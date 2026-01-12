"""
Improved Konkani Audio Processing Pipeline
===========================================
Key improvements:
- Better resource management
- Thread-safe operations
- Reduced memory usage
- Cleaner error handling
"""

import os
import sys

# Add CUDA libraries to PATH before importing anything
nvidia_base_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia')
cuda_bin_paths = [
    os.path.join(nvidia_base_path, 'cublas', 'bin'),
    os.path.join(nvidia_base_path, 'cudnn', 'bin'),
]
for bin_path in cuda_bin_paths:
    if os.path.exists(bin_path):
        os.add_dll_directory(bin_path)  # Python 3.8+ method
        os.environ['PATH'] = bin_path + os.pathsep + os.environ.get('PATH', '')

import numpy as np
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import warnings
import difflib
import unicodedata
import google.generativeai as genai
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Centralized configuration"""
    # Model paths
    whisper_model_path: str = "..\whisper-small-konkani-faster1"
    translation_model_path: str = r"..\models\KE\T_BASE_KE_17_07\translator"
    emotion_model_source: str = "..\speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
    konkani_vocab_path: str = r"..\TranslateKar-English-to-Konkani\vocabulary\bert_gom.vocab"
    
    # Gemini API settings
    gemini_api_key: str = ""  # Set this or use environment variable
    gemini_enabled: bool = True  # Enable Gemini correction
    
    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    recording_duration: float = 5.0
    
    # VAD settings
    energy_threshold: float = 0.01
    silence_duration: float = 1.5
    
    # Processing
    max_queue_size: int = 5
    audio_chunk_size: int = 1600  # 0.1s at 16kHz
    fuzzy_matcher_enabled: bool = False  # Set by user at startup

# ============================================================================
# TRANSLATOR (IMPROVED)
# ============================================================================

class KonkaniToEnglishTranslator:
    """Thread-safe Konkani to English translator"""
    
    def __init__(self, model_path: str):
        import tensorflow as tf
        import tensorflow_text
        
        # Force CPU usage for TensorFlow (avoid device assignment issues with protobuf)
        tf.config.set_visible_devices([], 'GPU')
        
        # Load model directly without device context
        self.model = tf.saved_model.load(model_path)
        self.translator = self.model.signatures["serving_default"]
        
        self._lock = threading.Lock()  # Thread safety
        print("✓ Translator loaded (CPU)")
    
    def translate(self, konkani_text: str) -> str:
        """Thread-safe translation"""
        if not konkani_text or not konkani_text.strip():
            return "[Empty input]"
        
        try:
            with self._lock:  # Prevent concurrent access
                import tensorflow as tf
                output = self.translator(tf.constant([konkani_text]))
                
                for val in output.values():
                    result = val.numpy()
                    if hasattr(result, 'ndim') and result.ndim > 0:
                        result = result[0]
                    if isinstance(result, bytes):
                        return result.decode('utf-8')
                    return str(result)
                    
        except Exception as e:
            return f"[Translation error: {type(e).__name__}]"

# ============================================================================
# EMOTION DETECTOR (IMPROVED)
# ============================================================================

class EmotionDetector:
    """Thread-safe emotion detection"""
    
    def __init__(self, model_source: str, savedir: str):
        import torch
        from speechbrain.inference.classifiers import EncoderClassifier
        
        self.enabled = False
        self.classifier = None
        self._lock = threading.Lock()
        
        try:
            print("Loading emotion model (first run may take time)...")
            self.classifier = EncoderClassifier.from_hparams(
                source=model_source,
                savedir=savedir,
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
            self.emotions = ['neutral', 'happy', 'sad', 'angry']
            self.enabled = True
            print("✓ Emotion model loaded")
        except Exception as e:
            print(f"⚠ Emotion detection disabled: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def detect_from_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """Thread-safe emotion detection"""
        if not self.enabled:
            return "disabled"
        
        try:
            with self._lock:
                import torch
                import torchaudio
                
                # Prepare audio
                if audio_array.ndim > 1:
                    audio_array = audio_array.flatten()
                
                signal = torch.from_numpy(audio_array.astype(np.float32))
                if signal.ndim == 1:
                    signal = signal.unsqueeze(0)
                
                # Resample if needed
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    signal = resampler(signal)
                
                # Inference
                with torch.no_grad():
                    device = next(self.classifier.mods.wav2vec2.parameters()).device
                    signal = signal.to(device)
                    wav_lens = torch.ones(1, device=device)
                    
                    features = self.classifier.mods.wav2vec2(signal, wav_lens)
                    pooled = self.classifier.mods.avg_pool(features, wav_lens)
                    logits = self.classifier.mods.output_mlp(pooled)
                    probs = torch.softmax(logits.squeeze(), dim=-1).cpu().numpy()
                    
                    if probs.ndim > 1:
                        probs = probs.flatten()
                    
                    idx = int(np.argmax(probs))
                    emotion = self.emotions[idx]
                    confidence = float(probs[idx])
                    
                    return f"{emotion} ({confidence:.1%})"
                    
        except Exception as e:
            print(f"⚠ Emotion error: {type(e).__name__}")
            return "error"

# ============================================================================
# GEMINI CORRECTOR FOR KONKANI
# ============================================================================

class GeminiKonkaniCorrector:
    """Gemini-based Konkani text corrector"""
    
    def __init__(self, api_key: str = None):
        self.enabled = False
        self.model = None
        self._lock = threading.Lock()
        
        # Get API key from parameter or environment
        if not api_key:
            api_key = "AIzaSyDVGC9YnhS28_GuxmlRa37s1_TAFDWJF7c"
        
        if not api_key:
            print("⚠ Gemini API key not found. Set GEMINI_API_KEY environment variable.")
            return
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('models/gemini-2.5-flash')
            self.enabled = True
            print("✓ Gemini corrector loaded")
        except Exception as e:
            print(f"⚠ Gemini correction disabled: {type(e).__name__}: {str(e)}")
    
    def correct_text(self, konkani_text: str) -> str:
        """Correct Konkani text using Gemini"""
        if not self.enabled or not konkani_text or not konkani_text.strip():
            return konkani_text
        
        try:
            with self._lock:
                prompt = f"""You are a Konkani language corrector.

Task:
- The input text is in Konkani (written in Devanagari).
- The sentence is mostly correct but may contain:
  - spelling mistakes
  - incorrect verb forms
  - wrong gender/number agreement
  - minor word order issues
- DO NOT translate the sentence.
- DO NOT add new information.
- DO NOT remove words unless they are clearly incorrect.
- Preserve the original meaning exactly.

Rules:
1. Output ONLY the corrected Konkani sentence in Devanagari.
2. Do not explain the corrections.
3. If a word is ambiguous, choose the most commonly used Goan Konkani form.
4. Keep punctuation minimal and natural.

Input:
{konkani_text}
"""
                
                response = self.model.generate_content(prompt)
                corrected = response.text.strip()
                return corrected if corrected else konkani_text
                
        except Exception as e:
            print(f"⚠ Gemini correction error: {type(e).__name__}")
            return konkani_text

# ============================================================================
# FUZZY MATCHER FOR KONKANI VOCABULARY
# ============================================================================

class KonkaniFuzzyMatcher:
    """Fuzzy matching with matra-aware scoring for Konkani words"""
    
    # Devanagari matra (diacritic) ranges
    MATRAS = set(range(0x093E, 0x094F + 1)) | set(range(0x0951, 0x0957 + 1))
    DEVANAGARI_RANGE = range(0x0900, 0x097F + 1)
    
    def __init__(self, vocab_path: str):
        self.vocab_path = vocab_path
        self.vocabulary = set()
        self.load_vocabulary()
        self._lock = threading.Lock()
        print(f"✓ Loaded {len(self.vocabulary)} Konkani vocabulary entries")
    
    def load_vocabulary(self):
        """Load Konkani vocabulary from file"""
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and self._is_devanagari_word(word):
                        self.vocabulary.add(word)
        except Exception as e:
            print(f"⚠ Warning: Could not load vocabulary: {e}")
    
    def _is_devanagari_word(self, word: str) -> bool:
        """Check if word contains Devanagari characters"""
        if not word:
            return False
        return any(ord(char) in self.DEVANAGARI_RANGE for char in word)
    
    def _separate_base_and_matras(self, word: str) -> Tuple[str, List[str]]:
        """Separate base characters from matras (diacritics)"""
        base_chars = []
        matras = []
        
        for char in word:
            char_code = ord(char)
            if char_code in self.MATRAS:
                matras.append(char)
            else:
                base_chars.append(char)
        
        return ''.join(base_chars), matras
    
    def _calculate_similarity(self, word1: str, word2: str) -> Dict[str, float]:
        """Calculate detailed similarity scores"""
        # Separate base and matras
        base1, matras1 = self._separate_base_and_matras(word1)
        base2, matras2 = self._separate_base_and_matras(word2)
        
        # Overall character-level similarity
        overall_ratio = difflib.SequenceMatcher(None, word1, word2).ratio()
        
        # Base character similarity
        base_ratio = difflib.SequenceMatcher(None, base1, base2).ratio() if base1 and base2 else 0.0
        
        # Matra similarity
        matras1_str = ''.join(sorted(matras1))
        matras2_str = ''.join(sorted(matras2))
        matra_ratio = difflib.SequenceMatcher(None, matras1_str, matras2_str).ratio() if matras1_str or matras2_str else 1.0
        
        # Weighted score: base characters are more important than matras
        weighted_score = (base_ratio * 0.7) + (matra_ratio * 0.3)
        
        return {
            'overall': overall_ratio * 100,
            'base': base_ratio * 100,
            'matra': matra_ratio * 100,
            'weighted': weighted_score * 100
        }
    
    def find_matches(self, word: str, top_n: int = 5, min_score: float = 50.0) -> List[Dict]:
        """Find top N fuzzy matches with detailed scoring"""
        if not self._is_devanagari_word(word):
            return []
        
        with self._lock:
            # Check exact match first
            if word in self.vocabulary:
                return [{
                    'word': word,
                    'rank': 1,
                    'overall': 100.0,
                    'base': 100.0,
                    'matra': 100.0,
                    'weighted': 100.0,
                    'status': 'exact_match'
                }]
            
            # Calculate similarities
            matches = []
            for vocab_word in self.vocabulary:
                scores = self._calculate_similarity(word, vocab_word)
                
                if scores['weighted'] >= min_score:
                    matches.append({
                        'word': vocab_word,
                        'overall': round(scores['overall'], 2),
                        'base': round(scores['base'], 2),
                        'matra': round(scores['matra'], 2),
                        'weighted': round(scores['weighted'], 2),
                        'status': 'fuzzy_match'
                    })
            
            # Sort by weighted score and rank
            matches.sort(key=lambda x: x['weighted'], reverse=True)
            matches = matches[:top_n]
            
            # Add ranks
            for i, match in enumerate(matches, 1):
                match['rank'] = i
            
            return matches
    
    def validate_and_correct(self, transcription: str, correction_threshold: float = 75.0) -> Dict[str, any]:
        """Validate and auto-correct transcription if matches >= threshold"""
        if not transcription:
            return {
                'original': '',
                'corrected': '',
                'words': [],
                'stats': {'total': 0, 'matched': 0, 'corrected': 0, 'unchanged': 0}
            }
        
        words = transcription.split()
        corrected_words = []
        results = []
        
        for word in words:
            # Keep non-Devanagari words as-is
            if not self._is_devanagari_word(word):
                corrected_words.append(word)
                continue
            
            # Find matches
            matches = self.find_matches(word, top_n=5, min_score=60.0)
            
            # Determine if correction should be applied
            corrected_word = word
            action = 'unchanged'
            
            if matches:
                top_match = matches[0]
                
                if top_match['status'] == 'exact_match':
                    # Already correct
                    corrected_word = word
                    action = 'exact_match'
                elif top_match['weighted'] >= correction_threshold:
                    # Auto-correct: top match meets threshold
                    corrected_word = top_match['word']
                    action = 'auto_corrected'
                else:
                    # Keep original: doesn't meet threshold
                    corrected_word = word
                    action = 'below_threshold'
            
            corrected_words.append(corrected_word)
            
            word_result = {
                'original': word,
                'corrected': corrected_word,
                'action': action,
                'matches': matches
            }
            results.append(word_result)
        
        # Statistics
        total = len(results)
        exact_matches = sum(1 for r in results if r['action'] == 'exact_match')
        auto_corrected = sum(1 for r in results if r['action'] == 'auto_corrected')
        unchanged = sum(1 for r in results if r['action'] in ['unchanged', 'below_threshold'])
        
        return {
            'original': transcription,
            'corrected': ' '.join(corrected_words),
            'words': results,
            'stats': {
                'total': total,
                'exact_matches': exact_matches,
                'auto_corrected': auto_corrected,
                'unchanged': unchanged,
                'accuracy': round((exact_matches / total * 100) if total > 0 else 0, 2)
            }
        }

# ============================================================================
# AUDIO PIPELINE (IMPROVED)
# ============================================================================

class AudioPipeline:
    """Main pipeline with better resource management"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.is_running = False
        
        # Thread-safe queues with size limits
        self.audio_queue = queue.Queue(maxsize=config.max_queue_size)
        self.transcription_queue = queue.Queue(maxsize=10)  # Increased from 2 to 10
        
        # Components (loaded lazily)
        self.whisper_model = None
        self.translator = None
        self.emotion_detector = None
        self.fuzzy_matcher = None
        self.gemini_corrector = None
        
        # Threading
        self.threads = []
    
    def initialize(self):
        """Load all models"""
        print("=" * 70)
        print("INITIALIZING PIPELINE")
        print("=" * 70)
        
        # Whisper - Use CPU to avoid CUDA DLL issues
        print("\n[1/3] Loading Whisper ASR...")
        from faster_whisper import WhisperModel
        
        # Try GPU first, but fall back to CPU gracefully
        try:
            import torch
            if torch.cuda.is_available():
                print("   Attempting GPU mode...")
                self.whisper_model = WhisperModel(
                    self.config.whisper_model_path,
                    device="cuda",
                    compute_type="float16"
                )
                print("✓ Whisper (GPU)")
            else:
                raise RuntimeError("CUDA not available")
        except Exception as e:
            print(f"   GPU failed ({type(e).__name__}), using CPU")
            self.whisper_model = WhisperModel(
                self.config.whisper_model_path,
                device="cpu",
                compute_type="int8"
            )
            print("✓ Whisper (CPU - Stable Mode)")
        
        # Translator
        print("\n[2/3] Loading translator...")
        self.translator = KonkaniToEnglishTranslator(
            self.config.translation_model_path
        )
        
        # Emotion
        print("\n[3/4] Loading emotion detector...")
        self.emotion_detector = EmotionDetector(
            self.config.emotion_model_source,
            self.config.emotion_model_savedir
        )
        
        # Gemini Corrector (Optional)
        print("\n[4/5] Loading Gemini corrector...")
        if self.config.gemini_enabled:
            self.gemini_corrector = GeminiKonkaniCorrector(
                self.config.gemini_api_key
            )
        else:
            print("   Gemini correction disabled")
        
        # Fuzzy Matcher (Optional)
        if self.config.fuzzy_matcher_enabled:
            print("\n[5/5] Loading Konkani vocabulary...")
            self.fuzzy_matcher = KonkaniFuzzyMatcher(
                self.config.konkani_vocab_path
            )
        else:
            print("\n[5/5] Fuzzy matcher disabled (skipping vocabulary load)")
        
        print("\n" + "=" * 70)
        print("✓ READY")
        print("=" * 70)
    
    def has_speech(self, audio_chunk: np.ndarray) -> bool:
        """Energy-based VAD"""
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        return energy > self.config.energy_threshold
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback"""
        if status:
            print(f"⚠ {status}", file=sys.stderr)
        
        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass  # Drop frames if queue full
    
    def recorder_thread(self):
        """Record and detect speech segments"""
        buffer_size = int(
            self.config.sample_rate * 
            self.config.recording_duration / 
            self.config.audio_chunk_size
        )
        audio_buffer = deque(maxlen=buffer_size)
        silence_counter = 0
        speech_detected = False
        
        print("\n🎤 LISTENING...")
        print("-" * 70)
        
        while self.is_running:
            try:
                # Get audio with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer.append(audio_chunk)
                
                # VAD
                has_voice = self.has_speech(audio_chunk)
                
                if has_voice:
                    if not speech_detected:
                        print("\n🎙️  Recording...", flush=True)
                    speech_detected = True
                    silence_counter = 0
                elif speech_detected:
                    silence_counter += 1
                
                # Check for end of speech
                silence_duration = (
                    silence_counter * 
                    len(audio_chunk) / 
                    self.config.sample_rate
                )
                
                if speech_detected and silence_duration >= self.config.silence_duration:
                    print("🔄 Processing...", flush=True)
                    
                    # Concatenate and submit
                    audio_data = np.concatenate(list(audio_buffer), axis=0)
                    try:
                        self.transcription_queue.put_nowait(audio_data)
                    except queue.Full:
                        print("⚠ Processing queue full, skipping")
                    
                    # Reset
                    speech_detected = False
                    silence_counter = 0
                    audio_buffer.clear()  # FIX: Clear buffer
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Recorder error: {e}")
    
    def processor_thread(self):
        """Process audio through pipeline"""
        while self.is_running:
            try:
                audio_data = self.transcription_queue.get(timeout=0.5)
                
                print("\n" + "=" * 70)
                start_time = time.time()
                
                # ASR
                print("[1/4] Transcribing...")
                transcription = self.transcribe_audio(audio_data)
                
                if not transcription:
                    print("⚪ No speech")
                    print("-" * 70)
                    continue
                
                print(f"✓ Konkani (Raw): {transcription}")
                
                # Gemini Correction (Optional)
                corrected_text = transcription  # Default to original
                if self.config.gemini_enabled and self.gemini_corrector and self.gemini_corrector.enabled:
                    print("\n[2/5] Correcting with Gemini...")
                    corrected_text = self.gemini_corrector.correct_text(transcription)
                    if corrected_text != transcription:
                        print(f"✓ Corrected: {corrected_text}")
                    else:
                        print("✓ No changes needed")
                else:
                    print("\n[2/5] Gemini correction disabled")
                
                # Fuzzy Matching Validation & Auto-Correction (Optional)
                if self.config.fuzzy_matcher_enabled and self.fuzzy_matcher:
                    print("\n[3/5] Validating vocabulary...")
                    validation = self.fuzzy_matcher.validate_and_correct(corrected_text, correction_threshold=75.0)
                    corrected_text = validation['corrected']
                    self._print_validation_results(validation)
                else:
                    print("\n[3/5] Fuzzy matching disabled")
                
                # Use corrected transcription for translation
                print("\n[4/5] Translating...")
                translation = self.translator.translate(corrected_text)
                print(f"✓ English: {translation}")
                
                # Emotion
                print("\n[5/5] Detecting emotion...")
                emotion = self.emotion_detector.detect_from_array(
                    audio_data.flatten(),
                    self.config.sample_rate
                )
                print(f"✓ Emotion: {emotion}")
                
                # Summary
                elapsed = time.time() - start_time
                print("\n" + "=" * 70)
                print("📝 RESULTS")
                print("=" * 70)
                if transcription != corrected_text:
                    print(f"Original:  {transcription}")
                    print(f"Corrected: {corrected_text}")
                else:
                    print(f"Konkani:   {transcription}")
                print(f"English:   {translation}")
                print(f"Emotion:   {emotion}")
                print(f"Time:      {elapsed:.2f}s")
                print("=" * 70)
                print("👂 Listening...\n")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Pipeline error: {e}")
    
    def _print_validation_results(self, validation: Dict):
        """Pretty print validation results with corrections"""
        stats = validation['stats']
        print(f"✓ Vocabulary: {stats['exact_matches']}/{stats['total']} exact, " +
              f"{stats['auto_corrected']} auto-corrected (≥75%), " +
              f"{stats['unchanged']} unchanged")
        
        # Show auto-corrected words
        auto_corrected = [w for w in validation['words'] if w['action'] == 'auto_corrected']
        if auto_corrected:
            print("\n  ✏️  Auto-corrected:")
            for word_data in auto_corrected:
                original = word_data['original']
                corrected = word_data['corrected']
                if word_data['matches']:
                    top_match = word_data['matches'][0]
                    print(f"    '{original}' → '{corrected}' " +
                          f"(Score: {top_match['weighted']}%, " +
                          f"Base: {top_match['base']}%, " +
                          f"Matra: {top_match['matra']}%)")
        
        # Show words below threshold (kept unchanged)
        below_threshold = [w for w in validation['words'] if w['action'] == 'below_threshold']
        if below_threshold:
            print("\n  ⚠ Below 75% threshold (unchanged):")
            for word_data in below_threshold[:3]:  # Show top 3
                original = word_data['original']
                if word_data['matches']:
                    top_match = word_data['matches'][0]
                    print(f"    '{original}' (Best match: '{top_match['word']}' " +
                          f"at {top_match['weighted']}%)")
    
    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio using Whisper"""
        try:
            # Prepare audio
            audio = audio_data.flatten().astype(np.float32)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            # Quick VAD check
            if not self.has_speech(audio):
                return None
            
            # Transcribe
            segments, _ = self.whisper_model.transcribe(
                audio,
                language="mr",
                beam_size=3,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.3,
                    min_speech_duration_ms=250
                )
            )
            
            text = " ".join(seg.text.strip() for seg in segments)
            return text.strip() if text else None
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None
    
    def start(self):
        """Start the pipeline"""
        import sounddevice as sd
        
        self.is_running = True
        
        # Show devices
        print("\n📢 Audio devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"  [{i}] {dev['name']}")
        
        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                callback=self.audio_callback,
                blocksize=self.config.audio_chunk_size,
                dtype='float32'
            ):
                # Start threads
                recorder = threading.Thread(
                    target=self.recorder_thread,
                    daemon=True
                )
                processor = threading.Thread(
                    target=self.processor_thread,
                    daemon=True
                )
                
                recorder.start()
                processor.start()
                
                self.threads = [recorder, processor]
                
                # Keep alive
                while self.is_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n\n✓ STOPPING...")
            self.stop()
        except Exception as e:
            print(f"\n❌ Fatal: {e}")
            self.stop()
    
    def stop(self):
        """Clean shutdown"""
        self.is_running = False
        
        # Wait for threads
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        print("👋 Goodbye!")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("KONKANI AUDIO PROCESSING PIPELINE")
    print("=" * 70)
    
    # Ask user about fuzzy matching
    print("\n🔍 Fuzzy Matcher Configuration")
    print("-" * 70)
    print("The fuzzy matcher validates and auto-corrects transcribed words")
    print("against Konkani vocabulary (auto-corrects if match ≥75%).")
    print("-" * 70)
    
    while True:
        response = input("\nEnable fuzzy matching? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            fuzzy_enabled = True
            print("✓ Fuzzy matcher will be enabled\n")
            break
        elif response in ['n', 'no']:
            fuzzy_enabled = False
            print("✓ Fuzzy matcher will be disabled\n")
            break
        else:
            print("Please enter 'y' or 'n'")
    
    config = PipelineConfig()
    config.fuzzy_matcher_enabled = fuzzy_enabled
    pipeline = AudioPipeline(config)
    
    pipeline.initialize()
    pipeline.start()

if __name__ == "__main__":
    main()