import os
import numpy as np
import tensorflow as tf
import tensorflow_text  # Required for text ops like RegexSplitWithOffsets
from faster_whisper import WhisperModel
from speechbrain.inference.classifiers import EncoderClassifier
from dataclasses import dataclass
from typing import Optional, Dict, List
import time

@dataclass
class PipelineConfig:
    """Configuration for the audio processing pipeline"""
    whisper_model_path: str = "../whisper-small-konkani-faster1"
    translator_model_path: str = "../models/KE/T_BASE_KE_17_07/translator"
    emotion_model_path: str = "../speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
    konkani_vocab_path: str = "../TranslateKar-English-to-Konkani/vocabulary/bert_gom.vocab"
    
    # Model settings
    whisper_device: str = "auto"  # "auto", "cuda", or "cpu"
    whisper_compute_type: str = "int8"
    use_gpu_for_translation: bool = False
    
    # Processing settings
    chunk_threshold: int = 15
    buffer_time: float = 1.5
    
    # Feature flags
    enable_gemini_correction: bool = False
    enable_fuzzy_matching: bool = False


class KonkaniToEnglishTranslator:
    """Handles Konkani to English translation using TensorFlow"""
    
    def __init__(self, model_path: str, vocab_path: str, use_gpu: bool = False):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.use_gpu = use_gpu
        self.model = None
        self.word_to_index = None
        self.index_to_word = None
        
    def initialize(self):
        """Load the translation model and vocabulary"""
        print("📚 Loading translation model...")
        
        # Force CPU for translation to avoid device conflicts
        if not self.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Load TensorFlow model
        self.model = tf.saved_model.load(self.model_path)
        
        # Load vocabulary
        self.word_to_index, self.index_to_word = self._load_vocab(self.vocab_path)
        print("✓ Translation model loaded")
        
    def _load_vocab(self, vocab_path: str):
        """Load vocabulary mappings"""
        word_to_index = {}
        index_to_word = {}
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                word = line.strip()
                word_to_index[word] = idx
                index_to_word[idx] = word
                
        return word_to_index, index_to_word
    
    def translate(self, konkani_text: str) -> str:
        """Translate Konkani text to English"""
        try:
            if not konkani_text or not konkani_text.strip():
                return ""
            
            # Tokenize input
            tokens = konkani_text.lower().split()
            indices = [self.word_to_index.get(token, self.word_to_index.get('[UNK]', 0)) 
                      for token in tokens]
            
            # Prepare input tensor
            input_tensor = tf.constant([indices], dtype=tf.int32)
            
            # Run translation
            output = self.model(input_tensor)
            output_indices = output.numpy()[0]
            
            # Decode output
            english_words = []
            for idx in output_indices:
                if idx == self.word_to_index.get('[END]', -1):
                    break
                word = self.index_to_word.get(idx, '')
                if word and word not in ['[START]', '[PAD]', '[UNK]']:
                    english_words.append(word)
            
            return ' '.join(english_words)
            
        except Exception as e:
            print(f"⚠ Translation error: {e}")
            return konkani_text  # Return original on error


class EmotionDetector:
    """Handles emotion detection from audio using SpeechBrain"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.classifier = None
        self.emotion_labels = ['neu', 'hap', 'sad', 'ang']
        
    def initialize(self):
        """Load the emotion detection model"""
        print("🎭 Loading emotion detection model...")
        
        # Convert relative path to absolute path for local loading
        model_path = os.path.abspath(self.model_path)
        
        # Check if it's a local path or HuggingFace repo
        if os.path.exists(model_path):
            # Local path - load directly
            self.classifier = EncoderClassifier.from_hparams(
                source=model_path,
                savedir=model_path
            )
        else:
            # HuggingFace repo - use default behavior
            self.classifier = EncoderClassifier.from_hparams(
                source=self.model_path,
                savedir="pretrained_models/emotion-recognition"
            )
        
        print("✓ Emotion model loaded")
    
    def detect_from_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Dict[str, float]:
        """Detect emotion from numpy audio array"""
        try:
            # Ensure audio is float32
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Convert to torch tensor (speechbrain expects this)
            import torch
            audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0)
            
            # Get predictions
            with torch.no_grad():
                out_prob, score, index, text_lab = self.classifier.classify_batch(audio_tensor)
                probabilities = out_prob.squeeze().cpu().numpy()
            
            # Create emotion scores dictionary
            emotion_scores = {
                label: float(prob) * 100 
                for label, prob in zip(self.emotion_labels, probabilities)
            }
            
            return emotion_scores
            
        except Exception as e:
            print(f"⚠ Emotion detection error: {e}")
            # Return neutral emotion on error
            return {label: 25.0 for label in self.emotion_labels}


class AudioPipeline:
    """Main pipeline orchestrating all audio processing components"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.whisper_model = None
        self.translator = None
        self.emotion_detector = None
        
    def initialize(self):
        """Initialize all models in the pipeline"""
        print("\n" + "="*50)
        print("🚀 Initializing Audio Processing Pipeline")
        print("="*50)
        
        start_time = time.time()
        
        # 1. Load Whisper ASR
        print("\n[1/3] 🎤 Loading Whisper ASR model...")
        device = self.config.whisper_device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        
        self.whisper_model = WhisperModel(
            self.config.whisper_model_path,
            device=device,
            compute_type=self.config.whisper_compute_type
        )
        print(f"✓ Whisper model loaded on {device}")
        
        # 2. Load Translator
        print("\n[2/3] 📚 Loading Konkani-English translator...")
        self.translator = KonkaniToEnglishTranslator(
            self.config.translator_model_path,
            self.config.konkani_vocab_path,
            use_gpu=self.config.use_gpu_for_translation
        )
        self.translator.initialize()
        
        # 3. Load Emotion Detector
        print("\n[3/3] 🎭 Loading emotion detection model...")
        self.emotion_detector = EmotionDetector(self.config.emotion_model_path)
        self.emotion_detector.initialize()
        
        elapsed = time.time() - start_time
        print("\n" + "="*50)
        print(f"✅ Pipeline initialized in {elapsed:.2f}s")
        print("="*50 + "\n")
    
    def transcribe_audio(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio array to Konkani text"""
        try:
            if audio_array is None or len(audio_array) == 0:
                return ""
            
            # Ensure correct format
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Transcribe using Whisper
            segments, info = self.whisper_model.transcribe(
                audio_array,
                language="mr",  # Marathi code for Konkani
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Combine segments
            transcript = " ".join([segment.text for segment in segments])
            return transcript.strip()
            
        except Exception as e:
            print(f"⚠ Transcription error: {e}")
            return ""
    
    def process_audio(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Dict:
        """Process audio through the full pipeline"""
        start_time = time.time()
        
        # 1. Transcribe
        print("🎤 Transcribing audio...")
        konkani_text = self.transcribe_audio(audio_array, sample_rate)
        print(f"  Konkani: {konkani_text}")
        
        # 2. Translate
        print("📚 Translating to English...")
        english_text = self.translator.translate(konkani_text)
        print(f"  English: {english_text}")
        
        # 3. Detect emotion
        print("🎭 Detecting emotion...")
        emotion_scores = self.emotion_detector.detect_from_array(audio_array, sample_rate)
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        print(f"  Emotion: {dominant_emotion} ({emotion_scores[dominant_emotion]:.1f}%)")
        
        elapsed = time.time() - start_time
        print(f"⏱ Processing completed in {elapsed:.2f}s")
        
        return {
            "konkani": konkani_text,
            "english": english_text,
            "emotion_scores": emotion_scores,
            "dominant_emotion": dominant_emotion,
            "processing_time": elapsed
        }


# Example usage
if __name__ == "__main__":
    # Create pipeline with default config
    config = PipelineConfig()
    pipeline = AudioPipeline(config)
    
    # Initialize models
    pipeline.initialize()
    
    print("\n✅ Pipeline ready for audio processing!")
    print("Use pipeline.process_audio(audio_array) to process audio.")
