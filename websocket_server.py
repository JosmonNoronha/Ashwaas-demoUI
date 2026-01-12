"""
IMPROVED WebSocket Server for Konkani Live Speech Recognition
Key fixes:
1. Added audio buffering per connection
2. Implemented VAD before processing
3. Better audio parsing
4. Aligned with live_pipeline.py architecture
"""

import os
import sys

# Add CUDA libraries to PATH
nvidia_base_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia')
cuda_bin_paths = [
    os.path.join(nvidia_base_path, 'cublas', 'bin'),
    os.path.join(nvidia_base_path, 'cudnn', 'bin'),
]
for bin_path in cuda_bin_paths:
    if os.path.exists(bin_path):
        os.add_dll_directory(bin_path)
        os.environ['PATH'] = bin_path + os.pathsep + os.environ.get('PATH', '')

import asyncio
import json
import numpy as np
import threading
import warnings
from dataclasses import dataclass
from typing import Optional, Dict
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
import io
import wave
import uvicorn

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Centralized configuration matching live_pipeline.py"""
    # Model paths
    whisper_model_path: str = "..\whisper-small-konkani-faster1"
    translation_model_path: str = r"..\models\KE\T_BASE_KE_17_07\translator"
    emotion_model_source: str = "..\speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
    emotion_model_savedir: str = "pretrained_models/emotion-recognition"
    
    # Audio settings (matching live_pipeline.py)
    sample_rate: int = 16000
    energy_threshold: float = 0.01  # VAD threshold
    min_audio_duration: float = 0.8 # Minimum seconds to process
    max_buffer_duration: float = 10.0  # Maximum seconds to buffer

# ============================================================================
# TRANSLATOR
# ============================================================================

class KonkaniToEnglishTranslator:
    """Thread-safe Konkani to English translator"""
    
    def __init__(self, model_path: str):
        import tensorflow as tf
        import tensorflow_text
        
        tf.config.set_visible_devices([], 'GPU')
        self.model = tf.saved_model.load(model_path)
        self.translator = self.model.signatures["serving_default"]
        self._lock = threading.Lock()
        print("✓ Translator loaded (CPU)")
    
    def translate(self, konkani_text: str) -> str:
        """Thread-safe translation"""
        if not konkani_text or not konkani_text.strip():
            return "[Empty input]"
        
        try:
            with self._lock:
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
# EMOTION DETECTOR
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
            print("Loading emotion model...")
            self.classifier = EncoderClassifier.from_hparams(
                source=model_source,
                savedir=savedir,
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
            self.emotions = ['neutral', 'happy', 'sad', 'angry']
            self.enabled = True
            print("✓ Emotion model loaded")
        except Exception as e:
            print(f"⚠ Emotion detection disabled: {type(e).__name__}")
    
    def detect_from_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Dict[str, float]:
        """Thread-safe emotion detection returning scores"""
        if not self.enabled:
            return {'neutral': 25.0, 'happy': 25.0, 'sad': 25.0, 'angry': 25.0}
        
        try:
            with self._lock:
                import torch
                import torchaudio
                
                if audio_array.ndim > 1:
                    audio_array = audio_array.flatten()
                
                signal = torch.from_numpy(audio_array.astype(np.float32))
                if signal.ndim == 1:
                    signal = signal.unsqueeze(0)
                
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    signal = resampler(signal)
                
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
                    
                    return {
                        emotion: float(prob * 100)
                        for emotion, prob in zip(self.emotions, probs)
                    }
                    
        except Exception as e:
            print(f"⚠ Emotion error: {type(e).__name__}")
            return {'neutral': 25.0, 'happy': 25.0, 'sad': 25.0, 'angry': 25.0}

# ============================================================================
# AUDIO BUFFER (NEW - matches live_pipeline.py)
# ============================================================================

class AudioBuffer:
    """Manages audio buffering for a single WebSocket connection"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.buffer = deque()
        self.total_samples = 0
        self.max_samples = int(config.max_buffer_duration * config.sample_rate)
        self._lock = threading.Lock()
        self.speech_detected = False
        self.silence_samples = 0
    
    def add_chunk(self, audio_chunk: np.ndarray):
        """Add audio chunk to buffer"""
        with self._lock:
            self.buffer.append(audio_chunk)
            self.total_samples += len(audio_chunk)
            
            # Prevent buffer overflow
            while self.total_samples > self.max_samples:
                removed = self.buffer.popleft()
                self.total_samples -= len(removed)
    
    def get_audio(self) -> Optional[np.ndarray]:
        """Get buffered audio if meets minimum duration"""
        with self._lock:
            if self.total_samples < int(self.config.min_audio_duration * self.config.sample_rate):
                return None
            
            # Concatenate all buffered chunks
            audio = np.concatenate(list(self.buffer))
            return audio
    
    def clear(self):
        """Clear buffer"""
        with self._lock:
            self.buffer.clear()
            self.total_samples = 0
    
    def has_speech(self, audio_chunk: np.ndarray) -> bool:
        """Energy-based VAD (matches live_pipeline.py)"""
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        return energy > self.config.energy_threshold

# ============================================================================
# GLOBAL PIPELINE
# ============================================================================

class GlobalPipeline:
    """Singleton pipeline shared across WebSocket connections"""
    
    def __init__(self):
        self.config = PipelineConfig()
        self.whisper_model = None
        self.translator = None
        self.emotion_detector = None
        self.initialized = False
        self._lock = threading.Lock()
    
    def initialize(self):
        """Initialize all models once"""
        if self.initialized:
            return
        
        print("\n" + "=" * 70)
        print("INITIALIZING PIPELINE")
        print("=" * 70)
        
        # Whisper ASR
        print("\n[1/3] Loading Whisper ASR...")
        from faster_whisper import WhisperModel
        
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
            print("✓ Whisper (CPU)")
        
        # Translator
        print("\n[2/3] Loading translator...")
        self.translator = KonkaniToEnglishTranslator(
            self.config.translation_model_path
        )
        
        # Emotion Detector
        print("\n[3/3] Loading emotion detector...")
        self.emotion_detector = EmotionDetector(
            self.config.emotion_model_source,
            self.config.emotion_model_savedir
        )
        
        self.initialized = True
        print("\n" + "=" * 70)
        print("✓ PIPELINE READY")
        print("=" * 70 + "\n")
    
    def transcribe_audio(self, audio_array: np.ndarray) -> str:
        """Transcribe audio using Whisper (matches live_pipeline.py)"""
        try:
            with self._lock:
                if audio_array.ndim > 1:
                    audio_array = audio_array.flatten()
                
                # Ensure float32
                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)
                
                # Normalize audio (skip near-silence to prevent amplifying noise)
                max_val = np.max(np.abs(audio_array))
                if max_val < 1e-4:
                    print(f"   ⚠ Skipping near-silence audio (max={max_val:.6f})")
                    return ""
                audio_array = audio_array / max_val
                
                duration = len(audio_array) / 16000
                print(f"   Audio: {duration:.2f}s, max={max_val:.3f}")
                
                # Transcribe (matching live_pipeline.py settings)
                segments, _ = self.whisper_model.transcribe(
                    audio_array,
                    language="mr",
                    beam_size=3,
                    temperature=0.0,
                    vad_filter=True,
                    vad_parameters=dict(
                        threshold=0.2,  # Lower threshold for better detection
                        min_speech_duration_ms=200  # Shorter minimum duration
                    )
                )
                
                text = " ".join(seg.text.strip() for seg in segments)
                return text.strip() if text else ""
                
        except Exception as e:
            print(f"⚠ Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return ""

# Global instance
pipeline = GlobalPipeline()

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Konkani Speech Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# AUDIO PROCESSING HELPERS
# ============================================================================

def parse_audio_from_blob(audio_data: bytes) -> Optional[np.ndarray]:
    """Parse audio from WebM/WAV blob to numpy array"""
    try:
        # Try WAV format first
        try:
            with io.BytesIO(audio_data) as audio_io:
                with wave.open(audio_io, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Resample if needed
                    if sample_rate != 16000:
                        try:
                            from scipy import signal
                            audio_array = signal.resample(
                                audio_array,
                                int(len(audio_array) * 16000 / sample_rate)
                            )
                        except ImportError:
                            # Simple downsampling
                            step = int(sample_rate / 16000)
                            audio_array = audio_array[::step]
                    
                    print(f"✓ Parsed WAV audio: {len(audio_array)/16000:.2f}s at {sample_rate}Hz")
                    return audio_array
        except:
            pass
        
        # Try using soundfile (better for WebM/Opus)
        try:
            import soundfile as sf
            with io.BytesIO(audio_data) as audio_io:
                audio_array, sample_rate = sf.read(audio_io)
                
                # Convert to mono if stereo
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=1)
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    from scipy import signal
                    audio_array = signal.resample(
                        audio_array,
                        int(len(audio_array) * 16000 / sample_rate)
                    )
                
                # Ensure float32 normalized to [-1, 1]
                audio_array = audio_array.astype(np.float32)
                if np.abs(audio_array).max() > 1.0:
                    audio_array = audio_array / 32768.0
                
                print(f"✓ Parsed audio with soundfile: {len(audio_array)/16000:.2f}s at {sample_rate}Hz")
                return audio_array
        except ImportError:
            pass
        except Exception as e:
            print(f"⚠ soundfile parsing failed: {e}")
        
        # Try WebM/MP4/OGG format using pydub/ffmpeg
        try:
            from pydub import AudioSegment
            with io.BytesIO(audio_data) as audio_io:
                # Try to load as various formats
                audio = None
                for fmt in ['webm', 'ogg', 'mp4', 'mp3']:
                    try:
                        audio = AudioSegment.from_file(audio_io, format=fmt)
                        print(f"✓ Loaded as {fmt} format")
                        break
                    except:
                        audio_io.seek(0)
                        continue
                
                if audio is not None:
                    # Convert to mono 16kHz
                    audio = audio.set_channels(1).set_frame_rate(16000)
                    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    
                    # Normalize based on sample width
                    if audio.sample_width == 2:  # 16-bit
                        samples = samples / 32768.0
                    elif audio.sample_width == 4:  # 32-bit
                        samples = samples / 2147483648.0
                    else:  # 8-bit
                        samples = (samples - 128) / 128.0
                    
                    print(f"✓ Parsed WebM/media audio: {len(samples)/16000:.2f}s")
                    return samples
                else:
                    print("⚠ pydub could not parse audio (ffmpeg might be missing)")
        except ImportError:
            print("⚠ pydub not installed - WebM format not supported")
        except Exception as e:
            print(f"⚠ WebM parsing failed: {e}")
        
        # Fallback: Raw 16-bit PCM
        print("⚠ Falling back to raw PCM parsing - this may not work for WebM")
        if len(audio_data) % 2 != 0:
            audio_data = audio_data + b'\x00'
        
        if len(audio_data) == 0:
            return None
            
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        print(f"⚠ Using raw PCM fallback: {len(audio_array)/16000:.2f}s")
        return audio_array
        
    except Exception as e:
        print(f"⚠ Audio parsing error: {e}")
        return None

def process_audio(audio_array: np.ndarray) -> Dict:
    """Process audio through the pipeline"""
    try:
        print("\n" + "="*70)
        print(f"🎤 Processing {len(audio_array)/16000:.2f}s of audio...")
        
        # 1. Transcribe
        konkani_text = pipeline.transcribe_audio(audio_array)
        
        if not konkani_text:
            print("  ⚠ No speech detected")
            print("="*70 + "\n")
            return {
                "konkani": "",
                "english": "[No speech detected]",
                "emotion": {"neutral": 25.0, "happy": 25.0, "sad": 25.0, "angry": 25.0}
            }
        
        print(f"✓ Konkani: {konkani_text}")
        
        # 2. Translate
        english_text = pipeline.translator.translate(konkani_text)
        print(f"✓ English: {english_text}")
        
        # 3. Detect emotion
        emotion_scores = pipeline.emotion_detector.detect_from_array(audio_array, 16000)
        dominant = max(emotion_scores.items(), key=lambda x: x[1])[0]
        print(f"✓ Emotion: {dominant} ({emotion_scores[dominant]:.1f}%)")
        
        print("="*70 + "\n")
        
        return {
            "konkani": konkani_text,
            "english": english_text,
            "emotion": emotion_scores
        }
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        print("="*70 + "\n")
        return {
            "konkani": "",
            "english": f"[Error: {type(e).__name__}]",
            "emotion": {"neutral": 25.0, "happy": 25.0, "sad": 25.0, "angry": 25.0}
        }

# ============================================================================
# WEBSOCKET ENDPOINT (IMPROVED WITH BUFFERING)
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✓ Client connected")
    
    # Create audio buffer for this connection
    audio_buffer = AudioBuffer(pipeline.config)
    
    try:
        while True:
            # Receive message
            message = await websocket.receive()
            
            # Handle different message types
            if "bytes" in message:
                # Audio data chunk
                data = message["bytes"]
                audio_array = parse_audio_from_blob(data)
                
                if audio_array is None or len(audio_array) == 0:
                    continue
                
                # Check if this is a complete recording/file (> 0.5 seconds) - process immediately
                # Small streaming chunks are typically < 200ms, so anything > 0.5s is a complete recording
                audio_duration = len(audio_array) / pipeline.config.sample_rate
                
                if audio_duration > 0.5:
                    # This is a complete file/recording - process immediately
                    print(f"🎵 Complete audio detected ({audio_duration:.1f}s) - processing immediately...")
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, process_audio, audio_array)
                    await websocket.send_json(result)
                    
                    # Clear any buffered data
                    audio_buffer.clear()
                    audio_buffer.speech_detected = False
                    audio_buffer.silence_samples = 0
                    continue
                
                # For smaller chunks (live recording), use buffering logic
                has_speech = audio_buffer.has_speech(audio_array)
                
                if has_speech:
                    audio_buffer.speech_detected = True
                    audio_buffer.silence_samples = 0
                elif audio_buffer.speech_detected:
                    audio_buffer.silence_samples += len(audio_array)
                
                # Add to buffer if we've detected speech OR are in speech state
                if audio_buffer.speech_detected:
                    audio_buffer.add_chunk(audio_array)
                
                # Only process AFTER silence following speech
                silence_duration = audio_buffer.silence_samples / pipeline.config.sample_rate
                
                if audio_buffer.speech_detected and silence_duration >= 0.8:
                    buffered_audio = audio_buffer.get_audio()
                    
                    if buffered_audio is not None:
                        # Process in thread pool
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(None, process_audio, buffered_audio)
                        
                        # Send results
                        await websocket.send_json(result)
                        
                        # Clear buffer and reset state after successful processing
                        audio_buffer.clear()
                        audio_buffer.speech_detected = False
                        audio_buffer.silence_samples = 0
            
            elif "text" in message:
                # Handle control messages
                msg_data = json.loads(message["text"])
                
                if msg_data.get("type") == "clear_buffer":
                    audio_buffer.clear()
                    await websocket.send_json({"status": "buffer_cleared"})
            
    except WebSocketDisconnect:
        print("✗ Client disconnected")
    except Exception as e:
        print(f"⚠ WebSocket error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# SERVE STATIC FILES
# ============================================================================

@app.get("/")
async def serve_frontend():
    """Serve the main HTML page"""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Frontend not found. Open index.html manually.</h1>")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_initialized": pipeline.initialized
    }

# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on server startup"""
    print("\n🚀 Starting Konkani Speech Recognition Server...")
    await asyncio.to_thread(pipeline.initialize)
    print("\n✓ Server ready at http://localhost:8000")
    print("✓ WebSocket endpoint: ws://localhost:8000/ws")
    print("\n📱 Open http://localhost:8000 in your browser\n")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "websocket_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False, # Disable reload to prevent model reloading
        log_level="info"
    )