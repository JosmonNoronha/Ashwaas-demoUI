"""
Test script to verify emotion detection is working
"""
import numpy as np
import torch
from speechbrain.inference.classifiers import EncoderClassifier

print("="*60)
print("Testing Emotion Detection Model")
print("="*60)

# Load model
print("\n1. Loading emotion model...")
try:
    classifier = EncoderClassifier.from_hparams(
        source="..\speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        savedir="pretrained_models/emotion-recognition",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Generate test audio
print("\n2. Generating test audio signals...")
sample_rate = 16000
duration = 3  # 3 seconds

# Create different test signals
test_signals = {
    "neutral_tone": np.random.randn(sample_rate * duration) * 0.1,  # Low amplitude
    "energetic_tone": np.sin(2 * np.pi * 440 * np.linspace(0, duration, sample_rate * duration)) * 0.5,  # 440Hz sine
    "varied_tone": np.random.randn(sample_rate * duration) * 0.5 + np.sin(2 * np.pi * 220 * np.linspace(0, duration, sample_rate * duration)) * 0.3,
}

# Test each signal
print("\n3. Testing emotion detection...")
emotions = ['neutral', 'happy', 'sad', 'angry']

for name, audio in test_signals.items():
    print(f"\n  Testing: {name}")
    
    # Convert to float32 and normalize
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # Convert to torch tensor
    signal = torch.FloatTensor(audio).unsqueeze(0)
    
    try:
        # Run inference
        with torch.no_grad():
            out_prob, score, index, text_lab = classifier.classify_batch(signal)
            probs = out_prob.squeeze().cpu().numpy()
            
            if probs.ndim > 1:
                probs = probs.flatten()
            
            # Print results
            print(f"    Raw probabilities: {probs}")
            print(f"    Predicted index: {index.item()}")
            print(f"    Text label: {text_lab}")
            print(f"    Scores:")
            for emotion, prob in zip(emotions, probs):
                print(f"      {emotion}: {prob*100:.2f}%")
            
            # Get dominant emotion
            dominant_idx = int(np.argmax(probs))
            dominant_emotion = emotions[dominant_idx]
            confidence = probs[dominant_idx]
            print(f"    → Dominant: {dominant_emotion} ({confidence*100:.1f}%)")
            
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("Test completed!")
print("="*60)
