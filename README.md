# Konkani Audio Processing - Setup Guide

## Prerequisites
Make sure you have Python installed with all the dependencies from your existing `live_pipeline.py`.

## Installation Steps

### 1. Install Additional Dependencies
```bash
pip install -r requirements_server.txt
```

This installs:
- FastAPI (WebSocket server)
- Uvicorn (ASGI server)
- WebSockets
- SciPy (audio processing)

### 2. Start the WebSocket Server
```bash
python websocket_server.py
```

You should see:
```
INITIALIZING PIPELINE
======================================================================
[1/5] Loading Whisper ASR...
✓ Whisper (GPU/CPU)
[2/5] Loading translator...
✓ Translator loaded (CPU)
[3/5] Loading emotion detector...
✓ Emotion model loaded
[4/5] Loading Gemini corrector...
✓ Gemini corrector loaded
[5/5] Loading Konkani vocabulary...
✓ Loaded XXXX Konkani vocabulary entries
======================================================================
✓ SERVER READY
======================================================================

🚀 Starting WebSocket server on ws://localhost:8000/ws
📡 Open your browser and navigate to your HTML file
Press Ctrl+C to stop
```

### 3. Open the Frontend
Open `index.html` in your browser:
- **Option A**: Double-click `index.html`
- **Option B**: Right-click → Open with → Chrome/Firefox
- **Option C**: Use a local server:
  ```bash
  python -m http.server 8080
  ```
  Then open: http://localhost:8080

### 4. Test the Connection
1. You should see "Connected" in green at the top of the page
2. Click the microphone button to start recording
3. Speak in Konkani
4. Results will appear automatically:
   - Konkani transcription
   - English translation
   - Emotion detection
   - Live charts updating

## How It Works

### Backend (Python)
- `websocket_server.py` runs a WebSocket server on `ws://localhost:8000/ws`
- Receives audio data from the browser
- Processes through the pipeline:
  1. Whisper ASR (transcription)
  2. Gemini correction (optional)
  3. Fuzzy matching (vocabulary validation)
  4. Translation (Konkani → English)
  5. Emotion detection
- Sends results back to frontend as JSON

### Frontend (JavaScript)
- `script.js` connects to the WebSocket server
- Captures microphone audio using MediaRecorder API
- Sends audio chunks every 1 second
- Receives results and updates:
  - Transcription display
  - Translation display
  - Emotion badge
  - Four charts (pie, line, bar, radar)

## Troubleshooting

### "Failed to connect to server"
- Make sure `websocket_server.py` is running
- Check that port 8000 is not in use
- Verify the WebSocket URL in `script.js` line 139

### "Microphone access denied"
- Allow microphone permissions in your browser
- Check browser settings → Privacy → Microphone

### No transcription appearing
- Speak louder or closer to the microphone
- Check the server console for errors
- Ensure audio is being sent (check browser console)

### Models not loading
- Verify model paths in `live_pipeline.py` PipelineConfig
- Check that all model files exist
- Ensure dependencies are installed

## Configuration

### Enable/Disable Features
Edit `websocket_server.py`, line 43-45:
```python
config.fuzzy_matcher_enabled = True  # Vocabulary validation
config.gemini_enabled = True         # Gemini correction
```

### Change WebSocket Port
Edit `websocket_server.py`, last line:
```python
uvicorn.run(app, host="0.0.0.0", port=8000)  # Change 8000 to your port
```

Then update `script.js`, line 139:
```javascript
const wsUrl = 'http://localhost:8000/ws';  // Match your port
```

### Adjust Audio Processing Interval
Edit `websocket_server.py`, line 269:
```python
if (current_time - last_process_time > 2.0):  # Process every 2 seconds
```

## Architecture

```
Browser (Frontend)
    ↓ Audio (WebM/Opus)
WebSocket Server (Python)
    ↓
Pipeline Processing
    ├─ Whisper ASR
    ├─ Gemini Correction
    ├─ Fuzzy Matching
    ├─ Translation
    └─ Emotion Detection
    ↓ JSON Results
WebSocket Server
    ↓
Browser (Update UI & Charts)
```

## Files Created

1. **websocket_server.py** - WebSocket server connecting backend to frontend
2. **requirements_server.txt** - Additional Python dependencies
3. **README.md** - This guide

## Existing Files Used

- **live_pipeline.py** - Your original pipeline code (imported)
- **index.html** - Frontend HTML structure
- **script.js** - Frontend JavaScript (modified for WebSocket)
- **styles.css** - Frontend styling

## Notes

- The server processes audio in 2-second intervals to reduce computational load
- Emotion distribution accumulates over the session (reset with Clear button)
- Charts update in real-time as new results arrive
- All processing happens on the backend (Python), frontend only handles UI

## Testing Without Backend (Mock Mode)

If you want to test the frontend without running the Python server:

1. Open `script.js`
2. Comment out line 408: `connectWebSocket();`
3. Uncomment the mock mode section (lines 413-454)
4. Replace line 408 with: `mockWebSocket();`

This will generate fake data for testing the UI.
