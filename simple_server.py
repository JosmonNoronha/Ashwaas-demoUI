"""
Minimal WebSocket server for testing - No ML models required
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Minimal Test Server",
        "websocket_url": "ws://localhost:8000/ws"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✓ Client connected!")
    
    try:
        while True:
            # Receive audio data (we'll just count bytes)
            data = await websocket.receive_bytes()
            print(f"📦 Received {len(data)} bytes of audio")
            
            # Send mock response immediately
            response = {
                "konkani": "हाँव बरें आसा (Test)",
                "english": "I am fine (Test)",
                "emotion": {
                    "label": "happy",
                    "confidence": 0.85
                },
                "emotionData": {
                    "label": "happy",
                    "confidence": 0.85,
                    "distribution": [30, 15, 10, 35, 5, 5],
                    "confidences": [0.3, 0.15, 0.1, 0.35, 0.05, 0.05],
                    "intensities": [0.85, 0.15, 0.1, 0.3, 0.05, 0.05],
                    "timestamp": time.strftime('%H:%M:%S'),
                    "score": 0.85
                }
            }
            
            await websocket.send_json(response)
            print("✓ Sent response to client")
            
    except WebSocketDisconnect:
        print("✗ Client disconnected")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MINIMAL TEST SERVER - NO ML MODELS")
    print("=" * 60)
    print("\n🚀 Starting on ws://localhost:8000/ws")
    print("📝 This sends mock data to test the frontend\n")
    
    uvicorn.run(app, host="localhost", port=8000)
