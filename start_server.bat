@echo off
echo ========================================
echo Konkani Audio Processing - Quick Start
echo ========================================
echo.

echo [1/3] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python first.
    pause
    exit /b 1
)
echo.

echo [2/3] Installing dependencies...
pip install -r requirements_server.txt
if errorlevel 1 (
    echo WARNING: Some dependencies may have failed to install.
    echo Continue anyway? Press Ctrl+C to cancel or
    pause
)
echo.

echo [3/3] Starting WebSocket Server...
echo.
echo ========================================
echo Server will start on ws://localhost:8000/ws
echo Open index.html in your browser
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python websocket_server.py
