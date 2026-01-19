# How to Start the Server

## The Error
If you see: `Error loading ASGI app. Could not import module "main"`

This usually means uvicorn can't find the module. Make sure you're running the command from the correct directory.

## Correct Way to Start the Server

### Option 1: Using Python (Recommended)
```powershell
cd "D:\Database answer\backend"
python main.py
```

### Option 2: Using uvicorn directly
Make sure you're in the `backend` directory:

```powershell
cd "D:\Database answer\backend"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Important**: The `main:app` means:
- `main` = the file `main.py` (without .py extension)
- `app` = the FastAPI instance named `app` inside main.py

### Option 3: Using uvicorn with full path
If you're not in the backend directory:

```powershell
cd "D:\Database answer"
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

## Verify It's Working

After starting, you should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Then test it:
```powershell
curl http://localhost:8000/health
```

Or open in browser: http://localhost:8000/health

