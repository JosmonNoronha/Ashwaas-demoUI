# Commands to Run the Project

## Prerequisites
- Python 3.10+ installed
- Node.js and npm installed

## Step 1: Fix Dependency Issue (if needed)

If you encounter the `cached_download` import error, update huggingface_hub:

```powershell
pip install --upgrade huggingface_hub
```

## Step 2: Run Backend Server

Open a **PowerShell** or **Command Prompt** terminal:

```powershell
# Navigate to backend directory
cd "D:\Database answer\backend"

# Install/update dependencies (if not already done)
pip install -r requirements.txt

# Run the FastAPI server
python main.py
```

Or using uvicorn directly:

```powershell
cd "D:\Database answer\backend"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at: **http://localhost:8000**

## Step 3: Run Frontend Server

Open a **NEW** PowerShell or Command Prompt terminal (keep backend running):

```powershell
# Navigate to frontend directory
cd "D:\Database answer\frontend"

# Install dependencies (first time only)
npm install

# Run the development server
npm run dev
```

The frontend will be available at: **http://localhost:3000**

## Quick Start (Both Servers)

### Terminal 1 - Backend:
```powershell
cd "D:\Database answer\backend"
python main.py
```

### Terminal 2 - Frontend:
```powershell
cd "D:\Database answer\frontend"
npm install
npm run dev
```

## Verify Everything is Running

1. **Backend Health Check:**
   ```powershell
   curl http://localhost:8000/health
   ```
   Or open in browser: http://localhost:8000/health

2. **Frontend:**
   Open browser: http://localhost:3000

## Troubleshooting

### Backend Issues:
- **Import errors**: Run `pip install -r requirements.txt` again
- **Port already in use**: Change port in `main.py` or kill the process using port 8000
- **huggingface_hub error**: Run `pip install --upgrade huggingface_hub`

### Frontend Issues:
- **npm not found**: Install Node.js from https://nodejs.org/
- **Port 3000 in use**: Vite will automatically use the next available port

## Environment Variables

The `.env` file in the `backend/` directory should contain:
```
LLM_API_URL=https://0bd44b20599c.ngrok-free.app
```

Make sure this file exists before starting the backend.


