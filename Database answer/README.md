# PDF Question Answering System using RAG

A complete web application for question answering over PDF documents using Retrieval-Augmented Generation (RAG).

## Features

- 📄 Upload multiple PDF files
- 🔍 Extract and chunk text with overlap
- 🧠 Generate embeddings using sentence-transformers
- 💾 Store embeddings in FAISS vector store (persistent)
- ❓ Ask natural language questions
- 🤖 Get answers using RAG with remote LLM (Mistral/Gemma via ngrok)

## Tech Stack

- **Backend**: Python + FastAPI
- **Frontend**: Vue 3 (Composition API) + Vite
- **Vector Store**: FAISS (local, persistent)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Remote API (Mistral/Gemma on Google Colab via ngrok)

## Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI application
│   ├── pdf_processor.py     # PDF extraction and chunking
│   ├── embeddings.py        # Embedding generation
│   ├── vector_store.py      # FAISS vector store management
│   ├── rag.py               # RAG pipeline (retrieval + prompt)
│   ├── llm_client.py        # HTTP client for remote LLM
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.vue          # Main Vue component
│   │   ├── api.js           # API client
│   │   ├── main.js          # Vue app entry
│   │   └── style.css        # Styles
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## Setup Instructions

### Backend Setup

1. **Create virtual environment** (recommended):
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set LLM API URL** (optional, defaults to placeholder):
```bash
export LLM_API_URL="http://your-ngrok-url.ngrok.io/generate"
# On Windows: set LLM_API_URL=http://your-ngrok-url.ngrok.io/generate
```

4. **Run the server**:
```bash
python main.py
# Or: uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Install dependencies**:
```bash
cd frontend
npm install
```

2. **Configure API URL** (if backend is not on localhost:8000):
Create a `.env` file:
```
VITE_API_URL=http://localhost:8000
```

3. **Run development server**:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## API Endpoints

### POST /upload-pdf
Upload and process a PDF file.

**Request**: Multipart form data with `file` field

**Response**:
```json
{
  "message": "PDF processed successfully",
  "chunks_count": 42,
  "filename": "document.pdf"
}
```

### POST /ask
Ask a question about uploaded PDFs.

**Request**:
```json
{
  "question": "What is the main topic of the document?"
}
```

**Response**:
```json
{
  "answer": "The main topic is...",
  "sources": ["document.pdf"]
}
```

### GET /health
Health check endpoint.

## RAG Prompt Template

The system constructs prompts with the following structure:

```
You are a helpful assistant that answers questions based ONLY on the provided context.

CONTEXT:
[Retrieved chunks from PDFs]

QUESTION: [User's question]

INSTRUCTIONS:
- Answer the question using ONLY the information provided in the context above.
- If the answer cannot be found in the context, respond with exactly: "Answer not found in the documents."
- Do not use any external knowledge or information not present in the context.
- Be concise and accurate.

ANSWER:
```

## Remote LLM Setup (Google Colab + ngrok)

### Example LLM API Implementation

Your LLM API should accept POST requests to `/generate`:

**Request**:
```json
{
  "prompt": "You are a helpful assistant..."
}
```

**Response**:
```json
{
  "response": "The answer to your question is..."
}
```

### Example curl Request

```bash
curl -X POST http://your-ngrok-url.ngrok.io/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "You are a helpful assistant that answers questions based ONLY on the provided context.\n\nCONTEXT:\n[Context here]\n\nQUESTION: What is the main topic?\n\nINSTRUCTIONS:\n- Answer using ONLY the context.\n- If not found, say: Answer not found in the documents.\n\nANSWER:"
  }'
```

### Google Colab Example

```python
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load model (Mistral or Gemma)
model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # or "google/gemma-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new generated text
    response = response[len(prompt):].strip()
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Then expose with ngrok:
```python
from pyngrok import ngrok
public_url = ngrok.connect(5000)
print(f"Public URL: {public_url}")
```

## Configuration

### Chunking Parameters

Default values in `backend/pdf_processor.py`:
- Chunk size: ~1000 tokens
- Overlap: ~200 tokens

### Embedding Model

Default: `all-MiniLM-L6-v2` (384 dimensions)

### Vector Store

- Location: `faiss_index` (in backend directory)
- Persists across restarts
- Uses cosine similarity (L2 normalized vectors)

## Usage

1. Start the backend server
2. Start the frontend development server
3. Open `http://localhost:3000` in your browser
4. Upload one or more PDF files
5. Wait for processing to complete
6. Ask questions about the uploaded documents

## Important Notes

- **No PDFs sent to LLM**: Only retrieved chunks are sent, not entire PDFs
- **Strict RAG**: Answers are based ONLY on provided context
- **Persistent storage**: FAISS index persists between restarts
- **Error handling**: Basic error handling included for common scenarios

## Troubleshooting

### Backend Issues

- **Import errors**: Ensure all dependencies are installed
- **FAISS errors**: Ensure `faiss-cpu` is installed (or `faiss-gpu` for GPU)
- **PDF parsing errors**: Ensure PDFs are not corrupted or password-protected

### Frontend Issues

- **CORS errors**: Backend CORS is configured to allow all origins (adjust for production)
- **API connection**: Check that backend is running on correct port

### LLM API Issues

- **Connection timeout**: Check ngrok URL is correct and active
- **Response format**: Ensure LLM API returns `{"response": "..."}` format

## License

MIT


