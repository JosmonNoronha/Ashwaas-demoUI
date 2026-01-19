# Project Structure

```
Database answer/
├── backend/
│   ├── __init__.py (optional)
│   ├── main.py                    # FastAPI application with endpoints
│   ├── pdf_processor.py          # PDF text extraction and chunking
│   ├── embeddings.py             # Sentence-transformers embedding generation
│   ├── vector_store.py           # FAISS vector store with persistence
│   ├── rag.py                    # RAG pipeline (retrieval + prompt construction)
│   ├── llm_client.py             # HTTP client for remote LLM API
│   ├── requirements.txt          # Python dependencies
│   ├── .gitignore                # Git ignore file
│   ├── faiss_index               # FAISS index (created at runtime)
│   ├── faiss_index_metadata.pkl  # Metadata storage (created at runtime)
│   ├── faiss_index_texts.pkl     # Text chunks storage (created at runtime)
│   └── uploads/                  # Temporary PDF uploads (created at runtime)
│
├── frontend/
│   ├── src/
│   │   ├── App.vue               # Main Vue component (file upload + chat)
│   │   ├── api.js                # Axios API client
│   │   ├── main.js               # Vue app entry point
│   │   └── style.css             # Application styles
│   ├── index.html                # HTML entry point
│   ├── package.json              # Node.js dependencies
│   ├── vite.config.js            # Vite configuration
│   └── .gitignore                # Git ignore file
│
├── example_llm_api.py           # Example LLM API for Google Colab
├── example_curl_request.sh       # Example curl request to LLM API
├── README.md                     # Complete documentation
└── PROJECT_STRUCTURE.md          # This file
```

## Key Files Description

### Backend

- **main.py**: FastAPI application with `/upload-pdf` and `/ask` endpoints
- **pdf_processor.py**: Uses PyPDFLoader to extract text, chunks with ~1000 tokens, ~200 overlap
- **embeddings.py**: Wraps sentence-transformers (all-MiniLM-L6-v2) for embedding generation
- **vector_store.py**: Manages FAISS index with persistence, stores texts and metadata
- **rag.py**: Retrieves relevant chunks and constructs strict RAG prompts
- **llm_client.py**: Async HTTP client for calling remote LLM API

### Frontend

- **App.vue**: Main component with file upload area and chat interface
- **api.js**: Axios-based API client for backend communication
- **style.css**: Modern, clean UI styling

### Examples

- **example_llm_api.py**: Complete Flask API for running Mistral/Gemma on Colab
- **example_curl_request.sh**: Example curl command for testing LLM API

## Data Flow

1. **Upload PDF** → `main.py` → `pdf_processor.py` → chunks
2. **Generate Embeddings** → `embeddings.py` → embeddings array
3. **Store in FAISS** → `vector_store.py` → persistent index
4. **User Question** → `main.py` → `rag.py` → retrieve chunks
5. **Construct Prompt** → `rag.py` → RAG prompt with context
6. **Call LLM** → `llm_client.py` → remote LLM API
7. **Return Answer** → `main.py` → frontend → display


