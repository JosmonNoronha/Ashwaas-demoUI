"""
Single-file PDF indexing script using FAISS.
"""

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
from typing import List


# -------------------------
# Config
# -------------------------
PDF_PATH = "C:\\Users\\Kavish\\OneDrive\\Desktop\\Sem prep\\syllabus\\IOT\\IOT level diagram.pdf"        # path to your PDF
INDEX_DIR = "faiss_index"      # where index + metadata will be saved
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000              # characters
CHUNK_OVERLAP = 200


# -------------------------
# PDF Loader
# -------------------------
def load_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


# -------------------------
# Text Chunking
# -------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


# -------------------------
# Main indexing logic
# -------------------------
def index_pdf(pdf_path: str):
    print("📄 Loading PDF...")
    text = load_pdf_text(pdf_path)

    print("✂️ Chunking text...")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    print(f"🔢 Total chunks: {len(chunks)}")

    print("🧬 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("📐 Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)

    print("💾 Saving FAISS index...")
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))

    print("💾 Saving metadata...")
    with open(os.path.join(INDEX_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print("✅ PDF indexed successfully!")


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    index_pdf(PDF_PATH)
