"""
Query a FAISS-indexed PDF using RAG + Google Gemini.
"""

import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


# -------------------------
# Config
# -------------------------
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
GEMINI_MODEL = "gemini-2.5-flash"


# -------------------------
# Safety checks
# -------------------------
if "GOOGLE_API_KEY" not in os.environ:
    raise EnvironmentError(
        "GOOGLE_API_KEY not found. Set it as an environment variable."
    )


# -------------------------
# Load FAISS index + chunks
# -------------------------
print("📦 Loading FAISS index...")
index = faiss.read_index(f"{INDEX_DIR}/index.faiss")

print("📄 Loading chunks...")
with open(f"{INDEX_DIR}/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"✅ Loaded {len(chunks)} chunks")


# -------------------------
# Load embedding model
# -------------------------
print("🧬 Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)


# -------------------------
# Configure Gemini
# -------------------------
print("🤖 Initializing Gemini...")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
llm = genai.GenerativeModel(GEMINI_MODEL)


# -------------------------
# Ask function
# -------------------------
def ask(question: str):
    print("\n❓ Question:", question)

    # Embed question
    query_embedding = embedder.encode([question])

    # Search FAISS
    distances, indices = index.search(query_embedding, TOP_K)

    retrieved_chunks = [chunks[i] for i in indices[0]]

    # Debug: show retrieved text
    print("\n📚 Retrieved chunks:")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk[:400].replace("\n", " "), "...")

    # Build strict RAG prompt
    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""
You are a helpful assistant that answers questions using ONLY the provided context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Use ONLY the information in the context.
- If the answer is not present, respond with exactly:
  "Answer not found in the documents."
- Do NOT use outside knowledge.
- Keep the answer concise.

ANSWER:
""".strip()

    # Call Gemini
    try:
        response = llm.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 800
            }
        )

        print("\n🤖 Answer:\n")
        
        # Check if response has text
        if hasattr(response, 'text'):
            print(response.text.strip())
        else:
            print("⚠️ No text in response. Full response:")
            print(response)
            
    except Exception as e:
        print(f"\n❌ Error generating response: {e}")


# -------------------------
# Interactive loop
# -------------------------
if __name__ == "__main__":
    print("\n✅ RAG system ready.")
    print("Type your question or 'exit' to quit.\n")

    while True:
        q = input("Ask a question: ").strip()
        if q.lower() == "exit":
            break
        ask(q)
