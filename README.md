# Agentic RAG Schedule Assistant

Render settings:

Root Directory: leave empty

Build:
pip install -r requirements.txt

Start:
uvicorn app:app --host 0.0.0.0 --port $PORT

This version uses ChromaDB with lightweight Python embeddings and does not use SentenceTransformers or PyTorch, reducing memory usage on Render Free.
