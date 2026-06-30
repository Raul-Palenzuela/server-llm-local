# RAG module — querying a real codebase

Adds retrieval-augmented generation on top of the same local vLLM server:
index a codebase, ask questions about it, get answers grounded in the actual
source, with cited file sources.

## Architecture

```
Source files → chunking → embeddings (MiniLM) → ChromaDB (local, persistent)
                                                        ↓
User question → embeddings → similarity search → top-N chunks
                                                        ↓
            chunks + question → vLLM → grounded answer + cited sources
```

## Files

- `rag_index.py` — one-off indexing script. Reads a directory, chunks files,
  embeds them, stores in a local ChromaDB instance.
- `backend_rag.py` — FastAPI backend. Retrieves relevant chunks for each
  question and injects them into the system prompt before calling vLLM.

## Usage

```bash
pip install chromadb sentence-transformers fastapi uvicorn httpx

# Index a codebase (run once, or again after the source changes)
python3 rag_index.py /path/to/source

# Start the RAG backend
python3 backend_rag.py
```

```bash
curl -s http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "your question about the codebase"}'
```

Response includes `"sources"`: the file paths the answer was grounded in.
Empty `sources` means no relevant chunk was found — the system is instructed
to say so explicitly rather than answer from the model's general knowledge.

## Finding: embedding model is sensitive to query/content language mismatch

Tested against a real codebase (Maya, a Spanish-language voice assistant)
with a Spanish question about a function whose internals are named in
English (`detect_mood`, `_POSITIVE`, `_NEGATIVE`):

- Query in Spanish, mixed natural language: **failed** to retrieve the
  relevant chunk (`state.py`) even after increasing `N_RESULTS` and reducing
  chunk size. The model correctly reported it had no grounding for the
  question rather than hallucinating — but the retrieval itself missed.
- Same question, rephrased with the function name and English technical
  terms: **succeeded**, retrieved the correct chunk, and the model
  reproduced the exact logic and source code.

Root cause: `all-MiniLM-L6-v2` is a general-purpose, primarily English
embedding model. Cross-lingual / code-specific semantic matching is weaker
than same-language matching. This is a known limitation of small general
embedding models, not a bug in the retrieval pipeline — confirmed by
isolating the variable (same index, same `N_RESULTS`, only the query
wording changed).

Production fix would be a multilingual or code-specific embedding model
(e.g. `multilingual-e5-large`, or a code-aware embedder), at the cost of
more VRAM/compute for the embedding step.

## Tuning notes

- `CHUNK_SIZE` / `CHUNK_OVERLAP` (in `rag_index.py`) and `N_RESULTS` (in
  `backend_rag.py`) trade off precision vs. context window usage.
  `--max-model-len` on the vLLM server (set in `docker-compose.yml`) caps
  how much retrieved context can fit per request — too many/large chunks
  will exceed it and the request fails with a token-limit error.
