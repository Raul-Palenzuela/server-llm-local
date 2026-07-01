"""
Backend RAG con sesiones aisladas.
Cada sesión del navegador tiene su propia colección en memoria — los documentos
subidos en una sesión no contaminan otras sesiones, y desaparecen al cerrar.

Instalación:
    pip install fastapi uvicorn httpx chromadb sentence-transformers python-multipart pypdf

Ejecución:
    python3 backend_rag.py  (sirve en http://localhost:8002)
"""

import io
import httpx
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import Optional

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
N_RESULTS = 6
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
# EphemeralClient: solo en memoria, nada persiste al disco
client = chromadb.EphemeralClient()


def get_collection(session_id: str):
    """Cada sesión tiene su propia colección aislada."""
    return client.get_or_create_collection(
        name=f"session_{session_id}",
        embedding_function=embedding_fn,
    )


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE]
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


class ChatRequest(BaseModel):
    message: str
    session_id: str
    history: list = []


def retrieve_context(query: str, session_id: str) -> tuple[str, list[str]]:
    collection = get_collection(session_id)
    if collection.count() == 0:
        return "", []

    results = collection.query(query_texts=[query], n_results=N_RESULTS)
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not docs:
        return "", []

    context_parts = []
    sources = []
    for doc, meta in zip(docs, metadatas):
        source = meta.get("source", "desconocido")
        context_parts.append(f"--- Fragmento de {source} ---\n{doc}")
        sources.append(source)

    return "\n\n".join(context_parts), list(dict.fromkeys(sources))


@app.get("/health")
def health():
    collections = client.list_collections()
    return {
        "status": "ok",
        "active_sessions": len(collections),
        "session_names": [c.name for c in collections],
    }


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """Recibe un archivo e indexa solo en la colección de esta sesión."""
    content_bytes = await file.read()

    if file.filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            return {"error": f"No se pudo leer el PDF: {e}"}
    else:
        text = content_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        return {"error": "El archivo está vacío o no se pudo leer."}

    chunks = chunk_text(text)
    collection = get_collection(session_id)
    existing = collection.count()

    collection.add(
        documents=chunks,
        metadatas=[{"source": file.filename} for _ in chunks],
        ids=[f"{session_id}_{existing + i}" for i in range(len(chunks))],
    )

    return {"status": "ok", "filename": file.filename, "chunks_indexed": len(chunks)}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Borra explícitamente la colección de una sesión."""
    try:
        client.delete_collection(f"session_{session_id}")
        return {"status": "deleted", "session_id": session_id}
    except Exception:
        return {"status": "not_found", "session_id": session_id}


@app.post("/chat")
async def chat(req: ChatRequest):
    context, sources = retrieve_context(req.message, req.session_id)

    if context:
        system_prompt = (
            "Eres un asistente que responde preguntas sobre documentos subidos por el usuario. "
            "Usa SOLO la información recuperada de esos documentos para responder. "
            "Si la información no es suficiente, dilo explícitamente en vez de inventar.\n\n"
            f"{context}"
        )
        messages = [{"role": "system", "content": system_prompt}] + req.history + [
            {"role": "user", "content": req.message}
        ]
    else:
        messages = req.history + [{"role": "user", "content": req.message}]

    async with httpx.AsyncClient(timeout=120) as http_client:
        response = await http_client.post(
            VLLM_URL,
            json={"model": MODEL_NAME, "messages": messages},
        )
        data = response.json()

    if "choices" not in data:
        return {"reply": f"[Error de vLLM] {data}", "sources": sources}

    return {
        "reply": data["choices"][0]["message"]["content"],
        "sources": sources,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
