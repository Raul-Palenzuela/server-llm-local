"""
Backend RAG: igual que backend.py, pero antes de mandar la pregunta al modelo,
busca fragmentos relevantes en la base de datos vectorial (indexada con rag_index.py)
y los añade como contexto.

Instalación:
    pip install fastapi uvicorn httpx chromadb sentence-transformers

Ejecución:
    python3 backend_rag.py
    (sirve en http://localhost:8002)
"""

import httpx
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "maya_codebase"
N_RESULTS = 6  # cuántos fragmentos recuperar por pregunta

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
client = chromadb.PersistentClient(path=DB_PATH)


def get_collection():
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


class ChatRequest(BaseModel):
    message: str
    history: list = []


def retrieve_context(query: str) -> tuple[str, list[str]]:
    """Busca los fragmentos más relevantes para la pregunta. Devuelve (texto_contexto, fuentes)."""
    try:
        collection = get_collection()
    except Exception:
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

    return "\n\n".join(context_parts), list(dict.fromkeys(sources))  # fuentes sin duplicados


@app.get("/health")
def health():
    try:
        collection = get_collection()
        count = collection.count()
        return {"status": "ok", "indexed_chunks": count}
    except Exception:
        return {"status": "ok", "indexed_chunks": 0, "warning": "No hay índice creado todavía. Ejecuta rag_index.py primero."}


@app.post("/chat")
async def chat(req: ChatRequest):
    context, sources = retrieve_context(req.message)

    if context:
        system_prompt = (
            "Eres un asistente que responde preguntas sobre el código del proyecto Maya. "
            "Usa SOLO la siguiente información recuperada del código real para responder. "
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
        return {
            "reply": f"[Error de vLLM] {data}",
            "sources": sources,
        }

    reply = data["choices"][0]["message"]["content"]

    return {
        "reply": reply,
        "sources": sources,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
