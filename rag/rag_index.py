"""
Indexa archivos de código/texto en una base de datos vectorial (ChromaDB)
para usarlos como contexto en el backend RAG.

Instalación:
    pip install chromadb sentence-transformers

Uso:
    python3 rag_index.py /ruta/a/la/carpeta/del/proyecto

Esto crea/actualiza una base de datos persistente en ./chroma_db
"""

import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

CHUNK_SIZE = 800        # caracteres por fragmento
CHUNK_OVERLAP = 150     # solapamiento entre fragmentos consecutivos
DB_PATH = "./chroma_db"
COLLECTION_NAME = "maya_codebase"
EXTENSIONS = {".py", ".md", ".txt", ".json", ".yml", ".yaml"}


def chunk_text(text: str, source: str) -> list[dict]:
    """Trocea un texto largo en fragmentos solapados, con metadata del origen."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({"text": chunk, "source": source, "start": start})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def index_directory(directory: str) -> None:
    path = Path(directory)
    if not path.exists():
        print(f"Error: la carpeta '{directory}' no existe.")
        sys.exit(1)

    print(f"Indexando archivos en: {path}")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=DB_PATH)
    # Recrear la colección desde cero cada vez (evita duplicados al reindexar)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    all_chunks = []
    files_indexed = 0

    for file_path in path.rglob("*"):
        if file_path.suffix not in EXTENSIONS:
            continue
        if "__pycache__" in str(file_path) or ".git" in str(file_path):
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not text.strip():
            continue

        rel_path = str(file_path.relative_to(path))
        chunks = chunk_text(text, rel_path)
        all_chunks.extend(chunks)
        files_indexed += 1
        print(f"  {rel_path}: {len(chunks)} fragmentos")

    if not all_chunks:
        print("No se encontraron archivos para indexar.")
        return

    collection.add(
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"], "start": c["start"]} for c in all_chunks],
        ids=[f"chunk_{i}" for i in range(len(all_chunks))],
    )

    print(f"\nListo: {files_indexed} archivos, {len(all_chunks)} fragmentos indexados en {DB_PATH}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python3 rag_index.py /ruta/a/la/carpeta")
        sys.exit(1)
    index_directory(sys.argv[1])
