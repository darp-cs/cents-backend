from __future__ import annotations

from threading import Lock

import chromadb
from chromadb.config import Settings as ChromaClientSettings

from app.config import settings

_client = None
_documents_collection = None
_tools_collection = None
_lock = Lock()


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaClientSettings(anonymized_telemetry=False),
        )
    return _client


def get_documents_collection():
    global _documents_collection
    if _documents_collection is None:
        client = _get_client()
        _documents_collection = client.get_or_create_collection(
            name=settings.chroma_documents_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _documents_collection


def get_tools_collection():
    global _tools_collection
    if _tools_collection is None:
        client = _get_client()
        _tools_collection = client.get_or_create_collection(
            name=settings.chroma_tools_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _tools_collection


def ensure_vector_store_ready():
    with _lock:
        get_documents_collection()
        get_tools_collection()


def zero_embedding() -> list[float]:
    return [0.0 for _ in range(settings.vector_dimension)]


def upsert_documents(user_id: str, source_filename: str, items: list[tuple[str, str]]):
    collection = get_documents_collection()
    embeddings = [zero_embedding() for _ in items]
    ids = [item_id for item_id, _ in items]
    documents = [chunk_text for _, chunk_text in items]
    metadatas = [
        {
            "user_id": user_id,
            "source_filename": source_filename,
        }
        for _ in items
    ]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


def query_documents(user_id: str, query_embedding: list[float], limit: int = 5) -> list[dict]:
    collection = get_documents_collection()
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        where={"user_id": user_id},
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]

    output = []
    for doc_id, doc_text, metadata, distance in zip(ids, docs, metas, distances):
        similarity = 1.0 - float(distance)
        output.append(
            {
                "id": doc_id,
                "chunk_text": doc_text,
                "source_filename": (metadata or {}).get("source_filename", "unknown"),
                "similarity": similarity,
            }
        )
    return output


def upsert_tool(tool_id: str, name: str, description: str):
    collection = get_tools_collection()
    collection.upsert(
        ids=[tool_id],
        documents=[description],
        metadatas=[{"name": name}],
        embeddings=[zero_embedding()],
    )


def query_tools(query_embedding: list[float], limit: int = 5) -> list[dict]:
    collection = get_tools_collection()
    result = collection.query(query_embeddings=[query_embedding], n_results=limit)

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]

    output = []
    for tool_id, description, metadata, distance in zip(ids, docs, metas, distances):
        similarity = 1.0 - float(distance)
        output.append(
            {
                "id": tool_id,
                "name": (metadata or {}).get("name", "unknown"),
                "description": description,
                "similarity": similarity,
                "source": "chroma",
            }
        )
    return output
