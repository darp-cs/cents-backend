from __future__ import annotations

import asyncio
import threading

from app.graph.state import GraphState
from app.llm.client import LLMClientError, embed_texts
from app.vector_store import query_documents


def _embed_query_sync(query_text: str) -> list[float] | None:
    async def _embed() -> list[list[float]]:
        return await embed_texts([query_text], metadata={"node": "doc_retrieval"})

    try:
        vectors = asyncio.run(_embed())
        return vectors[0] if vectors else None
    except RuntimeError:
        result: dict[str, list[list[float]] | None] = {"vectors": None}
        error: dict[str, Exception | None] = {"error": None}

        def _run() -> None:
            try:
                result["vectors"] = asyncio.run(_embed())
            except Exception as exc:  # pragma: no cover - defensive fallback path.
                error["error"] = exc

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        worker.join()
        if error["error"] is not None:
            raise error["error"]
        vectors = result["vectors"] or []
        return vectors[0] if vectors else None


def document_retrieval_node(state: GraphState) -> GraphState:
    query_text = ""
    for message in state.get("messages", []):
        if message.get("role") == "user":
            query_text = str(message.get("content", ""))
            break

    if not query_text.strip():
        state["retrieved_docs"] = []
        return state

    try:
        query_embedding = _embed_query_sync(query_text)
    except (LLMClientError, RuntimeError, ValueError, TypeError):
        query_embedding = None

    if not query_embedding:
        state["retrieved_docs"] = []
        return state

    user_id = str(state.get("user_id", ""))
    docs = query_documents(user_id=user_id, query_embedding=query_embedding, limit=5)
    state["retrieved_docs"] = [
        {
            "chunk_text": item["chunk_text"],
            "source_filename": item["source_filename"],
            "similarity": item["similarity"],
            "query": query_text,
        }
        for item in docs
    ]
    return state
