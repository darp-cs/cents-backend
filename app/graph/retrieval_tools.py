from __future__ import annotations

import asyncio
import threading

from app.llm.client import LLMClientError, embed_texts
from app.graph.state import GraphState
from app.vector_store import query_tools


def _embed_query_sync(query_text: str) -> list[float] | None:
    async def _embed() -> list[list[float]]:
        return await embed_texts([query_text], metadata={"node": "tool_retrieval"})

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


def tool_retrieval_node(state: GraphState) -> GraphState:
    query_text = ""
    for message in state.get("messages", []):
        if message.get("role") == "user":
            query_text = str(message.get("content", ""))
            break

    if not query_text.strip():
        state["retrieved_tools"] = []
        return state

    try:
        query_embedding = _embed_query_sync(query_text)
    except (LLMClientError, RuntimeError, ValueError, TypeError):
        query_embedding = None

    if not query_embedding:
        state["retrieved_tools"] = []
        return state

    tools = query_tools(query_embedding=query_embedding, limit=5, enabled_only=True)
    state["retrieved_tools"] = [
        {
            "name": item["name"],
            "description": item["description"],
            "similarity": item["similarity"],
            "source": item["source"],
            "query": query_text,
        }
        for item in tools
    ]
    return state
