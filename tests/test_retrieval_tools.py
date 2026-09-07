from __future__ import annotations

from app.graph.retrieval_tools import tool_retrieval_node
from app.llm.client import LLMClientError


def test_tool_retrieval_node_uses_embedded_query_and_enabled_filter(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _fake_embed_texts(texts: list[str], **kwargs) -> list[list[float]]:
        captured["texts"] = texts
        captured["embed_kwargs"] = kwargs
        return [[0.2, 0.3, 0.4]]

    def _fake_query_tools(query_embedding: list[float], limit: int = 5, *, enabled_only: bool = True):
        captured["query_embedding"] = query_embedding
        captured["limit"] = limit
        captured["enabled_only"] = enabled_only
        return [
            {
                "id": "tool-1",
                "name": "ledger_lookup",
                "description": "Find ledger entries",
                "enabled": True,
                "similarity": 0.91,
                "source": "chroma",
            }
        ]

    monkeypatch.setattr("app.graph.retrieval_tools.embed_texts", _fake_embed_texts)
    monkeypatch.setattr("app.graph.retrieval_tools.query_tools", _fake_query_tools)

    state = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "find ledger tools"},
        ]
    }

    updated = tool_retrieval_node(state)

    assert captured["texts"] == ["find ledger tools"]
    assert captured["enabled_only"] is True
    assert captured["query_embedding"] == [0.2, 0.3, 0.4]
    assert updated["retrieved_tools"] == [
        {
            "name": "ledger_lookup",
            "description": "Find ledger entries",
            "similarity": 0.91,
            "source": "chroma",
            "query": "find ledger tools",
        }
    ]


def test_tool_retrieval_node_falls_back_to_empty_on_embedding_error(monkeypatch) -> None:
    async def _failing_embed_texts(texts: list[str], **kwargs) -> list[list[float]]:
        del texts, kwargs
        raise LLMClientError("embedding down")

    monkeypatch.setattr("app.graph.retrieval_tools.embed_texts", _failing_embed_texts)

    state = {"messages": [{"role": "user", "content": "tools for budgets"}]}
    updated = tool_retrieval_node(state)

    assert updated["retrieved_tools"] == []
