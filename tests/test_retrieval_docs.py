from __future__ import annotations

from app.graph.retrieval_docs import document_retrieval_node
from app.llm.client import LLMClientError


def test_document_retrieval_node_uses_embedded_query(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _fake_embed_texts(texts: list[str], **kwargs) -> list[list[float]]:
        captured["texts"] = texts
        captured["embed_kwargs"] = kwargs
        return [[0.9, 0.1, 0.4]]

    def _fake_query_documents(user_id: str, query_embedding: list[float], limit: int = 5):
        captured["user_id"] = user_id
        captured["query_embedding"] = query_embedding
        captured["limit"] = limit
        return [
            {
                "id": "doc-1",
                "chunk_text": "Budget policy update",
                "source_filename": "policy.txt",
                "similarity": 0.77,
            }
        ]

    monkeypatch.setattr("app.graph.retrieval_docs.embed_texts", _fake_embed_texts)
    monkeypatch.setattr("app.graph.retrieval_docs.query_documents", _fake_query_documents)

    state = {
        "messages": [
            {"role": "user", "content": "show me budget policy"},
        ],
        "user_id": "user-1",
    }

    updated = document_retrieval_node(state)

    assert captured["texts"] == ["show me budget policy"]
    assert captured["query_embedding"] == [0.9, 0.1, 0.4]
    assert captured["user_id"] == "user-1"
    assert updated["retrieved_docs"] == [
        {
            "chunk_text": "Budget policy update",
            "source_filename": "policy.txt",
            "similarity": 0.77,
            "query": "show me budget policy",
        }
    ]


def test_document_retrieval_node_falls_back_to_empty_on_embedding_error(monkeypatch) -> None:
    async def _failing_embed_texts(texts: list[str], **kwargs) -> list[list[float]]:
        del texts, kwargs
        raise LLMClientError("embedding down")

    monkeypatch.setattr("app.graph.retrieval_docs.embed_texts", _failing_embed_texts)

    state = {
        "messages": [{"role": "user", "content": "find docs about tax forms"}],
        "user_id": "user-2",
    }
    updated = document_retrieval_node(state)

    assert updated["retrieved_docs"] == []
