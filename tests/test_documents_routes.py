from __future__ import annotations

import asyncio
from io import BytesIO
from types import SimpleNamespace
from typing import Any
import uuid

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile
from sqlalchemy import delete, func, select

from app.db.base import AsyncSessionLocal, init_db
from app.db.models import Document
from app.llm.client import LLMClientError
from app.routes.documents import upload_document


def _run(coro):
    return asyncio.run(coro)


def _fake_user() -> Any:
    return SimpleNamespace(id=uuid.uuid4())


def test_upload_document_embeds_chunks_and_upserts_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _test() -> None:
        await init_db()

        captured: dict[str, Any] = {}

        async def _fake_embed_texts(texts: list[str], **kwargs) -> list[list[float]]:
            captured["texts"] = texts
            captured["embed_kwargs"] = kwargs
            return [[0.1, 0.2], [0.3, 0.4]]

        def _fake_upsert_documents(user_id: str, source_filename: str, items: list[tuple[str, str]], **kwargs) -> None:
            captured["upsert"] = {
                "user_id": user_id,
                "source_filename": source_filename,
                "items": items,
                "kwargs": kwargs,
            }

        monkeypatch.setattr("app.routes.documents.embed_texts", _fake_embed_texts)
        monkeypatch.setattr("app.routes.documents.upsert_documents", _fake_upsert_documents)

        async with AsyncSessionLocal() as session:
            await session.execute(delete(Document))
            await session.commit()

            file = UploadFile(filename="notes.txt", file=BytesIO(b"First section\n\nSecond section"))
            response = await upload_document(user=_fake_user(), file=file, session=session)

            assert response["status"] == "ok"
            assert response["count"] == 2

            count_result = await session.execute(select(func.count()).select_from(Document))
            assert int(count_result.scalar_one()) == 2

            assert captured["texts"] == ["First section", "Second section"]
            assert captured["embed_kwargs"]["metadata"]["entity"] == "document_chunk"
            assert captured["upsert"]["source_filename"] == "notes.txt"
            assert len(captured["upsert"]["items"]) == 2
            assert captured["upsert"]["kwargs"]["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]

    _run(_test())


def test_upload_document_returns_502_when_embeddings_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _test() -> None:
        await init_db()

        async def _failing_embed_texts(texts: list[str], **kwargs) -> list[list[float]]:
            del texts, kwargs
            raise LLMClientError("embedding down")

        monkeypatch.setattr("app.routes.documents.embed_texts", _failing_embed_texts)

        async with AsyncSessionLocal() as session:
            await session.execute(delete(Document))
            await session.commit()

            file = UploadFile(filename="notes.txt", file=BytesIO(b"Only one section"))

            with pytest.raises(HTTPException) as exc:
                await upload_document(user=_fake_user(), file=file, session=session)

            assert exc.value.status_code == 502

    _run(_test())
