from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import delete, func, select

from app.config import settings
from app.db.base import AsyncSessionLocal, init_db
from app.db.models import PlatformConfig
from app.routes.configuration import (
    PlatformConfigUpdateRequest,
    get_configuration,
    set_configuration,
)


def _run(coro):
    return asyncio.run(coro)


def test_get_configuration_creates_default_row_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _test() -> None:
        await init_db()

        async with AsyncSessionLocal() as session:
            await session.execute(delete(PlatformConfig))
            await session.commit()

            previous_judge_enabled = settings.llm_judge_enabled
            monkeypatch.setattr(settings, "llm_judge_enabled", not previous_judge_enabled)

            payload = await get_configuration(user=SimpleNamespace(), session=session)

            assert payload["guidelines_text"] == ""
            assert payload["banned_topics"] == []
            assert payload["judge_enabled"] is (not previous_judge_enabled)
            assert payload["max_retries"] == 3

            count_result = await session.execute(select(func.count()).select_from(PlatformConfig))
            assert int(count_result.scalar_one()) == 1

    _run(_test())


def test_get_configuration_is_singleton_on_repeated_reads() -> None:
    async def _test() -> None:
        await init_db()

        async with AsyncSessionLocal() as session:
            await session.execute(delete(PlatformConfig))
            await session.commit()

            first_payload = await get_configuration(user=SimpleNamespace(), session=session)
            second_payload = await get_configuration(user=SimpleNamespace(), session=session)

            assert first_payload["created_at"] == second_payload["created_at"]

            count_result = await session.execute(select(func.count()).select_from(PlatformConfig))
            assert int(count_result.scalar_one()) == 1

    _run(_test())


def test_put_configuration_updates_existing_singleton_row() -> None:
    async def _test() -> None:
        await init_db()

        async with AsyncSessionLocal() as session:
            await session.execute(delete(PlatformConfig))
            await session.commit()

            await get_configuration(user=SimpleNamespace(), session=session)

            payload = PlatformConfigUpdateRequest(
                guidelines_text="Be concise and avoid prohibited content.",
                banned_topics=["medical advice", " legal advice ", ""],
                judge_enabled=True,
                max_retries=2,
            )
            updated = await set_configuration(payload=payload, user=SimpleNamespace(), session=session)

            assert updated["guidelines_text"] == "Be concise and avoid prohibited content."
            assert updated["banned_topics"] == ["medical advice", "legal advice"]
            assert updated["judge_enabled"] is True
            assert updated["max_retries"] == 2

            row_result = await session.execute(select(PlatformConfig).limit(1))
            row = row_result.scalar_one()
            assert row.max_retries == 2

    _run(_test())


def test_put_configuration_validates_max_retries_non_negative() -> None:
    with pytest.raises(ValidationError):
        PlatformConfigUpdateRequest(
            guidelines_text="ok",
            banned_topics=[],
            judge_enabled=False,
            max_retries=-1,
        )


def test_put_configuration_validates_guidelines_length_cap() -> None:
    with pytest.raises(ValidationError):
        PlatformConfigUpdateRequest(
            guidelines_text="x" * 8001,
            banned_topics=[],
            judge_enabled=False,
            max_retries=0,
        )
