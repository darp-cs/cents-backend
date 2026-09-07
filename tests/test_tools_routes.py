from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
import uuid

import pytest
from fastapi import HTTPException, status
from sqlalchemy import delete, func, select

from app.db.base import AsyncSessionLocal, init_db
from app.db.models import ToolDefinition
from app.routes.tools import (
    ToolCodeTestRunRequest,
    ToolEnabledPatchRequest,
    ToolRegisterRequest,
    ToolUpdateRequest,
    list_tools,
    register_tool,
    remove_tool,
    run_tool_code_test,
    set_tool_enabled,
    update_tool,
)


def _run(coro):
    return asyncio.run(coro)


def _fake_user() -> Any:
    return SimpleNamespace(id=uuid.uuid4(), is_active=True)


def test_register_tool_generates_embedding_and_indexes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _test() -> None:
        await init_db()

        captured: dict[str, Any] = {}

        async def _fake_embed_texts(texts: list[str], **kwargs) -> list[list[float]]:
            captured["texts"] = texts
            captured["embed_kwargs"] = kwargs
            return [[0.11, 0.22, 0.33]]

        def _fake_upsert_tool_embedding(tool_id: str, *, name: str, description: str, embedding: list[float], enabled: bool):
            captured["indexed"] = {
                "tool_id": tool_id,
                "name": name,
                "description": description,
                "embedding": embedding,
                "enabled": enabled,
            }

        monkeypatch.setattr("app.routes.tools.embed_texts", _fake_embed_texts)
        monkeypatch.setattr("app.routes.tools.upsert_tool_embedding", _fake_upsert_tool_embedding)

        async with AsyncSessionLocal() as session:
            await session.execute(delete(ToolDefinition))
            await session.commit()

            user = _fake_user()
            response = await register_tool(
                payload=ToolRegisterRequest(
                    name="ledger_lookup",
                    description="Look up latest ledger balances.",
                    python_code="def run(input_data, context):\n    return {'ok': True, 'amount': input_data.get('amount')}",
                    python_entrypoint="run",
                ),
                user=user,
                session=session,
            )

            assert response["name"] == "ledger_lookup"
            assert response["enabled"] is True
            assert response["has_python_code"] is True
            assert response["python_entrypoint"] == "run"

            count_result = await session.execute(select(func.count()).select_from(ToolDefinition))
            assert int(count_result.scalar_one()) == 1

            stored_result = await session.execute(select(ToolDefinition).where(ToolDefinition.name == "ledger_lookup"))
            stored = stored_result.scalar_one()
            assert stored.enabled is True
            assert stored.python_entrypoint == "run"
            assert "def run" in str(stored.python_code)

            assert captured["texts"] == ["ledger_lookup\nLook up latest ledger balances."]
            assert captured["embed_kwargs"]["metadata"]["entity"] == "tool_definition"
            assert captured["embed_kwargs"]["metadata"]["user_id"] == str(user.id)
            assert captured["indexed"]["tool_id"] == str(stored.id)
            assert captured["indexed"]["enabled"] is True

    _run(_test())


def test_list_tools_filters_by_enabled_flag() -> None:
    async def _test() -> None:
        await init_db()

        async with AsyncSessionLocal() as session:
            await session.execute(delete(ToolDefinition))
            await session.commit()

            session.add_all(
                [
                    ToolDefinition(name="enabled_tool", description="enabled", enabled=True),
                    ToolDefinition(name="disabled_tool", description="disabled", enabled=False),
                ]
            )
            await session.commit()

            all_tools = await list_tools(user=_fake_user(), enabled=None, session=session)
            enabled_tools = await list_tools(user=_fake_user(), enabled=True, session=session)
            disabled_tools = await list_tools(user=_fake_user(), enabled=False, session=session)

            assert len(all_tools) == 2
            assert [item["name"] for item in enabled_tools] == ["enabled_tool"]
            assert [item["name"] for item in disabled_tools] == ["disabled_tool"]

    _run(_test())


def test_update_and_enable_patch_reindex_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _test() -> None:
        await init_db()

        upsert_calls: list[dict[str, Any]] = []

        async def _fake_embed_texts(texts: list[str], **kwargs) -> list[list[float]]:
            del kwargs
            return [[0.5, 0.6, 0.7] for _ in texts]

        def _fake_upsert_tool_embedding(tool_id: str, *, name: str, description: str, embedding: list[float], enabled: bool):
            upsert_calls.append(
                {
                    "tool_id": tool_id,
                    "name": name,
                    "description": description,
                    "embedding": embedding,
                    "enabled": enabled,
                }
            )

        monkeypatch.setattr("app.routes.tools.embed_texts", _fake_embed_texts)
        monkeypatch.setattr("app.routes.tools.upsert_tool_embedding", _fake_upsert_tool_embedding)

        async with AsyncSessionLocal() as session:
            await session.execute(delete(ToolDefinition))
            await session.commit()

            seed = ToolDefinition(name="ledger_lookup", description="Old desc", enabled=True)
            session.add(seed)
            await session.commit()
            await session.refresh(seed)

            updated = await update_tool(
                tool_id=seed.id,
                payload=ToolUpdateRequest(
                    name="ledger_lookup_v2",
                    description="Updated lookup behavior.",
                    enabled=False,
                    python_code="def execute(payload, context):\n    return {'name': context.get('tool_name'), 'payload': payload}",
                    python_entrypoint="execute",
                ),
                user=_fake_user(),
                session=session,
            )

            assert updated["name"] == "ledger_lookup_v2"
            assert updated["enabled"] is False
            assert updated["python_entrypoint"] == "execute"

            patched = await set_tool_enabled(
                tool_id=seed.id,
                payload=ToolEnabledPatchRequest(enabled=True),
                user=_fake_user(),
                session=session,
            )
            assert patched["enabled"] is True

            refreshed = await session.execute(select(ToolDefinition).where(ToolDefinition.id == seed.id))
            row = refreshed.scalar_one()
            assert row.name == "ledger_lookup_v2"
            assert row.description == "Updated lookup behavior."
            assert row.enabled is True
            assert row.python_entrypoint == "execute"
            assert "def execute" in str(row.python_code)

            assert len(upsert_calls) == 2
            assert upsert_calls[0]["enabled"] is False
            assert upsert_calls[1]["enabled"] is True

    _run(_test())


def test_remove_tool_deletes_vector_and_database_record(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _test() -> None:
        await init_db()

        deleted_ids: list[str] = []

        def _fake_delete_tool(tool_id: str) -> None:
            deleted_ids.append(tool_id)

        monkeypatch.setattr("app.routes.tools.delete_tool", _fake_delete_tool)

        async with AsyncSessionLocal() as session:
            await session.execute(delete(ToolDefinition))
            await session.commit()

            seed = ToolDefinition(name="remove_me", description="to remove", enabled=True)
            session.add(seed)
            await session.commit()
            await session.refresh(seed)

            response = await remove_tool(tool_id=seed.id, user=_fake_user(), session=session)
            assert response is None

            count_result = await session.execute(select(func.count()).select_from(ToolDefinition))
            assert int(count_result.scalar_one()) == 0
            assert deleted_ids == [str(seed.id)]

    _run(_test())


def test_register_tool_rejects_duplicate_name() -> None:
    async def _test() -> None:
        await init_db()

        async with AsyncSessionLocal() as session:
            await session.execute(delete(ToolDefinition))
            await session.commit()

            session.add(ToolDefinition(name="duplicate", description="seed", enabled=True))
            await session.commit()

            with pytest.raises(HTTPException) as exc:
                await register_tool(
                    payload=ToolRegisterRequest(name="duplicate", description="new"),
                    user=_fake_user(),
                    session=session,
                )

            assert exc.value.status_code == status.HTTP_409_CONFLICT

    _run(_test())


def test_register_tool_rejects_python_syntax_errors() -> None:
    async def _test() -> None:
        await init_db()

        async with AsyncSessionLocal() as session:
            await session.execute(delete(ToolDefinition))
            await session.commit()

            with pytest.raises(HTTPException) as exc:
                await register_tool(
                    payload=ToolRegisterRequest(
                        name="broken_tool",
                        description="bad code",
                        python_code="def run(input_data, context)\n    return {}",
                    ),
                    user=_fake_user(),
                    session=session,
                )

            assert exc.value.status_code == status.HTTP_400_BAD_REQUEST
            assert "syntax error" in str(exc.value.detail).lower()

    _run(_test())


def test_tool_test_run_compiles_and_executes_python_code() -> None:
    async def _test() -> None:
        payload = ToolCodeTestRunRequest(
            python_code=(
                "def run(input_data, context):\n"
                "    total = input_data.get('left', 0) + input_data.get('right', 0)\n"
                "    return {'total': total, 'tool': context.get('tool_name', 'unknown')}"
            ),
            python_entrypoint="run",
            sample_input={"left": 2, "right": 3},
            sample_context={"tool_name": "adder"},
            execute=True,
        )

        response = await run_tool_code_test(payload=payload, user=_fake_user())

        assert response["compile_ok"] is True
        assert response["executed"] is True
        assert response["success"] is True
        assert response["output"] == {"total": 5, "tool": "adder"}

    _run(_test())


def test_tool_test_run_returns_compile_failure() -> None:
    async def _test() -> None:
        with pytest.raises(HTTPException) as exc:
            await run_tool_code_test(
                payload=ToolCodeTestRunRequest(python_code="def run(:\n    return 1", execute=True),
                user=_fake_user(),
            )

        assert exc.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "syntax error" in str(exc.value.detail).lower()

    _run(_test())
