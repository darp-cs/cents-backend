from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import delete

from app.agents.compiler import clear_compiled_agent_graph_cache
from app.config import settings
from app.db.base import AsyncSessionLocal, init_db
from app.db.models import AgentTemplate, PlatformConfig
from app.graph.checkpointer import reset_checkpointer_connections
from app.routes.agents import (
    AgentRunResumeRequest,
    AgentRunStartRequest,
    _clear_agent_run_registry,
    get_agent_run,
    resume_agent_run,
    start_agent_run,
)


def _run(coro):
    return asyncio.run(coro)


async def _create_template(
    *,
    session,
    name: str,
    raw_template: dict[str, Any],
    version: int = 1,
) -> None:
    session.add(
        AgentTemplate(
            name=name,
            version=version,
            raw_template=json.dumps(raw_template, ensure_ascii=True, separators=(",", ":")),
            is_valid=True,
            validation_errors=None,
            enabled=True,
        )
    )
    await session.commit()


async def _collect_sse_events(response: StreamingResponse) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async for chunk in response.body_iterator:
        payload = chunk.decode() if isinstance(chunk, bytes) else str(chunk)
        frames = [frame.strip() for frame in payload.split("\n\n") if frame.strip()]
        for frame in frames:
            assert frame.startswith("data: ")
            events.append(json.loads(frame[len("data: ") :]))
    return events


def test_start_run_streams_node_progress_and_done_terminal() -> None:
    async def _test() -> None:
        await init_db()
        clear_compiled_agent_graph_cache()
        _clear_agent_run_registry()

        async with AsyncSessionLocal() as session:
            await session.execute(delete(AgentTemplate))
            await session.execute(delete(PlatformConfig))
            await session.commit()

            template_payload = {
                "template_version": "1.0",
                "entry_node": "respond",
                "nodes": [
                    {
                        "id": "respond",
                        "type": "terminal_response",
                        "config": {"template": "done"},
                    }
                ],
            }
            await _create_template(session=session, name="status-agent", raw_template=template_payload)

            user = SimpleNamespace(id=uuid.uuid4())
            response = await start_agent_run(
                name="status-agent",
                payload=AgentRunStartRequest(input="hello"),
                user=user,
                session=session,
            )

            assert response.media_type == "text/event-stream"
            events = await _collect_sse_events(response)

            event_names = [str(event.get("event", "")) for event in events]
            assert "node_started" in event_names
            assert "node_completed" in event_names
            assert event_names[-1] == "done"

            run_id = str(events[0]["run_id"])
            thread_id = str(events[0]["thread_id"])
            assert thread_id.startswith(f"{user.id}:")

            done_event = events[-1]
            assert done_event["final_response"] == "done"

            status_payload = await get_agent_run(run_id=run_id, user=user)
            assert status_payload["status"] == "succeeded"
            assert status_payload["iteration_count"] == 1

    _run(_test())


def test_start_interrupt_then_resume_streams_until_terminal(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    async def _test() -> None:
        await init_db()
        clear_compiled_agent_graph_cache()
        _clear_agent_run_registry()

        checkpoint_db = tmp_path / "runs-checkpoints.db"
        previous_database_url = settings.database_url
        monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{checkpoint_db.as_posix()}")
        await reset_checkpointer_connections()

        try:
            async with AsyncSessionLocal() as session:
                await session.execute(delete(AgentTemplate))
                await session.execute(delete(PlatformConfig))
                await session.commit()

                template_payload = {
                    "template_version": "1.0",
                    "entry_node": "ask",
                    "nodes": [
                        {
                            "id": "ask",
                            "type": "user_interrupt",
                            "config": {
                                "prompt": "Approve?",
                                "output_key": "approval",
                                "expected_type": "confirmation",
                            },
                            "next": "respond",
                        },
                        {
                            "id": "respond",
                            "type": "terminal_response",
                            "config": {"template": "Approved: {{ parsed_data.approval }}"},
                        },
                    ],
                }
                await _create_template(session=session, name="interrupt-agent", raw_template=template_payload)

                user = SimpleNamespace(id=uuid.uuid4())
                start_response = await start_agent_run(
                    name="interrupt-agent",
                    payload=AgentRunStartRequest(input="start"),
                    user=user,
                    session=session,
                )
                start_events = await _collect_sse_events(start_response)

                start_event_names = [str(event.get("event", "")) for event in start_events]
                assert "node_started" in start_event_names
                assert "interrupt_requested" in start_event_names
                assert start_event_names[-1] == "done"
                assert start_events[-1]["awaiting_input"] is True

                run_id = str(start_events[0]["run_id"])

                awaiting_payload = await get_agent_run(run_id=run_id, user=user)
                assert awaiting_payload["status"] == "awaiting_input"

                resume_response = await resume_agent_run(
                    run_id=run_id,
                    payload=AgentRunResumeRequest(answer="yes"),
                    user=user,
                    session=session,
                )
                resume_events = await _collect_sse_events(resume_response)

                resume_event_names = [str(event.get("event", "")) for event in resume_events]
                assert "node_started" in resume_event_names
                assert "node_completed" in resume_event_names
                assert resume_event_names[-1] == "done"
                assert resume_events[-1]["final_response"] == "Approved: yes"

                status_payload = await get_agent_run(run_id=run_id, user=user)
                assert status_payload["status"] == "succeeded"
        finally:
            monkeypatch.setattr(settings, "database_url", previous_database_url)
            await reset_checkpointer_connections()

    _run(_test())


def test_failed_run_emits_error_event_and_marks_failed() -> None:
    async def _test() -> None:
        await init_db()
        clear_compiled_agent_graph_cache()
        _clear_agent_run_registry()

        async with AsyncSessionLocal() as session:
            await session.execute(delete(AgentTemplate))
            await session.execute(delete(PlatformConfig))
            await session.commit()

            template_payload = {
                "template_version": "1.0",
                "entry_node": "respond",
                "nodes": [
                    {
                        "id": "respond",
                        "type": "terminal_response",
                        "config": {"template": "Missing {{ parsed_data.absent }}"},
                    }
                ],
            }
            await _create_template(session=session, name="failing-agent", raw_template=template_payload)

            user = SimpleNamespace(id=uuid.uuid4())
            response = await start_agent_run(
                name="failing-agent",
                payload=AgentRunStartRequest(input="hello"),
                user=user,
                session=session,
            )
            events = await _collect_sse_events(response)

            error_events = [event for event in events if event.get("event") == "error"]
            assert error_events
            assert str(error_events[0].get("node_id", "")) == "respond"

            run_id = str(events[0]["run_id"])
            status_payload = await get_agent_run(run_id=run_id, user=user)
            assert status_payload["status"] == "failed"

    _run(_test())


def test_run_status_and_resume_are_scoped_to_requesting_user(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    async def _test() -> None:
        await init_db()
        clear_compiled_agent_graph_cache()
        _clear_agent_run_registry()

        checkpoint_db = tmp_path / "scope-checkpoints.db"
        previous_database_url = settings.database_url
        monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{checkpoint_db.as_posix()}")
        await reset_checkpointer_connections()

        try:
            async with AsyncSessionLocal() as session:
                await session.execute(delete(AgentTemplate))
                await session.execute(delete(PlatformConfig))
                await session.commit()

                template_payload = {
                    "template_version": "1.0",
                    "entry_node": "ask",
                    "nodes": [
                        {
                            "id": "ask",
                            "type": "user_interrupt",
                            "config": {
                                "prompt": "Approve?",
                                "output_key": "approval",
                                "expected_type": "confirmation",
                            },
                            "next": "respond",
                        },
                        {
                            "id": "respond",
                            "type": "terminal_response",
                            "config": {"template": "done"},
                        },
                    ],
                }
                await _create_template(session=session, name="scoped-agent", raw_template=template_payload)

                owner = SimpleNamespace(id=uuid.uuid4())
                intruder = SimpleNamespace(id=uuid.uuid4())

                start_response = await start_agent_run(
                    name="scoped-agent",
                    payload=AgentRunStartRequest(input="start"),
                    user=owner,
                    session=session,
                )
                start_events = await _collect_sse_events(start_response)
                run_id = str(start_events[0]["run_id"])

                with pytest.raises(HTTPException) as status_exc:
                    await get_agent_run(run_id=run_id, user=intruder)
                assert status_exc.value.status_code == 404

                with pytest.raises(HTTPException) as resume_exc:
                    await resume_agent_run(
                        run_id=run_id,
                        payload=AgentRunResumeRequest(answer="yes"),
                        user=intruder,
                        session=session,
                    )
                assert resume_exc.value.status_code == 404
        finally:
            monkeypatch.setattr(settings, "database_url", previous_database_url)
            await reset_checkpointer_connections()

    _run(_test())
