from __future__ import annotations

import asyncio
import json
import threading
from typing import Annotated, Any
import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.compiler import (
    clear_runtime_event_emitter,
    get_compiled_agent_graph,
    invalidate_compiled_agent_graph,
    set_runtime_event_emitter,
)
from app.agents.template_schema import validate_template
from app.auth.users import current_active_user
from app.config import settings
from app.db.base import get_async_session
from app.db.models import AgentTemplate, PlatformConfig, User

router = APIRouter()
_RUN_REGISTRY_LOCK = threading.Lock()
_RUN_REGISTRY: dict[str, dict[str, Any]] = {}


class AgentTemplateCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    raw_template: dict[str, Any]

    @field_validator("name")
    @classmethod
    def trim_name(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Name cannot be empty")
        return trimmed


class AgentTemplateUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_template: dict[str, Any]


class AgentTemplateEnabledPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    version: int | None = Field(default=None, ge=1)


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = Field(min_length=1)
    version: int | None = Field(default=None, ge=1)
    thread_id: str | None = Field(default=None, min_length=1)
    parsed_data: dict[str, Any] = Field(default_factory=dict)
    service_results: dict[str, Any] = Field(default_factory=dict)
    node_llm_configs: dict[str, dict[str, str | None]] = Field(default_factory=dict)


class AgentResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: Any
    thread_id: str = Field(min_length=1)
    version: int | None = Field(default=None, ge=1)


class AgentRunStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = Field(min_length=1)
    version: int | None = Field(default=None, ge=1)
    parsed_data: dict[str, Any] = Field(default_factory=dict)
    service_results: dict[str, Any] = Field(default_factory=dict)
    node_llm_configs: dict[str, dict[str, str | None]] = Field(default_factory=dict)


class AgentRunResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: Any


def _clear_agent_run_registry() -> None:
    with _RUN_REGISTRY_LOCK:
        _RUN_REGISTRY.clear()


def _coerce_iteration_count(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed >= 0 else 0


def _register_agent_run(
    *,
    run_id: str,
    thread_id: str,
    user_id: str,
    agent_name: str,
    version: int,
) -> None:
    with _RUN_REGISTRY_LOCK:
        _RUN_REGISTRY[run_id] = {
            "run_id": run_id,
            "thread_id": thread_id,
            "user_id": user_id,
            "agent_name": agent_name,
            "version": version,
            "status": "running",
            "iteration_count": 0,
            "final_response": "",
            "error": None,
        }


def _update_agent_run(run_id: str, **changes: Any) -> None:
    with _RUN_REGISTRY_LOCK:
        record = _RUN_REGISTRY.get(run_id)
        if record is None:
            return
        record.update(changes)


def _get_agent_run_for_user(run_id: str, user_id: str) -> dict[str, Any] | None:
    with _RUN_REGISTRY_LOCK:
        record = _RUN_REGISTRY.get(run_id)
        if record is None:
            return None
        if record.get("user_id") != user_id:
            return None
        return dict(record)


def _build_run_state(
    *,
    payload: AgentRunStartRequest,
    platform_guardrails: dict[str, Any],
    template_guardrails: dict[str, Any],
) -> dict[str, Any]:
    return {
        "input": payload.input,
        "parsed_data": dict(payload.parsed_data),
        "messages": [{"role": "user", "content": payload.input}],
        "iteration_count": 0,
        "service_results": dict(payload.service_results),
        "interrupt_payload": None,
        "final_response": "",
        "node_llm_configs": dict(payload.node_llm_configs),
        "platform_guardrails": platform_guardrails,
        "template_guardrails": template_guardrails,
    }


def _format_sse_data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"


def _queue_event(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[dict[str, Any]], payload: dict[str, Any]) -> None:
    loop.call_soon_threadsafe(queue.put_nowait, payload)


def _build_run_id(user_id: str) -> str:
    return f"{user_id}:{uuid.uuid4()}"


async def _build_run_streaming_response(
    *,
    run_id: str,
    thread_id: str,
    compiled_graph,
    invoke_input: dict[str, Any] | Command,
    config: dict[str, Any],
) -> StreamingResponse:
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    latest_node_id: dict[str, str | None] = {"value": None}

    def _emit(event: dict[str, Any]) -> None:
        event_name = str(event.get("event", "")).strip().lower()
        if event_name == "node_started":
            node_id = event.get("node_id")
            latest_node_id["value"] = str(node_id) if node_id is not None else None
        if event_name == "node_completed":
            _update_agent_run(
                run_id,
                iteration_count=_coerce_iteration_count(event.get("iteration_count", 0)),
            )

        payload = {
            "run_id": run_id,
            "thread_id": thread_id,
            **event,
        }
        _queue_event(loop, queue, payload)

    def _invoke_graph() -> dict[str, Any]:
        set_runtime_event_emitter(_emit)
        try:
            return compiled_graph.invoke(invoke_input, config)
        finally:
            clear_runtime_event_emitter()

    async def _runner() -> None:
        try:
            result = await asyncio.to_thread(_invoke_graph)
        except Exception as exc:
            _update_agent_run(run_id, status="failed", error=str(exc))
            _queue_event(
                loop,
                queue,
                {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "event": "error",
                    "node_id": latest_node_id["value"],
                    "message": str(exc),
                },
            )
            _queue_event(
                loop,
                queue,
                {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "event": "done",
                    "status": "failed",
                },
            )
            return

        iteration_count = _coerce_iteration_count(result.get("iteration_count", 0))
        _update_agent_run(run_id, iteration_count=iteration_count)

        interrupt_payload = _extract_first_interrupt_payload(compiled_graph, config)
        if interrupt_payload is not None:
            _update_agent_run(run_id, status="awaiting_input")
            _queue_event(
                loop,
                queue,
                {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "event": "interrupt_requested",
                    "node_id": interrupt_payload.get("node_id"),
                    "prompt": str(interrupt_payload.get("prompt", "")),
                },
            )
            _queue_event(
                loop,
                queue,
                {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "event": "done",
                    "awaiting_input": True,
                },
            )
            return

        error_event = result.get("error_event")
        if isinstance(error_event, dict):
            error_message = str(error_event.get("message", error_event.get("type", "execution failed")))
            _update_agent_run(run_id, status="failed", error=error_message)
            _queue_event(
                loop,
                queue,
                {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "event": "error",
                    "node_id": error_event.get("node_id"),
                    "message": error_message,
                },
            )
            _queue_event(
                loop,
                queue,
                {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "event": "done",
                    "status": "failed",
                },
            )
            return

        final_response = str(result.get("final_response", ""))
        _update_agent_run(run_id, status="succeeded", final_response=final_response)
        _queue_event(
            loop,
            queue,
            {
                "run_id": run_id,
                "thread_id": thread_id,
                "event": "done",
                "final_response": final_response,
            },
        )

    worker = asyncio.create_task(_runner())

    async def _event_stream():
        while True:
            payload = await queue.get()
            yield _format_sse_data(payload)
            if payload.get("event") == "done":
                break
        if not worker.done():
            await worker

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


def _parse_json(value: str | None) -> Any:
    if value is None:
        return None

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _serialize_record(record: AgentTemplate) -> dict[str, Any]:
    return {
        "id": str(record.id),
        "name": record.name,
        "version": record.version,
        "raw_template": _parse_json(record.raw_template),
        "is_valid": record.is_valid,
        "validation_errors": _parse_json(record.validation_errors),
        "enabled": record.enabled,
        "created_at": record.created_at.isoformat(),
    }


def _should_auto_enable_new_version(existing_enabled_count: int, is_valid: bool) -> bool:
    return existing_enabled_count == 0 and is_valid


def _apply_enabled_policy(
    records: list[AgentTemplate],
    target: AgentTemplate,
    desired_enabled: bool,
) -> None:
    if desired_enabled:
        if not target.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid templates cannot be enabled.",
            )

        # Enabling one version must atomically disable all other versions for that agent name.
        for record in records:
            record.enabled = record.version == target.version
        return

    currently_enabled = [record for record in records if record.enabled]
    if target.enabled and len(currently_enabled) <= 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one version must remain enabled for this agent.",
        )

    target.enabled = False


async def _get_latest_for_name(session: AsyncSession, name: str) -> AgentTemplate | None:
    result = await session.execute(
        select(AgentTemplate)
        .where(AgentTemplate.name == name)
        .order_by(AgentTemplate.version.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _get_versions_for_name(session: AsyncSession, name: str) -> list[AgentTemplate]:
    result = await session.execute(
        select(AgentTemplate)
        .where(AgentTemplate.name == name)
        .order_by(AgentTemplate.version.desc())
    )
    return list(result.scalars().all())


def _extract_first_interrupt_payload(compiled_graph, config: dict[str, Any]) -> dict[str, Any] | None:
    try:
        snapshot = compiled_graph.get_state(config)
    except ValueError:
        return None

    for task in getattr(snapshot, "tasks", ()):
        interrupts = getattr(task, "interrupts", ())
        if not interrupts:
            continue
        raw_value = interrupts[0].value
        return raw_value if isinstance(raw_value, dict) else {"value": raw_value}
    return None


def _deserialize_banned_topics(raw: str | None) -> list[str]:
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    return [str(item).strip() for item in parsed if str(item).strip()]


async def _get_or_create_platform_config(session: AsyncSession) -> PlatformConfig:
    result = await session.execute(select(PlatformConfig).order_by(PlatformConfig.id.asc()).limit(1))
    record = result.scalar_one_or_none()
    if record is not None:
        return record

    record = PlatformConfig(
        id=1,
        guidelines_text="",
        banned_topics="[]",
        judge_enabled=settings.llm_judge_enabled,
        max_retries=3,
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return record


async def _build_platform_guardrails(session: AsyncSession) -> dict[str, Any]:
    record = await _get_or_create_platform_config(session)
    return {
        "guidelines_text": record.guidelines_text,
        "banned_topics": _deserialize_banned_topics(record.banned_topics),
        "judge_enabled": record.judge_enabled,
        "max_iterations": record.max_retries,
    }


def _build_template_guardrails(raw_template: dict[str, Any]) -> dict[str, Any]:
    raw_guardrails = raw_template.get("guardrails")
    if not isinstance(raw_guardrails, dict):
        return {
            "max_iterations": None,
            "banned_topics_override": None,
            "judge_enabled_override": None,
        }

    template_max_iterations = raw_guardrails.get("max_iterations")
    if isinstance(template_max_iterations, bool):
        template_max_iterations = None
    elif template_max_iterations is not None:
        try:
            template_max_iterations = int(template_max_iterations)
        except (TypeError, ValueError):
            template_max_iterations = None

    raw_banned_topics_override = raw_guardrails.get("banned_topics_override")
    banned_topics_override = (
        [str(topic).strip() for topic in raw_banned_topics_override if str(topic).strip()]
        if isinstance(raw_banned_topics_override, list)
        else None
    )

    judge_enabled_override = raw_guardrails.get("judge_enabled_override")
    if judge_enabled_override is not None:
        judge_enabled_override = bool(judge_enabled_override)

    return {
        "max_iterations": template_max_iterations,
        "banned_topics_override": banned_topics_override,
        "judge_enabled_override": judge_enabled_override,
    }


async def _resolve_runnable_template(
    session: AsyncSession,
    *,
    name: str,
    requested_version: int | None,
) -> tuple[AgentTemplate, dict[str, Any]]:
    records = await _get_versions_for_name(session, name)
    if not records:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent template not found")

    if requested_version is None:
        record = next((item for item in records if item.enabled), None)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No enabled version is available for this agent.",
            )
    else:
        record = next((item for item in records if item.version == requested_version), None)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent template not found")
        if not record.enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Requested version is disabled.",
            )

    if not record.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot run an invalid template version.",
        )

    raw_template = _parse_json(record.raw_template)
    if not isinstance(raw_template, dict):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored template payload is invalid.",
        )

    parsed = validate_template(raw_template)
    if parsed.template is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored template failed validation.",
        )

    return record, raw_template


async def _create_version(
    session: AsyncSession,
    name: str,
    raw_template: dict[str, Any],
) -> AgentTemplate:
    result = validate_template(raw_template)
    validation_errors = result.errors if result.errors else None

    max_version_result = await session.execute(
        select(func.max(AgentTemplate.version)).where(AgentTemplate.name == name)
    )
    max_version = max_version_result.scalar_one()
    next_version = 1 if max_version is None else int(max_version) + 1

    enabled_count_result = await session.execute(
        select(func.count())
        .select_from(AgentTemplate)
        .where(AgentTemplate.name == name, AgentTemplate.enabled.is_(True))
    )
    enabled_count = int(enabled_count_result.scalar_one())

    record = AgentTemplate(
        name=name,
        version=next_version,
        raw_template=json.dumps(raw_template, ensure_ascii=True, separators=(",", ":")),
        is_valid=result.is_valid,
        validation_errors=(
            json.dumps(validation_errors, ensure_ascii=True, separators=(",", ":"))
            if validation_errors
            else None
        ),
        enabled=_should_auto_enable_new_version(enabled_count, result.is_valid),
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)

    if result.template is not None:
        # Warm cache so this version does not need recompilation at first execution.
        get_compiled_agent_graph(name=name, version=next_version, template=result.template)

    return record


@router.post("", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_agent_template(
    payload: AgentTemplateCreateRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    latest = await _get_latest_for_name(session, payload.name)
    if latest is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent '{payload.name}' already exists. Use PUT /agents/{payload.name} to create a new version.",
        )

    record = await _create_version(session, payload.name, payload.raw_template)
    return _serialize_record(record)


@router.get("", response_model=list[dict])
async def list_latest_agents(
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    # Subquery ensures one record (the max version) is returned for each agent name.
    latest_per_name = (
        select(
            AgentTemplate.name.label("name"),
            func.max(AgentTemplate.version).label("max_version"),
        )
        .group_by(AgentTemplate.name)
        .subquery()
    )

    result = await session.execute(
        select(AgentTemplate)
        .join(
            latest_per_name,
            (AgentTemplate.name == latest_per_name.c.name)
            & (AgentTemplate.version == latest_per_name.c.max_version),
        )
        .order_by(AgentTemplate.name.asc())
    )

    return [_serialize_record(record) for record in result.scalars().all()]


@router.get("/{name}/versions", response_model=list[dict])
async def list_agent_versions(
    name: str,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    normalized_name = name.strip()
    records = await _get_versions_for_name(session, normalized_name)
    if not records:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent template not found")

    return [_serialize_record(record) for record in records]


@router.get("/{name}", response_model=dict)
async def get_latest_agent(
    name: str,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    latest = await _get_latest_for_name(session, name.strip())
    if latest is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent template not found")

    return _serialize_record(latest)


@router.post("/{name}/runs")
async def start_agent_run(
    name: str,
    payload: AgentRunStartRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    normalized_name = name.strip()
    record, raw_template = await _resolve_runnable_template(
        session,
        name=normalized_name,
        requested_version=payload.version,
    )
    parsed = validate_template(raw_template)
    if parsed.template is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored template failed validation.",
        )

    user_id = str(user.id)
    run_id = _build_run_id(user_id)
    thread_id = run_id

    _register_agent_run(
        run_id=run_id,
        thread_id=thread_id,
        user_id=user_id,
        agent_name=normalized_name,
        version=record.version,
    )

    platform_guardrails = await _build_platform_guardrails(session)
    template_guardrails = _build_template_guardrails(raw_template)
    state = _build_run_state(
        payload=payload,
        platform_guardrails=platform_guardrails,
        template_guardrails=template_guardrails,
    )

    config = {"configurable": {"thread_id": thread_id}}
    compiled_graph = get_compiled_agent_graph(
        name=normalized_name,
        version=record.version,
        template=parsed.template,
    )

    return await _build_run_streaming_response(
        run_id=run_id,
        thread_id=thread_id,
        compiled_graph=compiled_graph,
        invoke_input=state,
        config=config,
    )


@router.post("/runs/{run_id}/resume")
async def resume_agent_run(
    run_id: str,
    payload: AgentRunResumeRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    user_id = str(user.id)
    run_record = _get_agent_run_for_user(run_id, user_id)
    if run_record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    if run_record.get("status") != "awaiting_input":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Run is not awaiting input.",
        )

    agent_name = str(run_record.get("agent_name", "")).strip()
    requested_version = _coerce_iteration_count(run_record.get("version", 0))
    if not agent_name or requested_version < 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Run metadata is invalid.",
        )

    record, raw_template = await _resolve_runnable_template(
        session,
        name=agent_name,
        requested_version=requested_version,
    )
    parsed = validate_template(raw_template)
    if parsed.template is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored template failed validation.",
        )

    thread_id = str(run_record.get("thread_id", "")).strip()
    if not thread_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Run thread_id is missing.",
        )

    _update_agent_run(run_id, status="running", error=None, final_response="")

    config = {"configurable": {"thread_id": thread_id}}
    compiled_graph = get_compiled_agent_graph(
        name=agent_name,
        version=record.version,
        template=parsed.template,
    )

    return await _build_run_streaming_response(
        run_id=run_id,
        thread_id=thread_id,
        compiled_graph=compiled_graph,
        invoke_input=Command(resume=payload.answer),
        config=config,
    )


@router.get("/runs/{run_id}", response_model=dict)
async def get_agent_run(
    run_id: str,
    user: Annotated[User, Depends(current_active_user)],
):
    user_id = str(user.id)
    run_record = _get_agent_run_for_user(run_id, user_id)
    if run_record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    return {
        "run_id": run_record["run_id"],
        "thread_id": run_record["thread_id"],
        "status": run_record["status"],
        "iteration_count": _coerce_iteration_count(run_record.get("iteration_count", 0)),
    }


@router.post("/{name}/run", response_model=dict)
async def run_agent(
    name: str,
    payload: AgentRunRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    normalized_name = name.strip()
    record, raw_template = await _resolve_runnable_template(
        session,
        name=normalized_name,
        requested_version=payload.version,
    )
    parsed = validate_template(raw_template)
    if parsed.template is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored template failed validation.",
        )

    thread_id = payload.thread_id.strip() if payload.thread_id else (
        f"user:{user.id}:agent:{normalized_name}:{uuid.uuid4()}"
    )
    platform_guardrails = await _build_platform_guardrails(session)
    template_guardrails = _build_template_guardrails(raw_template)

    state = {
        "input": payload.input,
        "parsed_data": dict(payload.parsed_data),
        "messages": [{"role": "user", "content": payload.input}],
        "iteration_count": 0,
        "service_results": dict(payload.service_results),
        "interrupt_payload": None,
        "final_response": "",
        "node_llm_configs": dict(payload.node_llm_configs),
        "platform_guardrails": platform_guardrails,
        "template_guardrails": template_guardrails,
    }
    config = {"configurable": {"thread_id": thread_id}}

    compiled_graph = get_compiled_agent_graph(
        name=normalized_name,
        version=record.version,
        template=parsed.template,
    )

    result = await asyncio.to_thread(compiled_graph.invoke, state, config)
    interrupt_payload = _extract_first_interrupt_payload(compiled_graph, config)
    if interrupt_payload is not None:
        return {
            "status": "interrupted",
            "thread_id": thread_id,
            "version": record.version,
            "interrupt": interrupt_payload,
            "state": result,
        }

    error_event = result.get("error_event")
    if isinstance(error_event, dict):
        return {
            "status": "failed",
            "thread_id": thread_id,
            "version": record.version,
            "error": error_event,
            "state": result,
        }

    return {
        "status": "completed",
        "thread_id": thread_id,
        "version": record.version,
        "state": result,
        "final_response": str(result.get("final_response", "")),
    }


@router.post("/{name}/resume", response_model=dict)
async def resume_agent(
    name: str,
    payload: AgentResumeRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    normalized_name = name.strip()
    record, raw_template = await _resolve_runnable_template(
        session,
        name=normalized_name,
        requested_version=payload.version,
    )
    parsed = validate_template(raw_template)
    if parsed.template is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored template failed validation.",
        )

    config = {"configurable": {"thread_id": payload.thread_id.strip()}}
    compiled_graph = get_compiled_agent_graph(
        name=normalized_name,
        version=record.version,
        template=parsed.template,
    )

    result = await asyncio.to_thread(compiled_graph.invoke, Command(resume=payload.answer), config)
    interrupt_payload = _extract_first_interrupt_payload(compiled_graph, config)
    if interrupt_payload is not None:
        return {
            "status": "interrupted",
            "thread_id": payload.thread_id.strip(),
            "version": record.version,
            "interrupt": interrupt_payload,
            "state": result,
        }

    error_event = result.get("error_event")
    if isinstance(error_event, dict):
        return {
            "status": "failed",
            "thread_id": payload.thread_id.strip(),
            "version": record.version,
            "error": error_event,
            "state": result,
        }

    return {
        "status": "completed",
        "thread_id": payload.thread_id.strip(),
        "version": record.version,
        "state": result,
        "final_response": str(result.get("final_response", "")),
    }


@router.put("/{name}", response_model=dict)
async def create_new_agent_version(
    name: str,
    payload: AgentTemplateUpdateRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    normalized_name = name.strip()
    latest = await _get_latest_for_name(session, normalized_name)
    if latest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{normalized_name}' not found. Use POST /agents to create version 1.",
        )

    record = await _create_version(session, normalized_name, payload.raw_template)
    return _serialize_record(record)


@router.patch("/{name}/enabled", response_model=dict)
async def set_agent_enabled(
    name: str,
    payload: AgentTemplateEnabledPatchRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    normalized_name = name.strip()

    records = await _get_versions_for_name(session, normalized_name)
    if not records:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent template not found")

    if payload.version is None:
        record = records[0]
    else:
        record = next((item for item in records if item.version == payload.version), None)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent template not found")

    _apply_enabled_policy(records, record, payload.enabled)
    await session.commit()
    await session.refresh(record)
    return _serialize_record(record)


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    name: str,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    normalized_name = name.strip()
    latest = await _get_latest_for_name(session, normalized_name)
    if latest is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent template not found")

    await session.execute(delete(AgentTemplate).where(AgentTemplate.name == normalized_name))
    await session.commit()
    invalidate_compiled_agent_graph(normalized_name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
