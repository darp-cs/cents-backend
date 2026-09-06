from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.template_schema import validate_template
from app.auth.users import current_active_user
from app.db.base import get_async_session
from app.db.models import AgentTemplate, User

router = APIRouter()


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
    return Response(status_code=status.HTTP_204_NO_CONTENT)
