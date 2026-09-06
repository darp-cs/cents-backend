from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.users import current_active_user
from app.config import settings
from app.db.base import get_async_session
from app.db.models import PlatformConfig, User

router = APIRouter()


class PlatformConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guidelines_text: str = Field(default="", max_length=8000)
    banned_topics: list[str] = Field(default_factory=list)
    judge_enabled: bool
    max_retries: int = Field(ge=0)

    @field_validator("banned_topics")
    @classmethod
    def normalize_banned_topics(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for topic in value:
            topic_text = str(topic).strip()
            if topic_text:
                normalized.append(topic_text)
        return normalized


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


def _serialize_banned_topics(topics: list[str]) -> str:
    return json.dumps(topics, ensure_ascii=True, separators=(",", ":"))


def _to_response_payload(record: PlatformConfig) -> dict:
    return {
        "guidelines_text": record.guidelines_text,
        "banned_topics": _deserialize_banned_topics(record.banned_topics),
        "judge_enabled": record.judge_enabled,
        "max_retries": record.max_retries,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }


async def _get_or_create_platform_config(session: AsyncSession) -> PlatformConfig:
    result = await session.execute(select(PlatformConfig).order_by(PlatformConfig.id.asc()).limit(1))
    record = result.scalar_one_or_none()
    if record is not None:
        return record

    record = PlatformConfig(
        id=1,
        guidelines_text="",
        banned_topics=_serialize_banned_topics([]),
        judge_enabled=settings.llm_judge_enabled,
        max_retries=3,
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return record


@router.get("", response_model=dict)
async def get_configuration(
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    record = await _get_or_create_platform_config(session)
    return _to_response_payload(record)


@router.put("", response_model=dict)
async def set_configuration(
    payload: PlatformConfigUpdateRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    record = await _get_or_create_platform_config(session)
    record.guidelines_text = payload.guidelines_text
    record.banned_topics = _serialize_banned_topics(payload.banned_topics)
    record.judge_enabled = payload.judge_enabled
    record.max_retries = payload.max_retries

    await session.commit()
    await session.refresh(record)
    return _to_response_payload(record)
