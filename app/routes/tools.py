from __future__ import annotations

from typing import Annotated, Any
import re
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.users import current_active_user
from app.config import settings
from app.db.base import get_async_session
from app.db.models import ToolDefinition, User
from app.llm.client import LLMClientError, embed_texts
from app.tools.python_runner import ToolExecutionError, compile_python_tool_code, execute_python_tool_code
from app.vector_store import delete_tool, upsert_tool_embedding

router = APIRouter()
_ENTRYPOINT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ToolRegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    description: str = Field(min_length=1, max_length=20000)
    enabled: bool = True
    python_code: str | None = Field(default=None, max_length=120000)
    python_entrypoint: str = Field(default="run", min_length=1, max_length=128)

    @field_validator("name", "description")
    @classmethod
    def trim_text_fields(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Field cannot be empty")
        return trimmed

    @field_validator("python_code")
    @classmethod
    def normalize_python_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.rstrip()
        return normalized or None

    @field_validator("python_entrypoint")
    @classmethod
    def validate_python_entrypoint(cls, value: str) -> str:
        normalized = value.strip()
        if not _ENTRYPOINT_PATTERN.fullmatch(normalized):
            raise ValueError("python_entrypoint must be a valid Python identifier.")
        return normalized


class ToolUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    description: str = Field(min_length=1, max_length=20000)
    enabled: bool | None = None
    python_code: str | None = Field(default=None, max_length=120000)
    python_entrypoint: str = Field(default="run", min_length=1, max_length=128)

    @field_validator("name", "description")
    @classmethod
    def trim_text_fields(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Field cannot be empty")
        return trimmed

    @field_validator("python_code")
    @classmethod
    def normalize_python_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.rstrip()
        return normalized or None

    @field_validator("python_entrypoint")
    @classmethod
    def validate_python_entrypoint(cls, value: str) -> str:
        normalized = value.strip()
        if not _ENTRYPOINT_PATTERN.fullmatch(normalized):
            raise ValueError("python_entrypoint must be a valid Python identifier.")
        return normalized


class ToolEnabledPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class ToolCodeTestRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python_code: str = Field(min_length=1, max_length=120000)
    python_entrypoint: str = Field(default="run", min_length=1, max_length=128)
    sample_input: dict[str, Any] = Field(default_factory=dict)
    sample_context: dict[str, Any] = Field(default_factory=dict)
    execute: bool = True

    @field_validator("python_code")
    @classmethod
    def normalize_code(cls, value: str) -> str:
        normalized = value.rstrip()
        if not normalized:
            raise ValueError("python_code cannot be empty")
        return normalized

    @field_validator("python_entrypoint")
    @classmethod
    def validate_entrypoint(cls, value: str) -> str:
        normalized = value.strip()
        if not _ENTRYPOINT_PATTERN.fullmatch(normalized):
            raise ValueError("python_entrypoint must be a valid Python identifier.")
        return normalized


async def get_db_session() -> AsyncSession:
    async for session in get_async_session():
        yield session


def _serialize_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "id": str(tool.id),
        "name": tool.name,
        "description": tool.description,
        "enabled": bool(tool.enabled),
        "python_code": tool.python_code,
        "python_entrypoint": tool.python_entrypoint,
        "has_python_code": bool(tool.python_code and tool.python_code.strip()),
    }


def _tool_embedding_input(name: str, description: str) -> str:
    return f"{name}\n{description}".strip()


async def _embed_tool(name: str, description: str, *, user_id: str) -> list[float]:
    try:
        vectors = await embed_texts(
            [_tool_embedding_input(name, description)],
            metadata={
                "user_id": user_id,
                "entity": "tool_definition",
            },
        )
    except LLMClientError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    if not vectors or not vectors[0]:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Embedding service returned an empty embedding.",
        )
    return vectors[0]


async def _get_tool_or_404(session: AsyncSession, tool_id: uuid.UUID) -> ToolDefinition:
    result = await session.execute(select(ToolDefinition).where(ToolDefinition.id == tool_id))
    tool = result.scalar_one_or_none()
    if tool is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tool not found")
    return tool


def _validate_python_tool_or_raise(code: str | None, entrypoint: str) -> None:
    del entrypoint

    if code is None:
        return

    if len(code) > settings.tool_code_max_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tool code exceeds maximum length of {settings.tool_code_max_length} characters.",
        )

    try:
        compile_python_tool_code(code)
    except SyntaxError as exc:
        line = exc.lineno or 1
        detail = f"Tool code syntax error on line {line}: {exc.msg}"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail) from exc


async def _ensure_name_available(session: AsyncSession, *, name: str, exclude_id: uuid.UUID | None = None) -> None:
    result = await session.execute(select(ToolDefinition).where(ToolDefinition.name == name))
    existing = result.scalar_one_or_none()
    if existing is None:
        return
    if exclude_id is not None and existing.id == exclude_id:
        return
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Tool name already exists")


@router.get("")
async def list_tools(
    user: Annotated[User, Depends(current_active_user)],
    enabled: bool | None = None,
    session: AsyncSession = Depends(get_db_session),
):
    del user

    statement = select(ToolDefinition).order_by(ToolDefinition.name.asc())
    if enabled is not None:
        statement = statement.where(ToolDefinition.enabled.is_(enabled))
    result = await session.execute(statement)
    return [_serialize_tool(tool) for tool in result.scalars().all()]


@router.post("", response_model=dict)
async def register_tool(
    payload: ToolRegisterRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_db_session),
):
    await _ensure_name_available(session, name=payload.name)
    _validate_python_tool_or_raise(payload.python_code, payload.python_entrypoint)

    embedding = await _embed_tool(payload.name, payload.description, user_id=str(user.id))
    tool = ToolDefinition(
        name=payload.name,
        description=payload.description,
        enabled=payload.enabled,
        python_code=payload.python_code,
        python_entrypoint=payload.python_entrypoint,
    )
    session.add(tool)

    await session.flush()
    upsert_tool_embedding(
        str(tool.id),
        name=tool.name,
        description=tool.description,
        embedding=embedding,
        enabled=tool.enabled,
    )

    await session.commit()
    await session.refresh(tool)
    return _serialize_tool(tool)


@router.post("/test-run", response_model=dict)
async def run_tool_code_test(
    payload: ToolCodeTestRunRequest,
    user: Annotated[User, Depends(current_active_user)],
):
    del user

    _validate_python_tool_or_raise(payload.python_code, payload.python_entrypoint)

    if not payload.execute:
        return {
            "compile_ok": True,
            "executed": False,
            "success": True,
            "output": None,
            "error": None,
            "traceback": None,
        }

    try:
        execution_result = execute_python_tool_code(
            payload.python_code,
            tool_input=payload.sample_input,
            context=payload.sample_context,
            entrypoint=payload.python_entrypoint,
            timeout_seconds=settings.tool_code_execution_timeout_seconds,
        )
    except ToolExecutionError as exc:
        return {
            "compile_ok": True,
            "executed": True,
            "success": False,
            "output": None,
            "error": str(exc),
            "traceback": exc.traceback_text,
        }

    return {
        "compile_ok": True,
        "executed": True,
        "success": True,
        "output": execution_result.output,
        "error": None,
        "traceback": None,
    }


@router.put("/{tool_id}")
async def update_tool(
    tool_id: uuid.UUID,
    payload: ToolUpdateRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_db_session),
):
    tool = await _get_tool_or_404(session, tool_id)
    await _ensure_name_available(session, name=payload.name, exclude_id=tool.id)
    _validate_python_tool_or_raise(payload.python_code, payload.python_entrypoint)

    embedding = await _embed_tool(payload.name, payload.description, user_id=str(user.id))
    tool.name = payload.name
    tool.description = payload.description
    tool.python_code = payload.python_code
    tool.python_entrypoint = payload.python_entrypoint
    if payload.enabled is not None:
        tool.enabled = payload.enabled

    await session.flush()
    upsert_tool_embedding(
        str(tool.id),
        name=tool.name,
        description=tool.description,
        embedding=embedding,
        enabled=tool.enabled,
    )

    await session.commit()
    await session.refresh(tool)
    return _serialize_tool(tool)


@router.patch("/{tool_id}/enabled")
async def set_tool_enabled(
    tool_id: uuid.UUID,
    payload: ToolEnabledPatchRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_db_session),
):
    tool = await _get_tool_or_404(session, tool_id)

    embedding = await _embed_tool(tool.name, tool.description, user_id=str(user.id))
    tool.enabled = payload.enabled

    await session.flush()
    upsert_tool_embedding(
        str(tool.id),
        name=tool.name,
        description=tool.description,
        embedding=embedding,
        enabled=tool.enabled,
    )

    await session.commit()
    await session.refresh(tool)
    return _serialize_tool(tool)


@router.delete("/{tool_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_tool(
    tool_id: uuid.UUID,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_db_session),
):
    del user

    tool = await _get_tool_or_404(session, tool_id)
    delete_tool(str(tool.id))
    await session.delete(tool)
    await session.commit()
    return None
