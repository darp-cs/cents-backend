from typing import Annotated
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.users import current_active_user
from app.config import settings
from app.db.base import AsyncSessionLocal
from app.db.models import Conversation, ConversationModelConfig, User
from app.services.conversation_models import (
    get_conversation_for_user,
    get_conversation_model_config,
    upsert_conversation_model_config,
)

router = APIRouter()
SUPPORTED_LLM_NODES = ("generation", "judge")


class NodeLLMConfigPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_type: str = Field(min_length=1)
    model: str | None = None


class ConversationModelConfigPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_llm_configs: dict[str, NodeLLMConfigPayload] | None = None


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = value.strip()
    return cleaned if cleaned else None


def _normalize_node_llm_configs(raw_configs: dict[str, Any] | None) -> dict[str, dict[str, str | None]]:
    if not raw_configs:
        return {}

    normalized: dict[str, dict[str, str | None]] = {}
    for key, value in raw_configs.items():
        node_name = str(key).strip()
        if not node_name:
            continue

        model_type: str | None = None
        model: str | None = None

        if isinstance(value, NodeLLMConfigPayload):
            model_type = _clean_optional_text(value.model_type)
            model = _clean_optional_text(value.model)
        elif isinstance(value, dict):
            raw_model_type = value.get("model_type")
            raw_model = value.get("model")
            model_type = _clean_optional_text(str(raw_model_type)) if isinstance(raw_model_type, str) else None
            model = _clean_optional_text(str(raw_model)) if isinstance(raw_model, str) else None

        if not model_type:
            continue

        normalized[node_name] = {
            "model_type": model_type,
            "model": model,
        }

    return normalized


def _build_default_node_llm_configs() -> dict[str, dict[str, str | None]]:
    generation_type = settings.llm_default_generation_model_type.strip()
    judge_type = settings.llm_default_judge_model_type.strip()

    if not generation_type or not judge_type:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM node default model types are not configured.",
        )

    return {
        "generation": {
            "model_type": generation_type,
            "model": _clean_optional_text(settings.llm_default_generation_model),
        },
        "judge": {
            "model_type": judge_type,
            "model": _clean_optional_text(settings.llm_default_judge_model),
        },
    }


async def get_db_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


@router.get("", response_model=list[dict])
async def list_conversations(
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_db_session),
):
    result = await session.execute(select(Conversation).where(Conversation.user_id == user.id))
    return [
        {
            "id": str(conversation.id),
            "title": conversation.title,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
        }
        for conversation in result.scalars().all()
    ]


@router.post("", response_model=dict)
async def create_conversation(
    payload: dict,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_db_session),
):
    title = str(payload.get("title") or "New conversation")
    conversation = Conversation(user_id=user.id, title=title)
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return {
        "id": str(conversation.id),
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
    }


@router.patch("/{conversation_id}", response_model=dict)
async def rename_conversation(
    conversation_id: str,
    payload: dict,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_db_session),
):
    conversation = await get_conversation_for_user(session, conversation_id, user.id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    conversation.title = str(payload.get("title") or conversation.title)
    await session.commit()
    await session.refresh(conversation)
    return {
        "id": str(conversation.id),
        "title": conversation.title,
        "updated_at": conversation.updated_at.isoformat(),
    }


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_db_session),
):
    conversation = await get_conversation_for_user(session, conversation_id, user.id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    await session.execute(
        delete(ConversationModelConfig).where(ConversationModelConfig.conversation_id == conversation.id)
    )
    await session.delete(conversation)
    await session.commit()


@router.get("/{conversation_id}/model-config", response_model=dict)
async def get_conversation_models(
    conversation_id: str,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_db_session),
):
    conversation = await get_conversation_for_user(session, conversation_id, user.id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    saved_node_llm_configs = await get_conversation_model_config(session, conversation.id)
    effective_node_llm_configs = {
        **_build_default_node_llm_configs(),
        **_normalize_node_llm_configs(saved_node_llm_configs),
    }

    return {
        "conversation_id": str(conversation.id),
        "node_llm_configs": effective_node_llm_configs,
        "supported_nodes": list(SUPPORTED_LLM_NODES),
    }


@router.put("/{conversation_id}/model-config", response_model=dict)
async def set_conversation_models(
    conversation_id: str,
    payload: ConversationModelConfigPayload,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_db_session),
):
    conversation = await get_conversation_for_user(session, conversation_id, user.id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    node_llm_configs = await upsert_conversation_model_config(
        session,
        conversation.id,
        _normalize_node_llm_configs(payload.node_llm_configs),
    )

    effective_node_llm_configs = {
        **_build_default_node_llm_configs(),
        **_normalize_node_llm_configs(node_llm_configs),
    }

    return {
        "conversation_id": str(conversation.id),
        "node_llm_configs": effective_node_llm_configs,
        "supported_nodes": list(SUPPORTED_LLM_NODES),
    }
