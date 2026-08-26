from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
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


class ConversationModelConfigPayload(BaseModel):
    default_model: str | None = None
    node_models: dict[str, str] | None = None


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

    default_model, node_models = await get_conversation_model_config(session, conversation.id)

    return {
        "conversation_id": str(conversation.id),
        "default_model": default_model or settings.llm_default_chat_model,
        "node_models": node_models,
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

    default_model, node_models = await upsert_conversation_model_config(
        session,
        conversation.id,
        payload.default_model,
        payload.node_models,
    )

    return {
        "conversation_id": str(conversation.id),
        "default_model": default_model or settings.llm_default_chat_model,
        "node_models": node_models,
    }
