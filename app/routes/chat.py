import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.users import current_active_user
from app.config import settings
from app.db.base import get_async_session
from app.db.models import User
from app.graph.graph import get_graph
from app.llm.client import LLMClientError, list_models
from app.services.conversation_models import get_conversation_for_user, get_conversation_model_config

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    model: str | None = None
    node_models: dict[str, str] | None = None


def _normalize_node_models(node_models: dict[str, str] | None) -> dict[str, str]:
    if not node_models:
        return {}

    normalized: dict[str, str] = {}
    for key, value in node_models.items():
        clean_key = str(key).strip()
        clean_value = str(value).strip()
        if clean_key and clean_value:
            normalized[clean_key] = clean_value

    return normalized


@router.get("/models")
async def get_available_models(
    user: Annotated[User, Depends(current_active_user)],
):
    del user

    try:
        models = await list_models()
    except LLMClientError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return {
        "models": models,
        "defaults": {
            "chat": settings.llm_default_chat_model,
            "nodes": {
                "generation": settings.llm_default_chat_model,
                "judge": settings.llm_default_judge_model,
            },
        },
        "supported_nodes": ["generation", "judge"],
    }


@router.post("")
async def chat(
    payload: ChatRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    # TODO: rate limiting / concurrency: a single local LLM worker or queue will serialize concurrent user requests.
    user_id = str(user.id)
    conversation_id = payload.conversation_id or "default"
    thread_id = f"{user_id}:{conversation_id}"
    graph = await get_graph()

    saved_model: str | None = None
    saved_node_models: dict[str, str] = {}

    if payload.conversation_id:
        conversation = await get_conversation_for_user(session, payload.conversation_id, user.id)
        if conversation is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

        saved_model, saved_node_models = await get_conversation_model_config(session, conversation.id)

    payload_node_models = _normalize_node_models(payload.node_models)
    effective_node_models = {**saved_node_models, **payload_node_models}

    effective_model = payload.model
    if effective_model:
        effective_model = effective_model.strip()
    if not effective_model:
        effective_model = saved_model or settings.llm_default_chat_model

    state = {
        "messages": [{"role": "user", "content": payload.message}],
        "user_id": user_id,
        "thread_id": thread_id,
        "chat_model": effective_model,
        "node_models": effective_node_models,
        "retry_count": 0,
    }

    async def event_stream():
        try:
            result = await graph.ainvoke(state)
            generated_response = str(result.get("generated_response", "")).strip()

            if not generated_response:
                yield f"data: {json.dumps({'error': 'No response generated.'})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            yield f"data: {json.dumps({'message': generated_response})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
