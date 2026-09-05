import json
from typing import Any
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.users import current_active_user
from app.config import settings
from app.db.base import get_async_session
from app.db.models import User
from app.graph.graph import get_graph
from app.llm.client import LLMClientError, list_models_payload
from app.services.conversation_models import get_conversation_for_user, get_conversation_model_config

router = APIRouter()
SUPPORTED_LLM_NODES = ("generation", "judge")


class NodeLLMConfigPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_type: str = Field(min_length=1)
    model: str | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    conversation_id: str | None = None
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


def _ensure_required_node_model_types(node_llm_configs: dict[str, dict[str, str | None]]) -> None:
    missing_nodes = [node for node in SUPPORTED_LLM_NODES if not node_llm_configs.get(node, {}).get("model_type")]
    if missing_nodes:
        missing = ", ".join(missing_nodes)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required model_type config for nodes: {missing}.",
        )


@router.get("/models")
async def get_available_models(
    user: Annotated[User, Depends(current_active_user)],
):
    del user

    try:
        payload = await list_models_payload()
    except LLMClientError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    models = payload.get("models", [])
    folders = payload.get("folders", {})

    return {
        "models": models,
        "folders": folders,
        "defaults": {
            "node_llm_configs": _build_default_node_llm_configs(),
        },
        "supported_nodes": list(SUPPORTED_LLM_NODES),
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

    saved_node_llm_configs: dict[str, dict[str, str | None]] = {}

    if payload.conversation_id:
        conversation = await get_conversation_for_user(session, payload.conversation_id, user.id)
        if conversation is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

        saved_node_llm_configs = await get_conversation_model_config(session, conversation.id)

    payload_node_llm_configs = _normalize_node_llm_configs(payload.node_llm_configs)
    effective_node_llm_configs = {
        **_build_default_node_llm_configs(),
        **_normalize_node_llm_configs(saved_node_llm_configs),
        **payload_node_llm_configs,
    }

    _ensure_required_node_model_types(effective_node_llm_configs)

    state = {
        "messages": [{"role": "user", "content": payload.message}],
        "user_id": user_id,
        "thread_id": thread_id,
        "node_llm_configs": effective_node_llm_configs,
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
