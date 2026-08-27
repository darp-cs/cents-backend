from __future__ import annotations

import json
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, ConversationModelConfig


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


def _parse_node_models(raw_json: str | None) -> dict[str, str]:
    if not raw_json:
        return {}

    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        return {}

    if not isinstance(payload, dict):
        return {}

    return _normalize_node_models({str(k): str(v) for k, v in payload.items()})


async def get_conversation_for_user(
    session: AsyncSession,
    conversation_id: str,
    user_id: uuid.UUID,
) -> Conversation | None:
    try:
        conversation_uuid = uuid.UUID(conversation_id)
    except ValueError:
        return None

    conversation = await session.get(Conversation, conversation_uuid)
    if conversation is None or conversation.user_id != user_id:
        return None

    return conversation


async def get_conversation_model_config(
    session: AsyncSession,
    conversation_id: uuid.UUID,
) -> tuple[str | None, dict[str, str]]:
    result = await session.execute(
        select(ConversationModelConfig).where(ConversationModelConfig.conversation_id == conversation_id)
    )
    config = result.scalar_one_or_none()

    if config is None:
        return None, {}

    default_model = config.default_model.strip() if config.default_model else None
    node_models = _parse_node_models(config.node_models_json)
    return default_model, node_models


async def upsert_conversation_model_config(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    default_model: str | None,
    node_models: dict[str, str] | None,
) -> tuple[str | None, dict[str, str]]:
    result = await session.execute(
        select(ConversationModelConfig).where(ConversationModelConfig.conversation_id == conversation_id)
    )
    config = result.scalar_one_or_none()

    normalized_default_model = default_model.strip() if default_model and default_model.strip() else None
    normalized_node_models = _normalize_node_models(node_models)
    node_models_json = json.dumps(normalized_node_models)

    if config is None:
        config = ConversationModelConfig(
            conversation_id=conversation_id,
            default_model=normalized_default_model,
            node_models_json=node_models_json,
        )
        session.add(config)
    else:
        config.default_model = normalized_default_model
        config.node_models_json = node_models_json

    await session.commit()
    return normalized_default_model, normalized_node_models
