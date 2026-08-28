from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, ConversationModelConfig


NodeLLMConfig = dict[str, str | None]


def _normalize_node_llm_configs(node_llm_configs: dict[str, Any] | None) -> dict[str, NodeLLMConfig]:
    if not node_llm_configs:
        return {}

    normalized: dict[str, NodeLLMConfig] = {}
    for key, value in node_llm_configs.items():
        node_name = str(key).strip()
        if not node_name or not isinstance(value, dict):
            continue

        raw_model_type = value.get("model_type")
        model_type = str(raw_model_type).strip() if isinstance(raw_model_type, str) else ""
        if not model_type:
            continue

        raw_model = value.get("model")
        model = str(raw_model).strip() if isinstance(raw_model, str) and raw_model.strip() else None
        normalized[node_name] = {
            "model_type": model_type,
            "model": model,
        }

    return normalized


def _parse_node_llm_configs(raw_json: str | None) -> dict[str, NodeLLMConfig]:
    if not raw_json:
        return {}

    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        return {}

    if not isinstance(payload, dict):
        return {}

    return _normalize_node_llm_configs(payload)


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
) -> dict[str, NodeLLMConfig]:
    result = await session.execute(
        select(ConversationModelConfig).where(ConversationModelConfig.conversation_id == conversation_id)
    )
    config = result.scalar_one_or_none()

    if config is None:
        return {}

    return _parse_node_llm_configs(config.node_models_json)


async def upsert_conversation_model_config(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    node_llm_configs: dict[str, Any] | None,
) -> dict[str, NodeLLMConfig]:
    result = await session.execute(
        select(ConversationModelConfig).where(ConversationModelConfig.conversation_id == conversation_id)
    )
    config = result.scalar_one_or_none()

    normalized_node_llm_configs = _normalize_node_llm_configs(node_llm_configs)
    node_models_json = json.dumps(normalized_node_llm_configs)

    if config is None:
        config = ConversationModelConfig(
            conversation_id=conversation_id,
            default_model=None,
            node_models_json=node_models_json,
        )
        session.add(config)
    else:
        config.default_model = None
        config.node_models_json = node_models_json

    await session.commit()
    return normalized_node_llm_configs
