import logging
import uuid
from typing import Any

from app.db.base import AsyncSessionLocal
from app.db.models import MetricEvent

logger = logging.getLogger(__name__)


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


async def record_metric_event(
    *,
    conversation_id: str | None,
    node_key: str,
    model: str | None = None,
    latency_ms: Any = None,
    usage: Any = None,
    retrieved_count: int | None = None,
    judge_verdict: str | None = None,
) -> None:
    try:
        conversation_uuid = uuid.UUID(conversation_id) if conversation_id else None
        usage_data = usage if isinstance(usage, dict) else {}
        prompt_tokens = _as_int(usage_data.get("prompt_tokens"))
        completion_tokens = _as_int(usage_data.get("completion_tokens"))
        total_tokens = _as_int(usage_data.get("total_tokens"))

        async with AsyncSessionLocal() as session:
            session.add(
                MetricEvent(
                    conversation_id=conversation_uuid,
                    node_key=node_key,
                    model=model,
                    latency_ms=_as_float(latency_ms),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    retrieved_count=retrieved_count,
                    judge_verdict=judge_verdict,
                )
            )
            await session.commit()
    except Exception:
        logger.exception("Failed to record metric event for node %s", node_key)