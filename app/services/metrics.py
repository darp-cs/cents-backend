from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MetricEvent, MetricRequest


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


async def create_metric_request(session: AsyncSession, user_id, started_at: datetime) -> MetricRequest:
    record = MetricRequest(user_id=user_id, started_at=started_at)
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return record


async def finalize_metric_request(
    session: AsyncSession,
    record: MetricRequest,
    *,
    completed_at: datetime,
    latency_ms: float,
    status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    result = result or {}
    record.completed_at = completed_at
    record.latency_ms = latency_ms
    record.status = status
    record.error = error[:1000] if error else None
    token_usage = result.get("token_usage", {})
    record.total_tokens = sum(
        int(value) for value in token_usage.values() if isinstance(value, (int, float)) and value >= 0
    ) if isinstance(token_usage, dict) else 0
    judge_verdict = result.get("judge_verdict")
    record.judge_verdict = str(judge_verdict.get("verdict")).strip().lower() if isinstance(judge_verdict, dict) else None
    record.retrieved_tools_json = json.dumps(result.get("retrieved_tools", []), ensure_ascii=True, separators=(",", ":"))
    record.retrieved_documents_json = json.dumps(result.get("retrieved_docs", []), ensure_ascii=True, separators=(",", ":"))

    for metric in result.get("node_metrics", []):
        if not isinstance(metric, dict) or not metric.get("node_key"):
            continue
        node_latency = metric.get("latency_ms")
        event = MetricEvent(
            request_id=record.id,
            node_key=str(metric["node_key"]),
            started_at=record.started_at,
            completed_at=completed_at,
            latency_ms=float(node_latency) if isinstance(node_latency, (int, float)) else None,
            tokens_used=int(metric.get("tokens_used", 0) or 0),
            status=str(metric.get("status", status)),
            metadata_json=json.dumps(metric, ensure_ascii=True, separators=(",", ":")),
        )
        session.add(event)

    await session.commit()
