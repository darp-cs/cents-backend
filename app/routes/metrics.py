from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.users import current_active_user
from app.db.base import get_async_session
from app.db.models import MetricEvent, MetricRequest, User

router = APIRouter()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _date_range(from_: datetime | None, to: datetime | None) -> tuple[datetime, datetime]:
    end = _as_utc(to) if to is not None else datetime.now(timezone.utc)
    start = _as_utc(from_) if from_ is not None else end - timedelta(hours=24)
    if start >= end:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The from date must be earlier than the to date.",
        )
    return start, end


def _require_admin(user: User) -> None:
    if not user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Metrics access requires an administrator.")


def _load_json(value: str | None, default: Any) -> Any:
    try:
        parsed = json.loads(value or "")
    except json.JSONDecodeError:
        return default
    return parsed


def _nearest_rank_p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = max(1, math.ceil(len(ordered) * 0.95))
    return ordered[position - 1]


def _item_key(item: Any, kind: str) -> str:
    if not isinstance(item, dict):
        return str(item)
    keys = ("name", "id") if kind == "tools" else ("source_filename", "id")
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def _rank_items(requests: list[MetricRequest], field: str, kind: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for request in requests:
        for item in _load_json(getattr(request, field), []):
            key = _item_key(item, kind)
            counts[key] = counts.get(key, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))[:10]
    return [{"key": key, "count": count} for key, count in ranked]


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


@router.get("/summary")
async def get_metrics_summary(
    from_: datetime | None = Query(default=None, alias="from"),
    to: datetime | None = Query(default=None),
    user: Annotated[User, Depends(current_active_user)] = None,
    session: AsyncSession = Depends(get_async_session),
):
    _require_admin(user)
    start, end = _date_range(from_, to)

    request_result = await session.execute(
        select(MetricRequest)
        .where(MetricRequest.started_at >= start, MetricRequest.started_at < end)
        .order_by(MetricRequest.started_at.asc(), MetricRequest.id.asc())
    )
    requests = list(request_result.scalars().all())
    request_ids = [request.id for request in requests]

    events: list[MetricEvent] = []
    if request_ids:
        event_result = await session.execute(
            select(MetricEvent)
            .where(MetricEvent.request_id.in_(request_ids))
            .order_by(MetricEvent.node_key.asc(), MetricEvent.id.asc())
        )
        events = list(event_result.scalars().all())

    node_stats: dict[str, dict[str, Any]] = {}
    for event in events:
        stats = node_stats.setdefault(event.node_key, {"count": 0, "latencies": []})
        stats["count"] += 1
        if event.latency_ms is not None:
            stats["latencies"].append(float(event.latency_ms))

    by_node = []
    for node_key in sorted(node_stats):
        stats = node_stats[node_key]
        latencies = stats["latencies"]
        by_node.append(
            {
                "node_key": node_key,
                "request_count": stats["count"],
                "average_latency_ms": sum(latencies) / len(latencies) if latencies else None,
                "p95_latency_ms": _nearest_rank_p95(latencies),
            }
        )

    pass_count = sum(request.judge_verdict == "pass" for request in requests)
    fail_count = sum(request.judge_verdict == "fail" for request in requests)
    judged_count = pass_count + fail_count

    return {
        "from": start.isoformat(),
        "to": end.isoformat(),
        "total_requests": len(requests),
        "total_tokens": sum(request.total_tokens or 0 for request in requests),
        "judge": {
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_rate": pass_count / judged_count if judged_count else None,
        },
        "nodes": by_node,
        "top_retrieved_tools": _rank_items(requests, "retrieved_tools_json", "tools"),
        "top_retrieved_documents": _rank_items(requests, "retrieved_documents_json", "documents"),
    }


@router.get("/events")
async def get_metrics_events(
    from_: datetime | None = Query(default=None, alias="from"),
    to: datetime | None = Query(default=None),
    node_key: str | None = Query(default=None, min_length=1, max_length=100),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: Annotated[User, Depends(current_active_user)] = None,
    session: AsyncSession = Depends(get_async_session),
):
    _require_admin(user)
    start, end = _date_range(from_, to)

    filters = [MetricRequest.started_at >= start, MetricRequest.started_at < end]
    if node_key is not None:
        filters.append(MetricEvent.node_key == node_key.strip())

    total_result = await session.execute(
        select(func.count(MetricEvent.id))
        .join(MetricRequest, MetricRequest.id == MetricEvent.request_id)
        .where(*filters)
    )
    total = int(total_result.scalar_one())

    result = await session.execute(
        select(MetricEvent, MetricRequest)
        .join(MetricRequest, MetricRequest.id == MetricEvent.request_id)
        .where(*filters)
        .order_by(MetricRequest.started_at.desc(), MetricEvent.id.desc())
        .offset(offset)
        .limit(limit)
    )

    items = []
    for event, request in result.all():
        items.append(
            {
                "id": str(event.id),
                "request_id": str(request.id),
                "user_id": str(request.user_id),
                "node_key": event.node_key,
                "started_at": _iso(event.started_at),
                "completed_at": _iso(event.completed_at),
                "latency_ms": event.latency_ms,
                "tokens_used": event.tokens_used,
                "status": event.status,
                "metadata": _load_json(event.metadata_json, {}),
                "request": {
                    "started_at": _iso(request.started_at),
                    "completed_at": _iso(request.completed_at),
                    "status": request.status,
                    "judge_verdict": request.judge_verdict,
                },
            }
        )

    return {"from": start.isoformat(), "to": end.isoformat(), "items": items, "total": total, "limit": limit, "offset": offset}
