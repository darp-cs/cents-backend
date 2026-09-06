from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from sqlalchemy import delete

from app.db.base import AsyncSessionLocal, init_db
from app.db.models import MetricEvent, MetricRequest
from app.routes.metrics import get_metrics_events, get_metrics_summary


def _run(coro):
    return asyncio.run(coro)


def test_metrics_summary_aggregates_request_latency_tokens_and_retrieved_items() -> None:
    async def _test() -> None:
        await init_db()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        admin = SimpleNamespace(id=uuid.uuid4(), is_superuser=True)

        async with AsyncSessionLocal() as session:
            await session.execute(delete(MetricEvent))
            await session.execute(delete(MetricRequest))
            first = MetricRequest(
                user_id=admin.id,
                started_at=now - timedelta(hours=1),
                completed_at=now - timedelta(hours=1) + timedelta(milliseconds=20),
                latency_ms=20,
                total_tokens=12,
                judge_verdict="pass",
                status="completed",
                retrieved_tools_json=json.dumps([{"name": "budget"}]),
                retrieved_documents_json=json.dumps([{"source_filename": "plan.md"}]),
            )
            second = MetricRequest(
                user_id=admin.id,
                started_at=now - timedelta(minutes=30),
                completed_at=now - timedelta(minutes=30) + timedelta(milliseconds=100),
                latency_ms=100,
                total_tokens=8,
                judge_verdict="fail",
                status="completed",
                retrieved_tools_json=json.dumps([{"name": "budget"}, {"name": "calendar"}]),
                retrieved_documents_json=json.dumps([{"source_filename": "plan.md"}]),
            )
            session.add_all([first, second])
            await session.flush()
            session.add_all(
                [
                    MetricEvent(request_id=first.id, node_key="generation", started_at=first.started_at, latency_ms=20),
                    MetricEvent(request_id=second.id, node_key="generation", started_at=second.started_at, latency_ms=100),
                ]
            )
            await session.commit()

            payload = await get_metrics_summary(
                from_=now - timedelta(hours=2), to=now, user=admin, session=session
            )

            assert payload["total_requests"] == 2
            assert payload["total_tokens"] == 20
            assert payload["judge"] == {"pass_count": 1, "fail_count": 1, "pass_rate": 0.5}
            assert payload["nodes"][0]["p95_latency_ms"] == 100
            assert payload["top_retrieved_tools"][0] == {"key": "budget", "count": 2}
            assert payload["top_retrieved_documents"][0] == {"key": "plan.md", "count": 2}

    _run(_test())


def test_metrics_events_filters_and_paginates() -> None:
    async def _test() -> None:
        await init_db()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        admin = SimpleNamespace(id=uuid.uuid4(), is_superuser=True)

        async with AsyncSessionLocal() as session:
            await session.execute(delete(MetricEvent))
            await session.execute(delete(MetricRequest))
            request = MetricRequest(user_id=admin.id, started_at=now - timedelta(minutes=5), status="completed")
            session.add(request)
            await session.flush()
            session.add_all(
                [
                    MetricEvent(request_id=request.id, node_key="generation", started_at=request.started_at),
                    MetricEvent(request_id=request.id, node_key="judge", started_at=request.started_at),
                ]
            )
            await session.commit()

            payload = await get_metrics_events(
                from_=now - timedelta(hours=1), to=now, node_key="judge", limit=1, offset=0, user=admin, session=session
            )

            assert payload["total"] == 1
            assert len(payload["items"]) == 1
            assert payload["items"][0]["node_key"] == "judge"

    _run(_test())
