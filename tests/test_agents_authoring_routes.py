from __future__ import annotations

import asyncio
import copy
import json
import uuid
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from app.auth.users import current_active_user
from app.db.base import AsyncSessionLocal, init_db
from app.db.models import AgentTemplate
from app.routes.agents import (
    AgentAuthoringValidateRequest,
    get_authoring_schema,
    router,
    validate_authoring_template,
)


def _run(coro):
    return asyncio.run(coro)


def _valid_template() -> dict[str, Any]:
    return {
        "template_version": "1.0",
        "entry_node": "parse_request",
        "nodes": [
            {
                "id": "parse_request",
                "type": "structured_parser",
                "config": {
                    "fields": [{"name": "amount", "type": "number"}],
                    "regex_patterns": {"amount": r"amount\\s*[:=]\\s*(-?\\d+(?:\\.\\d+)?)"},
                },
                "next": "respond",
            },
            {
                "id": "respond",
                "type": "terminal_response",
                "config": {"template": "Done {{ parsed_data.amount }}"},
            },
        ],
    }


def _fake_user() -> Any:
    return SimpleNamespace(id=uuid.uuid4(), is_active=True)


def test_authoring_schema_lists_all_node_types_and_metadata() -> None:
    async def _test() -> None:
        schema = await get_authoring_schema(user=_fake_user())

        assert schema.catalog_version == "1.0.0"
        assert schema.node_id_pattern
        assert schema.template_version_pattern

        node_types = {node.type: node for node in schema.node_types}
        assert set(node_types.keys()) == {
            "structured_parser",
            "condition",
            "service_call",
            "user_interrupt",
            "llm_step",
            "terminal_response",
        }

        for node in schema.node_types:
            assert node.label
            assert node.description
            assert node.config_fields
            for field in node.config_fields:
                assert field.name
                assert field.type
                assert isinstance(field.required, bool)

        condition_transitions = {item.kind for item in node_types["condition"].transitions}
        assert condition_transitions == {"branches"}

        llm_transitions = {item.kind for item in node_types["llm_step"].transitions}
        assert llm_transitions == {"next", "on_failure"}

    _run(_test())


def test_authoring_validate_returns_normalized_template_for_parseable_input() -> None:
    async def _test() -> None:
        template = _valid_template()
        payload = AgentAuthoringValidateRequest(raw_template=template)

        response = await validate_authoring_template(payload=payload, user=_fake_user())

        assert response.is_valid is True
        assert response.errors == []
        assert response.normalized_template is not None
        assert response.normalized_template["guardrails"]["max_iterations"] == 3
        assert response.normalized_template["nodes"][1]["config"]["status"] == "success"

    _run(_test())


def test_authoring_validate_maps_graph_errors_with_node_id_and_path() -> None:
    async def _test() -> None:
        template = _valid_template()
        template["nodes"][0]["next"] = "missing_node"

        response = await validate_authoring_template(
            payload=AgentAuthoringValidateRequest(raw_template=template),
            user=_fake_user(),
        )

        assert response.is_valid is False
        assert response.normalized_template is not None
        assert response.errors

        first_error = response.errors[0]
        assert first_error.node_id == "parse_request"
        assert first_error.path == "nodes.parse_request.next"
        assert "unknown node id" in first_error.message

    _run(_test())


def test_authoring_validate_maps_schema_errors_back_to_node_ids() -> None:
    async def _test() -> None:
        template = _valid_template()
        template["nodes"][1] = {
            "id": "respond",
            "type": "terminal_response",
            "config": {},
        }

        response = await validate_authoring_template(
            payload=AgentAuthoringValidateRequest(raw_template=template),
            user=_fake_user(),
        )

        assert response.is_valid is False
        assert response.normalized_template is None
        assert response.errors

        assert any(
            item.node_id == "respond"
            and item.path.startswith("nodes.1")
            and "Field required" in item.message
            for item in response.errors
        )

    _run(_test())


def test_authoring_validate_rejects_non_object_template_without_server_error() -> None:
    async def _test() -> None:
        response = await validate_authoring_template(
            payload=AgentAuthoringValidateRequest(raw_template=["not", "an", "object"]),
            user=_fake_user(),
        )

        assert response.is_valid is False
        assert response.normalized_template is None
        assert [item.model_dump() for item in response.errors] == [
            {
                "path": "template",
                "node_id": None,
                "message": "Template must be a JSON object.",
            }
        ]

    _run(_test())


def test_authoring_validate_does_not_write_or_compile(monkeypatch) -> None:
    async def _test() -> None:
        await init_db()

        async with AsyncSessionLocal() as session:
            await session.execute(delete(AgentTemplate))
            await session.commit()

            seed = _valid_template()
            session.add(
                AgentTemplate(
                    name="seed-agent",
                    version=1,
                    raw_template=json.dumps(seed, ensure_ascii=True, separators=(",", ":")),
                    is_valid=True,
                    validation_errors=None,
                    enabled=True,
                )
            )
            await session.commit()

            before_result = await session.execute(select(func.count()).select_from(AgentTemplate))
            before_count = int(before_result.scalar_one())

            def _fail_compile(**_):
                raise AssertionError("Dry-run validation should not compile templates")

            monkeypatch.setattr("app.routes.agents.get_compiled_agent_graph", _fail_compile)

            template = copy.deepcopy(seed)
            template["nodes"][0]["next"] = "unknown"
            response = await validate_authoring_template(
                payload=AgentAuthoringValidateRequest(raw_template=template),
                user=_fake_user(),
            )

            after_result = await session.execute(select(func.count()).select_from(AgentTemplate))
            after_count = int(after_result.scalar_one())

            assert response.is_valid is False
            assert before_count == after_count == 1

            persisted = await session.execute(select(AgentTemplate).where(AgentTemplate.name == "seed-agent"))
            row = persisted.scalar_one()
            assert row.version == 1
            assert row.enabled is True

    _run(_test())


def test_authoring_routes_require_auth_like_existing_agents_routes() -> None:
    app = FastAPI()
    app.include_router(router, prefix="/agents")

    with TestClient(app) as client:
        schema_response = client.get("/agents/authoring/schema")
        list_response = client.get("/agents")

    assert schema_response.status_code == 401
    assert list_response.status_code == 401


def test_authoring_routes_accept_active_user_dependency_override() -> None:
    app = FastAPI()
    app.include_router(router, prefix="/agents")
    app.dependency_overrides[current_active_user] = lambda: SimpleNamespace(id=uuid.uuid4(), is_active=True)

    with TestClient(app) as client:
        schema_response = client.get("/agents/authoring/schema")
        validate_response = client.post(
            "/agents/authoring/validate",
            json={"raw_template": _valid_template()},
        )

    assert schema_response.status_code == 200
    assert validate_response.status_code == 200
    assert validate_response.json()["is_valid"] is True
