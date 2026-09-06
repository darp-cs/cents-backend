from __future__ import annotations

import copy
from typing import Any
import uuid

import httpx
import pytest

from app.agents.compiler import (
    clear_compiled_agent_graph_cache,
    clear_tool_executors,
    compile_agent_graph,
    get_compiled_agent_graph,
    invalidate_compiled_agent_graph,
    register_tool_executor,
)
from app.config import settings
from app.agents.template_schema import AgentTemplate


@pytest.fixture(autouse=True)
def reset_compiler_cache() -> None:
    clear_compiled_agent_graph_cache()
    clear_tool_executors()
    yield
    clear_compiled_agent_graph_cache()
    clear_tool_executors()


@pytest.fixture(autouse=True)
def mock_http_service_request(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_http_request(**kwargs) -> httpx.Response:
        return httpx.Response(status_code=200, json={"ok": True, "echo": kwargs.get("json")})

    monkeypatch.setattr("app.agents.compiler.httpx.request", _fake_http_request)


@pytest.fixture
def linear_template_dict() -> dict:
    return {
        "template_version": "1.0",
        "entry_node": "parse",
        "nodes": [
            {
                "id": "parse",
                "type": "structured_parser",
                "config": {
                    "source_key": "input",
                    "strategy": "regex",
                    "regex_patterns": {"parsed_input": r"(hello)"},
                    "fields": [{"name": "parsed_input", "type": "string"}],
                },
                "next": "call_service",
            },
            {
                "id": "call_service",
                "type": "service_call",
                "config": {
                    "mode": "http",
                    "url": "http://localhost/mock",
                    "method": "POST",
                    "body_template": {"amount": 10},
                },
                "next": "respond",
            },
            {
                "id": "respond",
                "type": "terminal_response",
                "config": {"template": "ok"},
            },
        ],
    }


def _base_state() -> dict:
    return {
        "input": "hello",
        "parsed_data": {},
        "messages": [{"role": "user", "content": "hello"}],
        "iteration_count": 0,
        "service_results": {},
        "interrupt_payload": None,
        "final_response": "",
        "node_llm_configs": {},
    }


def test_compile_agent_graph_executes_linear_flow(linear_template_dict: dict) -> None:
    template = AgentTemplate.model_validate(linear_template_dict)
    compiled_graph = compile_agent_graph(template)

    result = compiled_graph.invoke(_base_state())

    assert result["final_response"] == "ok"
    assert result["parsed_data"]["parsed_input"] == "hello"
    assert result["service_results"]["call_service"]["mode"] == "http"
    assert result["service_results"]["call_service"]["status_code"] == 200
    assert result["iteration_count"] >= 3


def test_structured_parser_merges_new_fields_without_overwriting_existing_data(
    linear_template_dict: dict,
) -> None:
    template_payload = copy.deepcopy(linear_template_dict)
    template_payload["nodes"][0]["config"]["fields"] = [
        {"name": "amount", "type": "number"},
        {"name": "category", "type": "enum", "enum_values": ["food", "rent"]},
    ]
    template_payload["nodes"][0]["config"]["regex_patterns"] = {
        "amount": r"amount\s*[:=]\s*(-?\d+(?:\.\d+)?)",
    }

    template = AgentTemplate.model_validate(template_payload)
    compiled_graph = compile_agent_graph(template)

    state = _base_state()
    state["input"] = "amount: 42 and category food"
    state["parsed_data"] = {"existing": "keep_me"}

    result = compiled_graph.invoke(state)

    assert result["parsed_data"]["existing"] == "keep_me"
    assert result["parsed_data"]["amount"] == 42
    assert result["parsed_data"]["category"] == "food"


def test_structured_parser_llm_strategy_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    template_payload = {
        "template_version": "1.0",
        "entry_node": "parse",
        "nodes": [
            {
                "id": "parse",
                "type": "structured_parser",
                "config": {
                    "source_key": "input",
                    "strategy": "llm",
                    "fields": [
                        {"name": "amount", "type": "number"},
                        {"name": "category", "type": "string"},
                    ],
                },
                "next": "respond",
            },
            {
                "id": "respond",
                "type": "terminal_response",
                "config": {"template": "done"},
            },
        ],
    }
    template = AgentTemplate.model_validate(template_payload)

    captured_payload: dict[str, Any] = {}

    async def _fake_generate_text(payload: dict[str, Any]) -> dict[str, Any]:
        captured_payload.update(payload)
        return {"text": '{"amount": 73, "category": "groceries"}'}

    monkeypatch.setattr("app.agents.compiler.generate_text", _fake_generate_text)

    compiled_graph = compile_agent_graph(template)
    result = compiled_graph.invoke(_base_state())

    assert result["parsed_data"]["amount"] == 73
    assert result["parsed_data"]["category"] == "groceries"
    assert "return JSON only" in captured_payload["system_prompt"]


def test_structured_parser_on_failure_routes_to_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    template_payload = {
        "template_version": "1.0",
        "entry_node": "parse",
        "nodes": [
            {
                "id": "parse",
                "type": "structured_parser",
                "config": {
                    "source_key": "input",
                    "strategy": "llm",
                    "fields": [{"name": "amount", "type": "number"}],
                },
                "next": "success_response",
                "on_failure": "failure_response",
            },
            {
                "id": "success_response",
                "type": "terminal_response",
                "config": {"template": "success"},
            },
            {
                "id": "failure_response",
                "type": "terminal_response",
                "config": {"template": "fallback"},
            },
        ],
    }
    template = AgentTemplate.model_validate(template_payload)

    async def _fake_generate_text(_: dict[str, Any]) -> dict[str, Any]:
        return {"text": "not json"}

    monkeypatch.setattr("app.agents.compiler.generate_text", _fake_generate_text)

    compiled_graph = compile_agent_graph(template)
    result = compiled_graph.invoke(_base_state())

    assert result["final_response"] == "fallback"
    assert "parse" in result["parsed_data"]["__parser_errors__"]


def test_structured_parser_without_on_failure_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_payload = {
        "template_version": "1.0",
        "entry_node": "parse",
        "nodes": [
            {
                "id": "parse",
                "type": "structured_parser",
                "config": {
                    "source_key": "input",
                    "strategy": "llm",
                    "fields": [{"name": "amount", "type": "number"}],
                },
                "next": "success_response",
            },
            {
                "id": "success_response",
                "type": "terminal_response",
                "config": {"template": "success"},
            },
        ],
    }
    template = AgentTemplate.model_validate(template_payload)

    async def _fake_generate_text(_: dict[str, Any]) -> dict[str, Any]:
        return {"text": "{\"amount\": \"NaN\"}"}

    monkeypatch.setattr("app.agents.compiler.generate_text", _fake_generate_text)

    compiled_graph = compile_agent_graph(template)

    with pytest.raises(RuntimeError) as exc:
        compiled_graph.invoke(_base_state())

    assert "Structured parser node 'parse' failed" in str(exc.value)


def test_service_call_non_2xx_routes_to_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    template_payload = {
        "template_version": "1.0",
        "entry_node": "call_service",
        "nodes": [
            {
                "id": "call_service",
                "type": "service_call",
                "config": {
                    "mode": "http",
                    "url": "http://localhost/service",
                    "method": "GET",
                },
                "next": "success_response",
                "on_failure": "failure_response",
            },
            {
                "id": "success_response",
                "type": "terminal_response",
                "config": {"template": "success"},
            },
            {
                "id": "failure_response",
                "type": "terminal_response",
                "config": {"template": "failure"},
            },
        ],
    }
    template = AgentTemplate.model_validate(template_payload)

    def _fake_http_request(**_: Any) -> httpx.Response:
        return httpx.Response(status_code=503, text="service unavailable")

    monkeypatch.setattr("app.agents.compiler.httpx.request", _fake_http_request)

    compiled_graph = compile_agent_graph(template)
    result = compiled_graph.invoke(_base_state())

    assert result["final_response"] == "failure"
    assert result["service_results"]["call_service"]["__status__"] == "failure"


def test_service_call_disallowed_host_routes_to_on_failure() -> None:
    template_payload = {
        "template_version": "1.0",
        "entry_node": "call_service",
        "nodes": [
            {
                "id": "call_service",
                "type": "service_call",
                "config": {
                    "mode": "http",
                    "url": "http://internal-only.example/service",
                    "method": "GET",
                },
                "next": "success_response",
                "on_failure": "failure_response",
            },
            {
                "id": "success_response",
                "type": "terminal_response",
                "config": {"template": "success"},
            },
            {
                "id": "failure_response",
                "type": "terminal_response",
                "config": {"template": "failure"},
            },
        ],
    }
    template = AgentTemplate.model_validate(template_payload)

    compiled_graph = compile_agent_graph(template)
    result = compiled_graph.invoke(_base_state())

    assert result["final_response"] == "failure"
    assert "not in service_call_allowed_hosts" in result["service_results"]["call_service"]["error"]


def test_service_call_allow_unsafe_requires_server_opt_in() -> None:
    template_payload = {
        "template_version": "1.0",
        "entry_node": "call_service",
        "nodes": [
            {
                "id": "call_service",
                "type": "service_call",
                "config": {
                    "mode": "http",
                    "url": "http://external.example/service",
                    "method": "GET",
                    "allow_unsafe_destination": True,
                },
                "next": "success_response",
                "on_failure": "failure_response",
            },
            {
                "id": "success_response",
                "type": "terminal_response",
                "config": {"template": "success"},
            },
            {
                "id": "failure_response",
                "type": "terminal_response",
                "config": {"template": "failure"},
            },
        ],
    }
    template = AgentTemplate.model_validate(template_payload)

    previous_flag = settings.service_call_allow_unsafe_destinations
    settings.service_call_allow_unsafe_destinations = False
    try:
        compiled_graph = compile_agent_graph(template)
        result = compiled_graph.invoke(_base_state())
    finally:
        settings.service_call_allow_unsafe_destinations = previous_flag

    assert result["final_response"] == "failure"


def test_service_call_timeout_without_on_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    template_payload = {
        "template_version": "1.0",
        "entry_node": "call_service",
        "nodes": [
            {
                "id": "call_service",
                "type": "service_call",
                "config": {
                    "mode": "http",
                    "url": "http://localhost/service",
                    "method": "GET",
                    "timeout_seconds": 1,
                },
                "next": "success_response",
            },
            {
                "id": "success_response",
                "type": "terminal_response",
                "config": {"template": "success"},
            },
        ],
    }
    template = AgentTemplate.model_validate(template_payload)

    def _fake_http_request(**_: Any) -> httpx.Response:
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr("app.agents.compiler.httpx.request", _fake_http_request)

    compiled_graph = compile_agent_graph(template)
    with pytest.raises(RuntimeError) as exc:
        compiled_graph.invoke(_base_state())

    assert "Service call node 'call_service' failed" in str(exc.value)


def test_service_call_sensitive_header_requires_secret_reference() -> None:
    template_payload = {
        "template_version": "1.0",
        "entry_node": "call_service",
        "nodes": [
            {
                "id": "call_service",
                "type": "service_call",
                "config": {
                    "mode": "http",
                    "url": "http://localhost/service",
                    "method": "GET",
                    "headers_template": {"Authorization": "Bearer plain-text-key"},
                },
                "next": "success_response",
            },
            {
                "id": "success_response",
                "type": "terminal_response",
                "config": {"template": "success"},
            },
        ],
    }
    template = AgentTemplate.model_validate(template_payload)

    compiled_graph = compile_agent_graph(template)
    with pytest.raises(RuntimeError) as exc:
        compiled_graph.invoke(_base_state())

    assert "Sensitive header 'Authorization'" in str(exc.value)


def test_service_call_sensitive_body_field_requires_secret_reference() -> None:
    template_payload = {
        "template_version": "1.0",
        "entry_node": "call_service",
        "nodes": [
            {
                "id": "call_service",
                "type": "service_call",
                "config": {
                    "mode": "http",
                    "url": "http://localhost/service",
                    "method": "POST",
                    "body_template": {"api_key": "plain-text-key"},
                },
                "next": "success_response",
            },
            {
                "id": "success_response",
                "type": "terminal_response",
                "config": {"template": "success"},
            },
        ],
    }
    template = AgentTemplate.model_validate(template_payload)

    compiled_graph = compile_agent_graph(template)
    with pytest.raises(RuntimeError) as exc:
        compiled_graph.invoke(_base_state())

    assert "Sensitive body field 'api_key'" in str(exc.value)


def test_service_call_secret_placeholder_uses_server_side_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_payload = {
        "template_version": "1.0",
        "entry_node": "call_service",
        "nodes": [
            {
                "id": "call_service",
                "type": "service_call",
                "config": {
                    "mode": "http",
                    "url": "http://localhost/service",
                    "method": "GET",
                    "headers_template": {
                        "Authorization": "Bearer {{ secret.billing_api_key }}",
                    },
                },
                "next": "success_response",
            },
            {
                "id": "success_response",
                "type": "terminal_response",
                "config": {"template": "success"},
            },
        ],
    }
    template = AgentTemplate.model_validate(template_payload)
    captured_headers: dict[str, str] = {}

    def _fake_http_request(**kwargs: Any) -> httpx.Response:
        nonlocal captured_headers
        captured_headers = dict(kwargs.get("headers", {}))
        return httpx.Response(status_code=200, json={"ok": True})

    monkeypatch.setattr("app.agents.compiler.httpx.request", _fake_http_request)

    previous_secrets = dict(settings.service_call_secrets)
    settings.service_call_secrets = {"billing_api_key": "server-secret"}
    try:
        compiled_graph = compile_agent_graph(template)
        compiled_graph.invoke(_base_state())
    finally:
        settings.service_call_secrets = previous_secrets

    assert captured_headers["Authorization"] == "Bearer server-secret"


def test_service_call_tool_reference_uses_registered_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    template_payload = {
        "template_version": "1.0",
        "entry_node": "call_service",
        "nodes": [
            {
                "id": "call_service",
                "type": "service_call",
                "config": {
                    "mode": "tool",
                    "tool_name": "charge-card",
                    "tool_input_template": {"amount": "{{ parsed_data.amount }}"},
                },
                "next": "success_response",
            },
            {
                "id": "success_response",
                "type": "terminal_response",
                "config": {"template": "success"},
            },
        ],
    }
    template = AgentTemplate.model_validate(template_payload)

    async def _fake_fetch_tool_definition(tool_name: str | None, tool_id: str | None):
        del tool_id
        return type("ToolRow", (), {"id": uuid.uuid4(), "name": tool_name or "charge-card"})()

    monkeypatch.setattr("app.agents.compiler._fetch_tool_definition", _fake_fetch_tool_definition)

    register_tool_executor(
        tool_name="charge-card",
        executor=lambda payload, _: {"charged": True, "amount": payload["amount"]},
    )

    state = _base_state()
    state["parsed_data"] = {"amount": 120}

    compiled_graph = compile_agent_graph(template)
    result = compiled_graph.invoke(state)

    assert result["service_results"]["call_service"]["mode"] == "tool"
    assert result["service_results"]["call_service"]["result"]["charged"] is True
    assert result["service_results"]["call_service"]["result"]["amount"] == 120


def test_condition_node_routes_boolean_true_false() -> None:
    template_payload = {
        "template_version": "1.0",
        "entry_node": "route",
        "nodes": [
            {
                "id": "route",
                "type": "condition",
                "config": {"expression": "parsed_data.amount > 100"},
                "branches": {
                    "true": "high_response",
                    "false": "low_response",
                    "default": "fallback_response",
                },
            },
            {
                "id": "high_response",
                "type": "terminal_response",
                "config": {"template": "HIGH"},
            },
            {
                "id": "low_response",
                "type": "terminal_response",
                "config": {"template": "LOW"},
            },
            {
                "id": "fallback_response",
                "type": "terminal_response",
                "config": {"template": "FALLBACK"},
            },
        ],
    }

    template = AgentTemplate.model_validate(template_payload)
    compiled_graph = compile_agent_graph(template)

    high_state = _base_state()
    high_state["parsed_data"] = {"amount": 250}
    high_result = compiled_graph.invoke(high_state)

    low_state = _base_state()
    low_state["parsed_data"] = {"amount": 40}
    low_result = compiled_graph.invoke(low_state)

    assert high_result["final_response"] == "HIGH"
    assert low_result["final_response"] == "LOW"


def test_condition_node_routes_multi_branch_string_match() -> None:
    template_payload = {
        "template_version": "1.0",
        "entry_node": "route",
        "nodes": [
            {
                "id": "route",
                "type": "condition",
                "config": {"expression": "parsed_data.category"},
                "branches": {
                    "food": "food_response",
                    "rent": "rent_response",
                    "default": "other_response",
                },
            },
            {
                "id": "food_response",
                "type": "terminal_response",
                "config": {"template": "FOOD"},
            },
            {
                "id": "rent_response",
                "type": "terminal_response",
                "config": {"template": "RENT"},
            },
            {
                "id": "other_response",
                "type": "terminal_response",
                "config": {"template": "OTHER"},
            },
        ],
    }

    template = AgentTemplate.model_validate(template_payload)
    compiled_graph = compile_agent_graph(template)

    state = _base_state()
    state["parsed_data"] = {"category": "rent"}
    result = compiled_graph.invoke(state)

    assert result["final_response"] == "RENT"


def test_condition_missing_field_routes_to_default_and_records_error() -> None:
    template_payload = {
        "template_version": "1.0",
        "entry_node": "route",
        "nodes": [
            {
                "id": "route",
                "type": "condition",
                "config": {"expression": "parsed_data.amount > 100"},
                "branches": {
                    "true": "high_response",
                    "false": "low_response",
                    "default": "fallback_response",
                },
            },
            {
                "id": "high_response",
                "type": "terminal_response",
                "config": {"template": "HIGH"},
            },
            {
                "id": "low_response",
                "type": "terminal_response",
                "config": {"template": "LOW"},
            },
            {
                "id": "fallback_response",
                "type": "terminal_response",
                "config": {"template": "FALLBACK"},
            },
        ],
    }

    template = AgentTemplate.model_validate(template_payload)
    compiled_graph = compile_agent_graph(template)

    state = _base_state()
    state["parsed_data"] = {}
    result = compiled_graph.invoke(state)

    assert result["final_response"] == "FALLBACK"
    assert "route" in result["parsed_data"]["__condition_errors__"]


def test_compiled_graphs_are_isolated_per_template(linear_template_dict: dict) -> None:
    template_a = AgentTemplate.model_validate(linear_template_dict)

    template_b_payload = copy.deepcopy(linear_template_dict)
    template_b_payload["nodes"][2]["config"]["template"] = "different"
    template_b = AgentTemplate.model_validate(template_b_payload)

    graph_a = compile_agent_graph(template_a)
    graph_b = compile_agent_graph(template_b)

    result_a = graph_a.invoke(_base_state())
    result_b = graph_b.invoke(_base_state())

    assert graph_a is not graph_b
    assert result_a["final_response"] == "ok"
    assert result_b["final_response"] == "different"


def test_compiled_graph_cache_is_keyed_by_name_and_version(linear_template_dict: dict) -> None:
    template = AgentTemplate.model_validate(linear_template_dict)

    graph_v1 = get_compiled_agent_graph("budget-agent", 1, template)
    graph_v1_again = get_compiled_agent_graph("budget-agent", 1, template)
    graph_v2 = get_compiled_agent_graph("budget-agent", 2, template)
    graph_other = get_compiled_agent_graph("expense-agent", 1, template)

    assert graph_v1 is graph_v1_again
    assert graph_v2 is not graph_v1
    assert graph_other is not graph_v1


def test_compiled_graph_cache_can_invalidate_by_key(linear_template_dict: dict) -> None:
    template = AgentTemplate.model_validate(linear_template_dict)

    graph_v1 = get_compiled_agent_graph("budget-agent", 1, template)
    invalidate_compiled_agent_graph("budget-agent", version=1)
    rebuilt_v1 = get_compiled_agent_graph("budget-agent", 1, template)

    assert rebuilt_v1 is not graph_v1


def test_compiled_graph_cache_can_invalidate_all_versions_for_name(linear_template_dict: dict) -> None:
    template = AgentTemplate.model_validate(linear_template_dict)

    graph_v1 = get_compiled_agent_graph("budget-agent", 1, template)
    graph_v2 = get_compiled_agent_graph("budget-agent", 2, template)

    invalidate_compiled_agent_graph("budget-agent")

    rebuilt_v1 = get_compiled_agent_graph("budget-agent", 1, template)
    rebuilt_v2 = get_compiled_agent_graph("budget-agent", 2, template)

    assert rebuilt_v1 is not graph_v1
    assert rebuilt_v2 is not graph_v2
