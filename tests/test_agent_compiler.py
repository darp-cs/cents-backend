from __future__ import annotations

import copy

import pytest

from app.agents.compiler import (
    clear_compiled_agent_graph_cache,
    compile_agent_graph,
    get_compiled_agent_graph,
    invalidate_compiled_agent_graph,
)
from app.agents.template_schema import AgentTemplate


@pytest.fixture(autouse=True)
def reset_compiler_cache() -> None:
    clear_compiled_agent_graph_cache()
    yield
    clear_compiled_agent_graph_cache()


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
                    "output_key": "parsed_input",
                    "fields": [{"name": "parsed_input", "type": "string"}],
                },
                "next": "call_service",
            },
            {
                "id": "call_service",
                "type": "service_call",
                "config": {
                    "service": "ledger",
                    "operation": "record",
                    "parameters": {"amount": 10},
                    "output_key": "result",
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
    assert result["service_results"]["call_service"]["status"] == "stubbed"
    assert result["iteration_count"] >= 3


def test_compile_agent_graph_condition_branches(linear_template_dict: dict) -> None:
    template_payload = copy.deepcopy(linear_template_dict)
    template_payload["entry_node"] = "route"
    template_payload["nodes"] = [
        {
            "id": "route",
            "type": "condition",
            "config": {"expression": "stub"},
            "branches": {
                "yes": "yes_response",
                "no": "no_response",
            },
        },
        {
            "id": "yes_response",
            "type": "terminal_response",
            "config": {"template": "YES"},
        },
        {
            "id": "no_response",
            "type": "terminal_response",
            "config": {"template": "NO"},
        },
    ]

    template = AgentTemplate.model_validate(template_payload)
    compiled_graph = compile_agent_graph(template)

    yes_state = _base_state()
    yes_state["parsed_data"] = {"branch_overrides": {"route": "yes"}}
    yes_result = compiled_graph.invoke(yes_state)

    no_state = _base_state()
    no_state["parsed_data"] = {"route_branch": "no"}
    no_result = compiled_graph.invoke(no_state)

    assert yes_result["final_response"] == "YES"
    assert no_result["final_response"] == "NO"


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
