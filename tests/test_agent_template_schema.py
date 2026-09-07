from __future__ import annotations

import copy

import pytest

from app.agents.template_schema import (
    AgentTemplate,
    ConditionNode,
    TerminalResponseNode,
    validate_template,
)


@pytest.fixture
def valid_template() -> dict:
    return {
        "template_version": "1.0",
        "entry_node": "parse_request",
        "guardrails": {
            "max_iterations": 5,
            "banned_topics_override": ["medical advice"],
            "judge_enabled_override": True,
        },
        "nodes": [
            {
                "id": "parse_request",
                "type": "structured_parser",
                "config": {
                    "source_key": "last_message",
                    "strategy": "regex",
                    "regex_patterns": {"amount": r"amount\\s*[:=]\\s*(-?\\d+(?:\\.\\d+)?)"},
                    "fields": [
                        {"name": "amount", "type": "number"},
                        {"name": "category", "type": "string", "required": False},
                    ],
                },
                "next": "check_amount",
            },
            {
                "id": "check_amount",
                "type": "condition",
                "config": {
                    "expression": "parsed_request.amount > 1000",
                    "input_keys": ["parsed_request"],
                },
                "branches": {
                    "true": "confirm_with_user",
                    "false": "call_ledger",
                    "default": "call_ledger",
                },
            },
            {
                "id": "confirm_with_user",
                "type": "user_interrupt",
                "config": {
                    "prompt": "This is a large transaction. Continue?",
                    "output_key": "user_confirmation",
                    "expected_type": "confirmation",
                },
                "next": "call_ledger",
            },
            {
                "id": "call_ledger",
                "type": "service_call",
                "config": {
                    "mode": "http",
                    "url": "http://localhost/ledger",
                    "method": "POST",
                    "body_template": {"amount": "{{ parsed_data.amount }}"},
                },
                "next": "summarize",
            },
            {
                "id": "summarize",
                "type": "llm_step",
                "config": {
                    "model_type": "text-generation",
                    "system_prompt": "Summarize this result: {{ service_results.call_ledger.body }}",
                    "temperature": 0.2,
                    "max_tokens": 128,
                    "output_key": "summary",
                },
                "next": "respond",
            },
            {
                "id": "respond",
                "type": "terminal_response",
                "config": {"template": "{{ summary }}"},
            },
        ],
    }


def test_valid_template_parses(valid_template: dict) -> None:
    result = validate_template(valid_template)

    assert result.errors == []
    assert result.is_valid
    template = result.template
    assert isinstance(template, AgentTemplate)
    assert template.entry_node == "parse_request"
    assert template.guardrails.max_iterations == 5
    assert template.guardrails.banned_topics_override == ["medical advice"]
    assert template.guardrails.judge_enabled_override is True
    assert isinstance(template.nodes[1], ConditionNode)
    assert isinstance(template.nodes[-1], TerminalResponseNode)


def test_guardrails_default_when_omitted(valid_template: dict) -> None:
    valid_template.pop("guardrails")

    result = validate_template(valid_template)

    assert result.is_valid
    assert result.template is not None
    assert result.template.guardrails.max_iterations == 3
    assert result.template.guardrails.banned_topics_override is None
    assert result.template.guardrails.judge_enabled_override is None


def test_canvas_layout_is_accepted_when_node_ids_match(valid_template: dict) -> None:
    template = copy.deepcopy(valid_template)
    template["canvas_layout"] = {
        "node_positions": {
            "parse_request": {"x": 220, "y": 70},
            "check_amount": {"x": 220, "y": 250},
            "respond": {"x": 220, "y": 430},
        },
        "viewport": {"x": 0, "y": 0, "zoom": 1},
    }

    result = validate_template(template)

    assert result.is_valid
    assert result.template is not None
    assert result.template.canvas_layout is not None
    assert "parse_request" in result.template.canvas_layout.node_positions


def test_canvas_layout_rejects_unknown_node_ids(valid_template: dict) -> None:
    template = copy.deepcopy(valid_template)
    template["canvas_layout"] = {
        "node_positions": {
            "missing_node": {"x": 120, "y": 80},
        }
    }

    result = validate_template(template)

    assert not result.is_valid
    assert result.errors == [
        "canvas_layout.node_positions['missing_node'] points to unknown node id 'missing_node'."
    ]


def test_canvas_layout_rejects_invalid_node_position_keys(valid_template: dict) -> None:
    template = copy.deepcopy(valid_template)
    template["canvas_layout"] = {
        "node_positions": {
            "1-invalid": {"x": 120, "y": 80},
        }
    }

    result = validate_template(template)

    assert not result.is_valid
    assert any("canvas_layout.node_positions" in error for error in result.errors)


def test_dangling_edge_is_rejected(valid_template: dict) -> None:
    template = copy.deepcopy(valid_template)
    template["nodes"][3]["next"] = "does_not_exist"

    result = validate_template(template)

    assert not result.is_valid
    assert result.template is None
    assert result.errors == [
        "Node 'call_ledger' next points to unknown node id 'does_not_exist'."
    ]


def test_dangling_branch_target_is_rejected(valid_template: dict) -> None:
    template = copy.deepcopy(valid_template)
    template["nodes"][1]["branches"]["false"] = "missing_node"

    result = validate_template(template)

    assert not result.is_valid
    assert result.errors == [
        "Node 'check_amount' branches['false'] points to unknown node id 'missing_node'."
    ]


def test_dangling_on_failure_target_is_rejected(valid_template: dict) -> None:
    template = copy.deepcopy(valid_template)
    template["nodes"][0]["on_failure"] = "missing_fallback"

    result = validate_template(template)

    assert not result.is_valid
    assert result.errors == [
        "Node 'parse_request' on_failure points to unknown node id 'missing_fallback'."
    ]


def test_dangling_service_call_on_failure_target_is_rejected(valid_template: dict) -> None:
    template = copy.deepcopy(valid_template)
    template["nodes"][3]["on_failure"] = "missing_service_fallback"

    result = validate_template(template)

    assert not result.is_valid
    assert result.errors == [
        "Node 'call_ledger' on_failure points to unknown node id 'missing_service_fallback'."
    ]


def test_missing_terminal_node_is_rejected() -> None:
    template = {
        "template_version": "1.0",
        "entry_node": "loop_a",
        "nodes": [
            {
                "id": "loop_a",
                "type": "llm_step",
                "config": {
                    "model_type": "text-generation",
                    "system_prompt": "a",
                    "temperature": 0.2,
                    "max_tokens": 32,
                    "output_key": "a",
                },
                "next": "loop_b",
            },
            {
                "id": "loop_b",
                "type": "llm_step",
                "config": {
                    "model_type": "text-generation",
                    "system_prompt": "b",
                    "temperature": 0.2,
                    "max_tokens": 32,
                    "output_key": "b",
                },
                "next": "loop_a",
            },
        ],
    }

    result = validate_template(template)

    assert not result.is_valid
    assert result.errors == [
        "Template must define at least one 'terminal_response' node."
    ]


def test_unreachable_node_is_rejected(valid_template: dict) -> None:
    template = copy.deepcopy(valid_template)
    template["nodes"].append(
        {
            "id": "orphan",
            "type": "terminal_response",
            "config": {"template": "unused", "status": "failure"},
        }
    )

    result = validate_template(template)

    assert not result.is_valid
    assert result.errors == [
        "Node 'orphan' is unreachable from entry_node 'parse_request'."
    ]


def test_node_that_cannot_reach_terminal_is_rejected() -> None:
    template = {
        "template_version": "1.0",
        "entry_node": "start",
        "nodes": [
            {
                "id": "start",
                "type": "condition",
                "config": {"expression": "state.ok"},
                "branches": {"ok": "respond", "stuck": "dead_end_a", "default": "dead_end_a"},
            },
            {
                "id": "dead_end_a",
                "type": "llm_step",
                "config": {
                    "model_type": "text-generation",
                    "system_prompt": "a",
                    "temperature": 0.2,
                    "max_tokens": 32,
                    "output_key": "a",
                },
                "next": "dead_end_b",
            },
            {
                "id": "dead_end_b",
                "type": "llm_step",
                "config": {
                    "model_type": "text-generation",
                    "system_prompt": "b",
                    "temperature": 0.2,
                    "max_tokens": 32,
                    "output_key": "b",
                },
                "next": "dead_end_a",
            },
            {
                "id": "respond",
                "type": "terminal_response",
                "config": {"template": "done"},
            },
        ],
    }

    result = validate_template(template)

    assert not result.is_valid
    assert result.errors == [
        "Node 'dead_end_a' cannot reach a 'terminal_response' node.",
        "Node 'dead_end_b' cannot reach a 'terminal_response' node.",
    ]


def test_missing_entry_node_is_rejected(valid_template: dict) -> None:
    template = copy.deepcopy(valid_template)
    template["entry_node"] = "nowhere"

    result = validate_template(template)

    assert not result.is_valid
    assert result.errors == ["entry_node 'nowhere' does not match any node id."]


def test_condition_requires_default_branch(valid_template: dict) -> None:
    template = copy.deepcopy(valid_template)
    template["nodes"][1]["branches"].pop("default")

    result = validate_template(template)

    assert not result.is_valid
    assert any("Condition nodes must define a 'default' branch." in error for error in result.errors)


def test_duplicate_node_ids_are_rejected(valid_template: dict) -> None:
    template = copy.deepcopy(valid_template)
    template["nodes"].append(
        {
            "id": "respond",
            "type": "terminal_response",
            "config": {"template": "duplicate"},
        }
    )

    result = validate_template(template)

    assert not result.is_valid
    assert result.errors == ["Duplicate node id 'respond'."]


def test_schema_errors_are_human_readable() -> None:
    result = validate_template(
        {
            "template_version": "not-a-version",
            "entry_node": "respond",
            "nodes": [{"id": "respond", "type": "unknown_type", "config": {}}],
        }
    )

    assert not result.is_valid
    assert any("template_version" in error for error in result.errors)
    assert any("nodes.0" in error for error in result.errors)


def test_non_dict_input_is_rejected() -> None:
    result = validate_template("{}")  # type: ignore[arg-type]

    assert result.errors == ["Template must be a JSON object."]


def test_service_call_http_requires_url(valid_template: dict) -> None:
    template = copy.deepcopy(valid_template)
    template["nodes"][3]["config"] = {
        "mode": "http",
        "method": "GET",
    }

    result = validate_template(template)

    assert not result.is_valid
    assert any("mode='http' requires a non-empty url" in error for error in result.errors)


def test_service_call_tool_requires_name_or_id(valid_template: dict) -> None:
    template = copy.deepcopy(valid_template)
    template["nodes"][3]["config"] = {
        "mode": "tool",
    }

    result = validate_template(template)

    assert not result.is_valid
    assert any("mode='tool' requires tool_name or tool_id" in error for error in result.errors)


def test_llm_step_system_prompt_rejects_unscoped_placeholders(valid_template: dict) -> None:
    template = copy.deepcopy(valid_template)
    template["nodes"][4]["config"]["system_prompt"] = "Use {{ input }} and {{ parsed_data.amount }}"

    result = validate_template(template)

    assert not result.is_valid
    assert any(
        "llm_step system_prompt placeholders must reference parsed_data.* or service_results.*"
        in error
        for error in result.errors
    )
