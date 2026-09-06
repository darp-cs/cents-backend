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
                    "output_key": "parsed_request",
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
                    "service": "ledger",
                    "operation": "record_transaction",
                    "parameters": {"amount": "{{ parsed_request.amount }}"},
                    "output_key": "ledger_result",
                },
                "next": "summarize",
            },
            {
                "id": "summarize",
                "type": "llm_step",
                "config": {
                    "prompt_template": "Summarize {{ ledger_result }}",
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


def test_missing_terminal_node_is_rejected() -> None:
    template = {
        "template_version": "1.0",
        "entry_node": "loop_a",
        "nodes": [
            {
                "id": "loop_a",
                "type": "llm_step",
                "config": {"prompt_template": "a", "output_key": "a"},
                "next": "loop_b",
            },
            {
                "id": "loop_b",
                "type": "llm_step",
                "config": {"prompt_template": "b", "output_key": "b"},
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
                "branches": {"ok": "respond", "stuck": "dead_end_a"},
            },
            {
                "id": "dead_end_a",
                "type": "llm_step",
                "config": {"prompt_template": "a", "output_key": "a"},
                "next": "dead_end_b",
            },
            {
                "id": "dead_end_b",
                "type": "llm_step",
                "config": {"prompt_template": "b", "output_key": "b"},
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
