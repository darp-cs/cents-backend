from __future__ import annotations

from typing import Any, TypedDict


class NodeLLMConfig(TypedDict, total=False):
    model_type: str
    model: str | None


class SubAgentState(TypedDict, total=False):
    input: str
    parsed_data: dict[str, Any]
    messages: list[dict[str, Any]]
    iteration_count: int
    service_results: dict[str, Any]
    interrupt_payload: dict[str, Any] | None
    final_response: str
    node_llm_configs: dict[str, NodeLLMConfig]
    platform_guardrails: dict[str, Any]
    template_guardrails: dict[str, Any]
    guardrail_policy: dict[str, Any]
    guardrail_verdict: dict[str, Any] | None
    metric_events: list[dict[str, Any]]
    error_event: dict[str, Any] | None
    run_status: str
    _halt_execution: bool
