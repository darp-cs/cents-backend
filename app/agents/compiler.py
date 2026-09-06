from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.state import SubAgentState
from app.agents.template_schema import (
    AgentNode,
    AgentTemplate,
    ConditionNode,
    LLMStepNode,
    ServiceCallNode,
    StructuredParserNode,
    TerminalResponseNode,
    UserInterruptNode,
)

NodeExecutor = Callable[[AgentNode, SubAgentState], SubAgentState]
CompiledGraphCacheKey = tuple[str, int]

_COMPILED_GRAPH_CACHE: dict[CompiledGraphCacheKey, CompiledStateGraph] = {}


def compile_agent_graph(template: AgentTemplate) -> CompiledStateGraph:
    workflow = StateGraph(SubAgentState)
    nodes_by_id = {node.id: node for node in template.nodes}

    for node in template.nodes:
        workflow.add_node(node.id, _build_node_handler(node))

    workflow.set_entry_point(template.entry_node)

    for node in template.nodes:
        if isinstance(node, ConditionNode):
            workflow.add_conditional_edges(node.id, _build_condition_router(node), node.branches)
            continue

        if isinstance(node, TerminalResponseNode):
            workflow.add_edge(node.id, END)
            continue

        workflow.add_edge(node.id, node.next)

    return workflow.compile()


def get_compiled_agent_graph(name: str, version: int, template: AgentTemplate) -> CompiledStateGraph:
    cache_key = (name.strip(), version)
    cached_graph = _COMPILED_GRAPH_CACHE.get(cache_key)
    if cached_graph is not None:
        return cached_graph

    compiled_graph = compile_agent_graph(template)
    _COMPILED_GRAPH_CACHE[cache_key] = compiled_graph
    return compiled_graph


def invalidate_compiled_agent_graph(name: str, version: int | None = None) -> None:
    normalized_name = name.strip()
    if version is not None:
        _COMPILED_GRAPH_CACHE.pop((normalized_name, version), None)
        return

    keys_to_delete = [key for key in _COMPILED_GRAPH_CACHE if key[0] == normalized_name]
    for key in keys_to_delete:
        _COMPILED_GRAPH_CACHE.pop(key, None)


def clear_compiled_agent_graph_cache() -> None:
    _COMPILED_GRAPH_CACHE.clear()


def _build_node_handler(node: AgentNode) -> Callable[[SubAgentState], SubAgentState]:
    executor = _NODE_EXECUTORS[node.type]

    def _handler(state: SubAgentState) -> SubAgentState:
        return executor(node, state)

    return _handler


def _build_condition_router(node: ConditionNode) -> Callable[[SubAgentState], str]:
    def _router(state: SubAgentState) -> str:
        return _resolve_condition_branch(node, state)

    return _router


def _resolve_condition_branch(node: ConditionNode, state: SubAgentState) -> str:
    parsed_data = state.get("parsed_data", {})
    if isinstance(parsed_data, dict):
        branch_overrides = parsed_data.get("branch_overrides")
        if isinstance(branch_overrides, dict):
            candidate = branch_overrides.get(node.id)
            if isinstance(candidate, str) and candidate in node.branches:
                return candidate

        direct_candidate = parsed_data.get(f"{node.id}_branch")
        if isinstance(direct_candidate, str) and direct_candidate in node.branches:
            return direct_candidate

    # Default to the first configured branch for deterministic routing with stub handlers.
    return next(iter(node.branches))


def _touch_iteration(state: SubAgentState) -> SubAgentState:
    next_state = dict(state)
    next_state["iteration_count"] = int(next_state.get("iteration_count", 0)) + 1
    return next_state


def _execute_structured_parser(node: AgentNode, state: SubAgentState) -> SubAgentState:
    assert isinstance(node, StructuredParserNode)
    next_state = _touch_iteration(state)
    parsed_data = dict(next_state.get("parsed_data", {}))

    source_value: Any = next_state.get("input", "")
    if node.config.source_key == "last_message":
        messages = next_state.get("messages", [])
        if messages:
            source_value = messages[-1].get("content", source_value)

    parsed_data[node.config.output_key] = source_value
    next_state["parsed_data"] = parsed_data
    return next_state


def _execute_condition(node: AgentNode, state: SubAgentState) -> SubAgentState:
    assert isinstance(node, ConditionNode)
    return _touch_iteration(state)


def _execute_service_call(node: AgentNode, state: SubAgentState) -> SubAgentState:
    assert isinstance(node, ServiceCallNode)
    next_state = _touch_iteration(state)
    service_results = dict(next_state.get("service_results", {}))
    service_results[node.id] = {
        "service": node.config.service,
        "operation": node.config.operation,
        "status": "stubbed",
    }
    next_state["service_results"] = service_results
    return next_state


def _execute_user_interrupt(node: AgentNode, state: SubAgentState) -> SubAgentState:
    assert isinstance(node, UserInterruptNode)
    next_state = _touch_iteration(state)
    next_state["interrupt_payload"] = {
        "prompt": node.config.prompt,
        "expected_type": node.config.expected_type,
        "choices": node.config.choices,
    }
    return next_state


def _execute_llm_step(node: AgentNode, state: SubAgentState) -> SubAgentState:
    assert isinstance(node, LLMStepNode)
    next_state = _touch_iteration(state)
    messages = list(next_state.get("messages", []))
    messages.append(
        {
            "role": "assistant",
            "content": f"[stub:{node.id}] {node.config.prompt_template}",
        }
    )
    next_state["messages"] = messages
    return next_state


def _execute_terminal_response(node: AgentNode, state: SubAgentState) -> SubAgentState:
    assert isinstance(node, TerminalResponseNode)
    next_state = _touch_iteration(state)
    next_state["final_response"] = node.config.template
    return next_state


_NODE_EXECUTORS: dict[str, NodeExecutor] = {
    "structured_parser": _execute_structured_parser,
    "condition": _execute_condition,
    "service_call": _execute_service_call,
    "user_interrupt": _execute_user_interrupt,
    "llm_step": _execute_llm_step,
    "terminal_response": _execute_terminal_response,
}
