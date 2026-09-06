from __future__ import annotations

import asyncio
import json
import re
import threading
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
    ParsedField,
    ServiceCallNode,
    StructuredParserConfig,
    StructuredParserNode,
    TerminalResponseNode,
    UserInterruptNode,
)
from app.config import settings
from app.llm.client import LLMClientError, generate_text

NodeExecutor = Callable[[AgentNode, SubAgentState], SubAgentState]
CompiledGraphCacheKey = tuple[str, int]

_COMPILED_GRAPH_CACHE: dict[CompiledGraphCacheKey, CompiledStateGraph] = {}


def compile_agent_graph(template: AgentTemplate) -> CompiledStateGraph:
    workflow = StateGraph(SubAgentState)

    for node in template.nodes:
        workflow.add_node(node.id, _build_node_handler(node))

    workflow.set_entry_point(template.entry_node)

    for node in template.nodes:
        if isinstance(node, ConditionNode):
            workflow.add_conditional_edges(node.id, _build_condition_router(node), node.branches)
            continue

        if isinstance(node, StructuredParserNode) and node.on_failure:
            workflow.add_conditional_edges(
                node.id,
                _build_structured_parser_router(node),
                {
                    "success": node.next,
                    "failure": node.on_failure,
                },
            )
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


def _build_structured_parser_router(node: StructuredParserNode) -> Callable[[SubAgentState], str]:
    def _router(state: SubAgentState) -> str:
        return _resolve_structured_parser_route(node.id, state)

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


def _resolve_structured_parser_route(node_id: str, state: SubAgentState) -> str:
    parsed_data = state.get("parsed_data", {})
    if isinstance(parsed_data, dict):
        parser_status = parsed_data.get("__parser_status__", {})
        if isinstance(parser_status, dict) and parser_status.get(node_id) == "failure":
            return "failure"
    return "success"


def _touch_iteration(state: SubAgentState) -> SubAgentState:
    next_state = dict(state)
    next_state["iteration_count"] = int(next_state.get("iteration_count", 0)) + 1
    return next_state


def _execute_structured_parser(node: AgentNode, state: SubAgentState) -> SubAgentState:
    assert isinstance(node, StructuredParserNode)
    next_state = _touch_iteration(state)
    source_text = _resolve_source_text(node.config, next_state)

    try:
        extracted = _extract_structured_data(node=node, source_text=source_text, state=next_state)
        normalized = _normalize_structured_fields(node.config.fields, extracted)
    except RuntimeError as exc:
        return _handle_structured_parser_failure(node=node, state=next_state, message=str(exc))

    parsed_data = dict(next_state.get("parsed_data", {}))
    parsed_data.update(normalized)
    _set_parser_status(parsed_data, node.id, "success")
    next_state["parsed_data"] = parsed_data
    return next_state


def _resolve_source_text(config: StructuredParserConfig, state: SubAgentState) -> str:
    source_key = config.source_key.strip().lower()
    if source_key == "input":
        return str(state.get("input", ""))

    if source_key == "messages":
        messages = state.get("messages", [])
        rendered: list[str] = []
        for item in messages:
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if content:
                rendered.append(f"{role}: {content}" if role else content)
        return "\n".join(rendered)

    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        return str(last_message.get("content", state.get("input", "")))
    return str(state.get("input", ""))


def _extract_structured_data(
    node: StructuredParserNode,
    source_text: str,
    state: SubAgentState,
) -> dict[str, Any]:
    if node.config.strategy == "llm":
        return _extract_with_llm(node=node, source_text=source_text, state=state)
    return _extract_with_regex_or_keywords(node.config, source_text)


def _extract_with_regex_or_keywords(config: StructuredParserConfig, source_text: str) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    for field in config.fields:
        value = _extract_field_value(field, source_text, config.regex_patterns)
        if value is not None:
            extracted[field.name] = value
    return extracted


def _extract_field_value(
    field: ParsedField,
    source_text: str,
    regex_patterns: dict[str, str],
) -> Any:
    explicit_pattern = regex_patterns.get(field.name)
    if explicit_pattern:
        match = re.search(explicit_pattern, source_text, flags=re.IGNORECASE)
        if match is None:
            return None
        if match.groups():
            return match.group(1)
        return match.group(0)

    if field.type == "enum" and field.enum_values:
        source_lc = source_text.lower()
        for candidate in sorted(field.enum_values, key=len, reverse=True):
            if candidate.lower() in source_lc:
                return candidate
        return None

    if field.type == "number":
        match = re.search(r"-?\d+(?:\.\d+)?", source_text)
        return match.group(0) if match else None

    if field.type == "boolean":
        match = re.search(r"\b(true|false|yes|no)\b", source_text, flags=re.IGNORECASE)
        return match.group(1) if match else None

    if field.type == "date":
        match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", source_text)
        return match.group(0) if match else None

    if field.type == "array":
        if "," not in source_text:
            return None
        return [item.strip() for item in source_text.split(",") if item.strip()]

    cleaned = source_text.strip()
    return cleaned if cleaned else None


def _extract_with_llm(node: StructuredParserNode, source_text: str, state: SubAgentState) -> dict[str, Any]:
    model_type, model = _resolve_parser_model(node, state)
    system_prompt = _build_parser_system_prompt(node)
    request_payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": source_text}],
        "system_prompt": system_prompt,
        "model_folder": model_type,
        "temperature": (
            node.config.llm_temperature
            if node.config.llm_temperature is not None
            else settings.llm_default_temperature
        ),
        "max_tokens": (
            node.config.llm_max_tokens if node.config.llm_max_tokens is not None else settings.llm_default_max_tokens
        ),
        "metadata": {
            "node": node.id,
            "component": "structured_parser",
        },
    }
    if model:
        request_payload["model"] = model

    try:
        payload = _run_async(generate_text(request_payload))
    except LLMClientError as exc:
        raise RuntimeError(str(exc)) from exc

    if not isinstance(payload, dict):
        raise RuntimeError("LLM parser returned an invalid payload.")

    raw_text = str(payload.get("text", "")).strip()
    if not raw_text:
        raise RuntimeError("LLM parser returned empty text.")

    json_text = _unwrap_json_block(raw_text)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM parser returned invalid JSON.") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("LLM parser response must be a JSON object.")
    return parsed


def _resolve_parser_model(node: StructuredParserNode, state: SubAgentState) -> tuple[str, str | None]:
    config = node.config
    if config.llm_model_type and config.llm_model_type.strip():
        model_type = config.llm_model_type.strip()
    else:
        node_llm_configs = state.get("node_llm_configs", {})
        raw_model_type = node_llm_configs.get(node.id, {}).get("model_type")
        model_type = str(raw_model_type).strip() if raw_model_type else ""
        if not model_type:
            model_type = settings.llm_default_generation_model_type.strip()

    if not model_type:
        raise RuntimeError(f"Structured parser node '{node.id}' requires model_type for llm strategy.")

    if config.llm_model and config.llm_model.strip():
        model = config.llm_model.strip()
    else:
        node_llm_configs = state.get("node_llm_configs", {})
        raw_model = node_llm_configs.get(node.id, {}).get("model")
        model = str(raw_model).strip() if raw_model else ""
        model = model or settings.llm_default_generation_model.strip()

    return model_type, (model or None)


def _build_parser_system_prompt(node: StructuredParserNode) -> str:
    field_specs: list[str] = []
    for field in node.config.fields:
        requirement = "required" if field.required else "optional"
        enum_values = f"; enum_values={field.enum_values}" if field.enum_values else ""
        description = f"; description={field.description}" if field.description else ""
        field_specs.append(f"- {field.name}: {field.type} ({requirement}){enum_values}{description}")

    instructions = node.config.llm_prompt_instructions or ""
    return (
        "Extract structured data and return JSON only. Do not include markdown, code fences, or extra text.\n"
        f"Schema:\n{chr(10).join(field_specs)}\n"
        "Only include keys present in the schema."
        + (f"\nAdditional instructions: {instructions}" if instructions.strip() else "")
    )


def _unwrap_json_block(raw_text: str) -> str:
    if raw_text.startswith("```"):
        stripped = raw_text.strip("`")
        lines = stripped.splitlines()
        if lines and lines[0].lower().startswith("json"):
            lines = lines[1:]
        return "\n".join(lines).strip()
    return raw_text


def _normalize_structured_fields(fields: list[ParsedField], extracted: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    errors: list[str] = []

    for field in fields:
        if field.name not in extracted:
            if field.required:
                errors.append(f"Missing required field '{field.name}'.")
            continue

        try:
            normalized[field.name] = _coerce_field_value(field, extracted[field.name])
        except ValueError as exc:
            errors.append(f"{field.name}: {exc}")

    if errors:
        raise RuntimeError("; ".join(errors))
    return normalized


def _coerce_field_value(field: ParsedField, value: Any) -> Any:
    if field.type == "string":
        return str(value)

    if field.type == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if re.fullmatch(r"-?\d+", raw):
                return int(raw)
            if re.fullmatch(r"-?\d+\.\d+", raw):
                return float(raw)
        raise ValueError("expected number")

    if field.type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        raise ValueError("expected boolean")

    if field.type == "date":
        text = str(value).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return text
        raise ValueError("expected date in YYYY-MM-DD format")

    if field.type == "enum":
        if not field.enum_values:
            raise ValueError("enum field requires enum_values")
        text = str(value).strip()
        for candidate in field.enum_values:
            if candidate.lower() == text.lower():
                return candidate
        raise ValueError(f"expected one of {field.enum_values}")

    if field.type == "array":
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
            if items:
                return items
        raise ValueError("expected array")

    raise ValueError(f"unsupported field type '{field.type}'")


def _handle_structured_parser_failure(
    node: StructuredParserNode,
    state: SubAgentState,
    message: str,
) -> SubAgentState:
    if node.on_failure is None:
        raise RuntimeError(f"Structured parser node '{node.id}' failed: {message}")

    parsed_data = dict(state.get("parsed_data", {}))
    parser_errors = parsed_data.get("__parser_errors__", {})
    if not isinstance(parser_errors, dict):
        parser_errors = {}
    parser_errors[node.id] = message
    parsed_data["__parser_errors__"] = parser_errors
    _set_parser_status(parsed_data, node.id, "failure")

    next_state = dict(state)
    next_state["parsed_data"] = parsed_data
    return next_state


def _set_parser_status(parsed_data: dict[str, Any], node_id: str, status: str) -> None:
    parser_status = parsed_data.get("__parser_status__", {})
    if not isinstance(parser_status, dict):
        parser_status = {}
    parser_status[node_id] = status
    parsed_data["__parser_status__"] = parser_status


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_container: dict[str, Any] = {}
    error_container: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_container["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive propagation path.
            error_container["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in error_container:
        raise error_container["error"]
    return result_container.get("value")


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
