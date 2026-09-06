from __future__ import annotations

import asyncio
import ast
import json
import re
import threading
import uuid
from urllib.parse import urlparse
from collections.abc import Callable
from typing import Any, Hashable, cast

import httpx
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from sqlalchemy import select

from app.agents.state import SubAgentState
from app.agents.template_schema import (
    AgentNode,
    AgentTemplate,
    ConditionNode,
    LLMStepNode,
    ParsedField,
    ServiceCallConfig,
    ServiceCallNode,
    StructuredParserConfig,
    StructuredParserNode,
    TerminalResponseNode,
    UserInterruptNode,
)
from app.config import settings
from app.db.base import AsyncSessionLocal
from app.db.models import ToolDefinition
from app.graph.checkpointer import ensure_sync_checkpointer_ready
from app.llm.client import LLMClientError, generate_text

NodeExecutor = Callable[[AgentNode, SubAgentState], SubAgentState]
ToolExecutor = Callable[[dict[str, Any], SubAgentState], Any]
CompiledGraphCacheKey = tuple[str, int]

_COMPILED_GRAPH_CACHE: dict[CompiledGraphCacheKey, CompiledStateGraph] = {}
_TOOL_EXECUTORS_BY_ID: dict[str, ToolExecutor] = {}
_TOOL_EXECUTORS_BY_NAME: dict[str, ToolExecutor] = {}


def compile_agent_graph(template: AgentTemplate) -> CompiledStateGraph:
    workflow = StateGraph(SubAgentState)

    for node in template.nodes:
        workflow.add_node(node.id, _build_node_handler(node))

    workflow.set_entry_point(template.entry_node)

    for node in template.nodes:
        if isinstance(node, ConditionNode):
            workflow.add_conditional_edges(
                node.id,
                _build_condition_router(node),
                cast(dict[Hashable, str], node.branches),
            )
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

        if isinstance(node, ServiceCallNode) and node.on_failure:
            workflow.add_conditional_edges(
                node.id,
                _build_service_call_router(node.id),
                {
                    "success": node.next,
                    "failure": node.on_failure,
                },
            )
            continue

        if isinstance(node, LLMStepNode) and node.on_failure:
            workflow.add_conditional_edges(
                node.id,
                _build_llm_step_router(node.id),
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

    requires_checkpointing = any(isinstance(node, UserInterruptNode) for node in template.nodes)
    if not requires_checkpointing:
        return workflow.compile()

    checkpointer = ensure_sync_checkpointer_ready()
    return workflow.compile(checkpointer=checkpointer) if checkpointer is not None else workflow.compile()


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


def register_tool_executor(
    *,
    tool_name: str | None = None,
    tool_id: str | None = None,
    executor: ToolExecutor,
) -> None:
    if not tool_name and not tool_id:
        raise ValueError("register_tool_executor requires tool_name or tool_id.")

    if tool_name:
        _TOOL_EXECUTORS_BY_NAME[tool_name.strip()] = executor
    if tool_id:
        _TOOL_EXECUTORS_BY_ID[tool_id.strip()] = executor


def clear_tool_executors() -> None:
    _TOOL_EXECUTORS_BY_ID.clear()
    _TOOL_EXECUTORS_BY_NAME.clear()


def _build_node_handler(node: AgentNode) -> Callable[[SubAgentState], SubAgentState]:
    executor = _NODE_EXECUTORS[node.type]

    def _handler(state: SubAgentState) -> SubAgentState:
        return executor(node, state)

    return _handler


def _build_condition_router(node: ConditionNode) -> Callable[[SubAgentState], str]:
    def _router(state: SubAgentState) -> str:
        return _resolve_condition_route(node, state)

    return _router


def _build_structured_parser_router(node: StructuredParserNode) -> Callable[[SubAgentState], str]:
    def _router(state: SubAgentState) -> str:
        return _resolve_structured_parser_route(node.id, state)

    return _router


def _build_service_call_router(node_id: str) -> Callable[[SubAgentState], str]:
    def _router(state: SubAgentState) -> str:
        return _resolve_service_call_route(node_id, state)

    return _router


def _build_llm_step_router(node_id: str) -> Callable[[SubAgentState], str]:
    def _router(state: SubAgentState) -> str:
        return _resolve_llm_step_route(node_id, state)

    return _router


def _resolve_condition_route(node: ConditionNode, state: SubAgentState) -> str:
    parsed_data = state.get("parsed_data", {})
    if isinstance(parsed_data, dict):
        condition_branches = parsed_data.get("__condition_branches__", {})
        if isinstance(condition_branches, dict):
            candidate = condition_branches.get(node.id)
            if isinstance(candidate, str) and candidate in node.branches:
                return candidate

    return "default"


def _resolve_structured_parser_route(node_id: str, state: SubAgentState) -> str:
    parsed_data = state.get("parsed_data", {})
    if isinstance(parsed_data, dict):
        parser_status = parsed_data.get("__parser_status__", {})
        if isinstance(parser_status, dict) and parser_status.get(node_id) == "failure":
            return "failure"
    return "success"


def _resolve_service_call_route(node_id: str, state: SubAgentState) -> str:
    service_results = state.get("service_results", {})
    if isinstance(service_results, dict):
        node_result = service_results.get(node_id)
        if isinstance(node_result, dict) and node_result.get("__status__") == "failure":
            return "failure"
    return "success"


def _resolve_llm_step_route(node_id: str, state: SubAgentState) -> str:
    parsed_data = state.get("parsed_data", {})
    if isinstance(parsed_data, dict):
        llm_step_status = parsed_data.get("__llm_step_status__", {})
        if isinstance(llm_step_status, dict) and llm_step_status.get(node_id) == "failure":
            return "failure"
    return "success"


def _evaluate_condition_branch(node: ConditionNode, state: SubAgentState) -> str:
    parsed_data = state.get("parsed_data", {})
    service_results = state.get("service_results", {})

    if not isinstance(parsed_data, dict):
        raise RuntimeError("parsed_data must be an object for condition evaluation.")
    if not isinstance(service_results, dict):
        raise RuntimeError("service_results must be an object for condition evaluation.")

    expression_result = _safe_evaluate_expression(
        expression=node.config.expression,
        context={
            "parsed_data": parsed_data,
            "service_results": service_results,
        },
    )
    branch_key = _normalize_condition_result(expression_result)
    if branch_key in node.branches:
        return branch_key
    return "default"


def _safe_evaluate_expression(expression: str, context: dict[str, Any]) -> Any:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise RuntimeError(f"Invalid condition expression syntax: {exc.msg}.") from exc

    return _evaluate_ast_node(parsed.body, context)


def _evaluate_ast_node(node: ast.AST, context: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id in context:
            return context[node.id]
        if node.id == "true":
            return True
        if node.id == "false":
            return False
        if node.id == "null":
            return None
        raise RuntimeError(f"Unsupported name '{node.id}' in condition expression.")

    if isinstance(node, ast.Attribute):
        target = _evaluate_ast_node(node.value, context)
        if isinstance(target, dict):
            if node.attr in target:
                return target[node.attr]
            raise RuntimeError(f"Missing field '{node.attr}' in condition expression.")
        raise RuntimeError("Attribute access is only supported on objects in condition expressions.")

    if isinstance(node, ast.Subscript):
        target = _evaluate_ast_node(node.value, context)
        key = _evaluate_ast_node(node.slice, context)
        try:
            return target[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Missing field '{key}' in condition expression.") from exc

    if isinstance(node, ast.BoolOp):
        values = [_evaluate_ast_node(value_node, context) for value_node in node.values]
        if isinstance(node.op, ast.And):
            return all(bool(value) for value in values)
        if isinstance(node.op, ast.Or):
            return any(bool(value) for value in values)
        raise RuntimeError("Unsupported boolean operator in condition expression.")

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not bool(_evaluate_ast_node(node.operand, context))

    if isinstance(node, ast.BinOp):
        left = _evaluate_ast_node(node.left, context)
        right = _evaluate_ast_node(node.right, context)
        return _apply_binary_operator(node.op, left, right)

    if isinstance(node, ast.Compare):
        left = _evaluate_ast_node(node.left, context)
        for operator, comparator in zip(node.ops, node.comparators):
            right = _evaluate_ast_node(comparator, context)
            if not _apply_comparison_operator(operator, left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.List):
        return [_evaluate_ast_node(item, context) for item in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_evaluate_ast_node(item, context) for item in node.elts)

    raise RuntimeError(f"Unsupported expression construct: {node.__class__.__name__}.")


def _apply_binary_operator(operator: ast.operator, left: Any, right: Any) -> Any:
    try:
        if isinstance(operator, ast.Add):
            return left + right
        if isinstance(operator, ast.Sub):
            return left - right
        if isinstance(operator, ast.Mult):
            return left * right
        if isinstance(operator, ast.Div):
            return left / right
        if isinstance(operator, ast.Mod):
            return left % right
    except TypeError as exc:
        raise RuntimeError("Type mismatch while evaluating condition expression.") from exc

    raise RuntimeError("Unsupported binary operator in condition expression.")


def _apply_comparison_operator(operator: ast.cmpop, left: Any, right: Any) -> bool:
    try:
        if isinstance(operator, ast.Eq):
            return left == right
        if isinstance(operator, ast.NotEq):
            return left != right
        if isinstance(operator, ast.Gt):
            return left > right
        if isinstance(operator, ast.GtE):
            return left >= right
        if isinstance(operator, ast.Lt):
            return left < right
        if isinstance(operator, ast.LtE):
            return left <= right
        if isinstance(operator, ast.In):
            return left in right
        if isinstance(operator, ast.NotIn):
            return left not in right
    except TypeError as exc:
        raise RuntimeError("Type mismatch while evaluating condition expression.") from exc

    raise RuntimeError("Unsupported comparison operator in condition expression.")


def _normalize_condition_result(result: Any) -> str:
    if isinstance(result, bool):
        return "true" if result else "false"
    if result is None:
        return "null"
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        return str(result)
    if isinstance(result, str):
        cleaned = result.strip()
        if cleaned:
            return cleaned
        raise RuntimeError("Condition expression returned an empty string branch key.")
    raise RuntimeError(f"Condition expression returned unsupported type '{type(result).__name__}'.")


def _set_condition_branch(parsed_data: dict[str, Any], node_id: str, branch: str) -> None:
    condition_branches = parsed_data.get("__condition_branches__", {})
    if not isinstance(condition_branches, dict):
        condition_branches = {}
    condition_branches[node_id] = branch
    parsed_data["__condition_branches__"] = condition_branches


def _record_condition_error(parsed_data: dict[str, Any], node_id: str, message: str) -> None:
    condition_errors = parsed_data.get("__condition_errors__", {})
    if not isinstance(condition_errors, dict):
        condition_errors = {}
    condition_errors[node_id] = message
    parsed_data["__condition_errors__"] = condition_errors


def _touch_iteration(state: SubAgentState) -> SubAgentState:
    next_state = cast(SubAgentState, dict(state))
    raw_count = next_state.get("iteration_count", 0)
    try:
        current_count = int(cast(Any, raw_count))
    except (TypeError, ValueError):
        current_count = 0

    next_state["iteration_count"] = current_count + 1
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

    next_state = cast(SubAgentState, dict(state))
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
    next_state = _touch_iteration(state)
    parsed_data = next_state.get("parsed_data", {})
    normalized_parsed_data = dict(parsed_data) if isinstance(parsed_data, dict) else {}

    try:
        selected_branch = _evaluate_condition_branch(node, next_state)
    except RuntimeError as exc:
        selected_branch = "default"
        _record_condition_error(normalized_parsed_data, node.id, str(exc))

    # Persist branch decisions in state so routing remains deterministic and observable.
    _set_condition_branch(normalized_parsed_data, node.id, selected_branch)
    next_state["parsed_data"] = normalized_parsed_data
    return next_state


def _execute_service_call(node: AgentNode, state: SubAgentState) -> SubAgentState:
    assert isinstance(node, ServiceCallNode)
    next_state = _touch_iteration(state)
    try:
        response_payload = _execute_service_call_request(node.config, next_state)
    except RuntimeError as exc:
        return _handle_service_call_failure(node=node, state=next_state, message=str(exc))

    service_results = dict(next_state.get("service_results", {}))
    service_results[node.id] = {
        "__status__": "success",
        **response_payload,
    }
    next_state["service_results"] = service_results
    return next_state


def _handle_service_call_failure(
    node: ServiceCallNode,
    state: SubAgentState,
    message: str,
) -> SubAgentState:
    if node.on_failure is None:
        raise RuntimeError(f"Service call node '{node.id}' failed: {message}")

    service_results = dict(state.get("service_results", {}))
    service_results[node.id] = {
        "__status__": "failure",
        "error": message,
    }

    next_state = cast(SubAgentState, dict(state))
    next_state["service_results"] = service_results
    return next_state


def _execute_service_call_request(config: ServiceCallConfig, state: SubAgentState) -> dict[str, Any]:
    if config.mode == "tool":
        return _execute_tool_service_call(config, state)
    return _execute_http_service_call(config, state)


def _execute_http_service_call(config: ServiceCallConfig, state: SubAgentState) -> dict[str, Any]:
    if config.url is None:
        raise RuntimeError("service_call mode='http' requires a url.")

    rendered_url = str(_render_template_value(config.url, state)).strip()
    _assert_destination_allowed(rendered_url, config.allow_unsafe_destination)

    headers = _render_http_headers(config.headers_template, state)
    if config.body_template is not None:
        _enforce_secret_reference_for_sensitive_body(config.body_template)
    body = _render_template_value(config.body_template, state) if config.body_template is not None else None
    timeout = config.timeout_seconds or settings.service_call_default_timeout_seconds

    try:
        response = httpx.request(
            method=config.method,
            url=rendered_url,
            headers=headers,
            json=body,
            timeout=timeout,
            follow_redirects=False,
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError("HTTP service call timed out.") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"HTTP service call failed: {exc}") from exc

    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(
            f"HTTP service call returned non-2xx status {response.status_code}: {response.text.strip()}"
        )

    payload: Any
    try:
        payload = response.json()
    except ValueError:
        payload = response.text

    return {
        "mode": "http",
        "url": rendered_url,
        "method": config.method,
        "status_code": response.status_code,
        "body": payload,
    }


def _assert_destination_allowed(url: str, allow_unsafe_destination: bool) -> None:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise RuntimeError("Only http/https URLs are allowed for service_call http mode.")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise RuntimeError("service_call url must include a hostname.")

    allowed_hosts = {host.lower() for host in settings.service_call_allowed_hosts}
    if hostname in allowed_hosts:
        return

    if allow_unsafe_destination and settings.service_call_allow_unsafe_destinations:
        return

    raise RuntimeError(
        f"Destination host '{hostname}' is not in service_call_allowed_hosts. "
        "Use server-side opt-in to allow unsafe destinations."
    )


def _render_http_headers(headers_template: dict[str, str], state: SubAgentState) -> dict[str, str]:
    rendered: dict[str, str] = {}
    for raw_name, raw_value in headers_template.items():
        header_name = str(raw_name).strip()
        if not header_name:
            continue

        template_value = str(raw_value)
        _enforce_secret_reference_for_sensitive_header(header_name, template_value)
        rendered_value = _render_template_value(template_value, state)
        rendered[header_name] = str(rendered_value)
    return rendered


def _enforce_secret_reference_for_sensitive_header(header_name: str, template_value: str) -> None:
    sensitive_headers = {"authorization", "x-api-key", "api-key", "x-auth-token"}
    if header_name.strip().lower() not in sensitive_headers:
        return

    if "{{ secret." not in template_value and "{{ secrets." not in template_value:
        raise RuntimeError(
            f"Sensitive header '{header_name}' must use a server-side secret reference placeholder."
        )


def _enforce_secret_reference_for_sensitive_body(template_value: Any, path: str = "") -> None:
    sensitive_keys = {"api_key", "apikey", "token", "secret", "password"}

    if isinstance(template_value, dict):
        for key, value in template_value.items():
            key_name = str(key).strip()
            next_path = f"{path}.{key_name}" if path else key_name
            if key_name.lower() in sensitive_keys and isinstance(value, str):
                if "{{ secret." not in value and "{{ secrets." not in value:
                    raise RuntimeError(
                        f"Sensitive body field '{next_path}' must use a server-side secret placeholder."
                    )
            _enforce_secret_reference_for_sensitive_body(value, next_path)
        return

    if isinstance(template_value, list):
        for index, item in enumerate(template_value):
            _enforce_secret_reference_for_sensitive_body(item, f"{path}[{index}]")


def _render_template_value(template: Any, state: SubAgentState) -> Any:
    if isinstance(template, str):
        return _render_template_string(template, state)
    if isinstance(template, dict):
        return {str(key): _render_template_value(value, state) for key, value in template.items()}
    if isinstance(template, list):
        return [_render_template_value(item, state) for item in template]
    return template


def _render_template_string(template: str, state: SubAgentState) -> Any:
    pattern = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
    matches = list(pattern.finditer(template))
    if not matches:
        return template

    if len(matches) == 1 and matches[0].span() == (0, len(template)):
        return _resolve_template_reference(matches[0].group(1), state)

    rendered = template
    for match in matches:
        placeholder = match.group(0)
        resolved = _resolve_template_reference(match.group(1), state)
        rendered = rendered.replace(placeholder, str(resolved))
    return rendered


def _resolve_template_reference(reference: str, state: SubAgentState) -> Any:
    path = reference.strip()
    if path.startswith("state."):
        path = path[len("state.") :]

    if path.startswith("secret."):
        secret_key = path[len("secret.") :].strip()
        return _resolve_secret(secret_key)
    if path.startswith("secrets."):
        secret_key = path[len("secrets.") :].strip()
        return _resolve_secret(secret_key)

    return _resolve_state_reference(state, path)


def _resolve_secret(secret_key: str) -> str:
    if not secret_key:
        raise RuntimeError("Secret placeholder must include a key name.")

    secret_value = settings.service_call_secrets.get(secret_key)
    if secret_value is None:
        raise RuntimeError(f"Secret '{secret_key}' is not configured on the server.")
    return secret_value


def _resolve_state_reference(state: SubAgentState, path: str) -> Any:
    if not path:
        raise RuntimeError("Template placeholder path cannot be empty.")

    current: Any = state
    parts = [part.strip() for part in path.split(".") if part.strip()]
    if not parts:
        raise RuntimeError("Template placeholder path cannot be empty.")

    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                raise RuntimeError(f"Template placeholder '{path}' references missing field '{part}'.")
            current = current[part]
            continue

        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index < 0 or index >= len(current):
                raise RuntimeError(f"Template placeholder '{path}' references invalid list index '{part}'.")
            current = current[index]
            continue

        raise RuntimeError(f"Template placeholder '{path}' cannot resolve segment '{part}'.")

    return current


def _execute_tool_service_call(config: ServiceCallConfig, state: SubAgentState) -> dict[str, Any]:
    tool_definition = _run_async(_fetch_tool_definition(config.tool_name, config.tool_id))
    if tool_definition is None:
        raise RuntimeError("Referenced tool is not registered.")

    executor = _get_tool_executor(config, tool_definition)
    if executor is None:
        raise RuntimeError(
            "No executor registered for tool reference. Register a tool executor on the server first."
        )

    rendered_input = _render_template_value(config.tool_input_template, state)
    if not isinstance(rendered_input, dict):
        raise RuntimeError("Rendered tool_input_template must be a JSON object.")
    tool_input = cast(dict[str, Any], rendered_input)
    try:
        result = executor(tool_input, state)
    except Exception as exc:  # pragma: no cover - defensive path around integration code.
        raise RuntimeError(f"Tool executor failed: {exc}") from exc

    return {
        "mode": "tool",
        "tool_id": str(tool_definition.id),
        "tool_name": tool_definition.name,
        "result": result,
    }


def _get_tool_executor(config: ServiceCallConfig, tool_definition: ToolDefinition) -> ToolExecutor | None:
    if config.tool_id and config.tool_id in _TOOL_EXECUTORS_BY_ID:
        return _TOOL_EXECUTORS_BY_ID[config.tool_id]
    if config.tool_name and config.tool_name in _TOOL_EXECUTORS_BY_NAME:
        return _TOOL_EXECUTORS_BY_NAME[config.tool_name]

    tool_id = str(tool_definition.id)
    if tool_id in _TOOL_EXECUTORS_BY_ID:
        return _TOOL_EXECUTORS_BY_ID[tool_id]
    if tool_definition.name in _TOOL_EXECUTORS_BY_NAME:
        return _TOOL_EXECUTORS_BY_NAME[tool_definition.name]
    return None


async def _fetch_tool_definition(tool_name: str | None, tool_id: str | None) -> ToolDefinition | None:
    async with AsyncSessionLocal() as session:
        if tool_id:
            try:
                normalized_id = uuid.UUID(tool_id.strip())
            except ValueError as exc:
                raise RuntimeError(f"Invalid tool_id '{tool_id}'.") from exc

            result = await session.execute(select(ToolDefinition).where(ToolDefinition.id == normalized_id))
            tool = result.scalar_one_or_none()
            if tool is not None:
                return tool

        if tool_name:
            result = await session.execute(select(ToolDefinition).where(ToolDefinition.name == tool_name.strip()))
            return result.scalar_one_or_none()

    return None


def _execute_user_interrupt(node: AgentNode, state: SubAgentState) -> SubAgentState:
    assert isinstance(node, UserInterruptNode)
    rendered_prompt = str(_render_template_value(node.config.prompt, state)).strip()
    if not rendered_prompt:
        raise RuntimeError(f"User interrupt node '{node.id}' rendered an empty prompt.")

    answer = interrupt(
        {
            "node_id": node.id,
            "type": "user_interrupt",
            "prompt": rendered_prompt,
            "output_key": node.config.output_key,
            "expected_type": node.config.expected_type,
            "choices": node.config.choices,
        }
    )

    next_state = _touch_iteration(state)

    parsed_data = dict(next_state.get("parsed_data", {}))
    parsed_data[node.config.output_key] = answer
    next_state["parsed_data"] = parsed_data

    messages = list(next_state.get("messages", []))
    messages.append(
        {
            "role": "assistant",
            "content": rendered_prompt,
        }
    )
    messages.append(
        {
            "role": "user",
            "content": _stringify_interrupt_answer(answer),
        }
    )
    next_state["messages"] = messages

    next_state["interrupt_payload"] = {
        "node_id": node.id,
        "prompt": rendered_prompt,
        "expected_type": node.config.expected_type,
        "choices": node.config.choices,
    }

    return next_state


def _stringify_interrupt_answer(answer: Any) -> str:
    if isinstance(answer, str):
        return answer
    if answer is None:
        return ""
    try:
        return json.dumps(answer, ensure_ascii=True)
    except TypeError:
        return str(answer)


def _execute_llm_step(node: AgentNode, state: SubAgentState) -> SubAgentState:
    assert isinstance(node, LLMStepNode)
    next_state = _touch_iteration(state)

    rendered_system_prompt = _render_llm_step_system_prompt(node.config.system_prompt, next_state)
    request_payload: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": "Complete exactly the task described by the system prompt.",
            }
        ],
        "system_prompt": rendered_system_prompt,
        "model_folder": node.config.model_type,
        "temperature": node.config.temperature,
        "max_tokens": node.config.max_tokens,
        "metadata": {
            "node": node.id,
            "component": "llm_step",
        },
    }
    if node.config.model:
        request_payload["model"] = node.config.model

    try:
        payload = _run_async(generate_text(request_payload))
    except LLMClientError as exc:
        return _handle_llm_step_failure(node=node, state=next_state, message=str(exc))

    if not isinstance(payload, dict):
        raise RuntimeError("LLM step returned an invalid payload.")

    generated_text = str(payload.get("text", "")).strip()
    if not generated_text:
        raise RuntimeError("LLM step returned empty text.")

    messages = list(next_state.get("messages", []))
    messages.append(
        {
            "role": "assistant",
            "content": generated_text,
        }
    )
    next_state["messages"] = messages

    parsed_data = dict(next_state.get("parsed_data", {}))
    if node.config.output_key:
        parsed_data[node.config.output_key] = generated_text
    _set_llm_step_status(parsed_data, node.id, "success")
    next_state["parsed_data"] = parsed_data

    return next_state


def _handle_llm_step_failure(
    node: LLMStepNode,
    state: SubAgentState,
    message: str,
) -> SubAgentState:
    if node.on_failure is None:
        raise RuntimeError(f"LLM step node '{node.id}' failed: {message}")

    parsed_data = dict(state.get("parsed_data", {}))
    llm_step_errors = parsed_data.get("__llm_step_errors__", {})
    if not isinstance(llm_step_errors, dict):
        llm_step_errors = {}
    llm_step_errors[node.id] = message
    parsed_data["__llm_step_errors__"] = llm_step_errors
    _set_llm_step_status(parsed_data, node.id, "failure")

    next_state = cast(SubAgentState, dict(state))
    next_state["parsed_data"] = parsed_data
    return next_state


def _set_llm_step_status(parsed_data: dict[str, Any], node_id: str, status: str) -> None:
    llm_step_status = parsed_data.get("__llm_step_status__", {})
    if not isinstance(llm_step_status, dict):
        llm_step_status = {}
    llm_step_status[node_id] = status
    parsed_data["__llm_step_status__"] = llm_step_status


def _render_llm_step_system_prompt(template: str, state: SubAgentState) -> str:
    pattern = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
    rendered = template
    for match in pattern.finditer(template):
        placeholder = match.group(0)
        reference = match.group(1)
        resolved = _resolve_llm_step_reference(reference, state)
        rendered = rendered.replace(placeholder, str(resolved))
    return rendered


def _resolve_llm_step_reference(reference: str, state: SubAgentState) -> Any:
    parsed_data = state.get("parsed_data", {})
    service_results = state.get("service_results", {})
    if not isinstance(parsed_data, dict):
        raise RuntimeError("parsed_data must be an object for llm_step system_prompt rendering.")
    if not isinstance(service_results, dict):
        raise RuntimeError("service_results must be an object for llm_step system_prompt rendering.")

    path = reference.strip()
    if path.startswith("state."):
        path = path[len("state.") :]

    if not (
        path == "parsed_data"
        or path.startswith("parsed_data.")
        or path == "service_results"
        or path.startswith("service_results.")
    ):
        raise RuntimeError(
            "llm_step system_prompt placeholders may only reference "
            "parsed_data.* or service_results.*"
        )

    scoped_state = cast(
        SubAgentState,
        {
            "parsed_data": parsed_data,
            "service_results": service_results,
        },
    )
    return _resolve_state_reference(scoped_state, path)


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
