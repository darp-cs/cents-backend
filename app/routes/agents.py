from __future__ import annotations

import asyncio
import json
import re
import threading
from typing import Annotated, Any, Literal, get_args, get_origin
import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.compiler import (
    clear_runtime_event_emitter,
    get_compiled_agent_graph,
    invalidate_compiled_agent_graph,
    set_runtime_event_emitter,
)
from app.agents.template_schema import (
    AgentTemplate as AgentTemplateSchema,
    ConditionNode,
    Guardrails,
    LLMStepNode,
    NODE_ID_PATTERN,
    ServiceCallNode,
    StructuredParserNode,
    TEMPLATE_VERSION_PATTERN,
    TerminalResponseNode,
    UserInterruptNode,
    validate_template,
)
from app.auth.users import current_active_user
from app.config import settings
from app.db.base import get_async_session
from app.db.models import AgentTemplate, PlatformConfig, User
from app.llm.client import LLMClientError, generate_text

router = APIRouter()
_RUN_REGISTRY_LOCK = threading.Lock()
_RUN_REGISTRY: dict[str, dict[str, Any]] = {}


class AgentTemplateCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    raw_template: dict[str, Any]

    @field_validator("name")
    @classmethod
    def trim_name(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Name cannot be empty")
        return trimmed


class AgentTemplateUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_template: dict[str, Any]


class AgentTemplateEnabledPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    version: int | None = Field(default=None, ge=1)


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = Field(min_length=1)
    version: int | None = Field(default=None, ge=1)
    thread_id: str | None = Field(default=None, min_length=1)
    parsed_data: dict[str, Any] = Field(default_factory=dict)
    service_results: dict[str, Any] = Field(default_factory=dict)
    node_llm_configs: dict[str, dict[str, str | None]] = Field(default_factory=dict)


class AgentResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: Any
    thread_id: str = Field(min_length=1)
    version: int | None = Field(default=None, ge=1)


class AgentRunStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = Field(min_length=1)
    version: int | None = Field(default=None, ge=1)
    parsed_data: dict[str, Any] = Field(default_factory=dict)
    service_results: dict[str, Any] = Field(default_factory=dict)
    node_llm_configs: dict[str, dict[str, str | None]] = Field(default_factory=dict)


class AgentRunResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: Any


class AgentAuthoringValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_template: Any


class AgentAuthoringValidationError(BaseModel):
    path: str = Field(min_length=1)
    node_id: str | None = None
    message: str = Field(min_length=1)


class AgentAuthoringValidateResponse(BaseModel):
    is_valid: bool
    normalized_template: dict[str, Any] | None = None
    errors: list[AgentAuthoringValidationError] = Field(default_factory=list)


class AgentAuthoringGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=12000)
    current_template: dict[str, Any] | None = None

    @field_validator("prompt")
    @classmethod
    def trim_prompt(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Prompt cannot be empty")
        return trimmed


class AgentAuthoringGenerateResponse(BaseModel):
    message: str = Field(min_length=1)
    generated_template: dict[str, Any] | None = None
    is_valid: bool
    errors: list[AgentAuthoringValidationError] = Field(default_factory=list)
    referenced_nodes: list[str] = Field(default_factory=list)


class AgentAuthoringNodeAssistRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_type: str = Field(min_length=1)
    instruction: str = Field(min_length=1, max_length=4000)
    current_node: dict[str, Any]
    current_template: dict[str, Any] | None = None


class AgentAuthoringNodeAssistResponse(BaseModel):
    message: str = Field(min_length=1)
    node: dict[str, Any] | None = None
    is_valid: bool
    errors: list[AgentAuthoringValidationError] = Field(default_factory=list)


class AgentAuthoringSchemaField(BaseModel):
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    required: bool
    default: Any = None
    options: list[Any] | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)


class AgentAuthoringNodeTransition(BaseModel):
    kind: Literal["next", "on_failure", "branches"]
    required: bool
    details: dict[str, Any] | None = None


class AgentAuthoringNodeTypeSchema(BaseModel):
    type: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    config_fields: list[AgentAuthoringSchemaField]
    transitions: list[AgentAuthoringNodeTransition]


class AgentAuthoringSchemaResponse(BaseModel):
    catalog_version: str = Field(min_length=1)
    template_version_pattern: str = Field(min_length=1)
    node_id_pattern: str = Field(min_length=1)
    guardrails_fields: list[AgentAuthoringSchemaField]
    node_types: list[AgentAuthoringNodeTypeSchema]


_NODE_SCHEMA_MODELS = [
    StructuredParserNode,
    ConditionNode,
    ServiceCallNode,
    UserInterruptNode,
    LLMStepNode,
    TerminalResponseNode,
]

_UNKNOWN_EDGE_PATTERN = re.compile(
    r"^Node '([^']+)' (next|on_failure) points to unknown node id '[^']+'\.$"
)
_UNKNOWN_BRANCH_PATTERN = re.compile(
    r"^Node '([^']+)' branches\['([^']+)'\] points to unknown node id '[^']+'\.$"
)
_NODE_PATH_ONLY_PATTERN = re.compile(
    r"^Node '([^']+)' (is unreachable from entry_node '[^']+'|cannot reach a 'terminal_response' node)\.$"
)
_DUPLICATE_NODE_PATTERN = re.compile(r"^Duplicate node id '([^']+)'\.$")
_ENTRY_NODE_PATTERN = re.compile(r"^entry_node '[^']+' does not match any node id\.$")
_AUTHORING_DIRECTIVES = frozenset(
    {
        "start",
        "listen",
        "if",
        "else",
        "call",
        "interrupt",
        "think",
        "reply",
        "on_failure",
        "guardrails",
    }
)


def _resolve_node_type_for_model(node_model: type[BaseModel]) -> str:
    annotation = node_model.model_fields["type"].annotation
    target = annotation
    args = [arg for arg in get_args(target) if arg is not type(None)]
    if len(args) == 1:
        target = args[0]

    type_options = list(get_args(target)) if get_origin(target) is Literal else []
    if type_options:
        return str(type_options[0])
    type_schema = node_model.model_json_schema().get("properties", {}).get("type", {})
    return str(type_schema.get("const") or node_model.__name__)


_NODE_MODEL_BY_TYPE: dict[str, type[BaseModel]] = {
    _resolve_node_type_for_model(node_model): node_model for node_model in _NODE_SCHEMA_MODELS
}


def _clear_agent_run_registry() -> None:
    with _RUN_REGISTRY_LOCK:
        _RUN_REGISTRY.clear()


def _unwrap_optional_annotation(annotation: Any) -> Any:
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1:
        return args[0]
    return annotation


def _extract_literal_options(annotation: Any) -> list[Any] | None:
    target = _unwrap_optional_annotation(annotation)
    if get_origin(target) is Literal:
        return list(get_args(target))
    return None


def _humanize_identifier(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _describe_annotation(annotation: Any) -> str:
    target = _unwrap_optional_annotation(annotation)
    origin = get_origin(target)

    if origin is Literal:
        literal_values = list(get_args(target))
        if not literal_values:
            return "string"
        value_type = type(literal_values[0])
        if value_type is bool:
            return "boolean"
        if value_type in (int, float):
            return "number"
        return "string"

    if origin in (list, tuple, set):
        return "array"
    if origin is dict:
        return "object"

    if target is str:
        return "string"
    if target is bool:
        return "boolean"
    if target in (int, float):
        return "number"
    if isinstance(target, type) and issubclass(target, BaseModel):
        return "object"

    return "unknown"


def _extract_field_constraints(field_info) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    for metadata in field_info.metadata:
        for key in ("min_length", "max_length", "ge", "gt", "le", "lt", "pattern"):
            raw_value = getattr(metadata, key, None)
            if raw_value is None:
                continue
            if key == "pattern" and hasattr(raw_value, "pattern"):
                constraints[key] = str(raw_value.pattern)
            else:
                constraints[key] = raw_value
    return constraints


def _resolve_field_default(field_info) -> Any:
    if field_info.default_factory is not None:
        try:
            return field_info.default_factory()
        except Exception:
            return None
    if field_info.is_required():
        return None
    return field_info.default


def _build_model_field_catalog(model_cls: type[BaseModel]) -> list[AgentAuthoringSchemaField]:
    fields: list[AgentAuthoringSchemaField] = []
    for field_name, field_info in model_cls.model_fields.items():
        options = _extract_literal_options(field_info.annotation)
        constraints = _extract_field_constraints(field_info)
        fields.append(
            AgentAuthoringSchemaField(
                name=field_name,
                type=_describe_annotation(field_info.annotation),
                required=field_info.is_required(),
                default=_resolve_field_default(field_info),
                options=options,
                constraints=constraints,
            )
        )
    return fields


def _build_node_transition_catalog(node_model: type[BaseModel]) -> list[AgentAuthoringNodeTransition]:
    transitions: list[AgentAuthoringNodeTransition] = []
    for transition_name in ("next", "on_failure", "branches"):
        field_info = node_model.model_fields.get(transition_name)
        if field_info is None:
            continue

        details: dict[str, Any] | None = None
        if transition_name == "branches":
            details = {"branch_key_type": "string", "target_type": "node_id"}

        transitions.append(
            AgentAuthoringNodeTransition(
                kind=transition_name,
                required=field_info.is_required(),
                details=details,
            )
        )
    return transitions


def _build_authoring_schema_catalog() -> AgentAuthoringSchemaResponse:
    node_types: list[AgentAuthoringNodeTypeSchema] = []
    for node_model in _NODE_SCHEMA_MODELS:
        type_options = _extract_literal_options(node_model.model_fields["type"].annotation) or []
        if type_options:
            node_type = str(type_options[0])
        else:
            type_schema = node_model.model_json_schema().get("properties", {}).get("type", {})
            node_type = str(type_schema.get("const") or node_model.__name__)
        config_model = node_model.model_fields["config"].annotation
        config_fields = (
            _build_model_field_catalog(config_model)
            if isinstance(config_model, type) and issubclass(config_model, BaseModel)
            else []
        )
        transitions = _build_node_transition_catalog(node_model)
        transition_names = ", ".join(transition.kind for transition in transitions) or "none"

        node_types.append(
            AgentAuthoringNodeTypeSchema(
                type=node_type,
                label=_humanize_identifier(node_type),
                description=f"{_humanize_identifier(node_type)} node. Allowed transitions: {transition_names}.",
                config_fields=config_fields,
                transitions=transitions,
            )
        )

    return AgentAuthoringSchemaResponse(
        catalog_version="1.0.0",
        template_version_pattern=TEMPLATE_VERSION_PATTERN,
        node_id_pattern=NODE_ID_PATTERN,
        guardrails_fields=_build_model_field_catalog(Guardrails),
        node_types=node_types,
    )


def _build_authoring_generation_system_prompt(current_template: dict[str, Any] | None) -> str:
    schema_catalog = _build_authoring_schema_catalog().model_dump(mode="json")
    current_template_json = json.dumps(current_template, ensure_ascii=True) if current_template else "null"
    schema_json = json.dumps(schema_catalog, ensure_ascii=True)
    return (
        "You convert a user's workflow description into a Cents AgentTemplate JSON object. "
        "Return JSON only with exactly two top-level fields: message and template. "
        "message is a short summary of the changes. template is the complete resulting template.\n\n"
        "Readable workflow source conventions:\n"
        "- Treat the document as a complete workflow definition. Preserve line order and use indentation to "
        "associate statements with the nearest directive.\n"
        "- Plain text describes the workflow title, steps, actions, and responses. Infer the smallest useful "
        "set of components; do not add unrelated example components.\n"
        "- '@start <instruction>' marks the entry step when document order alone is not sufficient.\n"
        "- '@listen <fields and source>' creates a structured_parser. Infer field names, types, required flags, "
        "and regex or LLM extraction strategy from the description.\n"
        "- '@if <condition>' creates a condition component and begins its matching branch.\n"
        "- '@else if <condition>' adds another named branch. '@else' begins the required default branch.\n"
        "- '@call <HTTP request or tool name>' creates a service_call. Use HTTP mode for a method and URL, "
        "and tool mode when the description names a registered tool. Tool mode must include tool_name or tool_id "
        "and should provide tool_input_template when inputs are described.\n"
        "- '@interrupt <request>' creates a user_interrupt component that pauses for the requested input, "
        "including output key, expected type, and choices when stated, then continues to the next step.\n"
        "- '@think <instruction>' creates an llm_step. Capture model settings and output key when stated.\n"
        "- '@reply <response>' creates a terminal_response with the requested success, failure, or cancelled status.\n"
        "- '@on_failure <instruction>' attaches an error route to the preceding parser, service call, or LLM step.\n"
        "- '@guardrails <policy>' configures maximum iterations, banned topics, and judge overrides.\n"
        "- '{{ state.<path> }}' refers to graph state. Supported runtime roots are input, parsed_data, messages, "
        "and service_results; map each reference to the subset allowed by the target node schema and do not invent fields.\n"
        "- '@node_id' refers to an existing component only when the token is not a reserved directive. "
        "Preserve and modify that component when requested.\n"
        "Legacy #last_message, #parsed_data, #service_results, /listen, /if, /call, /ask, /think, and /reply "
        "symbols may still be accepted.\n"
        "Use concise stable node ids. Every non-terminal path must reach a terminal_response. "
        "Condition nodes must include a default branch. Preserve unrelated parts of the current template.\n\n"
        f"Current template:\n{current_template_json}\n\n"
        f"Authoring schema catalog:\n{schema_json}"
    )


def _build_node_assist_system_prompt(
    *,
    node_type: str,
    current_node: dict[str, Any],
    current_template: dict[str, Any] | None,
) -> str:
    schema_catalog = _build_authoring_schema_catalog().model_dump(mode="json")
    target_schema = next((item for item in schema_catalog["node_types"] if item["type"] == node_type), None)
    current_node_json = json.dumps(current_node, ensure_ascii=True)
    current_template_json = json.dumps(current_template, ensure_ascii=True) if current_template else "null"
    target_schema_json = json.dumps(target_schema or {}, ensure_ascii=True)

    return (
        "You help edit exactly one workflow node in Cents based on a natural-language instruction. "
        "Return JSON only with exactly two top-level fields: message and node. "
        "message is one short sentence. node is the fully-populated updated node object.\n\n"
        "Rules:\n"
        "- Keep node.id unchanged.\n"
        "- Keep node.type unchanged and equal to the requested node_type.\n"
        "- Preserve routing fields unless the instruction explicitly asks to change them.\n"
        "- Favor deterministic config values that can be validated server-side.\n"
        "- For condition nodes: convert natural language into config.expression and config.input_keys. "
        "Use plain deterministic expression syntax, not prose.\n"
        "- For service_call nodes: use mode='tool' when a registered tool is requested (set tool_name/tool_id and "
        "tool_input_template), otherwise use mode='http' with method/url plus headers/body templates.\n"
        "- Do not include markdown fences or explanatory prose outside the JSON object.\n\n"
        f"Requested node_type:\n{node_type}\n\n"
        f"Current node:\n{current_node_json}\n\n"
        f"Current template (for context only):\n{current_template_json}\n\n"
        f"Target node schema:\n{target_schema_json}"
    )


def _normalize_node_type(value: str) -> str:
    return value.strip()


def _merge_assisted_node(
    current_node: dict[str, Any],
    assisted_node: dict[str, Any],
    expected_node_type: str,
) -> dict[str, Any]:
    merged = dict(current_node)

    if "description" in assisted_node:
        merged["description"] = assisted_node.get("description")

    if isinstance(assisted_node.get("config"), dict):
        merged["config"] = assisted_node["config"]

    for transition_field in ("next", "on_failure", "branches"):
        if transition_field in assisted_node:
            merged[transition_field] = assisted_node[transition_field]

    merged["id"] = str(current_node.get("id", "")).strip()
    merged["type"] = expected_node_type
    return merged


def _validate_assisted_node(
    *,
    node_type: str,
    node_payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[AgentAuthoringValidationError]]:
    node_model = _NODE_MODEL_BY_TYPE.get(node_type)
    if node_model is None:
        return None, [
            AgentAuthoringValidationError(
                path="node.type",
                node_id=None,
                message=f"Unsupported node_type '{node_type}'.",
            )
        ]

    try:
        normalized = node_model.model_validate(node_payload).model_dump(mode="json")
        return normalized, []
    except ValidationError as exc:
        node_id = str(node_payload.get("id", "")).strip() or None
        errors = [
            AgentAuthoringValidationError(
                path="node." + ".".join(str(part) for part in issue["loc"]),
                node_id=node_id,
                message=issue["msg"],
            )
            for issue in exc.errors()
        ]
        return None, errors


def _extract_authoring_generation_payload(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, count=1, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate, count=1)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        object_start = candidate.find("{")
        if object_start < 0:
            raise ValueError("LLM response did not contain a JSON object.") from None
        try:
            parsed, _ = json.JSONDecoder().raw_decode(candidate[object_start:])
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response contained invalid JSON.") from exc

    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object.")
    return parsed


def _build_node_index_lookup(raw_template: Any) -> dict[int, str]:
    if not isinstance(raw_template, dict):
        return {}

    raw_nodes = raw_template.get("nodes")
    if not isinstance(raw_nodes, list):
        return {}

    lookup: dict[int, str] = {}
    for index, node in enumerate(raw_nodes):
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if isinstance(node_id, str) and node_id.strip():
            lookup[index] = node_id.strip()
    return lookup


def _extract_node_id_from_path(path: str, index_lookup: dict[int, str]) -> str | None:
    segments = path.split(".")
    if len(segments) < 2 or segments[0] != "nodes":
        return None

    try:
        node_index = int(segments[1])
    except ValueError:
        return segments[1] if segments[1] else None

    return index_lookup.get(node_index)


def _map_error_to_structured(
    error_text: str,
    node_index_lookup: dict[int, str],
) -> AgentAuthoringValidationError:
    if ": " in error_text and not error_text.startswith("Node '"):
        raw_path, message = error_text.split(": ", 1)
        return AgentAuthoringValidationError(
            path=raw_path,
            node_id=_extract_node_id_from_path(raw_path, node_index_lookup),
            message=message,
        )

    unknown_edge_match = _UNKNOWN_EDGE_PATTERN.match(error_text)
    if unknown_edge_match is not None:
        node_id = unknown_edge_match.group(1)
        transition = unknown_edge_match.group(2)
        return AgentAuthoringValidationError(path=f"nodes.{node_id}.{transition}", node_id=node_id, message=error_text)

    unknown_branch_match = _UNKNOWN_BRANCH_PATTERN.match(error_text)
    if unknown_branch_match is not None:
        node_id = unknown_branch_match.group(1)
        branch_name = unknown_branch_match.group(2)
        return AgentAuthoringValidationError(
            path=f"nodes.{node_id}.branches.{branch_name}",
            node_id=node_id,
            message=error_text,
        )

    node_only_match = _NODE_PATH_ONLY_PATTERN.match(error_text)
    if node_only_match is not None:
        node_id = node_only_match.group(1)
        return AgentAuthoringValidationError(path=f"nodes.{node_id}", node_id=node_id, message=error_text)

    duplicate_node_match = _DUPLICATE_NODE_PATTERN.match(error_text)
    if duplicate_node_match is not None:
        node_id = duplicate_node_match.group(1)
        return AgentAuthoringValidationError(path=f"nodes.{node_id}.id", node_id=node_id, message=error_text)

    if _ENTRY_NODE_PATTERN.match(error_text) is not None:
        return AgentAuthoringValidationError(path="entry_node", node_id=None, message=error_text)

    if error_text.startswith("Template "):
        return AgentAuthoringValidationError(path="template", node_id=None, message=error_text)

    return AgentAuthoringValidationError(path="template", node_id=None, message=error_text)


def _normalize_template_if_parseable(raw_template: Any, parsed_result) -> dict[str, Any] | None:
    if parsed_result.template is not None:
        return parsed_result.template.model_dump(mode="json")

    if not isinstance(raw_template, dict):
        return None

    try:
        return AgentTemplateSchema.model_validate(raw_template).model_dump(mode="json")
    except ValidationError:
        return None


def _build_structured_validation_errors(
    raw_template: Any,
    errors: list[str],
) -> list[AgentAuthoringValidationError]:
    node_index_lookup = _build_node_index_lookup(raw_template)
    return [_map_error_to_structured(error_text, node_index_lookup) for error_text in errors]


def _coerce_iteration_count(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed >= 0 else 0


def _register_agent_run(
    *,
    run_id: str,
    thread_id: str,
    user_id: str,
    agent_name: str,
    version: int,
) -> None:
    with _RUN_REGISTRY_LOCK:
        _RUN_REGISTRY[run_id] = {
            "run_id": run_id,
            "thread_id": thread_id,
            "user_id": user_id,
            "agent_name": agent_name,
            "version": version,
            "status": "running",
            "iteration_count": 0,
            "final_response": "",
            "error": None,
        }


def _update_agent_run(run_id: str, **changes: Any) -> None:
    with _RUN_REGISTRY_LOCK:
        record = _RUN_REGISTRY.get(run_id)
        if record is None:
            return
        record.update(changes)


def _get_agent_run_for_user(run_id: str, user_id: str) -> dict[str, Any] | None:
    with _RUN_REGISTRY_LOCK:
        record = _RUN_REGISTRY.get(run_id)
        if record is None:
            return None
        if record.get("user_id") != user_id:
            return None
        return dict(record)


def _build_run_state(
    *,
    payload: AgentRunStartRequest,
    platform_guardrails: dict[str, Any],
    template_guardrails: dict[str, Any],
) -> dict[str, Any]:
    return {
        "input": payload.input,
        "parsed_data": dict(payload.parsed_data),
        "messages": [{"role": "user", "content": payload.input}],
        "iteration_count": 0,
        "service_results": dict(payload.service_results),
        "interrupt_payload": None,
        "final_response": "",
        "node_llm_configs": dict(payload.node_llm_configs),
        "platform_guardrails": platform_guardrails,
        "template_guardrails": template_guardrails,
    }


def _format_sse_data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"


def _queue_event(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[dict[str, Any]], payload: dict[str, Any]) -> None:
    loop.call_soon_threadsafe(queue.put_nowait, payload)


def _build_run_id(user_id: str) -> str:
    return f"{user_id}:{uuid.uuid4()}"


async def _build_run_streaming_response(
    *,
    run_id: str,
    thread_id: str,
    compiled_graph,
    invoke_input: dict[str, Any] | Command,
    config: dict[str, Any],
) -> StreamingResponse:
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    latest_node_id: dict[str, str | None] = {"value": None}

    def _emit(event: dict[str, Any]) -> None:
        event_name = str(event.get("event", "")).strip().lower()
        if event_name == "node_started":
            node_id = event.get("node_id")
            latest_node_id["value"] = str(node_id) if node_id is not None else None
        if event_name == "node_completed":
            _update_agent_run(
                run_id,
                iteration_count=_coerce_iteration_count(event.get("iteration_count", 0)),
            )

        payload = {
            "run_id": run_id,
            "thread_id": thread_id,
            **event,
        }
        _queue_event(loop, queue, payload)

    def _invoke_graph() -> dict[str, Any]:
        set_runtime_event_emitter(_emit)
        try:
            return compiled_graph.invoke(invoke_input, config)
        finally:
            clear_runtime_event_emitter()

    async def _runner() -> None:
        try:
            result = await asyncio.to_thread(_invoke_graph)
        except Exception as exc:
            _update_agent_run(run_id, status="failed", error=str(exc))
            _queue_event(
                loop,
                queue,
                {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "event": "error",
                    "node_id": latest_node_id["value"],
                    "message": str(exc),
                },
            )
            _queue_event(
                loop,
                queue,
                {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "event": "done",
                    "status": "failed",
                },
            )
            return

        iteration_count = _coerce_iteration_count(result.get("iteration_count", 0))
        _update_agent_run(run_id, iteration_count=iteration_count)

        interrupt_payload = _extract_first_interrupt_payload(compiled_graph, config)
        if interrupt_payload is not None:
            _update_agent_run(run_id, status="awaiting_input")
            _queue_event(
                loop,
                queue,
                {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "event": "interrupt_requested",
                    "node_id": interrupt_payload.get("node_id"),
                    "prompt": str(interrupt_payload.get("prompt", "")),
                },
            )
            _queue_event(
                loop,
                queue,
                {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "event": "done",
                    "awaiting_input": True,
                },
            )
            return

        error_event = result.get("error_event")
        if isinstance(error_event, dict):
            error_message = str(error_event.get("message", error_event.get("type", "execution failed")))
            _update_agent_run(run_id, status="failed", error=error_message)
            _queue_event(
                loop,
                queue,
                {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "event": "error",
                    "node_id": error_event.get("node_id"),
                    "message": error_message,
                },
            )
            _queue_event(
                loop,
                queue,
                {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "event": "done",
                    "status": "failed",
                },
            )
            return

        final_response = str(result.get("final_response", ""))
        _update_agent_run(run_id, status="succeeded", final_response=final_response)
        _queue_event(
            loop,
            queue,
            {
                "run_id": run_id,
                "thread_id": thread_id,
                "event": "done",
                "final_response": final_response,
            },
        )

    worker = asyncio.create_task(_runner())

    async def _event_stream():
        while True:
            payload = await queue.get()
            yield _format_sse_data(payload)
            if payload.get("event") == "done":
                break
        if not worker.done():
            await worker

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


def _parse_json(value: str | None) -> Any:
    if value is None:
        return None

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _serialize_record(record: AgentTemplate) -> dict[str, Any]:
    return {
        "id": str(record.id),
        "name": record.name,
        "version": record.version,
        "raw_template": _parse_json(record.raw_template),
        "is_valid": record.is_valid,
        "validation_errors": _parse_json(record.validation_errors),
        "enabled": record.enabled,
        "created_at": record.created_at.isoformat(),
    }


def _should_auto_enable_new_version(existing_enabled_count: int, is_valid: bool) -> bool:
    return existing_enabled_count == 0 and is_valid


def _apply_enabled_policy(
    records: list[AgentTemplate],
    target: AgentTemplate,
    desired_enabled: bool,
) -> None:
    if desired_enabled:
        if not target.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid templates cannot be enabled.",
            )

        # Enabling one version must atomically disable all other versions for that agent name.
        for record in records:
            record.enabled = record.version == target.version
        return

    currently_enabled = [record for record in records if record.enabled]
    if target.enabled and len(currently_enabled) <= 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one version must remain enabled for this agent.",
        )

    target.enabled = False


async def _get_latest_for_name(session: AsyncSession, name: str) -> AgentTemplate | None:
    result = await session.execute(
        select(AgentTemplate)
        .where(AgentTemplate.name == name)
        .order_by(AgentTemplate.version.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _get_versions_for_name(session: AsyncSession, name: str) -> list[AgentTemplate]:
    result = await session.execute(
        select(AgentTemplate)
        .where(AgentTemplate.name == name)
        .order_by(AgentTemplate.version.desc())
    )
    return list(result.scalars().all())


def _extract_first_interrupt_payload(compiled_graph, config: dict[str, Any]) -> dict[str, Any] | None:
    try:
        snapshot = compiled_graph.get_state(config)
    except ValueError:
        return None

    for task in getattr(snapshot, "tasks", ()):
        interrupts = getattr(task, "interrupts", ())
        if not interrupts:
            continue
        raw_value = interrupts[0].value
        return raw_value if isinstance(raw_value, dict) else {"value": raw_value}
    return None


def _deserialize_banned_topics(raw: str | None) -> list[str]:
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    return [str(item).strip() for item in parsed if str(item).strip()]


async def _get_or_create_platform_config(session: AsyncSession) -> PlatformConfig:
    result = await session.execute(select(PlatformConfig).order_by(PlatformConfig.id.asc()).limit(1))
    record = result.scalar_one_or_none()
    if record is not None:
        return record

    record = PlatformConfig(
        id=1,
        guidelines_text="",
        banned_topics="[]",
        judge_enabled=settings.llm_judge_enabled,
        max_retries=3,
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return record


async def _build_platform_guardrails(session: AsyncSession) -> dict[str, Any]:
    record = await _get_or_create_platform_config(session)
    return {
        "guidelines_text": record.guidelines_text,
        "banned_topics": _deserialize_banned_topics(record.banned_topics),
        "judge_enabled": record.judge_enabled,
        "max_iterations": record.max_retries,
    }


def _build_template_guardrails(raw_template: dict[str, Any]) -> dict[str, Any]:
    raw_guardrails = raw_template.get("guardrails")
    if not isinstance(raw_guardrails, dict):
        return {
            "max_iterations": None,
            "banned_topics_override": None,
            "judge_enabled_override": None,
        }

    template_max_iterations = raw_guardrails.get("max_iterations")
    if isinstance(template_max_iterations, bool):
        template_max_iterations = None
    elif template_max_iterations is not None:
        try:
            template_max_iterations = int(template_max_iterations)
        except (TypeError, ValueError):
            template_max_iterations = None

    raw_banned_topics_override = raw_guardrails.get("banned_topics_override")
    banned_topics_override = (
        [str(topic).strip() for topic in raw_banned_topics_override if str(topic).strip()]
        if isinstance(raw_banned_topics_override, list)
        else None
    )

    judge_enabled_override = raw_guardrails.get("judge_enabled_override")
    if judge_enabled_override is not None:
        judge_enabled_override = bool(judge_enabled_override)

    return {
        "max_iterations": template_max_iterations,
        "banned_topics_override": banned_topics_override,
        "judge_enabled_override": judge_enabled_override,
    }


async def _resolve_runnable_template(
    session: AsyncSession,
    *,
    name: str,
    requested_version: int | None,
) -> tuple[AgentTemplate, dict[str, Any]]:
    records = await _get_versions_for_name(session, name)
    if not records:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent template not found")

    if requested_version is None:
        record = next((item for item in records if item.enabled), None)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No enabled version is available for this agent.",
            )
    else:
        record = next((item for item in records if item.version == requested_version), None)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent template not found")
        if not record.enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Requested version is disabled.",
            )

    if not record.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot run an invalid template version.",
        )

    raw_template = _parse_json(record.raw_template)
    if not isinstance(raw_template, dict):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored template payload is invalid.",
        )

    parsed = validate_template(raw_template)
    if parsed.template is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored template failed validation.",
        )

    return record, raw_template


async def _create_version(
    session: AsyncSession,
    name: str,
    raw_template: dict[str, Any],
) -> AgentTemplate:
    result = validate_template(raw_template)
    validation_errors = result.errors if result.errors else None

    max_version_result = await session.execute(
        select(func.max(AgentTemplate.version)).where(AgentTemplate.name == name)
    )
    max_version = max_version_result.scalar_one()
    next_version = 1 if max_version is None else int(max_version) + 1

    enabled_count_result = await session.execute(
        select(func.count())
        .select_from(AgentTemplate)
        .where(AgentTemplate.name == name, AgentTemplate.enabled.is_(True))
    )
    enabled_count = int(enabled_count_result.scalar_one())

    record = AgentTemplate(
        name=name,
        version=next_version,
        raw_template=json.dumps(raw_template, ensure_ascii=True, separators=(",", ":")),
        is_valid=result.is_valid,
        validation_errors=(
            json.dumps(validation_errors, ensure_ascii=True, separators=(",", ":"))
            if validation_errors
            else None
        ),
        enabled=_should_auto_enable_new_version(enabled_count, result.is_valid),
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)

    if result.template is not None:
        # Warm cache so this version does not need recompilation at first execution.
        get_compiled_agent_graph(name=name, version=next_version, template=result.template)

    return record


@router.get("/authoring/schema", response_model=AgentAuthoringSchemaResponse)
async def get_authoring_schema(
    user: Annotated[User, Depends(current_active_user)],
):
    del user
    return _build_authoring_schema_catalog()


@router.post("/authoring/generate", response_model=AgentAuthoringGenerateResponse)
async def generate_authoring_template(
    payload: AgentAuthoringGenerateRequest,
    user: Annotated[User, Depends(current_active_user)],
):
    referenced_nodes = sorted(
        {
            reference
            for reference in re.findall(r"@([A-Za-z][A-Za-z0-9_-]*)", payload.prompt)
            if reference.lower() not in _AUTHORING_DIRECTIVES
        }
    )
    request_payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": payload.prompt}],
        "system_prompt": _build_authoring_generation_system_prompt(payload.current_template),
        "model_folder": settings.llm_default_generation_model_type,
        "temperature": 0.1,
        "max_tokens": 4096,
        "metadata": {
            "user_id": str(user.id),
            "node": "agent_authoring",
        },
    }
    if settings.llm_default_generation_model.strip():
        request_payload["model"] = settings.llm_default_generation_model.strip()

    try:
        llm_response = await generate_text(request_payload)
    except LLMClientError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    generated_text = str(llm_response.get("text", "")).strip()
    if not generated_text:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM service returned an empty response.")

    try:
        generated_payload = _extract_authoring_generation_payload(generated_text)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    generated_template = generated_payload.get("template")
    if not isinstance(generated_template, dict):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM response is missing a template object.",
        )

    validation_result = validate_template(generated_template)  # type: ignore[arg-type]
    normalized_template = _normalize_template_if_parseable(generated_template, validation_result)
    structured_errors = _build_structured_validation_errors(generated_template, validation_result.errors)
    raw_message = generated_payload.get("message")
    message = str(raw_message).strip() if raw_message is not None else ""
    if not message:
        message = "Generated a workflow draft from your description."

    return AgentAuthoringGenerateResponse(
        message=message,
        generated_template=normalized_template or generated_template,
        is_valid=validation_result.is_valid,
        errors=structured_errors,
        referenced_nodes=referenced_nodes,
    )


@router.post("/authoring/node-assist", response_model=AgentAuthoringNodeAssistResponse)
async def assist_authoring_node(
    payload: AgentAuthoringNodeAssistRequest,
    user: Annotated[User, Depends(current_active_user)],
):
    normalized_node_type = _normalize_node_type(payload.node_type)
    current_node_type = _normalize_node_type(str(payload.current_node.get("type", "")))
    current_node_id = str(payload.current_node.get("id", "")).strip() or None

    if normalized_node_type not in _NODE_MODEL_BY_TYPE:
        return AgentAuthoringNodeAssistResponse(
            message="Node assist could not be generated.",
            node=None,
            is_valid=False,
            errors=[
                AgentAuthoringValidationError(
                    path="node_type",
                    node_id=current_node_id,
                    message=f"Unsupported node_type '{normalized_node_type}'.",
                )
            ],
        )

    if current_node_type != normalized_node_type:
        return AgentAuthoringNodeAssistResponse(
            message="Node assist could not be generated.",
            node=None,
            is_valid=False,
            errors=[
                AgentAuthoringValidationError(
                    path="current_node.type",
                    node_id=current_node_id,
                    message="current_node.type must match node_type.",
                )
            ],
        )

    request_payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": payload.instruction}],
        "system_prompt": _build_node_assist_system_prompt(
            node_type=normalized_node_type,
            current_node=payload.current_node,
            current_template=payload.current_template,
        ),
        "model_folder": settings.llm_default_generation_model_type,
        "temperature": 0.1,
        "max_tokens": 2048,
        "metadata": {
            "user_id": str(user.id),
            "node": "agent_authoring_node_assist",
            "node_type": normalized_node_type,
        },
    }
    if settings.llm_default_generation_model.strip():
        request_payload["model"] = settings.llm_default_generation_model.strip()

    try:
        llm_response = await generate_text(request_payload)
    except LLMClientError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    generated_text = str(llm_response.get("text", "")).strip()
    if not generated_text:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM service returned an empty response.")

    try:
        generated_payload = _extract_authoring_generation_payload(generated_text)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    assisted_node = generated_payload.get("node")
    if not isinstance(assisted_node, dict):
        return AgentAuthoringNodeAssistResponse(
            message="Node assist could not be generated.",
            node=None,
            is_valid=False,
            errors=[
                AgentAuthoringValidationError(
                    path="node",
                    node_id=current_node_id,
                    message="LLM response is missing a node object.",
                )
            ],
        )

    merged_node = _merge_assisted_node(payload.current_node, assisted_node, normalized_node_type)
    normalized_node, errors = _validate_assisted_node(node_type=normalized_node_type, node_payload=merged_node)

    raw_message = generated_payload.get("message")
    message = str(raw_message).strip() if raw_message is not None else ""
    if not message:
        message = "Generated node draft from natural language."

    return AgentAuthoringNodeAssistResponse(
        message=message,
        node=normalized_node,
        is_valid=normalized_node is not None,
        errors=errors,
    )


@router.post("/authoring/validate", response_model=AgentAuthoringValidateResponse)
async def validate_authoring_template(
    payload: AgentAuthoringValidateRequest,
    user: Annotated[User, Depends(current_active_user)],
):
    del user

    validation_result = validate_template(payload.raw_template)  # type: ignore[arg-type]
    normalized_template = _normalize_template_if_parseable(payload.raw_template, validation_result)
    structured_errors = _build_structured_validation_errors(payload.raw_template, validation_result.errors)

    return AgentAuthoringValidateResponse(
        is_valid=validation_result.is_valid,
        normalized_template=normalized_template,
        errors=structured_errors,
    )


@router.post("", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_agent_template(
    payload: AgentTemplateCreateRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    latest = await _get_latest_for_name(session, payload.name)
    if latest is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent '{payload.name}' already exists. Use PUT /agents/{payload.name} to create a new version.",
        )

    record = await _create_version(session, payload.name, payload.raw_template)
    return _serialize_record(record)


@router.get("", response_model=list[dict])
async def list_latest_agents(
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    # Subquery ensures one record (the max version) is returned for each agent name.
    latest_per_name = (
        select(
            AgentTemplate.name.label("name"),
            func.max(AgentTemplate.version).label("max_version"),
        )
        .group_by(AgentTemplate.name)
        .subquery()
    )

    result = await session.execute(
        select(AgentTemplate)
        .join(
            latest_per_name,
            (AgentTemplate.name == latest_per_name.c.name)
            & (AgentTemplate.version == latest_per_name.c.max_version),
        )
        .order_by(AgentTemplate.name.asc())
    )

    return [_serialize_record(record) for record in result.scalars().all()]


@router.get("/{name}/versions", response_model=list[dict])
async def list_agent_versions(
    name: str,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    normalized_name = name.strip()
    records = await _get_versions_for_name(session, normalized_name)
    if not records:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent template not found")

    return [_serialize_record(record) for record in records]


@router.get("/{name}", response_model=dict)
async def get_latest_agent(
    name: str,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    latest = await _get_latest_for_name(session, name.strip())
    if latest is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent template not found")

    return _serialize_record(latest)


@router.post("/{name}/runs")
async def start_agent_run(
    name: str,
    payload: AgentRunStartRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    normalized_name = name.strip()
    record, raw_template = await _resolve_runnable_template(
        session,
        name=normalized_name,
        requested_version=payload.version,
    )
    parsed = validate_template(raw_template)
    if parsed.template is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored template failed validation.",
        )

    user_id = str(user.id)
    run_id = _build_run_id(user_id)
    thread_id = run_id

    _register_agent_run(
        run_id=run_id,
        thread_id=thread_id,
        user_id=user_id,
        agent_name=normalized_name,
        version=record.version,
    )

    platform_guardrails = await _build_platform_guardrails(session)
    template_guardrails = _build_template_guardrails(raw_template)
    state = _build_run_state(
        payload=payload,
        platform_guardrails=platform_guardrails,
        template_guardrails=template_guardrails,
    )

    config = {"configurable": {"thread_id": thread_id}}
    compiled_graph = get_compiled_agent_graph(
        name=normalized_name,
        version=record.version,
        template=parsed.template,
    )

    return await _build_run_streaming_response(
        run_id=run_id,
        thread_id=thread_id,
        compiled_graph=compiled_graph,
        invoke_input=state,
        config=config,
    )


@router.post("/runs/{run_id}/resume")
async def resume_agent_run(
    run_id: str,
    payload: AgentRunResumeRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    user_id = str(user.id)
    run_record = _get_agent_run_for_user(run_id, user_id)
    if run_record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    if run_record.get("status") != "awaiting_input":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Run is not awaiting input.",
        )

    agent_name = str(run_record.get("agent_name", "")).strip()
    requested_version = _coerce_iteration_count(run_record.get("version", 0))
    if not agent_name or requested_version < 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Run metadata is invalid.",
        )

    record, raw_template = await _resolve_runnable_template(
        session,
        name=agent_name,
        requested_version=requested_version,
    )
    parsed = validate_template(raw_template)
    if parsed.template is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored template failed validation.",
        )

    thread_id = str(run_record.get("thread_id", "")).strip()
    if not thread_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Run thread_id is missing.",
        )

    _update_agent_run(run_id, status="running", error=None, final_response="")

    config = {"configurable": {"thread_id": thread_id}}
    compiled_graph = get_compiled_agent_graph(
        name=agent_name,
        version=record.version,
        template=parsed.template,
    )

    return await _build_run_streaming_response(
        run_id=run_id,
        thread_id=thread_id,
        compiled_graph=compiled_graph,
        invoke_input=Command(resume=payload.answer),
        config=config,
    )


@router.get("/runs/{run_id}", response_model=dict)
async def get_agent_run(
    run_id: str,
    user: Annotated[User, Depends(current_active_user)],
):
    user_id = str(user.id)
    run_record = _get_agent_run_for_user(run_id, user_id)
    if run_record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    return {
        "run_id": run_record["run_id"],
        "thread_id": run_record["thread_id"],
        "status": run_record["status"],
        "iteration_count": _coerce_iteration_count(run_record.get("iteration_count", 0)),
    }


@router.post("/{name}/run", response_model=dict)
async def run_agent(
    name: str,
    payload: AgentRunRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    normalized_name = name.strip()
    record, raw_template = await _resolve_runnable_template(
        session,
        name=normalized_name,
        requested_version=payload.version,
    )
    parsed = validate_template(raw_template)
    if parsed.template is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored template failed validation.",
        )

    thread_id = payload.thread_id.strip() if payload.thread_id else (
        f"user:{user.id}:agent:{normalized_name}:{uuid.uuid4()}"
    )
    platform_guardrails = await _build_platform_guardrails(session)
    template_guardrails = _build_template_guardrails(raw_template)

    state = {
        "input": payload.input,
        "parsed_data": dict(payload.parsed_data),
        "messages": [{"role": "user", "content": payload.input}],
        "iteration_count": 0,
        "service_results": dict(payload.service_results),
        "interrupt_payload": None,
        "final_response": "",
        "node_llm_configs": dict(payload.node_llm_configs),
        "platform_guardrails": platform_guardrails,
        "template_guardrails": template_guardrails,
    }
    config = {"configurable": {"thread_id": thread_id}}

    compiled_graph = get_compiled_agent_graph(
        name=normalized_name,
        version=record.version,
        template=parsed.template,
    )

    result = await asyncio.to_thread(compiled_graph.invoke, state, config)
    interrupt_payload = _extract_first_interrupt_payload(compiled_graph, config)
    if interrupt_payload is not None:
        return {
            "status": "interrupted",
            "thread_id": thread_id,
            "version": record.version,
            "interrupt": interrupt_payload,
            "state": result,
        }

    error_event = result.get("error_event")
    if isinstance(error_event, dict):
        return {
            "status": "failed",
            "thread_id": thread_id,
            "version": record.version,
            "error": error_event,
            "state": result,
        }

    return {
        "status": "completed",
        "thread_id": thread_id,
        "version": record.version,
        "state": result,
        "final_response": str(result.get("final_response", "")),
    }


@router.post("/{name}/resume", response_model=dict)
async def resume_agent(
    name: str,
    payload: AgentResumeRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    normalized_name = name.strip()
    record, raw_template = await _resolve_runnable_template(
        session,
        name=normalized_name,
        requested_version=payload.version,
    )
    parsed = validate_template(raw_template)
    if parsed.template is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored template failed validation.",
        )

    config = {"configurable": {"thread_id": payload.thread_id.strip()}}
    compiled_graph = get_compiled_agent_graph(
        name=normalized_name,
        version=record.version,
        template=parsed.template,
    )

    result = await asyncio.to_thread(compiled_graph.invoke, Command(resume=payload.answer), config)
    interrupt_payload = _extract_first_interrupt_payload(compiled_graph, config)
    if interrupt_payload is not None:
        return {
            "status": "interrupted",
            "thread_id": payload.thread_id.strip(),
            "version": record.version,
            "interrupt": interrupt_payload,
            "state": result,
        }

    error_event = result.get("error_event")
    if isinstance(error_event, dict):
        return {
            "status": "failed",
            "thread_id": payload.thread_id.strip(),
            "version": record.version,
            "error": error_event,
            "state": result,
        }

    return {
        "status": "completed",
        "thread_id": payload.thread_id.strip(),
        "version": record.version,
        "state": result,
        "final_response": str(result.get("final_response", "")),
    }


@router.put("/{name}", response_model=dict)
async def create_new_agent_version(
    name: str,
    payload: AgentTemplateUpdateRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    normalized_name = name.strip()
    latest = await _get_latest_for_name(session, normalized_name)
    if latest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{normalized_name}' not found. Use POST /agents to create version 1.",
        )

    record = await _create_version(session, normalized_name, payload.raw_template)
    return _serialize_record(record)


@router.patch("/{name}/enabled", response_model=dict)
async def set_agent_enabled(
    name: str,
    payload: AgentTemplateEnabledPatchRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    normalized_name = name.strip()

    records = await _get_versions_for_name(session, normalized_name)
    if not records:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent template not found")

    if payload.version is None:
        record = records[0]
    else:
        record = next((item for item in records if item.version == payload.version), None)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent template not found")

    _apply_enabled_policy(records, record, payload.enabled)
    await session.commit()
    await session.refresh(record)
    return _serialize_record(record)


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    name: str,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_async_session),
):
    del user

    normalized_name = name.strip()
    latest = await _get_latest_for_name(session, normalized_name)
    if latest is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent template not found")

    await session.execute(delete(AgentTemplate).where(AgentTemplate.name == normalized_name))
    await session.commit()
    invalidate_compiled_agent_graph(normalized_name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
