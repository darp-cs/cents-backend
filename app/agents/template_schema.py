from __future__ import annotations

from collections import deque
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

NODE_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]*$"
TEMPLATE_VERSION_PATTERN = r"^\d+\.\d+(\.\d+)?$"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParsedField(_StrictModel):
    name: str = Field(min_length=1)
    type: Literal["string", "number", "boolean", "date", "enum", "array"] = "string"
    required: bool = True
    enum_values: list[str] | None = None
    description: str | None = None


class StructuredParserConfig(_StrictModel):
    source_key: str = Field(default="last_message", min_length=1)
    strategy: Literal["regex", "llm"] = "regex"
    regex_patterns: dict[str, str] = Field(default_factory=dict)
    llm_prompt_instructions: str | None = None
    llm_model_type: str | None = None
    llm_model: str | None = None
    llm_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    llm_max_tokens: int | None = Field(default=None, ge=1, le=4096)
    fields: list[ParsedField] = Field(min_length=1)
    strict: bool = False


class ConditionConfig(_StrictModel):
    expression: str = Field(min_length=1)
    input_keys: list[str] = Field(default_factory=list)


class ServiceCallConfig(_StrictModel):
    mode: Literal["http", "tool"] = "http"
    url: str | None = None
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET"
    headers_template: dict[str, str] = Field(default_factory=dict)
    body_template: dict[str, Any] | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    tool_input_template: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int | None = Field(default=None, ge=1, le=600)
    allow_unsafe_destination: bool = False

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> ServiceCallConfig:
        if self.mode == "http":
            if not self.url or not self.url.strip():
                raise ValueError("service_call mode='http' requires a non-empty url.")
            if self.tool_name or self.tool_id:
                raise ValueError("service_call mode='http' cannot include tool_name or tool_id.")
            return self

        if not self.tool_name and not self.tool_id:
            raise ValueError("service_call mode='tool' requires tool_name or tool_id.")
        if self.url:
            raise ValueError("service_call mode='tool' cannot include url.")
        return self


class UserInterruptConfig(_StrictModel):
    prompt: str = Field(min_length=1)
    output_key: str = Field(min_length=1)
    expected_type: Literal["text", "choice", "confirmation"] = "text"
    choices: list[str] | None = None


class LLMStepConfig(_StrictModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    prompt_template: str = Field(min_length=1)
    output_key: str = Field(min_length=1)
    system_prompt: str | None = None
    model_type: str = "text-generation"
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)


class TerminalResponseConfig(_StrictModel):
    template: str = Field(min_length=1)
    status: Literal["success", "failure", "cancelled"] = "success"
    include_state_keys: list[str] = Field(default_factory=list)


class _BaseNode(_StrictModel):
    id: str = Field(pattern=NODE_ID_PATTERN)
    description: str | None = None


class StructuredParserNode(_BaseNode):
    type: Literal["structured_parser"]
    config: StructuredParserConfig
    next: str = Field(pattern=NODE_ID_PATTERN)
    on_failure: str | None = Field(default=None, pattern=NODE_ID_PATTERN)


class ConditionNode(_BaseNode):
    type: Literal["condition"]
    config: ConditionConfig
    branches: dict[str, str] = Field(min_length=1)

    @field_validator("branches")
    @classmethod
    def ensure_default_branch(cls, value: dict[str, str]) -> dict[str, str]:
        if "default" not in value:
            raise ValueError("Condition nodes must define a 'default' branch.")
        return value


class ServiceCallNode(_BaseNode):
    type: Literal["service_call"]
    config: ServiceCallConfig
    next: str = Field(pattern=NODE_ID_PATTERN)
    on_failure: str | None = Field(default=None, pattern=NODE_ID_PATTERN)


class UserInterruptNode(_BaseNode):
    type: Literal["user_interrupt"]
    config: UserInterruptConfig
    next: str = Field(pattern=NODE_ID_PATTERN)


class LLMStepNode(_BaseNode):
    type: Literal["llm_step"]
    config: LLMStepConfig
    next: str = Field(pattern=NODE_ID_PATTERN)


class TerminalResponseNode(_BaseNode):
    type: Literal["terminal_response"]
    config: TerminalResponseConfig


AgentNode = Annotated[
    Union[
        StructuredParserNode,
        ConditionNode,
        ServiceCallNode,
        UserInterruptNode,
        LLMStepNode,
        TerminalResponseNode,
    ],
    Field(discriminator="type"),
]


class Guardrails(_StrictModel):
    max_iterations: int = Field(default=3, ge=1, le=50)
    banned_topics_override: list[str] | None = None
    judge_enabled_override: bool | None = None


class AgentTemplate(_StrictModel):
    template_version: str = Field(pattern=TEMPLATE_VERSION_PATTERN)
    entry_node: str = Field(pattern=NODE_ID_PATTERN)
    nodes: list[AgentNode] = Field(min_length=1)
    guardrails: Guardrails = Field(default_factory=Guardrails)


class AgentTemplateValidationResult(BaseModel):
    template: AgentTemplate | None = None
    errors: list[str] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.template is not None and not self.errors


def validate_template(raw_json: dict) -> AgentTemplateValidationResult:
    if not isinstance(raw_json, dict):
        return AgentTemplateValidationResult(errors=["Template must be a JSON object."])

    try:
        template = AgentTemplate.model_validate(raw_json)
    except ValidationError as exc:
        return AgentTemplateValidationResult(errors=_format_schema_errors(exc))

    errors = _validate_graph(template)
    if errors:
        return AgentTemplateValidationResult(errors=errors)
    return AgentTemplateValidationResult(template=template)


def _format_schema_errors(exc: ValidationError) -> list[str]:
    errors: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "template"
        errors.append(f"{location}: {error['msg']}")
    return errors


def _outgoing_edges(node: AgentNode) -> list[tuple[str, str]]:
    """Returns (human-readable edge label, target node id) pairs for a node."""
    if isinstance(node, ConditionNode):
        return [(f"branches['{key}']", target) for key, target in node.branches.items()]
    if isinstance(node, TerminalResponseNode):
        return []
    if isinstance(node, ServiceCallNode) and node.on_failure:
        return [("next", node.next), ("on_failure", node.on_failure)]
    if isinstance(node, StructuredParserNode) and node.on_failure:
        return [("next", node.next), ("on_failure", node.on_failure)]
    return [("next", node.next)]


def _validate_graph(template: AgentTemplate) -> list[str]:
    errors: list[str] = []

    nodes_by_id: dict[str, AgentNode] = {}
    for node in template.nodes:
        if node.id in nodes_by_id:
            errors.append(f"Duplicate node id '{node.id}'.")
            continue
        nodes_by_id[node.id] = node

    if template.entry_node not in nodes_by_id:
        errors.append(f"entry_node '{template.entry_node}' does not match any node id.")

    for node in template.nodes:
        for label, target in _outgoing_edges(node):
            if target not in nodes_by_id:
                errors.append(
                    f"Node '{node.id}' {label} points to unknown node id '{target}'."
                )

    if errors:
        return errors

    reachable = _reachable_from(template.entry_node, nodes_by_id)
    for node_id in nodes_by_id:
        if node_id not in reachable:
            errors.append(f"Node '{node_id}' is unreachable from entry_node '{template.entry_node}'.")

    terminal_ids = {
        node_id
        for node_id, node in nodes_by_id.items()
        if isinstance(node, TerminalResponseNode)
    }
    if not terminal_ids:
        errors.append("Template must define at least one 'terminal_response' node.")
        return errors

    can_terminate = _can_reach_terminal(terminal_ids, nodes_by_id)
    for node_id in nodes_by_id:
        if node_id not in can_terminate:
            errors.append(f"Node '{node_id}' cannot reach a 'terminal_response' node.")

    return errors


def _reachable_from(start: str, nodes_by_id: dict[str, AgentNode]) -> set[str]:
    seen = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for _, target in _outgoing_edges(nodes_by_id[current]):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return seen


def _can_reach_terminal(
    terminal_ids: set[str], nodes_by_id: dict[str, AgentNode]
) -> set[str]:
    incoming: dict[str, list[str]] = {node_id: [] for node_id in nodes_by_id}
    for node_id, node in nodes_by_id.items():
        for _, target in _outgoing_edges(node):
            incoming[target].append(node_id)

    seen = set(terminal_ids)
    queue = deque(terminal_ids)
    while queue:
        current = queue.popleft()
        for predecessor in incoming[current]:
            if predecessor not in seen:
                seen.add(predecessor)
                queue.append(predecessor)
    return seen
