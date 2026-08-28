from typing import Any, TypedDict


class NodeLLMConfig(TypedDict, total=False):
    model_type: str
    model: str | None


class GraphState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    user_id: str
    thread_id: str
    node_llm_configs: dict[str, NodeLLMConfig]
    retrieved_docs: list[dict[str, Any]]
    retrieved_tools: list[dict[str, Any]]
    retry_count: int
    judge_verdict: dict[str, Any] | None
    next_route: str
    generated_response: str
