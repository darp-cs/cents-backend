from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    user_id: str
    thread_id: str
    chat_model: str
    node_models: dict[str, str]
    retrieved_docs: list[dict[str, Any]]
    retrieved_tools: list[dict[str, Any]]
    retry_count: int
    judge_verdict: dict[str, Any] | None
    next_route: str
    generated_response: str
