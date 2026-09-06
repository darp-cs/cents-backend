from app.config import settings
from app.graph.state import GraphState
from app.llm.client import LLMClientError, generate_text


def _extract_token_count(payload: dict) -> int:
    usage = payload.get("usage", payload.get("token_usage", {}))
    if not isinstance(usage, dict):
        return 0
    for key in ("total_tokens", "tokens", "total"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return max(0, int(value))
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    if isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
        return max(0, int(input_tokens) + int(output_tokens))
    return 0


def _build_context(retrieved_docs: list[dict], retrieved_tools: list[dict]) -> str:
    context_blocks = []

    for doc in retrieved_docs:
        context_blocks.append(f"Document: {doc.get('chunk_text', '')}")

    for tool in retrieved_tools:
        context_blocks.append(f"Tool: {tool.get('name', '')} - {tool.get('description', '')}")

    return "\n".join(context_blocks) if context_blocks else "No additional context."


def _normalize_messages(messages: list[dict]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []

    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()

        if role not in {"user", "assistant", "system"} or not content:
            continue

        normalized.append({"role": role, "content": content})

    return normalized


def _resolve_generation_model_config(state: GraphState) -> tuple[str, str | None]:
    node_llm_configs = state.get("node_llm_configs", {})
    generation_config = node_llm_configs.get("generation", {})

    model_type = str(generation_config.get("model_type", "")).strip()
    if not model_type:
        model_type = settings.llm_default_generation_model_type.strip()

    if not model_type:
        raise RuntimeError("Generation node model_type is required.")

    model = str(generation_config.get("model", "")).strip() if generation_config.get("model") else ""
    return model_type, (model or None)


async def generation_node(state: GraphState) -> GraphState:
    import time

    started = time.perf_counter()
    messages = list(state.get("messages", []))
    retrieved_docs = state.get("retrieved_docs", [])
    retrieved_tools = state.get("retrieved_tools", [])

    context = _build_context(retrieved_docs, retrieved_tools)
    system_prompt = (
        "You are a helpful assistant. Use the retrieved context when available.\n\n"
        f"Context:\n{context}\n\n"
        "Respond to the user using the conversation history and available context."
    )

    model_type, selected_model = _resolve_generation_model_config(state)

    request_payload = {
        "messages": _normalize_messages(messages),
        "system_prompt": system_prompt,
        "model_folder": model_type,
        "temperature": settings.llm_default_temperature,
        "max_tokens": settings.llm_default_max_tokens,
        "metadata": {
            "user_id": state.get("user_id", ""),
            "thread_id": state.get("thread_id", ""),
            "node": "generation",
        },
    }

    if selected_model:
        request_payload["model"] = selected_model

    try:
        payload = await generate_text(request_payload)
    except LLMClientError as exc:
        raise RuntimeError(str(exc)) from exc

    generated_text = str(payload.get("text", "")).strip()
    if not generated_text:
        raise RuntimeError("LLM service returned an empty response.")

    last_user_message = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user_message is not None:
        messages.append({"role": "assistant", "content": generated_text})

    state["messages"] = messages
    state["generated_response"] = generated_text
    tokens = _extract_token_count(payload)
    token_usage = dict(state.get("token_usage", {}))
    token_usage["generation"] = token_usage.get("generation", 0) + tokens
    state["token_usage"] = token_usage
    node_metrics = list(state.get("node_metrics", []))
    node_metrics.append(
        {
            "node_key": "generation",
            "latency_ms": (time.perf_counter() - started) * 1000,
            "tokens_used": tokens,
            "status": "completed",
        }
    )
    state["node_metrics"] = node_metrics
    return state
