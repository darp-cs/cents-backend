from app.config import settings
from app.graph.state import GraphState
from app.llm.client import LLMClientError, generate_text


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
    return state
