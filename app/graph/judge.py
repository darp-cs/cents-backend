import json

from app.config import settings
from app.graph.state import GraphState
from app.llm.client import LLMClientError, generate_text


def _heuristic_verdict(generated_response: str) -> tuple[str, str]:
    if not generated_response.strip():
        return "fail", "No response generated."
    return "pass", "Response is grounded in available context and does not require retry."


def _extract_context_summary(retrieved_docs: list[dict], retrieved_tools: list[dict]) -> str:
    document_snippets = [str(doc.get("chunk_text", "")).strip() for doc in retrieved_docs[:3]]
    tool_names = [str(tool.get("name", "")).strip() for tool in retrieved_tools[:5]]

    doc_text = "\n".join([snippet for snippet in document_snippets if snippet]) or "None"
    tools_text = ", ".join([name for name in tool_names if name]) or "None"
    return f"Documents:\n{doc_text}\n\nTools: {tools_text}"


async def _llm_judge_verdict(state: GraphState) -> tuple[str, str]:
    generated_response = state.get("generated_response", "")
    retrieved_docs = state.get("retrieved_docs", [])
    retrieved_tools = state.get("retrieved_tools", [])
    node_models = state.get("node_models", {})

    selected_model = node_models.get("judge") or settings.llm_default_judge_model
    prompt = (
        "Evaluate whether the assistant response is acceptable given the retrieved context. "
        "Return strict JSON with keys verdict and reason. verdict must be pass or fail."
    )

    payload = {
        "system_prompt": prompt,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Retrieved context:\n{_extract_context_summary(retrieved_docs, retrieved_tools)}\n\n"
                    f"Assistant response:\n{generated_response}"
                ),
            }
        ],
        "model": selected_model,
        "temperature": 0.0,
        "max_tokens": 160,
        "metadata": {
            "user_id": state.get("user_id", ""),
            "thread_id": state.get("thread_id", ""),
            "node": "judge",
        },
    }

    response = await generate_text(payload)
    raw_text = str(response.get("text", "")).strip()
    if not raw_text:
        return "fail", "Judge model returned an empty verdict."

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            verdict = str(parsed.get("verdict", "")).strip().lower()
            reason = str(parsed.get("reason", "")).strip() or "Judge model did not include a reason."
            if verdict in {"pass", "fail"}:
                return verdict, reason
    except json.JSONDecodeError:
        pass

    lowered = raw_text.lower()
    if "fail" in lowered and "pass" not in lowered:
        return "fail", "Judge model marked response as failed."
    return "pass", "Judge model marked response as acceptable."


async def judge_node(state: GraphState) -> GraphState:
    generated_response = state.get("generated_response", "")
    retry_count = state.get("retry_count", 0)

    verdict, reason = _heuristic_verdict(generated_response)

    if settings.llm_judge_enabled and generated_response.strip():
        try:
            verdict, reason = await _llm_judge_verdict(state)
        except LLMClientError as exc:
            verdict = "fail"
            reason = f"Judge model request failed: {exc}"

    state["judge_verdict"] = {"verdict": verdict, "reason": reason}
    state["retry_count"] = retry_count + 1 if verdict == "fail" else retry_count
    return state
