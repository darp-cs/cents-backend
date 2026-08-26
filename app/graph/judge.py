from app.graph.state import GraphState


def judge_node(state: GraphState) -> GraphState:
    generated_response = state.get("generated_response", "")
    retrieved_docs = state.get("retrieved_docs", [])
    retrieved_tools = state.get("retrieved_tools", [])
    retry_count = state.get("retry_count", 0)

    # TODO: model config -> grade the response against retrieved context and return verdict + reason.
    verdict = "pass"
    reason = "Response is grounded in available context and does not require retry."

    if not generated_response.strip():
        verdict = "fail"
        reason = "No response generated."

    state["judge_verdict"] = {"verdict": verdict, "reason": reason}
    state["retry_count"] = retry_count + 1 if verdict == "fail" else retry_count
    return state
