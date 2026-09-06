from app.graph.state import GraphState


def orchestrator_node(state: GraphState) -> GraphState:
    import time

    started = time.perf_counter()
    messages = state.get("messages", [])
    last_message = messages[-1].get("content", "") if messages else ""
    text = str(last_message).lower()

    if "tool" in text and "document" in text:
        state["next_route"] = "both"
    elif "tool" in text:
        state["next_route"] = "tools"
    elif "document" in text or "doc" in text:
        state["next_route"] = "docs"
    else:
        state["next_route"] = "direct"

    node_metrics = list(state.get("node_metrics", []))
    node_metrics.append(
        {
            "node_key": "orchestrator",
            "latency_ms": (time.perf_counter() - started) * 1000,
            "tokens_used": 0,
            "status": "completed",
        }
    )
    state["node_metrics"] = node_metrics

    return state
