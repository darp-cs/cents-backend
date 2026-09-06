import time

from app.config import settings
from app.graph.state import GraphState
from app.services.metrics import record_metric_event
from app.vector_store import query_tools


async def tool_retrieval_node(state: GraphState) -> GraphState:
    started_at = time.perf_counter()
    query_text = ""
    for message in state.get("messages", []):
        if message.get("role") == "user":
            query_text = str(message.get("content", ""))
            break

    # TODO: model config -> generate a tool embedding from query_text.
    query_embedding = [0.0 for _ in range(settings.vector_dimension)]
    tools = query_tools(query_embedding=query_embedding, limit=5)
    state["retrieved_tools"] = [
        {
            "name": item["name"],
            "description": item["description"],
            "similarity": item["similarity"],
            "source": item["source"],
            "query": query_text,
        }
        for item in tools
    ]
    await record_metric_event(
        conversation_id=state.get("conversation_id"),
        node_key="retrieved_tools",
        latency_ms=(time.perf_counter() - started_at) * 1000,
        retrieved_count=len(tools),
    )
    return state
