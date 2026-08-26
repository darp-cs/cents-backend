from app.config import settings
from app.graph.state import GraphState
from app.vector_store import query_tools


def tool_retrieval_node(state: GraphState) -> GraphState:
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
    return state
