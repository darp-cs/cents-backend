from app.config import settings
from app.graph.state import GraphState


def tool_retrieval_node(state: GraphState) -> GraphState:
    query_text = ""
    for message in state.get("messages", []):
        if message.get("role") == "user":
            query_text = str(message.get("content", ""))
            break

    # TODO: model config -> generate a tool embedding from query_text, output vector of length VECTOR_DIMENSION.
    query_embedding = [0.0 for _ in range(settings.vector_dimension)]

    # Example pgvector query shape for a tool table with an embedding column:
    # SELECT name, description, 1 - (embedding <=> :query_embedding) AS similarity
    # FROM tool_definitions ORDER BY embedding <=> :query_embedding LIMIT 5;
    state["retrieved_tools"] = [
        {
            "name": "tool_search",
            "description": "Generic tool lookup for user intent and operation selection.",
            "similarity": 0.9,
            "source": "placeholder",
            "query": query_text,
        }
    ]
    return state
