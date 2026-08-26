from app.config import settings
from app.graph.state import GraphState


def document_retrieval_node(state: GraphState) -> GraphState:
    query_text = ""
    for message in state.get("messages", []):
        if message.get("role") == "user":
            query_text = str(message.get("content", ""))
            break

    # TODO: model config -> generate a document embedding from query_text, output vector of length VECTOR_DIMENSION.
    query_embedding = [0.0 for _ in range(settings.vector_dimension)]

    # Example pgvector query scoped by user_id:
    # SELECT id, source_filename, chunk_text, 1 - (embedding <=> :query_embedding) AS similarity
    # FROM documents WHERE user_id = :user_id ORDER BY embedding <=> :query_embedding LIMIT 5;
    state["retrieved_docs"] = [
        {
            "chunk_text": "Relevant document context for the user query.",
            "source_filename": "placeholder.txt",
            "similarity": 0.88,
            "query": query_text,
        }
    ]
    return state
