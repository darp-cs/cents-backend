from app.config import settings
from app.graph.state import GraphState
from app.vector_store import query_documents


def document_retrieval_node(state: GraphState) -> GraphState:
    import time

    started = time.perf_counter()
    query_text = ""
    for message in state.get("messages", []):
        if message.get("role") == "user":
            query_text = str(message.get("content", ""))
            break

    # TODO: model config -> generate a document embedding from query_text.
    query_embedding = [0.0 for _ in range(settings.vector_dimension)]
    user_id = str(state.get("user_id", ""))
    docs = query_documents(user_id=user_id, query_embedding=query_embedding, limit=5)
    state["retrieved_docs"] = [
        {
            "chunk_text": item["chunk_text"],
            "source_filename": item["source_filename"],
            "similarity": item["similarity"],
            "query": query_text,
        }
        for item in docs
    ]
    node_metrics = list(state.get("node_metrics", []))
    node_metrics.append({"node_key": "doc_retrieval", "latency_ms": (time.perf_counter() - started) * 1000, "tokens_used": 0, "status": "completed"})
    state["node_metrics"] = node_metrics
    return state
