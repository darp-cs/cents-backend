from app.graph.state import GraphState


def generation_node(state: GraphState) -> GraphState:
    messages = list(state.get("messages", []))
    retrieved_docs = state.get("retrieved_docs", [])
    retrieved_tools = state.get("retrieved_tools", [])

    context_blocks = []
    for doc in retrieved_docs:
        context_blocks.append(f"Document: {doc.get('chunk_text', '')}")
    for tool in retrieved_tools:
        context_blocks.append(f"Tool: {tool.get('name', '')} - {tool.get('description', '')}")

    context = "\n".join(context_blocks) if context_blocks else "No additional context."

    prompt = (
        "You are a helpful assistant. Use the retrieved context when available.\n\n"
        f"Context:\n{context}\n\n"
        "Respond to the user using the conversation history and available context."
    )

    # TODO: model config -> generate response from conversation messages + prompt, output string.
    generated_text = "Generated response based on user intent and retrieved context."

    last_user_message = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user_message is not None:
        messages.append({"role": "assistant", "content": generated_text})
    state["messages"] = messages
    state["generated_response"] = generated_text
    return state
