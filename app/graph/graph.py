from typing import Literal

from langgraph.graph import END, StateGraph

from app.graph.checkpointer import ensure_checkpointer_ready
from app.graph.generation import generation_node
from app.graph.judge import judge_node
from app.graph.orchestrator import orchestrator_node
from app.graph.retrieval_docs import document_retrieval_node
from app.graph.retrieval_tools import tool_retrieval_node
from app.graph.state import GraphState


def should_continue(state: GraphState) -> Literal["end", "retry_orchestrator"]:
    verdict = state.get("judge_verdict", {})
    retry_count = state.get("retry_count", 0)
    if verdict.get("verdict") == "pass":
        return "end"
    if retry_count >= 3:
        return "end"
    return "retry_orchestrator"


workflow = StateGraph(GraphState)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("tool_retrieval", tool_retrieval_node)
workflow.add_node("doc_retrieval", document_retrieval_node)
workflow.add_node("generation", generation_node)
workflow.add_node("judge", judge_node)

workflow.set_entry_point("orchestrator")
workflow.add_conditional_edges(
    "orchestrator",
    lambda state: state.get("next_route", "direct"),
    {
        "tools": "tool_retrieval",
        "docs": "doc_retrieval",
        "both": "tool_retrieval",
        "direct": "generation",
    },
)
workflow.add_edge("tool_retrieval", "generation")
workflow.add_edge("doc_retrieval", "generation")
workflow.add_edge("generation", "judge")
workflow.add_conditional_edges(
    "judge",
    should_continue,
    {
        "end": END,
        "retry_orchestrator": "orchestrator",
    },
)

app_graph = None


async def get_graph():
    global app_graph
    if app_graph is None:
        app_graph = workflow.compile(checkpointer=await ensure_checkpointer_ready())
    return app_graph
