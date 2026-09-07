from __future__ import annotations

import asyncio
from typing import cast

from app.graph import orchestrator
from app.graph.state import GraphState
from app.llm.client import LLMClientError


def _run(coro):
    return asyncio.run(coro)


def _candidate(name: str, description: str, version: int = 1) -> orchestrator._AgentCandidate:
    return orchestrator._AgentCandidate(
        name=name,
        version=version,
        description=description,
        raw_template={
            "template_version": "1.0",
            "entry_node": "respond",
            "nodes": [
                {
                    "id": "respond",
                    "type": "terminal_response",
                    "config": {"template": "ok"},
                }
            ],
        },
    )


def test_extract_template_description_prefers_entry_node_description() -> None:
    raw_template = {
        "entry_node": "decide",
        "nodes": [
            {"id": "parse", "description": "Parse invoice fields"},
            {"id": "decide", "description": "Route invoice to approval or auto-post"},
            {"id": "reply", "description": "Respond with outcome"},
        ],
    }

    description = orchestrator._extract_template_description("InvoiceAgent", raw_template)

    assert description == "Route invoice to approval or auto-post"


def test_extract_template_description_merges_available_node_descriptions() -> None:
    raw_template = {
        "entry_node": "parse",
        "nodes": [
            {"id": "parse"},
            {"id": "lookup", "description": "Fetch account profile"},
            {"id": "reply", "description": "Return recommendation"},
        ],
    }

    description = orchestrator._extract_template_description("BudgetAgent", raw_template)

    assert "Fetch account profile" in description
    assert "Return recommendation" in description


def test_orchestrator_prefers_direct_name_match(monkeypatch) -> None:
    async def _test() -> None:
        picked = _candidate("BudgetAgent", "Handles budget questions")

        async def _fake_candidates():
            return [picked]

        async def _fake_execute(state, candidate):
            del state
            if candidate.name == picked.name:
                return "budget answer"
            return None

        monkeypatch.setattr(orchestrator, "_load_enabled_agent_candidates", _fake_candidates)
        monkeypatch.setattr(orchestrator, "_execute_sub_agent", _fake_execute)

        state = cast(GraphState, {"messages": [{"role": "user", "content": "Ask BudgetAgent for help"}]})
        result = await orchestrator.orchestrator_node(state)
        selected_agent = result.get("selected_agent")

        assert result.get("generated_response") == "budget answer"
        assert result.get("next_route") == "direct"
        assert selected_agent is not None
        assert selected_agent.get("name") == "BudgetAgent"
        assert selected_agent.get("selection_mode") == "name_match"
        assert selected_agent.get("selection_score") == 1.0

    _run(_test())


def test_orchestrator_tries_next_ranked_candidate_when_first_fails(monkeypatch) -> None:
    async def _test() -> None:
        first = _candidate("CollectionsAgent", "Collections and dunning")
        second = _candidate("BudgetAgent", "Budget planning and recommendations")
        call_order: list[str] = []

        async def _fake_candidates():
            return [first, second]

        async def _fake_rank(query, candidates):
            del query, candidates
            return [(first, 0.91), (second, 0.88)]

        async def _fake_execute(state, candidate):
            del state
            call_order.append(candidate.name)
            if candidate.name == first.name:
                return None
            return "fallback success"

        monkeypatch.setattr(orchestrator, "_load_enabled_agent_candidates", _fake_candidates)
        monkeypatch.setattr(orchestrator, "_pick_candidate_by_name_match", lambda query, candidates: None)
        monkeypatch.setattr(orchestrator, "_rank_candidates_by_embedding", _fake_rank)
        monkeypatch.setattr(orchestrator, "_execute_sub_agent", _fake_execute)

        state = cast(GraphState, {"messages": [{"role": "user", "content": "Need help with budget"}]})
        result = await orchestrator.orchestrator_node(state)
        selected_agent = result.get("selected_agent")

        assert call_order == ["CollectionsAgent", "BudgetAgent"]
        assert result.get("generated_response") == "fallback success"
        assert selected_agent is not None
        assert selected_agent.get("name") == "BudgetAgent"
        assert selected_agent.get("selection_mode") == "description_embedding"
        assert result.get("next_route") == "direct"

    _run(_test())


def test_orchestrator_falls_back_to_lexical_when_embedding_fails(monkeypatch) -> None:
    async def _test() -> None:
        lexical_pick = _candidate("DocsAgent", "Document retrieval and summarization")

        async def _fake_candidates():
            return [lexical_pick]

        async def _fail_embedding(query, candidates):
            del query, candidates
            raise LLMClientError("embedding unavailable")

        def _fake_lexical_rank(query, candidates):
            del query, candidates
            return [(lexical_pick, 0.5)]

        async def _fake_execute(state, candidate):
            del state
            if candidate.name == lexical_pick.name:
                return "lexical answer"
            return None

        monkeypatch.setattr(orchestrator, "_load_enabled_agent_candidates", _fake_candidates)
        monkeypatch.setattr(orchestrator, "_pick_candidate_by_name_match", lambda query, candidates: None)
        monkeypatch.setattr(orchestrator, "_rank_candidates_by_embedding", _fail_embedding)
        monkeypatch.setattr(orchestrator, "_rank_candidates_by_lexical_overlap", _fake_lexical_rank)
        monkeypatch.setattr(orchestrator, "_execute_sub_agent", _fake_execute)

        state = cast(GraphState, {"messages": [{"role": "user", "content": "Read docs and summarize"}]})
        result = await orchestrator.orchestrator_node(state)
        selected_agent = result.get("selected_agent")

        assert result.get("generated_response") == "lexical answer"
        assert selected_agent is not None
        assert selected_agent.get("name") == "DocsAgent"
        assert selected_agent.get("selection_mode") == "description_lexical"

    _run(_test())


def test_orchestrator_uses_fallback_route_when_no_agent_selected(monkeypatch) -> None:
    async def _test() -> None:
        async def _fake_candidates():
            return []

        monkeypatch.setattr(orchestrator, "_load_enabled_agent_candidates", _fake_candidates)

        state = cast(GraphState, {"messages": [{"role": "user", "content": "Use tool and document context"}]})
        result = await orchestrator.orchestrator_node(state)

        assert result.get("next_route") == "both"
        assert "selected_agent" not in result
        assert "generated_response" not in result

    _run(_test())
