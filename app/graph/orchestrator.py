from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import select

from app.agents.compiler import get_compiled_agent_graph
from app.agents.template_schema import validate_template
from app.db.base import AsyncSessionLocal
from app.db.models import AgentTemplate
from app.graph.state import GraphState
from app.llm.client import LLMClientError, embed_texts

_MIN_DESCRIPTION_SIMILARITY = 0.25
_MIN_LEXICAL_SIMILARITY = 0.08
_MAX_EMBEDDING_CANDIDATES = 5
_MAX_SUB_AGENT_EXECUTION_ATTEMPTS = 2
_SUB_AGENT_EXECUTION_TIMEOUT_SECONDS = 45
_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class _AgentCandidate:
    name: str
    version: int
    description: str
    raw_template: dict[str, Any]


def _tokenize(text: str) -> set[str]:
    return {token for token in _TOKEN_PATTERN.findall(text.lower()) if token}


def _safe_json_load(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _extract_template_description(name: str, raw_template: dict[str, Any]) -> str:
    direct_description = raw_template.get("description")
    if isinstance(direct_description, str) and direct_description.strip():
        return direct_description.strip()

    entry_node = str(raw_template.get("entry_node", "")).strip()
    nodes = raw_template.get("nodes", [])
    if isinstance(nodes, list):
        discovered_descriptions: list[str] = []
        entry_description: str | None = None

        for node in nodes:
            if not isinstance(node, dict):
                continue

            node_description = node.get("description")
            if isinstance(node_description, str) and node_description.strip():
                cleaned = node_description.strip()
                discovered_descriptions.append(cleaned)
                if entry_node and str(node.get("id", "")).strip() == entry_node and entry_description is None:
                    entry_description = cleaned

        if entry_description:
            return entry_description

        if discovered_descriptions:
            unique_descriptions = list(dict.fromkeys(discovered_descriptions))
            if len(unique_descriptions) == 1:
                return unique_descriptions[0]
            merged = " ".join(unique_descriptions[:4]).strip()
            return merged[:600] if merged else name.strip()

    return name.strip()


async def _load_enabled_agent_candidates() -> list[_AgentCandidate]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AgentTemplate)
            .where(AgentTemplate.enabled.is_(True), AgentTemplate.is_valid.is_(True))
            .order_by(AgentTemplate.name.asc(), AgentTemplate.version.desc())
        )
        records = list(result.scalars().all())

    latest_by_name: dict[str, _AgentCandidate] = {}
    for record in records:
        normalized_name = record.name.strip()
        if not normalized_name or normalized_name in latest_by_name:
            continue

        raw_template = _safe_json_load(record.raw_template)
        if not isinstance(raw_template, dict):
            continue

        latest_by_name[normalized_name] = _AgentCandidate(
            name=normalized_name,
            version=int(record.version),
            description=_extract_template_description(normalized_name, raw_template),
            raw_template=raw_template,
        )

    return list(latest_by_name.values())


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    dot_product = sum(lv * rv for lv, rv in zip(left, right))
    return dot_product / (left_norm * right_norm)


async def _rank_candidates_by_embedding(
    query: str,
    candidates: list[_AgentCandidate],
) -> list[tuple[_AgentCandidate, float]]:
    if not candidates:
        return []

    embedding_inputs = [query] + [f"{candidate.name}. {candidate.description}" for candidate in candidates]
    vectors = await embed_texts(
        embedding_inputs,
        metadata={"node": "orchestrator_agent_selector"},
    )
    if len(vectors) != len(embedding_inputs):
        return []

    query_vector = vectors[0]
    scored: list[tuple[_AgentCandidate, float]] = [
        (candidate, _cosine_similarity(query_vector, vectors[index + 1]))
        for index, candidate in enumerate(candidates)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)

    filtered = [
        (candidate, score)
        for candidate, score in scored
        if score >= _MIN_DESCRIPTION_SIMILARITY
    ]
    return filtered[:_MAX_EMBEDDING_CANDIDATES]


def _rank_candidates_by_lexical_overlap(
    query: str,
    candidates: list[_AgentCandidate],
) -> list[tuple[_AgentCandidate, float]]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored: list[tuple[_AgentCandidate, float]] = []
    for candidate in candidates:
        candidate_tokens = _tokenize(f"{candidate.name} {candidate.description}")
        if not candidate_tokens:
            continue
        overlap = len(query_tokens & candidate_tokens)
        score = overlap / max(len(query_tokens), 1)
        if score >= _MIN_LEXICAL_SIMILARITY:
            scored.append((candidate, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def _pick_candidate_by_name_match(query: str, candidates: list[_AgentCandidate]) -> _AgentCandidate | None:
    query_lc = query.lower()
    for candidate in sorted(candidates, key=lambda item: len(item.name), reverse=True):
        normalized_name = candidate.name.strip().lower()
        if not normalized_name:
            continue
        if normalized_name in query_lc:
            return candidate
    return None


def _extract_interrupt_payload(compiled_graph, config: dict[str, Any]) -> dict[str, Any] | None:
    try:
        snapshot = compiled_graph.get_state(config)
    except ValueError:
        return None

    for task in getattr(snapshot, "tasks", ()):
        interrupts = getattr(task, "interrupts", ())
        if not interrupts:
            continue
        raw_value = interrupts[0].value
        if isinstance(raw_value, dict):
            return raw_value
        return {"value": raw_value}

    return None


async def _execute_sub_agent(state: GraphState, candidate: _AgentCandidate) -> str | None:
    parsed_template = validate_template(candidate.raw_template)
    if parsed_template.template is None:
        return None

    user_message = _get_user_message_text(state)
    if not user_message:
        return None

    thread_id = str(state.get("thread_id", "")).strip() or "default"
    sub_agent_thread_id = f"{thread_id}:sub-agent:{candidate.name}:{candidate.version}"

    compiled_graph = get_compiled_agent_graph(
        name=candidate.name,
        version=candidate.version,
        template=parsed_template.template,
    )
    sub_state = {
        "input": user_message,
        "parsed_data": {},
        "messages": [{"role": "user", "content": user_message}],
        "iteration_count": 0,
        "service_results": {},
        "interrupt_payload": None,
        "final_response": "",
        "node_llm_configs": {},
    }
    config: dict[str, Any] = {"configurable": {"thread_id": sub_agent_thread_id}}
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(compiled_graph.invoke, sub_state, cast(Any, config)),
            timeout=_SUB_AGENT_EXECUTION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return None

    if _extract_interrupt_payload(compiled_graph, cast(Any, config)) is not None:
        return None

    if isinstance(result.get("error_event"), dict):
        return None

    response = str(result.get("final_response", "")).strip()
    return response or None


async def _try_route_to_sub_agent(state: GraphState, query_text: str) -> bool:
    candidates = await _load_enabled_agent_candidates()
    if not candidates:
        return False

    direct_name_match = _pick_candidate_by_name_match(query_text, candidates)
    ranked_candidates: list[tuple[str, _AgentCandidate, float]] = []

    if direct_name_match is not None:
        ranked_candidates.append(("name_match", direct_name_match, 1.0))
    else:
        try:
            embedding_ranked = await _rank_candidates_by_embedding(query_text, candidates)
            ranked_candidates.extend(
                ("description_embedding", candidate, score) for candidate, score in embedding_ranked
            )
        except (LLMClientError, RuntimeError, ValueError, TypeError):
            ranked_candidates = []

        if not ranked_candidates:
            lexical_ranked = _rank_candidates_by_lexical_overlap(query_text, candidates)
            ranked_candidates.extend(
                ("description_lexical", candidate, score) for candidate, score in lexical_ranked
            )

    for selection_mode, selected_candidate, selection_score in ranked_candidates[:_MAX_SUB_AGENT_EXECUTION_ATTEMPTS]:
        final_response = await _execute_sub_agent(state, selected_candidate)
        if not final_response:
            continue

        state["selected_agent"] = {
            "name": selected_candidate.name,
            "version": selected_candidate.version,
            "description": selected_candidate.description,
            "selection_mode": selection_mode,
            "selection_score": selection_score,
        }
        state["generated_response"] = final_response
        state["next_route"] = "direct"
        return True

    return False


def _get_user_message_text(state: GraphState) -> str:
    messages = state.get("messages", [])
    for message in reversed(messages):
        if str(message.get("role", "")).lower() == "user":
            content = str(message.get("content", "")).strip()
            if content:
                return content
    return ""


def _fallback_route(query_text: str) -> str:
    text = query_text.lower()
    if "tool" in text and "document" in text:
        return "both"
    if "tool" in text:
        return "tools"
    if "document" in text or "doc" in text:
        return "docs"
    return "direct"


async def orchestrator_node(state: GraphState) -> GraphState:
    query_text = _get_user_message_text(state)

    if query_text:
        routed_to_sub_agent = await _try_route_to_sub_agent(state, query_text)
        if routed_to_sub_agent:
            return state

    state["next_route"] = _fallback_route(query_text)
    return state
