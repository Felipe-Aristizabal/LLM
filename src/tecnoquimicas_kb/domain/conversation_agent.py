"""Orchestrator for conversational agent turns.

This module defines the high-level function `run_agent_turn`, which acts as
an abstraction layer between the Streamlit chat interface and the underlying
QA / retrieval systems. It manages conversation history, applies agent
settings, decides which tool to use (RAG vs datos estructurados) and
returns a structured result.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tecnoquimicas_kb.domain.context_builder import build_context_all
from tecnoquimicas_kb.domain.prompts import P_QA_COMPOSE, P_FOLLOWUP_CLASSIFIER
from tecnoquimicas_kb.domain.retriever import retrieve_for_question
from tecnoquimicas_kb.config.settings import settings
from tecnoquimicas_kb.domain import qa_service
from tecnoquimicas_kb.domain.models import (
    AgentSettings,
    AgentTurnResult,
    ChatMessage,
    QAResponse,
)
from tecnoquimicas_kb.domain.tool_router import choose_tool_llm
from tecnoquimicas_kb.infrastructure.structured.contact_store import (
    get_fact_by_id,
    load_all_facts,
    search_facts,
)

logger = logging.getLogger(__name__)


def _is_follow_up(
    user_input: str,
    history: List[ChatMessage],
    lookback: int = 4,
) -> bool:
    """LLM-based follow-up detector using a structured classifier prompt."""

    if not history:
        return False

    tail = history[-lookback:]
    if not any(m.role == "assistant" for m in tail):
        return False

    # Build short readable history
    lines: List[str] = []
    for msg in tail:
        who = "Usuario" if msg.role == "user" else "Asistente"
        txt = msg.content.replace("\n", " ").strip()
        if len(txt) > 250:
            txt = txt[:247] + "..."
        lines.append(f"{who}: {txt}")
    history_snippet = "\n".join(lines) if lines else "(sin historial)"

    llm = qa_service._ensure_llm()
    msg = P_FOLLOWUP_CLASSIFIER.format(history=history_snippet, question=user_input)

    try:
        out = llm.invoke(msg)
        raw = getattr(out, "content", str(out)).strip()
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        is_fu = bool(data.get("is_follow_up", False))
        reason = data.get("reason", "")
        logger.debug("Follow-up detector result: %s (reason=%s)", is_fu, reason)
        return is_fu
    except Exception:
        logger.warning(
            "Follow-up detection failed; defaulting to not follow-up.",
            exc_info=True,
        )
        return False


def _compose_answer(
    user_input: str,
    history: List[ChatMessage],
) -> QAResponse:
    """Compose mode: use RAG context + short history + exact structured facts.

    English comment:
    This is used for follow-up turns like:
    "Ahora vuelve a hacer el resumen ejecutivo pero incluye los productos principales".
    """

    # 1) Retrieve context from FAISS index
    docs = retrieve_for_question(user_input)
    context_text = build_context_all(docs)

    # 2) Build a short textual history window
    hist_tail = history[-4:]
    hist_lines: List[str] = []
    for msg in hist_tail:
        who = "Usuario" if msg.role == "user" else "Asistente"
        txt = msg.content.replace("\n", " ")
        if len(txt) > 250:
            txt = txt[:247] + "..."
        hist_lines.append(f"{who}: {txt}")
    history_snippet = "\n".join(hist_lines) if hist_lines else "(sin historial)"

    # 3) Select up to 3 structured facts that match the current question
    facts = search_facts(user_input, limit=3)
    facts_lines: List[str] = []
    sources: List[str] = []

    for fact in facts:
        # "label: value" format; do not rephrase the value
        facts_lines.append(f"- {fact.label}: {fact.value}")
        src = fact.metadata.get("source")
        if src:
            sources.append(src)

    facts_text = (
        "\n".join(facts_lines) if facts_lines else "(sin datos exactos relevantes)"
    )

    # 4) Call the LLM with the compose prompt
    llm = qa_service._ensure_llm()  # Reuse the same LLM factory as QA
    msg = P_QA_COMPOSE.format(
        history=history_snippet,
        context=context_text,
        facts=facts_text,
        q=user_input,
    )
    out = llm.invoke(msg)

    return QAResponse(
        question=user_input,
        answer=out.content,
        used_sources=sources,
        raw_context=context_text,
        extra={"mode": "compose_rag_plus_facts"},
    )


def _structured_data_file() -> Path:
    """Return the path to the structured-data JSON file.

    By convention we place it under:
        <data_root>/structured/structured_information.json

    where data_root is the parent of the configured clean data directory.
    """
    data_root = settings.paths.data_clean_dir.parent
    json_path = data_root / "structured" / "structured_information.json"
    logger.debug("Structured-data JSON path resolved to %s", json_path)
    return json_path


def _render_structured_answer(question: str, fact_id: str) -> Dict[str, Any]:
    """Build a natural-language answer from a structured fact.

    Returns a dictionary with:
    - "answer": text to show to the user.
    - "sources": list of source identifiers.
    - "tool_details": extra metadata about the tool execution.
    """
    logger.info(
        "Rendering structured-data answer for question=%r fact_id=%s",
        question,
        fact_id,
    )

    json_path = _structured_data_file()
    load_all_facts(json_path)  # Ensure cache is populated.
    fact = get_fact_by_id(fact_id)

    if fact is None:
        logger.warning("Structured fact with id %s not found.", fact_id)
        return {
            "answer": (
                "No encontré esa información en los datos estructurados. "
                "Intentaré responder usando los documentos disponibles."
            ),
            "sources": [],
            "tool_details": {
                "mode": "structured_data",
                "fact_id": fact_id,
                "status": "missing_fact",
            },
        }

    if fact.description:
        answer_text = f"{fact.description} {fact.value}"
    else:
        answer_text = f"{fact.label}: {fact.value}"

    sources: List[str] = []
    source_url = fact.metadata.get("source")
    if source_url:
        sources.append(source_url)
    else:
        sources.append(f"structured_data:{fact.id}")

    tool_details: Dict[str, Any] = {
        "mode": "structured_data",
        "fact_id": fact.id,
        "category": fact.category,
        "type": fact.type,
    }

    logger.debug(
        "Structured answer built: %r (sources=%r tool_details=%r)",
        answer_text,
        sources,
        tool_details,
    )

    return {
        "answer": answer_text,
        "sources": sources,
        "tool_details": tool_details,
    }


def _summarize_history(history: list[ChatMessage], prev_summary: str) -> str:
    """Update a rolling summary of the conversation.

    English comment: Keeps memory light by summarizing only last few turns.
    """
    recent = history[-6:]
    lines = []
    for m in recent:
        who = "Usuario" if m.role == "user" else "Asistente"
        lines.append(f"{who}: {m.content}")
    chunk = "\n".join(lines)

    llm = qa_service._ensure_llm()
    prompt = (
        "Resumen previo:\n"
        f"{prev_summary or '(vacío)'}\n\n"
        "Nuevos mensajes:\n"
        f"{chunk}\n\n"
        "Actualiza el resumen en 3–5 líneas, sin perder hechos importantes."
    )
    try:
        out = llm.invoke(prompt)
        return out.content.strip()
    except Exception:
        return prev_summary or ""


def run_agent_turn(
    user_input: str,
    history: List[ChatMessage],
    agent_settings: Optional[AgentSettings] = None,
) -> AgentTurnResult:
    logger.info("run_agent_turn called with user_input=%r", user_input)

    # 1) Effective settings
    cfg = agent_settings or AgentSettings()

    # 2) Truncate history (FIFO)
    orig_len = len(history)
    if orig_len > cfg.max_history_messages:
        history = history[-cfg.max_history_messages :]
        logger.debug(
            "History truncated from %d to %d messages.",
            orig_len,
            len(history),
        )

    # Build user message once
    user_msg = ChatMessage(role="user", content=user_input)

    # 3) PRIORITY STEP: follow-up → COMPOSE
    if cfg.allow_compose and _is_follow_up(
        user_input,
        history,
        cfg.followup_lookback,
    ):
        logger.info("Follow-up detected → using COMPOSE mode.")
        qa_resp = _compose_answer(user_input, history)

        tool_details: Dict[str, Any] = {
            "mode": "compose_rag_plus_facts",
            "router_reason": "Follow-up detected before router.",
            "num_sources": len(qa_resp.used_sources),
        }

        assistant_msg = ChatMessage(
            role="assistant",
            content=qa_resp.answer,
            used_tool="COMPOSE",
            sources=qa_resp.used_sources,
            metadata={"tool_details": tool_details},
        )

        updated_history = history + [user_msg, assistant_msg]

        return AgentTurnResult(
            answer=qa_resp.answer,
            used_tool="COMPOSE",
            tool_details=tool_details,
            updated_history=updated_history,
        )

    # 4) If NOT follow-up → normal router flow
    if cfg.use_router:
        choice = choose_tool_llm(user_input, history)
        tool = choice.tool or cfg.default_tool
        fact_id = choice.fact_id
        router_reason = choice.reason
        logger.debug(
            "Router decision: tool=%s fact_id=%s reason=%r",
            tool,
            fact_id,
            router_reason,
        )
    else:
        tool = cfg.default_tool
        fact_id = None
        router_reason = "Router disabled; using default tool."
        logger.debug("Router disabled. Using default tool=%s", tool)

    # 5) STRUCTURED_DATA branch
    if tool == "STRUCTURED_DATA" and fact_id:
        logger.info("Using STRUCTURED_DATA tool for this turn.")
        rendered = _render_structured_answer(user_input, fact_id)
        answer_text = rendered["answer"]
        sources = rendered["sources"]
        tool_details: Dict[str, Any] = rendered["tool_details"]
        tool_details["router_reason"] = router_reason

        assistant_msg = ChatMessage(
            role="assistant",
            content=answer_text,
            used_tool="STRUCTURED_DATA",
            sources=sources,
            metadata={"tool_details": tool_details},
        )

        updated_history = history + [user_msg, assistant_msg]

        return AgentTurnResult(
            answer=answer_text,
            used_tool="STRUCTURED_DATA",
            tool_details=tool_details,
            updated_history=updated_history,
        )

    # 6) Default: RAG_QA via QA service
    logger.info("Using RAG_QA tool for this turn.")
    qa_response = qa_service.answer_question(user_input)

    tool_details = {
        "mode": "rag_fullcontext",
        "router_reason": router_reason,
        "num_sources": len(qa_response.used_sources),
    }

    assistant_msg = ChatMessage(
        role="assistant",
        content=qa_response.answer,
        used_tool="RAG_QA",
        sources=qa_response.used_sources,
        metadata={"tool_details": tool_details},
    )

    updated_history = history + [user_msg, assistant_msg]

    return AgentTurnResult(
        answer=qa_response.answer,
        used_tool="RAG_QA",
        tool_details=tool_details,
        updated_history=updated_history,
    )
