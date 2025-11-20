"""Tool router for the Tecnoquímicas conversational agent.

This module decides whether a user question should be handled by:

- STRUCTURED_DATA: deterministic facts from the structured JSON file.
- RAG_QA: the document-based RAG pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from tecnoquimicas_kb.config.settings import settings
from tecnoquimicas_kb.domain.models import ChatMessage, ToolChoice

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured intents (heuristics)
# ---------------------------------------------------------------------------


@dataclass
class _StructuredIntent:
    """Internal representation of a structured-data intent."""

    fact_id: str
    description: str


# Minimal list of intents that are better answered with structured data.
_INTENTS: List[_StructuredIntent] = [
    _StructuredIntent(
        fact_id="phone_consumer_service_co",
        description="Teléfono de servicio al consumidor (Colombia).",
    ),
    _StructuredIntent(
        fact_id="company_nit",
        description="NIT de Tecnoquímicas (identificación tributaria).",
    ),
    _StructuredIntent(
        fact_id="hq_address_cali",
        description="Dirección de la sede principal en Cali.",
    ),
    _StructuredIntent(
        fact_id="hq_opening_hours",
        description="Horario de atención de la sede principal.",
    ),
    _StructuredIntent(
        fact_id="company_main_brands",
        description="Lista de marcas principales de Tecnoquímicas.",
    ),
]

_FACTS_DESCRIPTION = "\n".join(f"- {i.fact_id}: {i.description}" for i in _INTENTS)


# ---------------------------------------------------------------------------
# LLM-backed router
# ---------------------------------------------------------------------------


def _get_router_llm() -> BaseChatModel:
    """Create a chat model instance dedicated to routing decisions.

    For now we reuse the same provider/model configured for the main QA
    pipeline, but this function makes it easy to switch to a cheaper
    model in the future.
    """
    provider = settings.llm.provider
    logger.debug("Creating router LLM for provider=%s", provider)

    if provider == "ollama":
        return ChatOllama(model=settings.llm.ollama_model_id)

    # Default: Google Gemini chat model.
    return ChatGoogleGenerativeAI(model=settings.llm.google_model_id)


_ROUTER_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Eres un enrutador de herramientas para el asistente de Tecnoquímicas.\n"
                "Debes decidir QUÉ herramienta usar para responder una pregunta.\n\n"
                "Herramientas disponibles:\n"
                "1) STRUCTURED_DATA — usar si la pregunta pide un DATO puntual de la lista.\n"
                "   Identificadores (fact_id) disponibles:\n"
                f"{_FACTS_DESCRIPTION}\n\n"
                "2) RAG_QA — usar si la pregunta es abierta/narrativa (historia, productos,\n"
                "   sostenibilidad, procesos, comparaciones, etc.).\n\n"
                "Reglas:\n"
                "- Si la pregunta corresponde a un fact_id listado, elige STRUCTURED_DATA y ese fact_id.\n"
                "- Si la pregunta pide 'qué marcas maneja Tecnoquímicas' o 'marcas principales',\n"
                '  usa STRUCTURED_DATA con fact_id="company_main_brands".\n'
                "- Cuando tengas duda, usa RAG_QA.\n\n"
                "FORMATO DE SALIDA ESTRICTO (UNA sola línea, sin bloques de código):\n"
                '{{"tool": "STRUCTURED_DATA" | "RAG_QA", "fact_id": <string or null>, "reason": <string>}}'
            ),
        ),
        (
            "human",
            (
                "Historial de conversación (puede estar vacío):\n"
                "{history}\n\n"
                "Pregunta actual del usuario:\n"
                "{question}\n\n"
                "Devuelve SOLO el JSON pedido, sin ``` ni texto adicional."
            ),
        ),
    ]
)


def _history_to_text(history: List[ChatMessage], max_messages: int = 6) -> str:
    """Serialize recent history into a compact text representation."""
    if not history:
        return "(sin historial)"

    tail = history[-max_messages:]
    lines: List[str] = []
    for msg in tail:
        speaker = "Usuario" if msg.role == "user" else "Asistente"
        # Keep lines short to avoid wasting tokens.
        text = msg.content.replace("\n", " ").strip()
        if len(text) > 200:
            text = text[:197] + "..."
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _extract_json_object(text: str) -> str:
    """Extract the first JSON object found in the text.

    Many chat models wrap the JSON response in ```json ... ``` blocks.
    This helper tries to be robust by locating the first '{' and the
    last '}' and returning that slice.
    """
    s = text.strip()
    for fence in ("```json", "```JSON", "```", "json\n", "JSON\n"):
        s = s.replace(fence, "")
    s = s.strip()
    i, j = s.find("{"), s.rfind("}")
    return s if i == -1 or j == -1 or j <= i else s[i : j + 1]


def _parse_router_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse the JSON emitted by the LLM router, with safety guards."""
    try:
        return json.loads(_extract_json_object(raw_text))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Router JSON parse error for %r: %s", raw_text, exc, exc_info=True
        )
        return None


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def choose_tool_llm(question: str, history: List[ChatMessage]) -> ToolChoice:
    """LLM-backed router that chooses between structured data and RAG_QA."""
    llm = _get_router_llm()
    history_text = _history_to_text(history)
    msg = _ROUTER_PROMPT.format(question=question, history=history_text)

    try:
        out = llm.invoke(msg)
        raw_text = getattr(out, "content", str(out))
        data = _parse_router_json(raw_text)
        if not data:
            raise ValueError("Router LLM returned unparsable JSON.")
    except Exception:  # noqa: BLE001
        logger.warning(
            "Router LLM JSON parse error; falling back to RAG_QA.",
            exc_info=True,
        )
        return ToolChoice(
            tool="RAG_QA",
            fact_id=None,
            reason="Router LLM JSON parse error; falling back to RAG_QA.",
        )

    tool = str(data.get("tool", "RAG_QA")).upper()
    fact_id = data.get("fact_id")
    reason = str(data.get("reason", "") or "").strip()

    if tool not in {"RAG_QA", "STRUCTURED_DATA"}:
        logger.warning("Router LLM returned unknown tool=%r; forcing RAG_QA.", tool)
        return ToolChoice(
            tool="RAG_QA",
            fact_id=None,
            reason="Router LLM returned unknown tool; forcing RAG_QA.",
        )

    return ToolChoice(tool=tool, fact_id=fact_id, reason=reason or "LLM router choice.")
