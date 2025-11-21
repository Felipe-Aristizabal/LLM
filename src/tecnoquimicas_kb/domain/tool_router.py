"""
Tool selection logic (router) for the Tecnoquímicas conversational agent.

This module implements an LLM-based router using a structured
"function calling" style. The LLM receives explicit JSON Schemas
for the available tools and must return a JSON object like:

{
  "tool_name": "structured_data",
  "arguments": {"fact_id": "company_main_brands"},
  "reason": "..."
}
"""

from __future__ import annotations


import json
import logging
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, ValidationError

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from tecnoquimicas_kb.config.settings import settings
from tecnoquimicas_kb.domain.models import ChatMessage, ToolChoice
from tecnoquimicas_kb.domain.prompts import P_TOOL_ROUTER

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Pydantic models for tool argument validation
# ---------------------------------------------------------------------------

class RagQAArgs(BaseModel):
    question: str

class StructuredDataArgs(BaseModel):
    fact_id: str

class ComposeArgs(BaseModel):
    question: str

# Map tool_name to its argument model
TOOL_ARG_MODELS: Dict[str, Type[BaseModel]] = {
    "rag_qa": RagQAArgs,
    "structured_data": StructuredDataArgs,
    "compose": ComposeArgs,
}

# ---------------------------------------------------------------------------
# Tool schemas (JSON Schema) for function-calling style routing
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "rag_qa": {
        "name": "rag_qa",
        "description": (
            "Responder preguntas abiertas usando la base documental indexada en FAISS."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Pregunta del usuario en lenguaje natural. Debe ser "
                        "la pregunta original, sin reescribir."
                    ),
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
    "structured_data": {
        "name": "structured_data",
        "description": (
            "Recuperar un dato puntual desde el JSON de hechos estructurados "
            "(direcciones, NIT, teléfonos, países, marcas principales, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fact_id": {
                    "type": "string",
                    "description": (
                        "Identificador del fact estructurado, por ejemplo "
                        '"company_nit", "hq_address_cali", '
                        '"company_main_brands", "company_countries", '
                        '"company_employees".'
                    ),
                }
            },
            "required": ["fact_id"],
            "additionalProperties": False,
        },
    },
    "compose": {
        "name": "compose",
        "description": (
            "Combinar RAG (FAISS) + hechos estructurados + historial reciente "
            "para rehacer o ampliar una respuesta anterior."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Nueva instrucción del usuario que modifica o amplía "
                        "respuestas previas, por ejemplo: "
                        '"vuelve a hacer el resumen pero incluye las marcas".'
                    ),
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_router_llm() -> BaseChatModel:
    """Return a chat model instance to be used as router LLM.

    English comment:
    For simplicity we reuse the same provider configuration as the
    main QA model. If you want a cheaper or smaller model for routing,
    you can change the model ID here independently.
    """
    provider = settings.llm.provider

    if provider == "ollama":
        return ChatOllama(model=settings.llm.ollama_model_id)

    # Default: Google Gemini family.
    return ChatGoogleGenerativeAI(model=settings.llm.google_model_id)


def _history_to_text(history: List[ChatMessage], max_messages: int = 6) -> str:
    """Serialize the recent chat history into a compact text block."""
    if not history:
        return "(sin historial reciente)"

    tail = history[-max_messages:]
    lines: List[str] = []

    for msg in tail:
        speaker = "Usuario" if msg.role == "user" else "Asistente"
        text = msg.content.replace("\n", " ").strip()
        if len(text) > 200:
            text = text[:197] + "..."
        lines.append(f"{speaker}: {text}")

    return "\n".join(lines)


def _extract_json_object(text: str) -> str:
    """Extract the first JSON object found in the text.

    Many chat models wrap the JSON response in ```json ... ``` blocks.
    This helper tries to be robust by removing common fences and then
    taking the substring between the first '{' and the last '}'.
    """
    stripped = text.strip()
    for fence in ("```json", "```JSON", "```", "json\n", "JSON\n"):
        stripped = stripped.replace(fence, "")
    stripped = stripped.strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return stripped

    return stripped[start : end + 1]


def _parse_router_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse the JSON emitted by the LLM router, with safety guards."""
    try:
        candidate = _extract_json_object(raw_text)
        return json.loads(candidate)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not parse router JSON %r: %s", raw_text, exc, exc_info=True
        )
        return None


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def choose_tool_llm(question: str, history: List[ChatMessage]) -> ToolChoice:
    """LLM-backed router using structured tool-calling style.

    The model receives the JSON Schemas of the tools and the recent
    chat history, and must output a JSON object with:
    - tool_name: "rag_qa" | "structured_data" | "compose"
    - arguments: dict matching the schema of that tool
    - reason: short explanation in Spanish
    """
    llm = _get_router_llm()
    history_text = _history_to_text(history)
    schemas_json = json.dumps(TOOL_SCHEMAS, ensure_ascii=False, indent=2)

    prompt = P_TOOL_ROUTER.format(
        tool_schemas=schemas_json,
        history=history_text,
        question=question,
    )

    try:
        out = llm.invoke(prompt)
        raw_text = getattr(out, "content", str(out))
        data = _parse_router_json(raw_text)

        if not data:
            raise ValueError("Router LLM returned unparsable JSON.")

        tool_name = str(data.get("tool_name", "")).strip().lower()
        arguments = data.get("arguments") or {}
        reason = str(data.get("reason", "") or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Router LLM error; falling back to rag_qa. Error: %s",
            exc,
            exc_info=True,
        )
        return ToolChoice(
            tool_name="rag_qa",
            arguments={"question": question},
            reason="Router LLM error; falling back to rag_qa.",
        )

    # Validación estricta del nombre de herramienta
    if tool_name not in TOOL_ARG_MODELS:
        logger.warning(
            "Router LLM returned unknown tool_name=%r; forcing rag_qa.",
            tool_name,
        )
        return ToolChoice(
            tool_name="rag_qa",
            arguments={"question": question},
            reason="Router returned unknown tool; forcing rag_qa.",
        )

    # Validación estricta de argumentos usando Pydantic
    arg_model = TOOL_ARG_MODELS[tool_name]
    try:
        valid_args = arg_model(**arguments).dict()
    except ValidationError as ve:
        logger.warning(
            f"Argument validation failed for tool {tool_name}: {ve}. Degrading to rag_qa."
        )
        return ToolChoice(
            tool_name="rag_qa",
            arguments={"question": question},
            reason=f"Argument validation failed for tool {tool_name}; degraded to rag_qa.",
        )

    return ToolChoice(
        tool_name=tool_name,
        arguments=valid_args,
        reason=reason or "LLM router choice.",
    )
