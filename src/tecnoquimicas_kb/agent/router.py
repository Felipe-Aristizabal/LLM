"""Router del agente: orquesta herramientas y LLM.

Este módulo decide si responder con datos estructurados o con conocimiento
documental (RAG/Stuffing + LLM). Lee la configuración del modelo desde
variables de entorno para permitir conmutar entre proveedores (Gemini/Ollama).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

from dotenv import load_dotenv

from tecnoquimicas_kb.agent.memory import InMemorySessionStore
from tecnoquimicas_kb.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    DOCS_QA_INSTRUCTION,
)
from tecnoquimicas_kb.agent.tools import get_structured_answer

# Cargamos .env si existe (no impide sobreescritura por os.environ en tiempo de ejecución).
load_dotenv()


def _get_llm():
    """Construye y devuelve el cliente LLM según MODEL_PROVIDER.

    Lee las variables en tiempo de ejecución para que los cambios hechos
    desde la UI (sidebar) surtan efecto sin reiniciar el proceso.

    Env:
        MODEL_PROVIDER: "gemini" | "ollama"
        GEN_MODEL_ID: ID del modelo Gemini (ej.: "gemini-2.5-pro")
        GOOGLE_API_KEY: API key para Gemini (requerida si provider=gemini)
        OLLAMA_MODEL_ID: ID del modelo local de Ollama (ej.: "gemma3:4b")
        OLLAMA_BASE_URL: URL del daemon de Ollama (default: http://localhost:11434)

    Returns:
        Instancia de modelo compatible con LangChain.
    """
    provider = os.getenv("MODEL_PROVIDER", "gemini").strip().lower()

    if provider == "ollama":
        # Cliente para modelos locales de Ollama.
        from langchain_ollama import ChatOllama

        model_id = os.getenv("OLLAMA_MODEL_ID", "gemma3:4b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_id, temperature=0.2, base_url=base_url)

    # Por defecto: Gemini (servicio gestionado, requiere API key).
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY no está definido y MODEL_PROVIDER=gemini. "
            "Cambia a Ollama desde la barra lateral o define tu API key."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    model_id = os.getenv("GEN_MODEL_ID", "gemini-2.5-pro")
    return ChatGoogleGenerativeAI(
        model=model_id,
        temperature=0.2,
        google_api_key=api_key,
    )


def _build_messages(
    history: List[Dict[str, str]],
    docs_context: str,
) -> List[Dict[str, str]]:
    """Construye el arreglo de mensajes para el LLM.

    Args:
        history: Historial de la sesión en formato [{"role", "content"}, ...].
        docs_context: Contexto documental concatenado para la consulta.

    Returns:
        Lista de mensajes (diccionarios con 'role' y 'content').
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT}
    ]

    # Añadimos solo los últimos N turnos para evitar contextos gigantes.
    max_history = 10
    for turn in history[-max_history:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Instrucción y el contexto documental específico de la consulta.
    messages.append(
        {
            "role": "system",
            "content": f"{DOCS_QA_INSTRUCTION}\n\nCONTEXTO DOCUMENTAL:\n{docs_context}",
        }
    )
    return messages


def agent_dispatch(
    session_store: InMemorySessionStore,
    session_id: str,
    user_msg: str,
    docs_context_builder: Callable[[str], str],
) -> Dict[str, Any]:
    """Router principal del agente.

    Flujo:
        1) Intenta responder con datos estructurados (TOOL_STRUCT).
        2) Si no hay match, usa TOOL_DOCS (RAG/Stuffing + LLM) con memoria de sesión.

    Args:
        session_store: Implementación de memoria de sesión.
        session_id: Identificador único de la sesión.
        user_msg: Mensaje del usuario.
        docs_context_builder: Función que arma el contexto documental.

    Returns:
        Diccionario con:
            - answer: Respuesta en texto plano.
            - route: "TOOL_STRUCT" | "TOOL_DOCS" | "ERROR".
            - thought: Breve pista para depurar (no cadena de pensamiento).
    """
    # 1) Intento con datos estructurados deterministas.
    structured_answer = get_structured_answer(user_msg)
    if structured_answer:
        return {
            "answer": structured_answer,
            "route": "TOOL_STRUCT",
            "thought": "Consulta de datos estructurados.",
        }

    # 2) Ruta documental (RAG/Stuffing + LLM).
    try:
        llm = _get_llm()
        docs_context = docs_context_builder(user_msg)
        history = session_store.get_history(session_id)
        messages = _build_messages(history=history, docs_context=docs_context)

        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        return {
            "answer": answer,
            "route": "TOOL_DOCS",
            "thought": "Consulta documental con memoria de sesión.",
        }

    except Exception as exc:  # Captura de errores controlada para UI
        return {
            "answer": f"Error al procesar la consulta: {exc}",
            "route": "ERROR",
            "thought": "Fallo en la invocación del LLM o credenciales.",
        }
