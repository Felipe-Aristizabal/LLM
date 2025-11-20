"""Streamlit app for full-context QA over Tecnoquímicas documents.

This app:
- Loads all cleaned documents from the configured data directory.
- Builds a stuffed context focused on the user question.
- Calls the selected LLM provider/model to generate an answer.
- Optionally displays the raw context used for the answer.
"""

from __future__ import annotations

from typing import List, Dict, Any

import os
import streamlit as st
from urllib.parse import urlparse

from tecnoquimicas_kb.config.settings import settings
from tecnoquimicas_kb.domain.conversation_agent import run_agent_turn
from tecnoquimicas_kb.domain.models import ChatMessage, AgentSettings, Document
from tecnoquimicas_kb.infrastructure.ingestion.file_loader import load_all_docs


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Cargando base de conocimiento...", ttl=3600)
def _load_documents() -> List[Document]:
    """Load all documents from the configured clean data directory.

    The result is cached to avoid re-reading files on every interaction.

    Returns
    -------
    List[Document]
        List of all domain `Document` instances representing the
        cleaned knowledge base.
    """
    tuples = load_all_docs(settings.paths.data_clean_dir)
    docs: List[Document] = []
    for text, meta in tuples:
        docs.append(Document(text=text, metadata=meta))
    return docs


# ---------------------------------------------------------------------------
# Session / UI helpers
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    """Ensure that the required keys exist in Streamlit session state."""
    if "history" not in st.session_state:
        # We store ChatMessage objects directly; Streamlit can keep them
        # in memory between reruns.
        st.session_state["history"]: List[ChatMessage] = []


def _render_sidebar(doc_count: int) -> Dict[str, Any]:
    """Render the sidebar and return UI-level config for the agent and LLM."""

    st.sidebar.header("Configuración del asistente")

    # ---- LLM provider & model ----
    default_provider = settings.llm.provider
    if default_provider == "gemini":
        default_model = settings.llm.google_model_id
    else:
        default_model = settings.llm.ollama_model_id

    provider = st.sidebar.selectbox(
        "Proveedor LLM",
        options=["gemini", "ollama"],
        index=0 if default_provider == "gemini" else 1,
        key="llm_provider",
        help="Cambia entre modelos en la nube (Gemini) o locales (Ollama).",
    )

    model_label = (
        "ID de modelo (Gemini, p.ej. gemini-2.5-pro)"
        if provider == "gemini"
        else "ID de modelo (Ollama, p.ej. gemma3:4b)"
    )
    model_id = st.sidebar.text_input(
        model_label,
        value=default_model,
        key="llm_model_id",
    )

    # Set UI overrides so qa_service._ensure_llm() picks them up
    os.environ["UI_MODEL_PROVIDER"] = provider
    if provider == "gemini":
        os.environ["UI_GEN_MODEL_ID"] = model_id
        # Limpia el otro para evitar confusiones
        os.environ.pop("UI_OLLAMA_MODEL_ID", None)
    else:
        os.environ["UI_OLLAMA_MODEL_ID"] = model_id
        os.environ.pop("UI_GEN_MODEL_ID", None)

    st.sidebar.markdown(
        f"**Proveedor activo:** `{provider}`  \n**Modelo:** `{model_id}`"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Base de conocimiento: **{doc_count}** documentos  \n"
        f"Ruta: `{settings.paths.data_clean_dir}`"
    )

    # ---- Agent memory / behavior ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("Memoria y herramientas del agente")

    max_history = st.sidebar.slider(
        "Mensajes que recuerda el agente",
        min_value=4,
        max_value=40,
        value=st.session_state.get("agent_max_history", 20),
        step=1,
        key="agent_max_history",
        help="Limita cuántos mensajes previos se usan como contexto.",
    )

    followup_lookback = st.sidebar.slider(
        "Ventana para detectar seguimiento",
        min_value=2,
        max_value=20,
        value=st.session_state.get("agent_followup_lookback", 8),
        step=1,
        key="agent_followup_lookback",
        help="Cuántos turnos recientes se miran para decidir si una pregunta es seguimiento.",
    )

    use_router = st.sidebar.checkbox(
        "Usar router (RAG vs datos estructurados)",
        value=st.session_state.get("agent_use_router", True),
        key="agent_use_router",
    )

    allow_compose = st.sidebar.checkbox(
        "Permitir modo COMPOSE (fusionar resumen + datos exactos)",
        value=st.session_state.get("agent_allow_compose", True),
        key="agent_allow_compose",
    )

    st.sidebar.markdown("---")
    if st.sidebar.button(
        "Nueva conversación",
        type="secondary",
        use_container_width=True,
    ):
        st.session_state["history"] = []
        st.rerun()

    # Return a small config dict for main()
    return {
        "llm_provider": provider,
        "llm_model_id": model_id,
        "max_history_messages": int(max_history),
        "followup_lookback": int(followup_lookback),
        "use_router": bool(use_router),
        "allow_compose": bool(allow_compose),
    }


def _render_chat_history(history: List[ChatMessage]) -> None:
    """Render existing chat messages with full content and metadata."""
    for msg in history:
        role = "user" if msg.role == "user" else "assistant"

        with st.chat_message(role):
            # Mostrar el contenido principal del mensaje
            if msg.content:
                st.markdown(msg.content)
            else:
                st.markdown("_[sin contenido visible]_")

            # Mostrar metadatos y tool info si es respuesta del asistente
            if msg.role == "assistant":
                tool = msg.used_tool or "N/A"
                tool_details = msg.metadata.get("tool_details", {})
                mode = tool_details.get("mode", "")
                reason = tool_details.get("router_reason", "")

                meta_line = f"🧠 **Herramienta:** `{tool}`"
                if mode:
                    meta_line += f" · **Modo:** `{mode}`"
                if reason:
                    meta_line += f" · _{reason}_"

                st.caption(meta_line)

                # Mostrar fuentes si existen
                if msg.sources:
                    display_sources = _filter_display_sources(msg.sources)

                    if display_sources:
                        with st.expander("📚 Fuentes utilizadas"):
                            for src in display_sources:
                                # You can also turn them into clickable links if you want
                                st.markdown(f"- [{src}]({src})")


def _filter_display_sources(sources: list[str]) -> list[str]:
    """Return only HTTP/HTTPS URLs from the given sources list.

    Local file paths or internal identifiers are filtered out so
    the user only sees external links in the UI.
    """
    cleaned: list[str] = []
    seen: set[str] = set()

    for raw in sources:
        if not raw:
            continue
        s = raw.strip()
        parsed = urlparse(s)

        # Keep only http/https URLs
        if parsed.scheme not in ("http", "https"):
            continue

        # Deduplicate while preserving order
        if s not in seen:
            seen.add(s)
            cleaned.append(s)

    return cleaned


# ---------------------------------------------------------------------------
# Main app entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the Streamlit chat interface for Tecnoquímicas QA."""
    st.set_page_config(
        page_title="Tecnoquímicas KB – Agente conversacional",
        page_icon="💊",
        layout="wide",
    )

    _init_session_state()
    docs = _load_documents()
    doc_count = len(docs)

    ui_cfg = _render_sidebar(doc_count)

    st.title("Tecnoquímicas")
    st.write(
        "Pregunta en lenguaje natural y el asistente construirá un contexto "
        "a partir de la base de conocimiento interna para responder. "
        "Las respuestas se mantienen dentro de la conversación para que "
        "puedas hacer preguntas de seguimiento."
    )

    # Quick status about loaded documents.
    st.info(f"Has cargado **{doc_count}** documentos desde la base del conocimiento.")

    history: List[ChatMessage] = st.session_state["history"]

    # First render the previous messages so the user sees full history.
    _render_chat_history(history)

    # Chat input for the next user message.
    user_input = st.chat_input(
        "Escribe tu pregunta sobre Tecnoquímicas o sus productos..."
    )

    if user_input is not None and user_input.strip():
        # Configure agent settings for this session.
        agent_cfg = AgentSettings(
            max_history_messages=ui_cfg["max_history_messages"],
            use_router=ui_cfg["use_router"],
            default_tool="RAG_QA",
            allow_compose=ui_cfg["allow_compose"],
            followup_lookback=ui_cfg["followup_lookback"],
        )
        with st.spinner("Generando una respuesta..."):
            result = run_agent_turn(
                user_input=user_input.strip(),
                history=history,
                agent_settings=agent_cfg,
            )

        # Attach tool_details to the last assistant message so it can be
        # rendered in future turns as part of the history.
        updated_history = result.updated_history
        if updated_history and updated_history[-1].role == "assistant":
            updated_history[-1].metadata["tool_details"] = result.tool_details

        st.session_state["history"] = updated_history

        # Rerun the script so that the new messages appear rendered in
        # the chat history section at the top.
        st.rerun()


if __name__ == "__main__":
    # Standard guard to allow running `python -m` on this module.
    main()
