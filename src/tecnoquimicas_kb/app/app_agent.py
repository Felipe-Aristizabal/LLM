"""Aplicación Streamlit del agente conversacional TQ.

Incluye:
- Selector de proveedor de modelo (Gemini/Ollama) desde la barra lateral.
- Memoria por sesión en memoria.
- Llamada al router que decide TOOL_STRUCT (datos) o TOOL_DOCS (documental).
- Manejo de errores y controles mínimos de estado.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

from tecnoquimicas_kb.agent.memory import InMemorySessionStore
from tecnoquimicas_kb.agent.router import agent_dispatch


# ==============================================
# Lógica de RAG existente
# ==============================================
def build_docs_context(user_query: str) -> str:
    """Construye el contexto documental para la consulta.

    Nota:
        Sustituye esta función por la integración real de tu “targeted stuffing”:
        selección de top-k archivos, limpieza/síntesis, límites por tamaño, etc.

    Args:
        user_query: Pregunta del usuario.

    Returns:
        Cadena con el contexto concatenado y metadatos útiles.
    """
    # TODO: importar tu pipeline real (e.g. compose_context(user_query, k=..., limits=...))
    return (
        "CONTEXTO_DE_EJEMPLO: aquí iría el stuffing real construido "
        "desde los k archivos más relevantes."
    )


# ==============================================
# Utilidad: aplicar configuración de modelo y limpiar cachés
# ==============================================
def apply_model_config() -> None:
    """Aplica variables de entorno para el router y fuerza un rerun limpio."""
    os.environ["MODEL_PROVIDER"] = st.session_state.MODEL_PROVIDER
    os.environ["GEN_MODEL_ID"] = st.session_state.GEN_MODEL_ID
    os.environ["OLLAMA_MODEL_ID"] = st.session_state.OLLAMA_MODEL_ID

    # Limpieza segura de cachés de Streamlit (si se usan en índice/embeddings).
    try:
        st.cache_resource.clear()
        st.cache_data.clear()
    except Exception:
        pass

    # Rerun compatible con versiones nuevas/antiguas.
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - compatibilidad con versiones antiguas
        st.experimental_rerun()  # type: ignore[attr-defined]


# ==============================================
# Estado de sesión
# ==============================================
if "session_id" not in st.session_state:
    st.session_state.session_id = os.getenv("SESSION_ID", "default-session")

if "store" not in st.session_state:
    st.session_state.store = InMemorySessionStore()

if "chat" not in st.session_state:
    st.session_state.chat = []

# Configuración de página (debe ir antes de render principal).
st.set_page_config(page_title="Agente Tecnoquímicas", page_icon="💬")

st.title("Agente Conversacional — Tecnoquímicas")
st.caption("Memoria por sesión + Router de herramientas (Docs vs Datos Estructurados)")

# Valores por defecto de provider/modelos (se pueden sobreescribir en sidebar).
if "MODEL_PROVIDER" not in st.session_state:
    st.session_state.MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini")

if "GEN_MODEL_ID" not in st.session_state:
    st.session_state.GEN_MODEL_ID = os.getenv("GEN_MODEL_ID", "gemini-2.5-pro")

if "OLLAMA_MODEL_ID" not in st.session_state:
    st.session_state.OLLAMA_MODEL_ID = os.getenv("OLLAMA_MODEL_ID", "gemma3:4b")

# ==============================================
# Barra lateral (configuración)
# ==============================================
with st.sidebar:
    st.header("Modelo")

    provider = st.selectbox(
        "Provider",
        ["gemini", "ollama"],
        index=["gemini", "ollama"].index(st.session_state.MODEL_PROVIDER),
        key="provider_select",
    )

    if provider == "gemini":
        _ = st.text_input(
            "Gemini model id",
            value=st.session_state.GEN_MODEL_ID,
            help="Ej.: gemini-2.5-pro | Requiere GOOGLE_API_KEY en entorno/.env.",
            key="gen_id_input",
        )
        st.caption("Requiere GOOGLE_API_KEY en el entorno o .env.")
    else:
        _ = st.text_input(
            "Ollama model id",
            value=st.session_state.OLLAMA_MODEL_ID,
            help="Ej.: gemma3:4b, llama3.2:3b, mistral:7b, qwen2.5:3b (recuerda `ollama pull`).",
            key="ollama_id_input",
        )
        st.caption("Asegura `ollama pull <modelo>` y que Ollama esté corriendo.")

    # Botón para aplicar configuración seleccionada.
    if st.button("Aplicar configuración de modelo"):
        st.session_state.MODEL_PROVIDER = provider
        if provider == "gemini":
            st.session_state.GEN_MODEL_ID = (
                st.session_state.get("gen_id_input") or st.session_state.GEN_MODEL_ID
            )
        else:
            st.session_state.OLLAMA_MODEL_ID = (
                st.session_state.get("ollama_id_input")
                or st.session_state.OLLAMA_MODEL_ID
            )
        apply_model_config()

    st.markdown("---")
    st.subheader("Configuración activa")
    st.code(
        {
            "MODEL_PROVIDER": os.getenv("MODEL_PROVIDER"),
            "GEN_MODEL_ID": os.getenv("GEN_MODEL_ID"),
            "OLLAMA_MODEL_ID": os.getenv("OLLAMA_MODEL_ID"),
        },
        language="json",
    )

    # Limpieza de memoria de conversación.
    if st.button("Limpiar memoria"):
        st.session_state.store.clear(st.session_state.session_id)
        st.session_state.chat = []
        st.success("Memoria de la sesión limpiada.")

    # Diagnóstico simple del backend de modelo activo.
    st.markdown("---")
    st.subheader("Salud del modelo (diagnóstico)")
    if os.getenv("MODEL_PROVIDER", "gemini").lower() == "ollama":
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.ok:
                st.success("Ollama: OK (daemon accesible).")
            else:
                st.warning("Ollama: respuesta no exitosa del daemon.")
        except Exception:
            st.error("Ollama: no se pudo conectar a http://localhost:11434.")
    else:
        if os.getenv("GOOGLE_API_KEY"):
            st.success("Gemini: API key presente en entorno.")
        else:
            st.warning("Gemini: falta GOOGLE_API_KEY en entorno/.env.")

# ==============================================
# Render del historial
# ==============================================
for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# ==============================================
# Entrada del usuario
# ==============================================
user_msg = st.chat_input("Escribe tu pregunta...")

if user_msg:
    # Registramos el turno del usuario (historial/memoria).
    st.session_state.chat.append({"role": "user", "content": user_msg})
    st.session_state.store.append(st.session_state.session_id, "user", user_msg)

    # Despacho al router (TOOL_STRUCT → TOOL_DOCS → ERROR).
    result = agent_dispatch(
        session_store=st.session_state.store,
        session_id=st.session_state.session_id,
        user_msg=user_msg,
        docs_context_builder=build_docs_context,
    )

    # Manejo de error: mostramos en UI sin guardar como respuesta del asistente.
    if result.get("route") == "ERROR":
        st.error(result.get("answer", "Ocurrió un error inesperado."))
    else:
        assistant_msg = result.get("answer", "")
        # Añadimos al historial y memoria solo si no es error.
        st.session_state.chat.append({"role": "assistant", "content": assistant_msg})
        st.session_state.store.append(
            st.session_state.session_id,
            "assistant",
            assistant_msg,
        )
        # Render del último mensaje.
        with st.chat_message("assistant"):
            st.markdown(assistant_msg)

    # Bloque de depuración (no expone cadena de pensamiento, solo metadatos).
    with st.expander("🔎 Pensamiento del agente (debug)"):
        st.json(
            {
                "route": result.get("route"),
                "thought": result.get("thought"),
            }
        )
