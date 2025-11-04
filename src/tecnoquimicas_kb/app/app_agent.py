import os
import streamlit as st
from tecnoquimicas_kb.agent.memory import InMemorySessionStore
from tecnoquimicas_kb.agent.memory import InMemorySessionStore
from tecnoquimicas_kb.agent.router import agent_dispatch

# ===== puente con tu stuffing existente =====
def build_docs_context(user_query: str) -> str:
    """
    Llama a tu lógica existente de 'targeted stuffing' para Q&A.
    Debe devolver un string (contexto concatenado con metadatos).
    Si ya tienes funciones utilitarias, impórtalas aquí en vez de 'pass'.
    """
    # TODO: import real functions from your Módulo 1 (ej.: select_top_k_files, compose_context, etc.)
    # return compose_context(user_query, k=sidebar_value, limits=env_limits)
    return "CONTEXTO_DE_EJEMPLO: aquí iría el stuffing real construido desde los k archivos más relevantes."

# ===== Memoria por sesión =====
if "session_id" not in st.session_state:
    st.session_state.session_id = os.getenv("SESSION_ID", "default-session")
if "store" not in st.session_state:
    st.session_state.store = InMemorySessionStore()
if "chat" not in st.session_state:
    st.session_state.chat = []

st.set_page_config(page_title="Agente Tecnoquímicas", page_icon="💬")

st.title("Agente Conversacional — Tecnoquímicas")
st.caption("Memoria por sesión + Router de herramientas (Docs vs Datos Estructurados)")

# Sidebar config (proveedor, top-k, etc., según tu app existente)
with st.sidebar:
    st.subheader("Parámetros")
    st.write("Provider: Gemini u Ollama (config .env)")
    if st.button("Limpiar memoria"):
        st.session_state.store.clear(st.session_state.session_id)
        st.session_state.chat = []
        st.success("Memoria de la sesión limpiada.")

# Chat history render
for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# Input usuario
user_msg = st.chat_input("Escribe tu pregunta...")
if user_msg:
    # Append user
    st.session_state.chat.append({"role": "user", "content": user_msg})
    st.session_state.store.append(st.session_state.session_id, "user", user_msg)

    # Dispatch agente
    result = agent_dispatch(
        session_store=st.session_state.store,
        session_id=st.session_state.session_id,
        user_msg=user_msg,
        docs_context_builder=build_docs_context
    )
    assistant_msg = result["answer"]

    # Append assistant
    st.session_state.chat.append({"role": "assistant", "content": assistant_msg})
    st.session_state.store.append(st.session_state.session_id, "assistant", assistant_msg)

    # Render último
    with st.chat_message("assistant"):
        st.markdown(assistant_msg)

    with st.expander("🔎 Pensamiento del agente (debug)"):
        st.json({"route": result["route"], "thought": result["thought"]})