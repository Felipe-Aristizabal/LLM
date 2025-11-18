"""Streamlit app for full-context QA over Tecnoquímicas documents.

This app:
- Loads all cleaned documents from the configured data directory.
- Builds a stuffed context focused on the user question.
- Calls the selected LLM provider/model to generate an answer.
- Optionally displays the raw context used for the answer.
"""

from __future__ import annotations

from typing import List

import streamlit as st

from tecnoquimicas_kb.config.settings import settings
from tecnoquimicas_kb.domain.context_builder import build_context_for_question
from tecnoquimicas_kb.domain.models import Document
from tecnoquimicas_kb.domain.prompts import P_QA
from tecnoquimicas_kb.infrastructure.ingestion.file_loader import load_all_docs
from tecnoquimicas_kb.infrastructure.llm.llm_client import (
    get_llm,
    invoke_with_retry,
)


@st.cache_data(show_spinner="Loading knowledge base...", ttl=3600)
def _load_documents() -> List[Document]:
    """Load all documents from the configured clean data directory.

    The result is cached to avoid re-reading files on every interaction.
    """
    tuples = load_all_docs(settings.paths.data_clean_dir)
    docs: List[Document] = []
    for text, meta in tuples:
        docs.append(Document(text=text, metadata=meta))
    return docs


def _build_answer(
    question: str,
    provider: str,
    model_id: str,
    k_files: int,
    limit_chars: int,
) -> tuple[str, str]:
    """Build a context for the question and obtain an answer from the LLM.

    Returns
    -------
    tuple[str, str]
        A pair `(answer, context)` with the generated answer and the
        stuffed context that was sent to the model.
    """
    docs = _load_documents()
    if not docs:
        return (
            "No hay documentos cargados en la base de conocimiento.",
            "",
        )

    # Build a stuffed context focusing on the current question
    context = build_context_for_question(
        question,
        docs,
        k_files=k_files,
        limit_chars=limit_chars,
    )

    # Instantiate the LLM according to the selected provider and model
    llm = get_llm(provider=provider, model_id=model_id)

    # Prepare the chat prompt using the domain-level template
    prompt = P_QA.format(context=context, q=question)

    # Invoke the model with a small retry policy
    result = invoke_with_retry(llm, prompt)

    # LangChain chat models usually return an object with a `content`
    # attribute; fall back to string representation when needed.
    answer_text = getattr(result, "content", str(result))
    return answer_text, context


def main() -> None:
    """Render the Streamlit interface for full-context QA."""
    st.set_page_config(
        page_title="Tecnoquímicas KB – Full-context QA",
        page_icon="💊",
        layout="wide",
    )

    st.title("Tecnoquímicas – Full-context QA")
    st.write(
        "Pregunta en lenguaje natural y el sistema construirá un contexto "
        "a partir de la base de conocimiento interna para responder."
    )

    # Sidebar configuration for provider, model and context limits
    st.sidebar.header("Configuración del modelo y contexto")

    provider_label = st.sidebar.selectbox(
        "Proveedor del LLM ",
        options=["gemini", "ollama"],
        index=0 if settings.llm.provider == "gemini" else 1,
        help="Selecciona el proveedor que vas a usar para responder las preguntas.",
    )

    if provider_label == "gemini":
        default_model = settings.llm.google_model_id
    else:
        default_model = settings.llm.ollama_model_id

    model_id = st.sidebar.text_input(
        "Identificador del modelo",
        value=default_model,
        help="Id opcional del modelo; dejar el valor predeterminado en caso de que no lo necesites.",
    )

    k_files = st.sidebar.slider(
        "Cantidad máxima de documentos fuente",
        min_value=3,
        max_value=30,
        value=settings.context.default_k_files_qa,
        help="Número de documentos más relevantes a usar para construir el contexto.",
    )

    limit_chars = st.sidebar.slider(
        "Límite de caracteres para el contexto",
        min_value=5000,
        max_value=settings.context.max_context_chars_qa,
        value=min(20000, settings.context.max_context_chars_qa),
        step=1000,
        help=(
            "Número máximo de caracteres permitidos en el contexto final "
            "construido para la pregunta."
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Ruta docs : `{settings.paths.data_clean_dir}`")

    # Main QA interaction area
    docs = _load_documents()
    st.info(f"Has cargado **{len(docs)}** documentos desde la base del conocimiento.")

    question = st.text_area(
        "Tu pregunta:",
        placeholder="Ejemplo: ¿Qué programas de sostenibilidad maneja TQ?",
        height=120,
    )

    if st.button("Preguntar", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Por favor digita la información antes de preguntar.")
            return

        with st.spinner("Generando una respuesta..."):
            answer, context = _build_answer(
                question=question.strip(),
                provider=provider_label,
                model_id=model_id.strip() or default_model,
                k_files=k_files,
                limit_chars=limit_chars,
            )

        st.subheader("Respuesta")
        st.write(answer)

        with st.expander("Ver contexto usado para esta respuesta"):
            st.text(context)


if __name__ == "__main__":
    # Standard guard to allow running `python -m` on this module
    main()
