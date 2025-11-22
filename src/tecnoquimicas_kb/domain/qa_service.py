"""High-level QA services for the Tecnoquímicas RAG assistant.

This module exposes the main entrypoints for question answering and
content generation:

- answer_question:      responde una pregunta usando toda la base de
                        conocimiento limpia.
- summarize_for_onboarding: genera un resumen introductorio.
- generate_faqs:        produce una lista de preguntas frecuentes.
"""

from __future__ import annotations

import logging
import os
from typing import List, Sequence

from google.api_core.exceptions import ResourceExhausted
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from tecnoquimicas_kb.config.settings import settings
from tecnoquimicas_kb.domain.context_builder import (
    build_context_all,
    build_context_for_question,
)
from tecnoquimicas_kb.domain.models import DocLike, Document, QAResponse
from tecnoquimicas_kb.domain.prompts import P_FAQ, P_QA, P_SUMMARY
from tecnoquimicas_kb.domain.retriever import retrieve_for_question
from tecnoquimicas_kb.infrastructure.ingestion.file_loader import load_all_docs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _ensure_llm() -> BaseChatModel:
    """Create a chat model instance based on the configured provider.

    The selection is driven by `settings.llm.provider` and the associated
    model identifiers in the settings module.
    """
    ui_provider = os.getenv("UI_MODEL_PROVIDER")
    provider = (ui_provider or settings.llm.provider).lower()

    # Read sampling params from environment (set by UI)
    temperature = float(os.getenv("UI_LLM_TEMPERATURE", "0.7"))
    top_p = float(os.getenv("UI_LLM_TOP_P", "1.0"))
    top_k = int(os.getenv("UI_LLM_TOP_K", "40"))

    if provider == "ollama":
        ui_model = os.getenv("UI_OLLAMA_MODEL_ID")
        model_id = ui_model or settings.llm.ollama_model_id
        logger.debug(
            "Creating Ollama chat model. provider=%s model=%s (ui_provider=%r ui_model=%r)",
            provider,
            model_id,
            ui_provider,
            ui_model,
        )
        return ChatOllama(model=model_id, temperature=temperature, top_p=top_p, top_k=top_k)

    # Default - Gemini
    ui_model = os.getenv("UI_GEN_MODEL_ID")
    model_id = ui_model or settings.llm.google_model_id
    logger.debug(
        "Creating Gemini chat model. provider=%s model=%s (ui_provider=%r ui_model=%r)",
        provider,
        model_id,
        ui_provider,
        ui_model,
    )
    return ChatGoogleGenerativeAI(model=model_id, temperature=temperature, top_p=top_p, top_k=top_k)


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------


def _load_corpus_documents() -> List[Document]:
    """Load all knowledge-base documents from the clean data directory."""
    tuples = load_all_docs(settings.paths.data_clean_dir)
    docs: List[Document] = [Document(text=text, metadata=meta) for text, meta in tuples]
    logger.debug(
        "Loaded %d documents from %s",
        len(docs),
        settings.paths.data_clean_dir,
    )
    return docs


# ---------------------------------------------------------------------------
# Public utilities: summaries and FAQs
# ---------------------------------------------------------------------------


def summarize_for_onboarding(docs: Sequence[DocLike]) -> str:
    """Generate an onboarding summary for a new client.

    Parameters
    ----------
    docs:
        Iterable of documents to summarize. Can be `Document` instances
        or (text, metadata) tuples.

    Returns
    -------
    str
        Short Spanish summary. Returns an empty string if no documents
        are provided.
    """
    if not docs:
        logger.warning("summarize_for_onboarding called with no documents.")
        return ""

    ctx = build_context_all(docs)
    llm = _ensure_llm()
    msg = P_SUMMARY.format(context=ctx)

    try:
        out = llm.invoke(msg)
        return out.content
    except ResourceExhausted:
        logger.warning(
            "ResourceExhausted in summarize_for_onboarding; retrying once.",
        )
        out = llm.invoke(msg)
        return out.content


def generate_faqs(docs: Sequence[DocLike], n: int = 10) -> str:
    """Generate a FAQ list based on the provided documents.

    Parameters
    ----------
    docs:
        Iterable of documents to analyse.
    n:
        Desired number of FAQ entries.

    Returns
    -------
    str
        Text block with FAQs in Spanish. Returns an empty string if no
        documents are provided.
    """
    if not docs:
        logger.warning("generate_faqs called with no documents.")
        return ""

    ctx = build_context_all(docs)
    llm = _ensure_llm()
    msg = P_FAQ.format(context=ctx, n=n)

    try:
        out = llm.invoke(msg)
        return out.content
    except ResourceExhausted:
        logger.warning(
            "ResourceExhausted in generate_faqs; retrying once.",
        )
        out = llm.invoke(msg)
        return out.content


# ---------------------------------------------------------------------------
# Core QA logic
# ---------------------------------------------------------------------------


def answer_question_from_docs(
    question: str,
    docs: Sequence[DocLike],
) -> QAResponse:
    """Answer a question using the supplied documents as context.

    This function dont make I/O: Assume the caller has already loaded
    the documents.

    Parameters
    ----------
    question:
        User question in natural language.
    docs:
        Iterable of documents that form the knowledge base for this call.

    Returns
    -------
    QAResponse
        Object containing the final answer and basic metadata.
    """
    logger.debug(
        "answer_question_from_docs called with question=%r and %d docs.",
        question,
        len(docs),
    )

    if not docs:
        return QAResponse(question=question, answer="")

    ctx = build_context_for_question(question, docs)
    llm = _ensure_llm()
    msg = P_QA.format(context=ctx, q=question)

    try:
        out = llm.invoke(msg)
        answer_text = out.content
    except ResourceExhausted:
        logger.warning(
            "ResourceExhausted in answer_question_from_docs; retrying once.",
        )
        out = llm.invoke(msg)
        answer_text = out.content

    return QAResponse(question=question, answer=answer_text)


def answer_question(
    question: str,
    provider: str | None = None,
    model_id: str | None = None,
) -> QAResponse:  # <- en vez de str
    """Convenience entrypoint to answer a single question (FAISS-first)."""

    logger.info("answer_question called with question=%r", question)

    if provider:
        os.environ["MODEL_PROVIDER"] = provider.lower()
        logger.debug("Overriding MODEL_PROVIDER -> %s", provider.lower())
    if model_id:
        os.environ["GEN_MODEL_ID"] = model_id
        os.environ["OLLAMA_MODEL_ID"] = model_id
        logger.debug("Overriding model id -> %s", model_id)

    # 1) Retrieve with FAISS
    docs = retrieve_for_question(question)
    if not docs:
        logger.warning("Retriever returned 0 docs; answering defensively.")
        return QAResponse(
            question=question,
            answer=(
                "No encontré suficiente información en la base de conocimiento "
                "para responder con precisión."
            ),
            used_sources=[],
        )

    # 2) Build context
    ctx = build_context_all(docs, limit_chars=settings.context.max_context_chars_qa)

    # 3) Ask the model
    llm = _ensure_llm()
    msg = P_QA.format(context=ctx, q=question)
    try:
        out = llm.invoke(msg)
        answer_text = out.content
    except ResourceExhausted:
        logger.warning("ResourceExhausted in answer_question; retrying once.")
        out = llm.invoke(msg)
        answer_text = out.content

    # Optional: fill used_sources from docs metadata, only include web URLs
    sources: List[str] = []
    for d in docs:
        src = getattr(d, "metadata", {}).get("source")
        if src and src not in sources:
            # Only include if it looks like a web URL
            if isinstance(src, str) and (src.startswith("http://") or src.startswith("https://")):
                sources.append(src)

    return QAResponse(
        question=question,
        answer=answer_text,
        used_sources=sources,
    )
