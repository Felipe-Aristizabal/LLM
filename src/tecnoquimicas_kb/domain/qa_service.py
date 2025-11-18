"""High-level QA services for the Tecnoquímicas RAG assistant.

This module exposes functions such as `answer_question` and helpers for
generating onboarding summaries or FAQ lists. Internally, it reuses the
legacy stuffing-based implementations to keep behaviour consistent while
providing a cleaner entrypoint for other parts of the system.
"""

from __future__ import annotations

import os
from typing import List, Sequence

from google.api_core.exceptions import ResourceExhausted
from langchain_core.language_models.chat_models import BaseChatModel

from tecnoquimicas_kb.config.settings import settings
from tecnoquimicas_kb.domain.context_builder import (
    build_context_all,
    build_context_for_question,
)
from tecnoquimicas_kb.domain.models import DocLike, Document, QARequest, QAResponse
from tecnoquimicas_kb.domain.prompts import P_FAQ, P_QA, P_SUMMARY

# Import legacy implementations to delegate most of the work for now.
# This allows gradual refactoring without breaking existing behaviour.
try:  # pragma: no cover - thin wrappers
    from tecnoquimicas_kb.rag.stuffing import (  # type: ignore[import]
        load_all_docs as _legacy_load_all_docs,
        summarize_stuffing as _legacy_summarize_stuffing,
        faq_stuffing as _legacy_faq_stuffing,
        qa_stuffing as _legacy_qa_stuffing,
        get_llm as _legacy_get_llm,
    )
except ImportError:  # pragma: no cover
    _legacy_load_all_docs = None
    _legacy_summarize_stuffing = None
    _legacy_faq_stuffing = None
    _legacy_qa_stuffing = None
    _legacy_get_llm = None


def _ensure_llm() -> BaseChatModel:
    """Return a chat model instance based on the configured provider.

    When the legacy `get_llm` function is available, it is used directly.
    Otherwise a simple provider switch is implemented locally.
    """
    if _legacy_get_llm is not None:
        return _legacy_get_llm()

    # Local fallback keeps compatibility with the environment variables
    # used in the legacy code.
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_ollama import ChatOllama

    provider = settings.llm.provider
    if provider == "ollama":
        return ChatOllama(model=settings.llm.ollama_model_id)

    # Default to Google Gemini models
    return ChatGoogleGenerativeAI(model=settings.llm.google_model_id)


def _normalize_docs(docs: Sequence[DocLike]) -> List[Document]:
    """Convert DocLike items into Document instances."""
    normalized: List[Document] = []
    for item in docs:
        if isinstance(item, Document):
            normalized.append(item)
        else:
            text, meta = item
            normalized.append(Document(text=str(text), metadata=dict(meta)))
    return normalized


def summarize_for_onboarding(docs: Sequence[DocLike]) -> str:
    """Generate an onboarding summary for a new client based on documents."""
    if not docs:
        return ""

    if _legacy_summarize_stuffing is not None:
        # Use the existing, tuned implementation when available.
        legacy_docs = [(d.text, d.metadata) for d in _normalize_docs(docs)]
        return _legacy_summarize_stuffing(legacy_docs)

    # Fallback: build context and call the LLM directly.
    ctx = build_context_all(docs)
    llm = _ensure_llm()
    msg = P_SUMMARY.format(context=ctx)

    try:
        out = llm.invoke(msg)
        return out.content
    except ResourceExhausted:
        # On quota errors, apply a simple retry once.
        out = llm.invoke(msg)
        return out.content


def generate_faqs(docs: Sequence[DocLike], n: int = 10) -> str:
    """Generate a FAQ list for clients based on the provided documents."""
    if not docs:
        return ""

    if _legacy_faq_stuffing is not None:
        legacy_docs = [(d.text, d.metadata) for d in _normalize_docs(docs)]
        return _legacy_faq_stuffing(legacy_docs, n=n)

    ctx = build_context_all(docs)
    llm = _ensure_llm()
    msg = P_FAQ.format(context=ctx, n=n)

    try:
        out = llm.invoke(msg)
        return out.content
    except ResourceExhausted:
        out = llm.invoke(msg)
        return out.content


def answer_question_from_docs(
    question: str,
    docs: Sequence[DocLike],
) -> QAResponse:
    """Answer a question using the supplied documents as context.

    This function does not perform any I/O; all documents must be passed
    explicitly by the caller.
    """
    if not docs:
        return QAResponse(question=question, answer="")

    if _legacy_qa_stuffing is not None:
        legacy_docs = [(d.text, d.metadata) for d in _normalize_docs(docs)]
        answer_text = _legacy_qa_stuffing(legacy_docs, question)
        # The legacy function does not expose per-source info, so only
        # the question and answer are returned here.
        return QAResponse(question=question, answer=answer_text)

    ctx = build_context_for_question(question, docs)
    llm = _ensure_llm()
    msg = P_QA.format(context=ctx, q=question)

    try:
        out = llm.invoke(msg)
        answer_text = out.content
    except ResourceExhausted:
        out = llm.invoke(msg)
        answer_text = out.content

    return QAResponse(question=question, answer=answer_text)


def answer_question(
    question: str,
    provider: str | None = None,
    model_id: str | None = None,
) -> str:
    """High-level helper used by CLI tools to answer a single question.

    This function is opinionated and convenient:
    - It optionally overrides the model provider and model id by setting
      the corresponding environment variables.
    - It loads all documents from the configured `data_clean_dir`.
    - It delegates to the legacy stuffing-based QA implementation.
    """
    # Optionally override provider and model id for the duration of this call.
    if provider:
        os.environ["MODEL_PROVIDER"] = provider.lower()
    if model_id:
        # Both variables are set so that either provider can consume it.
        os.environ["GEN_MODEL_ID"] = model_id
        os.environ["OLLAMA_MODEL_ID"] = model_id

    if _legacy_load_all_docs is None or _legacy_qa_stuffing is None:
        # If the legacy code is not available, we cannot implement this
        # helper in a meaningful way, so we fall back to an empty answer.
        return ""

    # Load all documents from the configured clean data directory.
    docs = _legacy_load_all_docs(str(settings.paths.data_clean_dir))

    # Delegate to the existing implementation, which already handles
    # building the context and calling the LLM with the right prompts.
    answer_text = _legacy_qa_stuffing(docs, question)
    return answer_text
