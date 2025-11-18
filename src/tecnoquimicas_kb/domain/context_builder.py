"""Context building utilities for stuffing-style RAG.

This module provides helpers to transform a collection of documents into
a single stuffed context string, either by concatenating everything or
by selecting only the most relevant sources for a given question.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

from tecnoquimicas_kb.config.settings import settings
from tecnoquimicas_kb.domain.models import DocLike, Document

# Try to reuse the battle-tested implementations from the legacy module
# while still exposing a clean domain-level API.
try:  # pragma: no cover - thin wrapper
    from tecnoquimicas_kb.rag.stuffing import (  # type: ignore[import]
        build_context_all as _legacy_build_context_all,
        build_context_for_question as _legacy_build_context_for_question,
    )
except ImportError:  # pragma: no cover
    _legacy_build_context_all = None
    _legacy_build_context_for_question = None


def _normalize_docs(docs: Sequence[DocLike]) -> List[Tuple[str, Dict]]:
    """Convert DocLike instances into the legacy (text, metadata) tuples.

    This helper allows the new domain layer to interoperate with existing
    utilities that expect a simple tuple representation.
    """
    normalized: List[Tuple[str, Dict]] = []
    for item in docs:
        if isinstance(item, Document):
            normalized.append((item.text, dict(item.metadata)))
        else:
            text, meta = item
            normalized.append((str(text), dict(meta)))
    return normalized


def _score(question_tokens: Iterable[str], text: str) -> int:
    """Compute a very simple relevance score for a piece of text.

    The score is defined as the number of distinct question tokens that
    appear at least once in the candidate text.
    """
    score = 0
    for token in question_tokens:
        if token and token in text:
            score += 1
    return score


def build_context_all(
    docs: Sequence[DocLike],
    limit_chars: int | None = None,
) -> str:
    """Concatenate all documents into a single context string.

    Parameters
    ----------
    docs:
        Iterable of documents or `(text, metadata)` tuples.
    limit_chars:
        Optional maximum number of characters. If omitted, the default
        limit from `settings.context.max_context_chars_all` is used.
    """
    if not docs:
        return ""

    legacy_docs = _normalize_docs(docs)
    effective_limit = limit_chars or settings.context.max_context_chars_all

    if _legacy_build_context_all is not None:
        # Delegate to the existing implementation to avoid behaviour drift.
        return _legacy_build_context_all(legacy_docs, limit_chars=effective_limit)

    # Fallback implementation mirroring the legacy behaviour.
    parts: List[str] = []
    total = 0

    for text, meta in legacy_docs:
        source = meta.get("url") or meta.get("source", "desconocida")
        block = f"[FUENTE: {source}]\n{text}\n\n"
        if total + len(block) > effective_limit:
            remaining = effective_limit - total
            if remaining > 0:
                parts.append(block[:remaining])
            break
        parts.append(block)
        total += len(block)

    return "".join(parts)


def build_context_for_question(
    question: str,
    docs: Sequence[DocLike],
    k_files: int | None = None,
    limit_chars: int | None = None,
) -> str:
    """Build a stuffed context focused on a single question.

    Documents are grouped by source, scored using a simple token overlap
    heuristic, and only the top-k sources are concatenated until the
    character budget is exhausted.

    Parameters
    ----------
    question:
        User question used to rank sources.
    docs:
        Iterable of documents or `(text, metadata)` tuples.
    k_files:
        Maximum number of sources to include. Defaults to the value from
        `settings.context.default_k_files_qa`.
    limit_chars:
        Maximum number of characters allowed in the final context. If
        omitted, `settings.context.max_context_chars_qa` is used.
    """
    if not docs:
        return ""

    legacy_docs = _normalize_docs(docs)
    effective_k = k_files or settings.context.default_k_files_qa
    effective_limit = limit_chars or settings.context.max_context_chars_qa

    if _legacy_build_context_for_question is not None:
        # Delegate to the existing implementation when available.
        return _legacy_build_context_for_question(
            question,
            legacy_docs,
            k_files=effective_k,
            limit_chars=effective_limit,
        )

    # Fallback implementation mirroring the legacy ranking.
    tokens = {token.lower() for token in re.findall(r"[A-Za-zÀ-ÿ0-9_]+", question)}

    # Group chunks by source identifier.
    by_src: Dict[str, List[str]] = {}
    for text, meta in legacy_docs:
        src = meta.get("url") or meta.get("source", "desconocida")
        by_src.setdefault(src, []).append(text)

    scored: List[Tuple[str, int]] = []
    for src, texts in by_src.items():
        joined = " ".join(texts)
        scored.append((src, _score(tokens, joined)))

    # Sort sources by score in descending order.
    scored.sort(key=lambda item: item[1], reverse=True)

    # Select the top-k sources with a positive score, or fall back to
    # the first one if there were no matches.
    top_sources = [src for src, s in scored[:effective_k] if s > 0]
    if not top_sources and scored:
        top_sources = [scored[0][0]]

    parts: List[str] = []
    total = 0

    for src in top_sources:
        block = f"[FUENTE: {src}]\n" + " ".join(by_src[src]) + "\n\n"
        if total + len(block) > effective_limit:
            remaining = effective_limit - total
            if remaining > 0:
                parts.append(block[:remaining])
            break
        parts.append(block)
        total += len(block)

    # If we still have no content, fall back to concatenating everything.
    if not parts:
        return build_context_all(docs, limit_chars=effective_limit)

    return "".join(parts)
