"""Unified retriever that ALWAYS goes through FAISS.

This keeps RAG consistent and prevents "sosas" answers by ensuring
we fetch the most relevant passages first.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from tecnoquimicas_kb.config.settings import settings
from tecnoquimicas_kb.domain.models import Document
from tecnoquimicas_kb.infrastructure.ingestion.file_loader import load_all_docs
from tecnoquimicas_kb.infrastructure.vectorstore.faiss_store import (
    ensure_index,
    search,
)


def _load_corpus() -> List[Tuple[str, Dict]]:
    """Load the cleaned KB and return legacy tuples (text, metadata)."""
    tuples = load_all_docs(settings.paths.data_clean_dir)
    # Enrich metadata with raw_text to support MMR properly.
    enriched = []
    for text, meta in tuples:
        m = dict(meta)
        m.setdefault("raw_text", text)
        enriched.append((text, m))
    return enriched


# Simple module-level cache
_CORPUS: List[Tuple[str, Dict]] | None = None


def _get_corpus() -> List[Tuple[str, Dict]]:
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = _load_corpus()
    return _CORPUS


def ensure_faiss_ready() -> None:
    """Build FAISS index if missing (idempotent)."""
    corpus = _get_corpus()
    ensure_index(
        corpus,
        settings.vector.faiss_index_path,
        settings.vector.faiss_meta_path,
        settings.vector.embedding_model_name,
    )


def retrieve_for_question(question: str) -> List[Document]:
    """Return top-k Document objects for a question using FAISS+MMR."""
    ensure_faiss_ready()

    hits = search(
        query=question,
        index_path=settings.vector.faiss_index_path,
        meta_path=settings.vector.faiss_meta_path,
        model_name=settings.vector.embedding_model_name,
        top_k=settings.vector.top_k,
        fetch_k=settings.vector.fetch_k,
        use_mmr=settings.vector.use_mmr,
    )
    # Reconstruct Documents from corpus + meta
    corpus = _get_corpus()
    docs: List[Document] = []
    for idx, _score in hits:
        text, meta = corpus[idx]
        docs.append(Document(text=text, metadata=meta))
    return docs
