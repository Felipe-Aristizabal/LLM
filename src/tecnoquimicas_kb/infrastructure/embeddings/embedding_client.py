"""Embedding client helper used for index construction and search.

This module exposes a `get_embeddings` factory that returns a LangChain
Embeddings implementation based on environment variables. It mirrors the
behaviour of the original `_get_embeddings` helper used in the legacy
`index_builder.py` script.
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def _resolve_embed_provider(explicit: Optional[str] = None) -> str:
    """Resolve the embedding provider name.

    Priority:
    1. Explicit argument passed to the function.
    2. Environment variable EMBED_PROVIDER.
    3. The literal string 'local' (HuggingFace).
    """
    if explicit:
        return explicit.lower()

    env_value = os.getenv("EMBED_PROVIDER")
    if env_value:
        return env_value.lower()

    return "local"


def _resolve_embed_model_id(
    provider: str,
    explicit: Optional[str] = None,
) -> str:
    """Resolve the embedding model identifier for the given provider."""
    if explicit:
        return explicit

    if provider == "google":
        # Default model recommended for general purpose embeddings.
        return os.getenv("EMBED_MODEL_ID", "text-embedding-004")

    if provider == "ollama":
        return os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # Default model for local HuggingFace embeddings.
    return os.getenv("EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")


def get_embeddings(
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Embeddings:
    """Return an embedding model instance based on configuration.

    Parameters
    ----------
    provider:
        Optional provider name: 'google', 'ollama' or 'local'.
    model_id:
        Optional embedding model identifier; if omitted, a default is
        chosen based on the provider.

    Returns
    -------
    Embeddings
        A LangChain Embeddings implementation suitable for FAISS.
    """
    resolved_provider = _resolve_embed_provider(provider)
    resolved_model = _resolve_embed_model_id(resolved_provider, model_id)

    if resolved_provider == "google":
        return GoogleGenerativeAIEmbeddings(model=resolved_model)

    if resolved_provider == "ollama":
        return OllamaEmbeddings(model=resolved_model)

    # Fallback to local HuggingFace embeddings for 'local' or unknown.
    return HuggingFaceEmbeddings(model_name=resolved_model)
