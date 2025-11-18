"""Provider-specific factory functions for chat language models.

This module contains small helpers to construct chat models for each
supported provider (e.g. Google Gemini, Ollama). Higher-level code
should usually import and use `get_llm` from `llm_client.py` instead.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

try:
    # Google Gemini chat model
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore[assignment]

try:
    # Ollama local chat model
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover
    ChatOllama = None  # type: ignore[assignment]


def create_google_llm(model_id: str, **kwargs) -> BaseChatModel:
    """Create a Google Gemini chat model instance.

    Parameters
    ----------
    model_id:
        Identifier of the Gemini model (for example 'gemini-2.5-pro').

    Raises
    ------
    ImportError
        If the `langchain_google_genai` package is not installed.
    """
    if ChatGoogleGenerativeAI is None:  # pragma: no cover
        raise ImportError(
            "langchain_google_genai is required to use Google Gemini models."
        )

    # Additional keyword arguments are forwarded to the underlying client.
    return ChatGoogleGenerativeAI(model=model_id, **kwargs)


def create_ollama_llm(model_id: str, **kwargs) -> BaseChatModel:
    """Create an Ollama chat model instance.

    Parameters
    ----------
    model_id:
        Identifier of the local Ollama model (for example 'gemma3:4b').

    Raises
    ------
    ImportError
        If the `langchain_ollama` package is not installed.
    """
    if ChatOllama is None:  # pragma: no cover
        raise ImportError("langchain_ollama is required to use Ollama chat models.")

    return ChatOllama(model=model_id, **kwargs)
