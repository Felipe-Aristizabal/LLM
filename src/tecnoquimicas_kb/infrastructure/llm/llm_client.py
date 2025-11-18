"""LLM client helpers: provider selection and retry logic.

This module exposes a single, central `get_llm` function that should be
used by the rest of the codebase to obtain a chat model. It hides the
details of provider selection and model identifiers, using environment
variables and `config.settings` as the source of truth.

It also provides a small `invoke_with_retry` helper to apply a simple
retry policy around quota-related errors from Google APIs.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from google.api_core.exceptions import ResourceExhausted
from langchain_core.language_models.chat_models import BaseChatModel

from tecnoquimicas_kb.config.settings import settings
from tecnoquimicas_kb.infrastructure.llm.providers import (
    create_google_llm,
    create_ollama_llm,
)


def _resolve_provider(explicit: Optional[str] = None) -> str:
    """Resolve the active LLM provider with backward compatibility.

    The provider is resolved using this priority:
    1. Explicit argument passed to the function.
    2. Environment variable MODEL_PROVIDER.
    3. Value from settings.llm.provider.
    """
    if explicit:
        return explicit.lower()

    env_value = os.getenv("MODEL_PROVIDER")
    if env_value:
        return env_value.lower()

    return settings.llm.provider.lower()


def _resolve_model_id(provider: str, explicit: Optional[str] = None) -> str:
    """Resolve the model identifier for the given provider.

    The model id is resolved using this priority:
    1. Explicit argument passed to the function.
    2. Provider-specific environment variable (GEN_MODEL_ID or OLLAMA_MODEL_ID).
    3. Value from settings.llm.
    """
    if explicit:
        return explicit

    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL_ID", settings.llm.ollama_model_id)

    # Default branch for Google Gemini and any unknown value.
    return os.getenv("GEN_MODEL_ID", settings.llm.google_model_id)


def get_llm(
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
) -> BaseChatModel:
    """Return a chat model based on configuration and optional overrides.

    Parameters
    ----------
    provider:
        Optional provider name. Valid values include 'gemini' and 'ollama'.
        When omitted, configuration and environment variables are used.
    model_id:
        Optional model identifier. When omitted, configuration and
        provider-specific defaults are used.

    Returns
    -------
    BaseChatModel
        An instantiated chat model ready to be used with `.invoke()`.

    Raises
    ------
    ValueError
        If the provider is not recognized.
    ImportError
        If the required library for a provider is not installed.
    """
    resolved_provider = _resolve_provider(provider)
    resolved_model = _resolve_model_id(resolved_provider, model_id)

    if resolved_provider == "ollama":
        return create_ollama_llm(resolved_model)

    if resolved_provider == "gemini":
        return create_google_llm(resolved_model)

    # Unknown provider: fail fast to avoid confusing behaviour.
    raise ValueError(f"Unsupported LLM provider: {resolved_provider!r}")


def invoke_with_retry(
    llm: BaseChatModel,
    prompt: Any,
    max_retries: int = 2,
    delay_seconds: float = 6.0,
) -> Any:
    """Invoke a chat model with a simple retry policy.

    Parameters
    ----------
    llm:
        Chat model instance that implements `.invoke()`.
    prompt:
        Prompt or message object accepted by the underlying model.
    max_retries:
        Maximum number of attempts in total (including the first one).
    delay_seconds:
        Delay between retries in seconds when a retryable error occurs.

    Returns
    -------
    Any
        The result returned by `llm.invoke(prompt)`.

    Raises
    ------
    Exception
        Re-raises the last exception if all retries fail.
    """
    attempt = 0
    while True:
        try:
            return llm.invoke(prompt)
        except ResourceExhausted as exc:
            attempt += 1
            # Only retry for quota/limit errors, respecting the max_retries
            if attempt >= max_retries:
                raise exc
            time.sleep(delay_seconds)
        except Exception:
            # For unexpected errors, do not retry to avoid masking bugs.
            raise
