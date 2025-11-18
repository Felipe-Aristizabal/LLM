"""Application-wide configuration and environment settings.

This module centralizes access to environment variables and default paths
used across the Tecnoquímicas RAG project. Import the `settings` object
instead of calling `os.getenv` from many different places.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()


def _env_int(name: str, default: int) -> int:
    """Return an integer environment variable with a safe fallback.

    If the variable is not set or cannot be converted to an integer,
    the provided default value is returned.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_path(name: str, default: str) -> Path:
    """Return a Path built from an environment variable or a default value."""
    return Path(os.getenv(name, default)).expanduser().resolve()


@dataclass(frozen=True)
class LLMSettings:
    """Configuration options related to large language models."""

    provider: Literal["gemini", "ollama"]
    google_model_id: str
    ollama_model_id: str


@dataclass(frozen=True)
class ContextSettings:
    """Limits and heuristics used for context stuffing and ranking."""

    max_context_chars_all: int
    max_context_chars_qa: int
    default_k_files_qa: int
    default_k_files_sum: int
    default_k_files_faq: int


@dataclass(frozen=True)
class PathsSettings:
    """Filesystem paths used by the RAG pipeline."""

    data_clean_dir: Path
    data_raw_dir: Path
    index_dir: Path


@dataclass(frozen=True)
class Settings:
    """Top-level immutable container for all settings."""

    llm: LLMSettings
    context: ContextSettings
    paths: PathsSettings


def load_settings() -> Settings:
    """Load all settings from the environment and return a Settings instance."""
    # LLM provider selection; defaults match the existing legacy code.
    provider = os.getenv("MODEL_PROVIDER", "gemini").lower()
    if provider not in ("gemini", "ollama"):
        provider = "gemini"

    llm = LLMSettings(
        provider=provider,  # type: ignore[arg-type]
        google_model_id=os.getenv("GEN_MODEL_ID", "gemini-2.5-pro"),
        ollama_model_id=os.getenv("OLLAMA_MODEL_ID", "gemma3:4b"),
    )

    # Limits and ranking heuristics used by stuffing-style RAG.
    context = ContextSettings(
        max_context_chars_all=_env_int("MAX_CONTEXT_CHARS_ALL", 60000),
        max_context_chars_qa=_env_int("MAX_CONTEXT_CHARS_QA", 40000),
        default_k_files_qa=_env_int("DEFAULT_K_FILES_QA", 15),
        default_k_files_sum=_env_int("DEFAULT_K_FILES_SUM", 12),
        default_k_files_faq=_env_int("DEFAULT_K_FILES_FAQ", 15),
    )

    # Default filesystem layout for data and index artifacts.
    paths = PathsSettings(
        data_clean_dir=_env_path(
            "TQ_DATA_CLEAN_DIR", "src/tecnoquimicas_kb/data/clean"
        ),
        data_raw_dir=_env_path("TQ_DATA_RAW_DIR", "src/tecnoquimicas_kb/data/raw"),
        index_dir=_env_path("TQ_INDEX_DIR", "src/tecnoquimicas_kb/index/faiss"),
    )

    return Settings(llm=llm, context=context, paths=paths)


# Singleton-style settings instance that can be imported by other modules
settings: Settings = load_settings()
