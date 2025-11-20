"""Application-wide configuration and environment settings.

This module centralizes access to environment variables and default paths
used across the Tecnoquímicas project. Import the `settings` object
instead of calling `os.getenv` in multiple places.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()


# ----------------------------- helpers ------------------------------------ #
def _env_int(name: str, default: int) -> int:
    """Return an integer environment variable with a safe fallback."""
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


def bool_from_env(name: str, default: bool = False) -> bool:
    """Parse a boolean feature flag from environment variables."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_provider(raw: str) -> Literal["google", "ollama"]:
    """Normalize provider aliases to a stable set of literals.

    Accepts: "google", "gemini" -> "google"
              "ollama" -> "ollama"
    """
    val = (raw or "").strip().lower()
    if val in {"google", "gemini"}:
        return "google"
    return "ollama"


# ------------------------------ dataclasses -------------------------------- #
@dataclass(frozen=True)
class LLMSettings:
    """Configuration options related to large language models."""

    provider: Literal["google", "ollama"]
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


# ---- NEW: Vector / FAISS configuration ----------------------------------- #
@dataclass(frozen=True)
class VectorConfig:
    """FAISS / embeddings configuration."""

    # Index file (will be created if missing)
    faiss_index_path: Path
    # Where we persist doc metadata aligned with the index
    faiss_meta_path: Path
    # Sentence-Transformers model (multilingual E5 works well for ES)
    embedding_model_name: str
    # Retrieval settings
    top_k: int
    fetch_k: int
    use_mmr: bool
    score_threshold: float


# ---- NEW: Agent (router, compose, memory) configuration ------------------ #
@dataclass(frozen=True)
class AgentConfig:
    """Agent settings controlled by env flags."""

    use_router: bool  # enable LLM/heuristic router
    allow_compose: bool  # enable COMPOSE (merge RAG + facts)
    max_history_messages: int  # memory window for chat history
    followup_max_lookback: int  # how far to look for follow-up cues


@dataclass(frozen=True)
class Settings:
    """Top-level immutable container for all settings."""

    llm: LLMSettings
    context: ContextSettings
    paths: PathsSettings
    vector: VectorConfig  # NEW
    agent: AgentConfig  # NEW


# ------------------------------ loader ------------------------------------ #
def load_settings() -> Settings:
    """Load all settings from environment variables and return a Settings instance."""
    # --- LLM provider selection
    provider = _normalize_provider(os.getenv("MODEL_PROVIDER", "google"))

    llm = LLMSettings(
        provider=provider,
        google_model_id=os.getenv("GEN_MODEL_ID", "gemini-1.5-pro"),
        ollama_model_id=os.getenv("OLLAMA_MODEL_ID", "llama3.1"),
    )

    # --- Context / stuffing limits (legacy-friendly to keep compatibility)
    context = ContextSettings(
        max_context_chars_all=_env_int("MAX_CONTEXT_CHARS_ALL", 60000),
        max_context_chars_qa=_env_int("MAX_CONTEXT_CHARS_QA", 40000),
        default_k_files_qa=_env_int("DEFAULT_K_FILES_QA", 15),
        default_k_files_sum=_env_int("DEFAULT_K_FILES_SUM", 12),
        default_k_files_faq=_env_int("DEFAULT_K_FILES_FAQ", 15),
    )

    # --- Paths layout
    paths = PathsSettings(
        data_clean_dir=_env_path(
            "TQ_DATA_CLEAN_DIR", "src/tecnoquimicas_kb/data/clean"
        ),
        data_raw_dir=_env_path("TQ_DATA_RAW_DIR", "src/tecnoquimicas_kb/data/raw"),
        index_dir=_env_path("TQ_INDEX_DIR", "src/tecnoquimicas_kb/index/faiss"),
    )

    # --- Vector / FAISS config
    vector = VectorConfig(
        faiss_index_path=_env_path(
            "VECTOR_FAISS_INDEX", "data/vector/faiss_index.faiss"
        ),
        faiss_meta_path=_env_path("VECTOR_FAISS_META", "data/vector/faiss_meta.json"),
        embedding_model_name=os.getenv(
            "VECTOR_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
        ),
        top_k=_env_int("VECTOR_TOP_K", 8),
        fetch_k=_env_int("VECTOR_FETCH_K", 20),
        use_mmr=bool_from_env("VECTOR_USE_MMR", True),
        score_threshold=float(os.getenv("VECTOR_SCORE_THRESHOLD", "0.0")),
    )

    # --- Agent config (router / compose / memory window)
    agent = AgentConfig(
        use_router=bool_from_env("AGENT_USE_ROUTER", True),
        allow_compose=bool_from_env("AGENT_ALLOW_COMPOSE", True),
        max_history_messages=_env_int("AGENT_MAX_HISTORY", 10),
        followup_max_lookback=_env_int("AGENT_FOLLOWUP_LOOKBACK", 10),
    )

    return Settings(llm=llm, context=context, paths=paths, vector=vector, agent=agent)


# Singleton-style settings instance that can be imported by other modules
settings: Settings = load_settings()
