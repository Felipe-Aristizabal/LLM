"""Helper functions to resolve canonical data and index paths."""

from __future__ import annotations

from pathlib import Path

from tecnoquimicas_kb.config.settings import settings


def raw_base_dir() -> Path:
    """Return the base directory for raw data."""
    return settings.paths.data_raw_dir


def raw_html_dir() -> Path:
    """Return the directory where raw HTML files are stored."""
    return raw_base_dir() / "raw_html"


def clean_base_dir() -> Path:
    """Return the base directory for cleaned data and chunks."""
    return settings.paths.data_clean_dir


def clean_text_dir() -> Path:
    """Return the directory where cleaned text files are stored."""
    return clean_base_dir() / "clean_text"


def chunks_dir() -> Path:
    """Return the directory where JSON/JSONL chunks are stored."""
    return clean_base_dir() / "chunks"


def index_dir() -> Path:
    """Return the directory where vector indexes are stored."""
    return settings.paths.index_dir
