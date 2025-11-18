"""Utilities to load text documents and chunks from disk.

This module centralizes JSON/JSONL/TXT loading logic so that both the
RAG services and the indexing code can reuse the same behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tecnoquimicas_kb.config.settings import settings

# Canonical tuple type used by the RAG pipeline
DocTuple = Tuple[str, Dict[str, Any]]


def _clean_text(text: str) -> str:
    """Normalize whitespace in a string and strip leading/trailing spaces."""
    text = text.replace("\r", " ")
    # Collapse any run of whitespace into a single space
    text = " ".join(text.split())
    return text.strip()


def _doc_from_json_obj(obj: Dict[str, Any], source: str) -> DocTuple:
    """Extract text and metadata from a JSON object representing a chunk."""
    text = obj.get("text") or obj.get("content") or ""
    cleaned = _clean_text(str(text))

    # All other keys are treated as metadata
    meta: Dict[str, Any] = {
        k: v for k, v in obj.items() if k not in {"text", "content"}
    }
    meta.setdefault("source", source)
    return cleaned, meta


def load_json_file(path: Path) -> List[DocTuple]:
    """Load documents from a JSON or JSONL file.

    The function is tolerant to both JSONL and normal JSON structures:
    - JSONL: one JSON object per line.
    - JSON: a single object or a list of objects.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    docs: List[DocTuple] = []

    # First, try to interpret the file as JSONL
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # If one line is not valid JSON, just skip that line
            continue

        if isinstance(obj, dict):
            text, meta = _doc_from_json_obj(obj, str(path))
            if text:
                docs.append((text, meta))

    if docs:
        # If we successfully parsed at least one JSONL line, return now
        return docs

    # Otherwise, fall back to a single JSON value
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Invalid JSON content; nothing to load from this file
        return []

    if isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict):
                continue
            text, meta = _doc_from_json_obj(item, str(path))
            if text:
                docs.append((text, meta))
    elif isinstance(obj, dict):
        text, meta = _doc_from_json_obj(obj, str(path))
        if text:
            docs.append((text, meta))

    return docs


def load_txt_file(path: Path) -> List[DocTuple]:
    """Load a plain text file as a single document tuple."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    return [(cleaned, {"source": str(path)})]


def load_all_docs(root_dir: str | Path | None = None) -> List[DocTuple]:
    """Recursively load all documents under a directory.

    Parameters
    ----------
    root_dir:
        Base directory that will be scanned recursively. If omitted,
        the default clean data directory from `settings.paths` is used.
    """
    if root_dir is None:
        root = settings.paths.data_clean_dir
    else:
        root = Path(root_dir).expanduser().resolve()

    out: List[DocTuple] = []

    for fp in root.rglob("*"):
        suffix = fp.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            out.extend(load_json_file(fp))
        elif suffix == ".txt":
            out.extend(load_txt_file(fp))

    return out
