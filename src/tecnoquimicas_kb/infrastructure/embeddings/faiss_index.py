"""FAISS index utilities for Tecnoquímicas RAG.

This module provides small helpers to:
- Load chunked documents from disk.
- Build and save a FAISS vector index.
- Load an existing FAISS index and run similarity search.

The design is intentionally thin so that CLI scripts can call
`build_faiss_index` while the rest of the system can re-use the loader
for more advanced flows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from tecnoquimicas_kb.infrastructure.embeddings.embedding_client import (
    get_embeddings,
)


def _to_doc(text: str, metadata: Dict[str, Any]) -> Optional[Document]:
    """Convert raw text and metadata into a LangChain Document.

    Empty or whitespace-only strings are converted to `None` so that the
    caller can easily filter them out.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    return Document(page_content=cleaned, metadata=metadata)


def _load_txt_file(path: Path) -> List[Document]:
    """Load a plain text file as one or more Documents.

    For simplicity, the entire file content is stored as a single
    Document. More advanced splitting should happen earlier in the
    ingestion pipeline (e.g. during chunking).
    """
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Best-effort fallback for unexpected encodings.
        content = path.read_text(errors="ignore")

    meta: Dict[str, Any] = {"source": str(path)}
    doc = _to_doc(content, meta)
    return [doc] if doc else []


def _extract_text_and_meta(obj: Dict[str, Any]) -> Optional[Document]:
    """Extract text and metadata from a JSON object representing a chunk.

    The function supports multiple common key names for the text field:
    - 'text'
    - 'content'
    - 'page_content'
    """
    text = (
        obj.get("text")
        or obj.get("content")
        or obj.get("page_content")
        or ""
    )

    meta = {
        k: v
        for k, v in obj.items()
        if k not in {"text", "content", "page_content"}
    }
    return _to_doc(text, meta)


def _load_json_file(path: Path) -> List[Document]:
    """Load a JSON or JSONL file into a list of Documents."""
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(errors="ignore")

    docs: List[Document] = []

    try:
        # Heuristic: if the file contains many lines, treat it as JSONL.
        if "\n" in raw.strip():
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines instead of aborting the load.
                    continue

                if isinstance(obj, dict):
                    doc = _extract_text_and_meta(obj)
                    if doc:
                        docs.append(doc)
        else:
            # Single JSON value: either an object or a list of objects.
            obj = json.loads(raw)
            if isinstance(obj, list):
                for item in obj:
                    if not isinstance(item, dict):
                        continue
                    doc = _extract_text_and_meta(item)
                    if doc:
                        docs.append(doc)
            elif isinstance(obj, dict):
                doc = _extract_text_and_meta(obj)
                if doc:
                    docs.append(doc)
    except json.JSONDecodeError:
        # As a last resort, try parsing line by line as JSONL.
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                doc = _extract_text_and_meta(obj)
                if doc:
                    docs.append(doc)
    except Exception:
        # Unexpected errors are swallowed here to avoid failing the
        # entire indexing operation due to a single problematic file.
        return []

    return docs


def _load_chunks_recursive(root_dir: Path) -> List[Document]:
    """Recursively scan a directory for chunk files and load them."""
    docs: List[Document] = []
    for path in root_dir.rglob("*"):
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            docs.extend(_load_json_file(path))
        elif suffix == ".txt":
            docs.extend(_load_txt_file(path))

    # Filter out any `None` values that might have slipped through.
    return [doc for doc in docs if doc]


def build_faiss_index(
    data_dir: str | Path,
    index_dir: str | Path,
) -> int:
    """Build a FAISS index from documents found under `data_dir`.

    Parameters
    ----------
    data_dir:
        Directory containing chunk files (.json, .jsonl, .txt).
    index_dir:
        Directory where the FAISS index will be saved. The directory is
        created if it does not exist.

    Returns
    -------
    int
        Number of documents that were indexed.

    Raises
    ------
    RuntimeError
        If no chunk files are found in the given directory.
    """
    data_path = Path(data_dir).expanduser().resolve()
    index_path = Path(index_dir).expanduser().resolve()
    index_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Searching chunks in: {data_path}")
    docs = _load_chunks_recursive(data_path)
    print(f"[INFO] Loaded documents: {len(docs)}")

    if not docs:
        raise RuntimeError(
            f"No chunks found under {data_path}. "
            "Check folder structure and file extensions."
        )

    embeddings = get_embeddings()
    print(f"[INFO] Embeddings provider: {os.getenv('EMBED_PROVIDER', 'local')}")
    vs = FAISS.from_documents(docs, embeddings)

    vs.save_local(str(index_path))
    print(f"[OK] Index saved to: {index_path}")
    return len(docs)


def load_faiss_index(index_dir: str | Path) -> FAISS:
    """Load an existing FAISS index from the given directory."""
    index_path = Path(index_dir).expanduser().resolve()
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def search_index(
    index: FAISS,
    query: str,
    k: int = 5,
) -> List[Document]:
    """Run a similarity search over a loaded FAISS index.

    Parameters
    ----------
    index:
        FAISS vector store instance.
    query:
        Natural language query string.
    k:
        Number of top documents to return.

    Returns
    -------
    list[Document]
        List of retrieved documents ordered by similarity.
    """
    return index.similarity_search(query, k=k)
