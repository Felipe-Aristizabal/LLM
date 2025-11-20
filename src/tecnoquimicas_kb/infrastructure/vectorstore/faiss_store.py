"""FAISS vector store utilities (build, load, search).

This module centralizes FAISS index handling using sentence-transformers
embeddings (multilingual). It supports MMR retrieval and metadata sync.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

# Small in-memory caches
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_INDEX_CACHE: Dict[str, faiss.IndexFlatIP] = {}
_META_CACHE: Dict[str, List[Dict[str, Any]]] = {}

# ---------------- Embeddings ----------------


def _get_embedder(model_name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer model."""
    # NOTE: This loads model once per process; it's fast next calls.
    if model_name not in _MODEL_CACHE:
        # tip: 'intfloat/multilingual-e5-base' is good for ES; use "query: " prefix.
        _MODEL_CACHE[model_name] = SentenceTransformer(
            model_name, trust_remote_code=True
        )
    return _MODEL_CACHE[model_name]


def embed_queries(texts: List[str], model_name: str) -> np.ndarray:
    """Embed queries with instruction-style prefix if model expects it."""
    model = _get_embedder(model_name)
    # e5 expects "query: " prefix for queries (and "passage: " for docs)
    q = [f"query: {t}" for t in texts]
    vecs = model.encode(q, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")


def embed_passages(texts: List[str], model_name: str) -> np.ndarray:
    """Embed documents/chunks as passages."""
    model = _get_embedder(model_name)
    p = [f"passage: {t}" for t in texts]
    vecs = model.encode(p, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")


# ---------------- Build / Load ----------------


def build_index(
    docs: List[Tuple[str, Dict[str, Any]]],
    index_path: Path,
    meta_path: Path,
    model_name: str,
) -> None:
    """Build a fresh FAISS IP index + metadata file from docs.

    Parameters
    ----------
    docs : list of (text, metadata)
    """
    # Flatten documents into chunks (here we use whole docs; you can chunk upstream)
    texts = [t for t, _ in docs]
    metas = [m for _, m in docs]

    if not texts:
        raise ValueError("No documents to index.")

    emb = embed_passages(texts, model_name)  # [N, D]
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # Save index + metadata
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)


def _load_index(index_path: Path) -> faiss.IndexFlatIP:
    """Load FAISS index into memory (cached)."""
    key = str(index_path.resolve())
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = faiss.read_index(str(index_path))
    return _INDEX_CACHE[key]


def _load_meta(meta_path: Path) -> List[Dict[str, Any]]:
    """Load aligned metadata list (cached)."""
    key = str(meta_path.resolve())
    if key not in _META_CACHE:
        with meta_path.open("r", encoding="utf-8") as f:
            _META_CACHE[key] = json.load(f)
    return _META_CACHE[key]


def ensure_index(
    docs: List[Tuple[str, Dict[str, Any]]],
    index_path: Path,
    meta_path: Path,
    model_name: str,
) -> None:
    """Create index if missing; otherwise keep existing."""
    if not index_path.exists() or not meta_path.exists():
        build_index(docs, index_path, meta_path, model_name)


# ---------------- Search (MMR) ----------------


def search(
    query: str,
    index_path: Path,
    meta_path: Path,
    model_name: str,
    top_k: int = 8,
    fetch_k: int = 20,
    use_mmr: bool = True,
    lambda_mmr: float = 0.5,
) -> List[Tuple[int, float]]:
    """Return [(doc_idx, score)] using cosine/IP similarity + optional MMR.

    Notes
    -----
    - We assume normalized embeddings → IP == cosine.
    - If use_mmr, we re-rank fetch_k candidates with greedy MMR.
    """
    index = _load_index(index_path)
    meta = _load_meta(meta_path)
    if index.ntotal == 0:
        return []

    q = embed_queries([query], model_name)  # [1, D]
    # initial ANN
    k = min(fetch_k if use_mmr else top_k, index.ntotal)
    scores, idxs = index.search(q, k)  # [1, k]
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    if not use_mmr:
        return list(zip(idxs[:top_k], scores[:top_k]))

    # --- MMR (greedy) ---
    picked: List[int] = []
    picked_scores: List[float] = []
    candidates = idxs[:k]

    # Pre-compute doc vectors to compute diversity
    # (We only need vectors for candidate docs.)
    # For simplicity, re-embed candidate passages (cheap at k<=50).
    # If you want, persist passage embeddings to disk to avoid this.
    doc_texts = [meta[i].get("raw_text", meta[i].get("title", "")) for i in candidates]
    # Fallback: we may not have raw_text in metadata -> better to persist
    # the full original text as 'raw_text' when indexing.
    doc_embs = embed_passages(doc_texts, model_name)

    query_vec = q[0]
    cand_scores = np.array(scores, dtype="float32")

    # select first = highest similarity
    first = int(np.argmax(cand_scores))
    picked.append(candidates[first])
    picked_scores.append(cand_scores[first])

    # greedy selection
    remaining = [i for j, i in enumerate(candidates) if j != first]
    remaining_embs = np.delete(doc_embs, first, axis=0)

    while len(picked) < min(top_k, len(candidates)) and len(remaining) > 0:
        # similarity to query
        sim_to_query = cand_scores[[idxs.index(r) for r in remaining]]
        # diversity: max sim with already picked
        picked_embs = doc_embs[[idxs.index(p) for p in picked]]
        # compute max similarity for each remaining doc to picked set
        div = []
        for e in remaining_embs:
            # cosine with picked_embs
            sims = np.dot(picked_embs, e)
            div.append(np.max(sims) if len(sims) else 0.0)
        div = np.array(div)

        mmr_score = lambda_mmr * sim_to_query + (1.0 - lambda_mmr) * (1.0 - div)
        j = int(np.argmax(mmr_score))
        picked.append(remaining[j])
        picked_scores.append(float(sim_to_query[j]))

        # remove chosen
        remaining.pop(j)
        remaining_embs = np.delete(remaining_embs, j, axis=0)

    return list(zip(picked, picked_scores))
