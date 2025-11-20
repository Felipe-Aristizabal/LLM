"""
Structured data store for Tecnoquímicas contact and company information.

This module implements a deterministic tool that reads a JSON file with
predefined facts (phone numbers, NIT, addresses, schedules, etc.) and
exposes helper functions for the conversational agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from tecnoquimicas_kb.domain.models import StructuredFact

# Global cache to avoid reloading JSON on every call.
_FACT_CACHE: List[StructuredFact] = []


def _load_json(path: Path) -> dict:
    """Load JSON content from a file.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON object.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_all_facts(json_path: Path) -> List[StructuredFact]:
    """Load all structured facts from the JSON file into memory.

    The file is expected to have the following high-level structure::

        {
          "facts": [
            {
              "id": "contact_phone_consumer_service",
              "category": "contact",
              "type": "phone",
              "label": "...",
              "value": "...",
              "description": "...",
              "keywords": ["..."],
              "metadata": {"source": "..."}
            },
            ...
          ]
        }

    Parameters
    ----------
    json_path:
        Location of the JSON file on disk.

    Returns
    -------
    List[StructuredFact]
        List of StructuredFact objects representing all known facts.
    """
    global _FACT_CACHE

    if _FACT_CACHE:
        return _FACT_CACHE

    data = _load_json(json_path)
    facts_raw = data.get("facts", [])

    facts: List[StructuredFact] = []
    for item in facts_raw:
        fact = StructuredFact(
            id=item["id"],
            category=item.get("category", "generic"),
            type=item.get("type", "generic"),
            label=item.get("label", item["id"]),
            value=item.get("value", ""),
            description=item.get("description"),
            keywords=item.get("keywords", []),
            metadata=item.get("metadata", {}),
        )
        facts.append(fact)

    _FACT_CACHE = facts
    return facts


def get_fact_by_id(fact_id: str) -> Optional[StructuredFact]:
    """Return a structured fact by its identifier.

    Parameters
    ----------
    fact_id:
        Identifier of the fact (as defined in the JSON schema).

    Returns
    -------
    Optional[StructuredFact]
        The corresponding StructuredFact object, or None if not found.
    """
    for fact in _FACT_CACHE:
        if fact.id == fact_id:
            return fact
    return None


def search_facts(query: str, limit: int = 3) -> List[StructuredFact]:
    """Return the most relevant facts for a natural-language query.

    English comment:
    This is a simple keyword-based scorer:
    - Lowercase query
    - For each fact, count how many keywords / label tokens appear
    - Keep only those with score > 0 and return the top N
    """
    if not _FACT_CACHE:
        return []

    q = query.lower()

    scored: List[tuple[StructuredFact, int]] = []
    for fact in _FACT_CACHE:
        # Combine label + keywords as potential matches
        tokens: List[str] = list(fact.keywords)
        tokens.append(fact.label.lower())
        tokens.append(fact.category.lower())
        tokens.append(fact.type.lower())

        score = 0
        for tok in tokens:
            tok = tok.strip().lower()
            if tok and tok in q:
                score += 1

        if score > 0:
            scored.append((fact, score))

    # Sort by descending score, keep best N
    scored.sort(key=lambda x: x[1], reverse=True)
    return [f for (f, _) in scored[:limit]]
