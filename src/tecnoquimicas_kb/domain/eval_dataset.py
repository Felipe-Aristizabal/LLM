"""Evaluation dataset helpers for the Tecnoquímicas RAG assistant.

This module defines a small default set of evaluation questions and a
utility function to load questions from disk when needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Union


# Default question set used when no external file is provided. The goal
# is to cover typical client-facing topics such as identity, portfolio,
# coverage, sustainability and contact channels.
DEFAULT_QUESTIONS: List[str] = [
    "¿Quién es Tecnoquímicas y en qué sectores opera?",
    "¿Dónde puedo encontrar los puntos de contacto de TQ?",
    "¿Qué categorías de productos comercializa TQ?",
    "¿Tienen presencia en todo Colombia? ¿Cobertura?",
    "¿Qué programas de sostenibilidad o planeta maneja TQ?",
    "¿Cuál es la historia o resumen de Tecnoquímicas?",
    "¿Qué marcas propias maneja TQ?",
    "¿Cómo puedo contactar ventas institucionales?",
    "¿Cómo reporto un evento adverso de un medicamento?",
    "¿Cómo presentar una PQRS ante Tecnoquímicas?",
    "¿Dónde puedo ver noticias o actualizaciones de TQ?",
    "¿Qué políticas de calidad declara Tecnoquímicas?",
    "¿Cómo solicitar información para prensa?",
]


def _load_txt_questions(path: Path) -> List[str]:
    """Load evaluation questions from a plain text file.

    Each non-empty line is treated as a question. Lines starting with '#'
    are ignored as comments.
    """
    questions: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        questions.append(stripped)
    return questions


def _load_json_questions(path: Path) -> List[Union[str, dict]]:
    """Load evaluation questions from a JSON or JSONL file.

    The function supports:
    - JSON arrays of strings
    - JSON arrays of objects with a 'q' or 'question' field
    - JSONL files with one JSON object per line
    """
    if path.suffix.lower() == ".jsonl":
        items: List[Union[str, dict]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append(obj)
        return items

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return [data]


def load_eval_questions(
    path: Path | None = None,
) -> List[Union[str, dict]]:
    """Load evaluation questions from disk or return the default set.

    When a path is provided, the function infers the format based on the
    file extension and returns either plain strings or dictionaries. When
    `path` is None, the built-in DEFAULT_QUESTIONS list is returned.
    """
    if path is None:
        # Return a copy to avoid accidental in-place modifications.
        return list(DEFAULT_QUESTIONS)

    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".txt"}:
        return _load_txt_questions(path)

    if suffix in {".json", ".jsonl"}:
        return _load_json_questions(path)

    # Fallback: treat unknown extensions as plain text.
    return _load_txt_questions(path)
