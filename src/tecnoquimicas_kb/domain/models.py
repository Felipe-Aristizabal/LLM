"""Core domain models for the Tecnoquímicas RAG assistant.

These dataclasses express the main concepts used in the question-answering
flows. They are intentionally lightweight and independent from specific
infrastructure details such as vector stores or HTTP clients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union


@dataclass
class Document:
    """Simple document representation used for context stuffing.

    Attributes
    ----------
    text:
        The main textual content of the document.
    metadata:
        Arbitrary structured metadata associated with the document.
        Common keys include 'source' and 'url'.
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def source(self) -> str:
        """Return a human-readable source identifier for the document."""
        return str(
            self.metadata.get("url") or self.metadata.get("source") or "desconocida"
        )


@dataclass
class QARequest:
    """Value object describing a QA request coming from a user."""

    question: str
    # Additional fields (such as user id, channel, etc.) may be added later.


@dataclass
class QAResponse:
    """Value object describing a QA response produced by the system."""

    question: str
    answer: str
    used_sources: List[str] = field(default_factory=list)
    raw_context: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# Type alias used by other modules to accept either Document instances or
# (text, metadata) tuples coming from legacy utilities.
DocLike = Union[Document, Tuple[str, Mapping[str, Any]]]
