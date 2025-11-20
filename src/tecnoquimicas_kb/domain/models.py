"""Core domain models for the Tecnoquímicas conversational agent.

These dataclasses capture the main concepts used across the RAG and
structured-data tools. They are intentionally small and independent of
infrastructure details (HTTP, vector stores, etc.).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union


def bool_from_env(name: str, default: bool = False) -> bool:
    """Return a boolean read from an environment variable.

    Accepted truthy values (case-insensitive):
        "1", "true", "t", "yes", "y", "on"

    Accepted falsy values:
        "0", "false", "f", "no", "n", "off"

    Any other value falls back to `default`.
    """
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


@dataclass
class Document:
    """Simple document representation used for context stuffing.

    Attributes
    ----------
    text:
        Raw text content of the document.
    metadata:
        Arbitrary metadata associated with the document (file path, URL, etc.).
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# Type alias used by other modules to accept either Document instances or
# (text, metadata) tuples coming from legacy utilities.
DocLike = Union[Document, Tuple[str, Mapping[str, Any]]]


@dataclass
class ChatMessage:
    """Single message in the chat history.

    Attributes
    ----------
    role:
        Role of the message author (e.g. "user", "assistant").
    content:
        Text content of the message shown in the UI.
    used_tool:
        Optional name of the tool used to produce this message
        (for assistant messages only).
    sources:
        Optional list of source identifiers used for this answer.
    metadata:
        Free-form dictionary for additional information (debug data, scores, etc.).
    """

    role: Literal["user", "assistant"]
    content: str
    used_tool: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredFact:
    """Single fact stored in the structured-data JSON file.

    Attributes
    ----------
    id:
        Unique identifier for the fact. Used by the router and tools.
    category:
        High-level group (e.g. "contact", "company", "ethics").
    type:
        Subtype of the fact (e.g. "phone", "address", "email", "schedule").
    label:
        Human-readable label for the fact.
    value:
        Main value of the fact (phone number, NIT, address, etc.).
    description:
        Optional short description to give more context.
    keywords:
        Optional list of keywords useful for routing or search.
    metadata:
        Free-form dictionary for additional attributes (country, URL, etc.).
    """

    id: str
    category: str
    type: str
    label: str
    value: str
    description: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSettings:
    """Configuration options for the conversational agent.

    Attributes
    ----------
    max_history_messages:
        Maximum number of messages to keep in memory. Older messages are
        discarded in FIFO order.
    use_router:
        When True, the agent uses the tool router to decide which tool to
        call for each user question.
    default_tool:
        Fallback tool name to use when the router is disabled or returns
        an unknown tool.
    allow_compose:
        Whether COMPOSE mode (RAG + facts) is allowed at all
    followup_lookback:
        followup_lookback: int = 8
    """

    max_history_messages: int = 20
    use_router: bool = True
    default_tool: str = "RAG_QA"
    allow_compose: bool = True
    followup_lookback: int = 8


@dataclass
class ToolChoice:
    """Routing decision produced by the tool router."""

    tool: str
    fact_id: Optional[str] = None
    reason: str = ""


@dataclass
class QAResponse:
    """Value object describing a QA response produced by the system.

    Attributes
    ----------
    question:
        Original user question.
    answer:
        Final answer text to show to the user.
    used_sources:
        List of identifiers of the sources used for the answer.
    raw_context:
        Optional raw stuffed context used to generate the answer.
    extra:
        Free-form dictionary for additional metadata.
    """

    question: str
    answer: str
    used_sources: List[str] = field(default_factory=list)
    raw_context: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTurnResult:
    """Result of a single call to the conversational agent.

    Attributes
    ----------
    answer:
        Final answer text produced for this turn.
    used_tool:
        Name of the tool used ("RAG_QA", "STRUCTURED_DATA", etc.).
    tool_details:
        Additional structured information about the tool execution.
    updated_history:
        Full conversation history including the new user and assistant
        messages.
    """

    answer: str
    used_tool: str
    tool_details: Dict[str, Any]
    updated_history: List[ChatMessage]
