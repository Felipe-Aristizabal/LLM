from __future__ import annotations

from typing import Dict, List

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware

from tecnoquimicas_kb.domain.conversation_agent import run_agent_turn
from tecnoquimicas_kb.domain.models import AgentSettings, ChatMessage
from tecnoquimicas_kb.api.models import ChatRequest, ChatResponse

import os
from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# API Key Security Dependency
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("API_KEY")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

# English comment:
# Simple global dictionary that holds chat histories per session id.
# This is fine for the assignment and local development, but in a
# real production deployment you would replace this with Redis,
# a database, or another external store.
_SESSION_HISTORY: Dict[str, List[ChatMessage]] = {}


def _get_history(session_id: str) -> List[ChatMessage]:
    """Return the current history list for a session id.

    English comment:
    If the session does not exist yet, return an empty list.
    """
    return _SESSION_HISTORY.get(session_id, [])


def _set_history(session_id: str, history: List[ChatMessage]) -> None:
    """Persist the updated history for a given session id."""
    _SESSION_HISTORY[session_id] = history


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Tecnoquímicas KB – Conversational Agent API",
    version="1.0.0",
    description=(
        "REST API for the Tecnoquímicas RAG-based conversational agent. "
        "Designed to be used from external services (e.g. N8N, WhatsApp)."
    ),
)

# Optional CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in real deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health(dep: None = Depends(verify_api_key)) -> dict:
    """Simple health-check endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, dep: None = Depends(verify_api_key)) -> ChatResponse:
    """Main chat endpoint that delegates to the conversation agent.

    English comment:
    - Receives a user message + session id.
    - Loads the previous history for that session.
    - Calls run_agent_turn with the shared agent configuration.
    - Stores updated history back into the in-memory session store.
    - Returns a JSON payload with answer + metadata.
    """
    session_id = payload.session_id.strip()
    user_text = payload.message.strip()

    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id must not be empty.",
        )
    if not user_text:
        raise HTTPException(
            status_code=400,
            detail="message must not be empty.",
        )

    # Retrieve existing history for this session (or empty).
    history = _get_history(session_id)

    # Configure the agent (same defaults you use in Streamlit, or adjust).
    agent_cfg = AgentSettings(
        max_history_messages=20,
        use_router=True,
        default_tool="rag_qa",
        allow_compose=True,
        followup_lookback=6,
    )

    try:
        turn_result = run_agent_turn(
            user_input=user_text,
            history=history,
            agent_settings=agent_cfg,
        )
    except Exception as exc:  # noqa: BLE001
        # English comment:
        # Any unexpected exception is converted into a 500 error with a
        # short message. The actual logging should stay inside the agent.
        raise HTTPException(
            status_code=500,
            detail=f"Internal error while processing the message: {exc!r}",
        ) from exc

    # Persist updated history for this session.
    _set_history(session_id, turn_result.updated_history)

    # Try to obtain sources from the last assistant message.
    sources: list[str] = []
    last_assistant = None
    for msg in reversed(turn_result.updated_history):
        if msg.role == "assistant":
            last_assistant = msg
            break
    if last_assistant and last_assistant.sources:
        sources = last_assistant.sources

    return ChatResponse(
        session_id=session_id,
        answer=turn_result.answer,
        used_tool=turn_result.used_tool,
        tool_details=turn_result.tool_details or {},
        sources=sources,
        history_size=len(turn_result.updated_history),
    )
