from typing import List, Dict, Any
from dataclasses import dataclass
import time

@dataclass
class ChatTurn:
    role: str   # "user" | "assistant" | "system"
    content: str
    ts: float

class InMemorySessionStore:
    """
    Memoria por sesión en memoria (rápida y simple).
    Puedes sustituir por SQLite si necesitas persistencia entre reinicios.
    """
    def __init__(self):
        self._sessions: Dict[str, List[ChatTurn]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self._sessions.get(session_id, [])]

    def append(self, session_id: str, role: str, content: str):
        self._sessions.setdefault(session_id, []).append(ChatTurn(role, content, time.time()))

    def clear(self, session_id: str):
        self._sessions[session_id] = []