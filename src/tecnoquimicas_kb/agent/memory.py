"""Gestión de memoria en sesión (en memoria).

Este módulo define estructuras simples para almacenar el historial de chat
por sesión. Está optimizado para rapidez y simplicidad (sin persistencia
en disco por defecto), cumpliendo con PEP 8 y con documentación en español.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import time


@dataclass
class ChatTurn:
    """Representa un turno de conversación.

    Attributes:
        role: Rol del emisor: "user" | "assistant" | "system".
        content: Contenido textual del mensaje.
        ts: Marca de tiempo UNIX (float) cuando se añadió el turno.
    """

    role: str
    content: str
    ts: float


class InMemorySessionStore:
    """Almacén de conversaciones por sesión, residente en memoria.

    Notas:
        - Es apropiado para prototipos o apps que no requieren persistencia
          entre reinicios del proceso.
        - Si deseas persistencia, puedes extender esta clase con serialización
          a JSON o usar una base de datos (p. ej., SQLite).

    Ejemplo:
        store = InMemorySessionStore()
        store.append("session-1", "user", "Hola")
        history = store.get_history("session-1")
    """

    def __init__(self) -> None:
        # Diccionario: session_id -> lista de ChatTurn
        self._sessions: Dict[str, List[ChatTurn]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Obtiene el historial de una sesión como lista de dicts.

        Se devuelve en un formato compatible con varios clientes LLM
        que esperan {"role": ..., "content": ...}.

        Args:
            session_id: Identificador único de sesión.

        Returns:
            Lista de mensajes (dict) en orden de inserción.
        """
        return [
            {"role": t.role, "content": t.content}
            for t in self._sessions.get(session_id, [])
        ]

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        max_turns: int = 200,
    ) -> None:
        """Agrega un turno al historial y aplica poda por longitud.

        Args:
            session_id: Identificador de sesión.
            role: "user" | "assistant" | "system".
            content: Texto del mensaje.
            max_turns: Límite de turnos a conservar por sesión (poda FIFO).
        """
        sess = self._sessions.setdefault(session_id, [])
        sess.append(ChatTurn(role=role, content=content, ts=time.time()))
        if len(sess) > max_turns:
            # Eliminamos los más antiguos para mantener el tamaño máximo.
            del sess[: len(sess) - max_turns]

    def clear(self, session_id: str) -> None:
        """Borra el historial completo de una sesión.

        Args:
            session_id: Identificador de sesión.
        """
        self._sessions[session_id] = []
