"""Herramientas deterministas (datos estructurados).

Incluye utilidades para responder a preguntas de “catálogo” (teléfonos, NIT,
horarios, sedes, correos, etc.) a partir de un JSON local. Se aplica una
normalización simple del texto de entrada y se cachea el archivo para
evitar I/O repetido.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional

# Ruta del JSON estructurado (puedes cambiarla según tu proyecto).
STRUCTURED_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "structured" / "faqs.json"
)

# Cache de datos y de mtime para recargar solo cuando cambie el archivo.
_DATA_CACHE: Optional[Dict[str, Any]] = None
_DATA_MTIME: Optional[float] = None


def _normalize(text: str) -> str:
    """Normaliza texto a minúsculas sin acentos, útil para matching robusto."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def _load_structured() -> Dict[str, Any]:
    """Carga el JSON estructurado con cache por mtime.

    Returns:
        Diccionario con los datos estructurados. Si el archivo no existe,
        devuelve un dict vacío.
    """
    global _DATA_CACHE, _DATA_MTIME

    if not STRUCTURED_PATH.exists():
        _DATA_CACHE, _DATA_MTIME = {}, None
        return {}

    mtime = STRUCTURED_PATH.stat().st_mtime
    if _DATA_CACHE is None or _DATA_MTIME != mtime:
        _DATA_CACHE = json.loads(STRUCTURED_PATH.read_text(encoding="utf-8"))
        _DATA_MTIME = mtime

    return _DATA_CACHE or {}


def get_structured_answer(query: str) -> Optional[str]:
    """Responde consultas simples contra el JSON estructurado.

    Estrategia:
        - Normaliza la consulta (minúsculas sin acentos).
        - Aplica reglas de palabras clave para mapear a campos del JSON.
        - Devuelve None si no hay match para que el router use TOOL_DOCS.

    Args:
        query: Pregunta del usuario.

    Returns:
        Respuesta corta y determinista, o None si no hay coincidencia.
    """
    data = _load_structured()
    qnorm = _normalize(query)

    # Reglas sencillas (ajústalas a tu dominio real).
    if any(k in qnorm for k in ("tel", "numero", "número")):
        return f"Teléfono de servicio al cliente: {data.get('telefono_servicio_cliente', 'No disponible')}"

    if any(k in qnorm for k in ("horario", "atencion", "atención")):
        return f"Horarios de atención: {data.get('horarios_atencion', 'No disponible')}"

    if "sede" in qnorm and "cali" in qnorm:
        sedes = data.get("sedes_cali", [])
        return "Sedes en Cali:\n- " + "\n- ".join(sedes) if sedes else "No disponible"

    if "nit" in qnorm:
        return f"NIT de la empresa: {data.get('nit_empresa', 'No disponible')}"

    if any(k in qnorm for k in ("correo", "email")):
        return f"Correo de soporte: {data.get('correo_soporte', 'No disponible')}"

    if any(k in qnorm for k in ("sitio", "website", "web")):
        return f"Sitio web: {data.get('sitio_web', 'No disponible')}"

    # Sin match: que el router pruebe con RAG/Stuffing.
    return None
