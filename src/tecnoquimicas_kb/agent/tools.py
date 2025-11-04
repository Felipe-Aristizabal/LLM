import json
from pathlib import Path
from typing import Any, Dict, Optional

STRUCTURED_PATH = Path(__file__).resolve().parent.parent / "data" / "structured" / "faqs.json"

def _load_structured() -> Dict[str, Any]:
    if not STRUCTURED_PATH.exists():
        return {}
    return json.loads(STRUCTURED_PATH.read_text(encoding="utf-8"))

def get_structured_answer(query: str) -> Optional[str]:
    """
    Búsqueda simple determinista en JSON (puedes cambiar a claves → sinónimos).
    Estrategia: mapeo por palabras clave.
    """
    data = _load_structured()
    q = query.lower()

    # Reglas sencillas (ajústalas a tu dominio real)
    if "tel" in q or "número" in q or "numero" in q:
        return f"Teléfono de servicio al cliente: {data.get('telefono_servicio_cliente', 'No disponible')}"
    if "horario" in q or "atención" in q or "atencion" in q:
        return f"Horarios de atención: {data.get('horarios_atencion', 'No disponible')}"
    if "sede" in q and "cali" in q:
        sedes = data.get("sedes_cali", [])
        return "Sedes en Cali:\n- " + "\n- ".join(sedes) if sedes else "No disponible"
    if "nit" in q:
        return f"NIT de la empresa: {data.get('nit_empresa', 'No disponible')}"
    if "correo" in q or "email" in q:
        return f"Correo de soporte: {data.get('correo_soporte', 'No disponible')}"
    if "sitio" in q or "website" in q or "web" in q:
        return f"Sitio web: {data.get('sitio_web', 'No disponible')}"

    # Si no matchea, devolvemos None para que rote a RAG/Stuffing
    return None