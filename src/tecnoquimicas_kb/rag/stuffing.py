from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import json, os, re, time
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from google.api_core.exceptions import ResourceExhausted

# Límites más conservadores
MAX_CONTEXT_CHARS_ALL = int(os.getenv("MAX_CONTEXT_CHARS_ALL", "60000"))
MAX_CONTEXT_CHARS_QA = int(os.getenv("MAX_CONTEXT_CHARS_QA", "40000"))
DEFAULT_K_FILES_QA = int(os.getenv("DEFAULT_K_FILES_QA", "15"))
DEFAULT_K_FILES_SUM = int(os.getenv("DEFAULT_K_FILES_SUM", "12"))
DEFAULT_K_FILES_FAQ = int(os.getenv("DEFAULT_K_FILES_FAQ", "15"))

load_dotenv()

# # ---------------------------
# # Config alto para (gemini-2.5-pro o consumo modelos por API)
# # ---------------------------
# MAX_CONTEXT_CHARS_ALL = 400_000    # tope para Resumen/FAQ
# MAX_CONTEXT_CHARS_QA  = 200_000    # tope para Q&A (contexto dirigido)
# DEFAULT_K_FILES_QA    = 20         # top-k archivos por similitud simple

SYSTEM_BASE = (
    "Eres un asistente experto de Tecnoquímicas (TQ) para un sistema RAG. "
    "TU ÚNICA fuente de verdad es el CONTEXTO provisto en este turno. "
    "Está TERMINANTEMENTE PROHIBIDO usar conocimiento externo, suposiciones o datos no presentes en el CONTEXTO.\n\n"
    "CUANDO FALTE INFORMACIÓN:\n"
    "• Si no puedes responder con exactitud, debes decir textualmente: "
    "«No encontré esa información en las fuentes disponibles.»\n"
    "• Si el CONTEXTO es ambiguo o contradictorio, dilo explícitamente y muestra ambas versiones con su fuente; "
    "si hay fechas, prioriza la más reciente y justifícalo en una línea.\n\n"
    "FORMATO Y ESTILO DE RESPUESTA (OBLIGATORIO):\n"
    "• Idioma: español neutro, tono profesional, claro y conciso; sin emojis ni jerga.\n"
    "• Estructura: usa encabezados y viñetas cuando mejore la legibilidad. "
    "Puedes usar tablas Markdown SOLO si aportan claridad (p.ej., comparar canales o productos).\n"
    "• Números/fechas/nombres propios: respétalos EXACTAMENTE como aparecen en el CONTEXTO. "
    "No conviertas monedas, no infieras porcentajes, no traduzcas marcas.\n"
    "• Si citas cifras o hitos, incluye la FECHA tal cual esté en el CONTEXTO (si existe).\n"
    "• Respuestas breves pero completas: prioriza utilidad para un primer contacto (quiénes somos, portafolio, contacto, cobertura, sostenibilidad, procesos básicos).\n\n"
    "SECCIÓN «Fuentes» (OBLIGATORIA):\n"
    "• Al final agrega una sección titulada «Fuentes». Enumera nombres/URLs EXACTOS que aparezcan en el CONTEXTO y que hayas usado.\n"
    "Si no hay nombres/URLs en el CONTEXTO, escribe: «Fuentes: (no disponibles en el contexto)».\n\n"
    "REGLAS DE SEGURIDAD Y SECTOR (FARMACÉUTICO/CONSUMO):\n"
    "• No emitas diagnósticos médicos, no indiques dosis, tratamientos ni recomendaciones clínicas. "
    "Si el usuario lo pide y el CONTEXTO no lo soporta, responde: "
    "«No encontré esa información en las fuentes disponibles.» y sugiere contactar canales oficiales de TQ.\n"
    "• No prometas disponibilidad de productos ni tiempos de entrega si no están en el CONTEXTO.\n"
    "• No compartas datos personales ni información sensible.\n\n"
    "ALCANCE Y COHERENCIA:\n"
    "• Responde SOLO a lo que se pregunta. Si el usuario pide múltiples puntos, organiza la salida en secciones.\n"
    "• Si el CONTEXTO contiene información internacional y local, identifica el alcance (país/ciudad) cuando esté indicado.\n"
    "• Si el CONTEXTO menciona procesos (p.ej., pagos, servicio al cliente), resume pasos alto nivel; no inventes pasos faltantes.\n\n"
    "ESTILO DE VERIFICACIÓN INTERNA (SIN MOSTRAR RAZONAMIENTO):\n"
    "• Antes de responder, verifica: (1) que todos los datos citados existan en el CONTEXTO; "
    "(2) que no haya contradicciones sin avisar; (3) que la sección «Fuentes» esté presente.\n"
    "• NO muestres razonamientos intermedios; entrega SOLO la respuesta final.\n"
)


P_SUMMARY = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_BASE + "\n\nCONTEXTO:\n{context}"),
        (
            "human",
            "Elabora un RESUMEN DE ONBOARDING para un cliente nuevo de Tecnoquímicas.\n"
            "Objetivo: que un cliente entienda rápidamente quién es TQ y cómo interactuar.\n\n"
            "SALIDA (usa exactamente estos encabezados si hay datos; si faltan, indica «No encontré esa información en las fuentes disponibles.»):\n"
            "1) Quiénes somos (misión/visión/propósito, reseña corta)\n"
            "2) Portafolio y marcas (líneas de producto y ejemplos de marcas si aparecen)\n"
            "3) Cobertura y sedes (países/ciudades/planta/cluster si están en el contexto)\n"
            "4) Canales de contacto (servicio al cliente, encuéntranos, línea ética, otros)\n"
            "5) Procesos básicos (compras/pagos, atención a clientes, portales relevantes)\n"
            "6) Sostenibilidad (ejes/indicadores; incluir cifra/porcentaje/fecha si el contexto la trae)\n"
            "7) Noticias o hitos (máx. 3, con FECHA exacta tal como aparece en el contexto)\n"
            "8) Consideraciones regulatorias (si el contexto menciona restricciones del sector salud)\n\n"
            "REGLAS:\n"
            "• No inventes ni completes lagunas con supuestos.\n"
            "• Mantén cada punto en 1–3 viñetas concisas.\n"
            "• NUNCA cambies cifras/fechas ni extrapoles.\n",
        ),
    ]
)

P_FAQ = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_BASE + "\n\nCONTEXTO:\n{context}"),
        (
            "human",
            "Genera {n} PREGUNTAS FRECUENTES (FAQ) con sus respuestas, exclusivamente basadas en el CONTEXTO.\n"
            "Cobertura mínima: quién es TQ, portafolio/marcas, canales de contacto y servicio, cobertura/sedes, sostenibilidad, y (si existen) procesos básicos para clientes.\n\n"
            "FORMATO DE SALIDA (Markdown):\n"
            "• Lista numerada. En cada ítem:\n"
            "  **Pregunta:** <formulación clara y breve>\n"
            "  **Respuesta:** <respuesta precisa y completa, sin especular>\n"
            "  **Fuente:** <nombre y/o URL EXACTA presentes en el CONTEXTO>\n\n"
            "REGLAS ESTRICTAS:\n"
            "• Si una respuesta NO está en el CONTEXTO, responde SOLO: «No encontré esa información en las fuentes disponibles.»\n"
            "• Usa el MISMO idioma del CONTEXTO (si está en español, responde en español neutro).\n"
            "• No repitas preguntas idénticas; prioriza utilidad para un primer contacto.\n"
            "• Si hay números/porcentajes/fechas, reprodúcelos exactamente y señala la fuente.\n"
            "• Evita detalles regulatorios si no están citados en el CONTEXTO.\n\n",
        ),
    ]
)

P_QA = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_BASE + "\n\nCONTEXTO:\n{context}"), ("human", "Pregunta: {q}")]
)


def get_llm():
    provider = os.getenv("MODEL_PROVIDER", "gemini").lower()
    if provider == "ollama":
        model_id = os.getenv("OLLAMA_MODEL_ID", "gemma3:4b")
        return ChatOllama(model=model_id)
    # default: gemini
    model_id = os.getenv("GEN_MODEL_ID", "gemini-2.5-pro")
    return ChatGoogleGenerativeAI(model=model_id)


# ---------------------------
# Loaders
# ---------------------------
def _clean_text(s: str) -> str:
    s = s.replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _doc_from_json_obj(obj: Dict) -> Tuple[str, Dict]:
    text = obj.get("text") or obj.get("content") or ""
    meta = {k: v for k, v in obj.items() if k not in ("text", "content")}
    return _clean_text(text), meta


def _load_json_file(p: Path) -> List[Tuple[str, Dict]]:
    docs = []
    raw = p.read_text(encoding="utf-8", errors="ignore")
    # intenta jsonl primero
    any_line = False
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        any_line = True
        try:
            obj = json.loads(line)
            text, meta = _doc_from_json_obj(obj)
            if text:
                meta.setdefault("source", str(p))
                docs.append((text, meta))
        except Exception:
            pass
    if docs:
        return docs
    # si no era jsonl, intenta json normal
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict):
                    text, meta = _doc_from_json_obj(it)
                    if text:
                        meta.setdefault("source", str(p))
                        docs.append((text, meta))
        elif isinstance(obj, dict):
            text, meta = _doc_from_json_obj(obj)
            if text:
                meta.setdefault("source", str(p))
                docs.append((text, meta))
    except Exception:
        # no era json válido
        pass
    return docs


def _load_txt_file(p: Path) -> List[Tuple[str, Dict]]:
    text = p.read_text(encoding="utf-8", errors="ignore")
    text = _clean_text(text)
    return [(text, {"source": str(p)})] if text else []


def load_all_docs(
    data_dir: str = "src/tecnoquimicas_kb/data/clean",
) -> List[Tuple[str, Dict]]:
    root = Path(data_dir)
    out: List[Tuple[str, Dict]] = []
    for fp in root.rglob("*"):
        if fp.suffix.lower() in (".json", ".jsonl"):
            out.extend(_load_json_file(fp))
        elif fp.suffix.lower() == ".txt":
            out.extend(_load_txt_file(fp))
    return out


# ---------------------------
# Context building (stuffing)
# ---------------------------
def build_context_all(
    docs: List[Tuple[str, Dict]], limit_chars: int = MAX_CONTEXT_CHARS_ALL
) -> str:
    # concatena todo con pequeñas separaciones y corta por límite
    parts, total = [], 0
    for text, meta in docs:
        block = f"[FUENTE: {meta.get('url') or meta.get('source', 'desconocida')}]\n{text}\n\n"
        if total + len(block) > limit_chars:
            remain = limit_chars - total
            if remain > 0:
                parts.append(block[:remain])
            break
        parts.append(block)
        total += len(block)
    return "".join(parts)


def _score(question_tokens: set, text: str) -> int:
    # conteo muy simple de términos (rápido y sin deps)
    score = 0
    for t in question_tokens:
        if t and t in text:
            score += 1
    return score


def build_context_for_question(
    q: str,
    docs: List[Tuple[str, Dict]],
    k_files: int = DEFAULT_K_FILES_QA,
    limit_chars: int = MAX_CONTEXT_CHARS_QA,
) -> str:
    # ranking naive de archivos por coincidencia de tokens
    tokens = {t.lower() for t in re.findall(r"[A-Za-zÀ-ÿ0-9_]+", q)}
    # agrupa por archivo (source)
    by_src: Dict[str, List[str]] = {}
    for text, meta in docs:
        src = meta.get("url") or meta.get("source", "desconocida")
        by_src.setdefault(src, []).append(text)

    scored = []
    for src, texts in by_src.items():
        joined = " ".join(texts)[:200_000]  # no cargamos de más para el scoring
        scored.append((src, _score(tokens, joined)))

    # top-k por score (desc)
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [src for src, s in scored[:k_files] if s > 0] or [
        scored[0][0] if scored else ""
    ]

    # concatena solo los elegidos
    parts, total = [], 0
    for src in top:
        block = f"[FUENTE: {src}]\n" + " ".join(by_src[src]) + "\n\n"
        if total + len(block) > limit_chars:
            remain = limit_chars - total
            if remain > 0:
                parts.append(block[:remain])
            break
        parts.append(block)
        total += len(block)
    # fallback si no hubo match
    if not parts:
        return build_context_all(docs, limit_chars=limit_chars)
    return "".join(parts)


# ---------------------------
# Tasks
# ---------------------------
# def summarize_stuffing(docs: List[Tuple[str, Dict]]) -> str:
#     ctx = build_context_all(docs, limit_chars=MAX_CONTEXT_CHARS_ALL)
#     llm = get_llm()
#     msg = P_SUMMARY.format(context=ctx)
#     out = llm.invoke(msg)
#     return out.content


def summarize_stuffing(docs):
    # consulta guía para ranking
    guide = "información general Tecnoquímicas para onboarding de clientes nuevos"
    ctx = build_context_for_question(
        guide, docs, k_files=DEFAULT_K_FILES_SUM, limit_chars=MAX_CONTEXT_CHARS_ALL
    )
    llm = get_llm()
    msg = P_SUMMARY.format(context=ctx)
    try:
        out = llm.invoke(msg)
        return out.content
    except ResourceExhausted as e:
        time.sleep(6)
        out = llm.invoke(msg)
        return out.content


# def faq_stuffing(docs: List[Tuple[str, Dict]], n: int = 10) -> str:
#     ctx = build_context_all(docs, limit_chars=MAX_CONTEXT_CHARS_ALL)
#     llm = get_llm()
#     msg = P_FAQ.format(context=ctx, n=n)
#     out = llm.invoke(msg)
#     return out.content


def faq_stuffing(docs, n: int = 10):
    guide = "preguntas frecuentes clientes Tecnoquímicas contacto portafolio cobertura sostenibilidad"
    ctx = build_context_for_question(
        guide, docs, k_files=DEFAULT_K_FILES_FAQ, limit_chars=MAX_CONTEXT_CHARS_ALL
    )
    llm = get_llm()
    msg = P_FAQ.format(context=ctx, n=n)
    try:
        out = llm.invoke(msg)
        return out.content
    except ResourceExhausted as e:
        time.sleep(6)
        out = llm.invoke(msg)
        return out.content


# def qa_stuffing(docs: List[Tuple[str, Dict]], q: str) -> str:
#     ctx = build_context_for_question(q, docs, k_files=DEFAULT_K_FILES_QA, limit_chars=MAX_CONTEXT_CHARS_QA)
#     llm = get_llm()
#     msg = P_QA.format(context=ctx, q=q)
#     out = llm.invoke(msg)
#     return out.content


def qa_stuffing(docs, q: str):
    ctx = build_context_for_question(
        q, docs, k_files=DEFAULT_K_FILES_QA, limit_chars=MAX_CONTEXT_CHARS_QA
    )
    llm = get_llm()
    msg = P_QA.format(context=ctx, q=q)
    try:
        out = llm.invoke(msg)
        return out.content
    except ResourceExhausted as e:
        time.sleep(6)
        out = llm.invoke(msg)
        return out.content
