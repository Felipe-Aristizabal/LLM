
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
MAX_CONTEXT_CHARS_QA  = int(os.getenv("MAX_CONTEXT_CHARS_QA",  "40000"))
DEFAULT_K_FILES_QA    = int(os.getenv("DEFAULT_K_FILES_QA",   "15"))
DEFAULT_K_FILES_SUM   = int(os.getenv("DEFAULT_K_FILES_SUM",  "12"))
DEFAULT_K_FILES_FAQ   = int(os.getenv("DEFAULT_K_FILES_FAQ",  "15"))

load_dotenv()

# # ---------------------------
# # Config alto para (gemini-2.5-pro o consumo modelos por API)
# # ---------------------------
# MAX_CONTEXT_CHARS_ALL = 400_000    # tope para Resumen/FAQ
# MAX_CONTEXT_CHARS_QA  = 200_000    # tope para Q&A (contexto dirigido)
# DEFAULT_K_FILES_QA    = 20         # top-k archivos por similitud simple

SYSTEM_BASE = (
    "Eres un asistente de Tecnoquímicas (TQ). "
    "Responde SOLO con la información contenida en el CONTEXTO entregado. "
    "Si la respuesta no está en el contexto, di explícitamente: "
    "“No encontré esa información en las fuentes disponibles.” "
    "Al final agrega una sección 'Fuentes' con los nombres/urls de los archivos si es posible. "
    "Sé conciso y exacto. Responde en español neutro."
)

P_SUMMARY = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BASE + "\n\nCONTEXTO:\n{context}"),
    ("human", "Resume los puntos clave para onboarding de un cliente nuevo.")
])

P_FAQ = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BASE + "\n\nCONTEXTO:\n{context}"),
    ("human", "Genera {n} preguntas frecuentes con sus respuestas basadas en el CONTEXTO. "
              "Cubre: quién es TQ, productos/portafolio, canales de contacto, cobertura y sostenibilidad.")
])

P_QA = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BASE + "\n\nCONTEXTO:\n{context}"),
    ("human", "Pregunta: {q}")
])

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

def load_all_docs(data_dir: str = "src/tecnoquimicas_kb/data/clean") -> List[Tuple[str, Dict]]:
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
def build_context_all(docs: List[Tuple[str, Dict]], limit_chars: int = MAX_CONTEXT_CHARS_ALL) -> str:
    # concatena todo con pequeñas separaciones y corta por límite
    parts, total = [], 0
    for text, meta in docs:
        block = f"[FUENTE: {meta.get('url') or meta.get('source','desconocida')}]\n{text}\n\n"
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

def build_context_for_question(q: str, docs: List[Tuple[str, Dict]],
                               k_files: int = DEFAULT_K_FILES_QA,
                               limit_chars: int = MAX_CONTEXT_CHARS_QA) -> str:
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
    top = [src for src, s in scored[:k_files] if s > 0] or [scored[0][0] if scored else ""]

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
        guide, docs,
        k_files=DEFAULT_K_FILES_SUM,
        limit_chars=MAX_CONTEXT_CHARS_ALL
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
        guide, docs,
        k_files=DEFAULT_K_FILES_FAQ,
        limit_chars=MAX_CONTEXT_CHARS_ALL
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
        q, docs,
        k_files=DEFAULT_K_FILES_QA,
        limit_chars=MAX_CONTEXT_CHARS_QA
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

