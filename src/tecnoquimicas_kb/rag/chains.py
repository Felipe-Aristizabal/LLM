import os
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# LLM providers
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()

def get_llm():
    provider = os.getenv("MODEL_PROVIDER", "gemini").lower()
    if provider == "ollama":
        model_id = os.getenv("OLLAMA_MODEL_ID", "gemma3:4b")
        return ChatOllama(model=model_id)  # Ollama local
    # default: Gemini
    model_id = os.getenv("GEN_MODEL_ID", "gemini-2.5-pro")
    return ChatGoogleGenerativeAI(model=model_id)  # requiere GOOGLE_API_KEY

def _get_embeddings():
    provider = os.getenv("EMBED_PROVIDER", "local").lower()
    if provider == "google":
        return GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    elif provider == "ollama":
        model_id = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        return OllamaEmbeddings(model=model_id)
    else:
        model_name = os.getenv("EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model_name)

def load_index(index_dir="src/tecnoquimicas_kb/index/faiss"):
    embeddings = _get_embeddings()
    return FAISS.load_local(index_dir, embeddings=embeddings, allow_dangerous_deserialization=True)

# ------- PROMPTS (en español) -------
SYSTEM_BASE = """Eres un asistente de Tecnoquímicas (TQ).
Responde SOLO con la información contenida en el CONTEXTO.
Si la respuesta no está en el contexto, di de forma explícita: “No encontré esa información en las fuentes disponibles.” 
Añade al final una sección 'Fuentes' con los URLs/ids de los fragmentos usados.
Sé conciso y exacto. Responde en español neutro."""

P_SUMMARY = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BASE),
    ("human", "Resume los puntos clave para onboarding de un cliente nuevo.\n\nCONTEXTO:\n{context}")
])

P_FAQ = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BASE),
    ("human", "Genera {n} preguntas frecuentes con sus respuestas basadas en el CONTEXTO.\n"
              "Cubre: quién es TQ, productos/portafolio, canales de contacto, cobertura, sostenibilidad.\n\nCONTEXTO:\n{context}")
])

P_QA = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BASE),
    ("human", "Pregunta: {question}\n\nCONTEXTO:\n{context}")
])

def _format_sources(docs: List[Document]) -> str:
    uniq = []
    for d in docs:
        src = d.metadata.get("url") or d.metadata.get("source") or d.metadata.get("chunk_id")
        if src and src not in uniq:
            uniq.append(src)
    if not uniq: return "Fuentes: (no disponibles)"
    return "Fuentes:\n" + "\n".join(f"- {s}" for s in uniq[:10])

def summarize(vs: FAISS, k: int = 8) -> str:
    llm = get_llm()
    docs = vs.similarity_search("información general para clientes nuevos", k=k)
    context = "\n\n".join(d.page_content for d in docs)
    msg = P_SUMMARY.format(context=context)
    out = llm.invoke(msg)
    return f"{out.content}\n\n{_format_sources(docs)}"

def make_faq(vs: FAISS, n: int = 10, k: int = 12) -> str:
    llm = get_llm()
    docs = vs.similarity_search("preguntas frecuentes clientes TQ", k=k)
    context = "\n\n".join(d.page_content for d in docs)
    msg = P_FAQ.format(context=context, n=n)
    out = llm.invoke(msg)
    return f"{out.content}\n\n{_format_sources(docs)}"

def answer(vs: FAISS, question: str, k: int = 6) -> Tuple[str, List[Document]]:
    llm = get_llm()
    docs = vs.similarity_search(question, k=k)
    context = "\n\n".join(d.page_content for d in docs)
    msg = P_QA.format(context=context, question=question)
    out = llm.invoke(msg)
    return f"{out.content}\n\n{_format_sources(docs)}", docs
