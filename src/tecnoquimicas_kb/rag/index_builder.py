from pathlib import Path
import json, os
from typing import List, Dict

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Embeddings providers
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()

def _get_embeddings():
    provider = os.getenv("EMBED_PROVIDER", "local").lower()
    if provider == "google":
        # Usa el embedding oficial de Google (rápido y de alta calidad)
        # Alternativas: "text-embedding-004" o el exp de la cheatsheet
        return GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    elif provider == "ollama":
        model_id = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        return OllamaEmbeddings(model=model_id)
    else:
        # local: sin llaves, portable
        return HuggingFaceEmbeddings(model_name=os.getenv("EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"))

def _load_chunks_from_dir(data_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for p in sorted(data_dir.glob("*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                row: Dict = json.loads(line)
                text = row.get("text") or row.get("content") or ""
                meta = {k: v for k, v in row.items() if k != "text" and k != "content"}
                if text.strip():
                    docs.append(Document(page_content=text, metadata=meta))
    for p in sorted(data_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": p.name}))
    return docs

def build_faiss(data_dir="src/tecnoquimicas_kb/data/clean",
                index_dir="src/tecnoquimicas_kb/index/faiss"):
    data_dir = Path(data_dir); index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    docs = _load_chunks_from_dir(data_dir)
    if not docs:
        raise RuntimeError(f"No se hallaron chunks en {data_dir.resolve()}")

    embeddings = _get_embeddings()
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(str(index_dir))
    return len(docs)
