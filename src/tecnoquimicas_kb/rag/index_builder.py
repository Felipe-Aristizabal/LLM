from pathlib import Path
import json, os, argparse
from typing import List, Dict, Union

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

def _to_doc(text: str, meta: Dict) -> Union[Document, None]:
    text = (text or "").strip()
    if not text:
        return None
    return Document(page_content=text, metadata=meta)

def _load_json_file(p: Path) -> List[Document]:
    docs: List[Document] = []
    try:
        raw = p.read_text(encoding="utf-8")
        try:
            # .jsonl?
            if "\n" in raw.strip():
                for line in raw.splitlines():
                    line = line.strip()
                    if not line: 
                        continue
                    row = json.loads(line)
                    doc = _to_doc(row.get("text") or row.get("content"), {k:v for k,v in row.items() if k not in ("text","content")})
                    if doc: docs.append(doc)
            else:
                # .json (objeto o lista)
                obj = json.loads(raw)
                if isinstance(obj, list):
                    for row in obj:
                        if not isinstance(row, dict): 
                            continue
                        doc = _to_doc(row.get("text") or row.get("content"), {k:v for k,v in row.items() if k not in ("text","content")})
                        if doc: docs.append(doc)
                elif isinstance(obj, dict):
                    doc = _to_doc(obj.get("text") or obj.get("content"), {k:v for k,v in obj.items() if k not in ("text","content")})
                    if doc: docs.append(doc)
        except json.JSONDecodeError:
            # si es jsonl pero con BOM u otro encoding, intenta línea a línea
            for line in raw.splitlines():
                line = line.strip()
                if not line: 
                    continue
                try:
                    row = json.loads(line)
                    doc = _to_doc(row.get("text") or row.get("content"), {k:v for k,v in row.items() if k not in ("text","content")})
                    if doc: docs.append(doc)
                except Exception:
                    continue
    except Exception as e:
        print(f"[WARN] No pude leer {p.name}: {e}")
    return docs

def _load_txt_file(p: Path) -> List[Document]:
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
        return [_to_doc(text, {"source": str(p)})] if text.strip() else []
    except Exception as e:
        print(f"[WARN] No pude leer {p.name}: {e}")
        return []

def _load_chunks_recursive(root_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for p in root_dir.rglob("*"):
        if p.suffix.lower() in (".jsonl", ".json"):
            docs.extend(_load_json_file(p))
        elif p.suffix.lower() == ".txt":
            docs.extend(_load_txt_file(p))
    return [d for d in docs if d]

def build_faiss(data_dir="src/tecnoquimicas_kb/data/clean",
                index_dir="src/tecnoquimicas_kb/index/faiss"):
    data_dir = Path(data_dir); index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Buscando chunks en: {data_dir.resolve()}")
    docs = _load_chunks_recursive(data_dir)
    print(f"[INFO] Documentos cargados: {len(docs)}")
    if not docs:
        raise RuntimeError(f"No se hallaron chunks en {data_dir.resolve()} (revisa subcarpetas y extensiones)")

    embeddings = _get_embeddings()
    print(f"[INFO] Embeddings provider: {os.getenv('EMBED_PROVIDER','local')}")
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(str(index_dir))
    print(f"[OK] Índice guardado en: {index_dir.resolve()}")
    return len(docs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="src/tecnoquimicas_kb/data/clean")
    ap.add_argument("--index-dir", default="src/tecnoquimicas_kb/index/faiss")
    args = ap.parse_args()
    total = build_faiss(args.data_dir, args.index_dir)
    print(f"[DONE] Total documentos indexados: {total}")