
from tecnoquimicas_kb.rag.index_builder import build_faiss

if __name__ == "__main__":
    total = build_faiss()
    print(f"Index listo. Documentos: {total}")
