from tecnoquimicas_kb.infrastructure.vectorstore.faiss_store import build_index
from tecnoquimicas_kb.infrastructure.ingestion.file_loader import load_all_docs
from tecnoquimicas_kb.config.settings import settings


def main():
    docs = load_all_docs(settings.paths.data_clean_dir)
    build_index(
        docs,
        settings.vector.faiss_index_path,
        settings.vector.faiss_meta_path,
        settings.vector.embedding_model_name,
    )


if __name__ == "__main__":
    main()
