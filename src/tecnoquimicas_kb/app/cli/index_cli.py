"""CLI wrapper to build or rebuild the FAISS vector index."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tecnoquimicas_kb.config.logging_config import setup_logging
from tecnoquimicas_kb.config.settings import settings
from tecnoquimicas_kb.infrastructure.embeddings.faiss_index import (
    build_faiss_index,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments related to index building."""
    parser = argparse.ArgumentParser(
        description=("Build or rebuild the FAISS vector index from cleaned chunks.")
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.paths.data_clean_dir,
        help=(
            "Directory containing cleaned chunk files (.jsonl/.json/.txt). "
            "Recursively scanned. Default: TQ_DATA_CLEAN_DIR setting."
        ),
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=settings.paths.index_dir,
        help=(
            "Directory where the FAISS index will be saved. "
            "Default: TQ_INDEX_DIR setting."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Main entrypoint for the index building CLI."""
    setup_logging()
    args = parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    index_dir = args.index_dir.expanduser().resolve()

    logging.info("Building FAISS index.")
    logging.info("Data directory: %s", data_dir)
    logging.info("Index directory: %s", index_dir)

    try:
        total_docs = build_faiss_index(
            data_dir=str(data_dir),
            index_dir=str(index_dir),
        )
    except Exception as exc:  # noqa: BLE001
        logging.exception("Index construction failed: %s", exc)
        sys.exit(1)

    logging.info("Index successfully built. Documents indexed: %s", total_docs)


if __name__ == "__main__":
    # Standard CLI guard
    main()
