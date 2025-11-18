"""CLI entrypoint to build or rebuild the vector index."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tecnoquimicas_kb.infrastructure.embeddings.faiss_index import build_faiss_index


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments related to index building."""
    # Create a parser with a helpful description
    parser = argparse.ArgumentParser(
        description=("Build or rebuild the FAISS vector index from precomputed chunks.")
    )

    # Directory containing JSONL/JSON/TXT chunks to be indexed
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("src/tecnoquimicas_kb/data/clean"),
        help=(
            "Directory containing cleaned chunk files (.jsonl/.json/.txt). "
            "Recursively scanned (default: %(default)s)."
        ),
    )

    # Directory where the FAISS index will be stored
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("src/tecnoquimicas_kb/index/faiss"),
        help=(
            "Target directory where the FAISS index will be saved "
            "(default: %(default)s)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Main entrypoint for index construction."""
    # Initialize logging before any other operation
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    data_dir = args.data_dir.resolve()
    index_dir = args.index_dir.resolve()

    logging.info("Building FAISS index.")
    logging.info("Data directory: %s", data_dir)
    logging.info("Index directory: %s", index_dir)

    try:
        # Call build_faiss_index with the user-provided options
        total_docs = build_faiss_index(
            data_dir=str(data_dir),
            index_dir=str(index_dir),
        )
    except Exception as exc:  # noqa: BLE001
        logging.exception("Index construction failed: %s", exc)
        sys.exit(1)

    # Log success or failure and exit with the appropriate status code
    logging.info("Index successfully built. Documents indexed: %s", total_docs)


if __name__ == "__main__":
    # Only run main() when the script is executed directly
    main()
