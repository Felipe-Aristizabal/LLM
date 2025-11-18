"""CLI wrapper to run the web scraping pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tecnoquimicas_kb.config.logging_config import setup_logging
from tecnoquimicas_kb.infrastructure.ingestion.scraper.config import (
    PipelineConfig,
)
from tecnoquimicas_kb.infrastructure.ingestion.scraper.pipeline import (
    run_scraping_pipeline,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the scraping CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Tecnoquímicas KB – run the web scraping pipeline based on a "
            "links file containing one URL per line."
        )
    )

    parser.add_argument(
        "--links",
        required=True,
        help=(
            "Path to a text file containing one URL per line. "
            "Lines starting with '#' are ignored as comments."
        ),
    )

    parser.add_argument(
        "--out",
        default="out",
        help=(
            "Base output directory where raw_html/, clean_text/ and chunks/ "
            "subfolders will be created (default: %(default)s)."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2400,
        help="Approximate chunk size in characters for text splitting.",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=220,
        help="Number of overlapping characters between consecutive chunks.",
    )

    parser.add_argument(
        "--crawl",
        action="store_true",
        help="Follow internal links discovered on each page.",
    )

    parser.add_argument(
        "--max-pages-per-domain",
        type=int,
        default=25,
        help="Maximum number of pages to fetch per domain.",
    )

    parser.add_argument(
        "--max-total-pages",
        type=int,
        default=200,
        help="Global safety limit for the total number of pages to crawl.",
    )

    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run the HTTP client in a verbose mode (not headless browser).",
    )

    return parser.parse_args()


def main() -> None:
    """Main entrypoint for the scraping CLI."""
    setup_logging()
    args = parse_args()

    links_path = Path(args.links).expanduser().resolve()
    if not links_path.exists():
        logging.error("Links file not found: %s", links_path)
        sys.exit(1)

    out_dir = Path(args.out).expanduser().resolve()

    cfg = PipelineConfig(
        links_file=str(links_path),
        out_dir=str(out_dir),
        chunk_size_chars=args.chunk_size,
        overlap_chars=args.overlap,
        follow_internal_links=bool(args.crawl),
        max_pages_per_domain=args.max_pages_per_domain,
        max_total_pages=args.max_total_pages,
        # The `headful` flag is kept for compatibility; the current
        # implementation uses `requests` and does not differentiate.
    )

    logging.info("Starting scraping pipeline.")
    logging.info("Links file: %s", cfg.links_file)
    logging.info("Output directory: %s", cfg.out_dir)

    try:
        run_scraping_pipeline(cfg)
    except Exception as exc:  # noqa: BLE001
        logging.exception("Scraping pipeline failed: %s", exc)
        sys.exit(1)

    logging.info("Scraping pipeline finished successfully.")


if __name__ == "__main__":
    # Standard CLI guard
    main()
