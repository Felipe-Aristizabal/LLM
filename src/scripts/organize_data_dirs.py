"""CLI entrypoint to organize scraped data into the expected folder layout."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tecnoquimicas_kb.infrastructure.ingestion.file_organizer import (
    organize_scraped_data,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for data organization."""
    # Create a parser describing the purpose of the script
    parser = argparse.ArgumentParser(
        description=(
            "Organize scraped data from a temporary 'out' folder into "
            "canonical RAW and CLEAN directories."
        )
    )

    # Input directory produced by the scraper (e.g. contains raw_html/, clean_text/, chunks/)
    parser.add_argument(
        "--in-dir",
        "--in",
        dest="in_dir",
        required=True,
        help=(
            "Input folder produced by the scraper; usually contains "
            "raw_html/, clean_text/ and chunks/ subdirectories."
        ),
    )

    # Base directory for raw HTML (a raw_html/ subfolder will be created here)
    parser.add_argument(
        "--raw-dir",
        required=True,
        help="Base directory where the raw_html/ folder will be placed.",
    )

    # Base directory for clean text and chunks (clean_text/ and chunks/ will go here)
    parser.add_argument(
        "--clean-dir",
        required=True,
        help=("Base directory where clean_text/ and chunks/ folders will be placed."),
    )

    # Optional dry-run flag to preview what would be moved/copied
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually moving any files.",
    )

    return parser.parse_args()


def main() -> None:
    """Coordinate data directory organization based on CLI options."""
    # Configure logging so the user can see what is happening
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    in_dir = Path(args.in_dir).resolve()
    raw_base = Path(args.raw_dir).resolve()
    clean_base = Path(args.clean_dir).resolve()

    logging.info("Organizing scraped data.")
    logging.info("Input directory: %s", in_dir)
    logging.info("Raw base directory: %s", raw_base)
    logging.info("Clean base directory: %s", clean_base)
    if args.dry_run:
        logging.info("Dry-run mode enabled (no filesystem changes will be applied).")

    try:
        # Call organize_scraped_data with the parsed options
        result = organize_scraped_data(
            in_dir=in_dir,
            clean_base=clean_base,
            raw_base=raw_base,
            dry_run=bool(args.dry_run),
        )
    except TypeError:
        # Fallback for a simpler signature if your implementation differs
        # This keeps the CLI usable while you evolve the underlying function.
        result = organize_scraped_data(in_dir, clean_base, raw_base)  # type: ignore[call-arg]
    except Exception as exc:  # noqa: BLE001
        logging.exception("Data organization failed: %s", exc)
        sys.exit(1)

    # Log a summary of moved, copied or skipped files (if available)
    if isinstance(result, dict):
        # Convention: the organizer may return a dict of counters
        for key, value in result.items():
            logging.info("  %s: %s", key, value)

    logging.info("Data organization completed successfully.")


if __name__ == "__main__":
    # Execute the main function when the script is invoked directly
    main()
