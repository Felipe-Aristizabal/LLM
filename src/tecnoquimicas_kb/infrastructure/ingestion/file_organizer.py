"""Helpers to move scraped data into canonical RAW and CLEAN folders."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict


def _copy_dir(
    src: Path,
    dst: Path,
    *,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Copy the contents of one directory into another in a merge-safe way.

    Existing files in the destination are overwritten. Subdirectories are
    created as needed.
    """
    stats = {"files_copied": 0, "dirs_created": 0, "missing_src": 0}

    if not src.exists():
        stats["missing_src"] = 1
        return stats

    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)
    stats["dirs_created"] += 1

    for item in src.glob("*"):
        target = dst / item.name
        if item.is_dir():
            # For nested directories, replicate the folder structure
            if not dry_run:
                target.mkdir(parents=True, exist_ok=True)
            stats["dirs_created"] += 1
        else:
            if not dry_run:
                if target.exists():
                    target.unlink()
                shutil.copy(str(item), str(target))
            stats["files_copied"] += 1

    return stats


def organize_scraped_data(
    in_dir: Path,
    clean_base: Path,
    raw_base: Path,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Move scraper outputs from a temporary folder to canonical locations.

    Expected layout of `in_dir`:
    - in_dir/raw_html/
    - in_dir/clean_text/
    - in_dir/chunks/

    Target layout:
    - raw_base/raw_html/
    - clean_base/clean_text/
    - clean_base/chunks/
    """
    in_dir = in_dir.expanduser().resolve()
    clean_base = clean_base.expanduser().resolve()
    raw_base = raw_base.expanduser().resolve()

    # Source subfolders created by the scraper
    raw_html_src = in_dir / "raw_html"
    clean_text_src = in_dir / "clean_text"
    chunks_src = in_dir / "chunks"

    # Destination canonical folders
    raw_html_dst = raw_base / "raw_html"
    clean_text_dst = clean_base / "clean_text"
    chunks_dst = clean_base / "chunks"

    stats_total: Dict[str, int] = {
        "raw_files_copied": 0,
        "txt_files_copied": 0,
        "chunk_files_copied": 0,
        "dirs_created": 0,
        "missing_sources": 0,
    }

    raw_stats = _copy_dir(raw_html_src, raw_html_dst, dry_run=dry_run)
    clean_stats = _copy_dir(clean_text_src, clean_text_dst, dry_run=dry_run)
    chunk_stats = _copy_dir(chunks_src, chunks_dst, dry_run=dry_run)

    stats_total["raw_files_copied"] = raw_stats["files_copied"]
    stats_total["txt_files_copied"] = clean_stats["files_copied"]
    stats_total["chunk_files_copied"] = chunk_stats["files_copied"]
    stats_total["dirs_created"] = (
        raw_stats["dirs_created"]
        + clean_stats["dirs_created"]
        + chunk_stats["dirs_created"]
    )
    stats_total["missing_sources"] = (
        raw_stats["missing_src"]
        + clean_stats["missing_src"]
        + chunk_stats["missing_src"]
    )

    # Optionally remove the temporary input directory when not in dry-run
    if not dry_run:
        try:
            shutil.rmtree(in_dir)
        except Exception:
            # Best-effort cleanup; a failure here is not critical
            pass

    return stats_total
