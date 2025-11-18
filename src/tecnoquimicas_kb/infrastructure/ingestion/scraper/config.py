"""Configuration dataclass for the web scraping pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration options for the scraping and chunking pipeline."""

    # Input / output
    links_file: str
    out_dir: str = "out"

    # HTTP behaviour
    http_timeout_sec: int = 15
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )

    # Cleaning and chunking
    chunk_size_chars: int = 2400
    overlap_chars: int = 220

    # Crawling options
    follow_internal_links: bool = True
    max_pages_per_domain: int = 25
    same_host_only: bool = True
    allow_query_strings: bool = True

    # Global safety limit
    max_total_pages: int = 200
