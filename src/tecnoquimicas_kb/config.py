# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    links_file: str
    out_dir: str = "out"

    # Selenium
    selenium_wait_sec: int = 10
    headless: bool = True
    window_size: str = "1280,2400"
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )

    # Scroll para páginas dinámicas
    auto_scroll: bool = True
    scroll_pause: float = 1.2
    max_scroll_steps: int = 100

    # Limpieza y chunking
    chunk_size_chars: int = 2400
    overlap_chars: int = 220

    # Crawling
    follow_internal_links: bool = True
    max_pages_per_domain: int = 25
    same_host_only: bool = True
    allow_query_strings: bool = True

    # Salvaguarda global
    max_total_pages: int = 200
