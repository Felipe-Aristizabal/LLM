"""End-to-end scraping pipeline to fetch, clean and chunk web pages."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from tecnoquimicas_kb.infrastructure.ingestion.chunker import chunk_text
from tecnoquimicas_kb.infrastructure.ingestion.scraper.config import (
    PipelineConfig,
)
from tecnoquimicas_kb.infrastructure.ingestion.scraper.extract import (
    HEADERS,
    extract_social_content,
    html_to_clean_text,
    split_into_paragraphs,
)
from tecnoquimicas_kb.infrastructure.ingestion.scraper.url_utils import (
    identify_social,
    is_internal_link,
    normalize_url,
    safe_name_from_url,
)


def _read_links_file(path: Path) -> List[str]:
    """Read a links file containing one URL per line."""
    urls: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        urls.append(stripped)
    return urls


def _fetch_url(url: str, cfg: PipelineConfig) -> str | None:
    """Fetch a URL and return its HTML, or None on error."""
    try:
        resp = requests.get(
            url,
            headers={**HEADERS, "User-Agent": cfg.user_agent},
            timeout=cfg.http_timeout_sec,
        )
        if resp.status_code >= 400:
            print(f"[WARN] {url} -> HTTP {resp.status_code}")
            return None
        return resp.text
    except requests.RequestException as exc:
        print(f"[WARN] error fetching {url}: {exc}")
        return None


def run_scraping_pipeline(cfg: PipelineConfig) -> None:
    """Run the full scraping pipeline for the given configuration.

    Steps:
    1. Load seed URLs from the links file.
    2. Fetch each page, save raw HTML and cleaned text.
    3. Split cleaned text into paragraphs and chunks.
    4. Store chunks as JSONL files and build a consolidated CSV.
    5. Optionally follow internal links up to configured limits.
    """
    links_path = Path(cfg.links_file).expanduser().resolve()
    if not links_path.exists():
        raise FileNotFoundError(f"Links file not found: {links_path}")

    out_base = Path(cfg.out_dir).expanduser().resolve()
    raw_dir = out_base / "raw_html"
    clean_dir = out_base / "clean_text"
    chunks_dir = out_base / "chunks"

    # Ensure output directories exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    seeds = [normalize_url(u) for u in _read_links_file(links_path)]
    seeds = [u for u in seeds if u]

    if not seeds:
        print("[WARN] No valid URLs found in links file.")
        return

    print(f"[INFO] Starting scraper with {len(seeds)} seed URLs.")
    visited: Set[str] = set()
    per_domain_count: Dict[str, int] = defaultdict(int)
    queue: List[str] = list(seeds)
    all_rows: List[Dict[str, Any]] = []

    while queue and len(visited) < cfg.max_total_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if per_domain_count[domain] >= cfg.max_pages_per_domain:
            continue

        html = _fetch_url(url, cfg)
        if html is None:
            continue

        per_domain_count[domain] += 1
        print(f"[PAGE] {url}")

        soup = BeautifulSoup(html, "lxml")
        fname = safe_name_from_url(url)

        # 1) Save raw HTML
        raw_path = raw_dir / f"{fname}.html"
        raw_path.write_text(html, encoding="utf-8")

        # 2) Extract cleaned text, with special handling for social URLs
        if identify_social(url):
            text_for_chunks = extract_social_content(url, html, soup)
        else:
            text_for_chunks = html_to_clean_text(html)

        if not text_for_chunks.strip():
            print(f"[INFO] Empty content after cleaning for {url}")
            continue

        # Save the cleaned text to disk
        clean_path = clean_dir / f"{fname}.txt"
        clean_path.write_text(text_for_chunks, encoding="utf-8")

        # 3) Split into paragraphs and then into chunks
        paragraphs = split_into_paragraphs(text_for_chunks)
        chunks = chunk_text(
            paragraphs,
            chunk_size=cfg.chunk_size_chars,
            overlap=cfg.overlap_chars,
        )

        # 4) Store chunks as JSONL per page and append rows for CSV
        jsonl_path = chunks_dir / f"{fname}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for idx, chunk in enumerate(chunks):
                obj = {
                    "text": chunk,
                    "url": url,
                    "chunk_id": idx,
                    "source": str(jsonl_path),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                all_rows.append(
                    {
                        "url": url,
                        "chunk_id": idx,
                        "text": chunk,
                        "n_chars": len(chunk),
                        "file": str(jsonl_path),
                    }
                )

        # 5) Enqueue internal links if configured to crawl
        if cfg.follow_internal_links:
            try:
                for a in soup.find_all("a", href=True):
                    href = a.get("href", "").strip()
                    if not href:
                        continue
                    if not is_internal_link(
                        url,
                        href,
                        same_host_only=cfg.same_host_only,
                        allow_query_strings=cfg.allow_query_strings,
                    ):
                        continue

                    new_url = normalize_url(urljoin(url, href))
                    if not new_url:
                        continue

                    new_domain = urlparse(new_url).netloc.lower()
                    if per_domain_count[new_domain] >= cfg.max_pages_per_domain:
                        continue

                    if new_url not in visited and new_url not in queue:
                        queue.append(new_url)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] error collecting links in {url}: {exc}")

    # Consolidate all chunks into a CSV summary
    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = out_base / "chunks.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[INFO] Consolidated CSV saved to: {csv_path}")
        print(
            f"[INFO] Total pages: {len(visited)} | Chunks: {len(df)}",
        )
    else:
        print("[WARN] No chunks were produced; nothing to consolidate.")
