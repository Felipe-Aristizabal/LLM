# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup

from .chunk import chunk_text
from .config import PipelineConfig
from .extract import (
    extract_social_content,
    html_to_clean_text,
    identify_social,
    safe_name_from_url,
    split_into_paragraphs,
)
from .fetch import collect_links, fetch_html_selenium, same_host


def _ensure_dirs(base: Path) -> tuple[Path, Path, Path]:
    raw_dir = base / "raw_html"
    txt_dir = base / "clean_text"
    chunks_dir = base / "chunks"
    for d in (base, raw_dir, txt_dir, chunks_dir):
        d.mkdir(parents=True, exist_ok=True)
    return raw_dir, txt_dir, chunks_dir


def _read_links(path: Path) -> list[str]:
    assert path.exists(), f"No se encontró: {path}"
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tecnoquímicas KB (Selenium-only)")
    parser.add_argument(
        "--links", required=True, help="Ruta a links.txt (una URL por línea)"
    )
    parser.add_argument("--out", default="out", help="Directorio de salida")
    parser.add_argument("--chunk-size", type=int, default=1400)
    parser.add_argument("--overlap", type=int, default=220)
    parser.add_argument("--crawl", action="store_true", help="Seguir enlaces internos")
    parser.add_argument("--max-pages", type=int, default=25)
    parser.add_argument(
        "--headful", action="store_true", help="Ejecutar Chrome con UI (no headless)"
    )
    args = parser.parse_args()

    cfg = PipelineConfig(
        links_file=args.links,
        out_dir=args.out,
        chunk_size_chars=args.chunk_size,
        overlap_chars=args.overlap,
        follow_internal_links=args.crawl,
        max_pages_per_domain=args.max_pages,
        headless=not args.headful,
    )

    links_path = Path(cfg.links_file)
    out_base = Path(cfg.out_dir)
    raw_dir, txt_dir, chunks_dir = _ensure_dirs(out_base)

    urls = _read_links(links_path)
    print(f"Se cargaron {len(urls)} URLs raíz desde {links_path}")

    visited: Set[str] = set()
    queue: List[str] = list(urls)
    all_rows: List[Dict[str, str]] = []

    while queue and len(visited) < cfg.max_total_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        html, final_url, visible_text = fetch_html_selenium(url, cfg)

        if not html:
            print(f"[SKIP] {url}")
            continue

        name = safe_name_from_url(final_url or url)
        (raw_dir / f"{name}.html").write_text(html, encoding="utf-8", errors="ignore")

        soup = BeautifulSoup(html, "lxml")
        is_social = identify_social(final_url or url)

        if is_social:
            combined = extract_social_content(final_url or url, html, soup)
            (txt_dir / f"{name}.txt").write_text(combined, encoding="utf-8")
            paragraphs = split_into_paragraphs(combined)
        else:
            clean_text = html_to_clean_text(html)
            (txt_dir / f"{name}.txt").write_text(clean_text, encoding="utf-8")
            paragraphs = split_into_paragraphs(clean_text)

            if (not clean_text or len(clean_text) < 300) and visible_text:
                # si innerText es mayor, úsalo
                if len(visible_text.strip()) > len(clean_text.strip()):
                    clean_text = visible_text

            (txt_dir / f"{name}.txt").write_text(clean_text, encoding="utf-8")
            paragraphs = split_into_paragraphs(clean_text)

        chunks = chunk_text(
            paragraphs, chunk_size=cfg.chunk_size_chars, overlap=cfg.overlap_chars
        )

        if not chunks:
            fallback_text = "\n\n".join(paragraphs).strip()
            if not fallback_text and (visible_text or clean_text):
                fallback_text = (visible_text or clean_text).strip()
            if fallback_text:
                chunks = [fallback_text[: cfg.chunk_size_chars]]

        with (chunks_dir / f"{name}.jsonl").open("w", encoding="utf-8") as jf:
            for i, c in enumerate(chunks):
                row = {
                    "url": final_url or url,
                    "chunk_id": f"{name}_{i:04d}",
                    "text": c,
                }
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                all_rows.append(row)

        print(f"OK {len(chunks)} chunks -> {chunks_dir / f'{name}.jsonl'}")

        # Crawling interno (solo mismo host, límite por dominio)
        if cfg.follow_internal_links and not is_social:
            try:
                links = collect_links(soup, final_url or url)
                host = urlparse(final_url or url).netloc.lower()
                count_this_host = sum(
                    1 for u in visited if urlparse(u).netloc.lower() == host
                )
                for link in links:
                    if cfg.same_host_only and not same_host(link, final_url or url):
                        continue
                    if not cfg.allow_query_strings and urlparse(link).query:
                        continue
                    if (
                        link not in visited
                        and count_this_host < cfg.max_pages_per_domain
                    ):
                        queue.append(link)
                        count_this_host += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] error recolectando enlaces en {url}: {exc}")

    # Consolidado
    df = pd.DataFrame(all_rows)
    csv_path = out_base / "chunks.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print("Consolidado:", csv_path)
    print("Total URLs procesadas:", len(visited), " | Chunks:", len(df))
