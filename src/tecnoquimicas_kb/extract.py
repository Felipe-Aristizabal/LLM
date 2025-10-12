# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

import tldextract
from bs4 import BeautifulSoup, Comment

HEADERS: Dict[str, str] = {
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

REMOVE_TAGS = {
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "canvas",
    "form",
    "input",
    "button",
    "select",
    "option",
    "picture",
    "source",
    "track",
}

REMOVE_SELECTORS = [
    "[role='navigation']",
    ".nav",
    ".navbar",
    ".menu",
    ".breadcrumb",
    ".cookie",
    ".gdpr",
    ".advert",
    ".ads",
    ".promo",
    ".newsletter",
    ".subscribe",
    ".share",
    ".social",
]

SOCIAL_HOSTS = {
    "linkedin.com": "linkedin",
    "facebook.com": "facebook",
    "instagram.com": "instagram",
    "twitter.com": "twitter",
    "x.com": "twitter",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
}


def safe_name_from_url(url: str) -> str:
    sha = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    ext = tldextract.extract(url)
    host = ".".join([p for p in [ext.domain, ext.suffix] if p])
    path = urlparse(url).path.strip("/").replace("/", "_")[:60] or "index"
    return f"{host}_{path}_{sha}"


def identify_social(url: str) -> Optional[str]:
    host = urlparse(url).netloc.lower()
    for key, label in SOCIAL_HOSTS.items():
        if key in host:
            return label
    return None


def extract_meta_and_ldjson(soup: BeautifulSoup) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if soup.title and soup.title.string:
        data["title"] = soup.title.string.strip()
    for tag in soup.find_all("meta"):
        name = (tag.get("name") or tag.get("property") or "").strip().lower()
        if not name:
            continue
        val = tag.get("content")
        if not val:
            continue
        if (
            name in {"description", "keywords"}
            or name.startswith("og:")
            or name.startswith("twitter:")
        ):
            data[name] = val.strip()
    ldjson_blobs: List[str] = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            txt = script.string or script.get_text() or ""
            if txt.strip():
                ldjson_blobs.append(txt.strip())
        except Exception:
            continue
    if ldjson_blobs:
        data["ldjson"] = "\n\n".join(ldjson_blobs)
    return data


def html_to_clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for node in soup(text=lambda t: isinstance(t, Comment)):
        node.extract()

    for tag in soup.find_all(REMOVE_TAGS):
        tag.decompose()

    for selector in REMOVE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    root = (
        soup.select_one(
            "main, [role=main], article, section, #content, #main, [id*=content], [id*=main]"
        )
        or soup.body
        or soup
    )

    text = root.get_text(separator="\n")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t\u00A0]{2,}", " ", text)
    text = text.strip()
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and not re.fullmatch(r"[\W_]{1,}", ln)]
    return "\n".join(lines)


def split_into_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def extract_social_content(url: str, html: str, soup: BeautifulSoup) -> str:
    """Énfasis redes: OG/Twitter, JSON-LD, título + texto visible máximo posible."""
    meta = extract_meta_and_ldjson(soup)
    chunks: List[str] = []

    def add(k: str) -> None:
        if k in meta and meta[k].strip():
            chunks.append(f"{k}: {meta[k].strip()}")

    add("title")
    for k in [
        "description",
        "og:title",
        "og:description",
        "twitter:title",
        "twitter:description",
    ]:
        add(k)

    if "ldjson" in meta:
        chunks.append("ldjson:\n" + meta["ldjson"])

    visible = html_to_clean_text(html)
    if visible:
        chunks.append("texto_visible:\n" + visible)

    return "\n\n".join(chunks)
