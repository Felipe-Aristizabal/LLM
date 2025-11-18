"""HTML extraction and cleaning helpers for the scraping pipeline."""

from __future__ import annotations

import re
from typing import Dict, List

from bs4 import BeautifulSoup, Comment

# Basic HTTP headers; the scraper can reuse them for requests.
HEADERS: Dict[str, str] = {
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

# Tags that should always be removed from the DOM before extracting text
REMOVE_TAGS = {
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "canvas",
    "footer",
    "header",
    "nav",
}

# CSS selectors for elements that are usually boilerplate
REMOVE_SELECTORS = [
    "nav",
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


def html_to_clean_text(html: str) -> str:
    """Convert raw HTML into cleaned visible text.

    The function removes scripts, styles and common boilerplate sections,
    then extracts the visible text using '\n' as separator and applies
    basic whitespace normalization.
    """
    soup = BeautifulSoup(html, "lxml")

    # Strip HTML comments
    for node in soup(text=lambda t: isinstance(t, Comment)):
        node.extract()

    # Drop unwanted tags completely
    for tag in soup.find_all(REMOVE_TAGS):
        tag.decompose()

    # Drop elements matching common boilerplate selectors
    for selector in REMOVE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    # Try to focus on the main content; fallback to body or whole document
    root = (
        soup.select_one(
            "main, [role=main], article, section, #content, #main, "
            "[id*=content], [id*=main]"
        )
        or soup.body
        or soup
    )

    text = root.get_text(separator="\n")

    # Normalize carriage returns and consecutive line breaks
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize excessive spaces and non-breaking spaces
    text = re.sub(r"[ \t\u00A0]{2,}", " ", text)

    # Clean each line individually and drop empties
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]

    return "\n".join(lines).strip()


def split_into_paragraphs(text: str) -> List[str]:
    """Split a cleaned text into paragraphs using blank lines as breaks."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def extract_meta_and_ldjson(soup: BeautifulSoup) -> Dict[str, str]:
    """Extract useful meta tags and JSON-LD blobs from a DOM tree."""
    data: Dict[str, str] = {}

    # Page title
    if soup.title and soup.title.string:
        data["title"] = soup.title.string.strip()

    # Meta tags for description and social previews
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

    # JSON-LD structured data blobs
    ldjson_blobs: List[str] = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            txt = (script.string or script.get_text() or "").strip()
            if not txt:
                continue
            ldjson_blobs.append(txt)
        except Exception:
            # Ignore malformed or non-text nodes
            continue

    if ldjson_blobs:
        data["ldjson"] = "\n\n".join(ldjson_blobs)

    return data


def extract_social_content(url: str, html: str, soup: BeautifulSoup) -> str:
    """Build a text representation tailored to social network pages.

    The output emphasizes OpenGraph/Twitter cards, JSON-LD data and the
    visible text for the page. This is useful for LinkedIn/Facebook/etc.
    where the main content may live in meta tags instead of the body.
    """
    meta = extract_meta_and_ldjson(soup)
    chunks: List[str] = []

    def add(key: str) -> None:
        """Append a labeled line for the given meta key if it exists."""
        if key in meta and meta[key].strip():
            chunks.append(f"{key}: {meta[key].strip()}")

    # Title is always useful
    add("title")

    # Common social preview fields
    for key in [
        "description",
        "og:title",
        "og:description",
        "twitter:title",
        "twitter:description",
    ]:
        add(key)

    # Include any JSON-LD payloads when available
    if "ldjson" in meta:
        chunks.append("ldjson:\n" + meta["ldjson"])

    # Finally, append the visible text extracted from the HTML
    visible = html_to_clean_text(html)
    if visible:
        chunks.append("texto_visible:\n" + visible)

    return "\n\n".join(chunks)
