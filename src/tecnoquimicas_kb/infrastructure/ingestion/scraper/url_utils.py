"""URL-related helpers for the scraping pipeline."""

from __future__ import annotations

import hashlib
from typing import Dict, Optional
from urllib.parse import urljoin, urlparse

import tldextract

# Known social hosts to detect special handling for social pages
SOCIAL_HOSTS: Dict[str, str] = {
    "linkedin.com": "linkedin",
    "facebook.com": "facebook",
    "instagram.com": "instagram",
    "twitter.com": "twitter",
    "x.com": "twitter",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
}


def normalize_url(url: str) -> str:
    """Return a normalized absolute-looking URL string.

    The function trims whitespace and adds a default scheme when missing.
    It does not attempt full canonicalization.
    """
    url = url.strip()
    if not url:
        return ""

    parsed = urlparse(url)
    if not parsed.scheme:
        # Assume HTTPS for scheme-less URLs
        url = "https://" + url.lstrip("/")

    return url


def safe_name_from_url(url: str) -> str:
    """Generate a filesystem-safe name derived from the URL.

    The function keeps a short host+path fingerprint plus a hash suffix,
    which makes it convenient to store per-URL files on disk.
    """
    sha = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    ext = tldextract.extract(url)
    host = ".".join(part for part in [ext.domain, ext.suffix] if part)
    path = urlparse(url).path.strip("/").replace("/", "_")[:60] or "index"
    return f"{host}_{path}_{sha}"


def identify_social(url: str) -> Optional[str]:
    """Return a label for the social network, if the URL belongs to one."""
    host = urlparse(url).netloc.lower()
    for key, label in SOCIAL_HOSTS.items():
        if key in host:
            return label
    return None


def is_internal_link(
    base_url: str,
    candidate_url: str,
    *,
    same_host_only: bool = True,
    allow_query_strings: bool = True,
) -> bool:
    """Return True if `candidate_url` should be considered internal.

    Parameters
    ----------
    base_url:
        URL of the current page.
    candidate_url:
        Raw href value found in an <a> tag.
    same_host_only:
        When True, only links pointing to the same host are allowed.
    allow_query_strings:
        When False, links differing only by query parameters are treated
        as the same canonical path.
    """
    if not candidate_url:
        return False

    absolute = urljoin(base_url, candidate_url)
    parsed = urlparse(absolute)

    if parsed.scheme not in {"http", "https"}:
        return False

    base_host = urlparse(base_url).netloc.lower()
    cand_host = parsed.netloc.lower()

    if same_host_only and cand_host != base_host:
        return False

    if not allow_query_strings:
        # Ignore query string differences by clearing them
        absolute = parsed._replace(query="", fragment="").geturl()

    # A simple check: a valid HTTP/HTTPS URL that passed host filters
    return absolute.startswith(("http://", "https://"))
