"""Crossref lookup, used only to resolve a pasted DOI.

Crossref is not harvested — it is the fallback for a published paper whose
preprint is outside the arXiv categories and ePrint. Abstracts are often absent
or wrapped in JATS markup, both of which are handled here.
"""

from __future__ import annotations

import re

import httpx

from advisor.models import Paper

API = "https://api.crossref.org/works"
# Crossref asks for a contact address in the User-Agent for the polite pool.
USER_AGENT = "research-advisor/0.1 (personal reading recommender; mailto:advisor@example.invalid)"

_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _strip_jats(text: str | None) -> str | None:
    """Crossref abstracts arrive as JATS XML; keep the prose."""
    if not text:
        return None
    return _WS.sub(" ", _TAGS.sub(" ", text)).strip() or None


def _date(item: dict, *keys: str) -> str | None:
    for key in keys:
        parts = (item.get(key) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            y, m, d = (list(parts[0]) + [1, 1])[:3]
            return f"{int(y):04d}-{int(m or 1):02d}-{int(d or 1):02d}"
    return None


def parse_work(item: dict) -> Paper | None:
    titles = item.get("title") or []
    title = _WS.sub(" ", titles[0]).strip() if titles else None
    if not title:
        return None

    authors = []
    for author in item.get("author") or []:
        name = " ".join(
            part for part in (author.get("given"), author.get("family")) if part
        ).strip()
        if name:
            authors.append(name)

    containers = item.get("container-title") or []

    return Paper(
        title=title,
        abstract=_strip_jats(item.get("abstract")),
        authors=authors,
        published_at=_date(item, "published-print", "published-online", "created"),
        doi=item.get("DOI"),
        categories=list(item.get("subject") or []),
        venue=containers[0] if containers else item.get("publisher"),
        url=item.get("URL") or (f"https://doi.org/{item['DOI']}" if item.get("DOI") else None),
    )


async def fetch(doi: str) -> Paper | None:
    async with httpx.AsyncClient(
        timeout=30.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    ) as client:
        response = await client.get(f"{API}/{doi}")

    if response.status_code == 404:
        return None
    response.raise_for_status()

    message = response.json().get("message")
    return parse_work(message) if message else None
