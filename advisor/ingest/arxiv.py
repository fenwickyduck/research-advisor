"""arXiv client.

Two endpoints, each for what it is designed for:

* **OAI-PMH** (``oaipmh.arxiv.org``) for the corpus. Its ``cs:cs:CR``-style sets
  give per-category harvesting, ``from`` filters on modification datestamp so
  revisions and late cross-lists come back on their own, withdrawals arrive as
  deleted headers, and there is no result ceiling.
* **The search API** (``export.arxiv.org``) for resolving one pasted ID, which
  is a query rather than a bulk transfer — and it handles both the modern
  (``2401.12345``) and pre-2007 (``cs.CR/0512345``) ID schemes.

The search API asks for a 3-second delay between requests, enforced globally
here rather than left to each caller.
"""

from __future__ import annotations

import asyncio
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import httpx

from advisor.ingest import oai
from advisor.models import Paper

SEARCH_API = "https://export.arxiv.org/api/query"
OAI_API = "https://oaipmh.arxiv.org/oai"
MIN_INTERVAL = 3.0  # seconds between search-API requests, per arXiv's guidance
USER_AGENT = "research-advisor/0.1 (personal reading recommender)"

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}
ARXIV_NS = "http://arxiv.org/OAI/arXiv/"

_throttle = asyncio.Lock()
_last_request = 0.0

_VERSION = re.compile(r"v\d+$")
# oai:arXiv.org:2508.12220 — also covers old-style oai:arXiv.org:cs.CR/0512345
_OAI_ID = re.compile(r"^oai:arXiv\.org:(.+)$")


def set_spec(category: str) -> str:
    """``cs.CR`` -> ``cs:cs:CR``, arXiv's OAI set naming."""
    archive, _, subject = category.partition(".")
    return f"{archive}:{archive}:{subject}" if subject else archive


def oai_identifier(identifier: str) -> str | None:
    match = _OAI_ID.match(identifier.strip())
    return match.group(1) if match else None


# ---------------------------------------------------------------- OAI-PMH (bulk)


def _authors(element: ET.Element) -> list[str]:
    """arXiv splits names into forenames and keyname; join in reading order."""
    authors = []
    for author in element.findall(f"{{{ARXIV_NS}}}author"):
        forenames = oai.clean(author.findtext(f"{{{ARXIV_NS}}}forenames"))
        keyname = oai.clean(author.findtext(f"{{{ARXIV_NS}}}keyname"))
        name = " ".join(part for part in (forenames, keyname) if part)
        if name:
            authors.append(name)
    return authors


def record_to_paper(record: oai.Record) -> Paper | None:
    meta = record.metadata

    arxiv_id = oai.clean(meta.findtext(f"{{{ARXIV_NS}}}id"))
    title = oai.clean(meta.findtext(f"{{{ARXIV_NS}}}title"))
    if not arxiv_id or not title:
        return None

    authors_element = meta.find(f"{{{ARXIV_NS}}}authors")
    authors = _authors(authors_element) if authors_element is not None else []

    raw_categories = oai.clean(meta.findtext(f"{{{ARXIV_NS}}}categories")) or ""

    return Paper(
        title=title,
        abstract=oai.paragraphs(meta.findtext(f"{{{ARXIV_NS}}}abstract")),
        authors=authors,
        published_at=oai.clean(meta.findtext(f"{{{ARXIV_NS}}}created")),
        # `updated` is absent for a paper still at v1; fall back to the header.
        updated_at=oai.clean(meta.findtext(f"{{{ARXIV_NS}}}updated")) or record.datestamp,
        arxiv_id=arxiv_id,
        doi=oai.clean(meta.findtext(f"{{{ARXIV_NS}}}doi")),
        categories=raw_categories.split(),
        venue=oai.clean(meta.findtext(f"{{{ARXIV_NS}}}journal-ref")),
        url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def parse_records(xml: str) -> oai.Batch:
    return oai.parse(xml, record_to_paper, oai_identifier)


async def list_records(
    category: str,
    since: str | None = None,
    token: str | None = None,
) -> oai.Batch:
    """One page of records for a category, modified since ``since``."""
    params = oai.request_params(
        "arXiv", since=since, set_spec=set_spec(category), token=token
    )
    return parse_records(await oai.get(OAI_API, params))


# ------------------------------------------------------- search API (single ID)


@dataclass
class Page:
    papers: list[Paper]
    total: int


async def _search_get(params: dict[str, str | int]) -> str:
    global _last_request
    async with _throttle:
        wait = MIN_INTERVAL - (time.monotonic() - _last_request)
        if wait > 0:
            await asyncio.sleep(wait)
        async with httpx.AsyncClient(
            timeout=60.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True
        ) as client:
            response = await client.get(SEARCH_API, params=params)
        _last_request = time.monotonic()
    response.raise_for_status()
    return response.text


def _parse_entry(entry: ET.Element) -> Paper | None:
    title = oai.clean(entry.findtext("atom:title", namespaces=ATOM_NS))
    if not title:
        return None

    # <id> is the abs URL: http://arxiv.org/abs/2401.12345v2
    raw_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
    arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else None
    if not arxiv_id:
        # An <entry> with no usable id is arXiv's shape for "not found".
        return None
    arxiv_id = _VERSION.sub("", arxiv_id)

    authors = [
        name
        for author in entry.findall("atom:author", ATOM_NS)
        if (name := oai.clean(author.findtext("atom:name", namespaces=ATOM_NS)))
    ]
    categories = [
        term for category in entry.findall("atom:category", ATOM_NS)
        if (term := category.get("term"))
    ]

    pdf_url = None
    for link in entry.findall("atom:link", ATOM_NS):
        if link.get("title") == "pdf":
            pdf_url = link.get("href")

    published = entry.findtext("atom:published", namespaces=ATOM_NS)
    updated = entry.findtext("atom:updated", namespaces=ATOM_NS)

    return Paper(
        title=title,
        abstract=oai.clean(entry.findtext("atom:summary", namespaces=ATOM_NS)),
        authors=authors,
        published_at=published[:10] if published else None,
        updated_at=updated[:10] if updated else None,
        arxiv_id=arxiv_id,
        doi=oai.clean(entry.findtext("arxiv:doi", namespaces=ATOM_NS)),
        categories=categories,
        venue=oai.clean(entry.findtext("arxiv:journal_ref", namespaces=ATOM_NS)),
        url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
    )


def parse_feed(xml: str) -> Page:
    root = ET.fromstring(xml)

    papers = []
    for entry in root.findall("atom:entry", ATOM_NS):
        if paper := _parse_entry(entry):
            papers.append(paper)

    raw_total = root.findtext("opensearch:totalResults", namespaces=ATOM_NS)
    try:
        total = int(raw_total) if raw_total else len(papers)
    except ValueError:
        total = len(papers)

    return Page(papers, total)


async def fetch(arxiv_id: str) -> Paper | None:
    """Fetch a single paper by ID, for resolving something the user pasted."""
    page = parse_feed(await _search_get({"id_list": arxiv_id, "max_results": 1}))
    return page.papers[0] if page.papers else None
