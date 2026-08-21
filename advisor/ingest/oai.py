"""A small OAI-PMH client, shared by the arXiv and ePrint harvesters.

OAI-PMH is the right protocol for keeping a local mirror in step with a
repository: ``from`` filters on *modification* datestamp rather than submission
date, so revisions come back automatically, and withdrawn items are reported
explicitly as deleted headers instead of just vanishing.

This module owns the envelope — paging via resumption tokens, error codes, and
the deleted-record convention. Each source supplies its own function to turn a
metadata element into a :class:`~advisor.models.Paper`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from advisor.models import Paper

OAI_NS = "http://www.openarchives.org/OAI/2.0/"
USER_AGENT = "advisor/0.1 (personal reading recommender)"

_WS = re.compile(r"\s+")


def clean(text: str | None) -> str | None:
    """Collapse whitespace, returning None for anything empty."""
    if text is None:
        return None
    return _WS.sub(" ", text).strip() or None


def paragraphs(text: str | None) -> str | None:
    """Collapse whitespace but keep paragraph breaks — abstracts have structure."""
    if text is None:
        return None
    blocks = [_WS.sub(" ", block).strip() for block in text.split("\n\n")]
    return "\n\n".join(block for block in blocks if block) or None


@dataclass
class Record:
    identifier: str
    datestamp: str
    metadata: ET.Element


@dataclass
class Batch:
    papers: list[Paper] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    """Source identifiers of records the repository reports as withdrawn."""
    resumption_token: str | None = None


@dataclass
class Identity:
    earliest_datestamp: str
    """Oldest datestamp the repository will serve, as ``YYYY-MM-DD``.

    This is a *modification* stamp, not a submission date, so clamping a
    backfill to it loses nothing: arXiv reports 2005-09-16 because that is when
    its OAI service began, yet a paper submitted in 1999 is still served — with
    a datestamp from whenever its record was last touched.
    """


class OAIError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"OAI error {code}: {message}")
        self.code = code


async def get(base_url: str, params: dict[str, str], timeout: float = 120.0) -> str:
    async with httpx.AsyncClient(
        timeout=timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    ) as client:
        response = await client.get(base_url, params=params)
    response.raise_for_status()
    return response.text


def parse_identity(xml: str) -> Identity:
    root = ET.fromstring(xml)
    raw = root.findtext(f".//{{{OAI_NS}}}earliestDatestamp") or ""
    # Repositories differ in granularity: ePrint reports a full timestamp,
    # arXiv a bare date. A date is a valid `from` value for both.
    return Identity(earliest_datestamp=(clean(raw) or "")[:10])


async def identify(base_url: str) -> Identity:
    return parse_identity(await get(base_url, {"verb": "Identify"}, timeout=30.0))


def parse(
    xml: str,
    to_paper: Callable[[Record], Paper | None],
    to_identifier: Callable[[str], str | None],
) -> Batch:
    """Parse a ListRecords/GetRecord response.

    ``to_paper`` maps one record's metadata to a Paper; ``to_identifier``
    extracts the source's own ID from an OAI identifier, so deleted records
    (which carry no metadata at all) can still be matched to a local row.
    """
    root = ET.fromstring(xml)

    if (error := root.find(f"{{{OAI_NS}}}error")) is not None:
        code = error.get("code") or "unknown"
        # A cursor with nothing new since it is the normal quiet-day response.
        if code == "noRecordsMatch":
            return Batch()
        raise OAIError(code, (error.text or "").strip())

    batch = Batch()

    for record in root.iter(f"{{{OAI_NS}}}record"):
        header = record.find(f"{{{OAI_NS}}}header")
        if header is None:
            continue

        identifier = header.findtext(f"{{{OAI_NS}}}identifier", default="")
        datestamp = header.findtext(f"{{{OAI_NS}}}datestamp", default="")

        if header.get("status") == "deleted":
            if source_id := to_identifier(identifier):
                batch.deleted.append(source_id)
            continue

        metadata = record.find(f"{{{OAI_NS}}}metadata")
        if metadata is None or len(metadata) == 0:
            continue

        if paper := to_paper(Record(identifier, datestamp, metadata[0])):
            batch.papers.append(paper)

    token = root.find(f".//{{{OAI_NS}}}resumptionToken")
    batch.resumption_token = clean(token.text) if token is not None else None

    return batch


def request_params(
    metadata_prefix: str,
    since: str | None = None,
    set_spec: str | None = None,
    token: str | None = None,
) -> dict[str, str]:
    """Build ListRecords parameters.

    A resumption token encodes the original request, so the protocol forbids
    sending it alongside ``from``/``set``/``metadataPrefix``.
    """
    if token:
        return {"verb": "ListRecords", "resumptionToken": token}

    params = {"verb": "ListRecords", "metadataPrefix": metadata_prefix}
    if since:
        params["from"] = since
    if set_spec:
        params["set"] = set_spec
    return params
