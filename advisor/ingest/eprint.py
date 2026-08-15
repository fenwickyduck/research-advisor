"""Cryptology ePrint Archive client, via OAI-PMH.

The archive exposes ``oai_dc`` only (verified with ``verb=ListMetadataFormats``),
which is enough: records carry the title, every author, the full abstract in
``dc:description``, and the IACR category in ``dc:subject``. Earliest datestamp
is 1996-01-01, so a complete backfill is possible, and ``deletedRecord`` is
``persistent`` — withdrawals are reported rather than silently disappearing.

Harvesting is permitted subject to attributing IACR and the authors — see
https://eprint.iacr.org/operations.html — which the UI footer does.
"""

from __future__ import annotations

import re

from advisor.ingest import oai
from advisor.models import Paper

OAI_API = "https://eprint.iacr.org/oai"
DC_NS = "http://purl.org/dc/elements/1.1/"

# oai:eprint.iacr.org:2026/1688
_OAI_ID = re.compile(r"((?:19|20)\d{2})/(\d+)$")


def canonical_id(identifier: str) -> str | None:
    """Normalise an ePrint identifier to a zero-padded ``YYYY/NNNN``."""
    match = _OAI_ID.search(identifier.strip())
    if not match:
        return None
    return f"{match.group(1)}/{int(match.group(2)):04d}"


def record_to_paper(record: oai.Record) -> Paper | None:
    eprint_id = canonical_id(record.identifier)
    if not eprint_id:
        return None

    meta = record.metadata
    title = oai.clean(meta.findtext(f"{{{DC_NS}}}title"))
    if not title:
        return None

    authors = [
        name
        for creator in meta.findall(f"{{{DC_NS}}}creator")
        if (name := oai.clean(creator.text))
    ]
    categories = [
        subject
        for element in meta.findall(f"{{{DC_NS}}}subject")
        if (subject := oai.clean(element.text))
    ]

    # ePrint numbers sequentially within a year but oai_dc carries no submission
    # date, so the year from the ID is the only reliable publication signal.
    year = eprint_id.split("/")[0]

    return Paper(
        title=title,
        abstract=oai.paragraphs(meta.findtext(f"{{{DC_NS}}}description")),
        authors=authors,
        published_at=f"{year}-01-01",
        updated_at=record.datestamp[:10] if record.datestamp else None,
        eprint_id=eprint_id,
        categories=categories,
        venue="Cryptology ePrint Archive",
        url=f"https://eprint.iacr.org/{eprint_id}",
        pdf_url=f"https://eprint.iacr.org/{eprint_id}.pdf",
    )


def parse_records(xml: str) -> oai.Batch:
    return oai.parse(xml, record_to_paper, canonical_id)


async def list_records(since: str | None = None, token: str | None = None) -> oai.Batch:
    """One page of records. Pass ``token`` from the previous batch to continue."""
    params = oai.request_params("oai_dc", since=since, token=token)
    return parse_records(await oai.get(OAI_API, params))


async def fetch(eprint_id: str) -> Paper | None:
    """Fetch a single paper, e.g. ``2024/0123``."""
    year, number = eprint_id.split("/")
    xml = await oai.get(
        OAI_API,
        {
            "verb": "GetRecord",
            # ePrint's own identifiers are unpadded, e.g. .../2024/123
            "identifier": f"oai:eprint.iacr.org:{year}/{int(number)}",
            "metadataPrefix": "oai_dc",
        },
    )
    batch = parse_records(xml)
    return batch.papers[0] if batch.papers else None
