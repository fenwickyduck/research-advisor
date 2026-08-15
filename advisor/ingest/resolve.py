"""Turn whatever the user pasted into a canonical paper reference.

Accepts bare IDs, prefixed IDs, and full URLs for arXiv, the Cryptology ePrint
Archive, and DOIs. Resolution then looks in the local corpus first — after a
harvest most pastes are already there — and only hits the network for papers
outside the harvested categories.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Literal

from advisor.models import Paper

Kind = Literal["arxiv", "eprint", "doi"]


@dataclass(frozen=True)
class Ref:
    kind: Kind
    id: str

    def __str__(self) -> str:
        return {"arxiv": "arXiv:", "eprint": "ePrint ", "doi": "doi:"}[self.kind] + self.id


# arXiv, post-2007 scheme: 2401.12345 or 2401.12345v3 (4 digits + dot + 4-5 digits).
_ARXIV_NEW = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
# arXiv, pre-2007 scheme: cs.CR/0512345, math.GT/0309136, hep-th/9901001.
_ARXIV_OLD = re.compile(r"\b([a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?\b")
# ePrint: 2024/123 — a 4-digit year, a slash, then the sequence number.
_EPRINT = re.compile(r"\b((?:19|20)\d{2})/(\d{1,4})\b")
# DOI: 10.<registrant>/<suffix>.
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)\b")


def parse(text: str) -> Ref | None:
    """Extract a single reference from one line of user input.

    Host names in a URL disambiguate the cases that would otherwise collide —
    ``arxiv.org/abs/2401.12345`` is unambiguous, and a bare ``2024/123`` can
    only be ePrint since arXiv IDs always carry a dot.
    """
    text = text.strip()
    if not text:
        return None

    lowered = text.lower()

    # A DOI is checked first: a DOI suffix can contain digits that look like
    # other schemes, and the 10.x prefix is unambiguous.
    if match := _DOI.search(text):
        # Lowercased so a pasted DOI matches the stored one — see normalize_doi.
        return Ref("doi", match.group(1).rstrip(".,;)").lower())

    if "eprint.iacr.org" in lowered or "ia.cr" in lowered or lowered.startswith("eprint"):
        if match := _EPRINT.search(text):
            return Ref("eprint", f"{match.group(1)}/{int(match.group(2)):04d}")
        return None

    if match := _ARXIV_NEW.search(text):
        return Ref("arxiv", match.group(1))
    if match := _ARXIV_OLD.search(text):
        return Ref("arxiv", match.group(1))

    # No host and no arXiv-shaped ID left: a bare NNNN/NNN must be ePrint.
    if match := _EPRINT.fullmatch(text):
        return Ref("eprint", f"{match.group(1)}/{int(match.group(2)):04d}")

    return None


def parse_many(text: str) -> tuple[list[Ref], list[str]]:
    """Parse a newline-separated paste. Returns (references, unparseable lines)."""
    refs: list[Ref] = []
    seen: set[tuple[str, str]] = set()
    bad: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        ref = parse(line)
        if ref is None:
            bad.append(line)
        elif (ref.kind, ref.id) not in seen:
            seen.add((ref.kind, ref.id))
            refs.append(ref)

    return refs, bad


def find_local(conn: sqlite3.Connection, ref: Ref) -> Paper | None:
    """Look the reference up in the already-harvested corpus."""
    column = {"arxiv": "arxiv_id", "eprint": "eprint_id", "doi": "doi"}[ref.kind]
    row = conn.execute(f"SELECT * FROM papers WHERE {column} = ?", (ref.id,)).fetchone()
    return Paper.from_row(row) if row else None


async def resolve(conn: sqlite3.Connection, ref: Ref) -> Paper | None:
    """Return the paper for ``ref``, fetching it from the source if not local."""
    if local := find_local(conn, ref):
        return local

    if ref.kind == "arxiv":
        from advisor.ingest import arxiv

        return await arxiv.fetch(ref.id)
    if ref.kind == "eprint":
        from advisor.ingest import eprint

        return await eprint.fetch(ref.id)

    # DOIs are resolved through Crossref, which covers the published versions of
    # papers whose preprint never made it into our corpus.
    from advisor.ingest import crossref

    return await crossref.fetch(ref.id)
