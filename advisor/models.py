"""Domain objects and the paper-upsert logic shared by every ingestion path."""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_PUNCT = re.compile(r"[^a-z0-9]+")


def normalize_title(title: str) -> str:
    """Fold a title to a dedupe key.

    The same paper on arXiv and ePrint routinely differs in capitalisation,
    hyphenation, and LaTeX residue, so strip everything but alphanumerics.
    """
    folded = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return _PUNCT.sub(" ", folded.lower()).strip()


def normalize_doi(doi: str | None) -> str | None:
    """Case-fold a DOI.

    DOIs are case-insensitive, and registrars disagree in practice — Crossref
    returns ``10.1007/3-540-48910-x_16`` for a DOI printed on the paper as
    ``...-X_16``. Folding at every boundary keeps the UNIQUE constraint and the
    lookup-by-DOI path honest.
    """
    if not doi:
        return None
    return doi.strip().lower() or None


def surnames(authors: list[str]) -> set[str]:
    """Last whitespace-delimited token of each author name, lowercased."""
    out = set()
    for author in authors:
        parts = author.strip().split()
        if parts:
            out.add(parts[-1].lower())
    return out


@dataclass
class Paper:
    title: str
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    published_at: str | None = None
    updated_at: str | None = None
    arxiv_id: str | None = None
    eprint_id: str | None = None
    doi: str | None = None
    categories: list[str] = field(default_factory=list)
    venue: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    id: int | None = None

    def __post_init__(self) -> None:
        # Normalise here rather than in each source module, so every current and
        # future ingestion path gets it for free.
        self.doi = normalize_doi(self.doi)

    @property
    def title_norm(self) -> str:
        return normalize_title(self.title)

    @property
    def source_label(self) -> str:
        parts = []
        if self.arxiv_id:
            parts.append(f"arXiv:{self.arxiv_id}")
        if self.eprint_id:
            parts.append(f"ePrint {self.eprint_id}")
        return " · ".join(parts) or (self.doi or "")

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Paper:
        def jl(value: str | None) -> list:
            if not value:
                return []
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return []
            return decoded if isinstance(decoded, list) else []

        return cls(
            id=row["id"],
            title=row["title"],
            abstract=row["abstract"],
            authors=jl(row["authors"]),
            published_at=row["published_at"],
            updated_at=row["updated_at"],
            arxiv_id=row["arxiv_id"],
            eprint_id=row["eprint_id"],
            doi=row["doi"],
            categories=jl(row["categories"]),
            venue=row["venue"],
            url=row["url"],
            pdf_url=row["pdf_url"],
        )


# Columns that a later sighting of the same paper may fill in or refresh.
_MERGEABLE = (
    "title",
    "abstract",
    "authors",
    "published_at",
    "updated_at",
    "arxiv_id",
    "eprint_id",
    "doi",
    "categories",
    "venue",
    "url",
    "pdf_url",
)


def find_existing(
    conn: sqlite3.Connection, paper: Paper
) -> tuple[sqlite3.Row, str] | tuple[None, None]:
    """Locate an existing row for ``paper``, and report how it was matched.

    The *how* decides the merge policy in :func:`upsert_paper`: an identifier
    match means this is the same record arriving again from the same source (a
    revision), while a title match means a different source describing the same
    work. Those want opposite treatment.

    Identifier matches are exact. The title fallback additionally requires at
    least one shared author surname, so two unrelated papers that happen to
    share a generic title are not merged.
    """
    for column, value in (
        ("arxiv_id", paper.arxiv_id),
        ("eprint_id", paper.eprint_id),
        ("doi", paper.doi),
    ):
        if not value:
            continue
        row = conn.execute(
            f"SELECT * FROM papers WHERE {column} = ?", (value,)
        ).fetchone()
        if row:
            return row, column

    incoming = surnames(paper.authors)
    if not incoming:
        return None, None

    for row in conn.execute(
        "SELECT * FROM papers WHERE title_norm = ?", (paper.title_norm,)
    ):
        if surnames(Paper.from_row(row).authors) & incoming:
            return row, "title"

    return None, None


def upsert_paper(conn: sqlite3.Connection, paper: Paper) -> int:
    """Insert ``paper``, or merge it into the existing row for the same work.

    Two merge policies, chosen by how the existing row was matched:

    * **Same source** (matched on an identifier this paper carries) — the
      source is handing us this record again, so it is authoritative. Content
      fields are replaced. This is what makes a revised abstract actually
      replace the stale one; without it a harvest keyed on modification date
      would fetch the revision and then throw it away.
    * **Cross source** (matched on title and authors) — a different repository
      describing the same work. Fill in gaps and adopt the new identifier, but
      do not let a terser record clobber a fuller one.

    Returns the paper's row id.
    """
    existing, matched_on = find_existing(conn, paper)

    if existing is None:
        cursor = conn.execute(
            """INSERT INTO papers
               (title, abstract, authors, published_at, updated_at, arxiv_id,
                eprint_id, doi, categories, venue, url, pdf_url, title_norm)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                paper.title,
                paper.abstract,
                json.dumps(paper.authors),
                paper.published_at,
                paper.updated_at,
                paper.arxiv_id,
                paper.eprint_id,
                paper.doi,
                json.dumps(paper.categories),
                paper.venue,
                paper.url,
                paper.pdf_url,
                paper.title_norm,
            ),
        )
        return int(cursor.lastrowid)

    merged = Paper.from_row(existing)
    same_source = matched_on != "title"
    changed: dict[str, object] = {}

    for name in _MERGEABLE:
        incoming = getattr(paper, name)
        current = getattr(merged, name)

        if name in ("authors", "categories"):
            # Always a union, keeping existing order and appending anything new
            # — even on a same-source revision. These lists are genuinely
            # multi-source: a paper on both arXiv and ePrint carries arXiv's
            # taxonomy and IACR's, and letting an arXiv revision replace the
            # list wholesale would silently drop the IACR categories.
            combined = list(current) + [c for c in incoming if c not in current]
            if combined != current:
                changed[name] = json.dumps(combined)
        elif incoming and (not current or (same_source and incoming != current)):
            changed[name] = incoming

    if changed:
        assignments = ", ".join(f"{name} = ?" for name in changed)
        conn.execute(
            f"UPDATE papers SET {assignments} WHERE id = ?",
            (*changed.values(), existing["id"]),
        )

    return int(existing["id"])
