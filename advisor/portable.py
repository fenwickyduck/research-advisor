"""Everything the advisor knows about *you*, as one file you can carry.

The split is the same one :mod:`advisor.reset` draws. The corpus and its
vectors are large, public and rebuildable — anyone can harvest their own. Your
library, ratings, notes, profile and follows are small, private, and exist
nowhere else. This moves the second kind between machines, and keeps it out of
the repository the program is shared from.

The hard part is that paper ids are local autoincrement numbers, so exporting
them would be meaningless on any other install. Each remembered paper travels
with its real identifiers (arXiv, ePrint, DOI) *and* enough metadata to
recreate the row outright — so importing works against a corpus that has not
been harvested yet, or one harvested from different sources, rather than
silently dropping the half it cannot resolve.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from advisor import db
from advisor.models import Paper, now, upsert_paper

FORMAT = "research-advisor/personal"
VERSION = 1


@dataclass
class Report:
    """What an import actually did, per table."""

    papers_created: int = 0
    library: int = 0
    feedback: int = 0
    profiles: int = 0
    authors: int = 0
    skipped: int = 0

    def __str__(self) -> str:
        parts = [
            f"{self.library} library entries",
            f"{self.feedback} ratings",
            f"{self.profiles} profile versions",
            f"{self.authors} followed authors",
        ]
        line = "imported " + ", ".join(parts)
        if self.papers_created:
            line += f"\n{self.papers_created} paper(s) were not in this corpus and were added"
        if self.skipped:
            line += f"\n{self.skipped} entr(y/ies) already present, left alone"
        return line


def _paper_payload(row: sqlite3.Row) -> dict[str, Any]:
    """A paper as identity plus enough metadata to recreate it."""
    return {
        "title": row["title"],
        "abstract": row["abstract"],
        "authors": db.json_list(row["authors"]),
        "categories": db.json_list(row["categories"]),
        "published_at": row["published_at"],
        "arxiv_id": row["arxiv_id"],
        "eprint_id": row["eprint_id"],
        "doi": row["doi"],
        "url": row["url"],
    }


def export(conn: sqlite3.Connection) -> dict[str, Any]:
    """Everything personal, as a JSON-serialisable dict."""
    papers: dict[int, dict[str, Any]] = {}

    def remember(paper_id: int) -> None:
        if paper_id in papers:
            return
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if row is not None:
            papers[paper_id] = _paper_payload(row)

    library = []
    for row in conn.execute("SELECT * FROM library ORDER BY added_at"):
        remember(row["paper_id"])
        library.append(
            {
                "paper": row["paper_id"],
                "status": row["status"],
                "added_at": row["added_at"],
                "read_at": row["read_at"],
            }
        )

    feedback = []
    for row in conn.execute("SELECT * FROM feedback ORDER BY id"):
        remember(row["paper_id"])
        feedback.append(
            {
                "paper": row["paper_id"],
                "rating": row["rating"],
                "tags": db.json_list(row["tags"]),
                "note": row["note"],
                "created_at": row["created_at"],
            }
        )

    profiles = [
        {
            "content": row["content"],
            "written_by": row["written_by"],
            "created_at": row["created_at"],
        }
        for row in conn.execute("SELECT * FROM profile_versions ORDER BY id")
    ]

    authors = [
        {"name": row["name"], "added_at": row["added_at"]}
        for row in conn.execute("SELECT * FROM followed_authors ORDER BY added_at")
    ]

    return {
        "format": FORMAT,
        "version": VERSION,
        "exported_at": now(),
        # Keyed by the local id the entries below refer to. The ids are
        # meaningless elsewhere; they are join keys within this file only.
        "papers": {str(pid): payload for pid, payload in papers.items()},
        "library": library,
        "feedback": feedback,
        "profile_versions": profiles,
        "followed_authors": authors,
    }


def dumps(conn: sqlite3.Connection) -> str:
    return json.dumps(export(conn), indent=2, ensure_ascii=False) + "\n"


def _resolve(conn: sqlite3.Connection, payload: dict[str, Any]) -> tuple[int, bool]:
    """Find this paper in the local corpus, or create it. Returns (id, created).

    ``upsert_paper`` already matches on identifier and on title+author, so it
    both finds an existing row and fills in anything the local copy was
    missing. The only question left is whether a row existed beforehand, which
    is what the caller reports.
    """
    existing = None
    for column in ("arxiv_id", "eprint_id", "doi"):
        value = payload.get(column)
        if value:
            existing = conn.execute(
                f"SELECT id FROM papers WHERE {column} = ?", (value,)
            ).fetchone()
            if existing:
                break

    paper_id = upsert_paper(
        conn,
        Paper(
            title=payload["title"],
            abstract=payload.get("abstract"),
            authors=payload.get("authors") or [],
            categories=payload.get("categories") or [],
            published_at=payload.get("published_at"),
            arxiv_id=payload.get("arxiv_id"),
            eprint_id=payload.get("eprint_id"),
            doi=payload.get("doi"),
            url=payload.get("url"),
        ),
    )
    return paper_id, existing is None


def load(conn: sqlite3.Connection, data: dict[str, Any], replace: bool = False) -> Report:
    """Import a personal export. Merges by default.

    Merging is the default because the common case is carrying your history to
    a machine that already has some — and because an import that silently
    discarded what was already there would be unrecoverable. ``replace`` is
    the explicit way to say otherwise.
    """
    if data.get("format") != FORMAT:
        raise ValueError(
            f"not a research-advisor export (format={data.get('format')!r})"
        )
    if data.get("version", 0) > VERSION:
        raise ValueError(
            f"export is version {data['version']}, this install understands {VERSION}"
        )

    report = Report()

    with db.transaction(conn):
        if replace:
            from advisor.reset import PERSONAL_TABLES

            for table in PERSONAL_TABLES:
                conn.execute(f"DELETE FROM {table}")

        # Map the file's local ids onto this database's ids, once.
        mapping: dict[str, int] = {}
        for key, payload in (data.get("papers") or {}).items():
            paper_id, created = _resolve(conn, payload)
            mapping[key] = paper_id
            report.papers_created += int(created)

        for entry in data.get("library") or []:
            paper_id = mapping.get(str(entry["paper"]))
            if paper_id is None:
                report.skipped += 1
                continue
            cursor = conn.execute(
                """INSERT INTO library (paper_id, status, added_at, read_at)
                   VALUES (?,?,?,?) ON CONFLICT(paper_id) DO NOTHING""",
                (paper_id, entry["status"], entry["added_at"], entry.get("read_at")),
            )
            report.library += cursor.rowcount
            report.skipped += 1 - cursor.rowcount

        for entry in data.get("feedback") or []:
            paper_id = mapping.get(str(entry["paper"]))
            if paper_id is None:
                report.skipped += 1
                continue
            # Feedback is append-only and has no natural key, so re-importing
            # the same file would duplicate every rating. Match on the exact
            # (paper, timestamp) pair instead.
            duplicate = conn.execute(
                "SELECT 1 FROM feedback WHERE paper_id = ? AND created_at = ?",
                (paper_id, entry["created_at"]),
            ).fetchone()
            if duplicate:
                report.skipped += 1
                continue
            conn.execute(
                """INSERT INTO feedback (paper_id, rating, tags, note, created_at)
                   VALUES (?,?,?,?,?)""",
                (
                    paper_id,
                    entry["rating"],
                    json.dumps(entry.get("tags") or [], ensure_ascii=False),
                    entry.get("note"),
                    entry["created_at"],
                ),
            )
            report.feedback += 1

        for entry in data.get("profile_versions") or []:
            duplicate = conn.execute(
                "SELECT 1 FROM profile_versions WHERE content = ? AND created_at = ?",
                (entry["content"], entry["created_at"]),
            ).fetchone()
            if duplicate:
                report.skipped += 1
                continue
            conn.execute(
                """INSERT INTO profile_versions (content, written_by, created_at)
                   VALUES (?,?,?)""",
                (entry["content"], entry.get("written_by") or "user", entry["created_at"]),
            )
            report.profiles += 1

        for entry in data.get("followed_authors") or []:
            from advisor.authors import key as author_key

            key = author_key(entry["name"])
            if not key:
                report.skipped += 1
                continue
            cursor = conn.execute(
                """INSERT INTO followed_authors (key, name, added_at) VALUES (?,?,?)
                   ON CONFLICT(key) DO NOTHING""",
                (key, entry["name"], entry["added_at"]),
            )
            report.authors += cursor.rowcount
            report.skipped += 1 - cursor.rowcount

    return report


def loads(conn: sqlite3.Connection, text: str, replace: bool = False) -> Report:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return load(conn, data, replace=replace)
