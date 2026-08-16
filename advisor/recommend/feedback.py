"""What a rating means, and how much it should move the recommendations.

A bare thumbs-down is ambiguous in a way that matters. "I already know this"
and "wrong field entirely" are close to opposite signals — the first says the
retrieval was *right* and the paper is merely redundant, the second says it was
wrong — yet a plain -1 treats them identically and pushes the query away from
your interests in both cases.

So each reason carries a weight saying how far it should move the preference
vector. Reasons are optional; an unlabelled thumbs-down just gets the default.

Vectors cannot represent "too theoretical" directly, so those reasons act as a
partial push here and are handed to Claude in phase 5, where the free-text note
does most of the work.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from advisor.models import now


@dataclass(frozen=True)
class Reason:
    key: str
    label: str
    weight: float
    """How strongly this pushes the preference vector away, 0.0 to 1.0."""
    recency: bool = False
    """Whether it argues for newer papers rather than different ones."""


# Ordered as they appear in the UI.
REASONS: tuple[Reason, ...] = (
    Reason(
        "already-known",
        "I already know this",
        # Retrieval was correct — the paper is simply redundant. Pushing the
        # query away here would punish the system for being right.
        weight=0.0,
    ),
    Reason("wrong-area", "Wrong subfield", weight=1.0),
    Reason("too-theoretical", "Too theoretical", weight=0.5),
    Reason("too-applied", "Too applied", weight=0.5),
    Reason("not-rigorous", "Not rigorous enough", weight=0.5),
    Reason(
        "too-old",
        "Want newer work",
        # Nothing wrong with the topic, so leave the query alone and lean the
        # recency prior instead.
        weight=0.0,
        recency=True,
    ),
)

BY_KEY = {reason.key: reason for reason in REASONS}
DEFAULT_WEIGHT = 1.0


def valid_tags(tags: list[str]) -> list[str]:
    """Drop anything not in the vocabulary, so the UI cannot inject junk."""
    return [tag for tag in tags if tag in BY_KEY]


def negative_weight(tags: list[str]) -> float:
    """How far a thumbs-down with these reasons should move the query.

    With several reasons the strongest wins rather than the sum: "wrong
    subfield *and* too theoretical" is still one paper's worth of evidence.
    """
    known = [BY_KEY[tag] for tag in tags if tag in BY_KEY]
    if not known:
        return DEFAULT_WEIGHT
    return max(reason.weight for reason in known)


def wants_newer(tags: list[str]) -> bool:
    return any(BY_KEY[tag].recency for tag in tags if tag in BY_KEY)


def record(
    conn: sqlite3.Connection,
    paper_id: int,
    rating: int,
    tags: list[str] | None = None,
    note: str | None = None,
) -> None:
    """Append a feedback event. History is kept so you can change your mind."""
    conn.execute(
        "INSERT INTO feedback (paper_id, rating, tags, note, created_at) VALUES (?,?,?,?,?)",
        (
            paper_id,
            rating,
            json.dumps(valid_tags(tags or [])),
            (note or "").strip() or None,
            now(),
        ),
    )


def rated_papers(conn: sqlite3.Connection) -> int:
    """How many *papers* you have rated.

    Feedback is append-only so you can change your mind, which makes the row
    count a count of clicks rather than of opinions: rating one paper three
    times is one verdict, not three. Anything shown to a person as "ratings"
    should use this; only a deletion count should use the raw rows.
    """
    return conn.execute("SELECT count(DISTINCT paper_id) FROM feedback").fetchone()[0]


def latest_for(conn: sqlite3.Connection) -> dict[int, tuple[int, list[str]]]:
    """Each paper's most recent rating and reasons, keyed by paper id."""
    rows = conn.execute(
        """SELECT paper_id, rating, tags FROM feedback f
            WHERE f.id = (SELECT max(id) FROM feedback g WHERE g.paper_id = f.paper_id)"""
    ).fetchall()

    out: dict[int, tuple[int, list[str]]] = {}
    for row in rows:
        try:
            tags = json.loads(row["tags"] or "[]")
        except json.JSONDecodeError:
            tags = []
        out[row["paper_id"]] = (row["rating"], tags if isinstance(tags, list) else [])
    return out
