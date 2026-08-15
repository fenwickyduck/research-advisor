"""Following people, not just topics.

Embeddings answer "what is like what I have read". They cannot answer "what has
this group published since", and that is how a lot of research reading actually
works: you follow perhaps a dozen people whose output you want to see whatever
it happens to be about. A vector search cannot express that, because the paper
you want may not resemble anything in your library yet — which is precisely why
it is worth seeing.

So a followed author's new work is retrieved *directly*, by name, and merged
into the shortlist rather than competing with it on cosine similarity.

Matching is on surname plus given name, lowercased and stripped of accents and
punctuation, because the same person appears as "Henry Corrigan-Gibbs" on arXiv
and "H. Corrigan-Gibbs" on ePrint and a literal match would miss half their
work. Initials are compared as initials only when one side actually abbreviates:
"Ada Rao" matches "A. Rao" but not "Alan Rao", who is a different person.
Collapsing both to ``lee|k`` — the obvious first implementation — turned one
followed author into four on a corpus this size.

What remains is that a bare surname matches everyone who has it, and two people
with the same surname *and* given name are indistinguishable from metadata
alone. Both err toward showing you too much, which is the right direction: an
extra paper costs a glance, a missing one is invisible.
"""

from __future__ import annotations

import sqlite3
import unicodedata

from advisor.db import json_list
from advisor.models import now


def key(name: str) -> str:
    """The match key for an author name: ``'corrigan-gibbs|henry'``.

    Given name kept in full, not reduced to an initial — the reduction is what
    conflates distinct people. Returns "" for anything unusable, which callers
    treat as "no match" rather than a key colliding with every other bad name.
    """
    cleaned = unicodedata.normalize("NFKD", name or "")
    cleaned = "".join(ch for ch in cleaned if not unicodedata.combining(ch))
    parts = [part for part in cleaned.replace(".", " ").split() if part]
    if not parts:
        return ""

    surname = parts[-1].lower().strip("-'")
    if not surname:
        return ""
    # A single-token name has no given name to record; key on the surname.
    given = parts[0].lower().strip("-'") if len(parts) > 1 else ""
    return f"{surname}|{given}"


def matches(followed: str, candidate: str) -> bool:
    """Whether a candidate author key refers to the followed person.

    Abbreviation is only assumed when one side is actually a single letter.
    Two spelled-out given names that differ are two different people.
    """
    if not followed or not candidate:
        return False

    followed_surname, _, followed_given = followed.partition("|")
    candidate_surname, _, candidate_given = candidate.partition("|")
    if followed_surname != candidate_surname:
        return False

    # A bare surname matches anyone with it — but only when *you* typed one.
    # The reverse (metadata giving just "Lee") is not a statement of intent,
    # it is a truncated record, and treating it as a wildcard means every
    # common surname drags in strangers.
    if not followed_given:
        return True
    if not candidate_given:
        return False
    if len(followed_given) == 1 or len(candidate_given) == 1:
        return followed_given[0] == candidate_given[0]
    return followed_given == candidate_given


def follow(conn: sqlite3.Connection, name: str) -> bool:
    """Start following ``name``. False if already followed, or unusable.

    "Already followed" uses :func:`matches`, not key equality: following
    "Wei Chen" and then "W. Chen" is one person asked for twice, and
    storing both would report their papers twice.
    """
    author_key = key(name)
    if not author_key or is_followed(conn, name):
        return False

    conn.execute(
        "INSERT INTO followed_authors (key, name, added_at) VALUES (?,?,?)",
        (author_key, name.strip(), now()),
    )
    return True


def unfollow(conn: sqlite3.Connection, name: str) -> bool:
    """Stop following whoever ``name`` refers to, however it is spelled."""
    candidate = key(name)
    if not candidate:
        return False

    removed = [row["key"] for row in following(conn) if matches(row["key"], candidate)]
    for stored in removed:
        conn.execute("DELETE FROM followed_authors WHERE key = ?", (stored,))
    return bool(removed)


def following(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT key, name, added_at FROM followed_authors ORDER BY name"
    ).fetchall()


def followed_keys(conn: sqlite3.Connection) -> set[str]:
    return {row["key"] for row in conn.execute("SELECT key FROM followed_authors")}


def is_followed(conn: sqlite3.Connection, name: str) -> bool:
    candidate = key(name)
    if not candidate:
        return False
    return any(matches(followed, candidate) for followed in followed_keys(conn))


def papers_by(
    conn: sqlite3.Connection, exclude: set[int] | None = None, limit: int = 50
) -> list[tuple[int, str]]:
    """Recent un-seen papers by anyone you follow, newest first.

    Returns (paper_id, matched author name). Scanning every paper's author list
    in Python would mean decoding 76,000 JSON arrays, so the surname is matched
    in SQL first and only the survivors are decoded — the initial cannot be
    checked in SQL, and a surname alone is too loose to trust.
    """
    keys = followed_keys(conn)
    if not keys:
        return []

    surnames = {author_key.partition("|")[0] for author_key in keys}
    clauses = " OR ".join(["lower(authors) LIKE ?"] * len(surnames))
    patterns = [f"%{surname}%" for surname in surnames]

    rows = conn.execute(
        f"""SELECT id, authors FROM papers
             WHERE withdrawn_at IS NULL AND ({clauses})
             ORDER BY coalesce(published_at, '') DESC
             LIMIT 4000""",
        patterns,
    ).fetchall()

    exclude = exclude or set()
    out: list[tuple[int, str]] = []
    for row in rows:
        if row["id"] in exclude:
            continue
        for name in json_list(row["authors"]):
            candidate = key(name)
            if any(matches(followed, candidate) for followed in keys):
                out.append((row["id"], name))
                break
        if len(out) >= limit:
            break

    return out


def suggestions(conn: sqlite3.Connection, minimum: int = 2) -> list[tuple[str, int]]:
    """Authors appearing more than once in your library, most frequent first.

    Reading two of someone's papers is the evidence that you might want the
    third, so the suggestion costs nothing to compute and needs no judgement.
    Anyone already followed is left out.
    """
    rows = conn.execute(
        """SELECT p.authors FROM library l JOIN papers p ON p.id = l.paper_id
            WHERE p.authors IS NOT NULL"""
    ).fetchall()

    already = followed_keys(conn)
    counts: dict[str, int] = {}
    display: dict[str, str] = {}
    for row in rows:
        # A name repeated within one paper must not count twice.
        for author_key, name in {
            key(name): name for name in json_list(row["authors"]) if key(name)
        }.items():
            if any(matches(followed, author_key) for followed in already):
                continue
            counts[author_key] = counts.get(author_key, 0) + 1
            display.setdefault(author_key, name)

    ranked = sorted(
        ((display[k], n) for k, n in counts.items() if n >= minimum),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return ranked
