"""Full-text search over the corpus.

Without this the corpus is write-only: 76,000 papers you can only reach by
waiting for the advisor to volunteer one. Searching is the other half of using
a library — "did anyone ever do X", "what was that paper called" — and it is
the fastest way to find people worth following.

FTS5 with a contentless index, so the abstracts are not stored twice. Queries
go through :func:`sanitise` rather than reaching FTS5 raw, because its query
language treats plenty of ordinary punctuation as syntax and a stray quote is
a syntax error rather than a search for a quote.
"""

from __future__ import annotations

import html
import re
import sqlite3
from dataclasses import dataclass

# Bare words, quoted phrases, and a leading - to exclude a term. Everything
# else in the FTS5 grammar (NEAR, ^, *, boolean operators, column filters) is
# deliberately not exposed — this is a search box, not a query console.
TOKEN = re.compile(r'-?"[^"]*"|-?[\w][\w\'-]*', re.UNICODE)


# FTS5 marks matched terms by wrapping them in strings we choose. Choosing
# control characters rather than "<mark>" is the whole defence: the snippet is
# built from abstract text, which is not HTML and is not sanitised by arXiv, so
# a paper whose abstract contains a script tag would otherwise be rendered as
# markup by any page that trusts the result. Everything is escaped first and
# the highlight is put back afterwards, so only these two markers survive.
OPEN, CLOSE = "\x02", "\x03"


@dataclass
class Hit:
    paper_id: int
    title: str
    authors: str
    published_at: str | None
    url: str | None
    snippet: str

    @property
    def text(self) -> str:
        """The snippet as plain text, for terminals."""
        return self.snippet.replace(OPEN, "").replace(CLOSE, "")

    @property
    def marked(self) -> str:
        """The snippet as HTML, escaped, with only the highlight re-added."""
        escaped = html.escape(self.snippet)
        return escaped.replace(OPEN, "<mark>").replace(CLOSE, "</mark>")


def sanitise(query: str) -> str:
    """Turn user input into a valid FTS5 MATCH expression.

    Every token is quoted, which makes punctuation inert: ``lattice-based``
    searches for that phrase instead of being read as syntax. A leading ``-``
    excludes a term — spelled ``NOT`` here, because FTS5 has no ``-term`` form
    (that is web-search convention, and feeding it to FTS5 raises "no such
    column" rather than doing anything useful).
    """
    positive: list[str] = []
    negative: list[str] = []

    for raw in TOKEN.findall(query or ""):
        negated = raw.startswith("-")
        term = (raw[1:] if negated else raw).strip('"').replace('"', "")
        if not term:
            continue
        (negative if negated else positive).append(f'"{term}"')

    # Nothing but exclusions matches the whole corpus minus a word, which is
    # not what someone typing "-survey" meant, and scans everything to say so.
    if not positive:
        return ""

    expression = " AND ".join(positive)
    if negative:
        # NOT binds tighter than AND in FTS5, so the exclusions are grouped to
        # apply to the whole query rather than only the last term.
        expression += " NOT (" + " OR ".join(negative) + ")"
    return expression


def search(
    conn: sqlite3.Connection, query: str, limit: int = 25, offset: int = 0
) -> list[Hit]:
    """Best matches for ``query``, ranked by BM25 with the title weighted up."""
    expression = sanitise(query)
    if not expression:
        return []

    # A malformed expression is a bad query, not a broken program: FTS5's
    # grammar has corners this cannot all anticipate, and a search box must
    # never take down the page that hosts it.
    try:
        rows = conn.execute(
            """SELECT p.id, p.title, p.authors, p.published_at, p.url,
                      snippet(papers_fts, 1, char(2), char(3), '…', 24) AS snippet
                 FROM papers_fts
                 JOIN papers p ON p.id = papers_fts.rowid
                WHERE papers_fts MATCH ?
                  AND p.withdrawn_at IS NULL
            ORDER BY bm25(papers_fts, 10.0, 1.0, 3.0)
            LIMIT ? OFFSET ?""",
            (expression, limit, offset),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    from advisor.db import json_list

    return [
        Hit(
            paper_id=row["id"],
            title=row["title"],
            authors=", ".join(json_list(row["authors"])[:4]),
            published_at=row["published_at"],
            url=row["url"],
            snippet=row["snippet"] or "",
        )
        for row in rows
    ]


def count(conn: sqlite3.Connection, query: str) -> int:
    expression = sanitise(query)
    if not expression:
        return 0
    try:
        return conn.execute(
            """SELECT count(*) FROM papers_fts JOIN papers p ON p.id = papers_fts.rowid
                WHERE papers_fts MATCH ? AND p.withdrawn_at IS NULL""",
            (expression,),
        ).fetchone()[0]
    except sqlite3.OperationalError:
        return 0
