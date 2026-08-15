"""Clearing your data without throwing away the expensive parts.

The corpus and its vectors cost hours to build; your library, ratings and
recommendation history cost a few clicks. Those deserve very different
treatment, so resetting is split in two rather than offered as one button that
destroys everything.

The split matters most when starting over after experimenting: you want the
advisor to forget what it learned about you, not to re-harvest 76,000 papers
and re-encode them overnight.
"""

from __future__ import annotations

import sqlite3

from advisor import db
from advisor.config import Config

# Your side of the database: what you have read, what you thought of it, what
# was recommended, the profile written from all of it, and who you follow.
# Every table that is not corpus belongs here — see test_reset_covers_every_table.
PERSONAL_TABLES = (
    "recommendations",
    "runs",
    "profile_versions",
    "feedback",
    "library",
    "followed_authors",
)

# The corpus side: harvested papers, where the harvest got to, and the vectors.
CORPUS_TABLES = ("vector_index", "harvest_state", "papers")


def counts(conn: sqlite3.Connection, tables: tuple[str, ...]) -> dict[str, int]:
    return {
        table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in tables
    }


def clear_personal(conn: sqlite3.Connection, cfg: Config | None = None) -> dict[str, int]:
    """Forget everything about *you*, keeping the corpus and its vectors.

    Deleted in dependency order so the foreign keys never see a dangling row.
    """
    removed = counts(conn, PERSONAL_TABLES)
    with db.transaction(conn):
        for table in PERSONAL_TABLES:
            conn.execute(f"DELETE FROM {table}")

    # Encoded copies of your own profile phrases — derived from personal data,
    # so it goes with it rather than lingering after everything else is gone.
    if cfg is not None:
        (cfg.data_dir / "profile_steer.npz").unlink(missing_ok=True)
    return removed


def clear_corpus(conn: sqlite3.Connection, cfg: Config) -> dict[str, int]:
    """Discard the harvested corpus, its cursors and its vectors.

    Only safe once the personal tables are gone, since they reference papers —
    :func:`clear_all` is the supported way in.
    """
    removed = counts(conn, CORPUS_TABLES)
    with db.transaction(conn):
        conn.execute("DELETE FROM vector_index")
        conn.execute("DELETE FROM harvest_state")
        conn.execute("DELETE FROM papers")
    cfg.vectors_path.unlink(missing_ok=True)
    return removed


def clear_all(conn: sqlite3.Connection, cfg: Config) -> dict[str, int]:
    """Back to an empty database."""
    removed = clear_personal(conn, cfg)
    removed.update(clear_corpus(conn, cfg))
    return removed
