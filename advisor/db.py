"""SQLite access and schema.

One database file, WAL mode, foreign keys on. The schema is applied with
``CREATE TABLE IF NOT EXISTS`` and versioned through ``PRAGMA user_version`` so
later phases can add migrations without a framework.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 2

SCHEMA = """
-- The corpus: everything harvested, most of it never seen by you.
CREATE TABLE IF NOT EXISTS papers (
  id            INTEGER PRIMARY KEY,
  title         TEXT NOT NULL,
  abstract      TEXT,
  authors       TEXT,          -- JSON array of strings
  published_at  TEXT,          -- ISO date
  updated_at    TEXT,
  arxiv_id      TEXT UNIQUE,
  eprint_id     TEXT UNIQUE,
  doi           TEXT UNIQUE,
  categories    TEXT,          -- JSON array
  venue         TEXT,
  url           TEXT,
  pdf_url       TEXT,
  title_norm    TEXT,          -- for cross-source dedupe
  withdrawn_at  TEXT           -- set when the source reports the paper deleted
);
CREATE INDEX IF NOT EXISTS idx_papers_title_norm ON papers(title_norm);
CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published_at);

-- Papers you have read / want to read.
CREATE TABLE IF NOT EXISTS library (
  paper_id   INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
  status     TEXT NOT NULL CHECK (status IN ('read', 'queued', 'skipped')),
  added_at   TEXT NOT NULL,
  read_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_library_added ON library(added_at DESC);

-- Your reactions. Append-only: history is kept, never overwritten.
CREATE TABLE IF NOT EXISTS feedback (
  id         INTEGER PRIMARY KEY,
  paper_id   INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  rating     INTEGER NOT NULL CHECK (rating IN (-1, 0, 1)),
  tags       TEXT,               -- JSON array of quick reasons
  note       TEXT,               -- free text: the highest-signal field
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_paper ON feedback(paper_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at DESC);

-- The advisor's model of you: a markdown doc, versioned.
CREATE TABLE IF NOT EXISTS profile_versions (
  id         INTEGER PRIMARY KEY,
  content    TEXT NOT NULL,
  written_by TEXT NOT NULL CHECK (written_by IN ('claude', 'user')),
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  id            INTEGER PRIMARY KEY,
  created_at    TEXT NOT NULL,
  model         TEXT,
  profile_id    INTEGER REFERENCES profile_versions(id),
  n_candidates  INTEGER,
  input_tokens  INTEGER,
  output_tokens INTEGER
);

CREATE TABLE IF NOT EXISTS recommendations (
  id         INTEGER PRIMARY KEY,
  run_id     INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  paper_id   INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  rank       INTEGER,
  score      REAL,
  rationale  TEXT,
  action     TEXT,   -- NULL | 'liked' | 'disliked' | 'queued' | 'dismissed'
  acted_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_recs_paper ON recommendations(paper_id);
CREATE INDEX IF NOT EXISTS idx_recs_run ON recommendations(run_id);

-- paper_id -> row offset in vectors.npy
CREATE TABLE IF NOT EXISTS vector_index (
  paper_id  INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
  row       INTEGER NOT NULL UNIQUE,
  model     TEXT NOT NULL
);

-- Harvest bookkeeping, so incremental runs are cheap.
CREATE TABLE IF NOT EXISTS harvest_state (
  source    TEXT PRIMARY KEY,   -- 'arxiv:cs.CR' | 'eprint'
  cursor    TEXT,
  last_run  TEXT
);
"""


def connect(path: Path) -> sqlite3.Connection:
    """Open (and initialise) the database at ``path``.

    ``check_same_thread=False`` because a single FastAPI request legitimately
    hops threads — a sync dependency runs in the threadpool while an ``async``
    endpoint runs on the event loop. Access is still serialised: each request
    gets its own connection and uses it sequentially, so no two threads ever
    touch one connection at the same time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    _init(conn)
    return conn


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


def _init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    current = conn.execute("PRAGMA user_version").fetchone()[0]

    if current < 2:
        # v2: OAI-PMH harvesting reports withdrawn papers, so record them rather
        # than leaving them to be recommended forever. Existing databases were
        # created before this column, hence the guard.
        if not _column_exists(conn, "papers", "withdrawn_at"):
            conn.execute("ALTER TABLE papers ADD COLUMN withdrawn_at TEXT")

    if current < SCHEMA_VERSION:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Wrap a unit of work. ``isolation_level=None`` means we manage this ourselves."""
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def json_list(value: str | None) -> list:
    """Decode a JSON-array column, tolerating NULL and malformed values."""
    if not value:
        return []
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return []
    return decoded if isinstance(decoded, list) else []
