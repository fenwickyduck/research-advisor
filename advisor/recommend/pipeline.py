"""Producing a batch of recommendations and recording it.

A *run* is one batch. Recording it matters for more than history: the feed
reads the latest run, and every paper ever recommended is excluded from future
retrieval, so nothing is suggested twice.

Everything here is local. There is no model call and no network access in this
path — a run is retrieval, diversification, and attribution, and it costs
nothing but CPU.
"""

from __future__ import annotations

import sqlite3

from advisor import db
from advisor.config import Config
from advisor.models import now
from advisor.recommend import retrieve


def latest_profile_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT max(id) AS id FROM profile_versions").fetchone()
    return row["id"] if row else None


def _rationale(candidate: retrieve.Candidate, source) -> str | None:
    """Why this paper is here, in the terms that actually chose it."""
    if candidate.via:
        return f"By {candidate.via}, whom you follow."
    return source.sentence() if source else None


def create_run(
    conn: sqlite3.Connection,
    candidates: list[retrieve.Candidate],
    cfg: Config,
    model: str | None = None,
) -> int:
    """Store a batch of retrieval-only recommendations, returning the run id.

    These carry a rationale too, just not a written one: which paper of yours
    each was matched to. It is the same answer a model would be paraphrasing,
    and it costs nothing.
    """
    attributions = retrieve.explain(conn, candidates, cfg)

    with db.transaction(conn):
        cursor = conn.execute(
            """INSERT INTO runs (created_at, model, profile_id, n_candidates)
               VALUES (?,?,?,?)""",
            (now(), model, latest_profile_id(conn), len(candidates)),
        )
        run_id = int(cursor.lastrowid)

        conn.executemany(
            """INSERT INTO recommendations (run_id, paper_id, rank, score, rationale)
               VALUES (?,?,?,?,?)""",
            [
                (
                    run_id,
                    c.paper_id,
                    rank,
                    # A followed-author pick has no similarity score, and
                    # showing one implies it was chosen by similarity.
                    None if c.via else c.score,
                    _rationale(c, attributions.get(c.paper_id)),
                )
                for rank, c in enumerate(candidates, 1)
            ],
        )

    return run_id


def run(
    conn: sqlite3.Connection,
    cfg: Config,
    limit: int | None = None,
) -> tuple[int | None, int]:
    """Generate and store a batch. Returns (run_id, count).

    Retrieval narrows the corpus to a shortlist, MMR spreads it across your
    interests, and each pick is attributed to the paper of yours that matched
    it. That is the whole pipeline.
    """
    shortlist = retrieve.recommend(conn, cfg, limit=cfg.n_candidates)
    if not shortlist:
        return None, 0

    picks = shortlist[: limit or cfg.n_recommendations]
    return create_run(conn, picks, cfg), len(picks)


def latest(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """The current feed: the most recent run, minus anything already acted on."""
    return conn.execute(
        """SELECT r.id AS rec_id, r.rank, r.score, r.rationale, p.*
             FROM recommendations r
             JOIN papers p ON p.id = r.paper_id
            WHERE r.run_id = (SELECT max(id) FROM runs)
              AND r.action IS NULL
            ORDER BY r.rank"""
    ).fetchall()


def record_action(
    conn: sqlite3.Connection, rec_id: int, action: str
) -> int | None:
    """Mark a recommendation acted on. Returns its paper id."""
    row = conn.execute(
        "SELECT paper_id FROM recommendations WHERE id = ?", (rec_id,)
    ).fetchone()
    if row is None:
        return None

    conn.execute(
        "UPDATE recommendations SET action = ?, acted_at = ? WHERE id = ?",
        (action, now(), rec_id),
    )
    return row["paper_id"]
