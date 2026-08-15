"""Corpus harvest: keep the local ``papers`` table in step with arXiv and ePrint.

Both sources speak OAI-PMH, so both are harvested the same way: ask for
everything modified since a stored cursor and follow resumption tokens to the
end. Because ``from`` filters on *modification* datestamp, this picks up
revised abstracts and late cross-listings as well as new papers — a harvest
keyed on submission date would see a v3 revision as nothing at all.

The cursor is recorded from before the first request and only committed once a
source finishes, so an interrupted run replays rather than skipping a range.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from advisor import db
from advisor.config import Config
from advisor.ingest import arxiv, eprint, oai
from advisor.models import Paper, now, upsert_paper

Progress = Callable[[str], None]


def _silent(message: str) -> None:
    pass


@dataclass
class Result:
    source: str
    seen: int = 0
    added: int = 0
    withdrawn: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        line = f"{self.source:<14} {self.seen:>6} seen  {self.added:>6} new"
        if self.withdrawn:
            line += f"  {self.withdrawn} withdrawn"
        if self.errors:
            line += f"  {len(self.errors)} error(s)"
        return line


# ------------------------------------------------------------------ cursor state


def get_cursor(conn: sqlite3.Connection, source: str) -> str | None:
    row = conn.execute(
        "SELECT cursor FROM harvest_state WHERE source = ?", (source,)
    ).fetchone()
    return row["cursor"] if row else None


def set_cursor(conn: sqlite3.Connection, source: str, cursor: str) -> None:
    conn.execute(
        """INSERT INTO harvest_state (source, cursor, last_run) VALUES (?,?,?)
           ON CONFLICT(source) DO UPDATE SET cursor = excluded.cursor,
                                             last_run = excluded.last_run""",
        (source, cursor, now()),
    )


# --------------------------------------------------------------------- storage


def _store(conn: sqlite3.Connection, papers: list[Paper]) -> tuple[int, int]:
    """Persist a page. Returns (seen, newly created).

    Each page is its own transaction, so an interrupt costs at most one page.
    New rows are counted from the table size rather than from ``upsert_paper``,
    since merging a revision into an existing row is not a new paper.
    """
    if not papers:
        return 0, 0

    with db.transaction(conn):
        before = conn.execute("SELECT count(*) FROM papers").fetchone()[0]
        for paper in papers:
            upsert_paper(conn, paper)
        after = conn.execute("SELECT count(*) FROM papers").fetchone()[0]

    return len(papers), after - before


def _mark_withdrawn(conn: sqlite3.Connection, column: str, ids: list[str]) -> int:
    """Flag papers the source reports as deleted, so they stop being recommended."""
    if not ids:
        return 0

    stamp = now()
    with db.transaction(conn):
        cursor = conn.executemany(
            f"UPDATE papers SET withdrawn_at = ? WHERE {column} = ? AND withdrawn_at IS NULL",
            [(stamp, source_id) for source_id in ids],
        )
    return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0


# --------------------------------------------------------------------- harvest


async def _harvest_source(
    conn: sqlite3.Connection,
    source: str,
    id_column: str,
    base_url: str,
    fetch_page: Callable[[str | None, str | None], object],
    since: str,
    progress: Progress,
) -> Result:
    result = Result(source)

    # Recorded before the first request: anything modified mid-harvest then
    # falls after the cursor and is picked up next run rather than skipped.
    started = datetime.now(timezone.utc).date().isoformat()

    # Asking for a date before the repository's earliest is a hard error, not an
    # empty result — arXiv serves nothing before 2005-09-16. Clamping costs
    # nothing, since those are modification stamps and every paper still appears.
    try:
        identity = await oai.identify(base_url)
    except Exception as exc:
        result.errors.append(f"Identify failed: {exc}")
        return result

    if identity.earliest_datestamp and since < identity.earliest_datestamp:
        progress(
            f"{source}: {since} precedes the repository's earliest record; "
            f"starting at {identity.earliest_datestamp}"
        )
        since = identity.earliest_datestamp

    progress(f"{source}: harvesting changes since {since}")

    token: str | None = None
    while True:
        try:
            batch: oai.Batch = await fetch_page(since if token is None else None, token)
        except Exception as exc:
            result.errors.append(str(exc))
            return result

        seen, added = _store(conn, batch.papers)
        result.seen += seen
        result.added += added
        result.withdrawn += _mark_withdrawn(conn, id_column, batch.deleted)

        if batch.papers or batch.deleted:
            progress(f"  {result.seen} records")

        token = batch.resumption_token
        if not token:
            break

    set_cursor(conn, source, started)
    return result


async def harvest_arxiv(
    conn: sqlite3.Connection,
    cfg: Config,
    category: str,
    progress: Progress = _silent,
) -> Result:
    source = f"arxiv:{category}"
    since = get_cursor(conn, source) or cfg.backfill_from

    async def fetch_page(since_: str | None, token: str | None):
        return await arxiv.list_records(category, since=since_, token=token)

    return await _harvest_source(
        conn, source, "arxiv_id", arxiv.OAI_API, fetch_page, since, progress
    )


async def harvest_eprint(
    conn: sqlite3.Connection,
    cfg: Config,
    progress: Progress = _silent,
) -> Result:
    since = get_cursor(conn, "eprint") or cfg.backfill_from

    async def fetch_page(since_: str | None, token: str | None):
        return await eprint.list_records(since=since_, token=token)

    return await _harvest_source(
        conn, "eprint", "eprint_id", eprint.OAI_API, fetch_page, since, progress
    )


async def harvest_all(
    conn: sqlite3.Connection,
    cfg: Config,
    progress: Progress = _silent,
) -> list[Result]:
    results = []

    for category in cfg.arxiv_categories:
        results.append(await harvest_arxiv(conn, cfg, category, progress))

    if cfg.harvest_eprint:
        results.append(await harvest_eprint(conn, cfg, progress))

    return results
