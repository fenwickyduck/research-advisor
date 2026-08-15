"""Harvest orchestration: cursors, resumption-token paging, and withdrawals.

The network is stubbed throughout — these tests are about the control flow that
decides which requests to make and what to do with the answers.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from advisor.config import Config
from advisor.ingest import arxiv, eprint, harvest, oai
from advisor.models import Paper


def run(coro):
    return asyncio.run(coro)


def make_papers(n: int, prefix: str) -> list[Paper]:
    return [
        Paper(title=f"{prefix} paper {i}", authors=[f"Author {i}"], arxiv_id=f"{prefix}.{i:05d}")
        for i in range(n)
    ]


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(data_dir=tmp_path, arxiv_categories=("cs.CR",), backfill_from="2024-01-01")


@pytest.fixture(autouse=True)
def offline_identify(monkeypatch):
    """Every harvest issues an Identify probe; keep it off the network.

    Defaults to an earliest datestamp well before the tests' backfill_from, so
    clamping is a no-op unless a test overrides it.
    """

    async def fake_identify(base_url: str) -> oai.Identity:
        return oai.Identity(earliest_datestamp="1990-01-01")

    monkeypatch.setattr(oai, "identify", fake_identify)


# ------------------------------------------------------------- paging & cursors


def test_arxiv_follows_resumption_tokens(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    pages = [
        oai.Batch(make_papers(2, "a"), [], "tok1"),
        oai.Batch(make_papers(2, "b"), [], "tok2"),
        oai.Batch(make_papers(1, "c"), [], None),
    ]
    calls: list[tuple] = []

    async def fake_list(category, since=None, token=None):
        calls.append((category, since, token))
        return pages[len(calls) - 1]

    monkeypatch.setattr(arxiv, "list_records", fake_list)

    result = run(harvest.harvest_arxiv(conn, cfg, "cs.CR"))

    assert result.seen == 5
    assert result.added == 5
    # `since` only on the first call; a resumption token carries the original
    # request's parameters, and sending both is an OAI protocol error.
    assert calls == [
        ("cs.CR", "2024-01-01", None),
        ("cs.CR", None, "tok1"),
        ("cs.CR", None, "tok2"),
    ]


def test_cursor_is_used_on_the_next_run(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    seen_since: list[str | None] = []

    async def fake_list(category, since=None, token=None):
        seen_since.append(since)
        return oai.Batch(make_papers(1, "x"), [], None)

    monkeypatch.setattr(arxiv, "list_records", fake_list)

    run(harvest.harvest_arxiv(conn, cfg, "cs.CR"))
    cursor = harvest.get_cursor(conn, "arxiv:cs.CR")
    assert cursor is not None

    run(harvest.harvest_arxiv(conn, cfg, "cs.CR"))

    assert seen_since == ["2024-01-01", cursor], "the second run resumes from the cursor"


def test_failure_leaves_the_cursor_unset(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    """A failed run must be retried in full, not silently skipped over."""

    async def failing(category, since=None, token=None):
        raise RuntimeError("OAI 503")

    monkeypatch.setattr(arxiv, "list_records", failing)

    result = run(harvest.harvest_arxiv(conn, cfg, "cs.CR"))

    assert result.errors == ["OAI 503"]
    assert harvest.get_cursor(conn, "arxiv:cs.CR") is None


def test_failure_partway_through_keeps_earlier_pages(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    """Pages commit as they arrive, so a later failure does not lose them."""
    calls = 0

    async def flaky(category, since=None, token=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return oai.Batch(make_papers(3, "a"), [], "tok1")
        raise RuntimeError("connection reset")

    monkeypatch.setattr(arxiv, "list_records", flaky)

    result = run(harvest.harvest_arxiv(conn, cfg, "cs.CR"))

    assert result.added == 3
    assert result.errors
    assert conn.execute("SELECT count(*) FROM papers").fetchone()[0] == 3
    # But the cursor stays unset, so the next run re-reads from the start.
    assert harvest.get_cursor(conn, "arxiv:cs.CR") is None


def test_reharvest_counts_seen_but_not_added(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    async def fake_list(category, since=None, token=None):
        return oai.Batch(make_papers(3, "same"), [], None)

    monkeypatch.setattr(arxiv, "list_records", fake_list)

    first = run(harvest.harvest_arxiv(conn, cfg, "cs.CR"))
    second = run(harvest.harvest_arxiv(conn, cfg, "cs.CR"))

    assert (first.seen, first.added) == (3, 3)
    assert (second.seen, second.added) == (3, 0), "re-harvest must not duplicate rows"
    assert conn.execute("SELECT count(*) FROM papers").fetchone()[0] == 3


# ------------------------------------------------------------------- revisions


def test_a_revised_abstract_replaces_the_stale_one(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    """The whole point of harvesting on modification date.

    A v1 abstract already in the corpus must be replaced when the author posts
    a revision, because the abstract is what gets embedded.
    """
    v1 = Paper(title="A Paper", abstract="First version.", authors=["Ann Author"],
               arxiv_id="2401.00001")
    v2 = Paper(title="A Paper", abstract="Substantially rewritten.", authors=["Ann Author"],
               arxiv_id="2401.00001")
    batches = [oai.Batch([v1], [], None), oai.Batch([v2], [], None)]

    async def fake_list(category, since=None, token=None):
        return batches.pop(0)

    monkeypatch.setattr(arxiv, "list_records", fake_list)

    run(harvest.harvest_arxiv(conn, cfg, "cs.CR"))
    run(harvest.harvest_arxiv(conn, cfg, "cs.CR"))

    row = conn.execute("SELECT * FROM papers WHERE arxiv_id = '2401.00001'").fetchone()
    assert Paper.from_row(row).abstract == "Substantially rewritten."
    assert conn.execute("SELECT count(*) FROM papers").fetchone()[0] == 1


# ------------------------------------------------------------------ withdrawals


def test_withdrawn_papers_are_flagged(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    batches = [
        oai.Batch(make_papers(3, "a"), [], None),
        oai.Batch([], ["a.00001"], None),
    ]

    async def fake_list(category, since=None, token=None):
        return batches.pop(0)

    monkeypatch.setattr(arxiv, "list_records", fake_list)

    run(harvest.harvest_arxiv(conn, cfg, "cs.CR"))
    result = run(harvest.harvest_arxiv(conn, cfg, "cs.CR"))

    assert result.withdrawn == 1
    row = conn.execute("SELECT withdrawn_at FROM papers WHERE arxiv_id='a.00001'").fetchone()
    assert row["withdrawn_at"] is not None
    # The row survives — a paper you have already read stays in your library.
    assert conn.execute("SELECT count(*) FROM papers").fetchone()[0] == 3


def test_withdrawing_an_unknown_paper_is_harmless(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    async def fake_list(category, since=None, token=None):
        return oai.Batch([], ["never.seen"], None)

    monkeypatch.setattr(arxiv, "list_records", fake_list)

    result = run(harvest.harvest_arxiv(conn, cfg, "cs.CR"))

    assert result.withdrawn == 0
    assert not result.errors


# ---------------------------------------------------------------------- ePrint


def test_eprint_harvest_pages_and_sets_its_cursor(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    pages = [
        oai.Batch([Paper(title="A", authors=["X"], eprint_id="2024/0001")], [], "tok1"),
        oai.Batch([Paper(title="B", authors=["Y"], eprint_id="2024/0002")], [], None),
    ]
    calls: list[tuple] = []

    async def fake_list(since=None, token=None):
        calls.append((since, token))
        return pages[len(calls) - 1]

    monkeypatch.setattr(eprint, "list_records", fake_list)

    result = run(harvest.harvest_eprint(conn, cfg))

    assert result.added == 2
    assert calls == [("2024-01-01", None), (None, "tok1")]
    assert harvest.get_cursor(conn, "eprint") is not None


# ------------------------------------------------- repository earliest datestamp


def test_backfill_is_clamped_to_the_repository_earliest(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    """Asking arXiv for anything before 2005-09-16 is a hard badArgument error.

    Clamping loses nothing: OAI datestamps are modification times, so a 1999
    paper is still served, stamped whenever its record was last touched.
    """

    async def fake_identify(base_url: str) -> oai.Identity:
        return oai.Identity(earliest_datestamp="2005-09-16")

    monkeypatch.setattr(oai, "identify", fake_identify)

    asked: list[str | None] = []

    async def fake_list(category, since=None, token=None):
        asked.append(since)
        return oai.Batch([], [], None)

    monkeypatch.setattr(arxiv, "list_records", fake_list)

    early = Config(data_dir=cfg.data_dir, arxiv_categories=("cs.CR",),
                   backfill_from="1996-01-01")
    result = run(harvest.harvest_arxiv(conn, early, "cs.CR"))

    assert asked == ["2005-09-16"]
    assert not result.errors


def test_a_cursor_after_the_earliest_is_left_alone(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    async def fake_identify(base_url: str) -> oai.Identity:
        return oai.Identity(earliest_datestamp="2005-09-16")

    monkeypatch.setattr(oai, "identify", fake_identify)

    asked: list[str | None] = []

    async def fake_list(category, since=None, token=None):
        asked.append(since)
        return oai.Batch([], [], None)

    monkeypatch.setattr(arxiv, "list_records", fake_list)
    run(harvest.harvest_arxiv(conn, cfg, "cs.CR"))

    assert asked == ["2024-01-01"], "a later cursor must not be dragged back"


def test_identify_failure_is_reported_without_harvesting(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    async def failing_identify(base_url: str):
        raise RuntimeError("connection refused")

    async def fake_list(category, since=None, token=None):
        raise AssertionError("must not harvest when Identify failed")

    monkeypatch.setattr(oai, "identify", failing_identify)
    monkeypatch.setattr(arxiv, "list_records", fake_list)

    result = run(harvest.harvest_arxiv(conn, cfg, "cs.CR"))

    assert result.errors and "Identify failed" in result.errors[0]
    assert harvest.get_cursor(conn, "arxiv:cs.CR") is None
