"""Carrying your own data between installs.

The thing these guard is that a paper id is a local autoincrement number: an
export that recorded ids would restore into whatever paper happened to occupy
that row on the other machine.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from advisor import db, portable
from advisor.models import Paper, now, upsert_paper


def seed(conn: sqlite3.Connection) -> None:
    read = upsert_paper(
        conn,
        Paper(title="Doubly-Efficient PIR", abstract="A DEPIR scheme.",
              authors=["Wei Chen"], eprint_id="2025/1305", published_at="2025-01-01"),
    )
    rated = upsert_paper(
        conn,
        Paper(title="Succinct Arguments", abstract="SNARKs.",
              authors=["Ada Rao"], arxiv_id="2401.00001", published_at="2024-01-01"),
    )
    conn.execute(
        "INSERT INTO library (paper_id, status, added_at, read_at) VALUES (?,'read',?,?)",
        (read, now(), now()),
    )
    conn.execute(
        "INSERT INTO feedback (paper_id, rating, tags, note, created_at) VALUES (?,?,?,?,?)",
        (rated, -1, '["wrong-area"]', "I want attacks, not surveys", now()),
    )
    conn.execute(
        "INSERT INTO profile_versions (content, written_by, created_at) VALUES (?,?,?)",
        ("## More of\nlattice PIR", "user", now()),
    )
    conn.execute(
        "INSERT INTO followed_authors (key, name, added_at) VALUES (?,?,?)",
        ("chen|wei", "Wei Chen", now()),
    )
    conn.commit()


@pytest.fixture
def target(tmp_path) -> sqlite3.Connection:
    """A second, empty install."""
    conn = db.connect(tmp_path / "other" / "advisor.db")
    yield conn
    conn.close()


# -------------------------------------------------------------------- exporting


def test_an_export_carries_no_corpus(conn: sqlite3.Connection) -> None:
    """The large, public, rebuildable half stays behind."""
    seed(conn)
    for i in range(20):
        upsert_paper(conn, Paper(title=f"Bulk {i}", authors=["X"], arxiv_id=f"b.{i}"))

    data = portable.export(conn)

    # Only papers actually referenced by personal rows travel.
    assert len(data["papers"]) == 2
    assert "Bulk 0" not in json.dumps(data)


def test_an_export_carries_identifiers_not_row_ids(conn: sqlite3.Connection) -> None:
    seed(conn)
    data = portable.export(conn)

    payloads = list(data["papers"].values())
    assert any(p["eprint_id"] == "2025/1305" for p in payloads)
    assert any(p["arxiv_id"] == "2401.00001" for p in payloads)
    # And enough to recreate the row where the corpus lacks it.
    assert all(p["title"] and p["abstract"] is not None for p in payloads)


# -------------------------------------------------------------------- importing


def test_it_restores_into_an_empty_install(
    conn: sqlite3.Connection, target: sqlite3.Connection
) -> None:
    """The headline case: a fresh machine with nothing harvested yet."""
    seed(conn)
    report = portable.load(target, portable.export(conn))

    assert report.library == 1 and report.feedback == 1
    assert report.profiles == 1 and report.authors == 1
    assert report.papers_created == 2, "papers must be recreated, not dropped"

    row = target.execute(
        "SELECT p.title FROM library l JOIN papers p ON p.id = l.paper_id"
    ).fetchone()
    assert row["title"] == "Doubly-Efficient PIR"


def test_it_matches_papers_the_target_already_harvested(
    conn: sqlite3.Connection, target: sqlite3.Connection
) -> None:
    """The ids differ between installs; the identifiers do not."""
    # Give the target a different id for the same paper.
    for i in range(7):
        upsert_paper(target, Paper(title=f"Filler {i}", authors=["X"], arxiv_id=f"f.{i}"))
    existing = upsert_paper(
        target,
        Paper(title="Doubly-Efficient PIR", authors=["Wei Chen"], eprint_id="2025/1305"),
    )
    seed(conn)
    source_id = conn.execute(
        "SELECT paper_id FROM library"
    ).fetchone()["paper_id"]
    assert source_id != existing, "the two installs must disagree on the id"

    report = portable.load(target, portable.export(conn))

    assert report.papers_created == 1, "only the unseen paper is created"
    assert target.execute(
        "SELECT paper_id FROM library"
    ).fetchone()["paper_id"] == existing


def test_importing_twice_changes_nothing(
    conn: sqlite3.Connection, target: sqlite3.Connection
) -> None:
    """Feedback is append-only and has no natural key — the trap."""
    seed(conn)
    data = portable.export(conn)
    portable.load(target, data)
    second = portable.load(target, data)

    assert (second.library, second.feedback, second.profiles, second.authors) == (0, 0, 0, 0)
    assert target.execute("SELECT count(*) FROM feedback").fetchone()[0] == 1


def test_merging_keeps_what_was_already_there(
    conn: sqlite3.Connection, target: sqlite3.Connection
) -> None:
    seed(conn)
    mine = upsert_paper(target, Paper(title="Mine", authors=["Me"], arxiv_id="m.9"))
    target.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (mine, now()),
    )

    portable.load(target, portable.export(conn))

    titles = {r["title"] for r in target.execute(
        "SELECT p.title FROM library l JOIN papers p ON p.id = l.paper_id")}
    assert titles == {"Mine", "Doubly-Efficient PIR"}


def test_replace_discards_local_data_first(
    conn: sqlite3.Connection, target: sqlite3.Connection
) -> None:
    seed(conn)
    mine = upsert_paper(target, Paper(title="Mine", authors=["Me"], arxiv_id="m.9"))
    target.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (mine, now()),
    )

    portable.load(target, portable.export(conn), replace=True)

    titles = {r["title"] for r in target.execute(
        "SELECT p.title FROM library l JOIN papers p ON p.id = l.paper_id")}
    assert titles == {"Doubly-Efficient PIR"}
    # The corpus is not personal data and survives a replace.
    assert target.execute("SELECT count(*) FROM papers WHERE title='Mine'").fetchone()[0] == 1


def test_ratings_and_notes_survive_the_round_trip(
    conn: sqlite3.Connection, target: sqlite3.Connection
) -> None:
    """The notes are the least reproducible thing in the file."""
    seed(conn)
    portable.load(target, portable.export(conn))

    row = target.execute("SELECT rating, tags, note FROM feedback").fetchone()
    assert row["rating"] == -1
    assert json.loads(row["tags"]) == ["wrong-area"]
    assert row["note"] == "I want attacks, not surveys"


# ---------------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    "text,message",
    [
        ("not json at all", "not valid JSON"),
        ("[1, 2, 3]", "expected a JSON object"),
        ('{"format": "something/else"}', "not an advisor export"),
        ('{"format": "research-advisor/personal", "version": 999}', "version 999"),
    ],
)
def test_junk_is_refused_with_a_reason(
    target: sqlite3.Connection, text: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        portable.loads(target, text)


def test_an_export_written_before_the_rename_still_loads(
    conn: sqlite3.Connection, target: sqlite3.Connection
) -> None:
    """The program was renamed; files people already exported were not."""
    seed(conn)
    data = portable.export(conn)
    data["format"] = "research-advisor/personal"

    portable.load(target, data)

    assert target.execute("SELECT count(*) FROM library").fetchone()[0] > 0


def test_a_refused_import_leaves_nothing_behind(
    conn: sqlite3.Connection, target: sqlite3.Connection
) -> None:
    """A half-applied import would be worse than a rejected one."""
    seed(conn)
    data = portable.export(conn)
    data["feedback"][0]["rating"] = "not a number"

    with pytest.raises(Exception):
        portable.load(target, data)

    assert target.execute("SELECT count(*) FROM library").fetchone()[0] == 0


def test_papers_already_shown_are_not_offered_again_after_a_move(
    conn: sqlite3.Connection, target: sqlite3.Connection
) -> None:
    """Retrieval excludes what it has shown; a move must not reset that.

    Otherwise the first batch on the new machine is everything you already
    looked at and passed over.
    """
    from advisor.recommend import retrieve

    seed(conn)
    shown = upsert_paper(
        conn, Paper(title="Already Shown", authors=["X"], arxiv_id="shown.1")
    )
    run = conn.execute(
        "INSERT INTO runs (created_at, n_candidates) VALUES (?,1)", (now(),)
    ).lastrowid
    conn.execute(
        "INSERT INTO recommendations (run_id, paper_id, rank) VALUES (?,?,1)",
        (run, shown),
    )
    conn.commit()

    report = portable.load(target, portable.export(conn))

    assert report.seen == 1
    local = target.execute(
        "SELECT id FROM papers WHERE arxiv_id = 'shown.1'"
    ).fetchone()["id"]
    assert local in retrieve.excluded_ids(target)


def test_imported_history_does_not_become_the_feed(
    conn: sqlite3.Connection, target: sqlite3.Connection
) -> None:
    """The trap: latest() reads the newest run, so unacted rows would show."""
    from advisor.recommend import pipeline

    seed(conn)
    shown = upsert_paper(conn, Paper(title="Old News", authors=["X"], arxiv_id="old.1"))
    run = conn.execute(
        "INSERT INTO runs (created_at, n_candidates) VALUES (?,1)", (now(),)
    ).lastrowid
    conn.execute(
        "INSERT INTO recommendations (run_id, paper_id, rank) VALUES (?,?,1)",
        (run, shown),
    )
    conn.commit()

    portable.load(target, portable.export(conn))

    assert pipeline.latest(target) == []


def test_seen_papers_are_not_re_recorded_on_a_second_import(
    conn: sqlite3.Connection, target: sqlite3.Connection
) -> None:
    seed(conn)
    shown = upsert_paper(conn, Paper(title="Shown", authors=["X"], arxiv_id="s.9"))
    run = conn.execute(
        "INSERT INTO runs (created_at, n_candidates) VALUES (?,1)", (now(),)
    ).lastrowid
    conn.execute(
        "INSERT INTO recommendations (run_id, paper_id, rank) VALUES (?,?,1)",
        (run, shown),
    )
    conn.commit()
    data = portable.export(conn)

    portable.load(target, data)
    second = portable.load(target, data)

    assert second.seen == 0
    assert target.execute("SELECT count(*) FROM recommendations").fetchone()[0] == 1
