"""Keeping it running: the embed budget, the lock, resetting, and scheduling.

These are the pieces that only matter once the tool runs unattended, which is
also why they are easy to get wrong — nobody is watching when they misbehave.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from advisor import lock, reset, schedule
from advisor.config import Config
from advisor.embed import encoder, store
from advisor.embed import run as embed_run
from advisor.models import Paper, now, upsert_paper
from advisor.recommend import pipeline


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(data_dir=tmp_path, n_recommendations=3, n_clusters=2)


def corpus(conn: sqlite3.Connection, n: int) -> list[int]:
    return [
        upsert_paper(
            conn,
            Paper(title=f"Paper {i}", authors=[f"A{i}"], abstract="x",
                  arxiv_id=f"24{i:02d}.00001", published_at=f"20{10 + i % 15}-01-01"),
        )
        for i in range(n)
    ]


def fake_encoder(monkeypatch, dims: int = 2, delay: float = 0.0) -> None:
    """Encode without loading SPECTER; optionally pretend each batch is slow."""
    clock = {"t": 0.0}

    def encode(texts, cfg, progress=None):
        clock["t"] += delay
        return np.ones((len(texts), dims), dtype=np.float32)

    monkeypatch.setattr(encoder, "encode", encode)
    if delay:
        monkeypatch.setattr(embed_run.time, "monotonic", lambda: clock["t"])


# ------------------------------------------------------------------ embed budget


def test_embed_stops_at_its_budget(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    """A nightly job that overruns its night is worse than a partial one."""
    corpus(conn, 10)
    fake_encoder(monkeypatch, delay=60.0)  # one minute per batch

    done = embed_run.embed_pending(conn, cfg, batch_size=2, max_seconds=150)

    # Stops at the first boundary past the budget rather than mid-batch.
    assert done == 6
    assert encoder.count_pending(conn) == 4


def test_embed_without_a_budget_finishes(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    corpus(conn, 10)
    fake_encoder(monkeypatch, delay=60.0)

    done = embed_run.embed_pending(conn, cfg, batch_size=2)

    assert done == 10
    assert encoder.count_pending(conn) == 0


def test_a_budgeted_pass_resumes_where_it_stopped(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    """The whole point of stopping early is that the rest is not lost."""
    corpus(conn, 10)
    fake_encoder(monkeypatch, delay=60.0)

    first = embed_run.embed_pending(conn, cfg, batch_size=2, max_seconds=150)
    second = embed_run.embed_pending(conn, cfg, batch_size=2, max_seconds=600)

    assert first + second == 10
    assert store.load(cfg.vectors_path).shape[0] == 10


# ------------------------------------------------------------------------- lock


def test_two_embed_passes_cannot_overlap(cfg: Config) -> None:
    """Both rewrite vectors.npy wholesale, so the loser would erase the winner."""
    with lock.exclusive(cfg.embed_lock_path, "an embed pass"):
        with pytest.raises(lock.Busy, match="already running"):
            with lock.exclusive(cfg.embed_lock_path, "an embed pass"):
                pass


def test_the_lock_is_released_afterwards(cfg: Config) -> None:
    with lock.exclusive(cfg.embed_lock_path):
        pass
    with lock.exclusive(cfg.embed_lock_path):
        pass  # must not raise


# ------------------------------------------------------------------------ reset


def seed_everything(conn: sqlite3.Connection, cfg: Config) -> list[int]:
    ids = corpus(conn, 6)
    matrix = store.normalize(
        np.array([[1.0, 0.02 * i] for i in range(6)], dtype=np.float32)
    )
    store.record_rows(conn, ids, store.append(cfg.vectors_path, matrix), cfg.embedding_model)
    conn.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (ids[0], now()),
    )
    conn.execute(
        "INSERT INTO feedback (paper_id, rating, tags, note, created_at) VALUES (?,?,?,?,?)",
        (ids[1], -1, "[]", "not this", now()),
    )
    conn.execute(
        "INSERT INTO profile_versions (content, written_by, created_at) VALUES (?,?,?)",
        ("interested in lattices", "user", now()),
    )
    pipeline.run(conn, cfg)
    return ids


def test_reset_forgets_you_but_keeps_the_corpus(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """Re-harvesting and re-embedding costs hours; ratings cost a few clicks."""
    seed_everything(conn, cfg)

    removed = reset.clear_personal(conn)

    assert sum(removed.values()) > 0
    for table in reset.PERSONAL_TABLES:
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    # The expensive half survives.
    assert conn.execute("SELECT count(*) FROM papers").fetchone()[0] == 6
    assert conn.execute("SELECT count(*) FROM vector_index").fetchone()[0] == 6
    assert cfg.vectors_path.exists()


def test_reset_all_empties_the_database(conn: sqlite3.Connection, cfg: Config) -> None:
    seed_everything(conn, cfg)

    reset.clear_all(conn, cfg)

    for table in reset.PERSONAL_TABLES + reset.CORPUS_TABLES:
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    assert not cfg.vectors_path.exists()


def test_reset_leaves_a_usable_database(conn: sqlite3.Connection, cfg: Config) -> None:
    """Starting over must not mean recreating the schema by hand."""
    ids = seed_everything(conn, cfg)
    reset.clear_personal(conn)

    conn.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (ids[0], now()),
    )
    run_id, count = pipeline.run(conn, cfg)

    assert run_id is not None and count > 0


# --------------------------------------------------------------------- schedule


def test_the_schedule_points_at_this_installation(cfg: Config) -> None:
    """cron and systemd both run with a minimal PATH, so it must be absolute."""
    command = schedule.executable()

    assert command.is_absolute()
    assert str(command) in schedule.instructions()


def test_the_crontab_line_has_the_right_shape() -> None:
    line = schedule.crontab_line("/opt/advisor", at="06:00")

    assert line.startswith("0 6 * * *")
    assert "/opt/advisor update --quiet" in line
    assert schedule.crontab_line("/opt/advisor", at="23:30").startswith("30 23 * * *")


def test_the_timer_fires_at_the_requested_time() -> None:
    _, timer = schedule.systemd_units("/opt/advisor", at="23:30")

    assert "OnCalendar=*-*-* 23:30:00" in timer
    # Without this a machine asleep at 23:30 simply never updates.
    assert "Persistent=true" in timer


def test_the_service_runs_the_update_job() -> None:
    service, _ = schedule.systemd_units("/opt/advisor")

    assert "ExecStart=/opt/advisor update --quiet" in service


# ----------------------------------------------------------------- update (job)


def test_update_runs_all_three_steps(tmp_path, monkeypatch, capsys) -> None:
    """The scheduled job is only useful if the steps stay joined up.

    Harvesting without embedding leaves new papers invisible to retrieval, and
    embedding without a run leaves them out of the feed.
    """
    from advisor import cli, config, db
    from advisor.ingest import harvest

    cfg = Config(data_dir=tmp_path, n_recommendations=3, n_clusters=2)
    monkeypatch.setattr(config, "load", lambda *a, **k: cfg)
    fake_encoder(monkeypatch)

    harvested: list[int] = []

    async def fake_harvest_all(conn, cfg, progress=None):
        harvested.extend(corpus(conn, 8))
        conn.execute(
            "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
            (harvested[0], now()),
        )
        conn.commit()
        return [harvest.Result(source="arxiv:cs.CR", seen=8, added=8)]

    monkeypatch.setattr(harvest, "harvest_all", fake_harvest_all)

    assert cli.main(["update", "--max-minutes", "5"]) == 0

    conn = db.connect(cfg.db_path)
    assert conn.execute("SELECT count(*) FROM vector_index").fetchone()[0] == 8
    assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    assert pipeline.latest(conn), "the feed must be refreshed, not just embedded"
    conn.close()

    out = capsys.readouterr().out
    assert "harvesting" in out and "embedding" in out and "recommending" in out


def test_update_skips_embedding_when_a_manual_pass_holds_the_lock(
    tmp_path, monkeypatch, capsys
) -> None:
    """A busy lock is not an error — the other pass is doing the same work."""
    from advisor import cli, config
    from advisor.ingest import harvest

    cfg = Config(data_dir=tmp_path, n_recommendations=3, n_clusters=2)
    monkeypatch.setattr(config, "load", lambda *a, **k: cfg)
    fake_encoder(monkeypatch)

    async def fake_harvest_all(conn, cfg, progress=None):
        corpus(conn, 4)
        conn.commit()
        return [harvest.Result(source="arxiv:cs.CR", seen=4, added=4)]

    monkeypatch.setattr(harvest, "harvest_all", fake_harvest_all)

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    with lock.exclusive(cfg.embed_lock_path, "an embed pass"):
        assert cli.main(["update"]) == 0

    assert "skipped" in capsys.readouterr().out


# ------------------------------------------------------- explaining an empty feed


def test_an_unembedded_library_says_so(conn: sqlite3.Connection, cfg: Config) -> None:
    """The case that looks like a broken button.

    Corpus embedded, library seeded, but the library papers have no vectors yet
    — so interests cannot be built and retrieval returns nothing. Telling the
    user to add papers or harvest here is actively wrong.
    """
    from advisor.cli import _why_nothing

    ids = corpus(conn, 4)
    # Embed the corpus but not the paper that was just added to the library.
    matrix = store.normalize(np.array([[1.0, 0.0]] * 3, dtype=np.float32))
    store.record_rows(conn, ids[1:], store.append(cfg.vectors_path, matrix), cfg.embedding_model)
    conn.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (ids[0], now()),
    )

    message = _why_nothing(conn)

    assert "not been encoded" in message
    assert "advisor add" not in message and "harvest" not in message


def test_an_empty_library_says_to_add_papers(conn: sqlite3.Connection) -> None:
    from advisor.cli import _why_nothing

    assert "advisor add" in _why_nothing(conn)


def test_a_corpus_of_only_your_own_papers_says_to_harvest(
    conn: sqlite3.Connection,
) -> None:
    """'advisor add' files its papers under both, so counting rows is not enough."""
    from advisor.cli import _why_nothing

    for paper_id in corpus(conn, 3):
        conn.execute(
            "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
            (paper_id, now()),
        )

    assert "harvest" in _why_nothing(conn)


def test_the_feed_page_explains_an_unembedded_library(tmp_path, monkeypatch) -> None:
    """Same diagnosis in the browser, where the button appeared to do nothing."""
    from fastapi.testclient import TestClient

    from advisor import db
    from advisor.web import app as web

    cfg = Config(data_dir=tmp_path, n_recommendations=3, n_clusters=2)
    conn = db.connect(cfg.db_path)

    ids = corpus(conn, 4)
    matrix = store.normalize(np.array([[1.0, 0.0]] * 3, dtype=np.float32))
    store.record_rows(conn, ids[1:], store.append(cfg.vectors_path, matrix), cfg.embedding_model)
    conn.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (ids[0], now()),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(web, "CFG", cfg)
    html = TestClient(web.app).get("/").text

    assert "not been encoded" in html
    assert "Everything is in place" not in html


def test_reset_covers_every_table(conn: sqlite3.Connection) -> None:
    """The drift that actually happened: a table added after reset was written.

    followed_authors shipped without being added to PERSONAL_TABLES, so
    'advisor reset' reported clearing your data and quietly left your follow
    list behind — and 'import --replace' merged follows instead of replacing
    them. Anything new must land in one list or the other.
    """
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        # FTS5 shadow tables are managed by the virtual table, not by us.
        if not row[0].startswith("papers_fts")
    }

    assert tables == set(reset.PERSONAL_TABLES) | set(reset.CORPUS_TABLES)


def test_clearing_personal_data_forgets_who_you_follow(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    from advisor import authors

    seed_everything(conn, cfg)
    authors.follow(conn, "Wei Chen")

    reset.clear_personal(conn, cfg)

    assert authors.following(conn) == []


def test_clearing_personal_data_removes_the_encoded_profile(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """The steering cache holds vectors of your own words; it is personal too."""
    cache = cfg.data_dir / "profile_steer.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"pretend vectors")

    reset.clear_personal(conn, cfg)

    assert not cache.exists()
