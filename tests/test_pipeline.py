"""Recommendation runs: storing a batch, acting on it, and not repeating it."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from advisor.config import Config
from advisor.embed import store
from advisor.models import Paper, now, upsert_paper
from advisor.recommend import pipeline, retrieve


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(data_dir=tmp_path, n_recommendations=3, n_clusters=2)


def build(conn: sqlite3.Connection, cfg: Config, n: int = 12) -> list[int]:
    """A small corpus of embedded papers, spread over a semicircle."""
    ids, vectors = [], []
    for i in range(n):
        ids.append(
            upsert_paper(conn, Paper(title=f"Paper {i}", authors=[f"A{i}"],
                                     arxiv_id=f"24{i:02d}.00001"))
        )
        angle = i * (np.pi / (2 * n))
        vectors.append([float(np.cos(angle)), float(np.sin(angle))])

    matrix = store.normalize(np.array(vectors, dtype=np.float32))
    start = store.append(cfg.vectors_path, matrix)
    store.record_rows(conn, ids, start, cfg.embedding_model)
    return ids


def test_run_stores_a_batch(conn: sqlite3.Connection, cfg: Config) -> None:
    ids = build(conn, cfg)
    conn.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (ids[0], now()),
    )

    run_id, count = pipeline.run(conn, cfg)

    assert run_id is not None
    assert count == cfg.n_recommendations
    rows = pipeline.latest(conn)
    assert len(rows) == cfg.n_recommendations
    # Ranked best-first, and the seed paper is never recommended back.
    assert [row["rank"] for row in rows] == [1, 2, 3]
    assert ids[0] not in [row["id"] for row in rows]


def test_a_second_run_does_not_repeat_the_first(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """Seeing the same paper twice is what makes a feed feel broken."""
    ids = build(conn, cfg)
    conn.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (ids[0], now()),
    )

    pipeline.run(conn, cfg)
    first = {row["id"] for row in pipeline.latest(conn)}

    pipeline.run(conn, cfg)
    second = {row["id"] for row in pipeline.latest(conn)}

    assert first and second
    assert not (first & second), "runs must not overlap"


def test_acting_on_a_recommendation_removes_it_from_the_feed(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    ids = build(conn, cfg)
    conn.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (ids[0], now()),
    )
    pipeline.run(conn, cfg)

    rec = pipeline.latest(conn)[0]
    paper_id = pipeline.record_action(conn, rec["rec_id"], "dismissed")

    assert paper_id == rec["id"]
    assert rec["rec_id"] not in [r["rec_id"] for r in pipeline.latest(conn)]


def test_run_without_vectors_returns_nothing(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    run_id, count = pipeline.run(conn, cfg)
    assert (run_id, count) == (None, 0)


def test_feedback_from_the_feed_changes_the_next_run(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """The loop that makes this an advisor rather than a search engine.

    Disliking results should move the preference vector, so a later run
    retrieves from a different part of the space.
    """
    ids = build(conn, cfg, n=24)
    conn.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (ids[0], now()),
    )

    matrix = store.load(cfg.vectors_path)
    before = retrieve.preference_vectors(conn, matrix, cfg)[0].copy()

    # Dislike several papers clustered at the far end of the arc.
    for paper_id in ids[-4:]:
        conn.execute(
            "INSERT INTO feedback (paper_id, rating, tags, note, created_at) "
            "VALUES (?,?,?,?,?)",
            (paper_id, -1, "[]", None, now()),
        )

    after = retrieve.preference_vectors(conn, matrix, cfg)[0]

    assert not np.allclose(before, after), "negative feedback must move the query"
