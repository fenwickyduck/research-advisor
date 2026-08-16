"""Rating reasons, and how each one is meant to move the recommendations."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from advisor.config import Config
from advisor.embed import store
from advisor.models import Paper, now, upsert_paper
from advisor.recommend import feedback, retrieve


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(data_dir=tmp_path, n_clusters=1, recency_boost=0.05)


def build(conn: sqlite3.Connection, cfg: Config, vectors: dict[str, list[float]]):
    ids = {}
    for i, title in enumerate(vectors):
        ids[title] = upsert_paper(
            conn, Paper(title=title, authors=[title], arxiv_id=f"x.{i:05d}")
        )
    matrix = store.normalize(np.array(list(vectors.values()), dtype=np.float32))
    start = store.append(cfg.vectors_path, matrix)
    store.record_rows(conn, [ids[t] for t in vectors], start, cfg.embedding_model)
    return ids, store.load(cfg.vectors_path)


def add_to_library(conn: sqlite3.Connection, paper_id: int) -> None:
    conn.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (paper_id, now()),
    )


# ------------------------------------------------------------- reason vocabulary


def test_unlabelled_dislike_gets_the_default_weight() -> None:
    assert feedback.negative_weight([]) == feedback.DEFAULT_WEIGHT


def test_already_known_does_not_move_the_query() -> None:
    """That retrieval was correct — the paper is only redundant."""
    assert feedback.negative_weight(["already-known"]) == 0.0


def test_wrong_area_pushes_at_full_strength() -> None:
    assert feedback.negative_weight(["wrong-area"]) == 1.0


def test_several_reasons_take_the_strongest_not_the_sum() -> None:
    """Two complaints about one paper are still one paper's worth of evidence."""
    weight = feedback.negative_weight(["already-known", "wrong-area"])
    assert weight == 1.0


def test_unknown_tags_are_discarded() -> None:
    """The UI must not be able to inject arbitrary tags into the vocabulary."""
    assert feedback.valid_tags(["wrong-area", "'; DROP TABLE papers--"]) == ["wrong-area"]
    assert feedback.negative_weight(["nonsense"]) == feedback.DEFAULT_WEIGHT


def test_record_round_trips_tags_and_note(conn: sqlite3.Connection) -> None:
    paper_id = upsert_paper(conn, Paper(title="P", authors=["A"], arxiv_id="a.1"))
    feedback.record(conn, paper_id, -1, ["too-theoretical", "bogus"], "  want attacks  ")

    rating, tags = feedback.latest_for(conn)[paper_id]

    assert rating == -1
    assert tags == ["too-theoretical"]
    note = conn.execute("SELECT note FROM feedback WHERE paper_id=?", (paper_id,)).fetchone()
    assert note["note"] == "want attacks"


def test_blank_note_is_stored_as_null(conn: sqlite3.Connection) -> None:
    paper_id = upsert_paper(conn, Paper(title="P", authors=["A"], arxiv_id="a.1"))
    feedback.record(conn, paper_id, -1, [], "   ")

    row = conn.execute("SELECT note FROM feedback WHERE paper_id=?", (paper_id,)).fetchone()
    assert row["note"] is None


# --------------------------------------------------- reasons changing retrieval


def test_already_known_leaves_interests_intact(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """The distinction the plain thumbs-down could not express.

    Rejecting a paper as already-known must not steer the advisor away from the
    subject — it found the right area and deserves to keep looking there.
    """
    ids, matrix = build(conn, cfg, {"seed": [1.0, 0.0], "same_area": [0.98, 0.05],
                                    "elsewhere": [0.0, 1.0]})
    add_to_library(conn, ids["seed"])

    before = retrieve.preference_vectors(conn, matrix, cfg)[0].copy()
    feedback.record(conn, ids["same_area"], -1, ["already-known"], None)
    after = retrieve.preference_vectors(conn, matrix, cfg)[0]

    assert np.allclose(before, after), "an already-known rejection must not move the query"


def test_wrong_area_steers_the_query_away(conn: sqlite3.Connection, cfg: Config) -> None:
    ids, matrix = build(conn, cfg, {"seed": [1.0, 1.0], "offtopic": [0.0, 1.0]})
    add_to_library(conn, ids["seed"])

    before = retrieve.preference_vectors(conn, matrix, cfg)[0].copy()
    feedback.record(conn, ids["offtopic"], -1, ["wrong-area"], None)
    after = retrieve.preference_vectors(conn, matrix, cfg)[0]

    offtopic = np.array([0.0, 1.0], dtype=np.float32)
    assert after @ offtopic < before @ offtopic


def test_a_partial_reason_moves_less_than_a_full_one(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """"Too theoretical" is a milder complaint than "wrong subfield"."""

    def shift(tag: str) -> float:
        fresh = sqlite3_fresh(cfg)
        ids, matrix = build(fresh, cfg, {"seed": [1.0, 1.0], "other": [0.0, 1.0]})
        add_to_library(fresh, ids["seed"])
        before = retrieve.preference_vectors(fresh, matrix, cfg)[0].copy()
        feedback.record(fresh, ids["other"], -1, [tag], None)
        after = retrieve.preference_vectors(fresh, matrix, cfg)[0]
        fresh.close()
        return float(np.linalg.norm(after - before))

    assert shift("too-theoretical") < shift("wrong-area")


def sqlite3_fresh(cfg: Config) -> sqlite3.Connection:
    """A clean database and vector file, for comparing two feedback paths."""
    import uuid

    from advisor import db

    name = uuid.uuid4().hex[:8]
    cfg.vectors_path.unlink(missing_ok=True)
    return db.connect(cfg.data_dir / f"{name}.db")


# ------------------------------------------------------------- recency handling


def test_wanting_newer_leans_the_recency_prior(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """Age is the one complaint the vector space cannot express."""
    ids, matrix = build(conn, cfg, {"a": [1.0, 0.0], "b": [0.9, 0.1]})

    assert retrieve.effective_recency_boost(conn, cfg) == cfg.recency_boost

    feedback.record(conn, ids["b"], -1, ["too-old"], None)

    assert retrieve.effective_recency_boost(conn, cfg) > cfg.recency_boost


def test_recency_lean_is_capped(conn: sqlite3.Connection, cfg: Config) -> None:
    """A run of impatient clicks must not turn the feed reverse-chronological.

    Foundational papers you have not read are a large part of the value.
    """
    vectors = {f"p{i}": [1.0, 0.01 * i] for i in range(12)}
    ids, _ = build(conn, cfg, vectors)
    for name in vectors:
        feedback.record(conn, ids[name], -1, ["too-old"], None)

    assert retrieve.effective_recency_boost(conn, cfg) <= cfg.recency_boost * 3


def test_other_reasons_leave_recency_alone(conn: sqlite3.Connection, cfg: Config) -> None:
    ids, _ = build(conn, cfg, {"a": [1.0, 0.0]})
    feedback.record(conn, ids["a"], -1, ["wrong-area"], None)

    assert retrieve.effective_recency_boost(conn, cfg) == cfg.recency_boost


def test_rating_the_same_paper_twice_counts_once(conn: sqlite3.Connection) -> None:
    """Feedback is append-only, so rows are not the same thing as evidence.

    Clicking one thumb repeatedly must not make the profile look staler than
    it is.
    """
    from advisor.recommend import profile

    first = upsert_paper(conn, Paper(title="A", authors=["A"], arxiv_id="a.1"))
    second = upsert_paper(conn, Paper(title="B", authors=["B"], arxiv_id="b.1"))

    for _ in range(4):
        feedback.record(conn, first, 1)
    feedback.record(conn, second, -1)

    assert conn.execute("SELECT count(*) FROM feedback").fetchone()[0] == 5
    assert profile.feedback_since_last_profile(conn) == 2


def test_rated_papers_counts_opinions_not_clicks(conn: sqlite3.Connection) -> None:
    """Anything shown to a person as "ratings" must count papers, not rows.

    This has now been got wrong twice — on the profile page and on /data — so
    the count lives in one function and is asserted here.
    """
    from advisor.models import Paper, upsert_paper

    first = upsert_paper(conn, Paper(title="A", authors=["A"], arxiv_id="r.1"))
    second = upsert_paper(conn, Paper(title="B", authors=["B"], arxiv_id="r.2"))

    for _ in range(5):
        feedback.record(conn, first, 1)
    feedback.record(conn, first, -1)   # changed your mind: still one paper
    feedback.record(conn, second, 1)

    assert conn.execute("SELECT count(*) FROM feedback").fetchone()[0] == 7
    assert feedback.rated_papers(conn) == 2


def test_every_user_facing_rating_count_agrees(conn: sqlite3.Connection) -> None:
    """The three surfaces that report it must not drift apart again."""
    from advisor import portable
    from advisor.models import Paper, upsert_paper
    from advisor.recommend import profile

    paper = upsert_paper(conn, Paper(title="A", authors=["A"], arxiv_id="r.3"))
    for _ in range(3):
        feedback.record(conn, paper, 1)

    exported = portable.export(conn)

    assert feedback.rated_papers(conn) == 1
    assert profile.feedback_since_last_profile(conn) == 1
    assert len({entry["paper"] for entry in exported["feedback"]}) == 1
