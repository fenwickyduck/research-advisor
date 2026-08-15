"""Retrieval: preference vectors, cosine search, exclusions, and MMR.

These use small hand-built vectors rather than a real model, so the assertions
are about the arithmetic and the SQL — which is where the bugs actually are.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from advisor.config import Config
from advisor.embed import store
from advisor.models import Paper, now, upsert_paper
from advisor.recommend import retrieve


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(data_dir=tmp_path, n_clusters=2, n_retrieve_per_cluster=50)


def seed(conn: sqlite3.Connection, vectors: dict[str, list[float]]) -> dict[str, int]:
    """Create papers with the given unit vectors; returns title -> paper_id."""
    ids: dict[str, int] = {}
    rows = []
    for title, vector in vectors.items():
        paper_id = upsert_paper(
            conn, Paper(title=title, authors=[title], arxiv_id=f"id.{len(ids):05d}")
        )
        ids[title] = paper_id
        rows.append(vector)
    return ids


def build_matrix(
    conn: sqlite3.Connection, cfg: Config, vectors: dict[str, list[float]]
) -> tuple[dict[str, int], np.ndarray]:
    ids = seed(conn, vectors)
    matrix = store.normalize(np.array(list(vectors.values()), dtype=np.float32))
    start = store.append(cfg.vectors_path, matrix)
    store.record_rows(conn, [ids[t] for t in vectors], start, cfg.embedding_model)
    return ids, store.load(cfg.vectors_path)


def rate(conn: sqlite3.Connection, paper_id: int, rating: int) -> None:
    conn.execute(
        "INSERT INTO feedback (paper_id, rating, tags, note, created_at) VALUES (?,?,?,?,?)",
        (paper_id, rating, "[]", None, now()),
    )


def add_to_library(conn: sqlite3.Connection, paper_id: int) -> None:
    conn.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (paper_id, now()),
    )


# ------------------------------------------------------------------- normalising


def test_normalize_makes_dot_product_a_cosine() -> None:
    matrix = store.normalize(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    # 3-4-5 triangle against the x axis: cos = 0.6
    assert matrix[0] @ matrix[1] == pytest.approx(0.6)


def test_normalize_survives_a_zero_vector() -> None:
    """A paper with an empty abstract can embed to zeros; that must not divide by zero."""
    out = store.normalize(np.zeros((1, 4), dtype=np.float32))
    assert np.isfinite(out).all()


# ------------------------------------------------------------- preference vectors


def test_liked_papers_drive_the_query(conn: sqlite3.Connection, cfg: Config) -> None:
    ids, matrix = build_matrix(
        conn, cfg, {"crypto": [1.0, 0.0], "vision": [0.0, 1.0], "crypto2": [0.9, 0.1]}
    )
    rate(conn, ids["crypto"], 1)

    queries = retrieve.preference_vectors(conn, matrix, cfg)

    assert queries.shape[0] >= 1
    assert queries[0] @ np.array([1.0, 0.0], dtype=np.float32) > 0.9


def test_disliked_papers_push_the_query_away(conn: sqlite3.Connection, cfg: Config) -> None:
    """Rocchio: a thumbs-down should visibly move the query, not just be ignored."""
    ids, matrix = build_matrix(
        conn, cfg, {"a": [1.0, 1.0], "b": [0.0, 1.0], "c": [1.0, 0.0]}
    )
    rate(conn, ids["a"], 1)

    before = retrieve.preference_vectors(conn, matrix, cfg)[0].copy()
    rate(conn, ids["b"], -1)
    after = retrieve.preference_vectors(conn, matrix, cfg)[0]

    disliked = np.array([0.0, 1.0], dtype=np.float32)
    assert after @ disliked < before @ disliked


def test_library_seeds_preferences_before_any_feedback(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """The first run must work: adding a paper you read is itself a signal."""
    ids, matrix = build_matrix(conn, cfg, {"a": [1.0, 0.0], "b": [0.0, 1.0]})
    add_to_library(conn, ids["a"])

    queries = retrieve.preference_vectors(conn, matrix, cfg)

    assert queries.shape[0] == 1
    assert queries[0] @ np.array([1.0, 0.0], dtype=np.float32) > 0.99


def test_only_the_latest_rating_counts(conn: sqlite3.Connection, cfg: Config) -> None:
    """Feedback is append-only so you can change your mind."""
    ids, matrix = build_matrix(conn, cfg, {"a": [1.0, 0.0], "b": [0.0, 1.0]})
    rate(conn, ids["a"], 1)
    rate(conn, ids["a"], -1)

    liked, disliked = retrieve.rated_ids(conn)

    assert liked == []
    assert disliked == [ids["a"]]


def test_interest_clusters_keep_distinct_tastes_apart() -> None:
    """A single centroid over two subfields lands between them and serves neither."""
    positives = store.normalize(
        np.array([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [0.1, 0.99]], dtype=np.float32)
    )

    centroids = retrieve.interest_clusters(positives, k=2)

    assert centroids.shape[0] == 2
    # Each original interest should be well represented by some centroid.
    for direction in ([1.0, 0.0], [0.0, 1.0]):
        best = max(centroids @ np.array(direction, dtype=np.float32))
        assert best > 0.9


# -------------------------------------------------------------------- the search


def test_search_ranks_by_similarity_and_excludes_seen(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    ids, matrix = build_matrix(
        conn, cfg, {"near": [1.0, 0.0], "mid": [0.7, 0.7], "far": [0.0, 1.0]}
    )
    query = store.normalize(np.array([[1.0, 0.0]], dtype=np.float32))

    results = retrieve.search(conn, matrix, query, cfg, exclude=set())
    assert [c.paper_id for c in results][:2] == [ids["near"], ids["mid"]]

    results = retrieve.search(conn, matrix, query, cfg, exclude={ids["near"]})
    assert ids["near"] not in [c.paper_id for c in results]


def test_already_seen_papers_are_excluded(conn: sqlite3.Connection, cfg: Config) -> None:
    """Library, prior recommendations and rated papers all count as seen.

    Re-recommending something is the fastest way to make the feed feel broken.
    """
    ids, _ = build_matrix(conn, cfg, {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]})
    add_to_library(conn, ids["a"])
    rate(conn, ids["b"], -1)
    conn.execute("INSERT INTO runs (created_at) VALUES (?)", (now(),))
    conn.execute(
        "INSERT INTO recommendations (run_id, paper_id, rank) VALUES (1, ?, 1)", (ids["c"],)
    )

    assert retrieve.excluded_ids(conn) == {ids["a"], ids["b"], ids["c"]}


def test_recency_boost_is_a_nudge_not_a_filter(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """An old foundational paper must still be reachable — that is the point."""
    ids, matrix = build_matrix(conn, cfg, {"old": [1.0, 0.0], "new": [0.6, 0.8]})
    conn.execute("UPDATE papers SET published_at='2005-01-01' WHERE id=?", (ids["old"],))
    conn.execute("UPDATE papers SET published_at='2026-01-01' WHERE id=?", (ids["new"],))

    query = store.normalize(np.array([[1.0, 0.0]], dtype=np.float32))
    boosted = Config(data_dir=cfg.data_dir, recency_boost=0.05)
    results = retrieve.search(conn, matrix, query, boosted, exclude=set())

    # The old paper is a far better match, so a small boost must not unseat it.
    assert results[0].paper_id == ids["old"]
    assert ids["new"] in [c.paper_id for c in results]


# ------------------------------------------------------------------------- MMR


def test_diversify_drops_near_duplicates(conn: sqlite3.Connection, cfg: Config) -> None:
    """Three near-identical papers should not take all the slots.

    Scores here are deliberately in a narrow band, because that is what real
    retrieval produces — a candidate pool's cosines cluster around a similar
    value, so the diversity term is what separates them.
    """
    ids, matrix = build_matrix(
        conn,
        cfg,
        {
            "dup1": [1.0, 0.0],
            "dup2": [0.99, 0.01],
            "dup3": [0.98, 0.02],
            "other": [0.0, 1.0],
        },
    )
    candidates = [
        retrieve.Candidate(ids["dup1"], 0.78),
        retrieve.Candidate(ids["dup2"], 0.77),
        retrieve.Candidate(ids["dup3"], 0.76),
        retrieve.Candidate(ids["other"], 0.72),
    ]

    chosen = [c.paper_id for c in retrieve.diversify(conn, matrix, candidates, limit=2)]

    assert ids["dup1"] in chosen
    assert ids["other"] in chosen, "MMR should reach past the duplicate cluster"


def test_diversify_still_prefers_a_far_better_match(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """Diversity is a tiebreak, not an override.

    When one candidate is dramatically more relevant, MMR should keep it rather
    than trading it away for variety.
    """
    ids, matrix = build_matrix(
        conn, cfg, {"a": [1.0, 0.0], "a2": [0.99, 0.01], "unrelated": [0.0, 1.0]}
    )
    candidates = [
        retrieve.Candidate(ids["a"], 0.95),
        retrieve.Candidate(ids["a2"], 0.94),
        retrieve.Candidate(ids["unrelated"], 0.20),
    ]

    chosen = [c.paper_id for c in retrieve.diversify(conn, matrix, candidates, limit=2)]

    assert ids["unrelated"] not in chosen


def test_diversify_is_a_noop_below_the_limit(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    ids, matrix = build_matrix(conn, cfg, {"a": [1.0, 0.0], "b": [0.0, 1.0]})
    candidates = [retrieve.Candidate(ids["a"], 0.9), retrieve.Candidate(ids["b"], 0.8)]

    assert retrieve.diversify(conn, matrix, candidates, limit=10) == candidates


# ------------------------------------------------------------- model consistency


def test_switching_model_is_detected(conn: sqlite3.Connection, cfg: Config) -> None:
    """Vectors from two models are not comparable; mixing them must not be silent."""
    build_matrix(conn, cfg, {"a": [1.0, 0.0]})

    assert store.model_changed(conn, cfg.embedding_model) is None
    assert store.model_changed(conn, "some/other-model") == cfg.embedding_model


def test_append_rejects_a_dimension_change(conn: sqlite3.Connection, cfg: Config) -> None:
    store.append(cfg.vectors_path, np.ones((2, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="dimension mismatch"):
        store.append(cfg.vectors_path, np.ones((1, 8), dtype=np.float32))


def test_empty_corpus_yields_no_recommendations(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    assert retrieve.recommend(conn, cfg) == []


# ------------------------------------------------------- embedding work ordering


def test_pending_puts_library_papers_first(conn: sqlite3.Connection, cfg: Config) -> None:
    """Without the library embedded there are no preference vectors at all."""
    from advisor.embed import encoder

    old = upsert_paper(conn, Paper(title="Old", authors=["A"], arxiv_id="a.1",
                                   published_at="2005-01-01"))
    new = upsert_paper(conn, Paper(title="New", authors=["B"], arxiv_id="a.2",
                                   published_at="2026-01-01"))
    add_to_library(conn, old)

    first = next(encoder.pending(conn, batch_size=10))

    assert first[0].id == old, "a library paper outranks a newer non-library one"
    assert {p.id for p in first} == {old, new}


def test_pending_spreads_across_years(conn: sqlite3.Connection, cfg: Config) -> None:
    """Partial embedding must leave a representative pool, not one recent slice.

    Strict newest-first would embed only 2026 here, so a query about anything
    else would be matched against a pool containing none of it.
    """
    from advisor.embed import encoder

    for year in ("2010", "2018", "2026"):
        for i in range(5):
            upsert_paper(
                conn,
                Paper(title=f"{year}-{i}", authors=[f"A{i}"],
                      arxiv_id=f"{year}.{i}", published_at=f"{year}-06-01"),
            )

    batch = next(encoder.pending(conn, batch_size=6))
    years = {(p.published_at or "")[:4] for p in batch}

    assert years == {"2010", "2018", "2026"}, "each year should be represented early"


def test_pending_skips_withdrawn_papers(conn: sqlite3.Connection, cfg: Config) -> None:
    from advisor.embed import encoder

    live = upsert_paper(conn, Paper(title="Live", authors=["A"], arxiv_id="a.1"))
    gone = upsert_paper(conn, Paper(title="Gone", authors=["B"], arxiv_id="a.2"))
    conn.execute("UPDATE papers SET withdrawn_at = ? WHERE id = ?", (now(), gone))

    batch = next(encoder.pending(conn, batch_size=10))

    assert [p.id for p in batch] == [live]
    assert encoder.count_pending(conn) == 1


# ------------------------------------------- combining library and explicit ratings


def test_one_rating_does_not_erase_the_library(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """Regression guard.

    Ratings used to *replace* the library, so the first thumbs-up left the
    advisor recommending from that single paper — in testing, one stray click
    on an ML paper turned a post-quantum reading history into an ML feed.
    """
    vectors = {f"crypto{i}": [1.0, 0.05 * i] for i in range(8)}
    vectors["ml"] = [0.0, 1.0]
    ids, matrix = build_matrix(conn, cfg, vectors)

    for name in [f"crypto{i}" for i in range(8)]:
        add_to_library(conn, ids[name])
    rate(conn, ids["ml"], 1)

    queries = retrieve.preference_vectors(conn, matrix, cfg)
    crypto_dir = np.array([1.0, 0.0], dtype=np.float32)

    assert queries.shape[0] > 1, "the library should still contribute interests"
    assert max(q @ crypto_dir for q in queries) > 0.8, "library interests must survive"


def test_a_liked_paper_outweighs_a_merely_read_one(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """An explicit thumbs-up is a stronger signal than 'I read this'."""
    vectors = {f"a{i}": [1.0, 0.0] for i in range(3)}
    vectors["b"] = [0.0, 1.0]
    ids, matrix = build_matrix(conn, cfg, vectors)

    for i in range(3):
        add_to_library(conn, ids[f"a{i}"])
    add_to_library(conn, ids["b"])

    one = Config(data_dir=cfg.data_dir, n_clusters=1)
    before = retrieve.preference_vectors(conn, matrix, one)[0].copy()
    rate(conn, ids["b"], 1)
    after = retrieve.preference_vectors(conn, matrix, one)[0]

    b_dir = np.array([0.0, 1.0], dtype=np.float32)
    assert after @ b_dir > before @ b_dir


def test_disliking_a_read_paper_removes_it_from_interests(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """Having read something is not an endorsement of it."""
    ids, matrix = build_matrix(conn, cfg, {"keep": [1.0, 0.0], "reject": [0.0, 1.0]})
    add_to_library(conn, ids["keep"])
    add_to_library(conn, ids["reject"])
    rate(conn, ids["reject"], -1)

    one = Config(data_dir=cfg.data_dir, n_clusters=1)
    query = retrieve.preference_vectors(conn, matrix, one)[0]

    assert query @ np.array([1.0, 0.0], dtype=np.float32) > 0.9


def test_one_dense_interest_cannot_take_every_slot(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """Cosines are not comparable across clusters.

    A dense interest scores systematically higher than a sparse one, so a global
    sort would hand it every slot and undo the clustering. Round-robin keeps
    each interest represented.
    """
    vectors = {f"dense{i}": [1.0, 0.02 * i] for i in range(20)}
    vectors.update({f"sparse{i}": [0.35, 0.94 + 0.01 * i] for i in range(4)})
    ids, matrix = build_matrix(conn, cfg, vectors)

    queries = store.normalize(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    )
    results = retrieve.search(conn, matrix, queries, cfg, exclude=set())

    top6 = {c.paper_id for c in results[:6]}
    sparse_ids = {ids[f"sparse{i}"] for i in range(4)}
    assert top6 & sparse_ids, "the sparse interest must still appear near the top"


# ------------------------------------------------------------------- attribution


def test_explain_names_the_nearest_paper_of_yours(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """The free rationale: which paper you read pulled this one in.

    Two interests, far apart. Each recommendation must be attributed to the one
    it actually sits near, not to whichever source happens to come first.
    """
    ids, _ = build_matrix(
        conn,
        cfg,
        {
            "lattices": [1.0, 0.0],
            "mpc": [0.0, 1.0],
            "near lattices": [0.99, 0.14],
            "near mpc": [0.14, 0.99],
        },
    )
    add_to_library(conn, ids["lattices"])
    add_to_library(conn, ids["mpc"])

    candidates = [
        retrieve.Candidate(ids["near lattices"], 0.9),
        retrieve.Candidate(ids["near mpc"], 0.9),
    ]
    found = retrieve.explain(conn, candidates, cfg)

    assert found[ids["near lattices"]].paper_id == ids["lattices"]
    assert found[ids["near mpc"]].paper_id == ids["mpc"]
    assert found[ids["near lattices"]].similarity > 0.9


def test_explain_distinguishes_read_from_liked(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """Saying "which you read" about a paper you only rated up would be a lie."""
    ids, _ = build_matrix(
        conn, cfg, {"rated": [1.0, 0.0], "in library": [0.0, 1.0],
                    "near rated": [0.99, 0.14]}
    )
    add_to_library(conn, ids["in library"])
    rate(conn, ids["rated"], 1)

    found = retrieve.explain(conn, [retrieve.Candidate(ids["near rated"], 0.9)], cfg)
    source = found[ids["near rated"]]

    assert source.paper_id == ids["rated"]
    assert source.kind == "liked"
    assert "which you liked" in source.sentence()
    assert "rated" in source.sentence()


def test_explain_ignores_papers_you_disliked(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """A rejected paper must not be offered as the reason for a new one."""
    ids, _ = build_matrix(
        conn, cfg, {"rejected": [1.0, 0.0], "kept": [0.0, 1.0], "near rejected": [0.99, 0.14]}
    )
    add_to_library(conn, ids["rejected"])
    add_to_library(conn, ids["kept"])
    rate(conn, ids["rejected"], -1)

    found = retrieve.explain(conn, [retrieve.Candidate(ids["near rejected"], 0.9)], cfg)

    assert found[ids["near rejected"]].paper_id == ids["kept"]


def test_explain_is_silent_when_it_has_nothing_to_say(
    conn: sqlite3.Connection, cfg: Config
) -> None:
    """No library, or an unembedded candidate: no attribution, no crash."""
    ids, _ = build_matrix(conn, cfg, {"a": [1.0, 0.0], "b": [0.0, 1.0]})

    assert retrieve.explain(conn, [retrieve.Candidate(ids["b"], 0.9)], cfg) == {}

    add_to_library(conn, ids["a"])
    unembedded = upsert_paper(
        conn, Paper(title="never encoded", authors=["X"], arxiv_id="zz.1")
    )
    found = retrieve.explain(conn, [retrieve.Candidate(unembedded, 0.9)], cfg)

    assert found == {}


def test_a_long_source_title_is_elided(conn: sqlite3.Connection, cfg: Config) -> None:
    """The rationale sits in a feed card; a 300-character title would wreck it."""
    source = retrieve.Attribution(
        paper_id=1, title="Word " * 60, similarity=0.7123, kind="read"
    )
    sentence = source.sentence()

    assert len(sentence) < 140
    assert sentence.endswith("(cosine 0.71).")
    assert "…" in sentence
