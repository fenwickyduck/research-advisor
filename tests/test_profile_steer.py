"""The profile as an instruction, not just a description.

The steering sections are the only part of the advisor you can drive with
words rather than clicks, and the only part that works on a plain text edit —
so the assertions here are about a typed phrase actually moving retrieval.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from advisor.config import Config
from advisor.embed import encoder, store
from advisor.models import Paper, now, upsert_paper
from advisor.recommend import profile, retrieve


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(data_dir=tmp_path, n_clusters=2, n_retrieve_per_cluster=50)


# Two axes: "pir" points one way, "blockchain" the other.
DIRECTIONS = {"pir": [1.0, 0.0], "blockchain": [0.0, 1.0]}


def fake_phrase_encoder(monkeypatch) -> None:
    """Encode a phrase by which keyword it contains, in the same 2-D space."""

    def encode(texts, cfg, progress=None):
        out = []
        for text in texts:
            lowered = text.lower()
            key = next((k for k in DIRECTIONS if k in lowered), None)
            out.append(DIRECTIONS.get(key, [0.5, 0.5]))
        return np.array(out, dtype=np.float32)

    monkeypatch.setattr(encoder, "encode", encode)


def build(conn: sqlite3.Connection, cfg: Config, vectors: dict[str, list[float]]):
    ids = {}
    for title, _ in vectors.items():
        ids[title] = upsert_paper(
            conn, Paper(title=title, authors=[title], arxiv_id=f"id.{len(ids):05d}")
        )
    matrix = store.normalize(np.array(list(vectors.values()), dtype=np.float32))
    store.record_rows(conn, [ids[t] for t in vectors], store.append(cfg.vectors_path, matrix), cfg.embedding_model)
    return ids, store.load(cfg.vectors_path)


def set_profile(conn: sqlite3.Connection, content: str) -> None:
    profile.save(conn, content, written_by="user")


# ------------------------------------------------------------------------ parsing


def test_parse_reads_the_steering_sections() -> None:
    steer = profile.parse(
        """
## Working on
Lots of prose about what they do, which is not an instruction.

## More of
- doubly-efficient PIR
homomorphic encryption bootstrapping

## Less of
* blockchain deployment surveys
"""
    )

    assert steer.more == ["doubly-efficient PIR", "homomorphic encryption bootstrapping"]
    assert steer.less == ["blockchain deployment surveys"]
    assert steer


def test_prose_sections_are_never_mistaken_for_instructions() -> None:
    """An unrecognised heading must switch steering off, not capture what follows."""
    steer = profile.parse(
        """
## More of
lattice signatures

## Background
They already know RLWE hardness.
Introductions are a waste of their time.
"""
    )

    assert steer.more == ["lattice signatures"]
    assert steer.less == []


def test_a_profile_written_before_this_existed_steers_nothing() -> None:
    """Old prose profiles must keep working, unchanged."""
    steer = profile.parse("## Working on\nPIR.\n\n## Not interested in\nBlockchain.")

    assert not steer
    assert profile.parse(None) == profile.Steer([], [])


def test_the_item_cap_holds() -> None:
    steer = profile.parse("## More of\n" + "\n".join(f"topic {i}" for i in range(50)))

    assert len(steer.more) == profile.MAX_STEER_ITEMS


# ------------------------------------------------------------------------ vectors


def test_the_cache_is_keyed_on_the_text_and_the_model(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    """Re-encoding on every run would load the model every time you asked."""
    fake_phrase_encoder(monkeypatch)
    set_profile(conn, "## More of\npir")

    calls = {"n": 0}
    inner = encoder.encode

    def counting(texts, cfg, progress=None):
        calls["n"] += 1
        return inner(texts, cfg, progress)

    monkeypatch.setattr(encoder, "encode", counting)

    profile.steer_vectors(conn, cfg)
    profile.steer_vectors(conn, cfg)
    assert calls["n"] == 1, "the second call must come from cache"

    # Editing the profile invalidates it.
    set_profile(conn, "## More of\nblockchain")
    profile.steer_vectors(conn, cfg)
    assert calls["n"] == 2

    # So does changing the embedding model, whose vectors are not comparable.
    profile.steer_vectors(conn, Config(data_dir=cfg.data_dir, embedding_model="other"))
    assert calls["n"] == 3


# ---------------------------------------------------------------------- retrieval


def test_more_of_pulls_retrieval_toward_a_topic_you_named(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    """The headline behaviour: typing a phrase changes what you are shown."""
    fake_phrase_encoder(monkeypatch)
    ids, matrix = build(
        conn,
        cfg,
        {"read this": [0.0, 1.0], "pir paper": [1.0, 0.0], "chain paper": [0.0, 1.0]},
    )
    conn.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (ids["read this"], now()),
    )

    before = retrieve.preference_vectors(conn, matrix, cfg)
    set_profile(conn, "## More of\nprivate information retrieval (pir)")
    after = retrieve.preference_vectors(conn, matrix, cfg)

    pir = np.array([1.0, 0.0], dtype=np.float32)
    assert max(v @ pir for v in after) > max(v @ pir for v in before) + 0.3


def test_less_of_pushes_retrieval_away(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    fake_phrase_encoder(monkeypatch)
    ids, matrix = build(
        conn, cfg, {"read this": [0.7, 0.7], "pir paper": [1.0, 0.0]}
    )
    conn.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (ids["read this"], now()),
    )

    one = Config(data_dir=cfg.data_dir, n_clusters=1)
    before = retrieve.preference_vectors(conn, matrix, one)[0].copy()
    set_profile(conn, "## Less of\nblockchain deployment surveys")
    after = retrieve.preference_vectors(conn, matrix, one)[0]

    chain = np.array([0.0, 1.0], dtype=np.float32)
    assert after @ chain < before @ chain


def test_a_profile_alone_is_enough_to_get_recommendations(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    """With an empty library, saying what you want should still work."""
    fake_phrase_encoder(monkeypatch)
    _, matrix = build(conn, cfg, {"pir paper": [1.0, 0.0], "chain paper": [0.0, 1.0]})

    assert retrieve.preference_vectors(conn, matrix, cfg).shape[0] == 0

    set_profile(conn, "## More of\npir")
    queries = retrieve.preference_vectors(conn, matrix, cfg)

    assert queries.shape[0] > 0
    assert queries[0] @ np.array([1.0, 0.0], dtype=np.float32) > 0.9


def test_an_unusable_profile_never_breaks_a_run(
    conn: sqlite3.Connection, cfg: Config, monkeypatch
) -> None:
    """A missing encoder or stale vectors must cost the steer, not the run."""
    ids, matrix = build(conn, cfg, {"read this": [1.0, 0.0], "other": [0.0, 1.0]})
    conn.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (ids["read this"], now()),
    )
    set_profile(conn, "## More of\npir")

    def exploding(texts, cfg, progress=None):
        raise RuntimeError("no encoder installed")

    monkeypatch.setattr(encoder, "encode", exploding)
    assert retrieve.preference_vectors(conn, matrix, cfg).shape[0] > 0

    # Vectors of the wrong width (a changed embedding model) are ignored too.
    monkeypatch.setattr(
        encoder, "encode", lambda texts, cfg, progress=None: np.ones((len(texts), 7), dtype=np.float32)
    )
    assert retrieve.preference_vectors(conn, matrix, cfg).shape[0] > 0


# ---------------------------------------------------------------------- the page


def web_client(tmp_path):
    from fastapi.testclient import TestClient

    from advisor import db
    from advisor.web import app as web

    cfg = Config(data_dir=tmp_path, n_clusters=2)
    conn = db.connect(cfg.db_path)
    web.CFG = cfg
    return TestClient(web.app), conn, cfg


def test_the_page_offers_an_editor_with_no_profile_and_no_key(tmp_path) -> None:
    """Writing your own must not be hidden behind having a library or a key."""
    client, conn, _ = web_client(tmp_path)
    conn.close()

    html = client.get("/profile").text

    assert "## More of" in html and "## Less of" in html
    assert "<textarea" in html


def test_the_page_shows_what_is_steering(tmp_path) -> None:
    client, conn, _ = web_client(tmp_path)
    profile.save(conn, "## More of\ndoubly-efficient PIR\n\n## Less of\nblockchain", "user")
    conn.commit()
    conn.close()

    html = client.get("/profile").text

    assert "Steering your next batch" in html
    assert "doubly-efficient PIR" in html


def test_a_prose_only_profile_says_it_steers_nothing(tmp_path) -> None:
    """Otherwise a profile that does nothing looks identical to one that works."""
    client, conn, _ = web_client(tmp_path)
    profile.save(conn, "## Working on\nPIR and homomorphic encryption.", "user")
    conn.commit()
    conn.close()

    html = client.get("/profile").text

    assert "Not steering anything yet" in html


def test_saving_a_hand_written_profile_takes_effect(tmp_path) -> None:
    client, conn, _ = web_client(tmp_path)
    conn.close()

    client.post("/profile/edit", data={"content": "## More of\nlattice PIR"})

    html = client.get("/profile").text
    assert "lattice PIR" in html
    assert "Steering your next batch" in html
