"""Sharing the corpus and its vectors, so nobody re-does three hours of CPU."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from advisor import snapshot
from advisor.config import Config
from advisor.embed import store
from advisor.models import Paper, now, upsert_paper


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(data_dir=tmp_path / "source")


@pytest.fixture
def target(tmp_path):
    from advisor import db

    other = Config(data_dir=tmp_path / "target")
    conn = db.connect(other.db_path)
    yield conn, other
    conn.close()


def build(conn: sqlite3.Connection, cfg: Config, n: int = 6) -> list[int]:
    ids = [
        upsert_paper(
            conn,
            Paper(title=f"Paper {i}", abstract=f"About topic {i}.",
                  authors=[f"Author {i}"], arxiv_id=f"s.{i}",
                  published_at=f"202{i % 6}-01-01"),
        )
        for i in range(n)
    ]
    matrix = store.normalize(np.array([[1.0, 0.1 * i] for i in range(n)], dtype=np.float32))
    store.record_rows(conn, ids, store.append(cfg.vectors_path, matrix), cfg.embedding_model)
    conn.commit()
    return ids


# ------------------------------------------------------------------------ saving


def test_a_snapshot_carries_nothing_personal(
    conn: sqlite3.Connection, cfg: Config, tmp_path
) -> None:
    """The whole point of the split: this file is shareable, yours is not."""
    ids = build(conn, cfg)
    conn.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (ids[0], now()),
    )
    conn.execute(
        "INSERT INTO feedback (paper_id, rating, tags, note, created_at) VALUES (?,?,?,?,?)",
        (ids[1], -1, "[]", "a private note", now()),
    )
    conn.execute(
        "INSERT INTO profile_versions (content, written_by, created_at) VALUES (?,?,?)",
        ("## More of\nsecret interest", "user", now()),
    )
    conn.commit()

    path = tmp_path / "corpus.tar"
    snapshot.save(conn, cfg, path)

    blob = path.read_bytes()
    assert b"a private note" not in blob
    assert b"secret interest" not in blob


def test_saving_needs_vectors(conn: sqlite3.Connection, cfg: Config, tmp_path) -> None:
    with pytest.raises(ValueError, match="no vectors"):
        snapshot.save(conn, cfg, tmp_path / "corpus.tar")


def test_metadata_can_be_read_without_unpacking(
    conn: sqlite3.Connection, cfg: Config, tmp_path
) -> None:
    build(conn, cfg)
    path = tmp_path / "corpus.tar"
    snapshot.save(conn, cfg, path)

    meta = snapshot.inspect(path)

    assert meta["papers"] == 6
    assert meta["dtype"] == "float16"
    assert meta["embedding_model"] == cfg.embedding_model


# ----------------------------------------------------------------------- loading


def test_it_restores_into_an_empty_install(
    conn: sqlite3.Connection, cfg: Config, target, tmp_path
) -> None:
    """The headline case: skip the harvest and the three-hour encode."""
    build(conn, cfg)
    path = tmp_path / "corpus.tar"
    snapshot.save(conn, cfg, path)

    other, other_cfg = target
    snapshot.load(other, other_cfg, path)

    assert other.execute("SELECT count(*) FROM papers").fetchone()[0] == 6
    assert other.execute("SELECT count(*) FROM vector_index").fetchone()[0] == 6
    assert store.load(other_cfg.vectors_path).shape == (6, 2)


def test_restored_vectors_retrieve_the_same_neighbours(
    conn: sqlite3.Connection, cfg: Config, target, tmp_path
) -> None:
    """float16 is only acceptable if it changes nothing about the ranking."""
    build(conn, cfg, n=40)
    path = tmp_path / "corpus.tar"
    snapshot.save(conn, cfg, path)
    other, other_cfg = target
    snapshot.load(other, other_cfg, path)

    query = store.normalize(np.array([1.0, 0.35], dtype=np.float32))
    before = np.argsort(-(np.asarray(store.load(cfg.vectors_path)) @ query))[:10]
    after = np.argsort(-(np.asarray(store.load(other_cfg.vectors_path)) @ query))[:10]

    assert list(before) == list(after)


def test_a_snapshot_from_another_model_is_refused(
    conn: sqlite3.Connection, cfg: Config, target, tmp_path
) -> None:
    """Vectors from two models are not comparable; mixing them is silent."""
    build(conn, cfg)
    path = tmp_path / "corpus.tar"
    snapshot.save(conn, cfg, path)

    other, other_cfg = target
    mismatched = Config(data_dir=other_cfg.data_dir, embedding_model="something/else")

    with pytest.raises(ValueError, match="not comparable"):
        snapshot.load(other, mismatched, path)


def test_loading_over_existing_vectors_needs_replace(
    conn: sqlite3.Connection, cfg: Config, target, tmp_path
) -> None:
    build(conn, cfg)
    path = tmp_path / "corpus.tar"
    snapshot.save(conn, cfg, path)

    other, other_cfg = target
    build(other, other_cfg, n=3)

    with pytest.raises(ValueError, match="--replace"):
        snapshot.load(other, other_cfg, path)

    snapshot.load(other, other_cfg, path, replace=True)
    assert other.execute("SELECT count(*) FROM vector_index").fetchone()[0] == 6


def test_duplicates_are_reported_not_silently_dropped(
    conn: sqlite3.Connection, cfg: Config, target, tmp_path
) -> None:
    """The corpus holds a few papers posted twice; the count must add up."""
    build(conn, cfg, n=3)
    # A duplicate that escaped deduplication when it was harvested — inserted
    # raw, because upsert_paper would (correctly) refuse to create it.
    conn.execute(
        """INSERT INTO papers (title, title_norm, abstract, authors, categories,
                               eprint_id)
           VALUES ('Paper 0', 'zzz stale key', 'About topic 0.',
                   '["Author 0"]', '[]', '2025/0001')"""
    )
    twin = conn.execute("SELECT id FROM papers WHERE eprint_id='2025/0001'").fetchone()[0]
    matrix = store.normalize(np.array([[1.0, 0.0]], dtype=np.float32))
    store.record_rows(conn, [twin], store.append(cfg.vectors_path, matrix), cfg.embedding_model)
    conn.commit()

    path = tmp_path / "corpus.tar"
    snapshot.save(conn, cfg, path)
    other, other_cfg = target
    meta = snapshot.load(other, other_cfg, path)

    assert meta["merged"] == 1
    assert meta["restored"] == 3


def test_junk_is_refused(target, tmp_path) -> None:
    other, other_cfg = target
    path = tmp_path / "junk.tar"

    import tarfile, io
    with tarfile.open(path, "w") as tar:
        info = tarfile.TarInfo("meta.json")
        blob = b'{"format": "something/else"}'
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))

    with pytest.raises(ValueError, match="not a corpus snapshot"):
        snapshot.load(other, other_cfg, path)


# ---------------------------------------------------------------------- fetching


def test_fetch_prefers_an_anonymous_download(tmp_path, monkeypatch) -> None:
    """A public repository needs no credentials and no extra tooling."""
    monkeypatch.setattr(
        snapshot, "_public_asset", lambda repo: ("corpus-2026-01-01.tar", "https://x/y")
    )

    def pretend_stream(url, target, progress):
        target.write_bytes(b"x")

    monkeypatch.setattr(snapshot, "_stream", pretend_stream)

    assert snapshot.fetch(tmp_path).name == "corpus-2026-01-01.tar"


def test_fetch_falls_back_to_gh_for_a_private_repository(tmp_path, monkeypatch) -> None:
    import shutil
    import subprocess

    monkeypatch.setattr(snapshot, "_public_asset", lambda repo: None)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/gh")

    def pretend(*args, **kwargs):
        (tmp_path / "corpus-2026-02-02.tar").write_bytes(b"x")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", pretend)

    assert snapshot.fetch(tmp_path).name == "corpus-2026-02-02.tar"


def test_fetch_explains_itself_when_neither_route_works(tmp_path, monkeypatch) -> None:
    """The likeliest confusion: a private repo and no GitHub CLI."""
    import shutil

    monkeypatch.setattr(snapshot, "_public_asset", lambda repo: None)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(ValueError, match="GitHub CLI"):
        snapshot.fetch(tmp_path)


def test_a_broken_network_does_not_raise_from_the_lookup(monkeypatch) -> None:
    """An offline machine should fall through to gh, not crash."""
    import httpx

    def boom(*args, **kwargs):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(httpx, "get", boom)

    assert snapshot._public_asset("owner/repo") is None


def test_fetch_surfaces_the_gh_error(tmp_path, monkeypatch) -> None:
    import shutil
    import subprocess

    monkeypatch.setattr(snapshot, "_public_asset", lambda repo: None)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "gh: not authenticated"),
    )

    with pytest.raises(ValueError, match="not authenticated"):
        snapshot.fetch(tmp_path)
