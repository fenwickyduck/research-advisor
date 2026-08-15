"""Following people rather than topics."""

from __future__ import annotations

import sqlite3

import pytest

from advisor import authors
from advisor.models import Paper, now, upsert_paper


def add(conn: sqlite3.Connection, title: str, names: list[str], year: str = "2026") -> int:
    return upsert_paper(
        conn,
        Paper(title=title, authors=names, arxiv_id=f"x.{abs(hash(title)) % 99999}",
              published_at=f"{year}-01-01"),
    )


# ------------------------------------------------------------------------ matching


@pytest.mark.parametrize(
    "followed,candidate,expected",
    [
        # The same person, written the two ways the two archives write them.
        ("Henry Corrigan-Gibbs", "H. Corrigan-Gibbs", True),
        ("H. Corrigan-Gibbs", "Henry Corrigan-Gibbs", True),
        ("Henry Corrigan-Gibbs", "Henry Corrigan-Gibbs", True),
        # Accents differ between sources for the same author.
        ("Erik Mårtensson", "Erik Martensson", True),
        # Different people who share a surname and an initial. The first
        # implementation keyed on 'rao|a' and conflated all of these.
        ("Ada Rao", "Alan Rao", False),
        ("Ada Rao", "Amara Rao", False),
        ("Ada Rao", "A. Rao", True),
        # A surname you typed is a deliberate wildcard.
        ("Corrigan-Gibbs", "Henry Corrigan-Gibbs", True),
        # A surname in the *metadata* is a truncated record, not a wildcard.
        ("Ada Rao", "Lee", False),
        ("Wei Chen", "Wei Chen", True),
        ("Wei Chen", "Wen Chen", False),
    ],
)
def test_name_matching(followed: str, candidate: str, expected: bool) -> None:
    assert authors.matches(authors.key(followed), authors.key(candidate)) is expected


def test_unusable_names_match_nothing() -> None:
    for junk in ("", "   ", ".", "-"):
        assert authors.key(junk) == ""
        assert not authors.matches(authors.key(junk), authors.key("Wei Chen"))


# ------------------------------------------------------------------------ following


def test_following_is_idempotent(conn: sqlite3.Connection) -> None:
    assert authors.follow(conn, "Wei Chen")
    assert not authors.follow(conn, "Wei Chen")
    # The other spelling is the same person, so it must not create a second row.
    assert not authors.follow(conn, "W. Chen")
    assert len(authors.following(conn)) == 1


def test_unfollowing(conn: sqlite3.Connection) -> None:
    authors.follow(conn, "Wei Chen")
    assert authors.unfollow(conn, "Wei Chen")
    assert not authors.unfollow(conn, "Wei Chen")
    assert authors.following(conn) == []


def test_papers_by_finds_the_right_person(conn: sqlite3.Connection) -> None:
    """The headline behaviour, and the collision it must not make."""
    wanted = add(conn, "Function Secret Sharing", ["Wei Chen", "Niv Gilboa"])
    abbreviated = add(conn, "Topology-Hiding Computation", ["W. Chen"])
    other = add(conn, "Something Else Entirely", ["Wen Chen"])
    unrelated = add(conn, "Unrelated Work", ["Someone Else"])

    authors.follow(conn, "Wei Chen")
    found = {paper_id for paper_id, _ in authors.papers_by(conn)}

    assert found == {wanted, abbreviated}
    assert other not in found and unrelated not in found


def test_papers_by_respects_exclusions_and_limit(conn: sqlite3.Connection) -> None:
    ids = [add(conn, f"Paper {i}", ["Wei Chen"]) for i in range(5)]
    authors.follow(conn, "Wei Chen")

    kept = authors.papers_by(conn, exclude={ids[0], ids[1]})
    assert {pid for pid, _ in kept} == set(ids[2:])
    assert len(authors.papers_by(conn, limit=2)) == 2


def test_withdrawn_papers_are_not_offered(conn: sqlite3.Connection) -> None:
    paper_id = add(conn, "Retracted", ["Wei Chen"])
    conn.execute("UPDATE papers SET withdrawn_at = ? WHERE id = ?", (now(), paper_id))
    authors.follow(conn, "Wei Chen")

    assert authors.papers_by(conn) == []


def test_following_nobody_costs_nothing(conn: sqlite3.Connection) -> None:
    add(conn, "A Paper", ["Wei Chen"])
    assert authors.papers_by(conn) == []


# ---------------------------------------------------------------------- suggestions


def test_suggestions_come_from_repeated_authors(conn: sqlite3.Connection) -> None:
    for i in range(3):
        paper_id = add(conn, f"Boyle {i}", ["Wei Chen", f"Other {i}"])
        conn.execute(
            "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
            (paper_id, now()),
        )

    ranked = authors.suggestions(conn)

    assert ranked[0] == ("Wei Chen", 3)
    # Co-authors seen once are not suggestions.
    assert all(name == "Wei Chen" for name, _ in ranked)


def test_someone_you_already_follow_is_not_suggested(conn: sqlite3.Connection) -> None:
    for i in range(2):
        paper_id = add(conn, f"P{i}", ["Wei Chen"])
        conn.execute(
            "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
            (paper_id, now()),
        )
    authors.follow(conn, "W. Chen")

    assert authors.suggestions(conn) == []


# --------------------------------------------------------- follows in a batch


def embedded_corpus(conn: sqlite3.Connection, cfg, n: int = 12) -> list[int]:
    """A small embedded corpus plus one library paper to retrieve against."""
    import numpy as np

    from advisor.embed import store

    ids = [add(conn, f"Paper {i}", [f"Author {i}"]) for i in range(n)]
    matrix = store.normalize(
        np.array([[1.0, 0.02 * i] for i in range(n)], dtype=np.float32)
    )
    store.record_rows(conn, ids, store.append(cfg.vectors_path, matrix), cfg.embedding_model)
    conn.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (ids[0], now()),
    )
    return ids


def test_a_batch_never_exceeds_the_limit_it_was_asked_for(tmp_path, conn) -> None:
    """Follows are merged into the batch, not stacked on top of a full one."""
    from advisor.config import Config
    from advisor.recommend import retrieve

    cfg = Config(data_dir=tmp_path, n_clusters=2, n_followed=3)
    ids = embedded_corpus(conn, cfg)
    # Make three of the corpus papers by someone followed.
    for paper_id in ids[5:8]:
        conn.execute(
            'UPDATE papers SET authors = ? WHERE id = ?', ('["Wei Chen"]', paper_id)
        )
    authors.follow(conn, "Wei Chen")

    assert len(retrieve.recommend(conn, cfg, limit=6)) == 6
    assert len(retrieve.recommend(conn, cfg, limit=2)) == 2


def test_followed_picks_lead_the_batch_and_are_labelled(tmp_path, conn) -> None:
    from advisor.config import Config
    from advisor.recommend import retrieve

    cfg = Config(data_dir=tmp_path, n_clusters=2, n_followed=2)
    ids = embedded_corpus(conn, cfg)
    for paper_id in ids[9:11]:
        conn.execute(
            'UPDATE papers SET authors = ? WHERE id = ?', ('["Wei Chen"]', paper_id)
        )
    authors.follow(conn, "Wei Chen")

    batch = retrieve.recommend(conn, cfg, limit=6)

    assert [c.via for c in batch[:2]] == ["Wei Chen", "Wei Chen"]
    assert all(c.via is None for c in batch[2:]), "similarity picks carry no author"
