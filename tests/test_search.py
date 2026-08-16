"""Full-text search over the corpus."""

from __future__ import annotations

import sqlite3

import pytest

from advisor import search
from advisor.models import Paper, now, upsert_paper


@pytest.fixture
def corpus(conn: sqlite3.Connection) -> sqlite3.Connection:
    papers = [
        ("Doubly-Efficient Private Information Retrieval",
         "A DEPIR scheme with sublinear server work.", ["Wei Chen"]),
        ("A Survey of Private Information Retrieval",
         "We survey the PIR literature.", ["Someone Else"]),
        ("Lattice-Based Digital Signatures",
         "Signatures from module lattices.", ["Erik Mårtensson"]),
    ]
    for i, (title, abstract, names) in enumerate(papers):
        upsert_paper(
            conn, Paper(title=title, abstract=abstract, authors=names, arxiv_id=f"s.{i}")
        )
    return conn


# ------------------------------------------------------------------ query building


@pytest.mark.parametrize(
    "query,expected",
    [
        ("pir", '"pir"'),
        ("private information", '"private" AND "information"'),
        # Punctuation is inert, not syntax.
        ("lattice-based", '"lattice-based"'),
        # FTS5 has no -term form; it must become NOT, grouped so it applies to
        # the whole query rather than only the last term.
        ("pir -survey", '"pir" NOT ("survey")'),
        ("a b -c -d", '"a" AND "b" NOT ("c" OR "d")'),
        # Nothing positive to match: refuse rather than scan the whole corpus.
        ("-survey", ""),
        ("", ""),
        ("   ", ""),
        ("()*", ""),
    ],
)
def test_sanitise(query: str, expected: str) -> None:
    assert search.sanitise(query) == expected


# ------------------------------------------------------------------------ searching


def test_finds_papers_by_title_and_abstract(corpus: sqlite3.Connection) -> None:
    assert [h.title for h in search.search(corpus, "DEPIR")][0].startswith("Doubly-Efficient")
    assert search.count(corpus, "sublinear") == 1


def test_exclusion_removes_matches(corpus: sqlite3.Connection) -> None:
    assert search.count(corpus, "retrieval") == 2
    assert search.count(corpus, "retrieval -survey") == 1


def test_diacritics_are_folded(corpus: sqlite3.Connection) -> None:
    """The same author is spelled both ways across sources."""
    assert search.count(corpus, "Martensson") == 1
    assert search.count(corpus, "Mårtensson") == 1


def test_title_matches_outrank_abstract_matches(corpus: sqlite3.Connection) -> None:
    hits = search.search(corpus, "survey")

    assert hits[0].title.startswith("A Survey"), "title weighting must win"


def test_a_hostile_query_returns_nothing_rather_than_raising(
    corpus: sqlite3.Connection,
) -> None:
    """A search box must not be able to take down the page that hosts it."""
    for query in ['"', 'a"b)c(d*', "NEAR(a b)", "^title:", "*", "AND OR NOT", "a" * 500]:
        assert isinstance(search.search(corpus, query), list)
        assert isinstance(search.count(corpus, query), int)


def test_withdrawn_papers_are_not_returned(corpus: sqlite3.Connection) -> None:
    corpus.execute("UPDATE papers SET withdrawn_at = ? WHERE title LIKE 'A Survey%'", (now(),))

    assert search.count(corpus, "retrieval") == 1


def test_the_index_tracks_edits_and_deletes(corpus: sqlite3.Connection) -> None:
    """A contentless FTS table cannot repair itself, so the triggers matter."""
    assert search.count(corpus, "sublinear") == 1

    corpus.execute("UPDATE papers SET abstract = 'rewritten entirely' WHERE arxiv_id = 's.0'")
    assert search.count(corpus, "sublinear") == 0
    assert search.count(corpus, "rewritten") == 1

    corpus.execute("DELETE FROM papers WHERE arxiv_id = 's.0'")
    assert search.count(corpus, "rewritten") == 0


def test_reharvesting_a_paper_does_not_duplicate_it(corpus: sqlite3.Connection) -> None:
    """Harvest upserts constantly; the index must not accumulate copies."""
    for _ in range(3):
        upsert_paper(
            corpus,
            Paper(title="Doubly-Efficient Private Information Retrieval",
                  abstract="A DEPIR scheme with sublinear server work.",
                  authors=["Wei Chen"], arxiv_id="s.0"),
        )

    assert search.count(corpus, "DEPIR") == 1


def test_a_paper_cannot_inject_markup_into_the_results(
    corpus: sqlite3.Connection,
) -> None:
    """arXiv abstracts are not HTML and are not sanitised by anyone upstream.

    A hostile submission would otherwise be harvested into every reader's
    corpus and run in the browser of anyone whose search matched it.
    """
    upsert_paper(
        corpus,
        Paper(title="Zerotrust Protocols",
              abstract="We study zerotrust <script>alert(1)</script> systems.",
              authors=["A"], arxiv_id="x.9"),
    )

    hit = search.search(corpus, "zerotrust")[0]

    assert "<script>" not in hit.marked
    assert "&lt;script&gt;" in hit.marked
    assert "<mark>" in hit.marked, "the highlight must survive the escaping"
    # The terminal form carries neither the markers nor markup.
    assert "\x02" not in hit.text and "<mark>" not in hit.text
