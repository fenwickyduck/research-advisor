"""Cross-source dedupe: the same paper on arXiv and ePrint must become one row."""

from __future__ import annotations

import sqlite3

from advisor.models import Paper, normalize_title, upsert_paper


def test_normalize_title_folds_formatting_noise() -> None:
    variants = [
        "Fully Homomorphic Encryption over the Integers",
        "fully homomorphic encryption over the integers",
        "Fully  Homomorphic   Encryption over the Integers.",
        "Fully-Homomorphic Encryption over the Integers",
    ]
    assert len({normalize_title(v) for v in variants}) == 1


def test_normalize_title_strips_accents() -> None:
    assert normalize_title("Mårtensson") == normalize_title("Martensson")


def test_same_paper_on_both_sources_merges(conn: sqlite3.Connection) -> None:
    arxiv_version = Paper(
        title="Memory Checking Requires Logarithmic Overhead",
        abstract="We prove a lower bound.",
        authors=["Wei Chen", "Ilan Komargodski", "Neekon Vafa"],
        arxiv_id="2309.01900",
        categories=["cs.CR"],
        published_at="2023-09-05",
        url="https://arxiv.org/abs/2309.01900",
    )
    # Same work, different capitalisation and punctuation, different source.
    eprint_version = Paper(
        title="Memory checking requires logarithmic overhead.",
        authors=["W. Chen", "Ilan Komargodski", "Neekon Vafa"],
        eprint_id="2023/01345",
        categories=["Foundations"],
        venue="Cryptology ePrint Archive",
        pdf_url="https://eprint.iacr.org/2023/1345.pdf",
    )

    first = upsert_paper(conn, arxiv_version)
    second = upsert_paper(conn, eprint_version)

    assert first == second, "the two postings should collapse into one row"
    assert conn.execute("SELECT count(*) FROM papers").fetchone()[0] == 1

    row = conn.execute("SELECT * FROM papers WHERE id = ?", (first,)).fetchone()
    merged = Paper.from_row(row)
    # Both identifiers survive, so either paste form finds it later.
    assert merged.arxiv_id == "2309.01900"
    assert merged.eprint_id == "2023/01345"
    # Categories from both sources are unioned; the abstract is not lost.
    assert set(merged.categories) == {"cs.CR", "Foundations"}
    assert merged.abstract == "We prove a lower bound."
    # A NULL column is filled in from the later sighting.
    assert merged.pdf_url == "https://eprint.iacr.org/2023/1345.pdf"


def test_identical_title_without_shared_authors_stays_separate(
    conn: sqlite3.Connection,
) -> None:
    """Generic titles recur across unrelated work; require an author in common."""
    a = Paper(title="On Lattice Problems", authors=["Alice Smith"], arxiv_id="2401.00001")
    b = Paper(title="On Lattice Problems", authors=["Bob Jones"], eprint_id="2024/0002")

    assert upsert_paper(conn, a) != upsert_paper(conn, b)
    assert conn.execute("SELECT count(*) FROM papers").fetchone()[0] == 2


def test_reharvest_is_idempotent(conn: sqlite3.Connection) -> None:
    paper = Paper(title="A Paper", authors=["Ann Author"], arxiv_id="2401.00009")
    ids = {upsert_paper(conn, paper) for _ in range(3)}

    assert len(ids) == 1
    assert conn.execute("SELECT count(*) FROM papers").fetchone()[0] == 1


def test_existing_values_are_not_overwritten(conn: sqlite3.Connection) -> None:
    """A later sighting fills gaps; it does not clobber what we already have."""
    paper_id = upsert_paper(
        conn,
        Paper(title="A Paper", abstract="Full abstract.", authors=["Ann Author"],
              arxiv_id="2401.00010"),
    )
    upsert_paper(
        conn,
        Paper(title="A Paper", abstract="Truncated…", authors=["Ann Author"],
              eprint_id="2024/0010"),
    )

    row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
    assert Paper.from_row(row).abstract == "Full abstract."


# ------------------------------------------- merge policy: revision vs cross-source


def test_same_source_revision_replaces_content(conn: sqlite3.Connection) -> None:
    """A revised abstract must win — it is what gets embedded."""
    paper_id = upsert_paper(
        conn,
        Paper(title="A Paper", abstract="First version.", authors=["Ann Author"],
              arxiv_id="2401.00001", updated_at="2024-01-05"),
    )
    upsert_paper(
        conn,
        Paper(title="A Paper, Revised", abstract="Substantially rewritten.",
              authors=["Ann Author"], arxiv_id="2401.00001", updated_at="2024-06-01"),
    )

    row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
    merged = Paper.from_row(row)
    assert merged.abstract == "Substantially rewritten."
    assert merged.title == "A Paper, Revised"
    assert merged.updated_at == "2024-06-01"


def test_cross_source_merge_does_not_clobber_a_fuller_record(
    conn: sqlite3.Connection,
) -> None:
    """A terser posting on another repository must not overwrite a full abstract."""
    paper_id = upsert_paper(
        conn,
        Paper(title="A Paper", abstract="The full abstract, all of it.",
              authors=["Ann Author"], arxiv_id="2401.00002"),
    )
    upsert_paper(
        conn,
        Paper(title="A Paper", abstract="Truncated…", authors=["Ann Author"],
              eprint_id="2024/0002"),
    )

    row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
    assert Paper.from_row(row).abstract == "The full abstract, all of it."


def test_revision_keeps_the_other_sources_categories(conn: sqlite3.Connection) -> None:
    """Regression guard.

    Papers on both arXiv and ePrint carry both taxonomies. If a same-source
    revision replaced the category list wholesale, every one of those papers
    would silently lose its IACR categories on the next arXiv revision.
    """
    paper_id = upsert_paper(
        conn,
        Paper(title="Coin Tracing", authors=["Ann Author"], arxiv_id="2608.09249",
              categories=["cs.CR", "cs.DC"]),
    )
    upsert_paper(
        conn,
        Paper(title="Coin Tracing", authors=["Ann Author"], eprint_id="2026/1645",
              categories=["Applications"]),
    )
    # arXiv now posts a revision, carrying only arXiv's own categories.
    upsert_paper(
        conn,
        Paper(title="Coin Tracing", authors=["Ann Author"], arxiv_id="2608.09249",
              categories=["cs.CR", "cs.DC"], abstract="v2 text"),
    )

    row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
    merged = Paper.from_row(row)
    assert set(merged.categories) == {"cs.CR", "cs.DC", "Applications"}
    assert merged.abstract == "v2 text", "the revision still updates content"
    assert conn.execute("SELECT count(*) FROM papers").fetchone()[0] == 1
