from __future__ import annotations

import pytest

from advisor.ingest.resolve import Ref, parse, parse_many


@pytest.mark.parametrize(
    "text,expected",
    [
        # arXiv, modern scheme
        ("2401.12345", Ref("arxiv", "2401.12345")),
        ("2401.12345v3", Ref("arxiv", "2401.12345")),
        ("arXiv:2401.12345", Ref("arxiv", "2401.12345")),
        ("https://arxiv.org/abs/1706.03762", Ref("arxiv", "1706.03762")),
        ("https://arxiv.org/pdf/1706.03762v5", Ref("arxiv", "1706.03762")),
        ("http://export.arxiv.org/abs/2301.00001", Ref("arxiv", "2301.00001")),
        # arXiv, pre-2007 scheme
        ("cs.CR/0512345", Ref("arxiv", "cs.CR/0512345")),
        ("https://arxiv.org/abs/hep-th/9901001", Ref("arxiv", "hep-th/9901001")),
        # ePrint — the sequence number is zero-padded to the canonical width
        ("2024/123", Ref("eprint", "2024/0123")),
        ("2024/0123", Ref("eprint", "2024/0123")),
        ("eprint 2024/123", Ref("eprint", "2024/0123")),
        ("https://eprint.iacr.org/2024/123", Ref("eprint", "2024/0123")),
        ("https://eprint.iacr.org/2024/123.pdf", Ref("eprint", "2024/0123")),
        ("https://ia.cr/2026/1688", Ref("eprint", "2026/1688")),
        # DOI
        ("10.1007/3-540-48910-X_16", Ref("doi", "10.1007/3-540-48910-x_16")),
        ("https://doi.org/10.1145/3576915", Ref("doi", "10.1145/3576915")),
        ("doi:10.1145/3576915", Ref("doi", "10.1145/3576915")),
        # Nothing usable
        ("", None),
        ("   ", None),
        ("not a paper", None),
        ("see you at CRYPTO", None),
    ],
)
def test_parse(text: str, expected: Ref | None) -> None:
    assert parse(text) == expected


def test_doi_wins_over_embedded_digits() -> None:
    """A DOI suffix can contain arXiv-shaped digits; the 10.x prefix decides."""
    assert parse("https://doi.org/10.1234/2401.12345") == Ref("doi", "10.1234/2401.12345")


def test_trailing_punctuation_is_stripped_from_doi() -> None:
    assert parse("(10.1145/3576915)") == Ref("doi", "10.1145/3576915")


def test_parse_many_dedupes_and_reports_bad_lines() -> None:
    refs, bad = parse_many(
        """
        2401.12345
        arXiv:2401.12345v2
        2024/123

        garbage
        https://eprint.iacr.org/2024/0123
        """
    )
    # The two arXiv forms collapse to one ref, as do the two ePrint forms.
    assert refs == [Ref("arxiv", "2401.12345"), Ref("eprint", "2024/0123")]
    assert bad == ["garbage"]


def test_doi_case_is_folded() -> None:
    """DOIs are case-insensitive, so the printed and registered forms must agree."""
    upper = parse("10.1007/3-540-48910-X_16")
    lower = parse("https://doi.org/10.1007/3-540-48910-x_16")
    assert upper == lower
