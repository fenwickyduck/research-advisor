"""The profile is Markdown that a person or an assistant wrote.

It is therefore untrusted text being put back into a page, which is the shape
of the stored-XSS that already bit the search snippet once. These pin both
halves: that the markup people write comes out as markup, and that markup they
did not write cannot.
"""

from __future__ import annotations

import pytest

from advisor.web.render import richtext


def test_headings_start_below_the_page_title() -> None:
    """The page owns the <h1>; the document inside it cannot claim one too."""
    assert str(richtext("# Interest profile")) == "<h2>Interest profile</h2>"
    assert str(richtext("## Working on")) == "<h3>Working on</h3>"


def test_emphasis_and_code_survive() -> None:
    out = str(richtext("Both on **doubly-efficient** *PIR*, see `advisor embed`."))
    assert "<strong>doubly-efficient</strong>" in out
    assert "<em>PIR</em>" in out
    assert "<code>advisor embed</code>" in out


def test_steering_lines_each_become_their_own_paragraph() -> None:
    """They are separate search directions, not one run-on sentence."""
    out = str(richtext("## More of\nhardness of module learning with errors\nlattice sieving"))
    assert out.count("<p>") == 2


def test_bullets_become_a_list() -> None:
    assert "<ul><li>one</li><li>two</li></ul>" in str(richtext("- one\n- two"))


@pytest.mark.parametrize(
    "hostile",
    [
        "<script>alert(1)</script>",
        '<img src=x onerror="alert(1)">',
        "<a href='javascript:alert(1)'>click</a>",
        "**<script>alert(1)</script>**",
        "# <iframe src=//evil></iframe>",
    ],
)
def test_markup_in_the_profile_is_shown_not_run(hostile: str) -> None:
    out = str(richtext(hostile))
    assert "<script" not in out
    assert "<img" not in out and "<iframe" not in out
    assert "onerror" not in out or "&lt;img" in out
    assert "javascript:" not in out or "&lt;a" in out


def test_nothing_in_renders_nothing_out() -> None:
    assert str(richtext(None)) == ""
    assert str(richtext("   ")) == ""
