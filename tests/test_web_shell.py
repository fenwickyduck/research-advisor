"""The chrome around every page: what is in the top bar, and where.

The nav used to be seven flat links. The point of these is that the top level
stays the reading loop and everything else stays behind one control, because
that is the thing that quietly rots as pages get added.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from advisor import db
from advisor.config import Config
from advisor.web import app as web

TOP_LEVEL = ["/", "/library", "/search"]
TUNING = ["/profile", "/authors", "/data"]


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    cfg = Config(data_dir=tmp_path)
    db.connect(cfg.db_path).close()
    monkeypatch.setattr(web, "CFG", cfg)
    return TestClient(web.app)


def nav_targets(html: str) -> list[str]:
    nav = re.search(r'<nav class="nav".*?</nav>', html, re.S)
    assert nav, "no main nav in the page"
    return re.findall(r'href="([^"]+)"', nav.group(0))


def test_the_top_level_is_only_the_reading_loop(client: TestClient) -> None:
    assert nav_targets(client.get("/").text) == TOP_LEVEL


def test_everything_else_is_reachable_from_the_menu(client: TestClient) -> None:
    html = client.get("/").text
    panel = re.search(r'<div class="menu-panel".*?</div>', html, re.S)
    assert panel
    assert re.findall(r'href="([^"]+)"', panel.group(0)) == TUNING


def marked(html: str, selector: str) -> list[str]:
    """Which links inside one nav are marked as the current page."""
    block = re.search(rf'<nav class="{selector}".*?</nav>', html, re.S)
    assert block, f"no {selector} nav in the page"
    return re.findall(r'href="([^"]+)"[^>]*aria-current="page"', block.group(0))


@pytest.mark.parametrize("path", TOP_LEVEL)
def test_the_nav_marks_where_you_are(client: TestClient, path: str) -> None:
    """Without this a page renders fine and simply looks unvisited."""
    assert marked(client.get(path).text, "nav") == [path]


@pytest.mark.parametrize("path", TUNING + ["/add"])
def test_pages_outside_the_loop_mark_nothing_in_the_nav(
    client: TestClient, path: str
) -> None:
    assert marked(client.get(path).text, "nav") == []


@pytest.mark.parametrize("path", TUNING)
def test_the_tuning_pages_carry_their_own_tabs(client: TestClient, path: str) -> None:
    html = client.get(path).text
    tabs = re.search(r'<nav class="tabs".*?</nav>', html, re.S)
    assert tabs
    assert re.findall(r'href="([^"]+)"', tabs.group(0)) == TUNING
    assert marked(html, "tabs") == [path]


# --------------------------------------------------------------- asset safety


def test_the_stylesheet_url_changes_when_the_stylesheet_does(
    client: TestClient,
) -> None:
    """Markup and CSS ship as a pair.

    A browser holding the previous stylesheet against new markup does not
    degrade gracefully — inline SVG with no CSS lays out at 300x150, so the page
    becomes a wall of enormous icons. Keying the URL to the file prevents the
    pairing rather than relying on anyone to hard-refresh.
    """
    html = client.get("/").text
    assert re.search(r'href="/static/style\.css\?v=\d+"', html)


SVG_FILES = sorted(
    p for p in (Path(__file__).parent.parent / "advisor/web/templates").glob("*.html")
    if "<svg" in p.read_text()
)


@pytest.mark.parametrize("path", SVG_FILES, ids=lambda p: p.name)
def test_every_inline_svg_carries_its_own_size(path: Path) -> None:
    """The belt to the cache-buster's braces: sized icons even with no CSS."""
    for tag in re.findall(r"<svg\b[^>]*>", path.read_text(), re.S):
        assert 'width="' in tag and 'height="' in tag, f"{path.name}: {tag[:60]}"
