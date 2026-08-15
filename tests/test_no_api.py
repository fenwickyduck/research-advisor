"""The advisor makes no model calls and holds no credentials.

This is a deliberate constraint, not an accident of the current code, so it is
asserted rather than assumed. Every path here was reachable at some point in
this project's history; the tests exist so re-adding one is a decision someone
has to make on purpose.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from advisor.config import Config

SOURCE = Path(__file__).resolve().parent.parent / "advisor"
FORBIDDEN_IMPORTS = {"anthropic", "openai", "httpx_sse"}
# Environment variables that would carry a credential into the process.
CREDENTIAL_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY")


def python_files() -> list[Path]:
    return sorted(p for p in SOURCE.rglob("*.py") if "__pycache__" not in p.parts)


def imported_names(path: Path) -> set[str]:
    """Every module named by an import in this file, including inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_no_model_sdk_is_imported_anywhere() -> None:
    """Including lazily, inside a function — that is how it used to be done."""
    offenders = {
        path.relative_to(SOURCE).as_posix(): sorted(imported_names(path) & FORBIDDEN_IMPORTS)
        for path in python_files()
        if imported_names(path) & FORBIDDEN_IMPORTS
    }

    assert not offenders, f"model SDK imported: {offenders}"


def test_no_credential_is_read_from_the_environment() -> None:
    source = "\n".join(p.read_text(encoding="utf-8") for p in python_files())

    for name in CREDENTIAL_ENV:
        assert name not in source, f"{name} referenced in advisor/"


def test_the_config_has_no_model_or_key_settings() -> None:
    """A setting is an invitation; there should be nothing to point at a service."""
    fields = set(Config.__dataclass_fields__)

    assert not {"model", "effort", "api_key", "profile_refresh_every"} & fields
    # The embedding model is local and stays.
    assert "embedding_model" in fields


def test_a_run_needs_no_credentials_and_takes_no_client(
    conn: sqlite3.Connection, tmp_path
) -> None:
    """The pipeline has no seam a client could be passed through."""
    import inspect

    from advisor.recommend import pipeline

    parameters = set(inspect.signature(pipeline.run).parameters)

    assert parameters == {"conn", "cfg", "limit"}


def test_the_profile_briefing_is_text_you_carry_yourself(
    conn: sqlite3.Connection, tmp_path
) -> None:
    """The handover is a string to copy, not a request the advisor sends."""
    from advisor.models import Paper, now, upsert_paper
    from advisor.recommend import profile

    paper_id = upsert_paper(conn, Paper(title="A Paper", authors=["X"], arxiv_id="1.1"))
    conn.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (paper_id, now()),
    )

    brief = profile.briefing(conn)

    assert "A Paper" in brief and "## More of" in brief
    assert not hasattr(profile, "write"), "profile.write() called a model — keep it gone"


def test_an_empty_library_briefs_nothing(conn: sqlite3.Connection) -> None:
    from advisor.recommend import profile

    assert profile.briefing(conn) == ""


@pytest.mark.parametrize("module", ["claude", "rank"])
def test_the_removed_modules_stay_removed(module: str) -> None:
    assert not (SOURCE / "recommend" / f"{module}.py").exists()
