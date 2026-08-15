"""The MCP server: the advisor answering an assistant's questions.

Driven through a real client session over stdio rather than by calling the
functions directly — the point of these is that the protocol surface works,
not that the Python underneath it does.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys

import pytest

mcp = pytest.importorskip("mcp", reason="needs the optional [mcp] extra")

from mcp import ClientSession, StdioServerParameters, stdio_client  # noqa: E402

from advisor.models import Paper, now, upsert_paper  # noqa: E402

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def seeded(tmp_path) -> str:
    """A data directory with a small library, returned as a path string."""
    from advisor import db

    # config.load() resolves $XDG_DATA_HOME/advisor, so seed exactly there.
    conn = db.connect(tmp_path / "advisor" / "advisor.db")
    read = upsert_paper(
        conn,
        Paper(title="Doubly-Efficient PIR From Ring Learning",
              abstract="A doubly-efficient PIR scheme.",
              authors=["Ada Rao"], arxiv_id="m.1", published_at="2025-01-01"),
    )
    upsert_paper(
        conn,
        Paper(title="Succinct Arguments for Verifiable Computation",
              abstract="SNARKs for delegated computation.",
              authors=["Someone Else"], arxiv_id="m.2", published_at="2026-01-01"),
    )
    conn.execute(
        "INSERT INTO library (paper_id, status, added_at) VALUES (?,'read',?)",
        (read, now()),
    )
    conn.commit()
    conn.close()
    return str(tmp_path)


def client_for(data_dir: str) -> StdioServerParameters:
    """Launch the server against a throwaway data directory, not the real one."""
    env = dict(os.environ, XDG_DATA_HOME=data_dir, XDG_CONFIG_HOME=data_dir)
    return StdioServerParameters(
        command=sys.executable, args=["-m", "advisor.cli", "mcp"], env=env
    )


def payload(result):
    """A tool's return value, from the structured half of the response."""
    structured = result.structured_content or {}
    return structured.get("result", structured)


async def test_the_server_starts_and_advertises_its_tools(seeded: str) -> None:
    async with stdio_client(client_for(seeded)) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            listing = await session.list_tools()

            assert init.server_info.name == "research-advisor"
            names = {tool.name for tool in listing.tools}
            assert {"library", "search_corpus", "write_interest_profile"} <= names

            # Every tool must declare whether it changes anything: a client
            # decides whether to confirm with the user on that basis.
            for tool in listing.tools:
                assert tool.annotations is not None, f"{tool.name} has no annotations"
                assert tool.annotations.read_only_hint is not None
                assert tool.description


async def test_only_the_intended_tools_can_write(seeded: str) -> None:
    """A stray write tool is how a chat quietly edits someone's library."""
    async with stdio_client(client_for(seeded)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listing = await session.list_tools()

    writers = {t.name for t in listing.tools if not t.annotations.read_only_hint}

    assert writers == {"write_interest_profile", "follow_author", "unfollow_author"}


async def test_it_reads_the_library_and_the_corpus(seeded: str) -> None:
    async with stdio_client(client_for(seeded)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            library = payload(await session.call_tool("library", {}))
            assert library["count"] == 1
            assert library["papers"][0]["title"].startswith("Barely")

            found = payload(
                await session.call_tool("search_corpus", {"query": "SNARKs"})
            )
            assert found["total"] == 1
            assert "Succinct Arguments" in found["papers"][0]["title"]


async def test_writing_a_profile_takes_effect(seeded: str) -> None:
    """The headline use: discuss a change, then have it applied."""
    content = "## Working on\nPIR.\n\n## More of\nverifiable computation\n"

    async with stdio_client(client_for(seeded)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            saved = payload(
                await session.call_tool("write_interest_profile", {"content": content})
            )
            assert saved["steering"]["more_of"] == ["verifiable computation"]

            current = payload(await session.call_tool("interest_profile", {}))
            # save() strips, so compare against the stored form.
            assert current["content"] == content.strip()
            assert current["steering"]["more_of"] == ["verifiable computation"]


async def test_an_empty_profile_is_refused(seeded: str) -> None:
    """Saving "" would silently wipe the steering that drives retrieval."""
    async with stdio_client(client_for(seeded)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = payload(
                await session.call_tool("write_interest_profile", {"content": "   "})
            )

    assert "error" in result


async def test_a_prose_only_profile_says_it_steers_nothing(seeded: str) -> None:
    """Otherwise an assistant reports success on a profile that does nothing."""
    async with stdio_client(client_for(seeded)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            saved = payload(
                await session.call_tool(
                    "write_interest_profile", {"content": "## Working on\nPIR."}
                )
            )

    assert "does not change what they are shown" in saved["note"]


async def test_following_round_trips(seeded: str) -> None:
    async with stdio_client(client_for(seeded)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            assert payload(
                await session.call_tool("follow_author", {"name": "Ada Rao"})
            )["followed"] == "Ada Rao"
            # The same person spelled differently is not a second follow.
            assert payload(
                await session.call_tool("follow_author", {"name": "A. Rao"})
            )["followed"] is None
            assert payload(await session.call_tool("followed_authors", {}))[
                "following"
            ] == ["Ada Rao"]
            assert payload(
                await session.call_tool("unfollow_author", {"name": "Ada Rao"})
            )["unfollowed"] == "Ada Rao"


async def test_a_missing_paper_reports_rather_than_crashes(seeded: str) -> None:
    async with stdio_client(client_for(seeded)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = payload(await session.call_tool("paper", {"paper_id": 999999}))

    assert "error" in result
