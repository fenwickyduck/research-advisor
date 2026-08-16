"""An MCP server, so you can talk to an assistant *about* your reading.

The direction matters and is the opposite of the usual integration: the advisor
does not call a model. This exposes your library, your corpus and your profile
as tools, and an assistant you already run — Claude Desktop, say — connects to
it and calls them. Your machine answers questions; it never asks any. Nothing
here holds a credential, and the advisor keeps working with this server
switched off.

What it makes possible is the conversation the app itself cannot have. "What
have I been reading lately?" reads your library. "I want to move toward
verifiable computation" can search the corpus for what that would mean, then
write the ``## More of`` lines that actually steer retrieval — the part of the
profile you would otherwise have to phrase yourself.

Run it with ``advisor mcp``; ``advisor mcp --config`` prints the JSON to paste
into a client. Everything is stdio, so nothing listens on a port.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from advisor import authors as author_store
from advisor import config, db
from advisor import search as fts
from advisor.recommend import pipeline, profile, retrieve

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False)
WRITES = ToolAnnotations(read_only_hint=False, destructive_hint=False)

INSTRUCTIONS = """\
This is the user's personal research-paper advisor. It holds three things: a
library of papers they have read, a corpus of ~76,000 papers harvested from
arXiv and the Cryptology ePrint Archive, and an interest profile.

The profile is not a description — two of its sections are executable. Lines
under `## More of` and `## Less of` are encoded by the same model that encodes
the corpus and used directly as search directions, so editing them changes what
the user is shown tomorrow. Write those lines like paper titles ("doubly-
efficient private information retrieval"), never as sentences about the person
("they are interested in PIR"). Everything under other headings is prose.

Before rewriting the profile, read it (`interest_profile`) and the evidence
(`profile_briefing`), and prefer editing what is there to replacing it — every
save is a new version the user can see. When they describe wanting to read
something different, search the corpus first to find out what that literature
is actually called, and use the real vocabulary in the steering lines.
"""

server = MCPServer(
    name="research-advisor",
    title="Research Advisor",
    instructions=INSTRUCTIONS,
    version="0.1.0",
)

CFG = config.load()


def _conn() -> sqlite3.Connection:
    """A fresh connection per call: the server is long-lived, SQLite handles
    are not, and the web app may be writing at the same time."""
    return db.connect(CFG.db_path)


def _paper_row(row: sqlite3.Row, abstract: bool = False) -> dict[str, Any]:
    out = {
        "id": row["id"],
        "title": row["title"],
        "authors": db.json_list(row["authors"]),
        "year": (row["published_at"] or "")[:4] or None,
        "url": row["url"],
    }
    if abstract:
        out["abstract"] = row["abstract"]
    return out


# ------------------------------------------------------------------ what they read


@server.tool(
    description="Papers the user has read, newest first. Start here to ground "
    "any discussion of what they work on.",
    annotations=READ_ONLY,
)
def library(limit: int = 50, include_abstracts: bool = False) -> dict[str, Any]:
    conn = _conn()
    try:
        rows = conn.execute(
            """SELECT p.*, l.status, l.added_at FROM library l
                 JOIN papers p ON p.id = l.paper_id
                ORDER BY l.added_at DESC LIMIT ?""",
            (max(1, min(limit, 200)),),
        ).fetchall()
        return {
            "count": conn.execute("SELECT count(*) FROM library").fetchone()[0],
            "papers": [
                _paper_row(row, include_abstracts) | {"status": row["status"]}
                for row in rows
            ],
        }
    finally:
        conn.close()


@server.tool(
    description="The user's ratings and their free-text notes — what they "
    "thought, which is stronger evidence than what they read.",
    annotations=READ_ONLY,
)
def ratings(limit: int = 40) -> list[dict[str, Any]]:
    conn = _conn()
    try:
        rows = conn.execute(
            """SELECT f.rating, f.tags, f.note, f.created_at, p.title
                 FROM feedback f JOIN papers p ON p.id = f.paper_id
                WHERE f.id = (SELECT max(id) FROM feedback g WHERE g.paper_id = f.paper_id)
                ORDER BY f.id DESC LIMIT ?""",
            (max(1, min(limit, 200)),),
        ).fetchall()
        return [
            {
                "title": row["title"],
                "verdict": "liked" if row["rating"] > 0 else "rejected",
                "reasons": db.json_list(row["tags"]),
                "note": row["note"],
                "when": row["created_at"][:10],
            }
            for row in rows
        ]
    finally:
        conn.close()


# --------------------------------------------------------------------- the corpus


@server.tool(
    description="Full-text search over the ~76,000 harvested papers. Use this "
    "to find out what a literature is actually called before writing profile "
    "steering lines. Quote a phrase to keep it together; prefix a word with - "
    "to exclude it.",
    annotations=READ_ONLY,
)
def search_corpus(query: str, limit: int = 15) -> dict[str, Any]:
    conn = _conn()
    try:
        hits = fts.search(conn, query, limit=max(1, min(limit, 50)))
        return {
            "total": fts.count(conn, query),
            "showing": len(hits),
            "papers": [
                {
                    "id": hit.paper_id,
                    "title": hit.title,
                    "authors": hit.authors,
                    "year": (hit.published_at or "")[:4] or None,
                    "url": hit.url,
                    "excerpt": hit.text,
                }
                for hit in hits
            ],
        }
    finally:
        conn.close()


@server.tool(
    description="One paper in full, including its abstract, by the id returned "
    "from search or the feed.",
    annotations=READ_ONLY,
)
def paper(paper_id: int) -> dict[str, Any]:
    conn = _conn()
    try:
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if row is None:
            return {"error": f"no paper with id {paper_id}"}
        return _paper_row(row, abstract=True) | {
            "categories": db.json_list(row["categories"]),
            "in_library": conn.execute(
                "SELECT 1 FROM library WHERE paper_id = ?", (paper_id,)
            ).fetchone()
            is not None,
        }
    finally:
        conn.close()


@server.tool(
    description="What the advisor currently recommends, with the reason each "
    "paper was chosen. Read-only: it shows the last batch rather than "
    "generating one, so asking does not consume the user's feed.",
    annotations=READ_ONLY,
)
def current_recommendations() -> list[dict[str, Any]]:
    conn = _conn()
    try:
        return [
            {
                "title": row["title"],
                "why": row["rationale"],
                "score": row["score"],
                "url": row["url"],
                "id": row["id"],
            }
            for row in pipeline.latest(conn)
        ]
    finally:
        conn.close()


@server.tool(
    description="Preview what would be recommended right now, without "
    "recording it. Useful for 'what would I get if I changed my profile'.",
    annotations=READ_ONLY,
)
def preview_recommendations(limit: int = 10) -> list[dict[str, Any]]:
    conn = _conn()
    try:
        candidates = retrieve.recommend(conn, CFG, limit=max(1, min(limit, 50)))
        attributions = retrieve.explain(conn, candidates, CFG)
        out = []
        for candidate in candidates:
            row = conn.execute(
                "SELECT title, url FROM papers WHERE id = ?", (candidate.paper_id,)
            ).fetchone()
            source = attributions.get(candidate.paper_id)
            out.append(
                {
                    "title": row["title"],
                    "url": row["url"],
                    "why": f"By {candidate.via}, whom you follow."
                    if candidate.via
                    else (source.sentence() if source else None),
                }
            )
        return out
    finally:
        conn.close()


# --------------------------------------------------------------------- the profile


@server.tool(
    description="The user's current interest profile, plus which of its lines "
    "are actively steering retrieval.",
    annotations=READ_ONLY,
)
def interest_profile() -> dict[str, Any]:
    conn = _conn()
    try:
        current = profile.current(conn)
        steer = profile.parse(current.content if current else None)
        return {
            "content": current.content if current else None,
            "written_by": current.written_by if current else None,
            "written_at": current.created_at[:16] if current else None,
            "steering": {"more_of": steer.more, "less_of": steer.less},
            "ratings_since_written": profile.feedback_since_last_profile(conn),
        }
    finally:
        conn.close()


@server.tool(
    description="Everything a profile could be written from — library, "
    "ratings, notes — with the house instructions for writing one. Read this "
    "before rewriting the profile.",
    annotations=READ_ONLY,
)
def profile_briefing() -> str:
    conn = _conn()
    try:
        return profile.briefing(conn) or "Nothing to write a profile from yet."
    finally:
        conn.close()


@server.tool(
    description="Save a new version of the interest profile. Takes effect on "
    "the next batch: lines under '## More of' and '## Less of' are encoded and "
    "used as search directions, so write them like paper titles. Previous "
    "versions are kept and the user can revert at /profile.",
    annotations=WRITES,
)
def write_interest_profile(content: str) -> dict[str, Any]:
    if not content.strip():
        return {"error": "refusing to save an empty profile"}

    conn = _conn()
    try:
        version = profile.save(conn, content, written_by="user")
        steer = profile.parse(content)
        return {
            "saved_version": version,
            "steering": {"more_of": steer.more, "less_of": steer.less},
            "note": "No '## More of' or '## Less of' lines, so this profile "
            "describes the user but does not change what they are shown."
            if not steer
            else "In effect from the next batch.",
        }
    finally:
        conn.close()


# --------------------------------------------------------------------- the people


@server.tool(
    description="Authors the user follows. Their new work is pulled into every "
    "batch by name, whatever it is about.",
    annotations=READ_ONLY,
)
def followed_authors() -> dict[str, Any]:
    conn = _conn()
    try:
        return {
            "following": [row["name"] for row in author_store.following(conn)],
            "suggested": [
                {"name": name, "papers_read": n}
                for name, n in author_store.suggestions(conn)[:10]
            ],
        }
    finally:
        conn.close()


@server.tool(
    description="Follow an author, so their new papers appear regardless of "
    "topic. Give the fullest name you have — 'Wei Chen', not 'Chen'.",
    annotations=WRITES,
)
def follow_author(name: str) -> dict[str, Any]:
    conn = _conn()
    try:
        result = author_store.follow(conn, name)
        if result:
            return {"followed": name, "papers_in_corpus": result.papers}
        return {
            "followed": None,
            "reason": result.reason,
            "did_you_mean": list(result.suggestions),
        }
    finally:
        conn.close()


@server.tool(description="Stop following an author.", annotations=WRITES)
def unfollow_author(name: str) -> dict[str, Any]:
    conn = _conn()
    try:
        return {"unfollowed": name if author_store.unfollow(conn, name) else None}
    finally:
        conn.close()


# ------------------------------------------------------------------------ prompts
#
# A prompt is a workflow the user starts deliberately — a slash command rather
# than something the model decides to do. These are not static templates: each
# one reads the database first and hands over the evidence already gathered, so
# the conversation opens with the facts in hand instead of spending three tool
# calls collecting them. Each ends by asking for confirmation before writing,
# because the user invoked a discussion, not an edit.


@server.prompt(
    title="Refresh my interest profile",
    description="Rewrite the interest profile from everything read and rated "
    "since it was last written.",
)
def refresh_profile() -> str:
    conn = _conn()
    try:
        current = profile.current(conn)
        steer = profile.parse(current.content if current else None)
        pending = profile.feedback_since_last_profile(conn)
        brief = profile.briefing(conn)
    finally:
        conn.close()

    if not brief:
        return (
            "My research advisor has nothing to write a profile from yet — no "
            "papers in the library. Tell me to add some reading first, with "
            "`advisor add` or the /add page."
        )

    if current:
        standing = (
            f"My current profile was written on {current.created_at[:10]}. "
            f"I have rated {pending} paper(s) since.\n\n"
            f"It currently steers retrieval toward {steer.more or 'nothing'} "
            f"and away from {steer.less or 'nothing'}.\n\n"
            f"--- current profile ---\n{current.content}\n--- end ---\n\n"
        )
        task = (
            "Revise it against the evidence below. Keep what still holds and "
            "change only what the evidence contradicts — this is an edit, not "
            "a fresh start."
        )
    else:
        standing = "I have no interest profile yet.\n\n"
        task = "Write my first one from the evidence below."

    return (
        f"{standing}{task}\n\n"
        "Before you write the steering lines, use `search_corpus` to check that "
        "the phrasing you are about to use matches how the literature actually "
        "names itself — those lines are encoded and used as search directions, "
        "so a phrase nobody writes retrieves nothing.\n\n"
        "Show me the draft and what it would change about my steering. Save it "
        "with `write_interest_profile` only after I say so.\n\n"
        f"{brief}"
    )


@server.prompt(
    title="Explore a new direction",
    description="Work out what a topic is really called, and steer the advisor "
    "toward it.",
)
def explore(topic: str) -> str:
    conn = _conn()
    try:
        current = profile.current(conn)
        steer = profile.parse(current.content if current else None)
        hits = fts.search(conn, topic, limit=12)
        total = fts.count(conn, topic)
    finally:
        conn.close()

    found = (
        "\n".join(
            f"- {hit.title} ({(hit.published_at or '')[:4]}) — {hit.authors}"
            for hit in hits
        )
        or "Nothing matched that phrase, which is itself informative: either the "
        "corpus does not cover it, or the field calls it something else."
    )

    return (
        f"I want to read more about {topic}.\n\n"
        f"My profile currently steers toward {steer.more or 'nothing in particular'}"
        f" and away from {steer.less or 'nothing'}.\n\n"
        f"A search for “{topic}” in my corpus returns {total} papers. The first "
        f"few:\n\n{found}\n\n"
        "Help me work out whether this is worth steering toward, and what to "
        "call it. Search again with the vocabulary those titles actually use — "
        "my phrasing is probably not the field's. Then propose the exact "
        "`## More of` lines to add, written like paper titles rather than like "
        "sentences about me.\n\n"
        "Tell me what I would stop seeing as a result. Do not save anything "
        "until I agree to it."
    )


@server.prompt(
    title="Go through my feed",
    description="Review what the advisor is currently recommending, and why.",
)
def review_feed() -> str:
    conn = _conn()
    try:
        rows = pipeline.latest(conn)
        library_size = conn.execute("SELECT count(*) FROM library").fetchone()[0]
    finally:
        conn.close()

    if not rows:
        return (
            "My advisor has nothing in its feed right now. Check "
            "`preview_recommendations` to see whether that is because there is "
            "nothing to suggest or because no batch has been generated, and "
            "tell me which."
        )

    listing = "\n".join(
        f"{row['rank']}. {row['title']}\n   why: {row['rationale'] or 'no reason recorded'}"
        for row in rows
    )

    return (
        f"Here is what my research advisor is recommending. I have "
        f"{library_size} paper(s) in my library.\n\n{listing}\n\n"
        "Go through these with me. For any you cannot judge from the title, "
        "read the abstract with `paper`. I want to know which are worth my "
        "time and which are noise — and if several are noise for the same "
        "reason, say what that reason is and whether a `## Less of` line would "
        "prevent it. Be blunt; a list where everything is interesting is no "
        "use to me."
    )


# ------------------------------------------------------------------------- config


def client_config(command: str) -> str:
    """The JSON block a client needs in order to launch this server."""
    return json.dumps(
        {"mcpServers": {"research-advisor": {"command": command, "args": ["mcp"]}}},
        indent=2,
    )


def main() -> None:
    server.run(transport="stdio")
