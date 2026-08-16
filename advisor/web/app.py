"""FastAPI application: the browser UI.

Server-rendered Jinja2 with HTMX for the interactive bits, so there is no build
step and ``advisor serve`` is the whole story. One SQLite connection per request
— cheap under WAL, and it keeps the threadpool that FastAPI runs sync endpoints
in from sharing a connection across threads.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, Form, Request
from fastapi import UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from advisor import config, db
from advisor.ingest import resolve
from advisor.models import Paper, now, upsert_paper
from advisor.recommend import feedback as fb
from advisor.recommend import pipeline
from advisor.recommend import profile as prof

HERE = Path(__file__).parent

app = FastAPI(title="Research Advisor")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")

CFG = config.load()


def get_db() -> Iterator[sqlite3.Connection]:
    conn = db.connect(CFG.db_path)
    try:
        yield conn
    finally:
        conn.close()


def _render(request: Request, template: str, **context) -> HTMLResponse:
    return templates.TemplateResponse(request, template, {"cfg": CFG, **context})


def _local(target: str, fallback: str) -> str:
    """A redirect target that cannot leave this app.

    "starts with a slash" is not enough: `//evil.example.com` is a
    protocol-relative URL and browsers follow it off-site, as does `/\\evil`
    on some. Only a single leading slash counts as local.
    """
    if target.startswith("/") and not target.startswith(("//", "/\\")):
        return target
    return fallback


# --------------------------------------------------------------------------- feed


@app.get("/", response_class=HTMLResponse)
def feed(request: Request, conn: sqlite3.Connection = Depends(get_db)) -> HTMLResponse:
    rows = pipeline.latest(conn)

    return _render(
        request,
        "feed.html",
        recommendations=[
            {
                "rec_id": row["rec_id"],
                "score": row["score"],
                "rationale": row["rationale"],
                "paper": Paper.from_row(row),
            }
            for row in rows
        ],
        corpus_size=conn.execute("SELECT count(*) FROM papers").fetchone()[0],
        library_size=conn.execute("SELECT count(*) FROM library").fetchone()[0],
        embedded=conn.execute("SELECT count(*) FROM vector_index").fetchone()[0],
        # Retrieval builds your interests only from library papers that have a
        # vector. A freshly added paper has none until the embed pass reaches
        # it, and without this the page cannot tell "nothing to recommend" from
        # "not ready to recommend yet" — so the button appears to do nothing.
        library_embedded=conn.execute(
            """SELECT count(*) FROM library l
                 JOIN vector_index v ON v.paper_id = l.paper_id"""
        ).fetchone()[0],
    )


@app.post("/recommend")
def generate(conn: sqlite3.Connection = Depends(get_db)) -> RedirectResponse:
    pipeline.run(conn, CFG)
    return RedirectResponse("/", status_code=303)


@app.get("/feed/{rec_id}/why", response_class=HTMLResponse)
def why_form(request: Request, rec_id: int) -> HTMLResponse:
    """The reason chips revealed by a thumbs-down.

    Asked *after* the click rather than before it, so rejecting a paper stays a
    single keystroke and explaining why stays optional.
    """
    return _render(request, "_why.html", rec_id=rec_id, reasons=fb.REASONS)


@app.post("/feed/{rec_id}/{action}")
def act_on_recommendation(
    rec_id: int,
    action: str,
    request: Request,
    note: str = Form(""),
    tags: list[str] = Form(default=[]),
    conn: sqlite3.Connection = Depends(get_db),
) -> RedirectResponse:
    """Record a reaction from the feed.

    A thumbs up or down is both an action on this recommendation and a piece of
    feedback that reshapes the next run's preference vectors — hence both rows.
    """
    if action not in ("liked", "disliked", "queued", "dismissed"):
        return RedirectResponse("/", status_code=303)

    paper_id = pipeline.record_action(conn, rec_id, action)
    if paper_id is None:
        return RedirectResponse("/", status_code=303)

    if action in ("liked", "disliked"):
        fb.record(conn, paper_id, 1 if action == "liked" else -1, tags, note)
    if action == "queued":
        conn.execute(
            """INSERT INTO library (paper_id, status, added_at) VALUES (?, 'queued', ?)
               ON CONFLICT(paper_id) DO NOTHING""",
            (paper_id, now()),
        )

    return RedirectResponse("/", status_code=303)


# -------------------------------------------------------------------------- profile


@app.get("/profile", response_class=HTMLResponse)
def profile_page(
    request: Request, conn: sqlite3.Connection = Depends(get_db)
) -> HTMLResponse:
    current = prof.current(conn)
    return _render(
        request,
        "profile.html",
        profile=current,
        history=prof.history(conn)[1:],
        pending=prof.feedback_since_last_profile(conn),
        # What retrieval will actually act on, so the page can show that the
        # document is doing something rather than merely being stored.
        steer=prof.parse(current.content if current else None),
        starter=prof.TEMPLATE,
        # Evidence plus instructions, for handing to an assistant of your own
        # choosing. Offered as text to copy rather than a button that sends it:
        # nothing leaves this machine unless you carry it out yourself.
        brief=prof.briefing(conn),
        # Your own words, to fold into the profile — retrieval cannot read a
        # note, but it can read a line you lift out of one.
        notes=conn.execute(
            """SELECT f.note, f.rating, p.title FROM feedback f
                 JOIN papers p ON p.id = f.paper_id
                WHERE f.note IS NOT NULL AND trim(f.note) != ''
                ORDER BY f.id DESC LIMIT 10"""
        ).fetchall(),
    )


@app.post("/profile/edit")
def profile_edit(
    content: str = Form(""), conn: sqlite3.Connection = Depends(get_db)
) -> RedirectResponse:
    """A hand edit is stored as a new version, never an overwrite.

    You get the last word on how the advisor sees you, and the history shows
    where you disagreed with it.
    """
    if content.strip():
        prof.save(conn, content, written_by="user")
    return RedirectResponse("/profile", status_code=303)


# ------------------------------------------------------------------------- search


@app.get("/search", response_class=HTMLResponse)
def search_page(
    request: Request,
    q: str = "",
    n: int = 25,
    conn: sqlite3.Connection = Depends(get_db),
) -> HTMLResponse:
    from advisor import search as fts

    query = q.strip()
    limit = max(1, min(n, 200))
    return _render(
        request,
        "search.html",
        query=query,
        hits=fts.search(conn, query, limit=limit) if query else [],
        total=fts.count(conn, query) if query else 0,
        corpus_size=conn.execute("SELECT count(*) FROM papers").fetchone()[0],
    )


@app.post("/add/one")
def add_one(
    paper_id: int = Form(...),
    back: str = Form("/search"),
    conn: sqlite3.Connection = Depends(get_db),
) -> RedirectResponse:
    """Mark a paper already in the corpus as read, from a search result."""
    conn.execute(
        """INSERT INTO library (paper_id, status, added_at, read_at)
           VALUES (?, 'read', ?, ?)
           ON CONFLICT(paper_id) DO UPDATE SET status = 'read'""",
        (paper_id, now(), now()),
    )
    return RedirectResponse(_local(back, "/search"), status_code=303)


# --------------------------------------------------------------------- your data


@app.get("/data", response_class=HTMLResponse)
def data_page(
    request: Request, imported: str = "", conn: sqlite3.Connection = Depends(get_db)
) -> HTMLResponse:
    def count(table: str) -> int:
        return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    return _render(
        request,
        "data.html",
        counts={
            "library entries": count("library"),
            "papers rated": fb.rated_papers(conn),
            "profile versions": count("profile_versions"),
            "followed authors": count("followed_authors"),
        },
        corpus_size=count("papers"),
        imported=imported,
    )


@app.get("/data/export")
def data_export(conn: sqlite3.Connection = Depends(get_db)) -> Response:
    """Download everything personal as one file."""
    from advisor import portable

    stamp = now()[:10]
    return Response(
        content=portable.dumps(conn),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="advisor-{stamp}.json"'
        },
    )


@app.post("/data/import")
async def data_import(
    file: UploadFile,
    replace: str = Form(""),
    conn: sqlite3.Connection = Depends(get_db),
) -> RedirectResponse:
    from advisor import portable

    raw = await file.read()
    try:
        report = portable.loads(conn, raw.decode("utf-8"), replace=bool(replace))
    except (ValueError, UnicodeDecodeError) as exc:
        return RedirectResponse(f"/data?imported=error: {exc}", status_code=303)

    return RedirectResponse(f"/data?imported={report}", status_code=303)


# ------------------------------------------------------------------------ authors


@app.get("/authors", response_class=HTMLResponse)
def authors_page(
    request: Request, problem: str = "", conn: sqlite3.Connection = Depends(get_db)
) -> HTMLResponse:
    from advisor import authors

    return _render(
        request,
        "authors.html",
        following=authors.following(conn),
        suggestions=authors.suggestions(conn)[:15],
        problem=problem,
    )


@app.post("/authors/follow")
def authors_follow(
    name: str = Form(""), conn: sqlite3.Connection = Depends(get_db)
) -> RedirectResponse:
    from advisor import authors

    if not name.strip():
        return RedirectResponse("/authors", status_code=303)

    result = authors.follow(conn, name)
    if result:
        return RedirectResponse("/authors", status_code=303)

    problem = f"“{name.strip()}” — {result.reason}"
    if result.suggestions:
        problem += ". Did you mean: " + ", ".join(result.suggestions)
    return RedirectResponse(f"/authors?problem={quote(problem)}", status_code=303)


@app.post("/authors/unfollow")
def authors_unfollow(
    name: str = Form(""), conn: sqlite3.Connection = Depends(get_db)
) -> RedirectResponse:
    from advisor import authors

    authors.unfollow(conn, name)
    return RedirectResponse("/authors", status_code=303)


# ------------------------------------------------------------------------ library


@app.get("/library", response_class=HTMLResponse)
def library(
    request: Request,
    status: str = "",
    conn: sqlite3.Connection = Depends(get_db),
) -> HTMLResponse:
    query = """SELECT p.*, l.status, l.added_at,
                      (SELECT rating FROM feedback f
                        WHERE f.paper_id = p.id
                        ORDER BY f.created_at DESC, f.id DESC LIMIT 1) AS rating
                 FROM library l JOIN papers p ON p.id = l.paper_id"""
    params: tuple = ()
    if status in ("read", "queued", "skipped"):
        query += " WHERE l.status = ?"
        params = (status,)
    query += " ORDER BY l.added_at DESC, p.id DESC"

    rows = conn.execute(query, params).fetchall()
    entries = [
        {"paper": Paper.from_row(row), "status": row["status"], "rating": row["rating"]}
        for row in rows
    ]

    counts = dict(
        conn.execute("SELECT status, count(*) FROM library GROUP BY status").fetchall()
    )
    return _render(
        request, "library.html", entries=entries, counts=counts, active_status=status
    )


@app.post("/library/{paper_id}/status")
def set_status(
    paper_id: int,
    status: str = Form(...),
    conn: sqlite3.Connection = Depends(get_db),
) -> RedirectResponse:
    if status in ("read", "queued", "skipped"):
        read_at = now() if status == "read" else None
        conn.execute(
            "UPDATE library SET status = ?, read_at = ? WHERE paper_id = ?",
            (status, read_at, paper_id),
        )
    return RedirectResponse("/library", status_code=303)


@app.post("/library/{paper_id}/rate")
def rate(
    paper_id: int,
    rating: int = Form(...),
    note: str = Form(""),
    tags: list[str] = Form(default=[]),
    conn: sqlite3.Connection = Depends(get_db),
) -> RedirectResponse:
    if rating in (-1, 0, 1):
        fb.record(conn, paper_id, rating, tags, note)
    return RedirectResponse("/library", status_code=303)


@app.post("/library/{paper_id}/remove")
def remove(paper_id: int, conn: sqlite3.Connection = Depends(get_db)) -> RedirectResponse:
    # The paper stays in the corpus; only your library membership is dropped.
    conn.execute("DELETE FROM library WHERE paper_id = ?", (paper_id,))
    return RedirectResponse("/library", status_code=303)


# ---------------------------------------------------------------------------- add


@app.get("/add", response_class=HTMLResponse)
def add_form(request: Request) -> HTMLResponse:
    return _render(request, "add.html")


@app.post("/add/preview", response_class=HTMLResponse)
async def add_preview(
    request: Request,
    ids: str = Form(""),
    conn: sqlite3.Connection = Depends(get_db),
) -> HTMLResponse:
    refs, unparsed = resolve.parse_many(ids)

    resolved: list[dict] = []
    for ref in refs:
        try:
            paper = await resolve.resolve(conn, ref)
        except Exception as exc:  # network failure on one ID must not lose the rest
            resolved.append({"ref": str(ref), "paper": None, "error": str(exc)})
            continue

        already = False
        if paper is not None:
            local = resolve.find_local(conn, ref)
            if local and local.id:
                already = bool(
                    conn.execute(
                        "SELECT 1 FROM library WHERE paper_id = ?", (local.id,)
                    ).fetchone()
                )
        resolved.append({"ref": str(ref), "paper": paper, "error": None, "already": already})

    return _render(
        request,
        "_add_preview.html",
        resolved=resolved,
        unparsed=unparsed,
        raw_ids=ids,
    )


@app.post("/add/commit")
async def add_commit(
    ids: str = Form(""),
    status: str = Form("read"),
    conn: sqlite3.Connection = Depends(get_db),
) -> RedirectResponse:
    refs, _ = resolve.parse_many(ids)
    if status not in ("read", "queued", "skipped"):
        status = "read"

    for ref in refs:
        try:
            paper = await resolve.resolve(conn, ref)
        except Exception:
            continue
        if paper is None:
            continue

        with db.transaction(conn):
            paper_id = upsert_paper(conn, paper)
            conn.execute(
                """INSERT INTO library (paper_id, status, added_at, read_at)
                   VALUES (?,?,?,?)
                   ON CONFLICT(paper_id) DO UPDATE SET status = excluded.status""",
                (paper_id, status, now(), now() if status == "read" else None),
            )

    return RedirectResponse("/library", status_code=303)
