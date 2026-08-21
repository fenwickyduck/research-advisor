"""Command line entry point: ``advisor <command>``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from advisor import config, db
from advisor.models import now, upsert_paper


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    cfg = config.load()
    config.ensure_dirs(cfg)
    db.connect(cfg.db_path).close()  # create the schema before the first request

    print(f"database: {cfg.db_path}")
    print(f"serving:  http://{args.host}:{args.port}")
    uvicorn.run(
        "advisor.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )
    return 0


def _add(args: argparse.Namespace) -> int:
    from advisor.ingest import resolve

    cfg = config.load()
    config.ensure_dirs(cfg)
    conn = db.connect(cfg.db_path)

    text = "\n".join(args.ids) if args.ids else sys.stdin.read()
    refs, unparsed = resolve.parse_many(text)

    for line in unparsed:
        print(f"  ?  could not read: {line}", file=sys.stderr)

    added = 0

    async def run() -> None:
        nonlocal added
        for ref in refs:
            try:
                paper = await resolve.resolve(conn, ref)
            except Exception as exc:
                print(f"  !  {ref}: {exc}", file=sys.stderr)
                continue
            if paper is None:
                print(f"  !  {ref}: not found", file=sys.stderr)
                continue

            with db.transaction(conn):
                paper_id = upsert_paper(conn, paper)
                conn.execute(
                    """INSERT INTO library (paper_id, status, added_at, read_at)
                       VALUES (?,?,?,?)
                       ON CONFLICT(paper_id) DO UPDATE SET status = excluded.status""",
                    (
                        paper_id,
                        args.status,
                        now(),
                        now() if args.status == "read" else None,
                    ),
                )
            added += 1
            print(f"  +  {paper.title}")

    asyncio.run(run())
    conn.close()
    print(f"\nAdded {added} paper(s) as '{args.status}'.")
    return 0


def _harvest(args: argparse.Namespace) -> int:
    from advisor.ingest import harvest

    cfg = config.load()
    config.ensure_dirs(cfg)
    conn = db.connect(cfg.db_path)

    def progress(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    if args.reset:
        conn.execute("DELETE FROM harvest_state")
        print("Cursors cleared; harvesting from the beginning.")

    try:
        results = asyncio.run(harvest.harvest_all(conn, cfg, progress))
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved — rerun to resume.", file=sys.stderr)
        return 130
    finally:
        conn.close()

    print()
    failed = False
    for result in results:
        print(result)
        for error in result.errors:
            failed = True
            print(f"    ! {error}", file=sys.stderr)

    return 1 if failed else 0


def _embed(args: argparse.Namespace) -> int:
    from advisor import lock
    from advisor.embed import run, store

    cfg = config.load()
    config.ensure_dirs(cfg)
    conn = db.connect(cfg.db_path)

    def progress(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    try:
        with lock.exclusive(cfg.embed_lock_path, "an embed pass"):
            if args.reset:
                store.reset(conn, cfg.vectors_path)
                print("Cleared existing vectors.")

            done = run.embed_pending(conn, cfg, progress)
    except lock.Busy as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved — rerun to resume.", file=sys.stderr)
        return 130
    finally:
        conn.close()

    print(f"\nEmbedded {done} paper(s)." if done else "Nothing to embed.")
    return 0


def _print_candidates(conn, cfg, candidates) -> None:
    from advisor.recommend import retrieve

    attributions = retrieve.explain(conn, candidates, cfg)

    for rank, candidate in enumerate(candidates, 1):
        row = conn.execute(
            "SELECT title, published_at, url FROM papers WHERE id = ?", (candidate.paper_id,)
        ).fetchone()
        year = (row["published_at"] or "????")[:4]
        # A followed-author pick has no similarity score and was not chosen by
        # similarity, so it shows neither — same as the recorded feed.
        score = "  --  " if candidate.via else f"{candidate.score:.3f}"
        print(f"{rank:3d}. [{score}] ({year}) {row['title']}")
        if candidate.via:
            print(f"      By {candidate.via}, whom you follow.")
        elif source := attributions.get(candidate.paper_id):
            print(f"      {source.sentence()}")
        print(f"      {row['url']}")


def _why_nothing(conn) -> str:
    """Say which step is actually missing, not all of them.

    "Add papers, harvest and embed" is unhelpful once you have done two of the
    three — and actively misleading in the common case where the library is
    seeded but not yet encoded, which resolves itself on its own.
    """

    def count(sql: str) -> int:
        return conn.execute(sql).fetchone()[0]

    if not count("SELECT count(*) FROM library"):
        return "No recommendations yet — start with 'advisor add' to say what you have read."

    # Papers beyond your own library, since 'advisor add' puts its papers in
    # both: a corpus consisting only of what you have read has nothing to
    # recommend from, however many rows it has.
    if not count(
        """SELECT count(*) FROM papers p
             LEFT JOIN library l ON l.paper_id = p.id
            WHERE l.paper_id IS NULL"""
    ):
        return "No recommendations yet — run 'advisor harvest' to pull a corpus."

    in_library = count(
        "SELECT count(*) FROM library l JOIN vector_index v ON v.paper_id = l.paper_id"
    )
    if not in_library:
        return (
            "No recommendations yet — your library has not been encoded, and your "
            "interests are built from it. The embed pass takes your library first, "
            "so this clears shortly after 'advisor embed' is running."
        )
    if not count("SELECT count(*) FROM vector_index"):
        return "No recommendations yet — run 'advisor embed' to encode the corpus."

    return (
        "No recommendations left to make from what is embedded so far. "
        "Let 'advisor embed' get further through the corpus, or add more papers."
    )


def _recommend(args: argparse.Namespace) -> int:
    from advisor.recommend import pipeline, retrieve

    cfg = config.load()
    conn = db.connect(cfg.db_path)

    # --preview shows the same list without recording it. Recording is the
    # default because an unrecorded batch is shown again the next time you
    # ask — the terminal would repeat itself forever while the web feed, which
    # does record, moved on.
    if args.preview:
        candidates = retrieve.recommend(conn, cfg, limit=args.limit)
        if not candidates:
            print(_why_nothing(conn), file=sys.stderr)
            conn.close()
            return 1
        _print_candidates(conn, cfg, candidates)
        conn.close()
        return 0

    run_id, count = pipeline.run(conn, cfg, limit=args.limit)
    if not run_id or not count:
        print(_why_nothing(conn), file=sys.stderr)
        conn.close()
        return 1

    for row in pipeline.latest(conn):
        year = (row["published_at"] or "????")[:4]
        print(f"{row['rank']:3d}. [{row['score']:.3f}] ({year}) {row['title']}")
        if row["rationale"]:
            print(f"      {row['rationale']}")
        print(f"      {row['url']}")

    conn.close()
    return 0


def _profile(args: argparse.Namespace) -> int:
    from advisor.recommend import profile

    cfg = config.load()
    conn = db.connect(cfg.db_path)

    # --brief prints the evidence and the instructions for writing a profile
    # from it. Paste it into whatever assistant you already have, paste the
    # answer into 'advisor serve' at /profile. The advisor itself stays offline.
    if args.brief:
        brief = profile.briefing(conn)
        conn.close()
        if not brief:
            print(
                "Nothing to write a profile from yet — add papers and rate a few.",
                file=sys.stderr,
            )
            return 1
        print(brief)
        return 0

    current = profile.current(conn)
    if current is None:
        print(
            "No profile yet. Write one at /profile, or run "
            "'advisor profile --brief' for something to hand an assistant.",
            file=sys.stderr,
        )
        conn.close()
        return 1

    print(current.content)
    print(f"\n-- written by {current.written_by} on {current.created_at[:16]}")

    steer = profile.parse(current.content)
    if steer:
        print(f"-- steering retrieval: {len(steer.more)} more, {len(steer.less)} less")
    else:
        print("-- steering nothing: add a '## More of' section and it will")

    pending = profile.feedback_since_last_profile(conn)
    if pending:
        print(f"-- {pending} rating(s) since it was written")
    conn.close()
    return 0


def _search(args: argparse.Namespace) -> int:
    from advisor import search

    cfg = config.load()
    conn = db.connect(cfg.db_path)

    query = " ".join(args.query)
    hits = search.search(conn, query, limit=args.limit)
    if not hits:
        total = conn.execute("SELECT count(*) FROM papers").fetchone()[0]
        conn.close()
        print(
            f"No matches for {query!r} in {total} papers."
            if total
            else "Nothing to search yet — run 'advisor harvest'.",
            file=sys.stderr,
        )
        return 1

    total = search.count(conn, query)
    for hit in hits:
        year = (hit.published_at or "????")[:4]
        print(f"({year}) {hit.title}")
        print(f"      {hit.authors}")
        snippet = hit.text
        if snippet.strip():
            print(f"      {snippet.strip()}")
        print(f"      {hit.url}")

    if total > len(hits):
        print(f"\n{total} matches; showing {len(hits)}. Use -n to see more.")
    conn.close()
    return 0


def _follow(args: argparse.Namespace) -> int:
    from advisor import authors

    cfg = config.load()
    conn = db.connect(cfg.db_path)

    if args.suggest:
        ranked = authors.suggestions(conn)
        if not ranked:
            print("No repeated authors in your library yet.")
        for name, count in ranked[:20]:
            print(f"{count:3d}  {name}")
        conn.close()
        return 0

    if not args.names:
        rows = authors.following(conn)
        if not rows:
            print("Following nobody. Try 'advisor follow --suggest'.")
        for row in rows:
            print(f"  {row['name']}")
        conn.close()
        return 0

    failed = False
    for name in args.names:
        result = authors.follow(conn, name)
        if result:
            papers = f" ({result.papers} papers)" if result.papers else ""
            print(f"  + {name}{papers}")
            continue
        failed = True
        print(f"  ! {name}: {result.reason}", file=sys.stderr)
        if result.suggestions:
            print(f"    did you mean: {', '.join(result.suggestions)}", file=sys.stderr)
    conn.close()
    return 1 if failed else 0


def _unfollow(args: argparse.Namespace) -> int:
    from advisor import authors

    cfg = config.load()
    conn = db.connect(cfg.db_path)
    for name in args.names:
        print(f"  - {name}" if authors.unfollow(conn, name) else f"  . {name} (not followed)")
    conn.close()
    return 0


def _mcp(args: argparse.Namespace) -> int:
    """Serve the library over MCP, so an assistant you run can consult it.

    Nothing is sent anywhere: this speaks stdio to a client that launches it,
    and answers questions about your own database.
    """
    try:
        from advisor import mcp_server
    except ImportError:
        print(
            "The 'mcp' package is not installed. Run:\n"
            "  pip install -e '.[mcp]'",
            file=sys.stderr,
        )
        return 1

    if args.config:
        from advisor.schedule import executable

        command = executable()
        print(
            "Claude Code (the terminal tool; Linux, macOS and Windows)\n"
            "registers the server for you:\n\n"
            f"  claude mcp add advisor -- {command} mcp\n\n"
            "  Add --scope user to make it available in every directory rather\n"
            "  than only this project. Then start a new 'claude' session — MCP\n"
            "  servers are loaded at startup, so an open session will not see it.\n"
            "  Check it with 'claude mcp list'.\n"
        )
        print(
            "Claude Desktop (macOS and Windows) and other clients take the same\n"
            "server as JSON, in claude_desktop_config.json:\n"
        )
        print(mcp_server.client_config(str(command)))
        print(
            "\nMerge the 'advisor' entry into any existing 'mcpServers'\n"
            "object rather than replacing the file."
        )
        return 0

    mcp_server.main()
    return 0


def _export(args: argparse.Namespace) -> int:
    from advisor import portable

    cfg = config.load()
    conn = db.connect(cfg.db_path)
    try:
        text = portable.dumps(conn)
    finally:
        conn.close()

    if args.output in ("-", None):
        sys.stdout.write(text)
        return 0

    path = Path(args.output).expanduser()
    path.write_text(text, encoding="utf-8")
    data = json.loads(text)
    print(
        f"Wrote {path} — {len(data['library'])} library entries, "
        f"{len({entry['paper'] for entry in data['feedback']})} papers rated, "
        f"{len(data['profile_versions'])} profile versions, "
        f"{len(data['followed_authors'])} followed authors."
    )
    print("The corpus and its vectors are not included; they are rebuildable.")
    return 0


def _import(args: argparse.Namespace) -> int:
    from advisor import portable

    cfg = config.load()
    config.ensure_dirs(cfg)

    text = sys.stdin.read() if args.path == "-" else Path(args.path).expanduser().read_text()

    conn = db.connect(cfg.db_path)
    try:
        if args.replace and not args.yes:
            existing = conn.execute("SELECT count(*) FROM library").fetchone()[0]
            if existing:
                print(f"--replace will delete your current {existing} library entries,")
                print("along with your ratings, profile history and follows.")
                try:
                    if input("Type 'yes' to continue: ").strip().lower() != "yes":
                        print("Cancelled.")
                        return 1
                except EOFError:
                    print("Cancelled (no terminal to confirm on; pass --yes).",
                          file=sys.stderr)
                    return 1

        report = portable.loads(conn, text, replace=args.replace)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    print(report)
    return 0


def _snapshot(args: argparse.Namespace) -> int:
    from advisor import snapshot

    cfg = config.load()
    config.ensure_dirs(cfg)
    conn = db.connect(cfg.db_path)

    def say(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    try:
        if args.action == "fetch":
            import tempfile

            target = Path(args.path) if args.path else Path(tempfile.mkdtemp())
            downloaded = snapshot.fetch(target, progress=say)
            say(f"got {downloaded.name} ({downloaded.stat().st_size / 1048576:.0f} MB)")
            meta = snapshot.load(conn, cfg, downloaded, say, replace=args.replace)
            print(
                f"\nRestored {meta['restored']} papers embedded with "
                f"{meta['embedding_model']}."
            )
            print("Run 'advisor harvest' to pick up anything published since.")
        elif args.action == "save":
            meta = snapshot.save(conn, cfg, Path(args.path), say)
            size = Path(args.path).stat().st_size / 1048576
            print(
                f"\nWrote {args.path} — {meta['papers']} papers, "
                f"{meta['dimensions']}d {meta['dtype']} vectors, {size:.0f} MB."
            )
            print("Contains no library, ratings, notes or profile.")
        elif args.action == "show":
            meta = snapshot.inspect(Path(args.path))
            for key, value in meta.items():
                print(f"  {key:16} {value}")
        else:
            meta = snapshot.load(conn, cfg, Path(args.path), say, replace=args.replace)
            print(
                f"\nRestored {meta['restored']} papers embedded with "
                f"{meta['embedding_model']}."
            )
            if meta["merged"]:
                print(
                    f"{meta['merged']} of the snapshot's {meta['papers']} were "
                    f"duplicates of each other and merged."
                )
            print("Run 'advisor harvest' to pick up anything published since.")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    return 0


def _update(args: argparse.Namespace) -> int:
    """The scheduled job: harvest, embed within a budget, refresh the feed.

    One command because the three steps are only useful together — harvesting
    without embedding leaves the new papers invisible to retrieval, and
    embedding without a fresh run leaves them out of the feed until you happen
    to press the button.
    """
    from advisor import lock
    from advisor.embed import run as embed_run
    from advisor.ingest import harvest
    from advisor.recommend import pipeline

    cfg = config.load()
    config.ensure_dirs(cfg)
    conn = db.connect(cfg.db_path)

    def say(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    def progress(message: str) -> None:
        if args.verbose and not args.quiet:
            print(message, flush=True)

    failed = False
    try:
        say("harvesting...")
        results = asyncio.run(harvest.harvest_all(conn, cfg, progress))
        for result in results:
            say(f"  {result}")
            for error in result.errors:
                failed = True
                print(f"  ! {error}", file=sys.stderr)

        say(f"embedding (up to {args.max_minutes:g} minutes)...")
        try:
            with lock.exclusive(cfg.embed_lock_path, "an embed pass"):
                done = embed_run.embed_pending(
                    conn, cfg, progress, max_seconds=args.max_minutes * 60
                )
            say(f"  {done} paper(s) encoded")
        except lock.Busy as exc:
            # Not a failure: a manual pass is doing the same work right now.
            say(f"  skipped — {exc}")
        except RuntimeError as exc:
            failed = True
            print(f"  ! {exc}", file=sys.stderr)

        say("recommending...")
        run_id, count = pipeline.run(conn, cfg)
        say(f"  {count} recommendation(s)" if run_id else "  nothing to recommend yet")
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved — rerun to resume.", file=sys.stderr)
        return 130
    finally:
        conn.close()

    return 1 if failed else 0


def _schedule(args: argparse.Namespace) -> int:
    from advisor import schedule

    print(schedule.instructions(args.at))
    return 0


def _reset(args: argparse.Namespace) -> int:
    from advisor import reset

    cfg = config.load()
    conn = db.connect(cfg.db_path)

    tables = reset.PERSONAL_TABLES + (reset.CORPUS_TABLES if args.all else ())
    present = {name: n for name, n in reset.counts(conn, tables).items() if n}

    if not present:
        print("Nothing to clear.")
        conn.close()
        return 0

    scope = "everything" if args.all else "your library, ratings, profile and history"
    print(f"This will permanently delete {scope}:")
    for name, n in present.items():
        print(f"  {n} {name.replace('_', ' ')}")
    if not args.all:
        print("\nThe harvested corpus and its vectors are kept.")
    else:
        print("\nRe-harvesting and re-embedding the corpus takes hours.")

    if not args.yes:
        try:
            if input("\nType 'yes' to continue: ").strip().lower() != "yes":
                print("Cancelled.")
                conn.close()
                return 1
        except EOFError:
            print("Cancelled (no terminal to confirm on; pass --yes).", file=sys.stderr)
            conn.close()
            return 1

    removed = reset.clear_all(conn, cfg) if args.all else reset.clear_personal(conn, cfg)
    conn.close()

    print(f"\nCleared {sum(removed.values())} row(s).")
    return 0


def _status(args: argparse.Namespace) -> int:
    cfg = config.load()
    conn = db.connect(cfg.db_path)

    def count(table: str) -> int:
        return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    sources = conn.execute(
        """SELECT sum(arxiv_id IS NOT NULL) AS arxiv,
                  sum(eprint_id IS NOT NULL) AS eprint,
                  sum(arxiv_id IS NOT NULL AND eprint_id IS NOT NULL) AS both
             FROM papers"""
    ).fetchone()

    papers = count("papers")
    vectors = count("vector_index")

    print(f"database   {cfg.db_path}")
    print(
        f"corpus     {papers} papers "
        f"({sources['arxiv'] or 0} arXiv, {sources['eprint'] or 0} ePrint, "
        f"{sources['both'] or 0} on both)"
    )
    print(f"library    {count('library')} entries")
    from advisor.recommend.feedback import rated_papers

    rated = rated_papers(conn)
    events = count("feedback")
    # The two differ whenever you have changed your mind about a paper, and
    # showing only the row count reads as more opinions than you have given.
    revisions = f" ({events} ratings including changes)" if events != rated else ""
    print(f"feedback   {rated} paper(s) rated{revisions}")
    following = count("followed_authors")
    if following:
        print(f"following  {following} author(s)")

    # Coverage, not just a count: retrieval only ever sees the embedded part,
    # so "22016 vectors" alone does not tell you how much of the corpus is
    # actually reachable.
    if papers:
        pending = papers - vectors
        share = f"{100 * vectors / papers:.0f}% of the corpus"
        print(
            f"embedded   {vectors} vectors ({share}"
            + (f", {pending} pending)" if pending > 0 else ")")
        )
    else:
        print(f"embedded   {vectors} vectors")

    latest_run = conn.execute(
        "SELECT id, created_at, model, n_candidates FROM runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if latest_run:
        unread = conn.execute(
            "SELECT count(*) FROM recommendations WHERE run_id = ? AND action IS NULL",
            (latest_run["id"],),
        ).fetchone()[0]
        ranked_by = latest_run["model"] or "retrieval only"
        print(
            f"last run   {latest_run['created_at'][:16]} "
            f"({unread} unread, {ranked_by})"
        )
    else:
        print("last run   never — run 'advisor recommend'")

    profile_row = conn.execute(
        "SELECT created_at FROM profile_versions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if profile_row:
        print(f"profile    written {profile_row['created_at'][:16]}")

    rows = conn.execute("SELECT source, cursor, last_run FROM harvest_state").fetchall()
    if rows:
        print("\nharvest state:")
        for row in rows:
            print(f"  {row['source']:<18} cursor={row['cursor']}  last={row['last_run']}")
    conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="advisor", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_serve)

    add = sub.add_parser("add", help="add papers you have read, by ID/DOI/URL")
    add.add_argument("ids", nargs="*", help="omit to read from stdin")
    add.add_argument("--status", default="read", choices=["read", "queued", "skipped"])
    add.set_defaults(func=_add)

    status = sub.add_parser("status", help="show what is in the database")
    status.set_defaults(func=_status)

    harvest = sub.add_parser("harvest", help="pull new papers from arXiv and ePrint")
    harvest.add_argument("--reset", action="store_true", help="ignore saved cursors")
    harvest.add_argument("--quiet", action="store_true")
    harvest.set_defaults(func=_harvest)

    embed = sub.add_parser("embed", help="encode papers that have no vector yet")
    embed.add_argument("--reset", action="store_true", help="discard vectors and rebuild")
    embed.add_argument("--quiet", action="store_true")
    embed.set_defaults(func=_embed)

    recommend = sub.add_parser("recommend", help="show what to read next")
    recommend.add_argument("-n", "--limit", type=int, default=10)
    recommend.add_argument(
        "--preview",
        action="store_true",
        help="show without recording, so the papers can come up again",
    )
    recommend.set_defaults(func=_recommend)

    search_cmd = sub.add_parser("search", help="full-text search over the corpus")
    search_cmd.add_argument("query", nargs="+")
    search_cmd.add_argument("-n", "--limit", type=int, default=10)
    search_cmd.set_defaults(func=_search)

    follow = sub.add_parser("follow", help="follow an author, or list who you follow")
    follow.add_argument("names", nargs="*", help="omit to list; quote full names")
    follow.add_argument(
        "--suggest", action="store_true", help="authors you have read more than once"
    )
    follow.set_defaults(func=_follow)

    unfollow = sub.add_parser("unfollow", help="stop following an author")
    unfollow.add_argument("names", nargs="+")
    unfollow.set_defaults(func=_unfollow)

    mcp = sub.add_parser("mcp", help="serve your library to an assistant over MCP")
    mcp.add_argument(
        "--config", action="store_true", help="print the client config and exit"
    )
    mcp.set_defaults(func=_mcp)

    export = sub.add_parser(
        "export", help="write your library, ratings, profile and follows to a file"
    )
    export.add_argument("output", nargs="?", default="-", help="path, or - for stdout")
    export.set_defaults(func=_export)

    import_cmd = sub.add_parser(
        "import", help="load an export into this install (merges by default)"
    )
    import_cmd.add_argument("path", help="file to read, or - for stdin")
    import_cmd.add_argument(
        "--replace", action="store_true", help="discard local data first"
    )
    import_cmd.add_argument("--yes", action="store_true", help="skip the confirmation")
    import_cmd.set_defaults(func=_import)

    snapshot_cmd = sub.add_parser(
        "snapshot", help="share or restore the corpus and its vectors"
    )
    snapshot_cmd.add_argument("action", choices=["fetch", "save", "load", "show"])
    snapshot_cmd.add_argument(
        "path", nargs="?", help="file to read or write; omit when fetching"
    )
    snapshot_cmd.add_argument(
        "--replace", action="store_true", help="on load, discard local vectors first"
    )
    snapshot_cmd.add_argument("--quiet", action="store_true")
    snapshot_cmd.set_defaults(func=_snapshot)

    update = sub.add_parser(
        "update", help="harvest, embed and refresh the feed — the scheduled job"
    )
    update.add_argument(
        "--max-minutes",
        type=float,
        default=30.0,
        help="budget for the embed pass (default: 30)",
    )
    update.add_argument("--quiet", action="store_true")
    update.add_argument("--verbose", action="store_true", help="show per-step progress")
    update.set_defaults(func=_update)

    schedule_cmd = sub.add_parser(
        "schedule", help="print a systemd timer or crontab line for 'advisor update'"
    )
    schedule_cmd.add_argument("--at", default="06:00", help="time of day (default: 06:00)")
    schedule_cmd.set_defaults(func=_schedule)

    reset_cmd = sub.add_parser("reset", help="clear your library, ratings and history")
    reset_cmd.add_argument(
        "--all", action="store_true", help="also discard the corpus and its vectors"
    )
    reset_cmd.add_argument("--yes", action="store_true", help="skip the confirmation")
    reset_cmd.set_defaults(func=_reset)

    profile_cmd = sub.add_parser("profile", help="show your interest profile")
    profile_cmd.add_argument(
        "--brief",
        action="store_true",
        help="print evidence + instructions to hand an assistant of your choice",
    )
    profile_cmd.set_defaults(func=_profile)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
