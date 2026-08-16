"""The corpus and its vectors, as one downloadable file.

Building the corpus from scratch costs an hour of harvesting and about three
hours of CPU encoding it. None of that work is personal — everyone who runs
this ends up with the same vectors from the same public metadata — so doing it
once and sharing the result turns a half-day setup into a download.

Two decisions make the file a sensible size:

*Vectors are stored as float16.* They are L2-normalised unit vectors, so half
precision is ample: measured against the float32 originals over random queries,
the top-10 and top-50 neighbours are identical and the largest cosine error is
3e-5. It halves 223 MB to 112 MB for no measurable loss.

*The metadata is JSON Lines, gzipped*, which takes 134 MB of titles and
abstracts down to 42 MB.

The result is ~154 MB — over GitHub's 100 MB limit for a file in a repository,
comfortably under the 2 GB limit for a release asset, which is where it belongs
anyway: it is a build artifact, not source.

This carries nothing personal. It is the corpus, which is public, and vectors
derived from it — never a library, a rating, a note or a profile. Those move
with :mod:`advisor.portable` instead.
"""

from __future__ import annotations

import gzip
import io
import json
import sqlite3
import tarfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from advisor import db
from advisor.config import Config
from advisor.embed import store
from advisor.models import Paper, now, upsert_paper

FORMAT = "research-advisor/corpus"
VERSION = 1

SOURCES = [
    {
        "name": "arXiv",
        "url": "https://arxiv.org",
        "terms": "Descriptive metadata is dedicated to the public domain under "
        "CC0 1.0. See https://info.arxiv.org/help/api/tou.html",
    },
    {
        "name": "Cryptology ePrint Archive",
        "url": "https://eprint.iacr.org",
        "attribution": "© IACR and the respective authors",
        "terms": "Harvesting is supported subject to attribution being given "
        "to IACR and to the authors. See https://eprint.iacr.org/operations.html",
    },
]

META = "meta.json"
PAPERS = "papers.jsonl.gz"
VECTORS = "vectors.f16.npy"

Progress = Callable[[str], None]


def _silent(message: str) -> None:
    pass


# Columns carried for each paper. Deliberately explicit: a SELECT * would start
# shipping whatever a later migration adds, including personal columns.
COLUMNS = (
    "title",
    "abstract",
    "authors",
    "categories",
    "published_at",
    "updated_at",
    "arxiv_id",
    "eprint_id",
    "doi",
    "url",
    "venue",
    "withdrawn_at",
)


def save(
    conn: sqlite3.Connection, cfg: Config, path: Path, progress: Progress = _silent
) -> dict[str, Any]:
    """Write a corpus snapshot to ``path``. Returns its metadata."""
    matrix = store.load(cfg.vectors_path)
    if matrix is None:
        raise ValueError("nothing to snapshot — no vectors have been built yet")

    model = conn.execute("SELECT model FROM vector_index LIMIT 1").fetchone()
    embedded = conn.execute("SELECT count(*) FROM vector_index").fetchone()[0]

    progress(f"packing {embedded} papers and their vectors")

    # Only papers that actually have a vector: a title with no embedding is
    # something the recipient can harvest for themselves in minutes.
    rows = conn.execute(
        f"""SELECT p.id, v.row, {", ".join("p." + c for c in COLUMNS)}
              FROM papers p JOIN vector_index v ON v.paper_id = p.id
             ORDER BY v.row"""
    )

    # Rows are renumbered densely, 0..n-1. The live matrix is append-only and
    # can hold rows nothing points at any more — a paper that merged into
    # another after both were embedded leaves one behind. Copying the matrix
    # verbatim would ship those orphans and make the file self-inconsistent.
    source_rows: list[int] = []
    lines = io.BytesIO()
    with gzip.GzipFile(fileobj=lines, mode="wb", compresslevel=6, mtime=0) as gz:
        for position, row in enumerate(rows):
            source_rows.append(row["row"])
            record = {"row": position}
            record.update({column: row[column] for column in COLUMNS})
            gz.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
    papers_blob = lines.getvalue()

    progress(f"  metadata {len(papers_blob) / 1048576:.0f} MB")

    usable = [r for r in source_rows if r < matrix.shape[0]]
    if len(usable) != len(source_rows):
        raise ValueError(
            f"{len(source_rows) - len(usable)} paper(s) index vector rows that do "
            f"not exist — the matrix and the index have diverged"
        )

    half = io.BytesIO()
    np.save(half, np.asarray(matrix[usable]).astype(np.float16))
    vectors_blob = half.getvalue()

    progress(f"  vectors  {len(vectors_blob) / 1048576:.0f} MB")

    meta = {
        "format": FORMAT,
        "version": VERSION,
        "created_at": now(),
        "embedding_model": model["model"] if model else cfg.embedding_model,
        "papers": len(source_rows),
        "dimensions": int(matrix.shape[1]),
        "dtype": "float16",
        # Carried inside the file rather than left in a README nobody who
        # downloads a tarball will read. IACR permits harvesting on condition
        # that attribution is given, so the condition travels with the data.
        "sources": SOURCES,
    }

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for name, blob in (
            (META, json.dumps(meta, indent=2).encode("utf-8")),
            (PAPERS, papers_blob),
            (VECTORS, vectors_blob),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(blob))

    return meta


DEFAULT_REPO = "fenwickyduck/research-advisor"


def fetch(destination: Path, repo: str = DEFAULT_REPO, progress: Progress = _silent) -> Path:
    """Download the newest published snapshot, returning where it landed.

    Tries an anonymous HTTPS download first, which is all a public repository
    needs and keeps the dependency list at nothing. A private one answers 404
    to that, so the GitHub CLI is used instead — it already holds the reader's
    own credentials, which is why none have to be embedded here.
    """
    destination = Path(destination).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    progress(f"looking for the newest snapshot in {repo}")

    asset = _public_asset(repo)
    if asset is not None:
        name, url = asset
        target = destination / name
        progress(f"downloading {name}")
        _stream(url, target, progress)
        return target

    return _private_asset(destination, repo, progress)


def _public_asset(repo: str) -> tuple[str, str] | None:
    """(filename, url) of the newest .tar asset, if the repo is readable anonymously."""
    import httpx

    try:
        response = httpx.get(
            f"https://api.github.com/repos/{repo}/releases/latest",
            headers={"Accept": "application/vnd.github+json"},
            timeout=30.0,
            follow_redirects=True,
        )
        if response.status_code != 200:
            return None
        for asset in response.json().get("assets", []):
            if asset["name"].endswith(".tar"):
                return asset["name"], asset["browser_download_url"]
    except Exception:
        return None
    return None


def _stream(url: str, target: Path, progress: Progress) -> None:
    import httpx

    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        seen = 0
        step = 25 * 1024 * 1024
        next_report = step
        with target.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
                seen += len(chunk)
                if seen >= next_report:
                    of = f" of {total / 1048576:.0f}" if total else ""
                    progress(f"  {seen / 1048576:.0f}{of} MB")
                    next_report += step


def _private_asset(destination: Path, repo: str, progress: Progress) -> Path:
    import shutil
    import subprocess

    if shutil.which("gh") is None:
        raise ValueError(
            f"could not download the snapshot from {repo}.\n"
            "If the repository is private you need the GitHub CLI: install it "
            "from https://cli.github.com, run 'gh auth login', and try again.\n"
            f"Otherwise download the asset by hand from\n"
            f"  https://github.com/{repo}/releases\n"
            "and pass it to 'advisor snapshot load'."
        )

    progress("not readable anonymously — using the GitHub CLI")
    result = subprocess.run(
        ["gh", "release", "download", "--repo", repo, "--pattern", "*.tar",
         "--dir", str(destination), "--clobber"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            f"could not download from {repo}: {result.stderr.strip() or 'unknown error'}\n"
            "If this says authentication, run 'gh auth login' first."
        )

    downloaded = sorted(destination.glob("*.tar"), key=lambda p: p.stat().st_mtime)
    if not downloaded:
        raise ValueError(f"no snapshot asset found in the releases of {repo}")
    return downloaded[-1]


def inspect(path: Path) -> dict[str, Any]:
    """Read a snapshot's metadata without unpacking the rest of it."""
    with tarfile.open(Path(path).expanduser(), "r") as tar:
        handle = tar.extractfile(META)
        if handle is None:
            raise ValueError("not a corpus snapshot — no metadata")
        meta = json.loads(handle.read().decode("utf-8"))

    if meta.get("format") != FORMAT:
        raise ValueError(f"not a corpus snapshot (format={meta.get('format')!r})")
    if meta.get("version", 0) > VERSION:
        raise ValueError(
            f"snapshot is version {meta['version']}, this install understands {VERSION}"
        )
    return meta


def load(
    conn: sqlite3.Connection,
    cfg: Config,
    path: Path,
    progress: Progress = _silent,
    replace: bool = False,
) -> dict[str, Any]:
    """Restore a corpus snapshot. Refuses to merge into existing vectors.

    Merging two vector sets means reconciling two row orderings against one
    append-only matrix, and getting it subtly wrong yields recommendations
    computed from the wrong rows — silently. Refusing is the honest option;
    ``replace`` is the explicit way through.
    """
    meta = inspect(path)

    if meta["embedding_model"] != cfg.embedding_model:
        raise ValueError(
            f"snapshot was built with {meta['embedding_model']!r}, this install "
            f"uses {cfg.embedding_model!r}. Vectors from different models are "
            f"not comparable."
        )

    existing = conn.execute("SELECT count(*) FROM vector_index").fetchone()[0]
    if existing and not replace:
        raise ValueError(
            f"this install already has {existing} vectors. Loading a snapshot "
            f"over them needs --replace, which discards the local ones first."
        )

    if replace and existing:
        progress(f"discarding {existing} local vectors")
        store.reset(conn, cfg.vectors_path)

    with tarfile.open(Path(path).expanduser(), "r") as tar:
        progress(f"restoring {meta['papers']} papers")

        handle = tar.extractfile(PAPERS)
        if handle is None:
            raise ValueError("snapshot has no papers")

        # row -> local paper id. The snapshot's ids are not carried at all;
        # rows are the only join key, and they are internal to the file.
        by_row: dict[int, int] = {}
        started = time.monotonic()
        with gzip.GzipFile(fileobj=handle, mode="rb") as gz:
            with db.transaction(conn):
                for count, line in enumerate(gz, 1):
                    record = json.loads(line)
                    paper_id = upsert_paper(
                        conn,
                        Paper(
                            title=record["title"],
                            abstract=record.get("abstract"),
                            authors=db.json_list(record.get("authors")),
                            categories=db.json_list(record.get("categories")),
                            published_at=record.get("published_at"),
                            updated_at=record.get("updated_at"),
                            arxiv_id=record.get("arxiv_id"),
                            eprint_id=record.get("eprint_id"),
                            doi=record.get("doi"),
                            url=record.get("url"),
                            venue=record.get("venue"),
                        ),
                    )
                    if record.get("withdrawn_at"):
                        conn.execute(
                            "UPDATE papers SET withdrawn_at = ? WHERE id = ?",
                            (record["withdrawn_at"], paper_id),
                        )
                    by_row[record["row"]] = paper_id
                    if count % 10000 == 0:
                        rate = count / max(time.monotonic() - started, 1e-9)
                        progress(f"  {count}/{meta['papers']} ({rate:.0f}/s)")

        progress("restoring vectors")
        handle = tar.extractfile(VECTORS)
        if handle is None:
            raise ValueError("snapshot has no vectors")
        matrix = np.load(io.BytesIO(handle.read()))

    if matrix.shape[0] != len(by_row):
        raise ValueError(
            f"snapshot is inconsistent: {matrix.shape[0]} vectors for "
            f"{len(by_row)} papers"
        )

    # Two snapshot rows can land on one local paper: the corpus contains a few
    # genuine duplicates (one paper posted twice, or on both archives) that the
    # dedupe recognises on the way in. Their spare vector rows go unreferenced,
    # which is harmless — search skips rows with no paper — but the count would
    # otherwise differ from the snapshot's for no visible reason.
    merged = len(by_row) - len(set(by_row.values()))
    if merged:
        progress(f"  {merged} duplicate paper(s) merged into existing records")

    # Back to float32 on write: it is what the matrix is stored and searched in,
    # and the precision was never the reason for shipping half.
    start = store.append(cfg.vectors_path, matrix.astype(np.float32))
    store.record_rows(
        conn,
        [by_row[row] for row in sorted(by_row)],
        start,
        meta["embedding_model"],
    )
    conn.commit()

    return dict(meta, merged=merged, restored=len(set(by_row.values())))
