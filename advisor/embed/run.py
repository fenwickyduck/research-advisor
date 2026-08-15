"""The embed pass: turn every un-embedded paper into a vector.

Written to be interruptible. The corpus takes a while to encode on CPU, so
progress is committed batch by batch — stopping and rerunning picks up exactly
where it left off rather than starting the hour again.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable

from advisor import db
from advisor.config import Config
from advisor.embed import encoder, store

Progress = Callable[[str], None]


def _silent(message: str) -> None:
    pass


def _format_eta(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def embed_pending(
    conn: sqlite3.Connection,
    cfg: Config,
    progress: Progress = _silent,
    batch_size: int = 512,
    max_seconds: float | None = None,
) -> int:
    """Embed everything without a vector yet. Returns how many were encoded.

    ``max_seconds`` bounds the work rather than the result: the pass stops at
    the first batch boundary past the budget and leaves the rest for next time.
    A scheduled run needs that — the first one after a backfill would otherwise
    hold the machine for hours, and a nightly job that overruns its night is
    worse than one that makes steady partial progress.
    """
    previous = store.model_changed(conn, cfg.embedding_model)
    if previous:
        raise RuntimeError(
            f"stored vectors were built with '{previous}' but the configured model "
            f"is '{cfg.embedding_model}'. Vectors from different models are not "
            f"comparable — run 'advisor embed --reset' to rebuild."
        )

    total = encoder.count_pending(conn)
    if not total:
        return 0

    progress(f"embedding {total} papers with {cfg.embedding_model}")

    done = 0
    started = time.monotonic()

    for papers in encoder.pending(conn, batch_size=batch_size):
        texts = [encoder.paper_text(paper) for paper in papers]
        vectors = encoder.encode(texts, cfg)

        # Matrix first, then the index. If the process dies between the two the
        # matrix has orphan rows, which is harmless; the reverse would leave
        # vector_index pointing at rows that do not exist.
        start_row = store.append(cfg.vectors_path, vectors)
        with db.transaction(conn):
            store.record_rows(
                conn, [p.id for p in papers], start_row, cfg.embedding_model
            )

        done += len(papers)
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed else 0
        eta = (total - done) / rate if rate else 0
        progress(f"  {done}/{total}  ({rate:.0f}/s, ~{_format_eta(eta)} left)")

        if max_seconds is not None and elapsed >= max_seconds and done < total:
            progress(
                f"  stopping at the {_format_eta(max_seconds)} budget "
                f"— {total - done} left for the next run"
            )
            break

    return done
