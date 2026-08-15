"""Turning papers into vectors.

SPECTER is used by default: it is trained on the citation graph of scientific
papers rather than on general web text, and it expects exactly the input we
have — title and abstract joined by the tokenizer's separator. That framing is
why a paper's neighbours come out topically sensible rather than merely sharing
vocabulary.

The model is loaded lazily and once, since importing torch is slow and the web
app should not pay for it on a request that never embeds anything.
"""

from __future__ import annotations

import os
import sqlite3
import warnings
from collections.abc import Callable, Iterator

import numpy as np

from advisor.config import Config
from advisor.models import Paper

Progress = Callable[[str], None]

_model = None
_model_name: str | None = None


def load_model(
    name: str,
    backend: str = "onnx",
    onnx_file: str = "onnx/model_qint8_avx512_vnni.onnx",
    threads: int | None = None,
):
    """Load (and cache) the sentence-transformers model.

    Two things make the difference between minutes and hours on CPU:

    * **Thread count.** Torch's default is not the core count, and the gap is
      not marginal — measured at 6 papers/s versus 39 on the same machine.
    * **Backend.** An int8-quantised ONNX graph measured 4.5x faster than torch
      fp32 (2.4 -> 10.9 papers/s), for vectors agreeing at 0.988 mean cosine
      and ~85% overlap in top-10 neighbours. Since the shortlist is re-ranked
      afterwards, that is a good trade; ``backend="torch"`` opts out.

    ONNX is a soft dependency, so an unavailable runtime falls back to torch
    with a warning rather than failing the run.
    """
    global _model, _model_name

    key = f"{name}::{backend}"
    if _model is not None and _model_name == key:
        return _model

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        # The likeliest first-run failure: base install, corpus harvested, and
        # then a traceback from inside a library the user never asked for.
        raise RuntimeError(
            "The encoder is not installed, so papers cannot be embedded. Run:\n"
            "  pip install -e '.[embed]'\n"
            "Harvesting, searching and the web app work without it; "
            "recommendations do not."
        ) from exc

    torch.set_num_threads(threads or os.cpu_count() or 1)

    model = None
    if backend == "onnx":
        try:
            model = SentenceTransformer(
                name, backend="onnx", model_kwargs={"file_name": onnx_file}
            )
        except Exception as exc:
            warnings.warn(
                f"ONNX backend unavailable ({type(exc).__name__}); falling back to "
                f"torch, which is roughly 4x slower. Install the 'embed' extra to "
                f"enable it.",
                stacklevel=2,
            )

    _model = model if model is not None else SentenceTransformer(name)
    _model_name = key
    return _model


def paper_text(paper: Paper, separator: str = "[SEP]") -> str:
    """The string SPECTER expects: title, separator, abstract.

    A paper with no abstract still embeds usefully from its title alone, which
    matters because ePrint records occasionally omit one.
    """
    title = (paper.title or "").strip()
    abstract = (paper.abstract or "").strip()
    return f"{title} {separator} {abstract}".strip() if abstract else title


def encode(texts: list[str], cfg: Config, progress: Progress | None = None) -> np.ndarray:
    model = load_model(cfg.embedding_model, cfg.embedding_backend, cfg.embedding_onnx_file)
    return model.encode(
        texts,
        batch_size=cfg.embedding_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def pending(conn: sqlite3.Connection, batch_size: int = 512) -> Iterator[list[Paper]]:
    """Yield batches of papers that have no vector yet, most useful first.

    Order is the difference between a tool that works in minutes and one that
    works in hours. Encoding the corpus takes a while on CPU, but you do not
    need all of it — you need the parts a recommendation can come from:

    1. **Your library**, because preference vectors are built from it. Without
       these there are no recommendations at all, whatever else is embedded.
    2. **A spread across years**, newest first *within* each year — the newest
       paper of every year, then the second newest of every year, and so on.

    The stratification matters. Taking strictly the newest papers first makes
    the tool usable quickly but skews the candidate pool to whatever is
    currently fashionable: with only the last month embedded, a query about
    lattices is matched against a pool that is almost entirely something else,
    and the results look broken. Round-robin over years keeps the pool
    representative at every point, so partial results are still meaningful
    while the rest fills in.

    Withdrawn papers are skipped entirely — no point spending compute on
    something that must never be recommended.
    """
    while True:
        rows = conn.execute(
            """WITH ranked AS (
                 SELECT p.*,
                        (l.paper_id IS NOT NULL) AS in_library,
                        ROW_NUMBER() OVER (
                          PARTITION BY substr(coalesce(p.published_at, '0000'), 1, 4)
                          ORDER BY p.published_at DESC, p.id DESC
                        ) AS rank_in_year
                   FROM papers p
                   LEFT JOIN vector_index v ON v.paper_id = p.id
                   LEFT JOIN library l ON l.paper_id = p.id
                  WHERE v.paper_id IS NULL AND p.withdrawn_at IS NULL
               )
               SELECT * FROM ranked
                ORDER BY in_library DESC, rank_in_year, published_at DESC
                LIMIT ?""",
            (batch_size,),
        ).fetchall()

        if not rows:
            return
        yield [Paper.from_row(row) for row in rows]


def count_pending(conn: sqlite3.Connection) -> int:
    return conn.execute(
        """SELECT count(*) FROM papers p
            LEFT JOIN vector_index v ON v.paper_id = p.id
            WHERE v.paper_id IS NULL AND p.withdrawn_at IS NULL"""
    ).fetchone()[0]
