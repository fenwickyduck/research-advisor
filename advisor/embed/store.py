"""Vector storage: one ``.npy`` matrix plus a paper_id -> row map in SQLite.

At this corpus size a vector database would be pure overhead. 76,000 papers at
768 float32 dimensions is ~220 MB, and a brute-force cosine pass over that is
tens of milliseconds — comfortably faster than the round trip to Claude that
follows it. Vectors are L2-normalised on write, so cosine similarity is just a
matrix-vector product.

The matrix is append-only and its row order is authoritative; ``vector_index``
maps paper ids onto rows. Changing the embedding model invalidates both, which
:func:`model_changed` detects.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

DTYPE = np.float32


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit length so a dot product is the cosine similarity."""
    matrix = np.asarray(matrix, dtype=DTYPE)
    if matrix.ndim == 1:
        norm = np.linalg.norm(matrix)
        return matrix / norm if norm else matrix

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Guard the zero vector: an empty abstract would otherwise divide by zero.
    np.maximum(norms, 1e-12, out=norms)
    return matrix / norms


def load(path: Path) -> np.ndarray | None:
    """Memory-map the matrix, so a large corpus is not read into RAM up front."""
    if not path.exists():
        return None
    return np.load(path, mmap_mode="r")


def append(path: Path, vectors: np.ndarray) -> int:
    """Append rows, returning the row index the first new vector landed at.

    ``np.save`` cannot append, so this rewrites the file. That is acceptable
    because it happens once per harvest, not once per paper.
    """
    vectors = normalize(vectors)

    existing = load(path)
    if existing is None:
        start = 0
        combined = vectors
    else:
        start = existing.shape[0]
        if existing.shape[1] != vectors.shape[1]:
            raise ValueError(
                f"dimension mismatch: matrix has {existing.shape[1]}, "
                f"new vectors have {vectors.shape[1]}"
            )
        combined = np.vstack([np.asarray(existing), vectors])

    path.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the target and rename, so an interrupted save cannot leave a
    # truncated matrix that no longer matches vector_index.
    #
    # Written through an open handle deliberately: given a *path* that does not
    # end in .npy, np.save silently appends the extension, so the temp file
    # would land at "vectors.npy.tmp.npy" and the rename below would fail.
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, combined)
    tmp.replace(path)

    return start


def record_rows(
    conn: sqlite3.Connection, paper_ids: list[int], start_row: int, model: str
) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO vector_index (paper_id, row, model) VALUES (?,?,?)",
        [(paper_id, start_row + offset, model) for offset, paper_id in enumerate(paper_ids)],
    )


def embedded_ids(conn: sqlite3.Connection) -> set[int]:
    return {row["paper_id"] for row in conn.execute("SELECT paper_id FROM vector_index")}


def row_for(conn: sqlite3.Connection, paper_id: int) -> int | None:
    row = conn.execute(
        "SELECT row FROM vector_index WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    return row["row"] if row else None


def rows_for(conn: sqlite3.Connection, paper_ids: list[int]) -> dict[int, int]:
    if not paper_ids:
        return {}
    placeholders = ",".join("?" * len(paper_ids))
    return {
        row["paper_id"]: row["row"]
        for row in conn.execute(
            f"SELECT paper_id, row FROM vector_index WHERE paper_id IN ({placeholders})",
            paper_ids,
        )
    }


def ids_by_row(conn: sqlite3.Connection) -> np.ndarray:
    """Paper id for each row, as an array aligned with the matrix."""
    rows = conn.execute("SELECT paper_id, row FROM vector_index ORDER BY row").fetchall()
    ids = np.full(len(rows), -1, dtype=np.int64)
    for row in rows:
        if row["row"] < len(ids):
            ids[row["row"]] = row["paper_id"]
    return ids


def model_changed(conn: sqlite3.Connection, model: str) -> str | None:
    """Return the previously used model if it differs from ``model``.

    Vectors from two different models are not comparable, so mixing them would
    produce silently meaningless recommendations rather than an error.
    """
    row = conn.execute("SELECT model FROM vector_index LIMIT 1").fetchone()
    if row and row["model"] != model:
        return row["model"]
    return None


def reset(conn: sqlite3.Connection, path: Path) -> None:
    conn.execute("DELETE FROM vector_index")
    path.unlink(missing_ok=True)
