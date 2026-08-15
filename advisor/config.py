"""Configuration and filesystem paths.

Settings live in ``~/.config/advisor/config.toml``; anything absent falls back to
the defaults below. Data (database, vectors) lives in ``~/.local/share/advisor``
so the repo stays clean and a ``git clean`` never eats your library.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path


def _xdg(var: str, default: str) -> Path:
    return Path(os.environ.get(var) or Path.home() / default)


CONFIG_DIR = _xdg("XDG_CONFIG_HOME", ".config") / "advisor"
DATA_DIR = _xdg("XDG_DATA_HOME", ".local/share") / "advisor"
CONFIG_FILE = CONFIG_DIR / "config.toml"


@dataclass(frozen=True)
class Config:
    data_dir: Path = DATA_DIR

    # --- corpus ---
    # arXiv categories to harvest. cs.CR is cryptography and security.
    arxiv_categories: tuple[str, ...] = ("cs.CR",)
    harvest_eprint: bool = True
    # Earliest date to backfill from. ePrint's own earliest datestamp is 1996-01-01.
    backfill_from: str = "1996-01-01"

    # --- embeddings (phase 3) ---
    embedding_model: str = "sentence-transformers/allenai-specter"
    embedding_batch_size: int = 64
    # "onnx" runs an int8-quantised graph: measured 4.5x faster than torch fp32
    # on CPU (2.4 -> 10.9 papers/s), for a mean vector agreement of 0.988 and
    # ~85% overlap in top-10 neighbours. Set to "torch" for exact fp32 vectors.
    embedding_backend: str = "onnx"
    embedding_onnx_file: str = "onnx/model_qint8_avx512_vnni.onnx"

    # --- recommendations ---
    n_recommendations: int = 10
    n_candidates: int = 60  # what survives MMR and reaches Claude
    n_retrieve_per_cluster: int = 200
    n_clusters: int = 4  # interest clusters over your liked papers
    rocchio_alpha: float = 1.0
    rocchio_beta: float = 0.3
    # What one "More of" line is worth against one library paper when your
    # interests are clustered. Higher, because you typed it on purpose. Below
    # n_clusters positives there is no clustering to weight and every interest
    # already gets its own query vector, so this only bites as a library grows.
    profile_weight: float = 3.0
    recency_boost: float = 0.05  # added to cosine score for papers < 18 months old

    @property
    def db_path(self) -> Path:
        return self.data_dir / "advisor.db"

    @property
    def vectors_path(self) -> Path:
        return self.data_dir / "vectors.npy"

    @property
    def embed_lock_path(self) -> Path:
        """Guards the embed pass against a scheduled run overlapping a manual one."""
        return self.data_dir / "embed.lock"


_SCALARS = {
    "arxiv_categories": tuple,
    "backfill_from": str,
    "embedding_model": str,
}


def load(path: Path | None = None) -> Config:
    """Read the config file, falling back to defaults for anything unset."""
    path = path or CONFIG_FILE
    if not path.exists():
        return Config()

    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    known = {f.name for f in Config.__dataclass_fields__.values()}
    overrides: dict[str, object] = {}
    for key, value in raw.items():
        if key not in known:
            continue
        if key == "arxiv_categories":
            value = tuple(value)
        elif key == "data_dir":
            value = Path(value).expanduser()
        overrides[key] = value

    return replace(Config(), **overrides)


def ensure_dirs(cfg: Config) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
