"""One embed pass at a time.

Scheduling makes overlap possible for the first time: the nightly job can fire
while you are running ``advisor embed`` by hand, and the two would race. The
race is not benign — ``store.append`` rewrites the whole matrix, so the loser
silently discards whatever the winner had just added, leaving ``vector_index``
pointing at rows that no longer hold what it claims.

An advisory file lock is enough, because both contenders are this program on
this machine. Held by the running process and released even if it is killed,
since the kernel drops flocks when the file descriptor closes.
"""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class Busy(RuntimeError):
    """Another instance holds the lock."""


@contextmanager
def exclusive(path: Path, what: str = "another instance") -> Iterator[None]:
    """Hold an advisory lock on ``path``, or raise :class:`Busy` immediately.

    Non-blocking on purpose: a scheduled run that finds the lock taken should
    report it and let the next night's run pick the work up, rather than
    queueing behind an hours-long manual pass.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise Busy(f"{what} is already running ({path})") from exc
        yield
    finally:
        handle.close()
