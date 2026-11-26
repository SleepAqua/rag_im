"""One scheduled pass: take the lock, fetch, report, return an exit code.

The lock is advisory (``flock``) but sufficient for the real failure mode: a
slow fetch overlapping the next cron or systemd timer. The second process
exits with code 3 instead of interleaving writes.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:  # POSIX; the project targets Linux/macOS
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


class RunLocked(RuntimeError):
    """Another run already holds the lock."""


@contextmanager
def run_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    if fcntl is None:  # pragma: no cover
        try:
            yield
        finally:
            handle.close()
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RunLocked(f"another rag-im run holds {path}") from exc
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
