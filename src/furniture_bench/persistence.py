"""Validated single-file publication; concurrent writers need a separate policy."""
from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


class AtomicCommitError(OSError):
    """The new file is complete, but post-publication durability was not confirmed."""

    committed = True


def _sync_directory(directory):
    if os.name == "nt":
        return  # Windows does not expose POSIX directory fsync through os.open.
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path, writer, *, validator=None, overwrite=True):
    """Write/fsync/validate a unique sibling, then publish without a partial target.

    Before publication, failure leaves any existing target unchanged. After
    publication, directory-fsync failure raises AtomicCommitError(committed=True).
    Only this call's temporary file is cleaned up, never unrelated orphan files.
    overwrite=False uses a no-clobber hard link, not a racy existence check.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("atomic destinations must not be symbolic links")
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            if mode is not None:
                os.fchmod(stream.fileno(), mode)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if validator is not None:
            validator(temporary)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        try:
            _sync_directory(path.parent)
        except OSError as exc:
            raise AtomicCommitError("File committed; directory durability not confirmed") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_json(path, value, *, ensure_ascii=False, overwrite=True):
    # Serialize strictly before touching the destination or creating a temporary.
    payload = (json.dumps(value, indent=2, ensure_ascii=ensure_ascii, allow_nan=False) + "\n").encode("utf-8")

    def validate(temporary):
        with temporary.open(encoding="utf-8") as stream:
            json.load(stream)

    atomic_write(path, lambda stream: stream.write(payload), validator=validate, overwrite=overwrite)


@contextmanager
def artifact_lock(path, *, timeout=30.):
    """POSIX process lock; crash releases ownership, stable lock inode is retained.

    Only cooperating writers are serialized. Never unlink a lock file on exit:
    doing so could let later writers lock a different inode at the same path.
    """
    if os.name != 'posix':
        raise RuntimeError('Artifact locking requires POSIX/WSL; no unsafe fallback')
    import fcntl

    if not isinstance(timeout, (int, float)) or not 0 <= timeout < float('inf'):
        raise ValueError('Finite nonnegative lock timeout required')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Lock paths must not be symbolic links')
    with path.open('a+b') as stream:
        deadline = time.monotonic()+timeout
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Artifact is owned by another writer')
                time.sleep(.01)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
