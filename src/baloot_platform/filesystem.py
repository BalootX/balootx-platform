"""Small, durable filesystem writes shared by governed artifact producers."""

from __future__ import annotations

import errno
import os
from pathlib import Path
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager


_WINDOWS_REPLACE_ATTEMPTS = 40
_WINDOWS_REPLACE_INITIAL_BACKOFF_SECONDS = 0.002
_WINDOWS_REPLACE_MAXIMUM_BACKOFF_SECONDS = 0.025


@contextmanager
def staged_render_output(destination: Path) -> Iterator[Path]:
    """Let a renderer claim a fresh directory before moving its finished files."""
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".baloot-render-", dir=destination
    ) as temporary:
        rendered = Path(temporary) / "rendered"
        yield rendered
        for artifact in rendered.iterdir():
            artifact.replace(destination / artifact.name)


def _retryable_replace_error(exc: OSError) -> bool:
    return os.name == "nt" and (
        getattr(exc, "winerror", None) in {5, 32}
        or exc.errno == errno.EINVAL
    )


def _replace(source: Path, destination: Path) -> None:
    backoff = _WINDOWS_REPLACE_INITIAL_BACKOFF_SECONDS
    for attempt in range(_WINDOWS_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if (
                not _retryable_replace_error(exc)
                or attempt + 1 >= _WINDOWS_REPLACE_ATTEMPTS
            ):
                raise
            time.sleep(backoff)
            backoff = min(
                _WINDOWS_REPLACE_MAXIMUM_BACKOFF_SECONDS,
                backoff * 2,
            )


def atomic_write_text(
    path: Path,
    value: str,
    *,
    encoding: str = "utf-8",
    newline: str | None = "\n",
) -> None:
    """Write a complete sibling file, then atomically publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            encoding=encoding,
            newline=newline,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
        _replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
