"""Small, durable filesystem writes shared by governed artifact producers."""

from __future__ import annotations

from pathlib import Path
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager


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


def _replace(source: Path, destination: Path) -> None:
    try:
        source.replace(destination)
    except OSError as exc:
        raise OSError(
            exc.errno,
            "Could not atomically replace the output file. "
            "Close any program using it and check directory permissions.",
            None,
            getattr(exc, "winerror", None),
        ) from None


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
