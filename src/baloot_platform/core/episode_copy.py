"""Governed presentation copy used by generated episode contracts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def governed_episode_text(
    key: str,
    *,
    copy: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    **values: Any,
) -> str:
    """Resolve one non-empty string from the active video copy authority."""

    if copy is None:
        from baloot_platform.config import build_bundle

        copy = build_bundle(root).video["localizations"]
    text = copy.get(key)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"missing governed episode copy: {key}")
    try:
        return text.format(**values)
    except KeyError as exc:
        raise ValueError(f"missing governed episode copy value for {key}: {exc.args[0]}") from exc
