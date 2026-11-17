"""Git-visible product inputs used by release evidence and readiness."""

from __future__ import annotations

SOURCE_SCOPE = "git_visible_product_inputs_v1"
SOURCE_EXCLUDED_PATHS = (
    "build/",
    "dist/",
    "output/",
    "outputs/",
    "release/",
)


def source_pathspecs() -> tuple[str, ...]:
    return tuple(f":(exclude){path}**" for path in SOURCE_EXCLUDED_PATHS)
