#!/usr/bin/env python3
# ruff: noqa: E402
"""Run an honest dirty-worktree pilot for the exact finalist policy.

This runner never produces promotion evidence. It binds the aggregate pilot to
the current file contents so local iteration can expose regressions before the
clean permanent-holdout release gate is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from baloot_platform.benchmark import canonical_sha256, load_candidate_jsonl
from baloot_platform.config import platform_root
from baloot_platform.agents.qualification import workspace_source_paths

from scripts.benchmark_strength import run_preanalysis
from scripts.generate_holdout_candidates import DEFAULT_TRUSTED, SYNTHETIC_PURPOSE, load_replay_recipes


DEFAULT_OUTPUT = Path(".local-state/evidence/finalist-preanalysis.json")
DEFAULT_MANIFEST = Path(".local-state/evidence/finalist-source-manifest.json")
DEFAULT_CANDIDATES = Path("private-inputs/strength-candidates.jsonl")
DEFAULT_EXACT = Path("release/exact-labels.json")
_SOURCE_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".json"})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_workspace_manifest(
    root: Path,
    *,
    extra_paths: tuple[Path, ...],
) -> dict[str, Any]:
    coverage_roots = (root / "src" / "baloot_platform", root / "configs")
    explicit_paths = (
        root / "scripts" / "benchmark_strength.py",
        root / "scripts" / "run_local_finalist_preanalysis.py",
    )
    candidates = set(extra_paths)
    candidates.update(workspace_source_paths(coverage_roots, _SOURCE_SUFFIXES))
    candidates.update(explicit_paths)
    files = []
    for path in sorted(candidates, key=lambda item: item.as_posix()):
        resolved = path.resolve()
        if not resolved.is_file() or root.resolve() not in resolved.parents:
            raise ValueError(f"workspace manifest path is invalid: {path}")
        files.append(
            {
                "path": resolved.relative_to(root.resolve()).as_posix(),
                "bytes": resolved.stat().st_size,
                "sha256": _file_sha256(resolved),
            }
        )
    try:
        git_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        git_revision = None
    if (
        git_revision is not None
        and git_revision.returncode == 0
        and git_revision.stdout.strip()
    ):
        source_revision = git_revision.stdout.strip()
        source_revision_kind = "git_head"
    else:
        source_revision = f"workspace:{canonical_sha256({'files': files})}"
        source_revision_kind = "workspace_content"
    value: dict[str, Any] = {
        "schemaVersion": 2,
        "kind": "local_finalist_preanalysis_source_manifest",
        "sourceRevision": source_revision,
        "sourceRevisionKind": source_revision_kind,
        "coverage": {
            "roots": [path.relative_to(root).as_posix() for path in coverage_roots],
            "suffixes": sorted(_SOURCE_SUFFIXES),
            "explicitPaths": [
                path.relative_to(root).as_posix() for path in explicit_paths
            ],
            "inputPaths": [path.relative_to(root).as_posix() for path in extra_paths],
        },
        "files": files,
        "proofBoundary": (
            "Content-bound local dirty-worktree pilot identity. This is not a "
            "Git tree, clean release source, permanent holdout, professional "
            "review, or promotion attestation."
        ),
    }
    value["sha256"] = canonical_sha256(value)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--trusted", type=Path)
    parser.add_argument("--exact-labels", type=Path)
    parser.add_argument("--baseline", default="strong")
    parser.add_argument(
        "--comparison-policy",
        action="append",
        help=(
            "Additional governed policy to measure on the identical exact-label "
            "cohort. Repeat the option to compare more than one policy."
        ),
    )
    args = parser.parse_args()
    root = platform_root()
    candidate_path = (args.candidates or root / DEFAULT_CANDIDATES).resolve()
    candidates = load_candidate_jsonl(candidate_path)
    trusted_path = args.trusted.resolve() if args.trusted is not None else None
    if trusted_path is None and candidates.get("purpose") != SYNTHETIC_PURPOSE:
        trusted_path = root / DEFAULT_TRUSTED
    trusted_input = trusted_path or root / "scripts/generate_holdout_candidates.py"
    exact_path = (args.exact_labels or root / DEFAULT_EXACT).resolve()
    output = (args.output or root / DEFAULT_OUTPUT).resolve()
    manifest_path = (args.manifest or root / DEFAULT_MANIFEST).resolve()
    manifest = build_workspace_manifest(
        root,
        extra_paths=(candidate_path, trusted_input, exact_path),
    )
    comparison_policies = tuple(args.comparison_policy or ("foundry_balanced",))
    policy_ids = tuple(
        dict.fromkeys(("finalist_candidate", *comparison_policies, str(args.baseline)))
    )
    result = run_preanalysis(
        candidates,
        load_replay_recipes(candidates, root=root, path=trusted_path),
        json.loads(exact_path.read_text(encoding="utf-8")),
        root=root,
        policy_ids=policy_ids,
        source_revision=str(manifest["sourceRevision"]),
        source_tree=None,
        workspace_manifest_sha256=str(manifest["sha256"]),
    )
    result["localPilot"] = {
        "sourceManifestPath": manifest_path.relative_to(root).as_posix(),
        "sourceManifestSha256": manifest["sha256"],
        "candidatePolicy": "finalist_candidate",
        "comparisonPolicies": list(comparison_policies),
        "baselinePolicy": str(args.baseline),
        "qualificationEligible": False,
    }
    result["resultHash"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "resultHash"}
    )
    _write_json(manifest_path, manifest)
    _write_json(output, result)
    print(output)
    print(manifest_path)


if __name__ == "__main__":
    main()
