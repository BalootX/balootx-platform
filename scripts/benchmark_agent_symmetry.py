#!/usr/bin/env python3
# ruff: noqa: E402
"""Produce clean-source agent symmetry qualification evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY_ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.agents.symmetry_qualification import evaluate_agent_symmetry
from baloot_platform.benchmark import load_candidate_jsonl, validate_trusted_sidecar
from baloot_platform.config import build_bundle, platform_root
from baloot_platform.core.engine import BalootEngine


def _git_identity(root: Path) -> tuple[str, str]:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("agent symmetry benchmark requires a clean source worktree")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return revision, tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--trusted", type=Path, default=Path(".release-runtime/strength-holdout/trusted-replay-recipes.json"))
    parser.add_argument("--exact-labels", type=Path, default=Path("release/exact-labels.json"))
    parser.add_argument("--output", type=Path, default=Path("release/agent-qualification/symmetry.json"))
    args = parser.parse_args()
    root = platform_root()
    revision, tree = _git_identity(root)
    bundle = build_bundle(root)
    policy = bundle.component("benchmark")["paired_comparison"]
    symmetry = bundle.component("benchmark")["symmetry"]
    candidates = load_candidate_jsonl((root / args.candidates).resolve())
    trusted = json.loads((root / args.trusted).resolve().read_text(encoding="utf-8"))
    validate_trusted_sidecar(candidates, trusted)
    exact = json.loads((root / args.exact_labels).resolve().read_text(encoding="utf-8"))
    profile = AgentProfileRegistry.from_bundle(root).get(str(policy["candidate_profile_id"]))
    value = evaluate_agent_symmetry(
        engine=BalootEngine(bundle.rules), profile=profile,
        convention=bundle.convention, candidates=candidates,
        trusted=trusted, exact=exact,
        maximum_cases=int(symmetry["maximum_cases"]),
        tolerance=float(symmetry["tolerance"]),
    )
    value["sourceRevision"] = revision
    value["sourceTree"] = tree
    from baloot_platform.benchmark import canonical_sha256
    value["resultHash"] = canonical_sha256({
        key: item for key, item in value.items() if key != "resultHash"
    })
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(output)


if __name__ == "__main__":
    main()
