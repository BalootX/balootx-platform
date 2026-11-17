#!/usr/bin/env python3
# ruff: noqa: E402
"""Benchmark paired duplicate games from frozen inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY_ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from baloot_platform.benchmark.paired_games import (
    run_paired_duplicate_rounds,
    serialize_paired_evidence,
)
from baloot_platform.config import platform_root


DEFAULT_OUTPUT = Path(
    ".release-runtime/benchmark/paired-duplicate-rounds.json"
)


def _git_identity(root: Path) -> tuple[str, str]:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "paired benchmark requires a clean source worktree"
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return revision, tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate")
    parser.add_argument("--baseline")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=31_000_000)
    parser.add_argument(
        "--candidate-evaluation", type=Path,
        default=Path("release/agent-qualification/permanent-holdout-evaluation.json"),
    )
    parser.add_argument(
        "--baseline-evaluation", type=Path,
        default=Path("release/agent-qualification/permanent-holdout-baseline.json"),
    )
    parser.add_argument("--maximum-seconds", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = platform_root()
    from baloot_platform.config import build_bundle
    from baloot_platform.benchmark import canonical_sha256
    comparison = build_bundle(root).component("benchmark")["paired_comparison"]
    governed_candidate = str(comparison["candidate_profile_id"])
    governed_baseline = str(comparison["baseline_profile_id"])
    candidate_id = str(args.candidate or governed_candidate)
    baseline_id = str(args.baseline or governed_baseline)
    if candidate_id != governed_candidate or baseline_id != governed_baseline:
        parser.error("paired policies must match the governed candidate and baseline")
    candidate_evaluation = json.loads(
        (root / args.candidate_evaluation).resolve().read_text(encoding="utf-8")
    )
    baseline_evaluation = json.loads(
        (root / args.baseline_evaluation).resolve().read_text(encoding="utf-8")
    )
    for expected_id, value, label in (
        (candidate_id, candidate_evaluation, "candidate"),
        (baseline_id, baseline_evaluation, "baseline"),
    ):
        result_hash = str(value.get("resultHash") or "")
        payload = {key: item for key, item in value.items() if key != "resultHash"}
        if value.get("modelId") != expected_id or result_hash != canonical_sha256(payload):
            parser.error(f"{label} permanent-holdout evidence is invalid")
    source_revision, source_tree = _git_identity(root)
    value = run_paired_duplicate_rounds(
        root=root,
        candidate_id=candidate_id,
        baseline_id=baseline_id,
        pair_count=args.pairs,
        seed_start=args.seed_start,
        source_revision=source_revision,
        source_tree=source_tree,
        candidate_expected_regret=float(candidate_evaluation["expectedRegret"]),
        baseline_expected_regret=float(baseline_evaluation["expectedRegret"]),
        maximum_seconds=args.maximum_seconds,
        progress=lambda completed, total: (
            print(f"completed {completed}/{total} pairs")
            if completed == total or completed % 10 == 0
            else None
        ),
    )
    value["permanentHoldoutEvidence"] = {
        "candidate": candidate_evaluation["resultHash"],
        "baseline": baseline_evaluation["resultHash"],
    }
    value["evidenceSha256"] = canonical_sha256({
        key: item for key, item in value.items() if key != "evidenceSha256"
    })
    output = args.output or root / DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        serialize_paired_evidence(value),
        encoding="utf-8",
        newline="\n",
    )
    print(output)


if __name__ == "__main__":
    main()
