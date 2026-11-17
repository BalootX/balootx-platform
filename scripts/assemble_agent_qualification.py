#!/usr/bin/env python3
# ruff: noqa: E402
"""Assemble immutable machine evidence without fabricating human approval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY_ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.agents.qualification import FoundryAgentQualificationBundleV1
from baloot_platform.benchmark import canonical_sha256
from baloot_platform.config import build_bundle, platform_root


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _paired_comparison(path: Path) -> dict:
    evidence = _json(path)
    expected = str(evidence.get("evidenceSha256") or "")
    payload = {key: item for key, item in evidence.items() if key != "evidenceSha256"}
    if expected != canonical_sha256(payload):
        raise ValueError("paired benchmark evidence checksum mismatch")
    if evidence.get("completionStatus") != "completed":
        raise ValueError("paired benchmark did not complete")
    comparison = dict(evidence.get("comparison") or {})
    if not comparison.get("resultHash"):
        raise ValueError("paired benchmark has no qualifying comparison")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--holdout-evaluation", type=Path,
        default=Path("release/agent-qualification/permanent-holdout-evaluation.json"),
    )
    parser.add_argument(
        "--paired-evidence", type=Path,
        default=Path(".release-runtime/benchmark/paired-duplicate-rounds.json"),
    )
    parser.add_argument(
        "--symmetry-evidence", type=Path,
        default=Path("release/agent-qualification/symmetry.json"),
    )
    parser.add_argument("--professional-review", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("release/agent-qualification.json")
    )
    parser.add_argument("--release-id")
    args = parser.parse_args()
    root = platform_root()
    bundle = build_bundle(root)
    benchmark = bundle.component("benchmark")
    policy_id = str(benchmark["paired_comparison"]["candidate_profile_id"])
    registry = AgentProfileRegistry.from_bundle(root)
    profile = registry.get(policy_id)
    lineup = registry.snapshot_lineup({seat: policy_id for seat in (
        "north", "west", "south", "east"
    )})
    qualification = FoundryAgentQualificationBundleV1.build(
        release_id=(args.release_id or f"{policy_id}-qualification-candidate"),
        profile=profile,
        lineup_snapshot=lineup,
        benchmark_profile=benchmark,
        permanent_holdout=_json((root / args.holdout_evaluation).resolve()),
        paired_comparison=_paired_comparison((root / args.paired_evidence).resolve()),
        symmetry_evidence=_json((root / args.symmetry_evidence).resolve()),
        professional_review=(
            _json((root / args.professional_review).resolve())
            if args.professional_review is not None
            else None
        ),
    )
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            qualification.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(output)
    print(qualification.release.status)


if __name__ == "__main__":
    main()
