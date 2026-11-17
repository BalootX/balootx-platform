from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import ortools

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from baloot_platform.research.contracts import (  # noqa: E402
    RESEARCH_STRATEGIC_AUCTION_DEEP_BID_SAMPLES_V3,
    RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1,
    canonical_sha256,
)
from baloot_platform.service.container import build_container  # noqa: E402

from scripts.run_strategic_play_pilot import _metrics, _recipe, _wait  # noqa: E402


def _actor_safe_cp_sat_auction() -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "requestedSamples": RESEARCH_STRATEGIC_AUCTION_DEEP_BID_SAMPLES_V3,
        "integerScale": 1_000,
        "maximumConflicts": 50_000,
        "maximumSelectionSeconds": 5.0,
        "objective": "expected_round_margin",
        "tieBreak": RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1,
        "requireOptimalForQualification": True,
    }


def _candidate_recipe(
    research: Any,
    *,
    attempts: int,
    seed_start: int,
    candidates_per_seed: int,
    tactic_kind: str = "small_card_big_gain",
    tactic_mode: str = "qualification_gate",
    tactic_minimum_score: float = 10.0,
    minimum_truth: str = "robust_best_known",
    minimum_confidence: float = 0.75,
    minimum_surrounding_quality: float = 70.0,
    maximum_regret: float = 0.0,
    analysis_mode: str = "cascaded",
) -> dict[str, Any]:
    payload = copy.deepcopy(
        _recipe(
            research,
            mode="actor_safe_agent_verified",
            attempts=attempts,
            seed_start=seed_start,
        )
    )
    payload["titleAr"] = "اختبار جودة مرشحات OR-Tools"
    payload["output"]["resultCount"] = attempts * candidates_per_seed
    payload["tactics"] = [{
        "kind": tactic_kind,
        "mode": tactic_mode,
        "minimumScore": tactic_minimum_score,
        "weight": 1,
        "perspectives": [],
    }]
    payload["qualification"].update({
        "minimumTruth": minimum_truth,
        "minimumConfidence": minimum_confidence,
        "minimumSurroundingQuality": minimum_surrounding_quality,
        "maximumRegret": maximum_regret,
    })
    capabilities = research.capabilities()
    payload["profiles"]["analysisProfileHash"] = capabilities["research"][
        "analysisProfiles"
    ][analysis_mode]
    payload["resources"].update(
        {
            "candidatesPerSeed": candidates_per_seed,
            "auctionConstructionMode": "cp_sat_actor_safe_decision_v1",
            "strategicAuction": _actor_safe_cp_sat_auction(),
            "analysisMode": analysis_mode,
        }
    )
    return payload


def _target_metrics(research: Any, run: dict[str, Any], tactic_kind: str) -> dict[str, Any]:
    metrics = _metrics(research, run)
    items = research.list_results(
        run_code=str(run["runCode"]),
        sort="result_code_asc",
        limit=500,
    )["items"]
    verified = [
        item for item in items
        if tactic_kind in set(item.get("verifiedTacticalTags") or ())
    ]
    qualified = [item for item in verified if item["disposition"] == "qualified"]
    metrics.update({
        "targetTactic": tactic_kind,
        "verifiedTargetCount": len(verified),
        "qualifiedTargetCount": len(qualified),
        "distinctQualifiedTargetCount": len({
            str(item.get("gameHash") or item["resultHash"])
            for item in qualified
        }),
    })
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=202608080001)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--tactic", default="small_card_big_gain")
    parser.add_argument(
        "--tactic-mode",
        choices=("hard", "qualification_gate"),
        default="qualification_gate",
    )
    parser.add_argument("--tactic-minimum-score", type=float, default=10.0)
    parser.add_argument("--minimum-truth", default="robust_best_known")
    parser.add_argument("--minimum-confidence", type=float, default=0.75)
    parser.add_argument("--minimum-surrounding-quality", type=float, default=70.0)
    parser.add_argument("--maximum-regret", type=float, default=0.0)
    parser.add_argument(
        "--analysis-mode",
        choices=("screening", "cascaded", "deep"),
        default="cascaded",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.attempts <= 20:
        raise ValueError("attempts must be between 1 and 20")
    if not 2 <= args.candidate_count <= 8:
        raise ValueError("candidate count must be between 2 and 8")
    if not 0 <= args.tactic_minimum_score <= 100:
        raise ValueError("tactic minimum score must be between 0 and 100")
    if not 0 <= args.minimum_confidence <= 1:
        raise ValueError("minimum confidence must be between 0 and 1")
    if not 0 <= args.minimum_surrounding_quality <= 100:
        raise ValueError("minimum surrounding quality must be between 0 and 100")
    if args.maximum_regret < 0:
        raise ValueError("maximum regret cannot be negative")

    runtime = Path(tempfile.mkdtemp(prefix="baloot-candidate-quality-pilot-"))
    container = build_container(root=args.root.resolve(), writable_root=runtime)
    try:
        research = container.research
        runs: dict[str, dict[str, Any]] = {}
        recipe_hashes: dict[str, str] = {}
        for label, candidate_count in (
            ("single_candidate", 1),
            ("candidate_replacement", args.candidate_count),
        ):
            recipe = research.create_recipe(
                _candidate_recipe(
                    research,
                    attempts=args.attempts,
                    seed_start=args.seed_start,
                    candidates_per_seed=candidate_count,
                    tactic_kind=args.tactic,
                    tactic_mode=args.tactic_mode,
                    tactic_minimum_score=args.tactic_minimum_score,
                    minimum_truth=args.minimum_truth,
                    minimum_confidence=args.minimum_confidence,
                    minimum_surrounding_quality=args.minimum_surrounding_quality,
                    maximum_regret=args.maximum_regret,
                    analysis_mode=args.analysis_mode,
                )
            )
            recipe_hashes[label] = str(recipe["sha256"])
            submitted = research.submit(
                str(recipe["code"]),
                idempotency_key=(
                    f"candidate-quality:{label}:{args.seed_start}:"
                    f"{args.attempts}:{candidate_count}"
                ),
            )
            runs[label] = _target_metrics(
                research,
                _wait(research, str(submitted["runCode"])),
                args.tactic,
            )

        baseline = runs["single_candidate"]
        candidate = runs["candidate_replacement"]
        baseline_quality = baseline.get("meanSurroundingQuality")
        candidate_quality = candidate.get("meanSurroundingQuality")
        qualified_gain = int(candidate["distinctQualifiedCount"]) - int(
            baseline["distinctQualifiedCount"]
        )
        qualified_target_gain = int(candidate["distinctQualifiedTargetCount"]) - int(
            baseline["distinctQualifiedTargetCount"]
        )
        verified_target_gain = int(candidate["verifiedTargetCount"]) - int(
            baseline["verifiedTargetCount"]
        )
        quality_gain = (
            float(candidate_quality) - float(baseline_quality)
            if candidate_quality is not None and baseline_quality is not None
            else None
        )
        semantic_defects = int(baseline["semanticDefectCount"]) + int(
            candidate["semanticDefectCount"]
        )
        retain = semantic_defects == 0 and (
            qualified_target_gain > 0
            or (
                verified_target_gain > 0
                and quality_gain is not None
                and quality_gain >= 10.0
            )
        )
        capabilities = research.capabilities()
        stable = {
            "schemaVersion": 1,
            "pilot": "actor_safe_cp_sat_auction_candidate_replacement",
            "domainHash": capabilities["sha256"],
            "bundleHash": capabilities["bundleHash"],
            "matchedInputs": {
                "seedStart": args.seed_start,
                "attemptsPerMode": args.attempts,
                "generationLane": "ortools",
                "auctionConstructionMode": "cp_sat_actor_safe_decision_v1",
                "playConstructionMode": "actor_safe_agent_verified",
                "onlyControlledDifference": "candidatesPerSeed",
                "tactic": args.tactic,
                "tacticMode": args.tactic_mode,
                "tacticMinimumScore": args.tactic_minimum_score,
                "minimumTruth": args.minimum_truth,
                "minimumConfidence": args.minimum_confidence,
                "minimumSurroundingQuality": args.minimum_surrounding_quality,
                "maximumRegret": args.maximum_regret,
                "analysisMode": args.analysis_mode,
            },
            "recipeHashes": recipe_hashes,
            "modes": runs,
            "decision": {
                "retainCandidateReplacement": retain,
                "qualifiedDistinctGain": qualified_gain,
                "qualifiedDistinctTargetGain": qualified_target_gain,
                "verifiedTargetGain": verified_target_gain,
                "meanSurroundingQualityGain": quality_gain,
                "semanticDefectCount": semantic_defects,
                "requiredQualityGainWhenNoQualifiedGain": 10.0,
                "recommendation": "retain" if retain else "do_not_retain",
            },
            "proofBoundary": {
                "candidateReplacement": "deterministic_cp_sat_no_good_cuts",
                "qualificationAuthority": "actor_safe_complete_decision_audit",
                "globalOptimalityClaimed": False,
            },
        }
        result = {
            **stable,
            "environment": {
                "operatingSystem": platform.platform(),
                "python": platform.python_version(),
                "ortools": ortools.__version__,
                "logicalCpuCount": os.cpu_count(),
                "processor": platform.processor(),
                "storageMode": "sqlite_local",
            },
            "evidenceHash": canonical_sha256(stable),
        }
        rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8", newline="\n")
        print(rendered, end="")
        return 0
    finally:
        container.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
