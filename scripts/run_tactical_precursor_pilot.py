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

from baloot_platform.research.contracts import canonical_sha256  # noqa: E402
from baloot_platform.service.container import build_container  # noqa: E402
from scripts.run_ortools_candidate_quality_pilot import (  # noqa: E402
    _target_metrics,
)
from scripts.run_strategic_play_pilot import _recipe, _wait  # noqa: E402


def _pilot_recipe(
    research: Any,
    *,
    attempts: int,
    seed_start: int,
    precursor_enabled: bool,
) -> dict[str, Any]:
    payload = copy.deepcopy(
        _recipe(
            research,
            mode="actor_safe_agent_verified",
            attempts=attempts,
            seed_start=seed_start,
        )
    )
    capabilities = research.capabilities()
    payload["titleAr"] = "اختبار البناء التكتيكي في OR-Tools"
    payload["tactics"] = [{
        "kind": "small_card_big_gain",
        "mode": "qualification_gate",
        "minimumScore": 10,
        "weight": 1,
        "perspectives": [],
    }]
    # Freeze the tactical actor so the matched construction comparison cannot
    # be discarded because a different legal buyer won the generated auction.
    payload["situation"]["buyer"] = "south"
    payload["qualification"].update({
        "minimumTruth": "robust_best_known",
        "minimumConfidence": 0.75,
        "minimumSurroundingQuality": 70,
        "maximumRegret": 0,
    })
    payload["profiles"]["analysisProfileHash"] = capabilities["research"][
        "analysisProfiles"
    ]["deep"]
    payload["resources"].update({
        "auctionConstructionMode": "cp_sat_feasibility_line",
        "playConstructionMode": "cp_sat_feasibility_line",
        "analysisMode": "deep",
        "candidatesPerSeed": 1,
    })
    if precursor_enabled:
        payload["resources"]["tacticalConstruction"] = {
            "schemaVersion": 1,
            "id": "tactical_play_precursor_v1",
            "revision": 1,
            "enabled": True,
            "maximumPayoffTricks": 4,
            "minimumLaterRawPoints": 10,
            "requirePrecursorForGatedTactics": True,
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=202608090001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.attempts <= 10:
        raise ValueError("attempts must be between 1 and 10")

    runtime = Path(tempfile.mkdtemp(prefix="baloot-tactical-precursor-pilot-"))
    container = build_container(root=args.root.resolve(), writable_root=runtime)
    try:
        research = container.research
        modes: dict[str, dict[str, Any]] = {}
        recipe_hashes: dict[str, str] = {}
        for label, enabled in (("baseline", False), ("tactical_precursor", True)):
            recipe = research.create_recipe(
                _pilot_recipe(
                    research,
                    attempts=args.attempts,
                    seed_start=args.seed_start,
                    precursor_enabled=enabled,
                )
            )
            recipe_hashes[label] = str(recipe["sha256"])
            submitted = research.submit(
                str(recipe["code"]),
                idempotency_key=(
                    f"tactical-precursor:{label}:{args.seed_start}:{args.attempts}"
                ),
            )
            modes[label] = _target_metrics(
                research,
                _wait(research, str(submitted["runCode"])),
                "small_card_big_gain",
            )

        baseline = modes["baseline"]
        candidate = modes["tactical_precursor"]
        semantic_defects = int(baseline["semanticDefectCount"]) + int(
            candidate["semanticDefectCount"]
        )
        verified_gain = int(candidate["verifiedTargetCount"]) - int(
            baseline["verifiedTargetCount"]
        )
        qualified_gain = int(candidate["distinctQualifiedTargetCount"]) - int(
            baseline["distinctQualifiedTargetCount"]
        )
        quality_baseline = baseline.get("meanSurroundingQuality")
        quality_candidate = candidate.get("meanSurroundingQuality")
        quality_gain = (
            float(quality_candidate) - float(quality_baseline)
            if quality_candidate is not None and quality_baseline is not None
            else None
        )
        retain = (
            semantic_defects == 0
            and verified_gain > 0
            and (quality_gain is None or quality_gain >= -5)
        )
        capabilities = research.capabilities()
        stable = {
            "schemaVersion": 1,
            "pilot": "cp_sat_small_card_tactical_precursor",
            "domainHash": capabilities["sha256"],
            "bundleHash": capabilities["bundleHash"],
            "matchedInputs": {
                "seedStart": args.seed_start,
                "attemptsPerMode": args.attempts,
                "generationLane": "ortools",
                "auctionConstructionMode": "cp_sat_feasibility_line",
                "playConstructionMode": "cp_sat_feasibility_line",
                "analysisMode": "deep",
                "targetTactic": "small_card_big_gain",
                "buyer": "south",
                "onlyControlledDifference": "tacticalConstruction.enabled",
            },
            "recipeHashes": recipe_hashes,
            "modes": modes,
            "decision": {
                "retainTacticalPrecursor": retain,
                "verifiedTargetGain": verified_gain,
                "qualifiedDistinctTargetGain": qualified_gain,
                "meanSurroundingQualityGain": quality_gain,
                "semanticDefectCount": semantic_defects,
                "recommendation": "retain_experimental" if retain else "do_not_retain",
            },
            "proofBoundary": {
                "precursor": "complete_deal_oracle_structural_guidance",
                "qualificationAuthority": "actor_safe_complete_decision_audit_and_causal_verifier",
                "precursorIsTacticalTruth": False,
                "qualificationBypass": False,
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
