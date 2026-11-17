from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import ortools

from baloot_platform.research.contracts import canonical_sha256
from baloot_platform.service.container import build_container


def _strategic_profile() -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "minimumPlayedCards": 24,
        "requestedSamples": 32,
        "exactAllocationLimit": 500,
        "maximumSamplingAttempts": 500,
        "maximumWorlds": 500,
        "maximumContinuationSteps": 20_000,
        "maximumPlies": 8,
        "maximumSelectionSeconds": 30,
        "requireExactForQualification": True,
        "continuationPolicy": "observer_safe_heuristic_v1",
        "policyProfileId": "maximin_margin",
        "policyRevision": 1,
        "outcomeBoundary": 1,
        "utility": "actor_team_awarded_point_margin",
        "tieBreak": "maximin_then_total_then_card_id",
    }


def _recipe(research: Any, *, mode: str, attempts: int, seed_start: int) -> dict[str, Any]:
    capabilities = research.capabilities()
    active = dict(capabilities["research"])
    resources: dict[str, Any] = {
        "generationLane": "ortools",
        "seedStart": seed_start,
        "seedCount": attempts,
        "seedsPerShard": min(5, attempts),
        "candidatesPerSeed": 1,
        "solverTimeoutSeconds": 15,
        "maximumConflicts": 250_000,
        "analysisMode": "screening",
        "playConstructionMode": mode,
        "deepVerificationPolicy": "required_gates",
    }
    if mode == "cp_sat_actor_safe_late_game_v1":
        resources["strategicPlay"] = _strategic_profile()
    return {
        "schemaVersion": 1,
        "titleAr": (
            "اختبار OR-Tools الاستراتيجي"
            if mode == "cp_sat_actor_safe_late_game_v1"
            else "خط أساس اللاعبين المتساوين"
        ),
        "output": {"kind": "laffa", "resultCount": attempts},
        "profiles": {
            "domainHash": capabilities["sha256"],
            "bundleHash": capabilities["bundleHash"],
            "rulesHash": capabilities["rulesHash"],
            "conventionHash": capabilities["conventionHash"],
            "agentRelease": active["defaultAgentRelease"],
            "analysisProfileHash": active["analysisProfiles"]["screening"],
            "scoreProfileHash": active["defaultScoreProfileHash"],
        },
        "situation": {
            "dealer": "east",
            "buyer": None,
            "purchaseRound": None,
            "trumpSuit": None,
            "contracts": [],
            "matchScore": [0, 0],
            "requiredCards": {},
            "forbiddenCards": {},
            "projects": [],
            "escalations": [],
        },
        "tactics": [
            {
                "kind": "small_card_big_gain",
                "mode": "preference",
                "minimumScore": 1,
                "weight": 1,
                "perspectives": [],
            }
        ],
        "qualification": {
            "minimumTruth": "provisional",
            "minimumConfidence": 0,
            "minimumSurroundingQuality": 70,
            "requireObserverSafety": True,
            "requireAllDecisionAudit": True,
            "maximumRegret": 0,
        },
        "resources": resources,
        "status": "published",
    }


def _wait(research: Any, code: str, *, timeout: float = 900) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = research.get_run(code)
        if run is None:
            raise RuntimeError(f"Research run disappeared: {code}")
        if run["status"] in {"completed", "failed", "cancelled"}:
            return run
        time.sleep(0.1)
    raise TimeoutError(f"Research run exceeded the pilot deadline: {code}")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _metrics(research: Any, run: Mapping[str, Any]) -> dict[str, Any]:
    items = research.list_results(
        run_code=str(run["runCode"]),
        sort="result_code_asc",
        limit=500,
    )["items"]
    qualified = [item for item in items if item["disposition"] == "qualified"]
    fingerprints = {str(item.get("gameHash") or item["resultHash"]) for item in qualified}
    cpu_seconds = run.get("cpuSeconds")
    quality = [
        float(dict(item.get("qualification") or {}).get("minimumSurroundingQuality", 0))
        for item in items
    ]
    scores = [float(item["score"]) for item in items]
    qualification_failures = Counter(
        str(reason)
        for item in items
        for reason in dict(item.get("qualification") or {}).get("failures", ())
    )
    return {
        "runCode": str(run["runCode"]),
        "status": str(run["status"]),
        "attemptedSeeds": int(run.get("attemptedSeeds", 0)),
        "resultCount": len(items),
        "qualifiedCount": len(qualified),
        "distinctQualifiedCount": len(fingerprints),
        "meanScore": _mean(scores),
        "meanSurroundingQuality": _mean(quality),
        "minimumSurroundingQuality": min(quality) if quality else None,
        "semanticDefectCount": int(run.get("semanticDefectCount", 0)),
        "quarantineCount": int(run.get("quarantineCount", 0)),
        "elapsedWallSeconds": float(run.get("elapsedSeconds", 0)),
        "cpuSeconds": float(cpu_seconds) if cpu_seconds is not None else None,
        "qualifiedPerCpuHour": (
            len(fingerprints) * 3600.0 / float(cpu_seconds)
            if cpu_seconds is not None and float(cpu_seconds) > 0
            else None
        ),
        "cpuSecondsPerDistinctQualified": (
            float(cpu_seconds) / len(fingerprints)
            if cpu_seconds is not None and fingerprints
            else None
        ),
        "strategicSelectionDecisions": int(run.get("strategicSelectionDecisions", 0)),
        "strategicExactDecisions": int(run.get("strategicExactDecisions", 0)),
        "strategicProvisionalDecisions": int(run.get("strategicProvisionalDecisions", 0)),
        "strategicFallbackDecisions": int(run.get("strategicFallbackDecisions", 0)),
        "playCompilerOptimal": int(run.get("playCompilerOptimal", 0)),
        "constructionPreScreenCandidates": int(
            run.get("constructionPreScreenCandidates", 0)
        ),
        "constructionPreScreenRejected": int(
            run.get("constructionPreScreenRejected", 0)
        ),
        "tacticalViabilityPreScreenCandidates": int(
            run.get("tacticalViabilityPreScreenCandidates", 0)
        ),
        "tacticalViabilityPreScreenRejected": int(
            run.get("tacticalViabilityPreScreenRejected", 0)
        ),
        "constructionTargetMissCount": int(
            run.get("constructionTargetMissCount", 0)
        ),
        "analyzedCandidateCount": int(run.get("analyzedCandidateCount", 0)),
        "qualificationFailureCounts": dict(sorted(qualification_failures.items())),
        "resultSummaries": [
            {
                "resultCode": str(item["resultCode"]),
                "gameHash": str(item["gameHash"]),
                "score": float(item["score"]),
                "disposition": str(item["disposition"]),
                "truthLabel": str(item["truthLabel"]),
                "qualificationFailures": list(
                    dict(item.get("qualification") or {}).get("failures", ())
                ),
                "surroundingQuality": float(
                    dict(item.get("qualification") or {}).get(
                        "minimumSurroundingQuality", 0
                    )
                ),
                "strategicSelection": dict(item.get("strategicSelection") or {}),
            }
            for item in items
        ],
    }


def _decision(baseline: Mapping[str, Any], strategic: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if baseline["status"] != "completed" or strategic["status"] != "completed":
        reasons.append("run_incomplete")
    if int(baseline["semanticDefectCount"]) or int(strategic["semanticDefectCount"]):
        reasons.append("semantic_defect")
    if int(strategic["strategicExactDecisions"]) < 1:
        reasons.append("no_exact_policy_conditioned_decision")
    baseline_quality = baseline.get("meanSurroundingQuality")
    strategic_quality = strategic.get("meanSurroundingQuality")
    quality_preserved = (
        baseline_quality is not None
        and strategic_quality is not None
        and float(strategic_quality) >= float(baseline_quality)
    )
    if not quality_preserved:
        reasons.append("surrounding_quality_not_preserved")
    baseline_yield = baseline.get("qualifiedPerCpuHour")
    strategic_yield = strategic.get("qualifiedPerCpuHour")
    yield_multiplier = (
        float(strategic_yield) / float(baseline_yield)
        if baseline_yield is not None
        and strategic_yield is not None
        and float(baseline_yield) > 0
        else None
    )
    strategic_cost = strategic.get("cpuSecondsPerDistinctQualified")
    baseline_cost = baseline.get("cpuSecondsPerDistinctQualified")
    cost_reduction = (
        1.0 - float(strategic_cost) / float(baseline_cost)
        if strategic_cost is not None
        and baseline_cost is not None
        and float(baseline_cost) > 0
        else None
    )
    yield_passed = (
        yield_multiplier is not None and yield_multiplier >= 2.0
    ) or (
        int(baseline["distinctQualifiedCount"]) == 0
        and int(strategic["distinctQualifiedCount"]) > 0
    )
    cost_passed = cost_reduction is not None and cost_reduction >= 0.40
    if not (yield_passed or cost_passed):
        reasons.append("promotion_performance_gate_not_met")
    promoted = not reasons
    return {
        "promoteStrategicMode": promoted,
        "recommendation": "promote" if promoted else "keep_experimental",
        "qualityPreserved": quality_preserved,
        "yieldMultiplier": yield_multiplier,
        "costReduction": cost_reduction,
        "totalCpuChange": (
            float(strategic["cpuSeconds"]) / float(baseline["cpuSeconds"]) - 1.0
            if strategic.get("cpuSeconds") is not None
            and baseline.get("cpuSeconds") is not None
            and float(baseline["cpuSeconds"]) > 0
            else None
        ),
        "yieldGatePassed": yield_passed,
        "costGatePassed": cost_passed,
        "requiredYieldMultiplier": 2.0,
        "requiredCostReduction": 0.40,
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=202608080001)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("release/strategic-play-pilot-v1.json"),
    )
    args = parser.parse_args()
    if not 1 <= args.attempts <= 100:
        raise ValueError("attempts must be between 1 and 100")
    runtime = Path(tempfile.mkdtemp(prefix="baloot-strategic-pilot-"))
    container = build_container(root=args.root.resolve(), writable_root=runtime)
    try:
        research = container.research
        modes = (
            "actor_safe_agent_verified",
            "cp_sat_actor_safe_late_game_v1",
        )
        runs: dict[str, dict[str, Any]] = {}
        recipe_hashes: dict[str, str] = {}
        for mode in modes:
            recipe = research.create_recipe(
                _recipe(
                    research,
                    mode=mode,
                    attempts=args.attempts,
                    seed_start=args.seed_start,
                )
            )
            recipe_hashes[mode] = str(recipe["sha256"])
            submitted = research.submit(
                str(recipe["code"]),
                idempotency_key=f"strategic-play-pilot:{mode}:{args.seed_start}:{args.attempts}",
            )
            runs[mode] = _metrics(
                research,
                _wait(research, str(submitted["runCode"])),
            )
        capabilities = research.capabilities()
        stable = {
            "schemaVersion": 1,
            "pilot": "actor_safe_cp_sat_late_game_matched",
            "domainHash": capabilities["sha256"],
            "bundleHash": capabilities["bundleHash"],
            "matchedInputs": {
                "seedStart": args.seed_start,
                "attemptsPerMode": args.attempts,
                "generationLane": "ortools",
                "agentRelease": capabilities["research"]["defaultAgentRelease"],
                "analysisProfileHash": capabilities["research"]["analysisProfiles"][
                    "screening"
                ],
                "scoreProfileHash": capabilities["research"][
                    "defaultScoreProfileHash"
                ],
                "onlyControlledDifference": "playConstructionMode_and_strategicPlay",
            },
            "recipeHashes": recipe_hashes,
            "modes": runs,
            "decision": _decision(
                runs["actor_safe_agent_verified"],
                runs["cp_sat_actor_safe_late_game_v1"],
            ),
            "proofBoundary": {
                "strategicTruth": "policy_conditioned_actor_safe",
                "globalOptimalityClaimed": False,
                "fullLineCompilerPurpose": "legality_and_scoring_parity",
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
