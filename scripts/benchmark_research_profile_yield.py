from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import ortools

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baloot_platform.challenge.constraint_deal_search import (  # noqa: E402
    DealConstraintSpecV1,
    OrToolsCpSatDealGeneratorV1,
)
from baloot_platform.challenge.evidence_v3 import canonical_sha256  # noqa: E402
from baloot_platform.challenge.oracle import ActualDealOracleVerifierV1  # noqa: E402
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1  # noqa: E402
from baloot_platform.config import build_bundle  # noqa: E402
from baloot_platform.core.enums import Seat  # noqa: E402


PROFILES = ("project_400", "kabout_creation", "small_card_big_gain")


def _spec(profile: str, *, seed: int, project_value: int) -> DealConstraintSpecV1:
    project = (
        {
            "seat": "south",
            "contract": "sun",
            "kind": "four_hundred",
            "value": project_value,
            "highRank": "ace",
        }
        if profile == "project_400"
        else None
    )
    tactic = "kabout_creation" if profile == "project_400" else profile
    return DealConstraintSpecV1.build({
        "schemaVersion": 1,
        "id": f"research_yield_{profile}_v1",
        "revision": "1",
        "enabled": True,
        "backend": "ortools_cp_sat_v1",
        "assumedBuyer": "south",
        "purchaseSuits": [],
        "initialRequiredCards": {},
        "finalRequiredCards": {},
        "finalForbiddenCards": {},
        "initialSuitCountRanges": {},
        "finalSuitCountRanges": {},
        "projectPattern": project,
        "maximumSolutions": 1,
        "maximumConflicts": 250_000,
        "safetyTimeoutSeconds": 30.0,
        "solverWorkers": 1,
        "seedOffset": seed,
        "tags": [profile, "benchmark_only"],
        "tacticalObjectiveWeights": {tactic: 100},
        "objectiveTargetSeat": "south",
    })


def _detected_kinds(trace, oracle: ActualDealOracleVerifierV1) -> set[str]:
    probe = oracle.probe_verified_tactical_play(trace)
    return {
        evidence.kind.value
        for decision in probe.scanned_decisions
        for evidence in decision.evidence
    }


def _qualifies(profile: str, trace, oracle: ActualDealOracleVerifierV1) -> tuple[bool, str]:
    state = trace.trusted_final_state
    if profile == "project_400":
        matched = any(
            str(item.get("kind")) == "four_hundred"
            and str(item.get("owner")) == "south"
            for item in state.get("projects") or ()
        ) and state.get("buyer") == "south" and state.get("contract") == "sun"
        return matched, "verified_project_400" if matched else "project_not_realized"
    if profile == "kabout_creation":
        score = dict(state.get("roundScore") or {})
        matched = bool(score.get("kabout")) and score.get("winner") == "north_south"
        return matched, "engine_kabout" if matched else "no_engine_kabout"
    kinds = _detected_kinds(trace, oracle)
    matched = "small_card_big_gain" in kinds
    return matched, "exact_small_card_big_gain" if matched else "no_exact_small_card_gain"


def _lane(
    profile: str,
    *,
    seeds: tuple[int, ...],
    constrained: bool,
    simulator: NaturalLaffaSimulatorV1,
    generator: OrToolsCpSatDealGeneratorV1,
    oracle: ActualDealOracleVerifierV1,
    project_value: int,
) -> dict[str, object]:
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for seed in seeds:
        construction_started = time.perf_counter()
        if constrained:
            candidate = generator.generate(
                _spec(profile, seed=seed, project_value=project_value),
                root_seed=seed,
                dealer=Seat.WEST,
                rules=simulator.rules,
            )[0]
            construction_seconds = time.perf_counter() - construction_started
            simulation = simulator.simulate_preordered_deck(
                seed,
                candidate.deck_order,
                deal_generator_provenance=candidate.provenance_dict(),
                dealer=Seat.WEST,
            )
            structural_score = candidate.structural_objective_score
        else:
            simulation = simulator.simulate(seed, dealer=Seat.WEST)
            construction_seconds = time.perf_counter() - construction_started
            structural_score = 0
        if not simulation.publishable_laffa:
            qualified, reason = False, "non_publishable_trace"
        else:
            qualified, reason = _qualifies(profile, simulation.final_trace, oracle)
        rows.append({
            "seed": seed,
            "traceHash": simulation.final_trace.sha256,
            "qualified": qualified,
            "reason": reason,
            "buyer": simulation.final_trace.trusted_final_state.get("buyer"),
            "contract": simulation.final_trace.trusted_final_state.get("contract"),
            "structuralObjectiveScore": structural_score,
            "constructionSeconds": round(construction_seconds, 6),
        })
    wall = time.perf_counter() - started
    qualified_count = sum(bool(item["qualified"]) for item in rows)
    return {
        "seedCount": len(seeds),
        "qualifiedCount": qualified_count,
        "qualifiedPerCpuHour": round(qualified_count * 3600.0 / wall, 6),
        "secondsPerQualified": round(wall / qualified_count, 6) if qualified_count else None,
        "wallSeconds": round(wall, 6),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=202608061000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "release" / "research-profile-yield.json",
    )
    args = parser.parse_args()
    if not 2 <= args.seeds <= 100:
        parser.error("--seeds must be in [2, 100]")
    bundle = build_bundle(ROOT)
    simulator = NaturalLaffaSimulatorV1.from_root(
        ROOT,
        profile_id="foundry_balanced",
        maximum_deal_attempts=1,
    )
    generator = OrToolsCpSatDealGeneratorV1()
    oracle = ActualDealOracleVerifierV1(
        bundle.rules,
        maximum_remaining_cards=8,
        maximum_nodes=100_000,
        maximum_seconds=1.0,
        maximum_cache_entries=50_000,
    )
    project_value = int(
        bundle.rules.data["projects"]["four_of_kind"]
        ["values_game_points"]["sun"]["ace"]
    )
    profiles: dict[str, object] = {}
    for index, profile in enumerate(PROFILES):
        seeds = tuple(args.seed_start + index * args.seeds + offset for offset in range(args.seeds))
        organic = _lane(
            profile,
            seeds=seeds,
            constrained=False,
            simulator=simulator,
            generator=generator,
            oracle=oracle,
            project_value=project_value,
        )
        guided = _lane(
            profile,
            seeds=seeds,
            constrained=True,
            simulator=simulator,
            generator=generator,
            oracle=oracle,
            project_value=project_value,
        )
        organic_yield = float(organic["qualifiedPerCpuHour"])
        guided_yield = float(guided["qualifiedPerCpuHour"])
        organic_cost = organic["secondsPerQualified"]
        guided_cost = guided["secondsPerQualified"]
        yield_ratio = guided_yield / organic_yield if organic_yield > 0 else None
        zero_baseline_gain = organic_yield == 0 and guided_yield > 0
        cost_reduction = (
            1.0 - float(guided_cost) / float(organic_cost)
            if organic_cost is not None and guided_cost is not None
            else None
        )
        screening_threshold_passed = bool(
            zero_baseline_gain
            or (yield_ratio is not None and yield_ratio >= 1.5)
            or (cost_reduction is not None and cost_reduction >= 0.25)
        )
        profiles[profile] = {
            "organic": organic,
            "orTools": guided,
            "yieldRatio": round(yield_ratio, 6) if yield_ratio is not None else None,
            "organicZeroBaselineGain": zero_baseline_gain,
            "costReduction": round(cost_reduction, 6) if cost_reduction is not None else None,
            "screeningYieldThresholdPassed": screening_threshold_passed,
            "promotionGatePassed": False,
            "promotionBlockers": ["observer_safe_full_audit_not_run"],
        }
    normalized = {
        "schemaVersion": 1,
        "kind": "matched_research_profile_yield_v1",
        "proofBoundary": (
            "Bounded local comparison of engine-realized project/kabout outcomes and exact late-game "
            "small-card evidence. It is not professional acceptance or a global optimality claim."
        ),
        "bundleHash": bundle.sha256,
        "rulesHash": bundle.rules.sha256,
        "agentRelease": simulator.agent_release,
        "seedCountPerLane": args.seeds,
        "profiles": profiles,
    }
    evidence = {
        **normalized,
        "normalizedSha256": canonical_sha256(normalized),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "ortools": str(ortools.__version__),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({
        "output": str(args.output),
        "normalizedSha256": evidence["normalizedSha256"],
        "promotion": {
            key: value["promotionGatePassed"]
            for key, value in profiles.items()
        },
        "screeningYield": {
            key: value["screeningYieldThresholdPassed"]
            for key, value in profiles.items()
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
