from __future__ import annotations

import argparse
import copy
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
from baloot_platform.challenge.guided_search import (  # noqa: E402
    GuidedLegalSearchV1,
    GuidedSearchProfileV1,
)
from baloot_platform.challenge.oracle import ActualDealOracleVerifierV1  # noqa: E402
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1  # noqa: E402
from baloot_platform.config import build_bundle  # noqa: E402
from baloot_platform.core.cards import Card  # noqa: E402
from baloot_platform.core.dealing import DealLayoutV1  # noqa: E402
from baloot_platform.core.enums import Seat  # noqa: E402
from baloot_platform.core.rules import RuleBook  # noqa: E402


def _spec() -> DealConstraintSpecV1:
    return DealConstraintSpecV1.build(
        {
            "schemaVersion": 1,
            "id": "cp_sat_structural_benchmark_v1",
            "revision": "1",
            "enabled": True,
            "backend": "ortools_cp_sat_v1",
            "assumedBuyer": "south",
            "purchaseSuits": ["spades"],
            "initialRequiredCards": {"south": ["SJ", "S9"]},
            "initialSuitCountRanges": {
                "south": {"spades": {"minimum": 2, "maximum": 5}},
            },
            "finalSuitCountRanges": {
                "south": {"spades": {"minimum": 3, "maximum": 8}},
            },
            "projectPattern": None,
            "maximumSolutions": 1,
            "maximumConflicts": 1_000_000,
            "safetyTimeoutSeconds": 30.0,
            "solverWorkers": 1,
            "seedOffset": 0,
            "tags": ["structural_benchmark", "engine_validation_required"],
        }
    )


def _structural_hit(deck: tuple[int, ...], layout: DealLayoutV1) -> bool:
    purchase_spade = deck[layout.purchase_position] // 8 == 0
    initial_spades = sum(
        deck[position] // 8 == 0
        for position in layout.positions_for(Seat.SOUTH, final=False)
    )
    final_spades = sum(
        deck[position] // 8 == 0
        for position in layout.positions_for(Seat.SOUTH, final=True)
    )
    initial_cards = {
        deck[position] for position in layout.positions_for(Seat.SOUTH, final=False)
    }
    key_cards = {Card.parse("SJ").id, Card.parse("S9").id}
    return (
        purchase_spade
        and key_cards.issubset(initial_cards)
        and 2 <= initial_spades <= 5
        and 3 <= final_spades <= 8
    )


def _benchmark_layout(rules: RuleBook) -> DealLayoutV1:
    return DealLayoutV1.build(
        rules.structural_domain,
        dealer=Seat.EAST,
        buyer=Seat.SOUTH,
    )


def _guided_profile(
    bundle,
    *,
    constraint_enabled: bool,
    cp_candidates: int = 1,
) -> GuidedSearchProfileV1:
    value = copy.deepcopy(bundle.component("challenge")["guided_search"])
    value["dealConstraint"]["enabled"] = constraint_enabled
    if constraint_enabled:
        value["candidatesPerSeed"] = cp_candidates
        value["dealConstraint"]["maximumSolutions"] = cp_candidates
    return GuidedSearchProfileV1.build(value)


def _guided_row(searcher: GuidedLegalSearchV1, root_seed: int) -> dict[str, object]:
    started = time.perf_counter()
    result = searcher.search(root_seed)
    elapsed = time.perf_counter() - started
    selected = next(
        item
        for item in result.evaluated
        if str(item["traceHash"]) == result.selected_trace_hash
    )
    opportunities = tuple(selected.get("actualDealOpportunities") or ())
    complete_gaps = tuple(
        float(item["outcomeGap"])
        for item in opportunities
        if bool(item.get("complete"))
    )
    return {
        "rootSeed": int(root_seed),
        "profileHash": result.profile_hash,
        "traceHash": result.selected_trace_hash,
        "searchHash": result.sha256,
        "provenance": result.provenance,
        "targetSatisfied": result.target_satisfied,
        "selectedScore": result.selected_score,
        "actualDealOpportunityPassed": bool(
            selected.get("actualDealOpportunityPassed", False)
        ),
        "completeOpportunityCount": len(complete_gaps),
        "peakActualDealOutcomeGap": max(complete_gaps, default=0.0),
        "constraintBuyerMatched": selected.get("constraintBuyerMatched"),
        "elapsedSeconds": round(elapsed, 6),
    }


def _guided_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    elapsed = sum(float(item["elapsedSeconds"]) for item in rows)
    targets = sum(bool(item["targetSatisfied"]) for item in rows)
    return {
        "rootCount": len(rows),
        "targetSatisfied": targets,
        "actualDealOpportunityHits": sum(
            bool(item["actualDealOpportunityPassed"]) for item in rows
        ),
        "completeOpportunityCount": sum(
            int(item["completeOpportunityCount"]) for item in rows
        ),
        "buyerMatches": sum(item["constraintBuyerMatched"] is True for item in rows),
        "meanSelectedScore": round(
            sum(float(item["selectedScore"]) for item in rows) / len(rows),
            6,
        ),
        "meanPeakActualDealOutcomeGap": round(
            sum(float(item["peakActualDealOutcomeGap"]) for item in rows) / len(rows),
            6,
        ),
        "wallSeconds": round(elapsed, 6),
        "secondsPerTarget": (round(elapsed / targets, 6) if targets else None),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=int, default=12)
    parser.add_argument(
        "--challenge-roots",
        type=int,
        default=0,
        help="also compare guided tactical-opportunity yield on this many roots",
    )
    parser.add_argument(
        "--cp-candidates",
        type=int,
        default=1,
        help="deterministic CP-SAT candidates evaluated per challenge root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "release" / "cp-sat-guided-spike.json",
    )
    args = parser.parse_args()
    if not 1 <= args.roots <= 100:
        raise ValueError("benchmark roots must be in [1, 100]")
    if not 0 <= args.challenge_roots <= args.roots:
        raise ValueError("challenge benchmark roots must be in [0, roots]")
    if not 1 <= args.cp_candidates <= 100:
        raise ValueError("CP-SAT candidates must be in [1, 100]")

    bundle = build_bundle(ROOT)
    simulator = NaturalLaffaSimulatorV1.from_root(
        ROOT,
        profile_id="foundry_throughput",
        maximum_deal_attempts=1,
    )
    generator = OrToolsCpSatDealGeneratorV1()
    spec = _spec()
    layout = _benchmark_layout(bundle.rules)
    roots = tuple(202608060000 + index for index in range(args.roots))

    organic_rows = []
    organic_started = time.perf_counter()
    for seed in roots:
        simulation = simulator.simulate(seed, dealer=Seat.EAST)
        trace = simulation.final_trace
        deck = tuple(
            int(item) for item in trace.trusted_final_state["shuffle"]["deckOrder"]
        )
        organic_rows.append(
            {
                "rootSeed": seed,
                "traceHash": trace.sha256,
                "deckCommitment": trace.seed_provenance["deckCommitment"],
                "publishable": simulation.publishable_laffa,
                "buyer": trace.trusted_final_state.get("buyer"),
                "structuralHit": _structural_hit(deck, layout),
            }
        )
    organic_seconds = time.perf_counter() - organic_started

    cp_rows = []
    cp_generation_seconds = 0.0
    cp_replay_seconds = 0.0
    for seed in roots:
        started = time.perf_counter()
        candidate = generator.generate(
            spec,
            root_seed=seed,
            dealer=Seat.EAST,
            rules=bundle.rules,
        )[0]
        cp_generation_seconds += time.perf_counter() - started
        repeated = generator.generate(
            spec,
            root_seed=seed,
            dealer=Seat.EAST,
            rules=bundle.rules,
        )[0]
        if repeated.candidate_hash != candidate.candidate_hash:
            raise AssertionError("CP-SAT candidate generation is not deterministic")
        started = time.perf_counter()
        simulation = simulator.simulate_preordered_deck(
            seed,
            candidate.deck_order,
            deal_generator_provenance=candidate.provenance_dict(),
            dealer=Seat.EAST,
        )
        cp_replay_seconds += time.perf_counter() - started
        trace = simulation.final_trace
        cp_rows.append(
            {
                "rootSeed": seed,
                "candidateHash": candidate.candidate_hash,
                "traceHash": trace.sha256,
                "deckCommitment": candidate.deck_commitment,
                "publishable": simulation.publishable_laffa,
                "buyer": trace.trusted_final_state.get("buyer"),
                "assumedBuyerMatched": trace.trusted_final_state.get("buyer")
                == "south",
                "structuralHit": _structural_hit(candidate.deck_order, layout),
            }
        )

    normalized = {
        "schemaVersion": 1,
        "kind": "cp_sat_guided_structural_spike_v1",
        "proofBoundary": (
            "Structural deal yield and deterministic engine replay only; "
            "this does not prove challenge quality or finalist yield."
        ),
        "rootCount": len(roots),
        "rulesHash": bundle.rules.sha256,
        "bundleHash": bundle.sha256,
        "agentRelease": simulator.agent_release,
        "constraintProfile": spec.to_dict(),
        "organic": {
            "structuralHits": sum(row["structuralHit"] for row in organic_rows),
            "publishable": sum(row["publishable"] for row in organic_rows),
            "buyerMatches": sum(row["buyer"] == "south" for row in organic_rows),
            "rows": organic_rows,
        },
        "cpSat": {
            "structuralHits": sum(row["structuralHit"] for row in cp_rows),
            "publishable": sum(row["publishable"] for row in cp_rows),
            "buyerMatches": sum(row["assumedBuyerMatched"] for row in cp_rows),
            "rows": cp_rows,
        },
    }
    if args.challenge_roots:
        guided_simulator = NaturalLaffaSimulatorV1.from_root(
            ROOT,
            profile_id="foundry_balanced",
            maximum_deal_attempts=1,
        )
        scanner = ActualDealOracleVerifierV1(bundle.rules)
        ordinary_search = GuidedLegalSearchV1(
            guided_simulator,
            _guided_profile(bundle, constraint_enabled=False),
            opportunity_scanner=scanner,
            deal_generator=generator,
        )
        constrained_search = GuidedLegalSearchV1(
            guided_simulator,
            _guided_profile(
                bundle,
                constraint_enabled=True,
                cp_candidates=args.cp_candidates,
            ),
            opportunity_scanner=scanner,
            deal_generator=generator,
        )
        challenge_seeds = roots[: args.challenge_roots]
        ordinary_rows = [_guided_row(ordinary_search, seed) for seed in challenge_seeds]
        constrained_rows = [
            _guided_row(constrained_search, seed) for seed in challenge_seeds
        ]
        normalized["guidedChallengeYield"] = {
            "proofBoundary": (
                "Exact actual-deal tactical-opportunity discovery only; "
                "observer-safe finalist validation is still required."
            ),
            "cpCandidatesPerRoot": args.cp_candidates,
            "ordinary": _guided_summary(ordinary_rows),
            "cpSat": _guided_summary(constrained_rows),
        }
    evidence = {
        **normalized,
        "normalizedSha256": canonical_sha256(normalized),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "ortools": str(ortools.__version__),
        },
        "timing": {
            "organicSeconds": round(organic_seconds, 6),
            "organicCompleteTracesPerSecond": round(len(roots) / organic_seconds, 6),
            "cpSatConstructionSeconds": round(cp_generation_seconds, 6),
            "cpSatCandidatesPerSecond": round(len(roots) / cp_generation_seconds, 6),
            "cpSatEngineReplaySeconds": round(cp_replay_seconds, 6),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "normalizedSha256": evidence["normalizedSha256"],
                "organicStructuralHits": evidence["organic"]["structuralHits"],
                "cpSatStructuralHits": evidence["cpSat"]["structuralHits"],
                "cpSatBuyerMatches": evidence["cpSat"]["buyerMatches"],
                "guidedChallengeYield": evidence.get("guidedChallengeYield"),
                "timing": evidence["timing"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
