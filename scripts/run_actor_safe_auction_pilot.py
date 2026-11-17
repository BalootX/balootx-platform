from __future__ import annotations

import argparse
import copy
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat
from baloot_platform.research.contracts import (
    RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1,
    ResearchRecipeV1,
)
from baloot_platform.research.strategic_auction import (
    ActorSafeCpSatAuctionProfileV1,
    ActorSafeCpSatAuctionSelectorV1,
)
from baloot_platform.service.container import build_container

from scripts.run_strategic_play_pilot import _recipe


def _context_bound_auction_selector(
    context: Any,
    profile: ActorSafeCpSatAuctionProfileV1,
) -> ActorSafeCpSatAuctionSelectorV1:
    deep_profile = context.analyzers["deep"].profile
    return ActorSafeCpSatAuctionSelectorV1(
        profile,
        exact_search_backend=context.exact_search_backend,
        trace_analysis_profile=deep_profile,
        analysis_identity_hash=str(context.analysis_identity_hashes["deep"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=202608080001)
    parser.add_argument(
        "--mode",
        choices=("agent", "cp_sat"),
        default="agent",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.attempts <= 20:
        raise ValueError("attempts must be between 1 and 20")

    runtime = Path(tempfile.mkdtemp(prefix="baloot-actor-safe-auction-pilot-"))
    container = build_container(root=args.root.resolve(), writable_root=runtime)
    research = container.research
    if research.execution_context_factory is None:
        raise RuntimeError("the actor-safe auction pilot requires Research context authority")
    selected_bundle = container.configuration.resolve()["bundle"]
    _, context = research.execution_context_factory.resolve_bundle_revision(
        str(selected_bundle["revisionId"])
    )
    deep_profile = context.analyzers["deep"].profile
    base_payload = _recipe(
        research,
        mode="actor_safe_agent_verified",
        attempts=args.attempts,
        seed_start=args.seed_start,
    )
    candidate_recipe = ResearchRecipeV1.build(copy.deepcopy(base_payload))
    payload = copy.deepcopy(base_payload)
    strategic_profile: ActorSafeCpSatAuctionProfileV1 | None = None
    if args.mode == "cp_sat":
        payload["resources"].update(
            {
                "auctionConstructionMode": "cp_sat_actor_safe_decision_v1",
                "strategicAuction": {
                    "schemaVersion": 1,
                    "requestedSamples": deep_profile.bid_samples,
                    "integerScale": 1_000,
                    "maximumConflicts": 50_000,
                    "maximumSelectionSeconds": 5.0,
                    "objective": "expected_round_margin",
                    "tieBreak": RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1,
                    "requireOptimalForQualification": True,
                },
            }
        )
        strategic_profile = ActorSafeCpSatAuctionProfileV1.build(
            payload["resources"]["strategicAuction"]
        )
    else:
        payload["resources"]["auctionConstructionMode"] = (
            "actor_safe_agent_verified"
        )
    recipe = ResearchRecipeV1.build(payload)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    cpu_started = time.thread_time()
    try:
        for seed in range(args.seed_start, args.seed_start + args.attempts):
            spec = research._compile_deal_spec(candidate_recipe, seed=seed)
            candidate = research.deal_generator.generate(
                spec,
                root_seed=seed,
                dealer=Seat.EAST,
                rules=research.simulator.rules,
            )[0]
            simulation = research.simulator.simulate_preordered_deck(
                seed,
                candidate.deck_order,
                deal_generator_provenance={
                    **candidate.provenance_dict(),
                    "auctionConstructionMode": recipe.resources[
                        "auctionConstructionMode"
                    ],
                    "playConstructionMode": "actor_safe_agent_verified",
                    "playPolicyRelease": research.simulator.agent_release,
                },
                dealer=Seat.EAST,
                match_score=(0, 0),
                bid_selector=(
                    _context_bound_auction_selector(context, strategic_profile)
                    if strategic_profile is not None
                    else None
                ),
            )
            row: dict[str, object] = {
                "seed": seed,
                "candidateHash": candidate.candidate_hash,
                "publishable": simulation.publishable_laffa,
            }
            if not simulation.publishable_laffa:
                row["status"] = "all_pass"
                rows.append(row)
                continue

            state = simulation.final_trace.trusted_final_state
            exact_actions = tuple(
                {
                    "ordinal": index,
                    "roundIndex": int(item["round"]),
                    "actor": str(item["seat"]),
                    "action": str(item["action"]),
                    "trumpSuit": item.get("trumpSuit"),
                }
                for index, item in enumerate(
                    state.get("biddingHistory") or (), start=1
                )
            )
            auction = research.auction_compiler.compile(
                dealer=Seat.EAST,
                purchase_suit=Card.parse(state["purchaseCard"]).suit,
                rules=research.simulator.rules,
                required_actions=exact_actions,
                maximum_conflicts=int(recipe.resources["maximumConflicts"]),
                timeout_seconds=float(recipe.resources["solverTimeoutSeconds"]),
            )
            if auction is None:
                row["status"] = "auction_verify_failed"
                rows.append(row)
                continue
            escalation = research._compile_escalation_plan(recipe, auction)
            if escalation is None or escalation.actions:
                row["status"] = "escalation_verify_failed"
                rows.append(row)
                continue
            cards, ekka = research._played_line(simulation)
            _, play, _ = research._compile_complete_play(
                recipe,
                candidate.deck_order,
                auction,
                escalation,
                required_cards=cards,
                required_ekka_leads=ekka,
            )
            mismatches = sorted(
                set(
                    [
                        *research._semantic_mismatches(recipe, simulation),
                        *research._compiled_play_mismatches(play, simulation),
                    ]
                )
            )
            if mismatches:
                row.update(status="semantic_defect", mismatches=mismatches)
                rows.append(row)
                continue
            analysis = research.analyzers["screening"].analyze_trace(
                simulation.final_trace
            )
            tactical = research._verify_tactics(
                simulation.final_trace,
                analysis,
                recipe=recipe,
                cancel_check=None,
            )
            audit = research._decision_audit(
                simulation.final_trace,
                analysis,
                tactical,
            )
            qualification = research._qualification(recipe, audit, analysis)
            non_forced = [
                item for item in audit["decisions"] if not bool(item["forced"])
            ]
            bid_regrets = [
                float(item["expectedRegret"])
                for item in non_forced
                if item["decisionKind"] == "bid"
            ]
            play_regrets = [
                float(item["expectedRegret"])
                for item in non_forced
                if item["decisionKind"] == "play"
            ]
            row.update(
                status="qualified" if qualification["qualified"] else "rejected",
                qualified=qualification["qualified"],
                failures=qualification["failures"],
                minimumSurroundingQuality=qualification["minimumSurroundingQuality"],
                truthLabel=qualification["truthLabel"],
                confidence=qualification["confidence"],
                auctionActionCount=len(exact_actions),
                maximumBidRegret=max(bid_regrets, default=0.0),
                maximumPlayRegret=max(play_regrets, default=0.0),
                auctionPlanHash=auction.sha256,
                traceHash=simulation.final_trace.sha256,
            )
            rows.append(row)
    finally:
        container.shutdown()

    qualities = [
        float(row["minimumSurroundingQuality"])
        for row in rows
        if "minimumSurroundingQuality" in row
    ]
    result = {
        "schemaVersion": 1,
        "experiment": (
            "actor_safe_cp_sat_auction_decision_with_exact_trace_verification"
            if args.mode == "cp_sat"
            else "actor_safe_agent_auction_with_exact_cp_sat_trace_verification"
        ),
        "auctionMode": args.mode,
        "strategicAuctionProfile": (
            strategic_profile.to_dict() if strategic_profile is not None else None
        ),
        "seedStart": args.seed_start,
        "seedCount": args.attempts,
        "rows": rows,
        "qualifiedCount": sum(bool(row.get("qualified")) for row in rows),
        "semanticDefectCount": sum(
            row.get("status") == "semantic_defect" for row in rows
        ),
        "meanSurroundingQuality": (
            sum(qualities) / len(qualities) if qualities else None
        ),
        "minimumSurroundingQuality": min(qualities, default=None),
        "wallSeconds": time.perf_counter() - started,
        "threadCpuSeconds": time.thread_time() - cpu_started,
    }
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
