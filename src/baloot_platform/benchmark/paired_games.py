from __future__ import annotations

import json
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from baloot_platform.agents.profiles import (
    AdaptivePlayStrategy,
    AgentProfileRegistry,
)
from baloot_platform.analysis.search_errors import require_uncensored_play_trace
from baloot_platform.config import ProfileBundle, build_bundle, platform_root
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit, Team
from baloot_platform.version import __version__

from .adjudication import canonical_sha256
from .strength import StrengthBenchmark


RUNNER_ID = "paired_duplicate_rounds"
RUNNER_VERSION = "1.1.0"


def _deal_hash(state) -> str:
    return canonical_sha256(
        {
            "rulesSha256": state.rules_hash,
            "seed": state.seed,
            "dealer": state.dealer.name.lower(),
            "cutPosition": (
                state.shuffle.cut_position
                if state.shuffle is not None
                else None
            ),
            "initialHands": list(state.initial_hands),
            "purchaseCard": (
                state.purchase_card.id
                if state.purchase_card is not None
                else None
            ),
            "contract": state.contract.value,
            "buyer": state.buyer.name.lower(),
            "trumpSuit": (
                state.trump_suit.name.lower()
                if state.trump_suit is not None
                else None
            ),
        }
    )


def _policy_snapshot(
    registry: AgentProfileRegistry,
    policy_id: str,
) -> dict[str, Any]:
    if policy_id == "heuristic":
        return {
            "id": "transparent_heuristic",
            "version": "1.0.0",
            "deterministic": True,
        }
    return registry.get(policy_id).to_dict()


def _choose(
    *,
    engine: BalootEngine,
    state,
    policy_id: str,
    registry: AgentProfileRegistry,
    convention: dict[str, Any],
    seed: int,
    cancel_check: Callable[[], bool] | None,
) -> tuple[Any, dict[str, Any]]:
    if cancel_check is not None and cancel_check():
        raise InterruptedError("paired benchmark cancelled")
    if policy_id == "heuristic":
        card = HeuristicBot(
            engine,
            BotProfile(deterministic_seed=seed),
        ).choose(state)
        return card, {
            "policyId": "heuristic",
            "decisionSource": "transparent_heuristic",
            "mistakeApplied": False,
            "deceptionApplied": False,
        }
    card, trace = AdaptivePlayStrategy(
        registry.get(policy_id),
        convention,
    ).choose_with_trace(
        engine,
        state,
        seed=seed,
        cancel_check=cancel_check,
    )
    require_uncensored_play_trace(trace)
    return card, {
        "policyId": policy_id,
        "decisionSource": trace["decisionSource"],
        "mistakeApplied": bool(trace["mistakeApplied"]),
        "deceptionApplied": bool(trace["deceptionApplied"]),
    }


def _play_round(
    initial_state,
    engine: BalootEngine,
    *,
    candidate_team: Team,
    candidate_id: str,
    baseline_id: str,
    registry: AgentProfileRegistry,
    convention: dict[str, Any],
    seed: int,
    cancel_check: Callable[[], bool] | None,
) -> dict[str, Any]:
    state = initial_state.clone()
    initial_state_sha256 = engine.state_hash(state)
    actions: list[dict[str, Any]] = []
    while state.current_player is not None:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("paired benchmark cancelled")
        seat = state.current_player
        policy_id = (
            candidate_id
            if seat.team == candidate_team
            else baseline_id
        )
        decision_seed = (
            seed * 1_000_003
            + len(state.played_cards) * 65_537
            + int(seat)
        )
        legal = tuple(card.code for card in engine.legal_actions(state))
        card, trace = _choose(
            engine=engine,
            state=state,
            policy_id=policy_id,
            registry=registry,
            convention=convention,
            seed=decision_seed,
            cancel_check=cancel_check,
        )
        if card.code not in legal:
            raise AssertionError("paired benchmark policy returned illegal card")
        actions.append(
            {
                "sequence": len(actions),
                "seat": seat.name.lower(),
                "team": seat.team.name.lower(),
                "policyId": policy_id,
                "legalActions": list(legal),
                "selectedAction": card.code,
                "decisionSeed": decision_seed,
                **trace,
            }
        )
        engine.play_card(state, card)
    if state.round_score is None:
        raise AssertionError("paired benchmark round did not score")
    candidate_score = state.round_score.awarded_points[int(candidate_team)]
    baseline_score = state.round_score.awarded_points[
        int(candidate_team.other)
    ]
    return {
        "candidateTeam": candidate_team.name.lower(),
        "initialStateSha256": initial_state_sha256,
        "finalStateSha256": engine.state_hash(state),
        "candidateScore": candidate_score,
        "baselineScore": baseline_score,
        "actions": actions,
        "roundScore": state.round_score.to_dict(),
    }


def run_paired_duplicate_rounds(
    *,
    root: str | Path | None = None,
    candidate_id: str,
    baseline_id: str,
    pair_count: int,
    seed_start: int,
    source_revision: str,
    source_tree: str,
    candidate_expected_regret: float | None = None,
    baseline_expected_regret: float | None = None,
    progress: Callable[[int, int], None] | None = None,
    maximum_seconds: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    if seed_start < 0:
        raise ValueError("seed_start cannot be negative")
    if candidate_id == baseline_id:
        raise ValueError("candidate and baseline policies must differ")
    if not source_revision.strip() or not source_tree.strip():
        raise ValueError("source revision and tree identities are required")
    resolved_root = platform_root(root)
    bundle: ProfileBundle = build_bundle(resolved_root)
    paired_config = bundle.component("benchmark")["paired_comparison"]
    resolved_maximum_seconds = float(
        paired_config["maximum_seconds"]
        if maximum_seconds is None
        else maximum_seconds
    )
    if resolved_maximum_seconds <= 0:
        raise ValueError("maximum_seconds must be positive")
    started = monotonic()

    def cancelled() -> bool:
        return bool(
            (cancel_check is not None and cancel_check())
            or monotonic() - started >= resolved_maximum_seconds
        )

    registry = AgentProfileRegistry.from_bundle(resolved_root)
    candidate_snapshot = _policy_snapshot(registry, candidate_id)
    baseline_snapshot = _policy_snapshot(registry, baseline_id)
    configuration_hash = canonical_sha256(
        {
            "bundleSha256": bundle.sha256,
            "runnerId": RUNNER_ID,
            "runnerVersion": RUNNER_VERSION,
            "sourceRevision": source_revision,
            "sourceTree": source_tree,
            "candidate": candidate_snapshot,
            "baseline": baseline_snapshot,
            "maximumSeconds": resolved_maximum_seconds,
        }
    )
    engine = BalootEngine(bundle.rules)
    pairs: list[dict[str, Any]] = []
    completion_status = "completed"
    for index in range(pair_count):
        if cancelled():
            completion_status = "safety_cancelled"
            break
        seed = seed_start + index
        dealer = Seat(index % 4)
        cut_position = (index * 17 % 31) + 1
        contract = Contract.SUN if index % 2 == 0 else Contract.HOKUM
        buyer = Seat((index * 3 + 1) % 4)
        trump = (
            None
            if contract == Contract.SUN
            else Suit((index * 5 + 1) % 4)
        )
        state = engine.create_round(seed=seed, dealer=dealer)
        engine.shuffle_and_initial_deal(
            state,
            cut_position=cut_position,
        )
        engine.bid_and_complete_deal(
            state,
            forced_contract=contract,
            forced_buyer=buyer,
            forced_trump=trump,
        )
        engine.prepare_projects(state)
        deal_hash = _deal_hash(state)
        try:
            games = [
                _play_round(
                    state,
                    engine,
                    candidate_team=team,
                    candidate_id=candidate_id,
                    baseline_id=baseline_id,
                    registry=registry,
                    convention=bundle.convention,
                    seed=seed + game_index * 10_000_019,
                    cancel_check=cancelled,
                )
                for game_index, team in enumerate(
                    (Team.NORTH_SOUTH, Team.WEST_EAST)
                )
            ]
        except InterruptedError:
            completion_status = "safety_cancelled"
            break
        pairs.append(
            {
                "pairId": f"pair-{seed:010d}",
                "candidateId": candidate_id,
                "baselineId": baseline_id,
                "dealHash": deal_hash,
                "configurationHash": configuration_hash,
                "rulesSha256": bundle.rules.sha256,
                "engineVersion": __version__,
                "seedGroup": f"paired-round-seed-{seed}",
                "recipe": {
                    "seed": seed,
                    "dealer": dealer.name.lower(),
                    "cutPosition": cut_position,
                    "contract": contract.value,
                    "buyer": buyer.name.lower(),
                    "trumpSuit": (
                        trump.name.lower() if trump is not None else None
                    ),
                },
                "games": games,
            }
        )
        if progress is not None:
            progress(len(pairs), pair_count)
    value: dict[str, Any] = {
        "schemaVersion": 1,
        "runnerId": RUNNER_ID,
        "runnerVersion": RUNNER_VERSION,
        "sourceRevision": source_revision,
        "sourceTree": source_tree,
        "candidateId": candidate_id,
        "baselineId": baseline_id,
        "requestedPairCount": pair_count,
        "candidatePolicySha256": canonical_sha256(candidate_snapshot),
        "baselinePolicySha256": canonical_sha256(baseline_snapshot),
        "configurationSha256": configuration_hash,
        "rulesSha256": bundle.rules.sha256,
        "bundleSha256": bundle.sha256,
        "pairCount": len(pairs),
        "completionStatus": completion_status,
        "maximumSeconds": resolved_maximum_seconds,
        "pairs": pairs,
        "automaticPromotion": False,
        "proofBoundary": (
            "Deterministic local play-only paired duplicate-round evidence "
            "with forced contracts and candidate-team swapping. It is not "
            "bidding strength, a full-match result, permanent-holdout proof, "
            "cross-hardware performance proof, external certification, or "
            "a promotion decision. The wall-clock limit is safety "
            "cancellation only; incomplete evidence cannot promote."
        ),
    }
    if (
        candidate_expected_regret is not None
        and baseline_expected_regret is not None
    ):
        benchmark = StrengthBenchmark(
            bundle.component("benchmark")
        )
        minimum_pairs = int(
            bundle.component("benchmark")["paired_comparison"][
                "minimum_pairs"
            ]
        )
        if completion_status != "completed":
            value["comparison"] = {
                "status": "incomplete_run",
                "completionStatus": completion_status,
                "minimumPairs": minimum_pairs,
                "observedPairs": len(pairs),
                "requestedPairs": pair_count,
            }
        elif len(pairs) >= minimum_pairs:
            value["comparison"] = benchmark.evaluate_symmetric_pairs(
                candidate_id,
                baseline_id,
                pairs,
                candidate_expected_regret=candidate_expected_regret,
                baseline_expected_regret=baseline_expected_regret,
            ).to_dict()
        else:
            value["comparison"] = {
                "status": "insufficient_pairs",
                "minimumPairs": minimum_pairs,
                "observedPairs": len(pairs),
            }
    value["evidenceSha256"] = canonical_sha256(value)
    return value


def serialize_paired_evidence(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
