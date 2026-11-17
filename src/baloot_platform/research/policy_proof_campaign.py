"""Deterministic bounded qualification campaign for reduced policy proofs."""

from __future__ import annotations

import math
import platform
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from baloot_platform.analysis.evaluator import analyze
from baloot_platform.analysis.hidden_deals import generate_hidden_deals
from baloot_platform.analysis.models import InformationState
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.models import RoundState
from baloot_platform.core.rules import RuleBook

from .best_response_proof import (
    EngineReducedBestResponseProblemBuilderV1,
    OrToolsBoundedBestResponseProofV1,
    ReducedBestResponseProfileV1,
)
from .contracts import canonical_sha256


@dataclass(frozen=True, slots=True)
class PolicyProofCampaignProfileV1:
    profile_id: str = "bounded_late_game_qualification"
    revision: int = 1
    played_card_depths: tuple[int, ...] = (20, 24)
    seeds_per_cell: int = 2
    requested_samples: int = 1
    exact_allocation_limit: int = 1
    maximum_sampling_attempts: int = 100
    maximum_fixture_seed_attempts: int = 512
    maximum_nodes: int = 50_000
    maximum_case_seconds: float = 5.0
    current_exact_search_remaining: int = 8
    current_exact_search_seconds: float = 0.5
    promotion_minimum_cases: int = 24
    promotion_minimum_agreement: float = 0.99
    promotion_minimum_yield_ratio: float = 2.0
    promotion_minimum_cost_reduction: float = 0.40

    def __post_init__(self) -> None:
        if not self.profile_id or self.revision < 1:
            raise ValueError("campaign profile identity is invalid")
        if not self.played_card_depths or any(
            not 0 <= depth <= 28 for depth in self.played_card_depths
        ):
            raise ValueError("campaign played-card depths are invalid")
        if min(
            self.seeds_per_cell,
            self.requested_samples,
            self.exact_allocation_limit,
            self.maximum_sampling_attempts,
            self.maximum_fixture_seed_attempts,
            self.maximum_nodes,
            self.promotion_minimum_cases,
        ) < 1:
            raise ValueError("campaign count budgets must be positive")
        if self.maximum_case_seconds <= 0 or self.current_exact_search_seconds <= 0:
            raise ValueError("campaign time budgets must be positive")
        if not 0 <= self.promotion_minimum_agreement <= 1:
            raise ValueError("campaign agreement threshold is invalid")
        if self.promotion_minimum_yield_ratio <= 0:
            raise ValueError("campaign yield threshold must be positive")
        if not 0 <= self.promotion_minimum_cost_reduction < 1:
            raise ValueError("campaign cost threshold is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "profileId": self.profile_id,
            "revision": self.revision,
            "playedCardDepths": list(self.played_card_depths),
            "seedsPerCell": self.seeds_per_cell,
            "requestedSamples": self.requested_samples,
            "exactAllocationLimit": self.exact_allocation_limit,
            "maximumSamplingAttempts": self.maximum_sampling_attempts,
            "maximumFixtureSeedAttempts": self.maximum_fixture_seed_attempts,
            "maximumNodes": self.maximum_nodes,
            "maximumCaseSeconds": self.maximum_case_seconds,
            "currentExactSearchRemaining": self.current_exact_search_remaining,
            "currentExactSearchSeconds": self.current_exact_search_seconds,
            "promotionMinimumCases": self.promotion_minimum_cases,
            "promotionMinimumAgreement": self.promotion_minimum_agreement,
            "promotionMinimumYieldRatio": self.promotion_minimum_yield_ratio,
            "promotionMinimumCostReduction": self.promotion_minimum_cost_reduction,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _fixture(
    rules: RuleBook,
    *,
    contract: Contract,
    actor: Seat,
    played_cards: int,
    seed_start: int,
    maximum_attempts: int,
) -> tuple[BalootEngine, RoundState, int]:
    for offset in range(maximum_attempts):
        seed = seed_start + offset
        engine = BalootEngine(rules)
        state = engine.create_round(seed=seed, dealer=Seat.EAST)
        engine.shuffle_and_initial_deal(state)
        engine.bid_and_complete_deal(
            state,
            forced_contract=contract,
            forced_buyer=Seat.SOUTH,
            forced_trump=Suit.HEARTS if contract == Contract.HOKUM else None,
        )
        engine.prepare_projects(state)
        bot = HeuristicBot(engine, BotProfile(seed))
        while len(state.played_cards) < played_cards:
            engine.play_card(state, bot.choose(state))
        if state.current_player == actor and len(engine.legal_actions(state)) >= 2:
            return engine, state, seed
    raise RuntimeError("no deterministic non-forced fixture found within the seed budget")


def _opportunities(problem, proof) -> tuple[str, ...]:
    selected = proof.policy_proof.selected_action
    roots = problem.worlds[0].root_actions
    selected_root = next(item for item in roots if item.action == selected)
    action_evidence = proof.policy_proof.action_evidence
    selected_evidence = next(item for item in action_evidence if item.action == selected)
    alternative_worst = max(
        item.worst_margin for item in action_evidence if item.action != selected
    )
    tags = {"only_move_opportunity"}
    if selected_root.strength_percentile_bps <= 3_500:
        tags.add("small_card_opportunity")
    if selected_root.strength_percentile_bps >= 6_500:
        tags.add("large_card_sacrifice_opportunity")
    if selected_evidence.worst_margin > alternative_worst:
        tags.add("material_counterfactual_gap")
    return tuple(sorted(tags))


def run_policy_proof_qualification_campaign(
    rules: RuleBook,
    convention: Mapping[str, Any],
    *,
    profile: PolicyProofCampaignProfileV1 | None = None,
) -> dict[str, Any]:
    cfg = profile or PolicyProofCampaignProfileV1()
    cases: list[dict[str, Any]] = []
    campaign_wall_start = time.perf_counter_ns()
    campaign_cpu_start = time.process_time_ns()
    case_index = 0
    for contract in (Contract.HOKUM, Contract.SUN):
        for actor in Seat:
            for depth in cfg.played_card_depths:
                for repetition in range(cfg.seeds_per_cell):
                    case_index += 1
                    identity = {
                        "contract": contract.value,
                        "actor": actor.name.lower(),
                        "playedCards": depth,
                        "repetition": repetition,
                    }
                    seed_start = 1 + int(canonical_sha256(identity)[:8], 16)
                    try:
                        engine, state, seed = _fixture(
                            rules,
                            contract=contract,
                            actor=actor,
                            played_cards=depth,
                            seed_start=seed_start,
                            maximum_attempts=cfg.maximum_fixture_seed_attempts,
                        )
                        information = InformationState.from_complete(state, actor)

                        kernel_wall_start = time.perf_counter_ns()
                        kernel_cpu_start = time.process_time_ns()
                        cohort = generate_hidden_deals(
                            information,
                            engine,
                            requested_samples=cfg.requested_samples,
                            exact_allocation_limit=cfg.exact_allocation_limit,
                            maximum_attempts=cfg.maximum_sampling_attempts,
                            seed=seed,
                        )
                        deadline = time.monotonic() + cfg.maximum_case_seconds
                        problem = EngineReducedBestResponseProblemBuilderV1().build(
                            states=cohort.states,
                            engine=engine,
                            actor=actor,
                            coverage_method=cohort.method + "+hard_facts_only",
                            profile=ReducedBestResponseProfileV1(
                                maximum_nodes=cfg.maximum_nodes,
                            ),
                            maximum_plies=32 - depth,
                            cancel_check=lambda: time.monotonic() >= deadline,
                        )
                        proof = OrToolsBoundedBestResponseProofV1().prove(
                            problem,
                            cancel_check=lambda: time.monotonic() >= deadline,
                            maximum_seconds_per_subproblem=cfg.maximum_case_seconds,
                        )
                        kernel_wall = (time.perf_counter_ns() - kernel_wall_start) / 1e9
                        kernel_cpu = (time.process_time_ns() - kernel_cpu_start) / 1e9

                        analysis_wall_start = time.perf_counter_ns()
                        analysis_cpu_start = time.process_time_ns()
                        current = analyze(
                            information,
                            rules,
                            dict(convention),
                            requested_samples=cfg.requested_samples,
                            exact_allocation_limit=cfg.exact_allocation_limit,
                            exact_search_max_remaining=cfg.current_exact_search_remaining,
                            maximum_sampling_attempts=cfg.maximum_sampling_attempts,
                            exact_search_max_seconds=cfg.current_exact_search_seconds,
                            seed=seed,
                        )
                        analysis_wall = (time.perf_counter_ns() - analysis_wall_start) / 1e9
                        analysis_cpu = (time.process_time_ns() - analysis_cpu_start) / 1e9

                        kernel_exact = (
                            proof.policy_proof.strategic_truth
                            == "exact_reduced_actor_safe"
                        )
                        current_exact = bool(
                            current.method.startswith("exhaustive_")
                            and current.actions
                            and all(
                                item.search_completion_status == "exact"
                                and item.exact_share >= 1.0 - 1e-12
                                for item in current.actions
                            )
                        )
                        agreement = (
                            proof.policy_proof.selected_action == current.recommended_card
                        )
                        cases.append(
                            {
                                **identity,
                                "caseIndex": case_index,
                                "seed": seed,
                                "stateHash": engine.state_hash(state),
                                "informationStateHash": problem.information_state_hash,
                                "cohortMethod": cohort.method,
                                "currentCohortMethod": current.method,
                                "matchedCohortMethod": (
                                    current.method.split("+", 1)[0] == cohort.method
                                ),
                                "cohortSize": len(cohort.states),
                                "legalActionCount": len(engine.legal_actions(state)),
                                "opportunities": list(_opportunities(problem, proof)),
                                "provenTactics": [
                                    kind
                                    for kind, present in (
                                        (
                                            "only_winning_move",
                                            proof.policy_proof.only_winning_move is not None,
                                        ),
                                        (
                                            "small_card_big_gain",
                                            proof.small_card_big_gain is not None,
                                        ),
                                    )
                                    if present
                                ],
                                "kernelRecommendation": proof.policy_proof.selected_action.code,
                                "currentRecommendation": current.recommended_card.code,
                                "agreement": agreement,
                                "kernelTruth": proof.policy_proof.strategic_truth,
                                "currentExact": current_exact,
                                "kernelQualified": kernel_exact and agreement,
                                "currentQualified": current_exact,
                                "kernelWallSeconds": kernel_wall,
                                "kernelCpuSeconds": kernel_cpu,
                                "currentWallSeconds": analysis_wall,
                                "currentCpuSeconds": analysis_cpu,
                                "status": "completed",
                            }
                        )
                    except Exception as exc:
                        cases.append(
                            {
                                **identity,
                                "caseIndex": case_index,
                                "status": "failed",
                                "failureType": type(exc).__name__,
                                "failure": str(exc),
                            }
                        )

    completed = [case for case in cases if case["status"] == "completed"]
    failures = [case for case in cases if case["status"] == "failed"]
    kernel_wall_values = [case["kernelWallSeconds"] for case in completed]
    current_wall_values = [case["currentWallSeconds"] for case in completed]
    kernel_cpu = sum(case["kernelCpuSeconds"] for case in completed)
    current_cpu = sum(case["currentCpuSeconds"] for case in completed)
    kernel_qualified = sum(case["kernelQualified"] for case in completed)
    current_qualified = sum(case["currentQualified"] for case in completed)
    agreements = sum(case["agreement"] for case in completed)
    agreement_rate = agreements / len(completed) if completed else 0.0
    kernel_yield = kernel_qualified / kernel_cpu * 3600 if kernel_cpu else 0.0
    current_yield = current_qualified / current_cpu * 3600 if current_cpu else 0.0
    yield_ratio = kernel_yield / current_yield if current_yield > 0 else None
    kernel_cost = kernel_cpu / kernel_qualified if kernel_qualified else None
    current_cost = current_cpu / current_qualified if current_qualified else None
    cost_reduction = (
        1.0 - kernel_cost / current_cost
        if kernel_cost is not None and current_cost is not None and current_cost > 0
        else None
    )
    enough_cases = len(completed) >= cfg.promotion_minimum_cases
    performance_pass = bool(
        (
            yield_ratio is not None
            and yield_ratio >= cfg.promotion_minimum_yield_ratio
        )
        or (
            cost_reduction is not None
            and cost_reduction >= cfg.promotion_minimum_cost_reduction
        )
    )
    promote = bool(
        enough_cases
        and not failures
        and agreement_rate >= cfg.promotion_minimum_agreement
        and performance_pass
    )
    summary = {
        "requestedCases": len(cases),
        "completedCases": len(completed),
        "failedCases": len(failures),
        "agreementCount": agreements,
        "agreementRate": agreement_rate,
        "kernelExactCount": sum(
            case["kernelTruth"] == "exact_reduced_actor_safe" for case in completed
        ),
        "kernelProvisionalCount": sum(
            case["kernelTruth"] != "exact_reduced_actor_safe" for case in completed
        ),
        "kernelQualifiedCount": kernel_qualified,
        "currentExactCount": sum(case["currentExact"] for case in completed),
        "currentProvisionalCount": sum(
            not case["currentExact"] for case in completed
        ),
        "currentQualifiedCount": current_qualified,
        "kernelWallP50Ms": _percentile(kernel_wall_values, 0.50) * 1000,
        "kernelWallP95Ms": _percentile(kernel_wall_values, 0.95) * 1000,
        "currentWallP50Ms": _percentile(current_wall_values, 0.50) * 1000,
        "currentWallP95Ms": _percentile(current_wall_values, 0.95) * 1000,
        "kernelCpuSeconds": kernel_cpu,
        "currentCpuSeconds": current_cpu,
        "kernelQualifiedPerCpuHour": kernel_yield,
        "currentQualifiedPerCpuHour": current_yield,
        "qualifiedYieldRatio": yield_ratio,
        "kernelCpuSecondsPerQualified": kernel_cost,
        "currentCpuSecondsPerQualified": current_cost,
        "qualifiedCostReduction": cost_reduction,
        "opportunityCoverage": sorted(
            {tag for case in completed for tag in case["opportunities"]}
        ),
        "provenTacticCoverage": sorted(
            {tag for case in completed for tag in case["provenTactics"]}
        ),
        "failureCounts": {
            kind: sum(case.get("failureType") == kind for case in failures)
            for kind in sorted({case.get("failureType") for case in failures})
        },
    }
    wall_seconds = (time.perf_counter_ns() - campaign_wall_start) / 1e9
    cpu_seconds = (time.process_time_ns() - campaign_cpu_start) / 1e9
    normalized_cases = [
        {
            key: value
            for key, value in case.items()
            if not key.endswith("Seconds")
        }
        for case in cases
    ]
    normalized_core = {
        "schemaVersion": 1,
        "profileHash": cfg.sha256,
        "rulesHash": rules.sha256,
        "cases": normalized_cases,
        "promotion": {
            "promote": promote,
            "minimumCasesPassed": enough_cases,
            "agreementPassed": agreement_rate >= cfg.promotion_minimum_agreement,
            "performancePassed": performance_pass,
            "failureGatePassed": not failures,
        },
    }
    return {
        "schemaVersion": 1,
        "campaign": "reduced_policy_proof_qualification",
        "profile": cfg.to_dict(),
        "profileHash": cfg.sha256,
        "rulesHash": rules.sha256,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "cases": cases,
        "summary": summary,
        "campaignWallSeconds": wall_seconds,
        "campaignCpuSeconds": cpu_seconds,
        "promotion": normalized_core["promotion"],
        "decision": "eligible_for_integration" if promote else "verification_only",
        "normalizedEvidenceHash": canonical_sha256(normalized_core),
    }
