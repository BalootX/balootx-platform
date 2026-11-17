"""Honest local benchmark for the reduced proof kernel and current analyzer."""

from __future__ import annotations

import platform
import statistics
import time
from dataclasses import dataclass
from typing import Any, Mapping

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
class PolicyProofBenchmarkCaseV1:
    contract: str
    fixture_state_hash: str
    actor: str
    legal_actions: tuple[str, ...]
    kernel_median_ms: float
    current_analysis_median_ms: float
    kernel_recommendation: str
    current_analysis_recommendation: str
    recommendation_agreement: bool
    kernel_truth: str
    current_analysis_method: str
    kernel_backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "fixtureStateHash": self.fixture_state_hash,
            "actor": self.actor,
            "legalActions": list(self.legal_actions),
            "kernelMedianMs": self.kernel_median_ms,
            "currentAnalysisMedianMs": self.current_analysis_median_ms,
            "kernelRecommendation": self.kernel_recommendation,
            "currentAnalysisRecommendation": self.current_analysis_recommendation,
            "recommendationAgreement": self.recommendation_agreement,
            "kernelTruth": self.kernel_truth,
            "currentAnalysisMethod": self.current_analysis_method,
            "kernelBackend": self.kernel_backend,
        }


def authoritative_late_game_fixture(
    rules: RuleBook,
    contract: Contract,
    *,
    seed: int = 1,
) -> tuple[BalootEngine, RoundState]:
    """Create the governed حكم/صن two-card fixture through normal play."""
    if contract not in {Contract.HOKUM, Contract.SUN}:
        raise ValueError("benchmark fixture supports only Hokum and Sun")
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
    while len(state.played_cards) < 24:
        engine.play_card(state, bot.choose(state))
    if state.current_player is None or len(engine.legal_actions(state)) < 2:
        raise AssertionError("governed late-game fixture is no longer non-forced")
    return engine, state


def benchmark_reduced_policy_proof(
    rules: RuleBook,
    convention: Mapping[str, Any],
    *,
    iterations: int = 3,
) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("benchmark iterations must be positive")
    cases: list[PolicyProofBenchmarkCaseV1] = []
    for contract in (Contract.HOKUM, Contract.SUN):
        engine, state = authoritative_late_game_fixture(rules, contract)
        assert state.current_player is not None
        kernel_times: list[float] = []
        analysis_times: list[float] = []
        kernel_action = None
        current = None
        for _ in range(iterations):
            started = time.perf_counter_ns()
            cohort = generate_hidden_deals(
                InformationState.from_complete(state, state.current_player),
                engine,
                requested_samples=1,
                exact_allocation_limit=1,
                maximum_attempts=100,
                seed=1,
            )
            problem = EngineReducedBestResponseProblemBuilderV1().build(
                states=cohort.states,
                engine=engine,
                actor=state.current_player,
                coverage_method=cohort.method + "+hard_facts_only",
                profile=ReducedBestResponseProfileV1(maximum_nodes=20_000),
                maximum_plies=8,
            )
            proof = OrToolsBoundedBestResponseProofV1().prove(problem)
            kernel_times.append((time.perf_counter_ns() - started) / 1_000_000)
            kernel_action = proof.policy_proof.selected_action

            started = time.perf_counter_ns()
            current = analyze(
                InformationState.from_complete(state, state.current_player),
                rules,
                dict(convention),
                requested_samples=1,
                exact_allocation_limit=1,
                exact_search_max_remaining=8,
                maximum_sampling_attempts=100,
                exact_search_max_seconds=2.0,
                seed=1,
            )
            analysis_times.append((time.perf_counter_ns() - started) / 1_000_000)
        assert kernel_action is not None and current is not None
        cases.append(
            PolicyProofBenchmarkCaseV1(
                contract.value,
                engine.state_hash(state),
                state.current_player.name.lower(),
                tuple(card.code for card in engine.legal_actions(state)),
                round(statistics.median(kernel_times), 3),
                round(statistics.median(analysis_times), 3),
                kernel_action.code,
                current.recommended_card.code,
                kernel_action == current.recommended_card,
                proof.policy_proof.strategic_truth,
                current.method,
                problem.profile.solver_backend,
            )
        )
    payload = {
        "schemaVersion": 1,
        "benchmark": "reduced_policy_proof_vs_current_analysis",
        "scopeNote": (
            "Both paths use the same governed observer-safe hidden-deal generator, seed, "
            "one requested sample, exact-allocation limit, and attempt budget. The kernel "
            "method and truth remain explicit per case. Timing and recommendation "
            "differences are matched observations, not publication qualification."
        ),
        "iterations": iterations,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "rulesHash": rules.sha256,
        "cases": [case.to_dict() for case in cases],
    }
    payload["evidenceHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "environment"}
    )
    return payload
