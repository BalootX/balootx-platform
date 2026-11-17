from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml

from baloot_platform.config import platform_root
from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.rules import RuleBook

from .belief_v7 import (
    MAXIMUM_HIDDEN_DEAL_ATTEMPTS,
    BeliefPolicy,
    build_particle_belief,
)
from .deductions import derive_hard_deductions
from .match_value_v7 import MatchValueConfig, estimate_match_win_probability
from .models import ActionEvaluation, AnalysisResult, InformationState
from .posterior import weighted_mean
from .search import SearchResult, exact_team_value, heuristic_rollout_result
from .search_backend import ExactSearchBackend
from .signals import detect_signals

ANALYSIS_OBJECTIVES = frozenset(
    {
        "match_point_margin",
        "round_margin",
        "contract_success",
        "match_win_probability",
        "cvar",
        "low_volatility",
    }
)


@dataclass(frozen=True, slots=True)
class AnalysisPolicy:
    profile_id: str
    profile_version: str
    requested_samples: int
    exact_allocation_limit: int
    exact_search_max_remaining: int
    maximum_sampling_attempts: int
    objective: str

    @classmethod
    def from_profile(
        cls,
        profile: dict[str, Any],
        *,
        belief_policy: BeliefPolicy,
    ) -> "AnalysisPolicy":
        metadata = dict(profile.get("profile") or {})
        policy = cls(
            profile_id=str(metadata.get("id", "")).strip(),
            profile_version=str(metadata.get("version", "")).strip(),
            requested_samples=int(profile.get("requested_samples", 0)),
            exact_allocation_limit=int(
                profile.get("exact_allocation_limit", 0)
            ),
            exact_search_max_remaining=int(
                profile.get("exact_search_max_remaining", -1)
            ),
            maximum_sampling_attempts=int(
                profile.get("maximum_sampling_attempts", 0)
            ),
            objective=str(profile.get("objective", "")).strip(),
        )
        policy.validate_against(belief_policy)
        return policy

    def validate_against(self, belief_policy: BeliefPolicy) -> None:
        policy = self
        if not policy.profile_id or not policy.profile_version:
            raise ValueError("analysis profile identity is required")
        if not (
            1
            <= policy.requested_samples
            <= belief_policy.maximum_particles
        ):
            raise ValueError("analysis sample count violates belief bounds")
        if not (
            1
            <= policy.exact_allocation_limit
            <= belief_policy.maximum_exact_allocation_limit
        ):
            raise ValueError(
                "analysis exact-allocation limit violates safety bounds"
            )
        if not 0 <= policy.exact_search_max_remaining <= 32:
            raise ValueError(
                "analysis exact-search threshold must be in [0, 32]"
            )
        if not (
            policy.requested_samples
            <= policy.maximum_sampling_attempts
            <= MAXIMUM_HIDDEN_DEAL_ATTEMPTS
        ):
            raise ValueError(
                "analysis sampling attempts violate safety bounds"
            )
        if policy.objective not in ANALYSIS_OBJECTIVES:
            raise ValueError(
                f"unsupported analysis objective: {policy.objective}"
            )


@lru_cache(maxsize=8)
def _active_analysis_profile(resolved_root: str) -> dict[str, Any]:
    root = Path(resolved_root).resolve()
    bundle = yaml.safe_load(
        (root / "configs/bundle.yaml").read_text(encoding="utf-8")
    )
    relative = str(
        dict(bundle.get("components") or {}).get("analysis", "")
    ).strip()
    if not relative:
        raise ValueError("active bundle has no analysis profile")
    config_root = (root / "configs").resolve()
    profile_path = (config_root / relative).resolve()
    try:
        profile_path.relative_to(config_root)
    except ValueError as exc:
        raise ValueError(
            "analysis profile path escapes the configuration root"
        ) from exc
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError("active analysis profile must be a mapping")
    return profile


@lru_cache(maxsize=8)
def _active_analysis_policy(resolved_root: str) -> AnalysisPolicy:
    root = Path(resolved_root).resolve()
    return AnalysisPolicy.from_profile(
        _active_analysis_profile(str(root)),
        belief_policy=BeliefPolicy.from_profile(root=root),
    )


def analysis_objective_key(
    action: ActionEvaluation,
    objective: str,
) -> tuple[float, ...]:
    if objective in {"match_point_margin", "round_margin"}:
        primary = action.expected_round_margin
    elif objective == "contract_success":
        primary = action.contract_success_probability
    elif objective == "match_win_probability":
        primary = action.match_win_probability
    elif objective == "cvar":
        primary = action.downside_cvar
    elif objective == "low_volatility":
        primary = -action.volatility
    else:
        raise ValueError(f"unsupported analysis objective: {objective}")
    return (
        primary,
        action.match_win_probability,
        action.expected_round_margin,
        action.contract_success_probability,
        action.round_lead_probability,
        action.best_share,
        -float(action.card.id),
    )


def _objective_description(objective: str) -> str:
    return {
        "match_point_margin": "expected round margin",
        "round_margin": "expected round margin",
        "contract_success": "contract-success probability",
        "match_win_probability": "policy-conditioned match-win probability",
        "cvar": "lower-tail round margin",
        "low_volatility": "minimum outcome volatility",
    }[objective]


def _wilson_from_probability(probability: float, effective_n: float, z: float = 1.96) -> tuple[float, float]:
    if effective_n <= 0:
        return 0.0, 0.0
    p = min(1.0, max(0.0, probability))
    denominator = 1 + z * z / effective_n
    center = (p + z * z / (2 * effective_n)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * effective_n)) / effective_n) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _weighted_cvar(values: list[float], weights: list[float], fraction: float = 0.10) -> float:
    """Return the weighted lower-tail conditional mean.

    The implementation is deterministic and uses only the legal hidden-deal
    cohort shared by every candidate action.
    """
    if not values:
        return 0.0
    pairs = sorted(zip(values, weights), key=lambda item: item[0])
    total = sum(max(0.0, weight) for _, weight in pairs)
    if total <= 0:
        total = float(len(pairs))
        pairs = [(value, 1.0) for value, _ in pairs]
    target = max(1e-12, total * max(1e-6, min(1.0, fraction)))
    consumed = 0.0
    weighted = 0.0
    for value, weight in pairs:
        take = min(max(0.0, weight), target - consumed)
        if take > 0:
            weighted += value * take
            consumed += take
        if consumed >= target:
            break
    return weighted / consumed if consumed > 0 else float(pairs[0][0])


def _weighted_volatility(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    mean = weighted_mean(values, weights)
    total = sum(max(0.0, weight) for weight in weights)
    if total <= 0:
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    variance = sum(max(0.0, weight) * (value - mean) ** 2 for value, weight in zip(values, weights)) / total
    return math.sqrt(max(0.0, variance))


def analyze(
    info: InformationState,
    rules: RuleBook,
    convention: dict[str, Any] | None = None,
    *,
    requested_samples: int | None = None,
    exact_allocation_limit: int | None = None,
    exact_search_max_remaining: int | None = None,
    maximum_sampling_attempts: int | None = None,
    exact_search_max_nodes: int = 120_000,
    exact_search_max_seconds: float = 1.0,
    exact_search_max_depth: int | None = None,
    exact_search_max_cache_entries: int = 100_000,
    seed: int = 1,
    match_value_profile: dict[str, Any] | None = None,
    objective: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
    exact_search_backend: ExactSearchBackend | None = None,
    analysis_policy: AnalysisPolicy | None = None,
    belief_policy: BeliefPolicy | None = None,
) -> AnalysisResult:
    if cancel_check is not None and cancel_check():
        raise InterruptedError("analysis cancelled")
    resolved_belief_policy = (
        belief_policy
        if belief_policy is not None
        else BeliefPolicy.from_profile(root=platform_root())
    )
    if analysis_policy is not None:
        policy = analysis_policy
    elif belief_policy is None:
        policy = _active_analysis_policy(str(platform_root()))
    else:
        policy = AnalysisPolicy.from_profile(
            _active_analysis_profile(str(platform_root())),
            belief_policy=resolved_belief_policy,
        )
    policy.validate_against(resolved_belief_policy)
    requested_samples = (
        policy.requested_samples
        if requested_samples is None
        else int(requested_samples)
    )
    exact_allocation_limit = (
        policy.exact_allocation_limit
        if exact_allocation_limit is None
        else int(exact_allocation_limit)
    )
    exact_search_max_remaining = (
        policy.exact_search_max_remaining
        if exact_search_max_remaining is None
        else int(exact_search_max_remaining)
    )
    maximum_sampling_attempts = (
        policy.maximum_sampling_attempts
        if maximum_sampling_attempts is None
        else int(maximum_sampling_attempts)
    )
    if not 0 <= exact_search_max_remaining <= 32:
        raise ValueError(
            "exact-search remaining-card threshold must be in [0, 32]"
        )
    resolved_objective = (
        policy.objective
        if objective is None
        else str(objective).strip()
    )
    if resolved_objective not in ANALYSIS_OBJECTIVES:
        raise ValueError(
            f"unsupported analysis objective: {resolved_objective}"
        )
    engine = BalootEngine(rules)
    detected_observations = detect_signals(info, rules.play_domain)
    observations = (
        ()
        if (convention or {}).get("display_observations") is False
        else detected_observations
    )
    belief = build_particle_belief(
        info,
        engine,
        detected_observations,
        convention,
        particles=requested_samples,
        exact_allocation_limit=exact_allocation_limit,
        maximum_attempts=maximum_sampling_attempts,
        seed=seed,
        belief_policy=resolved_belief_policy,
        cancel_check=cancel_check,
    )
    if cancel_check is not None and cancel_check():
        raise InterruptedError("analysis cancelled")
    states = tuple(item.state for item in belief.particles)
    first = states[0]
    legal = engine.legal_moves(first)
    if not legal:
        raise ValueError("observer has no legal action")
    if first.current_player != info.observer:
        raise ValueError("analysis requires the observer to be the current player")

    weights = [item.weight for item in belief.particles]
    effective_n = belief.effective_sample_size
    conventions_applied = bool((convention or {}).get("apply_to_probability", False))

    margins: dict[Card, list[float]] = defaultdict(list)
    own_points: dict[Card, list[float]] = defaultdict(list)
    opponent_points: dict[Card, list[float]] = defaultdict(list)
    wins: dict[Card, list[float]] = defaultdict(list)
    draws: dict[Card, list[float]] = defaultdict(list)
    exact_flags: dict[Card, list[float]] = defaultdict(list)
    contract_successes: dict[Card, list[float]] = defaultdict(list)
    match_win_probabilities: dict[Card, list[float]] = defaultdict(list)
    best_flags: dict[Card, list[float]] = defaultdict(list)
    principal_variations: dict[Card, list[tuple[str, ...]]] = defaultdict(list)
    search_nodes: dict[Card, list[int]] = defaultdict(list)
    search_completion_statuses: dict[Card, list[str]] = defaultdict(list)
    match_config = MatchValueConfig.from_profile(match_value_profile)
    match_estimate_cache: dict[
        tuple[tuple[int, int], int],
        float,
    ] = {}
    exact_search = (
        exact_team_value
        if exact_search_backend is None
        else exact_search_backend.search
    )
    remaining_after_action = sum(mask.bit_count() for mask in first.hands) - 1
    batched_rollouts: dict[tuple[int, int], SearchResult] = {}
    full_rollout_batch = (
        None
        if exact_search_backend is None
        else getattr(exact_search_backend, "heuristic_rollout_batch", None)
    )
    compact_rollout_batch = (
        None
        if exact_search_backend is None
        else getattr(exact_search_backend, "heuristic_rollout_batch_compact", None)
    )
    rollout_details = (
        None
        if exact_search_backend is None
        else getattr(exact_search_backend, "heuristic_rollout_batch_details", None)
    )
    compact_rollouts_available = bool(
        exact_search_backend is not None
        and getattr(exact_search_backend, "supports_compact_rollout", False)
        and callable(compact_rollout_batch)
        and callable(rollout_details)
    )
    rollout_batch = (
        compact_rollout_batch if compact_rollouts_available else full_rollout_batch
    )
    compact_batch_used = False
    if remaining_after_action > exact_search_max_remaining and callable(rollout_batch):
        task_states = tuple(deal for deal in states for _card in legal)
        task_cards = tuple(card for _deal in states for card in legal)
        task_seeds = tuple(
            seed + deal_index * 97 + card.id
            for deal_index, _deal in enumerate(states)
            for card in legal
        )
        task_results = rollout_batch(
            task_states,
            task_cards,
            task_seeds,
            engine,
            info.observer.team,
            cancel_check=cancel_check,
        )
        if len(task_results) != len(task_states):
            raise RuntimeError("exact-search backend returned an incomplete rollout batch")
        compact_batch_used = compact_rollouts_available
        cursor = 0
        for deal_index, _deal in enumerate(states):
            for card in legal:
                batched_rollouts[(deal_index, card.id)] = task_results[cursor]
                cursor += 1

    for deal_index, deal in enumerate(states):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("analysis cancelled")
        per_deal: dict[Card, float] = {}
        per_outcome = {}
        for card in legal:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("analysis cancelled")
            outcome = batched_rollouts.get((deal_index, card.id))
            if outcome is None:
                child = deal.clone()
                child.events = []
                engine.play_card(
                    child,
                    card,
                    ekka=False,
                    emit_events=False,
                )
                remaining = sum(mask.bit_count() for mask in child.hands)
                if remaining <= exact_search_max_remaining:
                    outcome = exact_search(
                        child,
                        engine,
                        info.observer.team,
                        max_nodes=max(1, int(exact_search_max_nodes)),
                        max_seconds=max(0.001, float(exact_search_max_seconds)),
                        max_depth=(
                            None
                            if exact_search_max_depth is None
                            else max(1, int(exact_search_max_depth))
                        ),
                        max_cache_entries=max(
                            0,
                            int(exact_search_max_cache_entries),
                        ),
                        cancel_check=cancel_check,
                    )
                else:
                    outcome = heuristic_rollout_result(
                        child,
                        engine,
                        info.observer.team,
                        seed + deal_index * 97 + card.id,
                    )
            if outcome.completion_status == "cancelled":
                raise InterruptedError("analysis cancelled")
            per_deal[card] = outcome.value
            per_outcome[card] = outcome
            margins[card].append(outcome.value)
            own_points[card].append(outcome.own_points)
            opponent_points[card].append(outcome.opponent_points)
            wins[card].append(float(outcome.value > 0))
            draws[card].append(float(outcome.value == 0))
            exact_flags[card].append(float(outcome.exact))
            contract_successes[card].append(float(outcome.contract_succeeded))
            match_key = (
                tuple(outcome.score_after_round),
                deal_index,
            )
            match_probability = match_estimate_cache.get(match_key)
            if match_probability is None:
                match_probability = estimate_match_win_probability(
                    outcome.score_after_round,
                    info.observer.team,
                    config=match_config,
                    seed_parts=(
                        info.rules_hash,
                        int(info.observer),
                        deal_index,
                    ),
                ).probability
                match_estimate_cache[match_key] = match_probability
            match_win_probabilities[card].append(match_probability)
            if not compact_batch_used:
                principal_variations[card].append(tuple(outcome.principal_variation))
            search_nodes[card].append(int(outcome.nodes))
            search_completion_statuses[card].append(
                str(outcome.completion_status)
            )
        best = max(per_deal.values())
        for card, value in per_deal.items():
            best_flags[card].append(float(value == best))

    representative_indices = {
        card: min(
            range(len(margins[card])),
            key=lambda index, expected=weighted_mean(margins[card], weights): (
                abs(margins[card][index] - expected),
                -weights[index],
                index,
            ),
        )
        for card in legal
    }
    representative_details: dict[Card, SearchResult] = {}
    if compact_batch_used:
        assert callable(rollout_details)
        detail_states = tuple(states[representative_indices[card]] for card in legal)
        detail_cards = tuple(legal)
        detail_seeds = tuple(
            seed + representative_indices[card] * 97 + card.id for card in legal
        )
        details = rollout_details(
            detail_states,
            detail_cards,
            detail_seeds,
            engine,
            info.observer.team,
            cancel_check=cancel_check,
        )
        if len(details) != len(legal):
            raise RuntimeError("exact-search backend returned incomplete rollout details")
        for card, detail in zip(legal, details, strict=True):
            summary = batched_rollouts[(representative_indices[card], card.id)]
            summary_evidence = (
                summary.value,
                summary.own_points,
                summary.opponent_points,
                summary.exact,
                summary.nodes,
                summary.contract_succeeded,
                summary.round_lead,
                summary.score_after_round,
                summary.match_winner,
                summary.completion_status,
            )
            detail_evidence = (
                detail.value,
                detail.own_points,
                detail.opponent_points,
                detail.exact,
                detail.nodes,
                detail.contract_succeeded,
                detail.round_lead,
                detail.score_after_round,
                detail.match_winner,
                detail.completion_status,
            )
            if summary_evidence != detail_evidence:
                raise RuntimeError(
                    "compact rollout detail did not match its numeric evidence"
                )
            representative_details[card] = detail

    evaluations: list[ActionEvaluation] = []
    for card in legal:
        win_probability = weighted_mean(wins[card], weights)
        expected_margin = weighted_mean(margins[card], weights)
        low, high = _wilson_from_probability(win_probability, effective_n)
        representative_index = representative_indices[card]
        representative_detail = representative_details.get(card)
        match_probability = weighted_mean(match_win_probabilities[card], weights)
        match_low, match_high = _wilson_from_probability(match_probability, effective_n)
        evaluations.append(
            ActionEvaluation(
                card=card,
                expected_round_margin=expected_margin,
                expected_team_awarded_points=weighted_mean(own_points[card], weights),
                expected_opponent_awarded_points=weighted_mean(opponent_points[card], weights),
                round_lead_probability=win_probability,
                draw_probability=weighted_mean(draws[card], weights),
                confidence_low=low,
                confidence_high=high,
                best_share=weighted_mean(best_flags[card], weights),
                samples=len(states),
                effective_samples=effective_n,
                exact_share=weighted_mean(exact_flags[card], weights),
                contract_success_probability=weighted_mean(contract_successes[card], weights),
                match_win_probability=match_probability,
                match_confidence_low=match_low,
                match_confidence_high=match_high,
                match_value_model="policy_conditioned_exact_dynamic_program_v2",
                deal_margins=tuple(float(value) for value in margins[card]),
                downside_cvar=_weighted_cvar(margins[card], weights),
                volatility=_weighted_volatility(margins[card], weights),
                principal_variation=(
                    representative_detail.principal_variation
                    if representative_detail is not None
                    else principal_variations[card][representative_index]
                ),
                search_nodes=(
                    representative_detail.nodes
                    if representative_detail is not None
                    else search_nodes[card][representative_index]
                ),
                search_completion_status=(
                    representative_detail.completion_status
                    if representative_detail is not None
                    else search_completion_statuses[card][representative_index]
                ),
            )
        )

    evaluations.sort(
        key=lambda action: analysis_objective_key(
            action,
            resolved_objective,
        ),
        reverse=True,
    )
    recommended = evaluations[0].card
    gap = (
        evaluations[0].expected_round_margin - evaluations[1].expected_round_margin
        if len(evaluations) > 1
        else 99.0
    )
    uncertainty = max(0.0, 1.0 - min(1.0, effective_n / 100.0))
    difficulty = max(
        1,
        min(
            100,
            round(
                24
                + 12 * (len(evaluations) - 1)
                + max(0, 34 - 4 * abs(gap))
                + 12 * uncertainty
                + (8 if observations else 0)
            ),
        ),
    )
    debate = (
        max(1, min(100, round(100 - min(80, 8 * abs(gap)))))
        if len(evaluations) > 1
        else 5
    )
    method = belief.method + ("+weighted_conventions" if conventions_applied else "+hard_facts_only")
    explanation = (
        f"Compared {len(evaluations)} legal actions across {len(states)} legal hidden deals.",
        f"Hidden-deal method: {belief.method}; effective sample size {effective_n:.1f}.",
        "The exposed purchase card and revealed projects are pinned to their known owners.",
        "Opponent legal alternatives are reconstructed per hypothetical deal and never copied from the true hidden hands.",
        "Hard deductions restrict legality; convention evidence only reweights legal deals when explicitly enabled.",
        (
            f"Recommended {recommended.code} by the configured objective: "
            f"{_objective_description(resolved_objective)}."
        ),
        "Round lead probability, contract success, and match-to-target probability are reported separately.",
    )
    return AnalysisResult(
        info.observer,
        recommended,
        tuple(evaluations),
        derive_hard_deductions(info),
        observations,
        method,
        len(states),
        effective_n,
        difficulty,
        debate,
        conventions_applied,
        explanation,
        resolved_objective,
        policy.profile_id,
        policy.profile_version,
    )
