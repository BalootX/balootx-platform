"""Decision interestingness metrics for Baloot challenge curation.

This module deliberately contains no game rules.  It consumes normalized action
outcomes produced by an authoritative evaluator and returns auditable metrics.
The same metrics work for play, buy/pass, escalation, and project decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import exp, log
from statistics import fmean, pstdev
from typing import Iterable, Mapping, Sequence


class ChallengeArchetype(str, Enum):
    CLIFF = "cliff"
    DEBATE = "debate"
    COMEBACK = "comeback"
    BUY_OR_PASS = "buy_or_pass"
    PROJECT_DISCLOSURE = "project_disclosure"
    ESCALATION = "escalation"
    SIGNAL_READING = "signal_reading"
    ENDGAME = "endgame"
    GENERAL = "general"


@dataclass(frozen=True)
class ActionOutcome:
    action_id: str
    expected_match_utility: float
    match_win_probability: float
    expected_round_margin: float
    downside_cvar: float = 0.0
    best_deal_share: float = 0.0
    hidden_deal_values: tuple[float, ...] = ()
    is_obvious_heuristic_choice: bool = False
    explanation_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextFeatures:
    decision_kind: str = "play"
    contract: str = "unknown"
    buyer_relationship: str = "unknown"
    opponent_bought: bool = False
    score_pressure: float = 0.0
    tricks_remaining: int = 8
    depends_on_signal: bool = False
    counterintuitive: bool = False
    project_relevant: bool = False
    escalation_relevant: bool = False
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class InterestingnessVector:
    best_action_id: str
    second_action_id: str | None
    action_count: int
    regret_gap: float
    win_probability_gap: float
    disaster_gap: float
    cliff_score: float
    debate_score: float
    comeback_score: float
    surprise_score: float
    hidden_information_score: float
    educational_score: float
    robustness_score: float
    difficulty_prior: float
    publishability_score: float
    archetype_scores: Mapping[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def _normalized_entropy(weights: Sequence[float]) -> float:
    clean = [max(0.0, float(v)) for v in weights]
    total = sum(clean)
    if total <= 0 or len(clean) <= 1:
        return 0.0
    probs = [v / total for v in clean if v > 0]
    entropy = -sum(p * log(p) for p in probs)
    return _clip01(entropy / log(len(clean)))


def _deal_disagreement(actions: Sequence[ActionOutcome]) -> float:
    lengths = [len(a.hidden_deal_values) for a in actions if a.hidden_deal_values]
    if len(lengths) < 2 or min(lengths) == 0:
        return 0.0
    n = min(lengths)
    winners: list[str] = []
    margins: list[float] = []
    for i in range(n):
        ranked = sorted(
            ((a.hidden_deal_values[i], a.action_id) for a in actions if len(a.hidden_deal_values) >= n),
            reverse=True,
        )
        if not ranked:
            continue
        winners.append(ranked[0][1])
        if len(ranked) > 1:
            margins.append(ranked[0][0] - ranked[1][0])
    if not winners:
        return 0.0
    counts: dict[str, int] = {}
    for winner in winners:
        counts[winner] = counts.get(winner, 0) + 1
    entropy = _normalized_entropy(list(counts.values()))
    close = 1.0 - _clip01(fmean(margins) / 0.25) if margins else 0.0
    return _clip01(0.7 * entropy + 0.3 * close)


def score_interestingness(
    outcomes: Iterable[ActionOutcome],
    context: ContextFeatures | None = None,
) -> InterestingnessVector:
    """Score one decision point using evaluator outputs.

    Utilities are expected to be normalized to roughly ``[-1, 1]``.  The
    function remains bounded if callers provide larger values.  No hidden hand
    is inspected here; per-deal vectors must already be observer-safe evaluator
    outputs.
    """
    context = context or ContextFeatures()
    actions = sorted(
        list(outcomes),
        key=lambda a: (a.expected_match_utility, a.match_win_probability, a.expected_round_margin),
        reverse=True,
    )
    if len(actions) < 2:
        raise ValueError("at least two legal actions are required for a challenge")

    best, second, worst = actions[0], actions[1], actions[-1]
    regret_gap = max(0.0, best.expected_match_utility - second.expected_match_utility)
    win_gap = max(0.0, best.match_win_probability - second.match_win_probability)
    disaster_gap = max(0.0, best.expected_match_utility - worst.expected_match_utility)

    flips_outcome = best.match_win_probability >= 0.55 and worst.match_win_probability <= 0.35
    crosses_even = best.expected_match_utility > 0 >= worst.expected_match_utility
    catastrophic_tail = max(0.0, best.downside_cvar - worst.downside_cvar)

    cliff_score = _clip01(
        0.35 * _sigmoid((disaster_gap - 0.30) * 7.0)
        + 0.30 * _sigmoid((best.match_win_probability - worst.match_win_probability - 0.25) * 8.0)
        + 0.20 * (1.0 if flips_outcome or crosses_even else 0.0)
        + 0.15 * _sigmoid((catastrophic_tail - 0.15) * 8.0)
    )

    closeness = 1.0 - _clip01(regret_gap / 0.18)
    deal_disagreement = _deal_disagreement(actions)
    best_share_entropy = _normalized_entropy([max(0.001, a.best_deal_share) for a in actions])
    debate_score = _clip01(0.48 * closeness + 0.37 * deal_disagreement + 0.15 * best_share_entropy)

    comeback_score = _clip01(
        (0.30 if context.opponent_bought else 0.0)
        + 0.35 * _sigmoid((best.match_win_probability - 0.55) * 8.0)
        + 0.20 * (1.0 if crosses_even else 0.0)
        + 0.15 * _clip01(context.score_pressure)
    )

    obvious = next((a for a in actions if a.is_obvious_heuristic_choice), None)
    surprise_gap = 0.0 if obvious is None else max(0.0, best.expected_match_utility - obvious.expected_match_utility)
    surprise_score = _clip01(
        0.65 * _sigmoid((surprise_gap - 0.08) * 12.0)
        + 0.20 * (1.0 if context.counterintuitive else 0.0)
        + 0.15 * (1.0 if best.action_id != getattr(obvious, "action_id", best.action_id) else 0.0)
    )

    hidden_information_score = _clip01(
        0.65 * deal_disagreement
        + 0.20 * (1.0 if context.depends_on_signal else 0.0)
        + 0.15 * (1.0 if any(a.hidden_deal_values for a in actions) else 0.0)
    )

    explanation_union = set().union(*(set(a.explanation_features) for a in actions))
    educational_score = _clip01(
        0.18 * min(4, len(explanation_union)) / 4
        + 0.20 * (1.0 if context.project_relevant else 0.0)
        + 0.20 * (1.0 if context.escalation_relevant else 0.0)
        + 0.20 * (1.0 if context.depends_on_signal else 0.0)
        + 0.22 * min(1.0, context.tricks_remaining / 5.0)
    )

    samples = [a.hidden_deal_values for a in actions if a.hidden_deal_values]
    volatility = 0.0
    if samples:
        stdevs = [pstdev(v) for v in samples if len(v) > 1]
        volatility = fmean(stdevs) if stdevs else 0.0
    robustness_score = _clip01(1.0 - volatility / 0.55)

    cognitive_load = _clip01(
        0.18 * min(1.0, (len(actions) - 1) / 4.0)
        + 0.20 * min(1.0, context.tricks_remaining / 6.0)
        + 0.22 * hidden_information_score
        + 0.20 * surprise_score
        + 0.20 * (1.0 if context.depends_on_signal else 0.0)
    )
    difficulty_prior = 100.0 * cognitive_load

    archetypes = {
        ChallengeArchetype.CLIFF.value: cliff_score,
        ChallengeArchetype.DEBATE.value: debate_score,
        ChallengeArchetype.COMEBACK.value: comeback_score,
        ChallengeArchetype.BUY_OR_PASS.value: _clip01(
            0.60 * cliff_score + 0.25 * debate_score + 0.15 * surprise_score
        ) if context.decision_kind == "bid" else 0.0,
        ChallengeArchetype.PROJECT_DISCLOSURE.value: _clip01(
            0.45 * cliff_score + 0.30 * hidden_information_score + 0.25 * educational_score
        ) if context.decision_kind == "project" else 0.0,
        ChallengeArchetype.ESCALATION.value: _clip01(
            0.50 * cliff_score + 0.30 * comeback_score + 0.20 * debate_score
        ) if context.decision_kind == "escalation" else 0.0,
        ChallengeArchetype.SIGNAL_READING.value: _clip01(
            0.45 * hidden_information_score + 0.30 * debate_score + 0.25 * educational_score
        ) if context.depends_on_signal else 0.0,
        ChallengeArchetype.ENDGAME.value: _clip01(
            0.55 * cliff_score + 0.25 * surprise_score + 0.20 * robustness_score
        ) if context.tricks_remaining <= 3 else 0.0,
    }
    archetypes[ChallengeArchetype.GENERAL.value] = _clip01(
        0.30 * cliff_score
        + 0.20 * debate_score
        + 0.15 * surprise_score
        + 0.15 * educational_score
        + 0.10 * hidden_information_score
        + 0.10 * robustness_score
    )
    publishability = max(archetypes.values())

    reasons: list[str] = []
    if cliff_score >= 0.70:
        reasons.append("one legal choice creates materially better outcomes than another")
    if debate_score >= 0.70:
        reasons.append("multiple actions remain defensible across plausible hidden deals")
    if comeback_score >= 0.65:
        reasons.append("the decision can reverse an opponent-bought or score-pressure position")
    if surprise_score >= 0.65:
        reasons.append("the strongest action differs from the obvious heuristic choice")
    if context.depends_on_signal:
        reasons.append("interpretation depends on optional playing-pattern evidence")
    if not reasons:
        reasons.append("balanced educational decision with at least two meaningful actions")

    return InterestingnessVector(
        best_action_id=best.action_id,
        second_action_id=second.action_id,
        action_count=len(actions),
        regret_gap=regret_gap,
        win_probability_gap=win_gap,
        disaster_gap=disaster_gap,
        cliff_score=cliff_score,
        debate_score=debate_score,
        comeback_score=comeback_score,
        surprise_score=surprise_score,
        hidden_information_score=hidden_information_score,
        educational_score=educational_score,
        robustness_score=robustness_score,
        difficulty_prior=difficulty_prior,
        publishability_score=publishability,
        archetype_scores=archetypes,
        reasons=tuple(reasons),
    )
