"""Observer-safe, configurable buy/pass policies.

Policies see only the acting player's cards, the public purchase card, auction
history, score, and rules profile.  They never inspect partner/opponent hands.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import Mapping, Sequence


class BidAction(str, Enum):
    PASS = "pass"
    SUN = "sun"
    HOKUM = "hokum"
    ASHKAL = "ashkal"


@dataclass(frozen=True)
class BidCandidate:
    action: BidAction
    suit: str | None = None
    estimated_success: float = 0.0
    expected_margin: float = 0.0
    downside: float = 0.0
    information_cost: float = 0.0


@dataclass(frozen=True)
class BuyPolicyConfig:
    plugin_id: str = "threshold_buy_v4"
    minimum_success: float = 0.56
    minimum_expected_margin: float = 2.0
    score_pressure_aggression: float = 0.12
    trailing_aggression: float = 0.10
    leading_conservatism: float = 0.08
    risk_tolerance: float = 0.50
    bluff_probability: float = 0.0
    mistake_probability: float = 0.0
    deterministic_seed: int = 1
    prefer_sun_with_close_value: float = 0.025


@dataclass(frozen=True)
class BuyContext:
    actor: str
    hand: tuple[str, ...]
    purchase_card: str | None
    legal_candidates: tuple[BidCandidate, ...]
    team_score: int = 0
    opponent_score: int = 0
    target_score: int = 152
    round_index: int = 1
    auction_history: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True)
class BuyDecision:
    action: BidAction
    suit: str | None
    score: float
    confidence: float
    reason_codes: tuple[str, ...]
    plugin_id: str


def decide_buy(context: BuyContext, config: BuyPolicyConfig | None = None) -> BuyDecision:
    config = config or BuyPolicyConfig()
    if not context.legal_candidates:
        raise ValueError("legal_candidates cannot be empty")
    rng = Random(config.deterministic_seed + context.round_index * 1009 + sum(map(ord, context.actor)))

    target = max(1, context.target_score)
    trailing = max(0.0, (context.opponent_score - context.team_score) / target)
    leading = max(0.0, (context.team_score - context.opponent_score) / target)
    near_end = max(context.team_score, context.opponent_score) / target
    success_threshold = (
        config.minimum_success
        - config.trailing_aggression * trailing
        - config.score_pressure_aggression * near_end * trailing
        + config.leading_conservatism * leading
    )

    ranked: list[tuple[float, BidCandidate]] = []
    for candidate in context.legal_candidates:
        if candidate.action is BidAction.PASS:
            ranked.append((0.0, candidate))
            continue
        risk_penalty = (1.0 - config.risk_tolerance) * max(0.0, -candidate.downside) / 100.0
        score = (
            1.55 * (candidate.estimated_success - success_threshold)
            + candidate.expected_margin / 100.0
            - risk_penalty
            - candidate.information_cost * 0.04
        )
        if candidate.action is BidAction.SUN:
            score += config.prefer_sun_with_close_value
        ranked.append((score, candidate))

    ranked.sort(key=lambda item: (item[0], item[1].estimated_success, item[1].expected_margin), reverse=True)
    best_score, best = ranked[0]

    # Bounded mistakes and bluffs are deliberate profile behavior.  They are
    # seeded and auditable, and never bypass the supplied legal action list.
    non_pass = [item for item in ranked if item[1].action is not BidAction.PASS]
    if non_pass and rng.random() < config.bluff_probability:
        best_score, best = non_pass[min(len(non_pass) - 1, 1)]
    elif len(ranked) > 1 and rng.random() < config.mistake_probability:
        best_score, best = ranked[1]

    pass_candidate = next((item for item in ranked if item[1].action is BidAction.PASS), None)
    if best.action is not BidAction.PASS and (
        best.estimated_success < success_threshold or best.expected_margin < config.minimum_expected_margin
    ):
        if pass_candidate is not None:
            best_score, best = pass_candidate

    reasons = [f"success_threshold={success_threshold:.3f}"]
    if trailing > 0:
        reasons.append("trailing_score_increases_aggression")
    if leading > 0:
        reasons.append("leading_score_increases_conservatism")
    if best.action is BidAction.PASS:
        reasons.append("all_non_pass_candidates_below_profile_threshold")
    else:
        reasons.extend(("best_legal_contract_estimate", f"estimated_success={best.estimated_success:.3f}"))
    confidence = max(0.0, min(1.0, abs(best_score) * 2.0 + 0.5))
    return BuyDecision(best.action, best.suit, best_score, confidence, tuple(reasons), config.plugin_id)
