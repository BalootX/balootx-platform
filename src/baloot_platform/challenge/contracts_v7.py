"""Truth contracts for Baloot challenge publication.

Interestingness alone is not enough: a challenge must satisfy an explicit,
auditable content contract before its copy may imply winning, losing, comeback,
debate, or damage control.  This module contains no Baloot legality rules; it
consumes normalized evaluator outputs from the authoritative engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

import yaml


class ChallengeContract(str, Enum):
    OUTCOME_FLIP = "outcome_flip"
    ACTUAL_DEAL_FLIP = "actual_deal_flip"
    BIG_WIN_SMALL_WIN = "big_win_small_win"
    COMEBACK = "comeback"
    DEBATE = "debate"
    DAMAGE_CONTROL = "damage_control"
    BUY_OR_PASS = "buy_or_pass"

    @classmethod
    def parse(cls, value: str | "ChallengeContract") -> "ChallengeContract":
        return value if isinstance(value, cls) else cls(str(value).strip().lower())


class _ActionLike(Protocol):
    match_win_probability: float
    expected_round_margin: float
    confidence_low: float
    confidence_high: float
    exact_share: float
    deal_margins: tuple[float, ...]
    best_share: float


@dataclass(frozen=True, slots=True)
class ContractVerification:
    contract: ChallengeContract
    accepted: bool
    reasons: tuple[str, ...]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    truth_level: str = "observer_model"
    copy_policy: str = "comparative"
    winning_language_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract.value,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "truthLevel": self.truth_level,
            "copyPolicy": self.copy_policy,
            "winningLanguageAllowed": self.winning_language_allowed,
        }


@dataclass(frozen=True, slots=True)
class ContractProfile:
    raw: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ContractProfile":
        data = dict(value)
        if "contracts" not in data:
            raise ValueError(
                "challenge contract profile must contain contracts"
            )
        return cls(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ContractProfile":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("challenge contract profile must be a mapping")
        return cls.from_mapping(data)

    def settings(self, contract: ChallengeContract) -> Mapping[str, Any]:
        result = self.raw.get("contracts", {}).get(contract.value)
        if not isinstance(result, dict):
            raise KeyError(f"missing challenge contract settings: {contract.value}")
        return result

    @property
    def winning_contracts(self) -> frozenset[str]:
        return frozenset(self.raw.get("copy_policy", {}).get("winning_language_allowed_for", ()))


@dataclass(frozen=True, slots=True)
class NormalizedAction:
    action_id: str
    match_win_probability: float
    match_confidence_low: float
    match_confidence_high: float
    expected_round_margin: float
    round_lead_probability: float
    downside_cvar: float
    best_share: float
    exact_share: float
    deal_margins: tuple[float, ...]


def normalize_play_actions(actions: Iterable[Any]) -> tuple[NormalizedAction, ...]:
    normalized = tuple(
        NormalizedAction(
            getattr(item.card, "code", str(getattr(item, "card", "unknown"))),
            float(getattr(item, "match_win_probability", getattr(item, "win_probability", 0.0))),
            float(getattr(item, "match_confidence_low", getattr(item, "confidence_low", 0.0))),
            float(getattr(item, "match_confidence_high", getattr(item, "confidence_high", 1.0))),
            float(getattr(item, "expected_round_margin", getattr(item, "expected_match_margin", 0.0))),
            float(getattr(item, "round_lead_probability", getattr(item, "win_probability", 0.0))),
            float(getattr(item, "downside_cvar", 0.0)),
            float(getattr(item, "best_share", 0.0)),
            float(getattr(item, "exact_share", 0.0)),
            tuple(float(value) for value in getattr(item, "deal_margins", ())),
        )
        for item in actions
    )
    return tuple(sorted(normalized, key=lambda item: (item.match_win_probability, item.expected_round_margin), reverse=True))


def normalize_bid_actions(actions: Iterable[Any]) -> tuple[NormalizedAction, ...]:
    normalized = tuple(
        NormalizedAction(
            getattr(item.choice, "id", str(getattr(item, "choice", "unknown"))),
            0.5,
            0.0,
            1.0,
            float(getattr(item, "expected_match_margin", 0.0)),
            float(getattr(item, "team_win_probability", 0.0)),
            float(getattr(item, "downside_cvar10", 0.0)),
            float(getattr(item, "best_share", 0.0)),
            0.0,
            (),
        )
        for item in actions
    )
    return tuple(sorted(normalized, key=lambda item: (item.match_win_probability, item.expected_round_margin), reverse=True))


def _verification(
    contract: ChallengeContract,
    accepted: bool,
    reasons: list[str],
    metrics: Mapping[str, Any],
    profile: ContractProfile,
    *,
    truth_level: str = "observer_model",
) -> ContractVerification:
    winning = contract.value in profile.winning_contracts and accepted
    if contract == ChallengeContract.DAMAGE_CONTROL:
        copy_policy = "damage_control"
    elif winning:
        copy_policy = "winning"
    else:
        copy_policy = "comparative"
    return ContractVerification(contract, accepted, tuple(reasons), metrics, truth_level, copy_policy, winning)


def verify_contract(
    contract: str | ChallengeContract,
    actions: Iterable[NormalizedAction],
    profile: ContractProfile,
    *,
    opponent_bought: bool = False,
    hidden_deal_disagreement: float = 0.0,
) -> ContractVerification:
    contract = ChallengeContract.parse(contract)
    ranked = tuple(actions)
    if len(ranked) < 2:
        return _verification(contract, False, ["at least two legal actions are required"], {}, profile)
    ranked = tuple(sorted(ranked, key=lambda item: (item.match_win_probability, item.expected_round_margin), reverse=True))
    best, second, worst = ranked[0], ranked[1], ranked[-1]
    probability_swing = best.match_win_probability - worst.match_win_probability
    margin_swing = best.expected_round_margin - worst.expected_round_margin
    utility_gap = best.match_win_probability - second.match_win_probability
    metrics: dict[str, Any] = {
        "bestAction": best.action_id,
        "alternativeAction": worst.action_id,
        "bestMatchWinProbability": best.match_win_probability,
        "alternativeMatchWinProbability": worst.match_win_probability,
        "bestMatchConfidence95": [best.match_confidence_low, best.match_confidence_high],
        "alternativeMatchConfidence95": [worst.match_confidence_low, worst.match_confidence_high],
        "probabilitySwing": probability_swing,
        "bestExpectedRoundMargin": best.expected_round_margin,
        "alternativeExpectedRoundMargin": worst.expected_round_margin,
        "expectedRoundMarginSwing": margin_swing,
        "hiddenDealDisagreement": hidden_deal_disagreement,
        "opponentBought": opponent_bought,
        "bestExactShare": best.exact_share,
        "alternativeExactShare": worst.exact_share,
    }
    cfg = profile.settings(contract)
    reasons: list[str] = []

    if contract == ChallengeContract.OUTCOME_FLIP:
        checks = {
            "best action clears the minimum win-probability threshold": best.match_confidence_low >= float(cfg["best_match_win_probability_min"]),
            "alternative stays below the losing threshold": worst.match_confidence_high <= float(cfg["alternative_match_win_probability_max"]),
            "probability swing is large enough": probability_swing >= float(cfg["probability_swing_min"]),
            "round-margin swing is large enough": margin_swing >= float(cfg["expected_round_margin_swing_min"]),
        }
        accepted = all(checks.values())
        reasons.extend(label for label, value in checks.items() if not value)
        return _verification(contract, accepted, reasons or ["robust observer-level winning-versus-losing flip verified"], metrics, profile)

    if contract == ChallengeContract.ACTUAL_DEAL_FLIP:
        exact = best.exact_share >= 0.999999 and worst.exact_share >= 0.999999
        single = len(best.deal_margins) == 1 and len(worst.deal_margins) == 1
        best_actual = best.deal_margins[0] if single else best.expected_round_margin
        worst_actual = worst.deal_margins[0] if single else worst.expected_round_margin
        metrics.update({"bestActualMargin": best_actual, "alternativeActualMargin": worst_actual, "singleCompatibleDeal": single})
        checks = {
            "analysis is exact": exact or not bool(cfg.get("require_exact", True)),
            "exactly one compatible actual deal is identified": single,
            "best action wins the actual deal": best_actual >= float(cfg["best_actual_margin_min"]),
            "alternative loses the actual deal": worst_actual <= float(cfg["alternative_actual_margin_max"]),
        }
        accepted = all(checks.values())
        reasons.extend(label for label, value in checks.items() if not value)
        return _verification(contract, accepted, reasons or ["exact actual-deal outcome flip verified"], metrics, profile, truth_level="actual_deal_exact")

    if contract == ChallengeContract.BIG_WIN_SMALL_WIN:
        checks = {
            "best action is strongly winning": best.match_confidence_low >= float(cfg["best_match_win_probability_min"]),
            "alternative remains winning": worst.match_confidence_low >= float(cfg["alternative_match_win_probability_min"]),
            "margin swing is educationally material": margin_swing >= float(cfg["expected_round_margin_swing_min"]),
        }
        accepted = all(checks.values())
        reasons.extend(label for label, value in checks.items() if not value)
        return _verification(contract, accepted, reasons or ["both actions win but the best action wins substantially more"], metrics, profile)

    if contract == ChallengeContract.COMEBACK:
        checks = {
            "opponents bought the contract": opponent_bought is bool(cfg.get("opponent_bought", True)),
            "best action creates a winning path": best.match_confidence_low >= float(cfg["best_match_win_probability_min"]),
            "alternative remains losing": worst.match_confidence_high <= float(cfg["alternative_match_win_probability_max"]),
        }
        accepted = all(checks.values())
        reasons.extend(label for label, value in checks.items() if not value)
        return _verification(contract, accepted, reasons or ["opponent-bought comeback path verified"], metrics, profile)

    if contract == ChallengeContract.DEBATE:
        checks = {
            "top actions remain close": utility_gap <= float(cfg["maximum_utility_gap"]),
            "hidden deals disagree often enough": hidden_deal_disagreement >= float(cfg["minimum_hidden_deal_disagreement"]),
        }
        accepted = all(checks.values())
        reasons.extend(label for label, value in checks.items() if not value)
        return _verification(contract, accepted, reasons or ["multiple actions remain defensible under uncertainty"], metrics, profile)

    if contract == ChallengeContract.DAMAGE_CONTROL:
        accepted = best.match_confidence_high <= float(cfg["maximum_best_match_win_probability"])
        if not accepted:
            reasons.append("best action is not clearly a losing/damage-control position")
        return _verification(contract, accepted, reasons or ["all credible paths lose; the task is to minimize damage"], metrics, profile)

    if contract == ChallengeContract.BUY_OR_PASS:
        round_probability_swing = best.round_lead_probability - worst.round_lead_probability
        metrics["roundLeadProbabilitySwing"] = round_probability_swing
        metrics["matchProbabilityAvailable"] = False
        accepted = round_probability_swing >= float(cfg["probability_swing_min"])
        if not accepted:
            reasons.append("buy/pass round-lead probability swing is too small")
        return _verification(
            contract,
            accepted,
            reasons or ["buy/pass round-level threshold is materially consequential"],
            metrics,
            profile,
            truth_level="observer_round_model",
        )

    raise AssertionError(contract)


def safest_matching_contract(
    actions: Iterable[NormalizedAction],
    profile: ContractProfile,
    *,
    opponent_bought: bool = False,
    hidden_deal_disagreement: float = 0.0,
    decision_kind: str = "play",
) -> ContractVerification:
    order = (
        (ChallengeContract.BUY_OR_PASS,) if decision_kind == "bid" else (
            ChallengeContract.ACTUAL_DEAL_FLIP,
            ChallengeContract.OUTCOME_FLIP,
            ChallengeContract.COMEBACK,
            ChallengeContract.BIG_WIN_SMALL_WIN,
            ChallengeContract.DEBATE,
            ChallengeContract.DAMAGE_CONTROL,
        )
    )
    normalized = tuple(actions)
    failures: list[str] = []
    for contract in order:
        result = verify_contract(contract, normalized, profile, opponent_bought=opponent_bought, hidden_deal_disagreement=hidden_deal_disagreement)
        if result.accepted:
            return result
        failures.extend(f"{contract.value}: {reason}" for reason in result.reasons)
    fallback = ChallengeContract.DAMAGE_CONTROL if decision_kind != "bid" else ChallengeContract.BUY_OR_PASS
    return ContractVerification(fallback, False, tuple(failures), {}, "unverified", "unpublishable", False)
