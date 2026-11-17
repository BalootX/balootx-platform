from __future__ import annotations

from typing import Any

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat
from baloot_platform.core.models import RoundState
from baloot_platform.core.play_ir import PlayDomainIRV1

from .models import InformationState, SignalObservation


def _relationship_factor(profile: dict[str, Any], relationship: str) -> float:
    defaults = {"self": 1.0, "partner": 1.0, "left_opponent": 0.80, "right_opponent": 0.80}
    return float(profile.get("relationship_reliability", {}).get(relationship, defaults.get(relationship, 0.75)))


def _forcedness_factor(profile: dict[str, Any], legal_count: int) -> float:
    if legal_count <= 1:
        label = "forced"
    elif legal_count == 2:
        label = "near_forced"
    elif legal_count <= 4:
        label = "meaningful_choice"
    else:
        label = "highly_optional"
    defaults = {"forced": 0.0, "near_forced": 0.15, "meaningful_choice": 0.45, "highly_optional": 0.70}
    return float(profile.get("forcedness", {}).get(label, defaults[label]))


def _remaining_suit_cards(state: RoundState, seat: Seat, suit) -> tuple[Card, ...]:
    return tuple(card for card in state.hand(seat) if card.suit == suit)


def deal_likelihood(
    state: RoundState,
    info: InformationState,
    observations: tuple[SignalObservation, ...],
    profile: dict[str, Any] | None,
    *,
    play_domain: PlayDomainIRV1 | None = None,
) -> float:
    """Return a conservative likelihood weight for one legal hidden deal.

    These weights never establish legality. They only reweight already legal
    deals when an explicitly enabled convention profile is selected.
    """

    profile = profile or {}
    if not bool(profile.get("apply_to_probability", False)):
        return 1.0
    total = 1.0
    signals_cfg = profile.get("signals", {})
    for observation in observations:
        cfg = signals_cfg.get(observation.kind, {})
        if not bool(cfg.get("enabled", False)):
            continue
        ratios = cfg.get("likelihood_ratios", {})
        relationship = _relationship_factor(profile, observation.relationship)
        matched = [
            play
            for play in state.played_cards
            if play.seat == observation.seat and play.card in observation.cards
        ]
        optionality = max(
            (_forcedness_factor(profile, len(play.legal_cards)) for play in matched),
            default=0.0,
        )
        strength = optionality * relationship
        if strength <= 0:
            continue

        ratio = 1.0
        suit = observation.cards[-1].suit if observation.cards else None
        remaining = _remaining_suit_cards(state, observation.seat, suit) if suit is not None else ()
        if observation.kind == "low_then_high":
            if play_domain is None or not observation.cards:
                continue
            observed_strength = play_domain.strength(
                observation.cards[-1], info.contract, info.trump_suit
            )
            has_preserved_strength = any(
                play_domain.strength(card, info.contract, info.trump_suit)
                > observed_strength
                for card in remaining
            )
            ratio = float(ratios.get("encourage_suit" if has_preserved_strength else "no_signal_or_deception", 1.0))
        elif observation.kind == "high_then_low":
            ratio = float(ratios.get("short_suit" if not remaining else "discourage_suit", 1.0))
        elif observation.kind == "partner_return":
            if play_domain is None:
                continue
            high_remaining = any(
                play_domain.strength(card, info.contract, info.trump_suit) >= 5
                for card in remaining
            )
            ratio = float(ratios.get("support_partner_suit" if high_remaining else "tactical_return", 1.0))
        elif observation.kind == "minimum_winner":
            ratio = float(ratios.get("preserve_strength", 1.0))
        elif observation.kind == "high_expenditure":
            played = next(
                (
                    item
                    for item in matched
                    if item.card == observation.cards[0]
                    and item.legal_cards
                ),
                None,
            )
            if played is None or play_domain is None:
                continue
            played_strength = play_domain.strength(
                played.card, info.contract, info.trump_suit
            )
            maximum_strength = max(
                play_domain.strength(card, info.contract, info.trump_suit)
                for card in played.legal_cards
            )
            branch = (
                "urgency_or_signal"
                if played_strength == maximum_strength
                else "error_or_deception"
            )
            ratio = float(ratios.get(branch, 1.0))
        else:
            continue

        # Temper each convention rather than multiplying raw cultural priors.
        adjusted = 1.0 + (ratio - 1.0) * strength
        total *= min(1.25, max(0.80, adjusted))
    return min(4.0, max(0.25, total))


def effective_sample_size(weights: list[float]) -> float:
    if not weights:
        return 0.0
    total = sum(weights)
    squared = sum(weight * weight for weight in weights)
    return (total * total / squared) if squared > 0 else 0.0


def weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0:
        raise ValueError("posterior weights have zero mass")
    return sum(value * weight for value, weight in zip(values, weights)) / total
