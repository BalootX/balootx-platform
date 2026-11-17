from __future__ import annotations

from collections import defaultdict

from baloot_platform.core.enums import Contract, Suit
from .models import HardDeduction, InformationState


def derive_hard_deductions(info: InformationState) -> tuple[HardDeduction, ...]:
    deductions: list[HardDeduction] = []
    tricks: dict[int, list] = defaultdict(list)
    for play in info.plays:
        tricks[play.trick_index].append(play)
    seen: set[tuple] = set()
    for index, plays in tricks.items():
        plays.sort(key=lambda p: p.position)
        if not plays:
            continue
        led = plays[0].card.suit
        for play in plays[1:]:
            if play.card.suit != led:
                key = ("void", play.seat, led)
                if key not in seen:
                    deductions.append(HardDeduction("void_suit", play.seat, led, 1.0, f"trick_{index + 1}_failed_to_follow"))
                    seen.add(key)
                # Not cutting is not automatically proof of having no trump: in
                # positional Baloot the player's partner may already be winning,
                # or the player may be unable to overtrump and be permitted to
                # discard. Exact no-trump deductions are therefore established
                # only by replaying a hypothetical hand through the legal-move
                # generator in hidden_deals.py, never as a soft "hard fact".
    return tuple(deductions)


def void_map(deductions: tuple[HardDeduction, ...]) -> dict:
    result: dict = defaultdict(set)
    for d in deductions:
        if d.kind == "void_suit" and d.suit is not None:
            result[d.seat].add(d.suit)
    return result
