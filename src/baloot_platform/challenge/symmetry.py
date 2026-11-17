"""Deterministic seat/team/suit transforms for Foundry invariance gates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat, Suit, Team

from .evidence_v3 import DecisionPerspectiveEvidenceV3, canonical_sha256


@dataclass(frozen=True, slots=True)
class SeatRotationV1:
    quarter_turns: int

    def __post_init__(self) -> None:
        if self.quarter_turns not in range(4):
            raise ValueError("seat rotation must be between zero and three quarter turns")

    def map_seat(self, seat: Seat) -> Seat:
        return Seat((int(seat) + self.quarter_turns) % 4)

    def inverse_seat(self, seat: Seat) -> Seat:
        return Seat((int(seat) - self.quarter_turns) % 4)

    def map_team(self, team: Team) -> Team:
        return team if self.quarter_turns % 2 == 0 else team.other

    def map_team_values(self, values: tuple[float, float]) -> tuple[float, float]:
        if len(values) != 2:
            raise ValueError("team values must contain exactly two entries")
        if self.quarter_turns % 2 == 0:
            return float(values[0]), float(values[1])
        return float(values[1]), float(values[0])

    @staticmethod
    def observer_offset(actor: Seat, observer: Seat) -> int:
        return (int(observer) - int(actor)) % 4


@dataclass(frozen=True, slots=True)
class SuitRelabelingV1:
    mapped_suits: tuple[Suit, Suit, Suit, Suit]

    def __post_init__(self) -> None:
        if set(self.mapped_suits) != set(Suit):
            raise ValueError("suit relabeling must be a complete bijection")

    @classmethod
    def from_mapping(cls, mapping: Mapping[Suit, Suit]) -> "SuitRelabelingV1":
        if set(mapping) != set(Suit):
            raise ValueError("suit relabeling must map every suit")
        return cls(tuple(mapping[suit] for suit in Suit))

    def map_suit(self, suit: Suit) -> Suit:
        return self.mapped_suits[int(suit)]

    def map_card(self, card: Card) -> Card:
        return Card.of(self.map_suit(card.suit), card.rank)


@dataclass(frozen=True, slots=True)
class SymmetryAuditV1:
    rotation: int
    compared_records: int
    maximum_delta: float
    failures: tuple[str, ...]
    sha256: str

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "rotation": self.rotation,
            "comparedRecords": self.compared_records,
            "maximumDelta": self.maximum_delta,
            "failures": list(self.failures),
            "passed": self.passed,
            "sha256": self.sha256,
        }


def audit_rotated_perspective_evidence(
    original: Iterable[DecisionPerspectiveEvidenceV3],
    rotated: Iterable[DecisionPerspectiveEvidenceV3],
    *,
    quarter_turns: int,
    tolerance: float = 1e-9,
) -> SymmetryAuditV1:
    """Compare two independently analyzed evidence sets after inverse mapping."""
    if tolerance < 0:
        raise ValueError("symmetry tolerance cannot be negative")
    transform = SeatRotationV1(quarter_turns)
    source = tuple(original)
    target = tuple(rotated)
    target_index = {
        (
            item.decision_id,
            item.acting_seat,
            item.observer_seat,
            item.truth_mode,
        ): item
        for item in target
    }
    failures: list[str] = []
    maximum_delta = 0.0
    compared = 0
    for item in source:
        key = (
            item.decision_id,
            transform.map_seat(item.acting_seat),
            transform.map_seat(item.observer_seat),
            item.truth_mode,
        )
        counterpart = target_index.get(key)
        if counterpart is None:
            failures.append(f"missing:{item.decision_id}:{item.acting_seat.name}:{item.observer_seat.name}")
            continue
        compared += 1
        if item.forced != counterpart.forced or item.truth_label != counterpart.truth_label:
            failures.append(f"truth:{item.decision_id}:{item.observer_seat.name}")
        keys = set(item.evidence) | set(counterpart.evidence)
        for dimension in keys:
            delta = abs(float(item.evidence.get(dimension, 0.0)) - float(counterpart.evidence.get(dimension, 0.0)))
            maximum_delta = max(maximum_delta, delta)
            if delta > tolerance:
                failures.append(f"evidence:{item.decision_id}:{item.observer_seat.name}:{dimension}")
        confidence_delta = abs(float(item.confidence) - float(counterpart.confidence))
        maximum_delta = max(maximum_delta, confidence_delta)
        if confidence_delta > tolerance:
            failures.append(f"confidence:{item.decision_id}:{item.observer_seat.name}")
    if len(target) != len(source):
        failures.append("record_count")
    core = {
        "schemaVersion": 1,
        "rotation": quarter_turns,
        "comparedRecords": compared,
        "maximumDelta": maximum_delta,
        "failures": sorted(set(failures)),
    }
    return SymmetryAuditV1(
        quarter_turns,
        compared,
        maximum_delta,
        tuple(core["failures"]),
        canonical_sha256(core),
    )
