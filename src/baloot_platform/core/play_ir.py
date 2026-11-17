from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .cards import Card
from .enums import Contract, Seat, Suit
from .structural_ir import StructuralDomainIRV1


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PlayDomainIRV1:
    """Validated card-play authority shared by runtime, reference, and CP-SAT."""

    structural_domain_ir_hash: str
    first_leader: str
    configured_first_leader: Seat
    must_follow_suit: bool
    must_trump_when_void: bool
    must_overtrump_when_possible: bool
    partner_winning_exception: str
    locked_hokum_blocks_trump_lead: bool
    ekka_enabled: bool
    allow_discard_when_cannot_overtrump: bool
    sun_strength: tuple[int, ...]
    hokum_non_trump_strength: tuple[int, ...]
    hokum_trump_strength: tuple[int, ...]
    sun_points: tuple[int, ...]
    hokum_non_trump_points: tuple[int, ...]
    hokum_trump_points: tuple[int, ...]
    sha256: str

    @classmethod
    def from_rules(
        cls,
        rules: Mapping[str, Any],
        *,
        structural_domain: StructuralDomainIRV1 | None = None,
    ) -> "PlayDomainIRV1":
        structural = (
            structural_domain
            if structural_domain is not None
            else StructuralDomainIRV1.from_rules(rules)
        )
        if not isinstance(structural, StructuralDomainIRV1):
            raise TypeError("play compilation requires StructuralDomainIRV1")
        structural.require_integrity()
        try:
            play = dict(rules["play"])
            cards = dict(rules["card_rules"])
            core = {
                "schemaVersion": 1,
                "structuralDomainIrHash": structural.sha256,
                "firstLeader": str(play["first_leader"]),
                "configuredFirstLeader": Seat.parse(
                    play["configured_first_leader"]
                ).name.lower(),
                "mustFollowSuit": cls._boolean(play["must_follow_suit"]),
                "mustTrumpWhenVoid": cls._boolean(
                    play["must_trump_when_void"]
                ),
                "mustOvertrumpWhenPossible": cls._boolean(
                    play["must_overtrump_when_possible"]
                ),
                "partnerWinningException": str(
                    play["partner_winning_exception"]
                ),
                "lockedHokumBlocksTrumpLead": cls._boolean(
                    play["locked_hokum_blocks_trump_lead"]
                ),
                "ekkaEnabled": cls._boolean(play["ekka_enabled"]),
                "allowDiscardWhenCannotOvertrump": cls._boolean(
                    play["allow_discard_when_cannot_overtrump"]
                ),
                "sunStrength": cls._table(cards["sun_strength"], points=False),
                "hokumNonTrumpStrength": cls._table(
                    cards["hokum_non_trump_strength"], points=False
                ),
                "hokumTrumpStrength": cls._table(
                    cards["hokum_trump_strength"], points=False
                ),
                "sunPoints": cls._table(cards["sun_points"], points=True),
                "hokumNonTrumpPoints": cls._table(
                    cards["hokum_non_trump_points"], points=True
                ),
                "hokumTrumpPoints": cls._table(
                    cards["hokum_trump_points"], points=True
                ),
            }
        except KeyError as exc:
            raise ValueError(f"missing play domain configuration: {exc}") from exc
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(str(exc)) from exc

        if core["firstLeader"] not in {"dealer_right", "buyer", "configured"}:
            raise ValueError("unsupported first-leader primitive")
        required_true = (
            "mustFollowSuit",
            "mustTrumpWhenVoid",
            "mustOvertrumpWhenPossible",
            "lockedHokumBlocksTrumpLead",
            "ekkaEnabled",
        )
        if not all(core[key] for key in required_true):
            raise ValueError("unsupported disabled card-play obligation")
        if core["partnerWinningException"] != "position_aware_ace_or_ekka":
            raise ValueError("unsupported partner-winning exception primitive")

        return cls(
            core["structuralDomainIrHash"],
            core["firstLeader"],
            Seat.parse(core["configuredFirstLeader"]),
            core["mustFollowSuit"],
            core["mustTrumpWhenVoid"],
            core["mustOvertrumpWhenPossible"],
            core["partnerWinningException"],
            core["lockedHokumBlocksTrumpLead"],
            core["ekkaEnabled"],
            core["allowDiscardWhenCannotOvertrump"],
            tuple(core["sunStrength"]),
            tuple(core["hokumNonTrumpStrength"]),
            tuple(core["hokumTrumpStrength"]),
            tuple(core["sunPoints"]),
            tuple(core["hokumNonTrumpPoints"]),
            tuple(core["hokumTrumpPoints"]),
            _canonical_sha256(core),
        )

    @staticmethod
    def _boolean(value: Any) -> bool:
        if type(value) is not bool:
            raise TypeError("play switches must be booleans")
        return value

    @staticmethod
    def _table(value: Any, *, points: bool) -> list[int]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError("card rule tables must be sequences")
        if len(value) != 8 or any(type(item) is not int for item in value):
            raise TypeError("card rule tables must contain eight integers")
        result = [int(item) for item in value]
        if points and any(not 0 <= item <= 32_767 for item in result):
            raise ValueError("card point tables must be in the native-safe range [0, 32767]")
        if not points and set(result) != set(range(8)):
            raise ValueError(
                "card strength tables must be a permutation of zero through seven"
            )
        return result

    def require_integrity(
        self,
        structural_domain: StructuralDomainIRV1,
    ) -> "PlayDomainIRV1":
        """Reject stale, cross-bundle, or caller-forged play IR objects."""

        if not isinstance(structural_domain, StructuralDomainIRV1):
            raise TypeError("play integrity requires StructuralDomainIRV1")
        structural_domain.require_integrity()
        if self.structural_domain_ir_hash != structural_domain.sha256:
            raise ValueError(
                "PlayDomainIRV1 structural dependency mismatch; rebuild it from rules"
            )
        rebuilt = type(self).from_rules(
            {
                "play": {
                    "first_leader": self.first_leader,
                    "configured_first_leader": self.configured_first_leader.name.lower(),
                    "must_follow_suit": self.must_follow_suit,
                    "must_trump_when_void": self.must_trump_when_void,
                    "must_overtrump_when_possible": self.must_overtrump_when_possible,
                    "partner_winning_exception": self.partner_winning_exception,
                    "locked_hokum_blocks_trump_lead": self.locked_hokum_blocks_trump_lead,
                    "ekka_enabled": self.ekka_enabled,
                    "allow_discard_when_cannot_overtrump": self.allow_discard_when_cannot_overtrump,
                },
                "card_rules": {
                    "sun_strength": list(self.sun_strength),
                    "hokum_non_trump_strength": list(self.hokum_non_trump_strength),
                    "hokum_trump_strength": list(self.hokum_trump_strength),
                    "sun_points": list(self.sun_points),
                    "hokum_non_trump_points": list(self.hokum_non_trump_points),
                    "hokum_trump_points": list(self.hokum_trump_points),
                },
            },
            structural_domain=structural_domain,
        )
        if rebuilt != self:
            raise ValueError(
                "PlayDomainIRV1 integrity mismatch; rebuild it from rules"
            )
        return self

    def resolve_first_leader(
        self,
        *,
        dealer: Seat,
        buyer: Seat,
        structural_domain: StructuralDomainIRV1,
    ) -> Seat:
        self.require_integrity(structural_domain)
        if self.first_leader == "dealer_right":
            return structural_domain.dealer_right(Seat.parse(dealer))
        if self.first_leader == "buyer":
            return Seat.parse(buyer)
        if self.first_leader == "configured":
            return self.configured_first_leader
        raise AssertionError("validated play domain contains an invalid first leader")

    def strength(self, card: Card, contract: Contract, trump: Suit | None) -> int:
        parsed = Card.parse(card)
        effective = Contract.parse(contract).effective
        if effective == Contract.HOKUM and trump is not None and parsed.suit == trump:
            table = self.hokum_trump_strength
        elif effective == Contract.HOKUM:
            table = self.hokum_non_trump_strength
        else:
            table = self.sun_strength
        return table[int(parsed.rank)]

    def points(self, card: Card, contract: Contract, trump: Suit | None) -> int:
        parsed = Card.parse(card)
        effective = Contract.parse(contract).effective
        if effective == Contract.HOKUM and trump is not None and parsed.suit == trump:
            table = self.hokum_trump_points
        elif effective == Contract.HOKUM:
            table = self.hokum_non_trump_points
        else:
            table = self.sun_points
        return table[int(parsed.rank)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "structuralDomainIrHash": self.structural_domain_ir_hash,
            "firstLeader": self.first_leader,
            "configuredFirstLeader": self.configured_first_leader.name.lower(),
            "mustFollowSuit": self.must_follow_suit,
            "mustTrumpWhenVoid": self.must_trump_when_void,
            "mustOvertrumpWhenPossible": self.must_overtrump_when_possible,
            "partnerWinningException": self.partner_winning_exception,
            "lockedHokumBlocksTrumpLead": self.locked_hokum_blocks_trump_lead,
            "ekkaEnabled": self.ekka_enabled,
            "allowDiscardWhenCannotOvertrump": self.allow_discard_when_cannot_overtrump,
            "sunStrength": list(self.sun_strength),
            "hokumNonTrumpStrength": list(self.hokum_non_trump_strength),
            "hokumTrumpStrength": list(self.hokum_trump_strength),
            "sunPoints": list(self.sun_points),
            "hokumNonTrumpPoints": list(self.hokum_non_trump_points),
            "hokumTrumpPoints": list(self.hokum_trump_points),
            "sha256": self.sha256,
        }


__all__ = ["PlayDomainIRV1"]
