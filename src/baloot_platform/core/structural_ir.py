from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .enums import Rank, Seat


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class StructuralDomainIRV1:
    """Validated deck, seating, and dealing authority for every compiler target."""

    suits: tuple[str, ...]
    ranks: tuple[str, ...]
    players: int
    cards_per_player: int
    play_order: tuple[Seat, ...]
    dealer_rotation: str
    partnerships: tuple[Seat, ...]
    shuffle_algorithm: str
    cut_enabled: bool
    cutter: str
    allow_no_cut: bool
    initial_packets: tuple[int, ...]
    exposed_purchase_cards: int
    final_cards_for_buyer: int
    final_cards_for_others: int
    first_recipient: str
    all_pass_redeal: bool
    annulling_enabled: bool
    annulling_names: tuple[str, ...]
    annulling_eligible_ranks: tuple[str, ...]
    annulling_declaration_window: str
    annulling_dealer_advances: bool
    sha256: str

    _SEAT_KEYS = ("north", "west", "south", "east")

    @staticmethod
    def _closed_mapping(
        value: Any,
        *,
        path: str,
        required: frozenset[str],
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be an object")
        result = {str(key): child for key, child in value.items()}
        missing = sorted(required - set(result))
        unknown = sorted(set(result) - required)
        if missing:
            raise ValueError(f"missing structural fields at {path}: {missing}")
        if unknown:
            raise ValueError(f"unknown structural fields at {path}: {unknown}")
        return result

    @staticmethod
    def _string_list(value: Any, *, path: str) -> list[str]:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path} must be an array")
        if any(type(item) is not str or not item for item in value):
            raise ValueError(f"{path} must contain non-empty strings")
        return [str(item) for item in value]

    @classmethod
    def from_rules(cls, rules: Mapping[str, Any]) -> "StructuralDomainIRV1":
        try:
            seating = cls._closed_mapping(
                rules["seating"],
                path="seating",
                required=frozenset(
                    {"play_order", "dealer_rotation", "partnerships"}
                ),
            )
            deck = cls._closed_mapping(
                rules["deck"],
                path="deck",
                required=frozenset(
                    {"suits", "ranks", "players", "cards_per_player"}
                ),
            )
            dealing = cls._closed_mapping(
                rules["dealing"],
                path="dealing",
                required=frozenset(
                    {
                        "shuffle_algorithm",
                        "cut_enabled",
                        "cutter",
                        "allow_no_cut",
                        "initial_packets",
                        "exposed_purchase_cards",
                        "final_cards_for_buyer",
                        "final_cards_for_others",
                        "first_recipient",
                        "all_pass_redeal",
                        "annulling",
                    }
                ),
            )
            annulling = cls._closed_mapping(
                dealing["annulling"],
                path="dealing.annulling",
                required=frozenset(
                    {
                        "enabled",
                        "names",
                        "eligible_ranks",
                        "declaration_window",
                        "dealer_advances",
                    }
                ),
            )
            play_order = cls._string_list(
                seating["play_order"], path="seating.play_order"
            )
            order = tuple(Seat.parse(value) for value in play_order)
            partnerships_value = cls._closed_mapping(
                seating["partnerships"],
                path="seating.partnerships",
                required=frozenset(cls._SEAT_KEYS),
            )
            partnerships = tuple(
                Seat.parse(partnerships_value[Seat(index).name.lower()])
                for index in range(4)
            )
            suits = cls._string_list(deck["suits"], path="deck.suits")
            ranks = cls._string_list(deck["ranks"], path="deck.ranks")
            annulling_names = cls._string_list(
                annulling["names"], path="dealing.annulling.names"
            )
            eligible_ranks = cls._string_list(
                annulling["eligible_ranks"],
                path="dealing.annulling.eligible_ranks",
            )
            core = {
                "schemaVersion": 1,
                "suits": suits,
                "ranks": ranks,
                "players": cls._integer(deck["players"]),
                "cardsPerPlayer": cls._integer(deck["cards_per_player"]),
                "playOrder": [seat.name.lower() for seat in order],
                "dealerRotation": str(seating["dealer_rotation"]),
                "partnerships": [seat.name.lower() for seat in partnerships],
                "shuffleAlgorithm": str(dealing["shuffle_algorithm"]),
                "cutEnabled": cls._boolean(dealing["cut_enabled"]),
                "cutter": str(dealing["cutter"]),
                "allowNoCut": cls._boolean(dealing["allow_no_cut"]),
                "initialPackets": cls._integer_list(dealing["initial_packets"]),
                "exposedPurchaseCards": cls._integer(
                    dealing["exposed_purchase_cards"]
                ),
                "finalCardsForBuyer": cls._integer(
                    dealing["final_cards_for_buyer"]
                ),
                "finalCardsForOthers": cls._integer(
                    dealing["final_cards_for_others"]
                ),
                "firstRecipient": str(dealing["first_recipient"]),
                "allPassRedeal": cls._boolean(dealing["all_pass_redeal"]),
                "annullingEnabled": cls._boolean(annulling["enabled"]),
                "annullingNames": annulling_names,
                "annullingEligibleRanks": [
                    Rank.parse(value).name.lower()
                    for value in eligible_ranks
                ],
                "annullingDeclarationWindow": str(
                    annulling["declaration_window"]
                ),
                "annullingDealerAdvances": cls._boolean(
                    annulling["dealer_advances"]
                ),
            }
        except ValueError:
            raise
        except KeyError as exc:
            raise ValueError(
                f"missing or unknown structural value: {exc}"
            ) from exc
        except (TypeError, AttributeError) as exc:
            raise ValueError(
                f"invalid structural domain value: {exc}"
            ) from exc

        supported_invariants = (
            ("deck.suits", "suits", ["spades", "hearts", "diamonds", "clubs"]),
            (
                "deck.ranks",
                "ranks",
                ["seven", "eight", "nine", "ten", "jack", "queen", "king", "ace"],
            ),
            ("deck.players", "players", 4),
            ("deck.cards_per_player", "cardsPerPlayer", 8),
            (
                "seating.play_order",
                "playOrder",
                ["north", "west", "south", "east"],
            ),
            ("seating.dealer_rotation", "dealerRotation", "right"),
            (
                "seating.partnerships",
                "partnerships",
                ["south", "east", "north", "west"],
            ),
            (
                "dealing.shuffle_algorithm",
                "shuffleAlgorithm",
                "splitmix64_fisher_yates_v1",
            ),
            ("dealing.cutter", "cutter", "dealer_left"),
            ("dealing.exposed_purchase_cards", "exposedPurchaseCards", 1),
            ("dealing.final_cards_for_buyer", "finalCardsForBuyer", 2),
            ("dealing.final_cards_for_others", "finalCardsForOthers", 3),
            ("dealing.first_recipient", "firstRecipient", "dealer_right"),
            ("dealing.all_pass_redeal", "allPassRedeal", True),
            (
                "dealing.annulling.names",
                "annullingNames",
                ["kawesh", "saneen"],
            ),
            (
                "dealing.annulling.declaration_window",
                "annullingDeclarationWindow",
                "first_round_own_turn",
            ),
        )
        for path, key, value in supported_invariants:
            if core[key] != value:
                raise ValueError(
                    f"unsupported {path}={core[key]!r}; supported value is {value!r}"
                )
        if not core["initialPackets"] or any(
            value <= 0 for value in core["initialPackets"]
        ) or sum(core["initialPackets"]) != 5:
            raise ValueError(
                "dealing.initial_packets must be positive and total five"
            )
        if not core["cutEnabled"] and not core["allowNoCut"]:
            raise ValueError(
                "dealing.cut_enabled=false requires dealing.allow_no_cut=true"
            )
        card_count = len(core["suits"]) * len(core["ranks"])
        final_count = (
            core["players"] * sum(core["initialPackets"])
            + core["exposedPurchaseCards"]
            + core["finalCardsForBuyer"]
            + (core["players"] - 1) * core["finalCardsForOthers"]
        )
        if card_count != core["players"] * core["cardsPerPlayer"]:
            raise ValueError(
                "deck.suits and deck.ranks must equal players times cards_per_player"
            )
        if final_count != card_count:
            raise ValueError(
                "dealing packet counts must consume the complete deck"
            )
        if len(set(core["annullingNames"])) != len(core["annullingNames"]):
            raise ValueError("dealing.annulling.names must be unique")
        if not core["annullingEligibleRanks"]:
            raise ValueError(
                "dealing.annulling.eligible_ranks must be non-empty"
            )
        if len(set(core["annullingEligibleRanks"])) != len(
            core["annullingEligibleRanks"]
        ):
            raise ValueError(
                "dealing.annulling.eligible_ranks must be unique"
            )

        return cls(
            tuple(core["suits"]),
            tuple(core["ranks"]),
            core["players"],
            core["cardsPerPlayer"],
            order,
            core["dealerRotation"],
            partnerships,
            core["shuffleAlgorithm"],
            core["cutEnabled"],
            core["cutter"],
            core["allowNoCut"],
            tuple(core["initialPackets"]),
            core["exposedPurchaseCards"],
            core["finalCardsForBuyer"],
            core["finalCardsForOthers"],
            core["firstRecipient"],
            core["allPassRedeal"],
            core["annullingEnabled"],
            tuple(core["annullingNames"]),
            tuple(core["annullingEligibleRanks"]),
            core["annullingDeclarationWindow"],
            core["annullingDealerAdvances"],
            _canonical_sha256(core),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StructuralDomainIRV1":
        """Rebuild a closed canonical structural-domain snapshot."""

        if not isinstance(value, Mapping):
            raise ValueError("structural domain snapshot must be an object")
        document = {str(key): child for key, child in value.items()}
        expected = {
            "schemaVersion", "suits", "ranks", "players", "cardsPerPlayer",
            "playOrder", "dealerRotation", "partnerships", "shuffleAlgorithm",
            "cutEnabled", "cutter", "allowNoCut", "initialPackets",
            "exposedPurchaseCards", "finalCardsForBuyer", "finalCardsForOthers",
            "firstRecipient", "allPassRedeal", "annullingEnabled",
            "annullingNames", "annullingEligibleRanks",
            "annullingDeclarationWindow", "annullingDealerAdvances", "sha256",
        }
        if set(document) != expected:
            raise ValueError("structural domain snapshot has unknown or missing fields")
        known_hash = document.pop("sha256")
        if (
            type(known_hash) is not str
            or len(known_hash) != 64
            or any(character not in "0123456789abcdef" for character in known_hash)
            or type(document.get("schemaVersion")) is not int
            or document["schemaVersion"] != 1
        ):
            raise ValueError("structural domain snapshot identity is invalid")
        list_fields = (
            "suits", "ranks", "playOrder", "partnerships", "initialPackets",
            "annullingNames", "annullingEligibleRanks",
        )
        if any(type(document.get(field)) is not list for field in list_fields):
            raise ValueError("structural domain snapshot arrays must be canonical lists")
        partnerships = document["partnerships"]
        if len(partnerships) != 4:
            raise ValueError("structural domain snapshot partnerships are invalid")
        rebuilt = cls.from_rules({
            "seating": {
                "play_order": document["playOrder"],
                "dealer_rotation": document["dealerRotation"],
                "partnerships": {
                    Seat(index).name.lower(): partnerships[index]
                    for index in range(4)
                },
            },
            "deck": {
                "suits": document["suits"],
                "ranks": document["ranks"],
                "players": document["players"],
                "cards_per_player": document["cardsPerPlayer"],
            },
            "dealing": {
                "shuffle_algorithm": document["shuffleAlgorithm"],
                "cut_enabled": document["cutEnabled"],
                "cutter": document["cutter"],
                "allow_no_cut": document["allowNoCut"],
                "initial_packets": document["initialPackets"],
                "exposed_purchase_cards": document["exposedPurchaseCards"],
                "final_cards_for_buyer": document["finalCardsForBuyer"],
                "final_cards_for_others": document["finalCardsForOthers"],
                "first_recipient": document["firstRecipient"],
                "all_pass_redeal": document["allPassRedeal"],
                "annulling": {
                    "enabled": document["annullingEnabled"],
                    "names": document["annullingNames"],
                    "eligible_ranks": document["annullingEligibleRanks"],
                    "declaration_window": document[
                        "annullingDeclarationWindow"
                    ],
                    "dealer_advances": document["annullingDealerAdvances"],
                },
            },
        })
        if rebuilt.sha256 != known_hash or _canonical_sha256(document) != known_hash:
            raise ValueError("structural domain snapshot hash is invalid")
        return rebuilt

    @staticmethod
    def _boolean(value: Any) -> bool:
        if type(value) is not bool:
            raise TypeError("structural switches must be booleans")
        return value

    @staticmethod
    def _integer(value: Any) -> int:
        if type(value) is not int:
            raise TypeError("structural values must be integers")
        return value

    @classmethod
    def _integer_list(cls, value: Any) -> list[int]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("initial packets must be a list")
        return [cls._integer(item) for item in value]

    @property
    def card_count(self) -> int:
        return len(self.suits) * len(self.ranks)

    @property
    def card_ids(self) -> tuple[int, ...]:
        return tuple(range(self.card_count))

    def offset(self, seat: Seat, steps: int) -> Seat:
        index = self.play_order.index(Seat.parse(seat))
        return self.play_order[(index + int(steps)) % len(self.play_order)]

    def dealer_right(self, dealer: Seat) -> Seat:
        return self.offset(dealer, 1)

    def dealer_left(self, dealer: Seat) -> Seat:
        return self.offset(dealer, -1)

    def next_dealer(self, dealer: Seat) -> Seat:
        if self.dealer_rotation != "right":
            raise ValueError(
                f"unsupported seating.dealer_rotation={self.dealer_rotation!r}"
            )
        return self.dealer_right(dealer)

    def cutter_for(self, dealer: Seat) -> Seat:
        if self.cutter != "dealer_left":
            raise ValueError(f"unsupported dealing.cutter={self.cutter!r}")
        return self.dealer_left(dealer)

    def recipient_order(self, dealer: Seat) -> tuple[Seat, ...]:
        if self.first_recipient != "dealer_right":
            raise ValueError(
                f"unsupported dealing.first_recipient={self.first_recipient!r}"
            )
        return tuple(
            self.offset(dealer, step) for step in range(1, self.players + 1)
        )

    def partner(self, seat: Seat) -> Seat:
        return self.partnerships[int(Seat.parse(seat))]

    def annulment_actor(self, dealer: Seat) -> Seat:
        """Return the only actor exposed by the governed declaration window."""
        if self.annulling_declaration_window != "first_round_own_turn":
            raise ValueError(
                "unsupported dealing.annulling.declaration_window="
                f"{self.annulling_declaration_window!r}"
            )
        return self.dealer_right(dealer)

    def can_declare_annulment(self, *, dealer: Seat, seat: Seat) -> bool:
        return Seat.parse(seat) == self.annulment_actor(Seat.parse(dealer))

    def next_dealer_after_annulment(self, dealer: Seat) -> Seat:
        dealer = Seat.parse(dealer)
        return self.next_dealer(dealer) if self.annulling_dealer_advances else dealer

    def validate_legacy_all_pass_dealer_policy(self, policy: str) -> None:
        """Fail closed while the legacy game profile is being retired."""
        expected = "counter_clockwise_next" if self.all_pass_redeal else "same"
        if str(policy) != expected:
            raise ValueError(
                "game all-pass dealer policy conflicts with the structural domain: "
                f"expected {expected!r}"
            )

    def validate_cut_position(self, value: int) -> int:
        position = self._integer(value)
        if not 1 <= position < self.card_count:
            raise ValueError(
                f"cut position must be between 1 and {self.card_count - 1}"
            )
        return position

    def validate_deck_order(self, values: Sequence[int]) -> tuple[int, ...]:
        if not isinstance(values, (list, tuple)):
            raise TypeError("deck order must be a complete array")
        deck = tuple(self._integer(value) for value in values)
        if len(deck) != self.card_count or sorted(deck) != list(self.card_ids):
            raise ValueError("deck order must contain every governed card exactly once")
        return deck

    def can_annul_ranks(self, hand_ranks: Sequence[str | Rank]) -> bool:
        ranks = tuple(Rank.parse(value) for value in hand_ranks)
        eligible = {Rank.parse(value) for value in self.annulling_eligible_ranks}
        return (
            self.annulling_enabled
            and len(ranks) == sum(self.initial_packets)
            and all(rank in eligible for rank in ranks)
        )

    def accepts_annulment_reason(self, reason: str) -> bool:
        return str(reason) in self.annulling_names

    def require_integrity(self) -> "StructuralDomainIRV1":
        """Reject stale or caller-forged structural objects at target boundaries."""
        rebuilt = type(self).from_rules(
            {
                "seating": {
                    "play_order": [
                        seat.name.lower() for seat in self.play_order
                    ],
                    "dealer_rotation": self.dealer_rotation,
                    "partnerships": {
                        Seat(index).name.lower(): self.partnerships[
                            index
                        ].name.lower()
                        for index in range(self.players)
                    },
                },
                "deck": {
                    "suits": list(self.suits),
                    "ranks": list(self.ranks),
                    "players": self.players,
                    "cards_per_player": self.cards_per_player,
                },
                "dealing": {
                    "shuffle_algorithm": self.shuffle_algorithm,
                    "cut_enabled": self.cut_enabled,
                    "cutter": self.cutter,
                    "allow_no_cut": self.allow_no_cut,
                    "initial_packets": list(self.initial_packets),
                    "exposed_purchase_cards": self.exposed_purchase_cards,
                    "final_cards_for_buyer": self.final_cards_for_buyer,
                    "final_cards_for_others": self.final_cards_for_others,
                    "first_recipient": self.first_recipient,
                    "all_pass_redeal": self.all_pass_redeal,
                    "annulling": {
                        "enabled": self.annulling_enabled,
                        "names": list(self.annulling_names),
                        "eligible_ranks": list(
                            self.annulling_eligible_ranks
                        ),
                        "declaration_window": (
                            self.annulling_declaration_window
                        ),
                        "dealer_advances": (
                            self.annulling_dealer_advances
                        ),
                    },
                },
            }
        )
        if rebuilt != self:
            raise ValueError(
                "StructuralDomainIRV1 integrity mismatch; rebuild it from rules"
            )
        return self

    def semantic_observation(self, *, dealer: Seat) -> dict[str, Any]:
        dealer = Seat.parse(dealer)
        return {
            "structuralDomainIrHash": self.sha256,
            "deck": {
                "suits": list(self.suits),
                "ranks": list(self.ranks),
                "players": self.players,
                "cardsPerPlayer": self.cards_per_player,
                "cardIds": list(self.card_ids),
            },
            "seating": {
                "playOrder": [seat.name.lower() for seat in self.play_order],
                "dealerRotation": self.dealer_rotation,
                "nextDealer": self.next_dealer(dealer).name.lower(),
                "partnerships": [
                    self.partner(Seat(index)).name.lower()
                    for index in range(self.players)
                ],
            },
            "dealing": {
                "shuffleAlgorithm": self.shuffle_algorithm,
                "cutEnabled": self.cut_enabled,
                "cutterPolicy": self.cutter,
                "cutter": self.cutter_for(dealer).name.lower(),
                "allowNoCut": self.allow_no_cut,
                "initialPackets": list(self.initial_packets),
                "exposedPurchaseCards": self.exposed_purchase_cards,
                "finalCardsForBuyer": self.final_cards_for_buyer,
                "finalCardsForOthers": self.final_cards_for_others,
                "firstRecipient": self.first_recipient,
                "allPassRedeal": self.all_pass_redeal,
                "annulling": {
                    "enabled": self.annulling_enabled,
                    "names": list(self.annulling_names),
                    "eligibleRanks": list(self.annulling_eligible_ranks),
                    "declarationWindow": self.annulling_declaration_window,
                    "declarationActor": self.annulment_actor(dealer).name.lower(),
                    "dealerAdvances": self.annulling_dealer_advances,
                },
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "suits": list(self.suits),
            "ranks": list(self.ranks),
            "players": self.players,
            "cardsPerPlayer": self.cards_per_player,
            "playOrder": [seat.name.lower() for seat in self.play_order],
            "dealerRotation": self.dealer_rotation,
            "partnerships": [seat.name.lower() for seat in self.partnerships],
            "shuffleAlgorithm": self.shuffle_algorithm,
            "cutEnabled": self.cut_enabled,
            "cutter": self.cutter,
            "allowNoCut": self.allow_no_cut,
            "initialPackets": list(self.initial_packets),
            "exposedPurchaseCards": self.exposed_purchase_cards,
            "finalCardsForBuyer": self.final_cards_for_buyer,
            "finalCardsForOthers": self.final_cards_for_others,
            "firstRecipient": self.first_recipient,
            "allPassRedeal": self.all_pass_redeal,
            "annullingEnabled": self.annulling_enabled,
            "annullingNames": list(self.annulling_names),
            "annullingEligibleRanks": list(self.annulling_eligible_ranks),
            "annullingDeclarationWindow": self.annulling_declaration_window,
            "annullingDealerAdvances": self.annulling_dealer_advances,
            "sha256": self.sha256,
        }


__all__ = ["StructuralDomainIRV1"]
