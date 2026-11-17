from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Suit

# Stable action-space contract. Existing identifiers must never be renumbered.
# 0..31: play card without Ekka
# 32..63: play same card while declaring Ekka
# 64..70: bidding
# 71..74: escalation
# 75: annul eligible opening hand
ACTION_COUNT = 76
ACTION_SCHEMA_VERSION = "1.0.0"


class ActionKind(str, Enum):
    PLAY = "play"
    BID = "bid"
    ESCALATION = "escalation"
    ANNUL = "annul"


@dataclass(frozen=True, slots=True)
class Action:
    id: int
    kind: ActionKind
    card: Card | None = None
    ekka: bool = False
    name: str | None = None
    trump_suit: Suit | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "card": self.card.to_dict() if self.card is not None else None,
            "ekka": self.ekka,
            "name": self.name,
            "trumpSuit": self.trump_suit.name.lower() if self.trump_suit is not None else None,
        }


class ActionCodec:
    BID_IDS = {
        64: ("pass", None),
        65: ("sun", None),
        66: ("ashkal", None),
        67: ("hokum", Suit.SPADES),
        68: ("hokum", Suit.HEARTS),
        69: ("hokum", Suit.DIAMONDS),
        70: ("hokum", Suit.CLUBS),
    }
    ESCALATION_IDS = {71: "double", 72: "triple", 73: "four", 74: "gahwa"}

    @staticmethod
    def play(card: Card | int | str | dict[str, Any], *, ekka: bool = False) -> int:
        parsed = Card.parse(card)
        return parsed.id + (32 if ekka else 0)

    @classmethod
    def bid(cls, action: str, trump_suit: Suit | str | int | None = None) -> int:
        action = action.strip().lower()
        if action in {"pass", "sun", "ashkal"}:
            return {"pass": 64, "sun": 65, "ashkal": 66}[action]
        if action != "hokum" or trump_suit is None:
            raise ValueError("Hokum action requires a trump suit")
        return 67 + int(Suit.parse(trump_suit))

    @classmethod
    def escalation(cls, action: str) -> int:
        action = action.strip().lower()
        for identifier, name in cls.ESCALATION_IDS.items():
            if name == action:
                return identifier
        raise ValueError(f"unknown escalation action {action}")

    @classmethod
    def decode(cls, identifier: int) -> Action:
        identifier = int(identifier)
        if not 0 <= identifier < ACTION_COUNT:
            raise ValueError(f"action id outside stable action space: {identifier}")
        if identifier < 64:
            return Action(identifier, ActionKind.PLAY, Card.parse(identifier % 32), identifier >= 32)
        if identifier in cls.BID_IDS:
            name, trump = cls.BID_IDS[identifier]
            return Action(identifier, ActionKind.BID, name=name, trump_suit=trump)
        if identifier in cls.ESCALATION_IDS:
            return Action(identifier, ActionKind.ESCALATION, name=cls.ESCALATION_IDS[identifier])
        return Action(identifier, ActionKind.ANNUL, name="kawesh")

    @classmethod
    def legal_play_mask(cls, legal_cards: tuple[Card, ...], *, ekka_cards: tuple[Card, ...] = ()) -> tuple[int, ...]:
        mask = [0] * ACTION_COUNT
        for card in legal_cards:
            mask[cls.play(card)] = 1
        for card in ekka_cards:
            mask[cls.play(card, ekka=True)] = 1
        return tuple(mask)

    @classmethod
    def legal_bid_mask(cls, actions: tuple[str, ...], *, purchase_suit: Suit, second_round: bool) -> tuple[int, ...]:
        mask = [0] * ACTION_COUNT
        for action in actions:
            if action == "hokum":
                suits = [purchase_suit] if not second_round else [s for s in Suit if s != purchase_suit]
                for suit in suits:
                    mask[cls.bid("hokum", suit)] = 1
            else:
                mask[cls.bid(action)] = 1
        return tuple(mask)
