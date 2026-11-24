from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Iterable, Iterator

from .enums import Rank, Suit

_CARD_CODES = tuple(f"{suit.name[0]}{rank.short}" for suit in Suit for rank in Rank)


@dataclass(frozen=True, order=True, slots=True)
class Card:
    id: int

    def __post_init__(self) -> None:
        if not 0 <= int(self.id) < 32:
            raise ValueError(f"card id out of range: {self.id}")

    @property
    def suit(self) -> Suit:
        return Suit(self.id // 8)

    @property
    def rank(self) -> Rank:
        return Rank(self.id % 8)

    @property
    def code(self) -> str:
        if type(self.id) is int:
            return _CARD_CODES[self.id]
        return f"{self.suit.name[0]}{self.rank.short}"

    @classmethod
    def of(cls, suit: Suit, rank: Rank) -> "Card":
        return cls(int(suit) * 8 + int(rank))

    @classmethod
    def parse(cls, value: "Card | int | str | dict") -> "Card":
        if isinstance(value, cls):
            return value
        try:
            if isinstance(value, bool):
                raise ValueError("card id must be an integer")
            if isinstance(value, int):
                return cls(value)
            if isinstance(value, dict):
                if "id" in value:
                    raw_id = value["id"]
                    card_id = int(raw_id)
                    if isinstance(raw_id, bool) or (
                        isinstance(raw_id, Number) and raw_id != card_id
                    ):
                        raise ValueError("card id must be an integer")
                    return cls(card_id)
                return cls.of(
                    Suit.parse(value["suit"]),
                    Rank.parse(value["rank"]),
                )
            text = str(value).strip().upper()
            if len(text) < 2:
                raise ValueError("card code must include suit and rank")
            return cls.of(Suit.parse(text[0]), Rank.parse(text[1:]))
        except (KeyError, TypeError, IndexError, ValueError, OverflowError) as exc:
            try:
                description = repr(value)
            except ValueError:
                description = f"<{type(value).__name__}>"
            raise ValueError(f"invalid card: {description}") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "code": self.code,
            "suit": self.suit.name.lower(),
            "rank": self.rank.name.lower(),
            "symbol": self.suit.symbol,
            "label": f"{self.rank.short}{self.suit.symbol}",
        }


FULL_DECK: tuple[Card, ...] = tuple(Card(i) for i in range(32))


def cards_to_mask(cards: Iterable[Card]) -> int:
    mask = 0
    for card in cards:
        mask |= 1 << Card.parse(card).id
    return mask


def iter_mask(mask: int) -> Iterator[Card]:
    value = int(mask)
    while value:
        bit = value & -value
        yield Card(bit.bit_length() - 1)
        value ^= bit


def mask_to_cards(mask: int) -> tuple[Card, ...]:
    return tuple(iter_mask(mask))


def sorted_hand(cards: Iterable[Card]) -> tuple[Card, ...]:
    return tuple(sorted((Card.parse(c) for c in cards), key=lambda c: (int(c.suit), int(c.rank))))
