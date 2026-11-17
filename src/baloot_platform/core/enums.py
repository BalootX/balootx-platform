from __future__ import annotations

from enum import Enum, IntEnum


class Team(IntEnum):
    NORTH_SOUTH = 0
    WEST_EAST = 1

    @property
    def other(self) -> "Team":
        return Team(1 - int(self))

    @property
    def arabic(self) -> str:
        return "لنا" if self == Team.NORTH_SOUTH else "لهم"


class Seat(IntEnum):
    NORTH = 0
    WEST = 1
    SOUTH = 2
    EAST = 3

    def next(self) -> "Seat":
        return Seat((int(self) + 1) % 4)

    def previous(self) -> "Seat":
        return Seat((int(self) - 1) % 4)

    @property
    def partner(self) -> "Seat":
        return Seat((int(self) + 2) % 4)

    @property
    def team(self) -> Team:
        return Team.NORTH_SOUTH if self in {Seat.NORTH, Seat.SOUTH} else Team.WEST_EAST

    @property
    def arabic(self) -> str:
        return {
            Seat.NORTH: "الشريك",
            Seat.WEST: "الخصم الأيسر",
            Seat.SOUTH: "أنت",
            Seat.EAST: "الخصم الأيمن",
        }[self]

    @classmethod
    def parse(cls, value: "Seat | str | int") -> "Seat":
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[str(value).strip().upper()]


class Suit(IntEnum):
    SPADES = 0
    HEARTS = 1
    DIAMONDS = 2
    CLUBS = 3

    @property
    def symbol(self) -> str:
        return "♠♥♦♣"[int(self)]

    @property
    def arabic(self) -> str:
        return ("سباتي", "هاص", "دايمن", "كلب")[int(self)]

    @classmethod
    def parse(cls, value: "Suit | str | int") -> "Suit":
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        text = str(value).strip().upper()
        aliases = {"S": cls.SPADES, "H": cls.HEARTS, "D": cls.DIAMONDS, "C": cls.CLUBS}
        if text in aliases:
            return aliases[text]
        return cls[text]


class Rank(IntEnum):
    SEVEN = 0
    EIGHT = 1
    NINE = 2
    TEN = 3
    JACK = 4
    QUEEN = 5
    KING = 6
    ACE = 7

    @property
    def short(self) -> str:
        return ("7", "8", "9", "10", "J", "Q", "K", "A")[int(self)]

    @classmethod
    def parse(cls, value: "Rank | str | int") -> "Rank":
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        text = str(value).strip().upper()
        aliases = {r.short: r for r in cls}
        if text in aliases:
            return aliases[text]
        return cls[text]


class Contract(str, Enum):
    SUN = "sun"
    HOKUM = "hokum"
    ASHKAL = "ashkal"

    @property
    def effective(self) -> "Contract":
        return Contract.SUN if self == Contract.ASHKAL else self

    @property
    def arabic(self) -> str:
        return {Contract.SUN: "صن", Contract.HOKUM: "حكم", Contract.ASHKAL: "أشكل"}[self]

    @classmethod
    def parse(cls, value: "Contract | str") -> "Contract":
        return value if isinstance(value, cls) else cls(str(value).strip().lower())


class GamePhase(str, Enum):
    CREATED = "created"
    SHUFFLED = "shuffled"
    INITIAL_DEALT = "initial_dealt"
    BIDDING = "bidding"
    FINAL_DEALT = "final_dealt"
    PROJECTS = "projects"
    PLAYING = "playing"
    SCORED = "scored"
    COMPLETE = "complete"
