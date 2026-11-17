from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .cards import FULL_DECK, Card, iter_mask, mask_to_cards
from .enums import Contract, GamePhase, Seat, Suit, Team
from .shuffle import ShuffleRecord


@dataclass(frozen=True, slots=True)
class Event:
    sequence: int
    type: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"sequence": self.sequence, "type": self.type, **copy.deepcopy(self.data)}


@dataclass(frozen=True, slots=True)
class DealRecord:
    sequence: int
    stage: str
    packet: int
    seat: Seat | None
    cards: tuple[Card, ...]
    face_up: bool = False

    def to_dict(self, *, reveal_cards: bool = False) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "stage": self.stage,
            "packet": self.packet,
            "seat": self.seat.name.lower() if self.seat is not None else None,
            "count": len(self.cards),
            "faceUp": self.face_up,
            "cards": [c.to_dict() for c in self.cards] if reveal_cards or self.face_up else None,
        }


@dataclass(frozen=True, slots=True)
class BidRecord:
    round_index: int
    turn_index: int
    seat: Seat
    action: str
    contract: Contract | None = None
    trump: Suit | None = None
    bidder: Seat | None = None
    buyer: Seat | None = None
    purchase_recipient: Seat | None = None
    declarer: Seat | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "round": self.round_index,
            "turn": self.turn_index,
            "seat": self.seat.name.lower(),
            "action": self.action,
            "contract": self.contract.value if self.contract else None,
            "trumpSuit": self.trump.name.lower() if self.trump is not None else None,
            "bidder": self.bidder.name.lower() if self.bidder is not None else None,
            "buyer": self.buyer.name.lower() if self.buyer is not None else None,
        }
        if (
            self.purchase_recipient is not None
            and self.purchase_recipient != self.buyer
        ):
            payload["purchaseRecipient"] = self.purchase_recipient.name.lower()
        return payload


@dataclass(frozen=True, slots=True)
class LegalPlayActionV1:
    """One authoritative card-play action, including an Ekka declaration."""

    card: Card
    ekka: bool = False

    @property
    def action_id(self) -> int:
        return self.card.id + (len(FULL_DECK) if self.ekka else 0)

    @property
    def code(self) -> str:
        return f"{self.card.code}+ekka" if self.ekka else self.card.code

    @classmethod
    def from_code(cls, value: str) -> "LegalPlayActionV1":
        raw = str(value).strip()
        ekka = raw.endswith("+ekka")
        card_code = raw.removesuffix("+ekka") if ekka else raw
        if "+" in card_code:
            raise ValueError("legal play action code is invalid")
        return cls(Card.parse(card_code), ekka)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actionId": self.action_id,
            "code": self.code,
            "card": self.card.to_dict(),
            "ekka": self.ekka,
        }


@dataclass(frozen=True, slots=True)
class PlayedCard:
    seat: Seat
    card: Card
    trick_index: int
    position: int
    legal_cards: tuple[Card, ...] = ()
    ekka: bool = False
    baloot: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "seat": self.seat.name.lower(),
            "card": self.card.to_dict(),
            "trickIndex": self.trick_index,
            "position": self.position,
            "legalCards": [c.to_dict() for c in self.legal_cards],
            "ekka": self.ekka,
            "baloot": self.baloot,
        }

    def public_dict(self) -> dict[str, Any]:
        return {
            "seat": self.seat.name.lower(),
            "card": self.card.to_dict(),
            "trickIndex": self.trick_index,
            "position": self.position,
            "ekka": self.ekka,
            "baloot": self.baloot,
        }


@dataclass(slots=True)
class Trick:
    index: int
    leader: Seat
    plays: list[PlayedCard] = field(default_factory=list)
    winner: Seat | None = None
    raw_points: int = 0

    @property
    def complete(self) -> bool:
        return len(self.plays) == 4

    @property
    def led_suit(self) -> Suit | None:
        return self.plays[0].card.suit if self.plays else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "leader": self.leader.name.lower(),
            "plays": [p.to_dict() for p in self.plays],
            "winner": self.winner.name.lower() if self.winner is not None else None,
            "rawPoints": self.raw_points,
        }

    def public_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "leader": self.leader.name.lower(),
            "plays": [play.public_dict() for play in self.plays],
            "winner": (
                self.winner.name.lower()
                if self.winner is not None
                else None
            ),
            "rawPoints": self.raw_points,
        }


@dataclass(frozen=True, slots=True)
class Project:
    owner: Seat
    kind: str
    cards: tuple[Card, ...]
    value: int
    high_rank: int
    category_priority: int
    declared: bool = False
    declaration_order: int = 999
    scoring: bool = False
    revealed: bool = False

    @property
    def team(self) -> Team:
        return self.owner.team

    @property
    def card_mask(self) -> int:
        mask = 0
        for card in self.cards:
            mask |= 1 << card.id
        return mask

    def public_declaration(self) -> dict[str, Any]:
        return {
            "owner": self.owner.name.lower(),
            "kind": self.kind,
            "value": self.value,
            "highRank": self.high_rank,
            "declarationOrder": self.declaration_order,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.public_declaration(),
            "team": self.team.name.lower(),
            "cards": [c.to_dict() for c in self.cards] if self.revealed else None,
            "scoring": self.scoring,
            "revealed": self.revealed,
        }


@dataclass(slots=True)
class RoundScore:
    raw_card_points: list[int]
    card_game_points: list[int]
    project_points: list[int]
    baloot_points: list[int]
    total_before_award: list[int]
    winner: Team
    awarded_points: list[int]
    counting_team: Team
    multiplier: int
    kabout: bool = False
    gahwa: bool = False
    match_winner: Team | None = None
    explanation: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rawCardPoints": list(self.raw_card_points),
            "cardGamePoints": list(self.card_game_points),
            "projectPoints": list(self.project_points),
            "balootPoints": list(self.baloot_points),
            "totalBeforeAward": list(self.total_before_award),
            "winner": self.winner.name.lower(),
            "awardedPoints": list(self.awarded_points),
            "countingTeam": self.counting_team.name.lower(),
            "multiplier": self.multiplier,
            "kabout": self.kabout,
            "gahwa": self.gahwa,
            "matchWinner": self.match_winner.name.lower() if self.match_winner is not None else None,
            "explanation": list(self.explanation),
        }


@dataclass(slots=True)
class RoundState:
    rules_hash: str
    seed: int
    round_number: int
    dealer: Seat
    phase: GamePhase = GamePhase.CREATED
    match_score: list[int] = field(default_factory=lambda: [0, 0])
    shuffle: ShuffleRecord | None = None
    deal_trace: list[DealRecord] = field(default_factory=list)
    purchase_card: Card | None = None
    undealt: tuple[Card, ...] = ()
    initial_five: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    initial_hands: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    hands: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    bid_history: list[BidRecord] = field(default_factory=list)
    contract: Contract | None = None
    buyer: Seat | None = None
    card_taker: Seat | None = None
    declarer: Seat | None = None
    trump_suit: Suit | None = None
    escalation: str = "normal"
    escalation_history: list[dict[str, Any]] = field(default_factory=list)
    double_seat: Seat | None = None
    locked: bool = False
    first_leader: Seat | None = None
    current_player: Seat | None = None
    current_trick: Trick | None = None
    completed_tricks: list[Trick] = field(default_factory=list)
    projects: list[Project] = field(default_factory=list)
    project_rank_visibility: dict[str, Any] | None = None
    _project_declaration_state: Any | None = None
    project_winning_team: Team | None = None
    baloot_teams: set[Team] = field(default_factory=set)
    trump_broken: bool = False
    round_score: RoundScore | None = None
    match_winner: Team | None = None
    annulled_by: Seat | None = None
    annul_reason: str | None = None
    events: list[Event] = field(default_factory=list)

    def emit(self, event_type: str, **data: Any) -> None:
        self.events.append(Event(len(self.events), event_type, copy.deepcopy(data)))

    def clone(self) -> "RoundState":
        return copy.deepcopy(self)

    def hand(self, seat: Seat) -> tuple[Card, ...]:
        return mask_to_cards(self.hands[int(seat)])

    @property
    def played_cards(self) -> tuple[PlayedCard, ...]:
        result: list[PlayedCard] = []
        for trick in self.completed_tricks:
            result.extend(trick.plays)
        if self.current_trick is not None:
            result.extend(self.current_trick.plays)
        return tuple(result)

    @property
    def winning_bid(self) -> BidRecord | None:
        """The final standing non-pass bid, derived from the canonical history."""
        return next((bid for bid in reversed(self.bid_history) if bid.action != "pass"), None)

    @property
    def winning_bidder(self) -> Seat | None:
        bid = self.winning_bid
        if bid is None:
            return None
        return bid.bidder if bid.bidder is not None else bid.seat

    @property
    def contract_owner(self) -> Seat | None:
        """Seat legally responsible for the purchased contract.

        The bidding policy resolves Ashkal and other variants into ``buyer``;
        this derived alias makes the role explicit without duplicating state.
        """
        return self.buyer

    @property
    def purchase_recipient(self) -> Seat | None:
        """Seat that physically receives the exposed purchase card."""
        return self.card_taker if self.card_taker is not None else self.buyer

    def validate_conservation(self) -> None:
        masks = list(self.hands)
        played_mask = 0
        for played in self.played_cards:
            bit = 1 << played.card.id
            if played_mask & bit:
                raise AssertionError(f"card played twice: {played.card.code}")
            played_mask |= bit
        union = played_mask
        total = played_mask.bit_count()
        for mask in masks:
            if union & mask:
                raise AssertionError("card exists in more than one location")
            union |= mask
            total += mask.bit_count()
        if self.initial_hands and any(self.initial_hands):
            if total != 32 or union != (1 << 32) - 1:
                raise AssertionError(f"card conservation failed: {total}/32")
            for seat in Seat:
                initial = self.initial_hands[int(seat)]
                own_played = 0
                for played in self.played_cards:
                    if played.seat == seat:
                        own_played |= 1 << played.card.id
                if initial != (self.hands[int(seat)] | own_played):
                    raise AssertionError(f"initial hand mismatch for {seat.name}")

    def to_dict(self, *, reveal_hands: bool = True) -> dict[str, Any]:
        return {
            **({"projectRankVisibility": copy.deepcopy(self.project_rank_visibility)} if self.project_rank_visibility is not None else {}),
            "rulesHash": self.rules_hash,
            "seed": self.seed,
            "roundNumber": self.round_number,
            "dealer": self.dealer.name.lower(),
            "phase": self.phase.value,
            "matchScore": list(self.match_score),
            "shuffle": self.shuffle.to_dict(reveal_order=reveal_hands) if self.shuffle else None,
            "dealTrace": [d.to_dict(reveal_cards=reveal_hands) for d in self.deal_trace],
            "purchaseCard": self.purchase_card.to_dict() if self.purchase_card else None,
            "hands": {
                seat.name.lower(): [c.to_dict() for c in iter_mask(self.hands[int(seat)])]
                if reveal_hands else {"count": self.hands[int(seat)].bit_count()}
                for seat in Seat
            },
            "initialHands": {
                seat.name.lower(): [c.to_dict() for c in iter_mask(self.initial_hands[int(seat)])]
                if reveal_hands else None
                for seat in Seat
            },
            "biddingHistory": [b.to_dict() for b in self.bid_history],
            "contract": self.contract.value if self.contract else None,
            "winningBidder": self.winning_bidder.name.lower() if self.winning_bidder is not None else None,
            "buyer": self.buyer.name.lower() if self.buyer is not None else None,
            "contractOwner": self.contract_owner.name.lower() if self.contract_owner is not None else None,
            "purchaseRecipient": self.purchase_recipient.name.lower() if self.purchase_recipient is not None else None,
            "declarer": self.declarer.name.lower() if self.declarer is not None else None,
            "trumpSuit": self.trump_suit.name.lower() if self.trump_suit is not None else None,
            "escalation": self.escalation,
            "escalationHistory": copy.deepcopy(self.escalation_history),
            "locked": self.locked,
            "firstLeader": self.first_leader.name.lower() if self.first_leader is not None else None,
            "currentPlayer": self.current_player.name.lower() if self.current_player is not None else None,
            "currentTrick": self.current_trick.to_dict() if self.current_trick is not None else None,
            "completedTricks": [t.to_dict() for t in self.completed_tricks],
            "projects": [p.to_dict() for p in self.projects],
            "projectWinningTeam": self.project_winning_team.name.lower() if self.project_winning_team is not None else None,
            "balootTeams": [team.name.lower() for team in sorted(self.baloot_teams, key=int)],
            "trumpBroken": self.trump_broken,
            "roundScore": self.round_score.to_dict() if self.round_score else None,
            "matchWinner": self.match_winner.name.lower() if self.match_winner is not None else None,
            "annulledBy": self.annulled_by.name.lower() if self.annulled_by is not None else None,
            "annulReason": self.annul_reason,
            "events": [event.to_dict() for event in self.events],
        }
