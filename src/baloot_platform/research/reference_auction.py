from __future__ import annotations

from dataclasses import dataclass, field
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.enums import Contract, Rank, Seat, Suit


@dataclass(frozen=True, slots=True)
class ReferenceBidRecordV1:
    round_index: int
    position: int
    seat: Seat
    action: str
    contract: Contract | None = None
    trump: Suit | None = None
    bidder: Seat | None = None
    buyer: Seat | None = None
    purchase_recipient: Seat | None = None
    declarer: Seat | None = None


@dataclass(frozen=True, slots=True)
class ReferenceAuctionResultV1:
    records: tuple[ReferenceBidRecordV1, ...]
    all_passed: bool
    contract: Contract | None = None
    buyer: Seat | None = None
    declarer: Seat | None = None
    trump: Suit | None = None
    purchase_recipient: Seat | None = None
    caller: Seat | None = None


@dataclass(slots=True)
class ReferenceAuctionInterpreterV1:
    """Clear Domain-IR interpreter kept independent from ``BiddingMachine``."""

    dealer: Seat
    purchase_suit: Suit
    domain: AuctionDomainIRV1
    purchase_rank: Rank | None = None
    round_index: int = 1
    cursor: int = 0
    records: list[ReferenceBidRecordV1] = field(default_factory=list)
    finished: bool = False
    result: ReferenceAuctionResultV1 | None = None
    _candidate: ReferenceBidRecordV1 | None = None
    _recall_queue: list[Seat] = field(default_factory=list)
    _recall_position: int = 0

    def __post_init__(self) -> None:
        if not self.domain.require_preselected:
            return
        preselected = self.domain.preselected_bid(
            dealer=self.dealer, purchase_suit=self.purchase_suit
        )
        bidder = preselected.bidder
        resolved = preselected.resolution
        if (
            resolved.contract is None
            or resolved.buyer is None
        ):
            raise AssertionError("preselected auction did not resolve a contract")
        for position, seat in enumerate(self.order):
            if seat == bidder:
                self.records.append(
                    ReferenceBidRecordV1(
                        1,
                        position,
                        seat,
                        resolved.contract.value,
                        resolved.contract,
                        resolved.trump,
                        bidder,
                        resolved.buyer,
                        resolved.purchase_recipient,
                        resolved.declarer,
                    )
                )
                self.finished = True
                self.result = ReferenceAuctionResultV1(
                    tuple(self.records),
                    False,
                    resolved.contract,
                    resolved.buyer,
                    resolved.declarer,
                    resolved.trump,
                    resolved.purchase_recipient,
                    resolved.caller,
                )
                return
            self.records.append(ReferenceBidRecordV1(1, position, seat, "pass"))
        raise AssertionError("preselected bidder is not in seat order")

    @property
    def order(self) -> tuple[Seat, ...]:
        start = self.dealer.next()
        return (start, start.next(), start.next().next(), self.dealer)

    @property
    def in_recall(self) -> bool:
        return bool(self._recall_queue) and self._recall_position < len(
            self._recall_queue
        )

    @property
    def actor(self) -> Seat | None:
        if self.finished:
            return None
        if self.in_recall:
            return self._recall_queue[self._recall_position]
        return self.order[self.cursor]

    def legal_actions(self) -> tuple[str, ...]:
        if self.finished:
            return ()
        return self.domain.legal_actions(
            round_index=self.round_index,
            position=self.order.index(self.actor) + 1,
            in_recall=self.in_recall,
            candidate_contract=(
                self._candidate.contract if self._candidate is not None else None
            ),
            prior_round_actions=tuple(
                item.action
                for item in self.records
                if item.round_index == self.round_index
            ),
            purchase_rank=self.purchase_rank,
        )

    def submit(
        self, action: str, *, trump: Suit | None = None
    ) -> ReferenceAuctionResultV1 | None:
        if self.finished or self.actor is None:
            raise ValueError("bidding is already complete")
        action = action.strip().lower()
        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(f"illegal bid {action}; legal={legal}")
        seat = self.actor
        resolved = self.domain.resolve_action(
            action,
            round_index=self.round_index,
            actor=seat,
            purchase_suit=self.purchase_suit,
            trump=trump,
        )
        contract = resolved.contract
        resolved_trump = resolved.trump
        buyer = resolved.buyer

        record = ReferenceBidRecordV1(
            self.round_index,
            len(
                [
                    item
                    for item in self.records
                    if item.round_index == self.round_index
                ]
            ),
            seat,
            action,
            contract,
            resolved_trump,
            seat if contract is not None else None,
            buyer,
            resolved.purchase_recipient,
            resolved.declarer,
        )
        self.records.append(record)

        if self.in_recall:
            if contract == Contract.SUN:
                self._candidate = record
            self._recall_position += 1
            if not self.in_recall:
                self._finish_candidate()
            return self.result

        if self.round_index == 1:
            if contract == Contract.HOKUM:
                self._candidate = record
                self.cursor += 1
            elif contract in {Contract.SUN, Contract.ASHKAL}:
                self._candidate = record
                self._start_recall(before_index=self.cursor)
                if not self.in_recall:
                    self._finish_candidate()
                return self.result
            else:
                self.cursor += 1
            if self.cursor < 4:
                return None
            if self._candidate is None:
                if self.domain.rounds < 2:
                    self.finished = True
                    self.result = ReferenceAuctionResultV1(
                        tuple(self.records), True
                    )
                else:
                    self.round_index = 2
                    self.cursor = 0
                return self.result
            bidder_index = self.order.index(self._candidate.seat)
            self._start_recall(before_index=bidder_index)
            if not self.in_recall:
                self._finish_candidate()
            return self.result

        if contract in {Contract.SUN, Contract.ASHKAL}:
            self._candidate = record
            self._finish_candidate()
            return self.result
        if contract == Contract.HOKUM:
            self._candidate = record
        self.cursor += 1
        if self.cursor < 4:
            return None
        if self._candidate is None:
            self.finished = True
            self.result = ReferenceAuctionResultV1(tuple(self.records), True)
        else:
            self._finish_candidate()
        return self.result

    def _start_recall(self, *, before_index: int) -> None:
        self._recall_queue = [
            self.order[index] for index in range(before_index - 1, -1, -1)
        ]
        self._recall_position = 0

    def _finish_candidate(self) -> None:
        if (
            self._candidate is None
            or self._candidate.contract is None
            or self._candidate.bidder is None
            or self._candidate.buyer is None
            or self._candidate.declarer is None
        ):
            raise AssertionError("cannot finish without a standing bid")
        record = self._candidate
        self.finished = True
        self.result = ReferenceAuctionResultV1(
            tuple(self.records),
            False,
            record.contract,
            record.buyer,
            record.declarer,
            record.trump,
            record.purchase_recipient,
            record.bidder,
        )


__all__ = [
    "AuctionDomainIRV1",
    "ReferenceAuctionInterpreterV1",
    "ReferenceAuctionResultV1",
    "ReferenceBidRecordV1",
]
