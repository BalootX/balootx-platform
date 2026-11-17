from __future__ import annotations

from dataclasses import dataclass, field

from .auction_ir import AuctionDomainIRV1
from .bidding import BidResult
from .enums import Contract, Rank, Seat, Suit
from .models import BidRecord


@dataclass(slots=True)
class BiddingMachine:
    """Authoritative interactive two-round Baloot auction.

    The first round supports the Saudi recall rule: after Hokum or a later Sun,
    players whose original turns were earlier are offered a Sun overcall window
    in reverse order until the highest-priority willing Sun bidder is known.
    The second round has no recall; a passed player cannot later re-enter.
    """

    dealer: Seat
    purchase_suit: Suit
    domain: AuctionDomainIRV1
    purchase_rank: Rank | None = None
    round_index: int = 1
    cursor: int = 0
    records: list[BidRecord] = field(default_factory=list)
    finished: bool = False
    result: BidResult | None = None
    _candidate: BidRecord | None = None
    _recall_queue: list[Seat] = field(default_factory=list)
    _recall_position: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.domain, AuctionDomainIRV1):
            raise TypeError("BiddingMachine requires a compiled AuctionDomainIRV1")
        if not self.domain.require_preselected:
            return
        preselected = self.domain.preselected_bid(
            dealer=self.dealer, purchase_suit=self.purchase_suit
        )
        bidder = preselected.bidder
        resolved = preselected.resolution
        for seat in self.order:
            if seat == bidder:
                record = self._record(
                    seat,
                    resolved.contract.value,
                    resolved.contract,
                    resolved.trump,
                    resolved.buyer,
                    resolved.purchase_recipient,
                    resolved.declarer,
                )
                self._candidate = record
                self._finish_candidate()
                return
            self._record(seat, "pass", None, None, None)
        raise AssertionError("preselected bidder is not in seat order")

    @property
    def order(self) -> list[Seat]:
        start = self.dealer.next()
        return [start, start.next(), start.next().next(), self.dealer]

    @property
    def in_recall(self) -> bool:
        return bool(self._recall_queue) and self._recall_position < len(self._recall_queue)

    @property
    def actor(self) -> Seat | None:
        if self.finished:
            return None
        if self.in_recall:
            return self._recall_queue[self._recall_position]
        return self.order[self.cursor]

    def _record(
        self, seat: Seat, action: str, contract: Contract | None,
        trump: Suit | None, buyer: Seat | None,
        purchase_recipient: Seat | None = None,
        declarer: Seat | None = None,
    ) -> BidRecord:
        record = BidRecord(
            self.round_index,
            len([item for item in self.records if item.round_index == self.round_index]),
            seat,
            action,
            contract,
            trump,
            seat if contract else None,
            buyer,
            purchase_recipient,
            declarer,
        )
        self.records.append(record)
        return record

    def legal_actions(self) -> tuple[str, ...]:
        if self.finished:
            return ()
        actor = self.actor
        if actor is None:
            raise AssertionError("unfinished auction has no actor")
        return self.domain.legal_actions(
            round_index=self.round_index,
            position=self.order.index(actor) + 1,
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

    def submit(self, action: str, *, trump: Suit | None = None) -> BidResult | None:
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

        record = self._record(
            seat,
            action,
            contract,
            resolved_trump,
            buyer,
            resolved.purchase_recipient,
            resolved.declarer,
        )

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
                # Only players who acted earlier can supersede this Sun-like bid.
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
                    self.result = BidResult(tuple(self.records), True)
                else:
                    self.round_index = 2
                    self.cursor = 0
                return self.result
            # Hokum survived later players; offer earlier passers the Sun recall.
            bidder_index = self.order.index(self._candidate.seat)
            self._start_recall(before_index=bidder_index)
            if not self.in_recall:
                self._finish_candidate()
            return self.result

        # Second round has no recall.
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
            self.result = BidResult(tuple(self.records), True)
        else:
            self._finish_candidate()
        return self.result

    def _start_recall(self, *, before_index: int) -> None:
        # Example: if dealer at index 3 bids Hokum, the recall order is index
        # 2, 1, 0 (the published West→North→East example).
        self._recall_queue = [self.order[index] for index in range(before_index - 1, -1, -1)]
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
        self.result = BidResult(
            tuple(self.records),
            False,
            record.contract,
            record.buyer,
            record.declarer,
            record.trump,
            record.purchase_recipient,
            record.bidder,
        )
