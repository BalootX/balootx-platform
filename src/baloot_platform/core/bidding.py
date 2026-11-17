from __future__ import annotations

from dataclasses import dataclass

from .auction_ir import AuctionDomainIRV1
from .cards import Card
from .enums import Contract, Rank, Seat, Suit
from .models import BidRecord
from .rules import RuleBook


@dataclass(frozen=True, slots=True)
class BidResult:
    history: tuple[BidRecord, ...]
    all_passed: bool
    contract: Contract | None = None
    buyer: Seat | None = None
    declarer: Seat | None = None
    trump: Suit | None = None
    purchase_recipient: Seat | None = None
    caller: Seat | None = None


def _seat_order(dealer: Seat) -> list[Seat]:
    start = dealer.next()
    return [start, start.next(), start.next().next(), dealer]


def _sun_score(cards: tuple[Card, ...], rules: RuleBook) -> float:
    heuristics = rules.data["bidding"]["heuristics"]
    values = tuple(float(value) for value in heuristics["sun_rank_weights"])
    score = sum(values[int(card.rank)] for card in cards)
    suits = {suit: sum(card.suit == suit for card in cards) for suit in Suit}
    extra_after = int(heuristics["sun_extra_suit_card_after"])
    extra_weight = float(heuristics["sun_extra_suit_card_weight"])
    score += sum(extra_weight * max(0, count - extra_after) for count in suits.values())
    return score


def _hokum_score(cards: tuple[Card, ...], trump: Suit, rules: RuleBook) -> float:
    heuristics = rules.data["bidding"]["heuristics"]
    values = tuple(float(value) for value in heuristics["hokum_trump_rank_weights"])
    score = sum(values[int(card.rank)] for card in cards if card.suit == trump)
    trump_count = sum(card.suit == trump for card in cards)
    score += float(heuristics["hokum_extra_trump_card_weight"]) * max(
        0,
        trump_count - int(heuristics["hokum_extra_trump_card_after"]),
    )
    score += float(heuristics["hokum_off_trump_ace_ten_weight"]) * sum(
        card.rank in {Rank.ACE, Rank.TEN} and card.suit != trump
        for card in cards
    )
    return score


def _forced_history(
    dealer: Seat, contract: Contract, buyer: Seat, trump: Suit | None,
    domain: AuctionDomainIRV1,
) -> BidResult:
    order = _seat_order(dealer)
    bidder = (
        buyer.partner
        if contract == Contract.ASHKAL and domain.ashkal_contract_owner == "taker"
        else buyer
    )
    resolved = domain.resolve_action(
        contract.value, round_index=1, actor=bidder,
        purchase_suit=trump or Suit.SPADES, trump=trump,
    )
    if contract == Contract.ASHKAL and order.index(bidder) + 1 not in domain.ashkal_priority_positions:
        raise ValueError("forced Ashkal buyer must be the partner of the third or fourth bidder")
    history: list[BidRecord] = []
    for turn, seat in enumerate(order):
        if seat == bidder:
            history.append(BidRecord(
                1, turn, seat, contract.value, contract, trump, bidder,
                resolved.buyer, resolved.purchase_recipient, resolved.declarer,
            ))
            return BidResult(
                tuple(history), False, contract, resolved.buyer,
                resolved.declarer, trump, resolved.purchase_recipient,
                resolved.caller,
            )
        history.append(BidRecord(1, turn, seat, "pass"))
    raise ValueError("forced bidder is not in seat order")


def _choose_action(machine, five_cards: dict[Seat, tuple[Card, ...]], purchase_card: Card, rules: RuleBook) -> tuple[str, Suit | None]:
    actor = machine.actor
    if actor is None:
        return "pass", None
    legal = machine.legal_actions()
    cards = five_cards[actor]
    heuristics = rules.data["bidding"].get("heuristics", {})
    round_two = machine.round_index == 2
    sun_threshold = float(heuristics.get("second_sun_threshold" if round_two else "sun_threshold", 9.8 if round_two else 10.5))
    sun_margin = _sun_score(cards, rules) - sun_threshold

    choices: list[tuple[float, str, Suit | None]] = []
    if "sun" in legal:
        choices.append((sun_margin, "sun", None))
    if "hokum" in legal:
        if round_two:
            for suit in Suit:
                if suit != purchase_card.suit:
                    margin = _hokum_score(cards, suit, rules) - float(heuristics.get("second_hokum_threshold", 9.3))
                    choices.append((margin, "hokum", suit))
        else:
            margin = _hokum_score(cards, purchase_card.suit, rules) - float(heuristics.get("hokum_threshold", 9.0))
            choices.append((margin, "hokum", purchase_card.suit))
    if "ashkal" in legal:
        # Ashkal is received by the bidder's partner, but that hand is private.
        # The baseline therefore uses only the bidder's own support cards and
        # the public purchase card.  Looking at the partner's true five cards
        # would be hidden-information leakage.
        public_view = (*cards, purchase_card)
        margin = _sun_score(tuple(public_view), rules) - float(heuristics.get("ashkal_partner_threshold", 12.0))
        choices.append((margin, "ashkal", None))
    best = max(choices, default=(-999.0, "pass", None), key=lambda item: item[0])
    return (best[1], best[2]) if best[0] >= 0 else ("pass", None)


def run_bidding(
    five_cards: dict[Seat, tuple[Card, ...]],
    purchase_card: Card,
    dealer: Seat,
    rules: RuleBook,
    *,
    forced_contract: Contract | None = None,
    forced_buyer: Seat | None = None,
    forced_trump: Suit | None = None,
) -> BidResult:
    allowed = {Contract.parse(value) for value in rules.data["contracts"]["allowed"]}
    if forced_contract is not None:
        contract = Contract.parse(forced_contract)
        if contract not in allowed:
            raise ValueError(f"contract {contract.value} disabled by profile")
        buyer = dealer.next() if forced_buyer is None else forced_buyer
        trump = (purchase_card.suit if forced_trump is None else forced_trump) if contract == Contract.HOKUM else None
        if contract == Contract.HOKUM and forced_trump is not None and forced_trump != purchase_card.suit:
            # A different suit implies a second-round forced bid. Build a valid
            # compact history with a full first-round pass and the second-round bid.
            history = [BidRecord(1, index, seat, "pass") for index, seat in enumerate(_seat_order(dealer))]
            bidder = buyer
            for index, seat in enumerate(_seat_order(dealer)):
                if seat == bidder:
                    history.append(BidRecord(
                        2, index, seat, "hokum", contract, forced_trump,
                        bidder, buyer, buyer, buyer,
                    ))
                    return BidResult(
                        tuple(history), False, contract, buyer, buyer,
                        forced_trump, buyer, bidder,
                    )
                history.append(BidRecord(2, index, seat, "pass"))
        return _forced_history(
            dealer, contract, buyer, trump,
            AuctionDomainIRV1.from_rules(rules.data),
        )

    # Import locally to avoid a module cycle: the machine depends on BidResult.
    from .bidding_machine import BiddingMachine

    machine = BiddingMachine(
        dealer,
        purchase_card.suit,
        AuctionDomainIRV1.from_rules(rules.data),
        purchase_card.rank,
    )
    if machine.result is not None:
        return machine.result
    while machine.result is None:
        action, trump = _choose_action(machine, five_cards, purchase_card, rules)
        machine.submit(action, trump=trump)
    return machine.result
