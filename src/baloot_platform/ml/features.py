from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from baloot_platform.analysis.models import InformationState
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.models import RoundState

FEATURE_SCHEMA_VERSION = "2.0.0"
# Base public play contract (274) plus a shared 48-feature decision envelope:
# phase(4), observer(4), auction actor(4), bidding round(2), per-seat bid counts
# (16), standing bid(4), purchase suit(4), last bidder(4), last bid(4),
# auction progress(1), pass streak(1) = 322.
FEATURE_COUNT = 322


def _one_hot(index: int | None, length: int) -> list[float]:
    result = [0.0] * length
    if index is not None and 0 <= int(index) < length:
        result[int(index)] = 1.0
    return result


def _mask_bits(mask: int, length: int = 32) -> list[float]:
    return [1.0 if mask & (1 << index) else 0.0 for index in range(length)]


def _bid_record_value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        # Accept both canonical API casing and Python field names.
        aliases = {
            "round_index": ("roundIndex", "round_index"),
            "seat": ("seat",),
            "action": ("action",),
            "contract": ("contract",),
        }
        for candidate in aliases.get(key, (key,)):
            if candidate in record:
                return record[candidate]
        return default
    return getattr(record, key, default)


def _bid_summary(records: Iterable[Any]) -> tuple[list[float], list[float], list[float], list[float], float, float]:
    items = list(records)
    action_index = {"pass": 0, "sun": 1, "hokum": 2, "ashkal": 3}
    counts = [0.0] * 16
    last_actor: int | None = None
    last_action: int | None = None
    standing: int | None = None
    pass_streak = 0
    for record in items:
        try:
            seat = int(Seat.parse(_bid_record_value(record, "seat")))
        except Exception:
            continue
        action = str(_bid_record_value(record, "action", "")).lower()
        if action not in action_index:
            continue
        index = action_index[action]
        counts[seat * 4 + index] = min(1.0, counts[seat * 4 + index] + 0.5)
        last_actor = seat
        last_action = index
        if action != "pass":
            standing = index
            pass_streak = 0
        else:
            pass_streak += 1
    return (
        counts,
        _one_hot(standing, 4),
        _one_hot(last_actor, 4),
        _one_hot(last_action, 4),
        min(1.0, len(items) / 12.0),
        min(1.0, pass_streak / 4.0),
    )


def _decision_envelope(
    *,
    phase: int,
    observer: Seat,
    purchase_suit: Suit | None,
    records: Iterable[Any],
    auction_actor: Seat | None = None,
    bidding_round: int | None = None,
) -> list[float]:
    counts, standing, last_actor, last_action, progress, pass_streak = _bid_summary(records)
    values: list[float] = []
    values.extend(_one_hot(phase, 4))
    values.extend(_one_hot(int(observer), 4))
    values.extend(_one_hot(int(auction_actor) if auction_actor is not None else None, 4))
    values.extend(_one_hot((int(bidding_round) - 1) if bidding_round in {1, 2} else None, 2))
    values.extend(counts)
    values.extend(standing)
    values.extend(_one_hot(int(purchase_suit) if purchase_suit is not None else None, 4))
    values.extend(last_actor)
    values.extend(last_action)
    values.extend((progress, pass_streak))
    if len(values) != 48:
        raise AssertionError(f"decision envelope drift: {len(values)} != 48")
    return values


def _base_features(
    *,
    observer_hand: int,
    played_mask: int,
    known_unplayed: tuple[int, int, int, int],
    hand_sizes: tuple[int, int, int, int],
    dealer: Seat,
    buyer: Seat | None,
    declarer: Seat | None,
    current_player: Seat | None,
    first_leader: Seat | None,
    contract: Contract | None,
    trump_suit: Suit | None,
    escalation: str,
    match_score: tuple[int, int],
    trick_index: int,
    locked: bool,
    trump_broken: bool,
    trick_position: float,
    baloot_team_count: int,
    purchase_card_id: int | None,
) -> list[float]:
    values: list[float] = []
    values.extend(_mask_bits(observer_hand))
    values.extend(_mask_bits(played_mask))
    for seat in Seat:
        values.extend(_mask_bits(known_unplayed[int(seat)]))
    values.extend(size / 8.0 for size in hand_sizes)
    values.extend(_one_hot(int(dealer), 4))
    values.extend(_one_hot(int(buyer) if buyer is not None else None, 4))
    values.extend(_one_hot(int(declarer) if declarer is not None else None, 4))
    values.extend(_one_hot(int(current_player) if current_player is not None else None, 4))
    values.extend(_one_hot(int(first_leader) if first_leader is not None else None, 4))
    contract_index = {Contract.SUN: 0, Contract.HOKUM: 1, Contract.ASHKAL: 2}.get(contract)
    values.extend(_one_hot(contract_index, 3))
    values.extend(_one_hot(int(trump_suit) if trump_suit is not None else None, 4))
    values.extend(_one_hot({"normal": 0, "double": 1, "triple": 2, "four": 3, "gahwa": 4}.get(escalation, 0), 5))
    values.extend(min(2.0, max(0.0, score / 152.0)) for score in match_score)
    values.extend(_one_hot(max(0, min(7, trick_index)), 8))
    values.extend((
        1.0 if locked else 0.0,
        1.0 if trump_broken else 0.0,
        max(0.0, min(1.0, trick_position)),
        max(0.0, min(1.0, baloot_team_count / 2.0)),
    ))
    values.extend(_one_hot(purchase_card_id, 32))
    if len(values) != 274:
        raise AssertionError(f"base feature drift: {len(values)} != 274")
    return values


def encode_information_state(state: InformationState) -> np.ndarray:
    """Encode observer-safe play information using the unified V2 contract."""
    played_mask = 0
    for play in state.plays:
        played_mask |= 1 << play.card.id
    values = _base_features(
        observer_hand=state.observer_hand,
        played_mask=played_mask,
        known_unplayed=state.known_unplayed,
        hand_sizes=state.hand_sizes,
        dealer=state.dealer,
        buyer=state.buyer,
        declarer=state.declarer,
        current_player=state.current_player,
        first_leader=state.first_leader,
        contract=state.contract,
        trump_suit=state.trump_suit,
        escalation=state.escalation,
        match_score=state.match_score,
        trick_index=min(7, len(state.plays) // 4),
        locked=state.locked,
        trump_broken=state.trump_broken,
        trick_position=(len(state.plays) % 4) / 3.0,
        baloot_team_count=len(state.baloot_teams),
        purchase_card_id=state.purchase_card.id if state.purchase_card is not None else None,
    )
    values.extend(
        _decision_envelope(
            phase=2,
            observer=state.observer,
            purchase_suit=state.purchase_card.suit if state.purchase_card is not None else None,
            records=state.bid_history,
        )
    )
    if len(values) != FEATURE_COUNT:
        raise AssertionError(f"feature contract drift: {len(values)} != {FEATURE_COUNT}")
    return np.asarray(values, dtype=np.float32)


def encode_bid_state(state: RoundState, machine: BiddingMachine, actor: Seat) -> np.ndarray:
    """Encode a bidding decision without reading partner or opponent hands.

    The only private cards included are the acting player's current five cards.
    Other hands are represented only by public sizes. The purchase card and
    auction history are public. This function is safe for supervised learning,
    policy inference, and challenge mining.
    """
    if state.purchase_card is None:
        raise ValueError("purchase card is required for bidding features")
    if machine.finished or machine.actor != actor:
        raise ValueError("actor must be the current bidder")
    known = [0, 0, 0, 0]
    known[int(actor)] = state.hands[int(actor)]
    values = _base_features(
        observer_hand=state.hands[int(actor)],
        played_mask=0,
        known_unplayed=tuple(known),
        hand_sizes=(5, 5, 5, 5),
        dealer=state.dealer,
        buyer=None,
        declarer=None,
        current_player=actor,
        first_leader=None,
        contract=None,
        trump_suit=None,
        escalation="normal",
        match_score=tuple(state.match_score),
        trick_index=0,
        locked=False,
        trump_broken=False,
        trick_position=0.0,
        baloot_team_count=0,
        purchase_card_id=state.purchase_card.id,
    )
    values.extend(
        _decision_envelope(
            phase=0 if machine.round_index == 1 else 1,
            observer=actor,
            purchase_suit=state.purchase_card.suit,
            records=machine.records,
            auction_actor=actor,
            bidding_round=machine.round_index,
        )
    )
    if len(values) != FEATURE_COUNT:
        raise AssertionError(f"bid feature contract drift: {len(values)} != {FEATURE_COUNT}")
    return np.asarray(values, dtype=np.float32)


def feature_names() -> tuple[str, ...]:
    names: list[str] = []
    names.extend(f"observer_hand_{i}" for i in range(32))
    names.extend(f"played_{i}" for i in range(32))
    for seat in Seat:
        names.extend(f"known_{seat.name.lower()}_{i}" for i in range(32))
    names.extend(f"hand_size_{seat.name.lower()}" for seat in Seat)
    for group in ("dealer", "buyer", "declarer", "current", "first_leader"):
        names.extend(f"{group}_{seat.name.lower()}" for seat in Seat)
    names.extend(("contract_sun", "contract_hokum", "contract_ashkal"))
    names.extend(f"trump_{suit.name.lower()}" for suit in Suit)
    names.extend(("escalation_normal", "escalation_double", "escalation_triple", "escalation_four", "escalation_gahwa"))
    names.extend(("match_us", "match_them"))
    names.extend(f"trick_{i}" for i in range(8))
    names.extend(("locked", "trump_broken", "trick_position", "baloot_team_count"))
    names.extend(f"purchase_{i}" for i in range(32))
    names.extend(("phase_bid_round_1", "phase_bid_round_2", "phase_play", "phase_terminal"))
    names.extend(f"observer_{seat.name.lower()}" for seat in Seat)
    names.extend(f"auction_actor_{seat.name.lower()}" for seat in Seat)
    names.extend(("bidding_round_1", "bidding_round_2"))
    for seat in Seat:
        for action in ("pass", "sun", "hokum", "ashkal"):
            names.append(f"bid_count_{seat.name.lower()}_{action}")
    names.extend(("standing_pass", "standing_sun", "standing_hokum", "standing_ashkal"))
    names.extend(f"purchase_suit_{suit.name.lower()}" for suit in Suit)
    names.extend(f"last_bidder_{seat.name.lower()}" for seat in Seat)
    names.extend(("last_action_pass", "last_action_sun", "last_action_hokum", "last_action_ashkal"))
    names.extend(("auction_progress", "pass_streak"))
    if len(names) != FEATURE_COUNT:
        raise AssertionError(f"feature-name contract drift: {len(names)} != {FEATURE_COUNT}")
    return tuple(names)
