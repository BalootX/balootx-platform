from __future__ import annotations

import copy

import pytest

from baloot_platform.analysis.bidding import analyze_bid
from baloot_platform.analysis.search_backend import NativeExactSearchBackendV3
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.native.accelerator import NativeAccelerator, NativeUnavailable


def _auction(rules, *, seed: int = 73001):
    engine = BalootEngine(rules)
    state = engine.create_round(seed=seed, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state, cut_position=7)
    assert state.purchase_card is not None
    machine = BiddingMachine(
        state.dealer,
        state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(rules.data),
    )
    assert machine.actor is not None
    return state, machine, machine.actor


def _native_backend(rules) -> NativeExactSearchBackendV3:
    try:
        native = NativeAccelerator(rules)
    except (NativeUnavailable, OSError):
        pytest.skip("native V3 library is not built in this test lane")
    if not native.supports_v3:
        pytest.skip("native library does not expose rollout ABI V3")
    return NativeExactSearchBackendV3(native)


def _evidence(result):
    return tuple(
        (
            item.choice.id,
            item.expected_match_margin,
            item.team_win_probability,
            item.contract_success_probability,
            item.downside_cvar10,
            item.volatility,
            item.best_share,
            item.samples,
        )
        for item in result.choices
    )


def test_native_bid_batch_matches_reference_and_preserves_input(rules):
    state, machine, actor = _auction(rules)
    original_state = copy.deepcopy(state)
    original_machine = copy.deepcopy(machine)
    reference = analyze_bid(
        state,
        machine,
        actor,
        rules,
        requested_samples=6,
        seed=913,
    )
    accelerated = analyze_bid(
        state,
        machine,
        actor,
        rules,
        requested_samples=6,
        seed=913,
        exact_search_backend=_native_backend(rules),
    )

    assert accelerated.method == "paired_observer_safe_bid_native_batch_v2"
    assert accelerated.recommended == reference.recommended
    assert _evidence(accelerated) == _evidence(reference)
    assert accelerated.difficulty == reference.difficulty
    assert accelerated.debate_score == reference.debate_score
    assert state == original_state
    assert machine == original_machine


def test_native_bid_batch_is_deterministic(rules):
    state, machine, actor = _auction(rules, seed=73002)
    backend = _native_backend(rules)
    first = analyze_bid(
        state,
        copy.deepcopy(machine),
        actor,
        rules,
        requested_samples=5,
        seed=191,
        exact_search_backend=backend,
    )
    second = analyze_bid(
        state,
        copy.deepcopy(machine),
        actor,
        rules,
        requested_samples=5,
        seed=191,
        exact_search_backend=backend,
    )
    assert first == second


def test_native_bid_batch_preserves_second_round_hokum_results(rules):
    state, machine, _actor = _auction(rules, seed=73004)
    for _ in range(4):
        assert machine.submit("pass") is None
    assert machine.round_index == 2
    assert machine.actor is not None
    actor = machine.actor
    reference = analyze_bid(
        state,
        copy.deepcopy(machine),
        actor,
        rules,
        requested_samples=4,
        seed=419,
    )
    accelerated = analyze_bid(
        state,
        copy.deepcopy(machine),
        actor,
        rules,
        requested_samples=4,
        seed=419,
        exact_search_backend=_native_backend(rules),
    )
    assert accelerated.recommended == reference.recommended
    assert _evidence(accelerated) == _evidence(reference)
    assert {item.choice.id for item in accelerated.choices} == {
        "pass",
        "sun",
        *(f"hokum:{suit.name.lower()}" for suit in type(state.purchase_card.suit) if suit != state.purchase_card.suit),
    }


def test_native_bid_batch_honors_cancellation(rules):
    state, machine, actor = _auction(rules, seed=73003)
    with pytest.raises(InterruptedError, match="bid analysis cancelled"):
        analyze_bid(
            state,
            machine,
            actor,
            rules,
            requested_samples=4,
            seed=29,
            cancel_check=lambda: True,
            exact_search_backend=_native_backend(rules),
        )
