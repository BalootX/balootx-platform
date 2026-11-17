from __future__ import annotations

import copy
import hashlib
import json

import pytest

from baloot_platform.analysis.bidding import analyze_bid
from baloot_platform.analysis.evaluator import analyze
from baloot_platform.analysis.models import InformationState
from baloot_platform.analysis.search_backend import NativeExactSearchBackendV3
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.native.accelerator import NativeAccelerator, NativeUnavailable


def _native_backend(rules) -> NativeExactSearchBackendV3:
    try:
        native = NativeAccelerator(rules)
    except (NativeUnavailable, OSError):
        pytest.skip("native V3 library is not built in this test lane")
    if not native.supports_compact_rollout:
        pytest.skip("native library does not expose compact rollout ABI V3")
    return NativeExactSearchBackendV3(native)


def _playing_state(rules, *, seed: int = 73021):
    engine = BalootEngine(rules)
    state = engine.create_round(seed=seed, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state, cut_position=7)
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.HOKUM,
        forced_buyer=Seat.SOUTH,
        forced_trump=Suit.HEARTS,
    )
    engine.prepare_projects(state)
    assert state.current_player is not None
    return engine, state


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_compact_native_results_match_full_results_without_retaining_pv(rules):
    engine, state = _playing_state(rules)
    native = _native_backend(rules).native
    cards = tuple(engine.legal_moves(state))[:3]
    states = tuple(state for _card in cards)

    full = native.heuristic_rollout_batch(states, cards, state.current_player.team)
    compact = native.heuristic_rollout_batch_compact(
        states, cards, state.current_player.team
    )

    assert len(compact) == len(full)
    for summary, detail in zip(compact, full, strict=True):
        assert (
            summary.value,
            summary.own_points,
            summary.opponent_points,
            summary.exact,
            summary.nodes,
            summary.contract_succeeded,
            summary.round_lead,
            summary.score_after_round,
            summary.match_winner,
            summary.completion_status,
        ) == (
            detail.value,
            detail.own_points,
            detail.opponent_points,
            detail.exact,
            detail.nodes,
            detail.contract_succeeded,
            detail.round_lead,
            detail.score_after_round,
            detail.match_winner,
            detail.completion_status,
        )
        assert not hasattr(summary, "principal_variation")


def test_compact_play_analysis_preserves_exact_evidence_hash(rules, convention):
    _engine, state = _playing_state(rules, seed=73022)
    assert state.current_player is not None
    info = InformationState.from_complete(state, state.current_player)
    compact_backend = _native_backend(rules)
    full_backend = _native_backend(rules)
    full_backend.supports_compact_rollout = False
    kwargs = {
        "requested_samples": 12,
        "exact_search_max_remaining": 4,
        "maximum_sampling_attempts": 20_000,
        "seed": 913,
    }

    full = analyze(
        info,
        rules,
        convention,
        exact_search_backend=full_backend,
        **kwargs,
    )
    compact = analyze(
        info,
        rules,
        convention,
        exact_search_backend=compact_backend,
        **kwargs,
    )

    assert compact == full
    assert _hash(compact.to_dict()) == _hash(full.to_dict())


def test_compact_bid_analysis_preserves_exact_evidence_hash(rules):
    engine = BalootEngine(rules)
    state = engine.create_round(seed=73023, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state, cut_position=7)
    assert state.purchase_card is not None
    machine = BiddingMachine(
        state.dealer,
        state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(rules.data),
    )
    assert machine.actor is not None
    compact_backend = _native_backend(rules)
    full_backend = _native_backend(rules)
    full_backend.supports_compact_rollout = False

    full = analyze_bid(
        state,
        copy.deepcopy(machine),
        machine.actor,
        rules,
        requested_samples=12,
        seed=191,
        exact_search_backend=full_backend,
    )
    compact = analyze_bid(
        state,
        copy.deepcopy(machine),
        machine.actor,
        rules,
        requested_samples=12,
        seed=191,
        exact_search_backend=compact_backend,
    )

    assert compact == full
    assert _hash(compact.to_dict()) == _hash(full.to_dict())


def test_compact_native_batch_honors_cancellation(rules):
    engine, state = _playing_state(rules, seed=73024)
    native = _native_backend(rules).native
    card = engine.legal_moves(state)[0]

    with pytest.raises(InterruptedError, match="native rollout batch cancelled"):
        native.heuristic_rollout_batch_compact(
            (state,),
            (card,),
            state.current_player.team,
            cancel_check=lambda: True,
        )
