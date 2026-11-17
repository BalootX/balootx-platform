from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from baloot_platform.core.cards import Card, cards_to_mask
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, GamePhase, Rank, Seat, Suit
from baloot_platform.core.models import PlayedCard, RoundState, Trick
from baloot_platform.core.play_ir import PlayDomainIRV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.native.accelerator import NativeAccelerator
from baloot_platform.research import (
    OrToolsCpSatPlayCompilerV1,
    ReferencePlayInterpreterV1,
    ReferencePlayV1,
)
from baloot_platform.service.container import build_container


@pytest.fixture(scope="module")
def ekka_trace_fixture():
    container = build_container()
    simulator = container.research.simulator
    return simulator.rules, simulator.simulate(3).final_trace.trusted_final_state


def _runtime_state(
    rules: RuleBook,
    hand: tuple[Card, ...],
    plays: tuple[ReferencePlayV1, ...],
    *,
    actor: Seat,
    trump: Suit,
    locked: bool = False,
) -> RoundState:
    state = RoundState(rules.sha256, 1, 1, Seat.EAST)
    state.contract = Contract.HOKUM
    state.buyer = Seat.SOUTH
    state.declarer = Seat.SOUTH
    state.trump_suit = trump
    state.first_leader = plays[0].seat if plays else actor
    state.phase = GamePhase.PLAYING
    state.locked = locked
    state.hands = [0, 0, 0, 0]
    state.hands[int(actor)] = cards_to_mask(hand)
    state.initial_hands = list(state.hands)
    trick = Trick(0, plays[0].seat if plays else actor)
    for ordinal, play in enumerate(plays):
        trick.plays.append(
            PlayedCard(play.seat, play.card, 0, ordinal, (), play.ekka)
        )
        state.initial_hands[int(play.seat)] |= 1 << play.card.id
    state.current_trick = trick
    state.current_player = actor
    return state


def test_rulebook_owns_one_exact_play_and_project_ir(rules: RuleBook):
    engine = BalootEngine(rules)
    reference = ReferencePlayInterpreterV1(
        rules.play_domain,
        rules.structural_domain,
    )
    assert engine.play_domain is rules.play_domain
    assert engine.project_domain is rules.project_domain
    assert reference.domain is rules.play_domain
    assert reference.structural_domain is rules.structural_domain
    assert (
        rules.play_domain.structural_domain_ir_hash
        == rules.structural_domain.sha256
    )


def test_play_ir_rejects_forged_and_cross_bundle_dependencies(rules: RuleBook):
    forged = replace(
        rules.play_domain,
        allow_discard_when_cannot_overtrump=False,
    )
    with pytest.raises(ValueError, match="integrity mismatch"):
        forged.require_integrity(rules.structural_domain)

    changed = copy.deepcopy(rules.data)
    changed["dealing"]["cut_enabled"] = False
    changed_rules = RuleBook(changed)
    with pytest.raises(ValueError, match="structural dependency mismatch"):
        rules.play_domain.require_integrity(changed_rules.structural_domain)
    with pytest.raises(ValueError, match="structural dependency mismatch"):
        BalootEngine(changed_rules, play_domain=rules.play_domain)


@pytest.mark.parametrize(
    ("table", "value", "message"),
    (
        ("sun_strength", [0, 1, 2, 3, 4, 5, 6, 6], "permutation"),
        ("hokum_trump_strength", [-1, 1, 2, 3, 4, 5, 6, 7], "permutation"),
        ("sun_points", [0, 0, 0, 10, 2, 3, 4, 32_768], "native-safe"),
        ("hokum_trump_points", [0, 0, 14, 10, True, 3, 4, 11], "eight integers"),
    ),
)
def test_play_ir_table_bounds_fail_closed(
    rules: RuleBook,
    table: str,
    value: list[int],
    message: str,
):
    changed = copy.deepcopy(rules.data)
    changed["card_rules"][table] = value
    with pytest.raises(ValueError, match=message):
        PlayDomainIRV1.from_rules(
            changed,
            structural_domain=rules.structural_domain,
        )


def test_sequence_strength_is_owned_only_by_project_ir(rules: RuleBook):
    changed = copy.deepcopy(rules.data)
    changed["card_rules"]["sequence_strength"] = [0, 2, 1, 3, 4, 5, 6, 7]
    changed_rules = RuleBook(changed)
    assert changed_rules.play_domain.sha256 == rules.play_domain.sha256
    assert changed_rules.project_domain.sha256 != rules.project_domain.sha256
    assert (
        changed_rules.sequence_strength(Card.of(Suit.SPADES, Rank.EIGHT))
        == 2
    )


def test_reference_card_algorithms_do_not_delegate_to_runtime_ir_methods(
    rules: RuleBook,
    monkeypatch: pytest.MonkeyPatch,
):
    reference = ReferencePlayInterpreterV1(
        rules.play_domain,
        rules.structural_domain,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("reference delegated to the runtime IR algorithm")

    monkeypatch.setattr(PlayDomainIRV1, "strength", forbidden)
    monkeypatch.setattr(PlayDomainIRV1, "points", forbidden)
    assert reference.strength(Card.parse("HJ"), Contract.HOKUM, Suit.HEARTS) == 7
    assert reference.points(Card.parse("HJ"), Contract.HOKUM, Suit.HEARTS) == 20


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("trump", tuple(Suit))
def test_partner_exception_is_seat_and_suit_symmetric(
    rules: RuleBook,
    rotation: int,
    trump: Suit,
):
    structural = rules.structural_domain
    leader = structural.offset(Seat.NORTH, rotation)
    actor = structural.partner(leader)
    second = structural.offset(leader, 1)
    led_suit = Suit((int(trump) + 1) % 4)
    discard_suit = Suit((int(trump) + 2) % 4)
    plays = (
        ReferencePlayV1(leader, Card.of(led_suit, Rank.SEVEN)),
        ReferencePlayV1(second, Card.of(discard_suit, Rank.SEVEN)),
    )
    hand = (
        Card.of(trump, Rank.SEVEN),
        Card.of(discard_suit, Rank.EIGHT),
    )
    reference = ReferencePlayInterpreterV1(rules.play_domain, structural)
    expected = reference.legal_moves(
        hand,
        plays,
        actor=actor,
        contract=Contract.HOKUM,
        trump=trump,
    )
    runtime = BalootEngine(rules).legal_moves(
        _runtime_state(rules, hand, plays, actor=actor, trump=trump)
    )
    assert expected == runtime == (Card.of(trump, Rank.SEVEN),)


def test_native_rules_are_lowered_from_the_exact_play_ir(rules: RuleBook):
    lowered = NativeAccelerator._build_rules_v2(rules)
    domain = rules.play_domain
    assert [*lowered.strengths[0]] == list(domain.sun_strength)
    assert [*lowered.strengths[1]] == list(domain.hokum_non_trump_strength)
    assert [*lowered.strengths[2]] == list(domain.hokum_trump_strength)
    assert [*lowered.points[0]] == list(domain.sun_points)
    assert [*lowered.points[1]] == list(domain.hokum_non_trump_points)
    assert [*lowered.points[2]] == list(domain.hokum_trump_points)
    assert lowered.must_follow_suit == int(domain.must_follow_suit)
    assert lowered.must_trump_when_void == int(domain.must_trump_when_void)
    assert lowered.must_overtrump_when_possible == int(
        domain.must_overtrump_when_possible
    )
    assert lowered.allow_discard_when_cannot_overtrump == int(
        domain.allow_discard_when_cannot_overtrump
    )
    assert lowered.partner_winning_exception == 1
    assert lowered.locked_hokum_blocks_trump_lead == 1
    assert lowered.ekka_enabled == 1


@pytest.mark.parametrize(("trick_index", "accepted"), ((0, False), (4, True)))
def test_cp_sat_full_trace_enforces_ekka_legality(
    trick_index: int,
    accepted: bool,
    ekka_trace_fixture,
):
    rules, trace_state = ekka_trace_fixture
    state = copy.deepcopy(trace_state)
    assert state["contract"] == "hokum"
    state["completedTricks"][trick_index]["plays"][0]["ekka"] = True
    compiler = OrToolsCpSatPlayCompilerV1()
    if accepted:
        plan = compiler.verify_trace(state, rules=rules)
        assert plan.tricks[trick_index].ekka_lead is True
    else:
        with pytest.raises(ValueError, match="semantically infeasible"):
            compiler.verify_trace(state, rules=rules)


def test_cp_sat_trace_rejects_non_lead_ekka(ekka_trace_fixture):
    rules, trace_state = ekka_trace_fixture
    state = copy.deepcopy(trace_state)
    state["completedTricks"][4]["plays"][1]["ekka"] = True
    with pytest.raises(ValueError, match="only permitted on a trick lead"):
        OrToolsCpSatPlayCompilerV1().verify_trace(state, rules=rules)
