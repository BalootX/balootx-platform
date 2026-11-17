from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, GamePhase, Rank, Seat, Suit, Team
from baloot_platform.core.models import RoundState, Trick
from baloot_platform.research.ortools_play_state_predicates import (
    OBSERVER_VISIBILITY_CLASSIFICATIONS_V1,
    PLAY_STATE_PREDICATE_KINDS_V1,
    PLAY_STATE_PREDICATE_REGISTRATION_HASH_V1,
    PLAY_STATE_PREDICATE_REGISTRATIONS_V1,
    UNSUPPORTED_PLAY_STATE_PREDICATE_KINDS_V1,
    CanonicalObserverScopeV1,
    CpSatPlayPrefixActionV1,
    CpSatPlayStatePredicateContextV1,
    OrToolsPlayStatePredicateLoweringV1,
)
from baloot_platform.research.scenario_contracts import ScenarioPredicateV1


def _predicate(kind: str, arguments: dict) -> ScenarioPredicateV1:
    return ScenarioPredicateV1.build(
        {
            "predicateId": "PRED-" + kind.replace(".", "-").upper(),
            "kind": kind,
            "version": 1,
            "hardness": "hard",
            "arguments": arguments,
        }
    )


def _action(
    model: cp_model.CpModel,
    card: int,
    actor: Seat,
    *,
    ekka: bool = False,
    present: bool = True,
) -> CpSatPlayPrefixActionV1:
    return CpSatPlayPrefixActionV1(
        model.new_constant(card),
        model.new_constant(int(actor)),
        model.new_constant(int(ekka)),
        model.new_constant(int(present)),
    )


def _context(
    *,
    include_visibility: bool = True,
    variable_next_actor: bool = False,
    observer_scope: CanonicalObserverScopeV1 | None = None,
) -> CpSatPlayStatePredicateContextV1:
    model = cp_model.CpModel()
    cards = [
        Card.of(Suit.SPADES, Rank.SEVEN).id,
        Card.of(Suit.SPADES, Rank.ACE).id,
        Card.of(Suit.SPADES, Rank.EIGHT).id,
        Card.of(Suit.SPADES, Rank.NINE).id,
        Card.of(Suit.HEARTS, Rank.SEVEN).id,
        Card.of(Suit.HEARTS, Rank.ACE).id,
    ]
    actors = [Seat.NORTH, Seat.WEST, Seat.SOUTH, Seat.EAST, Seat.WEST, Seat.NORTH]
    present = [
        _action(model, card, actor) for card, actor in zip(cards, actors, strict=True)
    ]
    absent = [_action(model, 0, Seat.NORTH, present=False) for _ in range(26)]
    slots = tuple([*present, *absent])
    current = tuple([slots[4], slots[5], absent[0], absent[1]])
    legal_cards = {
        card: model.new_constant(
            int(
                card
                in {
                    Card.of(Suit.DIAMONDS, Rank.SEVEN).id,
                    Card.of(Suit.CLUBS, Rank.SEVEN).id,
                }
            )
        )
        for card in range(32)
    }
    legal_actions = {
        (card, ekka): model.new_constant(
            int(
                not ekka
                and card
                in {
                    Card.of(Suit.DIAMONDS, Rank.SEVEN).id,
                    Card.of(Suit.CLUBS, Rank.SEVEN).id,
                }
            )
        )
        for card in range(32)
        for ekka in (False, True)
    }
    visibility = (
        {
            (Seat.SOUTH, Card.of(Suit.DIAMONDS, Rank.SEVEN).id): model.new_constant(1),
            (Seat.SOUTH, Card.of(Suit.CLUBS, Rank.ACE).id): model.new_constant(0),
        }
        if include_visibility
        else None
    )
    return CpSatPlayStatePredicateContextV1(
        model=model,
        context_id="fixed-full-prefix",
        play_slots=slots,
        prefix_length=model.new_constant(6),
        trick_index=model.new_constant(1),
        next_actor=(
            model.new_int_var(0, 4, "live_next_actor")
            if variable_next_actor
            else model.new_constant(int(Seat.SOUTH))
        ),
        remaining_hand_sizes={
            Seat.NORTH: model.new_constant(6),
            Seat.WEST: model.new_constant(6),
            Seat.SOUTH: model.new_constant(7),
            Seat.EAST: model.new_constant(7),
        },
        current_trick_slots=current,
        current_trick_size=model.new_constant(2),
        current_trick_active=model.new_constant(1),
        current_trick_leader=model.new_constant(int(Seat.WEST)),
        current_trick_suit_led=model.new_constant(int(Suit.HEARTS)),
        trump_suit=model.new_constant(int(Suit.HEARTS)),
        completed_trick_literals={
            index: model.new_constant(int(index == 0)) for index in range(8)
        },
        completed_trick_winners={
            index: model.new_constant(int(Seat.WEST if index == 0 else Seat.NORTH))
            for index in range(8)
        },
        legal_card_literals=legal_cards,
        legal_action_literals=legal_actions,
        phase=model.new_constant(tuple(GamePhase).index(GamePhase.PLAYING)),
        match_scores={
            Team.NORTH_SOUTH: model.new_constant(12),
            Team.WEST_EAST: model.new_constant(18),
        },
        raw_points={
            Team.NORTH_SOUTH: model.new_constant(0),
            Team.WEST_EAST: model.new_constant(31),
        },
        project_points={
            Team.NORTH_SOUTH: model.new_constant(0),
            Team.WEST_EAST: model.new_constant(100),
        },
        game_points={
            Team.NORTH_SOUTH: model.new_constant(0),
            Team.WEST_EAST: model.new_constant(3),
        },
        card_visibility_literals=visibility,
        observer_scope=observer_scope,
    )


def _complete_state(rules) -> dict:
    state = RoundState(rules.sha256, 91, 3, Seat.EAST)
    state.phase = GamePhase.PLAYING
    state.match_score = [42, 31]
    state.purchase_card = Card(0)
    state.initial_hands = [
        sum(1 << card for card in range(int(seat) * 8, (int(seat) + 1) * 8))
        for seat in Seat
    ]
    state.hands = list(state.initial_hands)
    state.contract = Contract.SUN
    state.buyer = Seat.NORTH
    state.declarer = Seat.NORTH
    state.first_leader = Seat.SOUTH
    state.current_player = Seat.SOUTH
    state.current_trick = Trick(0, Seat.SOUTH)
    return state.to_dict(reveal_hands=True)


def _swap_cards(state: dict, first: str, second: str) -> None:
    for key in ("hands", "initialHands"):
        state[key][first][1], state[key][second][1] = (
            state[key][second][1],
            state[key][first][1],
        )


def _solve(model: cp_model.CpModel) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    return solver


TRUE_CASES = [
    ("play.trick_index", {"minimum": 1, "maximum": 1}),
    ("play.current_player", {"seat": "south"}),
    (
        "play.remaining_hand_size",
        {"seat": "south", "minimum": 7, "maximum": 7},
    ),
    (
        "play.card_prefix",
        {
            "cards": [
                Card.of(Suit.SPADES, Rank.SEVEN).id,
                Card.of(Suit.SPADES, Rank.ACE).id,
            ]
        },
    ),
    (
        "play.action_prefix",
        {
            "actions": [
                {
                    "card": Card.of(Suit.SPADES, Rank.SEVEN).id,
                    "ekka": False,
                },
                {"card": Card.of(Suit.SPADES, Rank.ACE).id, "ekka": False},
            ]
        },
    ),
    ("play.current_trick_leader", {"seat": "west"}),
    ("play.current_trick_suit_led", {"suit": "hearts"}),
    ("play.trump_use_count", {"minimum": 2, "maximum": 2}),
    (
        "play.current_trick_card",
        {
            "position": 1,
            "seat": "north",
            "card": Card.of(Suit.HEARTS, Rank.ACE).id,
            "ekka": False,
        },
    ),
    ("play.completed_trick_winner", {"trickIndex": 0, "seat": "west"}),
    ("state.legal_action_count", {"minimum": 2, "maximum": 2}),
    (
        "state.match_score",
        {"team": "west_east", "minimum": 18, "maximum": 18},
    ),
    ("state.phase", {"state": "playing"}),
    ("state.next_actor", {"seat": "south"}),
    (
        "state.observer_card_visibility",
        {
            "observer": "south",
            "card": Card.of(Suit.CLUBS, Rank.ACE).id,
            "visibility": "hidden",
        },
    ),
    (
        "state.current_raw_points",
        {"team": "west_east", "minimum": 31, "maximum": 31},
    ),
    (
        "state.current_project_points",
        {"team": "west_east", "minimum": 100, "maximum": 100},
    ),
    (
        "state.current_game_points",
        {"team": "west_east", "minimum": 3, "maximum": 3},
    ),
    (
        "state.legal_cards_include",
        {"cards": [Card.of(Suit.DIAMONDS, Rank.SEVEN).id]},
    ),
    (
        "state.legal_play_actions_include",
        {
            "actions": [
                {
                    "card": Card.of(Suit.CLUBS, Rank.SEVEN).id,
                    "ekka": False,
                }
            ]
        },
    ),
]


@pytest.mark.parametrize(("kind", "arguments"), TRUE_CASES)
def test_every_supported_play_state_predicate_has_exact_lowering_and_evidence(
    kind, arguments
):
    context = _context()
    lowering = OrToolsPlayStatePredicateLoweringV1().lower_predicate(
        _predicate(kind, arguments), context
    )
    context.model.add(lowering.literal == 1)
    solver = _solve(context.model)
    evidence = lowering.extract(solver)

    assert solver.boolean_value(lowering.literal)
    assert evidence["satisfied"] is True
    assert evidence["kind"] == kind
    assert len(evidence["sha256"]) == 64


FALSE_CASES = [
    ("play.trick_index", {"minimum": 2, "maximum": 2}),
    ("play.current_player", {"seat": "north"}),
    ("play.remaining_hand_size", {"seat": "south", "minimum": 6, "maximum": 6}),
    ("play.card_prefix", {"cards": [Card.of(Suit.SPADES, Rank.ACE).id]}),
    ("play.action_prefix", {"actions": [{"card": Card.of(Suit.SPADES, Rank.SEVEN).id, "ekka": True}]}),
    ("play.current_trick_leader", {"seat": "north"}),
    ("play.current_trick_suit_led", {"suit": "spades"}),
    ("play.trump_use_count", {"minimum": 1, "maximum": 1}),
    ("play.current_trick_card", {"position": 1, "seat": "south",
        "card": Card.of(Suit.HEARTS, Rank.ACE).id, "ekka": False}),
    ("play.completed_trick_winner", {"trickIndex": 0, "seat": "south"}),
    ("state.legal_action_count", {"minimum": 1, "maximum": 1}),
    ("state.match_score", {"team": "west_east", "minimum": 19, "maximum": 19}),
    ("state.phase", {"state": "bidding"}),
    ("state.next_actor", {"seat": "north"}),
    ("state.observer_card_visibility", {"observer": "south",
        "card": Card.of(Suit.CLUBS, Rank.ACE).id, "visibility": "visible"}),
    ("state.current_raw_points", {"team": "west_east", "minimum": 30, "maximum": 30}),
    ("state.current_project_points", {"team": "west_east", "minimum": 99, "maximum": 99}),
    ("state.current_game_points", {"team": "west_east", "minimum": 2, "maximum": 2}),
    ("state.legal_cards_include", {"cards": [Card.of(Suit.CLUBS, Rank.ACE).id]}),
    ("state.legal_play_actions_include", {"actions": [
        {"card": Card.of(Suit.CLUBS, Rank.SEVEN).id, "ekka": True}]}),
]


@pytest.mark.parametrize(("kind", "arguments"), FALSE_CASES)
def test_false_play_state_predicate_cannot_qualify(kind, arguments):
    context = _context()
    lowering = OrToolsPlayStatePredicateLoweringV1().lower_predicate(
        _predicate(kind, arguments), context)
    solver = _solve(context.model)
    assert not solver.boolean_value(lowering.literal)
    assert lowering.extract(solver)["satisfied"] is False
    context.model.add(lowering.literal == 1)
    assert solver.solve(context.model) == cp_model.INFEASIBLE


def test_registration_inventory_covers_all_21_v1_kinds_without_overclaiming():
    assert len(PLAY_STATE_PREDICATE_KINDS_V1) == 21
    assert len(PLAY_STATE_PREDICATE_REGISTRATIONS_V1) == 21
    assert not UNSUPPORTED_PLAY_STATE_PREDICATE_KINDS_V1
    assert {kind for kind, _ in TRUE_CASES} | {"state.observer_visibility"} == set(
        PLAY_STATE_PREDICATE_REGISTRATIONS_V1
    )
    assert {kind for kind, _ in FALSE_CASES} == {kind for kind, _ in TRUE_CASES}
    assert len(FALSE_CASES) == len({kind for kind, _ in FALSE_CASES})
    assert dict(OBSERVER_VISIBILITY_CLASSIFICATIONS_V1) == {
        "complete_state": "public_fact",
        "actor_information_set": "legal_deduction",
    }
    assert len(PLAY_STATE_PREDICATE_REGISTRATION_HASH_V1) == 64


def test_missing_observer_scope_and_unbound_card_visibility_fail_before_mutation():
    registry = OrToolsPlayStatePredicateLoweringV1()
    context = _context(include_visibility=False)
    before = len(context.model.proto.constraints)

    with pytest.raises(ValueError, match="requires a canonical observer scope"):
        registry.lower_predicate(
            _predicate(
                "state.observer_visibility",
                {"observer": "south", "classification": "legal_deduction"},
            ),
            context,
        )
    with pytest.raises(ValueError, match="requires exact observation variables"):
        registry.lower_predicate(
            _predicate(
                "state.observer_card_visibility",
                {
                    "observer": "south",
                    "card": Card.of(Suit.CLUBS, Rank.ACE).id,
                    "visibility": "hidden",
                },
            ),
            context,
        )
    assert len(context.model.proto.constraints) == before


def test_indistinguishable_worlds_bind_actor_information_scope_without_card_leakage(
    rules,
):
    first = _complete_state(rules)
    second = copy.deepcopy(first)
    _swap_cards(second, "north", "west")
    scope = CanonicalObserverScopeV1.build(
        visibility_scope="actor_information_set",
        acting_observer=Seat.SOUTH,
        complete_states=(first, second),
    )
    context = _context(observer_scope=scope)
    lowering = OrToolsPlayStatePredicateLoweringV1().lower_predicate(
        _predicate(
            "state.observer_visibility",
            {"observer": "south", "classification": "legal_deduction"},
        ),
        context,
    )
    context.model.add(lowering.literal == 1)
    evidence = lowering.extract(_solve(context.model))
    encoded = json.dumps(evidence, sort_keys=True)

    assert evidence["observed"]["visibilityScope"] == "actor_information_set"
    assert evidence["observed"]["actingObserver"] == "south"
    assert evidence["observed"]["hiddenWorldCount"] == 2
    assert len(evidence["observed"]["observerProjectionHash"]) == 64
    assert len(evidence["observed"]["hiddenWorldScopeHash"]) == 64
    assert "worldHashes" not in encoded
    assert "hands" not in encoded
    assert first["hands"]["north"][1]["code"] not in encoded
    assert second["hands"]["west"][1]["code"] not in encoded


def test_complete_state_scope_is_one_actor_bound_reference_world(rules):
    state = _complete_state(rules)
    scope = CanonicalObserverScopeV1.build(
        visibility_scope="complete_state",
        acting_observer="south",
        complete_states=(state,),
    )
    context = _context(observer_scope=scope)
    true_lowering = OrToolsPlayStatePredicateLoweringV1().lower_predicate(
        _predicate(
            "state.observer_visibility",
            {"observer": "south", "classification": "public_fact"},
        ),
        context,
    )
    false_lowering = OrToolsPlayStatePredicateLoweringV1().lower_predicate(
        _predicate(
            "state.observer_visibility",
            {"observer": "south", "classification": "policy_assumption"},
        ),
        context,
    )
    solver = _solve(context.model)

    assert solver.boolean_value(true_lowering.literal)
    assert not solver.boolean_value(false_lowering.literal)
    assert scope.hidden_world_count == 1
    context.model.add(false_lowering.literal == 1)
    assert solver.solve(context.model) == cp_model.INFEASIBLE


@pytest.mark.parametrize("observer,classification,expected", [
    ("south", "legal_deduction", True),
    ("north", "legal_deduction", False),
    ("south", "public_fact", False),
])
def test_native_observer_metadata_is_reified_not_python_truth(
    rules, monkeypatch, observer, classification, expected,
):
    first = _complete_state(rules)
    second = copy.deepcopy(first)
    _swap_cards(second, "north", "west")
    scope = CanonicalObserverScopeV1.build(visibility_scope="actor_information_set",
        acting_observer="south", complete_states=(first, second))
    context = replace(_context(observer_scope=scope), native_observer_scope=True)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("native observer lowering must not precompute truth constants")
    monkeypatch.setattr(context.model, "new_constant", forbidden)
    lowering = OrToolsPlayStatePredicateLoweringV1().lower_predicate(_predicate(
        "state.observer_visibility", {"observer": observer, "classification": classification}), context)
    solver = _solve(context.model)
    assert solver.boolean_value(lowering.literal) is expected
    observed = lowering.extract(solver)["observed"]
    assert observed == scope.to_safe_dict()
    assert set(observed) == {"schemaVersion", "visibilityScope", "actingObserver", "classification",
        "observerProjectionHash", "hiddenWorldScopeHash", "hiddenWorldCount", "sha256"}
    context.model.add(lowering.literal == int(not expected))
    assert solver.solve(context.model) == cp_model.INFEASIBLE


def test_default_observer_context_preserves_legacy_identity(rules):
    scope = CanonicalObserverScopeV1.build(visibility_scope="complete_state",
        acting_observer="south", complete_states=(_complete_state(rules),))
    context = _context(observer_scope=scope)
    assert replace(context, native_observer_scope=False).sha256 == context.sha256
    assert replace(context, native_observer_scope=True).sha256 != context.sha256


def test_observer_scope_rejects_unsupported_or_distinguishable_worlds(rules):
    state = _complete_state(rules)
    distinguishable = copy.deepcopy(state)
    _swap_cards(distinguishable, "south", "east")

    with pytest.raises(ValueError, match="scope is unsupported"):
        CanonicalObserverScopeV1.build(
            visibility_scope="sampled_information_set",
            acting_observer="south",
            complete_states=(state,),
        )
    with pytest.raises(ValueError, match="distinguishable"):
        CanonicalObserverScopeV1.build(
            visibility_scope="actor_information_set",
            acting_observer="south",
            complete_states=(state, distinguishable),
        )
    with pytest.raises(ValueError, match="bind the acting observer"):
        CanonicalObserverScopeV1.build(
            visibility_scope="complete_state",
            acting_observer="west",
            complete_states=(state,),
        )
    with pytest.raises(ValueError, match="exactly one world"):
        CanonicalObserverScopeV1.build(
            visibility_scope="complete_state",
            acting_observer="south",
            complete_states=(state, distinguishable),
        )


def test_lowering_constrains_live_cp_variable_not_only_fixed_evidence():
    context = _context(variable_next_actor=True)
    lowering = OrToolsPlayStatePredicateLoweringV1().lower_predicate(
        _predicate("state.next_actor", {"seat": "east"}), context
    )
    context.model.add(lowering.literal == 1)
    solver = _solve(context.model)

    assert solver.value(context.next_actor) == int(Seat.EAST)


def test_module_has_no_runtime_engine_or_bot_dependency():
    path = "src/baloot_platform/research/ortools_play_state_predicates.py"
    source = open(path, encoding="utf-8").read()
    forbidden = ("BalootEngine", "HeuristicBot", "runtime_verifier")

    assert not any(name in source for name in forbidden)
