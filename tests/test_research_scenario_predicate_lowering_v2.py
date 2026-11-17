from __future__ import annotations

import itertools

import pytest
from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Rank, Seat, Suit, Team
from baloot_platform.research.scenario_contracts import (
    ScenarioExpressionV1,
    ScenarioPredicateV1,
)
from baloot_platform.research.scenario_predicate_lowering_v2 import (
    CpSatActionVariablesV1,
    CpSatPredicateModelContextV1,
    CpSatTrickVariablesV1,
    ScenarioPredicateLoweringRegistryV2,
    _selector_matches,
)
from baloot_platform.research.scenario_registry_v2 import (
    EXACT_V1_COMPATIBILITY_KINDS,
    GENERAL_PREDICATE_KINDS_V2,
    CardSelectorV1,
    FLAGSHIP_PREDICATE_KINDS_V2,
    flagship_predicate_definitions_v2,
    scenario_predicate_definitions_v2,
)


def _predicate(predicate_id: str, kind: str, arguments: dict) -> dict:
    return {
        "predicateId": predicate_id,
        "kind": kind,
        "version": 1,
        "hardness": "hard",
        "arguments": arguments,
    }


def _action(
    model: cp_model.CpModel,
    action_id: str,
    card: int,
    actor: Seat,
    *,
    owner: Seat | None = None,
    ekka: bool = False,
    present: bool = True,
    winning: bool | None = None,
) -> CpSatActionVariablesV1:
    return CpSatActionVariablesV1(
        action_id,
        model.new_constant(card),
        model.new_constant(int(actor)),
        model.new_constant(int(owner if owner is not None else actor)),
        model.new_constant(int(ekka)),
        model.new_constant(int(present)),
        model.new_constant(int(winning)) if winning is not None else None,
    )


def _context(
    *,
    proof_complete: bool = True,
    root_wins: tuple[bool, ...] = (False, False, True),
) -> CpSatPredicateModelContextV1:
    model = cp_model.CpModel()
    trick_six_actions = (
        _action(model, "T6-N", Card.of(Suit.SPADES, Rank.KING).id, Seat.NORTH),
        _action(model, "T6-S", Card.of(Suit.HEARTS, Rank.TEN).id, Seat.SOUTH),
        _action(model, "T6-E", Card.of(Suit.DIAMONDS, Rank.NINE).id, Seat.EAST),
        _action(model, "T6-W", Card.of(Suit.CLUBS, Rank.ACE).id, Seat.WEST),
    )
    trick_seven_actions = (
        _action(model, "T7-W", Card.of(Suit.SPADES, Rank.SEVEN).id, Seat.WEST),
        _action(model, "T7-N", Card.of(Suit.HEARTS, Rank.KING).id, Seat.NORTH),
        _action(model, "T7-S", Card.of(Suit.DIAMONDS, Rank.QUEEN).id, Seat.SOUTH),
        _action(model, "T7-E", Card.of(Suit.CLUBS, Rank.EIGHT).id, Seat.EAST),
    )
    trick_eight_actions = (
        _action(model, "T8-W", Card.of(Suit.SPADES, Rank.SEVEN).id, Seat.WEST),
        _action(model, "T8-N", Card.of(Suit.HEARTS, Rank.ACE).id, Seat.NORTH),
        _action(model, "T8-S", Card.of(Suit.DIAMONDS, Rank.ACE).id, Seat.SOUTH),
        _action(model, "T8-E", Card.of(Suit.CLUBS, Rank.EIGHT).id, Seat.EAST),
    )
    root_cards = (
        Card.of(Suit.DIAMONDS, Rank.SEVEN).id,
        Card.of(Suit.DIAMONDS, Rank.EIGHT).id,
        Card.of(Suit.CLUBS, Rank.ACE).id,
    )
    root_actions = tuple(
        _action(
            model,
            f"ROOT-{index}",
            card,
            Seat.WEST,
            winning=winning,
        )
        for index, (card, winning) in enumerate(
            zip(root_cards, root_wins, strict=True)
        )
    )
    return CpSatPredicateModelContextV1(
        model,
        "flagship-fixed-context",
        {
            5: CpSatTrickVariablesV1(5, trick_six_actions, trick_six_actions[0]),
            6: CpSatTrickVariablesV1(6, trick_seven_actions, trick_seven_actions[0]),
            7: CpSatTrickVariablesV1(7, trick_eight_actions, trick_eight_actions[0]),
        },
        root_actions,
        model.new_constant(int(proof_complete)),
    )


def _solve(model: cp_model.CpModel) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    return solver


def _independent_selector_oracle(
    selector: CardSelectorV1,
    card_id: int,
    actor_id: int,
    owner_id: int,
    ekka: bool,
) -> bool:
    rank = tuple(Rank)[card_id % 8].name.lower()
    suit = tuple(Suit)[card_id // 8].name.lower()
    actor = Seat(actor_id)
    owner = Seat(owner_id)
    clauses = [
        not selector.cards or card_id in selector.cards,
        not selector.ranks or rank in selector.ranks,
        not selector.suits or suit in selector.suits,
        selector.ekka == "any"
        or (selector.ekka == "required") == bool(ekka),
    ]
    if selector.owner_relation == "actor":
        clauses.append(owner_id == actor_id)
    elif selector.owner_relation == "partner":
        clauses.append(owner_id == (actor_id + 2) % 4)
    elif selector.owner_relation == "opponent_team":
        clauses.append(owner.team != actor.team)
    elif selector.owner_relation == "seat":
        clauses.append(owner == Seat.parse(selector.owner_seat))
    elif selector.owner_relation == "team":
        clauses.append(owner.team == Team[selector.owner_team.upper()])
    return all(clauses)


@pytest.mark.parametrize(
    "raw_selector",
    [
        {"cards": [0, 31]},
        {"ranks": ["ace"]},
        {"suits": ["hearts", "clubs"]},
        {"ranks": ["seven", "nine"], "suits": ["spades"]},
        {"ekka": "required"},
        {"ekka": "forbidden"},
        {"owner": {"relation": "actor"}},
        {"owner": {"relation": "partner"}},
        {"owner": {"relation": "opponent_team"}},
        {"owner": {"relation": "seat", "seat": "west"}},
        {"owner": {"relation": "team", "team": "north_south"}},
        {
            "cards": [31],
            "ranks": ["ace"],
            "suits": ["clubs"],
            "ekka": "forbidden",
            "owner": {"relation": "actor"},
        },
    ],
)
def test_selector_semantics_cover_the_complete_finite_card_action_domain(raw_selector):
    selector = CardSelectorV1.build(raw_selector)

    for card_id, actor_id, owner_id, ekka in itertools.product(
        range(32), range(4), range(4), (False, True)
    ):
        assert _selector_matches(
            selector,
            card_id=card_id,
            actor_id=actor_id,
            owner_id=owner_id,
            ekka=ekka,
        ) is _independent_selector_oracle(
            selector, card_id, actor_id, owner_id, ekka
        )


def test_registered_catalog_is_exact_only_with_lowering_and_extractor_pairs():
    hidden = {row.kind: row for row in flagship_predicate_definitions_v2()}
    hidden_catalog = {row.kind: row for row in scenario_predicate_definitions_v2()}
    registry = ScenarioPredicateLoweringRegistryV2()
    registered = {row.kind: row for row in registry.definitions}
    registered_catalog = {row.kind: row for row in registry.catalog_definitions}

    assert set(hidden) == FLAGSHIP_PREDICATE_KINDS_V2
    assert set(registered) == GENERAL_PREDICATE_KINDS_V2
    assert len(hidden_catalog) == len(registered_catalog) == 78
    assert all(row.cp_sat_support.search_status == "unsupported" for row in hidden.values())
    assert all(row.publication_issues() for row in hidden.values())
    assert all(not row.publication_issues() for row in registered.values())
    assert all(row.cp_sat_support.evidence_extractor for row in registered.values())
    assert {
        kind
        for kind, row in registered_catalog.items()
        if row.cp_sat_support.search_status in {"exact", "derived"}
    } == EXACT_V1_COMPATIBILITY_KINDS | GENERAL_PREDICATE_KINDS_V2
    assert len(registry.manifest_hash) == 64


def test_flagship_predicates_lower_exactly_and_extract_engine_free_evidence():
    context = _context()
    registry = ScenarioPredicateLoweringRegistryV2()
    expression = ScenarioExpressionV1.build(
        {
            "op": "all",
            "children": [
                _predicate(
                    "PRED-ACTION",
                    "play.action_matches",
                    {
                        "trickIndex": 5,
                        "position": 3,
                        "actingSeat": "west",
                        "selector": {"ranks": ["ace"], "suits": ["clubs"]},
                    },
                ),
                _predicate(
                    "PRED-WINNER",
                    "play.trick_winner_matches",
                    {
                        "trickIndex": 6,
                        "team": "west_east",
                        "selector": {"ranks": ["seven", "eight", "nine"]},
                    },
                ),
                _predicate(
                    "PRED-TWO-ACES",
                    "play.trick_card_count",
                    {
                        "trickIndex": 7,
                        "selector": {
                            "ranks": ["ace"],
                            "owner": {
                                "relation": "team",
                                "team": "north_south",
                            },
                        },
                        "minimum": 2,
                        "maximum": 2,
                    },
                ),
                _predicate(
                    "PRED-ROOT-ACE",
                    "proof.winning_root_action_matches",
                    {
                        "selector": {
                            "ranks": ["ace"],
                            "owner": {"relation": "actor"},
                        }
                    },
                ),
            ],
        }
    )

    root, lowerings = registry.lower_expression(expression, context)
    context.model.add(root == 1)
    solver = _solve(context.model)
    evidence = {item.kind: item.extract(solver) for item in lowerings}

    assert solver.boolean_value(root)
    assert evidence["play.action_matches"]["observed"]["action"]["card"]["code"] == "CA"
    assert evidence["play.trick_winner_matches"]["observed"]["winner"]["card"]["rank"] == "seven"
    assert evidence["play.trick_card_count"]["observed"]["matchingCount"] == 2
    assert evidence["proof.winning_root_action_matches"]["observed"]["matchingWinningActionIds"] == ["ROOT-2"]
    assert all(len(item["sha256"]) == 64 for item in evidence.values())


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            {
                "op": "any",
                "children": [
                    _predicate(
                        "PRED-FALSE",
                        "play.trick_card_count",
                        {
                            "trickIndex": 7,
                            "selector": {"ranks": ["ace"]},
                            "minimum": 0,
                            "maximum": 0,
                        },
                    ),
                    _predicate(
                        "PRED-TRUE",
                        "play.trick_card_count",
                        {
                            "trickIndex": 7,
                            "selector": {"ranks": ["ace"]},
                            "minimum": 2,
                            "maximum": 2,
                        },
                    ),
                ],
            },
            True,
        ),
        (
            {
                "op": "count",
                "minimum": 1,
                "maximum": 1,
                "children": [
                    _predicate(
                        "PRED-FALSE",
                        "play.trick_card_count",
                        {
                            "trickIndex": 7,
                            "selector": {"ranks": ["ace"]},
                            "minimum": 0,
                            "maximum": 0,
                        },
                    ),
                    _predicate(
                        "PRED-TRUE",
                        "play.trick_card_count",
                        {
                            "trickIndex": 7,
                            "selector": {"ranks": ["ace"]},
                            "minimum": 2,
                            "maximum": 2,
                        },
                    ),
                ],
            },
            True,
        ),
        (
            {
                "op": "not",
                "children": [
                    _predicate(
                        "PRED-FALSE",
                        "play.trick_card_count",
                        {
                            "trickIndex": 7,
                            "selector": {"ranks": ["ace"]},
                            "minimum": 0,
                            "maximum": 0,
                        },
                    )
                ],
            },
            True,
        ),
    ],
)
def test_any_count_and_safe_not_are_exactly_reified(raw, expected):
    context = _context()
    root, _ = ScenarioPredicateLoweringRegistryV2().lower_expression(
        ScenarioExpressionV1.build(raw), context
    )
    solver = _solve(context.model)

    assert solver.boolean_value(root) is expected


def test_unsafe_not_and_missing_lowering_fail_before_model_mutation():
    registry = ScenarioPredicateLoweringRegistryV2()
    context = _context()
    unsafe = ScenarioExpressionV1.build(
        {
            "op": "not",
            "children": [
                _predicate(
                    "PRED-UNSAFE",
                    "play.action_matches",
                    {
                        "trickIndex": 5,
                        "position": 3,
                        "selector": {"ranks": ["ace"]},
                    },
                )
            ],
        }
    )
    constraints_before = len(context.model.proto.constraints)

    with pytest.raises(ValueError, match="safe complement"):
        registry.lower_expression(unsafe, context)
    assert len(context.model.proto.constraints) == constraints_before

    missing = ScenarioExpressionV1.build(
        _predicate(
            "PRED-MISSING",
            "outcome.team_round",
            {"team": "west_east", "status": "won"},
        )
    )
    with pytest.raises(ValueError, match="no exact CP-SAT lowering"):
        registry.lower_expression(missing, context)
    assert len(context.model.proto.constraints) == constraints_before


@pytest.mark.parametrize(
    "proof_complete,root_wins,kind,arguments",
    [
        (
            True,
            (False, False, False),
            "proof.winning_root_action_matches",
            {"selector": {"ranks": ["ace"]}},
        ),
        (
            False,
            (False, False, True),
            "proof.winning_root_action_matches",
            {"selector": {"ranks": ["ace"]}},
        ),
    ],
)
def test_root_proof_predicates_are_non_vacuous_and_fail_closed_when_incomplete(
    proof_complete,
    root_wins,
    kind,
    arguments,
):
    context = _context(proof_complete=proof_complete, root_wins=root_wins)
    lowering = ScenarioPredicateLoweringRegistryV2().lower_predicate(
        ScenarioPredicateV1.build(_predicate("PRED-PROOF", kind, arguments)),
        context,
    )
    solver = _solve(context.model)

    assert not solver.boolean_value(lowering.literal)
