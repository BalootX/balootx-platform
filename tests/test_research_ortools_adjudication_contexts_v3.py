from __future__ import annotations

from dataclasses import dataclass, replace

from ortools.sat.python import cp_model
import pytest

from baloot_platform.adjudication.research import AuthoredAdjudicationEventV1
from baloot_platform.core.enums import Seat
from baloot_platform.research.ortools_adjudication_contexts_v3 import (
    wrap_auction_adjudication_context_v3,
    wrap_play_adjudication_context_v3,
)
from baloot_platform.research.ortools_scenario_model import OrToolsScenarioModelV1
from baloot_platform.research.ortools_play_state_predicates import (
    CpSatPlayPrefixActionV1,
)
from baloot_platform.research.scenario_adjudication_predicates_v3 import (
    FALSE_EKKA_PREDICATE_V3,
    OUT_OF_TURN_PREDICATE_V3,
    PURCHASE_BEFORE_TURN_PREDICATE_V3,
    shared_scenario_predicate_catalog_lowering_manifest_v3,
)
from baloot_platform.research.scenario_contracts import ScenarioPredicateV1
from baloot_platform.research.scenario_predicate_auction_escalation_v2 import (
    CpSatAuctionPredicateContextV1,
)


def _predicate(kind: str, arguments: dict, predicate_id: str) -> ScenarioPredicateV1:
    return ScenarioPredicateV1.build(
        {
            "predicateId": predicate_id,
            "kind": kind,
            "version": 1,
            "hardness": "hard",
            "arguments": arguments,
        }
    )


def _solve(model: cp_model.CpModel) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    return solver


def test_purchase_attempt_uses_selected_current_actor_and_live_token_table(rules):
    scenario = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    fragment = scenario.add_auction_fragment(prefix_length=0)
    base = CpSatAuctionPredicateContextV1(
        scenario.model, fragment, Seat.EAST, view="prefix"
    )
    predicate = _predicate(
        PURCHASE_BEFORE_TURN_PREDICATE_V3,
        {"attemptedActor": "west", "purchaseToken": "sun"},
        "PRED-ADJ-AUCTION",
    )
    wrapped = wrap_auction_adjudication_context_v3(
        base, (predicate,), origin="search"
    )
    lowering = shared_scenario_predicate_catalog_lowering_manifest_v3().lower_predicate(
        predicate, wrapped
    )
    scenario.model.add(lowering.literal == 1)
    solver = _solve(scenario.model)
    evidence = lowering.extract(solver)["observed"]
    assert evidence["expectedActor"] == "north"
    assert evidence["attemptedActor"] == "west"
    assert evidence["attemptedAction"] == {
        "type": "auction_token",
        "token": "sun",
    }
    assert evidence["attemptedActionAvailable"] is True
    assert evidence["ordinaryActionHistoryMutated"] is False


@pytest.mark.parametrize("origin,sequence,expected_actor,token,satisfied", (
    ("search", None, "north", "sun", True),
    ("exact_trace_prefix", 0, "north", "sun", True),
    ("exact_trace_prefix", 0, "west", "sun", False),
    ("exact_trace_prefix", 1, "north", "sun", False),
    ("exact_trace_prefix", 2, "north", "sun", False),
    ("exact_trace_prefix", None, "north", "sun", False),
    ("exact_trace_prefix", 0, "north", "no", False),
    ("exact_trace_prefix", 0, "north", "hokum", True),
    ("exact_trace_prefix", 4, "north", "hokum", True),
))
def test_completed_native_auction_keeps_purchase_attempt_at_its_boundary(
    rules, origin, sequence, expected_actor, token, satisfied,
):
    scenario = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    actions = tuple(
        {"actor": seat.name.lower(), "action": "pass"}
        for seat in (Seat.NORTH, Seat.WEST, Seat.SOUTH, Seat.EAST)
    ) if sequence == 4 else ()
    actions += ({"actor": "north", "action": "sun"},)
    fragment = scenario.add_native_auction_fragment(
        required_actions=actions,
        required_actions_complete=True,
        required_purchase_card="S7",
    )
    base = CpSatAuctionPredicateContextV1(
        scenario.model, fragment, Seat.EAST, view="completion",
    )
    predicate = _predicate(
        PURCHASE_BEFORE_TURN_PREDICATE_V3,
        {"attemptedActor": "west", "purchaseToken": token},
        "PRED-HISTORICAL-PURCHASE",
    )
    events = () if origin == "search" or sequence is None else (
        AuthoredAdjudicationEventV1.build({
            "schemaVersion": 1,
            "kind": "invalid_bid",
            "actionSequence": sequence,
            "actorSeat": "west",
            "expectedActorSeat": expected_actor,
            "attemptedAction": {"token": token},
            "reasonCode": "purchase_before_turn",
        }),
    )
    wrapped = wrap_auction_adjudication_context_v3(
        base, (predicate,), origin=origin, exact_events=events,
    )
    lowering = shared_scenario_predicate_catalog_lowering_manifest_v3().lower_predicate(
        predicate, wrapped,
    )
    solver = _solve(scenario.model)
    assert solver.boolean_value(lowering.literal) is satisfied
    observed = lowering.extract(solver)["observed"]
    assert observed["expectedActor"] == "north"
    assert observed["actionSequence"] == (4 if sequence == 4 else 0)
    assert len(fragment.extract_actions(solver)) == len(actions)


@pytest.mark.parametrize("complete_prefix", (False, True))
def test_generated_historical_purchase_uses_first_actual_witness_and_safe_complement(rules, complete_prefix):
    scenario = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    fragment = scenario.add_native_auction_fragment(
        required_actions=(
            {"actor": "north", "action": "pass"},
            {"actor": "west", "action": "sun"},
            {"actor": "north", "action": "pass"},
        ) if complete_prefix else (),
        required_actions_complete=complete_prefix,
        required_purchase_card="S7",
    )
    if not complete_prefix:
        scenario.model.add(fragment.choices[0]["pass"] == 1)
        scenario.model.add(fragment.choices[1]["sun"] == 1)
        scenario.model.add(fragment.choices[2]["pass"] == 1)
    predicate = _predicate(PURCHASE_BEFORE_TURN_PREDICATE_V3,
        {"attemptedActor": "north", "purchaseToken": "sun"}, "PRED-LATER-PURCHASE")
    wrapped = wrap_auction_adjudication_context_v3(
        CpSatAuctionPredicateContextV1(scenario.model, fragment, Seat.EAST, view="completion"),
        (predicate,), origin="search",
    )
    lowering = shared_scenario_predicate_catalog_lowering_manifest_v3().lower_predicate(predicate, wrapped)
    solver = _solve(scenario.model)
    assert solver.boolean_value(lowering.literal) is True
    observed = lowering.extract(solver)["observed"]
    assert observed["actionSequence"] == 1
    assert observed["expectedActor"] == "west"
    context = wrapped.adjudication_contexts[predicate.predicate_id]
    foreign = cp_model.CpModel().new_int_var(0, 7, "foreign_boundary")
    negative = scenario.model.new_int_var(-1, -1, "negative_boundary")
    for value in (foreign, negative):
        with pytest.raises(ValueError, match="adjudication search boundary"):
            replace(context, action_sequence=value)
    with pytest.raises(ValueError, match="adjudication search boundary"):
        replace(context, origin="exact_trace_prefix")
    scenario.model.add(lowering.literal == 0)
    assert solver.solve(scenario.model) == cp_model.INFEASIBLE


def test_purchase_attempt_rejects_same_actor_or_unavailable_round_token(rules):
    for actor, token in (("north", "sun"), ("west", "no")):
        scenario = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
        fragment = scenario.add_auction_fragment(prefix_length=0)
        base = CpSatAuctionPredicateContextV1(
            scenario.model, fragment, Seat.EAST, view="prefix"
        )
        predicate = _predicate(
            PURCHASE_BEFORE_TURN_PREDICATE_V3,
            {"attemptedActor": actor, "purchaseToken": token},
            f"PRED-ADJ-{actor.upper()}-{token.upper()}",
        )
        wrapped = wrap_auction_adjudication_context_v3(
            base, (predicate,), origin="search"
        )
        lowering = (
            shared_scenario_predicate_catalog_lowering_manifest_v3().lower_predicate(
                predicate, wrapped
            )
        )
        scenario.model.add(lowering.literal == 1)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        assert solver.solve(scenario.model) == cp_model.INFEASIBLE


@dataclass(frozen=True)
class _PlayContext:
    model: cp_model.CpModel
    context_id: str
    play_slots: tuple[CpSatPlayPrefixActionV1, ...]
    next_actor: cp_model.IntVar
    legal_card_literals: dict[int, cp_model.IntVar]
    legal_action_literals: dict[tuple[int, bool], cp_model.IntVar]


def _play_context() -> tuple[_PlayContext, tuple[cp_model.IntVar, ...]]:
    model = cp_model.CpModel()
    owners = tuple(model.new_constant(card // 8) for card in range(32))
    absent = CpSatPlayPrefixActionV1(
        model.new_constant(0),
        model.new_constant(0),
        model.new_constant(0),
        model.new_constant(0),
    )
    slots = [absent for _ in range(32)]
    slots[0] = CpSatPlayPrefixActionV1(
        model.new_constant(0),
        model.new_constant(int(Seat.NORTH)),
        model.new_constant(0),
        model.new_constant(1),
    )
    legal_cards = {
        card: model.new_constant(int(card in {1, 2})) for card in range(32)
    }
    legal_actions = {
        (card, ekka): model.new_constant(
            int(card in {1, 2} and (not ekka or card == 2))
        )
        for card in range(32)
        for ekka in (False, True)
    }
    return (
        _PlayContext(
            model,
            "test-play",
            tuple(slots),
            model.new_constant(int(Seat.SOUTH)),
            legal_cards,
            legal_actions,
        ),
        owners,
    )


def test_out_of_turn_attempt_requires_owned_unplayed_card_and_other_actor():
    base, owners = _play_context()
    predicate = _predicate(
        OUT_OF_TURN_PREDICATE_V3,
        {"attemptedActor": "north", "attemptedCardId": 1},
        "PRED-ADJ-PLAY",
    )
    wrapped = wrap_play_adjudication_context_v3(
        base, owners, (predicate,), origin="search"
    )
    lowering = shared_scenario_predicate_catalog_lowering_manifest_v3().lower_predicate(
        predicate, wrapped
    )
    base.model.add(lowering.literal == 1)
    evidence = lowering.extract(_solve(base.model))["observed"]
    assert evidence["attemptedActionAvailable"] is True
    assert evidence["expectedActor"] == "south"
    assert evidence["ordinaryActionHistoryMutated"] is False


def test_requested_remedy_on_false_child_preserves_a_valid_disjunction():
    base, owners = _play_context()
    inactive = _predicate(
        OUT_OF_TURN_PREDICATE_V3,
        {"attemptedActor": "south", "attemptedCardId": 16, "requiredRemedy": "continue_play"},
        "PRED-INACTIVE",
    )
    active = _predicate(
        OUT_OF_TURN_PREDICATE_V3,
        {"attemptedActor": "north", "attemptedCardId": 1, "requiredRemedy": "continue_play"},
        "PRED-ACTIVE",
    )
    wrapped = wrap_play_adjudication_context_v3(
        base, owners, (inactive, active), origin="search"
    )
    manifest = shared_scenario_predicate_catalog_lowering_manifest_v3()
    first, second = (manifest.lower_predicate(predicate, wrapped) for predicate in (inactive, active))
    base.model.add_bool_or((first.literal, second.literal))
    solver = _solve(base.model)
    assert not solver.boolean_value(first.literal)
    assert solver.boolean_value(second.literal)
    assert "selectedRemedy" not in first.extract(solver)["observed"]
    assert second.extract(solver)["observed"]["selectedRemedy"] == "continue_play"


def test_out_of_turn_attempt_rejects_played_unowned_and_same_actor_cards():
    cases = (
        ("north", 0),
        ("west", 1),
        ("south", 16),
    )
    for actor, card in cases:
        base, owners = _play_context()
        predicate = _predicate(
            OUT_OF_TURN_PREDICATE_V3,
            {"attemptedActor": actor, "attemptedCardId": card},
            f"PRED-ADJ-{actor.upper()}-{card}",
        )
        wrapped = wrap_play_adjudication_context_v3(
            base, owners, (predicate,), origin="search"
        )
        lowering = (
            shared_scenario_predicate_catalog_lowering_manifest_v3().lower_predicate(
                predicate, wrapped
            )
        )
        base.model.add(lowering.literal == 1)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        assert solver.solve(base.model) == cp_model.INFEASIBLE


def test_false_ekka_requires_current_actor_legal_card_and_illegal_ekka_action():
    for actor, card, expected in (
        ("south", 1, cp_model.OPTIMAL),
        ("west", 1, cp_model.INFEASIBLE),
        ("south", 2, cp_model.INFEASIBLE),
        ("south", 3, cp_model.INFEASIBLE),
    ):
        base, owners = _play_context()
        predicate = _predicate(
            FALSE_EKKA_PREDICATE_V3,
            {"attemptedActor": actor, "attemptedCardId": card},
            f"PRED-FALSE-EKKA-{actor.upper()}-{card}",
        )
        wrapped = wrap_play_adjudication_context_v3(
            base, owners, (predicate,), origin="search"
        )
        lowering = shared_scenario_predicate_catalog_lowering_manifest_v3(
            include_false_ekka=True
        ).lower_predicate(predicate, wrapped)
        base.model.add(lowering.literal == 1)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        assert solver.solve(base.model) == expected
        if expected == cp_model.OPTIMAL:
            evidence = lowering.extract(solver)["observed"]
            assert evidence["kind"] == "false_ekka_attempt"
            assert evidence["attemptedActionAvailable"] is True
            assert evidence["ordinaryActionHistoryMutated"] is False


def test_exact_attempt_sidecar_must_match_current_stage_boundary():
    event = AuthoredAdjudicationEventV1.build(
        {
            "schemaVersion": 1,
            "kind": "out_of_turn",
            "actionSequence": 4,
            "actorSeat": "north",
            "expectedActorSeat": "south",
            "attemptedAction": {"cardId": 1},
            "reasonCode": "play_before_turn",
        }
    )
    for boundary, expected in ((4, cp_model.OPTIMAL), (5, cp_model.INFEASIBLE)):
        base, owners = _play_context()
        predicate = _predicate(
            OUT_OF_TURN_PREDICATE_V3,
            {"attemptedActor": "north", "attemptedCardId": 1},
            f"PRED-ADJ-EXACT-{boundary}",
        )
        wrapped = wrap_play_adjudication_context_v3(
            base,
            owners,
            (predicate,),
            origin="exact_trace_prefix",
            exact_events=(event,),
            action_sequence=boundary,
        )
        lowering = (
            shared_scenario_predicate_catalog_lowering_manifest_v3().lower_predicate(
                predicate, wrapped
            )
        )
        base.model.add(lowering.literal == 1)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        assert solver.solve(base.model) == expected
