from __future__ import annotations

from ortools.sat.python import cp_model
import pytest

from baloot_platform.adjudication.research import AuthoredAdjudicationEventV1
from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat
from baloot_platform.research.scenario_adjudication_predicates_v3 import (
    ADJUDICATION_AUTHORITY_ID_V3,
    ADJUDICATION_AUTHORITY_SHA256_V3,
    AdjudicationPredicateModelContextV3,
    FALSE_EKKA_PREDICATE_V3,
    OUT_OF_TURN_PREDICATE_V3,
    PURCHASE_BEFORE_TURN_PREDICATE_V3,
    ScenarioPredicateCatalogLoweringManifestV3,
)
from baloot_platform.research.scenario_predicate_manifest_v2 import (
    ScenarioPredicateCatalogLoweringManifestV2,
)


FROZEN_V2_MANIFEST_HASH = (
    "d36815fde96af7246bfb67153f75dd0cc7962fc5b18827a3e6582b764a9a600d"
)


def _predicate(kind: str, arguments: dict, predicate_id: str = "PRED-ADJ") -> dict:
    return {
        "predicateId": predicate_id,
        "kind": kind,
        "version": 1,
        "hardness": "hard",
        "arguments": arguments,
    }


def _solve(
    kind: str,
    arguments: dict,
    expected: Seat,
    *,
    events=(),
    action_available: bool = True,
):
    manifest = ScenarioPredicateCatalogLoweringManifestV3(
        include_false_ekka=kind == FALSE_EKKA_PREDICATE_V3
    )
    predicate = manifest.parse_predicate(_predicate(kind, arguments))
    model = cp_model.CpModel()
    expected_actor = model.new_int_var(int(expected), int(expected), "expected_actor")
    attempted_actor = model.new_int_var(0, 3, "domain_attempted_actor")
    maximum_action = (
        31
        if kind in {OUT_OF_TURN_PREDICATE_V3, FALSE_EKKA_PREDICATE_V3}
        else 3
    )
    attempted_action = model.new_int_var(0, maximum_action, "domain_attempted_action")
    context = AdjudicationPredicateModelContextV3(
        model,
        "test-context",
        "exact_trace_prefix" if events else "search",
        expected_actor,
        attempted_actor,
        attempted_action,
        model.new_constant(int(action_available)),
        tuple(events),
    )
    lowering = manifest.lower_predicate(predicate, context)
    solver = cp_model.CpSolver()
    assert solver.solve(model) in {cp_model.OPTIMAL, cp_model.FEASIBLE}
    return solver.boolean_value(lowering.literal), lowering.extract(solver)


def test_v3_merged_registry_preserves_frozen_v2_and_exposes_only_two_native_events():
    base = ScenarioPredicateCatalogLoweringManifestV2()
    before = base.catalog()
    merged = ScenarioPredicateCatalogLoweringManifestV3(base)

    assert base.manifest_hash == FROZEN_V2_MANIFEST_HASH
    assert base.catalog() == before
    assert merged.definition(OUT_OF_TURN_PREDICATE_V3) is not None
    assert merged.definition(PURCHASE_BEFORE_TURN_PREDICATE_V3) is not None
    assert merged.definition("adjudication.incorrect_highest_card_naming") is None
    assert merged.definition("adjudication.exposed_card") is None
    assert len(merged.exposed_definitions) == len(base.exposed_definitions) + 2
    assert merged.capability_manifest.capability(OUT_OF_TURN_PREDICATE_V3).decision_kinds == ("play",)
    assert merged.capability_manifest.capability(
        PURCHASE_BEFORE_TURN_PREDICATE_V3
    ).decision_kinds == ("bid", "play")


def test_false_ekka_predicate_is_visible_only_for_unratified_variant():
    official = ScenarioPredicateCatalogLoweringManifestV3()
    variant = ScenarioPredicateCatalogLoweringManifestV3(include_false_ekka=True)
    assert official.definition(FALSE_EKKA_PREDICATE_V3) is None
    assert variant.definition(FALSE_EKKA_PREDICATE_V3) is not None
    for external in (
        "adjudication.incorrect_highest_card_naming",
        "adjudication.exposed_card",
        "adjudication.signalling",
    ):
        assert variant.definition(external) is None


def test_v3_catalog_and_manifest_share_one_canonical_selected_authority_hash():
    first = ScenarioPredicateCatalogLoweringManifestV3(
        adjudication_authority_id="authority-v2",
        adjudication_authority_sha256="2" * 64,
    )
    repeated = ScenarioPredicateCatalogLoweringManifestV3(
        adjudication_authority_id="authority-v2",
        adjudication_authority_sha256="2" * 64,
    )
    historical = ScenarioPredicateCatalogLoweringManifestV3(
        adjudication_authority_id=ADJUDICATION_AUTHORITY_ID_V3,
        adjudication_authority_sha256=ADJUDICATION_AUTHORITY_SHA256_V3,
    )

    assert first.catalog()["sha256"] == first.manifest_hash
    assert repeated.catalog() == first.catalog()
    assert repeated.manifest_hash == first.manifest_hash
    assert historical.manifest_hash != first.manifest_hash
    assert first.capability_manifest.extensions[0].authority_id == "authority-v2"
    assert first.capability_manifest.extensions[0].authority_sha256 == "2" * 64
    assert ScenarioPredicateCatalogLoweringManifestV2().manifest_hash == FROZEN_V2_MANIFEST_HASH


@pytest.mark.parametrize(
    ("kind", "arguments", "expected", "satisfied"),
    (
        (OUT_OF_TURN_PREDICATE_V3, {"attemptedActor": "west", "attemptedCardId": 12}, Seat.SOUTH, True),
        (OUT_OF_TURN_PREDICATE_V3, {"attemptedActor": "south", "attemptedCardId": 12}, Seat.SOUTH, False),
        (PURCHASE_BEFORE_TURN_PREDICATE_V3, {"attemptedActor": "north", "purchaseToken": "sun"}, Seat.EAST, True),
        (PURCHASE_BEFORE_TURN_PREDICATE_V3, {"attemptedActor": "east", "purchaseToken": "sun"}, Seat.EAST, False),
    ),
)
def test_v3_generated_search_lowers_attempted_action_truth_natively(kind, arguments, expected, satisfied):
    observed, evidence = _solve(kind, arguments, expected)
    assert observed is satisfied
    assert evidence["satisfied"] is satisfied
    assert evidence["observed"]["authorityHash"] == ADJUDICATION_AUTHORITY_SHA256_V3
    assert evidence["observed"]["ordinaryActionHistoryMutated"] is False
    assert evidence["observed"]["requiresTableReferee"] is True


def test_v3_generated_attempt_requires_domain_available_action():
    observed, evidence = _solve(
        OUT_OF_TURN_PREDICATE_V3,
        {"attemptedActor": "west", "attemptedCardId": 12},
        Seat.SOUTH,
        action_available=False,
    )
    assert observed is False
    assert evidence["observed"]["attemptedActionAvailable"] is False


@pytest.mark.parametrize(
    ("kind", "arguments", "expected", "action_available"),
    (
        (OUT_OF_TURN_PREDICATE_V3, {"attemptedActor": "south", "attemptedCardId": 12}, Seat.SOUTH, True),
        (PURCHASE_BEFORE_TURN_PREDICATE_V3, {"attemptedActor": "east", "purchaseToken": "sun"}, Seat.EAST, True),
        (FALSE_EKKA_PREDICATE_V3, {"attemptedActor": "west", "attemptedCardId": 12}, Seat.SOUTH, True),
        (OUT_OF_TURN_PREDICATE_V3, {"attemptedActor": "west", "attemptedCardId": 12}, Seat.SOUTH, False),
    ),
)
def test_requested_remedy_does_not_force_an_inactive_predicate(
    kind, arguments, expected, action_available,
):
    satisfied, evidence = _solve(
        kind, {**arguments, "requiredRemedy": "continue_play"}, expected,
        action_available=action_available,
    )
    assert satisfied is False
    assert "selectedRemedy" not in evidence["observed"]


def test_false_ekka_is_current_actor_illegal_declaration_evidence():
    observed, evidence = _solve(
        FALSE_EKKA_PREDICATE_V3,
        {"attemptedActor": "south", "attemptedCardId": 12},
        Seat.SOUTH,
    )
    assert observed is True
    assert evidence["observed"]["kind"] == "false_ekka_attempt"
    assert evidence["observed"]["ordinaryActionHistoryMutated"] is False

    wrong_actor, _ = _solve(
        FALSE_EKKA_PREDICATE_V3,
        {"attemptedActor": "west", "attemptedCardId": 12},
        Seat.SOUTH,
    )
    assert wrong_actor is False


def test_v3_exact_input_requires_matching_typed_sidecar():
    event = AuthoredAdjudicationEventV1.build(
        {
            "schemaVersion": 1,
            "kind": "out_of_turn",
            "actionSequence": 10,
            "actorSeat": "west",
            "expectedActorSeat": "south",
            "attemptedAction": {"card": Card(12).code},
            "reasonCode": "play_before_turn",
        }
    )
    arguments = {"attemptedActor": "west", "attemptedCardId": 12}
    satisfied, evidence = _solve(OUT_OF_TURN_PREDICATE_V3, arguments, Seat.SOUTH, events=(event,))
    assert satisfied is True
    assert evidence["observed"]["exactSidecarSha256"] == event.sha256

    missing, _ = _solve(OUT_OF_TURN_PREDICATE_V3, arguments, Seat.SOUTH, events=(
        AuthoredAdjudicationEventV1.build({
            "schemaVersion": 1,
            "kind": "out_of_turn",
            "actionSequence": 10,
            "actorSeat": "west",
            "expectedActorSeat": "south",
            "attemptedAction": {"cardId": 13},
            "reasonCode": "play_before_turn",
        }),
    ))
    assert missing is False


def test_requested_remedy_without_matching_exact_sidecar_is_not_reported_as_selected():
    event = AuthoredAdjudicationEventV1.build({
        "schemaVersion": 1, "kind": "out_of_turn", "actionSequence": 10,
        "actorSeat": "west", "expectedActorSeat": "south",
        "attemptedAction": {"cardId": 12}, "reasonCode": "play_before_turn",
    })
    satisfied, evidence = _solve(
        OUT_OF_TURN_PREDICATE_V3,
        {"attemptedActor": "west", "attemptedCardId": 12, "requiredRemedy": "continue_play"},
        Seat.SOUTH, events=(event,),
    )
    assert not satisfied
    assert evidence["observed"]["eventPresent"] is True
    assert evidence["observed"]["exactSidecarSha256"] is None
    assert "selectedRemedy" not in evidence["observed"]


def test_v3_purchase_exact_input_uses_invalid_bid_purchase_sidecar_only():
    event = AuthoredAdjudicationEventV1.build(
        {
            "schemaVersion": 1,
            "kind": "invalid_bid",
            "actionSequence": 2,
            "actorSeat": "north",
            "expectedActorSeat": "east",
            "attemptedAction": {"token": "sun"},
            "reasonCode": "purchase_before_turn",
        }
    )
    arguments = {"attemptedActor": "north", "purchaseToken": "sun"}
    satisfied, evidence = _solve(
        PURCHASE_BEFORE_TURN_PREDICATE_V3,
        arguments,
        Seat.EAST,
        events=(event,),
    )
    assert satisfied is True
    assert evidence["observed"]["kind"] == "purchase_before_turn"
    assert evidence["observed"]["exactSidecarSha256"] == event.sha256

    wrong_expected, _ = _solve(
        PURCHASE_BEFORE_TURN_PREDICATE_V3,
        arguments,
        Seat.SOUTH,
        events=(event,),
    )
    assert wrong_expected is False


@pytest.mark.parametrize("origin", ("search", "exact_trace_prefix"))
def test_v3_official_hard_predicates_have_grouped_positive_and_negative_evidence(
    origin,
):
    manifest = ScenarioPredicateCatalogLoweringManifestV3()
    definitions = tuple(
        definition
        for definition in manifest.exposed_definitions
        if definition.kind.startswith("adjudication.")
    )
    assert {definition.kind for definition in definitions} == {
        OUT_OF_TURN_PREDICATE_V3,
        PURCHASE_BEFORE_TURN_PREDICATE_V3,
    }
    model = cp_model.CpModel()
    lowerings = []
    for index, definition in enumerate(definitions):
        if origin == "search":
            assert definition.cp_sat_support.search_status == "exact"
        else:
            assert definition.cp_sat_support.exact_input_status == "exact"
        if definition.kind == OUT_OF_TURN_PREDICATE_V3:
            action_arguments = {"attemptedCardId": 12}
            positive_actor = Seat.WEST
            expected_actor = Seat.SOUTH
            event = AuthoredAdjudicationEventV1.build(
                {
                    "schemaVersion": 1,
                    "kind": "out_of_turn",
                    "actionSequence": 10,
                    "actorSeat": "west",
                    "expectedActorSeat": "south",
                    "attemptedAction": {"card": Card(12).code},
                    "reasonCode": "play_before_turn",
                }
            )
        else:
            action_arguments = {"purchaseToken": "sun"}
            positive_actor = Seat.NORTH
            expected_actor = Seat.EAST
            event = AuthoredAdjudicationEventV1.build(
                {
                    "schemaVersion": 1,
                    "kind": "invalid_bid",
                    "actionSequence": 2,
                    "actorSeat": "north",
                    "expectedActorSeat": "east",
                    "attemptedAction": {"token": "sun"},
                    "reasonCode": "purchase_before_turn",
                }
            )
        for expected_satisfied, attempted_actor in (
            (True, positive_actor),
            (False, expected_actor),
        ):
            predicate = manifest.parse_predicate(
                _predicate(
                    definition.kind,
                    {
                        "attemptedActor": attempted_actor.name.lower(),
                        **action_arguments,
                    },
                    predicate_id=(
                        f"PRED-ADJ-MATRIX-{index}-"
                        f"{'POSITIVE' if expected_satisfied else 'NEGATIVE'}"
                    ),
                )
            )
            prefix = f"adjudication_matrix_{index}_{int(expected_satisfied)}"
            context = AdjudicationPredicateModelContextV3(
                model,
                prefix,
                origin,
                model.new_constant(int(expected_actor)),
                model.new_int_var(0, 3, f"{prefix}_attempted_actor"),
                model.new_int_var(0, 31, f"{prefix}_attempted_action"),
                model.new_constant(1),
                ((event,) if origin == "exact_trace_prefix" else ()),
            )
            lowerings.append(
                (
                    definition,
                    expected_satisfied,
                    manifest.lower_predicate(predicate, context),
                )
            )

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    for definition, expected_satisfied, lowering in lowerings:
        evidence = lowering.extract(solver)
        assert solver.boolean_value(lowering.literal) is expected_satisfied
        assert evidence["satisfied"] is expected_satisfied
        assert lowering.evidence_extractor_id == (
            definition.cp_sat_support.evidence_extractor
        )


@pytest.mark.parametrize(
    "value",
    (
        _predicate(OUT_OF_TURN_PREDICATE_V3, {"attemptedActor": "west", "attemptedCardId": 32}),
        _predicate(PURCHASE_BEFORE_TURN_PREDICATE_V3, {"attemptedActor": "west", "purchaseToken": "ashkal"}),
        _predicate("adjudication.exposed_card", {"attemptedActor": "west"}),
    ),
)
def test_v3_parser_rejects_invalid_or_report_only_adjudication_predicates(value):
    with pytest.raises(ValueError):
        ScenarioPredicateCatalogLoweringManifestV3().parse_predicate(value)
