from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Seat
from baloot_platform.research.ortools_scenario_model import OrToolsScenarioModelV1
from baloot_platform.research.scenario_contracts import ScenarioPredicateV1, canonical_sha256
from baloot_platform.research.scenario_adjudication_predicates_v3 import (
    ScenarioPredicateCatalogLoweringManifestV3,
    ScenarioPredicateStageContextV3,
)
from baloot_platform.research.scenario_predicate_manifest_v2 import (
    ScenarioPredicateCatalogLoweringManifestV2,
)
from baloot_platform.research.scenario_deal_predicates_v2 import (
    DEAL_PREDICATE_REGISTRATIONS_V2,
    DealPredicateLoweringRegistryV2,
    DealPredicateModelContextV2,
)


def _predicate(kind: str, arguments: dict, predicate_id: str = "PRED-DEAL"):
    return ScenarioPredicateV1.build(
        {
            "predicateId": predicate_id,
            "kind": kind,
            "version": 1,
            "hardness": "hard",
            "arguments": arguments,
        }
    )


def _context(rules, *, bind_remaining: bool = True):
    scenario = OrToolsScenarioModelV1(
        rules=rules,
        dealer=Seat.NORTH,
        required_buyer=Seat.SOUTH,
    )
    for position, card in enumerate(scenario.card_at_position):
        scenario.model.add(card == position)
    remaining = None
    if bind_remaining:
        layout = DealLayoutV1.build(
            rules.structural_domain,
            dealer=Seat.NORTH,
            buyer=Seat.SOUTH,
        )
        owners = [4] * 32
        for seat in Seat:
            for card in layout.positions_for(seat, final=True):
                owners[card] = int(seat)
        owners[20] = 4
        owners[24] = 4
        remaining = tuple(scenario.model.new_constant(owner) for owner in owners)
    return scenario, DealPredicateModelContextV2(
        scenario,
        remaining,
        "identity-deal",
    )


def _solve(model: cp_model.CpModel) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    return solver


TRUE_DEAL_PREDICATES = (
    (
        "deal.card_owner",
        {"card": 9, "seat": "north", "handStage": "initial_five"},
    ),
    (
        "deal.card_allowed",
        {
            "card": 20,
            "seats": ["south", "east"],
            "handStage": "post_purchase",
        },
    ),
    (
        "deal.card_forbidden",
        {"card": 20, "seat": "west", "handStage": "post_purchase"},
    ),
    (
        "deal.suit_count",
        {
            "seat": "north",
            "suit": "hearts",
            "minimum": 3,
            "maximum": 3,
            "handStage": "initial_five",
        },
    ),
    (
        "deal.rank_count",
        {
            "seat": "north",
            "rank": "nine",
            "minimum": 2,
            "maximum": 2,
            "handStage": "post_purchase",
        },
    ),
    (
        "deal.high_card_count",
        {
            "seat": "north",
            "minimum": 4,
            "maximum": 4,
            "handStage": "post_purchase",
        },
    ),
    (
        "deal.low_card_count",
        {
            "seat": "north",
            "minimum": 3,
            "maximum": 3,
            "handStage": "initial_five",
        },
    ),
    ("deal.purchase_card", {"card": 20}),
    ("deal.purchase_suit", {"suit": "diamonds"}),
    ("deal.remaining_card_owner", {"card": 25, "seat": "south"}),
    (
        "deal.remaining_suit_count",
        {"seat": "south", "suit": "clubs", "minimum": 1, "maximum": 1},
    ),
    (
        "deal.remaining_rank_count",
        {"seat": "south", "rank": "eight", "minimum": 1, "maximum": 1},
    ),
    (
        "deal.team_high_card_balance",
        {
            "team": "north_south",
            "maximumDifference": 1,
            "handStage": "post_purchase",
        },
    ),
    (
        "deal.seat_relation",
        {"seat": "south", "otherSeat": "north", "relation": "partner"},
    ),
)

FALSE_DEAL_PREDICATES = (
    (
        "deal.card_owner",
        {"card": 9, "seat": "west", "handStage": "initial_five"},
    ),
    (
        "deal.card_allowed",
        {
            "card": 20,
            "seats": ["west", "east"],
            "handStage": "post_purchase",
        },
    ),
    (
        "deal.card_forbidden",
        {"card": 20, "seat": "south", "handStage": "post_purchase"},
    ),
    (
        "deal.suit_count",
        {
            "seat": "north",
            "suit": "hearts",
            "minimum": 2,
            "maximum": 2,
            "handStage": "initial_five",
        },
    ),
    (
        "deal.rank_count",
        {
            "seat": "north",
            "rank": "nine",
            "minimum": 1,
            "maximum": 1,
            "handStage": "post_purchase",
        },
    ),
    (
        "deal.high_card_count",
        {
            "seat": "north",
            "minimum": 3,
            "maximum": 3,
            "handStage": "post_purchase",
        },
    ),
    (
        "deal.low_card_count",
        {
            "seat": "north",
            "minimum": 2,
            "maximum": 2,
            "handStage": "initial_five",
        },
    ),
    ("deal.purchase_card", {"card": 19}),
    ("deal.purchase_suit", {"suit": "hearts"}),
    ("deal.remaining_card_owner", {"card": 25, "seat": "north"}),
    (
        "deal.remaining_suit_count",
        {"seat": "south", "suit": "clubs", "minimum": 0, "maximum": 0},
    ),
    (
        "deal.remaining_rank_count",
        {"seat": "south", "rank": "eight", "minimum": 0, "maximum": 0},
    ),
    (
        "deal.team_high_card_balance",
        {
            "team": "north_south",
            "maximumDifference": 0,
            "handStage": "post_purchase",
        },
    ),
    (
        "deal.seat_relation",
        {"seat": "south", "otherSeat": "north", "relation": "next"},
    ),
)


@pytest.mark.parametrize(("kind", "arguments"), TRUE_DEAL_PREDICATES)
def test_all_fourteen_deal_predicates_lower_exactly_and_extract_evidence(
    rules,
    kind,
    arguments,
):
    scenario, context = _context(rules)
    lowering = DealPredicateLoweringRegistryV2().lower_predicate(
        _predicate(kind, arguments),
        context,
    )
    solver = _solve(scenario.model)
    evidence = lowering.extract(solver)

    assert solver.boolean_value(lowering.literal)
    assert evidence["kind"] == kind
    assert evidence["satisfied"] is True
    assert evidence["evidenceExtractor"].startswith("cp_sat.deal_")
    assert len(evidence["sha256"]) == len(lowering.lowering_hash) == 64
    scenario.model.add(lowering.literal == 1)
    assert solver.solve(scenario.model) == cp_model.OPTIMAL


@pytest.mark.parametrize(("kind", "arguments"), FALSE_DEAL_PREDICATES)
def test_all_fourteen_deal_predicate_complements_are_exact(rules, kind, arguments):
    scenario, context = _context(rules)
    lowering = DealPredicateLoweringRegistryV2().lower_predicate(
        _predicate(kind, arguments),
        context,
    )
    solver = _solve(scenario.model)

    assert not solver.boolean_value(lowering.literal)
    assert lowering.extract(solver)["satisfied"] is False
    scenario.model.add(lowering.literal == 1)
    assert solver.solve(scenario.model) == cp_model.INFEASIBLE


def test_registration_is_complete_stable_and_engine_free():
    registry = DealPredicateLoweringRegistryV2()
    kinds = tuple(item.kind for item in DEAL_PREDICATE_REGISTRATIONS_V2)

    assert len(kinds) == len(set(kinds))
    published = {definition.kind for definition in
        ScenarioPredicateCatalogLoweringManifestV2().exposed_definitions
        if definition.kind.startswith("deal.") and "hard" in definition.allowed_hardness}
    assert set(kinds) == published
    for fixtures in (TRUE_DEAL_PREDICATES, FALSE_DEAL_PREDICATES):
        fixture_kinds = [kind for kind, _ in fixtures]
        assert len(fixture_kinds) == len(set(fixture_kinds))
        assert set(fixture_kinds) == published
    assert registry.registrations == DEAL_PREDICATE_REGISTRATIONS_V2
    assert len(registry.manifest_hash) == 64

    source = Path(
        "src/baloot_platform/research/scenario_deal_predicates_v2.py"
    ).read_text(encoding="utf-8")
    assert "BalootEngine" not in source
    assert "HeuristicBot" not in source
    assert "scenario_evidence" not in source


def test_named_assumptions_report_a_contradictory_owner_core(rules):
    scenario, context = _context(rules)
    registry = DealPredicateLoweringRegistryV2()
    north = registry.lower_predicate(
        _predicate(
            "deal.card_owner",
            {"card": 9, "seat": "north", "handStage": "initial_five"},
            "PRED-NORTH",
        ),
        context,
        enforce_as_assumption=True,
    )
    west = registry.lower_predicate(
        _predicate(
            "deal.card_owner",
            {"card": 9, "seat": "west", "handStage": "initial_five"},
            "PRED-WEST",
        ),
        context,
        enforce_as_assumption=True,
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1

    assert solver.solve(scenario.model) == cp_model.INFEASIBLE
    assert north.assumption is not None and north.assumption.name == "assume_pred_north"
    assert west.assumption is not None and west.assumption.name == "assume_pred_west"
    core = set(solver.sufficient_assumptions_for_infeasibility())
    assert core <= {north.assumption.index, west.assumption.index}
    assert west.assumption.index in core


@pytest.mark.parametrize(
    "predicate",
    [
        {
            "predicateId": "PRED-BAD",
            "kind": "deal.purchase_card",
            "version": 1,
            "hardness": "hard",
            "arguments": {"card": 20, "unknown": True},
        },
        {
            "predicateId": "PRED-BAD",
            "kind": "deal.purchase_card",
            "version": 1,
            "hardness": "hard",
            "arguments": {"card": True},
        },
        {
            "predicateId": "PRED-BAD",
            "kind": "deal.suit_count",
            "version": 1,
            "hardness": "hard",
            "arguments": {
                "seat": "north",
                "suit": "hearts",
                "minimum": 4,
                "maximum": 3,
                "handStage": "initial_five",
            },
        },
    ],
)
def test_deal_argument_schemas_reject_unknown_coercive_and_inverted_ranges(
    rules,
    predicate,
):
    scenario, context = _context(rules)
    parsed = ScenarioPredicateV1.build(predicate)

    with pytest.raises((TypeError, ValueError)):
        DealPredicateLoweringRegistryV2().lower_predicate(parsed, context)

    assert scenario.model is context.model


def test_remaining_predicates_fail_closed_without_play_ownership(rules):
    _, context = _context(rules, bind_remaining=False)

    with pytest.raises(ValueError, match="bound play ownership"):
        DealPredicateLoweringRegistryV2().lower_predicate(
            _predicate(
                "deal.remaining_card_owner",
                {"card": 25, "seat": "south"},
            ),
            context,
        )


def test_evidence_and_lowering_hashes_are_repeatable(rules):
    observed = []
    for _ in range(2):
        scenario, context = _context(rules)
        lowering = DealPredicateLoweringRegistryV2().lower_predicate(
            _predicate("deal.purchase_card", {"card": 20}),
            context,
        )
        observed.append((lowering.lowering_hash, lowering.extract(_solve(scenario.model))))

    assert observed[0] == observed[1]


@pytest.mark.parametrize("seat", list(Seat))
@pytest.mark.parametrize("other", list(Seat))
@pytest.mark.parametrize("relation", ["same", "partner", "next", "previous", "opponent"])
def test_v3_seat_relation_is_native_and_exact(seat, other, relation):
    context = SimpleNamespace(model=cp_model.CpModel(), sha256="1" * 64)
    predicate = _predicate("deal.seat_relation", {
        "seat": seat.name.lower(), "otherSeat": other.name.lower(), "relation": relation,
    })
    manifest = ScenarioPredicateCatalogLoweringManifestV3()
    lowering = manifest.lower_predicate(predicate, context)
    assert list(context.model.proto.variables[lowering.literal.index].domain) == [0, 1]
    assert "int_mod" in str(context.model.proto)
    solver = _solve(context.model)
    expected = {
        "same": other == seat, "partner": other == seat.partner,
        "next": other == seat.next(), "previous": other == seat.previous(),
        "opponent": other.team != seat.team,
    }[relation]
    assert solver.boolean_value(lowering.literal) == expected
    assert lowering.extract(solver)["observed"]["matches"] == expected
    context.model.add(lowering.literal == int(not expected))
    assert solver.solve(context.model) == cp_model.INFEASIBLE


def test_v3_seat_relation_binds_native_identity_and_keeps_v2_frozen(rules):
    predicate = _predicate("deal.seat_relation", {
        "seat": "south", "otherSeat": "north", "relation": "partner",
    })
    scenario, context = _context(rules)
    legacy = DealPredicateLoweringRegistryV2().lower_predicate(predicate, context)
    definition = DealPredicateLoweringRegistryV2()._schema_registry.require(predicate)
    assert legacy.lowering_hash == canonical_sha256({
        "schemaVersion": 2, "predicateHash": predicate.sha256,
        "contextHash": context.sha256, "definitionHash": definition.sha256,
        "evidenceExtractor": legacy.evidence_extractor_id,
    })
    assert list(scenario.model.proto.variables[legacy.literal.index].domain) == [1, 1]
    manifest = ScenarioPredicateCatalogLoweringManifestV3()
    native = manifest.lower_predicate(predicate, ScenarioPredicateStageContextV3(context, {}))
    assert native.lowering_hash != legacy.lowering_hash
    assert native.evidence_extractor_id == legacy.evidence_extractor_id
    assert native.extract(_solve(scenario.model))["observed"] == legacy.extract(_solve(scenario.model))["observed"]
    assert manifest.catalog()["nativeLoweringImplementations"][predicate.kind] == "cp_sat.seat_relation_modulo_delta_v1"
    assert ScenarioPredicateCatalogLoweringManifestV2().manifest_hash == "d36815fde96af7246bfb67153f75dd0cc7962fc5b18827a3e6582b764a9a600d"
