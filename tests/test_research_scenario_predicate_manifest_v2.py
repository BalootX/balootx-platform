from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.research.ortools_escalation_stage import OrToolsEscalationStageV1
from baloot_platform.research.ortools_scenario_model import OrToolsScenarioModelV1
from baloot_platform.research.scenario_contracts import (
    ScenarioPredicateV1,
    canonical_sha256,
)
from baloot_platform.research.scenario_predicate_auction_escalation_v2 import (
    CpSatAuctionPredicateContextV1,
    CpSatEscalationPredicateContextV1,
)
from baloot_platform.research.scenario_predicate_manifest_v2 import (
    EXPECTED_CATEGORY_COUNTS_V2,
    ScenarioPredicateCatalogLoweringManifestV2,
    shared_scenario_predicate_catalog_lowering_manifest_v2,
)
from baloot_platform.research import scenario_predicate_manifest_v2 as manifest_module
from baloot_platform.research.scenario_registry_v2 import (
    GENERAL_PREDICATE_KINDS_V2,
    scenario_predicate_definitions_v2,
    registered_scenario_predicate_definitions_v2,
)


def test_merged_manifest_dynamically_covers_authoritative_definitions(
    root: Path,
):
    manifest = ScenarioPredicateCatalogLoweringManifestV2()
    authoritative = {row.kind for row in scenario_predicate_definitions_v2()}
    registered = {row.kind for row in manifest.registrations}
    exposed = {row.kind for row in manifest.exposed_definitions}
    hidden = {row.kind for row in manifest.hidden_definitions}
    actual_category_counts = Counter(row.category for row in manifest.registrations)

    assert len(manifest.registrations) == len(registered) == len(authoritative)
    assert len(manifest.exposed_definitions) == len(exposed) == len(authoritative)
    assert hidden == set()
    assert exposed | hidden == authoritative
    assert exposed & hidden == set()
    assert registered == exposed
    assert dict(EXPECTED_CATEGORY_COUNTS_V2) == dict(actual_category_counts)
    assert manifest.catalog()["categoryCounts"] == dict(actual_category_counts)

    for definition in manifest.exposed_definitions:
        assert "hard" in definition.allowed_hardness
        assert definition.cp_sat_support.search_status in {"exact", "derived"}
        assert definition.cp_sat_support.exact_input_status == "exact"
        assert definition.cp_sat_support.evidence_extractor
        assert definition.publication_issues() == ()
        registration = next(
            row for row in manifest.registrations if row.kind == definition.kind
        )
        assert registration.owning_test in definition.tests
        assert (root / registration.owning_test).is_file()


def test_manifest_fails_closed_when_authoritative_catalog_grows_without_lowering(
    monkeypatch: pytest.MonkeyPatch,
):
    definitions = scenario_predicate_definitions_v2()
    monkeypatch.setattr(
        manifest_module,
        "scenario_predicate_definitions_v2",
        lambda: (*definitions, SimpleNamespace(kind="future.unlowered")),
    )

    with pytest.raises(ValueError, match=r"missing=\['future\.unlowered'\]"):
        ScenarioPredicateCatalogLoweringManifestV2()


def test_observer_visibility_is_exposed_only_with_exact_observation_binding():
    manifest = ScenarioPredicateCatalogLoweringManifestV2()
    definition = manifest.definition("state.observer_visibility")

    assert definition is not None
    assert definition.allowed_hardness == ("hard", "report")
    assert definition.cp_sat_support.search_status == "derived"
    assert definition.cp_sat_support.exact_input_status == "exact"
    assert definition.cp_sat_support.evidence_extractor
    assert definition.publication_issues() == ()
    registration = next(
        row for row in manifest.registrations if row.kind == "state.observer_visibility"
    )
    assert registration.category == "play_state"


def test_flagship_predicates_are_fail_closed_until_registered_then_hard_exact():
    base = {row.kind: row for row in scenario_predicate_definitions_v2()}
    manifest = ScenarioPredicateCatalogLoweringManifestV2()

    for kind in GENERAL_PREDICATE_KINDS_V2:
        assert base[kind].allowed_hardness == ("report",)
        assert base[kind].cp_sat_support.search_status == "unsupported"
        assert base[kind].cp_sat_support.exact_input_status == "unsupported"

        definition = manifest.definition(kind)
        assert definition is not None
        assert definition.allowed_hardness == ("hard", "report")
        assert definition.cp_sat_support.search_status in {"exact", "derived"}
        assert definition.cp_sat_support.exact_input_status == "exact"
        assert definition.cp_sat_support.evidence_extractor


def test_remaining_deal_predicates_are_truthfully_play_prefix_derived():
    manifest = ScenarioPredicateCatalogLoweringManifestV2()

    for kind in (
        "deal.remaining_card_owner",
        "deal.remaining_suit_count",
        "deal.remaining_rank_count",
    ):
        definition = manifest.definition(kind)
        assert definition is not None
        assert definition.stage == "deal"
        assert definition.cp_sat_support.search_status == "derived"
        assert definition.cp_sat_support.qualification_source == "derived_state"
        assert definition.cp_sat_support.search_stages == ("deal", "playPrefix")
        assert definition.publication_issues() == ()


def test_manifest_hash_is_canonical_deterministic_and_binds_every_owner():
    first = ScenarioPredicateCatalogLoweringManifestV2()
    second = ScenarioPredicateCatalogLoweringManifestV2()
    catalog = first.catalog()
    declared = catalog.pop("sha256")

    assert first.manifest_hash == second.manifest_hash == declared
    assert declared == canonical_sha256(catalog)
    assert len(declared) == 64
    assert {row.category for row in first.registrations} == set(
        EXPECTED_CATEGORY_COUNTS_V2
    )
    assert all(
        row.lowerer_owner and row.evidence_extractor_id for row in first.registrations
    )
    assert (
        next(
            row
            for row in first.registrations
            if row.kind == "proof.winning_root_action_count"
        ).category
        == "proof_tactic"
    )
    assert (
        next(
            row
            for row in first.registrations
            if row.kind == "proof.winning_root_action_matches"
        ).category
        == "general"
    )


def test_shared_manifest_reuses_immutable_catalog_authority():
    first = shared_scenario_predicate_catalog_lowering_manifest_v2()
    second = shared_scenario_predicate_catalog_lowering_manifest_v2()
    expected = first.catalog()
    projection = first.catalog()

    projection["categoryCounts"]["deal"] = 0
    projection["registrations"].clear()

    assert first is second
    assert first.catalog() == expected


def test_registration_upgrade_requires_matching_extractors_tests_and_binding():
    with pytest.raises(ValueError, match="matching extractor and test sets"):
        registered_scenario_predicate_definitions_v2(
            {"deal.card_owner": "cp_sat.test"},
            {},
        )
    with pytest.raises(ValueError, match="unknown scenario predicate"):
        registered_scenario_predicate_definitions_v2(
            {"unknown.kind": "cp_sat.unknown"},
            {"unknown.kind": "tests/unknown.py"},
        )


def test_manifest_dispatches_actual_auction_and_escalation_lowerings(rules):
    manifest = ScenarioPredicateCatalogLoweringManifestV2()

    scenario = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    fragment = scenario.add_auction_fragment()
    auction = manifest.lower_predicate(
        ScenarioPredicateV1.build(
            {
                "predicateId": "PRED-MANIFEST-AUCTION",
                "kind": "auction.dealer",
                "version": 1,
                "hardness": "hard",
                "arguments": {"seat": "east"},
            }
        ),
        CpSatAuctionPredicateContextV1(
            scenario.model,
            fragment,
            Seat.EAST,
        ),
    )
    scenario.model.add(auction.literal == 1)
    auction_solver = cp_model.CpSolver()
    auction_solver.parameters.num_search_workers = 1
    assert auction_solver.solve(scenario.model) == cp_model.OPTIMAL
    assert auction.extract(auction_solver)["satisfied"] is True

    stage = OrToolsEscalationStageV1(
        EscalationDomainIRV1.from_rules(rules.data),
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
    )
    escalation_context = CpSatEscalationPredicateContextV1.attach(stage)
    escalation = manifest.lower_predicate(
        ScenarioPredicateV1.build(
            {
                "predicateId": "PRED-MANIFEST-ESCALATION",
                "kind": "escalation.level",
                "version": 1,
                "hardness": "hard",
                "arguments": {"level": "normal"},
            }
        ),
        escalation_context,
    )
    escalation_context.model.add(escalation.literal == 1)
    escalation_solver = cp_model.CpSolver()
    escalation_solver.parameters.num_search_workers = 1
    assert escalation_solver.solve(escalation_context.model) == cp_model.OPTIMAL
    assert escalation.extract(escalation_solver)["satisfied"] is True
