from __future__ import annotations

import copy

import pytest

from baloot_platform.research.scenario_registry import (
    ScenarioPredicateRegistryV1,
)
from baloot_platform.research.scenario_registry_v2 import (
    EXACT_V1_COMPATIBILITY_KINDS,
    CardSelectorV1,
    CpSatSupportV1,
    ScenarioPredicateDefinitionV2,
    compatible_scenario_predicates_v2,
    play_action_matches_definition_v2,
    validate_predicate_publication_v2,
)


def _exact_support() -> dict:
    return {
        "schemaVersion": 1,
        "search": {"status": "exact", "stages": ["deal"]},
        "exactInput": {"status": "exact", "stages": ["deal"]},
        "qualificationSource": "construction",
        "actorInformationSet": "public",
        "proof": {
            "exact": "exact",
            "bounded": "exact",
            "policyConditioned": "exact",
        },
        "evidenceExtractor": "scenario_structural_compiler_v1.predicate_evidence",
    }


def _definition_payload(definition: ScenarioPredicateDefinitionV2) -> dict:
    payload = definition.to_dict()
    payload.pop("schemaVersion")
    return payload


def test_cp_sat_support_is_closed_strict_and_round_trips():
    support = CpSatSupportV1.build(_exact_support())

    assert support.search_status == "exact"
    assert support.search_stages == ("deal",)
    assert support.to_dict() == _exact_support()

    unknown = _exact_support()
    unknown["runtimeVerifier"] = "not_part_of_v2"
    with pytest.raises(ValueError, match="unknown"):
        CpSatSupportV1.build(unknown)

    coerced = _exact_support()
    coerced["schemaVersion"] = "1"
    with pytest.raises(ValueError, match="schemaVersion"):
        CpSatSupportV1.build(coerced)

    contradictory = _exact_support()
    contradictory["search"] = {"status": "unsupported", "stages": ["deal"]}
    with pytest.raises(ValueError, match="must be empty"):
        CpSatSupportV1.build(contradictory)

    missing_extractor = _exact_support()
    missing_extractor["evidenceExtractor"] = ""
    with pytest.raises(ValueError, match="evidence extractor"):
        CpSatSupportV1.build(missing_extractor)

    mismatched_source = _exact_support()
    mismatched_source["qualificationSource"] = "derived_state"
    with pytest.raises(ValueError, match="derived search support"):
        CpSatSupportV1.build(mismatched_source)


def test_card_selector_is_closed_canonical_and_non_vacuous():
    value = {
        "cards": [31, 0],
        "ranks": ["ace", "seven"],
        "suits": ["clubs"],
        "ekka": "forbidden",
        "owner": {"relation": "seat", "seat": "west"},
    }

    first = CardSelectorV1.build(value)
    second = CardSelectorV1.build(copy.deepcopy(value))

    assert first == second
    assert first.sha256 == second.sha256
    assert first.cards == (31, 0)
    assert first.owner_relation == "seat"
    assert first.to_dict()["owner"] == {"relation": "seat", "seat": "west"}
    assert CardSelectorV1.build(first.to_dict()) == first

    with pytest.raises(ValueError, match="effective filter"):
        CardSelectorV1.build({})
    with pytest.raises(ValueError, match="unknown fields"):
        CardSelectorV1.build({"ranks": ["ace"], "oracle": True})
    with pytest.raises(TypeError, match="integers"):
        CardSelectorV1.build({"cards": ["31"]})
    with pytest.raises(ValueError, match="unique"):
        CardSelectorV1.build({"ranks": ["ace", "ace"]})
    with pytest.raises(ValueError, match="requires only seat"):
        CardSelectorV1.build(
            {"owner": {"relation": "seat", "seat": "west", "team": "west_east"}}
        )
    with pytest.raises(ValueError, match="does not accept"):
        CardSelectorV1.build(
            {"owner": {"relation": "actor", "seat": "west"}}
        )


def test_v1_adapter_preserves_identity_and_complements_without_overclaiming():
    v1 = ScenarioPredicateRegistryV1()
    adapted = compatible_scenario_predicates_v2()
    rows = {definition.kind: definition for definition in adapted}

    assert len(rows) == 74
    assert {
        kind
        for kind, definition in rows.items()
        if definition.cp_sat_support.search_status == "exact"
    } == EXACT_V1_COMPATIBILITY_KINDS
    assert len(EXACT_V1_COMPATIBILITY_KINDS) == 10
    for kind, definition in rows.items():
        source = v1.definition(kind)
        assert source is not None
        assert definition.safe_complement is source.safe_complement
        assert definition.stage == source.stage
        assert definition.argument_schema == source.argument_schema

    assert rows["deal.card_owner"].publication_issues() == ()
    assert rows["project.exists"].cp_sat_support.search_status == "unsupported"
    assert rows["auction.action_sequence"].cp_sat_support.search_status == "unsupported"
    assert rows["escalation.action_sequence"].cp_sat_support.search_status == "unsupported"
    assert "hard" not in rows["project.exists"].allowed_hardness
    assert "search_not_exactly_supported" in rows["project.exists"].publication_issues()


def test_only_truthfully_supported_compatibility_rows_can_be_published():
    rows = compatible_scenario_predicates_v2()
    supported = tuple(
        row for row in rows if row.kind in EXACT_V1_COMPATIBILITY_KINDS
    )

    first_hash = validate_predicate_publication_v2(supported)
    second_hash = validate_predicate_publication_v2(tuple(reversed(supported)))

    assert first_hash == second_hash
    assert len(first_hash) == 64
    with pytest.raises(ValueError, match="project.exists"):
        validate_predicate_publication_v2(
            (*supported, next(row for row in rows if row.kind == "project.exists"))
        )
    with pytest.raises(ValueError, match="duplicate"):
        validate_predicate_publication_v2((supported[0], supported[0]))


def test_definition_v2_rejects_drift_and_publication_metadata_gaps():
    definition = next(
        row
        for row in compatible_scenario_predicates_v2()
        if row.kind == "deal.card_owner"
    )
    assert ScenarioPredicateDefinitionV2.build(_definition_payload(definition)) == definition

    unknown = _definition_payload(definition)
    unknown.pop("sha256")
    unknown["runtimeVerifier"] = "legacy_authority"
    with pytest.raises(ValueError, match="unknown"):
        ScenarioPredicateDefinitionV2.build(unknown)

    stale_hash = _definition_payload(definition)
    stale_hash["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        ScenarioPredicateDefinitionV2.build(stale_hash)

    no_docs = _definition_payload(definition)
    no_docs.pop("sha256")
    no_docs["documentation"] = []
    with pytest.raises(ValueError, match="documentation"):
        ScenarioPredicateDefinitionV2.build(no_docs)

    unsupported_hard = _definition_payload(definition)
    unsupported_hard.pop("sha256")
    unsupported_hard["cpSatSupport"] = {
        "schemaVersion": 1,
        "search": {"status": "unsupported", "stages": []},
        "exactInput": {"status": "unsupported", "stages": []},
        "qualificationSource": "report_only",
        "actorInformationSet": "public",
        "proof": {
            "exact": "unsupported",
            "bounded": "unsupported",
            "policyConditioned": "unsupported",
        },
        "evidenceExtractor": "",
    }
    unpublishable = ScenarioPredicateDefinitionV2.build(unsupported_hard)
    with pytest.raises(ValueError, match="not publishable"):
        unpublishable.require_publishable()


def test_play_action_matches_reuses_card_selector_and_is_fail_closed_until_lowered():
    definition = play_action_matches_definition_v2()

    assert definition.kind == "play.action_matches"
    assert definition.stage == "playPrefix"
    assert definition.allowed_hardness == ("report",)
    assert definition.safe_complement is False
    assert definition.cp_sat_support.search_status == "unsupported"
    assert definition.cp_sat_support.exact_input_status == "unsupported"
    assert definition.publication_issues() == (
        "search_not_exactly_supported",
        "exact_input_not_supported",
        "evidence_extractor_missing",
    )

    definition.validate_arguments(
        {
            "trickIndex": 5,
            "position": 3,
            "actingSeat": "west",
            "selector": {
                "ranks": ["ace"],
                "owner": {"relation": "actor"},
            },
        }
    )
    with pytest.raises(ValueError, match="effective filter"):
        definition.validate_arguments(
            {"trickIndex": 5, "position": 3, "selector": {}}
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        definition.validate_arguments(
            {
                "trickIndex": 5,
                "position": 3,
                "selector": {"ranks": ["ace"]},
                "hiddenOracle": True,
            }
        )
