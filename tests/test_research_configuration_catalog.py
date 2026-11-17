from __future__ import annotations

import copy
import hashlib

import pytest
from pydantic import ValidationError

from baloot_platform.research.contracts import (
    ResearchRecipeV1,
    build_legacy_research_recipe_for_read,
    research_recipe_configuration_v1,
)
from baloot_platform.service.api import (
    RESEARCH,
    ResearchCapabilityResponse,
    ResearchQualificationRequest,
    ResearchRecipeRequest,
    StoredResearchRecipeV3Response,
    app,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _recipe() -> dict:
    return {
        "schemaVersion": 1,
        "titleAr": "لفة بحث تجريبية",
        "output": {"kind": "laffa", "resultCount": 1},
        "profiles": {
            "domainHash": _hash("domain"),
            "bundleHash": _hash("bundle"),
            "rulesHash": _hash("rules"),
            "conventionHash": _hash("convention"),
            "agentRelease": "agent@frozen",
            "analysisProfileHash": _hash("analysis"),
            "scoreProfileHash": _hash("score"),
        },
        "situation": {"dealer": "east", "contracts": ["sun"]},
        "tactics": [],
        "qualification": {},
        "resources": {},
        "status": "draft",
    }


def _strategic_play_profile() -> dict:
    return {
        "schemaVersion": 1,
        "minimumPlayedCards": 24,
        "requestedSamples": 2,
        "exactAllocationLimit": 2,
        "maximumSamplingAttempts": 2,
        "maximumWorlds": 2,
        "maximumContinuationSteps": 100,
        "maximumPlies": 8,
        "maximumSelectionSeconds": 1.0,
        "requireExactForQualification": True,
        "continuationPolicy": "observer_safe_heuristic_v1",
        "policyProfileId": "maximin_margin",
        "policyRevision": 1,
        "outcomeBoundary": 1,
        "utility": "actor_team_awarded_point_margin",
        "tieBreak": "maximin_then_total_then_card_id",
    }


def test_recipe_catalog_is_complete_versioned_and_self_describing():
    catalog = research_recipe_configuration_v1()
    assert catalog["recipeSchemaVersion"] == 3
    assert catalog["supportedRecipeSchemaVersions"] == [1, 2, 3]
    assert catalog["legacyReadSchemaVersions"] == [1, 2]
    assert len(catalog["sha256"]) == 64

    for path, field in catalog["fields"].items():
        assert {
            "control",
            "valueType",
            "mode",
            "helpKey",
            "example",
        } <= set(field), path
        assert ("default" in field) != ("required" in field), path

    assert catalog["fields"]["configuration.bundleRevisionId"]["required"] is True
    assert catalog["fields"]["resources.seedCount"]["default"] == 10
    assert catalog["fields"]["situation.purchaseRound"]["valueType"] == (
        "integer_or_null"
    )
    assert catalog["fields"]["situation.escalations"]["valueType"] == "array"
    assert catalog["fields"]["qualification.requireObserverSafety"]["default"] is True
    assert catalog["fields"]["qualification.requireAllDecisionAudit"]["default"] is True
    assert catalog["fields"]["renderPolicy.mode"]["value"] == "on_demand"
    assert catalog["fields"]["renderPolicy.mode"]["default"] == "on_demand"
    auction_samples = catalog["fields"]["resources.strategicAuction.requestedSamples"]
    assert auction_samples["mode"] == "derived"
    assert auction_samples["default"] == 192
    assert auction_samples["value"] == 192
    assert auction_samples["derivedFrom"] == (
        "configuration.challenge.confirmation.bid_samples"
    )
    assert auction_samples["semantics"] == "compatibility_mirror"
    assert "values" not in auction_samples
    strategic_auction_default = catalog["fields"]["resources.strategicAuction"][
        "default"
    ]
    assert strategic_auction_default["requestedSamples"] == 192
    assert strategic_auction_default["tieBreak"] == (
        "round_lead_probability_then_best_share_then_legal_action_order_v1"
    )
    assert catalog["fields"]["tactics[].mode"]["values"] == [
        "preference",
        "qualification_gate",
    ]
    semantics = catalog["fields"]["tactics"]["modeSemantics"]
    assert semantics["preference"]["effect"] == "selection_score"
    assert semantics["preference"]["qualificationGate"] is False
    assert semantics["qualification_gate"]["effect"] == ("post_replay_qualification")
    assert semantics["qualification_gate"]["qualificationGate"] is True


def test_recipe_catalog_owns_relations_presets_and_unsupported_variants():
    catalog = research_recipe_configuration_v1()
    relationships = catalog["relationships"]
    codes = [str(item["code"]) for item in relationships]
    assert len(codes) == len(set(codes))
    assert {
        "actor_safe_auction_requires_ortools",
        "strategic_play_requires_ortools",
        "tactical_precursor_supports_only_small_card_gate",
        "tactical_deal_balance_requires_supported_tactic",
        "at_most_one_hard_project",
    } <= set(codes)
    for relationship in relationships:
        assert relationship["labelKey"]
        assert relationship["helpKey"]
        assert set(relationship["example"]) == {"valid", "invalid"}

    assert [item["code"] for item in catalog["recipePresets"]] == [
        "project_400",
        "small_card_gain",
        "only_winning_move",
        "large_card_sacrifice",
        "critical_score",
    ]
    recipe_presets = {item["code"]: item["patch"] for item in catalog["recipePresets"]}
    assert (
        recipe_presets["small_card_gain"]["resources"]["tacticalDealBalance"]["enabled"]
        is True
    )
    resource_presets = {
        item["code"]: item["patch"]["resources"] for item in catalog["resourcePresets"]
    }
    assert resource_presets["quick_exploration"] == {
        "seedCount": 20,
        "candidatesPerSeed": 1,
        "seedsPerShard": 10,
        "solverTimeoutSeconds": 2.0,
        "maximumConflicts": 50_000,
    }
    assert resource_presets["standard_research"]["seedCount"] == 100
    assert resource_presets["thorough_campaign"]["seedCount"] == 1_000
    assert {item["code"] for item in catalog["unsupportedVariants"]} == {
        "direct_contextualized_laffa",
        "direct_sakka_monolith",
        "actor_safe_auction_escalation",
        "soft_project_targeting",
        "multiple_hard_projects",
        "tactical_precursor_other_than_small_card_gain",
        "full_round_global_optimal_play",
        "alternate_strategic_auction_policy",
        "alternate_strategic_play_policy",
        "placeholder_diversity_dimensions",
        "automatic_research_render",
        "uncompiled_hard_tactical_mode",
    }
    assert all(
        tactic["mode"] != "hard"
        for preset in catalog["recipePresets"]
        for tactic in preset["patch"]["tactics"]
    )


@pytest.mark.parametrize(
    "gate",
    ["requireObserverSafety", "requireAllDecisionAudit"],
)
def test_mandatory_qualification_gates_are_legacy_readable_but_fail_closed_for_requests(
    gate: str,
):
    payload = _recipe()
    payload["qualification"] = {gate: False}
    legacy = build_legacy_research_recipe_for_read(payload)
    assert isinstance(legacy, ResearchRecipeV1)
    assert legacy.qualification[gate] is False
    with pytest.raises(ValidationError):
        ResearchQualificationRequest.model_validate({gate: False})


def test_tactical_precursor_rejects_uncompiled_gated_tactic_mix():
    payload = _recipe()
    payload["tactics"] = [
        {
            "kind": "small_card_big_gain",
            "mode": "qualification_gate",
            "minimumScore": 70,
            "weight": 1,
            "perspectives": [],
        },
        {
            "kind": "only_winning_move",
            "mode": "hard",
            "minimumScore": 70,
            "weight": 1,
            "perspectives": [],
        },
    ]
    payload["resources"] = {
        "generationLane": "ortools",
        "playConstructionMode": "cp_sat_feasibility_line",
        "tacticalConstruction": {
            "schemaVersion": 1,
            "id": "tactical_play_precursor_v1",
            "revision": 1,
            "enabled": True,
            "maximumPayoffTricks": 4,
            "minimumLaterRawPoints": 10,
            "requirePrecursorForGatedTactics": True,
        },
    }
    with pytest.raises(ValueError, match="cannot compile every gated tactic"):
        ResearchRecipeV1.build(payload)


def test_recipe_api_uses_explicit_discriminators_and_stored_lifecycle():
    openapi = app.openapi()
    create = openapi["paths"]["/api/v2/research-recipes"]["post"]
    validate = openapi["paths"]["/api/v2/research-recipes:validate"]["post"]
    request_schema = create["requestBody"]["content"]["application/json"]["schema"]
    response_schema = create["responses"]["201"]["content"]["application/json"][
        "schema"
    ]
    assert request_schema == {
        "$ref": "#/components/schemas/ResearchRecipeV4CreateRequest"
    }
    assert response_schema == {
        "$ref": "#/components/schemas/StoredResearchRecipeV4Response"
    }
    validation_schema = validate["requestBody"]["content"]["application/json"]["schema"]
    assert len(validation_schema["oneOf"]) == 4
    assert validation_schema["discriminator"]["propertyName"] == "schemaVersion"
    assert set(validation_schema["discriminator"]["mapping"]) == {
        "1",
        "2",
        "3",
        "4",
    }

    minimal = copy.deepcopy(_recipe())
    minimal.pop("schemaVersion")
    with pytest.raises(ValidationError):
        ResearchRecipeRequest.model_validate(minimal)
    with pytest.raises(ValidationError):
        StoredResearchRecipeV3Response.model_validate({"schemaVersion": 3})


def test_v4_scenario_catalog_is_closed_typed_and_runtime_current():
    payload = RESEARCH.capabilities(include_payloads=False)["research"]
    model = ResearchCapabilityResponse.model_validate(payload)
    scenario = model.scenarioConfiguration
    assert scenario.recipeSchemaVersion == 4
    assert scenario.supportedLegacyReadVersions == [1, 2, 3]
    assert len(scenario.predicateRegistry.definitions) >= 36
    assert set(scenario.proofBudgets.profiles) == {
        "interactive_validation_v1",
        "bounded_v1",
        "late_game_exact_v1",
        "actor_information_set_exact_v1",
        "offline_heavyweight_v1",
    }
    bounded_budget = scenario.proofBudgets.profiles["bounded_v1"]
    assert bounded_budget.maximumDepth == 40
    assert bounded_budget.maximumHorizonActions == 40
    assert bounded_budget.maximumCpuSeconds == 10.0
    assert bounded_budget.maximumCacheBytes == 67_108_864
    assert bounded_budget.cancellationCheckIntervalNodes == 64
    assert bounded_budget.maximumWorkers == 1
    assert bounded_budget.maximumStateSpace == 50_000_000
    assert scenario.proofClaims == [
        "outcome_reachable",
        "outcome_forced",
        "winning_root_action_count",
        "winning_continuation_line_count",
        "selected_action_is_unique_winner",
        "target_team_has_no_salvage",
    ]
    assert scenario.proofUtilities == [
        "awarded_point_margin",
        "contract_success",
        "round_win",
        "match_win",
        "kabout",
    ]
    assert scenario.proofBoundaryEvaluators == ["current_raw_point_margin_v1"]
    assert scenario.proofLineUniverses == [
        "all_legal",
        "optimal_strategy",
        "policy_conditioned",
    ]
    assert scenario.proofHorizonUnits == [
        "plies",
        "decisions",
        "tricks",
        "round_end",
    ]
    assert set(scenario.defaultPolicyProfiles) == {
        "north",
        "west",
        "south",
        "east",
    }
    assert [item.id for item in scenario.cardOptions] == list(range(32))
    assert len({item.code for item in scenario.cardOptions}) == 32
    assert all(item.label and item.symbol for item in scenario.cardOptions)
    assert (
        scenario.tacticOptions == payload["tacticalVerification"]["supportedPatterns"]
    )
    assert scenario.projectOptions
    assert len(scenario.projectDomainHash) == 64
    assert {item.contract for item in scenario.projectOptions} == {
        "sun",
        "hokum",
        "ashkal",
    }
    assert any(
        item.contract == "sun"
        and item.kind == "hundred_sequence"
        and item.value == 20
        and item.highRank == "ace"
        and item.highRankId == 7
        for item in scenario.projectOptions
    )
    assert scenario.ruleOptions.contracts == ["sun", "hokum", "ashkal"]
    assert [
        item.model_dump() for item in scenario.ruleOptions.auctionActionsByRound
    ] == [
        {"round": 1, "actions": ["pass", "sun", "hokum", "ashkal"]},
        {"round": 2, "actions": ["pass", "sun", "hokum"]},
    ]
    assert {
        item.contract: {"levels": item.levels, "actions": item.actions}
        for item in scenario.ruleOptions.escalationLevelsByContract
    } == {
        "sun": {
            "levels": ["normal", "double"],
            "actions": ["decline", "double"],
        },
        "hokum": {
            "levels": ["normal", "double", "triple", "four", "gahwa"],
            "actions": ["decline", "double", "triple", "four", "gahwa"],
        },
        "ashkal": {
            "levels": ["normal", "double"],
            "actions": ["decline", "double"],
        },
    }
    assert scenario.ruleOptions.ashkalBuyerRelations == ["bidder_partner"]
    assert scenario.ruleOptions.trumpContracts == ["hokum"]
    assert scenario.ruleOptions.balootContracts == ["hokum"]
    assert len(scenario.ruleOptions.playActionOptions) == 64
    assert [item.actionId for item in scenario.ruleOptions.playActionOptions] == list(
        range(64)
    )
    assert all(
        item.code.endswith("+ekka") == item.ekka
        for item in scenario.ruleOptions.playActionOptions
    )
    assert [
        item.card.id for item in scenario.ruleOptions.playActionOptions[:32]
    ] == list(range(32))
    assert scenario.ruleOptions.projectsEnabled is True
    assert scenario.ruleOptions.kaboutEnabled is True

    schemas = app.openapi()["components"]["schemas"]
    capability = schemas["ResearchCapabilityResponse"]
    assert capability["properties"]["scenarioConfiguration"] == {
        "$ref": "#/components/schemas/ResearchScenarioConfigurationV1Response"
    }
    for schema_name in (
        "ResearchScenarioConfigurationV1Response",
        "ScenarioPredicateRegistryV1Response",
        "ScenarioPredicateDefinitionV1Response",
        "ScenarioProofBudgetRegistryV1Response",
        "ScenarioProofBudgetV1Response",
        "ScenarioPolicyProfileV1Response",
        "ScenarioCardOptionV1Response",
        "ScenarioPlayActionOptionV1Response",
        "ScenarioProjectOptionV1Response",
        "ScenarioAuctionActionsByRoundV1Response",
        "ScenarioEscalationOptionsByContractV1Response",
        "ScenarioRuleOptionsV1Response",
    ):
        assert schemas[schema_name]["additionalProperties"] is False
