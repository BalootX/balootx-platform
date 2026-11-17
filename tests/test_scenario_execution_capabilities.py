from __future__ import annotations

from pathlib import Path

import yaml

from baloot_platform.core.rules import RuleBook
from baloot_platform.research.scenario_backend import (
    PHASE4_FULL_SERVICE_PROVEN_COMBINATIONS_V1,
    PHASE4_REQUIRED_COMBINATIONS_V1,
    OrToolsScenarioResearchBackendV2,
)
from baloot_platform.research.scenario_contracts import canonical_sha256
from baloot_platform.research.scenario_contracts import (
    PLAYING_QUALITY_HORIZON_UNITS_BY_DECISION_V1,
    PROOF_BOUNDARY_EVALUATOR_DEFINITIONS,
    ScenarioPlayingQualityPolicyV1,
    SCENARIO_DECISION_KINDS,
)
from baloot_platform.research.scenario_execution_capabilities import (
    _sample_recipe,
    build_scenario_execution_capabilities_v1,
    project_scenario_configuration_execution_v1,
)
from baloot_platform.service.api import ScenarioExecutionCapabilitiesV1Response


def test_quality_authoring_reuses_frozen_proof_and_execution_authorities():
    backend = _backend("intelligent_research.yaml")
    catalog = build_scenario_execution_capabilities_v1(backend)["authoringCatalog"]
    quality = catalog["playingQuality"]
    assert quality["defaultBudgetProfileByModeAndVisibility"]["exact"]["complete_state"] == "offline_heavyweight_v1"
    assert quality["policySchemaVersion"] == 2
    assert quality["modesByOrigin"] == {"search": ["budgeted", "exact"], "exact_trace_prefix": ["budgeted", "exact"]}
    assert quality["visibilities"] == catalog["proof"]["visibilities"]
    assert quality["legacyVisibility"] == "actor_information_set"
    for visibility in quality["visibilities"]:
        for mode in quality["modes"]:
            budget = quality["defaultBudgetProfileByModeAndVisibility"][mode][visibility]
            assert budget in catalog["proof"]["budgetProfiles"]
            if mode == "exact":
                tier = catalog["proof"]["budgetProfiles"][budget]
                card_count = RuleBook.load(Path(__file__).resolve().parents[1]
                    / "configs/rules/saudi_federation_2023_v2.yaml").structural_domain.card_count
                assert tier["maximumDepth"] >= card_count
                assert tier["maximumHorizonActions"] >= card_count
            policy = ScenarioPlayingQualityPolicyV1.build({
                "schemaVersion": quality["policySchemaVersion"], "mode": mode, "visibility": visibility,
                "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": budget,
                    **({"horizon": {"unit": "plies", "value": 1},
                        "boundaryEvaluator": quality["defaultBoundaryEvaluatorByDecision"][kind]}
                       if mode == "budgeted" else {})} for kind in quality["decisionKinds"]},
            })
            assert all(policy.request_for(kind, "north").visibility == visibility
                       for kind in quality["decisionKinds"])
    assert quality["modes"] == ["budgeted", "exact"]
    assert quality["decisionKinds"] == list(SCENARIO_DECISION_KINDS)
    assert quality["defaultBudgetProfile"] == "bounded_v1"
    assert quality["defaultBudgetProfileByMode"] == {
        "budgeted": "bounded_v1",
        "exact": "offline_heavyweight_v1",
    }
    assert quality["defaultBoundaryEvaluatorByDecision"] == {
        "bid": "auction_hand_margin_v1",
        "escalation": "current_raw_point_margin_v1",
        "project": "current_raw_point_margin_v1",
        "play": "current_raw_point_margin_v1",
    }
    assert quality["horizonUnitsByDecision"] == {
        kind: list(units)
        for kind, units in PLAYING_QUALITY_HORIZON_UNITS_BY_DECISION_V1.items()
    }
    assert quality["execution"] == backend.authority.quality_execution.to_dict()
    definitions = quality["boundaryEvaluatorDefinitions"]
    assert set(definitions) == set(quality["boundaryEvaluators"])
    assert "auction_hand_margin_v1" not in catalog["proof"]["boundaryEvaluators"]
    assert definitions["auction_hand_margin_v1"]["kind"] == "heuristic"
    for name, definition in definitions.items():
        assert definition == {key: list(value) if isinstance(value, tuple) else value
                              for key, value in PROOF_BOUNDARY_EVALUATOR_DEFINITIONS[name].items()}
        assert definition["qualityOperatorVisible"] is True
        assert set(definition["decisionKinds"]) <= set(quality["decisionKinds"])
    for name, profile in catalog["proof"]["budgetProfiles"].items():
        bounds = quality["horizonBounds"][name]
        maximum = min(profile["maximumDepth"], profile["maximumHorizonActions"])
        assert bounds["plies"]["maximum"] == bounds["decisions"]["maximum"] == maximum
        assert bounds["tricks"]["maximum"] == maximum // 4
        assert bounds["round_end"] == {"minimum": 1, "step": 1}
    assert quality["recordedEscalationOpportunities"]["maxItems"] == backend.authority.quality_execution.maximum_escalation_decisions
    for path in quality["fieldSources"].values():
        section, field = path.split(".")
        assert catalog[section][field]
    assert catalog["sha256"] == canonical_sha256({key: value for key, value in catalog.items() if key != "sha256"})


def test_historical_quality_defaults_keep_late_game_budget():
    backend = _backend("intelligent_research.yaml", historical_quality_defaults=True)
    quality = build_scenario_execution_capabilities_v1(backend)["authoringCatalog"]["playingQuality"]
    assert quality["modesByOrigin"]["exact_trace_prefix"] == ["budgeted"]
    assert quality["defaultBudgetProfileByModeAndVisibility"]["exact"] == {
        "complete_state": "late_game_exact_v1",
        "actor_information_set": "actor_information_set_exact_v1",
    }


def _backend(
    profile_name: str = "progressive_mining_ortools.yaml",
    *,
    rules_profile: str | None = None,
    historical_quality_defaults: bool = False,
) -> OrToolsScenarioResearchBackendV2:
    root = Path(__file__).resolve().parents[1]
    profile = yaml.safe_load(
        (root / "configs/challenge" / profile_name).read_text(
            encoding="utf-8"
        )
    )
    if historical_quality_defaults:
        profile["profile"]["version"] = "3.7.0"
        profile["scenarioResearchV4"]["authoringDefaults"]["playingQuality"].pop("exactBudgetProfileByVisibility")
        profile["scenarioResearchV4"]["qualityExecution"].pop("allowExactInputQuality")
    return OrToolsScenarioResearchBackendV2(
        challenge_profile=profile,
        bundle_identity={
            "bundleRevisionId": "CBR-EXECUTION-CAPABILITY-TEST",
            "semanticBundleHash": "a" * 64,
            "effectiveRuntimeConfigHash": "b" * 64,
        },
        rules=(
            RuleBook.load(root / "configs/rules" / rules_profile)
            if rules_profile is not None
            else None
        ),
    )


def test_execution_capabilities_are_derived_from_the_backend_gate() -> None:
    backend = _backend()

    first = build_scenario_execution_capabilities_v1(backend)
    second = build_scenario_execution_capabilities_v1(backend)

    assert first == second
    assert first["sha256"] == canonical_sha256(
        {key: value for key, value in first.items() if key != "sha256"}
    )
    assert first["combinations"]
    assert {item["decisionKind"] for item in first["combinations"]} == {
        "bid",
        "escalation",
        "play",
    }
    assert first["decisionUnavailableReasons"] == {
        "project": "project_declaration_mode_missing_or_mandatory"
    }


    serialized = ScenarioExecutionCapabilitiesV1Response.model_validate(
        first
    ).model_dump(mode="json")
    assert serialized["decisionUnavailableReasons"] == first[
        "decisionUnavailableReasons"
    ]
    assert "predicateCapabilityManifest" not in first["authoringCatalog"]
    bid = [
        item for item in first["combinations"] if item["decisionKind"] == "bid"
    ]
    assert {(item["origin"], item["mode"], item["visibility"]) for item in bid} == {
        ("search", "exact", "complete_state"),
        ("search", "bounded", "complete_state"),
        ("search", "policy_conditioned", "complete_state"),
        ("exact_trace_prefix", "exact", "complete_state"),
        ("exact_trace_prefix", "bounded", "complete_state"),
        ("exact_trace_prefix", "policy_conditioned", "complete_state"),
        ("search", "exact", "actor_information_set"),
        ("exact_trace_prefix", "exact", "actor_information_set"),
    }
    assert all(
        item["claims"]
        == [
            "outcome_reachable",
            "outcome_forced",
            "winning_root_action_count",
            "winning_continuation_line_count",
            "selected_action_is_unique_winner",
            "target_team_has_no_salvage",
        ]
        and item["lineUniverses"] == ["policy_conditioned"]
        for item in bid
        if item["mode"] == "policy_conditioned"
    )
    assert all(
        item["claims"]
        == [
            "outcome_reachable",
            "outcome_forced",
            "winning_root_action_count",
            "winning_continuation_line_count",
            "selected_action_is_unique_winner",
            "target_team_has_no_salvage",
        ]
        and item["lineUniverses"] == ["all_legal", "optimal_strategy"]
        for item in bid
        if item["mode"] in {"exact", "bounded"}
        and item["visibility"] == "complete_state"
    )
    assert all(
        "winning_continuation_line_count" not in item["claims"]
        and item["lineUniverses"] == []
        for item in bid
        if item["visibility"] == "actor_information_set"
    )
    assert any(
        item["origin"] == "exact_trace_prefix"
        and item["decisionKind"] == "escalation"
        and item["mode"] == "policy_conditioned"
        and item["visibility"] == "complete_state"
        for item in first["combinations"]
    )
    complete_state_play = [
        item
        for item in first["combinations"]
        if item["decisionKind"] == "play"
        and item["visibility"] == "complete_state"
    ]
    assert complete_state_play
    assert all(
        "winning_root_action_count" in item["claims"]
        for item in complete_state_play
    )
    assert all(
        "winning_root_action_count" in item["claims"]
        for item in first["combinations"]
        if item["decisionKind"] == "escalation"
        and item["mode"] in {"exact", "bounded"}
    )
    assert all(
        "winning_root_action_count" not in item["claims"]
        for item in first["combinations"]
        if item["decisionKind"] == "escalation"
        and item["mode"] == "policy_conditioned"
    )
    policy_play = [
        item for item in complete_state_play if item["mode"] == "policy_conditioned"
    ]
    assert {item["origin"] for item in policy_play} == {
        "search",
        "exact_trace_prefix",
    }
    assert all(
        item["claims"]
        == [
            "outcome_reachable",
            "outcome_forced",
            "winning_root_action_count",
            "winning_continuation_line_count",
            "selected_action_is_unique_winner",
            "target_team_has_no_salvage",
        ]
        and item["lineUniverses"] == ["policy_conditioned"]
        for item in policy_play
    )
    assert all(
        not (
            item["visibility"] == "actor_information_set"
            and "winning_continuation_line_count" in item["claims"]
        )
        for item in first["combinations"]
    )
    actor_combinations = [
        item
        for item in first["combinations"]
        if item["visibility"] == "actor_information_set"
    ]
    assert len(actor_combinations) == 6
    assert {item["decisionKind"] for item in actor_combinations} == {
        "bid",
        "escalation",
        "play",
    }
    assert {item["origin"] for item in actor_combinations} == {
        "search",
        "exact_trace_prefix",
    }
    assert all(
        item["mode"] == "exact"
        and "winning_root_action_count" in item["claims"]
        and "winning_continuation_line_count" not in item["claims"]
        and item["lineUniverses"] == []
        for item in actor_combinations
    )
    assert all(
        item["contractValues"] == ["sun", "hokum"]
        for item in actor_combinations
        if item["decisionKind"] == "play"
    )
    assert all(
        item["contractValues"] is None
        for item in first["combinations"]
        if item["visibility"] == "complete_state"
    )
    play = first["decisionSelectorSchemas"]["play"]
    bid_selector = first["decisionSelectorSchemas"]["bid"]
    assert bid_selector["required"] == ["actionIndex"]
    assert bid_selector["properties"]["actionIndex"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 10000,
        "default": 0,
    }
    assert play["required"] == [
        "trickIndex",
        "remainingCards",
    ]
    assert "actingSeat" in play["properties"]
    assert play["properties"]["trickIndex"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 7,
        "default": 0,
    }
    assert play["properties"]["position"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 3,
    }
    assert play["properties"]["remainingCards"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 8,
        "default": 8,
    }
    authoring = first["authoringCatalog"]
    assert authoring["schemaVersion"] == 1
    assert authoring["recipeSchemaVersion"] == 4
    assert authoring["tacticKinds"] == [
        "contract_rescue",
        "debate",
        "entry_unblocking",
        "kabout_creation",
        "kabout_preservation",
        "kabout_prevention",
        "large_card_sacrifice",
        "only_winning_move",
        "partner_enablement",
        "project_bidding",
        "small_card_big_gain",
        "trump_control",
    ]
    assert authoring["executionCombinations"] == first["combinations"]
    assert authoring["decisionSelectorSchemas"] == first["decisionSelectorSchemas"]
    assert authoring["expressionLimits"] == {
        "maximumDepth": 8,
        "maximumNodes": 256,
        "maximumChildren": 64,
        "maximumSerializedBytes": 131072,
        "maximumCountRange": 64,
    }
    assert authoring["relationships"]
    assert all(
        field.get("labelKey") and field.get("helpKey")
        for field in authoring["fields"].values()
    )
    assert authoring["fields"]["titleAr"] == {
        "control": "text",
        "minimumLength": 3,
        "maximumLength": 160,
        "required": True,
        "defaultKey": "research_scenario_default_title",
        "labelKey": "research_scenario_recipe_title",
        "helpKey": "research_scenario_help_title",
    }
    assert (
        authoring["fields"]["resources.proofBudgetProfile"]["default"]
        == "late_game_exact_v1"
    )
    assert authoring["policyCatalog"]["defaultPolicyId"] == "fixed_action_map_v1"
    assert set(authoring["policyCatalog"]["defaultProfiles"]) == {
        "north",
        "west",
        "south",
        "east",
    }
    assert authoring["policyCatalog"]["seed"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 2147483647,
    }
    assert authoring["exactTrace"]["matchScore"]["items"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 10000,
    }
    assert authoring["localizationKeys"]["predicateHelp"] == {
        "play.action_matches": "research_scenario_predicate_play_action_matches_help",
        "play.trick_card_count": "research_scenario_predicate_play_trick_card_count_help",
        "play.trick_winner_matches": "research_scenario_predicate_play_trick_winner_matches_help",
        "proof.winning_root_action_matches": "research_scenario_predicate_proof_winning_root_action_matches_help",
    }
    assert authoring["sha256"] == canonical_sha256(
        {key: value for key, value in authoring.items() if key != "sha256"}
    )


def test_optional_shard_worker_control_tracks_selected_bundle_copy() -> None:
    backend = _backend()
    current = build_scenario_execution_capabilities_v1(backend)
    historical = build_scenario_execution_capabilities_v1(
        backend, expose_shard_worker_cap=False,
    )
    field = current["authoringCatalog"]["fields"]["resources.shardWorkerCount"]
    assert field["default"] is None
    assert field["maximumByBudgetProfile"]["offline_heavyweight_v1"] == 6
    assert "resources.shardWorkerCount" not in historical["authoringCatalog"]["fields"]


def test_execution_capability_sample_recipe_is_a_mapping() -> None:
    backend = _backend()

    sample = _sample_recipe(
        backend,
        origin="search",
        decision_kind="bid",
        mode="exact",
        visibility="complete_state",
        claim="outcome_reachable",
        line_universe=None,
    )

    assert isinstance(sample, dict)
    assert sample["scenario"]["decisionPoint"]["kind"] == "bid"


def test_v3_execution_capability_serializes_native_predicate_contexts() -> None:
    backend = _backend("progressive_mining_ortools_v3.yaml")
    capabilities = build_scenario_execution_capabilities_v1(backend)

    serialized = ScenarioExecutionCapabilitiesV1Response.model_validate(
        capabilities
    ).model_dump(mode="json")
    manifest = serialized["authoringCatalog"]["predicateCapabilityManifest"]

    assert manifest["backendId"] == "ortools_cp_sat_v3"
    assert manifest["baseManifestHash"] == backend.lowering_manifest.catalog()[
        "baseManifestHash"
    ]
    assert manifest["capabilities"]
    assert {
        (
            item["origin"],
            item["decisionKind"],
            item["mode"],
            item["visibility"],
        )
        for item in serialized["combinations"]
        } == {
            combination
            for combination in PHASE4_FULL_SERVICE_PROVEN_COMBINATIONS_V1
            if combination[1] != "project"
        }
    manifest_kinds = {item["kind"] for item in manifest["capabilities"]}
    assert "adjudication.out_of_turn_attempt" in manifest_kinds
    assert "adjudication.purchase_before_turn_attempt" in manifest_kinds
    assert all(
        context["decisionKind"] != "project"
        for capability in manifest["capabilities"]
        for context in capability["executionContexts"]
    )
    assert serialized["decisionUnavailableReasons"] == {
        "project": "project_declaration_mode_missing_or_mandatory",
    }


def test_v3_authoring_catalog_profiles_tactic_subtype_prerequisites() -> None:
    standard = build_scenario_execution_capabilities_v1(
        _backend("progressive_mining_ortools_v3.yaml")
    )
    optional = build_scenario_execution_capabilities_v1(
        _backend(
            "progressive_mining_ortools_v3.yaml",
            rules_profile="saudi_project_optional_unratified_v1.yaml",
        )
    )

    def argument_contexts(catalog: dict) -> dict[str, dict]:
        capability = next(
            item
            for item in catalog["authoringCatalog"][
                "predicateCapabilityManifest"
            ]["capabilities"]
            if item["kind"] == "tactic.applicable"
        )
        return {
            item["argumentValue"]: item
            for item in capability["argumentContexts"]
        }

    standard_rows = argument_contexts(standard)
    optional_rows = argument_contexts(optional)
    assert "project_bidding" not in standard_rows
    project = optional_rows["project_bidding"]
    assert project["decisionKinds"] == ["project"]
    assert project["requiredUtility"] == "awarded_point_margin"
    assert {"binding": "root_project_consequence_variables", "minimumCount": 2} in project["requiredProofBindings"]
    assert {(row["origin"], row["proofMode"], row["visibility"])
            for row in project["executionContexts"]} == {
        (origin, "exact", "complete_state")
        for origin in ("search", "exact_trace_prefix")
    }
    # The shared grammar includes the native producer; profile execution contexts
    # above keep the forced mandatory declaration out of usable controls.
    assert "project_bidding" in optional["authoringCatalog"]["tacticKinds"]
    for backend in (_backend("progressive_mining_ortools_v3.yaml"), _backend(
        "progressive_mining_ortools_v3.yaml",
        rules_profile="saudi_project_optional_unratified_v1.yaml",
    )):
        catalog = backend.catalog(
            origin="search", proof_mode="exact", visibility="complete_state"
        )
        for definition in catalog["definitions"]:
            if definition["kind"] in {"tactic.applicable", "tactic.verified"}:
                assert "project_bidding" in definition["argumentSchema"]["properties"]["kind"]["enum"]
    assert standard_rows["contract_rescue"]["requiredUtility"] == (
        "contract_success"
    )
    for tactic in (
        "kabout_creation",
        "kabout_preservation",
        "kabout_prevention",
    ):
        assert standard_rows[tactic]["requiredUtility"] == "kabout"
    for tactic in (
        "entry_unblocking",
        "partner_enablement",
        "trump_control",
        "small_card_big_gain",
        "large_card_sacrifice",
    ):
        assert standard_rows[tactic]["decisionKinds"] == ["play"]
        assert standard_rows[tactic]["requiredProofBindings"]


def test_v3_rejects_project_tactic_on_bid_decision() -> None:
    backend = _backend("progressive_mining_ortools_v3.yaml")
    recipe = _sample_recipe(
        backend, origin="search", decision_kind="bid", mode="exact",
        visibility="complete_state", claim="outcome_reachable",
        line_universe=None,
    )
    recipe["scenario"]["constraints"] = {
        "op": "all",
        "children": [{
            "predicateId": "PRED-PROJECT-BIDDING-CAUSALITY",
            "kind": "tactic.verified", "version": 1, "hardness": "hard",
            "arguments": {"kind": "project_bidding", "minimumScore": 100,
                          "minimumEvidence": 1},
        }],
    }
    assert any(
        issue["code"] == "tactic_decision_kind_unsupported"
        for issue in backend.validate(recipe)["issues"]
    )


def test_v3_admits_native_bid_and_escalation_state_only() -> None:
    backend = _backend("progressive_mining_ortools_v3.yaml")
    recipe = _sample_recipe(
        backend, origin="search", decision_kind="bid", mode="exact",
        visibility="complete_state", claim="outcome_reachable",
        line_universe=None,
    )
    recipe["scenario"]["constraints"] = {
        "predicateId": "PRED-BID-LEGAL-COUNT",
        "kind": "state.legal_action_count", "version": 1, "hardness": "hard",
        "arguments": {"minimum": 1, "maximum": 8},
    }
    assert not any(
        issue["code"] == "predicate_native_nonplay_state_unavailable"
        for issue in backend.validate(recipe)["issues"]
    )
    recipe["scenario"]["constraints"]["kind"] = "state.match_score"
    recipe["scenario"]["constraints"]["arguments"] = {
        "team": "north_south", "minimum": 0, "maximum": 152,
    }
    assert not any(
        issue["code"] == "predicate_native_nonplay_state_unavailable"
        for issue in backend.validate(recipe)["issues"]
    )
    recipe["scenario"]["constraints"]["kind"] = "state.current_raw_points"
    assert not any(
        issue["code"] == "predicate_native_nonplay_state_unavailable"
        for issue in backend.validate(recipe)["issues"]
    )
    recipe["scenario"]["constraints"]["kind"] = "state.current_project_points"
    assert not any(
        issue["code"] == "predicate_native_nonplay_state_unavailable"
        for issue in backend.validate(recipe)["issues"]
    )
    recipe["scenario"]["constraints"]["kind"] = "state.current_game_points"
    assert not any(
        issue["code"] == "predicate_native_nonplay_state_unavailable"
        for issue in backend.validate(recipe)["issues"]
    )
    escalation = _sample_recipe(
        backend, origin="search", decision_kind="escalation", mode="exact",
        visibility="complete_state", claim="outcome_reachable",
        line_universe=None,
    )
    escalation["scenario"]["constraints"] = {
        "predicateId": "PRED-ESCALATION-PHASE",
        "kind": "state.phase", "version": 1, "hardness": "hard",
        "arguments": {"state": "final_dealt"},
    }
    assert not any(
        issue["code"] == "predicate_native_nonplay_state_unavailable"
        for issue in backend.validate(escalation)["issues"]
    )


def test_optional_project_capabilities_are_published_after_full_service_evidence() -> None:
    backend = _backend(
        "progressive_mining_ortools_v3.yaml",
        rules_profile="saudi_project_optional_unratified_v1.yaml",
    )

    serialized = ScenarioExecutionCapabilitiesV1Response.model_validate(
        build_scenario_execution_capabilities_v1(backend)
    ).model_dump(mode="json")
    project = [
        item
        for item in serialized["combinations"]
        if item["decisionKind"] == "project"
    ]

    assert {
        (
            item["origin"],
            item["mode"],
            item["visibility"],
        )
        for item in project
    } == {
        ("search", "bounded", "complete_state"),
        ("search", "policy_conditioned", "complete_state"),
        ("exact_trace_prefix", "bounded", "complete_state"),
        ("exact_trace_prefix", "policy_conditioned", "complete_state"),
    }
    assert "project" not in serialized["decisionUnavailableReasons"]
    assert "project" in serialized["decisionSelectorSchemas"]
    manifest_contexts = [
        context
        for capability in serialized["authoringCatalog"][
            "predicateCapabilityManifest"
        ]["capabilities"]
        for context in capability["executionContexts"]
    ]
    assert any(
        context["decisionKind"] == "project"
        and context["proofMode"] in {"bounded", "policy_conditioned"}
        for context in manifest_contexts
    )
    predicate_contexts = {
        row["kind"]: {context["decisionKind"] for context in row["executionContexts"]}
        for row in serialized["authoringCatalog"]["predicateCapabilityManifest"]["capabilities"]
    }
    assert "project" in predicate_contexts["state.phase"]
    assert "project" in predicate_contexts["state.next_actor"]
    assert "project" in predicate_contexts["state.legal_action_count"]
    recipe = _sample_recipe(
        backend, origin="search", decision_kind="project", mode="bounded",
        visibility="complete_state", claim="outcome_reachable",
        line_universe=None,
    )
    recipe["scenario"]["constraints"] = {
        "predicateId": "PRED-PROJECT-COUNT",
        "kind": "state.legal_action_count", "version": 1,
        "hardness": "hard", "arguments": {"minimum": 1, "maximum": 64},
    }
    assert not any(
        issue["code"] == "predicate_native_nonplay_state_unavailable"
        for issue in backend.validate(recipe)["issues"]
    )
    recipe["scenario"]["constraints"]["kind"] = "state.match_score"
    recipe["scenario"]["constraints"]["arguments"] = {
        "team": "north_south", "minimum": 0, "maximum": 152,
    }
    assert not any(
        issue["code"] == "predicate_native_nonplay_state_unavailable"
        for issue in backend.validate(recipe)["issues"]
    )
    recipe["scenario"]["constraints"]["kind"] = "state.current_raw_points"
    assert not any(
        issue["code"] == "predicate_native_nonplay_state_unavailable"
        for issue in backend.validate(recipe)["issues"]
    )
    recipe["scenario"]["constraints"]["kind"] = "state.current_game_points"
    assert not any(
        issue["code"] == "predicate_native_nonplay_state_unavailable"
        for issue in backend.validate(recipe)["issues"]
    )


def test_v3_phase4_gaps_are_machine_readable_and_rejected() -> None:
    backend = _backend(
        "progressive_mining_ortools_v3.yaml",
        rules_profile="saudi_project_optional_unratified_v1.yaml",
    )
    capabilities = build_scenario_execution_capabilities_v1(backend)
    gaps = capabilities["phase4AcceptanceGaps"]
    gap_keys = {
        (
            item["origin"],
            item["decisionKind"],
            item["mode"],
            item["visibility"],
        )
        for item in gaps
    }

    assert gap_keys == (
        PHASE4_REQUIRED_COMBINATIONS_V1
        - PHASE4_FULL_SERVICE_PROVEN_COMBINATIONS_V1
    )
    assert all(
        item["acceptanceId"] == "phase4_strategic_proof_matrix_v1"
        and item["reason"] == "non_vacuous_full_service_fixture_missing"
        and item["requiredEvidence"]
        for item in gaps
    )
    for origin, decision_kind, mode, visibility in gap_keys:
        support = backend.execution_support(
            _sample_recipe(
                backend,
                origin=origin,
                decision_kind=decision_kind,
                mode=mode,
                visibility=visibility,
                claim="outcome_reachable",
                line_universe=None,
            )
        )
        assert support["executionReady"] is False
        assert "phase4_substantive_full_service_evidence_missing" in support[
            "reasons"
        ]


def test_scenario_configuration_projection_hides_non_executable_controls() -> None:
    capabilities = build_scenario_execution_capabilities_v1(_backend())
    original = {
        "schemaVersion": 1,
        "origins": ["search", "exact_trace_prefix"],
        "decisionCapabilities": [],
        "proofModes": ["exact", "bounded", "policy_conditioned"],
        "proofStrategyClasses": ["pure_deterministic", "frozen_policy"],
        "proofVisibilities": ["complete_state", "actor_information_set"],
        "proofClaims": ["winning_root_action_count"],
        "proofLineUniverses": [
            "all_legal",
            "optimal_strategy",
            "policy_conditioned",
        ],
        "unchanged": {"value": True},
        "sha256": "c" * 64,
    }

    projected = project_scenario_configuration_execution_v1(
        original,
        capabilities,
    )

    assert original["proofClaims"] == ["winning_root_action_count"]
    assert projected["unchanged"] == {"value": True}
    decisions = {item["kind"]: item for item in projected["decisionCapabilities"]}
    assert decisions["play"]["authoringSupported"] is True
    assert decisions["bid"]["authoringSupported"] is True
    assert decisions["escalation"]["authoringSupported"] is True
    assert decisions["project"] == {
        "kind": "project",
        "authoringSupported": False,
        "exactTraceNormalizationSupported": False,
        "searchGenerationSupported": False,
        "proofModes": [],
        "renderPreparationSupported": False,
        "unavailableReason": "project_declaration_mode_missing_or_mandatory",
    }
    assert "winning_root_action_count" in projected["proofClaims"]
    assert projected["sha256"] == canonical_sha256(
        {key: value for key, value in projected.items() if key != "sha256"}
    )


def test_actor_information_capability_refuses_unsupported_dimensions_explicitly() -> None:
    backend = _backend()

    def support(*, origin: str, decision: str, claim: str) -> dict:
        selector = (
            {
                "trickIndex": 0,
                "position": 0,
                "remainingCards": 8,
                "actingSeat": "south",
            }
            if decision == "play"
            else {"actionIndex": 0}
        )
        return backend.execution_support(
            {
                "scenario": {
                    "origin": {"kind": origin},
                    "decisionPoint": {"kind": decision, "selector": selector},
                },
                "proof": {
                    "mode": "exact",
                    "visibility": "actor_information_set",
                    "strategyClass": "pure_deterministic",
                    "claim": claim,
                    "targetTeam": "north_south",
                    "utility": "round_win",
                    "successThreshold": 1,
                    "budgetProfile": "actor_information_set_exact_v1",
                    **(
                        {"lineUniverse": "all_legal"}
                        if claim == "winning_continuation_line_count"
                        else {}
                    ),
                },
                "resources": {
                    "proofBudgetProfile": "actor_information_set_exact_v1"
                },
            }
        )

    admitted = support(
        origin="exact_trace_prefix",
        decision="play",
        claim="winning_root_action_count",
    )
    assert admitted["executionReady"] is True
    assert admitted["selectionConstraints"] == {
        "contractValues": ["sun", "hokum"]
    }
    assert support(
        origin="search", decision="play", claim="winning_root_action_count"
    )["executionReady"] is True
    nonplay = support(
        origin="exact_trace_prefix",
        decision="bid",
        claim="winning_root_action_count",
    )
    assert nonplay["executionReady"] is True
    assert "actor_information_set_continuation_count_unsupported" in support(
        origin="exact_trace_prefix",
        decision="play",
        claim="winning_continuation_line_count",
    )["reasons"]
