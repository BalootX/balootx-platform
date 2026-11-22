from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import pytest
import yaml

from baloot_platform.analysis.models import InformationState
from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat, Team
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.exact_state import ExactStateNormalizerV1
from baloot_platform.research.proof_service import (
    UnifiedScenarioProofServiceV1,
)
from baloot_platform.research.scenario_contracts import (
    ExactTracePrefixV1,
    ProofOutcomeV1,
    ResearchScenarioV1,
    ScenarioProofRequestV1,
    ScenarioPlayingQualityPolicyV1,
    SCENARIO_DECISION_KINDS,
    canonical_sha256,
    OPERATOR_PROOF_BOUNDARY_EVALUATORS,
    PROOF_BOUNDARY_EVALUATOR_DEFINITIONS,
    validate_proof_boundary_scope_v1,
)
from baloot_platform.research.scenario_registry import (
    ScenarioCompilerV1,
    ScenarioPredicateRegistryV1,
)
from baloot_platform.research.scenario_evidence import ScenarioRuntimeVerifierV1
from baloot_platform.research.scenario_search import (
    ScenarioGuidancePlanV1,
    ScenarioGuidedActorSafePolicyV1,
    ScenarioSearchExecutorV1,
)
from baloot_platform.research.recipe_v4 import ResearchRecipeV4
from baloot_platform.research.render_service import ResearchRenderPreparationServiceV1
from baloot_platform.research.service import (
    ResearchFoundryServiceV1,
    _load_owned_result_json_artifact_v1,
    _load_result_json_artifact_v1,
)
from baloot_platform.research.store import SQLiteResearchStoreV1
from baloot_platform.research.scenario_result import (
    public_scenario_result_v4,
    public_proof_outcome_v1,
    scenario_decision_audit_v1,
    validate_scenario_result_v4,
)
from baloot_platform.service import api
from baloot_platform.service.api import ExactTracePrefixV1Request
from baloot_platform.service.laffa_library import LaffaLibraryServiceV1
from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer
from baloot_platform.video.render_bundle import RenderBundleV1

from research_scenario_v4_helpers import (
    _ScenarioArtifactStore,
    _ScenarioRunStore,
    _exact_scenario_recipe,
    _late_prefix,
    _policy_proof_for,
    _predicate,
    _proof,
    _published_v4_configuration,
    _run_exact_scenario_recipe,
    _scenario,
)


@pytest.mark.parametrize("threshold_count", [0, 1, 2])
def test_actor_margin_copy_is_worst_case_not_a_game_win_count(threshold_count):
    from baloot_platform.video.scenario_adapter import _proof_reason_key

    key = "video.analysis.worst_case_awarded_margin"
    assert _proof_reason_key({"completionStatus": "exact_complete", "achievedMode": "exact",
                              "utility": "awarded_point_margin", "achievedVisibility": "actor_information_set",
                              "winningRootActionCount": threshold_count}, {key: "worst-case guarantee"}) == key


def test_video_proof_start_and_featured_play_are_distinct_and_derived():
    from baloot_platform.video.scenario_adapter import _proof_scope_position_values

    receipt = {"postAuction": {"projectPlay": {"coveredFromPlayIndex": 0}},
               "qualityExecution": {"exactTailProof": True, "exactPlayFromTrick": 7}}
    assert _proof_scope_position_values(receipt, 27) == {
        "proofStartTrick": 7, "proofStartPlay": 1, "featuredTrick": 7, "featuredPlay": 4}
    with pytest.raises(ValueError, match="boundary is invalid"):
        _proof_scope_position_values(receipt, 23)
    assert _proof_scope_position_values({}, 27) is None


@pytest.mark.parametrize("encoding", ["unknown", 1, True])
def test_contract_success_encoding_rejects_unknown_or_coercive_values(encoding):
    with pytest.raises(ValueError, match="utilityEncoding"):
        _proof(utility="contract_success", utilityEncoding=encoding)


def test_contract_success_signed_encoding_contract_api_and_quality_policy():
    request = _proof(utility="contract_success", utilityEncoding="signed_win_loss_v1")
    assert ScenarioProofRequestV1.build(request.to_dict()) == request
    assert request.sha256 != _proof(utility="contract_success").sha256
    wire = api.ScenarioProofRequestV1Request.model_validate(request.to_dict())
    assert wire.utilityEncoding == "signed_win_loss_v1"
    with pytest.raises(ValueError, match="utilityEncoding"):
        _proof(utility="awarded_point_margin", utilityEncoding="signed_win_loss_v1")
    with pytest.raises(ValueError, match="utilityEncoding"):
        api.ScenarioProofRequestV1Request.model_validate({**request.to_dict(), "utility": "round_win"})
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2, "mode": "exact",
        "visibility": "actor_information_set", "decisions": {kind: {
            "utility": "contract_success", "utilityEncoding": "signed_win_loss_v1",
            "budgetProfile": request.budget_profile} for kind in SCENARIO_DECISION_KINDS}})
    assert policy.request_for("play", "west").utility_encoding == "signed_win_loss_v1"
    assert api.PlayingQualityPolicyV1Request.model_validate(policy.to_dict()).decisions["play"].utilityEncoding == "signed_win_loss_v1"


@pytest.mark.parametrize("overrides", [
    {"horizon": {"unit": "tricks", "value": 1}},
    {"horizon": {"unit": "plies", "value": 2}},
    {"utility": "round_win"}, {"mode": "exact"},
])
def test_auction_hand_boundary_contract_rejects_unsupported_scope(overrides):
    raw = {"mode": "bounded", "horizon": {"unit": "plies", "value": 1},
           "boundaryEvaluator": "auction_hand_margin_v1"}
    raw.update(overrides)
    with pytest.raises(ValueError):
        _proof(**raw)


@pytest.mark.parametrize("kind", ["bid", "play", "project", "escalation"])
def test_auction_hand_boundary_is_bid_only_and_not_operator_visible(kind):
    request = _proof(mode="bounded", horizon={"unit": "decisions", "value": 1},
                     boundaryEvaluator="auction_hand_margin_v1")
    if kind == "bid":
        validate_proof_boundary_scope_v1(request, kind)
    else:
        with pytest.raises(ValueError, match="does not support"):
            validate_proof_boundary_scope_v1(request, kind)
    assert "auction_hand_margin_v1" not in OPERATOR_PROOF_BOUNDARY_EVALUATORS
    assert "current_raw_point_margin_v1" in OPERATOR_PROOF_BOUNDARY_EVALUATORS


def test_auction_hand_quality_policy_rejects_wrong_decision_kind():
    raw = {"schemaVersion": 1, "mode": "budgeted", "decisions": {
        kind: {"utility": "awarded_point_margin", "budgetProfile": "test",
               "horizon": {"unit": "plies", "value": 1}, "boundaryEvaluator": "current_raw_point_margin_v1"}
        for kind in SCENARIO_DECISION_KINDS}}
    raw["decisions"]["bid"]["boundaryEvaluator"] = "auction_hand_margin_v1"
    policy = ScenarioPlayingQualityPolicyV1.build(raw)
    assert policy.request_for("bid", "north").boundary_evaluator == "auction_hand_margin_v1"
    raw["decisions"]["play"]["boundaryEvaluator"] = "auction_hand_margin_v1"
    with pytest.raises(ValueError, match="does not support"):
        ScenarioPlayingQualityPolicyV1.build(raw)


def test_boundary_evaluator_registry_owns_horizon_and_evidence_classification():
    hand = PROOF_BOUNDARY_EVALUATOR_DEFINITIONS["auction_hand_margin_v1"]
    assert hand["kind"] == "heuristic"
    assert hand["operatorVisible"] is False
    assert hand["horizonUnits"] == ("plies", "decisions")
    assert hand["horizonValue"] == 1
    assert hand["utilities"] == ("awarded_point_margin",)
    for unit in hand["horizonUnits"]:
        request = _proof(mode="bounded", horizon={"unit": unit, "value": hand["horizonValue"]},
                         boundaryEvaluator="auction_hand_margin_v1", utility=hand["utilities"][0])
        validate_proof_boundary_scope_v1(request, "bid")
    assert PROOF_BOUNDARY_EVALUATOR_DEFINITIONS["current_raw_point_margin_v1"]["kind"] == "state_value"


def _bid_prefix() -> dict:
    return {
        "schemaVersion": 1,
        "deckOrder": list(range(32)),
        "dealer": "east",
        "roundNumber": 1,
        "matchScore": [0, 0],
        "auctionActions": [],
        "escalationActions": [],
        "projectDeclarations": [],
        "projectAssertions": [],
        "playActions": [],
        "stateAssertions": {"phase": "bidding"},
    }


def _escalation_prefix() -> dict:
    value = _bid_prefix()
    value["matchScore"] = [101, 0]
    value["auctionActions"] = [
        {"action": "pass"},
        {"action": "pass"},
        {"action": "sun"},
        {"action": "pass"},
        {"action": "sun"},
    ]
    value["stateAssertions"] = {"phase": "final_dealt"}
    return value


def _project_prefix() -> dict:
    value = _bid_prefix()
    value["auctionActions"] = [
        {"action": "pass"},
        {"action": "pass"},
        {"action": "sun"},
        {"action": "pass"},
        {"action": "sun"},
    ]
    value["stateAssertions"] = {"phase": "playing"}
    return value


def _awarded_partner_project_prefix() -> dict:
    """Sun buyer has the winning 100 while an undeclared partner Siri is awarded."""

    return {
        "schemaVersion": 1,
        "deckOrder": [
            10,
            13,
            23,
            5,
            9,
            0,
            7,
            22,
            31,
            6,
            27,
            20,
            1,
            12,
            4,
            18,
            25,
            24,
            28,
            8,
            11,
            14,
            16,
            17,
            3,
            19,
            29,
            26,
            15,
            2,
            21,
            30,
        ],
        "dealer": "east",
        "roundNumber": 1,
        "matchScore": [0, 0],
        "auctionActions": [{"actor": "north", "action": "sun"}],
        "escalationActions": [],
        "projectDeclarations": [
            {
                "owner": "north",
                "kind": "hundred_sequence",
                "value": 20,
                "highRank": 6,
                "declarationOrder": 1,
            }
        ],
        "projectAssertions": [],
        "playActions": [
            {"actor": "north", "card": "S8"},
            {"actor": "west", "card": "S7"},
            {"actor": "south", "card": "SA"},
            {"actor": "east", "card": "S9"},
        ],
        "stateAssertions": {"phase": "playing"},
    }


def test_v4_registry_is_closed_complete_localized_and_hash_stable(root: Path):
    registry = ScenarioPredicateRegistryV1()
    first = registry.catalog()
    second = ScenarioPredicateRegistryV1().catalog()
    assert first == second
    assert len(first["definitions"]) >= 36
    assert all(
        row["runtimeVerifier"]
        and row["messageArKey"]
        and row["messageEnKey"]
        and row["documentation"]
        and row["tests"]
        for row in first["definitions"]
    )
    product = yaml.safe_load(
        (root / "configs" / "product" / "studio_ortools.yaml").read_text(
            encoding="utf-8"
        )
    )
    workflow_copy = product["foundry_copy"]["groups"]["workflow"]
    expected_keys = {
        key
        for row in first["definitions"]
        for key in (row["messageArKey"], row["messageEnKey"])
    }
    for row in first["definitions"]:
        for property_schema in row["argumentSchema"]["properties"].values():
            expected_keys.update(
                {
                    property_schema["x-messageArKey"],
                    property_schema["x-messageEnKey"],
                }
            )
            enum_values = property_schema.get("enum", [])
            ar_enum_keys = property_schema.get("x-enumMessageArKeys", {})
            en_enum_keys = property_schema.get("x-enumMessageEnKeys", {})
            assert set(ar_enum_keys) == set(en_enum_keys)
            if enum_values and ar_enum_keys:
                assert {str(value) for value in enum_values} <= set(ar_enum_keys)
            expected_keys.update(ar_enum_keys.values())
            expected_keys.update(en_enum_keys.values())
    assert expected_keys <= set(workflow_copy)
    assert all(workflow_copy[key].strip() for key in expected_keys)
    with pytest.raises(ValueError, match="unsupported scenario predicate"):
        registry.require(
            _scenario(
                _predicate(
                    "PRED-CARD",
                    "deal.card_owner",
                    {"card": 0, "seat": "north", "handStage": "initial_five"},
                )
            )
            .constraints.predicates()[0]
            .__class__.build(_predicate("PRED-UNKNOWN", "made.up", {}))
        )


def test_scenario_configuration_exposes_frozen_default_policy_profiles():
    configuration = ResearchFoundryServiceV1._scenario_recipe_configuration()
    assert configuration["outputPolicy"] == {
        "kinds": ["laffa", "laffa_collection"],
        "resultCount": {"minimum": 1, "maximum": 1000, "default": 1},
        "renderModes": ["on_demand", "disabled"],
        "videoCount": {
            "minimum": 0,
            "maximum": 100,
            "default": 1,
            "maximumRelation": "output.resultCount",
        },
    }
    capabilities = {
        item["kind"]: item for item in configuration["decisionCapabilities"]
    }
    assert capabilities["play"] == {
        "kind": "play",
        "authoringSupported": True,
        "exactTraceNormalizationSupported": True,
        "searchGenerationSupported": True,
        "proofModes": ["exact", "bounded", "policy_conditioned"],
        "renderPreparationSupported": True,
        "unavailableReason": None,
    }
    assert capabilities["bid"] == {
        "kind": "bid",
        "authoringSupported": True,
        "exactTraceNormalizationSupported": True,
        "searchGenerationSupported": True,
        "proofModes": ["policy_conditioned"],
        "renderPreparationSupported": True,
        "unavailableReason": None,
    }
    assert capabilities["escalation"] == {
        "kind": "escalation",
        "authoringSupported": True,
        "exactTraceNormalizationSupported": True,
        "searchGenerationSupported": True,
        "proofModes": ["policy_conditioned"],
        "renderPreparationSupported": True,
        "unavailableReason": None,
    }
    assert capabilities["project"] == {
        "kind": "project",
        "authoringSupported": True,
        "exactTraceNormalizationSupported": True,
        "searchGenerationSupported": True,
        "proofModes": ["policy_conditioned"],
        "renderPreparationSupported": True,
        "unavailableReason": None,
    }
    profiles = configuration["defaultPolicyProfiles"]
    assert set(profiles) == {"north", "west", "south", "east"}
    for seed, seat in enumerate(("north", "west", "south", "east"), start=1):
        core = {
            "agentId": "heuristic_bot_v1",
            "version": "1.0.0",
            "seed": seed,
        }
        assert profiles[seat] == {
            **core,
            "profileHash": canonical_sha256(core),
        }


def test_generation_recipe_is_hash_bound_optional_and_requires_search_play_quality():
    configuration, _context = _published_v4_configuration()
    raw = {'schemaVersion': 4, 'titleAr': 'Typed family tag generation',
        'configuration': configuration,
        'scenario': _scenario(_predicate('PRED-GEN', 'auction.contract', {'contract': 'sun'})).to_dict(),
        'proof': _proof().to_dict(), 'status': 'published',
        'playingQuality': {'schemaVersion': 1, 'mode': 'exact', 'decisions': {
            kind: {'utility': 'awarded_point_margin', 'budgetProfile': 'late_game_exact_v1'}
            for kind in SCENARIO_DECISION_KINDS}}}
    legacy = ResearchRecipeV4.build(raw)
    assert 'generation' not in legacy.to_dict()
    assert ResearchRecipeV4.build(legacy.to_dict()).sha256 == legacy.sha256
    request = {'schemaVersion': 1, 'catalogHash': 'a' * 64, 'tagQuery': {'impact': ['round-flip']}}
    generated = ResearchRecipeV4.build({**raw, 'generation': request})
    assert generated.sha256 != legacy.sha256
    assert ResearchRecipeV4.build(generated.to_dict()).sha256 == generated.sha256
    with pytest.raises(ValueError, match='playingQuality'):
        ResearchRecipeV4.build({key: value for key, value in {**raw, 'generation': request}.items()
                                if key != 'playingQuality'})


def test_v4_output_and_render_counts_are_closed_hash_bound_and_cross_validated():
    configuration, _context = _published_v4_configuration()
    base = {
        "schemaVersion": 4,
        "titleAr": "بحث مخرجات متعددة موثق",
        "configuration": configuration,
        "scenario": _scenario(
            _predicate(
                "PRED-CONTRACT-OUTPUT",
                "auction.contract",
                {"contract": "sun"},
            )
        ).to_dict(),
        "proof": _proof().to_dict(),
        "status": "published",
    }
    default_recipe = ResearchRecipeV4.build(base)
    assert default_recipe.output == {"kind": "laffa", "resultCount": 1}
    assert default_recipe.render_policy == {"mode": "on_demand", "videoCount": 1}
    preserved = default_recipe.to_dict()
    assert "playingQuality" not in preserved
    quality = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 1, "mode": "exact", "decisions": {
            kind: {"utility": "awarded_point_margin", "budgetProfile": "late_game_exact_v1"}
            for kind in SCENARIO_DECISION_KINDS
        },
    })
    with_quality = ResearchRecipeV4.build({**base, "playingQuality": quality.to_dict()})
    assert with_quality.sha256 != default_recipe.sha256
    assert with_quality.proof.sha256 == default_recipe.proof.sha256
    assert ResearchRecipeV4.build(with_quality.to_dict()).to_dict() == with_quality.to_dict()
    assert ResearchRecipeV4.build(preserved).to_dict() == preserved
    create_payload = {**base, "configuration": {
        "schemaVersion": 1, "bundleRevisionId": configuration["bundleRevisionId"],
    }, "playingQuality": quality.to_dict()}
    request = api.ResearchRecipeV4CreateRequest.model_validate(create_payload)
    response = api.ResearchRecipeV4Response.model_validate(with_quality.to_dict())
    assert request.playingQuality.model_dump(exclude_none=True) == quality.to_dict()
    assert response.playingQuality.model_dump(exclude_none=True) == quality.to_dict()
    for model, payload in ((api.ResearchRecipeV4CreateRequest, create_payload),
                           (api.ResearchRecipeV4Response, with_quality.to_dict())):
        with pytest.raises(ValueError, match="playingQuality must be an object"):
            model.model_validate({**payload, "playingQuality": None})

    legacy = default_recipe.to_dict()
    legacy.pop("output")
    legacy["renderPolicy"] = {"mode": legacy["renderPolicy"]["mode"]}
    legacy.pop("sha256")
    legacy["sha256"] = canonical_sha256(legacy)
    assert ResearchRecipeV4.build(legacy).to_dict() == legacy

    collection_payload = copy.deepcopy(base)
    collection_payload["output"] = {"kind": "laffa_collection", "resultCount": 3}
    collection_payload["renderPolicy"] = {"mode": "on_demand", "videoCount": 2}
    collection_recipe = ResearchRecipeV4.build(collection_payload)
    assert collection_recipe.sha256 != default_recipe.sha256
    assert ResearchRecipeV4.build(collection_recipe.to_dict()).to_dict() == (
        collection_recipe.to_dict()
    )

    invalid_cases = (
        ("output", {"kind": "laffa", "resultCount": 2}, "single laffa"),
        (
            "output",
            {"kind": "laffa_collection", "resultCount": True},
            "output.resultCount must be an integer",
        ),
        (
            "output",
            {"kind": "laffa_collection", "resultCount": 1_001},
            "cannot exceed 1000",
        ),
        (
            "output",
            {"kind": "laffa_collection", "resultCount": 2, "unknown": 1},
            "contains unknown fields",
        ),
        (
            "renderPolicy",
            {"mode": "on_demand", "videoCount": 4},
            "cannot exceed output.resultCount",
        ),
        (
            "renderPolicy",
            {"mode": "disabled", "videoCount": 1},
            "disabled rendering",
        ),
        (
            "renderPolicy",
            {"mode": "on_demand", "videoCount": 1.0},
            "renderPolicy.videoCount must be an integer",
        ),
        (
            "renderPolicy",
            {"mode": "on_demand", "videoCount": 1, "unknown": 1},
            "contains unknown fields",
        ),
    )
    for section, value, message in invalid_cases:
        invalid = copy.deepcopy(collection_payload)
        invalid[section] = value
        with pytest.raises((TypeError, ValueError), match=message):
            ResearchRecipeV4.build(invalid)

    invalid_action = copy.deepcopy(base)
    invalid_action["proof"] = {
        **invalid_action["proof"],
        "claim": "selected_action_is_unique_winner",
        "selectedAction": 7,
    }
    invalid_action["proof"].pop("sha256", None)
    with pytest.raises(ValueError, match="selectedAction must be a string"):
        ResearchRecipeV4.build(invalid_action)

    invalid_action["proof"]["selectedAction"] = "pass"
    with pytest.raises(ValueError, match="selectedAction is invalid for play"):
        ResearchRecipeV4.build(invalid_action)

    exact = copy.deepcopy(base)
    exact["scenario"] = {
        **exact["scenario"],
        "origin": {
            "kind": "exact_trace_prefix",
            "exactTracePrefix": _late_prefix(_context.bundle.rules),
        },
    }
    exact["scenario"].pop("sha256", None)
    exact["output"] = {"kind": "laffa_collection", "resultCount": 2}
    exact["renderPolicy"] = {"mode": "on_demand", "videoCount": 1}
    with pytest.raises(ValueError, match="exact trace prefix can produce exactly one"):
        ResearchRecipeV4.build(exact)


@pytest.mark.parametrize("decision_kind", ["bid", "escalation", "project"])
def test_v4_compiler_fails_unsupported_non_play_proof_before_execution(
    decision_kind: str,
):
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _predicate(
                "PRED-CONTRACT",
                "auction.contract",
                {"contract": "sun"},
            ),
            "decisionPoint": {"kind": decision_kind, "selector": {}},
            "outcome": _predicate(
                "PRED-LEGAL-COUNT",
                "state.legal_action_count",
                {"minimum": 0, "maximum": 32},
            ),
        }
    )
    plan = ScenarioCompilerV1().compile(scenario, _proof())
    issue = next(
        item
        for item in plan.diagnostics
        if item.code == "decision_point_proof_unavailable"
    )
    assert plan.valid is False
    assert issue.predicate_ids == ()
    assert issue.field_paths == ("scenario.decisionPoint.kind",)
    assert issue.details["decisionKind"] == decision_kind


@pytest.mark.parametrize("decision_kind", ["bid", "escalation", "project"])
def test_v4_compiler_accepts_policy_conditioned_non_play_search(
    decision_kind: str,
):
    selector = (
        {"actingSeat": "west"}
        if decision_kind == "escalation"
        else {"owner": "north", "projectIndex": 0}
        if decision_kind == "project"
        else {"actionIndex": 0}
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _predicate(
                "PRED-DEALER", "auction.dealer", {"seat": "east"}
            ),
            "decisionPoint": {"kind": decision_kind, "selector": selector},
            "outcome": _predicate(
                "PRED-LEGAL-COUNT",
                "state.legal_action_count",
                {"minimum": 0, "maximum": 32},
            ),
        }
    )
    assert (
        ScenarioCompilerV1().compile(scenario, _policy_proof_for("north_south")).valid
        is True
    )


@pytest.mark.parametrize("decision_kind", ["bid", "escalation", "project"])
def test_v4_search_builds_policy_conditioned_non_play_candidates(
    decision_kind: str,
    rules,
):
    selector = {
        "bid": {"actionIndex": 1},
        "escalation": {"actionIndex": 1, "actingSeat": "north"},
        "project": {"owner": "north", "projectIndex": 0},
    }[decision_kind]
    constraints = [_predicate("PRED-DEALER", "auction.dealer", {"seat": "east"})]
    if decision_kind != "bid":
        constraints.extend(
            [
                _predicate("PRED-BUYER", "auction.buyer", {"seat": "north"}),
                _predicate("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
            ]
        )
    else:
        constraints.append(
            _predicate(
                "PRED-AUCTION-PREFIX",
                "auction.action_sequence",
                {"actions": [{"action": "pass"}], "complete": False},
            )
        )
    if decision_kind == "project":
        constraints.append(
            _predicate(
                "PRED-PROJECT",
                "project.exists",
                {
                    "team": "north_south",
                    "owner": "north",
                    "kind": "hundred_sequence",
                    "value": 20,
                    "state": "eligible",
                },
            )
        )
    if decision_kind == "escalation":
        constraints.extend(
            [
                _predicate(
                    "PRED-MATCH-SCORE",
                    "state.match_score",
                    {"team": "north_south", "minimum": 101, "maximum": 101},
                ),
                _predicate(
                    "PRED-ESCALATION-PREFIX",
                    "escalation.action_sequence",
                    {
                        "actions": [{"actor": "west", "action": "double"}],
                        "complete": False,
                    },
                ),
            ]
        )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {"op": "all", "children": constraints},
            "decisionPoint": {"kind": decision_kind, "selector": selector},
            "outcome": _predicate(
                "PRED-LEGAL-COUNT",
                "state.legal_action_count",
                {"minimum": 0, "maximum": 32},
            ),
        }
    )
    proof = _policy_proof_for("north_south")
    plan = ScenarioCompilerV1().compile(scenario, proof)
    assert plan.valid is True, [item.to_dict() for item in plan.diagnostics]
    recipe = ResearchRecipeV4(
        f"RCP-V4{decision_kind.upper()}SEARCH",
        "بحث قرار عام موثق",
        {"bundleRevisionId": "CBR-TEST"},
        scenario,
        proof,
        {},
        {
            "generationLane": "ortools",
            "generationPolicy": "natural_policy",
            "seedStart": 1,
            "seedCount": 1,
            "seedsPerShard": 1,
            "candidatesPerSeed": 1,
            "solverTimeoutSeconds": 10.0,
            "maximumConflicts": 250000,
            "analysisMode": "cascaded",
            "proofBudgetProfile": "bounded_v1",
        },
        {},
        {"mode": "on_demand"},
        "published",
        plan,
        "e" * 64,
    )

    candidate = ScenarioSearchExecutorV1().build_candidate(
        recipe,
        rules=rules,
        bundle_hash="b" * 64,
        seed=12001,
    )

    assert candidate.status == "satisfied", (
        [
            (item.predicate_id, item.kind, item.observed)
            for item in candidate.satisfaction.evidence
            if item.status != "satisfied"
        ]
        if candidate.satisfaction
        else candidate.construction_receipt
    )
    assert candidate.normalized is not None
    assert candidate.normalized.decision_kind == decision_kind
    assert candidate.proof is not None
    assert candidate.proof.payload["achievedMode"] == "policy_conditioned"
    assert candidate.construction_receipt["proofOutcomeHash"] == (
        candidate.proof.payload["semanticSha256"]
    )
    assert candidate.construction_receipt["policy"]["observerSafe"] is True


def test_v4_bid_policy_proof_evaluates_every_legal_public_action(rules):
    normalized = ExactStateNormalizerV1(rules).normalize(_bid_prefix())
    actor = normalized.next_actor
    assert actor is not None
    target_team = "north_south" if Seat.parse(actor).team.value == 0 else "west_east"
    proof_request = _policy_proof_for(target_team)
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": _bid_prefix(),
            },
            "constraints": _predicate(
                "PRED-DEALER",
                "auction.dealer",
                {"seat": "east"},
            ),
            "decisionPoint": {"kind": "bid", "selector": {"actionIndex": 0}},
            "outcome": _predicate(
                "PRED-WINNING-BIDS",
                "proof.winning_root_action_count",
                {"minimum": 1, "maximum": len(normalized.legal_actions)},
            ),
        }
    )
    plan = ScenarioCompilerV1().compile(scenario, proof_request)
    assert plan.valid is True
    proof = UnifiedScenarioProofServiceV1.from_default_authority().prove(
        normalized,
        proof_request,
        bundle_hash="b" * 64,
    )
    payload = proof.to_dict()
    assert payload["achievedMode"] == "policy_conditioned"
    assert payload["completionStatus"] == "policy_complete"
    assert payload["claimSatisfied"] is True
    assert {item["action"] for item in payload["rootActions"]} == {
        item["code"] for item in normalized.legal_actions
    }
    assert payload["winningRootActionCount"] == len(normalized.legal_actions)
    assert payload["continuationLines"] == {
        "complete": True,
        "exactCount": "1",
        "lineUniverse": "policy_conditioned",
    }
    assert payload["principalVariation"][0].startswith("bid:")
    assert payload["identities"]["policyHashes"] == {
        seat: profile["profileHash"] for seat, profile in proof_request.policies.items()
    }


def test_v4_bid_exact_input_runs_to_typed_result_and_observer_safe_render(
    root: Path,
    rules,
):
    normalized = ExactStateNormalizerV1(rules).normalize(_bid_prefix())
    actor = normalized.next_actor
    assert actor is not None
    target_team = "north_south" if Seat.parse(actor).team.value == 0 else "west_east"
    proof_request = _policy_proof_for(target_team)
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": _bid_prefix(),
            },
            "constraints": _predicate(
                "PRED-DEALER",
                "auction.dealer",
                {"seat": "east"},
            ),
            "decisionPoint": {"kind": "bid", "selector": {"actionIndex": 0}},
            "outcome": _predicate(
                "PRED-WINNING-BIDS",
                "proof.winning_root_action_count",
                {"minimum": 1, "maximum": len(normalized.legal_actions)},
            ),
        }
    )
    plan = ScenarioCompilerV1().compile(scenario, proof_request)
    configuration, context = _published_v4_configuration()
    assert context.bundle.rules.sha256 == rules.sha256
    recipe = ResearchRecipeV4(
        "RCP-V4BID001",
        "سيناريو قرار شراء موثق",
        configuration,
        scenario,
        proof_request,
        {},
        {
            "generationLane": "ortools",
            "generationPolicy": "natural_policy",
            "seedStart": 1,
            "seedCount": 1,
            "seedsPerShard": 1,
            "candidatesPerSeed": 1,
                    "solverTimeoutSeconds": 1.0,
            "maximumConflicts": 1,
            "analysisMode": "cascaded",
            "proofBudgetProfile": "bounded_v1",
        },
        {},
        {"mode": "on_demand"},
        "published",
        plan,
        "f" * 64,
    )
    run = {
        "runCode": "RUN-BID000000001",
        "runHash": "a" * 64,
        "recipeCode": recipe.code,
        "recipeHash": recipe.sha256,
        "status": "queued",
        "progress": 0.0,
        "events": [],
        "createdAt": "2026-08-12T00:00:00.000000+00:00",
        "configurationAuthority": configuration,
    }
    harness = SimpleNamespace(
        store=_ScenarioRunStore(run),
        artifact_store=_ScenarioArtifactStore(),
        _cancel_check=lambda _run_code: False,
    )
    ResearchFoundryServiceV1._execute_v4_exact_run(harness, run, recipe, context)
    result = harness.store.results[0]
    assert result["disposition"] == "qualified"
    assert result["scenarioProjection"]["decisionKind"] == "bid"
    assert result["proofOutcome"]["achievedMode"] == "policy_conditioned"

    profile = yaml.safe_load(
        (root / "configs" / "video" / "premium.yaml").read_text(encoding="utf-8")
    )
    research = SimpleNamespace(
        get_result=lambda _code: result,
        get_recipe=lambda _code: recipe.to_dict(),
    )
    render_service = ResearchRenderPreparationServiceV1(
        research=research,
        artifact_store=harness.artifact_store,
        publication_service=SimpleNamespace(),
        renderer=RemotionProfessionalRenderer(profile, repository_root=root),
        rules=rules,
        configuration_store=api.CONTAINER.configuration_store,
    )
    prepared = render_service.prepare(result["resultCode"])
    assert prepared["publicationEligible"] is True
    assert prepared["source"]["kind"] == "research_scenario_result"
    question = next(
        scene for scene in prepared["preview"]["scenes"] if scene["stage"] == "question"
    )
    prompt = next(
        layer["content"]
        for layer in question["layers"]
        if layer["id"] == "decision-prompt"
    )
    assert prompt["textKey"] == "video.bid.challenge_question"
    assert {item["code"] for item in prompt["legalChoices"]} == {
        item["action"] for item in result["proofOutcome"]["rootActions"]
    }
    encoded = json.dumps(prepared["preview"], sort_keys=True, separators=(",", ":"))
    exact_document = json.loads(
        harness.artifact_store.get(result["artifacts"]["exactState"]["uri"])
    )
    for seat, cards in exact_document["state"]["hands"].items():
        if seat == actor:
            continue
        assert all(f'"{card}"' not in encoded for card in cards)


def test_v4_escalation_policy_proof_evaluates_decline_and_legal_announcements(
    rules,
):
    prefix = _escalation_prefix()
    prefix_without_assertion = {**prefix, "stateAssertions": {}}
    playing = ExactStateNormalizerV1(rules).normalize(prefix_without_assertion)
    declarer = playing.round_state.declarer
    assert declarer is not None
    actor = next(seat for seat in Seat if seat.team != declarer.team)
    decision_point = {
        "kind": "escalation",
        "selector": {"actionIndex": 0, "actingSeat": actor.name.lower()},
    }
    normalized = ExactStateNormalizerV1(rules).normalize(
        prefix,
        decision_point=decision_point,
    )
    assert normalized.phase == "final_dealt"
    assert normalized.next_actor == actor.name.lower()
    assert normalized.legal_actions[0] == {
        "code": "decline",
        "action": "decline",
        "mode": None,
    }
    assert {item["code"] for item in normalized.legal_actions} == {
        "decline",
        "double:open",
    }
    proof_request = _policy_proof_for(
        "north_south" if actor.team.value == 0 else "west_east"
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": prefix,
            },
            "constraints": _predicate(
                "PRED-ESCALATION",
                "escalation.level",
                {"level": "normal"},
            ),
            "decisionPoint": decision_point,
            "outcome": _predicate(
                "PRED-WINNING-ESCALATIONS",
                "proof.winning_root_action_count",
                {"minimum": 1, "maximum": len(normalized.legal_actions)},
            ),
        }
    )
    assert ScenarioCompilerV1().compile(scenario, proof_request).valid is True
    proof = (
        UnifiedScenarioProofServiceV1.from_default_authority()
        .prove(
            normalized,
            proof_request,
            bundle_hash="b" * 64,
        )
        .to_dict()
    )
    assert proof["completionStatus"] == "policy_complete"
    assert {item["action"] for item in proof["rootActions"]} == {
        item["code"] for item in normalized.legal_actions
    }
    assert proof["principalVariation"][0].startswith("escalation:")


def test_v4_escalation_result_prepares_governed_observer_safe_render(
    root: Path,
    rules,
):
    prefix = _escalation_prefix()
    playing = ExactStateNormalizerV1(rules).normalize({**prefix, "stateAssertions": {}})
    declarer = playing.round_state.declarer
    assert declarer is not None
    actor = next(seat for seat in Seat if seat.team != declarer.team)
    decision_point = {
        "kind": "escalation",
        "selector": {"actionIndex": 0, "actingSeat": actor.name.lower()},
    }
    normalized = ExactStateNormalizerV1(rules).normalize(
        prefix,
        decision_point=decision_point,
    )
    proof_request = _policy_proof_for(
        "north_south" if actor.team.value == 0 else "west_east"
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": prefix,
            },
            "constraints": _predicate(
                "PRED-ESCALATION",
                "escalation.level",
                {"level": "normal"},
            ),
            "decisionPoint": decision_point,
            "outcome": _predicate(
                "PRED-WINNING-ESCALATIONS",
                "proof.winning_root_action_count",
                {"minimum": 1, "maximum": len(normalized.legal_actions)},
            ),
        }
    )
    recipe = _exact_scenario_recipe(
        "RCP-V4ESCAL01",
        "سيناريو قرار دبل موثق",
        scenario,
        proof_request,
    )
    result, harness = _run_exact_scenario_recipe(rules, recipe)
    assert result["disposition"] == "qualified"
    assert result["scenarioProjection"]["decisionKind"] == "escalation"
    prepared = _scenario_render_service(
        root,
        rules,
        result,
        harness,
        recipe,
    ).prepare(result["resultCode"])
    assert prepared["publicationEligible"] is True
    question = next(
        scene for scene in prepared["preview"]["scenes"] if scene["stage"] == "question"
    )
    prompt = next(
        layer["content"]
        for layer in question["layers"]
        if layer["id"] == "decision-prompt"
    )
    assert prompt["textKey"] == "video.escalation.challenge_question"
    assert {item["code"] for item in prompt["legalChoices"]} == {
        item["action"] for item in result["proofOutcome"]["rootActions"]
    }


def test_v4_project_declaration_is_engine_derived_proven_and_rendered(
    root: Path,
    rules,
):
    prefix = _project_prefix()
    decision_point = {
        "kind": "project",
        "selector": {"owner": "north", "projectIndex": 0},
    }
    normalized = ExactStateNormalizerV1(rules).normalize(
        prefix,
        decision_point=decision_point,
    )
    assert normalized.decision_kind == "project"
    assert normalized.next_actor == "north"
    assert len(normalized.legal_actions) == 1
    declaration = dict(normalized.legal_actions[0])
    assert declaration["action"] == "declare"
    assert declaration["owner"] == "north"
    proof_request = _policy_proof_for("north_south")
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": prefix,
            },
            "constraints": _predicate(
                "PRED-PROJECT",
                "project.exists",
                {
                    "team": "north_south",
                    "owner": "north",
                    "kind": declaration["kind"],
                    "value": declaration["value"],
                    "state": "eligible",
                },
            ),
            "decisionPoint": decision_point,
            "outcome": _predicate(
                "PRED-WINNING-PROJECT-LINES",
                "proof.winning_root_action_count",
                {"minimum": 1, "maximum": 1},
            ),
        }
    )
    assert ScenarioCompilerV1().compile(scenario, proof_request).valid is True
    recipe = _exact_scenario_recipe(
        "RCP-V4PROJECT1",
        "سيناريو إعلان مشروع موثق",
        scenario,
        proof_request,
    )
    result, harness = _run_exact_scenario_recipe(rules, recipe)
    assert result["disposition"] == "qualified"
    assert result["proofOutcome"]["rootActions"] == [
        {
            "action": declaration["code"],
            "value": result["proofOutcome"]["rootActions"][0]["value"],
            "satisfies": True,
            "completionStatus": "policy_complete",
        }
    ]
    assert "principalVariation" not in result["proofOutcome"]
    prepared = _scenario_render_service(
        root,
        rules,
        result,
        harness,
        recipe,
    ).prepare(result["resultCode"])
    question = next(
        scene for scene in prepared["preview"]["scenes"] if scene["stage"] == "question"
    )
    prompt = next(
        layer["content"]
        for layer in question["layers"]
        if layer["id"] == "decision-prompt"
    )
    assert prompt["textKey"] == "video.project.challenge_question"
    assert prompt["legalChoices"] == [declaration]


def test_v4_project_lifecycle_distinguishes_awarded_from_comparison_winner(rules):
    prefix = _awarded_partner_project_prefix()
    normalized = ExactStateNormalizerV1(rules).normalize(prefix)
    state = normalized.round_state

    assert state.contract is not None and state.contract.value == "sun"
    assert state.buyer == Seat.NORTH
    assert state.project_winning_team == Team.NORTH_SOUTH
    north_hundred = next(
        item
        for item in state.projects
        if item.owner == Seat.NORTH and item.kind == "hundred_sequence"
    )
    south_siri = next(
        item
        for item in state.projects
        if item.owner == Seat.SOUTH and item.kind == "siri"
    )
    assert north_hundred.declared is True
    assert north_hundred.scoring is True
    assert south_siri.declared is False
    assert south_siri.scoring is True
    assert (
        rules.project_domain.comparison_winner(
            state.projects,
            trump_suit=state.trump_suit,
        )
        is north_hundred
    )

    accepted = json.loads(json.dumps(prefix))
    accepted["projectAssertions"] = [
        {
            "owner": "north",
            "kind": "hundred_sequence",
            "value": 20,
            "state": "comparison_winner",
        },
        {
            "owner": "north",
            "kind": "hundred_sequence",
            "value": 20,
            "state": "awarded",
        },
        {
            "owner": "south",
            "kind": "siri",
            "value": 4,
            "state": "eligible",
        },
        {
            "owner": "south",
            "kind": "siri",
            "value": 4,
            "state": "awarded",
        },
    ]
    assert ExactStateNormalizerV1(rules).normalize(
        accepted
    ).project_assertions_verified == len(accepted["projectAssertions"])

    for invalid_state in ("declared", "comparison_winner"):
        rejected = json.loads(json.dumps(prefix))
        rejected["projectAssertions"] = [
            {
                "owner": "south",
                "kind": "siri",
                "value": 4,
                "state": invalid_state,
            }
        ]
        with pytest.raises(
            ValueError,
            match="project assertion does not match authoritative projects",
        ):
            ExactStateNormalizerV1(rules).normalize(rejected)

    predicates = [
        _predicate(
            "PRED-NORTH-COMPARISON-WINNER",
            "project.exists",
            {
                "team": "north_south",
                "owner": "north",
                "kind": "hundred_sequence",
                "value": 20,
                "state": "comparison_winner",
            },
        ),
        _predicate(
            "PRED-SOUTH-ELIGIBLE",
            "project.exists",
            {
                "team": "north_south",
                "owner": "south",
                "kind": "siri",
                "value": 4,
                "state": "eligible",
            },
        ),
        _predicate(
            "PRED-SOUTH-DECLARED",
            "project.exists",
            {
                "team": "north_south",
                "owner": "south",
                "kind": "siri",
                "value": 4,
                "state": "declared",
            },
        ),
        _predicate(
            "PRED-SOUTH-COMPARISON-WINNER",
            "project.exists",
            {
                "team": "north_south",
                "owner": "south",
                "kind": "siri",
                "value": 4,
                "state": "comparison_winner",
            },
        ),
        _predicate(
            "PRED-SOUTH-AWARDED",
            "project.exists",
            {
                "team": "north_south",
                "owner": "south",
                "kind": "siri",
                "value": 4,
                "state": "awarded",
            },
        ),
    ]
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {"op": "all", "children": predicates},
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-SUN",
                "auction.contract",
                {"contract": "sun"},
            ),
        }
    )
    evidence = {
        item.predicate_id: item.status
        for item in ScenarioRuntimeVerifierV1().verify(scenario, normalized).evidence
    }
    assert evidence == {
        "PRED-NORTH-COMPARISON-WINNER": "satisfied",
        "PRED-SOUTH-ELIGIBLE": "satisfied",
        "PRED-SOUTH-DECLARED": "failed",
        "PRED-SOUTH-COMPARISON-WINNER": "failed",
        "PRED-SOUTH-AWARDED": "satisfied",
        "PRED-SUN": "satisfied",
    }


def test_v4_project_lifecycle_rejects_declared_comparison_loser(root: Path, rules):
    example = json.loads(
        (root / "examples/research-scenario-v4-ace-sacrifice-seven.json").read_text(
            encoding="utf-8"
        )
    )
    prefix = example["exactTracePrefix"]
    normalized = ExactStateNormalizerV1(rules).normalize(prefix)
    state = normalized.round_state
    east_hundred = next(item for item in state.projects if item.owner == Seat.EAST)
    assert east_hundred.kind == "hundred_sequence"
    assert east_hundred.declared is True
    assert east_hundred.scoring is False

    accepted = json.loads(json.dumps(prefix))
    accepted["projectAssertions"] = [
        {
            "owner": "east",
            "kind": "hundred_sequence",
            "value": 20,
            "state": "declared",
        },
        {
            "owner": "north",
            "kind": "hundred_sequence",
            "value": 20,
            "state": "comparison_winner",
        },
    ]
    assert ExactStateNormalizerV1(rules).normalize(
        accepted
    ).project_assertions_verified == 2

    for invalid_state in ("comparison_winner", "awarded"):
        rejected = json.loads(json.dumps(prefix))
        rejected["projectAssertions"] = [
            {
                "owner": "east",
                "kind": "hundred_sequence",
                "value": 20,
                "state": invalid_state,
            }
        ]
        with pytest.raises(
            ValueError,
            match="project assertion does not match authoritative projects",
        ):
            ExactStateNormalizerV1(rules).normalize(rejected)

    predicates = [
        _predicate(
            "PRED-EAST-DECLARED",
            "project.exists",
            {
                "team": "west_east",
                "owner": "east",
                "kind": "hundred_sequence",
                "value": 20,
                "state": "declared",
            },
        ),
        _predicate(
            "PRED-EAST-COMPARISON-WINNER",
            "project.exists",
            {
                "team": "west_east",
                "owner": "east",
                "kind": "hundred_sequence",
                "value": 20,
                "state": "comparison_winner",
            },
        ),
        _predicate(
            "PRED-EAST-AWARDED",
            "project.exists",
            {
                "team": "west_east",
                "owner": "east",
                "kind": "hundred_sequence",
                "value": 20,
                "state": "awarded",
            },
        ),
    ]
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {"op": "all", "children": predicates},
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-SUN",
                "auction.contract",
                {"contract": "sun"},
            ),
        }
    )
    evidence = {
        item.predicate_id: item.status
        for item in ScenarioRuntimeVerifierV1().verify(scenario, normalized).evidence
    }
    assert evidence == {
        "PRED-EAST-DECLARED": "satisfied",
        "PRED-EAST-COMPARISON-WINNER": "failed",
        "PRED-EAST-AWARDED": "failed",
        "PRED-SUN": "satisfied",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("owner", "up", "owner is invalid"),
        ("team", "north", "team is invalid"),
        ("kind", 7, "kind is invalid"),
        ("value", 20.0, "value must be an integer"),
        ("highRank", True, "highRank must be an integer"),
        ("state", 1, "state must be a string"),
    ],
)
def test_v4_exact_project_assertions_reject_coercive_values(field, value, message):
    prefix = _awarded_partner_project_prefix()
    prefix["projectAssertions"] = [
        {
            "owner": "north",
            "kind": "hundred_sequence",
            "value": 20,
            "state": "awarded",
            field: value,
        }
    ]
    with pytest.raises((TypeError, ValueError), match=message):
        ExactTracePrefixV1.build(prefix)


@pytest.mark.parametrize(
    "state_assertions",
    [
        {"roundNumber": 1.0},
        {"roundNumber": True},
        {"matchScore": [0.0, 0.0]},
    ],
)
def test_v4_exact_state_assertions_use_strict_json_type_equality(
    rules,
    state_assertions,
):
    prefix = _awarded_partner_project_prefix()
    prefix["stateAssertions"] = state_assertions
    with pytest.raises(ValueError, match="exact-state assertion mismatch"):
        ExactStateNormalizerV1(rules).normalize(prefix)


def test_v4_exact_trace_rejects_boolean_card_identity():
    prefix = _awarded_partner_project_prefix()
    prefix["playActions"][0]["card"] = True
    with pytest.raises(
        TypeError,
        match="play action card must be an integer or string",
    ):
        ExactTracePrefixV1.build(prefix)
    with pytest.raises(ValueError, match="playActions.0.card"):
        ExactTracePrefixV1Request.model_validate(prefix)


@pytest.mark.parametrize("declared", [None, "", 0, False])
def test_v4_exact_trace_rejects_falsey_or_non_string_checksum(declared):
    prefix = _awarded_partner_project_prefix()
    prefix["sha256"] = declared
    with pytest.raises(
        ValueError,
        match="sha256 must be a lowercase SHA-256 string",
    ):
        ExactTracePrefixV1.build(prefix)


def test_v4_exact_trace_api_rejects_explicit_null_checksum():
    prefix = _awarded_partner_project_prefix()
    prefix["sha256"] = None
    with pytest.raises(
        ValueError,
        match="sha256 must be a lowercase SHA-256 string",
    ):
        ExactTracePrefixV1Request.model_validate(prefix)


def test_v4_exact_trace_accepts_only_its_matching_checksum():
    prefix = _awarded_partner_project_prefix()
    expected = ExactTracePrefixV1.build(prefix).sha256
    prefix["sha256"] = expected
    assert ExactTracePrefixV1.build(prefix).sha256 == expected
    prefix["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prefix checksum mismatch"):
        ExactTracePrefixV1.build(prefix)


def test_exact_trace_wire_actions_are_closed_and_json_native(rules):
    prefix = _late_prefix(rules)
    validated = ExactTracePrefixV1Request.model_validate(prefix)
    assert validated.model_dump(mode="json")["matchScore"] == [0, 0]
    assert (
        validated.model_dump(mode="json")["projectDeclarations"]
        == prefix["projectDeclarations"]
    )
    invalid = _late_prefix(rules)
    invalid["playActions"][0]["ignoredOracleHint"] = "SA"
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ExactTracePrefixV1Request.model_validate(invalid)


def test_v4_compiler_reports_card_ownership_contradiction_by_predicate_id():
    scenario = _scenario(
        _predicate(
            "PRED-CARD-N",
            "deal.card_owner",
            {"card": 0, "seat": "north", "handStage": "initial_five"},
        ),
        _predicate(
            "PRED-CARD-S",
            "deal.card_owner",
            {"card": 0, "seat": "south", "handStage": "initial_five"},
        ),
    )
    plan = ScenarioCompilerV1().compile(scenario, _proof())
    assert plan.valid is False
    issue = next(
        item for item in plan.diagnostics if item.code == "card_required_by_two_seats"
    )
    assert issue.diagnostic_class == "contradictory_constraints"
    assert issue.predicate_ids == ("PRED-CARD-N", "PRED-CARD-S")


def test_v4_compiler_reports_singleton_and_project_identity_contradictions():
    scenario = _scenario(
        _predicate("PRED-BUYER-N", "auction.buyer", {"seat": "north"}),
        _predicate("PRED-BUYER-S", "auction.buyer", {"seat": "south"}),
        _predicate(
            "PRED-PROJECT",
            "project.exists",
            {
                "team": "west_east",
                "owner": "south",
                "kind": "hundred_sequence",
                "value": 20,
                "state": "eligible",
            },
        ),
    )
    plan = ScenarioCompilerV1().compile(scenario, _proof())
    assert plan.valid is False
    issues = {item.code: item for item in plan.diagnostics}
    assert issues["conflicting_singleton_predicates"].predicate_ids == (
        "PRED-BUYER-N",
        "PRED-BUYER-S",
    )
    assert issues["project_owner_team_mismatch"].predicate_ids == ("PRED-PROJECT",)


def test_v4_compiler_rejects_play_selector_outside_requested_trick():
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _predicate(
                "PRED-TRICK-SIX",
                "play.trick_index",
                {"minimum": 5, "maximum": 5},
            ),
            "decisionPoint": {
                "kind": "play",
                "selector": {"remainingCards": 8, "actingSeat": "west"},
            },
            "outcome": _predicate(
                "PRED-ROOT-COUNT",
                "proof.winning_root_action_count",
                {"minimum": 1, "maximum": 1},
            ),
        }
    )
    plan = ScenarioCompilerV1().compile(scenario, _proof(targetTeam="west_east"))
    issue = next(
        item
        for item in plan.diagnostics
        if item.code == "play_decision_remaining_cards_conflicts_with_trick_index"
    )
    assert plan.valid is False
    assert issue.predicate_ids == ("PRED-TRICK-SIX",)
    assert issue.details["allowedRemainingCards"] == [9, 12]


def test_v4_compiler_rejects_play_selector_actor_mismatch():
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-TRICK-SIX",
                        "play.trick_index",
                        {"minimum": 5, "maximum": 5},
                    ),
                    _predicate(
                        "PRED-ACTOR",
                        "play.current_player",
                        {"seat": "west"},
                    ),
                ],
            },
            "decisionPoint": {
                "kind": "play",
                "selector": {"remainingCards": 9, "actingSeat": "south"},
            },
            "outcome": _predicate(
                "PRED-ROOT-COUNT",
                "proof.winning_root_action_count",
                {"minimum": 1, "maximum": 1},
            ),
        }
    )
    plan = ScenarioCompilerV1().compile(scenario, _proof(targetTeam="west_east"))
    issue = next(
        item
        for item in plan.diagnostics
        if item.code == "play_decision_actor_conflicts_with_current_player"
    )
    assert plan.valid is False
    assert issue.predicate_ids == ("PRED-ACTOR",)


def test_v4_compiler_rejects_future_trick_as_already_completed():
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _predicate(
                "PRED-FUTURE-COMPLETED",
                "play.completed_trick_winner",
                {"trickIndex": 5, "seat": "south"},
            ),
            "decisionPoint": {
                "kind": "play",
                "selector": {
                    "trickIndex": 5,
                    "position": 1,
                    "remainingCards": 3,
                    "actingSeat": "south",
                },
            },
            "outcome": _predicate(
                "PRED-ROOT-COUNT",
                "proof.winning_root_action_count",
                {"minimum": 1, "maximum": 1},
            ),
        }
    )

    plan = ScenarioCompilerV1().compile(scenario, _proof())

    issue = next(
        item
        for item in plan.diagnostics
        if item.code == "completed_trick_is_not_before_play_decision"
    )
    assert plan.valid is False
    assert issue.predicate_ids == ("PRED-FUTURE-COMPLETED",)
    assert issue.details == {"decisionTrickIndex": 5, "completedTrickIndex": 5}


def _scenario_render_service(root: Path, rules, result, harness, recipe):
    profile = yaml.safe_load(
        (root / "configs" / "video" / "premium.yaml").read_text(encoding="utf-8")
    )
    return ResearchRenderPreparationServiceV1(
        research=SimpleNamespace(
            get_result=lambda _code: result,
            get_recipe=lambda _code: recipe.to_dict(),
        ),
        artifact_store=harness.artifact_store,
        publication_service=SimpleNamespace(),
        renderer=RemotionProfessionalRenderer(profile, repository_root=root),
        rules=rules,
        configuration_store=api.CONTAINER.configuration_store,
    )


def _execute_exact_v4_result(rules):
    prefix = _late_prefix(rules)
    normalized = ExactStateNormalizerV1(rules).normalize(prefix)
    request = _proof(
        targetTeam=(
            "north_south"
            if normalized.round_state.current_player.team.value == 0
            else "west_east"
        )
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": prefix,
            },
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-ACTOR",
                        "play.current_player",
                        {"seat": normalized.next_actor},
                    )
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-COUNT",
                "proof.winning_root_action_count",
                {"minimum": 0, "maximum": len(normalized.legal_actions)},
            ),
        }
    )
    plan = ScenarioCompilerV1().compile(scenario, request)
    configuration, context = _published_v4_configuration()
    assert context.bundle.rules.sha256 == rules.sha256
    recipe = ResearchRecipeV4(
        "RCP-V4EXACT1",
        "سيناريو دقيق قابل للإثبات",
        configuration,
        scenario,
        request,
        {},
        {
            "generationLane": "ortools",
            "generationPolicy": "natural_policy",
            "seedStart": 1,
            "seedCount": 1,
            "seedsPerShard": 1,
            "candidatesPerSeed": 1,
            "solverTimeoutSeconds": 1.0,
            "maximumConflicts": 1,
            "analysisMode": "cascaded",
            "proofBudgetProfile": "late_game_exact_v1",
        },
        {},
        {"mode": "on_demand"},
        "published",
        plan,
        "f" * 64,
    )
    run = {
        "runCode": "RUN-123456789ABC",
        "runHash": "a" * 64,
        "recipeCode": recipe.code,
        "recipeHash": recipe.sha256,
        "status": "queued",
        "progress": 0.0,
        "events": [],
        "createdAt": "2026-08-12T00:00:00.000000+00:00",
        "configurationAuthority": configuration,
    }
    harness = SimpleNamespace(
        store=_ScenarioRunStore(run),
        artifact_store=_ScenarioArtifactStore(),
        _cancel_check=lambda _run_code: False,
    )
    ResearchFoundryServiceV1._execute_v4_exact_run(
        harness,
        run,
        recipe,
        context,
    )
    assert harness.store.run["status"] == "completed"
    assert harness.store.run["resultCount"] == 1
    result = harness.store.results[0]
    assert result["schemaVersion"] == 2
    assert result["disposition"] == "qualified"
    assert result["scenarioSatisfaction"]["status"] == "satisfied"
    assert result["proofOutcome"]["completionStatus"] == "exact_complete"
    assert result["scenarioProjection"]["origin"] == "exact_trace_prefix"
    assert set(result["artifacts"]) == {
        "exactState",
        "scenarioSatisfaction",
        "constraintEvidence",
        "proofOutcome",
    }
    assert all(
        reference["uri"] in harness.artifact_store.values
        for reference in result["artifacts"].values()
    )
    assert (
        validate_scenario_result_v4(
            result,
            artifact_store=harness.artifact_store,
        )
        == result
    )
    decisions = scenario_decision_audit_v1(result)
    assert len(decisions) == 1
    assert decisions[0]["decisionId"] == result["resultCode"]
    assert len(decisions[0]["legalActions"]) == len(
        result["proofOutcome"]["rootActions"]
    )
    assert decisions[0]["provenance"]["observerSafe"] is True
    proof_uri = result["artifacts"]["proofOutcome"]["uri"]
    original_proof = harness.artifact_store.values[proof_uri]
    harness.artifact_store.values[proof_uri] += b"\n"
    with pytest.raises(ValueError, match="artifact integrity mismatch"):
        validate_scenario_result_v4(
            result,
            artifact_store=harness.artifact_store,
        )
    harness.artifact_store.values[proof_uri] = original_proof
    return result, harness, recipe, normalized


def test_exact_v4_native_execution_retains_typed_artifacts_and_projection(rules):
    result, harness, _recipe, normalized = _execute_exact_v4_result(rules)
    service = SimpleNamespace(
        get_result=lambda code: (
            validate_scenario_result_v4(
                result,
                artifact_store=harness.artifact_store,
            )
            if code == result["resultCode"]
            else None
        ),
        artifact_store=harness.artifact_store,
    )
    view = ResearchFoundryServiceV1.get_exact_state(
        service,
        result["resultCode"],
        observer_seat=normalized.next_actor,
    )
    assert "exactStateId" not in view
    assert view["decisionReference"] == {
        "resultCode": result["resultCode"],
        "decisionOrdinal": 1,
    }
    assert view["publicProjectionHash"] == result["scenarioProjection"][
        "publicProjectionHash"
    ]
    assert view["decisionKind"] == "play"
    assert view["observerSeat"] == normalized.next_actor
    assert view["legalActions"] == normalized.to_dict()["legalActions"]
    exact_document = json.loads(
        harness.artifact_store.get(result["artifacts"]["exactState"]["uri"])
    )
    for seat, hand in view["publicState"]["hands"].items():
        if seat == normalized.next_actor:
            assert isinstance(hand, list)
        else:
            assert hand == {"count": len(exact_document["state"]["hands"][seat])}
    other = next(
        seat
        for seat in ("north", "west", "south", "east")
        if seat != normalized.next_actor
    )
    with pytest.raises(ValueError, match="acting observer"):
        ResearchFoundryServiceV1.get_exact_state(
            service,
            result["resultCode"],
            observer_seat=other,
        )
    library = LaffaLibraryServiceV1(
        learning_store=SimpleNamespace(
            list_games=lambda **_kwargs: [],
            list_video_records=lambda **_kwargs: [],
        ),
        research=SimpleNamespace(
            list_results=lambda **_kwargs: {
                "items": [result],
                "nextOffset": None,
            },
            artifact_store=harness.artifact_store,
        ),
        foundry_catalogs=None,
        render_batches=None,
    )
    assert library.get(result["resultCode"])["replayValid"] is True


def test_exact_v4_play_result_prepares_observer_safe_governed_render(
    root: Path,
    rules,
):
    result, harness, recipe, normalized = _execute_exact_v4_result(rules)
    projected_result = copy.deepcopy(result)
    admission = {"policyHash": "a" * 64, "evidenceHash": "b" * 64, "passed": True, "score": 134}
    projected_result["qualification"]["interestAdmission"] = {**admission, "privateBaselineAction": "CA"}
    assert public_scenario_result_v4(projected_result)["qualification"]["interestAdmission"] == admission
    profile = yaml.safe_load(
        (root / "configs" / "video" / "premium.yaml").read_text(encoding="utf-8")
    )
    profile_hash = canonical_sha256(profile)
    service = _scenario_render_service(root, rules, result, harness, recipe)

    prepared = service.prepare(result["resultCode"])
    assert prepared["exportPreset"] == "hooked_flagship"
    assert prepared["basename"].endswith("-scenario-play-full")
    assert prepared["publicationEligible"] is True
    assert prepared["renderPurpose"] == "content_candidate"
    assert prepared["qualificationFailures"] == []
    assert prepared["publicationBlockers"] == []
    assert prepared["source"]["kind"] == "research_scenario_result"
    assert prepared["source"]["sourceSchemaVersion"] == 3
    assert prepared["source"]["decisionReference"] == {
        "resultCode": result["resultCode"],
        "decisionOrdinal": 1,
    }
    assert prepared["source"]["proofReference"] == result["resultCode"]
    assert prepared["source"]["publicProjectionHash"]
    assert all(
        field not in prepared["source"]
        for field in (
            "decisionId",
            "exactStateId",
            "traceHash",
            "resultHash",
            "scenarioHash",
            "proofHash",
            "constraintEvidenceHash",
            "exactStateHash",
        )
    )
    assert prepared["source"]["qrActionable"] is False
    assert prepared["locator"]["universalLink"] == f"/c/{result['resultCode']}"
    assert [
        segment["stage"] for segment in prepared["preview"]["timeline"]["segments"]
    ] == [
        "hook",
        "rewind",
        "replay",
        "question",
        "reveal",
        "continuation",
        "outcome",
    ]

    decision_clip = service.prepare(result["resultCode"], decision_ordinal=1)
    assert decision_clip["exportPreset"] == "decision_clip"
    assert decision_clip["basename"].endswith("-scenario-play-decision-01")

    payload = harness.artifact_store.get(prepared["bundleArtifact"]["uri"])
    bundle = RenderBundleV1(json.loads(payload.decode("utf-8")))
    assert bundle.verify()
    encoded_bundle = payload.decode("utf-8")
    assert normalized.exact_state_id not in encoded_bundle
    assert normalized.state_hash not in encoded_bundle
    assert result["resultHash"] not in encoded_bundle
    assert result["artifacts"]["proofOutcome"]["sha256"] not in encoded_bundle
    assert prepared["authorityReceipt"]["exactStateId"] == normalized.exact_state_id
    assert prepared["proofArtifactSha256"] == result["artifacts"][
        "proofOutcome"
    ]["sha256"]
    scenes = bundle.value["scenes"]
    assert [scene["stage"] for scene in scenes] == [
        "context",
        "question",
        "reveal",
        "continuation",
        "outcome",
    ]
    actor = normalized.next_actor
    exact_document = json.loads(
        harness.artifact_store.get(result["artifacts"]["exactState"]["uri"])
    )
    private_proof = json.loads(
        harness.artifact_store.get(result["artifacts"]["proofOutcome"]["uri"])
    )
    actor_cards = exact_document["state"]["hands"][actor]
    for scene in scenes:
        observer_layer = next(
            layer for layer in scene["layers"] if layer["id"] == "observer-hand"
        )
        assert observer_layer["content"] == {"seat": actor, "cards": actor_cards}
        scene_json = json.dumps(scene, sort_keys=True, separators=(",", ":"))
        for seat, cards in exact_document["state"]["hands"].items():
            if seat == actor:
                continue
            assert all(f'"{card}"' not in scene_json for card in cards)
    context = next(scene for scene in scenes if scene["stage"] == "continuation")
    public_lifecycle = next(
        layer["content"]
        for layer in context["layers"]
        if layer["id"] == "public-lifecycle"
    )
    observer_lifecycle = next(
        layer["content"]
        for layer in context["layers"]
        if layer["id"] == "observer-lifecycle"
    )
    pause = public_lifecycle["pauseBeforePlayIndex"]
    initial_actor_cards = [
        *actor_cards,
        *[
            item["card"]
            for item in public_lifecycle["plays"][:pause]
            if item["seat"] == actor
        ],
    ]
    initial_actor_cards.sort(key=lambda card: card["id"])
    assert observer_lifecycle["initialHandCards"] == initial_actor_cards
    assert len(initial_actor_cards) == 8
    assert [
        item["card"]["code"] for item in public_lifecycle["plays"][pause:]
    ] == private_proof["principalVariation"]
    reveal = next(scene for scene in scenes if scene["stage"] == "reveal")
    analysis = next(
        layer["content"] for layer in reveal["layers"] if layer["id"] == "analysis"
    )
    assert analysis["completionStatus"] == "exact_complete"
    assert analysis["reasonKey"] in {
        "video.analysis.no_salvage",
        "video.analysis.one_winning_action",
        "video.analysis.multiple_winning_actions",
    }

    other = next(seat for seat in ("north", "west", "south", "east") if seat != actor)
    with pytest.raises(ValueError, match="acting observer"):
        service.prepare(result["resultCode"], observer_seat=other)

    proof_reference = result["artifacts"]["proofOutcome"]
    proof_payload = harness.artifact_store.values[proof_reference["uri"]]
    harness.artifact_store.values[proof_reference["uri"]] = proof_payload + b"\n"
    with pytest.raises(ValueError, match="artifact integrity mismatch"):
        service.prepare(result["resultCode"])
    harness.artifact_store.values[proof_reference["uri"]] = proof_payload

    incomplete = {**result["proofOutcome"], "completionStatus": "proof_incomplete"}
    eligible, _, _, blockers, policy = service._scenario_publication_governance(
        result,
        proof=incomplete,
        exact_state_id=normalized.exact_state_id,
        video_profile_hash=profile_hash,
    )
    assert eligible is False
    assert blockers == ["scenario_proof_incomplete"]
    assert policy["decisionSelection"]["gatePassed"] is False

    counterproof = {
        **result["proofOutcome"],
        "completionStatus": "proof_failed",
        "claimSatisfied": False,
    }
    eligible, _, _, blockers, policy = service._scenario_publication_governance(
        result,
        proof=counterproof,
        exact_state_id=normalized.exact_state_id,
        video_profile_hash=profile_hash,
    )
    assert eligible is False
    assert blockers == ["scenario_claim_not_satisfied"]
    assert policy["decisionSelection"]["gatePassed"] is False


def test_v4_render_resolves_frozen_bundle_and_rejects_substitution():
    container = api.CONTAINER
    revision_id = container.configuration.resolve()["bundle"]["revisionId"]
    authority, context = container.research.execution_context_factory.freeze_selector(
        bundle_revision_id=revision_id,
        analysis_mode="cascaded",
    )
    configuration = authority.to_dict()
    configuration_hash = canonical_sha256(configuration)
    recipe_code = "RCP-V4RENDER1"
    recipe_hash = "a" * 64
    result = {
        "recipeCode": recipe_code,
        "proofOutcome": {
            "identities": {
                "rulesHash": context.bundle.rules.sha256,
                "bundleHash": context.bundle.sha256,
            }
        },
        "provenance": {
            "recipeHash": recipe_hash,
            "configurationAuthorityHash": configuration_hash,
            "resultIdentity": {
                "recipeCode": recipe_code,
                "recipeHash": recipe_hash,
                "configurationAuthorityHash": configuration_hash,
            },
        },
    }
    recipe = {
        "code": recipe_code,
        "sha256": recipe_hash,
        "configuration": configuration,
    }
    active_payload = json.loads(json.dumps(context.bundle.rules.data))
    active_payload["profile"]["version"] = "99.0.0"
    active_rules = RuleBook(active_payload)
    service = ResearchRenderPreparationServiceV1(
        research=SimpleNamespace(get_recipe=lambda _code: recipe),
        artifact_store=container.research.artifact_store,
        publication_service=SimpleNamespace(),
        renderer=SimpleNamespace(),
        rules=active_rules,
        configuration_store=container.configuration_store,
    )

    frozen_rules, video, video_hash, bundle_hash, quality_settings = (
        service._scenario_render_authority(
            result,
            exact_state={"rulesHash": context.bundle.rules.sha256},
        )
    )
    assert frozen_rules.sha256 == context.bundle.rules.sha256
    assert frozen_rules.sha256 != active_rules.sha256
    assert video == context.bundle.video
    assert video_hash == context.bundle.components["video"].sha256
    assert bundle_hash == context.bundle.sha256
    frozen_challenge = container.configuration_store.get_profile(
        authority.component_revisions["challenge"]["revisionId"]
    )
    assert quality_settings == frozen_challenge.payload.get("scenarioResearchV4", {})

    minimal_configuration = {
        "schemaVersion": 1,
        "bundleRevisionId": authority.bundle_revision_id,
    }
    minimal_hash = canonical_sha256(minimal_configuration)
    minimal_recipe = {
        **recipe,
        "configuration": minimal_configuration,
    }
    minimal_result = copy.deepcopy(result)
    minimal_result["provenance"]["configurationAuthorityHash"] = minimal_hash
    minimal_result["provenance"]["resultIdentity"][
        "configurationAuthorityHash"
    ] = minimal_hash
    service.research = SimpleNamespace(get_recipe=lambda _code: minimal_recipe)
    with pytest.raises(
        ValueError,
        match="research configuration bundle identity is incomplete",
    ):
        service._scenario_render_authority(
            minimal_result,
            exact_state={"rulesHash": context.bundle.rules.sha256},
        )
    service.research = SimpleNamespace(get_recipe=lambda _code: recipe)

    for family in ("rules", "video"):
        revision_id = authority.component_revisions[family]["revisionId"]
        original = container.configuration_store.get_profile(revision_id)
        assert original is not None
        substituted_payload = json.loads(json.dumps(original.payload))
        substituted_payload["profile"]["version"] = "99.0.0"

        def substituted_profile(
            selected,
            frozen_revision_id=revision_id,
            frozen_original=original,
            frozen_payload=substituted_payload,
        ):
            if selected != frozen_revision_id:
                return container.configuration_store.get_profile(selected)
            return SimpleNamespace(
                revision_id=frozen_original.revision_id,
                family=frozen_original.family,
                payload=frozen_payload,
                payload_sha256=canonical_sha256(frozen_payload),
            )

        service.configuration_store = SimpleNamespace(
            get_bundle=container.configuration_store.get_bundle,
            get_profile=substituted_profile,
        )
        with pytest.raises(ValueError, match="frozen profile substitution detected"):
            service._scenario_render_authority(
                result,
                exact_state={"rulesHash": context.bundle.rules.sha256},
            )


def test_search_v4_constructs_replays_and_proves_one_actor_safe_candidate(rules):
    request = _proof(targetTeam="north_south")
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-CARD",
                        "deal.card_owner",
                        {"card": 0, "seat": "north", "handStage": "initial_five"},
                    ),
                    _predicate("PRED-BUYER", "auction.buyer", {"seat": "south"}),
                    _predicate(
                        "PRED-CONTRACT",
                        "auction.contract",
                        {"contract": "sun"},
                    ),
                    _predicate(
                        "PRED-PROJECT",
                        "project.exists",
                        {
                            "team": "north_south",
                            "owner": "south",
                            "kind": "hundred_sequence",
                            "value": 20,
                            "state": "eligible",
                        },
                    ),
                    _predicate(
                        "PRED-PLAY-PREFIX",
                        "play.card_prefix",
                        {"cards": [0]},
                    ),
                    _predicate(
                        "PRED-PLAY-ACTION-PREFIX",
                        "play.action_prefix",
                        {"actions": [{"card": 0, "ekka": False}]},
                    ),
                ],
            },
            "decisionPoint": {
                "kind": "play",
                "selector": {"remainingCards": 4},
            },
            "outcome": _predicate(
                "PRED-ROOT-COUNT",
                "proof.winning_root_action_count",
                {"minimum": 0, "maximum": 8},
            ),
        }
    )
    plan = ScenarioCompilerV1().compile(scenario, request)
    assert plan.valid is True
    recipe = ResearchRecipeV4(
        "RCP-V4SEARCH1",
        "بحث موجه وآمن لسيناريو قابل للإثبات",
        {"bundleRevisionId": "CBR-TEST"},
        scenario,
        request,
        {},
        {
            "generationLane": "ortools",
            "generationPolicy": "scenario_guided_actor_safe",
            "seedStart": 1,
            "seedCount": 1,
            "seedsPerShard": 1,
            "candidatesPerSeed": 1,
            "solverTimeoutSeconds": 10.0,
            "maximumConflicts": 250000,
            "analysisMode": "cascaded",
            "proofBudgetProfile": "late_game_exact_v1",
        },
        {},
        {"mode": "on_demand"},
        "published",
        plan,
        "e" * 64,
    )
    candidate = ScenarioSearchExecutorV1().build_candidate(
        recipe,
        rules=rules,
        bundle_hash="b" * 64,
        seed=12001,
    )
    assert candidate.status == "satisfied"
    assert candidate.normalized is not None
    assert candidate.proof is not None
    assert candidate.satisfaction is not None
    assert candidate.satisfaction.status == "satisfied"
    assert candidate.construction_receipt["policy"]["observerSafe"] is True
    assert candidate.construction_receipt["policy"]["id"] == (
        "scenario_guided_actor_safe_v13"
    )
    assert candidate.construction_receipt["policy"]["guidancePlanHash"]
    assert all(
        row["guidancePlanHash"]
        == candidate.construction_receipt["policy"]["guidancePlanHash"]
        for row in candidate.construction_receipt["decisionReceipts"]
    )
    assert all(
        isinstance(score["total"], int)
        for row in candidate.construction_receipt["decisionReceipts"]
        for score in row["actionScores"].values()
    )
    assert candidate.construction_receipt["structural"]["solverStatus"] == "optimal"
    assert candidate.construction_receipt["auction"]["contract"] == "sun"
    assert candidate.construction_receipt["auction"]["buyer"] == "south"
    project = next(
        item
        for item in candidate.satisfaction.evidence
        if item.predicate_id == "PRED-PROJECT"
    )
    assert project.status == "satisfied"
    assert project.observed["projects"][0]["kind"] == "hundred_sequence"


def test_generated_sun_hundred_project_has_exact_no_salvage_contract_loss(rules):
    proof = _proof(
        claim="target_team_has_no_salvage",
        targetTeam="north_south",
        utility="contract_success",
        successThreshold=1,
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate("PRED-BUYER", "auction.buyer", {"seat": "south"}),
                    _predicate(
                        "PRED-CONTRACT",
                        "auction.contract",
                        {"contract": "sun"},
                    ),
                    _predicate(
                        "PRED-PROJECT",
                        "project.exists",
                        {
                            "team": "north_south",
                            "owner": "south",
                            "kind": "hundred_sequence",
                            "value": 20,
                            "state": "awarded",
                        },
                    ),
                ],
            },
            "decisionPoint": {
                "kind": "play",
                "selector": {"remainingCards": 8},
            },
            "outcome": _predicate(
                "PRED-ROOT",
                "proof.winning_root_action_count",
                {"minimum": 0, "maximum": 0},
            ),
        }
    )
    plan = ScenarioCompilerV1().compile(scenario, proof)
    assert plan.valid is True
    recipe = ResearchRecipeV4(
        "RCP-V4SUN100GEN",
        "سيناريو مولد لمشروع مية في الصن بلا إنقاذ",
        {"bundleRevisionId": "CBR-TEST"},
        scenario,
        proof,
        {},
        {
            "generationLane": "ortools",
            "generationPolicy": "scenario_guided_actor_safe",
            "seedStart": 12004,
            "seedCount": 1,
            "seedsPerShard": 1,
            "candidatesPerSeed": 1,
            "solverTimeoutSeconds": 10.0,
            "maximumConflicts": 250000,
            "analysisMode": "cascaded",
            "proofBudgetProfile": "late_game_exact_v1",
        },
        {},
        {"mode": "on_demand"},
        "published",
        plan,
        "f" * 64,
    )
    executor = ScenarioSearchExecutorV1()
    first = executor.build_candidate(
        recipe,
        rules=rules,
        bundle_hash="b" * 64,
        seed=12004,
    )
    second = executor.build_candidate(
        recipe,
        rules=rules,
        bundle_hash="b" * 64,
        seed=12004,
    )
    for candidate in (first, second):
        assert candidate.status == "satisfied"
        assert candidate.normalized is not None
        assert candidate.proof is not None
        assert candidate.satisfaction is not None
        assert candidate.normalized.round_state.contract.value == "sun"
        assert candidate.normalized.round_state.buyer == Seat.SOUTH
        assert any(
            project.owner == Seat.SOUTH
            and project.kind == "hundred_sequence"
            and project.value == 20
            and project.declared
            and project.scoring
            for project in candidate.normalized.round_state.projects
        )
        assert candidate.proof.payload["completionStatus"] == "exact_complete"
        assert candidate.proof.payload["claimSatisfied"] is True
        assert candidate.proof.payload["winningRootActionCount"] == 0
        assert all(
            item["satisfies"] is False
            for item in candidate.proof.payload["rootActions"]
        )
        assert candidate.construction_receipt["policy"]["observerSafe"] is True
        project_evidence = next(
            item
            for item in candidate.construction_receipt["structural"][
                "predicateEvidence"
            ]
            if item["predicateId"] == "PRED-PROJECT"
        )
        assert project_evidence["lowering"] == "cp_sat_necessary_project_pattern"
    assert first.normalized.exact_state_id == second.normalized.exact_state_id
    assert first.proof.payload["rootActions"] == second.proof.payload["rootActions"]
    assert (
        first.proof.payload["continuationLines"]
        == (second.proof.payload["continuationLines"])
    )
    assert (
        first.construction_receipt["structural"]["deckOrder"]
        == (second.construction_receipt["structural"]["deckOrder"])
    )


def test_scenario_guidance_changes_actor_safe_choices_without_hidden_truth(rules):
    def scenario(status: str) -> ResearchScenarioV1:
        return ResearchScenarioV1.build(
            {
                "schemaVersion": 1,
                "origin": {"kind": "search"},
                "constraints": {
                    "op": "all",
                    "children": [
                        _predicate(
                            "PRED-CARD-GUIDANCE",
                            "deal.card_owner",
                            {
                                "card": 0,
                                "seat": "north",
                                "handStage": "initial_five",
                            },
                        )
                    ],
                },
                "decisionPoint": {"kind": "play", "selector": {}},
                "outcome": _predicate(
                    f"PRED-OUTCOME-{status.upper()}",
                    "outcome.team_round",
                    {"team": "north_south", "status": status},
                ),
            }
        )

    proof = _proof(targetTeam="north_south")
    won_policy = ScenarioGuidedActorSafePolicyV1(
        ScenarioGuidancePlanV1.build(scenario("won"), proof)
    )
    lost_policy = ScenarioGuidedActorSafePolicyV1(
        ScenarioGuidancePlanV1.build(scenario("lost"), proof)
    )
    base = _late_prefix(rules)
    selected = None
    for removed in range(4, 25, 4):
        prefix = json.loads(json.dumps(base))
        prefix["playActions"] = prefix["playActions"][:-removed]
        normalized = ExactStateNormalizerV1(rules).normalize(prefix)
        state = normalized.round_state
        engine = BalootEngine(rules)
        if len(engine.legal_actions(state)) < 2:
            continue
        won_card, won_receipt = won_policy.choose_with_receipt(state, engine)
        lost_card, lost_receipt = lost_policy.choose_with_receipt(state, engine)
        if won_card != lost_card:
            selected = (state, engine, won_card, won_receipt, lost_card, lost_receipt)
            break
    assert selected is not None, "guided win/loss goals must influence a legal choice"
    state, engine, won_card, won_receipt, lost_card, lost_receipt = selected
    assert won_card != lost_card
    assert won_receipt["desiredTeam"] == "north_south"
    assert lost_receipt["desiredTeam"] == "west_east"

    actor = state.current_player
    assert actor is not None
    information = InformationState.from_complete(state, actor)
    hidden_by_seat = {
        seat: [
            card_id
            for card_id in range(32)
            if state.hands[int(seat)] & (1 << card_id)
            and not information.known_unplayed[int(seat)] & (1 << card_id)
        ]
        for seat in Seat
        if seat != actor
    }
    swappable = [(seat, cards[0]) for seat, cards in hidden_by_seat.items() if cards]
    assert len(swappable) >= 2
    (first_seat, first_card), (second_seat, second_card) = swappable[:2]
    hidden_variant = state.clone()
    hidden_variant.hands[int(first_seat)] ^= (1 << first_card) | (1 << second_card)
    hidden_variant.hands[int(second_seat)] ^= (1 << first_card) | (1 << second_card)
    assert InformationState.from_complete(hidden_variant, actor).to_dict() == (
        information.to_dict()
    )
    variant_card, variant_receipt = won_policy.choose_with_receipt(
        hidden_variant, engine
    )
    assert variant_card == won_card
    assert variant_receipt == won_receipt


def test_unique_round_win_guidance_targets_public_score_trajectory(rules):
    base = _late_prefix(rules)
    selected = None
    for removed in (1, 5, 9, 13):
        prefix = json.loads(json.dumps(base))
        prefix["playActions"] = prefix["playActions"][:-removed]
        state = ExactStateNormalizerV1(rules).normalize(prefix).round_state
        engine = BalootEngine(rules)
        if len(ScenarioGuidedActorSafePolicyV1._current_trick(
            InformationState.from_complete(state, state.current_player)
        )) != 3 or len(engine.legal_actions(state)) < 2:
            continue
        target_trick = len(state.completed_tricks) + 1
        scenario = ResearchScenarioV1.build(
            {
                "schemaVersion": 1,
                "origin": {"kind": "search"},
                "constraints": _predicate(
                    "PRED-TARGET-TRICK",
                    "play.trick_index",
                    {"minimum": target_trick, "maximum": target_trick},
                ),
                "decisionPoint": {"kind": "play", "selector": {}},
                "outcome": {
                    "op": "all",
                    "children": [
                        _predicate(
                            "PRED-ROUND",
                            "outcome.team_round",
                            {"team": state.current_player.team.name.lower(), "status": "won"},
                        ),
                        _predicate(
                            "PRED-UNIQUE",
                            "proof.winning_root_action_count",
                            {"minimum": 1, "maximum": 1},
                        ),
                        _predicate(
                            "PRED-RAW-AT-DECISION",
                            "state.current_raw_points",
                            {
                                "team": state.current_player.team.name.lower(),
                                "minimum": 40,
                                "maximum": 60,
                            },
                        ),
                    ],
                },
            }
        )
        guidance = ScenarioGuidancePlanV1.build(
            scenario,
            _proof(
                targetTeam=state.current_player.team.name.lower(),
                utility="round_win",
            ),
        )
        policy = ScenarioGuidedActorSafePolicyV1(guidance)
        _, receipt = policy.choose_with_receipt(state, engine)
        scores = {
            row["decisiveScoreTrajectory"]
            for row in receipt["actionScores"].values()
        }
        if len(scores) > 1:
            selected = (guidance, receipt)
            break
    assert selected is not None
    guidance, receipt = selected
    trajectory = next(
        objective
        for objective in guidance.objectives
        if objective["sourceKind"] == "actor_visible_round_score_trajectory"
    )
    assert trajectory["weight"] == 4
    raw_trajectory = next(
        objective
        for objective in guidance.objectives
        if objective["sourceKind"] == "actor_visible_raw_points_at_decision"
    )
    assert raw_trajectory["minimum"] == 40
    assert raw_trajectory["maximum"] == 60
    assert receipt["observerSafe"] is True
    assert max(
        row["decisiveScoreTrajectory"]
        for row in receipt["actionScores"].values()
    ) <= 0
    assert all(
        row["decisionRawTrajectory"] <= 0
        for row in receipt["actionScores"].values()
    )


def test_scenario_guidance_preserves_actor_visible_decision_cards(rules, root: Path):
    fixture = json.loads(
        (root / "examples" / "research-scenario-v4-ace-sacrifice-seven.json").read_text(
            encoding="utf-8"
        )
    )
    prefix = fixture["exactTracePrefix"]
    prefix["playActions"] = prefix["playActions"][:-4]
    normalized = ExactStateNormalizerV1(rules).normalize(prefix)
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-ACTOR",
                        "play.current_player",
                        {"seat": "west"},
                    ),
                    _predicate(
                        "PRED-DECISION-CARDS",
                        "state.legal_cards_include",
                        {"cards": [23]},
                    ),
                    _predicate(
                        "PRED-REMAINING-CARD",
                        "deal.remaining_card_owner",
                        {"card": 11, "seat": "west"},
                    ),
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-WINNING-ACTION",
                "proof.winning_root_action_count",
                {"minimum": 1, "maximum": 1},
            ),
        }
    )
    policy = ScenarioGuidedActorSafePolicyV1(
        ScenarioGuidancePlanV1.build(
            scenario,
            _proof(
                claim="selected_action_is_unique_winner",
                selectedAction="DA",
                targetTeam="west_east",
            ),
        )
    )
    selected, receipt = policy.choose_with_receipt(
        normalized.round_state,
        BalootEngine(rules),
    )
    assert selected.code == "DK"
    assert receipt["actionScores"]["H10"]["decisionCardPreservation"] == -1
    assert receipt["actionScores"]["DA"]["decisionCardPreservation"] == -1
    assert receipt["actionScores"]["DK"]["decisionCardPreservation"] == 0


def test_scenario_guidance_preserves_requested_root_rank_until_decision(rules):
    prefix = _late_prefix(rules)
    prefix["playActions"] = prefix["playActions"][:-8]
    state = ExactStateNormalizerV1(rules).normalize(prefix).round_state
    actor = state.current_player
    assert actor is not None
    engine = BalootEngine(rules)
    legal = engine.legal_actions(state)
    target = next(card for card in legal if card.rank.name.lower() == "ace")
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-ACTOR",
                        "play.current_player",
                        {"seat": actor.name.lower()},
                    ),
                    _predicate(
                        "PRED-TARGET-TRICK",
                        "play.trick_index",
                        {
                            "minimum": len(state.completed_tricks) + 1,
                            "maximum": len(state.completed_tricks) + 1,
                        },
                    ),
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-WINNING-ACE",
                "proof.winning_root_card_rank_count",
                {"rank": ["ace"], "minimum": 1, "maximum": 1},
            ),
        }
    )
    policy = ScenarioGuidedActorSafePolicyV1(
        ScenarioGuidancePlanV1.build(
            scenario,
            _proof(targetTeam=actor.team.name.lower()),
        )
    )
    selected, receipt = policy.choose_with_receipt(state, engine)
    assert selected != target
    assert receipt["actionScores"][target.code]["decisionRankPreservation"] == -1
    assert receipt["actionScores"][selected.code]["decisionRankPreservation"] == 0


def test_scenario_guidance_derives_partial_trick_leader_from_actor_position():
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-TRICK-SIX",
                        "play.trick_index",
                        {"minimum": 5, "maximum": 5},
                    ),
                    _predicate(
                        "PRED-ACTOR",
                        "play.current_player",
                        {"seat": "west"},
                    ),
                    _predicate(
                        "PRED-SOUTH-H8",
                        "deal.remaining_card_owner",
                        {"card": Card.parse("H8").id, "seat": "south"},
                    ),
                    _predicate(
                        "PRED-SOUTH-HA",
                        "deal.remaining_card_owner",
                        {"card": Card.parse("HA").id, "seat": "south"},
                    ),
                    _predicate(
                        "PRED-SOUTH-C8",
                        "deal.remaining_card_owner",
                        {"card": Card.parse("C8").id, "seat": "south"},
                    ),
                ],
            },
            "decisionPoint": {
                "kind": "play",
                "selector": {"remainingCards": 9, "actingSeat": "west"},
            },
            "outcome": _predicate(
                "PRED-WINNING-ACE",
                "proof.winning_root_card_rank_count",
                {"rank": ["ace"], "minimum": 1, "maximum": 1},
            ),
        }
    )
    guidance = ScenarioGuidancePlanV1.build(
        scenario,
        _proof(targetTeam="west_east"),
    )
    leader = next(
        objective
        for objective in guidance.objectives
        if objective["sourceKind"] == "target_partial_trick_leader"
    )
    assert leader["targetTrickIndex"] == 5
    assert leader["targetLeader"] == "south"
    assert leader["bridgeSuit"] == "clubs"
    assert leader["turnoverSuit"] == "hearts"


def test_scenario_guidance_preserves_an_actor_visible_leader_bridge(rules, root: Path):
    fixture = json.loads(
        (root / "examples" / "research-scenario-v4-ace-sacrifice-seven.json").read_text(
            encoding="utf-8"
        )
    )
    prefix = fixture["exactTracePrefix"]
    prefix["playActions"] = prefix["playActions"][:-7]
    state = ExactStateNormalizerV1(rules).normalize(prefix).round_state
    assert state.current_player == Seat.SOUTH
    guidance = ScenarioGuidancePlanV1(
        "s" * 64,
        (
            {
                "sourceId": "DECISION-POSITION",
                "sourceKind": "target_partial_trick_leader",
                "targetLeader": "south",
                "targetTrickIndex": 6,
                "bridgeSuit": "diamonds",
                "weight": 4,
            },
            {
                "sourceId": "DECISION-CARDS",
                "sourceKind": "actor_visible_card_preservation",
                "actingSeat": "south",
                "protectedCards": ["H8", "D9"],
                "weight": 1,
            },
        ),
        "g" * 64,
    )
    _, receipt = ScenarioGuidedActorSafePolicyV1(guidance).choose_with_receipt(
        state,
        BalootEngine(rules),
    )
    assert receipt["actionScores"]["DQ"]["targetLeaderBridgePreservation"] == -4
    assert receipt["actionScores"]["D7"]["targetLeaderBridgePreservation"] == 0
    assert receipt["observerSafe"] is True


def test_scenario_guidance_prefers_requested_public_lead_suit_and_winning_rank(rules):
    prefix = _late_prefix(rules)
    prefix["playActions"] = prefix["playActions"][:-8]
    normalized = ExactStateNormalizerV1(rules).normalize(prefix)
    state = normalized.round_state
    actor = state.current_player
    assert actor is not None
    legal = BalootEngine(rules).legal_actions(state)
    target = min(legal, key=lambda card: card.id)
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _predicate(
                "PRED-LEAD-SUIT",
                "play.current_trick_suit_led",
                {"suit": target.suit.name.lower()},
            ),
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-LOW-WINNER",
                "proof.principal_variation_trick_winner",
                {
                    "trickIndex": len(state.completed_tricks),
                    "team": actor.team.name.lower(),
                    "rank": [target.rank.name.lower()],
                },
            ),
        }
    )
    policy = ScenarioGuidedActorSafePolicyV1(
        ScenarioGuidancePlanV1.build(
            scenario, _proof(targetTeam=actor.team.name.lower())
        )
    )
    selected, receipt = policy.choose_with_receipt(state, BalootEngine(rules))
    assert selected.suit == target.suit
    assert selected.rank == target.rank
    assert receipt["actionScores"][selected.code]["leadSuitAlignment"] > 0
    assert receipt["actionScores"][selected.code]["proofRankAlignment"] > 0


def test_scenario_guidance_selects_requested_current_trick_card(rules):
    prefix = _late_prefix(rules)
    prefix["playActions"] = prefix["playActions"][:-8]
    normalized = ExactStateNormalizerV1(rules).normalize(prefix)
    state = normalized.round_state
    actor = state.current_player
    assert actor is not None
    target = min(BalootEngine(rules).legal_actions(state), key=lambda card: card.id)
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-TARGET-TRICK",
                        "play.trick_index",
                        {
                            "minimum": len(state.completed_tricks),
                            "maximum": len(state.completed_tricks),
                        },
                    ),
                    _predicate(
                        "PRED-TARGET-CARD",
                        "play.current_trick_card",
                        {
                            "position": 0,
                            "seat": actor.name.lower(),
                            "card": target.id,
                        },
                    ),
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-WIN",
                "proof.winning_root_action_count",
                {"minimum": 1, "maximum": 1},
            ),
        }
    )
    policy = ScenarioGuidedActorSafePolicyV1(
        ScenarioGuidancePlanV1.build(
            scenario,
            _proof(targetTeam=actor.team.name.lower()),
        )
    )
    selected, receipt = policy.choose_with_receipt(state, BalootEngine(rules))
    assert selected == target
    assert receipt["actionScores"][target.code]["targetCardAlignment"] > 0


def test_scenario_guidance_preserves_requested_lead_suit_until_target_trick(rules):
    prefix = _late_prefix(rules)
    prefix["playActions"] = prefix["playActions"][:-8]
    state = ExactStateNormalizerV1(rules).normalize(prefix).round_state
    actor = state.current_player
    assert actor is not None
    legal = BalootEngine(rules).legal_actions(state)
    target = next(
        card for card in legal if any(other.suit != card.suit for other in legal)
    )
    target_trick = len(state.completed_tricks) + 1
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-TARGET-LEADER",
                        "play.current_trick_leader",
                        {"seat": actor.name.lower()},
                    ),
                    _predicate(
                        "PRED-TARGET-TRICK",
                        "play.trick_index",
                        {"minimum": target_trick, "maximum": target_trick},
                    ),
                    _predicate(
                        "PRED-TARGET-SUIT",
                        "play.current_trick_suit_led",
                        {"suit": target.suit.name.lower()},
                    ),
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-WIN",
                "proof.winning_root_action_count",
                {"minimum": 1, "maximum": 1},
            ),
        }
    )
    policy = ScenarioGuidedActorSafePolicyV1(
        ScenarioGuidancePlanV1.build(
            scenario,
            _proof(targetTeam=actor.team.name.lower()),
        )
    )
    selected, receipt = policy.choose_with_receipt(state, BalootEngine(rules))
    assert selected.suit != target.suit
    assert receipt["actionScores"][target.code]["targetLeadPreservation"] < 0


def test_v4_search_run_preserves_misses_no_goods_and_stage_yields(rules):
    proof = _proof(targetTeam="north_south", successThreshold=-1_000_000)
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate("PRED-BUYER", "auction.buyer", {"seat": "south"}),
                    _predicate(
                        "PRED-CONTRACT",
                        "auction.contract",
                        {"contract": "sun"},
                    ),
                ],
            },
            "decisionPoint": {
                "kind": "play",
                "selector": {"remainingCards": 4},
            },
            "outcome": _predicate(
                "PRED-IMPOSSIBLE-LEGAL-COUNT",
                "state.legal_action_count",
                {"minimum": 9, "maximum": 9},
            ),
        }
    )
    plan = ScenarioCompilerV1().compile(scenario, proof)
    configuration, context = _published_v4_configuration()
    assert context.bundle.rules.sha256 == rules.sha256
    recipe = ResearchRecipeV4(
        "RCP-V4-NO-GOODS",
        "بحث محدود يحفظ إخفاقات البناء",
        configuration,
        scenario,
        proof,
        {},
        {
            "generationLane": "ortools",
            "generationPolicy": "scenario_guided_actor_safe",
            "seedStart": 7,
            "seedCount": 1,
            "seedsPerShard": 1,
            "candidatesPerSeed": 2,
            "solverTimeoutSeconds": 10.0,
            "maximumConflicts": 250000,
            "analysisMode": "cascaded",
            "proofBudgetProfile": "late_game_exact_v1",
        },
        {},
        {"mode": "on_demand"},
        "published",
        plan,
        "c" * 64,
    )
    run = {
        "runCode": "RUN-V4NOGOODS01",
        "runHash": "d" * 64,
        "recipeCode": recipe.code,
        "recipeHash": recipe.sha256,
        "status": "queued",
        "progress": 0.0,
        "events": [],
        "createdAt": "2026-08-12T00:00:00.000000+00:00",
        "configurationAuthority": configuration,
    }
    harness = SimpleNamespace(
        store=_ScenarioRunStore(run),
        artifact_store=_ScenarioArtifactStore(),
        _cancel_check=lambda _run_code: False,
    )
    ResearchFoundryServiceV1._execute_v4_search_run(
        harness,
        run,
        recipe,
        context,
    )
    frozen = harness.store.run
    assert frozen["status"] == "completed"
    assert frozen["scenarioSearchStatus"] == "not_found_within_budget"
    assert frozen["attemptedCandidates"] == 2
    assert frozen["resultCount"] == 2
    assert frozen["rejectedCount"] == 2
    assert frozen["scenarioYieldCounts"] == {
        "attempted": 2,
        "structurallyFeasible": 2,
        "legalReplay": 2,
        "semanticSatisfied": 0,
        "proofCompleted": 2,
        "qualified": 0,
    }
    assert len(frozen["noGoodReceiptHashes"]) == 2
    evidence_ref = frozen["scenarioSearchEvidence"]
    evidence = json.loads(harness.artifact_store.get(evidence_ref["uri"]))
    assert (
        evidence["sha256"]
        == hashlib.sha256(
            json.dumps(
                {key: value for key, value in evidence.items() if key != "sha256"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    assert [row["scope"] for row in evidence["noGoodReceipts"]] == [
        "exact_deck",
        "exact_deck",
    ]
    first_deck_hash = evidence["noGoodReceipts"][0]["deckHash"]
    assert evidence["constructionReceipts"][0]["structural"]["noGoodHashes"] == []
    assert evidence["constructionReceipts"][1]["structural"]["noGoodHashes"] == [
        first_deck_hash
    ]


def test_v4_stored_result_normalizes_missing_artifact_to_fail_closed(rules):
    result, harness, _recipe, _normalized = _execute_exact_v4_result(rules)
    proof_uri = result["artifacts"]["proofOutcome"]["uri"]
    del harness.artifact_store.values[proof_uri]
    service = SimpleNamespace(artifact_store=harness.artifact_store)
    with pytest.raises(RuntimeError, match="artifact is unavailable"):
        ResearchFoundryServiceV1._validate_stored_scenario_result_v4(
            service,
            result,
        )


@pytest.mark.parametrize("missing_error", [FileNotFoundError, KeyError])
def test_native_result_artifact_loaders_fail_closed_without_a_server_error(
    missing_error,
):
    def missing(_uri):
        raise missing_error("missing CAS object")

    store = SimpleNamespace(get=missing)
    reference = {
        "uri": "cas://sha256/" + "a" * 64,
        "sha256": "a" * 64,
        "bytes": 1,
        "mediaType": "application/json",
        "name": "missing.json",
    }
    with pytest.raises(RuntimeError, match="artifact is unavailable"):
        _load_result_json_artifact_v1(
            store, reference, media_type="application/json"
        )
    with pytest.raises(RuntimeError, match="artifact is unavailable"):
        _load_owned_result_json_artifact_v1(
            store,
            {**reference, "ownerKind": "semantic", "ownerHash": "b" * 64},
            media_type="application/json",
            owner_kind="semantic",
        )


def test_sqlite_rejects_valid_v4_result_moved_to_another_owner_row(
    rules,
    tmp_path: Path,
):
    result_a, harness, _recipe, _normalized = _execute_exact_v4_result(rules)
    result_b = json.loads(json.dumps(result_a))
    identity_b = result_b["provenance"]["resultIdentity"]
    identity_b["runCode"] = "RUN-SUBSTITUTEB"
    identity_b["runHash"] = "b" * 64
    result_b["runCode"] = identity_b["runCode"]
    result_b["resultHash"] = canonical_sha256(identity_b)
    result_b["resultCode"] = f"LAF-{result_b['resultHash'][:10].upper()}"
    assert validate_scenario_result_v4(
        result_b,
        artifact_store=harness.artifact_store,
    ) == result_b

    store = SQLiteResearchStoreV1(tmp_path / "research.sqlite3")
    try:
        store.put_recipe(
            {
                "code": result_a["recipeCode"],
                "sha256": result_a["provenance"]["recipeHash"],
                "status": "published",
            }
        )
        for value in (result_a, result_b):
            store.create_run(
                {
                    "runCode": value["runCode"],
                    "runHash": value["provenance"]["resultIdentity"]["runHash"],
                    "recipeCode": value["recipeCode"],
                    "recipeHash": value["provenance"]["recipeHash"],
                    "status": "completed",
                    "progress": 1.0,
                }
            )
            store.put_result(value)
        payload_b = store._db.execute(
            "SELECT payload_json FROM research_results WHERE result_code=?",
            (result_b["resultCode"],),
        ).fetchone()["payload_json"]
        store._db.execute(
            "UPDATE research_results SET payload_json=? WHERE result_code=?",
            (payload_b, result_a["resultCode"]),
        )
        store._db.commit()
        with pytest.raises(
            RuntimeError,
            match="stored research result identity mismatch",
        ):
            store.get_result(result_a["resultCode"])
    finally:
        store.close()


def test_v4_consistently_resigned_wrong_bundle_fails_frozen_authority(rules):
    result, harness, recipe, _normalized = _execute_exact_v4_result(rules)
    original_proof = json.loads(
        harness.artifact_store.values[
            result["artifacts"]["proofOutcome"]["uri"]
        ].decode("utf-8")
    )
    configuration = {
        "schemaVersion": 1,
        "semanticBundleHash": original_proof["identities"]["bundleHash"],
        "identities": {
            "rules": {"sha256": original_proof["identities"]["rulesHash"]}
        },
        "sha256": "c" * 64,
    }
    recipe_value = recipe.to_dict()
    recipe_value["configuration"] = configuration
    identity = result["provenance"]["resultIdentity"]
    authority_hash = canonical_sha256(configuration)
    identity["configurationAuthorityHash"] = authority_hash
    result["provenance"]["configurationAuthorityHash"] = authority_hash

    proof = json.loads(json.dumps(original_proof))
    proof["identities"]["bundleHash"] = "d" * 64
    proof.pop("semanticSha256", None)
    proof.pop("sha256", None)
    proof = ProofOutcomeV1.build(proof).to_dict()
    proof_bytes = json.dumps(
        proof,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    proof_reference = harness.artifact_store.put(
        proof_bytes,
        media_type=result["artifacts"]["proofOutcome"]["mediaType"],
        name=f"{proof['sha256']}.json",
    )
    result["proofOutcome"] = public_proof_outcome_v1(proof)
    result["artifacts"]["proofOutcome"] = proof_reference
    result["qualification"]["proofOutcomeHash"] = proof["semanticSha256"]
    observer = result["qualification"]["observerSafetyEvidence"]
    observer["proofOutcomeHash"] = proof["semanticSha256"]
    observer["sha256"] = canonical_sha256(
        {key: value for key, value in observer.items() if key != "sha256"}
    )
    result["scoreEvidence"]["proofOutcomeHash"] = proof["semanticSha256"]
    result["scoreEvidence"]["observerSafetyEvidenceHash"] = observer["sha256"]
    result["scoreEvidence"]["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in result["scoreEvidence"].items()
            if key != "sha256"
        }
    )
    identity["proofOutcomeHash"] = proof["semanticSha256"]
    identity["observerSafetyEvidenceHash"] = observer["sha256"]
    identity["scoreEvidenceHash"] = result["scoreEvidence"]["sha256"]
    verdict_qualification = {
        key: value
        for key, value in result["qualification"].items()
        if key != "observerSafetyEvidence"
    }
    verdict_qualification["observerSafetyEvidenceHash"] = observer["sha256"]
    identity["verdictHash"] = canonical_sha256(
        {
            "schemaVersion": 1,
            "score": result["score"],
            "truthLabel": result["truthLabel"],
            "confidence": result["confidence"],
            "disposition": result["disposition"],
            "qualification": verdict_qualification,
        }
    )
    identity["artifactHashes"]["proofOutcome"] = proof_reference["sha256"]
    result["resultHash"] = canonical_sha256(identity)
    result["resultCode"] = f"LAF-{result['resultHash'][:10].upper()}"
    assert validate_scenario_result_v4(
        result,
        artifact_store=harness.artifact_store,
    ) == result

    run = {
        "runCode": result["runCode"],
        "runHash": identity["runHash"],
        "recipeCode": result["recipeCode"],
        "recipeHash": recipe.sha256,
        "configurationAuthority": configuration,
        "generationLane": "exact_input",
        "createdAt": result["createdAt"],
    }
    from baloot_platform.research.result_library import (
        validate_scenario_result_authority_v1,
    )

    with pytest.raises(RuntimeError, match="bundle authority mismatch"):
        validate_scenario_result_authority_v1(
            result,
            run=run,
            recipe=recipe_value,
        )
