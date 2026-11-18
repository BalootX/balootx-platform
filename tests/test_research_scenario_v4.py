from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

import pytest
import yaml
from fastapi.testclient import TestClient

from baloot_platform.analysis.models import InformationState
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import GamePhase, Seat, Team
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.structural_ir import StructuralDomainIRV1
from baloot_platform.research.exact_state import ExactStateNormalizerV1
from baloot_platform.research.auction_compiler import OrToolsCpSatAuctionCompilerV1
from baloot_platform.research.proof_service import (
    ScenarioProofBudgetRegistryV1,
    ScenarioProofBudgetV1,
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
from baloot_platform.research.scenario_construction import ScenarioStructuralCompilerV1
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
from baloot_platform.video.audio_postprocess import build_audio_cue_plan
from baloot_platform.video.render_bundle import RenderBundleV1
from scripts.benchmark_ortools_research_v4 import _publish_explicit_bundle


_ORTOOLS_V4_REVISION_ID: str | None = None
_API_CP_SAT_BUDGET_SECONDS = 900.0


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


def _published_ortools_v4_revision_id() -> str:
    global _ORTOOLS_V4_REVISION_ID
    if _ORTOOLS_V4_REVISION_ID is None:
        root = Path(__file__).parents[1]
        authority, context = _publish_explicit_bundle(
            api.CONTAINER,
            root,
            {
                "bundle": {
                    "id": "baloot_ortools_research_native_v3",
                    "source": "configs/bundle_intelligent_research.yaml",
                    "backend": "ortools_cp_sat_v3",
                }
            },
        )
        configuration = api.CONTAINER.configuration
        challenge = copy.deepcopy(context.bundle.component("challenge"))
        historical = yaml.safe_load((root / "configs/challenge/progressive_mining_ortools_v3_1.yaml").read_text(encoding="utf-8"))
        challenge["scenarioResearchV4"]["searchBudgets"]["profiles"]["late_game_exact_v1"] = (
            historical["scenarioResearchV4"]["searchBudgets"]["profiles"]["late_game_exact_v1"]
        )
        assert challenge["scenarioResearchV4"]["qualityExecution"]["featuredPlayNativeProof"] is True
        assert challenge["scenarioResearchV4"]["searchBudgets"]["profiles"]["late_game_exact_v1"]["maximumWallSeconds"] == 60
        _apply_measured_api_budgets(challenge)
        component = configuration.create_profile(
            family="challenge", payload=challenge, created_by="TEST-RESEARCH-API", publish=True,
            scope="environment", environment="research-api-quality-fixture",
        )
        base = configuration.store.get_bundle(authority.bundle_revision_id)
        assert base is not None
        fixture = configuration.create_bundle(
            bundle_id="baloot_research_api_quality_fixture", version="1.0.0", mode=base.mode,
            component_revisions={**base.component_revisions, "challenge": component["revisionId"]},
            created_by="TEST-RESEARCH-API", publish=True,
            scope="environment", environment="research-api-quality-fixture",
        )
        _ORTOOLS_V4_REVISION_ID = fixture["revisionId"]
    return _ORTOOLS_V4_REVISION_ID


def _apply_measured_api_budgets(challenge: dict) -> dict:
    before = copy.deepcopy(challenge)
    scenario = challenge["scenarioResearchV4"]
    scenario["qualityExecution"]["maximumWallSeconds"] = max(
        float(scenario["qualityExecution"]["maximumWallSeconds"]),
        _API_CP_SAT_BUDGET_SECONDS,
    )
    for group, keys in (
        ("proofBudgets", ("maximumWallSeconds", "maximumCpuSeconds")),
        ("searchBudgets", ("maximumWallSeconds", "maximumDeterministicTime")),
    ):
        for profile in scenario[group]["profiles"].values():
            for key in keys:
                profile[key] = max(float(profile[key]), _API_CP_SAT_BUDGET_SECONDS)
                assert profile[key] >= _API_CP_SAT_BUDGET_SECONDS

    def admission_projection(value: dict) -> dict:
        projected = copy.deepcopy(value)
        section = projected["scenarioResearchV4"]
        section["qualityExecution"].pop("maximumWallSeconds", None)
        for group, keys in (
            ("proofBudgets", ("maximumWallSeconds", "maximumCpuSeconds")),
            ("searchBudgets", ("maximumWallSeconds", "maximumDeterministicTime")),
        ):
            for profile in section[group]["profiles"].values():
                for key in keys:
                    profile.pop(key, None)
        return projected

    assert admission_projection(before) == admission_projection(challenge)
    return challenge


def _api_budgeted_quality_policy() -> dict:
    return {"schemaVersion": 1, "mode": "budgeted", "decisions": {
        kind: {
            "utility": "awarded_point_margin", "budgetProfile": "bounded_v1",
            "horizon": {"unit": "tricks" if kind == "play" else "plies", "value": 1},
            "boundaryEvaluator": "auction_hand_margin_v1" if kind == "bid" else "current_raw_point_margin_v1",
        } for kind in SCENARIO_DECISION_KINDS
    }}


def _api_budgeted_quality_prefix(root: Path) -> dict:
    """Keep positive input bound to the current bounded policy, not its filename."""
    fixture = json.loads((root / "tests/fixtures/research-native-observer-budgeted-round-v3.json").read_text(encoding="utf-8"))
    assert fixture["playingQuality"] == ScenarioPlayingQualityPolicyV1.build(_api_budgeted_quality_policy()).to_dict()
    assert fixture["maximumHiddenWorldsPerDecision"] == 8
    return fixture["exactTracePrefix"]


def _predicate(predicate_id: str, kind: str, arguments: dict) -> dict:
    return {
        "predicateId": predicate_id,
        "kind": kind,
        "version": 1,
        "hardness": "hard",
        "arguments": arguments,
    }


def _assert_public_launch_admission(admission: dict, run_code: str) -> None:
    run = api.RESEARCH.store.get_run(run_code)
    assert run is not None
    receipt = run["launchAdmission"]
    assert admission["sha256"] == receipt["sha256"] == canonical_sha256({
        key: value for key, value in receipt.items() if key != "sha256"
    })
    assert admission == api.ResearchRunLaunchAdmissionResponse.model_validate(receipt).model_dump(exclude_none=True)


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


def _proof(**overrides) -> ScenarioProofRequestV1:
    value = {
        "schemaVersion": 1,
        "mode": "exact",
        "visibility": "complete_state",
        "claim": "winning_root_action_count",
        "targetTeam": "north_south",
        "utility": "awarded_point_margin",
        "successThreshold": 1,
        "lineUniverse": "all_legal",
        "budgetProfile": "late_game_exact_v1",
    }
    value.update(overrides)
    if value.get("visibility") == "actor_information_set":
        value.setdefault(
            "strategyClass",
            "frozen_policy"
            if value.get("mode") == "policy_conditioned"
            else "pure_deterministic",
        )
    return ScenarioProofRequestV1.build(value)


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


def _published_v4_configuration():
    revision_id = api.CONTAINER.configuration.resolve(bundle_id="baloot_default")[
        "bundle"
    ]["revisionId"]
    authority, context = api.RESEARCH.execution_context_factory.freeze_selector(
        bundle_revision_id=revision_id,
        analysis_mode="cascaded",
    )
    return authority.to_dict(), context


def _scenario(*predicates: dict) -> ResearchScenarioV1:
    return ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {"op": "all", "children": list(predicates)},
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-OUTCOME",
                "outcome.team_round",
                {"team": "north_south", "status": "won"},
            ),
        }
    )


def _late_prefix(rules) -> dict:
    deck = list(range(32))
    engine = BalootEngine(rules)
    state = engine.create_round(seed=8172, dealer=Seat.EAST)
    engine.initial_deal_from_preordered(state, deck)
    assert state.purchase_card is not None
    state.phase = GamePhase.BIDDING
    machine = BiddingMachine(
        state.dealer,
        state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(rules.data),
    )
    auction_actions = [
        {"action": "pass"},
        {"action": "pass"},
        {"action": "sun"},
        {"action": "pass"},
        {"action": "sun"},
    ]
    for action in auction_actions:
        machine.submit(action["action"])
    assert machine.result is not None
    engine.complete_deal_from_bid_result(state, machine.result)
    engine.prepare_projects(state)
    bot = HeuristicBot(engine, BotProfile(8172))
    play_actions = []
    while sum(mask.bit_count() for mask in state.hands) > 4:
        assert state.current_player is not None
        card = bot.choose(state)
        play_actions.append(
            {"actor": state.current_player.name.lower(), "card": card.code}
        )
        engine.play_card(state, card)
    project_declarations = [
        {
            key: event[key]
            for key in (
                "owner",
                "kind",
                "value",
                "highRank",
                "declarationOrder",
            )
        }
        for event in state.to_dict(reveal_hands=True)["events"]
        if event["type"] == "project_declared"
    ]
    return {
        "schemaVersion": 1,
        "deckOrder": deck,
        "dealer": "east",
        "roundNumber": 1,
        "matchScore": [0, 0],
        "auctionActions": auction_actions,
        "escalationActions": [],
        "projectDeclarations": project_declarations,
        "projectAssertions": [],
        "playActions": play_actions,
        "stateAssertions": {"phase": "playing"},
    }


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


def _policy_proof_for(team: str) -> ScenarioProofRequestV1:
    policies = ResearchFoundryServiceV1._scenario_recipe_configuration()[
        "defaultPolicyProfiles"
    ]
    return ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": team,
            "utility": "awarded_point_margin",
            "successThreshold": -10000.0,
            "lineUniverse": "policy_conditioned",
            "policies": policies,
            "budgetProfile": "bounded_v1",
        }
    )


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


def test_exact_trace_prefix_replays_and_proves_every_root_action(rules, root: Path):
    normalized = ExactStateNormalizerV1(rules).normalize(_late_prefix(rules))
    assert normalized.phase == "playing"
    assert sum(mask.bit_count() for mask in normalized.round_state.hands) == 4
    assert normalized.state_hash == canonical_sha256(normalized.state)
    assert normalized.project_declarations_verified == len(
        [
            event
            for event in normalized.state["events"]
            if event["type"] == "project_declared"
        ]
    )
    invalid_declaration = _late_prefix(rules)
    invalid_declaration["projectDeclarations"][0]["owner"] = "south"
    with pytest.raises(
        ValueError,
        match="project declarations do not match authoritative events",
    ):
        ExactStateNormalizerV1(rules).normalize(invalid_declaration)
    service = UnifiedScenarioProofServiceV1.from_repository(root)
    request = _proof(
        targetTeam=(
            "north_south"
            if normalized.round_state.current_player.team.value == 0
            else "west_east"
        )
    )
    outcome = service.prove(normalized, request, bundle_hash="a" * 64)
    payload = outcome.to_dict()
    assert payload["completionStatus"] == "exact_complete"
    assert payload["achievedMode"] == "exact"
    assert len(payload["rootActions"]) == len(normalized.legal_actions)
    assert payload["winningRootActionCount"] == sum(
        bool(item["satisfies"]) for item in payload["rootActions"]
    )
    assert payload["continuationLines"]["complete"] is True
    assert ProofOutcomeRoundTrip(payload) == payload

    terminal = normalized.round_state.clone()
    engine = BalootEngine(rules)
    for code in payload["principalVariation"]:
        engine.play_card(terminal, Card.parse(code), emit_events=False)
    assert normalized.round_state.round_score is None
    assert terminal.round_score is not None
    assert terminal.declarer is not None
    buyer_status = (
        "won"
        if terminal.round_score.winner == terminal.declarer.team
        else "lost"
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _predicate(
                "PRED-ACTOR",
                "play.current_player",
                {"seat": normalized.next_actor},
            ),
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-BUYER-OUTCOME",
                "outcome.buyer_contract",
                {"status": buyer_status},
            ),
        }
    )
    satisfaction = ScenarioRuntimeVerifierV1().verify(
        scenario,
        normalized,
        proof_outcome=outcome,
    )
    assert satisfaction.status == "satisfied"
    outcome_evidence = next(
        item for item in satisfaction.evidence if item.kind == "outcome.buyer_contract"
    )
    assert outcome_evidence.observed["stateSource"] == "proof_principal_variation"


def test_golden_sun_hundred_project_has_exactly_no_salvage(rules, root: Path):
    fixture = json.loads(
        (root / "examples" / "research-scenario-v4-golden.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    expected = fixture["expected"]
    assert normalized.phase == expected["phase"]
    assert normalized.round_state.buyer == Seat.NORTH
    assert normalized.round_state.contract.value == expected["contract"]
    awarded = [
        project
        for project in normalized.round_state.projects
        if project.owner == Seat.NORTH
        and project.kind == "hundred_sequence"
        and project.value == 20
        and project.scoring
    ]
    assert len(awarded) == 1

    proof_request = ScenarioProofRequestV1.build(fixture["proof"])
    proof = UnifiedScenarioProofServiceV1.from_repository(root).prove(
        normalized,
        proof_request,
        bundle_hash="b" * 64,
    )
    payload = proof.payload
    assert payload["completionStatus"] == expected["completionStatus"]
    assert payload["claimSatisfied"] is expected["claimSatisfied"]
    assert payload["winningRootActionCount"] == expected["winningRootActionCount"]
    assert sorted(item["action"] for item in payload["rootActions"]) == sorted(
        expected["legalRootActions"]
    )
    assert [item["value"] for item in payload["rootActions"]] == expected["rootValues"]
    assert all(item["satisfies"] is False for item in payload["rootActions"])


def test_golden_sun_hundred_project_has_one_saving_move(rules, root: Path):
    fixture = json.loads(
        (root / "examples" / "research-scenario-v4-unique-saving.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    assert normalized.round_state.buyer == Seat.NORTH
    assert normalized.round_state.contract.value == "sun"
    assert any(
        project.owner == Seat.NORTH
        and project.kind == "hundred_sequence"
        and project.value == 20
        and project.scoring
        for project in normalized.round_state.projects
    )
    proof = UnifiedScenarioProofServiceV1.from_repository(root).prove(
        normalized,
        ScenarioProofRequestV1.build(fixture["proof"]),
        bundle_hash="b" * 64,
    )
    expected = fixture["expected"]
    payload = proof.payload
    assert payload["completionStatus"] == expected["completionStatus"]
    assert payload["claimSatisfied"] is True
    assert payload["selectedAction"] == expected["selectedAction"]
    assert payload["winningRootActionCount"] == 1
    assert payload["continuationLines"] == {
        "complete": True,
        "exactCount": expected["winningContinuationLineCount"],
        "lineUniverse": "all_legal",
    }
    root_actions = {item["action"]: item for item in payload["rootActions"]}
    assert root_actions["C10"]["satisfies"] is False
    assert root_actions["C10"]["value"] == 0
    assert root_actions["H10"]["satisfies"] is True
    assert root_actions["H10"]["value"] == 10

    wrong_action_request = ScenarioProofRequestV1.build(
        {**fixture["proof"], "selectedAction": "C10"}
    )
    wrong_action = UnifiedScenarioProofServiceV1.from_repository(root).prove(
        normalized,
        wrong_action_request,
        bundle_hash="b" * 64,
    )
    assert wrong_action.payload["completionStatus"] == "proof_failed"
    assert wrong_action.payload["claimSatisfied"] is False
    assert wrong_action.payload["selectedAction"] == "C10"

    illegal_action_request = ScenarioProofRequestV1.build(
        {**fixture["proof"], "selectedAction": "D7"}
    )
    with pytest.raises(ValueError, match="not a legal root action"):
        UnifiedScenarioProofServiceV1.from_repository(root).prove(
            normalized,
            illegal_action_request,
            bundle_hash="b" * 64,
        )


def test_golden_sun_hundred_project_requires_ace_sacrifice_for_partner_seven(
    rules,
    root: Path,
):
    fixture = json.loads(
        (root / "examples" / "research-scenario-v4-ace-sacrifice-seven.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    expected = fixture["expected"]
    assert normalized.phase == expected["phase"]
    assert normalized.next_actor == expected["actingSeat"]
    assert normalized.round_state.buyer == Seat.NORTH
    assert normalized.round_state.contract.value == expected["contract"]
    assert normalized.round_state.current_trick is not None
    assert (
        sum(mask.bit_count() for mask in normalized.round_state.hands)
        + len(normalized.round_state.current_trick.plays)
        == 8
    )
    assert any(
        project.owner == Seat.NORTH
        and project.kind == "hundred_sequence"
        and project.value == 20
        and project.scoring
        for project in normalized.round_state.projects
    )

    payload = (
        UnifiedScenarioProofServiceV1.from_repository(root)
        .prove(
            normalized,
            ScenarioProofRequestV1.build(fixture["proof"]),
            bundle_hash="b" * 64,
        )
        .payload
    )
    roots = {item["action"]: item for item in payload["rootActions"]}
    assert payload["completionStatus"] == expected["completionStatus"]
    assert payload["claimSatisfied"] is True
    assert payload["selectedAction"] == expected["selectedAction"]
    assert payload["winningRootActionCount"] == 1
    assert payload["continuationLines"] == {
        "complete": True,
        "exactCount": expected["winningContinuationLineCount"],
        "lineUniverse": "all_legal",
    }
    assert {action: item["value"] for action, item in roots.items()} == expected[
        "rootValues"
    ]
    assert roots["DA"]["satisfies"] is True
    assert roots["H10"]["satisfies"] is False
    assert payload["principalVariation"] == expected["principalVariation"]


def test_golden_ace_sacrifice_is_only_win_and_final_nine_takes_two_aces(
    rules,
    root: Path,
):
    fixture = json.loads(
        (
            root
            / "examples"
            / "research-scenario-v4-ace-sacrifice-two-aces.json"
        ).read_text(encoding="utf-8")
    )
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    state = normalized.round_state
    expected = fixture["expected"]
    assert normalized.next_actor == expected["actingSeat"]
    assert sorted(action["code"] for action in normalized.legal_actions) == sorted(
        expected["legalRootActions"]
    )
    assert [
        sum(
            trick.raw_points
            for trick in state.completed_tricks
            if int(trick.winner.team) == team
        )
        for team in range(2)
    ] == expected["rawPoints"]
    assert any(
        project.owner == Seat.NORTH
        and project.kind == "hundred_sequence"
        and project.value == 20
        and project.scoring
        for project in state.projects
    )

    proof = UnifiedScenarioProofServiceV1.from_repository(root).prove(
        normalized,
        ScenarioProofRequestV1.build(fixture["proof"]),
        bundle_hash="b" * 64,
    )
    payload = proof.payload
    assert payload["completionStatus"] == expected["completionStatus"]
    assert payload["claimSatisfied"] is expected["claimSatisfied"]
    assert payload["winningRootActionCount"] == 1
    assert payload["continuationLines"]["exactCount"] == expected[
        "winningContinuationLineCount"
    ]
    assert payload["principalVariation"] == expected["principalVariation"]
    assert {row["action"]: row["value"] for row in payload["rootActions"]} == expected[
        "rootValues"
    ]

    replay = state.clone()
    engine = BalootEngine(rules)
    for action in payload["principalVariation"]:
        engine.play_card(replay, Card.parse(action), emit_events=False)
    final_three = replay.completed_tricks[-3:]
    assert [trick.winner for trick in final_three] == [
        Seat.EAST,
        Seat.EAST,
        Seat.EAST,
    ]
    assert [trick.plays[-1].card.code for trick in final_three] == ["CA", "S7", "SA"]
    assert final_three[1].plays[0].card.code == "C8"
    assert final_three[2].plays[0].card.code == "C9"
    assert sorted(
        play.card.code
        for play in final_three[2].plays
        if play.seat.team == Team.NORTH_SOUTH and play.card.rank.name == "ACE"
    ) == ["DA", "SA"]


def test_principal_variation_trick_winner_predicate_replays_exact_proof(
    rules,
    root: Path,
):
    fixture = json.loads(
        (root / "examples" / "research-scenario-v4-ace-sacrifice-seven.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    proof = UnifiedScenarioProofServiceV1.from_repository(root).prove(
        normalized,
        ScenarioProofRequestV1.build(fixture["proof"]),
        bundle_hash="b" * 64,
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-TRICK-7",
                        "proof.principal_variation_trick_winner",
                        {
                            "trickIndex": 6,
                            "team": "west_east",
                            "rank": ["seven", "eight", "nine"],
                        },
                    ),
                    _predicate(
                        "PRED-TRICK-7-ACE-COUNT",
                        "proof.principal_variation_trick_team_rank_count",
                        {
                            "trickIndex": 6,
                            "team": "west_east",
                            "rank": ["ace"],
                            "minimum": 1,
                            "maximum": 1,
                        },
                    ),
                    _predicate(
                        "PRED-ROOT-ACE",
                        "proof.winning_root_card_rank_count",
                        {
                            "rank": ["ace"],
                            "minimum": 1,
                            "maximum": 1,
                        },
                    ),
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-TRICK-8",
                        "proof.principal_variation_trick_winner",
                        {
                            "trickIndex": 7,
                            "team": "west_east",
                            "rank": ["seven", "eight", "nine"],
                        },
                    ),
                    _predicate(
                        "PRED-TRICK-8-OPPONENT-ACES",
                        "proof.principal_variation_trick_team_rank_count",
                        {
                            "trickIndex": 7,
                            "team": "north_south",
                            "rank": ["ace"],
                            "minimum": 2,
                            "maximum": 2,
                        },
                    ),
                ],
            },
        }
    )
    result = ScenarioRuntimeVerifierV1().verify(
        scenario,
        normalized,
        proof_outcome=proof,
    )
    evidence = {item.predicate_id: item for item in result.evidence}
    assert evidence["PRED-TRICK-7"].status == "satisfied"
    assert evidence["PRED-TRICK-7"].observed["card"] == "C7"
    assert evidence["PRED-TRICK-7-ACE-COUNT"].status == "satisfied"
    assert evidence["PRED-TRICK-7-ACE-COUNT"].observed["matchingCards"] == [
        {"seat": "west", "card": "DA", "rank": "ace"}
    ]
    assert evidence["PRED-ROOT-ACE"].status == "satisfied"
    assert evidence["PRED-ROOT-ACE"].observed["matchingActions"] == ["DA"]
    assert evidence["PRED-TRICK-8"].status == "failed"
    assert evidence["PRED-TRICK-8"].observed["card"] == "H10"
    assert evidence["PRED-TRICK-8-OPPONENT-ACES"].status == "failed"
    assert evidence["PRED-TRICK-8-OPPONENT-ACES"].observed["count"] == 0


def test_search_candidate_skips_illegal_selected_action_without_aborting_budget(
    rules,
    root: Path,
):
    fixture = json.loads(
        (root / "examples" / "research-scenario-v4-ace-sacrifice-seven.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    proof = _proof(
        claim="selected_action_is_unique_winner",
        selectedAction="D7",
        targetTeam="west_east",
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-CONTRACT",
                        "auction.contract",
                        {"contract": "sun"},
                    )
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-COUNT",
                "proof.winning_root_action_count",
                {"minimum": 1, "maximum": 1},
            ),
        }
    )
    recipe = _exact_scenario_recipe("RCP-V4MISS01", "مرشح غير مطابق", scenario, proof)
    candidate = ScenarioSearchExecutorV1()._finish_candidate(
        recipe,
        normalized=normalized,
        bundle_hash="b" * 64,
        seed=1,
        receipt={"schemaVersion": 1, "structural": {"status": "satisfied"}},
        auction=SimpleNamespace(to_dict=lambda: {"status": "satisfied"}),
        cancel_check=None,
        policy={"id": "test", "observerSafe": True},
    )
    assert candidate.status == "semantic_postcondition_miss"
    assert candidate.proof is None
    assert candidate.satisfaction is not None
    assert candidate.construction_receipt["failure"] == (
        "proof_selected_action_not_legal"
    )


def test_golden_media_script_builds_verified_observer_safe_bundles(
    rules,
    root: Path,
):
    path = root / "scripts" / "render_research_scenario_v4_goldens.py"
    spec = importlib.util.spec_from_file_location("scenario_v4_golden_media", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    profile_hash = canonical_sha256(profile)
    renderer = RemotionProfessionalRenderer(profile, repository_root=root)
    counts = []
    public_projection_hashes = []
    for name, fixture_path in module.GOLDENS:
        bundle = module.build_golden_bundle(
            name=name,
            fixture_path=fixture_path,
            rules=rules,
            renderer=renderer,
            video_profile_hash=profile_hash,
        )
        repeated = module.build_golden_bundle(
            name=name,
            fixture_path=fixture_path,
            rules=rules,
            renderer=renderer,
            video_profile_hash=profile_hash,
        )
        assert repeated.bundle_sha256 == bundle.bundle_sha256
        assert bundle.verify() is True
        assert bundle.value["exportPreset"] == "hooked_flagship"
        assert [
            segment["stage"] for segment in bundle.value["timeline"]["segments"]
        ] == [
            "hook",
            "rewind",
            "replay",
            "question",
            "reveal",
            "continuation",
            "outcome",
        ]
        assert bundle.value["source"]["publicationEligible"] is True
        assert bundle.value["source"]["auditOnly"] is False
        assert bundle.value["source"]["qualificationFailures"] == []
        assert bundle.value["source"]["publicationBlockers"] == []
        assert [scene["stage"] for scene in bundle.value["scenes"]] == [
            "context",
            "question",
            "reveal",
            "continuation",
            "outcome",
        ]
        context = next(
            scene for scene in bundle.value["scenes"] if scene["stage"] == "continuation"
        )
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
        assert len(observer_lifecycle["initialHandCards"]) == 8
        assert len(public_lifecycle["plays"]) == 32
        reveal = next(
            scene for scene in bundle.value["scenes"] if scene["stage"] == "reveal"
        )
        analysis = next(
            layer["content"] for layer in reveal["layers"] if layer["id"] == "analysis"
        )
        counts.append(analysis["winningRootActionCount"])
        assert "exactStateId" not in bundle.value["source"]
        public_projection_hashes.append(
            bundle.value["source"]["publicProjectionHash"]
        )
        encoded = json.dumps(bundle.to_dict(), sort_keys=True)
        assert '"deckOrder"' not in encoded
        assert '"initialHands"' not in encoded
        audio_plan = build_audio_cue_plan(bundle)
        assert {item["kind"] for item in audio_plan["events"]} == {
            "card",
            "collect",
            "outcome",
            "timer",
            "reveal",
        }
    assert counts == [0, 1, 1]
    assert len(set(public_projection_hashes)) == 3
    assert module.GOLDENS[-1][1].name == (
        "research-scenario-v4-ace-sacrifice-two-aces.json"
    )


def test_golden_media_script_cli_is_source_checkout_safe(root: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(root / "scripts" / "render_research_scenario_v4_goldens.py"),
            "--help",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Research V4 golden scenarios" in result.stdout


def test_exact_proof_budget_exhaustion_returns_only_incomplete_bounds(
    rules, root: Path
):
    fixture = json.loads(
        (root / "examples" / "research-scenario-v4-unique-saving.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    budget = ScenarioProofBudgetV1.build(
        "test_one_node_v1",
        {
            "maximumNodes": 1,
            "maximumDepth": 16,
            "maximumHorizonActions": 16,
            "maximumHiddenWorlds": 1,
            "maximumWallSeconds": 10.0,
            "maximumCpuSeconds": 10.0,
            "maximumCacheEntries": 1,
            "maximumCacheBytes": 1024,
            "cancellationCheckIntervalNodes": 1,
            "maximumWorkers": 1,
            "maximumLineCountNodes": 1,
            "maximumStateSpace": 1000000,
        },
    )
    request = dict(fixture["proof"])
    request["budgetProfile"] = budget.profile_id
    outcome = UnifiedScenarioProofServiceV1(
        ScenarioProofBudgetRegistryV1(
            {budget.profile_id: budget},
            canonical_sha256({"profiles": [budget.sha256]}),
        )
    ).prove(
        normalized,
        ScenarioProofRequestV1.build(request),
        bundle_hash="b" * 64,
    )
    assert outcome.payload["completionStatus"] == "proof_incomplete"
    assert outcome.payload["achievedMode"] == "none"
    assert outcome.payload["claimSatisfied"] is False
    assert outcome.payload["continuationLines"] == {
        "complete": False,
        "lowerBound": "0",
        "upperBound": None,
        "lineUniverse": "all_legal",
    }


def ProofOutcomeRoundTrip(payload: dict) -> dict:
    from baloot_platform.research.scenario_contracts import ProofOutcomeV1

    return ProofOutcomeV1.build(payload).to_dict()


def test_actor_information_set_requires_exhaustive_indistinguishable_coverage(
    rules,
    root: Path,
):
    prefix = _late_prefix(rules)
    prefix["playActions"] = prefix["playActions"][:-4]
    normalized = ExactStateNormalizerV1(rules).normalize(prefix)
    actor = normalized.round_state.current_player
    assert actor is not None
    observation_hash = canonical_sha256(
        InformationState.from_complete(normalized.round_state, actor).to_dict()
    )
    request = _proof(
        visibility="actor_information_set",
        budgetProfile="actor_information_set_exact_v1",
    )
    service = UnifiedScenarioProofServiceV1.from_repository(root)
    automatic = service.prove(
        normalized,
        request,
        bundle_hash="b" * 64,
    )
    assert automatic.payload["completionStatus"] == "exact_complete"
    assert automatic.payload["hiddenWorlds"]["count"] > 1
    assert automatic.payload["hiddenWorlds"]["coverage"]["complete"] is True
    assert automatic.payload["strategyClass"] == "pure_deterministic"
    assert automatic.payload["hiddenWorlds"]["coverage"]["worldHashes"] == (
        automatic.payload["hiddenWorlds"]["hashes"]
    )
    with pytest.raises(ValueError, match="generated internally"):
        service.prove(
            normalized,
            request,
            bundle_hash="b" * 64,
            hidden_worlds=(normalized,),
        )
    with pytest.raises(ValueError, match="generated internally"):
        service.prove(
            normalized,
            request,
            bundle_hash="b" * 64,
            hidden_worlds=(normalized,),
            hidden_world_coverage={
                "method": "exhaustive_reference_fixture",
                "complete": True,
                "expectedWorldCount": 1,
                "observationHash": observation_hash,
            },
        )


def test_bounded_and_policy_conditioned_proofs_keep_their_labels(rules, root: Path):
    normalized = ExactStateNormalizerV1(rules).normalize(_late_prefix(rules))
    service = UnifiedScenarioProofServiceV1.from_repository(root)
    bounded = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "bounded",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -1_000_000,
            "horizon": {"unit": "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1",
            "budgetProfile": "bounded_v1",
        }
    )
    bounded_outcome = service.prove(normalized, bounded, bundle_hash="c" * 64)
    assert bounded_outcome.payload["completionStatus"] == "bounded_complete"
    assert bounded_outcome.payload["achievedMode"] == "bounded"

    policies = {}
    for index, seat in enumerate(("north", "west", "south", "east")):
        core = {"agentId": "heuristic_bot_v1", "version": "1.0.0", "seed": index + 1}
        policies[seat] = {**core, "profileHash": canonical_sha256(core)}
    policy = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -1_000_000,
            "policies": policies,
            "budgetProfile": "bounded_v1",
        }
    )
    policy_outcome = service.prove(normalized, policy, bundle_hash="d" * 64)
    assert policy_outcome.payload["completionStatus"] == "policy_complete"
    assert policy_outcome.payload["achievedMode"] == "policy_conditioned"
    assert policy_outcome.payload["continuationLines"] == {
        "complete": True,
        "exactCount": "1",
        "lineUniverse": "policy_conditioned",
    }

    false_claim = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": 1_000_000,
            "budgetProfile": "late_game_exact_v1",
        }
    )
    failed = service.prove(normalized, false_claim, bundle_hash="d" * 64)
    assert failed.payload["completionStatus"] == "proof_failed"
    assert failed.payload["claimSatisfied"] is False


def test_policy_continuation_count_is_for_selected_root_not_all_roots(
    rules,
    root: Path,
):
    fixture = json.loads(
        (root / "examples" / "research-scenario-v4-unique-saving.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    policies = {}
    for index, seat in enumerate(("north", "west", "south", "east")):
        core = {
            "agentId": "heuristic_bot_v1",
            "version": "1.0.0",
            "seed": index + 101,
        }
        policies[seat] = {**core, "profileHash": canonical_sha256(core)}
    request = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -1_000_000,
            "policies": policies,
            "lineUniverse": "policy_conditioned",
            "budgetProfile": "bounded_v1",
        }
    )
    outcome = UnifiedScenarioProofServiceV1.from_repository(root).prove(
        normalized,
        request,
        bundle_hash="d" * 64,
    )
    assert len(outcome.payload["rootActions"]) == 2
    assert outcome.payload["winningRootActionCount"] == 2
    assert outcome.payload["continuationLines"]["exactCount"] == "1"


def test_runtime_satisfaction_requires_real_state_and_proof_evidence(rules, root: Path):
    normalized = ExactStateNormalizerV1(rules).normalize(_late_prefix(rules))
    target_team = (
        "north_south"
        if normalized.round_state.current_player.team.value == 0
        else "west_east"
    )
    request = _proof(targetTeam=target_team)
    proof = UnifiedScenarioProofServiceV1.from_repository(root).prove(
        normalized,
        request,
        bundle_hash="e" * 64,
    )
    count = int(proof.payload["winningRootActionCount"])
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
                        {"seat": normalized.next_actor},
                    ),
                    _predicate(
                        "PRED-LEGAL",
                        "state.legal_action_count",
                        {
                            "minimum": len(normalized.legal_actions),
                            "maximum": len(normalized.legal_actions),
                        },
                    ),
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-PROOF",
                "proof.winning_root_action_count",
                {"minimum": count, "maximum": count},
            ),
        }
    )
    satisfied = ScenarioRuntimeVerifierV1().verify(
        scenario,
        normalized,
        proof_outcome=proof,
    )
    assert satisfied.status == "satisfied"
    assert satisfied.hard_predicates_satisfied is True
    missing = ScenarioRuntimeVerifierV1().verify(scenario, normalized)
    assert missing.status == "semantic_postcondition_miss"
    proof_row = next(
        item for item in missing.evidence if item.kind.startswith("proof.")
    )
    assert proof_row.status == "not_evaluated"


def test_runtime_verifies_extended_observable_predicate_inventory(rules):
    prefix = _late_prefix(rules)
    prefix["playActions"] = prefix["playActions"][:-1]
    normalized = ExactStateNormalizerV1(rules).normalize(prefix)
    state = normalized.round_state
    assert state.current_player is not None
    assert state.current_trick is not None
    assert state.current_trick.led_suit is not None
    assert state.completed_tricks and state.completed_tricks[0].winner is not None
    legal_card = int(normalized.legal_actions[0]["card"]["id"])
    owner = state.current_player.name.lower()
    trick = state.completed_tricks[0]
    constraints = [
        _predicate(
            "PRED-REMAINING-OWNER",
            "deal.remaining_card_owner",
            {"card": legal_card, "seat": owner},
        ),
        _predicate(
            "PRED-HIGH-BALANCE",
            "deal.team_high_card_balance",
            {
                "team": "north_south",
                "maximumDifference": 8,
                "handStage": "post_purchase",
            },
        ),
        _predicate(
            "PRED-PASSES",
            "auction.pass_count",
            {"minimum": 0, "maximum": 8},
        ),
        _predicate("PRED-AUCTION-END", "auction.terminal", {"terminal": True}),
        _predicate(
            "PRED-ESCALATION-LOCK",
            "escalation.locked",
            {"locked": state.locked},
        ),
        _predicate(
            "PRED-BALOOT",
            "project.baloot",
            {
                "team": "north_south",
                "achieved": Team.NORTH_SOUTH in state.baloot_teams,
            },
        ),
        _predicate(
            "PRED-OVERLAP",
            "project.overlap_count",
            {"owner": "north", "minimum": 0, "maximum": 32},
        ),
        _predicate(
            "PRED-LEADER",
            "play.current_trick_leader",
            {"seat": state.current_trick.leader.name.lower()},
        ),
        _predicate(
            "PRED-LED-SUIT",
            "play.current_trick_suit_led",
            {"suit": state.current_trick.led_suit.name.lower()},
        ),
        _predicate(
            "PRED-CURRENT-TRICK-CARD",
            "play.current_trick_card",
            {
                "position": 0,
                "seat": state.current_trick.plays[0].seat.name.lower(),
                "card": state.current_trick.plays[0].card.id,
            },
        ),
        _predicate(
            "PRED-TRICK-WINNER",
            "play.completed_trick_winner",
            {
                "trickIndex": trick.index,
                "seat": trick.winner.name.lower(),
            },
        ),
        _predicate(
            "PRED-RAW-NOW",
            "state.current_raw_points",
            {"team": "north_south", "minimum": 0, "maximum": 200},
        ),
        _predicate(
            "PRED-PROJECT-NOW",
            "state.current_project_points",
            {"team": "north_south", "minimum": 0, "maximum": 500},
        ),
        _predicate(
            "PRED-LEGAL-INCLUDE",
            "state.legal_cards_include",
            {"cards": [legal_card]},
        ),
        _predicate(
            "PRED-TACTIC-APPLICABLE",
            "tactic.applicable",
            {"kind": "small_card_big_gain", "applicable": True},
        ),
    ]
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "exact_trace_prefix", "exactTracePrefix": prefix},
            "constraints": {"op": "all", "children": constraints},
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-LEGAL-COUNT",
                "state.legal_action_count",
                {
                    "minimum": len(normalized.legal_actions),
                    "maximum": len(normalized.legal_actions),
                },
            ),
        }
    )
    plan = ScenarioCompilerV1().compile(scenario, _proof())
    assert plan.valid is True
    result = ScenarioRuntimeVerifierV1().verify(
        scenario,
        normalized,
        tactical_evidence={"small_card_big_gain": [{"applicable": True}]},
    )
    assert result.status == "satisfied"
    assert all(item.status == "satisfied" for item in result.evidence)
    assert all(
        row["requiredDomainPrimitives"]
        for row in ScenarioPredicateRegistryV1().catalog()["definitions"]
        if row["stage"]
        in {"deal", "auction", "escalation", "projects", "playPrefix", "outcome"}
    )


def test_structural_solver_proves_feasible_deal_and_exact_infeasible_core(rules):
    compiler = ScenarioStructuralCompilerV1()
    feasible = _scenario(
        _predicate(
            "PRED-CARD-N",
            "deal.card_owner",
            {"card": 0, "seat": "north", "handStage": "initial_five"},
        ),
        _predicate(
            "PRED-SUIT-N",
            "deal.suit_count",
            {
                "seat": "north",
                "suit": "spades",
                "minimum": 2,
                "maximum": 8,
                "handStage": "initial_five",
            },
        ),
    )
    result = compiler.solve(
        feasible,
        rules=rules,
        dealer=Seat.EAST,
        seed=91,
        maximum_seconds=10,
    )
    assert result.status == "satisfied"
    assert result.solver_status == "optimal"
    assert result.deck_order is not None
    engine = BalootEngine(rules)
    state = engine.create_round(seed=91, dealer=Seat.EAST)
    engine.initial_deal_from_preordered(state, result.deck_order)
    assert state.initial_five[int(Seat.NORTH)] & 1

    continued = compiler.solve(
        feasible,
        rules=rules,
        dealer=Seat.EAST,
        seed=91,
        maximum_seconds=10,
        excluded_deck_orders=[result.deck_order],
    )
    assert continued.status == "satisfied"
    assert continued.deck_order is not None
    assert continued.deck_order != result.deck_order
    assert continued.no_good_hashes == (canonical_sha256(list(result.deck_order)),)
    assert continued.to_dict()["noGoodHashes"] == list(continued.no_good_hashes)

    impossible = _scenario(
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
        _predicate(
            "PRED-IRRELEVANT",
            "deal.card_owner",
            {"card": 1, "seat": "east", "handStage": "initial_five"},
        ),
    )
    failed = compiler.solve(
        impossible,
        rules=rules,
        dealer=Seat.EAST,
        seed=92,
        maximum_seconds=10,
    )
    assert failed.status == "structurally_infeasible"
    assert failed.solver_status == "infeasible"
    assert failed.deck_order is None
    assert failed.infeasible_core_predicate_ids == (
        "PRED-CARD-N",
        "PRED-CARD-S",
    )


def test_structural_solver_prefers_actor_safe_buffers_for_protected_cards(rules):
    protected = Card.parse("SA")
    scenario = _scenario(
        _predicate("PRED-BUYER", "auction.buyer", {"seat": "south"}),
        _predicate("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
        _predicate(
            "PRED-OWNER",
            "deal.card_owner",
            {
                "card": protected.id,
                "seat": "north",
                "handStage": "post_purchase",
            },
        ),
        _predicate(
            "PRED-REMAINING",
            "deal.remaining_card_owner",
            {"card": protected.id, "seat": "north"},
        ),
    )
    result = ScenarioStructuralCompilerV1().solve(
        scenario,
        rules=rules,
        dealer=Seat.EAST,
        seed=93,
        maximum_seconds=10,
    )
    assert result.status == "satisfied"
    assert result.deck_order is not None
    layout = DealLayoutV1.build(
        rules.structural_domain,
        dealer=Seat.EAST,
        buyer=Seat.SOUTH,
    )
    north = [
        Card(result.deck_order[position])
        for position in layout.positions_for(Seat.NORTH, final=True)
    ]
    assert protected in north
    assert sum(card.suit == protected.suit for card in north) >= 2


def test_structural_solver_guides_multiple_engine_verified_projects(rules):
    scenario = _scenario(
        _predicate("PRED-BUYER", "auction.buyer", {"seat": "south"}),
        _predicate("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
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
            "PRED-SECOND-PROJECT",
            "project.exists",
            {
                "team": "north_south",
                "owner": "north",
                "kind": "four_hundred",
                "value": 40,
                "state": "eligible",
            },
        ),
    )
    compiler = ScenarioStructuralCompilerV1()
    result = compiler.solve(
        scenario,
        rules=rules,
        dealer=Seat.EAST,
        seed=100,
        maximum_seconds=1,
    )
    assert result.status == "satisfied"
    assert result.solver_status == "optimal"
    repeated = compiler.solve(
        scenario,
        rules=rules,
        dealer=Seat.EAST,
        seed=100,
        maximum_seconds=1,
    )
    varied = compiler.solve(
        scenario,
        rules=rules,
        dealer=Seat.EAST,
        seed=101,
        maximum_seconds=1,
    )
    assert repeated.deck_order == result.deck_order
    assert varied.deck_order != result.deck_order
    project_evidence = next(
        item
        for item in result.predicate_evidence
        if item["predicateId"] == "PRED-PROJECT"
    )
    assert project_evidence["lowering"] == "cp_sat_exact_eligible_project_pattern"

    engine = BalootEngine(rules)
    state = engine.create_round(seed=100, dealer=Seat.EAST)
    engine.initial_deal_from_preordered(state, result.deck_order)
    assert state.purchase_card is not None
    machine = BiddingMachine(
        state.dealer,
        state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(rules.data),
    )
    auction = OrToolsCpSatAuctionCompilerV1().compile(
        dealer=state.dealer,
        purchase_suit=state.purchase_card.suit,
        rules=rules,
        required_buyer=Seat.SOUTH,
        required_contracts=("sun",),
    )
    assert auction is not None
    for action in auction.actions:
        machine.submit(action.action, trump=action.trump)
    assert machine.result is not None
    engine.complete_deal_from_bid_result(state, machine.result)
    engine.prepare_projects(state)
    assert any(
        project.owner == Seat.SOUTH
        and project.kind == "hundred_sequence"
        and project.value == 20
        for project in state.projects
    )
    assert any(
        project.owner == Seat.NORTH
        and project.kind == "four_hundred"
        and project.value == 40
        for project in state.projects
    )


def test_v4_rules_applicability_rejects_unknown_project_before_execution(rules):
    request = _proof()
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-CONTRACT",
                        "auction.contract",
                        {"contract": "sun"},
                    ),
                    _predicate(
                        "PRED-UNKNOWN-PROJECT",
                        "project.exists",
                        {
                            "team": "north_south",
                            "kind": "imaginary_hundred",
                            "value": 100,
                            "state": "eligible",
                        },
                    ),
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-PROOF",
                "proof.winning_root_action_count",
                {"minimum": 0, "maximum": 8},
            ),
        }
    )
    plan = ScenarioCompilerV1().compile(scenario, request)
    recipe = ResearchRecipeV4(
        "RCP-V4RULES1",
        "Ø³ÙŠÙ†Ø§Ø±ÙŠÙˆ ÙŠØ±ÙØ¶ Ù…Ø´Ø±ÙˆØ¹Ù‹Ø§ ØºÙŠØ± Ù…Ø¯Ø¹ÙˆÙ…",
        {"bundleRevisionId": "CBR-TEST"},
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
    issues = ResearchFoundryServiceV1._scenario_rules_applicability_issues(
        recipe,
        rules,
    )
    assert [item["code"] for item in issues] == [
        "project_kind_value_not_enabled_by_rules"
    ]
    assert issues[0]["predicateIds"] == ["PRED-UNKNOWN-PROJECT"]


class _ScenarioArtifactStore:
    def __init__(self):
        self.values = {}

    def put(self, data, *, media_type, name=None):
        digest = hashlib.sha256(data).hexdigest()
        uri = f"cas://sha256/{digest}"
        self.values[uri] = bytes(data)
        return {
            "uri": uri,
            "sha256": digest,
            "bytes": len(data),
            "mediaType": media_type,
            "name": name,
        }

    def get(self, uri):
        return self.values[uri]


class _ScenarioRunStore:
    def __init__(self, run):
        self.run = dict(run)
        self.results = []

    def update_run(self, run_code, **changes):
        assert run_code == self.run["runCode"]
        self.run.update(changes)
        return dict(self.run)

    def get_run(self, run_code):
        assert run_code == self.run["runCode"]
        return dict(self.run)

    def put_result(self, value):
        self.results.append(dict(value))
        return dict(value)


def _run_exact_scenario_recipe(rules, recipe: ResearchRecipeV4):
    configuration, context = _published_v4_configuration()
    assert recipe.configuration == configuration
    assert context.bundle.rules.sha256 == rules.sha256
    run = {
        "runCode": "RUN-" + recipe.code.removeprefix("RCP-")[:12].ljust(12, "0"),
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
    return harness.store.results[0], harness


def _exact_scenario_recipe(
    code: str,
    title: str,
    scenario: ResearchScenarioV1,
    proof: ScenarioProofRequestV1,
) -> ResearchRecipeV4:
    configuration, _context = _published_v4_configuration()
    return ResearchRecipeV4(
        code,
        title,
        configuration,
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
            "solverTimeoutSeconds": 1.0,
            "maximumConflicts": 1,
            "analysisMode": "cascaded",
            "proofBudgetProfile": proof.budget_profile,
        },
        {},
        {"mode": "on_demand"},
        "published",
        ScenarioCompilerV1().compile(scenario, proof),
        "f" * 64,
    )


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


@pytest.mark.real_clock
def test_v4_exact_public_api_journey_retains_and_queries_proof(root):
    revision_id = _published_ortools_v4_revision_id()
    _authority, context = api.RESEARCH.execution_context_factory.resolve_bundle_revision(revision_id)
    prefix = _api_budgeted_quality_prefix(root)
    prefix["playActions"] = prefix["playActions"][:28]
    rules = context.bundle.rules
    normalized = ExactStateNormalizerV1(rules).normalize(prefix)
    target_team = (
        "north_south"
        if normalized.round_state.current_player.team.value == 0
        else "west_east"
    )
    payload = {
        "schemaVersion": 4,
        "code": "RCP-V4APIE2E3",
        "playingQuality": _api_budgeted_quality_policy(),
        "titleAr": "سيناريو دقيق عبر واجهة البحث",
            "configuration": {
                "schemaVersion": 1,
                "bundleRevisionId": revision_id,
        },
        "scenario": {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": prefix,
                "recordedEscalationOpportunities": [],
            },
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-API-ACTOR",
                        "play.current_player",
                        {"seat": normalized.next_actor},
                    ),
                    _predicate(
                        "PRED-API-LEGAL",
                        "state.legal_action_count",
                        {
                            "minimum": len(normalized.legal_actions),
                            "maximum": len(normalized.legal_actions),
                        },
                    ),
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-API-PROOF",
                "proof.winning_root_action_count",
                {"minimum": 0, "maximum": len(normalized.legal_actions)},
            ),
        },
        "proof": {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "claim": "winning_root_action_count",
            "targetTeam": target_team,
            "utility": "awarded_point_margin",
            "successThreshold": 1,
            "lineUniverse": "all_legal",
            "budgetProfile": "offline_heavyweight_v1",
        },
            "resources": {
                "generationLane": "ortools",
                "generationPolicy": "ortools_cp_sat_v3",
                "seedStart": 1,
                "seedCount": 1,
                "seedsPerShard": 1,
                "solverTimeoutSeconds": _API_CP_SAT_BUDGET_SECONDS,
                "proofBudgetProfile": "offline_heavyweight_v1",
            },
        "status": "published",
    }
    with TestClient(api.app) as client:
        validated = client.post("/api/v2/research-recipes:validate", json=payload)
        assert validated.status_code == 200, validated.text
        assert validated.json()["valid"] is True, json.dumps(
            validated.json()["issues"], ensure_ascii=False, indent=2
        )
        assert validated.json()["recipe"]["schemaVersion"] == 4
        assert validated.json()["recipe"]["output"] == {
            "kind": "laffa",
            "resultCount": 1,
        }
        assert validated.json()["recipe"]["renderPolicy"] == {
            "mode": "on_demand",
            "videoCount": 1,
        }
        assert validated.json()["normalizedExactState"]["exactStateId"].startswith(
            "STATE-"
        )

        estimated = client.post("/api/v2/research-recipes:estimate", json=payload)
        assert estimated.status_code == 200, estimated.text
        assert estimated.json()["schemaVersion"] == 2
        assert estimated.json()["valid"] is True
        assert estimated.json()["proof"]["requestedMode"] == "exact"

        created = client.post(
            "/api/v2/research-recipes",
            json=payload,
            headers={"Idempotency-Key": "research-v4-api-e2e3-create"},
        )
        assert created.status_code == 201, created.text
        assert created.json()["schemaVersion"] == 4
        assert created.json()["output"]["resultCount"] == 1
        assert created.json()["renderPolicy"]["videoCount"] == 1

        submitted = client.post(
            "/api/v2/research-runs",
            json={"recipeCode": created.json()["code"]},
            headers={"Idempotency-Key": "research-v4-api-e2e3-run"},
        )
        assert submitted.status_code == 202, submitted.text
        submitted_run = submitted.json()
        assert submitted_run["requestedResultCount"] == 1
        assert submitted_run["requestedVideoCount"] == 1
        admission = submitted_run["launchAdmission"]
        assert admission["recipeHash"] == created.json()["sha256"]
        assert admission["estimate"]["launchAdmissible"] is True
        assert admission["estimate"]["proof"]["admissible"] is True
        assert admission["estimate"]["configurationAuthority"] == created.json()[
            "configuration"
        ]
        _assert_public_launch_admission(admission, submitted_run["runCode"])
        run_code = submitted_run["runCode"]
        run = submitted_run
        deadline = time.monotonic() + 2 * admission["estimate"]["maximumCampaignSeconds"]
        while time.monotonic() < deadline:
            run_response = client.get(f"/api/v2/research-runs/{run_code}")
            assert run_response.status_code == 200, run_response.text
            run = run_response.json()
            if run["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.1)
        assert run["status"] == "completed", run
        assert run["campaignAccounting"]["maximumCampaignSeconds"] == _API_CP_SAT_BUDGET_SECONDS
        assert run["resultCount"] == 1
        assert run["qualifiedCount"] == 1
        assert run["selectedCount"] == 1
        assert run["outputReceipt"]["resultTargetSatisfied"] is True
        assert run["outputReceipt"]["sha256"] == canonical_sha256(
            {
                key: value
                for key, value in run["outputReceipt"].items()
                if key != "sha256"
            }
        )

        result_page = client.get(
            "/api/v2/research-results",
            params={
                "runCode": run_code,
                "scenarioOrigin": "exact_trace_prefix",
                "scenarioStatus": "satisfied",
                "proofCompletion": "exact_complete",
                "exactStateAvailable": True,
            },
        )
        assert result_page.status_code == 200, result_page.text
        assert result_page.json()["total"] == 1
        result_code = result_page.json()["items"][0]["resultCode"]

        result_response = client.get(f"/api/v2/research-results/{result_code}")
        assert result_response.status_code == 200, result_response.text
        result = result_response.json()
        assert result["proofOutcome"]["completionStatus"] == "exact_complete"
        assert result["scenarioProjection"]["exactStateAvailable"] is True
        assert result["scenarioProjection"]["predicateEvidenceCount"] == 3
        assert result["decisionReference"] == {
            "resultCode": result_code,
            "decisionOrdinal": 1,
        }
        assert "scenarioSatisfaction" not in result
        assert "exactStateId" not in result["scenarioProjection"]

        prepared = api.RESEARCH_RENDERS.prepare_selected_run(
            run_code,
            reveal_policy="after_commit",
            export_preset="decision_clip",
        )
        assert prepared["runCode"] == run_code
        assert prepared["runHash"] == run["runHash"]
        assert prepared["recipeHash"] == created.json()["sha256"]
        assert prepared["outputReceiptHash"] == run["outputReceipt"]["sha256"]
        assert prepared["diversitySelectionHash"] == run["diversitySelection"][
            "sha256"
        ]
        assert prepared["requestedVideoCount"] == 1
        assert prepared["selectedResultCodes"] == [result_code]
        assert len(prepared["items"]) == 1
        assert prepared["items"][0]["source"]["code"] == result_code
        assert prepared["items"][0]["locator"] == {
            "schemaVersion": 1,
            "code": result_code,
            "universalLink": f"/c/{result_code}",
        }
        assert prepared["items"][0]["proofArtifact"]["sha256"] == prepared[
            "items"
        ][0]["authorityReceipt"]["proofArtifactSha256"]
        assert prepared["sha256"] == canonical_sha256(
            {key: value for key, value in prepared.items() if key != "sha256"}
        )

        actor = validated.json()["normalizedExactState"]["nextActor"]
        exact_state_response = client.get(
            f"/api/v2/research-results/{result_code}/exact-state",
            params={"observerSeat": actor},
        )
        assert exact_state_response.status_code == 200, exact_state_response.text
        exact_state_view = exact_state_response.json()
        assert exact_state_view["observerSeat"] == actor
        assert exact_state_view["decisionReference"] == result["decisionReference"]
        assert exact_state_view["publicProjectionHash"] == result[
            "publicProjectionHash"
        ]
        assert "exactStateId" not in exact_state_view
        assert "stateHash" not in exact_state_view
        assert (
            exact_state_view["legalActions"]
            == validated.json()["normalizedExactState"]["legalActions"]
        )
        for seat, hand in exact_state_view["publicState"]["hands"].items():
            if seat == actor:
                assert isinstance(hand, list)
            else:
                assert set(hand) == {"count"}
        other = next(
            seat for seat in ("north", "west", "south", "east") if seat != actor
        )
        refused_view = client.get(
            f"/api/v2/research-results/{result_code}/exact-state",
            params={"observerSeat": other},
        )
        assert refused_view.status_code == 409

        decisions = client.get(f"/api/v2/research-results/{result_code}/decisions")
        assert decisions.status_code == 200, decisions.text
        assert len(decisions.json()["items"]) == 1
        assert decisions.json()["items"][0]["proof"]["achievedMode"] == "exact"


def test_estimate_v2_refuses_unsafe_exact_search_before_run_creation():
    authority, _context = _publish_explicit_bundle(
        api.CONTAINER, Path(__file__).parents[1], {"bundle": {
            "id": "baloot_ortools_research_native_v3",
            "source": "configs/bundle_intelligent_research.yaml",
            "backend": "ortools_cp_sat_v3",
        }},
    )
    revision_id = authority.bundle_revision_id
    recipe_code = "RCP-V4ADMISSION1"
    payload = {
        "schemaVersion": 4,
        "code": recipe_code,
        "playingQuality": {"schemaVersion": 1, "mode": "budgeted", "decisions": {
            kind: {
                "utility": "awarded_point_margin", "budgetProfile": "bounded_v1",
                "horizon": {"unit": "tricks" if kind == "play" else "plies", "value": 1},
                "boundaryEvaluator": "auction_hand_margin_v1" if kind == "bid" else "current_raw_point_margin_v1",
            } for kind in SCENARIO_DECISION_KINDS
        }},
        "titleAr": "رفض بحث دقيق يتجاوز ميزانية الإثبات",
        "configuration": {
            "schemaVersion": 1,
            "bundleRevisionId": revision_id,
        },
        "scenario": {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-ADMISSION-CONTRACT",
                        "auction.contract",
                        {"contract": "sun"},
                    )
                ],
            },
            "decisionPoint": {
                "kind": "play",
                "selector": {
                    "remainingCards": 6,
                    "trickIndex": 2,
                    "actingSeat": "south",
                },
            },
            "outcome": _predicate(
                "PRED-ADMISSION-PROOF",
                "proof.winning_root_action_count",
                {"minimum": 1, "maximum": 1},
            ),
        },
        "proof": {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "claim": "winning_root_action_count",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": 1,
            "lineUniverse": "all_legal",
            "budgetProfile": "late_game_exact_v1",
        },
            "resources": {
                "generationLane": "ortools",
                "generationPolicy": "ortools_cp_sat_v3",
                "seedStart": 1,
                "seedCount": 1,
                "seedsPerShard": 1,
                "solverTimeoutSeconds": 1.0,
                "proofBudgetProfile": "late_game_exact_v1",
            },
        "status": "published",
    }

    with TestClient(api.app) as client:
        estimated_response = client.post(
            "/api/v2/research-recipes:estimate", json=payload
        )
        assert estimated_response.status_code == 200, estimated_response.text
        estimated = estimated_response.json()
        assert estimated["valid"] is True, estimated["issues"]
        assert estimated["launchAdmissible"] is False
        assert estimated["proof"]["admissible"] is False
        assert estimated["proof"]["strategyClass"] == "pure_deterministic"
        assert estimated["proof"]["reason"] == "requested_depth_exceeds_proof_budget"
        assert estimated["proof"]["recommendedModes"] == []
        assert estimated["configurationAuthority"]["bundleRevisionId"] == revision_id
        assert estimated["estimateHash"] == canonical_sha256(
            {key: value for key, value in estimated.items() if key != "estimateHash"}
        )

        admitted_payload = json.loads(json.dumps(payload))
        admitted_payload["scenario"]["decisionPoint"]["selector"][
            "remainingCards"
        ] = 2
        admitted_payload["scenario"]["decisionPoint"]["selector"][
            "trickIndex"
        ] = 6
        admitted_response = client.post(
            "/api/v2/research-recipes:estimate", json=admitted_payload
        )
        assert admitted_response.status_code == 200, admitted_response.text
        admitted = admitted_response.json()
        assert admitted["launchAdmissible"] is True
        assert admitted["proof"]["admissible"] is True
        assert admitted["proof"]["recommendedModes"] == []
        assert client.post(
            "/api/v2/research-recipes:estimate", json=admitted_payload
        ).json() == admitted

        policy_payload = json.loads(json.dumps(admitted_payload))
        policy_payload["proof"].update(
            {
                "mode": "policy_conditioned",
                "strategyClass": "frozen_policy",
                "lineUniverse": "policy_conditioned",
                "policies": ResearchFoundryServiceV1._scenario_recipe_configuration()[
                    "defaultPolicyProfiles"
                ],
            }
        )
        policy_estimate = client.post(
            "/api/v2/research-recipes:estimate", json=policy_payload
        ).json()
        assert policy_estimate["launchAdmissible"] is False
        assert policy_estimate["proof"]["strategyClass"] == "frozen_policy"
        assert policy_estimate["proof"]["requestedDepth"] == 8
        assert policy_estimate["proof"]["reason"] == "policy_profile_invalid"
        assert "estimatedTreeStateUpperBound" not in policy_estimate["proof"]
        assert "informationTreeExceedsBudget" not in policy_estimate["proof"]

        unsafe_policy_payload = json.loads(json.dumps(policy_payload))
        unsafe_policy_payload["scenario"]["decisionPoint"]["selector"][
            "remainingCards"
        ] = 6
        unsafe_policy_payload["scenario"]["decisionPoint"]["selector"][
            "trickIndex"
        ] = 2
        unsafe_policy_estimate = client.post(
            "/api/v2/research-recipes:estimate", json=unsafe_policy_payload
        ).json()
        assert unsafe_policy_estimate["launchAdmissible"] is False
        assert (
            unsafe_policy_estimate["proof"]["reason"]
            == "policy_profile_invalid"
        )

        output_budget_payload = json.loads(json.dumps(admitted_payload))
        output_budget_payload["code"] = "RCP-V4OUTPUTBUDGET"
        output_budget_payload["output"] = {
            "kind": "laffa_collection",
            "resultCount": 2,
        }
        output_budget_estimate = client.post(
            "/api/v2/research-recipes:estimate", json=output_budget_payload
        ).json()
        assert output_budget_estimate["proof"]["admissible"] is True
        assert output_budget_estimate["launchAdmissible"] is False
        assert output_budget_estimate["output"] == {
            "kind": "laffa_collection",
            "requestedResultCount": 2,
            "requestedVideoCount": 1,
            "maximumCandidateCount": 10_000,
            "maximumVariantCount": 100,
            "admissible": False,
            "reason": "requested_result_count_exceeds_candidate_budget",
            "recommendedResultCount": 1,
            "requiresResourceExpansion": True,
        }
        output_budget_created = client.post(
            "/api/v2/research-recipes",
            json=output_budget_payload,
            headers={"Idempotency-Key": "research-v4-output-budget-create"},
        )
        assert output_budget_created.status_code == 201, output_budget_created.text
        output_budget_run = client.post(
            "/api/v2/research-runs",
            json={"recipeCode": output_budget_payload["code"]},
            headers={"Idempotency-Key": "research-v4-output-budget-run"},
        )
        assert output_budget_run.status_code == 409, output_budget_run.text
        assert (
            output_budget_run.json()["detail"]["reason"]
            == "requested_result_count_exceeds_candidate_budget"
        )
        assert output_budget_run.json()["detail"]["recommendedModes"] == []
        assert output_budget_run.json()["detail"]["output"] == output_budget_estimate[
            "output"
        ]

        actor_payload = json.loads(json.dumps(admitted_payload))
        actor_payload["proof"].pop("lineUniverse")
        actor_payload["proof"]["visibility"] = "actor_information_set"
        actor_payload["proof"]["strategyClass"] = "pure_deterministic"
        actor_payload["proof"][
            "budgetProfile"
        ] = "actor_information_set_exact_v1"
        actor_payload["resources"][
            "proofBudgetProfile"
        ] = "actor_information_set_exact_v1"
        actor_estimate = client.post(
            "/api/v2/research-recipes:estimate", json=actor_payload
        ).json()
        assert actor_estimate["launchAdmissible"] is True
        assert actor_estimate["proof"]["admissible"] is True
        assert actor_estimate["proof"]["recommendedModes"] == []

        policy_space_payload = json.loads(json.dumps(actor_payload))
        policy_space_payload["proof"][
            "budgetProfile"
        ] = "actor_information_set_exact_v1"
        policy_space_payload["resources"][
            "proofBudgetProfile"
        ] = "actor_information_set_exact_v1"
        policy_space_response = client.post(
            "/api/v2/research-recipes:estimate", json=policy_space_payload
        )
        assert policy_space_response.status_code == 200, policy_space_response.text
        policy_space = policy_space_response.json()
        policy_forecast = policy_space["proof"]
        assert policy_space["launchAdmissible"] is True
        assert policy_forecast["strategyClass"] == "pure_deterministic"
        assert policy_forecast["estimatedHiddenWorldUpperBound"] == 90
        assert policy_forecast["actionDomainUpperBound"] == 2
        assert policy_forecast["estimatedEkkaActionVariantUpperBound"] == 0
        assert policy_forecast["ekkaDeclarationSequenceUpperBound"] == "not_compiled"
        assert "targetInformationSetUpperBound" not in policy_forecast
        assert "opponentInformationSetUpperBound" not in policy_forecast
        assert "targetPolicyCountUpperBound" not in policy_forecast
        assert "opponentPolicyCountUpperBound" not in policy_forecast
        assert "policyProductUpperBound" not in policy_forecast
        assert "estimatedTreeStateUpperBound" not in policy_forecast
        assert "informationTreeExceedsBudget" not in policy_forecast
        assert "policyProductExceedsBudget" not in policy_forecast
        assert policy_forecast.get("reason") is None
        assert policy_forecast["recommendedModes"] == []

        hokum_payload = json.loads(json.dumps(policy_space_payload))
        hokum_payload["scenario"]["constraints"]["children"][0]["arguments"][
            "contract"
        ] = "hokum"
        hokum_response = client.post(
            "/api/v2/research-recipes:estimate", json=hokum_payload
        )
        assert hokum_response.status_code == 200, hokum_response.text
        hokum_forecast = hokum_response.json()["proof"]
        assert hokum_forecast["actionDomainUpperBound"] == 2
        assert hokum_forecast["estimatedEkkaActionVariantUpperBound"] == 0
        assert hokum_forecast["ekkaDeclarationSequenceUpperBound"] == "not_compiled"
        assert "policyProductExceedsBudget" not in hokum_forecast

        bounded_payload = json.loads(json.dumps(payload))
        bounded_payload["proof"].update(
            {
                "mode": "bounded",
                "horizon": {"unit": "tricks", "value": 5},
                "boundaryEvaluator": "current_raw_point_margin_v1",
            }
        )
        bounded_estimate = client.post(
            "/api/v2/research-recipes:estimate", json=bounded_payload
        ).json()
        assert bounded_estimate["launchAdmissible"] is False
        assert (
            bounded_estimate["proof"]["reason"]
            == "requested_depth_exceeds_proof_budget"
        )
        assert bounded_estimate["proof"]["recommendedModes"] == []

        mismatched_payload = json.loads(json.dumps(admitted_payload))
        mismatched_payload["resources"]["proofBudgetProfile"] = "bounded_v1"
        mismatched_estimate = client.post(
            "/api/v2/research-recipes:estimate", json=mismatched_payload
        ).json()
        assert mismatched_estimate["launchAdmissible"] is False
        assert (
            mismatched_estimate["proof"]["reason"]
            == "proof_budget_binding_mismatch"
        )
        assert mismatched_estimate["proof"]["recommendedModes"] == []

        created = client.post(
            "/api/v2/research-recipes",
            json=payload,
            headers={"Idempotency-Key": "research-v4-admission-create"},
        )
        assert created.status_code == 201, created.text
        frozen_authority = created.json()["configuration"]
        assert not any(
            item["recipeCode"] == recipe_code
            for item in api.RESEARCH.list_runs(limit=10_000)
        )

        submitted = client.post(
            "/api/v2/research-runs",
            json={"recipeCode": recipe_code},
            headers={"Idempotency-Key": "research-v4-admission-run"},
        )
        assert submitted.status_code == 409, submitted.text
        error = submitted.json()
        api.ResearchLaunchAdmissionErrorResponse.model_validate(error)
        diagnostic = error["detail"]
        assert diagnostic["reason"] == "requested_depth_exceeds_proof_budget"
        assert diagnostic["recommendedModes"] == []
        assert diagnostic["configurationAuthority"] == frozen_authority
        assert diagnostic["estimateHash"] == estimated["estimateHash"]
        assert diagnostic["sha256"] == canonical_sha256(
            {key: value for key, value in diagnostic.items() if key != "sha256"}
        )
        assert not any(
            item["recipeCode"] == recipe_code
            for item in api.RESEARCH.list_runs(limit=10_000)
        )


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


def _retained_late_play_scenario(rules):
    prefix = _late_prefix(rules)
    normalized = ExactStateNormalizerV1(rules).normalize(prefix)
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": prefix,
            },
            "constraints": _predicate(
                "PRED-RETAINED-ACTOR",
                "play.current_player",
                {"seat": normalized.next_actor},
            ),
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-RETAINED-ROOT-COUNT",
                "proof.winning_root_action_count",
                {"minimum": 0, "maximum": len(normalized.legal_actions)},
            ),
        }
    )
    return scenario, normalized


def test_bounded_v4_exact_prefix_retains_qualified_result(rules):
    scenario, normalized = _retained_late_play_scenario(rules)
    actor = normalized.round_state.current_player
    assert actor is not None
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "bounded",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": actor.team.name.lower(),
            "utility": "awarded_point_margin",
            "successThreshold": -1_000_000,
            "horizon": {"unit": "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1",
            "budgetProfile": "bounded_v1",
        }
    )
    recipe = _exact_scenario_recipe(
        "RCP-V4BOUNDED1",
        "إثبات محدود محفوظ",
        scenario,
        proof,
    )

    result, harness = _run_exact_scenario_recipe(rules, recipe)

    assert harness.store.run["status"] == "completed"
    assert harness.store.run["resultCount"] == 1
    assert harness.store.run["qualifiedCount"] == 1
    assert harness.store.results == [result]
    assert result["disposition"] == "qualified"
    assert result["truthLabel"] == "robust_best_known"
    assert result["scenarioSatisfaction"]["status"] == "satisfied"
    assert result["qualification"]["qualified"] is True
    assert result["qualification"]["observerSafe"] is True
    assert result["scenarioProjection"]["origin"] == "exact_trace_prefix"
    assert result["scenarioProjection"]["requestedProofMode"] == "bounded"
    assert result["scenarioProjection"]["achievedProofMode"] == "bounded"
    assert result["scenarioProjection"]["proofCompletion"] == "bounded_complete"
    assert result["scenarioProjection"]["proofVisibility"] == "complete_state"
    assert result["proofOutcome"]["claimSatisfied"] is True
    assert result["proofOutcome"]["horizon"] == {"unit": "plies", "value": 1}
    assert (
        result["proofOutcome"]["boundaryEvaluator"]
        == "current_raw_point_margin_v1"
    )
    assert (
        validate_scenario_result_v4(
            result,
            artifact_store=harness.artifact_store,
        )
        == result
    )


def test_policy_conditioned_v4_exact_prefix_retains_qualified_result(rules):
    scenario, normalized = _retained_late_play_scenario(rules)
    actor = normalized.round_state.current_player
    assert actor is not None
    proof = _policy_proof_for(actor.team.name.lower())
    recipe = _exact_scenario_recipe(
        "RCP-V4POLICY01",
        "إثبات مشروط بالسياسات محفوظ",
        scenario,
        proof,
    )

    result, harness = _run_exact_scenario_recipe(rules, recipe)

    assert harness.store.run["status"] == "completed"
    assert harness.store.run["resultCount"] == 1
    assert harness.store.run["qualifiedCount"] == 1
    assert harness.store.results == [result]
    assert result["disposition"] == "qualified"
    assert result["truthLabel"] == "robust_best_known"
    assert result["scenarioSatisfaction"]["status"] == "satisfied"
    assert result["qualification"]["qualified"] is True
    assert result["qualification"]["observerSafe"] is True
    assert result["scenarioProjection"]["origin"] == "exact_trace_prefix"
    assert (
        result["scenarioProjection"]["requestedProofMode"]
        == "policy_conditioned"
    )
    assert (
        result["scenarioProjection"]["achievedProofMode"]
        == "policy_conditioned"
    )
    assert result["scenarioProjection"]["proofCompletion"] == "policy_complete"
    assert result["scenarioProjection"]["proofVisibility"] == "complete_state"
    assert result["proofOutcome"]["claimSatisfied"] is True
    assert result["proofOutcome"]["continuationLines"] == {
        "complete": True,
        "exactCount": "1",
        "lineUniverse": "policy_conditioned",
    }
    assert result["proofOutcome"]["identities"]["policyHashes"] == {
        seat: profile["profileHash"] for seat, profile in proof.policies.items()
    }
    assert (
        validate_scenario_result_v4(
            result,
            artifact_store=harness.artifact_store,
        )
        == result
    )


@pytest.mark.real_clock
def test_v4_generated_search_public_api_journey_retains_result(root):
    revision_id = _published_ortools_v4_revision_id()
    _authority, context = api.RESEARCH.execution_context_factory.resolve_bundle_revision(revision_id)
    prefix = _api_budgeted_quality_prefix(root)
    layout = DealLayoutV1.build(
        StructuralDomainIRV1.from_rules(context.bundle.rules.data),
        dealer=Seat.parse(prefix["dealer"]), buyer=Seat.WEST,
    )
    ownership = [
        _predicate(f"PRED-API-GEN-OWNER-{prefix['deckOrder'][position]}", "deal.card_owner", {
            "card": prefix["deckOrder"][position], "seat": seat.name.lower(),
            "handStage": "initial_five" if position in layout.positions_for(seat, final=False) else "post_purchase",
        }) for seat in Seat for position in layout.positions_for(seat, final=True)
    ]
    payload = {
        "schemaVersion": 4,
        "code": "RCP-V4APIGEN1",
        "playingQuality": _api_budgeted_quality_policy(),
        "titleAr": "بحث مولد عبر واجهة البحث",
        "output": {"kind": "laffa", "resultCount": 1},
        "configuration": {
            "schemaVersion": 1,
            "bundleRevisionId": revision_id,
        },
        "scenario": {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-API-GEN-DEALER", "auction.dealer", {"seat": prefix["dealer"]},
                    ),
                    *ownership,
                    _predicate("PRED-API-GEN-BUYER", "auction.buyer", {"seat": "west"}),
                    _predicate("PRED-API-GEN-PURCHASE", "deal.purchase_card", {"card": prefix["deckOrder"][layout.purchase_position]}),
                    _predicate("PRED-API-GEN-AUCTION", "auction.action_sequence", {"actions": prefix["auctionActions"], "complete": True}),
                    _predicate(
                        "PRED-API-GEN-CONTRACT",
                        "auction.contract",
                        {"contract": "sun"},
                    ),
                ],
            },
            "decisionPoint": {
                "kind": "play",
                "selector": {"trickIndex": 6},
            },
            "outcome": {
                "predicateId": "PRED-API-GEN-OUTCOME",
                "kind": "outcome.team_round",
                "version": 1,
                "hardness": "report",
                "arguments": {"team": "north_south", "status": "won"},
            },
        },
        "proof": {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -100_000,
            "budgetProfile": "offline_heavyweight_v1",
        },
            "resources": {
                "generationLane": "ortools",
                "generationPolicy": "ortools_cp_sat_v3",
                "seedStart": 1,
                "seedCount": 1,
                "seedsPerShard": 1,
                "solverTimeoutSeconds": _API_CP_SAT_BUDGET_SECONDS,
                "proofBudgetProfile": "offline_heavyweight_v1",
            },
        "renderPolicy": {"mode": "on_demand", "videoCount": 1},
        "status": "published",
    }

    with TestClient(api.app) as client:
        validated = client.post("/api/v2/research-recipes:validate", json=payload)
        assert validated.status_code == 200, validated.text
        validation = validated.json()
        assert validation["valid"] is True
        assert validation["recipe"]["scenario"]["origin"]["kind"] == "search"
        assert "normalizedExactState" not in validation

        estimated = client.post("/api/v2/research-recipes:estimate", json=payload)
        assert estimated.status_code == 200, estimated.text
        estimate = estimated.json()
        assert estimate["schemaVersion"] == 2
        assert estimate["valid"] is True
        assert estimate["launchAdmissible"] is True
        assert estimate["candidateCount"] == 1
        assert estimate["proof"]["requestedMode"] == "exact"
        assert "remainingCardCount" not in estimate["proof"]
        assert estimate["proof"]["admissible"] is True

        created = client.post(
            "/api/v2/research-recipes",
            json=payload,
            headers={"Idempotency-Key": "research-v4-api-generated-create"},
        )
        assert created.status_code == 201, created.text

        submitted = client.post(
            "/api/v2/research-runs",
            json={"recipeCode": created.json()["code"]},
            headers={"Idempotency-Key": "research-v4-api-generated-run"},
        )
        assert submitted.status_code == 202, submitted.text
        submitted_run = submitted.json()
        admission = submitted_run["launchAdmission"]
        assert admission["recipeHash"] == created.json()["sha256"]
        assert admission["estimate"]["launchAdmissible"] is True
        assert admission["estimate"]["proof"]["admissible"] is True
        assert admission["estimate"]["configurationAuthority"] == created.json()[
            "configuration"
        ]
        _assert_public_launch_admission(admission, submitted_run["runCode"])

        run_code = submitted_run["runCode"]
        run = submitted_run
        deadline = time.monotonic() + 2 * admission["estimate"]["maximumCampaignSeconds"]
        while time.monotonic() < deadline:
            response = client.get(f"/api/v2/research-runs/{run_code}")
            assert response.status_code == 200, response.text
            run = response.json()
            if run["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.1)

        assert run["status"] == "completed", run
        assert run["campaignAccounting"]["maximumCampaignSeconds"] == _API_CP_SAT_BUDGET_SECONDS
        assert run["scenarioResearchStatus"] == "satisfied", json.dumps(
            {
                "scenarioResearchStatus": run["scenarioResearchStatus"],
                "campaignProgress": run.get("campaignProgress"),
            },
            ensure_ascii=False,
            indent=2,
        )
        assert run["attemptedCandidates"] == 1
        assert run["resultCount"] == 1
        assert run["qualifiedCount"] == 1
        assert run["selectedCount"] == 1
        assert run["scenarioYieldCounts"] == {
            "attempted": 1,
            "structurallyFeasible": 1,
            "legalReplay": 0,
            "semanticSatisfied": 1,
            "proofCompleted": 1,
            "qualified": 1,
        }
        assert run["outputReceipt"]["requestedResultCount"] == 1
        assert run["outputReceipt"]["requestedVideoCount"] == 1
        assert run["outputReceipt"]["resultTargetSatisfied"] is True

        page = client.get(
            "/api/v2/research-results",
            params={
                "runCode": run_code,
                "contract": "sun",
                "generationLane": "ortools",
                "scenarioOrigin": "search",
                "scenarioStatus": "satisfied",
                "proofCompletion": "exact_complete",
                "exactStateAvailable": True,
            },
        )
        assert page.status_code == 200, page.text
        assert page.json()["total"] == 1

        result_code = page.json()["items"][0]["resultCode"]
        response = client.get(f"/api/v2/research-results/{result_code}")
        assert response.status_code == 200, response.text
        result = response.json()

        assert result["disposition"] == "qualified"
        assert result["truthLabel"] == "exact_optimal"
        assert result["proofOutcome"]["claimSatisfied"] is True
        assert result["proofOutcome"]["completionStatus"] == "exact_complete"
        assert result["scenarioProjection"]["origin"] == "search"
        assert result["scenarioProjection"]["achievedProofMode"] == "exact"
        assert result["scenarioProjection"]["exactStateAvailable"] is True
        assert result["libraryProjection"]["contract"] == "sun"
        assert result["libraryProjection"]["generationLane"] == "ortools"
        assert result["decisionReference"] == {
            "resultCode": result_code,
            "decisionOrdinal": 1,
        }
        assert "scenarioSatisfaction" not in result
        assert "constraintEvidence" not in result
        assert "artifacts" not in result
        assert "provenance" not in result
