from __future__ import annotations

from pathlib import Path
import json

from service_test_client import OperatorClient as TestClient

from baloot_platform.research.scenario_contracts import (
    SCENARIO_DECISION_KINDS,
    canonical_sha256,
)
from baloot_platform.research.service import (
    ResearchFoundryServiceV1,
)
from baloot_platform.service import api
from scripts.benchmark_ortools_research_v4 import _publish_explicit_bundle

from research_scenario_v4_helpers import _predicate


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
