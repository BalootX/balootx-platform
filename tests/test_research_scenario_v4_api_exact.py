from __future__ import annotations

import json
import time

import pytest
from service_test_client import OperatorClient as TestClient

from baloot_platform.research.exact_state import ExactStateNormalizerV1
from baloot_platform.research.scenario_contracts import (
    canonical_sha256,
)
from baloot_platform.service import api

from research_scenario_v4_helpers import (
    _API_CP_SAT_BUDGET_SECONDS,
    _api_budgeted_quality_policy,
    _api_budgeted_quality_prefix,
    _assert_public_launch_admission,
    _predicate,
    _published_ortools_v4_revision_id,
)


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
