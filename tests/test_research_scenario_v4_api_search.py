from __future__ import annotations

import json
import time

import pytest
from service_test_client import OperatorClient as TestClient

from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Seat
from baloot_platform.core.structural_ir import StructuralDomainIRV1
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
