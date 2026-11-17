from __future__ import annotations

import json

from baloot_platform.service.container import build_container
from scripts.run_strategic_play_pilot import _recipe, _wait


def test_actor_safe_auction_is_selected_by_agents_then_verified_by_cp_sat(
    tmp_path,
) -> None:
    container = build_container(writable_root=tmp_path / "runtime")
    try:
        research = container.research
        payload = _recipe(
            research,
            mode="actor_safe_agent_verified",
            attempts=1,
            seed_start=202608080019,
        )
        payload["schemaVersion"] = 3
        payload.pop("profiles")
        payload["configuration"] = {
            "schemaVersion": 1,
            "bundleRevisionId": container.configuration.resolve()["bundle"][
                "revisionId"
            ],
        }
        payload["tactics"] = []
        payload["resources"]["auctionConstructionMode"] = (
            "actor_safe_agent_verified"
        )
        payload["qualification"] = {
            "minimumTruth": "provisional",
            "minimumConfidence": 0,
            "minimumSurroundingQuality": 0,
            "requireObserverSafety": True,
            "requireAllDecisionAudit": True,
            "maximumRegret": 999,
        }
        validated = research.validate_recipe(payload)
        assert validated["valid"] is True, validated["issues"]
        created = research.store.put_recipe(validated["recipe"])
        submitted = research.submit(
            created["code"],
            idempotency_key="actor-safe-auction-integration",
        )
        run = _wait(research, submitted["runCode"], timeout=900)

        assert run["status"] == "completed"
        assert run["auctionFeasible"] == 1
        assert run["quarantineCount"] == 0
        results = research.list_results(run_code=run["runCode"], limit=10)[
            "items"
        ]
        assert len(results) == 1
        result = results[0]
        provenance = result["provenance"]
        assert provenance["auctionConstructionMode"] == (
            "actor_safe_agent_verified"
        )
        assert provenance["auctionCompilerSolverStatus"] == "optimal"
        assert len(provenance["auctionCompilerPlanHash"]) == 64
        assert len(provenance["auctionCompilerModelHash"]) == 64

        trace = json.loads(
            research.artifact_store.get(
                result["artifacts"]["completeTrace"]["uri"]
            )
        )
        bid_count = sum(
            item["decisionKind"] == "bid"
            for item in trace["decisionLedger"]["decisions"]
        )
        policy_traces = trace["trustedFinalState"]["decisionPolicyTraces"]
        assert bid_count > 0
        assert all(
            item["trace"].get("selectionReason") != "compiled_legal_auction"
            for item in policy_traces[:bid_count]
        )
    finally:
        container.shutdown()


def test_actor_safe_auction_is_exposed_as_an_opt_in_capability(tmp_path) -> None:
    container = build_container(writable_root=tmp_path / "runtime")
    try:
        research = container.research
        assert research.capabilities()["research"]["auctionConstructionModes"] == [
            "actor_safe_agent_verified",
            "cp_sat_actor_safe_decision_v1",
            "cp_sat_feasibility_line",
        ]
    finally:
        container.shutdown()
