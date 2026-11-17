from __future__ import annotations

from scripts.run_ortools_candidate_quality_pilot import _candidate_recipe


class _ResearchStub:
    @staticmethod
    def capabilities():
        return {
            "sha256": "1" * 64,
            "bundleHash": "2" * 64,
            "rulesHash": "3" * 64,
            "conventionHash": "4" * 64,
            "research": {
                "defaultAgentRelease": "foundry_balanced@test",
                "analysisProfiles": {
                    "screening": "5" * 64,
                    "cascaded": "7" * 64,
                },
                "defaultScoreProfileHash": "6" * 64,
            },
        }


def test_candidate_quality_recipe_changes_only_candidate_budget():
    payload = _candidate_recipe(
        _ResearchStub(),
        attempts=5,
        seed_start=202608080001,
        candidates_per_seed=4,
    )
    resources = payload["resources"]
    assert payload["output"]["resultCount"] == 20
    assert resources["candidatesPerSeed"] == 4
    assert resources["auctionConstructionMode"] == "cp_sat_actor_safe_decision_v1"
    assert resources["playConstructionMode"] == "actor_safe_agent_verified"
    assert resources["analysisMode"] == "cascaded"
    assert payload["profiles"]["analysisProfileHash"] == "7" * 64
    assert resources["strategicAuction"]["objective"] == "expected_round_margin"
    assert resources["strategicAuction"]["requestedSamples"] == 192
    assert resources["strategicAuction"]["tieBreak"] == (
        "round_lead_probability_then_best_share_then_legal_action_order_v1"
    )
    assert resources["strategicAuction"]["requireOptimalForQualification"] is True
    assert payload["titleAr"] == "اختبار جودة مرشحات OR-Tools"
    assert payload["tactics"] == [{
        "kind": "small_card_big_gain",
        "mode": "qualification_gate",
        "minimumScore": 10.0,
        "weight": 1,
        "perspectives": [],
    }]
    assert payload["qualification"]["minimumTruth"] == "robust_best_known"
    assert payload["qualification"]["minimumConfidence"] == 0.75
