from __future__ import annotations

import json

import pytest
from ortools.sat.python import cp_model  # noqa: F401

from baloot_platform.research.policy_proof_campaign import (
    PolicyProofCampaignProfileV1,
    run_policy_proof_qualification_campaign,
)


def _smoke_profile() -> PolicyProofCampaignProfileV1:
    return PolicyProofCampaignProfileV1(
        played_card_depths=(24,),
        seeds_per_cell=1,
        promotion_minimum_cases=8,
    )


@pytest.mark.real_clock
def test_campaign_covers_every_seat_and_contract_with_matched_evidence(
    rules,
    convention,
):
    result = run_policy_proof_qualification_campaign(
        rules,
        convention,
        profile=_smoke_profile(),
    )
    completed = [case for case in result["cases"] if case["status"] == "completed"]

    assert len(completed) == 8
    assert {
        (case["contract"], case["actor"])
        for case in completed
    } == {
        (contract, seat)
        for contract in ("hokum", "sun")
        for seat in ("north", "west", "south", "east")
    }
    assert all(case["cohortSize"] == 1 for case in completed)
    assert all(case["matchedCohortMethod"] for case in completed)
    assert all(case["agreement"] for case in completed)
    assert result["summary"]["failedCases"] == 0
    assert result["summary"]["agreementRate"] == 1.0
    assert result["summary"]["kernelWallP50Ms"] > 0
    assert result["summary"]["kernelWallP95Ms"] > 0
    assert result["summary"]["kernelCpuSeconds"] > 0
    assert {
        "only_move_opportunity",
        "small_card_opportunity",
        "large_card_sacrifice_opportunity",
    }.issubset(result["summary"]["opportunityCoverage"])


def test_campaign_does_not_promote_provisional_evidence(rules, convention):
    result = run_policy_proof_qualification_campaign(
        rules,
        convention,
        profile=_smoke_profile(),
    )

    assert result["summary"]["kernelExactCount"] == 0
    assert result["summary"]["kernelProvisionalCount"] == 8
    assert result["summary"]["currentExactCount"] == 0
    assert result["summary"]["currentProvisionalCount"] == 8
    assert result["summary"]["kernelQualifiedCount"] == 0
    assert result["summary"]["kernelQualifiedPerCpuHour"] == 0
    assert result["summary"]["qualifiedYieldRatio"] is None
    assert result["summary"]["qualifiedCostReduction"] is None
    json.dumps(result, allow_nan=False)
    assert result["promotion"] == {
        "promote": False,
        "minimumCasesPassed": True,
        "agreementPassed": True,
        "performancePassed": False,
        "failureGatePassed": True,
    }
    assert result["decision"] == "verification_only"


@pytest.mark.real_clock
def test_campaign_normalized_evidence_is_timing_independent(rules, convention):
    first = run_policy_proof_qualification_campaign(
        rules,
        convention,
        profile=_smoke_profile(),
    )
    second = run_policy_proof_qualification_campaign(
        rules,
        convention,
        profile=_smoke_profile(),
    )

    assert first["normalizedEvidenceHash"] == second["normalizedEvidenceHash"]
    assert first["campaignWallSeconds"] != second["campaignWallSeconds"]


@pytest.mark.parametrize(
    "overrides",
    (
        {"played_card_depths": ()},
        {"seeds_per_cell": 0},
        {"maximum_case_seconds": 0},
        {"promotion_minimum_agreement": 1.1},
    ),
)
def test_campaign_profile_rejects_invalid_resource_and_promotion_values(overrides):
    with pytest.raises(ValueError):
        PolicyProofCampaignProfileV1(**overrides)
