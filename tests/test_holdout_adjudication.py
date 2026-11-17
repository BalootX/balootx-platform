from __future__ import annotations

import copy

import pytest

from baloot_platform.benchmark import (
    IndependentAdjudication,
    IntelligenceBenchmarkV18,
    canonical_sha256,
    submission_hash,
)


STRATA = (
    "bidding",
    "opening_lead",
    "play_obligation",
    "projects",
    "midgame",
    "endgame",
    "contract_rescue",
    "risk",
    "outcome_flip",
)


def _profile() -> dict:
    return {
        "strata": list(STRATA),
        "permanent_holdout": {
            "minimum_positions": 18,
            "minimum_per_archetype": 2,
            "minimum_adjudication_agreement": 0.8,
            "minimum_independent_authorities": 2,
            "require_full_candidate_coverage": True,
        },
    }


def _candidates() -> dict:
    positions = []
    for index in range(18):
        positions.append(
            {
                "candidateId": f"candidate-{index:03d}",
                "stateSha256": f"{index + 1:064x}",
                "rulesSha256": "a" * 64,
                "configurationSha256": "b" * 64,
                "engineVersion": "21.0.0",
                "observer": ("north", "west", "south", "east")[index % 4],
                "observerState": {"currentPlayer": "south"},
                "dealHash": f"{1000 + index:064x}",
                "episodeId": f"episode-{index:03d}",
                "sourceGroup": f"source-{index:03d}",
                "seedGroup": f"seed-{index:03d}",
                "archetype": STRATA[index % len(STRATA)],
                "legalActions": ["A", "B"],
            }
        )
    digest = canonical_sha256(positions)
    return {
        "datasetId": f"candidate-{digest[:20]}",
        "datasetSha256": digest,
        "positions": positions,
    }


def _submission(
    candidates: dict,
    *,
    reviewer_id: str,
    authority_id: str,
    action: str = "A",
) -> dict:
    attestation = (
        "I reviewed this corpus independently and did not coordinate labels."
    )
    value = {
        "schemaVersion": 1,
        "datasetId": candidates["datasetId"],
        "datasetSha256": candidates["datasetSha256"],
        "reviewer": {
            "id": reviewer_id,
            "authorityId": authority_id,
            "independenceAttestation": attestation,
            "independenceAttestationSha256": canonical_sha256(
                {
                    "reviewerId": reviewer_id,
                    "authorityId": authority_id,
                    "attestation": attestation,
                }
            ),
        },
        "labels": [
            {
                "candidateId": item["candidateId"],
                "action": action,
                "confidence": 0.9,
                "actionRegrets": (
                    {"A": 0.0, "B": 1.0}
                    if action == "A"
                    else {"A": 1.0, "B": 0.0}
                ),
            }
            for item in candidates["positions"]
        ],
    }
    value["submissionHash"] = submission_hash(value)
    return value


def _exact_evidence(candidates: dict) -> dict:
    candidate = candidates["positions"][0]
    value = {
        "schemaVersion": 1,
        "candidateDatasetId": candidates["datasetId"],
        "candidateDatasetSha256": candidates["datasetSha256"],
        "oracleId": "independent-clone-minimax-v1",
        "oracleAuthoritySha256": "d" * 64,
        "exactLabelCount": 1,
        "positions": [
            {
                "candidateId": candidate["candidateId"],
                "stateSha256": candidate["stateSha256"],
                "observer": candidate["observer"],
                "legalActions": list(candidate["legalActions"]),
                "referenceActions": ["A", "B"],
                "actionValues": [
                    {"action": "A", "expectedTeamMargin": 10.0},
                    {"action": "B", "expectedTeamMargin": 10.0},
                ],
                "labelSource": "exact",
                "labelAuthorityId": "independent-clone-minimax-v1",
                "labelAuthoritySha256": "d" * 64,
                "searchCompletion": "exact",
                "nodes": 42,
            }
        ],
    }
    value["evidenceSha256"] = canonical_sha256(value)
    return value


def test_two_complete_independent_reviews_publish_agreed_holdout():
    candidates = _candidates()
    left = _submission(
        candidates,
        reviewer_id="reviewer-left",
        authority_id="authority-left",
    )
    right = _submission(
        candidates,
        reviewer_id="reviewer-right",
        authority_id="authority-right",
    )

    result = IndependentAdjudication(_profile()).finalize(
        candidates,
        [right, left],
    ).to_dict()

    assert result["positionCount"] == 18
    assert result["agreementRate"] == 1.0
    assert result["disagreements"] == 0
    assert result["automaticPromotion"] is False
    assert len(result["datasetSha256"]) == 64
    assert len(result["manifestSha256"]) == 64
    for item in result["positions"]:
        assert item["split"] == "permanent_holdout"
        assert item["labelSource"] == "independent_adjudication"
        assert item["referenceAction"] == "A"
        assert item["adjudicatorCount"] == 2


def test_exact_labels_are_merged_without_penalizing_optimal_ties():
    candidates = _candidates()
    left = _submission(
        candidates,
        reviewer_id="reviewer-left",
        authority_id="authority-left",
    )
    right = _submission(
        candidates,
        reviewer_id="reviewer-right",
        authority_id="authority-right",
    )
    right["labels"][0]["action"] = "B"
    right["labels"][0]["actionRegrets"] = {"A": 1.0, "B": 0.0}
    right["submissionHash"] = submission_hash(right)

    result = IndependentAdjudication(_profile()).finalize(
        candidates,
        [left, right],
        exact_evidence=_exact_evidence(candidates),
    ).to_dict()

    assert result["positionCount"] == 18
    assert result["agreementRate"] == 1.0
    assert result["disagreements"] == 0
    assert result["labelProvenance"] == {
        "exact": 1,
        "independentlyAdjudicated": 17,
    }
    exact = next(
        item for item in result["positions"]
        if item["labelSource"] == "exact"
    )
    assert exact["referenceActions"] == ["A", "B"]


def test_adjudicated_holdout_feeds_strength_evaluator_without_relabelling():
    candidates = _candidates()
    submissions = [
        _submission(
            candidates,
            reviewer_id=f"reviewer-{side}",
            authority_id=f"authority-{side}",
        )
        for side in ("left", "right")
    ]
    holdout = IndependentAdjudication(_profile()).finalize(
        candidates,
        submissions,
    ).to_dict()
    predictions = [
        {
            "positionId": item["positionId"],
            "predictedAction": item["referenceAction"],
            "rankedActions": list(item["legalActions"]),
            "predictedConfidence": 0.9,
            "predictedRegret": 0.0,
            "searchCompletion": "not_applicable",
            "latencyMs": 1.0,
            "memoryBytes": 1024.0,
            "nodes": 0.0,
        }
        for item in holdout["positions"]
    ]
    evidence = IntelligenceBenchmarkV18(_profile()).evaluate_permanent_holdout(
        "candidate-model",
        holdout["positions"],
        predictions,
        dataset_id=holdout["datasetId"],
        excluded_groups={
            "dealHash": [],
            "episodeId": [],
            "sourceGroup": [],
            "seedGroup": [],
        },
    ).to_dict()

    assert evidence["examples"] == 18
    assert evidence["policyAccuracy"] == 1.0
    assert evidence["labelProvenance"] == {
        "exact": 0,
        "independentlyAdjudicated": 18,
    }
    assert evidence["automaticPromotion"] is False


def test_tampered_or_nonindependent_submissions_fail_closed():
    candidates = _candidates()
    left = _submission(
        candidates,
        reviewer_id="reviewer-left",
        authority_id="shared-authority",
    )
    right = _submission(
        candidates,
        reviewer_id="reviewer-right",
        authority_id="shared-authority",
    )
    with pytest.raises(ValueError, match="authorities must be distinct"):
        IndependentAdjudication(_profile()).finalize(
            candidates,
            [left, right],
        )

    tampered = copy.deepcopy(left)
    tampered["labels"][0]["action"] = "B"
    with pytest.raises(ValueError, match="submission hash mismatch"):
        IndependentAdjudication(_profile()).finalize(
            candidates,
            [
                tampered,
                _submission(
                    candidates,
                    reviewer_id="reviewer-right",
                    authority_id="authority-right",
                ),
            ],
        )


def test_missing_illegal_or_disagreed_labels_cannot_publish():
    candidates = _candidates()
    left = _submission(
        candidates,
        reviewer_id="reviewer-left",
        authority_id="authority-left",
    )
    missing = _submission(
        candidates,
        reviewer_id="reviewer-right",
        authority_id="authority-right",
    )
    missing["labels"].pop()
    missing["submissionHash"] = submission_hash(missing)
    with pytest.raises(ValueError, match="complete corpus"):
        IndependentAdjudication(_profile()).finalize(
            candidates,
            [left, missing],
        )

    illegal = _submission(
        candidates,
        reviewer_id="reviewer-right",
        authority_id="authority-right",
    )
    illegal["labels"][0]["action"] = "ILLEGAL"
    illegal["labels"][0]["actionRegrets"] = {
        "A": 1.0,
        "B": 1.0,
        "ILLEGAL": 0.0,
    }
    illegal["submissionHash"] = submission_hash(illegal)
    with pytest.raises(ValueError, match="illegal action"):
        IndependentAdjudication(_profile()).finalize(
            candidates,
            [left, illegal],
        )

    disagreed = _submission(
        candidates,
        reviewer_id="reviewer-right",
        authority_id="authority-right",
        action="B",
    )
    with pytest.raises(ValueError, match="agreement is below policy"):
        IndependentAdjudication(_profile()).finalize(
            candidates,
            [left, disagreed],
        )
