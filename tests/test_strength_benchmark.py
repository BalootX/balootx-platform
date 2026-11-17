from __future__ import annotations

import copy

import pytest

from baloot_platform.benchmark import IntelligenceBenchmarkV18


STRATA = ("bidding", "opening_lead", "midgame", "endgame")


def _profile(
    *,
    minimum_positions: int = 8,
    minimum_per_archetype: int = 2,
    minimum_pairs: int = 8,
) -> dict:
    return {
        "strata": list(STRATA),
        "permanent_holdout": {
            "minimum_positions": minimum_positions,
            "minimum_per_archetype": minimum_per_archetype,
            "minimum_adjudication_agreement": 0.8,
            "calibration_bins": 5,
        },
        "paired_comparison": {
            "minimum_pairs": minimum_pairs,
            "minimum_win_rate": 0.55,
            "minimum_win_rate_lower_bound": 0.50,
            "minimum_regret_improvement": 0.25,
        },
    }


def _holdout_rows() -> list[dict]:
    rows = []
    for index in range(8):
        correct = index != 7
        reference = f"A{index}"
        alternative = f"B{index}"
        rows.append(
            {
                "positionId": f"position-{index:03d}",
                "split": "permanent_holdout",
                "stateSha256": f"{index + 1:064x}",
                "rulesSha256": "a" * 64,
                "configurationSha256": "b" * 64,
                "engineVersion": "21.0.0",
                "observer": ("north", "west", "south", "east")[index % 4],
                "dealHash": f"{1000 + index:064x}",
                "episodeId": f"episode-{index:03d}",
                "sourceGroup": f"source-{index:03d}",
                "seedGroup": f"seed-{index:03d}",
                "archetype": STRATA[index % len(STRATA)],
                "labelSource": (
                    "exact" if index < 4 else "independent_adjudication"
                ),
                "labelAuthorityId": (
                    "independent-clone-minimax-v1"
                    if index < 4
                    else "expert-panel-v1"
                ),
                "labelAuthoritySha256": "c" * 64,
                "adjudicatorCount": 2,
                "adjudicators": ["reviewer-a", "reviewer-b"],
                "adjudicationAgreement": 1.0,
                "legalActions": [reference, alternative],
                "referenceAction": reference,
                "actionRegrets": {
                    reference: 0.0,
                    alternative: 2.0,
                },
                "predictedAction": reference if correct else alternative,
                "rankedActions": [reference, alternative],
                "predictedConfidence": 0.9 if correct else 0.6,
                "predictedRegret": 0.0 if correct else 2.0,
                "searchCompletion": "exact",
                "latencyMs": 10.0 + index,
                "memoryBytes": 1024.0 + index,
                "nodes": 100.0 + index,
            }
        )
    return rows


def _excluded_groups() -> dict[str, list[str]]:
    return {
        "dealHash": ["f" * 64],
        "episodeId": ["tuning-episode"],
        "sourceGroup": ["training-source"],
        "seedGroup": ["training-seed"],
    }


def _predictions(rows: list[dict]) -> list[dict]:
    prediction_fields = {
        "predictedAction",
        "rankedActions",
        "predictedConfidence",
        "predictedRegret",
        "searchCompletion",
        "latencyMs",
        "memoryBytes",
        "nodes",
    }
    predictions = []
    for row in rows:
        prediction = {"positionId": row["positionId"]}
        for field in prediction_fields:
            prediction[field] = row.pop(field)
        predictions.append(prediction)
    return predictions


def _paired_rows(count: int = 100, wins: int = 90) -> list[dict]:
    rows = []
    for index in range(count):
        if index < wins:
            candidate_scores = (10.0, 8.0)
            baseline_scores = (5.0, 4.0)
        else:
            candidate_scores = (4.0, 5.0)
            baseline_scores = (8.0, 7.0)
        rows.append(
            {
                "pairId": f"pair-{index:03d}",
                "candidateId": "candidate-v21",
                "baselineId": "baseline-v20",
                "dealHash": f"{2000 + index:064x}",
                "configurationHash": "d" * 64,
                "rulesSha256": "a" * 64,
                "engineVersion": "21.0.0",
                "seedGroup": f"seed-{index:03d}",
                "games": [
                    {
                        "candidateTeam": "north_south",
                        "candidateScore": candidate_scores[0],
                        "baselineScore": baseline_scores[0],
                    },
                    {
                        "candidateTeam": "west_east",
                        "candidateScore": candidate_scores[1],
                        "baselineScore": baseline_scores[1],
                    },
                ],
            }
        )
    return rows


def test_permanent_holdout_is_deterministic_and_never_promotes():
    benchmark = IntelligenceBenchmarkV18(_profile())
    rows = _holdout_rows()
    predictions = _predictions(rows)
    first = benchmark.evaluate_permanent_holdout(
        "candidate-v21",
        rows,
        predictions,
        dataset_id="holdout-v21",
        excluded_groups=_excluded_groups(),
    ).to_dict()
    second = benchmark.evaluate_permanent_holdout(
        "candidate-v21",
        reversed(rows),
        reversed(predictions),
        dataset_id="holdout-v21",
        excluded_groups=_excluded_groups(),
    ).to_dict()

    assert first == second
    assert first["examples"] == 8
    assert first["labelProvenance"] == {
        "exact": 4,
        "independentlyAdjudicated": 4,
    }
    assert first["illegalActionRate"] == 0.0
    assert first["automaticPromotion"] is False
    assert first["groupLeakageDetected"] is False
    assert len(first["resultHash"]) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(dealHash="f" * 64), "leakage"),
        (lambda row: row.update(adjudicatorCount=1), "reviewers"),
        (lambda row: row.update(adjudicationAgreement=0.5), "agreement"),
        (lambda row: row.update(legalActions=["only"]), "two legal choices"),
    ],
)
def test_permanent_holdout_rejects_invalid_evidence(mutation, message: str):
    benchmark = IntelligenceBenchmarkV18(_profile())
    rows = _holdout_rows()
    predictions = _predictions(rows)
    target = rows[4]
    mutation(target)

    with pytest.raises(ValueError, match=message):
        benchmark.evaluate_permanent_holdout(
            "candidate-v21",
            rows,
            predictions,
            dataset_id="holdout-v21",
            excluded_groups=_excluded_groups(),
        )


def test_permanent_holdout_rejects_invalid_prediction_evidence():
    benchmark = IntelligenceBenchmarkV18(_profile())
    rows = _holdout_rows()
    predictions = _predictions(rows)
    predictions[0]["searchCompletion"] = "unknown"

    with pytest.raises(ValueError, match="completion"):
        benchmark.evaluate_permanent_holdout(
            "candidate-v21",
            rows,
            predictions,
            dataset_id="holdout-v21",
            excluded_groups=_excluded_groups(),
        )


def test_dataset_identity_does_not_depend_on_candidate_predictions():
    benchmark = IntelligenceBenchmarkV18(_profile())
    rows = _holdout_rows()
    predictions = _predictions(rows)
    alternate = copy.deepcopy(predictions)
    alternate[0]["predictedAction"] = "B0"
    alternate[0]["predictedRegret"] = 1.0

    first = benchmark.evaluate_permanent_holdout(
        "candidate-a",
        rows,
        predictions,
        dataset_id="holdout-v21",
        excluded_groups=_excluded_groups(),
    ).to_dict()
    second = benchmark.evaluate_permanent_holdout(
        "candidate-b",
        rows,
        alternate,
        dataset_id="holdout-v21",
        excluded_groups=_excluded_groups(),
    ).to_dict()

    assert first["datasetSha256"] == second["datasetSha256"]
    assert first["policyAccuracy"] != second["policyAccuracy"]


def test_exact_tied_reference_actions_are_all_scored_as_correct():
    benchmark = IntelligenceBenchmarkV18(_profile())
    rows = _holdout_rows()
    rows[0]["referenceActions"] = ["A0", "B0"]
    rows[0]["actionRegrets"]["B0"] = 0.0
    predictions = _predictions(rows)
    predictions[0]["predictedAction"] = "B0"
    predictions[0]["rankedActions"] = ["B0", "A0"]

    result = benchmark.evaluate_permanent_holdout(
        "candidate-v21",
        rows,
        predictions,
        dataset_id="holdout-v21",
        excluded_groups=_excluded_groups(),
    ).to_dict()

    assert result["policyAccuracy"] == 0.875


def test_regret_is_owned_by_hidden_labels_not_candidate_predictions():
    benchmark = IntelligenceBenchmarkV18(_profile())
    rows = _holdout_rows()
    predictions = _predictions(rows)
    for prediction in predictions:
        prediction["predictedRegret"] = 0.0

    result = benchmark.evaluate_permanent_holdout(
        "candidate-v21",
        rows,
        predictions,
        dataset_id="holdout-v21",
        excluded_groups=_excluded_groups(),
    ).to_dict()

    assert result["expectedRegret"] == 0.25


def test_permanent_holdout_rejects_missing_archetype_coverage():
    benchmark = IntelligenceBenchmarkV18(_profile(minimum_per_archetype=2))
    rows = _holdout_rows()
    predictions = _predictions(rows)
    for row in rows:
        if row["archetype"] == "endgame":
            row["archetype"] = "midgame"

    with pytest.raises(ValueError, match="archetype coverage"):
        benchmark.evaluate_permanent_holdout(
            "candidate-v21",
            rows,
            predictions,
            dataset_id="holdout-v21",
            excluded_groups=_excluded_groups(),
        )


def test_symmetric_pairs_require_exact_seat_swap_and_unique_units():
    benchmark = IntelligenceBenchmarkV18(_profile(minimum_pairs=2))
    rows = _paired_rows(count=2, wins=2)
    rows[0]["games"][1]["candidateTeam"] = "north_south"
    with pytest.raises(ValueError, match="swap candidate teams"):
        benchmark.evaluate_symmetric_pairs(
            "candidate-v21",
            "baseline-v20",
            rows,
            candidate_expected_regret=0.5,
            baseline_expected_regret=1.0,
        )

    rows = _paired_rows(count=2, wins=2)
    rows[1]["dealHash"] = rows[0]["dealHash"]
    rows[1]["seedGroup"] = rows[0]["seedGroup"]
    with pytest.raises(ValueError, match="duplicate unit"):
        benchmark.evaluate_symmetric_pairs(
            "candidate-v21",
            "baseline-v20",
            rows,
            candidate_expected_regret=0.5,
            baseline_expected_regret=1.0,
        )


def test_symmetric_pairs_report_promotion_target_without_promoting():
    benchmark = IntelligenceBenchmarkV18(
        _profile(minimum_pairs=100)
    )
    result = benchmark.evaluate_symmetric_pairs(
        "candidate-v21",
        "baseline-v20",
        _paired_rows(count=100, wins=90),
        candidate_expected_regret=0.70,
        baseline_expected_regret=1.0,
    ).to_dict()

    assert result["wins"] == 90
    assert result["losses"] == 10
    assert result["ties"] == 0
    assert result["confidence95"]["decisiveWinRate"][0] > 0.5
    assert result["regretImprovement"] == pytest.approx(0.3)
    assert result["promotionTargetMet"] is True
    assert result["automaticPromotion"] is False


def test_symmetric_pairs_report_ties_and_failed_target():
    rows = _paired_rows(count=8, wins=4)
    tied = copy.deepcopy(rows[7])
    tied["pairId"] = "pair-tied"
    tied["dealHash"] = "e" * 64
    tied["seedGroup"] = "seed-tied"
    for game in tied["games"]:
        game["candidateScore"] = game["baselineScore"]
    rows[7] = tied

    result = IntelligenceBenchmarkV18(_profile()).evaluate_symmetric_pairs(
        "candidate-v21",
        "baseline-v20",
        rows,
        candidate_expected_regret=0.9,
        baseline_expected_regret=1.0,
    ).to_dict()

    assert result["ties"] == 1
    assert result["promotionTargetMet"] is False
