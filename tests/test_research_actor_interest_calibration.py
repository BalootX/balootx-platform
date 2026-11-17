"""Editorial-only regressions over byte-pinned, measured actor receipts.

These projections cannot substitute for replay, graph validation, or Research
qualification. Missing companions on rejected source receipts stay missing.
"""

import json
from pathlib import Path

import pytest

from baloot_platform.research.decision_selection import (
    evaluate_round_interestingness_v1,
    interest_admission_summary_v1,
)


CALIBRATION = json.loads(
    (Path(__file__).parent / "fixtures" / "actor_interest_calibration.json").read_text(
        encoding="utf-8"
    )
)["cases"]


@pytest.mark.parametrize("case_id", ["cj", "a3", "7189", "gap2"])
def test_measured_actor_interest_gate_matches_recorded_featured_decision(case_id):
    case = CALIBRATION[case_id]
    verdict = evaluate_round_interestingness_v1(
        case["receipt"], case["terminalScore"], case["policy"],
        featured_state_id=case["featuredStateId"],
    )
    # The saved historical CJ verdict predates strict baseline-score admission.
    expected = {**case["expected"]["featured"]}
    if case_id == "cj":
        expected["blockers"] = ["interestingness_baseline_regret_insufficient"]
    assert verdict["passed"] is False
    rows = verdict["meaningfulDecisions"] + verdict["rejectedDecisions"]
    assert rows == [expected]
    admission = interest_admission_summary_v1(verdict)
    assert admission["score"] == 0


def test_measured_cj_tied_baseline_and_basic_policy_a3_and_7189_are_not_nonobvious():
    case = CALIBRATION["cj"]
    group = case["receipt"]["postAuction"]["projectPlay"]
    row = group["decisions"][0]
    baseline = group["baselineEvidence"][row["proofArtifactHash"]]["proofModel"]["actionQuality"]
    assert baseline["actionScores"] == [{"action": "CJ", "value": 20302},
                                         {"action": "HJ", "value": 20302}]
    assert baseline["selectedPolicyAction"] == "HJ"
    assert baseline["selectionTieBreak"] == "stable_action_id"
    verdict = evaluate_round_interestingness_v1(
        case["receipt"], case["terminalScore"], case["policy"],
        featured_state_id=case["featuredStateId"],
    )
    assert verdict["passed"] is False
    assert "interestingness_baseline_regret_insufficient" in verdict["blockers"]
    cj = CALIBRATION["cj"]["expected"]["featured"]
    assert cj["pointSwing"] == cj["baselineRegret"] == 36
    assert cj["consistentWorlds"] == 2
    assert cj["tieRuleSensitive"] is False
    assert cj["outcomeChanged"] is True
    for case_id, action, worlds, swing in [("a3", "DK", 6, 34), ("7189", "H9", 3, 40)]:
        case = CALIBRATION[case_id]
        group = case["receipt"]["postAuction"]["projectPlay"]
        row = group["decisions"][0]
        baseline = group["baselineEvidence"][row["proofArtifactHash"]]["proofModel"]["actionQuality"]
        assert baseline["selectedPolicyAction"] == row["action"] == action
        focal = case["expected"]["featured"]
        assert focal["baselineRegret"] == 0
        assert focal["consistentWorlds"] == worlds
        assert focal["pointSwing"] == swing
        assert "interestingness_baseline_regret_insufficient" in focal["blockers"]
    gap = CALIBRATION["gap2"]["expected"]["featured"]
    assert gap["pointSwing"] == 2
    assert "interestingness_next_best_regret_insufficient" in gap["blockers"]
