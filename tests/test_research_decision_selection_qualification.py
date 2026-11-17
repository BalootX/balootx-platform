from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from baloot_platform.challenge.evidence_v3 import (
    DecisionPerspectiveEvidenceV3,
    TruthLabel,
    TruthMode,
    canonical_sha256,
)
from baloot_platform.challenge.profiles import SelectionProfileV1
from baloot_platform.core.enums import Seat
from baloot_platform.research.decision_selection import (
    apply_decision_selection_qualification_v1,
    baseline_strictly_prefers_alternative_v1,
    evaluate_decision_selection_v1,
    evaluate_round_interestingness_v1,
    interest_admission_summary_v1,
)


def test_actor_interest_admission_ranks_only_gate_passers_without_hidden_values():
    from baloot_platform.challenge.quality_diversity import QualityDiversityProfileV1
    from baloot_platform.research.result_diversity import select_research_results

    policy, receipt = _actor_interest_fixture()
    verdict = evaluate_round_interestingness_v1(receipt, {"awardedPoints": [28, 8]},
                                               policy, featured_state_id="featured")
    admission = interest_admission_summary_v1(verdict)
    assert admission["passed"] is True and admission["score"] == 68
    assert set(admission) == {"policyHash", "evidenceHash", "passed", "score"}
    assert admission["evidenceHash"] == verdict["sha256"]
    rejected = interest_admission_summary_v1({**verdict, "passed": False})
    assert rejected["score"] == 0
    values = [{"resultCode": code, "traceHash": canonical_sha256(code), "score": score,
               "disposition": "qualified", "qualification": {"qualified": True, "interestAdmission": gate}}
              for code, score, gate in (("PASS", admission["score"], admission), ("FAIL", 1000, rejected))]
    profile = QualityDiversityProfileV1.build({"schemaVersion": 1, "cellDimensions": ["contract"],
                                              "topKPerCell": 2, "rejectedAuditModulus": 1})
    selected = select_research_results(values, profile=profile, requested=2)
    assert selected.selected_result_codes == ("PASS",)
    assert selected.rejected == ({"resultCode": "FAIL", "reason": "interest_gate_rejected"},)


def test_structural_motif_is_bound_and_deduplicated_without_losing_unknowns():
    from baloot_platform.challenge.quality_diversity import QualityDiversityProfileV1
    from baloot_platform.research.result_diversity import select_research_results

    policy, receipt = _actor_interest_fixture()
    quality = receipt['postAuction']['projectPlay']['baselineEvidence']['proof']['proofModel']['actionQuality']
    flags = dict(ownMaster=False, partnerEntry=False, stranded=False,
                 preserveMaster=False, forcedReturnGuard=False)
    quality['structuralFeatures'] = dict(schemaVersion=1, contract='sun', currentWinnerRelation='opponent',
        actions=[dict(flags, action='DK', role='discard', forcedReturnGuard=True),
                 dict(flags, action='CA', role='discard', ownMaster=True)])
    verdict = evaluate_round_interestingness_v1(receipt, {'awardedPoints': [28, 8]}, policy,
                                               featured_state_id='featured')
    admission = interest_admission_summary_v1(verdict)
    assert admission['structuralMotifHash'] == canonical_sha256(verdict['structuralMotif'])
    assert 'DK' not in str(verdict['structuralMotif'])
    unknown = {key: value for key, value in admission.items() if key != 'structuralMotifHash'}
    different = dict(admission, structuralMotifHash=canonical_sha256('different-role'))
    rows = [dict(resultCode=code, traceHash=canonical_sha256(code), disposition='qualified',
                 qualification=dict(qualified=True, interestAdmission=gate))
            for code, gate in [('A', admission), ('B', dict(admission, policyHash=canonical_sha256('new-baseline'))), ('C', different),
                               ('D', unknown), ('E', unknown)]]
    profile = QualityDiversityProfileV1.build(dict(schemaVersion=1, cellDimensions=['contract'],
                                                  topKPerCell=10, rejectedAuditModulus=1))
    selected = select_research_results(rows, profile=profile, requested=5)
    assert selected.selected_result_codes == ('A', 'C', 'D', 'E')
    assert selected.duplicate_count == 1
    assert selected.rejected == ({'resultCode': 'B', 'reason': 'duplicate_motif'},)
    assert selected == select_research_results(reversed(rows), profile=profile, requested=5)
    assert len(rows) == 5


def _actor_interest_fixture():
    # Synthetic gate evidence; unlike measured A3, its baseline misses the best move.
    policy = {"schemaVersion": 4, "maximumFinalMargin": 40, "minimumPointSwing": 20,
              "maximumFeaturedTrick": 7, "minimumMeaningfulDecisions": 1,
              "minimumConsistentWorlds": 2, "minimumBaselineRegret": 20,
              "minimumNextBestRegret": 20, "baselinePolicyId": "observer_safe_trick_control_v1",
              "allowTieRuleOnlyLessons": False}
    request = {"mode": "exact", "visibility": "actor_information_set", "strategyClass": "pure_deterministic",
               "utility": "awarded_point_margin", "targetTeam": "west_east"}
    proof = {"completionStatus": "exact_complete", "hiddenWorlds": {"complete": True, "count": 6,
             "coverage": {"method": "exhaustive_cp_sat_hidden_allocations_v1"}},
             "rootActions": [{"action": "DK", "value": -20}, {"action": "CA", "value": -54}]}
    artifact = {"request": request, "proof": proof}
    win = copy.deepcopy(artifact)
    win["request"]["utility"] = "round_win"
    for root in win["proof"]["rootActions"]:
        root["value"] = -1
    group = {"decisions": [{"exactStateId": "featured", "playIndex": 27, "action": "DK", "proofArtifactHash": "proof"}],
             "artifacts": {"proof": artifact}, "roundWinEvidence": {"proof": win},
             "baselineEvidence": {"proof": {"proofModel": {"actionQuality": {
                 "policyId": policy["baselinePolicyId"], "scope": "observer_safe_policy_conditioned",
                 "solverStatus": "optimal", "selectedPolicyAction": "CA",
                 "actionScores": [{"action": "CA", "value": 2},
                                  {"action": "DK", "value": 1}]}}}},
             "tieRuleEvidence": {"proof": {rule: {"margin": copy.deepcopy(artifact), "roundWin": copy.deepcopy(win)}
                                            for rule in ("buyer_makes", "buyer_loses")}}}
    return policy, {"sha256": "f" * 64, "postAuction": {"projectPlay": group}}


@pytest.mark.parametrize("scores", [
    None, [], {}, [None],
    [{"action": "CA", "value": 29595}],
    [{"action": "CA", "value": 20302}, {"action": "DK", "value": 20302}],
    [{"action": "CA", "value": 20301}, {"action": "DK", "value": 20302}],
    [{"action": "CA", "value": True}, {"action": "DK", "value": 0}],
    [{"action": "CA", "value": "29595"}, {"action": "DK", "value": 28893}],
    [{"action": "CA", "value": float("inf")}, {"action": "DK", "value": 28893}],
    [{"action": "CA", "value": float("nan")}, {"action": "DK", "value": 28893}],
    [{"action": "CA", "value": 29595}, {"action": "CA", "value": 28893}],
    [{"action": [], "value": 29595}, {"action": "DK", "value": 28893}],
    [{"action": "CA", "value": 29595}, {"action": "DK", "value": 28893},
     {"action": "D9", "value": 30000}],
])
def test_actor_interest_baseline_scores_fail_closed(scores):
    policy, receipt = _actor_interest_fixture()
    quality = receipt["postAuction"]["projectPlay"]["baselineEvidence"]["proof"]["proofModel"]["actionQuality"]
    quality["actionScores"] = scores
    assert baseline_strictly_prefers_alternative_v1(quality, "DK") is False
    verdict = evaluate_round_interestingness_v1(receipt, {"awardedPoints": [28, 8]},
                                               policy, featured_state_id="featured")
    assert verdict["passed"] is False
    assert "interestingness_baseline_regret_insufficient" in verdict["blockers"]
    assert interest_admission_summary_v1(verdict)["score"] == 0


@pytest.mark.parametrize("tamper", ["missing", "flip", "rank", "unverified"])
def test_native_interest_readback_rejects_rehashed_admission_or_score(tamper):
    from baloot_platform.research.result_authority import _native_interest_admission_v1, _validate_native_interest_admission_v1

    policy, receipt = _actor_interest_fixture()
    snapshot = {"components": {"challenge": {"payload": {"scenarioResearchV4": {
        "qualityExecution": {"videoInterestingness": policy}}}}}}
    trace = {"score": {"awardedPoints": [28, 8]}, "playingQualityEvidence": {"receipt": receipt}}
    admission = _native_interest_admission_v1(snapshot, trace, verified_quality=True, featured_state_id="featured")
    result = {"qualification": {"interestAdmission": admission}, "score": 68.0,
              "disposition": "qualified", "scenarioProjection": {"exactStateId": "featured"},
              "provenance": {"resultIdentity": {"authorityArtifactHashes": {
                  "interestAdmission": "a" * 64}}}}
    _validate_native_interest_admission_v1(result, trace, snapshot, verified_quality=True)
    if tamper == "missing":
        result["qualification"].pop("interestAdmission")
    elif tamper == "flip":
        admission["passed"] = False
    elif tamper == "rank":
        result["score"] = 100.0
    with pytest.raises(ValueError, match="interest (admission|score) binding mismatch"):
        _validate_native_interest_admission_v1(result, trace, snapshot, verified_quality=tamper != "unverified")


@pytest.mark.parametrize("defect,blocker", [
    ("missing_baseline", "interestingness_baseline_evidence_unavailable"),
    ("obvious", "interestingness_baseline_regret_insufficient"),
    ("deducible", "interestingness_consistent_worlds_insufficient"),
    ("next_best", "interestingness_next_best_regret_insufficient"),
    ("missing_winner", "interestingness_winner_evidence_unavailable"),
    ("missing_tie", "interestingness_tie_rule_evidence_unavailable"),
    ("tie_only", "interestingness_tie_rule_only_lesson"),
    ("frozen_opponent", "interestingness_competent_opponent_proof_unavailable"),
])
def test_actor_interest_gate_rejects_weak_or_unverified_lessons(defect, blocker):
    policy, receipt = _actor_interest_fixture()
    group = receipt["postAuction"]["projectPlay"]
    if defect == "missing_baseline":
        group.pop("baselineEvidence")
    elif defect == "obvious":
        group["baselineEvidence"]["proof"]["proofModel"]["actionQuality"]["selectedPolicyAction"] = "DK"
    elif defect == "deducible":
        group["artifacts"]["proof"]["proof"]["hiddenWorlds"]["count"] = 1
    elif defect == "next_best":
        group["artifacts"]["proof"]["proof"]["rootActions"].append({"action": "D9", "value": -21})
    elif defect == "missing_winner":
        group.pop("roundWinEvidence")
    elif defect == "missing_tie":
        group.pop("tieRuleEvidence")
    elif defect == "tie_only":
        group["tieRuleEvidence"]["proof"]["buyer_loses"]["margin"]["proof"]["rootActions"][1]["value"] = -20
    else:
        group["artifacts"]["proof"]["request"]["strategyClass"] = "frozen_policy"
    result = evaluate_round_interestingness_v1(receipt, {"awardedPoints": [28, 8]}, policy, featured_state_id="featured")
    assert not result["passed"] and blocker in result["blockers"]


def test_actor_interest_gate_measures_baseline_and_nearest_competitor_not_threshold_wins():
    policy, receipt = _actor_interest_fixture()
    quality = receipt["postAuction"]["projectPlay"]["baselineEvidence"]["proof"]["proofModel"]["actionQuality"]
    assert baseline_strictly_prefers_alternative_v1(quality, "DK") is True
    result = evaluate_round_interestingness_v1(receipt, {"awardedPoints": [28, 8]}, policy, featured_state_id="featured")
    assert result["passed"] and result["interestScore"] == 68
    featured = result["meaningfulDecisions"][0]
    assert featured["baselineRegret"] == featured["pointSwing"] == 34
    assert featured["outcomeChanged"] is False  # A meaningful damage-limiting sacrifice, not a claimed win.
    assert result["sha256"] == canonical_sha256({key: value for key, value in result.items() if key != "sha256"})


def _actor_interest_three_root_fixture():
    policy, receipt = _actor_interest_fixture()
    policy["minimumPointSwing"] = 40
    group = receipt["postAuction"]["projectPlay"]
    comparisons = [(group["artifacts"]["proof"], group["roundWinEvidence"]["proof"]),
                   *((pair["margin"], pair["roundWin"])
                     for pair in group["tieRuleEvidence"]["proof"].values())]
    for margin, winner in comparisons:
        margin["proof"]["rootActions"] = [{"action": action, "value": value}
            for action, value in (("DK", 40), ("CA", 20), ("D9", -20))]
        winner["proof"]["rootActions"] = [{"action": action, "value": value}
            for action, value in (("DK", 1), ("CA", 1), ("D9", -1))]
    quality = group["baselineEvidence"]["proof"]["proofModel"]["actionQuality"]
    quality["actionScores"].append({"action": "D9", "value": 0})
    return policy, receipt


def test_actor_interest_outcome_change_uses_baseline_not_irrelevant_losing_move():
    policy, receipt = _actor_interest_three_root_fixture()
    verdict = evaluate_round_interestingness_v1(receipt, {"awardedPoints": [28, 8]},
                                               policy, featured_state_id="featured")
    assert verdict["passed"] is False
    featured = verdict["rejectedDecisions"][0]
    assert featured["baselineRegret"] == featured["pointSwing"] == 20
    assert featured["outcomeChanged"] is False
    assert "interestingness_outcome_or_margin_swing_insufficient" in featured["blockers"]


@pytest.mark.parametrize("comparison", ["primary", "buyer_makes", "buyer_loses", None])
def test_actor_interest_binds_each_winner_companion_to_baseline(comparison):
    policy, receipt = _actor_interest_three_root_fixture()
    group = receipt["postAuction"]["projectPlay"]
    comparisons = [("primary", group["artifacts"]["proof"], group["roundWinEvidence"]["proof"]),
                   *((rule, pair["margin"], pair["roundWin"])
                     for rule, pair in group["tieRuleEvidence"]["proof"].items())]
    for name, margin, winner in comparisons:
        if name == comparison:
            continue
        for row, value in zip(margin["proof"]["rootActions"], (20, -5, -30)):
            row["value"] = value
        winner["proof"]["rootActions"][1]["value"] = -1
    verdict = evaluate_round_interestingness_v1(receipt, {"awardedPoints": [28, 8]},
                                               policy, featured_state_id="featured")
    assert verdict["passed"] is (comparison is None)
    if comparison is not None:
        blocker = ("interestingness_outcome_or_margin_swing_insufficient" if comparison == "primary"
                   else "interestingness_tie_rule_only_lesson")
        assert blocker in verdict["blockers"]


@pytest.mark.parametrize("backend_id,version,historical", [
    ("ortools_cp_sat_v3", "3.1.140", True),
    ("ortools_cp_sat_v3", "3.1.141", False),
    ("ortools_cp_sat_v3", "3.1.54", False),
    ("ortools_cp_sat_v3", None, False),
    ("ortools_cp_sat_v3", "invalid", False),
    ("ortools_cp_sat_v2", "3.1.140", False),
])
def test_actor_interest_retained_verdict_uses_bound_backend_version(backend_id, version, historical):
    from baloot_platform.research.result_authority import (
        _native_interest_evidence_v1, _validate_native_interest_admission_v1,
        validate_native_interest_artifact_v1,
    )

    policy, receipt = _actor_interest_three_root_fixture()
    snapshot = {"components": {"challenge": {"payload": {"scenarioResearchV4": {
        "qualityExecution": {"videoInterestingness": policy}}}}}}
    trace = {"score": {"awardedPoints": [28, 8]}, "playingQualityEvidence": {"receipt": receipt}}
    authority = {"backendId": backend_id, "backendVersion": version}
    evidence = _native_interest_evidence_v1(snapshot, trace, verified_quality=True,
        featured_state_id="featured", research_authority=authority)
    assert evidence["passed"] is historical
    result = {"researchAuthority": authority, "qualification": {
        "allDecisionPolicyAuditComplete": True, "interestAdmission": interest_admission_summary_v1(evidence)},
        "score": 40.0 if historical else 0.0, "disposition": "qualified",
        "scenarioProjection": {"exactStateId": "featured"},
        "provenance": {"resultIdentity": {"authorityArtifactHashes": {"interestAdmission": "a" * 64}}}}
    original = copy.deepcopy((receipt, result, evidence))
    _validate_native_interest_admission_v1(result, trace, snapshot, verified_quality=True)
    validate_native_interest_artifact_v1(result, trace, snapshot, evidence)
    assert (receipt, result, evidence) == original
    # Historical read compatibility never opts fresh render admission into the old gate.
    assert evaluate_round_interestingness_v1(receipt, trace["score"], policy,
        featured_state_id="featured")["passed"] is False


@pytest.mark.parametrize("baseline_policy_id", ["observer_safe_trick_control_v1",
    "observer_safe_partner_feed_established_control_v1"])
def test_a3_educational_but_obvious_is_not_curated(baseline_policy_id):
    # Measured A3: six worlds, DK -20 versus CA -54; both basic baselines find DK.
    policy, receipt = _actor_interest_fixture()
    policy["baselinePolicyId"] = baseline_policy_id
    group = receipt["postAuction"]["projectPlay"]
    group["baselineEvidence"]["proof"]["proofModel"]["actionQuality"].update(
        policyId=baseline_policy_id, selectedPolicyAction="DK")
    verdict = evaluate_round_interestingness_v1(receipt, {"awardedPoints": [28, 8]},
                                               policy, featured_state_id="featured")
    assert verdict["passed"] is False
    focal = verdict["rejectedDecisions"][0]
    assert focal["consistentWorlds"] == 6 and focal["pointSwing"] == 34
    assert focal["baselineRegret"] == 0
    assert "interestingness_baseline_regret_insufficient" in focal["blockers"]


@pytest.mark.parametrize("field,value", [("minimumConsistentWorlds", 1), ("minimumBaselineRegret", True),
                                         ("allowTieRuleOnlyLessons", 0), ("baselinePolicyId", "unknown")])
def test_actor_interest_policy_rejects_coercion_and_unsupported_baselines(field, value):
    from baloot_platform.research.scenario_contracts import build_scenario_video_interestingness_v1
    policy, _ = _actor_interest_fixture()
    policy[field] = value
    with pytest.raises(ValueError):
        build_scenario_video_interestingness_v1(policy)


@pytest.mark.parametrize("change,expected_blocker", [
    ({}, None),
    ({"implicitPlay": True, "featuredIndex": 19}, None),
    ({"points": [64, 6], "values": [58, 50]}, "interestingness_result_not_contested"),
    ({"featuredIndex": 24}, "interestingness_early_featured_decision_unavailable"),
    ({"policyVersion": 3, "featuredIndex": 23, "visibility": "actor_information_set", "hiddenComplete": True}, None),
    ({"policyVersion": 3, "featuredIndex": 27, "visibility": "actor_information_set", "hiddenComplete": True}, None),
    ({"policyVersion": 3, "featuredIndex": 27, "visibility": "actor_information_set", "hiddenComplete": True,
      "decisionCount": 1, "minimumDecisions": 1, "maximumMargin": 40, "points": [40, 0]}, None),
    ({"policyVersion": 3, "featuredIndex": 27, "visibility": "actor_information_set", "hiddenComplete": True,
      "decisionCount": 1, "minimumDecisions": 1, "maximumMargin": 40, "points": [41, 0]}, "interestingness_result_not_contested"),
    ({"policyVersion": 3, "featuredIndex": 28}, "interestingness_early_featured_decision_unavailable"),
    ({"policyVersion": 3, "featuredIndex": 23, "points": [64, 6]}, "interestingness_result_not_contested"),
    ({"policyVersion": 3, "featuredIndex": 23, "completion": "bounded_complete"}, "interestingness_meaningful_decisions_insufficient"),
    ({"policyVersion": 3, "featuredIndex": 23, "visibility": "actor_information_set", "hiddenComplete": False}, "interestingness_meaningful_decisions_insufficient"),
    ({"decisionCount": 1}, "interestingness_meaningful_decisions_insufficient"),
    ({"completion": "bounded_complete"}, "interestingness_meaningful_decisions_insufficient"),
    ({"values": [10, None]}, "interestingness_meaningful_decisions_insufficient"),
    ({"values": [10, 10]}, "interestingness_meaningful_decisions_insufficient"),
    ({"visibility": "actor_information_set", "hiddenComplete": True}, None),
    ({"visibility": "actor_information_set"}, "interestingness_meaningful_decisions_insufficient"),
    ({"visibility": "actor_information_set", "hiddenComplete": False}, "interestingness_meaningful_decisions_insufficient"),
    ({"visibility": "actor_information_set", "hiddenComplete": 1}, "interestingness_meaningful_decisions_insufficient"),
    ({"visibility": "actor_information_set", "hiddenComplete": True,
      "completion": "bounded_complete"}, "interestingness_meaningful_decisions_insufficient"),
])
def test_round_interest_screen_requires_certified_early_and_multiple_choices(change, expected_blocker):
    policy = {"schemaVersion": 1, "maximumFinalMargin": 20, "minimumPointSwing": 20,
              "maximumFeaturedTrick": 5, "minimumMeaningfulDecisions": 2}
    if change.get("policyVersion") == 3:
        policy.update(schemaVersion=3, maximumFeaturedTrick=7)
        policy.update(minimumMeaningfulDecisions=change.get("minimumDecisions", 2),
                      maximumFinalMargin=change.get("maximumMargin", 20))
    decisions, artifacts = [], {}
    for index in range(change.get("decisionCount", 2)):
        key = str(index)
        decisions.append({"exactStateId": key, "playIndex": change.get("featuredIndex", 16) + index,
                          "decisionKind": "play", "action": "A", "proofArtifactHash": key})
        if change.get("implicitPlay"):
            decisions[-1].pop("decisionKind")
        values = change.get("values", [10, -12])
        artifacts[key] = {"request": {"mode": "exact", "visibility": change.get("visibility", "complete_state"),
                                      "utility": "awarded_point_margin"},
            "proof": {"completionStatus": change.get("completion", "exact_complete"),
                      **({"hiddenWorlds": {"complete": change["hiddenComplete"]}}
                         if "hiddenComplete" in change else {}),
                      "rootActions": [{"action": action, "value": value}
                                      for action, value in zip(("A", "B"), values)]}}
    receipt = {"sha256": "f" * 64, "postAuction": {"projectPlay": {
        "decisions": decisions, "artifacts": artifacts}}}
    screen = evaluate_round_interestingness_v1(receipt, {"awardedPoints": change.get("points", [25, 15])},
                                               policy, featured_state_id="0")
    assert screen["sha256"] == canonical_sha256({k: v for k, v in screen.items() if k != "sha256"})
    if expected_blocker is None:
        assert screen["passed"] and len(screen["meaningfulDecisions"]) == change.get("decisionCount", 2)
    else:
        assert not screen["passed"] and expected_blocker in screen["blockers"]
    if change.get("featuredIndex") == 24:
        decisions[0]["decisionKind"] = "project"
        assert evaluate_round_interestingness_v1(receipt, {"awardedPoints": [25, 15]},
            policy, featured_state_id="0")["passed"]
    if not change:
        assert evaluate_round_interestingness_v1(receipt, {"awardedPoints": [64, 6]},
            policy, featured_state_id="0")["passed"]  # Exact outcome-changing alternative, despite a realized blowout.
    if change.get("implicitPlay"):
        assert all(row["decisionKind"] == "play" for row in screen["meaningfulDecisions"])
        assert next(row for row in screen["meaningfulDecisions"] if row["exactStateId"] == "0")["early"]
        misplaced = {"auction": receipt["postAuction"]["projectPlay"]}
        assert not evaluate_round_interestingness_v1(misplaced, {"awardedPoints": [25, 15]},
            policy, featured_state_id="0")["passed"]


@pytest.mark.parametrize("values,raw_points,index,completion,passed", [
    ([10, -12], [85, 45], 23, "exact_complete", True),
    ([10, -12], [110, 20], 23, "exact_complete", True),
    ([30, 10], [85, 45], 23, "exact_complete", False),
    ([10, -12], [85, 45], 24, "exact_complete", False),
    ([10, -12], None, 23, "exact_complete", False),
    ([10, -12], [85, 45], 23, "bounded_complete", False),
])
def test_raw_point_interest_screen_keeps_exact_outcome_branch_without_awarded_swing(
    values, raw_points, index, completion, passed,
):
    policy = {"schemaVersion": 2, "maximumFinalMargin": 40, "minimumPointSwing": 15,
              "maximumFeaturedTrick": 6, "minimumMeaningfulDecisions": 1}
    receipt = {"sha256": "f" * 64, "postAuction": {"projectPlay": {
        "decisions": [{"exactStateId": "featured", "playIndex": index, "decisionKind": "play",
                       "action": "A", "proofArtifactHash": "proof"}],
        "artifacts": {"proof": {"request": {"mode": "exact", "visibility": "complete_state",
                                           "utility": "awarded_point_margin"},
                               "proof": {"completionStatus": completion, "rootActions": [
                                   {"action": action, "value": value}
                                   for action, value in zip(("A", "B"), values)]}}}}}}
    screen = evaluate_round_interestingness_v1(receipt,
        {"rawCardPoints": raw_points, "awardedPoints": [30, 0]}, policy, featured_state_id="featured")
    assert screen["passed"] is passed
    assert screen["pointUnit"] == "raw_points"
    assert screen["finalMargin"] == (abs(raw_points[0] - raw_points[1]) if raw_points else None)
    assert all(row["pointSwing"] is None for row in screen["meaningfulDecisions"])
    assert screen["sha256"] == canonical_sha256({key: value for key, value in screen.items() if key != "sha256"})


@pytest.mark.parametrize("values,completion,utility,actions,passed", [
    ([20, 5], "exact_complete", "_internal_raw_point_margin_v1", ["A", "B"], True),
    ([20, 6], "exact_complete", "_internal_raw_point_margin_v1", ["A", "B"], False),
    ([5, 20], "exact_complete", "_internal_raw_point_margin_v1", ["A", "B"], False),
    ([20, 5], "bounded_complete", "_internal_raw_point_margin_v1", ["A", "B"], False),
    ([20, 5], "exact_complete", "awarded_point_margin", ["A", "B"], False),
    ([20, 5], "exact_complete", "_internal_raw_point_margin_v1", ["A", "C"], False),
])
@pytest.mark.parametrize("visibility,hidden_complete", [
    ("complete_state", None), ("actor_information_set", True), ("actor_information_set", False),
])
def test_raw_point_screen_requires_owned_utility_and_recorded_raw_best(values, completion, utility, actions, passed, visibility, hidden_complete):
    policy = {"schemaVersion": 2, "maximumFinalMargin": 40, "minimumPointSwing": 15,
              "maximumFeaturedTrick": 6, "minimumMeaningfulDecisions": 1}
    group = {"decisions": [{"exactStateId": "featured", "playIndex": 23, "decisionKind": "play",
                          "action": "A", "proofArtifactHash": "normal"}],
        "artifacts": {"normal": {"request": {"mode": "exact", "visibility": visibility,
                                               "utility": "awarded_point_margin"},
            "proof": {"completionStatus": "exact_complete", "rootActions": [
                {"action": "A", "value": 30}, {"action": "B", "value": 10}]}}},
        "rawPointEvidence": {"normal": {"request": {"mode": "exact", "visibility": visibility, "utility": utility},
            "proof": {"completionStatus": completion, "rootActions": [
                {"action": action, "value": value} for action, value in zip(actions, values)]}}}}
    if visibility == "actor_information_set":
        group["artifacts"]["normal"]["proof"]["hiddenWorlds"] = {"complete": True}
        group["rawPointEvidence"]["normal"]["proof"]["hiddenWorlds"] = {
            "complete": hidden_complete, "count": 6,
            "coverage": {"method": "exhaustive_cp_sat_hidden_allocations_v1"}}
    screen = evaluate_round_interestingness_v1({"postAuction": {"projectPlay": group}},
        {"rawCardPoints": [85, 45]}, policy, featured_state_id="featured")
    assert screen["passed"] is (passed and hidden_complete is not False)
    if screen["passed"]:
        assert screen["meaningfulDecisions"][0]["pointSwing"] == 15


def _profile(**overrides) -> SelectionProfileV1:
    value = {
        "schemaVersion": 1,
        "id": "research-decision-selection",
        "revision": "1",
        "weights": {"outcomeImpact": 1.0},
        "allowedTruthLabels": ["exact_optimal", "robust_best_known"],
        "minimumConfidence": 0.75,
        "minimumScore": 70.0,
        "requireVerifiedTacticalPlay": False,
    }
    value.update(overrides)
    return SelectionProfileV1.build(value)


def _perspective(
    decision_id: str,
    *,
    score: float = 70.0,
    confidence: float = 0.75,
    truth: TruthLabel = TruthLabel.ROBUST_BEST_KNOWN,
    dimensions: dict[str, float] | None = None,
) -> DecisionPerspectiveEvidenceV3:
    return DecisionPerspectiveEvidenceV3(
        decision_id=decision_id,
        acting_seat=Seat.SOUTH,
        observer_seat=Seat.SOUTH,
        truth_mode=TruthMode.ACTOR_OBSERVER,
        projection_hash="a" * 64,
        forced=False,
        truth_label=truth,
        confidence=confidence,
        evidence=(
            dimensions
            if dimensions is not None
            else {"outcomeImpact": score}
        ),
        recommended_action={"action": "pass"},
    )


def _analysis(
    decision_id: str,
    ordinal: int,
    *,
    regret: float = 0.0,
    probability_drop: float = 0.0,
    confidence: float = 1.0,
    catastrophic: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        decision_id=decision_id,
        ordinal=ordinal,
        forced=False,
        expected_regret=regret,
        probability_drop=probability_drop,
        confidence=confidence,
        catastrophic_crossing=catastrophic,
        alternatives=(
            {"expectedMargin": 10.0, "winProbability": 0.75},
            {"expectedMargin": 5.0, "winProbability": 0.50},
        ),
        difficulty=50.0,
        debate_score=0.0,
    )


def _authority(
    *,
    competent: tuple[str, ...] = (),
    unresolved: tuple[str, ...] = (),
) -> dict[str, object]:
    core = {
        "schemaVersion": 2,
        "competentDecisionIds": list(competent),
        "unresolvedDecisionIds": list(unresolved),
    }
    return {**core, "sha256": canonical_sha256(core)}


def _evaluate(
    profile: SelectionProfileV1 | None,
    perspectives,
    analyses,
    *,
    kinds=None,
    tactical=(),
    competent=(),
    unresolved=(),
    authority_hash=None,
    authority=None,
):
    return evaluate_decision_selection_v1(
        profile,
        perspectives=perspectives,
        analyses=analyses,
        analysis_hash="b" * 64,
        decision_kinds=(
            kinds
            if kinds is not None
            else {item.decision_id: "play" for item in perspectives}
        ),
        tactical_verifications=tactical,
        competent_decision_ids=competent,
        unresolved_decision_ids=unresolved,
        authority_evidence_hash=authority_hash,
        authority_evidence=authority,
    )


def _base_qualification() -> dict[str, object]:
    return {
        "qualified": True,
        "truthLabel": "robust_best_known",
        "failures": [],
    }


def test_one_passing_decision_satisfies_the_frozen_selection_profile() -> None:
    evidence = (
        _perspective("D-pass", score=80.0),
        _perspective("D-low", score=20.0),
    )
    evaluation = _evaluate(
        _profile(),
        evidence,
        (_analysis("D-pass", 1), _analysis("D-low", 2)),
    )
    result = apply_decision_selection_qualification_v1(
        _base_qualification(), evaluation
    )

    assert evaluation["passed"] is True
    assert evaluation["qualifiedDecisionIds"] == ["D-pass"]
    assert result["qualified"] is True
    assert result["failures"] == []


def test_minimum_score_and_confidence_boundaries_are_inclusive() -> None:
    evidence = (_perspective("D-edge", score=70.0, confidence=0.75),)
    analyses = (_analysis("D-edge", 1),)

    at_boundary = _evaluate(_profile(), evidence, analyses)
    above_score = _evaluate(
        _profile(minimumScore=70.000001), evidence, analyses
    )
    above_confidence = _evaluate(
        _profile(minimumConfidence=0.750001), evidence, analyses
    )

    assert at_boundary["passed"] is True
    assert above_score["passed"] is False
    assert "minimum_score" in above_score["rejectionReasons"]
    assert above_confidence["passed"] is False
    assert "confidence" in above_confidence["rejectionReasons"]


def test_profile_change_influences_disposition_without_changing_raw_score() -> None:
    evidence = (_perspective("D1", score=70.0),)
    analyses = (_analysis("D1", 1),)
    open_profile = _profile(minimumScore=70.0)
    closed_profile = _profile(minimumScore=70.000001)

    opened = _evaluate(open_profile, evidence, analyses)
    closed = _evaluate(closed_profile, evidence, analyses)
    opened_result = apply_decision_selection_qualification_v1(
        _base_qualification(), opened
    )
    closed_result = apply_decision_selection_qualification_v1(
        _base_qualification(), closed
    )

    assert open_profile.sha256 != closed_profile.sha256
    assert opened["decisions"][0]["score"]["score"] == pytest.approx(70.0)
    assert closed["decisions"][0]["score"]["score"] == pytest.approx(70.0)
    assert opened_result["qualified"] is True
    assert closed_result["qualified"] is False
    assert "decision_score_profile_gate:minimum_score" in closed_result[
        "failures"
    ]


def test_profile_weights_influence_score_gate_and_final_disposition() -> None:
    evidence = (
        _perspective(
            "D1",
            dimensions={"outcomeImpact": 100.0, "tacticalDepth": 0.0},
        ),
    )
    analyses = (_analysis("D1", 1),)
    outcome_weighted = _profile(
        weights={"outcomeImpact": 3.0, "tacticalDepth": 1.0},
        minimumScore=60.0,
    )
    tactical_weighted = _profile(
        weights={"outcomeImpact": 1.0, "tacticalDepth": 3.0},
        minimumScore=60.0,
    )

    accepted = _evaluate(outcome_weighted, evidence, analyses)
    rejected = _evaluate(tactical_weighted, evidence, analyses)

    assert accepted["decisions"][0]["score"]["score"] == pytest.approx(75.0)
    assert rejected["decisions"][0]["score"]["score"] == pytest.approx(25.0)
    assert accepted["passed"] is True
    assert rejected["passed"] is False
    assert "minimum_score" in rejected["rejectionReasons"]


def test_truth_label_gate_changes_final_disposition() -> None:
    evidence = (_perspective("D1"),)
    analyses = (_analysis("D1", 1),)
    result = _evaluate(
        _profile(allowedTruthLabels=["exact_optimal"]),
        evidence,
        analyses,
    )

    assert result["passed"] is False
    assert "truth_label" in result["rejectionReasons"]


def test_verified_tactical_play_gate_is_applied_candidate_specifically() -> None:
    evidence = (_perspective("D-play", score=90.0),)
    analyses = (_analysis("D-play", 1),)
    profile = _profile(requireVerifiedTacticalPlay=True)

    missing = _evaluate(profile, evidence, analyses)
    verification = SimpleNamespace(
        status="verified",
        oracle=SimpleNamespace(decision_id="D-play", complete=True),
        evidence=(SimpleNamespace(),),
        sha256="c" * 64,
    )
    verified = _evaluate(
        profile,
        evidence,
        analyses,
        tactical=(verification,),
    )
    bid = _evaluate(
        profile,
        evidence,
        analyses,
        kinds={"D-play": "bid"},
    )

    assert missing["passed"] is False
    assert "verified_tactical_play" in missing["rejectionReasons"]
    assert verified["passed"] is True
    assert bid["passed"] is True


def test_surrounding_gate_and_its_limit_influence_final_disposition() -> None:
    perspectives = (
        _perspective("D-target", score=90.0),
        _perspective("D-weak", score=10.0),
    )
    analyses = (
        _analysis("D-target", 1),
        _analysis("D-weak", 2, regret=5.0, probability_drop=0.20),
    )
    strict = _profile(
        minimumSurroundingQuality=75.0,
        maximumSurroundingMaterialMistakes=0,
    )
    tolerant = _profile(
        minimumSurroundingQuality=75.0,
        maximumSurroundingMaterialMistakes=1,
    )

    rejected = _evaluate(strict, perspectives, analyses)
    accepted = _evaluate(tolerant, perspectives, analyses)

    assert rejected["passed"] is False
    assert "surrounding_material_mistakes" in rejected["rejectionReasons"]
    assert accepted["passed"] is True
    assert strict.sha256 != tolerant.sha256


def test_surrounding_confidence_and_materiality_boundaries_are_applied() -> None:
    perspectives = (
        _perspective("D-target", score=90.0),
        _perspective("D-weak", score=10.0),
    )
    analyses = (
        _analysis("D-target", 1),
        _analysis("D-weak", 2, confidence=0.49),
    )

    confidence_boundary = _evaluate(
        _profile(
            minimumSurroundingConfidence=0.49,
            minimumSurroundingMateriality=50.0,
            maximumSurroundingUnverifiedDecisions=0,
        ),
        perspectives,
        analyses,
    )
    confidence_above = _evaluate(
        _profile(
            minimumSurroundingConfidence=0.490001,
            minimumSurroundingMateriality=50.0,
            maximumSurroundingUnverifiedDecisions=0,
        ),
        perspectives,
        analyses,
    )
    materiality_above = _evaluate(
        _profile(
            minimumSurroundingConfidence=0.490001,
            minimumSurroundingMateriality=50.000001,
            maximumSurroundingUnverifiedDecisions=0,
        ),
        perspectives,
        analyses,
    )

    assert confidence_boundary["passed"] is True
    assert confidence_above["passed"] is False
    assert "surrounding_unverified_decisions" in confidence_above[
        "rejectionReasons"
    ]
    assert materiality_above["passed"] is True


def test_surrounding_failure_count_limits_influence_final_disposition() -> None:
    perspectives = (
        _perspective("D-target", score=90.0),
        _perspective("D-weak", score=10.0),
    )
    analyses = (
        _analysis("D-target", 1),
        _analysis("D-weak", 2, catastrophic=True),
    )

    strict = _evaluate(
        _profile(maximumSurroundingCatastrophicCrossings=0),
        perspectives,
        analyses,
    )
    tolerant = _evaluate(
        _profile(maximumSurroundingCatastrophicCrossings=1),
        perspectives,
        analyses,
    )

    assert strict["passed"] is False
    assert "surrounding_catastrophic_crossings" in strict[
        "rejectionReasons"
    ]
    assert tolerant["passed"] is True


def test_competence_can_support_surrounding_but_is_never_featured() -> None:
    perspectives = (
        _perspective("D-target", score=90.0),
        _perspective("D-support", score=90.0),
    )
    analyses = (
        _analysis("D-target", 1),
        _analysis("D-support", 2, confidence=0.10),
    )

    authority = _authority(competent=("D-support",))
    result = _evaluate(
        _profile(
            minimumSurroundingConfidence=0.75,
            minimumSurroundingMateriality=0.0,
            maximumSurroundingUnverifiedDecisions=0,
        ),
        perspectives,
        analyses,
        competent=("D-support",),
        authority_hash=authority["sha256"],
        authority=authority,
    )

    rows = {row["decisionId"]: row for row in result["decisions"]}
    assert result["passed"] is True
    assert result["qualifiedDecisionIds"] == ["D-target"]
    assert result["competentDecisionIds"] == ["D-support"]
    assert result["featuredTruthBlockedDecisionIds"] == ["D-support"]
    assert rows["D-target"]["surroundingCompetence"][
        "unverifiedDecisions"
    ] == 0
    assert rows["D-support"]["gatePassed"] is False
    assert "adjudication_featured_truth_blocked" in rows["D-support"][
        "gateFailures"
    ]


def test_unresolved_precedence_and_status_conflicts_fail_closed() -> None:
    perspectives = (
        _perspective("D-target", score=90.0),
        _perspective("D-support", score=90.0),
    )
    analyses = (
        _analysis("D-target", 1),
        _analysis("D-support", 2, confidence=0.10),
    )

    authority = _authority(
        competent=("D-support",),
        unresolved=("D-support",),
    )
    result = _evaluate(
        _profile(
            minimumSurroundingConfidence=0.75,
            minimumSurroundingMateriality=0.0,
            maximumSurroundingUnverifiedDecisions=0,
        ),
        perspectives,
        analyses,
        competent=("D-support",),
        unresolved=("D-support",),
        authority_hash=authority["sha256"],
        authority=authority,
    )

    rows = {row["decisionId"]: row for row in result["decisions"]}
    assert result["passed"] is False
    assert result["competentDecisionIds"] == []
    assert result["unresolvedDecisionIds"] == ["D-support"]
    assert result["featuredTruthBlockedDecisionIds"] == ["D-support"]
    assert "adjudication_authority_status_conflict" in result[
        "rejectionReasons"
    ]
    assert rows["D-target"]["surroundingCompetence"][
        "unverifiedDecisions"
    ] == 1
    assert "surrounding_unverified_decisions" in rows["D-target"][
        "gateFailures"
    ]


def test_adjudication_status_ids_require_known_decisions_and_bound_hash() -> None:
    perspectives = (_perspective("D-target", score=90.0),)
    analyses = (_analysis("D-target", 1),)

    authority = _authority(competent=("D-unknown",))
    unknown = _evaluate(
        _profile(),
        perspectives,
        analyses,
        competent=("D-unknown",),
        authority_hash=authority["sha256"],
        authority=authority,
    )
    unbound = _evaluate(
        _profile(),
        perspectives,
        analyses,
        unresolved=("D-target",),
        authority_hash="not-a-sha256",
    )

    assert unknown["passed"] is False
    assert "adjudication_authority_unknown_decision" in unknown[
        "rejectionReasons"
    ]
    assert unbound["passed"] is False
    assert "adjudication_authority_evidence_unbound" in unbound[
        "rejectionReasons"
    ]
    assert unbound["qualifiedDecisionIds"] == []


def test_missing_profile_fails_closed_and_preserves_legacy_readjudication() -> None:
    evaluation = _evaluate(
        None,
        (_perspective("D1"),),
        (_analysis("D1", 1),),
    )
    existing = {
        "qualified": False,
        "truthLabel": "rejected",
        "failures": ["historical_record_requires_readjudication"],
    }

    result = apply_decision_selection_qualification_v1(existing, evaluation)

    assert result["qualified"] is False
    assert "historical_record_requires_readjudication" in result["failures"]
    assert "decision_score_profile_gate:profile_unavailable" in result[
        "failures"
    ]
