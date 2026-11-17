from __future__ import annotations

import copy
import math
from typing import Any, Iterable, Mapping

from baloot_platform.challenge.evidence_v3 import TruthMode, canonical_sha256
from baloot_platform.research.scenario_contracts import build_scenario_video_interestingness_v1


DECISION_SELECTION_PROFILE_UNAVAILABLE = "profile_unavailable"
DECISION_SELECTION_NO_ACTOR_EVIDENCE = "no_actor_observer_decision"
DECISION_SELECTION_NO_QUALIFIED_DECISION = "no_qualified_decision"


def evaluate_round_interestingness_v1(
    receipt: Mapping[str, Any], terminal_score: Mapping[str, Any],
    policy: Mapping[str, Any], *, featured_state_id: str,
    _historical_any_alternative: bool = False,
) -> dict[str, Any]:
    """Cheap editorial screen over freshly validated, immutable proof variables.

    Missing alternative values cannot establish a swing. No new strategic proof
    is run here, and this screen cannot change Research qualification.
    """
    policy = build_scenario_video_interestingness_v1(policy)
    if policy["schemaVersion"] == 4:
        return _evaluate_actor_round_interest_v4(receipt, terminal_score, policy,
            featured_state_id=featured_state_id,
            _historical_any_alternative=_historical_any_alternative)
    raw_point_policy = policy["schemaVersion"] == 2
    post = receipt.get("postAuction") or {}
    project_play = post.get("projectPlay") or {}
    groups = [receipt.get("auction") or {}, project_play]
    # Escalation receipts bind each decision through its artifact reference.
    escalation = post.get("escalation") or {}
    for reference in escalation.get("decisions") or ():
        artifact = escalation.get("artifacts", {}).get(reference.get("artifactHash"), {})
        groups.append({"decisions": [{"decisionKind": "escalation",
            "exactStateId": artifact.get("exactStateId"),
            "action": artifact.get("proof", {}).get("selectedAction"),
            "proofArtifactHash": reference.get("artifactHash")}],
            "artifacts": escalation.get("artifacts") or {}})
    meaningful: dict[str, dict[str, Any]] = {}
    contested_by_choice = False
    for group in groups:
        for row in group.get("decisions") or ():
            artifact = group.get("artifacts", {}).get(row.get("proofArtifactHash"), {})
            request, proof = artifact.get("request") or {}, artifact.get("proof") or {}
            roots = proof.get("rootActions") or []
            visibility = request.get("visibility")
            if (request.get("mode") != "exact" or visibility not in {"complete_state", "actor_information_set"}
                    or proof.get("completionStatus") != "exact_complete" or len(roots) < 2
                    or any(type(root.get("value")) not in {int, float}
                           or not math.isfinite(root["value"]) for root in roots)):
                continue
            if visibility == "actor_information_set" and (proof.get("hiddenWorlds") or {}).get("complete") is not True:
                continue
            selected = [root for root in roots if root.get("action") == row.get("action")]
            if len(selected) != 1 or not isinstance(row.get("exactStateId"), str):
                continue
            value = selected[0]["value"]
            alternatives = [root["value"] for root in roots if root["action"] != row["action"]]
            if not alternatives or value < max(alternatives):
                continue
            utility = request.get("utility")
            outcome_changed = utility in {"awarded_point_margin", "round_win", "contract_success"} and (
                value > 0 and any(other < 0 for other in alternatives))
            kabout_changed = utility == "kabout" and value > 0 and any(other < 0 for other in alternatives)
            contested_by_choice |= outcome_changed or kabout_changed
            swing = value - min(alternatives) if (
                not raw_point_policy and utility == "awarded_point_margin") else None
            if raw_point_policy:
                from .ortools_complete_state_graph import INTERNAL_RAW_POINT_MARGIN_UTILITY_V1
                companion = group.get("rawPointEvidence", {}).get(row.get("proofArtifactHash"), {})
                raw_request, raw_proof = companion.get("request") or {}, companion.get("proof") or {}
                raw_roots = raw_proof.get("rootActions") or []
                raw_values = {root.get("action"): root.get("value") for root in raw_roots}
                worlds = raw_proof.get("hiddenWorlds") or {}
                if (raw_request.get("mode") == "exact" and raw_request.get("visibility") == request.get("visibility")
                        and (request.get("visibility") == "complete_state" or worlds.get("complete") is True
                            and (worlds.get("coverage") or {}).get("method") == "exhaustive_cp_sat_hidden_allocations_v1")
                        and raw_request.get("utility") == INTERNAL_RAW_POINT_MARGIN_UTILITY_V1
                        and raw_proof.get("completionStatus") == "exact_complete"
                        and len(raw_values) == len(raw_roots) and set(raw_values) == {root["action"] for root in roots}
                        and all(type(point) is int for point in raw_values.values())
                        and row["action"] in raw_values and raw_values[row["action"]] == max(raw_values.values())):
                    swing = raw_values[row["action"]] - min(raw_values.values())
            if not (outcome_changed or kabout_changed
                    or swing is not None and swing >= policy["minimumPointSwing"]):
                continue
            kind = row.get("decisionKind", "play" if group is project_play else None)
            index = row.get("playIndex")
            early = kind in {"bid", "project"} or (
                kind == "play" and type(index) is int and 0 <= index < 4 * policy["maximumFeaturedTrick"])
            meaningful[row["exactStateId"]] = {"exactStateId": row["exactStateId"],
                "decisionKind": kind, "playIndex": index, "pointSwing": swing,
                "outcomeChanged": outcome_changed, "kaboutChanged": kabout_changed,
                "early": early, "proofArtifactHash": row["proofArtifactHash"]}
    points = terminal_score.get("rawCardPoints" if raw_point_policy else "awardedPoints")
    margin = abs(points[0] - points[1]) if (
        isinstance(points, (list, tuple)) and len(points) == 2
        and all(type(point) is int for point in points)) else None
    blockers = []
    if margin is None:
        blockers.append("interestingness_terminal_score_unavailable")
    if not (margin is not None and margin <= policy["maximumFinalMargin"]
            or policy["schemaVersion"] != 3 and contested_by_choice):
        blockers.append("interestingness_result_not_contested")
    if not meaningful.get(featured_state_id, {}).get("early"):
        blockers.append("interestingness_early_featured_decision_unavailable")
    if len(meaningful) < policy["minimumMeaningfulDecisions"]:
        blockers.append("interestingness_meaningful_decisions_insufficient")
    core = {"schemaVersion": 1, "policy": policy, "qualityReceiptHash": receipt.get("sha256"),
            "featuredExactStateId": featured_state_id, "finalMargin": margin,
            "meaningfulDecisions": [meaningful[key] for key in sorted(meaningful)],
            "passed": not blockers, "blockers": blockers}
    if raw_point_policy:
        core["pointUnit"] = "raw_points"
    return {**core, "sha256": canonical_sha256(core)}


def interest_admission_summary_v1(verdict: Mapping[str, Any]) -> dict[str, Any]:
    """Public, hash-bound admission without hidden alternative actions or values."""
    summary = {"policyHash": canonical_sha256(verdict["policy"]),
            "evidenceHash": verdict["sha256"], "passed": verdict["passed"],
            "score": verdict.get("interestScore") if verdict["passed"] else 0}
    if verdict.get("structuralMotif") is not None:
        summary["structuralMotifHash"] = canonical_sha256(verdict["structuralMotif"])
    return summary


def _structural_motif(quality: Mapping[str, Any], action: str) -> dict[str, Any] | None:
    """Group public action structures, not claimed causal explanations."""
    features = quality.get("structuralFeatures")
    if not isinstance(features, Mapping) or features.get("schemaVersion") != 1:
        return None
    if features.get("contract") not in {"sun", "hokum"} or features.get("currentWinnerRelation") not in {"none", "self", "partner", "opponent"}:
        return None
    rows = features.get("actions")
    if not isinstance(rows, (list, tuple)) or any(not isinstance(row, Mapping) for row in rows):
        return None
    flags = ("ownMaster", "partnerEntry", "stranded", "preserveMaster", "forcedReturnGuard")
    selected = []
    for code in (action, quality.get("selectedPolicyAction")):
        matches = [row for row in rows if row.get("action") == code]
        if len(matches) != 1:
            return None
        row = matches[0]
        if row.get("role") not in {"lead", "follow", "discard"} or any(type(row.get(key)) is not bool for key in flags):
            return None
        selected.append({key: row[key] for key in ("role", *flags)})
    if not any(row[key] for row in selected for key in flags):
        return None
    return {"schemaVersion": 1, "kind": "public_action_structure", "contract": features["contract"],
            "currentWinnerRelation": features["currentWinnerRelation"],
            "optimal": selected[0], "baseline": selected[1]}


def baseline_strictly_prefers_alternative_v1(
    quality: Mapping[str, Any], optimal_action: str,
) -> bool:
    """Stable-ID tie breaking is not evidence that the optimal move is non-obvious."""
    rows = quality.get("actionScores")
    if not isinstance(rows, (list, tuple)) or len(rows) < 2:
        return False
    scores = {}
    for row in rows:
        if (not isinstance(row, Mapping) or not isinstance(row.get("action"), str)
                or type(row.get("value")) is not int or row["action"] in scores):
            return False
        scores[row["action"]] = row["value"]
    selected = quality.get("selectedPolicyAction")
    return (isinstance(selected, str) and selected in scores and optimal_action in scores
            and scores[selected] == max(scores.values())
            and scores[selected] > scores[optimal_action])


def actor_interest_primary_failures_v1(action, artifact, policy):
    """Necessary primary-proof checks; these never admit a round for rendering."""
    request, proof = artifact.get("request") or {}, artifact.get("proof") or {}
    worlds, roots = proof.get("hiddenWorlds") or {}, proof.get("rootActions") or []
    failures = []
    if not (request.get("mode") == "exact" and request.get("visibility") == "actor_information_set"
            and request.get("strategyClass") == "pure_deterministic"
            and request.get("utility") == "awarded_point_margin"
            and proof.get("completionStatus") == "exact_complete" and worlds.get("complete") is True
            and (worlds.get("coverage") or {}).get("method") == "exhaustive_cp_sat_hidden_allocations_v1"):
        failures.append("interestingness_exact_actor_proof_unavailable")
    values = {root.get("action"): root.get("value") for root in roots}
    if (len(values) != len(roots) or len(values) < 2 or action not in values
            or any(type(value) is not int for value in values.values())):
        return [*failures, "interestingness_root_values_unavailable"]
    best = values[action]
    if best != max(values.values()):
        failures.append("interestingness_selected_action_not_optimal")
    if best - max(value for key, value in values.items() if key != action) < policy["minimumNextBestRegret"]:
        failures.append("interestingness_next_best_regret_insufficient")
    count = worlds.get("count")
    if type(count) is not int or count < policy["minimumConsistentWorlds"]:
        failures.append("interestingness_consistent_worlds_insufficient")
    return failures


def actor_interest_prerequisite_failures_v1(action, artifact, baseline, policy):
    """Necessary editorial checks only; these never admit a round for rendering."""
    failures = actor_interest_primary_failures_v1(action, artifact, policy)
    if "interestingness_root_values_unavailable" in failures:
        return failures
    values = {root["action"]: root["value"] for root in artifact["proof"]["rootActions"]}
    best = values[action]
    quality = (baseline.get("proofModel") or {}).get("actionQuality") or {}
    baseline_action = quality.get("selectedPolicyAction")
    if (quality.get("policyId") != policy["baselinePolicyId"]
            or quality.get("scope") != "observer_safe_policy_conditioned"
            or quality.get("solverStatus") != "optimal" or baseline_action not in values):
        failures.append("interestingness_baseline_evidence_unavailable")
    elif (not baseline_strictly_prefers_alternative_v1(quality, action)
          or best - values[baseline_action] < policy["minimumBaselineRegret"]):
        failures.append("interestingness_baseline_regret_insufficient")
    return failures


def _evaluate_actor_round_interest_v4(receipt, terminal_score, policy, *, featured_state_id,
                                    _historical_any_alternative=False):
    """Editorial admission over already verified native companion evidence."""
    group = (receipt.get("postAuction") or {}).get("projectPlay") or {}
    admitted, rejected = [], []
    for row in group.get("decisions") or ():
        source_hash = row.get("proofArtifactHash")
        artifact = (group.get("artifacts") or {}).get(source_hash, {})
        request, proof = artifact.get("request") or {}, artifact.get("proof") or {}
        roots = proof.get("rootActions") or []
        worlds = proof.get("hiddenWorlds") or {}
        baseline = (group.get("baselineEvidence") or {}).get(source_hash, {})
        failures = actor_interest_prerequisite_failures_v1(row.get("action"), artifact, baseline, policy)
        exact = (request.get("mode") == "exact" and request.get("visibility") == "actor_information_set"
                 and request.get("strategyClass") == "pure_deterministic"
                 and request.get("utility") == "awarded_point_margin"
                 and proof.get("completionStatus") == "exact_complete"
                 and worlds.get("complete") is True
                 and (worlds.get("coverage") or {}).get("method") == "exhaustive_cp_sat_hidden_allocations_v1")
        values = {root.get("action"): root.get("value") for root in roots}
        action = row.get("action")
        if "interestingness_root_values_unavailable" in failures:
            rejected.append({"exactStateId": row.get("exactStateId"), "blockers": failures})
            continue
        best, alternative = values[action], max(value for key, value in values.items() if key != action)
        gap = best - alternative
        count = worlds.get("count")
        baseline_quality = (baseline.get("proofModel") or {}).get("actionQuality") or {}
        baseline_action = baseline_quality.get("selectedPolicyAction")
        baseline_regret = best - values[baseline_action] if baseline_action in values else None
        win = (group.get("roundWinEvidence") or {}).get(source_hash, {})
        win_request, win_proof = win.get("request") or {}, win.get("proof") or {}
        win_values = {root.get("action"): root.get("value") for root in win_proof.get("rootActions") or []}
        outcome_changed = False
        if (win_request.get("mode") != "exact" or win_request.get("visibility") != "actor_information_set"
                or win_request.get("utility") != "round_win"
                or win_request.get("targetTeam") != request.get("targetTeam")
                or win_proof.get("completionStatus") != "exact_complete"
                or (win_proof.get("hiddenWorlds") or {}).get("complete") is not True
                or set(win_values) != set(values) or any(type(value) is not int or value not in {-1, 1}
                                                       for value in win_values.values())):
            failures.append("interestingness_winner_evidence_unavailable")
        else:
            outcome_changed = (any(value != win_values[action] for key, value in win_values.items() if key != action)
                if _historical_any_alternative else baseline_action in win_values
                and win_values[baseline_action] != win_values[action])
        if not outcome_changed and gap < policy["minimumPointSwing"]:
            failures.append("interestingness_outcome_or_margin_swing_insufficient")
        # Exhaustive nonanticipative minimax, not a frozen opponent rollout, supplies robustness.
        if not exact or request.get("boundaryEvaluator") is not None:
            failures.append("interestingness_competent_opponent_proof_unavailable")
        tie = (group.get("tieRuleEvidence") or {}).get(source_hash, {})
        tie_lessons = []
        for rule in ("buyer_makes", "buyer_loses"):
            comparison = tie.get(rule) or {}
            margin_artifact, winner_artifact = comparison.get("margin") or {}, comparison.get("roundWin") or {}
            comparison_request, comparison_proof = margin_artifact.get("request") or {}, margin_artifact.get("proof") or {}
            winner_request, winner_proof = winner_artifact.get("request") or {}, winner_artifact.get("proof") or {}
            comparison_values = {root.get("action"): root.get("value")
                                 for root in comparison_proof.get("rootActions") or []}
            winner_values = {root.get("action"): root.get("value")
                             for root in winner_proof.get("rootActions") or []}
            if (comparison_request.get("utility") != "awarded_point_margin" or winner_request.get("utility") != "round_win"
                    or any(document.get("mode") != "exact" or document.get("visibility") != "actor_information_set"
                           or document.get("targetTeam") != request.get("targetTeam")
                           for document in (comparison_request, winner_request))
                    or any(document.get("completionStatus") != "exact_complete"
                           or (document.get("hiddenWorlds") or {}).get("complete") is not True
                           for document in (comparison_proof, winner_proof))
                    or set(comparison_values) != set(values) or set(winner_values) != set(values)
                    or any(type(value) is not int for value in comparison_values.values())
                    or any(type(value) is not int or value not in {-1, 1} for value in winner_values.values())):
                break
            comparison_gap = comparison_values[action] - max(value for key, value in comparison_values.items() if key != action)
            comparison_baseline = (comparison_values[action] - comparison_values[baseline_action]
                                   if baseline_action in comparison_values else -1)
            comparison_outcome = (any(value != winner_values[action] for key, value in winner_values.items() if key != action)
                if _historical_any_alternative else baseline_action in winner_values
                and winner_values[baseline_action] != winner_values[action])
            tie_lessons.append(comparison_gap >= policy["minimumNextBestRegret"]
                               and comparison_baseline >= policy["minimumBaselineRegret"]
                               and (comparison_outcome or comparison_gap >= policy["minimumPointSwing"]))
        tie_sensitive = not all(tie_lessons) if len(tie_lessons) == 2 else None
        if tie_sensitive is None:
            failures.append("interestingness_tie_rule_evidence_unavailable")
        elif tie_sensitive and not policy["allowTieRuleOnlyLessons"]:
            failures.append("interestingness_tie_rule_only_lesson")
        index = row.get("playIndex")
        if type(index) is not int or not 0 <= index < 4 * policy["maximumFeaturedTrick"]:
            failures.append("interestingness_early_featured_decision_unavailable")
        evidence = {"exactStateId": row.get("exactStateId"), "proofArtifactHash": source_hash,
                    "playIndex": index, "pointSwing": gap, "baselineRegret": baseline_regret,
                    "consistentWorlds": count, "outcomeChanged": outcome_changed,
                    "tieRuleSensitive": tie_sensitive,
                    "interestScore": gap + (baseline_regret or 0)}
        motif = _structural_motif(baseline_quality, action)
        if motif is not None:
            evidence["structuralMotif"] = motif
        if failures:
            rejected.append({**evidence, "blockers": list(dict.fromkeys(failures))})
        else:
            admitted.append(evidence)
    admitted.sort(key=lambda row: (-row["interestScore"], row["exactStateId"]))
    points = terminal_score.get("awardedPoints")
    margin = abs(points[0] - points[1]) if (isinstance(points, (list, tuple)) and len(points) == 2
                                          and all(type(point) is int for point in points)) else None
    blockers = []
    if margin is None or margin > policy["maximumFinalMargin"]:
        blockers.append("interestingness_result_not_contested")
    if len(admitted) < policy["minimumMeaningfulDecisions"]:
        blockers.append("interestingness_meaningful_decisions_insufficient")
    featured = next((row for row in admitted if row["exactStateId"] == featured_state_id), None)
    if featured is None:
        blockers.extend(next((row["blockers"] for row in rejected if row["exactStateId"] == featured_state_id),
                             ["interestingness_featured_evidence_unavailable"]))
    core = {"schemaVersion": 1, "policy": policy, "qualityReceiptHash": receipt.get("sha256"),
            "featuredExactStateId": featured_state_id, "finalMargin": margin,
            "meaningfulDecisions": admitted, "rejectedDecisions": rejected,
            "interestScore": featured["interestScore"] if featured else None,
            "passed": not blockers, "blockers": list(dict.fromkeys(blockers))}
    if featured is not None and "structuralMotif" in featured:
        core["structuralMotif"] = featured["structuralMotif"]
    return {**core, "sha256": canonical_sha256(core)}


def _verified_tactical_play(value: Any | None) -> bool:
    return bool(
        value is not None
        and str(getattr(value, "status", "")) == "verified"
        and bool(getattr(getattr(value, "oracle", None), "complete", False))
        and tuple(getattr(value, "evidence", ()) or ())
    )


def evaluate_decision_selection_v1(
    profile: Any | None,
    *,
    perspectives: Iterable[Any],
    analyses: Iterable[Any],
    analysis_hash: str,
    decision_kinds: Mapping[str, str],
    tactical_verifications: Iterable[Any] = (),
    competent_decision_ids: Iterable[str] = (),
    unresolved_decision_ids: Iterable[str] = (),
    authority_evidence_hash: str | None = None,
    authority_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one frozen decision-selection profile without changing scores.

    A complete Laffa needs at least one selectable actor-observer decision.
    Other decisions remain fully audited, but they do not all need to be
    editorial challenge candidates.
    """

    if profile is None:
        core = {
            "schemaVersion": 2,
            "profileHash": None,
            "analysisHash": str(analysis_hash),
            "evaluatedDecisionCount": 0,
            "qualifiedDecisionCount": 0,
            "qualifiedDecisionIds": [],
            "passed": False,
            "rejectionReasons": [DECISION_SELECTION_PROFILE_UNAVAILABLE],
            "decisions": [],
            "authorityEvidenceHash": authority_evidence_hash,
            "competentDecisionIds": [],
            "unresolvedDecisionIds": [],
            "featuredTruthBlockedDecisionIds": [],
        }
        return {**core, "sha256": canonical_sha256(core)}

    analysis_rows = tuple(analyses)
    known_decision_ids = {str(item) for item in decision_kinds}
    supplied_competent_ids = {
        str(item) for item in competent_decision_ids
    }
    supplied_unresolved_ids = {
        str(item) for item in unresolved_decision_ids
    }
    authority = dict(authority_evidence or {})
    evidence_competent_ids = {
        str(item) for item in authority.get("competentDecisionIds") or ()
    }
    evidence_unresolved_ids = {
        str(item) for item in authority.get("unresolvedDecisionIds") or ()
    }
    status_binding_mismatch = bool(authority) and (
        supplied_competent_ids != evidence_competent_ids
        or supplied_unresolved_ids != evidence_unresolved_ids
    )
    # Union both sides when the caller and evidence disagree. The mismatch
    # rejects the evaluation globally, while the union prevents either side
    # from leaking a featured candidate before that rejection is consumed.
    competent_ids = supplied_competent_ids | evidence_competent_ids
    unresolved_ids = supplied_unresolved_ids | evidence_unresolved_ids
    status_conflicts = competent_ids & unresolved_ids
    # Unresolved is the fail-closed precedence; never permit a conflicting ID
    # to retain competence-only resolution.
    competent_ids -= unresolved_ids
    unknown_authority_ids = (competent_ids | unresolved_ids) - known_decision_ids
    featured_blocked_ids = competent_ids | unresolved_ids
    declared_authority_hash = str(
        authority_evidence_hash
        or authority.get("sha256")
        or ""
    )
    authority_core = {
        key: value for key, value in authority.items() if key != "sha256"
    }
    valid_authority_hash = bool(
        authority
        and len(declared_authority_hash) == 64
        and declared_authority_hash == declared_authority_hash.lower()
        and all(
            character in "0123456789abcdef"
            for character in declared_authority_hash
        )
        and str(authority.get("sha256") or "")
        == declared_authority_hash
        and canonical_sha256(authority_core) == declared_authority_hash
    )
    tactical_by_decision = {
        str(item.oracle.decision_id): item
        for item in tactical_verifications
    }
    rows: list[dict[str, Any]] = []
    for item in perspectives:
        if item.truth_mode != TruthMode.ACTOR_OBSERVER or bool(item.forced):
            continue
        decision_id = str(item.decision_id)
        decision_kind = str(decision_kinds.get(decision_id) or "")
        score = profile.score(item)
        try:
            breakdown = profile.breakdown(item).to_dict()
        except ValueError:
            breakdown = None
        gate_failures = list(score.rejection_reasons)
        if decision_id in featured_blocked_ids:
            gate_failures.append("adjudication_featured_truth_blocked")
        tactical_required = bool(
            profile.require_verified_tactical_play and decision_kind == "play"
        )
        verification = tactical_by_decision.get(decision_id)
        tactical_verified = _verified_tactical_play(verification)
        if not decision_kind:
            gate_failures.append("decision_kind_unavailable")
        if tactical_required and not tactical_verified:
            gate_failures.append("verified_tactical_play")

        surrounding = None
        if (
            not gate_failures
            and bool(profile.requires_surrounding_competence)
        ):
            try:
                surrounding_value = profile.qualify_surrounding(
                    analysis_rows,
                    candidate_decision_id=decision_id,
                    trace_analysis_hash=str(analysis_hash),
                    unresolved_decision_ids=unresolved_ids,
                    competence_verified_decision_ids=competent_ids,
                )
            except ValueError:
                gate_failures.append("surrounding_evidence_unavailable")
            else:
                surrounding = surrounding_value.to_dict()
                gate_failures.extend(
                    f"surrounding_{reason}"
                    for reason in surrounding_value.rejection_reasons
                )

        gate_failures = list(dict.fromkeys(gate_failures))
        rows.append({
            "decisionId": decision_id,
            "actingSeat": item.acting_seat.name.lower(),
            "decisionKind": decision_kind or None,
            "truthLabel": item.truth_label.value,
            "confidence": float(item.confidence),
            "score": score.to_dict(),
            "breakdown": breakdown,
            "verifiedTacticalPlayRequired": tactical_required,
            "verifiedTacticalPlay": (
                tactical_verified if tactical_required else None
            ),
            "tacticalVerificationHash": (
                str(verification.sha256) if verification is not None else None
            ),
            "surroundingCompetence": surrounding,
            "gatePassed": not gate_failures,
            "gateFailures": gate_failures,
        })

    rows.sort(
        key=lambda value: (
            not bool(value["gatePassed"]),
            -float(value["score"]["score"]),
            str(value["decisionId"]),
        )
    )
    qualified_ids = sorted(
        str(item["decisionId"])
        for item in rows
        if bool(item["gatePassed"])
    )
    reasons: list[str] = []
    if featured_blocked_ids and not valid_authority_hash:
        reasons.append("adjudication_authority_evidence_unbound")
    if unknown_authority_ids:
        reasons.append("adjudication_authority_unknown_decision")
    if status_conflicts:
        reasons.append("adjudication_authority_status_conflict")
    if status_binding_mismatch:
        reasons.append("adjudication_authority_status_mismatch")
    if not rows:
        reasons.append(DECISION_SELECTION_NO_ACTOR_EVIDENCE)
    elif not qualified_ids:
        reasons.append(DECISION_SELECTION_NO_QUALIFIED_DECISION)
        reasons.extend(
            str(reason)
            for row in rows
            for reason in row["gateFailures"]
        )
    reasons = list(dict.fromkeys(reasons))
    core = {
        "schemaVersion": 2,
        "profileHash": str(profile.sha256),
        "analysisHash": str(analysis_hash),
        "evaluatedDecisionCount": len(rows),
        "qualifiedDecisionCount": len(qualified_ids),
        "qualifiedDecisionIds": qualified_ids,
        "passed": not reasons,
        "rejectionReasons": reasons,
        "authorityEvidenceHash": declared_authority_hash or None,
        "competentDecisionIds": sorted(competent_ids),
        "unresolvedDecisionIds": sorted(unresolved_ids),
        "featuredTruthBlockedDecisionIds": sorted(featured_blocked_ids),
        "decisions": copy.deepcopy(rows),
    }
    return {**core, "sha256": canonical_sha256(core)}


def apply_decision_selection_qualification_v1(
    qualification: Mapping[str, Any],
    decision_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge decision-selection evidence without losing earlier blockers."""

    current = copy.deepcopy(dict(qualification))
    passed = decision_selection.get("passed")
    if not isinstance(passed, bool):
        passed = False
        reasons = [DECISION_SELECTION_PROFILE_UNAVAILABLE]
    else:
        reasons = [
            str(item)
            for item in decision_selection.get("rejectionReasons") or ()
        ]
    failures = list(current.get("failures") or ())
    if not passed:
        failures.extend(f"decision_score_profile_gate:{item}" for item in reasons)
    current["failures"] = list(dict.fromkeys(failures))
    current["qualified"] = bool(current.get("qualified")) and bool(passed)
    if not current["qualified"]:
        current["truthLabel"] = "rejected"
    current["decisionSelection"] = {
        "schemaVersion": 2,
        "profileHash": decision_selection.get("profileHash"),
        "evidenceHash": decision_selection.get("sha256"),
        "evaluatedDecisionCount": int(
            decision_selection.get("evaluatedDecisionCount", 0)
        ),
        "qualifiedDecisionCount": int(
            decision_selection.get("qualifiedDecisionCount", 0)
        ),
        "qualifiedDecisionIds": list(
            decision_selection.get("qualifiedDecisionIds") or ()
        ),
        "passed": bool(passed),
        "rejectionReasons": reasons,
    }
    return current
