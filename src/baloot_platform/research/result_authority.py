"""Immutable authority and execution contracts for OR-Tools Research results.

Research qualification is bound to semantic CP-SAT artifacts.  Downstream
engine compatibility and machine execution measurements are integrity checked
separately and therefore cannot change the semantic Research result identity.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
import json
import math
import re
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from baloot_platform.core.exact_state_view import build_observer_safe_exact_state_v1
from baloot_platform.core.enums import Seat, Suit, Team
from baloot_platform.core.cards import Card
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.project_ir import ProjectDeclarationActionV1
from baloot_platform.tactics import DELAYED_PAYOFF_TACTIC_IDS, NATIVE_PROFILE_TACTIC_IDS, TacticalDetectorProfileV1, TacticalKind

from .scenario_contracts import ExactTracePrefixV1, ScenarioExpressionV1, ScenarioPlayingQualityPolicyV1, ScenarioProofRequestV1, ScenarioQualityExecutionBudgetV1, canonical_sha256
from .scenario_registry_v2 import scenario_predicate_definitions_v2
from .scenario_proof_tactic_predicates_v2 import PROOF_TACTIC_PREDICATE_REGISTRATIONS_V2
from .scenario_contracts import ScenarioProofBudgetV1
from .decision_selection import evaluate_round_interestingness_v1, interest_admission_summary_v1


def validated_trace_lesson_tags_v1(trace, *, rules):
    """Reproduce tags from an already validated trace before projecting them."""
    if 'lessonTags' not in trace:
        return None
    from .lesson_tags import derive_trace_lesson_tags_v1
    from .ortools_exact_state import OrToolsNormalizedExactStateV1, OrToolsExactNormalizationReceiptV1
    raw = trace.get('exactNormalization')
    receipt = trace.get('normalizationReceipt')
    if not isinstance(raw, Mapping) or not isinstance(receipt, Mapping) or rules is None:
        raise ValueError('lesson tags require owned normalization and frozen rules')
    try:
        normalized = OrToolsNormalizedExactStateV1(
            exact_state_id=raw['exactStateId'], prefix_hash=raw['prefixHash'], rules_hash=raw['rulesHash'],
            state_hash=raw['stateHash'], phase=raw['phase'], decision_kind=raw['decisionKind'],
            next_actor=raw['nextActor'], legal_actions=tuple(raw['legalActions']), state=raw['state'],
            project_declarations_verified=raw['projectDeclarationsVerified'],
            project_assertions_verified=raw['projectAssertionsVerified'], event_chain_hash=raw['eventChainHash'],
            sha256=raw['sha256'], normalization_receipt=OrToolsExactNormalizationReceiptV1(receipt))
        derived = derive_trace_lesson_tags_v1(trace, normalized=normalized, rules=rules)
    except (KeyError, TypeError) as exc:
        raise ValueError('lesson tag normalization is malformed') from exc
    if derived is None or derived != trace['lessonTags']:
        raise ValueError('lesson tags disagree with the owned proof and public history')
    return copy.deepcopy(derived)


def validate_generation_result_binding_v1(recipe, trace, snapshot, *, qualified):
    from .family_catalog import build_family_generation_request_v1, ResearchFamilyCatalogV1, TagQueryV1

    generation = recipe.get('generation')
    if generation is None:
        if 'generationRequest' in trace:
            raise ValueError('generation trace has no owning recipe request')
        return
    request = build_family_generation_request_v1(generation)
    catalog = (snapshot or {}).get('components', {}).get('challenge', {}).get('payload', {}).get(
        'scenarioResearchV4', {}).get('qualityExecution', {}).get('familyCatalog')
    if not isinstance(catalog, Mapping) or ResearchFamilyCatalogV1.build(catalog).sha256 != request['catalogHash']:
        raise ValueError('generation request differs from the frozen bundle catalog')
    if trace.get('generationRequest') != request:
        raise ValueError('generation trace differs from its owning recipe request')
    if qualified and (not isinstance(trace.get('lessonTags'), Mapping) or not
            TagQueryV1.build(request['tagQuery']).matches(trace['lessonTags'].get('tags', {}))):
        raise ValueError('qualified generation result does not match its requested tags')


ORTOOLS_RESULT_IDENTITY_KIND = "ortools_research_result_identity_v2"
ORTOOLS_RESULT_IDENTITY_SCHEMA_VERSION = 2
ORTOOLS_RESULT_IDENTITY_V3_KIND = "ortools_research_result_identity_v3"
ORTOOLS_RESULT_IDENTITY_V3_SCHEMA_VERSION = 3
ORTOOLS_RESEARCH_BACKEND = "ortools_cp_sat_v2"
ORTOOLS_RESEARCH_BACKENDS = frozenset({ORTOOLS_RESEARCH_BACKEND, "ortools_cp_sat_v3"})
CP_SAT_QUALIFICATION_AUTHORITY = "cp_sat"
_RUNTIME_COMPATIBILITY_EVIDENCE_KIND = "runtime_compatibility_evidence_v1"
_HASH = re.compile(r"[0-9a-f]{64}")


def _native_interest_evidence_v1(snapshot, trace, *, verified_quality, featured_state_id,
                                 research_authority=None):
    policy = snapshot["components"]["challenge"]["payload"].get(
        "scenarioResearchV4", {}).get("qualityExecution", {}).get("videoInterestingness")
    if not isinstance(policy, Mapping) or policy.get("schemaVersion") != 4:
        return None
    receipt = dict(trace.get("playingQualityEvidence") or {}).get("receipt", {}) if verified_quality else {}
    version = _backend_version_tuple((research_authority or {}).get("backendVersion"))
    # Preserve owned historical hashes; fresh search/render never request this mode.
    historical = ((research_authority or {}).get("backendId") == "ortools_cp_sat_v3"
        and version is not None and _V3_INTEREST_ARTIFACT_MINIMUM_BACKEND_VERSION <= version
        < _V3_BASELINE_OUTCOME_INTEREST_MINIMUM_BACKEND_VERSION)
    return evaluate_round_interestingness_v1(
        receipt, trace.get("score") or {}, policy, featured_state_id=featured_state_id or "",
        _historical_any_alternative=historical)


def _native_interest_admission_v1(snapshot, trace, *, verified_quality, featured_state_id,
                                  research_authority=None):
    evidence = _native_interest_evidence_v1(snapshot, trace,
        verified_quality=verified_quality, featured_state_id=featured_state_id,
        research_authority=research_authority)
    return interest_admission_summary_v1(evidence) if evidence is not None else None


def validate_native_interest_artifact_v1(result, trace, snapshot, evidence):
    """Bind retained editorial evidence to the result's verified source documents."""
    qualification = dict(result.get("qualification") or {})
    expected = _native_interest_evidence_v1(snapshot, trace,
        verified_quality=qualification.get("allDecisionPolicyAuditComplete") is True,
        featured_state_id=dict(result.get("scenarioProjection") or {}).get("exactStateId"),
        research_authority=result.get("researchAuthority"))
    if (expected is None or evidence != expected
            or qualification.get("interestAdmission") != interest_admission_summary_v1(evidence)):
        raise RuntimeError("stored OR-Tools Research V3 interest evidence binding mismatch")


def _validate_native_interest_admission_v1(result, trace, snapshot, *, verified_quality):
    identity = dict(dict(result.get("provenance") or {}).get("resultIdentity") or {})
    if "interestAdmission" not in dict(identity.get("authorityArtifactHashes") or {}):
        authority = dict(result.get("researchAuthority") or {})
        version = _backend_version_tuple(authority.get("backendVersion"))
        policy = (snapshot.get("components", {}).get("challenge", {}).get("payload", {})
                  .get("scenarioResearchV4", {}).get("qualityExecution", {}).get("videoInterestingness")
                  if isinstance(snapshot, Mapping) else None)
        if (authority.get("backendId") == "ortools_cp_sat_v3"
                and (version is None or version >= _V3_INTEREST_ARTIFACT_MINIMUM_BACKEND_VERSION)
                and isinstance(policy, Mapping) and policy.get("schemaVersion") == 4):
            raise ValueError("stored OR-Tools Research V3 interest evidence artifact is missing")
        # Historical summaries have no retained gate document; never reinterpret them with a newer gate.
        return
    recorded = dict(result.get("qualification") or {}).get("interestAdmission")
    if not isinstance(snapshot, Mapping):
        if recorded is not None:
            raise ValueError("stored OR-Tools Research V3 interest admission snapshot is missing")
        return
    expected = _native_interest_admission_v1(snapshot, trace, verified_quality=verified_quality,
        featured_state_id=dict(result.get("scenarioProjection") or {}).get("exactStateId"),
        research_authority=result.get("researchAuthority"))
    if expected != recorded:
        raise ValueError("stored OR-Tools Research V3 interest admission binding mismatch")
    if expected is not None:
        score = min(100.0, float(expected["score"])) if result.get("disposition") == "qualified" else 0.0
        if result.get("score") != score:
            raise ValueError("stored OR-Tools Research V3 interest score binding mismatch")


_NATIVE_OBSERVED_BID_STRATEGY_UPPERS = frozenset({
    "native_observed_bid_opponent_policy_upper_v1", "native_observed_bid_refuted_upper_v1",
})


def validate_native_prefix_threshold_v1(value: Mapping[str, Any], *, rules: RuleBook,
        scenario, source_query: Mapping[str, Any], target, prefix_play_count: int,
        required_threshold: int, maximum_seconds: float, maximum_conflicts: int,
        cancel_check=None) -> int:
    """Reconstruct the owned native query; a rehashed claim alone is not proof."""
    from .ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1

    if (not isinstance(value, Mapping)
            or value.get("sha256") != canonical_sha256({k: v for k, v in value.items() if k != "sha256"})
            or value.get("kind") != "native_counterexample_prefix_threshold_v1"
            or value.get("status") != "upper_bound_complete"
            or value.get("solverStatus") != "infeasible"):
        raise ValueError("prefix threshold requires a complete owned absence receipt")
    limits = value.get("limits")
    if (not isinstance(limits, Mapping) or set(limits) != {"seconds", "conflicts"}
            or type(limits["seconds"]) not in (int, float) or not math.isfinite(limits["seconds"])
            or limits["seconds"] <= 0 or type(limits["conflicts"]) is not int or limits["conflicts"] <= 0):
        raise ValueError("prefix threshold resource receipt is invalid")
    expected = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._bound_native_counterexample_prefix(
        scenario, source_query, target, prefix_play_count=prefix_play_count,
        required_threshold=required_threshold, maximum_seconds=maximum_seconds,
        maximum_conflicts=maximum_conflicts, cancel_check=cancel_check)
    if expected["status"] != "upper_bound_complete":
        raise ValueError("prefix threshold native reverification did not establish absence")
    def semantic(document):
        return canonical_sha256({k: v for k, v in document.items() if k not in {"sha256", "limits"}})
    if semantic(value) != semantic(expected):
        raise ValueError("prefix threshold native evidence mismatch")
    return expected["upperBound"]


def validate_native_prefix_minimax_v1(value: Mapping[str, Any], *, rules: RuleBook,
        scenario, source_query: Mapping[str, Any], target, prefix_play_count: int,
        minimax_budget: ScenarioProofBudgetV1, maximum_seconds: float,
        maximum_conflicts: int, cancel_check=None) -> int:
    """Rebuild the owned complete-state suffix; a rehashed value is not proof."""
    from .ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1

    if (not isinstance(value, Mapping)
            or value.get("sha256") != canonical_sha256({k: v for k, v in value.items() if k != "sha256"})
            or value.get("kind") != "native_counterexample_prefix_minimax_v1"
            or value.get("status") != "upper_bound_complete"):
        raise ValueError("prefix minimax requires a complete owned receipt")
    limits = value.get("limits")
    if (not isinstance(limits, Mapping) or set(limits) != {"seconds", "conflicts"}
            or type(limits["seconds"]) not in (int, float) or not math.isfinite(limits["seconds"])
            or limits["seconds"] <= 0 or type(limits["conflicts"]) is not int or limits["conflicts"] <= 0):
        raise ValueError("prefix minimax resource receipt is invalid")
    expected = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._bound_native_counterexample_prefix(
        scenario, source_query, target, prefix_play_count=prefix_play_count,
        minimax_budget=minimax_budget, maximum_seconds=maximum_seconds,
        maximum_conflicts=maximum_conflicts, cancel_check=cancel_check)
    if expected["status"] != "upper_bound_complete":
        raise ValueError("prefix minimax native reverification did not complete")
    def semantic(document):
        return canonical_sha256({k: v for k, v in document.items() if k not in {"sha256", "limits"}})
    if semantic(value) != semantic(expected):
        raise ValueError("prefix minimax native evidence mismatch")
    return expected["upperBound"]


def validate_native_constant_escalation_bounds_v1(value: Mapping[str, Any], *, rules: RuleBook,
        source_prefix: ExactTracePrefixV1, normalized, opportunity_order, escalation_events,
        maximum_seconds: float, cancel_check=None) -> None:
    """Verify an internal constant-bound receipt by fresh native reconstruction.

    Rehashed status/value claims are not proof. This reexecutes the owned CP-SAT
    models under an explicit verification deadline; no runtime engine is used.
    It is not full-round admission or an information-set coverage certificate.
    """
    import time
    from .ortools_exact_state import OrToolsNormalizedExactStateV1, _effective_prefix_hash
    from .ortools_information_set_compiler import InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1

    if (not isinstance(rules, RuleBook) or not isinstance(source_prefix, ExactTracePrefixV1)
            or not isinstance(normalized, OrToolsNormalizedExactStateV1)
            or not isinstance(value, Mapping)):
        raise TypeError("constant escalation bounds require typed frozen context")
    if (type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
            or maximum_seconds <= 0 or cancel_check is not None and not callable(cancel_check)):
        raise ValueError("constant escalation verification requires a finite deadline")
    deadline = time.monotonic() + maximum_seconds
    normalized.require_integrity()
    if (normalized.decision_kind != "escalation" or normalized.rules_hash != rules.sha256
            or normalized.prefix_hash != _effective_prefix_hash(source_prefix)):
        raise ValueError("constant escalation normalized ownership mismatch")
    target = Seat.parse(normalized.next_actor).team
    version = value.get("schemaVersion")
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("constant escalation receipt version is unsupported")
    expected = {"schemaVersion": version, "kind": "native_post_auction_policy_bounds_v1",
        "status": "scoped_constant_bound_converged", "visibility": "complete_state",
        "rulesHash": rules.sha256, "prefixHash": normalized.prefix_hash,
        "sourcePrefixHash": source_prefix.sha256, "exactStateId": normalized.exact_state_id,
        "stateHash": normalized.state_hash, "decisionKind": "escalation",
        "utility": "awarded_point_margin", "continuationBoundary": "current_deal_completion",
        "targetTeam": target.name.lower()}
    if (any(value.get(key) != item for key, item in expected.items())
            or type(value.get("schemaVersion")) is not int or value.get("strategicCoverage") is not False):
        raise ValueError("constant escalation receipt ownership or completion mismatch")
    limits, proof = value.get("limits"), value.get("strategy")
    if (not isinstance(limits, Mapping) or not isinstance(proof, Mapping)
            or any(type(limits.get(key)) is not int or limits[key] < 1 for key in
                ("iterations", "queries", "observations", "modelBytes", "conflictsPerSolve", "depth"))
            or type(limits.get("seconds")) not in {int, float} or not math.isfinite(limits["seconds"])
            or limits["seconds"] <= 0 or type(value.get("queriesUsed")) is not int
            or not 0 < value["queriesUsed"] <= limits["queries"]
            or proof.get("kind") != "native_constant_escalation_bounds_v1"
            or proof.get("status") != "bounds_coincide" or proof.get("targetTeam") != target.name.lower()
            or type(proof.get("lowerBound")) is not int or type(proof.get("upperBound")) is not int
            or proof["lowerBound"] != proof["upperBound"]
            or proof.get("selectedAction") not in {row["code"] for row in normalized.legal_actions}
            or value.get("selectedRootAction") not in {None, proof["selectedAction"]}):
        raise ValueError("constant escalation bounds or limits are incomplete")

    def semantic(document):
        if isinstance(document, Mapping):
            canonical_receipt = document.get("kind") in {
                "native_post_auction_policy_bounds_v1", "native_constant_escalation_bounds_v1",
                "native_continuation_depth_query_v1", "native_round_outcome_query_v1",
                "native_constant_policy_round_query_v1", "native_constant_escalation_policy_binding_v1"}
            if canonical_receipt and document.get("sha256") != canonical_sha256(
                    {key: item for key, item in document.items() if key != "sha256"}):
                raise ValueError("constant escalation nested receipt checksum mismatch")
            # Runtime allowances differ on fresh replay; constraints and outcomes
            # must not. Verify each recorded allowance before excluding it.
            runtime_receipt = document.get("kind") == "native_continuation_depth_query_v1"
            if runtime_receipt and (type(document.get("maximumSeconds")) not in {int, float}
                    or not 0 < document["maximumSeconds"] <= limits["seconds"]):
                raise ValueError("constant escalation runtime allowance mismatch")
            return {key: "verified" if key == "sha256" and canonical_receipt else semantic(item)
                    for key, item in document.items() if not (key == "maximumSeconds" and runtime_receipt)}
        if isinstance(document, (list, tuple)):
            return [semantic(item) for item in document]
        return document

    recorded = semantic(value)
    if cancel_check is not None and cancel_check():
        raise InformationSetCompilationStopped("cancelled", stage="constant_escalation_verification")
    if time.monotonic() >= deadline:
        raise InformationSetCompilationStopped("maximum_seconds", stage="constant_escalation_verification")
    try:
        replay = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._solve_native_post_auction_policy_bounds(
            normalized, source_prefix, target, selected_root_action=value.get("selectedRootAction"),
            escalation_receipt_version=version,
            opportunity_order=opportunity_order, escalation_events=escalation_events,
            maximum_iterations=limits["iterations"], maximum_queries=limits["queries"],
            maximum_observations=limits["observations"], maximum_model_bytes=limits["modelBytes"],
            maximum_conflicts=limits["conflictsPerSolve"], maximum_depth=limits["depth"],
            maximum_seconds=limits["seconds"],
            cancel_check=lambda: time.monotonic() >= deadline or bool(cancel_check and cancel_check()))
    except InformationSetCompilationStopped as stopped:
        if time.monotonic() >= deadline and not (cancel_check and cancel_check()):
            raise InformationSetCompilationStopped("maximum_seconds", stage="constant_escalation_verification",
                evidence={"sourceReceipt": stopped.partial_receipt}) from stopped
        raise
    if semantic(replay) != recorded:
        raise ValueError("constant escalation native reconstruction differs from retained evidence")


def validate_native_constant_escalation_budget_v1(certificate, budget, *, maximum_conflicts):
    """Keep retained native query limits inside the selected frozen tier."""
    if (not isinstance(budget, ScenarioProofBudgetV1) or budget.native_observed is None
            or type(maximum_conflicts) is not int or maximum_conflicts < 1):
        raise ValueError("constant escalation quality requires frozen native proof and conflict budgets")
    native = budget.native_observed
    caps = {"iterations": native.maximum_iterations, "queries": native.maximum_queries,
        "observations": native.maximum_observations, "modelBytes": budget.maximum_cache_bytes,
        "conflictsPerSolve": maximum_conflicts,
        "depth": min(budget.maximum_depth, budget.maximum_horizon_actions)}
    limits = certificate.get("limits")
    if not isinstance(limits, Mapping) or any(type(limits.get(key)) is not int
            or not 0 < limits[key] <= cap for key, cap in caps.items()):
        raise ValueError("constant escalation quality exceeds frozen proof limits")
    seconds = limits.get("seconds")
    if (type(seconds) not in {int, float} or not math.isfinite(seconds)
            or not 0 < seconds <= min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds)):
        raise ValueError("constant escalation quality exceeds frozen execution limits")


def build_native_constant_escalation_quality_artifact_v1(
        certificate, *, rules, source_prefix, normalized, request, opportunity_order,
        escalation_events, budget,
        maximum_seconds, maximum_conflicts, cancel_check=None):
    """Bind verified native bounds to selected-action quality, not root counts."""
    import time
    from ortools.sat.python import cp_model
    from .ortools_information_set_compiler import InformationSetCompilationStopped
    from .ortools_information_set_proof import build_native_bid_dominance_model_v1, _solve_cp_sat_with_cancel_v1

    if (not isinstance(request, ScenarioProofRequestV1) or request.mode != "exact"
            or request.visibility != "complete_state" or request.utility != "awarded_point_margin"
            or request.target_team != certificate.get("targetTeam")
            or request.selected_action != certificate.get("selectedRootAction")
            or type(maximum_conflicts) is not int or maximum_conflicts < 1
            or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
            or maximum_seconds <= 0):
        raise ValueError("constant escalation quality requires an owned exact request and finite limits")
    validate_native_constant_escalation_budget_v1(certificate, budget, maximum_conflicts=maximum_conflicts)
    if request.budget_profile != budget.profile_id:
        raise ValueError("constant escalation quality proof budget identity mismatch")
    deadline = time.monotonic() + maximum_seconds
    validate_native_constant_escalation_bounds_v1(certificate, rules=rules, source_prefix=source_prefix,
        normalized=normalized, opportunity_order=opportunity_order, escalation_events=escalation_events,
        maximum_seconds=maximum_seconds, cancel_check=cancel_check)
    bound = certificate["strategy"]
    selected = bound["selectedAction"]
    retained = next(row for row in certificate["preparation"]["partitions"] if row["retained"])
    information = retained["rootQuery"]["candidate"]["escalationObservations"][len(escalation_events)]["information"]
    actions = sorted(row["code"] for row in normalized.legal_actions)
    alternatives = [{"action": action, "upperBound": bound["upperBound"],
        "sourceProofHash": certificate["sha256"]} for action in actions if action != selected]
    # Reuse the numerical dominance compiler; no bid rules enter this model.
    model, selected_value, claim, best, regret, bindings = build_native_bid_dominance_model_v1(
        bound["lowerBound"], alternatives)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise InformationSetCompilationStopped("maximum_seconds", stage="escalation_action_quality")
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = remaining
    solver.parameters.max_number_of_conflicts = maximum_conflicts
    status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
    if status in {cp_model.INFEASIBLE, cp_model.MODEL_INVALID}:
        raise ValueError("constant escalation qualification model disagreement")
    if status != cp_model.OPTIMAL or time.monotonic() >= deadline:
        raise InformationSetCompilationStopped("qualification_incomplete", stage="escalation_action_quality")
    if cancel_check is not None and cancel_check():
        raise InformationSetCompilationStopped("cancelled", stage="escalation_action_quality")
    if not solver.boolean_value(claim):
        raise ValueError("constant escalation certified bounds do not establish optimality")
    scope = "selected_escalation_current_deal"
    quality = {"schemaVersion": 1, "authority": "cp_sat", "scope": "compiled_complete_state",
        "rootActorRelation": "target", "actualAction": selected,
        "optimalActions": [selected], "optimalActionsComplete": False, "optimalActionCount": None,
        "optimalActionCountLowerBound": 1, "optimalActionCountUpperBound": len(actions),
        "bestValue": int(solver.value(best)), "actualValue": int(solver.value(selected_value)),
        "regret": int(solver.value(regret)), "actualActionOptimal": bool(solver.boolean_value(claim)),
        "solverStatus": "optimal", "variableCount": len(model.proto.variables),
        "constraintCount": len(model.proto.constraints),
        "analysisScope": {"mode": "exact", "coverage": scope,
            "utility": request.utility, "exactStateId": normalized.exact_state_id},
        "sourceProofHash": certificate["sha256"]}
    proto = str(model.proto)
    quality["sha256"] = canonical_sha256({**quality, "modelProto": proto})
    model_document = {"schemaVersion": 1, "kind": "native_escalation_selected_action_quality_model_v1",
        "rootActorSeat": normalized.next_actor, "sourceProofHash": certificate["sha256"],
        "modelProto": proto, "selectedValueVariable": selected_value.name, "claimVariable": claim.name,
        "bestValueVariable": best.name, "regretVariable": regret.name,
        "alternatives": bindings, "actionQuality": quality}
    allocation = {"schemaVersion": 1, "kind": "native_complete_state_quality_v1",
        "sourcePrefixHash": source_prefix.sha256, "exactStateId": normalized.exact_state_id,
        "stateHash": normalized.state_hash, "rulesHash": rules.sha256,
        "sourcePrefix": source_prefix.to_dict(), "normalizedState": normalized.to_dict(),
        "normalizationReceipt": normalized.normalization_receipt.to_dict(),
        "opportunityOrder": list(opportunity_order), "escalationEvents": [dict(row) for row in escalation_events],
        "priorNonAnnouncingSeats": list(information["completeWindowDeclineOrder"][-1]),
        "priorWindowRecall": {seat: list(windows) for seat, windows in information["completePriorWindowDeclines"].items()
                              if windows}}
    allocation["sha256"] = canonical_sha256(allocation)
    proof = {"schemaVersion": 1, "completionStatus": "exact_complete", "completionScope": scope,
        "claimSatisfied": None, "selectedAction": selected, "winningRootActionCount": None,
        "winningRootActionLowerBound": 0, "winningRootActionUpperBound": len(actions),
        "rootActions": [{"action": action, "value": None, "satisfies": None} for action in actions],
        "identities": {"graphHash": certificate["sha256"], "modelHash": canonical_sha256(model_document),
            "proofRequestHash": request.sha256}}
    proof["sha256"] = canonical_sha256(proof)
    return {"request": request.to_dict(), "proof": proof, "proofModel": model_document,
        "graph": copy.deepcopy(dict(certificate)), "allocationModel": allocation}


def validate_native_constant_escalation_quality_artifact_v1(artifact, **context):
    """Rebuild the full envelope from caller-owned context and fresh native proof."""
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("graph"), Mapping):
        raise ValueError("constant escalation quality artifact is malformed")
    expected = build_native_constant_escalation_quality_artifact_v1(artifact["graph"], **context)
    if canonical_sha256(artifact) != canonical_sha256(expected):
        raise ValueError("constant escalation quality artifact binding mismatch")


def build_native_project_dominance_quality_artifact_v1(certificate, *, rules, source_prefix,
        normalized, request, maximum_seconds, budget, maximum_conflicts, cancel_check=None, maximum_depth=None):
    """Adapt fresh pointwise dominance to action quality, never outcome truth."""
    from .ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1

    if not isinstance(certificate, Mapping) or certificate.get("status") != "strictly_dominant":
        raise ValueError("project quality requires strictly dominant selected-action evidence")
    generated = certificate.get("kind") == "native_last_project_selection_v1"
    selection_matches = (request.selected_action is None and certificate.get("request") == request.to_dict()
        if generated else request.selected_action == certificate.get("selectedAction"))
    if not selection_matches:
        raise ValueError("project quality requires strictly dominant selected-action evidence")
    if (not isinstance(budget, ScenarioProofBudgetV1) or budget.native_observed is None
            or request.budget_profile != budget.profile_id
            or maximum_depth is not None and (type(maximum_depth) is not int or maximum_depth < 1)
            or certificate.get("limits") != {"profileId": budget.profile_id, "budgetHash": budget.sha256,
                "maximumQueries": budget.native_observed.maximum_queries,
                "maximumModelBytes": budget.maximum_cache_bytes, "maximumConflicts": maximum_conflicts,
                "maximumDepth": min(budget.maximum_depth, budget.maximum_horizon_actions,
                    maximum_depth if maximum_depth is not None else budget.maximum_depth)}):
        raise ValueError("project quality frozen proof budget mismatch")
    selected = certificate["selectedAction"]
    raw = request.to_dict()
    raw.pop("sha256")
    raw["selectedAction"] = selected
    owned_request = ScenarioProofRequestV1.build(raw)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    if generated:
        fresh = compiler._select_last_project_dominance(normalized, source_prefix, request,
            maximum_seconds=maximum_seconds, cancel_check=cancel_check,
            budget=budget, maximum_conflicts=maximum_conflicts, maximum_depth=maximum_depth)
        if fresh != certificate:
            raise ValueError("project selection differs from fresh native proof")
    else:
        compiler._verify_last_project_dominance(certificate, normalized, source_prefix, owned_request,
            maximum_seconds=maximum_seconds, cancel_check=cancel_check,
            budget=budget, maximum_conflicts=maximum_conflicts, maximum_depth=maximum_depth)
    actions = list(certificate["legalActions"])
    scope = "selected_final_project_complete_state"
    quality = {"schemaVersion": 1, "authority": "cp_sat", "scope": "compiled_complete_state",
        "rootActorRelation": "target", "actualAction": selected, "optimalActions": [selected],
        "optimalActionsComplete": True, "optimalActionCount": 1,
        "bestValue": None, "actualValue": None, "regret": 0, "actualActionOptimal": True,
        "solverStatus": "optimal", "sourceProofHash": certificate["sha256"],
        "analysisScope": {"mode": "exact", "coverage": scope, "utility": request.utility,
            "exactStateId": normalized.exact_state_id}}
    # The native model is reconstructed by the verifier; no constant-valued
    # qualification model substitutes for its SAT/INFEASIBLE query evidence.
    quality["sha256"] = canonical_sha256({**quality, "modelProto": None})
    model = {"schemaVersion": 1, "kind": "native_project_dominance_quality_model_v1",
        "rootActorSeat": normalized.next_actor, "sourceProofHash": certificate["sha256"],
        "modelProto": None, "actionQuality": quality}
    allocation = {"schemaVersion": 1, "kind": "native_complete_state_quality_v1",
        "sourcePrefixHash": source_prefix.sha256, "sourcePrefix": source_prefix.to_dict(),
        "exactStateId": normalized.exact_state_id, "stateHash": normalized.state_hash,
        "rulesHash": rules.sha256, "proofBudgetProfile": budget.profile_id, "proofBudgetHash": budget.sha256}
    allocation["sha256"] = canonical_sha256(allocation)
    proof = {"schemaVersion": 1, "completionStatus": "exact_complete",
        "completionScope": "action_optimality_only", "claimSatisfied": None,
        "selectedAction": selected, "winningRootActionCount": None,
        "winningRootActionLowerBound": 0, "winningRootActionUpperBound": len(actions),
        "rootActions": [{"action": action, "value": None, "satisfies": None} for action in actions],
        "identities": {"graphHash": certificate["sha256"], "modelHash": canonical_sha256(model),
            "proofRequestHash": request.sha256}}
    proof["sha256"] = canonical_sha256(proof)
    return {"request": request.to_dict(), "proof": proof, "proofModel": model,
        "graph": copy.deepcopy(dict(certificate)), "allocationModel": allocation}


def validate_native_project_dominance_quality_artifact_v1(artifact, **context):
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("graph"), Mapping):
        raise ValueError("project quality artifact is malformed")
    fresh = build_native_project_dominance_quality_artifact_v1(artifact["graph"], **context)
    if canonical_sha256(fresh) != canonical_sha256(artifact):
        raise ValueError("project quality artifact binding mismatch")


def reverify_native_observed_play_bounds_v1(value, *, rules, source_prefix, normalized,
        selected_root_action, opportunity_order, escalation_events, budget,
        maximum_conflicts, maximum_seconds, cancel_check=None):
    if getattr(normalized, "decision_kind", None) != "play":
        raise ValueError("observed play verification requires a play root")
    return _reverify_native_observed_post_auction_bounds_v1(value, rules=rules, source_prefix=source_prefix,
        normalized=normalized, selected_root_action=selected_root_action, opportunity_order=opportunity_order,
        escalation_events=escalation_events, budget=budget, maximum_conflicts=maximum_conflicts,
        maximum_seconds=maximum_seconds, cancel_check=cancel_check)


def reverify_native_observed_project_bounds_v1(value, *, rules, source_prefix, normalized,
        selected_root_action, opportunity_order, escalation_events, budget,
        maximum_conflicts, maximum_seconds, cancel_check=None):
    if getattr(normalized, "decision_kind", None) != "project":
        raise ValueError("observed project verification requires a project root")
    return _reverify_native_observed_post_auction_bounds_v1(value, rules=rules, source_prefix=source_prefix,
        normalized=normalized, selected_root_action=selected_root_action, opportunity_order=opportunity_order,
        escalation_events=escalation_events, budget=budget, maximum_conflicts=maximum_conflicts,
        maximum_seconds=maximum_seconds, cancel_check=cancel_check)


def _native_observed_bounds_semantics(value, *, maximum_seconds):
    """Compare proof content, preserving stored bytes and validating timed receipts.

    Reference hashes must follow their normalized owners; otherwise an execution
    allowance leaks back into semantic comparison through a cut or refinement.
    """
    if (type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
            or maximum_seconds <= 0):
        raise ValueError("observed bounds comparison requires a finite positive allowance")
    timed = {"native_post_auction_policy_bounds_v1", "native_symbolic_round_policy_bounds_v2",
        "native_continuation_depth_query_v1", "native_counterexample_prefix_bound_v1",
        "native_counterexample_prefix_threshold_v1", "native_bid_policy_bounds_v1",
        "native_counterexample_prefix_minimax_v1",
        "native_bid_branch_preparation_v1", "native_incumbent_threshold_floor_v1"}
    direct_time = {"native_continuation_depth_query_v1", "native_bid_policy_bounds_v1"}
    owners, memo, active = {}, {}, set()

    def collect(document):
        if isinstance(document, Mapping):
            digest = document.get("sha256")
            if isinstance(digest, str) and digest == canonical_sha256(
                    {k: v for k, v in document.items() if k != "sha256"}):
                owners[digest] = document
            for item in document.values():
                collect(item)
        elif isinstance(document, (list, tuple)):
            for item in document:
                collect(item)

    def normalize(document):
        if isinstance(document, str) and document in owners:
            return normalize_owner(document)["sha256"]
        if isinstance(document, (list, tuple)):
            return [normalize(item) for item in document]
        if not isinstance(document, Mapping):
            return document
        digest = document.get("sha256")
        if isinstance(digest, str) and digest in owners and document == owners[digest]:
            return normalize_owner(digest)
        return normalize_mapping(document, owned=False)

    def normalize_mapping(document, *, owned):
        kind = document.get("kind")
        if kind in timed:
            if not owned:
                raise ValueError("observed bounds nested receipt checksum mismatch")
            container = document if kind in direct_time else document.get("limits")
            field = "maximumSeconds" if kind in direct_time else "seconds"
            if (not isinstance(container, Mapping) or type(container.get(field)) not in {int, float}
                    or not math.isfinite(container[field]) or not 0 < container[field] <= maximum_seconds):
                raise ValueError("observed bounds runtime allowance mismatch")
        result = {normalize(key): (item if key == "sha256" else normalize(item))
            for key, item in document.items()}
        if kind in timed:
            if kind in direct_time:
                result.pop("maximumSeconds")
            else:
                result["limits"].pop("seconds")
        if owned:
            result["sha256"] = canonical_sha256({k: v for k, v in result.items() if k != "sha256"})
        return result

    def normalize_owner(digest):
        if digest in memo:
            return memo[digest]
        if digest in active:
            raise ValueError("observed bounds cyclic receipt references")
        active.add(digest)
        result = normalize_mapping(owners[digest], owned=True)
        active.remove(digest)
        memo[digest] = result
        return result

    collect(value)
    return normalize(value)


def _reverify_native_observed_post_auction_bounds_v1(value, *, rules, source_prefix, normalized,
        selected_root_action, opportunity_order, escalation_events, budget,
        maximum_conflicts, maximum_seconds, cancel_check=None):
    """Freshly rebuild internal observer bounds; this is not production admission."""
    import time
    from .ortools_exact_state import OrToolsExactStateNormalizerV1, OrToolsNormalizedExactStateV1
    from .ortools_information_set_compiler import InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1

    if (not isinstance(value, Mapping) or not isinstance(rules, RuleBook)
            or not isinstance(source_prefix, ExactTracePrefixV1)
            or not isinstance(normalized, OrToolsNormalizedExactStateV1)
            or not isinstance(budget, ScenarioProofBudgetV1)
            or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
            or not 0 < maximum_seconds <= min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds)
            or cancel_check is not None and not callable(cancel_check)):
        raise ValueError("observed bounds verification requires owned context and a finite frozen budget")
    normalized.require_integrity()
    if (normalized.decision_kind not in {"play", "project", "escalation"} or value.get("visibility") != "actor_information_set"
            or value.get("kind") != "native_post_auction_policy_bounds_v1"
            or value.get("status") != "scoped_bound_converged" or value.get("strategicCoverage") is not False
            or value.get("selectedRootAction") != selected_root_action
            or selected_root_action is not None and selected_root_action not in {row["code"] for row in normalized.legal_actions}
            or not isinstance(value.get("strategy"), Mapping)):
        raise ValueError("observed bounds verification scope or selected action mismatch")
    # These native query caps are shared with the existing post-auction lane.
    validate_native_constant_escalation_budget_v1(value, budget, maximum_conflicts=maximum_conflicts)
    preparation = value.get("preparation")
    if (not isinstance(preparation, Mapping) or "ownershipCompaction" in preparation
            and preparation["ownershipCompaction"] != "public_fixed_v1"):
        raise ValueError("observed ownership compaction identity mismatch")
    limits = value["limits"]
    deadline = time.monotonic() + maximum_seconds
    cpu_deadline = time.process_time() + maximum_seconds

    def stopped():
        return bool(cancel_check and cancel_check()) or time.monotonic() >= deadline or time.process_time() >= cpu_deadline

    def remaining():
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage="observed_bounds_verification")
        seconds = min(deadline - time.monotonic(), cpu_deadline - time.process_time())
        if seconds <= 0:
            raise InformationSetCompilationStopped("maximum_seconds", stage="observed_bounds_verification")
        return seconds

    def semantic(document):
        return _native_observed_bounds_semantics(document, maximum_seconds=limits["seconds"])

    recorded = semantic(value)
    rebuilt = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(source_prefix,
        decision_point={"kind": normalized.decision_kind,
            **({"selector": {"owner": normalized.next_actor}} if normalized.decision_kind == "project" else
               {"selector": {"actingSeat": normalized.next_actor}} if normalized.decision_kind == "escalation" else {})},
        maximum_seconds=remaining(), cancel_check=stopped)
    if canonical_sha256(rebuilt.to_dict()) != canonical_sha256(normalized.to_dict()):
        raise ValueError("observed bounds source differs from fresh native normalization")
    remaining()
    replay = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._solve_native_post_auction_policy_bounds(
        rebuilt, source_prefix, Seat.parse(rebuilt.next_actor).team, selected_root_action=selected_root_action,
        compact_public_ownership=preparation.get("ownershipCompaction") == "public_fixed_v1",
        reuse_source_observation=value.get("preparation", {}).get("rootObservationSource") == "source_observation_projection",
        observed_play_bounds_version=value.get("preparation", {}).get("observedPlayBoundsVersion", 1),
        path_floor_version=value.get("preparation", {}).get("pathFloorVersion", 1),
        visibility="actor_information_set", opportunity_order=opportunity_order, escalation_events=escalation_events,
        maximum_iterations=limits["iterations"], maximum_queries=limits["queries"],
        maximum_observations=limits["observations"], maximum_model_bytes=limits["modelBytes"],
        maximum_conflicts=limits["conflictsPerSolve"], maximum_depth=limits["depth"],
        maximum_seconds=limits["seconds"], cancel_check=stopped)
    remaining()
    if canonical_sha256(semantic(replay)) != canonical_sha256(recorded):
        raise ValueError("observed bounds differ from fresh native reconstruction")


def native_observed_best_strategy_action_v1(bounds):
    """Locate a proved strategy's root choice; fresh verification is still required."""
    if (not isinstance(bounds, Mapping) or bounds.get("status") != "scoped_bound_converged"
            or bounds.get("selectedRootAction") is not None):
        return None
    strategy, preparation = bounds.get("strategy"), bounds.get("preparation")
    if not isinstance(strategy, Mapping) or not isinstance(preparation, Mapping):
        return None
    policy = strategy.get("policy")
    if not isinstance(policy, Mapping):
        return None
    root = preparation.get("rootObservationHash")
    action = policy.get(root)
    floor = strategy.get("pathFloor")
    if type(action) is str and (strategy.get("finalBranchProofs") or (
            isinstance(floor, Mapping) and floor.get("value") == strategy.get("upperBound")
            and isinstance(floor.get("policy"), Mapping) and floor["policy"].get(root) == action)):
        return action
    return None


def compose_native_observed_play_optimality_v1(selected_bounds, best_bounds, *, rules,
        source_prefix, normalized, selected_root_action, opportunity_order, escalation_events,
        budget, maximum_conflicts, maximum_seconds, cancel_check=None, decision_kind="play", receipt_version=1):
    """Freshly verify external premises before composing their optimality claim."""
    return _compose_native_observed_play_optimality_v1(selected_bounds, best_bounds, rules=rules,
        source_prefix=source_prefix, normalized=normalized, selected_root_action=selected_root_action,
        opportunity_order=opportunity_order, escalation_events=escalation_events, budget=budget,
        maximum_conflicts=maximum_conflicts, maximum_seconds=maximum_seconds, cancel_check=cancel_check,
        decision_kind=decision_kind, receipt_version=receipt_version)


def _compose_native_observed_play_optimality_v1(selected_bounds, best_bounds, *, rules,
        source_prefix, normalized, selected_root_action, opportunity_order, escalation_events,
        budget, maximum_conflicts, maximum_seconds, cancel_check=None, decision_kind="play",
        receipt_version=1, verified_bounds=frozenset()):
    """Compare verified values, reusing an unrestricted proof's certified choice."""
    import time
    from ortools.sat.python import cp_model
    from .ortools_information_set_compiler import InformationSetCompilationStopped
    from .ortools_information_set_proof import build_native_bid_dominance_model_v1, _solve_cp_sat_with_cancel_v1

    if (type(receipt_version) is not int or receipt_version not in {1, 2}
            or verified_bounds and receipt_version != 2
            or decision_kind not in {"play", "project", "escalation"}
            or not isinstance(budget, ScenarioProofBudgetV1) or budget.native_observed is None
            or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
            or not 0 < maximum_seconds <= min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds)
            or selected_root_action is None):
        raise ValueError("observed optimality requires a selected action and finite frozen limits")
    deadline = time.monotonic() + maximum_seconds
    cpu_deadline = time.process_time() + maximum_seconds
    reuse_best = (isinstance(selected_bounds, Mapping) and isinstance(best_bounds, Mapping)
        and selected_bounds.get("selectedRootAction") is None
        and (selected_bounds is best_bounds or canonical_sha256(selected_bounds) == canonical_sha256(best_bounds)))
    if reuse_best and (native_observed_best_strategy_action_v1(best_bounds) != selected_root_action
            or selected_root_action not in {row["code"] for row in normalized.legal_actions}):
        raise ValueError("observed best strategy does not certify the selected root action")
    premises = ((best_bounds, None),) if reuse_best else ((selected_bounds, selected_root_action), (best_bounds, None))
    counts = [row.get("queriesUsed") if isinstance(row, Mapping) else None for row, _ in premises]
    allowances = [row.get("limits", {}).get("queries") if isinstance(row, Mapping)
        and isinstance(row.get("limits"), Mapping) else None for row, _ in premises]
    if (any(type(count) is not int or count < 1 for count in counts)
            or any(type(limit) is not int or limit < 1 for limit in allowances)
            or any(count > limit for count, limit in zip(counts, allowances))
            or sum(counts) + sum(allowances) + 1 > budget.native_observed.maximum_queries):
        raise ValueError("observed optimality exceeds combined construction and verification query budget")
    def stopped():
        return bool(cancel_check and cancel_check()) or time.monotonic() >= deadline or time.process_time() >= cpu_deadline

    def remaining():
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage="observed_play_optimality")
        seconds = min(deadline - time.monotonic(), cpu_deadline - time.process_time())
        if seconds <= 0:
            raise InformationSetCompilationStopped("maximum_seconds", stage="observed_play_optimality")
        return seconds

    for receipt, action in premises:
        remaining()
        if receipt.get("sha256") in verified_bounds:
            normalized.require_integrity()
            expected = {"kind": "native_post_auction_policy_bounds_v1",
                "rulesHash": rules.sha256, "sourcePrefixHash": source_prefix.sha256,
                "prefixHash": normalized.prefix_hash, "stateHash": normalized.state_hash,
                "exactStateId": normalized.exact_state_id, "decisionKind": decision_kind,
                "visibility": "actor_information_set", "targetTeam": Seat.parse(normalized.next_actor).team.name.lower(),
                "selectedRootAction": action, "utility": "awarded_point_margin",
                "status": "scoped_bound_converged", "strategicCoverage": False}
            preparation = receipt.get("preparation", {})
            contexts = preparation.get("partitions", ()) if decision_kind == "escalation" else (preparation,)
            if (receipt.get("sha256") != canonical_sha256({key: value for key, value in receipt.items() if key != "sha256"})
                    or any(receipt.get(key) != value for key, value in expected.items())
                    or preparation.get("rootLegalActions") != sorted(row["code"] for row in normalized.legal_actions)
                    or not contexts or any(
                        context.get("opportunityOrder") != [Seat.parse(seat).name.lower() for seat in opportunity_order]
                        or context.get("escalationEvents") != [dict(row) for row in escalation_events]
                        for context in contexts)):
                raise ValueError("observed owned proof binding mismatch")
            validate_native_constant_escalation_budget_v1(receipt, budget, maximum_conflicts=maximum_conflicts)
            continue
        verify = (reverify_native_observed_play_bounds_v1 if decision_kind == "play"
                  else reverify_native_observed_project_bounds_v1 if decision_kind == "project"
                  else _reverify_native_observed_post_auction_bounds_v1)
        verify(receipt, rules=rules, source_prefix=source_prefix,
            normalized=normalized, selected_root_action=action, opportunity_order=opportunity_order,
            escalation_events=escalation_events, budget=budget, maximum_conflicts=maximum_conflicts,
            maximum_seconds=remaining(), cancel_check=stopped)
    actions = sorted(row["code"] for row in normalized.legal_actions)
    optimum = best_bounds["strategy"]["upperBound"]
    alternatives = [{"action": action, "upperBound": optimum,
        "sourceProofHash": best_bounds["sha256"]} for action in actions if action != selected_root_action]
    model, selected, claim, best, regret, bindings = build_native_bid_dominance_model_v1(
        selected_bounds["strategy"]["lowerBound"], alternatives)
    model.add(selected <= optimum)
    model.add(best == optimum)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = remaining()
    solver.parameters.max_number_of_conflicts = maximum_conflicts
    status = _solve_cp_sat_with_cancel_v1(solver, model, stopped)
    remaining()
    if status in {cp_model.MODEL_INVALID, cp_model.INFEASIBLE}:
        raise ValueError("observed optimality native proof values disagree")
    if status != cp_model.OPTIMAL:
        raise InformationSetCompilationStopped("qualification_incomplete", stage="observed_play_optimality")
    core = {"schemaVersion": receipt_version, "kind": f"native_observed_{decision_kind}_optimality_v1",
        "status": "selected_action_optimal" if solver.boolean_value(claim) else "selected_action_suboptimal",
        "strategicCoverage": False, "visibility": "actor_information_set", "scope": f"selected_{decision_kind}_current_deal",
        "utility": "awarded_point_margin", "targetTeam": Seat.parse(normalized.next_actor).team.name.lower(),
        "rulesHash": rules.sha256, "sourcePrefixHash": source_prefix.sha256,
        "exactStateId": normalized.exact_state_id, "budgetProfile": budget.profile_id, "budgetHash": budget.sha256,
        "selectedAction": selected_root_action, "actualValue": int(solver.value(selected)),
        "bestValue": int(solver.value(best)), "regret": int(solver.value(regret)),
        "actualActionOptimal": bool(solver.boolean_value(claim)),
        "selectedProofHash": selected_bounds["sha256"], "bestProofHash": best_bounds["sha256"],
        "selectedProof": copy.deepcopy(selected_bounds), "bestProof": copy.deepcopy(best_bounds),
        "modelProto": str(model.proto), "alternatives": bindings, "solverStatus": "optimal",
        "queriesUsed": (2 if receipt_version == 1 else 1) * sum(counts) + 1,
        **({"queryCountScope": "certificate_construction_only"} if receipt_version == 2 else {}),
        **({"selectionSource": "verified_best_strategy"} if reuse_best else {})}
    return {**core, "sha256": canonical_sha256(core)}


def build_native_observed_play_quality_artifact_v1(selected_bounds, best_bounds, *, rules,
        source_prefix, normalized, request, opportunity_order, escalation_events,
        budget, maximum_conflicts, maximum_seconds, cancel_check=None, decision_kind="play", receipt_version=1):
    """Freshly reconstruct external bounds; callers cannot supply verification tokens."""
    return _build_native_observed_play_quality_artifact_v1(selected_bounds, best_bounds, rules=rules,
        source_prefix=source_prefix, normalized=normalized, request=request, opportunity_order=opportunity_order,
        escalation_events=escalation_events, budget=budget, maximum_conflicts=maximum_conflicts,
        maximum_seconds=maximum_seconds, cancel_check=cancel_check, decision_kind=decision_kind,
        receipt_version=receipt_version)


def _build_native_observed_play_quality_artifact_v1(selected_bounds, best_bounds, *, rules,
        source_prefix, normalized, request, opportunity_order, escalation_events,
        budget, maximum_conflicts, maximum_seconds, cancel_check=None, decision_kind="play",
        receipt_version=2, verified_bounds=frozenset()):
    """Bind freshly reconstructed post-auction optimality to its exact request."""
    from .ortools_exact_state import OrToolsNormalizedExactStateV1
    selected_action = selected_bounds.get("selectedRootAction") if isinstance(selected_bounds, Mapping) else None
    if (selected_action is None and isinstance(selected_bounds, Mapping) and isinstance(best_bounds, Mapping)
            and (selected_bounds is best_bounds or canonical_sha256(selected_bounds) == canonical_sha256(best_bounds))):
        selected_action = native_observed_best_strategy_action_v1(best_bounds)
    if (not isinstance(request, ScenarioProofRequestV1) or not isinstance(budget, ScenarioProofBudgetV1)
            or not isinstance(normalized, OrToolsNormalizedExactStateV1)
            or decision_kind not in {"play", "project", "escalation"} or normalized.decision_kind != decision_kind
            or not isinstance(selected_bounds, Mapping)
            or request.mode != "exact" or request.visibility != "actor_information_set"
            or request.utility != "awarded_point_margin" or request.horizon is not None
            or request.line_universe is not None or request.budget_profile != budget.profile_id
            or request.target_team != Seat.parse(normalized.next_actor).team.name.lower()
            or request.selected_action not in {None, selected_action}):
        raise ValueError("observed play quality requires an owned exact selected-action request")
    graph = _compose_native_observed_play_optimality_v1(selected_bounds, best_bounds, rules=rules,
        source_prefix=source_prefix, normalized=normalized,
        selected_root_action=selected_action, opportunity_order=opportunity_order,
        escalation_events=escalation_events, budget=budget, maximum_conflicts=maximum_conflicts,
        maximum_seconds=maximum_seconds, cancel_check=cancel_check, decision_kind=decision_kind,
        receipt_version=receipt_version, verified_bounds=verified_bounds)
    return _project_native_observed_quality_artifact_v1(graph, rules=rules,
        source_prefix=source_prefix, normalized=normalized, request=request,
        opportunity_order=opportunity_order, escalation_events=escalation_events,
        budget=budget, decision_kind=decision_kind)


def _project_native_bounded_play_quality_artifact_v1(execution, *, rules, source_prefix, normalized,
        request, budget):
    """Project owned compiler evidence; external readers must reconstruct it first."""
    from .ortools_exact_state import _effective_prefix_hash
    normalized.require_integrity()
    ScenarioProofRequestV1.build(request.to_dict())
    if (request.mode != "bounded" or request.visibility != "actor_information_set"
            or request.horizon is None or request.claim not in {"outcome_reachable", "winning_root_action_count"}
            or request.line_universe is not None or normalized.decision_kind != "play"
            or normalized.rules_hash != rules.sha256
            or normalized.prefix_hash != _effective_prefix_hash(source_prefix)
            or request.budget_profile != budget.profile_id
            or execution.get("kind") != "native_observed_play_selection_execution_v1"
            or execution.get("sourceStateHash") != normalized.state_hash
            or execution.get("requestHash") != request.sha256
            or execution.get("budgetHash") != budget.sha256
            or execution.get("sha256") != canonical_sha256({
                key: value for key, value in execution.items() if key != "sha256"})):
        raise ValueError("bounded selected play quality ownership mismatch")
    quality = execution["actionQuality"]
    selected = quality["actualAction"]
    actions = sorted(row["code"] for row in normalized.legal_actions)
    if (selected not in actions or request.selected_action not in {None, selected}
            or request.selected_action is None and quality["actualActionOptimal"] is not True):
        raise ValueError("bounded selected play quality action mismatch")
    model = {"schemaVersion": 1, "kind": "native_bounded_play_quality_model_v1",
        "rootActorSeat": normalized.next_actor,
        "qualificationModelHash": quality["qualificationModelHash"], "actionQuality": quality}
    allocation = {"schemaVersion": 1, "kind": "native_bounded_play_quality_context_v1",
        "sourcePrefixHash": source_prefix.sha256, "rulesHash": rules.sha256,
        "exactStateId": normalized.exact_state_id, "stateHash": normalized.state_hash,
        "budgetProfile": budget.profile_id, "budgetHash": budget.sha256}
    allocation["sha256"] = canonical_sha256(allocation)
    proof = {"schemaVersion": 1, "completionStatus": "bounded_complete",
        "completionScope": "selected_action_supplied_worlds_bounded",
        "claimSatisfied": None, "selectedAction": selected, "winningRootActionCount": None,
        "winningRootActionLowerBound": 0, "winningRootActionUpperBound": len(actions),
        "rootActions": [{"action": action, "value": quality["actualValue"] if action == selected else None,
            "satisfies": None} for action in actions],
        "identities": {"graphHash": execution["sha256"], "modelHash": canonical_sha256(model),
            "proofRequestHash": request.sha256}}
    proof["sha256"] = canonical_sha256(proof)
    return {"request": request.to_dict(), "proof": proof, "proofModel": model,
        "graph": execution, "allocationModel": allocation}


def _project_native_observed_quality_artifact_v1(graph, *, rules, source_prefix, normalized,
        request, opportunity_order, escalation_events, budget, decision_kind):
    """Bind an already verified graph; never an external evidence entry point."""
    actions = sorted(row["code"] for row in normalized.legal_actions)
    selected, optimal = graph["selectedAction"], graph["actualActionOptimal"]
    if request.selected_action is None and not optimal:
        raise ValueError("observed play generation cannot select a proved suboptimal action")
    quality = {"schemaVersion": 1, "authority": "cp_sat", "scope": "compiled_actor_information_set",
        "rootActorRelation": "target", "actualAction": selected,
        "actualActionOptimal": optimal, "actualValue": graph["actualValue"],
        "bestValue": graph["bestValue"], "regret": graph["regret"],
        "optimalActions": [selected] if optimal else [], "optimalActionsComplete": False,
        "optimalActionCount": None, "optimalActionCountLowerBound": 1,
        "optimalActionCountUpperBound": len(actions), "qualificationModelHash": graph["sha256"],
        "analysisScope": {"mode": "exact", "coverage": graph["scope"],
            "utility": request.utility, "exactStateId": normalized.exact_state_id}}
    quality["sha256"] = canonical_sha256(quality)
    model = {"schemaVersion": 1, "kind": f"native_observed_{decision_kind}_quality_model_v1",
        "rootActorSeat": normalized.next_actor, "qualificationModelHash": graph["sha256"],
        "actionQuality": quality}
    allocation = {"schemaVersion": 1, "kind": f"native_observed_{decision_kind}_quality_context_v1",
        "sourcePrefixHash": source_prefix.sha256, "rulesHash": rules.sha256,
        "exactStateId": normalized.exact_state_id, "stateHash": normalized.state_hash,
        "sourcePrefix": source_prefix.to_dict(), "normalizedState": normalized.to_dict(),
        "normalizationReceipt": normalized.normalization_receipt.to_dict(),
        "budgetProfile": budget.profile_id, "budgetHash": budget.sha256,
        "opportunityOrder": [Seat.parse(seat).name.lower() for seat in opportunity_order],
        "escalationEvents": [dict(row) for row in escalation_events]}
    allocation["sha256"] = canonical_sha256(allocation)
    proof = {"schemaVersion": 1, "completionStatus": "exact_complete", "completionScope": graph["scope"],
        "claimSatisfied": None, "selectedAction": selected, "winningRootActionCount": None,
        "winningRootActionLowerBound": 0, "winningRootActionUpperBound": len(actions),
        "rootActions": [{"action": action, "value": graph["actualValue"] if action == selected else None,
            "satisfies": None} for action in actions],
        "identities": {"graphHash": graph["sha256"], "modelHash": canonical_sha256(model),
            "proofRequestHash": request.sha256}}
    proof["sha256"] = canonical_sha256(proof)
    return {"request": request.to_dict(), "proof": proof, "proofModel": model,
        "graph": graph, "allocationModel": allocation}


def validate_native_observed_play_quality_artifact_v1(artifact, **context):
    """Rebuild the envelope from caller-owned context, never its embedded authority."""
    graph = artifact.get("graph") if isinstance(artifact, Mapping) else None
    if not isinstance(graph, Mapping) or not all(isinstance(graph.get(key), Mapping)
            for key in ("selectedProof", "bestProof")):
        raise ValueError("observed play quality artifact is malformed")
    expected = build_native_observed_play_quality_artifact_v1(graph["selectedProof"], graph["bestProof"],
        receipt_version=graph.get("schemaVersion", 1), **context)
    if canonical_sha256(artifact) != canonical_sha256(expected):
        raise ValueError("observed play quality artifact binding mismatch")


def validate_native_post_auction_bounds_v1(value: Mapping[str, Any], *, rules: RuleBook,
        source_prefix: ExactTracePrefixV1, normalized, opportunity_order, escalation_events) -> None:
    """Reconstruct the native continuation owner; never widen its proof scope.

    This checks retained solver evidence, not a fresh solve or whole-round
    qualification. A fixed-action bound alone does not establish optimality.
    """
    from .ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from .ortools_exact_state import OrToolsNormalizedExactStateV1, _effective_prefix_hash

    def hashed(document):
        return (isinstance(document, Mapping) and document.get("sha256") == canonical_sha256(
            {key: item for key, item in document.items() if key != "sha256"}))

    if (not isinstance(rules, RuleBook) or not isinstance(source_prefix, ExactTracePrefixV1)
            or not isinstance(normalized, OrToolsNormalizedExactStateV1)):
        raise TypeError("post-auction bounds require typed frozen context")
    normalized.require_integrity()
    actions = sorted(row["code"] for row in normalized.legal_actions)
    if (normalized.decision_kind not in {"project", "play"} or normalized.next_actor is None
            or normalized.rules_hash != rules.sha256
            or normalized.prefix_hash != _effective_prefix_hash(source_prefix)):
        raise ValueError("post-auction normalized context mismatch")
    expected = {"schemaVersion": 1, "kind": "native_post_auction_policy_bounds_v1",
        "status": "scoped_bound_converged", "strategicCoverage": False, "visibility": "complete_state",
        "rulesHash": rules.sha256, "sourcePrefixHash": source_prefix.sha256,
        "prefixHash": normalized.prefix_hash, "exactStateId": normalized.exact_state_id,
        "stateHash": normalized.state_hash, "decisionKind": normalized.decision_kind,
        "targetTeam": Seat.parse(normalized.next_actor).team.name.lower(),
        "utility": "awarded_point_margin", "continuationBoundary": "current_deal_completion"}
    if (not hashed(value) or type(value.get("schemaVersion")) is not int or value.get("strategicCoverage") is not False
            or any(value.get(key) != item for key, item in expected.items())
            or value.get("selectedRootAction") is not None and value["selectedRootAction"] not in actions):
        raise ValueError("post-auction bounds ownership or completion mismatch")
    preparation, strategy, limits = value.get("preparation"), value.get("strategy"), value.get("limits")
    if (not isinstance(preparation, Mapping) or not hashed(strategy) or not isinstance(limits, Mapping)
            or any(type(limits.get(key)) is not int or limits[key] < 1 for key in
                ("iterations", "queries", "observations", "modelBytes", "conflictsPerSolve", "depth"))
            or type(limits.get("seconds")) not in {int, float} or not math.isfinite(limits["seconds"])
            or limits["seconds"] <= 0 or type(value.get("queriesUsed")) is not int
            or not 0 < value["queriesUsed"] <= limits["queries"]
            or type(strategy.get("queriesUsed")) is not int
            or value["queriesUsed"] != 2 + strategy["queriesUsed"]
            or preparation.get("opportunityOrder") != [Seat.parse(seat).name.lower() for seat in opportunity_order]
            or preparation.get("escalationEvents") != [dict(row) for row in escalation_events]
            or preparation.get("rootLegalActions") != actions):
        raise ValueError("post-auction preparation or resource evidence mismatch")
    depth, witness, scope = preparation.get("depthQuery"), preparation.get("rootQuery"), strategy.get("scope")
    if (not hashed(depth) or not hashed(witness) or not isinstance(scope, Mapping)
            or depth.get("status") != "depth_bound_proven" or depth.get("solverStatus") != "infeasible"
            or depth.get("maximumDepth") != limits["depth"]
            or depth.get("unit") != "remaining_decision_actions"
            or depth.get("includes") != ["auction", "escalation", "project", "play"]
            or witness.get("status") != "witness_found" or witness.get("solverStatus") not in {"optimal", "feasible"}
            or strategy.get("status") != "scoped_bound_converged"
            or type(strategy.get("lowerBound")) is not int or type(strategy.get("upperBound")) is not int
            or strategy["lowerBound"] != strategy["upperBound"]
            or strategy.get("scopeHash") != canonical_sha256(scope)):
        raise ValueError("post-auction native bounds are incomplete or inconsistent")
    candidate = witness.get("candidate")
    if not isinstance(candidate, Mapping) or candidate.get("deckOrder") != list(source_prefix.deck_order):
        raise ValueError("post-auction root witness deal mismatch")
    rows = candidate.get(normalized.decision_kind + "Observations")
    index = len(source_prefix.project_actions if normalized.decision_kind == "project" else source_prefix.play_actions)
    if (not isinstance(rows, list) or len(rows) <= index or not isinstance(rows[index], Mapping)
            or rows[index].get("actor") != normalized.next_actor
            or sorted(rows[index].get("legalActions", [])) != actions
            or rows[index].get("authoredPrefix", False)
            or rows[index].get("observationHash") != preparation.get("rootObservationHash")):
        raise ValueError("post-auction root observation mismatch")
    model = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._native_post_auction_root_model(
        normalized, source_prefix, opportunity_order=opportunity_order, escalation_events=escalation_events)
    model_hash = model._model_hash(search_branching="automatic")
    branch_scope, points = model._native_round_query_context()
    maximum = max(max(map(int, point.proto.domain)) for point in points)
    information = rows[index].get("information")
    if not isinstance(information, Mapping):
        raise ValueError("post-auction observation information missing")
    information, _ = (model._normalize_project_information(information) if normalized.decision_kind == "project"
        else model._normalize_play_information(information))
    if preparation["rootObservationHash"] != model._round_observation_hash(normalized.decision_kind, information):
        raise ValueError("post-auction observation identity mismatch")
    expected_scope = {"kind": "native_symbolic_round_policy_scope_v1", "rulesHash": rules.sha256,
        "targetTeam": expected["targetTeam"], "coverage": "supplied_deal_branches_only",
        "utility": "awarded_point_margin", "utilityBounds": scope.get("utilityBounds"),
        "cooperativeCeiling": True,
        "branches": [{"modelHash": model_hash, "scope": dict(branch_scope)}]}
    selected = value.get("selectedRootAction")
    if selected is not None:
        expected_scope["fixedRoot"] = {"observationHash": preparation["rootObservationHash"],
            "legalActions": actions, "selectedAction": selected}
    if (scope != expected_scope or preparation.get("sourceModelHash") != model_hash
            or preparation.get("modelBytes") != len(str(model.model.proto).encode("utf-8"))
            or preparation["modelBytes"] > limits["modelBytes"]
            or depth.get("sourceModelHash") != model_hash or depth.get("scopeHash") != branch_scope["sha256"]
            or witness.get("baseModelHash") != model_hash or witness.get("scope") != branch_scope
            or witness.get("targetTeam") != expected["targetTeam"] or witness.get("threshold") != maximum + 1):
        raise ValueError("post-auction reconstructed model coverage mismatch")
    queries, masters, policy = strategy.get("queries"), strategy.get("masters"), strategy.get("policy")
    if (not isinstance(queries, list) or len(queries) < 2 or not isinstance(masters, list) or not masters
            or not isinstance(policy, Mapping) or len(masters) > limits["iterations"]
            or any(not isinstance(row, Mapping) or row.get("status") != "optimal" for row in masters)
            or masters[-1].get("upperBound") != strategy["upperBound"]
            or selected is not None and policy.get(preparation["rootObservationHash"]) != selected):
        raise ValueError("post-auction strategy completion mismatch")
    if (type(strategy.get("bindingQueriesUsed")) is not int or strategy["bindingQueriesUsed"] < 0
            or strategy["queriesUsed"] != len(queries) + len(masters) + strategy["bindingQueriesUsed"]
            or not isinstance(strategy.get("domains"), Mapping) or len(strategy["domains"]) > limits["observations"]):
        raise ValueError("post-auction strategy resource accounting mismatch")
    query_hashes = set()
    prefix_bounds = {}
    for row in queries:
        if (not isinstance(row, Mapping) or row.get("branchModelHash") != model_hash
                or not hashed(row.get("receipt"))):
            raise ValueError("post-auction strategy query ownership mismatch")
        receipt = row["receipt"]
        if receipt.get("kind") in {"native_counterexample_prefix_bound_v1", "native_counterexample_prefix_threshold_v1"}:
            if (receipt.get("scope") != "fixed_world_cooperative_upper_bound"
                    or receipt.get("rulesHash") != rules.sha256 or receipt.get("targetTeam") != expected["targetTeam"]
                    or receipt.get("sourceQueryHash") not in query_hashes
                    or type(receipt.get("prefixPlayCount")) is not int
                    or not max(4, len(source_prefix.play_actions)) <= receipt["prefixPlayCount"] < 32):
                raise ValueError("post-auction prefix bound ownership mismatch")
            if receipt["kind"] == "native_counterexample_prefix_threshold_v1":
                threshold = receipt.get("requiredThreshold")
                if (type(threshold) is not int
                        or receipt.get("complementThreshold") != 1 - threshold
                        or receipt.get("complementTargetTeam") != Seat.parse(normalized.next_actor).team.other.name.lower()
                        or receipt.get("queryMode") != "feasibility_v1"
                        or receipt.get("status") == "upper_bound_complete" and (
                            receipt.get("solverStatus") != "infeasible"
                            or type(receipt.get("upperBound")) is not int
                            or receipt["upperBound"] != threshold - 1)):
                    raise ValueError("post-auction prefix threshold evidence mismatch")
            prefix_bounds[receipt["sha256"]] = receipt
        elif receipt.get("scope") != branch_scope:
            raise ValueError("post-auction strategy query scope mismatch")
        query_hashes.add(receipt["sha256"])
    for cut in strategy.get("cuts", []):
        if cut.get("prefixBoundQueryHash") is not None:
            bound = prefix_bounds.get(cut["prefixBoundQueryHash"])
            if (bound is None or bound.get("status") != "upper_bound_complete"
                    or type(bound.get("upperBound")) is not int or cut.get("utility") != bound["upperBound"]
                    or cut.get("targetActions") != bound.get("targetActions")):
                raise ValueError("post-auction shortened cut lacks a completed bound")
    ceiling, final = queries[0]["receipt"], queries[-1]["receipt"]
    opposite = Seat.parse(normalized.next_actor).team.other.name.lower()
    if (queries[0].get("purpose") != "cooperative_upper_bound"
            or ceiling.get("baseModelHash") != model_hash or ceiling.get("targetTeam") != opposite
            or ceiling.get("status") != "witness_found" or ceiling.get("solverStatus") != "optimal"
            or not isinstance(ceiling.get("candidate"), Mapping)
            or type(ceiling["candidate"].get("utility")) is not int
            or scope["utilityBounds"] != [-maximum, -ceiling["candidate"]["utility"]]
            or not -maximum <= strategy["upperBound"] <= scope["utilityBounds"][1]
            or final.get("targetTeam") != expected["targetTeam"] or final.get("threshold") != strategy["upperBound"]
            or final.get("status") != "no_matching_outcome" or final.get("solverStatus") != "infeasible"):
        raise ValueError("post-auction strategy bounds lack terminal solver evidence")
    for kind in ("auction", "escalation", "project", "play"):
        bindings = final.get(kind + "PolicyBindings")
        if (not isinstance(bindings, list) or any(not hashed(binding)
                or binding.get("kind") != "native_" + kind + "_policy_binding_v1"
                or binding.get("rulesHash") != rules.sha256 or binding.get("status") != "bound"
                or binding.get("solverStatus") != "INFEASIBLE"
                or binding.get("action") != policy.get(binding.get("observationHash")) for binding in bindings)):
            raise ValueError("post-auction final query differs from the certified policy")


def validate_native_bid_dominance_resume_basis_v1(value, *, rules, source_prefix,
        normalized, selected_action, opponent_policy, opponent_play_policy_id,
        limits):
    """Validate complete retained alternative caps before threshold resumption."""
    expected = {"schemaVersion": 1, "kind": "native_bid_selected_action_dominance_v3",
        "status": "incomplete", "rulesHash": rules.sha256,
        "sourcePrefixHash": source_prefix.sha256, "exactStateId": normalized.exact_state_id,
        "stateHash": normalized.state_hash, "targetTeam": Seat.parse(normalized.next_actor).team.name.lower(),
        "selectedAction": selected_action, "visibility": "complete_state",
        "utility": "awarded_point_margin", "completionScope": "selected_bid_current_deal",
        "legalActions": sorted(row["code"] for row in normalized.legal_actions),
        "opponentStrategy": opponent_policy}
    if (not isinstance(value, Mapping) or any(value.get(key) != item for key, item in expected.items())
            or type(value.get("schemaVersion")) is not int or "incumbentAttempt" in value
            or value.get("sha256") != canonical_sha256({k: v for k, v in value.items() if k != "sha256"})
            or value.get("opponentPlayPolicyId") != opponent_play_policy_id):
        raise ValueError("bid threshold resume basis ownership mismatch")
    prior = value.get("limits")
    used = value.get("queriesUsed")
    alternatives = value.get("alternatives")
    if (not isinstance(prior, Mapping) or type(used) is not int or used < 0
            or type(prior.get("maximumQueries")) is not int
            or type(limits.get("maximumQueries")) is not int
            or not 0 < limits["maximumQueries"] <= prior["maximumQueries"] - used
            or any(limits.get(key) != prior.get(key) for key in (
                "maximumIterations", "maximumObservations", "maximumBranchesPerPreparation",
                "maximumModelBytesPerPreparation", "maximumConflictsPerSolve", "maximumDepth"))
            or not isinstance(alternatives, list) or not alternatives
            or any(not isinstance(row, Mapping) for row in alternatives)
            or [row.get("action") for row in alternatives] != [a for a in expected["legalActions"] if a != selected_action]):
        raise ValueError("bid threshold resume coverage or budget mismatch")
    total = 0
    for row in alternatives:
        preparation = row.get("preparation")
        upper = validate_native_bid_world_upper_v1(preparation, rules=rules, exact=normalized,
            source_prefix=source_prefix, action=row["action"], opponent_policy=opponent_policy,
            opponent_play_policy_id=opponent_play_policy_id,
            maximum_model_bytes=limits["maximumModelBytesPerPreparation"],
            maximum_depth=limits.get("maximumDepth"), opportunity_order=value.get("escalationOpportunityOrder"))
        retained = [branch for branch in preparation["branches"] if branch["retained"]]
        if (type(row.get("upperBound")) is not int or row["upperBound"] != upper
                or row.get("queries") != [branch["alternativeQuery"] for branch in retained]
                or row.get("sourceModelHashes") != [branch["sourceModelHash"] for branch in retained]):
            raise ValueError("bid threshold resume alternative substitution")
        total += preparation["queriesUsed"]
    if total > used:
        raise ValueError("bid threshold resume undercharges retained alternatives")


def validate_native_bid_dominance_identity_v1(
    value: Mapping[str, Any], *, rules_hash: str, source_prefix_hash: str,
    exact_state_id: str, state_hash: str, target_team: str, selected_action: str,
) -> None:
    """Validate retained certificate ownership, not strategic truth or admission."""
    def hashed(document):
        return (isinstance(document, Mapping) and document.get("sha256") == canonical_sha256(
            {key: item for key, item in document.items() if key != "sha256"}))

    kind = value.get("kind") if isinstance(value, Mapping) else None
    if kind not in {"native_bid_selected_action_dominance_v1", "native_bid_selected_action_dominance_v2",
                    "native_bid_selected_action_dominance_v3", "native_bid_selected_action_dominance_v4"}:
        raise ValueError("bid dominance certificate kind is unsupported")
    expected = {"kind": kind, "schemaVersion": 1,
        "rulesHash": rules_hash, "sourcePrefixHash": source_prefix_hash,
        "exactStateId": exact_state_id, "stateHash": state_hash, "targetTeam": target_team,
        "selectedAction": selected_action, "visibility": "complete_state",
        "utility": "awarded_point_margin", "completionScope": "selected_bid_current_deal"}
    if (not hashed(value) or type(value.get("schemaVersion")) is not int
            or any(value.get(key) != item for key, item in expected.items())):
        raise ValueError("bid dominance certificate ownership mismatch")
    selected = value.get("selectedValueProof")
    qualification = value.get("qualificationModel")
    alternatives = value.get("alternatives")
    if (not hashed(selected) or not hashed(selected.get("preparation")) or not hashed(selected.get("strategy"))
            or not hashed(qualification) or not isinstance(alternatives, list)
            or qualification.get("selectedProofHash") != canonical_sha256(selected)):
        raise ValueError("bid dominance selected proof identity mismatch")
    bindings = qualification.get("alternatives")
    if not isinstance(bindings, list) or len(bindings) != len(alternatives):
        raise ValueError("bid dominance alternative identity coverage mismatch")
    for row, binding in zip(alternatives, bindings):
        if (not isinstance(row, Mapping) or not isinstance(binding, Mapping)
                or binding.get("action") != row.get("action")
                or binding.get("sourceProofHash") != canonical_sha256(row)
                or not hashed(row.get("preparation"))
                or not isinstance(row.get("queries"), list)
                or any(not hashed(query) for query in row["queries"])):
            raise ValueError("bid dominance alternative proof identity mismatch")
    quality = value.get("actionQuality")
    if not isinstance(quality, Mapping):
        raise ValueError("bid dominance quality identity missing")
    quality_core = {key: item for key, item in quality.items() if key != "sha256"}
    if (quality.get("qualificationModelHash") != qualification["sha256"]
            or quality.get("sha256") != canonical_sha256({**quality_core,
                "modelProto": qualification.get("modelProto")})):
        raise ValueError("bid dominance quality identity mismatch")


def _validate_native_final_branch_proofs_v2(strategy: Mapping[str, Any], *, target_team: str,
        required_threshold: int | None = None) -> None:
    """Check retained reference integrity, not replay or strategic admission."""
    def fail():
        raise ValueError("native final branch proof ownership or coverage mismatch")

    def hashed(value):
        return isinstance(value, Mapping) and value.get("sha256") == canonical_sha256(
            {key: item for key, item in value.items() if key != "sha256"})

    scope = strategy.get("scope")
    proofs, refinements, queries = (strategy.get(key) for key in ("finalBranchProofs", "refinements", "queries"))
    policy = strategy.get("policy")
    value = strategy.get("upperBound") if required_threshold is None else required_threshold
    fallback = scope.get("playFallback") if isinstance(scope, Mapping) else None
    representation = scope.get("policyRepresentation") if isinstance(scope, Mapping) else None
    expected_kind = ("native_symbolic_round_policy_bounds_v2" if required_threshold is None
                     else "native_guarded_round_strategy_threshold_v2" if fallback is not None
                          and representation == "native_fallback_plus_deviations_v1"
                     else "native_guarded_round_strategy_threshold_v1" if fallback is not None
                     else "native_symbolic_round_strategy_threshold_v1")
    if (not hashed(strategy) or strategy.get("kind") != expected_kind
            or representation is not None and (fallback is None
                or representation != "native_fallback_plus_deviations_v1")
            or type(strategy.get("schemaVersion")) is not int or strategy["schemaVersion"] != 1
            or strategy.get("strategicCoverage") is not False or not isinstance(scope, Mapping)
            or strategy.get("scopeHash") != canonical_sha256(scope)
            or scope.get("targetTeam") != target_team or not isinstance(scope.get("branches"), list)
            or not isinstance(proofs, list) or not isinstance(refinements, Mapping)
            or not isinstance(queries, list) or not isinstance(policy, Mapping) or type(value) is not int):
        fail()
    if fallback is not None:
        from .ortools_play_tail import native_observer_safe_policy_identity_v1
        if (required_threshold is None or not isinstance(fallback, Mapping)
                or fallback != native_observer_safe_policy_identity_v1(
                    fallback.get("policyId"), Team[target_team.upper()])):
            fail()
        contexts = {}
        for row in queries:
            if not isinstance(row, Mapping) or not isinstance(row.get("receipt"), Mapping):
                fail()
            for observation in (row["receipt"].get("candidate") or {}).get("playObservations", ()):
                if observation.get("authoredPrefix", False) or Seat.parse(observation["actor"]).team.name.lower() != target_team:
                    continue
                key, info = observation["observationHash"], observation["information"]
                if contexts.setdefault(key, info) != info:
                    fail()
        # A warm proposal need not appear in an adverse witness. Its context is
        # policy input, not a claim of visitation; require a fresh owned native
        # legality binding in a final branch, then re-prove its effect on read.
        for proof in proofs:
            retained = refinements.get(proof.get("refinementHash"), {}) if isinstance(proof, Mapping) else {}
            core = retained.get("core", {}) if isinstance(retained, Mapping) else {}
            overrides = core.get("playOverrides", {}) if isinstance(core, Mapping) else {}
            bindings = core.get("bindings", ()) if isinstance(core, Mapping) else ()
            if not isinstance(overrides, Mapping) or not isinstance(bindings, (list, tuple)):
                fail()
            for key, row in overrides.items():
                if (not isinstance(row, Mapping) or set(row) != {"information", "action"}
                        or not any(hashed(binding) and binding.get("kind") == "native_play_policy_binding_v1"
                            and binding.get("status") == "bound" and binding.get("solverStatus") == "INFEASIBLE"
                            and binding.get("observationHash") == key and binding.get("action") == row["action"]
                            for binding in bindings)
                        or contexts.setdefault(key, row["information"]) != row["information"]):
                    fail()
        if strategy.get("playContexts") != contexts:
            fail()
    branches = scope["branches"]
    if (any(not isinstance(row, Mapping) or type(row.get("modelHash")) is not str
                or not isinstance(row.get("scope"), Mapping) for row in branches)
            or any(not isinstance(row, Mapping) for row in proofs)):
        fail()
    models = [row["modelHash"] for row in branches]
    if not models or len(set(models)) != len(models) or [row.get("branchModelHash") for row in proofs] != models:
        fail()
    owners = {row["modelHash"]: row.get("scope") for row in branches}
    lookup = {}
    for wrapper in queries:
        if (not isinstance(wrapper, Mapping) or type(wrapper.get("branchModelHash")) is not str
                or wrapper["branchModelHash"] not in owners or not isinstance(wrapper.get("receipt"), Mapping)):
            fail()
        query = wrapper["receipt"]
        lookup[wrapper["branchModelHash"], canonical_sha256(query)] = query
    configured = scope.get("opponentCeilingPolicy")
    if configured is not None and not isinstance(configured, Mapping):
        fail()
    world_policy_rows = []
    for row in queries:
        if row.get("purpose") != "fixed_world_opponent_upper_bound" or "sourceWorldQueryHash" not in row:
            continue
        policy = row["receipt"].get("policy")
        identity = policy.get("nativePlayPolicy") if isinstance(policy, Mapping) else None
        if not isinstance(identity, Mapping):
            fail()
        if (configured is not None and row["sourceWorldQueryHash"] == scope.get("opponentCeilingWorldQueryHash")
                and identity.get("policyId") == configured.get("native_play_policy_id")):
            world_policy_rows.append(row)
    if (world_policy_rows or "opponentCeilingProjectActions" in scope
            or isinstance(configured, Mapping) and "native_play_policy_id" in configured
            and any(row.get("purpose") == "fixed_world_opponent_upper_bound"
                and "sourceWorldQueryHash" in row for row in queries)):
        from .ortools_play_tail import native_observer_safe_policy_identity_v1

        floors = [row for row in queries if row.get("purpose") in
            {"root_policy_path_floor", "universal_path_floor", "adversarial_world_witness"}
            and row["receipt"].get("sha256") == scope.get("opponentCeilingWorldQueryHash")]
        if len(models) != 1 or len(world_policy_rows) != 1 or len(floors) != 1:
            fail()
        wrapper, floor = world_policy_rows[0], floors[0]["receipt"]
        receipt = wrapper["receipt"]
        configured, policy, outcome = scope.get("opponentCeilingPolicy"), receipt.get("policy"), receipt.get("query")
        opponent = Seat.NORTH.team.other if target_team == "north_south" else Seat.WEST.team.other
        if (not isinstance(configured, Mapping) or set(configured) != {
                "auction_action", "escalation_action", "native_play_policy_id"}
                or not isinstance(policy, Mapping) or not isinstance(outcome, Mapping)):
            fail()
        try:
            identity = native_observer_safe_policy_identity_v1(configured["native_play_policy_id"], opponent)
        except (TypeError, ValueError):
            fail()
        binding, bounds = policy.get("nativePlayPolicyBinding"), scope.get("utilityBounds")
        candidate = outcome.get("candidate")
        expected_bindings = sorted(("native_constant_" + kind + "_policy_binding_v1", action)
            for kind, key in (("auction", "auction_action"), ("escalation", "escalation_action"))
            if (action := configured[key]) is not None)
        projects = scope.get("opponentCeilingProjectActions", {})
        floor_candidate = floor.get("candidate")
        if not isinstance(floor_candidate, Mapping):
            fail()
        try:
            witnessed_projects = {row["actor"]: row["selectedAction"]
                for row in floor_candidate.get("projectObservations", ())
                if not row.get("authoredPrefix", False) and Seat.parse(row["actor"]).team == opponent}
        except (AttributeError, KeyError, TypeError, ValueError):
            fail()
        if (not isinstance(projects, Mapping) or policy.get("projectActions", {}) != projects
                or "opponentCeilingProjectActions" in scope and projects != witnessed_projects):
            fail()
        expected_bindings.extend(("native_constant_project_policy_binding_v1", action) for action in projects.values())
        expected_bindings.sort()
        bindings = receipt.get("bindings")
        if (not hashed(receipt) or not hashed(outcome) or not hashed(floor) or not hashed(binding)
                or receipt.get("kind") != "native_constant_policy_round_query_v1"
                or type(receipt.get("schemaVersion")) is not int or receipt["schemaVersion"] != 1
                or receipt.get("strategicCoverage") is not False
                or receipt.get("status") != "queried" or receipt.get("rulesHash") != scope.get("rulesHash")
                or receipt.get("scope") != owners[models[0]] or outcome.get("scope") != owners[models[0]]
                or receipt.get("targetTeam") != opponent.name.lower()
                or outcome.get("targetTeam") != opponent.name.lower()
                or outcome.get("solverStatus") != "optimal" or outcome.get("status") != "witness_found"
                or outcome.get("queryMode") == "feasibility_witness"
                or policy.get("team") != opponent.name.lower() or policy.get("nativePlayPolicy") != identity
                or policy.get("scope") != "future_nonplay_constants_and_native_visible_play"
                or type(policy.get("playPrefixLength")) is not int or policy["playPrefixLength"] < 0
                or policy.get("auctionAction") != configured["auction_action"]
                or policy.get("escalationAction") != configured["escalation_action"]
                or binding.get("kind") != "native_visible_play_policy_binding_v1" or binding.get("policy") != identity
                or set(binding) != {"schemaVersion", "kind", "policy", "sourceCpModelProtoHash",
                    "boundCpModelProtoHash", "sha256"}
                or type(binding.get("schemaVersion")) is not int or binding["schemaVersion"] != 1
                or any(type(binding.get(key)) is not str or _HASH.fullmatch(binding[key]) is None
                    for key in ("sourceCpModelProtoHash", "boundCpModelProtoHash"))
                or binding["sourceCpModelProtoHash"] == binding["boundCpModelProtoHash"]
                or type(policy.get("playPolicyModelHash")) is not str
                or _HASH.fullmatch(policy["playPolicyModelHash"]) is None
                or outcome.get("baseModelHash") != policy["playPolicyModelHash"]
                or not isinstance(bindings, list) or any(not hashed(item) or item.get("status") != "bound" for item in bindings)
                or sorted((item.get("kind"), item.get("action")) for item in bindings) != expected_bindings
                or wrapper.get("sourceWorldQueryHash") != floor["sha256"]
                or floor.get("status") != "witness_found" or floor.get("solverStatus") not in (
                    {"optimal", "feasible"} if floors[0].get("purpose") == "adversarial_world_witness" else {"optimal"})
                or floor.get("targetTeam") != target_team or floor.get("scope") != owners[models[0]]
                or not isinstance(candidate, Mapping) or not isinstance(floor.get("candidate"), Mapping)
                or candidate.get("deckOrder") != floor["candidate"].get("deckOrder")
                or type(candidate.get("utility")) is not int or type(floor["candidate"].get("utility")) is not int
                or not isinstance(bounds, list) or len(bounds) != 2
                or any(type(item) is not int for item in bounds) or bounds[0] > bounds[1]
                or bounds[1] != -candidate["utility"] or floor["candidate"]["utility"] > bounds[1]):
            fail()
    fixed_ceiling = scope.get("fixedWorldCooperativeCeiling")
    if fixed_ceiling is not None:
        bounds = scope.get("utilityBounds")
        if (len(models) != 1 or not isinstance(fixed_ceiling, Mapping)
                or set(fixed_ceiling) != {"value", "queryHash", "sourceWorldQueryHash"}
                or type(fixed_ceiling["value"]) is not int
                or not isinstance(bounds, list) or len(bounds) != 2
                or bounds[1] != fixed_ceiling["value"]):
            fail()
        floors = [row for row in queries if row.get("purpose") in
            {"root_policy_path_floor", "universal_path_floor"}
            and row["receipt"].get("sha256") == fixed_ceiling["sourceWorldQueryHash"]]
        ceilings = [row for row in queries if row.get("purpose") == "adversarial_world_upper_bound"
            and row["receipt"].get("sha256") == fixed_ceiling["queryHash"]]
        if len(floors) != 1 or len(ceilings) != 1:
            fail()
        floor, ceiling = floors[0]["receipt"], ceilings[0]["receipt"]
        floor_candidate, ceiling_candidate = floor.get("candidate"), ceiling.get("candidate")
        if (not hashed(floor) or not hashed(ceiling)
                or ceilings[0].get("sourceWorldQueryHash") != floor["sha256"]
                or floor.get("status") != "witness_found" or floor.get("solverStatus") != "optimal"
                or ceiling.get("status") != "witness_found" or ceiling.get("solverStatus") != "optimal"
                or floor.get("targetTeam") != target_team
                or ceiling.get("targetTeam") != ("west_east" if target_team == "north_south" else "north_south")
                or floor.get("scope") != owners[models[0]] or ceiling.get("scope") != owners[models[0]]
                or not isinstance(floor_candidate, Mapping) or not isinstance(ceiling_candidate, Mapping)
                or floor_candidate.get("deckOrder") != ceiling_candidate.get("deckOrder")
                or type(floor_candidate.get("utility")) is not int
                or type(ceiling_candidate.get("utility")) is not int
                or floor_candidate["utility"] > fixed_ceiling["value"]
                or -ceiling_candidate["utility"] != fixed_ceiling["value"]):
            fail()
    restored = {}
    for identity, retained in refinements.items():
        if not isinstance(retained, Mapping) or set(retained) != {"core", "queryHashes"} or not isinstance(retained["core"], Mapping):
            fail()
        core, references = retained["core"], retained["queryHashes"]
        if {"queries", "sha256"} & core.keys():
            fail()
        owner = core.get("sourceModelHash")
        if (type(owner) is not str or owner not in owners or not isinstance(references, list) or not references
                or any(type(key) is not str for key in references) or len(set(references)) != len(references)
                or any((owner, key) not in lookup for key in references)):
            fail()
        receipt = {**core, "queries": [lookup[owner, key] for key in references]}
        if (canonical_sha256(receipt) != identity
                or any(not hashed(query) or query.get("scope") != owners[owner] for query in receipt["queries"])):
            fail()
        restored[identity] = receipt
    for proof in proofs:
        owner = proof["branchModelHash"]
        if proof.get("infeasible") is True:
            if set(proof) != {"branchModelHash", "infeasible", "queryHash"} or type(proof.get("queryHash")) is not str:
                fail()
            query = lookup.get((owner, proof.get("queryHash")), {})
            bounds = scope.get("utilityBounds")
            if (not isinstance(bounds, list) or len(bounds) != 2
                    or any(type(bound) is not int for bound in bounds) or bounds[0] > bounds[1]
                    or not hashed(query) or query.get("baseModelHash") != owner
                    or query.get("scope") != owners[owner] or query.get("solverStatus") != "infeasible"
                    or query.get("status") != "no_matching_outcome"
                    or type(query.get("threshold")) is not int or query["threshold"] <= -bounds[0]
                    or query.get("targetTeam") != ("west_east" if target_team == "north_south" else "north_south")
                    or any(query.get(kind + "PolicyBindings") != [] for kind in ("auction", "escalation", "project", "play"))):
                fail()
            continue
        if (set(proof) != {"branchModelHash", "refinementHash", "policyIndependent"}
                or type(proof.get("refinementHash")) is not str or type(proof.get("policyIndependent")) is not bool):
            fail()
        receipt = restored.get(proof.get("refinementHash"), {})
        terminal = receipt.get("queries", [{}])[-1]
        bindings = receipt.get("bindings")
        inventories = [terminal.get(kind + "PolicyBindings") for kind in ("auction", "escalation", "project", "play")]
        if (receipt.get("sourceModelHash") != owner or receipt.get("rulesHash") != scope.get("rulesHash")
                or receipt.get("scope") != owners[owner] or receipt.get("targetTeam") != target_team
                or receipt.get("status") != "no_adverse_round_in_branch"
                or type(receipt.get("threshold")) is not int or receipt["threshold"] < value
                or not isinstance(bindings, list) or any(not isinstance(rows, list) for rows in inventories)
                or sorted(canonical_sha256(row) for row in bindings) != sorted(canonical_sha256(row) for rows in inventories for row in rows)
                or proof["policyIndependent"] != (not bindings and fallback is None)
                or not hashed(terminal) or terminal.get("status") != "no_matching_outcome" or terminal.get("solverStatus") != "infeasible"
                or terminal.get("scope") != owners[owner] or terminal.get("targetTeam") != target_team
                or type(terminal.get("threshold")) is not int or terminal["threshold"] != receipt["threshold"]
                or type(receipt.get("boundModelHash")) is not str or not receipt["boundModelHash"]
                or terminal.get("baseModelHash") != receipt["boundModelHash"]):
            fail()
        if fallback is not None:
            overrides, binding = receipt.get("playOverrides"), receipt.get("fallbackBinding")
            witnessed_play_keys = {row["observationHash"] for wrapper in queries
                if wrapper["branchModelHash"] == owner
                for row in (wrapper["receipt"].get("candidate") or {}).get("playObservations", ())
                if not row.get("authoredPrefix", False)
                and Seat.parse(row["actor"]).team.name.lower() == target_team}
            if (receipt.get("kind") != "native_guarded_round_refinement_v1"
                    or receipt.get("playFallback") != fallback or not isinstance(overrides, Mapping)
                    or not witnessed_play_keys.intersection(policy) <= overrides.keys()
                    or any(key not in contexts or key not in policy or row != {
                        "information": contexts[key], "action": policy[key]} for key, row in overrides.items())):
                fail()
            if owners[owner].get("allPassedCovered") is True:
                if binding is not None or overrides:
                    fail()
            elif (not hashed(binding) or set(binding) != {"schemaVersion", "kind", "policy", "overridesHash",
                    "sourceCpModelProtoHash", "boundCpModelProtoHash", "sha256"}
                    or type(binding["schemaVersion"]) is not int or binding["schemaVersion"] != 1
                    or binding["kind"] != "native_guarded_play_policy_binding_v1"
                    or binding["policy"] != fallback or binding["overridesHash"] != canonical_sha256(overrides)
                    or any(type(binding[key]) is not str or _HASH.fullmatch(binding[key]) is None
                        for key in ("sourceCpModelProtoHash", "boundCpModelProtoHash"))):
                fail()
            play_bindings = inventories[-1]
            if ({row.get("observationHash") for row in play_bindings} != set(overrides)
                    or len(play_bindings) != len(overrides)):
                fail()
        elif any(key in receipt for key in ("playFallback", "playOverrides", "fallbackBinding")):
            fail()
        for kind, rows in zip(("auction", "escalation", "project", "play"), inventories, strict=True):
            if any(not isinstance(row, Mapping) or type(row.get("schemaVersion")) is not int
                    or row["schemaVersion"] != 1 or row.get("kind") != f"native_{kind}_policy_binding_v1"
                    or row.get("solverStatus") != "INFEASIBLE" for row in rows):
                fail()
        if proof["policyIndependent"]:
            if receipt["boundModelHash"] != owner:
                fail()
        elif (receipt.get("policyHash") != canonical_sha256(
                {"fallback": fallback, "policy": policy} if fallback is not None else policy)
                or any(not hashed(row) or row.get("status") != "bound" or row.get("rulesHash") != scope.get("rulesHash")
                    or row.get("observationHash") not in policy or row.get("action") != policy[row["observationHash"]] for row in bindings)):
            fail()


def verify_native_complete_state_policy_pair_v1(
        proofs, *, rules, normalized, source_prefix, request, budget,
        cancel_check=None, maximum_queries=None, maximum_nodes=None,
        reuse_joint_witness=True):
    """Recheck two fixed-state guarantees; not whole-round or media admission."""
    import time
    from .ortools_exact_state import OrToolsExactStateNormalizerV1
    from .ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1

    if (not isinstance(budget, ScenarioProofBudgetV1) or budget.native_observed is None
            or request.mode != "exact" or request.visibility != "complete_state"
            or request.utility != "awarded_point_margin"
            or request.budget_profile != budget.profile_id
            or type(reuse_joint_witness) is not bool):
        raise ValueError("paired policies require terminal complete-state play coverage")
    started, cpu_started = time.monotonic(), time.process_time()

    def remaining():
        if cancel_check is not None and cancel_check():
            raise TimeoutError("paired policy verification cancelled")
        seconds = min(budget.maximum_wall_seconds - (time.monotonic() - started),
                      budget.maximum_cpu_seconds - (time.process_time() - cpu_started))
        if seconds <= 0:
            raise TimeoutError("paired policy verification budget exhausted")
        return seconds

    fresh = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
        source_prefix, maximum_seconds=remaining(), cancel_check=cancel_check)
    normalized.require_integrity()
    if fresh != normalized:
        raise ValueError("paired policy exact state differs from native normalization")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    worlds, contexts, preparation = compiler._prepare_complete_state_play(
        fresh, source_prefix, request, budget=budget, maximum_seconds=remaining(),
        maximum_nodes=maximum_nodes)
    if len(worlds) != 1:
        raise ValueError("paired policy verification requires one fixed complete state")
    if (preparation["modelBytes"] > budget.maximum_cache_bytes
            or preparation["remainingActions"] > min(budget.maximum_depth, budget.maximum_horizon_actions)):
        raise ValueError("paired policy model exceeds the selected resource tier")
    start = worlds[0][1].start
    project_context = contexts.get(worlds[0][1].model_hash)
    oracle_args = ({"play_step_count": len(start.cards), "project_prefix": source_prefix,
        "project_continuation_actions": worlds[0][1].project_continuation_actions}
        if project_context is not None else {})
    actor_team = Seat.parse(fresh.next_actor).team
    teams = (actor_team, actor_team.other)
    if not isinstance(proofs, Mapping) or set(proofs) != {team.name.lower() for team in teams}:
        raise ValueError("paired policy team coverage mismatch")
    checks, values = {}, []
    query_limit = budget.native_observed.maximum_queries if maximum_queries is None else maximum_queries
    if type(query_limit) is not int or not 0 < query_limit <= budget.native_observed.maximum_queries:
        raise ValueError("paired policy query allowance exceeds the frozen tier")
    queries_left = query_limit
    for team in teams:
        proof = proofs[team.name.lower()]
        oracle = compiler.graph_compiler._path_oracle(start, target=team,
            utility=request.utility, cancel_check=cancel_check, **oracle_args)
        scope = proof.get("scope", {}) if isinstance(proof, Mapping) else {}
        expected_utility = {"utility": oracle.utility, "boundaryEvaluator": None,
            "playStepCount": len(start.cards), "basePrefixLength": oracle.base_length,
            "cardCount": len(start.cards)}
        if (not isinstance(proof, Mapping)
                or not isinstance(scope, Mapping)
                or proof.get("sha256") != canonical_sha256({k: v for k, v in proof.items() if k != "sha256"})
                or proof.get("status") != "converged" or type(proof.get("value")) is not int
                or proof.get("worldCount") != 1 or proof.get("scopeHash") != canonical_sha256(scope)
                or scope.get("kind") != ("native_visible_path_policy_floor_scope_v1"
                    if "nativePlayPolicy" in proof or "rankedPlayPolicies" in proof else "native_observed_world_set_v1")
                or scope.get("rulesHash") != rules.sha256
                or scope.get("targetTeam") != team.name.lower()
                or scope.get("coverage") != "supplied_worlds_only"
                or scope.get("worlds") != [canonical_sha256({"state": fresh.state, "recall": {}})
                    if project_context is not None else canonical_sha256({"state": fresh.state, "model": oracle.model_hash, "recall": {}})]
                or scope.get("utilityScope") != expected_utility
                or "selectedRootAction" in scope
                or (scope.get("declarationCoverage") != {project_context[2]["sha256"]: project_context[2]}
                    or scope.get("branches") != [{"branchId": canonical_sha256({"state": fresh.state, "model": oracle.model_hash, "recall": {}}),
                        "worldId": canonical_sha256({"state": fresh.state, "recall": {}}),
                        "modelHash": oracle.model_hash, "coverageHash": project_context[2]["sha256"]}]
                    if project_context is not None else "branches" in scope or "declarationCoverage" in scope)):
            raise ValueError("paired policy receipt ownership or coverage mismatch")
        policy, domains = proof.get("policy"), proof.get("domains")
        if "nativePlayPolicy" in proof or "rankedPlayPolicies" in proof:
            from .ortools_play_tail import native_observer_safe_policy_identity_v1
            identity = proof.get("nativePlayPolicy")
            ranked = proof.get("rankedPlayPolicies")
            if (policy != [] or domains != {} or proof.get("kind") != "native_visible_path_policy_floor_v1"
                    or ("nativePlayPolicy" in proof) == ("rankedPlayPolicies" in proof)
                    or identity is not None and (not isinstance(identity, Mapping)
                        or identity != native_observer_safe_policy_identity_v1(identity.get("policyId"), team))
                    or "rankedPlayPolicies" in proof and not isinstance(ranked, Mapping)):
                raise ValueError("paired native policy identity mismatch")
            bound = oracle.with_native_play_policy(team=team,
                policy_id=identity["policyId"] if identity is not None else None, play_policies=ranked)
            if (proof.get("nativePolicyModelHash") != bound.model_hash
                    or len(str(bound.build.model.proto).encode("utf-8")) + preparation["modelBytes"] > budget.maximum_cache_bytes):
                raise ValueError("paired native policy model ownership or budget mismatch")
            if queries_left <= 0:
                raise TimeoutError("paired policy query budget exhausted")
            queries_left -= 1
            witness, value, query_hash = bound.find_path_below(bound.query_utility_bounds()[1] + 1,
                path=(), conditional_actions={}, maximum_seconds=remaining(), cancelled=cancel_check)
            remaining()
            if witness is None or value != proof["value"] or proof.get("queryHashes") != [query_hash]:
                raise ValueError("paired native policy guarantee was not established")
            checks[team.name.lower()] = {"policyReceiptHash": proof["sha256"],
                "queryHashes": [query_hash], "value": value}
            values.append(value)
            continue
        if (not isinstance(policy, (list, tuple)) or not isinstance(domains, Mapping)
                or any(not isinstance(row, (list, tuple)) or len(row) != 2
                       or any(type(item) is not str for item in row) for row in policy)
                or len(dict(policy)) != len(policy)
                or any(not isinstance(domain, (list, tuple)) or not domain
                       or any(type(action) is not str for action in domain)
                       or len(set(domain)) != len(domain) for domain in domains.values())
                or any(key not in domains or action not in domains[key] for key, action in policy)):
            raise ValueError("paired policy action domain mismatch")
        from .ortools_complete_state_graph import iter_native_path_query_guards_v1
        history = [row["receipt"] for row in proof.get("queries", ())
            if isinstance(row, Mapping) and isinstance(row.get("receipt"), Mapping)
            and ("conditionalActions" in row["receipt"] or "conditionalActionsAdded" in row["receipt"])]
        for _, _guards in iter_native_path_query_guards_v1(history,
                maximum_bindings=budget.native_observed.maximum_bound_prefixes):
            remaining()
        if queries_left <= 0:
            raise TimeoutError("paired policy query budget exhausted")
        check = compiler._find_observed_policy_counterexample(fresh.state, oracle, team,
            dict(policy), {}, threshold=proof["value"],
            project_context=project_context,
            known_domains={key: tuple(actions) for key, actions in domains.items()},
            maximum_queries=queries_left,
            maximum_evidence_bytes=budget.maximum_cache_bytes,
            maximum_prefixes=min(budget.native_observed.maximum_observations, preparation["maximumNodeObservations"]),
            maximum_seconds=remaining(), cancel_check=cancel_check)
        queries_left -= len(check["queryHashes"])
        remaining()
        if check["status"] != "no_counterexample":
            raise ValueError("paired policy guarantee was not established")
        checks[team.name.lower()] = {"policyReceiptHash": proof["sha256"],
            "queryHashes": list(check["queryHashes"]), "value": proof["value"]}
        values.append(proof["value"])
    if values[0] != -values[1]:
        raise ValueError("paired policy guarantees do not meet")
    joint = worlds[0][1]
    for team in teams:
        if "nativePlayPolicy" in proofs[team.name.lower()] or "rankedPlayPolicies" in proofs[team.name.lower()]:
            joint = joint.with_native_play_policy(team=team,
                policy_id=proofs[team.name.lower()].get("nativePlayPolicy", {}).get("policyId"),
                play_policies=proofs[team.name.lower()].get("rankedPlayPolicies"))
    path, decisions, path_queries = (), [], []
    cached_joint = None
    projected_policies = {}

    def reachable():
        nonlocal queries_left, cached_joint
        remaining()
        if (reuse_joint_witness and cached_joint is not None
                and cached_joint[0].plays[joint.base_length:joint.base_length + len(path)] == path):
            return cached_joint
        if queries_left <= 0:
            raise TimeoutError("paired policy query budget exhausted")
        queries_left -= 1
        witness, value, query_hash = joint.find_path_below(joint.query_utility_bounds()[1] + 1,
            path=path, conditional_actions={}, maximum_seconds=remaining(),
            cancelled=cancel_check, feasibility_only=True)
        remaining()
        if witness is None:
            raise ValueError("paired policy joint continuation is unreachable")
        path_queries.append(query_hash)
        cached_joint = witness, value
        projected_policies.clear()
        return cached_joint

    for _ in range(len(start.cards) - joint.base_length):
        witness, _ = reachable()
        actor = witness.at_prefix(joint.base_length + len(path)).next_actor
        proof = proofs[actor.team.name.lower()]
        if not reuse_joint_witness or actor.team not in projected_policies:
            observations = {}
            bindings, _, _ = compiler._observed_policy_bindings(fresh.state, joint, witness,
                actor.team, dict(proof["policy"]), {}, project_context=project_context,
                known_domains={key: tuple(actions) for key, actions in proof["domains"].items()},
                maximum_seconds=remaining(), cancel_check=cancel_check,
                _prefix_observations=observations)
            projected_policies[actor.team] = bindings, observations
        bindings, observations = projected_policies[actor.team]
        _, _, observation, actions = observations[path]
        # Unbound observations were universally covered by the fresh adverse
        # search; choose a stable legal completion, not an invented intention.
        native = "nativePlayPolicy" in proof or "rankedPlayPolicies" in proof
        action = witness.plays[joint.base_length + len(path)] if native else bindings.get(path, actions[min(actions)])
        if action is None:
            raise ValueError("paired policy excludes its joint continuation")
        decisions.append({"playIndex": len(source_prefix.play_actions) + len(path),
            "actor": actor.name.lower(), "observationHash": observation,
            "legalActions": sorted(actions), "action": action.to_dict(),
            "selection": "native_policy" if native else "policy" if path in bindings else "universally_covered_completion",
            "policyReceiptHash": proof["sha256"]})
        path = (*path, action)
    witness, terminal_value = reachable()
    if terminal_value != values[0]:
        raise ValueError("paired policy terminal value disagrees with guarantees")
    continuation = {"baseModelHash": joint.model_hash, "queryHashes": path_queries,
        "decisions": decisions, "plays": [action.to_dict() for action in witness.plays],
        "utilityTeam": actor_team.name.lower(), "terminalValue": terminal_value,
        **({"projectCoverageHash": project_context[2]["sha256"],
            "projectActions": [*source_prefix.project_actions, *project_context[2]["sequences"][0]]}
            if project_context is not None else {})}
    receipt = {"schemaVersion": 1, "kind": "native_complete_state_policy_pair_v1",
        "coverage": "fixed_state_terminal_play_suffix", "wholeRoundCertified": False,
        "rulesHash": rules.sha256, "stateHash": canonical_sha256(fresh.state),
        "sourcePrefixHash": source_prefix.sha256, "exactStateId": fresh.exact_state_id,
        "proofRequestHash": request.sha256, "proofBudgetHash": budget.sha256,
        "budgetProfile": budget.profile_id,
        "maximumNodeObservations": preparation["maximumNodeObservations"],
        "queryLimit": query_limit, "counterexampleAndReachabilityQueries": query_limit - queries_left,
        **({"jointWitnessReuse": True} if reuse_joint_witness else {}),
        "preparationHash": preparation["sha256"], "guarantees": checks,
        "continuation": {**continuation, "sha256": canonical_sha256(continuation)}}
    return {**receipt, "sha256": canonical_sha256(receipt)}


def validate_native_complete_state_policy_pair_v1(
        receipt, proofs, *, rules, normalized, source_prefix, request, budget,
        cancel_check=None):
    """Rebuild stored paired evidence from caller-owned frozen context."""
    if (not isinstance(receipt, Mapping)
            or receipt.get("sha256") != canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})
            or receipt.get("sourcePrefixHash") != source_prefix.sha256
            or receipt.get("exactStateId") != normalized.exact_state_id
            or receipt.get("proofRequestHash") != request.sha256
            or receipt.get("proofBudgetHash") != budget.sha256
            or receipt.get("rulesHash") != rules.sha256):
        raise ValueError("paired policy stored ownership mismatch")
    rebuilt = verify_native_complete_state_policy_pair_v1(proofs, rules=rules,
        normalized=normalized, source_prefix=source_prefix, request=request,
        budget=budget, cancel_check=cancel_check, maximum_queries=receipt.get("queryLimit"),
        maximum_nodes=receipt.get("maximumNodeObservations"),
        reuse_joint_witness=receipt.get("jointWitnessReuse", False))
    if canonical_sha256(rebuilt) != canonical_sha256(receipt):
        raise ValueError("paired policy stored evidence differs from native verification")


def build_native_paired_play_action_artifact_v1(pair, *, pair_hash, source_prefix,
        normalized, request):
    """Project one action from an independently verified, owned paired receipt.

    The containing trace reader must verify the shared pair, not trust this
    projection as a standalone solver certificate.
    """
    from .ortools_exact_state import _effective_prefix_hash
    normalized.require_integrity()
    if (request.mode != "exact" or request.visibility != "complete_state"
            or request.utility != "awarded_point_margin" or normalized.decision_kind != "play"
            or normalized.prefix_hash != _effective_prefix_hash(source_prefix)
            or normalized.rules_hash != pair["rulesHash"]):
        raise ValueError("paired play projection context mismatch")
    matches = [row for row in pair["continuation"]["decisions"]
               if row["playIndex"] == len(source_prefix.play_actions)]
    if len(matches) != 1:
        raise ValueError("paired play projection is outside verified coverage")
    row = matches[0]
    action = LegalPlayActionV1(Card.parse(row["action"]["card"]), row["action"].get("ekka", False)).code
    if (row["actor"] != normalized.next_actor
            or row["legalActions"] != sorted(item["code"] for item in normalized.legal_actions)
            or request.target_team != Seat.parse(normalized.next_actor).team.name.lower()
            or request.selected_action not in {None, action}):
        raise ValueError("paired play projection action mismatch")
    quality = {"schemaVersion": 1, "authority": "cp_sat", "scope": "compiled_complete_state",
        "rootActorRelation": "target", "actualAction": action, "actualActionOptimal": True,
        "regret": 0, "optimalActions": [action], "optimalActionsComplete": False,
        "optimalActionCountLowerBound": 1, "optimalActionCountUpperBound": len(row["legalActions"]),
        "bestValue": pair["guarantees"][request.target_team]["value"],
        "actualValue": pair["guarantees"][request.target_team]["value"]}
    quality["sha256"] = canonical_sha256(quality)
    model = {"schemaVersion": 1, "kind": "native_complete_state_policy_pair_quality_v1",
        "rootActorSeat": normalized.next_actor, "pairEvidenceHash": pair_hash,
        "playIndex": row["playIndex"], "actionQuality": quality}
    allocation = {"schemaVersion": 1, "kind": "native_complete_state_quality_v1",
        "sourcePrefixHash": source_prefix.sha256, "exactStateId": normalized.exact_state_id,
        "rulesHash": normalized.rules_hash, "sourcePrefix": source_prefix.to_dict()}
    allocation["sha256"] = canonical_sha256(allocation)
    graph = {"reference": "pairedPolicyEvidence", "sha256": pair_hash}
    proof = {"schemaVersion": 1, "completionStatus": "exact_complete",
        "completionScope": "selected_play_fixed_state_terminal_continuation",
        "claimSatisfied": None, "selectedAction": action, "winningRootActionCount": None,
        "rootActions": [{"action": code, "value": None, "satisfies": None} for code in row["legalActions"]],
        "identities": {"graphHash": pair_hash,
            "modelHash": canonical_sha256(model), "proofRequestHash": request.sha256}}
    proof["sha256"] = canonical_sha256(proof)
    return {"request": request.to_dict(), "proof": proof, "proofModel": model,
        "allocationModel": allocation, "graph": graph}


def validate_native_strategy_threshold_v1(strategy: Mapping[str, Any], *, expected_scope: Mapping[str, Any]) -> int:
    """Validate an owned threshold receipt, not exact value or global admission.

    The caller supplies the scope rebuilt from its frozen models. Receipt hashes
    check integrity; replay of those owned models remains a separate gate.
    """
    def fail():
        raise ValueError("native strategy threshold ownership or coverage mismatch")

    if not isinstance(strategy, Mapping) or not isinstance(expected_scope, Mapping):
        fail()
    scope = strategy.get("scope")
    threshold, bounds = expected_scope.get("requiredThreshold"), expected_scope.get("utilityBounds")
    target = expected_scope.get("targetTeam")
    if (type(threshold) is not int or type(target) is not str or target not in {"north_south", "west_east"}
            or not isinstance(bounds, list) or len(bounds) != 2
            or any(type(value) is not int for value in bounds) or bounds[0] > bounds[1]
            or not -(2**62) <= threshold <= bounds[1]
            or expected_scope.get("objective") != "strategy_threshold_satisfaction"
            or expected_scope.get("coverage") != "supplied_deal_branches_only"
            or scope != expected_scope or strategy.get("status") != "scoped_strategy_threshold_complete"
            or any(type(strategy.get(key)) is not int or strategy[key] != threshold
                   for key in ("requiredThreshold", "lowerBound", "masterSearchValue"))
            or type(strategy.get("upperBound")) is not int or strategy["upperBound"] != bounds[1]):
        fail()
    masters, policy, domains = (strategy.get(key) for key in ("masters", "policy", "domains"))
    if (not isinstance(masters, list) or not masters or any(not isinstance(row, Mapping)
            or row.get("status") != "optimal" or type(row.get("searchValue")) is not int
            or row["searchValue"] != threshold for row in masters)
            or not isinstance(policy, Mapping) or not isinstance(domains, Mapping)
            or any(not isinstance(actions, (list, tuple)) or not actions
                   or any(type(action) is not str for action in actions) or len(set(actions)) != len(actions)
                   for actions in domains.values())
            or any(type(key) is not str or type(action) is not str or key not in domains
                   or action not in domains[key] for key, action in policy.items())):
        fail()
    fixed = expected_scope.get("fixedRoot")
    if fixed is not None:
        if (not isinstance(fixed, Mapping) or set(fixed) != {"observationHash", "legalActions", "selectedAction"}
                or type(fixed["observationHash"]) is not str or not isinstance(fixed["legalActions"], list)
                or fixed["selectedAction"] not in fixed["legalActions"]
                or policy.get(fixed["observationHash"]) != fixed["selectedAction"]
                or list(domains.get(fixed["observationHash"], ())) != fixed["legalActions"]):
            fail()
    _validate_native_final_branch_proofs_v2(strategy, target_team=target, required_threshold=threshold)
    return threshold


def validate_native_visible_policy_threshold_v1(value, *, rules, scenario, target,
        policy_team, native_play_policy_id, auction_action, escalation_action,
        required_threshold, maximum_seconds, maximum_conflicts,
        record_model_bytes=False, cancel_check=None, project_actions=None, native_project_policy_id=None,
        native_auction_policy_id=None, record_policy_model_hashes=False, native_play_overrides=None):
    """Freshly verify one native-policy branch floor, not universe coverage."""
    from .ortools_scenario_model import OrToolsScenarioModelV1
    from .ortools_information_set_compiler import InformationSetCompilationStopped

    if native_play_overrides is not None and not isinstance(native_play_overrides, (list, tuple)):
        raise ValueError("native visible policy overrides require an explicit sequence")
    guarded = bool(native_play_overrides)
    if (target != policy_team
            or not isinstance(scenario, OrToolsScenarioModelV1) or scenario.rules.sha256 != rules.sha256
            or not isinstance(value, Mapping)
            or value.get("kind") != ("native_guarded_play_policy_round_query_v1" if guarded
                else "native_constant_policy_round_query_v1")
            or value.get("status") != "queried"
            or not isinstance(value.get("policy"), Mapping)
            or value["policy"].get("scope") != ("future_nonplay_constants_and_guarded_native_visible_play" if guarded
                else "future_nonplay_constants_and_native_visible_play")
            or value.get("sha256") != canonical_sha256({k: v for k, v in value.items() if k != "sha256"})):
        raise ValueError("native visible policy threshold ownership mismatch")
    rebuilt = scenario.find_native_round_with_constant_policy_below(
        target=target, policy_team=policy_team, native_play_policy_id=native_play_policy_id,
        auction_action=auction_action, escalation_action=escalation_action,
        threshold=required_threshold, optimize_margin=False, maximum_seconds=maximum_seconds,
        maximum_conflicts=maximum_conflicts, record_model_bytes=record_model_bytes, cancel_check=cancel_check,
        **({"record_policy_model_hashes": True} if record_policy_model_hashes else {}),
        **({"project_actions": project_actions} if project_actions is not None else {}),
        **({"native_project_policy_id": native_project_policy_id} if native_project_policy_id is not None else {}),
        **({"native_auction_policy_id": native_auction_policy_id} if native_auction_policy_id is not None else {}),
        **({"native_play_overrides": native_play_overrides} if native_play_overrides is not None else {}))
    outcome = rebuilt.get("query") or {}
    if rebuilt.get("status") == "incomplete":
        raise InformationSetCompilationStopped(rebuilt.get("exhaustionReason") or "solver_unknown",
            stage="native_visible_policy_threshold_reader", evidence=rebuilt)
    if (rebuilt.get("status") != "queried" or outcome.get("status") != "no_matching_outcome"
            or outcome.get("solverStatus") != "infeasible" or outcome.get("candidate") is not None
            or outcome.get("queryMode") != "feasibility_witness"):
        raise ValueError("native visible policy threshold did not survive reverification")
    if canonical_sha256(value) != canonical_sha256(rebuilt):
        raise ValueError("native visible policy threshold evidence mismatch")
    return required_threshold


def reverify_native_bid_incumbent_floor_v1(value, *, rules, source_prefix, normalized,
        native_play_policy_id, budget, maximum_conflicts, maximum_seconds,
        opportunity_order=None, cancel_check=None, nonplay_policy=None):
    """Fresh native root/partition/policy verification; not bid optimality."""
    import time
    from .ortools_exact_state import OrToolsExactStateNormalizerV1, OrToolsNormalizedExactStateV1
    from .ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1, InformationSetCompilationStopped
    from .ortools_play_tail import native_observer_safe_policy_identity_v1

    def fail():
        raise ValueError("native bid incumbent verification ownership or budget mismatch")

    if (not isinstance(value, Mapping) or not isinstance(rules, RuleBook)
            or not isinstance(source_prefix, ExactTracePrefixV1)
            or not isinstance(normalized, OrToolsNormalizedExactStateV1)
            or not isinstance(budget, ScenarioProofBudgetV1) or budget.native_observed is None
            or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
            or not 0 < maximum_seconds <= min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds)
            or type(maximum_conflicts) is not int or maximum_conflicts < 1
            or cancel_check is not None and not callable(cancel_check)):
        fail()
    deadline, cpu_deadline = time.monotonic() + maximum_seconds, time.process_time() + maximum_seconds
    normalized.require_integrity()
    if (normalized.decision_kind != "bid" or normalized.rules_hash != rules.sha256
            or value.get("kind") != "native_bid_policy_bounds_v1"
            or value.get("visibility") != "complete_state" or value.get("strategicCoverage") is not False
            or value.get("targetTeam") != Seat.parse(normalized.next_actor).team.name.lower()
            or value.get("selectedRootAction") not in {row["code"] for row in normalized.legal_actions}
            or value.get("status") not in {"scoped_incumbent_bound_complete", "scoped_incumbent_threshold_complete"}
            or any(key in value for key in ("rootActionValues", "requestedRootActions", "cause"))):
        fail()
    target = Seat.parse(normalized.next_actor).team
    if nonplay_policy is not None:
        from .scenario_contracts import validate_native_bid_nonplay_policy_v1
        nonplay_policy = validate_native_bid_nonplay_policy_v1(nonplay_policy)
    identity = native_observer_safe_policy_identity_v1(native_play_policy_id, target)
    preparation, strategy = value.get("preparation"), value.get("strategy")
    if not isinstance(preparation, Mapping) or not isinstance(strategy, Mapping):
        fail()
    limits, strategy_limits, scope = preparation.get("limits"), strategy.get("limits"), strategy.get("scope")
    seconds = value.get("maximumSeconds")
    if (not isinstance(limits, Mapping) or not isinstance(strategy_limits, Mapping) or not isinstance(scope, Mapping)
            or type(seconds) not in {int, float} or not math.isfinite(seconds)
            or not 0 < seconds <= min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds)
            or scope.get("incumbentNativePlayPolicy") != identity
            or scope.get("incumbentNonplayPolicy") != nonplay_policy
            or "incumbentPlayPoliciesHash" in scope or strategy.get("status") != value["status"]):
        fail()
    native = budget.native_observed
    caps = ((limits, "branches", budget.maximum_state_space), (limits, "modelBytes", budget.maximum_cache_bytes),
        (limits, "queries", native.maximum_queries), (limits, "conflictsPerSolve", maximum_conflicts),
        (limits, "maximumDepth", min(budget.maximum_depth, budget.maximum_horizon_actions)),
        (strategy_limits, "masterIterations", native.maximum_iterations),
        (strategy_limits, "observations", native.maximum_observations),
        (strategy_limits, "queries", native.maximum_queries))
    if any(type(document.get(key)) is not int or not 0 < document[key] <= cap for document, key, cap in caps):
        fail()
    used = (value.get("queriesUsed"), preparation.get("queriesUsed"), strategy.get("queriesUsed"))
    if (any(type(count) is not int or count < 0 for count in used)
            or used[0] != used[1] + used[2] or used[0] > limits["queries"]
            or strategy_limits["queries"] != limits["queries"] - used[1]
            or strategy_limits.get("branches") != limits["branches"]
            or strategy_limits.get("conflictsPerSolve") != limits["conflictsPerSolve"]):
        fail()
    threshold = scope.get("incumbentThreshold")
    if (type(threshold) is not int if value["status"] == "scoped_incumbent_threshold_complete"
            else "incumbentThreshold" in scope):
        fail()
    recorded = _native_observed_bounds_semantics(value, maximum_seconds=seconds)
    def stopped():
        return bool(cancel_check and cancel_check()) or time.monotonic() >= deadline or time.process_time() >= cpu_deadline

    def remaining():
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage="native_bid_incumbent_verification")
        available = min(deadline - time.monotonic(), cpu_deadline - time.process_time())
        if available <= 0:
            raise InformationSetCompilationStopped("maximum_seconds", stage="native_bid_incumbent_verification")
        return available

    rebuilt = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(source_prefix,
        decision_point={"kind": "bid"}, maximum_seconds=remaining(), cancel_check=stopped)
    if rebuilt.to_dict() != normalized.to_dict():
        fail()
    replay = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._solve_native_bid_policy_bounds(
        rebuilt, source_prefix, target, visibility="complete_state", selected_root_action=value["selectedRootAction"],
        compact_purchase_invariant_ownership=preparation.get("scope", {}).get("ownershipCompaction") == "purchase_recipient_invariant_v1",
        maximum_iterations=strategy_limits["masterIterations"], maximum_queries=limits["queries"],
        maximum_observations=strategy_limits["observations"], maximum_branches=limits["branches"],
        maximum_model_bytes=limits["modelBytes"], maximum_conflicts=limits["conflictsPerSolve"],
        maximum_depth=limits["maximumDepth"], maximum_seconds=seconds, cancel_check=stopped,
        opportunity_order=opportunity_order, incumbent_native_play_policy_id=native_play_policy_id,
        **({"incumbent_nonplay_policy": nonplay_policy} if nonplay_policy is not None else {}),
        incumbent_only=True, **({"incumbent_threshold": threshold} if threshold is not None else {}))
    remaining()
    replayed = _native_observed_bounds_semantics(replay, maximum_seconds=seconds)
    remaining()
    if replayed != recorded:
        raise ValueError("native bid incumbent did not survive fresh reconstruction")
    return replay["strategy"]["lowerBound"]


def validate_native_incumbent_lower_bound_v2(strategy, *, target_team, play_policies=None,
        rules_hash, source_model_hashes, native_play_policy_id=None, source_models=None, nonplay_policy=None) -> int:
    return _validate_native_incumbent_floor(strategy, target_team=target_team, play_policies=play_policies,
        rules_hash=rules_hash, source_model_hashes=source_model_hashes, threshold_mode=False,
        native_play_policy_id=native_play_policy_id, source_models=source_models, nonplay_policy=nonplay_policy)


def validate_native_incumbent_threshold_floor_v1(strategy, *, target_team, play_policies=None,
        rules_hash, source_model_hashes, native_play_policy_id=None, source_models=None, nonplay_policy=None) -> int:
    return _validate_native_incumbent_floor(strategy, target_team=target_team, play_policies=play_policies,
        rules_hash=rules_hash, source_model_hashes=source_model_hashes, threshold_mode=True,
        native_play_policy_id=native_play_policy_id, source_models=source_models, nonplay_policy=nonplay_policy)


def _validate_native_incumbent_floor(strategy, *, target_team, play_policies,
        rules_hash, source_model_hashes, threshold_mode, native_play_policy_id=None, source_models=None, nonplay_policy=None) -> int:
    """Read an owned policy floor, not convergence, optimality, or admission."""
    from .ortools_proof import DeclarativeActionPolicyV1

    def fail():
        raise ValueError("native incumbent lower bound ownership or coverage mismatch")

    def hashed(value):
        return isinstance(value, Mapping) and value.get("sha256") == canonical_sha256(
            {key: item for key, item in value.items() if key != "sha256"})

    if (type(rules_hash) is not str or not rules_hash
            or not isinstance(source_model_hashes, tuple) or not source_model_hashes
            or any(type(item) is not str or not item for item in source_model_hashes)
            or len(set(source_model_hashes)) != len(source_model_hashes)
            or type(target_team) is not str or target_team not in {"north_south", "west_east"}
            or not hashed(strategy) or strategy.get("kind") != ("native_incumbent_threshold_floor_v1"
                if threshold_mode else "native_symbolic_round_policy_bounds_v2")
            or type(strategy.get("schemaVersion")) is not int or strategy["schemaVersion"] != 1
            or strategy.get("strategicCoverage") is not False):
        fail()
    native_policy = None
    if nonplay_policy is not None:
        from .scenario_contracts import validate_native_bid_nonplay_policy_v1
        nonplay_policy = validate_native_bid_nonplay_policy_v1(nonplay_policy)
        if native_play_policy_id is None:
            fail()
    models_by_hash = {}
    if native_play_policy_id is not None:
        if play_policies is not None:
            fail()
        from .ortools_scenario_model import OrToolsScenarioModelV1
        if (not isinstance(source_models, tuple) or len(source_models) != len(source_model_hashes)
                or any(not isinstance(model, OrToolsScenarioModelV1) or model.rules.sha256 != rules_hash
                       for model in source_models)):
            fail()
        models_by_hash = {model._model_hash(search_branching="automatic"): model for model in source_models}
        if set(models_by_hash) != set(source_model_hashes):
            fail()
        from .ortools_play_tail import native_observer_safe_policy_identity_v1
        native_policy = native_observer_safe_policy_identity_v1(native_play_policy_id,
            Seat.NORTH.team if target_team == "north_south" else Seat.WEST.team)
        identities = None
    else:
        if source_models is not None:
            fail()
        if (not isinstance(play_policies, Mapping) or set(play_policies) != (
                {"north", "south"} if target_team == "north_south" else {"west", "east"})):
            fail()
        policies = {seat: DeclarativeActionPolicyV1.build(seat, raw) for seat, raw in play_policies.items()}
        if any(policy.choices or policy.materialization is None for policy in policies.values()):
            fail()
        identities = {seat: policy.identity_dict() for seat, policy in policies.items()}
    scope, queries = strategy.get("scope"), strategy.get("queries")
    if (not isinstance(scope, Mapping) or strategy.get("scopeHash") != canonical_sha256(scope)
            or scope.get("targetTeam") != target_team or scope.get("rulesHash") != rules_hash
            or scope.get("incumbentNonplayPolicy") != nonplay_policy
            or (scope.get("incumbentPlayPoliciesHash") != canonical_sha256(play_policies)
                or "incumbentNativePlayPolicy" in scope if native_policy is None else
                scope.get("incumbentNativePlayPolicy") != native_policy or "incumbentPlayPoliciesHash" in scope)
            or not isinstance(queries, list) or any(not isinstance(row, Mapping) for row in queries)):
        fail()
    branches, bounds = scope.get("branches"), scope.get("utilityBounds")
    if (not isinstance(branches, list) or not branches
            or any(not isinstance(row, Mapping) or type(row.get("modelHash")) is not str
                   or not hashed(row.get("scope")) or row["scope"].get("rulesHash") != rules_hash for row in branches)
            or len({row["modelHash"] for row in branches}) != len(branches)
            or tuple(row["modelHash"] for row in branches) != source_model_hashes
            or not isinstance(bounds, list) or len(bounds) != 2
            or any(type(bound) is not int for bound in bounds) or bounds[0] > bounds[1]):
        fail()
    threshold = scope.get("incumbentThreshold") if threshold_mode else bounds[1] + 1
    if threshold_mode and (type(threshold) is not int or not bounds[0] <= threshold <= bounds[1]
            or strategy.get("status") != "scoped_incumbent_threshold_complete"
            or scope.get("kind") != "native_symbolic_round_policy_scope_v1"
            or scope.get("coverage") != "supplied_deal_branches_only"
            or scope.get("utility") != "awarded_point_margin"):
        fail()
    owned = [row for row in queries if row.get("purpose") == (
        "incumbent_threshold_floor" if threshold_mode else "incumbent_lower_bound")]
    if nonplay_policy and any(not isinstance(row.get("receipt"), Mapping)
            or not isinstance(row["receipt"].get("bindings"), list) for row in owned):
        fail()
    binding_count = sum(len(row.get("receipt", {}).get("bindings", ())) for row in owned) if nonplay_policy else 0
    if (queries != owned or type(strategy.get("queriesUsed")) is not int
            or strategy["queriesUsed"] != len(queries) + binding_count
            or type(strategy.get("bindingQueriesUsed")) is not int or strategy["bindingQueriesUsed"] != binding_count
            or strategy.get("masters") != []
            or [row.get("branchModelHash") for row in owned] != [row["modelHash"] for row in branches]):
        fail()
    floors = []
    for branch, wrapper in zip(branches, owned):
        receipt = wrapper.get("receipt")
        if (not hashed(receipt) or receipt.get("kind") != "native_constant_policy_round_query_v1"
                or type(receipt.get("schemaVersion")) is not int or receipt["schemaVersion"] != 1
                or receipt.get("strategicCoverage") is not False or receipt.get("status") != "queried"
                or receipt.get("sourceModelHash") != branch["modelHash"]
                or receipt.get("scope") != branch["scope"] or receipt.get("rulesHash") != scope.get("rulesHash")
                or receipt.get("targetTeam") != target_team or type(receipt.get("threshold")) is not int
                or receipt["threshold"] != threshold
                or nonplay_policy is None and receipt.get("bindings") != []):
            fail()
        policy, query = receipt.get("policy"), receipt.get("query")
        if not isinstance(policy, Mapping):
            fail()
        if (native_policy is not None
                and models_by_hash[branch["modelHash"]]._native_round_query_context()[0] != branch["scope"]):
            fail()
        no_play = branch["scope"].get("allPassedCovered") is True
        if no_play:
            if (not isinstance(policy, Mapping) or "nativePlayPolicyBinding" in policy
                    or "playPolicyModelHash" in policy or not isinstance(query, Mapping)
                    or query.get("baseModelHash") != (receipt.get("bindings", [{}])[-1].get("boundModelHash")
                        if nonplay_policy and receipt.get("bindings") else branch["modelHash"])
                    or query.get("candidate") is not None and (
                        not isinstance(query["candidate"], Mapping)
                        or query["candidate"].get("playObservations") != [])):
                fail()
        if native_policy is not None:
            binding = policy.get("nativePlayPolicyBinding") if isinstance(policy, Mapping) else None
            if not no_play and (not hashed(binding) or set(binding) != {"schemaVersion", "kind", "policy",
                    "sourceCpModelProtoHash", "boundCpModelProtoHash", "sha256"}
                    or type(binding.get("schemaVersion")) is not int or binding["schemaVersion"] != 1
                    or binding.get("kind") != "native_visible_play_policy_binding_v1"
                    or binding.get("policy") != native_policy
                    or any(type(binding.get(key)) is not str or not re.fullmatch(r"[0-9a-f]{64}", binding[key])
                           for key in ("sourceCpModelProtoHash", "boundCpModelProtoHash"))):
                fail()
            elif not no_play:
                source_model = models_by_hash[branch["modelHash"]]
                if ((source_model._model_hash(search_branching="automatic",
                        proto_hash=binding["sourceCpModelProtoHash"]) != receipt["sourceModelHash"] if nonplay_policy is None else
                        not hashed(policy.get("nativeProjectPolicyBinding"))
                        or set(policy["nativeProjectPolicyBinding"]) != {"schemaVersion", "kind", "policyId", "team",
                            "sourceCpModelProtoHash", "boundCpModelProtoHash", "sha256"}
                        or type(policy["nativeProjectPolicyBinding"].get("schemaVersion")) is not int
                        or policy["nativeProjectPolicyBinding"]["schemaVersion"] != 1
                        or policy["nativeProjectPolicyBinding"].get("kind") != "native_visible_project_policy_binding_v1"
                        or policy["nativeProjectPolicyBinding"].get("policyId") != nonplay_policy["projectPolicyId"]
                        or policy["nativeProjectPolicyBinding"].get("team") != target_team
                        or policy["nativeProjectPolicyBinding"].get("boundCpModelProtoHash") != binding["sourceCpModelProtoHash"])
                        or source_model._model_hash(search_branching="automatic",
                        proto_hash=binding["boundCpModelProtoHash"]) != policy.get("playPolicyModelHash")):
                    fail()
        excluded = (not threshold_mode and nonplay_policy is not None and isinstance(query, Mapping)
            and query.get("status") == "no_matching_outcome" and query.get("solverStatus") == "infeasible")
        if nonplay_policy is not None:
            bindings = receipt.get("bindings")
            expected = [("auction", nonplay_policy["auctionAction"])] + ([] if no_play else
                [("escalation", nonplay_policy["escalationAction"])])
            if (not isinstance(bindings, list) or len(bindings) != len(expected)
                    or policy.get("nativeProjectPolicyId") != nonplay_policy["projectPolicyId"]
                    or any(not hashed(row) or row.get("kind") != "native_constant_" + kind + "_policy_binding_v1"
                        or row.get("action") != action or row.get("status") != "bound"
                        or row.get("solverStatus") != "INFEASIBLE" or row.get("rulesHash") != rules_hash
                        or type(row.get("boundModelHash")) is not str or not re.fullmatch(r"[0-9a-f]{64}", row["boundModelHash"])
                        for row, (kind, action) in zip(bindings, expected))):
                fail()
            if not no_play and source_model._model_hash(search_branching="fixed",
                    proto_hash=policy["nativeProjectPolicyBinding"].get("sourceCpModelProtoHash")) != bindings[-1].get("modelHash"):
                fail()
            if not no_play and source_model._model_hash(search_branching="automatic",
                    proto_hash=policy["nativeProjectPolicyBinding"].get("sourceCpModelProtoHash")) != bindings[-1]["boundModelHash"]:
                fail()
        if (not isinstance(policy, Mapping) or policy.get("team") != target_team
                or not {"auctionAction", "escalationAction"} <= policy.keys()
                or policy["auctionAction"] != (nonplay_policy["auctionAction"] if nonplay_policy else None)
                or policy["escalationAction"] != (nonplay_policy["escalationAction"] if nonplay_policy else None)
                or nonplay_policy is None and any(key in policy for key in ("nativeProjectPolicyId", "nativeProjectPolicyBinding"))
                or any(key in policy for key in ("projectActions", "escalationObservations"))
                or (policy.get("scope") != "future_nonplay_constants_and_ranked_play"
                    or policy.get("playPolicies") != identities or "nativePlayPolicy" in policy
                    if native_policy is None else
                    policy.get("scope") != "future_nonplay_constants_and_native_visible_play"
                    or policy.get("nativePlayPolicy") != native_policy or "playPolicies" in policy)
                or not no_play and (type(policy.get("playPolicyModelHash")) is not str or not policy["playPolicyModelHash"])
                or not hashed(query) or query.get("kind") != "native_round_outcome_query_v1"
                or type(query.get("schemaVersion")) is not int or query["schemaVersion"] != 1
                or query.get("baseModelHash") != ((receipt["bindings"][-1]["boundModelHash"]
                    if nonplay_policy else branch["modelHash"]) if no_play else policy["playPolicyModelHash"])
                or type(query.get("queryModelHash")) is not str or not query["queryModelHash"]
                or query.get("scope") != branch["scope"] or query.get("targetTeam") != target_team
                or query.get("utility") != "awarded_point_margin"
                or (query.get("queryMode") != "feasibility_witness" if threshold_mode else "queryMode" in query)
                or type(query.get("threshold")) is not int or query["threshold"] != receipt["threshold"]
                or (query.get("status") != "no_matching_outcome" or query.get("solverStatus") != "infeasible"
                    if threshold_mode else not excluded and (query.get("status") != "witness_found"
                    or query.get("solverStatus") not in {"optimal", "feasible"}))
                or any(query.get(kind + "PolicyBindings") != [] for kind in ("auction", "escalation", "project", "play"))):
            fail()
        floor, candidate = query.get("objectiveLowerBound"), query.get("candidate")
        if excluded:
            if candidate is not None or floor is not None:
                fail()
            continue
        if threshold_mode:
            if candidate is not None or "objectiveLowerBound" in query or query.get("exhaustionReason") is not None:
                fail()
            floors.append(threshold)
            continue
        if (type(floor) is not int or not bounds[0] <= floor <= bounds[1]
                or not isinstance(candidate, Mapping) or type(candidate.get("utility")) is not int
                or not floor <= candidate["utility"] <= bounds[1]
                or query["solverStatus"] == "optimal" and floor != candidate["utility"]):
            fail()
        floors.append(floor)
    if not floors:
        fail()
    floor = min(floors)
    if type(strategy.get("lowerBound")) is not int or strategy["lowerBound"] != floor:
        fail()
    return floor


def validate_native_bid_dominance_claim_v1(value: Mapping[str, Any], *, legal_actions: tuple[str, ...],
        native_incumbent_models=None) -> None:
    """Check claim/receipt consistency after identity validation; not solver replay."""
    if (not isinstance(value, Mapping) or not isinstance(legal_actions, tuple)
            or any(type(action) is not str or not action for action in legal_actions)
            or value.get("targetTeam") not in ("north_south", "west_east")
            or not isinstance(value.get("qualificationModel"), Mapping)
            or not isinstance(value.get("alternatives"), list)
            or any(not isinstance(row, Mapping) for row in value["alternatives"])
            or not isinstance(value.get("selectedValueProof"), Mapping)
            or not isinstance(value["selectedValueProof"].get("strategy"), Mapping)
            or not isinstance(value.get("actionQuality"), Mapping)):
        raise ValueError("bid dominance claim shape is invalid")
    selected = value.get("selectedAction")
    actions = sorted(legal_actions)
    alternatives = value.get("alternatives", [])
    strategy = value.get("selectedValueProof", {}).get("strategy", {})
    resumed = value.get("kind") == "native_bid_selected_action_dominance_v4"
    threshold_mode = resumed or value.get("kind") == "native_bid_selected_action_dominance_v3"
    if strategy.get("kind") not in ({"native_symbolic_round_strategy_threshold_v1"} if resumed else
            {"native_incumbent_threshold_floor_v1"} if threshold_mode else
            {"native_symbolic_round_policy_bounds_v1", "native_symbolic_round_policy_bounds_v2"}):
        raise ValueError("bid dominance strategy evidence kind is unsupported")
    quality = value.get("actionQuality", {})
    interval = resumed or value.get("kind") in {"native_bid_selected_action_dominance_v2", "native_bid_selected_action_dominance_v3"}
    if resumed and any(key in value for key in ("incumbentNativePlayPolicyId", "incumbentNonplayPolicy", "incumbentPlayPolicies")):
        raise ValueError("resumed bid strategy must not claim a fixed incumbent policy")
    native_id = value.get("incumbentNativePlayPolicyId")
    nonplay = value.get("incumbentNonplayPolicy")
    if "incumbentNonplayPolicy" in value:
        from .scenario_contracts import validate_native_bid_nonplay_policy_v1
        nonplay = validate_native_bid_nonplay_policy_v1(nonplay)
        if native_id is None or not interval or value.get("visibility") != "complete_state":
            raise ValueError("nonplay bid incumbent has no complete-state native owner")
    if "incumbentNativePlayPolicyId" in value and (
            not interval or type(native_id) is not str or "incumbentPlayPolicies" in value):
        raise ValueError("bid dominance incumbent must have one policy authority")
    if "opponentPlayPolicyId" in value and (not interval or type(value["opponentPlayPolicyId"]) is not str):
        raise ValueError("bid dominance opponent play policy requires an interval owner")
    if native_incumbent_models is not None and native_id is None:
        raise ValueError("bid dominance native models have no policy owner")
    score = strategy.get("lowerBound" if interval else "upperBound")
    expected_status = ("scoped_strategy_threshold_complete" if resumed else "scoped_incumbent_threshold_complete" if threshold_mode else
                       "scoped_incumbent_bound_complete" if interval else "scoped_bound_converged")
    if interval:
        if (strategy.get("kind") != ("native_symbolic_round_strategy_threshold_v1" if resumed else "native_incumbent_threshold_floor_v1" if threshold_mode else
                                     "native_symbolic_round_policy_bounds_v2")
                or value["selectedValueProof"].get("status") != expected_status
                or type(strategy.get("upperBound")) is not int
                or type(score) is not int or score > strategy["upperBound"]
                or not {"actualValue", "bestValue"} <= quality.keys()
                or quality["actualValue"] is not None or quality["bestValue"] is not None
                or quality.get("selectedValueBounds") != {"lower": score, "upper": strategy["upperBound"]}):
            raise ValueError("bid dominance interval values must remain explicit bounds")
    if (not actions or len(set(actions)) != len(actions) or selected not in actions
            or value.get("legalActions") != actions
            or value.get("status") != "selected_action_optimal"
            or value.get("qualificationModel", {}).get("solverStatus") != "optimal"
            or strategy.get("status") != expected_status
            or type(score) is not int or type(strategy.get("lowerBound")) is not int
            or strategy["lowerBound"] != score
            or [row.get("action") for row in alternatives] != [action for action in actions if action != selected]
            or value.get("optimalActionCount") is not None
            or quality.get("optimalActionCount") is not None
            or quality.get("optimalActionsComplete") is not False
            or quality.get("optimalActions") != [selected]
            or quality.get("actualAction") != selected
            or quality.get("actualActionOptimal") is not True
            or not interval and (type(quality.get("actualValue")) is not int or type(quality.get("bestValue")) is not int
                or quality.get("actualValue") != score or quality.get("bestValue") != score)
            or type(quality.get("regret")) is not int or quality["regret"] != 0):
        raise ValueError("bid dominance claim coverage or value mismatch")
    for document in (value, quality):
        if (type(document.get("optimalActionCountLowerBound")) is not int
                or document["optimalActionCountLowerBound"] != 1
                or type(document.get("optimalActionCountUpperBound")) is not int
                or document["optimalActionCountUpperBound"] != len(actions)):
            raise ValueError("bid dominance optimal-action bounds mismatch")

    def retained_models(preparation):
        if not isinstance(preparation, Mapping) or preparation.get("constructionComplete") is not True:
            raise ValueError("bid dominance preparation incomplete")
        branches = preparation.get("branches")
        if not isinstance(branches, list) or not branches or any(not isinstance(row, Mapping) for row in branches):
            raise ValueError("bid dominance preparation branches missing")
        hashes = [row.get("sourceModelHash") for row in branches if row.get("retained") is True]
        if not hashes or any(type(item) is not str for item in hashes) or len(set(hashes)) != len(hashes):
            raise ValueError("bid dominance preparation models invalid")
        return hashes

    selected_models = retained_models(value["selectedValueProof"].get("preparation"))
    if not isinstance(strategy.get("scope"), Mapping):
        raise ValueError("bid dominance selected strategy scope missing")
    strategy_branches = strategy["scope"].get("branches", [])
    if (not isinstance(strategy_branches, list) or any(not isinstance(row, Mapping) for row in strategy_branches)
            or any(type(row.get("modelHash")) is not str for row in strategy_branches)
            or sorted(selected_models) != sorted(row["modelHash"] for row in strategy_branches)):
        raise ValueError("bid dominance selected strategy model coverage mismatch")
    if interval:
        counts = (value["selectedValueProof"].get("queriesUsed"),
            value["selectedValueProof"].get("preparation", {}).get("queriesUsed"), strategy.get("queriesUsed"))
        if any(type(count) is not int or count < 0 for count in counts) or counts[0] != counts[1] + counts[2]:
            raise ValueError("bid dominance incumbent query accounting mismatch")
        if resumed:
            validate_native_strategy_threshold_v1(strategy, expected_scope={**strategy["scope"],
                "rulesHash": value["rulesHash"], "targetTeam": value["targetTeam"],
                "utility": "awarded_point_margin", "requiredThreshold": score})
        else:
            validate_floor = (validate_native_incumbent_threshold_floor_v1 if threshold_mode else
                              validate_native_incumbent_lower_bound_v2)
            validate_floor(strategy, target_team=value["targetTeam"],
                play_policies=value.get("incumbentPlayPolicies"), rules_hash=value.get("rulesHash"),
                **({"native_play_policy_id": native_id, "source_models": native_incumbent_models}
                   if native_id is not None else {}),
                **({"nonplay_policy": nonplay} if nonplay is not None else {}),
                source_model_hashes=tuple(sorted(selected_models)))
        if strategy["upperBound"] != strategy["scope"]["utilityBounds"][1]:
            raise ValueError("bid dominance selected upper bound differs from the native domain")
        if threshold_mode and (not alternatives
                or any(type(row.get("upperBound")) is not int for row in alternatives)
                or score != max(row["upperBound"] for row in alternatives)):
            raise ValueError("bid dominance threshold differs from the alternative bounds")
    elif strategy["kind"] == "native_symbolic_round_policy_bounds_v2":
        try:
            _validate_native_final_branch_proofs_v2(strategy, target_team=value["targetTeam"])
        except ValueError as exc:
            raise ValueError("bid dominance final branch proof is invalid") from exc
    opponent = Seat.parse("north" if value.get("targetTeam") == "north_south" else "west").team.other.name.lower()
    ceiling_policy = strategy["scope"].get("opponentCeilingPolicy")
    if ceiling_policy is not None:
        if (not isinstance(ceiling_policy, Mapping)
                or set(ceiling_policy) not in ({"auction_action", "escalation_action"},
                    {"auction_action", "escalation_action", "native_play_policy_id"})
                or any(action is not None and type(action) is not str for action in ceiling_policy.values())
                or not isinstance(strategy.get("queries"), list)
                or any(not isinstance(row, Mapping) for row in strategy["queries"])):
            raise ValueError("bid dominance opponent ceiling shape is invalid")
        ceilings = [row for row in strategy.get("queries", [])
            if row.get("purpose") == "opponent_policy_upper_bound"]
        if len(selected_models) != 1 or len(ceilings) != 1:
            raise ValueError("bid dominance opponent ceiling requires one owned branch")
        row = ceilings[0]
        query = row.get("receipt", {})
        bounds = strategy["scope"].get("utilityBounds")
        if (not isinstance(query, Mapping) or not isinstance(query.get("query"), Mapping)
                or not isinstance(query["query"].get("candidate"), Mapping)
                or not isinstance(query.get("policy"), Mapping)
                or not isinstance(query.get("bindings"), list)
                or any(not isinstance(binding, Mapping) or type(binding.get("kind")) is not str
                    or type(binding.get("action")) is not str for binding in query["bindings"])
                or not isinstance(bounds, list) or len(bounds) != 2
                or any(type(bound) is not int for bound in bounds)):
            raise ValueError("bid dominance opponent ceiling receipt shape is invalid")
        outcome = query.get("query", {})
        candidate = outcome.get("candidate") or {}
        query_scope = query.get("scope")
        no_play = isinstance(query_scope, Mapping) and query_scope.get("allPassedCovered") is True
        if no_play and (query_scope.get("kind") != "native_all_pass_deal_branch_v1"
                or query_scope.get("rulesHash") != value.get("rulesHash")
                or query_scope.get("sha256") != canonical_sha256({key: item for key, item in query_scope.items() if key != "sha256"})
                or outcome.get("scope") != query_scope
                or query_scope.get("continuationBoundary") != "current_deal_completion"
                or query_scope.get("nextDealAnalyzed") is not False
                or candidate.get("allPassed") is not True or candidate.get("utility") != 0
                or candidate.get("awardedPoints") != [0, 0]
                or any(candidate.get(key) != [] for key in (
                    "escalationObservations", "projectObservations", "playObservations"))):
            raise ValueError("bid dominance all-pass ceiling is not owned")
        expected_bindings = sorted(("native_constant_" + kind + "_policy_binding_v1", action)
            for kind, key in (("auction", "auction_action"), ("escalation", "escalation_action"))
            if (action := ceiling_policy.get(key)) is not None and not (no_play and kind == "escalation"))
        def hashed(document):
            return isinstance(document, Mapping) and document.get("sha256") == canonical_sha256(
                {key: item for key, item in document.items() if key != "sha256"})
        native_id = ceiling_policy.get("native_play_policy_id")
        if "native_play_policy_id" in ceiling_policy:
            from .ortools_play_tail import native_observer_safe_policy_identity_v1

            try:
                native_identity = native_observer_safe_policy_identity_v1(
                    native_id, Seat.parse("north" if opponent == "north_south" else "west").team)
            except (TypeError, ValueError) as exc:
                raise ValueError("bid dominance opponent ceiling play policy is unregistered") from exc
            policy = query.get("policy")
            binding = policy.get("nativePlayPolicyBinding") if isinstance(policy, Mapping) else None
            if (not isinstance(policy, Mapping) or policy.get("scope") != "future_nonplay_constants_and_native_visible_play"
                    or policy.get("nativePlayPolicy") != native_identity or "playPolicies" in policy
                    or type(policy.get("playPrefixLength")) is not int or policy["playPrefixLength"] < 0
                    or query.get("rulesHash") != value.get("rulesHash")
                    or not isinstance(query_scope, Mapping) or not isinstance(outcome, Mapping)
                    or outcome.get("scope") != query_scope):
                raise ValueError("bid dominance opponent ceiling play policy is not owned")
            if no_play:
                if ("nativePlayPolicyBinding" in policy or "playPolicyModelHash" in policy
                        or outcome.get("baseModelHash") != selected_models[0]
                        or isinstance(candidate, Mapping) and candidate.get("playObservations") != []):
                    raise ValueError("bid dominance all-pass ceiling contains play evidence")
            elif (not hashed(binding) or set(binding) != {"schemaVersion", "kind", "policy",
                    "sourceCpModelProtoHash", "boundCpModelProtoHash", "sha256"}
                    or type(binding.get("schemaVersion")) is not int or binding["schemaVersion"] != 1
                    or binding.get("kind") != "native_visible_play_policy_binding_v1"
                    or binding.get("policy") != native_identity
                    or any(type(binding.get(key)) is not str or _HASH.fullmatch(binding[key]) is None
                           for key in ("sourceCpModelProtoHash", "boundCpModelProtoHash"))
                    or binding["sourceCpModelProtoHash"] == binding["boundCpModelProtoHash"]
                    or type(policy.get("playPolicyModelHash")) is not str
                    or _HASH.fullmatch(policy["playPolicyModelHash"]) is None
                    or outcome.get("baseModelHash") != policy["playPolicyModelHash"]):
                raise ValueError("bid dominance opponent ceiling play binding is not certified")
        if (row.get("branchModelHash") != selected_models[0]
                or query.get("sourceModelHash") != selected_models[0]
                or not hashed(query) or not hashed(outcome)
                or query.get("status") != "queried" or query.get("targetTeam") != opponent
                or query.get("policy", {}).get("team") != opponent
                or query.get("policy", {}).get("auctionAction") != ceiling_policy.get("auction_action")
                or query.get("policy", {}).get("escalationAction") != ceiling_policy.get("escalation_action")
                or outcome.get("targetTeam") != opponent or outcome.get("solverStatus") != "optimal"
                or outcome.get("queryMode") == "feasibility_witness"
                or type(candidate.get("utility")) is not int
                or strategy["upperBound"] > -candidate["utility"]
                or not isinstance(query.get("bindings"), list)
                or sorted((binding.get("kind"), binding.get("action")) for binding in query["bindings"]) != expected_bindings
                or any(not hashed(binding) or binding.get("status") != "bound"
                    for binding in query["bindings"])):
            raise ValueError("bid dominance opponent ceiling is not certified")
    for row in alternatives:
        queries = row.get("queries", [])
        if (type(row.get("upperBound")) is not int or row["upperBound"] > score
                or not isinstance(queries, list) or not queries
                or any(not isinstance(query, Mapping) for query in queries)
                or retained_models(row.get("preparation")) != row.get("sourceModelHashes")
                or row.get("sourceModelHashes") != [query.get("sourceModelHash") for query in queries]):
            raise ValueError("bid dominance alternative bound coverage mismatch")
        if row["preparation"].get("kind") == "native_bid_streamed_world_upper_v1":
            if not interval or not threshold_mode and "opponentPlayPolicyId" not in value:
                raise ValueError("bid dominance world upper requires a paired policy owner")
            _validate_streamed_bid_preparation(row["preparation"], row,
                maximum_model_bytes=value.get("limits", {}).get("maximumModelBytesPerPreparation"), world_upper=True)
            outcomes = [query.get("query", {}) for query in queries]
            if (any(outcome.get("status") != "witness_found" or outcome.get("solverStatus") != "optimal"
                    or not isinstance(outcome.get("candidate"), Mapping)
                    or type(outcome.get("candidate", {}).get("utility")) is not int for outcome in outcomes)
                    or row["upperBound"] != max(-outcome["candidate"]["utility"] for outcome in outcomes)):
                raise ValueError("bid dominance alternative world upper is not certified")
            continue
        for query in queries:
            outcome = query.get("query", {})
            if (not isinstance(outcome, Mapping) or not isinstance(query.get("bindings"), list)
                    or any(not isinstance(binding, Mapping) for binding in query["bindings"])
                    or query.get("status") != "queried" or query.get("targetTeam") != opponent
                    or type(query.get("threshold")) is not int or type(outcome.get("threshold")) is not int
                    or query.get("threshold") != -row["upperBound"]
                    or outcome.get("status") != "no_matching_outcome"
                    or outcome.get("solverStatus") != "infeasible"
                    or outcome.get("targetTeam") != opponent or outcome.get("threshold") != -row["upperBound"]
                    or any(binding.get("status") != "bound" for binding in query.get("bindings", []))):
                raise ValueError("bid dominance alternative bound is not certified")
    from .ortools_information_set_proof import build_native_bid_dominance_model_v1
    model, selected_variable, claim, best, regret, bindings = build_native_bid_dominance_model_v1(score, alternatives)
    expected_model = {"modelProto": str(model.proto), "selectedValueVariable": selected_variable.name,
        "claimVariable": claim.name, "claimLiteralIndex": int(claim.index),
        "bestValueVariable": best.name, "regretVariable": regret.name, "alternatives": bindings}
    if any(value["qualificationModel"].get(key) != item for key, item in expected_model.items()):
        raise ValueError("bid dominance qualification model or variable binding mismatch")


def _native_bid_play_binding_matches_v1(branch, policy, purchased):
    binding = branch.get("playPolicyBinding")
    if policy is None or not purchased or not branch.get("retained"):
        return binding is None
    return (isinstance(binding, Mapping) and set(binding) == {"schemaVersion", "kind", "policy",
            "sourceCpModelProtoHash", "boundCpModelProtoHash", "sha256"}
        and binding.get("sha256") == canonical_sha256({k: v for k, v in binding.items() if k != "sha256"})
        and type(binding["schemaVersion"]) is int and binding["schemaVersion"] == 1
        and binding["kind"] == "native_visible_play_policy_binding_v1" and binding["policy"] == policy
        and all(type(binding[key]) is str and _HASH.fullmatch(binding[key]) is not None
                for key in ("sourceCpModelProtoHash", "boundCpModelProtoHash"))
        and binding["sourceCpModelProtoHash"] != binding["boundCpModelProtoHash"]
        and binding["boundCpModelProtoHash"] == branch.get("cpModelProtoHash"))


def _validate_streamed_bid_preparation(receipt, alternative, *, maximum_model_bytes, world_upper=False):
    """Reconcile streamed evidence without charging its proof queries twice."""
    def fail():
        raise ValueError("streamed bid preparation ownership or resource accounting mismatch")

    def hashed(document):
        return isinstance(document, Mapping) and document.get("sha256") == canonical_sha256(
            {key: item for key, item in document.items() if key != "sha256"})

    def size(document, key):
        value = document.get(key) if isinstance(document, Mapping) else None
        if type(value) is not int or value < 0:
            fail()
        return value

    if (not hashed(receipt) or receipt.get("constructionComplete") is not True
            or receipt.get("kind") != ("native_bid_streamed_world_upper_v1" if world_upper
                else "native_bid_streamed_alternative_preparation_v1")
            or receipt.get("modelByteUnit") != "utf8_cp_model_proto"
            or receipt.get("includesSolverWorkingMemory") is not False
            or type(receipt.get("upperBound" if world_upper else "alternativeUpperBound")) is not int
            or receipt["upperBound" if world_upper else "alternativeUpperBound"] != alternative.get("upperBound")
            or not isinstance(receipt.get("branches"), list)
            or not isinstance(receipt.get("limits"), Mapping)
            or type(maximum_model_bytes) is not int or maximum_model_bytes < 1):
        fail()
    phases, queries, proofs, constructed, owned = [0], 0, 0, 0, []
    for branch in receipt["branches"]:
        if not isinstance(branch, Mapping):
            fail()
        reach = branch.get("reachability")
        if not isinstance(reach, Mapping) or not isinstance(reach.get("policyBindings"), list):
            fail()
        queries += 1 + len(reach["policyBindings"])
        phases.append(size(reach, "modelBytes"))
        for binding in reach["policyBindings"]:
            if not hashed(binding) or binding.get("status") != "bound":
                fail()
            sizes = binding.get("modelBytes")
            phases.append(size(sizes, "owner") + size(sizes, "legalityProbe"))
        if not branch.get("retained"):
            if "alternativeQuery" in branch or size(branch, "modelBytes") != 0:
                fail()
            continue
        source_bytes = size(branch, "modelBytes")
        constructed += source_bytes
        phases.append(source_bytes)
        depth = branch.get("depthQuery")
        if depth is not None:
            queries += 1
            sizes = depth.get("modelBytes") if isinstance(depth, Mapping) else None
            if size(sizes, "owner") != source_bytes:
                fail()
            phases.append(source_bytes + size(sizes, "depthQuery"))
        query = branch.get("alternativeQuery")
        if (not hashed(query) or query.get("sourceModelHash") != branch.get("sourceModelHash")
                or not isinstance(query.get("bindings"), list) or not hashed(query.get("query"))):
            fail()
        owned.append(query)
        used = len(query["bindings"]) + 1
        proofs += used
        queries += used
        sizes = query.get("modelBytes")
        if (size(sizes, "source") != source_bytes or sizes.get("unit") != "utf8_cp_model_proto"
                or sizes.get("includesSolverWorkingMemory") is not False):
            fail()
        copies = [2 * source_bytes]
        for binding in query["bindings"]:
            if not hashed(binding):
                fail()
            bound_sizes = binding.get("modelBytes")
            copies.append(source_bytes + size(bound_sizes, "owner") + size(bound_sizes, "legalityProbe"))
        outcome_sizes = query["query"].get("modelBytes")
        copies.append(source_bytes + size(outcome_sizes, "owner") + size(outcome_sizes, "outcomeQuery"))
        if size(sizes, "retainedUpperBound") != max(copies):
            fail()
        phases.append(max(copies))
    limits = receipt["limits"]
    if (not owned or owned != alternative.get("queries")
            or size(receipt, "modelBytes") != constructed
            or size(receipt, "retainedModelBytes") != max(phases)
            or size(receipt, "proofQueriesUsed") != proofs or size(receipt, "queriesUsed") != queries
            or not 0 < size(limits, "queries") or queries > limits["queries"]
            or not 0 < size(limits, "modelBytes") <= maximum_model_bytes
            or max(phases) > limits["modelBytes"]):
        fail()


def _prior_bid_alternative_receipts_v1(prior, *, source_prefix_hash, rules_hash, maximum_seconds):
    """Execution-local proposals; every reused premise still needs fresh replay."""
    retained = {}
    for row in prior or ():
        parent = row.get("receipt", {}).get("evidence", {}).get("nativeObservedBidDominance", {})
        if (parent.get("kind") != "native_observed_bid_dominance_attempt_v1"
                or parent.get("status") != "incomplete"
                or parent.get("sourcePrefixHash") != source_prefix_hash
                or parent.get("rulesHash") != rules_hash
                or parent.get("sha256") != canonical_sha256({k: v for k, v in parent.items() if k != "sha256"})
                or type(parent.get("maximumSeconds")) not in {int, float}
                or not 0 < parent["maximumSeconds"] <= maximum_seconds):
            continue
        for alternative in parent.get("alternatives", ()):
            receipt = alternative.get("receipt", {})
            if (receipt.get("kind") == "native_bid_streamed_world_upper_v1"
                    and receipt.get("constructionComplete") is True
                    and receipt.get("sha256") == canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})):
                retained[receipt["sha256"]] = (alternative["action"], receipt, parent["maximumSeconds"])
    return retained


def validate_native_bid_world_upper_v1(receipt, *, rules, exact, source_prefix,
        action, opponent_policy, maximum_model_bytes, opportunity_order=None, opponent_play_policy_id=None,
        maximum_depth=None) -> int:
    """Check world/branch/claim ownership; admission still requires model replay."""
    from .ortools_exact_state import OrToolsNormalizedExactStateV1, _effective_prefix_hash
    from .ortools_information_set_proof import native_bid_branch_specs_v1
    from baloot_platform.core.exact_state_view import canonical_observer_information_hash_v1

    def fail():
        raise ValueError("native bid world upper ownership or coverage mismatch")

    def hashed(document):
        return isinstance(document, Mapping) and document.get("sha256") == canonical_sha256(
            {key: value for key, value in document.items() if key != "sha256"})

    if not isinstance(exact, OrToolsNormalizedExactStateV1) or not isinstance(source_prefix, ExactTracePrefixV1):
        fail()
    if maximum_depth is not None and (type(maximum_depth) is not int or maximum_depth < 1):
        fail()
    exact.require_integrity()
    if (exact.decision_kind != "bid" or exact.rules_hash != rules.sha256
            or exact.prefix_hash != _effective_prefix_hash(source_prefix)
            or type(action) is not str or action not in {row["code"] for row in exact.legal_actions}
            or not isinstance(opponent_policy, Mapping)
            or set(opponent_policy) != {"auction_action", "escalation_action"}
            or any(value is not None and type(value) is not str for value in opponent_policy.values())):
        fail()
    actor = Seat.parse(exact.next_actor)
    play_policy = None
    if opponent_play_policy_id is not None:
        from .ortools_play_tail import native_observer_safe_policy_identity_v1
        play_policy = native_observer_safe_policy_identity_v1(opponent_play_policy_id, actor.team.other)
    if opportunity_order is not None:
        if (not isinstance(opportunity_order, (tuple, list)) or len(opportunity_order) != len(Seat)
                or any(type(seat) not in {Seat, str, int} for seat in opportunity_order)):
            fail()
        opportunity_order = tuple(Seat.parse(seat).name.lower() for seat in opportunity_order)
        if len(set(opportunity_order)) != len(Seat):
            fail()
    specs = native_bid_branch_specs_v1(rules, Seat.parse(source_prefix.dealer))
    partition = [None if row is None else {"contract": row[0].value,
        "trump": row[1].name.lower() if row[1] is not None else None,
        "firstLeader": row[2].name.lower(), "locked": row[3]} for row in specs]
    expected_scope = {"rulesHash": rules.sha256, "visibility": "complete_state", "rootHash": exact.state_hash,
        "matchScore": list(source_prefix.match_score), "continuationBoundary": "current_deal_completion",
        "nextDealAnalyzed": False, "selectedRootAction": action,
        "nonplayPolicy": {"team": actor.team.other.name.lower(), **opponent_policy}, "partition": partition,
        **({"opponentPlayPolicy": play_policy} if play_policy is not None else {}),
        **({"escalationOpportunityOrder": list(opportunity_order)} if opportunity_order is not None else {})}
    receipt_scope = receipt.get("scope") if isinstance(receipt, Mapping) else None
    if not isinstance(receipt_scope, Mapping):
        fail()
    compaction = receipt_scope.get("ownershipCompaction")
    if "ownershipCompaction" in receipt_scope:
        if compaction != "purchase_recipient_invariant_v1":
            fail()
        expected_scope["ownershipCompaction"] = compaction
    if (not hashed(receipt) or type(receipt.get("schemaVersion")) is not int or receipt["schemaVersion"] != 1
            or receipt.get("strategicCoverage") is not False
            or receipt.get("worldStateHash") != exact.state_hash or receipt.get("worldExactStateId") != exact.exact_state_id
            or receipt.get("worldPrefixHash") != source_prefix.sha256 or receipt.get("worldEffectivePrefixHash") != exact.prefix_hash
            or receipt.get("observerRootHash") != canonical_observer_information_hash_v1(
                complete_state=exact.state, acting_seat=actor.name.lower())
            or receipt.get("targetTeam") != actor.team.name.lower()
            or receipt.get("targetDecisionScope") != "complete_information_cooperative_relaxation"
            or receipt.get("scope") != expected_scope or receipt.get("scopeHash") != canonical_sha256(expected_scope)
            or "alternativeUpperBound" in receipt):
        fail()
    branches = receipt.get("branches")
    if (not isinstance(branches, list) or len(branches) != len(partition)
            or any(not isinstance(row, Mapping) or type(row.get("partitionIndex")) is not int
                or row["partitionIndex"] != index or type(row.get("retained")) is not bool
                for index, row in enumerate(branches))):
        fail()
    upper_values, queries = [], []
    for branch, cell in zip(branches, partition):
        if maximum_depth is not None and branch["retained"]:
            depth = branch.get("depthQuery")
            if (receipt.get("limits", {}).get("maximumDepth") != maximum_depth or not hashed(depth)
                    or depth.get("kind") != "native_continuation_depth_query_v1"
                    or depth.get("status") != "depth_bound_proven" or depth.get("solverStatus") != "infeasible"
                    or type(depth.get("maximumDepth")) is not int or depth["maximumDepth"] != maximum_depth
                    or depth.get("sourceModelHash") != branch.get("sourceModelHash")
                    or depth.get("unit") != "remaining_decision_actions"
                    or depth.get("includes") != ["auction", "escalation", "project", "play"]):
                fail()
        if not _native_bid_play_binding_matches_v1(branch, play_policy, cell is not None):
            fail()
        reach = branch.get("reachability")
        expected_bindings = sorted(("native_constant_" + kind + "_policy_binding_v1", selected)
            for kind in ("auction", "escalation")
            if (selected := opponent_policy[kind + "_action"]) is not None and (kind != "escalation" or cell is not None))
        def legal_bindings(bindings, context):
            return (isinstance(bindings, list) and all(hashed(row) and row.get("status") == "bound"
                    and row.get("rulesHash") == rules.sha256 and row.get("solverStatus") == "INFEASIBLE"
                    and type(row.get("boundModelHash")) is str and row["boundModelHash"]
                    and row.get("observationHash") == canonical_sha256({**context,
                        "stage": "auction" if row.get("kind") == "native_constant_auction_policy_binding_v1" else "escalation"})
                    for row in bindings)
                and [(row.get("kind"), row.get("action")) for row in bindings] == expected_bindings)
        if (not hashed(reach) or reach.get("scope") != {"preparationScopeHash": canonical_sha256(expected_scope),
                "partitionIndex": branch["partitionIndex"], "partition": cell}
                or not legal_bindings(reach.get("policyBindings"), {"scope": expected_scope})
                or (reach["policyBindings"] and reach.get("boundModelHash") != reach["policyBindings"][-1]["boundModelHash"])):
            fail()
        if not branch["retained"]:
            if reach.get("solverStatus") != "infeasible":
                fail()
            continue
        if reach.get("solverStatus") not in {"optimal", "feasible"}:
            fail()
        query = branch.get("alternativeQuery")
        outcome = query.get("query") if isinstance(query, Mapping) else None
        expected_policy = {"team": actor.team.other.name.lower(), "auctionAction": opponent_policy["auction_action"],
            "escalationAction": opponent_policy["escalation_action"], "auctionPrefixLength": len(source_prefix.auction_actions),
            "scope": "all_future_nonplay_information_sets"}
        if (not hashed(query) or not hashed(outcome) or query.get("status") != "queried"
                or query.get("sourceModelHash") != branch.get("sourceModelHash")
                or query.get("policy") != expected_policy or query.get("rulesHash") != rules.sha256
                or query.get("targetTeam") != actor.team.other.name.lower()
                or not legal_bindings(query.get("bindings"), {"policy": expected_policy})
                or not hashed(query.get("scope")) or outcome.get("scope") != query["scope"]
                or outcome.get("targetTeam") != actor.team.other.name.lower()
                or outcome.get("utility") != "awarded_point_margin" or "queryMode" in outcome
                or outcome.get("status") != "witness_found" or outcome.get("solverStatus") != "optimal"
                or outcome.get("exhaustionReason") is not None
                or type(outcome.get("threshold")) is not int or outcome["threshold"] != query.get("threshold")
                or not isinstance(outcome.get("candidate"), Mapping)):
            fail()
        if outcome.get("baseModelHash") != (query["bindings"][-1]["boundModelHash"]
                if query["bindings"] else query["sourceModelHash"]):
            fail()
        branch_scope = {"rulesHash": rules.sha256, "matchScore": list(source_prefix.match_score),
            "allPassedCovered": cell is None, "strategicCoverage": False,
            **({"kind": "native_all_pass_deal_branch_v1", "continuationBoundary": "current_deal_completion",
                "nextDealAnalyzed": False} if cell is None else {
                "kind": "native_purchased_round_branch_v1", "contract": cell["contract"],
                "trumpSuit": cell["trump"], "firstLeader": cell["firstLeader"], "locked": cell["locked"],
                "escalationOpportunityOrder": list(opportunity_order) if opportunity_order is not None else "symbolic_per_window"})}
        if any(query["scope"].get(key) != value for key, value in branch_scope.items()):
            fail()
        utility = outcome["candidate"].get("utility")
        if (type(utility) is not int or type(outcome.get("objectiveLowerBound")) is not int
                or outcome["objectiveLowerBound"] != utility or type(branch.get("alternativeUpperBound")) is not int
                or branch["alternativeUpperBound"] != -utility):
            fail()
        upper_values.append(-utility)
        queries.append(query)
    if not upper_values or type(receipt.get("upperBound")) is not int or receipt["upperBound"] != max(upper_values):
        fail()
    _validate_streamed_bid_preparation(receipt, {"upperBound": receipt["upperBound"], "queries": queries},
        maximum_model_bytes=maximum_model_bytes, world_upper=True)
    return receipt["upperBound"]


def reverify_native_bid_world_upper_v1(receipt, *, rules, exact, source_prefix,
        action, opponent_policy, maximum_model_bytes, maximum_branches,
        maximum_conflicts, maximum_seconds, maximum_queries, maximum_depth=None,
        opportunity_order=None, cancel_check=None, _on_queries_used=None,
        opponent_play_policy_id=None) -> int:
    """Recompute a retained world cap; never establish observer-wide optimality."""
    from .ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1

    if _on_queries_used is not None and not callable(_on_queries_used):
        raise TypeError("query accounting callback must be callable")

    owner = dict(rules=rules, exact=exact, source_prefix=source_prefix, action=action,
        opponent_policy=opponent_policy, maximum_model_bytes=maximum_model_bytes,
        opportunity_order=opportunity_order, maximum_depth=maximum_depth,
        opponent_play_policy_id=opponent_play_policy_id)
    threshold_only = isinstance(receipt, Mapping) and receipt.get("kind") == "native_bid_streamed_alternative_preparation_v1"
    if threshold_only:
        from .ortools_play_tail import native_observer_safe_policy_identity_v1
        if opponent_play_policy_id is not None:
            from .ortools_exact_state import OrToolsNormalizedExactStateV1
            if not isinstance(exact, OrToolsNormalizedExactStateV1):
                raise ValueError("native bid opponent play policy requires an owned exact state")
        identity = (native_observer_safe_policy_identity_v1(opponent_play_policy_id,
            Seat.parse(exact.next_actor).team.other) if opponent_play_policy_id is not None else None)
        if receipt.get("scope", {}).get("opponentPlayPolicy") != identity:
            raise ValueError("threshold-only bid opponent play policy substitution")
        if (type(receipt.get("schemaVersion")) is not int or receipt["schemaVersion"] != 1
                or receipt.get("strategicCoverage") is not False
                or not isinstance(receipt.get("branches"), list)
                or any(not isinstance(row, Mapping) for row in receipt["branches"])):
            raise ValueError("native bid threshold upper schema mismatch")
        retained = receipt.get("alternativeUpperBound")
        queries = [row.get("alternativeQuery") for row in receipt.get("branches", []) if row.get("retained")]
        _validate_streamed_bid_preparation(receipt, {"upperBound": retained, "queries": queries},
            maximum_model_bytes=maximum_model_bytes)
    else:
        retained = validate_native_bid_world_upper_v1(receipt, **owner)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    models, rebuilt = compiler._prepare_native_bid_branches(exact, source_prefix,
        visibility="complete_state", selected_root_action=action,
        policy_team=Seat.parse(exact.next_actor).team.other,
        nonplay_policy=opponent_policy, maximum_branches=maximum_branches,
        maximum_model_bytes=maximum_model_bytes, maximum_conflicts=maximum_conflicts,
        maximum_seconds=maximum_seconds, maximum_queries=maximum_queries,
        maximum_depth=maximum_depth, opportunity_order=opportunity_order,
        **({"compact_purchase_invariant_ownership": True}
           if receipt.get("scope", {}).get("ownershipCompaction") == "purchase_recipient_invariant_v1" else {}),
        **({"opponent_play_policy_id": opponent_play_policy_id} if opponent_play_policy_id is not None else {}),
        cancel_check=cancel_check, **({"alternative_upper_bound": retained} if threshold_only
                                    else {"optimize_alternative_upper": True}))
    if threshold_only:
        def semantic(document):
            branches = copy.deepcopy(document["branches"])
            for branch in branches:
                reach = branch.get("reachability", {})
                reach.pop("branches", None)
                reach.pop("conflicts", None)
                depth = branch.get("depthQuery")
                if isinstance(depth, dict):
                    if depth.get("sha256") != canonical_sha256({k: v for k, v in depth.items() if k != "sha256"}):
                        raise ValueError("native bid threshold depth integrity mismatch")
                    depth.pop("sha256")
                    depth.pop("maximumSeconds", None)
            return {"scope": document["scope"], "branches": branches,
                "alternativeUpperBound": document["alternativeUpperBound"]}
        if models or canonical_sha256(semantic(receipt)) != canonical_sha256(semantic(rebuilt)):
            raise ValueError("native bid threshold upper differs from reconstructed universe")
    elif models or validate_native_bid_world_upper_v1(rebuilt, **owner) != retained:
        raise ValueError("native bid world upper failed fresh verification")
    # Comparing each cell also detects substitutions hidden by the aggregate max.
    for old, fresh in zip(receipt["branches"], rebuilt["branches"], strict=True):
        if any(old.get(key) != fresh.get(key) for key in (
                "partitionIndex", "retained", "sourceModelHash", "cpModelProtoHash",
                "alternativeUpperBound")):
            raise ValueError("native bid world upper branch differs from reconstructed universe")
    if _on_queries_used is not None:
        _on_queries_used(rebuilt["queriesUsed"])
    return retained


def validate_native_bid_threshold_refutation_v1(receipt, *, observer_exact,
        required_threshold: int, **world_owner) -> int:
    """Read a certified counterexample-world cap, not an exact action value.

    Even a fully informed target cannot exceed this cap against the owned
    opponent policy. The same visible root therefore cannot guarantee a higher
    floor. Validity still relies on replay of the owned native models.
    """
    from .ortools_exact_state import OrToolsNormalizedExactStateV1
    from baloot_platform.core.exact_state_view import canonical_observer_information_hash_v1

    if type(required_threshold) is not int or not isinstance(observer_exact, OrToolsNormalizedExactStateV1):
        raise ValueError("native bid threshold refutation requires an owned observer and integer threshold")
    observer_exact.require_integrity()
    upper = validate_native_bid_world_upper_v1(receipt, **world_owner)
    if (observer_exact.decision_kind != "bid" or observer_exact.rules_hash != world_owner["rules"].sha256
            or receipt["observerRootHash"] != canonical_observer_information_hash_v1(
                complete_state=observer_exact.state, acting_seat=observer_exact.next_actor)):
        raise ValueError("native bid threshold refutation observer mismatch")
    if upper >= required_threshold:
        raise ValueError("native bid world upper bound does not refute the requested threshold")
    return upper


def validate_native_observed_bid_preparation_v1(value: Mapping[str, Any], *, rules,
        exact, source_prefix: ExactTracePrefixV1, selected_action: str,
        maximum_branches: int, maximum_model_bytes: int, maximum_conflicts: int,
        maximum_seconds: float, maximum_queries: int, maximum_depth: int,
        opportunity_order=None, cancel_check=None, opponent_policy=None,
        fixed_observed_world=False, target_policy=None, target_native_auction_policy_id=None) -> tuple[Any, ...]:
    """Reconstruct an actor-visible preparation; never certify strategic truth.

    Resource authority comes from the caller, not the retained receipt. Fresh
    preparation checks omitted cells and variable bindings against native models.
    """
    from .ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1

    if (not isinstance(value, Mapping)
            or type(value.get("schemaVersion")) is not int or value.get("schemaVersion") != 1
            or value.get("kind") != "native_bid_branch_preparation_v1"
            or value.get("constructionComplete") is not True
            or value.get("strategicCoverage") is not False
            or value.get("sha256") != canonical_sha256(
                {key: item for key, item in value.items() if key != "sha256"})):
        raise ValueError("observed bid preparation identity mismatch")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    if type(selected_action) is not str:
        raise ValueError("observed bid preparation requires a selected action")
    scope = value.get("scope")
    if (type(fixed_observed_world) is not bool or not isinstance(scope, Mapping)
            or ("fixedObservedWorld" in scope) != fixed_observed_world):
        raise ValueError("observed bid preparation world scope mismatch")
    if fixed_observed_world and scope["fixedObservedWorld"] != {
            "coverage": "single_compatible_deal", "stateHash": exact.state_hash,
            "prefixHash": exact.prefix_hash, "exactStateId": exact.exact_state_id}:
        raise ValueError("observed bid preparation fixed world identity mismatch")
    if opponent_policy is not None and target_policy is not None:
        raise ValueError("observed bid preparation cannot mix target and opponent policies")
    policy_team = (Seat.parse(exact.next_actor).team if target_policy is not None else
        None if opponent_policy is None else Seat.parse(exact.next_actor).team.other)
    models, rebuilt = compiler._prepare_native_bid_branches(exact, source_prefix,
        visibility="actor_information_set", selected_root_action=selected_action,
        maximum_branches=maximum_branches, maximum_model_bytes=maximum_model_bytes,
        maximum_conflicts=maximum_conflicts, maximum_seconds=maximum_seconds,
        maximum_queries=maximum_queries, maximum_depth=maximum_depth,
        opportunity_order=opportunity_order, cancel_check=cancel_check,
        policy_team=policy_team, nonplay_policy=target_policy if target_policy is not None else opponent_policy,
        fixed_observed_world=fixed_observed_world,
        **({"target_native_auction_policy_id": target_native_auction_policy_id}
           if target_native_auction_policy_id is not None else {}),
        **({"compact_purchase_invariant_ownership": True}
           if scope.get("ownershipCompaction") == "purchase_recipient_invariant_v1" else {}))

    def semantic(document):
        limits = document.get("limits")
        seconds = limits.get("seconds") if isinstance(limits, Mapping) else None
        if type(seconds) not in {int, float} or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("observed bid preparation time receipt mismatch")
        result = copy.deepcopy({key: item for key, item in document.items()
            if key not in {"sha256", "limits"}})
        for branch in result.get("branches", []):
            reachability = branch.get("reachability")
            if isinstance(reachability, dict):
                # Search effort is not partition or information-set identity.
                reachability.pop("branches", None)
                reachability.pop("conflicts", None)
            depth = branch.get("depthQuery")
            if isinstance(depth, dict):
                if (depth.get("sha256") != canonical_sha256({
                        key: item for key, item in depth.items() if key != "sha256"})
                        or type(depth.get("maximumSeconds")) not in {int, float}
                        or not math.isfinite(depth["maximumSeconds"])
                        or not 0 < depth["maximumSeconds"] <= seconds):
                    raise ValueError("observed bid preparation depth receipt mismatch")
                # Each reconstruction reaches this solve with a different
                # remaining deadline; its constraints and conclusion must match.
                depth.pop("sha256")
                depth.pop("maximumSeconds")
        return result

    if canonical_sha256(semantic(value)) != canonical_sha256(semantic(rebuilt)):
        raise ValueError("observed bid preparation differs from reconstructed universe")
    return tuple(models)


def _native_observed_bid_strategy_configuration(scope, *, policy_team,
        strategy_play_fallback_id, minimax_budget):
    """Bind strategy metadata to the caller, never enable it from a receipt."""
    if not isinstance(scope, Mapping):
        raise ValueError("observed bid strategy requires a typed scope")
    expected = {}
    if minimax_budget is not None:
        if (not isinstance(minimax_budget, ScenarioProofBudgetV1)
                or scope.get("counterexampleShortening") is not True):
            raise ValueError("observed bid strategy minimax requires a typed caller budget and explicit shortening")
        expected["minimaxProofBudgetHash"] = minimax_budget.sha256
    if {key: scope[key] for key in ("minimaxProofBudgetHash",) if key in scope} != expected:
        raise ValueError("observed bid strategy minimax budget differs from caller authority")
    if strategy_play_fallback_id is not None:
        from .ortools_play_tail import native_observer_safe_policy_identity_v1
        expected.update(playFallback=native_observer_safe_policy_identity_v1(strategy_play_fallback_id, policy_team),
            policyRepresentation="native_fallback_plus_deviations_v1")
    if {key: scope[key] for key in ("playFallback", "policyRepresentation") if key in scope} != {
            key: expected[key] for key in ("playFallback", "policyRepresentation") if key in expected}:
        raise ValueError("observed bid strategy fallback differs from caller authority")
    return expected


def validate_native_observed_bid_threshold_v1(strategy: Mapping[str, Any], preparation: Mapping[str, Any], *,
        rules, exact, source_prefix: ExactTracePrefixV1, selected_action: str, required_threshold: int,
        maximum_branches: int, maximum_model_bytes: int, maximum_conflicts: int,
        maximum_seconds: float, maximum_queries: int, maximum_depth: int, maximum_observations: int,
        opportunity_order=None, cancel_check=None, _on_queries_used=None,
        strategy_play_fallback_id=None, minimax_budget=None, _refuted_upper_bound=None) -> int:
    """Freshly verify a retained visible policy floor; not optimal-bid admission."""
    import time
    from .ortools_information_set_compiler import (
        OrToolsActorInformationSetCompilerV1)

    if _on_queries_used is not None and not callable(_on_queries_used):
        raise TypeError("query accounting callback must be callable")

    if (not isinstance(strategy, Mapping) or not isinstance(strategy.get("scope"), Mapping)
            or type(required_threshold) is not int or type(maximum_observations) is not int
            or maximum_observations < 1 or type(maximum_seconds) not in {int, float}
            or not math.isfinite(maximum_seconds) or maximum_seconds <= 0):
        raise ValueError("observed bid threshold requires typed finite limits")
    expected_configuration = _native_observed_bid_strategy_configuration(strategy["scope"],
        policy_team=Seat.parse(exact.next_actor).team,
        strategy_play_fallback_id=strategy_play_fallback_id, minimax_budget=minimax_budget)
    deadline = time.monotonic() + maximum_seconds
    models = validate_native_observed_bid_preparation_v1(preparation, rules=rules, exact=exact,
        source_prefix=source_prefix, selected_action=selected_action, maximum_branches=maximum_branches,
        maximum_model_bytes=maximum_model_bytes, maximum_conflicts=maximum_conflicts,
        maximum_seconds=maximum_seconds, maximum_queries=maximum_queries, maximum_depth=maximum_depth,
        opportunity_order=opportunity_order, cancel_check=cancel_check)
    if not models:
        raise ValueError("observed bid threshold has no reachable branch")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    actor = Seat.parse(exact.next_actor)
    maximum = max(max(map(int, variable.proto.domain)) for model in models
        for variable in model._native_round_query_context()[1])
    observation = models[0]._auction_observation_hash(actor,
        [compiler._card(card) for card in exact.state["hands"][actor.name.lower()]],
        compiler._card(exact.state["purchaseCard"]),
        [row["action"] + (":" + row["trumpSuit"] if row.get("trumpSuit") is not None else "")
            for row in source_prefix.auction_actions])
    scope = {"kind": "native_symbolic_round_policy_scope_v1", "rulesHash": rules.sha256,
        "targetTeam": actor.team.name.lower(), "coverage": "supplied_deal_branches_only",
        "utility": "awarded_point_margin", "utilityBounds": [-maximum, maximum],
        "requiredThreshold": required_threshold, "objective": "strategy_threshold_satisfaction",
        **expected_configuration,
        "branches": sorted([{"modelHash": model._model_hash(search_branching="automatic"),
            "scope": dict(model._native_round_query_context()[0])} for model in models],
            key=lambda row: row["modelHash"]),
        "fixedRoot": {"observationHash": observation,
            "legalActions": sorted(row["code"] for row in exact.legal_actions), "selectedAction": selected_action}}
    if strategy["scope"].get("counterexampleShortening") is True:
        scope["counterexampleShortening"] = True
    if strategy["scope"].get("readbackWitnessHints") == "guarded_query_chain_v1":
        scope["readbackWitnessHints"] = "guarded_query_chain_v1"
    if _refuted_upper_bound is not None:
        return _reverify_native_strategy_refuted_upper_v1(strategy, models, compiler=compiler,
            scope=scope, target=actor.team, upper_bound=_refuted_upper_bound,
            preparation_queries=preparation["queriesUsed"], deadline=deadline,
            maximum_queries=maximum_queries, maximum_observations=maximum_observations,
            maximum_conflicts=maximum_conflicts, cancel_check=cancel_check,
            on_queries_used=_on_queries_used)
    return _reverify_observed_bid_policy_threshold(strategy, preparation, models,
        compiler=compiler, scope=scope, target=actor.team, required_threshold=required_threshold,
        deadline=deadline, maximum_queries=maximum_queries, maximum_observations=maximum_observations,
        maximum_conflicts=maximum_conflicts, cancel_check=cancel_check, on_queries_used=_on_queries_used)


def _reverify_native_strategy_refuted_upper_v1(strategy, models, *, compiler, scope,
        target, upper_bound, preparation_queries, deadline, maximum_queries,
        maximum_observations, maximum_conflicts, cancel_check=None, on_queries_used=None):
    """Re-solve shared-information cuts on caller-owned legal models, not retained truth.

    This certifies only the supplied universe. The bid reader independently
    reconstructs the complete observer-compatible branch partition first.
    """
    import time
    from .ortools_information_set_compiler import InformationSetCompilationStopped

    threshold = scope.get("requiredThreshold")
    bounds = scope.get("utilityBounds")
    masters = strategy.get("masters") if isinstance(strategy, Mapping) else None
    if (type(upper_bound) is not int or type(threshold) is not int or threshold != upper_bound + 1
            or not isinstance(bounds, list) or len(bounds) != 2
            or any(type(value) is not int for value in bounds) or bounds[0] > bounds[1]
            or not isinstance(strategy, Mapping)
            or type(strategy.get("schemaVersion")) is not int or strategy["schemaVersion"] != 1
            or strategy.get("kind") != "native_symbolic_round_strategy_threshold_v1"
            or strategy.get("status") != "scoped_strategy_threshold_refuted"
            or strategy.get("scope") != scope or strategy.get("scopeHash") != canonical_sha256(scope)
            or strategy.get("sha256") != canonical_sha256({
                key: value for key, value in strategy.items() if key != "sha256"})
            or strategy.get("requiredThreshold") != threshold
            or type(strategy.get("upperBound")) is not int
            or not bounds[0] <= strategy["upperBound"] <= upper_bound
            or strategy.get("lowerBound") is not None or strategy.get("strategicCoverage") is not False
            or strategy.get("finalBranchProofs") != []
            or not isinstance(masters, list) or not masters
            or any(not isinstance(master, Mapping) for master in masters)
            or masters[-1].get("status") != "optimal"
            or masters[-1].get("searchValue") != strategy["upperBound"]
            or strategy.get("masterSearchValue") != strategy["upperBound"]
            or scope.get("targetTeam") != target.name.lower()
            or scope.get("coverage") != "supplied_deal_branches_only"
            or scope.get("objective") != "strategy_threshold_satisfaction"
            or scope.get("utility") != "awarded_point_margin"
            or set(scope) - {"kind", "rulesHash", "targetTeam", "coverage", "utility", "utilityBounds",
                "requiredThreshold", "objective", "branches", "fixedRoot", "counterexampleShortening"}
            or type(preparation_queries) is not int or not 0 <= preparation_queries < maximum_queries):
        raise ValueError("native information-set refutation ownership or coverage mismatch")
    # No fallback, incumbent, construction filter or retained cut restricts the
    # fresh target strategy. An upper bound must cover every legal target policy.
    if cancel_check is not None and cancel_check() or time.monotonic() >= deadline:
        raise InformationSetCompilationStopped("refutation_reverification_budget",
            stage="native_information_set_refutation_reader")
    root = scope.get("fixedRoot")
    fresh = compiler._solve_symbolic_round_policy_bounds(models, target,
        required_threshold=threshold, maximum_iterations=maximum_queries - preparation_queries,
        maximum_queries=maximum_queries - preparation_queries, maximum_observations=maximum_observations,
        maximum_branches=len(models), maximum_conflicts=maximum_conflicts,
        maximum_seconds=deadline - time.monotonic(), cancel_check=cancel_check,
        shorten_counterexamples=scope.get("counterexampleShortening") is True,
        fixed_root=None if root is None else (
            root["observationHash"], tuple(root["legalActions"]), root["selectedAction"]))
    if (fresh["scope"] != scope or fresh["status"] != "scoped_strategy_threshold_refuted"
            or type(fresh.get("upperBound")) is not int or fresh["upperBound"] > upper_bound):
        raise ValueError("native information-set upper did not survive fresh strategy solving")
    if on_queries_used is not None:
        on_queries_used(preparation_queries + fresh["queriesUsed"])
    return upper_bound


def validate_native_observed_bid_refuted_upper_v1(strategy, preparation, *, upper_bound, **context):
    """An all-world strategy refutation is a cap, never a selected-policy floor."""
    if type(upper_bound) is not int:
        raise ValueError("observed bid refutation upper must be an integer")
    return validate_native_observed_bid_threshold_v1(strategy, preparation,
        required_threshold=upper_bound + 1, _refuted_upper_bound=upper_bound, **context)


def _validate_native_observed_bid_refutation_search_v1(search, receipt, *, configuration,
        source_prefix, rules, action, target_team, limits, maximum_evidence_bytes):
    """Check search ownership/accounting; composition freshly proves the cap separately."""
    def hashed(value):
        return isinstance(value, Mapping) and value.get("sha256") == canonical_sha256({
            key: item for key, item in value.items() if key != "sha256"})

    expected = {"queries": configuration["maximumQueries"], "masterIterations": limits["maximumIterations"],
        "observations": limits["maximumObservations"], "branches": limits["maximumBranches"],
        "modelBytes": limits["maximumModelBytes"], "evidenceBytes": maximum_evidence_bytes,
        "conflicts": limits["maximumConflicts"], "depth": limits["maximumDepth"]}
    bounds = search.get("limits") if isinstance(search, Mapping) else None
    generation = search.get("generation") if isinstance(search, Mapping) else None
    if (not hashed(search) or search.get("kind") != "native_observed_bid_information_set_search_v1"
            or type(search.get("schemaVersion")) is not int or search["schemaVersion"] != 1
            or search.get("status") != "upper_bound_proven" or search.get("stage") != "fresh_verification"
            or search.get("queryCountComplete") is not True
            or search.get("strategicCoverage") is not False or search.get("impossibilityProven") is not False
            or search.get("rulesHash") != rules.sha256 or search.get("sourcePrefixHash") != source_prefix.sha256
            or search.get("action") != action or type(search.get("upperBound")) is not int
            or search["upperBound"] != configuration["upperBound"]
            or not isinstance(bounds, Mapping) or set(bounds) != set(expected) | {"seconds"}
            or any(type(bounds[key]) is not int or not 0 < bounds[key] <= cap for key, cap in expected.items())
            or type(bounds["seconds"]) not in {int, float} or not math.isfinite(bounds["seconds"])
            or not 0 < bounds["seconds"] <= configuration["maximumSeconds"]
            or not hashed(receipt) or receipt.get("kind") != "native_observed_bid_refuted_upper_v1"
            or type(receipt.get("upperBound")) is not int or receipt["upperBound"] != search["upperBound"]
            or not hashed(receipt.get("preparation")) or not hashed(receipt.get("strategy"))
            or not hashed(generation) or generation.get("kind") != "native_bid_policy_bounds_v1"
            or generation.get("status") != "scoped_strategy_threshold_refuted"
            or generation.get("visibility") != "actor_information_set" or generation.get("targetTeam") != target_team
            or generation.get("selectedRootAction") != action or generation.get("strategicCoverage") is not False
            or generation.get("strategy") != receipt.get("strategy")
            or generation.get("preparation") != receipt.get("preparation")
            or type(generation.get("maximumSeconds")) not in {int, float}
            or not 0 < generation["maximumSeconds"] <= bounds["seconds"]):
        raise ValueError("observed bid information-set search ownership or frozen limits mismatch")
    counts = [generation.get("queriesUsed"), search.get("verificationQueriesUsed")]
    stage_counts = [receipt["preparation"].get("queriesUsed"), receipt["strategy"].get("queriesUsed")]
    if (any(type(count) is not int or count < 1 for count in [*counts, *stage_counts])
            or counts[0] != sum(stage_counts) or type(search.get("queriesCharged")) is not int
            or search["queriesCharged"] != sum(counts) or sum(counts) > bounds["queries"]):
        raise ValueError("observed bid information-set search query accounting mismatch")
    for stage in (receipt["preparation"], receipt["strategy"]):
        declared = stage.get("limits")
        caps = {"branches": bounds["branches"], "queries": bounds["queries"],
            "conflictsPerSolve": bounds["conflicts"]}
        if (not isinstance(declared, Mapping)
                or any(type(declared.get(key)) is not int or not 0 < declared[key] <= cap
                    for key, cap in caps.items())
                or type(declared.get("seconds")) not in {int, float}
                or not 0 < declared["seconds"] <= generation["maximumSeconds"]):
            raise ValueError("observed bid information-set stage exceeds its search allowance")
    return search["queriesCharged"]


def _reverify_observed_bid_policy_threshold(strategy, preparation, models, *, compiler, scope,
        target, required_threshold, deadline, maximum_queries, maximum_observations,
        maximum_conflicts, cancel_check, on_queries_used):
    """Replay a scoped policy against every freshly reconstructed branch."""
    import time
    from .ortools_information_set_compiler import InformationSetCompilationStopped

    validate_native_strategy_threshold_v1(strategy, expected_scope=scope)
    used = preparation["queriesUsed"]
    for model in models:
        if (cancel_check is not None and cancel_check()) or time.monotonic() >= deadline or used >= maximum_queries:
            raise InformationSetCompilationStopped("observed_threshold_reverification_budget",
                stage="native_observed_bid_threshold_reader")
        hint_args = {}
        owner = model._model_hash(search_branching="automatic")
        if scope.get("readbackWitnessHints") == "guarded_query_chain_v1":
            hints = [row["receipt"] for row in strategy["queries"]
                if row["branchModelHash"] == owner and row["receipt"].get("status") == "witness_found"]
            # Hints install only freshly legality-checked guarded choices. The
            # refiner must still run a fresh native query before returning proof.
            hint_args["observation_hints"] = hints[:maximum_queries - used]
        result = compiler._refine_symbolic_round_policy(model, target, strategy["policy"],
            threshold=required_threshold, maximum_queries=maximum_queries - used,
            maximum_observations=maximum_observations, maximum_conflicts=maximum_conflicts,
            maximum_seconds=deadline - time.monotonic(), cancel_check=cancel_check, **hint_args,
            # Rebuild the same guarded auction choices as generation. Contexts
            # describe observations; the refiner freshly checks their legality.
            _auction_contexts={row["observationHash"]: row
                for entry in strategy["queries"]
                for hint in (entry["receipt"].get("query") or entry["receipt"],)
                if hint.get("status") == "witness_found"
                for row in (hint.get("candidate") or {}).get("auctionObservations", ())
                if not row.get("authoredPrefix", False)
                and Seat.parse(row["actor"]).team == target
                and row["observationHash"] in strategy["policy"]
                and row["selectedAction"] != strategy["policy"][row["observationHash"]]},
            **({"_play_fallback_id": scope["playFallback"]["policyId"],
                "_play_contexts": strategy["playContexts"],
                "_search_hint": next((row["receipt"] for row in reversed(strategy["queries"])
                    if model._native_round_fragment is not None and row["branchModelHash"] == owner
                    and row["receipt"].get("status") == "witness_found"), None)}
               if "playFallback" in scope else {}))
        used += result["queriesUsed"]
        if result["status"] != "no_adverse_round_in_branch":
            raise ValueError("observed bid policy threshold did not survive native reverification")
        if "playFallback" in scope:
            proof = next(row for row in strategy["finalBranchProofs"] if row["branchModelHash"] == owner)
            retained = strategy["refinements"][proof["refinementHash"]]["core"]
            if any(result.get(key) != retained.get(key) for key in
                   ("playFallback", "playOverrides", "fallbackBinding", "policyHash", "boundModelHash")):
                raise ValueError("guarded strategy differs from fresh native binding")
    if on_queries_used is not None:
        on_queries_used(used)
    return required_threshold


def validate_native_observed_bid_opponent_upper_v1(strategy, preparation, *, rules,
        observer_exact, exact, source_prefix, action, upper_bound,
        maximum_branches, maximum_model_bytes, maximum_conflicts, maximum_seconds,
        maximum_queries, maximum_depth, maximum_observations,
        opportunity_order=None, cancel_check=None, _on_queries_used=None,
        strategy_play_fallback_id=None, minimax_budget=None):
    """Verify a one-world opponent policy cap, never an all-world policy floor."""
    import time
    from .ortools_exact_state import OrToolsNormalizedExactStateV1
    from .ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.core.exact_state_view import canonical_observer_information_hash_v1

    if (not isinstance(observer_exact, OrToolsNormalizedExactStateV1)
            or not isinstance(exact, OrToolsNormalizedExactStateV1)
            or type(upper_bound) is not int
            or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
            or maximum_seconds <= 0
            or any(type(value) is not int or value < 1 for value in (maximum_branches,
                maximum_model_bytes, maximum_conflicts, maximum_queries, maximum_depth, maximum_observations))
            or _on_queries_used is not None and not callable(_on_queries_used)):
        raise ValueError("observed opponent upper requires normalized contexts and typed budgets")
    for context in (observer_exact, exact):
        context.require_integrity()
    def observed(context):
        return canonical_observer_information_hash_v1(
            complete_state=context.state, acting_seat=context.next_actor)
    if (observer_exact.decision_kind != "bid" or exact.decision_kind != "bid"
            or observer_exact.rules_hash != rules.sha256 or exact.rules_hash != rules.sha256
            or observer_exact.next_actor != exact.next_actor or observed(observer_exact) != observed(exact)
            or sorted(row["code"] for row in observer_exact.legal_actions)
                != sorted(row["code"] for row in exact.legal_actions)):
        raise ValueError("observed opponent upper world differs from observer root")
    expected_configuration = _native_observed_bid_strategy_configuration(
        strategy.get("scope") if isinstance(strategy, Mapping) else None,
        policy_team=Seat.parse(exact.next_actor).team.other,
        strategy_play_fallback_id=strategy_play_fallback_id, minimax_budget=minimax_budget)
    deadline = time.monotonic() + maximum_seconds
    models = validate_native_observed_bid_preparation_v1(preparation, rules=rules, exact=exact,
        source_prefix=source_prefix, selected_action=action, fixed_observed_world=True,
        maximum_branches=maximum_branches, maximum_model_bytes=maximum_model_bytes,
        maximum_conflicts=maximum_conflicts, maximum_seconds=maximum_seconds,
        maximum_queries=maximum_queries, maximum_depth=maximum_depth,
        opportunity_order=opportunity_order, cancel_check=cancel_check)
    if not models:
        raise ValueError("observed opponent upper has no reachable branch")
    target = Seat.parse(exact.next_actor).team.other
    maximum = max(max(map(int, variable.proto.domain)) for model in models
        for variable in model._native_round_query_context()[1])
    scope = {"kind": "native_symbolic_round_policy_scope_v1", "rulesHash": rules.sha256,
        "targetTeam": target.name.lower(), "coverage": "supplied_deal_branches_only",
        "utility": "awarded_point_margin", "utilityBounds": [-maximum, maximum],
        "requiredThreshold": -upper_bound, "objective": "strategy_threshold_satisfaction",
        **expected_configuration,
        "branches": sorted([{"modelHash": model._model_hash(search_branching="automatic"),
            "scope": dict(model._native_round_query_context()[0])} for model in models],
            key=lambda row: row["modelHash"])}
    if strategy["scope"].get("counterexampleShortening") is True:
        scope["counterexampleShortening"] = True
    if strategy["scope"].get("readbackWitnessHints") == "guarded_query_chain_v1":
        scope["readbackWitnessHints"] = "guarded_query_chain_v1"
    _reverify_observed_bid_policy_threshold(strategy, preparation, models,
        compiler=OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True), scope=scope,
        target=target, required_threshold=-upper_bound, deadline=deadline,
        maximum_queries=maximum_queries, maximum_observations=maximum_observations,
        maximum_conflicts=maximum_conflicts, cancel_check=cancel_check, on_queries_used=_on_queries_used)
    return upper_bound


def validate_native_observed_bid_visible_floor_v1(strategy, preparation, *, rules,
        exact, source_prefix, selected_action, required_threshold,
        maximum_branches, maximum_model_bytes, maximum_conflicts, maximum_seconds,
        maximum_queries, maximum_depth, maximum_observations,
        opportunity_order=None, cancel_check=None, _on_queries_used=None):
    """Rebuild the entire observed universe and verify each native-policy floor."""
    import time
    from .ortools_exact_state import OrToolsNormalizedExactStateV1
    from .ortools_information_set_compiler import InformationSetCompilationStopped
    from .ortools_play_tail import native_observer_safe_policy_identity_v1

    fields = {"schemaVersion", "kind", "preparationHash", "nativePlayPolicyId",
        "nonplayPolicy", "requiredThreshold", "branches", "sha256"}
    if (not isinstance(exact, OrToolsNormalizedExactStateV1)
            or not isinstance(strategy, Mapping) or not fields <= set(strategy) <= fields | {"observedProjectPolicyId", "observedAuctionPolicyId"}
            or type(strategy["schemaVersion"]) is not int or strategy["schemaVersion"] != 1
            or strategy["kind"] != "native_observed_bid_visible_policy_floor_v1"
            or strategy["sha256"] != canonical_sha256({k: v for k, v in strategy.items() if k != "sha256"})
            or strategy["preparationHash"] != canonical_sha256(preparation)
            or type(required_threshold) is not int or type(strategy["requiredThreshold"]) is not int
            or strategy["requiredThreshold"] != required_threshold
            or not isinstance(strategy["branches"], list)
            or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
            or maximum_seconds <= 0 or type(maximum_observations) is not int or maximum_observations < 1
            or _on_queries_used is not None and not callable(_on_queries_used)):
        raise ValueError("observed native-policy floor identity or limits mismatch")
    nonplay = strategy["nonplayPolicy"]
    if (not isinstance(nonplay, Mapping) or set(nonplay) != {"auction_action", "escalation_action"}
            or any(action is not None and type(action) is not str for action in nonplay.values())):
        raise ValueError("observed native-policy floor requires a closed nonplay policy")
    exact.require_integrity()
    native_observer_safe_policy_identity_v1(strategy["nativePlayPolicyId"], Seat.parse(exact.next_actor).team)
    project_id = strategy.get("observedProjectPolicyId")
    auction_id = strategy.get("observedAuctionPolicyId")
    if (not isinstance(preparation, Mapping) or not isinstance(preparation.get("scope"), Mapping)
            or preparation["scope"].get("targetNativeAuctionPolicyId") != auction_id
            or ("targetNativeAuctionPolicyId" in preparation["scope"]) != (auction_id is not None)):
        raise ValueError("native visible auction strategy differs from preparation")
    if "observedAuctionPolicyId" in strategy:
        from .ortools_scenario_model import validate_native_observed_auction_policy_id_v1
        validate_native_observed_auction_policy_id_v1(auction_id)
        if nonplay["auction_action"] is not None:
            raise ValueError("native visible auction strategy mixes auction authorities")
    from .ortools_project_fragment import NATIVE_FULL_HAND_PROJECT_POLICY_ID
    if "observedProjectPolicyId" in strategy:
        from .ortools_project_fragment import validate_native_observed_project_policy_id_v1
        validate_native_observed_project_policy_id_v1(project_id)
    deadline = time.monotonic() + maximum_seconds
    # Historical floors prepared a policy-filtered universe. New floors bind
    # that policy only in their owned queries over the broader root universe.
    prepared_policy = (nonplay if isinstance(preparation, Mapping)
        and isinstance(preparation.get("scope"), Mapping)
        and "nonplayPolicy" in preparation["scope"] else None)
    models = validate_native_observed_bid_preparation_v1(preparation, rules=rules, exact=exact,
        source_prefix=source_prefix, selected_action=selected_action, target_policy=prepared_policy,
        maximum_branches=maximum_branches, maximum_model_bytes=maximum_model_bytes,
        maximum_conflicts=maximum_conflicts, maximum_seconds=maximum_seconds,
        maximum_queries=maximum_queries, maximum_depth=maximum_depth,
        opportunity_order=opportunity_order, cancel_check=cancel_check,
        **({"target_native_auction_policy_id": auction_id} if auction_id is not None else {}))
    hashes = [model._model_hash(search_branching="automatic") for model in models]
    rows = strategy["branches"]
    if (not models or len(set(hashes)) != len(hashes)
            or any(not isinstance(row, Mapping) or set(row) != {"sourceModelHash", "receipt"} |
                   ({"projectActions"} if project_id is not None and project_id != NATIVE_FULL_HAND_PROJECT_POLICY_ID
                    else set()) for row in rows)
            or [row["sourceModelHash"] for row in rows] != hashes):
        raise ValueError("observed native-policy floor branch coverage mismatch")
    target = Seat.parse(exact.next_actor).team
    used = preparation["queriesUsed"]
    for model, row in zip(models, rows, strict=True):
        project_args = {}
        if auction_id is not None:
            project_args["native_auction_policy_id"] = auction_id
        if project_id == NATIVE_FULL_HAND_PROJECT_POLICY_ID:
            project_args["native_project_policy_id"] = project_id
        elif project_id is not None:
            actor = Seat.parse(exact.next_actor)
            fragment = model._native_round_fragment
            project_args["project_actions"] = (fragment.projects.known_project_policy_actions(
                project_id, actor, tuple(Card.parse(card) for card in exact.state["hands"][actor.name.lower()]))
                if fragment is not None else {})
            if row["projectActions"] != project_args["project_actions"]:
                raise ValueError("observed native-policy floor substitutes actor-visible declarations")
        query_cost = 1 + int(nonplay["auction_action"] is not None) + int(
            nonplay["escalation_action"] is not None and model._native_round_fragment is not None)
        query_cost += len(project_args.get("project_actions", {}))
        query_cost += int(auction_id is not None)
        if (cancel_check is not None and cancel_check()) or time.monotonic() >= deadline or used + query_cost > maximum_queries:
            raise InformationSetCompilationStopped("native_floor_reverification_budget",
                stage="native_observed_bid_visible_floor_reader")
        if not isinstance(row["receipt"], Mapping):
            raise ValueError("observed native-policy floor requires a query receipt")
        sizes = row["receipt"].get("modelBytes")
        if sizes is not None:
            if (not isinstance(sizes, Mapping) or sizes.get("unit") != "utf8_cp_model_proto"
                    or sizes.get("includesSolverWorkingMemory") is not False
                    or any(type(sizes.get(key)) is not int or sizes[key] < 0
                           for key in ("source", "retainedUpperBound"))
                    or sizes["source"] > preparation["modelBytes"]
                    or sizes["retainedUpperBound"] < sizes["source"]):
                raise ValueError("observed native-policy floor model-byte evidence mismatch")
            if preparation["modelBytes"] - sizes["source"] + sizes["retainedUpperBound"] > maximum_model_bytes:
                raise InformationSetCompilationStopped("maximum_model_bytes", stage="native_observed_bid_visible_floor_reader")
        validate_native_visible_policy_threshold_v1(row["receipt"], rules=rules, scenario=model,
            target=target, policy_team=target, native_play_policy_id=strategy["nativePlayPolicyId"],
            auction_action=nonplay["auction_action"], escalation_action=nonplay["escalation_action"],
            required_threshold=required_threshold, maximum_seconds=deadline - time.monotonic(),
            maximum_conflicts=maximum_conflicts, cancel_check=cancel_check, record_model_bytes=sizes is not None,
            **project_args)
        used += query_cost
    if _on_queries_used is not None:
        _on_queries_used(used)
    return required_threshold


def compose_native_observed_bid_dominance_v1(*, rules, exact, source_prefix,
        selected_action, strategy, preparation, required_threshold, alternatives,
        opponent_policy, maximum_branches, maximum_model_bytes, maximum_conflicts,
        maximum_seconds, maximum_queries, maximum_depth, maximum_observations,
        opportunity_order=None, cancel_check=None, opponent_play_policy_id=None,
        strategy_play_fallback_id=None, minimax_budget=None):
    """Compose freshly proved bounds; not a full-round or publication receipt."""
    import time
    from ortools.sat.python import cp_model
    from .ortools_exact_state import OrToolsNormalizedExactStateV1, _effective_prefix_hash
    from .ortools_information_set_compiler import InformationSetCompilationStopped
    from .ortools_information_set_proof import build_native_bid_dominance_model_v1, _solve_cp_sat_with_cancel_v1
    from baloot_platform.core.exact_state_view import canonical_observer_information_hash_v1

    if (not isinstance(exact, OrToolsNormalizedExactStateV1)
            or not isinstance(source_prefix, ExactTracePrefixV1)
            or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
            or maximum_seconds <= 0 or type(maximum_queries) is not int
            or type(required_threshold) is not int
            or any(type(value) is not int or value < 1 for value in (maximum_branches,
                maximum_model_bytes, maximum_conflicts, maximum_depth, maximum_observations))
            or opponent_policy is not None and (
                not isinstance(opponent_policy, Mapping)
                or set(opponent_policy) != {"auction_action", "escalation_action"}
                or any(value is not None and type(value) is not str for value in opponent_policy.values()))):
        raise ValueError("observed dominance requires typed context and finite budgets")
    exact.require_integrity()
    actions = sorted(row["code"] for row in exact.legal_actions)
    if (exact.decision_kind != "bid" or exact.rules_hash != rules.sha256
            or exact.prefix_hash != _effective_prefix_hash(source_prefix)
            or selected_action not in actions or not isinstance(alternatives, (tuple, list))
            or any(not isinstance(row, Mapping) or set(row) != {"action", "exact", "sourcePrefix", "receipt"}
                   or type(row["action"]) is not str or not isinstance(row["receipt"], Mapping)
                   for row in alternatives)
            or sorted(row["action"] for row in alternatives) != [a for a in actions if a != selected_action]):
        raise ValueError("observed dominance root or alternative coverage mismatch")
    if opponent_policy is None and (not alternatives or any(row["receipt"].get("kind")
            not in _NATIVE_OBSERVED_BID_STRATEGY_UPPERS for row in alternatives)):
        raise ValueError("fixed opponent bounds require their declared policy")
    actor = Seat.parse(exact.next_actor)
    if strategy_play_fallback_id is not None:
        from .ortools_play_tail import native_observer_safe_policy_identity_v1
        native_observer_safe_policy_identity_v1(strategy_play_fallback_id, actor.team)
    if minimax_budget is not None and (not isinstance(minimax_budget, ScenarioProofBudgetV1)
            or strategy_play_fallback_id is None):
        raise ValueError("observed dominance minimax requires a typed budget and governed fallback")
    play_policy = None
    if opponent_play_policy_id is not None:
        from .ortools_play_tail import native_observer_safe_policy_identity_v1
        play_policy = native_observer_safe_policy_identity_v1(opponent_play_policy_id, actor.team.other)
        if any(row["receipt"].get("kind") in _NATIVE_OBSERVED_BID_STRATEGY_UPPERS
               for row in alternatives):
            raise ValueError("fixed play policy cannot relabel synthesized opponent bounds")
    observer = canonical_observer_information_hash_v1(complete_state=exact.state, acting_seat=actor.name.lower())
    for row in alternatives:
        world = row["exact"]
        if not isinstance(world, OrToolsNormalizedExactStateV1):
            raise ValueError("observed dominance alternative requires a normalized world")
        world.require_integrity()
        if (world.next_actor != exact.next_actor or world.rules_hash != rules.sha256
                or sorted(a["code"] for a in world.legal_actions) != actions
                or canonical_observer_information_hash_v1(complete_state=world.state,
                    acting_seat=actor.name.lower()) != observer):
            raise ValueError("observed dominance alternative observer mismatch")
    # Reserve one native comparison query; each recheck gets a disjoint slice.
    per_check = (maximum_queries - 1) // len(actions)
    if per_check < 1:
        raise ValueError("observed dominance query budget cannot cover every action")
    deadline = time.monotonic() + maximum_seconds

    def remaining():
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage="native_observed_bid_dominance")
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise InformationSetCompilationStopped("maximum_seconds", stage="native_observed_bid_dominance")
        return seconds

    common = dict(rules=rules, maximum_branches=maximum_branches, maximum_model_bytes=maximum_model_bytes,
        maximum_conflicts=maximum_conflicts, maximum_queries=per_check, maximum_depth=maximum_depth,
        opportunity_order=opportunity_order, cancel_check=cancel_check)
    measured = []
    lower_reader = (validate_native_observed_bid_visible_floor_v1
        if isinstance(strategy, Mapping) and strategy.get("kind") == "native_observed_bid_visible_policy_floor_v1"
        else validate_native_observed_bid_threshold_v1)
    floor = lower_reader(strategy, preparation, exact=exact,
        source_prefix=source_prefix, selected_action=selected_action, required_threshold=required_threshold,
        maximum_observations=maximum_observations, maximum_seconds=remaining(),
        **({"strategy_play_fallback_id": strategy_play_fallback_id}
           if strategy_play_fallback_id is not None and lower_reader is validate_native_observed_bid_threshold_v1 else {}),
        **({"minimax_budget": minimax_budget} if minimax_budget is not None
           and lower_reader is validate_native_observed_bid_threshold_v1 else {}),
        _on_queries_used=measured.append, **common)
    bounds = []
    synthesized_upper = any(isinstance(row["receipt"], Mapping) and row["receipt"].get("kind")
        in _NATIVE_OBSERVED_BID_STRATEGY_UPPERS for row in alternatives)
    for row in sorted(alternatives, key=lambda item: item["action"]):
        receipt = row["receipt"]
        if isinstance(receipt, Mapping) and receipt.get("kind") in _NATIVE_OBSERVED_BID_STRATEGY_UPPERS:
            if (set(receipt) != {"schemaVersion", "kind", "strategy", "preparation", "upperBound", "sha256"}
                    or type(receipt["schemaVersion"]) is not int or receipt["schemaVersion"] != 1
                    or type(receipt["upperBound"]) is not int
                    or receipt["sha256"] != canonical_sha256({
                        key: value for key, value in receipt.items() if key != "sha256"})):
                raise ValueError("observed opponent policy upper receipt integrity mismatch")
            refuted = receipt["kind"] == "native_observed_bid_refuted_upper_v1"
            upper_reader = (validate_native_observed_bid_refuted_upper_v1 if refuted
                else validate_native_observed_bid_opponent_upper_v1)
            upper = upper_reader(
                receipt["strategy"], receipt["preparation"], exact=row["exact"],
                source_prefix=row["sourcePrefix"], upper_bound=receipt["upperBound"],
                **({"selected_action": row["action"]} if refuted else {
                    "observer_exact": exact, "action": row["action"]}),
                maximum_observations=maximum_observations, maximum_seconds=remaining(),
                **({"strategy_play_fallback_id": strategy_play_fallback_id}
                   if "playFallback" in receipt["strategy"].get("scope", {}) else {}),
                **({"minimax_budget": minimax_budget}
                   if "minimaxProofBudgetHash" in receipt["strategy"].get("scope", {}) else {}),
                _on_queries_used=measured.append, **common)
        else:
            upper = reverify_native_bid_world_upper_v1(receipt, exact=row["exact"],
                source_prefix=row["sourcePrefix"], action=row["action"], opponent_policy=opponent_policy,
                **({"opponent_play_policy_id": opponent_play_policy_id} if play_policy is not None else {}),
                maximum_seconds=remaining(), _on_queries_used=measured.append, **common)
        bounds.append({"action": row["action"], "upperBound": upper,
            "sourceProofHash": canonical_sha256(row["receipt"]),
            "worldExactStateId": row["exact"].exact_state_id,
            **({"sourceProofKind": receipt["kind"]}
                if isinstance(receipt, Mapping) and receipt.get("kind") in _NATIVE_OBSERVED_BID_STRATEGY_UPPERS
                else {"opponentPolicy": dict(opponent_policy)} if synthesized_upper else {})})
    if (len(measured) != len(actions)
            or any(type(count) is not int or not 0 <= count <= per_check for count in measured)):
        raise ValueError("observed dominance verification query accounting mismatch")
    model, _, claim, _, _, bindings = build_native_bid_dominance_model_v1(floor, bounds)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_number_of_conflicts = maximum_conflicts
    solver.parameters.max_time_in_seconds = remaining()
    status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
    remaining()
    if status != cp_model.OPTIMAL:
        raise InformationSetCompilationStopped("comparison_incomplete", stage="native_observed_bid_dominance")
    proven = solver.boolean_value(claim)
    core = {"schemaVersion": 1, "kind": "native_observed_bid_selected_action_dominance_v1",
        "status": "selected_action_optimal" if proven else "bounds_do_not_establish_dominance",
        "visibility": "actor_information_set", "completionScope": "selected_bid_current_deal",
        "wholeRoundCertified": False, "rulesHash": rules.sha256,
        "sourcePrefixHash": source_prefix.sha256, "exactStateId": exact.exact_state_id,
        "observerRootHash": observer, "targetTeam": actor.team.name.lower(),
        "legalActions": actions, "selectedAction": selected_action,
        "selectedLowerBound": floor, "selectedProofHash": canonical_sha256(strategy),
        "preparationHash": canonical_sha256(preparation), "alternatives": bounds,
        "premises": {"sourcePrefix": source_prefix.to_dict(),
            "selectedStrategy": copy.deepcopy(dict(strategy)),
            "selectedPreparation": copy.deepcopy(dict(preparation)),
            "alternativeWorlds": [{"action": row["action"],
                "sourcePrefix": row["sourcePrefix"].to_dict(),
                "receipt": copy.deepcopy(dict(row["receipt"]))}
                for row in sorted(alternatives, key=lambda item: item["action"])]},
        "limits": {"maximumSeconds": maximum_seconds, "maximumQueries": maximum_queries,
            "queriesPerRecheck": per_check, "comparisonQueriesReserved": 1,
            "maximumBranches": maximum_branches, "maximumModelBytes": maximum_model_bytes,
            "maximumConflicts": maximum_conflicts, "maximumDepth": maximum_depth,
            "maximumObservations": maximum_observations, "workersPerSolve": 1},
        "executionReceipt": {"queriesUsed": sum(measured) + 1, "comparisonQueriesUsed": 1,
            "rechecks": [{"action": action, "queriesUsed": count} for action, count in zip(
                [selected_action, *(row["action"] for row in bounds)], measured, strict=True)]},
        "escalationOpportunityOrder": None if opportunity_order is None else [
            Seat.parse(seat).name.lower() for seat in opportunity_order],
        "opponentPolicy": None if synthesized_upper else dict(opponent_policy),
        **({"opponentPlayPolicy": play_policy} if play_policy is not None else {}),
        "actualActionOptimal": True if proven else None,
        "actualValue": None, "bestValue": None, "optimalActionCount": None,
        "modelProto": str(model.proto), "claimLiteralIndex": int(claim.index),
        "variableBindings": bindings, "solverStatus": "optimal"}
    return {**core, "sha256": canonical_sha256(core)}


def reverify_native_observed_bid_dominance_v1(value, *, rules, exact, source_prefix,
        selected_action, required_threshold, opponent_policy, alternative_worlds, **limits):
    """Read owned premises using independently normalized caller world contexts."""
    if (not isinstance(value, Mapping) or value.get("sha256") != canonical_sha256({
            key: item for key, item in value.items() if key != "sha256"})
            or value.get("kind") != "native_observed_bid_selected_action_dominance_v1"
            or not isinstance(value.get("premises"), Mapping)):
        raise ValueError("observed dominance certificate integrity mismatch")
    premises = value["premises"]
    if (set(premises) != {"sourcePrefix", "selectedStrategy", "selectedPreparation", "alternativeWorlds"}
            or premises["sourcePrefix"] != source_prefix.to_dict()
            or not isinstance(premises["alternativeWorlds"], list)
            or not isinstance(alternative_worlds, Mapping)
            or any(not isinstance(row, Mapping) or set(row) != {"action", "sourcePrefix", "receipt"}
                or type(row["action"]) is not str for row in premises["alternativeWorlds"])
            or [row["action"] for row in premises["alternativeWorlds"]] != sorted(alternative_worlds)):
        raise ValueError("observed dominance premise ownership mismatch")
    alternatives = []
    for row in premises["alternativeWorlds"]:
        world, prefix = alternative_worlds[row["action"]]
        if row["sourcePrefix"] != prefix.to_dict():
            raise ValueError("observed dominance alternative prefix mismatch")
        alternatives.append({"action": row["action"], "sourcePrefix": prefix,
            "exact": world, "receipt": row["receipt"]})
    if opponent_policy is None:
        bounds = value.get("alternatives")
        if not isinstance(bounds, list) or any(not isinstance(row, Mapping) for row in bounds):
            raise ValueError("observed dominance alternative policy metadata mismatch")
        policies = [row.get("opponentPolicy") for row in bounds
            if row.get("sourceProofKind") not in _NATIVE_OBSERVED_BID_STRATEGY_UPPERS]
        if policies:
            if not isinstance(policies[0], Mapping) or any(policy != policies[0] for policy in policies):
                raise ValueError("observed dominance fixed policies disagree")
            opponent_policy = policies[0]
    rebuilt = compose_native_observed_bid_dominance_v1(rules=rules, exact=exact,
        source_prefix=source_prefix, selected_action=selected_action,
        strategy=premises["selectedStrategy"], preparation=premises["selectedPreparation"],
        required_threshold=required_threshold, opponent_policy=opponent_policy,
        alternatives=alternatives, **limits)
    if canonical_sha256(rebuilt) != canonical_sha256(value):
        raise ValueError("observed dominance certificate differs from fresh verification")
    return rebuilt


def build_native_observed_bid_quality_artifact_v1(attempt, *, rules, source_prefix,
        normalized, request, budget, maximum_conflicts, opportunity_order,
        maximum_seconds, cancel_check=None, opponent_play_policy_id=None, execution_budget=None,
        receipt_version=1):
    """Reconstruct external evidence; no caller-supplied proof-reuse tokens."""
    return _build_native_observed_bid_quality_artifact_v1(attempt, rules=rules,
        source_prefix=source_prefix, normalized=normalized, request=request, budget=budget,
        maximum_conflicts=maximum_conflicts, opportunity_order=opportunity_order,
        maximum_seconds=maximum_seconds, cancel_check=cancel_check,
        opponent_play_policy_id=opponent_play_policy_id, execution_budget=execution_budget,
        receipt_version=receipt_version)


def _build_native_observed_bid_quality_artifact_v1(attempt, *, rules, source_prefix,
        normalized, request, budget, maximum_conflicts, opportunity_order,
        maximum_seconds, cancel_check=None, opponent_play_policy_id=None, execution_budget=None,
        receipt_version=2, verified_certificates=frozenset()):
    """Freshly verify owned observed bid evidence; never certify the whole round."""
    import time
    from .ortools_exact_state import OrToolsExactStateNormalizerV1, OrToolsNormalizedExactStateV1, _effective_prefix_hash
    from .ortools_information_set_compiler import InformationSetCompilationStopped

    def hashed(value):
        return isinstance(value, Mapping) and value.get("sha256") == canonical_sha256({
            key: item for key, item in value.items() if key != "sha256"})

    if (type(receipt_version) is not int or receipt_version not in {1, 2}
            or verified_certificates and receipt_version != 2
            or not isinstance(rules, RuleBook) or not isinstance(source_prefix, ExactTracePrefixV1)
            or not isinstance(normalized, OrToolsNormalizedExactStateV1)
            or not isinstance(request, ScenarioProofRequestV1)
            or not isinstance(budget, ScenarioProofBudgetV1) or budget.native_observed is None
            or type(maximum_conflicts) is not int or maximum_conflicts < 1
            or not isinstance(opportunity_order, (tuple, list)) or len(opportunity_order) != len(Seat)
            or any(type(seat) is not str for seat in opportunity_order)
            or set(opportunity_order) != {seat.name.lower() for seat in Seat}
            or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
            or maximum_seconds <= 0):
        raise ValueError("observed bid quality requires typed frozen context and budgets")
    normalized.require_integrity()
    selected_action = attempt.get("selectedAction") if isinstance(attempt, Mapping) else None
    play_policy = None
    if opponent_play_policy_id is not None:
        from .ortools_play_tail import native_observer_safe_policy_identity_v1
        play_policy = native_observer_safe_policy_identity_v1(
            opponent_play_policy_id, Seat.parse(normalized.next_actor).team.other)
    if (not isinstance(attempt, Mapping) or not isinstance(attempt.get("certificate"), Mapping)
            or attempt.get("opponentPlayPolicy") != play_policy
            or attempt["certificate"].get("opponentPlayPolicy") != play_policy):
        raise ValueError("observed bid opponent play policy differs from frozen configuration")
    if (normalized.decision_kind != "bid" or normalized.rules_hash != rules.sha256
            or normalized.prefix_hash != _effective_prefix_hash(source_prefix)
            or request.mode != "exact" or request.visibility != "actor_information_set"
            or request.utility != "awarded_point_margin" or request.line_universe is not None
            or request.budget_profile != budget.profile_id
            or request.target_team != Seat.parse(normalized.next_actor).team.name.lower()
            or not hashed(attempt) or attempt.get("kind") != "native_observed_bid_dominance_attempt_v1"
            or type(attempt.get("schemaVersion")) is not int or attempt["schemaVersion"] != 1
            or attempt.get("status") != "selected_action_optimal"
            or attempt.get("stage") != "fresh_verification" or attempt.get("queryCountComplete") is not True
            or attempt.get("sourcePrefixHash") != source_prefix.sha256
            or attempt.get("rulesHash") != rules.sha256
            or selected_action not in {row["code"] for row in normalized.legal_actions}
            or request.selected_action not in {None, selected_action}):
        raise ValueError("observed bid quality request or attempt ownership mismatch")
    native = budget.native_observed
    prior = attempt.get("priorSelectionAttempts")
    schedule = attempt.get("selectionSchedule")
    if schedule not in (None, "incumbent_then_deep_v1"):
        raise ValueError("unsupported generated bid selection schedule")
    selection_queries = 0
    if request.selected_action is None:
        legal_order = [row["code"] for row in normalized.legal_actions]
        if (not isinstance(prior, list) or len(prior) not in (
                {legal_order.index(selected_action), len(legal_order) + legal_order.index(selected_action)}
                if schedule is not None else {legal_order.index(selected_action)})
                or any(not isinstance(row, Mapping)
                    or set(row) != {"action", "queryAllowance", "queriesCharged", "receipt"}
                    or row.get("action") != legal_order[index % len(legal_order)]
                    or type(row.get("queryAllowance")) is not int
                    or not 0 < row["queryAllowance"] <= native.maximum_queries
                    or type(row.get("queriesCharged")) is not int
                    or row["queriesCharged"] != row["queryAllowance"]
                    or not hashed(row.get("receipt"))
                    or row["receipt"].get("status") != "incomplete"
                    for index, row in enumerate(prior))):
            raise ValueError("generated bid selection attempts or resource charges mismatch")
        selection_queries = sum(row["queriesCharged"] for row in prior)
    elif prior is not None or schedule is not None:
        raise ValueError("recorded bid cannot substitute generated selection attempts")
    caps = {"maximumIterations": native.maximum_iterations,
        "maximumObservations": native.maximum_observations, "maximumBranches": budget.maximum_state_space,
        "maximumModelBytes": budget.maximum_cache_bytes, "maximumConflicts": maximum_conflicts,
        "maximumDepth": min(budget.maximum_depth, budget.maximum_horizon_actions),
        "workersPerSolve": min(1, budget.maximum_workers)}
    limits, certificate, selected = (attempt.get(key) for key in ("limits", "certificate", "selectedProof"))
    if (not isinstance(limits, Mapping) or set(limits) != set(caps)
            or any(type(limits[key]) is not int or not 0 < limits[key] <= cap for key, cap in caps.items())
            or type(attempt.get("maximumQueries")) is not int
            or not 0 < attempt["maximumQueries"] <= native.maximum_queries
            or type(attempt.get("maximumSeconds")) not in {int, float}
            or not math.isfinite(attempt["maximumSeconds"])
            or not 0 < attempt["maximumSeconds"] <= min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds)
            or not hashed(certificate) or not hashed(selected)
            or not isinstance(selected.get("strategy"), Mapping)
            or not isinstance(selected.get("preparation"), Mapping)
            or selected.get("status") != "scoped_strategy_threshold_complete"
            or attempt.get("activePreparation") is not None
            or certificate.get("status") != "selected_action_optimal"
            or certificate.get("escalationOpportunityOrder") != list(opportunity_order)
            or certificate.get("premises", {}).get("selectedStrategy") != selected.get("strategy")
            or certificate.get("premises", {}).get("selectedPreparation") != selected.get("preparation")):
        raise ValueError("observed bid quality exceeds frozen limits or substitutes premises")
    strategy_fallback = (ScenarioQualityExecutionBudgetV1.build(execution_budget.to_dict()).observed_bid_strategy_play_fallback_id
        if isinstance(execution_budget, ScenarioQualityExecutionBudgetV1) else None)
    if (attempt.get("strategyPlayFallbackId") != strategy_fallback
            or "strategyPlayFallbackId" in attempt and strategy_fallback is None):
        raise ValueError("observed bid strategy fallback differs from frozen configuration")
    strategy_scope = selected["strategy"].get("scope", {})
    if not isinstance(strategy_scope, Mapping) or ("minimaxProofBudgetHash" in strategy_scope
            and (strategy_scope["minimaxProofBudgetHash"] != budget.sha256 or strategy_fallback is None)):
        raise ValueError("observed bid minimax budget differs from frozen configuration")
    native_floor = selected["strategy"].get("kind") == "native_observed_bid_visible_policy_floor_v1"
    fallback_attempt = attempt.get("incumbentAttempt")
    if fallback_attempt is not None:
        incumbent = (ScenarioQualityExecutionBudgetV1.build(execution_budget.to_dict()).observed_bid_incumbent_proof
            if isinstance(execution_budget, ScenarioQualityExecutionBudgetV1) else None)
        if (native_floor or incumbent is None or incumbent["schemaVersion"] != 2
                or attempt.get("incumbentProof") != incumbent or not hashed(fallback_attempt)
                or fallback_attempt.get("kind") != "native_bid_policy_bounds_v1"
                or type(fallback_attempt.get("schemaVersion")) is not int or fallback_attempt["schemaVersion"] != 1
                or fallback_attempt.get("status") != "incomplete"
                or fallback_attempt.get("visibility") != "actor_information_set"
                or fallback_attempt.get("targetTeam") != request.target_team
                or fallback_attempt.get("selectedRootAction") != selected_action
                or fallback_attempt.get("strategicCoverage") is not False
                or type(fallback_attempt.get("maximumSeconds")) not in {int, float}
                or not 0 < fallback_attempt["maximumSeconds"] <= incumbent["maximumSeconds"]
                or type(fallback_attempt.get("queriesUsed")) is not int
                or not 0 <= fallback_attempt["queriesUsed"] <= incumbent["maximumQueries"]):
            raise ValueError("observed bid incumbent fallback differs from frozen allowance or root")
        interrupted, refutation = fallback_attempt.get("partialReceipt"), fallback_attempt.get("strategy")
        if interrupted is not None:
            partial_evidence = interrupted.get("evidence") if isinstance(interrupted, Mapping) else None
            if (refutation is not None or not hashed(interrupted)
                    or not isinstance(partial_evidence, Mapping)
                    or interrupted.get("stage") != "native_observed_bid_floor"
                    or interrupted.get("status") != "incomplete"
                    or interrupted.get("exhaustionReason") not in {
                        "maximum_seconds", "maximum_queries", "selected_policy_floor_not_certified"}
                    or type(partial_evidence.get("queriesUsed")) is not int
                    or partial_evidence["queriesUsed"] != fallback_attempt["queriesUsed"]
                    or partial_evidence.get("requiredThreshold") != certificate["selectedLowerBound"]
                    or partial_evidence.get("nativePlayPolicyId") != incumbent["nativePlayPolicyId"]
                    or partial_evidence.get("observedAuctionPolicyId") != incumbent.get("observedAuctionPolicyId")
                    or partial_evidence.get("observedProjectPolicyId") != incumbent.get("observedProjectPolicyId")):
                raise ValueError("observed bid fallback substitutes the interrupted incumbent")
        elif (not hashed(refutation) or refutation.get("kind") != "native_observed_bid_policy_floor_refutation_v1"
                or refutation.get("status") != "refuted"
                or refutation.get("nativePlayPolicyId") != incumbent["nativePlayPolicyId"]
                or refutation.get("observedAuctionPolicyId") != incumbent.get("observedAuctionPolicyId")
                or refutation.get("observedProjectPolicyId") != incumbent.get("observedProjectPolicyId")
                or refutation.get("nonplayPolicy") != {"auction_action": None, "escalation_action": None}
                or refutation.get("requiredThreshold") != certificate["selectedLowerBound"]
                or refutation.get("queriesUsed") != fallback_attempt["queriesUsed"]):
            raise ValueError("observed bid fallback requires a refuted or incomplete incumbent")
    elif native_floor or "incumbentProof" in attempt:
        incumbent = (ScenarioQualityExecutionBudgetV1.build(execution_budget.to_dict()).observed_bid_incumbent_proof
            if isinstance(execution_budget, ScenarioQualityExecutionBudgetV1) else None)
        if (not native_floor or incumbent is None or incumbent["schemaVersion"] != 2
                or attempt.get("incumbentProof") != incumbent
                or selected["strategy"].get("nativePlayPolicyId") != incumbent["nativePlayPolicyId"]
                or selected["strategy"].get("observedAuctionPolicyId") != incumbent.get("observedAuctionPolicyId")
                or selected["strategy"].get("observedProjectPolicyId") != incumbent.get("observedProjectPolicyId")
                or selected["strategy"].get("nonplayPolicy") != {"auction_action": None, "escalation_action": None}
                or type(selected.get("maximumSeconds")) not in {int, float}
                or not 0 < selected["maximumSeconds"] <= incumbent["maximumSeconds"]
                or type(selected.get("queriesUsed")) is not int
                or not 0 <= selected["queriesUsed"] <= incumbent["maximumQueries"]
                or type(selected["preparation"].get("queriesUsed")) is not int
                or not isinstance(selected["preparation"].get("limits"), Mapping)
                or type(selected["preparation"]["limits"].get("queries")) is not int
                or not isinstance(selected["strategy"].get("branches"), list)
                or any(not isinstance(row, Mapping) or not isinstance(row.get("receipt"), Mapping)
                       or not isinstance(row["receipt"].get("bindings"), list)
                       for row in selected["strategy"]["branches"])
                or selected["queriesUsed"] != selected["preparation"]["queriesUsed"] + sum(
                    len(row["receipt"]["bindings"]) + 1 for row in selected["strategy"]["branches"])
                or selected["preparation"]["limits"]["queries"] > incumbent["maximumQueries"]):
            raise ValueError("observed bid incumbent differs from frozen policy or allowance")
    verification = certificate["limits"]
    if (any(verification.get(key) != limits[key] for key in caps if key != "maximumIterations")
            or type(verification.get("maximumSeconds")) not in {int, float}
            or not 0 < verification["maximumSeconds"] <= attempt["maximumSeconds"]):
        raise ValueError("observed bid verification budget mismatch")
    alternatives = attempt.get("alternatives")
    worlds = certificate["premises"]["alternativeWorlds"]
    if (not isinstance(worlds, list)
            or any(not isinstance(row, Mapping) or set(row) != {"action", "sourcePrefix", "receipt"}
                or type(row["action"]) is not str for row in worlds)
            or not isinstance(alternatives, list) or alternatives != [
                {"action": row["action"], "receipt": row["receipt"]} for row in worlds]
            or [row["action"] for row in worlds] != sorted(
                row["code"] for row in normalized.legal_actions if row["code"] != selected_action)):
        raise ValueError("observed bid alternative ownership mismatch")
    prefixes = {}
    for row in worlds:
        world_prefix = ExactTracePrefixV1.build(row["sourcePrefix"])
        if world_prefix != source_prefix:
            expected = source_prefix.to_dict()
            expected.pop("sha256")
            expected.update(deckOrder=list(world_prefix.deck_order), stateAssertions={}, projectAssertions=[])
            if world_prefix != ExactTracePrefixV1.build(expected):
                raise ValueError("observed bid alternative changes the public prefix")
        prefixes[row["action"]] = world_prefix
    normalization_queries = 2 * len({value.sha256 for value in prefixes.values() if value != source_prefix})
    stage_caps = {"branches": limits["maximumBranches"], "conflictsPerSolve": limits["maximumConflicts"],
        "maximumDepth": limits["maximumDepth"], "modelBytes": limits["maximumModelBytes"],
        "queries": attempt["maximumQueries"]}
    strategy_caps = {key: value for key, value in stage_caps.items()
        if key not in {"maximumDepth", "modelBytes"}} | {
            "masterIterations": limits["maximumIterations"], "observations": limits["maximumObservations"]}
    stages = [(selected["preparation"], stage_caps)]
    counts = [*([fallback_attempt["queriesUsed"]] if fallback_attempt is not None else []),
        selected.get("queriesUsed")]
    configured_probe = (ScenarioQualityExecutionBudgetV1.build(execution_budget.to_dict()).bid_threshold_refutation
        if isinstance(execution_budget, ScenarioQualityExecutionBudgetV1) else None)
    probe = attempt.get("thresholdRefutationProbe")
    if configured_probe is not None or probe is not None:
        from .scenario_contracts import build_native_bid_refutation_probe_v1
        if (configured_probe is None or not hashed(probe)
                or probe.get("kind") != "native_bid_threshold_probe_v1"
                or type(probe.get("schemaVersion")) is not int or probe["schemaVersion"] != 1
                or probe.get("status") not in {"incomplete", "threshold_not_refuted", "not_found_within_budget"}
                or probe.get("strategicCoverage") is not False
                or probe.get("sourcePrefixHash") != source_prefix.sha256 or probe.get("rulesHash") != rules.sha256
                or probe.get("selectedAction") != selected_action
                or type(probe.get("requiredThreshold")) is not int
                or probe["requiredThreshold"] != certificate["selectedLowerBound"]):
            raise ValueError("observed bid probe differs from frozen configuration or root")
        configuration = build_native_bid_refutation_probe_v1(probe.get("configuration"),
            maximum_seconds=attempt["maximumSeconds"])
        if "worldPrefix" in probe:
            counterworld = ExactTracePrefixV1.build(probe["worldPrefix"])
            public_prefix = source_prefix.to_dict()
            public_prefix.pop("sha256")
            public_prefix.update(deckOrder=list(counterworld.deck_order), stateAssertions={}, projectAssertions=[])
            if (counterworld.sha256 != probe.get("worldPrefixHash")
                    or counterworld != ExactTracePrefixV1.build(public_prefix)
                    or isinstance(probe.get("upperProof"), Mapping)
                    and probe["upperProof"].get("worldPrefixHash") != counterworld.sha256):
                raise ValueError("observed bid probe counterworld ownership mismatch")
        if (any(configuration[key] != configured_probe[key] for key in
                ("schemaVersion", "worldRankingId", "opponentAuctionAction"))
                or configuration["maximumSeconds"] > configured_probe["maximumSeconds"]
                or configuration["maximumQueries"] > min(configured_probe["maximumQueries"], attempt["maximumQueries"])
                or type(probe.get("queriesCharged")) is not int
                or not 0 <= probe["queriesCharged"] <= configuration["maximumQueries"]
                or probe.get("queryCountComplete") is not (probe["status"] != "incomplete")
                or probe.get("stage") not in {"selection", "normalization", "world_upper",
                    "fresh_verification", "threshold_refutation"}
                or probe["status"] == "incomplete" and probe["queriesCharged"] != configuration["maximumQueries"]):
            raise ValueError("observed bid probe exceeds its frozen allowance or changes accounting")
        if probe["status"] != "incomplete":
            upper = probe.get("upperProof")
            selection_count = (probe.get("selection") or {}).get("queriesUsed")
            upper_count = upper.get("queriesUsed") if isinstance(upper, Mapping) else None
            if (type(selection_count) is not int or selection_count < 1
                    or probe["status"] == "not_found_within_budget" and (
                        probe["stage"] != "selection" or probe["queriesCharged"] != selection_count or upper is not None)
                    or probe["status"] == "threshold_not_refuted" and (
                        probe["stage"] != "world_upper" or not hashed(upper)
                        or type(upper_count) is not int or upper_count < 1
                        or probe["queriesCharged"] != selection_count + 2 + upper_count)):
                raise ValueError("observed bid probe generation accounting mismatch")
        # Inconclusive probe evidence cannot qualify a move; the main proof is freshly reverified below.
        counts.append(probe["queriesCharged"])
    configured_search = (ScenarioQualityExecutionBudgetV1.build(execution_budget.to_dict()).bid_alternative_proof
        if isinstance(execution_budget, ScenarioQualityExecutionBudgetV1) else None)
    search_counts = None
    if configured_search is not None or "alternativeProof" in attempt or "alternativeSearches" in attempt:
        searches = attempt.get("alternativeSearches")
        if (configured_search is None or attempt.get("alternativeProof") != configured_search
                or play_policy is not None or not isinstance(searches, list)
                or len(searches) != len(alternatives)):
            raise ValueError("observed bid alternative search differs from frozen configuration")
        search_counts = []
        for row, search in zip(alternatives, searches, strict=True):
            receipt = row["receipt"]
            if configured_search["schemaVersion"] == 2:
                search_counts.append(_validate_native_observed_bid_refutation_search_v1(search, receipt,
                    configuration=configured_search, source_prefix=source_prefix, rules=rules,
                    action=row["action"], target_team=certificate["targetTeam"], limits=limits,
                    maximum_evidence_bytes=execution_budget.maximum_evidence_bytes))
                continue
            bounds = search.get("limits") if isinstance(search, Mapping) else None
            attempts = search.get("attempts") if isinstance(search, Mapping) else None
            expected = {"worlds": configured_search["maximumWorlds"],
                "queries": configured_search["maximumQueries"], "worldQueries": configured_search["maximumWorldQueries"],
                "iterationsPerWorld": limits["maximumIterations"], "observationsPerWorld": limits["maximumObservations"],
                "branches": limits["maximumBranches"], "modelBytes": limits["maximumModelBytes"],
                "evidenceBytes": execution_budget.maximum_evidence_bytes,
                "conflicts": limits["maximumConflicts"], "depth": limits["maximumDepth"]}
            if (not hashed(search) or search.get("kind") != "native_observed_bid_world_search_v1"
                    or type(search.get("schemaVersion")) is not int or search["schemaVersion"] != 1
                    or search.get("status") != "upper_bound_proven"
                    or search.get("strategicCoverage") is not False or search.get("impossibilityProven") is not False
                    or search.get("sourcePrefixHash") != source_prefix.sha256 or search.get("rulesHash") != rules.sha256
                    or search.get("action") != row["action"] or type(search.get("upperBound")) is not int
                    or search.get("worldRankingId") != configured_search.get("worldRankingId")
                    or search["upperBound"] != configured_search["upperBound"]
                    or not isinstance(bounds, Mapping)
                    or any(type(bounds.get(key)) is not int or not 0 < bounds[key] <= cap for key, cap in expected.items())
                    or any(type(bounds.get(key)) not in {int, float} or not math.isfinite(bounds[key])
                        or not 0 < bounds[key] <= cap for key, cap in {
                            "seconds": configured_search["maximumSeconds"],
                            "worldSeconds": configured_search["maximumWorldSeconds"]}.items())
                    or not isinstance(attempts, list) or not 0 < len(attempts) <= bounds["worlds"]
                    or type(search.get("queriesCharged")) is not int or not 0 < search["queriesCharged"] <= bounds["queries"]
                    or receipt.get("kind") != "native_observed_bid_opponent_policy_upper_v1"
                    or receipt.get("upperBound") != search["upperBound"]):
                raise ValueError("observed bid search receipt exceeds frozen limits or substitutes its cap")
            for index, child in enumerate(attempts):
                if (not isinstance(child, Mapping) or type(child.get("index")) is not int or child["index"] != index
                        or type(child.get("queryAllowance")) is not int or not 0 < child["queryAllowance"] <= bounds["worldQueries"]
                        or type(child.get("queriesCharged")) is not int or not 0 <= child["queriesCharged"] <= child["queryAllowance"]):
                    raise ValueError("observed bid search query accounting mismatch")
                if child.get("status") == "incomplete":
                    if child["queriesCharged"] != child["queryAllowance"]:
                        raise ValueError("observed bid incomplete search must preserve conservative charges")
                elif child.get("status") in {"upper_bound_proven", "opponent_threshold_refuted"}:
                    checked_count = child.get("verificationQueriesUsed", 0)
                    child_counts = [child.get("selection", {}).get("queriesUsed"), 2,
                        child.get("preparation", {}).get("queriesUsed"), child.get("strategy", {}).get("queriesUsed"),
                        checked_count]
                    if (any(type(count) is not int or count < 0 for count in child_counts)
                            or child["queriesCharged"] != sum(child_counts)
                            or child["status"] == "upper_bound_proven" and checked_count < 1):
                        raise ValueError("observed bid search generation/readback accounting mismatch")
                else:
                    raise ValueError("observed bid search contains an unsupported attempt status")
            final = attempts[-1]
            if (final.get("status") != "upper_bound_proven" or final.get("stage") != "fresh_verification"
                    or final.get("preparation") != receipt.get("preparation") or final.get("strategy") != receipt.get("strategy")
                    or final.get("worldPrefix") != certificate["premises"]["alternativeWorlds"][len(search_counts)]["sourcePrefix"]
                    or sum(child["queriesCharged"] for child in attempts) != search["queriesCharged"]):
                raise ValueError("observed bid search substitutes its retained world or proof")
            search_counts.append(search["queriesCharged"])
        if (sum(search_counts) > configured_search["maximumQueries"]
                or configured_search["schemaVersion"] == 1
                and sum(len(search["attempts"]) for search in searches) > configured_search["maximumWorlds"]):
            raise ValueError("observed bid alternative search exceeds its shared query/world allowance")
        counts.extend(search_counts)
    for row in alternatives:
        receipt = row["receipt"]
        if receipt.get("kind") in _NATIVE_OBSERVED_BID_STRATEGY_UPPERS:
            if (set(receipt) != {"schemaVersion", "kind", "strategy", "preparation", "upperBound", "sha256"}
                    or type(receipt["schemaVersion"]) is not int or receipt["schemaVersion"] != 1
                    or type(receipt["upperBound"]) is not int or not hashed(receipt)
                    or not hashed(receipt["preparation"]) or not hashed(receipt["strategy"])):
                raise ValueError("observed opponent policy upper receipt integrity mismatch")
            stages.extend(((receipt["preparation"], stage_caps), (receipt["strategy"], strategy_caps)))
            if search_counts is None:
                counts.extend((receipt["preparation"].get("queriesUsed"), receipt["strategy"].get("queriesUsed")))
        else:
            stages.append((receipt, stage_caps))
            counts.append(receipt.get("queriesUsed"))
    if not native_floor:
        stages.append((selected["strategy"], strategy_caps))
    prior_bounds = _prior_bid_alternative_receipts_v1(prior,
        source_prefix_hash=source_prefix.sha256, rules_hash=rules.sha256,
        maximum_seconds=min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds))
    for stage, ceilings in stages:
        declared = stage.get("limits")
        prior_bound = prior_bounds.get(stage.get("sha256"))
        seconds_ceiling = (prior_bound[2] if prior_bound is not None and stage == prior_bound[1]
            else attempt["maximumSeconds"])
        if (not isinstance(declared, Mapping)
                or any(type(declared.get(key)) is not int or not 0 < declared[key] <= cap
                    for key, cap in ceilings.items())
                or type(declared.get("seconds")) not in {int, float}
                or not 0 < declared["seconds"] <= seconds_ceiling):
            raise ValueError("observed bid stage exceeds its frozen budget")
    checked = certificate.get("executionReceipt", {}).get("queriesUsed")
    reserved = attempt.get("artifactVerificationReserved", False)
    if (any(type(count) is not int or count < 0 for count in [*counts, checked])
            or type(reserved) is not bool
            or type(attempt.get("queriesUsed")) is not int
            or attempt["queriesUsed"] != sum(counts) + checked
            or verification.get("maximumQueries") != (
                attempt["maximumQueries"] - sum(counts) - (normalization_queries if reserved else 0)) // (2 if reserved else 1)
            or attempt["queriesUsed"] + checked + normalization_queries > attempt["maximumQueries"]
            or selection_queries + attempt["maximumQueries"] > native.maximum_queries):
        raise ValueError("observed bid query accounting mismatch")
    deadline = time.monotonic() + maximum_seconds

    def cancelled():
        return time.monotonic() >= deadline or cancel_check is not None and cancel_check()

    if cancelled():
        raise InformationSetCompilationStopped("cancelled", stage="observed_bid_quality")
    expected = {"rulesHash": rules.sha256, "sourcePrefixHash": source_prefix.sha256,
        "exactStateId": normalized.exact_state_id, "selectedAction": selected_action,
        "visibility": "actor_information_set", "completionScope": "selected_bid_current_deal",
        "targetTeam": request.target_team, "legalActions": sorted(row["code"] for row in normalized.legal_actions),
        "wholeRoundCertified": False, "actualActionOptimal": True}
    if (any(certificate.get(key) != value for key, value in expected.items())
            or certificate["premises"].get("sourcePrefix") != source_prefix.to_dict()):
        raise ValueError("observed bid owned certificate context mismatch")
    reuse = certificate["sha256"] in verified_certificates
    normalized_worlds = {source_prefix.sha256: normalized}
    for world_prefix in (() if reuse else prefixes.values()):
        if world_prefix.sha256 not in normalized_worlds:
            if cancelled():
                raise InformationSetCompilationStopped("maximum_seconds", stage="observed_bid_quality")
            normalized_worlds[world_prefix.sha256] = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
                world_prefix, decision_point={"kind": "bid"}, maximum_seconds=deadline - time.monotonic(),
                cancel_check=cancelled)
    if cancelled():
        raise InformationSetCompilationStopped("maximum_seconds", stage="observed_bid_quality")
    graph = copy.deepcopy(certificate) if reuse else reverify_native_observed_bid_dominance_v1(certificate, rules=rules, exact=normalized,
        source_prefix=source_prefix, selected_action=selected_action,
        required_threshold=certificate["selectedLowerBound"], opponent_policy=certificate["opponentPolicy"],
        alternative_worlds={action: (normalized_worlds[value.sha256], value) for action, value in prefixes.items()},
        maximum_seconds=verification["maximumSeconds"], maximum_queries=verification["maximumQueries"],
        maximum_branches=limits["maximumBranches"], maximum_model_bytes=limits["maximumModelBytes"],
        maximum_conflicts=limits["maximumConflicts"], maximum_depth=limits["maximumDepth"],
        maximum_observations=limits["maximumObservations"], opportunity_order=opportunity_order,
        **({"strategy_play_fallback_id": strategy_fallback} if strategy_fallback is not None else {}),
        **({"minimax_budget": budget}
           if "minimaxProofBudgetHash" in selected["strategy"].get("scope", {}) else {}),
        **({"opponent_play_policy_id": opponent_play_policy_id} if play_policy is not None else {}),
        cancel_check=cancelled)
    if cancelled():
        raise InformationSetCompilationStopped("maximum_seconds", stage="observed_bid_quality")
    actions = sorted(row["code"] for row in normalized.legal_actions)
    quality = {"schemaVersion": 1, "authority": "cp_sat", "scope": "compiled_actor_information_set",
        "rootActorRelation": "target", "actualAction": selected_action,
        "actualActionOptimal": True, "regret": 0, "actualValue": None, "bestValue": None,
        "optimalActions": [selected_action], "optimalActionsComplete": False,
        "optimalActionCount": None, "optimalActionCountLowerBound": 1,
        "optimalActionCountUpperBound": len(actions), "qualificationModelHash": graph["sha256"]}
    quality["sha256"] = canonical_sha256(quality)
    model = {"schemaVersion": 1, "kind": "native_observed_bid_quality_model_v1",
        "rootActorSeat": normalized.next_actor, "actionQuality": quality,
        "qualificationModelHash": graph["sha256"]}
    allocation = {"schemaVersion": receipt_version, "kind": "native_observed_bid_quality_context_v1",
        "sourcePrefixHash": source_prefix.sha256, "exactStateId": normalized.exact_state_id,
        "stateHash": normalized.state_hash, "rulesHash": rules.sha256,
        "sourcePrefix": source_prefix.to_dict(), "normalizedState": normalized.to_dict(),
        "normalizationReceipt": normalized.normalization_receipt.to_dict(), "attempt": copy.deepcopy(attempt),
        "generationQueriesUsed": selection_queries + attempt["queriesUsed"] + checked + normalization_queries,
        **({"generationQueryCountUnit": "conservative_budget_charges"}
           if receipt_version == 2 or configured_search is not None or configured_probe is not None else {})}
    allocation["sha256"] = canonical_sha256(allocation)
    proof = {"schemaVersion": 1, "completionStatus": "exact_complete",
        "completionScope": "selected_bid_current_deal", "claimSatisfied": None,
        "selectedAction": selected_action, "winningRootActionCount": None,
        "winningRootActionLowerBound": 0, "winningRootActionUpperBound": len(actions),
        "rootActions": [{"action": action, "value": None, "satisfies": None} for action in actions],
        "identities": {"graphHash": graph["sha256"], "modelHash": canonical_sha256(model),
            "proofRequestHash": request.sha256}}
    proof["sha256"] = canonical_sha256(proof)
    return {"request": request.to_dict(), "proof": proof, "proofModel": model,
        "graph": graph, "allocationModel": allocation}


def validate_native_observed_bid_quality_artifact_v1(artifact, *, rules, request,
        source_prefix_hash, exact_state_id, budget, maximum_conflicts, opportunity_order,
        maximum_seconds, cancel_check=None, opponent_play_policy_id=None, execution_budget=None):
    """Re-normalize the owned prefix and reprove stored observer-visible bid evidence."""
    import time
    from .ortools_exact_state import OrToolsExactStateNormalizerV1
    from .ortools_information_set_compiler import InformationSetCompilationStopped

    if (not isinstance(artifact, Mapping) or not isinstance(rules, RuleBook)
            or not isinstance(request, ScenarioProofRequestV1)
            or not isinstance(budget, ScenarioProofBudgetV1)
            or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
            or maximum_seconds <= 0):
        raise ValueError("observed bid readback requires frozen context and finite time")
    allocation = artifact.get("allocationModel")
    if (not isinstance(allocation, Mapping) or artifact.get("request") != request.to_dict()
            or type(allocation.get("schemaVersion")) is not int or allocation["schemaVersion"] not in {1, 2}
            or allocation.get("sha256") != canonical_sha256({
                key: value for key, value in allocation.items() if key != "sha256"})
            or allocation.get("sourcePrefixHash") != source_prefix_hash
            or allocation.get("exactStateId") != exact_state_id
            or allocation.get("rulesHash") != rules.sha256
            or not isinstance(allocation.get("attempt"), Mapping)
            or artifact.get("graph") != allocation["attempt"].get("certificate")):
        raise ValueError("observed bid readback artifact ownership mismatch")
    prefix = ExactTracePrefixV1.build(allocation.get("sourcePrefix"))
    if prefix.sha256 != source_prefix_hash:
        raise ValueError("observed bid readback source substitution")
    deadline = time.monotonic() + min(maximum_seconds, budget.maximum_wall_seconds, budget.maximum_cpu_seconds)

    def remaining():
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage="observed_bid_readback")
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise InformationSetCompilationStopped("maximum_seconds", stage="observed_bid_readback")
        return seconds

    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=remaining(), cancel_check=cancel_check)
    if (normalized.exact_state_id != exact_state_id
            or normalized.to_dict() != allocation.get("normalizedState")
            or normalized.normalization_receipt.to_dict() != allocation.get("normalizationReceipt")):
        raise ValueError("observed bid readback derived state substitution")
    expected = build_native_observed_bid_quality_artifact_v1(allocation["attempt"], rules=rules,
        source_prefix=prefix, normalized=normalized, request=request, budget=budget,
        maximum_conflicts=maximum_conflicts, opportunity_order=opportunity_order,
        maximum_seconds=remaining(), cancel_check=cancel_check,
        execution_budget=execution_budget,
        receipt_version=allocation["schemaVersion"],
        **({"opponent_play_policy_id": opponent_play_policy_id} if opponent_play_policy_id is not None else {}))
    remaining()
    if canonical_sha256(artifact) != canonical_sha256(expected):
        raise ValueError("observed bid readback differs from fresh native evidence")


def validate_native_bid_dominance_preparations_v1(value: Mapping[str, Any], *, rules,
        source_prefix: ExactTracePrefixV1, state_hash: str, normalized=None) -> None:
    """Bind prepared universes to caller-owned frozen rules and state."""
    from .ortools_information_set_proof import native_bid_branch_specs_v1
    specs = native_bid_branch_specs_v1(rules, Seat.parse(source_prefix.dealer))
    partition = [None if row is None else {"contract": row[0].value,
        "trump": row[1].name.lower() if row[1] is not None else None,
        "firstLeader": row[2].name.lower(), "locked": row[3]} for row in specs]
    common = {"rulesHash": rules.sha256, "rootHash": state_hash, "visibility": "complete_state",
        "matchScore": list(source_prefix.match_score), "continuationBoundary": "current_deal_completion",
        "nextDealAnalyzed": False, "partition": partition}
    if "escalationOpportunityOrder" in value:
        order = value["escalationOpportunityOrder"]
        if (not isinstance(order, list) or len(order) != len(Seat)
                or any(type(seat) is not str for seat in order)
                or set(order) != {seat.name.lower() for seat in Seat}):
            raise ValueError("bid dominance escalation order is invalid")
        common["escalationOpportunityOrder"] = order
    if value.get("targetTeam") not in {"north_south", "west_east"}:
        raise ValueError("bid dominance preparation target is invalid")
    opponent = "west_east" if value["targetTeam"] == "north_south" else "north_south"
    policy = value.get("opponentStrategy", {})
    if not isinstance(policy, Mapping) or not {"auction_action", "escalation_action"} <= policy.keys():
        raise ValueError("bid dominance preparation policy missing")
    play_policy_id = value.get("opponentPlayPolicyId")
    streamed_kinds = {"native_bid_streamed_alternative_preparation_v1", "native_bid_streamed_world_upper_v1"}
    preparations = [(value.get("selectedValueProof", {}).get("preparation"), value.get("selectedAction"), None)]
    preparations.extend((row.get("preparation"), row.get("action"), {"team": opponent,
        **{key: policy[key] for key in ("auction_action", "escalation_action")}}) for row in value.get("alternatives", []))
    depth_limit = value.get("limits", {}).get("maximumDepth")
    if depth_limit is not None and (type(depth_limit) is not int or depth_limit < 1):
        raise ValueError("bid dominance depth limit is invalid")
    resumed = value.get("kind") == "native_bid_selected_action_dominance_v4"
    if resumed:
        validate_native_bid_dominance_resume_basis_v1(value.get("incumbentAttempt"), rules=rules,
            source_prefix=source_prefix, normalized=normalized, selected_action=value["selectedAction"],
            opponent_policy=policy, opponent_play_policy_id=play_policy_id, limits=value["limits"])
        if value["alternatives"] != value["incumbentAttempt"]["alternatives"]:
            raise ValueError("resumed bid substitutes its retained alternatives")
    for receipt, action, nonplay in preparations:
        if isinstance(receipt, Mapping) and receipt.get("kind") == "native_bid_streamed_world_upper_v1":
            if (value.get("kind") not in {"native_bid_selected_action_dominance_v2", "native_bid_selected_action_dominance_v3", "native_bid_selected_action_dominance_v4"}
                    or nonplay is None or (value.get("kind") == "native_bid_selected_action_dominance_v2"
                    and play_policy_id is None) or policy.get("project_actions")):
                raise ValueError("bid dominance world upper requires a paired interval alternative")
            validate_native_bid_world_upper_v1(receipt, rules=rules, exact=normalized,
                source_prefix=source_prefix, action=action, opponent_policy=policy,
                opponent_play_policy_id=play_policy_id, maximum_depth=depth_limit,
                maximum_model_bytes=value.get("limits", {}).get("maximumModelBytesPerPreparation"),
                opportunity_order=value.get("escalationOpportunityOrder"))
            if normalized.state_hash != state_hash:
                raise ValueError("bid dominance world upper state substitution")
            continue
        expected = {**common, "selectedRootAction": action,
            **({"nonplayPolicy": nonplay} if nonplay is not None else {})}
        if nonplay is not None and play_policy_id is not None:
            from .ortools_play_tail import native_observer_safe_policy_identity_v1
            if not isinstance(receipt, Mapping) or receipt.get("kind") != "native_bid_streamed_alternative_preparation_v1":
                raise ValueError("bid dominance paired policy lacks a streamed bound")
            expected["opponentPlayPolicy"] = native_observer_safe_policy_identity_v1(
                play_policy_id, Seat.parse("north" if opponent == "north_south" else "west").team)
        if (isinstance(receipt, Mapping) and isinstance(receipt.get("scope"), Mapping)
                and "ownershipCompaction" in receipt["scope"]):
            if receipt["scope"]["ownershipCompaction"] != "purchase_recipient_invariant_v1":
                raise ValueError("bid dominance ownership compaction is invalid")
            expected["ownershipCompaction"] = "purchase_recipient_invariant_v1"
        if (isinstance(receipt, Mapping) and isinstance(receipt.get("scope"), Mapping)
                and receipt["scope"].get("compactKnownInitialOwnership") is True):
            expected["compactKnownInitialOwnership"] = True
        if (not isinstance(receipt, Mapping) or receipt.get("constructionComplete") is not True
                or receipt.get("scope") != expected or receipt.get("scopeHash") != canonical_sha256(expected)):
            raise ValueError("bid dominance preparation scope mismatch")
        if receipt.get("kind") == "native_bid_streamed_alternative_preparation_v1":
            alternatives = [row for row in value.get("alternatives", []) if row.get("action") == action]
            if (value.get("kind") not in {"native_bid_selected_action_dominance_v1", "native_bid_selected_action_dominance_v2", "native_bid_selected_action_dominance_v3"} or nonplay is None
                    or policy.get("project_actions") or len(alternatives) != 1):
                raise ValueError("streamed bid preparation requires an owned nonplay alternative")
            _validate_streamed_bid_preparation(receipt, alternatives[0],
                maximum_model_bytes=value.get("limits", {}).get("maximumModelBytesPerPreparation"))
        elif "proofQueriesUsed" in receipt or any("alternativeQuery" in row
                for row in receipt.get("branches", []) if isinstance(row, Mapping)):
            raise ValueError("streamed bid preparation kind mismatch")
        branches = receipt.get("branches")
        if (not isinstance(branches, list) or len(branches) != len(partition)
                or any(not isinstance(row, Mapping) or type(row.get("partitionIndex")) is not int
                    or row["partitionIndex"] != index or type(row.get("retained")) is not bool
                    for index, row in enumerate(branches))):
            raise ValueError("bid dominance preparation partition coverage mismatch")
        for row in branches:
            if not _native_bid_play_binding_matches_v1(row, expected.get("opponentPlayPolicy"),
                    partition[row["partitionIndex"]] is not None):
                raise ValueError("bid dominance alternative play policy binding mismatch")
            if row["retained"]:
                if nonplay is not None:
                    alternative = next(item for item in value["alternatives"] if item["action"] == action)
                    owned = [query for query in alternative.get("queries", [])
                        if isinstance(query, Mapping) and query.get("sourceModelHash") == row.get("sourceModelHash")]
                    expected_policy = {"team": opponent, "auctionAction": policy["auction_action"],
                        "escalationAction": policy["escalation_action"],
                        "auctionPrefixLength": len(source_prefix.auction_actions),
                        "scope": "all_future_nonplay_information_sets"}
                    if policy.get("project_actions"):
                        expected_policy["projectActions"] = policy["project_actions"]
                    cell = partition[row["partitionIndex"]]
                    expected_scope = {"rulesHash": rules.sha256, "matchScore": list(source_prefix.match_score),
                        "allPassedCovered": cell is None, "strategicCoverage": False,
                        **({"kind": "native_all_pass_deal_branch_v1", "continuationBoundary": "current_deal_completion",
                            "nextDealAnalyzed": False} if cell is None else {
                            "kind": "native_purchased_round_branch_v1", "contract": cell["contract"],
                            "trumpSuit": cell["trump"], "firstLeader": cell["firstLeader"], "locked": cell["locked"],
                            "escalationOpportunityOrder": value.get("escalationOpportunityOrder", "symbolic_per_window")})}
                    expected_bindings = [("native_constant_" + kind + "_policy_binding_v1", selected)
                        for kind, selected in (("auction", policy["auction_action"]),
                            ("escalation", policy["escalation_action"]))
                        if selected is not None and (kind != "escalation" or cell is not None)]
                    expected_bindings += [("native_constant_project_policy_binding_v1", selected)
                        for selected in (policy.get("project_actions") or {}).values()]
                    query = owned[0] if len(owned) == 1 else {}
                    branch_scope = query.get("scope")
                    outcome = query.get("query")
                    if (query.get("policy") != expected_policy or query.get("rulesHash") != rules.sha256
                            or query.get("targetTeam") != opponent or not isinstance(branch_scope, Mapping)
                            or any(branch_scope.get(key) != item for key, item in expected_scope.items())
                            or branch_scope.get("sha256") != canonical_sha256({k: v for k, v in branch_scope.items() if k != "sha256"})
                            or not isinstance(outcome, Mapping) or outcome.get("scope") != branch_scope
                            or sorted((binding.get("kind"), binding.get("action")) for binding in query.get("bindings", []))
                                != sorted(expected_bindings)
                            or any(binding.get("rulesHash") != rules.sha256 or binding.get("status") != "bound"
                                for binding in query.get("bindings", []))):
                        raise ValueError("bid dominance alternative policy or branch ownership mismatch")
                if depth_limit is not None:
                    query = row.get("depthQuery")
                    if (receipt.get("limits", {}).get("maximumDepth") != depth_limit
                            or not isinstance(query, Mapping)
                            or query.get("sha256") != canonical_sha256({k: v for k, v in query.items() if k != "sha256"})
                            or query.get("kind") != "native_continuation_depth_query_v1"
                            or query.get("status") != "depth_bound_proven"
                            or query.get("solverStatus") != "infeasible"
                            or type(query.get("maximumDepth")) is not int or query["maximumDepth"] != depth_limit
                            or query.get("sourceModelHash") != row.get("sourceModelHash")
                            or query.get("unit") != "remaining_decision_actions"
                            or query.get("includes") != ["auction", "escalation", "project", "play"]):
                        raise ValueError("bid dominance retained depth bound is not certified")
                continue
            reachability = row.get("reachability")
            if (not isinstance(reachability, Mapping) or reachability.get("solverStatus") != "infeasible"
                    or any(not isinstance(binding, Mapping) or binding.get("status") != "bound"
                        for binding in reachability.get("policyBindings", []))):
                raise ValueError("bid dominance omitted partition is not certified")
    if any(receipt.get("kind") in streamed_kinds
            for receipt, _, _ in preparations):
        total = value.get("selectedValueProof", {}).get("queriesUsed")
        if type(total) is not int or total < 0:
            raise ValueError("streamed bid total query accounting mismatch")
        for alternative in value["alternatives"]:
            preparation = alternative["preparation"]
            used = preparation.get("queriesUsed")
            if type(used) is not int or used < 0:
                raise ValueError("streamed bid total query accounting mismatch")
            if not resumed:
                total += used
            if preparation.get("kind") not in streamed_kinds:
                total += sum(len(query["bindings"]) + 1 for query in alternative["queries"])
        if type(value.get("queriesUsed")) is not int or value["queriesUsed"] != total + 1:
            raise ValueError("streamed bid total query accounting mismatch")


def validate_native_bid_dominance_evidence_v1(
    value: Mapping[str, Any], *, rules, source_prefix: ExactTracePrefixV1,
    exact_state_id: str, state_hash: str, target_team: str,
    selected_action: str, legal_actions: tuple[str, ...],
    native_incumbent_models=None, normalized=None,
) -> None:
    """Validate the selected-bid certificate, not whole-round or tie-count claims.

    Context must come from the caller's frozen bundle and normalized state.
    """
    validate_native_bid_dominance_identity_v1(value, rules_hash=rules.sha256,
        source_prefix_hash=source_prefix.sha256, exact_state_id=exact_state_id,
        state_hash=state_hash, target_team=target_team, selected_action=selected_action)
    validate_native_bid_dominance_claim_v1(value, legal_actions=legal_actions,
        native_incumbent_models=native_incumbent_models)
    validate_native_bid_dominance_preparations_v1(value, rules=rules,
        source_prefix=source_prefix, state_hash=state_hash, normalized=normalized)


def build_native_bid_dominance_quality_artifact_v1(
    certificate: Mapping[str, Any], *, rules: RuleBook, source_prefix: ExactTracePrefixV1,
    normalized, request: ScenarioProofRequestV1, native_incumbent_models=None,
) -> dict[str, Any]:
    """Bind selected-action evidence without claiming the request's outcome/count."""
    from .ortools_exact_state import OrToolsNormalizedExactStateV1, _effective_prefix_hash
    if (not isinstance(rules, RuleBook) or not isinstance(source_prefix, ExactTracePrefixV1)
            or not isinstance(normalized, OrToolsNormalizedExactStateV1)
            or not isinstance(request, ScenarioProofRequestV1)):
        raise TypeError("bid quality requires typed frozen context")
    normalized.require_integrity()
    if (normalized.decision_kind != "bid" or normalized.next_actor is None
            or normalized.rules_hash != rules.sha256
            or normalized.prefix_hash != _effective_prefix_hash(source_prefix)
            or request.mode != "exact" or request.visibility != "complete_state"
            or request.utility != "awarded_point_margin"
            or request.target_team != Seat.parse(normalized.next_actor).team.name.lower()
            or request.selected_action not in {None, certificate.get("selectedAction")}):
        raise ValueError("bid quality request or normalized context mismatch")
    actions = tuple(row["code"] for row in normalized.legal_actions)
    selected_action = certificate.get("selectedAction")
    prior = certificate.get("priorSelectionAttempts")
    schedule = certificate.get("selectionSchedule")
    if schedule not in (None, "incumbent_then_deep_v1"):
        raise ValueError("unsupported generated bid selection schedule")
    if request.selected_action is None:
        if (selected_action not in actions or not isinstance(prior, list)
                or len(prior) not in ({actions.index(selected_action), len(actions) + actions.index(selected_action)}
                    if schedule is not None else {actions.index(selected_action)})
                or any(not isinstance(row, Mapping)
                    or set(row) != {"action", "queryAllowance", "queriesCharged", "receipt"}
                    or row.get("action") != actions[index % len(actions)]
                    or type(row.get("queryAllowance")) is not int or row["queryAllowance"] < 1
                    or type(row.get("queriesCharged")) is not int
                    or row["queriesCharged"] != row["queryAllowance"]
                    or not isinstance(row.get("receipt"), Mapping)
                    or row["receipt"].get("status") != "incomplete"
                    or row["receipt"].get("sha256") != canonical_sha256({
                        key: value for key, value in row["receipt"].items() if key != "sha256"})
                    for index, row in enumerate(prior))):
            raise ValueError("generated bid selection attempts or resource charges mismatch")
    elif prior is not None or schedule is not None:
        raise ValueError("recorded bid cannot substitute generated selection attempts")
    validate_native_bid_dominance_evidence_v1(certificate, rules=rules, source_prefix=source_prefix,
        exact_state_id=normalized.exact_state_id, state_hash=normalized.state_hash,
        target_team=request.target_team, selected_action=selected_action, legal_actions=actions,
        native_incumbent_models=native_incumbent_models, normalized=normalized)
    graph = copy.deepcopy(dict(certificate))
    model = {**copy.deepcopy(dict(certificate["qualificationModel"])),
        "kind": "native_bid_selected_action_quality_model_v1", "rootActorSeat": normalized.next_actor,
        "actionQuality": copy.deepcopy(dict(certificate["actionQuality"]))}
    # The qualification model's own hash remains in the certificate, not on this envelope.
    model.pop("sha256")
    allocation = {"schemaVersion": 1, "kind": "native_complete_state_quality_v1",
        "sourcePrefixHash": source_prefix.sha256, "exactStateId": normalized.exact_state_id,
        "stateHash": normalized.state_hash, "rulesHash": rules.sha256,
        "sourcePrefix": source_prefix.to_dict(), "normalizedState": normalized.to_dict(),
        "normalizationReceipt": normalized.normalization_receipt.to_dict()}
    allocation["sha256"] = canonical_sha256(allocation)
    proof = {"schemaVersion": 1, "completionStatus": "exact_complete",
        "completionScope": "selected_bid_current_deal", "claimSatisfied": None,
        "selectedAction": selected_action, "winningRootActionCount": None,
        "winningRootActionLowerBound": 0, "winningRootActionUpperBound": len(actions),
        "rootActions": [{"action": action, "value": None, "satisfies": None} for action in sorted(actions)],
        "identities": {"graphHash": graph["sha256"], "modelHash": canonical_sha256(model),
            "proofRequestHash": request.sha256}}
    proof["sha256"] = canonical_sha256(proof)
    return {"request": request.to_dict(), "proof": proof, "proofModel": model,
        "graph": graph, "allocationModel": allocation}


def validate_native_bid_dominance_budget_v1(certificate: Mapping[str, Any], budget: ScenarioProofBudgetV1,
        *, maximum_conflicts: int, opportunity_order: tuple[str, ...] | None, execution_budget=None,
        _incomplete_incumbent=False) -> None:
    """Compare declared native limits with the caller-owned frozen proof tier."""
    if not isinstance(budget, ScenarioProofBudgetV1) or budget.native_observed is None:
        raise ValueError("selected bid quality requires a frozen native proof budget")
    if type(maximum_conflicts) is not int or maximum_conflicts < 1:
        raise ValueError("selected bid quality requires a frozen conflict budget")
    if (not isinstance(opportunity_order, (tuple, list)) or len(opportunity_order) != len(Seat)
            or any(type(seat) is not str for seat in opportunity_order)
            or set(opportunity_order) != {seat.name.lower() for seat in Seat}
            or certificate.get("escalationOpportunityOrder") != list(opportunity_order)):
        raise ValueError("selected bid quality differs from frozen escalation order")
    limits = certificate.get("limits")
    native = budget.native_observed
    caps = {"maximumDepth": min(budget.maximum_depth, budget.maximum_horizon_actions),
        "maximumIterations": native.maximum_iterations, "maximumQueries": native.maximum_queries,
        "maximumObservations": native.maximum_observations,
        "maximumBranchesPerPreparation": budget.maximum_state_space,
        "maximumModelBytesPerPreparation": budget.maximum_cache_bytes,
        "maximumConflictsPerSolve": maximum_conflicts,
        "workersPerSolve": min(1, budget.maximum_workers)}
    if not isinstance(limits, Mapping) or any(type(limits.get(key)) is not int
            or not 0 < limits[key] <= cap for key, cap in caps.items()):
        raise ValueError("selected bid quality exceeds its frozen proof limits")
    seconds = limits.get("maximumSeconds")
    prior = certificate.get("priorSelectionAttempts", [])
    selection_attempt = certificate.get("incumbentAttempt")
    selection_limit = (selection_attempt.get("limits", limits).get("maximumQueries")
        if isinstance(selection_attempt, Mapping) else limits["maximumQueries"])
    if (not isinstance(prior, list) or any(not isinstance(row, Mapping)
            or type(row.get("queriesCharged")) is not int or row["queriesCharged"] < 1 for row in prior)
            or type(selection_limit) is not int
            or sum(row["queriesCharged"] for row in prior) + selection_limit > native.maximum_queries):
        raise ValueError("generated bid selection exceeds its shared query budget")
    if (type(seconds) not in {int, float} or not math.isfinite(seconds)
            or not 0 < seconds <= min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds)
            or type(certificate.get("queriesUsed")) is not int
            or not 0 <= certificate["queriesUsed"] <= limits["maximumQueries"]):
        raise ValueError("selected bid quality exceeds its frozen execution limits")
    incumbent = None
    opponent_play_policy_id = None
    if execution_budget is not None:
        if not isinstance(execution_budget, ScenarioQualityExecutionBudgetV1):
            raise ValueError("selected bid quality requires frozen execution configuration")
        frozen_execution = ScenarioQualityExecutionBudgetV1.build(execution_budget.to_dict())
        incumbent = frozen_execution.bid_incumbent_proof
        opponent_play_policy_id = frozen_execution.bid_opponent_play_policy_id
    interval = certificate.get("kind") in {"native_bid_selected_action_dominance_v2", "native_bid_selected_action_dominance_v3"}
    resumed = certificate.get("kind") == "native_bid_selected_action_dominance_v4"
    attempt = certificate.get("incumbentAttempt")
    if "sharedDecisionMaximumSeconds" in limits and "incumbentAttempt" not in certificate:
        raise ValueError("selected bid shared budget requires an incumbent fallback owner")
    if "incumbentAttempt" in certificate:
        if (incumbent is None or interval or _incomplete_incumbent
                or certificate.get("kind") not in {"native_bid_selected_action_dominance_v1", "native_bid_selected_action_dominance_v4"}
                or not isinstance(attempt, Mapping) or "incumbentAttempt" in attempt
                or attempt.get("kind") not in {"native_bid_selected_action_dominance_v2", "native_bid_selected_action_dominance_v3"}
                or attempt.get("status") != "incomplete"
                or attempt.get("sha256") != canonical_sha256({key: value for key, value in attempt.items() if key != "sha256"})
                or any(attempt.get(key) != certificate.get(key) for key in (
                    "rulesHash", "sourcePrefixHash", "exactStateId", "stateHash", "targetTeam",
                    "selectedAction", "legalActions", "visibility", "utility", "completionScope", "opponentStrategy"))):
            raise ValueError("selected bid fallback substitutes its incumbent owner")
        validate_native_bid_dominance_budget_v1(attempt, budget, maximum_conflicts=maximum_conflicts,
            opportunity_order=opportunity_order, execution_budget=execution_budget, _incomplete_incumbent=True)
        shared_seconds = limits.get("sharedDecisionMaximumSeconds")
        if "sharedDecisionMaximumSeconds" in limits and (
                type(shared_seconds) not in {int, float} or not math.isfinite(shared_seconds)
                or shared_seconds != attempt["limits"]["maximumSeconds"]):
            raise ValueError("selected bid fallback substitutes its shared deadline")
        # Legacy receipts reserved the full incumbent cap. New receipts declare
        # one decision-wide deadline; the runtime deducts actual elapsed time.
        fallback_cap = (shared_seconds if shared_seconds is not None else
            attempt["limits"]["maximumSeconds"] - incumbent["maximumSeconds"])
        if (seconds > fallback_cap
                or limits["maximumQueries"] > attempt["limits"]["maximumQueries"] - attempt["queriesUsed"]
                or certificate["queriesUsed"] + attempt["queriesUsed"] > attempt["limits"]["maximumQueries"]):
            raise ValueError("selected bid fallback exceeds its shared budget")
        incumbent = None
    if resumed and (not isinstance(attempt, Mapping)
            or attempt.get("kind") != "native_bid_selected_action_dominance_v3"
            or certificate.get("opponentPlayPolicyId") != opponent_play_policy_id
            or certificate.get("opponentPlayPolicyId") != attempt.get("opponentPlayPolicyId")):
        raise ValueError("resumed bid lacks its configured alternative basis")
    if interval != (incumbent is not None):
        raise ValueError("selected bid quality differs from frozen incumbent configuration")
    if _incomplete_incumbent and (not interval or certificate.get("status") != "incomplete"):
        raise ValueError("selected bid incomplete receipt must be an incumbent attempt")
    if interval and (certificate.get("opponentPlayPolicyId") != opponent_play_policy_id
            or ("opponentPlayPolicyId" in certificate) != (opponent_play_policy_id is not None)):
        raise ValueError("selected bid quality differs from frozen opponent play policy")
    selected_proof = certificate.get("selectedValueProof")
    selected_strategy = selected_proof.get("strategy") if isinstance(selected_proof, Mapping) else None
    selected_scope = selected_strategy.get("scope") if isinstance(selected_strategy, Mapping) else None
    ceiling = selected_scope.get("opponentCeilingPolicy") if isinstance(selected_scope, Mapping) else None
    unavailable_ceiling = selected_scope.get("opponentCeilingUnavailable") if isinstance(selected_scope, Mapping) else None
    if resumed:
        if ceiling is not None or unavailable_ceiling is not None:
            raise ValueError("resumed bid threshold must not substitute exact-value ceiling evidence")
    elif opponent_play_policy_id is not None and not interval:
        unavailable_matches = (isinstance(unavailable_ceiling, Mapping)
            and unavailable_ceiling == {"reason": "multiple_contract_branches", "policy": {
                "auction_action": "pass", "escalation_action": "decline",
                "native_play_policy_id": opponent_play_policy_id}}
            and ceiling is None and "cooperativeCeiling" not in selected_scope
            and isinstance(selected_scope.get("branches"), list) and len(selected_scope["branches"]) > 1)
        if not unavailable_matches and (not isinstance(ceiling, Mapping)
                or ceiling.get("native_play_policy_id") != opponent_play_policy_id):
            raise ValueError("selected bid quality differs from frozen opponent play policy")
    elif isinstance(ceiling, Mapping) and "native_play_policy_id" in ceiling:
        raise ValueError("selected bid quality has an unconfigured opponent play policy")
    elif unavailable_ceiling is not None:
        raise ValueError("selected bid quality has an unconfigured unavailable ceiling")
    if interval:
        team = certificate.get("targetTeam")
        if team not in ("north_south", "west_east"):
            raise ValueError("selected bid quality incumbent team is invalid")
        if incumbent["schemaVersion"] == 2:
            policy_matches = (certificate.get("incumbentNativePlayPolicyId") == incumbent["nativePlayPolicyId"]
                and certificate.get("incumbentNonplayPolicy") == incumbent.get("completeStateNonplayPolicy")
                and "incumbentPlayPolicies" not in certificate)
        else:
            policies = {seat: raw for seat, raw in incumbent["playPolicies"].items()
                if Seat.parse(seat).team.name.lower() == team}
            policy_matches = certificate.get("incumbentPlayPolicies") == policies and "incumbentNativePlayPolicyId" not in certificate
        if (not policy_matches
                or limits.get("incumbentMaximumSeconds") != incumbent["maximumSeconds"]
                or limits.get("incumbentMaximumQueries") != incumbent["maximumQueries"]):
            raise ValueError("selected bid quality differs from frozen incumbent policy or allowances")
        allowance = limits.get("incumbentMaximumSeconds")
        query_allowance = limits.get("incumbentMaximumQueries")
        selected_queries = selected_proof.get("queriesUsed") if isinstance(selected_proof, Mapping) else None
        selected_seconds = selected_proof.get("maximumSeconds") if isinstance(selected_proof, Mapping) else None
        if (type(allowance) not in {int, float} or not math.isfinite(allowance)
                or allowance <= 0
                or type(query_allowance) is not int or not 0 < query_allowance < limits["maximumQueries"]
                or not _incomplete_incumbent and (type(selected_seconds) not in {int, float}
                    or not math.isfinite(selected_seconds) or not 0 < selected_seconds <= min(allowance, seconds)
                    or type(selected_queries) is not int or not 0 <= selected_queries <= query_allowance)):
            raise ValueError("selected bid quality exceeds its frozen incumbent allowance")


def validate_native_bid_dominance_quality_artifact_v1(
    artifact: Mapping[str, Any], *, rules: RuleBook, request: ScenarioProofRequestV1,
    source_prefix_hash: str, exact_state_id: str,
    native_incumbent_models=None,
) -> None:
    """Rebuild the owned envelope; never reinterpret it as an exhaustive proof."""
    from .ortools_exact_state import OrToolsNormalizedExactStateV1, OrToolsExactNormalizationReceiptV1
    if not isinstance(rules, RuleBook):
        raise ValueError("selected bid quality requires frozen rules")
    allocation = artifact.get("allocationModel")
    if not isinstance(allocation, Mapping):
        raise ValueError("selected bid quality allocation missing")
    source = ExactTracePrefixV1.build(allocation.get("sourcePrefix"))
    raw = allocation.get("normalizedState")
    if (source.sha256 != source_prefix_hash or not isinstance(raw, Mapping)
            or raw.get("exactStateId") != exact_state_id or type(raw.get("schemaVersion")) is not int
            or raw["schemaVersion"] != 1 or not isinstance(raw.get("legalActions"), list)
            or not isinstance(allocation.get("normalizationReceipt"), Mapping)):
        raise ValueError("selected bid quality normalized ownership mismatch")
    try:
        normalized = OrToolsNormalizedExactStateV1(
            exact_state_id=raw["exactStateId"], prefix_hash=raw["prefixHash"], rules_hash=raw["rulesHash"],
            state_hash=raw["stateHash"], phase=raw["phase"], decision_kind=raw["decisionKind"],
            next_actor=raw["nextActor"], legal_actions=tuple(raw["legalActions"]), state=raw["state"],
            project_declarations_verified=raw["projectDeclarationsVerified"],
            project_assertions_verified=raw["projectAssertionsVerified"], event_chain_hash=raw["eventChainHash"],
            sha256=raw["sha256"], normalization_receipt=OrToolsExactNormalizationReceiptV1(allocation["normalizationReceipt"]))
        expected = build_native_bid_dominance_quality_artifact_v1(artifact["graph"], rules=rules,
            source_prefix=source, normalized=normalized, request=request,
            native_incumbent_models=native_incumbent_models)
    except (KeyError, TypeError) as exc:
        raise ValueError("selected bid quality artifact is malformed") from exc
    if canonical_sha256(artifact) != canonical_sha256(expected):
        raise ValueError("selected bid quality artifact binding mismatch")


def reverify_native_bid_dominance_quality_v1(artifact, *, rules, request, budget,
        maximum_conflicts, execution_budget, opportunity_order, maximum_seconds,
        cancel_check=None):
    """Re-solve retained bid bounds; envelope checks are not solver evidence."""
    import time
    from .ortools_exact_state import OrToolsExactStateNormalizerV1
    from .ortools_information_set_compiler import InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1

    graph = artifact["graph"]
    if not isinstance(request, ScenarioProofRequestV1) or request.budget_profile != budget.profile_id:
        raise ValueError("selected bid readback request budget substitution")
    validate_native_bid_dominance_budget_v1(graph, budget,
        maximum_conflicts=maximum_conflicts, opportunity_order=opportunity_order,
        execution_budget=execution_budget)
    if (type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
            or not 0 < maximum_seconds <= min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds)):
        raise ValueError("selected bid readback requires a governed deadline")
    deadline, cpu_end = time.monotonic() + maximum_seconds, time.process_time() + maximum_seconds
    limits, used = graph["limits"], 0

    def remaining():
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage="native_bid_quality_readback")
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise InformationSetCompilationStopped("maximum_seconds", stage="native_bid_quality_readback")
        cpu_seconds = cpu_end - time.process_time()
        if cpu_seconds <= 0:
            raise InformationSetCompilationStopped("maximum_cpu_seconds", stage="native_bid_quality_readback")
        return min(seconds, cpu_seconds)

    def charge(count):
        nonlocal used
        used += count
        if used > limits["maximumQueries"]:
            raise InformationSetCompilationStopped("maximum_queries", stage="native_bid_quality_readback")

    def available():
        if used >= limits["maximumQueries"]:
            raise InformationSetCompilationStopped("maximum_queries", stage="native_bid_quality_readback")
        return limits["maximumQueries"] - used

    allocation = artifact["allocationModel"]
    prefix = ExactTracePrefixV1.build(allocation["sourcePrefix"])
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix,
        decision_point={"kind": "bid"}, maximum_seconds=remaining(), cancel_check=cancel_check)
    if (exact.to_dict() != allocation["normalizedState"]
            or exact.normalization_receipt.to_dict() != allocation["normalizationReceipt"]):
        raise ValueError("selected bid readback derived state substitution")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    target = Seat.parse(exact.next_actor).team
    strategy = graph["selectedValueProof"]["strategy"]
    selected_floor = strategy["lowerBound"]
    native_models = None
    common = dict(maximum_branches=limits["maximumBranchesPerPreparation"],
        maximum_model_bytes=limits["maximumModelBytesPerPreparation"],
        maximum_conflicts=limits["maximumConflictsPerSolve"], maximum_depth=limits["maximumDepth"],
        opportunity_order=opportunity_order, cancel_check=cancel_check)

    def require_absence(receipt):
        query = receipt.get("query") or {}
        charge(len(receipt.get("bindings", [])) + int(bool(query)))
        remaining()
        if receipt.get("status") == "queried" and query.get("status") == "witness_found":
            raise ValueError("selected bid readback native bound refuted")
        if (receipt.get("status") != "queried" or query.get("status") != "no_matching_outcome"
                or query.get("solverStatus") != "infeasible"):
            raise InformationSetCompilationStopped("native_bound_not_verified", stage="native_bid_quality_readback",
                evidence={"query": receipt})

    if graph["kind"] == "native_bid_selected_action_dominance_v4":
        preparation = graph["selectedValueProof"]["preparation"]
        models, rebuilt = compiler._prepare_native_bid_branches(exact, prefix,
            visibility="complete_state", selected_root_action=graph["selectedAction"],
            compact_purchase_invariant_ownership=True,
            maximum_seconds=remaining(), maximum_queries=available(), **common)
        if ([row.get("sourceModelHash") for row in rebuilt["branches"] if row["retained"]]
                != [row.get("sourceModelHash") for row in preparation["branches"] if row["retained"]]):
            raise ValueError("resumed bid readback selected model substitution")
        maximum = max(max(map(int, variable.proto.domain)) for model in models
            for variable in model._native_round_query_context()[1])
        actor = Seat.parse(exact.next_actor)
        observation = models[0]._auction_observation_hash(actor,
            [compiler._card(card) for card in exact.state["hands"][actor.name.lower()]],
            compiler._card(exact.state["purchaseCard"]),
            [row["action"] + (":" + row["trumpSuit"] if row.get("trumpSuit") is not None else "")
             for row in prefix.auction_actions])
        scope = {"kind": "native_symbolic_round_policy_scope_v1", "rulesHash": rules.sha256,
            "targetTeam": target.name.lower(), "coverage": "supplied_deal_branches_only",
            "utility": "awarded_point_margin", "utilityBounds": [-maximum, maximum],
            "requiredThreshold": selected_floor, "objective": "strategy_threshold_satisfaction",
            "branches": sorted([{"modelHash": model._model_hash(search_branching="automatic"),
                "scope": dict(model._native_round_query_context()[0])} for model in models], key=lambda row: row["modelHash"]),
            "fixedRoot": {"observationHash": observation,
                "legalActions": sorted(row["code"] for row in exact.legal_actions),
                "selectedAction": graph["selectedAction"]},
            # Older single-branch receipts omitted this search aid. Either
            # form still requires fresh verification of every owned proof.
            **({"counterexampleShortening": True} if len(models) > 1
               or strategy.get("scope", {}).get("counterexampleShortening") is True else {})}
        _reverify_observed_bid_policy_threshold(strategy, rebuilt, models, compiler=compiler, scope=scope,
            target=target, required_threshold=selected_floor, deadline=time.monotonic() + remaining(),
            maximum_queries=available(), maximum_observations=limits["maximumObservations"],
            maximum_conflicts=maximum_conflicts, cancel_check=cancel_check, on_queries_used=charge)
        del models
    elif graph["kind"] in {"native_bid_selected_action_dominance_v2", "native_bid_selected_action_dominance_v3"}:
        preparation = graph["selectedValueProof"]["preparation"]
        models, rebuilt = compiler._prepare_native_bid_branches(exact, prefix,
            visibility="complete_state", selected_root_action=graph["selectedAction"],
            compact_known_initial_ownership=preparation["scope"].get("compactKnownInitialOwnership", False),
            **({"compact_purchase_invariant_ownership": True}
               if preparation["scope"].get("ownershipCompaction") == "purchase_recipient_invariant_v1" else {}),
            maximum_seconds=remaining(), maximum_queries=available(), **common)
        charge(rebuilt["queriesUsed"])
        if ([row.get("sourceModelHash") for row in rebuilt["branches"] if row["retained"]]
                != [row.get("sourceModelHash") for row in preparation["branches"] if row["retained"]]):
            raise ValueError("selected bid readback selected model substitution")
        native_id = graph.get("incumbentNativePlayPolicyId")
        nonplay = graph.get("incumbentNonplayPolicy")
        if native_id is not None:
            native_models = tuple(models)
            validate_floor = (validate_native_incumbent_threshold_floor_v1
                if graph["kind"] == "native_bid_selected_action_dominance_v3" else validate_native_incumbent_lower_bound_v2)
            validate_floor(strategy, target_team=target.name.lower(),
                native_play_policy_id=native_id, rules_hash=rules.sha256,
                **({"nonplay_policy": nonplay} if nonplay is not None else {}),
                source_model_hashes=tuple(sorted(model._model_hash(search_branching="automatic") for model in models)),
                source_models=native_models)
        for model in models:
            available()
            require_absence(model.find_native_round_with_constant_policy_below(target=target,
                policy_team=target,
                auction_action=nonplay["auctionAction"] if nonplay else None,
                escalation_action=nonplay["escalationAction"] if nonplay else None,
                **({"native_project_policy_id": nonplay["projectPolicyId"]} if nonplay else {}),
                **({"record_policy_model_hashes": True} if nonplay else {}),
                **({"native_play_policy_id": native_id} if native_id is not None else
                   {"play_policies": graph["incumbentPlayPolicies"]}), threshold=selected_floor,
                maximum_seconds=remaining(), maximum_conflicts=common["maximum_conflicts"],
                cancel_check=cancel_check, optimize_margin=False))
        del models
    else:
        rebuilt = compiler._solve_native_bid_policy_bounds(exact, prefix, target,
            visibility="complete_state", selected_root_action=graph["selectedAction"],
            compact_purchase_invariant_ownership=graph["selectedValueProof"]["preparation"]["scope"].get("ownershipCompaction") == "purchase_recipient_invariant_v1",
            maximum_iterations=limits["maximumIterations"], maximum_observations=limits["maximumObservations"],
            maximum_seconds=remaining(), maximum_queries=available(),
            opponent_ceiling_policy=strategy["scope"].get("opponentCeilingPolicy"), **common)
        charge(rebuilt["queriesUsed"])
        if rebuilt["status"] != "scoped_bound_converged":
            raise InformationSetCompilationStopped("selected_value_not_verified", stage="native_bid_quality_readback",
                evidence={"selectedValueProof": rebuilt})
        if (rebuilt["strategy"]["lowerBound"] != selected_floor
                or rebuilt["strategy"]["upperBound"] != strategy["upperBound"]):
            raise ValueError("selected bid readback native selected value mismatch")

    for alternative in graph["alternatives"]:
        preparation = alternative["preparation"]
        if preparation["kind"] in {"native_bid_streamed_alternative_preparation_v1", "native_bid_streamed_world_upper_v1"}:
            reverify_native_bid_world_upper_v1(preparation, rules=rules, exact=exact, source_prefix=prefix,
                action=alternative["action"], opponent_policy=graph["opponentStrategy"],
                opponent_play_policy_id=graph.get("opponentPlayPolicyId"),
                maximum_seconds=remaining(), maximum_queries=available(), _on_queries_used=charge, **common)
        else:
            models, rebuilt = compiler._prepare_native_bid_branches(exact, prefix, visibility="complete_state",
                selected_root_action=alternative["action"], policy_team=target.other,
                nonplay_policy={key: graph["opponentStrategy"][key] for key in ("auction_action", "escalation_action")},
                maximum_seconds=remaining(), maximum_queries=available(), **common)
            charge(rebuilt["queriesUsed"])
            if [model._model_hash(search_branching="automatic") for model in models] != alternative["sourceModelHashes"]:
                raise ValueError("selected bid readback alternative model substitution")
            for model in models:
                available()
                require_absence(model.find_native_round_with_constant_policy_below(target=target.other,
                    policy_team=target.other, threshold=-alternative["upperBound"],
                    maximum_seconds=remaining(), maximum_conflicts=common["maximum_conflicts"],
                    cancel_check=cancel_check, optimize_margin=False, **graph["opponentStrategy"]))
            del models
        remaining()

    return native_models


def build_native_complete_state_play_quality_artifact_v1(value, *, rules, source_prefix, normalized, request, budget):
    """Bind compact native terminal roots to an owned complete-state result."""
    from .ortools_exact_state import _effective_prefix_hash
    from .ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    def hashed(document):
        return isinstance(document, Mapping) and document.get("sha256") == canonical_sha256({
            key: item for key, item in document.items() if key != "sha256"})
    normalized.require_integrity()
    if (not isinstance(budget, ScenarioProofBudgetV1) or budget.native_observed is None
            or request.budget_profile != budget.profile_id or request.mode != "exact"
            or request.visibility != "complete_state" or normalized.decision_kind != "play"
            or normalized.prefix_hash != _effective_prefix_hash(source_prefix)
            or normalized.rules_hash != rules.sha256 or not source_prefix.project_actions_present
            or not hashed(value) or value.get("kind") != "native_complete_state_play_roots_v1"
            or value.get("strategicCoverage") is not False):
        raise ValueError("compact quality source or proof identity mismatch")
    preparation, roots, certificate = value["preparation"], value["roots"], value["certificate"]
    if not all(hashed(item) for item in (preparation, roots, certificate)):
        raise ValueError("compact quality nested evidence checksum mismatch")
    node_limit = preparation.get("maximumNodeObservations")
    if type(node_limit) is not int or not 0 < node_limit <= min(budget.maximum_nodes, budget.maximum_state_space):
        raise ValueError("compact quality node budget mismatch")
    native_limits = budget.native_observed.to_dict()
    native_limits["maximumObservations"] = min(budget.native_observed.maximum_observations, node_limit)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    worlds, contexts, expected = compiler._prepare_complete_state_play(normalized, source_prefix, request,
        budget=budget, maximum_seconds=min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds),
        maximum_nodes=node_limit)
    start, world = worlds[0][1].start, expected["worldId"]
    actions = {row["code"] for row in normalized.legal_actions}
    if (preparation != expected or expected["modelBytes"] > budget.maximum_cache_bytes
            or expected["remainingActions"] > min(budget.maximum_depth, budget.maximum_horizon_actions)
            or roots.get("status") != "complete" or roots.get("limits") != native_limits
            or set(roots.get("legalActions", [])) != actions or set(roots.get("roots", {})) != actions
            or certificate.get("sourceArtifactHash") != roots["sha256"]
            or value.get("status") != certificate.get("status")):
        raise ValueError("compact quality model, budget, or root coverage mismatch")
    query_models = ({row["branchId"]: row["modelHash"] for row in expected["branches"]}
                    if contexts else {None: expected["modelHash"]})

    def owned_query(query):
        if not isinstance(query, Mapping):
            return False
        if "receipt" not in query:
            receipt = query
            if query.get("kind") not in {"native_path_extremum_v1", "native_exhaustive_path_enumeration_v1"}:
                return False
        else:
            fields = {"worldId", "receipt", *({"branchId"} if contexts else set())}
            receipt = query.get("receipt")
            if (set(query) != fields or not isinstance(receipt, Mapping)
                    or receipt.get("kind") not in {"native_observed_path_query_v2", "native_observed_path_query_v3"}):
                return False
        branch = query.get("branchId")
        return (query.get("worldId") == world and hashed(receipt)
                and (type(branch) is str if contexts else branch is None)
                and branch in query_models and receipt.get("baseModelHash") == query_models[branch])

    for action, proof in roots["roots"].items():
        scope = proof.get("scope", {})
        if (not hashed(proof) or proof.get("status") != "converged" or type(proof.get("value")) is not int
                or scope.get("worlds") != [world] or scope.get("selectedRootAction") != action
                or scope.get("rulesHash") != rules.sha256 or scope.get("targetTeam") != request.target_team
                or scope.get("utilityScope") != {"utility": request.utility, "boundaryEvaluator": None,
                    "playStepCount": len(start.cards), "basePrefixLength": len(start.base_prefix), "cardCount": len(start.cards)}
                or scope.get("branches") != expected.get("branches")
                or scope.get("declarationCoverage") != ({expected["projectCoverage"]["sha256"]:
                    expected["projectCoverage"]} if contexts else None)
                or proof.get("scopeHash") != canonical_sha256(scope)
                or any(not owned_query(query) for query in proof.get("queries", []))):
            raise ValueError("compact quality root ownership or terminal proof mismatch")
    if (set(row["action"] for row in certificate.get("roots", [])) != actions
            or any(row["value"] != roots["roots"][row["action"]]["value"] for row in certificate["roots"])):
        raise ValueError("compact quality certificate values disagree with root proofs")
    from .ortools_information_set_compiler import _integral_success_threshold
    rebuilt_certificate = compiler._certify_observed_root_actions(roots,
        success_threshold=_integral_success_threshold(request.success_threshold), claim=request.claim,
        selected_action=request.selected_action,
        maximum_seconds=min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds),
        action_quality=True, quality_scope="compiled_complete_state")
    if rebuilt_certificate != certificate:
        raise ValueError("compact quality claim or action-quality model mismatch")
    outcome = compiler._native_certificate_outcome(value, request, certificate,
        {action: LegalPlayActionV1.from_code(action).action_id for action in actions})
    allocation = {"schemaVersion": 1, "kind": "native_complete_state_quality_v1",
        "sourcePrefixHash": source_prefix.sha256, "exactStateId": normalized.exact_state_id,
        "rulesHash": rules.sha256, "sourcePrefix": source_prefix.to_dict(),
        "normalizedState": normalized.to_dict(), "proofBudget": {"profileId": budget.profile_id,
            "sha256": budget.sha256}, "preparationHash": preparation["sha256"]}
    return {"request": request.to_dict(), "proof": outcome.to_dict(), "proofModel": outcome.model_document(),
        "graph": value, "allocationModel": {**allocation, "sha256": canonical_sha256(allocation)}}


def native_raw_point_quality_request_v1(request):
    """Derive a private editorial utility without widening the public catalog."""
    from dataclasses import replace
    from .ortools_complete_state_graph import INTERNAL_RAW_POINT_MARGIN_UTILITY_V1
    if (request.mode != "exact" or request.visibility not in {"complete_state", "actor_information_set"}
            or request.line_universe is not None):
        raise ValueError("raw quality companion requires exact roots")
    core = request.to_dict()
    core.pop("sha256")
    core.pop("selectedAction", None)
    core.pop("utilityEncoding", None)
    core.update(utility=INTERNAL_RAW_POINT_MARGIN_UTILITY_V1, claim="winning_root_action_count", successThreshold=0.0)
    return replace(request, utility=INTERNAL_RAW_POINT_MARGIN_UTILITY_V1,
        claim="winning_root_action_count", selected_action=None, success_threshold=0.0,
        utility_encoding=None, sha256=canonical_sha256(core))


def native_round_win_quality_request_v1(request):
    """Keep game-win truth separate from awarded-margin threshold success."""
    if (request.mode != "exact" or request.visibility not in {"complete_state", "actor_information_set"}
            or request.line_universe is not None):
        raise ValueError("round-win quality companion requires exact roots")
    core = request.to_dict()
    core.pop("sha256")
    core.pop("selectedAction", None)
    core.pop("utilityEncoding", None)
    core.update(utility="round_win", claim="winning_root_action_count", successThreshold=0.0)
    return ScenarioProofRequestV1.build(core)


def native_basic_policy_quality_request_v1(request):
    core = request.to_dict()
    for key in ("sha256", "selectedAction", "lineUniverse", "policies", "utilityEncoding"):
        core.pop(key, None)
    core.update(mode="bounded", visibility="actor_information_set", strategyClass="pure_deterministic",
        utility="awarded_point_margin", claim="winning_root_action_count", successThreshold=0.0,
        horizon={"unit": "plies", "value": 1}, boundaryEvaluator="current_raw_point_margin_v1")
    return ScenarioProofRequestV1.build(core)


def native_project_counterexample_request_v1(request):
    core = native_round_win_quality_request_v1(request).to_dict()
    core.pop('sha256')
    core['visibility'] = 'complete_state'
    return ScenarioProofRequestV1.build(core)


def native_tie_margin_quality_request_v1(request):
    if request.mode != "exact" or request.visibility != "actor_information_set" or request.utility != "awarded_point_margin":
        raise ValueError("tie comparison requires exact actor-visible awarded-margin roots")
    core = request.to_dict()
    for key in ("sha256", "selectedAction", "lineUniverse"):
        core.pop(key, None)
    core.update(claim="winning_root_action_count", successThreshold=0.0)
    return ScenarioProofRequestV1.build(core)


def native_tie_rule_profile_v1(rules, tie_rule):
    if not isinstance(rules, RuleBook) or tie_rule not in {"buyer_makes", "buyer_loses"}:
        raise ValueError("tie comparison requires a frozen rule profile and a supported tie rule")
    data = copy.deepcopy(rules.data)
    data["scoring"]["contract"]["buyer_tie_outcome"] = tie_rule
    return RuleBook(data)


def native_tie_allocation_universe_hash_v1(artifact):
    allocation = artifact.get("allocationModel") or {}
    worlds = artifact.get("proof", {}).get("hiddenWorlds") or {}
    fields = ("observer", "pool", "handSizes", "fixedOwnership", "acceptedWorlds",
        "reachability", "auctionDomainIrHash", "escalationDomainIrHash")
    if (allocation.get("reachability") != "deal_auction_escalation_project_play_cp_sat_v1"
            or allocation.get("terminalSolverStatus") != "infeasible"
            or worlds.get("complete") is not True or type(worlds.get("count")) is not int
            or worlds["count"] < 1 or any(key not in allocation for key in fields)):
        raise ValueError("tie comparison lacks an exhaustive native allocation universe")
    return canonical_sha256({**{key: allocation[key] for key in fields},
        "worldCount": worlds["count"], "priorHistoryWorldsHash": allocation.get("priorHistoryWorldsHash")})


def _validate_raw_point_quality_companions_v1(value, *, rules, execution_budget, remaining, cancel_check, private_recall,
        evidence_key="rawPointEvidence", request_factory=None, verified_artifacts=frozenset()):
    companions = value.get(evidence_key, {})
    round_win = evidence_key == "roundWinEvidence"
    if (not isinstance(companions, Mapping) or len(companions) > 1
            or companions and (not isinstance(rules, RuleBook) or execution_budget is None
                or (execution_budget.video_interestingness or {}).get("schemaVersion") != (4 if round_win else 2)
                or not round_win and value.get("status") != "complete")):
        raise ValueError("raw quality companion policy or coverage mismatch")
    if not companions:
        return
    from .ortools_exact_state import OrToolsExactStateNormalizerV1
    from .ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    for digest, artifact in companions.items():
        rows = [row for row in value["decisions"] if row.get("proofArtifactHash") == digest]
        if (len(rows) != 1 or rows[0].get("decisionKind", "play") != "play"
                or not isinstance(artifact, Mapping)):
            raise ValueError("raw quality companion source reference mismatch")
        row = rows[0]
        request = (request_factory or (native_round_win_quality_request_v1 if round_win else native_raw_point_quality_request_v1))(
            ScenarioProofRequestV1.build(value["artifacts"][digest]["request"]))
        observed = request.visibility == "actor_information_set"
        if set(artifact) != {"request", "proof", "proofModel", "graph", "allocationModel"} | (
                {"sourcePrefix"} if observed else set()) | (
                {"rootWitnesses"} if observed and "rootWitnesses" in artifact else set()):
            raise ValueError("raw quality companion source reference mismatch")
        budget = ScenarioProofBudgetV1.build(request.budget_profile,
            value["proofBudgetProfiles"][request.budget_profile])
        prefix = ExactTracePrefixV1.build(artifact["sourcePrefix"] if observed else artifact["allocationModel"]["sourcePrefix"])
        if (artifact["request"] != request.to_dict() or prefix.sha256 != row["sourcePrefixHash"]
                or row.get("proofBudgetHash") != budget.sha256
                or not observed and (artifact["allocationModel"].get("sourcePrefixHash") != row["sourcePrefixHash"]
                    or artifact["allocationModel"].get("exactStateId") != row["exactStateId"]
                    or artifact["allocationModel"].get("rulesHash") != rules.sha256)
                or artifact["proof"].get("completionStatus") != "exact_complete"):
            raise ValueError("raw quality companion request, prefix or budget substitution")
        read_remaining = _quality_readback_clock(min(value["limits"]["maximumDecisionSeconds"],
            budget.maximum_wall_seconds, budget.maximum_cpu_seconds, remaining()), cancel_check)
        normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
            prefix, maximum_seconds=read_remaining(), cancel_check=cancel_check)
        if (normalized.exact_state_id != row["exactStateId"] or normalized.decision_kind != "play"
                or normalized.next_actor != row["actor"] or len(normalized.legal_actions) < 2
                or request.target_team != row["targetTeam"]
                or row["action"] not in {action["code"] for action in normalized.legal_actions}):
            raise ValueError("raw quality companion state or actor substitution")
        nodes = min(value["limits"]["maximumNodesPerDecision"], budget.maximum_nodes, budget.maximum_state_space)
        if observed:
            owned = {key: item for key, item in artifact.items() if key != "sourcePrefix"}
            context = dict(rules=rules, source_prefix=prefix, normalized=normalized, request=request,
                maximum_nodes=nodes, maximum_depth=min(value["limits"]["maximumDepth"], budget.maximum_depth),
                maximum_worlds=execution_budget.resolve_hidden_world_cap(
                    min(value["limits"]["maximumWorldsPerDecision"], budget.maximum_hidden_worlds),
                    play_index=len(prefix.play_actions)),
                declared_maximum_seconds=min(value["limits"]["maximumDecisionSeconds"], budget.maximum_wall_seconds),
                private_recall=private_recall, policy_search=budget.policy_search)
            token = _exact_observed_verification_token_v1(owned, **context)
            if token not in verified_artifacts:
                reverify_native_exact_observed_play_artifact_v1(owned, **context,
                    maximum_seconds=read_remaining(), cancel_check=cancel_check)
                if isinstance(verified_artifacts, set):
                    verified_artifacts.add(token)
            read_remaining()
            continue
        preparation = artifact["graph"]["preparation"]
        if (type(preparation.get("remainingActions")) is not int
                or preparation["remainingActions"] > min(value["limits"]["maximumDepth"],
                    budget.maximum_depth, budget.maximum_horizon_actions)):
            raise ValueError("raw quality companion depth budget substitution")
        if preparation.get("maximumNodeObservations") != nodes:
            raise ValueError("raw quality companion node budget substitution")
        fresh = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._prove_complete_state_play(
            normalized, prefix, request, budget=budget, maximum_nodes=nodes,
            maximum_seconds=read_remaining(), cancel_check=cancel_check)
        if (_native_observed_bounds_semantics(fresh, maximum_seconds=budget.maximum_wall_seconds)
                != _native_observed_bounds_semantics(artifact["graph"], maximum_seconds=budget.maximum_wall_seconds)):
            raise ValueError("raw quality companion differs from fresh native proof")
        rebuilt = build_native_complete_state_play_quality_artifact_v1(artifact["graph"], rules=rules,
            source_prefix=prefix, normalized=normalized, request=request, budget=budget)
        if canonical_sha256(rebuilt) != canonical_sha256(artifact):
            raise ValueError("raw quality companion artifact binding mismatch")
        read_remaining()


def _validate_tie_rule_quality_companions_v1(value, *, rules, execution_budget, remaining, cancel_check, private_recall,
        verified_artifacts=frozenset()):
    companions = value.get("tieRuleEvidence", {})
    if (not isinstance(companions, Mapping) or len(companions) > 1
            or companions and (not isinstance(rules, RuleBook) or execution_budget is None
                or (execution_budget.video_interestingness or {}).get("schemaVersion") != 4)):
        raise ValueError("tie-rule companion policy mismatch")
    from .ortools_exact_state import OrToolsExactStateNormalizerV1
    from .ortools_research_executor import _observer_safe_play_policy_artifact_v1
    for digest, comparison in companions.items():
        rows = [row for row in value["decisions"] if row.get("proofArtifactHash") == digest]
        baseline = value.get("baselineEvidence", {}).get(digest, {})
        if (len(rows) != 1 or rows[0].get("decisionKind", "play") != "play"
                or not isinstance(comparison, Mapping)
                or set(comparison) != {"sourceArtifactHash", "sourcePrefixHash", "baselineAction", "buyer_makes", "buyer_loses", "sha256"}
                or comparison["sha256"] != canonical_sha256({key: item for key, item in comparison.items() if key != "sha256"})
                or comparison["sourceArtifactHash"] != digest
                or comparison["sourcePrefixHash"] != rows[0]["sourcePrefixHash"]
                or comparison["baselineAction"] != baseline.get("proof", {}).get("selectedAction")):
            raise ValueError("tie-rule companion source or baseline substitution")
        source = value["artifacts"][digest]
        source_request = ScenarioProofRequestV1.build(source["request"])
        universe_hash = native_tie_allocation_universe_hash_v1(source)
        baseline_request = native_basic_policy_quality_request_v1(source_request)
        for tie_rule in ("buyer_makes", "buyer_loses"):
            variant = native_tie_rule_profile_v1(rules, tie_rule)
            entry = comparison[tie_rule]
            if (not isinstance(entry, Mapping)
                    or set(entry) != {"margin", "roundWin", "rulesSnapshot", "rulesHash", "exactStateId", "allocationUniverseHash"}
                    or entry["rulesSnapshot"] != variant.data or entry["rulesHash"] != variant.sha256
                    or entry["allocationUniverseHash"] != universe_hash
                    or any(native_tie_allocation_universe_hash_v1(entry[key]) != universe_hash for key in ("margin", "roundWin"))):
                raise ValueError("tie-rule companion profile or legal-world substitution")
            prefix = ExactTracePrefixV1.build(entry["margin"]["sourcePrefix"])
            if prefix.sha256 != rows[0]["sourcePrefixHash"]:
                raise ValueError("tie-rule companion prefix substitution")
            state = OrToolsExactStateNormalizerV1(variant, native_cp_sat=True).normalize_play_or_project(
                prefix, maximum_seconds=remaining(), cancel_check=cancel_check)
            if state.exact_state_id != entry["exactStateId"]:
                raise ValueError("tie-rule companion exact-state substitution")
            _, rebuilt_baseline = _observer_safe_play_policy_artifact_v1(variant, state, baseline_request,
                actual_action=None, policy_id=execution_budget.video_interestingness["baselinePolicyId"],
                maximum_seconds=remaining(), cancel_check=cancel_check)
            if rebuilt_baseline["proof"]["selectedAction"] != comparison["baselineAction"]:
                raise ValueError("tie-rule companion changed the baseline action")
            checked = {**value, "decisions": [{**rows[0], "exactStateId": state.exact_state_id}]}
            for key in ("margin", "roundWin"):
                factory = (native_round_win_quality_request_v1 if key == "roundWin" else
                    (lambda request: request) if variant.sha256 == rules.sha256 else native_tie_margin_quality_request_v1)
                _validate_raw_point_quality_companions_v1({**checked, "roundWinEvidence": {digest: entry[key]}},
                    rules=variant, execution_budget=execution_budget, remaining=remaining,
                    cancel_check=cancel_check, private_recall=private_recall,
                    evidence_key="roundWinEvidence", request_factory=factory, verified_artifacts=verified_artifacts)
            remaining()


def _validate_basic_policy_quality_companions_v1(value, *, rules, execution_budget, remaining, cancel_check):
    companions = value.get("baselineEvidence", {})
    if (not isinstance(companions, Mapping) or len(companions) > len(value["decisions"])
            or companions and (not isinstance(rules, RuleBook) or execution_budget is None
                or (execution_budget.video_interestingness or {}).get("schemaVersion") != 4)):
        raise ValueError("basic policy companion policy mismatch")
    if not companions:
        return
    from .ortools_exact_state import OrToolsExactStateNormalizerV1
    from .ortools_research_executor import _observer_safe_play_policy_artifact_v1
    for digest, artifact in companions.items():
        rows = [row for row in value["decisions"] if row.get("proofArtifactHash") == digest]
        if (len(rows) != 1 or rows[0].get("decisionKind", "play") != "play"
                or not isinstance(artifact, Mapping)
                or set(artifact) != {"request", "proof", "proofModel", "graph", "allocationModel", "sourcePrefix"}):
            raise ValueError("basic policy companion source reference mismatch")
        row = rows[0]
        request = native_basic_policy_quality_request_v1(
            ScenarioProofRequestV1.build(value["artifacts"][digest]["request"]))
        prefix = ExactTracePrefixV1.build(artifact["sourcePrefix"])
        if artifact["request"] != request.to_dict() or prefix.sha256 != row["sourcePrefixHash"]:
            raise ValueError("basic policy companion request or prefix substitution")
        normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
            prefix, maximum_seconds=remaining(), cancel_check=cancel_check)
        if (normalized.exact_state_id != row["exactStateId"] or normalized.next_actor != row["actor"]
                or request.target_team != row["targetTeam"]):
            raise ValueError("basic policy companion actor or state substitution")
        _, rebuilt = _observer_safe_play_policy_artifact_v1(rules, normalized, request,
            actual_action=None, policy_id=execution_budget.video_interestingness["baselinePolicyId"],
            maximum_seconds=remaining(), cancel_check=cancel_check)
        if canonical_sha256({**rebuilt, "sourcePrefix": prefix.to_dict()}) != canonical_sha256(artifact):
            raise ValueError("basic policy companion differs from observer-safe selection")
        remaining()


def _validate_project_inference_quality_companions_v1(value, *, rules, execution_budget,
        search_budget_profiles, remaining, cancel_check):
    companions = value.get('projectInferenceEvidence', {})
    if (not isinstance(companions, Mapping) or len(companions) > 1
            or companions and (not isinstance(rules, RuleBook) or execution_budget is None
                or (execution_budget.video_interestingness or {}).get('schemaVersion') != 4)):
        raise ValueError('project inference companion policy mismatch')
    from .ortools_exact_state import OrToolsExactStateNormalizerV1
    from .ortools_research_executor import _project_inference_quality_evidence_v1
    for digest, evidence in companions.items():
        rows = [row for row in value['decisions'] if row.get('proofArtifactHash') == digest]
        if (len(rows) != 1 or rows[0].get('decisionKind', 'play') != 'play'
                or not isinstance(evidence, Mapping)
                or evidence.get('sha256') != canonical_sha256({k: v for k, v in evidence.items() if k != 'sha256'})
                or evidence.get('sourceArtifactHash') != digest
                or evidence.get('sourcePrefixHash') != rows[0]['sourcePrefixHash']
                or evidence.get('exactStateId') != rows[0]['exactStateId']
                or evidence.get('action') != rows[0]['action'] or evidence.get('rulesHash') != rules.sha256):
            raise ValueError('project inference companion source substitution')
        row = rows[0]
        source = value['artifacts'][digest]
        request = ScenarioProofRequestV1.build(source['request'])
        prefix = ExactTracePrefixV1.build(value.get('baselineEvidence', {}).get(digest, {}).get('sourcePrefix', {}))
        if prefix.sha256 != row['sourcePrefixHash']:
            raise ValueError('project inference companion prefix substitution')
        budget = ScenarioProofBudgetV1.build(request.budget_profile, value['proofBudgetProfiles'][request.budget_profile])
        if row.get('proofBudgetHash') != budget.sha256 or not search_budget_profiles:
            raise ValueError('project inference companion budget mismatch')
        read_remaining = _quality_readback_clock(min(remaining(), value['limits']['maximumDecisionSeconds'],
            budget.maximum_wall_seconds, budget.maximum_cpu_seconds), cancel_check)
        state = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
            prefix, maximum_seconds=read_remaining(), cancel_check=cancel_check)
        if state.exact_state_id != row['exactStateId']:
            raise ValueError('project inference companion exact-state substitution')
        rebuilt = _project_inference_quality_evidence_v1(rules, state, prefix, source,
            value.get('roundWinEvidence', {}).get(digest, {}), budget=budget,
            maximum_nodes=min(value['limits']['maximumNodesPerDecision'], budget.maximum_nodes, budget.maximum_state_space),
            maximum_depth=min(value['limits']['maximumDepth'], budget.maximum_depth),
            maximum_conflicts=search_budget_profiles[request.budget_profile]['maximumConflicts'],
            remaining_seconds=read_remaining, cancel_check=cancel_check)
        read_remaining()
        if rebuilt is None or rebuilt != evidence:
            raise ValueError('project inference companion differs from fresh native proof')


def _quality_readback_clock(maximum_seconds, cancel_check):
    """Share a caller's verification allowance without changing frozen budgets."""
    import time
    from .ortools_information_set_compiler import InformationSetCompilationStopped
    if maximum_seconds is not None and (type(maximum_seconds) not in {int, float}
            or not math.isfinite(maximum_seconds) or maximum_seconds <= 0):
        raise ValueError("quality readback deadline must be finite and positive")
    if cancel_check is not None and not callable(cancel_check):
        raise TypeError("quality readback cancellation check must be callable")
    deadline = math.inf if maximum_seconds is None else time.monotonic() + maximum_seconds
    def remaining():
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage="quality_readback")
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise InformationSetCompilationStopped("maximum_seconds", stage="quality_readback")
        return seconds
    return remaining


def validate_native_decision_quality_evidence_v1(
    value: Mapping[str, Any], *, source_prefix: ExactTracePrefixV1,
    rules_hash: str, request: ScenarioProofRequestV1,
    policy: ScenarioPlayingQualityPolicyV1 | None = None,
    rules: RuleBook | None = None,
    search_budget_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    execution_budget: ScenarioQualityExecutionBudgetV1 | None = None,
    private_recall=None,
    escalation_events=None,
    opportunity_order=None,
    maximum_seconds=None, cancel_check=None,
) -> None:
    """Verify external/stored evidence with fresh native reconstruction."""
    _validate_native_decision_quality_evidence_v1(value, source_prefix=source_prefix,
        rules_hash=rules_hash, request=request, policy=policy, rules=rules,
        search_budget_profiles=search_budget_profiles, execution_budget=execution_budget,
        private_recall=private_recall, escalation_events=escalation_events,
        opportunity_order=opportunity_order, maximum_seconds=maximum_seconds, cancel_check=cancel_check,
        verified_artifacts=set())


def _validate_native_decision_quality_evidence_v1(
    value: Mapping[str, Any], *, source_prefix: ExactTracePrefixV1,
    rules_hash: str, request: ScenarioProofRequestV1,
    policy: ScenarioPlayingQualityPolicyV1 | None = None,
    rules: RuleBook | None = None,
    search_budget_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    execution_budget: ScenarioQualityExecutionBudgetV1 | None = None,
    private_recall=None, escalation_events=None, opportunity_order=None,
    verified_artifacts=frozenset(), verified_normalizations=None,
    maximum_seconds=None, cancel_check=None,
) -> None:
    """Check owned decision evidence, not solver truth or whole-round quality."""
    if maximum_seconds is None and execution_budget is not None:
        maximum_seconds = execution_budget.maximum_wall_seconds
    remaining = _quality_readback_clock(maximum_seconds, cancel_check)
    remaining()
    if rules is not None and (not isinstance(rules, RuleBook) or rules.sha256 != rules_hash):
        raise ValueError("decision quality frozen rules mismatch")
    if "escalationHistoryHash" in value and (escalation_events is None
            or value["escalationHistoryHash"] != canonical_sha256(escalation_events)):
        raise ValueError("decision quality escalation history substitution")
    auction = value.get("kind") == "native_quality_auction_continuation_v1"
    if value.get("kind") not in {"native_quality_auction_continuation_v1", "native_quality_play_continuation_v1"}:
        raise ValueError("decision quality evidence kind is unsupported")
    core = {key: item for key, item in value.items() if key not in {"sha256", "executionReceipt"}}
    core_hash = canonical_sha256(core)
    if (value.get("sha256") != core_hash
            or value.get("scope") not in ({"auction_suffix_only"} if auction else {"play_suffix_only", "project_play_suffix"})
            or value.get("qualificationAuthority") != "cp_sat"
            or value.get("rulesHash") != rules_hash
            or value.get("sourcePrefixHash") != source_prefix.sha256
            or value.get("qualityRequestHash") != request.sha256):
        raise ValueError("auction quality source ownership mismatch")
    if value.get("operation") not in {"generate", "audit_recorded", "audit_then_generate"}:
        raise ValueError("auction quality operation is invalid")
    stage_token = None
    if value.get("status") == "complete" and isinstance(rules, RuleBook):
        stage_token = _quality_stage_verification_token_v1(core_hash, rules=rules,
            source_prefix=source_prefix, request=request, policy=policy,
            execution_budget=execution_budget, search_budget_profiles=search_budget_profiles,
            private_recall=private_recall, escalation_events=escalation_events,
            opportunity_order=opportunity_order)
        remaining()
        if stage_token in verified_artifacts:
            return
    final = ExactTracePrefixV1.build(value["finalPrefix"])
    submitted = None
    if value["operation"] == "audit_then_generate":
        submitted = ExactTracePrefixV1.build(value["submittedPrefix"])
        if value.get("recordedPrefixHash") != submitted.sha256:
            raise ValueError("quality submitted prefix ownership mismatch")
        action_fields = ("auctionActions",) if auction else ("playActions", "projectActions")
        immutable = {"sha256", "stateAssertions", "projectAssertions", *action_fields}
        if ({key: item for key, item in submitted.to_dict().items() if key not in immutable}
                != {key: item for key, item in source_prefix.to_dict().items() if key not in immutable}):
            raise ValueError("quality submitted prefix context mismatch")
        for field in action_fields:
            source_rows = source_prefix.to_dict().get(field, [])
            submitted_rows = submitted.to_dict().get(field, [])
            final_rows = final.to_dict().get(field, [])
            common = min(len(submitted_rows), len(final_rows))
            if (submitted_rows[:len(source_rows)] != source_rows
                    or submitted_rows[:common] != final_rows[:common]
                    or value.get("status") == "complete" and len(final_rows) < len(submitted_rows)):
                raise ValueError("quality submitted prefix actions mismatch")
    mutable = {"sha256", "stateAssertions", "projectAssertions"} | ({"auctionActions"} if auction else {"playActions", "projectActions"})
    if ({key: item for key, item in final.to_dict().items() if key not in mutable}
            != {key: item for key, item in source_prefix.to_dict().items() if key not in mutable}
            or final.auction_actions[:len(source_prefix.auction_actions)] != source_prefix.auction_actions
            or final.play_actions[:len(source_prefix.play_actions)] != source_prefix.play_actions
            or final.project_actions[:len(source_prefix.project_actions)] != source_prefix.project_actions):
        raise ValueError("auction quality trace source mismatch")
    rows, artifacts = value["decisions"], value["artifacts"]
    start, end = (value.get("coveredFromBidIndex"), value.get("coveredToBidIndex")) if auction else (
        value.get("coveredFromPlayIndex"), value.get("coveredToPlayIndex"))
    source_actions = source_prefix.auction_actions if auction else source_prefix.play_actions
    final_actions = final.auction_actions if auction else final.play_actions
    project_count = 0 if auction else len(final.project_actions) - len(source_prefix.project_actions)
    if (type(start) is not int or type(end) is not int
            or start != len(source_actions) or end != len(final_actions)
            or not isinstance(rows, list) or end - start + project_count != len(rows)):
        raise ValueError("auction quality decision coverage mismatch")
    if policy is not None and value.get("playingQuality") != policy.to_dict():
        raise ValueError("decision quality policy mismatch")
    for digest, artifact in artifacts.items():
        if digest != canonical_sha256(artifact):
            raise ValueError("auction quality artifact checksum mismatch")
        if "nativeExecutionBindings" in artifact:
            bindings = {"rootArtifact": "graph", "preparation": "allocationModel", "outcome": "proof"}
            if artifact["nativeExecutionBindings"] != bindings:
                raise ValueError("native execution reference binding mismatch")
            execution = dict(artifact["nativeExecution"])
            digest = execution.pop("sha256", None)
            if any(key in execution for key in bindings):
                raise ValueError("native execution has duplicate reference content")
            execution.update({key: artifact[field] for key, field in bindings.items()})
            allocation = artifact["allocationModel"]
            if allocation.get("reference") == "graph.preparations":
                preparation = artifact["graph"].get("preparations")
                if (set(allocation) != {"reference", "sha256"} or not isinstance(preparation, Mapping)
                        or allocation["sha256"] != canonical_sha256(preparation)):
                    raise ValueError("native preparation reference mismatch")
                execution["preparation"] = preparation
            if digest != canonical_sha256(execution):
                raise ValueError("native execution reference checksum mismatch")
    paired_contexts = {}
    paired_evidence = value.get("pairedPolicyEvidence", {})
    references = {artifact.get("proofModel", {}).get("pairEvidenceHash") for artifact in artifacts.values()
        if artifact.get("proofModel", {}).get("kind") == "native_complete_state_policy_pair_quality_v1"}
    if not isinstance(paired_evidence, Mapping) or set(paired_evidence) != references or len(paired_evidence) > 1:
        raise ValueError("paired quality evidence references mismatch")
    for digest, evidence in paired_evidence.items():
        import time
        from .ortools_exact_state import OrToolsExactStateNormalizerV1
        if not isinstance(rules, RuleBook) or digest != canonical_sha256(evidence):
            raise ValueError("paired quality evidence ownership mismatch")
        pair_prefix = ExactTracePrefixV1.build(evidence["sourcePrefix"])
        pair_request = ScenarioProofRequestV1.build(evidence["request"])
        pair_budget = ScenarioProofBudgetV1.build(pair_request.budget_profile,
            value["proofBudgetProfiles"][pair_request.budget_profile])
        wall_end = time.monotonic() + min(pair_budget.maximum_wall_seconds, remaining())
        cpu_end = time.process_time() + pair_budget.maximum_cpu_seconds

        def exhausted():
            return bool(cancel_check and cancel_check()) or time.monotonic() >= wall_end or time.process_time() >= cpu_end

        def pair_remaining():
            seconds = min(wall_end - time.monotonic(), cpu_end - time.process_time(), remaining())
            if seconds <= 0:
                raise TimeoutError("paired quality read budget exhausted")
            return seconds

        normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
        pair_state = normalizer.normalize_play_or_project(pair_prefix,
            maximum_seconds=pair_remaining(), cancel_check=exhausted)
        synthesis = evidence.get("synthesis")
        if synthesis is not None:
            usage = [proof["resources"]["usage"] for proof in evidence["proofs"].values()]
            preparation = synthesis.get("preparation", {})
            if (synthesis.get("sha256") != canonical_sha256({key: item for key, item in synthesis.items() if key != "sha256"})
                    or synthesis.get("kind") != "native_complete_state_policy_pair_synthesis_v1"
                    or synthesis.get("status") != "complete"
                    or canonical_sha256(synthesis.get("proofs")) != canonical_sha256(evidence["proofs"])
                    or preparation.get("sourcePrefixHash") != pair_prefix.sha256
                    or preparation.get("exactStateId") != pair_state.exact_state_id
                    or preparation.get("requestHash") != pair_request.sha256
                    or preparation.get("budgetHash") != pair_budget.sha256
                    or preparation.get("sha256") != evidence["receipt"]["preparationHash"]
                    or synthesis.get("queriesUsed") != sum(row["counterexampleQueries"] for row in usage)
                    or synthesis.get("iterationsUsed") != sum(row["masterIterations"] for row in usage)
                    or synthesis["queriesUsed"] + evidence["receipt"]["queryLimit"] > pair_budget.native_observed.maximum_queries
                    or synthesis["iterationsUsed"] > pair_budget.native_observed.maximum_iterations
                    or type(synthesis.get("modelBytes")) is not int
                    or not 0 < synthesis["modelBytes"] <= pair_budget.maximum_cache_bytes):
                raise ValueError("paired quality synthesis ownership or budget mismatch")
        verified_pair = canonical_sha256({"kind": "verified_native_complete_state_policy_pair_v1",
            "evidenceHash": digest})
        # Only the generation wrapper owns these ephemeral verified identities.
        # Public/stored reads always reprove the pair; state checks below remain.
        if verified_pair not in verified_artifacts:
            validate_native_complete_state_policy_pair_v1(evidence["receipt"], evidence["proofs"],
                rules=rules, normalized=pair_state, source_prefix=pair_prefix,
                request=pair_request, budget=pair_budget, cancel_check=exhausted)
        contexts = {}
        for decision in evidence["receipt"]["continuation"]["decisions"]:
            if pair_state.decision_kind == "project":
                projects = evidence["receipt"]["continuation"].get("projectActions", ())
                if len(pair_state.legal_actions) != 1 or len(pair_prefix.project_actions) >= len(projects):
                    raise ValueError("paired quality requires a forced owned project continuation")
                action = ProjectDeclarationActionV1.build(projects[len(pair_prefix.project_actions)]).code
                if action != pair_state.legal_actions[0]["code"]:
                    raise ValueError("paired quality project continuation substitution")
                pair_prefix = normalizer.append_normalized_action(pair_prefix, pair_state, action)
                pair_state = normalizer.normalize_play_or_project(pair_prefix,
                    maximum_seconds=pair_remaining(), cancel_check=exhausted)
            before_prefix, before_state = pair_prefix, pair_state
            code = LegalPlayActionV1(Card.parse(decision["action"]["card"]),
                decision["action"].get("ekka", False)).code
            pair_prefix = normalizer.append_normalized_action(pair_prefix, pair_state, code)
            play_index = len(before_prefix.play_actions)
            recorded_boundary = final if value["operation"] == "audit_recorded" else submitted
            if (recorded_boundary is not None
                    and play_index < len(recorded_boundary.play_actions)
                    and before_prefix.play_actions == recorded_boundary.play_actions[:play_index]):
                recorded = recorded_boundary.play_actions[play_index]
                if (recorded.get("actor", before_state.next_actor) == before_state.next_actor
                        and LegalPlayActionV1(Card.parse(recorded["card"]), recorded.get("ekka", False)).code == code):
                    # Bind the same certified choice to its immutable authored syntax.
                    # Card codes and integer IDs must not silently rewrite trace hashes.
                    document = pair_prefix.to_dict()
                    document.pop("sha256")
                    document["playActions"] = [*document["playActions"][:-1], dict(recorded)]
                    at_boundary = (play_index + 1 == len(recorded_boundary.play_actions)
                        and len(pair_prefix.project_actions) == len(recorded_boundary.project_actions))
                    document["stateAssertions"] = dict(recorded_boundary.state_assertions) if at_boundary else {}
                    document["projectAssertions"] = [dict(row) for row in recorded_boundary.project_assertions] if at_boundary else []
                    pair_prefix = ExactTracePrefixV1.build(document)
            pair_state = normalizer.normalize_play_or_project(pair_prefix,
                maximum_seconds=pair_remaining(), cancel_check=exhausted)
            contexts[decision["playIndex"]] = (before_prefix, before_state, pair_prefix, pair_state)
        pair_remaining()
        paired_contexts[digest] = contexts
    previous_prefix, previous_state = source_prefix.sha256, None
    index, project_index = start, len(source_prefix.project_actions)
    decision_normalizer = None

    def normalize_decision(owned_prefix, *, maximum_seconds, cancel_check, automatic_stage=False):
        from .ortools_exact_state import OrToolsExactStateNormalizerV1, OrToolsNormalizedExactStateV1, _effective_prefix_hash
        nonlocal decision_normalizer
        digest = row["proofArtifactHash"]
        cached = (verified_normalizations or {}).get(digest) if digest in verified_artifacts else None
        if cached is not None:
            if not isinstance(cached, OrToolsNormalizedExactStateV1):
                raise ValueError("owned quality normalization has invalid type")
            cached.require_integrity()
            if (cached.rules_hash != rules_hash or cached.prefix_hash != _effective_prefix_hash(owned_prefix)
                    or cached.decision_kind != kind or cached.exact_state_id != row["exactStateId"]):
                raise ValueError("owned quality normalization context substitution")
            return cached
        if decision_normalizer is None:
            decision_normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
        if automatic_stage:
            return decision_normalizer.normalize_play_or_project(owned_prefix,
                maximum_seconds=maximum_seconds, cancel_check=cancel_check)
        return decision_normalizer.normalize(owned_prefix,
            decision_point={"kind": kind, **({"selector": {"owner": row["actor"]}} if kind == "project" else {})},
            maximum_seconds=maximum_seconds, cancel_check=cancel_check)

    for row in rows:
        remaining()
        kind = row.get("decisionKind", "play")
        if kind not in ({"bid"} if auction else {"project", "play"}):
            raise ValueError("decision quality decision kind mismatch")
        if (project_index >= len(final.project_actions) if kind == "project" else index >= len(final_actions)):
            raise ValueError("decision quality action coverage mismatch")
        action = final.project_actions[project_index] if kind == "project" else final_actions[index]
        if kind == "project":
            action_code = ProjectDeclarationActionV1.build(action).code
        elif kind == "play":
            action_code = LegalPlayActionV1(Card.parse(action["card"]), action.get("ekka", False)).code
        else:
            action_code = action["action"] + (":" + action["trumpSuit"] if action.get("trumpSuit") is not None else "")
            if action_code == "hokum" and index < len(Seat):
                artifact = artifacts.get(row.get("proofArtifactHash"), {})
                purchases = {model.get("purchaseSuit") for world in
                    artifact.get("allocationModel", {}).get("nativeAuctionModels", [])
                    for model in world.get("models", [])}
                if artifact.get("proofModel", {}).get("kind") in {"native_bid_selected_action_quality_model_v1", "native_observed_bid_quality_model_v1"}:
                    purchases = {item["code"].split(":", 1)[1] for item in
                        artifact.get("allocationModel", {}).get("normalizedState", {}).get("legalActions", [])
                        if isinstance(item, Mapping) and isinstance(item.get("code"), str)
                        and item["code"].startswith("hokum:")}
                if len(purchases) != 1 or next(iter(purchases)) not in {suit.name.lower() for suit in Suit}:
                    raise ValueError("implicit Hokum action lacks a unique native purchase suit")
                action_code += ":" + next(iter(purchases))
        if row.get("action") != action_code or action.get("actor", row.get("actor")) != row.get("actor"):
            raise ValueError("auction quality recorded action mismatch")
        if (row.get("sha256") != canonical_sha256({key: item for key, item in row.items() if key != "sha256"})
                or row.get("bidIndex" if auction else "playIndex") != index
                or row.get("sourcePrefixHash") != previous_prefix
                or (previous_state is not None and row.get("exactStateId") != previous_state)):
            raise ValueError("auction quality decision chain mismatch")
        artifact = artifacts.get(row.get("proofArtifactHash"))
        if not isinstance(artifact, Mapping):
            raise ValueError("auction quality decision artifact is missing")
        expected = policy.request_for(kind, row["actor"],
            play_index=index if kind == "play" else None,
            execution_budget=execution_budget).to_dict() if policy is not None else request.to_dict()
        expected.pop("sha256")
        expected["targetTeam"] = Seat.parse(row["actor"]).team.name.lower()
        if (execution_budget is not None and execution_budget.exact_tail_proof
                and kind == "play" and expected["mode"] == "exact"
                and expected["visibility"] == "actor_information_set"
                and artifact["proofModel"].get("kind") == "native_observed_play_quality_model_v1"):
            raise ValueError("exact actor-visible tail requires complete hidden-world coverage, not current-deal bounds")
        audited = value["operation"] == "audit_recorded" or submitted is not None and (
            index < len(submitted.auction_actions) if auction else
            index < len(submitted.play_actions) or project_index < len(submitted.project_actions))
        if audited:
            expected["selectedAction"] = row["action"]
        if artifact.get("request") != ScenarioProofRequestV1.build(expected).to_dict():
            raise ValueError("auction quality decision request mismatch")
        bounded_selected = artifact["proofModel"].get("kind") == "native_bounded_play_quality_model_v1"
        if ((kind in {"play", "project"} and expected.get("mode") == "exact"
                and expected.get("visibility") == "actor_information_set"
                and artifact["proofModel"].get("kind") in {
                    "actor_information_set_cegis_model_v1", "native_observed_play_quality_model_v1", "native_observed_project_quality_model_v1"}
                and (kind == "play" or artifact["proofModel"]["kind"] == "native_observed_project_quality_model_v1"))
                or (bounded_selected and kind == "play" and expected.get("mode") == "bounded"
                    and expected.get("visibility") == "actor_information_set")):
            import time
            from .ortools_exact_state import OrToolsExactStateNormalizerV1
            from .ortools_nonplay_graphs import normalize_escalation_recall_v1
            if not isinstance(rules, RuleBook):
                raise ValueError("exact observed quality requires frozen rules")
            recall = normalize_escalation_recall_v1(private_recall)
            if value.get("privateRecallHash") != (canonical_sha256(recall) if recall else None):
                raise ValueError("exact observed quality private recall mismatch")
            owned_request = ScenarioProofRequestV1.build(expected)
            limits = value["limits"]
            budget_raw = value.get("proofBudgetProfiles", {}).get(owned_request.budget_profile)
            budget = ScenarioProofBudgetV1.build(owned_request.budget_profile, budget_raw) if budget_raw is not None else None
            if row.get("proofBudgetHash") != (budget.sha256 if budget is not None else None):
                raise ValueError("exact observed quality proof budget substitution")
            nodes, depth, worlds = (limits[key] for key in
                ("maximumNodesPerDecision", "maximumDepth", "maximumWorldsPerDecision"))
            if execution_budget is not None:
                worlds = execution_budget.resolve_hidden_world_cap(worlds,
                    play_index=index if kind == "play" else None)
            seconds = limits["maximumDecisionSeconds"]
            if budget is not None:
                nodes = min(nodes, budget.maximum_nodes, budget.maximum_state_space)
                depth, worlds = min(depth, budget.maximum_depth), min(worlds, budget.maximum_hidden_worlds)
                seconds = min(seconds, budget.maximum_wall_seconds)
            read_remaining = _quality_readback_clock(min(seconds, remaining()), cancel_check)
            deadline = time.monotonic() + read_remaining()
            cpu_end = time.process_time() + (budget.maximum_cpu_seconds if budget is not None else seconds)

            def exhausted():
                return bool(cancel_check and cancel_check()) or time.monotonic() >= deadline or time.process_time() >= cpu_end

            raw = source_prefix.to_dict()
            raw.pop("sha256")
            raw["playActions"] = final.to_dict()["playActions"][:index]
            if "projectActions" in final.to_dict():
                raw["projectActions"] = final.to_dict()["projectActions"][:project_index]
            if index != start or project_index != len(source_prefix.project_actions):
                raw["stateAssertions"], raw["projectAssertions"] = {}, []
                if value["operation"] == "generate" and project_index == len(source_prefix.project_actions):
                    raw["projectAssertions"] = [dict(item) for item in source_prefix.project_assertions]
                elif submitted is not None and index >= len(submitted.play_actions) and project_index == len(submitted.project_actions):
                    raw["projectAssertions"] = [dict(item) for item in submitted.project_assertions]
                boundary = submitted if submitted is not None else final if value["operation"] == "audit_recorded" else None
                if boundary is not None and index == len(boundary.play_actions) and project_index == len(boundary.project_actions):
                    raw["stateAssertions"] = dict(boundary.state_assertions)
                    raw["projectAssertions"] = [dict(item) for item in boundary.project_assertions]
            owned_prefix = ExactTracePrefixV1.build(raw)
            owned_state = normalize_decision(
                owned_prefix,
                maximum_seconds=read_remaining(), cancel_check=exhausted)
            if owned_prefix.sha256 != row["sourcePrefixHash"] or owned_state.exact_state_id != row["exactStateId"]:
                raise ValueError("exact observed quality source state substitution")
            remaining()
            remaining_seconds = min(deadline - time.monotonic(), cpu_end - time.process_time(), read_remaining())
            if remaining_seconds <= 0:
                raise TimeoutError("exact observed quality read budget exhausted")
            if bounded_selected:
                if budget is None or budget.native_observed is None:
                    raise ValueError("bounded selected play requires caller-owned native budget")
                if row["proofArtifactHash"] not in verified_artifacts:
                    from .ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
                    OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True,
                        prune_own_project_pass_worlds=bool(execution_budget and execution_budget.prune_own_project_pass_worlds),
                    )._verify_observed_play_selection(artifact["graph"], owned_state, owned_request,
                        budget=budget, maximum_seconds=remaining_seconds, cancel_check=exhausted,
                        private_recall=private_recall, maximum_worlds=worlds,
                        maximum_depth=depth, project_prefix=owned_prefix)
                rebuilt = _project_native_bounded_play_quality_artifact_v1(artifact["graph"], rules=rules,
                    source_prefix=owned_prefix, normalized=owned_state, request=owned_request, budget=budget)
                if canonical_sha256(rebuilt) != canonical_sha256(artifact):
                    raise ValueError("bounded selected play artifact binding mismatch")
                read_remaining()
            elif artifact["proofModel"]["kind"] in {"native_observed_play_quality_model_v1", "native_observed_project_quality_model_v1"}:
                if (budget is None or escalation_events is None or opportunity_order is None
                        or search_budget_profiles is None or owned_request.budget_profile not in search_budget_profiles
                        or value.get("escalationHistoryHash") != canonical_sha256(escalation_events)):
                    raise ValueError("selected play quality requires caller-owned history and frozen budgets")
                if any(type(artifact["graph"].get(key, {}).get("limits", {}).get("depth")) is not int
                        or not 0 < artifact["graph"][key]["limits"]["depth"] <= min(depth, budget.maximum_horizon_actions)
                        for key in ("selectedProof", "bestProof")):
                    raise ValueError("selected play quality exceeds caller depth")
                if row["proofArtifactHash"] in verified_artifacts:
                    for premise in ("selectedProof", "bestProof"):
                        validate_native_constant_escalation_budget_v1(artifact["graph"][premise], budget,
                            maximum_conflicts=search_budget_profiles[owned_request.budget_profile]["maximumConflicts"])
                    expected_artifact = _project_native_observed_quality_artifact_v1(artifact["graph"], rules=rules,
                        source_prefix=owned_prefix, normalized=owned_state, request=owned_request, budget=budget,
                        opportunity_order=opportunity_order, escalation_events=escalation_events, decision_kind=kind)
                    if canonical_sha256(artifact) != canonical_sha256(expected_artifact):
                        raise ValueError("observed quality owned artifact binding mismatch")
                else:
                    validate_native_observed_play_quality_artifact_v1(artifact, rules=rules, source_prefix=owned_prefix,
                        normalized=owned_state, request=owned_request, budget=budget,
                        opportunity_order=opportunity_order, escalation_events=escalation_events,
                        maximum_conflicts=search_budget_profiles[owned_request.budget_profile]["maximumConflicts"],
                        maximum_seconds=remaining_seconds, cancel_check=exhausted, decision_kind=kind)
            else:
                token = _exact_observed_verification_token_v1(artifact, rules=rules, source_prefix=owned_prefix,
                    normalized=owned_state, request=owned_request, maximum_nodes=nodes, maximum_depth=depth,
                    maximum_worlds=worlds, declared_maximum_seconds=seconds, private_recall=private_recall,
                    policy_search=budget.policy_search if budget is not None else None)
                if token not in verified_artifacts:
                    reverify_native_exact_observed_play_artifact_v1(artifact, rules=rules, source_prefix=owned_prefix,
                        normalized=owned_state, request=owned_request, maximum_nodes=nodes, maximum_depth=depth,
                        maximum_worlds=worlds, maximum_seconds=remaining_seconds, declared_maximum_seconds=seconds,
                        private_recall=private_recall, cancel_check=exhausted,
                        policy_search=budget.policy_search if budget is not None else None)
                    if isinstance(verified_artifacts, set):
                        verified_artifacts.add(token)
        previous_prefix, previous_state = row["nextPrefixHash"], row["nextExactStateId"]
        if kind == "project":
            project_index += 1
        else:
            index += 1
        if (submitted is not None and audited and
                (index == len(submitted.auction_actions) if auction else
                 index == len(submitted.play_actions) and project_index == len(submitted.project_actions))
                and previous_prefix != submitted.sha256):
            raise ValueError("quality submitted boundary state mismatch")
        if artifact["proofModel"].get("kind") in {"native_observed_play_quality_model_v1", "native_observed_project_quality_model_v1",
                "native_bounded_play_quality_model_v1"}:
            quality = artifact["proofModel"]["actionQuality"]
            if (auction or kind not in {"play", "project"} or expected["mode"] != ("bounded" if bounded_selected else "exact")
                    or artifact["proofModel"]["kind"] != ("native_bounded_play_quality_model_v1"
                        if bounded_selected and kind == "play" else f"native_observed_{kind}_quality_model_v1")
                    or expected["visibility"] != "actor_information_set"
                    or row.get("actor") != artifact["proofModel"]["rootActorSeat"]
                    or row.get("targetTeam") != expected["targetTeam"]
                    or row.get("actionQualityHash") != quality["sha256"]
                    or row.get("action") != quality["actualAction"]
                    or type(row.get("actualActionOptimal")) is not bool
                    or row["actualActionOptimal"] is not quality["actualActionOptimal"]
                    or type(row.get("regret")) is not int or row["regret"] != quality["regret"]
                    or "optimalActionCount" not in row or row["optimalActionCount"] is not None
                    or any(type(row.get(key)) is not int or row[key] != quality[key] for key in
                        ("optimalActionCountLowerBound", "optimalActionCountUpperBound"))
                    or row.get("selectionReason") != ("recorded_action" if audited else "native_actor_information_set_best_value")
                    or row.get("selectionTieBreak") is not None
                    or not audited and row["actualActionOptimal"] is not True):
                raise ValueError("selected play quality decision binding mismatch")
            continue
        if artifact["proofModel"].get("kind") == "native_complete_state_policy_pair_quality_v1":
            digest = artifact["proofModel"].get("pairEvidenceHash")
            if auction or kind != "play" or digest not in paired_contexts:
                raise ValueError("paired quality decision lacks owned evidence")
            owned_prefix, owned_state, next_prefix, next_state = paired_contexts[digest][row["playIndex"]]
            if (owned_prefix.sha256 != row["sourcePrefixHash"] or owned_state.exact_state_id != row["exactStateId"]
                    or next_prefix.sha256 != row["nextPrefixHash"] or next_state.exact_state_id != row["nextExactStateId"]):
                raise ValueError("paired quality continuation substitution")
            owned_request = ScenarioProofRequestV1.build(expected)
            evidence = value["pairedPolicyEvidence"][digest]
            if (owned_request.budget_profile != evidence["receipt"]["budgetProfile"]
                    or row.get("proofBudgetHash") != evidence["receipt"]["proofBudgetHash"]):
                raise ValueError("paired quality budget substitution")
            rebuilt = build_native_paired_play_action_artifact_v1(evidence["receipt"], pair_hash=digest,
                source_prefix=owned_prefix, normalized=owned_state, request=owned_request)
            quality = rebuilt["proofModel"]["actionQuality"]
            if (canonical_sha256(rebuilt) != canonical_sha256(artifact)
                    or row.get("targetTeam") != owned_request.target_team
                    or row.get("actionQualityHash") != quality["sha256"]
                    or row.get("actualActionOptimal") is not True
                    or type(row.get("regret")) is not int or row["regret"] != 0
                    or "optimalActionCount" not in row or row["optimalActionCount"] is not None
                    or row.get("selectionReason") != ("recorded_action" if audited else "native_complete_state_equilibrium_path")
                    or row.get("selectionTieBreak") is not None
                    or any(type(row.get(key)) is not int or row[key] != quality[key] for key in
                        ("optimalActionCountLowerBound", "optimalActionCountUpperBound"))):
                raise ValueError("paired quality selected-action binding mismatch")
            continue
        if artifact["proofModel"].get("kind") in {"native_bid_selected_action_quality_model_v1", "native_observed_bid_quality_model_v1"}:
            if not auction:
                raise ValueError("selected bid quality requires an auction action")
            owned_request = ScenarioProofRequestV1.build(expected)
            if artifact["proofModel"]["kind"] == "native_observed_bid_quality_model_v1":
                allocation = artifact["allocationModel"]
                if (allocation.get("sourcePrefixHash") != row["sourcePrefixHash"]
                        or allocation.get("exactStateId") != row["exactStateId"]):
                    raise ValueError("observed bid quality decision ownership mismatch")
                profile = owned_request.budget_profile
                if (not isinstance(execution_budget, ScenarioQualityExecutionBudgetV1)
                        or search_budget_profiles is None or profile not in search_budget_profiles):
                    raise ValueError("observed bid quality requires frozen execution and search profiles")
                budget = ScenarioProofBudgetV1.build(profile, value["proofBudgetProfiles"][profile])
                if row.get("proofBudgetHash") != budget.sha256:
                    raise ValueError("observed bid quality proof budget substitution")
                # Reuse only the exact builder-owned digest and frozen context
                # in the private immediate wrapper. Public reads have no token.
                verified_context = canonical_sha256({
                    "artifactHash": row["proofArtifactHash"], "proofBudgetHash": budget.sha256,
                    "executionBudgetHash": ScenarioQualityExecutionBudgetV1.build(execution_budget.to_dict()).sha256,
                    "maximumConflicts": search_budget_profiles[profile]["maximumConflicts"],
                })
                if verified_context not in verified_artifacts:
                    validate_native_observed_bid_quality_artifact_v1(artifact, rules=rules,
                        request=owned_request, source_prefix_hash=row["sourcePrefixHash"], exact_state_id=row["exactStateId"],
                        budget=budget, maximum_conflicts=search_budget_profiles[profile]["maximumConflicts"],
                        opportunity_order=execution_budget.escalation_opportunity_order,
                        execution_budget=execution_budget,
                        **({"opponent_play_policy_id": execution_budget.bid_opponent_play_policy_id}
                           if execution_budget.bid_opponent_play_policy_id is not None else {}),
                        maximum_seconds=min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds, remaining()),
                        cancel_check=cancel_check)
            else:
                native_incumbent_models = None
                # The builder already checked this exact digest against its owned
                # native models. Only the private immediate wrapper can reuse it;
                # public/stored reads always reconstruct and re-prove the bound.
                owned_native_incumbent = (
                    "incumbentNativePlayPolicyId" in artifact.get("graph", {})
                    and row["proofArtifactHash"] in verified_artifacts
                )
                if "incumbentNativePlayPolicyId" in artifact.get("graph", {}) and not owned_native_incumbent:
                    profile = owned_request.budget_profile
                    if search_budget_profiles is None:
                        raise ValueError("selected bid quality requires frozen search limits for native readback")
                    try:
                        budget = ScenarioProofBudgetV1.build(profile, value["proofBudgetProfiles"][profile])
                        conflicts = search_budget_profiles[profile]["maximumConflicts"]
                    except (KeyError, TypeError) as exc:
                        raise ValueError("selected bid quality frozen budget is unavailable") from exc
                    if row.get("proofBudgetHash") != budget.sha256:
                        raise ValueError("selected bid quality proof budget substitution")
                    native_incumbent_models = reverify_native_bid_dominance_quality_v1(artifact,
                        rules=rules, request=owned_request, budget=budget, maximum_conflicts=conflicts,
                        execution_budget=execution_budget,
                        opportunity_order=(execution_budget.escalation_opportunity_order
                            if isinstance(execution_budget, ScenarioQualityExecutionBudgetV1) else opportunity_order),
                        maximum_seconds=min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds, remaining()),
                        cancel_check=cancel_check)
                if not owned_native_incumbent:
                    validate_native_bid_dominance_quality_artifact_v1(artifact, rules=rules,
                        request=owned_request, source_prefix_hash=row["sourcePrefixHash"], exact_state_id=row["exactStateId"],
                        **({"native_incumbent_models": native_incumbent_models} if native_incumbent_models is not None else {}))
                if search_budget_profiles is not None:
                    profile = owned_request.budget_profile
                    try:
                        budget = ScenarioProofBudgetV1.build(profile, value["proofBudgetProfiles"][profile])
                        conflicts = search_budget_profiles[profile]["maximumConflicts"]
                    except (KeyError, TypeError) as exc:
                        raise ValueError("selected bid quality frozen budget is unavailable") from exc
                    if row.get("proofBudgetHash") != budget.sha256:
                        raise ValueError("selected bid quality proof budget substitution")
                    validate_native_bid_dominance_budget_v1(artifact["graph"], budget,
                        maximum_conflicts=conflicts,
                        opportunity_order=(execution_budget.escalation_opportunity_order
                            if isinstance(execution_budget, ScenarioQualityExecutionBudgetV1) else opportunity_order),
                        execution_budget=execution_budget)
                    if row["proofArtifactHash"] not in verified_artifacts and native_incumbent_models is None:
                        reverify_native_bid_dominance_quality_v1(artifact, rules=rules, request=owned_request,
                            budget=budget, maximum_conflicts=conflicts, execution_budget=execution_budget,
                            opportunity_order=(execution_budget.escalation_opportunity_order
                                if isinstance(execution_budget, ScenarioQualityExecutionBudgetV1) else opportunity_order),
                            maximum_seconds=min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds, remaining()),
                            cancel_check=cancel_check)
                else:
                    raise ValueError("selected bid quality requires frozen search limits for native readback")
            quality = artifact["proofModel"]["actionQuality"]
            if (row.get("actor") != artifact["proofModel"]["rootActorSeat"]
                    or row.get("targetTeam") != expected["targetTeam"]
                    or row.get("actionQualityHash") != quality["sha256"]
                    or row.get("action") != quality["actualAction"]
                    or row.get("actualActionOptimal") is not True
                    or type(row.get("regret")) is not int or row["regret"] != 0
                    or "optimalActionCount" not in row or row["optimalActionCount"] is not None
                    or any(type(row.get(key)) is not int or row[key] != quality[key] for key in
                        ("optimalActionCountLowerBound", "optimalActionCountUpperBound"))
                    or row.get("selectionReason") != ("recorded_action" if audited else
                        "native_complete_state_best_value" if expected["visibility"] == "complete_state" else
                        "native_actor_information_set_best_value")
                    or row.get("selectionTieBreak") is not None):
                raise ValueError("selected bid quality decision binding mismatch")
            continue
        if artifact["proofModel"].get("kind") == "native_project_dominance_quality_model_v1":
            from .ortools_exact_state import OrToolsExactStateNormalizerV1
            generated = artifact["graph"].get("kind") == "native_last_project_selection_v1"
            if auction or kind != "project" or audited == generated or not isinstance(rules, RuleBook):
                raise ValueError("project dominance quality requires matching generated or recorded evidence")
            owned_request = ScenarioProofRequestV1.build(expected)
            profile = owned_request.budget_profile
            if search_budget_profiles is None or profile not in search_budget_profiles:
                raise ValueError("project dominance quality requires frozen search limits")
            conflicts = search_budget_profiles[profile].get("maximumConflicts")
            caller_depth = value.get("limits", {}).get("maximumDepth")
            if type(conflicts) is not int or conflicts < 1 or type(caller_depth) is not int or caller_depth < 1:
                raise ValueError("project dominance quality conflict limit is invalid")
            budget = ScenarioProofBudgetV1.build(profile, value["proofBudgetProfiles"][profile])
            owned_prefix = ExactTracePrefixV1.build(artifact["allocationModel"]["sourcePrefix"])
            if owned_prefix.sha256 != row["sourcePrefixHash"] or row.get("proofBudgetHash") != budget.sha256:
                raise ValueError("project dominance quality prefix or budget substitution")
            proof_remaining = _quality_readback_clock(min(budget.maximum_wall_seconds,
                budget.maximum_cpu_seconds, remaining()), cancel_check)
            normalized = normalize_decision(
                owned_prefix, maximum_seconds=proof_remaining(), cancel_check=cancel_check, automatic_stage=True)
            if normalized.exact_state_id != row["exactStateId"]:
                raise ValueError("project dominance quality state substitution")
            validate_native_project_dominance_quality_artifact_v1(artifact, rules=rules,
                source_prefix=owned_prefix, normalized=normalized, request=owned_request, budget=budget,
                maximum_conflicts=conflicts, maximum_depth=caller_depth,
                maximum_seconds=proof_remaining(), cancel_check=cancel_check)
            proof_remaining()
        if artifact["proofModel"].get("kind") == "native_complete_state_root_outcome_v1":
            from .ortools_exact_state import OrToolsExactStateNormalizerV1
            if not isinstance(rules, RuleBook):
                raise ValueError("compact quality read requires frozen rules")
            allocation = artifact["allocationModel"]
            owned_prefix = ExactTracePrefixV1.build(allocation["sourcePrefix"])
            if owned_prefix.sha256 != row["sourcePrefixHash"]:
                raise ValueError("compact quality source prefix substitution")
            owned_request = ScenarioProofRequestV1.build(expected)
            budget = ScenarioProofBudgetV1.build(owned_request.budget_profile,
                value["proofBudgetProfiles"][owned_request.budget_profile])
            proof_remaining = _quality_readback_clock(min(budget.maximum_wall_seconds,
                budget.maximum_cpu_seconds, remaining()), cancel_check)
            normalized = normalize_decision(
                owned_prefix, maximum_seconds=proof_remaining(), cancel_check=cancel_check, automatic_stage=True)
            from .ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
            if row["proofArtifactHash"] not in verified_artifacts:
                fresh_roots = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._prove_complete_state_play(
                    normalized, owned_prefix, owned_request, budget=budget,
                    maximum_nodes=artifact["graph"]["preparation"]["maximumNodeObservations"],
                    maximum_seconds=proof_remaining(), cancel_check=cancel_check)
                if (_native_observed_bounds_semantics(fresh_roots, maximum_seconds=budget.maximum_wall_seconds)
                        != _native_observed_bounds_semantics(artifact["graph"], maximum_seconds=budget.maximum_wall_seconds)):
                    raise ValueError("compact quality roots differ from fresh native proof")
            rebuilt = build_native_complete_state_play_quality_artifact_v1(artifact["graph"], rules=rules,
                source_prefix=owned_prefix, normalized=normalized, request=owned_request, budget=budget)
            if canonical_sha256(rebuilt) != canonical_sha256(artifact):
                raise ValueError("compact quality artifact binding mismatch")
            proof_remaining()
        quality = artifact["proofModel"].get("actionQuality", {})
        if not isinstance(quality, Mapping) or not isinstance(artifact.get("proof"), Mapping):
            raise ValueError("decision quality proof must be an object")
        if quality.get("optimalActionsComplete", True) is not True:
            raise ValueError("decision quality cannot infer an exact count from partial optimal actions")
        if type(row.get("optimalActionCount")) is not int or row["optimalActionCount"] < 1:
            raise ValueError("decision quality optimal action count must be a positive integer")
        if artifact["proofModel"].get("kind") == "forced_action_optimality_model_v1":
            forced = artifact["proof"]
            roots = forced.get("rootActions")
            if (not {"claimSatisfied", "winningRootActionCount"} <= forced.keys()
                    or not {"bestValue", "actualValue"} <= quality.keys()
                    or not isinstance(roots, list) or len(roots) != 1
                    or not isinstance(roots[0], Mapping) or not {"value", "satisfies"} <= roots[0].keys()
                    or roots[0].get("action") != row.get("action")
                    or roots[0].get("value") is not None or roots[0].get("satisfies") is not None
                    or forced.get("completionScope") != "action_optimality_only"
                    or forced.get("claimSatisfied") is not None
                    or forced.get("winningRootActionCount") is not None
                    or type(forced.get("winningRootActionLowerBound")) is not int
                    or forced["winningRootActionLowerBound"] != 0
                    or type(forced.get("winningRootActionUpperBound")) is not int
                    or forced["winningRootActionUpperBound"] != 1
                    or quality.get("bestValue") is not None or quality.get("actualValue") is not None
                    or quality.get("optimalActions") != [row.get("action")]
                    or type(quality.get("regret")) is not int or quality["regret"] != 0):
                raise ValueError("forced optimality cannot claim an evaluated outcome")
        policy_conditioned_quality = (
            artifact["proofModel"].get("kind") == "observer_safe_play_policy_model_v1"
        )
        if "nativeRootArtifactHash" in artifact["proofModel"]:
            graph = artifact["graph"]
            if (artifact["proofModel"]["nativeRootArtifactHash"] != graph.get("sha256")
                    or graph.get("sha256") != canonical_sha256({key: item for key, item in graph.items() if key != "sha256"})):
                raise ValueError("native root artifact reference mismatch")
        quality_core = {key: item for key, item in quality.items() if key != "sha256"}
        quality_core["modelProto"] = artifact["proofModel"].get("modelProto")
        identities = artifact["proof"].get("identities", {})
        worlds = artifact["allocationModel"].get("nativeAuctionModels", [])
        completion = "exact_complete" if expected["mode"] == "exact" else "bounded_complete"
        if expected["visibility"] == "complete_state":
            allocation = artifact["allocationModel"]
            if (quality.get("scope") != "compiled_complete_state"
                    or allocation.get("kind") != "native_complete_state_quality_v1"
                    or allocation.get("sourcePrefixHash") != row["sourcePrefixHash"]
                    or allocation.get("exactStateId") != row["exactStateId"]
                    or allocation.get("rulesHash") != rules_hash
                    or allocation.get("sha256") != canonical_sha256({
                        key: item for key, item in allocation.items() if key != "sha256"})):
                raise ValueError("complete-state quality source binding mismatch")
        if policy_conditioned_quality:
            from .ortools_play_tail import NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS
            graph = artifact["graph"]
            allocation = artifact["allocationModel"]
            if (expected["mode"] != "bounded"
                    or expected["visibility"] != "actor_information_set"
                    or kind != "play"
                    or quality.get("scope") != "observer_safe_policy_conditioned"
                    or quality.get("policyId") not in NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS
                    or quality.get("observerInformationHash") != graph.get("observationHash")
                    or allocation.get("observationHash") != graph.get("observationHash")
                    or graph.get("sha256") != canonical_sha256({key: item for key, item in graph.items() if key != "sha256"})
                    or allocation.get("sha256") != canonical_sha256({key: item for key, item in allocation.items() if key != "sha256"})
                    or quality.get("sha256") != row.get("actionQualityHash")
                    or quality.get("sha256") != canonical_sha256(quality_core)
                    or identities.get("proofRequestHash") != artifact["request"]["sha256"]
                    or identities.get("modelHash") != canonical_sha256(artifact["proofModel"])
                    or identities.get("graphHash") != graph.get("sha256")
                    or artifact["proofModel"].get("rootActorSeat") != row.get("actor")
                    or row.get("targetTeam") != expected["targetTeam"]
                    or quality.get("rootActorRelation") != "target"
                    or row.get("regret") != quality.get("regret")
                    or row.get("optimalActionCount") != len(quality.get("optimalActions", []))
                    or artifact["proof"].get("selectedAction") != row.get("action")
                    or quality.get("actualAction") != row.get("action")
                    or quality.get("actualActionOptimal") is not row.get("actualActionOptimal")
                    or type(row.get("actualActionOptimal")) is not bool
                    or artifact["proof"].get("completionStatus") != "bounded_complete"
                    or row.get("selectionReason") != ("recorded_action" if audited else "native_observer_safe_policy_value")
                    or (not audited and row["actualActionOptimal"] is not True)):
                raise ValueError("observer-safe policy quality action binding mismatch")
        elif (quality.get("sha256") != row.get("actionQualityHash")
                or quality.get("sha256") != canonical_sha256(quality_core)
                or identities.get("proofRequestHash") != artifact["request"]["sha256"]
                or identities.get("modelHash") != canonical_sha256(artifact["proofModel"])
                or identities.get("graphHash") != artifact["graph"].get("sha256")
                or artifact["proofModel"].get("rootActorSeat") != row.get("actor")
                or row.get("targetTeam") != expected["targetTeam"]
                or quality.get("rootActorRelation") != "target"
                or row.get("regret") != quality.get("regret")
                or row.get("optimalActionCount") != len(quality.get("optimalActions", []))
                or (auction and (not worlds or any(world.get("sourcePrefixHash") != row["sourcePrefixHash"] for world in worlds)))
                or artifact["proof"].get("selectedAction") != row.get("action")
                or quality.get("actualAction") != row.get("action")
                or quality.get("actualActionOptimal") is not row.get("actualActionOptimal")
                or type(row.get("actualActionOptimal")) is not bool
                or artifact["proof"].get("completionStatus") != completion
                or row.get("selectionReason") != ("recorded_action" if audited else
                    "native_complete_state_best_value" if expected["visibility"] == "complete_state"
                    else "native_actor_information_set_best_value")
                or (not audited and row["actualActionOptimal"] is not True)):
            raise ValueError("auction quality action proof binding mismatch")
    if previous_prefix != final.sha256 or (rows and previous_state != value.get("finalExactStateId")):
        raise ValueError("auction quality final state binding mismatch")
    if value.get("status") == "complete" and any(not row["actualActionOptimal"] for row in rows):
        raise ValueError("auction quality incomplete qualification binding")
    _validate_raw_point_quality_companions_v1(value, rules=rules,
        execution_budget=execution_budget, remaining=remaining, cancel_check=cancel_check,
        private_recall=private_recall, verified_artifacts=verified_artifacts)
    _validate_raw_point_quality_companions_v1(value, rules=rules,
        execution_budget=execution_budget, remaining=remaining, cancel_check=cancel_check,
        private_recall=private_recall, evidence_key="roundWinEvidence", verified_artifacts=verified_artifacts)
    _validate_basic_policy_quality_companions_v1(value, rules=rules,
        execution_budget=execution_budget, remaining=remaining, cancel_check=cancel_check)
    _validate_tie_rule_quality_companions_v1(value, rules=rules,
        execution_budget=execution_budget, remaining=remaining, cancel_check=cancel_check,
        private_recall=private_recall, verified_artifacts=verified_artifacts)
    _validate_project_inference_quality_companions_v1(value, rules=rules, execution_budget=execution_budget,
        search_budget_profiles=search_budget_profiles, remaining=remaining, cancel_check=cancel_check)
    remaining()
    if stage_token is not None and isinstance(verified_artifacts, set):
        verified_artifacts.add(stage_token)


def validate_native_auction_quality_evidence_v1(value, **kwargs) -> None:
    if value.get("kind") != "native_quality_auction_continuation_v1":
        raise ValueError("auction quality evidence kind mismatch")
    validate_native_decision_quality_evidence_v1(value, **kwargs)

def validate_native_escalation_quality_evidence_v1(
    value: Mapping[str, Any], *, source_prefix: ExactTracePrefixV1,
    rules_hash: str, request: ScenarioProofRequestV1,
    rules: RuleBook | None = None, search_budget_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    maximum_seconds=None, cancel_check=None,
) -> None:
    """Freshly verify external/stored escalation evidence."""
    _validate_native_escalation_quality_evidence_v1(value, source_prefix=source_prefix,
        rules_hash=rules_hash, request=request, rules=rules, search_budget_profiles=search_budget_profiles,
        maximum_seconds=maximum_seconds, cancel_check=cancel_check)


def _validate_native_escalation_quality_evidence_v1(
    value: Mapping[str, Any], *, source_prefix: ExactTracePrefixV1,
    rules_hash: str, request: ScenarioProofRequestV1,
    rules: RuleBook | None = None, search_budget_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    maximum_seconds=None, cancel_check=None, verified_artifacts=frozenset(),
) -> None:
    """Bind retained opportunities; compact certificates require fresh native verification."""
    remaining = _quality_readback_clock(maximum_seconds, cancel_check)
    remaining()
    if (value.get("kind") != "native_quality_escalation_continuation_v1"
            or value.get("sha256") != canonical_sha256({k: v for k, v in value.items() if k != "sha256"})
            or value.get("sourcePrefixHash") != source_prefix.sha256
            or value.get("rulesHash") != rules_hash
            or value.get("qualityRequestHash") != request.sha256
            or value.get("operation") not in {"generate", "audit_recorded", "audit_then_generate"}):
        raise ValueError("escalation quality source ownership mismatch")
    artifacts, rows = value["artifacts"], value["decisions"]
    if value["operation"] == "audit_then_generate":
        recorded = value.get("recordedEvents")
        if not isinstance(recorded, list):
            raise ValueError("escalation quality recorded prefix is missing")
        common = min(len(recorded), len(value["events"]))
        if (recorded[:common] != value["events"][:common]
                or value["status"] == "complete" and len(value["events"]) < len(recorded)):
            raise ValueError("escalation quality recorded prefix mismatch")
    for digest, artifact in artifacts.items():
        if digest != artifact.get("sha256") or digest != canonical_sha256(
                {k: v for k, v in artifact.items() if k != "sha256"}):
            raise ValueError("escalation quality artifact checksum mismatch")
    current, events, recall = source_prefix, [], {}
    if source_prefix.state_assertions and value.get("initialNormalization") is not None:
        normalized_source = source_prefix.to_dict()
        normalized_source.pop("sha256")
        normalized_source["stateAssertions"] = {}
        current = ExactTracePrefixV1.build(normalized_source)
    for index, row in enumerate(rows):
        remaining()
        artifact = artifacts.get(row.get("artifactHash"))
        if (not isinstance(artifact, Mapping) or row.get("index") != index
                or row.get("sourcePrefixHash") != current.sha256
                or artifact.get("sourcePrefixHash") != current.sha256):
            raise ValueError("escalation quality decision chain mismatch")
        actor = Seat.parse(row["actor"])
        raw = request.to_dict()
        raw.pop("sha256")
        raw.pop("selectedAction", None)
        raw["targetTeam"] = actor.team.name.lower()
        if value["operation"] == "audit_recorded" or (
                value["operation"] == "audit_then_generate" and index < len(value["recordedEvents"])):
            recorded = value["recordedEvents"][index]
            if recorded["actor"] != row["actor"]:
                raise ValueError("escalation quality recorded actor mismatch")
            raw["selectedAction"] = recorded["actionCode"]
        if artifact.get("request") != ScenarioProofRequestV1.build(raw).to_dict():
            raise ValueError("escalation quality request mismatch")
        model, proof = artifact["proofModel"], artifact["proof"]
        identities = proof.get("identities", {})
        if (model.get("rootActorSeat") != row["actor"]
                or identities.get("proofRequestHash") != artifact["request"]["sha256"]
                or identities.get("modelHash") != canonical_sha256(model)
                or identities.get("graphHash") != artifact["graph"].get("sha256")):
            raise ValueError("escalation quality proof ownership mismatch")
        observed_native = model.get("kind") == "native_observed_escalation_quality_model_v1"
        if observed_native or model.get("kind") == "native_escalation_selected_action_quality_model_v1":
            if (not isinstance(rules, RuleBook) or rules.sha256 != rules_hash
                    or not isinstance(search_budget_profiles, Mapping)
                    or not isinstance(search_budget_profiles.get(request.budget_profile), Mapping)
                    or type(search_budget_profiles[request.budget_profile].get("maximumConflicts")) is not int
                    or search_budget_profiles[request.budget_profile]["maximumConflicts"] < 1):
                raise ValueError("selected escalation quality requires caller-owned frozen rules and search limits")
            from .ortools_exact_state import OrToolsExactStateNormalizerV1
            owned_request = ScenarioProofRequestV1.build(raw)
            budget = ScenarioProofBudgetV1.build(owned_request.budget_profile,
                value["proofBudgetProfiles"][owned_request.budget_profile])
            allocation = artifact["allocationModel"]
            if (source_prefix.escalation_actions or allocation.get("escalationEvents") != events
                    or allocation.get("sourcePrefix") != current.to_dict()):
                raise ValueError("selected escalation quality requires the complete actual history")
            allowance = min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds, remaining())
            normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(current,
                decision_point={"kind": "escalation", "selector": {"actingSeat": row["actor"]}},
                maximum_seconds=allowance, cancel_check=cancel_check)
            verify = (validate_native_observed_play_quality_artifact_v1 if observed_native
                      else validate_native_constant_escalation_quality_artifact_v1)
            owned_artifact = {key: artifact[key] for key in ("request", "proof", "proofModel", "graph", "allocationModel")}
            conflicts = search_budget_profiles[owned_request.budget_profile]["maximumConflicts"]
            if row["artifactHash"] in verified_artifacts:
                for premise in ((artifact["graph"]["selectedProof"], artifact["graph"]["bestProof"])
                                if observed_native else (artifact["graph"],)):
                    validate_native_constant_escalation_budget_v1(premise, budget, maximum_conflicts=conflicts)
                if observed_native:
                    expected = _project_native_observed_quality_artifact_v1(artifact["graph"], rules=rules,
                        source_prefix=current, normalized=normalized, request=owned_request, budget=budget,
                        opportunity_order=value["opportunityOrder"], escalation_events=events, decision_kind="escalation")
                    if canonical_sha256(expected) != canonical_sha256(owned_artifact):
                        raise ValueError("observed escalation owned artifact binding mismatch")
            else:
                verify(owned_artifact,
                    rules=rules, source_prefix=current, normalized=normalized, request=owned_request,
                    budget=budget, opportunity_order=value["opportunityOrder"], escalation_events=events,
                    maximum_seconds=min(allowance, remaining()), cancel_check=cancel_check,
                    maximum_conflicts=conflicts,
                    **({"decision_kind": "escalation"} if observed_native else {}))
        if artifact["status"] != "recorded":
            if index != len(rows) - 1 or value["status"] != artifact["status"]:
                raise ValueError("escalation quality incomplete status mismatch")
            break
        quality = model.get("actionQuality", {})
        core = {k: v for k, v in quality.items() if k != "sha256"}
        if not observed_native:
            core["modelProto"] = model.get("modelProto")
        transition = artifact["transition"]
        event = transition["event"]
        if (quality.get("sha256") != canonical_sha256(core)
                or quality.get("scope") != "compiled_" + request.visibility
                or quality.get("actualActionOptimal") is not True
                or quality.get("actualAction") != proof.get("selectedAction")
                or event.get("actor") != row["actor"]
                or event.get("actionCode") != proof.get("selectedAction")
                or transition.get("status") != "satisfied"
                or proof.get("completionStatus") != ("exact_complete" if request.mode == "exact" else "bounded_complete")):
            raise ValueError("escalation quality action binding mismatch")
        if request.visibility == "complete_state":
            allocation = artifact["allocationModel"]
            if (allocation.get("kind") != "native_complete_state_quality_v1"
                    or allocation.get("sha256") != canonical_sha256(
                        {k: v for k, v in allocation.items() if k != "sha256"})
                    or allocation.get("sourcePrefixHash") != current.sha256
                    or allocation.get("rulesHash") != rules_hash
                    or allocation.get("exactStateId") != artifact.get("exactStateId")
                    or allocation.get("opportunityOrder") != value["opportunityOrder"]
                    or allocation.get("priorNonAnnouncingSeats") != transition["before"]["opportunityState"]["nonAnnouncingSeats"]):
                raise ValueError("escalation quality complete-state context mismatch")
        events.append({k: event[k] for k in ("actor", "actionCode")})
        if event.get("visibility") == "private":
            from .ortools_nonplay_graphs import EscalationGraphActionV1, _escalation_window_key
            history = tuple(EscalationGraphActionV1.build(item["actor"], item["action"], item["requestedMode"])
                for item in transition["before"]["observation"]["actions"])
            recall.setdefault(event["actor"], []).append(_escalation_window_key(history))
        current = ExactTracePrefixV1.build(artifact["nextPublicPrefix"])
    if value.get("privateRecall", {}) != recall:
        raise ValueError("escalation quality private recall differs from retained transitions")
    if events != value["events"] or current.to_dict() != value["finalPrefix"]:
        raise ValueError("escalation quality final trace mismatch")
    if value["status"] == "complete" and (value.get("projectPlayHandoff") is None
            or not value["finalState"]["observation"]["terminal"]):
        raise ValueError("escalation quality completion mismatch")


def validate_native_round_quality_evidence_v1(
    value: Mapping[str, Any], *, source_prefix: ExactTracePrefixV1,
    rules_hash: str, policy: ScenarioPlayingQualityPolicyV1, decision_point: Mapping[str, Any],
    rules: RuleBook | None = None,
    search_budget_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    maximum_seconds=None, cancel_check=None,
) -> None:
    """Stored and external receipts always require fresh native verification."""
    _validate_native_round_quality_evidence_v1(value, source_prefix=source_prefix,
        rules_hash=rules_hash, policy=policy, decision_point=decision_point, rules=rules,
        search_budget_profiles=search_budget_profiles, maximum_seconds=maximum_seconds,
        cancel_check=cancel_check, verified_artifacts=set())


def _validate_native_round_quality_evidence_v1(
    value: Mapping[str, Any], *, source_prefix: ExactTracePrefixV1,
    rules_hash: str, policy: ScenarioPlayingQualityPolicyV1, decision_point: Mapping[str, Any],
    rules: RuleBook | None = None,
    search_budget_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    maximum_seconds=None, cancel_check=None, verified_artifacts=frozenset(),
) -> None:
    """Validate the composed receipt before storage; does not grant admission."""
    execution_budget = (ScenarioQualityExecutionBudgetV1.build(value["qualityExecution"])
        if isinstance(value.get("qualityExecution"), Mapping) else None)
    if maximum_seconds is None and execution_budget is not None:
        maximum_seconds = execution_budget.maximum_wall_seconds
    remaining = _quality_readback_clock(maximum_seconds, cancel_check)
    remaining()
    if rules is not None and (not isinstance(rules, RuleBook) or rules.sha256 != rules_hash):
        raise ValueError("round quality frozen rules mismatch")
    def envelope(document, kind, prefix):
        if (document.get("kind") != kind
                or document.get("sha256") != canonical_sha256({k: v for k, v in document.items()
                    if k not in {"sha256", "executionReceipt"}})
                or document.get("sourcePrefixHash") != prefix.sha256
                or document.get("rulesHash") != rules_hash
                or document.get("status") not in {"complete", "incomplete", "quality_failed", "proof_failed", "model_error"}
                or document.get("operation") not in {"generate", "audit_recorded", "audit_then_generate"}
                or document.get("playingQuality") != policy.to_dict()):
            raise ValueError("round quality stage ownership mismatch")
    envelope(value, "native_quality_round_continuation_v1", source_prefix)
    mixed = value["operation"] == "audit_then_generate"
    submitted = ExactTracePrefixV1.build(value["submittedPrefix"]) if mixed else None
    if mixed and value.get("recordedPrefixHash") != submitted.sha256:
        raise ValueError("round quality submitted prefix mismatch")
    if mixed:
        action_fields = {"auctionActions", "escalationActions", "projectActions", "playActions"}
        ignored = {"sha256", "stateAssertions", "projectAssertions", *action_fields}
        original, start = submitted.to_dict(), source_prefix.to_dict()
        if (any(original.get(key) != start.get(key) for key in original.keys() | start.keys() if key not in ignored)
                or any(original.get(key, [])[:len(start.get(key, []))] != start.get(key, []) for key in action_fields)):
            raise ValueError("round quality submitted source mismatch")
    auction = value["auction"]
    auction_prefix_only = mixed and auction.get("operation") == "audit_then_generate"
    escalation_prefix_only = bool(mixed and not auction_prefix_only
        and not (submitted.project_actions or submitted.play_actions))
    if auction_prefix_only and (submitted.escalation_actions or submitted.project_actions or submitted.play_actions):
        raise ValueError("round quality auction prefix contains later actions")
    if escalation_prefix_only and (submitted.state_assertions or submitted.project_assertions):
        raise ValueError("partial escalation boundary assertions are not yet supported")
    early_operation = "audit_then_generate" if auction_prefix_only else "audit_recorded" if mixed else value["operation"]
    if auction_prefix_only and (auction.get("submittedPrefix") != submitted.to_dict()
            or auction.get("recordedPrefixHash") != submitted.sha256):
        raise ValueError("round quality submitted auction mismatch")
    if value.get("scope") != "all_decisions" or auction.get("operation") != early_operation:
        raise ValueError("round quality operation mismatch")
    _validate_native_decision_quality_evidence_v1(auction, source_prefix=source_prefix,
        rules_hash=rules_hash, request=policy.request_for("bid", "north"), policy=policy, rules=rules,
        search_budget_profiles=search_budget_profiles,
        execution_budget=execution_budget, cancel_check=cancel_check, verified_artifacts=verified_artifacts,
        maximum_seconds=remaining() if maximum_seconds is not None else None)
    last = auction
    post = value["postAuction"]
    if post is not None:
        if auction["status"] not in {"complete", "quality_failed"} or auction["auctionOutcome"]["allPassed"]:
            raise ValueError("round quality invalid stage continuation")
        prefix = ExactTracePrefixV1.build(auction["finalPrefix"])
        envelope(post, "native_quality_post_auction_continuation_v1", prefix)
        if mixed and not auction_prefix_only and (prefix.auction_actions != submitted.auction_actions or
                post.get("submittedPrefix") != submitted.to_dict() or
                post.get("recordedPrefixHash") != submitted.sha256):
            raise ValueError("round quality submitted stage mismatch")
        post_operation = "generate" if auction_prefix_only else value["operation"]
        if post.get("scope") != "post_auction_only" or post.get("operation") != post_operation:
            raise ValueError("round quality operation mismatch")
        escalation = post["escalation"]
        if escalation.get("operation") != (
                "generate"
                if auction_prefix_only
                else "audit_then_generate"
                if mixed
                else early_operation
        ):
            raise ValueError("round quality operation mismatch")
        actor = (
            decision_point.get("selector", {}).get("actingSeat") or "north"
            if decision_point.get("kind") == "escalation"
            else "north"
        )
        validate_native_escalation_quality_evidence_v1(escalation, source_prefix=prefix,
            rules_hash=rules_hash, request=policy.request_for("escalation", actor),
            rules=rules, search_budget_profiles=search_budget_profiles,
            maximum_seconds=remaining() if maximum_seconds is not None else None, cancel_check=cancel_check)
        if escalation_prefix_only and escalation["status"] == "complete":
            count = len(escalation["recordedEvents"])
            boundary = (escalation["artifacts"][escalation["decisions"][count - 1]["artifactHash"]]["nextPublicPrefix"]
                if count else prefix.to_dict())
            ignored = {"sha256", "stateAssertions", "projectAssertions"}
            if ({key: item for key, item in boundary.items() if key not in ignored}
                    != {key: item for key, item in submitted.to_dict().items() if key not in ignored}):
                raise ValueError("round quality recorded escalation boundary mismatch")
        last = escalation
        play = post["projectPlay"]
        if play is not None:
            if mixed and not (auction_prefix_only or escalation_prefix_only) and (play.get("submittedPrefix") != submitted.to_dict() or
                    play.get("recordedPrefixHash") != submitted.sha256):
                raise ValueError("round quality submitted play mismatch")
            if play.get("operation") != ("generate" if escalation_prefix_only else post_operation):
                raise ValueError("round quality operation mismatch")
            if escalation["status"] != "complete":
                raise ValueError("round quality invalid play continuation")
            _validate_native_decision_quality_evidence_v1(play,
                source_prefix=ExactTracePrefixV1.build(escalation["finalPrefix"]),
                rules_hash=rules_hash, request=policy.request_for("play", "north"), policy=policy, rules=rules,
                search_budget_profiles=search_budget_profiles, private_recall=escalation.get("privateRecall"),
                execution_budget=execution_budget, escalation_events=escalation.get("events"),
                opportunity_order=escalation.get("opportunityOrder"), cancel_check=cancel_check,
                verified_artifacts=verified_artifacts,
                maximum_seconds=remaining() if maximum_seconds is not None else None)
            last = play
        if post["finalPrefix"] != last["finalPrefix"] or (
                post["status"] != last["status"] and post["status"] != "incomplete"):
            raise ValueError("round quality post-auction binding mismatch")
        last = post
    if value["finalPrefix"] != last["finalPrefix"]:
        raise ValueError("round quality final trace mismatch")
    expected_status = last["status"]
    if expected_status == "complete" and auction["status"] == "quality_failed":
        expected_status = "quality_failed"
    if value["status"] != expected_status and not (
            value["status"] == "incomplete" and value.get("partialReceipt") is not None):
        raise ValueError("round quality status binding mismatch")
    if value["status"] == "complete" and (last["status"] != "complete"
            or auction["status"] != "complete"
            or (post is None and not auction["auctionOutcome"]["allPassed"])
            or (post is not None and (post["projectPlay"] is None
                or len(ExactTracePrefixV1.build(value["finalPrefix"]).play_actions) != 32))):
        raise ValueError("round quality false completion")


def _quality_stage_verification_token_v1(core_hash, *, rules, source_prefix, request, policy,
        execution_budget, search_budget_profiles, private_recall, escalation_events, opportunity_order):
    """Reuse a completed readback only inside its owning execution and full context."""
    from .ortools_nonplay_graphs import normalize_escalation_recall_v1
    return canonical_sha256({"kind": "quality_stage_immediate_verification_v1",
        "receiptHash": core_hash, "rulesHash": rules.sha256, "rules": rules.data,
        "structuralDomain": rules.structural_domain.to_dict(),
        "playDomain": rules.play_domain.to_dict(), "projectDomain": rules.project_domain.to_dict(),
        "sourcePrefix": source_prefix.to_dict(), "request": request.to_dict(),
        "policy": policy.to_dict() if policy is not None else None,
        "executionBudget": execution_budget.to_dict() if execution_budget is not None else None,
        "searchProfiles": search_budget_profiles, "privateRecall": normalize_escalation_recall_v1(private_recall),
        "escalationEvents": escalation_events, "opportunityOrder": opportunity_order})


def _exact_observed_verification_token_v1(artifact, *, rules, source_prefix, normalized, request,
        maximum_nodes, maximum_depth, maximum_worlds, declared_maximum_seconds,
        private_recall=None, policy_search=None):
    """Bind immediate reuse to the complete freshly reverified artifact and context."""
    from .ortools_nonplay_graphs import normalize_escalation_recall_v1
    return canonical_sha256({"kind": "exact_observed_immediate_verification_v1",
        "artifactHash": canonical_sha256(artifact), "rulesHash": rules.sha256,
        "sourcePrefixHash": source_prefix.sha256, "normalizedHash": canonical_sha256(normalized.to_dict()),
        "requestHash": request.sha256, "maximumNodes": maximum_nodes, "maximumDepth": maximum_depth,
        "maximumWorlds": maximum_worlds, "declaredSeconds": declared_maximum_seconds,
        "privateRecall": normalize_escalation_recall_v1(private_recall), "policySearch": policy_search})


def reverify_native_exact_observed_play_artifact_v1(artifact, *, rules, source_prefix,
        normalized, request, maximum_nodes, maximum_depth, maximum_worlds,
        maximum_seconds, declared_maximum_seconds, private_recall=None, cancel_check=None, graph_workers=1,
        worker_cpu_callback=None, policy_search=None):
    """Reconstruct native reachability and strategy; hashes alone do not prove either."""
    import time
    from .ortools_exact_state import OrToolsExactStateNormalizerV1, OrToolsNormalizedExactStateV1, _effective_prefix_hash
    from .ortools_information_set_compiler import InformationSetCompilationStopped, prove_actor_information_set_v1
    from .ortools_research_executor import compile_observed_decision_v1
    if (not isinstance(artifact, Mapping) or not isinstance(rules, RuleBook)
            or not isinstance(source_prefix, ExactTracePrefixV1)
            or not isinstance(normalized, OrToolsNormalizedExactStateV1)
            or not isinstance(request, ScenarioProofRequestV1)
            or request.mode != "exact" or request.visibility != "actor_information_set"
            or request.horizon is not None or normalized.decision_kind != "play"
            or any(type(limit) is not int or limit < 1 for limit in (maximum_nodes, maximum_depth, maximum_worlds))
            or any(type(seconds) not in {int, float} or not math.isfinite(seconds) or seconds <= 0
                for seconds in (maximum_seconds, declared_maximum_seconds))
            or maximum_seconds > declared_maximum_seconds):
        raise ValueError("exact observed readback requires owned context and finite declared limits")
    normalized.require_integrity()
    if (not all(isinstance(artifact.get(key), Mapping) for key in ("proofModel", "allocationModel", "proof", "graph"))
            or normalized.rules_hash != rules.sha256 or normalized.prefix_hash != _effective_prefix_hash(source_prefix)
            or artifact.get("request") != request.to_dict()
            or artifact.get("proofModel", {}).get("kind") != "actor_information_set_cegis_model_v1"
            or artifact.get("allocationModel", {}).get("reachability") != "deal_auction_escalation_project_play_cp_sat_v1"
            or artifact.get("proof", {}).get("completionStatus") != "exact_complete"):
        raise ValueError("exact observed artifact source or native coverage mismatch")
    deadline = time.monotonic() + maximum_seconds

    def remaining():
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage="exact_observed_readback")
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise InformationSetCompilationStopped("maximum_seconds", stage="exact_observed_readback")
        return seconds

    rebuilt = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        source_prefix, decision_point={"kind": "play"}, maximum_seconds=remaining(), cancel_check=cancel_check)
    if rebuilt.to_dict() != normalized.to_dict():
        raise ValueError("exact observed state differs from fresh native normalization")
    compilation = compile_observed_decision_v1(rules, rebuilt, source_prefix, {"kind": "play"}, request,
        maximum_nodes=maximum_nodes, maximum_depth=maximum_depth, maximum_worlds=maximum_worlds,
        maximum_seconds=remaining(), declared_maximum_seconds=declared_maximum_seconds,
        native_cp_sat=True, private_recall=private_recall, cancel_check=cancel_check, graph_workers=graph_workers,
        worker_cpu_callback=worker_cpu_callback)
    if (canonical_sha256(compilation.graph.to_dict()) != canonical_sha256(artifact.get("graph"))
            or canonical_sha256(compilation.graph.model_document) != canonical_sha256(artifact["allocationModel"])):
        raise ValueError("exact observed graph differs from fresh native reachability")
    root_witnesses = []
    proof = prove_actor_information_set_v1(compilation, request, maximum_nodes=maximum_nodes,
        maximum_worlds=maximum_worlds, maximum_seconds=remaining(),
        declared_maximum_seconds=declared_maximum_seconds, native_action_quality=True, cancel_check=cancel_check,
        policy_search=policy_search,
        **({"_root_witnesses": root_witnesses} if "rootWitnesses" in artifact else {}))
    remaining()
    _require_matching_exact_observed_reproof_v1(artifact, proof.to_dict(), proof.model_document(),
        fresh_root_witnesses=root_witnesses)


def _require_matching_exact_observed_reproof_v1(artifact, fresh_proof, fresh_model, *, fresh_root_witnesses=None):
    """Incomplete rechecks are not semantic contradictions; complete mismatches fail closed."""
    from .ortools_information_set_compiler import InformationSetCompilationStopped
    completion = fresh_proof.get("completionStatus")
    if completion != "exact_complete":
        raise InformationSetCompilationStopped("proof_incomplete", stage="exact_observed_readback",
            evidence={"completionStatus": completion})
    changed = []
    for section, actual in (("proof", fresh_proof), ("proofModel", fresh_model)):
        expected = artifact[section]
        if canonical_sha256(actual) != canonical_sha256(expected):
            changed.extend(f"{section}.{key}" for key in sorted(set(actual) | set(expected))
                           if canonical_sha256(actual.get(key)) != canonical_sha256(expected.get(key))
                           or (key in actual) != (key in expected))
    if "rootWitnesses" in artifact and (not fresh_root_witnesses
            or canonical_sha256(artifact["rootWitnesses"]) != canonical_sha256(fresh_root_witnesses)):
        changed.append("rootWitnesses")
    if changed:
        # Field names only: diagnostics must not disclose hidden cards or strategy values.
        raise ValueError("exact observed strategy differs from fresh native proof: " + ", ".join(changed))


def native_featured_play_proof_supported_v1(artifact: Mapping[str, Any]) -> bool:
    """Check featured-proof scope after owned full-round evidence validation."""
    if not isinstance(artifact, Mapping):
        return False
    request, model, proof = (artifact.get(key) for key in ("request", "proofModel", "proof"))
    if not all(isinstance(item, Mapping) for item in (request, model, proof)):
        return False
    roots = proof.get("rootActions")
    if not isinstance(roots, list) or len(roots) < 2:
        return False
    if request.get("visibility") == "complete_state":
        return (request.get("mode") == "exact"
            and model.get("kind") == "native_complete_state_root_outcome_v1"
            and proof.get("completionStatus") == "exact_complete"
            and proof.get("hiddenWorlds") is None
            and model.get("rootActionCountScope") == "exact_fixed_deal_terminal_tail")
    worlds = proof.get("hiddenWorlds")
    if request.get("visibility") == "actor_information_set" and request.get("mode") == "exact":
        allocation = artifact.get("allocationModel")
        return (request.get("horizon") is None
            and model.get("kind") == "actor_information_set_cegis_model_v1"
            and proof.get("completionStatus") == "exact_complete"
            and isinstance(worlds, Mapping) and worlds.get("complete") is True
            and type(worlds.get("count")) is int and worlds["count"] > 0
            and isinstance(worlds.get("coverage"), Mapping)
            and worlds["coverage"].get("method") == "exhaustive_cp_sat_hidden_allocations_v1"
            and isinstance(allocation, Mapping)
            and allocation.get("reachability") == "deal_auction_escalation_project_play_cp_sat_v1")
    return (request.get("visibility") == "actor_information_set" and request.get("mode") == "bounded"
        and model.get("kind") == "native_observed_root_outcome_v1"
        and proof.get("completionStatus") == "bounded_complete"
        and isinstance(worlds, Mapping) and worlds.get("method") == "supplied_worlds_only"
        and worlds.get("complete") is False and type(worlds.get("count")) is int and worlds["count"] > 0)


def validate_trace_quality_evidence_v1(
    trace: Mapping[str, Any], *, rules_hash: str,
    required_policy: ScenarioPlayingQualityPolicyV1 | None = None,
    required_proof_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    required_search_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    required_execution: ScenarioQualityExecutionBudgetV1 | None = None,
    rules: RuleBook | None = None,
    maximum_seconds=None, cancel_check=None, require_family_opening=False,
) -> None:
    """Stored and external traces never receive builder-owned proof reuse."""
    _validate_trace_quality_evidence_v1(trace, rules_hash=rules_hash,
        required_policy=required_policy, required_proof_profiles=required_proof_profiles,
        required_search_profiles=required_search_profiles, required_execution=required_execution,
        rules=rules, maximum_seconds=maximum_seconds, cancel_check=cancel_check, verified_artifacts=set(),
        require_family_opening=require_family_opening)


def _validate_trace_quality_evidence_v1(
    trace: Mapping[str, Any], *, rules_hash: str,
    required_policy: ScenarioPlayingQualityPolicyV1 | None = None,
    required_proof_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    required_search_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    required_execution: ScenarioQualityExecutionBudgetV1 | None = None,
    rules: RuleBook | None = None,
    maximum_seconds=None, cancel_check=None, verified_artifacts=frozenset(), require_family_opening=False,
) -> None:
    """Validate optional quality evidence before the existing trace hash binds it."""
    remaining = _quality_readback_clock(maximum_seconds, cancel_check)
    remaining()
    if rules is not None and (not isinstance(rules, RuleBook) or rules.sha256 != rules_hash):
        raise ValueError("trace quality frozen rules mismatch")
    if (require_family_opening and required_execution is not None
            and required_execution.family_opening_mode is not None and 'familyOpening' not in trace):
        raise ValueError('native family execution requires allocation and auction provenance')
    if 'familyOpening' in trace:
        from .scenario_contracts import build_family_opening_evidence_v1
        prefix = trace.get('completeExactTracePrefix', trace.get('exactTracePrefix'))
        if prefix is None:
            raise ValueError('native family opening requires its bound exact trace prefix')
        build_family_opening_evidence_v1(trace['familyOpening'], prefix=ExactTracePrefixV1.build(prefix),
            rules_hash=rules_hash, source_seed=trace['familyOpening']['seed'])
    if "playingQualityEvidence" not in trace:
        if required_policy is not None:
            raise ValueError("qualified trace requires quality evidence")
        return
    value = trace["playingQualityEvidence"]
    if not isinstance(value, Mapping) or set(value) != {"sourcePrefix", "decisionPoint", "receipt"}:
        raise ValueError("trace quality evidence shape mismatch")
    receipt = value["receipt"]
    source = ExactTracePrefixV1.build(value["sourcePrefix"])
    policy = ScenarioPlayingQualityPolicyV1.build(receipt["playingQuality"])
    if required_policy is not None:
        if not isinstance(required_policy, ScenarioPlayingQualityPolicyV1):
            raise TypeError("required quality policy must be typed")
        if policy != required_policy:
            raise ValueError("trace quality policy differs from requested policy")
        if receipt.get("status") != "complete":
            raise ValueError("qualified trace requires complete quality evidence")
    if required_execution is not None:
        if not isinstance(required_execution, ScenarioQualityExecutionBudgetV1):
            raise TypeError("required quality execution budget must be typed")
        if receipt.get("qualityExecution") != required_execution.to_dict():
            raise ValueError("trace quality execution budget differs from frozen budget")
    execution = required_execution
    if execution is None and isinstance(receipt.get("qualityExecution"), Mapping):
        execution = ScenarioQualityExecutionBudgetV1.build(receipt["qualityExecution"])
    if maximum_seconds is None and execution is not None:
        maximum_seconds = execution.maximum_wall_seconds
        remaining = _quality_readback_clock(maximum_seconds, cancel_check)
        remaining()
    decision_point = value["decisionPoint"]
    if not isinstance(decision_point, Mapping):
        raise ValueError("trace quality decision point is invalid")
    if (required_execution is not None and required_execution.featured_play_native_proof
            and decision_point.get("kind") == "play" and "exactTracePrefix" not in trace):
        raise ValueError("featured native play exact prefix is missing")
    if (execution is not None and execution.featured_play_native_proof
            and decision_point.get("kind") == "play"
            and "exactTracePrefix" in trace):
        selector = dict(decision_point.get("selector") or {})
        featured_prefix = ExactTracePrefixV1.build(trace["exactTracePrefix"])
        play = dict(dict(receipt.get("postAuction") or {}).get("projectPlay") or {})
        rows = [row for row in play.get("decisions") or ()
                if row.get("decisionKind", "play") == "play"
                and type(row.get("playIndex")) is int
                and (selector or row["playIndex"] == len(featured_prefix.play_actions))
                and selector.get("trickIndex", row["playIndex"] // 4) == row["playIndex"] // 4
                and selector.get("position", row["playIndex"] % 4) == row["playIndex"] % 4
                and selector.get("actingSeat", row.get("actor")) == row.get("actor")
                and selector.get("remainingCards", 8 - row["playIndex"] // 4)
                == 8 - row["playIndex"] // 4]
        if len(rows) != 1 and len(featured_prefix.play_actions) < 32:
            rows = [row for row in rows
                    if row["playIndex"] == len(featured_prefix.play_actions)]
        if len(rows) != 1:
            raise ValueError("featured native play decision is not covered once")
        artifact = dict(play.get("artifacts") or {}).get(rows[0].get("proofArtifactHash"))
        if not isinstance(artifact, Mapping):
            raise ValueError("featured native play proof artifact is missing")
        root_actions = dict(artifact.get("proof") or {}).get("rootActions") or ()
        kind = dict(artifact.get("proofModel") or {}).get("kind")
        allocation = dict(artifact.get("allocationModel") or {})
        supported = (native_featured_play_proof_supported_v1(artifact) if len(root_actions) > 1
            else kind in {"forced_legal_action_model_v1", "forced_action_optimality_model_v1"})
        if (not supported
                or rows[0].get("actualActionOptimal") is not True
                or (dict(artifact.get("request") or {}).get("visibility") == "complete_state"
                    and (rows[0].get("sourcePrefixHash") != allocation.get("sourcePrefixHash")
                         or rows[0].get("exactStateId") != allocation.get("exactStateId")))):
            raise ValueError("featured play lacks native scope-matched quality proof")
    if required_proof_profiles is not None:
        stages = [(receipt["auction"], set(policy.decisions))]
        post = receipt["postAuction"]
        if post is not None:
            stages.append((post["escalation"], {"escalation"}))
            if post["projectPlay"] is not None:
                stages.append((post["projectPlay"], set(policy.decisions)))
        for stage, kinds in stages:
            profile_ids = {policy.decisions[kind]["budgetProfile"] for kind in kinds}
            if execution is not None and execution.exact_tail_budget_profile is not None and "play" in kinds:
                profile_ids.add(execution.exact_tail_budget_profile)
            if not profile_ids <= required_proof_profiles.keys():
                raise ValueError("required quality proof profile is unavailable")
            expected = {key: required_proof_profiles[key] for key in profile_ids}
            if stage.get("proofBudgetProfiles") != expected:
                raise ValueError("trace quality proof budgets differ from frozen profiles")
            for artifact in stage.get("artifacts", {}).values():
                if artifact.get("proofModel", {}).get("kind") == "native_bid_selected_action_quality_model_v1":
                    profile_id = artifact["request"]["budgetProfile"]
                    if profile_id not in expected:
                        raise ValueError("selected bid quality proof profile is unavailable")
                    if required_search_profiles is None or profile_id not in required_search_profiles:
                        raise ValueError("selected bid quality frozen search profile is unavailable")
                    validate_native_bid_dominance_budget_v1(artifact["graph"],
                        ScenarioProofBudgetV1.build(profile_id, expected[profile_id]),
                        maximum_conflicts=required_search_profiles[profile_id].get("maximumConflicts"),
                        opportunity_order=(required_execution.escalation_opportunity_order
                            if required_execution is not None else None), execution_budget=execution)
    _validate_native_round_quality_evidence_v1(receipt, source_prefix=source,
        rules_hash=rules_hash, policy=policy, decision_point=value["decisionPoint"], rules=rules,
        search_budget_profiles=required_search_profiles, cancel_check=cancel_check,
        verified_artifacts=verified_artifacts,
        maximum_seconds=remaining() if maximum_seconds is not None else None)
    complete = ExactTracePrefixV1.build(trace["completeExactTracePrefix"])
    covered = ExactTracePrefixV1.build(receipt["finalPrefix"])
    if receipt["operation"] == "audit_recorded":
        if receipt.get("recordedPrefixHash") != complete.sha256:
            raise ValueError("trace quality final prefix ownership mismatch")
        action_fields = {"auctionActions", "escalationActions", "projectActions", "playActions"}
        full, partial = complete.to_dict(), covered.to_dict()
        for key in full.keys() | partial.keys():
            if key in {"sha256", "stateAssertions", "projectAssertions"}:
                continue
            if key in action_fields:
                if full.get(key, [])[:len(partial.get(key, []))] != partial.get(key, []):
                    raise ValueError("trace quality covered actions mismatch")
            elif full.get(key) != partial.get(key):
                raise ValueError("trace quality covered context mismatch")
    if (receipt["operation"] != "audit_recorded" or receipt["status"] == "complete") and complete != covered:
        raise ValueError("trace quality final prefix ownership mismatch")


def native_full_round_quality_issue_v1(
    backend_id: str, quality_mode: str | None = None
) -> dict[str, str] | None:
    """Report the missing native all-decision producer, not a false game claim.

    A selected-root proof cannot certify the preceding round. Historical
    validation remains separate so immutable evidence stays readable.
    """
    if backend_id != "ortools_cp_sat_v3" or quality_mode in {"budgeted", "exact"}:
        return None
    return {
        "code": "all_decision_quality_evidence_unavailable",
        "path": "qualification.requireAllDecisionAudit",
        "detail": "selected_decision_proof_only",
    }


_BASE_IDENTITY_FIELDS = frozenset(
    {
        "schemaVersion",
        "kind",
        "runCode",
        "runHash",
        "recipeCode",
        "recipeHash",
        "configurationAuthorityHash",
        "scenarioHash",
        "exactStateHash",
        "scenarioSatisfactionHash",
        "constraintEvidenceHash",
        "proofOutcomeHash",
        "observerSafetyEvidenceHash",
        "scenarioProjectionHash",
        "scoreEvidenceHash",
        "verdictHash",
        "artifactHashes",
        "researchAuthorityHash",
        "backendId",
        "backendVersion",
        "qualificationAuthority",
        "bundleRevisionId",
        "bundleHash",
        "domainAuthorityHash",
        "domainIrHashesHash",
        "loweringManifestHash",
        "constructionModelHash",
        "proofModelHash",
        "graphHash",
        "traceHash",
        "executorSemanticHash",
        "candidateHash",
        "predicateEvaluationHash",
        "exactStateId",
        "qualificationHash",
    }
)
_REQUIRED_IDENTITY_FIELDS = frozenset(
    {
        "schemaVersion",
        "kind",
        "runCode",
        "runHash",
        "recipeCode",
        "recipeHash",
        "configurationAuthorityHash",
        "scenarioHash",
        "exactStateHash",
        "proofOutcomeHash",
        "artifactHashes",
        "researchAuthorityHash",
        "backendId",
        "backendVersion",
        "qualificationAuthority",
        "bundleRevisionId",
        "bundleHash",
        "domainAuthorityHash",
        "domainIrHashesHash",
        "loweringManifestHash",
        "constructionModelHash",
        "proofModelHash",
        "graphHash",
        "traceHash",
        "executorSemanticHash",
        "candidateHash",
        "predicateEvaluationHash",
        "exactStateId",
        "scenarioProjectionHash",
        "qualificationHash",
        "verdictHash",
    }
)
_SEMANTIC_ARTIFACT_FIELDS = {
    "constructionModelHash": "constructionModel",
    "proofModelHash": "proofModel",
    "graphHash": "proofGraph",
    "traceHash": "completeTrace",
    "proofOutcomeHash": "proofOutcome",
}
_SEMANTIC_ARTIFACT_NAMES = frozenset(_SEMANTIC_ARTIFACT_FIELDS.values())
ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES = {
    "constructionModel": "application/vnd.balootx.ortools-construction-model-v1+json",
    "proofModel": "application/vnd.balootx.ortools-proof-model-v1+json",
    "proofGraph": "application/vnd.balootx.ortools-proof-graph-v1+json",
    "completeTrace": "application/vnd.balootx.ortools-complete-trace-v1+json",
    "proofOutcome": "application/vnd.balootx.ortools-proof-outcome-v1+json",
}
_CAS_REFERENCE_FIELDS = frozenset({"uri", "sha256", "bytes", "mediaType", "name"})
_OWNED_REFERENCE_FIELDS = _CAS_REFERENCE_FIELDS | frozenset({"ownerKind", "ownerHash"})
_SEMANTIC_ARTIFACT_OWNER_KIND = "ortools_research_semantic_artifact_v1"
_AUTHORITY_ARTIFACT_OWNER_KIND = "ortools_research_authority_artifact_v1"
ORTOOLS_AUTHORITY_ARTIFACT_MEDIA_TYPES = {
    "loweringManifest": "application/vnd.balootx.ortools-lowering-manifest-v2+json",
    "configurationSnapshot": "application/vnd.balootx.configuration-snapshot-v1+json",
    "resultIdentity": "application/vnd.balootx.ortools-result-identity-v3+json",
    "executionReceipt": "application/vnd.balootx.ortools-execution-receipt-v1+json",
    "interestAdmission": "application/vnd.balootx.round-interest-evidence-v1+json",
}
_AUTHORITY_ARTIFACT_NAMES = frozenset(ORTOOLS_AUTHORITY_ARTIFACT_MEDIA_TYPES) - {"interestAdmission"}
_V3_COMPILATION_RECEIPT_KIND = "ortools_research_compilation_receipt_v3"
_V3_COMPILATION_RECEIPT_MINIMUM_BACKEND_VERSION = (3, 1, 0)
_V3_TACTICAL_PROFILE_MINIMUM_BACKEND_VERSION = (3, 1, 124)
_V3_PARTNER_PROFILE_MINIMUM_BACKEND_VERSION = (3, 1, 125)
_V3_ENTRY_PROFILE_MINIMUM_BACKEND_VERSION = (3, 1, 126)
_V3_INTEREST_ARTIFACT_MINIMUM_BACKEND_VERSION = (3, 1, 55)
_V3_BASELINE_OUTCOME_INTEREST_MINIMUM_BACKEND_VERSION = (3, 1, 141)
_V3_REQUIRED_DOMAIN_IR_NAMES = frozenset(
    {"auction", "escalation", "play", "projects", "scoring", "structural"}
)
_V3_OPTIONAL_DOMAIN_IR_NAMES = frozenset({"ruleSources"})
_CP_SAT_SOLVER_STATUSES = frozenset(
    {"optimal", "feasible", "infeasible", "unknown", "model_invalid"}
)
_V3_EXECUTION_STATUSES_BY_TERMINAL = {
    "satisfied": frozenset({"qualified"}),
    "structurally_infeasible": frozenset({"infeasible"}),
    "not_found_within_budget": frozenset(
        {"not_found", "infeasible", "qualified", "incomplete"}
    ),
    "semantic_postcondition_miss": frozenset(
        {"feasible", "semantic_postcondition_miss"}
    ),
    "proof_incomplete": frozenset(
        {
            "incomplete",
            "compiler_disagreement",
            "feasible",
            "semantic_postcondition_miss",
            "qualified",
        }
    ),
    "proof_failed": frozenset({"proof_failed"}),
}
_V3_STAGE_RECEIPT_V2_FIELDS = frozenset(
    {
        "schemaVersion",
        "stage",
        "modelHash",
        "evidenceHash",
        "semanticHash",
        "solverStatus",
        "variableCount",
        "constraintCount",
        "namedAssumptions",
        "infeasibleCore",
        "evidenceBindings",
        "elapsedSeconds",
        "branches",
        "conflicts",
        "sha256",
    }
)
_V3_NAMED_ASSUMPTION_FIELDS = frozenset(
    {"assumptionId", "literalName", "literalIndex", "predicateIds"}
)
_V3_EVIDENCE_BINDING_FIELDS = frozenset(
    {
        "predicateId",
        "sourceEvidenceHash",
        "loweringHash",
        "modelHash",
        "literalName",
        "literalIndex",
        "evidenceHash",
    }
)
_V3_COMPILATION_AUTHORITY_FIELDS = frozenset(
    {"bundleRevisionId", "bundleHash", "domainIrHashes", "loweringManifestHash"}
)
_V3_COMPILATION_ARTIFACT_HASH_FIELDS = frozenset(
    {"proofModel", "proofGraph", "completeTrace", "proofOutcome"}
)
_V3_CANDIDATE_IDENTITY_FIELDS = frozenset(
    {"schemaVersion", "origin", "noGoodVector", "exactStateId", "stateHash"}
)
_V3_COMPILATION_RECEIPT_FIELDS = frozenset(
    {
        "schemaVersion",
        "kind",
        "backendId",
        "backendVersion",
        "status",
        "scenarioHash",
        "preflightHash",
        "compiledStages",
        "executedStages",
        "configurationAuthority",
        "stageReceipts",
        "candidateIdentity",
        "predicateEvaluation",
        "semanticArtifactHashes",
        "sha256",
    }
)
_TERMINAL_STATUSES = frozenset(
    {
        "satisfied",
        "structurally_infeasible",
        "not_found_within_budget",
        "semantic_postcondition_miss",
        "proof_incomplete",
        "proof_failed",
    }
)


@lru_cache(maxsize=1)
def _predicate_categories_v2() -> dict[str, str]:
    return {
        definition.kind: definition.category
        for definition in scenario_predicate_definitions_v2()
    }


def _scenario_categories_v2(recipe: Mapping[str, Any]) -> list[str]:
    scenario = dict(recipe.get("scenario") or {})
    kinds: set[str] = set()

    def collect(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        if type(value.get("predicateId")) is str and type(value.get("kind")) is str:
            kinds.add(str(value["kind"]))
        for child in value.get("children") or ():
            collect(child)

    collect(scenario.get("constraints"))
    collect(scenario.get("outcome"))
    categories = {
        category
        for kind in kinds
        if (category := _predicate_categories_v2().get(kind)) is not None
    }
    decision_category = {
        "bid": "auction",
        "escalation": "escalation",
        "project": "projects",
        "play": "playPrefix",
    }.get(str(dict(scenario.get("decisionPoint") or {}).get("kind") or ""))
    if decision_category is not None:
        categories.add(decision_category)
    if isinstance(recipe.get("proof"), Mapping):
        categories.add("proof")
    return sorted(categories)


def _require_hash(value: Any, label: str) -> str:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _require_text(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty canonical string")
    return value


def _closed_document(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    document = copy.deepcopy(dict(value))
    unknown = sorted(set(document) - required)
    missing = sorted(required - set(document))
    if unknown or missing:
        raise ValueError(
            f"{label} shape mismatch; missing={missing}, unknown={unknown}"
        )
    return document


def _verify_integrity(document: dict[str, Any], label: str) -> dict[str, Any]:
    known = _require_hash(document.pop("sha256", None), f"{label} sha256")
    if canonical_sha256(document) != known:
        raise ValueError(f"{label} integrity mismatch")
    return {**document, "sha256": known}


def _backend_version_tuple(value: Any) -> tuple[int, int, int] | None:
    if type(value) is not str:
        return None
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        return None
    return tuple(int(item) for item in match.groups())


def _requires_v3_compilation_receipt(authority: Mapping[str, Any]) -> bool:
    version = _backend_version_tuple(authority.get("backendVersion"))
    return (
        authority.get("backendId") == "ortools_cp_sat_v3"
        and version is not None
        and version >= _V3_COMPILATION_RECEIPT_MINIMUM_BACKEND_VERSION
    )


def _non_negative_integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite_non_negative_number(value: Any, label: str) -> float | int:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return value


def _canonical_text_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    items = [_require_text(item, label) for item in value]
    if len(items) != len(set(items)):
        raise ValueError(f"{label} must not contain duplicates")
    return items


def validate_ortools_stage_receipt_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one closed V3.1 stage receipt without changing frozen V1 reads."""

    document = _verify_integrity(
        _closed_document(
            value,
            required=_V3_STAGE_RECEIPT_V2_FIELDS,
            label="OR-Tools V3 stage receipt",
        ),
        "OR-Tools V3 stage receipt",
    )
    if document["schemaVersion"] != 2:
        raise ValueError("OR-Tools V3 stage receipt schemaVersion must be 2")
    _require_text(document["stage"], "OR-Tools V3 stage receipt stage")
    for field in ("modelHash", "evidenceHash", "semanticHash"):
        _require_hash(document[field], f"OR-Tools V3 stage receipt {field}")
    if document["solverStatus"] not in _CP_SAT_SOLVER_STATUSES:
        raise ValueError("OR-Tools V3 stage receipt solverStatus is unsupported")
    for field in ("variableCount", "constraintCount", "branches", "conflicts"):
        _non_negative_integer(
            document[field], f"OR-Tools V3 stage receipt {field}"
        )
    _finite_non_negative_number(
        document["elapsedSeconds"], "OR-Tools V3 stage receipt elapsedSeconds"
    )

    assumptions_raw = document["namedAssumptions"]
    if not isinstance(assumptions_raw, list):
        raise TypeError("OR-Tools V3 stage receipt namedAssumptions must be an array")
    assumption_ids: set[str] = set()
    assumption_indexes: set[int] = set()
    assumptions: list[dict[str, Any]] = []
    for raw in assumptions_raw:
        assumption = _closed_document(
            raw,
            required=_V3_NAMED_ASSUMPTION_FIELDS,
            label="OR-Tools V3 named assumption",
        )
        assumption_id = _require_text(
            assumption["assumptionId"], "OR-Tools V3 assumptionId"
        )
        literal_name = _require_text(
            assumption["literalName"], "OR-Tools V3 assumption literalName"
        )
        literal_index = _non_negative_integer(
            assumption["literalIndex"], "OR-Tools V3 assumption literalIndex"
        )
        predicate_ids = _canonical_text_list(
            assumption["predicateIds"], "OR-Tools V3 assumption predicateIds"
        )
        if assumption_id in assumption_ids or literal_index in assumption_indexes:
            raise ValueError(
                "OR-Tools V3 named assumptions must have unique IDs and literal indexes"
            )
        assumption_ids.add(assumption_id)
        assumption_indexes.add(literal_index)
        assumptions.append(
            {
                "assumptionId": assumption_id,
                "literalName": literal_name,
                "literalIndex": literal_index,
                "predicateIds": predicate_ids,
            }
        )
    document["namedAssumptions"] = assumptions

    core = _canonical_text_list(
        document["infeasibleCore"], "OR-Tools V3 stage receipt infeasibleCore"
    )
    if not set(core) <= assumption_ids:
        raise ValueError(
            "OR-Tools V3 stage receipt infeasibleCore references an unknown assumption"
        )
    if document["solverStatus"] == "infeasible" and assumption_ids and not core:
        raise ValueError(
            "OR-Tools V3 infeasible stage receipt requires its mapped assumption core"
        )
    if document["solverStatus"] != "infeasible" and core:
        raise ValueError(
            "OR-Tools V3 feasible stage receipt cannot carry an infeasible core"
        )
    document["infeasibleCore"] = core

    bindings_raw = document["evidenceBindings"]
    if not isinstance(bindings_raw, list):
        raise TypeError("OR-Tools V3 stage receipt evidenceBindings must be an array")
    binding_keys: set[tuple[str, int]] = set()
    bindings: list[dict[str, Any]] = []
    for raw in bindings_raw:
        binding = _closed_document(
            raw,
            required=_V3_EVIDENCE_BINDING_FIELDS,
            label="OR-Tools V3 evidence binding",
        )
        predicate_id = _require_text(
            binding["predicateId"], "OR-Tools V3 evidence predicateId"
        )
        literal_name = _require_text(
            binding["literalName"], "OR-Tools V3 evidence literalName"
        )
        literal_index = _non_negative_integer(
            binding["literalIndex"], "OR-Tools V3 evidence literalIndex"
        )
        for field in ("sourceEvidenceHash", "loweringHash", "modelHash"):
            _require_hash(binding[field], f"OR-Tools V3 evidence {field}")
        evidence_hash = binding["evidenceHash"]
        if evidence_hash is not None:
            _require_hash(evidence_hash, "OR-Tools V3 evidence evidenceHash")
        if binding["modelHash"] != document["modelHash"]:
            raise ValueError("OR-Tools V3 evidence binding modelHash mismatch")
        if binding["sourceEvidenceHash"] != document["evidenceHash"]:
            raise ValueError("OR-Tools V3 evidence binding source evidence mismatch")
        key = (predicate_id, literal_index)
        if key in binding_keys:
            raise ValueError("OR-Tools V3 evidence binding is duplicated")
        binding_keys.add(key)
        bindings.append(
            {
                "predicateId": predicate_id,
                "sourceEvidenceHash": binding["sourceEvidenceHash"],
                "loweringHash": binding["loweringHash"],
                "modelHash": binding["modelHash"],
                "literalName": literal_name,
                "literalIndex": literal_index,
                "evidenceHash": evidence_hash,
            }
        )
    document["evidenceBindings"] = bindings
    return document


@dataclass(frozen=True, slots=True)
class ResearchAuthorityV1:
    """Canonical CP-SAT authority and semantic artifact ownership."""

    _canonical_json: str

    @classmethod
    def build(
        cls,
        *,
        backend_id: str,
        backend_version: str,
        solver_version: str,
        bundle_revision_id: str,
        bundle_hash: str,
        domain_ir_hashes: Mapping[str, str],
        lowering_manifest_hash: str,
        construction_model_hash: str,
        proof_model_hash: str,
        graph_hash: str,
        trace_hash: str,
        proof_outcome_hash: str,
    ) -> "ResearchAuthorityV1":
        core = {
            "schemaVersion": 1,
            "backendId": backend_id,
            "backendVersion": backend_version,
            "qualificationAuthority": CP_SAT_QUALIFICATION_AUTHORITY,
            "solverVersion": solver_version,
            "bundleRevisionId": bundle_revision_id,
            "bundleHash": bundle_hash,
            "domainIrHashes": dict(sorted(domain_ir_hashes.items())),
            "loweringManifestHash": lowering_manifest_hash,
            "constructionModelHash": construction_model_hash,
            "proofModelHash": proof_model_hash,
            "graphHash": graph_hash,
            "traceHash": trace_hash,
            "proofOutcomeHash": proof_outcome_hash,
        }
        return cls.from_dict({**core, "sha256": canonical_sha256(core)})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResearchAuthorityV1":
        fields = frozenset(
            {
                "schemaVersion",
                "backendId",
                "backendVersion",
                "qualificationAuthority",
                "solverVersion",
                "bundleRevisionId",
                "bundleHash",
                "domainIrHashes",
                "loweringManifestHash",
                "constructionModelHash",
                "proofModelHash",
                "graphHash",
                "traceHash",
                "proofOutcomeHash",
                "sha256",
            }
        )
        document = _verify_integrity(
            _closed_document(value, required=fields, label="researchAuthority"),
            "researchAuthority",
        )
        if type(document["schemaVersion"]) is not int or document["schemaVersion"] != 1:
            raise ValueError("researchAuthority schemaVersion must be 1")
        if document["backendId"] not in ORTOOLS_RESEARCH_BACKENDS:
            raise ValueError("researchAuthority backendId is unsupported")
        if document["qualificationAuthority"] != CP_SAT_QUALIFICATION_AUTHORITY:
            raise ValueError("researchAuthority qualificationAuthority is unsupported")
        for field in ("backendVersion", "solverVersion", "bundleRevisionId"):
            _require_text(document[field], f"researchAuthority {field}")
        for field in (
            "bundleHash",
            "loweringManifestHash",
            "constructionModelHash",
            "proofModelHash",
            "graphHash",
            "traceHash",
            "proofOutcomeHash",
        ):
            _require_hash(document[field], f"researchAuthority {field}")
        hashes = document["domainIrHashes"]
        if not isinstance(hashes, Mapping) or not hashes:
            raise ValueError(
                "researchAuthority domainIrHashes must be a non-empty object"
            )
        canonical_hashes: dict[str, str] = {}
        for name, digest in hashes.items():
            canonical_name = _require_text(name, "researchAuthority domain IR name")
            canonical_hashes[canonical_name] = _require_hash(
                digest, f"researchAuthority domainIrHashes.{canonical_name}"
            )
        document["domainIrHashes"] = dict(sorted(canonical_hashes.items()))
        return cls(
            json.dumps(
                document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)


def _validate_v3_domain_ir_hashes(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("OR-Tools V3 domainIrHashes must be an object")
    hashes = dict(value)
    names = set(hashes)
    if not _V3_REQUIRED_DOMAIN_IR_NAMES <= names or not names <= (
        _V3_REQUIRED_DOMAIN_IR_NAMES | _V3_OPTIONAL_DOMAIN_IR_NAMES
    ):
        raise ValueError("OR-Tools V3 domainIrHashes shape mismatch")
    return {
        _require_text(name, "OR-Tools V3 domain IR name"): _require_hash(
            digest, f"OR-Tools V3 domainIrHashes.{name}"
        )
        for name, digest in sorted(hashes.items())
    }


def _validate_predicate_evaluation_v1(
    value: Any, *, expected_hash: str | None, scenario: Mapping[str, Any]
) -> dict[str, Any] | None:
    if value is None:
        if expected_hash is not None:
            raise ValueError("OR-Tools V3 predicate evaluation preimage is missing")
        return None
    if not isinstance(value, Mapping):
        raise TypeError("OR-Tools V3 predicate evaluation must be an object")
    document = copy.deepcopy(dict(value))
    known = _require_hash(
        document.pop("sha256", None), "OR-Tools V3 predicate evaluation sha256"
    )
    if canonical_sha256(document) != known or (
        expected_hash is not None and known != expected_hash
    ):
        raise ValueError("OR-Tools V3 predicate evaluation integrity mismatch")
    evidence = document.get("evidence")
    if not isinstance(evidence, list):
        raise TypeError("OR-Tools V3 predicate evaluation evidence must be an array")
    # Empty historical all roots carry no hard claim.
    expressions = tuple(
        ScenarioExpressionV1.build(scenario[section])
        for section in ("constraints", "outcome")
        if not (scenario[section].get("op") == "all" and scenario[section].get("children") == [])
    )
    expected = {
        predicate.predicate_id: predicate
        for expression in expressions for predicate in expression.predicates()
    }
    actual: dict[str, Mapping[str, Any]] = {}
    for item in evidence:
        if not isinstance(item, Mapping):
            raise TypeError("OR-Tools V3 predicate evidence must be an object")
        evidence_document = copy.deepcopy(dict(item))
        evidence_hash = _require_hash(
            evidence_document.pop("sha256", None),
            "OR-Tools V3 predicate evidence sha256",
        )
        if canonical_sha256(evidence_document) != evidence_hash:
            raise ValueError("OR-Tools V3 predicate evidence integrity mismatch")
        predicate_id = item.get("predicateId")
        if not isinstance(predicate_id, str) or predicate_id not in expected or predicate_id in actual:
            raise ValueError("OR-Tools V3 predicate evidence is unowned or duplicated")
        if item.get("kind") != expected[predicate_id].kind or type(item.get("satisfied")) is not bool:
            raise ValueError("OR-Tools V3 predicate evidence kind or truth is invalid")
        actual[predicate_id] = item
    if type(document.get("satisfied")) is not bool or (
        document.get("status") == "satisfied"
    ) != document["satisfied"]:
        raise ValueError("OR-Tools V3 predicate evaluation status disagrees with its truth")
    if document["satisfied"]:
        if actual.keys() != expected.keys():
            raise ValueError("OR-Tools V3 satisfied predicate evidence is incomplete")

        def hard_truth(node):
            if node.predicate is not None:
                return (actual[node.predicate.predicate_id]["satisfied"]
                    if node.predicate.hardness == "hard" else None)
            children = [value for child in node.children if (value := hard_truth(child)) is not None]
            if not children:
                return None
            if node.op == "all":
                return all(children)
            if node.op == "any":
                return any(children)
            if node.op == "not":
                return not children[0]
            return node.minimum <= sum(children) <= node.maximum

        if any(hard_truth(expression) is False for expression in expressions):
            raise ValueError("OR-Tools V3 predicate evidence contradicts the hard expression")
    return {**document, "sha256": known}


def _validate_native_delayed_payoff_profile_v1(detector, profile, proof_outcome, proof_model, proof_request, scale):
    native = detector.get("nativeDelayedPayoff")
    metric = profile.native_delayed_gain_metric
    minimum = profile.native_delayed_minimum_gain
    metric_scale = 1 if metric == "raw_card_points" else scale
    if (metric is None or minimum is None or not isinstance(native, Mapping)
            or set(native) != {"gainMetric", "minimumGain"}
            or type(native["gainMetric"]) is not str or native["gainMetric"] != metric
            or type(native["minimumGain"]) not in (int, float) or native["minimumGain"] != minimum
            or type(detector.get("nativeGainScale")) is not int or detector["nativeGainScale"] != metric_scale
            or type(detector.get("minimumNativeGainScaled")) is not int
            or detector["minimumNativeGainScaled"] != max(1, math.ceil(Fraction(str(minimum)) * metric_scale))
            or detector.get("rawPayoffUnit") != "raw_card_points_excluding_last_trick_bonus"
            or detector.get("continuationObjective") != "awarded_point_margin"
            or detector.get("causalProofStatus") != "paired_native_delayed_payoff"
            or detector.get("qualificationSource") != "native_cp_sat_paired_delayed_payoff_variables"
            or detector.get("pairedRawPayoffScope") != "canonical_complete_state_continuations"
            or detector.get("pairedRawPayoffUnit") != "raw_card_points"
            or any(proof_model.get(key) != expected for key, expected in (
                ("proofMode", "exact"), ("visibility", "complete_state"), ("utility", "awarded_point_margin")))):
        raise ValueError("native tactical profile delayed payoff settings or scope mismatch")
    small = detector["detectorId"] == TacticalKind.SMALL_CARD_BIG_GAIN.value
    field = "lowCardMaxStrength" if small else "highCardMinStrength"
    cutoff = profile.low_card_max_strength if small else profile.high_card_min_strength
    if (type(detector.get(field)) not in (int, float) or detector[field] != cutoff
            or detector.get("strengthNormalization") != "within_suit_0_100"):
        raise ValueError("native tactical profile delayed payoff strength mismatch")
    pairs = proof_model["pairedRawPayoffVariables"]
    winners = detector["pairedWinnerVariables"]
    labels = sorted(key.removesuffix("WinnerSeat") for key in winners[0])
    expected_control = [{key: value for key, value in pair["variables"].items()
        if key.endswith(("WinningCard", "OpponentTrumpsRemaining"))} for pair in pairs]
    if (not labels or any(re.fullmatch(r"trick[1-8]", label) is None for label in labels)
            or labels != [f"trick{index}" for index in range(int(labels[0][5:]), 9)]
            or any(set(row) != {label + "WinnerSeat" for label in labels} for row in winners)
            or any(pair.get("delayedControl") is not True for pair in pairs)
            or detector.get("pairedDelayedControlVariables") != expected_control
            or any(set(row) != {label + suffix for label in labels
                for suffix in ("WinningCard", "OpponentTrumpsRemaining")} for row in expected_control)):
        raise ValueError("native tactical profile paired delayed control ownership mismatch")
    roots = proof_outcome["rootActions"]
    root_variables = proof_model["rootActionVariables"]
    if (any(not isinstance(root, Mapping)
                or type(root.get("action")) is not str or not root["action"]
                or type(root.get("actionId")) is not int
                or (root.get("actionId"), root.get("action")) != (variable.get("actionId"), variable.get("action"))
                or type(root.get("value")) not in (int, float)
                or type(root["value"]) is float and not math.isfinite(root["value"])
                or not isinstance(variable.get("valueVariable"), str) or not variable["valueVariable"]
                for root, variable in zip(roots, root_variables, strict=True))
            or type(detector.get("rootActionCount")) is not int or detector["rootActionCount"] != len(roots)
            or len({root["action"] for root in roots}) != len(roots)
            or detector.get("rootValueVariables") != [root["valueVariable"] for root in root_variables]
            or detector.get("selectedRootAction") != proof_outcome.get("selectedAction")
            or type(proof_model.get("rootActorTeam")) is not str
            or proof_model["rootActorTeam"] not in {"north_south", "west_east"}
            or type(proof_request.get("targetTeam")) is not str
            or proof_request["targetTeam"] not in {"north_south", "west_east"}):
        raise ValueError("native tactical profile delayed comparator root ownership mismatch")
    selected = next((index for index, root in enumerate(roots)
        if root["action"] == detector.get("selectedRootAction")), None)
    if selected is None or detector.get("selectedValueVariable") != root_variables[selected]["valueVariable"]:
        raise ValueError("native tactical profile delayed selected root mismatch")
    direction = 1 if proof_model["rootActorTeam"] == proof_request["targetTeam"] else -1
    comparator = max((index for index in range(len(roots)) if index != selected),
        key=lambda index: direction * roots[index]["value"])
    if type(detector.get("bestAlternativeRootIndex")) is not int or detector["bestAlternativeRootIndex"] != comparator:
        raise ValueError("native tactical profile delayed best comparator mismatch")
    suffix_bounds = {"WinnerSeat": 3, "LeaderSeat": 3, "LeadCard": 31,
        "WinningCard": 31, "OpponentTrumpsRemaining": 8, "CardRawMargin": None}
    for index, pair in enumerate(pairs):
        for label in labels:
            for suffix, maximum in suffix_bounds.items():
                key = label + suffix
                variable = pair["variables"].get(key)
                value = detector.get(f"root{index}{key[0].upper() + key[1:]}")
                if (not isinstance(variable, str) or not variable or type(value) is not int
                        or maximum is not None and not 0 <= value <= maximum):
                    raise ValueError("native tactical profile delayed paired solver value is missing or invalid")
    differences = [detector[f"root{selected}{label.capitalize()}CardRawMargin"]
        - detector[f"root{comparator}{label.capitalize()}CardRawMargin"] for label in labels]
    awarded_gain = round(direction * (Fraction(str(roots[selected]["value"]))
        - Fraction(str(roots[comparator]["value"]))) * scale)
    expected_gain = sum(differences) if metric == "raw_card_points" else awarded_gain
    if (any(type(detector.get(key)) is not int for key in ("nativeGain", "laterCardMarginGain", "firstTrickConcession"))
            or detector["nativeGain"] != expected_gain
            or detector["laterCardMarginGain"] != sum(differences[1:])
            or detector["firstTrickConcession"] != -differences[0]
            or any(type(detector.get(key)) is not bool for key in (
                "applicable", "verified", "causalProven", "entry", "forcedReturn", "partnerEnablement", "promotion", "trumpExhaustion"))):
        raise ValueError("native tactical profile delayed payoff solver values disagree")
    if detector["causalProven"] and (
            not detector["applicable"] or not detector["verified"]
            or not any(detector[key] for key in ("entry", "forcedReturn", "partnerEnablement", "promotion", "trumpExhaustion"))
            or detector["nativeGain"] < detector["minimumNativeGainScaled"]
            or awarded_gain < detector["minimumCausalGainScaled"]
            or detector["laterCardMarginGain"] <= 0
            or not small and (detector["firstTrickConcession"] <= 0
                or detector[f"root{selected}{labels[0].capitalize()}WinnerSeat"] % 2
                == (0 if proof_model["rootActorTeam"] == "north_south" else 1))):
        raise ValueError("native tactical profile delayed causal claim lacks its owned payoff witness")


def _validate_native_tactical_profile_v1(
    predicate, proof_outcome, snapshot, *, bundle_hash, snapshot_hash, proof_graph=None, proof_request=None, proof_model=None,
    backend_version=None, scenario=None,
):
    version = _backend_version_tuple(backend_version)
    current_profile_contract = backend_version is not None and (
        version is None or version >= _V3_TACTICAL_PROFILE_MINIMUM_BACKEND_VERSION)
    if current_profile_contract and not isinstance(scenario, Mapping):
        raise ValueError("native tactical profile scenario ownership is missing")
    expected_detectors = _scenario_tactic_requests(scenario or {}) if current_profile_contract else {}
    for row in (predicate or {}).get("evidence", []):
        observed = row.get("observed") or {}
        # Same-solve evidence wraps the signed native predicate evidence once.
        detector = observed.get("observed", observed) if isinstance(observed, Mapping) else {}
        expected_detector = expected_detectors.get(row.get("predicateId"))
        minimum_version = (_V3_ENTRY_PROFILE_MINIMUM_BACKEND_VERSION if expected_detector == TacticalKind.ENTRY_UNBLOCKING.value
            else _V3_PARTNER_PROFILE_MINIMUM_BACKEND_VERSION if expected_detector == TacticalKind.PARTNER_ENABLEMENT.value
            else _V3_TACTICAL_PROFILE_MINIMUM_BACKEND_VERSION)
        required = current_profile_contract and expected_detector in NATIVE_PROFILE_TACTIC_IDS and (
            version is None or version >= minimum_version)
        if required and (not isinstance(detector, Mapping) or detector.get("detectorId") != expected_detector):
            raise ValueError("native tactical profile detector does not match the scenario")
        if not isinstance(detector, Mapping):
            continue
        delayed = detector.get("detectorVersion") == "3.1.0"
        if delayed and detector.get("detectorId") not in DELAYED_PAYOFF_TACTIC_IDS:
            raise ValueError("native tactical profile delayed detector version mismatch")
        if detector.get("detectorId") not in NATIVE_PROFILE_TACTIC_IDS:
            continue
        roots = proof_outcome.get("rootActions")
        if (not delayed and isinstance(roots, list) and len(roots) == 1
                and detector.get("rootActionCount") == 1
                and detector.get("minimumRequiredRootActions") == 2
                and detector.get("inapplicabilityReason") == "required_alternative_absent"
                and all(detector.get(key) is False for key in (
                    "applicable", "verified", "causalProven"))):
            continue
        if required and detector.get("detectorVersion") != "3.0.1" and not delayed:
            raise ValueError("native tactical profile detector version mismatch")
        if detector.get("detectorVersion") != "3.0.1" and not delayed:
            continue
        if (not isinstance(snapshot, Mapping)
                or snapshot.get("sha256") != snapshot_hash
                or snapshot.get("bundleHash") != bundle_hash
                or canonical_sha256({key: value for key, value in snapshot.items()
                                     if key != "sha256"}) != snapshot_hash):
            raise ValueError("native tactical profile snapshot authority mismatch")
        try:
            raw = snapshot["components"]["challenge"]["payload"]["tactical_detectors"]
        except (KeyError, TypeError) as exc:
            raise ValueError("native tactical profile is missing from the frozen bundle") from exc
        try:
            profile = TacticalDetectorProfileV1.build(raw)
        except (TypeError, ValueError) as exc:
            if not delayed:
                raise
            raise ValueError("native tactical profile delayed frozen settings are invalid") from exc
        if (not delayed and detector["detectorId"] in DELAYED_PAYOFF_TACTIC_IDS
                and profile.native_delayed_gain_metric is not None):
            raise ValueError("native tactical profile opted-in delayed detector must use version 3.1.0")
        if detector["detectorId"] == TacticalKind.PROJECT_BIDDING.value:
            scale = proof_graph.get("utilityScale") if isinstance(proof_graph, Mapping) else None
            if (type(scale) is not int or scale < 1
                    or type(detector.get("projectConsequenceScale")) is not int
                    or detector["projectConsequenceScale"] != scale
                    or type(detector.get("minimumProjectConsequenceScaled")) is not int
                    or detector["minimumProjectConsequenceScaled"] != max(
                        1, math.ceil(Fraction(str(profile.project_consequence_delta)) * scale))
                    or detector.get("projectConsequenceUnit") != "awarded_point_margin"):
                raise ValueError("native tactical profile project consequence units disagree with the proof")
            field, expected = "projectConsequenceDelta", profile.project_consequence_delta
        elif delayed or detector["detectorId"] in {TacticalKind.TRUMP_CONTROL.value, TacticalKind.PARTNER_ENABLEMENT.value, TacticalKind.ENTRY_UNBLOCKING.value}:
            scale = proof_graph.get("utilityScale") if isinstance(proof_graph, Mapping) else None
            if (type(scale) is not int or scale < 1
                    or type(detector.get("causalGainScale")) is not int
                    or detector["causalGainScale"] != scale
                    or type(detector.get("minimumCausalGainScaled")) is not int
                    or detector["minimumCausalGainScaled"] != max(
                        1, math.ceil(Fraction(str(profile.minimum_causal_gain)) * scale))
                    or detector.get("causalGainUnit") != "awarded_point_margin"
                    or not isinstance(proof_graph, Mapping) or proof_graph.get("scope") != "exact"
                    or not isinstance(proof_request, Mapping)
                    or proof_request.get("mode") != "exact"
                    or proof_request.get("visibility") != "complete_state"
                    or proof_request.get("utility") != "awarded_point_margin"):
                raise ValueError("native tactical profile causal-gain units disagree with the proof")
            if delayed or detector["detectorId"] in {TacticalKind.PARTNER_ENABLEMENT.value, TacticalKind.ENTRY_UNBLOCKING.value}:
                pairs = proof_model.get("pairedRawPayoffVariables") if isinstance(proof_model, Mapping) else None
                root_variables = proof_model.get("rootActionVariables") if isinstance(proof_model, Mapping) else None
                if (not isinstance(pairs, list) or not isinstance(root_variables, list)
                        or len(pairs) != len(roots or ()) or len(pairs) != len(root_variables)
                        or len(pairs) < 2 or any(not isinstance(pair, Mapping)
                            or pair.get("scope") != "canonical_complete_state_continuation"
                            or pair.get("unit") != "raw_card_points"
                            or not isinstance(pair.get("variables"), Mapping)
                            or not isinstance(root, Mapping)
                            or (pair.get("actionId"), pair.get("action")) != (root.get("actionId"), root.get("action"))
                            for pair, root in zip(pairs, root_variables, strict=True))
                        or detector.get("pairedWinnerVariables") != [{key: value for key, value in pair["variables"].items()
                            if key.endswith("WinnerSeat")} for pair in pairs]
                        or not all(detector["pairedWinnerVariables"])
                        or detector.get("mechanismKind") != detector["detectorId"]
                        or detector.get("mechanismScope") != "paired_canonical_continuations"
                        or detector.get("comparisonPolicy") != "best_actor_alternative_stable_root_order"):
                    raise ValueError("native tactical profile paired partner winner ownership mismatch")
                if delayed or detector["detectorId"] == TacticalKind.ENTRY_UNBLOCKING.value:
                    expected_entry = [{key: value for key, value in pair["variables"].items()
                        if key.endswith(("LeaderSeat", "LeadCard", "CardRawMargin"))} for pair in pairs]
                    if (detector.get("pairedEntryVariables") != expected_entry
                            or any(set(entry) != {key.removesuffix("WinnerSeat") + suffix
                                for key in winners for suffix in ("LeaderSeat", "LeadCard", "CardRawMargin")}
                                for entry, winners in zip(expected_entry, detector["pairedWinnerVariables"], strict=True))
                            or any(type(pair.get("firstLeader")) is not int or not 0 <= pair["firstLeader"] < 4 for pair in pairs)
                            or len({pair["firstLeader"] for pair in pairs}) != 1
                            or detector.get("payoffPointUnit") != "raw_card_points_excluding_last_trick_bonus"):
                        raise ValueError("native tactical profile paired entry ownership mismatch")
            if delayed:
                _validate_native_delayed_payoff_profile_v1(detector, profile, proof_outcome, proof_model, proof_request, scale)
            field, expected = "minimumCausalGain", profile.minimum_causal_gain
        else:
            small = detector["detectorId"] == TacticalKind.SMALL_CARD_BIG_GAIN.value
            field = "lowCardMaxStrength" if small else "highCardMinStrength"
            expected = profile.low_card_max_strength if small else profile.high_card_min_strength
            if detector.get("strengthNormalization") != "within_suit_0_100":
                raise ValueError("native tactical profile strength normalization differs")
        if (detector.get("detectorProfileHash") != profile.sha256
                or type(detector.get(field)) not in (int, float)
                or detector[field] != expected):
            raise ValueError("native tactical profile evidence differs from the frozen bundle")


def _validate_campaign_semantic_document_v1(
    value: Mapping[str, Any], *, identity: Mapping[str, Any], satisfied: bool
) -> dict[str, Any]:
    document = copy.deepcopy(dict(value))
    semantic_fields = (
        "schemaVersion",
        "status",
        "requestedResultCount",
        "maximumCandidates",
        "symmetryPolicyId",
        "distancePolicyId",
        "minimumDistance",
        "executionSemanticHashes",
        "acceptedSemanticHashes",
        "variantReceipts",
    )
    if any(field not in document for field in semantic_fields):
        raise ValueError("OR-Tools V3 campaign receipt is incomplete")
    no_goods = document.get("noGoodVectors")
    if not isinstance(no_goods, list):
        raise TypeError("OR-Tools V3 campaign noGoodVectors must be an array")
    semantic_core = {
        field: copy.deepcopy(document[field]) for field in semantic_fields
    }
    semantic_core["noGoodHashes"] = [canonical_sha256(item) for item in no_goods]
    if "coordinatorReceipt" in document:
        coordinator = document.get("coordinatorReceipt")
        semantic_core["coordinatorSemanticHash"] = (
            dict(coordinator).get("semanticSha256")
            if isinstance(coordinator, Mapping)
            else None
        )
    semantic_hash = _require_hash(
        document.get("semanticHash"), "OR-Tools V3 campaign semanticHash"
    )
    if canonical_sha256(semantic_core) != semantic_hash:
        raise ValueError("OR-Tools V3 campaign semantic hash mismatch")
    receipt_core = {
        **semantic_core,
        "semanticHash": semantic_hash,
        "wallTimeSeconds": document.get("wallTimeSeconds"),
        "cpuSeconds": document.get("cpuSeconds"),
        **(
            {"coordinatorReceipt": copy.deepcopy(document.get("coordinatorReceipt"))}
            if "coordinatorReceipt" in document
            else {}
        ),
    }
    if canonical_sha256(receipt_core) != _require_hash(
        document.get("sha256"), "OR-Tools V3 campaign sha256"
    ):
        raise ValueError("OR-Tools V3 campaign receipt integrity mismatch")
    executor_hash = identity.get("executorSemanticHash")
    execution_hashes = document.get("executionSemanticHashes")
    accepted_hashes = document.get("acceptedSemanticHashes")
    if (
        semantic_hash != identity.get("campaignSemanticHash")
        or not isinstance(execution_hashes, list)
        or executor_hash not in execution_hashes
        or (
            satisfied
            and (not isinstance(accepted_hashes, list) or executor_hash not in accepted_hashes)
        )
    ):
        raise ValueError("OR-Tools V3 campaign/result semantic binding mismatch")
    return document


def _verify_project_parent_paths(graph, *, rules, source_prefix, frontiers,
                                 maximum_nodes, maximum_seconds):
    """Verify declaration/play edges before the retained full-play graphs."""
    import time
    from .ortools_exact_state import OrToolsExactStateNormalizerV1
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    deadline = time.monotonic() + maximum_seconds
    nodes = {node.node_id: node for node in graph.nodes}
    if graph.root_node_id in frontiers:
        raise ValueError("project parent is missing its declaration root")
    pending = [(graph.root_node_id, source_prefix)]
    seen = set()
    ignored = {"sha256", "stateAssertions", "projectAssertions", "projectDeclarations"}
    def context(prefix):
        return {key: value for key, value in prefix.to_dict().items() if key not in ignored}
    while pending:
        node_id, prefix = pending.pop()
        key = (node_id, prefix.sha256)
        if key in seen:
            continue
        seen.add(key)
        remaining = deadline - time.monotonic()
        if remaining <= 0 or len(seen) > maximum_nodes:
            raise ValueError("project parent verification budget exhausted")
        if node_id in frontiers:
            if not any(context(prefix) == context(candidate) for candidate in frontiers[node_id]):
                actual, expected = context(prefix), context(frontiers[node_id][0])
                changed = sorted(key for key in actual.keys() | expected.keys() if actual.get(key) != expected.get(key))
                raise ValueError("project parent path differs from compilation prefix: " + ", ".join(changed))
            continue
        node = nodes[node_id]
        if node.is_leaf:
            raise ValueError("project parent has an unverified terminal")
        declaration_count = len(rules.project_domain.declaration_seats(Seat.parse(prefix.dealer)))
        kind = ("project" if len(prefix.project_actions) < declaration_count
                and len(prefix.project_actions) <= len(prefix.play_actions) else "play")
        if node_id == graph.root_node_id and kind != "project":
            raise ValueError("project parent root is not a declaration decision")
        normalized = normalizer.normalize(prefix, decision_point={"kind": kind,
            **({"selector": {"owner": node.actor_seat}} if kind == "project" else {})},
            maximum_seconds=remaining)
        if (node.actor_seat != normalized.next_actor
                or node.actor_team != Seat.parse(normalized.next_actor).team.name.lower()
                or {edge.action_code for edge in node.edges} != {row["code"] for row in normalized.legal_actions}):
            raise ValueError("project parent legal actions or actor differ")
        for edge in node.edges:
            pending.append((edge.child_node_id, normalizer.append_normalized_action(prefix, normalized, edge.action_code)))


def validate_ortools_result_artifact_documents_v3(
    payload: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any],
    documents: Mapping[str, Mapping[str, Any]],
    campaign_document: Mapping[str, Any] | None = None,
    configuration_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate V3.1 semantic preimages after exact CAS loading.

    V3.0 artifacts remain readable. New V3.1 executions must provide the closed
    compilation receipt that joins stage evidence to the existing five-artifact
    authority rather than creating another artifact family.
    """

    result = copy.deepcopy(dict(payload))
    authority = ResearchAuthorityV1.from_dict(
        dict(result.get("researchAuthority") or {})
    ).to_dict()
    identity = dict(dict(result.get("provenance") or {}).get("resultIdentity") or {})
    if identity.get("kind") != ORTOOLS_RESULT_IDENTITY_V3_KIND:
        return result
    if set(documents) != _SEMANTIC_ARTIFACT_NAMES:
        raise RuntimeError("stored OR-Tools Research V3 semantic preimages are incomplete")
    semantic_documents = {
        name: copy.deepcopy(dict(document)) for name, document in documents.items()
    }
    for authority_field, artifact_name in _SEMANTIC_ARTIFACT_FIELDS.items():
        if canonical_sha256(semantic_documents[artifact_name]) != authority[authority_field]:
            raise RuntimeError(
                f"stored OR-Tools Research V3 {artifact_name} preimage mismatch"
            )
    trace = semantic_documents["completeTrace"]
    proof_model = semantic_documents["proofModel"]
    proof_outcome = semantic_documents["proofOutcome"]
    requested_tactics, verified_tactics = _native_tactic_projection(
        recipe["scenario"], list((result.get("predicateEvaluation") or {}).get("evidence", [])),
        proof_outcome=proof_outcome,
    )
    if (result.get("tacticalTags") != requested_tactics
            or result.get("verifiedTacticalTags") != verified_tactics):
        raise RuntimeError("stored OR-Tools Research V3 tactical tag projection mismatch")
    _validate_native_tactical_profile_v1(
        result.get("predicateEvaluation"), proof_outcome, configuration_snapshot,
        bundle_hash=authority["bundleHash"],
        snapshot_hash=identity.get("configurationSnapshotHash"),
        proof_graph=semantic_documents["proofGraph"],
        proof_request=recipe.get("proof"),
        proof_model=proof_model,
        backend_version=authority["backendVersion"], scenario=recipe["scenario"],
    )
    if "projectCompilationEvidence" in proof_model and "projectCounterfactualGraph" not in proof_model:
        raise ValueError("stored project compilation lacks its counterfactual graph")
    if ("projectCounterfactualGraph" in proof_model) != ("rootProjectConsequenceVariables" in proof_model):
        raise ValueError("stored project counterfactual proof bindings are incomplete")
    if "projectCounterfactualGraph" in proof_model:
        from .ortools_proof import ProofGraphV1
        graph_document = semantic_documents["proofGraph"]
        graph = ProofGraphV1.from_dict({**graph_document, "sha256": authority["graphHash"]})
        counterfactual = ProofGraphV1.from_dict(proof_model["projectCounterfactualGraph"])
        graph.require_same_legal_graph(counterfactual)
        if (graph.scope != "exact"
                or dict(recipe.get("scenario") or {}).get("decisionPoint", {}).get("kind") != "project"
                or proof_model.get("graphHash") != graph.sha256
                or proof_model.get("proofMode") != "exact"
                or proof_model.get("visibility") != "complete_state"
                or proof_model.get("utility") != "awarded_point_margin"
                or recipe["proof"].get("mode") != "exact"
                or recipe["proof"].get("visibility") != "complete_state"
                or recipe["proof"].get("utility") != "awarded_point_margin"
                or proof_model.get("proofRequestHash") != recipe["proof"]["sha256"]
                or proof_model.get("rootProjectConsequenceVariables") != [
                    f"root_project_consequence_{index}"
                    for index in range(len(next(node for node in graph.nodes if node.node_id == graph.root_node_id).edges))
                ]):
            raise ValueError("stored project counterfactual proof ownership mismatch")
        if "projectCompilationEvidence" in proof_model:
            rows = proof_model["projectCompilationEvidence"]
            if not isinstance(rows, list) or not rows:
                raise ValueError("stored project compilation evidence is empty")
            parent_nodes = {node.node_id: node for node in graph.nodes}
            parent_counter_nodes = {node.node_id: node for node in counterfactual.nodes}
            covered_terminals: set[str] = set()
            parent_frontiers: dict[str, list[ExactTracePrefixV1]] = {}
            for row in rows:
                fields = {"model", "receipt", "seed", "request", "graph", "counterfactualGraph", "exactTracePrefix"}
                if not isinstance(row, Mapping) or set(row) != fields or any(not isinstance(value, Mapping) for value in row.values()):
                    raise ValueError("stored project compilation evidence shape mismatch")
                if "exactTracePrefix" in row:
                    prefix = ExactTracePrefixV1.build(row["exactTracePrefix"])
                    if prefix.to_dict() != row["exactTracePrefix"]:
                        raise ValueError("stored project compilation prefix is noncanonical")
                    origin = dict(dict(recipe.get("scenario") or {}).get("origin") or {})
                    source_raw = (origin.get("exactTracePrefix") if origin.get("kind") == "exact_trace_prefix"
                                  else trace.get("exactTracePrefix"))
                    if not isinstance(source_raw, Mapping):
                        raise ValueError("stored project compilation lacks its scenario prefix")
                    source_prefix = ExactTracePrefixV1.build(source_raw)
                    mutable = {"sha256", "stateAssertions", "projectAssertions", "playActions", "projectActions"}
                    if ({key: value for key, value in prefix.to_dict().items() if key not in mutable}
                            != {key: value for key, value in source_prefix.to_dict().items() if key not in mutable}
                            or prefix.play_actions[:len(source_prefix.play_actions)] != source_prefix.play_actions
                            or prefix.project_actions[:len(source_prefix.project_actions)] != source_prefix.project_actions):
                        raise ValueError("stored project compilation changed its scenario prefix")
                model, receipt, seed, request = (row[key] for key in ("model", "receipt", "seed", "request"))
                from .ortools_complete_state_graph import CompleteStateGraphSeedV1
                CompleteStateGraphSeedV1.from_dict(seed)
                compiled_graph = ProofGraphV1.from_dict(row["graph"])
                compiled_counter = ProofGraphV1.from_dict(row["counterfactualGraph"])
                compiled_graph.require_same_legal_graph(compiled_counter)
                prefixes = [node_id[:-len(compiled_graph.root_node_id)] for node_id in parent_nodes
                            if node_id == compiled_graph.root_node_id
                            or node_id.endswith(":" + compiled_graph.root_node_id)]
                matching_prefixes = [prefix for prefix in prefixes if all(parent.get(prefix + node.node_id) == replace(node,
                            node_id=prefix + node.node_id,
                            edges=tuple(replace(edge, child_node_id=prefix + edge.child_node_id)
                                        for edge in node.edges))
                        for child, parent in ((compiled_graph, parent_nodes), (compiled_counter, parent_counter_nodes))
                        for node in child.nodes)]
                if not matching_prefixes:
                    raise ValueError("stored project compilation is not part of the parent proof")
                covered_terminals.update(prefix + node.node_id for prefix in matching_prefixes
                                         for node in compiled_graph.nodes if node.is_leaf)
                for namespace in matching_prefixes:
                    parent_frontiers.setdefault(namespace + compiled_graph.root_node_id, []).append(
                        ExactTracePrefixV1.build(row["exactTracePrefix"]))
                projection = {key: value for key, value in seed.items()
                              if key not in {"exactStateId", "proofStateProjectionHash"}}
                if (compiled_graph.scope != "exact"
                        or receipt.get("status") != "complete"
                        or receipt.get("exhaustionReason") is not None
                        or "elapsedSeconds" in receipt or "sha256" in receipt
                        or canonical_sha256(model) != receipt.get("modelHash")
                        or compiled_graph.sha256 != receipt.get("graphHash")
                        or compiled_counter.sha256 != model.get("projectCounterfactualGraphHash")
                        or seed.get("rulesHash") != dict(dict(recipe["configuration"].get("identities") or {}).get("rules") or {}).get("sha256")
                        or seed.get("exactStateId") != model.get("exactStateId")
                        or seed.get("exactStateId") != receipt.get("exactStateId")
                        or canonical_sha256(projection) != seed.get("proofStateProjectionHash")
                        or seed.get("proofStateProjectionHash") != model.get("proofStateProjectionHash")
                        or seed.get("proofStateProjectionHash") != receipt.get("proofStateProjectionHash")
                        or request.get("sha256") != canonical_sha256({key: value for key, value in request.items() if key != "sha256"})
                        or request.get("sha256") != model.get("requestHash")
                        or request.get("sha256") != receipt.get("requestHash")
                        or any(request.get(key) != recipe["proof"].get(key)
                               for key in ("mode", "visibility", "utility", "targetTeam"))):
                    raise ValueError("stored project compilation evidence ownership mismatch")
                for name in ("play", "scoring", "structural"):
                    if (dict(model.get("domainIrHashes") or {}).get(name) != authority["domainIrHashes"].get(name)
                            or dict(receipt.get("domainIrHashes") or {}).get(name) != authority["domainIrHashes"].get(name)):
                        raise ValueError("stored project compilation domain mismatch")
            if covered_terminals != {node.node_id for node in graph.nodes if node.is_leaf}:
                raise ValueError("stored project compilation terminal coverage is incomplete")
            snapshot = configuration_snapshot
            if (not isinstance(snapshot, Mapping)
                    or snapshot.get("sha256") != identity.get("configurationSnapshotHash")
                    or snapshot.get("bundleHash") != authority["bundleHash"]
                    or canonical_sha256({key: value for key, value in snapshot.items() if key != "sha256"}) != snapshot.get("sha256")):
                raise ValueError("project compilation configuration snapshot authority mismatch")
            frozen_rules = RuleBook(snapshot["components"]["rules"]["payload"])
            if frozen_rules.sha256 != snapshot["components"]["rules"]["sha256"]:
                raise ValueError("project compilation rules payload mismatch")
            settings = snapshot["components"]["challenge"]["payload"]["scenarioResearchV4"]
            if settings.get("backend") != authority["backendId"]:
                raise ValueError("project compilation backend mismatch")
            from .ortools_complete_state_graph import OrToolsCompleteStateGraphCompilerV1
            from baloot_platform.core.scoring_ir import ScoringDomainIRV1
            import time
            budget = ScenarioProofBudgetV1.build(recipe["proof"]["budgetProfile"],
                settings["proofBudgets"]["profiles"][recipe["proof"]["budgetProfile"]])
            deadline = time.monotonic() + min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds)
            compiler = OrToolsCompleteStateGraphCompilerV1(play_domain=frozen_rules.play_domain,
                scoring_domain=ScoringDomainIRV1.from_rules(frozen_rules.data),
                structural_domain=frozen_rules.structural_domain, project_domain=frozen_rules.project_domain,
                native_conversion=True)
            for row in rows:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError("project compilation reconstruction budget exhausted")
                if "exactTracePrefix" in row:
                    from .ortools_nonplay_play_graft import OrToolsNonPlayPlayGraphGraftV1
                    from .ortools_exact_state import OrToolsExactStateNormalizerV1
                    prefix = ExactTracePrefixV1.build(row["exactTracePrefix"])
                    normalized = OrToolsExactStateNormalizerV1(frozen_rules, native_cp_sat=True).normalize(
                        prefix, decision_point={"kind": "play"}, maximum_seconds=remaining)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ValueError("project compilation normalization budget exhausted")
                    graft = OrToolsNonPlayPlayGraphGraftV1(frozen_rules,
                        ScenarioProofRequestV1.build(row["request"]), native_cp_sat=True,
                        maximum_nodes=budget.maximum_nodes, maximum_depth=budget.maximum_depth,
                        maximum_seconds=remaining)
                    _, derived_seed = graft.native_project_play_seed(normalized, prefix)
                    if derived_seed.to_dict() != row["seed"]:
                        raise ValueError("project compilation prefix does not derive the retained seed")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ValueError("project compilation lifecycle budget exhausted")
                rebuilt = compiler.compile(CompleteStateGraphSeedV1.from_dict(row["seed"]),
                    ScenarioProofRequestV1.build(row["request"]), expected_rules_hash=frozen_rules.sha256,
                    maximum_nodes=min(budget.maximum_nodes, budget.maximum_state_space),
                    maximum_depth=budget.maximum_depth, maximum_seconds=remaining,
                    project_counterfactual=True,
                    _compact_forced="representation" in row["model"])
                if (not rebuilt.complete or rebuilt.graph.sha256 != row["graph"]["sha256"]
                        or rebuilt.project_counterfactual_graph is None
                        or rebuilt.project_counterfactual_graph.sha256 != row["counterfactualGraph"]["sha256"]
                        or rebuilt.receipt.model_hash != row["receipt"]["modelHash"]):
                    raise ValueError("project compilation reconstruction differs or is incomplete")
            _verify_project_parent_paths(graph, rules=frozen_rules, source_prefix=source_prefix,
                frontiers=parent_frontiers, maximum_nodes=min(budget.maximum_nodes, budget.maximum_state_space),
                maximum_seconds=deadline - time.monotonic())
    counterstrategy_label = "native_round_win_counterstrategy" in dict(
        result.get("qualification") or {}).get("failures", [])
    if proof_model.get("kind") == "native_counterstrategy_models":
        certificate = dict(proof_model.get("certificate") or {})
        certificate_hash = certificate.pop("sha256", None)
        expected_domains = {name: authority["domainIrHashes"].get(name)
                            for name in ("play", "structural", "projects", "scoring")}
        rules_identity = dict(dict(recipe["configuration"].get("identities") or {}).get("rules") or {})
        certified = certificate.get("status") == "certified"
        if counterstrategy_label and not certified:
            raise ValueError("stored native counterstrategy explanation lacks certification")
        facts = certificate.get("facts")
        if (certificate_hash != canonical_sha256(certificate)
                or certificate.get("schemaVersion") != 1
                or certificate.get("theoremId") != "SUN-ACE-LEAD-AWARDED-PROJECT-V1"
                or certificate.get("scenarioHash") != recipe["scenario"]["sha256"]
                or certificate.get("proofRequestHash") != recipe["proof"]["sha256"]
                or certificate.get("rulesHash") != rules_identity.get("sha256")
                or certificate.get("domainIrHashes") != expected_domains
                or certificate.get("solverVersion") != authority["solverVersion"]
                or certificate.get("status") not in {"certified", "incomplete", "not_certified"}
                or not isinstance(facts, list)
                or any(not isinstance(fact, Mapping) for fact in facts)):
            raise ValueError("stored native counterstrategy identity mismatch")
        if certified and (len(facts) != 6 or len({fact.get("factId") for fact in facts}) != 6
                          or any(fact.get("solverStatus") != "infeasible" for fact in facts)):
            raise ValueError("stored native counterstrategy certification is incomplete")
        for fact in facts:
            _require_hash(fact.get("modelHash"), "counterstrategy model hash")
        if (proof_outcome.get("kind") != "native_counterstrategy_outcome"
                or proof_outcome.get("certificateHash") != certificate_hash
                or proof_outcome.get("claimSatisfied") is not (False if certified else None)
                or proof_outcome.get("stateSpecificRootActionsEvaluated") is not False
                or (certified and result.get("terminalStatus") != "proof_failed")
                or result.get("disposition") == "qualified"
                or dict(result.get("scenarioProjection") or {}).get("exactStateAvailable") is not False):
            raise ValueError("stored native counterstrategy outcome mismatch")
    elif proof_outcome.get("kind") == "native_counterstrategy_outcome":
        raise ValueError("stored native counterstrategy model is missing")
    elif counterstrategy_label:
        raise ValueError("stored native counterstrategy explanation lacks a model")
    required_policy = (
        ScenarioPlayingQualityPolicyV1.build(recipe["playingQuality"])
        if recipe.get("playingQuality") is not None and (
            result.get("disposition") == "qualified"
            or dict(result.get("qualification") or {}).get("allDecisionPolicyAuditComplete") is True
        )
        else None
    )
    quality_verified = False
    if "playingQualityEvidence" in trace or required_policy is not None:
        proof_profiles = None
        search_profiles = None
        execution = None
        frozen_rules = None
        if configuration_snapshot is not None or required_policy is not None and "playingQualityEvidence" in trace:
            snapshot = configuration_snapshot
            if (not isinstance(snapshot, Mapping)
                    or snapshot.get("sha256") != identity.get("configurationSnapshotHash")
                    or snapshot.get("bundleHash") != authority["bundleHash"]
                    or canonical_sha256({k: v for k, v in snapshot.items() if k != "sha256"}) != snapshot.get("sha256")):
                raise ValueError("quality configuration snapshot authority mismatch")
            frozen_rules = RuleBook(snapshot["components"]["rules"]["payload"])
            if frozen_rules.sha256 != snapshot["components"]["rules"]["sha256"]:
                raise ValueError("quality configuration rules payload mismatch")
            settings = snapshot["components"]["challenge"]["payload"]["scenarioResearchV4"]
            if settings.get("backend") != authority["backendId"]:
                raise ValueError("quality configuration backend mismatch")
            proof_profiles = settings["proofBudgets"]["profiles"]
            search_profiles = settings.get("searchBudgets", {}).get("profiles")
            execution = ScenarioQualityExecutionBudgetV1.build(settings["qualityExecution"])
        normalization = trace.get("exactNormalization") or {}
        rules_hash = _require_hash(dict(normalization.get("state") or {}).get("rulesHash"),
                                   "trace quality rulesHash")
        validate_trace_quality_evidence_v1(trace, rules_hash=rules_hash,
            required_policy=required_policy, required_proof_profiles=proof_profiles,
            required_search_profiles=search_profiles,
            required_execution=execution, rules=frozen_rules, require_family_opening=True)
        quality_verified = dict(dict(trace.get("playingQualityEvidence") or {}).get("receipt") or {}).get("status") == "complete"
        if required_policy is not None:
            receipt = trace["playingQualityEvidence"]["receipt"]
            qualification = dict(result.get("qualification") or {})
            if (
                qualification.get("allDecisionPolicyAuditComplete") is not True
                or qualification.get("playingQualityMode") != required_policy.mode
                or (required_policy.schema_version == 2 and (
                    qualification.get("playingQualityVisibility") != required_policy.visibility
                    or qualification.get("actionSelectionObserverSafe") is not
                    (required_policy.visibility == "actor_information_set")))
                or qualification.get("playingQualityEvidenceHash")
                != receipt.get("sha256")
            ):
                raise ValueError(
                    "stored OR-Tools Research V3 playing-quality binding is incomplete"
                )
    _validate_native_interest_admission_v1(result, trace, configuration_snapshot,
                                         verified_quality=quality_verified)
    if 'lessonTags' in trace or 'lessonTags' in dict(result.get('scenarioProjection') or {}):
        if not quality_verified:
            raise ValueError('lesson tags require completed quality evidence')
        lesson_tags = validated_trace_lesson_tags_v1(trace, rules=frozen_rules)
        if lesson_tags != dict(result.get('scenarioProjection') or {}).get('lessonTags'):
            raise ValueError('stored lesson tag projection substitution detected')
    validate_generation_result_binding_v1(recipe, trace, configuration_snapshot,
        qualified=dict(result.get('qualification') or {}).get('qualified') is True)
    if not _requires_v3_compilation_receipt(authority):
        return result

    try:
        raw_construction = _closed_document(
            semantic_documents["constructionModel"],
            required=_V3_COMPILATION_RECEIPT_FIELDS,
            label="OR-Tools V3 compilation receipt",
        )
        if canonical_sha256(raw_construction) != authority["constructionModelHash"]:
            raise ValueError("OR-Tools V3 compilation receipt artifact hash mismatch")
        construction = _verify_integrity(
            raw_construction, "OR-Tools V3 compilation receipt"
        )
        if (
            construction["schemaVersion"] != 2
            or construction["kind"] != _V3_COMPILATION_RECEIPT_KIND
            or construction["backendId"] != authority["backendId"]
            or construction["backendVersion"] != authority["backendVersion"]
            or construction["scenarioHash"] != identity.get("scenarioHash")
            or construction["status"]
            not in _V3_EXECUTION_STATUSES_BY_TERMINAL.get(
                str(result.get("terminalStatus") or ""), frozenset()
            )
            or (
                result.get("terminalStatus") == "not_found_within_budget"
                and construction["status"] == "incomplete"
                and construction["candidateIdentity"] is not None
            )
        ):
            raise ValueError("OR-Tools V3 compilation receipt identity mismatch")
        _require_hash(construction["preflightHash"], "OR-Tools V3 preflightHash")
        compiled_stages = _canonical_text_list(
            construction["compiledStages"], "OR-Tools V3 compiledStages"
        )
        executed_stages = _canonical_text_list(
            construction["executedStages"], "OR-Tools V3 executedStages"
        )
        stage_receipts_raw = construction["stageReceipts"]
        if not isinstance(stage_receipts_raw, Mapping):
            raise TypeError("OR-Tools V3 stageReceipts must be an object")
        stage_receipts = {
            _require_text(stage, "OR-Tools V3 stage receipt key"):
            validate_ortools_stage_receipt_v2(receipt)
            for stage, receipt in sorted(stage_receipts_raw.items())
        }
        if (
            set(executed_stages) != set(stage_receipts)
            or any(receipt["stage"] != stage for stage, receipt in stage_receipts.items())
            or (
                result.get("terminalStatus") == "satisfied"
                and not set(compiled_stages) <= set(executed_stages)
            )
        ):
            raise ValueError("OR-Tools V3 compiled stage receipt coverage mismatch")
        configuration_authority = _closed_document(
            construction["configurationAuthority"],
            required=_V3_COMPILATION_AUTHORITY_FIELDS,
            label="OR-Tools V3 compilation configuration authority",
        )
        domain_hashes = _validate_v3_domain_ir_hashes(
            configuration_authority["domainIrHashes"]
        )
        if (
            configuration_authority["bundleRevisionId"]
            != authority["bundleRevisionId"]
            or configuration_authority["bundleHash"] != authority["bundleHash"]
            or domain_hashes != authority["domainIrHashes"]
            or configuration_authority["loweringManifestHash"]
            != authority["loweringManifestHash"]
        ):
            raise ValueError("OR-Tools V3 compilation configuration authority mismatch")
        artifact_hashes = _closed_document(
            construction["semanticArtifactHashes"],
            required=_V3_COMPILATION_ARTIFACT_HASH_FIELDS,
            label="OR-Tools V3 compilation semantic artifact hashes",
        )
        expected_artifact_hashes = {
            "proofModel": authority["proofModelHash"],
            "proofGraph": authority["graphHash"],
            "completeTrace": authority["traceHash"],
            "proofOutcome": authority["proofOutcomeHash"],
        }
        if artifact_hashes != expected_artifact_hashes:
            raise ValueError("OR-Tools V3 compilation semantic artifact mismatch")

        predicate = _validate_predicate_evaluation_v1(
            construction["predicateEvaluation"],
            expected_hash=identity.get("predicateEvaluationHash"),
            scenario=recipe["scenario"],
        )
        if predicate != result.get("predicateEvaluation"):
            raise ValueError("OR-Tools V3 predicate evaluation substitution detected")
        satisfied = result.get("terminalStatus") == "satisfied"
        candidate_identity_raw = construction["candidateIdentity"]
        if candidate_identity_raw is None:
            if satisfied or any(
                identity.get(field) is not None
                for field in ("candidateHash", "exactStateHash", "exactStateId")
            ):
                raise ValueError("OR-Tools V3 candidate identity preimage is missing")
        else:
            candidate_identity = _closed_document(
                candidate_identity_raw,
                required=_V3_CANDIDATE_IDENTITY_FIELDS,
                label="OR-Tools V3 candidate identity",
            )
            if candidate_identity["schemaVersion"] != 1:
                raise ValueError("OR-Tools V3 candidate identity schemaVersion must be 1")
            origin = _require_text(
                candidate_identity["origin"], "OR-Tools V3 candidate origin"
            )
            no_good = candidate_identity["noGoodVector"]
            if not isinstance(no_good, list) or any(type(item) is not int for item in no_good):
                raise ValueError("OR-Tools V3 candidate noGoodVector is invalid")
            exact_state_id = candidate_identity["exactStateId"]
            if exact_state_id is not None:
                _require_text(exact_state_id, "OR-Tools V3 candidate exactStateId")
            trace = semantic_documents["completeTrace"]
            exact_normalization = trace.get("exactNormalization")
            state = (
                dict(exact_normalization.get("state") or {})
                if isinstance(exact_normalization, Mapping)
                else dict(trace.get("partialConstructionState") or {})
            )
            state_hash = canonical_sha256(state)
            if (
                candidate_identity["stateHash"] != state_hash
                or identity.get("exactStateHash") != (state_hash if exact_state_id is not None else None)
                or candidate_identity["exactStateId"] != identity.get("exactStateId")
                or (exact_state_id is None and (satisfied or exact_normalization is not None
                                               or "partialConstructionState" not in trace))
                or (exact_state_id is not None and (not state or not isinstance(exact_normalization, Mapping)))
            ):
                raise ValueError("OR-Tools V3 exact-state identity mismatch")
            candidate_core = {
                "schemaVersion": 1,
                "origin": origin,
                "trace": trace,
                "state": state,
                "noGoodVector": no_good,
                "stageReceipts": stage_receipts,
                "exactStateId": exact_state_id,
            }
            if canonical_sha256(candidate_core) != identity.get("candidateHash"):
                raise ValueError("OR-Tools V3 candidate hash substitution detected")

        proof_document = semantic_documents["proofOutcome"]
        if satisfied:
            proof_identities = dict(proof_document.get("identities") or {})
            if (
                proof_document.get("completionStatus")
                not in {"exact_complete", "bounded_complete", "policy_complete"}
                or proof_document.get("claimSatisfied") is not True
                or proof_identities.get("graphHash") != authority["graphHash"]
                or proof_identities.get("modelHash") != authority["proofModelHash"]
            ):
                raise ValueError("OR-Tools V3 proof artifact semantic binding mismatch")
        proof_request = dict(recipe.get("proof") or {})
        expected_proof_receipt = _proof_receipt(
            raw_proof=_terminal_proof_source_v3(
                raw_proof=proof_document if "completionStatus" in proof_document else None,
                terminal_status=str(result.get("terminalStatus") or ""),
                proof_request=proof_request,
                exact_normalization=trace.get("exactNormalization"),
                backend_version=authority.get("backendVersion"),
            ),
            proof_request=proof_request,
            configuration=dict(recipe.get("configuration") or {}),
            authority=authority,
        )
        if expected_proof_receipt != result.get("proofOutcome"):
            raise ValueError("OR-Tools V3 proof receipt substitution detected")
        if campaign_document is None:
            raise ValueError("OR-Tools V3 campaign semantic preimage is missing")
        _validate_campaign_semantic_document_v1(
            campaign_document,
            identity=identity,
            satisfied=satisfied,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"stored OR-Tools Research V3 compilation receipt is invalid: {exc}"
        ) from exc
    return result


@dataclass(frozen=True, slots=True)
class RuntimeCompatibilityV1:
    """Downstream engine compatibility evidence; never qualification evidence."""

    _canonical_json: str

    @classmethod
    def build(
        cls,
        *,
        status: str,
        bundle_hash: str,
        trace_hash: str,
        engine_version: str | None = None,
        evidence_hash: str | None = None,
        failure_codes: tuple[str, ...] = (),
    ) -> "RuntimeCompatibilityV1":
        core = {
            "schemaVersion": 1,
            "status": status,
            "engineVersion": engine_version,
            "bundleHash": bundle_hash,
            "traceHash": trace_hash,
            "evidenceHash": evidence_hash,
            "failureCodes": list(failure_codes),
        }
        return cls.from_dict({**core, "sha256": canonical_sha256(core)})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeCompatibilityV1":
        fields = frozenset(
            {
                "schemaVersion",
                "status",
                "engineVersion",
                "bundleHash",
                "traceHash",
                "evidenceHash",
                "failureCodes",
                "sha256",
            }
        )
        document = _verify_integrity(
            _closed_document(value, required=fields, label="runtimeCompatibility"),
            "runtimeCompatibility",
        )
        if type(document["schemaVersion"]) is not int or document["schemaVersion"] != 1:
            raise ValueError("runtimeCompatibility schemaVersion must be 1")
        status = document["status"]
        if status not in {"not_checked", "passed", "failed"}:
            raise ValueError("runtimeCompatibility status is invalid")
        _require_hash(document["bundleHash"], "runtimeCompatibility bundleHash")
        _require_hash(document["traceHash"], "runtimeCompatibility traceHash")
        engine = document["engineVersion"]
        evidence = document["evidenceHash"]
        codes = document["failureCodes"]
        if not isinstance(codes, list) or any(type(item) is not str for item in codes):
            raise TypeError("runtimeCompatibility failureCodes must be strings")
        canonical_codes = sorted(
            {
                _require_text(item, "runtime compatibility failure code")
                for item in codes
            }
        )
        if codes != canonical_codes:
            raise ValueError(
                "runtimeCompatibility failureCodes must be sorted and unique"
            )
        if status == "not_checked":
            if engine is not None or evidence is not None or codes:
                raise ValueError(
                    "unchecked runtimeCompatibility cannot contain evidence"
                )
        else:
            _require_text(engine, "runtimeCompatibility engineVersion")
            _require_hash(evidence, "runtimeCompatibility evidenceHash")
            if status == "passed" and codes:
                raise ValueError(
                    "passed runtimeCompatibility cannot contain failure codes"
                )
            if status == "failed" and not codes:
                raise ValueError("failed runtimeCompatibility requires failure codes")
        return cls(
            json.dumps(
                document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)


@dataclass(frozen=True, slots=True)
class ExecutionReceiptV1:
    """Integrity-bound resource receipt excluded from semantic result identity."""

    _canonical_json: str

    @classmethod
    def build(
        cls,
        *,
        semantic_result_hash: str,
        stage_timings: Mapping[str, float | int],
        cpu_seconds: float | int,
        peak_memory_bytes: int,
        conflicts: int,
        branches: int,
        cache_bytes: int,
        completed_variants: int,
        machine: Mapping[str, Any],
    ) -> "ExecutionReceiptV1":
        core = {
            "schemaVersion": 1,
            "semanticResultHash": semantic_result_hash,
            "stageTimings": dict(sorted(stage_timings.items())),
            "cpuSeconds": cpu_seconds,
            "peakMemoryBytes": peak_memory_bytes,
            "conflicts": conflicts,
            "branches": branches,
            "cacheBytes": cache_bytes,
            "completedVariants": completed_variants,
            "machine": copy.deepcopy(dict(machine)),
        }
        return cls.from_dict({**core, "sha256": canonical_sha256(core)})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionReceiptV1":
        fields = frozenset(
            {
                "schemaVersion",
                "semanticResultHash",
                "stageTimings",
                "cpuSeconds",
                "peakMemoryBytes",
                "conflicts",
                "branches",
                "cacheBytes",
                "completedVariants",
                "machine",
                "sha256",
            }
        )
        document = _verify_integrity(
            _closed_document(value, required=fields, label="executionReceipt"),
            "executionReceipt",
        )
        if type(document["schemaVersion"]) is not int or document["schemaVersion"] != 1:
            raise ValueError("executionReceipt schemaVersion must be 1")
        _require_hash(
            document["semanticResultHash"], "executionReceipt semanticResultHash"
        )
        timings = document["stageTimings"]
        if not isinstance(timings, Mapping):
            raise ValueError("executionReceipt stageTimings must be an object")
        for stage, duration in timings.items():
            _require_text(stage, "executionReceipt stage name")
            if (
                type(duration) not in {int, float}
                or not math.isfinite(duration)
                or duration < 0
            ):
                raise ValueError(
                    "executionReceipt stage timings must be finite and non-negative"
                )
        document["stageTimings"] = dict(sorted(timings.items()))
        cpu = document["cpuSeconds"]
        if type(cpu) not in {int, float} or not math.isfinite(cpu) or cpu < 0:
            raise ValueError(
                "executionReceipt cpuSeconds must be finite and non-negative"
            )
        for field in (
            "peakMemoryBytes",
            "conflicts",
            "branches",
            "cacheBytes",
            "completedVariants",
        ):
            if type(document[field]) is not int or document[field] < 0:
                raise ValueError(
                    f"executionReceipt {field} must be a non-negative integer"
                )
        if not isinstance(document["machine"], Mapping):
            raise TypeError("executionReceipt machine must be an object")
        return cls(
            json.dumps(
                document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)


def build_ortools_result_identity_v2(
    base_identity: Mapping[str, Any],
    research_authority: ResearchAuthorityV1 | Mapping[str, Any],
    *,
    domain_authority_hash: str,
    executor_semantic_hash: str,
    candidate_hash: str,
    predicate_evaluation_hash: str,
) -> dict[str, Any]:
    """Build the closed semantic identity without runtime measurements."""

    identity = copy.deepcopy(dict(base_identity))
    unknown = sorted(set(identity) - _BASE_IDENTITY_FIELDS)
    if unknown:
        raise ValueError(f"OR-Tools result identity base mismatch; unknown={unknown}")
    authority = (
        research_authority
        if isinstance(research_authority, ResearchAuthorityV1)
        else ResearchAuthorityV1.from_dict(research_authority)
    ).to_dict()
    if authority["backendId"] != ORTOOLS_RESEARCH_BACKEND:
        raise ValueError(
            "OR-Tools V2 result identity requires explicit V2 backend authority"
        )
    artifact_hashes = dict(identity.get("artifactHashes") or {})
    if set(artifact_hashes) != _SEMANTIC_ARTIFACT_NAMES:
        raise ValueError("OR-Tools result identity requires exactly five artifacts")
    identity.update(
        {
            "schemaVersion": ORTOOLS_RESULT_IDENTITY_SCHEMA_VERSION,
            "kind": ORTOOLS_RESULT_IDENTITY_KIND,
            "backendId": authority["backendId"],
            "backendVersion": authority["backendVersion"],
            "qualificationAuthority": authority["qualificationAuthority"],
            "bundleRevisionId": authority["bundleRevisionId"],
            "bundleHash": authority["bundleHash"],
            "domainAuthorityHash": _require_hash(
                domain_authority_hash, "domain authority hash"
            ),
            "domainIrHashesHash": canonical_sha256(authority["domainIrHashes"]),
            "loweringManifestHash": authority["loweringManifestHash"],
            "constructionModelHash": authority["constructionModelHash"],
            "proofModelHash": authority["proofModelHash"],
            "graphHash": authority["graphHash"],
            "traceHash": authority["traceHash"],
            "proofOutcomeHash": authority["proofOutcomeHash"],
            "executorSemanticHash": _require_hash(
                executor_semantic_hash, "executor semantic hash"
            ),
            "candidateHash": _require_hash(candidate_hash, "candidate hash"),
            "predicateEvaluationHash": _require_hash(
                predicate_evaluation_hash, "predicate evaluation hash"
            ),
            "researchAuthorityHash": authority["sha256"],
        }
    )
    missing = sorted(_REQUIRED_IDENTITY_FIELDS - set(identity))
    if missing:
        raise ValueError(f"OR-Tools result identity is incomplete; missing={missing}")
    return identity


_V3_IDENTITY_FIELDS = frozenset(
    {
        "schemaVersion",
        "kind",
        "terminalStatus",
        "runCode",
        "runHash",
        "recipeCode",
        "recipeHash",
        "configurationAuthorityHash",
        "configurationSnapshotHash",
        "scenarioHash",
        "exactStateHash",
        "exactStateId",
        "scenarioSatisfactionHash",
        "scenarioProjectionHash",
        "qualificationHash",
        "verdictHash",
        "proofOutcomeReceiptHash",
        "artifactHashes",
        "authorityArtifactHashes",
        "researchAuthorityHash",
        "backendId",
        "backendVersion",
        "qualificationAuthority",
        "bundleRevisionId",
        "bundleHash",
        "domainAuthorityHash",
        "domainIrHashesHash",
        "loweringManifestHash",
        "constructionModelHash",
        "proofModelHash",
        "graphHash",
        "traceHash",
        "proofOutcomeHash",
        "executorSemanticHash",
        "campaignSemanticHash",
        "candidateHash",
        "predicateEvaluationHash",
    }
)


def build_ortools_result_identity_v3(
    base_identity: Mapping[str, Any],
    research_authority: ResearchAuthorityV1 | Mapping[str, Any],
    *,
    terminal_status: str,
    configuration_snapshot_hash: str,
    authority_artifact_hashes: Mapping[str, str],
    domain_authority_hash: str,
    executor_semantic_hash: str,
    campaign_semantic_hash: str,
    candidate_hash: str | None,
    predicate_evaluation_hash: str | None,
) -> dict[str, Any]:
    """Build the V3 identity shared by qualified and terminal audit results."""

    if terminal_status not in _TERMINAL_STATUSES:
        raise ValueError("OR-Tools result terminal status is unsupported")
    authority = (
        research_authority
        if isinstance(research_authority, ResearchAuthorityV1)
        else ResearchAuthorityV1.from_dict(research_authority)
    ).to_dict()
    identity = copy.deepcopy(dict(base_identity))
    semantic_hashes = dict(identity.get("artifactHashes") or {})
    if set(semantic_hashes) != _SEMANTIC_ARTIFACT_NAMES:
        raise ValueError("OR-Tools V3 result identity requires five semantic artifacts")
    authority_hashes = dict(authority_artifact_hashes)
    if set(authority_hashes) - {"interestAdmission"} != {"loweringManifest", "configurationSnapshot"}:
        raise ValueError(
            "OR-Tools V3 result identity authority artifacts are incomplete"
        )
    for name, digest in {**semantic_hashes, **authority_hashes}.items():
        _require_hash(digest, f"OR-Tools V3 {name} artifact hash")
    nullable_hashes = {
        "candidateHash": candidate_hash,
        "predicateEvaluationHash": predicate_evaluation_hash,
        "exactStateHash": identity.get("exactStateHash"),
    }
    for name, digest in nullable_hashes.items():
        if digest is not None:
            _require_hash(digest, f"OR-Tools V3 {name}")
    identity.update(
        {
            "schemaVersion": ORTOOLS_RESULT_IDENTITY_V3_SCHEMA_VERSION,
            "kind": ORTOOLS_RESULT_IDENTITY_V3_KIND,
            "terminalStatus": terminal_status,
            "configurationSnapshotHash": _require_hash(
                configuration_snapshot_hash, "configuration snapshot hash"
            ),
            "authorityArtifactHashes": dict(sorted(authority_hashes.items())),
            "researchAuthorityHash": authority["sha256"],
            "backendId": authority["backendId"],
            "backendVersion": authority["backendVersion"],
            "qualificationAuthority": authority["qualificationAuthority"],
            "bundleRevisionId": authority["bundleRevisionId"],
            "bundleHash": authority["bundleHash"],
            "domainAuthorityHash": _require_hash(
                domain_authority_hash, "domain authority hash"
            ),
            "domainIrHashesHash": canonical_sha256(authority["domainIrHashes"]),
            "loweringManifestHash": authority["loweringManifestHash"],
            "constructionModelHash": authority["constructionModelHash"],
            "proofModelHash": authority["proofModelHash"],
            "graphHash": authority["graphHash"],
            "traceHash": authority["traceHash"],
            "proofOutcomeHash": authority["proofOutcomeHash"],
            "executorSemanticHash": _require_hash(
                executor_semantic_hash, "executor semantic hash"
            ),
            "campaignSemanticHash": _require_hash(
                campaign_semantic_hash, "campaign semantic hash"
            ),
            "candidateHash": candidate_hash,
            "predicateEvaluationHash": predicate_evaluation_hash,
        }
    )
    unknown = sorted(set(identity) - _V3_IDENTITY_FIELDS)
    missing = sorted(_V3_IDENTITY_FIELDS - set(identity))
    if unknown or missing:
        raise ValueError(
            f"OR-Tools V3 result identity shape mismatch; missing={missing}, unknown={unknown}"
        )
    if (
        identity.get("exactStateId") is not None
        and re.fullmatch(r"STATE-[A-F0-9]{20}", str(identity["exactStateId"])) is None
    ):
        raise ValueError("OR-Tools V3 exact-state ID is not canonical")
    return identity


def _as_document(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(f"{label} must be a mapping or expose to_dict()")
    document = to_dict()
    if not isinstance(document, Mapping):
        raise TypeError(f"{label} to_dict() must return an object")
    return copy.deepcopy(dict(document))


def _configuration_domain_hash(configuration: Mapping[str, Any]) -> str:
    identities = configuration.get("identities")
    domain = (
        dict(identities.get("domain") or {}) if isinstance(identities, Mapping) else {}
    )
    return _require_hash(domain.get("sha256"), "configuration domain authority hash")


def _validate_cas_uri(uri: Any, digest: str, label: str) -> str:
    value = _require_text(uri, f"{label} uri")
    parsed = urlparse(value)
    local = (
        parsed.scheme == "cas"
        and parsed.netloc == "sha256"
        and parsed.path == f"/{digest}"
    )
    s3 = (
        parsed.scheme == "s3"
        and bool(parsed.netloc)
        and parsed.path not in {"", "/"}
        and parsed.path.rsplit("/", 1)[-1] == digest
    )
    if parsed.params or parsed.query or parsed.fragment or not (local or s3):
        raise ValueError(f"{label} URI does not match its hash")
    return value


def _bind_semantic_artifacts(
    *,
    documents: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    authority: Mapping[str, Any],
    owner_hash: str,
) -> dict[str, dict[str, Any]]:
    if set(documents) != _SEMANTIC_ARTIFACT_NAMES:
        raise ValueError(
            "semantic artifact documents must contain exactly five artifacts"
        )
    if set(references) != _SEMANTIC_ARTIFACT_NAMES:
        raise ValueError(
            "semantic artifact references must contain exactly five artifacts"
        )
    owned: dict[str, dict[str, Any]] = {}
    for authority_field, name in _SEMANTIC_ARTIFACT_FIELDS.items():
        document = copy.deepcopy(dict(documents[name]))
        digest = canonical_sha256(document)
        if digest != authority[authority_field]:
            raise ValueError(f"{name} semantic artifact preimage mismatch")
        reference = _closed_document(
            references[name], required=_CAS_REFERENCE_FIELDS, label=f"{name} artifact"
        )
        if reference["sha256"] != digest:
            raise ValueError(f"{name} artifact CAS hash mismatch")
        payload = json.dumps(
            document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if type(reference["bytes"]) is not int or reference["bytes"] != len(payload):
            raise ValueError(f"{name} artifact CAS byte count mismatch")
        if reference["mediaType"] != ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES[name]:
            raise ValueError(f"{name} artifact media type mismatch")
        if reference["name"] is not None:
            _require_text(reference["name"], f"{name} artifact name")
        _validate_cas_uri(reference["uri"], digest, f"{name} artifact")
        owned[name] = {
            **reference,
            "ownerKind": _SEMANTIC_ARTIFACT_OWNER_KIND,
            "ownerHash": owner_hash,
        }
    return owned


def _runtime_check_documents(
    value: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if isinstance(value, Mapping):
        raw = dict(value)
        receipt_raw = raw.get("runtimeCompatibility") or raw.get("receipt")
        evidence_raw = raw.get("evidence")
    else:
        receipt_raw = getattr(value, "receipt", None)
        evidence_raw = getattr(value, "evidence", None)
    receipt = RuntimeCompatibilityV1.from_dict(
        _as_document(receipt_raw, "runtime compatibility receipt")
    ).to_dict()
    evidence = (
        _as_document(evidence_raw, "runtime compatibility evidence")
        if evidence_raw is not None
        else None
    )
    return receipt, evidence


def _quarantined_runtime_check(
    *,
    semantic_result_hash: str,
    bundle_hash: str,
    trace_hash: str,
    failure_code: str,
    failure: Exception,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Local import keeps the compatibility checker downstream of this authority module.
    from .runtime_compatibility import quarantine_runtime_compatibility_failure_v1

    return _runtime_check_documents(
        quarantine_runtime_compatibility_failure_v1(
            semantic_result_hash=semantic_result_hash,
            bundle_hash=bundle_hash,
            trace_hash=trace_hash,
            failure_code=failure_code,
            failure=failure,
        )
    )


def _validate_runtime_evidence(
    *,
    compatibility: Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
    semantic_result_hash: str,
) -> None:
    status = compatibility["status"]
    if status == "not_checked":
        if evidence is not None:
            raise ValueError("unchecked runtime compatibility cannot contain evidence")
        return
    if evidence is None:
        raise ValueError("checked runtime compatibility requires evidence")
    document = copy.deepcopy(dict(evidence))
    digest = _require_hash(document.pop("sha256", None), "runtime evidence sha256")
    if canonical_sha256(document) != digest or compatibility["evidenceHash"] != digest:
        raise ValueError("runtime compatibility evidence integrity mismatch")
    if (
        document.get("semanticResultHash") != semantic_result_hash
        or document.get("bundleHash") != compatibility["bundleHash"]
        or document.get("traceHash") != compatibility["traceHash"]
        or document.get("failureCodes") != compatibility["failureCodes"]
    ):
        raise ValueError("runtime compatibility evidence ownership mismatch")
    if status == "passed":
        required = {
            "schemaVersion",
            "kind",
            "semanticResultHash",
            "bundleHash",
            "traceHash",
            "engineVersion",
            "rulesHash",
            "exactStateId",
            "exactStateDocumentHash",
            "declaredStateHash",
            "derivedStateHash",
            "expectedStateProjectionHash",
            "derivedStateProjectionHash",
            "expectedTraceProjectionHash",
            "derivedTraceProjectionHash",
            "replayStatus",
            "replayFailureHash",
            "failureCodes",
        }
        if set(document) != required:
            raise ValueError("passed runtime compatibility evidence is incomplete")
        if (
            document["schemaVersion"] != 1
            or document["kind"] != _RUNTIME_COMPATIBILITY_EVIDENCE_KIND
            or document["engineVersion"] != compatibility["engineVersion"]
            or document["replayStatus"] != "completed"
            or document["replayFailureHash"] is not None
            or document["failureCodes"] != []
            or document["expectedStateProjectionHash"]
            != document["derivedStateProjectionHash"]
            or document["expectedTraceProjectionHash"]
            != document["derivedTraceProjectionHash"]
        ):
            raise ValueError(
                "passed runtime compatibility evidence is not a matching replay"
            )
        _require_text(document["exactStateId"], "runtime evidence exact state id")
        for field in (
            "rulesHash",
            "exactStateDocumentHash",
            "declaredStateHash",
            "derivedStateHash",
            "expectedStateProjectionHash",
            "derivedStateProjectionHash",
            "expectedTraceProjectionHash",
            "derivedTraceProjectionHash",
        ):
            _require_hash(document[field], f"runtime evidence {field}")


def _execution_receipt(
    executor_receipt: Mapping[str, Any],
    *,
    semantic_result_hash: str,
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    timings = dict(executor_receipt.get("stageTimings") or {})
    stage_receipts = dict((candidate or {}).get("stageReceipts") or {})
    return ExecutionReceiptV1.build(
        semantic_result_hash=semantic_result_hash,
        stage_timings=timings,
        cpu_seconds=executor_receipt.get("cpuSeconds", 0),
        peak_memory_bytes=executor_receipt.get("peakMemoryBytes", 0),
        conflicts=sum(
            _non_negative_integer(dict(item).get("conflicts", 0), "stage conflicts")
            for item in stage_receipts.values()
        ),
        branches=sum(
            _non_negative_integer(dict(item).get("branches", 0), "stage branches")
            for item in stage_receipts.values()
        ),
        cache_bytes=executor_receipt.get("cacheBytes", 0),
        completed_variants=executor_receipt.get(
            "completedVariants", executor_receipt.get("completedCandidates", 1)
        ),
        machine=dict(executor_receipt.get("machine") or {}),
    ).to_dict()


def _proof_receipt(
    *,
    raw_proof: Mapping[str, Any],
    proof_request: Mapping[str, Any],
    configuration: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    completion = str(raw_proof.get("completionStatus") or "")
    achieved_mode = {
        "exact_complete": "exact",
        "bounded_complete": "bounded",
        "policy_complete": "policy_conditioned",
        "proof_failed": str(proof_request.get("mode") or "none"),
    }.get(completion, "none")
    visibility = str(proof_request.get("visibility") or "complete_state")
    hidden_worlds = raw_proof.get("hiddenWorlds")
    if not isinstance(hidden_worlds, Mapping):
        if visibility != "complete_state":
            raise ValueError("information-set proof must report hidden-world coverage")
        hidden_worlds = {
            "count": 1,
            "complete": True,
            "coverage": {"method": "complete_state"},
        }
    continuation = raw_proof.get("continuationLines")
    if isinstance(continuation, Mapping):
        continuation_value = {
            key: copy.deepcopy(continuation.get(key))
            for key in (
                "complete",
                "exactCount",
                "lowerBound",
                "upperBound",
                "lineUniverse",
            )
        }
    else:
        continuation_value = {
            "complete": False,
            "exactCount": None,
            "lowerBound": "0",
            "upperBound": None,
            "lineUniverse": str(proof_request.get("lineUniverse") or "all_legal"),
        }
    identities = dict(configuration.get("identities") or {})
    rules_hash = _require_hash(
        dict(identities.get("rules") or {}).get("sha256"),
        "configuration rules authority hash",
    )
    policy_identities = dict(raw_proof.get("policyIdentities") or {})
    policy_hashes = {
        str(seat): _require_hash(
            dict(policy).get("profileHash"), f"proof policy {seat} hash"
        )
        for seat, policy in sorted(policy_identities.items())
    }
    proof_identities = {
        "rulesHash": rules_hash,
        "bundleHash": authority["bundleHash"],
        "graphHash": authority["graphHash"],
        "modelHash": authority["proofModelHash"],
        "proofRequestHash": str(
            dict(raw_proof.get("identities") or {}).get("proofRequestHash")
            or proof_request.get("sha256")
            or ""
        ),
        "policyHashes": policy_hashes,
    }
    _require_hash(proof_identities["proofRequestHash"], "proof request hash")
    strategy_class = str(
        proof_request.get("strategyClass")
        or (
            "frozen_policy"
            if proof_request.get("mode") == "policy_conditioned"
            else "pure_deterministic"
        )
    )
    model_size = dict(raw_proof.get("modelSize") or {})
    raw_winning_count = raw_proof.get("winningRootActionCount")
    winning_count = (
        None if raw_winning_count is None else int(raw_winning_count)
    )
    raw_lower_bound = raw_proof.get("winningRootActionLowerBound")
    raw_upper_bound = raw_proof.get("winningRootActionUpperBound")
    winning_lower_bound = (
        winning_count
        if raw_lower_bound is None and winning_count is not None
        else int(raw_lower_bound or 0)
    )
    winning_upper_bound = (
        winning_count
        if raw_upper_bound is None and winning_count is not None
        else len(raw_proof.get("rootActions") or ())
        if raw_upper_bound is None
        else int(raw_upper_bound)
    )
    core = {
        "schemaVersion": 1,
        "requestedMode": proof_request.get("mode"),
        "achievedMode": achieved_mode,
        "requestedVisibility": visibility,
        "achievedVisibility": visibility if achieved_mode != "none" else "none",
        "strategyClass": strategy_class,
        "claim": proof_request.get("claim"),
        "targetTeam": proof_request.get("targetTeam"),
        "utility": proof_request.get("utility"),
        "successThreshold": proof_request.get("successThreshold"),
        "horizon": copy.deepcopy(proof_request.get("horizon")),
        "boundaryEvaluator": proof_request.get("boundaryEvaluator"),
        "rootActions": copy.deepcopy(list(raw_proof.get("rootActions") or ())),
        "selectedAction": raw_proof.get("selectedAction"),
        "winningRootActionCount": winning_count,
        "winningRootActionLowerBound": winning_lower_bound,
        "winningRootActionUpperBound": winning_upper_bound,
        "continuationLines": continuation_value,
        "hiddenWorlds": copy.deepcopy(dict(hidden_worlds)),
        "resources": {
            "nodes": int(model_size.get("variables", 0)),
            "strategyClass": strategy_class,
            "workers": 1,
        },
        "completionStatus": completion,
        "claimSatisfied": raw_proof.get("claimSatisfied") is True,
        "principalVariation": copy.deepcopy(
            list(raw_proof.get("principalVariation") or ())
        ),
        "identities": proof_identities,
        "semanticSha256": authority["proofOutcomeHash"],
        "artifactSha256": authority["proofOutcomeHash"],
    }
    return {**core, "sha256": canonical_sha256(core)}


def _terminal_proof_source_v3(
    *,
    raw_proof: Mapping[str, Any] | None,
    terminal_status: str,
    proof_request: Mapping[str, Any],
    exact_normalization: Mapping[str, Any] | None,
    backend_version: Any,
) -> dict[str, Any]:
    proof_source = copy.deepcopy(dict(raw_proof or {}))
    if not proof_source:
        proof_source = {
            "completionStatus": (
                "proof_failed"
                if terminal_status == "proof_failed"
                else "proof_incomplete"
            ),
            "claimSatisfied": False,
            "rootActions": [],
            "selectedAction": None,
            "winningRootActionCount": None,
            "winningRootActionLowerBound": 0,
            "winningRootActionUpperBound": None,
            "principalVariation": [],
            "hiddenWorlds": {
                "count": 0,
                "complete": False,
                "coverage": {"method": "not_run"},
            },
            "modelSize": {"variables": 0},
            "identities": {"proofRequestHash": proof_request.get("sha256")},
        }
    elif terminal_status in {"proof_incomplete", "proof_failed"}:
        proof_source["completionStatus"] = terminal_status
        proof_source["claimSatisfied"] = False
    if (
        isinstance(exact_normalization, Mapping)
        and terminal_status == "proof_incomplete"
        and not list(proof_source.get("rootActions") or ())
    ):
        legal_actions = [
            str(dict(action).get("code") or "")
            for action in exact_normalization.get("legalActions") or ()
            if str(dict(action).get("code") or "")
        ]
        if (_backend_version_tuple(backend_version) or (0, 0, 0)) < (3, 1, 137):
            # Preserve historical receipt hashes, not new unproved valuations.
            proof_source["rootActions"] = [
                {"action": action, "value": 0, "satisfies": False}
                for action in legal_actions
            ]
        elif proof_source.get("winningRootActionUpperBound") is None:
            proof_source["winningRootActionUpperBound"] = len(legal_actions)
    return proof_source


def _scenario_project_projection(scenario: Mapping[str, Any]) -> dict[str, list[str]]:
    predicates: list[Mapping[str, Any]] = []

    def collect(node: Any) -> None:
        if not isinstance(node, Mapping):
            return
        if "predicateId" in node and "kind" in node:
            predicates.append(node)
            return
        for child in node.get("children") or ():
            collect(child)

    collect(scenario.get("constraints"))
    collect(scenario.get("outcome"))
    projected = {"projectKinds": set(), "projectValues": set(), "projectStates": set()}
    for predicate in predicates:
        if not str(predicate.get("kind") or "").startswith("project."):
            continue
        arguments = dict(predicate.get("arguments") or {})
        for key, field in (
            ("kind", "projectKinds"),
            ("value", "projectValues"),
            ("state", "projectStates"),
        ):
            if arguments.get(key) is not None:
                projected[field].add(str(arguments[key]))
    return {key: sorted(values) for key, values in projected.items()}


def _scenario_tactic_requests(scenario: Mapping[str, Any]) -> dict[str, str]:
    allowed = {item.value for item in TacticalKind}
    predicate_detectors: dict[str, str] = {}

    def collect(node: Any) -> None:
        if not isinstance(node, Mapping):
            return
        if str(node.get("kind") or "").startswith("tactic."):
            detector = str(dict(node.get("arguments") or {}).get("kind") or "")
            if detector not in allowed:
                raise ValueError("scenario tactical detector is not registered")
            predicate_detectors[str(node.get("predicateId") or "")] = detector
            return
        for child in node.get("children") or ():
            collect(child)

    collect(scenario.get("constraints"))
    collect(scenario.get("outcome"))
    return predicate_detectors


def _native_tactic_projection(
    scenario: Mapping[str, Any],
    predicate_evidence: list[Mapping[str, Any]],
    *,
    proof_outcome: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    allowed = {item.value for item in TacticalKind}
    predicate_detectors = _scenario_tactic_requests(scenario)
    requested = set(predicate_detectors.values())
    verified: set[str] = set()
    for raw in predicate_evidence:
        row = copy.deepcopy(dict(raw))
        if row.get("kind") != "tactic.verified":
            continue
        declared_hash = row.pop("sha256", None)
        if declared_hash != canonical_sha256(row):
            raise ValueError("predicate tactical evidence checksum mismatch")
        same_solve = row.get("evidenceExtractor") == "cp_sat.native_proof_same_solve_v1"
        if not same_solve and row.get("satisfied") is not True:
            continue
        observed = dict(row.get("observed") or {})
        if same_solve:
            inner = dict(observed)
            inner_hash = inner.pop("sha256", None)
            if inner_hash != canonical_sha256(inner):
                raise ValueError("predicate tactical evidence inner checksum mismatch")
            if (type(row.get("schemaVersion")) is not int
                    or type(row.get("satisfied")) is not bool
                    or type(inner.get("schemaVersion")) is not int or inner["schemaVersion"] != 1
                    or type(inner.get("satisfied")) is not bool
                    or any(inner.get(key) != row.get(key)
                           for key in ("schemaVersion", "predicateId", "kind", "satisfied"))
                    or inner.get("evidenceExtractor") != PROOF_TACTIC_PREDICATE_REGISTRATIONS_V2["tactic.verified"]):
                raise ValueError("predicate tactical evidence inner ownership mismatch")
            owned = (proof_outcome or {}).get("nativeQualification")
            if not isinstance(owned, Mapping) or owned.get(row.get("predicateId")) != observed:
                raise ValueError("predicate tactical evidence does not match the proof")
            observed = dict(inner.get("observed") or {})
        detector = str(observed.get("detectorId") or "")
        if detector not in allowed:
            raise ValueError("predicate tactical evidence detector is not registered")
        if detector != predicate_detectors.get(row.get("predicateId")):
            raise ValueError("predicate tactical evidence does not match the scenario")
        if (row.get("satisfied") is True
                and observed.get("applicable") is True and observed.get("verified") is True):
            verified.add(detector)
    return sorted(requested), sorted(verified)


def build_ortools_research_result_v1(
    *,
    run: Mapping[str, Any],
    recipe: Mapping[str, Any],
    executor_result: Mapping[str, Any] | Any,
    artifact_documents: Mapping[str, Mapping[str, Any]],
    artifact_references: Mapping[str, Mapping[str, Any]],
    runtime_compatibility_check: Callable[[str, str], Any],
    created_at: str,
) -> dict[str, Any]:
    """Persist one qualified exact CP-SAT result without legacy qualification calls."""

    run_value = copy.deepcopy(dict(run))
    recipe_value = copy.deepcopy(dict(recipe))
    executor = _as_document(executor_result, "executor result")
    if executor.get("status") != "qualified":
        raise ValueError("only a qualified OR-Tools result can be retained")
    candidate = dict(executor.get("candidate") or {})
    predicate = dict(executor.get("predicateEvaluation") or {})
    proof = dict(executor.get("proof") or {})
    if not candidate or not predicate or not proof:
        raise ValueError("qualified OR-Tools result is incomplete")
    exact_state_id = _require_text(
        candidate.get("exactStateId"), "candidate exactStateId"
    )
    if re.fullmatch(r"STATE-[A-F0-9]{20}", exact_state_id) is None:
        raise ValueError("candidate exactStateId is not canonical")
    authority = ResearchAuthorityV1.from_dict(
        dict(executor.get("researchAuthority") or {})
    ).to_dict()
    configuration = dict(recipe_value.get("configuration") or {})
    if run_value.get("recipeCode") != recipe_value.get("code") or run_value.get(
        "recipeHash"
    ) != recipe_value.get("sha256"):
        raise ValueError("run recipe authority does not match recipe")
    if run_value.get("configurationAuthority") != configuration:
        raise ValueError("run configuration authority does not match recipe")
    if authority["bundleRevisionId"] != configuration.get(
        "bundleRevisionId"
    ) or authority["bundleHash"] != configuration.get("semanticBundleHash"):
        raise ValueError("executor bundle authority does not match recipe")
    executor_semantic_hash = _require_hash(
        executor.get("semanticHash"), "executor semantic hash"
    )
    owned_artifacts = _bind_semantic_artifacts(
        documents=artifact_documents,
        references=artifact_references,
        authority=authority,
        owner_hash=executor_semantic_hash,
    )
    proof_preimage = copy.deepcopy(proof)
    proof_preimage.pop("sha256", None)
    if dict(artifact_documents["proofOutcome"]) != proof_preimage:
        raise ValueError("proofOutcome artifact does not match executor proof")
    if dict(artifact_documents["completeTrace"]) != dict(candidate.get("trace") or {}):
        raise ValueError("completeTrace artifact does not match executor candidate")
    scenario = dict(recipe_value.get("scenario") or {})
    state = dict(candidate.get("state") or {})
    exact_state_hash = canonical_sha256(state)
    proof_request = dict(recipe_value.get("proof") or {})
    proof_receipt = _proof_receipt(
        raw_proof=proof,
        proof_request=proof_request,
        configuration=configuration,
        authority=authority,
    )
    exact_normalization = dict(candidate.get("trace") or {}).get("exactNormalization")
    actor = str(
        (
            exact_normalization.get("nextActor")
            if isinstance(exact_normalization, Mapping)
            else None
        )
        or state.get("currentPlayer")
        or state.get("nextActor")
        or ""
    )
    if actor not in {"north", "west", "south", "east"}:
        raise ValueError("qualified candidate does not identify its acting seat")
    predicate_evidence = copy.deepcopy(list(predicate.get("evidence") or ()))
    requested_tactics, verified_tactics = _native_tactic_projection(
        scenario, predicate_evidence, proof_outcome=proof
    )
    satisfaction_core = {
        "schemaVersion": 1,
        "scenarioHash": _require_hash(scenario.get("sha256"), "scenario hash"),
        "exactStateId": exact_state_id,
        "status": "satisfied",
        "predicates": predicate_evidence,
        "qualificationSource": CP_SAT_QUALIFICATION_AUTHORITY,
        "predicateEvaluationHash": _require_hash(
            predicate.get("sha256"), "predicate evaluation hash"
        ),
    }
    satisfaction = {
        **satisfaction_core,
        "sha256": canonical_sha256(satisfaction_core),
    }
    requested_mode = str(proof_request.get("mode") or "")
    completion = str(proof_receipt["completionStatus"])
    achieved_mode = str(proof_receipt["achievedMode"])
    project_projection = _scenario_project_projection(scenario)
    decision_point = dict(scenario.get("decisionPoint") or {})
    trace_document = dict(artifact_documents["completeTrace"])
    exact_state = trace_document.get("exactNormalization")
    if (
        not isinstance(exact_state, Mapping)
        or dict(exact_state.get("state") or {}) != state
    ):
        raise ValueError(
            "completeTrace exact normalization does not match candidate state"
        )
    public_projection_hash = build_observer_safe_exact_state_v1(
        exact_state=exact_state,
        scenario_projection={
            "exactStateId": exact_state_id,
            "decisionKind": decision_point.get("kind"),
        },
        observer_seat=actor,
    )["publicProjectionHash"]
    complete_exact_state = trace_document.get("completeExactNormalization")
    terminal_state = (
        dict(complete_exact_state.get("state") or {})
        if isinstance(complete_exact_state, Mapping)
        else {}
    )
    projected_state = terminal_state or state
    round_score = dict(projected_state.get("roundScore") or {})
    terminal_score = dict(trace_document.get("score") or {})
    round_winner = round_score.get("winner") or terminal_score.get("roundWinner")
    declarer = str(
        projected_state.get("declarer")
        or projected_state.get("buyer")
        or trace_document.get("buyer")
        or ""
    )
    declarer_team = (
        "north_south"
        if declarer in {"north", "south"}
        else "west_east"
        if declarer in {"west", "east"}
        else None
    )
    contract_outcome = (
        "won"
        if round_winner == declarer_team
        else "lost"
        if declarer_team is not None and round_winner is not None
        else None
    )
    scenario_projection_core = {
        "schemaVersion": 1,
        "scenarioTemplateId": f"SCN-{scenario['sha256'][:12].upper()}",
        "origin": candidate.get("origin"),
        "status": "satisfied",
        "decisionKind": decision_point.get("kind"),
        "targetTeam": proof_receipt["targetTeam"],
        "requestedProofMode": requested_mode,
        "achievedProofMode": achieved_mode,
        "proofCompletion": completion,
        "proofVisibility": proof_receipt["achievedVisibility"],
        "winningRootActionCount": proof_receipt["winningRootActionCount"],
        "winningRootActionLowerBound": proof_receipt.get(
            "winningRootActionLowerBound",
            proof_receipt["winningRootActionCount"],
        ),
        "winningRootActionUpperBound": proof_receipt.get(
            "winningRootActionUpperBound",
            proof_receipt["winningRootActionCount"],
        ),
        "contractOutcome": contract_outcome,
        "scenarioCategories": _scenario_categories_v2(recipe_value),
        **project_projection,
        "exactStateId": exact_state_id,
        "exactStateAvailable": True,
        "publicProjectionHash": public_projection_hash,
    }
    scenario_projection = {
        **scenario_projection_core,
        "sha256": canonical_sha256(scenario_projection_core),
    }
    qualification_core = {
        "schemaVersion": 1,
        "authority": CP_SAT_QUALIFICATION_AUTHORITY,
        "status": "qualified",
        "qualified": True,
        "failures": [],
        "observerSafe": True,
        "decisionPointAuditComplete": True,
        "legacyObserverSafetyUnverified": False,
        "runtimeCompatibilityDoesNotQualify": True,
    }
    qualification = {
        **qualification_core,
        "sha256": canonical_sha256(qualification_core),
    }
    truth_label = (
        "exact_optimal" if completion == "exact_complete" else "robust_best_known"
    )
    decision_count = 1 if list(proof_receipt.get("rootActions") or ()) else 0
    verdict = {
        "truthLabel": truth_label,
        "confidence": 1.0,
        "score": 100.0,
        "disposition": "qualified",
        "classification": "scenario_v4",
        "criticalDecisionIds": [exact_state_id],
        "interestingSeats": [actor],
        "tacticalTags": requested_tactics,
        "verifiedTacticalTags": verified_tactics,
        "decisionCount": decision_count,
    }
    base_identity = {
        "runCode": _require_text(run_value.get("runCode"), "run code"),
        "runHash": _require_hash(run_value.get("runHash"), "run hash"),
        "recipeCode": _require_text(recipe_value.get("code"), "recipe code"),
        "recipeHash": _require_hash(recipe_value.get("sha256"), "recipe hash"),
        "configurationAuthorityHash": canonical_sha256(configuration),
        "scenarioHash": _require_hash(scenario.get("sha256"), "scenario hash"),
        "exactStateHash": exact_state_hash,
        "exactStateId": exact_state_id,
        "scenarioProjectionHash": scenario_projection["sha256"],
        "qualificationHash": qualification["sha256"],
        "verdictHash": canonical_sha256(verdict),
        "proofOutcomeHash": authority["proofOutcomeHash"],
        "artifactHashes": {
            name: reference["sha256"]
            for name, reference in sorted(owned_artifacts.items())
        },
    }
    identity = build_ortools_result_identity_v2(
        base_identity,
        authority,
        domain_authority_hash=_configuration_domain_hash(configuration),
        executor_semantic_hash=executor_semantic_hash,
        candidate_hash=_require_hash(candidate.get("sha256"), "candidate hash"),
        predicate_evaluation_hash=_require_hash(
            predicate.get("sha256"), "predicate evaluation hash"
        ),
    )
    result_hash = canonical_sha256(identity)
    try:
        check = runtime_compatibility_check(result_hash, authority["traceHash"])
    except Exception as exc:
        compatibility, compatibility_evidence = _quarantined_runtime_check(
            semantic_result_hash=result_hash,
            bundle_hash=authority["bundleHash"],
            trace_hash=authority["traceHash"],
            failure_code="runtime_compatibility_callback_failed",
            failure=exc,
        )
    else:
        try:
            compatibility, compatibility_evidence = _runtime_check_documents(check)
            if (
                compatibility["bundleHash"] != authority["bundleHash"]
                or compatibility["traceHash"] != authority["traceHash"]
            ):
                raise ValueError("runtime compatibility authority mismatch")
            _validate_runtime_evidence(
                compatibility=compatibility,
                evidence=compatibility_evidence,
                semantic_result_hash=result_hash,
            )
        except Exception as exc:
            compatibility, compatibility_evidence = _quarantined_runtime_check(
                semantic_result_hash=result_hash,
                bundle_hash=authority["bundleHash"],
                trace_hash=authority["traceHash"],
                failure_code="runtime_compatibility_callback_malformed",
                failure=exc,
            )
    executor_receipt = dict(executor.get("executionReceipt") or {})
    receipt = _execution_receipt(
        executor_receipt, semantic_result_hash=result_hash, candidate=candidate
    )
    timestamp = _require_text(created_at, "createdAt")
    if timestamp != run_value.get("createdAt"):
        raise ValueError("result createdAt must match its run")
    library_projection = {
        "schemaVersion": 1,
        "actingSeats": [actor],
        "observerSeats": [actor],
        "contract": projected_state.get("contract") or trace_document.get("contract"),
        "purchaseRound": projected_state.get("purchaseRound"),
        "trumpSuit": projected_state.get("trumpSuit")
        or trace_document.get("trumpSuit"),
        "escalation": projected_state.get("escalation") or "normal",
        "projectKinds": project_projection["projectKinds"],
        "generationLane": (
            "exact_input"
            if candidate.get("origin") == "exact_trace_prefix"
            else "ortools"
        ),
    }
    result = {
        "schemaVersion": 2,
        "resultCode": f"LAF-{result_hash[:10].upper()}",
        "resultHash": result_hash,
        "runCode": run_value["runCode"],
        "recipeCode": recipe_value["code"],
        "traceHash": authority["traceHash"],
        "createdAt": timestamp,
        "predicateEvaluation": predicate,
        "scenarioSatisfaction": satisfaction,
        "proofOutcome": proof_receipt,
        "scenarioProjection": scenario_projection,
        "libraryProjection": library_projection,
        "artifacts": owned_artifacts,
        "researchAuthority": authority,
        "runtimeCompatibility": compatibility,
        "runtimeCompatibilityEvidence": compatibility_evidence,
        "executionReceipt": receipt,
        **verdict,
        "qualification": qualification,
        "provenance": {
            "recipeHash": recipe_value["sha256"],
            "scenarioHash": scenario["sha256"],
            "exactStateId": exact_state_id,
            "configurationAuthorityHash": canonical_sha256(configuration),
            "executorSemanticHash": executor_semantic_hash,
            "resultIdentity": identity,
        },
    }
    return validate_ortools_result_sections_v1(result, recipe=recipe_value)


def _authority_artifact_reference_v1(
    reference: Mapping[str, Any],
    *,
    name: str,
    digest: str,
    owner_hash: str,
) -> dict[str, Any]:
    raw = _closed_document(
        reference, required=_CAS_REFERENCE_FIELDS, label=f"{name} authority artifact"
    )
    if raw["sha256"] != digest:
        raise ValueError(f"{name} authority artifact CAS hash mismatch")
    if raw["mediaType"] != ORTOOLS_AUTHORITY_ARTIFACT_MEDIA_TYPES[name]:
        raise ValueError(f"{name} authority artifact media type mismatch")
    if type(raw["bytes"]) is not int or raw["bytes"] < 1:
        raise ValueError(f"{name} authority artifact byte count is invalid")
    _validate_cas_uri(raw["uri"], digest, f"{name} authority artifact")
    return {
        **raw,
        "ownerKind": _AUTHORITY_ARTIFACT_OWNER_KIND,
        "ownerHash": owner_hash,
    }


def build_ortools_research_result_v3(
    *,
    run: Mapping[str, Any],
    recipe: Mapping[str, Any],
    executor_result: Mapping[str, Any] | Any,
    campaign_semantic_hash: str,
    terminal_status: str,
    artifact_documents: Mapping[str, Mapping[str, Any]],
    artifact_references: Mapping[str, Mapping[str, Any]],
    configuration_snapshot: Mapping[str, Any],
    lowering_manifest: Mapping[str, Any],
    authority_artifact_writer: Callable[[bytes, str, str], Mapping[str, Any]],
    runtime_compatibility_check: Callable[[str, str], Any] | None,
    created_at: str,
) -> dict[str, Any]:
    """Build external results with independent, fresh quality verification."""
    return _build_ortools_research_result_v3(
        run=run, recipe=recipe, executor_result=executor_result,
        campaign_semantic_hash=campaign_semantic_hash, terminal_status=terminal_status,
        artifact_documents=artifact_documents, artifact_references=artifact_references,
        configuration_snapshot=configuration_snapshot, lowering_manifest=lowering_manifest,
        authority_artifact_writer=authority_artifact_writer,
        runtime_compatibility_check=runtime_compatibility_check, created_at=created_at,
    )


def _build_ortools_research_result_v3(
    *,
    run: Mapping[str, Any],
    recipe: Mapping[str, Any],
    executor_result: Mapping[str, Any] | Any,
    campaign_semantic_hash: str,
    terminal_status: str,
    artifact_documents: Mapping[str, Mapping[str, Any]],
    artifact_references: Mapping[str, Mapping[str, Any]],
    configuration_snapshot: Mapping[str, Any],
    lowering_manifest: Mapping[str, Any],
    authority_artifact_writer: Callable[[bytes, str, str], Mapping[str, Any]],
    runtime_compatibility_check: Callable[[str, str], Any] | None,
    created_at: str,
    verified_artifacts: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Build one immutable V3 result for success or a terminal audit outcome."""

    if terminal_status not in _TERMINAL_STATUSES:
        raise ValueError("OR-Tools terminal result status is unsupported")
    run_value = copy.deepcopy(dict(run))
    recipe_value = copy.deepcopy(dict(recipe))
    executor = _as_document(executor_result, "executor result")
    if (
        terminal_status == "satisfied"
        and dict(recipe_value.get("qualification") or {}).get("requireAllDecisionAudit")
        is True
    ):
        quality_issue = native_full_round_quality_issue_v1(
            str(dict(executor.get("researchAuthority") or {}).get("backendId") or ""),
            str(dict(recipe_value.get("playingQuality") or {}).get("mode") or "") or None,
        )
        if (
            recipe_value.get("playingQuality") is None
            or "playingQualityEvidence" not in artifact_documents.get("completeTrace", {})
        ):
            raise ValueError(
                quality_issue["code"]
                if quality_issue is not None
                else "all_decision_quality_evidence_unavailable"
            )
    candidate = dict(executor.get("candidate") or {}) or None
    predicate = dict(executor.get("predicateEvaluation") or {}) or None
    raw_proof = dict(executor.get("proof") or {}) or None
    if candidate is not None:
        candidate_core = copy.deepcopy(candidate)
        candidate_hash = _require_hash(
            candidate_core.pop("sha256", None), "candidate sha256"
        )
        if canonical_sha256(candidate_core) != candidate_hash:
            raise ValueError("candidate integrity mismatch")
    predicate = _validate_predicate_evaluation_v1(
        predicate,
        expected_hash=(predicate.get("sha256") if predicate is not None else None),
        scenario=recipe_value["scenario"],
    )
    if raw_proof is not None:
        proof_core = copy.deepcopy(raw_proof)
        proof_hash = proof_core.pop("sha256", None)
        if proof_hash is not None and (
            _require_hash(proof_hash, "proof sha256") != canonical_sha256(proof_core)
        ):
            raise ValueError("proof integrity mismatch")
    authority = ResearchAuthorityV1.from_dict(
        dict(executor.get("researchAuthority") or {})
    ).to_dict()
    configuration = dict(recipe_value.get("configuration") or {})
    scenario = dict(recipe_value.get("scenario") or {})
    proof_request = dict(recipe_value.get("proof") or {})
    if (
        run_value.get("recipeCode") != recipe_value.get("code")
        or run_value.get("recipeHash") != recipe_value.get("sha256")
        or run_value.get("configurationAuthority") != configuration
    ):
        raise ValueError("run, recipe, and configuration authority do not match")
    if authority["bundleRevisionId"] != configuration.get(
        "bundleRevisionId"
    ) or authority["bundleHash"] != configuration.get("semanticBundleHash"):
        raise ValueError("executor bundle authority does not match recipe")
    executor_semantic_hash = _require_hash(
        executor.get("semanticHash"), "executor semantic hash"
    )
    owned_artifacts = _bind_semantic_artifacts(
        documents=artifact_documents,
        references=artifact_references,
        authority=authority,
        owner_hash=executor_semantic_hash,
    )

    snapshot = copy.deepcopy(dict(configuration_snapshot))
    snapshot_intrinsic = _require_hash(
        snapshot.get("sha256"), "configuration snapshot sha256"
    )
    if (
        canonical_sha256({k: v for k, v in snapshot.items() if k != "sha256"})
        != snapshot_intrinsic
        or snapshot.get("bundleHash") != authority["bundleHash"]
    ):
        raise ValueError("configuration snapshot authority mismatch")
    _validate_native_tactical_profile_v1(
        predicate, artifact_documents["proofOutcome"], snapshot,
        bundle_hash=authority["bundleHash"], snapshot_hash=snapshot_intrinsic,
        proof_graph=artifact_documents["proofGraph"],
        proof_request=recipe_value.get("proof"),
        proof_model=artifact_documents["proofModel"],
        backend_version=authority["backendVersion"], scenario=recipe_value["scenario"],
    )
    quality_policy = None
    quality_evidence_hash = None
    trace_quality = dict(artifact_documents["completeTrace"].get("playingQualityEvidence") or {})
    quality_complete = dict(trace_quality.get("receipt") or {}).get("status") == "complete"
    if recipe_value.get("playingQuality") is not None and (
        terminal_status == "satisfied" or quality_complete
    ):
        settings = snapshot["components"]["challenge"]["payload"]["scenarioResearchV4"]
        if settings.get("backend") != authority["backendId"]:
            raise ValueError("quality configuration backend mismatch")
        quality_policy = ScenarioPlayingQualityPolicyV1.build(
            recipe_value["playingQuality"]
        )
        validator = (_validate_trace_quality_evidence_v1
            if verified_artifacts else validate_trace_quality_evidence_v1)
        validator(
            artifact_documents["completeTrace"],
            require_family_opening=True,
            rules_hash=_require_hash(snapshot["components"]["rules"]["sha256"], "quality rules hash"),
            required_policy=quality_policy,
            required_proof_profiles=settings["proofBudgets"]["profiles"],
            required_search_profiles=settings.get("searchBudgets", {}).get("profiles"),
            required_execution=ScenarioQualityExecutionBudgetV1.build(settings["qualityExecution"]),
            rules=RuleBook(snapshot["components"]["rules"]["payload"]),
            **({"verified_artifacts": verified_artifacts} if verified_artifacts else {}),
        )
        quality_evidence_hash = _require_hash(
            artifact_documents["completeTrace"]["playingQualityEvidence"][
                "receipt"
            ]["sha256"],
            "playing quality evidence hash",
        )
    manifest = copy.deepcopy(dict(lowering_manifest))
    manifest_intrinsic = _require_hash(
        manifest.get("sha256"), "lowering manifest sha256"
    )
    if (
        canonical_sha256({k: v for k, v in manifest.items() if k != "sha256"})
        != manifest_intrinsic
        or manifest_intrinsic != authority["loweringManifestHash"]
    ):
        raise ValueError("lowering manifest authority mismatch")

    def store_authority(
        name: str, document: Mapping[str, Any]
    ) -> tuple[dict[str, Any], str]:
        payload = json.dumps(
            dict(document), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = canonical_sha256(dict(document))
        reference = authority_artifact_writer(
            payload,
            ORTOOLS_AUTHORITY_ARTIFACT_MEDIA_TYPES[name],
            f"{digest}.json",
        )
        return dict(reference), digest

    snapshot_reference, snapshot_cas_hash = store_authority(
        "configurationSnapshot", snapshot
    )
    manifest_reference, manifest_cas_hash = store_authority(
        "loweringManifest", manifest
    )

    candidate_exact_state_id = candidate.get("exactStateId") if candidate else None
    state = dict(candidate.get("state") or {}) if candidate else {}
    trace_document = dict(artifact_documents["completeTrace"])
    exact_normalization = trace_document.get("exactNormalization")
    replay_valid = (
        candidate is not None
        and isinstance(exact_normalization, Mapping)
        and candidate_exact_state_id is not None
    )
    exact_state_id = candidate_exact_state_id if replay_valid else None
    strict_candidate_identity = (
        artifact_documents["constructionModel"].get("kind")
        == "ortools_research_compilation_receipt_v3"
    )
    if strict_candidate_identity and candidate_exact_state_id is not None and not replay_valid:
        raise ValueError("exact-state identity requires an exact normalization")
    if replay_valid and (strict_candidate_identity or terminal_status == "satisfied"):
        if re.fullmatch(r"STATE-[A-F0-9]{20}", str(exact_state_id)) is None:
            raise ValueError("V3 result exactStateId is not canonical")
        if canonical_sha256(dict(candidate.get("trace") or {})) != canonical_sha256(
            trace_document
        ):
            raise ValueError("completeTrace artifact does not match V3 candidate")
        if dict(exact_normalization.get("state") or {}) != state:
            raise ValueError("completeTrace exact state does not match V3 candidate")
    if terminal_status == "satisfied":
        if (
            executor.get("status") != "qualified"
            or candidate is None
            or predicate is None
            or raw_proof is None
            or not replay_valid
            or predicate.get("status") != "satisfied"
            or predicate.get("satisfied") is not True
            or raw_proof.get("completionStatus")
            not in {"exact_complete", "bounded_complete", "policy_complete"}
            or raw_proof.get("claimSatisfied") is not True
        ):
            raise ValueError(
                "satisfied V3 result requires a candidate, satisfied predicate, "
                "exact state, and complete true proof"
            )
    if raw_proof is not None or "completionStatus" in artifact_documents["proofOutcome"]:
        proof_preimage = copy.deepcopy(raw_proof or {})
        proof_preimage.pop("sha256", None)
        if proof_preimage != dict(artifact_documents["proofOutcome"]):
            raise ValueError("proofOutcome artifact does not match V3 proof")
    actor = str(
        (exact_normalization.get("nextActor") if replay_valid else None)
        or state.get("currentPlayer")
        or state.get("nextActor")
        or ""
    )
    if actor not in {"north", "west", "south", "east"}:
        actor = ""

    proof_source = _terminal_proof_source_v3(
        raw_proof=raw_proof, terminal_status=terminal_status,
        proof_request=proof_request,
        exact_normalization=exact_normalization if replay_valid else None,
        backend_version=authority.get("backendVersion"),
    )
    proof_receipt = _proof_receipt(
        raw_proof=proof_source,
        proof_request=proof_request,
        configuration=configuration,
        authority=authority,
    )
    predicate_hash = predicate.get("sha256") if predicate else None
    predicate_evidence = copy.deepcopy(list((predicate or {}).get("evidence") or ()))
    requested_tactics, verified_tactics = _native_tactic_projection(
        scenario, predicate_evidence, proof_outcome=raw_proof
    )
    satisfaction_core = {
        "schemaVersion": 1,
        "scenarioHash": _require_hash(scenario.get("sha256"), "scenario hash"),
        "exactStateId": exact_state_id,
        "status": terminal_status,
        "predicates": predicate_evidence,
        "qualificationSource": CP_SAT_QUALIFICATION_AUTHORITY,
        "predicateEvaluationHash": predicate_hash,
    }
    satisfaction = {**satisfaction_core, "sha256": canonical_sha256(satisfaction_core)}
    project_projection = _scenario_project_projection(scenario)
    decision_point = dict(scenario.get("decisionPoint") or {})
    public_projection_hash = (
        build_observer_safe_exact_state_v1(
            exact_state=exact_normalization,
            scenario_projection={
                "exactStateId": exact_state_id,
                "decisionKind": decision_point.get("kind"),
            },
            observer_seat=actor,
        )["publicProjectionHash"]
        if replay_valid and actor
        else canonical_sha256(
            {
                "schemaVersion": 1,
                "scenarioHash": scenario["sha256"],
                "terminalStatus": terminal_status,
                "exactStateAvailable": False,
            }
        )
    )
    requested_mode = str(proof_request.get("mode") or "")
    completion = str(proof_receipt["completionStatus"])
    achieved_mode = str(proof_receipt["achievedMode"])
    complete_exact_state = trace_document.get("completeExactNormalization")
    terminal_state = (
        dict(complete_exact_state.get("state") or {})
        if isinstance(complete_exact_state, Mapping)
        else {}
    )
    score_state = terminal_state or state
    round_score = dict(score_state.get("roundScore") or {})
    terminal_score = dict(trace_document.get("score") or {})
    round_winner = round_score.get("winner") or terminal_score.get("roundWinner")
    declarer = str(
        score_state.get("declarer")
        or score_state.get("buyer")
        or trace_document.get("buyer")
        or ""
    )
    declarer_team = (
        "north_south"
        if declarer in {"north", "south"}
        else "west_east"
        if declarer in {"west", "east"}
        else None
    )
    contract_outcome = (
        "won"
        if round_winner == declarer_team
        else "lost"
        if declarer_team is not None and round_winner is not None
        else None
    )
    scenario_projection_core = {
        "schemaVersion": 1,
        "scenarioTemplateId": f"SCN-{scenario['sha256'][:12].upper()}",
        "origin": (
            candidate.get("origin")
            if candidate
            else dict(scenario.get("origin") or {}).get("kind")
        ),
        "status": terminal_status,
        "decisionKind": decision_point.get("kind"),
        "targetTeam": proof_receipt["targetTeam"],
        "requestedProofMode": requested_mode,
        "achievedProofMode": achieved_mode,
        "proofCompletion": completion,
        "proofVisibility": proof_receipt["achievedVisibility"],
        "winningRootActionCount": proof_receipt["winningRootActionCount"],
        "winningRootActionLowerBound": proof_receipt.get(
            "winningRootActionLowerBound",
            proof_receipt["winningRootActionCount"],
        ),
        "winningRootActionUpperBound": proof_receipt.get(
            "winningRootActionUpperBound",
            proof_receipt["winningRootActionCount"],
        ),
        "contractOutcome": contract_outcome,
        "scenarioCategories": _scenario_categories_v2(recipe_value),
        **project_projection,
        "exactStateId": exact_state_id,
        "exactStateAvailable": bool(replay_valid),
        "publicProjectionHash": public_projection_hash,
    }
    if 'lessonTags' in trace_document:
        if quality_evidence_hash is None:
            raise ValueError('lesson tags require completed quality evidence')
        scenario_projection_core['lessonTags'] = validated_trace_lesson_tags_v1(
            trace_document, rules=RuleBook(snapshot['components']['rules']['payload']))
    scenario_projection = {
        **scenario_projection_core,
        "sha256": canonical_sha256(scenario_projection_core),
    }
    qualified = terminal_status == "satisfied"
    validate_generation_result_binding_v1(recipe_value, trace_document, snapshot, qualified=qualified)
    counterstrategy = dict(dict(artifact_documents.get("proofModel") or {}).get("certificate") or {})
    counterstrategy_refuted = (
        terminal_status == "proof_failed"
        and dict(artifact_documents.get("proofModel") or {}).get("kind") == "native_counterstrategy_models"
        and counterstrategy.get("status") == "certified"
    )
    interest_evidence = _native_interest_evidence_v1(snapshot, trace_document,
        verified_quality=quality_evidence_hash is not None, featured_state_id=exact_state_id,
        research_authority=authority)
    interest_admission = (interest_admission_summary_v1(interest_evidence)
                          if interest_evidence is not None else None)
    interest_reference = None
    if interest_evidence is not None:
        interest_reference, interest_cas_hash = store_authority("interestAdmission", interest_evidence)
    qualification_core = {
        "schemaVersion": 1,
        "authority": CP_SAT_QUALIFICATION_AUTHORITY,
        "status": "qualified" if qualified else "rejected",
        "qualified": qualified,
        "failures": [] if qualified else [terminal_status, *(
            ["native_round_win_counterstrategy"] if counterstrategy_refuted else [])],
        "observerSafe": True,
        "decisionPointAuditComplete": bool(replay_valid and predicate is not None),
        "allDecisionPolicyAuditComplete": quality_evidence_hash is not None,
        "legacyObserverSafetyUnverified": False,
        "runtimeCompatibilityDoesNotQualify": True,
        **({"interestAdmission": interest_admission} if interest_admission is not None else {}),
        **(
            {
                "playingQualityMode": quality_policy.mode,
                "playingQualityEvidenceHash": quality_evidence_hash,
                **({"playingQualityVisibility": quality_policy.visibility,
                    "actionSelectionObserverSafe": quality_policy.visibility == "actor_information_set"}
                   if quality_policy.schema_version == 2 else {}),
            }
            if quality_policy is not None and quality_evidence_hash is not None
            else {}
        ),
    }
    qualification = {
        **qualification_core,
        "sha256": canonical_sha256(qualification_core),
    }
    verdict = {
        "truthLabel": (
            "exact_optimal"
            if qualified and completion == "exact_complete"
            else "robust_best_known"
            if qualified
            else "rejected"
        ),
        "confidence": 1.0 if qualified else 0.0,
        "score": (min(100.0, float(interest_admission["score"])) if qualified and interest_admission is not None
                  else 100.0 if qualified else 0.0),
        "disposition": "qualified" if qualified else "rejected",
        "classification": "scenario_v4",
        "criticalDecisionIds": [exact_state_id] if exact_state_id else [],
        "interestingSeats": [actor] if actor else [],
        "tacticalTags": requested_tactics,
        "verifiedTacticalTags": verified_tactics,
        "decisionCount": 1 if replay_valid else 0,
    }
    base_identity = {
        "runCode": _require_text(run_value.get("runCode"), "run code"),
        "runHash": _require_hash(run_value.get("runHash"), "run hash"),
        "recipeCode": _require_text(recipe_value.get("code"), "recipe code"),
        "recipeHash": _require_hash(recipe_value.get("sha256"), "recipe hash"),
        "configurationAuthorityHash": canonical_sha256(configuration),
        "scenarioHash": scenario["sha256"],
        "exactStateHash": canonical_sha256(state) if replay_valid else None,
        "exactStateId": exact_state_id if replay_valid else None,
        "scenarioSatisfactionHash": satisfaction["sha256"],
        "scenarioProjectionHash": scenario_projection["sha256"],
        "qualificationHash": qualification["sha256"],
        "verdictHash": canonical_sha256(verdict),
        "proofOutcomeReceiptHash": proof_receipt["sha256"],
        "artifactHashes": {
            name: reference["sha256"]
            for name, reference in sorted(owned_artifacts.items())
        },
    }
    identity = build_ortools_result_identity_v3(
        base_identity,
        authority,
        terminal_status=terminal_status,
        configuration_snapshot_hash=snapshot_intrinsic,
        authority_artifact_hashes={
            "configurationSnapshot": snapshot_cas_hash,
            "loweringManifest": manifest_cas_hash,
            **({"interestAdmission": interest_cas_hash} if interest_reference is not None else {}),
        },
        domain_authority_hash=_configuration_domain_hash(configuration),
        executor_semantic_hash=executor_semantic_hash,
        campaign_semantic_hash=campaign_semantic_hash,
        candidate_hash=candidate.get("sha256") if candidate else None,
        predicate_evaluation_hash=predicate_hash,
    )
    result_hash = canonical_sha256(identity)
    identity_reference_raw, identity_cas_hash = store_authority(
        "resultIdentity", identity
    )
    if identity_cas_hash != result_hash:
        raise AssertionError("result identity CAS digest drifted")
    receipt = _execution_receipt(
        dict(executor.get("executionReceipt") or {}),
        semantic_result_hash=result_hash,
        candidate=candidate,
    )
    receipt_reference_raw, receipt_cas_hash = store_authority(
        "executionReceipt", receipt
    )
    authority_artifacts = {
        **({"interestAdmission": _authority_artifact_reference_v1(
            interest_reference, name="interestAdmission", digest=interest_cas_hash,
            owner_hash=result_hash)} if interest_reference is not None else {}),
        "configurationSnapshot": _authority_artifact_reference_v1(
            snapshot_reference,
            name="configurationSnapshot",
            digest=snapshot_cas_hash,
            owner_hash=result_hash,
        ),
        "loweringManifest": _authority_artifact_reference_v1(
            manifest_reference,
            name="loweringManifest",
            digest=manifest_cas_hash,
            owner_hash=result_hash,
        ),
        "resultIdentity": _authority_artifact_reference_v1(
            identity_reference_raw,
            name="resultIdentity",
            digest=result_hash,
            owner_hash=result_hash,
        ),
        "executionReceipt": _authority_artifact_reference_v1(
            receipt_reference_raw,
            name="executionReceipt",
            digest=receipt_cas_hash,
            owner_hash=result_hash,
        ),
    }

    if replay_valid and runtime_compatibility_check is not None:
        try:
            check = runtime_compatibility_check(result_hash, authority["traceHash"])
            compatibility, compatibility_evidence = _runtime_check_documents(check)
            if (
                compatibility["bundleHash"] != authority["bundleHash"]
                or compatibility["traceHash"] != authority["traceHash"]
            ):
                raise ValueError("runtime compatibility authority mismatch")
            _validate_runtime_evidence(
                compatibility=compatibility,
                evidence=compatibility_evidence,
                semantic_result_hash=result_hash,
            )
        except Exception as exc:
            compatibility, compatibility_evidence = _quarantined_runtime_check(
                semantic_result_hash=result_hash,
                bundle_hash=authority["bundleHash"],
                trace_hash=authority["traceHash"],
                failure_code="runtime_compatibility_callback_failed",
                failure=exc,
            )
    else:
        compatibility = RuntimeCompatibilityV1.build(
            status="not_checked",
            bundle_hash=authority["bundleHash"],
            trace_hash=authority["traceHash"],
        ).to_dict()
        compatibility_evidence = None

    projected_state = terminal_state or state
    timestamp = _require_text(created_at, "createdAt")
    if timestamp != run_value.get("createdAt"):
        raise ValueError("result createdAt must match its run")
    result = {
        "schemaVersion": 2,
        "resultCode": f"LAF-{result_hash[:10].upper()}",
        "resultHash": result_hash,
        "runCode": run_value["runCode"],
        "recipeCode": recipe_value["code"],
        "traceHash": authority["traceHash"] if replay_valid else None,
        "createdAt": timestamp,
        "terminalStatus": terminal_status,
        "predicateEvaluation": predicate,
        "scenarioSatisfaction": satisfaction,
        "proofOutcome": proof_receipt,
        "scenarioProjection": scenario_projection,
        "libraryProjection": {
            "schemaVersion": 1,
            "actingSeats": [actor] if actor else [],
            "observerSeats": [actor] if actor else [],
            "contract": projected_state.get("contract"),
            "purchaseRound": projected_state.get("purchaseRound"),
            "trumpSuit": projected_state.get("trumpSuit"),
            "escalation": projected_state.get("escalation") or "normal",
            "projectKinds": project_projection["projectKinds"],
            "generationLane": (
                "exact_input"
                if dict(scenario.get("origin") or {}).get("kind")
                == "exact_trace_prefix"
                else "ortools"
            ),
        },
        "artifacts": owned_artifacts,
        "authorityArtifacts": authority_artifacts,
        "researchAuthority": authority,
        "runtimeCompatibility": compatibility,
        "runtimeCompatibilityEvidence": compatibility_evidence,
        "executionReceipt": receipt,
        **verdict,
        "qualification": qualification,
        "provenance": {
            "recipeHash": recipe_value["sha256"],
            "scenarioHash": scenario["sha256"],
            "exactStateId": exact_state_id if replay_valid else None,
            "configurationAuthorityHash": canonical_sha256(configuration),
            "executorSemanticHash": executor_semantic_hash,
            "campaignSemanticHash": campaign_semantic_hash,
            "resultIdentity": identity,
        },
    }
    return validate_ortools_result_sections_v1(result, recipe=recipe_value)


def validate_ortools_result_sections_v1(
    payload: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate cross-artifact ownership for one new-backend stored result."""

    value = copy.deepcopy(dict(payload))
    identity_kind = dict(
        dict(value.get("provenance") or {}).get("resultIdentity") or {}
    ).get("kind")
    if identity_kind == ORTOOLS_RESULT_IDENTITY_V3_KIND:
        return _validate_ortools_result_sections_v3(value, recipe=recipe)
    if identity_kind != ORTOOLS_RESULT_IDENTITY_KIND:
        raise RuntimeError(
            "stored OR-Tools Research result identity kind is missing or unsupported"
        )
    sections = {
        name: value.get(name)
        for name in ("researchAuthority", "runtimeCompatibility", "executionReceipt")
    }
    if any(section is None for section in sections.values()):
        raise RuntimeError("stored OR-Tools Research result sections are incomplete")
    try:
        authority = ResearchAuthorityV1.from_dict(
            sections["researchAuthority"]
        ).to_dict()
        compatibility = RuntimeCompatibilityV1.from_dict(
            sections["runtimeCompatibility"]
        ).to_dict()
        receipt = ExecutionReceiptV1.from_dict(sections["executionReceipt"]).to_dict()
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"stored OR-Tools Research contract is invalid: {exc}"
        ) from exc

    identity = dict(dict(value.get("provenance") or {}).get("resultIdentity") or {})
    if (
        set(identity) - _BASE_IDENTITY_FIELDS
        or _REQUIRED_IDENTITY_FIELDS - set(identity)
        or identity.get("schemaVersion") != ORTOOLS_RESULT_IDENTITY_SCHEMA_VERSION
        or identity.get("kind") != ORTOOLS_RESULT_IDENTITY_KIND
        or identity.get("researchAuthorityHash") != authority["sha256"]
        or authority["backendId"] != ORTOOLS_RESEARCH_BACKEND
        or identity.get("backendId") != authority["backendId"]
        or identity.get("backendVersion") != authority["backendVersion"]
        or identity.get("qualificationAuthority") != authority["qualificationAuthority"]
        or identity.get("bundleRevisionId") != authority["bundleRevisionId"]
        or identity.get("bundleHash") != authority["bundleHash"]
        or canonical_sha256(identity) != value.get("resultHash")
        or receipt["semanticResultHash"] != value.get("resultHash")
    ):
        raise RuntimeError("stored OR-Tools Research semantic identity mismatch")

    configuration = dict(dict(recipe).get("configuration") or {})
    domain_authority_hash = _configuration_domain_hash(configuration)
    if (
        authority["bundleRevisionId"] != configuration.get("bundleRevisionId")
        or authority["bundleHash"] != configuration.get("semanticBundleHash")
        or identity.get("configurationAuthorityHash") != canonical_sha256(configuration)
        or identity.get("domainAuthorityHash") != domain_authority_hash
        or identity.get("domainIrHashesHash")
        != canonical_sha256(authority["domainIrHashes"])
    ):
        raise RuntimeError("stored OR-Tools Research bundle authority mismatch")

    artifacts = dict(value.get("artifacts") or {})
    identity_artifacts = dict(identity.get("artifactHashes") or {})
    if (
        set(artifacts) != _SEMANTIC_ARTIFACT_NAMES
        or set(identity_artifacts) != _SEMANTIC_ARTIFACT_NAMES
    ):
        raise RuntimeError("stored OR-Tools Research semantic artifact set mismatch")
    for authority_field, artifact_name in _SEMANTIC_ARTIFACT_FIELDS.items():
        reference = dict(artifacts.get(artifact_name) or {})
        artifact_hash = reference.get("sha256")
        if (
            set(reference) != _OWNED_REFERENCE_FIELDS
            or artifact_hash != authority[authority_field]
            or identity_artifacts.get(artifact_name) != artifact_hash
            or reference.get("ownerKind") != _SEMANTIC_ARTIFACT_OWNER_KIND
            or reference.get("ownerHash") != identity.get("executorSemanticHash")
            or reference.get("mediaType")
            != ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES[artifact_name]
            or type(reference.get("bytes")) is not int
            or int(reference.get("bytes", 0)) < 1
        ):
            raise RuntimeError(
                f"stored OR-Tools Research {artifact_name} substitution detected"
            )
        try:
            _validate_cas_uri(
                reference.get("uri"), str(artifact_hash), f"{artifact_name} artifact"
            )
        except ValueError as exc:
            raise RuntimeError(
                f"stored OR-Tools Research {artifact_name} substitution detected"
            ) from exc

    proof = dict(value.get("proofOutcome") or {})
    proof_integrity = proof.get("sha256")
    proof_hash = proof.get("artifactSha256")
    predicate = dict(value.get("predicateEvaluation") or {})
    candidate_hash = identity.get("candidateHash")
    if (
        canonical_sha256({key: item for key, item in proof.items() if key != "sha256"})
        != proof_integrity
        or authority["proofOutcomeHash"] != proof_hash
        or identity.get("proofOutcomeHash") != proof_hash
        or identity.get("proofModelHash") != authority["proofModelHash"]
        or identity.get("graphHash") != authority["graphHash"]
        or authority["traceHash"] != value.get("traceHash")
        or identity.get("traceHash") != authority["traceHash"]
        or identity.get("constructionModelHash") != authority["constructionModelHash"]
        or identity.get("loweringManifestHash") != authority["loweringManifestHash"]
        or identity.get("predicateEvaluationHash") != predicate.get("sha256")
        or not _HASH.fullmatch(str(candidate_hash or ""))
    ):
        raise RuntimeError("stored OR-Tools Research proof or trace authority mismatch")

    satisfaction = dict(value.get("scenarioSatisfaction") or {})
    projection = dict(value.get("scenarioProjection") or {})
    qualification = dict(value.get("qualification") or {})
    verdict = {
        key: copy.deepcopy(value.get(key))
        for key in (
            "truthLabel",
            "confidence",
            "score",
            "disposition",
            "classification",
            "criticalDecisionIds",
            "interestingSeats",
            "tacticalTags",
            "verifiedTacticalTags",
            "decisionCount",
        )
    }
    qualification_hash = qualification.get("sha256")
    if (
        canonical_sha256(
            {key: item for key, item in qualification.items() if key != "sha256"}
        )
        != qualification_hash
        or identity.get("qualificationHash") != qualification_hash
        or canonical_sha256(
            {key: item for key, item in projection.items() if key != "sha256"}
        )
        != projection.get("sha256")
        or identity.get("scenarioProjectionHash") != projection.get("sha256")
        or identity.get("verdictHash") != canonical_sha256(verdict)
        or identity.get("exactStateId") != projection.get("exactStateId")
        or satisfaction.get("exactStateId") != projection.get("exactStateId")
        or satisfaction.get("scenarioHash") != identity.get("scenarioHash")
        or satisfaction.get("predicateEvaluationHash")
        != identity.get("predicateEvaluationHash")
        or canonical_sha256(
            {key: item for key, item in satisfaction.items() if key != "sha256"}
        )
        != satisfaction.get("sha256")
    ):
        raise RuntimeError(
            "stored OR-Tools Research scenario or verdict authority mismatch"
        )

    if (
        compatibility["bundleHash"] != authority["bundleHash"]
        or compatibility["traceHash"] != authority["traceHash"]
    ):
        raise RuntimeError(
            "stored OR-Tools Research compatibility substitution detected"
        )
    evidence = value.get("runtimeCompatibilityEvidence")
    try:
        _validate_runtime_evidence(
            compatibility=compatibility,
            evidence=dict(evidence) if isinstance(evidence, Mapping) else None,
            semantic_result_hash=str(value.get("resultHash") or ""),
        )
    except ValueError as exc:
        raise RuntimeError(
            f"stored OR-Tools Research compatibility evidence invalid: {exc}"
        ) from exc
    return value


def _validate_ortools_result_sections_v3(
    value: Mapping[str, Any], *, recipe: Mapping[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    try:
        authority = ResearchAuthorityV1.from_dict(
            dict(result.get("researchAuthority") or {})
        ).to_dict()
        compatibility = RuntimeCompatibilityV1.from_dict(
            dict(result.get("runtimeCompatibility") or {})
        ).to_dict()
        receipt = ExecutionReceiptV1.from_dict(
            dict(result.get("executionReceipt") or {})
        ).to_dict()
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"stored OR-Tools Research V3 contract is invalid: {exc}"
        ) from exc
    provenance = dict(result.get("provenance") or {})
    identity = dict(provenance.get("resultIdentity") or {})
    terminal_status = result.get("terminalStatus")
    if (
        set(identity) != _V3_IDENTITY_FIELDS
        or identity.get("schemaVersion") != ORTOOLS_RESULT_IDENTITY_V3_SCHEMA_VERSION
        or identity.get("kind") != ORTOOLS_RESULT_IDENTITY_V3_KIND
        or terminal_status not in _TERMINAL_STATUSES
        or identity.get("terminalStatus") != terminal_status
        or canonical_sha256(identity) != result.get("resultHash")
        or receipt.get("semanticResultHash") != result.get("resultHash")
        or identity.get("researchAuthorityHash") != authority["sha256"]
        or identity.get("backendId") != authority["backendId"]
        or identity.get("backendVersion") != authority["backendVersion"]
        or identity.get("qualificationAuthority") != authority["qualificationAuthority"]
        or identity.get("bundleRevisionId") != authority["bundleRevisionId"]
        or identity.get("bundleHash") != authority["bundleHash"]
        or identity.get("loweringManifestHash") != authority["loweringManifestHash"]
        or identity.get("constructionModelHash") != authority["constructionModelHash"]
        or identity.get("proofModelHash") != authority["proofModelHash"]
        or identity.get("graphHash") != authority["graphHash"]
        or identity.get("traceHash") != authority["traceHash"]
        or identity.get("proofOutcomeHash") != authority["proofOutcomeHash"]
    ):
        raise RuntimeError("stored OR-Tools Research V3 semantic identity mismatch")
    scenario = dict(dict(recipe).get("scenario") or {})
    try:
        scenario_hash = _require_hash(scenario.get("sha256"), "recipe scenario hash")
        _require_hash(identity.get("campaignSemanticHash"), "campaign semantic hash")
        _require_hash(identity.get("executorSemanticHash"), "executor semantic hash")
        if identity.get("candidateHash") is not None:
            _require_hash(identity.get("candidateHash"), "candidate hash")
        if identity.get("exactStateHash") is not None:
            _require_hash(identity.get("exactStateHash"), "exact state hash")
        predicate = _validate_predicate_evaluation_v1(
            result.get("predicateEvaluation"),
            expected_hash=identity.get("predicateEvaluationHash"),
            scenario=scenario,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"stored OR-Tools Research V3 semantic preimage is invalid: {exc}"
        ) from exc
    if (
        identity.get("recipeHash") != dict(recipe).get("sha256")
        or identity.get("scenarioHash") != scenario_hash
        or provenance.get("scenarioHash") != scenario_hash
        or provenance.get("campaignSemanticHash")
        != identity.get("campaignSemanticHash")
        or provenance.get("executorSemanticHash")
        != identity.get("executorSemanticHash")
    ):
        raise RuntimeError("stored OR-Tools Research V3 recipe identity mismatch")
    configuration = dict(dict(recipe).get("configuration") or {})
    if (
        authority["bundleRevisionId"] != configuration.get("bundleRevisionId")
        or authority["bundleHash"] != configuration.get("semanticBundleHash")
        or identity.get("configurationAuthorityHash") != canonical_sha256(configuration)
        or identity.get("domainAuthorityHash")
        != _configuration_domain_hash(configuration)
        or identity.get("domainIrHashesHash")
        != canonical_sha256(authority["domainIrHashes"])
    ):
        raise RuntimeError("stored OR-Tools Research V3 bundle authority mismatch")
    artifacts = dict(result.get("artifacts") or {})
    identity_artifacts = dict(identity.get("artifactHashes") or {})
    if (
        set(artifacts) != _SEMANTIC_ARTIFACT_NAMES
        or set(identity_artifacts) != _SEMANTIC_ARTIFACT_NAMES
    ):
        raise RuntimeError("stored OR-Tools Research V3 semantic artifact set mismatch")
    for authority_field, artifact_name in _SEMANTIC_ARTIFACT_FIELDS.items():
        reference = dict(artifacts.get(artifact_name) or {})
        digest = reference.get("sha256")
        if (
            set(reference) != _OWNED_REFERENCE_FIELDS
            or digest != authority[authority_field]
            or identity_artifacts.get(artifact_name) != digest
            or reference.get("ownerKind") != _SEMANTIC_ARTIFACT_OWNER_KIND
            or reference.get("ownerHash") != identity.get("executorSemanticHash")
            or reference.get("mediaType")
            != ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES[artifact_name]
            or type(reference.get("bytes")) is not int
            or int(reference.get("bytes", 0)) < 1
        ):
            raise RuntimeError(
                f"stored OR-Tools Research V3 {artifact_name} substitution detected"
            )
        try:
            _validate_cas_uri(
                reference.get("uri"), str(digest), f"{artifact_name} artifact"
            )
        except ValueError as exc:
            raise RuntimeError(
                f"stored OR-Tools Research V3 {artifact_name} substitution detected"
            ) from exc

    authority_artifacts = dict(result.get("authorityArtifacts") or {})
    semantic_authority_hashes = dict(identity.get("authorityArtifactHashes") or {})
    optional_interest = {"interestAdmission"} if "interestAdmission" in semantic_authority_hashes else set()
    if set(authority_artifacts) != _AUTHORITY_ARTIFACT_NAMES | optional_interest or set(
        semantic_authority_hashes
    ) != {"loweringManifest", "configurationSnapshot"} | optional_interest:
        raise RuntimeError(
            "stored OR-Tools Research V3 authority artifact set mismatch"
        )
    for name, reference_value in authority_artifacts.items():
        reference = dict(reference_value or {})
        digest = reference.get("sha256")
        expected_digest = (
            result.get("resultHash")
            if name == "resultIdentity"
            else canonical_sha256(receipt)
            if name == "executionReceipt"
            else semantic_authority_hashes.get(name)
        )
        if (
            set(reference) != _OWNED_REFERENCE_FIELDS
            or digest != expected_digest
            or reference.get("ownerKind") != _AUTHORITY_ARTIFACT_OWNER_KIND
            or reference.get("ownerHash") != result.get("resultHash")
            or reference.get("mediaType")
            != ORTOOLS_AUTHORITY_ARTIFACT_MEDIA_TYPES[name]
            or type(reference.get("bytes")) is not int
            or int(reference.get("bytes", 0)) < 1
        ):
            raise RuntimeError(
                f"stored OR-Tools Research V3 {name} authority substitution detected"
            )
        try:
            _validate_cas_uri(
                reference.get("uri"), str(digest), f"{name} authority artifact"
            )
        except ValueError as exc:
            raise RuntimeError(
                f"stored OR-Tools Research V3 {name} authority substitution detected"
            ) from exc

    proof = dict(result.get("proofOutcome") or {})
    proof_identities = dict(proof.get("identities") or {})
    satisfaction = dict(result.get("scenarioSatisfaction") or {})
    projection = dict(result.get("scenarioProjection") or {})
    qualification = dict(result.get("qualification") or {})
    verdict = {
        key: copy.deepcopy(result.get(key))
        for key in (
            "truthLabel",
            "confidence",
            "score",
            "disposition",
            "classification",
            "criticalDecisionIds",
            "interestingSeats",
            "tacticalTags",
            "verifiedTacticalTags",
            "decisionCount",
        )
    }

    def canonical_section(document: Mapping[str, Any]) -> str:
        return canonical_sha256(
            {key: item for key, item in document.items() if key != "sha256"}
        )

    if (
        canonical_section(proof) != proof.get("sha256")
        or identity.get("proofOutcomeReceiptHash") != proof.get("sha256")
        or proof.get("semanticSha256") != authority["proofOutcomeHash"]
        or proof.get("artifactSha256") != authority["proofOutcomeHash"]
        or proof_identities.get("bundleHash") != authority["bundleHash"]
        or proof_identities.get("graphHash") != authority["graphHash"]
        or proof_identities.get("modelHash") != authority["proofModelHash"]
        or canonical_section(satisfaction) != satisfaction.get("sha256")
        or identity.get("scenarioSatisfactionHash") != satisfaction.get("sha256")
        or canonical_section(projection) != projection.get("sha256")
        or identity.get("scenarioProjectionHash") != projection.get("sha256")
        or canonical_section(qualification) != qualification.get("sha256")
        or identity.get("qualificationHash") != qualification.get("sha256")
        or identity.get("verdictHash") != canonical_sha256(verdict)
        or satisfaction.get("scenarioHash") != identity.get("scenarioHash")
        or satisfaction.get("predicateEvaluationHash")
        != identity.get("predicateEvaluationHash")
        or satisfaction.get("predicates") != (predicate["evidence"] if predicate is not None else [])
        or projection.get("status") != terminal_status
        or projection.get("exactStateId") != identity.get("exactStateId")
        or bool(projection.get("exactStateAvailable"))
        != (identity.get("exactStateId") is not None)
    ):
        raise RuntimeError("stored OR-Tools Research V3 scenario authority mismatch")
    should_qualify = terminal_status == "satisfied"
    if (
        qualification.get("qualified") is not should_qualify
        or result.get("disposition") != ("qualified" if should_qualify else "rejected")
        or (
            not should_qualify
            and terminal_status not in qualification.get("failures", [])
        )
    ):
        raise RuntimeError("stored OR-Tools Research V3 qualification mismatch")
    if should_qualify and (
        identity.get("candidateHash") is None
        or identity.get("predicateEvaluationHash") is None
        or identity.get("exactStateHash") is None
        or identity.get("exactStateId") is None
        or predicate is None
        or predicate.get("status") != "satisfied"
        or predicate.get("satisfied") is not True
        or proof.get("completionStatus")
        not in {"exact_complete", "bounded_complete", "policy_complete"}
        or proof.get("claimSatisfied") is not True
        or qualification.get("decisionPointAuditComplete") is not True
        or projection.get("exactStateAvailable") is not True
    ):
        raise RuntimeError("stored OR-Tools Research V3 qualified shape is incomplete")
    if (
        compatibility["bundleHash"] != authority["bundleHash"]
        or compatibility["traceHash"] != authority["traceHash"]
    ):
        raise RuntimeError(
            "stored OR-Tools Research V3 compatibility substitution detected"
        )
    try:
        evidence = result.get("runtimeCompatibilityEvidence")
        _validate_runtime_evidence(
            compatibility=compatibility,
            evidence=dict(evidence) if isinstance(evidence, Mapping) else None,
            semantic_result_hash=str(result.get("resultHash") or ""),
        )
    except ValueError as exc:
        raise RuntimeError(
            f"stored OR-Tools Research V3 compatibility evidence invalid: {exc}"
        ) from exc
    return result


__all__ = [
    "CP_SAT_QUALIFICATION_AUTHORITY",
    "ExecutionReceiptV1",
    "ORTOOLS_AUTHORITY_ARTIFACT_MEDIA_TYPES",
    "ORTOOLS_RESEARCH_BACKEND",
    "ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES",
    "ORTOOLS_RESULT_IDENTITY_KIND",
    "ORTOOLS_RESULT_IDENTITY_SCHEMA_VERSION",
    "ORTOOLS_RESULT_IDENTITY_V3_KIND",
    "ORTOOLS_RESULT_IDENTITY_V3_SCHEMA_VERSION",
    "ResearchAuthorityV1",
    "RuntimeCompatibilityV1",
    "build_ortools_research_result_v1",
    "build_ortools_research_result_v3",
    "build_ortools_result_identity_v2",
    "build_ortools_result_identity_v3",
    "validate_ortools_result_artifact_documents_v3",
    "validate_ortools_result_sections_v1",
    "validate_native_interest_artifact_v1",
    "validate_ortools_stage_receipt_v2",
]
