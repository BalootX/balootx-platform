"""Pure finalist competence adjudication shared by online and replay analysis."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from baloot_platform.analysis.decision_scores import (
    PrimaryDecisionScoreEvidenceV1,
    primary_decision_score_confidence_v1,
)

from .evidence_v3 import canonical_sha256


_FAILURE_PROBABILITY_KEYS = (
    "matchWinProbability",
    "contractSuccessProbability",
    "roundLeadProbability",
    "winProbability",
)

_ADJUDICATION_PROFILE_CORE_KEYS_V2 = frozenset({
    "schemaVersion",
    "primaryProfileHash",
    "reanalysisProfileHash",
    "maximumDecisionsPerTrace",
    "seedOffset",
    "minimumHiddenDeals",
    "minimumConfidence",
    "minimumQuality",
    "minimumBestShare",
    "maximumCompetenceExpectedMarginRegret",
    "maximumCompetenceMatchWinDrop",
    "maximumExpectedMarginDrift",
    "maximumCohorts",
    "minimumAgreement",
    "thirdCohortOnDisagreement",
    "requireSameRecommendedAction",
})
_ADJUDICATION_PROFILE_KEYS_V2 = frozenset({
    *_ADJUDICATION_PROFILE_CORE_KEYS_V2,
    "sha256",
})


def _canonical_sha256_hex_v1(value: Any, *, field: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{field} must be a canonical lowercase SHA-256 digest")
    return digest


def _integer_v1(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _finite_number_v1(value: Any, *, field: str) -> float:
    if not isinstance(value, float):
        raise ValueError(f"{field} must be a normalized floating-point number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be finite")
    return normalized


def normalize_finalist_adjudication_profile_v2(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize the complete adjudication policy snapshot.

    The snapshot is intentionally owned by this pipeline-free module so live
    adjudication and receipt reload share one authority without importing one
    another.
    """

    raw = dict(value)
    if set(raw) != _ADJUDICATION_PROFILE_KEYS_V2:
        raise ValueError(
            "finalist adjudication profile is incomplete or contains unknown keys"
        )
    if raw["schemaVersion"] != 2 or isinstance(raw["schemaVersion"], bool):
        raise ValueError("unsupported finalist adjudication profile schema")
    primary_hash = _canonical_sha256_hex_v1(
        raw["primaryProfileHash"], field="finalist primary profile hash"
    )
    reanalysis_hash = _canonical_sha256_hex_v1(
        raw["reanalysisProfileHash"], field="finalist reanalysis profile hash"
    )
    maximum_decisions = _integer_v1(
        raw["maximumDecisionsPerTrace"],
        field="finalist adjudication decision bound",
    )
    seed_offset = _integer_v1(
        raw["seedOffset"], field="finalist adjudication seed offset"
    )
    minimum_hidden = _integer_v1(
        raw["minimumHiddenDeals"],
        field="finalist adjudication hidden-deal threshold",
    )
    maximum_cohorts = _integer_v1(
        raw["maximumCohorts"], field="finalist adjudication cohort bound"
    )
    minimum_agreement = _integer_v1(
        raw["minimumAgreement"],
        field="finalist adjudication agreement threshold",
    )
    minimum_confidence = _finite_number_v1(
        raw["minimumConfidence"],
        field="finalist adjudication confidence threshold",
    )
    minimum_quality = _finite_number_v1(
        raw["minimumQuality"], field="finalist adjudication quality threshold"
    )
    minimum_best_share = _finite_number_v1(
        raw["minimumBestShare"],
        field="finalist adjudication best-share threshold",
    )
    maximum_regret = _finite_number_v1(
        raw["maximumCompetenceExpectedMarginRegret"],
        field="finalist adjudication competence regret threshold",
    )
    maximum_match_drop = _finite_number_v1(
        raw["maximumCompetenceMatchWinDrop"],
        field="finalist adjudication competence match-win threshold",
    )
    maximum_drift = _finite_number_v1(
        raw["maximumExpectedMarginDrift"],
        field="finalist adjudication margin-drift threshold",
    )
    third_on_disagreement = raw["thirdCohortOnDisagreement"]
    require_same = raw["requireSameRecommendedAction"]
    if not isinstance(third_on_disagreement, bool) or not isinstance(
        require_same, bool
    ):
        raise ValueError("finalist adjudication boolean policy is invalid")
    if not 1 <= maximum_decisions <= 64:
        raise ValueError("finalist adjudication decision bound must be in [1, 64]")
    if seed_offset < 1:
        raise ValueError("finalist adjudication seed offset must be positive")
    if minimum_hidden < 2:
        raise ValueError("finalist adjudication requires at least two hidden deals")
    if not 0.0 <= minimum_confidence <= 1.0 or not 0.0 <= minimum_quality <= 100.0:
        raise ValueError("finalist adjudication confidence or quality is out of range")
    if not 0.0 <= minimum_best_share <= 1.0:
        raise ValueError("finalist adjudication best-share threshold is invalid")
    if maximum_regret < 0.0:
        raise ValueError("finalist adjudication competence regret is invalid")
    if not 0.0 <= maximum_match_drop <= 1.0:
        raise ValueError("finalist adjudication competence match-win drop is invalid")
    if maximum_drift < 0.0:
        raise ValueError("finalist adjudication margin drift must be non-negative")
    if not 2 <= maximum_cohorts <= 3:
        raise ValueError("finalist adjudication cohort bound must be in [2, 3]")
    if not 2 <= minimum_agreement <= maximum_cohorts:
        raise ValueError("finalist adjudication agreement threshold is invalid")
    if maximum_cohorts == 3 and not third_on_disagreement:
        raise ValueError(
            "three-cohort finalist adjudication requires adaptive third analysis"
        )
    if require_same is not True:
        raise ValueError(
            "finalist adjudication requires independent recommendation agreement"
        )
    core = {
        "schemaVersion": 2,
        "primaryProfileHash": primary_hash,
        "reanalysisProfileHash": reanalysis_hash,
        "maximumDecisionsPerTrace": maximum_decisions,
        "seedOffset": seed_offset,
        "minimumHiddenDeals": minimum_hidden,
        "minimumConfidence": minimum_confidence,
        "minimumQuality": minimum_quality,
        "minimumBestShare": minimum_best_share,
        "maximumCompetenceExpectedMarginRegret": maximum_regret,
        "maximumCompetenceMatchWinDrop": maximum_match_drop,
        "maximumExpectedMarginDrift": maximum_drift,
        "maximumCohorts": maximum_cohorts,
        "minimumAgreement": minimum_agreement,
        "thirdCohortOnDisagreement": third_on_disagreement,
        "requireSameRecommendedAction": True,
    }
    declared_hash = _canonical_sha256_hex_v1(
        raw["sha256"], field="finalist adjudication profile sha256"
    )
    if canonical_sha256(core) != declared_hash:
        raise ValueError("finalist adjudication profile checksum mismatch")
    return {**core, "sha256": declared_hash}


def build_finalist_adjudication_profile_v2(
    stability: Mapping[str, Any],
    *,
    primary_profile_hash: str,
    reanalysis_profile_hash: str,
    maximum_decisions_per_trace: int,
    minimum_confidence: float,
    minimum_quality: float,
) -> dict[str, Any]:
    core = {
        "schemaVersion": 2,
        "primaryProfileHash": str(primary_profile_hash),
        "reanalysisProfileHash": str(reanalysis_profile_hash),
        "maximumDecisionsPerTrace": int(maximum_decisions_per_trace),
        "seedOffset": int(stability.get("seed_offset", 0)),
        "minimumHiddenDeals": int(stability.get("minimum_hidden_deals", 0)),
        "minimumConfidence": float(minimum_confidence),
        "minimumQuality": float(minimum_quality),
        "minimumBestShare": float(stability.get("minimum_best_share", -1.0)),
        "maximumCompetenceExpectedMarginRegret": float(
            stability.get("maximum_competence_expected_margin_regret", -1.0)
        ),
        "maximumCompetenceMatchWinDrop": float(
            stability.get("maximum_competence_match_win_drop", -1.0)
        ),
        "maximumExpectedMarginDrift": float(
            stability.get("maximum_expected_margin_drift", -1.0)
        ),
        "maximumCohorts": int(stability.get("maximum_cohorts", 0)),
        "minimumAgreement": int(stability.get("minimum_agreement", 0)),
        "thirdCohortOnDisagreement": stability.get(
            "third_cohort_on_disagreement"
        ),
        "requireSameRecommendedAction": stability.get(
            "require_same_recommended_action"
        ),
    }
    return normalize_finalist_adjudication_profile_v2({
        **core,
        "sha256": canonical_sha256(core),
    })


def _action_hash(value: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(value))


def canonical_score_metrics_by_action_v1(
    evidence: Mapping[str, Any] | PrimaryDecisionScoreEvidenceV1,
) -> dict[str, dict[str, Any]]:
    """Normalize the validated score-row authority for finalist evaluation."""

    frozen = (
        evidence
        if isinstance(evidence, PrimaryDecisionScoreEvidenceV1)
        else PrimaryDecisionScoreEvidenceV1.from_mapping(evidence)
    )
    result: dict[str, dict[str, Any]] = {}
    for row in frozen.rows:
        metrics: dict[str, Any] = {
            "expectedMargin": float(row["expectedRoundMargin"]),
            "roundLeadProbability": float(row["roundLeadProbability"]),
            "contractSuccessProbability": float(
                row["contractSuccessProbability"]
            ),
            "bestShare": float(row["bestShare"]),
        }
        if frozen.decision_kind == "play":
            metrics.update({
                "matchWinProbability": float(row["matchWinProbability"]),
                "confidence95": [
                    float(value) for value in row["roundLeadConfidence95"]
                ],
                "matchWinConfidence95": [
                    float(value) for value in row["matchWinConfidence95"]
                ],
                "roundLeadConfidence95": [
                    float(value) for value in row["roundLeadConfidence95"]
                ],
            })
        result[_action_hash(dict(row["action"]))] = metrics
    return result


def canonical_score_support_classification_v1(
    evidence: Mapping[str, Any] | PrimaryDecisionScoreEvidenceV1,
) -> dict[str, Any]:
    """Derive one closed support class from canonical decision-score rows.

    Search-completion flags carried beside a trace are descriptive mirrors,
    not an independent authority.  Play support is therefore derived from the
    validated row population and method.  Bid analysis has no exact-search
    completion columns and is conservatively classified as sampled evidence.
    """

    frozen = (
        evidence
        if isinstance(evidence, PrimaryDecisionScoreEvidenceV1)
        else PrimaryDecisionScoreEvidenceV1.from_mapping(evidence)
    )
    if frozen.decision_kind == "bid":
        return {
            "exhaustiveExactSupport": False,
            "populationCoverageComplete": False,
            "solverCompletion": "sampled",
            "truthClass": "unresolved",
            "budgetExhausted": False,
            "completionStatuses": ["sampled"],
        }
    statuses = sorted({
        str(row["searchCompletionStatus"]) for row in frozen.rows
    })
    one_status = statuses[0] if len(statuses) == 1 else "mixed_or_unknown"
    exhaustive_method = str(frozen.method).startswith("exhaustive_")
    exhaustive_exact = bool(
        exhaustive_method
        and statuses == ["exact"]
        and all(float(row["exactShare"]) >= 1.0 for row in frozen.rows)
    )
    complete_population = bool(
        exhaustive_method
        and statuses in (["exact"], ["heuristic_complete"])
    )
    truth_class = (
        "exact_optimal"
        if exhaustive_exact
        else (
            "full_population_best_known"
            if complete_population
            else "unresolved"
        )
    )
    return {
        "exhaustiveExactSupport": exhaustive_exact,
        "populationCoverageComplete": complete_population,
        "solverCompletion": one_status,
        "truthClass": truth_class,
        "budgetExhausted": "budget_exhausted" in statuses,
        "completionStatuses": statuses,
    }


def canonical_score_confidence_v1(
    evidence: Mapping[str, Any] | PrimaryDecisionScoreEvidenceV1,
) -> float:
    """Recompute analyzer confidence exclusively from canonical score rows."""
    return primary_decision_score_confidence_v1(evidence)


def canonical_surrounding_score_projection_v1(
    evidence: Mapping[str, Any] | PrimaryDecisionScoreEvidenceV1,
    *,
    actual_action: Mapping[str, Any],
    best_action: Mapping[str, Any],
) -> dict[str, Any]:
    """Project every surrounding-qualification metric from score rows."""

    frozen = (
        evidence
        if isinstance(evidence, PrimaryDecisionScoreEvidenceV1)
        else PrimaryDecisionScoreEvidenceV1.from_mapping(evidence)
    )
    cohort = canonical_finalist_cohort_v1(
        frozen,
        cohort_index=0,
        evidence_hash=frozen.sha256,
        actual_action=actual_action,
        best_action=best_action,
    )
    best = dict(cohort["bestMetrics"])
    actual = dict(cohort["primaryMetrics"])
    if not best or not actual:
        raise ValueError(
            "canonical surrounding score rows do not contain best and actual actions"
        )
    expected_regret = max(
        0.0,
        float(best["expectedMargin"]) - float(actual["expectedMargin"]),
    )
    probability_drop = max(
        0.0,
        float(best["roundLeadProbability"])
        - float(actual["roundLeadProbability"]),
    )
    catastrophic = finalist_failure_boundary_crossed_v1(
        cohort, best, actual
    )
    rows = tuple(frozen.rows)
    best_row = rows[0]
    second_row = rows[min(1, len(rows) - 1)]
    worst_row = rows[-1]
    materiality = min(100.0, max(
        max(
            0.0,
            float(best_row["expectedRoundMargin"])
            - float(second_row["expectedRoundMargin"]),
        )
        * 6.0,
        max(
            0.0,
            float(best_row["roundLeadProbability"])
            - float(second_row["roundLeadProbability"]),
        )
        * 100.0,
        max(
            0.0,
            float(best_row["expectedRoundMargin"])
            - float(worst_row["expectedRoundMargin"]),
        )
        * 2.5,
        100.0 if catastrophic else 0.0,
    ))
    return {
        "expectedRegret": expected_regret,
        "probabilityDrop": probability_drop,
        "quality": max(
            0.0,
            100.0 - expected_regret * 5.0 - probability_drop * 100.0,
        ),
        "catastrophicCrossing": catastrophic,
        "materiality": materiality,
        "confidence": canonical_score_confidence_v1(frozen),
    }


def canonical_finalist_cohort_v1(
    evidence: Mapping[str, Any] | PrimaryDecisionScoreEvidenceV1 | None,
    *,
    cohort_index: int,
    evidence_hash: str,
    actual_action: Mapping[str, Any],
    best_action: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the sole row-authoritative cohort projection used by verdicts."""

    best: dict[str, Any] = {}
    actual: dict[str, Any] = {}
    value_interval: list[float] | None = None
    if evidence is None:
        classification = {
            "exhaustiveExactSupport": False,
            "populationCoverageComplete": False,
            "solverCompletion": "missing",
            "truthClass": "unresolved",
            "budgetExhausted": False,
            "supportClassificationValid": False,
        }
        hidden_deals: int | None = None
        trace_seed: int | None = None
        confidence = 0.0
    else:
        frozen = (
            evidence
            if isinstance(evidence, PrimaryDecisionScoreEvidenceV1)
            else PrimaryDecisionScoreEvidenceV1.from_mapping(evidence)
        )
        by_action = canonical_score_metrics_by_action_v1(frozen)
        best = dict(by_action.get(_action_hash(best_action)) or {})
        actual = dict(by_action.get(_action_hash(actual_action)) or {})
        raw_interval = actual.pop("confidence95", None)
        best.pop("confidence95", None)
        value_interval = (
            [float(item) for item in raw_interval]
            if isinstance(raw_interval, (list, tuple)) and len(raw_interval) == 2
            else None
        )
        classification = {
            **canonical_score_support_classification_v1(frozen),
            "supportClassificationValid": True,
        }
        hidden_deals = frozen.hidden_deal_support
        trace_seed = frozen.cohort_seed
        confidence = canonical_score_confidence_v1(frozen)
    return {
        "cohortIndex": int(cohort_index),
        "evidenceHash": str(evidence_hash),
        "traceSeed": trace_seed,
        "bestAction": dict(best_action),
        "primaryAction": dict(actual_action),
        "bestMetrics": best,
        "primaryMetrics": actual,
        "valueConfidence95": value_interval,
        "hiddenDeals": hidden_deals,
        "confidence": float(confidence),
        "bestShare": best.get("bestShare", 0.0),
        "canonicalScoreEvidencePresent": evidence is not None,
        **classification,
        # Failure boundaries are derived from the same canonical metrics below.
        "catastrophicCrossing": False,
    }


def finalist_adjudication_target_required_v2(
    item: Any,
    *,
    adjudication_profile: Mapping[str, Any],
    selection_profile: Mapping[str, Any],
) -> bool:
    """Project the exact governed set of decisions requiring adjudication."""

    if bool(item.forced):
        return False
    profile = normalize_finalist_adjudication_profile_v2(adjudication_profile)
    evidence = item.primary_decision_score_evidence
    if evidence is None:
        return True
    projection = canonical_surrounding_score_projection_v1(
        evidence,
        actual_action=item.actual_action,
        best_action=item.best_action,
    )
    cohort = canonical_finalist_cohort_v1(
        evidence,
        cohort_index=0,
        evidence_hash=item.sha256,
        actual_action=item.actual_action,
        best_action=item.best_action,
    )
    actual_is_best = _action_hash(item.actual_action) == _action_hash(
        item.best_action
    )
    low_support = bool(
        int(cohort["hiddenDeals"]) < int(profile["minimumHiddenDeals"])
        and not cohort["exhaustiveExactSupport"]
    )
    return bool(
        not actual_is_best
        or projection["catastrophicCrossing"]
        or projection["materiality"]
        >= float(selection_profile["minimumSurroundingMateriality"])
        or projection["quality"] < float(profile["minimumQuality"])
        or projection["confidence"] < float(profile["minimumConfidence"])
        or low_support
    )


def finalist_support_classification_reasons_v1(
    cohort: Mapping[str, Any],
) -> list[str]:
    if cohort.get("supportClassificationValid") is False:
        return ["invalid_support_truth_classification"]
    exhaustive = bool(cohort.get("exhaustiveExactSupport"))
    complete_population = bool(cohort.get("populationCoverageComplete"))
    solver = str(cohort.get("solverCompletion") or "unknown")
    truth = str(cohort.get("truthClass") or "unresolved")
    valid = (
        exhaustive
        and complete_population
        and solver == "exact"
        and truth == "exact_optimal"
    ) or (
        not exhaustive
        and complete_population
        and solver == "heuristic_complete"
        and truth == "full_population_best_known"
    ) or (
        not exhaustive
        and not complete_population
        and truth == "unresolved"
    )
    return [] if valid else ["invalid_support_truth_classification"]


def _strict_support_reasons(
    cohort: Mapping[str, Any],
    *,
    minimum_hidden_deals: int,
    minimum_confidence: float,
    minimum_best_share: float,
) -> list[str]:
    """Retain the strict-confirmation reasons beside competence evidence."""

    exhaustive = bool(cohort.get("exhaustiveExactSupport"))
    complete_population = bool(cohort.get("populationCoverageComplete"))
    reasons: list[str] = finalist_support_classification_reasons_v1(cohort)
    # Complete population with best-known continuations is useful secondary
    # proof, but it is not exact truth and can never become featured truth.
    if complete_population and not exhaustive:
        reasons.append("population_solver_best_known_only")
    hidden = cohort.get("hiddenDeals")
    if not exhaustive and not complete_population and (
        hidden is None or int(hidden) < int(minimum_hidden_deals)
    ):
        reasons.append("low_hidden_deal_support")
    confidence = float(cohort.get("confidence", 0.0))
    if not math.isfinite(confidence):
        reasons.append("invalid_confidence")
    elif not exhaustive and not complete_population and confidence < float(
        minimum_confidence
    ):
        reasons.append("low_confidence")
    # Exact exhaustive proof does not need a sampled best-share floor.  A
    # heuristic complete population remains best-known and must retain it.
    best_share = float(cohort.get("bestShare", 0.0))
    if not math.isfinite(best_share):
        reasons.append("invalid_best_share")
    elif not exhaustive and best_share < float(minimum_best_share):
        reasons.append("low_best_share")
    if bool(cohort.get("budgetExhausted")):
        reasons.append("analysis_not_converged")
        if "low_hidden_deal_support" in reasons:
            reasons.append("budget_exhausted_low_support")
    return sorted(set(reasons))


def finalist_strict_support_reasons_v1(
    cohort: Mapping[str, Any],
    *,
    minimum_hidden_deals: int,
    minimum_confidence: float,
    minimum_best_share: float,
) -> list[str]:
    """Public strict-support projection shared by online and replay verdicts."""

    return _strict_support_reasons(
        cohort,
        minimum_hidden_deals=minimum_hidden_deals,
        minimum_confidence=minimum_confidence,
        minimum_best_share=minimum_best_share,
    )


def _competence_support_reasons(
    cohort: Mapping[str, Any],
    *,
    minimum_hidden_deals: int,
    minimum_confidence: float,
) -> list[str]:
    """Apply the published secondary-proof support authority.

    Best-share and convergence remain strict-confirmation evidence, but the
    Adaptive Finalist's competence proof is intentionally governed by hidden
    support/confidence (waived only for exact exhaustive or complete finite
    population support) plus its utility, boundary, interval, and quorum
    checks below.
    """

    exhaustive = bool(cohort.get("exhaustiveExactSupport"))
    complete_population = bool(cohort.get("populationCoverageComplete"))
    reasons: list[str] = finalist_support_classification_reasons_v1(cohort)
    hidden = cohort.get("hiddenDeals")
    if not exhaustive and not complete_population and (
        hidden is None or int(hidden) < int(minimum_hidden_deals)
    ):
        reasons.append("low_hidden_deal_support")
    confidence = float(cohort.get("confidence", 0.0))
    if not math.isfinite(confidence):
        reasons.append("invalid_confidence")
    elif not exhaustive and not complete_population and confidence < float(
        minimum_confidence
    ):
        reasons.append("low_confidence")
    return sorted(set(reasons))


def finalist_failure_boundary_crossed_v1(
    cohort: Mapping[str, Any],
    best: Mapping[str, Any],
    primary: Mapping[str, Any],
) -> bool:
    if bool(cohort.get("catastrophicCrossing")) or bool(
        primary.get("catastrophicCrossing")
    ):
        return True
    if "expectedMargin" in best and "expectedMargin" in primary:
        if float(best["expectedMargin"]) >= 0.0 > float(primary["expectedMargin"]):
            return True
    return any(
        key in best
        and key in primary
        and float(best[key]) >= 0.5 > float(primary[key])
        for key in _FAILURE_PROBABILITY_KEYS
    )


def evaluate_strict_cohort_agreement_v1(
    cohorts: Sequence[Mapping[str, Any]],
    *,
    minimum_hidden_deals: int,
    minimum_confidence: float,
    minimum_best_share: float,
    maximum_expected_margin_drift: float,
    minimum_agreement: int,
) -> dict[str, Any]:
    """Evaluate the recommendation-agreement proof over normalized cohorts.

    Callers own the meaning of the frozen primary recommendation: online
    generation confirms the recorded primary action, while post-trace analysis
    confirms the primary analysis recommendation and may separately label a
    confirmed mistake.  This pure projection owns the shared support, quorum,
    margin-drift, and failure-boundary semantics used before that caller-owned
    comparison.
    """

    if not 2 <= len(cohorts) <= 3:
        raise ValueError("strict finalist agreement requires two or three cohorts")
    if not 2 <= int(minimum_agreement) <= 3:
        raise ValueError("strict finalist agreement threshold is invalid")
    if float(maximum_expected_margin_drift) < 0.0:
        raise ValueError("strict finalist margin drift must be non-negative")

    qualified_by_action: dict[str, list[dict[str, Any]]] = {}
    support_reasons: set[str] = set()
    best_action_hashes: list[str] = []
    for index, raw in enumerate(cohorts):
        cohort = dict(raw)
        best_action = dict(cohort.get("bestAction") or {})
        action_hash = _action_hash(best_action)
        best_action_hashes.append(action_hash)
        reasons = (
            ["missing_primary_decision_score_evidence"]
            if cohort.get("canonicalScoreEvidencePresent") is False
            else finalist_strict_support_reasons_v1(
                cohort,
                minimum_hidden_deals=minimum_hidden_deals,
                minimum_confidence=minimum_confidence,
                minimum_best_share=minimum_best_share,
            )
        )
        support_reasons.update(reasons)
        if not reasons:
            cohort["cohortIndex"] = int(cohort.get("cohortIndex", index))
            qualified_by_action.setdefault(action_hash, []).append(cohort)

    ranked = sorted(
        qualified_by_action.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    if not ranked or len(ranked[0][1]) < int(minimum_agreement):
        reasons = set(support_reasons)
        reasons.add("insufficient_qualified_agreement")
        if len(set(best_action_hashes)) > 1:
            reasons.add("cohort_reversal")
        core = {
            "schemaVersion": 1,
            "status": "unresolved",
            "strictReasons": sorted(reasons),
            "agreementCount": 0,
            "majorityActionHash": None,
            "selectedAction": None,
            "expectedMarginDrift": None,
            "supporterCohortIndices": [],
            "catastrophicClassification": None,
        }
        return {**core, "sha256": canonical_sha256(core)}

    majority_hash, supporters = ranked[0]
    reasons: set[str] = set()
    raw_margins = [
        dict(item.get("bestMetrics") or {}).get("expectedMargin")
        for item in supporters
    ]
    if any(
        item is None or not math.isfinite(float(item))
        for item in raw_margins
    ):
        reasons.add("missing_majority_margin")
        margin_drift: float | None = None
    else:
        margins = [float(item) for item in raw_margins]
        margin_drift = max(margins) - min(margins)
        if margin_drift > float(maximum_expected_margin_drift):
            reasons.add("expected_margin_drift")

    catastrophic_classes: set[bool] = set()
    for item in supporters:
        best = dict(item.get("bestMetrics") or {})
        primary = dict(item.get("primaryMetrics") or {})
        if not best or not primary:
            reasons.add("missing_primary_boundary_metrics")
            continue
        boundary_keys = ("expectedMargin", *_FAILURE_PROBABILITY_KEYS)
        if any(
            (key in best) != (key in primary)
            or (
                key in best
                and (
                    not math.isfinite(float(best[key]))
                    or not math.isfinite(float(primary[key]))
                )
            )
            for key in boundary_keys
        ):
            reasons.add("missing_primary_boundary_metrics")
            continue
        catastrophic_classes.add(
            finalist_failure_boundary_crossed_v1(item, best, primary)
        )
    if len(catastrophic_classes) > 1:
        reasons.add("catastrophic_classification_disagreement")

    selected = dict(supporters[0].get("bestAction") or {})
    strict_reasons = sorted(reasons)
    core = {
        "schemaVersion": 1,
        "status": "strict_confirmed" if not strict_reasons else "unresolved",
        "strictReasons": strict_reasons,
        "agreementCount": len(supporters),
        "majorityActionHash": majority_hash,
        "selectedAction": selected,
        "expectedMarginDrift": margin_drift,
        "supporterCohortIndices": [
            int(item["cohortIndex"]) for item in supporters
        ],
        "catastrophicClassification": (
            next(iter(catastrophic_classes))
            if len(catastrophic_classes) == 1
            else None
        ),
    }
    return {**core, "sha256": canonical_sha256(core)}


def evaluate_primary_competence_v1(
    primary_action: Mapping[str, Any],
    cohorts: Sequence[Mapping[str, Any]],
    *,
    minimum_hidden_deals: int,
    minimum_confidence: float,
    minimum_best_share: float,
    maximum_expected_margin_regret: float,
    maximum_match_win_drop: float,
    maximum_expected_margin_drift: float,
    minimum_agreement: int,
) -> dict[str, Any]:
    """Decide whether the recorded primary action is robustly competent.

    This function never searches for or substitutes another action.  Callers
    may invoke it only after strict recommendation confirmation has failed.
    Every cohort keeps its strict support, confidence, best-share, convergence,
    failure-boundary, and value-interval evidence.  The strict reasons do not
    silently become secondary-proof blockers: competence uses the published
    Adaptive Finalist support ladder.
    """

    if not 2 <= len(cohorts) <= 3:
        raise ValueError("finalist competence requires two or three cohorts")
    if not 2 <= int(minimum_agreement) <= 3:
        raise ValueError("finalist competence agreement threshold is invalid")
    primary = dict(primary_action)
    if not primary:
        raise ValueError("finalist competence requires the recorded primary action")
    primary_hash = _action_hash(primary)
    evaluations: list[dict[str, Any]] = []
    qualified: list[dict[str, Any]] = []
    for index, raw in enumerate(cohorts):
        cohort = dict(raw)
        strict_reasons = _strict_support_reasons(
            cohort,
            minimum_hidden_deals=minimum_hidden_deals,
            minimum_confidence=minimum_confidence,
            minimum_best_share=minimum_best_share,
        )
        reasons = _competence_support_reasons(
            cohort,
            minimum_hidden_deals=minimum_hidden_deals,
            minimum_confidence=minimum_confidence,
        )
        cohort_primary = dict(cohort.get("primaryAction") or {})
        if _action_hash(cohort_primary) != primary_hash:
            reasons.append("primary_action_drift")
        best_action = dict(cohort.get("bestAction") or {})
        best = dict(cohort.get("bestMetrics") or {})
        candidate = dict(cohort.get("primaryMetrics") or {})
        regret: float | None = None
        match_drop: float | None = None
        value_interval: list[float] | None = None
        if "expectedMargin" not in best or "expectedMargin" not in candidate:
            reasons.append("missing_competence_utility")
        elif not all(
            math.isfinite(float(value))
            for value in (
                *best.values(),
                *candidate.values(),
            )
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ):
            reasons.append("invalid_competence_utility")
        else:
            regret = max(
                0.0,
                float(best["expectedMargin"])
                - float(candidate["expectedMargin"]),
            )
            if regret > float(maximum_expected_margin_regret):
                reasons.append("competence_regret")
            match_supported = (
                "matchWinProbability" in best,
                "matchWinProbability" in candidate,
            )
            if match_supported[0] != match_supported[1]:
                reasons.append("incomplete_match_win_metric")
            elif all(match_supported):
                match_drop = max(
                    0.0,
                    float(best["matchWinProbability"])
                    - float(candidate["matchWinProbability"]),
                )
                if match_drop > float(maximum_match_win_drop):
                    reasons.append("competence_match_win_drop")
            one_sided_boundary = any(
                (key in best) != (key in candidate)
                for key in _FAILURE_PROBABILITY_KEYS
            )
            if one_sided_boundary:
                reasons.append("incomplete_failure_boundary_metric")
            if finalist_failure_boundary_crossed_v1(cohort, best, candidate):
                reasons.append("failure_boundary_crossing")
            raw_interval = cohort.get("valueConfidence95")
            if raw_interval is not None:
                if (
                    not isinstance(raw_interval, (list, tuple))
                    or len(raw_interval) != 2
                    or not all(math.isfinite(float(value)) for value in raw_interval)
                    or float(raw_interval[0]) > float(raw_interval[1])
                ):
                    reasons.append("invalid_value_interval")
                else:
                    value_interval = [
                        float(raw_interval[0]),
                        float(raw_interval[1]),
                    ]
        reasons = sorted(set(reasons))
        evaluation = {
            "cohortIndex": int(cohort.get("cohortIndex", index)),
            "evidenceHash": cohort.get("evidenceHash"),
            "traceSeed": cohort.get("traceSeed"),
            "bestActionHash": _action_hash(best_action),
            "primaryActionHash": primary_hash,
            "expectedMarginRegret": regret,
            "matchWinDrop": match_drop,
            "matchWinMetricSupported": match_drop is not None,
            "valueConfidence95": value_interval,
            "budgetExhausted": bool(cohort.get("budgetExhausted")),
            "populationCoverageComplete": bool(
                cohort.get("populationCoverageComplete")
            ),
            "solverCompletion": str(
                cohort.get("solverCompletion") or "unknown"
            ),
            "truthClass": str(cohort.get("truthClass") or "unresolved"),
            "exhaustiveExactSupport": bool(cohort.get("exhaustiveExactSupport")),
            "strictReasons": strict_reasons,
            "reasons": reasons,
            "qualified": not reasons,
        }
        evaluations.append(evaluation)
        if not reasons:
            qualified.append(evaluation)

    regrets = [
        float(item["expectedMarginRegret"])
        for item in qualified
        if item["expectedMarginRegret"] is not None
    ]
    regret_drift = max(regrets) - min(regrets) if regrets else 0.0
    intervals = [
        item["valueConfidence95"]
        for item in qualified
        if item["valueConfidence95"] is not None
    ]
    intervals_stable = bool(
        len(intervals) < int(minimum_agreement)
        or max(float(item[0]) for item in intervals)
        <= min(float(item[1]) for item in intervals)
    )
    resolution_reasons: list[str] = []
    if len(qualified) < int(minimum_agreement):
        resolution_reasons.append("insufficient_competence_agreement")
    if regret_drift > float(maximum_expected_margin_drift):
        resolution_reasons.append("competence_expected_margin_drift")
    if not intervals_stable:
        resolution_reasons.append("competence_value_interval_instability")
    passed = not resolution_reasons
    core = {
        "schemaVersion": 1,
        "policy": "recorded_primary_robust_competence_v1",
        "status": "primary_robust_competent" if passed else "unresolved",
        "primaryActionHash": primary_hash,
        "agreementCount": len(qualified),
        "minimumAgreement": int(minimum_agreement),
        "expectedMarginRegretDrift": regret_drift,
        "maximumExpectedMarginDrift": float(maximum_expected_margin_drift),
        "valueIntervalsStable": intervals_stable,
        "resolutionReasons": resolution_reasons,
        "cohortEvaluations": evaluations,
    }
    return {**core, "sha256": canonical_sha256(core)}


def project_finalist_adjudication_v2(
    cohorts: Sequence[Mapping[str, Any]],
    *,
    primary_best_action: Mapping[str, Any],
    primary_actual_action: Mapping[str, Any],
    adjudication_profile: Mapping[str, Any],
    initial_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """Project one complete finalist verdict from rows and the frozen policy.

    This is deliberately serialization-free.  The live writer adds immutable
    analysis objects and the reload parser reconstructs these exact inputs,
    then compares every returned verdict field.
    """

    profile = normalize_finalist_adjudication_profile_v2(adjudication_profile)
    frozen_cohorts = tuple(dict(item) for item in cohorts)
    if not 2 <= len(frozen_cohorts) <= int(profile["maximumCohorts"]):
        raise ValueError("finalist adjudication cohort count is outside policy")
    strict = evaluate_strict_cohort_agreement_v1(
        frozen_cohorts,
        minimum_hidden_deals=int(profile["minimumHiddenDeals"]),
        minimum_confidence=float(profile["minimumConfidence"]),
        minimum_best_share=float(profile["minimumBestShare"]),
        maximum_expected_margin_drift=float(
            profile["maximumExpectedMarginDrift"]
        ),
        minimum_agreement=int(profile["minimumAgreement"]),
    )
    reasons = set(str(item) for item in strict["strictReasons"])
    majority_hash = strict["majorityActionHash"]
    strict_agreement_count = int(strict["agreementCount"])
    supporter_indices = [
        int(item) for item in strict["supporterCohortIndices"]
    ]
    if (
        majority_hash is not None
        and majority_hash != _action_hash(primary_best_action)
    ):
        reasons.add("primary_recommendation_not_confirmed")
    strict_reasons = sorted(reasons)
    competence: dict[str, Any] | None = None
    if reasons:
        competence = evaluate_primary_competence_v1(
            primary_actual_action,
            frozen_cohorts,
            minimum_hidden_deals=int(profile["minimumHiddenDeals"]),
            minimum_confidence=float(profile["minimumConfidence"]),
            minimum_best_share=float(profile["minimumBestShare"]),
            maximum_expected_margin_regret=float(
                profile["maximumCompetenceExpectedMarginRegret"]
            ),
            maximum_match_win_drop=float(
                profile["maximumCompetenceMatchWinDrop"]
            ),
            maximum_expected_margin_drift=float(
                profile["maximumExpectedMarginDrift"]
            ),
            minimum_agreement=int(profile["minimumAgreement"]),
        )
        if competence["status"] == "primary_robust_competent":
            majority_hash = _action_hash(primary_actual_action)
            supporter_indices = [
                int(item["cohortIndex"])
                for item in competence["cohortEvaluations"]
                if item["qualified"]
            ]
            reasons.clear()
        else:
            reasons.update(
                str(item) for item in competence.get("resolutionReasons") or ()
            )
            for evaluation in competence.get("cohortEvaluations") or ():
                reasons.update(str(item) for item in evaluation.get("reasons") or ())
    status = (
        "unresolved"
        if reasons
        else (
            "primary_robust_competent"
            if competence is not None
            and competence["status"] == "primary_robust_competent"
            else "strict_confirmed"
        )
    )
    strict_primary_action_confirmed = bool(
        status == "strict_confirmed"
        and majority_hash == _action_hash(primary_actual_action)
    )
    strict_recommendation_status = (
        (
            "confirmed_primary"
            if strict_primary_action_confirmed
            else "confirmed_mistake"
        )
        if status == "strict_confirmed"
        else None
    )
    return {
        "status": status,
        "strictRecommendationStatus": strict_recommendation_status,
        "strictPrimaryActionConfirmed": strict_primary_action_confirmed,
        "reasons": sorted(reasons),
        "strictReasons": strict_reasons,
        "competenceEvidence": competence,
        "initialReasons": sorted(set(str(item) for item in initial_reasons)),
        "agreementCount": (
            int(competence["agreementCount"])
            if competence is not None and status == "primary_robust_competent"
            else strict_agreement_count
        ),
        "supporterCohortIndices": supporter_indices,
        "majorityActionHash": majority_hash,
        "featuredTruthEligible": status == "strict_confirmed",
        "surroundingVerified": status != "unresolved",
        "expectedMarginDrift": strict["expectedMarginDrift"],
    }
