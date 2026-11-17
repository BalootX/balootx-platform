"""Versioned scoring and perspective policy contracts.

Raw evidence remains immutable. These profiles only rank already-valid
evidence and never convert a failed truth gate into a passing candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .evidence_v3 import (
    DecisionPerspectiveEvidenceV3,
    LaffaEvidenceV2,
    TruthLabel,
    TruthMode,
    canonical_sha256,
)
from .finalist_competence import canonical_surrounding_score_projection_v1


_DECISION_DIMENSIONS = frozenset({
    "outcomeImpact", "uniqueness", "disasterAvoidance", "kaboutImpact",
    "sacrificeValue", "delayedGain", "tacticalDepth", "informationBurden",
    "executionDifficulty", "surprise", "robustness", "confidence",
    "explanationFidelity", "visualClarity", "novelty", "coverage",
})
_LAFFA_DIMENSIONS = frozenset({
    "overallPeak", "secondPeak", "topKThrillMass", "criticalDensity",
    "perspectiveBreadth", "multiAngleYield", "causalChainStrength",
    "outcomeImportance", "visualClarity", "narrativeArc",
})


def decision_materiality_v1(item: Any) -> float:
    """Return the governed importance of confirming one surrounding decision."""
    if hasattr(item, "primary_decision_score_evidence"):
        evidence = item.primary_decision_score_evidence
        if evidence is None:
            return 100.0 if not bool(item.forced) else 0.0
        return float(canonical_surrounding_score_projection_v1(
            evidence,
            actual_action=item.actual_action,
            best_action=item.best_action,
        )["materiality"])
    alternatives = tuple(item.alternatives)
    if bool(item.forced) or len(alternatives) < 2:
        return 0.0
    best, second, worst = alternatives[0], alternatives[1], alternatives[-1]
    best_margin = float(best.get("expectedMargin", 0.0))
    second_margin = float(second.get("expectedMargin", 0.0))
    worst_margin = float(worst.get("expectedMargin", 0.0))
    best_probability = float(best.get("winProbability", 0.5))
    second_probability = float(second.get("winProbability", 0.5))
    return min(100.0, max(
        max(0.0, best_margin - second_margin) * 6.0,
        max(0.0, best_probability - second_probability) * 100.0,
        max(0.0, best_margin - worst_margin) * 2.5,
        float(item.difficulty),
        float(item.debate_score) * 0.75,
        100.0 if bool(item.catastrophic_crossing) else 0.0,
    ))


def _surrounding_qualification_projection_v1(item: Any) -> dict[str, Any]:
    if hasattr(item, "primary_decision_score_evidence"):
        evidence = item.primary_decision_score_evidence
        if evidence is None:
            return {
                "quality": 0.0,
                "materiality": 100.0,
                "confidence": 0.0,
                "catastrophicCrossing": True,
            }
        return canonical_surrounding_score_projection_v1(
            evidence,
            actual_action=item.actual_action,
            best_action=item.best_action,
        )
    return {
        "quality": max(
            0.0,
            100.0
            - float(item.expected_regret) * 5.0
            - float(item.probability_drop) * 100.0,
        ),
        "materiality": decision_materiality_v1(item),
        "confidence": float(item.confidence),
        "catastrophicCrossing": bool(item.catastrophic_crossing),
    }


def _weights(value: Mapping[str, Any], allowed: frozenset[str]) -> tuple[tuple[str, float], ...]:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unsupported score dimensions: {sorted(unknown)}")
    result = tuple(sorted((str(key), float(weight)) for key, weight in value.items()))
    if not result or any(weight <= 0 for _, weight in result):
        raise ValueError("score weights must be non-empty and strictly positive")
    return result


@dataclass(frozen=True, slots=True)
class ContentSelectionScoreV1:
    score_family: str
    profile_hash: str
    evidence_hash: str
    score: float
    passed: bool
    rejection_reasons: tuple[str, ...]
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        score_family: str,
        profile_hash: str,
        evidence_hash: str,
        score: float,
        passed: bool,
        rejection_reasons: tuple[str, ...] = (),
    ) -> "ContentSelectionScoreV1":
        if not 0.0 <= float(score) <= 100.0:
            raise ValueError("content selection score must be between 0 and 100")
        core = {
            "schemaVersion": 1,
            "scoreFamily": score_family,
            "profileHash": profile_hash,
            "evidenceHash": evidence_hash,
            "score": round(float(score), 9),
            "passed": bool(passed),
            "rejectionReasons": list(rejection_reasons),
        }
        return cls(
            score_family, profile_hash, evidence_hash, float(score), bool(passed),
            rejection_reasons, canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "scoreFamily": self.score_family,
            "profileHash": self.profile_hash,
            "evidenceHash": self.evidence_hash,
            "score": self.score,
            "passed": self.passed,
            "rejectionReasons": list(self.rejection_reasons),
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContentSelectionScoreV1":
        if int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported content selection score schema")
        result = cls.build(
            score_family=str(value.get("scoreFamily") or ""),
            profile_hash=str(value.get("profileHash") or ""),
            evidence_hash=str(value.get("evidenceHash") or ""),
            score=float(value.get("score", -1.0)),
            passed=bool(value.get("passed")),
            rejection_reasons=tuple(
                str(item) for item in value.get("rejectionReasons") or ()
            ),
        )
        if str(value.get("sha256") or "") != result.sha256:
            raise ValueError("content selection score checksum mismatch")
        return result


@dataclass(frozen=True, slots=True)
class DecisionScoreBreakdownV1:
    """Immutable explanation of one profile-weighted decision score.

    Missing dimensions are non-applicable rather than artificial zeroes. The
    remaining governed weights are normalized so profile intent stays stable.
    """

    profile_hash: str
    evidence_hash: str
    dimensions: tuple[tuple[str, float], ...]
    normalized_weights: tuple[tuple[str, float], ...]
    contributions: tuple[tuple[str, float], ...]
    score: float
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        profile_hash: str,
        evidence_hash: str,
        dimensions: Mapping[str, float],
        weights: Mapping[str, float],
    ) -> "DecisionScoreBreakdownV1":
        applicable = tuple(sorted(
            (str(key), float(value))
            for key, value in dimensions.items()
            if key in weights and float(weights[key]) > 0.0
        ))
        if not applicable:
            raise ValueError("decision score has no applicable dimensions")
        weight_total = sum(float(weights[key]) for key, _ in applicable)
        normalized = tuple(
            (key, float(weights[key]) / weight_total)
            for key, _ in applicable
        )
        values = dict(applicable)
        contributions = tuple(
            (key, values[key] * weight)
            for key, weight in normalized
        )
        score = sum(value for _, value in contributions)
        core = {
            "schemaVersion": 1,
            "profileHash": str(profile_hash),
            "evidenceHash": str(evidence_hash),
            "dimensions": dict(applicable),
            "normalizedWeights": dict(normalized),
            "contributions": dict(contributions),
            "score": round(score, 9),
        }
        return cls(
            str(profile_hash), str(evidence_hash), applicable, normalized,
            contributions, score, canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "profileHash": self.profile_hash,
            "evidenceHash": self.evidence_hash,
            "dimensions": dict(self.dimensions),
            "normalizedWeights": dict(self.normalized_weights),
            "contributions": dict(self.contributions),
            "score": self.score,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SurroundingCompetenceEvidenceV1:
    selection_profile_hash: str
    trace_analysis_hash: str
    candidate_decision_id: str
    evaluated_decisions: int
    preceding_decisions: int
    following_decisions: int
    minimum_quality: float
    mean_quality: float
    material_mistakes: int
    unverified_decisions: int
    catastrophic_crossings: int
    passed: bool
    rejection_reasons: tuple[str, ...]
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        selection_profile_hash: str,
        trace_analysis_hash: str,
        candidate_decision_id: str,
        evaluated_decisions: int,
        preceding_decisions: int,
        following_decisions: int,
        minimum_quality: float,
        mean_quality: float,
        material_mistakes: int,
        unverified_decisions: int,
        catastrophic_crossings: int,
        passed: bool,
        rejection_reasons: tuple[str, ...] = (),
    ) -> "SurroundingCompetenceEvidenceV1":
        if min(
            evaluated_decisions, preceding_decisions, following_decisions,
            material_mistakes, unverified_decisions, catastrophic_crossings,
        ) < 0:
            raise ValueError("surrounding competence counts cannot be negative")
        if preceding_decisions + following_decisions != evaluated_decisions:
            raise ValueError("surrounding competence decision counts are inconsistent")
        if not 0.0 <= float(minimum_quality) <= 100.0:
            raise ValueError("minimum surrounding quality must be in [0, 100]")
        if not 0.0 <= float(mean_quality) <= 100.0:
            raise ValueError("mean surrounding quality must be in [0, 100]")
        core = {
            "schemaVersion": 1,
            "selectionProfileHash": str(selection_profile_hash),
            "traceAnalysisHash": str(trace_analysis_hash),
            "candidateDecisionId": str(candidate_decision_id),
            "evaluatedDecisions": int(evaluated_decisions),
            "precedingDecisions": int(preceding_decisions),
            "followingDecisions": int(following_decisions),
            "minimumQuality": round(float(minimum_quality), 9),
            "meanQuality": round(float(mean_quality), 9),
            "materialMistakes": int(material_mistakes),
            "unverifiedDecisions": int(unverified_decisions),
            "catastrophicCrossings": int(catastrophic_crossings),
            "passed": bool(passed),
            "rejectionReasons": list(rejection_reasons),
        }
        return cls(
            str(selection_profile_hash), str(trace_analysis_hash),
            str(candidate_decision_id), int(evaluated_decisions),
            int(preceding_decisions), int(following_decisions),
            float(minimum_quality), float(mean_quality), int(material_mistakes),
            int(unverified_decisions), int(catastrophic_crossings), bool(passed),
            tuple(rejection_reasons), canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "selectionProfileHash": self.selection_profile_hash,
            "traceAnalysisHash": self.trace_analysis_hash,
            "candidateDecisionId": self.candidate_decision_id,
            "evaluatedDecisions": self.evaluated_decisions,
            "precedingDecisions": self.preceding_decisions,
            "followingDecisions": self.following_decisions,
            "minimumQuality": self.minimum_quality,
            "meanQuality": self.mean_quality,
            "materialMistakes": self.material_mistakes,
            "unverifiedDecisions": self.unverified_decisions,
            "catastrophicCrossings": self.catastrophic_crossings,
            "passed": self.passed,
            "rejectionReasons": list(self.rejection_reasons),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SelectionProfileV1:
    profile_id: str
    revision: str
    weights: tuple[tuple[str, float], ...]
    allowed_truth_labels: tuple[TruthLabel, ...]
    minimum_confidence: float
    minimum_score: float
    require_verified_tactical_play: bool
    minimum_surrounding_quality: float
    minimum_surrounding_confidence: float
    minimum_surrounding_materiality: float
    maximum_surrounding_material_mistakes: int
    maximum_surrounding_unverified_decisions: int
    maximum_surrounding_catastrophic_crossings: int
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "SelectionProfileV1":
        allowed_keys = {
            "schemaVersion", "id", "revision", "weights", "allowedTruthLabels",
            "minimumConfidence", "minimumScore", "requireVerifiedTacticalPlay",
            "minimumSurroundingQuality", "minimumSurroundingConfidence",
            "minimumSurroundingMateriality",
            "maximumSurroundingMaterialMistakes",
            "maximumSurroundingUnverifiedDecisions",
            "maximumSurroundingCatastrophicCrossings",
        }
        unknown = set(value) - allowed_keys
        if unknown:
            raise ValueError(f"unknown selection profile keys: {sorted(unknown)}")
        if int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported selection profile schema version")
        profile_id = str(value.get("id") or "").strip()
        revision = str(value.get("revision") or "").strip()
        if not profile_id or not revision:
            raise ValueError("selection profile identity is required")
        weights = _weights(dict(value.get("weights") or {}), _DECISION_DIMENSIONS)
        labels = tuple(sorted(
            {TruthLabel(str(item)) for item in value.get("allowedTruthLabels") or ()},
            key=lambda item: item.value,
        ))
        if not labels or TruthLabel.UNRESOLVED in labels or TruthLabel.REJECTED in labels:
            raise ValueError("selection profile may admit only supported truth labels")
        confidence = float(value.get("minimumConfidence", 0.0))
        minimum_score = float(value.get("minimumScore", 0.0))
        require_tactical = value.get("requireVerifiedTacticalPlay", False)
        if not isinstance(require_tactical, bool):
            raise ValueError("selection tactical-play gate must be boolean")
        if not 0.0 <= confidence <= 1.0 or not 0.0 <= minimum_score <= 100.0:
            raise ValueError("selection confidence and score gates are out of range")
        surrounding_quality = float(value.get("minimumSurroundingQuality", 0.0))
        surrounding_confidence = float(value.get("minimumSurroundingConfidence", 0.0))
        surrounding_materiality = float(value.get("minimumSurroundingMateriality", 0.0))
        surrounding_maximums = (
            int(value.get("maximumSurroundingMaterialMistakes", 2_147_483_647)),
            int(value.get("maximumSurroundingUnverifiedDecisions", 2_147_483_647)),
            int(value.get("maximumSurroundingCatastrophicCrossings", 2_147_483_647)),
        )
        if not 0.0 <= surrounding_quality <= 100.0:
            raise ValueError("minimum surrounding quality must be in [0, 100]")
        if not 0.0 <= surrounding_confidence <= 1.0:
            raise ValueError("minimum surrounding confidence must be in [0, 1]")
        if not 0.0 <= surrounding_materiality <= 100.0:
            raise ValueError("minimum surrounding materiality must be in [0, 100]")
        if min(surrounding_maximums) < 0:
            raise ValueError("maximum surrounding failure counts cannot be negative")
        core = {
            "schemaVersion": 1,
            "id": profile_id,
            "revision": revision,
            "weights": dict(weights),
            "allowedTruthLabels": [item.value for item in labels],
            "minimumConfidence": confidence,
            "minimumScore": minimum_score,
            "requireVerifiedTacticalPlay": require_tactical,
            "minimumSurroundingQuality": surrounding_quality,
            "minimumSurroundingConfidence": surrounding_confidence,
            "minimumSurroundingMateriality": surrounding_materiality,
            "maximumSurroundingMaterialMistakes": surrounding_maximums[0],
            "maximumSurroundingUnverifiedDecisions": surrounding_maximums[1],
            "maximumSurroundingCatastrophicCrossings": surrounding_maximums[2],
        }
        return cls(
            core["id"], core["revision"], weights, labels, confidence,
            minimum_score, require_tactical, surrounding_quality,
            surrounding_confidence, surrounding_materiality,
            *surrounding_maximums, canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "weights": dict(self.weights),
            "allowedTruthLabels": [item.value for item in self.allowed_truth_labels],
            "minimumConfidence": self.minimum_confidence,
            "minimumScore": self.minimum_score,
            "requireVerifiedTacticalPlay": self.require_verified_tactical_play,
            "minimumSurroundingQuality": self.minimum_surrounding_quality,
            "minimumSurroundingConfidence": self.minimum_surrounding_confidence,
            "minimumSurroundingMateriality": self.minimum_surrounding_materiality,
            "maximumSurroundingMaterialMistakes": self.maximum_surrounding_material_mistakes,
            "maximumSurroundingUnverifiedDecisions": self.maximum_surrounding_unverified_decisions,
            "maximumSurroundingCatastrophicCrossings": self.maximum_surrounding_catastrophic_crossings,
            "sha256": self.sha256,
        }

    @property
    def requires_surrounding_competence(self) -> bool:
        return (
            self.minimum_surrounding_quality > 0.0
            or self.minimum_surrounding_confidence > 0.0
            or self.minimum_surrounding_materiality > 0.0
            or self.maximum_surrounding_material_mistakes < 2_147_483_647
            or self.maximum_surrounding_unverified_decisions < 2_147_483_647
            or self.maximum_surrounding_catastrophic_crossings < 2_147_483_647
        )

    def qualify_surrounding(
        self,
        analyses: Any,
        *,
        candidate_decision_id: str,
        trace_analysis_hash: str,
        unresolved_decision_ids: Any = (),
        competence_verified_decision_ids: Any = (),
    ) -> SurroundingCompetenceEvidenceV1:
        items = tuple(
            item for item in analyses
            if not bool(item.forced) and str(item.decision_id) != candidate_decision_id
        )
        candidate = next(
            (item for item in analyses if str(item.decision_id) == candidate_decision_id),
            None,
        )
        if candidate is None:
            raise ValueError("candidate decision is absent from trace analysis")
        projections = {
            str(item.decision_id): _surrounding_qualification_projection_v1(item)
            for item in items
        }
        qualities = tuple(
            float(projections[str(item.decision_id)]["quality"])
            for item in items
        )
        material = sum(value < self.minimum_surrounding_quality for value in qualities)
        item_ids = {str(item.decision_id) for item in items}
        competence_verified_ids = {
            str(item) for item in competence_verified_decision_ids
            if str(item) in item_ids
        }
        unverified_ids = {
            decision_id
            for decision_id, projection in projections.items()
            if float(projection["materiality"])
            >= self.minimum_surrounding_materiality
            and float(projection["confidence"])
            < self.minimum_surrounding_confidence
            and decision_id not in competence_verified_ids
        }
        unverified_ids.update(
            str(item) for item in unresolved_decision_ids
            if str(item) in item_ids
        )
        unverified = len(unverified_ids)
        catastrophic = sum(
            bool(projection["catastrophicCrossing"])
            for projection in projections.values()
        )
        reasons: list[str] = []
        if not items:
            reasons.append("no_surrounding_decisions")
        if material > self.maximum_surrounding_material_mistakes:
            reasons.append("material_mistakes")
        if unverified > self.maximum_surrounding_unverified_decisions:
            reasons.append("unverified_decisions")
        if catastrophic > self.maximum_surrounding_catastrophic_crossings:
            reasons.append("catastrophic_crossings")
        preceding = sum(int(item.ordinal) < int(candidate.ordinal) for item in items)
        return SurroundingCompetenceEvidenceV1.build(
            selection_profile_hash=self.sha256,
            trace_analysis_hash=trace_analysis_hash,
            candidate_decision_id=candidate_decision_id,
            evaluated_decisions=len(items),
            preceding_decisions=preceding,
            following_decisions=len(items) - preceding,
            minimum_quality=min(qualities, default=0.0),
            mean_quality=(sum(qualities) / len(qualities) if qualities else 0.0),
            material_mistakes=material,
            unverified_decisions=unverified,
            catastrophic_crossings=catastrophic,
            passed=not reasons,
            rejection_reasons=tuple(reasons),
        )

    def score(self, evidence: DecisionPerspectiveEvidenceV3) -> ContentSelectionScoreV1:
        reasons: list[str] = []
        if evidence.truth_mode != TruthMode.ACTOR_OBSERVER:
            reasons.append("not_actor_truth")
        if evidence.forced:
            reasons.append("forced_decision")
        if evidence.truth_label not in self.allowed_truth_labels:
            reasons.append("truth_label")
        if evidence.confidence < self.minimum_confidence:
            reasons.append("confidence")
        try:
            breakdown = self.breakdown(evidence)
            score = breakdown.score
        except ValueError:
            reasons.append("no_applicable_dimensions")
            score = 0.0
        if score < self.minimum_score:
            reasons.append("minimum_score")
        evidence_hash = canonical_sha256(evidence.to_dict())
        return ContentSelectionScoreV1.build(
            score_family="decision_content_selection",
            profile_hash=self.sha256,
            evidence_hash=evidence_hash,
            score=score,
            passed=not reasons,
            rejection_reasons=tuple(reasons),
        )

    def breakdown(
        self,
        evidence: DecisionPerspectiveEvidenceV3,
    ) -> DecisionScoreBreakdownV1:
        evidence_hash = canonical_sha256(evidence.to_dict())
        return DecisionScoreBreakdownV1.build(
            profile_hash=self.sha256,
            evidence_hash=evidence_hash,
            dimensions=evidence.evidence,
            weights=dict(self.weights),
        )


@dataclass(frozen=True, slots=True)
class ThrillProfileV1:
    profile_id: str
    revision: str
    weights: tuple[tuple[str, float], ...]
    minimum_critical_points: int
    minimum_score: float
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ThrillProfileV1":
        allowed_keys = {
            "schemaVersion", "id", "revision", "weights",
            "minimumCriticalPoints", "minimumScore",
        }
        unknown = set(value) - allowed_keys
        if unknown:
            raise ValueError(f"unknown thrill profile keys: {sorted(unknown)}")
        if int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported thrill profile schema version")
        profile_id = str(value.get("id") or "").strip()
        revision = str(value.get("revision") or "").strip()
        if not profile_id or not revision:
            raise ValueError("thrill profile identity is required")
        weights = _weights(dict(value.get("weights") or {}), _LAFFA_DIMENSIONS)
        minimum_points = int(value.get("minimumCriticalPoints", 1))
        minimum_score = float(value.get("minimumScore", 0.0))
        if minimum_points < 1 or not 0.0 <= minimum_score <= 100.0:
            raise ValueError("thrill profile gates are out of range")
        core = {
            "schemaVersion": 1,
            "id": profile_id,
            "revision": revision,
            "weights": dict(weights),
            "minimumCriticalPoints": minimum_points,
            "minimumScore": minimum_score,
        }
        return cls(
            core["id"], core["revision"], weights, minimum_points,
            minimum_score, canonical_sha256(core),
        )

    def score(self, evidence: LaffaEvidenceV2) -> ContentSelectionScoreV1:
        dimensions = {
            "overallPeak": float(evidence.overall_peak or 0.0),
            "secondPeak": float(evidence.second_peak or 0.0),
            "topKThrillMass": min(100.0, evidence.top_k_thrill_mass / 3.0),
            "criticalDensity": evidence.critical_density * 100.0,
            "perspectiveBreadth": evidence.perspective_breadth * 25.0,
            "multiAngleYield": evidence.multi_angle_yield * 25.0,
            "causalChainStrength": evidence.causal_chain_strength,
            "outcomeImportance": evidence.outcome_importance,
            "visualClarity": evidence.visual_clarity,
            "narrativeArc": evidence.narrative_arc,
        }
        score = sum(dimensions[key] * weight for key, weight in self.weights) / sum(
            weight for _, weight in self.weights
        )
        reasons: list[str] = []
        if len(evidence.critical_decision_ids) < self.minimum_critical_points:
            reasons.append("critical_points")
        if score < self.minimum_score:
            reasons.append("minimum_score")
        return ContentSelectionScoreV1.build(
            score_family="laffa_content_selection",
            profile_hash=self.sha256,
            evidence_hash=evidence.sha256,
            score=score,
            passed=not reasons,
            rejection_reasons=tuple(reasons),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "weights": dict(self.weights),
            "minimumCriticalPoints": self.minimum_critical_points,
            "minimumScore": self.minimum_score,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ResearchComparisonProfileV1:
    """Fail-closed promotion contract for matched Research lanes."""

    profile_id: str
    revision: str
    minimum_qualified_per_lane: int
    minimum_distinct_selected_finalists_per_lane: int
    minimum_attempted_seeds_per_lane: int
    minimum_yield_multiplier: float
    minimum_cost_reduction: float
    required_promotion_cpu_boundary: str
    require_zero_semantic_defects: bool
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ResearchComparisonProfileV1":
        allowed = {
            "schemaVersion",
            "id",
            "revision",
            "minimumQualifiedPerLane",
            "minimumDistinctSelectedFinalistsPerLane",
            "minimumAttemptedSeedsPerLane",
            "minimumYieldMultiplier",
            "minimumCostReduction",
            "requiredPromotionCpuBoundary",
            "requireZeroSemanticDefects",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"unknown Research comparison profile keys: {sorted(unknown)}"
            )
        if int(value.get("schemaVersion", 0)) != 2:
            raise ValueError("unsupported Research comparison profile schema")
        profile_id = str(value.get("id") or "").strip()
        revision = str(value.get("revision") or "").strip()
        if not profile_id or not revision:
            raise ValueError("Research comparison profile identity is required")
        minimum_qualified = int(value.get("minimumQualifiedPerLane", 0))
        minimum_distinct = int(
            value.get("minimumDistinctSelectedFinalistsPerLane", 0)
        )
        minimum_attempted = int(value.get("minimumAttemptedSeedsPerLane", 0))
        minimum_yield = float(value.get("minimumYieldMultiplier", 0.0))
        minimum_cost = float(value.get("minimumCostReduction", -1.0))
        required_cpu_boundary = str(
            value.get("requiredPromotionCpuBoundary") or ""
        ).strip()
        zero_defects = value.get("requireZeroSemanticDefects")
        if not 1 <= minimum_qualified <= 1_000_000:
            raise ValueError(
                "Research comparison finalist minimum is out of range"
            )
        if not 1 <= minimum_distinct <= 1_000_000:
            raise ValueError(
                "Research comparison distinct-finalist minimum is out of range"
            )
        if not 1 <= minimum_attempted <= 10_000_000:
            raise ValueError("Research comparison attempt minimum is out of range")
        if not 1.0 <= minimum_yield <= 100.0:
            raise ValueError("Research comparison yield multiplier is out of range")
        if not 0.0 <= minimum_cost < 1.0:
            raise ValueError("Research comparison cost reduction is out of range")
        if required_cpu_boundary != "isolated_process_cpu_v1":
            raise ValueError(
                "Research comparison promotion requires isolated process CPU evidence"
            )
        if zero_defects is not True:
            raise ValueError(
                "Research comparison must require zero semantic defects"
            )
        core = {
            "schemaVersion": 2,
            "id": profile_id,
            "revision": revision,
            "minimumQualifiedPerLane": minimum_qualified,
            "minimumDistinctSelectedFinalistsPerLane": minimum_distinct,
            "minimumAttemptedSeedsPerLane": minimum_attempted,
            "minimumYieldMultiplier": minimum_yield,
            "minimumCostReduction": minimum_cost,
            "requiredPromotionCpuBoundary": required_cpu_boundary,
            "requireZeroSemanticDefects": True,
        }
        return cls(
            profile_id,
            revision,
            minimum_qualified,
            minimum_distinct,
            minimum_attempted,
            minimum_yield,
            minimum_cost,
            required_cpu_boundary,
            True,
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 2,
            "id": self.profile_id,
            "revision": self.revision,
            "minimumQualifiedPerLane": self.minimum_qualified_per_lane,
            "minimumDistinctSelectedFinalistsPerLane": (
                self.minimum_distinct_selected_finalists_per_lane
            ),
            "minimumAttemptedSeedsPerLane": self.minimum_attempted_seeds_per_lane,
            "minimumYieldMultiplier": self.minimum_yield_multiplier,
            "minimumCostReduction": self.minimum_cost_reduction,
            "requiredPromotionCpuBoundary": self.required_promotion_cpu_boundary,
            "requireZeroSemanticDefects": self.require_zero_semantic_defects,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class PerspectivePolicyV1:
    profile_id: str
    revision: str
    actor_prompt: str
    seat_experience_prompts: tuple[str, ...]
    require_four_projections: bool
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "PerspectivePolicyV1":
        allowed_keys = {
            "schemaVersion", "id", "revision", "actorPrompt",
            "seatExperiencePrompts", "requireFourProjections",
        }
        unknown = set(value) - allowed_keys
        if unknown:
            raise ValueError(f"unknown perspective profile keys: {sorted(unknown)}")
        if int(value.get("schemaVersion", 1)) != 1:
            raise ValueError("unsupported perspective profile schema version")
        actor_prompt = str(value.get("actorPrompt") or "").strip()
        prompts = tuple(sorted({str(item).strip() for item in value.get("seatExperiencePrompts") or () if str(item).strip()}))
        if actor_prompt != "choose_acting_action":
            raise ValueError("actor truth requires the acting-action prompt contract")
        if not prompts or actor_prompt in prompts:
            raise ValueError("seat-experience prompts must be non-actor contracts")
        core = {
            "schemaVersion": 1,
            "id": str(value["id"]),
            "revision": str(value["revision"]),
            "actorPrompt": actor_prompt,
            "seatExperiencePrompts": list(prompts),
            "requireFourProjections": bool(value.get("requireFourProjections", True)),
        }
        return cls(
            core["id"], core["revision"], actor_prompt, prompts,
            core["requireFourProjections"], canonical_sha256(core),
        )
