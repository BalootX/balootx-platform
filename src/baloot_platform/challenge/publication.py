"""Governed answer evidence for one exact actor-perspective challenge."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .analysis_pipeline import TraceDecisionAnalysisV3
from .evidence_v3 import (
    DecisionPerspectiveEvidenceV3,
    TruthLabel,
    TruthMode,
    canonical_sha256,
)


_PUBLIC_METRICS = frozenset({
    "expectedRoundMargin",
    "expectedTeamAwardedPoints",
    "expectedOpponentAwardedPoints",
    "roundLeadProbability",
    "contractSuccessProbability",
    "matchWinProbability",
    "downsideCvar10",
    "volatility",
    "bestShare",
    "exactShare",
})


def _action_key(value: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(value))


@dataclass(frozen=True, slots=True)
class ChallengeAnswerPolicyV1:
    allowed_truth_labels: tuple[TruthLabel, ...]
    minimum_confidence: float
    accepted_margin_gap: float
    accepted_probability_gap: float
    public_metrics: tuple[str, ...]
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ChallengeAnswerPolicyV1":
        allowed_keys = {
            "schemaVersion",
            "allowedTruthLabels",
            "minimumConfidence",
            "acceptedMarginGap",
            "acceptedProbabilityGap",
            "publicMetrics",
        }
        unknown = set(value) - allowed_keys
        if unknown:
            raise ValueError(f"unknown challenge answer policy keys: {sorted(unknown)}")
        if int(value.get("schemaVersion", 1)) != 1:
            raise ValueError("unsupported challenge answer policy schema version")
        labels = tuple(sorted(
            {TruthLabel(str(item)) for item in value.get("allowedTruthLabels") or ()},
            key=lambda item: item.value,
        ))
        if not labels or any(
            label in {TruthLabel.UNRESOLVED, TruthLabel.REJECTED}
            for label in labels
        ):
            raise ValueError("answer policy may admit only supported truth labels")
        confidence = float(value.get("minimumConfidence", 0.0))
        margin_gap = float(value.get("acceptedMarginGap", 0.0))
        probability_gap = float(value.get("acceptedProbabilityGap", 0.0))
        metrics = tuple(sorted({str(item) for item in value.get("publicMetrics") or ()}))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("answer policy confidence must be between zero and one")
        if margin_gap < 0.0 or not 0.0 <= probability_gap <= 1.0:
            raise ValueError("answer policy equivalence tolerances are out of range")
        unsupported = set(metrics) - _PUBLIC_METRICS
        if unsupported:
            raise ValueError(f"unsupported public action metrics: {sorted(unsupported)}")
        core = {
            "schemaVersion": 1,
            "allowedTruthLabels": [item.value for item in labels],
            "minimumConfidence": confidence,
            "acceptedMarginGap": margin_gap,
            "acceptedProbabilityGap": probability_gap,
            "publicMetrics": list(metrics),
        }
        return cls(
            labels,
            confidence,
            margin_gap,
            probability_gap,
            metrics,
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "allowedTruthLabels": [item.value for item in self.allowed_truth_labels],
            "minimumConfidence": self.minimum_confidence,
            "acceptedMarginGap": self.accepted_margin_gap,
            "acceptedProbabilityGap": self.accepted_probability_gap,
            "publicMetrics": list(self.public_metrics),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ChallengeAnswerEvidenceV1:
    trace_hash: str
    decision_id: str
    observer_seat: str
    projection_hash: str
    truth_label: TruthLabel
    confidence: float
    recommendation: Mapping[str, Any]
    accepted_actions: tuple[Mapping[str, Any], ...]
    action_evidence: tuple[Mapping[str, Any], ...]
    engine_cognitive_difficulty: float
    analysis_hash: str
    actor_evidence_hash: str
    policy_hash: str
    answer_mode: str
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        trace_hash: str,
        analysis: TraceDecisionAnalysisV3,
        actor_evidence: DecisionPerspectiveEvidenceV3,
        policy: ChallengeAnswerPolicyV1,
    ) -> "ChallengeAnswerEvidenceV1":
        if len(str(trace_hash)) != 64:
            raise ValueError("challenge answer evidence requires a trace hash")
        if actor_evidence.truth_mode != TruthMode.ACTOR_OBSERVER:
            raise ValueError("challenge answer evidence requires actor-observer truth")
        if actor_evidence.decision_id != analysis.decision_id:
            raise ValueError("challenge answer decision identity mismatch")
        if actor_evidence.acting_seat.name.lower() != analysis.acting_seat:
            raise ValueError("challenge answer acting-seat identity mismatch")
        if actor_evidence.forced or analysis.forced:
            raise ValueError("forced decisions cannot become actor challenges")
        if actor_evidence.truth_label not in policy.allowed_truth_labels:
            raise ValueError("challenge answer truth label is not publishable by policy")
        if actor_evidence.confidence < policy.minimum_confidence:
            raise ValueError("challenge answer confidence is below publication policy")
        if abs(actor_evidence.confidence - analysis.confidence) > 1e-12:
            raise ValueError("actor evidence confidence does not match paired analysis")
        if actor_evidence.recommended_action is None:
            raise ValueError("challenge answer evidence requires a recommendation")
        if _action_key(actor_evidence.recommended_action) != _action_key(analysis.best_action):
            raise ValueError("actor recommendation does not match paired analysis")
        if not analysis.alternatives:
            raise ValueError("challenge answer evidence requires analyzed alternatives")
        best = analysis.alternatives[0]
        best_margin = float(best["expectedMargin"])
        best_probability = float(best["winProbability"])
        accepted = tuple(
            dict(item["action"])
            for item in analysis.alternatives
            if best_margin - float(item["expectedMargin"]) <= policy.accepted_margin_gap
            and best_probability - float(item["winProbability"]) <= policy.accepted_probability_gap
        )
        if not accepted:
            raise ValueError("challenge answer policy accepted no legal alternative")
        if actor_evidence.truth_label == TruthLabel.DEBATABLE and len(accepted) < 2:
            raise ValueError("debatable challenge must expose multiple supported answers")
        metric_sources = {
            "expectedRoundMargin": "expectedMargin",
            "expectedTeamAwardedPoints": "expectedTeamAwardedPoints",
            "expectedOpponentAwardedPoints": "expectedOpponentAwardedPoints",
            "roundLeadProbability": "winProbability",
            "contractSuccessProbability": "contractSuccessProbability",
            "matchWinProbability": "matchWinProbability",
            "downsideCvar10": "downsideCvar10",
            "volatility": "volatility",
            "bestShare": "bestShare",
            "exactShare": "exactShare",
        }
        metric_support = {
            "expectedRoundMargin": "expected_round_margin",
            "expectedTeamAwardedPoints": "expected_team_awarded_points",
            "expectedOpponentAwardedPoints": "expected_opponent_awarded_points",
            "roundLeadProbability": "round_lead_probability",
            "contractSuccessProbability": "contract_success_probability",
            "matchWinProbability": "match_win_probability",
            "downsideCvar10": "downside_cvar",
            "volatility": "volatility",
            "bestShare": "best_deal_share",
            "exactShare": "exact_solve_share",
        }
        unsupported_requested = [
            name
            for name in policy.public_metrics
            if metric_support[name] not in analysis.supported_metrics
        ]
        if unsupported_requested:
            raise ValueError(
                "analysis does not support requested public metrics: "
                + ", ".join(unsupported_requested)
            )
        action_rows: list[dict[str, Any]] = []
        for alternative in analysis.alternatives:
            row: dict[str, Any] = {
                "action": dict(alternative["action"]),
                "rank": int(alternative["rank"]),
            }
            for public_name in policy.public_metrics:
                source_name = metric_sources[public_name]
                if source_name in alternative:
                    row[public_name] = float(alternative[source_name])
            action_rows.append(row)
        mode = "single_supported" if len(accepted) == 1 else "multiple_supported"
        core = {
            "schemaVersion": 1,
            "traceHash": str(trace_hash),
            "decisionId": analysis.decision_id,
            "observerSeat": actor_evidence.observer_seat.name.lower(),
            "projectionHash": actor_evidence.projection_hash,
            "truthLabel": actor_evidence.truth_label.value,
            "confidence": float(actor_evidence.confidence),
            "recommendation": dict(actor_evidence.recommended_action),
            "acceptedActions": [dict(item) for item in accepted],
            "actionEvidence": action_rows,
            "engineCognitiveDifficulty": float(analysis.difficulty),
            "analysisHash": analysis.sha256,
            "actorEvidenceHash": canonical_sha256(actor_evidence.to_dict()),
            "policyHash": policy.sha256,
            "answerMode": mode,
        }
        return cls(
            str(trace_hash),
            analysis.decision_id,
            actor_evidence.observer_seat.name.lower(),
            actor_evidence.projection_hash,
            actor_evidence.truth_label,
            float(actor_evidence.confidence),
            dict(actor_evidence.recommended_action),
            accepted,
            tuple(action_rows),
            float(analysis.difficulty),
            analysis.sha256,
            canonical_sha256(actor_evidence.to_dict()),
            policy.sha256,
            mode,
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "traceHash": self.trace_hash,
            "decisionId": self.decision_id,
            "observerSeat": self.observer_seat,
            "projectionHash": self.projection_hash,
            "truthLabel": self.truth_label.value,
            "confidence": self.confidence,
            "recommendation": dict(self.recommendation),
            "acceptedActions": [dict(item) for item in self.accepted_actions],
            "actionEvidence": [dict(item) for item in self.action_evidence],
            "engineCognitiveDifficulty": self.engine_cognitive_difficulty,
            "analysisHash": self.analysis_hash,
            "actorEvidenceHash": self.actor_evidence_hash,
            "policyHash": self.policy_hash,
            "answerMode": self.answer_mode,
            "sha256": self.sha256,
        }
