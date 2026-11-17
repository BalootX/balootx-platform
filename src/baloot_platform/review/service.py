from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from baloot_platform.audit.models import action_key
from .models import FeedbackReason, ReviewRole, StructuredDecisionReview


class DecisionReviewService:
    """Structured human feedback with legality and counterfactual validation.

    Reviews are evidence, not ground truth.  They are stored separately from
    engine analyses and become training-eligible only after legality, consensus,
    confidence and counterfactual gates are satisfied.
    """

    def __init__(self, learning_store: Any, audit_service: Any | None = None):
        self.store = learning_store
        self.audit_service = audit_service

    @staticmethod
    def _is_legal(preferred: Any, legal_actions: Sequence[Any]) -> bool:
        key = action_key(preferred)
        return any(action_key(item) == key for item in legal_actions)

    def context(self, decision_id: str, *, run_id: str | None = None, view_mode: str = "observer") -> dict[str, Any]:
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise KeyError(decision_id)
        return {
            "decision": decision,
            "structuredReviews": self.store.list_structured_reviews(decision_id),
            "consensus": self.store.review_consensus(decision_id),
            "runId": run_id,
            "viewMode": view_mode,
            "reasonTaxonomy": [
                {"code": reason.value, "labelAr": reason.label_ar}
                for reason in FeedbackReason
            ],
        }

    def _validate_against_engine(self, decision_id: str, preferred_action: Any) -> tuple[str, float | None, float | None, Any | None]:
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise KeyError(decision_id)
        audit = (decision.get("audits") or [None])[0]
        if audit is None and self.audit_service is not None:
            try:
                audit = self.audit_service.audit_decision(decision_id, deep=False, force=False)
            except Exception:
                audit = None
        if audit is None:
            return "unverified", None, None, None
        best = audit.get("bestAction")
        preferred_key = action_key(preferred_action)
        if preferred_key == action_key(best):
            return "supported", 0.0, 0.0, best
        alternative = next(
            (item for item in audit.get("alternatives", []) if action_key(item.get("action")) == preferred_key),
            None,
        )
        if alternative is None:
            return "unverified", None, None, best
        regret = float(alternative.get("regret", 0.0))
        win_drop = float(alternative.get("winProbabilityDrop", 0.0))
        if regret <= 1.0 and win_drop <= 0.02:
            status = "plausible"
        elif regret >= 8.0 or win_drop >= 0.15:
            status = "refuted_by_current_engine"
        else:
            status = "uncertain"
        return status, regret, win_drop, best

    def submit(
        self,
        decision_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str = "player",
        preferred_action: Any,
        confidence: float,
        reason_codes: Sequence[str],
        rationale: str,
        evidence: Sequence[Mapping[str, Any]] = (),
        view_mode: str = "observer",
        source: str = "decision_workbench",
    ) -> dict[str, Any]:
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise KeyError(decision_id)
        if not self._is_legal(preferred_action, decision.get("legalActions") or []):
            raise ValueError("preferred action was not legal at this decision")
        if view_mode not in {"observer", "public", "oracle_after_completion"}:
            raise ValueError("invalid review view mode")
        status, regret, win_drop, engine_action = self._validate_against_engine(decision_id, preferred_action)
        review = StructuredDecisionReview.create(
            decision_id=decision_id,
            reviewer_id=reviewer_id,
            reviewer_role=reviewer_role,
            preferred_action=preferred_action,
            confidence=confidence,
            reason_codes=reason_codes,
            rationale=rationale,
            evidence=evidence,
            view_mode=view_mode,
            source=source,
            engine_validation=status,
            engine_regret=regret,
            engine_win_drop=win_drop,
        )
        stored = self.store.add_structured_review(review.to_dict())
        self.store.add_feedback(
            decision_id,
            preferred_action=preferred_action,
            engine_action=engine_action,
            confidence=confidence,
            rationale=rationale,
            source=source,
            reviewer=reviewer_id,
            interestingness=max(0.0, min(1.0, (regret or 0.0) / 20.0 + (win_drop or 0.0))),
            model_error=1.0 if status in {"supported", "plausible"} and action_key(preferred_action) != action_key(engine_action) else 0.0,
        )
        return {
            "review": stored,
            "consensus": self.store.review_consensus(decision_id),
            "trainingPolicy": {
                "feedbackIsEvidenceNotTruth": True,
                "automaticProductionPromotion": False,
                "requiresLegalAction": True,
                "requiresConsensusOrExpertReview": True,
            },
        }

    def consensus(self, decision_id: str) -> dict[str, Any]:
        return self.store.review_consensus(decision_id)
