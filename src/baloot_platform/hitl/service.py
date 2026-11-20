from __future__ import annotations

import hashlib
import hmac
import math
import secrets
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from baloot_platform.audit.models import action_key
from baloot_platform.review.models import FeedbackReason
from baloot_platform.version import __version__

from baloot_platform.identity.ownership import owns_subject

from .models import (
    AdjudicationRecord,
    BlindAssignment,
    ChallengerEvaluation,
    ConsensusSnapshot,
    ConsentRecord,
    GovernedReviewCase,
    ImprovementEvaluation,
    ImprovementProposal,
    ModerationFlag,
    ReviewSubmission,
    ReviewerProfile,
    TrainingCandidate,
    ValidationResult,
    identifier,
    utc_now,
)


def _cluster(value: str, secret: bytes) -> str:
    if not value:
        return ""
    return hmac.new(secret, value.encode("utf-8"), hashlib.sha256).hexdigest()[:24]


def _wilson_lower(success_weight: float, total_weight: float, z: float = 1.96) -> float:
    if total_weight <= 0:
        return 0.0
    p = max(0.0, min(1.0, success_weight / total_weight))
    n = max(1.0, total_weight)
    denominator = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, (center - margin) / denominator)


class ReviewGovernanceServiceV18:
    """Governed Human-in-the-Loop review and engine-improvement lifecycle.

    Reviews are evidence. They cannot mutate rules, scoring, or the production
    engine. Accepted labels still require independent challenger validation,
    release evidence, explicit promotion, and rollback readiness.
    """

    REASON_TAXONOMY = tuple(
        {"code": reason.value, "labelAr": reason.label_ar} for reason in FeedbackReason
    )
    IMPROVEMENT_TARGETS = {
        "model_policy",
        "search_configuration",
        "belief_priors",
        "conventions",
        "engine_bug",
        "rule_profile",
    }
    CORRECTION_TAXONOMY = (
        "rule_defect",
        "engine_legality_scoring_defect",
        "analysis_defect",
        "tactical_convention",
        "perspective_interpretation",
        "explanation_defect",
        "metadata_video_defect",
    )

    def __init__(
        self,
        store: Any,
        learning_store: Any,
        audit_service: Any | None,
        profile: Mapping[str, Any],
        *,
        cluster_secret: bytes | None = None,
    ):
        self.store = store
        self.learning_store = learning_store
        self.audit_service = audit_service
        self.profile = dict(profile)
        self.cluster_secret = cluster_secret or secrets.token_bytes(32)

    def _decision(self, decision_id: str) -> dict[str, Any]:
        decision = self.learning_store.get_decision(decision_id)
        if decision is None:
            raise KeyError(decision_id)
        return decision

    def create_case(
        self,
        decision_id: str,
        *,
        created_by: str,
        correction_type: str = "analysis_defect",
        high_impact: bool = False,
        completed: bool = True,
    ) -> dict[str, Any]:
        if correction_type not in self.CORRECTION_TAXONOMY:
            raise ValueError("unsupported correction type")
        decision = self._decision(decision_id)
        public_context = {
            "decisionId": decision_id,
            "decisionKind": decision.get("decisionKind"),
            "actor": decision.get("actor"),
            "phase": decision.get("phase"),
            "publicState": decision.get("publicState") or {},
            "selectedAction": decision.get("selectedAction"),
            "createdAt": decision.get("createdAt"),
        }
        analyses = decision.get("analyses") or []
        production_action = None
        bundle_hash = ""
        if analyses:
            result = analyses[0].get("result") or {}
            production_action = result.get("recommendedCard") or result.get("recommendedAction")
            bundle_hash = str(analyses[0].get("bundleHash") or "")
        case = GovernedReviewCase.create(
            decision_id=decision_id,
            created_by=created_by,
            public_context=public_context,
            legal_actions=tuple(decision.get("legalActions") or ()),
            production_action=production_action,
            engine_version=__version__,
            bundle_hash=bundle_hash,
            correction_type=correction_type,
            high_impact=high_impact,
            completed=completed,
        )
        return self.store.put_case(case)

    def case(self, case_id: str, *, blind: bool = False) -> dict[str, Any]:
        case = self.store.get_case(case_id)
        if case is None:
            raise KeyError(case_id)
        result = dict(case)
        if blind:
            result.pop("productionAction", None)
            result["blind"] = True
        result["submissions"] = [] if blind else self.store.list_submissions(case_id)
        result["consensus"] = None if blind else self.store.latest_consensus(case_id)
        result["moderationFlags"] = [] if blind else self.store.list_flags(case_id)
        result["verification"] = self.store.verify_events("case", case_id)
        return result

    def list_cases(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list_cases(limit=limit)

    def assign(
        self,
        case_id: str,
        *,
        reviewer_id: str,
        device_id: str = "",
        network_id: str = "",
        ttl_hours: int = 72,
    ) -> dict[str, Any]:
        if self.store.get_case(case_id) is None:
            raise KeyError(case_id)
        existing = self.store.find_assignment(case_id, reviewer_id)
        if existing and existing.get("status") == "active":
            return {
                **existing,
                "assignmentToken": None,
                "reused": True,
                "blindContext": self.case(case_id, blind=True),
                "reasonTaxonomy": list(self.REASON_TAXONOMY),
            }
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        assigned_at = utc_now()
        expires_at = (
            datetime.now(timezone.utc) + timedelta(hours=max(1, int(ttl_hours)))
        ).isoformat(timespec="seconds")
        assignment = BlindAssignment(
            identifier(
                "ASN",
                {"case": case_id, "reviewer": reviewer_id, "at": assigned_at},
            ),
            case_id,
            reviewer_id,
            token_hash,
            _cluster(device_id, self.cluster_secret),
            _cluster(network_id, self.cluster_secret),
            assigned_at=assigned_at,
            expires_at=expires_at,
        )
        stored = self.store.put_assignment(assignment)
        return {
            **stored,
            "assignmentToken": token,
            "blindContext": self.case(case_id, blind=True),
            "reasonTaxonomy": list(self.REASON_TAXONOMY),
        }

    @staticmethod
    def _is_legal(preferred_action: Any, legal_actions: Sequence[Any]) -> bool:
        preferred_key = action_key(preferred_action)
        return any(action_key(item) == preferred_key for item in legal_actions)

    @staticmethod
    def _information_set_safe(
        claims: Sequence[Mapping[str, Any]],
        evidence: Sequence[Mapping[str, Any]],
        view_mode: str,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if view_mode not in {"observer", "public"}:
            reasons.append("view_mode_not_observer_safe")
        forbidden = {
            "hiddenHand",
            "hiddenCards",
            "oracleState",
            "partnerHand",
            "opponentHand",
            "privateBotTrace",
            "completeState",
        }

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if str(key) in forbidden:
                        reasons.append("private_information_reference")
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(list(claims))
        walk(list(evidence))
        return not reasons, reasons

    def _engine_check(
        self, case: Mapping[str, Any], preferred_action: Any
    ) -> tuple[str, float | None, float | None]:
        decision = self._decision(str(case["decisionId"]))
        audits = decision.get("audits") or []
        audit = audits[0] if audits else None
        if audit is None and self.audit_service is not None:
            try:
                audit = self.audit_service.audit_decision(
                    str(case["decisionId"]), deep=False, force=False
                )
            except Exception:
                audit = None
        if audit is None:
            return "unverified", None, None
        if action_key(preferred_action) == action_key(audit.get("bestAction")):
            return "supported", 0.0, 0.0
        alternative = next(
            (
                item
                for item in audit.get("alternatives", [])
                if action_key(item.get("action")) == action_key(preferred_action)
            ),
            None,
        )
        if alternative is None:
            return "unverified", None, None
        regret = float(alternative.get("regret", 0.0))
        win_drop = float(alternative.get("winProbabilityDrop", 0.0))
        if regret <= 1.0 and win_drop <= 0.02:
            return "plausible", regret, win_drop
        if regret >= 8.0 or win_drop >= 0.15:
            return "refuted_by_current_engine", regret, win_drop
        return "uncertain", regret, win_drop

    def submit(
        self,
        case_id: str,
        *,
        assignment_id: str,
        assignment_token: str,
        reviewer_id: str,
        preferred_action: Any,
        confidence: float,
        reason_codes: Sequence[str],
        rationale_ar: str,
        claims: Sequence[Mapping[str, Any]] = (),
        evidence: Sequence[Mapping[str, Any]] = (),
        view_mode: str = "observer",
        hidden_cards_seen: bool = False,
        final_result_seen: bool = False,
        disclosure_timing: str = "before_reveal",
    ) -> dict[str, Any]:
        case = self.store.get_case(case_id)
        assignment = self.store.get_assignment(assignment_id)
        if case is None:
            raise KeyError(case_id)
        if (
            assignment is None
            or assignment.get("caseId") != case_id
            or not owns_subject(reviewer_id, assignment.get("reviewerId"))
        ):
            raise ValueError("invalid blind assignment")
        if assignment.get("status") != "active":
            raise ValueError("assignment is not active")
        expected_hash = assignment.get("tokenHash")
        supplied_hash = hashlib.sha256(assignment_token.encode("utf-8")).hexdigest()
        if not assignment_token or not expected_hash or not hmac.compare_digest(
            str(expected_hash), supplied_hash
        ):
            raise ValueError("invalid assignment token")

        # Disclosure status is server-derived when authoritative reveal metadata
        # exists. A client cannot make an after-reveal review training-safe by
        # merely submitting false booleans.
        hidden_cards_seen = bool(hidden_cards_seen or case.get("hiddenRevealedAt"))
        final_result_seen = bool(final_result_seen or case.get("resultRevealedAt"))
        if hidden_cards_seen or final_result_seen:
            disclosure_timing = "after_reveal"
        legal = self._is_legal(preferred_action, case.get("legalActions") or [])
        information_safe, information_reasons = self._information_set_safe(
            claims, evidence, view_mode
        )
        hindsight_safe = (
            not hidden_cards_seen
            and not final_result_seen
            and disclosure_timing in {"before_reveal", "during_play"}
        )
        if legal:
            engine_status, regret, win_drop = self._engine_check(case, preferred_action)
        else:
            engine_status, regret, win_drop = "illegal", None, None
        submission = ReviewSubmission.create(
            case_id=case_id,
            assignment_id=assignment_id,
            reviewer_id=reviewer_id,
            preferred_action=preferred_action,
            confidence=confidence,
            reason_codes=reason_codes,
            rationale_ar=rationale_ar,
            claims=claims,
            evidence=evidence,
            view_mode=view_mode,
            hidden_cards_seen=hidden_cards_seen,
            final_result_seen=final_result_seen,
            disclosure_timing=disclosure_timing,
        )
        stored = self.store.put_submission(submission)
        self.store.update_assignment(assignment_id, status="submitted")
        reasons: list[str] = []
        if not legal:
            reasons.append("illegal_action")
        reasons.extend(information_reasons)
        if not hindsight_safe:
            reasons.append("hindsight_disclosed")
        validation = ValidationResult(
            identifier("VAL", {"submission": submission.submission_id}),
            submission.submission_id,
            legal,
            information_safe,
            hindsight_safe,
            engine_status,
            regret,
            win_drop,
            reasons=tuple(reasons),
        )
        validation_value = self.store.put_validation(validation)
        status = "engine_checked" if legal and information_safe else "blocked"
        self.store.update_submission(submission.submission_id, status=status)
        flags = self._detect_collusion(case_id)
        return {
            "submission": {**stored, "status": status},
            "validation": validation_value,
            "moderationFlags": flags,
            "consensus": self.compute_consensus(case_id),
            "automaticProductionPromotion": False,
        }

    def _profile(self, subject_id: str) -> ReviewerProfile:
        payload = self.store.get_profile(subject_id)
        if payload is None:
            profile = ReviewerProfile(subject_id)
            self.store.put_profile(profile)
            return profile
        return ReviewerProfile(
            subject_id=payload["subjectId"],
            verified_role=payload.get("verifiedRole", "player"),
            role_verified=bool(payload.get("roleVerified")),
            legality_accuracy=float(payload.get("legalityAccuracy", 0.5)),
            information_set_discipline=float(
                payload.get("informationSetDiscipline", 0.5)
            ),
            strategic_accuracy=float(payload.get("strategicAccuracy", 0.5)),
            calibration=float(payload.get("calibration", 0.5)),
            independence=float(payload.get("independence", 0.5)),
            adjudicated_cases=int(payload.get("adjudicatedCases", 0)),
            updated_at=payload.get("updatedAt", utc_now()),
        )

    def set_reviewer_profile(self, subject_id: str, **values: Any) -> dict[str, Any]:
        current = self._profile(subject_id)
        allowed = {
            key: value
            for key, value in values.items()
            if key
            in {
                "verified_role",
                "role_verified",
                "legality_accuracy",
                "information_set_discipline",
                "strategic_accuracy",
                "calibration",
                "independence",
                "adjudicated_cases",
            }
        }
        profile = replace(current, **allowed, updated_at=utc_now())
        return self.store.put_profile(profile)

    def get_reviewer_profile(self, subject_id: str) -> dict[str, Any]:
        return self._profile(subject_id).to_dict()

    def set_consent(self, subject_id: str, **values: Any) -> dict[str, Any]:
        existing = self.store.get_consent(subject_id) or {}
        revoke = bool(values.get("revoke", False))
        revoked_at = utc_now() if revoke else values.get("revoked_at")
        stored_subject = existing.get("subjectId", subject_id)
        consent = ConsentRecord(
            subject_id=stored_subject,
            version=str(values.get("version", existing.get("version", "1"))),
            allow_training=False if revoke else bool(values.get("allow_training", existing.get("allowTraining", False))),
            allow_research=False if revoke else bool(values.get("allow_research", existing.get("allowResearch", False))),
            allow_public_challenge=False if revoke else bool(values.get("allow_public_challenge", existing.get("allowPublicChallenge", False))),
            allow_voice=False if revoke else bool(values.get("allow_voice", existing.get("allowVoice", False))),
            allow_identity_attribution=False if revoke else bool(values.get("allow_identity_attribution", existing.get("allowIdentityAttribution", False))),
            allow_private_coaching=False if revoke else bool(values.get("allow_private_coaching", existing.get("allowPrivateCoaching", True))),
            effective_at=str(existing.get("effectiveAt") or utc_now()),
            revoked_at=revoked_at,
        )
        value = self.store.put_consent(consent)
        if revoke:
            value["withdrawalImpact"] = self._invalidate_consent_lineage(subject_id)
        self.store.append_event(
            "consent", stored_subject, "consent_revoked" if revoke else "consent_updated",
            {
                "subjectId": stored_subject,
                "version": value.get("version"),
                "revokedAt": value.get("revokedAt"),
                "withdrawalImpact": value.get("withdrawalImpact"),
            },
        )
        return value

    def _invalidate_consent_lineage(self, subject_id: str) -> dict[str, Any]:
        """Fail closed across all known descendants of withdrawn human evidence."""
        candidate_ids: list[str] = []
        case_ids: set[str] = set()
        for candidate in self.store.list_candidates():
            if not any(owns_subject(subject_id, value) for value in candidate.get("consentSubjects") or ()):
                continue
            candidate_id = str(candidate["candidateId"])
            case_id = str(candidate["caseId"])
            candidate_ids.append(candidate_id)
            case_ids.add(case_id)
            self.store.update_candidate(
                candidate_id,
                status="consent_withdrawn",
                invalidatedAt=utc_now(),
                invalidatedByConsentSubject=subject_id,
                automaticProductionPromotion=False,
            )
            self.store.append_event(
                "case",
                case_id,
                "training_lineage_invalidated",
                {"candidateId": candidate_id, "reason": "consent_withdrawn", "subjectId": subject_id},
            )

        dataset_ids: list[str] = []
        model_ids: list[str] = []
        list_datasets = getattr(self.learning_store, "list_datasets", None)
        if callable(list_datasets):
            for dataset in list_datasets():
                manifest = dict(dataset.get("manifest") or dataset.get("manifestData") or {})
                metadata = dict(manifest.get("metadata") or {})
                lineage = list(metadata.get("governedHitlLineage") or [])
                if any(
                    any(owns_subject(subject_id, value) for value in item.get("consentSubjects") or ())
                    or str(item.get("candidateId")) in candidate_ids
                    for item in lineage
                ):
                    dataset_ids.append(str(dataset.get("id")))
        list_models = getattr(self.learning_store, "list_models", None)
        if callable(list_models):
            for model in list_models():
                if str(model.get("datasetId")) in dataset_ids:
                    model_ids.append(str(model.get("modelId")))

        impact = {
            "schemaVersion": 1,
            "subjectId": subject_id,
            "candidateIds": sorted(candidate_ids),
            "caseIds": sorted(case_ids),
            "datasetIds": sorted(dataset_ids),
            "modelIds": sorted(model_ids),
            "futureTrainingEligible": False,
            "productionMutationPerformed": False,
            "recordedAt": utc_now(),
        }
        self.store.append_event("consent", subject_id, "lineage_impact_recorded", impact)
        return impact

    def get_consent(self, subject_id: str) -> dict[str, Any]:
        value = self.store.get_consent(subject_id)
        return value or ConsentRecord(subject_id).to_dict()

    def _detect_collusion(self, case_id: str) -> list[dict[str, Any]]:
        submissions = self.store.list_submissions(case_id)
        buckets: dict[str, list[str]] = defaultdict(list)
        for submission in submissions:
            normalized = " ".join(
                str(submission.get("rationaleAr", "")).lower().split()
            )
            if normalized:
                key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
                buckets[key].append(submission["submissionId"])
        for similarity_key, submission_ids in buckets.items():
            if len(submission_ids) < 2:
                continue
            existing = [
                item
                for item in self.store.list_flags(case_id)
                if item.get("code") == "copied_rationale"
                and set(item.get("submissionIds", [])) == set(submission_ids)
            ]
            if not existing:
                flag = ModerationFlag(
                    identifier(
                        "FLG",
                        {
                            "case": case_id,
                            "code": "copy",
                            "ids": submission_ids,
                        },
                    ),
                    case_id,
                    tuple(submission_ids),
                    "copied_rationale",
                    "medium",
                    {"similarityKey": similarity_key},
                )
                self.store.put_flag(flag)
        return self.store.list_flags(case_id, open_only=True)

    def compute_consensus(self, case_id: str) -> dict[str, Any]:
        case = self.store.get_case(case_id)
        if case is None:
            raise KeyError(case_id)
        submissions = self.store.list_submissions(case_id)
        valid: list[dict[str, Any]] = []
        for submission in submissions:
            validation = self.store.get_validation_for_submission(
                submission["submissionId"]
            )
            if (
                validation
                and validation.get("legal")
                and validation.get("informationSetSafe")
                and validation.get("hindsightSafe")
            ):
                valid.append(submission)

        by_action: dict[str, list[tuple[dict[str, Any], float, ReviewerProfile]]] = defaultdict(list)
        total_weight = 0.0
        reviewer_ids: set[str] = set()
        clusters: set[tuple[str, str]] = set()
        expert_coverage = False
        for submission in valid:
            profile = self._profile(submission["reviewerId"])
            weight = (
                profile.reputation
                * profile.role_prior
                * max(0.2, float(submission.get("confidence", 0.5)))
            )
            total_weight += weight
            reviewer_ids.add(submission["reviewerId"])
            by_action[action_key(submission["preferredAction"])].append(
                (submission, weight, profile)
            )
            assignment = self.store.get_assignment(submission["assignmentId"]) or {}
            clusters.add(
                (
                    assignment.get("deviceClusterHash")
                    or submission["reviewerId"],
                    assignment.get("networkClusterHash")
                    or submission["reviewerId"],
                )
            )
            expert_coverage = expert_coverage or (
                profile.role_verified
                and profile.verified_role in {"expert", "referee"}
            )

        best_key: str | None = None
        support_weight = 0.0
        for key, rows in by_action.items():
            current = sum(weight for _submission, weight, _profile in rows)
            if current > support_weight:
                best_key, support_weight = key, current
        preferred_action = (
            by_action[best_key][0][0]["preferredAction"] if best_key else None
        )
        expert_opposition = False
        if best_key:
            for key, rows in by_action.items():
                if key == best_key:
                    continue
                if any(
                    profile.role_verified
                    and profile.verified_role in {"expert", "referee"}
                    and float(submission.get("confidence", 0)) >= 0.7
                    for submission, _weight, profile in rows
                ):
                    expert_opposition = True
                    break

        policy = dict(self.profile.get("consensus") or self.profile.get("review") or {})
        minimum_reviewers = int(policy.get("minimum_independent_reviewers", 5))
        minimum_support = float(policy.get("minimum_weighted_support", 0.70))
        minimum_lower_bound = float(
            policy.get("minimum_wilson_lower_bound", policy.get("minimum_wilson_lower", 0.55))
        )
        minimum_clusters = int(policy.get("minimum_cluster_diversity", policy.get("minimum_independent_clusters", 3)))
        require_expert = bool(policy.get("require_verified_expert", policy.get("require_verified_expert_or_two_trusted", True)))
        support_ratio = support_weight / total_weight if total_weight else 0.0
        lower_bound = _wilson_lower(support_weight, total_weight)
        reasons: list[str] = []
        if len(reviewer_ids) < minimum_reviewers:
            reasons.append("insufficient_reviewers")
        if support_ratio < minimum_support:
            reasons.append("insufficient_weighted_support")
        if lower_bound < minimum_lower_bound:
            reasons.append("confidence_bound_too_low")
        if len(clusters) < minimum_clusters:
            reasons.append("insufficient_cluster_diversity")
        if require_expert and not expert_coverage:
            reasons.append("verified_expert_required")
        if expert_opposition:
            reasons.append("expert_opposition")
        if self.store.list_flags(case_id, open_only=True):
            reasons.append("unresolved_moderation")
        eligible = not reasons
        consensus = ConsensusSnapshot(
            identifier(
                "CON",
                {
                    "case": case_id,
                    "count": len(submissions),
                    "preferred": preferred_action,
                    "at": utc_now(),
                },
            ),
            case_id,
            preferred_action,
            len(submissions),
            len(reviewer_ids),
            round(support_ratio, 6),
            round(lower_bound, 6),
            expert_coverage,
            expert_opposition,
            len(clusters),
            eligible,
            tuple(reasons),
        )
        value = self.store.put_consensus(consensus)
        self.store.update_case(
            case_id,
            status="consensus_candidate" if eligible else "blind_peer_review",
        )
        return value

    def open_moderation_flag(
        self,
        case_id: str,
        *,
        submission_ids: Sequence[str],
        code: str,
        severity: str = "medium",
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.store.get_case(case_id) is None:
            raise KeyError(case_id)
        flag = ModerationFlag(
            identifier(
                "FLG",
                {
                    "case": case_id,
                    "code": code,
                    "ids": list(submission_ids),
                    "at": utc_now(),
                },
            ),
            case_id,
            tuple(submission_ids),
            code,
            severity,
            dict(details or {}),
        )
        return self.store.put_flag(flag)

    def resolve_moderation_flag(
        self, flag_id: str, *, resolved_by: str, resolution: str
    ) -> dict[str, Any]:
        return self.store.update_flag(
            flag_id,
            status="resolved",
            resolved_at=utc_now(),
            resolved_by=resolved_by,
            resolution=resolution,
        )

    def adjudicate(
        self,
        case_id: str,
        *,
        adjudicator_id: str,
        outcome: str,
        preferred_action: Any | None,
        rationale_ar: str,
        certification_manifest: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        case = self.store.get_case(case_id)
        if case is None:
            raise KeyError(case_id)
        if outcome not in {"accepted", "rejected", "disputed"}:
            raise ValueError("unsupported adjudication outcome")
        if outcome == "accepted":
            consensus = self.store.latest_consensus(case_id)
            if not consensus or not consensus.get("eligible"):
                raise ValueError("accepted adjudication requires eligible blind consensus")
            if preferred_action is None or not self._is_legal(
                preferred_action, case.get("legalActions") or ()
            ):
                raise ValueError("accepted adjudication requires a legal preferred action")
            if action_key(preferred_action) != action_key(consensus.get("preferredAction")):
                raise ValueError("accepted adjudication must preserve the supported blind consensus")
        adjudication = AdjudicationRecord(
            identifier(
                "ADJ", {"case": case_id, "by": adjudicator_id, "at": utc_now()}
            ),
            case_id,
            adjudicator_id,
            outcome,
            preferred_action,
            rationale_ar,
            certification_manifest=certification_manifest,
        )
        value = self.store.put_adjudication(adjudication)
        status = {"accepted": "accepted", "rejected": "rejected"}.get(
            outcome, "disputed"
        )
        self.store.update_case(case_id, status=status)
        return value

    def create_training_candidate(self, case_id: str) -> dict[str, Any]:
        case = self.store.get_case(case_id)
        consensus = self.store.latest_consensus(case_id)
        adjudication = self.store.latest_adjudication(case_id)
        if case is None:
            raise KeyError(case_id)
        if not consensus or not consensus.get("eligible"):
            raise ValueError("case has no eligible governed consensus")
        if self.store.list_flags(case_id, open_only=True):
            raise ValueError("unresolved moderation flags")
        if adjudication and adjudication.get("outcome") != "accepted":
            raise ValueError("adjudication does not accept this correction")
        if case.get("highImpact") and not adjudication:
            raise ValueError("high-impact correction requires expert adjudication")
        submissions: list[dict[str, Any]] = []
        for item in self.store.list_submissions(case_id):
            validation = self.store.get_validation_for_submission(item["submissionId"])
            if not validation:
                continue
            if not all(
                bool(validation.get(key))
                for key in ("legal", "informationSetSafe", "hindsightSafe")
            ):
                continue
            submissions.append(item)
        preferred_key = action_key(consensus.get("preferredAction"))
        supporting = [
            item for item in submissions
            if action_key(item.get("preferredAction")) == preferred_key
        ]
        if not supporting:
            raise ValueError("eligible consensus has no validated supporting submissions")
        subjects = tuple(sorted({item["reviewerId"] for item in supporting}))
        for subject_id in subjects:
            consent = self.store.get_consent(subject_id)
            if (
                not consent
                or not consent.get("allowTraining")
                or consent.get("revokedAt")
            ):
                raise ValueError(f"training consent missing for {subject_id}")
        labels = {
            "human_preference": [
                {
                    "reviewerId": item["reviewerId"],
                    "action": item["preferredAction"],
                }
                for item in supporting
            ],
            "review_consensus": consensus,
            "expert_adjudication": adjudication,
            "production_engine_action": case.get("productionAction"),
            "actual_outcome": None,
            "deep_search_label": None,
            "exact_solver_label": None,
            "correction_type": case.get("correctionType", "analysis_defect"),
            "decision_id": case.get("decisionId"),
            "case_hash": case.get("caseHash"),
            "validated_submission_ids": [item["submissionId"] for item in supporting],
        }
        action = (adjudication or {}).get("preferredAction") or consensus.get(
            "preferredAction"
        )
        candidate_id = identifier(
            "TRN",
            {
                "case": case_id,
                "action": action,
                "validatedSubmissions": labels["validated_submission_ids"],
                "caseHash": case.get("caseHash"),
            },
        )
        existing_candidate = self.store.get_candidate(candidate_id)
        if existing_candidate is not None:
            if existing_candidate.get("status") == "consent_withdrawn":
                raise ValueError("candidate lineage was permanently invalidated by consent withdrawal")
            self._require_current_candidate_consent(existing_candidate)
            return existing_candidate
        candidate = TrainingCandidate(
            candidate_id,
            case_id,
            labels,
            action,
            subjects,
        )
        value = self.store.put_candidate(candidate)
        self.store.update_case(case_id, status="training_candidate")
        return value

    def record_challenger_evaluation(
        self,
        candidate_id: str,
        *,
        validator_id: str,
        validator_type: str,
        status: str,
        metrics: Mapping[str, Any],
        independent: bool = True,
    ) -> dict[str, Any]:
        candidate = self.store.get_candidate(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        self._require_current_candidate_consent(candidate)
        if candidate.get("status") not in {"training_candidate", "challenger_evaluated"}:
            raise ValueError("training candidate is not open for challenger evaluation")
        if status not in {"passed", "failed"}:
            raise ValueError("challenger evaluation status must be passed or failed")
        evaluation = ChallengerEvaluation(
            identifier(
                "CEV",
                {
                    "candidate": candidate_id,
                    "validator": validator_id,
                    "type": validator_type,
                },
            ),
            candidate_id,
            validator_id,
            validator_type,
            status,
            dict(metrics),
            independent,
        )
        value = self.store.put_challenger_evaluation(evaluation)
        self.store.update_candidate(candidate_id, status="challenger_evaluated")
        self.store.append_event(
            "case",
            str(candidate["caseId"]),
            "challenger_evaluated",
            {"candidateId": candidate_id, "evaluationId": value["evaluationId"], "status": status},
        )
        return {**value, "automaticProductionPromotion": False}

    def accept_training_candidate(
        self, candidate_id: str, *, accepted_by: str
    ) -> dict[str, Any]:
        candidate = self.store.get_candidate(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        self._require_current_candidate_consent(candidate)
        if candidate.get("status") == "accepted":
            return candidate
        if candidate.get("status") != "challenger_evaluated":
            raise ValueError("training candidate has not completed challenger evaluation")
        evaluations = self.store.list_challenger_evaluations(candidate_id)
        if any(
            item.get("independent") and item.get("status") != "passed"
            for item in evaluations
        ):
            raise ValueError("an independent challenger evaluation failed")
        independent_validators = {
            item["validatorId"]
            for item in evaluations
            if item.get("independent") and item.get("status") == "passed"
        }
        challenger_policy = self.profile.get("challenger") or self.profile.get("training") or {}
        required = int(
            challenger_policy.get(
                "minimum_independent_challenger_evaluations",
                challenger_policy.get("minimum_independent_validators", 2),
            )
        )
        if len(independent_validators) < required:
            raise ValueError("insufficient independent challenger evaluations")
        accepted = self.store.update_candidate(
            candidate_id,
            status="accepted",
            acceptedBy=accepted_by,
            acceptedAt=utc_now(),
            automaticProductionPromotion=False,
        )
        self.store.append_event(
            "case",
            str(candidate["caseId"]),
            "training_candidate_accepted",
            {"candidateId": candidate_id, "acceptedBy": accepted_by, "automaticProductionPromotion": False},
        )
        return accepted

    def _require_current_candidate_consent(self, candidate: Mapping[str, Any]) -> None:
        if candidate.get("status") == "consent_withdrawn":
            raise ValueError("training candidate consent was withdrawn")
        for subject_id in candidate.get("consentSubjects") or ():
            consent = self.store.get_consent(str(subject_id))
            if not consent or not consent.get("allowTraining") or consent.get("revokedAt"):
                raise ValueError(f"training consent missing for {subject_id}")

    def create_improvement_proposal(
        self,
        *,
        created_by: str,
        target: str,
        title_ar: str,
        rationale_ar: str,
        source_case_ids: Sequence[str],
        requested_change: Mapping[str, Any],
    ) -> dict[str, Any]:
        if target not in self.IMPROVEMENT_TARGETS:
            raise ValueError("unsupported improvement target")
        if target == "rule_profile" and not requested_change.get(
            "certificationManifestRequired", True
        ):
            raise ValueError("rule-profile proposals require external certification")
        proposal = ImprovementProposal(
            identifier(
                "IMP",
                {
                    "by": created_by,
                    "target": target,
                    "title": title_ar,
                    "at": utc_now(),
                },
            ),
            created_by,
            target,
            title_ar,
            rationale_ar,
            tuple(source_case_ids),
            dict(requested_change),
        )
        return self.store.put_proposal(proposal)

    def evaluate_improvement(
        self,
        proposal_id: str,
        *,
        validator_id: str,
        status: str,
        evidence: Mapping[str, Any],
        independent: bool = True,
    ) -> dict[str, Any]:
        if self.store.get_proposal(proposal_id) is None:
            raise KeyError(proposal_id)
        evaluation = ImprovementEvaluation(
            identifier(
                "IEV", {"proposal": proposal_id, "validator": validator_id}
            ),
            proposal_id,
            validator_id,
            status,
            dict(evidence),
            independent,
        )
        return self.store.put_improvement_evaluation(evaluation)

    def approve_improvement(
        self, proposal_id: str, *, approved_by: str, role: str
    ) -> dict[str, Any]:
        proposal = self.store.get_proposal(proposal_id)
        if proposal is None:
            raise KeyError(proposal_id)
        if role not in {"expert", "referee", "admin"}:
            raise PermissionError("expert or administrative approval required")
        evaluations = self.store.list_improvement_evaluations(proposal_id)
        independent_validators = {
            item["validatorId"]
            for item in evaluations
            if item.get("independent") and item.get("status") == "passed"
        }
        required = int(
            (self.profile.get("improvement") or {}).get(
                "minimum_independent_validators", 2
            )
        )
        if len(independent_validators) < required:
            raise ValueError("insufficient independent proposal evaluations")
        return self.store.update_proposal(
            proposal_id,
            status="approved_for_challenger",
            approvedBy=approved_by,
            approvedAt=utc_now(),
            automaticProductionApply=False,
        )

    def finalize_improvement(
        self,
        proposal_id: str,
        *,
        release_governor_id: str,
        change_manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        proposal = self.store.get_proposal(proposal_id)
        if proposal is None:
            raise KeyError(proposal_id)
        if proposal.get("status") != "approved_for_challenger":
            raise ValueError("proposal is not approved for challenger packaging")
        required = {
            "artifactId",
            "bundleHash",
            "rollbackManifest",
            "regressionPassed",
            "shadowPassed",
            "canaryPassed",
        }
        missing = required - set(change_manifest)
        if missing:
            raise ValueError(
                "incomplete change manifest: " + ",".join(sorted(missing))
            )
        if not all(
            bool(change_manifest[key])
            for key in ("regressionPassed", "shadowPassed", "canaryPassed")
        ):
            raise ValueError("release gates did not pass")
        if proposal.get("target") == "rule_profile":
            certification = change_manifest.get("certificationManifest")
            if not isinstance(certification, dict) or not certification.get(
                "externalSignatures"
            ):
                raise ValueError(
                    "externally signed certification manifest is required"
                )
        return self.store.update_proposal(
            proposal_id,
            status="release_change_ready",
            releaseGovernorId=release_governor_id,
            finalizedAt=utc_now(),
            changeManifest=dict(change_manifest),
            releaseChangeReady=True,
            automaticProductionApply=False,
        )

    def list_proposals(self) -> list[dict[str, Any]]:
        return self.store.list_proposals()

    def proposal(self, proposal_id: str) -> dict[str, Any]:
        proposal = self.store.get_proposal(proposal_id)
        if proposal is None:
            raise KeyError(proposal_id)
        return {
            **proposal,
            "evaluations": self.store.list_improvement_evaluations(proposal_id),
            "verification": self.store.verify_events("proposal", proposal_id),
        }
