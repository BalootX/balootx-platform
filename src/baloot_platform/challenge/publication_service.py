"""Publish exact actor challenges from checksum-verified Foundry artifacts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from baloot_platform.core.rules import RuleBook
from baloot_platform.plugins.base import PluginDescriptor

from .analysis_pipeline import TraceAnalysisResultV1
from .evidence_v3 import TruthMode, canonical_sha256
from .locator import ChallengeLocatorServiceV1, ChallengeLocatorV1
from .publication import ChallengeAnswerEvidenceV1, ChallengeAnswerPolicyV1
from .trace import CompleteGameTraceV1


@dataclass(frozen=True, slots=True)
class PublishedActorChallengeV1:
    locator: ChallengeLocatorV1
    trace_hash: str
    decision_id: str
    analysis_hash: str
    answer_evidence_hash: str
    policy_hash: str
    sha256: str

    def to_dict(self, *, trusted: bool = False) -> dict[str, Any]:
        value = {
            "schemaVersion": 1,
            "locator": self.locator.to_dict(),
            "sha256": self.sha256,
        }
        if trusted:
            value["provenance"] = {
                "traceHash": self.trace_hash,
                "decisionId": self.decision_id,
                "analysisHash": self.analysis_hash,
                "answerEvidenceHash": self.answer_evidence_hash,
                "policyHash": self.policy_hash,
            }
        return value


class ChallengePublicationServiceV1:
    descriptor = PluginDescriptor(
        id="challenge_publication_service_v1",
        kind="challenge_publisher",
        version="1.0.0",
        description="Publish governed actor challenges from verified analysis artifacts",
        capabilities=(
            "content_addressed_analysis", "critical_actor_truth",
            "governed_answer", "opaque_locator",
        ),
        resource_class="io",
        deterministic=True,
        timeout_seconds=60,
        maximum_memory_mb=256,
    )

    def __init__(
        self,
        artifact_store: Any,
        locator_service: ChallengeLocatorServiceV1,
        policy: ChallengeAnswerPolicyV1,
        *,
        player_names: Mapping[str, str] | None = None,
    ):
        self.artifact_store = artifact_store
        self.locator_service = locator_service
        self.policy = policy
        self.player_names = dict(player_names) if player_names is not None else None

    def _load_analysis(
        self,
        reference: Mapping[str, Any],
        trace: CompleteGameTraceV1,
        *,
        expected_adaptive_profile_hash: str | None = None,
    ) -> TraceAnalysisResultV1:
        if str(reference.get("ownerKind") or "") != "trace_analysis_v1":
            raise ValueError("challenge publication requires a trace analysis artifact")
        if str(reference.get("mediaType") or "") != "application/vnd.balootx.trace-analysis-v1+json":
            raise ValueError("challenge publication analysis media type is invalid")
        data = self.artifact_store.get(str(reference.get("uri") or ""))
        if len(data) != int(reference.get("bytes", -1)):
            raise ValueError("challenge publication analysis size mismatch")
        if hashlib.sha256(data).hexdigest() != str(reference.get("sha256") or ""):
            raise ValueError("challenge publication analysis artifact checksum mismatch")
        document = json.loads(data)
        if not isinstance(document, Mapping):
            raise ValueError("challenge publication analysis must be a JSON object")
        analysis = TraceAnalysisResultV1.from_dict(
            document,
            ledger=trace.decision_ledger,
            expected_adaptive_profile_hash=expected_adaptive_profile_hash,
        )
        if analysis.trace_hash != trace.sha256:
            raise ValueError("challenge publication analysis belongs to another trace")
        if analysis.sha256 != str(reference.get("ownerHash") or ""):
            raise ValueError("challenge publication analysis owner identity mismatch")
        return analysis

    def publish_actor_challenge(
        self,
        trace: CompleteGameTraceV1,
        *,
        analysis_reference: Mapping[str, Any],
        decision_id: str,
        reveal_policy: str = "after_commit",
        expires_at: int | None = None,
        answer_explanation: str | None = None,
        rules: RuleBook | None = None,
        expected_adaptive_profile_hash: str | None = None,
    ) -> PublishedActorChallengeV1:
        CompleteGameTraceV1.from_dict(trace.to_dict(trusted=True))
        analysis = self._load_analysis(
            analysis_reference,
            trace,
            expected_adaptive_profile_hash=expected_adaptive_profile_hash,
        )
        if (
            analysis.analyzer_id == "adaptive_finalist_trace_analyzer_v1"
            and analysis.analyzer_version != "1.3.0"
        ):
            raise ValueError(
                "historical adaptive finalist analysis is publication-ineligible"
            )
        if decision_id in analysis.unresolved_decision_ids:
            raise ValueError("an unresolved decision cannot be published")
        if decision_id not in analysis.laffa_evidence.critical_decision_ids:
            raise ValueError("only a verified critical decision may be published")
        decision = next(
            (item for item in analysis.decisions if item.decision_id == decision_id),
            None,
        )
        if decision is None:
            raise KeyError(decision_id)
        actor = next(
            (
                item for item in analysis.laffa_evidence.perspectives
                if item.decision_id == decision_id
                and item.truth_mode == TruthMode.ACTOR_OBSERVER
            ),
            None,
        )
        if actor is None:
            raise ValueError("critical decision lacks actor-observer evidence")
        answer = ChallengeAnswerEvidenceV1.build(
            trace_hash=trace.sha256,
            analysis=decision,
            actor_evidence=actor,
            policy=self.policy,
        )
        locator = self.locator_service.publish(
            trace,
            decision_id=decision_id,
            observer_seat=actor.observer_seat,
            prompt_contract="choose_acting_action",
            reveal_policy=reveal_policy,
            expires_at=expires_at,
            answer_evidence=answer,
            answer_explanation=answer_explanation,
            player_names=self.player_names,
            rules=rules,
        )
        core = {
            "schemaVersion": 1,
            "locatorCode": locator.code,
            "traceHash": trace.sha256,
            "decisionId": decision_id,
            "analysisHash": analysis.sha256,
            "answerEvidenceHash": answer.sha256,
            "policyHash": self.policy.sha256,
        }
        return PublishedActorChallengeV1(
            locator,
            trace.sha256,
            decision_id,
            analysis.sha256,
            answer.sha256,
            self.policy.sha256,
            canonical_sha256(core),
        )
