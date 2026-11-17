"""Deterministic trace-to-candidate Foundry analysis funnel."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from baloot_platform.core.enums import Seat
from baloot_platform.plugins.base import PluginDescriptor

from .analysis_pipeline import (
    CascadedTraceAnalyzerV1,
    StagedTraceAnalyzerV1,
    TraceAnalysisResultV1,
)
from .evidence_v3 import (
    DecisionPerspectiveEvidenceV3,
    LaffaEvidenceV2,
    TruthMode,
    canonical_sha256,
)
from .oracle import (
    ActualDealActionOracleV1,
    ActualDealOracleVerifierV1,
    TacticalVerificationResultV1,
)
from .perspective_pipeline import build_observer_projection
from .profiles import (
    SelectionProfileV1,
    SurroundingCompetenceEvidenceV1,
)
from .quality_diversity import (
    QualityDiversityCandidateV1,
    QualityDiversityProfileV1,
    QualityDiversitySelectionV1,
    QualityDiversitySelectorV1,
)
from .replay import TraceDecisionProjectionStoreV1
from .scoring import ProfileChallengeScorerV1
from .tactical_integration import integrate_verified_tactical_evidence
from .trace import CompleteGameTraceV1


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class FoundryAnalysisFunnelResultV1:
    analysis_profile_hash: str
    selection_profile_hash: str
    diversity_profile_hash: str
    plugins: tuple[Mapping[str, str], ...]
    trace_hashes: tuple[str, ...]
    analysis_hashes: tuple[str, ...]
    tactical_verification_hashes: tuple[str, ...]
    verified_laffa_evidence_hashes: tuple[str, ...]
    candidate_hashes: tuple[str, ...]
    selection: QualityDiversitySelectionV1
    artifact_references: tuple[Mapping[str, Any], ...]
    sha256: str
    surrounding_competence_evidence_hashes: tuple[str, ...] = ()
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schemaVersion": self.schema_version,
            "analysisProfileHash": self.analysis_profile_hash,
            "selectionProfileHash": self.selection_profile_hash,
            "diversityProfileHash": self.diversity_profile_hash,
            "plugins": [dict(item) for item in self.plugins],
            "traceHashes": list(self.trace_hashes),
            "analysisHashes": list(self.analysis_hashes),
            "tacticalVerificationHashes": list(self.tactical_verification_hashes),
            "verifiedLaffaEvidenceHashes": list(self.verified_laffa_evidence_hashes),
            "candidateHashes": list(self.candidate_hashes),
            "selection": self.selection.to_dict(),
            "artifactReferences": [dict(item) for item in self.artifact_references],
            "sha256": self.sha256,
        }
        if self.schema_version >= 2:
            value["surroundingCompetenceEvidenceHashes"] = list(
                self.surrounding_competence_evidence_hashes
            )
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FoundryAnalysisFunnelResultV1":
        schema_version = int(value.get("schemaVersion", 0))
        if schema_version not in {1, 2}:
            raise ValueError("unsupported Foundry analysis funnel schema")
        selection = QualityDiversitySelectionV1.from_dict(
            dict(value.get("selection") or {})
        )
        analysis_profile_hash = str(value.get("analysisProfileHash") or "")
        selection_profile_hash = str(value.get("selectionProfileHash") or "")
        diversity_profile_hash = str(value.get("diversityProfileHash") or "")
        if selection.profile_hash != diversity_profile_hash:
            raise ValueError("funnel selection uses another diversity profile")
        plugins = tuple(dict(item) for item in value.get("plugins") or ())
        trace_hashes = tuple(str(item) for item in value.get("traceHashes") or ())
        analysis_hashes = tuple(str(item) for item in value.get("analysisHashes") or ())
        tactical_hashes = tuple(
            str(item) for item in value.get("tacticalVerificationHashes") or ()
        )
        verified_laffa_hashes = tuple(
            str(item) for item in value.get("verifiedLaffaEvidenceHashes") or ()
        )
        surrounding_hashes = tuple(
            str(item)
            for item in value.get("surroundingCompetenceEvidenceHashes") or ()
        )
        candidate_hashes = tuple(
            str(item) for item in value.get("candidateHashes") or ()
        )
        if not {item.sha256 for item in selection.selected}.issubset(candidate_hashes):
            raise ValueError("funnel selection contains an undeclared candidate")
        artifacts = tuple(
            dict(item) for item in value.get("artifactReferences") or ()
        )
        core = {
            "schemaVersion": schema_version,
            "analyzerProfileHash": analysis_profile_hash,
            "selectionProfileHash": selection_profile_hash,
            "diversityProfileHash": diversity_profile_hash,
            "plugins": [dict(item) for item in plugins],
            "traceHashes": list(trace_hashes),
            "analysisHashes": list(analysis_hashes),
            "tacticalVerificationHashes": list(tactical_hashes),
            "verifiedLaffaEvidenceHashes": list(verified_laffa_hashes),
            "candidateHashes": list(candidate_hashes),
            "selectionHash": selection.sha256,
        }
        if schema_version >= 2:
            core["surroundingCompetenceEvidenceHashes"] = list(surrounding_hashes)
        digest = canonical_sha256(core)
        if str(value.get("sha256") or "") != digest:
            raise ValueError("Foundry analysis funnel checksum mismatch")
        return cls(
            analysis_profile_hash=analysis_profile_hash,
            selection_profile_hash=selection_profile_hash,
            diversity_profile_hash=diversity_profile_hash,
            plugins=plugins,
            trace_hashes=trace_hashes,
            analysis_hashes=analysis_hashes,
            tactical_verification_hashes=tactical_hashes,
            verified_laffa_evidence_hashes=verified_laffa_hashes,
            candidate_hashes=candidate_hashes,
            selection=selection,
            artifact_references=artifacts,
            sha256=digest,
            surrounding_competence_evidence_hashes=surrounding_hashes,
            schema_version=schema_version,
        )


class FoundryAnalysisFunnelV1:
    """Analyze immutable traces and reduce qualified candidates deterministically."""

    descriptor = PluginDescriptor(
        id="foundry_analysis_funnel_v1",
        kind="trace_analysis_funnel",
        version="1.3.2",
        description="All-seat trace analysis, surrounding-competence qualification, causal verification, and quality-diversity reduction",
        capabilities=(
            "immutable_trace_ingestion",
            "all_decisions",
            "four_isolated_observers",
            "post_reveal_actual_deal_oracle",
            "verified_causal_scenario_graphs",
            "candidate_specific_surrounding_competence",
            "profile_gated_selection",
            "order_independent_reduction",
        ),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=900,
        maximum_memory_mb=4096,
    )

    def __init__(
        self,
        artifact_store: Any,
        analyzer: StagedTraceAnalyzerV1 | CascadedTraceAnalyzerV1,
        tactical_verifier: ActualDealOracleVerifierV1,
        selection_profile: SelectionProfileV1,
        diversity_profile: QualityDiversityProfileV1,
        *,
        scorer: ProfileChallengeScorerV1 | None = None,
        selector: QualityDiversitySelectorV1 | None = None,
        maximum_oracle_decisions_per_trace: int = 1,
        allow_pending_surrounding_nominations: bool = False,
    ):
        if int(maximum_oracle_decisions_per_trace) < 0:
            raise ValueError("maximum oracle decisions per trace cannot be negative")
        self.artifact_store = artifact_store
        self.analyzer = analyzer
        self.tactical_verifier = tactical_verifier
        self.selection_profile = selection_profile
        self.diversity_profile = diversity_profile
        self.scorer = scorer or ProfileChallengeScorerV1()
        self.selector = selector or QualityDiversitySelectorV1()
        self.maximum_oracle_decisions_per_trace = int(maximum_oracle_decisions_per_trace)
        self.allow_pending_surrounding_nominations = bool(
            allow_pending_surrounding_nominations
        )

    @staticmethod
    def _artifact_reference(owner_kind: str, owner_hash: str, value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "ownerKind": owner_kind,
            "ownerHash": owner_hash,
            "uri": str(value["uri"]),
            "sha256": str(value["sha256"]),
            "bytes": int(value["bytes"]),
            "mediaType": str(value["mediaType"]),
            "name": value.get("name"),
        }

    def _put(self, owner_kind: str, owner_hash: str, document: Mapping[str, Any]) -> dict[str, Any]:
        reference = self.artifact_store.put(
            _canonical_bytes(document),
            media_type=f"application/vnd.balootx.{owner_kind.replace('_', '-')}+json",
            name=f"{owner_hash}.{owner_kind}.json",
        )
        return self._artifact_reference(owner_kind, owner_hash, reference)

    @staticmethod
    def _featured_truth_blocked_ids(
        analysis: TraceAnalysisResultV1,
    ) -> tuple[str, ...]:
        adjudication = dict(analysis.adjudication_evidence or {})
        unresolved = {
            *(str(item) for item in analysis.unresolved_decision_ids),
            *(
                str(item)
                for item in adjudication.get("unresolvedDecisionIds") or ()
            ),
        }
        competence_only = tuple(
            str(item)
            for item in adjudication.get("competentDecisionIds") or ()
        )
        return tuple(sorted({
            *unresolved,
            *competence_only,
        }))

    def _load_trace(self, reference: Mapping[str, Any]) -> CompleteGameTraceV1:
        if str(reference.get("ownerKind")) != "complete_trace_v1":
            raise ValueError("analysis funnel accepts only complete trace artifacts")
        if str(reference.get("mediaType")) != "application/vnd.balootx.complete-trace-v1+json":
            raise ValueError("analysis funnel trace artifact has the wrong media type")
        data = self.artifact_store.get(str(reference.get("uri") or ""))
        if len(data) != int(reference.get("bytes", -1)):
            raise ValueError("complete trace artifact size mismatch")
        digest = hashlib.sha256(data).hexdigest()
        if digest != str(reference.get("sha256") or ""):
            raise ValueError("complete trace artifact checksum mismatch")
        document = json.loads(data)
        if not isinstance(document, Mapping):
            raise ValueError("complete trace artifact must contain a JSON object")
        trace = CompleteGameTraceV1.from_dict(document)
        if trace.sha256 != str(reference.get("ownerHash") or ""):
            raise ValueError("complete trace owner identity mismatch")
        return trace

    @staticmethod
    def _buyer_relationship(actor: Seat, buyer: Any) -> str:
        if not buyer:
            return "unknown"
        seat = Seat.parse(str(buyer))
        if seat == actor:
            return "self"
        return "partner" if seat.team == actor.team else "opponent"

    @staticmethod
    def _score_pressure(actor: Seat, match_score: Any) -> str:
        values = tuple(int(item) for item in (match_score or (0, 0)))
        if len(values) != 2 or values[0] == values[1]:
            return "level"
        return "leading" if values[int(actor.team)] > values[int(actor.team.other)] else "trailing"

    @staticmethod
    def _difficulty_band(value: float | None) -> str:
        score = float(value or 0.0)
        if score >= 80.0:
            return "elite"
        if score >= 60.0:
            return "hard"
        if score >= 35.0:
            return "medium"
        return "low"

    def _candidate(
        self,
        trace: CompleteGameTraceV1,
        store: TraceDecisionProjectionStoreV1,
        evidence: DecisionPerspectiveEvidenceV3,
        tactical: Mapping[str, TacticalVerificationResultV1],
        perspective_breadth: int,
        source: str,
        surrounding_competence: SurroundingCompetenceEvidenceV1 | None = None,
        unresolved_decision_ids: Iterable[str] = (),
        selection_score: Any | None = None,
    ) -> QualityDiversityCandidateV1 | None:
        if evidence.decision_id in {str(item) for item in unresolved_decision_ids}:
            return None
        score = (
            selection_score
            if selection_score is not None
            else self.scorer.score(evidence, self.selection_profile)
        )
        if not score.passed:
            return None
        surrounding_status = "not_required"
        if self.selection_profile.requires_surrounding_competence:
            if surrounding_competence is None:
                return None
            if surrounding_competence.passed:
                surrounding_status = "passed"
            elif self.allow_pending_surrounding_nominations:
                surrounding_status = (
                    "needs_finalist_validation"
                    if surrounding_competence.material_mistakes == 0
                    and surrounding_competence.catastrophic_crossings == 0
                    and surrounding_competence.minimum_quality
                    >= self.selection_profile.minimum_surrounding_quality
                    and set(surrounding_competence.rejection_reasons)
                    == {"unverified_decisions"}
                    else "needs_strong_replay"
                )
            else:
                return None
        record = next(item for item in trace.decision_ledger.records if item.decision_id == evidence.decision_id)
        verification = tactical.get(record.decision_id)
        if (
            self.selection_profile.require_verified_tactical_play
            and record.decision_kind == "play"
            and (
                verification is None
                or verification.status != "verified"
                or not verification.oracle.complete
                or not verification.evidence
            )
        ):
            return None
        projection = build_observer_projection(store, record.decision_id, record.acting_seat)
        public = projection.information_state
        detectors = sorted(
            item.kind.value for item in (verification.evidence if verification is not None else ())
        )
        plays = tuple(public.get("plays") or ())
        projects = tuple(public.get("projects") or ())
        bidding_round = public.get("biddingRound")
        structural_fingerprint = canonical_sha256({
            "schemaVersion": 2,
            "kind": "actor_projection_fingerprint_v1",
            "decisionKind": record.decision_kind,
            "phase": record.phase,
            "projectionHash": projection.sha256,
            "legalActions": [dict(item) for item in record.legal_actions],
        })
        dimensions = {
            "decision_kind": record.decision_kind,
            "contract": str(public.get("contract") or "not_established"),
            "purchase_round": (
                "first" if bidding_round == 0 else "second" if bidding_round == 1 else "not_applicable"
            ),
            "trump_suit": str(public.get("trumpSuit") or "none"),
            "project": "+".join(sorted({str(item.get("kind")) for item in projects})) or "none",
            "escalation": str(public.get("escalation") or "normal"),
            "tactical_detector": "+".join(detectors) or "none_verified",
            "trick_number": str(len(plays) // 4 + 1 if record.decision_kind == "play" else 0),
            "acting_seat": record.acting_seat.name.lower(),
            "buyer_relationship": self._buyer_relationship(record.acting_seat, public.get("buyer")),
            "score_pressure": self._score_pressure(record.acting_seat, public.get("matchScore")),
            "difficulty_band": self._difficulty_band(evidence.peak),
            "perspective_breadth": str(int(perspective_breadth)),
            "source": str(source),
            "video_duration": "decision_clip",
            "surrounding_competence": surrounding_status,
            "candidate_scope": (
                "decision_candidate"
                if surrounding_status in {"passed", "not_required"}
                else "deal_replay_nomination"
            ),
        }
        return QualityDiversityCandidateV1.build(
            trace_hash=trace.sha256,
            decision_id=record.decision_id,
            observer_seat=record.acting_seat.name.lower(),
            truth_label=evidence.truth_label,
            confidence=evidence.confidence,
            structural_fingerprint=structural_fingerprint,
            dimensions=dimensions,
            selection_score=score,
        )

    def _verify(
        self,
        trace: CompleteGameTraceV1,
        analysis: TraceAnalysisResultV1,
        store: TraceDecisionProjectionStoreV1,
        cancel_check: Callable[[], bool] | None,
        precomputed_actions: Mapping[
            str, tuple[ActualDealActionOracleV1, ...]
        ] | None = None,
        featured_truth_blocked_ids: Iterable[str] = (),
    ) -> tuple[TacticalVerificationResultV1, ...]:
        if self.maximum_oracle_decisions_per_trace == 0:
            return ()
        analyses = {item.decision_id: item for item in analysis.decisions}
        actor_evidence = {
            item.decision_id: item
            for item in analysis.laffa_evidence.perspectives
            if item.truth_mode == TruthMode.ACTOR_OBSERVER
        }
        critical = set(analysis.laffa_evidence.critical_decision_ids)
        featured_blocked = {
            *(str(item) for item in featured_truth_blocked_ids),
            *self._featured_truth_blocked_ids(analysis),
        }
        eligible = [
            snapshot for snapshot in store.replay.decisions
            if snapshot.decision.decision_kind == "play"
            and not snapshot.decision.forced
            and snapshot.decision.decision_id not in featured_blocked
            and sum(mask.bit_count() for mask in snapshot.state.hands)
            <= self.tactical_verifier.maximum_remaining_cards
        ]
        eligible.sort(
            key=lambda snapshot: (
                -(snapshot.decision.decision_id in critical),
                -self.scorer.score(
                    actor_evidence[snapshot.decision.decision_id],
                    self.selection_profile,
                ).score,
                -(actor_evidence[snapshot.decision.decision_id].peak or 0.0),
                snapshot.decision.ordinal,
                snapshot.decision.decision_id,
            )
        )
        return tuple(
            self.tactical_verifier.verify_decision(
                snapshot,
                analyses[snapshot.decision.decision_id],
                cancel_check=cancel_check,
                precomputed_actions=(
                    tuple(precomputed_actions[snapshot.decision.decision_id])
                    if precomputed_actions is not None
                    and snapshot.decision.decision_id in precomputed_actions
                    else None
                ),
            )
            for snapshot in eligible[: self.maximum_oracle_decisions_per_trace]
        )

    def process(
        self,
        trace_references: Iterable[Mapping[str, Any]],
        *,
        cancel_check: Callable[[], bool] | None = None,
        source_by_trace: Mapping[str, str] | None = None,
        progress_callback: Callable[[float], None] | None = None,
        oracle_action_cache_by_trace: Mapping[
            str, Mapping[str, tuple[ActualDealActionOracleV1, ...]]
        ] | None = None,
    ) -> FoundryAnalysisFunnelResultV1:
        references = sorted(
            (dict(item) for item in trace_references),
            key=lambda item: (str(item.get("ownerHash", "")), str(item.get("sha256", ""))),
        )
        if not references:
            raise ValueError("analysis funnel requires at least one complete trace")
        traces = [self._load_trace(item) for item in references]
        if len({item.sha256 for item in traces}) != len(traces):
            raise ValueError("analysis funnel input contains duplicate complete traces")

        analyses: list[TraceAnalysisResultV1] = []
        verifications: list[TacticalVerificationResultV1] = []
        verified_laffat: list[LaffaEvidenceV2] = []
        surrounding_evidence: list[SurroundingCompetenceEvidenceV1] = []
        candidates: list[QualityDiversityCandidateV1] = []
        artifacts: list[dict[str, Any]] = []
        for trace_index, trace in enumerate(traces):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Foundry analysis funnel cancelled")
            analysis = self.analyzer.analyze_trace(
                trace,
                cancel_check=cancel_check,
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda fraction, index=trace_index: progress_callback(
                        (index + 0.90 * fraction) / len(traces)
                    )
                ),
            )
            unresolved_decision_ids = tuple(
                str(item) for item in analysis.unresolved_decision_ids
            )
            competence_verified_decision_ids = tuple(
                str(item)
                for item in dict(
                    analysis.adjudication_evidence or {}
                ).get("competentDecisionIds") or ()
            )
            featured_truth_blocked_ids = self._featured_truth_blocked_ids(
                analysis
            )
            analyses.append(analysis)
            artifacts.append(self._put("trace_analysis_v1", analysis.sha256, analysis.to_dict()))
            store = TraceDecisionProjectionStoreV1(trace, self.analyzer.rules)
            verified = self._verify(
                trace,
                analysis,
                store,
                cancel_check,
                (
                    dict(oracle_action_cache_by_trace[trace.sha256])
                    if oracle_action_cache_by_trace is not None
                    and trace.sha256 in oracle_action_cache_by_trace
                    else None
                ),
                featured_truth_blocked_ids,
            )
            verifications.extend(verified)
            tactical_by_decision = {item.oracle.decision_id: item for item in verified}
            for item in verified:
                artifacts.append(self._put("tactical_verification_v1", item.sha256, item.to_dict()))
            verified_laffa = integrate_verified_tactical_evidence(
                analysis.laffa_evidence,
                trace.decision_ledger,
                verified,
                critical_threshold=self.analyzer.critical_threshold,
            )
            verified_laffat.append(verified_laffa)
            artifacts.append(
                self._put(
                    "verified_laffa_evidence_v2",
                    verified_laffa.sha256,
                    verified_laffa.to_dict(),
                )
            )
            for evidence in verified_laffa.perspectives:
                if evidence.truth_mode != TruthMode.ACTOR_OBSERVER:
                    continue
                qualification = None
                decision_score = self.scorer.score(
                    evidence,
                    self.selection_profile,
                )
                if (
                    decision_score.passed
                    and evidence.decision_id
                    not in set(featured_truth_blocked_ids)
                ):
                    qualification = self.selection_profile.qualify_surrounding(
                        analysis.decisions,
                        candidate_decision_id=evidence.decision_id,
                        trace_analysis_hash=analysis.sha256,
                        unresolved_decision_ids=unresolved_decision_ids,
                        competence_verified_decision_ids=(
                            competence_verified_decision_ids
                        ),
                    )
                    surrounding_evidence.append(qualification)
                    artifacts.append(
                        self._put(
                            "surrounding_competence_evidence_v1",
                            qualification.sha256,
                            qualification.to_dict(),
                        )
                    )
                candidate = self._candidate(
                    trace,
                    store,
                    evidence,
                    tactical_by_decision,
                    verified_laffa.perspective_breadth,
                    str((source_by_trace or {}).get(trace.sha256, "organic")),
                    qualification,
                    featured_truth_blocked_ids,
                    decision_score,
                )
                if candidate is not None:
                    candidates.append(candidate)
            if progress_callback is not None:
                progress_callback((trace_index + 1) / len(traces))

        selection = self.selector.select(candidates, self.diversity_profile)
        if progress_callback is not None:
            progress_callback(1.0)
        artifacts.append(self._put("quality_diversity_selection_v1", selection.sha256, selection.to_dict()))
        artifacts.sort(key=lambda item: (item["ownerKind"], item["ownerHash"], item["sha256"]))
        trace_hashes = tuple(sorted(item.sha256 for item in traces))
        analysis_hashes = tuple(sorted(item.sha256 for item in analyses))
        verification_hashes = tuple(sorted(item.sha256 for item in verifications))
        verified_laffa_hashes = tuple(sorted(item.sha256 for item in verified_laffat))
        surrounding_hashes = tuple(sorted(item.sha256 for item in surrounding_evidence))
        candidate_hashes = tuple(sorted(item.sha256 for item in candidates))
        plugins = tuple(sorted((
            {
                "kind": "trace_analyzer",
                "id": self.analyzer.descriptor.id,
                "version": self.analyzer.descriptor.version,
            },
            {
                "kind": "tactical_verifier",
                "id": self.tactical_verifier.descriptor.id,
                "version": self.tactical_verifier.descriptor.version,
            },
            {
                "kind": "challenge_scorer",
                "id": self.scorer.descriptor.id,
                "version": self.scorer.descriptor.version,
            },
            {
                "kind": "selection_policy",
                "id": self.selector.descriptor.id,
                "version": self.selector.descriptor.version,
            },
        ), key=lambda item: (item["kind"], item["id"], item["version"])))
        core = {
            "schemaVersion": 2,
            "analyzerProfileHash": self.analyzer.profile.sha256,
            "selectionProfileHash": self.selection_profile.sha256,
            "diversityProfileHash": self.diversity_profile.sha256,
            "plugins": [dict(item) for item in plugins],
            "traceHashes": list(trace_hashes),
            "analysisHashes": list(analysis_hashes),
            "tacticalVerificationHashes": list(verification_hashes),
            "verifiedLaffaEvidenceHashes": list(verified_laffa_hashes),
            "surroundingCompetenceEvidenceHashes": list(surrounding_hashes),
            "candidateHashes": list(candidate_hashes),
            "selectionHash": selection.sha256,
        }
        return FoundryAnalysisFunnelResultV1(
            analysis_profile_hash=self.analyzer.profile.sha256,
            selection_profile_hash=self.selection_profile.sha256,
            diversity_profile_hash=self.diversity_profile.sha256,
            plugins=plugins,
            trace_hashes=trace_hashes,
            analysis_hashes=analysis_hashes,
            tactical_verification_hashes=verification_hashes,
            verified_laffa_evidence_hashes=verified_laffa_hashes,
            candidate_hashes=candidate_hashes,
            selection=selection,
            artifact_references=tuple(artifacts),
            sha256=canonical_sha256(core),
            surrounding_competence_evidence_hashes=surrounding_hashes,
            schema_version=2,
        )

    def process_completed_campaign(
        self,
        campaign_store: Any,
        campaign_hash: str,
        *,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> FoundryAnalysisFunnelResultV1:
        """Reduce only a fully completed campaign; partial evidence fails closed."""
        campaign = campaign_store.get_campaign(str(campaign_hash))
        if campaign is None:
            raise KeyError(campaign_hash)
        if campaign.get("status") != "completed" or bool(campaign.get("cancelRequested")):
            raise ValueError("only a fully completed, non-cancelled campaign may enter analysis")
        shards = tuple(campaign_store.list_shards(str(campaign_hash)))
        if not shards or any(item.get("status") != "completed" for item in shards):
            raise ValueError("campaign analysis requires complete evidence from every shard")
        references_by_trace: dict[str, Mapping[str, Any]] = {}
        sources_by_trace: dict[str, set[str]] = {}
        for shard in shards:
            result = shard.get("result")
            if not isinstance(result, Mapping):
                raise ValueError("completed campaign shard is missing its immutable result")
            lane = str(dict(shard.get("spec") or {}).get("lane") or "organic")
            declared_candidates = {
                str(item) for item in result.get("candidateHashes") or ()
            }
            if any(
                len(item) != 64
                or any(char not in "0123456789abcdef" for char in item)
                for item in declared_candidates
            ):
                raise ValueError("campaign contains an invalid candidate trace hash")
            shard_references: dict[str, Mapping[str, Any]] = {}
            for raw in tuple(result.get("artifactReferences") or ()):
                reference = dict(raw)
                if str(reference.get("ownerKind")) != "complete_trace_v1":
                    continue
                trace_hash = str(reference.get("ownerHash") or "")
                existing = shard_references.get(trace_hash)
                if existing is not None and dict(existing) != reference:
                    raise ValueError(
                        "one complete trace has conflicting shard artifacts"
                    )
                shard_references[trace_hash] = reference
            missing = declared_candidates - set(shard_references)
            if missing:
                raise ValueError(
                    "campaign candidate is missing its complete-trace artifact"
                )
            for trace_hash in sorted(declared_candidates):
                reference = shard_references[trace_hash]
                existing = references_by_trace.get(trace_hash)
                if existing is not None and dict(existing) != reference:
                    raise ValueError("one complete trace has conflicting campaign artifacts")
                references_by_trace[trace_hash] = reference
                sources_by_trace.setdefault(trace_hash, set()).add(lane)
        if not references_by_trace:
            raise ValueError("completed campaign contains no retained candidate traces")
        sources = {
            trace_hash: "+".join(sorted(lanes))
            for trace_hash, lanes in sources_by_trace.items()
        }
        return self.process(
            references_by_trace.values(),
            cancel_check=cancel_check,
            source_by_trace=sources,
            progress_callback=progress_callback,
        )
