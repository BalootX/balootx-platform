"""Governed exact-deal replay with a stronger equal-seat finalist agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from baloot_platform.core.enums import GamePhase, Seat
from baloot_platform.plugins.base import PluginDescriptor

from .evidence_v3 import LaffaEvidenceV2, TruthMode, canonical_sha256
from .profiles import ContentSelectionScoreV1
from .simulation import NaturalLaffaSimulatorV1
from .trace import CompleteGameTraceV1


@dataclass(frozen=True, slots=True)
class FinalistValidationProfileV1:
    profile_id: str
    revision: str
    agent_profile_id: str
    maximum_traces_per_job: int
    maximum_confirmation_decisions_per_trace: int
    maximum_trace_seconds: int
    require_qualified_agent_release_for_publication: bool
    allow_local_preview: bool
    enable_exact_tactical_viability_precheck: bool
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "FinalistValidationProfileV1":
        allowed = {
            "schemaVersion",
            "id",
            "revision",
            "agentProfileId",
            "maximumTracesPerJob",
            "maximumConfirmationDecisionsPerTrace",
            "maximumTraceSeconds",
            "requireQualifiedAgentReleaseForPublication",
            "allowLocalPreview",
            "enableExactTacticalViabilityPrecheck",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"unknown finalist validation profile keys: {sorted(unknown)}"
            )
        core = {
            "schemaVersion": int(value.get("schemaVersion", 0)),
            "id": str(value.get("id") or "").strip(),
            "revision": str(value.get("revision") or "").strip(),
            "agentProfileId": str(value.get("agentProfileId") or "").strip(),
            "maximumTracesPerJob": int(value.get("maximumTracesPerJob", 0)),
            "maximumConfirmationDecisionsPerTrace": int(
                value.get("maximumConfirmationDecisionsPerTrace", 64)
            ),
            "maximumTraceSeconds": int(value.get("maximumTraceSeconds", 300)),
            "requireQualifiedAgentReleaseForPublication": bool(
                value.get("requireQualifiedAgentReleaseForPublication", False)
            ),
            "allowLocalPreview": bool(value.get("allowLocalPreview", False)),
            "enableExactTacticalViabilityPrecheck": bool(
                value.get("enableExactTacticalViabilityPrecheck", False)
            ),
        }
        if core["schemaVersion"] != 1:
            raise ValueError("unsupported finalist validation profile schema")
        if not core["id"] or not core["revision"] or not core["agentProfileId"]:
            raise ValueError("finalist validation profile identity is required")
        if not 1 <= core["maximumTracesPerJob"] <= 100:
            raise ValueError("finalist validation trace bound is out of range")
        if not 1 <= core["maximumConfirmationDecisionsPerTrace"] <= 64:
            raise ValueError("finalist validation confirmation bound is out of range")
        if not 30 <= core["maximumTraceSeconds"] <= 900:
            raise ValueError("finalist validation trace timeout is out of range")
        return cls(
            core["id"],
            core["revision"],
            core["agentProfileId"],
            core["maximumTracesPerJob"],
            core["maximumConfirmationDecisionsPerTrace"],
            core["maximumTraceSeconds"],
            core["requireQualifiedAgentReleaseForPublication"],
            core["allowLocalPreview"],
            core["enableExactTacticalViabilityPrecheck"],
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "agentProfileId": self.agent_profile_id,
            "maximumTracesPerJob": self.maximum_traces_per_job,
            "maximumConfirmationDecisionsPerTrace": (
                self.maximum_confirmation_decisions_per_trace
            ),
            "maximumTraceSeconds": self.maximum_trace_seconds,
            "requireQualifiedAgentReleaseForPublication": (
                self.require_qualified_agent_release_for_publication
            ),
            "allowLocalPreview": self.allow_local_preview,
            "enableExactTacticalViabilityPrecheck": (
                self.enable_exact_tactical_viability_precheck
            ),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class FinalistReplayEvidenceV1:
    parent_trace_hash: str
    descendant_trace_hash: str
    parent_game_hash: str
    descendant_game_hash: str
    deck_commitment: str
    validator_agent_profile_hash: str
    validator_agent_release: str
    validator_status: str
    decision_agreement_profile_hash: str | None
    decision_agreement_decision_count: int
    decision_agreement_unresolved_count: int
    replay_complete: bool
    local_preview_allowed: bool
    publication_ready: bool
    blockers: tuple[str, ...]
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        parent: CompleteGameTraceV1,
        descendant: CompleteGameTraceV1,
        simulator: NaturalLaffaSimulatorV1,
        require_qualified_agent_release_for_publication: bool,
        allow_local_preview: bool,
    ) -> "FinalistReplayEvidenceV1":
        parent_commitment = str(parent.seed_provenance.get("deckCommitment") or "")
        descendant_commitment = str(
            descendant.seed_provenance.get("deckCommitment") or ""
        )
        generation = dict(descendant.trusted_final_state.get("generation") or {})
        if parent_commitment != descendant_commitment:
            raise ValueError("finalist descendant changed the exact dealt cards")
        if generation.get("kind") != "finalist_validation_descendant":
            raise ValueError("finalist descendant has the wrong lineage kind")
        if str(generation.get("parentTraceHash") or "") != parent.sha256:
            raise ValueError("finalist descendant does not reference its parent trace")
        qualified = bool(
            simulator.agent_qualification is not None
            and simulator.agent_qualification.expert_generation_allowed
        )
        scored = (
            str(descendant.trusted_final_state.get("phase"))
            == GamePhase.SCORED.value
        )
        agreement_profile_hash = (
            str(generation.get("decisionAgreementProfileHash") or "") or None
        )
        agreement_decisions = int(
            generation.get("decisionAgreementDecisionCount", 0)
        )
        agreement_unresolved = int(
            generation.get("decisionAgreementUnresolvedCount", 0)
        )
        blockers: list[str] = []
        if not scored:
            blockers.append("descendant_not_scored")
        if require_qualified_agent_release_for_publication and not qualified:
            blockers.append("professional_agent_qualification_missing")
        if agreement_unresolved:
            blockers.append("unresolved_agent_decisions")
        validator_status = (
            "configured_candidate_unresolved"
            if agreement_unresolved
            else ("expert_qualified" if qualified else "configured_candidate")
        )
        publication_ready = scored and (
            qualified or not require_qualified_agent_release_for_publication
        ) and agreement_unresolved == 0
        core = {
            "schemaVersion": 1,
            "parentTraceHash": parent.sha256,
            "descendantTraceHash": descendant.sha256,
            "parentGameHash": parent.game_hash,
            "descendantGameHash": descendant.game_hash,
            "deckCommitment": parent_commitment,
            "validatorAgentProfileHash": simulator.profile_hash,
            "validatorAgentRelease": simulator.agent_release,
            "validatorStatus": validator_status,
            "decisionAgreementProfileHash": agreement_profile_hash,
            "decisionAgreementDecisionCount": agreement_decisions,
            "decisionAgreementUnresolvedCount": agreement_unresolved,
            "replayComplete": scored,
            "localPreviewAllowed": bool(allow_local_preview and scored),
            "publicationReady": publication_ready,
            "blockers": blockers,
        }
        return cls(
            parent.sha256,
            descendant.sha256,
            parent.game_hash,
            descendant.game_hash,
            parent_commitment,
            simulator.profile_hash,
            simulator.agent_release,
            validator_status,
            agreement_profile_hash,
            agreement_decisions,
            agreement_unresolved,
            scored,
            bool(allow_local_preview and scored),
            publication_ready,
            tuple(blockers),
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "parentTraceHash": self.parent_trace_hash,
            "descendantTraceHash": self.descendant_trace_hash,
            "parentGameHash": self.parent_game_hash,
            "descendantGameHash": self.descendant_game_hash,
            "deckCommitment": self.deck_commitment,
            "validatorAgentProfileHash": self.validator_agent_profile_hash,
            "validatorAgentRelease": self.validator_agent_release,
            "validatorStatus": self.validator_status,
            "decisionAgreementProfileHash": self.decision_agreement_profile_hash,
            "decisionAgreementDecisionCount": self.decision_agreement_decision_count,
            "decisionAgreementUnresolvedCount": self.decision_agreement_unresolved_count,
            "replayComplete": self.replay_complete,
            "localPreviewAllowed": self.local_preview_allowed,
            "publicationReady": self.publication_ready,
            "blockers": list(self.blockers),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class FinalistAnalysisEvidenceV1:
    descendant_trace_hash: str
    funnel_hash: str
    analysis_profile_hash: str
    selection_profile_hash: str
    diversity_profile_hash: str
    decision_count: int
    analyzed_decision_count: int
    perspective_count: int
    acting_seats: tuple[str, ...]
    critical_decision_count: int
    qualified_candidate_count: int
    selected_candidate_hashes: tuple[str, ...]
    surrounding_competence_evidence_count: int
    adjudicated_decision_count: int
    unresolved_decision_count: int
    laffa_selection_score: ContentSelectionScoreV1
    all_decisions_analyzed: bool
    four_observer_projections: bool
    all_seats_analyzed: bool
    challenge_ready: bool
    blockers: tuple[str, ...]
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        trace: CompleteGameTraceV1,
        funnel: Any,
        analysis_document: Mapping[str, Any],
        verified_laffa: LaffaEvidenceV2,
        laffa_selection_score: ContentSelectionScoreV1,
        expected_thrill_profile_hash: str,
    ) -> "FinalistAnalysisEvidenceV1":
        if str(analysis_document.get("traceHash") or "") != trace.sha256:
            raise ValueError("finalist analysis references the wrong descendant trace")
        ledger_by_id = {
            item.decision_id: item for item in trace.decision_ledger.records
        }
        analyzed = tuple(dict(item) for item in analysis_document.get("decisions") or ())
        analyzed_by_id = {str(item.get("decisionId") or ""): item for item in analyzed}
        all_decisions = (
            len(analyzed_by_id) == len(analyzed)
            and set(analyzed_by_id) == set(ledger_by_id)
        )
        if all_decisions and any(
            str(analyzed_by_id[decision_id].get("actor") or "")
            != record.acting_seat.name.lower()
            for decision_id, record in ledger_by_id.items()
        ):
            raise ValueError("finalist analysis acting-seat identity mismatch")

        acting_seats = tuple(
            sorted({item.acting_seat.name.lower() for item in ledger_by_id.values()})
        )
        all_seats = set(acting_seats) == {
            seat.name.lower() for seat in Seat
        }
        perspective_count = len(verified_laffa.perspectives)
        observer_safe_perspectives = tuple(
            item
            for item in verified_laffa.perspectives
            if item.truth_mode
            in {TruthMode.ACTOR_OBSERVER, TruthMode.SEAT_EXPERIENCE}
        )
        projections_by_decision = {
            decision_id: tuple(
                item
                for item in observer_safe_perspectives
                if item.decision_id == decision_id
            )
            for decision_id in ledger_by_id
        }
        four_projections = (
            verified_laffa.ledger_hash == trace.decision_ledger.sha256
            and len(observer_safe_perspectives) == len(ledger_by_id) * 4
            and all(
                len(items) == 4
                and {item.observer_seat for item in items} == set(Seat)
                and len(
                    tuple(
                        item
                        for item in items
                        if item.truth_mode == TruthMode.ACTOR_OBSERVER
                        and item.observer_seat == ledger_by_id[decision_id].acting_seat
                    )
                )
                == 1
                for decision_id, items in projections_by_decision.items()
            )
        )
        selected = tuple(funnel.selection.selected)
        if any(item.trace_hash != trace.sha256 for item in selected):
            raise ValueError("finalist selection contains a different trace")
        selected_hashes = tuple(sorted(item.sha256 for item in selected))
        critical_count = len(verified_laffa.critical_decision_ids)
        qualified_count = len(tuple(funnel.candidate_hashes))
        surrounding_count = len(
            tuple(funnel.surrounding_competence_evidence_hashes)
        )
        adjudication = dict(analysis_document.get("adjudicationEvidence") or {})
        adjudicated_count = len(tuple(adjudication.get("evaluations") or ()))
        unresolved_count = len(tuple(
            analysis_document.get("unresolvedDecisionIds") or ()
        ))
        if laffa_selection_score.score_family != "laffa_content_selection":
            raise ValueError("finalist Laffa score uses the wrong score family")
        if laffa_selection_score.profile_hash != expected_thrill_profile_hash:
            raise ValueError("finalist Laffa score uses another thrill profile")
        if laffa_selection_score.evidence_hash != verified_laffa.sha256:
            raise ValueError("finalist Laffa score references different evidence")
        blockers: list[str] = []
        if not all_decisions:
            blockers.append("incomplete_decision_analysis")
        if not four_projections:
            blockers.append("incomplete_four_observer_coverage")
        if not all_seats:
            blockers.append("incomplete_acting_seat_coverage")
        if critical_count < 1:
            blockers.append("no_verified_critical_decision")
        if not selected_hashes:
            blockers.append("no_finalist_candidate_survived")
        if unresolved_count:
            blockers.append("unresolved_finalist_analysis")
        if not laffa_selection_score.passed:
            blockers.append("laffa_selection_failed")
        challenge_ready = not blockers
        core = {
            "schemaVersion": 2,
            "descendantTraceHash": trace.sha256,
            "funnelHash": str(funnel.sha256),
            "analysisProfileHash": str(funnel.analysis_profile_hash),
            "selectionProfileHash": str(funnel.selection_profile_hash),
            "diversityProfileHash": str(funnel.diversity_profile_hash),
            "decisionCount": len(ledger_by_id),
            "analyzedDecisionCount": len(analyzed),
            "perspectiveCount": perspective_count,
            "actingSeats": list(acting_seats),
            "criticalDecisionCount": critical_count,
            "qualifiedCandidateCount": qualified_count,
            "selectedCandidateHashes": list(selected_hashes),
            "surroundingCompetenceEvidenceCount": surrounding_count,
            "adjudicatedDecisionCount": adjudicated_count,
            "unresolvedDecisionCount": unresolved_count,
            "laffaSelectionScore": laffa_selection_score.to_dict(),
            "allDecisionsAnalyzed": all_decisions,
            "fourObserverProjections": four_projections,
            "allSeatsAnalyzed": all_seats,
            "challengeReady": challenge_ready,
            "blockers": blockers,
        }
        return cls(
            trace.sha256,
            str(funnel.sha256),
            str(funnel.analysis_profile_hash),
            str(funnel.selection_profile_hash),
            str(funnel.diversity_profile_hash),
            len(ledger_by_id),
            len(analyzed),
            perspective_count,
            acting_seats,
            critical_count,
            qualified_count,
            selected_hashes,
            surrounding_count,
            adjudicated_count,
            unresolved_count,
            laffa_selection_score,
            all_decisions,
            four_projections,
            all_seats,
            challenge_ready,
            tuple(blockers),
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 2,
            "descendantTraceHash": self.descendant_trace_hash,
            "funnelHash": self.funnel_hash,
            "analysisProfileHash": self.analysis_profile_hash,
            "selectionProfileHash": self.selection_profile_hash,
            "diversityProfileHash": self.diversity_profile_hash,
            "decisionCount": self.decision_count,
            "analyzedDecisionCount": self.analyzed_decision_count,
            "perspectiveCount": self.perspective_count,
            "actingSeats": list(self.acting_seats),
            "criticalDecisionCount": self.critical_decision_count,
            "qualifiedCandidateCount": self.qualified_candidate_count,
            "selectedCandidateHashes": list(self.selected_candidate_hashes),
            "surroundingCompetenceEvidenceCount": (
                self.surrounding_competence_evidence_count
            ),
            "adjudicatedDecisionCount": self.adjudicated_decision_count,
            "unresolvedDecisionCount": self.unresolved_decision_count,
            "laffaSelectionScore": self.laffa_selection_score.to_dict(),
            "allDecisionsAnalyzed": self.all_decisions_analyzed,
            "fourObserverProjections": self.four_observer_projections,
            "allSeatsAnalyzed": self.all_seats_analyzed,
            "challengeReady": self.challenge_ready,
            "blockers": list(self.blockers),
            "sha256": self.sha256,
        }


class FinalistLaffaReplayV1:
    descriptor = PluginDescriptor(
        id="finalist_laffa_replay_v1",
        kind="finalist_replay",
        version="1.0.0",
        description=(
            "Exact-deal descendant replay with an equal-seat validator agent "
            "and fail-closed publication qualification"
        ),
        capabilities=(
            "exact_deal_descendant",
            "equal_seat_validator",
            "configuration_snapshot_preserved",
            "local_candidate_preview",
            "publication_qualification_gate",
        ),
        deterministic=True,
        timeout_seconds=900,
        maximum_memory_mb=4096,
    )

    def __init__(
        self,
        simulator: NaturalLaffaSimulatorV1,
        *,
        require_qualified_agent_release_for_publication: bool = True,
        allow_local_preview: bool = True,
    ):
        self.simulator = simulator
        self.require_qualified_agent_release_for_publication = bool(
            require_qualified_agent_release_for_publication
        )
        self.allow_local_preview = bool(allow_local_preview)

    def replay(
        self,
        parent: CompleteGameTraceV1,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[CompleteGameTraceV1, FinalistReplayEvidenceV1]:
        generation: Mapping[str, Any] = dict(
            parent.trusted_final_state.get("generation") or {}
        )
        match_score = tuple(
            int(item) for item in generation.get("matchScore") or (0, 0)
        )
        if len(match_score) != 2:
            raise ValueError("parent finalist context has an invalid match score")
        descendant = self.simulator.resimulate_exact_deal_as_finalist_descendant(
            parent,
            match_score=(match_score[0], match_score[1]),
            round_number=int(generation.get("roundNumber", 1)),
            cancel_check=cancel_check,
        )
        evidence = FinalistReplayEvidenceV1.build(
            parent=parent,
            descendant=descendant,
            simulator=self.simulator,
            require_qualified_agent_release_for_publication=(
                self.require_qualified_agent_release_for_publication
            ),
            allow_local_preview=self.allow_local_preview,
        )
        return descendant, evidence
