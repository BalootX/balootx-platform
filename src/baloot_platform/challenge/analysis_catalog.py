"""Trusted bridge from completed Foundry analysis to play and video products."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from baloot_platform.core.enums import Seat
from baloot_platform.plugins.base import PluginDescriptor
from baloot_platform.video.batch_service import RENDER_BUNDLE_MEDIA_TYPE

from .analysis_pipeline import TraceAnalysisResultV1
from .deep_analysis import DeepAnalysisBatchSpecV1
from .evidence_v3 import RenderVariantV1, TruthMode
from .finalist_adjudication import frozen_adaptive_finalist_profile_hash_v1
from .perspective_pipeline import build_observer_projection
from .publication_service import ChallengePublicationServiceV1
from .funnel import FoundryAnalysisFunnelResultV1
from .quality_diversity import QualityDiversityCandidateV1
from .replay import TraceDecisionProjectionStoreV1
from .trace import CompleteGameTraceV1


_FUNNEL_MEDIA_TYPE = "application/vnd.balootx.foundry-analysis-funnel-v1+json"
_TRACE_MEDIA_TYPE = "application/vnd.balootx.complete-trace-v1+json"
_ANALYSIS_MEDIA_TYPE = "application/vnd.balootx.trace-analysis-v1+json"


class NoPlayableFinalistCandidateError(ValueError):
    """The qualified descendant has no card-play decision for local rendering."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def complete_observer_safe_lifecycle(
    trace: CompleteGameTraceV1,
    *,
    observer: Seat,
    pause_before_play_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project one complete trace into the public and single-seat video facts."""

    state = dict(trace.trusted_final_state)
    seat = observer.name.lower()
    initial_hands = state.get("initialHands")
    if not isinstance(initial_hands, Mapping) or not isinstance(initial_hands.get(seat), list):
        raise ValueError("complete Laffa render requires the observer's frozen initial hand")
    completed_tricks = state.get("completedTricks")
    if not isinstance(completed_tricks, list) or sum(
        len(dict(trick).get("plays") or ()) for trick in completed_tricks
    ) != 32:
        raise ValueError("complete Laffa render requires all 32 public card plays")
    projects = [
        dict(project) for project in state.get("projects") or ()
        if bool(dict(project).get("declared")) or bool(dict(project).get("revealed"))
    ]
    generation = dict(state.get("generation") or {})
    public_tricks = []
    for trick in completed_tricks:
        value = dict(trick)
        value["plays"] = [
            {
                key: dict(play)[key]
                for key in (
                    "seat", "card", "trickIndex", "position", "ekka", "baloot"
                )
                if key in dict(play)
            }
            for play in value.get("plays") or ()
        ]
        public_tricks.append(value)
    public = {
        "roundNumber": int(state.get("roundNumber", 1)),
        "pauseBeforePlayIndex": int(pause_before_play_index),
        "dealer": state.get("dealer"),
        "phase": state.get("phase"),
        "matchScore": list(state.get("matchScore") or (0, 0)),
        "preDecisionMatchScore": list(generation.get("matchScore") or (0, 0)),
        "purchaseCard": state.get("purchaseCard"),
        "bidHistory": list(state.get("biddingHistory") or ()),
        "contract": state.get("contract"),
        "buyer": state.get("buyer"),
        "declarer": state.get("declarer"),
        "trumpSuit": state.get("trumpSuit"),
        "escalation": state.get("escalation"),
        "escalationHistory": list(state.get("escalationHistory") or ()),
        "locked": bool(state.get("locked", False)),
        "firstLeader": state.get("firstLeader"),
        "completedTricks": public_tricks,
        "projects": projects,
        "projectWinningTeam": state.get("projectWinningTeam"),
        "balootTeams": list(state.get("balootTeams") or ()),
        "roundScore": state.get("roundScore"),
        "matchWinner": state.get("matchWinner"),
    }
    observer_facts = {
        "seat": seat,
        "initialHandCards": [dict(card) for card in initial_hands[seat]],
        "privateProjects": [
            project for project in projects if str(project.get("owner")) == seat
        ],
    }
    return public, observer_facts


# Compatibility for existing internal callers while the public helper name migrates.
_complete_observer_safe_lifecycle = complete_observer_safe_lifecycle


class FoundryAnalysisCandidateServiceV1:
    """Resolve selected candidates without copying engine or hidden-state truth."""

    descriptor = PluginDescriptor(
        id="foundry_analysis_candidate_service_v1",
        kind="candidate_materializer",
        version="1.0.0",
        description="Materialize selected observer-safe challenges and render bundles",
        capabilities=(
            "completed_analysis_only",
            "content_addressed_candidate_resolution",
            "exact_challenge_locator",
            "observer_safe_render_bundle",
        ),
        resource_class="io",
        deterministic=True,
        timeout_seconds=60,
        maximum_memory_mb=512,
    )

    def __init__(
        self,
        *,
        deep_analysis: Any,
        campaign_store: Any,
        artifact_store: Any,
        publication_service: ChallengePublicationServiceV1,
        renderer: Any,
        rules: Any,
    ) -> None:
        self.deep_analysis = deep_analysis
        self.campaign_store = campaign_store
        self.artifact_store = artifact_store
        self.publication_service = publication_service
        self.renderer = renderer
        self.rules = rules

    def _load_document(
        self,
        reference: Mapping[str, Any],
        *,
        owner_kind: str,
        media_type: str,
    ) -> Mapping[str, Any]:
        if (
            str(reference.get("ownerKind") or "") != owner_kind
            or str(reference.get("mediaType") or "") != media_type
        ):
            raise ValueError(f"unexpected {owner_kind} artifact contract")
        payload = self.artifact_store.get(str(reference.get("uri") or ""))
        if (
            len(payload) != int(reference.get("bytes", -1))
            or hashlib.sha256(payload).hexdigest()
            != str(reference.get("sha256") or "")
        ):
            raise ValueError(f"{owner_kind} artifact integrity check failed")
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{owner_kind} artifact is not canonical JSON") from exc
        if not isinstance(document, Mapping):
            raise ValueError(f"{owner_kind} artifact must contain an object")
        return document

    def _completed_job(self, job_id: str) -> Mapping[str, Any]:
        job = self.deep_analysis.get(str(job_id))
        if job is None:
            raise KeyError(job_id)
        if job.get("status") != "completed" or not isinstance(job.get("result"), Mapping):
            raise RuntimeError("analysis candidates require a completed batch")
        return job

    def _funnel_result(
        self, job: Mapping[str, Any]
    ) -> FoundryAnalysisFunnelResultV1:
        spec = DeepAnalysisBatchSpecV1.from_dict(dict(job["request"]))
        result = dict(job["result"])
        if str(result.get("batchHash") or "") != spec.sha256:
            raise ValueError("completed analysis batch identity mismatch")
        references = tuple(dict(item) for item in result.get("artifactReferences") or ())
        matches = [
            item for item in references
            if item.get("ownerKind") == "foundry_analysis_funnel_v1"
        ]
        if len(matches) != 1:
            raise ValueError("completed analysis must reference one funnel manifest")
        document = self._load_document(
            matches[0],
            owner_kind="foundry_analysis_funnel_v1",
            media_type=_FUNNEL_MEDIA_TYPE,
        )
        if (
            str(matches[0].get("ownerHash") or "")
            != str(result.get("funnelResultHash") or "")
            or str(document.get("sha256") or "")
            != str(result.get("funnelResultHash") or "")
        ):
            raise ValueError("funnel manifest identity mismatch")
        parsed = FoundryAnalysisFunnelResultV1.from_dict(document)
        if parsed.analysis_profile_hash != spec.analysis_profile_hash:
            raise ValueError("funnel analyzer profile does not match its frozen batch")
        if parsed.selection.sha256 != str(result.get("selectionHash") or ""):
            raise ValueError("funnel selection identity mismatch")
        if set(parsed.candidate_hashes) != {
            str(item) for item in result.get("candidateHashes") or ()
        }:
            raise ValueError("funnel candidate membership mismatch")
        referenced_analysis_hashes = {
            str(item.get("ownerHash") or "")
            for item in references
            if item.get("ownerKind") == "trace_analysis_v1"
        }
        if referenced_analysis_hashes != set(parsed.analysis_hashes):
            raise ValueError("funnel analysis artifact membership mismatch")
        return parsed

    def _selected(
        self,
        job: Mapping[str, Any],
        funnel: FoundryAnalysisFunnelResultV1 | None = None,
    ) -> tuple[QualityDiversityCandidateV1, ...]:
        resolved = funnel or self._funnel_result(job)
        candidates = resolved.selection.selected
        declared = set(str(item) for item in dict(job["result"]).get("candidateHashes") or ())
        if any(item.sha256 not in declared for item in candidates):
            raise ValueError("selected candidate is absent from the completed batch")
        return candidates

    def _trace_reference(
        self,
        campaign_hash: str,
        trace_hash: str,
    ) -> Mapping[str, Any]:
        matches: list[Mapping[str, Any]] = []
        for shard in self.campaign_store.list_shards(campaign_hash):
            result = dict(shard.get("result") or {})
            for raw in result.get("artifactReferences") or ():
                reference = dict(raw)
                if (
                    reference.get("ownerKind") == "complete_trace_v1"
                    and reference.get("ownerHash") == trace_hash
                ):
                    matches.append(reference)
        if not matches:
            raise KeyError(trace_hash)
        if any(dict(item) != dict(matches[0]) for item in matches[1:]):
            raise ValueError("candidate trace has conflicting artifact references")
        return matches[0]

    def _trace(self, reference: Mapping[str, Any]) -> CompleteGameTraceV1:
        document = self._load_document(
            reference,
            owner_kind="complete_trace_v1",
            media_type=_TRACE_MEDIA_TYPE,
        )
        trace = CompleteGameTraceV1.from_dict(document)
        if trace.sha256 != str(reference.get("ownerHash") or ""):
            raise ValueError("candidate trace owner identity mismatch")
        return trace

    def _analysis_reference(
        self,
        job: Mapping[str, Any],
        trace_hash: str,
        funnel: FoundryAnalysisFunnelResultV1,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        if trace_hash not in set(funnel.trace_hashes):
            raise ValueError("candidate trace is absent from the verified funnel")
        matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for raw in dict(job["result"]).get("artifactReferences") or ():
            reference = dict(raw)
            if reference.get("ownerKind") != "trace_analysis_v1":
                continue
            document = self._load_document(
                reference,
                owner_kind="trace_analysis_v1",
                media_type=_ANALYSIS_MEDIA_TYPE,
            )
            if str(document.get("sha256") or "") != str(
                reference.get("ownerHash") or ""
            ):
                raise ValueError("trace analysis owner identity mismatch")
            if str(document.get("sha256") or "") not in set(
                funnel.analysis_hashes
            ):
                raise ValueError("trace analysis is absent from the verified funnel")
            if str(document.get("traceHash") or "") == trace_hash:
                matches.append((reference, document))
        if len(matches) != 1:
            raise ValueError("candidate requires exactly one trace analysis artifact")
        return matches[0]

    @staticmethod
    def _summary(
        candidate: QualityDiversityCandidateV1,
        analysis_document: Mapping[str, Any],
        *,
        compatibility_blockers: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        decision = next(
            (
                dict(item) for item in analysis_document.get("decisions") or ()
                if str(item.get("decisionId") or "") == candidate.decision_id
            ),
            None,
        )
        if decision is None:
            raise ValueError("candidate decision is absent from its analysis")
        laffa = dict(analysis_document.get("laffaEvidence") or {})
        critical = candidate.decision_id in set(
            str(item) for item in laffa.get("criticalDecisionIds") or ()
        )
        unresolved = candidate.decision_id in {
            str(item)
            for item in analysis_document.get("unresolvedDecisionIds") or ()
        }
        historical_adaptive = (
            str(analysis_document.get("analyzerId") or "")
            == "adaptive_finalist_trace_analyzer_v1"
            and str(analysis_document.get("analyzerVersion") or "") != "1.3.0"
        )
        return {
            "candidateHash": candidate.sha256,
            "shortCode": candidate.sha256[:8].upper(),
            "observerSeat": candidate.observer_seat,
            "decisionKind": str(decision.get("decisionKind") or ""),
            "truthLabel": candidate.truth_label.value,
            "confidence": candidate.confidence,
            "selectionScore": candidate.selection_score.score,
            "dimensions": dict(candidate.dimensions),
            "bestAction": dict(decision.get("bestAction") or {}),
            "actualAction": dict(decision.get("actualAction") or {}),
            "alternativeCount": len(tuple(decision.get("alternatives") or ())),
            "laffaClassification": str(laffa.get("classification") or ""),
            "critical": critical,
            "playReady": (
                critical
                and str(decision.get("decisionKind") or "") == "play"
                and not unresolved
                and not historical_adaptive
                and not compatibility_blockers
                and dict(candidate.dimensions).get(
                    "surrounding_competence", "not_required"
                )
                in {"passed", "not_required"}
            ),
            "finalistValidationRequired": (
                not compatibility_blockers
                and
                dict(candidate.dimensions).get("surrounding_competence")
                in {"needs_finalist_validation", "needs_strong_replay"}
            ),
            "compatibilityBlockers": list(compatibility_blockers),
            "technical": {
                "traceHash": candidate.trace_hash,
                "decisionId": candidate.decision_id,
                "analysisHash": str(analysis_document.get("sha256") or ""),
            },
        }

    def _trace_compatibility_blockers(
        self,
        trace: CompleteGameTraceV1,
    ) -> tuple[str, ...]:
        blockers: list[str] = []
        if trace.rules_hash != self.rules.sha256:
            blockers.append("rules_profile_mismatch")
        state = dict(trace.trusted_final_state)
        generation = dict(state.get("generation") or {})
        snapshot = state.get("configurationSnapshot")
        snapshot_hash = str(generation.get("configurationSnapshotHash") or "")
        if not snapshot_hash or not isinstance(snapshot, Mapping):
            blockers.append("frozen_configuration_snapshot_missing")
        elif str(dict(snapshot).get("sha256") or "") != snapshot_hash:
            blockers.append("frozen_configuration_snapshot_mismatch")
        return tuple(blockers)

    @staticmethod
    def _reveal_reason(
        candidate: QualityDiversityCandidateV1,
        analysis_document: Mapping[str, Any],
    ) -> tuple[str, dict[str, str]]:
        decision = next(
            (
                dict(item) for item in analysis_document.get("decisions") or ()
                if str(item.get("decisionId") or "") == candidate.decision_id
            ),
            None,
        )
        if decision is None:
            raise ValueError("candidate decision is absent from its analysis")
        best = dict(decision.get("bestAction") or {})
        actual = dict(decision.get("actualAction") or {})
        alternatives = tuple(dict(item) for item in decision.get("alternatives") or ())
        alternative_action = actual if actual and actual != best else {}
        if not alternative_action:
            alternative_action = next(
                (
                    dict(item.get("action") or {}) for item in alternatives
                    if dict(item.get("action") or {}) != best
                ),
                {},
            )
        best_label = str(best.get("label") or best.get("code") or "").strip()
        alternative_label = str(
            alternative_action.get("label") or alternative_action.get("code") or ""
        ).strip()
        if not best_label:
            raise ValueError("candidate best action requires a display label or code")
        values = {"best": best_label, "alternative": alternative_label}
        detectors = {
            item for item in str(dict(candidate.dimensions).get("tactical_detector") or "").split("+")
            if item and item != "none"
        }
        if {"only_winning_move", "contract_rescue"}.issubset(detectors):
            if not alternative_label:
                raise ValueError(
                    "combined tactical explanation requires an alternative action label or code"
                )
            key = "video.analysis.only_winning_contract_rescue"
        elif "only_winning_move" in detectors:
            key = "video.analysis.only_winning_move"
        elif "contract_rescue" in detectors:
            key = "video.analysis.contract_rescue"
        else:
            priority = (
                "small_card_big_gain",
                "large_card_sacrifice",
                "kabout_creation_prevention",
                "entry_unblocking",
                "trump_control",
                "partner_enablement",
            )
            selected = next((item for item in priority if item in detectors), None)
            key = (
                "video.analysis.kabout_creation_prevention"
                if selected == "kabout_creation_prevention"
                else f"video.analysis.{selected}"
                if selected is not None
                else "video.analysis.verified_tactical"
            )
        return key, values

    def list_candidates(self, job_id: str) -> dict[str, Any]:
        job = self._completed_job(job_id)
        funnel = self._funnel_result(job)
        candidates = self._selected(job, funnel)
        summaries = []
        trace_compatibility: dict[str, tuple[str, ...]] = {}
        for candidate in candidates:
            _, analysis_document = self._analysis_reference(
                job,
                candidate.trace_hash,
                funnel,
            )
            blockers = trace_compatibility.get(candidate.trace_hash)
            if blockers is None:
                campaign_hash = str(dict(job["request"])["campaignHash"])
                trace = self._trace(
                    self._trace_reference(campaign_hash, candidate.trace_hash)
                )
                blockers = self._trace_compatibility_blockers(trace)
                trace_compatibility[candidate.trace_hash] = blockers
            summaries.append(
                self._summary(
                    candidate,
                    analysis_document,
                    compatibility_blockers=blockers,
                )
            )
        return {
            "schemaVersion": 1,
            "jobId": str(job["jobId"]),
            "campaignHash": str(dict(job["request"])["campaignHash"]),
            "count": len(summaries),
            "items": summaries,
        }

    def candidate_trace(
        self,
        job_id: str,
        candidate_hash: str,
    ) -> CompleteGameTraceV1:
        """Resolve the complete authoritative parent Laffa for a selected candidate."""
        job = self._completed_job(job_id)
        funnel = self._funnel_result(job)
        candidate = next(
            (
                item
                for item in self._selected(job, funnel)
                if item.sha256 == candidate_hash
            ),
            None,
        )
        if candidate is None:
            raise KeyError(candidate_hash)
        campaign_hash = str(dict(job["request"])["campaignHash"])
        return self._trace(
            self._trace_reference(campaign_hash, candidate.trace_hash)
        )

    def prepare_actor_candidate(
        self,
        job_id: str,
        candidate_hash: str,
        *,
        reveal_policy: str = "after_commit",
        export_preset: str = "decision_clip",
    ) -> dict[str, Any]:
        job = self._completed_job(job_id)
        funnel = self._funnel_result(job)
        candidate = next(
            (
                item
                for item in self._selected(job, funnel)
                if item.sha256 == candidate_hash
            ),
            None,
        )
        if candidate is None:
            raise KeyError(candidate_hash)
        if dict(candidate.dimensions).get("surrounding_competence") in {
            "needs_finalist_validation",
            "needs_strong_replay",
        }:
            raise ValueError(
                "candidate requires governed finalist validation before challenge or video preparation"
            )
        campaign_hash = str(dict(job["request"])["campaignHash"])
        trace = self._trace(self._trace_reference(campaign_hash, candidate.trace_hash))
        compatibility_blockers = self._trace_compatibility_blockers(trace)
        if compatibility_blockers:
            raise ValueError(
                "candidate cannot be replayed with the active governed profiles: "
                + ", ".join(compatibility_blockers)
            )
        analysis_reference, analysis_document = self._analysis_reference(
            job,
            candidate.trace_hash,
            funnel,
        )
        return self._prepare_verified_candidate(
            trace=trace,
            candidate=candidate,
            analysis_reference=analysis_reference,
            analysis_document=analysis_document,
            expected_analysis_profile_hash=funnel.analysis_profile_hash,
            reveal_policy=reveal_policy,
            export_preset=export_preset,
        )

    def _prepare_verified_candidate(
        self,
        *,
        trace: CompleteGameTraceV1,
        candidate: QualityDiversityCandidateV1,
        analysis_reference: Mapping[str, Any],
        analysis_document: Mapping[str, Any],
        expected_analysis_profile_hash: str,
        reveal_policy: str,
        export_preset: str = "decision_clip",
    ) -> dict[str, Any]:
        if (
            str(analysis_document.get("analyzerId") or "")
            == "adaptive_finalist_trace_analyzer_v1"
        ):
            if str(analysis_document.get("analyzerVersion") or "") != "1.3.0":
                raise ValueError(
                    "historical adaptive finalist analysis is preview-ineligible"
                )
            frozen_profile_hash = frozen_adaptive_finalist_profile_hash_v1(trace)
            if frozen_profile_hash != expected_analysis_profile_hash:
                raise ValueError(
                    "finalist funnel profile does not match the frozen trace authority"
                )
        analysis = TraceAnalysisResultV1.from_dict(
            analysis_document,
            ledger=trace.decision_ledger,
            expected_adaptive_profile_hash=expected_analysis_profile_hash,
        )
        reason_key, reason_values = self._reveal_reason(candidate, analysis_document)
        resolved_reason = self.renderer.localized_reveal_reason(
            reason_key, reason_values
        )
        published = self.publication_service.publish_actor_challenge(
            trace,
            analysis_reference=analysis_reference,
            decision_id=candidate.decision_id,
            reveal_policy=reveal_policy,
            answer_explanation=resolved_reason,
            expected_adaptive_profile_hash=expected_analysis_profile_hash,
        )
        decision = next(
            item for item in trace.decision_ledger.records
            if item.decision_id == candidate.decision_id
        )
        actor = next(
            item for item in analysis.laffa_evidence.perspectives
            if item.decision_id == candidate.decision_id
            and item.truth_mode == TruthMode.ACTOR_OBSERVER
        )
        store = TraceDecisionProjectionStoreV1(trace, self.rules)
        projection = build_observer_projection(
            store, candidate.decision_id, Seat.parse(candidate.observer_seat)
        )
        complete_public_lifecycle = None
        complete_observer_lifecycle = None
        if export_preset == "hooked_flagship":
            complete_public_lifecycle, complete_observer_lifecycle = (
                complete_observer_safe_lifecycle(
                    trace,
                    observer=Seat.parse(candidate.observer_seat),
                    pause_before_play_index=len(
                        dict(projection.information_state).get("plays") or ()
                    ),
                )
            )
        variant = RenderVariantV1(
            analysis.laffa_evidence.sha256,
            candidate.decision_id,
            Seat.parse(candidate.observer_seat),
            "actor_challenge",
            "choose_acting_action",
            reveal_policy,
            published.locator.universal_link,
        )
        bundle = self.renderer.build_perspective_render_request(
            projection=projection,
            evidence=actor,
            variant=variant,
            effective_config_sha256=trace.bundle_hash,
            game_serial=published.locator.code,
            legal_actions=tuple(dict(item) for item in decision.legal_actions),
            reveal_reason_key=reason_key,
            reveal_reason_values=reason_values,
            export_preset=export_preset,
            complete_public_lifecycle=complete_public_lifecycle,
            complete_observer_lifecycle=complete_observer_lifecycle,
        )
        payload = _canonical_bytes(bundle.to_dict())
        stored = self.artifact_store.put(
            payload,
            media_type=RENDER_BUNDLE_MEDIA_TYPE,
            name=f"{bundle.bundle_sha256}.render-bundle.json",
        )
        bundle_reference = {
            "ownerKind": "render_bundle_v1",
            "ownerHash": bundle.bundle_sha256,
            "uri": str(stored["uri"]),
            "sha256": str(stored["sha256"]),
            "bytes": int(stored["bytes"]),
            "mediaType": str(stored["mediaType"]),
            "name": stored.get("name"),
        }
        if bundle_reference["sha256"] != hashlib.sha256(payload).hexdigest():
            raise ValueError("render bundle artifact checksum mismatch")
        return {
            "schemaVersion": 1,
            "exportPreset": export_preset,
            "candidate": self._summary(candidate, analysis_document),
            "locator": published.locator.to_dict(),
            "bundleArtifact": bundle_reference,
            "basename": (
                f"balootx-{published.locator.code.lower()}"
                if export_preset == "decision_clip"
                else f"balootx-{published.locator.code.lower()}-laffa"
            ),
        }

    def prepare_best_finalist_candidate(
        self,
        finalist_job: Mapping[str, Any],
        *,
        reveal_policy: str = "after_commit",
        export_preset: str = "decision_clip",
    ) -> dict[str, Any]:
        """Materialize the strongest locally qualified descendant candidate."""
        if (
            finalist_job.get("status") != "completed"
            or not isinstance(finalist_job.get("result"), Mapping)
        ):
            raise RuntimeError("finalist candidate preparation requires a completed job")
        request = dict(finalist_job.get("request") or {})
        expected_analysis_profile_hash = str(
            request.get("finalistAnalysisProfileHash") or ""
        )
        if len(expected_analysis_profile_hash) != 64:
            raise ValueError(
                "finalist candidate lacks its frozen analysis profile authority"
            )
        result = dict(finalist_job["result"])
        eligible = [
            dict(item) for item in result.get("items") or ()
            if bool(dict(item).get("localPreviewReady"))
        ]
        if not eligible:
            raise ValueError("finalist validation has no locally qualified candidate")

        prepared: list[
            tuple[
                float,
                CompleteGameTraceV1,
                QualityDiversityCandidateV1,
                Mapping[str, Any],
                Mapping[str, Any],
                Mapping[str, Any],
            ]
        ] = []
        for item in eligible:
            trace_reference = dict(item.get("traceArtifact") or {})
            trace = self._trace(trace_reference)
            funnel_reference = dict(item.get("analysisFunnelArtifact") or {})
            funnel_document = self._load_document(
                funnel_reference,
                owner_kind="foundry_analysis_funnel_v1",
                media_type=_FUNNEL_MEDIA_TYPE,
            )
            funnel = FoundryAnalysisFunnelResultV1.from_dict(funnel_document)
            if (
                str(funnel_reference.get("ownerHash") or "") != funnel.sha256
                or funnel.analysis_profile_hash != expected_analysis_profile_hash
                or trace.sha256 not in set(funnel.trace_hashes)
            ):
                raise ValueError("finalist funnel authority or trace membership mismatch")
            analysis_references = [
                dict(reference) for reference in item.get("analysisArtifacts") or ()
                if dict(reference).get("ownerKind") == "trace_analysis_v1"
            ]
            if len(analysis_references) != 1:
                raise ValueError("finalist candidate requires one trace analysis artifact")
            analysis_reference = analysis_references[0]
            analysis_document = self._load_document(
                analysis_reference,
                owner_kind="trace_analysis_v1",
                media_type=_ANALYSIS_MEDIA_TYPE,
            )
            analysis_hash = str(analysis_document.get("sha256") or "")
            if (
                analysis_hash != str(analysis_reference.get("ownerHash") or "")
                or analysis_hash not in set(funnel.analysis_hashes)
                or str(analysis_document.get("traceHash") or "") != trace.sha256
            ):
                raise ValueError("finalist analysis membership mismatch")
            analysis_evidence = dict(item.get("analysisEvidence") or {})
            if analysis_evidence and (
                str(analysis_evidence.get("descendantTraceHash") or "")
                != trace.sha256
                or str(analysis_evidence.get("funnelHash") or "")
                != funnel.sha256
                or str(analysis_evidence.get("analysisProfileHash") or "")
                != expected_analysis_profile_hash
            ):
                raise ValueError("finalist analysis evidence authority mismatch")
            for candidate in funnel.selection.selected:
                # The public duplicate player and current Remotion decision
                # composition support card-play choices. Bidding evidence is
                # retained in the Foundry, but cannot win a request that
                # promises a playable and renderable local preview.
                if dict(candidate.dimensions).get("decision_kind") != "play":
                    continue
                if candidate.trace_hash != trace.sha256:
                    raise ValueError("finalist candidate trace membership mismatch")
                prepared.append((
                    float(candidate.selection_score.score),
                    trace,
                    candidate,
                    analysis_reference,
                    analysis_document,
                    item,
                ))
        if not prepared:
            raise NoPlayableFinalistCandidateError(
                "finalist validation has no surviving playable candidate"
            )
        _, trace, candidate, analysis_reference, analysis_document, finalist_item = max(
            prepared,
            key=lambda item: (item[0], item[2].sha256),
        )
        result = self._prepare_verified_candidate(
            trace=trace,
            candidate=candidate,
            analysis_reference=analysis_reference,
            analysis_document=analysis_document,
            expected_analysis_profile_hash=expected_analysis_profile_hash,
            reveal_policy=reveal_policy,
            export_preset=export_preset,
        )
        publication_blockers = tuple(
            dict.fromkeys(str(value) for value in finalist_item.get("blockers") or ())
        )
        evidence = dict(finalist_item.get("evidence") or {})
        analysis_evidence = dict(finalist_item.get("analysisEvidence") or {})
        return {
            **result,
            "qualification": {
                "schemaVersion": 1,
                "localPreviewReady": bool(finalist_item.get("localPreviewReady")),
                "publicationReady": bool(finalist_item.get("publicationReady")),
                "publicationBlockers": list(publication_blockers),
                "finalistManifestHash": str(result.get("manifestHash") or finalist_job.get("result", {}).get("manifestHash") or ""),
                "replayEvidenceHash": str(evidence.get("sha256") or ""),
                "analysisEvidenceHash": str(analysis_evidence.get("sha256") or ""),
            },
        }
