"""Verified natural-history search for critical Sakka contexts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from baloot_platform.core.enums import GamePhase, Seat
from baloot_platform.plugins.base import PluginDescriptor

from .analysis_pipeline import StagedTraceAnalyzerV1, TraceDecisionAnalysisV3
from .evidence_v3 import DecisionLedgerRecordV1, canonical_sha256
from .sakka_context import ContextualizedLaffaV1, SakkaContextV1
from .simulation import NaturalLaffaSimulatorV1
from .score_reachability import VerifiedEnumeratedContextV1
from .trace import CompleteGameTraceV1


def _start(trace: CompleteGameTraceV1) -> tuple[tuple[int, int], Seat, int]:
    generation = dict(trace.trusted_final_state.get("generation") or {})
    score = tuple(int(item) for item in generation.get("matchScore") or ())
    if len(score) != 2:
        raise ValueError("trace lacks a two-team starting match score")
    return score, Seat.parse(str(generation.get("dealer") or "")), int(generation.get("roundNumber", 0))


def _finish(trace: CompleteGameTraceV1) -> tuple[int, int]:
    score = tuple(int(item) for item in trace.trusted_final_state.get("matchScore") or ())
    if len(score) != 2:
        raise ValueError("trace lacks a two-team final match score")
    return score


def _action_key(value: Mapping[str, Any]) -> str:
    if "action" in value and set(value).intersection({"trumpSuit", "action"}):
        return f"bid:{value.get('action')}:{value.get('trumpSuit') or ''}"
    return canonical_sha256(dict(value))


@dataclass(frozen=True, slots=True)
class CriticalContextSearchProfileV1:
    profile_id: str
    revision: str
    minimum_analysis_confidence: float
    minimum_match_probability_amplification: float
    require_same_recommendation: bool
    maximum_contexts: int
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "CriticalContextSearchProfileV1":
        allowed = {
            "schemaVersion", "id", "revision", "minimumAnalysisConfidence",
            "minimumMatchProbabilityAmplification", "requireSameRecommendation",
            "maximumContexts",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown context search profile keys: {sorted(unknown)}")
        if int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported context search profile schema version")
        confidence = float(value.get("minimumAnalysisConfidence", -1.0))
        amplification = float(value.get("minimumMatchProbabilityAmplification", -1.0))
        maximum = int(value.get("maximumContexts", 0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("context search confidence must be in [0, 1]")
        if not 0.0 <= amplification <= 1.0:
            raise ValueError("context amplification must be in [0, 1]")
        if not 1 <= maximum <= 10_000:
            raise ValueError("context search maximum contexts must be in [1, 10000]")
        if not bool(value.get("requireSameRecommendation", True)):
            raise ValueError("contextualized challenges require a stable recommendation")
        core = {
            "schemaVersion": 1,
            "id": str(value.get("id") or ""),
            "revision": str(value.get("revision") or ""),
            "minimumAnalysisConfidence": confidence,
            "minimumMatchProbabilityAmplification": amplification,
            "requireSameRecommendation": bool(value.get("requireSameRecommendation", True)),
            "maximumContexts": maximum,
        }
        if not core["id"] or not core["revision"]:
            raise ValueError("context search profile identity is required")
        return cls(
            core["id"], core["revision"], confidence, amplification,
            core["requireSameRecommendation"], maximum, canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "minimumAnalysisConfidence": self.minimum_analysis_confidence,
            "minimumMatchProbabilityAmplification": self.minimum_match_probability_amplification,
            "requireSameRecommendation": self.require_same_recommendation,
            "maximumContexts": self.maximum_contexts,
            "sha256": self.sha256,
        }


class NaturalHistoryContextVerifierV1:
    """Build a context only from an uninterrupted authoritative history."""

    descriptor = PluginDescriptor(
        id="natural_history_context_verifier_v1",
        kind="context_search",
        version="1.0.0",
        description="Prove a Sakka starting context from complete scored prior rounds",
        capabilities=("natural_history", "score_continuity", "dealer_continuity"),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=60,
        maximum_memory_mb=256,
    )

    @classmethod
    def build_context(
        cls,
        previous_rounds: Iterable[CompleteGameTraceV1],
        *,
        target: int,
    ) -> "VerifiedNaturalContextV1":
        traces = tuple(previous_rounds)
        if not traces:
            raise ValueError("natural history context requires preceding rounds")
        expected_score = (0, 0)
        expected_round = 1
        expected_dealer: Seat | None = None
        identities: set[tuple[str, str, str, tuple[tuple[str, str], ...]]] = set()
        for trace in traces:
            if str(trace.trusted_final_state.get("phase")) != GamePhase.SCORED.value:
                raise ValueError("natural history contains an incomplete Laffa")
            start_score, dealer, round_number = _start(trace)
            if start_score != expected_score or round_number != expected_round:
                raise ValueError("natural history score or round continuity is broken")
            if expected_dealer is not None and dealer != expected_dealer:
                raise ValueError("natural history dealer continuity is broken")
            if max(start_score) >= int(target):
                raise ValueError("natural history continues after match termination")
            identities.add((trace.rules_hash, trace.bundle_hash, trace.engine_version, trace.agent_profiles))
            expected_score = _finish(trace)
            expected_round += 1
            expected_dealer = Seat.parse(str(trace.trusted_final_state["dealer"])).next()
        if len(identities) != 1:
            raise ValueError("natural history changed rules, configuration, engine, or agents")
        if max(expected_score) >= int(target):
            raise ValueError("natural history already reached the declared target")
        proof_core = {
            "schemaVersion": 1,
            "verifierId": cls.descriptor.id,
            "verifierVersion": cls.descriptor.version,
            "target": int(target),
            "roundHashes": [trace.sha256 for trace in traces],
            "nextScore": list(expected_score),
            "nextDealer": expected_dealer.name.lower() if expected_dealer else None,
            "nextRoundNumber": expected_round,
        }
        context = SakkaContextV1.build(
            starting_score=expected_score,
            dealer=expected_dealer,
            round_number=expected_round,
            target=int(target),
            reachability_mode="natural_history",
            reachability_proof_hash=canonical_sha256(proof_core),
            previous_round_hashes=(trace.sha256 for trace in traces),
        )
        rules_hash, bundle_hash, engine_version, agent_profiles = next(iter(identities))
        core = {
            "schemaVersion": 1,
            "contextHash": context.sha256,
            "historyTraceHashes": [trace.sha256 for trace in traces],
            "rulesHash": rules_hash,
            "bundleHash": bundle_hash,
            "engineVersion": engine_version,
            "agentProfiles": dict(agent_profiles),
            "verifierId": cls.descriptor.id,
            "verifierVersion": cls.descriptor.version,
        }
        return VerifiedNaturalContextV1(
            context,
            tuple(trace.sha256 for trace in traces),
            rules_hash,
            bundle_hash,
            engine_version,
            agent_profiles,
            canonical_sha256(core),
        )


@dataclass(frozen=True, slots=True)
class VerifiedNaturalContextV1:
    context: SakkaContextV1
    history_trace_hashes: tuple[str, ...]
    rules_hash: str
    bundle_hash: str
    engine_version: str
    agent_profiles: tuple[tuple[str, str], ...]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "context": self.context.to_dict(),
            "historyTraceHashes": list(self.history_trace_hashes),
            "rulesHash": self.rules_hash,
            "bundleHash": self.bundle_hash,
            "engineVersion": self.engine_version,
            "agentProfiles": dict(self.agent_profiles),
            "verifierId": NaturalHistoryContextVerifierV1.descriptor.id,
            "verifierVersion": NaturalHistoryContextVerifierV1.descriptor.version,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class CriticalContextSearchResultV1:
    parent_trace_hash: str
    original_decision_id: str
    profile_hash: str
    analysis_profile_hash: str
    evaluated: tuple[Mapping[str, Any], ...]
    accepted: tuple[ContextualizedLaffaV1, ...]
    selected_hash: str | None
    sha256: str
    descendant_traces: tuple[CompleteGameTraceV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "parentTraceHash": self.parent_trace_hash,
            "originalDecisionId": self.original_decision_id,
            "profileHash": self.profile_hash,
            "analysisProfileHash": self.analysis_profile_hash,
            "evaluated": [dict(item) for item in self.evaluated],
            "accepted": [item.to_dict() for item in self.accepted],
            "selectedHash": self.selected_hash,
            "sha256": self.sha256,
        }


class CriticalContextSearchV1:
    descriptor = PluginDescriptor(
        id="critical_context_search_v1",
        kind="context_search",
        version="1.0.0",
        description="Re-simulate an exact Laffa under proven natural or enumerated Sakka contexts",
        capabilities=(
            "exact_deal_resimulation", "natural_history",
            "verified_enumeration",
            "actor_observer_reanalysis", "match_probability_amplification",
        ),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=900,
        maximum_memory_mb=4096,
    )

    def __init__(
        self,
        simulator: NaturalLaffaSimulatorV1,
        analyzer: StagedTraceAnalyzerV1,
        profile: CriticalContextSearchProfileV1,
    ):
        if simulator.rules.sha256 != analyzer.rules.sha256:
            raise ValueError("context simulator and analyzer must use the same rules")
        self.simulator = simulator
        self.analyzer = analyzer
        self.profile = profile
        match_profile = dict(getattr(analyzer, "match_value_profile", {}) or {})
        self.match_target = int(match_profile["target"]) if "target" in match_profile else None

    @staticmethod
    def _record(trace: CompleteGameTraceV1, decision_id: str) -> DecisionLedgerRecordV1:
        record = next(
            (item for item in trace.decision_ledger.records if item.decision_id == decision_id),
            None,
        )
        if record is None:
            raise ValueError("critical decision is absent from the parent trace")
        return record

    @staticmethod
    def _structural_ordinal(trace: CompleteGameTraceV1, record: DecisionLedgerRecordV1) -> int:
        matches = [
            item for item in trace.decision_ledger.records
            if item.decision_kind == record.decision_kind and item.acting_seat == record.acting_seat
        ]
        return next(index for index, item in enumerate(matches) if item.decision_id == record.decision_id)

    @staticmethod
    def _mapped_record(
        trace: CompleteGameTraceV1,
        source: DecisionLedgerRecordV1,
        occurrence: int,
    ) -> DecisionLedgerRecordV1 | None:
        matches = [
            item for item in trace.decision_ledger.records
            if item.decision_kind == source.decision_kind and item.acting_seat == source.acting_seat
        ]
        return matches[occurrence] if occurrence < len(matches) else None

    @staticmethod
    def _analysis(decisions: Iterable[TraceDecisionAnalysisV3], decision_id: str) -> TraceDecisionAnalysisV3:
        analysis = next(
            (item for item in decisions if item.decision_id == decision_id),
            None,
        )
        if analysis is None:
            raise ValueError("critical decision lacks actor-safe analysis")
        return analysis

    @staticmethod
    def _match_swing(item: TraceDecisionAnalysisV3) -> float:
        values = [
            float(row["matchWinProbability"])
            for row in item.alternatives
            if "matchWinProbability" in row
        ]
        if len(values) < 2 or "match_win_probability" not in item.supported_metrics:
            raise ValueError("critical context search currently requires supported play match probabilities")
        return max(values) - min(values)

    def search(
        self,
        parent: CompleteGameTraceV1,
        *,
        decision_id: str,
        contexts: Iterable[VerifiedNaturalContextV1 | VerifiedEnumeratedContextV1],
        cancel_check: Callable[[], bool] | None = None,
    ) -> CriticalContextSearchResultV1:
        candidates = tuple(contexts)
        if not candidates:
            raise ValueError("critical context search requires at least one proven context")
        if len(candidates) > self.profile.maximum_contexts:
            raise ValueError("critical context search exceeds its governed context bound")
        accepted_types = (VerifiedNaturalContextV1, VerifiedEnumeratedContextV1)
        if any(not isinstance(item, accepted_types) for item in candidates):
            raise ValueError("context search requires verified reachability artifacts")
        for verified in candidates:
            if isinstance(verified, VerifiedNaturalContextV1):
                if verified.context.reachability_mode != "natural_history":
                    raise ValueError("natural context has the wrong proof mode")
                if verified.history_trace_hashes != verified.context.previous_round_hashes:
                    raise ValueError("verified context history binding is inconsistent")
                if (
                    verified.rules_hash != parent.rules_hash
                    or verified.bundle_hash != parent.bundle_hash
                    or verified.engine_version != parent.engine_version
                    or verified.agent_profiles != parent.agent_profiles
                ):
                    raise ValueError("verified natural context is incompatible with the parent Laffa")
            else:
                verified.verify()
                proof = verified.proof
                if verified.context.reachability_mode != "verified_enumeration":
                    raise ValueError("enumerated context has the wrong proof mode")
                if verified.context.reachability_proof_hash != proof.sha256:
                    raise ValueError("enumerated context proof binding is inconsistent")
                if (
                    proof.rules_hash != parent.rules_hash
                    or proof.bundle_hash != parent.bundle_hash
                    or proof.engine_version != parent.engine_version
                ):
                    raise ValueError("verified enumerated context is incompatible with the parent Laffa")
            if self.match_target is not None and verified.context.target != self.match_target:
                raise ValueError("verified context target does not match the analysis model")
        parent_record = self._record(parent, decision_id)
        if parent_record.forced:
            raise ValueError("forced decisions cannot become critical context challenges")
        occurrence = self._structural_ordinal(parent, parent_record)
        parent_result = self.analyzer.analyze_trace(parent, cancel_check=cancel_check)
        parent_item = self._analysis(parent_result.decisions, decision_id)
        parent_swing = self._match_swing(parent_item)
        evaluated: list[dict[str, Any]] = []
        accepted_pairs: list[tuple[ContextualizedLaffaV1, CompleteGameTraceV1]] = []

        parent_dealer = Seat.parse(str(parent.trusted_final_state["generation"]["dealer"]))
        for verified in sorted(candidates, key=lambda item: item.sha256):
            context = verified.context
            if cancel_check is not None and cancel_check():
                raise InterruptedError("critical context search cancelled")
            if context.dealer != parent_dealer:
                evaluated.append({
                    "contextHash": context.sha256,
                    "startingScore": list(context.starting_score),
                    "accepted": False,
                    "rejectionReason": "dealer_incompatible_with_frozen_deal",
                })
                continue
            descendant = self.simulator.resimulate_exact_deal(
                parent,
                match_score=context.starting_score,
                round_number=context.round_number,
                cancel_check=cancel_check,
            )
            row: dict[str, Any] = {
                "contextHash": context.sha256,
                "startingScore": list(context.starting_score),
                "descendantTraceHash": descendant.sha256,
                "accepted": False,
            }
            if str(descendant.trusted_final_state.get("phase")) != GamePhase.SCORED.value:
                row["rejectionReason"] = "context_changed_auction_to_all_pass"
                evaluated.append(row)
                continue
            mapped = self._mapped_record(descendant, parent_record, occurrence)
            if mapped is None:
                row["rejectionReason"] = "critical_decision_no_longer_exists"
                evaluated.append(row)
                continue
            descendant_result = self.analyzer.analyze_trace(descendant, cancel_check=cancel_check)
            descendant_item = self._analysis(descendant_result.decisions, mapped.decision_id)
            stable = _action_key(parent_item.best_action) == _action_key(descendant_item.best_action)
            amplification = self._match_swing(descendant_item) - parent_swing
            row.update({
                "descendantDecisionId": mapped.decision_id,
                "recommendationStable": stable,
                "analysisConfidence": descendant_item.confidence,
                "parentMatchProbabilitySwing": parent_swing,
                "descendantMatchProbabilitySwing": parent_swing + amplification,
                "matchProbabilityAmplification": amplification,
            })
            if descendant_item.confidence < self.profile.minimum_analysis_confidence:
                row["rejectionReason"] = "insufficient_analysis_confidence"
            elif self.profile.require_same_recommendation and not stable:
                row["rejectionReason"] = "recommendation_changed"
            elif amplification < self.profile.minimum_match_probability_amplification:
                row["rejectionReason"] = "insufficient_context_amplification"
            else:
                analysis_hash = canonical_sha256({
                    "parentAnalysisHash": parent_result.sha256,
                    "descendantAnalysisHash": descendant_result.sha256,
                    "profileHash": self.profile.sha256,
                    "originalDecisionId": decision_id,
                    "descendantDecisionId": mapped.decision_id,
                })
                contextualized = ContextualizedLaffaV1.build(
                    parent=parent,
                    descendant=descendant,
                    context=context,
                    original_decision_id=decision_id,
                    descendant_decision_id=mapped.decision_id,
                    analysis_hash=analysis_hash,
                    context_amplification={
                        "matchProbabilitySwing": 100.0 * amplification,
                    },
                    strategy_changed=(
                        _action_key(parent_record.chosen_action) != _action_key(mapped.chosen_action)
                    ),
                    recommendation_stable=stable,
                )
                row["accepted"] = True
                row["contextualizedLaffaHash"] = contextualized.sha256
                accepted_pairs.append((contextualized, descendant))
            evaluated.append(row)

        accepted_pairs.sort(
            key=lambda pair: (
                -float(pair[0].context_amplification["matchProbabilitySwing"]),
                pair[0].sha256,
            )
        )
        accepted = tuple(item for item, _ in accepted_pairs)
        descendants = tuple(trace for _, trace in accepted_pairs)
        core = {
            "schemaVersion": 1,
            "parentTraceHash": parent.sha256,
            "originalDecisionId": decision_id,
            "profileHash": self.profile.sha256,
            "analysisProfileHash": self.analyzer.profile.sha256,
            "evaluated": evaluated,
            "acceptedHashes": [item.sha256 for item in accepted],
            "selectedHash": accepted[0].sha256 if accepted else None,
        }
        return CriticalContextSearchResultV1(
            parent.sha256,
            decision_id,
            self.profile.sha256,
            self.analyzer.profile.sha256,
            tuple(evaluated),
            accepted,
            core["selectedHash"],
            canonical_sha256(core),
            descendants,
        )
