"""Evidence and profile-specific ranking for complete natural Sakkas."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Mapping

from .evidence_v3 import LaffaEvidenceV2, TruthMode, canonical_sha256
from .profiles import ContentSelectionScoreV1
from .sakka_context import NaturalSakkaV1
from .trace import CompleteGameTraceV1


_DIMENSIONS = frozenset({
    "minimumLaffaQuality",
    "meanLaffaQuality",
    "peakLaffaQuality",
    "criticalDensity",
    "perspectiveCoverage",
    "teamOpportunityCoverage",
    "leadChangeRate",
    "matchTension",
    "contractDiversity",
    "projectDiversity",
    "tacticalDiversity",
    "finalRoundConsequence",
})


@dataclass(frozen=True, slots=True)
class NaturalSakkaEvidenceV1:
    sakka_hash: str
    round_references: tuple[Mapping[str, Any], ...]
    round_count: int
    minimum_laffa_quality: float
    mean_laffa_quality: float
    peak_laffa_quality: float
    critical_density: float
    perspective_coverage: float
    team_opportunity_coverage: float
    lead_change_rate: float
    match_tension: float
    contract_diversity: float
    project_diversity: float
    tactical_diversity: float
    final_round_consequence: float
    critical_decision_count: int
    sha256: str

    @classmethod
    def build(
        cls,
        sakka: NaturalSakkaV1,
        *,
        traces: Iterable[CompleteGameTraceV1],
        laffa_evidence: Mapping[str, LaffaEvidenceV2],
        laffa_selection_scores: Mapping[str, ContentSelectionScoreV1],
    ) -> "NaturalSakkaEvidenceV1":
        trace_values = tuple(traces)
        expected_hashes = tuple(str(item["traceHash"]) for item in sakka.rounds)
        if tuple(item.sha256 for item in trace_values) != expected_hashes:
            raise ValueError("Sakka evidence trace order does not match natural history")
        if set(laffa_evidence) != set(expected_hashes):
            raise ValueError("Sakka evidence requires one Laffa evidence record per round")
        if set(laffa_selection_scores) != set(expected_hashes):
            raise ValueError("Sakka evidence requires one Laffa selection score per round")
        for trace in trace_values:
            evidence = laffa_evidence[trace.sha256]
            score = laffa_selection_scores[trace.sha256]
            if (
                evidence.game_id != trace.game_id
                or evidence.ledger_hash != trace.decision_ledger.sha256
            ):
                raise ValueError("Sakka Laffa evidence does not match its complete trace")
            if score.evidence_hash != evidence.sha256:
                raise ValueError("Sakka Laffa score does not reference its evidence")
        profile_hashes = {
            item.profile_hash for item in laffa_selection_scores.values()
        }
        if len(profile_hashes) != 1:
            raise ValueError("Sakka Laffa quality scores must use one selection profile")
        if any(
            item.score_family != "laffa_content_selection"
            for item in laffa_selection_scores.values()
        ):
            raise ValueError("Sakka evidence requires labeled Laffa content scores")

        qualities = [
            float(laffa_selection_scores[trace_hash].score)
            for trace_hash in expected_hashes
        ]
        critical_count = sum(
            len(laffa_evidence[trace_hash].critical_decision_ids)
            for trace_hash in expected_hashes
        )
        meaningful_count = sum(
            1
            for trace_hash in expected_hashes
            for item in laffa_evidence[trace_hash].perspectives
            if item.truth_mode == TruthMode.ACTOR_OBSERVER and not item.forced
        )
        critical_seats = {
            item.acting_seat
            for trace_hash in expected_hashes
            for item in laffa_evidence[trace_hash].perspectives
            if item.truth_mode == TruthMode.ACTOR_OBSERVER
            and item.decision_id in laffa_evidence[trace_hash].critical_decision_ids
        }
        critical_teams = {seat.team for seat in critical_seats}
        contracts = {
            str(trace.trusted_final_state.get("contract") or "unknown")
            for trace in trace_values
        }
        project_kinds = {
            str(project.get("kind") or project.get("project") or "unknown")
            for trace in trace_values
            for project in trace.trusted_final_state.get("projects") or ()
        }
        tactical_kinds = {
            node.kind
            for trace_hash in expected_hashes
            for node in (
                laffa_evidence[trace_hash].scenario_graph.nodes
                if laffa_evidence[trace_hash].scenario_graph is not None else ()
            )
        }
        tension_values = [
            max(
                0.0,
                100.0
                * (1.0 - abs(int(item["startingScore"][0]) - int(item["startingScore"][1])) / sakka.target),
            )
            for item in sakka.rounds
        ]
        lead_change_rate = (
            100.0 * sakka.lead_changes / max(1, len(sakka.rounds) - 1)
        )
        dimensions = {
            "minimumLaffaQuality": min(qualities),
            "meanLaffaQuality": fmean(qualities),
            "peakLaffaQuality": max(qualities),
            "criticalDensity": 100.0 * critical_count / meaningful_count if meaningful_count else 0.0,
            "perspectiveCoverage": 25.0 * len(critical_seats),
            "teamOpportunityCoverage": 50.0 * len(critical_teams),
            "leadChangeRate": min(100.0, lead_change_rate),
            "matchTension": fmean(tension_values),
            "contractDiversity": min(100.0, 100.0 * len(contracts) / 3.0),
            "projectDiversity": min(100.0, 25.0 * len(project_kinds)),
            "tacticalDiversity": min(100.0, 20.0 * len(tactical_kinds)),
            "finalRoundConsequence": 100.0,
        }
        references = tuple({
            "ordinal": index + 1,
            "traceHash": trace_hash,
            "laffaEvidenceHash": laffa_evidence[trace_hash].sha256,
            "laffaSelectionScoreHash": laffa_selection_scores[trace_hash].sha256,
        } for index, trace_hash in enumerate(expected_hashes))
        core = {
            "schemaVersion": 1,
            "sakkaHash": sakka.sha256,
            "roundReferences": [dict(item) for item in references],
            "roundCount": len(expected_hashes),
            **dimensions,
            "criticalDecisionCount": critical_count,
        }
        return cls(
            sakka.sha256,
            references,
            len(expected_hashes),
            dimensions["minimumLaffaQuality"],
            dimensions["meanLaffaQuality"],
            dimensions["peakLaffaQuality"],
            dimensions["criticalDensity"],
            dimensions["perspectiveCoverage"],
            dimensions["teamOpportunityCoverage"],
            dimensions["leadChangeRate"],
            dimensions["matchTension"],
            dimensions["contractDiversity"],
            dimensions["projectDiversity"],
            dimensions["tacticalDiversity"],
            dimensions["finalRoundConsequence"],
            critical_count,
            canonical_sha256(core),
        )

    def dimensions(self) -> dict[str, float]:
        return {
            "minimumLaffaQuality": self.minimum_laffa_quality,
            "meanLaffaQuality": self.mean_laffa_quality,
            "peakLaffaQuality": self.peak_laffa_quality,
            "criticalDensity": self.critical_density,
            "perspectiveCoverage": self.perspective_coverage,
            "teamOpportunityCoverage": self.team_opportunity_coverage,
            "leadChangeRate": self.lead_change_rate,
            "matchTension": self.match_tension,
            "contractDiversity": self.contract_diversity,
            "projectDiversity": self.project_diversity,
            "tacticalDiversity": self.tactical_diversity,
            "finalRoundConsequence": self.final_round_consequence,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "sakkaHash": self.sakka_hash,
            "roundReferences": [dict(item) for item in self.round_references],
            "roundCount": self.round_count,
            **self.dimensions(),
            "criticalDecisionCount": self.critical_decision_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SakkaSelectionProfileV1:
    profile_id: str
    revision: str
    weights: tuple[tuple[str, float], ...]
    minimum_rounds: int
    minimum_laffa_quality: float
    minimum_final_round_quality: float
    require_both_team_opportunities: bool
    minimum_score: float
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "SakkaSelectionProfileV1":
        allowed = {
            "schemaVersion", "id", "revision", "weights", "minimumRounds",
            "minimumLaffaQuality", "minimumFinalRoundQuality",
            "requireBothTeamOpportunities", "minimumScore",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown Sakka selection profile keys: {sorted(unknown)}")
        if int(value.get("schemaVersion", 1)) != 1:
            raise ValueError("unsupported Sakka selection profile schema version")
        weights = tuple(sorted(
            (str(key), float(weight))
            for key, weight in dict(value.get("weights") or {}).items()
        ))
        if not weights or any(weight <= 0 for _, weight in weights):
            raise ValueError("Sakka selection weights must be positive and non-empty")
        unsupported = {key for key, _ in weights} - _DIMENSIONS
        if unsupported:
            raise ValueError(f"unsupported Sakka evidence dimensions: {sorted(unsupported)}")
        minimum_rounds = int(value.get("minimumRounds", 1))
        minimum_laffa = float(value.get("minimumLaffaQuality", 0.0))
        minimum_final = float(value.get("minimumFinalRoundQuality", 0.0))
        minimum_score = float(value.get("minimumScore", 0.0))
        if minimum_rounds < 1 or any(
            not 0.0 <= item <= 100.0
            for item in (minimum_laffa, minimum_final, minimum_score)
        ):
            raise ValueError("Sakka selection gates are out of range")
        core = {
            "schemaVersion": 1,
            "id": str(value["id"]),
            "revision": str(value["revision"]),
            "weights": dict(weights),
            "minimumRounds": minimum_rounds,
            "minimumLaffaQuality": minimum_laffa,
            "minimumFinalRoundQuality": minimum_final,
            "requireBothTeamOpportunities": bool(value.get("requireBothTeamOpportunities", True)),
            "minimumScore": minimum_score,
        }
        return cls(
            core["id"], core["revision"], weights, minimum_rounds,
            minimum_laffa, minimum_final,
            core["requireBothTeamOpportunities"], minimum_score,
            canonical_sha256(core),
        )

    def score(
        self,
        evidence: NaturalSakkaEvidenceV1,
        *,
        final_round_quality: float,
    ) -> ContentSelectionScoreV1:
        final_quality = float(final_round_quality)
        if not 0.0 <= final_quality <= 100.0:
            raise ValueError("final round quality must be between zero and 100")
        dimensions = evidence.dimensions()
        score = sum(dimensions[key] * weight for key, weight in self.weights) / sum(
            weight for _, weight in self.weights
        )
        reasons: list[str] = []
        if evidence.round_count < self.minimum_rounds:
            reasons.append("round_count")
        if evidence.minimum_laffa_quality < self.minimum_laffa_quality:
            reasons.append("filler_laffa")
        if final_quality < self.minimum_final_round_quality:
            reasons.append("final_round_quality")
        if self.require_both_team_opportunities and evidence.team_opportunity_coverage < 100.0:
            reasons.append("team_opportunity_coverage")
        if score < self.minimum_score:
            reasons.append("minimum_score")
        return ContentSelectionScoreV1.build(
            score_family="sakka_content_selection",
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
            "minimumRounds": self.minimum_rounds,
            "minimumLaffaQuality": self.minimum_laffa_quality,
            "minimumFinalRoundQuality": self.minimum_final_round_quality,
            "requireBothTeamOpportunities": self.require_both_team_opportunities,
            "minimumScore": self.minimum_score,
            "sha256": self.sha256,
        }
