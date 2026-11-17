"""Canonical multi-perspective challenge evidence contracts.

These contracts are derived from the trusted game trace. They never replace
engine state, rules, scoring, or history. Actor recommendations and observer
experience are deliberately different truth modes so private information and
decision authority cannot be mixed accidentally.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from baloot_platform.analysis.models import InformationState
from baloot_platform.analysis.search_errors import require_uncensored_play_trace
from baloot_platform.core.enums import Seat, Team


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _seat(value: Seat | str) -> Seat:
    return value if isinstance(value, Seat) else Seat.parse(value)


def relative_team_value(team_values: Sequence[float], perspective: Seat | Team) -> float:
    team = perspective.team if isinstance(perspective, Seat) else perspective
    if len(team_values) != 2:
        raise ValueError("team values must contain exactly two entries")
    return float(team_values[int(team)]) - float(team_values[int(team.other)])


class TruthMode(str, Enum):
    ACTOR_OBSERVER = "actor_observer"
    SEAT_EXPERIENCE = "seat_experience"
    ACTUAL_DEAL_ORACLE = "actual_deal_oracle"


class TruthLabel(str, Enum):
    EXACT_OPTIMAL = "exact_optimal"
    ROBUST_BEST_KNOWN = "robust_best_known"
    DEBATABLE = "debatable"
    UNRESOLVED = "unresolved"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class DecisionLedgerRecordV1:
    decision_id: str
    ordinal: int
    acting_seat: Seat
    decision_kind: str
    phase: str
    public_prefix_hash: str
    complete_state_hash: str | None
    legal_actions: tuple[Mapping[str, Any], ...]
    chosen_action: Mapping[str, Any]
    forced: bool
    resulting_event_hash: str | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DecisionLedgerRecordV1":
        require_uncensored_play_trace(value.get("decisionMetadata") or {})
        legal = tuple(dict(item) for item in value.get("legalActions", ()))
        chosen = value.get("selectedAction") or value.get("chosenAction")
        if not isinstance(chosen, Mapping):
            raise ValueError("decision ledger chosen action must be a mapping")
        actor = value.get("actor") or value.get("actingSeat")
        if not actor:
            raise ValueError("decision ledger acting seat is required")
        forced = len(legal) <= 1
        if "forced" in value and bool(value["forced"]) != forced:
            raise ValueError("decision ledger forcedness does not match its legal actions")
        public_state = value.get("publicState") or {}
        prefix_hash = str(value.get("publicPrefixHash") or canonical_sha256(public_state))
        return cls(
            decision_id=str(value["decisionId"]),
            ordinal=int(value["ordinal"]),
            acting_seat=_seat(str(actor)),
            decision_kind=str(value["decisionKind"]),
            phase=str(value.get("phase", "unknown")),
            public_prefix_hash=prefix_hash,
            complete_state_hash=(str(value["completeStateHash"]) if value.get("completeStateHash") else None),
            legal_actions=legal,
            chosen_action=dict(chosen),
            forced=forced,
            resulting_event_hash=(str(value["resultingEventHash"]) if value.get("resultingEventHash") else None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisionId": self.decision_id,
            "ordinal": self.ordinal,
            "actingSeat": self.acting_seat.name.lower(),
            "actingTeam": self.acting_seat.team.name.lower(),
            "decisionKind": self.decision_kind,
            "phase": self.phase,
            "publicPrefixHash": self.public_prefix_hash,
            "completeStateHash": self.complete_state_hash,
            "legalActions": [dict(item) for item in self.legal_actions],
            "chosenAction": dict(self.chosen_action),
            "forced": self.forced,
            "resultingEventHash": self.resulting_event_hash,
        }


@dataclass(frozen=True, slots=True)
class DecisionLedgerV1:
    game_id: str
    records: tuple[DecisionLedgerRecordV1, ...]
    sha256: str

    @classmethod
    def build(cls, game_id: str, records: Iterable[Mapping[str, Any] | DecisionLedgerRecordV1]) -> "DecisionLedgerV1":
        normalized = tuple(
            item if isinstance(item, DecisionLedgerRecordV1) else DecisionLedgerRecordV1.from_mapping(item)
            for item in records
        )
        if not normalized:
            raise ValueError("decision ledger cannot be empty")
        ordinals = [item.ordinal for item in normalized]
        if ordinals != sorted(ordinals) or len(ordinals) != len(set(ordinals)):
            raise ValueError("decision ledger ordinals must be unique and ordered")
        if len({item.decision_id for item in normalized}) != len(normalized):
            raise ValueError("decision ledger IDs must be unique")
        payload = {"schemaVersion": 1, "gameId": game_id, "decisions": [item.to_dict() for item in normalized]}
        return cls(str(game_id), normalized, canonical_sha256(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "gameId": self.game_id,
            "decisions": [item.to_dict() for item in self.records],
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ObserverProjectionV1:
    decision_id: str
    observer_seat: Seat
    information_state: Mapping[str, Any]
    sha256: str

    @classmethod
    def from_information_state(cls, decision_id: str, state: InformationState) -> "ObserverProjectionV1":
        payload = state.to_dict()
        if payload.get("observer") != state.observer.name.lower():
            raise ValueError("observer projection identity mismatch")
        return cls(str(decision_id), state.observer, payload, canonical_sha256(payload))

    @classmethod
    def from_public_mapping(cls, decision_id: str, observer: Seat | str, payload: Mapping[str, Any]) -> "ObserverProjectionV1":
        seat = _seat(observer)
        projected = dict(payload)
        if projected.get("observer") != seat.name.lower():
            raise ValueError("observer projection identity mismatch")
        return cls(str(decision_id), seat, projected, canonical_sha256(projected))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "decisionId": self.decision_id,
            "observerSeat": self.observer_seat.name.lower(),
            "informationState": dict(self.information_state),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class DecisionPerspectiveEvidenceV3:
    decision_id: str
    acting_seat: Seat
    observer_seat: Seat
    truth_mode: TruthMode
    projection_hash: str
    forced: bool
    truth_label: TruthLabel
    confidence: float
    evidence: Mapping[str, float]
    recommended_action: Mapping[str, Any] | None = None
    prompt_contract: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("evidence confidence must be between zero and one")
        for key, value in self.evidence.items():
            if not 0.0 <= float(value) <= 100.0:
                raise ValueError(f"evidence dimension {key} must be between 0 and 100")
        if self.truth_mode == TruthMode.ACTOR_OBSERVER:
            if self.observer_seat != self.acting_seat:
                raise ValueError("actor-observer truth must use the acting seat projection")
            if not self.forced and not self.recommended_action:
                raise ValueError("non-forced actor truth requires a recommendation")
        elif self.truth_mode == TruthMode.SEAT_EXPERIENCE:
            if self.observer_seat == self.acting_seat:
                raise ValueError("seat-experience truth is reserved for non-acting observers")
            if self.recommended_action is not None:
                raise ValueError("non-acting observers cannot receive an acting-player recommendation")
            if not str(self.prompt_contract or "").strip():
                raise ValueError("seat-experience evidence requires a typed prompt contract")
        elif self.truth_mode == TruthMode.ACTUAL_DEAL_ORACLE:
            if self.recommended_action is not None:
                raise ValueError(
                    "actual-deal oracle evidence cannot replace the observer recommendation"
                )
            if self.prompt_contract != "post_reveal_analysis":
                raise ValueError(
                    "actual-deal oracle evidence is restricted to post-reveal analysis"
                )

    @property
    def peak(self) -> float | None:
        actor_dimensions = {
            "outcomeImpact", "uniqueness", "disasterAvoidance", "kaboutImpact",
            "sacrificeValue", "tacticalDepth", "informationBurden",
            "executionDifficulty", "surprise", "visualClarity",
        }
        observer_dimensions = {
            "expectedConsequence", "inferenceAbility", "partnerSignalSignificance",
            "opponentThreatVisibility", "threatVisibility", "surpriseAfterAction",
            "nextDecisionUrgency", "educationalValue", "visualValue",
        }
        allowed = (
            actor_dimensions
            if self.truth_mode in {
                TruthMode.ACTOR_OBSERVER,
                TruthMode.ACTUAL_DEAL_ORACLE,
            }
            else observer_dimensions
        )
        return max(
            (float(value) for key, value in self.evidence.items() if key in allowed),
            default=None,
        )

    def selection_score(self, weights: Mapping[str, float]) -> float:
        selected = [(float(self.evidence[key]), float(weight)) for key, weight in weights.items() if key in self.evidence and float(weight) > 0]
        if not selected:
            raise ValueError("selection profile has no supported positive weights")
        return sum(value * weight for value, weight in selected) / sum(weight for _, weight in selected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 3,
            "decisionId": self.decision_id,
            "actingSeat": self.acting_seat.name.lower(),
            "actingTeam": self.acting_seat.team.name.lower(),
            "observerSeat": self.observer_seat.name.lower(),
            "truthMode": self.truth_mode.value,
            "projectionHash": self.projection_hash,
            "forced": self.forced,
            "truthLabel": self.truth_label.value,
            "confidence": self.confidence,
            "evidence": {key: float(value) for key, value in sorted(self.evidence.items())},
            "recommendedAction": dict(self.recommended_action) if self.recommended_action is not None else None,
            "promptContract": self.prompt_contract,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecisionPerspectiveEvidenceV3":
        if int(value.get("schemaVersion", 0)) != 3:
            raise ValueError("unsupported decision perspective evidence schema")
        recommendation = value.get("recommendedAction")
        return cls(
            decision_id=str(value.get("decisionId") or ""),
            acting_seat=_seat(value.get("actingSeat")),
            observer_seat=_seat(value.get("observerSeat")),
            truth_mode=TruthMode(str(value.get("truthMode"))),
            projection_hash=str(value.get("projectionHash") or ""),
            forced=bool(value.get("forced")),
            truth_label=TruthLabel(str(value.get("truthLabel"))),
            confidence=float(value.get("confidence", -1.0)),
            evidence={str(key): float(item) for key, item in dict(value.get("evidence") or {}).items()},
            recommended_action=(dict(recommendation) if isinstance(recommendation, Mapping) else None),
            prompt_contract=(
                str(value["promptContract"]) if value.get("promptContract") is not None else None
            ),
        )


@dataclass(frozen=True, slots=True)
class SeatJourneyEvidenceV1:
    seat: Seat
    decision_count: int
    non_forced_count: int
    peak: float | None
    top_k_mean: float | None
    status: str

    @classmethod
    def build(cls, seat: Seat, evidence: Iterable[DecisionPerspectiveEvidenceV3], *, top_k: int = 3) -> "SeatJourneyEvidenceV1":
        actor_items = [item for item in evidence if item.acting_seat == seat and item.truth_mode == TruthMode.ACTOR_OBSERVER]
        meaningful = sorted((item.peak for item in actor_items if not item.forced and item.peak is not None), reverse=True)
        return cls(
            seat=seat,
            decision_count=len(actor_items),
            non_forced_count=len(meaningful),
            peak=meaningful[0] if meaningful else None,
            top_k_mean=fmean(meaningful[: max(1, int(top_k))]) if meaningful else None,
            status="evaluated" if meaningful else "not_applicable",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "seat": self.seat.name.lower(),
            "decisionCount": self.decision_count,
            "nonForcedCount": self.non_forced_count,
            "peak": self.peak,
            "topKMean": self.top_k_mean,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class TeamJourneyEvidenceV1:
    team: Team
    decision_count: int
    non_forced_count: int
    peak: float | None
    top_k_mean: float | None
    status: str

    @classmethod
    def build(cls, team: Team, evidence: Iterable[DecisionPerspectiveEvidenceV3], *, top_k: int = 5) -> "TeamJourneyEvidenceV1":
        actor_items = [item for item in evidence if item.acting_seat.team == team and item.truth_mode == TruthMode.ACTOR_OBSERVER]
        meaningful = sorted((item.peak for item in actor_items if not item.forced and item.peak is not None), reverse=True)
        return cls(
            team,
            len(actor_items),
            len(meaningful),
            meaningful[0] if meaningful else None,
            fmean(meaningful[: max(1, int(top_k))]) if meaningful else None,
            "evaluated" if meaningful else "not_applicable",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "team": self.team.name.lower(),
            "decisionCount": self.decision_count,
            "nonForcedCount": self.non_forced_count,
            "peak": self.peak,
            "topKMean": self.top_k_mean,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ScenarioNodeV1:
    node_id: str
    ordinal: int
    kind: str
    decision_id: str | None
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodeId": self.node_id,
            "ordinal": self.ordinal,
            "kind": self.kind,
            "decisionId": self.decision_id,
            "evidenceHash": self.evidence_hash,
        }


@dataclass(frozen=True, slots=True)
class ScenarioEdgeV1:
    source: str
    target: str
    relation: str
    strength: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 100.0:
            raise ValueError("scenario edge strength must be between 0 and 100")

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "relation": self.relation, "strength": self.strength}


@dataclass(frozen=True, slots=True)
class ScenarioGraphV1:
    nodes: tuple[ScenarioNodeV1, ...]
    edges: tuple[ScenarioEdgeV1, ...]
    sha256: str

    @classmethod
    def build(cls, nodes: Iterable[ScenarioNodeV1], edges: Iterable[ScenarioEdgeV1]) -> "ScenarioGraphV1":
        node_values = tuple(nodes)
        edge_values = tuple(edges)
        by_id = {item.node_id: item for item in node_values}
        if len(by_id) != len(node_values):
            raise ValueError("scenario graph node IDs must be unique")
        for edge in edge_values:
            if edge.source not in by_id or edge.target not in by_id:
                raise ValueError("scenario graph edge references an unknown node")
            if by_id[edge.source].ordinal > by_id[edge.target].ordinal:
                raise ValueError("scenario graph causal edges cannot travel backwards in time")
            if edge.source == edge.target:
                raise ValueError("scenario graph self edges are forbidden")
        core = {
            "schemaVersion": 1,
            "nodes": [item.to_dict() for item in node_values],
            "edges": [item.to_dict() for item in edge_values],
        }
        return cls(node_values, edge_values, canonical_sha256(core))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScenarioGraphV1":
        graph = cls.build(
            (
                ScenarioNodeV1(
                    str(item.get("nodeId") or ""),
                    int(item.get("ordinal", 0)),
                    str(item.get("kind") or ""),
                    str(item["decisionId"]) if item.get("decisionId") is not None else None,
                    str(item.get("evidenceHash") or ""),
                )
                for item in value.get("nodes") or ()
            ),
            (
                ScenarioEdgeV1(
                    str(item.get("source") or ""),
                    str(item.get("target") or ""),
                    str(item.get("relation") or ""),
                    float(item.get("strength", -1.0)),
                )
                for item in value.get("edges") or ()
            ),
        )
        if value.get("sha256") and str(value["sha256"]) != graph.sha256:
            raise ValueError("scenario graph checksum mismatch")
        return graph


@dataclass(frozen=True, slots=True)
class LaffaEvidenceV2:
    game_id: str
    ledger_hash: str
    perspectives: tuple[DecisionPerspectiveEvidenceV3, ...]
    seat_journeys: tuple[SeatJourneyEvidenceV1, ...]
    team_journeys: tuple[TeamJourneyEvidenceV1, ...]
    critical_decision_ids: tuple[str, ...]
    perspective_breadth: int
    overall_peak: float | None
    second_peak: float | None
    top_k_thrill_mass: float
    critical_density: float
    multi_angle_yield: int
    classification: str
    minimum_critical_confidence: float
    scenario_graph_sha256: str | None
    scenario_graph: ScenarioGraphV1 | None
    sha256: str

    @classmethod
    def build(
        cls,
        ledger: DecisionLedgerV1,
        perspectives: Iterable[DecisionPerspectiveEvidenceV3],
        *,
        critical_threshold: float = 70.0,
        minimum_critical_confidence: float = 0.75,
        require_four_projections: bool = True,
        scenario_graph: ScenarioGraphV1 | None = None,
    ) -> "LaffaEvidenceV2":
        items = tuple(perspectives)
        if not 0.0 <= float(minimum_critical_confidence) <= 1.0:
            raise ValueError("minimum critical confidence must be between zero and one")
        by_decision: dict[str, list[DecisionPerspectiveEvidenceV3]] = {}
        for item in items:
            by_decision.setdefault(item.decision_id, []).append(item)
        expected = {item.decision_id for item in ledger.records}
        if set(by_decision) != expected:
            raise ValueError("laffa evidence must cover every ledger decision exactly by ID")
        for record in ledger.records:
            decision_items = by_decision[record.decision_id]
            actor_truth = [item for item in decision_items if item.truth_mode == TruthMode.ACTOR_OBSERVER]
            if len(actor_truth) != 1 or actor_truth[0].acting_seat != record.acting_seat:
                raise ValueError("every decision requires exactly one acting-seat truth record")
            observers = {item.observer_seat for item in decision_items if item.truth_mode != TruthMode.ACTUAL_DEAL_ORACLE}
            if require_four_projections and observers != set(Seat):
                raise ValueError("every decision requires four isolated observer projections")

        journeys = tuple(SeatJourneyEvidenceV1.build(seat, items) for seat in Seat)
        team_journeys = tuple(TeamJourneyEvidenceV1.build(team, items) for team in Team)
        usable_labels = {
            TruthLabel.EXACT_OPTIMAL,
            TruthLabel.ROBUST_BEST_KNOWN,
            TruthLabel.DEBATABLE,
        }
        actor_items = [
            item for item in items
            if item.truth_mode == TruthMode.ACTOR_OBSERVER
            and not item.forced
            and item.truth_label in usable_labels
        ]
        qualified_actor_items = [
            item for item in actor_items
            if item.confidence >= float(minimum_critical_confidence)
        ]
        # Low-confidence evidence must never inflate the Laffa rank. It remains
        # present for audit and in the density denominator, but only evidence
        # that clears the governed confidence gate contributes to peaks.
        peaks = sorted(
            (
                item.peak
                for item in qualified_actor_items
                if item.peak is not None
            ),
            reverse=True,
        )
        critical = tuple(
            item.decision_id
            for item in qualified_actor_items
            if (item.peak or 0.0) >= float(critical_threshold)
        )
        exciting_seats = {item.acting_seat for item in items if item.decision_id in critical}
        breadth = len(exciting_seats)
        observer_angles = {
            item.observer_seat
            for item in items
            if item.decision_id in critical
            and item.truth_mode == TruthMode.SEAT_EXPERIENCE
            and item.truth_label in usable_labels
            and item.confidence > 0.0
            and (item.peak or 0.0) >= float(critical_threshold) / 2.0
        }
        classification = {
            0: "no_verified_critical_point",
            1: "single_perspective_masterpiece",
            2: "two_seat_duel",
            3: "multi_angle_laffa",
            4: "four_seat_pressure",
        }[breadth]
        thrill_mass = float(sum(peaks[:3]))
        core = {
            "schemaVersion": 2,
            "gameId": ledger.game_id,
            "ledgerHash": ledger.sha256,
            "perspectives": [item.to_dict() for item in items],
            "seatJourneys": [item.to_dict() for item in journeys],
            "teamJourneys": [item.to_dict() for item in team_journeys],
            "criticalDecisionIds": list(critical),
            "perspectiveBreadth": breadth,
            "overallPeak": peaks[0] if peaks else None,
            "secondPeak": peaks[1] if len(peaks) > 1 else None,
            "topKThrillMass": thrill_mass,
            "criticalDensity": len(critical) / len(actor_items) if actor_items else 0.0,
            "multiAngleYield": len(observer_angles),
            "classification": classification,
            "minimumCriticalConfidence": float(minimum_critical_confidence),
            "scenarioGraphSha256": scenario_graph.sha256 if scenario_graph is not None else None,
        }
        return cls(
            ledger.game_id, ledger.sha256, items, journeys, team_journeys,
            critical, breadth, peaks[0] if peaks else None,
            peaks[1] if len(peaks) > 1 else None, thrill_mass,
            len(critical) / len(actor_items) if actor_items else 0.0,
            len(observer_angles), classification,
            float(minimum_critical_confidence),
            scenario_graph.sha256 if scenario_graph is not None else None,
            scenario_graph,
            canonical_sha256(core),
        )

    def _critical_actor_evidence(self) -> tuple[DecisionPerspectiveEvidenceV3, ...]:
        critical = set(self.critical_decision_ids)
        return tuple(
            item
            for item in self.perspectives
            if item.decision_id in critical
            and item.truth_mode == TruthMode.ACTOR_OBSERVER
        )

    @property
    def outcome_importance(self) -> float:
        """Strongest verified outcome swing among the critical decisions."""
        return max(
            (
                float(item.evidence.get("outcomeImpact", 0.0))
                for item in self._critical_actor_evidence()
            ),
            default=0.0,
        )

    @property
    def visual_clarity(self) -> float:
        """Mean presentation clarity of the verified critical decisions."""
        values = tuple(
            float(item.evidence.get("visualClarity", 0.0))
            for item in self._critical_actor_evidence()
        )
        return fmean(values) if values else 0.0

    @property
    def causal_chain_strength(self) -> float:
        """Strongest verified causal relation attached to a critical point.

        The post-reveal verification edge proves identity, not tactical value,
        so it is deliberately excluded. Novelty and corpus coverage are also
        excluded because they are library-relative selection evidence.
        """
        graph = self.scenario_graph
        if graph is None:
            return 0.0
        critical = set(self.critical_decision_ids)
        nodes = {item.node_id: item for item in graph.nodes}
        return max(
            (
                float(edge.strength)
                for edge in graph.edges
                if edge.relation != "post_reveal_verification"
                and (
                    nodes[edge.source].decision_id in critical
                    or nodes[edge.target].decision_id in critical
                )
            ),
            default=0.0,
        )

    @property
    def narrative_arc(self) -> float:
        """Deterministic setup-to-payoff shape derived from the causal graph."""
        if not self.critical_decision_ids:
            return 0.0
        graph = self.scenario_graph
        critical_count = min(3, len(self.critical_decision_ids)) / 3.0
        if graph is None or not graph.nodes:
            return min(100.0, 40.0 + 30.0 * critical_count)
        critical = set(self.critical_decision_ids)
        nodes = {item.node_id: item for item in graph.nodes}
        timeline_ordinals = tuple(item.ordinal for item in graph.nodes)
        decision_ordinals = tuple(
            item.ordinal
            for item in graph.nodes
            if item.decision_id in critical and item.kind == "decision"
        )
        total_span = max(1, max(timeline_ordinals) - min(timeline_ordinals))
        critical_span = (
            (max(decision_ordinals) - min(decision_ordinals)) / total_span
            if len(decision_ordinals) > 1
            else 0.0
        )
        delayed_payoff = any(
            edge.relation != "post_reveal_verification"
            and nodes[edge.target].ordinal > nodes[edge.source].ordinal
            and (
                nodes[edge.source].decision_id in critical
                or nodes[edge.target].decision_id in critical
            )
            for edge in graph.edges
        )
        return min(
            100.0,
            40.0
            + 30.0 * critical_count
            + 20.0 * critical_span
            + (10.0 if delayed_payoff else 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 2,
            "gameId": self.game_id,
            "ledgerHash": self.ledger_hash,
            "perspectives": [item.to_dict() for item in self.perspectives],
            "seatJourneys": [item.to_dict() for item in self.seat_journeys],
            "teamJourneys": [item.to_dict() for item in self.team_journeys],
            "criticalDecisionIds": list(self.critical_decision_ids),
            "perspectiveBreadth": self.perspective_breadth,
            "overallPeak": self.overall_peak,
            "secondPeak": self.second_peak,
            "topKThrillMass": self.top_k_thrill_mass,
            "criticalDensity": self.critical_density,
            "multiAngleYield": self.multi_angle_yield,
            "classification": self.classification,
            "minimumCriticalConfidence": self.minimum_critical_confidence,
            "scenarioGraphSha256": self.scenario_graph_sha256,
            "scenarioGraph": self.scenario_graph.to_dict() if self.scenario_graph is not None else None,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        ledger: DecisionLedgerV1,
    ) -> "LaffaEvidenceV2":
        if int(value.get("schemaVersion", 0)) != 2:
            raise ValueError("unsupported Laffa evidence schema")
        perspectives = tuple(
            DecisionPerspectiveEvidenceV3.from_dict(item)
            for item in value.get("perspectives") or ()
        )
        expected = {item.decision_id for item in ledger.records}
        by_decision: dict[str, list[DecisionPerspectiveEvidenceV3]] = {}
        for item in perspectives:
            by_decision.setdefault(item.decision_id, []).append(item)
        if set(by_decision) != expected:
            raise ValueError("loaded Laffa evidence does not cover the exact decision ledger")
        for record in ledger.records:
            items = by_decision[record.decision_id]
            actors = [item for item in items if item.truth_mode == TruthMode.ACTOR_OBSERVER]
            observers = {
                item.observer_seat for item in items
                if item.truth_mode != TruthMode.ACTUAL_DEAL_ORACLE
            }
            if len(actors) != 1 or actors[0].acting_seat != record.acting_seat:
                raise ValueError("loaded Laffa evidence has invalid actor truth")
            if observers != set(Seat):
                raise ValueError("loaded Laffa evidence lacks four isolated projections")
        seat_journeys = tuple(
            SeatJourneyEvidenceV1(
                _seat(item.get("seat")),
                int(item.get("decisionCount", 0)),
                int(item.get("nonForcedCount", 0)),
                None if item.get("peak") is None else float(item["peak"]),
                None if item.get("topKMean") is None else float(item["topKMean"]),
                str(item.get("status") or ""),
            )
            for item in value.get("seatJourneys") or ()
        )
        team_journeys = tuple(
            TeamJourneyEvidenceV1(
                Team[str(item.get("team") or "").upper()],
                int(item.get("decisionCount", 0)),
                int(item.get("nonForcedCount", 0)),
                None if item.get("peak") is None else float(item["peak"]),
                None if item.get("topKMean") is None else float(item["topKMean"]),
                str(item.get("status") or ""),
            )
            for item in value.get("teamJourneys") or ()
        )
        graph_value = value.get("scenarioGraph")
        graph = ScenarioGraphV1.from_dict(graph_value) if isinstance(graph_value, Mapping) else None
        declared_graph_hash = value.get("scenarioGraphSha256")
        if (graph.sha256 if graph is not None else None) != declared_graph_hash:
            raise ValueError("Laffa evidence scenario graph identity mismatch")
        core = {
            "schemaVersion": 2,
            "gameId": str(value.get("gameId") or ""),
            "ledgerHash": str(value.get("ledgerHash") or ""),
            "perspectives": [item.to_dict() for item in perspectives],
            "seatJourneys": [item.to_dict() for item in seat_journeys],
            "teamJourneys": [item.to_dict() for item in team_journeys],
            "criticalDecisionIds": list(value.get("criticalDecisionIds") or ()),
            "perspectiveBreadth": int(value.get("perspectiveBreadth", 0)),
            "overallPeak": value.get("overallPeak"),
            "secondPeak": value.get("secondPeak"),
            "topKThrillMass": float(value.get("topKThrillMass", 0.0)),
            "criticalDensity": float(value.get("criticalDensity", 0.0)),
            "multiAngleYield": int(value.get("multiAngleYield", 0)),
            "classification": str(value.get("classification") or ""),
            "minimumCriticalConfidence": float(value.get("minimumCriticalConfidence", 0.0)),
            "scenarioGraphSha256": declared_graph_hash,
        }
        if core["gameId"] != ledger.game_id or core["ledgerHash"] != ledger.sha256:
            raise ValueError("Laffa evidence ledger identity mismatch")
        digest = canonical_sha256(core)
        declared_digest = str(value.get("sha256") or "")
        if declared_digest != digest:
            # Early V2 writers serialized an empty thrill mass as integer zero.
            # Accept that byte-level legacy representation without changing the
            # normalized in-memory numeric contract.
            legacy_core = dict(core)
            legacy_core["topKThrillMass"] = value.get("topKThrillMass", 0)
            if declared_digest != canonical_sha256(legacy_core):
                raise ValueError("Laffa evidence checksum mismatch")
            digest = declared_digest
        return cls(
            core["gameId"],
            core["ledgerHash"],
            perspectives,
            seat_journeys,
            team_journeys,
            tuple(str(item) for item in core["criticalDecisionIds"]),
            core["perspectiveBreadth"],
            None if core["overallPeak"] is None else float(core["overallPeak"]),
            None if core["secondPeak"] is None else float(core["secondPeak"]),
            core["topKThrillMass"],
            core["criticalDensity"],
            core["multiAngleYield"],
            core["classification"],
            core["minimumCriticalConfidence"],
            declared_graph_hash,
            graph,
            digest,
        )


@dataclass(frozen=True, slots=True)
class RenderVariantV1:
    laffa_evidence_hash: str
    decision_id: str
    observer_seat: Seat
    rendition_type: str
    prompt_contract: str
    reveal_policy: str
    challenge_locator: str

    def __post_init__(self) -> None:
        if len(self.laffa_evidence_hash) != 64 or any(
            value not in "0123456789abcdef" for value in self.laffa_evidence_hash
        ):
            raise ValueError("render variant requires a lowercase laffa evidence SHA-256")
        if not str(self.decision_id).strip():
            raise ValueError("render variant requires a decision identity")
        supported_renditions = {
            "actor_challenge", "partner_inference", "opponent_prediction",
            "seat_narrative", "post_reveal_analysis",
        }
        observer_prompts = {
            "predict_opponent_play", "read_partner_signal", "identify_threat",
            "plan_next_response", "infer_distribution", "post_reveal_analysis",
        }
        if self.rendition_type not in supported_renditions:
            raise ValueError("render variant rendition type is unsupported")
        actor_only = self.rendition_type == "actor_challenge"
        if actor_only and self.prompt_contract != "choose_acting_action":
            raise ValueError("actor challenges require the acting-action prompt contract")
        if not actor_only and self.prompt_contract not in observer_prompts:
            raise ValueError("non-actor render variants require a seat-experience prompt contract")
        if self.rendition_type == "post_reveal_analysis" and self.prompt_contract != "post_reveal_analysis":
            raise ValueError("post-reveal renditions require the post-reveal prompt contract")
        if not str(self.reveal_policy).strip():
            raise ValueError("render variant requires a reveal policy")
        if not str(self.challenge_locator).strip():
            raise ValueError("render variant requires an exact challenge locator")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "laffaEvidenceHash": self.laffa_evidence_hash,
            "decisionId": self.decision_id,
            "observerSeat": self.observer_seat.name.lower(),
            "renditionType": self.rendition_type,
            "promptContract": self.prompt_contract,
            "revealPolicy": self.reveal_policy,
            "challengeLocator": self.challenge_locator,
        }
