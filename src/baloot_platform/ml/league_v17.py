from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class LeagueAgent:
    agent_id: str
    model_id: str
    version: str
    artifact_hash: str
    created_at: str
    parent_agent_id: str | None = None
    role: str = "challenger"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, *, model_id: str, version: str, artifact_hash: str,
               parent_agent_id: str | None = None, role: str = "challenger",
               metadata: Mapping[str, Any] | None = None) -> "LeagueAgent":
        return cls(
            agent_id=f"AGT-{uuid.uuid4().hex[:20].upper()}", model_id=model_id, version=version,
            artifact_hash=artifact_hash, created_at=utc_now(), parent_agent_id=parent_agent_id,
            role=role, metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agentId": self.agent_id, "modelId": self.model_id, "version": self.version,
            "artifactHash": self.artifact_hash, "createdAt": self.created_at,
            "parentAgentId": self.parent_agent_id, "role": self.role, "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class LeagueMatch:
    match_id: str
    board_set_hash: str
    agent_a: str
    agent_b: str
    score_a: float
    score_b: float
    decisions: int
    illegal_actions_a: int
    illegal_actions_b: int
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, *, board_set_hash: str, agent_a: str, agent_b: str,
               score_a: float, score_b: float, decisions: int,
               illegal_actions_a: int = 0, illegal_actions_b: int = 0,
               metadata: Mapping[str, Any] | None = None) -> "LeagueMatch":
        return cls(
            match_id=f"LGM-{uuid.uuid4().hex[:20].upper()}", board_set_hash=board_set_hash,
            agent_a=agent_a, agent_b=agent_b, score_a=float(score_a), score_b=float(score_b),
            decisions=max(0, int(decisions)), illegal_actions_a=max(0, int(illegal_actions_a)),
            illegal_actions_b=max(0, int(illegal_actions_b)), created_at=utc_now(), metadata=dict(metadata or {}),
        )

    def result_a(self) -> float:
        if self.score_a > self.score_b:
            return 1.0
        if self.score_a < self.score_b:
            return 0.0
        return 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "matchId": self.match_id, "boardSetHash": self.board_set_hash,
            "agentA": self.agent_a, "agentB": self.agent_b,
            "scoreA": self.score_a, "scoreB": self.score_b, "decisions": self.decisions,
            "illegalActionsA": self.illegal_actions_a, "illegalActionsB": self.illegal_actions_b,
            "createdAt": self.created_at, "metadata": dict(self.metadata),
        }


class SelfPlayLeague:
    def __init__(self, *, initial_rating: float = 1500.0, k_factor: float = 24.0):
        self.initial_rating = float(initial_rating)
        self.k_factor = float(k_factor)
        self.agents: dict[str, LeagueAgent] = {}
        self.ratings: dict[str, float] = {}
        self.matches: list[LeagueMatch] = []

    def register(self, agent: LeagueAgent) -> None:
        if agent.agent_id in self.agents:
            raise ValueError("agent already registered")
        self.agents[agent.agent_id] = agent
        self.ratings[agent.agent_id] = self.initial_rating

    def record(self, match: LeagueMatch) -> None:
        if match.agent_a not in self.agents or match.agent_b not in self.agents:
            raise KeyError("both agents must be registered")
        if match.illegal_actions_a or match.illegal_actions_b:
            # Illegal actions are a hard failure. Keep the match for audit but
            # do not let it improve an agent rating.
            result_a = 0.0 if match.illegal_actions_a else 1.0
        else:
            result_a = match.result_a()
        ra = self.ratings[match.agent_a]
        rb = self.ratings[match.agent_b]
        expected_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        delta = self.k_factor * (result_a - expected_a)
        self.ratings[match.agent_a] = ra + delta
        self.ratings[match.agent_b] = rb - delta
        self.matches.append(match)

    def leaderboard(self) -> list[dict[str, Any]]:
        games = {agent_id: 0 for agent_id in self.agents}
        illegal = {agent_id: 0 for agent_id in self.agents}
        for match in self.matches:
            games[match.agent_a] += 1; games[match.agent_b] += 1
            illegal[match.agent_a] += match.illegal_actions_a
            illegal[match.agent_b] += match.illegal_actions_b
        rows = [
            {
                "rank": 0, "agent": self.agents[agent_id].to_dict(), "rating": rating,
                "matches": games[agent_id], "illegalActions": illegal[agent_id],
            }
            for agent_id, rating in self.ratings.items()
        ]
        rows.sort(key=lambda item: (item["illegalActions"] == 0, item["rating"]), reverse=True)
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        return rows

    def cross_play_matrix(self) -> dict[str, dict[str, dict[str, float]]]:
        result: dict[str, dict[str, dict[str, float]]] = {agent_id: {} for agent_id in self.agents}
        aggregates: dict[tuple[str, str], list[float]] = {}
        for match in self.matches:
            aggregates.setdefault((match.agent_a, match.agent_b), []).append(match.result_a())
            aggregates.setdefault((match.agent_b, match.agent_a), []).append(1.0 - match.result_a())
        for (a, b), values in aggregates.items():
            result[a][b] = {"games": float(len(values)), "scoreRate": sum(values) / len(values)}
        return result

    def snapshot(self) -> dict[str, Any]:
        value = {
            "schemaVersion": 1,
            "agents": [agent.to_dict() for agent in self.agents.values()],
            "ratings": dict(self.ratings),
            "matches": [match.to_dict() for match in self.matches],
            "leaderboard": self.leaderboard(),
            "crossPlay": self.cross_play_matrix(),
        }
        value["snapshotHash"] = stable_hash(value)
        return value


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    model_id: str
    evaluation_id: str
    legal_violation_rate: float
    relative_score_rate: float
    lower_confidence_bound: float
    calibration_error: float
    p95_latency_ms: float
    ood_detection_recall: float
    board_sets: int
    decisions: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, *, model_id: str, legal_violation_rate: float,
               relative_score_rate: float, lower_confidence_bound: float,
               calibration_error: float, p95_latency_ms: float,
               ood_detection_recall: float, board_sets: int, decisions: int,
               metadata: Mapping[str, Any] | None = None) -> "ModelEvaluation":
        return cls(
            model_id=model_id, evaluation_id=f"EVAL-{uuid.uuid4().hex[:20].upper()}",
            legal_violation_rate=float(legal_violation_rate), relative_score_rate=float(relative_score_rate),
            lower_confidence_bound=float(lower_confidence_bound), calibration_error=float(calibration_error),
            p95_latency_ms=float(p95_latency_ms), ood_detection_recall=float(ood_detection_recall),
            board_sets=max(0, int(board_sets)), decisions=max(0, int(decisions)), metadata=dict(metadata or {}),
        )


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    approved: bool
    reasons: tuple[str, ...]
    evaluation_id: str
    policy_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {"approved": self.approved, "reasons": list(self.reasons), "evaluationId": self.evaluation_id, "policyHash": self.policy_hash}


class ModelPromotionPolicy:
    def __init__(self, *, minimum_score_rate: float = 0.52, minimum_lower_bound: float = 0.50,
                 maximum_calibration_error: float = 0.08, maximum_p95_latency_ms: float = 50.0,
                 minimum_ood_recall: float = 0.75, minimum_board_sets: int = 20,
                 minimum_decisions: int = 10_000):
        self.settings = {
            "minimumScoreRate": float(minimum_score_rate),
            "minimumLowerBound": float(minimum_lower_bound),
            "maximumCalibrationError": float(maximum_calibration_error),
            "maximumP95LatencyMs": float(maximum_p95_latency_ms),
            "minimumOodRecall": float(minimum_ood_recall),
            "minimumBoardSets": int(minimum_board_sets),
            "minimumDecisions": int(minimum_decisions),
            "maximumLegalViolationRate": 0.0,
        }
        self.sha256 = stable_hash(self.settings)

    def decide(self, evaluation: ModelEvaluation) -> PromotionDecision:
        reasons: list[str] = []
        if evaluation.legal_violation_rate > 0:
            reasons.append("illegal_action_rate_must_be_zero")
        if evaluation.relative_score_rate < self.settings["minimumScoreRate"]:
            reasons.append("score_rate_below_threshold")
        if evaluation.lower_confidence_bound < self.settings["minimumLowerBound"]:
            reasons.append("confidence_bound_below_threshold")
        if evaluation.calibration_error > self.settings["maximumCalibrationError"]:
            reasons.append("calibration_error_above_threshold")
        if evaluation.p95_latency_ms > self.settings["maximumP95LatencyMs"]:
            reasons.append("latency_above_threshold")
        if evaluation.ood_detection_recall < self.settings["minimumOodRecall"]:
            reasons.append("ood_recall_below_threshold")
        if evaluation.board_sets < self.settings["minimumBoardSets"]:
            reasons.append("insufficient_board_sets")
        if evaluation.decisions < self.settings["minimumDecisions"]:
            reasons.append("insufficient_decisions")
        return PromotionDecision(not reasons, tuple(reasons), evaluation.evaluation_id, self.sha256)
