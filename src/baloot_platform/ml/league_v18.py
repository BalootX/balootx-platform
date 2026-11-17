from __future__ import annotations

"""Deterministic self-play league evidence for the V18 release.

The V17 league remains available for compatibility.  V18 uses content-addressed
identities and SOURCE_DATE_EPOCH-derived timestamps so self-play evidence is
stable across clean-room verification paths and repeated release builds.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


def stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def release_time() -> str:
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "315532800"))
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def _id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}-{stable_hash(payload)[:20].upper()}"


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
    def create(
        cls,
        *,
        model_id: str,
        version: str,
        artifact_hash: str,
        parent_agent_id: str | None = None,
        role: str = "challenger",
        metadata: Mapping[str, Any] | None = None,
    ) -> "LeagueAgent":
        identity = {
            "modelId": str(model_id),
            "version": str(version),
            "artifactHash": str(artifact_hash),
            "parentAgentId": parent_agent_id,
            "role": str(role),
            "metadata": dict(metadata or {}),
        }
        return cls(
            agent_id=_id("AGT", identity),
            model_id=str(model_id),
            version=str(version),
            artifact_hash=str(artifact_hash),
            created_at=release_time(),
            parent_agent_id=parent_agent_id,
            role=str(role),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agentId": self.agent_id,
            "modelId": self.model_id,
            "version": self.version,
            "artifactHash": self.artifact_hash,
            "createdAt": self.created_at,
            "parentAgentId": self.parent_agent_id,
            "role": self.role,
            "metadata": dict(self.metadata),
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
    def create(
        cls,
        *,
        board_set_hash: str,
        agent_a: str,
        agent_b: str,
        score_a: float,
        score_b: float,
        decisions: int,
        illegal_actions_a: int = 0,
        illegal_actions_b: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> "LeagueMatch":
        identity = {
            "boardSetHash": str(board_set_hash),
            "agentA": str(agent_a),
            "agentB": str(agent_b),
            "scoreA": float(score_a),
            "scoreB": float(score_b),
            "decisions": max(0, int(decisions)),
            "illegalActionsA": max(0, int(illegal_actions_a)),
            "illegalActionsB": max(0, int(illegal_actions_b)),
            "metadata": dict(metadata or {}),
        }
        return cls(
            match_id=_id("LGM", identity),
            board_set_hash=str(board_set_hash),
            agent_a=str(agent_a),
            agent_b=str(agent_b),
            score_a=float(score_a),
            score_b=float(score_b),
            decisions=max(0, int(decisions)),
            illegal_actions_a=max(0, int(illegal_actions_a)),
            illegal_actions_b=max(0, int(illegal_actions_b)),
            created_at=release_time(),
            metadata=dict(metadata or {}),
        )

    def result_a(self) -> float:
        if self.score_a > self.score_b:
            return 1.0
        if self.score_a < self.score_b:
            return 0.0
        return 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "matchId": self.match_id,
            "boardSetHash": self.board_set_hash,
            "agentA": self.agent_a,
            "agentB": self.agent_b,
            "scoreA": self.score_a,
            "scoreB": self.score_b,
            "decisions": self.decisions,
            "illegalActionsA": self.illegal_actions_a,
            "illegalActionsB": self.illegal_actions_b,
            "createdAt": self.created_at,
            "metadata": dict(self.metadata),
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
            result_a = 0.0 if match.illegal_actions_a else 1.0
        else:
            result_a = match.result_a()
        rating_a = self.ratings[match.agent_a]
        rating_b = self.ratings[match.agent_b]
        expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
        delta = self.k_factor * (result_a - expected_a)
        self.ratings[match.agent_a] = rating_a + delta
        self.ratings[match.agent_b] = rating_b - delta
        self.matches.append(match)

    def leaderboard(self) -> list[dict[str, Any]]:
        games = {agent_id: 0 for agent_id in self.agents}
        illegal = {agent_id: 0 for agent_id in self.agents}
        for match in self.matches:
            games[match.agent_a] += 1
            games[match.agent_b] += 1
            illegal[match.agent_a] += match.illegal_actions_a
            illegal[match.agent_b] += match.illegal_actions_b
        rows = [
            {
                "rank": 0,
                "agent": self.agents[agent_id].to_dict(),
                "rating": rating,
                "matches": games[agent_id],
                "illegalActions": illegal[agent_id],
            }
            for agent_id, rating in self.ratings.items()
        ]
        rows.sort(key=lambda item: (item["illegalActions"] == 0, item["rating"], item["agent"]["agentId"]), reverse=True)
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        return rows

    def cross_play_matrix(self) -> dict[str, dict[str, dict[str, float]]]:
        result: dict[str, dict[str, dict[str, float]]] = {agent_id: {} for agent_id in self.agents}
        aggregates: dict[tuple[str, str], list[float]] = {}
        for match in self.matches:
            aggregates.setdefault((match.agent_a, match.agent_b), []).append(match.result_a())
            aggregates.setdefault((match.agent_b, match.agent_a), []).append(1.0 - match.result_a())
        for (agent_a, agent_b), values in sorted(aggregates.items()):
            result[agent_a][agent_b] = {"games": float(len(values)), "scoreRate": sum(values) / len(values)}
        return result

    def snapshot(self) -> dict[str, Any]:
        value = {
            "schemaVersion": 2,
            "version": "18.0.0",
            "generatedAt": release_time(),
            "agents": [self.agents[key].to_dict() for key in sorted(self.agents)],
            "ratings": {key: self.ratings[key] for key in sorted(self.ratings)},
            "matches": [match.to_dict() for match in self.matches],
            "leaderboard": self.leaderboard(),
            "crossPlay": self.cross_play_matrix(),
            "automaticProductionPromotion": False,
        }
        value["snapshotHash"] = stable_hash(value)
        return value
