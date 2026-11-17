from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from baloot_platform.hitl import ReviewGovernanceServiceV18, SQLiteReviewGovernanceStore


class DummyLearningStore:
    def __init__(self) -> None:
        self.decision = {
            "decisionId": "DEC-HITL-1",
            "gameId": "GAME-HITL-1",
            "gameHash": "g" * 64,
            "decisionKind": "play",
            "actor": "south",
            "phase": "playing",
            "publicState": {"trickIndex": 3, "matchScore": [70, 68]},
            "selectedAction": "C7",
            "legalActions": ["C7", "C8"],
            "createdAt": "2026-07-29T00:00:00+00:00",
            "analyses": [
                {
                    "bundleHash": "b" * 64,
                    "result": {"recommendedCard": "C7"},
                }
            ],
            "audits": [
                {
                    "bestAction": "C8",
                    "alternatives": [
                        {"action": "C7", "regret": 4.0, "winProbabilityDrop": 0.08},
                        {"action": "C8", "regret": 0.0, "winProbabilityDrop": 0.0},
                    ],
                }
            ],
        }

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        return copy.deepcopy(self.decision) if decision_id == self.decision["decisionId"] else None


def hitl_profile(root: Path, *, easy_consensus: bool = True) -> dict[str, Any]:
    profile = yaml.safe_load((root / "configs/hitl/governance_v18.yaml").read_text(encoding="utf-8"))
    if easy_consensus:
        profile["review"].update(
            {
                "minimum_independent_reviewers": 2,
                "minimum_weighted_support": 0.5,
                "minimum_wilson_lower": 0.0,
                "minimum_independent_clusters": 2,
                "require_verified_expert_or_two_trusted": False,
            }
        )
    return profile


def make_service(root: Path, tmp_path: Path, *, easy_consensus: bool = True):
    store = SQLiteReviewGovernanceStore(tmp_path / "hitl.sqlite3")
    learning = DummyLearningStore()
    service = ReviewGovernanceServiceV18(
        store,
        learning,
        None,
        hitl_profile(root, easy_consensus=easy_consensus),
        cluster_secret=b"hitl-test-cluster-secret-32-bytes!",
    )
    return service, store, learning


def submit_review(
    service: ReviewGovernanceServiceV18,
    case_id: str,
    reviewer_id: str,
    *,
    rationale: str,
    preferred_action: str = "C8",
    hidden_seen: bool = False,
    result_seen: bool = False,
    disclosure: str = "before_reveal",
    device: str | None = None,
    network: str | None = None,
):
    assignment = service.assign(
        case_id,
        reviewer_id=reviewer_id,
        device_id=device or f"device-{reviewer_id}",
        network_id=network or f"network-{reviewer_id}",
    )
    return service.submit(
        case_id,
        assignment_id=assignment["assignmentId"],
        assignment_token=assignment["assignmentToken"],
        reviewer_id=reviewer_id,
        preferred_action=preferred_action,
        confidence=0.92,
        reason_codes=["card_counting", "partner_signal"],
        rationale_ar=rationale,
        claims=[{"type": "strategic_goal", "goal": "preserve_entry"}],
        evidence=[{"type": "public_event", "sequence": 12}],
        view_mode="observer",
        hidden_cards_seen=hidden_seen,
        final_result_seen=result_seen,
        disclosure_timing=disclosure,
    )
