from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from baloot_platform.analysis.evaluator import analyze
from baloot_platform.challenge.miner import ProgressiveChallengeMiner
from baloot_platform.config import build_bundle
from baloot_platform.core.scenario import ScenarioRequest, generate_episode
from baloot_platform.learning.challenge_publication import (
    publication_policy_sha256,
    validate_manual_status_transition,
)
from baloot_platform.learning.store import LearningStore


def _action(
    code: str,
    margin: float,
    best_share: float,
    exact_share: float = 1.0,
):
    return SimpleNamespace(
        card=SimpleNamespace(code=code),
        expected_match_margin=margin,
        best_share=best_share,
        exact_share=exact_share,
    )


def _analysis(
    first: str = "SA",
    first_margin: float = 12.0,
    second_margin: float = 8.0,
    best_share: float = 0.70,
    hidden_deals: int = 96,
):
    second = "H10" if first != "H10" else "SA"
    return SimpleNamespace(
        actions=(
            _action(first, first_margin, best_share),
            _action(second, second_margin, 1.0 - best_share),
        ),
        hidden_deals=hidden_deals,
    )


def _policy() -> dict:
    return {
        "minimum_regret_gap": 1.0,
        "minimum_best_share": 0.5,
        "maximum_expected_margin_drift": 5.0,
        "minimum_hidden_deals": 8,
        "require_same_recommended_action": True,
        "publish_uncertain": False,
    }


def test_challenge_stability_accepts_repeatable_material_decision():
    result = ProgressiveChallengeMiner._stability_metrics(
        _analysis(),
        _analysis(first_margin=13.0, second_margin=8.5),
        _policy(),
    )

    assert result["status"] == "stable"
    assert result["accepted"] is True
    assert result["recommendedActionStable"] is True
    assert result["minimumRegretGap"] == 4.0
    assert result["reasons"] == []


def test_challenge_stability_rejects_changed_or_weak_decision():
    result = ProgressiveChallengeMiner._stability_metrics(
        _analysis(),
        _analysis(
            first="H10",
            first_margin=8.4,
            second_margin=8.0,
            best_share=0.45,
        ),
        _policy(),
    )

    assert result["status"] == "uncertain"
    assert result["accepted"] is False
    assert result["publishUncertain"] is False
    assert set(result["reasons"]) == {
        "recommended_action_changed",
        "regret_gap_below_policy",
        "best_share_below_policy",
    }


def test_active_challenge_profile_is_governed_v2(root):
    profile = build_bundle(root).component("challenge")

    assert profile["profile"] == {
        "id": "progressive_challenge_mining_v2",
        "version": "2.8.4",
        "status": "governed_candidate",
    }
    assert profile["stability"]["publish_uncertain"] is False
    assert profile["confirmation"]["exact_allocation_limit"] == 128
    assert profile["stability"]["exact_allocation_limit"] == 256


def test_generic_status_policy_cannot_bypass_independent_review():
    validate_manual_status_transition("candidate", "review")
    for target in ("approved", "published", "retired"):
        with pytest.raises(ValueError, match="independent-review"):
            validate_manual_status_transition("review", target)
    with pytest.raises(ValueError, match="not allowed"):
        validate_manual_status_transition("review", "candidate")
    with pytest.raises(ValueError, match="independent-review"):
        validate_manual_status_transition("candidate", "retired")


def test_learning_store_fails_closed_on_direct_publication(tmp_path):
    store = LearningStore(tmp_path / "learning.sqlite3")
    store.put_challenge(
        challenge_id="CHL-V21-GATE",
        decision_id=None,
        fingerprint="1" * 64,
        archetype="outcome_flip",
        publishability=0.91,
        payload={"observerSafe": True},
        episode=None,
    )
    assert store.update_challenge_status("CHL-V21-GATE", "review")[
        "status"
    ] == "review"
    with pytest.raises(ValueError, match="independent-review"):
        store.update_challenge_status("CHL-V21-GATE", "published")
    assert store.get_challenge("CHL-V21-GATE")["status"] == "review"


def _publication_policy() -> dict:
    return {
        "minimum_publishability": 0.64,
        "minimum_robustness": 0.35,
        "minimum_archetype_score": 0.50,
        "require_human_review": True,
        "reviewer_quorum": 2,
        "reviewer_roles": ["expert", "referee", "admin", "service"],
        "publisher_roles": ["referee", "admin", "service"],
        "require_governor_independence": True,
    }


def _publishable_payload() -> dict:
    policy = _publication_policy()
    return {
        "observerSafe": True,
        "contentContract": {"contract": "outcome_flip", "accepted": True},
        "gameRecordHash": "8" * 64,
        "decision": {
            "kind": "play",
            "best": {"matchWinConfidence95": [0.60, 0.75]},
        },
        "configurationBundleSha256": "9" * 64,
        "publicationPolicySha256": publication_policy_sha256(policy),
        "stability": {"accepted": True, "status": "stable"},
        "interestingness": {
            "robustnessScore": 0.80,
            "archetypeScores": {"outcome_flip": 0.90},
        },
    }


def test_governed_publication_requires_unique_independent_quorum(tmp_path):
    store = LearningStore(tmp_path / "learning.sqlite3")
    store.put_challenge(
        challenge_id="CHL-V21-PUBLISH", decision_id=None, fingerprint="2" * 64,
        archetype="outcome_flip", publishability=0.91,
        payload=_publishable_payload(), episode=None,
    )
    store.update_challenge_status("CHL-V21-PUBLISH", "review")
    policy = _publication_policy()
    store.record_challenge_review(
        "CHL-V21-PUBLISH", reviewer_id="EXPERT-1", reviewer_role="expert",
        decision="approve", evidence_sha256="a" * 64, comment="independent evidence one",
        policy=policy,
    )
    with pytest.raises(ValueError, match="quorum"):
        store.govern_challenge_status(
            "CHL-V21-PUBLISH", target="approved", actor_id="REFEREE-1",
            actor_role="referee", evidence_sha256="b" * 64, policy=policy,
        )
    with pytest.raises(ValueError, match="already submitted"):
        store.record_challenge_review(
            "CHL-V21-PUBLISH", reviewer_id="EXPERT-1", reviewer_role="expert",
            decision="approve", evidence_sha256="c" * 64, comment="duplicate",
            policy=policy,
        )
    store.record_challenge_review(
        "CHL-V21-PUBLISH", reviewer_id="EXPERT-2", reviewer_role="expert",
        decision="approve", evidence_sha256="d" * 64, comment="independent evidence two",
        policy=policy,
    )
    with pytest.raises(ValueError, match="independent from its reviewers"):
        store.govern_challenge_status(
            "CHL-V21-PUBLISH", target="approved", actor_id="EXPERT-1",
            actor_role="referee", evidence_sha256="e" * 64, policy=policy,
        )
    assert store.govern_challenge_status(
        "CHL-V21-PUBLISH", target="approved", actor_id="REFEREE-1",
        actor_role="referee", evidence_sha256="e" * 64, policy=policy,
    )["status"] == "approved"
    assert store.govern_challenge_status(
        "CHL-V21-PUBLISH", target="published", actor_id="REFEREE-1",
        actor_role="referee", evidence_sha256="f" * 64, policy=policy,
    )["status"] == "published"
    assert store.govern_challenge_status(
        "CHL-V21-PUBLISH", target="retired", actor_id="REFEREE-1",
        actor_role="referee", evidence_sha256="1" * 64, policy=policy,
    )["status"] == "retired"
    events = store.list_challenge_publication_events("CHL-V21-PUBLISH")
    assert [event["toStatus"] for event in events] == ["approved", "published", "retired"]


def test_governed_publication_rejects_unsafe_or_conflicted_evidence(tmp_path):
    store = LearningStore(tmp_path / "learning.sqlite3")
    policy = _publication_policy()
    store.put_challenge(
        challenge_id="CHL-V21-UNSAFE", decision_id=None, fingerprint="3" * 64,
        archetype="outcome_flip", publishability=0.91,
        payload={**_publishable_payload(), "observerSafe": False}, episode=None,
    )
    store.update_challenge_status("CHL-V21-UNSAFE", "review")
    for reviewer, decision in (("EXPERT-1", "approve"), ("EXPERT-2", "reject")):
        store.record_challenge_review(
            "CHL-V21-UNSAFE", reviewer_id=reviewer, reviewer_role="expert",
            decision=decision, evidence_sha256=("a" if decision == "approve" else "b") * 64,
            comment="evidence", policy=policy,
        )
    with pytest.raises(ValueError, match="observer-safety"):
        store.govern_challenge_status(
            "CHL-V21-UNSAFE", target="approved", actor_id="REFEREE-1",
            actor_role="referee", evidence_sha256="c" * 64, policy=policy,
        )


def test_governed_publication_rejects_policy_drift(tmp_path):
    store = LearningStore(tmp_path / "learning.sqlite3")
    policy = _publication_policy()
    store.put_challenge(
        challenge_id="CHL-V21-DRIFT", decision_id=None, fingerprint="5" * 64,
        archetype="outcome_flip", publishability=0.91,
        payload=_publishable_payload(), episode=None,
    )
    store.update_challenge_status("CHL-V21-DRIFT", "review")
    for reviewer, digest in (("EXPERT-1", "a" * 64), ("EXPERT-2", "b" * 64)):
        store.record_challenge_review(
            "CHL-V21-DRIFT", reviewer_id=reviewer, reviewer_role="expert",
            decision="approve", evidence_sha256=digest, comment="evidence", policy=policy,
        )
    changed = {**policy, "minimum_publishability": 0.65}
    with pytest.raises(ValueError, match="policy identity"):
        store.govern_challenge_status(
            "CHL-V21-DRIFT", target="approved", actor_id="REFEREE-1",
            actor_role="referee", evidence_sha256="c" * 64, policy=changed,
        )


def test_publication_migration_is_packaged_byte_identically(root):
    source = root / "migrations" / "014_v21_challenge_publication.sql"
    packaged = root / "src" / "baloot_platform" / "resources" / "migrations" / source.name
    assert source.read_bytes() == packaged.read_bytes()
    text = source.read_text(encoding="utf-8")
    assert "UNIQUE(challenge_id, reviewer_id)" in text
    assert "to_status IN ('approved','published','retired')" in text


def test_rest_publication_uses_authenticated_identity_not_request_reviewer(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("BALOOT_API_KEY", "v21-publication-key")
    # This route test does not construct media. Exact CI separately enforces
    # the real RAQM preflight and Arabic render gates.
    import baloot_platform.video.cards as video_cards
    monkeypatch.setattr(video_cards, "assert_arabic_ready", lambda: None)
    from baloot_platform.service import api

    store = LearningStore(tmp_path / "api-learning.sqlite3")
    monkeypatch.setattr(api, "LEARNING_STORE", store)
    store.put_challenge(
        challenge_id="CHL-V21-API", decision_id=None, fingerprint="4" * 64,
        archetype="outcome_flip", publishability=0.91,
        payload=_publishable_payload(), episode=None,
    )
    client = TestClient(api.app)
    base = {"x-api-key": "v21-publication-key"}
    # This is the original default SHA-256 subject for the fixture key.
    first = {**base, "x-baloot-subject-id": "SVC-22AE3A94AFA700B03C94"}
    second = {**base, "x-baloot-subject-id": "REVIEWER-B"}
    governor = {**base, "x-baloot-subject-id": "GOVERNOR-A"}
    assert client.post(
        "/api/v1/challenges/CHL-V21-API:status",
        headers=governor, json={"status": "review"},
    ).status_code == 200
    for headers, digest in ((first, "a" * 64), (second, "b" * 64)):
        response = client.post(
            "/api/v1/challenges/CHL-V21-API/publication-reviews",
            headers=headers,
            json={"decision": "approve", "evidenceSha256": digest, "comment": "review"},
        )
        assert response.status_code == 201
        assert response.json()["reviewerId"] == headers["x-baloot-subject-id"]
    duplicate = client.post(
        "/api/v1/challenges/CHL-V21-API/publication-reviews",
        headers=base,
        json={"decision": "approve", "evidenceSha256": "e" * 64, "comment": "duplicate review"},
    )
    assert duplicate.status_code == 409
    assert "already submitted" in duplicate.json()["detail"]
    assert client.post(
        "/api/v1/challenges/CHL-V21-API:approve", headers=governor,
        json={"evidenceSha256": "c" * 64},
    ).json()["status"] == "approved"
    assert client.post(
        "/api/v1/challenges/CHL-V21-API:publish", headers=governor,
        json={"evidenceSha256": "d" * 64},
    ).json()["status"] == "published"


def test_analysis_and_scenario_honor_immediate_cancellation(root):
    bundle = build_bundle(root)

    with pytest.raises(InterruptedError, match="analysis cancelled"):
        analyze(
            None,
            bundle.rules,
            cancel_check=lambda: True,
        )
    with pytest.raises(
        InterruptedError,
        match="scenario generation cancelled",
    ):
        generate_episode(
            bundle.rules,
            bundle.convention,
            bundle_hash=bundle.sha256,
            request=ScenarioRequest.governed(),
            cancel_check=lambda: True,
        )
