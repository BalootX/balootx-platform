from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from baloot_platform.config import build_bundle, load_profile_payload
from baloot_platform.learning.challenge_publication import (
    publication_policy_sha256,
    validate_governed_transition,
)
from baloot_platform.learning.store import LearningStore
from baloot_platform.sakka.store import _leaderboard_key
from baloot_platform.service.api import _accepted_actions_for_challenge


ROOT = Path(__file__).resolve().parents[1]


def _policy() -> dict:
    return {
        "minimum_publishability": 0.64,
        "minimum_robustness": 0.35,
        "minimum_archetype_score": 0.50,
        "reviewer_quorum": 2,
        "reviewer_roles": ["expert"],
        "publisher_roles": ["referee"],
        "require_governor_independence": True,
    }


def _challenge_payload() -> dict:
    policy = _policy()
    return {
        "observerSafe": True,
        "contentContract": {"contract": "outcome_flip", "accepted": True},
        "gameRecordHash": "a" * 64,
        "configurationBundleSha256": "b" * 64,
        "publicationPolicySha256": publication_policy_sha256(policy),
        "stability": {"accepted": True, "status": "stable"},
        "interestingness": {
            "robustnessScore": 0.8,
            "archetypeScores": {"outcome_flip": 0.9},
        },
        "difficulty": 71,
        "decision": {
            "kind": "play",
            "recommendedCard": {"code": "SA"},
            "best": {"matchWinConfidence95": [0.58, 0.74]},
        },
    }


def test_challenge_store_projects_payload_without_losing_audit_copy(tmp_path: Path):
    store = LearningStore(tmp_path / "learning.sqlite3")
    payload = _challenge_payload()
    stored = store.put_challenge(
        challenge_id="CHL-PROJECTION",
        decision_id=None,
        fingerprint="c" * 64,
        archetype="outcome_flip",
        publishability=0.9,
        payload=payload,
        episode=None,
    )
    assert stored["difficulty"] == 71
    assert stored["decision"]["recommendedCard"] == {"code": "SA"}
    assert stored["payload"] == payload
    assert store.list_challenges()[0]["interestingness"]["robustnessScore"] == 0.8


def test_simulated_challenge_has_a_nonempty_accepted_action():
    challenge = {"decisionId": None, "payload": _challenge_payload()}
    accepted, difficulty = _accepted_actions_for_challenge(challenge)
    assert accepted == [{"code": "SA"}]
    assert difficulty == 71


def test_publication_truth_contract_is_final_fail_closed_gate():
    payload = _challenge_payload()
    challenge = {
        "status": "review",
        "actorId": "AUTHOR",
        "publishability": 0.9,
        "archetype": "outcome_flip",
        "payload": payload,
    }
    reviews = [
        {"reviewerId": "A", "decision": "approve"},
        {"reviewerId": "B", "decision": "approve"},
    ]
    validate_governed_transition(
        challenge, reviews, target="approved", actor_role="referee", policy=_policy()
    )
    for missing in ("contentContract", "gameRecordHash", "decision"):
        broken = {**challenge, "payload": {**payload}}
        broken["payload"].pop(missing)
        with pytest.raises(ValueError):
            validate_governed_transition(
                broken, reviews, target="approved", actor_role="referee", policy=_policy()
            )


def test_leaderboard_key_is_observer_relative():
    team_one = {
        "result": {
            "matchWon": False,
            "matchScore": [120, 140],
            "observerTeam": 1,
            "totalRoundMargin": 1,
        }
    }
    team_zero = {
        "result": {
            "matchWon": False,
            "matchScore": [130, 110],
            "observerTeam": 0,
            "totalRoundMargin": 1,
        }
    }
    assert _leaderboard_key(team_one)[1] == 140
    assert _leaderboard_key(team_zero)[1] == 130
    assert _leaderboard_key(team_one) > _leaderboard_key(team_zero)


def test_rules_and_challenge_profiles_reject_unknown_or_invalid_values(tmp_path: Path):
    rules = yaml.safe_load(
        (ROOT / "configs/rules/saudi_federation_2023.yaml").read_text(
            encoding="utf-8"
        )
    )
    rules["unknown_switch"] = True
    bad_rules = tmp_path / "rules.yaml"
    bad_rules.write_text(yaml.safe_dump(rules, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown_switch"):
        load_profile_payload("rules", bad_rules, root=ROOT)

    challenge = yaml.safe_load((ROOT / "configs/challenge/progressive_mining_v2.yaml").read_text(encoding="utf-8"))
    challenge["publication"]["minimum_robustness"] = 9
    bad_challenge = tmp_path / "challenge.yaml"
    bad_challenge.write_text(yaml.safe_dump(challenge, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="minimum_robustness"):
        load_profile_payload("challenge", bad_challenge, root=ROOT)


def test_committed_profile_manifest_matches_current_bundle():
    manifest = json.loads((ROOT / "generated/profile-manifest.json").read_text(encoding="utf-8"))
    for mode in ("full", "sun_only", "hokum_only"):
        bundle = build_bundle(ROOT, mode=mode).to_dict()
        assert manifest["bundles"][mode] == bundle
        assert all(
            "\\" not in item["path"]
            for item in bundle["components"].values()
            if "path" in item
        )
