from __future__ import annotations

import copy
import hashlib
import json

import pytest

from baloot_platform.challenge.locator import (
    ChallengeAttemptConflictError,
    ChallengeLocatorServiceV1,
    SQLiteChallengeLocatorStore,
)
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.config import build_bundle
from baloot_platform.research.result_authority import RuntimeCompatibilityV1


def _service(root, tmp_path, clock, *, presentation=None):
    bundle = build_bundle(root)
    video = bundle.component("video")
    frozen_presentation = presentation or {
        "schemaVersion": 1,
        "bundleRevisionId": "CBR-TEST-FROZEN",
        "locale": video["render"]["locale"],
        "direction": video["render"]["direction"],
        "profileId": video["profile"]["id"],
        "profileVersion": video["profile"]["version"],
        "profileSha256": bundle.components["video"].sha256,
        "challengeCopy": copy.deepcopy(video["challenge_copy"]),
    }
    return ChallengeLocatorServiceV1(
        SQLiteChallengeLocatorStore(tmp_path / "locators.sqlite3"),
        rules=bundle.rules,
        secret=b"locator-test-secret-which-is-at-least-32-bytes",
        public_base_url="https://play.balootx.test",
        challenge_copy=bundle.component("video")["challenge_copy"],
        research_presentation_resolver=lambda _code: copy.deepcopy(
            frozen_presentation
        ),
        clock=lambda: clock[0],
    )


def _research_public_exact_state(code: str) -> dict:
    legal = [
        {
            "code": "C7",
            "action": "play_card",
            "card": {
                "id": 0,
                "code": "C7",
                "suit": "clubs",
                "rank": "seven",
                "symbol": "7♣",
                "label": "سبعة شيريا",
            },
        },
        {
            "code": "CA",
            "action": "play_card",
            "card": {
                "id": 7,
                "code": "CA",
                "suit": "clubs",
                "rank": "ace",
                "symbol": "A♣",
                "label": "آص شيريا",
            },
        },
    ]
    return {
        "schemaVersion": 1,
        "decisionReference": {"resultCode": code, "decisionOrdinal": 1},
        "publicProjectionHash": "a" * 64,
        "phase": "play",
        "decisionKind": "play",
        "observerSeat": "south",
        "nextActor": "south",
        "legalActions": legal,
        "publicState": {
            "phase": "play",
            "currentPlayer": "south",
            "hands": {
                "north": {"count": 2},
                "west": {"count": 2},
                "south": [item["card"] for item in legal],
                "east": {"count": 2},
            },
            "completedTricks": [
                {
                    "plays": [
                        {"seat": "north", "card": legal[0]["card"]},
                        {"seat": "west", "card": legal[1]["card"]},
                        {"seat": "south", "card": legal[0]["card"]},
                        {"seat": "east", "card": legal[1]["card"]},
                    ]
                }
            ],
            "currentTrick": {
                "plays": [{"seat": "north", "card": legal[0]["card"]}]
            },
        },
    }


def _research_result(code: str, digest: str) -> dict:
    exact_state_id = "STATE-1234567890ABCDEF1234"
    compatibility = RuntimeCompatibilityV1.build(
        status="passed",
        engine_version="4.0.0",
        evidence_hash="b" * 64,
        failure_codes=(),
        bundle_hash="c" * 64,
        trace_hash="d" * 64,
    ).to_dict()
    return {
        "schemaVersion": 2,
        "resultCode": code,
        "resultHash": digest,
        "classification": "scenario_v4",
        "disposition": "qualified",
        "truthLabel": "exact_optimal",
        "confidence": 1.0,
        "qualification": {
            "qualified": True,
            "observerSafe": True,
            "decisionPointAuditComplete": True,
            "failures": [],
        },
        "proofOutcome": {
            "achievedMode": "exact",
            "completionStatus": "exact_complete",
            "claimSatisfied": True,
            "selectedAction": "CA",
            "rootActions": [
                {"action": "CA", "value": 1, "satisfies": True},
                {"action": "C7", "value": -1, "satisfies": False},
            ],
        },
        "scenarioProjection": {
            "decisionKind": "play",
            "exactStateId": exact_state_id,
            "exactStateAvailable": True,
        },
        "interestingSeats": ["south"],
        "runtimeCompatibility": compatibility,
        "provenance": {"resultIdentity": {"exactStateId": exact_state_id}},
    }


def test_research_result_locator_is_stable_persisted_and_playable(root, tmp_path):
    service = _service(root, tmp_path, [1_000])
    digest = "a1b2c3d4e5" + "0" * 54
    code = "LAF-A1B2C3D4E5"
    exact = _research_public_exact_state(code)
    proof = _research_result(code, digest)["proofOutcome"]

    first = service.publish_research_result(
        result_code=code,
        result_hash=digest,
        exact_state_id="STATE-1234567890ABCDEF1234",
        public_exact_state=exact,
        proof_outcome=proof,
        truth_label="exact_optimal",
        confidence=1.0,
    )
    second = service.publish_research_result(
        result_code=code,
        result_hash=digest,
        exact_state_id="STATE-1234567890ABCDEF1234",
        public_exact_state=exact,
        proof_outcome=proof,
        truth_label="exact_optimal",
        confidence=1.0,
    )

    assert first == second
    stored = service.store.get(code)
    assert stored is not None
    assert stored["binding"]["exactStateId"] == "STATE-1234567890ABCDEF1234"
    public = service.resolve(code.lower())
    encoded = json.dumps(public, sort_keys=True)
    assert public["code"] == code
    assert public["decisionKind"] == "play"
    assert public["observerSeat"] == public["actingSeat"] == "south"
    assert public["legalActions"] == [
        item["card"] for item in exact["legalActions"]
    ]
    assert public["state"]["observerHand"] == [
        item["card"] for item in exact["legalActions"]
    ]
    assert public["state"]["handSizes"] == [2, 2, 2, 2]
    assert len(public["state"]["plays"]) == 5
    assert public["state"]["plays"][-1]["seat"] == "north"
    assert public["governedAnswerAvailable"] is True
    assert "exactStateId" not in encoded
    assert "resultHash" not in encoded
    attempt = service.commit_attempt(
        code,
        participant_id="research-player",
        chosen_action=public["legalActions"][1],
    )
    assert attempt["reveal"]["status"] == "governed_answer"
    assert attempt["reveal"]["chosenAccepted"] is True
    assert attempt["reveal"]["answerMode"] == "single_supported"
    assert attempt["reveal"]["recommendedAction"]["code"] == "CA"
    assert attempt["reveal"]["actionEvidence"][0]["action"]["code"] == "CA"
    assert attempt["reveal"]["engineCognitiveDifficulty"] is None
    assert attempt["reveal"]["proofBoundary"] == {
        "achievedMode": "exact",
        "completionStatus": "exact_complete",
        "horizon": None,
        "boundaryEvaluator": None,
    }


@pytest.mark.parametrize(
    ("decision_kind", "legal_actions", "selected_code"),
    [
        (
            "bid",
            [
                {"code": "pass", "action": "pass", "trumpSuit": None},
                {"code": "sun", "action": "sun", "trumpSuit": None},
            ],
            "sun",
        ),
        (
            "escalation",
            [
                {"code": "decline", "action": "decline", "mode": None},
                {"code": "double:open", "action": "double", "mode": "open"},
            ],
            "double:open",
        ),
        (
            "project",
            [
                {
                    "code": "declare:hundred_sequence:100:14",
                    "action": "declare",
                    "owner": "south",
                    "kind": "hundred_sequence",
                    "value": 100,
                    "highRank": 14,
                }
            ],
            "declare:hundred_sequence:100:14",
        ),
    ],
)
def test_research_nonplay_locator_preserves_typed_actions_and_proof_boundary(
    root, tmp_path, decision_kind, legal_actions, selected_code
):
    service = _service(root, tmp_path, [1_000])
    digest = {
        "bid": "b1d0000000",
        "escalation": "e5ca1a7e00",
        "project": "a0b0c0d0e0",
    }[decision_kind] + "0" * 54
    code = f"LAF-{digest[:10].upper()}"
    exact = _research_public_exact_state(code)
    exact["decisionKind"] = decision_kind
    exact["phase"] = decision_kind
    exact["legalActions"] = legal_actions
    exact["publicState"]["phase"] = decision_kind
    proof = {
        "achievedMode": "policy_conditioned",
        "completionStatus": "policy_complete",
        "claimSatisfied": True,
        "selectedAction": selected_code,
        "rootActions": [
            {
                "action": action["code"],
                "value": 1 if action["code"] == selected_code else -1,
                "satisfies": action["code"] == selected_code,
            }
            for action in legal_actions
        ],
    }
    service.publish_research_result(
        result_code=code,
        result_hash=digest,
        exact_state_id="STATE-1234567890ABCDEF1234",
        public_exact_state=exact,
        proof_outcome=proof,
        truth_label="robust_best_known",
        confidence=1.0,
    )

    public = service.resolve(code)
    assert public["decisionKind"] == decision_kind
    assert public["legalActions"] == legal_actions
    assert public["copyProfile"]["bundleRevisionId"] == "CBR-TEST-FROZEN"
    assert public["copy"]["proof_boundary_policy_conditioned"]
    encoded = json.dumps(public, sort_keys=True)
    assert "hiddenHands" not in encoded
    attempt = service.commit_attempt(
        code,
        participant_id=f"{decision_kind}-player",
        chosen_action=next(
            action for action in legal_actions if action["code"] == selected_code
        ),
    )
    assert attempt["reveal"]["chosenAccepted"] is True
    assert attempt["reveal"]["proofBoundary"]["achievedMode"] == "policy_conditioned"


def test_research_result_locator_requires_complete_frozen_presentation(root, tmp_path):
    bundle = build_bundle(root)
    video = bundle.component("video")
    presentation = {
        "schemaVersion": 1,
        "bundleRevisionId": "CBR-TEST-FROZEN",
        "locale": video["render"]["locale"],
        "direction": video["render"]["direction"],
        "profileId": video["profile"]["id"],
        "profileVersion": video["profile"]["version"],
        "profileSha256": bundle.components["video"].sha256,
        "challengeCopy": copy.deepcopy(video["challenge_copy"]),
    }
    presentation["challengeCopy"].pop("action_accessibility")
    service = _service(root, tmp_path, [1_000], presentation=presentation)
    digest = "b1d0000000" + "0" * 54
    with pytest.raises(ValueError, match="lacks governed copy: action_accessibility"):
        service.publish_research_result(
            result_code="LAF-B1D0000000",
            result_hash=digest,
            exact_state_id="STATE-1234567890ABCDEF1234",
            public_exact_state=_research_public_exact_state("LAF-B1D0000000"),
            proof_outcome=_research_result("LAF-B1D0000000", digest)["proofOutcome"],
            truth_label="exact_optimal",
            confidence=1.0,
        )


def test_research_result_locator_requires_frozen_presentation_resolver(root, tmp_path):
    service = _service(root, tmp_path, [1_000])
    service.research_presentation_resolver = None
    digest = "b1d1000000" + "0" * 54
    with pytest.raises(ValueError, match="frozen presentation is unavailable"):
        service.publish_research_result(
            result_code="LAF-B1D1000000",
            result_hash=digest,
            exact_state_id="STATE-1234567890ABCDEF1234",
            public_exact_state=_research_public_exact_state("LAF-B1D1000000"),
            proof_outcome=_research_result("LAF-B1D1000000", digest)["proofOutcome"],
            truth_label="exact_optimal",
            confidence=1.0,
        )


def test_research_result_locator_rejects_public_payload_substitution(root, tmp_path):
    service = _service(root, tmp_path, [1_000])
    digest = "a1b2c3d4e5" + "0" * 54
    code = "LAF-A1B2C3D4E5"
    service.publish_research_result(
        result_code=code,
        result_hash=digest,
        exact_state_id="STATE-1234567890ABCDEF1234",
        public_exact_state=_research_public_exact_state(code),
        proof_outcome=_research_result(code, digest)["proofOutcome"],
        truth_label="exact_optimal",
        confidence=1.0,
    )
    stored = service.store.get(code)
    assert stored is not None
    forged = dict(stored["public"])
    forged["hiddenHands"] = {"north": [{"code": "SA"}]}
    with service.store._connect() as connection:
        connection.execute(
            "UPDATE challenge_locators_v1 SET public_json=? WHERE code=?",
            (json.dumps(forged, sort_keys=True, separators=(",", ":")), code),
        )
    with pytest.raises(PermissionError, match="integrity verification failed"):
        service.resolve(code)


def test_research_result_locator_rejects_presentation_profile_substitution(root, tmp_path):
    service = _service(root, tmp_path, [1_000])
    digest = "a1b2c3d4e6" + "0" * 54
    code = "LAF-A1B2C3D4E6"
    service.publish_research_result(
        result_code=code,
        result_hash=digest,
        exact_state_id="STATE-1234567890ABCDEF1234",
        public_exact_state=_research_public_exact_state(code),
        proof_outcome=_research_result(code, digest)["proofOutcome"],
        truth_label="exact_optimal",
        confidence=1.0,
    )
    stored = service.store.get(code)
    assert stored is not None
    forged_public = copy.deepcopy(stored["public"])
    forged_public["copyProfile"]["profileSha256"] = "f" * 64
    forged_binding = copy.deepcopy(stored["binding"])
    forged_binding["publicChallengeHash"] = hashlib.sha256(
        json.dumps(
            forged_public,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    forged_binding["bindingSignature"] = service._research_binding_signature(
        forged_binding
    )
    with service.store._connect() as connection:
        connection.execute(
            """UPDATE challenge_locators_v1
               SET binding_json=?,public_json=? WHERE code=?""",
            (
                json.dumps(forged_binding, sort_keys=True, separators=(",", ":")),
                json.dumps(forged_public, sort_keys=True, separators=(",", ":")),
                code,
            ),
        )
    with pytest.raises(PermissionError, match="presentation integrity verification failed"):
        service.resolve(code)


def test_research_result_locator_api_publishes_on_first_resolve(root, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from baloot_platform.service import api

    digest = "a1b2c3d4e5" + "0" * 54
    code = "LAF-A1B2C3D4E5"
    result = _research_result(code, digest)
    exact = _research_public_exact_state(code)
    service = _service(root, tmp_path, [1_000])

    class ResearchResults:
        @staticmethod
        def get_result(requested_code):
            return result if requested_code == code else None

        @staticmethod
        def get_exact_state(requested_code, *, observer_seat):
            assert requested_code == code
            assert observer_seat == "south"
            return exact

    monkeypatch.setattr(api, "RESEARCH", ResearchResults())
    monkeypatch.setattr(api, "CHALLENGE_LOCATORS", service)
    with TestClient(api.app) as client:
        response = client.get(f"/api/v2/challenge-locators/{code.lower()}")
        repeated = client.get(f"/api/v2/challenge-locators/{code}")
        result["disposition"] = "rejected"
        blocked_after_publication = client.get(
            f"/api/v2/challenge-locators/{code}"
        )
    assert response.status_code == 200, response.text
    assert repeated.json() == response.json()
    assert response.json()["code"] == code
    assert response.json()["state"]["handSizes"] == [2, 2, 2, 2]
    assert len(response.json()["state"]["observerHand"]) == 2
    assert len(response.json()["state"]["plays"]) == 5
    assert [item["code"] for item in response.json()["legalActions"]] == [
        "C7",
        "CA",
    ]
    assert blocked_after_publication.status_code == 404
    assert service.store.get(code) is not None


@pytest.mark.parametrize(
    "mutate",
    (
        lambda result: result.update({"disposition": "rejected"}),
        lambda result: result["proofOutcome"].update(
            {"completionStatus": "proof_incomplete"}
        ),
        lambda result: result["runtimeCompatibility"].update({"status": "failed"}),
        lambda result: result["scenarioProjection"].update(
            {"exactStateAvailable": False}
        ),
    ),
)
def test_research_result_locator_api_fails_closed_for_ineligible_result(
    root, tmp_path, monkeypatch, mutate
):
    from fastapi.testclient import TestClient
    from baloot_platform.service import api

    digest = "b1b2c3d4e5" + "0" * 54
    code = "LAF-B1B2C3D4E5"
    result = _research_result(code, digest)
    mutate(result)
    service = _service(root, tmp_path, [1_000])

    class ResearchResults:
        @staticmethod
        def get_result(requested_code):
            return result if requested_code == code else None

        @staticmethod
        def get_exact_state(_requested_code, *, observer_seat):
            raise AssertionError(f"ineligible result reached exact state: {observer_seat}")

    monkeypatch.setattr(api, "RESEARCH", ResearchResults())
    monkeypatch.setattr(api, "CHALLENGE_LOCATORS", service)
    with TestClient(api.app) as client:
        response = client.get(f"/api/v2/challenge-locators/{code}")
    assert response.status_code == 404
    assert service.store.get(code) is None


def test_exact_locator_is_opaque_observer_safe_deterministic_and_revocable(root, tmp_path):
    trace = NaturalLaffaSimulatorV1.from_root(root).simulate(2).final_trace
    decision = next(
        item for item in trace.decision_ledger.records
        if item.decision_kind == "play" and not item.forced
    )
    clock = [1_000]
    service = _service(root, tmp_path, clock)

    first = service.publish(
        trace,
        decision_id=decision.decision_id,
        observer_seat=decision.acting_seat,
        prompt_contract="choose_acting_action",
        expires_at=2_000,
    )
    second = service.publish(
        trace,
        decision_id=decision.decision_id,
        observer_seat=decision.acting_seat,
        prompt_contract="choose_acting_action",
        expires_at=2_000,
    )
    assert first.code == second.code and len(first.code) == 16
    public = service.resolve(first.code.lower())
    encoded = json.dumps(public, sort_keys=True)
    assert first.universal_link.endswith(first.code)
    assert public["observerSeat"] == decision.acting_seat.name.lower()
    assert public["answerCommitted"] is False
    assert len(public["legalActions"]) == len(decision.legal_actions)
    assert all(
        forbidden not in encoded
        for forbidden in (
            "traceHash", "decisionId", "projectionHash", "completeStateHash",
            "recommendedAction", "chosenAction", "trustedFinalState",
        )
    )

    observer = decision.acting_seat.next()
    observer_locator = service.publish(
        trace,
        decision_id=decision.decision_id,
        observer_seat=observer,
        prompt_contract="identify_threat",
    )
    observer_public = service.resolve(observer_locator.code)
    assert observer_public["observerSeat"] == observer.name.lower()
    assert observer_public["legalActions"] == []

    assert service.revoke(first.code)
    with pytest.raises(PermissionError, match="revoked"):
        service.resolve(first.code)


def test_locator_binds_safe_player_names_without_internal_identity(root, tmp_path):
    trace = NaturalLaffaSimulatorV1.from_root(root).simulate(3).final_trace
    decision = next(
        item for item in trace.decision_ledger.records
        if item.decision_kind == "play" and not item.forced
    )
    service = _service(root, tmp_path, [1_000])
    names = {
        "north": "Player 01",
        "west": "Player 04",
        "south": "Player 03",
        "east": "Player 02",
    }
    locator = service.publish(
        trace,
        decision_id=decision.decision_id,
        observer_seat=decision.acting_seat,
        prompt_contract="choose_acting_action",
        player_names=names,
    )
    public = service.resolve(locator.code)
    assert public["playerNames"] == names
    assert "decisionId" not in json.dumps(public, ensure_ascii=False)

    with pytest.raises(ValueError, match="exactly four seats"):
        service.publish(
            trace,
            decision_id=decision.decision_id,
            observer_seat=decision.acting_seat,
            prompt_contract="choose_acting_action",
            player_names={"south": "Player 03"},
        )


def test_locator_rejects_wrong_perspective_for_actor_prompt_and_expires(root, tmp_path):
    trace = NaturalLaffaSimulatorV1.from_root(root).simulate(2).final_trace
    decision = next(
        item for item in trace.decision_ledger.records
        if item.decision_kind == "play" and not item.forced
    )
    clock = [1_000]
    service = _service(root, tmp_path, clock)
    with pytest.raises(ValueError, match="acting-seat"):
        service.publish(
            trace,
            decision_id=decision.decision_id,
            observer_seat=decision.acting_seat.next(),
            prompt_contract="choose_acting_action",
        )
    locator = service.publish(
        trace,
        decision_id=decision.decision_id,
        observer_seat=decision.acting_seat,
        prompt_contract="choose_acting_action",
        expires_at=1_001,
    )
    clock[0] = 1_001
    with pytest.raises(PermissionError, match="expired"):
        service.resolve(locator.code)


def test_locator_attempt_is_single_commit_and_statistics_are_post_choice_only(root, tmp_path):
    trace = NaturalLaffaSimulatorV1.from_root(root).simulate(7).final_trace
    decision = next(
        item for item in trace.decision_ledger.records
        if item.decision_kind == "play" and not item.forced
    )
    service = _service(root, tmp_path, [1_000])
    locator = service.publish(
        trace,
        decision_id=decision.decision_id,
        observer_seat=decision.acting_seat,
        prompt_contract="choose_acting_action",
    )
    public = service.resolve(locator.code)
    assert "statistics" not in public
    assert "sourceGameAction" not in json.dumps(public)

    first_action = dict(decision.legal_actions[0])
    first = service.commit_attempt(
        locator.code,
        participant_id="guest-one",
        chosen_action=first_action,
        response_seconds=12.5,
        confidence=0.8,
        changed_answer=True,
    )
    assert first["committed"] is True
    assert first["reveal"]["status"] == "source_game_only"
    assert first["reveal"]["truthLabel"] == "unresolved"
    assert first["reveal"]["recommendationStatus"] == "not_available"
    assert first["statistics"]["totalAttempts"] == 1
    assert first["statistics"]["distinctParticipants"] == 1
    assert first["statistics"]["choices"][0]["percentage"] == 100.0
    assert first["statistics"]["audienceDifficulty"]["status"] == "not_calibrated"
    assert "participantKey" not in json.dumps(first)

    retry = service.commit_attempt(
        locator.code,
        participant_id="guest-one",
        chosen_action=first_action,
        response_seconds=99.5,
        confidence=0.2,
        changed_answer=False,
    )
    assert retry["attemptId"] == first["attemptId"]
    assert retry["statistics"]["totalAttempts"] == 1
    restored = service.restore_attempt(
        locator.code,
        participant_id="guest-one",
    )
    assert restored is not None
    assert restored["attemptId"] == first["attemptId"]
    assert restored["chosenAction"] == first_action
    assert service.restore_attempt(
        locator.code,
        participant_id="guest-without-an-attempt",
    ) is None

    second_action = dict(decision.legal_actions[-1])
    if second_action != first_action:
        with pytest.raises(ChallengeAttemptConflictError, match="already committed"):
            service.commit_attempt(
                locator.code,
                participant_id="guest-one",
                chosen_action=second_action,
            )
        second = service.commit_attempt(
            locator.code,
            participant_id="guest-two",
            chosen_action=second_action,
            response_seconds=7.5,
            confidence=0.6,
        )
        assert second["statistics"]["totalAttempts"] == 2
        assert {item["percentage"] for item in second["statistics"]["choices"]} == {50.0}

    with pytest.raises(ValueError, match="not legal"):
        service.commit_attempt(
            locator.code,
            participant_id="guest-three",
            chosen_action={"rank": "ace", "suit": "not-a-suit"},
        )


def test_observer_prompt_attempt_fails_closed_until_option_contract_exists(root, tmp_path):
    trace = NaturalLaffaSimulatorV1.from_root(root).simulate(8).final_trace
    decision = next(
        item for item in trace.decision_ledger.records
        if item.decision_kind == "play" and not item.forced
    )
    service = _service(root, tmp_path, [1_000])
    locator = service.publish(
        trace,
        decision_id=decision.decision_id,
        observer_seat=decision.acting_seat.next(),
        prompt_contract="identify_threat",
    )
    with pytest.raises(ValueError, match="option schema"):
        service.commit_attempt(
            locator.code,
            participant_id="observer-one",
            chosen_action={"value": "threat"},
        )


def test_locator_api_creates_and_resolves_exact_public_state(root, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from baloot_platform.service import api

    trace = NaturalLaffaSimulatorV1.from_root(root).simulate(2).final_trace
    decision = next(
        item for item in trace.decision_ledger.records
        if item.decision_kind == "play" and not item.forced
    )
    service = _service(root, tmp_path, [1_000])

    class TraceSource:
        def complete_game_trace(self, _identifier):
            return trace

    monkeypatch.setattr(api, "LEARNING_STORE", TraceSource())
    monkeypatch.setattr(api, "CHALLENGE_LOCATORS", service)
    with TestClient(api.app) as client:
        created = client.post(
            "/api/v2/challenge-locators",
            json={
                "laffaIdentifier": trace.game_hash,
                "decisionId": decision.decision_id,
                "observerSeat": decision.acting_seat.name.lower(),
                "promptContract": "choose_acting_action",
                "revealPolicy": "after_commit",
            },
        )
        assert created.status_code == 201, created.text
        code = created.json()["code"]
        player = client.get(f"/c/{code}")
        assert player.status_code == 200, player.text
        assert player.headers["cache-control"] == "no-store"
        assert 'id="loading-copy"' in player.text
        copy_response = client.get("/api/v1/challenge-copy")
        assert copy_response.status_code == 200, copy_response.text
        governed_copy = copy_response.json()["copy"]
        assert governed_copy["page_title"] == "تحدّي BalootX"
        assert 'id="challenge-copy"' not in player.text
        assert governed_copy["page_title"] not in player.text
        assert "{{" not in player.text and "}}" not in player.text
        script = client.get("/challenge-assets/app.js")
        assert script.status_code == 200, script.text
        assert "/api/v2/challenge-locators/" in script.text
        assert "recommendedActionPercentage" in script.text
        assert "expertReason" in script.text
        assert "expectedRoundMargin" in script.text
        assert "pointermove" in script.text
        assert "fromSwipe" in script.text
        assert "bindFrozenCopy" in script.text
        assert "Missing governed challenge copy" in script.text
        assert not any("\u0600" <= character <= "\u06ff" for character in script.text)
        assert "/card-assets/" in script.text
        assert 'id="expert-reason"' in player.text
        assert 'id="drag-hint"' in player.text
        face = client.get("/card-assets/A_spades.svg")
        assert face.status_code == 200, face.text
        back = client.get("/challenge-video-assets/card-back.svg")
        assert back.status_code == 200, back.text
        resolved = client.get(f"/api/v2/challenge-locators/{code}")
        assert resolved.status_code == 200, resolved.text
        payload = resolved.json()
        assert payload["code"] == code
        assert payload["copy"] == governed_copy
        assert "decisionId" not in json.dumps(payload)
        action = payload["legalActions"][0]
        anonymous = client.post(
            f"/api/v2/challenge-locators/{code}/attempts",
            json={"chosenAction": action},
        )
        assert anonymous.status_code == 401, anonymous.text
        guest = client.post(
            "/api/v1/auth/guest",
            json={"displayName": "Locator guest"},
        )
        assert guest.status_code == 201, guest.text
        pristine = client.get(
            f"/api/v2/challenge-locators/{code}/attempt"
        )
        assert pristine.status_code == 200, pristine.text
        assert pristine.json()["committed"] is False
        rejected = client.post(
            f"/api/v2/challenge-locators/{code}/attempts",
            json={
                "participantId": "another-participant",
                "chosenAction": action,
            },
        )
        assert rejected.status_code == 403, rejected.text
        attempted = client.post(
            f"/api/v2/challenge-locators/{code}/attempts",
            json={
                "chosenAction": action,
                "responseSeconds": 9.25,
                "confidence": 0.75,
            },
        )
        assert attempted.status_code == 201, attempted.text
        attempt_payload = attempted.json()
        assert attempt_payload["statistics"]["totalAttempts"] == 1
        assert attempt_payload["reveal"]["truthLabel"] == "unresolved"
        assert "decisionId" not in json.dumps(attempt_payload)
        restored = client.get(
            f"/api/v2/challenge-locators/{code}/attempt"
        )
        assert restored.status_code == 200, restored.text
        assert restored.json()["attemptId"] == attempt_payload["attemptId"]
        assert restored.json()["statistics"]["totalAttempts"] == 1
        assert "decisionId" not in restored.text
        revoked = client.post(f"/api/v2/challenge-locators/{code}:revoke")
        assert revoked.status_code == 200
        assert client.get(f"/api/v2/challenge-locators/{code}").status_code == 410
