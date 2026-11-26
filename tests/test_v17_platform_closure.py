from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from baloot_platform.adjudication import (
    AdjudicationProfile,
    AdjudicationService,
    CaseStatus,
    RulingOutcome,
    SQLiteAdjudicationStore,
)
from baloot_platform.analysis.belief_calibration_v17 import (
    evaluate_multiclass_calibration,
    flatten_card_location_examples,
)
from baloot_platform.analysis.models import InformationState
from baloot_platform.analysis.team_public_cfr_v17 import analyze_team_public_coordinator
from baloot_platform.config import build_bundle
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.core.models import Event
from baloot_platform.core.scenario import ScenarioRequest, generate_episode
from baloot_platform.history.canonical_store_v17 import CanonicalHistoryStore
from baloot_platform.live import InMemoryLiveEventBroker, LiveStatus, LiveTableService, SQLiteLiveStore
from baloot_platform.ml.features import FEATURE_COUNT
from baloot_platform.ml.league_v17 import (
    LeagueAgent,
    LeagueMatch,
    ModelEvaluation,
    ModelPromotionPolicy,
    SelfPlayLeague,
)
from baloot_platform.ml.models import UniformLegalModel
from baloot_platform.security import (
    AesGcmPayloadCipher,
    AuthorizationPolicy,
    PasskeyChallengeService,
    PkceTransactionStore,
    Scope,
    SignedAccessTokenService,
)
from baloot_platform.service.game import GameManager
from baloot_platform.storage.game_store import SQLiteGameStore


def test_adjudication_lifecycle_hash_appeal_and_directive(root: Path, tmp_path: Path):
    store = SQLiteAdjudicationStore(tmp_path / "adjudication.sqlite3")
    profile = AdjudicationProfile.from_yaml(root / "configs/adjudication/saudi_researched_v1.yaml")
    service = AdjudicationService(store, profile)
    case = service.open_case(
        game_id="GAME-17", violation="illegal_card", reported_by="USR-A",
        description="لعب ورقة غير قانونية", event_sequence=12, accused_seat="west",
    )
    assert service.verify(case.case_id)["valid"] is True
    case = service.add_evidence(
        case.case_id, kind="video", submitted_by="USR-A", description="المقطع المتزامن",
        uri="artifact://GAME-17/decision-12", sha256="a" * 64,
    )
    assert case.status == CaseStatus.UNDER_REVIEW
    recommendation = service.recommended_ruling(case.case_id)
    assert recommendation["recommendedOutcome"] == "correct_action"
    ruled = service.rule(
        case.case_id, referee_id="REF-1", outcome=RulingOutcome.REPLAY_FROM_EVENT,
        reason_ar="نعيد القرار من آخر حالة سليمة", replay_from_event_sequence=11,
    )
    assert ruled.status == CaseStatus.RULED
    appealed = service.appeal(case.case_id, appealed_by="USR-B", reason="المقطع الكامل يغير السياق")
    assert appealed.status == CaseStatus.APPEALED
    final = service.resolve_appeal(
        case.case_id, referee_id="REF-2", outcome=RulingOutcome.CORRECT_ACTION,
        reason_ar="ثبت الخطأ وتصحح الورقة", replay_from_event_sequence=11,
    )
    assert final.status == CaseStatus.FINAL
    directive = service.directive(case.case_id)
    assert directive["mustCreateCorrectionLineage"] is True
    assert directive["replayFromEventSequence"] == 11
    assert service.verify(case.case_id)["valid"] is True


def test_live_four_human_room_full_round_and_reconnect(root: Path, tmp_path: Path):
    manager = GameManager(root, SQLiteGameStore(tmp_path / "games.sqlite3"))
    game = manager.create(seed=17001, dealer=Seat.EAST)
    service = LiveTableService(SQLiteLiveStore(tmp_path / "live.sqlite3"), manager)
    room = service.create_room(game_id=game.id, owner_subject_id="USR-N", turn_seconds=30)
    subjects = {
        Seat.NORTH: "USR-N", Seat.WEST: "USR-W", Seat.SOUTH: "USR-S", Seat.EAST: "USR-E",
    }
    tokens: dict[Seat, str] = {}
    for seat, subject in subjects.items():
        claimed = service.claim_seat(room.room_id, subject_id=subject, display_name=subject, seat=seat)
        tokens[seat] = claimed["reconnectToken"]
        service.set_ready(room.room_id, subject_id=subject)
    started = service.start(room.room_id, requested_by="USR-N")
    assert started.status == LiveStatus.ACTIVE

    # Deterministic legal auction: North passes, West buys Hokum, all others pass.
    auction = [
        (Seat.NORTH, {"kind": "bid", "action": "pass"}),
        (Seat.WEST, {"kind": "bid", "action": "hokum"}),
        (Seat.SOUTH, {"kind": "bid", "action": "pass"}),
        (Seat.EAST, {"kind": "bid", "action": "pass"}),
        (Seat.NORTH, {"kind": "bid", "action": "pass"}),
    ]
    for seat, action in auction:
        payload = service.submit_action(room.room_id, subject_id=subjects[seat], action=action)
        assert payload["room"]["status"] == "active"

    # Reconnect preserves the exact seat and room history.
    service.disconnect(room.room_id, subject_id="USR-E")
    with pytest.raises(PermissionError):
        service.reconnect(room.room_id, subject_id="USR-E", reconnect_token="wrong-token")
    service.reconnect(room.room_id, subject_id="USR-E", reconnect_token=tokens[Seat.EAST])

    for _ in range(40):
        session = manager.get(game.id)
        if session.state.current_player is None:
            break
        actor = session.state.current_player
        card = session.engine.legal_moves(session.state)[0]
        payload = service.submit_action(
            room.room_id, subject_id=subjects[actor],
            action={"kind": "play", "card": card.code, "expectedVersion": session.version},
        )
    final_room = service.store.get(room.room_id)
    assert final_room is not None and final_room.status == LiveStatus.COMPLETED
    verification = service.verify(room.room_id)
    assert verification["valid"] is True, verification
    assert service.view(room.room_id, subject_id="USR-S")["viewerSeat"] == "south"



@pytest.mark.real_clock
def test_live_timeout_creates_adjudication_case(root: Path, tmp_path: Path):
    adjudication = AdjudicationService(
        SQLiteAdjudicationStore(tmp_path / "adj-live.sqlite3"),
        AdjudicationProfile.from_yaml(root / "configs/adjudication/saudi_researched_v1.yaml"),
    )
    manager = GameManager(root, SQLiteGameStore(tmp_path / "games-timeout.sqlite3"))
    game = manager.create(seed=17002, dealer=Seat.EAST)
    service = LiveTableService(
        SQLiteLiveStore(tmp_path / "live-timeout.sqlite3"), manager, adjudication=adjudication,
    )
    room = service.create_room(game_id=game.id, owner_subject_id="N", turn_seconds=5)
    for seat, subject in ((Seat.NORTH, "N"), (Seat.WEST, "W"), (Seat.SOUTH, "S"), (Seat.EAST, "E")):
        service.claim_seat(room.room_id, subject_id=subject, display_name=subject, seat=seat)
        service.set_ready(room.room_id, subject_id=subject)
    service.start(room.room_id, requested_by="N")
    loaded = service.store.get(room.room_id)
    loaded.turn_deadline = "2000-01-01T00:00:00+00:00"
    loaded.rehash(); service.store.save(loaded)
    result = service.enforce_deadline(room.room_id)
    assert result["expired"] is True and result["adjudicationCaseId"]
    case = adjudication.store.get(result["adjudicationCaseId"])
    assert case is not None and case.violation.value == "timeout"


def test_access_tokens_oidc_pkce_passkeys_and_authorization():
    token_service = SignedAccessTokenService(b"v17-secure-secret-which-is-at-least-32-bytes")
    token = token_service.issue(
        subject_id="USR-17", scopes=[Scope.READ, Scope.PLAY, Scope.LIVE],
        role="player", session_id="SES-17", now=1_700_000_000,
    )
    claims = token_service.verify(token, now=1_700_000_010)
    assert claims.subject_id == "USR-17" and Scope.LIVE.value in claims.scopes
    with pytest.raises(ValueError):
        token_service.verify(token, now=1_700_001_000)
    policy = AuthorizationPolicy()
    policy.require(role="player", required=Scope.PLAY, claims=claims)
    with pytest.raises(PermissionError):
        policy.require(role="player", required=Scope.REFEREE, claims=claims)

    pkce = PkceTransactionStore(ttl_seconds=60)
    transaction = pkce.create("provider", return_to="/app")
    consumed = pkce.consume(transaction.state, now=transaction.created_at + 1)
    assert consumed.challenge == PkceTransactionStore._challenge(consumed.verifier)
    with pytest.raises(ValueError):
        pkce.consume(transaction.state)

    passkeys = PasskeyChallengeService(rp_id="baloot.local", origin="https://baloot.local")
    registration = passkeys.begin(subject_id="USR-17", purpose="register")
    credential = passkeys.register_verified(
        challenge=registration["challenge"], subject_id="USR-17", credential_id="CRED-17",
        public_key_cose=b"public-key", sign_count=1, transports=("internal",),
    )
    assert credential.sign_count == 1
    authentication = passkeys.begin(subject_id="USR-17", purpose="authenticate")
    updated = passkeys.record_verified_assertion(
        challenge=authentication["challenge"], subject_id="USR-17", credential_id="CRED-17", sign_count=2,
    )
    assert updated.sign_count == 2
    with pytest.raises(ValueError):
        passkeys.record_verified_assertion(
            challenge=passkeys.begin(subject_id="USR-17", purpose="authenticate")["challenge"],
            subject_id="USR-17", credential_id="CRED-17", sign_count=2,
        )


def test_canonical_event_store_snapshots_branches_and_corrections(tmp_path: Path):
    store = CanonicalHistoryStore(tmp_path / "canonical.sqlite3")
    events = (Event(0, "round_created", {"seed": 17}), Event(1, "card_played", {"card": "S7"}))
    appended = store.append(
        game_id="GAME-17:round:1", events=events,
        profile_bundle_hash="b" * 64, engine_version="17.0.0",
    )
    assert len(appended) == 2
    snapshot = store.snapshot(
        game_id="GAME-17:round:1", state={"phase": "playing", "played": ["S7"]},
        engine_version="17.0.0", profile_bundle_hash="b" * 64,
    )
    assert snapshot.event_count == 2
    branch = store.create_branch(
        parent_game_id="GAME-17:round:1", child_game_id="GAME-17-BRANCH",
        created_by="USR-17", reason="تجربة مسار آخر",
    )
    correction = store.create_correction(
        source_game_id="GAME-17:round:1", corrected_game_id="GAME-17-CORRECTED",
        created_by="REF-17", reason="قرار حكم", adjudication_case_id="ADJ-17",
    )
    lineage = store.lineage("GAME-17:round:1")
    assert lineage["branches"][0]["child_game_id"] == branch.child_game_id
    assert lineage["corrections"][0]["corrected_game_id"] == correction.corrected_game_id
    assert store.verify("GAME-17:round:1")["valid"] is True


def test_calibration_league_and_promotion_gates():
    report = evaluate_multiclass_calibration(
        [{"west": 0.8, "east": 0.2}, {"west": 0.3, "east": 0.7}],
        ["west", "east"], bins=5,
    )
    assert report.examples == 2 and report.top1_accuracy == 1.0
    rows, labels = flatten_card_location_examples(
        [{"S7": {"west": 0.9, "east": 0.1}}], [{"S7": "west"}],
    )
    assert labels == ["west"] and rows[0]["west"] == 0.9

    league = SelfPlayLeague()
    a = LeagueAgent.create(model_id="policy-a", version="1", artifact_hash="a" * 64)
    b = LeagueAgent.create(model_id="policy-b", version="1", artifact_hash="b" * 64)
    league.register(a); league.register(b)
    league.record(LeagueMatch.create(
        board_set_hash="c" * 64, agent_a=a.agent_id, agent_b=b.agent_id,
        score_a=0.6, score_b=0.4, decisions=256,
    ))
    assert league.leaderboard()[0]["agent"]["agentId"] == a.agent_id
    assert league.snapshot()["snapshotHash"]

    policy = ModelPromotionPolicy(minimum_board_sets=2, minimum_decisions=100)
    approved = policy.decide(ModelEvaluation.create(
        model_id="candidate", legal_violation_rate=0.0, relative_score_rate=0.55,
        lower_confidence_bound=0.52, calibration_error=0.03, p95_latency_ms=20,
        ood_detection_recall=0.9, board_sets=5, decisions=1000,
    ))
    rejected = policy.decide(ModelEvaluation.create(
        model_id="bad", legal_violation_rate=0.01, relative_score_rate=0.49,
        lower_confidence_bound=0.45, calibration_error=0.2, p95_latency_ms=80,
        ood_detection_recall=0.5, board_sets=1, decisions=10,
    ))
    assert approved.approved is True
    assert rejected.approved is False and "illegal_action_rate_must_be_zero" in rejected.reasons


def test_root_consistent_team_public_search_is_legal(root: Path):
    bundle = build_bundle(root)
    episode = generate_episode(
        bundle.rules, bundle.convention, bundle_hash=bundle.sha256,
        request=ScenarioRequest.governed(
            seed=7351, viewer=Seat.SOUTH, contract=Contract.HOKUM,
            requested_samples=4, exact_allocation_limit=500,
            pause_trick_min=6, pause_trick_max=6,
            require_project=False, max_seed_attempts=30, minimum_hidden_deals=1,
        ),
    )
    engine = BalootEngine(bundle.rules)
    info = InformationState.from_complete(episode.state_before_pause.clone(), episode.viewer)
    result = analyze_team_public_coordinator(
        info, engine, particles=4, exact_allocation_limit=500,
        iterations=20, max_nodes=10_000, max_seconds=0.1, seed=17,
    )
    legal = {card.id for card in engine.legal_moves(episode.state_before_pause)}
    assert result.recommended_card.id in legal
    assert result.method == "root_consistent_team_public_regret_matching_v1"
    assert result.guarantee.startswith("one_shared_root_policy")


def test_ml_profile_and_uniform_model_are_observer_safe(root: Path):
    bundle = build_bundle(root)
    ml = bundle.component("ml")
    assert ml["self_play"]["enabled"] is True
    assert ml["promotion"]["maximum_legal_violation_rate"] == 0.0
    model = UniformLegalModel()
    legal_mask = tuple(1 if index in {2, 5, 8} else 0 for index in range(46))
    output = model.infer(np.zeros(FEATURE_COUNT, dtype=np.float32), legal_mask)
    assert output.policy_logits[2] == 0
    assert output.policy_logits[0] < -1e8



def test_encrypted_canonical_snapshots_and_live_broker(tmp_path: Path):
    cipher = AesGcmPayloadCipher.from_secret("v17-hidden-state-encryption-secret")
    store = CanonicalHistoryStore(tmp_path / "encrypted.sqlite3", cipher=cipher)
    store.append(
        game_id="ENC:round:1", events=(Event(0, "round_created", {"seed": 9}),),
        profile_bundle_hash="d" * 64, engine_version="17.0.0",
    )
    original = {"hands": {"south": ["SA", "H10"]}, "phase": "playing"}
    store.snapshot(
        game_id="ENC:round:1", state=original, engine_version="17.0.0", profile_bundle_hash="d" * 64,
    )
    import sqlite3
    with sqlite3.connect(tmp_path / "encrypted.sqlite3") as conn:
        raw = conn.execute("SELECT state_json FROM canonical_snapshots_v17").fetchone()[0]
    assert raw.startswith("enc:v1:") and "SA" not in raw
    assert store.latest_snapshot("ENC:round:1").state_payload == original

    import asyncio
    broker = InMemoryLiveEventBroker()
    async def exercise():
        waiter = asyncio.create_task(broker.wait("ROOM", 0, timeout=1.0))
        await asyncio.sleep(0.01)
        broker.publish("ROOM", 2)
        assert await waiter == 2
    asyncio.run(exercise())
    broker.close()

def test_v17_api_routes_and_unified_arabic_app(monkeypatch):
    monkeypatch.setenv("BALOOT_ENV", "test")
    from baloot_platform.service.api import app
    client = TestClient(app)
    capabilities = client.get("/api/v1/security/capabilities")
    assert capabilities.status_code == 200 and capabilities.json()["passkeys"] is True
    guest = client.post("/api/v1/auth/guest", json={"displayName": "ضيف V17"})
    assert guest.status_code == 201
    app_page = client.get("/app")
    assert app_page.status_code == 200
    text = app_page.text
    for label in ("شارك لفة أو صكّة", "صكّة كاملة", "ليش لعبها؟", "طاولة مباشرة"):
        assert label in text
    assert '"request_timeout":' in text and "{{" not in text
    assert '<html lang="ar" dir="rtl">' in text and 'id="shareCode" dir="ltr"' in text
    shared = client.get("/shared-assets/request.js")
    assert shared.status_code == 200 and "REQUEST_DEADLINE_MS" in shared.text
    openapi = client.get("/openapi.json").json()
    paths = openapi["paths"]
    assert "/api/v1/live/rooms" in paths
    assert "/api/v1/adjudication/cases" in paths
    assert "/api/v1/security/capabilities" in paths
    assert "/api/v1/canonical-history/{identifier}/verification" in paths
