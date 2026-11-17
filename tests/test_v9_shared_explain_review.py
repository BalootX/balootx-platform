from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.core.enums import Seat
from baloot_platform.explain import ExplanationService
from baloot_platform.explain.arabic import explain_decision_ar
from baloot_platform.learning.service import LearningService
from baloot_platform.learning.store import LearningStore
from baloot_platform.review import DecisionReviewService
from baloot_platform.sakka import SakkaService, SQLiteSakkaStore
from baloot_platform.service.game import GameManager
from baloot_platform.shared import (
    SharedDecisionAccessError,
    SharedDecisionAccessV1,
    SharedScope,
    project_shared_decision,
    public_shared_environment,
    resolve_shared_decision_access,
    resolve_shared_game_access,
    verify_shared_environment,
)
from baloot_platform.storage.codec_v7 import decode
from baloot_platform.storage.game_store import SQLiteGameStore


class _NoAudit:
    def audit_decision(self, decision_id: str, *, deep: bool = False, force: bool = False):
        return {
            "decisionId": decision_id,
            "bestAction": None,
            "expectedRegret": 0.0,
            "winProbabilityDrop": 0.0,
            "auditConfidence": 0.8,
            "reasons": [],
        }


def test_active_opponent_projection_redacts_analysis_until_completion():
    decision = {
        "decisionId": "DEC-OPPONENT",
        "ordinal": 3,
        "decisionKind": "play",
        "actor": "east",
        "selectedAction": "S7",
        "opponentOptionCounts": {"east": 2},
        "actionValues": {"S7": -4, "SA": 8},
        "rootValues": {"S7": -9, "SA": 12},
        "legalActions": ["S7", "SA"],
        "legalActionSet": ["S7", "SA"],
        "audits": [{"bestAction": "SA", "principalVariation": ["SA", "H7"]}],
        "bestAction": "SA",
        "principalVariation": ["SA", "H7"],
        "principalVariations": {"SA": ["SA", "H7"]},
    }
    active = SharedDecisionAccessV1(
        run_id="RUN-1",
        run_status="active",
        owner_subject_id="USR-A",
        observer_seat=Seat.SOUTH,
        actor_relation="left_opponent",
    )

    hidden = project_shared_decision(decision, active, subject_id="USR-A")

    private_fields = {
        "opponentOptionCounts",
        "actionValues",
        "rootValues",
        "legalActions",
        "legalActionSet",
        "audits",
        "bestAction",
        "principalVariation",
        "principalVariations",
    }
    assert private_fields.isdisjoint(hidden)
    assert hidden["privateAnalysisRedacted"] is True

    completed = SharedDecisionAccessV1(
        run_id="RUN-1",
        run_status="completed",
        owner_subject_id="USR-A",
        observer_seat=Seat.SOUTH,
        actor_relation="left_opponent",
    )
    assert project_shared_decision(decision, completed, subject_id="USR-A") == decision


def test_shared_decision_access_fails_closed_for_unknown_identity_and_run_state():
    decision = {
        "decisionId": "DEC-1",
        "ordinal": 3,
        "decisionKind": "play",
        "actor": "south",
        "selectedAction": "SA",
        "legalActions": ["SA", "S7"],
        "audits": [{"bestAction": "SA"}],
    }
    active = SharedDecisionAccessV1(
        run_id="RUN-1",
        run_status="active",
        owner_subject_id="USR-A",
        observer_seat=Seat.SOUTH,
        actor_relation="self",
    )

    projected = project_shared_decision(decision, active, subject_id=None)

    assert projected["selectedAction"] == "SA"
    assert projected["privateAnalysisRedacted"] is True
    assert "legalActions" not in projected
    assert "audits" not in projected

    class UnknownStateStore:
        def get_run_for_decision(self, _decision_id):
            return (
                SimpleNamespace(
                    run_id="RUN-1",
                    status="unknown",
                    subject_id="USR-A",
                    board_id="BOARD-1",
                ),
                b"",
            )

        def get_run_for_game(self, _game_id):
            return None

        def get_board(self, _board_id):
            return (SimpleNamespace(observer_seat=Seat.SOUTH), b"")

    with pytest.raises(SharedDecisionAccessError, match="authority is invalid"):
        resolve_shared_decision_access(UnknownStateStore(), decision)


def test_aggregate_history_access_uses_shared_run_completion(monkeypatch):
    from fastapi import HTTPException
    from baloot_platform.service import api

    run = SimpleNamespace(run_id="RUN-AGG", status="active", subject_id="USR-A", board_id="BOARD-AGG")
    board = SimpleNamespace(observer_seat=Seat.SOUTH)
    game = {"gameId": "GAME-AGG", "gameHash": "HASH-AGG", "decisions": [{
        "decisionId": "DEC-OPP", "actor": "east", "legalActions": ["SA", "S7"],
    }]}
    report = {"gameId": "GAME-AGG", "decisions": game["decisions"]}
    shared_store = SimpleNamespace(
        get_run_for_game=lambda _: (run, b""), get_board=lambda _: (board, b""),
        get_run_for_decision=lambda _: None,
    )
    learning_store = SimpleNamespace(
        get_game=lambda _: game, get_game_audit=lambda _: report,
        get_decision=lambda _: {**game["decisions"][0], "gameId": "GAME-AGG"},
        list_game_audits=lambda *args, **kwargs: [report],
        verify_game=lambda _: {"valid": True},
        list_video_records=lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(api, "SAKKA", SimpleNamespace(store=shared_store))
    monkeypatch.setattr(api, "LEARNING_STORE", learning_store)
    calls = (
        lambda: api.history_game("HASH-AGG"),
        lambda: api.replay_history_package("HASH-AGG"),
        lambda: api.history_game_video_records("HASH-AGG"),
        lambda: api.verify_history_game("HASH-AGG"),
        lambda: api.reopen_history_game("HASH-AGG"),
        lambda: api.history_game_videos("HASH-AGG"),
        lambda: api.render_history_game("HASH-AGG", api.HistoricalReplayRenderRequest()),
        lambda: api.audit_history_game("HASH-AGG", api.ReplayAuditRequest()),
        lambda: api.list_history_game_audits("HASH-AGG"),
        lambda: api.get_replay_audit("AUD-AGG", None),
    )
    for call in calls:
        with pytest.raises(HTTPException) as denied:
            call()
        assert denied.value.status_code == 403

    # A completed archive inside an active match is not completed-run authority.
    game["status"] = "completed"
    with pytest.raises(HTTPException):
        api.history_game("HASH-AGG")
    run.status = "completed"
    assert api.history_game("HASH-AGG") == game
    assert api.replay_history_package("HASH-AGG")["decisions"] == game["decisions"]
    assert api.get_replay_audit("AUD-AGG", None) == report
    run.status = "active"
    shared_store.get_run_for_game = lambda _: None
    shared_store.get_run_for_decision = lambda _: (run, b"")
    with pytest.raises(HTTPException) as nested_denied:
        api.get_replay_audit("AUD-AGG", None)
    assert nested_denied.value.status_code == 403
    shared_store.get_run_for_decision = lambda _: None
    assert api.history_game("HASH-AGG") == game
    assert api.get_replay_audit("AUD-AGG", None) == report


def test_aggregate_shared_game_access_fails_closed_for_broken_links():
    run = SimpleNamespace(run_id="RUN-1", status="active", subject_id="USR-A", board_id="BOARD-1")
    store = SimpleNamespace(get_run_for_game=lambda _: (run, b""), get_board=lambda _: None)
    with pytest.raises(SharedDecisionAccessError, match="board is unavailable"):
        resolve_shared_game_access(store, "GAME-1")
    store.get_board = lambda _: (SimpleNamespace(observer_seat=Seat.SOUTH), b"")
    run.status = "unknown"
    with pytest.raises(SharedDecisionAccessError, match="authority is invalid"):
        resolve_shared_game_access(store, "GAME-1")
    with pytest.raises(SharedDecisionAccessError, match="identity is unavailable"):
        resolve_shared_game_access(store, "")


def test_grpc_aggregate_methods_deny_active_shared_game(root: Path):
    # Exercise the actual adapter bodies without starting its optional server.
    import ast
    import asyncio

    source = root / "src/baloot_platform/service/grpc_service.py"
    serve = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                 if isinstance(node, ast.AsyncFunctionDef) and node.name == "serve")
    guard = next(node for node in serve.body
                 if isinstance(node, ast.AsyncFunctionDef) and node.name == "_require_private_game")
    service = next(node for node in serve.body if isinstance(node, ast.ClassDef) and node.name == "Service")
    methods = [node for node in service.body if isinstance(node, ast.AsyncFunctionDef)
               and node.name in {"GetReplay", "ReopenGame", "AuditGame", "RenderGameReplay"}]
    assert len(methods) == 4
    run = SimpleNamespace(run_id="RUN-1", status="active", subject_id="USR-A", board_id="BOARD-1")
    async def allow_api(context):
        pass
    async def abort(code, detail):
        raise PermissionError(detail)
    namespace = {
        "Any": object, "_require_api": allow_api,
        "LEARNING_STORE": SimpleNamespace(get_game=lambda _: {"gameId": "GAME-1"}),
        "SAKKA": SimpleNamespace(store=SimpleNamespace(
            get_run_for_game=lambda _: (run, b""),
            get_board=lambda _: (SimpleNamespace(observer_seat=Seat.SOUTH), b""))),
        "resolve_shared_game_access": resolve_shared_game_access,
        "resolve_shared_decision_access": resolve_shared_decision_access,
        "SharedDecisionAccessError": SharedDecisionAccessError,
        "grpc": SimpleNamespace(StatusCode=SimpleNamespace(NOT_FOUND=404, PERMISSION_DENIED=403)),
    }
    exec(compile(ast.Module(body=[guard, *methods], type_ignores=[]), str(source), "exec"), namespace)
    request = SimpleNamespace(identifier="GAME-1", value="GAME-1")
    for method in methods:
        with pytest.raises(PermissionError, match="active shared run"):
            asyncio.run(namespace[method.name](None, request, SimpleNamespace(abort=abort)))
    context = SimpleNamespace(abort=abort)
    run.status = "completed"
    assert asyncio.run(namespace["_require_private_game"](context, "GAME-1")) == {"gameId": "GAME-1"}
    shared_store = namespace["SAKKA"].store
    shared_store.get_run_for_game = lambda _: None
    assert asyncio.run(namespace["_require_private_game"](context, "GAME-1")) == {"gameId": "GAME-1"}
    run.status = "active"
    namespace["LEARNING_STORE"].get_game = lambda _: {
        "gameId": "GAME-1", "decisions": [{"decisionId": "DEC-1"}],
    }
    namespace["LEARNING_STORE"].get_decision = lambda _: {"decisionId": "DEC-1", "actor": "east"}
    shared_store.get_run_for_decision = lambda _: (run, b"")
    with pytest.raises(PermissionError, match="active shared run"):
        asyncio.run(namespace["_require_private_game"](context, "GAME-1"))


def _services(root: Path, tmp_path: Path):
    learning_store = LearningStore(tmp_path / "learning.sqlite3")
    registry = AgentProfileRegistry.from_yaml(root / "configs/agents/adaptive_lineup_v1.yaml")
    manager = GameManager(
        root, SQLiteGameStore(tmp_path / "games.sqlite3"),
        learning_store=learning_store, agent_registry=registry,
    )
    learning = LearningService(root, tmp_path / "learning", store=learning_store, game_manager=manager)
    manager.learning_service = learning
    audit = _NoAudit()
    service = SakkaService(
        root, SQLiteSakkaStore(tmp_path / "shared.sqlite3"),
        game_manager=manager, audit_service=audit,
    )
    return service, manager, learning_store, audit


def test_canonical_terminology_and_public_environment_redaction(root: Path, tmp_path: Path):
    service, _manager, _learning, _audit = _services(root, tmp_path)
    board = service.create_board_from_new_game(
        created_by="USR-A", seed=99121, dealer=Seat.EAST,
        observer_seat=Seat.SOUTH, scope="sakka", maximum_rounds=5,
    )
    assert SharedScope.parse(board.scope) is SharedScope.SAKKA
    stored = service.store.get_board(board.board_id)
    assert stored is not None
    session = decode(stored[1], root=root)
    environment = session.shared_environment
    assert environment is not None and verify_shared_environment(environment)["valid"]
    assert environment["terminology"] == {"matchAr": "صكّة", "roundAr": "لفة", "trickAr": "أكلة"}
    public = public_shared_environment(environment)
    assert public["scope"] == "sakka" and public["scopeLabelAr"] == "صكّة كاملة"
    assert all("seed" not in item and "cutPosition" not in item for item in public["dealSchedule"])
    assert public["redaction"]["futureCardsHidden"] is True


def test_all_pass_consumes_exact_next_precommitted_attempt(root: Path, tmp_path: Path):
    service, manager, _learning, _audit = _services(root, tmp_path)
    session = manager.create(seed=77881, dealer=Seat.NORTH, mode="full")
    board = service.create_board(session, created_by="USR-A", observer_seat=Seat.SOUTH,
                                 scope="sakka", maximum_rounds=4)
    stored = service.store.get_board(board.board_id)
    assert stored is not None
    shared = decode(stored[1], root=root)
    environment = shared.shared_environment
    assert environment is not None
    expected = copy.deepcopy(environment["dealSchedule"][1])
    frozen_manager = service._manager_for_session(shared)
    frozen_manager.import_session(shared)
    current = shared
    # Force every public auction actor to pass until the engine performs its
    # authoritative all-pass redeal.
    for _ in range(16):
        assert current.bidding is not None
        actor = current.bidding.actor
        assert actor is not None
        current = frozen_manager.bid(current.id, seat=actor, action="pass", expected_version=current.version)
        if current.deal_schedule_index == 1:
            break
    assert current.deal_schedule_index == 1
    assert current.state.seed == expected["seed"]
    assert current.state.dealer.name.lower() == expected["dealer"]
    assert current.state.shuffle is not None
    assert current.state.shuffle.deck_commitment == expected["deckCommitment"]


def test_same_action_path_has_same_frozen_agents_and_state(root: Path, tmp_path: Path):
    service, _manager, _learning, _audit = _services(root, tmp_path)
    board = service.create_board_from_new_game(
        created_by="USR-A", seed=812345, dealer=Seat.WEST,
        observer_seat=Seat.SOUTH, scope="laffa", maximum_rounds=1,
        agent_profiles={"north": "beginner", "west": "beginner", "south": "beginner", "east": "beginner"},
    )
    room = service.open_room(board.board_id, created_by="USR-A")
    first = service.start_run(room.room_code, subject_id="USR-A", display_name="الأول", is_guest=False)
    second = service.start_run(room.room_code, subject_id="USR-B", display_name="الثاني", is_guest=False)
    assert first["run"]["stateHash"] == second["run"]["stateHash"]
    assert first["state"]["sharedEnvironment"] == second["state"]["sharedEnvironment"]
    for _ in range(40):
        if first["run"]["status"] == "completed":
            break
        action = copy.deepcopy(first["legalActions"][0])
        first = service.apply_action(first["run"]["runId"], subject_id="USR-A", action=action)
        second = service.apply_action(second["run"]["runId"], subject_id="USR-B", action=action)
        assert first["run"]["stateHash"] == second["run"]["stateHash"]
        assert first["legalActions"] == second["legalActions"]
    assert first["run"]["result"] == second["run"]["result"]
    assert service.verify_run(first["run"]["runId"])["valid"]
    assert service.verify_run(second["run"]["runId"])["valid"]


def test_simple_arabic_explanation_is_hidden_in_challenge_and_attributed(root: Path, tmp_path: Path):
    service, _manager, store, audit = _services(root, tmp_path)
    board = service.create_board_from_new_game(
        created_by="USR-A", seed=71331, dealer=Seat.EAST,
        observer_seat=Seat.SOUTH, scope="laffa",
    )
    room = service.open_room(board.board_id, created_by="USR-A")
    payload = service.start_run(room.room_code, subject_id="USR-A", display_name="الأول", is_guest=False)
    payload = service.apply_action(
        payload["run"]["runId"], subject_id="USR-A", action=payload["legalActions"][0],
    )
    decisions = store.list_decisions(decode(service.store.get_run(payload["run"]["runId"])[1], root=root).id)
    assert decisions
    decision = decisions[-1]
    explanations = ExplanationService(store, audit, {"reasoning": {"maximum_reasons": 5}})
    hidden = explanations.explain(
        decision["decisionId"], observer="south", surface="challenge",
        run_completed=False, challenge_revealed=False,
    )
    assert hidden["available"] is False
    assert "بدون تلميحات" in hidden["summaryAr"]
    visible = explanations.explain(
        decision["decisionId"], observer="south", surface="replay",
        run_completed=True, challenge_revealed=True,
    )
    assert visible["available"] is True
    assert visible["titleAr"].startswith("ليش")
    assert visible["attributionAr"] in {"أنت", "خويك", "الخصم اليمين", "الخصم اليسار", "اللاعب"}
    assert "الإشارات المحتملة لا تُعامل كحقيقة" in visible["caveatAr"]


def test_active_opponent_explanation_redacts_private_analysis():
    decision = {
        "decisionId": "DEC-PRIVATE",
        "decisionKind": "play",
        "actor": "east",
        "selectedAction": "7S",
        "legalActions": ["7S", "AS"],
        "feedback": [{"rationale": "private rationale", "confidence": .9}],
    }
    audit = {"bestAction": "AS", "expectedRegret": 11, "auditConfidence": .9}

    hidden = explain_decision_ar(
        decision, audit=audit, observer="south", level="expert", surface="replay",
        run_completed=False, challenge_revealed=False, view_mode="oracle_after_completion",
    ).to_dict()
    assert hidden["bestAction"] is None
    assert not {"LEGAL_CHOICE", "FORCED_MOVE", "BETTER_ALTERNATIVE", "HUMAN_RECORDED_REASON"} & {
        reason["code"] for reason in hidden["reasons"]
    }

    own = explain_decision_ar(
        decision, audit=audit, observer="east", level="expert", surface="replay",
        run_completed=False, challenge_revealed=False, view_mode="observer",
    ).to_dict()
    assert own["bestAction"] == "AS"
    assert "LEGAL_CHOICE" in {reason["code"] for reason in own["reasons"]}


def test_structured_feedback_requires_independent_consensus(root: Path, tmp_path: Path):
    service, _manager, store, audit = _services(root, tmp_path)
    board = service.create_board_from_new_game(
        created_by="USR-A", seed=55441, dealer=Seat.NORTH,
        observer_seat=Seat.SOUTH, scope="laffa",
    )
    room = service.open_room(board.board_id, created_by="USR-A")
    payload = service.start_run(room.room_code, subject_id="USR-A", display_name="الأول", is_guest=False)
    payload = service.apply_action(payload["run"]["runId"], subject_id="USR-A", action=payload["legalActions"][0])
    actions = service.store.actions(payload["run"]["runId"])
    decision_id = next(item["decisionId"] for item in actions if item.get("decisionId"))
    decision = store.get_decision(decision_id)
    assert decision is not None
    preferred = decision["legalActions"][0]
    reviews = DecisionReviewService(store, audit)
    first = reviews.submit(
        decision_id, reviewer_id="USR-A", reviewer_role="player",
        preferred_action=preferred, confidence=.95, reason_codes=["card_counting"],
        rationale="عدّيت الورق وشفت أن هذا اللعب يحفظ الورقة القوية.", evidence=(), view_mode="observer",
    )
    assert first["consensus"]["trainingEligible"] is False
    for reviewer in ("USR-B", "USR-C"):
        value = reviews.submit(
            decision_id, reviewer_id=reviewer, reviewer_role="player",
            preferred_action=preferred, confidence=.9, reason_codes=["card_counting"],
            rationale="نفس العد يعطي هذا المسار نتيجة أفضل.", evidence=(), view_mode="observer",
        )
    assert value["consensus"]["trainingEligible"] is True
    assert value["consensus"]["independentReviewers"] == 3


def test_canonical_shared_api_and_decision_workbench(monkeypatch):
    from baloot_platform.service.api import app
    with TestClient(app) as client:
        guest = client.post("/api/v1/auth/guest", json={"displayName": "مراجع"})
        assert guest.status_code in {200, 201}
        board = client.post("/api/v1/shared/boards", json={
            "source": "new", "seed": 987611, "dealer": "east", "rulesProfile": "sun_only",
            "observerSeat": "south", "scope": "laffa", "maximumRounds": 1,
            "visibility": "unlisted", "label": "لفة مراجعة",
        })
        assert board.status_code == 201, board.text
        body = board.json()
        assert body["scope"] == "laffa" and body["scopeLabelAr"] == "لفة واحدة"
        assert body["sharedEnvironment"]["redaction"]["dealSeedsHidden"] is True
        room = client.post("/api/v1/shared/rooms", json={"boardIdentifier": body["boardId"]})
        assert room.status_code == 201, room.text
        code = room.json()["room"]["roomCode"]
        run = client.post(f"/api/v1/shared/rooms/{code}/runs")
        assert run.status_code == 201, run.text
        payload = run.json()
        if payload["run"]["status"] == "active":
            acted = client.post(
                f"/api/v1/shared/runs/{payload['run']['runId']}/actions",
                json=payload["legalActions"][0],
            )
            assert acted.status_code == 200, acted.text
            payload = acted.json()
        package = client.get(f"/api/v1/shared/runs/{payload['run']['runId']}/review")
        assert package.status_code == 200, package.text
        package_body = package.json()
        assert package_body["scopeLabelAr"] == "لفة واحدة"
        hidden = next(item for item in package_body["decisions"] if not item["reviewable"])
        assert hidden["legalActions"] == []
        assert hidden["audits"] == []
        assert hidden["reviewConsensus"] == {}
        assert hidden["simpleExplanation"]["bestAction"] is None
        assert not {"LEGAL_CHOICE", "FORCED_MOVE", "BETTER_ALTERNATIVE"} & {
            reason["code"] for reason in hidden["simpleExplanation"]["reasons"]
        }
        reviews = client.get(
            f"/api/v1/shared/runs/{payload['run']['runId']}/decisions/{hidden['decisionId']}/reviews",
        )
        assert reviews.status_code == 403
        actor = next(item for item in package_body["decisions"] if item["reviewable"])
        assert actor["legalActions"]
        actor_raw = client.get(f"/api/v1/history/decisions/{actor['decisionId']}")
        assert actor_raw.status_code == 200, actor_raw.text
        assert actor_raw.json()["legalActions"]

        hidden_raw = client.get(f"/api/v1/history/decisions/{hidden['decisionId']}")
        assert hidden_raw.status_code == 200, hidden_raw.text
        assert hidden_raw.json() == {
            "decisionId": hidden["decisionId"],
            "ordinal": hidden["ordinal"],
            "decisionKind": hidden["decisionKind"],
            "actor": hidden["actor"],
            "selectedAction": hidden["selectedAction"],
            "actorRelation": hidden["actorRelation"],
            "reviewable": False,
            "privateAnalysisRedacted": True,
        }
        for method, path, body_payload in (
            ("GET", f"/api/v1/history/decisions/{hidden['decisionId']}/audits", None),
            ("GET", f"/api/v1/history/decisions/{hidden['decisionId']}/comparison", None),
            ("GET", f"/api/v1/history/decisions/{hidden['decisionId']}/difficulty", None),
            ("POST", f"/api/v1/history/decisions/{hidden['decisionId']}:explain", {}),
            ("POST", f"/api/v1/history/decisions/{hidden['decisionId']}:reanalyze", {}),
        ):
            response = client.request(method, path, json=body_payload)
            assert response.status_code == 403, (path, response.text)
        active_audit = client.post(
            f"/api/v1/sakka/runs/{payload['run']['runId']}:audit",
            json={},
        )
        assert active_audit.status_code == 403, active_audit.text
        cached_audit = client.get(
            f"/api/v1/sakka/runs/{payload['run']['runId']}/audit",
        )
        assert cached_audit.status_code == 403, cached_audit.text

        from baloot_platform.service import mcp_server

        def local_api(method, path, body_payload=None, *, extra_headers=None):
            del extra_headers
            response = client.request(method, path, json=body_payload)
            if response.status_code >= 400:
                raise mcp_server.MCPError(
                    -32000,
                    "guarded API request failed",
                    {"status": response.status_code},
                )
            return response.json()

        monkeypatch.setenv("BALOOT_MCP_ALLOW_MUTATIONS", "true")
        monkeypatch.setattr(mcp_server, "_api", local_api)
        monkeypatch.setattr(mcp_server, "_require_mutations", lambda: None)
        with pytest.raises(mcp_server.MCPError) as mcp_denied:
            mcp_server._call_tool(
                "audit_sakka_run",
                {"run_id": payload["run"]["runId"]},
            )
        assert mcp_denied.value.data == {"status": 403}

        game_id = str(actor_raw.json()["gameId"])
        from baloot_platform.service import api
        active_session = decode(api.SAKKA.store.get_run(payload["run"]["runId"])[1], root=api.ROOT)
        archive = api.LEARNING.archive_session(active_session, status="manual_archive")
        aggregate_audit = api.LEARNING_STORE.put_game_audit({
            "gameId": game_id, "gameHash": archive["gameHash"],
            "decisions": [{"decisionId": hidden["decisionId"], "bestAction": "SA"}],
        })
        aggregate_paths = (
            f"/api/v1/history/games/{archive['gameHash']}",
            f"/api/v1/history/games/{archive['gameHash']}/replay",
            f"/api/v1/history/games/{archive['gameHash']}/audits",
            f"/api/v1/audits/{aggregate_audit['auditId']}",
        )
        for path in aggregate_paths:
            denied = client.get(path)
            assert denied.status_code == 403, (path, denied.text)
        game_rows = client.get(f"/api/v1/games/{game_id}/decisions")
        assert game_rows.status_code == 200, game_rows.text
        projected = next(
            item for item in game_rows.json()["decisions"]
            if item["decisionId"] == hidden["decisionId"]
        )
        assert projected["privateAnalysisRedacted"] is True

        for _ in range(40):
            if payload["run"]["status"] == "completed":
                break
            acted = client.post(
                f"/api/v1/shared/runs/{payload['run']['runId']}/actions",
                json=payload["legalActions"][0],
            )
            assert acted.status_code == 200, acted.text
            payload = acted.json()
        assert payload["run"]["status"] == "completed"
        completed_package = client.get(
            f"/api/v1/shared/runs/{payload['run']['runId']}/review",
        )
        assert completed_package.status_code == 200, completed_package.text
        completed_hidden = next(
            item for item in completed_package.json()["decisions"]
            if item["decisionId"] == hidden["decisionId"]
        )
        assert completed_hidden["reviewable"] is True
        assert completed_hidden["legalActions"]
        completed_raw = client.get(
            f"/api/v1/history/decisions/{hidden['decisionId']}",
        )
        assert completed_raw.status_code == 200, completed_raw.text
        assert completed_raw.json()["legalActions"]
        for path in aggregate_paths:
            allowed = client.get(path)
            assert allowed.status_code == 200, (path, allowed.text)
        workbench = client.get("/decision-workbench", follow_redirects=False)
        assert workbench.status_code == 307
        assert workbench.headers["location"] == "/foundry"
