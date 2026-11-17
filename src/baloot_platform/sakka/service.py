from __future__ import annotations

import hashlib
import json
import secrets
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

from baloot_platform.config import platform_root
from baloot_platform.core.cards import Card, iter_mask
from baloot_platform.core.enums import GamePhase, Seat
from baloot_platform.service.game import GameManager, GameSession
from baloot_platform.storage.codec_v7 import decode, encode
from baloot_platform.shared import (
    SharedScope, build_shared_environment, environment_hash, verify_shared_environment,
)
from baloot_platform.agents.profiles import AgentProfileRegistry

from .models import SakkaBoard, SakkaRoom, SakkaRun, digest, now
from .store import SakkaStore, action_record_hash
from .signing import BoardSigner

_CODE_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def _share_code(length: int = 10) -> str:
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(max(8, length)))


class SakkaService:
    """Authoritative same-start Baloot backend.

    Canonical terminology:
    * لفة / ``laffa`` is one deal/round.
    * صكّة / ``sakka`` is a complete match, normally to 152.

    The historical class name is retained as an API compatibility boundary,
    while new interfaces use the canonical shared لفة/صكّة terminology.
    Every published board freezes the deal-attempt stream, rules, scoring,
    complete bot tunings, strategy/model identities, seeds and reveal policy.
    """

    def __init__(self, root: str | Path | None, store: SakkaStore, *,
                 game_manager: GameManager, audit_service: Any | None = None,
                 default_maximum_rounds: int = 16,
                 default_room_players: int = 16,
                 board_signer: BoardSigner | None = None,
                 player_names: Mapping[str, str] | None = None,
                 clock: Callable[[], str] | None = None,
                 id_factory: Callable[[str], str] | None = None,
                 share_code_factory: Callable[[int], str] | None = None):
        self.root = platform_root(root)
        self.store = store
        self.games = game_manager
        self.audit_service = audit_service
        self.default_maximum_rounds = max(1, int(default_maximum_rounds))
        self.default_room_players = max(2, int(default_room_players))
        self.board_signer = board_signer
        self.player_names = {
            str(seat).lower(): str(name).strip()
            for seat, name in dict(player_names or {}).items()
            if str(name).strip()
        }
        self.clock = clock or now
        self.id_factory = id_factory or (lambda prefix: f"{prefix}-{uuid.uuid4().hex[:20].upper()}")
        self.share_code_factory = share_code_factory or _share_code

    @staticmethod
    def _state_hash(session: GameSession) -> str:
        payload = session.to_dict(reveal_hands=True)
        # A Sakka state identity must be independent of a participant's local
        # run/game identifier and persistence version. Only game semantics,
        # public history, hands, policies and rule bundle belong in the hash.
        for transient in ("gameId", "version", "parentGameHash", "gameHash", "serial"):
            payload.pop(transient, None)
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def legal_actions(session: GameSession) -> list[dict[str, Any]]:
        if session.bidding is not None and not session.bidding.finished:
            actions: list[dict[str, Any]] = []
            for action in session.bidding.legal_actions():
                if action == "hokum" and session.bidding.round_index == 2:
                    from baloot_platform.core.enums import Suit
                    actions.extend(
                        {"kind": "bid", "action": action, "trumpSuit": suit.name.lower()}
                        for suit in Suit if suit != session.bidding.purchase_suit
                    )
                else:
                    actions.append({"kind": "bid", "action": action})
            return actions
        if session.state.current_player is not None:
            return [{"kind": "play", "card": card.code} for card in session.engine.legal_moves(session.state)]
        return []

    def _manager_for_session(self, session: GameSession) -> GameManager:
        registry = self.games.agent_registry
        environment = dict(session.shared_environment or {})
        if environment:
            check = verify_shared_environment(environment)
            if not check["valid"]:
                raise ValueError(f"shared environment failed verification: {check['issues']}")
            registry = AgentProfileRegistry.from_snapshot(environment["agentSnapshot"])
        manager = GameManager(
            self.root, learning_store=self.games.learning_store, agent_registry=registry,
            bundle_provider=self.games.bundle,
        )
        manager.learning_service = self.games.learning_service
        return manager

    def create_board(self, session: GameSession, *, created_by: str,
                     observer_seat: Seat = Seat.SOUTH, scope: str = "round",
                     maximum_rounds: int | None = None, label: str | None = None,
                     visibility: str = "unlisted", bot_policy_hashes: Mapping[str, str] | None = None,
                     scoring_profile: str = "duplicate_skill_v1",
                     time_control: Mapping[str, Any] | None = None,
                     reveal_policy: str = "after_completion") -> SakkaBoard:
        resolved_scope = SharedScope.parse(scope)
        legacy_scope = resolved_scope.legacy
        if visibility not in {"private", "unlisted", "public"}:
            raise ValueError("visibility must be private, unlisted or public")
        max_rounds = 1 if resolved_scope is SharedScope.LAFFA else max(1, int(maximum_rounds or self.default_maximum_rounds))
        environment = build_shared_environment(
            self.root, session, self.games.agent_registry, scope=resolved_scope,
            maximum_rounds=max_rounds, lineup=session.agent_profiles,
            scoring_profile=scoring_profile, time_control=time_control,
            reveal_policy=reveal_policy,
            bundle=self.games.bundle(session.mode),
        )
        environment["observerSeat"] = observer_seat.name.lower()
        environment["environmentHash"] = environment_hash(environment)
        session.shared_environment = environment
        session.deal_schedule_index = 0
        payload = encode(session)
        payload_hash = hashlib.sha256(payload).hexdigest()
        public_prefix = hashlib.sha256(
            json.dumps([event.to_dict() for event in session.state.events], ensure_ascii=False,
                       sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        seat_tunings = dict(environment.get("agentSnapshot", {}).get("seats") or {})
        effective_bot_hashes = {
            seat: digest(dict(entry.get("tunings") or {}))
            for seat, entry in sorted(seat_tunings.items())
        }
        if bot_policy_hashes:
            # User-supplied identifiers are recorded only as additional labels;
            # comparison is always bound to the full frozen tuning hashes.
            for seat, value in bot_policy_hashes.items():
                effective_bot_hashes[f"label:{str(seat).lower()}"] = str(value)
        identity = {
            "schema": 3,
            "payloadHash": payload_hash,
            "bundleHash": session.bundle_hash,
            "observer": observer_seat.name.lower(),
            "scope": legacy_scope,
            "maximumRounds": max_rounds,
            "startingMatchScore": list(session.state.match_score),
            "startingRoundNumber": session.state.round_number,
            "eventPrefix": public_prefix,
            "bots": dict(sorted(effective_bot_hashes.items())),
            "scoring": scoring_profile,
            "time": dict(time_control or {"decisionSeconds": 45}),
            "reveal": reveal_policy,
            "environmentHash": environment["environmentHash"],
            "dealScheduleHash": environment["dealScheduleHash"],
            "agentSnapshotHash": environment["agentSnapshotHash"],
            "sharedProfileHash": environment["sharedProfileHash"],
            "terminologyVersion": environment["terminologyVersion"],
        }
        board_hash = digest(identity)
        board_id = f"SKB-{board_hash[:20].upper()}"
        board = SakkaBoard(
            board_id=board_id, version=2, board_hash=board_hash,
            serial=f"SKA-{board_hash[:12].upper()}", share_code=self.share_code_factory(10),
            bundle_hash=session.bundle_hash, observer_seat=observer_seat, scope=legacy_scope,
            maximum_rounds=max_rounds, starting_match_score=tuple(session.state.match_score),
            starting_round_number=session.state.round_number, public_event_prefix_hash=public_prefix,
            bot_policy_hashes=effective_bot_hashes,
            scoring_profile=scoring_profile, time_control=dict(time_control or {"decisionSeconds": 45}),
            reveal_policy=reveal_policy, trusted_payload_hash=payload_hash,
            created_by=created_by, visibility=visibility, label=label,
            created_at=self.clock(), environment_hash=environment["environmentHash"],
            deal_schedule_hash=environment["dealScheduleHash"],
            agent_snapshot_hash=environment["agentSnapshotHash"],
            shared_profile_hash=environment["sharedProfileHash"],
            terminology_version=environment["terminologyVersion"],
        )
        if self.board_signer is not None:
            board = replace(
                board, signature_algorithm=self.board_signer.algorithm,
                signature_key_id=self.board_signer.key_id,
                board_signature=self.board_signer.sign(board),
            )
        self.store.put_board(board, payload)
        return board

    def verify_board(self, identifier: str) -> dict[str, Any]:
        item = self.store.get_board(identifier)
        if item is None:
            raise KeyError(identifier)
        board, payload = item
        issues: list[dict[str, Any]] = []
        actual_payload_hash = hashlib.sha256(payload).hexdigest()
        if actual_payload_hash != board.trusted_payload_hash:
            issues.append({
                "code": "trusted_payload_hash_mismatch",
                "expected": board.trusted_payload_hash, "actual": actual_payload_hash,
            })
        calculated_board_hash = digest(board.identity_payload())
        if calculated_board_hash != board.board_hash:
            issues.append({
                "code": "board_hash_mismatch",
                "expected": board.board_hash, "actual": calculated_board_hash,
            })
        signature_valid: bool | None = None
        if board.board_signature:
            if self.board_signer is None:
                issues.append({"code": "board_signer_unavailable"})
                signature_valid = False
            else:
                signature_valid = self.board_signer.verify(board)
                if not signature_valid:
                    issues.append({"code": "board_signature_invalid"})
        elif self.board_signer is not None:
            issues.append({"code": "board_signature_missing"})
            signature_valid = False
        try:
            session = decode(payload, root=self.root, bundle_provider=self.games.bundle)
            environment = dict(session.shared_environment or {})
            environment_check = verify_shared_environment(environment) if environment else {"valid": False, "issues": [{"code": "shared_environment_missing"}]}
            if not environment_check["valid"]:
                issues.extend(environment_check["issues"])
            if board.environment_hash != environment.get("environmentHash"):
                issues.append({"code": "board_environment_hash_mismatch"})
            if board.deal_schedule_hash != environment.get("dealScheduleHash"):
                issues.append({"code": "board_deal_schedule_hash_mismatch"})
            if board.agent_snapshot_hash != environment.get("agentSnapshotHash"):
                issues.append({"code": "board_agent_snapshot_hash_mismatch"})
        except Exception as exc:
            environment_check = {"valid": False, "issues": [{"code": "shared_environment_decode_failed", "message": str(exc)}]}
            issues.extend(environment_check["issues"])
        return {
            "boardId": board.board_id, "serial": board.serial,
            "valid": not issues, "signatureValid": signature_valid,
            "boardHash": board.board_hash, "trustedPayloadHash": board.trusted_payload_hash,
            "environmentHash": board.environment_hash,
            "dealScheduleHash": board.deal_schedule_hash,
            "agentSnapshotHash": board.agent_snapshot_hash,
            "sharedEnvironment": environment_check,
            "signature": {
                "algorithm": board.signature_algorithm, "keyId": board.signature_key_id,
                "value": board.board_signature,
            } if board.board_signature else None,
            "issues": issues,
        }

    def create_board_from_new_game(self, *, created_by: str, seed: int, dealer: Seat,
                                   mode: str = "full", match_score: tuple[int, int] = (0, 0),
                                   observer_seat: Seat = Seat.SOUTH, scope: str = "round",
                                   maximum_rounds: int | None = None, label: str | None = None,
                                   visibility: str = "unlisted",
                                   agent_profiles: Mapping[str, str] | None = None) -> SakkaBoard:
        session = self.games.create(
            seed=seed, dealer=dealer, mode=mode, match_score=match_score,
            agent_profiles=agent_profiles,
        )
        session = self.games.advance_bots(session.id, stop_before=observer_seat)
        return self.create_board(
            session, created_by=created_by, observer_seat=observer_seat, scope=scope,
            maximum_rounds=maximum_rounds, label=label, visibility=visibility,
        )

    def create_board_from_game(self, game_id: str, *, created_by: str,
                               observer_seat: Seat | None = None, scope: str = "round",
                               maximum_rounds: int | None = None, label: str | None = None,
                               visibility: str = "unlisted") -> SakkaBoard:
        session = self.games.get(game_id)
        observer = observer_seat or session.state.current_player or Seat.SOUTH
        return self.create_board(
            session, created_by=created_by, observer_seat=observer, scope=scope,
            maximum_rounds=maximum_rounds, label=label, visibility=visibility,
        )

    def open_room(self, board_identifier: str, *, created_by: str, mode: str = "async",
                  max_players: int | None = None, allow_guests: bool = True,
                  settings: Mapping[str, Any] | None = None) -> SakkaRoom:
        item = self.store.get_board(board_identifier)
        if item is None:
            raise KeyError(board_identifier)
        verification = self.verify_board(item[0].board_id)
        if not verification["valid"]:
            raise ValueError("shared لفة/صكّة board failed authenticity verification")
        if mode not in {"async", "live", "practice"}:
            raise ValueError("room mode must be async, live or practice")
        room = SakkaRoom(
            self.id_factory("SKR"), item[0].board_id, self.share_code_factory(10),
            created_by, mode, "open", max(2, int(max_players or self.default_room_players)),
            bool(allow_guests), dict(settings or {}), created_at=self.clock(),
        )
        self.store.put_room(room)
        self.store.append_room_event(
            room.room_id, "room_created", {"room": room.to_dict()}, subject_id=created_by,
        )
        return room

    def join_room(self, identifier: str, *, subject_id: str, display_name: str,
                  is_guest: bool) -> dict[str, Any]:
        room = self.store.get_room(identifier)
        if room is None:
            raise KeyError(identifier)
        if room.status != "open":
            raise ValueError("room is not open")
        if is_guest and not room.allow_guests:
            raise PermissionError("this room requires sign in")
        self.store.join(room.room_id, subject_id, display_name)
        self.store.append_room_event(
            room.room_id, "member_joined",
            {"subjectId": subject_id, "displayName": display_name, "isGuest": bool(is_guest)},
            subject_id=subject_id,
        )
        board = self.store.get_board(room.board_id)
        assert board is not None
        return {"room": room.to_dict(), "board": board[0].to_dict(include_hashes=False), "members": self.store.members(room.room_id)}

    def room(self, identifier: str) -> dict[str, Any]:
        room = self.store.get_room(identifier)
        if room is None:
            raise KeyError(identifier)
        board = self.store.get_board(room.board_id)
        assert board is not None
        return {
            "room": room.to_dict(), "board": board[0].to_dict(include_hashes=False),
            "members": self.store.members(room.room_id),
            "leaderboard": self.store.leaderboard(board[0].board_id, board[0].comparison_key),
        }

    def start_run(self, room_identifier: str, *, subject_id: str, display_name: str,
                  is_guest: bool) -> dict[str, Any]:
        self.join_room(
            room_identifier, subject_id=subject_id,
            display_name=display_name, is_guest=is_guest,
        )
        room = self.store.get_room(room_identifier)
        assert room is not None
        existing = getattr(self.store, "get_run_for_subject", lambda *_: None)(room.room_id, subject_id)
        if existing is not None:
            return self.reopen_run(existing[0].run_id, subject_id=subject_id)
        board, payload = self.store.get_board(room.board_id) or (None, None)
        assert board is not None and payload is not None
        verification = self.verify_board(board.board_id)
        if not verification["valid"]:
            raise ValueError("shared لفة/صكّة board failed authenticity verification")
        session = decode(payload, root=self.root, bundle_provider=self.games.bundle)
        session.id = self.id_factory("SKG")
        session.version = 1
        manager = self._manager_for_session(session)
        manager.import_session(session)
        session = manager.advance_bots(session.id, stop_before=board.observer_seat)
        run = SakkaRun(
            self.id_factory("RUN"), room.room_id, board.board_id, subject_id,
            board.comparison_key, "active", session.version, 0,
            self._state_hash(session), started_at=self.clock(), game_id=session.id,
        )
        self.store.put_run(run, encode(session))
        self.store.append_room_event(
            room.room_id, "run_started", {"run": run.to_dict()}, subject_id=subject_id,
        )
        return self._run_payload(board, room, run, session, reveal=False)

    def _latest_player_decision(self, game_id: str, subject_id: str) -> str | None:
        store = self.games.learning_store
        if store is None:
            return None
        decisions = store.list_decisions(game_id, limit=10_000)
        for decision in reversed(decisions):
            metadata = decision.get("decisionMetadata") or {}
            if metadata.get("source") == "sakka_player" and metadata.get("subjectId") == subject_id:
                return str(decision["decisionId"])
        return None

    def _is_round_complete(self, session: GameSession) -> bool:
        return session.state.phase in {GamePhase.SCORED, GamePhase.COMPLETE} or (
            session.state.current_player is None and session.state.round_score is not None
        )

    def _run_payload(self, board: SakkaBoard, room: SakkaRoom, run: SakkaRun,
                     session: GameSession, *, reveal: bool) -> dict[str, Any]:
        state = session.to_dict(reveal_hands=reveal)
        members = self.store.members(room.room_id)
        player_names = dict(self.player_names)
        observer_name = next(
            (
                str(item.get("display_name") or item.get("displayName") or "").strip()
                for item in members
                if str(item.get("subject_id") or item.get("subjectId") or "") == run.subject_id
            ),
            "",
        )
        if observer_name:
            player_names[board.observer_seat.name.lower()] = observer_name
        if not reveal:
            state["hands"][board.observer_seat.name.lower()] = [
                card.to_dict() for card in iter_mask(session.state.hands[int(board.observer_seat)])
            ]
            state["viewerSeat"] = board.observer_seat.name.lower()
        return {
            "board": board.to_dict(include_hashes=run.status == "completed"),
            "room": room.to_dict(),
            "run": run.to_dict(),
            "state": state,
            "legalActions": self.legal_actions(session) if run.status == "active" else [],
            "actions": self.store.actions(run.run_id),
            "members": members,
            "playerNames": player_names,
            "leaderboard": self.store.leaderboard(board.board_id, board.comparison_key),
            "shareUrl": f"/play?code={room.room_code}",
        }

    def apply_action(self, run_id: str, *, subject_id: str, action: Mapping[str, Any]) -> dict[str, Any]:
        loaded = self.store.get_run(run_id)
        if loaded is None:
            raise KeyError(run_id)
        run, payload = loaded
        if run.subject_id != subject_id:
            raise PermissionError("run belongs to another player")
        if run.status != "active":
            raise ValueError("shared run is not active")
        board_item = self.store.get_board(run.board_id)
        room = self.store.get_room(run.room_id)
        assert board_item is not None and room is not None
        board = board_item[0]
        verification = self.verify_board(board.board_id)
        if not verification["valid"]:
            raise ValueError("shared لفة/صكّة board failed authenticity verification")
        session = decode(payload, root=self.root, bundle_provider=self.games.bundle)
        pre_payload = encode(session)
        legal_actions = self.legal_actions(session)
        if not any(
            item.get("kind") == action.get("kind")
            and item.get("action") == action.get("action")
            and item.get("card") == action.get("card")
            and item.get("trumpSuit") == action.get("trumpSuit")
            for item in legal_actions
        ):
            raise ValueError("action is not legal in this shared لفة/صكّة state")
        manager = self._manager_for_session(session)
        manager.import_session(session)
        observer = board.observer_seat
        kind = str(action.get("kind", "play"))
        if kind == "bid":
            from baloot_platform.core.enums import Suit
            trump = Suit.parse(action["trumpSuit"]) if action.get("trumpSuit") else None
            session = manager.bid(
                session.id, seat=observer, action=str(action["action"]), trump=trump,
                expected_version=session.version,
                decision_metadata={"source": "sakka_player", "subjectId": subject_id, "boardId": board.board_id},
            )
        elif kind == "play":
            session = manager.play(
                session.id, seat=observer, card=Card.parse(action["card"]),
                ekka=bool(action.get("ekka", False)), expected_version=session.version,
                decision_metadata={"source": "sakka_player", "subjectId": subject_id, "boardId": board.board_id},
            )
        else:
            raise ValueError("action kind must be bid or play")
        session = manager.advance_bots(session.id, stop_before=observer)
        decision_id = self._latest_player_decision(session.id, subject_id)
        rounds_completed = run.rounds_completed
        status = "active"
        completed_at = None
        while self._is_round_complete(session):
            rounds_completed += 1
            match_done = session.state.match_winner is not None
            if board.scope == "round" or match_done or rounds_completed >= board.maximum_rounds:
                status = "completed"
                completed_at = self.clock()
                break
            session = manager.next_round(session.id)
            session = manager.advance_bots(session.id, stop_before=observer)
            if not self._is_round_complete(session):
                break
        state_hash = self._state_hash(session)
        sequence = len(self.store.actions(run_id))
        self.store.add_action(
            run_id, sequence, dict(action), state_hash, pre_payload, legal_actions,
            decision_id=decision_id, created_at=self.clock(),
        )
        result = None
        if status == "completed":
            score = tuple(session.state.match_score)
            team_index = int(observer.team)
            opponent_index = int(observer.team.other)
            last_round = session.state.round_score
            result = {
                "scope": board.scope,
                "roundsCompleted": rounds_completed,
                "matchScore": list(score),
                "matchWon": session.state.match_winner == observer.team if session.state.match_winner else score[team_index] > score[opponent_index],
                "ourMatchPoints": score[team_index],
                "opponentMatchPoints": score[opponent_index],
                "totalRoundMargin": score[team_index] - score[opponent_index],
                "lastRoundMargin": (
                    last_round.awarded_points[team_index] - last_round.awarded_points[opponent_index]
                    if last_round is not None else None
                ),
                "contractSucceeded": (
                    last_round.winner == session.state.buyer.team
                    if last_round is not None and session.state.buyer is not None else None
                ),
                "decisionRegret": None,
                "decisionQualityStatus": "available_for_post_run_audit",
            }
        updated = replace(
            run, status=status, version=session.version, rounds_completed=rounds_completed,
            state_hash=state_hash, result=result, completed_at=completed_at,
        )
        self.store.put_run(updated, encode(session))
        self.store.append_room_event(
            room.room_id,
            "run_completed" if status == "completed" else "run_action",
            {
                "runId": run_id, "sequence": sequence, "action": dict(action),
                "stateHash": state_hash, "decisionId": decision_id,
                "status": status, "result": result,
            },
            subject_id=subject_id,
        )
        reveal = status == "completed" or board.reveal_policy == "immediate"
        return self._run_payload(board, room, updated, session, reveal=reveal)

    def _verify_apply_action(self, session: GameSession, board: SakkaBoard, action: Mapping[str, Any],
                             rounds_completed: int) -> tuple[GameSession, int]:
        """Replay one stored player action without writing learning/history side effects."""
        manager = self._manager_for_session(session)
        manager.import_session(session)
        observer = board.observer_seat
        kind = str(action.get("kind", "play"))
        if kind == "bid":
            from baloot_platform.core.enums import Suit
            trump = Suit.parse(action["trumpSuit"]) if action.get("trumpSuit") else None
            session = manager.bid(
                session.id, seat=observer, action=str(action["action"]), trump=trump,
                expected_version=session.version,
            )
        elif kind == "play":
            session = manager.play(
                session.id, seat=observer, card=Card.parse(action["card"]),
                ekka=bool(action.get("ekka", False)), expected_version=session.version,
            )
        else:
            raise ValueError("action kind must be bid or play")
        session = manager.advance_bots(session.id, stop_before=observer)
        while self._is_round_complete(session):
            rounds_completed += 1
            if (board.scope == "round" or session.state.match_winner is not None
                    or rounds_completed >= board.maximum_rounds):
                break
            session = manager.next_round(session.id)
            session = manager.advance_bots(session.id, stop_before=observer)
            if not self._is_round_complete(session):
                break
        return session, rounds_completed

    def verify_run(self, run_id: str, *, subject_id: str | None = None) -> dict[str, Any]:
        """Verify the complete Sakka action chain and deterministic replay."""
        loaded = self.store.get_run(run_id)
        if loaded is None:
            raise KeyError(run_id)
        run, _payload = loaded
        if subject_id is not None and run.subject_id != subject_id:
            raise PermissionError("run belongs to another player")
        board_item = self.store.get_board(run.board_id)
        if board_item is None:
            raise RuntimeError("run references a missing board")
        board, board_payload = board_item
        session = decode(board_payload, root=self.root, bundle_provider=self.games.bundle)
        manager = self._manager_for_session(session)
        manager.import_session(session)
        session = manager.advance_bots(session.id, stop_before=board.observer_seat)
        expected_pre_state_hash = self._state_hash(session)
        previous_action_hash = ""
        rounds_completed = 0
        issues: list[dict[str, Any]] = []
        actions = self.store.actions(run_id, include_private=True)
        for expected_sequence, item in enumerate(actions):
            sequence = int(item["sequence"])
            if sequence != expected_sequence:
                issues.append({"sequence": sequence, "code": "non_contiguous_sequence", "expected": expected_sequence})
            pre_payload = item["preStatePayload"]
            pre_hash = hashlib.sha256(pre_payload).hexdigest()
            if pre_hash != item.get("preStatePayloadHash"):
                issues.append({"sequence": sequence, "code": "pre_state_payload_hash_mismatch"})
            if item.get("previousActionHash", "") != previous_action_hash:
                issues.append({"sequence": sequence, "code": "previous_action_hash_mismatch"})
            calculated_action_hash = action_record_hash(
                run_id=run_id, sequence=sequence, action=item["action"], state_hash=item["stateHash"],
                pre_state_payload_hash=pre_hash, legal_actions=item["legalActions"],
                decision_id=item.get("decisionId"), previous_action_hash=previous_action_hash,
            )
            if calculated_action_hash != item.get("actionHash"):
                issues.append({"sequence": sequence, "code": "action_hash_mismatch"})
            try:
                recorded_pre = decode(pre_payload, root=self.root, bundle_provider=self.games.bundle)
                recorded_pre_hash = self._state_hash(recorded_pre)
                if recorded_pre_hash != expected_pre_state_hash:
                    issues.append({
                        "sequence": sequence, "code": "pre_state_chain_mismatch",
                        "expected": expected_pre_state_hash, "actual": recorded_pre_hash,
                    })
                actual_legal = self.legal_actions(recorded_pre)
                if actual_legal != item["legalActions"]:
                    issues.append({"sequence": sequence, "code": "legal_actions_mismatch"})
                recorded_pre, rounds_completed = self._verify_apply_action(
                    recorded_pre, board, item["action"], rounds_completed,
                )
                calculated_state_hash = self._state_hash(recorded_pre)
                if calculated_state_hash != item["stateHash"]:
                    issues.append({
                        "sequence": sequence, "code": "post_state_hash_mismatch",
                        "expected": item["stateHash"], "actual": calculated_state_hash,
                    })
                expected_pre_state_hash = calculated_state_hash
                session = recorded_pre
            except Exception as exc:
                issues.append({"sequence": sequence, "code": "replay_error", "detail": str(exc)})
                break
            previous_action_hash = str(item.get("actionHash") or "")
        if run.state_hash != expected_pre_state_hash:
            issues.append({
                "code": "run_head_mismatch", "expected": run.state_hash,
                "actual": expected_pre_state_hash,
            })
        return {
            "runId": run_id, "boardId": board.board_id, "valid": not issues,
            "actionCount": len(actions), "roundsReplayed": rounds_completed,
            "headActionHash": previous_action_hash or None,
            "finalStateHash": expected_pre_state_hash, "issues": issues,
        }

    def reopen_run(self, run_id: str, *, subject_id: str) -> dict[str, Any]:
        loaded = self.store.get_run(run_id)
        if loaded is None:
            raise KeyError(run_id)
        run, payload = loaded
        if run.subject_id != subject_id:
            raise PermissionError("run belongs to another player")
        board_item = self.store.get_board(run.board_id)
        room = self.store.get_room(run.room_id)
        assert board_item is not None and room is not None
        board = board_item[0]
        verification = self.verify_board(board.board_id)
        if not verification["valid"]:
            raise ValueError("shared لفة/صكّة board failed authenticity verification")
        session = decode(payload, root=self.root, bundle_provider=self.games.bundle)
        reveal = run.status == "completed" or board.reveal_policy == "immediate"
        return self._run_payload(board, room, run, session, reveal=reveal)

    def leaderboard(self, room_identifier: str) -> dict[str, Any]:
        room = self.store.get_room(room_identifier)
        if room is None:
            raise KeyError(room_identifier)
        board = self.store.get_board(room.board_id)
        assert board is not None
        return {
            "room": room.to_dict(),
            "board": board[0].to_dict(include_hashes=False),
            "leaderboard": self.store.leaderboard(board[0].board_id, board[0].comparison_key),
        }

    def compare_runs(
        self,
        run_a_id: str,
        run_b_id: str,
        *,
        requester_subject_id: str | None = None,
    ) -> dict[str, Any]:
        """Compare two completed runs that began from the same immutable Sakka.

        The comparison is intentionally derived from authoritative run records,
        not from client-supplied scores.  Both action chains and the originating
        signed board are verified before a winner or divergence is reported.
        """
        if run_a_id == run_b_id:
            raise ValueError("two different runs are required")
        loaded_a = self.store.get_run(run_a_id)
        loaded_b = self.store.get_run(run_b_id)
        if loaded_a is None:
            raise KeyError(run_a_id)
        if loaded_b is None:
            raise KeyError(run_b_id)
        run_a, _ = loaded_a
        run_b, _ = loaded_b
        if run_a.board_id != run_b.board_id or run_a.comparison_key != run_b.comparison_key:
            raise ValueError("runs are not comparable because their Sakka identities differ")
        if run_a.status != "completed" or run_b.status != "completed":
            raise ValueError("both runs must be completed before comparison")

        room_a = self.store.get_room(run_a.room_id)
        room_b = self.store.get_room(run_b.room_id)
        if room_a is None or room_b is None:
            raise RuntimeError("run references a missing room")
        if requester_subject_id is not None:
            allowed = {
                str(item["subject_id"] if "subject_id" in item else item.get("subjectId"))
                for item in self.store.members(room_a.room_id) + self.store.members(room_b.room_id)
            }
            if requester_subject_id not in allowed:
                raise PermissionError("comparison is available only to room members")

        board_item = self.store.get_board(run_a.board_id)
        if board_item is None:
            raise RuntimeError("run references a missing board")
        board = board_item[0]
        board_verification = self.verify_board(board.board_id)
        verification_a = self.verify_run(run_a_id)
        verification_b = self.verify_run(run_b_id)
        if not board_verification["valid"] or not verification_a["valid"] or not verification_b["valid"]:
            raise ValueError("one or more Sakka records failed authenticity verification")

        actions_a = self.store.actions(run_a_id)
        actions_b = self.store.actions(run_b_id)
        divergences: list[dict[str, Any]] = []
        for sequence in range(max(len(actions_a), len(actions_b))):
            left = actions_a[sequence] if sequence < len(actions_a) else None
            right = actions_b[sequence] if sequence < len(actions_b) else None
            if left is not None and right is not None and left.get("action") == right.get("action"):
                continue
            divergences.append({
                "sequence": sequence,
                "runA": None if left is None else {
                    "action": left.get("action"),
                    "decisionId": left.get("decisionId"),
                    "preStatePayloadHash": left.get("preStatePayloadHash"),
                    "stateHash": left.get("stateHash"),
                },
                "runB": None if right is None else {
                    "action": right.get("action"),
                    "decisionId": right.get("decisionId"),
                    "preStatePayloadHash": right.get("preStatePayloadHash"),
                    "stateHash": right.get("stateHash"),
                },
            })

        result_a = dict(run_a.result or {})
        result_b = dict(run_b.result or {})

        def rank_key(result: Mapping[str, Any]) -> tuple[int, float, float, float]:
            return (
                int(bool(result.get("matchWon"))),
                float(result.get("ourMatchPoints", 0)) - float(result.get("opponentMatchPoints", 0)),
                float(result.get("ourMatchPoints", 0)),
                float(result.get("totalRoundMargin", result.get("lastRoundMargin", -9999))),
            )

        key_a = rank_key(result_a)
        key_b = rank_key(result_b)
        winner_run_id = run_a_id if key_a > key_b else run_b_id if key_b > key_a else None
        comparison_payload = {
            "schemaVersion": 2,
            "boardId": board.board_id,
            "boardSerial": board.serial,
            "boardHash": board.board_hash,
            "comparisonKey": board.comparison_key,
            "sharedScope": "laffa" if board.scope in {"laffa", "round"} else "sakka",
            "scopeLabelAr": "لفة" if board.scope in {"laffa", "round"} else "صكّة",
            "targetMatchPoints": int(self.games.bundle("full").rules.data.get("scoring", {}).get("match", {}).get("target", 152)),
            "environmentHash": board.environment_hash,
            "dealScheduleHash": board.deal_schedule_hash,
            "agentSnapshotHash": board.agent_snapshot_hash,
            "sharedProfileHash": board.shared_profile_hash,
            "modelSnapshotHash": None,
            "runA": {"runId": run_a_id, "subjectId": run_a.subject_id, "result": result_a},
            "runB": {"runId": run_b_id, "subjectId": run_b.subject_id, "result": result_b},
            "winnerRunId": winner_run_id,
            "firstDivergence": divergences[0] if divergences else None,
            "divergences": divergences,
            "sameStartVerified": True,
            "verification": {
                "board": board_verification,
                "runA": verification_a,
                "runB": verification_b,
            },
        }
        comparison_payload["comparisonHash"] = digest(comparison_payload)
        return comparison_payload

    def events(self, room_identifier: str, *, after_sequence: int = 0,
               limit: int = 200) -> dict[str, Any]:
        room = self.store.get_room(room_identifier)
        if room is None:
            raise KeyError(room_identifier)
        return {
            "roomId": room.room_id,
            "events": self.store.room_events(
                room.room_id, after_sequence=after_sequence, limit=limit,
            ),
        }

    def create_board_from_run_action(
        self, run_id: str, sequence: int, *, created_by: str,
        scope: str = "round", maximum_rounds: int | None = None,
        label: str | None = None, visibility: str = "unlisted",
    ) -> SakkaBoard:
        loaded = self.store.get_run(run_id)
        if loaded is None:
            raise KeyError(run_id)
        run, _payload = loaded
        if run.subject_id != created_by:
            raise PermissionError("run belongs to another player")
        actions = self.store.actions(run_id, include_private=True)
        item = next((value for value in actions if int(value["sequence"]) == int(sequence)), None)
        if item is None:
            raise KeyError(f"{run_id}@{sequence}")
        session = decode(
            item["preStatePayload"], root=self.root, bundle_provider=self.games.bundle,
        )
        board_item = self.store.get_board(run.board_id)
        assert board_item is not None
        source = board_item[0]
        return self.create_board(
            session, created_by=created_by, observer_seat=source.observer_seat,
            scope=scope, maximum_rounds=maximum_rounds, label=label,
            visibility=visibility, bot_policy_hashes=source.bot_policy_hashes,
            scoring_profile=source.scoring_profile, time_control=source.time_control,
            reveal_policy=source.reveal_policy,
        )

    def audit_run(self, run_id: str, *, subject_id: str, deep: bool = False,
                  force: bool = False, maximum_decisions: int = 32) -> dict[str, Any]:
        loaded = self.store.get_run(run_id)
        if loaded is None:
            raise KeyError(run_id)
        run, payload = loaded
        if run.subject_id != subject_id:
            raise PermissionError("run belongs to another player")
        if run.status != "completed":
            raise PermissionError("run analysis is available after completion only")
        existing = self.store.get_run_audit(run_id)
        if existing is not None and not force:
            return existing
        if self.audit_service is None:
            raise RuntimeError("replay audit service is unavailable")
        action_rows = self.store.actions(run_id)
        decision_sequences = {
            str(item["decisionId"]): int(item["sequence"])
            for item in action_rows if item.get("decisionId")
        }
        decision_ids = list(decision_sequences)[: max(1, int(maximum_decisions))]
        audits: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        for decision_id in decision_ids:
            try:
                audit = self.audit_service.audit_decision(
                    decision_id, deep=deep, force=force,
                )
                audits.append({**audit, "sakkaSequence": decision_sequences[decision_id]})
            except Exception as exc:  # an unavailable deep evaluator must not corrupt the run
                failures.append({"decisionId": decision_id, "error": str(exc)})
        total_regret = sum(float(item.get("expectedRegret", 0.0)) for item in audits)
        total_win_drop = sum(float(item.get("winProbabilityDrop", 0.0)) for item in audits)
        non_forced = [item for item in audits if not bool(item.get("forced"))]
        quality = (
            sum(float(item.get("qualityScore", 0.0)) for item in non_forced) / len(non_forced)
            if non_forced else 100.0
        )
        report = {
            "schemaVersion": 1,
            "runId": run_id,
            "boardId": run.board_id,
            "subjectId": subject_id,
            "deep": bool(deep),
            "decisionCount": len(audits),
            "failedDecisionCount": len(failures),
            "totalExpectedRegret": total_regret,
            "totalWinProbabilityDrop": total_win_drop,
            "averageDecisionQuality": quality,
            "catastrophicCount": sum(bool(item.get("catastrophicCrossing")) for item in audits),
            "criticalDecisionIds": [
                item["decisionId"] for item in sorted(
                    non_forced,
                    key=lambda value: (
                        bool(value.get("catastrophicCrossing")),
                        float(value.get("winProbabilityDrop", 0.0)),
                        float(value.get("expectedRegret", 0.0)),
                    ),
                    reverse=True,
                )[:8]
            ],
            "decisions": audits,
            "failures": failures,
        }
        saved = self.store.put_run_audit(run_id, report)
        result = dict(run.result or {})
        result.update({
            "decisionRegret": total_regret,
            "decisionQuality": quality,
            "decisionQualityStatus": "audited",
            "auditHash": saved["auditHash"],
        })
        updated = replace(run, result=result, analysis_status="complete")
        self.store.put_run(updated, payload)
        self.store.append_room_event(
            run.room_id, "run_audited",
            {"runId": run_id, "auditHash": saved["auditHash"],
             "averageDecisionQuality": quality, "totalExpectedRegret": total_regret},
            subject_id=subject_id,
        )
        return saved

    def get_run_audit(self, run_id: str, *, subject_id: str) -> dict[str, Any] | None:
        """Read a completed owner's audit without bypassing the analysis gate."""

        loaded = self.store.get_run(run_id)
        if loaded is None:
            raise KeyError(run_id)
        run, _payload = loaded
        if run.subject_id != subject_id:
            raise PermissionError("run belongs to another player")
        if run.status != "completed":
            raise PermissionError("run analysis is available after completion only")
        return self.store.get_run_audit(run_id)

    def feedback(self, run_id: str, sequence: int, *, subject_id: str,
                 preferred_action: Any, confidence: float, rationale: str,
                 interestingness: float = 0.0) -> dict[str, Any]:
        loaded = self.store.get_run(run_id)
        if loaded is None:
            raise KeyError(run_id)
        run, _payload = loaded
        if run.subject_id != subject_id:
            raise PermissionError("run belongs to another player")
        row = next(
            (item for item in self.store.actions(run_id) if int(item["sequence"]) == int(sequence)),
            None,
        )
        if row is None or not row.get("decisionId"):
            raise KeyError(f"{run_id}@{sequence}")
        learning = self.games.learning_store
        if learning is None:
            raise RuntimeError("learning store is unavailable")
        audit = self.store.get_run_audit(run_id)
        engine_action = None
        if audit is not None:
            decision = next(
                (item for item in audit.get("decisions", []) if item.get("decisionId") == row["decisionId"]),
                None,
            )
            if decision is not None:
                engine_action = decision.get("bestAction")
        return learning.add_feedback(
            str(row["decisionId"]), preferred_action=preferred_action,
            engine_action=engine_action, confidence=confidence,
            rationale=rationale, source="sakka_player_review", reviewer=subject_id,
            interestingness=interestingness,
            model_error=1.0 if engine_action is not None and preferred_action != engine_action else 0.0,
        )
