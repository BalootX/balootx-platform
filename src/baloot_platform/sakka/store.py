from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from baloot_platform.core.enums import Seat
from .models import SakkaBoard, SakkaRoom, SakkaRun


def action_record_hash(*, run_id: str, sequence: int, action: dict[str, Any], state_hash: str,
                       pre_state_payload_hash: str, legal_actions: list[dict[str, Any]],
                       decision_id: str | None, previous_action_hash: str) -> str:
    payload = {
        "runId": run_id, "sequence": int(sequence), "action": action,
        "stateHash": state_hash, "preStatePayloadHash": pre_state_payload_hash,
        "legalActions": legal_actions,
        "decisionId": decision_id, "previousActionHash": previous_action_hash,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _leaderboard_key(item: dict[str, Any]) -> tuple[int, int, float, float]:
    """Return the observer-relative comparison key used by every backend."""
    result = item["result"]
    our_points = result.get("ourMatchPoints")
    if our_points is None:
        observer_team = int(result.get("observerTeam", 0))
        scores = result.get("matchScore", [0, 0])
        our_points = scores[observer_team] if len(scores) > observer_team else 0
    return (
        int(bool(result.get("matchWon"))),
        int(our_points),
        float(result.get("totalRoundMargin", result.get("roundMargin", -9999))),
        -float(result.get("decisionRegret", 0.0) or 0.0),
    )


class SakkaStore(Protocol):
    def put_board(self, board: SakkaBoard, trusted_payload: bytes) -> None: ...
    def get_board(self, identifier: str) -> tuple[SakkaBoard, bytes] | None: ...
    def list_boards(self, *, subject_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]: ...
    def put_room(self, room: SakkaRoom) -> None: ...
    def get_room(self, identifier: str) -> SakkaRoom | None: ...
    def join(self, room_id: str, subject_id: str, display_name: str) -> None: ...
    def members(self, room_id: str) -> list[dict[str, Any]]: ...
    def put_run(self, run: SakkaRun, payload: bytes) -> None: ...
    def get_run(self, run_id: str) -> tuple[SakkaRun, bytes] | None: ...
    def get_run_for_decision(self, decision_id: str) -> tuple[SakkaRun, bytes] | None: ...
    def get_run_for_game(self, game_id: str) -> tuple[SakkaRun, bytes] | None: ...
    def add_action(self, run_id: str, sequence: int, action: dict[str, Any], state_hash: str,
                   pre_state_payload: bytes, legal_actions: list[dict[str, Any]],
                   decision_id: str | None = None, created_at: str | None = None) -> None: ...
    def actions(self, run_id: str, *, include_private: bool = False) -> list[dict[str, Any]]: ...
    def leaderboard(self, board_id: str, comparison_key: str) -> list[dict[str, Any]]: ...
    def append_room_event(self, room_id: str, event_type: str, payload: dict[str, Any],
                          *, subject_id: str | None = None) -> dict[str, Any]: ...
    def room_events(self, room_id: str, *, after_sequence: int = 0,
                    limit: int = 200) -> list[dict[str, Any]]: ...
    def put_run_audit(self, run_id: str, audit: dict[str, Any]) -> dict[str, Any]: ...
    def get_run_audit(self, run_id: str) -> dict[str, Any] | None: ...


class SQLiteSakkaStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS sakka_boards_v8(
                  board_id TEXT PRIMARY KEY,version INTEGER NOT NULL,board_hash TEXT NOT NULL UNIQUE,
                  serial TEXT NOT NULL UNIQUE,share_code TEXT NOT NULL UNIQUE,bundle_hash TEXT NOT NULL,
                  observer_seat TEXT NOT NULL,scope TEXT NOT NULL CHECK(scope IN ('round','match')),
                  maximum_rounds INTEGER NOT NULL,starting_match_score TEXT NOT NULL,
                  starting_round_number INTEGER NOT NULL,public_event_prefix_hash TEXT NOT NULL,
                  bot_policy_hashes TEXT NOT NULL,scoring_profile TEXT NOT NULL,time_control TEXT NOT NULL,
                  reveal_policy TEXT NOT NULL,trusted_payload BLOB NOT NULL,trusted_payload_hash TEXT NOT NULL,
                  created_by TEXT NOT NULL,visibility TEXT NOT NULL,status TEXT NOT NULL,label TEXT,created_at TEXT NOT NULL,
                  signature_algorithm TEXT,signature_key_id TEXT,board_signature TEXT
                );
                CREATE TABLE IF NOT EXISTS sakka_rooms_v8(
                  room_id TEXT PRIMARY KEY,board_id TEXT NOT NULL REFERENCES sakka_boards_v8(board_id),
                  room_code TEXT NOT NULL UNIQUE,created_by TEXT NOT NULL,mode TEXT NOT NULL,status TEXT NOT NULL,
                  max_players INTEGER NOT NULL,allow_guests INTEGER NOT NULL,settings_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,expires_at TEXT
                );
                CREATE TABLE IF NOT EXISTS sakka_members_v8(
                  room_id TEXT NOT NULL REFERENCES sakka_rooms_v8(room_id) ON DELETE CASCADE,
                  subject_id TEXT NOT NULL,display_name TEXT NOT NULL,joined_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  PRIMARY KEY(room_id,subject_id)
                );
                CREATE TABLE IF NOT EXISTS sakka_runs_v8(
                  run_id TEXT PRIMARY KEY,room_id TEXT NOT NULL REFERENCES sakka_rooms_v8(room_id),
                  board_id TEXT NOT NULL REFERENCES sakka_boards_v8(board_id),subject_id TEXT NOT NULL,
                  comparison_key TEXT NOT NULL,status TEXT NOT NULL,version INTEGER NOT NULL,
                  rounds_completed INTEGER NOT NULL,session_payload BLOB NOT NULL,state_hash TEXT,
                  result_json TEXT,analysis_status TEXT NOT NULL,started_at TEXT NOT NULL,completed_at TEXT,
                  game_id TEXT,
                  UNIQUE(room_id,subject_id)
                );
                CREATE TABLE IF NOT EXISTS sakka_run_actions_v8(
                  run_id TEXT NOT NULL REFERENCES sakka_runs_v8(run_id) ON DELETE CASCADE,
                  sequence INTEGER NOT NULL,action_json TEXT NOT NULL,state_hash TEXT NOT NULL,
                  pre_state_payload BLOB NOT NULL,legal_actions_json TEXT NOT NULL,
                  decision_id TEXT,pre_state_payload_hash TEXT NOT NULL DEFAULT '',previous_action_hash TEXT NOT NULL DEFAULT '',action_hash TEXT NOT NULL DEFAULT '',
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,PRIMARY KEY(run_id,sequence)
                );
                CREATE TABLE IF NOT EXISTS sakka_room_events_v8(
                  room_id TEXT NOT NULL REFERENCES sakka_rooms_v8(room_id) ON DELETE CASCADE,
                  sequence INTEGER NOT NULL,event_type TEXT NOT NULL,subject_id TEXT,
                  payload_json TEXT NOT NULL,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  PRIMARY KEY(room_id,sequence)
                );
                CREATE TABLE IF NOT EXISTS sakka_run_audits_v8(
                  run_id TEXT PRIMARY KEY REFERENCES sakka_runs_v8(run_id) ON DELETE CASCADE,
                  audit_hash TEXT NOT NULL,audit_json TEXT NOT NULL,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_sakka_runs_v8_leaderboard
                  ON sakka_runs_v8(board_id,comparison_key,status);
                CREATE INDEX IF NOT EXISTS idx_sakka_boards_v8_owner
                  ON sakka_boards_v8(created_by,created_at);
                """
            )
            board_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(sakka_boards_v8)").fetchall()
            }
            if "signature_algorithm" not in board_columns:
                conn.execute("ALTER TABLE sakka_boards_v8 ADD COLUMN signature_algorithm TEXT")
            if "signature_key_id" not in board_columns:
                conn.execute("ALTER TABLE sakka_boards_v8 ADD COLUMN signature_key_id TEXT")
            if "board_signature" not in board_columns:
                conn.execute("ALTER TABLE sakka_boards_v8 ADD COLUMN board_signature TEXT")
            for column in ("environment_hash", "deal_schedule_hash", "agent_snapshot_hash", "shared_profile_hash", "terminology_version"):
                if column not in board_columns:
                    conn.execute(f"ALTER TABLE sakka_boards_v8 ADD COLUMN {column} TEXT")
            action_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(sakka_run_actions_v8)").fetchall()
            }
            if "decision_id" not in action_columns:
                conn.execute("ALTER TABLE sakka_run_actions_v8 ADD COLUMN decision_id TEXT")
            if "pre_state_payload_hash" not in action_columns:
                conn.execute("ALTER TABLE sakka_run_actions_v8 ADD COLUMN pre_state_payload_hash TEXT NOT NULL DEFAULT ''")
            if "previous_action_hash" not in action_columns:
                conn.execute("ALTER TABLE sakka_run_actions_v8 ADD COLUMN previous_action_hash TEXT NOT NULL DEFAULT ''")
            if "action_hash" not in action_columns:
                conn.execute("ALTER TABLE sakka_run_actions_v8 ADD COLUMN action_hash TEXT NOT NULL DEFAULT ''")
            run_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(sakka_runs_v8)").fetchall()
            }
            if "game_id" not in run_columns:
                conn.execute("ALTER TABLE sakka_runs_v8 ADD COLUMN game_id TEXT")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_sakka_runs_v8_game "
                "ON sakka_runs_v8(game_id) WHERE game_id IS NOT NULL"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @staticmethod
    def _j(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _board(row: sqlite3.Row) -> SakkaBoard:
        keys = set(row.keys())
        return SakkaBoard(
            board_id=row["board_id"], version=int(row["version"]), board_hash=row["board_hash"],
            serial=row["serial"], share_code=row["share_code"], bundle_hash=row["bundle_hash"],
            observer_seat=Seat.parse(row["observer_seat"]), scope=row["scope"],
            maximum_rounds=int(row["maximum_rounds"]),
            starting_match_score=tuple(json.loads(row["starting_match_score"])),
            starting_round_number=int(row["starting_round_number"]),
            public_event_prefix_hash=row["public_event_prefix_hash"],
            bot_policy_hashes=json.loads(row["bot_policy_hashes"]),
            scoring_profile=row["scoring_profile"], time_control=json.loads(row["time_control"]),
            reveal_policy=row["reveal_policy"], trusted_payload_hash=row["trusted_payload_hash"],
            created_by=row["created_by"], visibility=row["visibility"], status=row["status"],
            label=row["label"], created_at=row["created_at"],
            signature_algorithm=row["signature_algorithm"] if "signature_algorithm" in keys else None,
            signature_key_id=row["signature_key_id"] if "signature_key_id" in keys else None,
            board_signature=row["board_signature"] if "board_signature" in keys else None,
            environment_hash=row["environment_hash"] if "environment_hash" in keys else None,
            deal_schedule_hash=row["deal_schedule_hash"] if "deal_schedule_hash" in keys else None,
            agent_snapshot_hash=row["agent_snapshot_hash"] if "agent_snapshot_hash" in keys else None,
            shared_profile_hash=row["shared_profile_hash"] if "shared_profile_hash" in keys else None,
            terminology_version=(row["terminology_version"] if "terminology_version" in keys and row["terminology_version"] else "ar_sa_baloot_v2"),
        )

    @staticmethod
    def _room(row: sqlite3.Row) -> SakkaRoom:
        return SakkaRoom(
            row["room_id"], row["board_id"], row["room_code"], row["created_by"], row["mode"],
            row["status"], int(row["max_players"]), bool(row["allow_guests"]),
            json.loads(row["settings_json"]), row["created_at"], row["expires_at"],
        )

    @staticmethod
    def _run(row: sqlite3.Row) -> SakkaRun:
        keys = set(row.keys())
        return SakkaRun(
            row["run_id"], row["room_id"], row["board_id"], row["subject_id"], row["comparison_key"],
            row["status"], int(row["version"]), int(row["rounds_completed"]), row["state_hash"],
            json.loads(row["result_json"]) if row["result_json"] else None,
            row["analysis_status"], row["started_at"], row["completed_at"],
            row["game_id"] if "game_id" in keys else None,
        )

    def put_board(self, board: SakkaBoard, trusted_payload: bytes) -> None:
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT board_hash,trusted_payload_hash FROM sakka_boards_v8 WHERE board_id=?", (board.board_id,)
            ).fetchone()
            if existing and (existing[0] != board.board_hash or existing[1] != board.trusted_payload_hash):
                raise RuntimeError("immutable Sakka board identity conflict")
            conn.execute(
                """INSERT OR IGNORE INTO sakka_boards_v8(
                   board_id,version,board_hash,serial,share_code,bundle_hash,observer_seat,scope,
                   maximum_rounds,starting_match_score,starting_round_number,public_event_prefix_hash,
                   bot_policy_hashes,scoring_profile,time_control,reveal_policy,trusted_payload,
                   trusted_payload_hash,created_by,visibility,status,label,created_at,
                   signature_algorithm,signature_key_id,board_signature,environment_hash,deal_schedule_hash,
                   agent_snapshot_hash,shared_profile_hash,terminology_version
                 ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    board.board_id, board.version, board.board_hash, board.serial, board.share_code,
                    board.bundle_hash, board.observer_seat.name.lower(), board.scope, board.maximum_rounds,
                    self._j(list(board.starting_match_score)), board.starting_round_number,
                    board.public_event_prefix_hash, self._j(dict(board.bot_policy_hashes)), board.scoring_profile,
                    self._j(dict(board.time_control)), board.reveal_policy, trusted_payload,
                    board.trusted_payload_hash, board.created_by, board.visibility, board.status,
                    board.label, board.created_at, board.signature_algorithm, board.signature_key_id,
                    board.board_signature, board.environment_hash, board.deal_schedule_hash,
                    board.agent_snapshot_hash, board.shared_profile_hash, board.terminology_version,
                ),
            )

    def get_board(self, identifier: str) -> tuple[SakkaBoard, bytes] | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM sakka_boards_v8
                   WHERE board_id=? OR serial=? OR share_code=? OR board_hash=?""",
                (identifier, identifier.upper(), identifier.upper(), identifier.lower()),
            ).fetchone()
        return None if row is None else (self._board(row), bytes(row["trusted_payload"]))

    def list_boards(self, *, subject_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if subject_id:
                rows = conn.execute(
                    """SELECT * FROM sakka_boards_v8
                       WHERE created_by=? OR visibility='public'
                       ORDER BY created_at DESC LIMIT ?""",
                    (subject_id, max(1, min(500, int(limit)))),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM sakka_boards_v8 WHERE visibility='public' ORDER BY created_at DESC LIMIT ?",
                    (max(1, min(500, int(limit))),),
                ).fetchall()
        return [self._board(row).to_dict(include_hashes=False) for row in rows]

    def put_room(self, room: SakkaRoom) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO sakka_rooms_v8 VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    room.room_id, room.board_id, room.room_code, room.created_by, room.mode, room.status,
                    room.max_players, int(room.allow_guests), self._j(dict(room.settings)), room.created_at,
                    room.expires_at,
                ),
            )

    def get_room(self, identifier: str) -> SakkaRoom | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sakka_rooms_v8 WHERE room_id=? OR room_code=?",
                (identifier, identifier.upper()),
            ).fetchone()
        return None if row is None else self._room(row)

    def join(self, room_id: str, subject_id: str, display_name: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            room = conn.execute("SELECT max_players FROM sakka_rooms_v8 WHERE room_id=?", (room_id,)).fetchone()
            if room is None:
                raise KeyError(room_id)
            existing = conn.execute(
                "SELECT 1 FROM sakka_members_v8 WHERE room_id=? AND subject_id=?", (room_id, subject_id)
            ).fetchone()
            count = int(conn.execute("SELECT COUNT(*) FROM sakka_members_v8 WHERE room_id=?", (room_id,)).fetchone()[0])
            if existing is None and count >= int(room["max_players"]):
                raise ValueError("room is full")
            conn.execute(
                "INSERT OR IGNORE INTO sakka_members_v8(room_id,subject_id,display_name) VALUES(?,?,?)",
                (room_id, subject_id, display_name),
            )
            conn.commit()

    def members(self, room_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT subject_id,display_name,joined_at FROM sakka_members_v8 WHERE room_id=? ORDER BY joined_at",
                (room_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def put_run(self, run: SakkaRun, payload: bytes) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO sakka_runs_v8(
                     run_id,room_id,board_id,subject_id,comparison_key,status,version,
                     rounds_completed,session_payload,state_hash,result_json,analysis_status,
                     started_at,completed_at,game_id
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(run_id) DO UPDATE SET
                     status=excluded.status,version=excluded.version,rounds_completed=excluded.rounds_completed,
                     session_payload=excluded.session_payload,state_hash=excluded.state_hash,
                     result_json=excluded.result_json,analysis_status=excluded.analysis_status,
                     completed_at=excluded.completed_at,game_id=excluded.game_id""",
                (
                    run.run_id, run.room_id, run.board_id, run.subject_id, run.comparison_key,
                    run.status, run.version, run.rounds_completed, payload, run.state_hash,
                    self._j(dict(run.result)) if run.result else None, run.analysis_status,
                    run.started_at, run.completed_at, run.game_id,
                ),
            )

    def get_run(self, run_id: str) -> tuple[SakkaRun, bytes] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sakka_runs_v8 WHERE run_id=?", (run_id,)).fetchone()
        return None if row is None else (self._run(row), bytes(row["session_payload"]))

    def get_run_for_decision(self, decision_id: str) -> tuple[SakkaRun, bytes] | None:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT run_id FROM sakka_run_actions_v8 WHERE decision_id=? LIMIT 2",
                (decision_id,),
            ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise RuntimeError("decision is linked to multiple shared runs")
        loaded = self.get_run(str(rows[0]["run_id"]))
        if loaded is None:
            raise RuntimeError("shared decision run is unavailable")
        return loaded

    def get_run_for_game(self, game_id: str) -> tuple[SakkaRun, bytes] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sakka_runs_v8 WHERE game_id=?",
                (game_id,),
            ).fetchone()
        return None if row is None else (self._run(row), bytes(row["session_payload"]))

    def get_run_for_subject(self, room_id: str, subject_id: str) -> tuple[SakkaRun, bytes] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sakka_runs_v8 WHERE room_id=? AND subject_id=?", (room_id, subject_id)
            ).fetchone()
        return None if row is None else (self._run(row), bytes(row["session_payload"]))

    def add_action(self, run_id: str, sequence: int, action: dict[str, Any], state_hash: str,
                   pre_state_payload: bytes, legal_actions: list[dict[str, Any]],
                   decision_id: str | None = None, created_at: str | None = None) -> None:
        recorded_at = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            previous = conn.execute(
                "SELECT action_hash FROM sakka_run_actions_v8 WHERE run_id=? ORDER BY sequence DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            previous_hash = str(previous[0]) if previous and previous[0] else ""
            pre_hash = hashlib.sha256(pre_state_payload).hexdigest()
            record_hash = action_record_hash(
                run_id=run_id, sequence=sequence, action=action, state_hash=state_hash,
                pre_state_payload_hash=pre_hash, legal_actions=legal_actions,
                decision_id=decision_id, previous_action_hash=previous_hash,
            )
            conn.execute(
                """INSERT INTO sakka_run_actions_v8(
                     run_id,sequence,action_json,state_hash,pre_state_payload,legal_actions_json,decision_id,
                     pre_state_payload_hash,previous_action_hash,action_hash,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, sequence, self._j(action), state_hash, pre_state_payload,
                 self._j(legal_actions), decision_id, pre_hash, previous_hash, record_hash, recorded_at),
            )
            conn.commit()

    def actions(self, run_id: str, *, include_private: bool = False) -> list[dict[str, Any]]:
        columns = "sequence,action_json,state_hash,legal_actions_json,decision_id,pre_state_payload_hash,previous_action_hash,action_hash,created_at"
        if include_private:
            columns += ",pre_state_payload"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {columns} FROM sakka_run_actions_v8 WHERE run_id=? ORDER BY sequence", (run_id,)
            ).fetchall()
        values: list[dict[str, Any]] = []
        for row in rows:
            item = {
                "sequence": int(row["sequence"]),
                "action": json.loads(row["action_json"]),
                "stateHash": row["state_hash"],
                "legalActions": json.loads(row["legal_actions_json"]),
                "decisionId": row["decision_id"],
                "preStatePayloadHash": row["pre_state_payload_hash"],
                "previousActionHash": row["previous_action_hash"],
                "actionHash": row["action_hash"],
                "createdAt": row["created_at"],
            }
            if include_private:
                item["preStatePayload"] = bytes(row["pre_state_payload"])
            values.append(item)
        return values

    def leaderboard(self, board_id: str, comparison_key: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT r.*,m.display_name FROM sakka_runs_v8 r
                   LEFT JOIN sakka_members_v8 m ON m.room_id=r.room_id AND m.subject_id=r.subject_id
                   WHERE r.board_id=? AND r.comparison_key=? AND r.status='completed'""",
                (board_id, comparison_key),
            ).fetchall()
        items = []
        for row in rows:
            result = json.loads(row["result_json"])
            items.append({
                "runId": row["run_id"], "subjectId": row["subject_id"],
                "displayName": row["display_name"] or "لاعب", "result": result,
                "completedAt": row["completed_at"],
            })
        items.sort(key=_leaderboard_key, reverse=True)
        for rank, item in enumerate(items, 1):
            item["rank"] = rank
        return items

    def append_room_event(self, room_id: str, event_type: str, payload: dict[str, Any],
                          *, subject_id: str | None = None) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM sakka_rooms_v8 WHERE room_id=?", (room_id,)).fetchone() is None:
                raise KeyError(room_id)
            sequence = int(conn.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 FROM sakka_room_events_v8 WHERE room_id=?",
                (room_id,),
            ).fetchone()[0])
            conn.execute(
                """INSERT INTO sakka_room_events_v8(room_id,sequence,event_type,subject_id,payload_json)
                   VALUES(?,?,?,?,?)""",
                (room_id, sequence, event_type, subject_id, self._j(payload)),
            )
            conn.commit()
        return {"roomId": room_id, "sequence": sequence, "eventType": event_type,
                "subjectId": subject_id, "payload": dict(payload)}

    def room_events(self, room_id: str, *, after_sequence: int = 0,
                    limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT sequence,event_type,subject_id,payload_json,created_at
                   FROM sakka_room_events_v8 WHERE room_id=? AND sequence>?
                   ORDER BY sequence LIMIT ?""",
                (room_id, max(0, int(after_sequence)), max(1, min(1000, int(limit)))),
            ).fetchall()
        return [{
            "roomId": room_id, "sequence": int(row["sequence"]), "eventType": row["event_type"],
            "subjectId": row["subject_id"], "payload": json.loads(row["payload_json"]),
            "createdAt": row["created_at"],
        } for row in rows]

    def put_run_audit(self, run_id: str, audit: dict[str, Any]) -> dict[str, Any]:
        import hashlib
        value = self._j(audit)
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO sakka_run_audits_v8(run_id,audit_hash,audit_json)
                   VALUES(?,?,?) ON CONFLICT(run_id) DO UPDATE SET
                     audit_hash=excluded.audit_hash,audit_json=excluded.audit_json,
                     created_at=CURRENT_TIMESTAMP""",
                (run_id, digest, value),
            )
        return {**audit, "auditHash": digest}

    def get_run_audit(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT audit_hash,audit_json,created_at FROM sakka_run_audits_v8 WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return {**json.loads(row["audit_json"]), "auditHash": row["audit_hash"],
                "createdAt": row["created_at"]}


class PostgresSakkaStore:
    """PostgreSQL production store for immutable boards and concurrent runs."""

    def __init__(self, dsn: str):
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('Install with: pip install "baloot-platform[postgres]"') from exc
        self.pool = ConnectionPool(dsn, min_size=1, max_size=16, open=True)

    @staticmethod
    def _board(row) -> SakkaBoard:
        return SakkaBoard(
            board_id=row[0], version=int(row[1]), board_hash=row[2], serial=row[3], share_code=row[4],
            bundle_hash=row[5], observer_seat=Seat.parse(row[6]), scope=row[7], maximum_rounds=int(row[8]),
            starting_match_score=tuple(row[9]), starting_round_number=int(row[10]),
            public_event_prefix_hash=row[11], bot_policy_hashes=row[12], scoring_profile=row[13],
            time_control=row[14], reveal_policy=row[15], trusted_payload_hash=row[17],
            created_by=row[18], visibility=row[19], status=row[20], label=row[21],
            created_at=row[22].isoformat() if hasattr(row[22], "isoformat") else str(row[22]),
            signature_algorithm=row[23] if len(row) > 23 else None,
            signature_key_id=row[24] if len(row) > 24 else None,
            board_signature=row[25] if len(row) > 25 else None,
            environment_hash=row[26] if len(row) > 26 else None,
            deal_schedule_hash=row[27] if len(row) > 27 else None,
            agent_snapshot_hash=row[28] if len(row) > 28 else None,
            shared_profile_hash=row[29] if len(row) > 29 else None,
            terminology_version=(row[30] if len(row) > 30 and row[30] else "ar_sa_baloot_v2"),
        )

    @staticmethod
    def _room(row) -> SakkaRoom:
        return SakkaRoom(
            row[0], row[1], row[2], row[3], row[4], row[5], int(row[6]), bool(row[7]), row[8],
            row[9].isoformat(), row[10].isoformat() if row[10] else None,
        )

    @staticmethod
    def _run(row) -> SakkaRun:
        return SakkaRun(
            row[0], row[1], row[2], row[3], row[4], row[5], int(row[6]), int(row[7]), row[9],
            row[10], row[11], row[12].isoformat(), row[13].isoformat() if row[13] else None,
            row[14] if len(row) > 14 else None,
        )

    def put_board(self, board: SakkaBoard, trusted_payload: bytes) -> None:
        with self.pool.connection() as conn, conn.transaction():
            existing = conn.execute(
                "SELECT board_hash,trusted_payload_hash FROM duplicate.boards WHERE board_id=%s FOR UPDATE",
                (board.board_id,),
            ).fetchone()
            if existing and (str(existing[0]) != board.board_hash or str(existing[1]) != board.trusted_payload_hash):
                raise RuntimeError("immutable Sakka board identity conflict")
            conn.execute(
                """INSERT INTO duplicate.boards(
                     board_id,version,board_hash,serial,share_code,bundle_hash,observer_seat,scope,
                     maximum_rounds,starting_match_score,starting_round_number,public_event_prefix_hash,
                     bot_policy_hashes,scoring_profile,time_control,reveal_policy,trusted_payload,
                     trusted_payload_hash,created_by,visibility,status,label,created_at,
                     signature_algorithm,signature_key_id,board_signature,environment_hash,deal_schedule_hash,
                     agent_snapshot_hash,shared_profile_hash,terminology_version
                   ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s::timestamptz,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT(board_id) DO NOTHING""",
                (
                    board.board_id, board.version, board.board_hash, board.serial, board.share_code,
                    board.bundle_hash, board.observer_seat.name.lower(), board.scope, board.maximum_rounds,
                    json.dumps(list(board.starting_match_score)), board.starting_round_number,
                    board.public_event_prefix_hash, json.dumps(dict(board.bot_policy_hashes)),
                    board.scoring_profile, json.dumps(dict(board.time_control)), board.reveal_policy,
                    trusted_payload, board.trusted_payload_hash, board.created_by, board.visibility,
                    board.status, board.label, board.created_at, board.signature_algorithm,
                    board.signature_key_id, board.board_signature, board.environment_hash,
                    board.deal_schedule_hash, board.agent_snapshot_hash, board.shared_profile_hash,
                    board.terminology_version,
                ),
            )

    def get_board(self, identifier: str) -> tuple[SakkaBoard, bytes] | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                """SELECT board_id,version,board_hash,serial,share_code,bundle_hash,observer_seat,scope,
                          maximum_rounds,starting_match_score,starting_round_number,public_event_prefix_hash,
                          bot_policy_hashes,scoring_profile,time_control,reveal_policy,trusted_payload,
                          trusted_payload_hash,created_by,visibility,status,label,created_at,
                          signature_algorithm,signature_key_id,board_signature,environment_hash,deal_schedule_hash,
                          agent_snapshot_hash,shared_profile_hash,terminology_version
                   FROM duplicate.boards
                   WHERE board_id=%s OR serial=%s OR share_code=%s OR board_hash=%s""",
                (identifier, identifier.upper(), identifier.upper(), identifier.lower()),
            ).fetchone()
        if not row:
            return None
        return self._board(row), bytes(row[16])

    def list_boards(self, *, subject_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            if subject_id:
                rows = conn.execute(
                    """SELECT board_id,version,board_hash,serial,share_code,bundle_hash,observer_seat,scope,
                              maximum_rounds,starting_match_score,starting_round_number,public_event_prefix_hash,
                              bot_policy_hashes,scoring_profile,time_control,reveal_policy,trusted_payload,
                              trusted_payload_hash,created_by,visibility,status,label,created_at,
                              signature_algorithm,signature_key_id,board_signature,environment_hash,deal_schedule_hash,
                          agent_snapshot_hash,shared_profile_hash,terminology_version
                       FROM duplicate.boards WHERE created_by=%s OR visibility='public'
                       ORDER BY created_at DESC LIMIT %s""",
                    (subject_id, max(1, min(500, int(limit)))),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT board_id,version,board_hash,serial,share_code,bundle_hash,observer_seat,scope,
                              maximum_rounds,starting_match_score,starting_round_number,public_event_prefix_hash,
                              bot_policy_hashes,scoring_profile,time_control,reveal_policy,trusted_payload,
                              trusted_payload_hash,created_by,visibility,status,label,created_at,
                              signature_algorithm,signature_key_id,board_signature,environment_hash,deal_schedule_hash,
                          agent_snapshot_hash,shared_profile_hash,terminology_version
                       FROM duplicate.boards WHERE visibility='public'
                       ORDER BY created_at DESC LIMIT %s""",
                    (max(1, min(500, int(limit))),),
                ).fetchall()
        return [self._board(row).to_dict(include_hashes=False) for row in rows]

    def put_room(self, room: SakkaRoom) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO duplicate.rooms(
                     room_id,board_id,room_code,created_by,mode,status,max_players,allow_guests,
                     settings_json,created_at,expires_at
                   ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::timestamptz,%s::timestamptz)""",
                (room.room_id,room.board_id,room.room_code,room.created_by,room.mode,room.status,
                 room.max_players,room.allow_guests,json.dumps(dict(room.settings)),room.created_at,room.expires_at),
            )

    def get_room(self, identifier: str) -> SakkaRoom | None:
        with self.pool.connection() as conn:
            row=conn.execute(
                """SELECT room_id,board_id,room_code,created_by,mode,status,max_players,allow_guests,
                          settings_json,created_at,expires_at
                   FROM duplicate.rooms WHERE room_id=%s OR room_code=%s""",
                (identifier,identifier.upper()),
            ).fetchone()
        return None if not row else self._room(row)

    def join(self, room_id: str, subject_id: str, display_name: str) -> None:
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (room_id,))
            room=conn.execute("SELECT max_players FROM duplicate.rooms WHERE room_id=%s",(room_id,)).fetchone()
            if not room:
                raise KeyError(room_id)
            exists=conn.execute("SELECT 1 FROM duplicate.members WHERE room_id=%s AND subject_id=%s",(room_id,subject_id)).fetchone()
            count=int(conn.execute("SELECT COUNT(*) FROM duplicate.members WHERE room_id=%s",(room_id,)).fetchone()[0])
            if not exists and count>=int(room[0]):
                raise ValueError("room is full")
            conn.execute(
                """INSERT INTO duplicate.members(room_id,subject_id,display_name)
                   VALUES(%s,%s,%s) ON CONFLICT(room_id,subject_id) DO NOTHING""",
                (room_id,subject_id,display_name),
            )

    def members(self, room_id: str) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            rows=conn.execute(
                "SELECT subject_id,display_name,joined_at FROM duplicate.members WHERE room_id=%s ORDER BY joined_at",
                (room_id,),
            ).fetchall()
        return [{"subject_id":r[0],"display_name":r[1],"joined_at":r[2].isoformat()} for r in rows]

    def put_run(self, run: SakkaRun, payload: bytes) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO duplicate.runs(
                     run_id,room_id,board_id,subject_id,comparison_key,status,version,rounds_completed,
                     session_payload,state_hash,result_json,analysis_status,started_at,completed_at
                     ,game_id
                   ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s::timestamptz,%s::timestamptz,%s)
                   ON CONFLICT(run_id) DO UPDATE SET
                     status=excluded.status,version=excluded.version,rounds_completed=excluded.rounds_completed,
                     session_payload=excluded.session_payload,state_hash=excluded.state_hash,
                     result_json=excluded.result_json,analysis_status=excluded.analysis_status,
                     completed_at=excluded.completed_at,game_id=excluded.game_id""",
                (run.run_id,run.room_id,run.board_id,run.subject_id,run.comparison_key,run.status,run.version,
                 run.rounds_completed,payload,run.state_hash,json.dumps(dict(run.result)) if run.result else None,
                 run.analysis_status,run.started_at,run.completed_at,run.game_id),
            )

    def get_run(self, run_id: str) -> tuple[SakkaRun, bytes] | None:
        with self.pool.connection() as conn:
            row=conn.execute(
                """SELECT run_id,room_id,board_id,subject_id,comparison_key,status,version,rounds_completed,
                          session_payload,state_hash,result_json,analysis_status,started_at,completed_at
                          ,game_id
                   FROM duplicate.runs WHERE run_id=%s""",(run_id,),
            ).fetchone()
        return None if not row else (self._run(row),bytes(row[8]))

    def get_run_for_decision(self, decision_id: str) -> tuple[SakkaRun, bytes] | None:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT run_id FROM duplicate.run_actions WHERE decision_id=%s LIMIT 2",
                (decision_id,),
            ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise RuntimeError("decision is linked to multiple shared runs")
        loaded = self.get_run(str(rows[0][0]))
        if loaded is None:
            raise RuntimeError("shared decision run is unavailable")
        return loaded

    def get_run_for_game(self, game_id: str) -> tuple[SakkaRun, bytes] | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                """SELECT run_id,room_id,board_id,subject_id,comparison_key,status,version,
                          rounds_completed,session_payload,state_hash,result_json,analysis_status,
                          started_at,completed_at,game_id
                   FROM duplicate.runs WHERE game_id=%s""",
                (game_id,),
            ).fetchone()
        return None if not row else (self._run(row), bytes(row[8]))

    def get_run_for_subject(self, room_id: str, subject_id: str) -> tuple[SakkaRun, bytes] | None:
        with self.pool.connection() as conn:
            row=conn.execute(
                """SELECT run_id,room_id,board_id,subject_id,comparison_key,status,version,rounds_completed,
                          session_payload,state_hash,result_json,analysis_status,started_at,completed_at,
                          game_id
                   FROM duplicate.runs WHERE room_id=%s AND subject_id=%s""",(room_id,subject_id),
            ).fetchone()
        return None if not row else (self._run(row),bytes(row[8]))

    def add_action(self, run_id: str, sequence: int, action: dict[str, Any], state_hash: str,
                   pre_state_payload: bytes, legal_actions: list[dict[str, Any]],
                   decision_id: str | None = None, created_at: str | None = None) -> None:
        recorded_at = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (run_id,))
            previous = conn.execute(
                "SELECT action_hash FROM duplicate.run_actions WHERE run_id=%s ORDER BY sequence DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            previous_hash = str(previous[0]) if previous and previous[0] else ""
            pre_hash = hashlib.sha256(pre_state_payload).hexdigest()
            record_hash = action_record_hash(
                run_id=run_id, sequence=sequence, action=action, state_hash=state_hash,
                pre_state_payload_hash=pre_hash, legal_actions=legal_actions,
                decision_id=decision_id, previous_action_hash=previous_hash,
            )
            conn.execute(
                """INSERT INTO duplicate.run_actions(
                     run_id,sequence,action_json,state_hash,pre_state_payload,legal_actions_json,decision_id,
                     pre_state_payload_hash,previous_action_hash,action_hash,created_at
                   ) VALUES(%s,%s,%s::jsonb,%s,%s,%s::jsonb,%s,%s,%s,%s,%s::timestamptz)""",
                (run_id,sequence,json.dumps(action),state_hash,pre_state_payload,
                 json.dumps(legal_actions),decision_id,pre_hash,previous_hash,record_hash,recorded_at),
            )

    def actions(self, run_id: str, *, include_private: bool = False) -> list[dict[str, Any]]:
        fields="sequence,action_json,state_hash,legal_actions_json,decision_id,pre_state_payload_hash,previous_action_hash,action_hash,created_at"
        if include_private:
            fields += ",pre_state_payload"
        with self.pool.connection() as conn:
            rows=conn.execute(f"SELECT {fields} FROM duplicate.run_actions WHERE run_id=%s ORDER BY sequence",(run_id,)).fetchall()
        values=[]
        for r in rows:
            item={"sequence":int(r[0]),"action":r[1],"stateHash":r[2],"legalActions":r[3],
                  "decisionId":r[4],"preStatePayloadHash":r[5],
                  "previousActionHash":r[6],"actionHash":r[7],
                  "createdAt":r[8].isoformat()}
            if include_private:
                item["preStatePayload"] = bytes(r[9])
            values.append(item)
        return values

    def leaderboard(self, board_id: str, comparison_key: str) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            rows=conn.execute(
                """SELECT r.run_id,r.subject_id,r.result_json,r.completed_at,m.display_name
                   FROM duplicate.runs r LEFT JOIN duplicate.members m
                     ON m.room_id=r.room_id AND m.subject_id=r.subject_id
                   WHERE r.board_id=%s AND r.comparison_key=%s AND r.status='completed'""",
                (board_id,comparison_key),
            ).fetchall()
        items=[{"runId":r[0],"subjectId":r[1],"displayName":r[4] or "لاعب","result":r[2],"completedAt":r[3].isoformat() if r[3] else None} for r in rows]
        items.sort(key=_leaderboard_key, reverse=True)
        for rank,item in enumerate(items,1):
            item["rank"] = rank
        return items

    def append_room_event(self, room_id: str, event_type: str, payload: dict[str, Any],
                          *, subject_id: str | None = None) -> dict[str, Any]:
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (room_id,))
            if conn.execute("SELECT 1 FROM duplicate.rooms WHERE room_id=%s", (room_id,)).fetchone() is None:
                raise KeyError(room_id)
            sequence = int(conn.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 FROM duplicate.room_events WHERE room_id=%s",
                (room_id,),
            ).fetchone()[0])
            conn.execute(
                """INSERT INTO duplicate.room_events(room_id,sequence,event_type,subject_id,payload_json)
                   VALUES(%s,%s,%s,%s,%s::jsonb)""",
                (room_id, sequence, event_type, subject_id, json.dumps(payload)),
            )
        return {"roomId": room_id, "sequence": sequence, "eventType": event_type,
                "subjectId": subject_id, "payload": dict(payload)}

    def room_events(self, room_id: str, *, after_sequence: int = 0,
                    limit: int = 200) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                """SELECT sequence,event_type,subject_id,payload_json,created_at
                   FROM duplicate.room_events WHERE room_id=%s AND sequence>%s
                   ORDER BY sequence LIMIT %s""",
                (room_id, max(0, int(after_sequence)), max(1, min(1000, int(limit)))),
            ).fetchall()
        return [{"roomId": room_id, "sequence": int(row[0]), "eventType": row[1],
                 "subjectId": row[2], "payload": row[3], "createdAt": row[4].isoformat()}
                for row in rows]

    def put_run_audit(self, run_id: str, audit: dict[str, Any]) -> dict[str, Any]:
        import hashlib
        value = json.dumps(audit, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO duplicate.run_audits(run_id,audit_hash,audit_json)
                   VALUES(%s,%s,%s::jsonb) ON CONFLICT(run_id) DO UPDATE SET
                     audit_hash=excluded.audit_hash,audit_json=excluded.audit_json,created_at=now()""",
                (run_id, digest, value),
            )
        return {**audit, "auditHash": digest}

    def get_run_audit(self, run_id: str) -> dict[str, Any] | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT audit_hash,audit_json,created_at FROM duplicate.run_audits WHERE run_id=%s",
                (run_id,),
            ).fetchone()
        return None if not row else {**row[1], "auditHash": row[0], "createdAt": row[2].isoformat()}
