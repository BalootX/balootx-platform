from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from baloot_platform.core.enums import Seat
from baloot_platform.duplicate.models import BoardDefinition, DuplicateRoom, DuplicateRun


class SQLiteDuplicateStore:
    def __init__(self, path: str | Path):
        self.path = str(path); Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS duplicate_boards_v7(
              board_id TEXT PRIMARY KEY,version INTEGER NOT NULL,board_hash TEXT NOT NULL UNIQUE,
              serial TEXT NOT NULL UNIQUE,bundle_hash TEXT NOT NULL,observer_seat TEXT NOT NULL,
              starting_match_score TEXT NOT NULL,public_event_prefix_hash TEXT NOT NULL,
              bot_policy_hashes TEXT NOT NULL,scoring_profile TEXT NOT NULL,time_control TEXT NOT NULL,
              reveal_policy TEXT NOT NULL,trusted_payload BLOB NOT NULL,trusted_payload_hash TEXT NOT NULL,
              status TEXT NOT NULL,label TEXT,created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS duplicate_rooms_v7(
              room_id TEXT PRIMARY KEY,board_id TEXT NOT NULL,room_code TEXT NOT NULL UNIQUE,
              mode TEXT NOT NULL,status TEXT NOT NULL,settings_json TEXT NOT NULL,created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS duplicate_members_v7(
              room_id TEXT NOT NULL,subject_id TEXT NOT NULL,joined_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(room_id,subject_id)
            );
            CREATE TABLE IF NOT EXISTS duplicate_runs_v7(
              run_id TEXT PRIMARY KEY,room_id TEXT NOT NULL,board_id TEXT NOT NULL,subject_id TEXT NOT NULL,
              comparison_key TEXT NOT NULL,status TEXT NOT NULL,version INTEGER NOT NULL,
              session_payload BLOB NOT NULL,state_hash TEXT,result_json TEXT,started_at TEXT NOT NULL,
              completed_at TEXT,UNIQUE(room_id,subject_id)
            );
            CREATE TABLE IF NOT EXISTS duplicate_run_actions_v7(
              run_id TEXT NOT NULL,sequence INTEGER NOT NULL,action_json TEXT NOT NULL,state_hash TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,PRIMARY KEY(run_id,sequence)
            );
            CREATE INDEX IF NOT EXISTS idx_duplicate_results_v7 ON duplicate_runs_v7(board_id,comparison_key,status);
            """)

    def _connect(self):
        conn=sqlite3.connect(self.path,timeout=30); conn.row_factory=sqlite3.Row; return conn

    @staticmethod
    def _j(value: Any) -> str:
        return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"))

    def put_board(self, board: BoardDefinition, trusted_payload: bytes) -> None:
        with self._lock,self._connect() as conn:
            existing=conn.execute("SELECT board_hash,trusted_payload_hash FROM duplicate_boards_v7 WHERE board_id=?",(board.board_id,)).fetchone()
            if existing and (existing[0]!=board.board_hash or existing[1]!=board.trusted_payload_hash):
                raise RuntimeError("immutable board identity conflict")
            conn.execute("""INSERT OR IGNORE INTO duplicate_boards_v7 VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",(
                board.board_id,board.version,board.board_hash,board.serial,board.bundle_hash,board.observer_seat.name.lower(),
                self._j(list(board.starting_match_score)),board.public_event_prefix_hash,self._j(dict(board.bot_policy_hashes)),
                board.scoring_profile,self._j(dict(board.time_control)),board.reveal_policy,trusted_payload,
                board.trusted_payload_hash,board.status,board.label,board.created_at))

    def get_board(self, identifier: str) -> tuple[BoardDefinition,bytes] | None:
        with self._connect() as conn:
            r=conn.execute("SELECT * FROM duplicate_boards_v7 WHERE board_id=? OR serial=? OR board_hash=?",(identifier,identifier,identifier)).fetchone()
        if not r:return None
        b=BoardDefinition(r['board_id'],int(r['version']),r['board_hash'],r['serial'],r['bundle_hash'],Seat.parse(r['observer_seat']),tuple(json.loads(r['starting_match_score'])),r['public_event_prefix_hash'],json.loads(r['bot_policy_hashes']),r['scoring_profile'],json.loads(r['time_control']),r['reveal_policy'],r['trusted_payload_hash'],r['status'],r['label'],r['created_at'])
        return b,bytes(r['trusted_payload'])

    def put_room(self, room: DuplicateRoom) -> None:
        with self._lock,self._connect() as conn:
            conn.execute("INSERT INTO duplicate_rooms_v7 VALUES(?,?,?,?,?,?,?)",(room.room_id,room.board_id,room.room_code,room.mode,room.status,self._j(dict(room.settings)),room.created_at))

    def get_room(self, identifier: str) -> DuplicateRoom | None:
        with self._connect() as conn:r=conn.execute("SELECT * FROM duplicate_rooms_v7 WHERE room_id=? OR room_code=?",(identifier,identifier)).fetchone()
        return None if not r else DuplicateRoom(r['room_id'],r['board_id'],r['room_code'],r['mode'],r['status'],json.loads(r['settings_json']),r['created_at'])

    def join(self,room_id:str,subject_id:str)->None:
        with self._lock,self._connect() as conn:conn.execute("INSERT OR IGNORE INTO duplicate_members_v7(room_id,subject_id) VALUES(?,?)",(room_id,subject_id))

    def put_run(self,run:DuplicateRun,payload:bytes)->None:
        with self._lock,self._connect() as conn:
            conn.execute("""INSERT INTO duplicate_runs_v7 VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id) DO UPDATE SET status=excluded.status,version=excluded.version,session_payload=excluded.session_payload,state_hash=excluded.state_hash,result_json=excluded.result_json,completed_at=excluded.completed_at""",(
                run.run_id,run.room_id,run.board_id,run.subject_id,run.comparison_key,run.status,run.version,payload,run.state_hash,self._j(dict(run.result)) if run.result else None,run.started_at,run.completed_at))

    def get_run(self,run_id:str)->tuple[DuplicateRun,bytes]|None:
        with self._connect() as conn:r=conn.execute("SELECT * FROM duplicate_runs_v7 WHERE run_id=?",(run_id,)).fetchone()
        if not r:return None
        return DuplicateRun(r['run_id'],r['room_id'],r['board_id'],r['subject_id'],r['comparison_key'],r['status'],int(r['version']),r['state_hash'],json.loads(r['result_json']) if r['result_json'] else None,r['started_at'],r['completed_at']),bytes(r['session_payload'])

    def add_action(self,run_id:str,sequence:int,action:dict[str,Any],state_hash:str)->None:
        with self._lock,self._connect() as conn:conn.execute("INSERT INTO duplicate_run_actions_v7 VALUES(?,?,?,?,CURRENT_TIMESTAMP)",(run_id,sequence,self._j(action),state_hash))

    def actions(self,run_id:str)->list[dict[str,Any]]:
        with self._connect() as conn:rows=conn.execute("SELECT * FROM duplicate_run_actions_v7 WHERE run_id=? ORDER BY sequence",(run_id,)).fetchall()
        return [{"sequence":int(r['sequence']),"action":json.loads(r['action_json']),"stateHash":r['state_hash'],"createdAt":r['created_at']} for r in rows]

    def leaderboard(self,board_id:str,comparison_key:str)->list[dict[str,Any]]:
        with self._connect() as conn:rows=conn.execute("SELECT * FROM duplicate_runs_v7 WHERE board_id=? AND comparison_key=? AND status='completed'",(board_id,comparison_key)).fetchall()
        items=[{"runId":r['run_id'],"subjectId":r['subject_id'],"result":json.loads(r['result_json']),"completedAt":r['completed_at']} for r in rows]
        items.sort(key=lambda x:(x['result'].get('awardedPoints',0),x['result'].get('roundMargin',-9999),-x['result'].get('decisionRegret',0)),reverse=True)
        for i,item in enumerate(items,1):item['rank']=i
        return items
