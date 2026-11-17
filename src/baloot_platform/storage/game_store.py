from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Any

from baloot_platform.storage.codec_v7 import decode as decode_payload, encode as encode_payload
from baloot_platform.history.event_store_v7 import SQLiteEventStore
from baloot_platform.version import __version__


class SQLiteGameStore:
    """Durable immutable game versions plus a current-snapshot pointer.

    Snapshots use the V7 typed canonical codec rather than executable pickle.
    Public interchange remains JSON and is replay-validated by the engine. Every version is immutable and
    linked through a SHA-256 chain so old games can be reopened or audited.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.events = SQLiteEventStore(self.path)
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS game_snapshots(
                    game_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    payload BLOB NOT NULL,
                    state_hash TEXT NOT NULL,
                    chain_hash TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS game_versions(
                    game_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    payload BLOB NOT NULL,
                    state_hash TEXT NOT NULL,
                    previous_chain_hash TEXT,
                    chain_hash TEXT NOT NULL,
                    event_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(game_id, version)
                );
                CREATE INDEX IF NOT EXISTS idx_game_versions_created
                    ON game_versions(created_at DESC);
                CREATE TABLE IF NOT EXISTS game_audit(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    state_hash TEXT NOT NULL,
                    event_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_game_audit_game_version
                    ON game_audit(game_id, version);
                """
            )
            # Compatibility migration for databases created before chain_hash.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(game_snapshots)")}
            if "chain_hash" not in columns:
                conn.execute("ALTER TABLE game_snapshots ADD COLUMN chain_hash TEXT")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @staticmethod
    def _encode(session: Any) -> tuple[bytes, str]:
        payload = encode_payload(session)
        return payload, hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _chain(previous: str | None, digest: str, version: int, event_count: int) -> str:
        material = f"{previous or ''}:{digest}:{int(version)}:{int(event_count)}".encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def save(self, session: Any) -> str:
        payload, digest = self._encode(session)
        version = int(session.version)
        event_count = len(session.state.events)
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT state_hash,chain_hash FROM game_versions WHERE game_id=? AND version=?",
                (session.id, version),
            ).fetchone()
            if existing is not None:
                if existing["state_hash"] != digest:
                    raise RuntimeError("immutable game version already exists with different state")
                chain_hash = str(existing["chain_hash"])
            else:
                previous = conn.execute(
                    "SELECT chain_hash FROM game_versions WHERE game_id=? AND version<? ORDER BY version DESC LIMIT 1",
                    (session.id, version),
                ).fetchone()
                previous_hash = str(previous[0]) if previous is not None else None
                chain_hash = self._chain(previous_hash, digest, version, event_count)
                conn.execute(
                    """INSERT INTO game_versions(
                           game_id,version,payload,state_hash,previous_chain_hash,chain_hash,event_count
                       ) VALUES(?,?,?,?,?,?,?)""",
                    (session.id, version, payload, digest, previous_hash, chain_hash, event_count),
                )
                conn.execute(
                    "INSERT INTO game_audit(game_id,version,state_hash,event_count) VALUES(?,?,?,?)",
                    (session.id, version, digest, event_count),
                )
            conn.execute(
                """INSERT INTO game_snapshots(game_id,version,payload,state_hash,chain_hash)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(game_id) DO UPDATE SET
                     version=excluded.version,payload=excluded.payload,
                     state_hash=excluded.state_hash,chain_hash=excluded.chain_hash,
                     updated_at=CURRENT_TIMESTAMP""",
                (session.id, version, payload, digest, chain_hash),
            )
        event_stream_id = f"{session.id}:round:{int(session.state.round_number)}"
        self.events.append_core_events(
            game_id=event_stream_id, events=session.state.events,
            profile_bundle_hash=str(getattr(session, "bundle_hash", "")), engine_version=__version__,
        )
        return digest

    @staticmethod
    def _decode(payload: bytes, expected: str) -> Any:
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise RuntimeError("stored game snapshot hash mismatch")
        return decode_payload(payload)

    def load(self, game_id: str) -> Any | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload,state_hash FROM game_snapshots WHERE game_id=?", (game_id,)
            ).fetchone()
        return None if row is None else self._decode(bytes(row["payload"]), str(row["state_hash"]))

    def load_version(self, game_id: str, version: int) -> Any | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload,state_hash FROM game_versions WHERE game_id=? AND version=?",
                (game_id, int(version)),
            ).fetchone()
        return None if row is None else self._decode(bytes(row["payload"]), str(row["state_hash"]))

    def list_versions(self, game_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT version,state_hash,previous_chain_hash,chain_hash,event_count,created_at
                   FROM game_versions WHERE game_id=? ORDER BY version""",
                (game_id,),
            ).fetchall()
        return [
            {
                "gameId": game_id,
                "version": int(row["version"]),
                "stateHash": row["state_hash"],
                "previousChainHash": row["previous_chain_hash"],
                "chainHash": row["chain_hash"],
                "eventCount": int(row["event_count"]),
                "createdAt": row["created_at"],
            }
            for row in rows
        ]

    def list_games(self, *, query: str = "", limit: int = 100) -> list[dict[str, Any]]:
        like = f"%{query.strip()}%"
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT game_id,version,state_hash,chain_hash,updated_at
                   FROM game_snapshots WHERE game_id LIKE ?
                   ORDER BY updated_at DESC LIMIT ?""",
                (like, max(1, min(5000, int(limit)))),
            ).fetchall()
        return [
            {
                "gameId": row["game_id"],
                "version": int(row["version"]),
                "stateHash": row["state_hash"],
                "chainHash": row["chain_hash"],
                "updatedAt": row["updated_at"],
            }
            for row in rows
        ]

    def verify(self, game_id: str) -> dict[str, Any]:
        versions = self.list_versions(game_id)
        previous: str | None = None
        for item in versions:
            session = self.load_version(game_id, item["version"])
            if session is None:
                return {"valid": False, "gameId": game_id, "error": "missing version payload"}
            payload, digest = self._encode(session)
            expected = self._chain(previous, digest, item["version"], item["eventCount"])
            if digest != item["stateHash"] or expected != item["chainHash"] or item["previousChainHash"] != previous:
                return {
                    "valid": False,
                    "gameId": game_id,
                    "failedVersion": item["version"],
                    "computedStateHash": digest,
                    "computedChainHash": expected,
                }
            previous = item["chainHash"]
        with self._connect() as conn:
            event_streams = [
                str(row[0])
                for row in conn.execute(
                    """SELECT game_id FROM game_event_heads_v7
                       WHERE game_id=? OR game_id LIKE ? ORDER BY game_id""",
                    (game_id, f"{game_id}:round:%"),
                ).fetchall()
            ]
        stream_results = [self.events.verify(stream_id) for stream_id in event_streams]
        event_verification = {
            "valid": bool(stream_results) and all(item.get("valid") for item in stream_results),
            "streamCount": len(stream_results),
            "streams": stream_results,
        }
        return {
            "valid": bool(event_verification.get("valid", False)),
            "gameId": game_id, "versionCount": len(versions), "headChainHash": previous,
            "eventVerification": event_verification,
        }

    def delete(self, game_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM game_snapshots WHERE game_id=?", (game_id,))
            conn.execute("DELETE FROM game_versions WHERE game_id=?", (game_id,))
            conn.execute("DELETE FROM game_audit WHERE game_id=?", (game_id,))

    def audit(self, game_id: str) -> list[dict[str, Any]]:
        return self.list_versions(game_id)
