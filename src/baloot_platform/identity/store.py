from __future__ import annotations

import sqlite3
import threading
from datetime import timezone
from pathlib import Path
from typing import Any

from .ownership import ownership_ids


class _IdentityLists:
    def _read_page_rows(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def search_subjects(
        self, query: str, *, limit: int, after: tuple[str, ...] = ()
    ) -> list[dict[str, Any]]:
        needle = f"%{query}%"
        select = """SELECT subject_id,kind,username,display_name,created_at
                    FROM identity_subjects_v8
                    WHERE disabled_at IS NULL AND (display_name LIKE ? OR username LIKE ?)"""
        order = "ORDER BY kind DESC,display_name,subject_id LIMIT ?"
        if not after:
            return self._read_page_rows(f"{select} {order}", (needle, needle, limit))
        kind, display_name, subject_id = after
        return self._read_page_rows(
            f"""SELECT * FROM (
                  SELECT * FROM ({select} AND kind=? AND (display_name,subject_id)>(?,?)
                                 {order}) AS same_kind
                  UNION ALL
                  SELECT * FROM ({select} AND kind<? {order}) AS later_kinds
                ) AS matches {order}""",
            (needle, needle, kind, display_name, subject_id, limit, needle, needle, kind, limit, limit),
        )

    def list_friends(
        self, subject_id: str, *, limit: int, after: tuple[str, ...] = ()
    ) -> list[dict[str, Any]]:
        owners = ownership_ids(subject_id)
        branches = []
        params: list[Any] = []
        for owner in dict.fromkeys(owners):
            for owner_column, friend_column in (("subject_low", "subject_high"), ("subject_high", "subject_low")):
                seek = f"AND {friend_column}>?" if after else ""
                branches.append(
                    f"""SELECT friend_id FROM (
                          SELECT {friend_column} AS friend_id FROM identity_friendships_v8
                          WHERE {owner_column}=? AND {friend_column} NOT IN (?,?) {seek}
                          ORDER BY {friend_column} LIMIT ?
                        ) AS neighbors_{len(branches)}"""
                )
                params.extend((owner, *owners, *after, limit))
        # Each branch contains unique neighbors. Deduplicate the bounded union across aliases.
        neighbors = " UNION ".join(branches)
        return self._read_page_rows(
            f"""SELECT u.subject_id,u.kind,u.username,u.display_name,u.created_at
                FROM ({neighbors} ORDER BY friend_id LIMIT ?) AS page
                JOIN identity_subjects_v8 u ON u.subject_id=page.friend_id
                ORDER BY page.friend_id""",
            (*params, limit),
        )

    def list_friend_requests(
        self, subject_id: str, *, limit: int, after: tuple[str, ...] = ()
    ) -> list[dict[str, Any]]:
        seek = "AND (created_at,request_id)<(?,?)" if after else ""
        order = "ORDER BY created_at DESC,request_id DESC LIMIT ?"
        branches = []
        params: list[Any] = []
        for owner in dict.fromkeys(ownership_ids(subject_id)):
            branches.append(
                f"""SELECT * FROM (
                      SELECT request_id,sender_id,recipient_id,status,created_at,responded_at
                      FROM identity_friend_requests_v8
                      WHERE recipient_id=? AND status='pending' {seek} {order}
                    ) AS requests_{len(branches)}"""
            )
            params.extend((owner, *after, limit))
        requests = " UNION ALL ".join(branches)
        return self._read_page_rows(
            f"""SELECT r.*,u.display_name AS sender_display_name,u.username AS sender_username
                FROM ({requests} {order}) AS r
                JOIN identity_subjects_v8 u ON u.subject_id=r.sender_id
                ORDER BY r.created_at DESC,r.request_id DESC""",
            (*params, limit),
        )


class SQLiteIdentityStore(_IdentityLists):
    """Durable development identity store.

    Password material is stored only as scrypt-derived hashes. Session tokens are
    stored only as SHA-256 hashes. Production deployments may replace this store
    through the identity provider interface without changing game or Sakka code.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS identity_subjects_v8(
                  subject_id TEXT PRIMARY KEY,
                  kind TEXT NOT NULL CHECK(kind IN ('guest','local','oidc','passkey','service')),
                  username TEXT UNIQUE,
                  display_name TEXT NOT NULL,
                  password_salt BLOB,
                  password_hash BLOB,
                  created_at TEXT NOT NULL,
                  disabled_at TEXT
                );
                CREATE TABLE IF NOT EXISTS identity_sessions_v8(
                  session_id TEXT PRIMARY KEY,
                  subject_id TEXT NOT NULL REFERENCES identity_subjects_v8(subject_id) ON DELETE CASCADE,
                  token_hash TEXT NOT NULL UNIQUE,
                  csrf_token TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  expires_at TEXT NOT NULL,
                  last_seen_at TEXT NOT NULL,
                  user_agent_hash TEXT,
                  revoked_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_identity_sessions_v8_subject
                  ON identity_sessions_v8(subject_id,expires_at);
                CREATE TABLE IF NOT EXISTS identity_friendships_v8(
                  subject_low TEXT NOT NULL REFERENCES identity_subjects_v8(subject_id) ON DELETE CASCADE,
                  subject_high TEXT NOT NULL REFERENCES identity_subjects_v8(subject_id) ON DELETE CASCADE,
                  created_at TEXT NOT NULL,
                  PRIMARY KEY(subject_low,subject_high),
                  CHECK(subject_low < subject_high)
                );
                CREATE TABLE IF NOT EXISTS identity_friend_requests_v8(
                  request_id TEXT PRIMARY KEY,
                  sender_id TEXT NOT NULL REFERENCES identity_subjects_v8(subject_id) ON DELETE CASCADE,
                  recipient_id TEXT NOT NULL REFERENCES identity_subjects_v8(subject_id) ON DELETE CASCADE,
                  status TEXT NOT NULL CHECK(status IN ('pending','accepted','declined','cancelled')),
                  created_at TEXT NOT NULL,
                  responded_at TEXT,
                  UNIQUE(sender_id,recipient_id,status)
                );
                CREATE INDEX IF NOT EXISTS idx_identity_active_search
                  ON identity_subjects_v8(kind DESC,display_name,subject_id) WHERE disabled_at IS NULL;
                CREATE INDEX IF NOT EXISTS idx_identity_friendships_high_low
                  ON identity_friendships_v8(subject_high,subject_low);
                CREATE INDEX IF NOT EXISTS idx_identity_pending_requests
                  ON identity_friend_requests_v8(recipient_id,created_at DESC,request_id DESC)
                  WHERE status='pending';
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def insert_subject(self, *, subject_id: str, kind: str, username: str | None,
                       display_name: str, password_salt: bytes | None,
                       password_hash: bytes | None, created_at: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO identity_subjects_v8(
                     subject_id,kind,username,display_name,password_salt,password_hash,created_at
                   ) VALUES(?,?,?,?,?,?,?)""",
                (subject_id, kind, username, display_name, password_salt, password_hash, created_at),
            )

    def get_subject(self, subject_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM identity_subjects_v8 WHERE subject_id=?", (subject_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_subject_by_username(self, username: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM identity_subjects_v8 WHERE lower(username)=lower(?)", (username,)
            ).fetchone()
        return dict(row) if row else None

    def _read_page_rows(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def upgrade_guest(self, *, subject_id: str, username: str, display_name: str,
                      password_salt: bytes, password_hash: bytes) -> None:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """UPDATE identity_subjects_v8 SET kind='local',username=?,display_name=?,
                           password_salt=?,password_hash=?
                   WHERE subject_id=? AND kind='guest' AND disabled_at IS NULL""",
                (username, display_name, password_salt, password_hash, subject_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(subject_id)

    def insert_session(self, row: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO identity_sessions_v8(
                     session_id,subject_id,token_hash,csrf_token,created_at,expires_at,last_seen_at,user_agent_hash
                   ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    row["session_id"], row["subject_id"], row["token_hash"], row["csrf_token"],
                    row["created_at"], row["expires_at"], row["last_seen_at"], row.get("user_agent_hash"),
                ),
            )

    def get_session_by_hash(self, token_hash: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT s.*,u.kind,u.username,u.display_name,u.created_at AS subject_created_at,u.disabled_at
                   FROM identity_sessions_v8 s JOIN identity_subjects_v8 u ON u.subject_id=s.subject_id
                   WHERE s.token_hash=?""",
                (token_hash,),
            ).fetchone()
        return dict(row) if row else None

    def touch_session(self, session_id: str, now: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE identity_sessions_v8 SET last_seen_at=? WHERE session_id=? AND revoked_at IS NULL",
                (now, session_id),
            )

    def revoke_session(self, session_id: str, now: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE identity_sessions_v8 SET revoked_at=? WHERE session_id=? AND revoked_at IS NULL",
                (now, session_id),
            )

    def create_friend_request(self, *, request_id: str, sender_id: str,
                              recipient_id: str, created_at: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO identity_friend_requests_v8(
                     request_id,sender_id,recipient_id,status,created_at
                   ) VALUES(?,?,?,'pending',?)""",
                (request_id, sender_id, recipient_id, created_at),
            )

    def respond_friend_request(self, *, request_id: str, recipient_id: str,
                               accept: bool, responded_at: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT * FROM identity_friend_requests_v8
                   WHERE request_id=? AND recipient_id IN (?,?) AND status='pending'""",
                (request_id, *ownership_ids(recipient_id)),
            ).fetchone()
            if row is None:
                raise KeyError(request_id)
            status = "accepted" if accept else "declined"
            conn.execute(
                "UPDATE identity_friend_requests_v8 SET status=?,responded_at=? WHERE request_id=?",
                (status, responded_at, request_id),
            )
            if accept:
                low, high = sorted((str(row["sender_id"]), str(row["recipient_id"])))
                conn.execute(
                    "INSERT OR IGNORE INTO identity_friendships_v8(subject_low,subject_high,created_at) VALUES(?,?,?)",
                    (low, high, responded_at),
                )
            conn.commit()

class PostgresIdentityStore(_IdentityLists):
    """PostgreSQL production identity store with the same service contract."""

    def __init__(self, dsn: str):
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('Install with: pip install "baloot-platform[postgres]"') from exc
        self.pool = ConnectionPool(dsn, min_size=1, max_size=16, open=True)

    @staticmethod
    def _dict(row, columns):
        return dict(zip(columns, row, strict=True))

    def insert_subject(self, *, subject_id: str, kind: str, username: str | None,
                       display_name: str, password_salt: bytes | None,
                       password_hash: bytes | None, created_at: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO identity.subjects(
                     subject_id,kind,username,display_name,password_salt,password_hash,created_at
                   ) VALUES(%s,%s,%s,%s,%s,%s,%s::timestamptz)""",
                (subject_id,kind,username,display_name,password_salt,password_hash,created_at),
            )

    def get_subject(self, subject_id: str) -> dict[str, Any] | None:
        cols=("subject_id","kind","username","display_name","password_salt","password_hash","created_at","disabled_at")
        with self.pool.connection() as conn:
            row=conn.execute(
                "SELECT subject_id,kind,username,display_name,password_salt,password_hash,created_at,disabled_at FROM identity.subjects WHERE subject_id=%s",
                (subject_id,),
            ).fetchone()
        if not row:return None
        value=self._dict(row,cols); value["created_at"]=value["created_at"].isoformat(); value["disabled_at"]=value["disabled_at"].isoformat() if value["disabled_at"] else None
        return value

    def get_subject_by_username(self, username: str) -> dict[str, Any] | None:
        cols=("subject_id","kind","username","display_name","password_salt","password_hash","created_at","disabled_at")
        with self.pool.connection() as conn:
            row=conn.execute(
                "SELECT subject_id,kind,username,display_name,password_salt,password_hash,created_at,disabled_at FROM identity.subjects WHERE lower(username)=lower(%s)",
                (username,),
            ).fetchone()
        if not row:return None
        value=self._dict(row,cols); value["created_at"]=value["created_at"].isoformat(); value["disabled_at"]=value["disabled_at"].isoformat() if value["disabled_at"] else None
        return value

    @staticmethod
    def _postgres_page_sql(sql: str) -> str:
        return (
            sql.replace("identity_subjects_v8", "identity.subjects")
            .replace("identity_friendships_v8", "identity.friendships")
            .replace("identity_friend_requests_v8", "identity.friend_requests")
            .replace(" LIKE ", " ILIKE ")
            .replace("?", "%s")
        )

    def _read_page_rows(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        from psycopg.rows import dict_row

        with self.pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                rows = cursor.execute(self._postgres_page_sql(sql), params).fetchall()
        for row in rows:
            row["created_at"] = row["created_at"].astimezone(timezone.utc).isoformat()
            # The PostgreSQL pending-request response omits the response timestamp.
            row.pop("responded_at", None)
        return rows

    def upgrade_guest(self, *, subject_id: str, username: str, display_name: str,
                      password_salt: bytes, password_hash: bytes) -> None:
        with self.pool.connection() as conn:
            row = conn.execute(
                """UPDATE identity.subjects SET kind='local',username=%s,display_name=%s,
                           password_salt=%s,password_hash=%s
                   WHERE subject_id=%s AND kind='guest' AND disabled_at IS NULL RETURNING subject_id""",
                (username, display_name, password_salt, password_hash, subject_id),
            ).fetchone()
            if row is None:
                raise KeyError(subject_id)

    def insert_session(self, row: dict[str, Any]) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO identity.sessions(
                     session_id,subject_id,token_hash,csrf_token,created_at,expires_at,last_seen_at,user_agent_hash
                   ) VALUES(%s,%s,%s,%s,%s::timestamptz,%s::timestamptz,%s::timestamptz,%s)""",
                (row["session_id"],row["subject_id"],row["token_hash"],row["csrf_token"],row["created_at"],row["expires_at"],row["last_seen_at"],row.get("user_agent_hash")),
            )

    def get_session_by_hash(self, token_hash: str) -> dict[str, Any] | None:
        cols=("session_id","subject_id","token_hash","csrf_token","created_at","expires_at","last_seen_at","user_agent_hash","revoked_at","kind","username","display_name","subject_created_at","disabled_at")
        with self.pool.connection() as conn:
            row=conn.execute(
                """SELECT s.session_id,s.subject_id,s.token_hash,s.csrf_token,s.created_at,s.expires_at,
                          s.last_seen_at,s.user_agent_hash,s.revoked_at,u.kind,u.username,u.display_name,
                          u.created_at AS subject_created_at,u.disabled_at
                   FROM identity.sessions s JOIN identity.subjects u ON u.subject_id=s.subject_id
                   WHERE s.token_hash=%s""",(token_hash,),
            ).fetchone()
        if not row:return None
        value=self._dict(row,cols)
        for key in ("created_at","expires_at","last_seen_at","revoked_at","subject_created_at","disabled_at"):
            if value.get(key) is not None:value[key]=value[key].isoformat()
        return value

    def touch_session(self, session_id: str, now: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("UPDATE identity.sessions SET last_seen_at=%s::timestamptz WHERE session_id=%s AND revoked_at IS NULL",(now,session_id))

    def revoke_session(self, session_id: str, now: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("UPDATE identity.sessions SET revoked_at=%s::timestamptz WHERE session_id=%s AND revoked_at IS NULL",(now,session_id))

    def create_friend_request(self, *, request_id: str, sender_id: str, recipient_id: str, created_at: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO identity.friend_requests(request_id,sender_id,recipient_id,status,created_at)
                   VALUES(%s,%s,%s,'pending',%s::timestamptz)""",
                (request_id,sender_id,recipient_id,created_at),
            )

    def respond_friend_request(self, *, request_id: str, recipient_id: str, accept: bool, responded_at: str) -> None:
        with self.pool.connection() as conn, conn.transaction():
            row=conn.execute(
                """SELECT sender_id,recipient_id FROM identity.friend_requests
                   WHERE request_id=%s AND recipient_id IN (%s,%s) AND status='pending' FOR UPDATE""",
                (request_id, *ownership_ids(recipient_id)),
            ).fetchone()
            if not row:raise KeyError(request_id)
            status='accepted' if accept else 'declined'
            conn.execute("UPDATE identity.friend_requests SET status=%s,responded_at=%s::timestamptz WHERE request_id=%s",(status,responded_at,request_id))
            if accept:
                low,high=sorted((str(row[0]),str(row[1])))
                conn.execute("INSERT INTO identity.friendships(subject_low,subject_high,created_at) VALUES(%s,%s,%s::timestamptz) ON CONFLICT DO NOTHING",(low,high,responded_at))
