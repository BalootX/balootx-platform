from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ownership import ownership_ids


class _IdentityLists:
    subjects: str
    friendships: str
    requests: str
    placeholder: str
    like: str
    legacy_pending_request_fields = ""

    def _read_page_rows(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def search_subjects(
        self, query: str, *, limit: int = 20, exact_username: str | None = None,
        maximum_results: int | None = None, after: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        p = self.placeholder
        fields = "subject_id,kind,username,display_name,created_at"
        query = query.strip()
        if maximum_results is None:
            needle = f"%{query}%"
            return self._read_page_rows(
                f"""SELECT {fields} FROM {self.subjects}
                    WHERE disabled_at IS NULL AND (display_name {self.like} {p} OR username {self.like} {p})
                    ORDER BY kind DESC,display_name LIMIT {p}""",
                (needle, needle, max(1, min(100, int(limit)))),
            )
        needle = "%" + query.translate(str.maketrans({"!": "!!", "%": "!%", "_": "!_"})) + "%"
        seek = f"WHERE (search_rank,subject_id)>({p},{p})" if after else ""
        # Cap the candidate set before applying the cursor, including for a forged anchor.
        return self._read_page_rows(
            f"""WITH exact AS (
                  SELECT '0' AS search_rank,{fields} FROM {self.subjects}
                  WHERE disabled_at IS NULL AND username = {p}
                  ORDER BY subject_id LIMIT 1
                ), matches AS (
                  SELECT '1' AS search_rank,{fields}
                  FROM {self.subjects}
                  WHERE disabled_at IS NULL AND
                    (display_name {self.like} {p} ESCAPE '!' OR username {self.like} {p} ESCAPE '!')
                    AND subject_id NOT IN (SELECT subject_id FROM exact)
                  ORDER BY subject_id LIMIT ({p} - (SELECT count(*) FROM exact))
                )
                SELECT * FROM (SELECT * FROM exact UNION ALL SELECT * FROM matches) AS candidates
                {seek} ORDER BY search_rank,subject_id LIMIT {p}""",
            (exact_username, needle, needle, maximum_results, *after, limit),
        )

    def _friend_neighbors(
        self, subject_id: str, *, limit: int | None = None, after: tuple[str, ...] = ()
    ) -> tuple[str, tuple[Any, ...]]:
        p = self.placeholder
        owners = ownership_ids(subject_id)
        branches = []
        params: list[Any] = []
        for owner in dict.fromkeys(owners):
            for owner_column, friend_column in (("subject_low", "subject_high"), ("subject_high", "subject_low")):
                seek = f"AND {friend_column}>{p}" if after else ""
                branch = (
                    f"SELECT {friend_column} AS friend_id FROM {self.friendships} "
                    f"WHERE {owner_column}={p} AND {friend_column} NOT IN ({p},{p}) {seek}"
                )
                params.extend((owner, *owners, *after))
                if limit is not None:
                    branch = (
                        f"SELECT friend_id FROM ({branch} ORDER BY {friend_column} LIMIT {p}) "
                        f"AS neighbors_{len(branches)}"
                    )
                    params.append(limit)
                branches.append(branch)
        # Both ownership aliases may link to the same friend.
        neighbors = " UNION ".join(branches)
        if limit is not None:
            neighbors += f" ORDER BY friend_id LIMIT {p}"
            params.append(limit)
        return neighbors, tuple(params)

    def list_friends(
        self, subject_id: str, *, limit: int | None = None, after: tuple[str, ...] = ()
    ) -> list[dict[str, Any]]:
        neighbors, params = self._friend_neighbors(subject_id, limit=limit, after=after)
        order = "u.display_name" if limit is None else "page.friend_id"
        return self._read_page_rows(
            f"""SELECT u.subject_id,u.kind,u.username,u.display_name,u.created_at
                FROM ({neighbors}) AS page
                JOIN {self.subjects} u ON u.subject_id=page.friend_id
                ORDER BY {order}""",
            params,
        )

    def friend_ids(self, subject_id: str) -> list[str]:
        sql, params = self._friend_neighbors(subject_id)
        return [row["friend_id"] for row in self._read_page_rows(sql, params)]

    def list_friend_requests(
        self, subject_id: str, *, limit: int | None = None, after: tuple[str, ...] = ()
    ) -> list[dict[str, Any]]:
        p = self.placeholder
        fields = "request_id,sender_id,recipient_id,status,created_at"
        order = f"ORDER BY created_at DESC,request_id DESC LIMIT {p}"
        if limit is None:
            requests = (
                f"SELECT {fields}{self.legacy_pending_request_fields} FROM {self.requests} "
                f"WHERE recipient_id IN ({p},{p}) AND status='pending'"
            )
            params = list(ownership_ids(subject_id))
        else:
            seek = f"AND (created_at,request_id)<({p},{p})" if after else ""
            branches = []
            params = []
            for owner in dict.fromkeys(ownership_ids(subject_id)):
                branches.append(
                    f"""SELECT * FROM (
                          SELECT {fields} FROM {self.requests}
                          WHERE recipient_id={p} AND status='pending' {seek} {order}
                        ) AS requests_{len(branches)}"""
                )
                params.extend((owner, *after, limit))
            requests = " UNION ALL ".join(branches) + f" {order}"
            params.append(limit)
        request_order = "r.created_at DESC" if limit is None else "r.created_at DESC,r.request_id DESC"
        return self._read_page_rows(
            f"""SELECT r.*,u.display_name AS sender_display_name,u.username AS sender_username
                FROM ({requests}) AS r
                JOIN {self.subjects} u ON u.subject_id=r.sender_id
                ORDER BY {request_order}""",
            tuple(params),
        )


class SQLiteIdentityStore(_IdentityLists):
    """Durable development identity store.

    Password material is stored only as scrypt-derived hashes. Session tokens are
    stored only as SHA-256 hashes. Production deployments may replace this store
    through the identity provider interface without changing game or Sakka code.
    """


    subjects = "identity_subjects_v8"
    friendships = "identity_friendships_v8"
    requests = "identity_friend_requests_v8"
    placeholder = "?"
    like = "LIKE"
    legacy_pending_request_fields = ",responded_at"

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


    subjects = "identity.subjects"
    friendships = "identity.friendships"
    requests = "identity.friend_requests"
    placeholder = "%s"
    like = "ILIKE"

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

    def _read_page_rows(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        from psycopg.rows import dict_row

        with self.pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                rows = cursor.execute(sql, params).fetchall()
        for row in rows:
            for column, value in row.items():
                if isinstance(value, datetime):
                    if value.tzinfo is None:
                        value = value.replace(tzinfo=timezone.utc)
                    row[column] = value.astimezone(timezone.utc).isoformat()
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
