from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from baloot_platform.identity.ownership import ownership_ids
from baloot_platform.storage.request_context import contextual_connection


class SQLiteSecurityStateStore:
    """Durable one-time security state for V20 development and tests."""
    def __init__(self, path: str | Path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True); self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS security_oidc_transactions_v20(
                  state TEXT PRIMARY KEY, payload_json TEXT NOT NULL, expires_at INTEGER NOT NULL, consumed_at INTEGER
                );
                CREATE TABLE IF NOT EXISTS security_websocket_tickets_v20(
                  nonce_hash TEXT PRIMARY KEY, subject_id TEXT NOT NULL, room_id TEXT NOT NULL,
                  audience TEXT NOT NULL, expires_at INTEGER NOT NULL, consumed_at INTEGER
                );
                CREATE TABLE IF NOT EXISTS security_passkey_challenges_v20(
                  challenge_hash TEXT PRIMARY KEY, subject_id TEXT NOT NULL, purpose TEXT NOT NULL,
                  payload_json TEXT NOT NULL, expires_at INTEGER NOT NULL, consumed_at INTEGER
                );
                CREATE TABLE IF NOT EXISTS security_passkey_credentials_v20(
                  credential_id TEXT PRIMARY KEY, subject_id TEXT NOT NULL, public_key_cose BLOB NOT NULL,
                  sign_count INTEGER NOT NULL, transports_json TEXT NOT NULL, created_at INTEGER NOT NULL,
                  last_used_at INTEGER, label TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS security_external_identities_v20(
                  provider_id TEXT NOT NULL, issuer TEXT NOT NULL, external_subject TEXT NOT NULL,
                  local_subject_id TEXT NOT NULL, claims_json TEXT NOT NULL, linked_at TEXT NOT NULL,
                  PRIMARY KEY(provider_id,issuer,external_subject), UNIQUE(local_subject_id,provider_id)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn=sqlite3.connect(self.path,timeout=30); conn.row_factory=sqlite3.Row; return conn

    def save_oidc(self, state: str, payload: dict[str, Any], expires_at: int) -> None:
        with self._lock,self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO security_oidc_transactions_v20(state,payload_json,expires_at,consumed_at) VALUES(?,?,?,NULL)",(state,json.dumps(payload,ensure_ascii=False,sort_keys=True),int(expires_at)))

    def consume_oidc(self, state: str, now: int) -> dict[str, Any] | None:
        with self._lock,self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row=conn.execute("SELECT * FROM security_oidc_transactions_v20 WHERE state=? AND consumed_at IS NULL",(state,)).fetchone()
            if row is None or int(row["expires_at"])<=int(now): conn.execute("ROLLBACK"); return None
            conn.execute("UPDATE security_oidc_transactions_v20 SET consumed_at=? WHERE state=? AND consumed_at IS NULL",(int(now),state)); conn.commit()
        return json.loads(row["payload_json"])

    def save_ticket(self, nonce_hash: str, subject_id: str, room_id: str, audience: str, expires_at: int) -> None:
        with self._lock,self._connect() as conn:
            conn.execute("INSERT INTO security_websocket_tickets_v20 VALUES(?,?,?,?,?,NULL)",(nonce_hash,subject_id,room_id,audience,int(expires_at)))

    def consume_ticket(self, nonce_hash: str, now: int) -> dict[str, Any] | None:
        with self._lock,self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row=conn.execute("SELECT * FROM security_websocket_tickets_v20 WHERE nonce_hash=? AND consumed_at IS NULL",(nonce_hash,)).fetchone()
            if row is None or int(row["expires_at"])<=int(now): conn.execute("ROLLBACK"); return None
            cursor=conn.execute("UPDATE security_websocket_tickets_v20 SET consumed_at=? WHERE nonce_hash=? AND consumed_at IS NULL",(int(now),nonce_hash))
            if cursor.rowcount!=1: conn.execute("ROLLBACK"); return None
            conn.commit()
        return dict(row)

    def save_challenge(self, challenge_hash: str, subject_id: str, purpose: str, payload: dict[str,Any], expires_at: int) -> None:
        with self._lock,self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO security_passkey_challenges_v20 VALUES(?,?,?,?,?,NULL)",(challenge_hash,subject_id,purpose,json.dumps(payload,ensure_ascii=False,sort_keys=True),int(expires_at)))

    def consume_challenge(self, challenge_hash: str, subject_id: str, purpose: str, now: int) -> dict[str,Any] | None:
        with self._lock,self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row=conn.execute("SELECT * FROM security_passkey_challenges_v20 WHERE challenge_hash=? AND subject_id IN (?,?) AND purpose=? AND consumed_at IS NULL",(challenge_hash,*ownership_ids(subject_id),purpose)).fetchone()
            if row is None or int(row["expires_at"])<=int(now): conn.execute("ROLLBACK"); return None
            conn.execute("UPDATE security_passkey_challenges_v20 SET consumed_at=? WHERE challenge_hash=? AND consumed_at IS NULL",(int(now),challenge_hash)); conn.commit()
        return json.loads(row["payload_json"])

    def put_credential(self, item: dict[str,Any]) -> None:
        with self._lock,self._connect() as conn:
            conn.execute("""INSERT INTO security_passkey_credentials_v20(credential_id,subject_id,public_key_cose,sign_count,transports_json,created_at,last_used_at,label)
              VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(credential_id) DO UPDATE SET sign_count=excluded.sign_count,last_used_at=excluded.last_used_at,label=excluded.label""",
              (item["credential_id"],item["subject_id"],item["public_key_cose"],int(item["sign_count"]),json.dumps(list(item.get("transports",()))),int(item["created_at"]),item.get("last_used_at"),item.get("label","Passkey")))

    def get_credential(self, credential_id: str) -> dict[str,Any] | None:
        with self._connect() as conn: row=conn.execute("SELECT * FROM security_passkey_credentials_v20 WHERE credential_id=?",(credential_id,)).fetchone()
        if not row:return None
        d=dict(row); d["transports"]=tuple(json.loads(d.pop("transports_json"))); return d

    def list_credentials(self, subject_id: str) -> list[dict[str,Any]]:
        with self._connect() as conn: rows=conn.execute("SELECT * FROM security_passkey_credentials_v20 WHERE subject_id IN (?,?) ORDER BY created_at",ownership_ids(subject_id)).fetchall()
        result=[]
        for row in rows:
            d=dict(row); d["transports"]=tuple(json.loads(d.pop("transports_json"))); result.append(d)
        return result

    def get_external_identity(self, provider_id: str, issuer: str, external_subject: str) -> dict[str,Any] | None:
        with self._connect() as conn: row=conn.execute("SELECT * FROM security_external_identities_v20 WHERE provider_id=? AND issuer=? AND external_subject=?",(provider_id,issuer,external_subject)).fetchone()
        if not row:return None
        d=dict(row); d["claims"]=json.loads(d.pop("claims_json")); return d

    def link_external_identity(self, *, provider_id: str, issuer: str, external_subject: str, local_subject_id: str, claims: dict[str,Any], linked_at: str) -> None:
        with self._lock,self._connect() as conn:
            conn.execute("INSERT INTO security_external_identities_v20 VALUES(?,?,?,?,?,?) ON CONFLICT(provider_id,issuer,external_subject) DO UPDATE SET local_subject_id=excluded.local_subject_id,claims_json=excluded.claims_json",(provider_id,issuer,external_subject,local_subject_id,json.dumps(claims,ensure_ascii=False,sort_keys=True),linked_at))

    def close(self) -> None: return None


class PostgresSecurityStateStore:
    def __init__(self, dsn: str):
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('Install with: pip install "baloot-platform[postgres]"') from exc
        self.pool=ConnectionPool(dsn,min_size=1,max_size=16,open=True)

    def save_oidc(self,state,payload,expires_at):
        with contextual_connection(self.pool) as conn: conn.execute("INSERT INTO security.oidc_transactions(state,payload_json,expires_at) VALUES(%s,%s::jsonb,to_timestamp(%s)) ON CONFLICT(state) DO UPDATE SET payload_json=excluded.payload_json,expires_at=excluded.expires_at,consumed_at=NULL",(state,json.dumps(payload,ensure_ascii=False),expires_at))
    def consume_oidc(self,state,now):
        with contextual_connection(self.pool) as conn:
            row=conn.execute("UPDATE security.oidc_transactions SET consumed_at=to_timestamp(%s) WHERE state=%s AND consumed_at IS NULL AND expires_at>to_timestamp(%s) RETURNING payload_json",(now,state,now)).fetchone()
        return dict(row[0]) if row else None
    def save_ticket(self,nonce_hash,subject_id,room_id,audience,expires_at):
        with contextual_connection(self.pool) as conn: conn.execute("INSERT INTO security.websocket_tickets(nonce_hash,subject_id,room_id,audience,expires_at) VALUES(%s,%s,%s,%s,to_timestamp(%s))",(nonce_hash,subject_id,room_id,audience,expires_at))
    def consume_ticket(self,nonce_hash,now):
        with contextual_connection(self.pool) as conn: row=conn.execute("UPDATE security.websocket_tickets SET consumed_at=to_timestamp(%s) WHERE nonce_hash=%s AND consumed_at IS NULL AND expires_at>to_timestamp(%s) RETURNING subject_id,room_id,audience,extract(epoch from expires_at)::bigint",(now,nonce_hash,now)).fetchone()
        return {"subject_id":row[0],"room_id":row[1],"audience":row[2],"expires_at":int(row[3])} if row else None
    def save_challenge(self,challenge_hash,subject_id,purpose,payload,expires_at):
        with contextual_connection(self.pool) as conn: conn.execute("INSERT INTO security.passkey_challenges(challenge_hash,subject_id,purpose,payload_json,expires_at) VALUES(%s,%s,%s,%s::jsonb,to_timestamp(%s)) ON CONFLICT(challenge_hash) DO UPDATE SET payload_json=excluded.payload_json,expires_at=excluded.expires_at,consumed_at=NULL",(challenge_hash,subject_id,purpose,json.dumps(payload,ensure_ascii=False),expires_at))
    def consume_challenge(self,challenge_hash,subject_id,purpose,now):
        with contextual_connection(self.pool) as conn: row=conn.execute("UPDATE security.passkey_challenges SET consumed_at=to_timestamp(%s) WHERE challenge_hash=%s AND subject_id IN (%s,%s) AND purpose=%s AND consumed_at IS NULL AND expires_at>to_timestamp(%s) RETURNING payload_json",(now,challenge_hash,*ownership_ids(subject_id),purpose,now)).fetchone()
        return dict(row[0]) if row else None
    def put_credential(self,item):
        with contextual_connection(self.pool) as conn: conn.execute("INSERT INTO security.passkey_credentials(credential_id,subject_id,public_key_cose,sign_count,transports_json,created_at,last_used_at,label) VALUES(%s,%s,%s,%s,%s::jsonb,to_timestamp(%s),CASE WHEN %s IS NULL THEN NULL ELSE to_timestamp(%s) END,%s) ON CONFLICT(credential_id) DO UPDATE SET sign_count=excluded.sign_count,last_used_at=excluded.last_used_at,label=excluded.label",(item["credential_id"],item["subject_id"],item["public_key_cose"],item["sign_count"],json.dumps(list(item.get("transports",()))),item["created_at"],item.get("last_used_at"),item.get("last_used_at"),item.get("label","Passkey")))
    def get_credential(self,credential_id):
        with contextual_connection(self.pool) as conn: row=conn.execute("SELECT credential_id,subject_id,public_key_cose,sign_count,transports_json,extract(epoch from created_at)::bigint,extract(epoch from last_used_at)::bigint,label FROM security.passkey_credentials WHERE credential_id=%s",(credential_id,)).fetchone()
        return {"credential_id":row[0],"subject_id":row[1],"public_key_cose":bytes(row[2]),"sign_count":int(row[3]),"transports":tuple(row[4]),"created_at":int(row[5]),"last_used_at":int(row[6]) if row[6] is not None else None,"label":row[7]} if row else None
    def list_credentials(self,subject_id):
        with contextual_connection(self.pool) as conn: rows=conn.execute("SELECT credential_id,subject_id,public_key_cose,sign_count,transports_json,extract(epoch from created_at)::bigint,extract(epoch from last_used_at)::bigint,label FROM security.passkey_credentials WHERE subject_id IN (%s,%s) ORDER BY created_at",ownership_ids(subject_id)).fetchall()
        return [{"credential_id":r[0],"subject_id":r[1],"public_key_cose":bytes(r[2]),"sign_count":int(r[3]),"transports":tuple(r[4]),"created_at":int(r[5]),"last_used_at":int(r[6]) if r[6] is not None else None,"label":r[7]} for r in rows]
    def get_external_identity(self,provider_id,issuer,external_subject):
        with contextual_connection(self.pool) as conn: row=conn.execute("SELECT local_subject_id,claims_json,linked_at FROM security.external_identities WHERE provider_id=%s AND issuer=%s AND external_subject=%s",(provider_id,issuer,external_subject)).fetchone()
        return {"local_subject_id":row[0],"claims":dict(row[1]),"linked_at":str(row[2])} if row else None
    def link_external_identity(self,*,provider_id,issuer,external_subject,local_subject_id,claims,linked_at):
        with contextual_connection(self.pool) as conn: conn.execute("INSERT INTO security.external_identities(provider_id,issuer,external_subject,local_subject_id,claims_json,linked_at) VALUES(%s,%s,%s,%s,%s::jsonb,%s::timestamptz) ON CONFLICT(provider_id,issuer,external_subject) DO UPDATE SET local_subject_id=excluded.local_subject_id,claims_json=excluded.claims_json",(provider_id,issuer,external_subject,local_subject_id,json.dumps(claims,ensure_ascii=False),linked_at))
    def close(self): self.pool.close()
