from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from baloot_platform.storage.request_context import contextual_connection

from .models import BundleRevision, ProfileRevision, canonical, digest, utc_now
from .schedule import normalize_schedule


_UNSET = object()

_POSTGRES_PROFILE_COLUMNS = """
revision_id,family,profile_id,version,scope,scope_key,tenant_id,environment,
status,payload,payload_sha256,sequence,effective_from,effective_until,
local_schedule,rollout_percentage,rollout_seed,created_by,created_at,updated_at,published_by,
published_at,supersedes_revision_id,etag
""".replace("\n", "")

_POSTGRES_BUNDLE_COLUMNS = """
revision_id,bundle_id,version,mode,scope,scope_key,tenant_id,environment,
status,components,content_sha256,sequence,effective_from,effective_until,
local_schedule,rollout_percentage,rollout_seed,created_by,created_at,updated_at,published_by,
published_at,supersedes_revision_id,etag
""".replace("\n", "")

_POSTGRES_EVENT_COLUMNS = (
    "event_id,aggregate_type,aggregate_id,sequence,event_type,actor_id,"
    "payload,previous_hash,event_hash,created_at"
)


def _loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def _aggregate_key(kind: str, identity: tuple[Any, ...]) -> str:
    return f"{kind}:{_aggregate_identity(identity)}"


def _aggregate_identity(identity: tuple[Any, ...]) -> str:
    # JSON preserves tuple boundaries, nulls, empty strings, and embedded
    # punctuation.  A colon-joined identity cannot distinguish those values.
    return canonical(list(identity))


def _timestamp_text(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _optional_timestamp_text(value: Any) -> str | None:
    return None if value is None else _timestamp_text(value)


def _require_same_schedule(
    revision: ProfileRevision | BundleRevision,
    schedule: tuple[str | None, str | None, dict[str, str] | None],
) -> None:
    start, end, local = schedule
    saved_local = revision.local_schedule
    if local is not None and saved_local is not None:
        local = {**local, "zoneDataVersion": saved_local.get("zoneDataVersion", "")}
    for field, saved, requested in zip(
        ("effectiveFrom", "effectiveUntil", "localSchedule"),
        (revision.effective_from, revision.effective_until, saved_local),
        (start, end, local),
        strict=True,
    ):
        if saved != requested:
            raise ValueError(
                f"{field} differs from existing revision {revision.revision_id}. "
                "Create a revision with different content to change its schedule"
            )


def _row_mapping(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return dict(row)
    except (TypeError, ValueError) as exc:
        raise TypeError("configuration store rows must be named mappings") from exc


def _event_from_row(row: Any, *, payload_column: str) -> dict[str, Any]:
    value = _row_mapping(row)
    return {
        "eventId": value["event_id"],
        "aggregateType": value["aggregate_type"],
        "aggregateId": value["aggregate_id"],
        "sequence": int(value["sequence"]),
        "eventType": value["event_type"],
        "actorId": value["actor_id"],
        "payload": _loads(value[payload_column], {}),
        "previousHash": value["previous_hash"],
        "eventHash": value["event_hash"],
        "createdAt": _timestamp_text(value["created_at"]),
    }


def _verify_event_values(values: list[dict[str, Any]]) -> dict[str, Any]:
    previous = ""
    issues: list[str] = []
    for expected, event in enumerate(values, start=1):
        body = {
            key: event[key]
            for key in (
                "aggregateType",
                "aggregateId",
                "sequence",
                "eventType",
                "actorId",
                "payload",
                "previousHash",
                "createdAt",
            )
        }
        if event["sequence"] != expected:
            issues.append(f"sequence:{expected}")
        if event["previousHash"] != previous:
            issues.append(f"previous:{expected}")
        if digest(body) != event["eventHash"]:
            issues.append(f"hash:{expected}")
        previous = event["eventHash"]
    return {
        "valid": not issues,
        "issues": issues,
        "eventCount": len(values),
        "eventHead": previous,
    }


def _validate_component_revisions(
    component_revisions: Mapping[str, str],
    rows: list[ProfileRevision],
    *,
    allowed_statuses: tuple[str, ...],
) -> None:
    by_id = {row.revision_id: row for row in rows}
    for family, revision_id in component_revisions.items():
        revision = by_id.get(revision_id)
        if revision is None or revision.family != family:
            raise ValueError(f"invalid component revision for {family}: {revision_id}")
        if revision.status not in allowed_statuses:
            raise ValueError(f"component revision is not eligible: {revision_id}")


class SQLiteConfigurationStore:
    """Append-only V20 configuration registry with aggregate-head sequencing.

    SQLite is the deterministic development/test implementation.  PostgreSQL is
    the production authority.  Both expose the same API and immutable rows.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS configuration_profile_revisions(
                  revision_id TEXT PRIMARY KEY,
                  family TEXT NOT NULL,
                  profile_id TEXT NOT NULL,
                  version TEXT NOT NULL,
                  scope TEXT NOT NULL,
                  scope_key TEXT NOT NULL DEFAULT '*',
                  tenant_id TEXT,
                  environment TEXT,
                  status TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  payload_sha256 TEXT NOT NULL,
                  sequence INTEGER NOT NULL,
                  effective_from TEXT,
                  effective_until TEXT,
                  local_schedule TEXT,
                  rollout_percentage REAL NOT NULL DEFAULT 100,
                  rollout_seed TEXT NOT NULL DEFAULT 'default',
                  created_by TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  published_by TEXT,
                  published_at TEXT,
                  supersedes_revision_id TEXT,
                  etag INTEGER NOT NULL DEFAULT 1,
                  UNIQUE(family,profile_id,version,scope,scope_key,tenant_id,environment,sequence)
                );
                DROP INDEX IF EXISTS uq_configuration_profile_content_v20;
                CREATE UNIQUE INDEX uq_configuration_profile_content_v20
                  ON configuration_profile_revisions(
                    family,scope,scope_key,COALESCE(tenant_id,''),
                    COALESCE(environment,''),payload_sha256
                  );
                CREATE INDEX IF NOT EXISTS ix_configuration_profiles_lookup_v20
                  ON configuration_profile_revisions(
                    family,profile_id,scope,scope_key,tenant_id,environment,status,sequence DESC
                  );
                CREATE UNIQUE INDEX IF NOT EXISTS uq_configuration_profile_published_v25
                  ON configuration_profile_revisions(
                    family,profile_id,scope,scope_key,COALESCE(tenant_id,''),COALESCE(environment,'')
                  ) WHERE status='published';

                CREATE TABLE IF NOT EXISTS configuration_bundle_revisions(
                  revision_id TEXT PRIMARY KEY,
                  bundle_id TEXT NOT NULL,
                  version TEXT NOT NULL,
                  mode TEXT NOT NULL,
                  scope TEXT NOT NULL,
                  scope_key TEXT NOT NULL DEFAULT '*',
                  tenant_id TEXT,
                  environment TEXT,
                  status TEXT NOT NULL,
                  components_json TEXT NOT NULL,
                  content_sha256 TEXT NOT NULL,
                  sequence INTEGER NOT NULL,
                  effective_from TEXT,
                  effective_until TEXT,
                  local_schedule TEXT,
                  rollout_percentage REAL NOT NULL DEFAULT 100,
                  rollout_seed TEXT NOT NULL DEFAULT 'default',
                  created_by TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  published_by TEXT,
                  published_at TEXT,
                  supersedes_revision_id TEXT,
                  etag INTEGER NOT NULL DEFAULT 1,
                  UNIQUE(bundle_id,version,scope,scope_key,tenant_id,environment,sequence)
                );
                DROP INDEX IF EXISTS uq_configuration_bundle_content_v20;
                DROP INDEX IF EXISTS ix_configuration_bundle_content_v21;
                CREATE INDEX ix_configuration_bundle_content_v21
                  ON configuration_bundle_revisions(
                    bundle_id,scope,scope_key,COALESCE(tenant_id,''),
                    COALESCE(environment,''),content_sha256,sequence DESC
                  );
                CREATE INDEX IF NOT EXISTS ix_configuration_bundles_lookup_v20
                  ON configuration_bundle_revisions(
                    bundle_id,scope,scope_key,tenant_id,environment,status,sequence DESC
                  );
                CREATE UNIQUE INDEX IF NOT EXISTS uq_configuration_bundle_published_v25
                  ON configuration_bundle_revisions(
                    bundle_id,scope,scope_key,COALESCE(tenant_id,''),COALESCE(environment,'')
                  ) WHERE status='published';

                CREATE TABLE IF NOT EXISTS configuration_aggregate_heads(
                  aggregate_key TEXT PRIMARY KEY,
                  next_sequence INTEGER NOT NULL,
                  etag INTEGER NOT NULL DEFAULT 1,
                  updated_at TEXT NOT NULL DEFAULT '1980-01-01T00:00:00+00:00'
                );
                CREATE TABLE IF NOT EXISTS configuration_invalidation_epochs(
                  namespace TEXT PRIMARY KEY,
                  epoch INTEGER NOT NULL,
                  updated_at TEXT NOT NULL
                );
                INSERT OR IGNORE INTO configuration_invalidation_epochs(namespace,epoch,updated_at)
                  VALUES('configuration',1,'1980-01-01T00:00:00+00:00');

                CREATE TABLE IF NOT EXISTS configuration_events(
                  event_id TEXT PRIMARY KEY,
                  aggregate_type TEXT NOT NULL,
                  aggregate_id TEXT NOT NULL,
                  sequence INTEGER NOT NULL,
                  event_type TEXT NOT NULL,
                  actor_id TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  previous_hash TEXT NOT NULL,
                  event_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  UNIQUE(aggregate_type,aggregate_id,sequence)
                );
                CREATE INDEX IF NOT EXISTS ix_configuration_events_aggregate
                  ON configuration_events(aggregate_type,aggregate_id,sequence);
                """
            )
            conn.execute("BEGIN IMMEDIATE")
            head_columns = {
                str(row[1])
                for row in conn.execute(
                    "PRAGMA table_info(configuration_aggregate_heads)"
                ).fetchall()
            }
            if "etag" not in head_columns:
                conn.execute(
                    "ALTER TABLE configuration_aggregate_heads "
                    "ADD COLUMN etag INTEGER NOT NULL DEFAULT 1"
                )
            if "updated_at" not in head_columns:
                conn.execute(
                    "ALTER TABLE configuration_aggregate_heads "
                    "ADD COLUMN updated_at TEXT NOT NULL "
                    "DEFAULT '1980-01-01T00:00:00+00:00'"
                )
            for kind in ("profile", "bundle"):
                table = f"configuration_{kind}_revisions"
                columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
                if "local_schedule" not in columns:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN local_schedule TEXT")
                    conn.execute(
                        f"UPDATE {table} SET effective_from=NULLIF(effective_from,''), "
                        "effective_until=NULLIF(effective_until,'') "
                        "WHERE effective_from='' OR effective_until=''"
                    )
                conn.execute(
                    f"CREATE TRIGGER IF NOT EXISTS trg_configuration_{kind}_local_schedule_immutable "
                    f"BEFORE UPDATE OF local_schedule ON {table} "
                    "FOR EACH ROW WHEN NEW.local_schedule IS NOT OLD.local_schedule "
                    "BEGIN SELECT RAISE(ABORT, 'local_schedule is immutable'); END"
                )
            conn.execute("COMMIT")

    @staticmethod
    def _profile(row: sqlite3.Row | None) -> ProfileRevision | None:
        if row is None:
            return None
        value = dict(row)
        return ProfileRevision(
            revision_id=value["revision_id"], family=value["family"], profile_id=value["profile_id"],
            version=value["version"], scope=value["scope"], scope_key=value["scope_key"],
            tenant_id=value["tenant_id"], environment=value["environment"], status=value["status"],
            payload=_loads(value["payload_json"], {}), payload_sha256=value["payload_sha256"],
            sequence=int(value["sequence"]), effective_from=_optional_timestamp_text(value["effective_from"]),
            effective_until=_optional_timestamp_text(value["effective_until"]), rollout_percentage=float(value["rollout_percentage"]),
            rollout_seed=value["rollout_seed"], created_by=value["created_by"], created_at=_timestamp_text(value["created_at"]),
            updated_at=_timestamp_text(value["updated_at"]), published_by=value["published_by"], published_at=_optional_timestamp_text(value["published_at"]),
            supersedes_revision_id=value["supersedes_revision_id"], etag=int(value["etag"]),
            local_schedule=_loads(value["local_schedule"], None),
        )

    @staticmethod
    def _bundle(row: sqlite3.Row | None) -> BundleRevision | None:
        if row is None:
            return None
        value = dict(row)
        return BundleRevision(
            revision_id=value["revision_id"], bundle_id=value["bundle_id"], version=value["version"],
            mode=value["mode"], scope=value["scope"], scope_key=value["scope_key"], tenant_id=value["tenant_id"],
            environment=value["environment"], status=value["status"],
            component_revisions=_loads(value["components_json"], {}), content_sha256=value["content_sha256"],
            sequence=int(value["sequence"]), effective_from=_optional_timestamp_text(value["effective_from"]),
            effective_until=_optional_timestamp_text(value["effective_until"]), rollout_percentage=float(value["rollout_percentage"]),
            rollout_seed=value["rollout_seed"], created_by=value["created_by"], created_at=_timestamp_text(value["created_at"]),
            updated_at=_timestamp_text(value["updated_at"]), published_by=value["published_by"], published_at=_optional_timestamp_text(value["published_at"]),
            supersedes_revision_id=value["supersedes_revision_id"], etag=int(value["etag"]),
            local_schedule=_loads(value["local_schedule"], None),
        )

    @staticmethod
    def _next_sequence(
        conn: sqlite3.Connection,
        aggregate_key: str,
        *,
        minimum_next_sequence: int = 1,
    ) -> int:
        now = utc_now()
        conn.execute(
            "INSERT OR IGNORE INTO configuration_aggregate_heads"
            "(aggregate_key,next_sequence,etag,updated_at) VALUES(?,1,1,?)",
            (aggregate_key, now),
        )
        row = conn.execute(
            "SELECT next_sequence FROM configuration_aggregate_heads WHERE aggregate_key=?", (aggregate_key,)
        ).fetchone()
        sequence = max(int(row[0]), int(minimum_next_sequence))
        conn.execute(
            "UPDATE configuration_aggregate_heads "
            "SET next_sequence=?,etag=etag+1,updated_at=? WHERE aggregate_key=?",
            (sequence + 1, now, aggregate_key),
        )
        return sequence

    @staticmethod
    def _ensure_head_next_sequence(
        conn: sqlite3.Connection, aggregate_key: str, minimum: int
    ) -> None:
        now = utc_now()
        conn.execute(
            "INSERT OR IGNORE INTO configuration_aggregate_heads"
            "(aggregate_key,next_sequence,etag,updated_at) VALUES(?,?,1,?)",
            (aggregate_key, int(minimum), now),
        )
        conn.execute(
            "UPDATE configuration_aggregate_heads "
            "SET next_sequence=CASE WHEN next_sequence<? THEN ? ELSE next_sequence END,"
            "etag=CASE WHEN next_sequence<? THEN etag+1 ELSE etag END,"
            "updated_at=CASE WHEN next_sequence<? THEN ? ELSE updated_at END "
            "WHERE aggregate_key=?",
            (
                int(minimum),
                int(minimum),
                int(minimum),
                int(minimum),
                now,
                aggregate_key,
            ),
        )

    def get_invalidation_epoch(self, namespace: str = "configuration") -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT epoch FROM configuration_invalidation_epochs WHERE namespace=?", (namespace,)
            ).fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _bump_invalidation_epoch_tx(
        conn: sqlite3.Connection, namespace: str = "configuration"
    ) -> int:
        now = utc_now()
        conn.execute(
            "INSERT OR IGNORE INTO configuration_invalidation_epochs"
            "(namespace,epoch,updated_at) VALUES(?,?,?)",
            (namespace, 0, now),
        )
        conn.execute(
            "UPDATE configuration_invalidation_epochs "
            "SET epoch=epoch+1,updated_at=? WHERE namespace=?",
            (now, namespace),
        )
        row = conn.execute(
            "SELECT epoch FROM configuration_invalidation_epochs WHERE namespace=?",
            (namespace,),
        ).fetchone()
        if row is None:
            raise RuntimeError("configuration invalidation epoch update failed")
        return int(row[0])

    def bump_invalidation_epoch(self, namespace: str = "configuration") -> int:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            epoch = self._bump_invalidation_epoch_tx(conn, namespace)
            conn.execute("COMMIT")
        return epoch

    @staticmethod
    def _append_event_tx(
        conn: sqlite3.Connection,
        aggregate_type: str,
        aggregate_id: str,
        event_type: str,
        actor_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        row = conn.execute(
            "SELECT sequence,event_hash FROM configuration_events "
            "WHERE aggregate_type=? AND aggregate_id=? "
            "ORDER BY sequence DESC LIMIT 1",
            (aggregate_type, aggregate_id),
        ).fetchone()
        sequence = int(row["sequence"]) + 1 if row else 1
        previous = str(row["event_hash"]) if row else ""
        created = utc_now()
        body = {
            "aggregateType": aggregate_type,
            "aggregateId": aggregate_id,
            "sequence": sequence,
            "eventType": event_type,
            "actorId": actor_id,
            "payload": payload,
            "previousHash": previous,
            "createdAt": created,
        }
        event_hash = digest(body)
        event_id = f"CFE-{event_hash[:24].upper()}"
        conn.execute(
            "INSERT INTO configuration_events"
            "(event_id,aggregate_type,aggregate_id,sequence,event_type,actor_id,"
            "payload_json,previous_hash,event_hash,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                event_id,
                aggregate_type,
                aggregate_id,
                sequence,
                event_type,
                actor_id,
                canonical(payload),
                previous,
                event_hash,
                created,
            ),
        )
        return {**body, "eventId": event_id, "eventHash": event_hash}

    def append_event(self, aggregate_type: str, aggregate_id: str, event_type: str,
                     actor_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            event = self._append_event_tx(
                conn, aggregate_type, aggregate_id, event_type, actor_id, payload
            )
            conn.execute("COMMIT")
        return event

    @staticmethod
    def _events_tx(
        conn: sqlite3.Connection, aggregate_type: str, aggregate_id: str
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT event_id,aggregate_type,aggregate_id,sequence,event_type,"
            "actor_id,payload_json,previous_hash,event_hash,created_at "
            "FROM configuration_events WHERE aggregate_type=? AND aggregate_id=? "
            "ORDER BY sequence",
            (aggregate_type, aggregate_id),
        ).fetchall()
        return [_event_from_row(row, payload_column="payload_json") for row in rows]

    def events_with_verification(
        self, aggregate_type: str, aggregate_id: str
    ) -> dict[str, Any]:
        with self._connect() as conn:
            values = self._events_tx(conn, aggregate_type, aggregate_id)
        return {"items": values, "verification": _verify_event_values(values)}

    def events(self, aggregate_type: str, aggregate_id: str) -> list[dict[str, Any]]:
        return self.events_with_verification(aggregate_type, aggregate_id)["items"]

    def verify_events(self, aggregate_type: str, aggregate_id: str) -> dict[str, Any]:
        return self.events_with_verification(aggregate_type, aggregate_id)["verification"]

    @classmethod
    def _supersede_published_profiles_tx(
        cls,
        conn: sqlite3.Connection,
        *,
        family: str,
        profile_id: str,
        scope: str,
        scope_key: str,
        tenant_id: str | None,
        environment: str | None,
        except_revision_id: str | None,
        actor_id: str,
    ) -> list[str]:
        rows = conn.execute(
            "SELECT * FROM configuration_profile_revisions "
            "WHERE family=? AND profile_id=? AND scope=? AND scope_key=? "
            "AND tenant_id IS ? AND environment IS ? AND status='published' "
            "AND (? IS NULL OR revision_id<>?) ORDER BY revision_id",
            (
                family,
                profile_id,
                scope,
                scope_key,
                tenant_id,
                environment,
                except_revision_id,
                except_revision_id,
            ),
        ).fetchall()
        changed: list[str] = []
        for row in rows:
            now = utc_now()
            cursor = conn.execute(
                "UPDATE configuration_profile_revisions "
                "SET status='superseded',updated_at=?,etag=etag+1 "
                "WHERE revision_id=? AND etag=? AND status='published'",
                (now, row["revision_id"], int(row["etag"])),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("configuration revision etag conflict")
            updated = conn.execute(
                "SELECT * FROM configuration_profile_revisions WHERE revision_id=?",
                (row["revision_id"],),
            ).fetchone()
            result = cls._profile(updated)
            if result is None:
                raise RuntimeError("configuration profile supersession failed")
            cls._append_event_tx(
                conn,
                "profile",
                result.revision_id,
                "superseded",
                actor_id,
                {"etag": result.etag, "status": result.status},
            )
            changed.append(result.revision_id)
        return changed

    @classmethod
    def _supersede_published_bundles_tx(
        cls,
        conn: sqlite3.Connection,
        *,
        bundle_id: str,
        scope: str,
        scope_key: str,
        tenant_id: str | None,
        environment: str | None,
        except_revision_id: str | None,
        actor_id: str,
    ) -> list[str]:
        rows = conn.execute(
            "SELECT * FROM configuration_bundle_revisions "
            "WHERE bundle_id=? AND scope=? AND scope_key=? "
            "AND tenant_id IS ? AND environment IS ? AND status='published' "
            "AND (? IS NULL OR revision_id<>?) ORDER BY revision_id",
            (
                bundle_id,
                scope,
                scope_key,
                tenant_id,
                environment,
                except_revision_id,
                except_revision_id,
            ),
        ).fetchall()
        changed: list[str] = []
        for row in rows:
            now = utc_now()
            cursor = conn.execute(
                "UPDATE configuration_bundle_revisions "
                "SET status='superseded',updated_at=?,etag=etag+1 "
                "WHERE revision_id=? AND etag=? AND status='published'",
                (now, row["revision_id"], int(row["etag"])),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("configuration revision etag conflict")
            updated = conn.execute(
                "SELECT * FROM configuration_bundle_revisions WHERE revision_id=?",
                (row["revision_id"],),
            ).fetchone()
            result = cls._bundle(updated)
            if result is None:
                raise RuntimeError("configuration bundle supersession failed")
            cls._append_event_tx(
                conn,
                "bundle",
                result.revision_id,
                "superseded",
                actor_id,
                {"etag": result.etag, "status": result.status},
            )
            changed.append(result.revision_id)
        return changed

    @classmethod
    def _lock_bundle_components_tx(
        cls,
        conn: sqlite3.Connection,
        component_revisions: Mapping[str, str],
        *,
        allowed_statuses: tuple[str, ...],
    ) -> None:
        revision_ids = sorted(set(component_revisions.values()))
        if not revision_ids:
            raise ValueError("configuration bundle requires component revisions")
        placeholders = ",".join("?" for _ in revision_ids)
        rows = conn.execute(
            "SELECT * FROM configuration_profile_revisions "
            f"WHERE revision_id IN ({placeholders}) ORDER BY revision_id",
            revision_ids,
        ).fetchall()
        profiles = [item for row in rows if (item := cls._profile(row)) is not None]
        _validate_component_revisions(
            component_revisions, profiles, allowed_statuses=allowed_statuses
        )

    def create_profile(self, *, family: str, profile_id: str, version: str, scope: str,
                       tenant_id: str | None, payload: dict[str, Any], status: str,
                       created_by: str, scope_key: str = "*", environment: str | None = None,
                       effective_from: str | None = None, effective_until: str | None = None,
                       local_schedule: Mapping[str, str] | None = None,
                       rollout_percentage: float = 100.0, rollout_seed: str = "default",
                       supersedes_revision_id: str | None = None) -> ProfileRevision:
        schedule = normalize_schedule(effective_from, effective_until, local_schedule)
        start, end, local = schedule
        payload_hash = digest(payload)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM configuration_profile_revisions "
                "WHERE family=? AND scope=? AND scope_key=? AND tenant_id IS ? "
                "AND environment IS ? AND payload_sha256=? "
                "ORDER BY sequence DESC,revision_id LIMIT 1",
                (family, scope, scope_key, tenant_id, environment, payload_hash),
            ).fetchone()
            if existing is not None:
                result = self._profile(existing)
                assert result is not None
                _require_same_schedule(result, schedule)
                conn.execute("COMMIT")
                return result
            maximum_row = conn.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM configuration_profile_revisions "
                "WHERE family=? AND profile_id=? AND scope=? AND scope_key=? "
                "AND tenant_id IS ? AND environment IS ?",
                (family, profile_id, scope, scope_key, tenant_id, environment),
            ).fetchone()
            sequence = self._next_sequence(
                conn,
                _aggregate_key(
                    "profile",
                    (family, profile_id, scope, scope_key, tenant_id, environment),
                ),
                minimum_next_sequence=int(maximum_row[0]) + 1,
            )
            identity = {"family": family, "profileId": profile_id, "version": version, "scope": scope,
                        "scopeKey": scope_key, "tenantId": tenant_id, "environment": environment,
                        "payloadSha256": payload_hash,
                        "sequence": sequence}
            revision_id = f"CPR-{digest(identity)[:24].upper()}"
            now = utc_now()
            if status == "published":
                self._supersede_published_profiles_tx(
                    conn,
                    family=family,
                    profile_id=profile_id,
                    scope=scope,
                    scope_key=scope_key,
                    tenant_id=tenant_id,
                    environment=environment,
                    except_revision_id=None,
                    actor_id=created_by,
                )
            conn.execute(
                """INSERT INTO configuration_profile_revisions(
                   revision_id,family,profile_id,version,scope,scope_key,tenant_id,environment,status,
                   payload_json,payload_sha256,sequence,effective_from,effective_until,local_schedule,rollout_percentage,rollout_seed,
                   created_by,created_at,updated_at,published_by,published_at,supersedes_revision_id,etag)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
                (revision_id, family, profile_id, version, scope, scope_key, tenant_id, environment, status,
                 canonical(payload), payload_hash, sequence, start, end, canonical(local) if local is not None else None,
                 float(rollout_percentage), rollout_seed, created_by, now, now,
                 created_by if status == "published" else None, now if status == "published" else None,
                 supersedes_revision_id),
            )
            row = conn.execute("SELECT * FROM configuration_profile_revisions WHERE revision_id=?", (revision_id,)).fetchone()
            result = self._profile(row)
            assert result is not None
            self._append_event_tx(
                conn,
                "profile",
                revision_id,
                "created",
                created_by,
                result.to_dict(include_payload=False),
            )
            if status == "published":
                self._append_event_tx(
                    conn,
                    "profile",
                    revision_id,
                    "published",
                    created_by,
                    {"payloadSha256": payload_hash},
                )
            self._bump_invalidation_epoch_tx(conn)
            conn.execute("COMMIT")
        return result

    def get_profile(self, revision_id: str) -> ProfileRevision | None:
        with self._connect() as conn:
            return self._profile(conn.execute("SELECT * FROM configuration_profile_revisions WHERE revision_id=?", (revision_id,)).fetchone())

    def update_profile_draft(
        self,
        revision_id: str,
        *,
        expected_etag: int,
        profile_id: str,
        version: str,
        payload: dict[str, Any],
        actor_id: str,
    ) -> ProfileRevision:
        """Replace a validated draft payload using optimistic concurrency."""

        payload_hash = digest(payload)
        now = utc_now()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM configuration_profile_revisions WHERE revision_id=?",
                (revision_id,),
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                raise KeyError(revision_id)
            if int(row["etag"]) != int(expected_etag):
                conn.execute("ROLLBACK")
                raise RuntimeError("configuration revision etag conflict")
            if row["status"] != "draft":
                conn.execute("ROLLBACK")
                raise ValueError("only draft profile revisions can be edited")
            maximum_row = conn.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM configuration_profile_revisions "
                "WHERE family=? AND profile_id=? AND scope=? AND scope_key=? "
                "AND tenant_id IS ? AND environment IS ?",
                (
                    row["family"],
                    profile_id,
                    row["scope"],
                    row["scope_key"],
                    row["tenant_id"],
                    row["environment"],
                ),
            ).fetchone()
            self._ensure_head_next_sequence(
                conn,
                _aggregate_key(
                    "profile",
                    (
                        row["family"],
                        profile_id,
                        row["scope"],
                        row["scope_key"],
                        row["tenant_id"],
                        row["environment"],
                    ),
                ),
                max(int(row["sequence"]), int(maximum_row[0])) + 1,
            )
            try:
                cursor = conn.execute(
                    """UPDATE configuration_profile_revisions
                       SET profile_id=?,version=?,payload_json=?,payload_sha256=?,updated_at=?,etag=etag+1
                       WHERE revision_id=? AND etag=? AND status='draft'""",
                    (
                        profile_id,
                        version,
                        canonical(payload),
                        payload_hash,
                        now,
                        revision_id,
                        expected_etag,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                conn.execute("ROLLBACK")
                raise ValueError(
                    "configuration profile payload duplicates an existing revision"
                ) from exc
            if cursor.rowcount != 1:
                conn.execute("ROLLBACK")
                raise RuntimeError("configuration revision etag conflict")
            updated = conn.execute(
                "SELECT * FROM configuration_profile_revisions WHERE revision_id=?",
                (revision_id,),
            ).fetchone()
            result = self._profile(updated)
            assert result is not None
            self._append_event_tx(
                conn,
                "profile",
                revision_id,
                "draft_updated",
                actor_id,
                {
                    "etag": result.etag,
                    "payloadSha256": result.payload_sha256,
                    "profileId": result.profile_id,
                    "version": result.version,
                },
            )
            self._bump_invalidation_epoch_tx(conn)
            conn.execute("COMMIT")
        return result

    def find_profile_by_content(self, *, family: str, scope: str, tenant_id: str | None,
                                payload_sha256: str, scope_key: str = "*",
                                environment: str | None = None) -> ProfileRevision | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM configuration_profile_revisions "
                "WHERE family=? AND scope=? AND scope_key=? AND tenant_id IS ? "
                "AND environment IS ? AND payload_sha256=? "
                "ORDER BY sequence DESC,revision_id LIMIT 1",
                (family, scope, scope_key, tenant_id, environment, payload_sha256),
            ).fetchone()
        return self._profile(row)

    def list_profiles(self, *, family: str | None = None, profile_id: str | None = None,
                      scope: str | None = None, scope_key: str | None = None,
                       tenant_id: str | None = None, environment: str | None = None,
                       status: str | None = None, limit: int = 200,
                       exact_dimensions: bool = False) -> list[ProfileRevision]:
        clauses: list[str] = []
        args: list[Any] = []
        for column, value in (("family", family), ("profile_id", profile_id), ("scope", scope),
                              ("scope_key", scope_key), ("status", status)):
            if value is not None:
                clauses.append(f"{column}=?")
                args.append(value)
        if exact_dimensions:
            clauses.extend(("tenant_id IS ?", "environment IS ?"))
            args.extend((tenant_id, environment))
        else:
            if tenant_id is not None:
                clauses.append("tenant_id=?")
                args.append(tenant_id)
            if environment is not None:
                clauses.append("environment=?")
                args.append(environment)
        sql = "SELECT * FROM configuration_profile_revisions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY sequence DESC,created_at DESC LIMIT ?"
        args.append(max(1, min(1000, int(limit))))
        with self._connect() as conn:
            return [self._profile(row) for row in conn.execute(sql, args).fetchall()]  # type: ignore[misc]

    def publish_profile(
        self, revision_id: str, *, expected_etag: int, actor_id: str
    ) -> ProfileRevision:
        now = utc_now()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM configuration_profile_revisions WHERE revision_id=?", (revision_id,)).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                raise KeyError(revision_id)
            if int(row["etag"]) != int(expected_etag):
                conn.execute("ROLLBACK")
                raise RuntimeError("configuration revision etag conflict")
            if row["status"] != "draft":
                conn.execute("ROLLBACK")
                raise ValueError("only draft revisions can be published")
            self._supersede_published_profiles_tx(
                conn,
                family=row["family"],
                profile_id=row["profile_id"],
                scope=row["scope"],
                scope_key=row["scope_key"],
                tenant_id=row["tenant_id"],
                environment=row["environment"],
                except_revision_id=revision_id,
                actor_id=actor_id,
            )
            cursor = conn.execute(
                "UPDATE configuration_profile_revisions "
                "SET status='published',updated_at=?,etag=etag+1,"
                "published_by=?,published_at=? "
                "WHERE revision_id=? AND etag=? AND status='draft'",
                (now, actor_id, now, revision_id, expected_etag),
            )
            if cursor.rowcount != 1:
                conn.execute("ROLLBACK")
                raise RuntimeError("configuration revision etag conflict")
            updated = conn.execute("SELECT * FROM configuration_profile_revisions WHERE revision_id=?", (revision_id,)).fetchone()
            result = self._profile(updated)
            assert result is not None
            self._append_event_tx(
                conn,
                "profile",
                revision_id,
                "published",
                actor_id,
                {"etag": result.etag, "status": result.status},
            )
            self._bump_invalidation_epoch_tx(conn)
            conn.execute("COMMIT")
        return result

    def set_profile_status(
        self,
        revision_id: str,
        *,
        expected_etag: int,
        status: str,
        actor_id: str,
    ) -> ProfileRevision:
        if status == "published":
            return self.publish_profile(
                revision_id, expected_etag=expected_etag, actor_id=actor_id
            )
        now = utc_now()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM configuration_profile_revisions WHERE revision_id=?",
                (revision_id,),
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                raise KeyError(revision_id)
            if int(row["etag"]) != int(expected_etag):
                conn.execute("ROLLBACK")
                raise RuntimeError("configuration revision etag conflict")
            cursor = conn.execute(
                "UPDATE configuration_profile_revisions "
                "SET status=?,updated_at=?,etag=etag+1 "
                "WHERE revision_id=? AND etag=?",
                (status, now, revision_id, expected_etag),
            )
            if cursor.rowcount != 1:
                conn.execute("ROLLBACK")
                raise RuntimeError("configuration revision etag conflict")
            updated = conn.execute(
                "SELECT * FROM configuration_profile_revisions WHERE revision_id=?",
                (revision_id,),
            ).fetchone()
            result = self._profile(updated)
            assert result is not None
            self._append_event_tx(
                conn,
                "profile",
                revision_id,
                status,
                actor_id,
                {"etag": result.etag, "status": result.status},
            )
            self._bump_invalidation_epoch_tx(conn)
            conn.execute("COMMIT")
        return result

    def supersede_published_profiles(self, *, family: str, profile_id: str, scope: str,
                                     tenant_id: str | None, except_revision_id: str, actor_id: str,
                                     scope_key: str = "*", environment: str | None = None) -> list[str]:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = self._supersede_published_profiles_tx(
                conn,
                family=family,
                profile_id=profile_id,
                scope=scope,
                scope_key=scope_key,
                tenant_id=tenant_id,
                environment=environment,
                except_revision_id=except_revision_id,
                actor_id=actor_id,
            )
            if changed:
                self._bump_invalidation_epoch_tx(conn)
            conn.execute("COMMIT")
        return changed

    def create_bundle(self, *, bundle_id: str, version: str, mode: str, scope: str,
                      tenant_id: str | None, component_revisions: dict[str, str], status: str,
                      created_by: str, scope_key: str = "*", environment: str | None = None,
                      effective_from: str | None = None, effective_until: str | None = None,
                      local_schedule: Mapping[str, str] | None = None,
                      rollout_percentage: float = 100.0, rollout_seed: str = "default",
                       supersedes_revision_id: str | None = None,
                       force_new_revision: bool = False,
                       allowed_component_statuses: tuple[str, ...] = ("published",)) -> BundleRevision:
        schedule = normalize_schedule(effective_from, effective_until, local_schedule)
        start, end, local = schedule
        content_hash = digest({"mode": mode, "components": component_revisions})
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._lock_bundle_components_tx(
                conn,
                component_revisions,
                allowed_statuses=tuple(sorted(set(allowed_component_statuses))),
            )
            existing = conn.execute(
                "SELECT * FROM configuration_bundle_revisions "
                "WHERE bundle_id=? AND scope=? AND scope_key=? AND tenant_id IS ? "
                "AND environment IS ? AND content_sha256=? "
                "ORDER BY sequence DESC,revision_id LIMIT 1",
                (bundle_id, scope, scope_key, tenant_id, environment, content_hash),
            ).fetchone()
            if existing is not None and not force_new_revision:
                result = self._bundle(existing)
                assert result is not None
                _require_same_schedule(result, schedule)
                conn.execute("COMMIT")
                return result
            maximum_row = conn.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM configuration_bundle_revisions "
                "WHERE bundle_id=? AND scope=? AND scope_key=? "
                "AND tenant_id IS ? AND environment IS ?",
                (bundle_id, scope, scope_key, tenant_id, environment),
            ).fetchone()
            sequence = self._next_sequence(
                conn,
                _aggregate_key(
                    "bundle",
                    (bundle_id, scope, scope_key, tenant_id, environment),
                ),
                minimum_next_sequence=int(maximum_row[0]) + 1,
            )
            identity = {"bundleId": bundle_id, "version": version, "mode": mode, "scope": scope,
                        "scopeKey": scope_key, "tenantId": tenant_id, "environment": environment,
                        "contentSha256": content_hash,
                        "sequence": sequence}
            revision_id = f"CBR-{digest(identity)[:24].upper()}"
            now = utc_now()
            if status == "published":
                self._supersede_published_bundles_tx(
                    conn,
                    bundle_id=bundle_id,
                    scope=scope,
                    scope_key=scope_key,
                    tenant_id=tenant_id,
                    environment=environment,
                    except_revision_id=None,
                    actor_id=created_by,
                )
            conn.execute(
                """INSERT INTO configuration_bundle_revisions(
                   revision_id,bundle_id,version,mode,scope,scope_key,tenant_id,environment,status,
                   components_json,content_sha256,sequence,effective_from,effective_until,local_schedule,rollout_percentage,rollout_seed,
                   created_by,created_at,updated_at,published_by,published_at,supersedes_revision_id,etag)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
                (revision_id, bundle_id, version, mode, scope, scope_key, tenant_id, environment, status,
                 canonical(component_revisions), content_hash, sequence, start, end, canonical(local) if local is not None else None,
                 float(rollout_percentage), rollout_seed, created_by, now, now,
                 created_by if status == "published" else None, now if status == "published" else None,
                 supersedes_revision_id),
            )
            row = conn.execute("SELECT * FROM configuration_bundle_revisions WHERE revision_id=?", (revision_id,)).fetchone()
            result = self._bundle(row)
            assert result is not None
            self._append_event_tx(
                conn, "bundle", revision_id, "created", created_by, result.to_dict()
            )
            if status == "published":
                self._append_event_tx(
                    conn,
                    "bundle",
                    revision_id,
                    "published",
                    created_by,
                    {"contentSha256": content_hash},
                )
            self._bump_invalidation_epoch_tx(conn)
            conn.execute("COMMIT")
        return result

    def get_bundle(self, revision_id: str) -> BundleRevision | None:
        with self._connect() as conn:
            return self._bundle(conn.execute("SELECT * FROM configuration_bundle_revisions WHERE revision_id=?", (revision_id,)).fetchone())

    def find_bundle_by_content(self, *, bundle_id: str, scope: str, tenant_id: str | None, content_sha256: str,
                               scope_key: str = "*", environment: str | None = None) -> BundleRevision | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM configuration_bundle_revisions "
                "WHERE bundle_id=? AND scope=? AND scope_key=? AND tenant_id IS ? "
                "AND environment IS ? AND content_sha256=? "
                "ORDER BY sequence DESC,revision_id LIMIT 1",
                (bundle_id, scope, scope_key, tenant_id, environment, content_sha256),
            ).fetchone()
        return self._bundle(row)

    def list_bundles(self, *, bundle_id: str | None = None, scope: str | None = None,
                     scope_key: str | None = None, tenant_id: str | None = None,
                     environment: str | None = None, status: str | None = None,
                     limit: int = 200, exact_dimensions: bool = False) -> list[BundleRevision]:
        clauses: list[str] = []
        args: list[Any] = []
        for column, value in (("bundle_id", bundle_id), ("scope", scope), ("scope_key", scope_key), ("status", status)):
            if value is not None:
                clauses.append(f"{column}=?")
                args.append(value)
        if exact_dimensions:
            clauses.extend(("tenant_id IS ?", "environment IS ?"))
            args.extend((tenant_id, environment))
        else:
            if tenant_id is not None:
                clauses.append("tenant_id=?")
                args.append(tenant_id)
            if environment is not None:
                clauses.append("environment=?")
                args.append(environment)
        sql = "SELECT * FROM configuration_bundle_revisions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY sequence DESC,created_at DESC LIMIT ?"
        args.append(max(1, min(1000, int(limit))))
        with self._connect() as conn:
            return [self._bundle(row) for row in conn.execute(sql, args).fetchall()]  # type: ignore[misc]

    def publish_bundle(
        self, revision_id: str, *, expected_etag: int, actor_id: str
    ) -> BundleRevision:
        now = utc_now()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM configuration_bundle_revisions WHERE revision_id=?", (revision_id,)).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                raise KeyError(revision_id)
            if int(row["etag"]) != int(expected_etag):
                conn.execute("ROLLBACK")
                raise RuntimeError("configuration revision etag conflict")
            if row["status"] != "draft":
                conn.execute("ROLLBACK")
                raise ValueError("only draft revisions can be published")
            component_revisions = _loads(row["components_json"], {})
            self._lock_bundle_components_tx(
                conn,
                component_revisions,
                allowed_statuses=("published",),
            )
            self._supersede_published_bundles_tx(
                conn,
                bundle_id=row["bundle_id"],
                scope=row["scope"],
                scope_key=row["scope_key"],
                tenant_id=row["tenant_id"],
                environment=row["environment"],
                except_revision_id=revision_id,
                actor_id=actor_id,
            )
            cursor = conn.execute(
                "UPDATE configuration_bundle_revisions "
                "SET status='published',updated_at=?,etag=etag+1,"
                "published_by=?,published_at=? "
                "WHERE revision_id=? AND etag=? AND status='draft'",
                (now, actor_id, now, revision_id, expected_etag),
            )
            if cursor.rowcount != 1:
                conn.execute("ROLLBACK")
                raise RuntimeError("configuration revision etag conflict")
            updated = conn.execute("SELECT * FROM configuration_bundle_revisions WHERE revision_id=?", (revision_id,)).fetchone()
            result = self._bundle(updated)
            assert result is not None
            self._append_event_tx(
                conn,
                "bundle",
                revision_id,
                "published",
                actor_id,
                {"etag": result.etag, "status": result.status},
            )
            self._bump_invalidation_epoch_tx(conn)
            conn.execute("COMMIT")
        return result

    def set_bundle_status(
        self,
        revision_id: str,
        *,
        expected_etag: int,
        status: str,
        actor_id: str,
    ) -> BundleRevision:
        if status == "published":
            return self.publish_bundle(
                revision_id, expected_etag=expected_etag, actor_id=actor_id
            )
        now = utc_now()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM configuration_bundle_revisions WHERE revision_id=?",
                (revision_id,),
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                raise KeyError(revision_id)
            if int(row["etag"]) != int(expected_etag):
                conn.execute("ROLLBACK")
                raise RuntimeError("configuration revision etag conflict")
            cursor = conn.execute(
                "UPDATE configuration_bundle_revisions "
                "SET status=?,updated_at=?,etag=etag+1 "
                "WHERE revision_id=? AND etag=?",
                (status, now, revision_id, expected_etag),
            )
            if cursor.rowcount != 1:
                conn.execute("ROLLBACK")
                raise RuntimeError("configuration revision etag conflict")
            updated = conn.execute(
                "SELECT * FROM configuration_bundle_revisions WHERE revision_id=?",
                (revision_id,),
            ).fetchone()
            result = self._bundle(updated)
            assert result is not None
            self._append_event_tx(
                conn,
                "bundle",
                revision_id,
                status,
                actor_id,
                {"etag": result.etag, "status": result.status},
            )
            self._bump_invalidation_epoch_tx(conn)
            conn.execute("COMMIT")
        return result

    def supersede_published_bundles(self, *, bundle_id: str, scope: str, tenant_id: str | None,
                                    except_revision_id: str, actor_id: str, scope_key: str = "*",
                                    environment: str | None = None) -> list[str]:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = self._supersede_published_bundles_tx(
                conn,
                bundle_id=bundle_id,
                scope=scope,
                scope_key=scope_key,
                tenant_id=tenant_id,
                environment=environment,
                except_revision_id=except_revision_id,
                actor_id=actor_id,
            )
            if changed:
                self._bump_invalidation_epoch_tx(conn)
            conn.execute("COMMIT")
        return changed

    def close(self) -> None:
        return None


class PostgresConfigurationStore:
    """PostgreSQL V20 control-plane store.

    The migration defines `configuration.aggregate_heads` and immutable scoped
    revisions.  Sequence creation uses a row-level aggregate-head lock rather
    than `MAX(sequence)+1`, making concurrent publishers safe.
    """

    def __init__(self, dsn: str):
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('Install with: pip install "baloot-platform[postgres]"') from exc
        self.pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=24,
            open=True,
            kwargs={"row_factory": dict_row},
        )

    @staticmethod
    def _profile(row: Any | None) -> ProfileRevision | None:
        if row is None:
            return None
        d = _row_mapping(row)
        return ProfileRevision(
            revision_id=d["revision_id"], family=d["family"], profile_id=d["profile_id"], version=d["version"],
            scope=d["scope"], scope_key=d["scope_key"], tenant_id=d.get("tenant_id"), environment=d.get("environment"),
            status=d["status"], payload=_loads(d["payload"], {}), payload_sha256=d["payload_sha256"],
            sequence=int(d["sequence"]), effective_from=_optional_timestamp_text(d.get("effective_from")),
            effective_until=_optional_timestamp_text(d.get("effective_until")),
            rollout_percentage=float(d.get("rollout_percentage", 100)), rollout_seed=d.get("rollout_seed", "default"),
            created_by=d["created_by"], created_at=_timestamp_text(d["created_at"]), updated_at=_timestamp_text(d["updated_at"]),
            published_by=d.get("published_by"), published_at=_optional_timestamp_text(d.get("published_at")),
            supersedes_revision_id=d.get("supersedes_revision_id"), etag=int(d["etag"]),
            local_schedule=_loads(d.get("local_schedule"), None),
        )

    @staticmethod
    def _bundle(row: Any | None) -> BundleRevision | None:
        if row is None:
            return None
        d = _row_mapping(row)
        return BundleRevision(
            revision_id=d["revision_id"], bundle_id=d["bundle_id"], version=d["version"], mode=d["mode"],
            scope=d["scope"], scope_key=d["scope_key"], tenant_id=d.get("tenant_id"), environment=d.get("environment"),
            status=d["status"], component_revisions=_loads(d["components"], {}),
            content_sha256=d["content_sha256"], sequence=int(d["sequence"]),
            effective_from=_optional_timestamp_text(d.get("effective_from")),
            effective_until=_optional_timestamp_text(d.get("effective_until")),
            rollout_percentage=float(d.get("rollout_percentage", 100)), rollout_seed=d.get("rollout_seed", "default"),
            created_by=d["created_by"], created_at=_timestamp_text(d["created_at"]), updated_at=_timestamp_text(d["updated_at"]),
            published_by=d.get("published_by"), published_at=_optional_timestamp_text(d.get("published_at")),
            supersedes_revision_id=d.get("supersedes_revision_id"), etag=int(d["etag"]),
            local_schedule=_loads(d.get("local_schedule"), None),
        )

    @staticmethod
    def _lock_head(conn: Any, aggregate_type: str, aggregate_id: str) -> dict[str, Any]:
        now = utc_now()
        conn.execute(
            "INSERT INTO configuration.aggregate_heads"
            "(aggregate_type,aggregate_id,next_sequence,etag,updated_at) "
            "VALUES(%s,%s,1,1,%s::timestamptz) "
            "ON CONFLICT(aggregate_type,aggregate_id) DO NOTHING",
            (aggregate_type, aggregate_id, now),
        )
        row = conn.execute(
            "SELECT next_sequence,etag,updated_at "
            "FROM configuration.aggregate_heads "
            "WHERE aggregate_type=%s AND aggregate_id=%s FOR UPDATE",
            (aggregate_type, aggregate_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("configuration aggregate head lock failed")
        return _row_mapping(row)

    @classmethod
    def _next_sequence(
        cls,
        conn: Any,
        aggregate_type: str,
        aggregate_id: str,
        *,
        minimum_next_sequence: int = 1,
    ) -> int:
        head = cls._lock_head(conn, aggregate_type, aggregate_id)
        sequence = max(int(head["next_sequence"]), int(minimum_next_sequence))
        now = utc_now()
        row = conn.execute(
            "UPDATE configuration.aggregate_heads "
            "SET next_sequence=%s,etag=etag+1,updated_at=%s::timestamptz "
            "WHERE aggregate_type=%s AND aggregate_id=%s AND etag=%s "
            "RETURNING next_sequence-1 AS sequence,etag,updated_at",
            (
                sequence + 1,
                now,
                aggregate_type,
                aggregate_id,
                int(head["etag"]),
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("configuration aggregate head etag conflict")
        return int(_row_mapping(row)["sequence"])

    @classmethod
    def _ensure_head_next_sequence(
        cls, conn: Any, aggregate_type: str, aggregate_id: str, minimum: int
    ) -> None:
        head = cls._lock_head(conn, aggregate_type, aggregate_id)
        if int(head["next_sequence"]) >= int(minimum):
            return
        row = conn.execute(
            "UPDATE configuration.aggregate_heads "
            "SET next_sequence=%s,etag=etag+1,updated_at=%s::timestamptz "
            "WHERE aggregate_type=%s AND aggregate_id=%s AND etag=%s "
            "RETURNING next_sequence,etag,updated_at",
            (
                int(minimum),
                utc_now(),
                aggregate_type,
                aggregate_id,
                int(head["etag"]),
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("configuration aggregate head etag conflict")

    def get_invalidation_epoch(self, namespace: str = "configuration") -> int:
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                "SELECT epoch FROM configuration.invalidation_epochs WHERE namespace=%s",
                (namespace,),
            ).fetchone()
        return int(_row_mapping(row)["epoch"]) if row else 0

    @staticmethod
    def _bump_invalidation_epoch_tx(
        conn: Any, namespace: str = "configuration"
    ) -> int:
        now = utc_now()
        row = conn.execute(
            "INSERT INTO configuration.invalidation_epochs"
            "(namespace,epoch,updated_at) VALUES(%s,1,%s::timestamptz) "
            "ON CONFLICT(namespace) DO UPDATE SET "
            "epoch=configuration.invalidation_epochs.epoch+1,"
            "updated_at=EXCLUDED.updated_at RETURNING epoch,updated_at",
            (namespace, now),
        ).fetchone()
        if row is None:
            raise RuntimeError("configuration invalidation epoch update failed")
        return int(_row_mapping(row)["epoch"])

    def bump_invalidation_epoch(self, namespace: str = "configuration") -> int:
        with contextual_connection(self.pool) as conn:
            return self._bump_invalidation_epoch_tx(conn, namespace)

    @classmethod
    def _append_event_tx(
        cls,
        conn: Any,
        aggregate_type: str,
        aggregate_id: str,
        event_type: str,
        actor_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if aggregate_type not in {"profile", "bundle"}:
            raise ValueError("configuration event aggregate type is invalid")
        head_type = f"event_{aggregate_type}"
        head = cls._lock_head(conn, head_type, aggregate_id)
        row = conn.execute(
            "SELECT sequence,event_hash FROM configuration.events "
            "WHERE aggregate_type=%s AND aggregate_id=%s "
            "ORDER BY sequence DESC LIMIT 1",
            (aggregate_type, aggregate_id),
        ).fetchone()
        previous_event = _row_mapping(row) if row is not None else None
        sequence = (
            int(previous_event["sequence"]) + 1 if previous_event is not None else 1
        )
        previous = (
            str(previous_event["event_hash"]) if previous_event is not None else ""
        )
        head_next = int(head["next_sequence"])
        if head_next != sequence:
            # Migration 012 introduced aggregate heads after event streams
            # already existed.  A brand-new head may be synchronized once;
            # any other divergence indicates corruption and fails closed.
            if head_next != 1 or int(head["etag"]) != 1 or sequence <= 1:
                raise RuntimeError("configuration event head is not contiguous")
        updated_head = conn.execute(
            "UPDATE configuration.aggregate_heads "
            "SET next_sequence=%s,etag=etag+1,updated_at=%s::timestamptz "
            "WHERE aggregate_type=%s AND aggregate_id=%s AND etag=%s "
            "RETURNING next_sequence,etag,updated_at",
            (
                sequence + 1,
                utc_now(),
                head_type,
                aggregate_id,
                int(head["etag"]),
            ),
        ).fetchone()
        if updated_head is None:
            raise RuntimeError("configuration event head etag conflict")
        created = utc_now()
        body = {
            "aggregateType": aggregate_type,
            "aggregateId": aggregate_id,
            "sequence": sequence,
            "eventType": event_type,
            "actorId": actor_id,
            "payload": payload,
            "previousHash": previous,
            "createdAt": created,
        }
        event_hash = digest(body)
        event_id = f"CFE-{event_hash[:24].upper()}"
        conn.execute(
            "INSERT INTO configuration.events"
            "(event_id,aggregate_type,aggregate_id,sequence,event_type,actor_id,"
            "payload,previous_hash,event_hash,created_at) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::timestamptz)",
            (
                event_id,
                aggregate_type,
                aggregate_id,
                sequence,
                event_type,
                actor_id,
                canonical(payload),
                previous,
                event_hash,
                created,
            ),
        )
        return {**body, "eventId": event_id, "eventHash": event_hash}

    def append_event(self, aggregate_type: str, aggregate_id: str, event_type: str,
                     actor_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with contextual_connection(self.pool) as conn:
            return self._append_event_tx(
                conn, aggregate_type, aggregate_id, event_type, actor_id, payload
            )

    @staticmethod
    def _events_tx(
        conn: Any, aggregate_type: str, aggregate_id: str
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            f"SELECT {_POSTGRES_EVENT_COLUMNS} FROM configuration.events "
            "WHERE aggregate_type=%s AND aggregate_id=%s ORDER BY sequence",
            (aggregate_type, aggregate_id),
        ).fetchall()
        return [_event_from_row(row, payload_column="payload") for row in rows]

    def events_with_verification(
        self, aggregate_type: str, aggregate_id: str
    ) -> dict[str, Any]:
        with contextual_connection(self.pool) as conn:
            values = self._events_tx(conn, aggregate_type, aggregate_id)
        return {"items": values, "verification": _verify_event_values(values)}

    def events(self, aggregate_type: str, aggregate_id: str) -> list[dict[str, Any]]:
        return self.events_with_verification(aggregate_type, aggregate_id)["items"]

    def verify_events(self, aggregate_type: str, aggregate_id: str) -> dict[str, Any]:
        return self.events_with_verification(aggregate_type, aggregate_id)["verification"]

    @classmethod
    def _supersede_published_profiles_tx(
        cls,
        conn: Any,
        *,
        family: str,
        profile_id: str,
        scope: str,
        scope_key: str,
        tenant_id: str | None,
        environment: str | None,
        except_revision_id: str | None,
        actor_id: str,
    ) -> list[str]:
        rows = conn.execute(
            f"SELECT {_POSTGRES_PROFILE_COLUMNS} "
            "FROM configuration.profile_revisions "
            "WHERE family=%s AND profile_id=%s AND scope=%s AND scope_key=%s "
            "AND tenant_id IS NOT DISTINCT FROM %s "
            "AND environment IS NOT DISTINCT FROM %s AND status='published' "
            "AND revision_id IS DISTINCT FROM %s ORDER BY revision_id FOR UPDATE",
            (
                family,
                profile_id,
                scope,
                scope_key,
                tenant_id,
                environment,
                except_revision_id,
            ),
        ).fetchall()
        changed: list[str] = []
        for row in rows:
            current = cls._profile(row)
            assert current is not None
            updated = conn.execute(
                "UPDATE configuration.profile_revisions "
                "SET status='superseded',updated_at=%s::timestamptz,etag=etag+1 "
                "WHERE revision_id=%s AND etag=%s AND status='published' "
                f"RETURNING {_POSTGRES_PROFILE_COLUMNS}",
                (utc_now(), current.revision_id, current.etag),
            ).fetchone()
            result = cls._profile(updated)
            if result is None:
                raise RuntimeError("configuration revision etag conflict")
            cls._append_event_tx(
                conn,
                "profile",
                result.revision_id,
                "superseded",
                actor_id,
                {"etag": result.etag, "status": result.status},
            )
            changed.append(result.revision_id)
        return changed

    @classmethod
    def _supersede_published_bundles_tx(
        cls,
        conn: Any,
        *,
        bundle_id: str,
        scope: str,
        scope_key: str,
        tenant_id: str | None,
        environment: str | None,
        except_revision_id: str | None,
        actor_id: str,
    ) -> list[str]:
        rows = conn.execute(
            f"SELECT {_POSTGRES_BUNDLE_COLUMNS} "
            "FROM configuration.bundle_revisions "
            "WHERE bundle_id=%s AND scope=%s AND scope_key=%s "
            "AND tenant_id IS NOT DISTINCT FROM %s "
            "AND environment IS NOT DISTINCT FROM %s AND status='published' "
            "AND revision_id IS DISTINCT FROM %s ORDER BY revision_id FOR UPDATE",
            (
                bundle_id,
                scope,
                scope_key,
                tenant_id,
                environment,
                except_revision_id,
            ),
        ).fetchall()
        changed: list[str] = []
        for row in rows:
            current = cls._bundle(row)
            assert current is not None
            updated = conn.execute(
                "UPDATE configuration.bundle_revisions "
                "SET status='superseded',updated_at=%s::timestamptz,etag=etag+1 "
                "WHERE revision_id=%s AND etag=%s AND status='published' "
                f"RETURNING {_POSTGRES_BUNDLE_COLUMNS}",
                (utc_now(), current.revision_id, current.etag),
            ).fetchone()
            result = cls._bundle(updated)
            if result is None:
                raise RuntimeError("configuration revision etag conflict")
            cls._append_event_tx(
                conn,
                "bundle",
                result.revision_id,
                "superseded",
                actor_id,
                {"etag": result.etag, "status": result.status},
            )
            changed.append(result.revision_id)
        return changed

    @classmethod
    def _lock_bundle_components_tx(
        cls,
        conn: Any,
        component_revisions: Mapping[str, str],
        *,
        allowed_statuses: tuple[str, ...],
    ) -> None:
        revision_ids = sorted(set(component_revisions.values()))
        if not revision_ids:
            raise ValueError("configuration bundle requires component revisions")
        rows = conn.execute(
            f"SELECT {_POSTGRES_PROFILE_COLUMNS} "
            "FROM configuration.profile_revisions "
            "WHERE revision_id=ANY(%s) ORDER BY revision_id FOR SHARE",
            (revision_ids,),
        ).fetchall()
        profiles = [item for row in rows if (item := cls._profile(row)) is not None]
        _validate_component_revisions(
            component_revisions, profiles, allowed_statuses=allowed_statuses
        )

    def create_profile(self, **kwargs: Any) -> ProfileRevision:
        schedule = normalize_schedule(kwargs.get("effective_from"), kwargs.get("effective_until"), kwargs.get("local_schedule"))
        start, end, local = schedule
        family = str(kwargs["family"])
        profile_id = str(kwargs["profile_id"])
        scope = str(kwargs["scope"])
        scope_key = str(kwargs.get("scope_key", "*"))
        tenant_id = kwargs.get("tenant_id")
        environment = kwargs.get("environment")
        payload_hash = digest(kwargs["payload"])
        logical_identity = _aggregate_identity(
            (family, profile_id, scope, scope_key, tenant_id, environment)
        )
        content_identity = _aggregate_identity(
            (family, scope, scope_key, tenant_id, environment, payload_hash)
        )
        with contextual_connection(self.pool) as conn:
            self._lock_head(conn, "profile_content", content_identity)
            existing = conn.execute(
                f"SELECT {_POSTGRES_PROFILE_COLUMNS} "
                "FROM configuration.profile_revisions "
                "WHERE family=%s AND scope=%s AND scope_key=%s "
                "AND tenant_id IS NOT DISTINCT FROM %s "
                "AND environment IS NOT DISTINCT FROM %s AND payload_sha256=%s "
                "ORDER BY sequence DESC,revision_id LIMIT 1",
                (family, scope, scope_key, tenant_id, environment, payload_hash),
            ).fetchone()
            if existing is not None:
                result = self._profile(existing)
                assert result is not None
                _require_same_schedule(result, schedule)
                return result
            self._lock_head(conn, "profile", logical_identity)
            maximum_row = conn.execute(
                "SELECT COALESCE(MAX(sequence),0) AS max_sequence "
                "FROM configuration.profile_revisions "
                "WHERE family=%s AND profile_id=%s AND scope=%s AND scope_key=%s "
                "AND tenant_id IS NOT DISTINCT FROM %s "
                "AND environment IS NOT DISTINCT FROM %s",
                (family, profile_id, scope, scope_key, tenant_id, environment),
            ).fetchone()
            maximum = int(_row_mapping(maximum_row)["max_sequence"])
            sequence = self._next_sequence(
                conn,
                "profile",
                logical_identity,
                minimum_next_sequence=maximum + 1,
            )
            identity = {
                "family": family,
                "profileId": profile_id,
                "version": kwargs["version"],
                "scope": scope,
                "scopeKey": scope_key,
                "tenantId": tenant_id,
                "environment": environment,
                "payloadSha256": payload_hash,
                "sequence": sequence,
            }
            revision_id = f"CPR-{digest(identity)[:24].upper()}"
            now = utc_now()
            if kwargs["status"] == "published":
                self._supersede_published_profiles_tx(
                    conn,
                    family=family,
                    profile_id=profile_id,
                    scope=scope,
                    scope_key=scope_key,
                    tenant_id=tenant_id,
                    environment=environment,
                    except_revision_id=None,
                    actor_id=kwargs["created_by"],
                )
            row = conn.execute(
                "INSERT INTO configuration.profile_revisions"
                "(revision_id,family,profile_id,version,scope,scope_key,tenant_id,"
                "environment,status,payload,payload_sha256,sequence,effective_from,"
                "effective_until,local_schedule,rollout_percentage,rollout_seed,created_by,created_at,"
                "updated_at,published_by,published_at,supersedes_revision_id,etag) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,"
                "%s::timestamptz,%s::timestamptz,%s::jsonb,%s,%s,%s,%s::timestamptz,"
                "%s::timestamptz,%s,%s::timestamptz,%s,1) "
                f"RETURNING {_POSTGRES_PROFILE_COLUMNS}",
                (
                    revision_id,
                    family,
                    profile_id,
                    kwargs["version"],
                    scope,
                    scope_key,
                    tenant_id,
                    environment,
                    kwargs["status"],
                    canonical(kwargs["payload"]),
                    payload_hash,
                    sequence,
                    start,
                    end,
                    canonical(local) if local is not None else None,
                    float(kwargs.get("rollout_percentage", 100)),
                    kwargs.get("rollout_seed", "default"),
                    kwargs["created_by"],
                    now,
                    now,
                    kwargs["created_by"] if kwargs["status"] == "published" else None,
                    now if kwargs["status"] == "published" else None,
                    kwargs.get("supersedes_revision_id"),
                ),
            ).fetchone()
            result = self._profile(row)
            assert result is not None
            self._append_event_tx(
                conn,
                "profile",
                revision_id,
                "created",
                kwargs["created_by"],
                result.to_dict(include_payload=False),
            )
            if kwargs["status"] == "published":
                self._append_event_tx(
                    conn,
                    "profile",
                    revision_id,
                    "published",
                    kwargs["created_by"],
                    {"payloadSha256": payload_hash},
                )
            self._bump_invalidation_epoch_tx(conn)
            return result

    def get_profile(self, revision_id: str) -> ProfileRevision | None:
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                f"SELECT {_POSTGRES_PROFILE_COLUMNS} "
                "FROM configuration.profile_revisions WHERE revision_id=%s",
                (revision_id,),
            ).fetchone()
        return self._profile(row)

    def update_profile_draft(
        self,
        revision_id: str,
        *,
        expected_etag: int,
        profile_id: str,
        version: str,
        payload: dict[str, Any],
        actor_id: str,
    ) -> ProfileRevision:
        payload_hash = digest(payload)
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                f"SELECT {_POSTGRES_PROFILE_COLUMNS} "
                "FROM configuration.profile_revisions "
                "WHERE revision_id=%s FOR UPDATE",
                (revision_id,),
            ).fetchone()
            if not row:
                raise KeyError(revision_id)
            current = self._profile(row)
            assert current is not None
            if current.etag != int(expected_etag):
                raise RuntimeError("configuration revision etag conflict")
            if current.status != "draft":
                raise ValueError("only draft profile revisions can be edited")
            content_identity = _aggregate_identity(
                (
                    current.family,
                    current.scope,
                    current.scope_key,
                    current.tenant_id,
                    current.environment,
                    payload_hash,
                )
            )
            logical_identity = _aggregate_identity(
                (
                    current.family,
                    profile_id,
                    current.scope,
                    current.scope_key,
                    current.tenant_id,
                    current.environment,
                )
            )
            self._lock_head(conn, "profile_content", content_identity)
            maximum_row = conn.execute(
                "SELECT COALESCE(MAX(sequence),0) AS max_sequence "
                "FROM configuration.profile_revisions "
                "WHERE family=%s AND profile_id=%s AND scope=%s AND scope_key=%s "
                "AND tenant_id IS NOT DISTINCT FROM %s "
                "AND environment IS NOT DISTINCT FROM %s",
                (
                    current.family,
                    profile_id,
                    current.scope,
                    current.scope_key,
                    current.tenant_id,
                    current.environment,
                ),
            ).fetchone()
            maximum = int(_row_mapping(maximum_row)["max_sequence"])
            self._ensure_head_next_sequence(
                conn,
                "profile",
                logical_identity,
                max(current.sequence, maximum) + 1,
            )
            try:
                updated = conn.execute(
                    """UPDATE configuration.profile_revisions
                       SET profile_id=%s,version=%s,payload=%s::jsonb,payload_sha256=%s,
                           updated_at=%s::timestamptz,etag=etag+1
                       WHERE revision_id=%s AND etag=%s AND status='draft'
                       RETURNING """ + _POSTGRES_PROFILE_COLUMNS,
                    (
                        profile_id,
                        version,
                        canonical(payload),
                        payload_hash,
                        utc_now(),
                        revision_id,
                        expected_etag,
                    ),
                ).fetchone()
            except Exception as exc:
                if getattr(exc, "sqlstate", None) == "23505":
                    raise ValueError(
                        "configuration profile payload duplicates an existing revision"
                    ) from exc
                raise
            if updated is None:
                raise RuntimeError("configuration revision etag conflict")
            result = self._profile(updated)
            assert result is not None
            self._append_event_tx(
                conn,
                "profile",
                revision_id,
                "draft_updated",
                actor_id,
                {
                    "etag": result.etag,
                    "payloadSha256": result.payload_sha256,
                    "profileId": result.profile_id,
                    "version": result.version,
                },
            )
            self._bump_invalidation_epoch_tx(conn)
            return result

    def find_profile_by_content(
        self,
        *,
        family: str,
        scope: str,
        tenant_id: str | None,
        payload_sha256: str,
        scope_key: str = "*",
        environment: str | None = None,
    ) -> ProfileRevision | None:
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                f"SELECT {_POSTGRES_PROFILE_COLUMNS} "
                "FROM configuration.profile_revisions "
                "WHERE family=%s AND scope=%s AND scope_key=%s "
                "AND tenant_id IS NOT DISTINCT FROM %s "
                "AND environment IS NOT DISTINCT FROM %s AND payload_sha256=%s "
                "ORDER BY sequence DESC,revision_id LIMIT 1",
                (family, scope, scope_key, tenant_id, environment, payload_sha256),
            ).fetchone()
        return self._profile(row)

    def list_profiles(
        self,
        *,
        family: str | None = None,
        profile_id: str | None = None,
        scope: str | None = None,
        scope_key: str | None = None,
        tenant_id: str | None = None,
        environment: str | None = None,
        status: str | None = None,
        limit: int = 200,
        exact_dimensions: bool = False,
    ) -> list[ProfileRevision]:
        clauses: list[str] = []
        args: list[Any] = []
        for column, value in (
            ("family", family),
            ("profile_id", profile_id),
            ("scope", scope),
            ("scope_key", scope_key),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column}=%s")
                args.append(value)
        if exact_dimensions:
            clauses.extend(
                (
                    "tenant_id IS NOT DISTINCT FROM %s",
                    "environment IS NOT DISTINCT FROM %s",
                )
            )
            args.extend((tenant_id, environment))
        else:
            if tenant_id is not None:
                clauses.append("tenant_id=%s")
                args.append(tenant_id)
            if environment is not None:
                clauses.append("environment=%s")
                args.append(environment)
        sql = f"SELECT {_POSTGRES_PROFILE_COLUMNS} FROM configuration.profile_revisions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY sequence DESC,created_at DESC,revision_id LIMIT %s"
        args.append(max(1, min(1000, int(limit))))
        with contextual_connection(self.pool) as conn:
            rows = conn.execute(sql, args).fetchall()
        return [item for row in rows if (item := self._profile(row)) is not None]

    def publish_profile(
        self, revision_id: str, *, expected_etag: int, actor_id: str
    ) -> ProfileRevision:
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                f"SELECT {_POSTGRES_PROFILE_COLUMNS} "
                "FROM configuration.profile_revisions "
                "WHERE revision_id=%s FOR UPDATE",
                (revision_id,),
            ).fetchone()
            if row is None:
                raise KeyError(revision_id)
            current = self._profile(row)
            assert current is not None
            if current.etag != int(expected_etag):
                raise RuntimeError("configuration revision etag conflict")
            if current.status != "draft":
                raise ValueError("only draft revisions can be published")
            logical_identity = _aggregate_identity(
                (
                    current.family,
                    current.profile_id,
                    current.scope,
                    current.scope_key,
                    current.tenant_id,
                    current.environment,
                )
            )
            self._lock_head(conn, "profile", logical_identity)
            self._supersede_published_profiles_tx(
                conn,
                family=current.family,
                profile_id=current.profile_id,
                scope=current.scope,
                scope_key=current.scope_key,
                tenant_id=current.tenant_id,
                environment=current.environment,
                except_revision_id=revision_id,
                actor_id=actor_id,
            )
            now = utc_now()
            updated = conn.execute(
                "UPDATE configuration.profile_revisions "
                "SET status='published',updated_at=%s::timestamptz,etag=etag+1,"
                "published_by=%s,published_at=%s::timestamptz "
                "WHERE revision_id=%s AND etag=%s AND status='draft' "
                f"RETURNING {_POSTGRES_PROFILE_COLUMNS}",
                (now, actor_id, now, revision_id, expected_etag),
            ).fetchone()
            result = self._profile(updated)
            if result is None:
                raise RuntimeError("configuration revision etag conflict")
            self._append_event_tx(
                conn,
                "profile",
                revision_id,
                "published",
                actor_id,
                {"etag": result.etag, "status": result.status},
            )
            self._bump_invalidation_epoch_tx(conn)
            return result

    def set_profile_status(
        self,
        revision_id: str,
        *,
        expected_etag: int,
        status: str,
        actor_id: str,
    ) -> ProfileRevision:
        if status == "published":
            return self.publish_profile(
                revision_id, expected_etag=expected_etag, actor_id=actor_id
            )
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                f"SELECT {_POSTGRES_PROFILE_COLUMNS} "
                "FROM configuration.profile_revisions "
                "WHERE revision_id=%s FOR UPDATE",
                (revision_id,),
            ).fetchone()
            if row is None:
                raise KeyError(revision_id)
            current = self._profile(row)
            assert current is not None
            if current.etag != int(expected_etag):
                raise RuntimeError("configuration revision etag conflict")
            updated = conn.execute(
                "UPDATE configuration.profile_revisions "
                "SET status=%s,updated_at=%s::timestamptz,etag=etag+1 "
                "WHERE revision_id=%s AND etag=%s "
                f"RETURNING {_POSTGRES_PROFILE_COLUMNS}",
                (status, utc_now(), revision_id, expected_etag),
            ).fetchone()
            result = self._profile(updated)
            if result is None:
                raise RuntimeError("configuration revision etag conflict")
            self._append_event_tx(
                conn,
                "profile",
                revision_id,
                status,
                actor_id,
                {"etag": result.etag, "status": result.status},
            )
            self._bump_invalidation_epoch_tx(conn)
            return result

    def supersede_published_profiles(
        self,
        *,
        family: str,
        profile_id: str,
        scope: str,
        tenant_id: str | None,
        except_revision_id: str,
        actor_id: str,
        scope_key: str = "*",
        environment: str | None = None,
    ) -> list[str]:
        logical_identity = _aggregate_identity(
            (family, profile_id, scope, scope_key, tenant_id, environment)
        )
        with contextual_connection(self.pool) as conn:
            self._lock_head(conn, "profile", logical_identity)
            changed = self._supersede_published_profiles_tx(
                conn,
                family=family,
                profile_id=profile_id,
                scope=scope,
                scope_key=scope_key,
                tenant_id=tenant_id,
                environment=environment,
                except_revision_id=except_revision_id,
                actor_id=actor_id,
            )
            if changed:
                self._bump_invalidation_epoch_tx(conn)
            return changed

    def create_bundle(self, **kwargs: Any) -> BundleRevision:
        schedule = normalize_schedule(kwargs.get("effective_from"), kwargs.get("effective_until"), kwargs.get("local_schedule"))
        start, end, local = schedule
        bundle_id = str(kwargs["bundle_id"])
        scope = str(kwargs["scope"])
        scope_key = str(kwargs.get("scope_key", "*"))
        tenant_id = kwargs.get("tenant_id")
        environment = kwargs.get("environment")
        content_hash = digest(
            {"mode": kwargs["mode"], "components": kwargs["component_revisions"]}
        )
        logical_identity = _aggregate_identity(
            (bundle_id, scope, scope_key, tenant_id, environment)
        )
        content_identity = _aggregate_identity(
            (bundle_id, scope, scope_key, tenant_id, environment, content_hash)
        )
        with contextual_connection(self.pool) as conn:
            self._lock_bundle_components_tx(
                conn,
                kwargs["component_revisions"],
                allowed_statuses=tuple(
                    sorted(set(kwargs.get("allowed_component_statuses", ("published",))))
                ),
            )
            self._lock_head(conn, "bundle_content", content_identity)
            existing = conn.execute(
                f"SELECT {_POSTGRES_BUNDLE_COLUMNS} "
                "FROM configuration.bundle_revisions "
                "WHERE bundle_id=%s AND scope=%s AND scope_key=%s "
                "AND tenant_id IS NOT DISTINCT FROM %s "
                "AND environment IS NOT DISTINCT FROM %s AND content_sha256=%s "
                "ORDER BY sequence DESC,revision_id LIMIT 1",
                (bundle_id, scope, scope_key, tenant_id, environment, content_hash),
            ).fetchone()
            if existing is not None and not bool(
                kwargs.get("force_new_revision", False)
            ):
                result = self._bundle(existing)
                assert result is not None
                _require_same_schedule(result, schedule)
                return result
            self._lock_head(conn, "bundle", logical_identity)
            maximum_row = conn.execute(
                "SELECT COALESCE(MAX(sequence),0) AS max_sequence "
                "FROM configuration.bundle_revisions "
                "WHERE bundle_id=%s AND scope=%s AND scope_key=%s "
                "AND tenant_id IS NOT DISTINCT FROM %s "
                "AND environment IS NOT DISTINCT FROM %s",
                (bundle_id, scope, scope_key, tenant_id, environment),
            ).fetchone()
            maximum = int(_row_mapping(maximum_row)["max_sequence"])
            sequence = self._next_sequence(
                conn,
                "bundle",
                logical_identity,
                minimum_next_sequence=maximum + 1,
            )
            identity = {
                "bundleId": bundle_id,
                "version": kwargs["version"],
                "mode": kwargs["mode"],
                "scope": scope,
                "scopeKey": scope_key,
                "tenantId": tenant_id,
                "environment": environment,
                "contentSha256": content_hash,
                "sequence": sequence,
            }
            revision_id = f"CBR-{digest(identity)[:24].upper()}"
            now = utc_now()
            if kwargs["status"] == "published":
                self._supersede_published_bundles_tx(
                    conn,
                    bundle_id=bundle_id,
                    scope=scope,
                    scope_key=scope_key,
                    tenant_id=tenant_id,
                    environment=environment,
                    except_revision_id=None,
                    actor_id=kwargs["created_by"],
                )
            row = conn.execute(
                "INSERT INTO configuration.bundle_revisions"
                "(revision_id,bundle_id,version,mode,scope,scope_key,tenant_id,"
                "environment,status,components,content_sha256,sequence,effective_from,"
                "effective_until,local_schedule,rollout_percentage,rollout_seed,created_by,created_at,"
                "updated_at,published_by,published_at,supersedes_revision_id,etag) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,"
                "%s::timestamptz,%s::timestamptz,%s::jsonb,%s,%s,%s,%s::timestamptz,"
                "%s::timestamptz,%s,%s::timestamptz,%s,1) "
                f"RETURNING {_POSTGRES_BUNDLE_COLUMNS}",
                (
                    revision_id,
                    bundle_id,
                    kwargs["version"],
                    kwargs["mode"],
                    scope,
                    scope_key,
                    tenant_id,
                    environment,
                    kwargs["status"],
                    canonical(kwargs["component_revisions"]),
                    content_hash,
                    sequence,
                    start,
                    end,
                    canonical(local) if local is not None else None,
                    float(kwargs.get("rollout_percentage", 100)),
                    kwargs.get("rollout_seed", "default"),
                    kwargs["created_by"],
                    now,
                    now,
                    kwargs["created_by"] if kwargs["status"] == "published" else None,
                    now if kwargs["status"] == "published" else None,
                    kwargs.get("supersedes_revision_id"),
                ),
            ).fetchone()
            result = self._bundle(row)
            assert result is not None
            self._append_event_tx(
                conn, "bundle", revision_id, "created", kwargs["created_by"], result.to_dict()
            )
            if kwargs["status"] == "published":
                self._append_event_tx(
                    conn,
                    "bundle",
                    revision_id,
                    "published",
                    kwargs["created_by"],
                    {"contentSha256": content_hash},
                )
            self._bump_invalidation_epoch_tx(conn)
            return result

    def get_bundle(self, revision_id: str) -> BundleRevision | None:
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                f"SELECT {_POSTGRES_BUNDLE_COLUMNS} "
                "FROM configuration.bundle_revisions WHERE revision_id=%s",
                (revision_id,),
            ).fetchone()
        return self._bundle(row)

    def find_bundle_by_content(
        self,
        *,
        bundle_id: str,
        scope: str,
        tenant_id: str | None,
        content_sha256: str,
        scope_key: str = "*",
        environment: str | None = None,
    ) -> BundleRevision | None:
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                f"SELECT {_POSTGRES_BUNDLE_COLUMNS} "
                "FROM configuration.bundle_revisions "
                "WHERE bundle_id=%s AND scope=%s AND scope_key=%s "
                "AND tenant_id IS NOT DISTINCT FROM %s "
                "AND environment IS NOT DISTINCT FROM %s AND content_sha256=%s "
                "ORDER BY sequence DESC,revision_id LIMIT 1",
                (
                    bundle_id,
                    scope,
                    scope_key,
                    tenant_id,
                    environment,
                    content_sha256,
                ),
            ).fetchone()
        return self._bundle(row)

    def list_bundles(
        self,
        *,
        bundle_id: str | None = None,
        scope: str | None = None,
        scope_key: str | None = None,
        tenant_id: str | None = None,
        environment: str | None = None,
        status: str | None = None,
        limit: int = 200,
        exact_dimensions: bool = False,
    ) -> list[BundleRevision]:
        clauses: list[str] = []
        args: list[Any] = []
        for column, value in (
            ("bundle_id", bundle_id),
            ("scope", scope),
            ("scope_key", scope_key),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column}=%s")
                args.append(value)
        if exact_dimensions:
            clauses.extend(
                (
                    "tenant_id IS NOT DISTINCT FROM %s",
                    "environment IS NOT DISTINCT FROM %s",
                )
            )
            args.extend((tenant_id, environment))
        else:
            if tenant_id is not None:
                clauses.append("tenant_id=%s")
                args.append(tenant_id)
            if environment is not None:
                clauses.append("environment=%s")
                args.append(environment)
        sql = f"SELECT {_POSTGRES_BUNDLE_COLUMNS} FROM configuration.bundle_revisions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY sequence DESC,created_at DESC,revision_id LIMIT %s"
        args.append(max(1, min(1000, int(limit))))
        with contextual_connection(self.pool) as conn:
            rows = conn.execute(sql, args).fetchall()
        return [item for row in rows if (item := self._bundle(row)) is not None]

    def publish_bundle(
        self, revision_id: str, *, expected_etag: int, actor_id: str
    ) -> BundleRevision:
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                f"SELECT {_POSTGRES_BUNDLE_COLUMNS} "
                "FROM configuration.bundle_revisions "
                "WHERE revision_id=%s FOR UPDATE",
                (revision_id,),
            ).fetchone()
            if row is None:
                raise KeyError(revision_id)
            current = self._bundle(row)
            assert current is not None
            if current.etag != int(expected_etag):
                raise RuntimeError("configuration revision etag conflict")
            if current.status != "draft":
                raise ValueError("only draft revisions can be published")
            self._lock_bundle_components_tx(
                conn,
                current.component_revisions,
                allowed_statuses=("published",),
            )
            logical_identity = _aggregate_identity(
                (
                    current.bundle_id,
                    current.scope,
                    current.scope_key,
                    current.tenant_id,
                    current.environment,
                )
            )
            self._lock_head(conn, "bundle", logical_identity)
            self._supersede_published_bundles_tx(
                conn,
                bundle_id=current.bundle_id,
                scope=current.scope,
                scope_key=current.scope_key,
                tenant_id=current.tenant_id,
                environment=current.environment,
                except_revision_id=revision_id,
                actor_id=actor_id,
            )
            now = utc_now()
            updated = conn.execute(
                "UPDATE configuration.bundle_revisions "
                "SET status='published',updated_at=%s::timestamptz,etag=etag+1,"
                "published_by=%s,published_at=%s::timestamptz "
                "WHERE revision_id=%s AND etag=%s AND status='draft' "
                f"RETURNING {_POSTGRES_BUNDLE_COLUMNS}",
                (now, actor_id, now, revision_id, expected_etag),
            ).fetchone()
            result = self._bundle(updated)
            if result is None:
                raise RuntimeError("configuration revision etag conflict")
            self._append_event_tx(
                conn,
                "bundle",
                revision_id,
                "published",
                actor_id,
                {"etag": result.etag, "status": result.status},
            )
            self._bump_invalidation_epoch_tx(conn)
            return result

    def set_bundle_status(
        self,
        revision_id: str,
        *,
        expected_etag: int,
        status: str,
        actor_id: str,
    ) -> BundleRevision:
        if status == "published":
            return self.publish_bundle(
                revision_id, expected_etag=expected_etag, actor_id=actor_id
            )
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                f"SELECT {_POSTGRES_BUNDLE_COLUMNS} "
                "FROM configuration.bundle_revisions "
                "WHERE revision_id=%s FOR UPDATE",
                (revision_id,),
            ).fetchone()
            if row is None:
                raise KeyError(revision_id)
            current = self._bundle(row)
            assert current is not None
            if current.etag != int(expected_etag):
                raise RuntimeError("configuration revision etag conflict")
            updated = conn.execute(
                "UPDATE configuration.bundle_revisions "
                "SET status=%s,updated_at=%s::timestamptz,etag=etag+1 "
                "WHERE revision_id=%s AND etag=%s "
                f"RETURNING {_POSTGRES_BUNDLE_COLUMNS}",
                (status, utc_now(), revision_id, expected_etag),
            ).fetchone()
            result = self._bundle(updated)
            if result is None:
                raise RuntimeError("configuration revision etag conflict")
            self._append_event_tx(
                conn,
                "bundle",
                revision_id,
                status,
                actor_id,
                {"etag": result.etag, "status": result.status},
            )
            self._bump_invalidation_epoch_tx(conn)
            return result

    def supersede_published_bundles(
        self,
        *,
        bundle_id: str,
        scope: str,
        tenant_id: str | None,
        except_revision_id: str,
        actor_id: str,
        scope_key: str = "*",
        environment: str | None = None,
    ) -> list[str]:
        logical_identity = _aggregate_identity(
            (bundle_id, scope, scope_key, tenant_id, environment)
        )
        with contextual_connection(self.pool) as conn:
            self._lock_head(conn, "bundle", logical_identity)
            changed = self._supersede_published_bundles_tx(
                conn,
                bundle_id=bundle_id,
                scope=scope,
                scope_key=scope_key,
                tenant_id=tenant_id,
                environment=environment,
                except_revision_id=except_revision_id,
                actor_id=actor_id,
            )
            if changed:
                self._bump_invalidation_epoch_tx(conn)
            return changed

    def close(self) -> None:
        self.pool.close()
