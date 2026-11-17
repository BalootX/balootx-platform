"""Immutable control-plane index for Foundry catalogs and reachability proofs."""

from __future__ import annotations

import hashlib
import base64
import json
import math
import sqlite3
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from .score_reachability import (
    ScoreReachabilityProofV1,
    ScoreReachabilityCatalogFactoryV1,
    VerifiedEnumeratedContextV1,
)
from .campaign_proof import CampaignEquivalenceProofV1
from .evidence_v3 import canonical_sha256
from .sakka_context import (
    ContextualizedLaffaV1,
    CuratedSakkaSignerV1,
    CuratedSakkaV1,
    NaturalSakkaV1,
)
from .sakka_generation import NaturalSakkaGenerationResultV1
from .simulation import NaturalLaffaSimulationResultV1
from .trace import CompleteGameTraceV1
from .trace_import import (
    TraceImportCatalogEntryV1,
    TraceImportCatalogFactoryV1,
    TraceImportCatalogV1,
    TraceImportProvenanceV1,
)


CATALOG_KINDS = frozenset(
    {
        "trace_import_catalog_v1",
        "score_reachability_catalog_v1",
        "score_reachability_proof_v1",
        "campaign_equivalence_proof_v1",
        "context_search_result_v1",
        "natural_sakka_v1",
        "curated_sakka_v1",
        "natural_sakka_generation_v1",
        "qualified_laffa_v1",
        "qualified_laffa_v2",
    }
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any, field: str) -> str:
    result = str(value or "").strip().lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return result


def _qualified_laffa_value(prepared_item: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the minimum immutable evidence required by the video library."""
    value = dict(prepared_item)
    artifact = dict(value.get("bundleArtifact") or {})
    bundle_hash = _digest(value.get("bundleHash"), "bundleHash")
    if (
        artifact.get("ownerKind") != "render_bundle_v1"
        or _digest(artifact.get("ownerHash"), "bundleArtifact.ownerHash") != bundle_hash
    ):
        raise ValueError("qualified Laffa bundle authority is invalid")
    _digest(value.get("traceHash"), "traceHash")
    _digest(value.get("candidateHash"), "candidateHash")
    if not str(value.get("decisionKind") or "").strip():
        raise ValueError("qualified Laffa decisionKind is required")
    if str(value.get("truthLabel") or "") not in {
        "exact_optimal",
        "robust_best_known",
        "debatable",
    }:
        raise ValueError("qualified Laffa truthLabel is not publishable")
    for field, minimum, maximum in (
        ("selectionScore", 0.0, 100.0),
        ("confidence", 0.0, 1.0),
    ):
        raw = value.get(field)
        if isinstance(raw, bool):
            raise ValueError(f"qualified Laffa {field} must be numeric")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"qualified Laffa {field} must be numeric") from exc
        if not math.isfinite(number) or not minimum <= number <= maximum:
            raise ValueError(f"qualified Laffa {field} is outside its range")
    laffa_score = dict(value.get("laffaSelectionScore") or {})
    if laffa_score.get("scoreFamily") != "laffa_content_selection":
        raise ValueError("qualified Laffa content-selection score is missing")
    if laffa_score.get("passed") is not True:
        raise ValueError("qualified Laffa did not pass complete-Laffa selection")
    for field in ("profileHash", "evidenceHash", "sha256"):
        _digest(laffa_score.get(field), f"laffaSelectionScore.{field}")
    raw_laffa_score = laffa_score.get("score")
    if isinstance(raw_laffa_score, bool):
        raise ValueError("qualified Laffa content-selection score must be numeric")
    try:
        numeric_laffa_score = float(raw_laffa_score)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "qualified Laffa content-selection score must be numeric"
        ) from exc
    if (
        not math.isfinite(numeric_laffa_score)
        or not 0.0 <= numeric_laffa_score <= 100.0
    ):
        raise ValueError("qualified Laffa content-selection score is outside its range")
    qualification = dict(value.get("qualification") or {})
    if qualification.get("localPreviewReady") is not True:
        raise ValueError("qualified Laffa local preview evidence is missing")
    return value


def _record(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = row["metadata_json"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return {
        "catalogHash": str(row["catalog_hash"]),
        "kind": str(row["kind"]),
        "artifact": {
            "uri": str(row["artifact_uri"]),
            "sha256": str(row["artifact_sha256"]),
            "bytes": int(row["artifact_bytes"]),
            "mediaType": str(row["media_type"]),
        },
        "metadata": dict(metadata or {}),
        "createdAt": str(row["created_at"]),
    }


_QUALIFIED_LAFFA_SORTS = {
    "laffa_score",
    "decision_score",
    "confidence",
}


def _qualified_laffa_cursor(
    *, sort: str, direction: str, value: float | str, catalog_hash: str
) -> str:
    payload = _canonical(
        {
            "schemaVersion": 1,
            "sort": sort,
            "direction": direction,
            "value": value,
            "catalogHash": _digest(catalog_hash, "catalogHash"),
        }
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _parse_qualified_laffa_cursor(
    value: str | None, *, sort: str, direction: str
) -> tuple[float | str, str] | None:
    if not value:
        return None
    try:
        padding = "=" * (-len(value) % 4)
        payload = json.loads(base64.urlsafe_b64decode(value + padding))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid qualified Laffa cursor") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schemaVersion") != 1
        or payload.get("sort") != sort
        or payload.get("direction") != direction
    ):
        raise ValueError("qualified Laffa cursor does not match the query")
    raw = payload.get("value")
    if isinstance(raw, bool):
        raise ValueError("invalid qualified Laffa numeric cursor")
    try:
        raw = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid qualified Laffa numeric cursor") from exc
    return raw, _digest(payload.get("catalogHash"), "catalogHash")


def _qualified_laffa_query_values(
    *,
    search: str,
    truth_label: str | None,
    classification: str | None,
    profile: str | None,
    sort: str,
    direction: str,
) -> tuple[str, str | None, str | None, str | None, str, str]:
    needle = str(search or "").strip().lower()
    if len(needle) > 100:
        raise ValueError("qualified Laffa search is too long")
    truth = str(truth_label).strip() if truth_label else None
    if truth not in {None, "exact_optimal", "robust_best_known", "debatable"}:
        raise ValueError("unsupported qualified Laffa truth filter")
    selected_classification = str(classification).strip() if classification else None
    if selected_classification and len(selected_classification) > 80:
        raise ValueError("qualified Laffa classification filter is too long")
    selected_profile = str(profile).strip() if profile else None
    if selected_profile and len(selected_profile) > 100:
        raise ValueError("qualified Laffa profile filter is too long")
    selected_sort = str(sort or "laffa_score")
    if selected_sort not in _QUALIFIED_LAFFA_SORTS:
        raise ValueError("unsupported qualified Laffa sort")
    selected_direction = str(direction or "desc")
    if selected_direction not in {"asc", "desc"}:
        raise ValueError("qualified Laffa direction must be asc or desc")
    return (
        needle,
        truth,
        selected_classification,
        selected_profile,
        selected_sort,
        selected_direction,
    )


def _qualified_laffa_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Build a rebuildable query projection from immutable Laffa evidence."""
    prepared = _qualified_laffa_value(value)
    return {
        "shortCode": str(prepared.get("shortCode") or ""),
        "truthLabel": str(prepared.get("truthLabel") or ""),
        "decisionKind": str(prepared.get("decisionKind") or ""),
        "laffaClassification": str(prepared.get("laffaClassification") or ""),
        "decisionSelectionScore": float(prepared.get("selectionScore") or 0.0),
        "laffaSelectionScore": float(
            dict(prepared.get("laffaSelectionScore") or {}).get("score") or 0.0
        ),
        "confidence": float(prepared.get("confidence") or 0.0),
        "interestingProfiles": list(prepared.get("interestingProfiles") or ()),
        "sourceCampaignHash": str(prepared.get("sourceCampaignHash") or ""),
    }


class SQLiteFoundryCatalogStoreV1:
    def __init__(self, path: str | Path):
        self.path = str(Path(path))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS foundry_catalogs_v1(
                    catalog_hash TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    artifact_uri TEXT NOT NULL,
                    artifact_sha256 TEXT NOT NULL,
                    artifact_bytes INTEGER NOT NULL,
                    media_type TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_foundry_catalogs_kind_v1
                  ON foundry_catalogs_v1(kind,catalog_hash);
                CREATE TABLE IF NOT EXISTS qualified_laffa_query_v1(
                    catalog_hash TEXT PRIMARY KEY,
                    short_code TEXT NOT NULL,
                    truth_label TEXT NOT NULL,
                    decision_kind TEXT NOT NULL,
                    laffa_classification TEXT NOT NULL,
                    decision_selection_score REAL NOT NULL,
                    laffa_selection_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    interesting_profiles_json TEXT NOT NULL,
                    source_campaign_hash TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_qualified_laffa_score_v1
                  ON qualified_laffa_query_v1(laffa_selection_score DESC,catalog_hash);
                CREATE INDEX IF NOT EXISTS idx_qualified_decision_score_v1
                  ON qualified_laffa_query_v1(decision_selection_score DESC,catalog_hash);
                CREATE INDEX IF NOT EXISTS idx_qualified_laffa_truth_v1
                  ON qualified_laffa_query_v1(truth_label,laffa_classification,catalog_hash);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def close(self) -> None:
        return None

    def put(
        self,
        catalog_hash: str,
        kind: str,
        artifact: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        identity = _digest(catalog_hash, "catalogHash")
        selected_kind = str(kind)
        if selected_kind not in CATALOG_KINDS:
            raise ValueError("unsupported Foundry catalog kind")
        values = (
            identity,
            selected_kind,
            str(artifact["uri"]),
            _digest(artifact["sha256"], "artifact sha256"),
            int(artifact["bytes"]),
            str(artifact["mediaType"]),
            _canonical(dict(metadata)),
        )
        if values[4] < 1 or not values[2] or not values[5]:
            raise ValueError("Foundry catalog artifact reference is incomplete")
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM foundry_catalogs_v1 WHERE catalog_hash=?", (identity,)
            ).fetchone()
            if existing is not None:
                result = _record(existing)
                if (
                    result["kind"] != selected_kind
                    or result["artifact"]
                    != {
                        "uri": values[2],
                        "sha256": values[3],
                        "bytes": values[4],
                        "mediaType": values[5],
                    }
                    or result["metadata"] != dict(metadata)
                ):
                    raise RuntimeError("immutable Foundry catalog hash collision")
                return result
            conn.execute(
                """INSERT INTO foundry_catalogs_v1(
                       catalog_hash,kind,artifact_uri,artifact_sha256,
                       artifact_bytes,media_type,metadata_json
                   ) VALUES(?,?,?,?,?,?,?)""",
                values,
            )
        return self.get(identity) or {}

    def get(self, catalog_hash: str) -> dict[str, Any] | None:
        identity = _digest(catalog_hash, "catalogHash")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM foundry_catalogs_v1 WHERE catalog_hash=?", (identity,)
            ).fetchone()
        return _record(row) if row is not None else None

    def list(
        self,
        *,
        kind: str | None = None,
        after: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        size = max(1, min(500, int(limit)))
        selected_kind = str(kind) if kind is not None else None
        if selected_kind is not None and selected_kind not in CATALOG_KINDS:
            raise ValueError("unsupported Foundry catalog kind")
        cursor = _digest(after, "after") if after else None
        clauses: list[str] = []
        parameters: list[Any] = []
        if selected_kind:
            clauses.append("kind=?")
            parameters.append(selected_kind)
        if cursor:
            clauses.append("catalog_hash>?")
            parameters.append(cursor)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM foundry_catalogs_v1{where} ORDER BY catalog_hash LIMIT ?",
                (*parameters, size),
            ).fetchall()
        return [_record(row) for row in rows]

    def put_qualified_laffa_projection(
        self, catalog_hash: str, projection: Mapping[str, Any]
    ) -> None:
        identity = _digest(catalog_hash, "catalogHash")
        values = (
            identity,
            str(projection["shortCode"]),
            str(projection["truthLabel"]),
            str(projection["decisionKind"]),
            str(projection["laffaClassification"]),
            float(projection["decisionSelectionScore"]),
            float(projection["laffaSelectionScore"]),
            float(projection["confidence"]),
            _canonical(list(projection.get("interestingProfiles") or ())),
            str(projection.get("sourceCampaignHash") or ""),
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO qualified_laffa_query_v1(
                       catalog_hash,short_code,truth_label,decision_kind,
                       laffa_classification,decision_selection_score,
                       laffa_selection_score,confidence,interesting_profiles_json,
                       source_campaign_hash
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(catalog_hash) DO UPDATE SET
                     short_code=excluded.short_code,
                     truth_label=excluded.truth_label,
                     decision_kind=excluded.decision_kind,
                     laffa_classification=excluded.laffa_classification,
                     decision_selection_score=excluded.decision_selection_score,
                     laffa_selection_score=excluded.laffa_selection_score,
                     confidence=excluded.confidence,
                     interesting_profiles_json=excluded.interesting_profiles_json,
                     source_campaign_hash=excluded.source_campaign_hash""",
                values,
            )

    def query_qualified_laffat(
        self,
        *,
        search: str = "",
        truth_label: str | None = None,
        classification: str | None = None,
        profile: str | None = None,
        sort: str = "laffa_score",
        direction: str = "desc",
        cursor: str | None = None,
        limit: int = 100,
    ) -> tuple[list[dict[str, Any]], str | None]:
        needle, truth, selected_classification, selected_profile, selected_sort, selected_direction = _qualified_laffa_query_values(
            search=search,
            truth_label=truth_label,
            classification=classification,
            profile=profile,
            sort=sort,
            direction=direction,
        )
        expressions = {
            "laffa_score": "q.laffa_selection_score",
            "decision_score": "q.decision_selection_score",
            "confidence": "q.confidence",
        }
        expression = expressions[selected_sort]
        parsed_cursor = _parse_qualified_laffa_cursor(
            cursor, sort=selected_sort, direction=selected_direction
        )
        clauses = ["c.kind=?"]
        parameters: list[Any] = ["qualified_laffa_v2"]
        if needle:
            clauses.append(
                "(LOWER(q.short_code) LIKE ? OR LOWER(q.truth_label) LIKE ? "
                "OR LOWER(q.laffa_classification) LIKE ? "
                "OR LOWER(q.interesting_profiles_json) LIKE ?)"
            )
            parameters.extend((f"%{needle}%",) * 4)
        if truth:
            clauses.append("q.truth_label=?")
            parameters.append(truth)
        if selected_classification:
            clauses.append("q.laffa_classification=?")
            parameters.append(selected_classification)
        if selected_profile:
            clauses.append(
                "EXISTS (SELECT 1 FROM json_each(q.interesting_profiles_json) WHERE value=?)"
            )
            parameters.append(selected_profile)
        if parsed_cursor is not None:
            cursor_value, cursor_hash = parsed_cursor
            operator = ">" if selected_direction == "asc" else "<"
            clauses.append(
                f"({expression}{operator}? OR ({expression}=? AND c.catalog_hash>?))"
            )
            parameters.extend((cursor_value, cursor_value, cursor_hash))
        size = max(1, min(500, int(limit)))
        order = "ASC" if selected_direction == "asc" else "DESC"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT c.*, {expression} AS query_sort_value FROM foundry_catalogs_v1 c "
                "JOIN qualified_laffa_query_v1 q ON q.catalog_hash=c.catalog_hash "
                f"WHERE {' AND '.join(clauses)} ORDER BY query_sort_value {order}, c.catalog_hash ASC LIMIT ?",
                (*parameters, size + 1),
            ).fetchall()
        selected = rows[:size]
        next_cursor = None
        if len(rows) > size and selected:
            last = selected[-1]
            next_cursor = _qualified_laffa_cursor(
                sort=selected_sort,
                direction=selected_direction,
                value=last["query_sort_value"],
                catalog_hash=str(last["catalog_hash"]),
            )
        return [_record(row) for row in selected], next_cursor


class PostgresFoundryCatalogStoreV1:
    def __init__(self, dsn: str):
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError(
                'Install with: pip install "baloot-platform[postgres]"'
            ) from exc
        self.pool = ConnectionPool(dsn, min_size=1, max_size=16, open=True)

    def close(self) -> None:
        self.pool.close()

    def put(
        self,
        catalog_hash: str,
        kind: str,
        artifact: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        identity = _digest(catalog_hash, "catalogHash")
        selected_kind = str(kind)
        if selected_kind not in CATALOG_KINDS:
            raise ValueError("unsupported Foundry catalog kind")
        artifact_hash = _digest(artifact["sha256"], "artifact sha256")
        artifact_bytes = int(artifact["bytes"])
        if (
            artifact_bytes < 1
            or not str(artifact["uri"])
            or not str(artifact["mediaType"])
        ):
            raise ValueError("Foundry catalog artifact reference is incomplete")
        from psycopg.rows import dict_row

        with (
            self.pool.connection() as conn,
            conn.transaction(),
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                "SELECT * FROM foundry_catalogs_v1 WHERE catalog_hash=%s FOR UPDATE",
                (identity,),
            )
            existing = cursor.fetchone()
            if existing is None:
                cursor.execute(
                    """INSERT INTO foundry_catalogs_v1(
                           catalog_hash,kind,artifact_uri,artifact_sha256,
                           artifact_bytes,media_type,metadata_json
                       ) VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb)""",
                    (
                        identity,
                        selected_kind,
                        str(artifact["uri"]),
                        artifact_hash,
                        artifact_bytes,
                        str(artifact["mediaType"]),
                        _canonical(dict(metadata)),
                    ),
                )
            else:
                result = _record(existing)
                expected = {
                    "uri": str(artifact["uri"]),
                    "sha256": artifact_hash,
                    "bytes": artifact_bytes,
                    "mediaType": str(artifact["mediaType"]),
                }
                if (
                    result["kind"] != selected_kind
                    or result["artifact"] != expected
                    or result["metadata"] != dict(metadata)
                ):
                    raise RuntimeError("immutable Foundry catalog hash collision")
                return result
        return self.get(identity) or {}

    def get(self, catalog_hash: str) -> dict[str, Any] | None:
        identity = _digest(catalog_hash, "catalogHash")
        from psycopg.rows import dict_row

        with (
            self.pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                "SELECT * FROM foundry_catalogs_v1 WHERE catalog_hash=%s", (identity,)
            )
            row = cursor.fetchone()
        return _record(row) if row is not None else None

    def list(
        self,
        *,
        kind: str | None = None,
        after: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        size = max(1, min(500, int(limit)))
        selected_kind = str(kind) if kind is not None else None
        if selected_kind is not None and selected_kind not in CATALOG_KINDS:
            raise ValueError("unsupported Foundry catalog kind")
        cursor_value = _digest(after, "after") if after else None
        clauses: list[str] = []
        parameters: list[Any] = []
        if selected_kind:
            clauses.append("kind=%s")
            parameters.append(selected_kind)
        if cursor_value:
            clauses.append("catalog_hash>%s")
            parameters.append(cursor_value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        from psycopg.rows import dict_row

        with (
            self.pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                f"SELECT * FROM foundry_catalogs_v1{where} ORDER BY catalog_hash LIMIT %s",
                (*parameters, size),
            )
            rows = list(cursor.fetchall())
        return [_record(row) for row in rows]

    def put_qualified_laffa_projection(
        self, catalog_hash: str, projection: Mapping[str, Any]
    ) -> None:
        identity = _digest(catalog_hash, "catalogHash")
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO qualified_laffa_query_v1(
                       catalog_hash,short_code,truth_label,decision_kind,
                       laffa_classification,decision_selection_score,
                       laffa_selection_score,confidence,interesting_profiles_json,
                       source_campaign_hash
                   ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
                   ON CONFLICT(catalog_hash) DO UPDATE SET
                     short_code=excluded.short_code,
                     truth_label=excluded.truth_label,
                     decision_kind=excluded.decision_kind,
                     laffa_classification=excluded.laffa_classification,
                     decision_selection_score=excluded.decision_selection_score,
                     laffa_selection_score=excluded.laffa_selection_score,
                     confidence=excluded.confidence,
                     interesting_profiles_json=excluded.interesting_profiles_json,
                     source_campaign_hash=excluded.source_campaign_hash""",
                (
                    identity,
                    str(projection["shortCode"]),
                    str(projection["truthLabel"]),
                    str(projection["decisionKind"]),
                    str(projection["laffaClassification"]),
                    float(projection["decisionSelectionScore"]),
                    float(projection["laffaSelectionScore"]),
                    float(projection["confidence"]),
                    _canonical(list(projection.get("interestingProfiles") or ())),
                    str(projection.get("sourceCampaignHash") or ""),
                ),
            )

    def query_qualified_laffat(
        self,
        *,
        search: str = "",
        truth_label: str | None = None,
        classification: str | None = None,
        profile: str | None = None,
        sort: str = "laffa_score",
        direction: str = "desc",
        cursor: str | None = None,
        limit: int = 100,
    ) -> tuple[list[dict[str, Any]], str | None]:
        needle, truth, selected_classification, selected_profile, selected_sort, selected_direction = _qualified_laffa_query_values(
            search=search,
            truth_label=truth_label,
            classification=classification,
            profile=profile,
            sort=sort,
            direction=direction,
        )
        expressions = {
            "laffa_score": "q.laffa_selection_score",
            "decision_score": "q.decision_selection_score",
            "confidence": "q.confidence",
        }
        expression = expressions[selected_sort]
        parsed_cursor = _parse_qualified_laffa_cursor(
            cursor, sort=selected_sort, direction=selected_direction
        )
        clauses = ["c.kind=%s"]
        parameters: list[Any] = ["qualified_laffa_v2"]
        if needle:
            clauses.append(
                "(LOWER(q.short_code) LIKE %s OR LOWER(q.truth_label) LIKE %s "
                "OR LOWER(q.laffa_classification) LIKE %s "
                "OR LOWER(q.interesting_profiles_json::text) LIKE %s)"
            )
            parameters.extend((f"%{needle}%",) * 4)
        if truth:
            clauses.append("q.truth_label=%s")
            parameters.append(truth)
        if selected_classification:
            clauses.append("q.laffa_classification=%s")
            parameters.append(selected_classification)
        if selected_profile:
            clauses.append("q.interesting_profiles_json @> %s::jsonb")
            parameters.append(_canonical([selected_profile]))
        if parsed_cursor is not None:
            cursor_value, cursor_hash = parsed_cursor
            operator = ">" if selected_direction == "asc" else "<"
            clauses.append(
                f"({expression}{operator}%s OR ({expression}=%s AND c.catalog_hash>%s))"
            )
            parameters.extend((cursor_value, cursor_value, cursor_hash))
        size = max(1, min(500, int(limit)))
        order = "ASC" if selected_direction == "asc" else "DESC"
        from psycopg.rows import dict_row

        with (
            self.pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as db_cursor,
        ):
            db_cursor.execute(
                f"SELECT c.*, {expression} AS query_sort_value FROM foundry_catalogs_v1 c "
                "JOIN qualified_laffa_query_v1 q ON q.catalog_hash=c.catalog_hash "
                f"WHERE {' AND '.join(clauses)} ORDER BY query_sort_value {order}, c.catalog_hash ASC LIMIT %s",
                (*parameters, size + 1),
            )
            rows = list(db_cursor.fetchall())
        selected = rows[:size]
        next_cursor = None
        if len(rows) > size and selected:
            last = selected[-1]
            next_cursor = _qualified_laffa_cursor(
                sort=selected_sort,
                direction=selected_direction,
                value=float(last["query_sort_value"]),
                catalog_hash=str(last["catalog_hash"]),
            )
        return [_record(row) for row in selected], next_cursor


class FoundryCatalogServiceV1:
    """Build, persist, reload, and prove immutable governed Foundry catalogs."""

    def __init__(
        self,
        store: SQLiteFoundryCatalogStoreV1 | PostgresFoundryCatalogStoreV1,
        artifact_store: Any,
        trace_import_factory: TraceImportCatalogFactoryV1,
        score_reachability_factory: ScoreReachabilityCatalogFactoryV1,
        sakka_signer: CuratedSakkaSignerV1 | None = None,
    ):
        self.store = store
        self.artifact_store = artifact_store
        self.trace_import_factory = trace_import_factory
        self.score_reachability_factory = score_reachability_factory
        self.sakka_signer = sakka_signer

    def _persist(
        self,
        *,
        identity: str,
        kind: str,
        document: Mapping[str, Any],
        media_type: str,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = _canonical(dict(document)).encode("utf-8")
        artifact = self.artifact_store.put(
            payload,
            media_type=media_type,
            name=f"{identity}.{kind}.json",
        )
        record = self.store.put(identity, kind, artifact, metadata)
        return {"record": record, "document": dict(document)}

    def _document(self, catalog_hash: str, expected_kind: str) -> dict[str, Any]:
        record = self.store.get(catalog_hash)
        if record is None or record["kind"] != expected_kind:
            raise KeyError(catalog_hash)
        reference = record["artifact"]
        data = self.artifact_store.get(reference["uri"])
        if len(data) != reference["bytes"]:
            raise ValueError("Foundry catalog artifact size mismatch")
        if hashlib.sha256(data).hexdigest() != reference["sha256"]:
            raise ValueError("Foundry catalog artifact checksum mismatch")
        value = json.loads(data)
        if not isinstance(value, Mapping):
            raise ValueError("Foundry catalog artifact must contain a JSON object")
        return dict(value)

    def create_trace_import_catalog(
        self,
        lane: str,
        entries: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        values: list[TraceImportCatalogEntryV1] = []
        for ordinal, entry in enumerate(entries):
            declared = int(entry.get("ordinal", ordinal))
            if declared != ordinal:
                raise ValueError(
                    "trace import catalog ordinals must be contiguous from zero"
                )
            provenance = TraceImportProvenanceV1.build(
                dict(entry.get("provenance") or {}),
                policy=self.trace_import_factory.policy,
            )
            values.append(
                TraceImportCatalogEntryV1.build(
                    ordinal=ordinal,
                    trace_reference=dict(entry.get("traceReference") or {}),
                    provenance=provenance,
                )
            )
        catalog = TraceImportCatalogV1.build(
            lane, values, policy=self.trace_import_factory.policy
        )
        return self._persist(
            identity=catalog.sha256,
            kind="trace_import_catalog_v1",
            document=catalog.to_dict(),
            media_type="application/vnd.balootx.trace-import-catalog-v1+json",
            metadata={
                "lane": catalog.lane,
                "entryCount": len(catalog.entries),
                "policyHash": catalog.policy_hash,
            },
        )

    def load_trace_import_catalog(self, catalog_hash: str) -> TraceImportCatalogV1:
        value = self._document(catalog_hash, "trace_import_catalog_v1")
        entries = []
        for item in value.get("entries") or ():
            provenance_value = dict(item.get("provenance") or {})
            provenance_value.pop("reviewState", None)
            provenance_value.pop("sha256", None)
            provenance = TraceImportProvenanceV1.build(
                provenance_value,
                policy=self.trace_import_factory.policy,
            )
            entries.append(
                TraceImportCatalogEntryV1.build(
                    ordinal=int(item["ordinal"]),
                    trace_reference=dict(item.get("traceReference") or {}),
                    provenance=provenance,
                )
            )
        catalog = TraceImportCatalogV1.build(
            str(value.get("lane") or ""),
            entries,
            policy=self.trace_import_factory.policy,
        )
        if catalog.sha256 != catalog_hash or value.get("sha256") != catalog.sha256:
            raise ValueError("persisted trace import catalog identity mismatch")
        return catalog

    def _load_trace_reference(self, value: Mapping[str, Any]) -> CompleteGameTraceV1:
        reference = dict(value)
        if (
            reference.get("ownerKind") != "complete_trace_v1"
            or reference.get("mediaType")
            != "application/vnd.balootx.complete-trace-v1+json"
        ):
            raise ValueError(
                "score reachability requires complete-trace artifact references"
            )
        owner = _digest(reference.get("ownerHash"), "trace ownerHash")
        artifact_hash = _digest(reference.get("sha256"), "trace artifact sha256")
        data = self.artifact_store.get(str(reference.get("uri") or ""))
        if len(data) != int(reference.get("bytes", -1)):
            raise ValueError("score reachability trace artifact size mismatch")
        if hashlib.sha256(data).hexdigest() != artifact_hash:
            raise ValueError("score reachability trace artifact checksum mismatch")
        trace = CompleteGameTraceV1.from_dict(json.loads(data))
        if trace.sha256 != owner:
            raise ValueError("score reachability trace owner identity mismatch")
        return trace

    def load_complete_trace(self, value: Mapping[str, Any]) -> CompleteGameTraceV1:
        """Load and verify a content-addressed complete trace reference."""
        return self._load_trace_reference(value)

    def persist_complete_trace(self, trace: CompleteGameTraceV1) -> dict[str, Any]:
        """Persist one canonical trusted trace and return its verified owner reference."""
        verified = CompleteGameTraceV1.from_dict(trace.to_dict(trusted=True))
        payload = _canonical(verified.to_dict(trusted=True)).encode("utf-8")
        artifact = self.artifact_store.put(
            payload,
            media_type="application/vnd.balootx.complete-trace-v1+json",
            name=f"{verified.sha256}.complete_trace_v1.json",
        )
        return {
            "ownerKind": "complete_trace_v1",
            "ownerHash": verified.sha256,
            **artifact,
        }

    def create_score_reachability_catalog(
        self,
        trace_references: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        references = tuple(
            sorted(
                (dict(item) for item in trace_references),
                key=lambda item: item["ownerHash"],
            )
        )
        if len({str(item.get("ownerHash")) for item in references}) != len(references):
            raise ValueError("score reachability witness references must be unique")
        traces = tuple(self._load_trace_reference(item) for item in references)
        engine = self.score_reachability_factory.build(traces)
        catalog = engine.catalog
        document = {
            "schemaVersion": 1,
            "catalog": catalog.to_dict(),
            "witnessTraceReferences": list(references),
        }
        return self._persist(
            identity=catalog.sha256,
            kind="score_reachability_catalog_v1",
            document=document,
            media_type="application/vnd.balootx.score-reachability-catalog-v1+json",
            metadata={
                "witnessCount": len(references),
                "outcomeCount": len(catalog.outcomes),
                "rulesHash": catalog.rules_hash,
                "bundleHash": catalog.bundle_hash,
                "engineVersion": catalog.engine_version,
                "profileHash": catalog.profile_hash,
            },
        )

    def _load_score_reachability_engine(self, catalog_hash: str):
        value = self._document(catalog_hash, "score_reachability_catalog_v1")
        traces = tuple(
            self._load_trace_reference(item)
            for item in value.get("witnessTraceReferences") or ()
        )
        engine = self.score_reachability_factory.build(traces)
        declared = dict(value.get("catalog") or {})
        if (
            engine.catalog.sha256 != catalog_hash
            or declared != engine.catalog.to_dict()
        ):
            raise ValueError("persisted score reachability catalog identity mismatch")
        return engine

    def prove_score_context(
        self,
        catalog_hash: str,
        target_score: tuple[int, int],
        *,
        target_dealer: str,
        initial_dealer: str,
    ) -> dict[str, Any]:
        engine = self._load_score_reachability_engine(catalog_hash)
        proof = engine.prove(
            target_score,
            target_dealer=target_dealer,
            initial_dealer=initial_dealer,
        )
        verified = VerifiedEnumeratedContextV1.build(proof)
        document = {
            "schemaVersion": 1,
            "catalogHash": catalog_hash,
            "proof": proof.to_dict(),
            "verifiedContext": verified.to_dict(),
        }
        return self._persist(
            identity=proof.sha256,
            kind="score_reachability_proof_v1",
            document=document,
            media_type="application/vnd.balootx.score-reachability-proof-v1+json",
            metadata={
                "catalogHash": catalog_hash,
                "contextHash": verified.context.sha256,
                "targetScore": list(proof.target_score),
                "targetDealer": proof.target_dealer.name.lower(),
                "initialDealer": proof.initial_dealer.name.lower(),
                "roundNumber": proof.round_number,
            },
        )

    def load_verified_score_context(
        self, proof_hash: str
    ) -> VerifiedEnumeratedContextV1:
        value = self._document(proof_hash, "score_reachability_proof_v1")
        proof = ScoreReachabilityProofV1.from_dict(dict(value.get("proof") or {}))
        verified = VerifiedEnumeratedContextV1.build(proof)
        if proof.sha256 != proof_hash:
            raise ValueError("persisted score reachability proof identity mismatch")
        if dict(value.get("verifiedContext") or {}) != verified.to_dict():
            raise ValueError("persisted verified context identity mismatch")
        return verified

    def persist_context_search_result(self, result: Any) -> dict[str, Any]:
        descendant_references: list[dict[str, Any]] = []
        for trace in result.descendant_traces:
            descendant_references.append(self.persist_complete_trace(trace))
        document = {
            "schemaVersion": 1,
            "result": result.to_dict(),
            "descendantTraceReferences": descendant_references,
        }
        return self._persist(
            identity=result.sha256,
            kind="context_search_result_v1",
            document=document,
            media_type="application/vnd.balootx.context-search-result-v1+json",
            metadata={
                "parentTraceHash": result.parent_trace_hash,
                "originalDecisionId": result.original_decision_id,
                "evaluatedCount": len(result.evaluated),
                "acceptedCount": len(result.accepted),
                "selectedHash": result.selected_hash,
            },
        )

    def create_natural_sakka(
        self,
        trace_references: Sequence[Mapping[str, Any]],
        *,
        target: int,
        attempt_trace_reference_groups: Sequence[Sequence[Mapping[str, Any]]]
        | None = None,
    ) -> dict[str, Any]:
        references = tuple(dict(item) for item in trace_references)
        traces = tuple(self._load_trace_reference(item) for item in references)
        reference_groups = (
            tuple(
                tuple(dict(reference) for reference in group)
                for group in attempt_trace_reference_groups
            )
            if attempt_trace_reference_groups is not None
            else None
        )
        trace_groups = (
            tuple(
                tuple(self._load_trace_reference(reference) for reference in group)
                for group in reference_groups
            )
            if reference_groups is not None
            else None
        )
        sakka = NaturalSakkaV1.build(
            traces,
            target=target,
            attempt_trace_groups=trace_groups,
        )
        document = {
            "schemaVersion": 1,
            "sakka": sakka.to_dict(),
            "traceReferences": list(references),
        }
        if reference_groups is not None:
            document["attemptTraceReferenceGroups"] = [
                list(group) for group in reference_groups
            ]
        return self._persist(
            identity=sakka.sha256,
            kind="natural_sakka_v1",
            document=document,
            media_type="application/vnd.balootx.natural-sakka-v1+json",
            metadata={
                "provenance": sakka.provenance,
                "roundCount": len(sakka.rounds),
                "target": sakka.target,
                "winner": sakka.winner,
                "finalScore": list(sakka.final_score),
                "rulesHash": sakka.rules_hash,
                "bundleHash": sakka.bundle_hash,
            },
        )

    def persist_natural_sakka_generation(
        self,
        result: NaturalSakkaGenerationResultV1,
    ) -> dict[str, Any]:
        reference_groups = [
            [
                self.persist_complete_trace(trace)
                for trace in round_result.attempt_traces
            ]
            for round_result in result.rounds
        ]
        sakka_record = self.create_natural_sakka(
            tuple(group[-1] for group in reference_groups),
            target=result.sakka.target,
            attempt_trace_reference_groups=reference_groups,
        )
        document = {
            "schemaVersion": 1,
            "generation": result.to_dict(),
            "sakkaCatalogHash": sakka_record["record"]["catalogHash"],
            "attemptTraceReferenceGroups": reference_groups,
        }
        return self._persist(
            identity=result.sha256,
            kind="natural_sakka_generation_v1",
            document=document,
            media_type="application/vnd.balootx.natural-sakka-generation-v1+json",
            metadata={
                "sakkaHash": result.sakka.sha256,
                "rootSeed": result.root_seed,
                "initialDealer": result.initial_dealer.name.lower(),
                "profileHash": result.profile_hash,
                "roundCount": len(result.rounds),
                "finalScore": list(result.sakka.final_score),
                "winner": result.sakka.winner,
            },
        )

    def load_natural_sakka_generation(
        self,
        generation_hash: str,
    ) -> NaturalSakkaGenerationResultV1:
        value = self._document(generation_hash, "natural_sakka_generation_v1")
        declared = dict(value.get("generation") or {})
        groups = tuple(
            tuple(self._load_trace_reference(reference) for reference in group)
            for group in value.get("attemptTraceReferenceGroups") or ()
        )
        round_values = tuple(declared.get("roundResults") or ())
        if not groups or len(groups) != len(round_values):
            raise ValueError("natural Sakka generation trace-group count mismatch")
        rounds = tuple(
            NaturalLaffaSimulationResultV1.build(
                root_seed=int(item["rootSeed"]),
                attempt_traces=group,
                publishable_laffa=bool(item["publishableLaffa"]),
                coverage_cells=tuple(item.get("coverageCells") or ()),
            )
            for item, group in zip(round_values, groups)
        )
        if [item.to_summary_dict() for item in rounds] != list(round_values):
            raise ValueError("natural Sakka generation round-result identity mismatch")
        sakka_hash = str(value.get("sakkaCatalogHash") or "")
        sakka = self.load_sakka(sakka_hash)
        if not isinstance(sakka, NaturalSakkaV1):
            raise ValueError("natural Sakka generation references the wrong Sakka kind")
        rebuilt = NaturalSakkaGenerationResultV1.build(
            root_seed=int(declared["rootSeed"]),
            initial_dealer=str(declared["initialDealer"]),
            profile_hash=str(declared["profileHash"]),
            sakka=sakka,
            rounds=rounds,
            round_seeds=tuple(declared.get("roundSeeds") or ()),
        )
        if rebuilt.sha256 != generation_hash or rebuilt.to_dict() != declared:
            raise ValueError("persisted natural Sakka generation identity mismatch")
        return rebuilt

    def _load_contextualized_selection(
        self,
        search_result_hash: str,
        contextualized_laffa_hash: str,
    ) -> tuple[ContextualizedLaffaV1, CompleteGameTraceV1, dict[str, Any]]:
        result_hash = _digest(search_result_hash, "contextSearchResultHash")
        selected_hash = _digest(contextualized_laffa_hash, "contextualizedLaffaHash")
        document = self._document(result_hash, "context_search_result_v1")
        result = dict(document.get("result") or {})
        accepted_values = tuple(result.get("accepted") or ())
        references = tuple(document.get("descendantTraceReferences") or ())
        if len(accepted_values) != len(references):
            raise ValueError("context search descendant reference count mismatch")
        accepted = tuple(
            ContextualizedLaffaV1.from_dict(item) for item in accepted_values
        )
        if len({item.sha256 for item in accepted}) != len(accepted):
            raise ValueError("context search accepted identities must be unique")
        declared_selected = result.get("selectedHash")
        if declared_selected is not None:
            declared_selected = _digest(declared_selected, "selectedHash")
            if declared_selected not in {item.sha256 for item in accepted}:
                raise ValueError("context search selected identity is not accepted")
        result_core = {
            "schemaVersion": 1,
            "parentTraceHash": _digest(
                result.get("parentTraceHash"), "parentTraceHash"
            ),
            "originalDecisionId": str(result.get("originalDecisionId") or ""),
            "profileHash": _digest(result.get("profileHash"), "profileHash"),
            "analysisProfileHash": _digest(
                result.get("analysisProfileHash"), "analysisProfileHash"
            ),
            "evaluated": [dict(item) for item in result.get("evaluated") or ()],
            "acceptedHashes": [item.sha256 for item in accepted],
            "selectedHash": declared_selected,
        }
        if (
            canonical_sha256(result_core) != result_hash
            or result.get("sha256") != result_hash
        ):
            raise ValueError("persisted context search result identity mismatch")
        for item, reference in zip(accepted, references):
            if item.sha256 != selected_hash:
                continue
            trace = self._load_trace_reference(reference)
            if trace.sha256 != item.descendant_trace_hash:
                raise ValueError("contextualized Laffa descendant reference mismatch")
            return item, trace, dict(reference)
        raise KeyError(selected_hash)

    def create_curated_sakka(
        self,
        selections: Sequence[Mapping[str, Any]],
        *,
        target: int,
    ) -> dict[str, Any]:
        if self.sakka_signer is None:
            raise RuntimeError("curated Sakka signing is not configured")
        selected = tuple(dict(item) for item in selections)
        if not selected:
            raise ValueError("curated Sakka requires at least one selection")
        loaded = tuple(
            self._load_contextualized_selection(
                str(item.get("contextSearchResultHash") or ""),
                str(item.get("contextualizedLaffaHash") or ""),
            )
            for item in selected
        )
        contexts = tuple(item[0] for item in loaded)
        traces = tuple(item[1] for item in loaded)
        references = tuple(item[2] for item in loaded)
        sakka = CuratedSakkaV1.build(
            contexts,
            traces,
            target=target,
            signer=self.sakka_signer,
        )
        document = {
            "schemaVersion": 1,
            "sakka": sakka.to_dict(),
            "selections": selected,
            "descendantTraceReferences": list(references),
        }
        return self._persist(
            identity=sakka.composition_hash,
            kind="curated_sakka_v1",
            document=document,
            media_type="application/vnd.balootx.curated-sakka-v1+json",
            metadata={
                "provenance": sakka.provenance,
                "roundCount": len(sakka.rounds),
                "target": sakka.target,
                "winner": sakka.winner,
                "finalScore": list(sakka.final_score),
                "rulesHash": sakka.rules_hash,
                "bundleHash": sakka.bundle_hash,
                "signatureKeyId": sakka.signature_key_id,
            },
        )

    def load_sakka(self, sakka_hash: str) -> NaturalSakkaV1 | CuratedSakkaV1:
        record = self.store.get(sakka_hash)
        if record is None or record["kind"] not in {
            "natural_sakka_v1",
            "curated_sakka_v1",
        }:
            raise KeyError(sakka_hash)
        value = self._document(sakka_hash, record["kind"])
        if record["kind"] == "natural_sakka_v1":
            traces = tuple(
                self._load_trace_reference(item)
                for item in value.get("traceReferences") or ()
            )
            declared_groups = value.get("attemptTraceReferenceGroups")
            trace_groups = (
                tuple(
                    tuple(self._load_trace_reference(item) for item in group)
                    for group in declared_groups
                )
                if declared_groups is not None
                else None
            )
            rebuilt = NaturalSakkaV1.build(
                traces,
                target=int(value["sakka"]["target"]),
                attempt_trace_groups=trace_groups,
            )
            if rebuilt.sha256 != sakka_hash or rebuilt.to_dict() != value.get("sakka"):
                raise ValueError("persisted natural Sakka identity mismatch")
            return rebuilt
        if self.sakka_signer is None:
            raise RuntimeError("curated Sakka signing is not configured")
        selected = tuple(value.get("selections") or ())
        contexts = []
        traces = []
        references = []
        for item in selected:
            context, trace, reference = self._load_contextualized_selection(
                item["contextSearchResultHash"], item["contextualizedLaffaHash"]
            )
            contexts.append(context)
            traces.append(trace)
            references.append(reference)
        rebuilt = CuratedSakkaV1.build(
            contexts,
            traces,
            target=int(value["sakka"]["target"]),
            signer=self.sakka_signer,
        )
        expected = {
            "schemaVersion": 1,
            "sakka": rebuilt.to_dict(),
            "selections": list(selected),
            "descendantTraceReferences": references,
        }
        if rebuilt.composition_hash != sakka_hash or expected != value:
            raise ValueError("persisted curated Sakka identity mismatch")
        return rebuilt

    def list_sakkas(
        self,
        *,
        provenance: str | None = None,
        after: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        if provenance not in {None, "natural", "curated"}:
            raise ValueError("unsupported Sakka provenance filter")
        kinds = (
            ("natural_sakka_v1", "curated_sakka_v1")
            if provenance is None
            else (
                ("natural_sakka_v1",)
                if provenance == "natural"
                else ("curated_sakka_v1",)
            )
        )
        values = [
            item
            for kind in kinds
            for item in self.store.list(kind=kind, after=after, limit=500)
        ]
        values.sort(key=lambda item: item["catalogHash"])
        size = max(1, min(500, int(limit)))
        selected = values[:size]
        return {
            "resourceKind": "foundry_sakka",
            "items": selected,
            "nextCursor": selected[-1]["catalogHash"] if len(values) > size else None,
        }

    def create_campaign_equivalence_proof(
        self,
        proof: CampaignEquivalenceProofV1,
    ) -> dict[str, Any]:
        return self._persist(
            identity=proof.sha256,
            kind="campaign_equivalence_proof_v1",
            document=proof.to_dict(),
            media_type="application/vnd.balootx.campaign-equivalence-proof-v1+json",
            metadata={
                "workloadHash": proof.workload_hash,
                "profileHash": proof.profile_hash,
                "executionCount": len(proof.executions),
                "normalizedEvidenceHash": proof.normalized_evidence_hash,
                "status": proof.status,
            },
        )

    def get(self, catalog_hash: str) -> dict[str, Any]:
        record = self.store.get(catalog_hash)
        if record is None:
            raise KeyError(catalog_hash)
        return {
            "record": record,
            "document": self._document(catalog_hash, record["kind"]),
        }

    def list(
        self,
        *,
        kind: str | None = None,
        after: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        size = max(1, min(500, int(limit)))
        values = self.store.list(kind=kind, after=after, limit=size)
        return {
            "items": values,
            "nextCursor": values[-1]["catalogHash"] if len(values) == size else None,
        }

    def put_qualified_laffa(self, prepared_item: Mapping[str, Any]) -> dict[str, Any]:
        """Persist one immutable, render-ready Laffa as a durable library item."""
        value = _qualified_laffa_value(prepared_item)
        projection = _qualified_laffa_projection(value)
        bundle_hash = _digest(value.get("bundleHash"), "bundleHash")
        laffa_score_hash = _digest(
            dict(value["laffaSelectionScore"]).get("sha256"),
            "laffaSelectionScore.sha256",
        )
        identity = hashlib.sha256(
            _canonical(
                {
                    "bundleHash": bundle_hash,
                    "laffaSelectionScoreHash": laffa_score_hash,
                }
            ).encode("utf-8")
        ).hexdigest()
        document = {"schemaVersion": 2, "preparedItem": value}
        existing = self.store.get(identity)
        if existing is not None:
            if existing["kind"] != "qualified_laffa_v2":
                raise RuntimeError("immutable qualified Laffa catalog kind collision")
            persisted = self._document(identity, "qualified_laffa_v2")
            if persisted != document:
                raise RuntimeError("immutable qualified Laffa document collision")
            self.store.put_qualified_laffa_projection(identity, projection)
            return {"record": existing, "document": document}
        created = self._persist(
            identity=identity,
            kind="qualified_laffa_v2",
            document=document,
            media_type="application/vnd.balootx.qualified-laffa-v2+json",
            metadata={
                "shortCode": str(value.get("shortCode") or ""),
                "candidateHash": str(value.get("candidateHash") or ""),
                "truthLabel": str(value.get("truthLabel") or ""),
                "decisionKind": str(value.get("decisionKind") or ""),
                "laffaClassification": str(value.get("laffaClassification") or ""),
                "decisionSelectionScore": float(value.get("selectionScore") or 0.0),
                "laffaSelectionScore": float(
                    dict(value.get("laffaSelectionScore") or {}).get("score") or 0.0
                ),
                "confidence": float(value.get("confidence") or 0.0),
                "interestingProfiles": list(value.get("interestingProfiles") or ()),
                "sourceCampaignHash": str(value.get("sourceCampaignHash") or ""),
            },
        )
        self.store.put_qualified_laffa_projection(identity, projection)
        return created

    def list_qualified_laffat(
        self,
        *,
        search: str = "",
        truth_label: str | None = None,
        classification: str | None = None,
        profile: str | None = None,
        sort: str = "laffa_score",
        direction: str = "desc",
        cursor: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        size = max(1, min(500, int(limit)))
        records, next_cursor = self.store.query_qualified_laffat(
            search=search,
            truth_label=truth_label,
            classification=classification,
            profile=profile,
            sort=sort,
            direction=direction,
            cursor=cursor,
            limit=size,
        )
        items = []
        for record in records:
            value = dict(
                self._document(record["catalogHash"], "qualified_laffa_v2")[
                    "preparedItem"
                ]
            )
            try:
                items.append(_qualified_laffa_value(value))
            except ValueError:
                # Historical records remain immutable audit evidence, but an
                # incomplete record must never inflate the usable library.
                continue
        return {
            "items": items,
            "nextCursor": next_cursor,
        }
