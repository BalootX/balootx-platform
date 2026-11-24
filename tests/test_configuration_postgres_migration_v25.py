from __future__ import annotations

import hashlib
import re
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from scripts import migrate_postgres


MIGRATION_NAME = "025_configuration_store_contract.sql"
LATEST_MIGRATION_NAME = "033_oidc_database_roles.sql"
FROZEN_PREDECESSOR_HASHES = {
    "011_configuration_registry_v19.sql": (
        "1e79c94b6244244afe0a2fd1f9ef421dcfadca6c32a48bd4c7f4b8fee97b9931"
    ),
    "012_v20_security_configuration_live.sql": (
        "797e87f438fc70b001ab5c104e33f07ba7cc525dac1281e6e399cbd14c943c2e"
    ),
    "013_v21_configuration_rollback.sql": (
        "eb395358736e5b494bb5bc44bcc3d7774a8c08aa429127d13886c514b580d099"
    ),
    "024_research_fenced_search_receipts.sql": (
        "913a41c16aaa695705e7f33b4fe579396eaef6c04280a538c34470adae84e33b"
    ),
    "025_configuration_store_contract.sql": (
        "a72345cb2593df997f8f4fb839450a8f34f5672c9ed1f51fdc3f1f5aa108a9fe"
    ),
    "026_research_scenario_v4_projections.sql": (
        "65ea8bb88ac5b22de45d9917b65d2b64203422d942b8e018e2d4208058fba3db"
    ),
    "027_research_result_library_projection.sql": (
        "462ff77337f9f041713130d907e74ff3442916517cf395751950ab44805464c4"
    ),
}


def _migration(root: Path) -> tuple[Path, str]:
    path = root / "migrations" / MIGRATION_NAME
    return path, path.read_text(encoding="utf-8")


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().upper()


def _statement(sql: str, marker: str) -> str:
    normalized = _normalized(sql)
    start = normalized.index(marker.upper())
    return normalized[start : normalized.index(";", start) + 1]


def _function_body(sql: str, function_name: str) -> str:
    normalized = _normalized(sql)
    start = normalized.index(f"CREATE OR REPLACE FUNCTION {function_name.upper()}")
    end = normalized.index("DROP TRIGGER", start)
    return normalized[start:end]


def test_v25_authored_and_packaged_inventory_are_byte_identical(root: Path) -> None:
    authored = root / "migrations"
    packaged = root / "src/baloot_platform/resources/migrations"
    authored_names = sorted(path.name for path in authored.glob("*.sql"))
    packaged_names = sorted(path.name for path in packaged.glob("*.sql"))

    assert authored_names == packaged_names
    assert MIGRATION_NAME in authored_names
    assert authored_names[-1] == LATEST_MIGRATION_NAME
    for name in authored_names:
        assert (authored / name).read_bytes() == (packaged / name).read_bytes()


def test_latest_shared_run_access_migration_matches_store_bootstrap(
    root: Path,
) -> None:
    sql = (root / "migrations" / "029_shared_run_game_access.sql").read_text(encoding="utf-8")
    normalized = _normalized(sql)

    assert normalized.startswith("BEGIN;")
    assert normalized.endswith("COMMIT;")
    assert "PG_ADVISORY_XACT_LOCK" in normalized
    assert "ALTER TABLE DUPLICATE.RUNS" in normalized
    assert "ADD COLUMN IF NOT EXISTS GAME_ID TEXT" in normalized
    assert "CREATE UNIQUE INDEX IF NOT EXISTS IDX_DUPLICATE_RUNS_GAME" in normalized
    assert "DROP " not in normalized
    assert "DELETE " not in normalized
    assert "TRUNCATE " not in normalized


def test_v25_preserves_published_predecessor_migration_bytes(root: Path) -> None:
    authored = root / "migrations"
    packaged = root / "src/baloot_platform/resources/migrations"
    for name, expected_hash in FROZEN_PREDECESSOR_HASHES.items():
        source = (authored / name).read_bytes()
        assert hashlib.sha256(source).hexdigest() == expected_hash
        assert (packaged / name).read_bytes() == source


def test_full_history_bootstrap_order_is_exactly_repeatable(
    root: Path, monkeypatch
) -> None:
    expected_paths = sorted((root / "migrations").glob("*.sql"))
    expected_sql = [path.read_text(encoding="utf-8") for path in expected_paths]
    executed: list[str] = []

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql: str) -> None:
            executed.append(sql)

    fake_psycopg = SimpleNamespace(connect=lambda _dsn: FakeConnection())
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setenv("BALOOT_DATABASE_URL", "postgresql://bootstrap.invalid/test")
    monkeypatch.setenv("BALOOT_ROOT", str(root))

    migrate_postgres.main()
    migrate_postgres.main()

    assert executed == expected_sql + expected_sql
    assert any(path.name == MIGRATION_NAME for path in expected_paths)
    assert expected_paths[-1].name == LATEST_MIGRATION_NAME


def test_v25_is_forward_only_and_schema_idempotent_by_construction(root: Path) -> None:
    _, sql = _migration(root)
    normalized = _normalized(sql)

    assert normalized.startswith("BEGIN;")
    assert normalized.endswith("COMMIT;")
    assert "PG_ADVISORY_XACT_LOCK" in normalized
    assert "SET_CONFIG('APP.ROLE','SERVICE',TRUE)" in normalized
    assert "DROP TABLE" not in normalized
    assert re.search(r"\bTRUNCATE\s+(?:TABLE\s+)?CONFIGURATION\.", normalized) is None
    assert re.search(r"\bDELETE\s+FROM\s+CONFIGURATION\.", normalized) is None
    assert normalized.count("CREATE OR REPLACE FUNCTION") == 2
    assert "ON CONFLICT(NAMESPACE) DO NOTHING" in normalized

    for constraint in (
        "configuration_profile_scope_v25",
        "configuration_profile_dimensions_v25",
        "configuration_profile_rollout_v25",
        "configuration_bundle_scope_v25",
        "configuration_bundle_dimensions_v25",
        "configuration_bundle_rollout_v25",
    ):
        assert f"DROP CONSTRAINT IF EXISTS {constraint.upper()}" in normalized
        assert f"ADD CONSTRAINT {constraint.upper()}" in normalized

    for trigger in (
        "trg_configuration_profile_immutable",
        "trg_configuration_bundle_immutable",
    ):
        assert f"DROP TRIGGER IF EXISTS {trigger.upper()}" in normalized
        assert f"CREATE TRIGGER {trigger.upper()}" in normalized


def test_v25_populated_upgrade_retains_rows_and_reconciles_legacy_state(
    root: Path,
) -> None:
    _, sql = _migration(root)
    normalized = _normalized(sql)

    assert normalized.count("SET SCOPE_KEY=TENANT_ID") == 2
    assert normalized.count("SCOPE='TENANT'") >= 4
    assert normalized.count("AS DUPLICATE_RANK") == 2
    assert normalized.count("AS PRIOR_MAXIMUM") == 2
    assert normalized.count("AS NEW_SEQUENCE") == 2
    assert normalized.count("WHERE DUPLICATE_RANK>1") == 2
    assert normalized.count("SET SEQUENCE=REASSIGNED.NEW_SEQUENCE") == 2
    assert normalized.count("ROW_NUMBER() OVER") == 6
    assert (
        "PARTITION BY FAMILY,PROFILE_ID,SCOPE,SCOPE_KEY,TENANT_ID,ENVIRONMENT"
        in normalized
    )
    assert (
        "PARTITION BY BUNDLE_ID,SCOPE,SCOPE_KEY,TENANT_ID,ENVIRONMENT"
        in normalized
    )
    assert normalized.count("SET STATUS='SUPERSEDED'") == 2
    assert "ORDER BY SEQUENCE DESC,CREATED_AT DESC,REVISION_ID DESC" in normalized
    assert (
        "VALUES('CONFIGURATION',1,TIMESTAMPTZ '1980-01-01T00:00:00+00:00')"
        in normalized
    )


def test_v25_rollout_type_and_exact_bounds_match_runtime(root: Path) -> None:
    _, sql = _migration(root)
    normalized = _normalized(sql)

    assert normalized.count("ALTER COLUMN ROLLOUT_PERCENTAGE TYPE NUMERIC") == 2
    assert normalized.count("USING ROLLOUT_PERCENTAGE::NUMERIC") == 2
    assert "DOUBLE PRECISION" not in normalized
    assert normalized.count(
        "ROLLOUT_PERCENTAGE>0::NUMERIC AND ROLLOUT_PERCENTAGE<=100::NUMERIC"
    ) == 2

    def accepts(value: str) -> bool:
        return Decimal("0") < Decimal(value) <= Decimal("100")

    assert not accepts("0")
    assert accepts("0.125")
    assert accepts("99.999999")
    assert accepts("100")
    assert not accepts("100.000001")


def test_v25_scope_constraints_match_canonical_normalization(root: Path) -> None:
    _, sql = _migration(root)
    normalized = _normalized(sql)

    for scope in (
        "'GLOBAL'",
        "'ENVIRONMENT'",
        "'TENANT'",
        "'USER_EXPERIMENT'",
        "'ROOM'",
        "'SHARED_GAME'",
        "'VIDEO_PROFILE'",
    ):
        assert normalized.count(scope) >= 2
    assert normalized.count(
        "SCOPE='GLOBAL' AND SCOPE_KEY='*' AND TENANT_ID IS NULL"
    ) == 2
    assert normalized.count(
        "SCOPE='ENVIRONMENT' AND TENANT_ID IS NULL AND ENVIRONMENT IS NOT NULL "
        "AND BTRIM(ENVIRONMENT)<>'' AND SCOPE_KEY=ENVIRONMENT"
    ) == 2
    assert normalized.count(
        "SCOPE='TENANT' AND TENANT_ID IS NOT NULL AND BTRIM(TENANT_ID)<>'' "
        "AND SCOPE_KEY=TENANT_ID"
    ) == 2


def test_v25_indexes_use_exact_nullable_runtime_identities(root: Path) -> None:
    _, sql = _migration(root)

    profile_content = _statement(
        sql, "CREATE UNIQUE INDEX uq_configuration_profile_content"
    )
    assert (
        "( FAMILY,SCOPE,SCOPE_KEY,TENANT_ID,ENVIRONMENT,PAYLOAD_SHA256 )"
        in profile_content
    )
    assert "NULLS NOT DISTINCT" in profile_content
    assert "PROFILE_ID" not in profile_content
    assert "COALESCE" not in profile_content

    profile_revision = _statement(
        sql, "CREATE UNIQUE INDEX uq_configuration_profile_v20"
    )
    assert (
        "( FAMILY,PROFILE_ID,SCOPE,SCOPE_KEY,TENANT_ID,ENVIRONMENT,SEQUENCE )"
        in profile_revision
    )
    assert "NULLS NOT DISTINCT" in profile_revision
    assert "VERSION" not in profile_revision
    assert "COALESCE" not in profile_revision

    bundle_revision = _statement(
        sql, "CREATE UNIQUE INDEX uq_configuration_bundle_v20"
    )
    assert (
        "( BUNDLE_ID,SCOPE,SCOPE_KEY,TENANT_ID,ENVIRONMENT,SEQUENCE )"
        in bundle_revision
    )
    assert "NULLS NOT DISTINCT" in bundle_revision
    assert "VERSION" not in bundle_revision
    assert "COALESCE" not in bundle_revision

    bundle_content = _statement(
        sql, "CREATE INDEX uq_configuration_bundle_content"
    )
    assert "CREATE UNIQUE INDEX" not in bundle_content
    assert (
        "( BUNDLE_ID,SCOPE,SCOPE_KEY,TENANT_ID,ENVIRONMENT, "
        "CONTENT_SHA256,SEQUENCE DESC,REVISION_ID )"
        in bundle_content
    )
    assert "COALESCE" not in bundle_content

    for marker, identity in (
        (
            "CREATE UNIQUE INDEX uq_configuration_profile_published_v25",
            "FAMILY,PROFILE_ID,SCOPE,SCOPE_KEY,TENANT_ID,ENVIRONMENT",
        ),
        (
            "CREATE UNIQUE INDEX uq_configuration_bundle_published_v25",
            "BUNDLE_ID,SCOPE,SCOPE_KEY,TENANT_ID,ENVIRONMENT",
        ),
    ):
        published = _statement(sql, marker)
        assert identity in published
        assert "NULLS NOT DISTINCT" in published
        assert "WHERE STATUS='PUBLISHED'" in published


def test_v25_historical_index_names_are_safe_full_replay_sentinels(root: Path) -> None:
    _, sql = _migration(root)
    normalized = _normalized(sql)

    # V11/V12 use IF NOT EXISTS with these names.  V25 intentionally retains
    # the corrected definitions under those names so populated 001..025 replay
    # cannot recreate an obsolete, over-broad uniqueness contract first.
    for name in (
        "uq_configuration_profile_content",
        "uq_configuration_profile_v20",
        "uq_configuration_bundle_v20",
        "uq_configuration_bundle_content",
    ):
        assert f"DROP INDEX IF EXISTS CONFIGURATION.{name.upper()}" in normalized
        assert f"INDEX {name.upper()}" in normalized


def test_v25_published_revision_triggers_freeze_complete_provenance(
    root: Path,
) -> None:
    _, sql = _migration(root)
    profile = _function_body(
        sql, "configuration.guard_published_profile_payload()"
    )
    bundle = _function_body(
        sql, "configuration.guard_published_bundle_payload()"
    )

    shared_fields = {
        "revision_id",
        "version",
        "scope",
        "scope_key",
        "tenant_id",
        "environment",
        "sequence",
        "effective_from",
        "effective_until",
        "rollout_percentage",
        "rollout_seed",
        "created_by",
        "created_at",
        "published_by",
        "published_at",
        "supersedes_revision_id",
    }
    for field in shared_fields | {"family", "profile_id", "payload", "payload_sha256"}:
        assert f"NEW.{field.upper()} IS DISTINCT FROM OLD.{field.upper()}" in profile
    for field in shared_fields | {
        "bundle_id",
        "mode",
        "components",
        "content_sha256",
    }:
        assert f"NEW.{field.upper()} IS DISTINCT FROM OLD.{field.upper()}" in bundle

    # Lifecycle transitions remain possible; their provenance fields do not.
    assert "NEW.STATUS IS DISTINCT FROM OLD.STATUS" not in profile
    assert "NEW.STATUS IS DISTINCT FROM OLD.STATUS" not in bundle


def test_v25_rls_removes_permissive_legacy_union_and_restores_forced_authority(
    root: Path,
) -> None:
    _, sql = _migration(root)
    normalized = _normalized(sql)

    for legacy_policy in (
        "configuration_profile_read",
        "configuration_profile_admin",
        "configuration_profile_read_v20",
        "configuration_profile_write_v20",
        "configuration_bundle_read",
        "configuration_bundle_admin",
        "configuration_bundle_read_v20",
        "configuration_bundle_write_v20",
        "configuration_events_read",
        "configuration_events_admin",
    ):
        assert f"DROP POLICY IF EXISTS {legacy_policy.upper()}" in normalized

    assert normalized.count("CREATE POLICY") == 6
    assert normalized.count("CURRENT_SETTING('APP.ROLE',TRUE) IN ('ADMIN','SERVICE')") >= 6
    assert normalized.count("TENANT_ID IS NOT DISTINCT FROM") == 2
    assert normalized.count("NULLIF(CURRENT_SETTING('APP.TENANT_ID',TRUE),'') IS NOT NULL") == 2
    assert normalized.count("FORCE ROW LEVEL SECURITY") == 3
    assert normalized.count("ENABLE ROW LEVEL SECURITY") == 3
