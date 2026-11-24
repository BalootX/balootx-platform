"""Configuration integration tests for the disposable PostgreSQL CI service."""
import hashlib
import os
import shutil
from pathlib import Path

import psycopg
from psycopg.conninfo import make_conninfo

from baloot_platform.configuration import PostgresConfigurationStore
from baloot_platform.storage.request_context import (
    CURRENT_DATABASE_IDENTITY,
    DatabaseIdentityContext,
)
from scripts.migrate_postgres import apply_migrations
from scripts.verify_live_postgres_research import (
    _prove_configuration_schedule,
    _prove_configuration_v25_schema,
)


def test_configuration_schedule_migrates_populated_postgres_and_round_trips(
    root: Path, tmp_path: Path,
):
    dsn = os.environ["BALOOT_TEST_POSTGRES_URL"]
    migration = root / "migrations/030_configuration_local_schedule.sql"
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute("CREATE ROLE configuration_migrator LOGIN NOSUPERUSER NOBYPASSRLS")
        connection.execute("CREATE ROLE configuration_app LOGIN NOSUPERUSER NOBYPASSRLS")
        connection.execute("GRANT CREATE ON DATABASE baloot_test TO configuration_migrator")
        connection.execute("GRANT USAGE, CREATE ON SCHEMA public TO configuration_migrator")

    migrator_dsn = make_conninfo(dsn, user="configuration_migrator")
    for predecessor in sorted(migration.parent.glob("*.sql")):
        if predecessor.name < migration.name:
            shutil.copyfile(predecessor, tmp_path / predecessor.name)
    apply_migrations(migrator_dsn, tmp_path)
    with psycopg.connect(migrator_dsn) as connection:
        connection.execute("SELECT set_config('app.role', 'service', true)")
        connection.execute(
            """INSERT INTO configuration.profile_revisions
               (revision_id, family, profile_id, version, scope, status, payload,
                payload_sha256, sequence, created_by, published_by, published_at, effective_from)
               VALUES ('CPR-LEGACY', 'rules', 'legacy', '1.0.0', 'global', 'published',
                       '{}', repeat('a', 64), 1, 'TEST', 'TEST', '2026-06-01T00:00Z',
                       '2026-07-01T12:00+03:00')"""
        )
        connection.execute(
            """INSERT INTO configuration.bundle_revisions
               (revision_id, bundle_id, mode, version, scope, status, components,
                content_sha256, sequence, created_by, effective_until, scope_key, tenant_id)
               VALUES ('CBR-LEGACY', 'legacy', 'full', '1.0.0', 'tenant', 'draft',
                       '{"rules":"CPR-LEGACY"}', repeat('b', 64), 1, 'TEST',
                       '2026-08-01T12:00+03:00', 'TEST-TENANT', 'TEST-TENANT')"""
        )
        connection.execute("GRANT USAGE ON SCHEMA configuration TO configuration_app")
        connection.execute("GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA configuration TO configuration_app")

    apply_migrations(migrator_dsn, migration.parent)
    schema_proof = _prove_configuration_v25_schema(migrator_dsn)
    assert schema_proof["immutabilityTriggerCount"] == 4
    store = PostgresConfigurationStore(make_conninfo(dsn, user="configuration_app"))
    try:
        with store.pool.connection() as connection:
            assert connection.execute(
                "SELECT current_user, row_security_active('configuration.bundle_revisions')"
            ).fetchone() == {
                "current_user": "configuration_app", "row_security_active": True,
            }
        assert store.get_bundle("CBR-LEGACY") is None
        token = CURRENT_DATABASE_IDENTITY.set(DatabaseIdentityContext(subject_id="TEST", role="service"))
        try:
            old_profile = store.get_profile("CPR-LEGACY")
            old_bundle = store.get_bundle("CBR-LEGACY")
            assert old_profile.effective_from == "2026-07-01T09:00:00+00:00"
            assert old_bundle.effective_until == "2026-08-01T09:00:00+00:00"
            assert "localSchedule" not in old_profile.to_dict()
            assert "localSchedule" not in old_bundle.to_dict()
            _prove_configuration_schedule(store, {"rules": old_profile.revision_id})
            apply_migrations(migrator_dsn, migration.parent)
        finally:
            CURRENT_DATABASE_IDENTITY.reset(token)
    finally:
        store.close()

    print(f"migration 030 sha256 {hashlib.sha256(migration.read_bytes()).hexdigest()}")
    print("configurationLocalScheduleRoundTrip: true")
