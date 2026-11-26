"""Configuration integration tests for the disposable PostgreSQL CI service."""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import psycopg
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from psycopg.conninfo import make_conninfo

from baloot_platform.config import build_bundle
from baloot_platform.configuration import (
    PostgresConfigurationStore,
    SQLiteConfigurationStore,
    VersionedConfigurationService,
    store as configuration_store,
)
from baloot_platform.configuration.schedule import ConfigurationScheduleConflict
from baloot_platform.identity import IdentityService, SQLiteIdentityStore
from baloot_platform.security import AuthorizationPolicy
from baloot_platform.service.routes.configuration import (
    build_configuration_router,
    configuration_schedule_conflict_response,
)
from baloot_platform.storage.request_context import (
    CURRENT_DATABASE_IDENTITY,
    DatabaseIdentityContext,
)
from scripts.migrate_postgres import apply_migrations
from scripts.verify_live_postgres_research import (
    _assert_schema_contract,
    _migration_schema_contract,
    _prove_configuration_schedule,
    _prove_configuration_v25_schema,
)


def _run_previous_migrator(dsn: str, previous: Path, migration_root: Path) -> None:
    subprocess.run(
        [sys.executable, str(previous / "scripts/migrate_postgres.py")],
        cwd=previous,
        env={
            **os.environ,
            "BALOOT_ROOT": str(migration_root),
            "BALOOT_DATABASE_URL": dsn,
            "PYTHONPATH": str(previous / "src"),
        },
        check=True,
    )


def _prove_previous_release(migrator_dsn: str, app_dsn: str, postgres_ids: tuple[str, str], path: Path) -> None:
    previous = Path(os.environ["BALOOT_PREVIOUS_ROOT"])
    store = SQLiteConfigurationStore(path)
    local = {"timeZone": "America/New_York", "effectiveFrom": "2026-07-01T12:00"}
    profile = store.create_profile(
        family="rules", profile_id="compatibility", version="1.0.0",
        scope="global", tenant_id=None, status="published", payload={"compatibility": True},
        created_by="TEST", local_schedule=local,
    )
    bundle = store.create_bundle(
        bundle_id="compatibility", version="1.0.0", mode="full", scope="global",
        tenant_id=None, status="draft", component_revisions={"rules": profile.revision_id},
        created_by="TEST", local_schedule=local,
    )
    old_env = {**os.environ, "BALOOT_ROOT": str(previous), "PYTHONPATH": str(previous / "src")}
    _run_previous_migrator(migrator_dsn, previous, previous)
    subprocess.run(
        [sys.executable, "-c", """
import os
import sys
from pathlib import Path

import baloot_platform
from baloot_platform.configuration import PostgresConfigurationStore, SQLiteConfigurationStore
from baloot_platform.storage.request_context import CURRENT_DATABASE_IDENTITY, DatabaseIdentityContext

previous_root = Path(os.environ["BALOOT_PREVIOUS_ROOT"]).resolve()
module_path = Path(baloot_platform.__file__).resolve()
assert module_path.is_relative_to(previous_root / "src"), "Previous-release package came from another checkout"
app_dsn, sqlite_path, postgres_profile_id, postgres_bundle_id, sqlite_profile_id, sqlite_bundle_id = sys.argv[1:]
CURRENT_DATABASE_IDENTITY.set(DatabaseIdentityContext(subject_id="TEST", role="service"))
for store, profile_id, bundle_id in (
    (PostgresConfigurationStore(app_dsn), postgres_profile_id, postgres_bundle_id),
    (SQLiteConfigurationStore(sqlite_path), sqlite_profile_id, sqlite_bundle_id),
):
    try:
        profile = store.get_profile(profile_id)
        bundle = store.get_bundle(bundle_id)
        assert profile.effective_from == bundle.effective_from == "2026-07-01T16:00:00+00:00"
    finally:
        store.close()
print(f"previousReleaseConfigurationModule: {module_path.relative_to(previous_root).as_posix()}")
""", app_dsn, str(path), *postgres_ids, profile.revision_id, bundle.revision_id],
        cwd=previous, env=old_env, check=True,
    )
    reopened = SQLiteConfigurationStore(path)
    assert reopened.get_profile(profile.revision_id) == profile
    assert reopened.get_bundle(bundle.revision_id) == bundle
    print("previousReleaseConfigurationCompatible: true")


@pytest.fixture(scope="module")
def migrated_postgres(root: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str]:
    dsn = os.environ["BALOOT_TEST_POSTGRES_URL"]
    migration = root / "migrations/033_configuration_local_schedule.sql"
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute("CREATE ROLE configuration_migrator LOGIN NOSUPERUSER NOBYPASSRLS")
        connection.execute("CREATE ROLE configuration_app LOGIN NOSUPERUSER NOBYPASSRLS")
        connection.execute("GRANT CREATE ON DATABASE baloot_test TO configuration_migrator")
        connection.execute("GRANT USAGE, CREATE ON SCHEMA public TO configuration_migrator")

    migrator_dsn = make_conninfo(dsn, user="configuration_migrator")
    predecessor_root = tmp_path_factory.mktemp("pre-schedule-migrations")
    predecessor_directory = predecessor_root / "migrations"
    predecessor_directory.mkdir()
    for predecessor in sorted(migration.parent.glob("*.sql")):
        if predecessor.name < migration.name:
            shutil.copyfile(predecessor, predecessor_directory / predecessor.name)
    previous = Path(os.environ["BALOOT_PREVIOUS_ROOT"])
    _run_previous_migrator(migrator_dsn, previous, predecessor_root)
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
    return migrator_dsn, make_conninfo(dsn, user="configuration_app")


def test_configuration_schedule_migrates_populated_postgres_and_round_trips(
    root: Path, tmp_path: Path, migrated_postgres: tuple[str, str],
) -> None:
    migrator_dsn, app_dsn = migrated_postgres
    migration = root / "migrations/033_configuration_local_schedule.sql"
    schema_contract = _migration_schema_contract(root)
    assert _assert_schema_contract(migrator_dsn, schema_contract)["tableCount"] == len(schema_contract["tables"])
    schema_proof = _prove_configuration_v25_schema(migrator_dsn)
    assert schema_proof["immutabilityTriggerCount"] == 4
    store = PostgresConfigurationStore(app_dsn)
    try:
        with store.pool.connection() as connection:
            assert connection.execute(
                "SELECT current_user, row_security_active('configuration.bundle_revisions')"
            ).fetchone() == {
                "current_user": "configuration_app", "row_security_active": True,
            }
            assert connection.execute(
                "SELECT indisunique FROM pg_index "
                "WHERE indexrelid = 'configuration.uq_configuration_profile_content'::regclass"
            ).fetchone() == {"indisunique": True}
        assert store.get_bundle("CBR-LEGACY") is None
        token = CURRENT_DATABASE_IDENTITY.set(DatabaseIdentityContext(subject_id="TEST", role="service"))
        try:
            old_profile = store.get_profile("CPR-LEGACY")
            old_bundle = store.get_bundle("CBR-LEGACY")
            assert old_profile.effective_from == "2026-07-01T09:00:00+00:00"
            assert old_bundle.effective_until == "2026-08-01T09:00:00+00:00"
            assert "localSchedule" not in old_profile.to_dict()
            assert "localSchedule" not in old_bundle.to_dict()
            postgres_ids = _prove_configuration_schedule(store, {"rules": old_profile.revision_id})
            apply_migrations(migrator_dsn, migration.parent)
        finally:
            CURRENT_DATABASE_IDENTITY.reset(token)
    finally:
        store.close()

    _prove_previous_release(migrator_dsn, app_dsn, postgres_ids, tmp_path / "configuration.sqlite3")
    assert _prove_configuration_v25_schema(migrator_dsn) == schema_proof
    with psycopg.connect(migrator_dsn) as connection:
        receipts = connection.execute("SELECT name, sha256 FROM baloot_migrations.applied ORDER BY name").fetchall()
    assert receipts == [
        (path.name, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(migration.parent.glob("*.sql"))
    ]
    print(f"{migration.name} sha256 {hashlib.sha256(migration.read_bytes()).hexdigest()}")
    print("configurationLocalScheduleRoundTrip: true")


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("schedule_kind", ["utc", "local"])
def test_rollback_publishes_new_revision_without_target_schedule(
    root: Path,
    tmp_path: Path,
    migrated_postgres: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    schedule_kind: str,
) -> None:
    monkeypatch.setattr(configuration_store, "utc_now", lambda: "2026-07-01T16:30:00+00:00")
    store = (
        PostgresConfigurationStore(migrated_postgres[1])
        if backend == "postgres"
        else SQLiteConfigurationStore(tmp_path / "rollback.sqlite3")
    )
    identity = DatabaseIdentityContext(subject_id="TEST", role="service")
    token = CURRENT_DATABASE_IDENTITY.set(identity)
    try:
        profile = build_bundle(root).component("configuration")
        service = VersionedConfigurationService(store, root=root, profile=profile)
        components = service.bootstrap_packaged_bundle(actor_id="TEST")["bundle"]["componentRevisions"]
        schedule = (
            {"local_schedule": {"timeZone": "America/New_York", "effectiveFrom": "2026-07-01T12:00"}}
            if schedule_kind == "local"
            else {"effective_from": "2026-07-01T16:00Z"}
        )
        target = service.create_bundle(
            bundle_id=f"scheduled-rollback-{schedule_kind}", version="1.0.0",
            mode="full", component_revisions=components, created_by="TEST",
            effective_until="2026-07-01T17:00Z", publish=True, **schedule,
        )
        identities = IdentityService(SQLiteIdentityStore(tmp_path / "identity.sqlite3"))
        app = FastAPI()
        app.add_exception_handler(ConfigurationScheduleConflict, configuration_schedule_conflict_response)
        app.include_router(build_configuration_router(service, identities, AuthorizationPolicy(), profile, {}))

        @app.middleware("http")
        async def service_request(request: Request, call_next):
            request.state.api_key_valid = True
            request_token = CURRENT_DATABASE_IDENTITY.set(identity)
            try:
                return await call_next(request)
            finally:
                CURRENT_DATABASE_IDENTITY.reset(request_token)

        with TestClient(app) as client:
            response = client.post(
                f"/api/v1/configuration-registry/bundles/{target['revisionId']}:rollback",
                headers={"x-baloot-role": "admin", "x-baloot-subject-id": "TEST"},
                json={"newVersion": "1.0.1"},
            )
        assert response.status_code == 201, response.text
        restored = response.json()
        assert restored["revisionId"] != target["revisionId"]
        assert restored["status"] == "published"
        assert restored["supersedesRevisionId"] == target["revisionId"]
        assert restored["contentSha256"] == target["contentSha256"]
        assert restored["componentRevisions"] == target["componentRevisions"]
        assert restored["effectiveFrom"] is None
        assert restored["effectiveUntil"] is None
        assert "localSchedule" not in restored
        previous = store.get_bundle(target["revisionId"]).to_dict()
        for field in ("effectiveFrom", "effectiveUntil", "localSchedule"):
            assert previous.get(field) == target.get(field)
        assert store.events("bundle", restored["revisionId"])[-1]["eventType"] == "rollback_created"
    finally:
        CURRENT_DATABASE_IDENTITY.reset(token)
        store.close()
