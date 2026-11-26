"""Configuration integration tests for the disposable PostgreSQL CI service."""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Event

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
    schedule as configuration_schedule,
    service as configuration_service,
    store as configuration_store,
)
from baloot_platform.configuration.schedule import ConfigurationIntegrityError, ConfigurationScheduleConflict
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
    _synthetic_runtime_environment,
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
    assert schema_proof["immutabilityTriggerCount"] == 5
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


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("operation", ["create", "publish", "rollback"])
def test_bundle_publication_refuses_invalid_component_without_changes(
    root: Path, tmp_path: Path, migrated_postgres: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch, backend: str, operation: str,
) -> None:
    store = (
        PostgresConfigurationStore(migrated_postgres[1]) if backend == "postgres"
        else SQLiteConfigurationStore(tmp_path / "publication.sqlite3")
    )
    token = CURRENT_DATABASE_IDENTITY.set(DatabaseIdentityContext(subject_id="TEST", role="service"))
    try:
        service = VersionedConfigurationService(store, root=root, profile=build_bundle(root).component("configuration"))
        baseline = service.bootstrap_packaged_bundle(actor_id="TEST")
        components = baseline["bundle"]["componentRevisions"]
        payload = copy.deepcopy(baseline["profiles"]["performance"])
        payload["profile"].update({"id": f"schedule-review-{operation}", "version": "99.7.1"})
        suspect = service.create_profile(
            family="performance", payload=payload, created_by="TEST", publish=True,
            local_schedule={"timeZone": "America/New_York", "effectiveFrom": "2026-07-01T12:00"},
        )
        bundle_id = f"schedule-review-{operation}"
        defaults = {"bundle_id": bundle_id, "mode": "full", "created_by": "TEST"}
        invalid_components = {**components, "performance": suspect["revisionId"]}
        if operation != "create":
            target = service.create_bundle(
                **defaults, component_revisions=invalid_components, version="1.0.0",
                publish=operation == "rollback",
            )
        working = service.create_bundle(
            **defaults, component_revisions=components, version="1.0.1", publish=True,
        )

        def snapshot():
            rows = store.list_bundles(bundle_id=bundle_id)
            return rows, {row.revision_id: store.events("bundle", row.revision_id) for row in rows}

        before = snapshot()
        zone = configuration_schedule._zone
        monkeypatch.setattr(
            configuration_schedule, "_zone", lambda name: zone("UTC" if name == "America/New_York" else name),
        )
        with pytest.raises(ConfigurationIntegrityError, match="maps to a different UTC instant"):
            if operation == "create":
                service.create_bundle(
                    **defaults, component_revisions=invalid_components, version="1.0.2", publish=True,
                )
            elif operation == "publish":
                store.publish_bundle(target["revisionId"], expected_etag=target["etag"], actor_id="TEST")
            else:
                service.rollback_bundle(target["revisionId"], new_version="1.0.2", actor_id="TEST")
        assert snapshot() == before
        assert service.resolve(bundle_id=bundle_id)["bundle"]["revisionId"] == working["revisionId"]
    finally:
        CURRENT_DATABASE_IDENTITY.reset(token)
        store.close()


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_rollback_audit_commits_with_the_actual_replacement(
    root: Path, tmp_path: Path, migrated_postgres: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    store = (
        PostgresConfigurationStore(migrated_postgres[1]) if backend == "postgres"
        else SQLiteConfigurationStore(tmp_path / "rollback-audit.sqlite3")
    )
    identity = DatabaseIdentityContext(subject_id="TEST", role="service")
    token = CURRENT_DATABASE_IDENTITY.set(identity)
    try:
        service = VersionedConfigurationService(store, root=root, profile=build_bundle(root).component("configuration"))
        original = service.bootstrap_packaged_bundle(actor_id="TEST")["bundle"]
        bundle_id = "atomic-rollback-audit"
        defaults = {"bundle_id": bundle_id, "mode": "full", "created_by": "TEST", "publish": True}
        components = original["componentRevisions"]
        target = service.create_bundle(**defaults, component_revisions=components, version="1.0.0")
        rollback_ready = Event()
        published = Event()
        create = store.create_bundle

        def ordered_create(**kwargs):
            if kwargs.get("rollback_target_revision_id") or kwargs.get("force_new_revision") and kwargs["version"] == "1.0.2":
                rollback_ready.set()
                assert published.wait(15), "competing publication did not commit"
            return create(**kwargs)

        def rollback():
            rollback_identity = CURRENT_DATABASE_IDENTITY.set(identity)
            try:
                return service.rollback_bundle(target["revisionId"], new_version="1.0.2", actor_id="TEST")
            finally:
                CURRENT_DATABASE_IDENTITY.reset(rollback_identity)

        monkeypatch.setattr(store, "create_bundle", ordered_create)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(rollback)
            try:
                assert rollback_ready.wait(15), "rollback did not reach its write boundary"
                intervening = service.create_bundle(
                    **defaults, component_revisions=components, version="1.0.1", force_new_revision=True,
                )
            finally:
                published.set()
            restored = pending.result()

        assert restored["supersedesRevisionId"] == intervening["revisionId"]
        receipt = store.events("bundle", restored["revisionId"])[-1]
        assert receipt["eventType"] == "rollback_created"
        assert receipt["payload"]["targetRevisionId"] == target["revisionId"]
        assert receipt["payload"]["replacedRevisionId"] == intervening["revisionId"]

        before = store.list_bundles(bundle_id=bundle_id)
        audit_before = [store.events("bundle", row.revision_id) for row in before]
        append = store._append_event_tx

        def fail_rollback_audit(conn, aggregate_type, aggregate_id, event_type, actor_id, payload):
            if event_type == "rollback_created":
                raise RuntimeError("rollback audit unavailable")
            return append(conn, aggregate_type, aggregate_id, event_type, actor_id, payload)

        monkeypatch.setattr(store, "_append_event_tx", fail_rollback_audit)
        with pytest.raises(RuntimeError, match="rollback audit unavailable"):
            service.rollback_bundle(target["revisionId"], new_version="1.0.3", actor_id="TEST")
        assert store.list_bundles(bundle_id=bundle_id) == before
        assert [store.events("bundle", row.revision_id) for row in before] == audit_before
        assert service.resolve(bundle_id=bundle_id)["bundle"]["revisionId"] == restored["revisionId"]
    finally:
        CURRENT_DATABASE_IDENTITY.reset(token)
        store.close()


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_runtime_boot_and_recovery_across_scheduled_handoff(
    root: Path, tmp_path: Path, migrated_postgres: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    store = (
        PostgresConfigurationStore(migrated_postgres[1]) if backend == "postgres"
        else SQLiteConfigurationStore(tmp_path / "configuration.sqlite3")
    )
    token = CURRENT_DATABASE_IDENTITY.set(DatabaseIdentityContext(subject_id="TEST", role="service"))
    instant = "2026-07-01T15:59:59+00:00"
    startup_environment = {
        **os.environ,
        **_synthetic_runtime_environment(root, tmp_path, tmp_path / "outputs"),
        "BALOOT_DATABASE_URL": os.environ["BALOOT_TEST_POSTGRES_URL"] if backend == "postgres" else "",
        "BALOOT_API_KEY": "test",
        "BALOOT_SHUTDOWN_CONTAINER_ON_APP_EXIT": "true",
    }
    previous = Path(os.environ["BALOOT_PREVIOUS_ROOT"])

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromisoformat(instant).astimezone(tz)

    def probe(expected: str, *, source: Path = root, recovery_target: str = "", diagnostic: str = "") -> None:
        subprocess.run(
            [sys.executable, "-c", """
import sys
from datetime import datetime
from pathlib import Path

import baloot_platform
from baloot_platform.configuration import service

instant, expected, recovery_target, diagnostic, source_path = sys.argv[1:]
assert Path(baloot_platform.__file__).resolve().is_relative_to(Path(source_path).resolve() / 'src')

class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.fromisoformat(instant).astimezone(tz)

service.datetime = Clock
from fastapi.testclient import TestClient
from baloot_platform.service.api import CONFIGURATION, FOUNDRY_PROFILE, RESEARCH, app
from baloot_platform.storage.request_context import CURRENT_DATABASE_IDENTITY, DatabaseIdentityContext

admin = {'x-api-key': 'test', 'x-baloot-role': 'admin'}
with TestClient(app) as client:
    assert client.get('/healthz').status_code == 200
    ready = client.get('/readyz')
    if not diagnostic:
        assert ready.status_code == 200, ready.text
        assert FOUNDRY_PROFILE['runtimeBundleRevisionId'] == expected
        assert client.get('/api/v1/options').status_code == 200
        _, research = RESEARCH.execution_context_factory.resolve_bundle_revision(expected)
        assert research.bundle_revision_id == expected
    else:
        assert ready.status_code == 503, ready.text
        assert expected not in ready.text
        assert ready.json()['requestId'] == ready.headers['x-request-id']
        diagnosis = client.get('/readyz', headers=admin).json()['detail']
        assert expected in diagnosis and diagnostic in diagnosis
    if recovery_target:
        rollback = f'/api/v1/configuration-registry/bundles/{recovery_target}:rollback'
        body = {'newVersion': '99.7.9'}

        def snapshot():
            token = CURRENT_DATABASE_IDENTITY.set(DatabaseIdentityContext(subject_id='TEST', role='service'))
            try:
                bundle = CONFIGURATION.store.get_bundle(recovery_target)
                rows = CONFIGURATION.store.list_bundles(bundle_id=bundle.bundle_id)
                events = [CONFIGURATION.store.events('bundle', row.revision_id) for row in rows]
                return rows, events
            finally:
                CURRENT_DATABASE_IDENTITY.reset(token)

        before = snapshot()
        for headers, expected_status in (({}, 401), ({**admin, 'x-baloot-role': 'player'}, 503)):
            denied = client.post(rollback, headers=headers, json=body)
            assert denied.status_code == expected_status, denied.text
            assert expected not in denied.text
            assert denied.json()['requestId'] == denied.headers['x-request-id']
        assert snapshot() == before
        restored = client.post(rollback, headers=admin, json=body)
        assert restored.status_code == 201, restored.text
        assert CONFIGURATION.resolve(bundle_id=restored.json()['bundleId'])['bundle']['revisionId'] == restored.json()['revisionId']
""", instant, expected, recovery_target, diagnostic, str(source)],
            cwd=source,
            env={**startup_environment, "BALOOT_ROOT": str(source), "PYTHONPATH": str(source / "src")},
            check=True,
        )

    monkeypatch.setattr(configuration_service, "datetime", Clock)
    try:
        service = VersionedConfigurationService(store, root=root, profile=build_bundle(root).component("configuration"))
        baseline = service.bootstrap_packaged_bundle(actor_id="TEST")
        original = baseline["bundle"]
        defaults = {"bundle_id": original["bundleId"], "mode": original["mode"], "created_by": "TEST", "publish": True}

        def new_components(version: str):
            payload = copy.deepcopy(baseline["profiles"]["performance"])
            payload["profile"].update({"id": f"schedule-handoff-performance-{version}", "version": version})
            revision = service.create_profile(family="performance", payload=payload, created_by="TEST", publish=True)
            return {**original["componentRevisions"], "performance": revision["revisionId"]}

        scheduled = service.create_bundle(
            **defaults, version="99.7.6", component_revisions=new_components("99.7.6"),
            local_schedule={"timeZone": "America/New_York", "effectiveFrom": "2026-07-01T12:00", "effectiveUntil": "2026-07-01T13:00"},
        )
        published_filter = {"bundle_id": original["bundleId"], "status": "published"}
        assert [row.revision_id for row in store.list_bundles(**published_filter)] == [original["revisionId"]]
        assert [row.revision_id for row in store.list_bundles(**published_filter, include_scheduled=True)] == [scheduled["revisionId"]]
        for instant, expected in (
            ("2026-07-01T15:59:59+00:00", original["revisionId"]),
            ("2026-07-01T16:00:00+00:00", scheduled["revisionId"]),
            ("2026-07-01T17:00:00+00:00", original["revisionId"]),
        ):
            assert service.resolve(bundle_id=original["bundleId"])["bundle"]["revisionId"] == expected
            probe(expected)
            probe(original["revisionId"], source=previous)
        print(f"previousReleaseScheduledHandoffStartup: {backend} before, during and after")

        stale_etag = scheduled["etag"]
        history = store.events("bundle", scheduled["revisionId"])
        transition = store.set_bundle_status
        competing = {}

        def publish_before_retiring(*args, **kwargs):
            competing.update(service.create_bundle(
                **defaults, version="99.7.10", component_revisions=new_components("99.7.10"),
                effective_from="2026-07-02T00:00:00Z",
            ))
            stale = store.get_bundle(scheduled["revisionId"])
            assert stale.etag == stale_etag and stale.status == "superseded"
            return transition(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(store, "set_bundle_status", publish_before_retiring)
            with pytest.raises(ValueError, match="only published bundle revisions can be retired"):
                service.retire_bundle(scheduled["revisionId"], expected_etag=stale_etag, actor_id="TEST", reason="Stale retirement")
        assert store.events("bundle", scheduled["revisionId"]) == history
        assert service.resolve(bundle_id=original["bundleId"], at_time="2026-07-01T16:00Z")["bundle"]["revisionId"] == scheduled["revisionId"]
        service.retire_bundle(competing["revisionId"], expected_etag=competing["etag"], actor_id="TEST", reason="Remove later window")

        scheduled_profile_id = scheduled["componentRevisions"]["performance"]
        scheduled_profile = store.get_profile(scheduled_profile_id)
        history = store.events("profile", scheduled_profile_id)
        with pytest.raises(ValueError, match="used by a published bundle"):
            service.retire_profile(
                scheduled_profile_id, expected_etag=scheduled_profile.etag, actor_id="TEST", reason="Test active reference",
            )
        assert store.get_profile(scheduled_profile_id) == scheduled_profile
        assert store.events("profile", scheduled_profile_id) == history
        # Legacy writers could retire a component that an approved window still uses.
        if backend == "postgres":
            with psycopg.connect(migrated_postgres[1]) as connection:
                connection.execute("SELECT set_config('app.role', 'service', true)")
                connection.execute("UPDATE configuration.profile_revisions SET status='retired' WHERE revision_id=%s", (scheduled_profile_id,))
        else:
            with store._connect() as connection:
                connection.execute("UPDATE configuration_profile_revisions SET status='retired' WHERE revision_id=?", (scheduled_profile_id,))
        instant = "2026-07-01T16:00:00+00:00"
        probe(scheduled_profile_id, recovery_target=original["revisionId"], diagnostic="unavailable component")

        interim = service.create_bundle(**defaults, version="99.7.7", component_revisions=new_components("99.7.7"))
        service.retire_bundle(interim["revisionId"], expected_etag=interim["etag"], actor_id="TEST", reason="Test expiry without predecessor")
        request = {
            **defaults, "version": "99.7.8", "component_revisions": new_components("99.7.8"),
            "effective_until": "2026-07-01T17:00:00Z",
        }
        before = store.list_bundles(bundle_id=original["bundleId"])
        audit_before = [store.events("bundle", row.revision_id) for row in before]
        with pytest.raises(ConfigurationScheduleConflict, match="requires an unbounded published bundle at full rollout"):
            service.create_bundle(**request)
        assert store.list_bundles(bundle_id=original["bundleId"]) == before
        assert [store.events("bundle", row.revision_id) for row in before] == audit_before

        draft = service.create_bundle(**{**request, "publish": False})
        draft_before = store.get_bundle(draft["revisionId"])
        draft_audit = store.events("bundle", draft["revisionId"])
        with pytest.raises(ConfigurationScheduleConflict, match="requires an unbounded published bundle at full rollout"):
            store.publish_bundle(draft["revisionId"], expected_etag=draft["etag"], actor_id="TEST")
        assert store.get_bundle(draft["revisionId"]) == draft_before
        assert store.events("bundle", draft["revisionId"]) == draft_audit

        # The previous release could write a window without an underlying publication.
        expired = subprocess.run(
            [sys.executable, "-c", """
import json
import os
import sys
from pathlib import Path

import baloot_platform
from baloot_platform.config import build_bundle
from baloot_platform.configuration import PostgresConfigurationStore, SQLiteConfigurationStore, VersionedConfigurationService
from baloot_platform.storage.request_context import CURRENT_DATABASE_IDENTITY, DatabaseIdentityContext

root = Path(os.environ['BALOOT_ROOT'])
assert Path(baloot_platform.__file__).resolve().is_relative_to(root.resolve() / 'src')
dsn = os.environ['BALOOT_DATABASE_URL']
store = PostgresConfigurationStore(dsn) if dsn else SQLiteConfigurationStore(Path(os.environ['BALOOT_RUNTIME']) / 'configuration.sqlite3')
CURRENT_DATABASE_IDENTITY.set(DatabaseIdentityContext(subject_id='TEST', role='service'))
try:
    service = VersionedConfigurationService(store, root=root, profile=build_bundle(root).component('configuration'))
    print(service.create_bundle(**json.loads(sys.argv[1]), force_new_revision=True)['revisionId'])
finally:
    store.close()
""", json.dumps(request)],
            cwd=previous,
            env={**startup_environment, "BALOOT_ROOT": str(previous), "PYTHONPATH": str(previous / "src")},
            check=True, capture_output=True, text=True,
        )
        expiring_id = expired.stdout.strip()
        instant = "2026-07-01T17:00:00+00:00"
        probe(expiring_id, recovery_target=original["revisionId"], diagnostic="No configuration bundle is effective")
        recovered = service.resolve(bundle_id=original["bundleId"])["bundle"]["revisionId"]
        assert recovered not in {expiring_id, original["revisionId"]}
        probe(recovered)
        probe(recovered, source=previous)

        partial = service.create_bundle(
            **defaults, version="99.7.11", component_revisions=new_components("99.7.11"), rollout_percentage=10,
        )
        exact = service.resolve(revision_id=partial["revisionId"])
        assert exact["bundle"]["scope"] == original["scope"]
        assert exact["resolutionTrace"] == [{"scope": "exact_revision", "revisionId": partial["revisionId"]}]
        with pytest.raises(KeyError, match="published configuration bundle not found"):
            service.resolve(bundle_id=original["bundleId"])
        with pytest.raises(ConfigurationIntegrityError, match="No configuration bundle is effective"):
            service.bootstrap_packaged_bundle(actor_id="TEST")
        probe(partial["revisionId"], diagnostic="No configuration bundle is effective")
        bounded_components = new_components("99.7.12")
        before = store.list_bundles(bundle_id=original["bundleId"])
        audit_before = [store.events("bundle", row.revision_id) for row in before]
        with pytest.raises(ConfigurationScheduleConflict, match="requires an unbounded published bundle at full rollout"):
            service.create_bundle(
                **defaults, version="99.7.12", component_revisions=bounded_components,
                effective_from="2026-07-02T00:00:00Z",
            )
        assert store.list_bundles(bundle_id=original["bundleId"]) == before
        assert [store.events("bundle", row.revision_id) for row in before] == audit_before
    finally:
        CURRENT_DATABASE_IDENTITY.reset(token)
        store.close()
