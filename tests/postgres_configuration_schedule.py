"""Configuration integration tests for the disposable PostgreSQL CI service."""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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
from baloot_platform.service.container import _PACKAGED_RESEARCH_BUNDLES
from baloot_platform.service.routes.configuration import (
    build_configuration_router,
    configuration_schedule_conflict_response,
)
from baloot_platform.storage.request_context import (
    CURRENT_DATABASE_IDENTITY,
    DatabaseIdentityContext,
    contextual_connection,
)
from scripts.migrate_postgres import apply_migrations, migration_batches
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
    previous = Path(os.environ["BALOOT_PREVIOUS_ROOT"])
    predecessors = {path.name: path for path in (previous / "migrations").glob("*.sql")}
    for batch in migration_batches(predecessors):
        if batch[-1] >= migration.name:
            break
        for name in batch:
            shutil.copyfile(predecessors[name], predecessor_directory / name)
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
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest, backend: str,
) -> None:
    app_dsn = request.getfixturevalue("migrated_postgres")[1] if backend == "postgres" else None
    store = (
        PostgresConfigurationStore(app_dsn) if app_dsn is not None
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

    def probe(
        expected: str, *, source: Path = root, recovery_target: str = "", diagnostic: str = "",
        research: tuple[str, str, str] = ("", "", ""),
        recovery_research_scopes: tuple[tuple[str, str], ...] = (),
        access_rows: dict[str, tuple[str, str]] | None = None,
        recovery_audit: tuple[str, str] | None = None,
    ) -> None:
        subprocess.run(
            [sys.executable, "-c", """
import json
import sys
from datetime import datetime
from os import environ
from pathlib import Path
from unittest.mock import patch

import baloot_platform
from baloot_platform.configuration import service
from baloot_platform.service import container

(instant, expected, recovery_target, diagnostic, source_path, bundle_id,
 environment, research_id, scope_json, access_json, audit_json) = sys.argv[1:]
recovery_research_scopes = json.loads(scope_json)
access_rows = json.loads(access_json)
recovery_audit = json.loads(audit_json)
assert Path(baloot_platform.__file__).resolve().is_relative_to(Path(source_path).resolve() / 'src')

class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.fromisoformat(instant).astimezone(tz)

service.datetime = Clock
from fastapi.testclient import TestClient
configuration_dsn = environ['BALOOT_TEST_CONFIGURATION_DSN']
if configuration_dsn:
    # Other platform stores initialize their own schemas. Configuration uses the migrated app role.
    configuration_store = container.PostgresConfigurationStore
    with patch.object(container, 'PostgresConfigurationStore', lambda _: configuration_store(configuration_dsn)):
        from baloot_platform.service import api
else:
    from baloot_platform.service import api
ACCESS_TOKENS, CONFIGURATION, FOUNDRY_PROFILE, RESEARCH, app = (
    api.ACCESS_TOKENS, api.CONFIGURATION, api.FOUNDRY_PROFILE, api.RESEARCH, api.app,
)
from baloot_platform.storage.request_context import CURRENT_DATABASE_IDENTITY, DatabaseIdentityContext

admin = {'x-api-key': 'test', 'x-baloot-role': 'admin'}

def check_configuration_access(client):
    from pytest import MonkeyPatch, raises
    from baloot_platform.generated import configuration_pb2 as proto
    from baloot_platform.generated.configuration_pb2_grpc import ConfigurationControlStub

    sys.path.insert(0, str(Path(source_path) / 'tests'))
    from service_test_client import grpc_probe

    if configuration_dsn:
        with CONFIGURATION.store.pool.connection() as connection:
            authority = connection.execute(
                "SELECT session_user AS login_role, current_user AS query_role, rolsuper, rolbypassrls, "
                "row_security_active('configuration.profile_revisions') AS profiles, "
                "row_security_active('configuration.bundle_revisions') AS bundles, "
                "row_security_active('configuration.events') AS events "
                "FROM pg_roles WHERE rolname=current_user"
            ).fetchone()
        assert authority == {
            'login_role': 'configuration_app', 'query_role': 'configuration_app', 'rolsuper': False,
            'rolbypassrls': False, 'profiles': True, 'bundles': True, 'events': True,
        }
        print('configurationApplicationRoleRls: active')

    registry = '/api/v1/configuration-registry'
    bundle_name = 'scoped-schedule'
    profile_name = 'scoped_schedule'
    tenant_a, tenant_b = 'fixture-tenant-a', 'fixture-tenant-b'
    tenant_rows = {tenant: access_rows[tenant] for tenant in (tenant_a, tenant_b)}
    shared_profile, shared_bundle = access_rows['global']
    default_profile, default_bundle = access_rows['default']
    isolated_profile, isolated_bundle = access_rows['environment']
    _, linked_bundle = access_rows['edge']
    handoff_anchor, handoff_successor = access_rows['handoff']
    chain_anchor, chain_predecessor = access_rows['chain-base']
    chain_visible, chain_head = access_rows['chain-window']
    rollback_anchor, rollback_head = access_rows['rollback-links']
    linked_profile, private_profile = access_rows['profile-links']
    visible_draft, hidden_draft = access_rows['drafts']

    def signed(tenant, role, scopes=('baloot.read',)):
        return {'authorization': 'Bearer ' + ACCESS_TOKENS.issue(
            subject_id='TEST-CONFIG-' + tenant, role=role, tenant_id=tenant, scopes=scopes,
        )}

    headers = {
        tenant_a: signed(tenant_a, 'player'),
        tenant_b: signed(tenant_b, 'admin', ('baloot.read', 'baloot.config.admin')),
    }
    elevated = signed(tenant_a, 'admin')
    no_read = {'authorization': 'Bearer ' + ACCESS_TOKENS.issue(
        subject_id='TEST-CONFIG-PLAY', tenant_id=tenant_a, scopes=['baloot.play'],
    )}
    profiles = {'family': 'performance', 'profileId': profile_name}
    bundles = {'bundleId': bundle_name}
    service_reader = {**admin, 'x-baloot-role': 'player'}
    default_view = client.get(f'{registry}/bundles', params=bundles, headers=service_reader)
    assert default_view.status_code == 200, default_view.text
    assert {item['revisionId'] for item in default_view.json()['items']} == {default_bundle, shared_bundle}
    default_exact = client.get(f'{registry}/bundles/{default_bundle}:resolve', headers=service_reader)
    assert default_exact.status_code == 200 and default_exact.json()['audit']['valid'], default_exact.text
    for tenant, (profile_id, bundle_id) in tenant_rows.items():
        cred = headers[tenant]
        for resource, query, own, shared in (
            ('profiles', profiles, profile_id, shared_profile), ('bundles', bundles, bundle_id, shared_bundle),
        ):
            listing = client.get(f'{registry}/{resource}', params=query, headers=cred)
            assert listing.status_code == 200, listing.text
            assert {item['revisionId'] for item in listing.json()['items']} == {own, shared}
            for revision in (own, shared):
                detail = client.get(f'{registry}/{resource}/{revision}', headers=cred)
                assert detail.status_code == 200 and detail.json()['revisionId'] == revision, detail.text
                events = client.get(f'{registry}/{resource}/{revision}/events', headers=cred)
                assert events.status_code == 200 and events.json()['verification']['valid'], events.text
        profile = client.get(f'{registry}/profiles:resolve', params=profiles, headers=cred)
        assert profile.status_code == 200 and profile.json()['revisionId'] == profile_id, profile.text
        resolved = client.get(f'{registry}/resolve', params=bundles, headers=cred)
        assert resolved.status_code == 200 and resolved.json()['bundle']['revisionId'] == bundle_id, resolved.text
        exact = client.get(f'{registry}/bundles/{bundle_id}:resolve-v2', headers=cred)
        assert exact.status_code == 200 and exact.json()['bundle']['revisionId'] == bundle_id, exact.text

    other_profile, other_bundle = access_rows[tenant_b]
    allowed = headers[tenant_a]
    for route in ('profiles', 'profiles:resolve', 'bundles', 'resolve', 'resolve-v2'):
        params = profiles if route.startswith('profiles') else bundles
        response = client.get(f'{registry}/{route}', params={**params, 'tenantId': tenant_b}, headers=allowed)
        assert response.status_code == 403, response.text
    for resource in (
        f'profiles/{other_profile}', f'profiles/{other_profile}/events',
        f'bundles/{other_bundle}', f'bundles/{other_bundle}/events',
        f'bundles/{other_bundle}:resolve', f'bundles/{other_bundle}:resolve-v2',
    ):
        response = client.get(f'{registry}/{resource}', headers=allowed)
        assert response.status_code == 404 and tenant_b not in response.text, response.text
    isolated = {'environment': 'fixture-isolated'}
    for cred in (allowed, elevated):
        for route, query in (('profiles:resolve', profiles), ('resolve', bundles), ('resolve-v2', bundles)):
            missing = client.get(f'{registry}/{route}', params={**query, **isolated}, headers=cred)
            assert missing.status_code == 404, missing.text
        for resource, revision in (('profiles', isolated_profile), ('bundles', isolated_bundle), ('bundles', linked_bundle)):
            for suffix in ('', '/events'):
                missing = client.get(f'{registry}/{resource}/{revision}{suffix}', headers=cred)
                assert missing.status_code == 404, missing.text
        for revision in (isolated_bundle, linked_bundle):
            for suffix in (':resolve', ':resolve-v2'):
                missing = client.get(f'{registry}/bundles/{revision}{suffix}', headers=cred)
                assert missing.status_code == 404, missing.text
        listing = client.get(f'{registry}/bundles', params={'bundleId': 'scoped-edge'}, headers=cred)
        assert listing.status_code == 200 and listing.json()['items'] == [], listing.text

    own_exact = f'{registry}/bundles/{tenant_rows[tenant_a][1]}:resolve'
    CONFIGURATION._invalidate_cache()
    with patch.object(CONFIGURATION.store, 'get_bundle', wraps=CONFIGURATION.store.get_bundle) as lookup:
        for cred in (elevated, elevated, allowed, allowed):
            response = client.get(own_exact, headers=cred)
            assert response.status_code == 200 and response.json()['audit']['valid'], response.text
        assert lookup.call_count == 2
    for revision in (isolated_bundle, linked_bundle):
        route = f'{registry}/bundles/{revision}:resolve'
        assert client.get(route, headers=admin).status_code == 200
        assert client.get(route, headers=elevated).status_code == 404
        assert client.get(route, headers=allowed).status_code == 404

    def assert_handoff_fallback(result):
        assert result['bundle']['revisionId'] == handoff_anchor
        assert [row['revisionId'] for row in result['resolutionTrace']] == [handoff_anchor]
        assert handoff_successor not in json.dumps(result)

    handoff = {'bundleId': 'scoped-handoff'}
    admin_handoff = client.get(f'{registry}/resolve', params={**handoff, 'atTime': '2026-07-01T18:00Z'}, headers=admin)
    assert admin_handoff.status_code == 200 and admin_handoff.json()['bundle']['revisionId'] == handoff_successor
    for cred in (allowed, elevated):
        for route in ('resolve', 'resolve-v2'):
            for at_time in ('2026-07-01T17:00Z', '2026-07-01T18:00Z'):
                result = client.get(f'{registry}/{route}', params={**handoff, 'atTime': at_time}, headers=cred)
                assert result.status_code == 200, result.text
                assert_handoff_fallback(result.json())
        for suffix in (':resolve', ':resolve-v2'):
            hidden = client.get(f'{registry}/bundles/{hidden_draft}{suffix}', headers=cred)
            visible = client.get(f'{registry}/bundles/{visible_draft}{suffix}', headers=cred)
            assert hidden.status_code == 404 and hidden_draft not in hidden.text, hidden.text
            assert visible.status_code == 403, visible.text
    preview = client.get(f'{registry}/bundles/{hidden_draft}:resolve', params={'allowDraft': 'true'}, headers=admin)
    assert preview.status_code == 200 and preview.json()['bundle']['status'] == 'draft', preview.text

    def check_event_links(resource, revision, event_type, field, hidden_id):
        path = f'{registry}/{resource}/{revision}/events'
        raw = client.get(path, headers=admin)
        assert raw.status_code == 200, raw.text
        raw_history = raw.json()
        assert raw_history['verification']['valid']
        for cred in (allowed, elevated):
            scoped = client.get(path, headers=cred)
            assert scoped.status_code == 200, scoped.text
            history = scoped.json()
            assert history['verification'] == raw_history['verification']
            raw_event, = (event for event in raw_history['items'] if event['eventType'] == event_type)
            event, = (event for event in history['items'] if event['eventType'] == event_type)
            assert raw_event['payload'][field] == hidden_id and event['payload'][field] is None
        assert client.get(path, headers=admin).json() == raw_history

    status_cases = (
        ('scoped-handoff', 'published', handoff_anchor),
        ('scoped-handoff', 'superseded', None),
        ('scoped-chain', 'published', chain_visible),
        ('scoped-chain', 'superseded', chain_anchor),
    )
    for name, status, scoped_id in status_cases:
        query = {'bundleId': name, 'status': status, 'limit': 1}
        result = client.get(f'{registry}/bundles', params=query, headers=allowed)
        assert result.status_code == 200, result.text
        assert [item['revisionId'] for item in result.json()['items']] == ([scoped_id] if scoped_id else [])
    assert client.get(
        f'{registry}/bundles', params={'bundleId': 'scoped-chain', 'status': 'published', 'limit': 1}, headers=admin,
    ).json()['items'][0]['revisionId'] == chain_head
    for revision in (handoff_anchor, chain_visible):
        path = f'{registry}/bundles/{revision}'
        assert client.get(path, headers=allowed).json()['status'] == 'published'
        assert client.get(f'{path}:resolve', headers=allowed).json()['bundle']['status'] == 'published'
    assert client.get(f'{registry}/bundles/{handoff_anchor}', headers=admin).json()['status'] == 'superseded'
    check_event_links('bundles', chain_visible, 'published', 'scheduledPredecessorId', chain_predecessor)
    check_event_links('profiles', linked_profile, 'created', 'supersedesRevisionId', private_profile)

    rolled = client.post(f'{registry}/bundles/{rollback_anchor}:rollback', json={'newVersion': '1.0.2'}, headers=admin)
    assert rolled.status_code == 201 and rolled.json()['supersedesRevisionId'] == rollback_head, rolled.text
    rollback_id = rolled.json()['revisionId']
    check_event_links('bundles', rollback_id, 'rollback_created', 'replacedRevisionId', rollback_head)
    check_event_links('bundles', rollback_id, 'created', 'supersedesRevisionId', rollback_head)
    linked_query = {'family': 'performance', 'profileId': 'scoped_linked_profile'}
    for cred in (allowed, elevated):
        for route, params in (
            (f'profiles/{linked_profile}', {}), ('profiles', linked_query),
            ('profiles:resolve', linked_query), ('profiles:compare', {'left': linked_profile, 'right': shared_profile}),
        ):
            response = client.get(f'{registry}/{route}', params=params, headers=cred)
            assert response.status_code == 200 and private_profile not in response.text, response.text
        selected = client.get(
            f'{registry}/resolve', params={'bundleId': 'scoped-chain', 'atTime': '2026-07-01T19:00Z'}, headers=cred,
        )
        assert selected.status_code == 200 and selected.json()['bundle']['revisionId'] == chain_visible, selected.text
        assert selected.json()['profileRevisions']['performance']['supersedesRevisionId'] is None
        assert chain_head not in selected.text and chain_predecessor not in selected.text
        for route, params in (
            (f'bundles/{rollback_id}', {}), (f'bundles/{rollback_id}:resolve', {}),
            (f'bundles/{rollback_id}:resolve-v2', {}), ('resolve', {'bundleId': 'scoped-rollback-link'}),
            ('resolve-v2', {'bundleId': 'scoped-rollback-link'}), ('bundles', {'bundleId': 'scoped-rollback-link'}),
        ):
            response = client.get(f'{registry}/{route}', params=params, headers=cred)
            assert response.status_code == 200 and rollback_head not in response.text, response.text
    assert client.get(f'{registry}/bundles', headers=no_read).status_code == 403
    assert client.get(f'{registry}/bundles/{access_rows[tenant_a][1]}:resolve', headers=no_read).status_code == 403
    assert client.get('/api/v2/configuration/bundles', headers=headers[tenant_b]).status_code == 403
    operator_view = client.get('/api/v2/configuration/bundles', params=bundles, headers=admin)
    assert operator_view.status_code == 200, operator_view.text
    assert {item['revisionId'] for item in operator_view.json()['items']} == {
        shared_bundle, isolated_bundle, default_bundle, *(item[1] for item in tenant_rows.values()),
    }

    async def rpc(grpc, stub):
        result = await stub.ResolveProfile(
            proto.ProfileResolution(family='performance', profile_id=profile_name),
            metadata=tuple(service_reader.items()), timeout=5,
        )
        assert json.loads(result.canonical_json)['revisionId'] == default_profile
        for tenant, (profile_id, bundle_id) in tenant_rows.items():
            cred = tuple(headers[tenant].items())
            for method, request, revisions in (
                (stub.ListProfiles, proto.ProfileQuery(family='performance', profile_id=profile_name), {profile_id, shared_profile}),
                (stub.ListBundles, proto.BundleQuery(bundle_id=bundle_name), {bundle_id, shared_bundle}),
                (stub.ResolveProfile, proto.ProfileResolution(family='performance', profile_id=profile_name), {profile_id}),
                (stub.ResolveBundle, proto.BundleResolution(bundle_id=bundle_name), {bundle_id}),
                (stub.ResolveEffectiveRuntimeConfigV2, proto.BundleResolution(revision_id=bundle_id), {bundle_id}),
            ):
                result = json.loads((await method(request, metadata=cred, timeout=5)).canonical_json)
                records = result.get('items', [result.get('bundle', result)])
                assert {row['revisionId'] for row in records} == revisions
        CONFIGURATION._invalidate_cache()
        own = proto.BundleResolution(revision_id=tenant_rows[tenant_a][1])
        with patch.object(CONFIGURATION.store, 'get_bundle', wraps=CONFIGURATION.store.get_bundle) as lookup:
            for headers_for_role in (elevated, elevated, allowed, allowed):
                result = await stub.ResolveBundle(own, metadata=tuple(headers_for_role.items()), timeout=5)
                assert json.loads(result.canonical_json)['audit']['valid']
            assert lookup.call_count == 2
        denied = grpc.StatusCode.PERMISSION_DENIED
        missing = grpc.StatusCode.NOT_FOUND
        cred = tuple(allowed.items())
        for method, request, metadata, code in (
            (stub.ListBundles, proto.BundleQuery(tenant_id=tenant_b), cred, denied),
            (stub.ResolveProfile, proto.ProfileResolution(tenant_id=tenant_b), cred, denied),
            (stub.ResolveBundle, proto.BundleResolution(revision_id=other_bundle), cred, missing),
            (stub.ResolveEffectiveRuntimeConfigV2, proto.BundleResolution(revision_id=other_bundle), cred, missing),
            (stub.CompareProfiles, proto.ProfileCompare(left=other_profile, right=other_profile), cred, missing),
            (stub.ResolveBundle, proto.BundleResolution(revision_id=access_rows[tenant_a][1], allow_draft=True), cred, denied),
            (stub.ListBundles, proto.BundleQuery(), tuple(no_read.items()), denied),
            (stub.ResolveBundle, proto.BundleResolution(revision_id=access_rows[tenant_a][1]), tuple(no_read.items()), denied),
            (stub.RollbackBundle, proto.BundleRollback(revision_id=other_bundle), tuple(headers[tenant_b].items()), denied),
        ):
            with raises(grpc.aio.AioRpcError) as failure:
                await method(request, metadata=metadata, timeout=5)
            assert failure.value.code() == code
        for revision in (isolated_bundle, linked_bundle):
            request = proto.BundleResolution(revision_id=revision)
            await stub.ResolveEffectiveRuntimeConfigV2(request, metadata=tuple(admin.items()), timeout=5)
            for role_headers in (elevated, allowed):
                with raises(grpc.aio.AioRpcError) as failure:
                    await stub.ResolveEffectiveRuntimeConfigV2(request, metadata=tuple(role_headers.items()), timeout=5)
                assert failure.value.code() == missing
        for role_headers in (elevated, allowed):
            listing = await stub.ListBundles(proto.BundleQuery(bundle_id='scoped-edge'), metadata=tuple(role_headers.items()), timeout=5)
            assert json.loads(listing.canonical_json)['items'] == []
            with raises(grpc.aio.AioRpcError) as failure:
                await stub.CompareProfiles(proto.ProfileCompare(left=shared_profile, right=isolated_profile),
                                           metadata=tuple(role_headers.items()), timeout=5)
            assert failure.value.code() == missing
        handoff_request = proto.BundleResolution(bundle_id='scoped-handoff')
        admin_metadata = tuple(admin.items())
        admin_result = await stub.ResolveBundle(handoff_request, metadata=admin_metadata, timeout=5)
        assert handoff_successor in {row['revisionId'] for row in json.loads(admin_result.canonical_json)['resolutionTrace']}
        preview = await stub.ResolveBundle(
            proto.BundleResolution(revision_id=hidden_draft, allow_draft=True), metadata=admin_metadata, timeout=5,
        )
        assert json.loads(preview.canonical_json)['bundle']['status'] == 'draft'
        for role_headers in (elevated, allowed):
            metadata = tuple(role_headers.items())
            for method in (stub.ResolveBundle, stub.ResolveEffectiveRuntimeConfigV2):
                result = await method(handoff_request, metadata=metadata, timeout=5)
                assert_handoff_fallback(json.loads(result.canonical_json))
                for draft, expected in ((hidden_draft, missing), (visible_draft, denied)):
                    with raises(grpc.aio.AioRpcError) as failure:
                        await method(proto.BundleResolution(revision_id=draft), metadata=metadata, timeout=5)
                    assert failure.value.code() == expected
                    assert hidden_draft not in failure.value.details()
        for name, status, scoped_id in status_cases:
            listed = await stub.ListBundles(
                proto.BundleQuery(bundle_id=name, status=status, limit=1), metadata=cred, timeout=5,
            )
            assert [item['revisionId'] for item in json.loads(listed.canonical_json)['items']] == ([scoped_id] if scoped_id else [])
        for method, request, protected_id in (
            (stub.ListProfiles, proto.ProfileQuery(family='performance', profile_id='scoped_linked_profile'), private_profile),
            (stub.ResolveProfile, proto.ProfileResolution(family='performance', profile_id='scoped_linked_profile'), private_profile),
            (stub.CompareProfiles, proto.ProfileCompare(left=linked_profile, right=shared_profile), private_profile),
            (stub.ListBundles, proto.BundleQuery(bundle_id='scoped-rollback-link'), rollback_head),
            (stub.ResolveBundle, proto.BundleResolution(revision_id=rollback_id), rollback_head),
            (stub.ResolveBundle, proto.BundleResolution(bundle_id='scoped-rollback-link'), rollback_head),
            (stub.ResolveEffectiveRuntimeConfigV2, proto.BundleResolution(revision_id=rollback_id), rollback_head),
        ):
            result = await method(request, metadata=cred, timeout=5)
            assert protected_id.encode() not in result.canonical_json
        exact = await stub.ResolveBundle(proto.BundleResolution(revision_id=chain_visible), metadata=cred, timeout=5)
        assert json.loads(exact.canonical_json)['bundle']['status'] == 'published'
        raw = await stub.ResolveBundle(proto.BundleResolution(revision_id=rollback_id), metadata=admin_metadata, timeout=5)
        assert json.loads(raw.canonical_json)['bundle']['supersedesRevisionId'] == rollback_head
    with MonkeyPatch.context() as grpc_patch:
        grpc_probe(Path(environ['BALOOT_RUNTIME']), grpc_patch, rpc, stub_type=ConfigurationControlStub)

with TestClient(app) as client:
    assert client.get('/healthz').status_code == 200
    ready = client.get('/readyz')
    if not diagnostic:
        assert ready.status_code == 200, ready.text
        assert FOUNDRY_PROFILE['runtimeBundleRevisionId'] == expected
        assert client.get('/api/v1/options').status_code == 200
        _, research = RESEARCH.execution_context_factory.resolve_bundle_revision(expected)
        assert research.bundle_revision_id == expected
        if recovery_audit is not None:
            recovered_bundle, replaced_revision = recovery_audit
            registry = '/api/v1/configuration-registry'
            effective = client.get(f'{registry}/resolve', headers=admin, params={'bundleId': recovered_bundle})
            assert effective.status_code == 200, effective.text
            assert effective.json()['bundle']['revisionId'] == expected
            events = client.get(f'{registry}/bundles/{expected}/events', headers=admin)
            assert events.status_code == 200 and events.json()['verification']['valid'], events.text
            rollback_event = events.json()['items'][-1]
            assert rollback_event['eventType'] == 'rollback_created'
            assert rollback_event['payload']['replacedRevisionId'] == replaced_revision
            print('scheduleRecoveryAuditAfterRestart: verified')
        if access_rows:
            check_configuration_access(client)
            print('configurationReadScope: HTTP and gRPC')
        if research_id:
            response = client.get('/api/v1/configuration-registry/resolve',
                                  headers=admin, params={'bundleId': bundle_id, 'environment': environment})
            assert response.status_code == 200, response.text
            scoped = response.json()
            assert scoped['bundle']['environment'] == environment
            assert scoped['bundle']['revisionId'] == research_id
    else:
        assert ready.status_code == 503, ready.text
        assert expected not in ready.text
        assert ready.json()['requestId'] == ready.headers['x-request-id']
        diagnosis = client.get('/readyz', headers=admin).json()['detail']
        assert expected in diagnosis and diagnostic in diagnosis
    if recovery_target:
        rollback = f'/api/v1/configuration-registry/bundles/{recovery_target}:rollback'
        body = {'newVersion': '99.7.9'}

        def snapshot(bundle_id):
            token = CURRENT_DATABASE_IDENTITY.set(DatabaseIdentityContext(subject_id='TEST', role='service'))
            try:
                rows = CONFIGURATION.store.list_bundles(bundle_id=bundle_id)
                events = [CONFIGURATION.store.events('bundle', row.revision_id) for row in rows]
                return rows, events
            finally:
                CURRENT_DATABASE_IDENTITY.reset(token)

        token = CURRENT_DATABASE_IDENTITY.set(DatabaseIdentityContext(subject_id='TEST', role='service'))
        try:
            target = CONFIGURATION.store.get_bundle(recovery_target)
            legacy = CONFIGURATION.store.get_bundle(expected) if recovery_research_scopes else None
        finally:
            CURRENT_DATABASE_IDENTITY.reset(token)
        before = snapshot(target.bundle_id)
        for headers, expected_status in (({}, 401), ({**admin, 'x-baloot-role': 'player'}, 403)):
            denied = client.post(rollback, headers=headers, json=body)
            assert denied.status_code == expected_status, denied.text
            assert expected not in denied.text
            assert denied.json()['requestId'] == denied.headers['x-request-id']
        assert snapshot(target.bundle_id) == before
        if recovery_research_scopes:
            for bundle_id, environment in recovery_research_scopes:
                research_before = snapshot(bundle_id)
                assert len(research_before[0]) == 1 and research_before[0][0].status == 'draft'
                assert len(research_before[1][0]) == 1
                payload = {
                    'bundleId': bundle_id, 'version': '99.8.0', 'scope': 'environment',
                    'environment': environment, 'componentRevisions': legacy.component_revisions,
                    'publish': True,
                }
                for bounds in ({}, {'effectiveFrom': '2026-07-02T00:00:00Z'}):
                    refused = client.post('/api/v1/configuration-registry/bundles', headers=admin, json={**payload, **bounds})
                    assert refused.status_code == 409, refused.text
                    assert 'previous-release startup' in refused.json()['detail']
                    assert snapshot(bundle_id) == research_before
            print('researchRecoveryRefusedUnsafeWrites: true')
        restored = client.post(rollback, headers=admin, json=body)
        assert restored.status_code == 201, restored.text
        if diagnostic == 'scheduled publication could not be verified':
            restored_id = restored.json()['revisionId']
            assert restored.json()['supersedesRevisionId'] == expected
            events = client.get(f'/api/v1/configuration-registry/bundles/{restored_id}/events', headers=admin)
            assert events.status_code == 503, events.text
            assert client.get('/readyz').status_code == 503
        assert CONFIGURATION.resolve(bundle_id=restored.json()['bundleId'])['bundle']['revisionId'] == restored.json()['revisionId']
""", instant, expected, recovery_target, diagnostic, str(source), *research, json.dumps(recovery_research_scopes),
             json.dumps(access_rows), json.dumps(recovery_audit)],
            cwd=source,
            env={
                **startup_environment, "BALOOT_ROOT": str(source), "PYTHONPATH": str(source / "src"),
                "BALOOT_TEST_CONFIGURATION_DSN": app_dsn or "",
            },
            check=True,
        )

    monkeypatch.setattr(configuration_service, "datetime", Clock)
    try:
        packaged = build_bundle(root)
        service = VersionedConfigurationService(store, root=root, profile=packaged.component("configuration"))
        baseline = service.bootstrap_packaged_bundle(actor_id="TEST", packaged_bundle=packaged)
        original = baseline["bundle"]
        defaults = {"bundle_id": original["bundleId"], "mode": original["mode"], "created_by": "TEST", "publish": True}

        def new_components(version: str):
            payload = copy.deepcopy(baseline["profiles"]["performance"])
            payload["profile"].update({"id": f"schedule-handoff-performance-{version}", "version": version})
            revision = service.create_profile(family="performance", payload=payload, created_by="TEST", publish=True)
            return {**original["componentRevisions"], "performance": revision["revisionId"]}

        legacy_request = {
            **defaults, "version": "99.7.13", "component_revisions": new_components("99.7.13"),
            "effective_until": "2026-07-01T17:00:00Z",
        }
        research_scopes = tuple(
            (build_bundle(root, bundle_path=path).bundle_profile["profile"]["id"], environment)
            for path, environment in _PACKAGED_RESEARCH_BUNDLES
        )
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
    request = json.loads(sys.argv[1])
    for bundle_id, environment in json.loads(sys.argv[2]):
        service.create_bundle(
            bundle_id=bundle_id, version='99.8.0', mode='full', scope='environment',
            environment=environment, component_revisions=json.loads(sys.argv[3]), created_by='TEST',
        )
    print(service.create_bundle(**request, force_new_revision=True)['revisionId'])
finally:
    store.close()
""", json.dumps(legacy_request), json.dumps(research_scopes), json.dumps(original["componentRevisions"])],
            cwd=previous,
            env={
                **startup_environment, "BALOOT_ROOT": str(previous), "PYTHONPATH": str(previous / "src"),
                "BALOOT_DATABASE_URL": app_dsn or "",
            },
            check=True, capture_output=True, text=True,
        )
        expiring_id = expired.stdout.strip()
        instant = "2026-07-01T17:00:00+00:00"
        probe(
            expiring_id, recovery_target=original["revisionId"],
            diagnostic="No configuration bundle is effective", recovery_research_scopes=research_scopes,
        )
        recovered = service.resolve(bundle_id=original["bundleId"])["bundle"]["revisionId"]
        assert recovered not in {expiring_id, original["revisionId"]}
        def create_access_rows(scope, *, tenant=None, environment=None):
            data = copy.deepcopy(baseline["profiles"]["performance"])
            data["profile"].update(id="scoped_schedule", version="99.0.0")
            dimensions = {"scope": scope, "tenant_id": tenant, "environment": environment}
            revision = service.create_profile(
                family="performance", payload=data, created_by="TEST", publish=True,
                effective_from="2026-07-01T16:00:00Z", **dimensions,
            )
            bundle = service.create_bundle(
                bundle_id="scoped-schedule", version="99.0.0", mode=original["mode"],
                component_revisions={**original["componentRevisions"], "performance": revision["revisionId"]},
                created_by="TEST", publish=True, effective_from="2026-07-01T16:00:00Z", **dimensions,
            )
            return revision["revisionId"], bundle["revisionId"]

        access_rows = {
            tenant: create_access_rows("tenant", tenant=tenant)
            for tenant in ("fixture-tenant-a", "fixture-tenant-b")
        }
        access_rows["global"] = create_access_rows("global")
        access_rows["default"] = create_access_rows("tenant", tenant="default")
        access_rows["environment"] = create_access_rows("environment", environment="fixture-isolated")
        isolated_profile = access_rows["environment"][0]
        linked_bundle = service.create_bundle(
            bundle_id="scoped-edge", version="99.0.0", mode=original["mode"], created_by="TEST", publish=True,
            component_revisions={**original["componentRevisions"], "performance": isolated_profile},
        )
        access_rows["edge"] = (isolated_profile, linked_bundle["revisionId"])
        access_bundle = {"mode": original["mode"], "created_by": "TEST"}
        shared_components = original["componentRevisions"]
        hidden_components = {**shared_components, "performance": isolated_profile}
        handoff_anchor = service.create_bundle(
            **access_bundle, bundle_id="scoped-handoff", version="1.0.0",
            component_revisions=shared_components, publish=True,
        )
        handoff_successor = service.create_bundle(
            **access_bundle, bundle_id="scoped-handoff", version="1.0.1",
            component_revisions=hidden_components, publish=True, effective_from="2026-07-01T18:00Z",
        )
        access_rows["handoff"] = (handoff_anchor["revisionId"], handoff_successor["revisionId"])
        linked_payload = copy.deepcopy(baseline["profiles"]["performance"])
        linked_payload["profile"].update(id="scoped_linked_profile", version="99.0.0")
        linked_profile = service.create_profile(
            family="performance", payload=linked_payload, created_by="TEST", publish=True,
            supersedes_revision_id=isolated_profile,
        )
        linked_payload["profile"]["id"] = "scoped_isolated_later"
        late_profile = service.create_profile(
            family="performance", payload=linked_payload, created_by="TEST", publish=True,
            scope="environment", environment="fixture-isolated",
        )
        access_rows["profile-links"] = (linked_profile["revisionId"], isolated_profile)
        for name, versions in (
            ("scoped-chain", (
                ("1.0.0", shared_components, None),
                ("1.0.1", hidden_components, "2026-07-01T18:00Z"),
                ("1.0.2", {**shared_components, "performance": linked_profile["revisionId"]}, "2026-07-01T19:00Z"),
                ("1.0.3", {**shared_components, "performance": late_profile["revisionId"]}, "2026-07-01T20:00Z"),
            )),
            ("scoped-rollback-link", (
                ("1.0.0", shared_components, None),
                ("1.0.1", hidden_components, "2026-07-01T18:00Z"),
            )),
        ):
            revisions = tuple(
                service.create_bundle(
                    **access_bundle, bundle_id=name, version=version, component_revisions=components,
                    effective_from=start, publish=True,
                )["revisionId"]
                for version, components, start in versions
            )
            if name == "scoped-chain":
                access_rows["chain-base"] = revisions[:2]
                access_rows["chain-window"] = revisions[2:]
            else:
                access_rows["rollback-links"] = revisions
        access_rows["drafts"] = tuple(
            service.create_bundle(
                **access_bundle, bundle_id=bundle_name, version="1.0.0", component_revisions=components,
            )["revisionId"]
            for bundle_name, components in (("scoped-draft-visible", shared_components), ("scoped-draft-edge", hidden_components))
        )
        probe(recovered, access_rows=access_rows)
        probe(recovered, source=previous)

        original = service.resolve(bundle_id=original["bundleId"])["bundle"]
        native_bundle_id, native_environment = research_scopes[0]
        native = service.resolve(bundle_id=native_bundle_id, environment=native_environment)
        native_anchor = native["bundle"]["revisionId"]
        anchor_history = store.events("bundle", native_anchor)
        with pytest.raises(ValueError, match="previous-release startup"):
            service.retire_bundle(
                native_anchor, expected_etag=native["bundle"]["etag"], actor_id="TEST", reason="Retire packaged anchor",
            )
        assert store.get_bundle(native_anchor).status == "published"
        assert store.events("bundle", native_anchor) == anchor_history
        native_defaults = {
            "bundle_id": native_bundle_id, "mode": native["bundle"]["mode"], "scope": "environment",
            "environment": native_environment, "created_by": "TEST",
        }
        global_performance = original["componentRevisions"]["performance"]
        assert store.get_profile(global_performance).payload_sha256 == native["profileRevisions"]["performance"]["payloadSha256"]
        cross_scope = service.create_bundle(
            **native_defaults, version="99.8.4", publish=True,
            component_revisions={**native["bundle"]["componentRevisions"], "performance": global_performance},
        )
        assert service.resolve(bundle_id=native_bundle_id, environment=native_environment)["bundle"]["revisionId"] == cross_scope["revisionId"]
        assert [row.revision_id for row in store.list_bundles(bundle_id=native_bundle_id, status="published")] == [native_anchor]

        def native_components(version: str) -> dict[str, str]:
            payload = copy.deepcopy(native["profiles"]["performance"])
            payload["profile"].update({"id": f"research-handoff-performance-{version}", "version": version})
            profile = service.create_profile(
                family="performance", payload=payload, created_by="TEST", publish=True,
                scope="environment", environment=native_environment,
            )
            return {**native["bundle"]["componentRevisions"], "performance": profile["revisionId"]}

        immediate = service.create_bundle(
            **native_defaults, component_revisions=native_components("99.8.1"), version="99.8.1", publish=True,
        )

        scheduled = service.create_bundle(
            **defaults, version="99.7.6", component_revisions=new_components("99.7.6"),
            local_schedule={"timeZone": "America/New_York", "effectiveFrom": "2026-07-01T12:00", "effectiveUntil": "2026-07-01T13:00"},
        )
        published_filter = {"bundle_id": original["bundleId"], "status": "published"}
        assert [row.revision_id for row in store.list_bundles(**published_filter)] == [original["revisionId"]]
        assert [row.revision_id for row in store.list_bundles(**published_filter, include_scheduled=True)] == [scheduled["revisionId"]]
        for index, (instant, expected) in enumerate((
            ("2026-07-01T15:59:59+00:00", original["revisionId"]),
            ("2026-07-01T16:00:00+00:00", scheduled["revisionId"]),
            ("2026-07-01T17:00:00+00:00", original["revisionId"]),
        )):
            native_expected = immediate["revisionId"]
            if index == 1:
                native_window = service.create_bundle(
                    **native_defaults, component_revisions=native_components("99.8.2"), version="99.8.2",
                    effective_from="2026-07-01T16:00:00Z", effective_until="2026-07-01T17:00:00Z",
                )
                promoted = store.publish_bundle(native_window["revisionId"], expected_etag=native_window["etag"], actor_id="TEST")
                native_expected = promoted.revision_id
            assert service.resolve(bundle_id=original["bundleId"])["bundle"]["revisionId"] == expected
            probe(expected, research=(native_bundle_id, native_environment, native_expected))
            probe(original["revisionId"], source=previous, research=(native_bundle_id, native_environment, native_anchor))
        print(f"previousReleaseScheduledHandoffStartup: {backend} before, during and after")
        print(f"previousReleaseResearchAnchorStartup: {backend} after immediate and timed publication")

        native_restore = service.rollback_bundle(native_anchor, new_version="99.8.3", actor_id="TEST")
        native_publication = store.list_bundles(bundle_id=native_bundle_id, status="published")
        assert [row.revision_id for row in native_publication] == [native_restore["revisionId"]]
        assert native_restore["supersedesRevisionId"] == promoted.revision_id

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

        competing_id = competing["revisionId"]
        selection = {"bundle_id": original["bundleId"], "at_time": "2026-07-02T00:00:00Z"}
        assert service.resolve(**selection)["bundle"]["revisionId"] == competing_id
        instant = selection["at_time"]
        decode_bundle = store._bundle
        broken_predecessors = ["CBR-MISSING", scheduled["revisionId"]]
        if backend == "sqlite":
            broken_predecessors.append(b"non-text predecessor")
        for broken_predecessor in broken_predecessors:
            def decode_broken_history(row):
                revision = decode_bundle(row)
                if revision is not None and revision.revision_id == scheduled["revisionId"]:
                    return replace(revision, scheduled_predecessor_id=broken_predecessor)
                return revision

            with monkeypatch.context() as patch:
                patch.setattr(store, "_bundle", decode_broken_history)
                service._invalidate_cache()
                with pytest.raises(ConfigurationIntegrityError, match="scheduled publication could not be verified"):
                    service.bootstrap_packaged_bundle(actor_id="TEST", packaged_bundle=packaged)
                with pytest.raises(ConfigurationIntegrityError, match="scheduled publication could not be verified"):
                    service.resolve(**selection)
        assert service.resolve(**selection)["bundle"]["revisionId"] == competing_id
        print(f"scheduleIntegrityBeforeSelection: {backend}")
        before = store.list_bundles(bundle_id=original["bundleId"])
        audit_before = {row.revision_id: store.events("bundle", row.revision_id) for row in before}
        epoch_before = store.get_invalidation_epoch()
        reason = "Remove later window. " + "x" * 2000
        retirement = {
            "revision_id": competing_id, "expected_etag": competing["etag"], "actor_id": "TEST", "reason": reason,
        }
        append_event = store._append_event_tx

        def fail_retirement_reason(conn, aggregate_type, aggregate_id, event_type, actor_id, payload):
            if event_type == "retirement_reason" and aggregate_id == competing_id:
                raise RuntimeError("retirement audit unavailable")
            return append_event(conn, aggregate_type, aggregate_id, event_type, actor_id, payload)

        with monkeypatch.context() as patch:
            patch.setattr(store, "_append_event_tx", fail_retirement_reason)
            with pytest.raises(RuntimeError, match="retirement audit unavailable"):
                service.retire_bundle(**retirement)
        assert store.list_bundles(bundle_id=original["bundleId"]) == before
        assert {key: store.events("bundle", key) for key in audit_before} == audit_before
        assert store.get_invalidation_epoch() == epoch_before
        assert service.resolve(**selection)["bundle"]["revisionId"] == competing_id

        instant = "2026-07-02T00:00:00+00:00"
        choose = service._effective_scheduled_bundle
        retired = {}
        competing_profile_id = competing["componentRevisions"]["performance"]
        competing_profile = store.get_profile(competing_profile_id)
        verified_before = store.verify_events("bundle", competing_id)

        def retire_between_snapshot_and_selection(*args, **kwargs):
            retired.update(service.retire_bundle(**retirement))
            assert store.get_bundle(competing_id).status == "retired"
            assert store.get_invalidation_epoch() == epoch_before + 1
            service.retire_profile(
                competing_profile_id, expected_etag=competing_profile.etag, actor_id="TEST",
                reason="Retire unused scheduled component",
            )
            assert store.get_profile(competing_profile_id).status == "retired"
            return choose(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(service, "_effective_scheduled_bundle", retire_between_snapshot_and_selection)
            startup = service.bootstrap_packaged_bundle(actor_id="TEST", packaged_bundle=packaged)
        assert startup["bundle"]["revisionId"] == competing_id
        assert startup["profileRevisions"]["performance"]["status"] == "published"
        assert startup["invalidationEpoch"] == epoch_before
        assert startup["audit"] == verified_before
        assert service.bootstrap_packaged_bundle(actor_id="TEST", packaged_bundle=packaged)["bundle"]["revisionId"] == original["revisionId"]
        assert retired["status"] == "retired" and retired["etag"] == competing["etag"] + 1
        events = store.events("bundle", competing_id)
        assert events[:-2] == audit_before[competing_id]
        assert [(event["eventType"], event["payload"]) for event in events[-2:]] == [
            ("retired", {"etag": retired["etag"], "status": "retired"}),
            ("retirement_reason", {"reason": reason[:2000]}),
        ]
        assert all(event["actorId"] == "TEST" for event in events[-2:])
        assert store.get_invalidation_epoch() == epoch_before + 2
        assert service.resolve(**selection)["bundle"]["revisionId"] == original["revisionId"]
        print(f"retirementAuditAtomicity: {backend}")

        anchor = store.get_bundle(original["revisionId"])
        def write_successor(anchor_id, successor):
            with (contextual_connection(store.pool) if app_dsn is not None else store._connect()) as conn:
                store._set_scheduled_successor_tx(conn, anchor_id, successor)

        invalid_successors = ["", " ", competing_id, anchor.revision_id]
        if backend == "sqlite":
            invalid_successors.append(b"non-text successor")
        for invalid_successor in invalid_successors:
            write_successor(anchor.revision_id, invalid_successor)
            service._invalidate_cache()
            try:
                with pytest.raises(ConfigurationIntegrityError, match="scheduled publication could not be verified"):
                    service.bootstrap_packaged_bundle(actor_id="TEST", packaged_bundle=packaged)
                with pytest.raises(ConfigurationIntegrityError, match="scheduled publication could not be verified"):
                    service.resolve(**selection)
            finally:
                write_successor(anchor.revision_id, anchor.scheduled_successor_id)
        try:
            write_successor(anchor.revision_id, None)
            service._invalidate_cache()
            assert service.resolve(**selection)["bundle"]["revisionId"] == anchor.revision_id
            assert service.bootstrap_packaged_bundle(actor_id="TEST", packaged_bundle=packaged)["bundle"]["revisionId"] == anchor.revision_id
        finally:
            write_successor(anchor.revision_id, anchor.scheduled_successor_id)
            service._invalidate_cache()
        print(f"scheduleLinkIntegrity: {backend}")
        print(f"startupRetirementSnapshot: {backend}")

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
        if app_dsn is not None:
            with psycopg.connect(app_dsn) as connection:
                connection.execute("SELECT set_config('app.role', 'service', true)")
                connection.execute("UPDATE configuration.profile_revisions SET status='retired' WHERE revision_id=%s", (scheduled_profile_id,))
        else:
            with store._connect() as connection:
                connection.execute("UPDATE configuration_profile_revisions SET status='retired' WHERE revision_id=?", (scheduled_profile_id,))
        instant = "2026-07-01T16:00:00+00:00"
        probe(scheduled_profile_id, recovery_target=original["revisionId"], diagnostic="unavailable component")

        interim = service.create_bundle(**defaults, version="99.7.7", component_revisions=new_components("99.7.7"))
        service.retire_bundle(interim["revisionId"], expected_etag=interim["etag"], actor_id="TEST", reason="Test expiry without predecessor")
        bounded_request = {
            **defaults, "version": "99.7.8", "component_revisions": new_components("99.7.8"),
            "effective_until": "2026-07-01T17:00:00Z",
        }
        before = store.list_bundles(bundle_id=original["bundleId"])
        audit_before = [store.events("bundle", row.revision_id) for row in before]
        with pytest.raises(ConfigurationScheduleConflict, match="requires an unbounded published bundle at full rollout"):
            service.create_bundle(**bounded_request)
        assert store.list_bundles(bundle_id=original["bundleId"]) == before
        assert [store.events("bundle", row.revision_id) for row in before] == audit_before
        draft = service.create_bundle(**{**bounded_request, "publish": False})
        draft_before = store.get_bundle(draft["revisionId"])
        draft_audit = store.events("bundle", draft["revisionId"])
        with pytest.raises(ConfigurationScheduleConflict, match="requires an unbounded published bundle at full rollout"):
            store.publish_bundle(draft["revisionId"], expected_etag=draft["etag"], actor_id="TEST")
        assert store.get_bundle(draft["revisionId"]) == draft_before
        assert store.events("bundle", draft["revisionId"]) == draft_audit

        partial = service.create_bundle(
            **defaults, version="99.7.11", component_revisions=new_components("99.7.11"), rollout_percentage=10,
        )
        exact = service.resolve(revision_id=partial["revisionId"])
        assert exact["bundle"]["scope"] == original["scope"]
        assert exact["resolutionTrace"] == [{"scope": "exact_revision", "revisionId": partial["revisionId"]}]
        with pytest.raises(KeyError, match="published configuration bundle not found"):
            service.resolve(bundle_id=original["bundleId"])
        with pytest.raises(ConfigurationIntegrityError, match="No configuration bundle is effective"):
            service.bootstrap_packaged_bundle(actor_id="TEST", packaged_bundle=packaged)
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
        other_scope = {**defaults, "scope": "tenant", "tenant_id": "fixture-tenant-a"}
        service.create_bundle(
            **other_scope, version="99.7.14", component_revisions=original["componentRevisions"],
        )
        wrong_scope_id = service.create_bundle(
            **other_scope, version="99.7.15", component_revisions=bounded_components,
            effective_from="2026-07-01T16:00:00Z",
        )["revisionId"]
        wrong_scope = store.get_bundle(wrong_scope_id)
        wrong_scope_audit = store.events("bundle", wrong_scope_id)
        missing_id = "CBR-000000000000000000000000"
        assert store.get_bundle(missing_id) is None
        invalid_links: list[tuple[str, str | bytes]] = [
            ("missing", missing_id), ("wrong-scope", wrong_scope_id),
        ]
        if backend == "sqlite":
            invalid_links.append(("non-text", b"non-text successor"))
        affected_id = partial["revisionId"]
        for case, successor in invalid_links:
            write_successor(affected_id, successor)
            service._invalidate_cache()
            with pytest.raises(ConfigurationIntegrityError, match="scheduled publication could not be verified"):
                service.resolve(bundle_id=original["bundleId"])
            prior_events = store.events("bundle", affected_id)
            probe(
                affected_id, recovery_target=original["revisionId"],
                diagnostic="scheduled publication could not be verified",
            )
            replacement = service.resolve(bundle_id=original["bundleId"])["bundle"]
            assert replacement["supersedesRevisionId"] == affected_id
            events = store.events("bundle", affected_id)
            assert events[:-1] == prior_events
            assert events[-1]["eventType"] == "superseded"
            assert store.get_bundle(wrong_scope_id) == wrong_scope
            assert store.events("bundle", wrong_scope_id) == wrong_scope_audit
            probe(replacement["revisionId"], recovery_audit=(original["bundleId"], affected_id))
            print(f"scheduleRecoveryLineage: {backend} {case}")
            affected_id = replacement["revisionId"]
    finally:
        CURRENT_DATABASE_IDENTITY.reset(token)
        store.close()
