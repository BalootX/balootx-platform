"""Configuration integration tests for the disposable PostgreSQL CI service."""
import hashlib
import os
from pathlib import Path

import psycopg

from baloot_platform.configuration import PostgresConfigurationStore
from baloot_platform.storage.request_context import (
    CURRENT_DATABASE_IDENTITY,
    DatabaseIdentityContext,
)
from scripts.verify_live_postgres_research import _prove_configuration_schedule


def test_configuration_schedule_migrates_populated_postgres_and_round_trips(root: Path):
    dsn = os.environ["BALOOT_TEST_POSTGRES_URL"]
    migration = root / "migrations/030_configuration_local_schedule.sql"
    with psycopg.connect(dsn, autocommit=True) as connection:
        for predecessor in sorted(migration.parent.glob("*.sql")):
            if predecessor.name < migration.name:
                connection.execute(predecessor.read_text(encoding="utf-8"))
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
                content_sha256, sequence, created_by, effective_until)
               VALUES ('CBR-LEGACY', 'legacy', 'full', '1.0.0', 'global', 'draft',
                       '{"rules":"CPR-LEGACY"}', repeat('b', 64), 1, 'TEST',
                       '2026-08-01T12:00+03:00')"""
        )
        connection.execute(migration.read_text(encoding="utf-8"))
        connection.execute(migration.read_text(encoding="utf-8"))
        assert connection.execute(
            "SELECT effective_from::text FROM configuration.profile_revisions WHERE revision_id='CPR-LEGACY'"
        ).fetchone() is not None

    token = CURRENT_DATABASE_IDENTITY.set(DatabaseIdentityContext(subject_id="TEST", role="service"))
    store = PostgresConfigurationStore(dsn)
    try:
        old_profile = store.get_profile("CPR-LEGACY")
        old_bundle = store.get_bundle("CBR-LEGACY")
        assert old_profile.effective_from == "2026-07-01T09:00:00+00:00"
        assert old_bundle.effective_until == "2026-08-01T09:00:00+00:00"
        assert "localSchedule" not in old_profile.to_dict()
        assert "localSchedule" not in old_bundle.to_dict()
        _prove_configuration_schedule(store, {"rules": old_profile.revision_id})
    finally:
        store.close()
        CURRENT_DATABASE_IDENTITY.reset(token)

    print(f"migration 030 sha256 {hashlib.sha256(migration.read_bytes()).hexdigest()}")
    print("configurationLocalScheduleRoundTrip: true")
