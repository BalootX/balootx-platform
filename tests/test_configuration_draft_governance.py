from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from baloot_platform.configuration import (
    PostgresConfigurationStore,
    SQLiteConfigurationStore,
    VersionedConfigurationService,
)
from baloot_platform.configuration.models import (
    BundleRevision,
    ProfileRevision,
    digest,
)
from baloot_platform.identity import IdentityService, SQLiteIdentityStore
from baloot_platform.security import AuthorizationPolicy
import baloot_platform.service.container as container_module
from baloot_platform.service.container import build_container
from baloot_platform.service.routes.configuration import build_configuration_router
from baloot_platform.storage.request_context import (
    CURRENT_DATABASE_IDENTITY,
    DatabaseIdentityContext,
)


def _service(root: Path, tmp_path: Path) -> VersionedConfigurationService:
    governance = yaml.safe_load(
        (root / "configs/configuration/control_plane_v19.yaml").read_text(
            encoding="utf-8"
        )
    )
    service = VersionedConfigurationService(
        SQLiteConfigurationStore(tmp_path / "configuration.sqlite3"),
        root=root,
        profile=governance,
    )
    service.bootstrap_packaged_bundle(actor_id="TEST")
    return service


def _draft_performance(service: VersionedConfigurationService) -> dict[str, Any]:
    payload = copy.deepcopy(service.resolve()["profiles"]["performance"])
    payload["profile"] = dict(payload["profile"])
    payload["profile"].update(
        {"id": "draft_governance_test", "version": "99.0.0", "status": "draft"}
    )
    return service.create_profile(
        family="performance", payload=payload, created_by="TEST"
    )


def test_native_challenge_clone_retains_supported_version(root: Path, tmp_path: Path) -> None:
    service = _service(root, tmp_path)
    payload = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
    original = service.create_profile(family="challenge", payload=payload, created_by="TEST", publish=True)
    draft_payload = copy.deepcopy(payload)
    draft_payload["profile"]["status"] = "draft"
    draft = service.create_profile(family="challenge", payload=draft_payload, created_by="TEST", supersedes_revision_id=original["revisionId"])
    assert draft["version"] == original["version"]
    assert draft["revisionId"] != original["revisionId"]
    assert draft["payloadSha256"] != original["payloadSha256"]
    assert service.validate_profile_revision(draft["revisionId"])["valid"] is True
    source = (root / "src/baloot_platform/service/foundry_dashboard/src/main.tsx").read_text(encoding="utf-8")
    clone = source.split("const cloneDraft = async () => {", 1)[1].split("const validateDraft", 1)[0]
    assert "nextPatchVersion" not in clone


def test_validation_budget_successor_keeps_published_revision_replayable(root, tmp_path):
    service = _service(root, tmp_path)
    selections = []
    for name in ("intelligent", "intelligent_validation_5s"):
        selections.append(service.bootstrap_packaged_bundle(
            actor_id="TEST-VALIDATION-BUDGET", scope="environment",
            environment="validation-budget-test",
            bundle_path=f"configs/bundle_ortools_research_native_v3_{name}.yaml",
        ))
    old, new = selections
    assert old["bundle"]["revisionId"] != new["bundle"]["revisionId"]
    assert old["profileRevisions"]["challenge"]["payloadSha256"] != new["profileRevisions"]["challenge"]["payloadSha256"]
    restored = service.resolve(revision_id=old["bundle"]["revisionId"])
    assert restored["profiles"] == old["profiles"]
    for selected, seconds in ((restored, 1.0), (new, 5.0)):
        block = selected["profiles"]["challenge"]["scenarioResearchV4"]
        assert block["searchBudgets"]["profiles"]["interactive_validation_v1"]["maximumWallSeconds"] == seconds
        assert block["proofBudgets"] == old["profiles"]["challenge"]["scenarioResearchV4"]["proofBudgets"]


def test_native_delayed_draft_roundtrips_without_runtime_activation(root, tmp_path):
    service = _service(root, tmp_path)
    before = service.resolve()
    payload = yaml.safe_load(
        (root / "configs/challenge/native_delayed_payoff_v3.yaml").read_text(encoding="utf-8")
    )
    draft = service.create_profile(
        family="challenge", payload=payload, created_by="TEST-DELAYED-DRAFT",
    )
    assert draft["status"] == "draft" and draft["publishedAt"] is None
    assert draft["payload"]["tactical_detectors"]["nativeDelayedPayoff"] == {
        "gainMetric": "raw_card_points", "minimumGain": 8,
    }
    cold = VersionedConfigurationService(
        SQLiteConfigurationStore(tmp_path / "configuration.sqlite3"),
        root=root, profile=service.profile,
    )
    saved = cold.store.get_profile(draft["revisionId"])
    assert saved is not None and saved.to_dict() == draft
    validation = cold.validate_profile_revision(draft["revisionId"])
    assert validation["valid"] is True
    assert validation["automaticProductionApply"] is False
    with pytest.raises(PermissionError, match="requires a published revision"):
        cold.resolve_profile(
            family="challenge", profile_id=draft["profileId"],
            revision_id=draft["revisionId"],
        )
    after = cold.resolve()
    assert after["bundle"]["revisionId"] == before["bundle"]["revisionId"]
    assert after["profiles"] == before["profiles"]


def test_sqlite_draft_patch_uses_etag_and_keeps_published_immutable(
    root: Path, tmp_path: Path
) -> None:
    service = _service(root, tmp_path)
    draft = _draft_performance(service)
    payload = copy.deepcopy(draft["payload"])
    payload["budgets"] = dict(payload["budgets"])
    payload["budgets"]["configuration_resolve_p95_ms"] += 1

    updated = service.update_profile(
        draft["revisionId"],
        payload=payload,
        expected_etag=draft["etag"],
        actor_id="TEST",
    )
    assert updated["etag"] == 2
    assert updated["payloadSha256"] != draft["payloadSha256"]
    assert service.validate_profile_revision(draft["revisionId"])["valid"] is True
    with pytest.raises(RuntimeError, match="etag conflict"):
        service.update_profile(
            draft["revisionId"],
            payload=payload,
            expected_etag=1,
            actor_id="TEST",
        )

    published = service.publish_profile(
        draft["revisionId"], expected_etag=2, actor_id="TEST"
    )
    with pytest.raises(ValueError, match="only draft"):
        service.update_profile(
            published["revisionId"],
            payload=payload,
            expected_etag=published["etag"],
            actor_id="TEST",
        )
    events = service.store.events("profile", draft["revisionId"])
    assert any(event["eventType"] == "draft_updated" for event in events)


class _FakeResult:
    def __init__(
        self,
        row: dict[str, Any] | None = None,
        rows: list[dict[str, Any]] | None = None,
    ):
        self.row = row
        self.rows = rows or []

    def fetchone(self) -> dict[str, Any] | None:
        return self.row

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self.rows)


class _FakeConnection:
    def __init__(self, current: dict[str, Any]):
        self.current = current
        self.heads: dict[tuple[str, str], dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.epoch = 1
        self.statements: list[str] = []

    def execute(self, sql: str, args: Any = None) -> _FakeResult:
        self.statements.append(" ".join(sql.split()))
        if sql.startswith("SELECT set_config"):
            return _FakeResult()
        if sql.startswith("INSERT INTO configuration.aggregate_heads"):
            aggregate_type, aggregate_id, *_ = args
            self.heads.setdefault(
                (aggregate_type, aggregate_id),
                {
                    "next_sequence": 1,
                    "etag": 1,
                    "updated_at": "2026-01-01T00:00:00+00:00",
                },
            )
            return _FakeResult()
        if "FROM configuration.aggregate_heads" in sql and "FOR UPDATE" in sql:
            return _FakeResult(dict(self.heads[(args[0], args[1])]))
        if sql.startswith("UPDATE configuration.aggregate_heads"):
            next_sequence, updated_at, aggregate_type, aggregate_id, expected_etag = (
                args
            )
            head = self.heads[(aggregate_type, aggregate_id)]
            assert head["etag"] == expected_etag
            head.update(
                {
                    "next_sequence": next_sequence,
                    "etag": expected_etag + 1,
                    "updated_at": updated_at,
                }
            )
            return _FakeResult(dict(head))
        if "COALESCE(MAX(sequence),0) AS max_sequence" in sql:
            return _FakeResult({"max_sequence": self.current["sequence"]})
        if "FROM configuration.profile_revisions" in sql and "FOR UPDATE" in sql:
            return _FakeResult(dict(self.current))
        if "UPDATE configuration.profile_revisions" in sql:
            profile_id, version, payload_json, payload_hash, updated_at, _, _ = args
            self.current.update(
                {
                    "profile_id": profile_id,
                    "version": version,
                    "payload": payload_json,
                    "payload_sha256": payload_hash,
                    "updated_at": updated_at,
                    "etag": self.current["etag"] + 1,
                }
            )
            return _FakeResult(dict(self.current))
        if "SELECT sequence,event_hash FROM configuration.events" in sql:
            matching = [
                event
                for event in self.events
                if event["aggregate_type"] == args[0]
                and event["aggregate_id"] == args[1]
            ]
            return _FakeResult(dict(matching[-1]) if matching else None)
        if sql.startswith("INSERT INTO configuration.events"):
            (
                event_id,
                aggregate_type,
                aggregate_id,
                sequence,
                event_type,
                actor_id,
                payload,
                previous_hash,
                event_hash,
                created_at,
            ) = args
            self.events.append(
                {
                    "event_id": event_id,
                    "aggregate_type": aggregate_type,
                    "aggregate_id": aggregate_id,
                    "sequence": sequence,
                    "event_type": event_type,
                    "actor_id": actor_id,
                    "payload": payload,
                    "previous_hash": previous_hash,
                    "event_hash": event_hash,
                    "created_at": created_at,
                }
            )
            return _FakeResult()
        if sql.startswith("INSERT INTO configuration.invalidation_epochs"):
            self.epoch += 1
            return _FakeResult({"epoch": self.epoch, "updated_at": args[-1]})
        raise AssertionError(sql)


class _FakePool:
    def __init__(self, connection: _FakeConnection):
        self._connection = connection

    @contextmanager
    def connection(self):
        yield self._connection


def test_postgres_draft_patch_uses_same_optimistic_contract() -> None:
    row = {
        "revision_id": "CPR-TEST",
        "family": "performance",
        "profile_id": "before",
        "version": "1.0.0",
        "scope": "global",
        "scope_key": "*",
        "tenant_id": None,
        "environment": None,
        "status": "draft",
        "payload": {"profile": {"id": "before", "version": "1.0.0"}},
        "payload_sha256": "0" * 64,
        "sequence": 1,
        "effective_from": None,
        "effective_until": None,
        "rollout_percentage": 100,
        "rollout_seed": "default",
        "created_by": "TEST",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "published_by": None,
        "published_at": None,
        "supersedes_revision_id": None,
        "etag": 1,
    }
    store = PostgresConfigurationStore.__new__(PostgresConfigurationStore)
    connection = _FakeConnection(row)
    store.pool = _FakePool(connection)
    result = store.update_profile_draft(
        "CPR-TEST",
        expected_etag=1,
        profile_id="after",
        version="2.0.0",
        payload={"profile": {"id": "after", "version": "2.0.0"}},
        actor_id="TEST",
    )
    assert result.profile_id == "after"
    assert result.version == "2.0.0"
    assert result.etag == 2
    assert [event["event_type"] for event in connection.events] == ["draft_updated"]
    assert connection.epoch == 2
    assert any("payload=%s::jsonb" in sql for sql in connection.statements)
    assert all("payload_json" not in sql for sql in connection.statements)


class _Compiled:
    def __init__(self, publishable: bool):
        self.publishable = publishable

    def to_dict(self, *, include_payloads: bool = False) -> dict[str, Any]:
        return {
            "sha256": "a" * 64,
            "compiledDomain": {
                "compilerVersion": "test_compiler_v1",
                "coverage": {
                    "covered": 3 if self.publishable else 2,
                    "total": 3,
                    "complete": self.publishable,
                },
                "publishable": self.publishable,
                "sha256": "b" * 64,
            },
        }


def test_draft_bundle_compile_is_preview_only_and_publication_is_gated(
    root: Path, tmp_path: Path
) -> None:
    service = _service(root, tmp_path)
    current = service.resolve()
    service.set_bundle_compiler(lambda _: _Compiled(False))
    draft = service.create_bundle(
        bundle_id="draft_domain_test",
        version="1.0.0",
        mode="full",
        component_revisions={
            family: item["revisionId"]
            for family, item in current["profileRevisions"].items()
        },
        created_by="TEST",
        force_new_revision=True,
    )
    preview = service.compile_bundle(draft["revisionId"])
    assert preview["previewOnly"] is True
    assert preview["automaticProductionApply"] is False
    assert preview["publishable"] is False
    assert service.resolve()["bundle"]["revisionId"] == current["bundle"]["revisionId"]
    with pytest.raises(ValueError, match="coverage is incomplete"):
        service.publish_bundle(draft["revisionId"], expected_etag=1, actor_id="TEST")
    assert service.store.get_bundle(draft["revisionId"]).status == "draft"

    service.set_bundle_compiler(lambda _: _Compiled(True))
    published = service.publish_bundle(
        draft["revisionId"], expected_etag=1, actor_id="TEST"
    )
    assert published["status"] == "published"


def test_v1_and_v2_profile_routes_share_the_draft_update_contract(
    root: Path, tmp_path: Path
) -> None:
    service = _service(root, tmp_path)
    identities = IdentityService(SQLiteIdentityStore(tmp_path / "identity.sqlite3"))
    app = FastAPI()

    @app.middleware("http")
    async def mark_api_key(request: Request, call_next):
        request.state.api_key_valid = True
        return await call_next(request)

    app.include_router(
        build_configuration_router(
            service,
            identities,
            AuthorizationPolicy(),
            {"id": "test"},
            {"id": "test"},
        )
    )
    client = TestClient(app)
    headers = {
        "x-api-key": "test",
        "x-baloot-role": "admin",
        "x-baloot-subject-id": "TEST",
    }
    draft = _draft_performance(service)
    payload = copy.deepcopy(draft["payload"])
    payload["profile"]["version"] = "99.0.1"
    response = client.patch(
        f"/api/v2/configuration/profiles/{draft['revisionId']}",
        headers=headers,
        json={"expectedEtag": 1, "payload": payload},
    )
    assert response.status_code == 200
    assert response.json()["etag"] == 2
    validated = client.post(
        f"/api/v2/configuration/profiles/{draft['revisionId']}:validate",
        headers=headers,
    )
    assert validated.status_code == 200
    assert validated.json()["valid"] is True
    legacy = client.get(
        f"/api/v1/configuration-registry/profiles/{draft['revisionId']}",
        headers=headers,
    )
    assert legacy.status_code == 200
    assert legacy.json()["etag"] == 2
    conflict = client.patch(
        f"/api/v1/configuration-registry/profiles/{draft['revisionId']}",
        headers=headers,
        json={"expectedEtag": 1, "payload": payload},
    )
    assert conflict.status_code == 409


def test_foundry_rules_operator_flow_is_api_backed_and_coverage_gated(
    root: Path, tmp_path: Path
) -> None:
    service = _service(root, tmp_path)
    service.set_bundle_compiler(lambda _: _Compiled(True))
    identities = IdentityService(
        SQLiteIdentityStore(tmp_path / "operator-identity.sqlite3")
    )
    app = FastAPI()

    @app.middleware("http")
    async def mark_api_key(request: Request, call_next):
        request.state.api_key_valid = True
        return await call_next(request)

    app.include_router(
        build_configuration_router(
            service,
            identities,
            AuthorizationPolicy(),
            {"id": "test"},
            {"id": "test"},
        )
    )
    client = TestClient(app)
    headers = {
        "x-api-key": "test",
        "x-baloot-role": "admin",
        "x-baloot-subject-id": "TEST",
    }
    current = service.resolve()
    source = current["profileRevisions"]["performance"]
    payload = copy.deepcopy(current["profiles"]["performance"])
    payload["profile"] = dict(payload["profile"])
    payload["profile"].update({"version": "98.0.0", "status": "draft"})

    schema = client.get(
        "/api/v2/configuration/profile-schemas/performance", headers=headers
    )
    assert schema.status_code == 200
    assert "properties" in schema.json()
    created = client.post(
        "/api/v2/configuration/profiles",
        headers=headers,
        json={
            "family": "performance",
            "payload": payload,
            "supersedesRevisionId": source["revisionId"],
            "publish": False,
        },
    )
    assert created.status_code == 201, created.text
    draft = created.json()
    payload["profile"]["version"] = "98.0.1"
    updated = client.patch(
        f"/api/v2/configuration/profiles/{draft['revisionId']}",
        headers=headers,
        json={"payload": payload, "expectedEtag": draft["etag"]},
    )
    assert updated.status_code == 200, updated.text
    draft = updated.json()
    validated = client.post(
        f"/api/v2/configuration/profiles/{draft['revisionId']}:validate",
        headers=headers,
    )
    assert validated.status_code == 200
    assert validated.json()["valid"] is True

    components = {
        family: revision["revisionId"]
        for family, revision in current["profileRevisions"].items()
    }
    components["performance"] = draft["revisionId"]
    bundle = client.post(
        "/api/v2/configuration/bundles",
        headers=headers,
        json={
            "bundleId": current["bundle"]["bundleId"],
            "version": "98.0.0",
            "componentRevisions": components,
            "supersedesRevisionId": current["bundle"]["revisionId"],
            "allowDraftComponents": True,
        },
    )
    assert bundle.status_code == 201, bundle.text
    preview = client.post(
        f"/api/v2/configuration/bundles/{bundle.json()['revisionId']}:compile",
        headers=headers,
    )
    assert preview.status_code == 200, preview.text
    assert preview.json()["previewOnly"] is True
    assert preview.json()["publishable"] is True
    assert preview.json()["coverage"]["complete"] is True

    published_profile = client.post(
        f"/api/v2/configuration/profiles/{draft['revisionId']}:publish",
        headers=headers,
        json={"expectedEtag": draft["etag"]},
    )
    assert published_profile.status_code == 200, published_profile.text
    published_bundle = client.post(
        f"/api/v2/configuration/bundles/{bundle.json()['revisionId']}:publish",
        headers=headers,
        json={"expectedEtag": bundle.json()["etag"]},
    )
    assert published_bundle.status_code == 200, published_bundle.text
    comparison = client.get(
        "/api/v1/configuration-registry/profiles:compare",
        headers=headers,
        params={"left": source["revisionId"], "right": draft["revisionId"]},
    )
    assert comparison.status_code == 200
    assert comparison.json()["changes"]
    rollback = client.post(
        f"/api/v1/configuration-registry/bundles/{current['bundle']['revisionId']}:rollback",
        headers=headers,
        json={"newVersion": "98.0.1"},
    )
    assert rollback.status_code == 201, rollback.text
    assert rollback.json()["status"] == "published"
    retired = client.post(
        f"/api/v2/configuration/profiles/{draft['revisionId']}:retire",
        headers=headers,
        json={
            "expectedEtag": published_profile.json()["etag"],
            "reason": "operator flow verified",
        },
    )
    assert retired.status_code == 200, retired.text
    assert retired.json()["status"] == "retired"


def test_real_container_compiles_draft_bundle_with_governed_conventions_family(
    root: Path, tmp_path: Path
) -> None:
    container = build_container(root=root, writable_root=tmp_path / "runtime")
    try:
        service = container.configuration
        current = service.resolve()
        rules = copy.deepcopy(current["profiles"]["rules"])
        rules["profile"] = dict(rules["profile"])
        rules["profile"].update({"version": "99.0.0", "status": "draft"})
        rules["dealing"] = dict(rules["dealing"])
        rules["dealing"]["allow_no_cut"] = not bool(rules["dealing"]["allow_no_cut"])
        revision = service.create_profile(
            family="rules",
            payload=rules,
            created_by="TEST",
            supersedes_revision_id=current["profileRevisions"]["rules"]["revisionId"],
        )
        components = dict(current["bundle"]["componentRevisions"])
        components["rules"] = revision["revisionId"]
        bundle = service.create_bundle(
            bundle_id=current["bundle"]["bundleId"],
            version="99.0.0",
            mode="full",
            component_revisions=components,
            created_by="TEST",
            supersedes_revision_id=current["bundle"]["revisionId"],
            allow_draft_components=True,
        )

        preview = service.compile_bundle(bundle["revisionId"])

        assert preview["previewOnly"] is True
        assert preview["automaticProductionApply"] is False
        assert preview["coverage"]["complete"] is True
    finally:
        container.shutdown()


def _minimal_control_plane(precedence: str) -> dict[str, Any]:
    return {
        "governance": {"maximum_payload_bytes": 1_048_576},
        "resolution": {
            "cache_entries": 0,
            "cache_ttl_seconds": 0,
            "precedence": [precedence],
        },
    }


@pytest.mark.parametrize(
    ("scope", "scope_key", "tenant_id", "environment", "message"),
    [
        ("global", "OTHER", None, None, "scope_key"),
        ("global", "*", "TEN-A", None, "tenant_id"),
        ("environment", "prod", "TEN-A", "prod", "tenant_id"),
        ("environment", "prod", None, "staging", "must match"),
        ("tenant", "TEN-A", "TEN-B", None, "must match"),
        ("room", "ROOM-A", "", None, "nonblank"),
        ("room", "ROOM-A", None, "", "nonblank"),
    ],
)
def test_scope_normalization_rejects_dimensions_it_would_otherwise_drop(
    scope: str,
    scope_key: str | None,
    tenant_id: str | None,
    environment: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        VersionedConfigurationService._scope(
            scope,
            scope_key=scope_key,
            tenant_id=tenant_id,
            environment=environment,
        )


def _profile_revision(
    revision_id: str,
    *,
    scope: str,
    scope_key: str,
    tenant_id: str | None,
    environment: str | None,
    sequence: int,
) -> ProfileRevision:
    payload = {"profile": {"id": "isolation", "version": "1.0.0"}}
    return ProfileRevision(
        revision_id=revision_id,
        family="performance",
        profile_id="isolation",
        version="1.0.0",
        scope=scope,
        scope_key=scope_key,
        tenant_id=tenant_id,
        environment=environment,
        status="published",
        payload=payload,
        payload_sha256=digest(payload),
        sequence=sequence,
        created_by="TEST",
        created_at="2026-08-10T00:00:00+00:00",
        updated_at="2026-08-10T00:00:00+00:00",
    )


def _bundle_revision(
    revision_id: str,
    *,
    scope: str,
    scope_key: str,
    tenant_id: str | None,
    environment: str | None,
    sequence: int,
) -> BundleRevision:
    components: dict[str, str] = {}
    return BundleRevision(
        revision_id=revision_id,
        bundle_id="isolation-bundle",
        version="1.0.0",
        mode="full",
        scope=scope,
        scope_key=scope_key,
        tenant_id=tenant_id,
        environment=environment,
        status="published",
        component_revisions=components,
        content_sha256=digest({"mode": "full", "components": components}),
        sequence=sequence,
        created_by="TEST",
        created_at="2026-08-10T00:00:00+00:00",
        updated_at="2026-08-10T00:00:00+00:00",
    )


class _ExactDimensionResolutionStore:
    def __init__(
        self,
        profiles: list[ProfileRevision],
        bundles: list[BundleRevision],
    ) -> None:
        self.profiles = profiles
        self.bundles = bundles
        self.profile_calls: list[dict[str, Any]] = []
        self.bundle_calls: list[dict[str, Any]] = []

    def get_invalidation_epoch(self) -> int:
        return 1

    def get_profile(self, revision_id: str) -> ProfileRevision | None:
        return next(
            (item for item in self.profiles if item.revision_id == revision_id), None
        )

    def get_bundle(self, revision_id: str) -> BundleRevision | None:
        return next(
            (item for item in self.bundles if item.revision_id == revision_id), None
        )

    def list_profiles(self, **filters: Any) -> list[ProfileRevision]:
        self.profile_calls.append(dict(filters))
        assert filters["exact_dimensions"] is True
        values = [
            item
            for item in self.profiles
            if item.family == filters["family"]
            and item.profile_id == filters["profile_id"]
            and item.scope == filters["scope"]
            and item.scope_key == filters["scope_key"]
            and item.tenant_id == filters["tenant_id"]
            and item.environment == filters["environment"]
            and item.status == filters["status"]
        ]
        return sorted(values, key=lambda item: item.sequence, reverse=True)

    def list_bundles(self, **filters: Any) -> list[BundleRevision]:
        self.bundle_calls.append(dict(filters))
        assert filters["exact_dimensions"] is True
        values = [
            item
            for item in self.bundles
            if item.bundle_id == filters["bundle_id"]
            and item.scope == filters["scope"]
            and item.scope_key == filters["scope_key"]
            and item.tenant_id == filters["tenant_id"]
            and item.environment == filters["environment"]
            and item.status == filters["status"]
        ]
        return sorted(values, key=lambda item: item.sequence, reverse=True)

    def verify_events(self, aggregate_type: str, aggregate_id: str) -> dict[str, Any]:
        return {
            "valid": True,
            "issues": [],
            "eventCount": 1,
            "eventHead": f"{aggregate_type}:{aggregate_id}",
        }


@pytest.mark.parametrize(
    ("precedence", "scope", "scope_key", "context"),
    [
        (
            "shared_game_published",
            "shared_game",
            "SAME",
            {"shared_game_id": "SAME", "tenant_id": "TEN-A", "environment": "prod"},
        ),
        (
            "room_published",
            "room",
            "SAME",
            {"room_id": "SAME", "tenant_id": "TEN-A", "environment": "prod"},
        ),
        (
            "room_published",
            "room",
            "NULL-ENV",
            {"room_id": "NULL-ENV", "tenant_id": "TEN-A"},
        ),
        (
            "room_published",
            "room",
            "NULL-TENANT",
            {"room_id": "NULL-TENANT", "environment": "prod"},
        ),
        (
            "user_experiment_published",
            "user_experiment",
            "SAME",
            {
                "user_experiment_key": "SAME",
                "tenant_id": "TEN-A",
                "environment": "prod",
            },
        ),
        (
            "video_profile_published",
            "video_profile",
            "SAME",
            {"video_profile_id": "SAME", "tenant_id": "TEN-A", "environment": "prod"},
        ),
        (
            "tenant_published",
            "tenant",
            "TEN-A",
            {"tenant_id": "TEN-A", "environment": "prod"},
        ),
        ("environment_published", "environment", "prod", {"environment": "prod"}),
        (
            "global_published",
            "global",
            "*",
            {"tenant_id": "TEN-A", "environment": "prod"},
        ),
    ],
)
def test_profile_and_bundle_resolution_preserve_exact_tenant_environment_dimensions(
    root: Path,
    precedence: str,
    scope: str,
    scope_key: str,
    context: dict[str, str],
) -> None:
    expected_tenant = (
        None if scope in {"environment", "global"} else context.get("tenant_id")
    )
    expected_environment = context.get("environment")
    wrong_environment = "staging" if expected_environment is not None else "prod"
    profiles = [
        _profile_revision(
            "CPR-WRONG-ENVIRONMENT",
            scope=scope,
            scope_key=scope_key,
            tenant_id=expected_tenant,
            environment=wrong_environment,
            sequence=30,
        ),
        _profile_revision(
            "CPR-EXPECTED",
            scope=scope,
            scope_key=scope_key,
            tenant_id=expected_tenant,
            environment=expected_environment,
            sequence=10,
        ),
    ]
    bundles = [
        _bundle_revision(
            "CBR-WRONG-ENVIRONMENT",
            scope=scope,
            scope_key=scope_key,
            tenant_id=expected_tenant,
            environment=wrong_environment,
            sequence=30,
        ),
        _bundle_revision(
            "CBR-EXPECTED",
            scope=scope,
            scope_key=scope_key,
            tenant_id=expected_tenant,
            environment=expected_environment,
            sequence=10,
        ),
    ]
    if scope not in {"environment", "global"}:
        profiles.insert(
            0,
            _profile_revision(
                "CPR-WRONG-TENANT",
                scope=scope,
                scope_key=scope_key,
                tenant_id="TEN-B",
                environment=expected_environment,
                sequence=40,
            ),
        )
        bundles.insert(
            0,
            _bundle_revision(
                "CBR-WRONG-TENANT",
                scope=scope,
                scope_key=scope_key,
                tenant_id="TEN-B",
                environment=expected_environment,
                sequence=40,
            ),
        )
    store = _ExactDimensionResolutionStore(profiles, bundles)
    service = VersionedConfigurationService(
        store,
        root=root,
        profile=_minimal_control_plane(precedence),
    )

    profile = service.resolve_profile(
        family="performance", profile_id="isolation", **context
    )
    bundle = service.resolve(bundle_id="isolation-bundle", **context)

    assert profile["revisionId"] == "CPR-EXPECTED"
    assert bundle["bundle"]["revisionId"] == "CBR-EXPECTED"
    assert store.profile_calls[-1]["tenant_id"] == expected_tenant
    assert store.profile_calls[-1]["environment"] == expected_environment
    assert store.bundle_calls[-1]["tenant_id"] == expected_tenant
    assert store.bundle_calls[-1]["environment"] == expected_environment


def test_exact_bundle_revision_lookup_never_falls_through_to_context_resolution(
    root: Path,
) -> None:
    store = _ExactDimensionResolutionStore(
        [],
        [
            _bundle_revision(
                "CBR-OTHER",
                scope="global",
                scope_key="*",
                tenant_id=None,
                environment=None,
                sequence=1,
            )
        ],
    )
    service = VersionedConfigurationService(
        store,
        root=root,
        profile=_minimal_control_plane("global_published"),
    )

    with pytest.raises(KeyError, match="CBR-MISSING"):
        service.resolve(revision_id="CBR-MISSING", bundle_id="isolation-bundle")
    with pytest.raises(KeyError, match="CBR-OTHER"):
        service.resolve(revision_id="CBR-OTHER", bundle_id="different-bundle")

    assert store.bundle_calls == []


class _AtomicPublicationStore:
    def __init__(self) -> None:
        self.profile = _profile_revision(
            "CPR-DRAFT",
            scope="global",
            scope_key="*",
            tenant_id=None,
            environment=None,
            sequence=1,
        )
        object.__setattr__(self.profile, "status", "draft")
        self.bundle = _bundle_revision(
            "CBR-DRAFT",
            scope="global",
            scope_key="*",
            tenant_id=None,
            environment=None,
            sequence=1,
        )
        object.__setattr__(self.bundle, "status", "draft")
        self.calls: list[str] = []

    def get_invalidation_epoch(self) -> int:
        return len(self.calls)

    def get_profile(self, revision_id: str) -> ProfileRevision | None:
        return self.profile if revision_id == self.profile.revision_id else None

    def get_bundle(self, revision_id: str) -> BundleRevision | None:
        return self.bundle if revision_id == self.bundle.revision_id else None

    def publish_profile(self, revision_id: str, **_: Any) -> ProfileRevision:
        self.calls.append("publish_profile")
        object.__setattr__(self.profile, "status", "published")
        return self.profile

    def publish_bundle(self, revision_id: str, **_: Any) -> BundleRevision:
        self.calls.append("publish_bundle")
        object.__setattr__(self.bundle, "status", "published")
        return self.bundle

    def set_profile_status(self, *_: Any, **__: Any) -> None:
        raise AssertionError("profile publication must be one atomic store call")

    def set_bundle_status(self, *_: Any, **__: Any) -> None:
        raise AssertionError("bundle publication must be one atomic store call")

    def supersede_published_profiles(self, *_: Any, **__: Any) -> None:
        raise AssertionError("profile supersession must not be a second service call")

    def supersede_published_bundles(self, *_: Any, **__: Any) -> None:
        raise AssertionError("bundle supersession must not be a second service call")


def test_service_delegates_publication_and_supersession_to_one_store_call(
    root: Path,
) -> None:
    store = _AtomicPublicationStore()
    service = VersionedConfigurationService(
        store,
        root=root,
        profile=_minimal_control_plane("global_published"),
    )
    service.validate_profile = lambda *_: {}  # type: ignore[method-assign]
    service.compile_bundle = lambda *_: {  # type: ignore[method-assign]
        "publishable": True
    }

    assert (
        service.publish_profile("CPR-DRAFT", expected_etag=1, actor_id="TEST")["status"]
        == "published"
    )
    assert (
        service.publish_bundle("CBR-DRAFT", expected_etag=1, actor_id="TEST")["status"]
        == "published"
    )
    assert store.calls == ["publish_profile", "publish_bundle"]


class _ConcurrentEventSnapshotStore:
    def __init__(self) -> None:
        self.items = [{"sequence": 1, "eventHash": "one"}]

    def get_profile(self, revision_id: str) -> object:
        return object()

    def get_bundle(self, revision_id: str) -> object:
        return object()

    def events_with_verification(
        self, aggregate_type: str, aggregate_id: str
    ) -> dict[str, Any]:
        snapshot = copy.deepcopy(self.items)
        self.items.append({"sequence": 2, "eventHash": "two"})
        return {
            "items": snapshot,
            "verification": {
                "valid": True,
                "issues": [],
                "eventCount": len(snapshot),
                "eventHead": snapshot[-1]["eventHash"],
            },
        }

    def events(self, *_: Any, **__: Any) -> None:
        raise AssertionError("event route must not perform a separate item read")

    def verify_events(self, *_: Any, **__: Any) -> None:
        raise AssertionError("event route must not perform a second snapshot read")


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/configuration-registry/profiles/CPR-ONE/events",
        "/api/v1/configuration-registry/bundles/CBR-ONE/events",
    ],
)
def test_event_route_projects_items_and_verification_from_one_snapshot(
    tmp_path: Path, path: str
) -> None:
    store = _ConcurrentEventSnapshotStore()
    service = SimpleNamespace(store=store)
    identities = IdentityService(
        SQLiteIdentityStore(tmp_path / "events-identity.sqlite3")
    )
    app = FastAPI()

    @app.middleware("http")
    async def mark_api_key(request: Request, call_next):
        request.state.api_key_valid = True
        return await call_next(request)

    app.include_router(
        build_configuration_router(
            service,  # type: ignore[arg-type]
            identities,
            AuthorizationPolicy(),
            {"id": "test"},
            {"id": "test"},
        )
    )
    response = TestClient(app).get(
        path,
        headers={
            "x-api-key": "test",
            "x-baloot-role": "admin",
            "x-baloot-subject-id": "TEST",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["items"] == [{"sequence": 1, "eventHash": "one"}]
    assert response.json()["verification"] == {
        "valid": True,
        "issues": [],
        "eventCount": 1,
        "eventHead": "one",
    }
    assert len(store.items) == 2


class _BootstrapIdentityGuard:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.seen: list[DatabaseIdentityContext] = []
        self.bundle_paths: list[str | None] = []

    def bootstrap_packaged_bundle(
        self, *, actor_id: str, bundle_path: str | None = None, **_: Any
    ) -> dict[str, Any]:
        context = CURRENT_DATABASE_IDENTITY.get()
        self.seen.append(context)
        self.bundle_paths.append(bundle_path)
        if context.role != "service" or not context.subject_id.startswith("SVC-"):
            raise PermissionError("anonymous configuration bootstrap denied")
        assert actor_id == context.subject_id
        if self.fail:
            raise RuntimeError("bootstrap failed")
        return {"bundle": {"revisionId": "CBR-BOOTSTRAP"}}


def test_configuration_bootstrap_uses_service_identity_and_restores_caller() -> None:
    guard = _BootstrapIdentityGuard()
    caller = DatabaseIdentityContext(subject_id="CALLER", role="admin")
    caller_token = CURRENT_DATABASE_IDENTITY.set(caller)
    try:
        anonymous_token = CURRENT_DATABASE_IDENTITY.set(DatabaseIdentityContext())
        try:
            with pytest.raises(PermissionError, match="anonymous"):
                guard.bootstrap_packaged_bundle(actor_id="ANONYMOUS")
        finally:
            CURRENT_DATABASE_IDENTITY.reset(anonymous_token)

        result = container_module._bootstrap_packaged_configuration(guard)  # type: ignore[arg-type]

        assert result["bundle"]["revisionId"] == "CBR-BOOTSTRAP"
        assert guard.bundle_paths == [
            None,  # The denied anonymous attempt is retained by the guard.
            None,
            container_module._NATIVE_RESEARCH_BUNDLE,
            container_module._INTELLIGENT_RESEARCH_BUNDLE,
        ]
        assert guard.seen[-1].role == "service"
        assert CURRENT_DATABASE_IDENTITY.get() == caller
    finally:
        CURRENT_DATABASE_IDENTITY.reset(caller_token)


def test_configuration_bootstrap_failure_also_restores_caller_identity() -> None:
    guard = _BootstrapIdentityGuard(fail=True)
    caller = DatabaseIdentityContext(subject_id="CALLER", role="admin")
    token = CURRENT_DATABASE_IDENTITY.set(caller)
    try:
        with pytest.raises(RuntimeError, match="bootstrap failed"):
            container_module._bootstrap_packaged_configuration(guard)  # type: ignore[arg-type]
        assert CURRENT_DATABASE_IDENTITY.get() == caller
    finally:
        CURRENT_DATABASE_IDENTITY.reset(token)


def test_build_container_failure_closes_store_and_worker_threads(monkeypatch) -> None:
    worker_threads: list[Any] = []

    class CloseProbe:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            assert all(not thread.is_alive() for thread in worker_threads)
            self.closed += 1

    class PoolProbe:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            assert all(not thread.is_alive() for thread in worker_threads)
            self.closed += 1

    store = CloseProbe()
    pool = PoolProbe()
    pool_store = SimpleNamespace(pool=pool)

    def fail_construction(*, owned_resources: list[Any], **_: Any) -> None:
        container_module._own(owned_resources, store)
        container_module._own(owned_resources, pool_store)
        executor = container_module._own(
            owned_resources,
            ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="config-build-failure"
            ),
        )
        executor.submit(lambda: None).result()
        worker_threads.extend(executor._threads)
        raise RuntimeError("construction failed")

    monkeypatch.setattr(container_module, "_build_container", fail_construction)

    with pytest.raises(RuntimeError, match="construction failed"):
        container_module.build_container()

    assert store.closed == 1
    assert pool.closed == 1
    assert worker_threads
    assert all(not thread.is_alive() for thread in worker_threads)
