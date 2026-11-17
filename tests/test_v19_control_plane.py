from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from baloot_platform.configuration import SQLiteConfigurationStore, VersionedConfigurationService
from baloot_platform.performance import PerformanceService
from baloot_platform.plugins.base import PluginDescriptor
from baloot_platform.plugins.registry import PluginPolicy, PluginRegistry


@dataclass
class DummyPlugin:
    descriptor: PluginDescriptor


def test_v19_row_configuration_lifecycle(root: Path, tmp_path: Path) -> None:
    store = SQLiteConfigurationStore(tmp_path / "configuration.sqlite3")
    profile = yaml.safe_load((root / "configs/configuration/control_plane_v19.yaml").read_text(encoding="utf-8"))
    service = VersionedConfigurationService(store, root=root, profile=profile)
    resolved = service.bootstrap_packaged_bundle(actor_id="TEST")
    assert resolved["bundle"]["status"] == "published"
    assert set(resolved["profiles"]) >= {"rules", "configuration", "performance"}
    assert resolved["effectiveHash"]
    performance = dict(resolved["profiles"]["performance"])
    performance["profile"] = dict(performance["profile"])
    performance["profile"].update({"id": "performance_test_v19", "version": "19.0.1", "status": "draft_candidate"})
    performance["budgets"] = dict(performance["budgets"])
    performance["budgets"]["configuration_resolve_p95_ms"] = float(performance["budgets"]["configuration_resolve_p95_ms"]) + 1
    created = service.create_profile(family="performance", payload=performance, created_by="TEST")
    assert created["status"] == "draft" and created["etag"] == 1
    published = service.publish_profile(created["revisionId"], expected_etag=1, actor_id="TEST")
    assert published["status"] == "published" and published["etag"] == 2
    with pytest.raises(RuntimeError):
        service.publish_profile(created["revisionId"], expected_etag=1, actor_id="TEST")
    resolved_profile = service.resolve_profile(family="performance", profile_id="performance_test_v19")
    assert resolved_profile["revisionId"] == created["revisionId"]
    verification = store.verify_events("profile", created["revisionId"])
    assert verification["valid"] is True and verification["eventCount"] >= 2


def test_v19_plugin_policy_isolates_external_plugins() -> None:
    policy = PluginPolicy.from_profile({
        "host_api_version": "1", "profile_schema_version": "1",
        "allowed_isolation": ["in_process", "process", "wasm_component"],
        "allowed_permissions": ["none"], "require_signed_external": True,
        "require_content_addressed_external": True,
        "maximum_timeout_seconds": 30, "maximum_response_bytes": 1048576,
        "maximum_memory_mb": 512, "maximum_cpu_seconds": 10,
        "maximum_stderr_bytes": 65536,
    })
    registry = PluginRegistry(policy=policy)
    unsafe = DummyPlugin(PluginDescriptor("unsafe", "analysis", "1.0.0", "unsafe", isolation="in_process"))
    with pytest.raises(PermissionError):
        registry.register(unsafe, external=True)
    unsigned = DummyPlugin(PluginDescriptor("unsigned", "analysis", "1.0.0", "unsigned", isolation="wasm_component"))
    with pytest.raises(PermissionError):
        registry.register(unsigned, external=True)
    safe = DummyPlugin(PluginDescriptor(
        "safe-wasm", "analysis", "1.0.0", "safe", isolation="wasm_component",
        artifact_sha256="a" * 64, signed=True, trust_class="external",
    ))
    registry.register(safe, external=True, source="test")
    value = registry.list("analysis")[0]
    assert value["id"] == "safe-wasm" and value["external"] is True


def test_v19_performance_service_reports_budget() -> None:
    service = PerformanceService({
        "profile": {"id": "test"}, "measurement": {"warmup_iterations": 0, "measured_iterations": 3, "percentile": 95},
        "budgets": {"noop": 50.0},
    })
    measurement = service.measure("noop", lambda: 1, iterations=3, warmup=0)
    assert measurement.iterations == 3 and measurement.passed is True
    assert service.report([measurement])["status"] == "passed"


def test_v19_quick_performance_does_not_select_arbitrary_published_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BALOOT_API_KEY", "v19-performance-key")
    from baloot_platform.service import api

    active = api.CONFIGURATION.resolve()
    scoped = api.CONFIGURATION.create_bundle(
        bundle_id="v19-performance-scoped-only",
        version="99.0.0",
        mode=str(active["bundle"]["mode"]),
        component_revisions={
            family: str(value["revisionId"])
            for family, value in active["profileRevisions"].items()
        },
        created_by="TEST",
        scope="tenant",
        tenant_id="V19-PERFORMANCE-TENANT",
        publish=True,
        force_new_revision=True,
    )
    with pytest.raises(KeyError, match="published configuration bundle not found"):
        api.CONFIGURATION.resolve(bundle_id=scoped["bundleId"])
    scoped_row = api.CONFIGURATION.store.get_bundle(scoped["revisionId"])
    assert scoped_row is not None
    list_bundles = api.CONFIGURATION.store.list_bundles

    def scoped_first(**filters):
        if filters == {"status": "published", "limit": 1}:
            return [scoped_row]
        return list_bundles(**filters)

    monkeypatch.setattr(api.CONFIGURATION.store, "list_bundles", scoped_first)

    response = TestClient(api.app).post(
        "/api/v1/performance/quick",
        headers={
            "x-api-key": "v19-performance-key",
            "x-baloot-role": "admin",
            "x-baloot-subject-id": "ADMIN-V19-PERFORMANCE",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] in {"passed", "failed"}


def test_v19_rest_control_plane_and_capability_parity(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BALOOT_API_KEY", "v19-admin-key")
    from baloot_platform.service.api import app
    headers = {"x-api-key": "v19-admin-key", "x-baloot-role": "admin", "x-baloot-subject-id": "ADMIN-V19"}
    client = TestClient(app)
    assert client.get("/healthz").status_code == 200
    profile = client.get("/api/v1/configuration-registry/profile", headers=headers)
    assert profile.status_code == 200 and profile.json()["automaticProductionApply"] is False
    listed = client.get("/api/v1/configuration-registry/profiles?family=performance", headers=headers)
    assert listed.status_code == 200 and listed.json()["items"]
    plugins = client.get("/api/v1/plugins", headers=headers).json()
    assert plugins["externalInProcessAllowed"] is False and plugins["policy"]["requireSignedExternal"] is True
    quick = client.post("/api/v1/performance/quick", headers=headers)
    assert quick.status_code == 200 and quick.json()["status"] in {"passed", "failed"}
    assert client.get("/api/v1/certification/profile", headers=headers).status_code == 200
    assert client.get("/api/v1/benchmark/profile", headers=headers).status_code == 200
    assert client.get("/api/v1/operations/profile", headers=headers).status_code == 200
    capability = client.get("/api/v1/platform/capabilities", headers=headers).json()
    assert capability["requirements"]["every_stable_capability_has_mcp"] is True
    studio = client.get("/configuration-studio", follow_redirects=False)
    assert studio.status_code == 307
    assert studio.headers["location"] == "/foundry?section=settings"
    parity = json.loads((root / "generated/capability-parity-v19.json").read_text(encoding="utf-8"))
    assert parity["status"] == "passed"
    assert all(item["passed"] for item in parity["capabilities"].values())


def test_v19_rejection_retirement_and_tenant_precedence(root: Path, tmp_path: Path) -> None:
    store = SQLiteConfigurationStore(tmp_path / "configuration-lifecycle.sqlite3")
    governance = yaml.safe_load((root / "configs/configuration/control_plane_v19.yaml").read_text(encoding="utf-8"))
    service = VersionedConfigurationService(store, root=root, profile=governance)
    service.bootstrap_packaged_bundle(actor_id="TEST")
    base = service.resolve()["profiles"]["performance"]

    rejected_payload = json.loads(json.dumps(base))
    rejected_payload["profile"] = {"id": "rejected_performance_v19", "version": "19.0.1", "status": "draft_candidate"}
    rejected = service.create_profile(family="performance", payload=rejected_payload, created_by="TEST")
    rejected = service.reject_profile(rejected["revisionId"], expected_etag=1, actor_id="TEST", reason="unsafe regression")
    assert rejected["status"] == "rejected"
    assert store.verify_events("profile", rejected["revisionId"])["valid"] is True

    global_payload = json.loads(json.dumps(base))
    global_payload["profile"] = {"id": "tenant_resolution_v19", "version": "19.1.0", "status": "production_candidate"}
    global_row = service.create_profile(family="performance", payload=global_payload, created_by="TEST", publish=True)
    tenant_payload = json.loads(json.dumps(global_payload))
    tenant_payload["profile"]["version"] = "19.2.0"
    tenant_payload["budgets"]["configuration_resolve_p95_ms"] = 12.0
    tenant_row = service.create_profile(
        family="performance", payload=tenant_payload, created_by="TEST", scope="tenant",
        tenant_id="TENANT-A", publish=True,
    )
    assert service.resolve_profile(family="performance", profile_id="tenant_resolution_v19")["revisionId"] == global_row["revisionId"]
    assert service.resolve_profile(family="performance", profile_id="tenant_resolution_v19", tenant_id="TENANT-A")["revisionId"] == tenant_row["revisionId"]
    changes = service.compare_payloads(global_payload, tenant_payload)
    assert any(item["path"] == "budgets.configuration_resolve_p95_ms" for item in changes)
    retired = service.retire_profile(tenant_row["revisionId"], expected_etag=1, actor_id="TEST", reason="superseded by experiment")
    assert retired["status"] == "retired"


def test_v19_migration_and_protocol_contracts(root: Path) -> None:
    assert not (root / "migrations/011_configuration_control_plane_v19.sql").exists()
    migration = (root / "migrations/011_configuration_registry_v19.sql").read_text(encoding="utf-8")
    assert "configuration.profile_revisions" in migration
    assert "ENABLE ROW LEVEL SECURITY" in migration
    proto = (root / "proto/baloot/v19/configuration.proto").read_text(encoding="utf-8")
    for method in (
        "RejectProfile", "RetireProfile", "RejectBundle", "RetireBundle",
        "RunPerformanceQuick", "GetCapabilityMatrix", "VerifyCertificationCandidate",
        "EvaluateIntelligenceBenchmark", "GetOperationsProfile",
    ):
        assert f"rpc {method}" in proto
    grpc_service = (root / "src/baloot_platform/service/grpc_service.py").read_text(encoding="utf-8")
    assert 'PERFORMANCE.measure("configuration_resolve_p95_ms", CONFIGURATION.resolve,' in grpc_service
    assert 'list_bundles(status="published", limit=1)' not in grpc_service
    from baloot_platform.service.mcp_server import TOOL_DEFINITIONS
    names = {item["name"] for item in TOOL_DEFINITIONS}
    assert {
        "health_check", "reject_configuration_profile", "retire_configuration_profile",
        "reject_configuration_bundle", "retire_configuration_bundle",
        "verify_certification_candidate", "evaluate_intelligence_benchmark",
        "get_operations_profile",
    } <= names


def test_v19_plugin_host_enforces_timeout(tmp_path: Path) -> None:
    import sys
    from baloot_platform.plugins.host import PluginHostPolicy, SubprocessPluginHost
    worker = tmp_path / "slow_plugin.py"
    worker.write_text(
        "import sys,time,json\n"
        "for line in sys.stdin:\n"
        " time.sleep(1.0)\n"
        " req=json.loads(line);print(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{}}),flush=True)\n",
        encoding="utf-8",
    )
    host = SubprocessPluginHost([sys.executable, str(worker)], policy=PluginHostPolicy(
        timeout_seconds=0.1, maximum_response_bytes=1048576,
        maximum_memory_mb=128, maximum_cpu_seconds=2,
        allow_network=False, allow_filesystem=False,
        maximum_stderr_bytes=65536,
    ))
    with pytest.raises(TimeoutError):
        host.call("choose")
    host.close()
