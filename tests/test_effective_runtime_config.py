from __future__ import annotations

import ast
import base64
import copy
import gzip
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
import yaml

from baloot_platform import config as config_module
from baloot_platform.config import (
    PROFILE_FAMILIES,
    build_bundle,
    profile_bundle_from_resolved,
    validate_profile_payload,
)
from baloot_platform.configuration import (
    EffectiveRuntimeConfigV2,
    SQLiteConfigurationStore,
    VersionedConfigurationService,
    migrate_effective_runtime_config_v1,
)
from baloot_platform.service.container import build_container
from baloot_platform.core.enums import Seat


def test_prepared_profile_references_preserve_all_family_validation(root: Path) -> None:
    bundle = build_bundle(root)
    for family in PROFILE_FAMILIES:
        schema_path = root / "schemas/profiles" / f"{family}.schema.json"
        if not schema_path.exists():
            schema_path = root / "schemas/profile.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        reference = jsonschema.Draft202012Validator(schema)
        for payload in (bundle.components[family].data, {"profile": False}):
            original = copy.deepcopy(payload)
            try:
                reference.validate(payload)
            except jsonschema.ValidationError as exc:
                if exc.context:
                    exc = jsonschema.exceptions.best_match(exc.context)
                dotted = ".".join(str(part) for part in exc.absolute_path)
                with pytest.raises(ValueError) as actual:
                    config_module._validate_profile(payload, schema, schema_path)
                assert str(actual.value) == f"invalid profile {schema_path}: {dotted or '<root>'}: {exc.message}"
            else:
                config_module._validate_profile(payload, schema, schema_path)
            assert payload == original


def test_prepared_profile_refs_revalidate_payload_semantics_and_changed_schema(tmp_path: Path, monkeypatch) -> None:
    schema_path = tmp_path / "schemas/profiles/analysis.schema.json"
    schema_path.parent.mkdir(parents=True)
    schema = {"$schema": jsonschema.Draft202012Validator.META_SCHEMA["$id"],
        "$id": "https://baloot.test/analysis.schema.json", "type": "object", "additionalProperties": False,
        "required": ["limit"], "properties": {"limit": {"$ref": "#limit"}},
        "$defs": {"limit": {"$anchor": "limit", "type": "integer", "minimum": 1}}}
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    config_module._profile_schema_registry.cache_clear()
    crawls, semantics = [], []
    crawl = config_module.Registry.crawl
    def counted(registry):
        crawls.append(True)
        return crawl(registry)
    monkeypatch.setattr(config_module.Registry, "crawl", counted)
    monkeypatch.setattr(config_module, "platform_root", lambda explicit: tmp_path)
    monkeypatch.setattr(config_module, "_semantic_validate", lambda family, data: semantics.append((family, data)))
    payload = {"limit": 1}
    for _ in range(2):
        assert validate_profile_payload("analysis", payload, root=tmp_path) == payload
    assert len(crawls) == 1 and len(semantics) == 2
    for bad in ({"limit": False}, {"limit": 1, "unexpected": True}):
        with pytest.raises(ValueError):
            validate_profile_payload("analysis", bad, root=tmp_path)
    schema["$defs"]["limit"]["minimum"] = 2
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_profile_payload("analysis", payload, root=tmp_path)
    assert validate_profile_payload("analysis", {"limit": 2}, root=tmp_path) == {"limit": 2}
    assert len(crawls) == 2 and len(semantics) == 3
    assert payload == {"limit": 1}
    ordered = {"$schema": schema["$schema"], "$id": schema["$id"], "$ref": "#node",
        "$defs": {"node": {"$anchor": "node", "type": "object",
            "properties": {"z": {"type": "integer"}, "a": {"type": "integer"}}}}}
    with pytest.raises(ValueError, match=r": z: .*integer"):
        config_module._validate_profile({"z": "x", "a": "x"}, ordered, schema_path)


def test_profile_reference_preparation_falls_back_for_other_reference_semantics() -> None:
    base = {"$schema": jsonschema.Draft202012Validator.META_SCHEMA["$id"],
            "$id": "https://baloot.test/profile.schema.json", "type": "object"}
    schemas = [True, {"type": "object"}, {**base, "$id": "relative.json"},
        {**base, "$id": base["$id"] + "#anchor"}, {**base, "$schema": "http://json-schema.org/draft-07/schema#"},
        {**base, "$ref": "https://example.invalid/remote.json"}, {**base, "$dynamicRef": "#node"},
        {**base, "$defs": {"other": {"$id": "child.json"}}}]
    for schema in schemas:
        assert config_module._profile_schema_registry(config_module._canonical(schema)) is None
    info = config_module._profile_schema_registry.cache_info()
    assert info.maxsize == len(PROFILE_FAMILIES) and info.currsize <= info.maxsize


def _service(root: Path, tmp_path: Path) -> VersionedConfigurationService:
    profile = yaml.safe_load(
        (root / "configs/configuration/control-plane.yaml").read_text(encoding="utf-8")
    )
    return VersionedConfigurationService(
        SQLiteConfigurationStore(tmp_path / "configuration.sqlite3"),
        root=root,
        profile=profile,
    )


def _validate(root: Path, value: dict) -> None:
    schema = json.loads(
        (root / "schemas/contracts/effective-runtime-config-v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(value)


def test_exact_historical_product_copy_alias_is_read_only_compatible(
    root: Path,
) -> None:
    historical = json.loads(
        gzip.decompress(
            base64.b64decode(
                (root / "tests/fixtures/historical-product-38b2eca1.json.gz.b64")
                .read_bytes()
            )
        )
    )

    admitted = validate_profile_payload(
        "product",
        historical,
        root=root,
        source="<configuration:product>",
    )
    assert admitted == historical

    with pytest.raises(ValueError, match="invalid profile"):
        validate_profile_payload(
            "product",
            historical,
            root=root,
            source="<row:product>",
        )
    drifted = copy.deepcopy(historical)
    drifted["foundry_copy"]["groups"]["workflow"]["research_deal_balance_enabled"] += (
        "!"
    )
    with pytest.raises(ValueError, match="invalid profile"):
        validate_profile_payload(
            "product",
            drifted,
            root=root,
            source="<configuration:product>",
        )


def test_exact_historical_three_signal_conventions_are_read_only_compatible(
    root: Path,
) -> None:
    historical = copy.deepcopy(build_bundle(root).component("conventions"))
    historical["signals"].pop("minimum_winner")
    historical["signals"].pop("high_expenditure")

    admitted = validate_profile_payload(
        "conventions",
        historical,
        root=root,
        source="<configuration:conventions>",
    )
    assert admitted == historical

    with pytest.raises(ValueError, match="invalid profile"):
        validate_profile_payload(
            "conventions",
            historical,
            root=root,
            source="<row:conventions>",
        )
    drifted = copy.deepcopy(historical)
    drifted["signals"]["partner_return"]["enabled"] = False
    with pytest.raises(ValueError, match="invalid profile"):
        validate_profile_payload(
            "conventions",
            drifted,
            root=root,
            source="<configuration:conventions>",
        )


def test_current_conventions_bootstrap_keeps_complete_signal_inventory(
    root: Path,
    tmp_path: Path,
) -> None:
    service = _service(root, tmp_path)
    resolved = service.bootstrap_packaged_bundle(actor_id="TEST")

    assert set(resolved["profiles"]["conventions"]["signals"]) == {
        "high_expenditure",
        "high_then_low",
        "low_then_high",
        "minimum_winner",
        "partner_return",
    }
    profile_bundle_from_resolved(resolved, root=root)

    incomplete = copy.deepcopy(resolved["profiles"]["conventions"])
    incomplete["signals"].pop("minimum_winner")
    with pytest.raises(ValueError, match="minimum_winner"):
        validate_profile_payload(
            "conventions",
            incomplete,
            root=root,
            source="<configuration:conventions>",
        )


def test_effective_runtime_config_v2_is_canonical_and_schema_valid(
    root: Path, tmp_path: Path
) -> None:
    service = _service(root, tmp_path)
    baseline = service.bootstrap_packaged_bundle(actor_id="TEST")
    sha = "a" * 64
    first = service.resolve_effective_v2(
        revision_id=baseline["bundle"]["revisionId"],
        plugin_identities=(
            {"id": "z-plugin", "version": "1.0.0", "artifactSha256": sha},
            {"id": "a-plugin", "version": "1.0.0", "artifactSha256": sha},
        ),
        model_identities=({"id": "policy", "version": "1.0.0", "configSha256": sha},),
    )
    second = service.resolve_effective_v2(
        revision_id=baseline["bundle"]["revisionId"],
        plugin_identities=(
            {"id": "a-plugin", "version": "1.0.0", "artifactSha256": sha},
            {"id": "z-plugin", "version": "1.0.0", "artifactSha256": sha},
        ),
        model_identities=({"id": "policy", "version": "1.0.0", "configSha256": sha},),
    )

    assert first == second
    assert first.verify() is True
    value = first.to_dict()
    assert value["schemaVersion"] == 2
    assert value["contentSha256"] == first.content_sha256
    assert [item["id"] for item in value["pluginIdentities"]] == [
        "a-plugin",
        "z-plugin",
    ]
    assert set(value["componentRevisions"]) == set(value["normalizedValues"])
    _validate(root, value)


def test_effective_runtime_hash_ignores_store_audit_provenance(
    root: Path, tmp_path: Path
) -> None:
    stores = (tmp_path / "first", tmp_path / "second")
    for path in stores:
        path.mkdir()
    first_service = _service(root, stores[0])
    second_service = _service(root, stores[1])
    first = first_service.bootstrap_packaged_bundle(actor_id="FIRST-ACTOR")
    second = second_service.bootstrap_packaged_bundle(actor_id="SECOND-ACTOR")

    assert first["bundle"]["createdBy"] != second["bundle"]["createdBy"]
    assert first["effectiveHash"] == second["effectiveHash"]
    assert first_service.resolve_effective_v2(
        revision_id=first["bundle"]["revisionId"]
    ) == second_service.resolve_effective_v2(
        revision_id=second["bundle"]["revisionId"]
    )


def test_row_runtime_bundle_hash_ignores_store_specific_provenance(
    root: Path, tmp_path: Path
) -> None:
    service = _service(root, tmp_path)
    resolved = service.bootstrap_packaged_bundle(actor_id="TEST")
    packaged = build_bundle(root)
    equivalent = copy.deepcopy(resolved)
    equivalent["bundle"]["revisionId"] = "another-store-bundle"
    equivalent["bundle"]["createdAt"] = "2099-01-01T00:00:00Z"
    for family, revision in equivalent["profileRevisions"].items():
        revision["revisionId"] = f"another-store-{family}"

    first = profile_bundle_from_resolved(resolved, root=root)
    second = profile_bundle_from_resolved(equivalent, root=root)
    packaged_profile = dict(packaged.bundle_profile["profile"])
    expected_profile_hashes = {
        family: artifact.sha256 for family, artifact in packaged.components.items()
    }

    assert first.sha256 == second.sha256
    assert first.sha256 != packaged.sha256
    assert first.rules.sha256 == second.rules.sha256
    assert first.component("challenge") == second.component("challenge")
    assert packaged_profile == {
        "id": "baloot_default",
        "version": "21.1.22",
        "status": "governed_candidate",
    }
    assert first.bundle_profile["profile"] == {
        **packaged_profile,
        "status": "published",
    }
    assert {
        "id": resolved["bundle"]["bundleId"],
        "version": resolved["bundle"]["version"],
        "mode": resolved["bundle"]["mode"],
        "status": resolved["bundle"]["status"],
    } == {
        "id": packaged_profile["id"],
        "version": packaged_profile["version"],
        "mode": packaged.mode,
        "status": "published",
    }
    assert len(expected_profile_hashes) == 32
    assert {
        family: artifact.sha256 for family, artifact in first.components.items()
    } == expected_profile_hashes
    assert {
        family: revision["payloadSha256"]
        for family, revision in resolved["profileRevisions"].items()
    } == expected_profile_hashes


def test_historical_profile_content_requires_explicit_bundle_authority(
    root: Path,
    tmp_path: Path,
) -> None:
    service = _service(root, tmp_path)
    baseline = service.bootstrap_packaged_bundle(actor_id="TEST")
    agents = copy.deepcopy(baseline["profiles"]["agents"])
    agents["profiles"]["strong"]["risk_tolerance"] += 0.01
    service.create_profile(
        family="agents",
        payload=agents,
        created_by="TEST",
        publish=True,
        supersedes_revision_id=baseline["profileRevisions"]["agents"]["revisionId"],
    )
    historical_components = dict(baseline["bundle"]["componentRevisions"])
    with pytest.raises(ValueError, match="component revision is not eligible"):
        service.create_bundle(
            bundle_id=baseline["bundle"]["bundleId"],
            version="21.9.0",
            mode="full",
            component_revisions=historical_components,
            created_by="TEST",
            force_new_revision=True,
        )
    restored = service.create_bundle(
        bundle_id=baseline["bundle"]["bundleId"],
        version="21.9.0",
        mode="full",
        component_revisions=historical_components,
        created_by="TEST",
        allow_historical_components=True,
        force_new_revision=True,
    )
    assert restored["status"] == "draft"


def test_container_activates_published_row_bundle_only_after_restart(
    root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BALOOT_RUNTIME", str(tmp_path / "ambient-runtime"))
    monkeypatch.setenv("BALOOT_OUTPUTS", str(tmp_path / "ambient-outputs"))
    first = build_container(root=root, writable_root=tmp_path)
    try:
        assert first.runtime == (tmp_path / "runtime").resolve()
        assert first.outputs == (tmp_path / "outputs").resolve()
        runtime_revision = first.foundry_profile["runtimeBundleRevisionId"]
        baseline = first.configuration.resolve(revision_id=runtime_revision)
        challenge = copy.deepcopy(baseline["profiles"]["challenge"])
        original = float(challenge["tactical_detectors"]["minimumDelayedGain"])
        challenge["tactical_detectors"]["minimumDelayedGain"] = original + 1.0
        revision = first.configuration.create_profile(
            family="challenge",
            payload=challenge,
            created_by="TEST",
            publish=True,
            supersedes_revision_id=baseline["profileRevisions"]["challenge"][
                "revisionId"
            ],
        )
        components = dict(baseline["bundle"]["componentRevisions"])
        components["challenge"] = revision["revisionId"]
        published = first.configuration.create_bundle(
            bundle_id=baseline["bundle"]["bundleId"],
            version="21.0.1",
            mode="full",
            component_revisions=components,
            created_by="TEST",
            supersedes_revision_id=runtime_revision,
            publish=True,
        )
        detector = first.registry.get(
            "tactical_detector", "causal_tactical_detector_v1"
        )
        assert detector.profile.minimum_delayed_gain == original
        assert first.foundry_profile["runtimeBundleRevisionId"] == runtime_revision
    finally:
        first.shutdown()

    second = build_container(root=root, writable_root=tmp_path)
    try:
        detector = second.registry.get(
            "tactical_detector", "causal_tactical_detector_v1"
        )
        assert (
            second.foundry_profile["runtimeBundleRevisionId"] == published["revisionId"]
        )
        assert detector.profile.minimum_delayed_gain == original + 1.0
        assert (
            second.challenge_miner.bundle.sha256 == second.foundry_profile["bundleHash"]
        )
        assert (
            second.games._bundle("full").sha256 == second.foundry_profile["bundleHash"]
        )
        board = second.sakka.create_board_from_new_game(
            created_by="TEST",
            seed=19073,
            dealer=Seat.EAST,
            mode="full",
        )
        assert board.bundle_hash == second.foundry_profile["bundleHash"]
        assert second.sakka.verify_board(board.board_id)["valid"] is True
        renderer = second.registry.get("video_renderer", "premium_video")
        assert (
            renderer.profile
            == second.configuration.resolve(revision_id=published["revisionId"])[
                "profiles"
            ]["video"]
        )
        assert renderer.descriptor.timeout_seconds == 3600
    finally:
        second.shutdown()


def test_effective_runtime_config_v1_adapter_is_read_only_and_exact(
    root: Path, tmp_path: Path
) -> None:
    service = _service(root, tmp_path)
    resolved = service.bootstrap_packaged_bundle(actor_id="TEST")
    frozen_v1 = service.freeze(revision_id=resolved["bundle"]["revisionId"])

    migrated = migrate_effective_runtime_config_v1(frozen_v1, resolved)
    direct = service.resolve_effective_v2(revision_id=resolved["bundle"]["revisionId"])
    assert migrated == direct

    mismatched = copy.deepcopy(frozen_v1)
    mismatched["effectiveHash"] = "0" * 64
    with pytest.raises(ValueError, match="effectiveHash"):
        migrate_effective_runtime_config_v1(mismatched, resolved)
    assert service.freeze(revision_id=resolved["bundle"]["revisionId"]) == frozen_v1


def test_effective_runtime_config_v2_preserves_existing_scope_precedence(
    root: Path, tmp_path: Path
) -> None:
    service = _service(root, tmp_path)
    baseline = service.bootstrap_packaged_bundle(actor_id="TEST")
    bundle = baseline["bundle"]
    shared = service.create_bundle(
        bundle_id=bundle["bundleId"],
        version="20.1.0",
        mode="full",
        component_revisions=dict(bundle["componentRevisions"]),
        created_by="TEST",
        scope="shared_game",
        scope_key="BOARD-V21",
        publish=True,
    )

    effective = service.freeze_v2(
        bundle_id=bundle["bundleId"],
        shared_game_id="BOARD-V21",
    )
    assert effective["bundle"]["revisionId"] == shared["revisionId"]
    assert effective["resolutionTrace"][0]["scope"] == "shared_game"
    _validate(root, effective)


def test_effective_runtime_config_v2_rejects_ambiguous_identity_keys(
    root: Path, tmp_path: Path
) -> None:
    service = _service(root, tmp_path)
    baseline = service.bootstrap_packaged_bundle(actor_id="TEST")
    with pytest.raises(ValueError, match="unknown keys"):
        service.resolve_effective_v2(
            revision_id=baseline["bundle"]["revisionId"],
            plugin_identities=(
                {"id": "plugin", "version": "1.0.0", "ambientPath": "C:/host"},
            ),
        )


def test_effective_runtime_config_v2_snapshot_parser_rejects_all_tamper(
    root: Path, tmp_path: Path
) -> None:
    service = _service(root, tmp_path)
    baseline = service.bootstrap_packaged_bundle(actor_id="TEST")
    effective = service.resolve_effective_v2(
        revision_id=baseline["bundle"]["revisionId"]
    )
    canonical_bytes = json.dumps(
        effective.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert EffectiveRuntimeConfigV2.from_bytes(canonical_bytes) == effective

    noncanonical = json.dumps(effective.to_dict(), indent=2).encode("utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        EffectiveRuntimeConfigV2.from_bytes(noncanonical)

    unknown = effective.to_dict()
    unknown["ambientFallback"] = True
    with pytest.raises(ValueError, match="missing or unknown keys"):
        EffectiveRuntimeConfigV2.from_dict(unknown)

    payload_tamper = effective.to_dict()
    family = next(iter(payload_tamper["normalizedValues"]))
    payload_tamper["normalizedValues"][family]["tampered"] = True
    with pytest.raises(ValueError, match=f"{family} payload checksum mismatch"):
        EffectiveRuntimeConfigV2.from_dict(payload_tamper)

    content_tamper = effective.to_dict()
    content_tamper["contentSha256"] = "0" * 64
    with pytest.raises(ValueError, match="content checksum mismatch"):
        EffectiveRuntimeConfigV2.from_dict(content_tamper)


def test_control_plane_has_no_private_cache_defaults(
    root: Path, tmp_path: Path
) -> None:
    profile = yaml.safe_load(
        (root / "configs/configuration/control-plane.yaml").read_text(encoding="utf-8")
    )
    del profile["resolution"]["cache_entries"]
    with pytest.raises(ValueError, match="cache_entries"):
        VersionedConfigurationService(
            SQLiteConfigurationStore(tmp_path / "configuration.sqlite3"),
            root=root,
            profile=profile,
        )


def test_scope_precedence_is_driven_by_the_validated_profile(
    root: Path, tmp_path: Path
) -> None:
    profile = yaml.safe_load(
        (root / "configs/configuration/control-plane.yaml").read_text(encoding="utf-8")
    )
    precedence = profile["resolution"]["precedence"]
    precedence.remove("tenant_published")
    precedence.insert(precedence.index("shared_game_published"), "tenant_published")
    service = VersionedConfigurationService(
        SQLiteConfigurationStore(tmp_path / "configuration.sqlite3"),
        root=root,
        profile=profile,
    )
    baseline = service.bootstrap_packaged_bundle(actor_id="TEST")
    bundle = baseline["bundle"]
    components = dict(bundle["componentRevisions"])
    tenant = service.create_bundle(
        bundle_id=bundle["bundleId"],
        version="20.2.0",
        mode="full",
        component_revisions=components,
        created_by="TEST",
        scope="tenant",
        tenant_id="TENANT-V21",
        publish=True,
    )
    service.create_bundle(
        bundle_id=bundle["bundleId"],
        version="20.2.1",
        mode="full",
        component_revisions=components,
        created_by="TEST",
        scope="shared_game",
        scope_key="BOARD-V21",
        tenant_id="TENANT-V21",
        publish=True,
    )

    resolved = service.resolve(
        bundle_id=bundle["bundleId"],
        tenant_id="TENANT-V21",
        shared_game_id="BOARD-V21",
    )
    assert resolved["bundle"]["revisionId"] == tenant["revisionId"]
    assert resolved["resolutionTrace"][0]["scope"] == "tenant"


def test_active_constructor_defaults_have_machine_readable_classification(
    root: Path,
) -> None:
    inventory = json.loads(
        (root / "docs/architecture/hard-code-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    entries = {entry["key"]: entry for entry in inventory["entries"]}
    detected: dict[str, str] = {}
    source_root = root / "src/baloot_platform"
    for package in ("configuration", "core", "analysis", "video", "plugins"):
        for path in sorted((source_root / package).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for class_node in (
                node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            ):
                constructors = (
                    node
                    for node in class_node.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "__init__"
                )
                for constructor in constructors:
                    positional = constructor.args.posonlyargs + constructor.args.args
                    defaults = (
                        list(
                            zip(
                                positional[-len(constructor.args.defaults) :],
                                constructor.args.defaults,
                            )
                        )
                        if constructor.args.defaults
                        else []
                    )
                    defaults.extend(
                        (argument, default)
                        for argument, default in zip(
                            constructor.args.kwonlyargs,
                            constructor.args.kw_defaults,
                        )
                        if default is not None
                    )
                    for argument, default in defaults:
                        relative = path.relative_to(root).as_posix()
                        key = f"{relative}:{class_node.name}.{argument.arg}"
                        detected[key] = ast.unparse(default)

    assert set(entries) == set(detected)
    allowed = set(inventory["allowedClassifications"])
    for key, default in detected.items():
        entry = entries[key]
        assert entry["default"] == default
        assert entry["classification"] in allowed
        assert entry["status"] in {"compliant", "migration_required"}
        assert entry["owner"] and entry["rationale"]


def test_bundle_rollback_creates_a_new_revision_and_preserves_frozen_history(
    root: Path, tmp_path: Path
) -> None:
    service = _service(root, tmp_path)
    baseline = service.bootstrap_packaged_bundle(actor_id="TEST")
    baseline_bundle = baseline["bundle"]
    baseline_components = dict(baseline_bundle["componentRevisions"])
    performance = copy.deepcopy(baseline["profiles"]["performance"])
    performance["profile"]["version"] = "20.0.1"
    performance["budgets"]["configuration_resolve_p95_ms"] += 1
    updated_profile = service.create_profile(
        family="performance",
        payload=performance,
        created_by="TEST",
        publish=True,
    )
    updated_components = dict(baseline_components)
    updated_components["performance"] = updated_profile["revisionId"]
    current = service.create_bundle(
        bundle_id=baseline_bundle["bundleId"],
        version="20.0.1",
        mode="full",
        component_revisions=updated_components,
        created_by="TEST",
        publish=True,
    )

    historical = service.resolve(revision_id=baseline_bundle["revisionId"])
    assert (
        historical["profileRevisions"]["performance"]["revisionId"]
        == baseline_components["performance"]
    )
    rollback = service.rollback_bundle(
        baseline_bundle["revisionId"],
        new_version="20.0.2",
        actor_id="TEST",
    )
    assert rollback["revisionId"] not in {
        baseline_bundle["revisionId"],
        current["revisionId"],
    }
    assert rollback["componentRevisions"] == baseline_components
    assert rollback["supersedesRevisionId"] == current["revisionId"]
    resolved = service.freeze_v2(revision_id=rollback["revisionId"])
    assert resolved["bundle"]["revisionId"] == rollback["revisionId"]
    events = service.store.events("bundle", rollback["revisionId"])
    assert events[-1]["eventType"] == "rollback_created"


def test_rollback_migration_preserves_content_hash_as_a_non_unique_lookup(
    root: Path,
) -> None:
    source = root / "migrations/013_v21_configuration_rollback.sql"
    packaged = root / "src/baloot_platform/resources/migrations" / source.name
    sql = source.read_text(encoding="utf-8")
    assert "DROP INDEX IF EXISTS configuration.uq_configuration_bundle_content" in sql
    assert "CREATE INDEX IF NOT EXISTS ix_configuration_bundle_content_v21" in sql
    assert "CREATE UNIQUE INDEX" not in sql
    assert packaged.read_bytes() == source.read_bytes()


def test_v2_contract_uses_existing_rest_grpc_and_mcp_configuration_authority(
    root: Path,
) -> None:
    routes = (root / "src/baloot_platform/service/routes/configuration.py").read_text(
        encoding="utf-8"
    )
    mcp = (root / "src/baloot_platform/service/mcp_server.py").read_text(
        encoding="utf-8"
    )
    proto = (root / "proto/baloot/v19/configuration.proto").read_text(encoding="utf-8")
    grpc_service = (root / "src/baloot_platform/service/grpc_service.py").read_text(
        encoding="utf-8"
    )
    assert '@router.get("/resolve-v2")' in routes
    assert '@router.get("/bundles/{revision_id}:resolve-v2")' in routes
    assert '@router.post("/bundles/{revision_id}:rollback"' in routes
    assert '"resolve_effective_runtime_config_v2"' in mcp
    assert '"rollback_configuration_bundle"' in mcp
    assert "rpc ResolveEffectiveRuntimeConfigV2" in proto
    assert "rpc RollbackBundle" in proto
    assert "async def ResolveEffectiveRuntimeConfigV2" in grpc_service
    assert "async def RollbackBundle" in grpc_service


def test_v1_deprecation_is_bounded_read_only_and_has_a_removal_gate(
    root: Path,
) -> None:
    policy = json.loads(
        (
            root / "docs/architecture/effective-runtime-config-v1-deprecation.json"
        ).read_text(encoding="utf-8")
    )
    assert policy["status"] == "read_compatibility_only"
    assert policy["adapterMode"] == "read_only_no_dual_write"
    assert policy["removalGate"]["earliestMajor"] > 21
    assert len(policy["removalGate"]["requires"]) >= 4


def test_analysis_and_plugin_tuning_are_profile_owned_and_fail_closed(
    root: Path,
) -> None:
    import inspect

    import yaml

    from baloot_platform.analysis.match_value_v7 import MatchValueConfig
    from baloot_platform.core.match_scenario import MatchScenarioRequest
    from baloot_platform.core.scenario import ScenarioRequest
    from baloot_platform.plugins.host import PluginHostPolicy
    from baloot_platform.plugins.registry import PluginPolicy

    assert all(
        field.default is inspect.Parameter.empty
        for field in inspect.signature(MatchValueConfig).parameters.values()
    )
    assert all(
        field.default is inspect.Parameter.empty
        for field in inspect.signature(PluginHostPolicy).parameters.values()
    )
    assert all(
        field.default is inspect.Parameter.empty
        for field in inspect.signature(PluginPolicy).parameters.values()
    )
    assert all(
        field.default is inspect.Parameter.empty
        for field in inspect.signature(ScenarioRequest).parameters.values()
    )
    assert all(
        field.default is inspect.Parameter.empty
        for field in inspect.signature(MatchScenarioRequest).parameters.values()
    )
    match_profile = yaml.safe_load(
        (root / "configs/match_value/exact_dynamic_v2.yaml").read_text(encoding="utf-8")
    )
    MatchValueConfig.from_profile(match_profile)
    match_profile["future_rounds"]["unexpected"] = 1
    with pytest.raises(ValueError, match="keys mismatch"):
        MatchValueConfig.from_profile(match_profile)

    runtime = yaml.safe_load(
        (root / "configs/runtime/docker_v1.yaml").read_text(encoding="utf-8")
    )
    assert runtime["workers"]["video_concurrency"] == 1
    assert runtime["workers"]["video_render_slots"] == 6
    PluginPolicy.from_profile(runtime["plugin_host"])
    runtime["plugin_host"]["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown plugin policy keys"):
        PluginPolicy.from_profile(runtime["plugin_host"])
    scenario = ScenarioRequest.governed(seed=123, contract="sun", require_project=False)
    assert scenario.seed == 123 and scenario.contract.value == "sun"
    match = MatchScenarioRequest.governed(maximum_rounds=3)
    assert match.maximum_rounds == 3
    with pytest.raises(ValueError, match="unknown scenario request keys"):
        ScenarioRequest.governed(unexpected=True)


def test_dataclass_default_inventory_is_current_and_explicit(
    root: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/generate_default_inventory.py", "--check"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    inventory = json.loads(
        (root / "docs/architecture/dataclass-default-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    assert inventory["summary"]["total"] == len(inventory["entries"])
    assert inventory["summary"]["migrationRequired"] == 0
    assert all(
        item["classification"]
        in {
            "protocol_or_mathematical_invariant",
            "safety_bound",
            "governed_tuning",
        }
        and item["status"] in {"compliant", "migration_required"}
        and item["owner"]
        and item["rationale"]
        for item in inventory["entries"]
    )


def test_rule_traceability_matrix_covers_v21_normative_areas(
    root: Path,
) -> None:
    matrix = json.loads(
        (root / "docs/rules/rule-traceability.json").read_text(encoding="utf-8")
    )
    required_prefixes = {
        "deal",
        "bid",
        "escalation",
        "play",
        "projects",
        "score",
        "match",
        "invalid",
        "variants",
    }
    actual = {item["id"].split(".", 1)[0] for item in matrix["clauses"]}
    assert required_prefixes <= actual
    assert matrix["certificationStatus"] == "candidate_external_review_required"
    assert all(item["profilePaths"] and item["evidence"] for item in matrix["clauses"])
    assert all(
        (root / evidence).is_file()
        for item in matrix["clauses"]
        for evidence in item["evidence"]
    )
    rules = yaml.safe_load(
        (root / "configs/rules/saudi_common_v1.yaml").read_text(encoding="utf-8")
    )
    for item in matrix["clauses"]:
        for profile_path in item["profilePaths"]:
            value = rules
            for segment in profile_path.split("."):
                assert segment in value, (
                    f"{item['id']} references missing rule path {profile_path}"
                )
                value = value[segment]
    assert all(
        item["status"] == "existing_executable_evidence" for item in matrix["clauses"]
    )
