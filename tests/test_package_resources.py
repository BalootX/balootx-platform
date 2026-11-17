from pathlib import Path
from fnmatch import fnmatchcase
import hashlib
import tomllib

import pytest
import yaml


@pytest.mark.parametrize("filename", [
    "gea_baloot_regulation_2023.pdf",
    "saudi_federation_gameplay_2020_archive.html.bin",
])
def test_retained_rule_source_is_packaged(root, filename):
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.core.rule_source_catalog import load_exact_rule_source_catalog
    relative = f"configs/rules/sources/{filename}"
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v9.yaml")
    for base in (root, root / "src/baloot_platform/resources"):
        catalog = load_exact_rule_source_catalog(rules, base)
        assert not (base / relative).exists()
        source = next(item for item in catalog.sources if item["id"] == filename.split(".")[0])
        assert source["snapshot_url"].startswith("https://")
        receipt = next(item for item in catalog.verify_retained_sources(base) if item["sourceId"] == source["id"])
        assert receipt["status"] == "not_retained"
        assert receipt["expectedSha256"] == source["content_sha256"]


@pytest.mark.parametrize("authority", ["research_events_v1", "research_events_v2"])
def test_adjudication_source_receipt_is_packaged_and_hash_bound(root, authority):
    data = yaml.safe_load(
        (root / f"configs/adjudication/{authority}.yaml").read_text(encoding="utf-8")
    )["sourceAuthority"]
    relative = data["verificationReceiptPath"]
    canonical = (root / relative).read_bytes()
    packaged = root / "src/baloot_platform/resources"
    assert (packaged / relative).read_bytes() == canonical
    assert hashlib.sha256(canonical).hexdigest() == data["verificationReceiptSha256"]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = project["tool"]["setuptools"]["package-data"]["baloot_platform"]
    assert any(fnmatchcase(f"resources/{relative}", pattern) for pattern in patterns)
    runtime = (root / "Dockerfile").read_text(encoding="utf-8").split(
        "FROM python-base AS runtime\n", 1
    )[1].split("FROM runtime AS", 1)[0]
    assert relative in runtime and "/app/release/" in runtime


def test_engine_guide_loads_from_packaged_authority(root):
    from baloot_platform.config import build_bundle
    from baloot_platform.guide import build_engine_guide

    packaged = root / "src/baloot_platform/resources"
    bundle_path = "configs/bundle_ortools_research_native_v3_1.yaml"
    bundle = build_bundle(packaged, bundle_path=packaged / bundle_path)
    guide = build_engine_guide(bundle, root=packaged)
    assert guide["audit"]["bundleHash"] == bundle.sha256
    assert guide["adjudicationAuthority"]["sourceAuthority"] == dict(
        bundle.component("adjudication")["sourceAuthority"]
    )


def test_generated_package_resources_match_authoritative_files(root):
    packaged = root / "src/baloot_platform/resources"
    for family in ("configs", "schemas", "migrations"):
        source = root / family
        target = packaged / family
        assert target.is_dir()
        source_files = sorted(
            path.relative_to(source) for path in source.rglob("*") if path.is_file()
        )
        target_files = sorted(
            path.relative_to(target) for path in target.rglob("*") if path.is_file()
        )
        assert target_files == source_files
        for relative in source_files:
            assert (target / relative).read_bytes() == (source / relative).read_bytes()
    for name in (
        "localization-usage-v1.json",
        "localization-usage-ortools-v4.json",
    ):
        manifest = Path(name)
        assert (packaged / "generated" / manifest).read_bytes() == (
            root / "generated" / manifest
        ).read_bytes()
    for history in (root / "generated/localization-history").glob("*.json"):
        relative = history.relative_to(root)
        assert (packaged / relative).read_bytes() == history.read_bytes()


def test_resource_sync_retains_pinned_receipts_and_localization_history(tmp_path, monkeypatch):
    from scripts import sync_package_resources

    monkeypatch.setattr(sync_package_resources, "__file__", str(tmp_path / "scripts/sync.py"))
    canonical = {
        "generated/localization-usage-ortools-v4.json": b"current\n",
        "generated/localization-history/frozen.json": b"historical\n",
        **{f"release/{name}": name.encode() for name in sync_package_resources.AUTHORITY_RECEIPTS},
    }
    for relative, payload in canonical.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    sync_package_resources.main()
    for relative, payload in canonical.items():
        assert (tmp_path / "src/baloot_platform/resources" / relative).read_bytes() == payload


def test_generated_profile_schemas_use_canonical_lf(root):
    for schema in sorted((root / "schemas/profiles").glob("*.schema.json")):
        payload = schema.read_bytes()
        assert payload.endswith(b"\n"), schema
        assert b"\r\n" not in payload, schema
