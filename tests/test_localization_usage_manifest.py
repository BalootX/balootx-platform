from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from baloot_platform.config import build_bundle
from baloot_platform.research import BalootDomainCompilerV1
from baloot_platform.research import domain_compiler


ROOT = Path(__file__).resolve().parents[1]
LEGACY_MANIFEST_PATH = ROOT / "generated/localization-usage-v1.json"
LEGACY_PACKAGED_PATH = (
    ROOT / "src/baloot_platform/resources/generated/localization-usage-v1.json"
)
MANIFEST_PATH = ROOT / "generated/localization-usage-ortools-v4.json"
PACKAGED_PATH = (
    ROOT / "src/baloot_platform/resources/generated/localization-usage-ortools-v4.json"
)


def _generator():
    path = ROOT / "scripts/generate_localization_usage_manifest.py"
    spec = importlib.util.spec_from_file_location("localization_usage_generator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _leaves(value, path=()):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _leaves(value[key], (*path, str(key)))
        return
    if isinstance(value, list):
        yield ".".join(path)
        return
    yield ".".join(path)


def test_localization_usage_manifest_is_deterministic_complete_and_packaged():
    generator = _generator()
    generated = generator.build_manifest()
    checked_in = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert generated == checked_in
    assert PACKAGED_PATH.read_bytes() == MANIFEST_PATH.read_bytes()
    generator._validate_frozen_legacy_manifest(LEGACY_MANIFEST_PATH)
    assert LEGACY_PACKAGED_PATH.read_bytes() == LEGACY_MANIFEST_PATH.read_bytes()

    product = yaml.safe_load(generator.PRODUCT_PATH.read_text(encoding="utf-8"))
    expected = (
        {
            f"localization.{path}"
            for path in _leaves(product["foundry_copy"], ("foundry_copy",))
        }
        | {
            f"localization.{path}"
            for path in _leaves(product["public_guide"], ("public_guide",))
        }
        | {
            "localization.schema_version",
            "localization.profile.id",
            "localization.profile.version",
            "localization.profile.status",
        }
    )
    receipts = {item["primitiveId"] for item in checked_in["receipts"]}
    assert receipts == expected
    assert checked_in["receiptCount"] == len(receipts)
    assert all(item["proofHash"] for item in checked_in["receipts"])
    assert not any("compiler_coverage_covered" in item for item in receipts)
    assert not any("compiler_coverage_row_complete" in item for item in receipts)

    node_usage = generator._node_usage()
    assert node_usage["schemaVersion"] == 2
    assert node_usage["sources"] == [
        path.relative_to(generator.FOUNDRY_DASHBOARD_ROOT).as_posix()
        for path in generator._foundry_source_paths()
    ]
    catalog_consumers = {
        generator._foundry_manifest_consumer(value)
        for value in node_usage["catalogConsumers"]
    }
    assert catalog_consumers
    receipts_by_id = {item["primitiveId"]: item for item in checked_in["receipts"]}
    for key in ('research_lesson_tag_template_predicate_id', 'research_lesson_tag_template_arguments'):
        assert receipts_by_id[f'localization.foundry_copy.groups.workflow.{key}'][
            'proofKind'] == 'registered_catalog_dynamic_vocabulary'
    for path in (
        generator._research_catalog_localization_keys()
        | generator._scenario_catalog_localization_keys()
    ):
        receipt = receipts_by_id[f"localization.foundry_copy.groups.{path}"]
        assert receipt["proofKind"] == "registered_catalog_dynamic_vocabulary"
        assert catalog_consumers.issubset(receipt["consumers"])
    frontend_consumers = generator._frontend_key_consumers(node_usage)
    for path, consumers in frontend_consumers.items():
        receipt = receipts_by_id[f"localization.foundry_copy.groups.{path}"]
        assert consumers.issubset(receipt["consumers"])
    copy_consumers = {
        generator._foundry_manifest_consumer(value)
        for value in node_usage["copyConsumers"]
    }
    for field in ("locale", "direction"):
        receipt = receipts_by_id[f"localization.foundry_copy.{field}"]
        assert set(receipt["consumers"]) == copy_consumers


def test_foundry_authored_source_discovery_is_sorted_and_excludes_generated_api(
    tmp_path,
):
    generator = _generator()
    source_root = tmp_path / "src"
    nested = source_root / "feature" / "research.tsx"
    main = source_root / "main.tsx"
    generated = source_root / "api" / "generated" / "openapi.ts"
    ignored = source_root / "readme.js"
    for path in (nested, main, generated, ignored):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("export {};\n", encoding="utf-8")

    assert [
        path.relative_to(source_root).as_posix()
        for path in generator._foundry_source_paths(source_root)
    ] == ["feature/research.tsx", "main.tsx"]
    with pytest.raises(
        ValueError, match="Generated API source cannot enter authored copy evidence"
    ):
        generator._node_usage([generated], source_root=source_root)


def test_profile_control_copy_receipts_bind_schema_and_both_languages():
    generator = _generator()
    schema = json.loads(generator.CHALLENGE_PROFILE_SCHEMA.read_text(encoding="utf-8"))
    control = schema["properties"]["tactical_detectors"]["properties"]["nativeDelayedPayoff"]
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    receipts = {item["primitiveId"]: item for item in manifest["receipts"]}
    keys = set()
    for node in (control, *control["properties"].values()):
        keys.update((node["x-messageArKey"], node["x-helpArKey"]))
        keys.update(node.get("x-enumMessageArKeys", {}).values())
    assert keys
    for key in keys:
        for suffix in ("", "_en"):
            receipt = receipts[f"localization.foundry_copy.groups.workflow.{key}{suffix}"]
            assert receipt["proofKind"] == "registered_catalog_dynamic_vocabulary"
            assert "foundry_dashboard/src/main.tsx:tCatalog" in receipt["consumers"]
    relative = generator.CHALLENGE_PROFILE_SCHEMA.relative_to(ROOT).as_posix()
    assert manifest["sources"][relative] == generator._file_hash(generator.CHALLENGE_PROFILE_SCHEMA)


def test_foundry_copy_extractor_preserves_exact_module_consumers(tmp_path):
    generator = _generator()
    source_root = tmp_path / "src"
    main = source_root / "main.tsx"
    research = source_root / "features" / "research.tsx"
    main.parent.mkdir(parents=True, exist_ok=True)
    research.parent.mkdir(parents=True, exist_ok=True)
    main.write_text(
        """
const foundryCopy = readFoundryCopy();
const foundryCopyVariants = {'workflow.variant': ['one']} as const;
t('workflow.main');
tCatalog(field.helpKey);
void foundryCopy;
""".strip()
        + "\n",
        encoding="utf-8",
    )
    research.write_text(
        """
t('workflow.research');
tVariant('workflow.variant', 'one');
""".strip()
        + "\n",
        encoding="utf-8",
    )

    usage = generator._node_usage(
        [research, main],
        source_root=source_root,
    )
    assert usage["sources"] == ["src/features/research.tsx", "src/main.tsx"]
    assert usage["staticKeyConsumers"]["workflow.main"] == ["src/main.tsx"]
    assert usage["staticKeyConsumers"]["workflow.research"] == [
        "src/features/research.tsx"
    ]
    assert usage["dynamicKeyConsumers"]["workflow.variant_one"] == [
        "src/features/research.tsx"
    ]
    assert usage["catalogConsumers"] == ["src/main.tsx:tCatalog"]
    assert usage["copyConsumers"] == ["src/main.tsx:readFoundryCopy"]


def test_only_presence_guarded_copy_can_be_absent(tmp_path):
    source_root = tmp_path / 'src'
    source_root.mkdir()
    source = source_root / 'main.tsx'
    source.write_text("""
const foundryCopy = readFoundryCopy();
tCatalog(field.helpKey);
function label(key) {
  if (key && hasFoundryCopy('optional')) return t('optional');
  if (hasFoundryCopy('unsafe') || key) return t('unsafe');
  if (hasFoundryCopy('else')) return ''; else return t('else');
  if (hasFoundryCopy('mixed')) return t('mixed');
  return t('mixed');
}
""", encoding='utf-8')
    usage = _generator()._node_usage([source], source_root=source_root)
    assert usage['optionalStaticKeys'] == ['optional']


def test_foundry_copy_extractor_fails_closed_for_dynamic_keys(tmp_path):
    generator = _generator()
    source_root = tmp_path / "src"
    source = source_root / "main.tsx"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "const foundryCopy=readFoundryCopy();tCatalog(field.helpKey);t(key);\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"src/main\.tsx:1: dynamic t\(\) call"):
        generator._node_usage([source], source_root=source_root)


def test_scenario_studio_copy_keys_are_governed_and_unknown_fails_closed(
    monkeypatch,
):
    generator = _generator()
    usage = generator._node_usage()
    scenario_source = "src/features/research/scenarioStudio.tsx"
    scenario_keys = {
        key
        for key, consumers in usage["staticKeyConsumers"].items()
        if scenario_source in consumers
    }
    product = yaml.safe_load(generator.PRODUCT_PATH.read_text(encoding="utf-8"))
    configured = {
        f"{group}.{key}"
        for group, values in product["foundry_copy"]["groups"].items()
        for key in values
    }
    assert scenario_keys
    assert scenario_keys <= configured

    unknown_key = "workflow.research_scenario_unknown_test_key"
    broken_usage = dict(usage)
    broken_usage["staticKeys"] = sorted([*usage["staticKeys"], unknown_key])
    broken_usage["staticKeyConsumers"] = {
        **usage["staticKeyConsumers"],
        unknown_key: [scenario_source],
    }
    monkeypatch.setattr(generator, "_node_usage", lambda *_args, **_kwargs: broken_usage)
    with pytest.raises(ValueError, match=unknown_key):
        generator.build_manifest()


def test_foundry_copy_extractor_fails_closed_for_duplicate_and_outside_sources(
    tmp_path,
):
    generator = _generator()
    source_root = tmp_path / "src"
    source = source_root / "main.tsx"
    outside = tmp_path / "outside.tsx"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("export {};\n", encoding="utf-8")
    outside.write_text("export {};\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains duplicate paths"):
        generator._node_usage([source, source], source_root=source_root)
    with pytest.raises(ValueError, match="escapes its root"):
        generator._node_usage([outside], source_root=source_root)


def test_hard_coded_arabic_scan_covers_nested_foundry_modules(tmp_path):
    generator = _generator()
    source_root = tmp_path / "src"
    source = source_root / "features" / "research.tsx"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("export const unsafe='نص';\n", encoding="utf-8")
    assert generator._hard_coded_arabic([source], display_root=tmp_path) == {
        "src/features/research.tsx": ["نص"]
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("research_example", "workflow.research_example"),
        ("workflow.research_example", "workflow.research_example"),
    ],
)
def test_catalog_localization_key_normalization(raw, expected):
    assert _generator()._normalize_catalog_localization_key(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        7,
        "",
        " ",
        " research_example",
        "research_example ",
        ".research_example",
        "workflow..research_example",
        "workflow.research.example",
        "workflow.bad-key",
    ],
)
def test_catalog_localization_key_normalization_fails_closed(raw):
    with pytest.raises(ValueError, match="invalid Research catalog localization key"):
        _generator()._normalize_catalog_localization_key(raw)


def test_localization_receipts_are_compiler_qualification_evidence():
    bundle = build_bundle(
        ROOT,
        bundle_path="configs/bundle_ortools_research_v4.yaml",
    )
    compiled = BalootDomainCompilerV1().compile(
        rules=bundle.rules.data,
        conventions=bundle.component("conventions"),
        localization=bundle.component("product"),
        bundle_hash=bundle.sha256,
    )
    rows = {
        row.primitive_id: row for row in compiled.compiled.domain_ir.primitive_instances
    }
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for receipt in manifest["receipts"]:
        row = rows[receipt["primitiveId"]]
        assert row.behavioral_or_invariant_status == receipt["status"]
        assert row.verified_behavioral_evidence == receipt["proofHash"]
        assert row.qualification_eligible is True


def test_frozen_legacy_localization_receipts_remain_exactly_selectable():
    bundle = build_bundle(ROOT)
    compiled = BalootDomainCompilerV1().compile(
        rules=bundle.rules.data,
        conventions=bundle.component("conventions"),
        localization=bundle.component("product"),
        bundle_hash=bundle.sha256,
    )
    rows = {
        row.primitive_id: row for row in compiled.compiled.domain_ir.primitive_instances
    }
    manifest = json.loads(LEGACY_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert all(
        rows[item["primitiveId"]].qualification_eligible
        for item in manifest["receipts"]
    )


def test_localization_receipt_selection_rejects_substitution_and_unknown_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeResources:
        def joinpath(self, *parts: str) -> Path:
            return tmp_path / parts[-1]

    payload = MANIFEST_PATH.read_text(encoding="utf-8")
    (tmp_path / "localization-usage-v1.json").write_text(payload, encoding="utf-8")
    (tmp_path / "localization-usage-ortools-v4.json").write_text(
        payload, encoding="utf-8"
    )
    monkeypatch.setattr(domain_compiler, "files", lambda _: FakeResources())
    domain_compiler._PRESENTATION_PROOF_CACHE.clear()
    product = yaml.safe_load(_generator().PRODUCT_PATH.read_text(encoding="utf-8"))
    assert BalootDomainCompilerV1._presentation_proofs(localization=product) == {}
    assert (
        BalootDomainCompilerV1._presentation_proofs(
            localization={"unknownProfile": True}
        )
        == {}
    )


def test_product_localization_protocol_fields_are_closed_invariants():
    schema = json.loads(
        (ROOT / "schemas/profiles/product.schema.json").read_text(encoding="utf-8")
    )
    properties = schema["properties"]
    assert properties["schema_version"]["const"] == 1
    guide = properties["public_guide"]["properties"]
    foundry = properties["foundry_copy"]["properties"]
    assert guide["schema_version"]["const"] == 1
    assert guide["locale"]["const"] == "ar-SA"
    assert guide["direction"]["const"] == "rtl"
    assert foundry["locale"]["const"] == "ar-SA"
    assert foundry["direction"]["const"] == "rtl"


def test_exact_historical_localization_receipt_is_selectable_and_tamper_checked(tmp_path, monkeypatch):
    class FakeResources:
        def joinpath(self, *parts):
            return tmp_path / parts[-1]

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    product = yaml.safe_load(_generator().PRODUCT_PATH.read_text(encoding="utf-8"))
    path = tmp_path / "localization-history" / f"{manifest['productHash']}.json"
    path.parent.mkdir()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(domain_compiler, "files", lambda _: FakeResources())
    domain_compiler._PRESENTATION_PROOF_CACHE.clear()
    assert len(BalootDomainCompilerV1._presentation_proofs(localization=product)) == manifest["receiptCount"]
    manifest["receipts"][0]["proofKind"] = "tampered"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    domain_compiler._PRESENTATION_PROOF_CACHE.clear()
    assert BalootDomainCompilerV1._presentation_proofs(localization=product) == {}
