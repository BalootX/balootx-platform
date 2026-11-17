#!/usr/bin/env python3
"""Generate deterministic executable usage receipts for governed product copy."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import re
import string
import subprocess
import sys
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baloot_platform.config import ProfileArtifact, build_bundle  # noqa: E402
from baloot_platform.adjudication.research import (  # noqa: E402
    AdjudicationResearchAuthorityV1,
)
from baloot_platform.guide import (  # noqa: E402
    PUBLIC_GUIDE_DYNAMIC_VOCABULARIES,
    build_engine_guide,
)
from baloot_platform.research.contracts import canonical_sha256  # noqa: E402
from baloot_platform.research.recipe_v4 import (  # noqa: E402
    research_recipe_v4_authoring_catalog,
)
from baloot_platform.research.scenario_registry import (  # noqa: E402
    SCENARIO_RULE_APPLICABILITY_CODES,
)
from baloot_platform.research.scenario_adjudication_predicates_v3 import (  # noqa: E402
    adjudication_predicate_definitions_v3,
)
from baloot_platform.research.scenario_predicate_manifest_v2 import (  # noqa: E402
    ScenarioPredicateCatalogLoweringManifestV2,
)

PRODUCT_PATH = ROOT / "configs/product/studio_ortools.yaml"
ADJUDICATION_PATH = ROOT / "configs/adjudication/research_events_v2.yaml"
FALSE_EKKA_ADJUDICATION_PATH = (
    ROOT / "configs/adjudication/research_events_false_ekka_unratified_v1.yaml"
)
ACTIVE_AUTHORING_BUNDLE_PATH = (
    ROOT / "configs/bundle_ortools_research_native_v3_1.yaml"
)
ACTIVE_BUNDLE_PATHS = (
    ROOT / "configs/bundle_ortools_research_v4.yaml",
    ACTIVE_AUTHORING_BUNDLE_PATH,
)
LEGACY_PRODUCT_PATH = ROOT / "configs/product/studio_v1.yaml"
FOUNDRY_DASHBOARD_ROOT = ROOT / "src/baloot_platform/service/foundry_dashboard"
FOUNDRY_SOURCE_ROOT = FOUNDRY_DASHBOARD_ROOT / "src"
FOUNDRY_EXTRACTOR = FOUNDRY_DASHBOARD_ROOT / "scripts/extract-copy-usage.mjs"
RESEARCH_SERVICE = ROOT / "src/baloot_platform/research/service.py"
GUIDE_SOURCE = ROOT / "src/baloot_platform/guide.py"
RESEARCH_CONTRACTS = ROOT / "src/baloot_platform/research/contracts.py"
RESEARCH_RECIPE_V4 = ROOT / "src/baloot_platform/research/recipe_v4.py"
SCENARIO_REGISTRY = ROOT / "src/baloot_platform/research/scenario_registry.py"
SCENARIO_REGISTRY_V2 = ROOT / "src/baloot_platform/research/scenario_registry_v2.py"
SCENARIO_MANIFEST_V2 = (
    ROOT / "src/baloot_platform/research/scenario_predicate_manifest_v2.py"
)
CHALLENGE_PROFILE_SCHEMA = ROOT / "schemas/profiles/challenge.schema.json"
LEGACY_OUTPUT = ROOT / "generated/localization-usage-v1.json"
LEGACY_PACKAGED_OUTPUT = (
    ROOT / "src/baloot_platform/resources/generated/localization-usage-v1.json"
)
OUTPUT = ROOT / "generated/localization-usage-ortools-v4.json"
PACKAGED_OUTPUT = (
    ROOT / "src/baloot_platform/resources/generated/localization-usage-ortools-v4.json"
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
        .replace("\r\n", "\n")
        .encode("utf-8")
        + b"\n"
    )


def _file_hash(path: Path) -> str:
    return canonical_sha256(
        {"bytesSha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}
    )


def _iter_leaves(
    value: Any, path: tuple[str, ...]
) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _iter_leaves(value[key], (*path, str(key)))
        return
    if isinstance(value, list):
        yield path, copy.deepcopy(value)
        return
    yield path, value


def _foundry_source_paths(
    source_root: Path = FOUNDRY_SOURCE_ROOT,
) -> tuple[Path, ...]:
    """Return deterministic authored Foundry modules, excluding generated API code."""

    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise ValueError(f"Foundry source root is missing: {source_root}")
    paths: list[Path] = []
    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix not in {".ts", ".tsx"}:
            continue
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(source_root)
        except ValueError as error:
            raise ValueError(f"Foundry source escapes its root: {path}") from error
        if relative.parts[:2] == ("api", "generated"):
            continue
        paths.append(resolved)
    ordered = tuple(
        sorted(paths, key=lambda item: item.relative_to(source_root).as_posix())
    )
    if not ordered:
        raise ValueError(f"Foundry authored source set is empty: {source_root}")
    if len(ordered) != len(set(ordered)):
        raise ValueError("Foundry authored source set contains duplicate paths")
    return ordered


def _validated_foundry_sources(
    source_paths: Sequence[Path], source_root: Path
) -> tuple[Path, ...]:
    source_root = source_root.resolve()
    validated: list[Path] = []
    for path in source_paths:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(source_root)
        except ValueError as error:
            raise ValueError(f"Foundry source escapes its root: {path}") from error
        if resolved.suffix not in {".ts", ".tsx"}:
            raise ValueError(f"Foundry source is not TypeScript: {relative.as_posix()}")
        if relative.parts[:2] == ("api", "generated"):
            raise ValueError(
                f"Generated API source cannot enter authored copy evidence: {relative.as_posix()}"
            )
        if not resolved.is_file():
            raise ValueError(f"Foundry source is missing: {relative.as_posix()}")
        validated.append(resolved)
    ordered = tuple(
        sorted(validated, key=lambda item: item.relative_to(source_root).as_posix())
    )
    if not ordered:
        raise ValueError("Foundry authored source set is empty")
    if len(ordered) != len(set(ordered)):
        raise ValueError("Foundry authored source set contains duplicate paths")
    return ordered


def _node_usage(
    source_paths: Sequence[Path] | None = None,
    *,
    source_root: Path = FOUNDRY_SOURCE_ROOT,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    paths = _validated_foundry_sources(
        source_paths
        if source_paths is not None
        else _foundry_source_paths(source_root),
        source_root,
    )
    command = ["node", str(FOUNDRY_EXTRACTOR), *(str(path) for path in paths)]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            command,
            cwd=source_root.parent,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            creationflags=creationflags,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or str(error)).strip()
        raise ValueError(f"Foundry copy extraction failed: {detail}") from error
    try:
        usage = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("Foundry copy extractor returned invalid JSON") from error
    expected_sources = [
        path.relative_to(source_root.parent).as_posix() for path in paths
    ]
    if usage.get("schemaVersion") != 2:
        raise ValueError("Foundry copy extractor schema is unsupported")
    if usage.get("sources") != expected_sources:
        raise ValueError(
            "Foundry copy extractor source set differs from authored source discovery"
        )
    source_set = set(expected_sources)
    key_sets: dict[str, set[str]] = {}
    for field in ("staticKeys", "dynamicKeys", "optionalStaticKeys"):
        keys = usage.get(field)
        if (
            not isinstance(keys, list)
            or any(not isinstance(key, str) or not key for key in keys)
            or keys != sorted(set(keys))
        ):
            raise ValueError(f"Foundry copy extractor has invalid {field}")
        key_sets[field] = set(keys)
    if not key_sets["optionalStaticKeys"].issubset(key_sets["staticKeys"]):
        raise ValueError("optional copy keys must have static consumer evidence")
    for field in ("staticKeyConsumers", "dynamicKeyConsumers"):
        consumers = usage.get(field)
        if not isinstance(consumers, dict):
            raise ValueError(f"Foundry copy extractor omitted {field}")
        for key, values in consumers.items():
            if not isinstance(key, str) or not isinstance(values, list) or not values:
                raise ValueError(f"Foundry copy extractor has invalid {field} entry")
            if values != sorted(set(values)) or not set(values).issubset(source_set):
                raise ValueError(
                    f"Foundry copy extractor has invalid {field} provenance"
                )
    metadata_fields = {"locale", "direction", "page_title"}
    if set(usage["staticKeyConsumers"]) != key_sets["staticKeys"] - metadata_fields:
        raise ValueError("Foundry copy extractor static key provenance is incomplete")
    if set(usage["dynamicKeyConsumers"]) != key_sets["dynamicKeys"]:
        raise ValueError("Foundry copy extractor dynamic key provenance is incomplete")
    catalog_consumers = usage.get("catalogConsumers")
    if not isinstance(catalog_consumers, list) or not catalog_consumers:
        raise ValueError("Foundry Research catalog consumer marker is missing")
    if catalog_consumers != sorted(set(catalog_consumers)):
        raise ValueError("Foundry Research catalog consumers are not deterministic")
    if any(
        not value.endswith(":tCatalog")
        or value.removesuffix(":tCatalog") not in source_set
        for value in catalog_consumers
    ):
        raise ValueError("Foundry Research catalog consumer provenance is invalid")
    copy_consumers = usage.get("copyConsumers")
    if not isinstance(copy_consumers, list) or not copy_consumers:
        raise ValueError("Foundry copy authority consumer provenance is invalid")
    if copy_consumers != sorted(set(copy_consumers)) or any(
        not value.endswith(":readFoundryCopy")
        or value.removesuffix(":readFoundryCopy") not in source_set
        for value in copy_consumers
    ):
        raise ValueError("Foundry copy authority consumer provenance is invalid")
    return usage


def _foundry_manifest_consumer(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"invalid Foundry consumer provenance: {value!r}")
    source, separator, symbol = value.partition(":")
    path = PurePosixPath(source)
    if (
        path.is_absolute()
        or not source.startswith("src/")
        or ".." in path.parts
        or any(not part for part in path.parts)
    ):
        raise ValueError(f"invalid Foundry consumer provenance: {value!r}")
    if separator and symbol not in {"tCatalog", "readFoundryCopy"}:
        raise ValueError(f"invalid Foundry consumer symbol: {value!r}")
    suffix = f":{symbol}" if separator else ""
    return f"foundry_dashboard/{source}{suffix}"


def _frontend_key_consumers(node_usage: dict[str, Any]) -> dict[str, set[str]]:
    consumers: dict[str, set[str]] = {}
    for field in ("staticKeyConsumers", "dynamicKeyConsumers"):
        for key, sources in dict(node_usage[field]).items():
            consumers.setdefault(str(key), set()).update(
                _foundry_manifest_consumer(str(source)) for source in sources
            )
    metadata_fields = {"locale", "direction", "page_title"}
    expected = (
        set(node_usage["staticKeys"]) | set(node_usage["dynamicKeys"])
    ) - metadata_fields
    if set(consumers) != expected or any(not values for values in consumers.values()):
        raise ValueError("Foundry copy extractor key provenance is incomplete")
    return consumers


def _hard_coded_arabic(
    paths: Sequence[Path], *, display_root: Path = ROOT
) -> dict[str, list[str]]:
    violations: dict[str, list[str]] = {}
    display_root = display_root.resolve()
    ordered = sorted(
        (Path(item).resolve() for item in paths), key=lambda item: item.as_posix()
    )
    for path in ordered:
        text = path.read_text(encoding="utf-8")
        matches = sorted(set(re.findall(r"[\u0600-\u06ff]+", text)))
        if not matches:
            continue
        try:
            label = path.relative_to(display_root).as_posix()
        except ValueError:
            label = path.as_posix()
        violations[label] = matches
    return violations


def _python_localized_calls() -> dict[str, set[str]]:
    tree = ast.parse(
        RESEARCH_SERVICE.read_text(encoding="utf-8"), filename=str(RESEARCH_SERVICE)
    )
    calls: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "_localized":
            continue
        if (
            not node.args
            or not isinstance(node.args[0], ast.Constant)
            or not isinstance(node.args[0].value, str)
        ):
            raise ValueError(f"dynamic _localized() key at line {node.lineno}")
        if any(keyword.arg is None for keyword in node.keywords):
            raise ValueError(f"expanded _localized() arguments at line {node.lineno}")
        key = str(node.args[0].value)
        placeholders = {str(keyword.arg) for keyword in node.keywords}
        previous = calls.setdefault(key, placeholders)
        if previous != placeholders:
            raise ValueError(f"inconsistent template arguments for {key}")
    return calls


def _template_fields(value: str) -> set[str]:
    fields: set[str] = set()
    for _, field, _, _ in string.Formatter().parse(value):
        if field:
            fields.add(field.split(".", 1)[0].split("[", 1)[0])
    return fields


def _normalize_catalog_localization_key(raw: Any) -> str:
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise ValueError(f"invalid Research catalog localization key: {raw!r}")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)?", raw):
        raise ValueError(f"invalid Research catalog localization key: {raw!r}")
    return raw if "." in raw else f"workflow.{raw}"


def _collect_catalog_localization_keys(value: Any, keys: set[str]) -> None:
    if isinstance(value, dict):
        for name, item in value.items():
            if name in {
                "labelKey",
                "helpKey",
                "defaultKey",
                "x-messageArKey",
                "x-messageEnKey",
                "x-helpArKey",
                "x-helpEnKey",
            }:
                keys.add(_normalize_catalog_localization_key(item))
            elif name in {"x-enumMessageArKeys", "x-enumMessageEnKeys"}:
                for key in item.values():
                    keys.add(_normalize_catalog_localization_key(key))
            else:
                _collect_catalog_localization_keys(item, keys)
    elif isinstance(value, list):
        for item in value:
            _collect_catalog_localization_keys(item, keys)


def _active_authoring_defaults() -> Mapping[str, Any]:
    bundle = build_bundle(ROOT, bundle_path=ACTIVE_AUTHORING_BUNDLE_PATH)
    research = bundle.components["challenge"].data.get("scenarioResearchV4")
    if not isinstance(research, Mapping):
        raise ValueError("active Research bundle has no scenarioResearchV4 authority")
    defaults = research.get("authoringDefaults", {})
    if not isinstance(defaults, Mapping):
        raise ValueError("active Research authoring defaults are invalid")
    return copy.deepcopy(dict(defaults))


def _research_catalog_localization_keys() -> set[str]:
    from baloot_platform.research.family_catalog import family_catalog_schema_v1
    manifest = ScenarioPredicateCatalogLoweringManifestV2()
    catalog = research_recipe_v4_authoring_catalog(
        execution_capabilities={"combinations": [], "decisionSelectorSchemas": {}},
        policy_catalog={},
        default_policy_profiles={},
        proof_budgets={"profiles": {"late_game_exact_v1": {}}},
        authoring_defaults=_active_authoring_defaults(),
        predicate_kinds=tuple(
            definition.kind for definition in manifest.exposed_definitions
        ),
        adjudication_catalog=(
            AdjudicationResearchAuthorityV1.from_yaml(
                ADJUDICATION_PATH
            ).authoring_catalog()
        ),
    )
    keys: set[str] = set()
    _collect_catalog_localization_keys(catalog, keys)
    _collect_catalog_localization_keys(family_catalog_schema_v1(), keys)
    _collect_catalog_localization_keys(
        json.loads(CHALLENGE_PROFILE_SCHEMA.read_text(encoding="utf-8")), keys
    )
    _collect_catalog_localization_keys(
        AdjudicationResearchAuthorityV1.from_yaml(
            FALSE_EKKA_ADJUDICATION_PATH
        ).authoring_catalog(),
        keys,
    )

    def collect_leaf(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                collect_leaf(item)
        elif isinstance(value, list):
            for item in value:
                collect_leaf(item)
        elif isinstance(value, str):
            keys.add(_normalize_catalog_localization_key(value))

    collect_leaf(catalog["localizationKeys"])
    return keys


def _scenario_catalog_localization_keys() -> set[str]:
    keys: set[str] = {
        _normalize_catalog_localization_key(f"research_scenario_issue_{code}{suffix}")
        for code in SCENARIO_RULE_APPLICABILITY_CODES
        for suffix in ("", "_en")
    }
    for definition in adjudication_predicate_definitions_v3(include_false_ekka=True):
        keys.add(_normalize_catalog_localization_key(definition.message_ar_key))
        keys.add(_normalize_catalog_localization_key(definition.message_en_key))
        _collect_catalog_localization_keys(definition.to_dict(), keys)
    for definition in ScenarioPredicateCatalogLoweringManifestV2().exposed_definitions:
        keys.add(_normalize_catalog_localization_key(definition.message_ar_key))
        keys.add(_normalize_catalog_localization_key(definition.message_en_key))
        _collect_catalog_localization_keys(definition.argument_schema, keys)
    return keys


def _alternative(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.125
    if isinstance(value, str):
        return value + "__usage_probe__"
    if isinstance(value, list):
        return [*reversed(value)] if len(value) > 1 else [*value, "__usage_probe__"]
    raise TypeError(f"unsupported public-guide leaf: {type(value).__name__}")


def _assign(root: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current: Any = root
    for key in path[:-1]:
        current = current[int(key)] if isinstance(current, list) else current[key]
    if isinstance(current, list):
        current[int(path[-1])] = value
    else:
        current[path[-1]] = value


def _semantic_guide(value: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(value)
    normalized.pop("guideHash", None)
    normalized.get("audit", {}).pop("productHash", None)
    return normalized


def _guide_receipts(product: dict[str, Any]) -> list[dict[str, Any]]:
    bundles = tuple(
        build_bundle(ROOT, bundle_path=path) for path in ACTIVE_BUNDLE_PATHS
    )
    baselines = tuple(_semantic_guide(build_engine_guide(bundle)) for bundle in bundles)
    configured_terms = {
        group: tuple(sorted(str(key) for key in values))
        for group, values in dict(product["public_guide"]["terms"]).items()
    }
    expected_terms = {
        group: tuple(sorted(values))
        for group, values in PUBLIC_GUIDE_DYNAMIC_VOCABULARIES.items()
    }
    if configured_terms != expected_terms:
        raise ValueError("public-guide dynamic vocabulary registration is stale")

    receipts: list[dict[str, Any]] = []
    for path, value in _iter_leaves(product["public_guide"], ("public_guide",)):
        primitive_id = "localization." + ".".join(path)
        if path == ("public_guide", "schema_version"):
            proof_kind = "closed_schema_invariant"
            status = "proven_invariant"
        else:
            mutated = copy.deepcopy(product)
            _assign(mutated, path, _alternative(value))
            artifact = ProfileArtifact(
                family="product",
                path="<localization-usage-probe>",
                data=mutated,
                sha256=canonical_sha256(mutated),
            )
            changed = any(
                _semantic_guide(
                    build_engine_guide(
                        replace(
                            bundle,
                            components={**bundle.components, "product": artifact},
                        )
                    )
                )
                != baseline
                for bundle, baseline in zip(bundles, baselines, strict=True)
            )
            if changed:
                proof_kind = "semantic_output_mutation"
                status = "proven_behavioral"
            elif len(path) >= 4 and path[1] == "terms":
                group, key = path[2], path[3]
                if key not in PUBLIC_GUIDE_DYNAMIC_VOCABULARIES.get(group, ()):
                    raise ValueError(
                        f"unregistered public-guide vocabulary: {group}.{key}"
                    )
                proof_kind = "complete_dynamic_vocabulary"
                status = "proven_invariant"
            else:
                raise ValueError(f"unused public-guide key: {'.'.join(path)}")
        proof = {
            "primitiveId": primitive_id,
            "status": status,
            "proofKind": proof_kind,
            "consumers": ["baloot_platform.guide:build_engine_guide"],
        }
        proof["proofHash"] = canonical_sha256(proof)
        receipts.append(proof)
    return receipts


def build_manifest(*, product: Mapping[str, Any] | None = None) -> dict[str, Any]:
    foundry_sources = _foundry_source_paths()
    hard_coded = _hard_coded_arabic((*foundry_sources, RESEARCH_SERVICE, GUIDE_SOURCE))
    if hard_coded:
        raise ValueError(f"hard-coded Arabic product copy detected: {hard_coded}")
    product = product if product is not None else yaml.safe_load(PRODUCT_PATH.read_text(encoding="utf-8"))
    messages = {
        f"{group}.{key}": str(value)
        for group, values in dict(product["foundry_copy"]["groups"]).items()
        for key, value in dict(values).items()
    }
    node_usage = _node_usage(foundry_sources)
    frontend = set(node_usage["staticKeys"]) | set(node_usage["dynamicKeys"])
    frontend_consumers = _frontend_key_consumers(node_usage)
    backend = _python_localized_calls()
    backend_paths = {f"workflow.{key}" for key in backend}
    metadata_fields = {"locale", "direction", "page_title"}
    frontend_messages = frontend - metadata_fields
    frontend_messages -= set(node_usage.get("optionalStaticKeys", ())) - set(messages)
    catalog_consumers = {
        _foundry_manifest_consumer(value) for value in node_usage["catalogConsumers"]
    }
    copy_consumers = {
        _foundry_manifest_consumer(value) for value in node_usage["copyConsumers"]
    }
    recipe_catalog_paths = _research_catalog_localization_keys()
    scenario_catalog_paths = _scenario_catalog_localization_keys()
    configured = set(messages)
    for paths in (recipe_catalog_paths, scenario_catalog_paths):
        paths.update(f"{path}_en" for path in tuple(paths) if f"{path}_en" in configured)
        paths.update(
            path.removesuffix("_en") for path in tuple(paths)
            if path.endswith("_en") and path.removesuffix("_en") in configured
        )
    catalog_paths = recipe_catalog_paths | scenario_catalog_paths
    unexpected_metadata = sorted((frontend & metadata_fields) ^ metadata_fields)
    if unexpected_metadata:
        raise ValueError(f"Foundry metadata usage mismatch: {unexpected_metadata}")
    used = frontend_messages | backend_paths | catalog_paths
    used |= {f"{path}_en" for path in tuple(used) if f"{path}_en" in configured}
    used |= {
        path.removesuffix("_en")
        for path in tuple(used)
        if path.endswith("_en") and path.removesuffix("_en") in configured
    }
    missing = sorted(used - configured)
    stale = sorted(configured - used)
    if missing or stale:
        raise ValueError(
            f"Foundry localization usage mismatch: missing={missing} stale={stale}"
        )

    receipts: list[dict[str, Any]] = []
    for path in sorted(configured):
        consumers: list[str] = []
        proof_kind = "static_ast_reference"
        if path in frontend_messages:
            consumers.extend(frontend_consumers[path])
        if path in catalog_paths:
            consumers.extend(catalog_consumers)
        workflow_key = path.removeprefix("workflow.")
        if path.startswith("workflow.") and workflow_key in backend:
            consumers.append("baloot_platform.research.service:_localized")
            expected_fields = backend[workflow_key]
            actual_fields = _template_fields(messages[path])
            if actual_fields != expected_fields:
                raise ValueError(
                    f"template fields differ for {path}: configured={sorted(actual_fields)} "
                    f"call={sorted(expected_fields)}"
                )
        if path in catalog_paths:
            proof_kind = "registered_catalog_dynamic_vocabulary"
        elif path in set(node_usage["dynamicKeys"]):
            proof_kind = "registered_dynamic_vocabulary"
        proof = {
            "primitiveId": f"localization.foundry_copy.groups.{path}",
            "status": "proven_behavioral",
            "proofKind": proof_kind,
            "consumers": sorted(set(consumers)),
            "templateFields": sorted(_template_fields(messages[path])),
        }
        proof["proofHash"] = canonical_sha256(proof)
        receipts.append(proof)

    for field, consumers in (
        ("locale", copy_consumers),
        ("direction", copy_consumers),
        ("page_title", {"baloot_platform.service.api:foundry_shell"}),
    ):
        proof = {
            "primitiveId": f"localization.foundry_copy.{field}",
            "status": "proven_behavioral",
            "proofKind": "static_ast_reference",
            "consumers": sorted(consumers),
        }
        proof["proofHash"] = canonical_sha256(proof)
        receipts.append(proof)

    for path, proof_kind, consumer in (
        (
            "schema_version",
            "closed_schema_invariant",
            "baloot_platform.config:validate_profile_payload",
        ),
        ("profile.id", "profile_identity", "baloot_platform.config:ProfileArtifact.id"),
        (
            "profile.version",
            "profile_identity",
            "baloot_platform.config:ProfileArtifact.version",
        ),
        (
            "profile.status",
            "profile_identity",
            "baloot_platform.config:ProfileArtifact.to_dict",
        ),
    ):
        proof = {
            "primitiveId": f"localization.{path}",
            "status": "proven_invariant",
            "proofKind": proof_kind,
            "consumers": [consumer],
        }
        proof["proofHash"] = canonical_sha256(proof)
        receipts.append(proof)

    receipts.extend(_guide_receipts(product))
    receipts.sort(key=lambda item: item["primitiveId"])
    sources = {
        path.relative_to(ROOT).as_posix(): _file_hash(path)
        for path in (
            *foundry_sources,
            FOUNDRY_EXTRACTOR,
            RESEARCH_SERVICE,
            GUIDE_SOURCE,
            RESEARCH_CONTRACTS,
            RESEARCH_RECIPE_V4,
            SCENARIO_REGISTRY,
            SCENARIO_REGISTRY_V2,
            SCENARIO_MANIFEST_V2,
            CHALLENGE_PROFILE_SCHEMA,
        )
    }
    core = {
        "schemaVersion": 1,
        "kind": "LocalizationUsageManifestV1",
        "productHash": canonical_sha256(product),
        "sources": dict(sorted(sources.items())),
        "dynamicVocabularies": {
            **{
                group: list(sorted(values))
                for group, values in sorted(PUBLIC_GUIDE_DYNAMIC_VOCABULARIES.items())
            },
            "researchRecipeConfiguration": sorted(recipe_catalog_paths),
            "researchScenarioConfiguration": sorted(scenario_catalog_paths),
        },
        "receiptCount": len(receipts),
        "receipts": receipts,
    }
    return {**core, "sha256": canonical_sha256(core)}


def _validate_frozen_legacy_manifest(path: Path) -> None:
    """Validate the immutable legacy receipt without reinterpreting its sources."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid frozen legacy localization manifest: {path}"
        ) from error
    legacy_product = yaml.safe_load(LEGACY_PRODUCT_PATH.read_text(encoding="utf-8"))
    expected_hash = str(payload.get("sha256") or "")
    core = dict(payload)
    core.pop("sha256", None)
    if (
        payload.get("schemaVersion") != 1
        or payload.get("kind") != "LocalizationUsageManifestV1"
        or payload.get("productHash") != canonical_sha256(legacy_product)
        or expected_hash != canonical_sha256(core)
    ):
        raise ValueError(f"invalid frozen legacy localization manifest: {path}")
    receipts = payload.get("receipts")
    if not isinstance(receipts, list) or payload.get("receiptCount") != len(receipts):
        raise ValueError(f"invalid frozen legacy localization manifest: {path}")
    primitive_ids: set[str] = set()
    for item in receipts:
        if not isinstance(item, dict):
            raise ValueError(f"invalid frozen legacy localization manifest: {path}")
        primitive_id = str(item.get("primitiveId") or "")
        proof_hash = str(item.get("proofHash") or "")
        proof_core = dict(item)
        proof_core.pop("proofHash", None)
        if (
            not primitive_id.startswith("localization.")
            or primitive_id in primitive_ids
            or item.get("status") not in {"proven_behavioral", "proven_invariant"}
            or proof_hash != canonical_sha256(proof_core)
        ):
            raise ValueError(f"invalid frozen legacy localization manifest: {path}")
        primitive_ids.add(primitive_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    _validate_frozen_legacy_manifest(LEGACY_OUTPUT)
    if LEGACY_PACKAGED_OUTPUT.read_bytes() != LEGACY_OUTPUT.read_bytes():
        raise SystemExit(
            f"stale frozen legacy localization manifest: {LEGACY_PACKAGED_OUTPUT}"
        )
    manifest = build_manifest()
    rendered = _canonical_bytes(manifest)
    if args.check:
        for path in (OUTPUT, PACKAGED_OUTPUT):
            if not path.exists() or path.read_bytes() != rendered:
                raise SystemExit(f"stale localization usage manifest: {path}")
        return 0
    for path in (OUTPUT, PACKAGED_OUTPUT):
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            previous_core = dict(previous)
            previous_hash = previous_core.pop("sha256", None)
            product_hash = str(previous.get("productHash") or "")
            if previous_hash != canonical_sha256(previous_core) or not re.fullmatch(r"[0-9a-f]{64}", product_hash):
                raise ValueError(f"invalid localization history source: {path}")
            if product_hash != manifest["productHash"]:
                history = path.parent / "localization-history" / f"{product_hash}.json"
                history.parent.mkdir(parents=True, exist_ok=True)
                if history.exists() and history.read_bytes() != path.read_bytes():
                    raise ValueError(f"conflicting immutable localization receipt: {history}")
                if not history.exists():
                    history.write_bytes(path.read_bytes())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rendered)
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
