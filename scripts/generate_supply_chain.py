#!/usr/bin/env python3
"""Generate the deterministic release SBOM and license inventory.

The inventory covers Python requirements, both shipped npm lockfiles, bundled
fonts, card packs, voice packs, and professional video assets. Optional source
requirements may be absent from the build environment; anything actually used
by the release must have a resolved license/provenance declaration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tomllib
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
UNRESOLVED = {"", "NOASSERTION", "UNKNOWN", "user_supplied_review_required"}
NPM_PROJECTS = (
    ROOT / "src/baloot_platform/video/remotion/package-lock.json",
    ROOT / "src/baloot_platform/service/foundry_dashboard/package-lock.json",
)


def dependency_name(spec: str) -> str:
    return re.split(r"[<>=!~;\[ ]", spec, maxsplit=1)[0].strip()


def license_for(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    for key in ("License-Expression", "License"):
        value = str(meta.get(key, "")).strip()
        if value and value.upper() != "UNKNOWN":
            return value
    classifiers = meta.get_all("Classifier") or []
    values = [item.split(" :: ")[-1] for item in classifiers if item.startswith("License ::")]
    return "; ".join(values) if values else "NOASSERTION"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def component(
    *,
    kind: str,
    name: str,
    version: str,
    license_value: str,
    scope: str,
    properties: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    props = [
        {"name": key, "value": json.dumps(value, sort_keys=True) if not isinstance(value, str) else value}
        for key, value in sorted(properties.items())
    ]
    bom = {
        "type": kind,
        "name": name,
        "version": version,
        "scope": "required" if scope in {"required", "runtime", "asset"} else "optional",
        "licenses": [{"license": {"name": license_value}}],
        "properties": props,
    }
    inventory = {
        "name": name,
        "version": version,
        "license": license_value,
        "scope": scope,
        **properties,
    }
    return bom, inventory


def python_components(pyproject: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    specs: dict[str, dict[str, Any]] = {}
    for spec in pyproject["project"]["dependencies"]:
        specs[dependency_name(spec).lower()] = {"spec": spec, "scope": "required"}
    for group, entries in pyproject["project"].get("optional-dependencies", {}).items():
        for spec in entries:
            name = dependency_name(spec).lower()
            specs.setdefault(name, {"spec": spec, "scope": "optional"})
            specs[name].setdefault("groups", []).append(group)

    values = []
    for name, item in sorted(specs.items()):
        try:
            dist = metadata.distribution(name)
            version = dist.version
            license_value = license_for(dist)
        except metadata.PackageNotFoundError:
            version = "not-installed-in-build-environment"
            license_value = "NOASSERTION"
        released = item["scope"] == "required"
        if released and license_value in UNRESOLVED:
            raise RuntimeError(f"required Python dependency has unresolved license: {name}")
        values.append(
            component(
                kind="library",
                name=name,
                version=version,
                license_value=license_value,
                scope=item["scope"],
                properties={
                    "baloot:ecosystem": "python",
                    "baloot:requirement": item["spec"],
                    "baloot:optional-groups": ",".join(sorted(item.get("groups", []))),
                    "baloot:released": str(released).lower(),
                },
            )
        )
    return values


def npm_components() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    values: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[tuple[str, str, str]] = set()
    for lock_path in NPM_PROJECTS:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        root_package = lock["packages"][""]
        direct = set(root_package.get("dependencies", {}))
        development = set(root_package.get("devDependencies", {}))
        project = str(lock_path.parent.relative_to(ROOT)).replace("\\", "/")
        for node_path, package in sorted(lock.get("packages", {}).items()):
            if not node_path or "node_modules/" not in node_path:
                continue
            name = node_path.rsplit("node_modules/", 1)[-1]
            version = str(package.get("version") or "UNKNOWN")
            key = (project, name, version)
            if key in seen:
                continue
            seen.add(key)
            scope = "runtime" if name in direct else "development" if name in development else "transitive"
            license_value = str(package.get("license") or "").strip()
            if not license_value and name.startswith("@remotion/compositor-"):
                license_value = "SEE LICENSE IN @remotion/renderer LICENSE.md"
            if license_value in UNRESOLVED:
                raise RuntimeError(f"npm package has unresolved license: {project}:{name}@{version}")
            values.append(
                component(
                    kind="library",
                    name=name,
                    version=version,
                    license_value=license_value,
                    scope=scope,
                    properties={
                        "baloot:ecosystem": "npm",
                        "baloot:project": project,
                        "baloot:lockfile": str(lock_path.relative_to(ROOT)).replace("\\", "/"),
                        "baloot:integrity": str(package.get("integrity") or ""),
                        "baloot:released": "true",
                    },
                )
            )
    return values


def card_components() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    config = yaml.safe_load((ROOT / "configs/assets/card_packs.yaml").read_text(encoding="utf-8"))
    configured = config["packs"]
    values = []
    for manifest_path in sorted((ROOT / "assets/cards/packs").glob("*/manifest.json")):
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        pack_id = str(data.get("packId") or manifest_path.parent.name)
        settings = configured.get(pack_id, {})
        enabled = bool(settings.get("enabled", False))
        license_value = str(data.get("licenseStatus") or settings.get("license_status") or "NOASSERTION")
        if enabled and license_value in UNRESOLVED:
            raise RuntimeError(f"enabled card pack has unresolved license: {pack_id}")
        values.append(
            component(
                kind="data",
                name=pack_id,
                version=str(data.get("schemaVersion", 1)),
                license_value=license_value,
                scope="asset" if enabled else "excluded-unresolved-asset",
                properties={
                    "baloot:asset-kind": "playing-card-pack",
                    "baloot:enabled": str(enabled).lower(),
                    "baloot:released": str(enabled).lower(),
                    "baloot:source": str(data.get("source") or ""),
                    "baloot:manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
                    "baloot:manifest-sha256": sha256(manifest_path),
                },
            )
        )
    return values


def bundled_asset_components() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    values = []
    fonts_path = ROOT / "assets/fonts/manifest.json"
    fonts = json.loads(fonts_path.read_text(encoding="utf-8"))
    font_licenses = {"noto": "OFL-1.1", "dejavu": "Bitstream-Vera-and-Arev"}
    for family, package in sorted(fonts["packages"].items()):
        license_file = ROOT / "assets/fonts" / package["license"]
        if not license_file.is_file():
            raise RuntimeError(f"font license file is missing: {license_file}")
        files = {
            name: detail["sha256"]
            for name, detail in sorted(fonts["fonts"].items())
            if detail["family"] == family
        }
        values.append(
            component(
                kind="data",
                name=f"font-family-{family}",
                version=str(package["version"]),
                license_value=font_licenses[family],
                scope="asset",
                properties={
                    "baloot:asset-kind": "font",
                    "baloot:files": files,
                    "baloot:license-file": str(license_file.relative_to(ROOT)).replace("\\", "/"),
                    "baloot:license-sha256": sha256(license_file),
                    "baloot:manifest-sha256": sha256(fonts_path),
                    "baloot:released": "true",
                },
            )
        )

    voice_manifests = (
        ROOT / "assets/audio/voice_packs/ar_sa_baloot_v1/generated-manifest.json",
        ROOT / "assets/audio/voice_packs/ar_sa_baloot_v2_smooth/manifest.json",
    )
    for manifest_path in voice_manifests:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        pack_id = str(data.get("packId") or manifest_path.parent.name)
        provenance = str(data.get("provenance") or data.get("license") or "")
        if not provenance:
            raise RuntimeError(f"voice pack provenance is unresolved: {pack_id}")
        values.append(
            component(
                kind="data",
                name=pack_id,
                version=str(data.get("schemaVersion", 1)),
                license_value="BalootX-project-generated",
                scope="asset",
                properties={
                    "baloot:asset-kind": "voice-pack",
                    "baloot:provenance": provenance,
                    "baloot:manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
                    "baloot:manifest-sha256": sha256(manifest_path),
                    "baloot:released": "true",
                },
            )
        )

    professional_path = ROOT / "assets/video/professional/manifest.json"
    professional = json.loads(professional_path.read_text(encoding="utf-8"))
    license_value = str(professional.get("licenseStatus") or "NOASSERTION")
    if license_value in UNRESOLVED:
        raise RuntimeError("professional video asset license is unresolved")
    values.append(
        component(
            kind="data",
            name=str(professional["assetSet"]),
            version=str(professional.get("schemaVersion", 1)),
            license_value=license_value,
            scope="asset",
            properties={
                "baloot:asset-kind": "professional-video-svg",
                "baloot:generation": str(professional["generation"]),
                "baloot:assets": professional["assets"],
                "baloot:manifest-sha256": sha256(professional_path),
                "baloot:released": "true",
            },
        )
    )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbom-output", default="release/sbom.cdx.json")
    parser.add_argument("--license-output", default="release/license-inventory.json")
    arguments = parser.parse_args()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pairs = [
        *python_components(pyproject),
        *npm_components(),
        *card_components(),
        *bundled_asset_components(),
    ]
    components = [pair[0] for pair in pairs]
    licenses = [pair[1] for pair in pairs]

    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    created = (
        datetime.fromtimestamp(int(epoch), timezone.utc)
        if epoch
        else datetime.now(timezone.utc)
    )
    project_version = pyproject["project"]["version"]
    serial_seed = hashlib.sha256(f"baloot-platform:{project_version}".encode()).hexdigest()[:32]
    serial = str(uuid.UUID(serial_seed))
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "timestamp": created.isoformat().replace("+00:00", "Z"),
            "component": {"type": "application", "name": "baloot-platform", "version": project_version},
            "properties": [
                {"name": "baloot:scope", "value": "Python requirements, npm lockfiles, bundled fonts, card packs, voice packs, and professional video assets"},
                {"name": "baloot:system-dependencies", "value": "FFmpeg and C++20 runtime are externally supplied and recorded by exact release evidence"},
            ],
        },
        "components": components,
    }
    inventory = {
        "schemaVersion": 3,
        "status": "passed",
        "componentCount": len(licenses),
        "unresolvedReleasedComponents": [],
        "components": licenses,
        "proofBoundary": "Inventory is generated from declared Python requirements, npm lockfiles, and bundled asset manifests. It does not itself approve the external Remotion production license.",
    }

    sbom_output = (ROOT / arguments.sbom_output).resolve()
    license_output = (ROOT / arguments.license_output).resolve()
    sbom_output.parent.mkdir(parents=True, exist_ok=True)
    license_output.parent.mkdir(parents=True, exist_ok=True)
    sbom_output.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    license_output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(sbom_output)


if __name__ == "__main__":
    main()
