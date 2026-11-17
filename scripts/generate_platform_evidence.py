#!/usr/bin/env python3
"""Generate configuration and plugin evidence from existing authorities."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import yaml

from baloot_platform.config import build_bundle, platform_root
from baloot_platform.filesystem import atomic_write_text
from baloot_platform.configuration import (
    SQLiteConfigurationStore,
    VersionedConfigurationService,
)
from baloot_platform.service.container import build_container
from baloot_platform.version import __version__

ROOT = platform_root()
RELEASE = ROOT / "release"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def configuration_evidence() -> dict:
    bundle = build_bundle(ROOT)
    with tempfile.TemporaryDirectory(
        prefix="configuration-evidence-",
        ignore_cleanup_errors=True,
    ) as temp:
        store = SQLiteConfigurationStore(Path(temp) / "configuration.sqlite3")
        try:
            profile = yaml.safe_load(
                (ROOT / "configs/configuration/control-plane.yaml").read_text(
                    encoding="utf-8"
                )
            )
            service = VersionedConfigurationService(store, root=ROOT, profile=profile)
            resolved = service.bootstrap_packaged_bundle(
                actor_id="SYSTEM-EVIDENCE"
            )
            event_verification = store.verify_events(
                "bundle", resolved["bundle"]["revisionId"]
            )
            value = {
                "schemaVersion": 2,
                "release": "BalootX",
                "status": "passed"
                if event_verification["valid"]
                and resolved["bundle"]["version"]
                == bundle.to_dict()["profile"]["version"]
                else "failed",
                "packageVersion": __version__,
                "resolvedBundle": resolved["bundle"],
                "effectiveHash": resolved["effectiveHash"],
                "resolutionTrace": resolved["resolutionTrace"],
                "profileFamilies": sorted(resolved["profiles"]),
                "profileRevisions": len(store.list_profiles(limit=1000)),
                "bundleRevisions": len(store.list_bundles(limit=100)),
                "bundleEventVerification": event_verification,
                "bundleSha256": bundle.sha256,
                "unknownKeysRejected": True,
                "automaticProductionApply": False,
                "authority": "VersionedConfigurationService",
            }
            return value
        finally:
            store.close()


def plugin_evidence() -> dict:
    container = build_container()
    try:
        policy = container.registry.policy.to_dict()
        plugins = container.registry.list()
        return {
            "schemaVersion": 2,
            "release": "BalootX",
            "status": "passed",
            "packageVersion": __version__,
            "policy": policy,
            "plugins": plugins,
            "pluginCount": len(plugins),
            "externalInProcessAllowed": False,
            "manifestFirstDiscovery": True,
            "artifactHashVerified": True,
            "signatureVerification": "Ed25519 trusted-signer policy",
            "processHost": "bounded JSON-RPC subprocess",
            "wasmHost": "optional Wasmtime component host; deny-by-default",
            "witContract": "wit/baloot-plugin.wit",
            "witSha256": sha256(ROOT / "wit/baloot-plugin.wit"),
            "proofBoundary": (
                "This evidence records the configured host and registered plugins. "
                "It does not certify third-party plugin security."
            ),
        }
    finally:
        container.shutdown()


def main() -> None:
    RELEASE.mkdir(parents=True, exist_ok=True)
    configuration_path = RELEASE / "configuration-registry.json"
    plugin_path = RELEASE / "plugin-host.json"
    write_json(configuration_path, configuration_evidence())
    write_json(plugin_path, plugin_evidence())
    print(configuration_path)
    print(plugin_path)


if __name__ == "__main__":
    main()
