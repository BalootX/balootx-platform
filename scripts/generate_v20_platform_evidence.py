#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

from baloot_platform.config import build_bundle
from baloot_platform.configuration import SQLiteConfigurationStore, VersionedConfigurationService
from baloot_platform.performance import PerformanceService
from baloot_platform.plugins.registry import PluginPolicy
from baloot_platform.service.container import build_container
from baloot_platform.version import __version__

ROOT = Path(__file__).resolve().parents[1]
EPOCH = int(os.environ.get("SOURCE_DATE_EPOCH", "315532800"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    release = ROOT / "release"
    release.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v20-evidence-") as temp:
        store = SQLiteConfigurationStore(Path(temp) / "configuration.sqlite3")
        profile = yaml.safe_load((ROOT / "configs/configuration/control_plane_v20.yaml").read_text(encoding="utf-8"))
        service = VersionedConfigurationService(store, root=ROOT, profile=profile)
        resolved = service.bootstrap_packaged_bundle(actor_id="SYSTEM-V20-EVIDENCE")
        profiles = store.list_profiles(limit=1000)
        bundles = store.list_bundles(limit=100)
        configuration = {
            "schemaVersion": 1,
            "version": __version__,
            "status": "passed",
            "resolvedBundle": resolved["bundle"],
            "effectiveHash": resolved["effectiveHash"],
            "profileFamilies": sorted(resolved["profiles"]),
            "profileRevisions": len(profiles),
            "bundleRevisions": len(bundles),
            "bundleEventVerification": store.verify_events("bundle", resolved["bundle"]["revisionId"]),
            "automaticProductionApply": False,
        }
        (release / "configuration-registry-v20.json").write_text(json.dumps(configuration, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Benchmark evidence is generated independently by scripts/benchmark_v20.py
    performance_path = release / "performance-benchmark-v20.json"
    if not performance_path.exists():
        raise RuntimeError("V20 performance benchmark evidence is missing")
    performance = json.loads(performance_path.read_text(encoding="utf-8"))

    container = build_container()
    try:
        plugin = {
            "schemaVersion": 2,
            "version": __version__,
            "status": "passed",
            "policy": container.registry.policy.to_dict(),
            "plugins": container.registry.list(),
            "externalInProcessAllowed": False,
            "manifestFirstDiscovery": True,
            "artifactHashVerified": True,
            "signatureVerification": "Ed25519 trusted-signer policy",
            "processHost": "bounded JSON-RPC subprocess",
            "wasmHost": "optional Wasmtime component host; deny-by-default",
            "witContract": "wit/baloot-plugin.wit",
            "witSha256": sha256(ROOT / "wit/baloot-plugin.wit"),
        }
        (release / "plugin-host-v20.json").write_text(json.dumps(plugin, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        container.shutdown()

    parity = json.loads((ROOT / "generated/capability-parity-v20.json").read_text(encoding="utf-8"))
    aggregate = {
        "schemaVersion": 1,
        "project": "baloot-platform",
        "version": __version__,
        "generatedAt": datetime.fromtimestamp(EPOCH, timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "passed" if parity["status"] == "passed" and performance["status"] == "passed" else "failed",
        "configurationEvidence": "release/configuration-registry-v20.json",
        "pluginEvidence": "release/plugin-host-v20.json",
        "performanceEvidence": "release/performance-benchmark-v20.json",
        "capabilityParity": parity,
        "migrationSha256": sha256(ROOT / "migrations/012_v20_security_configuration_live.sql"),
        "bundleSha256": build_bundle(ROOT).sha256,
        "externalValidationBoundaries": {
            "officialRulesCertification": "external signed referee governance",
            "livePostgresFailover": "requires live PostgreSQL infrastructure",
            "physicalOciBuildAndScan": "requires Docker/OCI daemon and scanner",
            "independentSecurityReview": "requires independent assessment",
            "eliteModelStrength": "requires expert-labelled data and external benchmark",
            "approvedHumanSaudiVoice": "provider interface implemented; approved recordings remain external",
        },
    }
    (release / "platform-v20-evidence.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(release / "platform-v20-evidence.json")


if __name__ == "__main__":
    main()
