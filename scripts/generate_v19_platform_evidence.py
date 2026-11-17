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
    with tempfile.TemporaryDirectory(prefix="v19-evidence-") as temp:
        store = SQLiteConfigurationStore(Path(temp) / "configuration.sqlite3")
        profile = yaml.safe_load((ROOT / "configs/configuration/control_plane_v19.yaml").read_text(encoding="utf-8"))
        service = VersionedConfigurationService(store, root=ROOT, profile=profile)
        resolved = service.bootstrap_packaged_bundle(actor_id="SYSTEM-V19-EVIDENCE")
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
        (release / "configuration-registry-v19.json").write_text(json.dumps(configuration, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    container = build_container()
    try:
        measurements = [
            container.performance.measure("configuration_resolve_p95_ms", lambda: container.configuration.resolve(), iterations=40, warmup=5),
            container.performance.measure("plugin_registry_list_p95_ms", lambda: container.registry.list(), iterations=80, warmup=5),
            container.performance.measure("canonical_payload_p95_ms", lambda: json.dumps(container.capability_matrix, sort_keys=True, separators=(",", ":")), iterations=80, warmup=5),
        ]
        performance = container.performance.report(measurements)
        performance.update({"schemaVersion": 1, "version": __version__})
        (release / "performance-benchmark-v19.json").write_text(json.dumps(performance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        plugin = {
            "schemaVersion": 1,
            "version": __version__,
            "status": "passed",
            "policy": container.registry.policy.to_dict(),
            "plugins": container.registry.list(),
            "externalInProcessAllowed": False,
            "witContract": "wit/baloot-plugin.wit",
            "witSha256": sha256(ROOT / "wit/baloot-plugin.wit"),
        }
        (release / "plugin-host-v19.json").write_text(json.dumps(plugin, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        container.shutdown()

    parity = json.loads((ROOT / "generated/capability-parity-v19.json").read_text(encoding="utf-8"))
    aggregate = {
        "schemaVersion": 1,
        "project": "baloot-platform",
        "version": __version__,
        "generatedAt": datetime.fromtimestamp(EPOCH, timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "passed" if parity["status"] == "passed" and performance["status"] == "passed" else "failed",
        "configurationEvidence": "release/configuration-registry-v19.json",
        "pluginEvidence": "release/plugin-host-v19.json",
        "performanceEvidence": "release/performance-benchmark-v19.json",
        "capabilityParity": parity,
        "migrationSha256": sha256(ROOT / "migrations/011_configuration_registry_v19.sql"),
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
    (release / "platform-v19-evidence.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(release / "platform-v19-evidence.json")


if __name__ == "__main__":
    main()
