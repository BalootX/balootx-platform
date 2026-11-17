#!/usr/bin/env python3
# ruff: noqa: E402
"""Generate source-bound automated security evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY_ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from baloot_platform.benchmark import canonical_sha256
from baloot_platform.config import platform_root
from baloot_platform.filesystem import atomic_write_text

try:
    from scripts.generate_performance_evidence import git_identity
except ModuleNotFoundError:
    from generate_performance_evidence import git_identity


ROOT = platform_root()
OUTPUT = ROOT / "release/security-report.json"
REQUIRED_TEST_MARKERS = (
    "test_v19_plugin_policy_isolates_external_plugins",
    "test_v19_plugin_host_enforces_timeout",
    "test_v20_external_plugin_manifest_verifies_hash_and_signature",
    "test_v20_oidc_callback_pkce_nonce_and_claim_validation",
    "test_v20_passkey_state_is_durable_and_replay_safe",
    "test_v20_security_contract_removes_query_bearer_and_forces_rls",
    "test_v20_websocket_ticket_is_room_bound_and_one_time",
    "test_v21_analysis_rejects_unknown_objective_before_sampling",
    "test_analysis_is_forbidden_before_reveal",
    "test_deep_link_is_minimal_signed_and_tamper_evident",
)
INPUT_ARTIFACTS = {
    "pluginHost": "release/plugin-host.json",
    "sbom": "release/sbom.cdx.json",
    "licenseInventory": "release/license-inventory.json",
    "testInventory": "release/test-inventory.json",
}


def load(root: Path, relative: str) -> dict[str, Any]:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def artifact_receipt(root: Path, relative: str) -> dict[str, Any]:
    payload = (root / relative).read_bytes()
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def build(
    *,
    root: Path = ROOT,
    ci_run_id: str,
    ci_head_sha: str,
    ci_conclusion: str,
    source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = (
        git_identity(root)
        if source_identity is None
        else copy.deepcopy(dict(source_identity))
    )
    revision = str(source["revision"])
    if ci_head_sha != revision or ci_conclusion != "success":
        raise ValueError("successful CI must match the exact source head")
    plugin = load(root, "release/plugin-host.json")
    sbom = load(root, "release/sbom.cdx.json")
    licenses = load(root, "release/license-inventory.json")
    inventory = load(root, "release/test-inventory.json")
    policy = plugin.get("policy", {})
    required_policy = {
        "externalInProcessAllowed": False,
        "manifestFirstDiscovery": True,
        "requireContentAddressedExternal": True,
        "requireSignedExternal": True,
    }
    if any(policy.get(key) is not expected for key, expected in required_policy.items()):
        raise ValueError("plugin security policy is incomplete")
    raw_node_ids = inventory.get("nodeIds")
    if not isinstance(raw_node_ids, list) or not all(
        isinstance(item, str) for item in raw_node_ids
    ):
        raise ValueError("security test inventory node IDs are invalid")
    node_ids = tuple(raw_node_ids)
    test_files = sorted({node_id.split("::", 1)[0] for node_id in node_ids})
    inventory_sha256 = hashlib.sha256(
        ("\n".join(node_ids) + "\n").encode("utf-8")
    ).hexdigest()
    if (
        inventory.get("inventorySha256") != inventory_sha256
        or inventory.get("testCases") != len(node_ids)
        or inventory.get("testFiles") != len(test_files)
        or inventory.get("files") != test_files
        or any(not (root / path).is_file() for path in test_files)
    ):
        raise ValueError("security test inventory is stale or invalid")
    missing_tests = [
        marker
        for marker in REQUIRED_TEST_MARKERS
        if not any(marker in node_id for node_id in node_ids)
    ]
    if inventory.get("status") != "passed" or missing_tests:
        raise ValueError(f"security test inventory is incomplete: {missing_tests}")
    if sbom.get("bomFormat") != "CycloneDX" or not sbom.get("components"):
        raise ValueError("CycloneDX SBOM is incomplete")
    if not licenses.get("components"):
        raise ValueError("license inventory is incomplete")
    value = {
        "schemaVersion": 1,
        "release": "BalootX",
        "status": "passed",
        "source": source,
        "continuousIntegration": {
            "provider": "GitHub Actions",
            "runId": str(ci_run_id),
            "headSha": ci_head_sha,
            "conclusion": ci_conclusion,
        },
        "automatedControls": {
            "observerSafety": "covered",
            "deepLinkTamperResistance": "covered",
            "authenticationReplayResistance": "covered",
            "tenantAndRowIsolationContracts": "covered",
            "pluginSignatureAndHashVerification": "covered",
            "pluginFailureIsolationAndTimeouts": "covered",
            "unknownConfigurationRejection": "covered",
        },
        "inputArtifacts": {
            key: artifact_receipt(root, relative)
            for key, relative in INPUT_ARTIFACTS.items()
        },
        "pluginPolicy": required_policy,
        "testInventory": {
            "path": "release/test-inventory.json",
            "cases": int(inventory["testCases"]),
            "inventorySha256": inventory["inventorySha256"],
            "requiredMarkers": list(REQUIRED_TEST_MARKERS),
        },
        "supplyChain": {
            "sbomPath": "release/sbom.cdx.json",
            "sbomFormat": sbom["bomFormat"],
            "sbomSpecVersion": sbom["specVersion"],
            "componentCount": len(sbom["components"]),
            "licenseInventoryPath": "release/license-inventory.json",
            "licenseComponentCount": len(licenses["components"]),
        },
        "deploymentSecurity": "not_evaluated_deployment_frozen",
        "independentPenetrationTest": "not_performed",
        "proofBoundary": (
            "Automated source and contract security evidence is bound to the recorded "
            "source identity. CI matches its Git-head base; a worktree snapshot does "
            "not claim that uncommitted bytes ran in CI and cannot satisfy the clean-"
            "source release gate. This is not an independent penetration test, external "
            "security certification, or deployment security proof."
        ),
    }
    value["evidenceSha256"] = canonical_sha256(value)
    return value


def verify(value: dict[str, Any], *, root: Path = ROOT) -> None:
    _verify_against_source(value, root=root, source=git_identity(root))


def verify_packaged(
    value: dict[str, Any],
    *,
    root: Path,
    source_identity: Mapping[str, Any],
) -> None:
    _verify_against_source(value, root=root, source=source_identity)


def _verify_against_source(
    value: dict[str, Any],
    *,
    root: Path,
    source: Mapping[str, Any],
) -> None:
    payload = {
        key: item for key, item in value.items() if key != "evidenceSha256"
    }
    if value.get("evidenceSha256") != canonical_sha256(payload):
        raise ValueError("security evidence hash mismatch")
    recorded_source = value.get("source")
    if not isinstance(recorded_source, dict) or canonical_sha256(
        recorded_source
    ) != canonical_sha256(
        source
    ):
        raise ValueError("security evidence source identity is stale")
    continuous_integration = value.get("continuousIntegration")
    if (
        not isinstance(continuous_integration, dict)
        or continuous_integration.get("headSha") != source.get("revision")
        or continuous_integration.get("conclusion") != "success"
    ):
        raise ValueError("security evidence CI identity is stale or failed")
    expected = build(
        root=root,
        ci_run_id=str(continuous_integration.get("runId") or ""),
        ci_head_sha=str(continuous_integration.get("headSha") or ""),
        ci_conclusion=str(continuous_integration.get("conclusion") or ""),
        source_identity=source,
    )
    if value != expected:
        raise ValueError("security evidence content is stale or invalid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci-run-id", required=True)
    parser.add_argument("--ci-head-sha", required=True)
    parser.add_argument("--ci-conclusion", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    value = build(
        ci_run_id=args.ci_run_id,
        ci_head_sha=args.ci_head_sha,
        ci_conclusion=args.ci_conclusion,
    )
    atomic_write_text(
        args.output,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(args.output)


if __name__ == "__main__":
    main()
