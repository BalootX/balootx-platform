#!/usr/bin/env python3
"""Aggregate exact, bounded performance evidence without overclaiming."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from baloot_platform.benchmark import canonical_sha256
from baloot_platform.config import build_bundle, platform_root
from baloot_platform.configuration import EffectiveRuntimeConfigV2
from baloot_platform.filesystem import atomic_write_text
from baloot_platform.research.contracts import ResearchConfigurationAuthorityV1
from baloot_platform.research.performance_stats import (
    qualified_scenario_identity_hash_v1,
)
from baloot_platform.research.proof_service import ScenarioProofBudgetRegistryV1
from baloot_platform.research.result_authority import ExecutionReceiptV1
from baloot_platform.research.scenario_result import canonical_json_bytes
from baloot_platform.release_scope import (
    SOURCE_EXCLUDED_PATHS,
    SOURCE_SCOPE,
    source_pathspecs,
)


ROOT = platform_root()
OUTPUT = ROOT / "release/performance-benchmark.json"
SEARCH = ROOT / "release/search-performance.json"
RESEARCH_V4 = ROOT / "release/research-scenario-v4-performance.json"
ORTOOLS_RESEARCH_V4 = ROOT / "release/ortools-research-v3-performance.json"
BASELINE = ROOT / "release/performance-benchmark-v20.json"
MEDIA = ROOT / "release/media-probe.json"
PROOF_BUDGETS = ROOT / "configs/research/proof_budgets_v1.yaml"
SEARCH_ORTOOLS_BUNDLE_SOURCE = "configs/bundle_ortools_research_native_v3.yaml"
ORTOOLS_BENCHMARK_BUNDLE_SOURCE = (
    "configs/bundle_ortools_research_native_v3_1.yaml"
)
ORTOOLS_STAGE_TIMINGS = {
    "validationEstimate": {"validation", "estimate", "total"},
    "originalFlagshipContradiction": {
        "validation",
        "estimate",
        "preflight",
        "total",
    },
    "exactNormalization": {"exactNormalization", "total"},
    "fixedAlternateProof": {"graph", "proof", "total"},
    "alternateFirstQualifiedResult": {
        "validation",
        "estimate",
        "construction",
        "tailConstruction",
        "proofBoundary",
        "prefixLift",
        "proof",
        "qualification",
        "compatibility",
        "total",
    },
    "alternateFiveVariants": {
        "validation",
        "estimate",
        "construction",
        "tailConstruction",
        "proofBoundary",
        "prefixLift",
        "proof",
        "qualification",
        "compatibility",
        "total",
    },
    "cancellationAcknowledgement": {"cancellationAcknowledgement", "total"},
}


def _ortools_raw_sample_receipts_are_valid(
    operation_name: str,
    operation: dict[str, Any],
) -> bool:
    try:
        for phase in ("cold", "warm"):
            phase_evidence = dict(operation[phase])
            samples = list(phase_evidence["samples"])
            receipt_hashes = list(phase_evidence["executionReceiptHashes"])
            if len(samples) != int(phase_evidence["latencyMs"]["sampleCount"]):
                return False
            if len(receipt_hashes) != len(samples):
                return False
            for sample, receipt_hash in zip(samples, receipt_hashes, strict=True):
                sample = dict(sample)
                receipt = ExecutionReceiptV1.from_dict(
                    dict(sample["executionReceipt"])
                ).to_dict()
                if (
                    receipt["sha256"] != receipt_hash
                    or receipt["semanticResultHash"] != operation["semanticHash"]
                    or receipt["machine"].get("contextHash")
                    != operation["contextHash"]
                    or set(receipt["stageTimings"])
                    != ORTOOLS_STAGE_TIMINGS[operation_name]
                ):
                    return False
                if operation_name == "exactNormalization":
                    resources = dict(sample["resources"])
                    authority = {
                        field: resources.get(field)
                        for field in (
                            "normalizationBackendId",
                            "normalizationReceiptHash",
                            "normalizationModelHash",
                        )
                    }
                    if (
                        authority["normalizationBackendId"] != "ortools_cp_sat_v3"
                        or any(
                            type(authority[field]) is not str
                            or len(authority[field]) != 64
                            or any(
                                character not in "0123456789abcdef"
                                for character in authority[field]
                            )
                            for field in (
                                "normalizationReceiptHash",
                                "normalizationModelHash",
                            )
                        )
                        or any(
                            receipt["machine"].get(field) != value
                            for field, value in authority.items()
                        )
                    ):
                        return False
                if operation_name == "alternateFiveVariants":
                    campaign = dict(sample["resources"]["campaignReceipt"])
                    coordinator = dict(campaign["coordinatorReceipt"])
                    if coordinator.get("workerDeclaration") != {
                        "shardWorkerCount": 5,
                        "cpSatWorkersPerSolve": 1,
                    }:
                        return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_identity(root: Path) -> dict[str, Any]:
    """Bind authored Git-visible inputs; downstream artifacts use own receipts."""

    def run(*args: str) -> bytes:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout

    revision = run("rev-parse", "HEAD").decode("ascii").strip()
    tree = run("rev-parse", "HEAD^{tree}").decode("ascii").strip()
    pathspecs = source_pathspecs()
    status = run(
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        ".",
        *pathspecs,
    )
    identity = {
        "revision": revision,
        "tree": tree,
        "scope": SOURCE_SCOPE,
        "excludedPaths": list(SOURCE_EXCLUDED_PATHS),
    }
    if not status:
        return {
            **identity,
            "binding": "exact_git_head",
        }

    listed = run(
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        ".",
        *pathspecs,
    )
    indexed = run(
        "ls-files",
        "-s",
        "-z",
        "--cached",
        "--",
        ".",
        *pathspecs,
    )
    index_modes = {}
    for record in indexed.split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        index_modes[encoded_path.decode("utf-8", errors="surrogateescape")] = (
            metadata.split(b" ", 1)[0].decode("ascii")
        )

    entries = []
    for encoded_path in sorted(value for value in listed.split(b"\0") if value):
        relative = encoded_path.decode("utf-8", errors="surrogateescape")
        logical = PurePosixPath(relative)
        if logical.is_absolute() or ".." in logical.parts:
            raise ValueError(f"unsafe Git worktree path: {relative}")
        path = root.joinpath(*logical.parts)
        entry: dict[str, Any] = {
            "path": logical.as_posix(),
            "indexMode": index_modes.get(relative),
        }
        if path.is_symlink():
            target = os.readlink(path).encode("utf-8", errors="surrogateescape")
            entry.update(
                {
                    "kind": "symlink",
                    "bytes": len(target),
                    "sha256": hashlib.sha256(target).hexdigest(),
                }
            )
        elif path.is_file():
            entry.update(
                {
                    "kind": "file",
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        elif not path.exists():
            entry["kind"] = "missing"
        else:
            raise ValueError(f"unsupported Git worktree entry: {relative}")
        entries.append(entry)

    snapshot = {
        "schemaVersion": 1,
        "scope": SOURCE_SCOPE,
        "excludedPaths": list(SOURCE_EXCLUDED_PATHS),
        "statusSha256": hashlib.sha256(status).hexdigest(),
        "files": entries,
    }
    return {
        **identity,
        "binding": "git_head_plus_worktree_snapshot",
        "worktreeSnapshot": {
            "algorithm": "sha256",
            "scope": snapshot["scope"],
            "excludedPaths": snapshot["excludedPaths"],
            "fileCount": len(entries),
            "statusSha256": snapshot["statusSha256"],
            "sha256": canonical_sha256(snapshot),
        },
    }


def build(root: Path = ROOT) -> dict[str, Any]:
    search = load(root / SEARCH.relative_to(ROOT))
    research_v4 = load(root / RESEARCH_V4.relative_to(ROOT))
    ortools_research_v4 = load(root / ORTOOLS_RESEARCH_V4.relative_to(ROOT))
    baseline = load(root / BASELINE.relative_to(ROOT))
    media = load(root / MEDIA.relative_to(ROOT))
    source = git_identity(root)
    bundle = build_bundle(root)
    search_ortools_bundle = build_bundle(
        root,
        bundle_path=SEARCH_ORTOOLS_BUNDLE_SOURCE,
    )
    ortools_bundle = build_bundle(
        root,
        bundle_path=ORTOOLS_BENCHMARK_BUNDLE_SOURCE,
    )
    search_payload = {
        key: value for key, value in search.items() if key != "evidenceSha256"
    }
    if search.get("evidenceSha256") != canonical_sha256(search_payload):
        raise ValueError("search evidence hash mismatch")
    if search.get("status") != "passed":
        raise ValueError("search performance budgets did not pass")
    if search.get("exactSearchObservation", {}).get("completionStatus") != "exact":
        raise ValueError("search benchmark did not complete exactly")
    search_source = search.get("source") or {}
    if not isinstance(search_source, dict) or canonical_sha256(
        search_source
    ) != canonical_sha256(source):
        raise ValueError("search performance source identity is stale")
    if search.get("engine") != {
        "bundleSha256": search_ortools_bundle.sha256,
        "rulesSha256": search_ortools_bundle.rules.sha256,
    }:
        raise ValueError("search performance configuration identity is stale")
    if search.get("researchAuthority") != {
        "backendId": "ortools_cp_sat_v3",
        "qualificationAuthority": "cp_sat",
        "bundleSource": SEARCH_ORTOOLS_BUNDLE_SOURCE,
    }:
        raise ValueError("search performance Research authority is stale")
    ortools_payload = {
        key: value
        for key, value in ortools_research_v4.items()
        if key != "evidenceSha256"
    }
    operations = ortools_research_v4.get("operations") or {}
    expected_operations = {
        "validationEstimate",
        "originalFlagshipContradiction",
        "exactNormalization",
        "fixedAlternateProof",
        "alternateFirstQualifiedResult",
        "alternateFiveVariants",
        "cancellationAcknowledgement",
    }
    benchmark_profile = dict(ortools_research_v4.get("benchmarkProfile") or {})
    profile_hash = benchmark_profile.pop("sha256", None)
    scenario_requests = dict(ortools_research_v4.get("scenarioRequests") or {})
    resource_boundary = dict(
        ortools_research_v4.get("resourceMeasurementBoundary") or {}
    )
    resource_profile_id = resource_boundary.get("processLifetimeMemoryProfileId")
    resource_profiles = ortools_bundle.component("challenge")["scenarioResearchV4"][
        "proofBudgets"
    ]["profiles"]
    expected_resource_boundary = {
        "conflictsAndBranches": "candidate_stage_receipts_only",
        "tailConstructionCounters": "unavailable",
        "maximumConflicts": "observed_gate_not_executor_enforcement_proof",
        "peakMemory": "process_lifetime_peak_rss",
        "processLifetimeMemoryProfileId": resource_profile_id,
        "cache": "executor_receipt_cache_bytes",
        "cancellation": "cooperative_acknowledgement",
    }
    expected_requests = {
        "original": (
            "examples/research-scenario-v4-flagship-exact-infeasible.json",
            "original_exact_structural_contradiction",
        ),
        "alternate": (
            "examples/research-scenario-v4-flagship-"
            "alternate-target-wins-six-search-five.json",
            "alternate_target_team_wins_trick_six",
        ),
    }
    requests_valid = set(scenario_requests) == set(expected_requests)
    for name, (recipe_source, classification) in expected_requests.items():
        identity = dict(scenario_requests.get(name) or {})
        request_hashes = [
            identity.get(field)
            for field in ("sourceSha256", "scenarioHash", "proofHash")
        ]
        requests_valid = requests_valid and (
            set(identity)
            == {
                "source",
                "sourceSha256",
                "recipeCode",
                "scenarioHash",
                "proofHash",
                "classification",
            }
            and identity.get("source") == recipe_source
            and identity.get("classification") == classification
            and str(identity.get("recipeCode") or "").startswith("RCP-")
            and all(
                type(value) is str
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in request_hashes
            )
        )
    if (
        ortools_research_v4.get("evidenceSha256")
        != canonical_sha256(ortools_payload)
        or ortools_research_v4.get("status") != "passed"
        or canonical_sha256(ortools_research_v4.get("source") or {})
        != canonical_sha256(source)
        or ortools_research_v4.get("researchAuthority")
        != {
            "backendId": "ortools_cp_sat_v3",
            "qualificationAuthority": "cp_sat",
            "bundleId": "baloot_ortools_research_native_v3",
            "sourceBundleHash": ortools_bundle.sha256,
            "rulesHash": ortools_bundle.rules.sha256,
        }
        or profile_hash != canonical_sha256(benchmark_profile)
        or not requests_valid
        or resource_profile_id not in resource_profiles
        or resource_boundary != expected_resource_boundary
        or set(operations) != expected_operations
        or any(
            item.get("passed") is not True
            or dict(item.get("resourceGate") or {}).get("passed") is not True
            or item.get("sha256")
            != canonical_sha256(
                {key: value for key, value in item.items() if key != "sha256"}
            )
            or int((item.get("cold") or {}).get("latencyMs", {}).get("sampleCount", 0))
            < 2
            or int((item.get("warm") or {}).get("latencyMs", {}).get("sampleCount", 0))
            < 2
            or set(
                dict((item.get("cold") or {}).get("stageTimingsSeconds") or {})
            )
            != ORTOOLS_STAGE_TIMINGS[name]
            or set(
                dict((item.get("warm") or {}).get("stageTimingsSeconds") or {})
            )
            != ORTOOLS_STAGE_TIMINGS[name]
            or not _ortools_raw_sample_receipts_are_valid(name, dict(item))
            or any(
                int(distribution.get("sampleCount", 0)) < 2
                for phase in ("cold", "warm")
                for distribution in dict(
                    (item.get(phase) or {}).get("stageTimingsSeconds") or {}
                ).values()
            )
            for name, item in operations.items()
        )
    ):
        raise ValueError("OR-Tools Research V4 performance authority is stale")
    research_payload = {
        key: value for key, value in research_v4.items() if key != "evidenceSha256"
    }
    if research_v4.get("evidenceSha256") != canonical_sha256(research_payload):
        raise ValueError("Research V4 performance evidence hash mismatch")
    try:
        authority = ResearchConfigurationAuthorityV1.build(
            research_v4.get("configurationAuthority") or {}
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Research V4 configuration authority is invalid") from exc
    snapshot = research_v4.get("configurationSnapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("Research V4 configuration snapshot is missing")
    snapshot_bytes = canonical_json_bytes(snapshot)
    snapshot_artifact = dict(authority.snapshot_artifact)
    if snapshot_artifact.get("sha256") != hashlib.sha256(
        snapshot_bytes
    ).hexdigest() or snapshot_artifact.get("bytes") != len(snapshot_bytes):
        raise ValueError("Research V4 configuration snapshot is stale")
    try:
        effective_snapshot = EffectiveRuntimeConfigV2.from_dict(snapshot)
    except (TypeError, ValueError) as exc:
        raise ValueError("Research V4 configuration snapshot is invalid") from exc
    snapshot_bundle = dict(snapshot["bundle"])
    if (
        effective_snapshot.content_sha256 != authority.effective_runtime_config_hash
        or snapshot["sourceEffectiveHash"] != authority.effective_config_hash
        or snapshot_bundle["revisionId"] != authority.bundle_revision_id
        or snapshot_bundle["bundleId"] != authority.bundle_id
        or snapshot_bundle["version"] != authority.bundle_version
        or snapshot["componentRevisions"]
        != {
            family: dict(revision)
            for family, revision in authority.component_revisions.items()
        }
    ):
        raise ValueError("Research V4 configuration snapshot authority mismatch")
    profile = dict(bundle.bundle_profile["profile"])
    expected_research_bundle = {
        "revisionId": authority.bundle_revision_id,
        "id": str(profile["id"]),
        "version": str(profile["version"]),
        "sha256": bundle.sha256,
        "semanticRuntimeSha256": authority.semantic_bundle_hash,
        "rulesSha256": bundle.rules.sha256,
        "componentSha256": {
            family: component.sha256
            for family, component in sorted(bundle.components.items())
        },
    }
    if (
        research_v4.get("schemaVersion") != 2
        or research_v4.get("kind") != "research_scenario_v4_performance"
        or research_v4.get("status") != "passed"
        or canonical_sha256(research_v4.get("source") or {}) != canonical_sha256(source)
        or research_v4.get("bundle") != expected_research_bundle
        or research_v4.get("proofBudgetProfileSha256")
        != hashlib.sha256(
            (root / PROOF_BUDGETS.relative_to(ROOT)).read_bytes()
        ).hexdigest()
    ):
        raise ValueError("Research V4 performance authority is stale")
    research_workloads = research_v4.get("workloads") or {}
    exact_no_salvage = research_workloads.get("exactNoSalvage") or {}
    exact_unique = research_workloads.get("exactUniqueSavingMove") or {}
    generated = research_workloads.get("generatedSunHundredNoSalvage") or {}
    generated_funnel = generated.get("funnel") or {}
    generated_result = generated.get("result") or {}
    generated_result_hash = generated_result.get("resultHash")
    generated_scenario_hash = generated.get("scenarioIdentityHash")
    summary = research_v4.get("summary") or {}
    proof_budgets = ScenarioProofBudgetRegistryV1.load(
        root / PROOF_BUDGETS.relative_to(ROOT)
    )
    used_budget_profiles = {
        str((item.get("proofBudget") or {}).get("profileId"))
        for item in research_workloads.values()
    }
    expected_budget_authority = {
        "registrySha256": proof_budgets.sha256,
        "profiles": {
            profile_id: proof_budgets.get(profile_id).sha256
            for profile_id in sorted(used_budget_profiles)
        },
    }
    try:
        expected_scenario_hashes = {
            name: qualified_scenario_identity_hash_v1(
                {
                    "scenarioProjection": {
                        "scenarioTemplateId": item.get("scenarioTemplateId"),
                        "exactStateId": item.get("exactStateId"),
                    }
                }
            )
            for name, item in research_workloads.items()
        }
    except ValueError as exc:
        raise ValueError("Research V4 performance workloads are incomplete") from exc
    if (
        set(research_workloads)
        != {
            "exactNoSalvage",
            "exactUniqueSavingMove",
            "generatedSunHundredNoSalvage",
        }
        or exact_no_salvage.get("origin") != "exact_input"
        or exact_no_salvage.get("status") != "exact_complete"
        or exact_no_salvage.get("winningRootActions") != 0
        or exact_unique.get("origin") != "exact_input"
        or exact_unique.get("status") != "exact_complete"
        or exact_unique.get("winningRootActions") != 1
        or generated.get("origin") != "generated_search"
        or generated.get("status") != "satisfied"
        or generated.get("observerSafe") is not True
        or generated_funnel
        != {
            "attempted": 1,
            "structurallyFeasible": 1,
            "legalReplay": 1,
            "semanticSatisfied": 1,
            "proofCompleted": 1,
            "qualified": 1,
        }
        or generated_result.get("disposition") != "qualified"
        or not isinstance(generated_result_hash, str)
        or len(generated_result_hash) != 64
        or any(
            item.get("scenarioIdentityHash") != expected_scenario_hashes[name]
            for name, item in research_workloads.items()
        )
        or any(
            (item.get("reproducibility") or {}).get("matched") is not True
            for item in research_workloads.values()
        )
        or any(
            (item.get("proofBudget") or {}).get("registrySha256")
            != proof_budgets.sha256
            or (item.get("proofBudget") or {}).get("profileSha256")
            != proof_budgets.get(
                str((item.get("proofBudget") or {}).get("profileId"))
            ).sha256
            for item in research_workloads.values()
        )
        or research_v4.get("proofBudgetAuthority") != expected_budget_authority
        or summary.get("workloadCount") != len(research_workloads)
        or summary.get("generatedFunnel") != generated_funnel
        or summary.get("qualifiedGeneratedResultCount") != 1
        or summary.get("qualifiedDistinctGeneratedScenarioCount") != 1
        or summary.get("qualifiedDistinctGeneratedScenarioHashes")
        != [generated_scenario_hash]
        or summary.get("qualifiedDistinctGeneratedResultHashes")
        != [generated_result_hash]
    ):
        raise ValueError("Research V4 performance workloads are incomplete")
    if baseline.get("status") != "passed":
        raise ValueError("V20 locked baseline did not pass")
    if media.get("status") != "passed":
        raise ValueError("media probe did not pass")
    invalid_media = []
    for item in media.get("media", []):
        video = item.get("mediaProbe", {}).get("video", {})
        width = item.get("width", video.get("width", 0))
        height = item.get("height", video.get("height", 0))
        if (int(width), int(height)) != (
            1080,
            1920,
        ):
            invalid_media.append(item.get("path"))
    if invalid_media:
        raise ValueError(f"non-vertical release media: {invalid_media}")
    value = {
        "schemaVersion": 1,
        "release": "BalootX",
        "status": "passed",
        "source": source,
        "bundleSha256": bundle.sha256,
        "rulesSha256": bundle.rules.sha256,
        "search": {
            "path": SEARCH.relative_to(ROOT).as_posix(),
            "evidenceSha256": search["evidenceSha256"],
            "measurementSource": search["source"],
            "hardware": search["hardware"],
            "measurements": search["measurements"],
            "throughput": search["throughput"],
            "exactSearchObservation": search["exactSearchObservation"],
        },
        "researchV4": {
            "path": RESEARCH_V4.relative_to(ROOT).as_posix(),
            "evidenceSha256": research_v4["evidenceSha256"],
            "source": research_v4["source"],
            "bundle": research_v4["bundle"],
            "configurationAuthority": research_v4["configurationAuthority"],
            "configurationSnapshotArtifact": snapshot_artifact,
            "proofBudgetProfileSha256": research_v4["proofBudgetProfileSha256"],
            "proofBudgetAuthority": research_v4["proofBudgetAuthority"],
            "environment": research_v4["environment"],
            "workloads": research_workloads,
            "summary": research_v4["summary"],
        },
        "ortoolsResearchV4": {
            "path": ORTOOLS_RESEARCH_V4.relative_to(ROOT).as_posix(),
            "evidenceSha256": ortools_research_v4["evidenceSha256"],
            "source": ortools_research_v4["source"],
            "researchAuthority": ortools_research_v4["researchAuthority"],
            "scenarioRequests": scenario_requests,
            "resourceMeasurementBoundary": resource_boundary,
            "benchmarkProfile": ortools_research_v4["benchmarkProfile"],
            "environment": ortools_research_v4["environment"],
            "operations": operations,
        },
        "lockedV20Baseline": {
            "path": BASELINE.relative_to(ROOT).as_posix(),
            "bundleSha256": baseline["bundleHash"],
            "environment": baseline["environment"],
            "measurements": baseline["measurements"],
        },
        "media": {
            "probePath": MEDIA.relative_to(ROOT).as_posix(),
            "items": media["media"],
            "aspectRatio": "9:16",
        },
        "promotionEvaluation": {
            "automaticPromotion": False,
            "nativeTwoTimesTarget": "not_comparable_across_declared_environments",
            "videoThirtyPercentTarget": "not_yet_measured_on_locked_same_hardware",
            "decision": "retain_prior_defaults_until_comparable_evidence",
        },
        "proofBoundary": (
            "The declared local budgets, exact-search completion, and release-media "
            "shape passed. The 2x native and 30% render improvement targets remain "
            "unproven until locked same-hardware comparisons are recorded."
        ),
    }
    value["evidenceSha256"] = canonical_sha256(value)
    return value


def main() -> None:
    value = build()
    atomic_write_text(
        OUTPUT,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
