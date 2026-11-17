#!/usr/bin/env python3
# ruff: noqa: E402
"""Build the deterministic, fail-closed release-admission manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY_ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from baloot_platform.benchmark import (
    benchmark_configuration_sha256,
    canonical_sha256,
    load_candidate_jsonl,
)
from baloot_platform.config import build_bundle, platform_root
from baloot_platform.research.result_authority import (
    validate_ortools_result_sections_v1,
)
from baloot_platform.release_scope import (
    SOURCE_EXCLUDED_PATHS,
    SOURCE_SCOPE,
    source_pathspecs,
)
try:
    from scripts.run_verifier_container import verify_receipt as verify_verifier_receipt
except ModuleNotFoundError:
    from run_verifier_container import verify_receipt as verify_verifier_receipt


ROOT = platform_root()
DEFAULT_CANDIDATES = ROOT / "private-inputs/strength-candidates.jsonl"
DEFAULT_EXACT = ROOT / "release/exact-labels.json"
DEFAULT_HOLDOUT = ROOT / "release/permanent-holdout.jsonl"
DEFAULT_OUTPUT = ROOT / ".release-runtime/release/readiness.json"

REQUIRED_RELEASE_EVIDENCE = (
    "release/chrome-qa.json",
    "release/configuration-registry.json",
    "release/exact-labels.json",
    "release/intelligence-benchmark.json",
    "release/license-inventory.json",
    "release/media-probe.json",
    "release/performance-benchmark.json",
    "release/plugin-host.json",
    "release/rules-certification-candidate.json",
    "release/sbom.cdx.json",
    "release/security-report.json",
    "release/test-inventory.json",
    "release/verification-report.json",
    "release/visual-qa.json",
)
REQUIRED_RELEASE_MEDIA = (
    "outputs/professional-video-prototypes/render-bundle-hooked_flagship.json",
    "outputs/professional-video-prototypes/render-bundle-decision.json",
    "outputs/professional-video-flagship/balootx-complete-laffah-audit.mp4",
    "outputs/professional-video-flagship/balootx-complete-laffah-audit.ar.vtt",
    "outputs/professional-video-flagship/manifest.json",
    "outputs/professional-video-decision/balootx-decision-clip.mp4",
    "outputs/professional-video-decision/balootx-decision-clip.ar.vtt",
    "outputs/professional-video-decision/manifest.json",
    "outputs/scene-preview/manifest.json",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        return {"path": relative, "status": "pending"}
    return {
        "path": relative,
        "status": "passed",
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _closed(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")
    return dict(value)


def _bound_artifact(root: Path, value: Any, label: str) -> tuple[dict[str, Any], Path]:
    reference = _closed(value, {"path", "sha256"}, label)
    relative = reference["path"]
    expected = reference["sha256"]
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} path must be repository-relative")
    if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
        raise ValueError(f"{label} sha256 is invalid")
    path = (root / relative).resolve()
    try:
        canonical = path.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the repository") from exc
    if canonical != Path(relative).as_posix() or not path.is_file():
        raise ValueError(f"{label} is missing or non-canonical: {relative}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash substitution detected: {relative}")
    return {
        "path": canonical,
        "bytes": path.stat().st_size,
        "sha256": actual,
    }, path


def _json_artifact(
    root: Path, value: Any, label: str
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    artifact, path = _bound_artifact(root, value, label)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    return artifact, document, path


def _canonical_document_hash(
    document: Mapping[str, Any], field: str, label: str
) -> str:
    declared = document.get(field)
    if not isinstance(declared, str) or not _SHA256.fullmatch(declared):
        raise ValueError(f"{label} {field} is invalid")
    actual = canonical_sha256(
        {key: item for key, item in document.items() if key != field}
    )
    if declared != actual:
        raise ValueError(f"{label} internal hash substitution detected")
    return declared


def _agent_qualification_gate(root: Path, bundle) -> dict[str, Any]:
    from baloot_platform.agents.profiles import AgentProfileRegistry
    from baloot_platform.agents.qualification import FoundryAgentQualificationBundleV1

    path = root / "release" / "agent-qualification.json"
    if not path.is_file():
        return {"status": "pending", "path": "release/agent-qualification.json"}
    try:
        benchmark = bundle.component("benchmark")
        policy_id = str(benchmark["paired_comparison"]["candidate_profile_id"])
        registry = AgentProfileRegistry.from_bundle(root)
        loaded = FoundryAgentQualificationBundleV1.load(
            path,
            profile=registry.get(policy_id),
            lineup_snapshot=registry.snapshot_lineup(
                {seat: policy_id for seat in ("north", "west", "south", "east")}
            ),
            benchmark_profile=benchmark,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "path": "release/agent-qualification.json",
            "error": str(exc),
        }
    release = loaded.release
    return {
        "status": "passed" if release.expert_generation_allowed else "pending",
        "path": "release/agent-qualification.json",
        "releaseStatus": release.status,
        "expertGenerationAllowed": release.expert_generation_allowed,
        "issues": list(release.issues),
        "sha256": loaded.sha256,
    }


def _evidence_artifact(root: Path, relative: str) -> dict[str, Any]:
    artifact = _artifact(root, relative)
    if artifact["status"] != "passed":
        return artifact
    path = root / relative
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if relative.endswith("sbom.cdx.json"):
            valid = value.get("bomFormat") == "CycloneDX" and bool(
                value.get("components")
            )
        elif relative.endswith("license-inventory.json"):
            valid = bool(value.get("components"))
        elif relative.endswith("exact-labels.json"):
            valid = int(value.get("exactLabelCount", 0)) > 0 and bool(
                value.get("evidenceSha256")
            )
        elif relative.endswith("intelligence-benchmark.json"):
            bundle = build_bundle(root)
            candidates = load_candidate_jsonl(
                root / "private-inputs/strength-candidates.jsonl"
            )
            exact = json.loads(
                (root / "release/exact-labels.json").read_text(encoding="utf-8")
            )
            source_manifest_path = root / "release/source-manifest.json"
            source_manifest = json.loads(
                source_manifest_path.read_text(encoding="utf-8")
            )
            source = dict(value.get("source") or {})
            semantic = {
                "candidateDatasetSha256": value.get("candidateDatasetSha256"),
                "exactEvidenceSha256": value.get("exactEvidenceSha256"),
                "policies": {
                    key: {
                        item_key: item_value
                        for item_key, item_value in item.items()
                        if item_key != "latencyMs"
                    }
                    for key, item in dict(value.get("policies") or {}).items()
                },
            }
            valid = (
                value.get("status") == "passed"
                and value.get("kind") == "observer_safe_exact_subset_preanalysis"
                and value.get("candidateDatasetId") == candidates["datasetId"]
                and value.get("candidateDatasetSha256") == candidates["datasetSha256"]
                and value.get("exactEvidenceSha256") == exact.get("evidenceSha256")
                and value.get("rulesSha256") == bundle.rules.sha256
                and value.get("configurationSha256")
                == benchmark_configuration_sha256(bundle)
                and source.get("binding") == "local_workspace_manifest"
                and source.get("revision")
                == source_manifest.get("source", {}).get("revision")
                and source.get("tree") is None
                and source.get("workspaceManifestSha256")
                == source_manifest.get("source", {})
                .get("worktreeSnapshot", {})
                .get("sha256")
                and value.get("automaticPromotion") is False
                and value.get("promotionDecision", {}).get("decision") == "not_eligible"
                and value.get("semanticResultSha256") == canonical_sha256(semantic)
            )
        elif relative.endswith("rules-certification-candidate.json"):
            bundle = build_bundle(root)
            candidate_payload = {
                key: item for key, item in value.items() if key != "candidateSha256"
            }
            valid = (
                value.get("status") == "passed"
                and value.get("official") is False
                and str(value.get("certificationStatus", "")).startswith("candidate")
                and value.get("automaticRuleActivation") is False
                and value.get("bundleSha256") == bundle.sha256
                and value.get("rulesSha256") == bundle.rules.sha256
                and value.get("unresolvedTraceabilityClauses") == []
                and value.get("candidateSha256") == canonical_sha256(candidate_payload)
            )
        elif relative.endswith("security-report.json"):
            try:
                from scripts.generate_security_report import verify as verify_security
            except ModuleNotFoundError:
                from generate_security_report import verify as verify_security
            verify_security(value, root=root)
            valid = True
        else:
            valid = value.get("status") == "passed"
        if not valid:
            raise ValueError("artifact does not declare an accepted evidence status")
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            **artifact,
            "status": "failed",
            "reason": str(exc),
        }
    return artifact


def _portable_path(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def _load_holdout_jsonl(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("holdout JSONL is empty or invalid")
    return {**rows[0], "positions": rows[1:]}


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _source_gate(root: Path) -> dict[str, Any]:
    try:
        commit = _git(root, "rev-parse", "HEAD")
        tree = _git(root, "rev-parse", "HEAD^{tree}")
        dirty = _git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            ".",
            *source_pathspecs(),
        )
    except RuntimeError as exc:
        return {"status": "failed", "reason": str(exc)}
    return {
        "status": "passed" if not dirty else "pending",
        "gitCommit": commit,
        "gitTree": tree,
        "scope": SOURCE_SCOPE,
        "excludedPaths": list(SOURCE_EXCLUDED_PATHS),
        "clean": not dirty,
        "reason": None
        if not dirty
        else "release product inputs are not frozen and clean",
    }


def _candidate_gate(
    path: Path,
    minimum: int,
    *,
    root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    display_path = _portable_path(path, root)
    if not path.is_file():
        return ({"status": "pending", "path": display_path}, None)
    try:
        candidates = load_candidate_jsonl(path)
        count = len(candidates["positions"])
        if count < minimum:
            raise ValueError(f"candidate count {count} is below policy {minimum}")
    except (KeyError, TypeError, ValueError) as exc:
        return ({"status": "failed", "path": display_path, "reason": str(exc)}, None)
    return (
        {
            "status": "passed",
            "path": display_path,
            "datasetId": candidates["datasetId"],
            "datasetSha256": candidates["datasetSha256"],
            "positionCount": count,
            "sha256": _sha256(path),
        },
        candidates,
    )


def _exact_gate(
    path: Path,
    candidates: dict[str, Any] | None,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    display_path = _portable_path(path, root)
    if not path.is_file():
        return {"status": "pending", "path": display_path}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        payload = {key: item for key, item in value.items() if key != "evidenceSha256"}
        if value.get("evidenceSha256") != canonical_sha256(payload):
            raise ValueError("exact-label evidence hash mismatch")
        if candidates is None or (
            value.get("candidateDatasetId") != candidates.get("datasetId")
            or value.get("candidateDatasetSha256") != candidates.get("datasetSha256")
        ):
            raise ValueError("exact-label candidate identity mismatch")
        positions = list(value.get("positions") or ())
        if value.get("exactLabelCount") != len(positions) or not positions:
            raise ValueError("exact-label count is invalid")
        if any(item.get("searchCompletion") != "exact" for item in positions):
            raise ValueError("non-exact result is present in exact evidence")
        candidate_by_id = {
            str(item["candidateId"]): item for item in candidates["positions"]
        }
        seen: set[str] = set()
        for position in positions:
            candidate_id = str(position.get("candidateId", ""))
            candidate = candidate_by_id.get(candidate_id)
            if not candidate_id or candidate_id in seen or candidate is None:
                raise ValueError("exact-label candidate identity is invalid")
            seen.add(candidate_id)
            if (
                position.get("stateSha256") != candidate.get("stateSha256")
                or position.get("observer") != candidate.get("observer")
                or list(position.get("legalActions") or ())
                != list(candidate.get("legalActions") or ())
            ):
                raise ValueError("exact-label source state mismatch")
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"status": "failed", "path": display_path, "reason": str(exc)}
    return {
        "status": "passed",
        "path": display_path,
        "exactLabelCount": len(positions),
        "evidenceSha256": value["evidenceSha256"],
        "sha256": _sha256(path),
    }


def _holdout_gate(
    path: Path,
    candidates: dict[str, Any] | None,
    *,
    minimum_positions: int,
    minimum_authorities: int,
    minimum_agreement: float,
    root: Path | None = None,
) -> dict[str, Any]:
    display_path = _portable_path(path, root)
    if not path.is_file():
        return {
            "status": "pending",
            "path": display_path,
            "reason": "independent adjudication has not been finalized",
        }
    try:
        value = _load_holdout_jsonl(path)
        positions = list(value.get("positions") or ())
        provenance = dict(value.get("labelProvenance") or {})
        authorities = list(value.get("authorityIds") or ())
        reviewers = list(value.get("reviewerIds") or ())
        manifest_payload = {
            key: item for key, item in value.items() if key != "manifestSha256"
        }
        if value.get("manifestSha256") != canonical_sha256(manifest_payload):
            raise ValueError("holdout manifest hash mismatch")
        if candidates is None or (
            value.get("candidateDatasetId") != candidates.get("datasetId")
            or value.get("candidateDatasetSha256") != candidates.get("datasetSha256")
        ):
            raise ValueError("holdout candidate identity mismatch")
        if int(value.get("positionCount", -1)) != len(positions):
            raise ValueError("holdout position count mismatch")
        if value.get("datasetSha256") != canonical_sha256(positions):
            raise ValueError("holdout dataset hash mismatch")
        if len(positions) < minimum_positions:
            raise ValueError("holdout position count is below policy")
        actual_provenance = {
            "exact": sum(item.get("labelSource") == "exact" for item in positions),
            "independentlyAdjudicated": sum(
                item.get("labelSource") == "independent_adjudication"
                for item in positions
            ),
        }
        if provenance != actual_provenance or sum(provenance.values()) != len(
            positions
        ):
            raise ValueError("holdout label provenance count mismatch")
        candidate_by_id = {
            str(item["candidateId"]): item for item in candidates["positions"]
        }
        seen: set[str] = set()
        for position in positions:
            position_id = str(position.get("positionId", ""))
            candidate = candidate_by_id.get(position_id)
            if not position_id or position_id in seen or candidate is None:
                raise ValueError("holdout position identity is invalid")
            seen.add(position_id)
            if (
                position.get("stateSha256") != candidate.get("stateSha256")
                or position.get("observer") != candidate.get("observer")
                or list(position.get("legalActions") or ())
                != list(candidate.get("legalActions") or ())
            ):
                raise ValueError("holdout position source state mismatch")
        if (
            len(set(reviewers)) < minimum_authorities
            or len(set(authorities)) < minimum_authorities
        ):
            raise ValueError("holdout lacks independent review authorities")
        if float(value.get("agreementRate", -1.0)) < minimum_agreement:
            raise ValueError("holdout agreement is below policy")
        if value.get("automaticPromotion") is not False:
            raise ValueError("holdout must not authorize automatic promotion")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {"status": "failed", "path": display_path, "reason": str(exc)}
    return {
        "status": "passed",
        "path": display_path,
        "datasetId": value["datasetId"],
        "datasetSha256": value["datasetSha256"],
        "manifestSha256": value["manifestSha256"],
        "positionCount": len(positions),
        "agreementRate": value["agreementRate"],
        "reviewerCount": len(set(reviewers)),
        "authorityCount": len(set(authorities)),
        "labelProvenance": provenance,
        "sha256": _sha256(path),
        "proofBoundary": value.get("proofBoundary"),
    }


def _media_gate(root: Path, width: int, height: int) -> dict[str, Any]:
    artifacts = [_artifact(root, relative) for relative in REQUIRED_RELEASE_MEDIA]
    probe_path = root / "release/media-probe.json"
    invalid: list[str] = []
    if probe_path.is_file():
        try:
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            if probe.get("schemaVersion") != 2 or probe.get("status") != "passed":
                raise ValueError("media probe is not the accepted schema or status")
            media = list(probe.get("media") or ())
            if not media:
                invalid.append("media probe has no release media")
            for item in media:
                video = dict((item.get("mediaProbe") or {}).get("video") or {})
                if (int(video.get("width", 0)), int(video.get("height", 0))) != (
                    width,
                    height,
                ):
                    invalid.append(str(item.get("path", "unknown")))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            invalid.append(f"invalid media probe: {exc}")
    pending = [item["path"] for item in artifacts if item["status"] != "passed"]
    status = "failed" if invalid else ("pending" if pending else "passed")
    return {
        "status": status,
        "requiredAspectRatio": "9:16",
        "requiredResolution": {"width": width, "height": height},
        "artifacts": artifacts,
        "missing": pending,
        "invalidAspectRatio": invalid,
        "forbiddenReleaseAspectRatios": ["16:9", "1:1"],
    }


def _archive_source_manifest(
    archive_path: Path,
    *,
    source: Mapping[str, str],
    label: str,
) -> tuple[str, dict[str, Any]]:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = [
                name
                for name in archive.namelist()
                if name.endswith("/release/source-manifest.json")
            ]
            if len(names) != 1:
                raise ValueError(f"{label} must contain one source manifest")
            raw = archive.read(names[0])
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{label} is not a valid repository archive") from exc
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} source manifest is invalid") from exc
    manifest_source = (
        dict(manifest.get("source") or {}) if isinstance(manifest, dict) else {}
    )
    if any(manifest_source.get(key) != expected for key, expected in source.items()):
        raise ValueError(f"{label} source manifest does not bind the frozen source")
    return hashlib.sha256(raw).hexdigest(), manifest


def _archive_launcher_receipt(
    root: Path,
    value: Any,
    *,
    archive_path: Path,
    archive_sha256: str,
    report_path: Path,
    report: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, receipt, _ = _json_artifact(root, value, f"archive {label} receipt")
    expected_fields = {
        "schemaVersion",
        "taskName",
        "archive",
        "expectedArchiveSha256",
        "actualArchiveSha256",
        "workerPid",
        "verifierPid",
        "affinityMask",
        "nativeThreadLimit",
        "startedAt",
        "finishedAt",
        "exitCode",
        "status",
        "reportStatus",
        "report",
        "stdout",
        "stderr",
        "workspace",
        "launcherError",
        "cleanupErrors",
    }
    _closed(receipt, expected_fields, f"archive {label} receipt")
    workspace = Path(str(receipt["workspace"])).resolve()
    clean_root = Path(str(report.get("cleanRoot") or "")).resolve()
    if (
        receipt["schemaVersion"] != 1
        or receipt["status"] != "passed"
        or receipt["reportStatus"] != "passed"
        or receipt["exitCode"] != 0
        or receipt["launcherError"] is not None
        or receipt["cleanupErrors"] != []
        or str(receipt["expectedArchiveSha256"]).lower() != archive_sha256
        or str(receipt["actualArchiveSha256"]).lower() != archive_sha256
        or Path(str(receipt["archive"])).resolve() != archive_path.resolve()
        or Path(str(receipt["report"])).resolve() != report_path.resolve()
        or not isinstance(receipt["taskName"], str)
        or not receipt["taskName"].strip()
        or type(receipt["workerPid"]) is not int
        or receipt["workerPid"] < 1
        or type(receipt["verifierPid"]) is not int
        or receipt["verifierPid"] < 1
        or not workspace.is_dir()
        or not clean_root.is_dir()
        or workspace not in clean_root.parents
        or any(
            not Path(str(receipt[field])).resolve().is_file()
            for field in ("stdout", "stderr")
        )
    ):
        raise ValueError(f"archive {label} launcher receipt is stale or invalid")
    return artifact, {
        "taskName": receipt["taskName"],
        "workspace": str(workspace),
        "cleanRoot": str(clean_root),
    }


def _require_archived_artifact(
    manifest: Mapping[str, Any], artifact: Mapping[str, Any], label: str
) -> None:
    files = manifest.get("files")
    metadata = (
        dict(files.get(artifact.get("path")) or {})
        if isinstance(files, Mapping)
        else {}
    )
    if metadata != {
        "bytes": artifact.get("bytes"),
        "sha256": artifact.get("sha256"),
    }:
        raise ValueError(f"{label} is not bound into the frozen source archive")


def _variant_handoff(
    root: Path,
    value: Any,
    *,
    seen_ids: set[str],
    seen_results: set[str],
    seen_videos: set[str],
) -> dict[str, Any]:
    item = _closed(
        value,
        {"variantId", "recipe", "result", "video", "videoManifest"},
        "variant",
    )
    variant_id = item["variantId"]
    if not isinstance(variant_id, str) or not variant_id or variant_id in seen_ids:
        raise ValueError("variantId must be a unique non-empty string")
    result_artifact, result, _ = _json_artifact(
        root, item["result"], f"variant {variant_id} result"
    )
    recipe_artifact, recipe, _ = _json_artifact(
        root, item["recipe"], f"variant {variant_id} recipe"
    )
    try:
        result = validate_ortools_result_sections_v1(result, recipe=recipe)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"variant {variant_id} Research result authority is invalid: {exc}"
        ) from exc
    video_artifact, video_path = _bound_artifact(
        root, item["video"], f"variant {variant_id} video"
    )
    manifest_artifact, manifest, _ = _json_artifact(
        root, item["videoManifest"], f"variant {variant_id} video manifest"
    )
    result_hash = result.get("resultHash")
    result_code = result.get("resultCode")
    identity = dict(dict(result.get("provenance") or {}).get("resultIdentity") or {})
    authority = dict(result.get("researchAuthority") or {})
    compatibility = dict(result.get("runtimeCompatibility") or {})
    qualification = dict(result.get("qualification") or {})
    projection = dict(result.get("scenarioProjection") or {})
    if (
        not isinstance(result_hash, str)
        or not _SHA256.fullmatch(result_hash)
        or canonical_sha256(identity) != result_hash
        or not isinstance(result_code, str)
        or not result_code
        or authority.get("backendId") != "ortools_cp_sat_v2"
        or authority.get("qualificationAuthority") != "cp_sat"
        or compatibility.get("status") != "passed"
        or compatibility.get("bundleHash") != authority.get("bundleHash")
        or compatibility.get("traceHash") != authority.get("traceHash")
        or qualification.get("status") != "qualified"
        or qualification.get("qualified") is not True
    ):
        raise ValueError(
            f"variant {variant_id} Research result is not qualified or identity-bound"
        )
    if result_hash in seen_results or video_artifact["sha256"] in seen_videos:
        raise ValueError("variant results and videos must be genuinely distinct")
    _canonical_document_hash(
        manifest, "manifestSha256", f"variant {variant_id} video manifest"
    )
    source = dict(manifest.get("source") or {})
    video_entry = dict(dict(manifest.get("files") or {}).get("video") or {})
    if (
        manifest.get("observerSafe") is not True
        or manifest.get("deterministic") is not True
        or source.get("code") != result_code
        or source.get("publicProjectionHash") != projection.get("publicProjectionHash")
        or source.get("qualificationStatus") != "qualified"
        or source.get("publicationEligible") is not True
        or source.get("auditOnly") is not False
        or video_entry.get("name") != video_path.name
        or video_entry.get("sha256") != video_artifact["sha256"]
    ):
        raise ValueError(
            f"variant {variant_id} video is not bound to its qualified result"
        )
    seen_ids.add(variant_id)
    seen_results.add(result_hash)
    seen_videos.add(video_artifact["sha256"])
    return {
        "variantId": variant_id,
        "resultCode": result_code,
        "resultHash": result_hash,
        "recipe": recipe_artifact,
        "result": result_artifact,
        "videoManifest": manifest_artifact,
        "video": video_artifact,
    }


def build_final_audit_handoff(
    root: Path, specification: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the final deterministic cross-artifact audit handoff."""

    spec = _closed(
        specification,
        {
            "schemaVersion",
            "source",
            "wheelReceipt",
            "archives",
            "postgresProof",
            "verifier",
            "variants",
            "ownerAcceptance",
        },
        "final handoff specification",
    )
    if spec["schemaVersion"] != 1:
        raise ValueError("final handoff specification schemaVersion must be 1")
    source = _closed(spec["source"], {"revision", "tree"}, "source")
    if not all(
        isinstance(source[key], str) and re.fullmatch(r"[0-9a-f]{40,64}", source[key])
        for key in ("revision", "tree")
    ):
        raise ValueError("source revision and tree must be lowercase Git object IDs")
    source_gate = _source_gate(root)
    if source_gate.get("status") != "passed" or any(
        source_gate.get(gate_key) != source[source_key]
        for gate_key, source_key in (("gitCommit", "revision"), ("gitTree", "tree"))
    ):
        raise ValueError(
            "final handoff source is not frozen at the declared revision and tree"
        )

    wheel_receipt_artifact, wheel_receipt, _ = _json_artifact(
        root, spec["wheelReceipt"], "wheel receipt"
    )
    wheel_source = dict(wheel_receipt.get("source") or {})
    builds = list(wheel_receipt.get("builds") or [])
    if (
        wheel_receipt.get("schemaVersion") != 1
        or wheel_receipt.get("status") != "passed"
        or wheel_receipt.get("byteIdentical") is not True
        or wheel_source.get("revision") != source["revision"]
        or wheel_source.get("tree") != source["tree"]
        or len(builds) != 2
        or builds[0].get("sha256") != builds[1].get("sha256")
    ):
        raise ValueError("wheel receipt is stale or not byte-identical")
    wheel_reference = {
        "path": wheel_receipt.get("wheel"),
        "sha256": builds[0].get("sha256") if builds else None,
    }
    wheel_artifact, _ = _bound_artifact(root, wheel_reference, "wheel")

    archive_spec = _closed(spec["archives"], {"A", "B"}, "archives")
    archives: dict[str, Any] = {}
    source_manifest_hashes: set[str] = set()
    archive_hashes: set[str] = set()
    for label in ("A", "B"):
        pair = _closed(
            archive_spec[label], {"archive", "report", "receipt"}, f"archive {label}"
        )
        archive_artifact, archive_path = _bound_artifact(
            root, pair["archive"], f"archive {label}"
        )
        report_artifact, report, _ = _json_artifact(
            root, pair["report"], f"archive {label} report"
        )
        checks = list(report.get("checks") or [])
        if (
            report.get("status") != "passed"
            or report.get("archiveSha256") != archive_artifact["sha256"]
            or not checks
            or any(
                not isinstance(check, Mapping) or check.get("status") != "passed"
                for check in checks
            )
        ):
            raise ValueError(f"archive {label} report is stale or incomplete")
        manifest_hash, source_manifest = _archive_source_manifest(
            archive_path, source=source, label=f"archive {label}"
        )
        receipt_artifact, execution = _archive_launcher_receipt(
            root,
            pair["receipt"],
            archive_path=archive_path,
            archive_sha256=archive_artifact["sha256"],
            report_path=(root / report_artifact["path"]).resolve(),
            report=report,
            label=label,
        )
        archive_hashes.add(archive_artifact["sha256"])
        source_manifest_hashes.add(manifest_hash)
        archives[label] = {
            "archive": archive_artifact,
            "report": report_artifact,
            "receipt": receipt_artifact,
            "execution": execution,
            "sourceManifestSha256": manifest_hash,
            "sourceManifest": source_manifest,
        }
    if len(archive_hashes) != 1 or len(source_manifest_hashes) != 1:
        raise ValueError("archives A and B are not independently byte-identical")
    archive_sha256 = next(iter(archive_hashes))
    source_manifest_sha256 = next(iter(source_manifest_hashes))
    if any(
        len({item["execution"][field] for item in archives.values()}) != 2
        for field in ("taskName", "workspace", "cleanRoot")
    ):
        raise ValueError("archive A and B do not have independent clean-root executions")

    postgres_artifact, postgres, _ = _json_artifact(
        root, spec["postgresProof"], "PostgreSQL proof"
    )
    _canonical_document_hash(postgres, "sha256", "PostgreSQL proof")
    postgres_source = dict(dict(postgres.get("environment") or {}).get("source") or {})
    if (
        postgres.get("kind") != "balootx_live_postgres_research_proof_v1"
        or postgres.get("status") != "passed"
        or postgres_source.get("sourceManifestSha256") != source_manifest_sha256
    ):
        raise ValueError("PostgreSQL proof is stale or belongs to another archive")

    verifier_spec = _closed(
        spec["verifier"], {"imageReceipt", "runReceipt"}, "verifier"
    )
    image_artifact, image_receipt, _ = _json_artifact(
        root, verifier_spec["imageReceipt"], "verifier image receipt"
    )
    run_artifact, run_receipt, _ = _json_artifact(
        root, verifier_spec["runReceipt"], "verifier run receipt"
    )
    image_receipt = verify_verifier_receipt(image_receipt)
    run_receipt = verify_verifier_receipt(run_receipt)
    image_id = image_receipt["image"]["id"]
    if (
        image_receipt.get("kind") != "verifier_image_build"
        or image_receipt.get("status") != "passed"
        or image_receipt["terminal"]["exitCode"] != 0
        or image_receipt["archive"]["sha256"] != archive_sha256
        or image_receipt["cleanRoot"]["sourceManifest"]["sha256"]
        != source_manifest_sha256
        or run_receipt.get("kind") != "verifier_container_run"
        or run_receipt.get("status") != "passed"
        or any(code != 0 for code in run_receipt["terminal"].values())
        or run_receipt["buildReceiptSha256"] != image_receipt["sha256"]
        or run_receipt["archive"] != image_receipt["archive"]
        or run_receipt["cleanRoot"] != image_receipt["cleanRoot"]
        or run_receipt["dockerfile"] != image_receipt["dockerfile"]
        or run_receipt["image"] != image_receipt["image"]
        or run_receipt["versions"] != image_receipt["versions"]
        or any(
            evidence is None or evidence.get("status") != "passed"
            for evidence in run_receipt["copiedEvidence"].values()
        )
    ):
        raise ValueError("verifier receipt/image is stale or cross-bound incorrectly")

    variants_value = spec["variants"]
    if not isinstance(variants_value, list) or len(variants_value) != 5:
        raise ValueError("final handoff requires exactly five variants")
    seen_ids: set[str] = set()
    seen_results: set[str] = set()
    seen_videos: set[str] = set()
    variants = [
        _variant_handoff(
            root,
            item,
            seen_ids=seen_ids,
            seen_results=seen_results,
            seen_videos=seen_videos,
        )
        for item in variants_value
    ]
    variants.sort(key=lambda item: item["variantId"])
    source_manifest = archives["A"].pop("sourceManifest")
    archives["B"].pop("sourceManifest")
    for item in variants:
        for field in ("recipe", "result", "videoManifest", "video"):
            _require_archived_artifact(
                source_manifest,
                item[field],
                f"variant {item['variantId']} {field}",
            )

    acceptance_reference = spec["ownerAcceptance"]
    acceptance_artifact: dict[str, Any] | None = None
    acceptance: dict[str, Any] | None = None
    if acceptance_reference is not None:
        acceptance_artifact, acceptance, _ = _json_artifact(
            root, acceptance_reference, "owner acceptance"
        )
    if acceptance is not None:
        _closed(
            acceptance,
            {
                "schemaVersion",
                "status",
                "ownerId",
                "source",
                "wheelSha256",
                "archiveSha256",
                "archiveReportSha256",
                "archiveReceiptSha256",
                "postgresProofSha256",
                "verifier",
                "variants",
                "sha256",
            },
            "owner acceptance",
        )
        acceptance_hash = _canonical_document_hash(
            acceptance, "sha256", "owner acceptance"
        )
    else:
        acceptance_hash = None
    if acceptance is not None:
        expected_acceptance = {
            "schemaVersion": 1,
            "status": "approved",
            "ownerId": acceptance.get("ownerId"),
            "source": source,
            "wheelSha256": wheel_artifact["sha256"],
            "archiveSha256": archive_sha256,
            "archiveReportSha256": {
                label: archives[label]["report"]["sha256"]
                for label in ("A", "B")
            },
            "archiveReceiptSha256": {
                label: archives[label]["receipt"]["sha256"]
                for label in ("A", "B")
            },
            "postgresProofSha256": postgres_artifact["sha256"],
            "verifier": {
                "imageReceiptSha256": image_artifact["sha256"],
                "runReceiptSha256": run_artifact["sha256"],
                "imageId": image_id,
            },
            "variants": [
                {
                    "variantId": item["variantId"],
                    "resultHash": item["resultHash"],
                    "recipeSha256": item["recipe"]["sha256"],
                    "resultSha256": item["result"]["sha256"],
                    "videoManifestSha256": item["videoManifest"]["sha256"],
                    "videoSha256": item["video"]["sha256"],
                }
                for item in variants
            ],
        }
        if (
            not isinstance(acceptance.get("ownerId"), str)
            or not acceptance["ownerId"].strip()
            or {key: item for key, item in acceptance.items() if key != "sha256"}
            != expected_acceptance
        ):
            raise ValueError(
                "owner acceptance does not bind the complete final artifact set"
            )

    handoff = {
        "schemaVersion": 1,
        "kind": "balootx_final_cross_artifact_audit_handoff_v1",
        "status": (
            "ready_for_independent_audit"
            if acceptance is not None
            else "pending_owner_acceptance"
        ),
        "source": source,
        "wheel": {"receipt": wheel_receipt_artifact, "artifact": wheel_artifact},
        "archives": archives,
        "postgresProof": postgres_artifact,
        "verifier": {
            "imageReceipt": image_artifact,
            "runReceipt": run_artifact,
            "imageId": image_id,
        },
        "variants": variants,
        "ownerAcceptance": (
            {
                **acceptance_artifact,
                "status": "approved",
                "ownerId": acceptance["ownerId"],
                "acceptanceSha256": acceptance_hash,
            }
            if acceptance is not None and acceptance_artifact is not None
            else {"status": "pending"}
        ),
        "proofBoundary": (
            "This handoff proves deterministic cross-artifact identity, exact local file hashes, "
            "five qualified OR-Tools results and bound videos. Owner acceptance is a "
            "separate explicit final gate. "
            "It does not perform an independent audit, deployment, publication, or rules certification."
        ),
    }
    handoff["handoffSha256"] = canonical_sha256(handoff)
    return handoff


def collect_readiness(
    root: Path,
    *,
    candidates_path: Path,
    exact_path: Path,
    holdout_path: Path,
) -> dict[str, Any]:
    bundle = build_bundle(root)
    benchmark = bundle.component("benchmark")
    governance = dict(benchmark.get("permanent_holdout") or {})
    video = bundle.component("video")
    candidate_gate, candidates = _candidate_gate(
        candidates_path,
        int(governance["minimum_positions"]),
        root=root,
    )
    gates = {
        "sourceFreeze": _source_gate(root),
        "agentQualification": _agent_qualification_gate(root, bundle),
        "candidateCorpus": candidate_gate,
        "exactLabels": _exact_gate(exact_path, candidates, root=root),
        "independentAdjudication": _holdout_gate(
            holdout_path,
            candidates,
            minimum_positions=int(governance["minimum_positions"]),
            minimum_authorities=int(governance["minimum_independent_authorities"]),
            minimum_agreement=float(governance["minimum_adjudication_agreement"]),
            root=root,
        ),
        "releaseEvidence": {
            "status": "passed",
            "artifacts": [
                _evidence_artifact(root, relative)
                for relative in REQUIRED_RELEASE_EVIDENCE
            ],
        },
        "releaseMedia": _media_gate(
            root,
            int(video["render"]["width"]),
            int(video["render"]["height"]),
        ),
    }
    evidence_missing = [
        item["path"]
        for item in gates["releaseEvidence"]["artifacts"]
        if item["status"] == "pending"
    ]
    evidence_failed = [
        item["path"]
        for item in gates["releaseEvidence"]["artifacts"]
        if item["status"] == "failed"
    ]
    if evidence_failed:
        gates["releaseEvidence"]["status"] = "failed"
        gates["releaseEvidence"]["failed"] = evidence_failed
    elif evidence_missing:
        gates["releaseEvidence"]["status"] = "pending"
        gates["releaseEvidence"]["missing"] = evidence_missing
    statuses = [str(item["status"]) for item in gates.values()]
    status = (
        "failed"
        if "failed" in statuses
        else ("pending" if "pending" in statuses else "ready")
    )
    value = {
        "schemaVersion": 1,
        "release": "BalootX",
        "status": status,
        "bundle": {
            "id": bundle.bundle_profile["profile"]["id"],
            "version": bundle.bundle_profile["profile"]["version"],
            "sha256": bundle.sha256,
        },
        "gates": gates,
        "proofBoundary": (
            "Readiness proves only the integrity and presence of declared local release inputs. "
            "It does not prove reviewer identity or expertise, official Baloot certification, "
            "deployment, external security review, or human premium-media acceptance."
        ),
    }
    value["manifestSha256"] = canonical_sha256(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--exact-labels", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument(
        "--final-handoff-spec",
        type=Path,
        help="build the final cross-artifact audit handoff from a closed JSON specification",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.final_handoff_spec is None:
        value = collect_readiness(
            root,
            candidates_path=args.candidates.resolve(),
            exact_path=args.exact_labels.resolve(),
            holdout_path=args.holdout.resolve(),
        )
        identity_field = "manifestSha256"
    else:
        try:
            specification = json.loads(
                args.final_handoff_spec.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid final handoff specification: {exc}") from exc
        try:
            value = build_final_audit_handoff(root, specification)
        except (OSError, TypeError, ValueError) as exc:
            raise SystemExit(f"final handoff rejected: {exc}") from exc
        identity_field = "handoffSha256"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": value["status"],
                identity_field: value[identity_field],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    if args.require_ready and value["status"] not in {
        "ready",
        "ready_for_independent_audit",
    }:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
