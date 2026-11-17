from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

from scripts import check_release_readiness as handoff
from scripts import run_verifier_container


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _ref(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _fixture(root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    tracked = root / "source.txt"
    tracked.write_text("frozen\n", encoding="utf-8", newline="\n")
    subprocess.run(["git", "add", "source.txt"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=BalootX Test",
            "-c",
            "user.email=balootx-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    source = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True
        ).strip(),
    }

    wheel = root / "dist/baloot_platform-1-py3-none-any.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"wheel")
    wheel_sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    wheel_receipt = root / "release/wheel.json"
    _write_json(
        wheel_receipt,
        {
            "schemaVersion": 1,
            "status": "passed",
            "byteIdentical": True,
            "source": source,
            "wheel": wheel.relative_to(root).as_posix(),
            "builds": [{"sha256": wheel_sha}, {"sha256": wheel_sha}],
        },
    )

    variants = []
    acceptance_variants = []
    paths: dict[str, Path] = {"wheel": wheel, "acceptance": root / "release/owner.json"}
    frozen_variant_paths: list[Path] = []
    for index in range(5):
        variant_id = f"variant-{index + 1}"
        authority = {
            "backendId": "ortools_cp_sat_v2",
            "qualificationAuthority": "cp_sat",
            "bundleHash": hashlib.sha256(b"bundle").hexdigest(),
            "traceHash": hashlib.sha256(f"trace-{index}".encode()).hexdigest(),
        }
        authority["sha256"] = handoff.canonical_sha256(authority)
        identity = {
            "schemaVersion": 1,
            "variant": variant_id,
            "researchAuthorityHash": authority["sha256"],
        }
        result_hash = handoff.canonical_sha256(identity)
        result_code = f"LAF-{result_hash[:10].upper()}"
        projection_hash = hashlib.sha256(f"projection-{index}".encode()).hexdigest()
        recipe = root / f"outputs/{variant_id}-recipe.json"
        _write_json(
            recipe,
            {
                "schemaVersion": 4,
                "code": f"RS4-{index + 1}",
                "configuration": {
                    "bundleRevisionId": "CBR-AUTHORITY",
                    "semanticBundleHash": authority["bundleHash"],
                },
            },
        )
        result = root / f"outputs/{variant_id}-result.json"
        _write_json(
            result,
            {
                "schemaVersion": 2,
                "resultCode": result_code,
                "resultHash": result_hash,
                "researchAuthority": authority,
                "runtimeCompatibility": {
                    "status": "passed",
                    "bundleHash": authority["bundleHash"],
                    "traceHash": authority["traceHash"],
                },
                "qualification": {"status": "qualified", "qualified": True},
                "scenarioProjection": {"publicProjectionHash": projection_hash},
                "provenance": {"resultIdentity": identity},
            },
        )
        video = root / f"outputs/{variant_id}.mp4"
        video.write_bytes(f"video-{index}".encode())
        video_sha = hashlib.sha256(video.read_bytes()).hexdigest()
        manifest = root / f"outputs/{variant_id}-manifest.json"
        manifest_value = {
            "schemaVersion": 1,
            "observerSafe": True,
            "deterministic": True,
            "source": {
                "code": result_code,
                "publicProjectionHash": projection_hash,
                "qualificationStatus": "qualified",
                "publicationEligible": True,
                "auditOnly": False,
            },
            "files": {"video": {"name": video.name, "sha256": video_sha}},
        }
        manifest_value["manifestSha256"] = handoff.canonical_sha256(manifest_value)
        _write_json(manifest, manifest_value)
        variants.append(
            {
                "variantId": variant_id,
                "recipe": _ref(root, recipe),
                "result": _ref(root, result),
                "videoManifest": _ref(root, manifest),
                "video": _ref(root, video),
            }
        )
        acceptance_variants.append(
            {
                "variantId": variant_id,
                "resultHash": result_hash,
                "recipeSha256": _ref(root, recipe)["sha256"],
                "resultSha256": _ref(root, result)["sha256"],
                "videoManifestSha256": _ref(root, manifest)["sha256"],
                "videoSha256": video_sha,
            }
        )
        paths[f"video-{index}"] = video
        frozen_variant_paths.extend((recipe, result, manifest, video))

    source_manifest = {
        "schemaVersion": 1,
        "source": source,
        "files": {
            path.relative_to(root).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in frozen_variant_paths
        },
    }
    source_manifest_bytes = (
        json.dumps(source_manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    archive_a = root / "release/archive-a.zip"
    archive_a.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_a, "w") as archive:
        info = zipfile.ZipInfo("baloot-platform-1/release/source-manifest.json")
        info.date_time = (1980, 1, 1, 0, 0, 0)
        archive.writestr(info, source_manifest_bytes)
        for path in frozen_variant_paths:
            archive.write(path, f"baloot-platform-1/{path.relative_to(root).as_posix()}")
    archive_b = root / "release/archive-b.zip"
    shutil.copyfile(archive_a, archive_b)
    archive_sha = hashlib.sha256(archive_a.read_bytes()).hexdigest()
    reports: dict[str, Path] = {}
    archive_receipts: dict[str, Path] = {}
    for index, label in enumerate(("A", "B"), start=1):
        archive = archive_a if label == "A" else archive_b
        workspace = root / f"archive-workspace-{label.lower()}"
        clean_root = workspace / f"extract-{label.lower()}" / "baloot-platform-1"
        clean_root.mkdir(parents=True)
        report = root / f"release/archive-{label.lower()}-report.json"
        _write_json(
            report,
            {
                "schemaVersion": 1,
                "status": "passed",
                "archive": archive.name,
                "archiveSha256": archive_sha,
                "checks": [{"name": "complete", "status": "passed"}],
                "cleanRoot": str(clean_root.resolve()),
            },
        )
        stdout = root / f"release/archive-{label.lower()}.stdout.log"
        stderr = root / f"release/archive-{label.lower()}.stderr.log"
        stdout.write_text("passed\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        receipt = root / f"release/archive-{label.lower()}-receipt.json"
        _write_json(
            receipt,
            {
                "schemaVersion": 1,
                "taskName": f"BalootX-ArchiveVerify-{label}",
                "archive": str(archive.resolve()),
                "expectedArchiveSha256": archive_sha.upper(),
                "actualArchiveSha256": archive_sha.upper(),
                "workerPid": 100 + index,
                "verifierPid": 200 + index,
                "affinityMask": "0x3",
                "nativeThreadLimit": 1,
                "startedAt": f"2026-08-25T00:0{index}:00+03:00",
                "finishedAt": f"2026-08-25T00:1{index}:00+03:00",
                "exitCode": 0,
                "status": "passed",
                "reportStatus": "passed",
                "report": str(report.resolve()),
                "stdout": str(stdout.resolve()),
                "stderr": str(stderr.resolve()),
                "workspace": str(workspace.resolve()),
                "launcherError": None,
                "cleanupErrors": [],
            },
        )
        reports[label] = report
        archive_receipts[label] = receipt

    postgres = root / "release/postgres.json"
    postgres_value = {
        "schemaVersion": 1,
        "kind": "balootx_live_postgres_research_proof_v1",
        "status": "passed",
        "environment": {
            "source": {
                "sourceManifestSha256": hashlib.sha256(
                    source_manifest_bytes
                ).hexdigest()
            }
        },
    }
    postgres_value["sha256"] = handoff.canonical_sha256(postgres_value)
    _write_json(postgres, postgres_value)

    image_receipt = root / "release/verifier-image.json"
    image_id = f"sha256:{hashlib.sha256(b'image').hexdigest()}"
    source_manifest_sha = hashlib.sha256(source_manifest_bytes).hexdigest()
    dockerfile_sha = hashlib.sha256(b"Dockerfile").hexdigest()
    labels = {
        "org.balootx.verifier.archive-sha256": archive_sha,
        "org.balootx.verifier.source-manifest-sha256": source_manifest_sha,
        "org.balootx.verifier.dockerfile-sha256": dockerfile_sha,
        "org.opencontainers.image.version": "1",
    }
    image_value = run_verifier_container.build_image_receipt(
        status="passed",
        archive={"sha256": archive_sha, "bytes": archive_a.stat().st_size},
        clean_root={
            "kind": "independent_archive_extraction",
            "archivePrefix": "baloot-platform-1",
            "sourceManifest": {
                "relativePath": "release/source-manifest.json",
                "sha256": source_manifest_sha,
                "fileCount": len(frozen_variant_paths),
                "sourceRevision": source["revision"],
                "sourceTree": source["tree"],
                "sourceIdentitySha256": handoff.canonical_sha256(source),
            },
        },
        dockerfile_sha256=dockerfile_sha,
        image_id=image_id,
        image_labels=labels,
        exit_code=0,
        versions={
            "receiptOwner": run_verifier_container.OWNER_ID,
            "project": "1",
            "python": "3.13.7",
            "dockerClient": "28.3.2",
            "dockerServer": "28.3.2",
        },
    )
    _write_json(image_receipt, image_value)
    run_receipt = root / "release/verifier-run.json"
    evidence = {"schemaVersion": 1, "status": "passed", "sha256": "c" * 64}
    run_value = run_verifier_container.build_run_receipt(
        status="passed",
        build_receipt=image_value,
        create_exit_code=0,
        container_exit_code=0,
        copy_exit_code=0,
        timeout_seconds=3600,
        archive_report=evidence,
        verifier_preflight=evidence,
    )
    _write_json(run_receipt, run_value)

    acceptance = {
        "schemaVersion": 1,
        "status": "approved",
        "ownerId": "balootx-owner",
        "source": source,
        "wheelSha256": wheel_sha,
        "archiveSha256": archive_sha,
        "archiveReportSha256": {
            label: _ref(root, reports[label])["sha256"] for label in ("A", "B")
        },
        "archiveReceiptSha256": {
            label: _ref(root, archive_receipts[label])["sha256"]
            for label in ("A", "B")
        },
        "postgresProofSha256": _ref(root, postgres)["sha256"],
        "verifier": {
            "imageReceiptSha256": _ref(root, image_receipt)["sha256"],
            "runReceiptSha256": _ref(root, run_receipt)["sha256"],
            "imageId": image_id,
        },
        "variants": acceptance_variants,
    }
    acceptance["sha256"] = handoff.canonical_sha256(acceptance)
    _write_json(paths["acceptance"], acceptance)

    specification = {
        "schemaVersion": 1,
        "source": source,
        "wheelReceipt": _ref(root, wheel_receipt),
        "archives": {
            "A": {
                "archive": _ref(root, archive_a),
                "report": _ref(root, reports["A"]),
                "receipt": _ref(root, archive_receipts["A"]),
            },
            "B": {
                "archive": _ref(root, archive_b),
                "report": _ref(root, reports["B"]),
                "receipt": _ref(root, archive_receipts["B"]),
            },
        },
        "postgresProof": _ref(root, postgres),
        "verifier": {
            "imageReceipt": _ref(root, image_receipt),
            "runReceipt": _ref(root, run_receipt),
        },
        "variants": variants,
        "ownerAcceptance": _ref(root, paths["acceptance"]),
    }
    return specification, paths


def _admit_fixture_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        handoff,
        "validate_ortools_result_sections_v1",
        lambda result, *, recipe: dict(result),
    )


def test_final_handoff_is_closed_deterministic_and_cross_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _admit_fixture_results(monkeypatch)
    specification, _ = _fixture(tmp_path)

    first = handoff.build_final_audit_handoff(tmp_path, specification)
    second = handoff.build_final_audit_handoff(tmp_path, specification)

    assert first == second
    assert first["status"] == "ready_for_independent_audit"
    assert len(first["variants"]) == 5
    assert (
        first["archives"]["A"]["archive"]["sha256"]
        == first["archives"]["B"]["archive"]["sha256"]
    )
    assert first["handoffSha256"] == handoff.canonical_sha256(
        {key: item for key, item in first.items() if key != "handoffSha256"}
    )


def test_final_handoff_rejects_missing_and_hash_substituted_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _admit_fixture_results(monkeypatch)
    specification, paths = _fixture(tmp_path)
    paths["video-0"].write_bytes(b"substituted")
    with pytest.raises(ValueError, match="hash substitution"):
        handoff.build_final_audit_handoff(tmp_path, specification)

    specification, paths = _fixture(tmp_path / "missing")
    paths["wheel"].unlink()
    with pytest.raises(ValueError, match="missing or non-canonical"):
        handoff.build_final_audit_handoff(tmp_path / "missing", specification)


def test_final_handoff_rejects_unknown_fields_and_stale_owner_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _admit_fixture_results(monkeypatch)
    specification, paths = _fixture(tmp_path)
    with pytest.raises(ValueError, match="must contain exactly"):
        handoff.build_final_audit_handoff(tmp_path, {**specification, "unknown": True})

    acceptance = json.loads(paths["acceptance"].read_text(encoding="utf-8"))
    acceptance["archiveSha256"] = "0" * 64
    acceptance["sha256"] = handoff.canonical_sha256(
        {key: item for key, item in acceptance.items() if key != "sha256"}
    )
    _write_json(paths["acceptance"], acceptance)
    specification["ownerAcceptance"] = _ref(tmp_path, paths["acceptance"])
    with pytest.raises(ValueError, match="does not bind"):
        handoff.build_final_audit_handoff(tmp_path, specification)


def test_final_handoff_can_stop_at_explicit_pending_owner_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _admit_fixture_results(monkeypatch)
    specification, _ = _fixture(tmp_path)
    specification["ownerAcceptance"] = None

    first = handoff.build_final_audit_handoff(tmp_path, specification)
    second = handoff.build_final_audit_handoff(tmp_path, specification)

    assert first == second
    assert first["status"] == "pending_owner_acceptance"
    assert first["status"] not in {"ready", "ready_for_independent_audit"}
    assert first["ownerAcceptance"] == {"status": "pending"}


def test_final_handoff_rejects_copied_archive_execution_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _admit_fixture_results(monkeypatch)
    specification, _ = _fixture(tmp_path)
    specification["ownerAcceptance"] = None
    report_a_path = tmp_path / specification["archives"]["A"]["report"]["path"]
    report_b_path = tmp_path / specification["archives"]["B"]["report"]["path"]
    receipt_a_path = tmp_path / specification["archives"]["A"]["receipt"]["path"]
    receipt_b_path = tmp_path / specification["archives"]["B"]["receipt"]["path"]
    report_a = json.loads(report_a_path.read_text(encoding="utf-8"))
    report_b = json.loads(report_b_path.read_text(encoding="utf-8"))
    report_b["cleanRoot"] = report_a["cleanRoot"]
    _write_json(report_b_path, report_b)
    receipt_a = json.loads(receipt_a_path.read_text(encoding="utf-8"))
    receipt_b = json.loads(receipt_b_path.read_text(encoding="utf-8"))
    receipt_b["workspace"] = receipt_a["workspace"]
    _write_json(receipt_b_path, receipt_b)
    specification["archives"]["B"]["report"] = _ref(tmp_path, report_b_path)
    specification["archives"]["B"]["receipt"] = _ref(tmp_path, receipt_b_path)

    with pytest.raises(ValueError, match="independent clean-root executions"):
        handoff.build_final_audit_handoff(tmp_path, specification)


def test_final_handoff_rejects_a_copied_archive_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _admit_fixture_results(monkeypatch)
    specification, _ = _fixture(tmp_path)
    specification["ownerAcceptance"] = None
    specification["archives"]["B"]["report"] = specification["archives"]["A"][
        "report"
    ]

    with pytest.raises(ValueError, match="launcher receipt is stale or invalid"):
        handoff.build_final_audit_handoff(tmp_path, specification)


def test_final_handoff_uses_canonical_ortools_result_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specification, _ = _fixture(tmp_path)
    called = []

    def reject(result: dict[str, Any], *, recipe: dict[str, Any]) -> dict[str, Any]:
        called.append((result["resultCode"], recipe["code"]))
        raise RuntimeError("canonical authority rejected")

    monkeypatch.setattr(handoff, "validate_ortools_result_sections_v1", reject)
    with pytest.raises(ValueError, match="canonical authority rejected"):
        handoff.build_final_audit_handoff(tmp_path, specification)
    assert len(called) == 1
    assert called[0][1] == "RS4-1"


def test_final_handoff_requires_variant_bytes_in_frozen_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _admit_fixture_results(monkeypatch)
    specification, _ = _fixture(tmp_path)
    specification["ownerAcceptance"] = None
    original = tmp_path / specification["variants"][0]["result"]["path"]
    outside = tmp_path / "outputs/unarchived-result.json"
    shutil.copyfile(original, outside)
    specification["variants"][0]["result"] = _ref(tmp_path, outside)

    with pytest.raises(ValueError, match="not bound into the frozen source archive"):
        handoff.build_final_audit_handoff(tmp_path, specification)
