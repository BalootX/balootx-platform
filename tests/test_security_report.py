from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from scripts import generate_security_report as report
from scripts import package_release
from scripts import verify_archive


def test_v21_security_report_is_source_bound_and_bounded(security_root: Path, monkeypatch) -> None:
    source = report.git_identity(security_root)
    monkeypatch.setattr(
        report,
        "git_identity",
        lambda root: copy.deepcopy(source),
    )
    value = report.build(
        root=security_root,
        ci_run_id="123",
        ci_head_sha=source["revision"],
        ci_conclusion="success",
    )

    assert value["status"] == "passed"
    assert value["source"] == source
    assert value["source"]["binding"] in {
        "exact_git_head",
        "git_head_plus_worktree_snapshot",
    }
    assert value["pluginPolicy"]["externalInProcessAllowed"] is False
    assert value["deploymentSecurity"] == "not_evaluated_deployment_frozen"
    assert value["independentPenetrationTest"] == "not_performed"
    report.verify(value, root=security_root)


def test_v21_security_report_rejects_stale_or_failed_ci(security_root: Path) -> None:
    with pytest.raises(ValueError, match="exact source head"):
        report.build(
            root=security_root,
            ci_run_id="123",
            ci_head_sha="0" * 40,
            ci_conclusion="success",
        )


def test_v21_security_report_rejects_stale_test_inventory(security_root: Path, monkeypatch) -> None:
    original_load = report.load

    def load_with_stale_inventory(root, relative):
        value = original_load(root, relative)
        if relative == "release/test-inventory.json":
            value = copy.deepcopy(value)
            value["nodeIds"][0] = "tests/missing.py::test_missing"
        return value

    monkeypatch.setattr(report, "load", load_with_stale_inventory)
    source = report.git_identity(security_root)
    monkeypatch.setattr(
        report,
        "git_identity",
        lambda root: copy.deepcopy(source),
    )
    with pytest.raises(ValueError, match="stale or invalid"):
        report.build(
            root=security_root,
            ci_run_id="123",
            ci_head_sha=source["revision"],
            ci_conclusion="success",
        )


def test_v21_security_report_rejects_resigned_tamper_and_source_drift(
    security_root: Path,
    monkeypatch,
) -> None:
    source = {
        "revision": "a" * 40,
        "tree": "b" * 40,
        "scope": "git_visible_product_inputs_v1",
        "excludedPaths": ["release/"],
        "binding": "git_head_plus_worktree_snapshot",
        "worktreeSnapshot": {
            "algorithm": "sha256",
            "scope": "git_visible_product_inputs_v1",
            "excludedPaths": ["release/"],
            "fileCount": 3,
            "statusSha256": "c" * 64,
            "sha256": "d" * 64,
        },
    }
    monkeypatch.setattr(report, "git_identity", lambda root: copy.deepcopy(source))
    value = report.build(
        root=security_root,
        ci_run_id="123",
        ci_head_sha=source["revision"],
        ci_conclusion="success",
    )
    report.verify(value, root=security_root)

    skeletal = {
        key: copy.deepcopy(value[key])
        for key in ("status", "source", "continuousIntegration")
    }
    skeletal["evidenceSha256"] = report.canonical_sha256(skeletal)
    with pytest.raises(ValueError, match="content is stale or invalid"):
        report.verify(skeletal, root=security_root)

    tampered = copy.deepcopy(value)
    tampered["source"]["worktreeSnapshot"]["sha256"] = "e" * 64
    with pytest.raises(ValueError, match="evidence hash mismatch"):
        report.verify(tampered, root=security_root)

    tampered["evidenceSha256"] = report.canonical_sha256(
        {
            key: item
            for key, item in tampered.items()
            if key != "evidenceSha256"
        }
    )
    with pytest.raises(ValueError, match="source identity is stale"):
        report.verify(tampered, root=security_root)

    changed_source = copy.deepcopy(source)
    changed_source["worktreeSnapshot"]["sha256"] = "f" * 64
    monkeypatch.setattr(report, "git_identity", lambda root: changed_source)
    with pytest.raises(ValueError, match="source identity is stale"):
        report.verify(value, root=security_root)


def test_security_evidence_authority_is_in_the_repository_package(tmp_path: Path, monkeypatch) -> None:
    assert package_release.include(
        report.REPOSITORY_ROOT / "scripts/generate_security_report.py"
    )
    assert package_release.include(
        report.REPOSITORY_ROOT / "scripts/generate_performance_evidence.py"
    )
    assert package_release.include(report.REPOSITORY_ROOT / __file__)
    monkeypatch.setattr(package_release, "ROOT", tmp_path)
    security_path = tmp_path / "release/security-report.json"
    security_path.parent.mkdir()
    security_path.write_text("{}\n", encoding="utf-8")
    assert package_release.include(security_path)


def test_repository_package_requires_verified_security_and_binds_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(package_release, "ROOT", tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/kept.py").write_text("kept = True\n", encoding="utf-8")
    (tmp_path / "release").mkdir()
    with pytest.raises(FileNotFoundError, match="security-report.json"):
        package_release.build(tmp_path / "missing.zip")

    security_path = tmp_path / "release/security-report.json"
    security_path.write_text("{}\n", encoding="utf-8", newline="\n")
    source = {
        "revision": "a" * 40,
        "tree": "b" * 40,
        "scope": "git_visible_product_inputs_v1",
        "excludedPaths": ["release/"],
        "binding": "exact_git_head",
    }
    verified = []
    monkeypatch.setattr(package_release, "git_identity", lambda root: source)
    monkeypatch.setattr(
        package_release,
        "verify_security",
        lambda value, **kwargs: verified.append((value, kwargs)),
    )

    archive = package_release.build(tmp_path / "candidate.zip")
    assert verified == [
        (
            {},
            {"root": tmp_path, "source_identity": source},
        )
    ]
    with zipfile.ZipFile(archive) as zipped:
        manifest_name = next(
            name
            for name in zipped.namelist()
            if name.endswith("/release/source-manifest.json")
        )
        manifest = json.loads(zipped.read(manifest_name))
        assert manifest["source"] == source
        assert any(
            name.endswith("/release/security-report.json")
            for name in zipped.namelist()
        )


@pytest.fixture
def security_root(
    tmp_path: Path,
) -> Path:
    paths = {
        "src/module.py": "source-base\n",
        "configs/rules.yaml": "config-base\n",
        "generated/contract.json": "generated-base\n",
        "tests/security.py": "# security inventory authority\n",
    }
    for relative, content in paths.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    for command in (
        ["git", "init"],
        ["git", "config", "user.email", "security-test@example.invalid"],
        ["git", "config", "user.name", "Security Test"],
        ["git", "add", "."],
        ["git", "commit", "-m", "baseline"],
    ):
        subprocess.run(
            command,
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    release = tmp_path / "release"
    release.mkdir()
    node_ids = [
        f"tests/security.py::{marker}" for marker in report.REQUIRED_TEST_MARKERS
    ]
    inventory_hash = hashlib.sha256(
        ("\n".join(node_ids) + "\n").encode("utf-8")
    ).hexdigest()
    artifacts = {
        "release/plugin-host.json": {
            "policy": {
                "externalInProcessAllowed": False,
                "manifestFirstDiscovery": True,
                "requireContentAddressedExternal": True,
                "requireSignedExternal": True,
            }
        },
        "release/sbom.cdx.json": {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "components": [{"name": "component", "version": "1"}],
        },
        "release/license-inventory.json": {
            "components": [{"name": "component", "license": "MIT"}]
        },
        "release/test-inventory.json": {
            "status": "passed",
            "nodeIds": node_ids,
            "inventorySha256": inventory_hash,
            "testCases": len(node_ids),
            "testFiles": 1,
            "files": ["tests/security.py"],
        },
    }
    for relative, document in artifacts.items():
        (tmp_path / relative).write_text(
            json.dumps(document, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return tmp_path


def test_security_verification_detects_included_source_config_and_generated_drift(
    security_root: Path,
) -> None:
    tmp_path = security_root
    candidate_source = tmp_path / "src/module.py"
    candidate_source.write_text(
        "source-candidate\n",
        encoding="utf-8",
        newline="\n",
    )
    source = report.git_identity(tmp_path)
    assert source["binding"] == "git_head_plus_worktree_snapshot"
    value = report.build(
        root=tmp_path,
        ci_run_id="123",
        ci_head_sha=source["revision"],
        ci_conclusion="success",
    )
    report.verify(value, root=tmp_path)

    baselines = {
        "src/module.py": "source-candidate\n",
        "configs/rules.yaml": "config-base\n",
        "generated/contract.json": "generated-base\n",
    }
    for relative, baseline in baselines.items():
        path = tmp_path / relative
        path.write_text(
            baseline.removesuffix("\n") + "-changed\n",
            encoding="utf-8",
            newline="\n",
        )
        with pytest.raises(ValueError, match="source identity is stale"):
            report.verify(value, root=tmp_path)
        path.write_text(baseline, encoding="utf-8", newline="\n")
        report.verify(value, root=tmp_path)

    for relative in report.INPUT_ARTIFACTS.values():
        path = tmp_path / relative
        baseline = path.read_bytes()
        path.write_bytes(baseline + b" \n")
        with pytest.raises(ValueError, match="content is stale or invalid"):
            report.verify(value, root=tmp_path)
        path.write_bytes(baseline)
        report.verify(value, root=tmp_path)

    security_path = tmp_path / "release/security-report.json"
    security_path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "source": source,
        "files": {
            "release/security-report.json": report.artifact_receipt(
                tmp_path,
                "release/security-report.json",
            )
        },
    }
    assert verify_archive._verify_packaged_security(
        tmp_path,
        manifest,
    )["status"] == "passed"

    with pytest.raises(
        verify_archive.ArchiveVerificationError,
        match="omits the required security report",
    ):
        verify_archive._verify_packaged_security(
            tmp_path,
            {"source": source, "files": {}},
        )

    skeletal = {
        key: copy.deepcopy(value[key])
        for key in ("status", "source", "continuousIntegration")
    }
    skeletal["evidenceSha256"] = report.canonical_sha256(skeletal)
    security_path.write_text(
        json.dumps(skeletal, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(
        verify_archive.ArchiveVerificationError,
        match="security report is stale or invalid",
    ):
        verify_archive._verify_packaged_security(tmp_path, manifest)
