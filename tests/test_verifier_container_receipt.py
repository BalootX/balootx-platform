from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts import run_verifier_container


HASH_A = "a" * 64
IMAGE_ID = "sha256:" + "b" * 64


def _source_archive(tmp_path: Path) -> tuple[Path, Path, str]:
    clean_root = tmp_path / "machine-specific" / "baloot-platform-21.1.0"
    manifest_path = clean_root / "release" / "source-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        "schemaVersion": 1,
        "project": "baloot-platform",
        "version": "21.1.0",
        "archivePrefix": "baloot-platform-21.1.0",
        "source": {
            "revision": "1" * 40,
            "tree": "2" * 40,
            "scope": "git_visible_product_inputs_v1",
            "excludedPaths": ["release/"],
            "binding": "exact_git_head",
        },
        "fileCount": 1,
        "files": {"Dockerfile": {"bytes": 6, "sha256": HASH_A}},
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (clean_root / "Dockerfile").write_text("FROM x", encoding="utf-8")
    archive = tmp_path / "candidate.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.write(
            manifest_path,
            "baloot-platform-21.1.0/release/source-manifest.json",
        )
    return archive, clean_root, hashlib.sha256(archive.read_bytes()).hexdigest()


def _build_receipt(tmp_path: Path) -> dict[str, object]:
    archive, clean_root, archive_sha = _source_archive(tmp_path)
    archive_identity, root_identity, version = run_verifier_container._source_identity(
        archive=archive,
        expected_archive_sha256=archive_sha,
        clean_root=clean_root,
    )
    return run_verifier_container.build_image_receipt(
        status="passed",
        archive=archive_identity,
        clean_root=root_identity,
        dockerfile_sha256=hashlib.sha256(b"FROM x").hexdigest(),
        image_id=IMAGE_ID,
        image_labels={
            "org.balootx.verifier.archive-sha256": archive_sha,
            "org.balootx.verifier.source-manifest-sha256": root_identity[
                "sourceManifest"
            ]["sha256"],
            "org.balootx.verifier.dockerfile-sha256": hashlib.sha256(
                b"FROM x"
            ).hexdigest(),
            "org.opencontainers.image.version": version,
        },
        exit_code=0,
        versions={
            "receiptOwner": run_verifier_container.OWNER_ID,
            "project": version,
            "python": "3.13.7",
            "dockerClient": "28.3.2",
            "dockerServer": "28.3.2",
        },
    )


def test_verifier_build_receipt_binds_source_image_and_has_no_host_identity(
    tmp_path: Path,
) -> None:
    receipt = _build_receipt(tmp_path)

    assert receipt["status"] == "passed"
    assert receipt["image"]["id"] == IMAGE_ID
    assert receipt["cleanRoot"]["sourceManifest"]["relativePath"] == (
        "release/source-manifest.json"
    )
    assert receipt == run_verifier_container.verify_receipt(receipt)
    serialized = json.dumps(receipt, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "startedAt" not in serialized
    assert "finishedAt" not in serialized


def test_verifier_run_receipt_binds_build_terminal_and_copied_reports(
    tmp_path: Path,
) -> None:
    build = _build_receipt(tmp_path)
    archive_report = {"schemaVersion": 1, "status": "passed", "sha256": "c" * 64}
    preflight = {"schemaVersion": 1, "status": "passed", "sha256": "d" * 64}

    first = run_verifier_container.build_run_receipt(
        status="passed",
        build_receipt=build,
        create_exit_code=0,
        container_exit_code=0,
        copy_exit_code=0,
        timeout_seconds=3600,
        archive_report=archive_report,
        verifier_preflight=preflight,
    )
    second = run_verifier_container.build_run_receipt(
        status="passed",
        build_receipt=build,
        create_exit_code=0,
        container_exit_code=0,
        copy_exit_code=0,
        timeout_seconds=3600,
        archive_report=archive_report,
        verifier_preflight=preflight,
    )

    assert first == second
    assert first["buildReceiptSha256"] == build["sha256"]
    assert first["copiedEvidence"]["archiveVerificationReport"] == archive_report
    assert first["copiedEvidence"]["verifierPreflight"] == preflight
    assert run_verifier_container.verify_receipt(first) == first


def test_verifier_receipts_reject_tampering_and_false_passes(tmp_path: Path) -> None:
    build = _build_receipt(tmp_path)
    tampered = copy.deepcopy(build)
    tampered["image"]["id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="semanticHash mismatch"):
        run_verifier_container.verify_receipt(tampered)

    passed = {"schemaVersion": 1, "status": "passed", "sha256": "c" * 64}
    with pytest.raises(ValueError, match="passed terminal evidence"):
        run_verifier_container.build_run_receipt(
            status="passed",
            build_receipt=build,
            create_exit_code=0,
            container_exit_code=1,
            copy_exit_code=0,
            timeout_seconds=3600,
            archive_report=passed,
            verifier_preflight=passed,
        )

    unknown = copy.deepcopy(build)
    unknown["hostPath"] = "C:/unsafe"
    with pytest.raises(ValueError, match="must contain exactly"):
        run_verifier_container.verify_receipt(unknown)

    with pytest.raises(ValueError, match="zero exit code"):
        run_verifier_container.build_image_receipt(
            status="passed",
            archive=build["archive"],
            clean_root=build["cleanRoot"],
            dockerfile_sha256=build["dockerfile"]["sha256"],
            image_id=IMAGE_ID,
            image_labels=build["image"]["labels"],
            exit_code=1,
            versions=build["versions"],
        )


def test_verifier_source_binding_rejects_another_archive(tmp_path: Path) -> None:
    archive, clean_root, archive_sha = _source_archive(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="does not match"):
        run_verifier_container._source_identity(
            archive=archive,
            expected_archive_sha256=archive_sha,
            clean_root=clean_root,
        )


@pytest.mark.parametrize("docker_host", [None, "ssh://root@example.invalid"])
def test_verifier_refuses_windows_before_source_or_docker_work(
    monkeypatch: pytest.MonkeyPatch, docker_host: str | None
) -> None:
    monkeypatch.setattr(run_verifier_container.platform, "system", lambda: "Windows")
    if docker_host is None:
        monkeypatch.delenv("DOCKER_HOST", raising=False)
    else:
        monkeypatch.setenv("DOCKER_HOST", docker_host)

    def unexpected_work(*args, **kwargs):
        pytest.fail("Windows refusal must precede source reads and Docker commands")

    monkeypatch.setattr(run_verifier_container, "_source_identity", unexpected_work)
    monkeypatch.setattr(run_verifier_container, "_run", unexpected_work)
    with pytest.raises(RuntimeError, match="PC verifier builds are disabled"):
        run_verifier_container.execute(argparse.Namespace())


@pytest.mark.parametrize("copy_exit_code", (0, 1))
def test_verifier_owner_writes_both_terminal_receipts_without_docker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, copy_exit_code: int,
) -> None:
    monkeypatch.setattr(run_verifier_container.platform, "system", lambda: "Linux")
    archive, clean_root, archive_sha = _source_archive(tmp_path)
    archive_report = tmp_path / "archive-report.json"
    archive_report.write_text(
        json.dumps({"schemaVersion": 1, "status": "passed"}), encoding="utf-8"
    )
    preflight = tmp_path / "verifier-preflight-container.json"
    build_receipt = tmp_path / "verifier-image-build-final.exit.json"
    run_receipt = tmp_path / "verifier-container-run-final.exit.json"
    dockerfile_sha = hashlib.sha256(b"FROM x").hexdigest()
    manifest_sha = hashlib.sha256(
        (clean_root / "release/source-manifest.json").read_bytes()
    ).hexdigest()
    labels = {
        "org.balootx.verifier.archive-sha256": archive_sha,
        "org.balootx.verifier.source-manifest-sha256": manifest_sha,
        "org.balootx.verifier.dockerfile-sha256": dockerfile_sha,
        "org.opencontainers.image.version": "21.1.0",
    }

    commands = []
    def fake_run(command: list[str], *, cwd: Path | None = None):
        del cwd
        commands.append(command)
        if command[1:3] == ["version", "--format"]:
            output = json.dumps(
                {"Client": {"Version": "28.3.2"}, "Server": {"Version": "28.3.2"}}
            )
        elif command[1:3] == ["image", "inspect"]:
            output = json.dumps([{"Id": IMAGE_ID, "Config": {"Labels": labels}}])
        elif command[1] == "create":
            output = "container-id\n"
        elif command[1:3] == ["inspect", "--format"]:
            output = "0\n"
        elif command[1] == "cp":
            if copy_exit_code:
                return subprocess.CompletedProcess(command, copy_exit_code, "", "copy failed")
            preflight.write_text(
                json.dumps({"schemaVersion": 1, "status": "passed"}),
                encoding="utf-8",
            )
            output = ""
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(run_verifier_container, "_run", fake_run)
    args = run_verifier_container.parse_args(
        [
            "--archive",
            str(archive),
            "--archive-sha256",
            archive_sha,
            "--clean-root",
            str(clean_root),
            "--archive-report",
            str(archive_report),
            "--preflight-output",
            str(preflight),
            "--build-receipt",
            str(build_receipt),
            "--run-receipt",
            str(run_receipt),
        ]
    )

    assert run_verifier_container.execute(args) == (0 if copy_exit_code == 0 else 1)
    built = run_verifier_container.verify_receipt(
        json.loads(build_receipt.read_text(encoding="utf-8"))
    )
    ran = run_verifier_container.verify_receipt(
        json.loads(run_receipt.read_text(encoding="utf-8"))
    )
    assert built["image"]["id"] == IMAGE_ID
    assert ran["terminal"] == {
        "createExitCode": 0,
        "containerExitCode": 0,
        "copyExitCode": copy_exit_code,
    }
    created = next(command for command in commands if command[1] == "create")
    assert created[2:5] == ["--init", "--tmpfs", "/tmp:rw,exec,size=2g"]
    copied = next(command for command in commands if command[1] == "cp")
    assert copied[2:] == ["container-id:/app/release/verifier-preflight.json", str(preflight)]
    removed = [command for command in commands if command[1] == "rm"]
    if copy_exit_code == 0:
        assert removed == [["docker", "rm", "container-id"]]
        assert commands.index(copied) < commands.index(removed[0])
    else:
        assert removed == []
