#!/usr/bin/env python3
"""Build and run the final verifier container with immutable receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
OWNER_ID = "balootx-verifier-container-receipt-v1"
HASH_KEYS = ("semanticHash", "sha256")
BUILD_FIELDS = {
    "schemaVersion",
    "kind",
    "status",
    "archive",
    "cleanRoot",
    "dockerfile",
    "image",
    "terminal",
    "commands",
    "versions",
    *HASH_KEYS,
}
RUN_FIELDS = {
    "schemaVersion",
    "kind",
    "status",
    "archive",
    "cleanRoot",
    "dockerfile",
    "image",
    "buildReceiptSha256",
    "terminal",
    "copiedEvidence",
    "commands",
    "versions",
    *HASH_KEYS,
}


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(value: Any, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 hash")
    return value


def _closed(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} must contain exactly {sorted(fields)}")
    return value


def _seal(core: Mapping[str, Any]) -> dict[str, Any]:
    semantic_hash = _sha256_bytes(_canonical_bytes(core))
    semantic = {**dict(core), "semanticHash": semantic_hash}
    return {**semantic, "sha256": _sha256_bytes(_canonical_bytes(semantic))}


def verify_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(receipt)
    kind = document.get("kind")
    fields = BUILD_FIELDS if kind == "verifier_image_build" else RUN_FIELDS
    if kind not in {"verifier_image_build", "verifier_container_run"}:
        raise ValueError("verifier receipt kind is unsupported")
    _closed(document, fields, "verifier receipt")
    if document["schemaVersion"] != SCHEMA_VERSION:
        raise ValueError("verifier receipt schemaVersion must be 1")
    if document["status"] not in {"passed", "failed"}:
        raise ValueError("verifier receipt status is invalid")
    _closed(document["archive"], {"sha256", "bytes"}, "archive identity")
    _closed(
        document["cleanRoot"],
        {"kind", "archivePrefix", "sourceManifest"},
        "clean-root identity",
    )
    _closed(
        document["cleanRoot"]["sourceManifest"],
        {
            "relativePath",
            "sha256",
            "fileCount",
            "sourceRevision",
            "sourceTree",
            "sourceIdentitySha256",
        },
        "source-manifest identity",
    )
    _closed(document["dockerfile"], {"relativePath", "sha256"}, "Dockerfile")
    _closed(document["image"], {"id", "labels"}, "image identity")
    _closed(
        document["versions"],
        {"receiptOwner", "project", "python", "dockerClient", "dockerServer"},
        "version identities",
    )
    if type(document["image"]["labels"]) is not dict:
        raise ValueError("image labels must be an object")
    archive_sha256 = _require_hash(document["archive"]["sha256"], "archive SHA-256")
    if (
        type(document["archive"]["bytes"]) is not int
        or document["archive"]["bytes"] < 0
    ):
        raise ValueError("archive bytes must be a non-negative integer")
    source_manifest = document["cleanRoot"]["sourceManifest"]
    source_sha256 = _require_hash(source_manifest["sha256"], "source-manifest SHA-256")
    _require_hash(source_manifest["sourceIdentitySha256"], "source identity SHA-256")
    if (
        document["cleanRoot"]["kind"] != "independent_archive_extraction"
        or source_manifest["relativePath"] != "release/source-manifest.json"
        or type(source_manifest["fileCount"]) is not int
        or source_manifest["fileCount"] < 0
        or type(source_manifest["sourceRevision"]) is not str
        or type(source_manifest["sourceTree"]) is not str
    ):
        raise ValueError("clean-root source-manifest identity is invalid")
    dockerfile_sha256 = _require_hash(
        document["dockerfile"]["sha256"], "Dockerfile SHA-256"
    )
    if document["versions"]["receiptOwner"] != OWNER_ID:
        raise ValueError("verifier receipt owner identity mismatch")
    if document["status"] == "passed":
        image_id = document["image"]["id"]
        if type(image_id) is not str or not image_id.startswith("sha256:"):
            raise ValueError("passed verifier receipt requires an immutable image ID")
        _require_hash(image_id.removeprefix("sha256:"), "verifier image ID")
        expected_labels = {
            "org.balootx.verifier.archive-sha256": archive_sha256,
            "org.balootx.verifier.source-manifest-sha256": source_sha256,
            "org.balootx.verifier.dockerfile-sha256": dockerfile_sha256,
            "org.opencontainers.image.version": document["versions"]["project"],
        }
        if document["image"]["labels"] != expected_labels:
            raise ValueError("verifier image labels do not bind the receipt source")
    if kind == "verifier_image_build":
        _closed(document["terminal"], {"exitCode"}, "build terminal")
        _closed(document["commands"], {"build", "inspect"}, "build commands")
        if type(document["terminal"]["exitCode"]) is not int:
            raise TypeError("build exit code must be an exact integer")
        if document["status"] == "passed" and document["terminal"]["exitCode"] != 0:
            raise ValueError("passed verifier build requires a zero exit code")
    else:
        _closed(
            document["terminal"],
            {"createExitCode", "containerExitCode", "copyExitCode"},
            "container terminal",
        )
        _closed(
            document["copiedEvidence"],
            {"archiveVerificationReport", "verifierPreflight"},
            "copied evidence",
        )
        _closed(
            document["commands"],
            {"create", "start", "copy", "inspect"},
            "container commands",
        )
        terminal = document["terminal"]
        if (
            type(terminal["createExitCode"]) is not int
            or type(terminal["copyExitCode"]) is not int
            or (
                terminal["containerExitCode"] is not None
                and type(terminal["containerExitCode"]) is not int
            )
        ):
            raise TypeError("container terminal codes must be exact integers or null")
        for label, evidence in document["copiedEvidence"].items():
            if evidence is not None:
                _closed(
                    evidence,
                    {"schemaVersion", "status", "sha256"},
                    f"{label} evidence",
                )
                _require_hash(evidence["sha256"], f"{label} SHA-256")
        if document["status"] == "passed" and (
            any(code != 0 for code in document["terminal"].values())
            or any(
                evidence is None or evidence["status"] != "passed"
                for evidence in document["copiedEvidence"].values()
            )
        ):
            raise ValueError("passed container run requires passed terminal evidence")
        _require_hash(document["buildReceiptSha256"], "build receipt SHA-256")
    for command in document["commands"].values():
        _closed(command, {"id", "schemaVersion", "arguments"}, "command identity")
        if command["schemaVersion"] != 1:
            raise ValueError("verifier command schemaVersion must be 1")
    expected_command_ids = (
        {"docker-build-verifier-v1", "docker-image-inspect-v1"}
        if kind == "verifier_image_build"
        else {
            "docker-create-verifier-v1",
            "docker-start-attach-verifier-v1",
            "docker-copy-verifier-preflight-v1",
            "docker-container-exit-inspect-v1",
        }
    )
    if {
        command["id"] for command in document["commands"].values()
    } != expected_command_ids:
        raise ValueError("verifier command identity mismatch")
    semantic = {key: value for key, value in document.items() if key not in HASH_KEYS}
    expected_semantic = _sha256_bytes(_canonical_bytes(semantic))
    if document["semanticHash"] != expected_semantic:
        raise ValueError("verifier receipt semanticHash mismatch")
    expected_receipt = _sha256_bytes(
        _canonical_bytes({**semantic, "semanticHash": expected_semantic})
    )
    if document["sha256"] != expected_receipt:
        raise ValueError("verifier receipt sha256 mismatch")
    return document


def _json_document(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unavailable or invalid") from exc
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _evidence_identity(path: Path, label: str) -> dict[str, Any]:
    document = _json_document(path, label)
    status = document.get("status")
    schema_version = document.get("schemaVersion")
    if status not in {"passed", "failed"} or type(schema_version) is not int:
        raise ValueError(f"{label} must have an exact schemaVersion and status")
    return {
        "schemaVersion": schema_version,
        "status": status,
        "sha256": sha256_file(path),
    }


def _source_identity(
    *, archive: Path, expected_archive_sha256: str, clean_root: Path
) -> tuple[dict[str, Any], dict[str, Any], str]:
    expected = _require_hash(expected_archive_sha256, "archive SHA-256")
    actual = sha256_file(archive)
    if actual != expected:
        raise ValueError("archive SHA-256 does not match the expected archive")
    manifest_path = clean_root / "release" / "source-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if type(manifest) is not dict:
        raise ValueError("source manifest must be a JSON object")
    archive_prefix = manifest.get("archivePrefix")
    if type(archive_prefix) is not str or not archive_prefix:
        raise ValueError("source manifest archivePrefix is invalid")
    member = f"{archive_prefix}/release/source-manifest.json"
    with zipfile.ZipFile(archive) as zipped:
        if zipped.read(member) != manifest_bytes:
            raise ValueError("clean-root source manifest does not match the archive")
    source = manifest.get("source")
    if type(source) is not dict:
        raise ValueError("source manifest source identity is invalid")
    revision = source.get("revision")
    tree = source.get("tree")
    if type(revision) is not str or type(tree) is not str:
        raise ValueError("source manifest revision or tree is invalid")
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    return (
        {"sha256": actual, "bytes": archive.stat().st_size},
        {
            "kind": "independent_archive_extraction",
            "archivePrefix": archive_prefix,
            "sourceManifest": {
                "relativePath": "release/source-manifest.json",
                "sha256": manifest_sha256,
                "fileCount": manifest.get("fileCount"),
                "sourceRevision": revision,
                "sourceTree": tree,
                "sourceIdentitySha256": _sha256_bytes(_canonical_bytes(source)),
            },
        },
        str(manifest.get("version")),
    )


def _command(id_: str, arguments: list[str]) -> dict[str, Any]:
    return {"id": id_, "schemaVersion": 1, "arguments": arguments}


def _versions(
    *, project_version: str, docker_client: str, docker_server: str
) -> dict[str, str]:
    return {
        "receiptOwner": OWNER_ID,
        "project": project_version,
        "python": platform.python_version(),
        "dockerClient": docker_client,
        "dockerServer": docker_server,
    }


def build_image_receipt(
    *,
    status: str,
    archive: Mapping[str, Any],
    clean_root: Mapping[str, Any],
    dockerfile_sha256: str,
    image_id: str | None,
    image_labels: Mapping[str, str],
    exit_code: int,
    versions: Mapping[str, str],
) -> dict[str, Any]:
    if status not in {"passed", "failed"} or type(exit_code) is not int:
        raise ValueError("build status or exit code is invalid")
    if status == "passed" and exit_code != 0:
        raise ValueError("passed verifier build requires a zero exit code")
    if status == "passed" and (
        type(image_id) is not str or not image_id.startswith("sha256:")
    ):
        raise ValueError("passed verifier build requires an immutable image ID")
    core = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "verifier_image_build",
        "status": status,
        "archive": dict(archive),
        "cleanRoot": dict(clean_root),
        "dockerfile": {
            "relativePath": "Dockerfile",
            "sha256": _require_hash(dockerfile_sha256, "Dockerfile SHA-256"),
        },
        "image": {"id": image_id, "labels": dict(sorted(image_labels.items()))},
        "terminal": {"exitCode": exit_code},
        "commands": {
            "build": _command(
                "docker-build-verifier-v1",
                [
                    "docker",
                    "build",
                    "--target",
                    "verifier",
                    "--label",
                    "<governed-labels>",
                    "--tag",
                    "<derived-tag>",
                    ".",
                ],
            ),
            "inspect": _command(
                "docker-image-inspect-v1",
                ["docker", "image", "inspect", "<derived-tag>"],
            ),
        },
        "versions": dict(versions),
    }
    return verify_receipt(_seal(core))


def build_run_receipt(
    *,
    status: str,
    build_receipt: Mapping[str, Any],
    create_exit_code: int,
    container_exit_code: int | None,
    copy_exit_code: int,
    timeout_seconds: int,
    archive_report: Mapping[str, Any] | None,
    verifier_preflight: Mapping[str, Any] | None,
) -> dict[str, Any]:
    build = verify_receipt(build_receipt)
    if build["kind"] != "verifier_image_build" or build["status"] != "passed":
        raise ValueError("container run requires a passed verifier image build receipt")
    if status not in {"passed", "failed"}:
        raise ValueError("container run status is invalid")
    if (
        type(create_exit_code) is not int
        or (container_exit_code is not None and type(container_exit_code) is not int)
        or type(copy_exit_code) is not int
    ):
        raise TypeError("container terminal codes must be exact integers or null")
    if type(timeout_seconds) is not int or not 300 <= timeout_seconds <= 3600:
        raise ValueError("container timeout must be between 300 and 3600 seconds")
    if status == "passed" and (
        create_exit_code != 0
        or container_exit_code != 0
        or copy_exit_code != 0
        or archive_report is None
        or verifier_preflight is None
        or archive_report.get("status") != "passed"
        or verifier_preflight.get("status") != "passed"
    ):
        raise ValueError("passed container run requires passed terminal evidence")
    core = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "verifier_container_run",
        "status": status,
        "archive": build["archive"],
        "cleanRoot": build["cleanRoot"],
        "dockerfile": build["dockerfile"],
        "image": build["image"],
        "buildReceiptSha256": build["sha256"],
        "terminal": {
            "createExitCode": create_exit_code,
            "containerExitCode": container_exit_code,
            "copyExitCode": copy_exit_code,
        },
        "copiedEvidence": {
            "archiveVerificationReport": archive_report,
            "verifierPreflight": verifier_preflight,
        },
        "commands": {
            "create": _command(
                "docker-create-verifier-v1",
                [
                    "docker",
                    "create",
                    "--init",
                    "--tmpfs",
                    "/tmp:rw,exec,size=2g",
                    "-e",
                    f"BALOOT_VERIFIER_TEST_TIMEOUT_SECONDS={timeout_seconds}",
                    "<image-id>",
                ],
            ),
            "start": _command(
                "docker-start-attach-verifier-v1",
                ["docker", "start", "--attach", "<container-id>"],
            ),
            "copy": _command(
                "docker-copy-verifier-preflight-v1",
                [
                    "docker",
                    "cp",
                    "<container-id>:/app/release/verifier-preflight.json",
                    "<receipt-output>",
                ],
            ),
            "inspect": _command(
                "docker-container-exit-inspect-v1",
                ["docker", "inspect", "<container-id>"],
            ),
        },
        "versions": build["versions"],
    }
    return verify_receipt(_seal(core))


def _write(path: Path, receipt: Mapping[str, Any]) -> None:
    verified = verify_receipt(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(verified, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _run(
    command: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _docker_versions() -> tuple[str, str]:
    completed = _run(["docker", "version", "--format", "{{json .}}"])
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "docker version failed")
    value = json.loads(completed.stdout)
    return str(value["Client"]["Version"]), str(value["Server"]["Version"])


def execute(args: argparse.Namespace) -> int:
    if platform.system() == "Windows":
        raise RuntimeError(
            "PC verifier builds are disabled; use the owner-admitted offline research "
            "lane after its image-build backend is resource-accounted and CI permits "
            "execution. A remote Docker endpoint does not establish that admission."
        )
    archive, clean_root, project_version = _source_identity(
        archive=args.archive,
        expected_archive_sha256=args.archive_sha256,
        clean_root=args.clean_root,
    )
    archive_report = _evidence_identity(args.archive_report, "archive report")
    dockerfile = args.clean_root / "Dockerfile"
    dockerfile_sha256 = sha256_file(dockerfile)
    docker_client, docker_server = _docker_versions()
    versions = _versions(
        project_version=project_version,
        docker_client=docker_client,
        docker_server=docker_server,
    )
    labels = {
        "org.balootx.verifier.archive-sha256": archive["sha256"],
        "org.balootx.verifier.source-manifest-sha256": clean_root["sourceManifest"][
            "sha256"
        ],
        "org.balootx.verifier.dockerfile-sha256": dockerfile_sha256,
        "org.opencontainers.image.version": project_version,
    }
    image_tag = (
        args.image_tag or f"balootx-verifier:{project_version}-{archive['sha256'][:12]}"
    )
    build_command = ["docker", "build", "--target", "verifier"]
    for key, value in sorted(labels.items()):
        build_command.extend(["--label", f"{key}={value}"])
    build_command.extend(["--tag", image_tag, "."])
    built = _run(build_command, cwd=args.clean_root)
    image_id: str | None = None
    inspected_labels: dict[str, str] = {}
    if built.returncode == 0:
        inspected = _run(["docker", "image", "inspect", image_tag])
        if inspected.returncode == 0:
            payload = json.loads(inspected.stdout)[0]
            image_id = str(payload["Id"])
            all_labels = dict(payload["Config"].get("Labels") or {})
            inspected_labels = {
                key: str(all_labels[key]) for key in sorted(labels) if key in all_labels
            }
            if any(inspected_labels.get(key) != value for key, value in labels.items()):
                raise RuntimeError(
                    "verifier image labels do not bind the requested source"
                )
        else:
            built = subprocess.CompletedProcess(
                build_command, inspected.returncode, built.stdout, inspected.stderr
            )
    build_status = "passed" if built.returncode == 0 and image_id else "failed"
    build_receipt = build_image_receipt(
        status=build_status,
        archive=archive,
        clean_root=clean_root,
        dockerfile_sha256=dockerfile_sha256,
        image_id=image_id,
        image_labels=inspected_labels,
        exit_code=built.returncode,
        versions=versions,
    )
    _write(args.build_receipt, build_receipt)
    if build_status != "passed":
        return 1

    created = _run(
        [
            "docker",
            "create",
            "--init",
            "--tmpfs",
            "/tmp:rw,exec,size=2g",
            "-e",
            f"BALOOT_VERIFIER_TEST_TIMEOUT_SECONDS={args.timeout_seconds}",
            image_id,
        ]
    )
    if created.returncode != 0:
        _write(
            args.run_receipt,
            build_run_receipt(
                status="failed",
                build_receipt=build_receipt,
                create_exit_code=created.returncode,
                container_exit_code=None,
                copy_exit_code=-1,
                timeout_seconds=args.timeout_seconds,
                archive_report=archive_report,
                verifier_preflight=None,
            ),
        )
        return created.returncode
    container_id = created.stdout.strip()
    started = _run(["docker", "start", "--attach", container_id])
    inspected = _run(
        ["docker", "inspect", "--format", "{{.State.ExitCode}}", container_id]
    )
    container_exit = (
        int(inspected.stdout.strip())
        if inspected.returncode == 0
        else started.returncode
    )
    copied = _run(
        [
            "docker",
            "cp",
            f"{container_id}:/app/release/verifier-preflight.json",
            str(args.preflight_output),
        ]
    )
    preflight = (
        _evidence_identity(args.preflight_output, "verifier preflight")
        if copied.returncode == 0
        else None
    )
    status = (
        "passed"
        if container_exit == 0
        and copied.returncode == 0
        and archive_report["status"] == "passed"
        and preflight is not None
        and preflight["status"] == "passed"
        else "failed"
    )
    run_receipt = build_run_receipt(
        status=status,
        build_receipt=build_receipt,
        create_exit_code=created.returncode,
        container_exit_code=container_exit,
        copy_exit_code=copied.returncode,
        timeout_seconds=args.timeout_seconds,
        archive_report=archive_report,
        verifier_preflight=preflight,
    )
    _write(args.run_receipt, run_receipt)
    if copied.returncode == 0:
        _run(["docker", "rm", container_id])
    return 0 if status == "passed" else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--clean-root", required=True, type=Path)
    parser.add_argument("--archive-report", required=True, type=Path)
    parser.add_argument("--preflight-output", required=True, type=Path)
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument("--run-receipt", required=True, type=Path)
    parser.add_argument("--image-tag")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    if not 300 <= args.timeout_seconds <= 3600:
        parser.error("--timeout-seconds must be between 300 and 3600")
    return args


def main() -> None:
    raise SystemExit(execute(parse_args()))


if __name__ == "__main__":
    main()
