from __future__ import annotations

import contextlib
import json
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from baloot_platform.research.result_library import (
    RESULT_PROJECTION_VERSION,
    stored_result_payload_hash_v1,
)
from baloot_platform.version import __version__
from scripts import verify_live_postgres_research as proof


def _runtime_versions() -> dict[str, str]:
    return {"python": "3.13", "psycopg": "3.3.4", "psycopgPool": "3.3.1"}


def _child_preflight() -> dict[str, object]:
    return {
        **_runtime_versions(),
        "pythonNoUserSite": True,
        "balootSourceVerified": True,
        "explicitDependencyProvenanceVerified": True,
        "explicitDependencyRootCount": 1,
        "explicitDependencyModuleCount": 3,
        "jsonschemaModuleSha256": "1" * 64,
        "psycopgModuleSha256": "2" * 64,
        "psycopgPoolModuleSha256": "3" * 64,
    }


def _source_preflight() -> dict[str, object]:
    return {
        "sourceManifestSha256": "d" * 64,
        "contentInventorySha256": "e" * 64,
        "verifiedFileCount": 1485,
        "project": "baloot-platform",
        "version": "21.1.0",
        "archivePrefix": "baloot-platform-21.1.0",
        "contentVerified": True,
    }


def _runtime_preflight() -> dict[str, object]:
    return {
        "runtimeImportsVerified": True,
        "configurationBundleVerified": True,
        "packageVersion": "21.1.0",
        "bundleSha256": "f" * 64,
        "rulesSha256": "0" * 64,
    }


def _completed(
    stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        args=["mock"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _context(tmp_path: Path) -> dict[str, object]:
    return {
        "proofId": "a" * 32,
        "containerName": "balootx-pg-proof-aaaaaaaaaaaa",
        "proxyContainerName": "balootx-pg-proof-proxy-aaaaaaaaaaaa",
        "networkName": "balootx-pg-proof-net-aaaaaaaaaaaa",
        "accessNetworkName": "balootx-pg-proof-access-aaaaaaaaaaaa",
        "networkId": "e" * 64,
        "accessNetworkId": "f" * 64,
        "dockerContext": "default",
        "database": "balootx_proof_aaaaaaaaaaaa",
        "user": "balootx_proof",
        "password": "b" * 48,
        "proofRoot": tmp_path,
    }


def _inspection(context: dict[str, object], container_id: str) -> dict:
    return {
        "Id": container_id,
        "Config": {
            "Image": "postgres@sha256:" + "c" * 64,
            "Labels": {
                proof.CONTAINER_LABEL_KIND: proof.CONTAINER_LABEL_KIND_VALUE,
                proof.CONTAINER_LABEL_ID: context["proofId"],
                proof.CONTAINER_LABEL_ROLE: proof.DATABASE_CONTAINER_ROLE,
            },
        },
        "HostConfig": {
            "Binds": None,
            "RestartPolicy": {"Name": "no"},
            "Memory": 1_073_741_824,
            "NanoCpus": 2_000_000_000,
            "PidsLimit": 256,
            "SecurityOpt": ["no-new-privileges=true"],
            "NetworkMode": context["networkName"],
            "Tmpfs": dict(proof.REQUIRED_TMPFS),
            "PortBindings": {},
        },
        "Mounts": [
            {
                "Type": "tmpfs",
                "Destination": "/var/lib/postgresql/data",
            },
            {"Type": "tmpfs", "Destination": "/var/run/postgresql"},
            {"Type": "tmpfs", "Destination": "/tmp"},
        ],
        "NetworkSettings": {
            "Ports": {"5432/tcp": None},
            "Networks": {context["networkName"]: {"NetworkID": context["networkId"]}},
        },
    }


def _proxy_inspection(context: dict[str, object], container_id: str) -> dict:
    return {
        "Id": container_id,
        "Config": {
            "Image": "postgres@sha256:" + "c" * 64,
            "User": "postgres",
            "Entrypoint": ["sh"],
            "Cmd": ["-c", proof._proxy_relay_command(context)],
            "Env": ["PATH=/usr/local/bin:/usr/bin:/bin"],
            "Labels": {
                proof.CONTAINER_LABEL_KIND: proof.CONTAINER_LABEL_KIND_VALUE,
                proof.CONTAINER_LABEL_ID: context["proofId"],
                proof.CONTAINER_LABEL_ROLE: proof.PROXY_CONTAINER_ROLE,
            },
        },
        "State": {"Status": "running"},
        "HostConfig": {
            "Binds": None,
            "RestartPolicy": {"Name": "no"},
            "Memory": 67_108_864,
            "NanoCpus": 250_000_000,
            "PidsLimit": 32,
            "CapDrop": ["ALL"],
            "ReadonlyRootfs": True,
            "SecurityOpt": ["no-new-privileges=true"],
            "NetworkMode": context["accessNetworkName"],
            "Tmpfs": dict(proof.PROXY_TMPFS),
            "PortBindings": {"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": ""}]},
        },
        "Mounts": [],
        "NetworkSettings": {
            "Ports": {"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": "49152"}]},
            "Networks": {
                context["networkName"]: {"NetworkID": context["networkId"]},
                context["accessNetworkName"]: {"NetworkID": context["accessNetworkId"]},
            },
        },
    }


def test_manifest_resolution_requires_one_matching_immutable_platform() -> None:
    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "digest": "sha256:" + "a" * 64,
        "manifests": [
            {
                "digest": "sha256:" + "b" * 64,
                "platform": {"os": "linux", "architecture": "amd64"},
                "annotations": {"org.opencontainers.image.version": "17.10-alpine3.24"},
            },
            {
                "digest": "sha256:" + "c" * 64,
                "platform": {"os": "linux", "architecture": "arm64"},
            },
        ],
    }

    resolved = proof._parse_manifest(
        json.dumps(manifest).encode(),
        server_os="linux",
        server_architecture="amd64",
        expected_index_digest="sha256:" + "a" * 64,
    )

    assert resolved["indexDigest"] == "sha256:" + "a" * 64
    assert resolved["resolvedReference"] == "postgres@sha256:" + "b" * 64
    assert resolved["platformDigest"] == "sha256:" + "b" * 64
    assert resolved["declaredVersion"] == "17.10-alpine3.24"

    manifest["digest"] = "postgres:17-alpine"
    with pytest.raises(proof.ProofError, match="immutable digest"):
        proof._parse_manifest(
            json.dumps(manifest).encode(),
            server_os="linux",
            server_architecture="amd64",
            expected_index_digest="sha256:" + "a" * 64,
        )


def test_container_command_is_internal_tmpfs_labeled_and_secret_free(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    image = "postgres@sha256:" + "c" * 64

    command, environment = proof._container_run_command(context, image)
    encoded = "\0".join(command)

    assert command[0:2] == ["run", "--detach"]
    assert image == command[-1]
    assert command[command.index("--network") + 1] == context["networkName"]
    assert "--publish" not in command
    assert f"{proof.CONTAINER_LABEL_ID}={context['proofId']}" in command
    assert f"{proof.CONTAINER_LABEL_ROLE}={proof.DATABASE_CONTAINER_ROLE}" in command
    assert "--tmpfs" in command
    assert "/var/lib/postgresql/data:rw,nosuid,nodev,noexec,size=1073741824" in command
    assert "--volume" not in command
    assert "--mount" not in command
    assert str(context["password"]) not in encoded
    assert environment["POSTGRES_PASSWORD"] == context["password"]

    database_url = proof._database_url(context, 49152)
    assert "@127.0.0.1:49152/balootx_proof_aaaaaaaaaaaa" in database_url
    assert "connect_timeout=5" in database_url
    assert "statement_timeout%3D30000" in database_url
    assert "lock_timeout%3D10000" in database_url


def test_proxy_command_is_digest_pinned_loopback_and_least_privilege(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    image = "postgres@sha256:" + "c" * 64

    command = proof._proxy_container_run_command(context, image)

    assert image in command
    assert command[command.index("--publish") + 1] == "127.0.0.1::5432"
    assert command.count("--network") == 2
    assert context["networkName"] in command
    assert context["accessNetworkName"] in command
    assert "--read-only" in command
    assert command[command.index("--user") + 1] == "postgres"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--pids-limit") + 1] == "32"
    assert command[-2:] == ["-c", proof._proxy_relay_command(context)]
    assert str(context["password"]) not in "\0".join(command)


def test_network_inspection_accepts_current_docker_not_found_wording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        proof,
        "_run_docker",
        lambda *_args, **_kwargs: _completed(
            stderr=b"Error response from daemon: network proof-net not found\n",
            returncode=1,
        ),
    )
    assert proof._inspect_network("proof-net", "default") is None


def test_container_validation_rejects_persistent_mounts_or_published_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    container_id = "d" * 64
    inspected = _inspection(context, container_id)
    monkeypatch.setattr(
        proof, "_inspect_container", lambda _reference, _docker_context: inspected
    )

    value = proof._validated_container(
        container_id, context, "postgres@sha256:" + "c" * 64
    )
    assert value["publishedPortCount"] == 0
    assert value["persistentMountCount"] == 0

    inspected["Mounts"][0]["Type"] = "volume"
    with pytest.raises(proof.ProofError, match="persistent mount"):
        proof._validated_container(container_id, context, "postgres@sha256:" + "c" * 64)

    inspected["Mounts"][0]["Type"] = "tmpfs"
    inspected["NetworkSettings"]["Ports"]["5432/tcp"] = [
        {"HostIp": "127.0.0.1", "HostPort": "49152"}
    ]
    with pytest.raises(proof.ProofError, match="published a port"):
        proof._validated_container(container_id, context, "postgres@sha256:" + "c" * 64)

    inspected["NetworkSettings"]["Ports"]["5432/tcp"] = None
    inspected["HostConfig"]["Tmpfs"].pop("/tmp")
    with pytest.raises(proof.ProofError, match="writable directories"):
        proof._validated_container(container_id, context, "postgres@sha256:" + "c" * 64)


def test_container_validation_accepts_docker_tmpfs_config_without_mount_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    container_id = "d" * 64
    inspected = _inspection(context, container_id)
    inspected["Mounts"] = []
    monkeypatch.setattr(
        proof, "_inspect_container", lambda _reference, _docker_context: inspected
    )

    value = proof._validated_container(
        container_id, context, "postgres@sha256:" + "c" * 64
    )

    assert value["dataMount"] == "tmpfs"
    assert value["persistentMountCount"] == 0


def test_proxy_validation_requires_exact_loopback_least_privilege_and_networks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    container_id = "d" * 64
    inspected = _proxy_inspection(context, container_id)
    monkeypatch.setattr(
        proof, "_inspect_container", lambda _reference, _docker_context: inspected
    )

    value = proof._validated_proxy_container(
        container_id, context, "postgres@sha256:" + "c" * 64
    )
    assert value["loopbackOnly"] is True
    assert value["readOnlyRoot"] is True
    assert value["capabilitiesDropped"] is True

    inspected["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostIp"] = "0.0.0.0"
    with pytest.raises(proof.ProofError, match="loopback-only"):
        proof._validated_proxy_container(
            container_id, context, "postgres@sha256:" + "c" * 64
        )

    inspected = _proxy_inspection(context, container_id)
    inspected["HostConfig"]["CapDrop"] = []
    monkeypatch.setattr(
        proof, "_inspect_container", lambda _reference, _docker_context: inspected
    )
    with pytest.raises(proof.ProofError, match="resource or isolation"):
        proof._validated_proxy_container(
            container_id, context, "postgres@sha256:" + "c" * 64
        )

    inspected = _proxy_inspection(context, container_id)
    del inspected["NetworkSettings"]["Networks"][context["networkName"]]
    monkeypatch.setattr(
        proof, "_inspect_container", lambda _reference, _docker_context: inspected
    )
    with pytest.raises(proof.ProofError, match="resource or isolation"):
        proof._validated_proxy_container(
            container_id, context, "postgres@sha256:" + "c" * 64
        )


def test_network_validation_binds_database_and_proxy_to_exact_networks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    database_id = "d" * 64
    proxy_id = "c" * 64
    internal = {
        "Id": context["networkId"],
        "Driver": "bridge",
        "Internal": True,
        "Labels": {
            proof.CONTAINER_LABEL_KIND: proof.CONTAINER_LABEL_KIND_VALUE,
            proof.CONTAINER_LABEL_ID: context["proofId"],
            proof.CONTAINER_LABEL_ROLE: proof.INTERNAL_NETWORK_ROLE,
        },
        "Containers": {database_id: {}, proxy_id: {}},
    }
    access = {
        "Id": context["accessNetworkId"],
        "Driver": "bridge",
        "Internal": False,
        "Labels": {
            proof.CONTAINER_LABEL_KIND: proof.CONTAINER_LABEL_KIND_VALUE,
            proof.CONTAINER_LABEL_ID: context["proofId"],
            proof.CONTAINER_LABEL_ROLE: proof.ACCESS_NETWORK_ROLE,
        },
        "Options": {"com.docker.network.bridge.host_binding_ipv4": "127.0.0.1"},
        "Containers": {proxy_id: {}},
    }

    monkeypatch.setattr(
        proof,
        "_inspect_network",
        lambda reference, _docker_context: (
            internal if reference == context["networkId"] else access
        ),
    )
    internal_value = proof._validated_network(
        str(context["networkId"]),
        context,
        expected_container_ids=(database_id, proxy_id),
    )
    access_value = proof._validated_access_network(
        str(context["accessNetworkId"]),
        context,
        expected_proxy_container_id=proxy_id,
    )
    assert internal_value["attachedContainerCount"] == 2
    assert access_value["defaultHostBinding"] == "127.0.0.1"

    access["Containers"][database_id] = {}
    with pytest.raises(proof.ProofError, match="access-network isolation"):
        proof._validated_access_network(
            str(context["accessNetworkId"]),
            context,
            expected_proxy_container_id=proxy_id,
        )


def test_cleanup_requires_exact_labels_and_removes_only_exact_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    container_id = "d" * 64
    inspected = _inspection(context, container_id)
    calls: list[list[str]] = []

    inspected["Config"]["Labels"][proof.CONTAINER_LABEL_ID] = "wrong"
    monkeypatch.setattr(
        proof, "_inspect_container", lambda _reference, _docker_context: inspected
    )
    monkeypatch.setattr(
        proof,
        "_run_command",
        lambda command, **_kwargs: calls.append(list(command)) or _completed(),
    )
    with pytest.raises(proof.ProofError, match="not owned"):
        proof._cleanup_container(container_id, context)
    assert calls == []

    inspected["Config"]["Labels"][proof.CONTAINER_LABEL_ID] = context["proofId"]
    inspections = iter((inspected, None))
    monkeypatch.setattr(
        proof,
        "_inspect_container",
        lambda _reference, _docker_context: next(inspections),
    )
    result = proof._cleanup_container(container_id, context)

    assert result["status"] == "removed"
    assert calls == [
        [
            "docker",
            "--context",
            "default",
            "rm",
            "--force",
            "--volumes",
            container_id,
        ]
    ]


def test_cleanup_removes_exact_owned_anonymous_volume_and_proves_no_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    container_id = "d" * 64
    volume_name = "a" * 64
    inspected = _proxy_inspection(context, container_id)
    inspected["Mounts"] = [
        {
            "Type": "volume",
            "Name": volume_name,
            "Destination": "/var/lib/postgresql/data",
        }
    ]
    container_inspections = iter((inspected, None))
    volume_inspections = iter(({"Name": volume_name}, None))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        proof,
        "_inspect_container",
        lambda _reference, _docker_context: next(container_inspections),
    )
    monkeypatch.setattr(
        proof,
        "_inspect_volume",
        lambda _reference, _docker_context: next(volume_inspections),
    )
    monkeypatch.setattr(
        proof,
        "_run_command",
        lambda command, **_kwargs: calls.append(list(command)) or _completed(),
    )

    result = proof._cleanup_container(
        container_id,
        context,
        expected_role=proof.PROXY_CONTAINER_ROLE,
    )

    assert result["ownedPersistentVolumeCount"] == 1
    assert result["persistentVolumeResidueCount"] == 0
    assert calls[0][-4:] == ["rm", "--force", "--volumes", container_id]


def test_python_child_environment_preserves_explicit_path_order_without_duplicates(
    tmp_path: Path,
) -> None:
    dependency_a = (tmp_path / "dependency-a").resolve()
    dependency_b = (tmp_path / "dependency-b").resolve()
    dependency_a.mkdir()
    dependency_b.mkdir()
    base = {
        "KEEP_ME": "yes",
        "PYTHONNOUSERSITE": "0",
        "PYTHONPATH": os.pathsep.join(
            (str(dependency_a), str(proof.SRC), str(dependency_a), str(dependency_b))
        ),
    }

    environment = proof._python_child_environment(base_environment=base)

    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(proof.SRC.resolve()),
        str(dependency_a),
        str(dependency_b),
    ]
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["KEEP_ME"] == "yes"
    with pytest.raises(proof.ProofError, match="cannot override import isolation"):
        proof._python_child_environment(
            {"PYTHONNOUSERSITE": "0"}, base_environment=base
        )


def test_source_manifest_evidence_binds_manifest_and_exact_content(
    tmp_path: Path,
) -> None:
    content = tmp_path / "src" / "example.py"
    content.parent.mkdir()
    content.write_text("value = 1\n", encoding="utf-8", newline="\n")
    files = {
        "src/example.py": {
            "bytes": content.stat().st_size,
            "sha256": proof._sha256_file(content),
        }
    }
    manifest = {
        "schemaVersion": 1,
        "project": "baloot-platform",
        "version": "21.1.0",
        "archivePrefix": "baloot-platform-21.1.0",
        "source": {"revision": "a" * 40, "binding": "exact_git_head"},
        "fileCount": 1,
        "files": files,
    }
    manifest_path = tmp_path / "release" / "source-manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    evidence = proof._source_manifest_evidence(tmp_path)

    assert evidence == {
        "sourceManifestSha256": proof._sha256_file(manifest_path),
        "contentInventorySha256": proof._sha256_text(proof._canonical_json(files)),
        "verifiedFileCount": 1,
        "project": "baloot-platform",
        "version": "21.1.0",
        "archivePrefix": "baloot-platform-21.1.0",
        "source": {"revision": "a" * 40, "binding": "exact_git_head"},
        "contentVerified": True,
    }
    content.write_text("value = 2\n", encoding="utf-8", newline="\n")
    with pytest.raises(proof.ProofError, match="content mismatch"):
        proof._source_manifest_evidence(tmp_path)


def _write_fake_postgres_dependencies(root: Path) -> None:
    for package, version in (
        ("jsonschema", None),
        ("psycopg", "3.3.4"),
        ("psycopg_pool", "3.3.1"),
    ):
        package_root = root / package
        package_root.mkdir(parents=True)
        (package_root / "__init__.py").write_text(
            "# dependency fixture\n"
            if version is None
            else f"__version__ = {version!r}\n",
            encoding="utf-8",
            newline="\n",
        )


def test_python_child_preflight_uses_explicit_dependencies_with_user_site_disabled(
    tmp_path: Path,
) -> None:
    dependency_root = tmp_path / "explicit-dependencies"
    _write_fake_postgres_dependencies(dependency_root)
    base = dict(os.environ)
    base.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(dependency_root),
        }
    )

    result = proof._probe_python_child_dependencies(base_environment=base)

    assert result == {
        "python": proof.sys.version.split()[0],
        "psycopg": "3.3.4",
        "psycopgPool": "3.3.1",
        "pythonNoUserSite": True,
        "balootSourceVerified": True,
        "explicitDependencyProvenanceVerified": True,
        "explicitDependencyRootCount": 1,
        "explicitDependencyModuleCount": 3,
        "jsonschemaModuleSha256": proof._sha256_file(
            dependency_root / "jsonschema" / "__init__.py"
        ),
        "psycopgModuleSha256": proof._sha256_file(
            dependency_root / "psycopg" / "__init__.py"
        ),
        "psycopgPoolModuleSha256": proof._sha256_file(
            dependency_root / "psycopg_pool" / "__init__.py"
        ),
    }


def test_python_child_preflight_without_explicit_dependencies_fails_closed() -> None:
    base = dict(os.environ)
    base.update({"PYTHONNOUSERSITE": "1", "PYTHONPATH": str(proof.SRC)})

    with pytest.raises(
        proof.ProofError,
        match="dependency preflight failed|did not load from an explicit dependency root",
    ):
        proof._probe_python_child_dependencies(base_environment=base)


def test_full_runtime_preflight_imports_and_validates_current_bundle() -> None:
    result = proof._probe_full_runtime_wiring()

    assert result["runtimeImportsVerified"] is True
    assert result["configurationBundleVerified"] is True
    assert result["packageVersion"] == __version__
    assert len(str(result["bundleSha256"])) == 64
    assert len(str(result["rulesSha256"])) == 64


@pytest.mark.parametrize(
    ("substituted_field", "message"),
    (
        (
            "jsonschemaPath",
            "jsonschema did not load from an explicit dependency root",
        ),
        ("psycopgPath", "psycopg did not load from an explicit dependency root"),
        ("balootPlatformPath", "Baloot source was substituted"),
    ),
)
def test_python_child_preflight_rejects_substituted_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substituted_field: str,
    message: str,
) -> None:
    dependency_root = tmp_path / "explicit-dependencies"
    outsider_root = tmp_path / "substituted"
    _write_fake_postgres_dependencies(dependency_root)
    outsider_root.mkdir()
    outsider = outsider_root / "__init__.py"
    outsider.write_text("# substitution\n", encoding="utf-8", newline="\n")
    payload = {
        "python": proof.sys.version.split()[0],
        "noUserSite": True,
        "balootPlatformPath": str(
            (proof.SRC / "baloot_platform" / "__init__.py").resolve()
        ),
        "jsonschemaPath": str(
            (dependency_root / "jsonschema" / "__init__.py").resolve()
        ),
        "psycopg": "3.3.4",
        "psycopgPath": str((dependency_root / "psycopg" / "__init__.py").resolve()),
        "psycopgPool": "3.3.1",
        "psycopgPoolPath": str(
            (dependency_root / "psycopg_pool" / "__init__.py").resolve()
        ),
    }
    payload[substituted_field] = str(outsider.resolve())
    monkeypatch.setattr(
        proof,
        "_run_command",
        lambda *_args, **_kwargs: _completed(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ),
    )
    base = dict(os.environ)
    base.update({"PYTHONNOUSERSITE": "1", "PYTHONPATH": str(dependency_root)})

    with pytest.raises(proof.ProofError, match=message):
        proof._probe_python_child_dependencies(base_environment=base)


def test_migration_child_preserves_explicit_dependency_path_and_install_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dependency_root = (tmp_path / "explicit-dependencies").resolve()
    dependency_root.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(dependency_root))
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    captured: dict[str, object] = {}

    def fail_with_install_message(command, **kwargs):
        captured["command"] = list(command)
        captured["environment"] = dict(kwargs["environment"])
        raise proof.ProofError(
            'command failed: pip install "baloot-platform[postgres]"'
        )

    monkeypatch.setattr(proof, "_run_command", fail_with_install_message)

    with pytest.raises(proof.ProofError, match="pip install"):
        proof._run_migration_once(
            "postgresql://secret",
            tmp_path / "snapshot",
            (),
            ("secret",),
        )

    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(proof.SRC.resolve()),
        str(dependency_root),
    ]
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert captured["command"][:3] == [proof.sys.executable, "-B", "-s"]


def test_migration_proof_runs_twice_and_rejects_schema_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = [{"name": "001.sql", "sha256": "a" * 64}]
    snapshot_root = tmp_path / "snapshot"
    calls: list[list[str]] = []
    monkeypatch.setattr(
        proof, "_stage_migrations", lambda _proof_root: (snapshot_root, inventory)
    )
    monkeypatch.setattr(proof, "_migration_inventory", lambda *_args: inventory)
    monkeypatch.setattr(
        proof,
        "_migration_schema_contract",
        lambda _root: {"sha256": "f" * 64},
    )
    monkeypatch.setattr(
        proof,
        "_run_migration_once",
        lambda _url, _root, names, _secrets: calls.append(list(names))
        or {
            "applied": 1,
            "executedInventorySha256": proof._sha256_text(
                proof._canonical_json(inventory)
            ),
        },
    )
    monkeypatch.setattr(
        proof,
        "_schema_fingerprint",
        lambda _url: {"sha256": "c" * 64, "counts": {"columns": 1}},
    )
    monkeypatch.setattr(proof, "_assert_relations", lambda *_args: None)
    monkeypatch.setattr(proof, "_assert_migration_constraints", lambda *_args: None)
    monkeypatch.setattr(proof, "_assert_schema_contract", lambda *_args: {})

    result = proof._prove_migrations(
        "postgresql://secret", tmp_path, secrets_to_remove=["secret"]
    )

    assert calls == [["001.sql"], ["001.sql"]]
    assert result == {
        "inventory": inventory,
        "migrationCount": 1,
        "executedInventorySha256": proof._sha256_text(proof._canonical_json(inventory)),
        "requiredMigrationHashes": dict(proof.REQUIRED_MIGRATION_HASHES),
        "passes": [
            {
                "applied": 1,
                "executedInventorySha256": proof._sha256_text(
                    proof._canonical_json(inventory)
                ),
            },
            {
                "applied": 1,
                "executedInventorySha256": proof._sha256_text(
                    proof._canonical_json(inventory)
                ),
            },
        ],
        "schemaFingerprint": {
            "sha256": "c" * 64,
            "counts": {"columns": 1},
        },
        "schemaContract": {},
        "immutableInventory": True,
        "idempotent": True,
    }

    fingerprints = iter(
        (
            {"sha256": "c" * 64, "counts": {}},
            {"sha256": "d" * 64, "counts": {}},
        )
    )
    monkeypatch.setattr(proof, "_schema_fingerprint", lambda _url: next(fingerprints))
    with pytest.raises(proof.ProofError, match="not schema-idempotent"):
        proof._prove_migrations(
            "postgresql://secret", tmp_path, secrets_to_remove=["secret"]
        )

    changed = [{"name": "001.sql", "sha256": "e" * 64}]
    monkeypatch.setattr(
        proof,
        "_schema_fingerprint",
        lambda _url: {"sha256": "c" * 64, "counts": {"columns": 1}},
    )

    def migration_inventory(root: Path = proof.ROOT) -> list[dict[str, object]]:
        return changed if root == proof.ROOT else inventory

    monkeypatch.setattr(proof, "_migration_inventory", migration_inventory)
    with pytest.raises(proof.ProofError, match="source migrations changed"):
        proof._prove_migrations(
            "postgresql://secret", tmp_path, secrets_to_remove=["secret"]
        )


def test_current_migration_stage_freezes_complete_inventory_and_required_tail(
    tmp_path: Path,
) -> None:
    snapshot_root, inventory = proof._stage_migrations(tmp_path)

    assert len(inventory) == 29
    assert proof._migration_inventory(snapshot_root) == inventory
    assert {
        item["name"]: item["sha256"]
        for item in inventory
        if item["name"] in proof.REQUIRED_MIGRATION_HASHES
    } == proof.REQUIRED_MIGRATION_HASHES
    assert inventory[-1]["name"] == "030_oidc_issuer_transition.sql"
    assert (
        proof.REQUIRED_MIGRATION_HASHES["029_shared_run_game_access.sql"]
        == "4612ba64ff50b969fc978a9671267aca8afbd12d817de0d586b5fab4edab8b5b"
    )


def test_policy_lifecycle_replays_canonical_file_and_statement_order(
    tmp_path: Path,
) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    first = migrations / "001_policy_create.sql"
    second = migrations / "002_policy_replace.sql"
    third = migrations / "003_policy_recreate.sql"
    first.write_text(
        """
        CREATE POLICY legacy_read ON Configuration.Profile_Revisions;
        CREATE POLICY retained_policy ON app.records;
        CREATE POLICY shared_name ON app.first_relation;
        CREATE POLICY shared_name ON app.second_relation;
        """,
        encoding="utf-8",
    )
    second.write_text(
        """
        dRoP
          PoLiCy IF
          EXISTS LEGACY_READ
          ON configuration.profile_revisions;
        DROP POLICY IF EXISTS shared_name ON app.first_relation;
        CREATE POLICY transient_policy ON records;
        DROP POLICY IF EXISTS transient_policy ON public.records;
        """,
        encoding="utf-8",
    )
    third.write_text(
        """
        CREATE
          POLICY replacement_read
          ON CONFIGURATION.PROFILE_REVISIONS;
        DROP POLICY IF EXISTS retained_policy ON app.records;
        CREATE POLICY RETAINED_POLICY ON APP.RECORDS;
        """,
        encoding="utf-8",
    )

    policies = proof._migration_policy_identities((third, first, second))

    assert policies == {
        ("app.records", "retained_policy"),
        ("app.second_relation", "shared_name"),
        ("configuration.profile_revisions", "replacement_read"),
    }


def test_current_migration_policy_contract_is_final_and_deterministic(
    tmp_path: Path,
) -> None:
    snapshot_root, inventory = proof._stage_migrations(tmp_path)

    first = proof._migration_schema_contract(snapshot_root)
    second = proof._migration_schema_contract(snapshot_root)
    policies = set(map(tuple, first["policies"]))
    legacy_configuration_policies = {
        ("configuration.bundle_revisions", "configuration_bundle_admin"),
        ("configuration.bundle_revisions", "configuration_bundle_read"),
        ("configuration.bundle_revisions", "configuration_bundle_read_v20"),
        ("configuration.bundle_revisions", "configuration_bundle_write_v20"),
        ("configuration.events", "configuration_events_admin"),
        ("configuration.events", "configuration_events_read"),
        ("configuration.profile_revisions", "configuration_profile_admin"),
        ("configuration.profile_revisions", "configuration_profile_read"),
        ("configuration.profile_revisions", "configuration_profile_read_v20"),
        ("configuration.profile_revisions", "configuration_profile_write_v20"),
    }
    configuration_v25_policies = {
        ("configuration.bundle_revisions", "configuration_bundle_context_read_v25"),
        ("configuration.bundle_revisions", "configuration_bundle_write_v25"),
        ("configuration.events", "configuration_events_read_v25"),
        ("configuration.events", "configuration_events_write_v25"),
        ("configuration.profile_revisions", "configuration_profile_context_read_v25"),
        ("configuration.profile_revisions", "configuration_profile_write_v25"),
    }
    unrelated_policies = {
        ("hitl.blind_assignments", "hitl_assignment_reviewer"),
        ("hitl.consent", "hitl_consent_subject"),
        ("hitl.review_cases", "hitl_case_owner"),
        ("hitl.review_submissions", "hitl_submission_reviewer"),
        ("live.outbox", "live_outbox_service_v20"),
        ("live.room_events", "live_event_read_v20"),
        ("live.room_events", "live_event_write_v20"),
        ("live.rooms", "live_room_read_v20"),
        ("live.rooms", "live_room_write_v20"),
        ("security.external_identities", "external_identity_owner_v20"),
        ("security.passkey_credentials", "passkey_owner_v20"),
        ("security.websocket_tickets", "websocket_ticket_service_v20"),
    }

    assert len(inventory) == 28
    assert first == second
    assert (
        first["sha256"]
        == "2c0d7519af7b36a7f63bef34c25612c3f49ab0ab3710beeff397d891a907c983"
    )
    assert policies.isdisjoint(legacy_configuration_policies)
    assert configuration_v25_policies.issubset(policies)
    assert unrelated_policies.issubset(policies)
    assert policies == configuration_v25_policies | unrelated_policies


def test_schema_contract_reports_missing_and_unexpected_policy_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        def __init__(self, rows: tuple[tuple[object, ...], ...]) -> None:
            self._rows = rows

        def fetchall(self) -> list[tuple[object, ...]]:
            return list(self._rows)

    class Connection:
        def __enter__(self) -> Connection:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def execute(self, sql: str) -> Result:
            if "relation.relkind IN ('r','p')" in sql and "relrowsecurity" not in sql:
                return Result((("public", "records"),))
            if "FROM pg_constraint" in sql:
                return Result(())
            if "FROM pg_proc" in sql:
                return Result(())
            if "FROM information_schema.triggers" in sql:
                return Result(())
            if "FROM pg_policies" in sql:
                return Result((("public", "records", "unexpected_policy"),))
            if "relrowsecurity" in sql:
                return Result((("public", "records", False, False),))
            raise AssertionError(sql)

    monkeypatch.setattr(proof, "_connect", lambda _url: Connection())
    expected = {
        "tables": ["public.records"],
        "constraints": [],
        "functions": [],
        "triggers": [],
        "policies": [["public.records", "required_policy"]],
        "rlsEnabled": [],
        "rlsForced": [],
        "sha256": "f" * 64,
    }

    with pytest.raises(proof.ProofError) as error:
        proof._assert_schema_contract("postgresql://unit", expected)

    assert str(error.value) == (
        "PostgreSQL policies do not match the frozen migration contract; "
        "missing=[public.records:required_policy]; "
        "unexpected=[public.records:unexpected_policy]"
    )


@pytest.mark.parametrize(
    ("fault", "message"),
    (
        ("missing-v25", "required PostgreSQL migration is missing"),
        ("substituted-v25", "migration identity mismatch"),
        ("wrong-v25-mirror", "not byte-identical"),
    ),
)
def test_migration_stage_rejects_missing_substituted_or_wrong_v25_mirror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    message: str,
) -> None:
    authored = tmp_path / "authored" / "migrations"
    packaged = tmp_path / "packaged"
    authored.mkdir(parents=True)
    packaged.mkdir()
    for name in proof.REQUIRED_MIGRATION_HASHES:
        if fault == "missing-v25" and name.startswith("025_"):
            continue
        payload = (proof.ROOT / "migrations" / name).read_bytes()
        if fault == "substituted-v25" and name.startswith("025_"):
            payload = b"substituted-v25\n"
        (authored / name).write_bytes(payload)
        mirror = payload
        if fault == "wrong-v25-mirror" and name.startswith("025_"):
            mirror = b"wrong-packaged-v25\n"
        (packaged / name).write_bytes(mirror)
    monkeypatch.setattr(proof, "ROOT", authored.parent)
    monkeypatch.setattr(proof, "PACKAGED_MIGRATIONS", packaged)

    with pytest.raises(proof.ProofError, match=message):
        proof._stage_migrations(tmp_path / "proof")


def test_migration_proof_rejects_stale_executed_inventory_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = [{"name": "001.sql", "sha256": "a" * 64}]
    snapshot_root = tmp_path / "snapshot"
    monkeypatch.setattr(
        proof, "_stage_migrations", lambda _proof_root: (snapshot_root, inventory)
    )
    monkeypatch.setattr(proof, "_migration_inventory", lambda *_args: inventory)
    monkeypatch.setattr(
        proof, "_migration_schema_contract", lambda _root: {"sha256": "f" * 64}
    )
    monkeypatch.setattr(
        proof,
        "_run_migration_once",
        lambda *_args: {"applied": 1, "inventorySha256": "b" * 64},
    )

    with pytest.raises(proof.ProofError, match="executed migration inventory"):
        proof._prove_migrations(
            "postgresql://secret", tmp_path, secrets_to_remove=["secret"]
        )


def _install_full_wiring_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    wrong_store: bool = False,
    unresolved: bool = False,
    invalid_event_chain: bool = False,
    authority_drift: str | None = None,
) -> tuple[SimpleNamespace, list[tuple[str, str]]]:
    import baloot_platform.config as config_module
    import baloot_platform.service.container as container_module
    import baloot_platform.storage.request_context as request_context
    from baloot_platform.configuration import PostgresConfigurationStore
    from baloot_platform.research.postgres_store import PostgresResearchStoreV1

    pool = SimpleNamespace(closed=False)
    configuration_store: object
    if wrong_store:
        configuration_store = SimpleNamespace(pool=pool)
    else:
        configuration_store = object.__new__(PostgresConfigurationStore)
        configuration_store.pool = pool  # type: ignore[attr-defined]
    research_store = object.__new__(PostgresResearchStoreV1)
    bundle_revision_id = "CBR-LIVE-PROOF"
    families = tuple(config_module.PROFILE_FAMILIES)
    assert len(families) == 32
    profile_revision_ids = {
        family: f"CPR-LIVE-PROOF-{family.upper()}" for family in families
    }
    profile_hashes = {
        family: proof._hash(f"full-wiring-profile:{family}") for family in families
    }
    packaged_profile = {
        "id": "baloot_default",
        "version": "21.1.22",
        "status": "governed_candidate",
    }
    active_profile = {
        "id": packaged_profile["id"],
        "version": packaged_profile["version"],
        "status": "published",
    }
    packaged_components = {
        family: SimpleNamespace(sha256=profile_hashes[family]) for family in families
    }
    active_components = {
        family: SimpleNamespace(sha256=profile_hashes[family]) for family in families
    }
    packaged = SimpleNamespace(
        sha256=proof._hash("full-wiring-packaged:governed_candidate"),
        rules=packaged_components["rules"],
        components=packaged_components,
        mode="full",
        bundle_profile={"profile": packaged_profile},
    )
    ortools_revision_id = "CBR-LIVE-ORTOOLS-PROOF"
    ortools_packaged = SimpleNamespace(
        sha256=proof._hash("full-wiring-ortools-packaged"),
        rules=packaged_components["rules"],
        components=packaged_components,
        mode="full",
        bundle_profile={
            "profile": {
                "id": "baloot_ortools_research_v4",
                "version": "22.0.0",
                "status": "governed_candidate",
            }
        },
    )
    ortools_runtime = SimpleNamespace(
        sha256=proof._hash("full-wiring-ortools-runtime"),
        component=lambda family: (
            {
                "scenarioResearchV4": {
                    "backend": "ortools_cp_sat_v2",
                    "proofBudgets": {
                        "profiles": {
                            "late_game_exact_v1": {
                                "maximumWallSeconds": 60.0
                            }
                        }
                    },
                    "searchBudgets": {
                        "profiles": {
                            "late_game_exact_v1": {
                                "maximumWallSeconds": 60.0
                            }
                        }
                    },
                }
            }
            if family == "challenge"
            else {}
        ),
    )
    ortools_resolved = {
        "bundle": {
            "revisionId": ortools_revision_id,
            "status": "published",
        }
    }
    active = SimpleNamespace(
        sha256=proof._hash("full-wiring-active:published"),
        rules=active_components["rules"],
        components=active_components,
        mode="full",
        bundle_profile={"profile": active_profile},
    )
    event_head = "e" * 64
    valid_chain = {
        "items": [
            {
                "aggregateType": "bundle",
                "aggregateId": bundle_revision_id,
            }
        ],
        "verification": {
            "valid": True,
            "issues": [],
            "eventCount": 1,
            "eventHead": event_head,
        },
    }
    resolved = {
        "bundle": {
            "revisionId": bundle_revision_id,
            "bundleId": packaged_profile["id"],
            "version": packaged_profile["version"],
            "mode": "full",
            "status": "published",
            "componentRevisions": dict(profile_revision_ids),
        },
        "profiles": {family: {"profile": {"id": family}} for family in families},
        "profileRevisions": {
            family: {
                "revisionId": profile_revision_ids[family],
                "payloadSha256": profile_hashes[family],
            }
            for family in families
        },
        "effectiveHash": "d" * 64,
        "audit": dict(valid_chain["verification"]),
    }

    if authority_drift == "row_id":
        resolved["bundle"]["bundleId"] = "drifted_bundle"
    elif authority_drift == "active_id":
        active_profile["id"] = "drifted_bundle"
    elif authority_drift == "row_version":
        resolved["bundle"]["version"] = "99.0.0"
    elif authority_drift == "active_version":
        active_profile["version"] = "99.0.0"
    elif authority_drift == "row_mode":
        resolved["bundle"]["mode"] = "sun_only"
    elif authority_drift == "active_mode":
        active.mode = "sun_only"
    elif authority_drift == "packaged_status":
        packaged_profile["status"] = "published"
    elif authority_drift == "row_status":
        resolved["bundle"]["status"] = "draft"
    elif authority_drift == "active_status":
        active_profile["status"] = "draft"
    elif authority_drift == "profiles_family":
        resolved["profiles"].pop("performance")
    elif authority_drift == "revisions_family":
        resolved["profileRevisions"].pop("performance")
    elif authority_drift == "component_revisions_family":
        resolved["bundle"]["componentRevisions"].pop("performance")
    elif authority_drift == "packaged_family":
        packaged_components.pop("performance")
    elif authority_drift == "active_family":
        active_components.pop("performance")
    elif authority_drift == "active_payload":
        active_components["analysis"].sha256 = "0" * 64
    elif authority_drift == "revision_payload":
        resolved["profileRevisions"]["analysis"]["payloadSha256"] = "0" * 64
    elif authority_drift is not None:
        raise AssertionError(f"unknown full-wiring authority drift: {authority_drift}")

    def resolve(**_kwargs):
        if unresolved:
            raise KeyError("unresolved")
        return resolved

    configuration = SimpleNamespace(resolve=resolve)

    def events_with_verification(aggregate_type: str, aggregate_id: str) -> dict:
        if aggregate_type == "bundle":
            if invalid_event_chain:
                return {
                    **valid_chain,
                    "verification": {
                        **valid_chain["verification"],
                        "valid": False,
                        "issues": ["hash:1"],
                    },
                }
            return valid_chain
        return {
            "items": [
                {
                    "aggregateType": aggregate_type,
                    "aggregateId": aggregate_id,
                }
            ],
            "verification": {
                "valid": True,
                "issues": [],
                "eventCount": 1,
                "eventHead": "f" * 64,
            },
        }

    if not wrong_store:
        configuration_store.events_with_verification = (  # type: ignore[attr-defined]
            events_with_verification
        )
    shutdown_calls: list[bool] = []

    def shutdown() -> None:
        pool.closed = True
        shutdown_calls.append(True)

    container = SimpleNamespace(
        research_store=research_store,
        configuration_store=configuration_store,
        configuration=configuration,
        research=SimpleNamespace(statistics=lambda **_kwargs: {"resultCount": 1}),
        shutdown=shutdown,
    )
    monkeypatch.setattr(
        container_module, "build_container", lambda **_kwargs: container
    )
    monkeypatch.setattr(
        config_module,
        "build_bundle",
        lambda _root, *, bundle_path=None: (
            ortools_packaged if bundle_path else packaged
        ),
    )
    monkeypatch.setattr(
        config_module,
        "profile_bundle_from_resolved",
        lambda value, **_kwargs: (
            ortools_runtime if value is ortools_resolved else active
        ),
    )
    monkeypatch.setattr(
        proof,
        "_publish_live_ortools_bundle",
        lambda _configuration, *, active_resolved, packaged: (
            ortools_resolved
            if active_resolved is resolved and packaged is ortools_packaged
            else (_ for _ in ()).throw(AssertionError("wrong OR-Tools authority"))
        ),
    )

    class ContextConnection:
        def execute(self, _sql: str):
            identity = request_context.CURRENT_DATABASE_IDENTITY.get()
            return SimpleNamespace(
                fetchone=lambda: (identity.subject_id, identity.role)
            )

    @contextlib.contextmanager
    def contextual_connection(_pool):
        yield ContextConnection()

    monkeypatch.setattr(request_context, "contextual_connection", contextual_connection)
    monkeypatch.setattr(
        proof,
        "_prove_configuration_v25_schema",
        lambda _url: {"contractSha256": "c" * 64},
    )
    immutable_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        proof,
        "_assert_configuration_immutability",
        lambda _pool,
        *,
        relation,
        revision_id,
        expected_message: immutable_calls.append((relation, revision_id)),
    )
    monkeypatch.setattr(
        proof,
        "_wait_for_no_client_sessions",
        lambda _url: {"remainingClientSessions": 0},
    )
    scenario_v4 = {
        "recipeCode": "RCP-PGV4PROOF",
        "runCode": "RUN-PGV4PROOF01",
        "resultCode": "LAF-PGV4PROOF",
        "resultHash": "7" * 64,
        "exactStateId": "STATE-1234567890ABCDEF1234",
        "proofOutcomeHash": "8" * 64,
        "proofArtifactSha256": "3" * 64,
        "winningRootActionCount": 0,
        "projectionVersion": RESULT_PROJECTION_VERSION,
        "facetCount": 14,
        "observerSafeExactState": True,
        "artifactOwnershipVerified": True,
        "bundleAuthorityVerified": True,
        "storedResultIdentityBound": True,
    }
    monkeypatch.setattr(
        proof,
        "_prove_research_scenario_v4_wiring",
        lambda *_args, **_kwargs: dict(scenario_v4),
    )
    container.shutdown_calls = shutdown_calls
    container.authority_fixture = {
        "families": families,
        "profileHashes": profile_hashes,
        "packagedSha256": packaged.sha256,
        "activeSha256": active.sha256,
        "ortoolsPackagedSha256": ortools_packaged.sha256,
        "ortoolsRuntimeSha256": ortools_runtime.sha256,
        "identity": {
            "id": "baloot_default",
            "version": "21.1.22",
            "mode": "full",
        },
    }
    return container, immutable_calls


def test_live_ortools_bundle_reuses_active_profiles_and_selects_exact_revision() -> (
    None
):
    active = {
        "profiles": {
            "rules": {"profile": {"id": "rules"}},
            "challenge": {"profile": {"id": "challenge-v1"}},
        },
        "profileRevisions": {
            "rules": {"revisionId": "CPR-RULES"},
            "challenge": {"revisionId": "CPR-CHALLENGE-V1"},
        },
    }
    packaged = SimpleNamespace(
        rules=SimpleNamespace(data=active["profiles"]["rules"]),
        components={
            "rules": SimpleNamespace(data=active["profiles"]["rules"]),
            "challenge": SimpleNamespace(data={"profile": {"id": "challenge-v2"}}),
        },
        bundle_profile={
            "profile": {"id": "baloot_ortools_research_v4", "version": "22.0.0"}
        },
        mode="full",
    )
    calls: list[tuple[str, dict]] = []

    class Configuration:
        def create_profile(self, **kwargs):
            calls.append(("profile", kwargs))
            return {"revisionId": "CPR-CHALLENGE-V2"}

        def create_bundle(self, **kwargs):
            calls.append(("bundle", kwargs))
            return {"revisionId": "CBR-ORTOOLS-V2"}

        def resolve(self, *, revision_id):
            assert revision_id == "CBR-ORTOOLS-V2"
            return {"bundle": {"revisionId": revision_id, "status": "published"}}

    resolved = proof._publish_live_ortools_bundle(
        Configuration(), active_resolved=active, packaged=packaged
    )

    assert resolved["bundle"]["revisionId"] == "CBR-ORTOOLS-V2"
    assert [kind for kind, _kwargs in calls] == ["profile", "bundle"]
    assert calls[0][1]["family"] == "challenge"
    assert calls[1][1]["component_revisions"] == {
        "rules": "CPR-RULES",
        "challenge": "CPR-CHALLENGE-V2",
    }
    assert calls[1][1]["bundle_id"] == "baloot_ortools_research_v4"
    assert calls[1][1]["publish"] is True


def test_full_wiring_proves_postgres_configuration_authority_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container, immutable_calls = _install_full_wiring_fixture(monkeypatch)

    result = proof._prove_full_wiring("postgresql://unit", tmp_path, "RUN-LIVE-ADMIT")
    fixture = container.authority_fixture
    authority_sha256 = proof._sha256_text(
        proof._canonical_json(
            {
                "schemaVersion": 1,
                "bundle": fixture["identity"],
                "profilePayloadSha256": fixture["profileHashes"],
            }
        )
    )

    assert result == {
        "postgresResearchStore": True,
        "postgresConfigurationStore": True,
        "configurationServiceRlsContext": True,
        "packagedBundleResolved": True,
        "activeBundleRevisionId": "CBR-LIVE-PROOF",
        "packagedBundleSha256": fixture["packagedSha256"],
        "activeBundleSha256": fixture["activeSha256"],
        "bundleLifecycleStatuses": {
            "packaged": "governed_candidate",
            "runtimeRow": "published",
            "active": "published",
        },
        "authoritySha256": authority_sha256,
        "profileAuthorityCount": 32,
        "configurationEventChains": {
            "bundle": {"eventCount": 1, "eventHead": "e" * 64},
            "profiles": {
                family: {"eventCount": 1, "eventHead": "f" * 64}
                for family in fixture["families"]
            },
        },
        "configurationV25Schema": {"contractSha256": "c" * 64},
        "publishedImmutabilityEnforced": True,
        "ortoolsResearchBundle": {
            "revisionId": "CBR-LIVE-ORTOOLS-PROOF",
            "backend": "ortools_cp_sat_v2",
            "authoredBundleSha256": fixture["ortoolsPackagedSha256"],
            "runtimeBundleSha256": fixture["ortoolsRuntimeSha256"],
            "activeLegacyParity": True,
        },
        "researchScenarioV4": {
            "recipeCode": "RCP-PGV4PROOF",
            "runCode": "RUN-PGV4PROOF01",
            "resultCode": "LAF-PGV4PROOF",
            "resultHash": "7" * 64,
            "exactStateId": "STATE-1234567890ABCDEF1234",
            "proofOutcomeHash": "8" * 64,
            "proofArtifactSha256": "3" * 64,
            "winningRootActionCount": 0,
            "projectionVersion": RESULT_PROJECTION_VERSION,
            "facetCount": 14,
            "observerSafeExactState": True,
            "artifactOwnershipVerified": True,
            "bundleAuthorityVerified": True,
            "storedResultIdentityBound": True,
        },
        "statisticsReadable": True,
        "configurationPoolClosed": True,
        "configurationSessionsClosed": {"remainingClientSessions": 0},
    }
    assert immutable_calls == [
        ("profile_revisions", "CPR-LIVE-PROOF-RULES"),
        ("bundle_revisions", "CBR-LIVE-PROOF"),
    ]
    assert container.shutdown_calls == [True]


@pytest.mark.parametrize(
    "missing_binding",
    (
        "artifactOwnershipVerified",
        "bundleAuthorityVerified",
        "storedResultIdentityBound",
    ),
)
def test_full_wiring_requires_every_research_result_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_binding: str,
) -> None:
    _install_full_wiring_fixture(monkeypatch)
    bindings = {
        "artifactOwnershipVerified": True,
        "bundleAuthorityVerified": True,
        "storedResultIdentityBound": True,
    }
    bindings[missing_binding] = False
    monkeypatch.setattr(
        proof,
        "_prove_research_scenario_v4_wiring",
        lambda *_args, **_kwargs: bindings,
    )

    with pytest.raises(proof.ProofError, match="result bindings are incomplete"):
        proof._prove_full_wiring("postgresql://unit", tmp_path, "RUN-LIVE-ADMIT")


def test_research_scenario_v4_wiring_proves_postgres_projection_and_safe_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_hash = "b" * 64
    rules_hash = "c" * 64
    recipe_hash = "6" * 64
    run_hash = "5" * 64
    configuration = {
        "semanticBundleHash": bundle_hash,
        "identities": {"rules": {"sha256": rules_hash}},
    }
    configuration_hash = proof._sha256_text(proof._canonical_json(configuration))
    artifacts = {
        key: {"sha256": character * 64}
        for key, character in zip(
            (
                "constraintEvidence",
                "exactState",
                "proofOutcome",
                "scenarioSatisfaction",
            ),
            "1234",
            strict=True,
        )
    }
    result_identity = {
        "runCode": "RUN-PGV4PROOF01",
        "runHash": run_hash,
        "recipeCode": "RCP-PGV4PROOF",
        "recipeHash": recipe_hash,
        "configurationAuthorityHash": configuration_hash,
        "scenarioSatisfactionHash": "a" * 64,
        "constraintEvidenceHash": "9" * 64,
        "proofOutcomeHash": "8" * 64,
        "artifactHashes": {
            key: reference["sha256"] for key, reference in sorted(artifacts.items())
        },
    }
    result = {
        "schemaVersion": 2,
        "resultCode": "LAF-PGV4PROOF",
        "resultHash": "7" * 64,
        "disposition": "qualified",
        "scenarioProjection": {
            "status": "satisfied",
            "decisionKind": "play",
            "exactStateId": "STATE-1234567890ABCDEF1234",
            "scenarioCategories": ["auction", "play", "projects", "proof"],
        },
        "researchAuthority": {
            "backendId": "ortools_cp_sat_v2",
            "qualificationAuthority": "cp_sat",
            "bundleRevisionId": "CBR-LIVE-PROOF",
        },
        "runtimeCompatibility": {"status": "passed"},
        "proofOutcome": {
            "winningRootActionCount": 0,
            "completionStatus": "exact_complete",
            "semanticSha256": "8" * 64,
            "artifactSha256": "3" * 64,
            "sha256": "e" * 64,
            "identities": {
                "bundleHash": bundle_hash,
                "rulesHash": rules_hash,
            },
        },
        "scenarioSatisfaction": {"sha256": "a" * 64},
        "constraintEvidence": {"sha256": "9" * 64},
        "artifacts": artifacts,
        "runCode": "RUN-PGV4PROOF01",
        "recipeCode": "RCP-PGV4PROOF",
        "provenance": {
            "recipeHash": recipe_hash,
            "configurationAuthorityHash": configuration_hash,
            "resultIdentity": result_identity,
        },
        "libraryProjection": {"observerSeats": ["north"]},
    }

    class Research:
        def __init__(self):
            self.filters = None
            self.exact_filters = None
            self.executed = []

        def create_recipe(self, payload, **_kwargs):
            assert payload["resources"]["solverTimeoutSeconds"] == 60.0
            assert payload["scenario"]["constraints"]["children"][2]["arguments"] == {
                "team": "north_south",
                "owner": "north",
                "kind": "hundred_sequence",
                "value": 20,
                "state": "awarded",
            }
            return {
                "code": "RCP-PGV4PROOF",
                "sha256": recipe_hash,
                "configuration": configuration,
            }

        def _submit_recipe_run(self, *_args, **kwargs):
            assert kwargs["schedule"] is False
            return {
                "runCode": "RUN-PGV4PROOF01",
                "runHash": run_hash,
                "recipeHash": recipe_hash,
                "configurationAuthority": configuration,
            }

        def _execute(self, code):
            self.executed.append(code)

        def get_run(self, _code):
            return {
                "status": "completed",
                "resultCount": 1,
                "qualifiedCount": 1,
            }

        def list_results(self, **filters):
            if "exact_state_id" in filters:
                self.exact_filters = filters
            else:
                self.filters = filters
            return {"items": [result], "total": 1}

        def get_exact_state(self, _code, *, observer_seat):
            assert observer_seat == "north"
            return {
                "observerSeat": "north",
                "publicState": {
                    "hands": {
                        "north": ["CA", "D10"],
                        "west": {"count": 2},
                        "south": {"count": 2},
                        "east": {"count": 2},
                    }
                },
            }

    required_facets = [
        ("configuration_bundle", "cbr-live-proof"),
        ("decision_kind", "play"),
        ("exact_state_id", "state-1234567890abcdef1234"),
        ("exact_state_available", "true"),
        ("project_kind", "hundred_sequence"),
        ("project_state", "awarded"),
        ("proof_completion", "exact_complete"),
        ("qualification_authority", "cp_sat"),
        ("research_backend", "ortools_cp_sat_v2"),
        ("runtime_compatibility_status", "passed"),
        ("scenario_category", "auction"),
        ("scenario_category", "play"),
        ("scenario_category", "projects"),
        ("scenario_category", "proof"),
        ("scenario_origin", "exact_trace_prefix"),
        ("scenario_status", "satisfied"),
        ("variant_collection", "run-pgv4proof01"),
        ("winning_root_action_band", "0"),
    ]

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql, _params):
            if "FROM research.results" in sql:
                return SimpleNamespace(
                    fetchone=lambda: (
                        result["resultCode"],
                        result["resultHash"],
                        result,
                        stored_result_payload_hash_v1(result),
                        {
                            "resultCode": result["resultCode"],
                            "resultHash": result["resultHash"],
                        },
                        RESULT_PROJECTION_VERSION,
                    )
                )
            return SimpleNamespace(
                fetchall=lambda: required_facets,
            )

    research = Research()
    monkeypatch.setattr(proof, "_connect", lambda _url: Connection())
    evidence = proof._prove_research_scenario_v4_wiring(
        "postgresql://unit",
        research,
        bundle_revision_id="CBR-LIVE-PROOF",
        solver_timeout_seconds=60.0,
    )
    assert research.executed == ["RUN-PGV4PROOF01"]
    assert research.filters["winning_root_action_band"] == "0"
    assert research.filters["exact_state_available"] == "true"
    assert research.filters["research_backend"] == "ortools_cp_sat_v2"
    assert research.filters["qualification_authority"] == "cp_sat"
    assert research.filters["runtime_compatibility_status"] == "passed"
    assert research.filters["configuration_bundle"] == "CBR-LIVE-PROOF"
    assert research.filters["variant_collection"] == "RUN-PGV4PROOF01"
    assert research.exact_filters == {
        "exact_state_id": "STATE-1234567890ABCDEF1234",
        "scenario_category": "auction",
        "offset": 0,
        "limit": 10,
    }
    assert evidence["exactStateId"] == "STATE-1234567890ABCDEF1234"
    assert evidence["projectionVersion"] == RESULT_PROJECTION_VERSION == 5
    assert evidence["proofOutcomeHash"] == "8" * 64
    assert evidence["proofArtifactSha256"] == "3" * 64
    assert evidence["observerSafeExactState"] is True
    assert evidence["artifactOwnershipVerified"] is True
    assert evidence["bundleAuthorityVerified"] is True
    assert evidence["storedResultIdentityBound"] is True
    assert evidence["exactStateFacetQueryVerified"] is True
    assert evidence["researchBackend"] == "ortools_cp_sat_v2"
    assert evidence["qualificationAuthority"] == "cp_sat"
    assert evidence["runtimeCompatibilityStatus"] == "passed"
    assert evidence["configurationBundle"] == "CBR-LIVE-PROOF"
    assert evidence["scenarioCategories"] == [
        "auction",
        "play",
        "projects",
        "proof",
    ]
    assert evidence["variantCollection"] == "RUN-PGV4PROOF01"
    assert evidence["facetCount"] == len(required_facets)


@pytest.mark.parametrize(
    ("authority_drift", "message"),
    (
        ("row_id", "runtime bundle row identity"),
        ("active_id", "active runtime bundle identity"),
        ("row_version", "runtime bundle row identity"),
        ("active_version", "active runtime bundle identity"),
        ("row_mode", "runtime bundle row identity"),
        ("active_mode", "active runtime bundle identity"),
        ("packaged_status", "packaged bundle profile authority"),
        ("row_status", "runtime bundle lifecycle status"),
        ("active_status", "runtime bundle lifecycle status"),
        ("profiles_family", "profile family authority"),
        ("revisions_family", "profile family authority"),
        ("component_revisions_family", "profile family authority"),
        ("packaged_family", "component family authority"),
        ("active_family", "component family authority"),
        ("active_payload", "active profile payloads"),
        ("revision_payload", "profile revision payloads"),
    ),
)
def test_full_wiring_rejects_configuration_authority_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority_drift: str,
    message: str,
) -> None:
    container, _immutable_calls = _install_full_wiring_fixture(
        monkeypatch, authority_drift=authority_drift
    )

    with pytest.raises(proof.ProofError, match=message):
        proof._prove_full_wiring("postgresql://unit", tmp_path, "RUN-LIVE-ADMIT")

    assert container.shutdown_calls == [True]


def test_full_wiring_rejects_non_postgres_configuration_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container, _calls = _install_full_wiring_fixture(monkeypatch, wrong_store=True)

    with pytest.raises(proof.ProofError, match="PostgreSQL configuration"):
        proof._prove_full_wiring("postgresql://unit", tmp_path, "RUN-LIVE-ADMIT")
    assert container.shutdown_calls == [True]


def test_full_wiring_rejects_unresolved_active_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container, _calls = _install_full_wiring_fixture(monkeypatch, unresolved=True)

    with pytest.raises(proof.ProofError, match="could not resolve"):
        proof._prove_full_wiring("postgresql://unit", tmp_path, "RUN-LIVE-ADMIT")
    assert container.shutdown_calls == [True]


def test_full_wiring_rejects_invalid_configuration_event_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container, _calls = _install_full_wiring_fixture(
        monkeypatch, invalid_event_chain=True
    )

    with pytest.raises(proof.ProofError, match="event chain is invalid"):
        proof._prove_full_wiring("postgresql://unit", tmp_path, "RUN-LIVE-ADMIT")
    assert container.shutdown_calls == [True]


def test_configuration_v25_schema_probe_is_exact_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"dropPolicy": False}

    class Result:
        def __init__(self, *, rows=(), row=None):
            self.rows = list(rows)
            self.row = row

        def fetchall(self):
            return list(self.rows)

        def fetchone(self):
            return self.row

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql: str):
            if "information_schema.columns" in sql:
                return Result(
                    rows=(
                        ("bundle_revisions", "numeric"),
                        ("profile_revisions", "numeric"),
                    )
                )
            if "FROM pg_constraint" in sql:
                return Result(
                    rows=tuple(
                        (*value, True) for value in proof.CONFIGURATION_V25_CONSTRAINTS
                    )
                )
            if "FROM pg_trigger" in sql:
                return Result(rows=proof.CONFIGURATION_V25_TRIGGERS)
            if "FROM pg_policies" in sql:
                rows = [
                    (*value, "PERMISSIVE") for value in proof.CONFIGURATION_V25_POLICIES
                ]
                return Result(rows=rows[1:] if state["dropPolicy"] else rows)
            if "FROM pg_class" in sql:
                return Result(
                    rows=(
                        (relation, True, True)
                        for relation in proof.CONFIGURATION_V25_RLS_RELATIONS
                    )
                )
            if "configuration.invalidation_epochs" in sql:
                return Result(row=(1,))
            raise AssertionError(sql)

    monkeypatch.setattr(proof, "_connect", lambda _url: Connection())

    result = proof._prove_configuration_v25_schema("postgresql://unit")

    assert result == {
        "contractSha256": proof._sha256_text(
            proof._canonical_json(
                {
                    "constraints": [
                        list(value)
                        for value in sorted(proof.CONFIGURATION_V25_CONSTRAINTS)
                    ],
                    "policies": [
                        list(value)
                        for value in sorted(proof.CONFIGURATION_V25_POLICIES)
                    ],
                    "rlsRelations": sorted(proof.CONFIGURATION_V25_RLS_RELATIONS),
                    "triggers": [
                        list(value)
                        for value in sorted(proof.CONFIGURATION_V25_TRIGGERS)
                    ],
                }
            )
        ),
        "rolloutPercentageNumeric": True,
        "validatedConstraintCount": 6,
        "immutabilityTriggerCount": 2,
        "policyCount": 6,
        "forcedRlsRelationCount": 3,
        "invalidationEpochPresent": True,
    }

    state["dropPolicy"] = True
    with pytest.raises(proof.ProofError, match="RLS policies drifted"):
        proof._prove_configuration_v25_schema("postgresql://unit")


def test_failed_child_preflight_stops_before_every_docker_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path / "proof-root")
    Path(context["proofRoot"]).mkdir()
    output = tmp_path / "preflight-failed.json"
    docker_called = False

    monkeypatch.setattr(proof, "_new_context", lambda: context)
    monkeypatch.setattr(proof, "_source_manifest_evidence", _source_preflight)
    monkeypatch.setattr(
        proof,
        "_probe_python_child_dependencies",
        lambda: (_ for _ in ()).throw(
            proof.ProofError("explicit dependency path is missing")
        ),
    )

    def unexpected_docker(*_args, **_kwargs):
        nonlocal docker_called
        docker_called = True
        raise AssertionError("Docker must not run before Python child preflight")

    monkeypatch.setattr(proof, "_local_docker_context", unexpected_docker)
    monkeypatch.setattr(
        proof,
        "_safe_remove_proof_root",
        lambda _path: {"status": "removed", "pathPersisted": False},
    )

    report = proof.run_live_proof(output=output)

    assert report["status"] == "failed"
    assert report["error"]["message"] == "explicit dependency path is missing"
    assert report["environment"] == {"source": _source_preflight()}
    assert docker_called is False
    assert report["cleanup"] == {
        "loopbackRelay": {"status": "not_started"},
        "container": {"status": "not_started"},
        "accessNetwork": {"status": "not_started"},
        "network": {"status": "not_started"},
        "proofRoot": {"status": "removed", "pathPersisted": False},
    }


def test_failed_full_runtime_preflight_stops_before_every_docker_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path / "proof-root")
    Path(context["proofRoot"]).mkdir()
    output = tmp_path / "runtime-preflight-failed.json"
    docker_called = False

    monkeypatch.setattr(proof, "_new_context", lambda: context)
    monkeypatch.setattr(proof, "_source_manifest_evidence", _source_preflight)
    monkeypatch.setattr(proof, "_probe_python_child_dependencies", _child_preflight)
    monkeypatch.setattr(proof, "_dependency_versions", _runtime_versions)
    monkeypatch.setattr(
        proof,
        "_probe_full_runtime_wiring",
        lambda: (_ for _ in ()).throw(
            proof.ProofError("full runtime dependency is missing")
        ),
    )

    def unexpected_docker(*_args, **_kwargs):
        nonlocal docker_called
        docker_called = True
        raise AssertionError("Docker must not run before full runtime preflight")

    monkeypatch.setattr(proof, "_local_docker_context", unexpected_docker)
    monkeypatch.setattr(
        proof,
        "_safe_remove_proof_root",
        lambda _path: {"status": "removed", "pathPersisted": False},
    )

    report = proof.run_live_proof(output=output)

    assert report["status"] == "failed"
    assert report["error"]["message"] == "full runtime dependency is missing"
    assert report["environment"] == {
        "source": _source_preflight(),
        "runtime": _runtime_versions(),
    }
    assert docker_called is False


def test_failed_run_cleans_up_and_persists_no_connection_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path / "proof-root")
    Path(context["proofRoot"]).mkdir()
    output = tmp_path / "failed-proof.json"
    container_id = "d" * 64
    cleanup_calls: list[str] = []

    monkeypatch.setattr(proof, "_new_context", lambda: context)
    monkeypatch.setattr(proof, "_source_manifest_evidence", _source_preflight)
    monkeypatch.setattr(proof, "_probe_python_child_dependencies", _child_preflight)
    monkeypatch.setattr(proof, "_dependency_versions", _runtime_versions)
    monkeypatch.setattr(proof, "_probe_full_runtime_wiring", _runtime_preflight)
    monkeypatch.setattr(
        proof,
        "_local_docker_context",
        lambda: {"name": "default", "endpointScheme": "npipe", "local": True},
    )
    monkeypatch.setattr(
        proof,
        "_docker_server",
        lambda _docker_context: {
            "platform": "Docker Desktop",
            "version": "29",
            "os": "linux",
            "architecture": "amd64",
        },
    )
    monkeypatch.setattr(
        proof,
        "_resolve_image",
        lambda **_kwargs: {"resolvedReference": "postgres@sha256:" + "c" * 64},
    )
    monkeypatch.setattr(
        proof,
        "_assert_container_name_unused",
        lambda _name, _docker_context: None,
    )
    monkeypatch.setattr(
        proof,
        "_assert_network_name_unused",
        lambda _name, _docker_context: None,
    )
    monkeypatch.setattr(
        proof, "_create_network", lambda _context: str(context["networkId"])
    )
    monkeypatch.setattr(
        proof,
        "_create_access_network",
        lambda _context: str(context["accessNetworkId"]),
    )
    monkeypatch.setattr(proof, "_start_container", lambda *_args: container_id)
    monkeypatch.setattr(
        proof,
        "_wait_for_health",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            proof.ProofError(
                f"failed with {context['password']} and postgresql://user:pw@host/db"
            )
        ),
    )
    monkeypatch.setattr(
        proof, "_inspect_container", lambda _reference, _docker_context: None
    )
    monkeypatch.setattr(
        proof,
        "_cleanup_container",
        lambda reference, _context, **_kwargs: cleanup_calls.append(reference)
        or {"status": "removed"},
    )
    monkeypatch.setattr(
        proof, "_cleanup_network", lambda *_args, **_kwargs: {"status": "removed"}
    )
    monkeypatch.setattr(
        proof,
        "_safe_remove_proof_root",
        lambda _path: {"status": "removed", "pathPersisted": False},
    )

    report = proof.run_live_proof(output=output)
    encoded = output.read_text(encoding="utf-8")

    assert report["status"] == "failed"
    assert report["environment"]["source"] == _source_preflight()
    assert report["environment"]["runtime"] == _runtime_versions()
    assert report["environment"]["dockerContext"]["name"] == "default"
    assert report["environment"]["docker"]["os"] == "linux"
    assert report["environment"]["image"] == {}
    assert cleanup_calls == [container_id]
    assert str(context["password"]) not in encoded
    assert "postgresql://" not in encoded
    assert "<redacted>" in encoded


def test_evidence_write_is_exclusive_and_sensitive_keys_fail_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "proof.json"
    report = {"schemaVersion": 1, "kind": proof.PROOF_KIND, "status": "passed"}
    proof._assert_sanitized(report, ["secret"])
    proof._write_evidence(output, report)

    with pytest.raises(proof.ProofError, match="overwrite"):
        proof._write_evidence(output, report)
    with pytest.raises(proof.ProofError, match="sensitive field"):
        proof._assert_sanitized({"databaseUrl": "hidden"}, ["secret"])


@pytest.mark.real_clock
def test_health_wait_is_bounded_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = iter(
        (
            {"Status": "created", "Health": {"Status": "starting"}},
            {"Status": "running", "Health": {"Status": "healthy"}},
        )
    )
    clock = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(
        proof, "_container_state", lambda _identity, _docker_context: next(states)
    )
    monkeypatch.setattr(proof.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(proof.time, "sleep", lambda _seconds: None)
    proof._wait_for_health("d" * 64, docker_context="default", timeout_seconds=30)

    monkeypatch.setattr(
        proof,
        "_container_state",
        lambda _identity, _docker_context: {
            "Status": "running",
            "Health": {"Status": "unhealthy"},
        },
    )
    monkeypatch.setattr(proof.time, "monotonic", lambda: 0.0)
    with pytest.raises(proof.ProofError, match="failed health"):
        proof._wait_for_health("d" * 64, docker_context="default", timeout_seconds=30)


def test_temp_cleanup_rejects_paths_outside_exact_proof_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(proof.tempfile, "gettempdir", lambda: str(tmp_path))
    safe = tmp_path / f"{proof.PROOF_ROOT_PREFIX}safe"
    safe.mkdir()
    read_only = safe / "value.txt"
    read_only.write_text("temporary", encoding="utf-8")
    read_only.chmod(0o444)

    assert proof._safe_remove_proof_root(safe)["status"] == "removed"
    assert not safe.exists()

    outside = tmp_path / "unrelated"
    outside.mkdir()
    with pytest.raises(proof.ProofError, match="unsafe"):
        proof._safe_remove_proof_root(outside)
    assert outside.exists()


def test_docker_inspection_error_is_not_treated_as_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        proof,
        "_run_command",
        lambda *_args, **_kwargs: _completed(
            stderr=b"permission denied while connecting to Docker", returncode=1
        ),
    )

    with pytest.raises(proof.ProofError, match="inspection failed"):
        proof._inspect_container("balootx-pg-proof-test", "default")


def test_name_collision_never_reads_logs_or_attempts_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path / "proof-root")
    Path(context["proofRoot"]).mkdir()
    output = tmp_path / "collision-proof.json"
    logs_called = False

    monkeypatch.setattr(proof, "_new_context", lambda: context)
    monkeypatch.setattr(proof, "_source_manifest_evidence", _source_preflight)
    monkeypatch.setattr(proof, "_probe_python_child_dependencies", _child_preflight)
    monkeypatch.setattr(proof, "_dependency_versions", _runtime_versions)
    monkeypatch.setattr(proof, "_probe_full_runtime_wiring", _runtime_preflight)
    monkeypatch.setattr(
        proof,
        "_local_docker_context",
        lambda: {"name": "default", "endpointScheme": "npipe", "local": True},
    )
    monkeypatch.setattr(
        proof,
        "_docker_server",
        lambda _docker_context: {
            "platform": "Docker Desktop",
            "version": "29",
            "os": "linux",
            "architecture": "amd64",
        },
    )
    monkeypatch.setattr(
        proof,
        "_resolve_image",
        lambda **_kwargs: {"resolvedReference": "postgres@sha256:" + "c" * 64},
    )
    monkeypatch.setattr(
        proof,
        "_assert_container_name_unused",
        lambda _name, _docker_context: (_ for _ in ()).throw(
            proof.ProofError("name in use")
        ),
    )

    def unexpected_logs(*_args, **_kwargs):
        nonlocal logs_called
        logs_called = True
        raise AssertionError("unowned logs must not be read")

    monkeypatch.setattr(proof, "_container_logs", unexpected_logs)
    monkeypatch.setattr(
        proof,
        "_cleanup_container",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pre-existing container must not be cleaned")
        ),
    )
    monkeypatch.setattr(
        proof,
        "_safe_remove_proof_root",
        lambda _path: {"status": "removed", "pathPersisted": False},
    )

    report = proof.run_live_proof(output=output)

    assert report["status"] == "failed"
    assert report["cleanup"]["container"] == {"status": "not_started"}
    assert logs_called is False


@pytest.mark.parametrize(
    ("cleanup_fails", "wiring_fails"),
    ((False, False), (True, False), (False, True)),
)
def test_mocked_success_path_is_downgraded_when_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_fails: bool,
    wiring_fails: bool,
) -> None:
    context = _context(tmp_path / f"proof-root-{cleanup_fails}-{wiring_fails}")
    Path(context["proofRoot"]).mkdir()
    output = tmp_path / f"proof-{cleanup_fails}-{wiring_fails}.json"
    container_id = "d" * 64
    proxy_container_id = "c" * 64
    monkeypatch.setattr(proof, "_new_context", lambda: context)
    monkeypatch.setattr(proof, "_source_manifest_evidence", _source_preflight)
    monkeypatch.setattr(proof, "_probe_python_child_dependencies", _child_preflight)
    monkeypatch.setattr(proof, "_dependency_versions", _runtime_versions)
    monkeypatch.setattr(proof, "_probe_full_runtime_wiring", _runtime_preflight)
    monkeypatch.setattr(
        proof,
        "_local_docker_context",
        lambda: {"name": "default", "endpointScheme": "npipe", "local": True},
    )
    monkeypatch.setattr(
        proof,
        "_docker_server",
        lambda _docker_context: {
            "platform": "Docker Desktop",
            "version": "29",
            "os": "linux",
            "architecture": "amd64",
        },
    )
    monkeypatch.setattr(
        proof,
        "_resolve_image",
        lambda **_kwargs: {
            "indexDigest": "sha256:" + "a" * 64,
            "resolvedDigest": "sha256:" + "c" * 64,
            "resolvedReference": "postgres@sha256:" + "c" * 64,
        },
    )
    monkeypatch.setattr(
        proof,
        "_assert_container_name_unused",
        lambda _name, _docker_context: None,
    )
    monkeypatch.setattr(
        proof,
        "_assert_network_name_unused",
        lambda _name, _docker_context: None,
    )
    monkeypatch.setattr(
        proof, "_create_network", lambda _context: str(context["networkId"])
    )
    monkeypatch.setattr(
        proof,
        "_create_access_network",
        lambda _context: str(context["accessNetworkId"]),
    )
    monkeypatch.setattr(proof, "_start_container", lambda *_args: container_id)
    monkeypatch.setattr(
        proof, "_start_proxy_container", lambda *_args: proxy_container_id
    )
    monkeypatch.setattr(proof, "_wait_for_health", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        proof,
        "_validated_network",
        lambda *_args, **_kwargs: {
            "networkIdSha256": "f" * 64,
            "internal": True,
            "driver": "bridge",
            "attachedContainerCount": 2,
        },
    )
    monkeypatch.setattr(
        proof,
        "_validated_access_network",
        lambda *_args, **_kwargs: {
            "networkIdSha256": "e" * 64,
            "internal": False,
            "driver": "bridge",
            "attachedContainerCount": 1,
        },
    )
    monkeypatch.setattr(
        proof,
        "_validated_container",
        lambda *_args: {"publishedPortCount": 0},
    )
    monkeypatch.setattr(
        proof,
        "_validated_proxy_container",
        lambda *_args: {"hostPort": 49152, "loopbackOnly": True},
    )
    monkeypatch.setattr(
        proof,
        "_database_url",
        lambda *_args: "postgresql://proof-user:proof-secret@127.0.0.1:49152/proof",
    )
    monkeypatch.setattr(proof, "_database_readiness", lambda *_args: {"ready": True})
    monkeypatch.setattr(
        proof,
        "_prove_migrations",
        lambda *_args, **_kwargs: {"schemaFingerprint": {}},
    )
    monkeypatch.setattr(proof, "_prove_research", lambda *_args, **_kwargs: {})

    def full_wiring(*_args):
        if wiring_fails:
            raise proof.ProofError("full wiring failed after resource validation")
        return {}

    monkeypatch.setattr(proof, "_prove_full_wiring", full_wiring)
    monkeypatch.setattr(
        proof, "_wait_for_no_client_sessions", lambda *_args: {"remaining": 0}
    )
    if cleanup_fails:
        monkeypatch.setattr(
            proof,
            "_cleanup_container",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                proof.ProofError("bounded cleanup failed")
            ),
        )
    else:
        monkeypatch.setattr(
            proof,
            "_cleanup_container",
            lambda *_args, **_kwargs: {"status": "removed"},
        )
    monkeypatch.setattr(
        proof, "_cleanup_network", lambda *_args, **_kwargs: {"status": "removed"}
    )
    monkeypatch.setattr(
        proof,
        "_safe_remove_proof_root",
        lambda _path: {"status": "removed", "pathPersisted": False},
    )

    report = proof.run_live_proof(output=output)

    assert report["status"] == ("failed" if cleanup_fails or wiring_fails else "passed")
    assert report["environment"]["source"] == _source_preflight()
    assert report["environment"]["image"]["resolvedDigest"] == "sha256:" + "c" * 64
    assert report["container"] == {"publishedPortCount": 0}
    assert report["loopbackRelay"] == {"hostPort": 49152, "loopbackOnly": True}
    assert report["network"]["internal"] is True
    assert report["accessNetwork"]["internal"] is False
    if wiring_fails:
        assert report["error"]["message"] == (
            "full wiring failed after resource validation"
        )
    assert output.is_file()
    assert "postgresql://" not in output.read_text(encoding="utf-8")


def test_recipe_mutation_probe_covers_replay_conflict_retire_and_cursor() -> None:
    from baloot_platform.research.recipe_api import ResearchMutationConflictError

    receipts: dict[str, dict] = {}
    statuses: dict[str, str] = {
        proof._proof_recipe_code(f"ABC123-{probe}"): "published"
        for probe in ("CLAIM", "FENCE", "RETRY")
    }

    def put_recipe_with_receipt(
        recipe: dict, *, idempotency_key: str, request_intent_hash: str
    ) -> dict:
        if idempotency_key in receipts:
            receipt = receipts[idempotency_key]
            if receipt["requestIntentHash"] != request_intent_hash:
                raise ResearchMutationConflictError("different intent")
            return dict(receipt["response"])
        response = {**recipe, "lifecycleStatus": "published"}
        statuses[recipe["code"]] = "published"
        receipts[idempotency_key] = {
            "mutationKind": "create",
            "requestIntentHash": request_intent_hash,
            "responseStatus": 201,
            "response": response,
        }
        return response

    def retire_recipe_with_receipt(
        code: str, *, idempotency_key: str, request_intent_hash: str
    ) -> dict:
        if idempotency_key in receipts:
            receipt = receipts[idempotency_key]
            if receipt["requestIntentHash"] != request_intent_hash:
                raise ResearchMutationConflictError("different intent")
            return dict(receipt["response"])
        statuses[code] = "retired"
        response = {
            "code": code,
            "sha256": proof._hash("recipe-mutation:ABC123-MUTATION"),
            "status": "published",
            "lifecycleStatus": "retired",
        }
        receipts[idempotency_key] = {
            "mutationKind": "retire",
            "requestIntentHash": request_intent_hash,
            "responseStatus": 200,
            "response": response,
        }
        return response

    def ordered_codes(status: str | None) -> list[str]:
        return sorted(
            code
            for code, lifecycle_status in statuses.items()
            if status is None or lifecycle_status == status
        )

    def list_recipes(*, status: str | None = None, limit: int = 100) -> list[dict]:
        return [{"code": code} for code in ordered_codes(status)[:limit]]

    def list_recipe_page(**values) -> dict:
        codes = ordered_codes(values.get("status"))
        after_recipe_code = values.get("after_recipe_code")
        start = codes.index(after_recipe_code) + 1 if after_recipe_code else 0
        limit = int(values.get("limit", 100))
        selected = codes[start : start + limit]
        has_more = start + limit < len(codes)
        return {
            "items": [{"recipe": {"code": code}} for code in selected],
            "nextAnchor": (
                {
                    "createdAt": "2026-08-09T00:00:00+00:00",
                    "recipeCode": selected[-1],
                }
                if has_more
                else None
            ),
        }

    store = SimpleNamespace(
        put_recipe_with_receipt=put_recipe_with_receipt,
        get_recipe_mutation_receipt=lambda key: receipts.get(key),
        list_recipes=list_recipes,
        list_recipe_page=list_recipe_page,
        retire_recipe_with_receipt=retire_recipe_with_receipt,
        recipe_status=lambda code: statuses.get(code),
    )

    result = proof._prove_recipe_mutations(
        store,
        suffix="ABC123-MUTATION",
        published_recipe_labels=(
            "ABC123-CLAIM",
            "ABC123-FENCE",
            "ABC123-RETRY",
        ),
    )

    assert result == {
        "createReceiptReplayed": True,
        "conflictingIntentRejected": True,
        "retireReceiptReplayed": True,
        "retireConflictingIntentRejected": True,
        "invalidCursorRejected": True,
        "stableKeysetPagination": True,
        "lifecycleListingsVerified": True,
    }


@pytest.mark.real_clock
def test_research_barrier_failure_names_stage_parties_and_timeout() -> None:
    barrier = threading.Barrier(2)

    with pytest.raises(proof.ProofError) as captured:
        proof._wait_at_research_barrier(
            barrier,
            stage="research.claim_concurrency.drain",
            timeout_seconds=0.001,
        )

    message = str(captured.value)
    assert "stage=research.claim_concurrency.drain" in message
    assert "parties=2" in message
    assert "waiting=0" in message
    assert "broken=True" in message
    assert "timeoutSeconds=0.001" in message
    assert isinstance(captured.value.__cause__, threading.BrokenBarrierError)
    assert proof.RESEARCH_BARRIER_TIMEOUT_SECONDS == 60.0


class _ConcurrentClaimStore:
    def __init__(self, *, failing_worker: str | None = None) -> None:
        self._lock = threading.Lock()
        self._shards: list[dict[str, object]] = []
        self.claim_workers: set[str] = set()
        self.failing_worker = failing_worker
        self.failure_raised = False

    def seed_shards(self, count: int) -> list[dict[str, object]]:
        with self._lock:
            self._shards = [
                {
                    "shardId": f"RSH-{index:03d}",
                    "seedCount": 1,
                    "status": "queued",
                    "attemptCount": 0,
                    "fencingToken": 0,
                    "workerId": None,
                }
                for index in range(count)
            ]
            return [dict(item) for item in self._shards]

    def claim_shard(
        self, _run_code: str, *, worker_id: str, lease_seconds: int
    ) -> dict[str, object] | None:
        assert lease_seconds == 30
        with self._lock:
            self.claim_workers.add(worker_id)
            if worker_id == self.failing_worker and not self.failure_raised:
                self.failure_raised = True
                raise RuntimeError("adversarial claim failure")
            for shard in self._shards:
                if shard["status"] == "queued":
                    shard["status"] = "running"
                    shard["attemptCount"] = 1
                    shard["fencingToken"] = 1
                    shard["workerId"] = worker_id
                    return dict(shard)
        return None

    def complete_shard(
        self,
        shard_id: str,
        *,
        worker_id: str,
        fencing_token: int,
        result: dict[str, object],
    ) -> dict[str, object]:
        assert fencing_token == 1
        assert result["attemptedSeeds"] == 1
        with self._lock:
            shard = next(item for item in self._shards if item["shardId"] == shard_id)
            assert shard["workerId"] == worker_id
            assert shard["status"] == "running"
            shard["status"] = "completed"
            return dict(shard)

    def list_shards(self, _run_code: str) -> list[dict[str, object]]:
        with self._lock:
            return [dict(item) for item in self._shards]


def _install_claim_probe_fixtures(
    monkeypatch: pytest.MonkeyPatch, store: _ConcurrentClaimStore
) -> None:
    monkeypatch.setattr(
        proof,
        "_put_recipe_and_run",
        lambda *_args, **_kwargs: ({"sha256": "a" * 64}, {"runCode": "RUN-PROOF"}),
    )
    monkeypatch.setattr(
        proof,
        "_create_shards",
        lambda *_args, count, **_kwargs: store.seed_shards(count),
    )


def test_claim_concurrency_shares_existing_pool_and_preserves_eight_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ConcurrentClaimStore()
    _install_claim_probe_fixtures(monkeypatch, store)
    delayed_initialization_entered = threading.Event()

    def delayed_store_initialization(_database_url: str) -> None:
        delayed_initialization_entered.set()
        raise AssertionError("claim concurrency created an avoidable store pool")

    monkeypatch.setattr(proof, "_store", delayed_store_initialization)

    result = proof._prove_claim_concurrency(
        "postgresql://unit", store, suffix="ABC123-CLAIM"
    )

    expected_workers = {f"pg-proof-worker-{index}" for index in range(8)}
    assert delayed_initialization_entered.is_set() is False
    assert store.claim_workers == expected_workers
    assert result == {
        "workers": 8,
        "workerIds": sorted(expected_workers),
        "shards": 64,
        "uniqueClaims": 64,
        "allCompleted": True,
    }


@pytest.mark.real_clock
def test_claim_concurrency_waits_for_executor_cleanup_on_worker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ConcurrentClaimStore(failing_worker="pg-proof-worker-0")
    _install_claim_probe_fixtures(monkeypatch, store)
    real_executor = proof.concurrent.futures.ThreadPoolExecutor
    executors: list[object] = []

    class RecordingExecutor(real_executor):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.shutdown_waits: list[bool] = []
            executors.append(self)

        def shutdown(self, wait: bool = True, **kwargs) -> None:
            self.shutdown_waits.append(wait)
            super().shutdown(wait=wait, **kwargs)

    monkeypatch.setattr(
        proof.concurrent.futures, "ThreadPoolExecutor", RecordingExecutor
    )

    with pytest.raises(RuntimeError, match="adversarial claim failure"):
        proof._prove_claim_concurrency(
            "postgresql://unit", store, suffix="ABC123-CLAIM"
        )

    assert len(executors) == 1
    assert executors[0].shutdown_waits == [True]
    assert store.failure_raised is True


def test_research_proof_stage_inventory_owns_every_emitted_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, dict[str, object]] = {}

    class Store:
        closed = False

        def close(self) -> None:
            self.closed = True

    store = Store()
    monkeypatch.setattr(proof, "_store", lambda _database_url: store)
    monkeypatch.setattr(
        proof,
        "_bootstrap_research",
        lambda *_args, **_kwargs: {"migrationEquivalent": True},
    )

    def stage(name: str):
        def invoke(*_args, suffix: str, **kwargs) -> dict[str, object]:
            calls[name] = {"suffix": suffix, **kwargs}
            return {"stage": name}

        return invoke

    for stage_name, function_name in (
        ("concurrency", "_prove_claim_concurrency"),
        ("fencing", "_prove_fencing"),
        ("retry", "_prove_retry"),
        ("recipeMutations", "_prove_recipe_mutations"),
        ("admissionAndStatistics", "_prove_admission_and_statistics"),
        ("fencedSearchReceipts", "_prove_fenced_search_receipts"),
        ("runLockSerialization", "_prove_run_lock_serialization"),
        ("artifactSerialization", "_prove_artifact_serialization"),
    ):
        monkeypatch.setattr(proof, function_name, stage(stage_name))

    result = proof._prove_research(
        "postgresql://unit",
        "abcdef0123456789abcdef0123456789",
        expected_schema_fingerprint={"sha256": "a" * 64},
    )
    expected_labels = proof._research_proof_stage_labels("ABCDEF0123")

    assert result["stageInventory"] == [
        {"stage": stage_name, "label": expected_labels[stage_name]}
        for stage_name in proof.RESEARCH_PROOF_STAGE_KEYS
    ]
    assert tuple(calls) == proof.RESEARCH_PROOF_STAGE_KEYS
    assert {
        stage_name: values["suffix"] for stage_name, values in calls.items()
    } == expected_labels
    assert calls["recipeMutations"]["published_recipe_labels"] == tuple(
        expected_labels[stage_name] for stage_name in proof.PUBLISHED_RECIPE_STAGE_KEYS
    )
    assert store.closed is True

    with pytest.raises(proof.ProofError, match="suffix is invalid"):
        proof._research_proof_stage_labels("ABC123")


def test_research_proof_closes_shared_store_when_a_stage_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Store:
        closed = False

        def close(self) -> None:
            self.closed = True

    store = Store()
    monkeypatch.setattr(proof, "_store", lambda _database_url: store)
    monkeypatch.setattr(
        proof,
        "_bootstrap_research",
        lambda *_args, **_kwargs: {"migrationEquivalent": True},
    )
    monkeypatch.setattr(
        proof,
        "_prove_claim_concurrency",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stage failed")),
    )

    with pytest.raises(RuntimeError, match="stage failed"):
        proof._prove_research(
            "postgresql://unit",
            "abcdef0123456789abcdef0123456789",
            expected_schema_fingerprint={"sha256": "a" * 64},
        )

    assert store.closed is True


def test_live_receipt_fixture_is_runner_owned_and_bound_to_frozen_contract() -> None:
    from baloot_platform.research.sequential_candidate_search import (
        SequentialCandidateSearchReceiptV1,
    )

    document = proof._empty_sequential_search_receipt(7)
    parsed = SequentialCandidateSearchReceiptV1.from_dict(document).verify()

    assert parsed.to_dict() == document
    assert document["rootSeed"] == 7
    assert document["stopReason"] == "candidates_exhausted"
    assert document["generatorDescriptor"]["version"] == "1.5.0"
    assert "structuralDomain" in document["searchIdentityDocument"]


def test_receipt_admission_assertion_preserves_identity_and_checks_transition() -> None:
    immutable = {
        "schemaVersion": 1,
        "occurrenceHash": "a" * 64,
        "receiptHash": "b" * 64,
        "runCode": "RUN-PROOF",
        "shardId": "RSH-PROOF",
        "attemptCount": 3,
        "fencingToken": 3,
        "seed": 7,
        "createdAt": "2026-08-09T00:00:00+00:00",
        "receiptBundle": {"sha256": "b" * 64},
    }
    staged = {**immutable, "admitted": False}
    admitted = {**immutable, "admitted": True}

    proof._assert_receipt_admission_transition(staged, admitted)

    with pytest.raises(proof.ProofError, match="rewrote immutable"):
        proof._assert_receipt_admission_transition(
            staged, {**admitted, "occurrenceHash": "c" * 64}
        )
    with pytest.raises(proof.ProofError, match="transition is invalid"):
        proof._assert_receipt_admission_transition(staged, staged)


def _valid_receipt_snapshot_assertion_inputs() -> dict[str, object]:
    return {
        "snapshot_batch_size": 1,
        "fixed_snapshot_seeds": [1, 2],
        "concurrent_admitted_seed": 3,
        "fresh_snapshot_seeds": [1, 2, 3],
        "writer_completed_while_snapshot_open": True,
        "final_occurrence_count": 5,
        "final_admitted_count": 3,
    }


def test_receipt_snapshot_assertion_emits_exact_live_evidence() -> None:
    assert proof._assert_receipt_snapshot_choreography(
        **_valid_receipt_snapshot_assertion_inputs()
    ) == {
        "snapshotBatchSize": 1,
        "fixedSnapshotSeeds": [1, 2],
        "concurrentAdmittedSeed": 3,
        "freshSnapshotSeeds": [1, 2, 3],
        "writerCompletedWhileSnapshotOpen": True,
        "snapshotExcludedConcurrentAdmission": True,
        "finalOccurrenceCount": 5,
        "finalAdmittedCount": 3,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"fixed_snapshot_seeds": [1, 2, 3]}, "leaked concurrent admission"),
        ({"fresh_snapshot_seeds": [1, 2]}, "omitted concurrent admission"),
        ({"fixed_snapshot_seeds": [2, 1]}, "duplicate or order drift"),
        ({"fresh_snapshot_seeds": [1, 1, 3]}, "duplicate or order drift"),
        (
            {"writer_completed_while_snapshot_open": False},
            "did not complete while snapshot was open",
        ),
        ({"final_occurrence_count": 4}, "occurrence count drifted"),
        ({"final_admitted_count": 2}, "admitted receipt count drifted"),
    ),
)
def test_receipt_snapshot_assertion_rejects_adversarial_evidence(
    overrides: dict[str, object], message: str
) -> None:
    values = _valid_receipt_snapshot_assertion_inputs()
    values.update(overrides)
    with pytest.raises(proof.ProofError, match=message):
        proof._assert_receipt_snapshot_choreography(**values)
