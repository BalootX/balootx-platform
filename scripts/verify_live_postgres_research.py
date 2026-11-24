#!/usr/bin/env python3
"""Run one isolated, live PostgreSQL proof for the Research control plane.

The proof intentionally uses a standalone, uniquely labelled container. It
does not use the repository Compose stack, host volumes, a fixed host port, or
an ambient database URL. The resolved image digest, schema fingerprints, and
normalized outcomes are retained; credentials and connection strings are not.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGED_MIGRATIONS = SRC / "baloot_platform" / "resources" / "migrations"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
POSTGRES_IMAGE_TAG = "postgres:17-alpine"
EXPECTED_POSTGRES_INDEX_DIGEST = (
    "sha256:18cfe3ef5e6815560c98237d6216d1e5119702fb0f3894c8785dd58b8bbe5d73"
)
PROOF_KIND = "balootx_live_postgres_research_proof_v1"
CONTAINER_LABEL_KIND = "com.balootx.validation"
CONTAINER_LABEL_ID = "com.balootx.validation-id"
CONTAINER_LABEL_ROLE = "com.balootx.validation-role"
CONTAINER_LABEL_KIND_VALUE = "research-postgres"
DATABASE_CONTAINER_ROLE = "database"
PROXY_CONTAINER_ROLE = "loopback-relay"
INTERNAL_NETWORK_ROLE = "database-internal"
ACCESS_NETWORK_ROLE = "loopback-access"
PROOF_ROOT_PREFIX = "BalootX-pg-proof-"
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
PORT_RE = re.compile(r"^127\.0\.0\.1:([0-9]{1,5})$")
DSN_RE = re.compile(r"postgres(?:ql)?://[^\s\"']+", re.IGNORECASE)
DOCKER_CONTEXT_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
MIGRATION_NAME_RE = re.compile(r"^[0-9]{3}_[a-z0-9_]+\.sql$")
REQUIRED_MIGRATION_HASHES = {
    "024_research_fenced_search_receipts.sql": (
        "913a41c16aaa695705e7f33b4fe579396eaef6c04280a538c34470adae84e33b"
    ),
    "025_configuration_store_contract.sql": (
        "a72345cb2593df997f8f4fb839450a8f34f5672c9ed1f51fdc3f1f5aa108a9fe"
    ),
    "026_research_scenario_v4_projections.sql": (
        "65ea8bb88ac5b22de45d9917b65d2b64203422d942b8e018e2d4208058fba3db"
    ),
    "027_research_result_library_projection.sql": (
        "462ff77337f9f041713130d907e74ff3442916517cf395751950ab44805464c4"
    ),
    "028_research_authority_projection.sql": (
        "ce643a64a4094849cca458bb11f3c4287476e0d32727995449f64919ff676763"
    ),
    "029_shared_run_game_access.sql": (
        "4612ba64ff50b969fc978a9671267aca8afbd12d817de0d586b5fab4edab8b5b"
    ),
    "030_oidc_issuer_transition.sql": (
        "5cba5166395de08d74ec5ba4dcd78eaee11756ac06f9a58acb14f7481e1ffa5e"
    ),
    "031_oidc_registry_access.sql": (
        "79f0419b6e14eff0320087847edf5688745041507c3b63df8f8e5254cc8871da"
    ),
    "032_oidc_identity_authority.sql": (
        "4bd244d098d27a3cf86097c50db52052f7cd4e1cd0375bb623a444d812b20f10"
    ),
}
CONFIGURATION_V25_CONSTRAINTS = frozenset(
    {
        ("bundle_revisions", "configuration_bundle_dimensions_v25"),
        ("bundle_revisions", "configuration_bundle_rollout_v25"),
        ("bundle_revisions", "configuration_bundle_scope_v25"),
        ("profile_revisions", "configuration_profile_dimensions_v25"),
        ("profile_revisions", "configuration_profile_rollout_v25"),
        ("profile_revisions", "configuration_profile_scope_v25"),
    }
)
CONFIGURATION_V25_TRIGGERS = frozenset(
    {
        (
            "bundle_revisions",
            "trg_configuration_bundle_immutable",
            "guard_published_bundle_payload",
            "O",
        ),
        (
            "profile_revisions",
            "trg_configuration_profile_immutable",
            "guard_published_profile_payload",
            "O",
        ),
    }
)
CONFIGURATION_V25_POLICIES = frozenset(
    {
        ("bundle_revisions", "configuration_bundle_context_read_v25", "SELECT"),
        ("bundle_revisions", "configuration_bundle_write_v25", "ALL"),
        ("events", "configuration_events_read_v25", "SELECT"),
        ("events", "configuration_events_write_v25", "INSERT"),
        ("profile_revisions", "configuration_profile_context_read_v25", "SELECT"),
        ("profile_revisions", "configuration_profile_write_v25", "ALL"),
    }
)
CONFIGURATION_V25_RLS_RELATIONS = frozenset(
    {"bundle_revisions", "events", "profile_revisions"}
)
REQUIRED_TMPFS = {
    "/var/lib/postgresql/data": "rw,nosuid,nodev,noexec,size=1073741824",
    "/var/run/postgresql": "rw,nosuid,nodev,noexec,size=16777216",
    "/tmp": "rw,nosuid,nodev,noexec,size=67108864",
}
PROXY_TMPFS = {
    "/var/lib/postgresql/data": "rw,nosuid,nodev,noexec,size=16777216",
    "/tmp": "rw,nosuid,nodev,noexec,size=16777216",
}
RESEARCH_PROOF_STAGE_SUFFIXES = (
    ("concurrency", "CLAIM"),
    ("fencing", "FENCE"),
    ("retry", "RETRY"),
    ("recipeMutations", "MUTATION"),
    ("admissionAndStatistics", "ADMIT"),
    ("fencedSearchReceipts", "RECEIPT"),
    ("runLockSerialization", "RUNLOCK"),
    ("artifactSerialization", "ARTIFACT"),
)
RESEARCH_PROOF_STAGE_KEYS = tuple(
    stage for stage, _suffix in RESEARCH_PROOF_STAGE_SUFFIXES
)
PUBLISHED_RECIPE_STAGE_KEYS = ("concurrency", "fencing", "retry")
RECEIPT_OCCURRENCE_IMMUTABLE_FIELDS = (
    "schemaVersion",
    "occurrenceHash",
    "receiptHash",
    "runCode",
    "shardId",
    "attemptCount",
    "fencingToken",
    "seed",
    "createdAt",
    "receiptBundle",
)
EXPECTED_RESEARCH_TABLES = frozenset(
    {
        "audit_traces",
        "comparison_campaigns",
        "domain_compilation_artifacts",
        "recipe_mutation_receipts",
        "recipes",
        "result_occurrences",
        "result_facets",
        "results",
        "runs",
        "semantic_disagreement_occurrences",
        "semantic_disagreements",
        "sequential_search_receipt_bundles",
        "sequential_search_receipt_occurrences",
        "shards",
    }
)
MIGRATION_RELATIONS = (
    "configuration.profile_revisions",
    "configuration.bundle_revisions",
    "foundry_campaigns_v1",
    "foundry_shards_v1",
    "foundry_audit_traces_v1",
    "research.domain_compilation_artifacts",
    "research.audit_traces",
    "research.recipes",
    "research.runs",
    "research.recipe_mutation_receipts",
    "research.results",
    "research.result_facets",
    "research.result_occurrences",
    "research.semantic_disagreements",
    "research.semantic_disagreement_occurrences",
    "research.shards",
    "research.sequential_search_receipt_bundles",
    "research.sequential_search_receipt_occurrences",
)


class ProofError(RuntimeError):
    """A fail-closed validation error safe to summarize after redaction."""


RESEARCH_BARRIER_TIMEOUT_SECONDS = 60.0


def _wait_at_research_barrier(
    barrier: threading.Barrier,
    *,
    stage: str,
    timeout_seconds: float = RESEARCH_BARRIER_TIMEOUT_SECONDS,
) -> None:
    """Fail closed with enough context to identify a research barrier site."""
    try:
        barrier.wait(timeout=timeout_seconds)
    except threading.BrokenBarrierError as exc:
        raise ProofError(
            "PostgreSQL research barrier failed "
            f"(stage={stage}, parties={barrier.parties}, "
            f"waiting={barrier.n_waiting}, broken={barrier.broken}, "
            f"timeoutSeconds={timeout_seconds:g})"
        ) from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_manifest_evidence(root: Path = ROOT) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    manifest_path = root / "release" / "source-manifest.json"
    try:
        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(raw_manifest.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProofError("source manifest is unavailable or invalid") from exc
    expected_keys = {
        "schemaVersion",
        "project",
        "version",
        "archivePrefix",
        "source",
        "fileCount",
        "files",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_keys:
        raise ProofError("source manifest contract drifted")
    files = manifest["files"]
    file_count = manifest["fileCount"]
    if (
        manifest["schemaVersion"] != 1
        or manifest["project"] != "baloot-platform"
        or not isinstance(manifest["version"], str)
        or not manifest["version"]
        or manifest["archivePrefix"] != f"baloot-platform-{manifest['version']}"
        or not isinstance(manifest["source"], Mapping)
        or type(file_count) is not int
        or file_count < 1
        or not isinstance(files, Mapping)
        or len(files) != file_count
    ):
        raise ProofError("source manifest identity is invalid")
    for relative, metadata in files.items():
        posix_path = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            posix_path is None
            or not relative
            or "\\" in relative
            or posix_path.is_absolute()
            or posix_path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in posix_path.parts)
            or not isinstance(metadata, Mapping)
            or set(metadata) != {"bytes", "sha256"}
            or type(metadata["bytes"]) is not int
            or metadata["bytes"] < 0
            or not isinstance(metadata["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"]) is None
        ):
            raise ProofError("source manifest file entry is invalid")
        candidate = root.joinpath(*posix_path.parts)
        try:
            resolved_candidate = candidate.resolve(strict=True)
        except OSError as exc:
            raise ProofError(f"source manifest content is missing: {relative}") from exc
        if (
            candidate.is_symlink()
            or not resolved_candidate.is_file()
            or root not in resolved_candidate.parents
            or resolved_candidate.stat().st_size != metadata["bytes"]
            or _sha256_file(resolved_candidate) != metadata["sha256"]
        ):
            raise ProofError(f"source manifest content mismatch: {relative}")
    return {
        "sourceManifestSha256": _sha256_bytes(raw_manifest),
        "contentInventorySha256": _sha256_text(_canonical_json(files)),
        "verifiedFileCount": file_count,
        "project": str(manifest["project"]),
        "version": str(manifest["version"]),
        "archivePrefix": str(manifest["archivePrefix"]),
        "source": dict(manifest["source"]),
        "contentVerified": True,
    }


def _hash(label: str) -> str:
    return _sha256_text(label)


def _proof_recipe_code(label: str) -> str:
    return f"RCP-{_hash(f'recipe-code:{label}')[:16].upper()}"


def _research_proof_stage_labels(root_suffix: str) -> dict[str, str]:
    if re.fullmatch(r"[0-9A-F]{10}", root_suffix) is None:
        raise ProofError("PostgreSQL research proof suffix is invalid")
    return {
        stage: f"{root_suffix}-{stage_suffix}"
        for stage, stage_suffix in RESEARCH_PROOF_STAGE_SUFFIXES
    }


def _assert_receipt_admission_transition(
    staged: Mapping[str, Any], admitted: Mapping[str, Any]
) -> None:
    staged_keys = set(staged)
    admitted_keys = set(admitted)
    expected_keys = {*RECEIPT_OCCURRENCE_IMMUTABLE_FIELDS, "admitted"}
    if staged_keys != expected_keys or admitted_keys != expected_keys:
        raise ProofError("PostgreSQL receipt occurrence projection drifted")
    if any(
        staged[field] != admitted[field]
        for field in RECEIPT_OCCURRENCE_IMMUTABLE_FIELDS
    ):
        raise ProofError(
            "PostgreSQL receipt admission rewrote immutable occurrence data"
        )
    if staged["admitted"] is not False or admitted["admitted"] is not True:
        raise ProofError("PostgreSQL receipt admission transition is invalid")


def _assert_receipt_snapshot_choreography(
    *,
    snapshot_batch_size: int,
    fixed_snapshot_seeds: Sequence[int],
    concurrent_admitted_seed: int,
    fresh_snapshot_seeds: Sequence[int],
    writer_completed_while_snapshot_open: bool,
    final_occurrence_count: int,
    final_admitted_count: int,
) -> dict[str, Any]:
    """Validate and project the bounded live receipt-snapshot proof."""

    fixed = list(fixed_snapshot_seeds)
    fresh = list(fresh_snapshot_seeds)
    for label, seeds in (("fixed", fixed), ("fresh", fresh)):
        if any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ProofError(f"PostgreSQL {label} receipt snapshot seed is invalid")
        if seeds != sorted(set(seeds)):
            raise ProofError(
                f"PostgreSQL {label} receipt snapshot has duplicate or order drift"
            )
    if type(snapshot_batch_size) is not int or snapshot_batch_size != 1:
        raise ProofError("PostgreSQL receipt snapshot batch size drifted")
    if type(concurrent_admitted_seed) is not int or concurrent_admitted_seed != 3:
        raise ProofError("PostgreSQL concurrent receipt admission seed drifted")
    if writer_completed_while_snapshot_open is not True:
        raise ProofError(
            "PostgreSQL concurrent receipt writer did not complete while snapshot was open"
        )
    if concurrent_admitted_seed in fixed:
        raise ProofError("PostgreSQL old receipt snapshot leaked concurrent admission")
    if fixed != [1, 2]:
        raise ProofError("PostgreSQL fixed receipt snapshot seeds drifted")
    if concurrent_admitted_seed not in fresh:
        raise ProofError(
            "PostgreSQL fresh receipt snapshot omitted concurrent admission"
        )
    if fresh != [1, 2, 3]:
        raise ProofError("PostgreSQL fresh receipt snapshot seeds drifted")
    if type(final_occurrence_count) is not int or final_occurrence_count != 5:
        raise ProofError("PostgreSQL final receipt occurrence count drifted")
    if type(final_admitted_count) is not int or final_admitted_count != 3:
        raise ProofError("PostgreSQL final admitted receipt count drifted")
    return {
        "snapshotBatchSize": snapshot_batch_size,
        "fixedSnapshotSeeds": fixed,
        "concurrentAdmittedSeed": concurrent_admitted_seed,
        "freshSnapshotSeeds": fresh,
        "writerCompletedWhileSnapshotOpen": True,
        "snapshotExcludedConcurrentAdmission": True,
        "finalOccurrenceCount": final_occurrence_count,
        "finalAdmittedCount": final_admitted_count,
    }


def _redact(value: str, secrets_to_remove: Sequence[str]) -> str:
    result = str(value)
    for secret in sorted(
        {item for item in secrets_to_remove if item}, key=len, reverse=True
    ):
        result = result.replace(secret, "<redacted>")
    return DSN_RE.sub("<redacted-dsn>", result)


def _run_command(
    command: Sequence[str],
    *,
    timeout: float,
    environment: Mapping[str, str] | None = None,
    check: bool = True,
    secrets_to_remove: Sequence[str] = (),
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            [str(item) for item in command],
            cwd=ROOT,
            env=None if environment is None else dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(timeout),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProofError(
            _redact(
                f"command failed to execute: {command[0]}: {type(exc).__name__}: {exc}",
                secrets_to_remove,
            )
        ) from exc
    if check and completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise ProofError(
            _redact(
                f"command failed ({completed.returncode}): {command[0]}: {stderr}",
                secrets_to_remove,
            )
        )
    return completed


def _command_text(completed: subprocess.CompletedProcess[bytes]) -> str:
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _python_child_environment(
    extra: Mapping[str, str] | None = None,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the one fail-closed environment used by every Python child."""
    environment = {
        str(key): str(value)
        for key, value in (
            os.environ if base_environment is None else base_environment
        ).items()
    }
    source = SRC.resolve()
    raw_entries = [str(source)]
    raw_entries.extend(
        entry for entry in environment.get("PYTHONPATH", "").split(os.pathsep) if entry
    )
    entries: list[str] = []
    identities: set[str] = set()
    for raw_entry in raw_entries:
        path = Path(raw_entry).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        resolved = path.resolve()
        identity = os.path.normcase(str(resolved))
        if identity in identities:
            continue
        identities.add(identity)
        entries.append(str(resolved))
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra:
        protected = {"PYTHONPATH", "PYTHONNOUSERSITE"}.intersection(extra)
        if protected:
            raise ProofError(
                "Python child environment cannot override import isolation"
            )
        environment.update({str(key): str(value) for key, value in extra.items()})
    return environment


def _run_python_child(
    arguments: Sequence[str],
    *,
    timeout: float,
    extra_environment: Mapping[str, str] | None = None,
    base_environment: Mapping[str, str] | None = None,
    secrets_to_remove: Sequence[str] = (),
) -> tuple[subprocess.CompletedProcess[bytes], dict[str, str]]:
    environment = _python_child_environment(
        extra_environment,
        base_environment=base_environment,
    )
    completed = _run_command(
        [sys.executable, "-B", "-s", *map(str, arguments)],
        timeout=timeout,
        environment=environment,
        secrets_to_remove=secrets_to_remove,
    )
    return completed, environment


def _probe_python_child_dependencies(
    *, base_environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    probe_source = "\n".join(
        (
            "import json, pathlib, sys",
            "import baloot_platform, jsonschema, psycopg, psycopg_pool",
            "print(json.dumps({",
            "'python': sys.version.split()[0],",
            "'noUserSite': bool(sys.flags.no_user_site),",
            "'balootPlatformPath': str(pathlib.Path(baloot_platform.__file__).resolve()),",
            "'jsonschemaPath': str(pathlib.Path(jsonschema.__file__).resolve()),",
            "'psycopg': str(psycopg.__version__),",
            "'psycopgPath': str(pathlib.Path(psycopg.__file__).resolve()),",
            "'psycopgPool': str(psycopg_pool.__version__),",
            "'psycopgPoolPath': str(pathlib.Path(psycopg_pool.__file__).resolve()),",
            "}, sort_keys=True))",
        )
    )
    try:
        completed, environment = _run_python_child(
            ["-c", probe_source],
            timeout=30,
            base_environment=base_environment,
        )
    except ProofError as exc:
        raise ProofError(f"Python child dependency preflight failed: {exc}") from exc
    try:
        payload = json.loads(_command_text(completed))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProofError("Python child dependency preflight output is invalid") from exc
    expected_keys = {
        "python",
        "noUserSite",
        "balootPlatformPath",
        "jsonschemaPath",
        "psycopg",
        "psycopgPath",
        "psycopgPool",
        "psycopgPoolPath",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ProofError("Python child dependency preflight output is incomplete")
    if payload["noUserSite"] is not True:
        raise ProofError("Python child user-site isolation is disabled")
    expected_source = (SRC / "baloot_platform" / "__init__.py").resolve()
    try:
        source_origin = Path(str(payload["balootPlatformPath"])).resolve(strict=True)
    except OSError as exc:
        raise ProofError("Python child Baloot source provenance is invalid") from exc
    if os.path.normcase(str(source_origin)) != os.path.normcase(str(expected_source)):
        raise ProofError("Python child Baloot source was substituted")

    source_identity = os.path.normcase(str(SRC.resolve()))
    dependency_roots = [
        Path(entry).resolve()
        for entry in environment["PYTHONPATH"].split(os.pathsep)
        if os.path.normcase(str(Path(entry).resolve())) != source_identity
    ]
    for package, field in (
        ("jsonschema", "jsonschemaPath"),
        ("psycopg", "psycopgPath"),
        ("psycopg_pool", "psycopgPoolPath"),
    ):
        try:
            origin = Path(str(payload[field])).resolve(strict=True)
        except OSError as exc:
            raise ProofError(f"Python child {package} provenance is invalid") from exc
        if not any(
            (root / package).resolve() in origin.parents for root in dependency_roots
        ):
            raise ProofError(
                f"Python child {package} did not load from an explicit dependency root"
            )
        payload[f"{package}ModuleSha256"] = _sha256_file(origin)
    for field in ("python", "psycopg", "psycopgPool"):
        if not isinstance(payload[field], str) or not str(payload[field]).strip():
            raise ProofError("Python child dependency version is invalid")
    return {
        "python": str(payload["python"]),
        "psycopg": str(payload["psycopg"]),
        "psycopgPool": str(payload["psycopgPool"]),
        "pythonNoUserSite": True,
        "balootSourceVerified": True,
        "explicitDependencyProvenanceVerified": True,
        "explicitDependencyRootCount": len(dependency_roots),
        "explicitDependencyModuleCount": 3,
        "jsonschemaModuleSha256": str(payload["jsonschemaModuleSha256"]),
        "psycopgModuleSha256": str(payload["psycopgModuleSha256"]),
        "psycopgPoolModuleSha256": str(payload["psycopg_poolModuleSha256"]),
    }


def _probe_full_runtime_wiring() -> dict[str, Any]:
    try:
        from baloot_platform.config import build_bundle
        from baloot_platform.configuration import PostgresConfigurationStore
        from baloot_platform.research.postgres_store import PostgresResearchStoreV1
        from baloot_platform.service.container import build_container
        from baloot_platform.storage.request_context import DatabaseIdentityContext
        from baloot_platform.version import __version__

        bundle = build_bundle(ROOT)
    except ModuleNotFoundError as exc:
        module = str(exc.name or "unknown")
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", module) is None:
            module = "unknown"
        raise ProofError(
            f"full runtime dependency and bundle preflight failed: missing module {module}"
        ) from exc
    except Exception as exc:
        raise ProofError(
            f"full runtime dependency and bundle preflight failed: {type(exc).__name__}"
        ) from exc
    owners = (
        PostgresConfigurationStore,
        PostgresResearchStoreV1,
        DatabaseIdentityContext,
        build_container,
    )
    if any(not callable(owner) for owner in owners):
        raise ProofError("full runtime wiring owner is invalid")
    return {
        "runtimeImportsVerified": True,
        "configurationBundleVerified": True,
        "packageVersion": str(__version__),
        "bundleSha256": str(bundle.sha256),
        "rulesSha256": str(bundle.rules.sha256),
    }


def _docker_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
    ):
        environment.pop(name, None)
    if extra:
        environment.update({str(key): str(value) for key, value in extra.items()})
    return environment


def _run_docker(
    docker_context: str,
    command: Sequence[str],
    *,
    timeout: float,
    check: bool = True,
    environment: Mapping[str, str] | None = None,
    secrets_to_remove: Sequence[str] = (),
) -> subprocess.CompletedProcess[bytes]:
    if not DOCKER_CONTEXT_RE.fullmatch(str(docker_context)):
        raise ProofError("Docker context identity is invalid")
    extra = None if environment is None else dict(environment)
    return _run_command(
        ["docker", "--context", str(docker_context), *map(str, command)],
        timeout=timeout,
        check=check,
        environment=_docker_environment(extra),
        secrets_to_remove=secrets_to_remove,
    )


def _local_docker_context() -> dict[str, Any]:
    environment = _docker_environment()
    current = _run_command(
        ["docker", "context", "show"], timeout=15, environment=environment
    )
    name = _command_text(current)
    if not DOCKER_CONTEXT_RE.fullmatch(name):
        raise ProofError("Docker context identity is invalid")
    inspected = _run_command(
        ["docker", "context", "inspect", name, "--format", "{{json .}}"],
        timeout=15,
        environment=environment,
    )
    try:
        value = json.loads(_command_text(inspected))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProofError("Docker context metadata is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ProofError("Docker context metadata is incomplete")
    endpoint = dict(dict(value.get("Endpoints") or {}).get("docker") or {})
    host = str(endpoint.get("Host") or "")
    if os.name == "nt":
        local = bool(
            re.fullmatch(
                r"npipe:////\./pipe/(?:dockerDesktopLinuxEngine|docker_engine)",
                host,
                re.IGNORECASE,
            )
        )
        scheme = "npipe"
    else:
        local = host.startswith("unix:///") and not host.startswith("unix://///")
        scheme = "unix"
    if (
        not local
        or bool(endpoint.get("SkipTLSVerify"))
        or bool(value.get("TLSMaterial"))
    ):
        raise ProofError(
            "the PostgreSQL proof requires an explicitly local Docker context"
        )
    return {"name": name, "endpointScheme": scheme, "local": True}


def _docker_server(docker_context: str) -> dict[str, str]:
    completed = _run_docker(
        docker_context,
        ["version", "--format", "{{json .Server}}"],
        timeout=15,
    )
    try:
        value = json.loads(_command_text(completed))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProofError("Docker server metadata is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ProofError("Docker server metadata is incomplete")
    platform = dict(value.get("Platform") or {})
    result = {
        "platform": str(platform.get("Name") or ""),
        "version": str(value.get("Version") or ""),
        "os": str(value.get("Os") or ""),
        "architecture": str(value.get("Arch") or ""),
    }
    if not all(result.values()) or result["os"] != "linux":
        raise ProofError("a healthy Linux Docker server is required")
    return result


def _parse_manifest(
    raw: bytes,
    *,
    server_os: str,
    server_architecture: str,
    expected_index_digest: str = EXPECTED_POSTGRES_INDEX_DIGEST,
) -> dict[str, Any]:
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProofError("PostgreSQL image manifest is not valid JSON") from exc
    if not isinstance(manifest, Mapping):
        raise ProofError("PostgreSQL image manifest is incomplete")
    digest = str(manifest.get("digest") or "")
    if not SHA256_RE.fullmatch(digest):
        raise ProofError("PostgreSQL image manifest has no immutable digest")
    if digest != expected_index_digest:
        raise ProofError("PostgreSQL image tag no longer matches the governed digest")
    manifests = manifest.get("manifests")
    if not isinstance(manifests, list):
        raise ProofError("PostgreSQL image manifest has no platform list")
    matches = []
    for item in manifests:
        if not isinstance(item, Mapping):
            continue
        platform = dict(item.get("platform") or {})
        if (
            platform.get("os") == server_os
            and platform.get("architecture") == server_architecture
            and SHA256_RE.fullmatch(str(item.get("digest") or ""))
        ):
            matches.append(item)
    if len(matches) != 1:
        raise ProofError("PostgreSQL image has no unique Docker-server platform image")
    selected = matches[0]
    annotations = dict(selected.get("annotations") or {})
    declared_version = str(annotations.get("org.opencontainers.image.version") or "")
    if not re.fullmatch(r"17\.[0-9]+-alpine[0-9.]+", declared_version):
        raise ProofError("PostgreSQL image version annotation is invalid")
    return {
        "sourceTag": POSTGRES_IMAGE_TAG,
        "indexDigest": digest,
        "resolvedDigest": str(selected["digest"]),
        "resolvedReference": f"postgres@{selected['digest']}",
        "manifestMediaType": str(manifest.get("mediaType") or ""),
        "platformDigest": str(selected["digest"]),
        "platform": {
            "os": server_os,
            "architecture": server_architecture,
        },
        "declaredVersion": declared_version,
    }


def _resolve_image(
    *, docker_context: str, server_os: str, server_architecture: str
) -> dict[str, Any]:
    completed = _run_docker(
        docker_context,
        [
            "buildx",
            "imagetools",
            "inspect",
            POSTGRES_IMAGE_TAG,
            "--format",
            "{{json .Manifest}}",
        ],
        timeout=90,
    )
    return _parse_manifest(
        completed.stdout,
        server_os=server_os,
        server_architecture=server_architecture,
    )


def _new_context() -> dict[str, Any]:
    proof_id = uuid.uuid4().hex
    return {
        "proofId": proof_id,
        "containerName": f"balootx-pg-proof-{proof_id[:12]}",
        "proxyContainerName": f"balootx-pg-proof-proxy-{proof_id[:12]}",
        "networkName": f"balootx-pg-proof-net-{proof_id[:12]}",
        "accessNetworkName": f"balootx-pg-proof-access-{proof_id[:12]}",
        "database": f"balootx_proof_{proof_id[:12]}",
        "user": "balootx_proof",
        "password": secrets.token_hex(24),
        "proofRoot": Path(tempfile.mkdtemp(prefix=PROOF_ROOT_PREFIX)).resolve(),
    }


def _container_run_command(
    context: Mapping[str, Any], image_reference: str
) -> tuple[list[str], dict[str, str]]:
    password = str(context["password"])
    command = [
        "run",
        "--detach",
        "--name",
        str(context["containerName"]),
        "--label",
        f"{CONTAINER_LABEL_KIND}={CONTAINER_LABEL_KIND_VALUE}",
        "--label",
        f"{CONTAINER_LABEL_ID}={context['proofId']}",
        "--label",
        f"{CONTAINER_LABEL_ROLE}={DATABASE_CONTAINER_ROLE}",
        "--restart",
        "no",
        "--network",
        str(context["networkName"]),
        "--tmpfs",
        f"/var/lib/postgresql/data:{REQUIRED_TMPFS['/var/lib/postgresql/data']}",
        "--tmpfs",
        f"/var/run/postgresql:{REQUIRED_TMPFS['/var/run/postgresql']}",
        "--tmpfs",
        f"/tmp:{REQUIRED_TMPFS['/tmp']}",
        "--memory",
        "1g",
        "--cpus",
        "2",
        "--pids-limit",
        "256",
        "--shm-size",
        "256m",
        "--security-opt",
        "no-new-privileges=true",
        "--health-cmd",
        (f"pg_isready -h 127.0.0.1 -U {context['user']} -d {context['database']}"),
        "--health-interval",
        "1s",
        "--health-timeout",
        "3s",
        "--health-start-period",
        "5s",
        "--health-retries",
        "60",
        "--env",
        f"POSTGRES_DB={context['database']}",
        "--env",
        f"POSTGRES_USER={context['user']}",
        "--env",
        "POSTGRES_PASSWORD",
        "--env",
        "POSTGRES_INITDB_ARGS=--data-checksums",
        "--env",
        "PGDATA=/var/lib/postgresql/data/pgdata",
        image_reference,
    ]
    environment: dict[str, str] = {}
    environment["POSTGRES_PASSWORD"] = password
    if password in "\0".join(command):
        raise ProofError("PostgreSQL password must not appear in Docker arguments")
    return command, environment


def _proxy_relay_command(context: Mapping[str, Any]) -> str:
    container_name = str(context["containerName"])
    if not re.fullmatch(r"balootx-pg-proof-[0-9a-f]{12}", container_name):
        raise ProofError("PostgreSQL proof container name is invalid")
    return f"exec nc -lk -p 5432 -e sh -c 'exec nc {container_name} 5432'"


def _proxy_container_run_command(
    context: Mapping[str, Any], image_reference: str
) -> list[str]:
    return [
        "run",
        "--detach",
        "--name",
        str(context["proxyContainerName"]),
        "--label",
        f"{CONTAINER_LABEL_KIND}={CONTAINER_LABEL_KIND_VALUE}",
        "--label",
        f"{CONTAINER_LABEL_ID}={context['proofId']}",
        "--label",
        f"{CONTAINER_LABEL_ROLE}={PROXY_CONTAINER_ROLE}",
        "--restart",
        "no",
        "--network",
        str(context["accessNetworkName"]),
        "--network",
        str(context["networkName"]),
        "--publish",
        "127.0.0.1::5432",
        "--read-only",
        "--tmpfs",
        (f"/var/lib/postgresql/data:{PROXY_TMPFS['/var/lib/postgresql/data']}"),
        "--tmpfs",
        f"/tmp:{PROXY_TMPFS['/tmp']}",
        "--memory",
        "64m",
        "--cpus",
        "0.25",
        "--pids-limit",
        "32",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges=true",
        "--user",
        "postgres",
        "--entrypoint",
        "sh",
        image_reference,
        "-c",
        _proxy_relay_command(context),
    ]


def _inspect_container(reference: str, docker_context: str) -> dict[str, Any] | None:
    completed = _run_docker(
        docker_context,
        ["inspect", str(reference)],
        timeout=15,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace")
        if re.search(r"No such (?:object|container)", message, re.IGNORECASE):
            return None
        raise ProofError(
            _redact(f"Docker container inspection failed: {message[-2000:]}", ())
        )
    try:
        value = json.loads(completed.stdout.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProofError("Docker container inspection is not valid JSON") from exc
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ProofError("Docker container inspection is ambiguous")
    return value[0]


def _owned_container(
    reference: str,
    context: Mapping[str, Any],
    *,
    expected_role: str = DATABASE_CONTAINER_ROLE,
) -> dict[str, Any] | None:
    inspected = _inspect_container(reference, str(context["dockerContext"]))
    if inspected is None:
        return None
    labels = dict(dict(inspected.get("Config") or {}).get("Labels") or {})
    if (
        labels.get(CONTAINER_LABEL_KIND) != CONTAINER_LABEL_KIND_VALUE
        or labels.get(CONTAINER_LABEL_ID) != context["proofId"]
        or labels.get(CONTAINER_LABEL_ROLE) != expected_role
    ):
        raise ProofError("container is not owned by this PostgreSQL proof")
    return inspected


def _assert_container_name_unused(name: str, docker_context: str) -> None:
    if _inspect_container(name, docker_context) is not None:
        raise ProofError("the unique PostgreSQL proof container name is already in use")


def _inspect_network(reference: str, docker_context: str) -> dict[str, Any] | None:
    completed = _run_docker(
        docker_context,
        ["network", "inspect", str(reference)],
        timeout=15,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace")
        if re.search(
            r"No such (?:network|object)|network .+ not found",
            message,
            re.IGNORECASE,
        ):
            return None
        raise ProofError(
            _redact(f"Docker network inspection failed: {message[-2000:]}", ())
        )
    try:
        value = json.loads(completed.stdout.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProofError("Docker network inspection is not valid JSON") from exc
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ProofError("Docker network inspection is ambiguous")
    return value[0]


def _inspect_volume(reference: str, docker_context: str) -> dict[str, Any] | None:
    completed = _run_docker(
        docker_context,
        ["volume", "inspect", str(reference)],
        timeout=15,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace")
        if re.search(r"No such volume", message, re.IGNORECASE):
            return None
        raise ProofError(
            _redact(f"Docker volume inspection failed: {message[-2000:]}", ())
        )
    try:
        value = json.loads(completed.stdout.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProofError("Docker volume inspection is not valid JSON") from exc
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ProofError("Docker volume inspection is ambiguous")
    return value[0]


def _owned_network(
    reference: str,
    context: Mapping[str, Any],
    *,
    expected_role: str = INTERNAL_NETWORK_ROLE,
) -> dict[str, Any] | None:
    inspected = _inspect_network(reference, str(context["dockerContext"]))
    if inspected is None:
        return None
    labels = dict(inspected.get("Labels") or {})
    if (
        labels.get(CONTAINER_LABEL_KIND) != CONTAINER_LABEL_KIND_VALUE
        or labels.get(CONTAINER_LABEL_ID) != context["proofId"]
        or labels.get(CONTAINER_LABEL_ROLE) != expected_role
    ):
        raise ProofError("network is not owned by this PostgreSQL proof")
    return inspected


def _assert_network_name_unused(name: str, docker_context: str) -> None:
    if _inspect_network(name, docker_context) is not None:
        raise ProofError("the unique PostgreSQL proof network name is already in use")


def _create_network(context: Mapping[str, Any]) -> str:
    completed = _run_docker(
        str(context["dockerContext"]),
        [
            "network",
            "create",
            "--driver",
            "bridge",
            "--internal",
            "--label",
            f"{CONTAINER_LABEL_KIND}={CONTAINER_LABEL_KIND_VALUE}",
            "--label",
            f"{CONTAINER_LABEL_ID}={context['proofId']}",
            "--label",
            f"{CONTAINER_LABEL_ROLE}={INTERNAL_NETWORK_ROLE}",
            str(context["networkName"]),
        ],
        timeout=30,
    )
    network_id = _command_text(completed)
    if not CONTAINER_ID_RE.fullmatch(network_id):
        raise ProofError("Docker did not return one exact network identity")
    return network_id


def _create_access_network(context: Mapping[str, Any]) -> str:
    completed = _run_docker(
        str(context["dockerContext"]),
        [
            "network",
            "create",
            "--driver",
            "bridge",
            "--opt",
            "com.docker.network.bridge.host_binding_ipv4=127.0.0.1",
            "--label",
            f"{CONTAINER_LABEL_KIND}={CONTAINER_LABEL_KIND_VALUE}",
            "--label",
            f"{CONTAINER_LABEL_ID}={context['proofId']}",
            "--label",
            f"{CONTAINER_LABEL_ROLE}={ACCESS_NETWORK_ROLE}",
            str(context["accessNetworkName"]),
        ],
        timeout=30,
    )
    network_id = _command_text(completed)
    if not CONTAINER_ID_RE.fullmatch(network_id):
        raise ProofError("Docker did not return one exact access-network identity")
    return network_id


def _start_container(context: Mapping[str, Any], image_reference: str) -> str:
    command, environment = _container_run_command(context, image_reference)
    completed = _run_docker(
        str(context["dockerContext"]),
        command,
        environment=environment,
        timeout=180,
        secrets_to_remove=(str(context["password"]),),
    )
    container_id = _command_text(completed)
    if not CONTAINER_ID_RE.fullmatch(container_id):
        raise ProofError("Docker did not return one exact container identity")
    return container_id


def _start_proxy_container(context: Mapping[str, Any], image_reference: str) -> str:
    completed = _run_docker(
        str(context["dockerContext"]),
        _proxy_container_run_command(context, image_reference),
        timeout=60,
    )
    container_id = _command_text(completed)
    if not CONTAINER_ID_RE.fullmatch(container_id):
        raise ProofError("Docker did not return one exact proxy-container identity")
    return container_id


def _container_state(container_id: str, docker_context: str) -> dict[str, Any]:
    inspected = _inspect_container(container_id, docker_context)
    if inspected is None:
        raise ProofError("PostgreSQL proof container disappeared")
    state = inspected.get("State")
    if not isinstance(state, Mapping):
        raise ProofError("PostgreSQL proof container state is unavailable")
    return dict(state)


def _wait_for_health(
    container_id: str, *, docker_context: str, timeout_seconds: float
) -> None:
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        state = _container_state(container_id, docker_context)
        status = str(state.get("Status") or "")
        health = dict(state.get("Health") or {})
        health_status = str(health.get("Status") or "")
        if status == "running" and health_status == "healthy":
            return
        if status in {"dead", "exited", "removing"} or health_status == "unhealthy":
            raise ProofError(
                f"PostgreSQL proof container failed health: {status}/{health_status}"
            )
        if time.monotonic() >= deadline:
            raise ProofError("PostgreSQL proof container health timed out")
        time.sleep(1.0)


def _validated_container(
    container_id: str,
    context: Mapping[str, Any],
    image_reference: str,
) -> dict[str, Any]:
    inspected = _owned_container(container_id, context)
    if inspected is None:
        raise ProofError("PostgreSQL proof container is unavailable")
    if str(inspected.get("Id") or "") != container_id:
        raise ProofError("PostgreSQL proof container identity changed")
    config = dict(inspected.get("Config") or {})
    if str(config.get("Image") or "") != image_reference:
        raise ProofError("PostgreSQL proof container image is not the resolved digest")
    host_config = dict(inspected.get("HostConfig") or {})
    if host_config.get("Binds") not in (None, []):
        raise ProofError("PostgreSQL proof container has a host bind")
    mounts = inspected.get("Mounts") or []
    if any(
        not isinstance(item, Mapping) or item.get("Type") != "tmpfs" for item in mounts
    ):
        raise ProofError("PostgreSQL proof container has a persistent mount")
    configured_tmpfs = dict(host_config.get("Tmpfs") or {})
    if configured_tmpfs != REQUIRED_TMPFS:
        raise ProofError("PostgreSQL proof writable directories are not all tmpfs")
    destinations = {str(item.get("Destination")) for item in mounts}
    if not destinations.issubset(REQUIRED_TMPFS):
        raise ProofError("PostgreSQL proof has an unexpected tmpfs mount")
    published_ports = dict(
        dict(inspected.get("NetworkSettings") or {}).get("Ports") or {}
    )
    port_bindings = dict(host_config.get("PortBindings") or {})
    if published_ports.get("5432/tcp") not in (None, []) or any(
        value for value in port_bindings.values()
    ):
        raise ProofError("PostgreSQL proof database container published a port")
    restart_policy = str(dict(host_config.get("RestartPolicy") or {}).get("Name"))
    memory_bytes = int(host_config.get("Memory") or 0)
    nano_cpus = int(host_config.get("NanoCpus") or 0)
    pids_limit = int(host_config.get("PidsLimit") or 0)
    security_options = set(host_config.get("SecurityOpt") or ())
    accepted_no_new_privileges = {
        "no-new-privileges",
        "no-new-privileges=true",
        "no-new-privileges:true",
    }
    attached_networks = dict(
        dict(inspected.get("NetworkSettings") or {}).get("Networks") or {}
    )
    expected_network = dict(attached_networks.get(str(context["networkName"])) or {})
    if (
        restart_policy != "no"
        or memory_bytes != 1_073_741_824
        or nano_cpus != 2_000_000_000
        or pids_limit != 256
        or not (security_options & accepted_no_new_privileges)
        or str(host_config.get("NetworkMode") or "") != context["networkName"]
        or set(attached_networks) != {str(context["networkName"])}
        or str(expected_network.get("NetworkID") or "") != context["networkId"]
    ):
        raise ProofError("PostgreSQL proof resource or isolation limits drifted")
    return {
        "containerIdSha256": _sha256_text(container_id),
        "publishedPortCount": 0,
        "persistentMountCount": 0,
        "dataMount": "tmpfs",
        "restartPolicy": restart_policy,
        "memoryBytes": memory_bytes,
        "nanoCpus": nano_cpus,
        "pidsLimit": pids_limit,
        "dedicatedInternalNetwork": True,
    }


def _validated_proxy_container(
    container_id: str,
    context: Mapping[str, Any],
    image_reference: str,
) -> dict[str, Any]:
    inspected = _owned_container(
        container_id, context, expected_role=PROXY_CONTAINER_ROLE
    )
    if inspected is None or str(inspected.get("Id") or "") != container_id:
        raise ProofError("PostgreSQL proof proxy-container identity changed")
    config = dict(inspected.get("Config") or {})
    if (
        str(config.get("Image") or "") != image_reference
        or str(config.get("User") or "") != "postgres"
        or list(config.get("Entrypoint") or ()) != ["sh"]
        or list(config.get("Cmd") or ()) != ["-c", _proxy_relay_command(context)]
        or any(
            str(item).startswith("POSTGRES_PASSWORD=")
            for item in (config.get("Env") or ())
        )
    ):
        raise ProofError("PostgreSQL proof proxy-container identity drifted")
    state = dict(inspected.get("State") or {})
    if str(state.get("Status") or "") != "running":
        raise ProofError("PostgreSQL proof proxy-container is not running")
    host_config = dict(inspected.get("HostConfig") or {})
    if host_config.get("Binds") not in (None, []):
        raise ProofError("PostgreSQL proof proxy-container has a host bind")
    mounts = inspected.get("Mounts") or []
    if any(
        not isinstance(item, Mapping)
        or item.get("Type") != "tmpfs"
        or str(item.get("Destination")) not in PROXY_TMPFS
        for item in mounts
    ):
        raise ProofError("PostgreSQL proof proxy-container has a persistent mount")
    if dict(host_config.get("Tmpfs") or {}) != PROXY_TMPFS:
        raise ProofError("PostgreSQL proof proxy scratch directory is not tmpfs")
    port_rows = (
        dict(dict(inspected.get("NetworkSettings") or {}).get("Ports") or {}).get(
            "5432/tcp"
        )
        or []
    )
    requested_bindings = dict(host_config.get("PortBindings") or {})
    requested_rows = requested_bindings.get("5432/tcp") or []
    if (
        not isinstance(port_rows, list)
        or len(port_rows) != 1
        or not isinstance(requested_rows, list)
        or len(requested_rows) != 1
    ):
        raise ProofError("PostgreSQL proof proxy port mapping is ambiguous")
    mapping = dict(port_rows[0])
    requested_mapping = dict(requested_rows[0])
    if (
        str(mapping.get("HostIp") or "") != "127.0.0.1"
        or str(requested_mapping.get("HostIp") or "") != "127.0.0.1"
        or str(requested_mapping.get("HostPort") or "") != ""
        or set(requested_bindings) != {"5432/tcp"}
    ):
        raise ProofError("PostgreSQL proof proxy port is not loopback-only")
    try:
        port = int(mapping.get("HostPort"))
    except (TypeError, ValueError) as exc:
        raise ProofError("PostgreSQL proof proxy host port is invalid") from exc
    if not 1 <= port <= 65535:
        raise ProofError("PostgreSQL proof proxy host port is out of range")
    restart_policy = str(dict(host_config.get("RestartPolicy") or {}).get("Name"))
    security_options = set(host_config.get("SecurityOpt") or ())
    attached_networks = dict(
        dict(inspected.get("NetworkSettings") or {}).get("Networks") or {}
    )
    internal_network = dict(attached_networks.get(str(context["networkName"])) or {})
    access_network = dict(
        attached_networks.get(str(context["accessNetworkName"])) or {}
    )
    if (
        restart_policy != "no"
        or host_config.get("ReadonlyRootfs") is not True
        or int(host_config.get("Memory") or 0) != 67_108_864
        or int(host_config.get("NanoCpus") or 0) != 250_000_000
        or int(host_config.get("PidsLimit") or 0) != 32
        or {str(item).upper() for item in (host_config.get("CapDrop") or ())} != {"ALL"}
        or not {
            "no-new-privileges",
            "no-new-privileges=true",
            "no-new-privileges:true",
        }.intersection(security_options)
        or str(host_config.get("NetworkMode") or "") != context["accessNetworkName"]
        or set(attached_networks)
        != {str(context["networkName"]), str(context["accessNetworkName"])}
        or str(internal_network.get("NetworkID") or "") != context["networkId"]
        or str(access_network.get("NetworkID") or "") != context["accessNetworkId"]
    ):
        raise ProofError("PostgreSQL proof proxy resource or isolation limits drifted")
    return {
        "containerIdSha256": _sha256_text(container_id),
        "imageDigestPinned": True,
        "loopbackOnly": True,
        "hostPort": port,
        "readOnlyRoot": True,
        "nonRootUser": True,
        "capabilitiesDropped": True,
        "persistentMountCount": 0,
        "memoryBytes": 67_108_864,
        "nanoCpus": 250_000_000,
        "pidsLimit": 32,
    }


def _validated_network(
    network_id: str,
    context: Mapping[str, Any],
    *,
    expected_container_ids: Sequence[str],
) -> dict[str, Any]:
    inspected = _owned_network(network_id, context)
    if inspected is None or str(inspected.get("Id") or "") != network_id:
        raise ProofError("PostgreSQL proof network identity changed")
    containers = dict(inspected.get("Containers") or {})
    if (
        str(inspected.get("Driver") or "") != "bridge"
        or inspected.get("Internal") is not True
        or set(containers) != set(expected_container_ids)
    ):
        raise ProofError("PostgreSQL proof network isolation drifted")
    return {
        "networkIdSha256": _sha256_text(network_id),
        "internal": True,
        "driver": "bridge",
        "attachedContainerCount": len(expected_container_ids),
    }


def _validated_access_network(
    network_id: str,
    context: Mapping[str, Any],
    *,
    expected_proxy_container_id: str,
) -> dict[str, Any]:
    inspected = _owned_network(network_id, context, expected_role=ACCESS_NETWORK_ROLE)
    if inspected is None or str(inspected.get("Id") or "") != network_id:
        raise ProofError("PostgreSQL proof access-network identity changed")
    containers = dict(inspected.get("Containers") or {})
    options = dict(inspected.get("Options") or {})
    if (
        str(inspected.get("Driver") or "") != "bridge"
        or inspected.get("Internal") is not False
        or set(containers) != {expected_proxy_container_id}
        or options.get("com.docker.network.bridge.host_binding_ipv4") != "127.0.0.1"
    ):
        raise ProofError("PostgreSQL proof access-network isolation drifted")
    return {
        "networkIdSha256": _sha256_text(network_id),
        "internal": False,
        "driver": "bridge",
        "defaultHostBinding": "127.0.0.1",
        "attachedContainerCount": 1,
    }


def _container_logs(
    container_id: str,
    docker_context: str,
    secrets_to_remove: Sequence[str],
) -> dict[str, Any]:
    completed = _run_docker(
        docker_context,
        ["logs", "--tail", "300", container_id],
        timeout=20,
        check=False,
        secrets_to_remove=secrets_to_remove,
    )
    combined = completed.stdout + completed.stderr
    text = _redact(combined.decode("utf-8", errors="replace"), secrets_to_remove)
    return {
        "sha256": _sha256_text(text),
        "tail": text[-4000:],
        "returnCode": int(completed.returncode),
    }


def _cleanup_container(
    reference: str,
    context: Mapping[str, Any],
    *,
    expected_role: str = DATABASE_CONTAINER_ROLE,
    wait_for_appearance_seconds: float = 0,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, float(wait_for_appearance_seconds))
    while True:
        inspected = _owned_container(reference, context, expected_role=expected_role)
        if inspected is not None or time.monotonic() >= deadline:
            break
        time.sleep(0.25)
    if inspected is None:
        completed = _run_docker(
            str(context["dockerContext"]),
            [
                "ps",
                "--all",
                "--quiet",
                "--no-trunc",
                "--filter",
                f"label={CONTAINER_LABEL_KIND}={CONTAINER_LABEL_KIND_VALUE}",
                "--filter",
                f"label={CONTAINER_LABEL_ID}={context['proofId']}",
                "--filter",
                f"label={CONTAINER_LABEL_ROLE}={expected_role}",
            ],
            timeout=15,
        )
        identities = [
            line.strip()
            for line in _command_text(completed).splitlines()
            if line.strip()
        ]
        if len(identities) > 1:
            raise ProofError("PostgreSQL proof container ownership is ambiguous")
        if identities:
            inspected = _owned_container(
                identities[0], context, expected_role=expected_role
            )
    if inspected is None:
        if wait_for_appearance_seconds > 0:
            raise ProofError(
                "Docker container start was attempted but absence is indeterminate"
            )
        return {"status": "already_absent"}
    container_id = str(inspected.get("Id") or "")
    if not CONTAINER_ID_RE.fullmatch(container_id):
        raise ProofError("refusing to remove a container with an invalid identity")
    owned_volume_names: list[str] = []
    for mount in inspected.get("Mounts") or ():
        if not isinstance(mount, Mapping) or mount.get("Type") != "volume":
            continue
        volume_name = str(mount.get("Name") or "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}", volume_name):
            raise ProofError("owned container has an invalid volume identity")
        volume = _inspect_volume(volume_name, str(context["dockerContext"]))
        if volume is None or str(volume.get("Name") or "") != volume_name:
            raise ProofError("owned container volume identity changed")
        owned_volume_names.append(volume_name)
    if len(owned_volume_names) != len(set(owned_volume_names)):
        raise ProofError("owned container volume identity is ambiguous")
    _run_docker(
        str(context["dockerContext"]),
        ["rm", "--force", "--volumes", container_id],
        timeout=60,
    )
    if _inspect_container(container_id, str(context["dockerContext"])) is not None:
        raise ProofError("PostgreSQL proof container cleanup was incomplete")
    if any(
        _inspect_volume(name, str(context["dockerContext"])) is not None
        for name in owned_volume_names
    ):
        raise ProofError("PostgreSQL proof volume cleanup was incomplete")
    return {
        "status": "removed",
        "containerIdSha256": _sha256_text(container_id),
        "ownedPersistentVolumeCount": len(owned_volume_names),
        "persistentVolumeResidueCount": 0,
    }


def _cleanup_network(
    reference: str,
    context: Mapping[str, Any],
    *,
    expected_role: str = INTERNAL_NETWORK_ROLE,
    wait_for_appearance_seconds: float = 0,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, float(wait_for_appearance_seconds))
    while True:
        inspected = _owned_network(reference, context, expected_role=expected_role)
        if inspected is not None or time.monotonic() >= deadline:
            break
        time.sleep(0.25)
    if inspected is None:
        if wait_for_appearance_seconds > 0:
            raise ProofError(
                "Docker network creation was attempted but absence is indeterminate"
            )
        return {"status": "already_absent"}
    network_id = str(inspected.get("Id") or "")
    if not CONTAINER_ID_RE.fullmatch(network_id):
        raise ProofError("refusing to remove a network with an invalid identity")
    if dict(inspected.get("Containers") or {}):
        raise ProofError("refusing to remove a non-empty PostgreSQL proof network")
    _run_docker(
        str(context["dockerContext"]),
        ["network", "rm", network_id],
        timeout=30,
    )
    if _inspect_network(network_id, str(context["dockerContext"])) is not None:
        raise ProofError("PostgreSQL proof network cleanup was incomplete")
    return {"status": "removed", "networkIdSha256": _sha256_text(network_id)}


def _safe_remove_proof_root(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ProofError("refusing to remove an unsafe PostgreSQL proof directory")
    root = path.resolve()
    expected_parent = Path(tempfile.gettempdir()).resolve()
    if root.parent != expected_parent or not root.name.startswith(PROOF_ROOT_PREFIX):
        raise ProofError("refusing to remove an unsafe PostgreSQL proof directory")
    if root.exists():
        descendants = sorted(
            root.rglob("*"), key=lambda item: len(item.parts), reverse=True
        )
        for item in descendants:
            if item.is_symlink():
                raise ProofError(
                    "refusing to remove an unsafe PostgreSQL proof directory"
                )
            resolved = item.resolve()
            if root not in resolved.parents:
                raise ProofError(
                    "refusing to remove an unsafe PostgreSQL proof directory"
                )
            item.chmod(0o700 if item.is_dir() else 0o600)
        root.chmod(0o700)
        shutil.rmtree(root)
    return {"status": "removed", "pathPersisted": False}


def _dependency_versions() -> dict[str, str]:
    try:
        import psycopg
        import psycopg_pool
    except ImportError as exc:
        raise ProofError(
            'PostgreSQL proof requires: pip install "baloot-platform[postgres]"'
        ) from exc
    return {
        "python": sys.version.split()[0],
        "psycopg": str(psycopg.__version__),
        "psycopgPool": str(psycopg_pool.__version__),
    }


def _database_url(context: Mapping[str, Any], port: int) -> str:
    if not 1 <= int(port) <= 65535:
        raise ProofError("PostgreSQL proof host port is invalid")
    user = str(context["user"])
    password = str(context["password"])
    database = str(context["database"])
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,62}", user):
        raise ProofError("PostgreSQL proof user is invalid")
    if not re.fullmatch(r"balootx_proof_[0-9a-f]{12}", database):
        raise ProofError("PostgreSQL proof database identity is invalid")
    if not re.fullmatch(r"[0-9a-f]{48}", password):
        raise ProofError("PostgreSQL proof password is invalid")
    session_options = quote(
        "-c statement_timeout=30000 "
        "-c lock_timeout=10000 "
        "-c idle_in_transaction_session_timeout=30000",
        safe="",
    )
    return (
        f"postgresql://{user}:{password}@127.0.0.1:{int(port)}/{database}"
        f"?connect_timeout=5&application_name=balootx_live_postgres_proof"
        f"&options={session_options}"
    )


def _connect(database_url: str):
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - preflight owns this path
        raise ProofError("psycopg is unavailable") from exc
    return psycopg.connect(database_url, connect_timeout=5)


def _database_readiness(database_url: str, expected_database: str) -> dict[str, Any]:
    with _connect(database_url) as connection:
        row = connection.execute(
            "SELECT current_database(),current_setting('server_version_num'),"
            "current_setting('data_checksums')"
        ).fetchone()
    if row is None or str(row[0]) != expected_database:
        raise ProofError("PostgreSQL proof connected to the wrong database")
    server_version_num = str(row[1])
    if not server_version_num.startswith("17"):
        raise ProofError("PostgreSQL proof did not start PostgreSQL 17")
    if str(row[2]).lower() != "on":
        raise ProofError("PostgreSQL proof data checksums are disabled")
    return {
        "databaseIdentitySha256": _sha256_text(expected_database),
        "serverMajor": 17,
        "serverVersionNum": server_version_num,
        "dataChecksums": True,
    }


SCHEMA_QUERIES = {
    "columns": """
        SELECT table_schema,table_name,ordinal_position,column_name,data_type,
               udt_name,is_nullable,COALESCE(column_default,'')
        FROM information_schema.columns
        WHERE table_schema NOT IN ('pg_catalog','information_schema')
        ORDER BY 1,2,3,4
    """,
    "constraints": """
        SELECT namespace.nspname,relation.relname,constraint_row.conname,
               constraint_row.contype,
               pg_get_constraintdef(constraint_row.oid,true)
        FROM pg_constraint AS constraint_row
        JOIN pg_class AS relation ON relation.oid=constraint_row.conrelid
        JOIN pg_namespace AS namespace ON namespace.oid=relation.relnamespace
        WHERE namespace.nspname NOT IN ('pg_catalog','information_schema')
        ORDER BY 1,2,3,4,5
    """,
    "indexes": """
        SELECT schemaname,tablename,indexname,indexdef
        FROM pg_indexes
        WHERE schemaname NOT IN ('pg_catalog','information_schema')
        ORDER BY 1,2,3,4
    """,
    "policies": """
        SELECT schemaname,tablename,policyname,permissive,roles::text,cmd,
               COALESCE(qual,''),COALESCE(with_check,'')
        FROM pg_policies
        WHERE schemaname NOT IN ('pg_catalog','information_schema')
        ORDER BY 1,2,3,4,5,6,7,8
    """,
    "triggers": """
        SELECT event_object_schema,event_object_table,trigger_name,
               event_manipulation,action_timing,action_statement
        FROM information_schema.triggers
        WHERE event_object_schema NOT IN ('pg_catalog','information_schema')
        ORDER BY 1,2,3,4,5,6
    """,
    "functions": """
        SELECT namespace.nspname,function_row.proname,
               pg_get_function_identity_arguments(function_row.oid),
               function_row.prokind,language.lanname,
               pg_get_functiondef(function_row.oid)
        FROM pg_proc AS function_row
        JOIN pg_namespace AS namespace
          ON namespace.oid=function_row.pronamespace
        JOIN pg_language AS language ON language.oid=function_row.prolang
        WHERE namespace.nspname NOT IN ('pg_catalog','information_schema')
        ORDER BY 1,2,3,4,5,6
    """,
    "rowSecurity": """
        SELECT namespace.nspname,relation.relname,
               relation.relrowsecurity,relation.relforcerowsecurity
        FROM pg_class AS relation
        JOIN pg_namespace AS namespace ON namespace.oid=relation.relnamespace
        WHERE namespace.nspname NOT IN ('pg_catalog','information_schema')
          AND relation.relkind IN ('r','p')
        ORDER BY 1,2,3,4
    """,
}


def _schema_fingerprint(database_url: str) -> dict[str, Any]:
    sections: dict[str, list[list[Any]]] = {}
    with _connect(database_url) as connection:
        for name, query in SCHEMA_QUERIES.items():
            rows = connection.execute(query).fetchall()
            sections[name] = [
                [None if value is None else str(value) for value in row] for row in rows
            ]
    return {
        "sha256": _sha256_text(_canonical_json(sections)),
        "counts": {name: len(rows) for name, rows in sorted(sections.items())},
    }


def _migration_inventory(root: Path = ROOT) -> list[dict[str, Any]]:
    migrations = sorted((root / "migrations").glob("*.sql"))
    if not migrations:
        raise ProofError("PostgreSQL migration inventory is empty")
    inventory = []
    for path in migrations:
        if (
            path.is_symlink()
            or not path.is_file()
            or not MIGRATION_NAME_RE.fullmatch(path.name)
        ):
            raise ProofError("PostgreSQL migration inventory contains an unsafe path")
        inventory.append(
            {"name": path.name, "sha256": _sha256_bytes(path.read_bytes())}
        )
    return inventory


def _stage_migrations(proof_root: Path) -> tuple[Path, list[dict[str, Any]]]:
    source_inventory = _migration_inventory()
    inventory_by_name = {
        str(item["name"]): str(item["sha256"]) for item in source_inventory
    }
    for name, expected_sha256 in REQUIRED_MIGRATION_HASHES.items():
        authored = ROOT / "migrations" / name
        if name not in inventory_by_name or not authored.is_file():
            raise ProofError(f"required PostgreSQL migration is missing: {name}")
        authored_payload = authored.read_bytes()
        if (
            inventory_by_name[name] != expected_sha256
            or _sha256_bytes(authored_payload) != expected_sha256
        ):
            raise ProofError(f"required PostgreSQL migration identity mismatch: {name}")
        packaged = PACKAGED_MIGRATIONS / name
        if packaged.is_symlink() or not packaged.is_file():
            raise ProofError(
                f"required packaged PostgreSQL migration is missing: {name}"
            )
        packaged_payload = packaged.read_bytes()
        if (
            packaged_payload != authored_payload
            or _sha256_bytes(packaged_payload) != expected_sha256
        ):
            raise ProofError(
                f"packaged PostgreSQL migration is not byte-identical: {name}"
            )
    snapshot_root = proof_root / "migration-snapshot"
    migrations_root = snapshot_root / "migrations"
    migrations_root.mkdir(parents=True, exist_ok=False)
    for item in source_inventory:
        source = ROOT / "migrations" / str(item["name"])
        payload = source.read_bytes()
        if _sha256_bytes(payload) != item["sha256"]:
            raise ProofError("PostgreSQL migration bytes changed while snapshotting")
        target = migrations_root / str(item["name"])
        with target.open("xb") as stream:
            stream.write(payload)
        target.chmod(0o444)
    snapshot_inventory = _migration_inventory(snapshot_root)
    if snapshot_inventory != source_inventory:
        raise ProofError("PostgreSQL migration snapshot is not byte-identical")
    return snapshot_root, snapshot_inventory


SQL_IDENTIFIER_PATTERN = r'(?:"(?:[^"]|"")*"|[a-z_][a-z0-9_$]*)'
SQL_RELATION_PATTERN = rf"{SQL_IDENTIFIER_PATTERN}(?:\s*\.\s*{SQL_IDENTIFIER_PATTERN})?"
POLICY_LIFECYCLE_RE = re.compile(
    rf"\b(?P<operation>CREATE|DROP)\s+POLICY"
    rf"(?:\s+IF\s+EXISTS)?\s+"
    rf"(?P<policy>{SQL_IDENTIFIER_PATTERN})\s+ON\s+"
    rf"(?P<relation>{SQL_RELATION_PATTERN})",
    re.IGNORECASE,
)


def _normalized_identifier(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith('"'):
        return normalized[1:-1].replace('""', '"')
    return normalized.lower()


def _qualified_relation(value: str) -> str:
    match = re.fullmatch(
        rf"(?P<first>{SQL_IDENTIFIER_PATTERN})"
        rf"(?:\s*\.\s*(?P<second>{SQL_IDENTIFIER_PATTERN}))?",
        value.strip(),
        re.IGNORECASE,
    )
    if match is None:
        raise ProofError("PostgreSQL migration relation identity is invalid")
    first = _normalized_identifier(match.group("first"))
    second = match.group("second")
    if second is None:
        return f"public.{first}"
    return f"{first}.{_normalized_identifier(second)}"


def _migration_policy_identities(migrations: Sequence[Path]) -> set[tuple[str, str]]:
    policies: set[tuple[str, str]] = set()
    for path in sorted(migrations, key=lambda item: item.name):
        source = path.read_text(encoding="utf-8")
        for match in POLICY_LIFECYCLE_RE.finditer(source):
            identity = (
                _qualified_relation(match.group("relation")),
                _normalized_identifier(match.group("policy")),
            )
            if match.group("operation").upper() == "DROP":
                policies.discard(identity)
            else:
                policies.add(identity)
    return policies


def _migration_schema_contract(snapshot_root: Path) -> dict[str, Any]:
    migrations = sorted((snapshot_root / "migrations").glob("*.sql"))
    source = "\n".join(path.read_text(encoding="utf-8") for path in migrations)
    identifier = r"[a-z_][a-z0-9_]*"
    relation = rf"(?:{identifier}\.)?{identifier}"
    tables = {
        _qualified_relation(match)
        for match in re.findall(
            rf"\bCREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+({relation})",
            source,
            re.IGNORECASE,
        )
    }
    constraints = {
        (_qualified_relation(table), name.lower())
        for table, name in re.findall(
            rf"\bALTER\s+TABLE\s+({relation})[^;]*?\bADD\s+CONSTRAINT\s+({identifier})",
            source,
            re.IGNORECASE,
        )
    }
    functions = {
        _qualified_relation(name)
        for name in re.findall(
            rf"\bCREATE\s+OR\s+REPLACE\s+FUNCTION\s+({relation})",
            source,
            re.IGNORECASE,
        )
    }
    triggers = {
        (_qualified_relation(table), name.lower())
        for name, table in re.findall(
            rf"\bCREATE\s+TRIGGER\s+({identifier})[^;]*?\bON\s+({relation})",
            source,
            re.IGNORECASE,
        )
    }
    policies = _migration_policy_identities(migrations)
    rls_enabled = {
        _qualified_relation(table)
        for table in re.findall(
            rf"\bALTER\s+TABLE\s+({relation})\s+ENABLE\s+ROW\s+LEVEL\s+SECURITY",
            source,
            re.IGNORECASE,
        )
    }
    rls_forced = {
        _qualified_relation(table)
        for table in re.findall(
            rf"\bALTER\s+TABLE\s+({relation})\s+FORCE\s+ROW\s+LEVEL\s+SECURITY",
            source,
            re.IGNORECASE,
        )
    }
    if not tables or not constraints or not rls_enabled:
        raise ProofError("PostgreSQL migration schema contract is incomplete")
    contract = {
        "tables": sorted(tables),
        "constraints": [list(item) for item in sorted(constraints)],
        "functions": sorted(functions),
        "triggers": [list(item) for item in sorted(triggers)],
        "policies": [list(item) for item in sorted(policies)],
        "rlsEnabled": sorted(rls_enabled),
        "rlsForced": sorted(rls_forced),
    }
    return {**contract, "sha256": _sha256_text(_canonical_json(contract))}


def _run_migration_once(
    database_url: str,
    snapshot_root: Path,
    expected_names: Sequence[str],
    secrets_to_remove: Sequence[str],
) -> dict[str, Any]:
    completed, _environment = _run_python_child(
        [str(ROOT / "scripts/migrate_postgres.py")],
        timeout=180,
        extra_environment={
            "BALOOT_DATABASE_URL": database_url,
            "BALOOT_ROOT": str(snapshot_root),
        },
        secrets_to_remove=secrets_to_remove,
    )
    lines = _command_text(completed).splitlines()
    applied = [
        line.removeprefix("applied ") for line in lines if line.startswith("applied ")
    ]
    if applied != list(expected_names):
        raise ProofError(
            "PostgreSQL migration output does not match the source inventory"
        )
    if not lines or lines[-1] != "Baloot PostgreSQL schema is current":
        raise ProofError("PostgreSQL migration completion marker is missing")
    inventory = _migration_inventory(snapshot_root)
    return {
        "applied": len(applied),
        "executedInventorySha256": _sha256_text(_canonical_json(inventory)),
    }


def _assert_relations(database_url: str, relations: Sequence[str]) -> None:
    with _connect(database_url) as connection:
        for relation in relations:
            row = connection.execute(
                "SELECT to_regclass(%s)::text", (relation,)
            ).fetchone()
            if row is None or str(row[0] or "") != relation:
                raise ProofError(f"required PostgreSQL relation is missing: {relation}")


def _assert_migration_constraints(database_url: str) -> None:
    with _connect(database_url) as connection:
        audit_fk = connection.execute(
            """SELECT COUNT(*)
               FROM pg_constraint AS foreign_key
               WHERE foreign_key.contype='f'
                 AND foreign_key.conrelid='research.audit_traces'::regclass
                 AND foreign_key.confrelid='research.runs'::regclass"""
        ).fetchone()
        receipt_checks = connection.execute(
            """SELECT COUNT(*)
               FROM pg_constraint AS check_row
               WHERE check_row.contype='c'
                 AND check_row.conrelid=
                     'research.recipe_mutation_receipts'::regclass"""
        ).fetchone()
        fenced_occurrence_fks = connection.execute(
            """SELECT foreign_key.conrelid::regclass::text,
                      ARRAY(
                        SELECT attribute.attname
                        FROM unnest(foreign_key.conkey) WITH ORDINALITY
                          AS local_key(attnum,position)
                        JOIN pg_attribute AS attribute
                          ON attribute.attrelid=foreign_key.conrelid
                         AND attribute.attnum=local_key.attnum
                        ORDER BY local_key.position
                      ),
                      foreign_key.confrelid::regclass::text,
                      ARRAY(
                        SELECT attribute.attname
                        FROM unnest(foreign_key.confkey) WITH ORDINALITY
                          AS remote_key(attnum,position)
                        JOIN pg_attribute AS attribute
                          ON attribute.attrelid=foreign_key.confrelid
                         AND attribute.attnum=remote_key.attnum
                        ORDER BY remote_key.position
                      )
               FROM pg_constraint AS foreign_key
               WHERE foreign_key.contype='f'
                 AND foreign_key.conrelid IN (
                   'research.result_occurrences'::regclass,
                   'research.semantic_disagreement_occurrences'::regclass,
                   'research.sequential_search_receipt_occurrences'::regclass
                 )"""
        ).fetchall()
        fenced_checks = connection.execute(
            """SELECT check_row.conrelid::regclass::text,
                      ARRAY(
                        SELECT attribute.attname
                        FROM unnest(check_row.conkey) WITH ORDINALITY
                          AS check_key(attnum,position)
                        JOIN pg_attribute AS attribute
                          ON attribute.attrelid=check_row.conrelid
                         AND attribute.attnum=check_key.attnum
                        ORDER BY check_key.position
                      ),
                      pg_get_expr(check_row.conbin,check_row.conrelid)
               FROM pg_constraint AS check_row
               WHERE check_row.contype='c'
                 AND check_row.conrelid IN (
                   'research.results'::regclass,
                   'research.semantic_disagreements'::regclass,
                   'research.result_occurrences'::regclass,
                   'research.semantic_disagreement_occurrences'::regclass,
                   'research.sequential_search_receipt_occurrences'::regclass
                 )"""
        ).fetchall()
        fenced_indexes = connection.execute(
            """SELECT table_relation.oid::regclass::text,
                      index_relation.relname,index_row.indisunique,
                      ARRAY(
                        SELECT attribute.attname
                        FROM unnest(index_row.indkey::smallint[]) WITH ORDINALITY
                          AS index_key(attnum,position)
                        JOIN pg_attribute AS attribute
                          ON attribute.attrelid=table_relation.oid
                         AND attribute.attnum=index_key.attnum
                        ORDER BY index_key.position
                      ),
                      pg_get_expr(index_row.indpred,index_row.indrelid)
               FROM pg_index AS index_row
               JOIN pg_class AS index_relation
                 ON index_relation.oid=index_row.indexrelid
               JOIN pg_class AS table_relation
                 ON table_relation.oid=index_row.indrelid
               WHERE index_relation.relname IN (
                 'ux_research_shards_identity_run',
                 'ux_research_receipt_bundle_hash_seed',
                 'ux_research_search_receipts_admitted_seed'
               )"""
        ).fetchall()
        nullable_legacy_fence_columns = connection.execute(
            """SELECT COUNT(*)
               FROM information_schema.columns
               WHERE table_schema='research'
                 AND table_name IN ('results','semantic_disagreements')
                 AND column_name IN ('attempt_count','fencing_token')
                 AND is_nullable='YES'"""
        ).fetchone()
    if audit_fk is None or int(audit_fk[0]) != 1:
        raise ProofError("Research audit-trace run foreign key is missing or ambiguous")
    if receipt_checks is None or int(receipt_checks[0]) < 3:
        raise ProofError("Research recipe mutation constraints are incomplete")
    actual_foreign_keys = {
        (str(row[0]), tuple(row[1]), str(row[2]), tuple(row[3]))
        for row in fenced_occurrence_fks
    }
    expected_foreign_keys = {
        (
            "research.result_occurrences",
            ("result_code",),
            "research.results",
            ("result_code",),
        ),
        (
            "research.result_occurrences",
            ("run_code",),
            "research.runs",
            ("run_code",),
        ),
        (
            "research.result_occurrences",
            ("shard_id", "run_code"),
            "research.shards",
            ("shard_id", "run_code"),
        ),
        (
            "research.semantic_disagreement_occurrences",
            ("disagreement_id",),
            "research.semantic_disagreements",
            ("disagreement_id",),
        ),
        (
            "research.semantic_disagreement_occurrences",
            ("run_code",),
            "research.runs",
            ("run_code",),
        ),
        (
            "research.semantic_disagreement_occurrences",
            ("shard_id", "run_code"),
            "research.shards",
            ("shard_id", "run_code"),
        ),
        (
            "research.sequential_search_receipt_occurrences",
            ("receipt_hash", "seed"),
            "research.sequential_search_receipt_bundles",
            ("receipt_hash", "seed"),
        ),
        (
            "research.sequential_search_receipt_occurrences",
            ("run_code",),
            "research.runs",
            ("run_code",),
        ),
        (
            "research.sequential_search_receipt_occurrences",
            ("shard_id", "run_code"),
            "research.shards",
            ("shard_id", "run_code"),
        ),
    }
    if actual_foreign_keys != expected_foreign_keys:
        raise ProofError("Research fenced occurrence foreign keys are not exact")

    def canonical_expression(value: Any) -> str:
        return re.sub(r"[\s()]", "", str(value or "").lower())

    actual_checks = {
        (str(row[0]), tuple(row[1]), canonical_expression(row[2]))
        for row in fenced_checks
    }
    paired_fence = canonical_expression(
        "(attempt_count IS NULL AND fencing_token IS NULL) OR "
        "(attempt_count IS NOT NULL AND fencing_token IS NOT NULL AND "
        "attempt_count >= 1 AND fencing_token >= 1)"
    )
    expected_checks = {
        ("research.results", ("attempt_count", "fencing_token"), paired_fence),
        (
            "research.semantic_disagreements",
            ("attempt_count", "fencing_token"),
            paired_fence,
        ),
        *{
            (table, (column,), canonical_expression(f"{column} >= 1"))
            for table in (
                "research.result_occurrences",
                "research.semantic_disagreement_occurrences",
                "research.sequential_search_receipt_occurrences",
            )
            for column in ("attempt_count", "fencing_token")
        },
    }
    if actual_checks != expected_checks:
        raise ProofError("Research fenced occurrence checks are not exact")

    actual_indexes = {
        (
            str(row[0]),
            str(row[1]),
            bool(row[2]),
            tuple(row[3]),
            canonical_expression(row[4]),
        )
        for row in fenced_indexes
    }
    expected_indexes = {
        (
            "research.shards",
            "ux_research_shards_identity_run",
            True,
            ("shard_id", "run_code"),
            "",
        ),
        (
            "research.sequential_search_receipt_bundles",
            "ux_research_receipt_bundle_hash_seed",
            True,
            ("receipt_hash", "seed"),
            "",
        ),
        (
            "research.sequential_search_receipt_occurrences",
            "ux_research_search_receipts_admitted_seed",
            True,
            ("run_code", "seed"),
            "admitted",
        ),
    }
    if actual_indexes != expected_indexes:
        raise ProofError("Research fenced receipt indexes are not exact")
    if (
        nullable_legacy_fence_columns is None
        or int(nullable_legacy_fence_columns[0]) != 4
    ):
        raise ProofError("Research legacy fence projections are not nullable")


def _assert_schema_contract(
    database_url: str, expected: Mapping[str, Any]
) -> dict[str, Any]:
    with _connect(database_url) as connection:
        table_rows = connection.execute(
            """SELECT namespace.nspname,relation.relname
               FROM pg_class AS relation
               JOIN pg_namespace AS namespace ON namespace.oid=relation.relnamespace
               WHERE namespace.nspname NOT IN ('pg_catalog','information_schema')
                 AND namespace.nspname NOT LIKE 'pg_toast%'
                 AND relation.relkind IN ('r','p')
               ORDER BY 1,2"""
        ).fetchall()
        constraint_rows = connection.execute(
            """SELECT namespace.nspname,relation.relname,constraint_row.conname
               FROM pg_constraint AS constraint_row
               JOIN pg_class AS relation ON relation.oid=constraint_row.conrelid
               JOIN pg_namespace AS namespace ON namespace.oid=relation.relnamespace
               WHERE namespace.nspname NOT IN ('pg_catalog','information_schema')
               ORDER BY 1,2,3"""
        ).fetchall()
        function_rows = connection.execute(
            """SELECT namespace.nspname,function_row.proname
               FROM pg_proc AS function_row
               JOIN pg_namespace AS namespace ON namespace.oid=function_row.pronamespace
               WHERE namespace.nspname NOT IN ('pg_catalog','information_schema')
                 AND namespace.nspname NOT LIKE 'pg_toast%'
               ORDER BY 1,2"""
        ).fetchall()
        trigger_rows = connection.execute(
            """SELECT event_object_schema,event_object_table,trigger_name
               FROM information_schema.triggers
               WHERE event_object_schema NOT IN ('pg_catalog','information_schema')
               ORDER BY 1,2,3"""
        ).fetchall()
        policy_rows = connection.execute(
            """SELECT schemaname,tablename,policyname
               FROM pg_policies
               WHERE schemaname NOT IN ('pg_catalog','information_schema')
               ORDER BY 1,2,3"""
        ).fetchall()
        rls_rows = connection.execute(
            """SELECT namespace.nspname,relation.relname,
                      relation.relrowsecurity,relation.relforcerowsecurity
               FROM pg_class AS relation
               JOIN pg_namespace AS namespace ON namespace.oid=relation.relnamespace
               WHERE namespace.nspname NOT IN ('pg_catalog','information_schema')
                 AND relation.relkind IN ('r','p')
               ORDER BY 1,2"""
        ).fetchall()
    actual_tables = {f"{row[0]}.{row[1]}" for row in table_rows}
    expected_tables = set(map(str, expected["tables"]))
    if actual_tables != expected_tables:
        raise ProofError("PostgreSQL tables do not match the frozen migration contract")
    actual_constraints = {
        (f"{row[0]}.{row[1]}", str(row[2])) for row in constraint_rows
    }
    actual_functions = {f"{row[0]}.{row[1]}" for row in function_rows}
    actual_triggers = {(f"{row[0]}.{row[1]}", str(row[2])) for row in trigger_rows}
    actual_policies = {(f"{row[0]}.{row[1]}", str(row[2])) for row in policy_rows}
    actual_rls_enabled = {f"{row[0]}.{row[1]}" for row in rls_rows if row[2] is True}
    actual_rls_forced = {f"{row[0]}.{row[1]}" for row in rls_rows if row[3] is True}
    subset_checks = (
        (set(map(tuple, expected["constraints"])), actual_constraints, "constraints"),
        (set(map(str, expected["functions"])), actual_functions, "functions"),
        (set(map(tuple, expected["triggers"])), actual_triggers, "triggers"),
    )
    for required, actual, label in subset_checks:
        if not required.issubset(actual):
            raise ProofError(
                f"PostgreSQL {label} do not match the frozen migration contract"
            )
    expected_policies = set(map(tuple, expected["policies"]))
    missing_policies = expected_policies - actual_policies
    unexpected_policies = actual_policies - expected_policies
    if missing_policies or unexpected_policies:
        missing = ",".join(
            f"{relation}:{name}" for relation, name in sorted(missing_policies)
        )
        unexpected = ",".join(
            f"{relation}:{name}" for relation, name in sorted(unexpected_policies)
        )
        raise ProofError(
            "PostgreSQL policies do not match the frozen migration contract; "
            f"missing=[{missing}]; unexpected=[{unexpected}]"
        )
    if set(map(str, expected["rlsEnabled"])) != actual_rls_enabled:
        raise ProofError(
            "PostgreSQL RLS enablement does not match the migration contract"
        )
    if set(map(str, expected["rlsForced"])) != actual_rls_forced:
        raise ProofError("PostgreSQL forced RLS does not match the migration contract")
    return {
        "contractSha256": str(expected["sha256"]),
        "tableCount": len(actual_tables),
        "constraintCount": len(actual_constraints),
        "functionCount": len(actual_functions),
        "triggerCount": len(actual_triggers),
        "policyCount": len(actual_policies),
        "rlsEnabledCount": len(actual_rls_enabled),
        "rlsForcedCount": len(actual_rls_forced),
    }


def _prove_migrations(
    database_url: str,
    proof_root: Path,
    *,
    secrets_to_remove: Sequence[str],
) -> dict[str, Any]:
    snapshot_root, inventory = _stage_migrations(proof_root)
    source_inventory = list(inventory)
    names = [item["name"] for item in inventory]
    executed_inventory_sha256 = _sha256_text(_canonical_json(inventory))
    expected_pass = {
        "applied": len(inventory),
        "executedInventorySha256": executed_inventory_sha256,
    }
    schema_contract = _migration_schema_contract(snapshot_root)
    passes = []
    fingerprints = []
    for _ in range(2):
        if _migration_inventory(snapshot_root) != inventory:
            raise ProofError("PostgreSQL migration snapshot changed before execution")
        migration_pass = _run_migration_once(
            database_url,
            snapshot_root,
            names,
            secrets_to_remove,
        )
        if migration_pass != expected_pass:
            raise ProofError("PostgreSQL executed migration inventory evidence drifted")
        passes.append(migration_pass)
        if _migration_inventory(snapshot_root) != inventory:
            raise ProofError("PostgreSQL migration snapshot changed during execution")
        fingerprints.append(_schema_fingerprint(database_url))
    if fingerprints[0] != fingerprints[1]:
        raise ProofError("PostgreSQL migrations are not schema-idempotent")
    if _migration_inventory() != source_inventory:
        raise ProofError("PostgreSQL source migrations changed during the proof")
    _assert_relations(database_url, MIGRATION_RELATIONS)
    _assert_migration_constraints(database_url)
    contract_evidence = _assert_schema_contract(database_url, schema_contract)
    return {
        "inventory": inventory,
        "migrationCount": len(inventory),
        "executedInventorySha256": executed_inventory_sha256,
        "requiredMigrationHashes": dict(REQUIRED_MIGRATION_HASHES),
        "passes": passes,
        "schemaFingerprint": fingerprints[0],
        "schemaContract": contract_evidence,
        "immutableInventory": True,
        "idempotent": True,
    }


def _store(database_url: str):
    from baloot_platform.research.postgres_store import PostgresResearchStoreV1

    return PostgresResearchStoreV1(database_url)


def _bootstrap_research(
    database_url: str,
    *,
    workers: int = 8,
    expected_schema_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    barrier = threading.Barrier(workers)

    def bootstrap(_worker: int) -> None:
        _wait_at_research_barrier(
            barrier,
            stage="research.bootstrap.concurrent_store_initialization",
        )
        store = _store(database_url)
        store.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(bootstrap, range(workers)))
    with _connect(database_url) as connection:
        rows = connection.execute(
            """SELECT table_name FROM information_schema.tables
               WHERE table_schema='research' ORDER BY table_name"""
        ).fetchall()
    tables = {str(row[0]) for row in rows}
    missing = sorted(EXPECTED_RESEARCH_TABLES - tables)
    if missing:
        raise ProofError(f"Research bootstrap tables are incomplete: {missing}")
    fingerprint = _schema_fingerprint(database_url)
    if expected_schema_fingerprint is not None and fingerprint != dict(
        expected_schema_fingerprint
    ):
        raise ProofError("Research store bootstrap drifted from migration schema")
    return {
        "workers": workers,
        "tableCount": len(tables),
        "requiredTableCount": len(EXPECTED_RESEARCH_TABLES),
        "schemaFingerprint": fingerprint,
        "migrationEquivalent": expected_schema_fingerprint is not None,
    }


def _put_recipe_and_run(
    store: Any,
    *,
    suffix: str,
    status: str = "running",
) -> tuple[dict[str, Any], dict[str, Any]]:
    recipe = {
        "code": _proof_recipe_code(suffix),
        "sha256": _hash(f"recipe:{suffix}"),
        "status": "published",
    }
    run = {
        "runCode": f"RUN-PG-{suffix}",
        "runHash": _hash(f"run:{suffix}"),
        "recipeCode": recipe["code"],
        "recipeHash": recipe["sha256"],
        "status": status,
        "progress": 0.0,
        "events": [],
    }
    store.put_recipe(recipe)
    store.create_run(run)
    return recipe, run


def _create_shards(
    store: Any,
    *,
    recipe: Mapping[str, Any],
    run: Mapping[str, Any],
    count: int,
) -> list[dict[str, Any]]:
    from baloot_platform.research.sharding import partition_research_shards

    specs = partition_research_shards(
        run_code=str(run["runCode"]),
        run_hash=str(run["runHash"]),
        recipe_hash=str(recipe["sha256"]),
        seed_start=1,
        seed_count=count,
        seeds_per_shard=1,
    )
    return store.create_shards(tuple(item.to_dict() for item in specs))


def _prove_claim_concurrency(
    database_url: str,
    store: Any,
    *,
    suffix: str,
    workers: int = 8,
    shard_count: int = 64,
) -> dict[str, Any]:
    recipe, run = _put_recipe_and_run(store, suffix=suffix)
    _create_shards(store, recipe=recipe, run=run, count=shard_count)
    barrier = threading.Barrier(workers)

    def drain(worker_index: int) -> list[str]:
        claimed: list[str] = []
        worker_id = f"pg-proof-worker-{worker_index}"
        _wait_at_research_barrier(
            barrier,
            stage="research.claim_concurrency.drain",
        )
        while True:
            shard = store.claim_shard(
                str(run["runCode"]), worker_id=worker_id, lease_seconds=30
            )
            if shard is None:
                return claimed
            store.complete_shard(
                shard["shardId"],
                worker_id=worker_id,
                fencing_token=shard["fencingToken"],
                result={
                    "attemptedSeeds": shard["seedCount"],
                    "workerIndex": worker_index,
                },
            )
            claimed.append(str(shard["shardId"]))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        claims = [
            shard_id
            for worker_claims in executor.map(drain, range(workers))
            for shard_id in worker_claims
        ]
    shards = store.list_shards(str(run["runCode"]))
    if (
        len(claims) != shard_count
        or len(set(claims)) != shard_count
        or len(shards) != shard_count
        or any(item["status"] != "completed" for item in shards)
        or any(item["attemptCount"] != 1 for item in shards)
        or any(item["fencingToken"] != 1 for item in shards)
    ):
        raise ProofError("PostgreSQL concurrent shard claims were not exactly-once")
    return {
        "workers": workers,
        "workerIds": [f"pg-proof-worker-{index}" for index in range(workers)],
        "shards": shard_count,
        "uniqueClaims": len(set(claims)),
        "allCompleted": True,
    }


def _prove_fencing(store: Any, *, suffix: str) -> dict[str, Any]:
    recipe, run = _put_recipe_and_run(store, suffix=suffix)
    _create_shards(store, recipe=recipe, run=run, count=1)
    started = datetime.now(timezone.utc)
    first = store.claim_shard(
        str(run["runCode"]),
        worker_id="old-worker",
        lease_seconds=1,
        now=started,
    )
    second = store.claim_shard(
        str(run["runCode"]),
        worker_id="new-worker",
        lease_seconds=30,
        now=started + timedelta(seconds=2),
    )
    if first is None or second is None:
        raise ProofError("PostgreSQL fencing probe could not claim its shard")
    if (
        second["shardId"] != first["shardId"]
        or second["fencingToken"] != first["fencingToken"] + 1
        or second["reclaimCount"] != 1
    ):
        raise ProofError("PostgreSQL lease reclaim did not advance its fence")
    stale_rejected = False
    try:
        store.complete_shard(
            first["shardId"],
            worker_id="old-worker",
            fencing_token=first["fencingToken"],
            result={"stale": True},
        )
    except RuntimeError:
        stale_rejected = True
    if not stale_rejected:
        raise ProofError("PostgreSQL accepted a stale shard owner")
    completed = store.complete_shard(
        second["shardId"],
        worker_id="new-worker",
        fencing_token=second["fencingToken"],
        result={"reclaimed": True},
    )
    if completed["status"] != "completed":
        raise ProofError("PostgreSQL reclaimed shard did not complete")
    return {
        "staleOwnerRejected": True,
        "fencingTokenAdvanced": True,
        "reclaimCount": 1,
    }


def _prove_retry(store: Any, *, suffix: str) -> dict[str, Any]:
    recipe, run = _put_recipe_and_run(store, suffix=suffix)
    _create_shards(store, recipe=recipe, run=run, count=1)
    first = store.claim_shard(
        str(run["runCode"]), worker_id="retry-old", lease_seconds=30
    )
    if first is None:
        raise ProofError("PostgreSQL retry probe could not claim its shard")
    failed = store.fail_shard(
        first["shardId"],
        worker_id="retry-old",
        fencing_token=first["fencingToken"],
        error={"category": "bounded_probe"},
        retryable=True,
    )
    second = store.claim_shard(
        str(run["runCode"]), worker_id="retry-new", lease_seconds=30
    )
    if (
        failed["status"] != "queued"
        or failed["retryCount"] != 1
        or failed["reclaimCount"] != 0
        or second is None
        or second["fencingToken"] != first["fencingToken"] + 1
    ):
        raise ProofError("PostgreSQL retry accounting or fencing is invalid")
    store.complete_shard(
        second["shardId"],
        worker_id="retry-new",
        fencing_token=second["fencingToken"],
        result={"retried": True},
    )
    return {
        "retryCount": 1,
        "reclaimCount": 0,
        "fencingTokenAdvanced": True,
    }


def _prove_recipe_mutations(
    store: Any,
    *,
    suffix: str,
    published_recipe_labels: Sequence[str],
) -> dict[str, Any]:
    from baloot_platform.research.recipe_api import (
        RecipeListCursorV1,
        ResearchMutationConflictError,
    )

    recipe = {
        "code": _proof_recipe_code(suffix),
        "sha256": _hash(f"recipe-mutation:{suffix}"),
        "status": "published",
    }
    create_key = f"pg-proof-create-{suffix.lower()}"
    create_intent = _hash(f"create-intent:{suffix}")
    created = store.put_recipe_with_receipt(
        recipe,
        idempotency_key=create_key,
        request_intent_hash=create_intent,
    )
    replayed_create = store.put_recipe_with_receipt(
        recipe,
        idempotency_key=create_key,
        request_intent_hash=create_intent,
    )
    create_receipt = store.get_recipe_mutation_receipt(create_key)
    if (
        created != replayed_create
        or created.get("lifecycleStatus") != "published"
        or create_receipt is None
        or create_receipt.get("mutationKind") != "create"
        or create_receipt.get("responseStatus") != 201
    ):
        raise ProofError("PostgreSQL recipe creation receipt did not replay exactly")
    conflict_rejected = False
    try:
        store.put_recipe_with_receipt(
            recipe,
            idempotency_key=create_key,
            request_intent_hash=_hash(f"different-create-intent:{suffix}"),
        )
    except ResearchMutationConflictError:
        conflict_rejected = True
    if not conflict_rejected:
        raise ProofError("PostgreSQL recipe creation receipt accepted a new intent")

    invalid_cursor_rejected = False
    try:
        RecipeListCursorV1.decode("not-an-issued-cursor", status="published")
    except ValueError:
        invalid_cursor_rejected = True
    if not invalid_cursor_rejected:
        raise ProofError("PostgreSQL recipe cursor accepted malformed input")

    def exhaust_pages(status: str) -> list[str]:
        expected_order = [
            str(item["code"]) for item in store.list_recipes(status=status, limit=100)
        ]
        collected: list[str] = []
        anchor: dict[str, Any] | None = None
        anchors: set[tuple[str, str]] = set()
        while True:
            parameters: dict[str, Any] = {"status": status, "limit": 2}
            if anchor is not None:
                parameters.update(
                    {
                        "after_created_at": str(anchor["createdAt"]),
                        "after_recipe_code": str(anchor["recipeCode"]),
                    }
                )
            page = store.list_recipe_page(**parameters)
            page_codes = [str(item["recipe"]["code"]) for item in page["items"]]
            next_anchor = page["nextAnchor"]
            if not page_codes and next_anchor is not None:
                raise ProofError(
                    "PostgreSQL recipe keyset cursor advanced an empty page"
                )
            if set(collected) & set(page_codes) or len(page_codes) != len(
                set(page_codes)
            ):
                raise ProofError("PostgreSQL recipe keyset pages overlap")
            collected.extend(page_codes)
            if next_anchor is None:
                break
            issued_cursor = RecipeListCursorV1.from_anchor(
                next_anchor["createdAt"],
                next_anchor["recipeCode"],
                status=status,
            ).encode()
            decoded_cursor = RecipeListCursorV1.decode(issued_cursor, status=status)
            anchor = {
                "createdAt": decoded_cursor.created_at,
                "recipeCode": decoded_cursor.recipe_code,
            }
            anchor_key = (str(anchor["createdAt"]), str(anchor["recipeCode"]))
            if (
                not page_codes
                or anchor_key[1] != page_codes[-1]
                or anchor_key in anchors
            ):
                raise ProofError("PostgreSQL recipe keyset cursor order is invalid")
            anchors.add(anchor_key)
        if collected != expected_order:
            raise ProofError(
                "PostgreSQL recipe keyset pages are incomplete or misordered"
            )
        return collected

    expected_published = {
        *(_proof_recipe_code(label) for label in published_recipe_labels),
        str(recipe["code"]),
    }
    if len(expected_published) != len(PUBLISHED_RECIPE_STAGE_KEYS) + 1:
        raise ProofError("PostgreSQL published recipe inventory is ambiguous")
    published_before = exhaust_pages("published")
    if set(published_before) != expected_published or len(published_before) != 4:
        raise ProofError("PostgreSQL published recipe projection is incomplete")

    retire_key = f"pg-proof-retire-{suffix.lower()}"
    retire_intent = _hash(f"retire-intent:{suffix}")
    retired = store.retire_recipe_with_receipt(
        str(recipe["code"]),
        idempotency_key=retire_key,
        request_intent_hash=retire_intent,
    )
    replayed_retire = store.retire_recipe_with_receipt(
        str(recipe["code"]),
        idempotency_key=retire_key,
        request_intent_hash=retire_intent,
    )
    retire_conflict_rejected = False
    try:
        store.retire_recipe_with_receipt(
            str(recipe["code"]),
            idempotency_key=retire_key,
            request_intent_hash=_hash(f"different-retire-intent:{suffix}"),
        )
    except ResearchMutationConflictError:
        retire_conflict_rejected = True
    if not retire_conflict_rejected:
        raise ProofError("PostgreSQL recipe retirement receipt accepted a new intent")
    retire_receipt = store.get_recipe_mutation_receipt(retire_key)
    if (
        retired != replayed_retire
        or retired.get("lifecycleStatus") != "retired"
        or store.recipe_status(str(recipe["code"])) != "retired"
        or retire_receipt is None
        or retire_receipt.get("mutationKind") != "retire"
        or retire_receipt.get("responseStatus") != 200
    ):
        raise ProofError("PostgreSQL recipe retirement receipt did not replay exactly")
    published_after = exhaust_pages("published")
    retired_after = exhaust_pages("retired")
    if (
        set(published_after) != expected_published - {str(recipe["code"])}
        or str(recipe["code"]) in published_after
        or retired_after != [str(recipe["code"])]
    ):
        raise ProofError("PostgreSQL recipe lifecycle listings are inconsistent")
    return {
        "createReceiptReplayed": True,
        "conflictingIntentRejected": True,
        "retireReceiptReplayed": True,
        "retireConflictingIntentRejected": True,
        "invalidCursorRejected": True,
        "stableKeysetPagination": True,
        "lifecycleListingsVerified": True,
    }


def _prove_admission_and_statistics(store: Any, *, suffix: str) -> dict[str, Any]:
    recipe, run = _put_recipe_and_run(store, suffix=suffix)
    _create_shards(store, recipe=recipe, run=run, count=1)
    lease = store.claim_shard(
        str(run["runCode"]), worker_id="admission-worker", lease_seconds=30
    )
    if lease is None:
        raise ProofError("PostgreSQL admission probe could not claim its shard")
    result = {
        "resultCode": f"LAF-PG-{suffix}",
        "resultHash": _hash(f"result:{suffix}"),
        "runCode": run["runCode"],
        "truthLabel": "robust_best_known",
        "confidence": 0.9,
        "score": 88.0,
        "disposition": "qualified",
        "provenance": {"researchShardId": lease["shardId"]},
    }
    store.put_result(
        result,
        attempt_count=int(lease["attemptCount"]),
        fencing_token=int(lease["fencingToken"]),
        worker_id="admission-worker",
    )
    hidden = store.list_results(run_code=str(run["runCode"]))
    hidden_stats = store.aggregate_stats(run_code=str(run["runCode"]))
    if (
        store.get_result(str(result["resultCode"])) is not None
        or hidden["total"] != 0
        or hidden_stats["resultCount"] != 0
    ):
        raise ProofError("PostgreSQL admitted an incomplete shard result")
    store.complete_shard(
        lease["shardId"],
        worker_id="admission-worker",
        fencing_token=lease["fencingToken"],
        result={"attemptedSeeds": 1},
    )
    visible = store.list_results(run_code=str(run["runCode"]))
    stats = store.aggregate_stats(run_code=str(run["runCode"]))
    if (
        store.get_result(str(result["resultCode"])) != result
        or visible["total"] != 1
        or stats["runStatus"] != {"running": 1}
        or stats["resultCount"] != 1
        or stats["resultDisposition"] != {"qualified": 1}
        or stats["semanticDefects"] != {}
    ):
        raise ProofError("PostgreSQL admission/statistics projection is inconsistent")
    return {
        "hiddenBeforeCompletion": True,
        "visibleAfterCompletion": True,
        "qualifiedResults": 1,
        "semanticDefects": 0,
    }


def _empty_sequential_search_receipt(seed: int) -> dict[str, Any]:
    from baloot_platform.challenge.constraint_deal_search import (
        DealConstraintSpecV1,
        OrToolsCpSatDealGeneratorV1,
    )
    from baloot_platform.config import build_bundle
    from baloot_platform.core.enums import Seat
    from baloot_platform.research.sequential_candidate_search import (
        run_sequential_candidate_search_v1,
    )
    import ortools

    rules = build_bundle().rules
    spec = DealConstraintSpecV1.build(
        {
            "schemaVersion": 1,
            "id": "postgres_receipt_proof_v1",
            "revision": "1",
            "enabled": True,
            "backend": "ortools_cp_sat_v1",
            "assumedBuyer": None,
            "purchaseSuits": [],
            "initialRequiredCards": {},
            "finalRequiredCards": {},
            "finalForbiddenCards": {},
            "initialSuitCountRanges": {},
            "finalSuitCountRanges": {},
            "projectPattern": None,
            "maximumSolutions": 2,
            "maximumConflicts": 1000,
            "safetyTimeoutSeconds": 1.0,
            "solverWorkers": 1,
            "seedOffset": 0,
            "tags": ["engine_validation_required"],
        }
    )

    class EmptyGenerator:
        descriptor = OrToolsCpSatDealGeneratorV1.descriptor

        @staticmethod
        def generate_next(*_args: Any, **_kwargs: Any) -> None:
            return None

        @staticmethod
        def probe_continuation(
            selected_spec: Any,
            *,
            root_seed: int,
            dealer: Any,
            rules: Any,
            rejected_candidates: Sequence[Any],
            **_kwargs: Any,
        ) -> Any:
            if rejected_candidates:
                raise ProofError(
                    "Empty receipt proof unexpectedly rejected a candidate"
                )
            model_hash = _hash("postgres-receipt-proof-model")
            backend_version = str(ortools.__version__)
            no_good_cut_type = "complete_deck_assignment_v1"
            search_identity = OrToolsCpSatDealGeneratorV1.search_identity_document(
                spec=selected_spec,
                model_hash=model_hash,
                root_seed=root_seed,
                dealer=dealer,
                rules=rules,
                backend_version=backend_version,
                no_good_cut_type=no_good_cut_type,
            )
            return OrToolsCpSatDealGeneratorV1._build_continuation_receipt(
                profile_hash=selected_spec.sha256,
                model_hash=model_hash,
                backend_version=backend_version,
                solver_status="infeasible",
                solution_ordinal=0,
                search_identity_hash=search_identity["sha256"],
                preceding_no_good_cut_hashes=(),
                no_good_cut_type=no_good_cut_type,
                exhausted=True,
                witness=None,
                structural_objective_score=0,
            )

    result = run_sequential_candidate_search_v1(
        EmptyGenerator(),
        spec,
        root_seed=int(seed),
        dealer=Seat.SOUTH,
        rules=rules,
        evaluate_candidate=lambda *_args: (_ for _ in ()).throw(
            ProofError("Empty receipt proof evaluated a candidate")
        ),
    )
    return result.receipt.verify().to_dict()


def _prove_fenced_search_receipts(
    database_url: str,
    store: Any,
    *,
    suffix: str,
) -> dict[str, Any]:
    recipe, run = _put_recipe_and_run(store, suffix=suffix)
    shards = _create_shards(store, recipe=recipe, run=run, count=3)
    if [(int(shard["seedStart"]), int(shard["seedCount"])) for shard in shards] != [
        (1, 1),
        (2, 1),
        (3, 1),
    ]:
        raise ProofError(
            "PostgreSQL receipt proof shards are not three one-seed shards"
        )
    receipts = {seed: _empty_sequential_search_receipt(seed) for seed in (1, 2, 3)}
    first = store.claim_shard(
        str(run["runCode"]), worker_id="receipt-worker-1", lease_seconds=30
    )
    if first is None or int(first["seedStart"]) != 1:
        raise ProofError("PostgreSQL receipt proof could not claim attempt one")
    result = {
        "resultCode": f"LAF-PG-{suffix}",
        "resultHash": _hash(f"receipt-result:{suffix}"),
        "runCode": run["runCode"],
        "truthLabel": "provisional",
        "confidence": 0.5,
        "score": 50.0,
        "disposition": "rejected",
        "provenance": {"researchShardId": first["shardId"]},
    }
    disagreement = {
        "disagreementId": f"DSG-PG-{suffix}",
        "sha256": _hash(f"receipt-disagreement:{suffix}"),
        "runCode": run["runCode"],
        "category": "compiler_parity_defect",
        "status": "open",
    }

    def put_attempt(
        lease: Mapping[str, Any],
        worker_id: str,
        *,
        seed: int,
        include_semantic_rows: bool,
        attempt_store: Any | None = None,
    ) -> dict[str, Any]:
        local = _store(database_url) if attempt_store is None else attempt_store
        try:
            retained_receipt = local.put_sequential_search_receipt(
                receipts[seed],
                run_code=str(run["runCode"]),
                shard_id=str(lease["shardId"]),
                attempt_count=int(lease["attemptCount"]),
                fencing_token=int(lease["fencingToken"]),
                worker_id=worker_id,
                seed=seed,
            )
            if include_semantic_rows:
                local.put_result(
                    result,
                    attempt_count=int(lease["attemptCount"]),
                    fencing_token=int(lease["fencingToken"]),
                    worker_id=worker_id,
                )
                local.put_disagreement(
                    disagreement,
                    shard_id=str(lease["shardId"]),
                    attempt_count=int(lease["attemptCount"]),
                    fencing_token=int(lease["fencingToken"]),
                    worker_id=worker_id,
                )
            return retained_receipt
        finally:
            if attempt_store is None:
                local.close()

    barrier = threading.Barrier(2)

    def concurrent_first(_index: int) -> dict[str, Any]:
        _wait_at_research_barrier(
            barrier,
            stage="research.fenced_search_receipts.concurrent_cas",
        )
        return put_attempt(
            first,
            "receipt-worker-1",
            seed=1,
            include_semantic_rows=True,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        concurrent_rows = list(executor.map(concurrent_first, range(2)))
    if concurrent_rows[0] != concurrent_rows[1]:
        raise ProofError("PostgreSQL concurrent receipt CAS replay diverged")

    with _connect(database_url) as connection:
        base_result_before = connection.execute(
            """SELECT payload_json,attempt_count,fencing_token
               FROM research.results WHERE result_code=%s""",
            (result["resultCode"],),
        ).fetchone()
        base_disagreement_before = connection.execute(
            """SELECT payload_json,attempt_count,fencing_token
               FROM research.semantic_disagreements WHERE disagreement_id=%s""",
            (disagreement["disagreementId"],),
        ).fetchone()
    if base_result_before is None or base_disagreement_before is None:
        raise ProofError("PostgreSQL receipt proof lost its semantic base rows")

    store.fail_shard(
        first["shardId"],
        worker_id="receipt-worker-1",
        fencing_token=first["fencingToken"],
        error={"category": "deterministic_retry"},
        retryable=True,
    )
    retry = store.claim_shard(
        str(run["runCode"]), worker_id="receipt-worker-2", lease_seconds=30
    )
    if (
        retry is None
        or retry["shardId"] != first["shardId"]
        or int(retry["seedStart"]) != 1
        or int(retry["attemptCount"]) != 2
        or int(retry["fencingToken"]) != 2
    ):
        raise ProofError("PostgreSQL receipt proof could not claim attempt two")

    stale_writes_rejected = 0
    stale_operations = (
        lambda: store.put_sequential_search_receipt(
            receipts[1],
            run_code=str(run["runCode"]),
            shard_id=str(first["shardId"]),
            attempt_count=int(first["attemptCount"]),
            fencing_token=int(first["fencingToken"]),
            worker_id="receipt-worker-1",
            seed=1,
        ),
        lambda: store.put_result(
            result,
            attempt_count=int(first["attemptCount"]),
            fencing_token=int(first["fencingToken"]),
            worker_id="receipt-worker-1",
        ),
        lambda: store.put_disagreement(
            disagreement,
            shard_id=str(first["shardId"]),
            attempt_count=int(first["attemptCount"]),
            fencing_token=int(first["fencingToken"]),
            worker_id="receipt-worker-1",
        ),
    )
    for operation in stale_operations:
        try:
            operation()
        except RuntimeError as exc:
            if str(exc) != "stale research shard occurrence admission":
                raise ProofError(
                    "PostgreSQL stale receipt write failed for an unexpected reason"
                ) from exc
            stale_writes_rejected += 1
    if stale_writes_rejected != len(stale_operations):
        raise ProofError("PostgreSQL receipt proof accepted a stale attempt write")

    put_attempt(
        retry,
        "receipt-worker-2",
        seed=1,
        include_semantic_rows=True,
    )
    with _connect(database_url) as connection:
        connection.execute(
            """UPDATE research.shards
               SET lease_expires_at=clock_timestamp()-interval '1 second'
               WHERE shard_id=%s""",
            (retry["shardId"],),
        )
    expired_heartbeat_rejected = False
    expired_completion_rejected = False
    for operation, expected_error in (
        (
            lambda: store.heartbeat_shard(
                retry["shardId"],
                worker_id="receipt-worker-2",
                fencing_token=retry["fencingToken"],
                lease_seconds=30,
            ),
            "stale research shard lease",
        ),
        (
            lambda: store.complete_shard(
                retry["shardId"],
                worker_id="receipt-worker-2",
                fencing_token=retry["fencingToken"],
                result={"attemptedSeeds": 1},
            ),
            "stale research shard completion",
        ),
    ):
        try:
            operation()
        except RuntimeError as exc:
            if str(exc) != expected_error:
                raise ProofError(
                    "PostgreSQL expired receipt operation failed for an unexpected reason"
                ) from exc
            if expected_error == "stale research shard lease":
                expired_heartbeat_rejected = True
            else:
                expired_completion_rejected = True
            continue
        raise ProofError("PostgreSQL expired receipt lease was renewed or admitted")

    admitted_retry = store.claim_shard(
        str(run["runCode"]), worker_id="receipt-worker-3", lease_seconds=30
    )
    if (
        admitted_retry is None
        or admitted_retry["shardId"] != first["shardId"]
        or int(admitted_retry["seedStart"]) != 1
        or int(admitted_retry["attemptCount"]) != 3
        or int(admitted_retry["fencingToken"]) != 3
    ):
        raise ProofError("PostgreSQL expired receipt attempt was not reclaimed exactly")
    admitted_retry_receipt = put_attempt(
        admitted_retry,
        "receipt-worker-3",
        seed=1,
        include_semantic_rows=True,
    )
    store.complete_shard(
        admitted_retry["shardId"],
        worker_id="receipt-worker-3",
        fencing_token=admitted_retry["fencingToken"],
        result={"attemptedSeeds": 1},
    )

    seed_two = store.claim_shard(
        str(run["runCode"]), worker_id="receipt-worker-4", lease_seconds=30
    )
    if (
        seed_two is None
        or int(seed_two["seedStart"]) != 2
        or int(seed_two["attemptCount"]) != 1
        or int(seed_two["fencingToken"]) != 1
    ):
        raise ProofError("PostgreSQL receipt proof could not claim seed two")
    seed_two_receipt = put_attempt(
        seed_two,
        "receipt-worker-4",
        seed=2,
        include_semantic_rows=False,
    )
    store.complete_shard(
        seed_two["shardId"],
        worker_id="receipt-worker-4",
        fencing_token=seed_two["fencingToken"],
        result={"attemptedSeeds": 1},
    )

    seed_three = store.claim_shard(
        str(run["runCode"]), worker_id="receipt-worker-5", lease_seconds=30
    )
    if (
        seed_three is None
        or int(seed_three["seedStart"]) != 3
        or int(seed_three["attemptCount"]) != 1
        or int(seed_three["fencingToken"]) != 1
    ):
        raise ProofError("PostgreSQL receipt proof could not claim seed three")

    snapshot_writer_store = _store(database_url)

    def admit_seed_three() -> dict[str, Any]:
        retained = put_attempt(
            seed_three,
            "receipt-worker-5",
            seed=3,
            include_semantic_rows=False,
            attempt_store=snapshot_writer_store,
        )
        completed = snapshot_writer_store.complete_shard(
            seed_three["shardId"],
            worker_id="receipt-worker-5",
            fencing_token=seed_three["fencingToken"],
            result={"attemptedSeeds": 1},
        )
        return {"retained": retained, "completed": completed}

    snapshot_batch_size = 1
    fixed_snapshot_seeds: list[int] = []
    concurrent_write: dict[str, Any] | None = None
    writer_completed_while_snapshot_open = False
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future: concurrent.futures.Future[dict[str, Any]] | None = None
    try:
        with store.iter_admitted_sequential_search_receipts(
            run_code=str(run["runCode"]), batch_size=snapshot_batch_size
        ) as snapshot_rows:
            try:
                fixed_snapshot_seeds.append(int(next(snapshot_rows)["seed"]))
            except StopIteration as exc:
                raise ProofError("PostgreSQL fixed receipt snapshot was empty") from exc
            future = executor.submit(admit_seed_three)
            try:
                concurrent_write = future.result(timeout=30)
            except concurrent.futures.TimeoutError as exc:
                raise ProofError(
                    "PostgreSQL concurrent receipt writer exceeded its bounded timeout"
                ) from exc
            completed_shard = concurrent_write.get("completed")
            writer_completed_while_snapshot_open = (
                future.done()
                and isinstance(completed_shard, Mapping)
                and completed_shard.get("status") == "completed"
                and completed_shard.get("shardId") == seed_three["shardId"]
            )
            fixed_snapshot_seeds.extend(int(item["seed"]) for item in snapshot_rows)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        if future is None or future.done():
            snapshot_writer_store.close()
        else:
            future.add_done_callback(lambda _future: snapshot_writer_store.close())
    if concurrent_write is None:
        raise ProofError("PostgreSQL concurrent receipt writer did not return evidence")

    with store.iter_admitted_sequential_search_receipts(
        run_code=str(run["runCode"]), batch_size=snapshot_batch_size
    ) as snapshot_rows:
        fresh_snapshot_seeds = [int(item["seed"]) for item in snapshot_rows]

    all_receipts = store.list_sequential_search_receipts(
        run_code=str(run["runCode"]), admitted_only=False
    )
    admitted_receipts = store.list_sequential_search_receipts(
        run_code=str(run["runCode"])
    )
    with _connect(database_url) as connection:
        base_result_after = connection.execute(
            """SELECT payload_json,attempt_count,fencing_token
               FROM research.results WHERE result_code=%s""",
            (result["resultCode"],),
        ).fetchone()
        base_disagreement_after = connection.execute(
            """SELECT payload_json,attempt_count,fencing_token
               FROM research.semantic_disagreements WHERE disagreement_id=%s""",
            (disagreement["disagreementId"],),
        ).fetchone()
        receipt_occurrences = connection.execute(
            """SELECT seed,attempt_count,fencing_token,admitted
               FROM research.sequential_search_receipt_occurrences
               WHERE run_code=%s
               ORDER BY seed,attempt_count,fencing_token,occurrence_hash""",
            (run["runCode"],),
        ).fetchall()
        result_occurrences = connection.execute(
            """SELECT attempt_count,fencing_token,admitted
               FROM research.result_occurrences
               WHERE run_code=%s ORDER BY attempt_count""",
            (run["runCode"],),
        ).fetchall()
        disagreement_occurrences = connection.execute(
            """SELECT attempt_count,fencing_token,admitted
               FROM research.semantic_disagreement_occurrences
               WHERE run_code=%s ORDER BY attempt_count""",
            (run["runCode"],),
        ).fetchall()
        bundle_rows = [
            connection.execute(
                """SELECT seed,receipt_json
                   FROM research.sequential_search_receipt_bundles
                   WHERE receipt_hash=%s AND seed=%s""",
                (receipts[seed]["sha256"], seed),
            ).fetchone()
            for seed in (1, 2, 3)
        ]
    expected_receipt_occurrences = [
        (1, 1, 1, False),
        (1, 2, 2, False),
        (1, 3, 3, True),
        (2, 1, 1, True),
        (3, 1, 1, True),
    ]
    expected_semantic_occurrences = [
        (1, 1, False),
        (2, 2, False),
        (3, 3, True),
    ]
    if (
        [tuple(row) for row in receipt_occurrences] != expected_receipt_occurrences
        or [tuple(row) for row in result_occurrences] != expected_semantic_occurrences
        or [tuple(row) for row in disagreement_occurrences]
        != expected_semantic_occurrences
    ):
        raise ProofError(
            "PostgreSQL receipt admission did not retain exact occurrences"
        )
    if (
        tuple(base_result_before) != tuple(base_result_after or ())
        or tuple(base_disagreement_before) != tuple(base_disagreement_after or ())
        or int(base_result_after[1]) != 1
        or int(base_result_after[2]) != 1
        or int(base_disagreement_after[1]) != 1
        or int(base_disagreement_after[2]) != 1
    ):
        raise ProofError("PostgreSQL deterministic retry rewrote a semantic base row")
    final_occurrence_count = store.count_sequential_search_receipts(
        run_code=str(run["runCode"]), admitted_only=False
    )
    final_admitted_count = store.count_sequential_search_receipts(
        run_code=str(run["runCode"])
    )
    expected_bundles = [(seed, receipts[seed]) for seed in (1, 2, 3)]
    if (
        any(row is None for row in bundle_rows)
        or [tuple(row) for row in bundle_rows] != expected_bundles
        or all_receipts["total"] != final_occurrence_count
        or admitted_receipts["total"] != final_admitted_count
        or len(admitted_receipts["items"]) != final_admitted_count
    ):
        raise ProofError("PostgreSQL receipt CAS/list/count projection is inconsistent")
    admitted_by_seed = {int(item["seed"]): item for item in admitted_receipts["items"]}
    if set(admitted_by_seed) != {1, 2, 3}:
        raise ProofError("PostgreSQL admitted receipt seed projection drifted")
    for seed, staged in (
        (1, admitted_retry_receipt),
        (2, seed_two_receipt),
        (3, concurrent_write["retained"]),
    ):
        _assert_receipt_admission_transition(staged, admitted_by_seed[seed])
    snapshot_evidence = _assert_receipt_snapshot_choreography(
        snapshot_batch_size=snapshot_batch_size,
        fixed_snapshot_seeds=fixed_snapshot_seeds,
        concurrent_admitted_seed=3,
        fresh_snapshot_seeds=fresh_snapshot_seeds,
        writer_completed_while_snapshot_open=writer_completed_while_snapshot_open,
        final_occurrence_count=final_occurrence_count,
        final_admitted_count=final_admitted_count,
    )
    return {
        "concurrentCasCallers": 2,
        "semanticBundleCount": 3,
        "staleWritesRejected": stale_writes_rejected,
        "expiredHeartbeatRejected": expired_heartbeat_rejected,
        "expiredCompletionRejected": expired_completion_rejected,
        "deterministicRetryAdmitted": True,
        "baseResultUnchanged": True,
        "baseDisagreementUnchanged": True,
        **snapshot_evidence,
    }


def _prove_run_lock_serialization(
    database_url: str,
    store: Any,
    *,
    suffix: str,
) -> dict[str, Any]:
    recipe, run = _put_recipe_and_run(store, suffix=suffix)
    disagreement = {
        "disagreementId": f"DSG-PG-{suffix}",
        "sha256": _hash(f"run-lock-disagreement:{suffix}"),
        "runCode": run["runCode"],
        "category": "compiler_parity_defect",
        "status": "open",
    }

    def unsharded_write() -> dict[str, Any]:
        local = _store(database_url)
        try:
            return local.put_disagreement(disagreement)
        finally:
            local.close()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        with _connect(database_url) as blocker:
            blocker.execute(
                "SELECT run_code FROM research.runs WHERE run_code=%s FOR UPDATE",
                (run["runCode"],),
            ).fetchone()
            unsharded_future = executor.submit(unsharded_write)
            time.sleep(0.2)
            unsharded_waited = not unsharded_future.done()
        if unsharded_future.result(timeout=30) != disagreement:
            raise ProofError("PostgreSQL serialized unsharded write changed payload")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if not unsharded_waited:
        raise ProofError("PostgreSQL unsharded write bypassed the run lock")

    from baloot_platform.research.sharding import partition_research_shards

    shard_spec = partition_research_shards(
        run_code=str(run["runCode"]),
        run_hash=str(run["runHash"]),
        recipe_hash=str(recipe["sha256"]),
        seed_start=1,
        seed_count=1,
        seeds_per_shard=1,
    )[0].to_dict()

    def create_shard() -> list[dict[str, Any]]:
        local = _store(database_url)
        try:
            return local.create_shards((shard_spec,))
        finally:
            local.close()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        with _connect(database_url) as blocker:
            blocker.execute(
                "SELECT run_code FROM research.runs WHERE run_code=%s FOR UPDATE",
                (run["runCode"],),
            ).fetchone()
            shard_future = executor.submit(create_shard)
            time.sleep(0.2)
            shard_waited = not shard_future.done()
        shard_rows = shard_future.result(timeout=30)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if not shard_waited or len(shard_rows) != 1:
        raise ProofError("PostgreSQL shard creation bypassed the run lock")

    rejected_after_sharding = False
    try:
        store.put_disagreement(
            {
                **disagreement,
                "disagreementId": f"{disagreement['disagreementId']}-AFTER",
                "sha256": _hash(f"run-lock-disagreement-after:{suffix}"),
            }
        )
    except ValueError:
        rejected_after_sharding = True
    if not rejected_after_sharding:
        raise ProofError("PostgreSQL unsharded write raced past shard creation")
    return {
        "unshardedWriteWaitedForRunLock": True,
        "shardCreationWaitedForRunLock": True,
        "unshardedWriteRejectedAfterSharding": True,
    }


def _artifact_payload(suffix: str, *, artifact_label: str) -> dict[str, Any]:
    artifact_hash = _hash(f"artifact:{artifact_label}")
    return {
        "schemaVersion": 1,
        "compilationHash": _hash(f"compilation:{suffix}"),
        "artifact": {
            "uri": f"cas://sha256/{artifact_hash}",
            "sha256": artifact_hash,
            "bytes": 1,
            "mediaType": "application/vnd.balootx.domain-compilation-v3+json",
            "name": "postgres-proof.json",
        },
        "bundleHash": _hash(f"bundle:{suffix}"),
        "effectiveRuntimeConfigHash": _hash(f"runtime:{suffix}"),
        "contextHash": _hash(f"context:{suffix}"),
        "domainHash": _hash(f"domain:{suffix}"),
        "createdAt": None,
    }


def _prove_artifact_serialization(store: Any, *, suffix: str) -> dict[str, Any]:
    payload = _artifact_payload(suffix, artifact_label="original")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        rows = list(
            executor.map(
                lambda _index: store.put_domain_compilation_artifact(payload),
                range(8),
            )
        )
    encodings = {_canonical_json(row) for row in rows}
    if len(encodings) != 1:
        raise ProofError("PostgreSQL serialized artifact writes are not deterministic")
    original = store.get_domain_compilation_artifact(payload["compilationHash"])
    conflict = _artifact_payload(suffix, artifact_label="conflict")
    conflict_rejected = False
    try:
        store.put_domain_compilation_artifact(conflict)
    except ValueError:
        conflict_rejected = True
    if (
        not conflict_rejected
        or original is None
        or store.get_domain_compilation_artifact(payload["compilationHash"]) != original
    ):
        raise ProofError("PostgreSQL artifact identity rollback failed")
    return {
        "concurrentWriters": 8,
        "storedRows": 1,
        "conflictingIdentityRejected": True,
        "originalPreserved": True,
    }


def _prove_research(
    database_url: str,
    proof_id: str,
    *,
    expected_schema_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    bootstrap = _bootstrap_research(
        database_url,
        expected_schema_fingerprint=expected_schema_fingerprint,
    )
    store = _store(database_url)
    root_suffix = proof_id[:10].upper()
    stage_labels = _research_proof_stage_labels(root_suffix)
    published_recipe_labels = tuple(
        stage_labels[stage] for stage in PUBLISHED_RECIPE_STAGE_KEYS
    )
    try:
        stage_results = {
            "concurrency": _prove_claim_concurrency(
                database_url, store, suffix=stage_labels["concurrency"]
            ),
            "fencing": _prove_fencing(store, suffix=stage_labels["fencing"]),
            "retry": _prove_retry(store, suffix=stage_labels["retry"]),
            "recipeMutations": _prove_recipe_mutations(
                store,
                suffix=stage_labels["recipeMutations"],
                published_recipe_labels=published_recipe_labels,
            ),
            "admissionAndStatistics": _prove_admission_and_statistics(
                store, suffix=stage_labels["admissionAndStatistics"]
            ),
            "fencedSearchReceipts": _prove_fenced_search_receipts(
                database_url,
                store,
                suffix=stage_labels["fencedSearchReceipts"],
            ),
            "runLockSerialization": _prove_run_lock_serialization(
                database_url,
                store,
                suffix=stage_labels["runLockSerialization"],
            ),
            "artifactSerialization": _prove_artifact_serialization(
                store, suffix=stage_labels["artifactSerialization"]
            ),
        }
        if tuple(stage_results) != RESEARCH_PROOF_STAGE_KEYS:
            raise ProofError("PostgreSQL research proof stage inventory drifted")
        return {
            "bootstrap": bootstrap,
            "stageInventory": [
                {"stage": stage, "label": stage_labels[stage]}
                for stage in RESEARCH_PROOF_STAGE_KEYS
            ],
            **stage_results,
        }
    finally:
        store.close()


@contextlib.contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update({key: str(value) for key, value in values.items()})
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _prove_configuration_v25_schema(database_url: str) -> dict[str, Any]:
    with _connect(database_url) as connection:
        rollout_rows = connection.execute(
            """SELECT table_name,data_type
               FROM information_schema.columns
               WHERE table_schema='configuration'
                 AND table_name IN ('profile_revisions','bundle_revisions')
                 AND column_name='rollout_percentage'
               ORDER BY table_name"""
        ).fetchall()
        constraint_rows = connection.execute(
            """SELECT relation.relname,constraint_row.conname,
                      constraint_row.convalidated
               FROM pg_constraint AS constraint_row
               JOIN pg_class AS relation
                 ON relation.oid=constraint_row.conrelid
               JOIN pg_namespace AS namespace
                 ON namespace.oid=relation.relnamespace
               WHERE namespace.nspname='configuration'
                 AND constraint_row.conname LIKE 'configuration_%_v25'
               ORDER BY 1,2"""
        ).fetchall()
        trigger_rows = connection.execute(
            """SELECT relation.relname,trigger_row.tgname,
                      function_row.proname,trigger_row.tgenabled
               FROM pg_trigger AS trigger_row
               JOIN pg_class AS relation ON relation.oid=trigger_row.tgrelid
               JOIN pg_namespace AS namespace
                 ON namespace.oid=relation.relnamespace
               JOIN pg_proc AS function_row
                 ON function_row.oid=trigger_row.tgfoid
               WHERE namespace.nspname='configuration'
                 AND NOT trigger_row.tgisinternal
                 AND trigger_row.tgname LIKE 'trg_configuration_%_immutable'
               ORDER BY 1,2"""
        ).fetchall()
        policy_rows = connection.execute(
            """SELECT tablename,policyname,cmd,permissive
               FROM pg_policies
               WHERE schemaname='configuration'
                 AND tablename IN (
                   'profile_revisions','bundle_revisions','events'
                 )
               ORDER BY 1,2"""
        ).fetchall()
        rls_rows = connection.execute(
            """SELECT relation.relname,relation.relrowsecurity,
                      relation.relforcerowsecurity
               FROM pg_class AS relation
               JOIN pg_namespace AS namespace
                 ON namespace.oid=relation.relnamespace
               WHERE namespace.nspname='configuration'
                 AND relation.relname IN (
                   'profile_revisions','bundle_revisions','events'
                 )
               ORDER BY 1"""
        ).fetchall()
        invalidation_row = connection.execute(
            """SELECT epoch FROM configuration.invalidation_epochs
               WHERE namespace='configuration'"""
        ).fetchone()
    if {(str(row[0]), str(row[1])) for row in rollout_rows} != {
        ("bundle_revisions", "numeric"),
        ("profile_revisions", "numeric"),
    }:
        raise ProofError("configuration v25 rollout column contract drifted")
    if {
        (str(row[0]), str(row[1])) for row in constraint_rows if row[2] is True
    } != CONFIGURATION_V25_CONSTRAINTS or any(
        row[2] is not True for row in constraint_rows
    ):
        raise ProofError("configuration v25 validated constraints drifted")
    if {
        (str(row[0]), str(row[1]), str(row[2]), str(row[3])) for row in trigger_rows
    } != CONFIGURATION_V25_TRIGGERS:
        raise ProofError("configuration v25 immutable triggers drifted")
    if {
        (str(row[0]), str(row[1]), str(row[2])) for row in policy_rows
    } != CONFIGURATION_V25_POLICIES or any(
        str(row[3]).upper() != "PERMISSIVE" for row in policy_rows
    ):
        raise ProofError("configuration v25 RLS policies drifted")
    if {
        str(row[0]) for row in rls_rows if row[1] is True and row[2] is True
    } != CONFIGURATION_V25_RLS_RELATIONS or any(
        row[1] is not True or row[2] is not True for row in rls_rows
    ):
        raise ProofError("configuration v25 forced RLS contract drifted")
    if invalidation_row is None or int(invalidation_row[0]) < 1:
        raise ProofError("configuration v25 invalidation epoch is missing")
    contract = {
        "constraints": [list(value) for value in sorted(CONFIGURATION_V25_CONSTRAINTS)],
        "policies": [list(value) for value in sorted(CONFIGURATION_V25_POLICIES)],
        "rlsRelations": sorted(CONFIGURATION_V25_RLS_RELATIONS),
        "triggers": [list(value) for value in sorted(CONFIGURATION_V25_TRIGGERS)],
    }
    return {
        "contractSha256": _sha256_text(_canonical_json(contract)),
        "rolloutPercentageNumeric": True,
        "validatedConstraintCount": len(CONFIGURATION_V25_CONSTRAINTS),
        "immutabilityTriggerCount": len(CONFIGURATION_V25_TRIGGERS),
        "policyCount": len(CONFIGURATION_V25_POLICIES),
        "forcedRlsRelationCount": len(CONFIGURATION_V25_RLS_RELATIONS),
        "invalidationEpochPresent": True,
    }


def _configuration_event_chain_evidence(
    value: Mapping[str, Any], *, aggregate_type: str, aggregate_id: str
) -> dict[str, Any]:
    if set(value) != {"items", "verification"}:
        raise ProofError("configuration event-chain projection drifted")
    items = list(value["items"])
    verification = dict(value["verification"])
    if set(verification) != {"valid", "issues", "eventCount", "eventHead"}:
        raise ProofError("configuration event-chain verification shape drifted")
    if (
        verification["valid"] is not True
        or list(verification["issues"]) != []
        or int(verification["eventCount"]) != len(items)
        or not items
    ):
        raise ProofError("configuration event chain is invalid")
    if any(
        str(item.get("aggregateType")) != aggregate_type
        or str(item.get("aggregateId")) != aggregate_id
        for item in items
    ):
        raise ProofError("configuration event chain aggregate identity drifted")
    event_head = str(verification["eventHead"] or "")
    if len(event_head) != 64 or any(
        character not in "0123456789abcdef" for character in event_head
    ):
        raise ProofError("configuration event-chain head is invalid")
    return {"eventCount": len(items), "eventHead": event_head}


def _assert_configuration_immutability(
    pool: Any, *, relation: str, revision_id: str, expected_message: str
) -> None:
    from baloot_platform.storage.request_context import contextual_connection

    succeeded = False
    try:
        with contextual_connection(pool) as connection:
            with connection.transaction():
                cursor = connection.execute(
                    f"UPDATE configuration.{relation} "
                    "SET rollout_seed=rollout_seed || '-proof' "
                    "WHERE revision_id=%s",
                    (revision_id,),
                )
                succeeded = int(cursor.rowcount) == 1
                raise ProofError(
                    "configuration immutable mutation unexpectedly succeeded"
                )
    except ProofError:
        raise
    except Exception as exc:
        if expected_message not in str(exc):
            raise ProofError(
                "configuration immutable mutation failed ambiguously"
            ) from exc
    if succeeded:
        raise ProofError("configuration immutable mutation unexpectedly succeeded")


def _prove_research_scenario_v4_wiring(
    database_url: str,
    research: Any,
    *,
    bundle_revision_id: str,
    solver_timeout_seconds: float,
) -> dict[str, Any]:
    """Persist, query, validate, and observer-project one real V4 result."""

    from baloot_platform.research.result_library import (
        RESULT_PROJECTION_VERSION,
        stored_result_payload_hash_v1,
    )

    golden_path = ROOT / "examples" / "research-scenario-v4-golden.json"
    try:
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProofError("Research V4 golden scenario is unavailable") from exc
    meaning = dict(golden.get("rulesMeaning") or {})
    prefix = dict(golden.get("exactTracePrefix") or {})
    proof_request = dict(golden.get("proof") or {})
    expected = dict(golden.get("expected") or {})
    payload = {
        "schemaVersion": 4,
        "code": "RCP-PGV4PROOF",
        "titleAr": "إثبات حي لسيناريو مشروع مية في الصن",
        "configuration": {
            "schemaVersion": 1,
            "bundleRevisionId": bundle_revision_id,
        },
        "scenario": {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": prefix,
            },
            "constraints": {
                "op": "all",
                "children": [
                    {
                        "predicateId": "PRED-PG-V4-BUYER",
                        "kind": "auction.buyer",
                        "version": 1,
                        "hardness": "hard",
                        "arguments": {"seat": meaning["buyer"]},
                    },
                    {
                        "predicateId": "PRED-PG-V4-CONTRACT",
                        "kind": "auction.contract",
                        "version": 1,
                        "hardness": "hard",
                        "arguments": {"contract": meaning["contract"]},
                    },
                    {
                        "predicateId": "PRED-PG-V4-PROJECT",
                        "kind": "project.exists",
                        "version": 1,
                        "hardness": "hard",
                        "arguments": {
                            "team": "north_south",
                            "owner": meaning["buyer"],
                            "kind": meaning["projectKind"],
                            "value": int(meaning["projectGovernedGamePoints"]),
                            "state": meaning["projectState"],
                        },
                    },
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": {
                "predicateId": "PRED-PG-V4-NO-SALVAGE",
                "kind": "proof.winning_root_action_count",
                "version": 1,
                "hardness": "hard",
                "arguments": {"minimum": 0, "maximum": 0},
            },
        },
        "proof": proof_request,
        "resources": {
            "generationLane": "ortools",
            "generationPolicy": "ortools_cp_sat_v2",
            "seedStart": 1,
            "seedCount": 1,
            "seedsPerShard": 1,
            "solverTimeoutSeconds": solver_timeout_seconds,
            "proofBudgetProfile": proof_request["budgetProfile"],
        },
        "status": "published",
    }
    try:
        created = research.create_recipe(
            payload,
            idempotency_key="live-postgres-v4-recipe",
        )
        run = research._submit_recipe_run(
            str(created["code"]),
            idempotency_key="live-postgres-v4-run",
            schedule=False,
        )
        research._execute(str(run["runCode"]))
        completed = research.get_run(str(run["runCode"]))
    except Exception as exc:
        raise ProofError("Research V4 PostgreSQL execution failed") from exc
    if (
        not isinstance(completed, Mapping)
        or completed.get("status") != "completed"
        or int(completed.get("resultCount") or 0) != 1
        or int(completed.get("qualifiedCount") or 0) != 1
    ):
        raise ProofError("Research V4 PostgreSQL run did not complete qualified")
    page = research.list_results(
        run_code=str(run["runCode"]),
        scenario_origin="exact_trace_prefix",
        scenario_status="satisfied",
        requested_proof_mode="exact",
        achieved_proof_mode="exact",
        proof_completion="exact_complete",
        proof_visibility="complete_state",
        target_team="north_south",
        project_kind=str(meaning["projectKind"]),
        project_value=str(meaning["projectGovernedGamePoints"]),
        project_state=str(meaning["projectState"]),
        winning_root_action_band="0",
        exact_state_available="true",
        decision_kind="play",
        research_backend="ortools_cp_sat_v2",
        qualification_authority="cp_sat",
        runtime_compatibility_status="passed",
        configuration_bundle=bundle_revision_id,
        variant_collection=str(run["runCode"]),
        offset=0,
        limit=10,
    )
    items = list(page.get("items") or ())
    if int(page.get("total") or 0) != 1 or len(items) != 1:
        raise ProofError(
            "Research V4 PostgreSQL facet query did not isolate its result"
        )
    result = dict(items[0])
    result_code = str(result.get("resultCode") or "")
    projection = dict(result.get("scenarioProjection") or {})
    proof_outcome = dict(result.get("proofOutcome") or {})
    research_authority = dict(result.get("researchAuthority") or {})
    runtime_compatibility = dict(result.get("runtimeCompatibility") or {})
    proof_artifact_hash = str(proof_outcome.get("artifactSha256") or "")
    proof_semantic_hash = str(
        proof_outcome.get("semanticSha256")
        or (proof_outcome.get("sha256") if not proof_artifact_hash else "")
        or ""
    )
    if (
        result.get("schemaVersion") != 2
        or result.get("disposition") != "qualified"
        or projection.get("status") != "satisfied"
        or projection.get("decisionKind") != "play"
        or int(proof_outcome.get("winningRootActionCount", -1))
        != int(expected["winningRootActionCount"])
        or proof_outcome.get("completionStatus") != expected["completionStatus"]
        or research_authority.get("backendId") != "ortools_cp_sat_v2"
        or research_authority.get("qualificationAuthority") != "cp_sat"
        or research_authority.get("bundleRevisionId") != bundle_revision_id
        or runtime_compatibility.get("status") != "passed"
    ):
        raise ProofError("Research V4 PostgreSQL result evidence drifted")
    provenance = dict(result.get("provenance") or {})
    identity = dict(provenance.get("resultIdentity") or {})
    artifacts = dict(result.get("artifacts") or {})
    artifact_hashes = {
        key: str(dict(reference).get("sha256") or "")
        for key, reference in sorted(artifacts.items())
    }
    semantic_hashes = {
        "scenarioSatisfactionHash": dict(result.get("scenarioSatisfaction") or {}).get(
            "sha256"
        ),
        "constraintEvidenceHash": dict(result.get("constraintEvidence") or {}).get(
            "sha256"
        ),
        "proofOutcomeHash": proof_semantic_hash,
    }
    identity_artifact_hashes = dict(identity.get("artifactHashes") or {})
    artifact_ownership_verified = (
        identity_artifact_hashes == artifact_hashes
        and bool(proof_semantic_hash)
        and proof_artifact_hash
        == artifact_hashes.get("proofOutcome")
        == identity_artifact_hashes.get("proofOutcome")
        and all(identity.get(key) == value for key, value in semantic_hashes.items())
    )
    configuration = dict(created.get("configuration") or {})
    configuration_hash = _sha256_text(_canonical_json(configuration))
    proof_identities = dict(proof_outcome.get("identities") or {})
    rules_identity = dict(
        dict(configuration.get("identities") or {}).get("rules") or {}
    )
    bundle_authority_verified = (
        dict(run.get("configurationAuthority") or {}) == configuration
        and identity.get("configurationAuthorityHash") == configuration_hash
        and provenance.get("configurationAuthorityHash") == configuration_hash
        and proof_identities.get("bundleHash")
        == configuration.get("semanticBundleHash")
        and proof_identities.get("rulesHash") == rules_identity.get("sha256")
    )
    result_owner_verified = (
        identity.get("runCode") == result.get("runCode") == run.get("runCode")
        and identity.get("runHash") == run.get("runHash")
        and identity.get("recipeCode")
        == result.get("recipeCode")
        == created.get("code")
        and identity.get("recipeHash")
        == provenance.get("recipeHash")
        == run.get("recipeHash")
        == created.get("sha256")
    )
    if not (
        artifact_ownership_verified
        and bundle_authority_verified
        and result_owner_verified
    ):
        raise ProofError("Research V4 PostgreSQL retained authority drifted")
    observer_rows = list(
        dict(result.get("libraryProjection") or {}).get("observerSeats") or ()
    )
    if len(observer_rows) != 1:
        raise ProofError("Research V4 PostgreSQL observer projection is not singular")
    view = research.get_exact_state(
        result_code,
        observer_seat=str(observer_rows[0]),
    )
    hands = dict(dict(view.get("publicState") or {}).get("hands") or {})
    exact_state_id = str(projection.get("exactStateId") or "")
    if (
        re.fullmatch(r"STATE-[A-F0-9]{20}", exact_state_id) is None
        or "exactStateId" in view
        or str(view.get("observerSeat") or "") != str(observer_rows[0])
        or sum(isinstance(hand, list) for hand in hands.values()) != 1
        or sum(
            isinstance(hand, Mapping) and set(hand) == {"count"}
            for hand in hands.values()
        )
        != 3
    ):
        raise ProofError(
            "Research V4 PostgreSQL exact-state projection leaked or drifted"
        )
    with _connect(database_url) as connection:
        row = connection.execute(
            """SELECT result_code,result_hash,payload_json,
                      payload_hash,library_projection_json,projection_version
               FROM research.results WHERE result_code=%s""",
            (result_code,),
        ).fetchone()
        facets = connection.execute(
            """SELECT facet_kind,facet_value
               FROM research.result_facets
               WHERE result_code=%s ORDER BY facet_kind,facet_value""",
            (result_code,),
        ).fetchall()
    if row is None or int(row[5]) != RESULT_PROJECTION_VERSION:
        raise ProofError("Research V4 PostgreSQL projection version drifted")
    stored_payload = dict(row[2]) if isinstance(row[2], Mapping) else {}
    stored_projection = dict(row[4]) if isinstance(row[4], Mapping) else {}
    stored_result_identity_bound = (
        str(row[0]) == result_code == str(stored_payload.get("resultCode") or "")
        and str(row[1])
        == str(result.get("resultHash") or "")
        == str(stored_payload.get("resultHash") or "")
        and str(row[3]) == stored_result_payload_hash_v1(stored_payload)
        and str(stored_projection.get("resultCode") or "") == result_code
        and str(stored_projection.get("resultHash") or "") == str(row[1])
    )
    if not stored_result_identity_bound:
        raise ProofError("Research V4 PostgreSQL stored result identity drifted")
    facet_set = {(str(kind), str(value)) for kind, value in facets}
    scenario_categories = {
        str(value).strip().lower()
        for value in projection.get("scenarioCategories") or ()
        if str(value).strip()
    }
    if not scenario_categories:
        raise ProofError("Research V4 PostgreSQL scenario categories are missing")
    exact_facet_page = research.list_results(
        exact_state_id=exact_state_id,
        scenario_category=sorted(scenario_categories)[0],
        offset=0,
        limit=10,
    )
    exact_facet_items = list(exact_facet_page.get("items") or ())
    if (
        int(exact_facet_page.get("total") or 0) != 1
        or len(exact_facet_items) != 1
        or str(exact_facet_items[0].get("resultCode") or "") != result_code
    ):
        raise ProofError(
            "Research V4 PostgreSQL exact-state/category facet query drifted"
        )
    required_facets = {
        ("scenario_origin", "exact_trace_prefix"),
        ("scenario_status", "satisfied"),
        ("proof_completion", "exact_complete"),
        ("project_kind", str(meaning["projectKind"])),
        ("project_state", str(meaning["projectState"])),
        ("winning_root_action_band", "0"),
        ("exact_state_available", "true"),
        ("decision_kind", "play"),
        ("research_backend", "ortools_cp_sat_v2"),
        ("qualification_authority", "cp_sat"),
        ("runtime_compatibility_status", "passed"),
        ("configuration_bundle", bundle_revision_id.lower()),
        ("exact_state_id", exact_state_id.lower()),
        ("variant_collection", str(run["runCode"]).lower()),
        *(("scenario_category", value) for value in scenario_categories),
    }
    if not required_facets <= facet_set:
        raise ProofError("Research V4 PostgreSQL facet projection drifted")
    return {
        "recipeCode": str(created["code"]),
        "runCode": str(run["runCode"]),
        "resultCode": result_code,
        "resultHash": str(result["resultHash"]),
        "exactStateId": exact_state_id,
        "proofOutcomeHash": proof_semantic_hash,
        "proofArtifactSha256": proof_artifact_hash,
        "winningRootActionCount": int(proof_outcome["winningRootActionCount"]),
        "projectionVersion": RESULT_PROJECTION_VERSION,
        "facetCount": len(facet_set),
        "observerSafeExactState": True,
        "artifactOwnershipVerified": artifact_ownership_verified,
        "bundleAuthorityVerified": bundle_authority_verified,
        "storedResultIdentityBound": stored_result_identity_bound,
        "exactStateFacetQueryVerified": True,
        "researchBackend": "ortools_cp_sat_v2",
        "qualificationAuthority": "cp_sat",
        "runtimeCompatibilityStatus": "passed",
        "configurationBundle": bundle_revision_id,
        "scenarioCategories": sorted(scenario_categories),
        "variantCollection": str(run["runCode"]),
    }


def _publish_live_ortools_bundle(
    configuration: Any,
    *,
    active_resolved: Mapping[str, Any],
    packaged: Any,
) -> Mapping[str, Any]:
    """Publish the explicit OR-Tools bundle without replacing the active bundle."""

    component_revisions: dict[str, str] = {}
    active_profiles = dict(active_resolved.get("profiles") or {})
    active_revisions = dict(active_resolved.get("profileRevisions") or {})
    for family in packaged.components:
        payload = (
            dict(packaged.rules.data)
            if family == "rules"
            else dict(packaged.components[family].data)
        )
        if active_profiles.get(family) == payload:
            component_revisions[family] = str(
                dict(active_revisions[family])["revisionId"]
            )
            continue
        created = configuration.create_profile(
            family=family,
            payload=payload,
            created_by="LIVE-POSTGRES-ORTOOLS-BUNDLE-PROOF",
            publish=True,
        )
        component_revisions[family] = str(created["revisionId"])
    profile = dict(packaged.bundle_profile.get("profile") or {})
    created_bundle = configuration.create_bundle(
        bundle_id=str(profile["id"]),
        version=str(profile["version"]),
        mode=str(packaged.mode),
        component_revisions=component_revisions,
        created_by="LIVE-POSTGRES-ORTOOLS-BUNDLE-PROOF",
        publish=True,
        allow_historical_components=True,
    )
    resolved = configuration.resolve(revision_id=str(created_bundle["revisionId"]))
    if dict(resolved.get("bundle") or {}).get("status") != "published":
        raise ProofError("explicit OR-Tools Research bundle is not published")
    return resolved


def _prove_full_wiring(
    database_url: str,
    proof_root: Path,
    admission_run_code: str,
) -> dict[str, Any]:
    from baloot_platform.config import (
        PROFILE_FAMILIES,
        build_bundle,
        profile_bundle_from_resolved,
    )
    from baloot_platform.configuration import PostgresConfigurationStore
    from baloot_platform.research.postgres_store import PostgresResearchStoreV1
    from baloot_platform.service.container import build_container
    from baloot_platform.storage.request_context import (
        CURRENT_DATABASE_IDENTITY,
        DatabaseIdentityContext,
        contextual_connection,
    )

    runtime = proof_root / "runtime"
    values = {
        "BALOOT_ROOT": str(ROOT),
        "BALOOT_DATA_DIR": str(runtime),
        "BALOOT_RUNTIME": str(runtime),
        "BALOOT_OUTPUTS": str(proof_root / "outputs"),
        "BALOOT_ENV": "development",
        "BALOOT_CHALLENGE_LOCATOR_KEY": secrets.token_urlsafe(48),
        "BALOOT_CURATED_SAKKA_SIGNING_KEY": secrets.token_urlsafe(48),
    }
    schema = _prove_configuration_v25_schema(database_url)
    with _temporary_environment(values):
        container = build_container(
            root=ROOT, writable_root=runtime, database_url=database_url
        )
        configuration_pool: Any | None = None
        try:
            if not isinstance(container.research_store, PostgresResearchStoreV1):
                raise ProofError("full container did not select PostgreSQL Research")
            if not isinstance(
                container.configuration_store, PostgresConfigurationStore
            ):
                raise ProofError(
                    "full container did not select PostgreSQL configuration"
                )
            configuration_pool = container.configuration_store.pool
            if getattr(configuration_pool, "closed", None) is not False:
                raise ProofError("PostgreSQL configuration pool was not open")
            identity = DatabaseIdentityContext(
                subject_id="LIVE-POSTGRES-CONFIGURATION-PROOF",
                role="service",
                request_id="live-postgres-configuration-proof",
            )
            token = CURRENT_DATABASE_IDENTITY.set(identity)
            try:
                with contextual_connection(configuration_pool) as connection:
                    context_row = connection.execute(
                        "SELECT current_setting('app.subject_id',true) AS subject_id,"
                        "current_setting('app.role',true) AS role"
                    ).fetchone()
                if isinstance(context_row, Mapping):
                    context_values = (
                        str(context_row.get("subject_id") or ""),
                        str(context_row.get("role") or ""),
                    )
                else:
                    context_values = (
                        tuple(map(str, context_row)) if context_row is not None else ()
                    )
                if context_values != (identity.subject_id, identity.role):
                    raise ProofError(
                        "configuration service RLS context was not applied"
                    )
                try:
                    resolved = container.configuration.resolve()
                except Exception as exc:
                    raise ProofError(
                        "full container could not resolve its active configuration bundle"
                    ) from exc
                bundle = dict(resolved.get("bundle") or {})
                profiles = dict(resolved.get("profiles") or {})
                revisions = dict(resolved.get("profileRevisions") or {})
                family_set = set(PROFILE_FAMILIES)
                if len(PROFILE_FAMILIES) != 32 or len(family_set) != 32:
                    raise ProofError("configuration profile family inventory drifted")
                component_revisions = dict(bundle.get("componentRevisions") or {})
                if (
                    set(profiles) != family_set
                    or set(revisions) != family_set
                    or set(component_revisions) != family_set
                ):
                    raise ProofError(
                        "resolved configuration profile family authority is not closed"
                    )
                if component_revisions != {
                    family: str(revisions[family].get("revisionId") or "")
                    for family in PROFILE_FAMILIES
                }:
                    raise ProofError(
                        "resolved configuration component revisions are inconsistent"
                    )
                packaged = build_bundle(ROOT)
                active = profile_bundle_from_resolved(resolved, root=ROOT)
                packaged_profile = dict(packaged.bundle_profile.get("profile") or {})
                if (
                    set(packaged_profile) != {"id", "version", "status"}
                    or not isinstance(packaged_profile.get("id"), str)
                    or not packaged_profile["id"]
                    or not isinstance(packaged_profile.get("version"), str)
                    or not packaged_profile["version"]
                    or packaged_profile.get("status") != "governed_candidate"
                ):
                    raise ProofError("packaged bundle profile authority is invalid")
                packaged_components = dict(packaged.components)
                active_components = dict(active.components)
                if (
                    set(packaged_components) != family_set
                    or set(active_components) != family_set
                ):
                    raise ProofError(
                        "configuration profile component family authority is not closed"
                    )
                expected_profile_hashes = {
                    family: str(packaged_components[family].sha256)
                    for family in PROFILE_FAMILIES
                }
                active_profile_hashes = {
                    family: str(active_components[family].sha256)
                    for family in PROFILE_FAMILIES
                }
                revision_profile_hashes = {
                    family: str(revisions[family].get("payloadSha256") or "")
                    for family in PROFILE_FAMILIES
                }
                if (
                    str(packaged.rules.sha256) != expected_profile_hashes["rules"]
                    or str(active.rules.sha256) != active_profile_hashes["rules"]
                ):
                    raise ProofError("configuration rules payload authority drifted")
                if active_profile_hashes != expected_profile_hashes:
                    raise ProofError(
                        "active profile payloads do not match packaged authority"
                    )
                if revision_profile_hashes != expected_profile_hashes:
                    raise ProofError(
                        "resolved profile revision payloads do not match packaged authority"
                    )
                active_profile = dict(active.bundle_profile.get("profile") or {})
                expected_identity = {
                    "id": packaged_profile["id"],
                    "version": packaged_profile["version"],
                    "mode": packaged.mode,
                }
                if {
                    "id": bundle.get("bundleId"),
                    "version": bundle.get("version"),
                    "mode": bundle.get("mode"),
                } != expected_identity:
                    raise ProofError(
                        "runtime bundle row identity does not match packaged authority"
                    )
                if (
                    set(active_profile) != {"id", "version", "status"}
                    or {
                        "id": active_profile.get("id"),
                        "version": active_profile.get("version"),
                        "mode": active.mode,
                    }
                    != expected_identity
                ):
                    raise ProofError(
                        "active runtime bundle identity does not match packaged authority"
                    )
                if (
                    bundle.get("status") != "published"
                    or active_profile.get("status") != "published"
                ):
                    raise ProofError("runtime bundle lifecycle status is not published")
                authority_document = {
                    "schemaVersion": 1,
                    "bundle": expected_identity,
                    "profilePayloadSha256": expected_profile_hashes,
                }
                authority_sha256 = _sha256_text(_canonical_json(authority_document))
                exact = container.configuration.resolve(
                    revision_id=str(bundle["revisionId"])
                )
                if str(exact.get("effectiveHash") or "") != str(
                    resolved.get("effectiveHash") or ""
                ):
                    raise ProofError("exact active configuration resolution drifted")
                bundle_chain = _configuration_event_chain_evidence(
                    container.configuration_store.events_with_verification(
                        "bundle", str(bundle["revisionId"])
                    ),
                    aggregate_type="bundle",
                    aggregate_id=str(bundle["revisionId"]),
                )
                profile_chains = {
                    family: _configuration_event_chain_evidence(
                        container.configuration_store.events_with_verification(
                            "profile", str(revisions[family]["revisionId"])
                        ),
                        aggregate_type="profile",
                        aggregate_id=str(revisions[family]["revisionId"]),
                    )
                    for family in PROFILE_FAMILIES
                }
                if dict(resolved.get("audit") or {}) != {
                    "valid": True,
                    "issues": [],
                    "eventCount": bundle_chain["eventCount"],
                    "eventHead": bundle_chain["eventHead"],
                }:
                    raise ProofError("resolved configuration audit chain drifted")
                _assert_configuration_immutability(
                    configuration_pool,
                    relation="profile_revisions",
                    revision_id=str(revisions["rules"]["revisionId"]),
                    expected_message=(
                        "published configuration profile revisions are immutable"
                    ),
                )
                _assert_configuration_immutability(
                    configuration_pool,
                    relation="bundle_revisions",
                    revision_id=str(bundle["revisionId"]),
                    expected_message=(
                        "published configuration bundle revisions are immutable"
                    ),
                )
                ortools_packaged = build_bundle(
                    ROOT,
                    bundle_path="configs/bundle_ortools_research_v4.yaml",
                )
                ortools_resolved = _publish_live_ortools_bundle(
                    container.configuration,
                    active_resolved=resolved,
                    packaged=ortools_packaged,
                )
                active_after_publish = profile_bundle_from_resolved(
                    container.configuration.resolve(
                        revision_id=str(bundle["revisionId"])
                    ),
                    root=ROOT,
                )
                ortools_runtime = profile_bundle_from_resolved(
                    ortools_resolved,
                    root=ROOT,
                )
                ortools_challenge = dict(
                    ortools_runtime.component("challenge").get("scenarioResearchV4")
                    or {}
                )
                if (
                    active_after_publish.sha256 != active.sha256
                    or ortools_challenge.get("backend") != "ortools_cp_sat_v2"
                ):
                    raise ProofError(
                        "explicit OR-Tools bundle changed active legacy authority"
                    )
                scenario_v4 = _prove_research_scenario_v4_wiring(
                    database_url,
                    container.research,
                    bundle_revision_id=str(
                        dict(ortools_resolved["bundle"])["revisionId"]
                    ),
                    solver_timeout_seconds=min(
                        float(
                            ortools_challenge["proofBudgets"]["profiles"][
                                "late_game_exact_v1"
                            ]["maximumWallSeconds"]
                        ),
                        float(
                            ortools_challenge["searchBudgets"]["profiles"][
                                "late_game_exact_v1"
                            ]["maximumWallSeconds"]
                        ),
                    ),
                )
                required_result_bindings = (
                    "artifactOwnershipVerified",
                    "bundleAuthorityVerified",
                    "storedResultIdentityBound",
                )
                if any(
                    scenario_v4.get(binding) is not True
                    for binding in required_result_bindings
                ):
                    raise ProofError(
                        "Research V4 PostgreSQL result bindings are incomplete"
                    )
            finally:
                CURRENT_DATABASE_IDENTITY.reset(token)
            stats = container.research.statistics(run_code=admission_run_code)
            if stats["resultCount"] != 1:
                raise ProofError("full container Research statistics are inconsistent")
        finally:
            container.shutdown()
        if (
            configuration_pool is None
            or getattr(configuration_pool, "closed", None) is not True
        ):
            raise ProofError("PostgreSQL configuration pool did not close cleanly")
    sessions = _wait_for_no_client_sessions(database_url)
    return {
        "postgresResearchStore": True,
        "postgresConfigurationStore": True,
        "configurationServiceRlsContext": True,
        "packagedBundleResolved": True,
        "activeBundleRevisionId": str(bundle["revisionId"]),
        "packagedBundleSha256": str(packaged.sha256),
        "activeBundleSha256": active.sha256,
        "bundleLifecycleStatuses": {
            "packaged": str(packaged_profile["status"]),
            "runtimeRow": str(bundle["status"]),
            "active": str(active_profile["status"]),
        },
        "authoritySha256": authority_sha256,
        "profileAuthorityCount": len(PROFILE_FAMILIES),
        "configurationEventChains": {
            "bundle": bundle_chain,
            "profiles": profile_chains,
        },
        "configurationV25Schema": schema,
        "publishedImmutabilityEnforced": True,
        "ortoolsResearchBundle": {
            "revisionId": str(dict(ortools_resolved["bundle"])["revisionId"]),
            "backend": "ortools_cp_sat_v2",
            "authoredBundleSha256": str(ortools_packaged.sha256),
            "runtimeBundleSha256": str(ortools_runtime.sha256),
            "activeLegacyParity": True,
        },
        "researchScenarioV4": scenario_v4,
        "statisticsReadable": True,
        "configurationPoolClosed": True,
        "configurationSessionsClosed": sessions,
    }


def _wait_for_no_client_sessions(
    database_url: str, *, timeout_seconds: float = 10
) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout_seconds)
    remaining = -1
    while True:
        with _connect(database_url) as connection:
            row = connection.execute(
                """SELECT COUNT(*) FROM pg_stat_activity
                   WHERE datname=current_database()
                     AND pid<>pg_backend_pid()
                     AND backend_type='client backend'"""
            ).fetchone()
        remaining = int(row[0]) if row is not None else -1
        if remaining == 0:
            return {"remainingClientSessions": 0}
        if time.monotonic() >= deadline:
            raise ProofError(f"PostgreSQL proof left {remaining} client sessions open")
        time.sleep(0.25)


def _admission_run_code(proof_id: str) -> str:
    return f"RUN-PG-{proof_id[:10].upper()}-ADMIT"


def _assert_sanitized(value: Any, secrets_to_remove: Sequence[str]) -> None:
    encoded = _canonical_json(value)
    if DSN_RE.search(encoded):
        raise ProofError("evidence contains a database connection string")
    if any(secret and secret in encoded for secret in secrets_to_remove):
        raise ProofError("evidence contains a proof credential")

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized = re.sub(r"[^a-z]", "", str(key).lower())
                if normalized in {"dsn", "databaseurl", "password"}:
                    raise ProofError("evidence contains a sensitive field")
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)


def _write_evidence(path: Path, report: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    if output.exists():
        raise ProofError("refusing to overwrite existing PostgreSQL proof evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["sha256"] = _sha256_text(_canonical_json(payload))
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )


def run_live_proof(
    *,
    output: Path,
    health_timeout_seconds: float = 90,
) -> dict[str, Any]:
    if not 30 <= float(health_timeout_seconds) <= 180:
        raise ProofError("health timeout must be between 30 and 180 seconds")
    if output.expanduser().resolve().exists():
        raise ProofError("refusing to overwrite existing PostgreSQL proof evidence")
    context = _new_context()
    password = str(context["password"])
    secrets_to_remove = [password]
    checks: dict[str, Any] = {}
    environment_evidence: dict[str, Any] = {}
    resource_evidence: dict[str, Any] = {}
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": PROOF_KIND,
        "status": "failed",
        "proofId": str(context["proofId"]),
        "checks": checks,
    }
    container_reference = str(context["containerName"])
    proxy_reference = str(context["proxyContainerName"])
    network_reference = str(context["networkName"])
    access_network_reference = str(context["accessNetworkName"])
    start_attempted = False
    proxy_start_attempted = False
    network_start_attempted = False
    access_network_start_attempted = False
    cleanup: dict[str, Any] = {}
    primary_error: Exception | None = None
    try:
        source_evidence = _source_manifest_evidence()
        environment_evidence["source"] = source_evidence
        child_dependencies = _probe_python_child_dependencies()
        dependencies = _dependency_versions()
        if any(
            child_dependencies[key] != dependencies[key]
            for key in ("python", "psycopg", "psycopgPool")
        ):
            raise ProofError("Python parent and child dependency versions disagree")
        checks["pythonChildPreflight"] = child_dependencies
        environment_evidence["runtime"] = dependencies
        runtime_wiring = _probe_full_runtime_wiring()
        if runtime_wiring["packageVersion"] != source_evidence["version"]:
            raise ProofError("runtime package and source manifest versions disagree")
        checks["fullRuntimePreflight"] = runtime_wiring
        docker_context = _local_docker_context()
        environment_evidence["dockerContext"] = docker_context
        context["dockerContext"] = str(docker_context["name"])
        docker_server = _docker_server(str(context["dockerContext"]))
        environment_evidence["docker"] = docker_server
        image = _resolve_image(
            docker_context=str(context["dockerContext"]),
            server_os=docker_server["os"],
            server_architecture=docker_server["architecture"],
        )
        environment_evidence["image"] = {
            key: value for key, value in image.items() if key != "resolvedReference"
        }
        _assert_container_name_unused(
            str(context["containerName"]), str(context["dockerContext"])
        )
        _assert_container_name_unused(
            str(context["proxyContainerName"]), str(context["dockerContext"])
        )
        _assert_network_name_unused(
            str(context["networkName"]), str(context["dockerContext"])
        )
        _assert_network_name_unused(
            str(context["accessNetworkName"]), str(context["dockerContext"])
        )
        network_start_attempted = True
        network_id = _create_network(context)
        context["networkId"] = network_id
        network_reference = network_id
        access_network_start_attempted = True
        access_network_id = _create_access_network(context)
        context["accessNetworkId"] = access_network_id
        access_network_reference = access_network_id
        start_attempted = True
        container_id = _start_container(context, str(image["resolvedReference"]))
        container_reference = container_id
        _wait_for_health(
            container_id,
            docker_context=str(context["dockerContext"]),
            timeout_seconds=health_timeout_seconds,
        )
        proxy_start_attempted = True
        proxy_container_id = _start_proxy_container(
            context, str(image["resolvedReference"])
        )
        proxy_reference = proxy_container_id
        network = _validated_network(
            network_id,
            context,
            expected_container_ids=(container_id, proxy_container_id),
        )
        resource_evidence["network"] = network
        access_network = _validated_access_network(
            access_network_id,
            context,
            expected_proxy_container_id=proxy_container_id,
        )
        resource_evidence["accessNetwork"] = access_network
        container = _validated_container(
            container_id, context, str(image["resolvedReference"])
        )
        resource_evidence["container"] = container
        proxy = _validated_proxy_container(
            proxy_container_id, context, str(image["resolvedReference"])
        )
        resource_evidence["loopbackRelay"] = proxy
        database_url = _database_url(context, int(proxy["hostPort"]))
        secrets_to_remove.append(database_url)
        checks["databaseReadiness"] = _database_readiness(
            database_url, str(context["database"])
        )
        checks["migrations"] = _prove_migrations(
            database_url,
            Path(context["proofRoot"]),
            secrets_to_remove=secrets_to_remove,
        )
        checks["research"] = _prove_research(
            database_url,
            str(context["proofId"]),
            expected_schema_fingerprint=checks["migrations"]["schemaFingerprint"],
        )
        checks["fullWiring"] = _prove_full_wiring(
            database_url,
            Path(context["proofRoot"]),
            _admission_run_code(str(context["proofId"])),
        )
        checks["sessionCleanup"] = _wait_for_no_client_sessions(database_url)
        report = {
            "schemaVersion": 1,
            "kind": PROOF_KIND,
            "status": "passed",
            "proofId": str(context["proofId"]),
            "environment": environment_evidence,
            **resource_evidence,
            "checks": checks,
            "proofBoundary": (
                "One internal-only ephemeral PostgreSQL 17 instance, reached through "
                "a least-privilege loopback relay, proves current migrations, "
                "Research bootstrap, real concurrent leases, fencing, retry, admission, "
                "statistics, immutable artifact serialization, and application wiring. "
                "It is not a failover, production-load, or deployment proof."
            ),
        }
    except Exception as exc:  # evidence owns the failure boundary
        primary_error = exc
        if proxy_start_attempted:
            try:
                inspected = _owned_container(
                    proxy_reference,
                    context,
                    expected_role=PROXY_CONTAINER_ROLE,
                )
                if inspected is not None:
                    identity = str(inspected.get("Id") or "")
                    if CONTAINER_ID_RE.fullmatch(identity):
                        checks["proxyLogs"] = _container_logs(
                            identity,
                            str(context["dockerContext"]),
                            secrets_to_remove,
                        )
            except Exception as diagnostic_error:
                checks["proxyLogs"] = {
                    "status": "unavailable",
                    "error": _redact(str(diagnostic_error), secrets_to_remove)[:1000],
                }
        if start_attempted:
            try:
                inspected = _owned_container(container_reference, context)
                if inspected is not None:
                    identity = str(inspected.get("Id") or "")
                    if CONTAINER_ID_RE.fullmatch(identity):
                        checks["containerLogs"] = _container_logs(
                            identity,
                            str(context["dockerContext"]),
                            secrets_to_remove,
                        )
            except Exception as diagnostic_error:
                checks["containerLogs"] = {
                    "status": "unavailable",
                    "error": _redact(str(diagnostic_error), secrets_to_remove)[:1000],
                }
        report = {
            "schemaVersion": 1,
            "kind": PROOF_KIND,
            "status": "failed",
            "proofId": str(context["proofId"]),
            **({"environment": environment_evidence} if environment_evidence else {}),
            **resource_evidence,
            "checks": checks,
            "error": {
                "type": type(exc).__name__,
                "message": _redact(str(exc), secrets_to_remove)[:4000],
            },
        }
    finally:
        try:
            cleanup["loopbackRelay"] = (
                _cleanup_container(
                    proxy_reference,
                    context,
                    expected_role=PROXY_CONTAINER_ROLE,
                    wait_for_appearance_seconds=5,
                )
                if proxy_start_attempted
                else {"status": "not_started"}
            )
        except Exception as exc:
            cleanup["loopbackRelay"] = {
                "status": "failed",
                "error": _redact(str(exc), secrets_to_remove)[:4000],
            }
            if primary_error is None:
                primary_error = exc
        try:
            cleanup["container"] = (
                _cleanup_container(
                    container_reference,
                    context,
                    wait_for_appearance_seconds=5,
                )
                if start_attempted
                else {"status": "not_started"}
            )
        except Exception as exc:
            cleanup["container"] = {
                "status": "failed",
                "error": _redact(str(exc), secrets_to_remove)[:4000],
            }
            if primary_error is None:
                primary_error = exc
        try:
            cleanup["accessNetwork"] = (
                _cleanup_network(
                    access_network_reference,
                    context,
                    expected_role=ACCESS_NETWORK_ROLE,
                    wait_for_appearance_seconds=5,
                )
                if access_network_start_attempted
                else {"status": "not_started"}
            )
        except Exception as exc:
            cleanup["accessNetwork"] = {
                "status": "failed",
                "error": _redact(str(exc), secrets_to_remove)[:4000],
            }
            if primary_error is None:
                primary_error = exc
        try:
            cleanup["network"] = (
                _cleanup_network(
                    network_reference,
                    context,
                    wait_for_appearance_seconds=5,
                )
                if network_start_attempted
                else {"status": "not_started"}
            )
        except Exception as exc:
            cleanup["network"] = {
                "status": "failed",
                "error": _redact(str(exc), secrets_to_remove)[:4000],
            }
            if primary_error is None:
                primary_error = exc
        try:
            cleanup["proofRoot"] = _safe_remove_proof_root(Path(context["proofRoot"]))
        except Exception as exc:
            cleanup["proofRoot"] = {
                "status": "failed",
                "error": _redact(str(exc), secrets_to_remove)[:4000],
            }
            if primary_error is None:
                primary_error = exc
        report["cleanup"] = cleanup
        if any(value.get("status") == "failed" for value in cleanup.values()):
            report["status"] = "failed"
            report.setdefault(
                "error",
                {
                    "type": type(primary_error).__name__
                    if primary_error
                    else "ProofError",
                    "message": _redact(
                        str(primary_error or "cleanup failed"), secrets_to_remove
                    )[:4000],
                },
            )
    _assert_sanitized(report, secrets_to_remove)
    _write_evidence(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the isolated BalootX live PostgreSQL Research proof"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--health-timeout-seconds", type=float, default=90)
    arguments = parser.parse_args()
    try:
        report = run_live_proof(
            output=arguments.output,
            health_timeout_seconds=arguments.health_timeout_seconds,
        )
    except Exception as exc:
        print(f"PostgreSQL proof could not start: {type(exc).__name__}: {exc}")
        return 1
    print(arguments.output.expanduser().resolve())
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
