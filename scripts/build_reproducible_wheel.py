#!/usr/bin/env python3
"""Build, compare, and clean-install the exact current BalootX wheel twice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from baloot_platform.filesystem import atomic_write_text  # noqa: E402

MIGRATION_028 = ROOT / "migrations/028_research_authority_projection.sql"
VERSION = str(
    tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
)
REPORT = ROOT / "release/wheel-reproducibility.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(
    command: list[str], *, cwd: Path = ROOT, environment: dict[str, str] | None = None
) -> None:
    merged = {**os.environ, **(environment or {})}
    subprocess.run(command, cwd=cwd, env=merged, check=True)


def _clean_python_staging() -> None:
    for path in (
        ROOT / "build/lib",
        *sorted((ROOT / "build").glob("bdist.*")),
        ROOT / "src/baloot_platform.egg-info",
        ROOT / "baloot_platform.egg-info",
    ):
        shutil.rmtree(path, ignore_errors=True)


def _install_wheel(python: Path, wheel: Path, *, cwd: Path) -> None:
    _run(
        [str(python), "-m", "pip", "install", str(wheel)],
        cwd=cwd,
        environment={"PYTHONPATH": "", "BALOOT_ROOT": ""},
    )


def _probe_stdout(completed: subprocess.CompletedProcess[str]) -> str:
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        completed.check_returncode()
    return completed.stdout


def _source_identity() -> dict[str, object]:
    source = ROOT / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        from scripts.generate_performance_evidence import git_identity
    except ModuleNotFoundError:
        from generate_performance_evidence import git_identity
    return git_identity(ROOT)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    source_identity = _source_identity()
    migration_028_sha256 = _sha256(MIGRATION_028)
    environment = {
        "PYTHONHASHSEED": "0",
        "SOURCE_DATE_EPOCH": "315532800",
    }
    with tempfile.TemporaryDirectory(prefix="baloot-wheel-pair-") as temporary:
        pair_root = Path(temporary)
        wheels: list[Path] = []
        for label in ("a", "b"):
            _clean_python_staging()
            destination = pair_root / label
            destination.mkdir()
            _run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    ".",
                    "--no-deps",
                    "--no-build-isolation",
                    "-w",
                    str(destination),
                ],
                environment=environment,
            )
            wheel = next(destination.glob(f"baloot_platform-{VERSION}-*.whl"))
            wheels.append(wheel)
        identities = [
            {"bytes": wheel.stat().st_size, "sha256": _sha256(wheel)}
            for wheel in wheels
        ]
        if identities[0] != identities[1] or wheels[0].name != wheels[1].name:
            raise RuntimeError(
                "clean wheel builds are not byte-identical: "
                + json.dumps(identities, sort_keys=True)
            )
        shutil.rmtree(ROOT / "dist", ignore_errors=True)
        (ROOT / "dist").mkdir()
        published = ROOT / "dist" / wheels[0].name
        shutil.copyfile(wheels[0], published)

    with tempfile.TemporaryDirectory(prefix="baloot-wheel-install-") as temporary:
        venv = Path(temporary) / "venv"
        _run([sys.executable, "-m", "venv", str(venv)])
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        _install_wheel(python, published, cwd=Path(temporary))
        runtime_probe = (
            "import hashlib, importlib.resources, json; "
            "from baloot_platform.config import build_bundle, platform_root; "
            "from baloot_platform.research import ScenarioPredicateRegistryV1; "
            "from baloot_platform.service.container import build_container; "
            "from baloot_platform.version import __version__; "
            "root=platform_root(); bundle=build_bundle(root); "
            "ortools_bundle=build_bundle(root,bundle_path='configs/bundle_ortools_research_v4.yaml'); "
            "catalog=ScenarioPredicateRegistryV1().catalog(); "
            "migration=importlib.resources.files('baloot_platform.resources').joinpath('migrations/028_research_authority_projection.sql'); "
            "migration_hash=hashlib.sha256(migration.read_bytes()).hexdigest(); "
            "container=build_container(); "
            "assert __version__==" + repr(VERSION) + "; "
            "assert catalog['registryVersion']=='scenario_predicate_registry_v1'; "
            "assert 'scenarioResearchV4' not in bundle.component('challenge'); "
            "assert ortools_bundle.component('challenge')['scenarioResearchV4']['backend']=='ortools_cp_sat_v2'; "
            "assert migration.is_file(); "
            "assert migration_hash==" + repr(migration_028_sha256) + "; "
            "assert container.registry.get('video_renderer','premium_video') is not None; "
            "container.shutdown(); "
            "print(json.dumps({'version':__version__,'bundleHash':bundle.sha256,'ortoolsBundleHash':ortools_bundle.sha256,'ortoolsBackend':'ortools_cp_sat_v2','scenarioPredicateCount':len(catalog['definitions']),'scenarioPredicateCatalogSha256':catalog['sha256'],'migration028Sha256':migration_hash,'videoPlugins':['premium_video']}))"
        )
        completed = subprocess.run(
            [str(python), "-c", runtime_probe],
            cwd=Path(temporary),
            env={
                **os.environ,
                "PYTHONPATH": "",
                "BALOOT_ROOT": "",
                "BALOOT_DATA_DIR": str(Path(temporary) / "data"),
                "BALOOT_RUNTIME": str(Path(temporary) / "runtime"),
                "BALOOT_OUTPUTS": str(Path(temporary) / "outputs"),
            },
            text=True,
            capture_output=True,
            check=False,
        )
        installed = json.loads(_probe_stdout(completed).strip())

    report = {
        "schemaVersion": 1,
        "status": "passed",
        "source": source_identity,
        "sourceDateEpoch": 315532800,
        "builds": identities,
        "byteIdentical": True,
        "wheel": published.relative_to(ROOT).as_posix(),
        "installedRuntime": {
            "version": installed["version"],
            "bundleHash": installed["bundleHash"],
            "ortoolsBundleHash": installed["ortoolsBundleHash"],
            "ortoolsBackend": installed["ortoolsBackend"],
            "packagedResources": True,
            "packagedRemotionRenderer": True,
            "migration028Sha256": installed["migration028Sha256"],
            "scenarioPredicateCatalogSha256": installed[
                "scenarioPredicateCatalogSha256"
            ],
            "scenarioPredicateCount": installed["scenarioPredicateCount"],
            "videoPlugins": installed["videoPlugins"],
        },
    }
    atomic_write_text(
        REPORT,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(REPORT)
    print(published)


if __name__ == "__main__":
    main()
