from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from zipfile import ZipFile

import yaml
import pytest
from packaging.version import Version

from baloot_platform.version import __version__


FOUNDRY_NPM = "service/foundry_dashboard"
REMOTION_NPM = "video/remotion"


@pytest.mark.parametrize(
    ("package_dir", "dependency", "first_patched"),
    [
        (FOUNDRY_NPM, "js-yaml", "4.3.2"),  # GHSA-2883-xcg3-v3hh
        (FOUNDRY_NPM, "@vitest/mocker", "4.1.12"),  # GHSA-qxg6-89mg-rqv8
        (FOUNDRY_NPM, "playwright", "1.55.1"),  # GHSA-7mvr-c777-76hp
        (FOUNDRY_NPM, "source-map-js", "1.2.2"),  # GHSA-68fv-2mgg-jv7q
        (FOUNDRY_NPM, "vite", "8.2.4"),  # GHSA-587h-7wmj-h5pm
        (FOUNDRY_NPM, "vitest", "4.1.11"),  # GHSA-82fw-gwwq-j7x9
        (REMOTION_NPM, "fast-uri", "3.1.9"),  # GHSA-w74x-v6f9-j3xg
        (REMOTION_NPM, "postcss-selector-parser", "7.1.6"),  # GHSA-rj75-hqrm-r3gf
        (REMOTION_NPM, "source-map-js", "1.2.2"),  # GHSA-68fv-2mgg-jv7q
        (REMOTION_NPM, "zod", "4.6.6"),  # GHSA-mrhx-c7xc-m3w5
    ],
)
def test_npm_lockfiles_keep_security_fixes(root, package_dir, dependency, first_patched):
    lockfile = root / "src/baloot_platform" / package_dir / "package-lock.json"
    packages = json.loads(lockfile.read_bytes())["packages"]
    package_path = f"node_modules/{dependency}"
    versions = {
        Version(entry["version"])
        for path, entry in packages.items()
        if path == package_path or path.endswith(f"/{package_path}")
    }
    vulnerable = sorted(version for version in versions if version < Version(first_patched))
    assert not vulnerable, (
        f"{dependency} needs at least {first_patched}: {vulnerable}"
    )


def test_runtime_and_build_directories_are_not_tracked(root: Path) -> None:
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    forbidden = (
        ".release-runtime/",
        ".pytest_cache/",
        ".ruff_cache/",
        "build/",
        "dist/",
    )

    assert not [path for path in tracked if path.startswith(forbidden)]


def test_workspace_root_is_the_repository_root(root: Path) -> None:
    git_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert Path(git_root).resolve() == root.resolve()
    assert not (root / "deliveries").exists()


def test_active_engine_video_surfaces_use_responsibility_names(root: Path) -> None:
    required = (
        "src/baloot_platform/benchmark/strength.py",
        "src/baloot_platform/video/remotion_renderer.py",
        "src/baloot_platform/video/render_bundle.py",
        "src/baloot_platform/video/scene.py",
        "src/baloot_platform/video/scene_adapter.py",
        "scripts/render_release.py",
        "scripts/check_release_readiness.py",
        "configs/video/premium.yaml",
    )

    assert all((root / relative).is_file() for relative in required)


def test_obsolete_pillow_renderer_chain_is_absent(root: Path) -> None:
    obsolete = (
        "src/baloot_platform/video/premium.py",
        "src/baloot_platform/video/premium_v8.py",
        "src/baloot_platform/video/premium_v9.py",
        "src/baloot_platform/video/premium_v10.py",
        "src/baloot_platform/video/premium_sakka_v12.py",
        "src/baloot_platform/video/premium_shared_v13.py",
        "src/baloot_platform/video/premium_natural_v14.py",
        "src/baloot_platform/video/premium_authentic_v15.py",
        "src/baloot_platform/video/flagship.py",
        "src/baloot_platform/video/match.py",
        "src/baloot_platform/video/bid.py",
    )

    assert not [relative for relative in obsolete if (root / relative).exists()]


def test_verifier_container_uses_current_responsibility_entrypoint(root: Path) -> None:
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert 'CMD ["python", "scripts/verifier_preflight.py"]' in dockerfile
    assert (root / "scripts/verifier_preflight.py").is_file()
    assert sorted(
        path.name for path in (root / "configs/video").glob("premium*.yaml")
    ) == ["premium.yaml"]
    assert sorted(
        path.name
        for path in (root / "src/baloot_platform/resources/configs/video").glob(
            "premium*.yaml"
        )
    ) == ["premium.yaml"]


def test_ci_shards_receive_builds_and_required_node_dependencies(root: Path) -> None:
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())["jobs"]
    steps = jobs["python-fast-run"]["steps"]
    installs = {step.get("working-directory") for step in steps
                if step.get("run", "").startswith("npm ci ")}
    assert installs == {"src/baloot_platform/service/foundry_dashboard",
                        "src/baloot_platform/video/remotion"}
    assert jobs["python-backend-run"]["steps"] == steps == jobs["python-heavy"]["steps"]
    for name in ("python-backend-run", "python-fast-run", "python-heavy"):
        assert {"plan", "python-wheelhouse", "native-build", "ffmpeg-packages"} <= set(jobs[name]["needs"])
    downloads = {step["with"]["name"] for step in steps if "download-artifact@" in step.get("uses", "")}
    assert downloads == {"python-runtime-${{ matrix.python }}", "native-release-library",
                         "ffmpeg-debs-6.1.1-3ubuntu5"}
    assert all(step.get("with", {}).get("digest-mismatch") == "error" for step in steps
               if "download-artifact@" in step.get("uses", ""))
    foundry = jobs["foundry"]["steps"]
    assert any(step.get("run", "").startswith("npm run test:research -- ") for step in foundry)
    e2e, = (step for step in foundry if step.get("run", "").startswith("npm run test:operator-e2e -- "))
    output_dir = e2e["env"]["BALOOT_OPERATOR_E2E_OUTPUT_DIR"]
    dashboard = e2e.get("working-directory", jobs["foundry"]["defaults"]["run"]["working-directory"])
    config = (root / dashboard / "e2e/playwright.config.cjs").read_text(encoding="utf-8")
    assert "\n  outputDir: resolve(root, process.env.BALOOT_OPERATOR_E2E_OUTPUT_DIR ||" in config
    assert "\n    trace: 'retain-on-failure'," in config
    receipts, = (step for step in foundry if step.get("with", {}).get("name") == "foundry-test-receipts")
    assert f"{output_dir}/**/audit-*.png" in receipts["with"]["path"].splitlines()
    diagnostics, = (step for step in foundry
                    if step.get("with", {}).get("name") == "foundry-playwright-failure-diagnostics")
    assert diagnostics["with"]["path"] == output_dir
    assert diagnostics["with"]["if-no-files-found"] == "ignore"
    assert diagnostics["with"]["retention-days"] == 7
    assert diagnostics["if"] == f"${{{{ !cancelled() && steps.{e2e['id']}.outcome == 'failure' }}}}"
    assert installs <= {step.get("working-directory") for step in foundry
                        if step.get("run", "").startswith("npm ci ")}
    assert jobs["media-render"]["strategy"]["matrix"]["shard"] == ["flagship", "decision", "companion", "parity"]
    assert jobs["media-release-smoke"]["needs"] == ["plan", "media-render"]
    assert any(step.get("name") == "Verify governed font hashes" for step in jobs["media-inputs"]["steps"])


def test_full_ci_reuses_only_the_run_local_installed_python_runtime(root: Path) -> None:
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())["jobs"]
    for name in ("foundry", "python-fast-run", "python-backend-run", "python-heavy",
                 "media-inputs", "media-render", "media-release-smoke"):
        steps = jobs[name]["steps"]
        index, = (i for i, step in enumerate(steps) if step.get("with", {}).get("name", "").startswith("python-runtime-"))
        matrix = name.startswith("python-")
        version = "${{ matrix.python }}" if matrix else "3.13"
        assert steps[index]["with"] == {
            "name": f"python-runtime-{version}",
            "path": "${{ runner.temp }}",
            "digest-mismatch": "error",
        }
        assert steps[index].get("if") == (None if matrix else "needs.plan.outputs.full == 'true'")
        assert steps[index + 1]["if"] == "needs.plan.outputs.full == 'true'"
        assert steps[index + 1]["name"] == "Activate installed Python runtime"
    upload = jobs["python-wheelhouse"]["steps"][-1]["with"]
    assert upload["name"] == "python-runtime-${{ matrix.python }}"
    assert upload["path"] == "${{ runner.temp }}/python-runtime.tar.gz"


def test_ci_runtime_installs_project_wheel_with_checkout_metadata_on_pythonpath(root: Path, tmp_path: Path) -> None:
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())["jobs"]
    build = next(step["run"] for step in jobs["python-wheelhouse"]["steps"]
                 if step.get("name") == "Build wheels and install the run-local Python runtime")
    install, verify, archive = build.splitlines()[-3:]
    assert "python-runtime.tar.gz" in archive
    _, *install_args, wheel_glob = shlex.split(install)
    _, *verify_args = shlex.split(verify)
    assert wheel_glob == "$RUNNER_TEMP/wheelhouse/*.whl"
    assert "--no-index" in install_args

    source = tmp_path / "src"
    egg = source / "baloot_platform.egg-info"
    egg.mkdir(parents=True)
    metadata = "Metadata-Version: 2.1\nName: baloot-platform\nVersion: 1.2.3\n"
    (egg / "PKG-INFO").write_text(metadata, encoding="utf-8")
    wheel = tmp_path / "baloot_platform-1.2.3-py3-none-any.whl"
    dist = "baloot_platform-1.2.3.dist-info"
    with ZipFile(wheel, "w") as archive:
        archive.writestr(f"{dist}/METADATA", metadata)
        archive.writestr(f"{dist}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{dist}/RECORD", "")

    runtime = tmp_path / "ci-venv"
    subprocess.run([sys.executable, "-I", "-m", "venv", str(runtime)], check=True, timeout=60)
    python = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    env = {**os.environ, "PYTHONPATH": str(source), "PIP_CONFIG_FILE": os.devnull,
           "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    missing = subprocess.run([python, *verify_args], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert missing.returncode and "PackageNotFoundError" in missing.stderr
    installed = subprocess.run([python, *install_args, wheel], cwd=tmp_path, env=env,
                               capture_output=True, text=True, timeout=30)
    assert installed.returncode == 0, installed.stdout + installed.stderr
    verified = subprocess.run([python, *verify_args], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert verified.returncode == 0, installed.stdout + verified.stderr
    assert verified.stdout.strip() == "1.2.3"


def test_workflows_use_node24_artifact_upload_action(root: Path) -> None:
    workflows = (root / ".github/workflows").glob("*.yml")
    sources = [path.read_text(encoding="utf-8") for path in workflows]
    uploads = [step["uses"] for source in sources
               for job in yaml.safe_load(source)["jobs"].values() for step in job.get("steps", ())
               if step.get("uses", "").startswith("actions/upload-artifact")]
    approved = {"actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9"}  # Node 24, v7.0.2
    assert uploads and set(uploads) <= approved


def test_upload_action_check_rejects_unapproved_sha_and_tags(tmp_path: Path) -> None:
    workflows = tmp_path / ".github/workflows"
    workflows.mkdir(parents=True)
    workflow = workflows / "probe.yml"
    for ref in ("0" * 40, "65c4c4a1ddee5b72f698fdd19549f0f0fb45cf08",
                "ea165f8d65b6e75b540449e92b4886f43607fa02", "v7"):
        workflow.write_text(f"jobs:\n  probe:\n    steps:\n      - uses: actions/upload-artifact@{ref}\n")
        with pytest.raises(AssertionError):
            test_workflows_use_node24_artifact_upload_action(tmp_path)


def test_workflows_pin_artifact_downloads(root: Path) -> None:
    downloads = [step["uses"] for path in (root / ".github/workflows").glob("*.yml")
                 for job in yaml.safe_load(path.read_text())["jobs"].values()
                 for step in job.get("steps", ()) if step.get("uses", "").startswith("actions/download-artifact@")]
    assert downloads and set(downloads) == {
        "actions/download-artifact@9000827ccba6bdab643e8b6fd33ac0654aef8333"  # Node 24, v8.0.2
    }


def test_package_and_openapi_share_the_authoritative_version(root: Path) -> None:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    openapi = json.loads((root / "api/openapi.json").read_text(encoding="utf-8"))
    assert {
        pyproject["project"]["version"],
        openapi["info"]["version"],
    } == {__version__}
