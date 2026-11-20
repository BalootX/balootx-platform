from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

import yaml
import pytest

from baloot_platform.version import __version__




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
        assert jobs[name]["needs"] == ["plan", "python-wheelhouse", "native-build", "ffmpeg-packages"]
    downloads = {step["with"]["name"] for step in steps if "download-artifact@" in step.get("uses", "")}
    assert downloads == {"python-wheels-${{ matrix.python }}", "native-release-library", "ffmpeg-debs-6.1.1-3ubuntu5"}
    assert all(step.get("with", {}).get("digest-mismatch") == "error" for step in steps
               if "download-artifact@" in step.get("uses", ""))
    foundry = jobs["foundry"]["steps"]
    assert any(step.get("run") == "npm run test:research" for step in foundry)
    assert installs <= {step.get("working-directory") for step in foundry
                        if step.get("run", "").startswith("npm ci ")}
    assert jobs["media-render"]["strategy"]["matrix"]["shard"] == ["flagship", "decision", "companion", "parity"]
    assert jobs["media-release-smoke"]["needs"] == ["plan", "media-render"]
    assert any(step.get("name") == "Verify governed font hashes" for step in jobs["media-inputs"]["steps"])


def test_workflows_use_node24_artifact_upload_action(root: Path) -> None:
    workflows = (root / ".github/workflows").glob("*.yml")
    sources = [path.read_text(encoding="utf-8") for path in workflows]
    uploads = [step["uses"] for source in sources
               for job in yaml.safe_load(source)["jobs"].values() for step in job["steps"]
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
                 for step in job["steps"] if step.get("uses", "").startswith("actions/download-artifact@")]
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
