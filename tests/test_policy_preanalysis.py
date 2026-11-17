from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def _module(root: Path):
    path = root / "scripts/benchmark_strength.py"
    spec = importlib.util.spec_from_file_location(
        "benchmark_strength",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_preanalysis_summary_is_aggregate_only(root: Path):
    module = _module(root)
    result = module.summarize(
        [
            {
                "archetype": "endgame",
                "hit": True,
                "regret": 0.0,
                "illegal": False,
                "latencyMs": 1.0,
                "decisionSource": "search",
            },
            {
                "archetype": "outcome_flip",
                "hit": False,
                "regret": 2.0,
                "illegal": False,
                "latencyMs": 3.0,
                "decisionSource": "heuristic",
            },
        ]
    )

    assert result["positions"] == 2
    assert result["top1Accuracy"] == 0.5
    assert result["expectedRegret"] == 1.0
    assert result["latencyMs"]["p50"] == 2.0
    assert "positionId" not in result
    assert "predictedAction" not in result


def test_policy_preanalysis_release_contract_is_honest(root: Path):
    module = _module(root)
    assert module.DEFAULT_OUTPUT.as_posix() == (
        "release/intelligence-benchmark.json"
    )


def test_policy_preanalysis_accepts_content_bound_local_source_identity(root: Path):
    assert "finalist_candidate" in (
        root / "scripts/benchmark_strength.py"
    ).read_text(encoding="utf-8")
    source = {
        "revision": "a" * 40,
        "tree": None,
        "workspaceManifestSha256": "b" * 64,
    }
    binding = (
        "exact_git_head"
        if source["revision"] and source["tree"]
        else "local_workspace_manifest"
        if source["workspaceManifestSha256"]
        else "unbound_test_invocation"
    )
    assert binding == "local_workspace_manifest"


def test_policy_preanalysis_accepts_a_valid_release_source_manifest(
    root: Path,
    tmp_path: Path,
):
    module = _module(root)
    path = tmp_path / "source-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "source": {
                    "binding": "git_head_plus_worktree_snapshot",
                    "revision": "a" * 40,
                    "tree": "b" * 40,
                    "worktreeSnapshot": {"sha256": "c" * 64},
                },
            }
        ),
        encoding="utf-8",
    )

    revision, manifest_sha256 = module.source_manifest_identity(path)

    assert revision == "a" * 40
    assert manifest_sha256 == "c" * 64


def test_local_finalist_manifest_is_deterministic_and_content_bound(root: Path):
    path = root / "scripts/run_local_finalist_preanalysis.py"
    spec = importlib.util.spec_from_file_location("local_finalist_preanalysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = (root / "configs/benchmark/strength.yaml",)
    first = module.build_workspace_manifest(root, extra_paths=inputs)
    second = module.build_workspace_manifest(root, extra_paths=inputs)
    assert first == second
    assert len(first["sha256"]) == 64
    assert first["proofBoundary"].startswith("Content-bound local")
    assert first["sourceRevisionKind"] in {"git_head", "workspace_content"}
    if first["sourceRevisionKind"] == "workspace_content":
        assert first["sourceRevision"].startswith("workspace:")


def test_local_finalist_manifest_excludes_generated_dependency_trees(
    root: Path, tmp_path: Path
):
    path = root / "scripts/run_local_finalist_preanalysis.py"
    spec = importlib.util.spec_from_file_location("local_finalist_source_pruning", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    (tmp_path / "src/baloot_platform/node_modules/pkg").mkdir(parents=True)
    (tmp_path / "configs/benchmark").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src/baloot_platform/owned.py").write_text("OWNED = True\n")
    (tmp_path / "src/baloot_platform/node_modules/pkg/package.json").write_text(
        "{}\n"
    )
    (tmp_path / "configs/benchmark/strength.yaml").write_text("schemaVersion: 1\n")
    (tmp_path / "scripts/benchmark_strength.py").write_text("# owned\n")
    (tmp_path / "scripts/run_local_finalist_preanalysis.py").write_text("# owned\n")

    manifest = module.build_workspace_manifest(
        tmp_path,
        extra_paths=(tmp_path / "configs/benchmark/strength.yaml",),
    )
    paths = {entry["path"] for entry in manifest["files"]}
    assert "src/baloot_platform/owned.py" in paths
    assert not any("node_modules" in candidate.split("/") for candidate in paths)


def test_local_finalist_manifest_is_content_bound_without_git(root: Path, monkeypatch):
    path = root / "scripts/run_local_finalist_preanalysis.py"
    spec = importlib.util.spec_from_file_location("local_finalist_preanalysis_archive", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=128, stdout="", stderr="not a repository"),
    )
    manifest = module.build_workspace_manifest(
        root,
        extra_paths=(root / "configs/benchmark/strength.yaml",),
    )
    assert manifest["sourceRevisionKind"] == "workspace_content"
    assert manifest["sourceRevision"].startswith("workspace:")
    assert len(manifest["sourceRevision"].removeprefix("workspace:")) == 64


def test_local_finalist_manifest_is_content_bound_when_git_is_unavailable(root: Path, monkeypatch):
    path = root / "scripts/run_local_finalist_preanalysis.py"
    spec = importlib.util.spec_from_file_location("local_finalist_preanalysis_no_git", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("git")),
    )
    manifest = module.build_workspace_manifest(
        root,
        extra_paths=(root / "configs/benchmark/strength.yaml",),
    )
    assert manifest["sourceRevisionKind"] == "workspace_content"
    assert manifest["sourceRevision"].startswith("workspace:")
