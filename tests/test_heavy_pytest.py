from collections import Counter
from copy import deepcopy
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from scripts import ci_scope
from scripts import run_heavy_pytest as heavy
from scripts import run_isolated_pytest as isolated


FILE = "tests/test_ortools_scenario_model.py"
BACKEND = "tests/test_research_ortools_backend_service.py"
UNIFIED = "tests/test_research_ortools_unified_executor.py"


def execution(nodes):
    return {"exitstatus": 0, "collected": nodes, "deselected": [], "collectionIssues": [],
            "reports": {node: [{"phase": phase, "outcome": "passed", "xfail": False}
                                for phase in ("setup", "call", "teardown")] for node in nodes}}


def receipts(nodes, file=FILE, python="3.13", seconds=None, shards=4):
    return [{"shard": shard, "head": "head", "python": f"{python}.9", "files": [file],
             "status": "passed", "collected": list(nodes),
             "selected": heavy.split_nodes(nodes, shard, seconds, shards),
             "execution": execution(heavy.split_nodes(nodes, shard, seconds, shards))}
            for shard in range(1, shards + 1)]


def test_node_shards_are_deterministic_disjoint_and_exhaustive():
    nodes = sorted(f"{FILE}::test_case[{index}]" for index in range(579))
    shards = [heavy.split_nodes(list(reversed(nodes)), shard) for shard in range(1, 5)]
    assert [len(shard) for shard in shards] == [145, 145, 145, 144]
    assert all(shard == nodes[index::4] for index, shard in enumerate(shards))
    assert sorted(node for shard in shards for node in shard) == nodes
    assert len(set(node for shard in shards for node in shard)) == 579
    assert heavy.validate_union(receipts(nodes), "head", [FILE])["executed"] == 579
    for invalid, shard in (([], 1), (nodes * 2, 1), (nodes, 0), (nodes, 5)):
        with pytest.raises(ValueError):
            heavy.split_nodes(invalid, shard)


def test_failed_traceback_is_persisted_before_session_finish(tmp_path, monkeypatch):
    target = tmp_path / "execution.json"
    with monkeypatch.context() as patch:
        patch.setenv("BALOOT_CI_NODE_RECEIPT", str(target))
        patch.setattr(heavy, "_nodes", {"collected": [], "deselected": [], "collectionIssues": [], "reports": {}})
        heavy.pytest_runtest_logreport(SimpleNamespace(nodeid="test_sample::test_failure", when="call",
            outcome="failed", duration=1.2, longrepr="AssertionError: normalization_incomplete"))
    assert not target.exists()  # Session finalization may never run after a timeout.
    event = json.loads(target.with_suffix(".jsonl").read_text())
    assert event["nodeid"] == "test_sample::test_failure"
    assert event["longrepr"] == "AssertionError: normalization_incomplete"
    assert event["outcome"] == "failed" and event["phase"] == "call"


def test_summary_rejects_missing_extra_failed_skipped_or_stale_execution():
    nodes = [f"{FILE}::test_case[{index}]" for index in range(8)]
    original = receipts(nodes)
    corruptions = (
        lambda rows: rows.pop(),
        lambda rows: rows.__setitem__(3, deepcopy(rows[0])),
        lambda rows: rows[0]["execution"]["reports"].pop(nodes[0]),
        lambda rows: rows[0]["execution"]["reports"].__setitem__("extra", []),
        lambda rows: rows[0]["execution"]["reports"][nodes[0]][0].__setitem__("outcome", "skipped"),
        lambda rows: rows[0]["execution"]["reports"][nodes[0]][2].__setitem__("outcome", "failed"),
        lambda rows: rows[0]["execution"]["reports"][nodes[0]][1].__setitem__("xfail", True),
        lambda rows: rows[0]["execution"]["reports"][nodes[0]].append({"phase": "call", "outcome": "passed", "xfail": False}),
        lambda rows: rows[0]["execution"]["deselected"].append(nodes[0]),
        lambda rows: rows[0]["execution"]["collectionIssues"].append({"outcome": "skipped"}),
        lambda rows: rows[0]["execution"].__setitem__("exitstatus", 1),
        lambda rows: rows[0].__setitem__("head", "stale"),
        lambda rows: rows[0].__setitem__("python", "3.12.15"),
        lambda rows: rows[0]["selected"].pop(),
        lambda rows: rows[0]["collected"].pop(),
    )
    for corrupt in corruptions:
        rows = deepcopy(original)
        corrupt(rows)
        with pytest.raises(ValueError):
            heavy.validate_union(rows, "head", [FILE])


@pytest.mark.parametrize("python", ["3.12", "3.13"])
def test_weighted_backend_union_preserves_every_node_and_rejects_corruption(python):
    nodes = [f"{BACKEND}::test_case[{index:03d}]" for index in range(210)]
    seconds = {node: (index % 17 + 1) ** 2 for index, node in enumerate(nodes)}
    original = receipts(nodes, BACKEND, python, seconds, 5)
    assert heavy.validate_union(original, "head", [BACKEND], python, seconds, 5)["executed"] == 210
    groups = [row["selected"] for row in original]
    assert sorted(node for group in groups for node in group) == nodes
    assert all(group == heavy.split_nodes(list(reversed(nodes)), index, seconds, 5)
               for index, group in enumerate(groups, 1))
    loads = [sum(seconds[node] for node in group) for group in groups]
    assert max(loads) - min(loads) <= max(seconds.values())
    assert any(group != nodes[index::5] for index, group in enumerate(groups))
    expanded = sorted([*nodes, f"{BACKEND}::test_new_case"])
    assert heavy.validate_union(receipts(expanded, BACKEND, python, seconds, 5),
                                "head", [BACKEND], python, seconds, 5)["executed"] == 211
    first = groups[0][0]
    corruptions = (
        lambda rows: rows.pop(),
        lambda rows: rows.append(deepcopy(rows[0])),
        lambda rows: rows.__setitem__(4, deepcopy(rows[0])),
        lambda rows: rows[0].__setitem__("head", "stale"),
        lambda rows: rows[0].__setitem__("python", "3.13.9" if python == "3.12" else "3.12.9"),
        lambda rows: rows[0].__setitem__("files", [FILE]),
        lambda rows: rows[0].__setitem__("status", "failed"),
        lambda rows: rows[0]["collected"].pop(),
        lambda rows: rows[0]["selected"].pop(),
        lambda rows: rows[0]["execution"]["reports"].pop(first),
        lambda rows: rows[0]["execution"]["reports"].__setitem__("extra", []),
        lambda rows: rows[0]["execution"]["reports"][first][1].__setitem__("outcome", "skipped"),
        lambda rows: rows[0]["execution"]["reports"][first][1].__setitem__("xfail", True),
    )
    for corrupt in corruptions:
        rows = deepcopy(original)
        corrupt(rows)
        with pytest.raises(ValueError):
            heavy.validate_union(rows, "head", [BACKEND], python, seconds, 5)
    for value in (-1, float("nan"), float("inf"), "1"):
        with pytest.raises(ValueError, match="finite nonnegative"):
            heavy.split_nodes(nodes, 1, {nodes[0]: value}, 5)


def test_backend_profile_retains_both_versions_and_legacy_heavy_scope():
    history = json.loads((heavy.ROOT / ".github/python-test-durations.json").read_text())
    expected = {
        BACKEND: 210, UNIFIED: 745,
        "tests/test_research_foundry.py": 29,
        "tests/test_research_ortools_information_set_proof.py": 342,
        "tests/test_research_ortools_nonplay_complete_graph.py": 177,
        "tests/test_research_ortools_real_executor_predicate_matrix.py": 429,
    }
    assert set(history["versionedNodeShardedFiles"]) == expected.keys()
    assert heavy.node_sharded_files(history) == {FILE}
    assert history["versionedNodeShards"] == 4
    assert set(history["nodeSeconds"]) == {"3.12", "3.13"}
    for weights in history["nodeSeconds"].values():
        assert Counter(node.split("::", 1)[0] for node in weights) == expected
    assert len(history["heavyNodeSeconds"]) == 582
    assert all(node.startswith(f"{FILE}::") for node in history["heavyNodeSeconds"])


def test_shard_parent_uses_separate_collection_and_execution_with_90min_limit(tmp_path, monkeypatch):
    nodes = sorted(f"{FILE}::test_case[{index}]" for index in range(8))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(16)), raising=False)
    affinity = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda _, cpus: affinity.append(cpus), raising=False)
    calls = []
    stale = [tmp_path / "receipt.json", *(tmp_path / f"{name}.{extension}"
             for name in ("collection", "execution") for extension in ("json", "jsonl", "log"))]
    for path in stale:
        path.write_text("stale shard output")

    def run(arguments, output, name, seconds):
        assert not any(path.exists() for path in stale)
        calls.append((arguments, name, seconds))
        return 0, execution(nodes if name == "collection" else arguments)

    monkeypatch.setattr(heavy, "_pytest", run)
    assert heavy.run_shard(2, tmp_path) == 0
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["selected"] == nodes[1::2] and receipt["status"] == "passed"
    assert calls == [(["--collect-only", FILE], "collection", 300), (nodes[1::2], "execution", 5400)]
    assert affinity == [[0, 1, 2, 3]]


@pytest.mark.parametrize("python", ["3.12", "3.13"])
def test_versioned_backend_parent_uses_weighted_nodes_and_unchanged_3600s_cap(tmp_path, monkeypatch, python):
    nodes = [f"{BACKEND}::test_case[{index:03d}]" for index in range(210)]
    weights = {node: index + 1 for index, node in enumerate(nodes)}
    history = {"seconds": {}, "versionedNodeShardedFiles": [BACKEND], "versionedNodeShards": 5,
               "nodeSeconds": {python: weights}}
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github/python-test-durations.json").write_text(json.dumps(history))
    monkeypatch.setattr(heavy, "ROOT", tmp_path)
    monkeypatch.setattr(heavy.platform, "python_version", lambda: f"{python}.9")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(16)), raising=False)
    affinity = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda _, cpus: affinity.append(cpus), raising=False)
    calls = []

    def run(arguments, output, name, seconds):
        calls.append((arguments, name, seconds))
        return 0, execution(nodes if name == "collection" else arguments)

    monkeypatch.setattr(heavy, "_pytest", run)
    output = tmp_path / "backend"
    assert heavy.run_shard(5, output, versioned=True) == 0
    selected = heavy.split_nodes(nodes, 5, weights, 5)
    assert calls == [(["--collect-only", BACKEND], "collection", 300), (selected, "execution", 3600)]
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["files"] == [BACKEND] and receipt["python"] == f"{python}.9"
    assert receipt["selected"] == selected and receipt["status"] == "passed"
    assert affinity == [[0, 1, 2, 3]]


@pytest.mark.parametrize("versioned", [False, True])
def test_cli_selects_only_the_requested_node_profile(tmp_path, monkeypatch, versioned):
    monkeypatch.setattr(sys, "argv", ["run_heavy_pytest.py", "run", "--shard", "3", "--output", str(tmp_path),
                                      *(["--versioned"] if versioned else [])])
    calls = []

    def run(shard, output, selected_versioned, *, workers):
        calls.append((shard, output, selected_versioned, workers))
        return 0

    monkeypatch.setattr(heavy, "run_shard", run)
    assert heavy.main() == 0
    assert calls == [(3, tmp_path, versioned, 1)]


@pytest.mark.parametrize("name", ["collection", "execution"])
def test_pytest_subprocess_cannot_reuse_stale_receipts(tmp_path, monkeypatch, name):
    monkeypatch.setenv("BALOOT_PYTEST_FILE_BUDGET", "1")
    target = tmp_path / f"{name}.json"
    target.write_text(json.dumps(execution(["stale::test_pass"])))
    target.with_suffix(".jsonl").write_text("stale phase")
    (tmp_path / f"{name}.log").write_text("stale log")

    def run(command, **kwargs):
        assert not target.exists() and not target.with_suffix(".jsonl").exists()
        assert (tmp_path / f"{name}.log").stat().st_size == 0
        assert kwargs["env"]["BALOOT_PYTEST_FILE_BUDGET"] == "0"
        return SimpleNamespace(returncode=0)  # No new receipt: old PASS must not survive.

    monkeypatch.setattr(heavy.subprocess, "run", run)
    assert heavy._pytest([FILE], tmp_path, name, 300) == (0, None)


def test_real_pytest_plugin_records_collection_and_rejects_skips(tmp_path):
    test = tmp_path / "test_small.py"
    test.write_text("import pytest\n@pytest.mark.parametrize('value', [1, 2])\ndef test_pass(value): assert value > 0\n@pytest.mark.skip(reason='negative fixture')\ndef test_skip(): pass\n")
    env = {**os.environ, "PYTHONPATH": str(heavy.ROOT), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""}
    command = [sys.executable, "-m", "pytest", "-q", "--override-ini=addopts=", "-p", "scripts.run_heavy_pytest",
               f"--rootdir={tmp_path}", f"--confcutdir={tmp_path}", f"--override-ini=cache_dir={tmp_path}/cache"]
    collected = tmp_path / "collection.json"
    result = subprocess.run([*command, "--collect-only", str(test)], cwd=tmp_path, env={**env, "BALOOT_CI_NODE_RECEIPT": str(collected)}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    nodes = json.loads(collected.read_text())["collected"]
    assert len(nodes) == 3
    passing = sorted(node for node in nodes if "::test_pass[" in node)
    positive = tmp_path / "passing.json"
    result = subprocess.run([*command, *passing], cwd=tmp_path, env={**env, "BALOOT_CI_NODE_RECEIPT": str(positive)}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    heavy.validate_execution(passing, json.loads(positive.read_text()))
    target = tmp_path / "execution.json"
    result = subprocess.run([*command, str(test)], cwd=tmp_path, env={**env, "BALOOT_CI_NODE_RECEIPT": str(target)}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0  # Pytest itself permits skips; the lane must not.
    with pytest.raises(ValueError, match="without skipping"):
        heavy.validate_execution(nodes, json.loads(target.read_text()))
    assert target.with_suffix(".jsonl").exists()


@pytest.mark.parametrize("fail_worker", [False, True])
def test_four_worker_shard_proves_exact_union(tmp_path, monkeypatch, fail_worker):
    nodes = sorted(f"{FILE}::test_case[{index}]" for index in range(16))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(4)), raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", lambda *_: None, raising=False)
    calls = {}

    def run(arguments, output, name, seconds):
        calls[name] = arguments
        if name == "execution-2" and fail_worker:
            return 1, None
        return 0, execution(nodes if name == "collection" else arguments)

    monkeypatch.setattr(heavy, "_pytest", run)
    assert heavy.run_shard(1, tmp_path, workers=4) == int(fail_worker)
    receipt = json.loads((tmp_path / "receipt.json").read_bytes())
    assert sorted(calls) == ["collection", "execution-1", "execution-2", "execution-3", "execution-4"]
    assert sorted(node for name, args in calls.items() if name != "collection" for node in args) == receipt["selected"]
    if fail_worker:
        assert receipt["status"] == "failed" and "worker 2" in receipt["error"]
    else:
        assert receipt["status"] == "passed"
        heavy.validate_execution(receipt["selected"], receipt["execution"])


def test_full_workflow_requires_coverage_for_all_matrices(root):
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())["jobs"]
    versions = """${{ fromJSON(github.event_name == 'pull_request' && '["3.13"]' || '["3.12","3.13"]') }}"""
    assert jobs["python-wheelhouse"]["strategy"]["matrix"] == {"python": versions}
    assert jobs["python-backend-run"]["strategy"]["matrix"] == {"shard": [1, 2, 3, 4], "python": versions}
    assert jobs["python-fast-run"]["strategy"]["matrix"] == {"python": versions, "shard": [1, 2, 3]}
    assert jobs["python-heavy"]["strategy"]["matrix"] == {"python": ["3.13"], "shard": [1, 2]}
    for name in ("python-backend-run", "python-fast-run", "python-heavy"):
        assert jobs[name]["strategy"]["fail-fast"] is False
        assert "max-parallel" not in jobs[name]["strategy"]
        assert any("--workers 4" in step.get("run", "") for step in jobs[name]["steps"])
    assert jobs["python-coverage"]["needs"] == ["plan", "python-fast-run", "python-backend-run", "python-heavy"]
    base_step = jobs["python-fast-run"]["steps"][1]
    assert base_step["if"] == "github.event_name == 'merge_group'"
    assert base_step["env"] == {"MERGE_GROUP_BASE": "${{ github.event.merge_group.base_sha }}"}
    assert 'git fetch --no-tags --depth=1 origin "$MERGE_GROUP_BASE"' in base_step["run"]
    assert all(jobs[name]["steps"][1] is base_step for name in ("python-backend-run", "python-heavy", "python-coverage"))
    assert jobs["python-coverage"]["steps"][-1]["run"].startswith("python -P scripts/ci_scope.py coverage ")


@pytest.mark.parametrize("event", ["push", "pull_request", "merge_group"])
def test_coverage_rejects_stale_version_missing_and_duplicate_receipts(tmp_path, monkeypatch, capsys, event):
    head_sha = "b" * 40
    files = [f"tests/test_sample_{index}.py" for index in range(12)]
    monkeypatch.setattr(ci_scope, "fast_files", lambda _history: files)
    backend_nodes = [f"{BACKEND}::test_case[{index:03d}]" for index in range(15)]
    heavy_nodes = [f"{FILE}::test_case[{index:03d}]" for index in range(12)]
    weights = {node: index + 1 for index, node in enumerate(backend_nodes)}
    heavy_weights = {node: index + 1 for index, node in enumerate(heavy_nodes)}
    history = {"seconds": {}, "incompleteFiles": [], "nodeShardedFiles": [FILE],
               "versionedNodeShardedFiles": [BACKEND], "versionedNodeShards": 4,
               "nodeSeconds": {version: weights for version in ci_scope.VERSIONS},
               "heavyNodeSeconds": heavy_weights}

    def write(name, filename, value):
        target = tmp_path / name / filename
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(value))
        return target

    versions = ("3.13",) if event == "pull_request" else ci_scope.VERSIONS
    for version in versions:
        for index, group in enumerate(isolated.split_files(files, history), 1):
            write(f"python-fast-run-{version}-{index}", "python-fast-status.json",
                  {"head": head_sha, "python": version, "result": "success", "shard": index, "files": group})
        for row in receipts(backend_nodes, BACKEND, version, weights, 4):
            write(f"python-backend-run-{version}-{row['shard']}", "receipt.json", {**row, "head": head_sha})
    for row in receipts(heavy_nodes, FILE, "3.13", heavy_weights, 2):
        write(f"python-heavy-3.13-{row['shard']}", "receipt.json", {**row, "head": head_sha})

    history_file = tmp_path / "history.json"
    history_file.write_text(json.dumps(history), encoding="utf-8")
    monkeypatch.setattr(ci_scope, "HISTORY", history_file)
    monkeypatch.setattr(ci_scope.sys, "argv", ["ci_scope.py", "coverage", str(tmp_path)])
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    monkeypatch.setenv("GITHUB_SHA", head_sha)
    ci_scope.main()
    coverage = json.loads(capsys.readouterr().out)
    assert all(value["head"] == head_sha for value in coverage.values())
    assert {key: value["executed"] for key, value in coverage.items()} == {
        **{f"backend-{version}": 15 for version in versions}, "heavy": 12,
    }
    if event == "pull_request":
        with pytest.raises(ValueError):
            ci_scope.validate_coverage(tmp_path, head_sha, history, "push")
        stray = write("python-fast-run-3.12-1", "python-fast-status.json", {})
        with pytest.raises(ValueError, match="unexpected Python version"):
            ci_scope.validate_coverage(tmp_path, head_sha, history, event)
        stray.unlink()
    for name, filename, changes in (
        (f"python-fast-run-{versions[0]}-1", "python-fast-status.json",
         {"head": "stale", "python": "3.11", "shard": 2, "files": [], "result": "cancelled"}),
        ("python-backend-run-3.13-1", "receipt.json",
         {"head": "stale", "python": "3.12.9", "shard": 2, "selected": [], "status": "failed", "files": [FILE]}),
        ("python-heavy-3.13-1", "receipt.json", {"status": "failed"}),
    ):
        path = tmp_path / name / filename
        saved = json.loads(path.read_bytes())
        for field, value in changes.items():
            write(name, filename, {**saved, field: value})
            with pytest.raises(ValueError):
                ci_scope.validate_coverage(tmp_path, head_sha, history, event)
        path.unlink()
        with pytest.raises(ValueError):
            ci_scope.validate_coverage(tmp_path, head_sha, history, event)
        write(name, filename, saved)
        duplicate = write(name + "-duplicate", filename, saved)
        with pytest.raises(ValueError):
            ci_scope.validate_coverage(tmp_path, head_sha, history, event)
        duplicate.unlink()
