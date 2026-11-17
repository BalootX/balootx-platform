from copy import deepcopy
import json
import os
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from scripts import run_heavy_pytest as heavy


FILE = "tests/test_ortools_scenario_model.py"
BACKEND = "tests/test_research_ortools_backend_service.py"


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
    assert history["versionedNodeShardedFiles"] == [BACKEND]
    assert heavy.node_sharded_files(history) == {FILE}
    assert history["versionedNodeShards"] == 5
    assert set(history["nodeSeconds"]) == {"3.12", "3.13"}
    for version, weights in history["nodeSeconds"].items():
        nodes = sorted(weights)
        assert len(nodes) == 210 and {node.split("::", 1)[0] for node in nodes} == {BACKEND}
        rows = receipts(nodes, BACKEND, version, weights, 5)
        assert heavy.validate_union(rows, "head", [BACKEND], version, weights, 5)["executed"] == 210


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
    assert receipt["selected"] == nodes[1::4] and receipt["status"] == "passed"
    assert calls == [(["--collect-only", FILE], "collection", 300), (nodes[1::4], "execution", 5400)]
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

    def run(shard, output, selected_versioned):
        calls.append((shard, output, selected_versioned))
        return 0

    monkeypatch.setattr(heavy, "run_shard", run)
    assert heavy.main() == 0
    assert calls == [(3, tmp_path, versioned)]


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


def test_heavy_workflow_keeps_required_fast_names_and_checks_complete_union(root):
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())["jobs"]
    assert jobs["python-backend-run"]["strategy"]["matrix"] == {"shard": [1, 2, 3, 4, 5], "python": ["3.12", "3.13"]}
    assert jobs["python-fast-run"]["strategy"]["matrix"] == {"python": ["3.12", "3.13"], "shard": [1, 2, 3, 4]}
    lane = jobs["python-heavy"]
    assert lane["runs-on"] == "ubuntu-24.04"
    assert lane["strategy"] == {"fail-fast": False, "max-parallel": "${{ fromJSON(vars.BALOOT_CI_MAX_PARALLEL || '4') }}", "matrix": {"shard": [1, 2, 3, 4]}}
    assert lane["env"] == {"OMP_NUM_THREADS": "4", "CMAKE_BUILD_PARALLEL_LEVEL": "4"}
    setup = next(step for step in lane["steps"] if "setup-python@" in step.get("uses", ""))
    assert setup["with"]["python-version"] == "3.13"
    run = next(step for step in lane["steps"] if step.get("name") == "Run deterministic heavy node shard")
    assert run["timeout-minutes"] == 100 and "--shard" in run["run"]
    summary = jobs["python-heavy-summary"]
    assert summary["runs-on"] == "ubuntu-24.04"
    assert summary["needs"] == "python-heavy"
    assert summary["if"] == "always() && (github.event_name != 'pull_request' || !github.event.pull_request.draft)"
    download = next(step for step in summary["steps"] if "download-artifact@" in step.get("uses", ""))
    assert download["with"]["merge-multiple"] is False
    verify = next(step for step in summary["steps"] if " summary " in step.get("run", ""))
    assert verify["env"] == {"HEAVY_RESULT": "${{ needs.python-heavy.result }}"}
    guard, proof = verify["run"].splitlines()
    assert "scripts/run_heavy_pytest.py summary" in proof
    command = shlex.split(guard)
    assert command[:2] == ["python", "-c"]
    for status in ("success", "failure", "cancelled", "skipped", ""):
        result = subprocess.run([sys.executable, *command[1:]], env={**os.environ, "HEAVY_RESULT": status},
            capture_output=True, text=True, timeout=30)
        assert (result.returncode == 0) == (status == "success"), result.stderr


def test_required_fast_checks_gate_each_version_and_heavy_summary(root, tmp_path):
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())["jobs"]
    gate = jobs["python-fast"]
    assert gate["name"] == "python-fast (${{ matrix.python }})"
    assert gate["strategy"]["matrix"] == {"python": ["3.12", "3.13"]}
    assert gate["needs"] == ["python-backend-run", "python-fast-run", "python-heavy-summary"] and gate["if"] == "always()"
    assert gate["runs-on"] == "ubuntu-24.04"
    assert gate["timeout-minutes"] == 10
    worker = jobs["python-fast-run"]
    record = next(step for step in worker["steps"] if step.get("name") == "Record final fast lane result")
    file_condition = "always() && github.job != 'python-backend-run'"
    assert record["if"] == file_condition and record["env"] == {
        "FAST_RESULT": "${{ job.status }}", "FAST_PYTHON": "${{ matrix.python }}", "FAST_SHARD": "${{ matrix.shard }}"}
    upload = next(step for step in worker["steps"] if step.get("name") == "Upload final fast lane result")
    download, backend_download = [step for step in gate["steps"] if "download-artifact@" in step.get("uses", "")]
    # Uploads skip on cancelled jobs (403 "job is completed"); the result record still runs always().
    assert upload["if"] == "!cancelled() && github.job != 'python-backend-run'"
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["name"] == "python-fast-status-${{ matrix.python }}-${{ matrix.shard }}"
    assert download["with"]["pattern"] == "python-fast-status-${{ matrix.python }}-*"
    assert download["with"]["merge-multiple"] is False
    assert backend_download["with"] == {"pattern": "backend-node-results-${{ matrix.python }}-*",
                                         "merge-multiple": False, "path": "backend-result"}
    require = gate["steps"][-1]
    assert require["env"] == {"FAST_PYTHON": "${{ matrix.python }}",
                             "BACKEND_RESULT": "${{ needs.python-backend-run.result }}",
                             "FAST_RUN_RESULT": "${{ needs.python-fast-run.result }}",
                             "HEAVY_RESULT": "${{ needs.python-heavy-summary.result }}"}
    status_dir = tmp_path / "fast-result"
    status_dir.mkdir()
    from scripts.run_isolated_pytest import split_files
    files = [f"tests/test_sample_{index}.py" for index in range(12)]
    (tmp_path / "tests").mkdir()
    for path in files + [FILE, BACKEND]:
        (tmp_path / path).touch()
    (tmp_path / ".github").mkdir()
    backend_nodes = [f"{BACKEND}::test_case[{index:03d}]" for index in range(210)]
    history = {"seconds": {}, "incompleteFiles": [], "nodeShardedFiles": [FILE],
               "versionedNodeShardedFiles": [BACKEND], "versionedNodeShards": 5,
               "nodeSeconds": {"3.12": {node: index + 1 for index, node in enumerate(backend_nodes)},
                               "3.13": {node: (index % 17 + 1) ** 2 for index, node in enumerate(backend_nodes)}}}
    (tmp_path / ".github/python-test-durations.json").write_text(json.dumps(history))
    record_script = record["run"].split("\n", 1)[1].rsplit("\nPY", 1)[0]
    require_script = require["run"].split("\n", 1)[1].rsplit("\nPY", 1)[0]
    for version in ("3.12", "3.13"):
        env = {**os.environ, "GITHUB_SHA": "head", "FAST_PYTHON": version, "FAST_RESULT": "success",
               "BACKEND_RESULT": "success", "FAST_RUN_RESULT": "success", "RUNNER_TEMP": str(status_dir), "PYTHONPATH": str(root)}
        for index, group in enumerate(split_files(files, history), 1):
            results = status_dir / "python-file-results"
            results.mkdir(exist_ok=True)
            (results / "results.json").write_text(json.dumps([{"file": path} for path in group]))
            result = subprocess.run([sys.executable, "-c", record_script], env={**env, "FAST_SHARD": str(index)},
                capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stderr
            artifact = status_dir / f"python-fast-status-{version}-{index}"
            artifact.mkdir()
            (status_dir / "python-fast-status.json").replace(artifact / "python-fast-status.json")
        backend_dir = tmp_path / "backend-result"
        backend_dir.mkdir(exist_ok=True)
        backend_rows = receipts(backend_nodes, BACKEND, version, history["nodeSeconds"][version], 5)
        for row in backend_rows:
            artifact = backend_dir / f"backend-node-results-{version}-{row['shard']}"
            artifact.mkdir()
            (artifact / "receipt.json").write_text(json.dumps(row))
        status_file = status_dir / f"python-fast-status-{version}-1/python-fast-status.json"
        saved = json.loads(status_file.read_text())
        for name in ("BACKEND_RESULT", "FAST_RUN_RESULT", "HEAVY_RESULT"):
            for status in ("success", "failure", "cancelled", "skipped", ""):
                result = subprocess.run([sys.executable, "-c", require_script], cwd=tmp_path,
                    env={**env, "HEAVY_RESULT": "success", name: status}, capture_output=True, text=True, timeout=30)
                assert (result.returncode == 0) == (status == "success"), result.stderr
        backend_file = backend_dir / f"backend-node-results-{version}-1/receipt.json"
        backend_saved = json.loads(backend_file.read_text())
        for field, value in (("head", "stale"), ("python", "3.13.9" if version == "3.12" else "3.12.9"),
                             ("shard", 2), ("selected", []), ("files", [FILE]), ("status", "failed")):
            backend_file.write_text(json.dumps({**backend_saved, field: value}))
            result = subprocess.run([sys.executable, "-c", require_script], cwd=tmp_path,
                env={**env, "HEAVY_RESULT": "success"}, capture_output=True, text=True, timeout=30)
            assert result.returncode != 0
        backend_file.unlink()
        result = subprocess.run([sys.executable, "-c", require_script], cwd=tmp_path,
            env={**env, "HEAVY_RESULT": "success"}, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        backend_file.write_text(json.dumps(backend_saved))
        duplicate = backend_dir / "duplicate"
        duplicate.mkdir(exist_ok=True)
        (duplicate / "receipt.json").write_text(json.dumps(backend_saved))
        result = subprocess.run([sys.executable, "-c", require_script], cwd=tmp_path,
            env={**env, "HEAVY_RESULT": "success"}, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        (duplicate / "receipt.json").unlink()
        for field, value in (("result", "failure"), ("result", "cancelled"), ("result", "skipped"),
                             ("head", "old"), ("python", "other"), ("shard", 2), ("files", []), ("files", files)):
            status_file.write_text(json.dumps({**saved, field: value}))
            result = subprocess.run([sys.executable, "-c", require_script], cwd=tmp_path,
                env={**env, "HEAVY_RESULT": "success"}, capture_output=True, text=True, timeout=30)
            assert result.returncode != 0
        status_file.unlink()
        result = subprocess.run([sys.executable, "-c", require_script], cwd=tmp_path,
            env={**env, "HEAVY_RESULT": "success"}, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0  # A skipped shard without an artifact must never pass.
        for path in status_dir.glob("*/python-fast-status.json"):
            path.unlink()
        for path in backend_dir.glob("*/receipt.json"):
            path.unlink()
