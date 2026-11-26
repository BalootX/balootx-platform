from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

from scripts import run_isolated_pytest as lane


@pytest.mark.parametrize("source,expected", [
    ("def test_nested():\n from ortools.sat.python import cp_model\n", True),
    ("from baloot_platform.research import service\n", True),
    ("from baloot_platform.analysis.search import SearchLimits\n", True),
    ("import json\n", False),
    ("text = 'import ortools'\n", False),
])
def test_solver_work_default_selects_imports(tmp_path, source, expected):
    from deterministic_work import solver_test_module

    module = tmp_path / "test_sample.py"
    module.write_text(source, encoding="utf-8")
    assert solver_test_module(module) is expected


def test_renderer_receipts_and_image_parity_run_without_loading_the_solver(root, tmp_path, capsys):
    probe = """
import json
import sys
from time import perf_counter

import pytest

class CollectionEnd:
    def pytest_collection_finish(self, session):
        self.finished = perf_counter()

collection = CollectionEnd()
started = perf_counter()
exit_code = pytest.main(sys.argv[1:], plugins=[collection])
if exit_code:
    raise SystemExit(exit_code)
assert "ortools" not in sys.modules, "renderer receipts or image parity imported OR-Tools"
print(json.dumps({
    "setup_and_collection_seconds": collection.finished - started,
    "execution_and_cleanup_seconds": perf_counter() - collection.finished,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, "--basetemp", str(tmp_path / "pytest"),
         "tests/test_research_notice.py", "tests/test_remotion_image_parity.py"],
        cwd=root, env=lane.isolated_environment(str(tmp_path / "runtime")),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with capsys.disabled():
        print("Renderer receipt and parity timings:", result.stdout.splitlines()[-1])


@pytest.mark.parametrize("elapsed,expected", [(0, 600), (500, 340), (839, 1), (840, None)])
def test_solver_file_budget_never_resets(monkeypatch, elapsed, expected):
    import deterministic_work as work

    monkeypatch.setitem(work._REAL, "monotonic", lambda: 1000 + elapsed)
    if expected is None:
        with pytest.raises(TimeoutError, match="840-second"):
            work.file_work_safety_seconds(1000)
    else:
        assert work.file_work_safety_seconds(1000) == expected


@pytest.mark.parametrize("timeout,seconds,shard_seconds,expected", [
    (900, 0, None, 840), (3600, 100, 120, 840), (900, 839, None, 840),
    (900, 840, None, 0), (3600, 900.087, 935.58, 0),
    (3600, 1489.03, 5.497, 0), (900, 100, 840, 0), (300, 0, None, 0),
])
def test_work_file_budget_uses_timeout_and_both_duration_histories(timeout, seconds, shard_seconds, expected):
    history = {"seconds": {"sample": seconds},
               "shardSeconds": {} if shard_seconds is None else {"sample": shard_seconds}}
    assert lane.work_file_budget("sample", timeout, history) == expected


def test_unified_executor_history_disables_file_cap_even_with_job_time():
    history = json.loads((lane.ROOT / ".github/python-test-durations.json").read_text())
    path = "tests/test_research_ortools_unified_executor.py"
    assert dict(lane.schedule([path], history))[path] is True
    assert history["shardSeconds"][path] >= lane.FILE_SAFETY_SECONDS
    assert lane.work_file_budget(path, 1320, history) == 0
    for invalid in (-1, float("inf"), float("nan"), "unknown", True):
        with pytest.raises(ValueError, match="finite nonnegative"):
            lane.work_file_budget("sample", 900, {"shardSeconds": {"sample": invalid}})


def test_solver_default_and_real_clock_scopes_restore_each_other(tmp_path):
    module = tmp_path / "test_clock_scopes.py"
    module.write_text('''import time
import pytest
from ortools.sat.python import cp_model
from baloot_platform.research import ortools_scenario_model as scenario

original_solve = cp_model.CpSolver.solve.__wrapped__

def test_default_before(request):
    assert request.getfixturevalue("deterministic_work") is not None
    assert scenario.time is not time
    assert cp_model.CpSolver.solve is not original_solve

@pytest.mark.real_clock
def test_real_clock(request):
    assert request.getfixturevalue("deterministic_work") is None
    assert scenario.time is time
    assert cp_model.CpSolver.solve is original_solve

def test_default_after(request):
    assert request.getfixturevalue("deterministic_work") is not None
    assert scenario.time is not time
    assert cp_model.CpSolver.solve is not original_solve
''', encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(root), str(root / "tests"), str(root / "src"))),
        "BALOOT_ROOT": str(root),
        "BALOOT_DATA_DIR": str(tmp_path / "runtime"),
        "BALOOT_RUNTIME": str(tmp_path / "runtime"),
        "BALOOT_OUTPUTS": str(tmp_path / "runtime" / "outputs"),
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "conftest", "-q", str(module),
         "-o", "markers=real_clock: preserve real timing"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("file_budget", [None, "0", "1"])
def test_solver_file_budget_is_opt_in_and_starts_before_module_import(tmp_path, file_budget):
    (tmp_path / "collection_probe.py").write_text('''import pytest
from conftest import _WORK_FILE_STARTED

@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_make_collect_report(collector):
    if isinstance(collector, pytest.Module):
        assert _WORK_FILE_STARTED in collector.stash
        collector.stash[_WORK_FILE_STARTED] -= 1050
    return (yield)
''', encoding="utf-8")
    module = tmp_path / "test_aged_module.py"
    module.write_text('''import pytest
from conftest import _WORK_FILE_STARTED
from baloot_platform.research import ortools_scenario_model

@pytest.fixture(scope="module", autouse=True)
def elapsed_module_setup(request):
    assert _WORK_FILE_STARTED in request.node.stash

@pytest.mark.real_clock
def test_real_first(deterministic_work):
    assert deterministic_work is None

def test_default_after(deterministic_work):
    assert deterministic_work.safety_seconds == 600
''', encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(root), str(root / "tests"), str(root / "src"))),
        "BALOOT_ROOT": str(root),
        "BALOOT_DATA_DIR": str(tmp_path / "runtime"),
        "BALOOT_RUNTIME": str(tmp_path / "runtime"),
        "BALOOT_OUTPUTS": str(tmp_path / "runtime" / "outputs"),
        "BALOOT_CI_NODE_RECEIPT": str(tmp_path / "node-receipt.json"),
    }
    environment.pop("BALOOT_PYTEST_FILE_BUDGET", None)
    if file_budget is not None:
        environment["BALOOT_PYTEST_FILE_BUDGET"] = file_budget
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "conftest", "-p", "scripts.run_heavy_pytest",
         "-p", "collection_probe",
         "-q", str(module), "-o", "markers=real_clock: preserve real timing"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == (1 if file_budget == "1" else 0), result.stdout + result.stderr
    if file_budget == "1":
        assert "840-second safety budget" in result.stdout


@pytest.mark.parametrize("collect_only", [False, True])
def test_solver_collection_solves_use_work_and_restore_captured_calls(tmp_path, collect_only):
    (tmp_path / "clock_probe.py").write_text('''from functools import wraps
from ortools.sat.python import cp_model

expect_work = True
original = cp_model.CpSolver.solve

@wraps(original)
def checked_solve(self, model, solution_callback=None):
    if expect_work:
        assert self.parameters.max_deterministic_time == 0.01
        assert 10 < self.parameters.max_time_in_seconds <= 600
    else:
        assert self.parameters.max_time_in_seconds == 0.01
        assert self.parameters.max_deterministic_time == float("inf")
        return cp_model.OPTIMAL  # This branch probes delegation, without a wall-limited solve.
    return original(self, model, solution_callback)

cp_model.CpSolver.solve = checked_solve
''', encoding="utf-8")
    module = tmp_path / "test_collection_solve.py"
    module.write_text('''import pytest
import clock_probe
from ortools.sat.python import cp_model

model = cp_model.CpModel()
model.new_bool_var("import_time")
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 0.01
captured_solve = cp_model.CpSolver.solve
assert solver.solve(model) == cp_model.OPTIMAL

@pytest.mark.real_clock
def test_real_clock_after_collection():
    clock_probe.expect_work = False
    assert cp_model.CpSolver.solve is clock_probe.checked_solve
    assert captured_solve(solver, model) == cp_model.OPTIMAL
''', encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(root), str(root / "tests"), str(root / "src"))),
        "BALOOT_ROOT": str(root),
        "BALOOT_DATA_DIR": str(tmp_path / "runtime"),
        "BALOOT_RUNTIME": str(tmp_path / "runtime"),
        "BALOOT_OUTPUTS": str(tmp_path / "runtime" / "outputs"),
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "clock_probe", "-p", "conftest", "-q",
         *(["--collect-only"] if collect_only else []), str(module),
         "-o", "markers=real_clock: preserve real timing"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("timeout", [300, 900])
def test_parallel_files_keep_all_failures_timeouts_and_runtime_isolation(tmp_path, monkeypatch, capsys, timeout):
    files = [f"tests/test_sample_{index}.py" for index in range(8)]
    for path in files:
        target = tmp_path / path
        target.parent.mkdir(exist_ok=True)
        target.touch()
    (tmp_path / ".github").mkdir()
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci/expected-pytest-skips.txt").write_text("")
    (tmp_path / ".github/python-test-durations.json").write_text(json.dumps({
        "seconds": {files[6]: 2400, files[5]: 1489, files[0]: 100},
        "shardSeconds": {files[4]: 935.58},
        "incompleteFiles": [files[6], files[5]],
    }))
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    monkeypatch.setattr(lane, "job_seconds_remaining", lambda _: 1320)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(16)), raising=False)
    affinities = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda _, cpus: affinities.append(cpus), raising=False)
    barrier, lock = Barrier(4), Lock()
    calls = []

    def execute(command, **kwargs):
        with lock:
            calls.append((command, kwargs))
        barrier.wait(timeout=10)  # Four independent subprocesses may proceed together.
        node = command[-1] + "::test_sample"
        expected_budget = int(command[3]) >= 900 and command[-1] not in {files[4], files[5], files[6]}
        assert kwargs["env"]["BALOOT_PYTEST_FILE_BUDGET"] == ("1" if expected_budget else "0")
        Path(kwargs["env"]["BALOOT_CI_NODE_RECEIPT"]).write_text(json.dumps({
            "exitstatus": 0, "deselected": [], "collectionIssues": [], "collected": [node], "reports": {node: [
                {"phase": phase, "outcome": "passed"} for phase in ("setup", "call", "teardown")
            ]},
        }))
        return SimpleNamespace(returncode={files[0]: 124, files[1]: 1}.get(command[-1], 0))

    monkeypatch.setattr(lane.subprocess, "run", execute)
    output = tmp_path / "results"
    assert lane.run(files, output, timeout=timeout) == 1
    results = json.loads((output / "results.json").read_text())
    assert [row["file"] for row in results] == files
    assert [row["returncode"] for row in results] == [124, 1, 0, 0, 0, 0, 0, 0]
    assert results[0]["timedOut"] and not results[1]["timedOut"]
    assert f"::error file={files[0]}::exit 124; timeout=True" in capsys.readouterr().out
    assert len((output / "results.jsonl").read_text().splitlines()) == 8
    assert affinities == [[0, 1, 2, 3]]
    assert len({kwargs["env"]["BALOOT_RUNTIME"] for _, kwargs in calls}) == 8
    assert {command[-1] for command, _ in calls} == set(files)
    assert {command[-1] for command, _ in calls[:4]} == {files[6], files[5], files[4], files[0]}
    assert [row["timeoutSeconds"] for row in results] == [timeout] * 5 + [1320, 1320, timeout]
    assert [row["tier"] for row in results] == ["light"] * 5 + ["heavy", "heavy", "light"]
    for command, kwargs in calls:
        limit = "1320" if command[-1] in {files[6], files[5]} else str(timeout)
        assert command[:4] == ["timeout", "--signal=TERM", "--kill-after=10s", limit]
        assert command[4:9] == [sys.executable, "-m", "pytest", "-q", "--override-ini=addopts="]
        runtime = kwargs["env"]["BALOOT_RUNTIME"]
        assert command[9:-1] == [f"--override-ini=cache_dir={runtime}/pytest-cache", "--basetemp", f"{runtime}/tmp",
                                "-p", "scripts.run_heavy_pytest"]
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CMAKE_BUILD_PARALLEL_LEVEL"):
            assert kwargs["env"][name] == "1"
        assert kwargs["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert kwargs["env"]["BALOOT_NATIVE_LIBRARY"] == str(tmp_path / "build/native-release/libbaloot_native.so")


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("parallel_count", [0, 4])
def test_timing_sensitive_files_wait_for_pool_join_and_keep_coverage_and_limits(tmp_path, monkeypatch, workers, parallel_count):
    parallel = [f"tests/test_parallel_{index}.py" for index in range(parallel_count)]
    serial = ["tests/test_serial_heavy.py", "tests/test_serial_light.py"]
    files = serial + parallel  # Serial files cannot enter the pool even when ordered first.
    for path in files:
        target = tmp_path / path
        target.parent.mkdir(exist_ok=True)
        target.touch()
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github/python-test-durations.json").write_text(json.dumps({
        "seconds": {serial[0]: 900.305}, "incompleteFiles": [serial[0]], "serialFiles": serial,
    }))
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci/expected-pytest-skips.txt").write_text("")
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    monkeypatch.setattr(lane, "job_seconds_remaining", lambda _: 1100 if not joined else 500)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(16)), raising=False)
    affinities = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda _, cpus: affinities.append(cpus), raising=False)
    joined = []

    class Pool(lane.ThreadPoolExecutor):
        def __exit__(self, *args):
            result = super().__exit__(*args)
            joined.append(True)
            return result

    monkeypatch.setattr(lane, "ThreadPoolExecutor", Pool)
    barrier, lock = Barrier(workers), Lock()
    active, completed, calls = set(), [], []

    def execute(command, **kwargs):
        path = command[-1]
        assert kwargs['start_new_session'] is True
        with lock:
            if path in serial:
                assert joined == [True] and not active
                assert set(parallel) <= set(completed)
            else:
                assert not set(serial) & set(completed)
            active.add(path)
            calls.append(command)
        if path in parallel:
            barrier.wait(timeout=10)
        node = path + "::test_sample"
        Path(kwargs["env"]["BALOOT_CI_NODE_RECEIPT"]).write_text(json.dumps({
            "exitstatus": 0, "deselected": [], "collectionIssues": [], "collected": [node], "reports": {node: [
                {"phase": phase, "outcome": "passed"} for phase in ("setup", "call", "teardown")
            ]},
        }))
        with lock:
            active.remove(path)
            completed.append(path)
        return SimpleNamespace(returncode=124 if path == serial[0] else 0)

    monkeypatch.setattr(lane.subprocess, "run", execute)
    output = tmp_path / "results"
    assert lane.run(files, output, workers=workers) == 1
    results = json.loads((output / "results.json").read_text())
    assert [row["file"] for row in results] == sorted(files)
    assert len(calls) == len(files) and completed[-2:] == serial
    assert len((output / "results.jsonl").read_text().splitlines()) == len(files)
    assert affinities == [list(range(workers))]
    assert {tuple(command[:4]) for command in calls} == {
        ("timeout", "--signal=TERM", "--kill-after=10s", limit)
        for limit in ({"500", "900"} if parallel else {"500"})}
    for row in results:
        assert row["timeoutSeconds"] == (500 if row["file"] in serial else 900)
        assert row["timedOut"] is (row["file"] == serial[0])
        assert row["callCount"] == (0 if row["timedOut"] else 1)


@pytest.mark.parametrize("serial", [["tests/missing.py"], ["tests/test_sample.py"] * 2])
def test_invalid_serial_inventory_fails_before_work(tmp_path, monkeypatch, serial):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_sample.py").touch()
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github/python-test-durations.json").write_text(json.dumps({
        "seconds": {}, "incompleteFiles": [], "serialFiles": serial,
    }))
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="serial files must be unique repository test files"):
        lane.run(["tests/test_sample.py"], tmp_path / "results")
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("allowlisted", [False, True])
def test_real_fast_receipts_reject_unexpected_skips_xfails_and_no_execution(tmp_path, monkeypatch, capsys, allowlisted):
    sources = {
        "test_mixed.py": """import pytest
def test_pass(): pass
@pytest.mark.skip(reason='Windows-only fixture')
def test_skip(): pass
""",
        "test_skipped.py": """import pytest
@pytest.mark.skip(reason='Windows-only fixture')
def test_skip(): pass
""",
        "test_xfail.py": """import pytest
@pytest.mark.xfail
def test_xfail(): assert False
""",
        "test_xpass.py": """import pytest
@pytest.mark.xfail
def test_xpass(): pass
""",
        "test_collect_only.py": "def test_not_run(): raise AssertionError('must not execute')\n",
        "test_empty.py": "# No test cases.\n",
        "test_deselected.py": "def test_kept(): pass\ndef test_filtered(): raise AssertionError('not selected')\n",
    }
    (tmp_path / "tests").mkdir()
    for name, source in sources.items():
        (tmp_path / "tests" / name).write_text(source)
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github/python-test-durations.json").write_text('{"seconds": {}, "incompleteFiles": []}')
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci/expected-pytest-skips.txt").write_text(
        "setup tests/test_skipped.py::test_skip # Windows-only fixture\n"
        "setup tests/test_mixed.py::test_skip # Windows-only fixture\n" if allowlisted else "")
    repository = lane.ROOT
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0}, raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", lambda *_: None, raising=False)
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
    monkeypatch.setenv("PYTEST_PLUGINS", "nonexistent_inherited_plugin")
    monkeypatch.setenv("PYTEST_DEBUG", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "inherited test")
    run = subprocess.run

    def execute(command, **kwargs):
        assert {key for key in kwargs["env"] if key.startswith("PYTEST_")} == {"PYTEST_DISABLE_PLUGIN_AUTOLOAD"}
        kwargs["env"]["PYTHONPATH"] = str(repository)
        command = [*command[4:-1], f"--rootdir={tmp_path}", f"--confcutdir={tmp_path}", command[-1]]
        if command[-1].endswith("test_collect_only.py"):
            command.insert(-1, "--collect-only")  # rc=0 must still fail without a call receipt.
        if command[-1].endswith("test_deselected.py"):
            command[-1:-1] = ["-k", "kept"]
        return run(command, **kwargs, timeout=30)

    monkeypatch.setattr(lane.subprocess, "run", execute)
    output = tmp_path / "results"
    assert lane.run([f"tests/{name}" for name in sources], output, workers=1) == 1
    results = {row["file"]: row for row in json.loads((output / "results.json").read_text())}
    assert results["tests/test_mixed.py"]["returncode"] == (0 if allowlisted else 1)
    assert results["tests/test_mixed.py"]["callCount"] == int(allowlisted)
    skipped = results["tests/test_skipped.py"]
    assert skipped["returncode"] == 1 and skipped["allSkipped"] is allowlisted and skipped["callCount"] == 0
    assert ("zero call phases" if allowlisted else "unexpected pytest skip") in skipped["error"]
    console = capsys.readouterr().out
    assert ("ALL SKIPPED tests/test_skipped.py" in console) is allowlisted
    for name in ("test_xfail", "test_xpass"):
        assert results[f"tests/{name}.py"]["returncode"] == 1
        assert "unexpected pytest xfail/xpass" in results[f"tests/{name}.py"]["error"]
    assert results["tests/test_collect_only.py"]["returncode"] == 1
    assert "no execution receipt" in results["tests/test_collect_only.py"]["error"]
    assert results["tests/test_empty.py"]["returncode"] == 5
    assert results["tests/test_deselected.py"]["returncode"] == 1
    assert "deselected tests" in results["tests/test_deselected.py"]["error"]


def test_fast_receipt_requires_complete_phases_even_with_a_passing_call():
    phases = [{"phase": phase, "outcome": "passed"} for phase in ("setup", "call", "teardown")]
    original = {"exitstatus": 0, "deselected": [], "collectionIssues": [],
                "collected": ["passed", "incomplete"], "reports": {"passed": phases, "incomplete": phases}}
    assert lane.validate_execution(original)["callCount"] == 2
    for incomplete in ([phases[0]], phases[:2], phases[1:], phases + [phases[-1]],
                       [{"phase": "setup", "outcome": "skipped"}],
                       [*phases[:2], {"phase": "teardown", "outcome": "failed"}]):
        receipt = deepcopy(original)
        receipt["reports"]["incomplete"] = incomplete
        with pytest.raises(ValueError, match="test phases"):
            lane.validate_execution(receipt)


@pytest.mark.parametrize("receipt", [None, "not JSON", json.dumps({"exitstatus": 0}),
    json.dumps({"exitstatus": 0, "deselected": [], "collectionIssues": [], "collected": ["node"],
                "reports": {"node": [{"phase": "setup", "outcome": "passed"}]}})])
def test_zero_exit_without_fresh_valid_execution_receipt_fails(tmp_path, monkeypatch, receipt):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_sample.py").touch()
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github/python-test-durations.json").write_text('{"seconds": {}, "incompleteFiles": []}')
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci/expected-pytest-skips.txt").write_text("")
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0}, raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", lambda *_: None, raising=False)
    output = tmp_path / "results"
    output.mkdir()
    target = output / "test_sample.execution.json"
    target.write_text(json.dumps({"exitstatus": 0, "deselected": [], "collectionIssues": [],
        "collected": ["node"], "reports": {"node": [
            {"phase": phase, "outcome": "passed"} for phase in ("setup", "call", "teardown")]}}))
    target.with_suffix(".jsonl").write_text("old phase data")
    (output / "test_sample.log").write_text("old log")
    (output / "results.json").write_text("old summary")
    (output / "results.jsonl").write_text("old summary rows")

    def execute(command, **kwargs):
        assert not target.exists() and not target.with_suffix(".jsonl").exists()
        assert (output / "test_sample.log").stat().st_size == 0
        assert not (output / "results.json").exists() and not (output / "results.jsonl").exists()
        if receipt is not None:
            Path(kwargs["env"]["BALOOT_CI_NODE_RECEIPT"]).write_text(receipt)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(lane.subprocess, "run", execute)
    assert lane.run(["tests/test_sample.py"], output, workers=1) == 1
    result = json.loads((tmp_path / "results/results.json").read_text())[0]
    assert result["returncode"] == 1 and result["error"].startswith("invalid execution receipt")


def test_expected_skips_require_a_unique_inventory_node_phase_and_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    (tmp_path / "ci").mkdir()
    target = tmp_path / "ci/expected-pytest-skips.txt"
    node = "tests/test_sample.py::test_case"
    row = f"setup {node} # Windows-only\n"
    target.write_text("# Explicit exceptions only\n" + row)
    assert lane.expected_skip_phases({"tests/test_sample.py"}) == {node: ("setup", "Windows-only")}
    for text in (node, f"setup {node} # ", "setup tests/missing.py::test_case # Windows-only",
                 "setup tests/test_sample.py # Windows-only", f"teardown {node} # Windows-only", row * 2):
        target.write_text(text)
        with pytest.raises(ValueError, match="unique repository test nodes, a phase, and a reason"):
            lane.expected_skip_phases({"tests/test_sample.py"})


def test_heavy_schedule_uses_history_threshold_and_incomplete_list_without_deselecting():
    files = ["fast", "scenario", "threshold", "isolated", "below_threshold", "unknown"]
    history = {"seconds": {"fast": 2, "scenario": 2400, "threshold": 900,
                           "isolated": 50, "below_threshold": 899}, "incompleteFiles": ["isolated"]}
    expected = [("scenario", True), ("threshold", True), ("isolated", True),
                ("below_threshold", False), ("fast", False), ("unknown", False)]
    assert lane.schedule(files, history) == expected
    assert lane.schedule(list(reversed(files)), history) == expected
    assert sorted(path for path, _ in expected) == sorted(files)
    for invalid in (-1, float("inf"), float("nan"), "unknown", True):
        with pytest.raises(ValueError):
            lane.schedule(files, {"seconds": {"scenario": invalid}, "incompleteFiles": []})


def test_ci_failure_wave_history_preserves_censored_heavy_files_and_serial_budgets():
    history = json.loads((lane.ROOT / ".github/python-test-durations.json").read_text())
    inventory = {path.relative_to(lane.ROOT).as_posix() for path in lane.ROOT.glob("tests/test_*.py")}
    excluded = lane.node_sharded_files(history)
    planned = dict(lane.schedule(sorted(inventory - excluded), history))
    heavy = {"tests/test_research_ortools_backend_service.py",
             "tests/test_research_ortools_real_executor_predicate_matrix.py",
             "tests/test_research_ortools_unified_executor.py", "tests/test_research_scenario_v4.py"}
    serial = {"tests/test_candidate_resimulation.py", "tests/test_isolated_pytest.py",
              "tests/test_research_ortools_authority_acceptance.py",
              "tests/test_search_timeout_boundaries.py", "tests/test_time_censored_service_boundaries.py",
              "tests/test_outcome_auction.py", "tests/test_outcome_dd_integration.py",
              "tests/test_outcome_generation.py", "tests/test_outcome_refinement.py",
              "tests/test_outcome_rollout.py", "tests/test_outcome_search.py", "tests/test_outcome_spec.py",
              "tests/test_research_cp_sat_campaign_coordinator.py", "tests/test_research_early_play_benchmark.py",
              "tests/test_research_ortools_backend_service.py", "tests/test_research_ortools_unified_executor.py",
              "tests/test_research_ortools_escalation_stage.py", "tests/test_research_ortools_play_tail.py",
              "tests/test_research_ortools_tail_first_lift.py",
              "tests/test_research_result_authority_fallbacks.py", "tests/test_search_performance.py"}
    assert set(planned) | excluded == inventory and set(planned).isdisjoint(excluded)
    assert excluded == {"tests/test_ortools_scenario_model.py"}
    assert set(history["serialFiles"]) == set(history["serialRationale"]) == serial
    assert heavy <= set(history["incompleteFiles"])
    assert all(planned[path] and 900 <= history["seconds"][path] < 901 for path in heavy)
    assert all("40161394797" in history["additionalSources"][path] and
               "censored" in history["additionalSources"][path].lower() for path in heavy)
    assert all(planned[path] is (path in heavy) for path in serial)
    scenario = "tests/test_research_scenario_v4.py"
    split_scenario = {f"tests/test_research_scenario_v4_{name}.py"
                      for name in ("api_estimate", "api_exact", "api_search", "proof")}
    assert history["splitFileLimitSources"] == dict.fromkeys(split_scenario, scenario)
    assert planned[scenario] and all(planned[path] for path in split_scenario)
    assert all(lane.work_file_budget(path, 1320, history) == 0 for path in split_scenario)


@pytest.mark.parametrize(("files", "workers", "timeout"), [([], 4, 900), (["../other.py"], 4, 900),
    (["tests/test_core.py"] * 2, 4, 900), (["tests/test_core.py"], 5, 900), (["tests/test_core.py"], 4, 901)])
def test_invalid_inventory_or_resource_limits_fail_before_work(tmp_path, files, workers, timeout):
    with pytest.raises(ValueError):
        lane.run(files, tmp_path / "unused", workers, timeout)
    assert not (tmp_path / "unused").exists()


def test_cli_partitions_inventory_between_fast_and_node_sharded_lanes(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_isolated_pytest.py", "--output", str(tmp_path)])
    inventory = {path.relative_to(lane.ROOT).as_posix() for path in lane.ROOT.glob("tests/test_*.py")}
    history = json.loads((lane.ROOT / ".github/python-test-durations.json").read_text())
    excluded = lane.node_sharded_files(history)
    versioned = set(history["versionedNodeShardedFiles"])
    expected = lane.fast_files(history)
    assert excluded == {"tests/test_ortools_scenario_model.py"}
    assert set(expected).isdisjoint(excluded | versioned) and set(expected) | excluded | versioned == inventory
    assert lane.node_sharded_files({"nodeShardedFiles": ["estimated"], "seconds": {"over": 3601, "at_limit": 3600}}) == {"estimated", "over"}

    def execute(files, output, workers, timeout):
        assert files == expected and len(files) == len(set(files))
        assert output == tmp_path and (workers, timeout) == (4, 900)
        return 0

    monkeypatch.setattr(lane, "run", execute)
    assert lane.main() == 0


def test_fast_inventory_rejects_overlapping_or_missing_node_profiles():
    history = json.loads((lane.ROOT / ".github/python-test-durations.json").read_text())
    with pytest.raises(ValueError, match="disjoint"):
        lane.fast_files({**history, "versionedNodeShardedFiles": history["nodeShardedFiles"]})
    with pytest.raises(ValueError, match="missing"):
        lane.fast_files({**history, "versionedNodeShardedFiles": ["tests/test_missing_file.py"]})


def test_fast_shards_balance_recorded_durations_and_cover_every_file_once():
    files = [f"tests/test_{index}.py" for index in range(12)]
    history = {"seconds": {}, "incompleteFiles": [],
               "shardSeconds": dict(zip(files, [20, 19, 18, 17, 8, 7, 6, 5, 4, 3, 2, 1]))}
    groups = lane.split_files(files, history)
    assert groups == lane.split_files(list(reversed(files)), history)
    assert sorted(path for group in groups for path in group) == sorted(files)
    # Pooled files overlap on four workers; balance their wall time, not summed CPU time.
    loads = [max(history["shardSeconds"][path] for path in group) for group in groups]
    assert max(loads) - min(loads) <= 3
    assert all(lane.split_files(files, {"seconds": {}, "incompleteFiles": []}))
    for bad_files, count in [([], 4), (files * 2, 4), (files, 0), (files, 13)]:
        with pytest.raises(ValueError):
            lane.split_files(bad_files, history, count)
    with pytest.raises(ValueError, match="finite nonnegative"):
        lane.split_files(files, {**history, "shardSeconds": {files[0]: float('nan')}})


def test_measured_slow_files_keep_exact_union_with_balanced_wall_time():
    history = json.loads((lane.ROOT / ".github/python-test-durations.json").read_text())
    files = lane.fast_files(history)
    groups = lane.split_files(files, history)
    backend = "tests/test_research_ortools_backend_service.py"
    unified = "tests/test_research_ortools_unified_executor.py"
    assert all(backend not in group and unified not in group for group in groups)
    assert {backend, unified} <= set(history["versionedNodeShardedFiles"])
    assert history["versionedNodeShards"] == 4
    assert {backend, unified} <= set(history["serialFiles"])
    assert history["shardTimingSourceRun"] == 44812185812
    assert history["shardSeconds"][unified] == 869.454
    scorer = "tests/test_engine_interest_score.py"
    assert scorer in files and history["seconds"][scorer] == history["shardSeconds"][scorer] == 3.0
    assert set(files) <= set(history["shardSeconds"])
    assert sorted(path for group in groups for path in group) == files
    inventory = {path.relative_to(lane.ROOT).as_posix() for path in lane.ROOT.glob("tests/test_*.py")}
    assert set(files) | lane.node_sharded_files(history) | set(history["versionedNodeShardedFiles"]) == inventory
    walls = []
    for group in groups:
        workers, serial = [0.0] * 4, 0.0
        for path, _ in lane.schedule(group, history):
            seconds = history["shardSeconds"].get(path, history["seconds"].get(path, 1.0))
            if path in history["serialFiles"]:
                serial += seconds
            else:
                workers[workers.index(min(workers))] += seconds
        walls.append(max(workers) + serial)
    assert max(walls) - min(walls) < 120 and max(walls) < 900
    assert lane.split_files(files, history) == lane.split_files(list(reversed(files)), history)


def test_outcome_generator_files_keep_timing_weights_and_exactly_once_fast_coverage():
    history = json.loads((lane.ROOT / ".github/python-test-durations.json").read_text())
    added = {f"tests/test_{name}.py" for name in (
        "double_dummy", "informed_project_visibility", "outcome_auction", "outcome_benchmark",
        "outcome_dd_benchmark", "outcome_dd_integration", "outcome_dd_policy", "outcome_generation",
        "outcome_refinement", "outcome_rollout", "outcome_search", "outcome_spec",
    )}
    scheduled = [path for group in lane.split_files(lane.fast_files(history), history) for path in group]
    assert all(scheduled.count(path) == 1 for path in added)
    assert all(history["seconds"][path] > 0 and history["shardSeconds"][path] > 0 for path in added)
    assert all("40356050071" in history["additionalSources"][path] for path in added)


def test_timeout_boundary_files_keep_weights_serial_isolation_and_exactly_once_coverage():
    history = json.loads((lane.ROOT / ".github/python-test-durations.json").read_text())
    added = {"tests/test_search_timeout_boundaries.py", "tests/test_time_censored_service_boundaries.py"}
    scheduled = [path for group in lane.split_files(lane.fast_files(history), history) for path in group]
    assert all(scheduled.count(path) == 1 for path in added)
    assert all(history["seconds"][path] > 0 and history["shardSeconds"][path] > 0 for path in added)
    assert all("PR17" in history["additionalSources"][path] for path in added)
    assert added <= set(history["serialFiles"]) == set(history["serialRationale"])
    assert not any(dict(lane.schedule(scheduled, history))[path] for path in added)


def test_balancer_counts_serial_tail_in_addition_to_four_worker_pool():
    files = ["serial"] + [f"pool-{index}" for index in range(16)]
    history = {"seconds": {}, "incompleteFiles": [], "serialFiles": ["serial"],
               "shardSeconds": {"serial": 80, **{path: 20 for path in files[1:]}}}
    groups = lane.split_files(files, history, shards=2)
    serial_group = next(group for group in groups if "serial" in group)
    assert serial_group == ["serial"]  # Its serial tail already equals the other group's pool wall time.
    assert sorted(path for group in groups for path in group) == sorted(files)
    assert lane.split_files(["serial"], history, shards=1) == [["serial"]]


def test_fast_gate_requires_three_successful_exact_head_and_version_inventories():
    files = [f"tests/test_{index}.py" for index in range(12)]
    history = {"seconds": {}, "incompleteFiles": []}
    receipts = [{"shard": index, "head": "head", "python": "3.13", "result": "success", "files": group}
                for index, group in enumerate(lane.split_files(files, history), 1)]
    lane.validate_shards(list(reversed(receipts)), "head", "3.13", files, history)
    for field, value in [("head", "old"), ("python", "3.12"), ("result", "failure"),
                         ("files", []), ("files", files), ("shard", 2)]:
        invalid = deepcopy(receipts)
        invalid[0][field] = value
        with pytest.raises(ValueError):
            lane.validate_shards(invalid, "head", "3.13", files, history)
    for invalid in (receipts[:-1], receipts + [receipts[0]]):
        with pytest.raises(ValueError):
            lane.validate_shards(invalid, "head", "3.13", files, history)


def test_cli_executes_exact_timing_shard(tmp_path, monkeypatch):
    history = json.loads((lane.ROOT / ".github/python-test-durations.json").read_text())
    files = lane.fast_files(history)
    groups = lane.split_files(files, history)
    assert sorted(path for group in groups for path in group) == files
    for index, group in enumerate(groups, 1):
        monkeypatch.setattr(sys, "argv", ["run_isolated_pytest.py", "--output", str(tmp_path), "--shard", str(index)])
        def execute(selected, output, workers, timeout):
            assert selected == group and (workers, timeout) == (4, 900)
            return 0
        monkeypatch.setattr(lane, "run", execute)
        assert lane.main() == 0


def test_resource_snapshot_keeps_affinity_quota_and_missing_pressure_explicit(monkeypatch):
    def read(path, **_):
        if str(path).replace("\\", "/").endswith("/cpu.max"):
            return "400000 100000\n"
        raise FileNotFoundError(path)
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {2, 4, 6}, raising=False)
    snapshot = lane.resource_snapshot()
    assert snapshot["affinity"] == [2, 4, 6]
    assert snapshot["files"]["/sys/fs/cgroup/cpu.max"] == "400000 100000"
    assert snapshot["files"]["/sys/fs/cgroup/cpu.stat"] is None
