from copy import deepcopy
import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import ci_known_failures as known
from scripts.run_isolated_pytest import validate_execution


RESOURCE_NODE = "tests/test_resource.py::test_reads_optional_fixture"


@pytest.fixture(autouse=True)
def trusted_test_baseline(monkeypatch, tmp_path):
    baseline = tmp_path / "ci" / "known-failures.json"
    baseline.parent.mkdir()
    baseline.write_text(json.dumps({
        "schemaVersion": 1,
        "source": {"runId": 44039515022},
        "tests": [
            {"nodeid": RESOURCE_NODE, "phase": "call",
             "signature": {"exception": "FileNotFoundError", "resourceKind": "missing_file",
                           "resource": f"assets/test-fixture-{version}.wav"},
             "observations": [{"artifactName": f"python-file-results-{version}-1"}]}
            for version in ("3.12", "3.13")
        ],
    }), encoding="utf-8")
    monkeypatch.setattr(known, "BASELINE", baseline)
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"base": {"sha": "a" * 40}}}), encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    def git_show(command, **kwargs):
        assert command == ["git", "show", "a" * 40 + ":ci/known-failures.json"]
        return SimpleNamespace(stdout=known.BASELINE.read_text(encoding="utf-8"))
    monkeypatch.setattr(known.subprocess, "run", git_show)
    known.entries.cache_clear()
    yield
    known.entries.cache_clear()


def _failure():
    version = ".".join(known.platform.python_version_tuple()[:2])
    row = next(row for row in json.loads(known.BASELINE.read_text(encoding="utf-8"))["tests"]
               if f"-{version}-" in row["observations"][0]["artifactName"])
    node, phase, signature = row["nodeid"], row["phase"], row["signature"]
    event = {"phase": phase, "outcome": "failed", "xfail": False,
             "longrepr": "E   FileNotFoundError: [Errno 2] No such file or directory: '/checkout/" + signature["resource"] + "'"}
    return node, event


def _execution(node, event):
    return {"exitstatus": 1, "collected": [node], "deselected": [], "collectionIssues": [],
            "reports": {node: [{"phase": "setup", "outcome": "passed"}, event,
                               {"phase": "teardown", "outcome": "passed"}]}}


def test_known_failure_keeps_the_actual_failed_report():
    node, event = _failure()
    execution = _execution(node, event)
    original = deepcopy(execution)
    assert validate_execution(execution) == {"callCount": 1, "allSkipped": False, "knownFailures": [node]}
    assert execution == original


@pytest.mark.parametrize("change", ["node", "phase", "resource", "exception", "xfail"])
def test_a_new_cause_inside_a_listed_test_stays_red(change):
    node, event = _failure()
    if change == "node":
        node += "_new"
    elif change == "phase":
        event["phase"] = "teardown"
    elif change == "resource":
        event["longrepr"] = event["longrepr"].replace("test-fixture-", "different-fixture-")
    elif change == "exception":
        event["longrepr"] = "E   AssertionError: new product failure"
    else:
        event["xfail"] = True
    assert not known.is_known(node, event)
    with pytest.raises(ValueError):
        validate_execution(_execution(node, event))


def test_a_known_failure_does_not_hide_another_failing_node():
    node, event = _failure()
    execution = _execution(node, event)
    new_node = node + "_new"
    execution["collected"].append(new_node)
    execution["reports"][new_node] = deepcopy(execution["reports"][node])
    with pytest.raises(ValueError):
        validate_execution(execution)


def test_a_traceback_source_mention_is_not_a_missing_resource_exception():
    assert known.failure_signature("assert 'outputs/research-scenario-v4-goldens/manifest.json'\nE   AssertionError: new failure") is None


def test_pull_request_cannot_admit_its_own_new_baseline(monkeypatch):
    def missing_target(command, **kwargs):
        raise subprocess.CalledProcessError(128, command)
    monkeypatch.setattr(known.subprocess, "run", missing_target)
    with pytest.raises(ValueError, match="absent from the trusted base"):
        known.entries()


def test_merge_group_reads_its_trusted_base(monkeypatch, tmp_path):
    event = tmp_path / "merge-event.json"
    event.write_text(json.dumps({"merge_group": {"base_sha": "a" * 40}}), encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "merge_group")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    assert known.entries()


@pytest.mark.parametrize("event,ref,admitted", [
    ("push", "refs/heads/main", True),
    ("push", "refs/heads/feature", False),
    ("workflow_dispatch", "refs/heads/main", False),
    ("workflow_dispatch", "refs/heads/feature", False),
    (None, "refs/heads/main", False),
])
def test_only_main_push_can_use_the_checked_out_baseline(monkeypatch, event, ref, admitted):
    if event is None:
        monkeypatch.delenv("GITHUB_EVENT_NAME")
    else:
        monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    monkeypatch.setenv("GITHUB_REF", ref)
    node, failure = _failure()
    assert known.is_known(node, failure) is admitted


@pytest.mark.parametrize("version,resource,accepted", [
    ("3.12", "test-fixture-3.12.wav", True), ("3.12", "test-fixture-3.13.wav", False),
    ("3.13", "test-fixture-3.12.wav", False), ("3.13", "test-fixture-3.13.wav", True),
])
def test_fixture_signature_is_bound_to_the_recorded_python_matrix(monkeypatch, version, resource, accepted):
    monkeypatch.setattr(known.platform, "python_version_tuple", lambda: (*version.split("."), "0"))
    event = {"phase": "call", "outcome": "failed", "xfail": False,
             "longrepr": "E   FileNotFoundError: [Errno 2] No such file or directory: '/checkout/assets/" + resource + "'"}
    assert known.is_known(RESOURCE_NODE, event) is accepted
