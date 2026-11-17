from copy import deepcopy
import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import ci_known_failures as known
from scripts.run_isolated_pytest import validate_execution


@pytest.fixture(autouse=True)
def trusted_test_baseline(monkeypatch, tmp_path):
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
    row = json.loads(known.BASELINE.read_text(encoding="utf-8"))["tests"][0]
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
        event["longrepr"] = event["longrepr"].replace("manifest.json", "another.json")
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


@pytest.mark.parametrize("version,resource,accepted", [
    ("3.12", "siri.wav", True), ("3.12", "triple.wav", False),
    ("3.13", "siri.wav", False), ("3.13", "triple.wav", True),
])
def test_voice_signature_is_bound_to_the_original_python_matrix(monkeypatch, version, resource, accepted):
    monkeypatch.setattr(known.platform, "python_version_tuple", lambda: (*version.split("."), "0"))
    event = {"phase": "call", "outcome": "failed", "xfail": False,
             "longrepr": "E   FileNotFoundError: [Errno 2] No such file or directory: '/checkout/assets/audio/voice_packs/ar_sa_baloot_v2_smooth/" + resource + "'"}
    node = "tests/test_video_motion_release.py::test_voice_pack_is_original_and_media_correct"
    assert known.is_known(node, event) is accepted
