"""Hosted workflow policy and its fail-closed producer contract."""
import json
import re

import pytest
import yaml

from scripts import ci_scope


def test_hosted_ci_contract(root):
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())
    triggers = workflow[True]  # PyYAML's YAML 1.1 spelling of on.
    assert triggers["merge_group"]["types"] == ["checks_requested"]
    assert "pull_request_target" not in triggers
    assert not {"paths", "paths-ignore"} & triggers["pull_request"].keys()
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"
    jobs = workflow["jobs"]
    for job in jobs.values():
        assert job["runs-on"] == "ubuntu-24.04"
        assert job["timeout-minutes"] > 0
        for step in job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r"[0-9a-f]{40}", step["uses"].rsplit("@", 1)[-1])
            assert not step.get("continue-on-error")
    assert jobs["ci-gate"]["name"] == "CI result"
    assert set(jobs["ci-gate"]["needs"]) == jobs.keys() - {"ci-gate"}
    assert jobs["ci-gate"]["if"] == "always()"


@pytest.mark.parametrize("event", ["pull_request", "merge_group", "push", "workflow_dispatch"])
def test_gate_rejects_unproven_full_producers(event):
    plan = {key: json.dumps(value) for key, value in ci_scope.current_plan().items()}
    results = {job: {"result": "success"} for job in {"plan", "preflight", "dependency-review",
               "python-affected", *ci_scope.FULL, *ci_scope.SCOPED}}
    results["plan"]["outputs"] = plan
    results["dependency-review"]["result"] = ("success" if event in {"pull_request", "merge_group"} else "skipped")
    results["python-affected"]["result"] = "skipped"
    ci_scope.validate_results(event, results)
    for job in results:
        old = results[job]["result"]
        for bad in {"failure", "cancelled", "skipped", "success", ""} - {old}:
            results[job]["result"] = bad
            with pytest.raises(ValueError):
                ci_scope.validate_results(event, results)
        results[job]["result"] = old
    for key in ("full", "sdk", "remotion"):
        plan[key] = "false"
        with pytest.raises(ValueError):
            ci_scope.validate_results(event, results)
        plan[key] = "true"
    results.pop("python-heavy")
    with pytest.raises(ValueError):
        ci_scope.validate_results(event, results)
