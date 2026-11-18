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
    assert jobs["ci-gate"]["name"] == "${{ github.event_name == 'workflow_dispatch' && 'CI diagnostic result' || 'CI result' }}"
    assert jobs["plan"]["if"] == "github.event_name != 'pull_request' || !github.event.pull_request.draft"
    gate_step = jobs["ci-gate"]["steps"][-1]
    assert gate_step["env"]["IS_DRAFT"] == "${{ github.event_name == 'pull_request' && github.event.pull_request.draft }}"
    assert "exit 1" in gate_step["run"].split("python3 scripts/ci_scope.py gate")[0]
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


def test_one_run_never_holds_more_than_twenty_hosted_jobs(root):
    # Shared Actions capacity: the largest set of mutually independent jobs, weighted by matrix width.
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())["jobs"]
    needs = {name: set([job["needs"]] if isinstance(job.get("needs"), str) else job.get("needs", ())) for name, job in jobs.items()}
    def ancestors(name):
        found = set(needs[name])
        for parent in needs[name]:
            found |= ancestors(parent)
        return found
    before = {name: ancestors(name) for name in jobs}
    def width(job):
        strategy = job.get("strategy") or {}
        size = 1
        for values in (strategy.get("matrix") or {}).values():
            size *= len(values) if isinstance(values, list) else 1
        return min(size, strategy.get("max-parallel", size))
    names = sorted(jobs)
    def peak(index, chosen):
        if index == len(names):
            return 0
        name = names[index]
        best = peak(index + 1, chosen)
        if all(name not in before[other] and other not in before[name] for other in chosen):
            best = max(best, width(jobs[name]) + peak(index + 1, chosen | {name}))
        return best
    assert peak(0, frozenset()) <= 20
