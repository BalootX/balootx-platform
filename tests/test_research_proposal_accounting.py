import copy
import json
from types import SimpleNamespace

import pytest

from baloot_platform.research import service as service_module
from baloot_platform.research.ortools_research_executor import OrToolsResearchCampaignResultV1
from baloot_platform.research.service import ResearchFoundryServiceV1


@pytest.mark.real_clock
@pytest.mark.parametrize("screen_seconds,native_wall,native_cpu", [(3.0, 2.0, 1.0), (12.0, 0.0, 0.0)])
@pytest.mark.parametrize("child_cpu", [0.0, 7.25])
@pytest.mark.parametrize("origin,decision", [("search", "play"), ("search", "bid"), ("exact_trace_prefix", "play")])
@pytest.mark.parametrize("streaming", [False, True])
def test_screening_cost_is_retained_without_rewriting_campaign_budget(
    monkeypatch, screen_seconds, native_wall, native_cpu, child_cpu, origin, decision, streaming,
):
    clock = {"wall": 100.0, "cpu": 20.0}
    monkeypatch.setattr(service_module, "time", SimpleNamespace(
        perf_counter=lambda: clock["wall"], thread_time=lambda: clock["cpu"]))
    monkeypatch.setattr(service_module, "partition_research_shards", lambda **kwargs: (object(),))
    parameters = dict(maximumCampaignSeconds=10.0, maximumSeconds=5.0, seed=10,
        maximumCandidates=2, symmetryPolicyId="rules_preserving_v1", allowSymmetryOnlyVariants=False,
        minimumDecisionStateDistance=1, maximumProofNodes=20, maximumProofDepth=32,
        maximumHiddenWorlds=4, maximumLineCountLimbs=64, maximumConflicts=100,
        maximumPrefixLiftAttempts=1, requestedShardWorkers=3 if streaming else 1, cancellationPollMilliseconds=25)
    original_parameters = copy.deepcopy(parameters)
    observed = {}
    def boundaries(campaign, recipe, effective_parameters):
        observed["parameters"] = dict(effective_parameters)
        return {}
    monkeypatch.setattr(service_module, "_ortools_line_count_boundaries_v1", boundaries)
    service = object.__new__(ResearchFoundryServiceV1)
    service.maximum_local_shard_workers = 2
    service._cancel_check = lambda code: False
    service._ortools_execution_parameters_v2 = lambda recipe, backend: parameters
    def screen(*args, **kwargs):
        clock["wall"] += screen_seconds
        clock["cpu"] += 1.5
        kwargs["worker_cpu_callback"](child_cpu)
        return {}
    service._runtime_proposal_queue = screen
    def stream(*args, **kwargs):
        assert args[2]['requestedShardWorkers'] == 1
        screen(*args, **kwargs)
        yield 10, {'witness': True}
    service._runtime_proposal_stream = stream
    # Stop after evidence persistence; no solver, result qualification, or runtime is needed.
    service.store = SimpleNamespace(list_shards=lambda code: (),
        get_run=lambda code: {"status": "cancelled"}, update_run=lambda *args, **kwargs: None)
    def put(payload, **kwargs):
        observed["document"] = json.loads(payload)
        observed["artifact_name"] = kwargs["name"]
        return {}
    service.artifact_store = SimpleNamespace(put=put)
    native = OrToolsResearchCampaignResultV1(
        "incomplete" if screen_seconds >= 10 else "not_found_within_budget",
        (), (), (), (), 1, 2, "rules_preserving_v1", "decision_state_card_hamming_v1",
        1, native_wall, native_cpu)
    def execute(*args, **kwargs):
        observed["native_budget"] = kwargs["maximum_campaign_seconds"]
        if 'runtime_proposal_stream' in kwargs:
            assert clock['wall'] == 100.0, 'stream was materialized before proof dispatch'
            assert kwargs['shard_worker_count'] == 1
            assert list(kwargs['runtime_proposal_stream']) == [(10, {'witness': True})]
        return native
    backend = SimpleNamespace(authority=SimpleNamespace(quality_execution=SimpleNamespace(
        candidate_source="runtime_proposal_stream_v1" if streaming else "runtime_proposal_v1",
        video_interestingness=None)), execute_campaign=execute)
    backend.for_run = lambda: backend
    recipe = SimpleNamespace(sha256="recipe", scenario=SimpleNamespace(
        origin=SimpleNamespace(kind=origin), decision_point={"kind": decision}),
        proof=object(), playing_quality=object(), generation=None, output={"resultCount": 1},
        resources={"seedStart": 10, "seedCount": 2, "seedsPerShard": 2})
    service.maximum_shard_attempts = 1
    context = SimpleNamespace(bundle=SimpleNamespace(rules=object(), components={}))
    service._execute_v4_ortools_run(
        {"runCode": "run", "runHash": "hash", "status": "running", "startedAt": "now"},
        recipe, context, backend)
    assert parameters == original_parameters == observed["parameters"]
    screened = origin == "search" and decision == "play"
    assert observed["native_budget"] == max(0.0, 10.0 - (screen_seconds if screened and not streaming else 0))
    assert observed["document"]["wallTimeSeconds"] == (screen_seconds if screened and not streaming else 0) + native_wall
    assert observed["document"]["cpuSeconds"] == native_cpu + ((child_cpu if streaming else 1.5 + child_cpu) if screened else 0)
    assert (observed["document"]["sha256"] != native.sha256) == (screened and (not streaming or child_cpu > 0))
    if streaming and screened:
        assert observed['document']['runtimeProposalQueue'] == [{'witness': True}]
    assert observed["artifact_name"] == observed["document"]["sha256"] + ".json"


@pytest.mark.parametrize(
    ("executor_status", "has_candidate", "expected"),
    (
        ("qualified", True, "satisfied"),
        ("infeasible", False, "structurally_infeasible"),
        ("not_found", False, "not_found_within_budget"),
        ("feasible", True, "semantic_postcondition_miss"),
        ("semantic_postcondition_miss", True, "semantic_postcondition_miss"),
        ("incomplete", True, "proof_incomplete"),
        ("incomplete", False, "not_found_within_budget"),
        ("proof_failed", True, "proof_failed"),
    ),
)
def test_ortools_terminal_statuses_do_not_change_run_lifecycle(
    executor_status: str,
    has_candidate: bool,
    expected: str,
) -> None:
    result = SimpleNamespace(
        status=executor_status,
        candidate=object() if has_candidate else None,
    )

    assert ResearchFoundryServiceV1._ortools_terminal_status_v2(result) == expected
