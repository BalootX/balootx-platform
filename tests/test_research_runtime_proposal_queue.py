from types import SimpleNamespace
import json
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import pytest

from baloot_platform.research import construction_pre_screen as screen
from baloot_platform.research.scenario_contracts import canonical_sha256
from baloot_platform.research.service import ResearchFoundryServiceV1
from test_research_family_proposal_stream import target as _family_target

target = _family_target


@pytest.mark.real_clock
@pytest.mark.parametrize("workers", [2, 4])
@pytest.mark.parametrize("stop", [None, "cancel", "deadline", "pre_cancel", "pre_deadline", "error", "limit"])
def test_parallel_seed_queue_is_bounded_ordered_and_cleans_up(monkeypatch, workers, stop):
    submitted, delivered, cleaned = [], [], []
    cancelled = [stop == "pre_cancel"]
    clock = [10.0 if stop == "pre_deadline" else 0.0]
    monkeypatch.setattr(screen, "time", SimpleNamespace(perf_counter=lambda: clock[0]))

    class Future:
        def __init__(self, seed):
            self.seed, self.calls = seed, 0

        def result(self, timeout):
            self.calls += 1
            # First seed is slow; later completed seeds must not jump ahead.
            if self.seed == 10 and self.calls == 1:
                if stop == "cancel":
                    cancelled[0] = True
                if stop == "deadline":
                    clock[0] = 10
                raise screen.FutureTimeout()
            if stop == "error":
                raise RuntimeError("worker failed")
            delivered.append(self.seed)
            return self.seed, 0, []

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == workers
            self.event, self.slots, _ = kwargs["initargs"]

        def submit(self, fn, payload):
            assert fn is screen._screen_refine_proposal_seed
            submitted.append(payload[1])
            assert len(submitted) - len(delivered) <= workers
            return Future(payload[1])

        def shutdown(self, **kwargs):
            assert self.event.is_set()
            assert kwargs == dict(wait=True, cancel_futures=True)
            self.slots[0] = 3.5  # Includes rejected/speculative work, not only queue entries.
            cleaned.append(True)

    monkeypatch.setattr(screen, "ProcessPoolExecutor", Pool)
    execution = SimpleNamespace(proposal_seed_count=7, proposal_maximum_seconds=0.5,
        early_play_policy_id="frozen", video_interestingness={})
    parameters = dict(seed=10, requestedShardWorkers=workers, maximumSeconds=5)
    cpu = []
    iterator = screen.parallel_runtime_proposal_seeds(SimpleNamespace(data={}), execution,
        parameters, deadline=10, cancel_check=lambda: cancelled[0], worker_cpu_callback=cpu.append)
    if stop == "error":
        with pytest.raises(RuntimeError, match="worker failed"):
            list(iterator)
    elif stop == "limit":
        assert next(iterator)[0] == 10
        iterator.close()
        assert len(submitted) == workers
    else:
        rows = list(iterator)
        assert [row[0] for row in rows] == ([] if stop else list(range(10, 17)))
    assert cleaned == [True] and cpu == [3.5]
    if stop in {"pre_cancel", "pre_deadline"}:
        assert submitted == []


@pytest.mark.real_clock
def test_proposal_worker_spawn_serializes_frozen_inputs_and_accounts_cpu(rules):
    context = multiprocessing.get_context("spawn")
    cancelled = context.Event()
    cancelled.set()
    slots, next_slot = context.Array("d", 1), context.Value("i", 0)
    with ProcessPoolExecutor(max_workers=1, mp_context=context,
            initializer=screen._initialize_proposal_worker,
            initargs=(cancelled, slots, next_slot)) as pool:
        row = pool.submit(screen._screen_refine_proposal_seed,
            (rules.data, 10, "unused", {}, {}, time.perf_counter() + 30, 1)).result(timeout=30)
    assert row == (10, 0, []) and slots[0] > 0


def test_parallel_queue_preserves_serial_data_hashes_and_closes_at_limit(monkeypatch):
    parameters = dict(seed=10, maximumCandidates=2, maximumSeconds=5,
        maximumCampaignSeconds=10, maximumProofNodes=20, maximumHiddenWorlds=4)
    execution = SimpleNamespace(early_play_policy_id="frozen", video_interestingness={},
        proposal_seed_count=20, proposal_maximum_seconds=0.5)
    rows = [{"status": "priority_candidate", "prefix": {"play": index},
        "prefixHash": "a" * 64, "rulesHash": "b" * 64, "policyHash": "c" * 64,
        "playIndex": index, "actor": "north", "action": "SA", "sha256": "d" * 64}
        for index in (26, 25, 24)]
    monkeypatch.setattr(screen, "screen_runtime_proposals_v1", lambda *a, **k: {"proposals": rows})
    monkeypatch.setattr(screen, "refine_runtime_proposal_v1", lambda rules, row, **k: row)
    closed = []
    def parallel(*args, **kwargs):
        try:
            yield 10, 3, rows
            pytest.fail("queue should stop at candidate limit")
        finally:
            closed.append(True)
    monkeypatch.setattr(screen, "parallel_runtime_proposal_seeds", parallel)
    def collect(workers):
        return ResearchFoundryServiceV1._runtime_proposal_queue(object(), execution,
            {**parameters, "requestedShardWorkers": workers}, campaign_started=time.perf_counter(),
            cancel_check=lambda: False, progress_callback=lambda row: None)
    assert collect(1) == collect(2) == collect(4)
    assert closed == [True, True]


@pytest.fixture
def rules(root):
    from baloot_platform.core.rules import RuleBook
    return RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")


@pytest.mark.real_clock
@pytest.mark.parametrize("stop", ["close", "cancel", "deadline"])
def test_proposal_stream_yields_before_exhaustion_and_closes_workers(monkeypatch, stop):
    from baloot_platform.research import service as service_module
    clock, cancelled, visited, closed = [0.0], [False], [], []
    monkeypatch.setattr(service_module, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    row = dict(prefix={}, prefixHash="a" * 64, rulesHash="b" * 64,
               policyHash="c" * 64, playIndex=24, actor="north", action="SA", sha256="d" * 64)
    def producer(*args, **kwargs):
        try:
            visited.append(10)
            yield 10, 2, [row, row]
            pytest.fail("consumer must stop before another seed")
        finally:
            closed.append(True)
    monkeypatch.setattr(screen, "parallel_runtime_proposal_seeds", producer)
    execution = SimpleNamespace(proposal_maximum_seconds=1)
    parameters = dict(seed=10, maximumCampaignSeconds=10, maximumSeconds=5,
                      requestedShardWorkers=2, maximumCandidates=3)
    stream = ResearchFoundryServiceV1._runtime_proposal_stream(
        object(), execution, parameters, campaign_started=0,
        cancel_check=lambda: cancelled[0], progress_callback=lambda row: None)
    assert visited == []
    ordinal, witness = next(stream)
    assert ordinal == 10 and visited == [10] and closed == []
    assert witness["sha256"] == canonical_sha256({k: v for k, v in witness.items() if k != "sha256"})
    if stop == "close":
        stream.close()
    else:
        cancelled[0] = stop == "cancel"
        clock[0] = 6 if stop == "deadline" else 0
        assert list(stream) == []
    assert closed == [True]


@pytest.mark.real_clock
def test_worker_expiry_between_cancel_check_and_budget_is_not_error(rules, monkeypatch):
    clock = iter([9.0, 11.0])
    monkeypatch.setattr(screen, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    monkeypatch.setattr(screen, "_sample_proposal_cpu", lambda: None)
    monkeypatch.setattr(screen, "_proposal_cancel", SimpleNamespace(is_set=lambda: False))
    monkeypatch.setattr(screen, "screen_runtime_proposals_v1",
                        lambda *a, **k: pytest.fail("expired worker must not start screening"))
    assert screen._screen_refine_proposal_seed((rules.data, 10, "unused", {}, {}, 10.0, 1)) == (10, 0, [])


@pytest.mark.real_clock
@pytest.mark.parametrize("stop", [None, "cancel", "expired"])
def test_proposal_queue_is_budgeted_data_only_and_scans_all_decisions(monkeypatch, stop):
    calls = []
    parameters = {"seed": 10, "maximumCandidates": 2, "maximumSeconds": 5,
                  "maximumCampaignSeconds": 10, "maximumProofNodes": 20, "maximumHiddenWorlds": 4}
    execution = SimpleNamespace(early_play_policy_id="frozen", video_interestingness={"frozen": True},
                                proposal_seed_count=20000, proposal_maximum_seconds=0.5)
    def collect(rules, **kwargs):
        calls.append(kwargs)
        assert 0 < kwargs["maximum_seconds"] <= 0.5
        assert kwargs["rollout_policy_id"] == "frozen"
        return {"proposals": [{"index": 24}, {"index": 25}]}
    def refine(rules, row, **kwargs):
        assert 0 < kwargs["maximum_seconds"] <= 0.5
        assert kwargs["maximum_nodes"] == 20 and kwargs["maximum_worlds"] == 4
        return {"status": "priority_candidate", "prefix": {"play": row["index"]},
                "prefixHash": "a" * 64, "rulesHash": "b" * 64, "policyHash": "c" * 64,
                "playIndex": row["index"], "actor": "north", "action": "SA", "sha256": "d" * 64,
                "ddValues": {"SA": 100}, "qualified": False}
    monkeypatch.setattr(screen, "screen_runtime_proposals_v1", collect)
    monkeypatch.setattr(screen, "refine_runtime_proposal_v1", refine)
    queue = ResearchFoundryServiceV1._runtime_proposal_queue(
        object(), execution, parameters, campaign_started=time.perf_counter() - (20 if stop == "expired" else 0),
        cancel_check=lambda: stop == "cancel", progress_callback=lambda row: None)
    if stop:
        assert queue == {} and calls == []
    else:
        assert list(queue) == [10, 11] and len(calls) == 1
        assert [row["playIndex"] for row in queue.values()] == [24, 25]
        assert [row["sourceSeed"] for row in queue.values()] == [10, 10]
        for row in queue.values():
            assert "qualified" not in row and "ddValues" not in row
            assert row["sha256"] == canonical_sha256({key: value for key, value in row.items() if key != "sha256"})


@pytest.mark.real_clock
def test_slow_proposal_advances_to_next_seed_with_campaign_time_remaining(monkeypatch):
    from baloot_platform.research import service as service_module
    clock = [0.0]
    monkeypatch.setattr(service_module, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    seeds = []
    def screen_one(rules, **kwargs):
        seeds.append(kwargs["seed"])
        clock[0] += kwargs["maximum_seconds"]
        return {"status": "time_budget", "proposals": []}
    monkeypatch.setattr(screen, "screen_runtime_proposals_v1", screen_one)
    execution = SimpleNamespace(proposal_seed_count=20000, proposal_maximum_seconds=0.5,
                                early_play_policy_id="frozen", video_interestingness={})
    parameters = dict(seed=10, maximumCampaignSeconds=2, maximumSeconds=2,
                      maximumProofNodes=1, maximumCandidates=2, maximumHiddenWorlds=2)
    result = ResearchFoundryServiceV1._runtime_proposal_queue(
        object(), execution, parameters, campaign_started=0,
        cancel_check=lambda: False, progress_callback=lambda row: None)
    assert result == {} and seeds == [10, 11, 12, 13] and clock[0] == 2


@pytest.fixture
def native_queue_inputs(root, rules):
    import json
    import yaml
    from baloot_platform.research.scenario_contracts import (
        ExactTracePrefixV1, ScenarioPlayingQualityPolicyV1, ScenarioQualityExecutionBudgetV1,
        SCENARIO_DECISION_KINDS,
    )
    profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))["scenarioResearchV4"]
    interestingness = json.loads((root / "tests/fixtures/actor_interest_calibration.json").read_text())["cases"]["7189"]["policy"]
    execution = ScenarioQualityExecutionBudgetV1.build({**profile["qualityExecution"],
        "candidateSource": "runtime_proposal_v1", "proposalSeedCount": 20000,
        "auditConstructedTrace": True, "earlyPlayPolicyId": "observer_safe_trick_control_v1",
        "exactPlayFromTrick": 7, "videoInterestingness": interestingness})
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2, "mode": "exact", "visibility": "complete_state",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "late_game_exact_v1"}
                      for kind in SCENARIO_DECISION_KINDS}})
    raw = json.loads((root / "tests/fixtures/research-native-qualified-round-v3.json").read_text())["exactTracePrefix"]
    raw.pop("sha256", None)
    from baloot_platform.core.cards import Card
    from baloot_platform.core.models import LegalPlayActionV1
    focal = raw["playActions"][24]
    action = LegalPlayActionV1(Card.parse(focal["card"]), focal.get("ekka", False)).code
    raw["playActions"] = raw["playActions"][:24]
    raw["stateAssertions"] = {}
    prefix = ExactTracePrefixV1.build(raw)
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "play", "selector": {}}, maximum_seconds=10)
    row = {"prefix": prefix.to_dict(), "prefixHash": prefix.sha256, "rulesHash": rules.sha256,
           "policyHash": canonical_sha256(interestingness), "playIndex": 24,
           "actor": normalized.next_actor, "action": action, "sourceHash": "d" * 64, "sourceSeed": 10}
    row["sha256"] = canonical_sha256(row)
    return dict(backend_id="ortools_cp_sat_v3", bundle_revision_id="CBR-PROPOSAL-TEST", bundle_hash="a" * 64,
                quality_policy=policy, quality_execution=execution,
                quality_proof_profiles=profile["proofBudgets"]["profiles"]), row


@pytest.mark.parametrize('workers', [1, 2, 4])
def test_proposal_native_normalization_requires_full_audit_without_rewriting_scenario(
    native_queue_inputs, rules, monkeypatch, workers,
):
    from baloot_platform.research.ortools_research_executor import OrToolsUnifiedResearchExecutorV1, OrToolsCandidateAttemptV1
    from test_research_ortools_unified_executor import _scenario, _leaf, _proof
    options, row = native_queue_inputs
    scenario = _scenario(constraints=_leaf("PRED-SUN", "auction.contract", {"contract": "sun"}),
        outcome=_leaf("PRED-PHASE", "state.phase", {"state": "playing"}, "report"),
        decision_point={"kind": "play", "selector": {}})
    original = scenario.to_dict()
    assert row["rulesHash"] == rules.sha256
    assert row["policyHash"] == canonical_sha256(options["quality_execution"].video_interestingness)
    assert row["sha256"] == canonical_sha256({key: value for key, value in row.items() if key != "sha256"})
    executor = OrToolsUnifiedResearchExecutorV1(rules, runtime_proposals={10: row}, **options)
    calls = []
    def audit(candidate, **kwargs):
        calls.append(candidate)
        assert candidate.origin == "search"
        assert kwargs["recorded_trace"].sha256 == row["prefixHash"]
        assert kwargs["recorded_events"] == () and kwargs["continue_recorded_prefix"] is True
        assert kwargs["decision_point"] == scenario.decision_point
        assert kwargs["stop_on_recorded_quality_failure"] is True
        assert kwargs.get("graph_workers", 1) == workers
        return OrToolsCandidateAttemptV1("not_found", None, reason="quality_test_refused")
    monkeypatch.setattr(executor, "audit_quality_candidate", audit)
    monkeypatch.setattr(executor, "generate_quality_candidate", lambda *a, **k: pytest.fail("proposal bypassed audit"))
    result = executor.execute(scenario, _proof(), seed=10, maximum_candidates=3, maximum_seconds=30,
        maximum_proof_nodes=1000, maximum_proof_depth=64, maximum_hidden_worlds=1,
        terminal_worker_count=workers)
    assert len(calls) == 1
    assert result.status != "qualified"
    assert scenario.to_dict() == original


@pytest.mark.parametrize("scope", ["family", "query", "general", "exact", "passing", "untrusted", "early_miss"])
def test_curated_editorial_miss_skips_only_redundant_scenario_proof(
    native_queue_inputs, root, rules, monkeypatch, scope, target,
):
    import copy
    import json
    from dataclasses import replace
    from baloot_platform.research import decision_selection, ortools_research_executor as module
    from baloot_platform.research import result_authority
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioQualityExecutionBudgetV1
    from test_research_family_proposal_stream import execution, query_execution, generation_request
    from test_research_ortools_unified_executor import _scenario, _leaf, _proof

    options, row = native_queue_inputs
    config = query_execution(root, target) if scope == "query" else ScenarioQualityExecutionBudgetV1.build(
        {**execution(root, target), **({"allowExactInputQuality": True} if scope == "exact" else {})})
    if scope == "general":
        config = options["quality_execution"]
    options = {**options, "quality_execution": config}
    if scope == "query":
        options["generation"] = generation_request(config, mode=["sun"])
    prefix = ExactTracePrefixV1.build(row["prefix"])
    scenario = _scenario(constraints=_leaf("PRED-SUN", "auction.contract", {"contract": "sun"}),
        outcome=_leaf("PRED-PHASE", "state.phase", {"state": "playing"}, "report"),
        decision_point={"kind": "play", "selector": {"trickIndex": 6, "position": 0, "actingSeat": row["actor"]}},
        origin={"kind": "exact_trace_prefix", "exactTracePrefix": row["prefix"],
                "recordedEscalationOpportunities": []} if scope == "exact" else None)
    executor = module.OrToolsUnifiedResearchExecutorV1(rules,
        runtime_proposals=None if scope == "exact" else {10: row}, **options)
    case = json.loads((root / "tests/fixtures/actor_interest_calibration.json").read_text())["cases"]["gap2"]
    steps, evidence, verdicts = [], [], []

    def audit(candidate, **kwargs):
        assert kwargs["recorded_trace"] == prefix
        steps.append("audit")
        if scope == "early_miss":
            partial = {"stage": "curatedSelection", "researchClaimStatus": "not_evaluated"}
            return module.OrToolsCandidateAttemptV1("not_found", None, reason="curated_interest_miss",
                partial_receipt={**partial, "sha256": canonical_sha256(partial)})
        receipt = json.loads(json.dumps(case["receipt"]).replace(case["featuredStateId"], candidate.exact_state_id))
        evidence.append({"receipt": receipt})
        return module.OrToolsCandidateAttemptV1("feasible", replace(candidate, trace={**candidate.trace,
            "playingQualityEvidence": copy.deepcopy(evidence[-1]), "score": case["terminalScore"]}))

    evaluate = decision_selection.evaluate_round_interestingness_v1
    def screen(receipt, score, policy, *, featured_state_id):
        steps.append("screen")
        assert receipt == evidence[0]["receipt"] and featured_state_id != case["featuredStateId"]
        verdict = evaluate(receipt, score, policy, featured_state_id=featured_state_id)
        assert verdict["passed"] is False
        if scope == "passing":
            verdict = {**verdict, "passed": True}  # Only exercise pass-through, not qualification.
        verdicts.append(verdict)
        return verdict

    def compile_graph(*args, **kwargs):
        steps.append("graph")
        raise RuntimeError("stop after reaching the existing scenario proof")

    def validate(trace, **kwargs):
        steps.append("validate")
        if scope == "early_miss":
            assert "playingQualityEvidence" not in trace  # No short suffix disguised as a full audit.
            return
        assert trace["playingQualityEvidence"] == evidence[0]
        if scope in {"family", "query", "untrusted"}:
            assert trace["curatedSelection"] == {
                "researchClaimStatus": "not_evaluated", "evidence": verdicts[0]}
        if scope == "untrusted":
            raise ValueError("quality ownership validation failed")

    monkeypatch.setattr(executor, "audit_quality_candidate", audit)
    monkeypatch.setattr(module, "_search_decision_prefix", lambda *a, **kw: prefix)
    monkeypatch.setattr(decision_selection, "evaluate_round_interestingness_v1", screen)
    monkeypatch.setattr(module, "compile_observed_decision_v1", compile_graph)
    monkeypatch.setattr(module, "validate_trace_quality_evidence_v1", validate)
    monkeypatch.setattr(result_authority, "_validate_trace_quality_evidence_v1", validate)
    request = _proof().to_dict()
    request.pop("sha256")
    request.update(visibility="actor_information_set", strategyClass="pure_deterministic")
    def execute_candidate():
        return executor.execute(scenario, module.ScenarioProofRequestV1.build(request), seed=10,
            maximum_candidates=1, maximum_seconds=30, maximum_proof_nodes=1000,
            maximum_proof_depth=64, maximum_hidden_worlds=16)
    if scope == "untrusted":
        with pytest.raises(ValueError, match="quality ownership validation failed"):
            execute_candidate()
        assert steps == ["audit", "screen", "validate"]
        return
    result = execute_candidate()
    if scope in {"family", "query", "early_miss"}:
        assert steps == (["audit", "validate"] if scope == "early_miss" else ["audit", "screen", "validate"])
        assert result.status == "not_found" and result.proof is result.predicate_evaluation is None
        assert result.execution_receipt["candidateAttemptReason"] == "curated_interest_miss"
        assert result.execution_receipt["completedCandidates"] == 1
        if scope == "early_miss":
            assert result.execution_receipt["incompleteCompilation"]["researchClaimStatus"] == "not_evaluated"
        else:
            assert result.artifact_preimages.to_dict()["trace"]["curatedSelection"]["evidence"] == verdicts[0]
    else:
        assert steps == ["audit", *(["screen"] if scope == "passing" else []), "graph", "validate"]
        assert result.status == "incomplete"


@pytest.mark.parametrize("scope", ["family", "query", "general", "exact", "generate", "complete_state"])
def test_native_focal_precheck_is_private_to_curated_actor_proposal_audits(
    native_queue_inputs, root, rules, target, monkeypatch, scope,
):
    from baloot_platform.research import ortools_research_executor as module
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioPlayingQualityPolicyV1, ScenarioQualityExecutionBudgetV1
    from test_research_family_proposal_stream import execution, query_execution, generation_request

    options, proposal = native_queue_inputs
    from baloot_platform.core.rules import RuleBook
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v8.yaml")
    config = query_execution(root, target) if scope == "query" else ScenarioQualityExecutionBudgetV1.build(
        {**execution(root, target), "allowExactInputQuality": True})
    if scope == "general":
        config = options["quality_execution"]
    raw = options["quality_policy"].to_dict()
    raw.pop("sha256", None)
    raw["visibility"] = "complete_state" if scope == "complete_state" else "actor_information_set"
    options = {**options, "quality_execution": config, "quality_policy": ScenarioPlayingQualityPolicyV1.build(raw)}
    if scope == "query":
        options["generation"] = generation_request(config, mode=["sun"])
    executor = module.OrToolsUnifiedResearchExecutorV1(rules, **options)
    recorded = ExactTracePrefixV1.build(proposal["prefix"])
    candidate = module.OrToolsNeutralCandidateV1("exact_trace_prefix" if scope == "exact" else "search",
        {key: recorded.to_dict()[key] for key in ("deckOrder", "dealer", "matchScore")}, {},
        recorded.deck_order, {"deal": module._stage_receipt("deal", model_hash="a" * 64,
            evidence_hash="b" * 64, semantic={}, solver_status="optimal")})
    expected = scope in {"family", "query"}
    evidence = {"researchClaimStatus": "not_evaluated", "blockers": [
        "interestingness_next_best_regret_insufficient", "interestingness_baseline_regret_insufficient"],
        "sourcePrefix": {"deckOrder": list(range(32))}}
    def round_quality(*args, **kwargs):
        assert kwargs["_curated_focal_precheck"] is expected
        assert kwargs["graph_workers"] == 4
        if expected:
            raise module._QualityEditorialMiss(evidence)
        raise RuntimeError("ordinary full quality path")
    monkeypatch.setattr(module, "resolve_quality_round_continuation_v1", round_quality)
    def run():
        return executor._resolve_quality_candidate(candidate, maximum_seconds=10,
            graph_workers=4,
            recorded_trace=recorded, recorded_events=(), continue_recorded_prefix=True,
            audit_recorded=scope != "generate", stop_on_recorded_quality_failure=True,
            decision_point={"kind": "play", "selector": {"trickIndex": 6, "position": 3}})
    if expected:
        result = run()
        assert result.status == "not_found" and result.candidate is None
        assert result.reason == "curated_interest_miss"
        assert result.partial_receipt["researchClaimStatus"] == "not_evaluated"
        assert result.partial_receipt["focalPrecheck"] == evidence
        reason = ", ".join(evidence["blockers"])
        assert result.partial_receipt["qualityFailureReason"] == reason
        from baloot_platform.research.service import _operator_candidate_progress_receipt_v1
        event = module._candidate_terminal_progress_v1(SimpleNamespace(
            candidate=None, status=result.status, proof=None, execution_receipt={
                "completedCandidates": 1, "candidateAttemptReason": result.reason,
                "incompleteCompilation": result.partial_receipt}), ordinal=1, decision_kind="play")
        public = _operator_candidate_progress_receipt_v1(event)
        assert public["classification"] == "rejected"
        assert public["incompleteCompilation"]["qualityFailureReason"] == reason
        assert "focalPrecheck" not in event["incompleteCompilation"]
        assert "deckOrder" not in json.dumps(public)
        assert result.partial_receipt["focalPrecheck"] == evidence
    else:
        with pytest.raises(RuntimeError, match="ordinary full quality path"):
            run()


@pytest.mark.parametrize("mutation", ["digest", "rulesHash", "policyHash", "prefixHash", "playIndex", "sourceSeed", "qualified"])
def test_proposal_data_rejects_tampering(native_queue_inputs, rules, mutation):
    from baloot_platform.research.ortools_research_executor import OrToolsUnifiedResearchExecutorV1
    options, row = native_queue_inputs
    if mutation == "digest":
        row["sha256"] = "0" * 64
    else:
        row[mutation] = -1 if mutation in {"playIndex", "sourceSeed"} else "unexpected"
        row["sha256"] = canonical_sha256({key: value for key, value in row.items() if key != "sha256"})
    with pytest.raises(ValueError, match="identity mismatch"):
        OrToolsUnifiedResearchExecutorV1(rules, runtime_proposals={10: row}, **options)
