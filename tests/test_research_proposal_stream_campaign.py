from types import SimpleNamespace
import threading

import pytest

from baloot_platform.research import ortools_research_executor as module


@pytest.fixture
def stream_executor(monkeypatch):
    def initialize(self, rules, runtime_proposals=None, **kwargs):
        self.rules = rules
        self.runtime_proposals = runtime_proposals
        for key, value in kwargs.items():
            setattr(self, key, value)
    monkeypatch.setattr(module.OrToolsUnifiedResearchExecutorV1, '__init__', initialize)
    return module.OrToolsUnifiedResearchExecutorV1(object(),
        backend_id='ortools_cp_sat_v3', bundle_revision_id='test', bundle_hash='hash',
        tactical_detector_profile=None,
        adjudication_authority_document=None, rule_source_catalog=None, witness_policies={},
        quality_policy=None, quality_execution=SimpleNamespace(candidate_source='runtime_proposal_v1'),
        quality_proof_profiles={}, quality_search_profiles={}, generation=None)


def run(executor, proposals, **options):
    return executor.execute_proposal_stream(
        SimpleNamespace(origin=SimpleNamespace(kind='search'), decision_point={'kind': 'play'}, sha256='scenario-hash'),
        None, proposals=proposals, proof_worker_count=options.pop('proof_worker_count', 2),
        maximum_campaign_seconds=options.pop('maximum_campaign_seconds', 10),
        maximum_candidates=options.pop('maximum_candidates', 3), result_count=1, symmetry_policy_id='rules_preserving_v1',
        allow_symmetry_only_variants=True, minimum_decision_state_distance=1, **options)


@pytest.mark.real_clock
@pytest.mark.parametrize('failure', [False, True])
def test_stream_starts_proof_before_exhaustion_and_closes_on_error(stream_executor, monkeypatch, failure):
    from baloot_platform.research.family_catalog import build_family_generation_request_v1
    stream_executor.generation = build_family_generation_request_v1(dict(schemaVersion=1,
        catalogHash='a' * 64, tagQuery={'mode': ['sun']}))
    started, closed, calls = threading.Event(), [], []
    def prove(task, request, **kwargs):
        calls.append(task)
        request.report_checkpoint('{"checkpoint":"serialized"}')
        request.report_worker_cpu_sample(1000000000)
        assert request.progress_snapshot()['qualityCheckpoints']
        started.set()
        if failure:
            raise RuntimeError('proof failure')
        return SimpleNamespace(status='not_found_within_budget', accepted_results=(),
            executions=(), no_good_vectors=(), cpu_seconds=0)
    monkeypatch.setattr(module, '_run_seed_campaign_process_v1', prove)
    def proposals():
        try:
            yield 10, {'witness': 1}
            assert started.wait(2), 'proof did not overlap producer'
            yield 11, {'witness': 2}
        finally:
            closed.append(True)
    if failure:
        with pytest.raises(RuntimeError, match='proof failure'):
            run(stream_executor, proposals())
    else:
        result = run(stream_executor, proposals())
        assert result.status == 'not_found_within_budget'
        assert result.cpu_seconds >= 2
        assert len(result.coordinator_receipt.attempts) == 2
        assert all(item.partial_evidence['qualityCheckpoints'] for item in result.coordinator_receipt.attempts)
        assert result.coordinator_receipt.semantic_sha256 == module.canonical_sha256(
            result.coordinator_receipt._semantic_core_document())
        from baloot_platform.research.service import _project_ortools_campaign_statistics_run
        projected = _project_ortools_campaign_statistics_run({
            'scenarioExecutorIdentity': {'backendIdentity': {'backendId': 'ortools_cp_sat_v3'}},
            'attemptedCandidates': 0, 'resultCount': 0}, result.to_dict())
        assert projected['attemptedCandidates'] == 0
    assert closed == [True]
    for task in calls:
        assert task['generation'] == stream_executor.generation
        assert len(task['runtimeProposals']) == 1
        assert task['options']['terminal_worker_count'] == 1
        assert 0 < task['options']['maximum_campaign_seconds'] <= 10


def test_cancelled_stream_never_dispatches(stream_executor, monkeypatch):
    monkeypatch.setattr(module, '_run_seed_campaign_process_v1',
        lambda *a, **k: pytest.fail('cancelled work dispatched'))
    assert run(stream_executor, iter([(10, {})]), cancel_check=lambda: True).status == 'incomplete'


@pytest.mark.parametrize('workers,candidates,per_candidate', [
    (1, 1, 1), (4, 1, 4), (8, 1, 4), (4, 2, 2),
    (3, 2, 1), (8, 3, 2), (8, 8, 1), (8, 16, 1),
])
def test_stream_shares_worker_grant_without_nested_oversubscription(
    stream_executor, monkeypatch, workers, candidates, per_candidate,
):
    grants = []
    def prove(task, request, **kwargs):
        grants.append(task['options']['terminal_worker_count'])
        return SimpleNamespace(status='not_found_within_budget', accepted_results=(),
            executions=(), no_good_vectors=(), cpu_seconds=0)
    monkeypatch.setattr(module, '_run_seed_campaign_process_v1', prove)
    run(stream_executor, iter((index, {}) for index in range(candidates)),
        proof_worker_count=workers, maximum_candidates=candidates)
    assert grants == [per_candidate] * candidates
    assert min(workers, candidates) * per_candidate <= workers


@pytest.mark.real_clock
def test_expired_stream_never_dispatches(stream_executor, monkeypatch):
    monkeypatch.setattr(module, '_run_seed_campaign_process_v1',
        lambda *a, **k: pytest.fail('expired work dispatched'))
    assert run(stream_executor, iter([(10, {})]), maximum_campaign_seconds=0).status == 'incomplete'


@pytest.mark.real_clock
def test_error_is_not_hidden_by_concurrent_cancellation(stream_executor, monkeypatch):
    cancelled = threading.Event()
    def fail(*args, **kwargs):
        cancelled.set()
        raise RuntimeError('invalid proof evidence')
    monkeypatch.setattr(module, '_run_seed_campaign_process_v1', fail)
    with pytest.raises(RuntimeError, match='invalid proof evidence'):
        run(stream_executor, iter([(10, {})]), cancel_check=cancelled.is_set)


@pytest.mark.real_clock
@pytest.mark.parametrize('proof_workers', [2, 8])
def test_cancelled_stream_retains_private_checkpoint_and_distinct_progress(stream_executor, monkeypatch, proof_workers):
    cancelled, both_started = threading.Event(), threading.Barrier(proof_workers)
    live = []
    def prove(task, request, **kwargs):
        request.report_checkpoint({'privateHand': ['SA']})
        request.report_worker_cpu_sample(1_000_000_000)
        request.report_progress({'currentStage': 'proof', 'completedCandidates': 1})
        both_started.wait(timeout=2)
        cancelled.set()
        raise RuntimeError('campaign shard process cancelled')
    monkeypatch.setattr(module, '_run_seed_campaign_process_v1', prove)
    result = run(stream_executor, iter((10 + index, {}) for index in range(proof_workers)),
                 proof_worker_count=proof_workers, maximum_candidates=proof_workers,
                 cancel_check=cancelled.is_set, progress_callback=live.append)
    assert result.status == 'incomplete'
    assert result.cpu_seconds >= proof_workers
    receipt = result.coordinator_receipt.to_dict()
    assert receipt['semanticSha256'] is None
    assert receipt['cpuTimeCoverage'] == 'sampled_lower_bound'
    assert len({row['shardId'] for row in live}) == proof_workers
    assert all(row['attempt'] == 1 and 'qualityCheckpoints' not in row for row in live)
    assert len(receipt['attempts']) == proof_workers
    assert [row['shardIndex'] for row in receipt['attempts']] == list(range(10, 10 + proof_workers))
    assert all(next(iter(row['partialEvidence']['qualityCheckpoints'].values())) ==
               {'privateHand': ['SA']} for row in receipt['attempts'])
    dispositions = [row for row in live if row.get('proposalDisposition')]
    assert len(dispositions) == proof_workers
    assert all(row['nativeAttemptStarted'] and row['proposalDisposition'] == 'incomplete'
               and row['dispositionReason'] == 'cancelled' and row['currentStage'] == 'proof'
               for row in dispositions)
    assert all(row['partialEvidence']['nativeAttemptStarted']
               and row['partialEvidence']['proposalDisposition'] == 'incomplete'
               for row in receipt['attempts'])
    assert receipt['receiptSha256'] == module.canonical_sha256({
        key: value for key, value in receipt.items() if key != 'receiptSha256'})


def test_proposal_expiring_during_generation_has_explicit_undispatched_receipt(stream_executor, monkeypatch):
    now, live = [0], []
    monkeypatch.setattr(module.time, 'perf_counter', lambda: now[0])
    monkeypatch.setattr(module, '_run_seed_campaign_process_v1',
        lambda *a, **k: pytest.fail('expired proposal dispatched'))
    def proposals():
        now[0] = 11
        yield 10, {}
    result = run(stream_executor, proposals(), progress_callback=live.append)
    assert result.status == 'incomplete' and result.executions == ()
    assert len(live) == 1 and live[0]['proposalOrdinal'] == 10
    assert live[0]['proposalDisposition'] == 'rejected_before_dispatch'
    assert live[0]['dispositionReason'] == 'campaign_budget_exhausted'
    assert live[0]['nativeAttemptStarted'] is False
    assert result.coordinator_receipt.attempts[0].partial_evidence == live[0]


@pytest.mark.real_clock
def test_progress_consumer_failure_cannot_cancel_native_proof(stream_executor, monkeypatch):
    monkeypatch.setattr(module, '_run_seed_campaign_process_v1', lambda *a, **k:
        SimpleNamespace(status='not_found_within_budget', accepted_results=(), executions=(),
                        no_good_vectors=(), cpu_seconds=0))
    def broken_progress(payload):
        raise RuntimeError('telemetry unavailable')
    result = run(stream_executor, iter([(10, {})]), progress_callback=broken_progress)
    evidence = result.coordinator_receipt.attempts[0].partial_evidence
    assert evidence['nativeAttemptStarted'] and evidence['proposalDisposition'] == 'executed'
    assert evidence['dispositionReason'] == 'not_found_within_budget'


@pytest.mark.parametrize('candidate_exists', [False, True])
def test_child_budget_exhaustion_without_execution_still_counts_native_attempt(stream_executor, monkeypatch, candidate_exists):
    live = []
    def prove(task, request, **kwargs):
        request.report_progress({'currentStage': 'shard_budget_exhausted',
            **({'candidateAttemptReason': 'shard_budget_exhausted',
                'incompleteCompilation': {'stage': 'playingQuality', 'exhaustionReason': 'maximum_seconds'}}
               if candidate_exists else {})})
        return SimpleNamespace(status='incomplete', accepted_results=(), executions=(),
                               no_good_vectors=(), cpu_seconds=0)
    monkeypatch.setattr(module, '_run_seed_campaign_process_v1', prove)
    result = run(stream_executor, iter([(10, {})]), progress_callback=live.append)
    terminal = [row for row in live if row.get('proposalDisposition')]
    assert len(terminal) == 1 and terminal[0]['nativeAttemptStarted']
    assert terminal[0]['proposalDisposition'] == 'incomplete'
    assert terminal[0]['dispositionReason'] == 'shard_budget_exhausted'
    if candidate_exists:
        assert terminal[0]['incompleteCompilation']['stage'] == 'playingQuality'
    else:
        assert 'incompleteCompilation' not in terminal[0]
    assert not result.executions and result.coordinator_receipt.attempts[0].status == 'incomplete'
