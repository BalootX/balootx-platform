import copy
from types import SimpleNamespace

import pytest

from baloot_platform.research import ortools_research_executor as module


@pytest.mark.parametrize("resolver", [module.resolve_quality_round_continuation_v1,
    module.resolve_quality_post_auction_continuation_v1])
@pytest.mark.parametrize("graph_workers", [0, 5, True])
def test_quality_stages_reject_invalid_world_worker_grants(resolver, graph_workers):
    with pytest.raises(ValueError, match="granted 1..4"):
        resolver(None, None, {}, None, opportunity_order=(), maximum_nodes=1,
            maximum_depth=1, maximum_worlds=1, maximum_seconds=1, maximum_decision_seconds=1,
            maximum_escalation_decisions=1, maximum_evidence_bytes=1, graph_workers=graph_workers)


def receipt(status='complete'):
    core = {'schemaVersion': 1, 'status': status, 'partialReceipt': None,
            'finalPrefix': {}, 'auctionOutcome': {'allPassed': True}}
    return {**core, 'sha256': module.canonical_sha256(core),
            'executionReceipt': {'evidenceBytes': 10}}


def test_bid_cache_is_bounded_immutable_and_never_reuses_unknown():
    cache = module._BidQualityCacheV1(1, 10000)
    complete = receipt()
    cache.put('first', complete)
    complete['status'] = 'corrupted'
    assert cache.get('first')['status'] == 'complete'
    loaded = cache.get('first')
    loaded['status'] = 'corrupted'
    assert cache.get('first')['status'] == 'complete'
    cache.put('second', receipt('incomplete'))
    assert cache.get('second') is None
    cache.put('second', receipt())
    assert cache.get('first') is None and cache.size <= cache.maximum_bytes
    entry = cache.snapshot()[0]
    entry['result']['status'] = 'corrupted'
    with pytest.raises(ValueError, match='checksum'):
        cache.merge([entry])
    tiny = module._BidQualityCacheV1(1, 1)
    tiny.put('too-large', receipt())
    assert not tiny.values


@pytest.mark.real_clock
def test_auction_generation_uses_frozen_native_policy_and_cache(monkeypatch):
    executor = object.__new__(module.OrToolsUnifiedResearchExecutorV1)
    executor.backend_id = 'ortools_cp_sat_v3'
    executor.rules = object()
    executor.quality_policy = SimpleNamespace(decisions={'bid': {'budgetProfile': 'bid'}})
    executor.quality_execution = SimpleNamespace(maximum_wall_seconds=30,
        maximum_evidence_bytes=10000, escalation_opportunity_order=('north', 'west', 'south', 'east'))
    executor.quality_proof_profiles = {'bid': {'frozen': True}}
    executor.quality_search_profiles = {'bid': {'frozenSearch': True}}
    executor.bid_quality_cache = object()
    monkeypatch.setattr(module, '_quality_budget_snapshots', lambda names, profiles, *, execution_budget=None: (
        {'bid': SimpleNamespace(maximum_nodes=100, maximum_depth=32, maximum_hidden_worlds=2,
            maximum_wall_seconds=5)}, profiles))
    calls = []
    monkeypatch.setattr(module, 'resolve_quality_auction_continuation_v1',
        lambda *a, **kw: calls.append((a, kw)) or {'status': 'incomplete'})
    assert executor.construct_qualified_auction('prefix', maximum_seconds=4) == {'status': 'incomplete'}
    args, options = calls[0]
    assert args == (executor.rules, 'prefix', executor.quality_policy)
    assert options['runtime_maximum_seconds'] == 4 and options['maximum_seconds'] == 30
    assert options['proof_budget_profiles'] is executor.quality_proof_profiles
    assert options['search_budget_profiles'] is executor.quality_search_profiles
    assert options['_bid_quality_cache'] is executor.bid_quality_cache
    assert options['search_generated'] is True and 'recorded_trace' not in options


def test_cached_bid_is_read_back_again_and_context_changes_miss(monkeypatch):
    calls, readbacks = [], []
    request = SimpleNamespace(budget_profile='bid', to_dict=lambda: {'mode': 'bounded', 'actor': 'north'})
    prefix = SimpleNamespace(sha256='prefix')
    rules = SimpleNamespace(sha256='rules')
    monkeypatch.setattr(module, '_resolve_quality_continuation_v1',
        lambda *a, **kw: calls.append(kw) or receipt())
    def read_back(result, remaining, **kwargs):
        readbacks.append(kwargs)
        return copy.deepcopy(result)
    monkeypatch.setattr(module, '_read_back_generated_quality_v1', read_back)
    cache = module._BidQualityCacheV1(10, 100000)
    kwargs = dict(maximum_seconds=10, maximum_decision_seconds=2,
        proof_budget_profiles={'bid': {'frozen': True}}, search_budget_profiles={'bid': {'conflicts': 100}})
    first = module.resolve_quality_auction_continuation_v1(rules, prefix, request, _bid_quality_cache=cache, **kwargs)
    second = module.resolve_quality_auction_continuation_v1(rules, prefix, request, _bid_quality_cache=cache, **kwargs)
    assert len(calls) == 1 and len(readbacks) == 2
    assert first['executionReceipt']['bidCacheHit'] is False and second['executionReceipt']['bidCacheHit'] is True
    assert second['executionReceipt']['decisionTimings'] == []
    assert readbacks[-1]['verified_artifacts'] == frozenset()
    module.resolve_quality_auction_continuation_v1(SimpleNamespace(sha256='different-rules'), prefix,
        request, _bid_quality_cache=cache, **kwargs)
    module.resolve_quality_auction_continuation_v1(rules, SimpleNamespace(sha256='different-prefix'),
        request, _bid_quality_cache=cache, **kwargs)
    module.resolve_quality_auction_continuation_v1(rules, prefix, request,
        _bid_quality_cache=cache, **{**kwargs, 'proof_budget_profiles': {'bid': {'frozen': False}}})
    module.resolve_quality_auction_continuation_v1(rules, prefix, request,
        _bid_quality_cache=cache, **{**kwargs, 'proof_budget_profiles': {**kwargs['proof_budget_profiles'], 'play': {'depth': 8}}})
    module.resolve_quality_auction_continuation_v1(rules, prefix, request,
        _bid_quality_cache=cache, **{**kwargs, 'search_budget_profiles': {**kwargs['search_budget_profiles'], 'play': {'conflicts': 20}}})
    assert len(calls) == 6


def test_bid_cache_binds_non_bid_policy(monkeypatch):
    from baloot_platform.research.scenario_contracts import ScenarioPlayingQualityPolicyV1
    calls = []
    monkeypatch.setattr(module, '_resolve_quality_continuation_v1',
        lambda *a, **kw: calls.append(kw) or receipt())
    monkeypatch.setattr(module, '_read_back_generated_quality_v1', lambda result, *a, **kw: result)
    raw = {'schemaVersion': 2, 'mode': 'budgeted', 'visibility': 'actor_information_set',
        'decisions': {kind: {'utility': 'awarded_point_margin', 'budgetProfile': 'bounded_v1',
            'horizon': {'unit': 'plies', 'value': 1},
            'boundaryEvaluator': 'auction_hand_margin_v1' if kind == 'bid' else 'current_raw_point_margin_v1'}
            for kind in ('bid', 'escalation', 'project', 'play')}}
    cache = module._BidQualityCacheV1(10, 100000)
    rules, prefix = SimpleNamespace(sha256='rules'), SimpleNamespace(sha256='prefix')
    first = ScenarioPlayingQualityPolicyV1.build(raw)
    module.resolve_quality_auction_continuation_v1(rules, prefix, first, maximum_seconds=10, _bid_quality_cache=cache)
    raw['decisions']['play']['horizon']['value'] = 2
    changed = ScenarioPlayingQualityPolicyV1.build(raw)
    assert first.request_for('bid', 'north').to_dict() == changed.request_for('bid', 'north').to_dict()
    module.resolve_quality_auction_continuation_v1(rules, prefix, changed, maximum_seconds=10, _bid_quality_cache=cache)
    assert len(calls) == 2


def test_completed_play_cache_rechecks_native_ownership_and_binds_hidden_context(monkeypatch):
    calls, readbacks, checkpoints = [], [], []
    request = SimpleNamespace(to_dict=lambda: {'mode': 'exact', 'visibility': 'actor_information_set'})
    prefix, rules = SimpleNamespace(sha256='hand-history'), SimpleNamespace(sha256='rules')
    def resolve(*args, **kwargs):
        calls.append(kwargs)
        kwargs['_verified_artifacts'].add('immediate-verification-only')
        return receipt()
    monkeypatch.setattr(module, '_resolve_quality_continuation_v1', resolve)
    def read_back(result, remaining, **kwargs):
        readbacks.append(kwargs)
        return copy.deepcopy(result)
    monkeypatch.setattr(module, '_read_back_generated_quality_v1', read_back)
    cache = module._BidQualityCacheV1(20, 100000)
    kwargs = dict(maximum_seconds=10, maximum_decision_seconds=2, maximum_worlds=2,
        proof_budget_profiles={'play': {'frozen': True}}, private_recall={},
        featured_play_index=24, checkpoint_callback=checkpoints.append)
    owned = set()
    first = module.resolve_quality_play_continuation_v1(rules, prefix, request,
        _play_quality_cache=cache, _verified_artifacts=owned, **kwargs)
    second = module.resolve_quality_play_continuation_v1(rules, prefix, request,
        _play_quality_cache=cache, _verified_artifacts=owned, **kwargs)
    assert len(calls) == 1 and len(readbacks) == 2
    assert first['executionReceipt']['playCacheHit'] is False and second['executionReceipt']['playCacheHit'] is True
    assert readbacks[0]['verified_artifacts'] == {'immediate-verification-only'}
    assert readbacks[-1]['verified_artifacts'] == frozenset()
    assert 'immediate-verification-only' not in str(cache.snapshot())
    assert len(checkpoints) == 2 and cache.snapshot()[0] == checkpoints[-1]['entry']
    for changed in (
        {'private_recall': {'actor': 'west'}}, {'featured_play_index': 25}, {'maximum_worlds': 3},
        {'proof_budget_profiles': {'play': {'frozen': False}}},
        {'recorded_trace': SimpleNamespace(sha256='different-line')},
        {'_curated_focal_only': True},
    ):
        module.resolve_quality_play_continuation_v1(rules, prefix, request,
            _play_quality_cache=cache, **{**kwargs, **changed})
    module.resolve_quality_play_continuation_v1(rules, SimpleNamespace(sha256='different-hand-history'), request,
        _play_quality_cache=cache, **kwargs)
    module.resolve_quality_play_continuation_v1(SimpleNamespace(sha256='different-rules'), prefix, request,
        _play_quality_cache=cache, **kwargs)
    assert len(calls) == 9
    # Readback can refute a cached entry; it is never an admission bypass.
    monkeypatch.setattr(module, '_read_back_generated_quality_v1', lambda *a, **kw: receipt('incomplete'))
    third = module.resolve_quality_play_continuation_v1(rules, prefix, request, _play_quality_cache=cache, **kwargs)
    assert third['status'] == 'incomplete'


def test_play_cache_replays_decision_and_project_checks_after_native_readback(monkeypatch):
    events, calls, readbacks = [], [], []
    checkpoint = {'kind': 'quality_decision_checkpoint_v1', 'stage': 'play',
        'decision': {'playIndex': 16, 'action': 'C8'}, 'sourcePrefix': {'playActions': []},
        'nextPrefix': {'playActions': [{'card': 'C8'}]}, 'publicAwardedProjects': [{'owner': 'north'}],
        'timing': {'elapsedSeconds': 5}}
    def resolve(*a, **kw):
        calls.append(kw)
        kw['checkpoint_callback'](checkpoint)
        return receipt()
    monkeypatch.setattr(module, '_resolve_quality_continuation_v1', resolve)
    monkeypatch.setattr(module, '_read_back_generated_quality_v1',
        lambda result, *a, **kw: readbacks.append(kw) or result)
    cache = module._BidQualityCacheV1(10, 100000)
    args = (SimpleNamespace(sha256='rules'), SimpleNamespace(sha256='hand-history'),
        SimpleNamespace(to_dict=lambda: {'mode': 'exact'}))
    for _ in range(2):
        module.resolve_quality_play_continuation_v1(*args, _play_quality_cache=cache,
            maximum_seconds=10, checkpoint_callback=events.append)
    decision_events = [event for event in events if event['kind'] == checkpoint['kind']]
    assert len(calls) == 1 and len(readbacks) == 2 and len(decision_events) == 2
    assert decision_events[1]['decision'] == decision_events[0]['decision']
    assert decision_events[1]['publicAwardedProjects'] == checkpoint['publicAwardedProjects']
    assert decision_events[1]['timing']['cacheHit'] is True
    merged = module._BidQualityCacheV1(10, 100000)
    merged.merge(cache.snapshot())
    def reject_project(event):
        if event.get('publicAwardedProjects'):
            raise module._QualityScenarioTargetMiss('required project mismatch')
    with pytest.raises(module._QualityScenarioTargetMiss, match='project mismatch'):
        module.resolve_quality_play_continuation_v1(*args, _play_quality_cache=merged,
            maximum_seconds=10, checkpoint_callback=reject_project)
    assert len(calls) == 1


@pytest.mark.real_clock
@pytest.mark.parametrize('exact_tail', [False, True])
@pytest.mark.parametrize('escalation_seconds', [0, 60, 100])
def test_play_stage_shares_remaining_round_deadline(monkeypatch, exact_tail, escalation_seconds):
    import json
    from pathlib import Path
    from baloot_platform.research.scenario_contracts import (
        ExactTracePrefixV1, ScenarioPlayingQualityPolicyV1, ScenarioQualityExecutionBudgetV1)
    raw = json.loads((Path(__file__).parent / 'fixtures/research-native-qualified-round-v3.json').read_text())['exactTracePrefix']
    raw.pop('sha256', None)
    raw.update(projectActions=[], projectDeclarations=[], projectAssertions=[], playActions=[], stateAssertions={}, escalationActions=[])
    prefix = ExactTracePrefixV1.build(raw)
    policy = ScenarioPlayingQualityPolicyV1.build({'schemaVersion': 2, 'mode': 'budgeted',
        'visibility': 'actor_information_set', 'decisions': {kind: {'utility': 'awarded_point_margin',
            'budgetProfile': 'bounded_v1', 'horizon': {'unit': 'plies', 'value': 1},
            'boundaryEvaluator': 'auction_hand_margin_v1' if kind == 'bid' else 'current_raw_point_margin_v1'}
            for kind in ('bid', 'escalation', 'project', 'play')}})
    execution = ScenarioQualityExecutionBudgetV1.build({'schemaVersion': 1, 'maximumWallSeconds': 120,
        'maximumEvidenceBytes': 10000, 'maximumEscalationDecisions': 4,
        'maximumHiddenWorldsPerDecision': 8,
        **({'exactTailProof': True, 'exactPlayFromTrick': 7,
            'earlyPlayPolicyId': 'observer_safe_trick_control_v1'} if exact_tail else {})})
    monkeypatch.setattr(module, '_quality_budget_snapshots', lambda *a, execution_budget=None: ({}, {}))
    clock = [0]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    def escalation(*a, **kw):
        assert kw['maximum_worlds'] == 8
        clock[0] = escalation_seconds
        return {'status': 'complete', 'partialReceipt': None, 'finalPrefix': prefix.to_dict(),
            'initialNormalization': {}, 'initialState': {}, 'projectPlayHandoff': {}, 'artifacts': {}, 'events': []}
    monkeypatch.setattr(module, 'resolve_quality_escalation_continuation_v1', escalation)
    seen = []
    def play(*a, **kw):
        seen.append(kw)
        assert kw['graph_workers'] == 4
        return {'status': 'incomplete', 'partialReceipt': {'exhaustionReason': 'maximum_seconds'},
            'finalPrefix': prefix.to_dict(), 'executionReceipt': {'evidenceBytes': 0}}
    monkeypatch.setattr(module, 'resolve_quality_play_continuation_v1', play)
    result = module.resolve_quality_post_auction_continuation_v1(SimpleNamespace(sha256='rules'), prefix,
        {'kind': 'escalation'}, policy, opportunity_order=('north', 'west', 'south', 'east'),
        maximum_nodes=100, maximum_depth=32, maximum_worlds=512, maximum_seconds=120,
        runtime_maximum_seconds=100, maximum_decision_seconds=60, maximum_escalation_decisions=4,
        maximum_evidence_bytes=10000, execution_budget=execution, graph_workers=4)
    assert result['status'] == 'incomplete'
    if escalation_seconds == 100:
        assert not seen
        assert result['projectPlay'] is None
        assert result['partialReceipt']['exhaustionReason'] == 'maximum_seconds'
        return
    expected = 100 - escalation_seconds
    assert seen[0]['runtime_maximum_seconds'] == expected
    assert seen[0]['maximum_seconds'] == 120
    assert seen[0]['maximum_decision_seconds'] == 60
    assert seen[0]['maximum_worlds'] == 8
    assert result['projectPlay']['status'] == 'incomplete'
    assert result['executionReceipt']['projectPlay']['playStageStatus'] == 'UNKNOWN'
    assert result['executionReceipt']['projectPlay']['playStageMaximumSeconds'] == expected


@pytest.mark.parametrize('reason', ['cancelled', 'maximum_seconds'])
@pytest.mark.parametrize('status', ['complete', 'quality_failed', 'incomplete'])
def test_native_readback_normalization_stop_is_honest_incomplete(monkeypatch, reason, status):
    from baloot_platform.research import result_authority
    def stopped(*a, **kw):
        raise module.ExactNormalizationStopped(reason, evidence={'ownedPrefix': 'prefix'})
    monkeypatch.setattr(result_authority, '_validate_native_decision_quality_evidence_v1', stopped)
    source = receipt(status)
    marker = {'stage': 'curated_focal_precheck', 'reason': 'focal_decision_complete'}
    if status == 'incomplete':
        source['partialReceipt'] = marker
    value = module._read_back_generated_quality_v1(source, lambda: 1)
    assert value['status'] == ('quality_failed' if status == 'quality_failed' else 'incomplete')
    assert value['partialReceipt']['exhaustionReason'] == reason
    retained = value['partialReceipt']
    if status == 'incomplete':
        assert retained['stage'] == 'quality_readback'
        assert retained['evidence']['generationReceipt'] == marker
        retained = retained['evidence']['readbackReceipt']
    assert retained['evidence']['normalizationReceipt']['evidence'] == {'ownedPrefix': 'prefix'}
    assert value['sha256'] == module.canonical_sha256({key: item for key, item in value.items()
        if key not in {'sha256', 'executionReceipt'}})


@pytest.mark.real_clock
@pytest.mark.parametrize('runtime,bid_limit,expected', [(120, 5, 5), (8, 20, 4)])
@pytest.mark.parametrize('suffix_stops', [False, True])
def test_bid_cap_preserves_suffix_work_without_admitting_unknown(monkeypatch, runtime, bid_limit, expected, suffix_stops):
    from baloot_platform.research.scenario_contracts import (
        ExactTracePrefixV1, ScenarioPlayingQualityPolicyV1, ScenarioQualityExecutionBudgetV1)
    import json
    from pathlib import Path
    prefix = json.loads((Path(__file__).parent / 'fixtures/research-native-qualified-round-v3.json').read_text(encoding='utf-8'))['exactTracePrefix']
    prefix.pop('sha256', None)
    prefix['stateAssertions'] = {}
    prefix['playActions'] = prefix['playActions'][:24]
    recorded = ExactTracePrefixV1.build(prefix)
    start = copy.deepcopy(prefix)
    start.update(auctionActions=[], escalationActions=[], projectActions=[], projectDeclarations=[],
                 projectAssertions=[], playActions=[])
    start = ExactTracePrefixV1.build(start)
    policy = ScenarioPlayingQualityPolicyV1.build({'schemaVersion': 2, 'mode': 'budgeted',
        'visibility': 'actor_information_set', 'decisions': {kind: {'utility': 'awarded_point_margin',
            'budgetProfile': 'bid' if kind == 'bid' else 'play',
            'horizon': {'unit': 'plies', 'value': 1},
            'boundaryEvaluator': 'auction_hand_margin_v1' if kind == 'bid' else 'current_raw_point_margin_v1'}
            for kind in ('bid', 'escalation', 'project', 'play')}})
    monkeypatch.setattr(module, '_validate_quality_recorded_trace', lambda *a, **kw: recorded)
    monkeypatch.setattr(module, '_quality_budget_snapshots', lambda *a, execution_budget=None: ({}, {'bid': {}}))
    monkeypatch.setattr(module.ScenarioProofBudgetV1, 'build', lambda *a: SimpleNamespace(
        maximum_wall_seconds=bid_limit, maximum_cpu_seconds=bid_limit))
    bid_args, suffix_args = [], []
    unknown = receipt('incomplete')
    unknown['finalPrefix'] = start.to_dict()
    unknown['sha256'] = module.canonical_sha256({key: value for key, value in unknown.items()
                                               if key not in {'sha256', 'executionReceipt'}})
    monkeypatch.setattr(module, 'resolve_quality_auction_continuation_v1',
        lambda *a, **kw: bid_args.append(kw) or copy.deepcopy(unknown))
    def suffix(*a, **kw):
        suffix_args.append((a, kw))
        if suffix_stops:
            raise module.ExactNormalizationStopped('maximum_seconds')
        return {'status': 'complete', 'scope': 'suffix'}
    monkeypatch.setattr(module, 'resolve_quality_play_continuation_v1', suffix)
    result = module.resolve_quality_round_continuation_v1(SimpleNamespace(sha256='rules'), start,
        {'kind': 'play'}, policy, opportunity_order=('north', 'west', 'south', 'east'),
        maximum_nodes=10, maximum_depth=32, maximum_worlds=2,
        maximum_seconds=120, runtime_maximum_seconds=runtime, maximum_decision_seconds=20,
        maximum_escalation_decisions=4, maximum_evidence_bytes=10000,
        proof_budget_profiles={}, recorded_trace=recorded, recorded_events=[],
        execution_budget=ScenarioQualityExecutionBudgetV1.build({'schemaVersion': 1,
            'maximumWallSeconds': 120, 'maximumEvidenceBytes': 10000, 'maximumEscalationDecisions': 4,
            'exactPlayFromTrick': 7, 'earlyPlayPolicyId': 'observer_safe_trick_control_v1', 'exactTailProof': True}),
        featured_play_index=24, featured_decision_point={'kind': 'play'}, graph_workers=4)
    assert bid_args[0]['runtime_maximum_seconds'] == expected
    assert len(suffix_args) == 1 and len(suffix_args[0][0][1].play_actions) == 24
    assert suffix_args[0][1]['graph_workers'] == 4
    assert result['status'] == 'incomplete' and result['postAuction'] is None
    assert result['executionReceipt']['bidStageStatus'] == 'UNKNOWN'
    if suffix_stops:
        assert result['executionReceipt']['playSuffixAudit']['status'] == 'incomplete'
    else:
        assert result['executionReceipt']['playSuffixAudit']['scope'] == 'suffix'


def test_proved_bad_bid_skips_post_auction_and_exact_tail(monkeypatch):
    import json
    from pathlib import Path
    from baloot_platform.research.scenario_contracts import (
        ExactTracePrefixV1, ScenarioPlayingQualityPolicyV1, ScenarioQualityExecutionBudgetV1)
    raw = json.loads((Path(__file__).parent / 'fixtures/research-native-qualified-round-v3.json').read_text())['exactTracePrefix']
    raw.pop('sha256', None)
    raw.update(auctionActions=[], escalationActions=[], projectActions=[], projectDeclarations=[],
        projectAssertions=[], playActions=[], stateAssertions={})
    start = ExactTracePrefixV1.build(raw)
    policy = ScenarioPlayingQualityPolicyV1.build({'schemaVersion': 2, 'mode': 'budgeted',
        'visibility': 'actor_information_set', 'decisions': {kind: {'utility': 'awarded_point_margin',
            'budgetProfile': 'bounded_v1', 'horizon': {'unit': 'plies', 'value': 1},
            'boundaryEvaluator': 'auction_hand_margin_v1' if kind == 'bid' else 'current_raw_point_margin_v1'}
            for kind in ('bid', 'escalation', 'project', 'play')}})
    failed = receipt('quality_failed')
    failed['finalPrefix'] = start.to_dict()
    monkeypatch.setattr(module, '_quality_budget_snapshots', lambda *a, execution_budget=None: ({}, {}))
    monkeypatch.setattr(module, 'resolve_quality_auction_continuation_v1', lambda *a, **kw: failed)
    monkeypatch.setattr(module, 'resolve_quality_post_auction_continuation_v1',
        lambda *a, **kw: pytest.fail('proved bad bid reached post-auction proof'))
    monkeypatch.setattr(module, 'resolve_quality_play_continuation_v1',
        lambda *a, **kw: pytest.fail('proved bad bid reached exact-tail fallback'))
    result = module.resolve_quality_round_continuation_v1(SimpleNamespace(sha256='rules'), start,
        {'kind': 'play'}, policy, opportunity_order=('north', 'west', 'south', 'east'),
        maximum_nodes=100, maximum_depth=32, maximum_worlds=2, maximum_seconds=120,
        maximum_decision_seconds=60, maximum_escalation_decisions=4, maximum_evidence_bytes=10000,
        stop_on_recorded_quality_failure=True, featured_play_index=24,
        execution_budget=ScenarioQualityExecutionBudgetV1.build({'schemaVersion': 1,
            'maximumWallSeconds': 120, 'maximumEvidenceBytes': 10000, 'maximumEscalationDecisions': 4,
            'exactTailProof': True, 'exactPlayFromTrick': 7, 'earlyPlayPolicyId': 'observer_safe_trick_control_v1'}))
    assert result['status'] == 'quality_failed' and result['postAuction'] is None
