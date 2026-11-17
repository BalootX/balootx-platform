import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from baloot_platform.config import build_bundle
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.family_catalog import ResearchFamilyCatalogV1
from baloot_platform.research.scenario_contracts import (
    ScenarioPlayingQualityPolicyV1, ScenarioQualityExecutionBudgetV1,
)


@pytest.fixture
def module(root):
    spec = importlib.util.spec_from_file_location('research_tag_experiment', root / 'scripts/research_tag_experiment.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def strict_challenge(root):
    raw = yaml.safe_load((root / 'configs/challenge/intelligent_research.yaml').read_text(encoding='utf-8'))
    interest = json.loads((root / 'tests/fixtures/actor_interest_calibration.json').read_text())['cases']['7189']['policy']
    execution = raw['scenarioResearchV4']['qualityExecution']
    execution.update(exactPlayFromTrick=7, exactTailProof=True, earlyPlayPolicyId='observer_safe_trick_control_v1',
        videoInterestingness={**interest, 'baselinePolicyId': 'observer_safe_public_two_trick_control_v6'})
    return raw


def test_strict_authority_does_not_mutate_or_relax_settings(module, strict_challenge):
    before = copy.deepcopy(strict_challenge)
    frozen = module.strict_authority(strict_challenge)
    assert strict_challenge == before
    assert frozen.quality_execution.video_interestingness == before['scenarioResearchV4']['qualityExecution']['videoInterestingness']


def test_experiment_bounds_earlier_decisions_but_proves_the_declared_tail_exactly(module, strict_challenge):
    raw = dict(schemaVersion=2, mode='budgeted', visibility='actor_information_set',
        decisions={kind: dict(utility='awarded_point_margin', budgetProfile='offline_heavyweight_v1',
            horizon={'unit': 'tricks' if kind == 'play' else 'plies', 'value': 1},
            boundaryEvaluator='auction_hand_margin_v1' if kind == 'bid' else 'current_raw_point_margin_v1')
            for kind in ('bid', 'escalation', 'project', 'play')})
    policy = module.tail_quality_policy(raw)
    execution = ScenarioQualityExecutionBudgetV1.build(strict_challenge['scenarioResearchV4']['qualityExecution'])
    for kind in ('bid', 'escalation', 'project'):
        assert policy.request_for(kind, 'north', execution_budget=execution).mode == 'bounded'
    start = 4 * (execution.exact_play_from_trick - 1)
    assert policy.request_for('play', 'north', play_index=start - 1, execution_budget=execution).mode == 'bounded'
    tail = policy.request_for('play', 'north', play_index=start, execution_budget=execution)
    assert tail.mode == 'exact' and tail.horizon is None and tail.boundary_evaluator is None
    with pytest.raises(ValueError, match='bounded prefix'):
        module.tail_quality_policy(dict(schemaVersion=2, mode='exact', visibility='actor_information_set',
            decisions={kind: {'utility': 'awarded_point_margin', 'budgetProfile': 'offline_heavyweight_v1'}
                for kind in ('bid', 'escalation', 'project', 'play')}))


def test_derivative_uses_canonical_fresh_rules_bundle_and_snapshot_hashes(module, root, strict_challenge):
    source = {'snapshot': build_bundle(root, bundle_path='configs/bundle_ortools_research_native_v3_intelligent.yaml').configuration_snapshot()}
    original = copy.deepcopy(source)
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')
    execution = ScenarioQualityExecutionBudgetV1.build(strict_challenge['scenarioResearchV4']['qualityExecution']).to_dict()
    corrupt = {**execution, 'maximumWallSeconds': execution['maximumWallSeconds'] + 1}
    with pytest.raises(ValueError, match='checksum mismatch'):
        module.freeze_inputs(source, corrupt, rules)
    bundle, authority, source_catalog = module.freeze_inputs(source, execution, rules)
    snapshot = bundle.configuration_snapshot()
    assert source == original
    assert bundle.sha256 != source['snapshot']['bundleHash']
    assert snapshot['sha256'] != source['snapshot']['sha256']
    assert snapshot['components']['rules']['sha256'] == rules.sha256
    assert bundle.bundle_profile['profile']['status'] == 'draft'
    assert authority.quality_execution.sha256 == execution['sha256']
    assert bundle.component('certification')['rule_source_catalog']['sha256'] == source_catalog.sha256
    target, _ = module.target_and_scenario(ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml'), module.LANES[1], 0)
    changed = {key: value for key, value in execution.items() if key != 'sha256'}
    changed.update(candidateSource='runtime_proposal_stream_v1', proposalSeedCount=1,
        auditConstructedTrace=True, familyTarget=target, exactPlayFromTrick=5)
    changed = ScenarioQualityExecutionBudgetV1.build(changed).to_dict()
    replacement, replacement_authority, _ = module.freeze_inputs({'snapshot': snapshot}, changed, rules)
    assert replacement.sha256 != bundle.sha256
    assert replacement_authority.sha256 != authority.sha256


def test_corrupt_source_snapshot_fails_before_resolving(module):
    with pytest.raises(ValueError, match='snapshot checksum'):
        module.freeze_inputs({'snapshot': {'components': {}, 'sha256': 'forged'}}, {}, None)


def test_legacy_source_configuration_cannot_be_upgraded_implicitly(module):
    payload = {'profile': {'id': 'historical'}}
    snapshot = {'components': {'challenge': {'payload': payload,
        'sha256': module.canonical_sha256(payload)}}}
    snapshot['sha256'] = module.canonical_sha256(snapshot)
    with pytest.raises(ValueError, match='retained native V3 source configuration'):
        module.freeze_inputs({'snapshot': snapshot}, {'sha256': 'present'}, None)


def test_catalog_selected_project_family_is_a_native_target_without_aliasing(module, root):
    raw = ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml').to_dict()
    raw.pop('sha256')
    project, = ResearchFamilyCatalogV1.load(root / 'configs/research/public_project_rank_inference.yaml').families
    raw['families'].append(project.to_dict())
    catalog = ResearchFamilyCatalogV1.build(raw)
    before = catalog.to_dict()
    lanes = (('round-flip', project.family_id, {'impact': ['round-flip']}), module.LANES[1])
    target, scenario = module.target_and_scenario(catalog, lanes[0], 18, lanes=lanes)
    module.validate_family_target_scenario_v1(target, scenario)
    assert target['familyId'] == project.family_id
    assert {row['id'] for row in target['catalog']['families'] if row['enabled']} == {
        project.family_id, module.LANES[1][1]}
    assert sum(p.kind == 'project.exists' for p in scenario.constraints.predicates()) == 2
    assert catalog.to_dict() == before


@pytest.mark.parametrize('index,family_id', [(0, 'absent'), (1, 'negative-project')])
def test_selected_family_must_exist_and_support_its_lane(module, root, index, family_id):
    catalog = ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml')
    lanes = list(module.LANES)
    lanes[index] = (lanes[index][0], family_id, lanes[index][2])
    with pytest.raises(ValueError):
        module.target_and_scenario(catalog, lanes[index], 0, lanes=lanes)


@pytest.mark.parametrize('families', [[], ['one'], [True, 'stopper_guard']])
def test_invalid_family_selection_fails_before_loading(module, tmp_path, families):
    with pytest.raises(ValueError, match='two catalog family IDs'):
        module.run(SimpleNamespace(proof_workers=1, seconds=2, slice_seconds=1, seed=1,
            families=families, authority=tmp_path / 'absent.json'))


@pytest.mark.parametrize('mutation', [
    lambda raw: raw['videoInterestingness'].update(baselinePolicyId='observer_safe_public_exhausted_suit_v5'),
    lambda raw: raw['videoInterestingness'].update(allowTieRuleOnlyLessons=True),
    lambda raw: raw.update(featuredPlayNativeProof=False),
    lambda raw: raw.update(exactTailProof=False),
])
def test_weak_authority_has_no_default_fallback(module, strict_challenge, mutation):
    mutation(strict_challenge['scenarioResearchV4']['qualityExecution'])
    with pytest.raises(ValueError, match='strict v6 authority'):
        module.strict_authority(strict_challenge)


@pytest.mark.parametrize('lane_index', [0, 1])
def test_private_targets_are_exact_conjuncts_and_keep_bundled_families_disabled(module, root, lane_index):
    catalog = ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml')
    before = catalog.to_dict()
    for ordinal in (0, 7, 12, 47):
        target, scenario = module.target_and_scenario(catalog, module.LANES[lane_index], ordinal)
        module.validate_family_target_scenario_v1(target, scenario)
        assert target['bindings']['contract'] == 'sun' and target['trump'] is None
        assert scenario.constraints.sha256 == next(row for row in ResearchFamilyCatalogV1.build(target['catalog']).families
            if row.family_id == target['familyId']).instantiate(target['bindings']).sha256
    assert catalog.to_dict() == before
    assert not any(row['enabled'] for row in before['families'])


def test_target_order_prioritizes_shorter_tails_without_dropping_bands(module, root):
    catalog = ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml')
    target_indices = [module.target_and_scenario(catalog, module.LANES[0], index)[0]['bindings']['playIndex']
        for index in range(12)]
    assert target_indices == list(range(27, 15, -1))


def test_library_is_only_seed_hints_and_target_freezes_matching_opening(module, tmp_path, root):
    prefix = {'dealer': 'south'}
    receipt = {'rulesHash': 'rules', 'status': 'complete', 'operation': 'generate', 'finalPrefix': prefix,
        'auctionOutcome': {'contract': 'sun', 'buyer': 'west', 'allPassed': False}, 'sha256': 'receipt'}
    module.retain(tmp_path / 'qualified-00001.json.gz', {'seed': 29, 'prefix': prefix, 'receipt': receipt})
    hints = module.auction_seed_hints(tmp_path, 'rules')
    assert hints == [{'seed': 29, 'dealer': 'south', 'buyer': 'west', 'receiptHash': 'receipt'}]
    catalog = ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml')
    target, scenario = module.target_and_scenario(catalog, module.LANES[0], 0, opening=hints[0])
    assert target['dealer'] == 'south' and target['buyer'] == 'west'
    assert target['bindings']['contract'] == 'sun'
    module.validate_family_target_scenario_v1(target, scenario)
    with pytest.raises(ValueError, match='same-rules'):
        module.auction_seed_hints(tmp_path, 'other-rules')


def test_hint_schedule_covers_opening_seed_binding_product_without_duplicate_contexts(module, root):
    hints = [
        {'seed': 29, 'dealer': 'north', 'buyer': 'north'},
        {'seed': 41, 'dealer': 'south', 'buyer': 'west'},
        {'seed': 31, 'dealer': 'north', 'buyer': 'north'},
        {'seed': 43, 'dealer': 'south', 'buyer': 'west'},
    ]
    before = copy.deepcopy(hints)
    catalog = ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml')
    covered = []
    for pair in range(16):
        opening, ordinal = module._opening_and_context_ordinal(hints, pair)
        assert opening == hints[pair % 2] and ordinal == pair // 2
        ordinal, seeds = module._binding_and_auction_seeds(hints, opening, ordinal)
        target, _ = module.target_and_scenario(catalog, module.LANES[0], ordinal, opening=opening)
        covered.append((target['dealer'], target['buyer'], target['bindings']['playIndex'], seeds[0]))
    assert len(set(covered)) == 16
    assert set(covered) == {(row['dealer'], row['buyer'], index, row['seed'])
                            for row in hints for index in range(24, 28)}
    assert module._opening_and_context_ordinal(None, 7) == ({'dealer': 'east', 'buyer': 'west'}, 0)
    assert module._opening_and_context_ordinal([], 7) == ({'dealer': 'east', 'buyer': 'west'}, 0)
    assert hints == before


@pytest.mark.parametrize('decision_count', [1, 4, 12])
def test_fresh_schedule_separates_actor_buyer_dealer_and_visits_positions_first(module, decision_count):
    seats = ('north', 'west', 'south', 'east')
    covered = []
    for pair in range(64 * decision_count):
        opening, ordinal = module._opening_and_context_ordinal(None, pair, decision_count=decision_count)
        binding, seeds = module._binding_and_auction_seeds(None, opening, ordinal, decision_count=decision_count)
        assert seeds is None
        covered.append((opening['dealer'], opening['buyer'], seats[binding // decision_count],
                        binding % decision_count))
    assert len(set(covered)) == len(covered)
    assert set(covered) == {(dealer, buyer, actor, position)
                            for dealer in seats for buyer in seats for actor in seats
                            for position in range(decision_count)}
    for start in range(0, len(covered), decision_count):
        group = covered[start:start + decision_count]
        assert len({row[:3] for row in group}) == 1
        assert [row[3] for row in group] == list(range(decision_count))


@pytest.mark.parametrize('binding_count', [2, 4])
def test_hint_seed_rotation_precedes_truncation_and_covers_each_binding(module, binding_count):
    hints = [
        {'seed': seed, 'dealer': dealer, 'buyer': buyer}
        for dealer, buyer, seeds in [('north', 'north', (29, 31, 37)),
                                     ('south', 'west', (41, 43))]
        for seed in seeds
    ]
    before = copy.deepcopy(hints)
    covered = []
    for opening, group in [(hints[0], [29, 31, 37]), (hints[3], [41, 43])]:
        for ordinal in range(binding_count * len(group)):
            binding, seeds = module._binding_and_auction_seeds(hints, opening, ordinal)
            assert binding == ordinal // len(group)
            assert seeds[0] == group[ordinal % len(group)]
            assert sorted(seeds) == sorted(group)
            covered.append((opening['dealer'], binding, seeds[:2][0]))
    assert len(covered) == len(set(covered))
    assert set(covered) == {(dealer, binding, seed)
                            for dealer, seeds in [('north', (29, 31, 37)), ('south', (41, 43))]
                            for binding in range(binding_count) for seed in seeds}
    assert module._binding_and_auction_seeds(None, None, 7) == (7, None)
    assert module._binding_and_auction_seeds([], None, 7) == (7, None)
    with pytest.raises(ValueError, match='matching opening'):
        module._binding_and_auction_seeds(hints, {'dealer': 'east', 'buyer': 'east'}, 0)
    assert hints == before


@pytest.mark.parametrize('decision_count', [1, 4, 12])
def test_hint_schedule_visits_positions_first_without_losing_ragged_seed_groups(module, decision_count):
    hints = [{'seed': seed, 'dealer': dealer, 'buyer': buyer}
             for dealer, buyer, seeds in [('north', 'north', (29, 31, 37)),
                                          ('south', 'west', (41, 43))]
             for seed in seeds]
    before = copy.deepcopy(hints)
    for pair in range(2 * decision_count):
        opening, ordinal = module._opening_and_context_ordinal(hints, pair, decision_count=decision_count)
        binding, seeds = module._binding_and_auction_seeds(hints, opening, ordinal, decision_count=decision_count)
        assert opening == hints[0 if pair // decision_count == 0 else 3]
        assert binding == pair % decision_count
        assert seeds[0] == opening['seed']
    for opening, source_seeds in [(hints[0], [29, 31, 37]), (hints[3], [41, 43])]:
        covered = []
        for ordinal in range(2 * decision_count * len(source_seeds)):
            binding, seeds = module._binding_and_auction_seeds(hints, opening, ordinal,
                                                            decision_count=decision_count)
            assert binding % decision_count == ordinal % decision_count
            assert seeds[0] == source_seeds[ordinal // decision_count % len(source_seeds)]
            assert sorted(seeds) == sorted(source_seeds)
            covered.append((binding, seeds[:1][0]))
        assert len(set(covered)) == len(covered)
        assert set(covered) == {(binding, seed) for binding in range(2 * decision_count) for seed in source_seeds}
    assert module._binding_and_auction_seeds(None, None, 7, decision_count=decision_count) == (7, None)
    assert hints == before


def test_empty_or_inconsistent_library_cannot_fall_back_to_random_openings(module, tmp_path):
    with pytest.raises(ValueError, match='compatible Sun'):
        module.auction_seed_hints(tmp_path, 'rules')
    receipt = {'rulesHash': 'rules', 'status': 'complete', 'operation': 'generate', 'finalPrefix': {'dealer': 'west'}}
    module.retain(tmp_path / 'qualified-00001.json.gz', {'seed': 29, 'prefix': {'dealer': 'south'}, 'receipt': receipt})
    with pytest.raises(ValueError, match='generated auction'):
        module.auction_seed_hints(tmp_path, 'rules')


@pytest.mark.parametrize('status', ['proposal', 'incomplete', 'semantic_postcondition_miss'])
def test_unaccepted_execution_cannot_be_a_witness(module, status):
    with pytest.raises(ValueError, match='not an accepted witness'):
        module.accepted_packet(SimpleNamespace(status=status, candidate=object()), rules=None,
            scenario=None, target=None, maximum_seconds=1)


def test_retained_evidence_is_immutable_and_compressed_bytes_replay(module, tmp_path):
    path = tmp_path / 'attempt.json.gz'
    document = {'action': 'SA:ekka', 'status': 'incomplete'}
    module.retain(path, document)
    with module.gzip.open(path, 'rt', encoding='utf-8') as stream:
        assert json.load(stream) == document
    with pytest.raises(ValueError, match='already exists'):
        module.retain(path, {'status': 'complete'})


@pytest.mark.parametrize('root_replays', [False, True])
def test_packet_retains_companions_but_does_not_invent_paired_replay_or_novelty(module, monkeypatch, root_replays):
    tags = {'tags': {'impact': ['round-flip'], 'technique': ['guard/protect']},
        'sources': {'proofArtifactHash': 'proof', 'baselineAction': 'SA:ekka'},
        'classifier': {'deductions': ['retained_return_control']}}
    group = {'artifacts': {'proof': {'native': True}}, 'baselineEvidence': {'proof': {'baseline': True}},
        'tieRuleEvidence': {'proof': {'buyer_makes': {'value': 1}, 'buyer_loses': {'value': 1}}}}
    if root_replays:
        group['artifacts']['proof'].update(proof={'selectedAction': 'CA',
            'rootActions': [{'action': 'CA', 'value': 30}, {'action': 'SA:ekka', 'value': -20}]},
            rootWitnesses=[{'action': action, 'worstCaseValue': value, 'worldId': world,
                'principalVariation': [action, 'H7']} for action, value, world in
                (('CA', 30, 'one'), ('SA:ekka', -20, 'two'))])
    trace = {'exactTracePrefix': {}, 'lessonTags': tags, 'playingQualityEvidence': {'receipt': {
        'postAuction': {'projectPlay': group}, 'finalPrefix': {'playActions': [{'card': 'HA'}]}}}}
    monkeypatch.setattr(module.ExactTracePrefixV1, 'build', lambda value: value)
    monkeypatch.setattr(module, 'OrToolsExactStateNormalizerV1', lambda *a, **kw: SimpleNamespace(
        normalize=lambda *a, **kw: SimpleNamespace(to_dict=lambda: {'native': True})))
    monkeypatch.setattr(module, 'derive_trace_lesson_tags_v1', lambda *a, **kw: copy.deepcopy(tags))
    item = SimpleNamespace(status='qualified', candidate=SimpleNamespace(trace=trace), sha256='execution', semantic_hash='semantic')
    packet = module.accepted_packet(item, rules=object(), scenario=SimpleNamespace(decision_point={'kind': 'play'}),
        target={'tagQuery': {'impact': ['round-flip']}}, maximum_seconds=1)
    assert packet['tieRuleEvidence'] == group['tieRuleEvidence']['proof']
    assert packet['pairedContinuations']['baselineAction'] == 'SA:ekka'
    if root_replays:
        pair = packet['pairedContinuations']
        assert pair['status'] == 'exact_root_worst_cases'
        assert pair['selected']['worldId'] != pair['baseline']['worldId']
        assert pair['selected']['worstCaseValue'] == 30 and pair['baseline']['worstCaseValue'] == -20
        assert pair['scope'] == 'independent_worst_case_worlds; not_actual_deal_or_inference_ablation'
        for field, value in (('principalVariation', ['D7']), ('worstCaseValue', -999), ('worldId', None)):
            row = group['artifacts']['proof']['rootWitnesses'][1]
            original = row[field]
            row[field] = value
            with pytest.raises(ValueError, match='does not bind'):
                module.accepted_packet(item, rules=object(), scenario=SimpleNamespace(decision_point={'kind': 'play'}),
                    target={'tagQuery': {'impact': ['round-flip']}}, maximum_seconds=1)
            row[field] = original
    else:
        assert packet['pairedContinuations']['status'] == 'missing_baseline_replay_line'
    assert packet['inferenceAblation']['status'] == 'not_run'
    assert packet['audit']['newLesson'] is None
    trace['lessonTags'] = {'tags': {'impact': ['round-flip']}}
    with pytest.raises(ValueError, match='independently derived'):
        module.accepted_packet(item, rules=object(), scenario=SimpleNamespace(decision_point={'kind': 'play'}),
            target={'tagQuery': {'impact': ['round-flip']}}, maximum_seconds=1)


@pytest.mark.parametrize('seconds,slice_seconds,seed', [(3601, 60, 1), (120, 61, 1), (120, 60, -1), (float('nan'), 60, 1)])
def test_invalid_budget_fails_before_loading_or_dispatch(module, tmp_path, seconds, slice_seconds, seed):
    with pytest.raises(ValueError):
        module.run(SimpleNamespace(seconds=seconds, slice_seconds=slice_seconds, seed=seed,
            proof_workers=8, authority=tmp_path / 'absent.json'))


@pytest.mark.parametrize('proof_workers', [0, 11, True, '8'])
def test_invalid_parallelism_fails_before_loading_or_dispatch(module, tmp_path, proof_workers):
    with pytest.raises(ValueError, match='proof_workers'):
        module.run(SimpleNamespace(proof_workers=proof_workers, authority=tmp_path / 'absent.json'))


@pytest.mark.parametrize('start_pair', [-1, True, 1.5])
def test_invalid_target_offset_fails_before_loading_or_dispatch(module, tmp_path, start_pair):
    with pytest.raises(ValueError, match='start_pair'):
        module.run(SimpleNamespace(proof_workers=1, start_pair=start_pair,
            authority=tmp_path / 'absent.json'))


@pytest.mark.real_clock
@pytest.mark.parametrize('cancelled_native', [False, True])
@pytest.mark.parametrize('proof_workers,tail_first', [(1, False), (8, False), (10, False), (1, True)])
def test_run_counts_canonical_accepted_not_proposals_and_records_freeze_uncertainty(module, root, strict_challenge, monkeypatch, tmp_path, cancelled_native, proof_workers, tail_first):
    start_pair = 2 if proof_workers == 1 else 0
    if proof_workers == 1:
        inherited = strict_challenge['scenarioResearchV4']['qualityExecution']
        inherited.update(candidateSource='runtime_proposal_stream_v1', proposalSeedCount=16, auditConstructedTrace=True,
            proposalPlayIndices=[24], familyCatalog=ResearchFamilyCatalogV1.load(
                root / 'configs/research/families.yaml').to_dict())
    policy_search = 'time_budgeted_symbolic_v1' if proof_workers == 8 else None
    if policy_search is not None:
        strict_challenge['scenarioResearchV4']['proofBudgets']['profiles']['bounded_v1']['policySearch'] = policy_search
    use_hints = proof_workers == 8 and not cancelled_native
    if use_hints:
        start_pair = 1
        strict_challenge['scenarioResearchV4']['searchBudgets']['profiles']['bounded_v1']['maximumCandidates'] = 2
    authority = module.strict_authority(strict_challenge)
    supplied = {'snapshot': {'components': {}, 'mode': 'test'}}
    input_path = tmp_path / 'input.json'
    input_path.write_text(json.dumps(supplied), encoding='utf-8')
    execution_path = tmp_path / 'execution.json'
    execution_path.write_text(json.dumps(authority.quality_execution.to_dict()), encoding='utf-8')
    policy_path = tmp_path / 'policy.json'
    policy_path.write_text(json.dumps(dict(schemaVersion=2, mode='budgeted', visibility='actor_information_set',
        decisions={kind: dict(utility='awarded_point_margin', budgetProfile='bounded_v1',
            horizon={'unit': 'tricks' if kind == 'play' else 'plies', 'value': 1},
            boundaryEvaluator='auction_hand_margin_v1' if kind == 'bid' else 'current_raw_point_margin_v1')
            for kind in ('bid', 'escalation', 'project', 'play')})), encoding='utf-8')
    rules = SimpleNamespace(data={}, sha256='rules')
    bundle = SimpleNamespace(sha256='bundle', configuration_snapshot=lambda: {})
    monkeypatch.setattr(module.RuleBook, 'load', lambda path: rules)
    def freeze_inputs(source, execution, rules):
        assert execution['sha256'] == ScenarioQualityExecutionBudgetV1.build(execution).sha256
        return bundle, authority, SimpleNamespace(data={}, sha256='catalog')
    monkeypatch.setattr(module, 'freeze_inputs', freeze_inputs)
    catalog = ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml')
    selected_families = ('positive-project', 'stopper_guard') if proof_workers == 10 and not cancelled_native else None
    selected_lanes = tuple((label, selected_families[index] if selected_families else family_id, query)
        for index, (label, family_id, query) in enumerate(module.LANES))
    targets = {lane[0]: module.target_and_scenario(catalog, lane, 0, lanes=selected_lanes) for lane in selected_lanes}
    hints = [{'seed': seed, 'dealer': 'north', 'buyer': 'north', 'receiptHash': f'receipt-{seed}'}
             for seed in (29, 31, 37)]
    monkeypatch.setattr(module, 'auction_seed_hints', lambda directory, rules_hash: hints)
    seen_openings = []
    def target_for_run(catalog, lane, ordinal, *, opening=None, lanes=None):
        assert lanes == selected_lanes and lane in selected_lanes
        assert opening is not None
        if use_hints:
            assert opening == hints[0]
        else:
            assert set(opening) == {'dealer', 'buyer'}
            assert all(value in {'north', 'west', 'south', 'east'} for value in opening.values())
        seen_openings.append(opening)
        return targets[lane[0]]
    monkeypatch.setattr(module, 'target_and_scenario', target_for_run)
    clock = [0.0]
    def now():
        clock[0] += .05
        return clock[0]
    monkeypatch.setattr(module.time, 'perf_counter', now)
    lanes = []
    def proposals(*a, **kw):
        if tail_first:
            assert 'auction_provider' not in kw and 'source_seeds' not in kw
            assert kw['request'].visibility == 'actor_information_set' and kw['request'].mode == 'exact'
            assert kw['scenario'].sha256 == targets[selected_lanes[len(lanes) % len(selected_lanes)][0]][1].sha256
        else:
            assert callable(kw['auction_provider'])
        assert kw['playing_quality'].mode == 'budgeted'
        assert kw['playing_quality'].visibility == 'actor_information_set'
        if use_hints:
            pair = start_pair + len(lanes) // len(selected_lanes)
            family = next(family for family in catalog.families
                          if family.family_id == selected_lanes[len(lanes) % len(selected_lanes)][1])
            rotation = pair // len(family.parameters.choices(('sun',))['playIndex'])
            offset = rotation % len(hints)
            assert kw['source_seeds'] == ([29, 31, 37][offset:] + [29, 31, 37][:offset])[:2]
        else:
            assert kw.get('source_seeds') is None
        kw['progress_callback']({'proposalCount': 1, 'constructionStatus': 'proposal'})
        yield 1, {'proposalOnly': True}
    monkeypatch.setattr(module, 'family_proposal_stream_v1', proposals)
    monkeypatch.setattr(module, 'tail_first_proposal_stream', proposals)
    class Executor:
        def __init__(self, *a, **kw):
            self.target = kw['quality_execution'].family_target
            assert kw['quality_execution'].family_catalog is None
            assert kw['quality_execution'].proposal_play_indices == (self.target['bindings']['playIndex'],)
        def construct_qualified_auction(self, *a, **kw):
            pytest.fail('mock proposal producer must not execute native generation')
        def execute_proposal_stream(self, *a, **kw):
            assert kw['proof_worker_count'] == proof_workers and not kw['allow_symmetry_only_variants']
            assert kw['policy_search'] == policy_search
            assert a[1].visibility == 'actor_information_set'
            assert list(kw['proposals']) == [(1, {'proposalOnly': True})]
            lanes.append(self.target['familyId'])
            item = SimpleNamespace(status='incomplete', execution_receipt={'reason': 'budget'}, to_dict=lambda: {'status': 'incomplete'})
            kw['progress_callback']({'currentStage': 'playingQuality', 'proposalOrdinal': 1,
                'nativeAttemptStarted': True, 'proposalDisposition': 'incomplete' if cancelled_native else 'executed',
                'dispositionReason': 'campaign_budget_exhausted' if cancelled_native else 'proof_incomplete'})
            return SimpleNamespace(cpu_seconds=.01, executions=() if cancelled_native else (item,), accepted_results=(),
                status='incomplete', to_dict=lambda: {'status': 'incomplete'})
    monkeypatch.setattr(module, 'OrToolsUnifiedResearchExecutorV1', Executor)
    result = module.run(SimpleNamespace(seconds=.5, slice_seconds=.2, seed=1, authority=input_path,
        proof_workers=proof_workers, start_pair=start_pair, auction_library=tmp_path if use_hints else None,
        families=selected_families, tail_first=tail_first,
        execution=execution_path, policy=policy_path, source_commit='exact-source', source_tree='exact-tree',
        rules=Path('rules'), catalog=root / 'configs/research/families.yaml', output=tmp_path / 'result'))
    assert lanes[:2] == [lane[1] for lane in selected_lanes]
    assert sum(row['constructed'] for row in result['funnels'].values()) > 0
    assert result['unionDistinct'] == 0 and not any(result['nativeTaggedDistinctByTag'].values())
    assert result['activeSeconds'] is None and result['auditedNewByTag'] is None
    assert result['proofWorkers'] == proof_workers
    assert result['startPair'] == start_pair
    assert result['slices'][0]['ordinal'] == start_pair * 2
    saved_authority = json.loads((tmp_path / 'result/authority.json').read_text(encoding='utf-8'))
    assert saved_authority['operationalBudget']['proofWorkers'] == proof_workers
    assert saved_authority['operationalBudget']['startPair'] == start_pair
    assert saved_authority['operationalBudget']['construction'] == (
        'native_tail_first_proposal_v1' if tail_first else 'qualified_history_v1')
    assert saved_authority['operationalBudget']['targetSchedule'] == (
        'opening_decision_seed_binding_cartesian_v3' if use_hints else 'opening_decision_binding_cartesian_v2')
    assert saved_authority['operationalBudget']['laneFamilies'] == {lane[0]: lane[1] for lane in selected_lanes}
    if use_hints:
        assert seen_openings[:2] == [hints[0], hints[0]]
        assert saved_authority['auctionSeedHints']['openings'] == hints
        assert saved_authority['auctionSeedHints']['sha256'] == module.canonical_sha256(hints)
    assert result['acceptanceStatus'].startswith('inconclusive')
    assert sum(row['proofAttempts'] for row in result['funnels'].values()) == len(lanes)
    assert sum(row['completedExecutions'] for row in result['funnels'].values()) == (0 if cancelled_native else len(lanes))
    reason = 'campaign_budget_exhausted' if cancelled_native else 'proof_incomplete'
    assert sum(row.get(reason, 0) for row in result['rejections'].values()) == len(lanes)
    with module.gzip.open(tmp_path / f'result/slice-{start_pair * 2:03}.json.gz', 'rt', encoding='utf-8') as stream:
        saved = json.load(stream)
    assert saved['proposalDispositions'][0]['dispositionReason'] == reason
    assert (tmp_path / 'result/authority.json').exists() and (tmp_path / 'result/summary.json').exists()


@pytest.mark.parametrize('status', ['feasible', 'incomplete'])
@pytest.mark.parametrize('bid_boundary', ['auction_hand_margin_v1', 'current_raw_point_margin_v1'])
def test_tail_constructor_exports_only_actions_and_preserves_actor_proof_scope(
    module, root, strict_challenge, monkeypatch, status, bid_boundary,
):
    from baloot_platform.research import ortools_tail_first_candidate as constructor
    from baloot_platform.research.family_proposal_stream import tail_first_proposal_stream_v1
    from baloot_platform.research.scenario_contracts import ScenarioProofRequestV1
    assert module.tail_first_proposal_stream is tail_first_proposal_stream_v1

    target, scenario = module.target_and_scenario(
        ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml'), module.LANES[1], 0)
    raw = copy.deepcopy(strict_challenge['scenarioResearchV4']['qualityExecution'])
    raw.update(candidateSource='runtime_proposal_stream_v1', proposalSeedCount=2,
        auditConstructedTrace=True, familyTarget=target)
    execution = ScenarioQualityExecutionBudgetV1.build(raw)
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode='exact', visibility='actor_information_set',
        strategyClass='pure_deterministic', claim='winning_root_action_count', targetTeam='north_south',
        utility='awarded_point_margin', successThreshold=1, budgetProfile='late_game_exact_v1'))
    before = (scenario.to_dict(), request.to_dict(), execution.to_dict())
    index = target['bindings']['playIndex']
    # Synthetic model output tests the transport boundary, not legality or strategic truth.
    cards = list(module.FULL_DECK)
    cards[index], cards[7] = cards[7], cards[index]
    trace = dict(deckOrder=list(range(32)), dealer=target['dealer'],
        projects={'projects': []}, auction={'completion': {'actions': []}},
        escalation={'observation': {'actions': []}},
        play={'plays': [dict(actor=target['bindings']['seat'], card=card.code, ekka=step == index)
                        for step, card in enumerate(cards)]})
    calls, progress = [], []
    class Compiler:
        def __init__(self, **kwargs):
            assert kwargs['native_cp_sat_scoring'] is True
            assert kwargs['proof_request'].visibility == 'complete_state'
            assert kwargs['maximum_proof_nodes'] == 100 and kwargs['maximum_proof_depth'] == 8
            assert kwargs['maximum_line_count_limbs'] == 12
            assert kwargs['maximum_prefix_lift_attempts'] == 2
            assert kwargs['require_optimal_auction_actions'] is (bid_boundary == 'auction_hand_margin_v1')
            assert kwargs['policy_prefix_steps'] == 24
            assert kwargs['plan'].boundary_combinations == ((target['dealer'], 'sun', target['buyer'], None),)
        def __call__(self, construction, excluded, **kwargs):
            assert construction.sha256 != scenario.sha256
            assert any(child.sha256 == scenario.constraints.sha256 for child in construction.constraints.children)
            assert kwargs['maximum_conflicts'] == 500 and 0 < kwargs['maximum_seconds'] <= .5
            assert len(excluded) == (len(calls) if status == 'feasible' else 0)
            calls.append(kwargs['seed'])
            return SimpleNamespace(status=status, reason='test stop' if status == 'incomplete' else None,
                candidate=SimpleNamespace(trace=trace) if status == 'feasible' else None,
                graph='UNTRUSTED', proof_outcome='UNTRUSTED', state='UNTRUSTED')
    monkeypatch.setattr(constructor, 'OrToolsTailFirstCandidateCompilerV1', Compiler)
    rules = SimpleNamespace(sha256='a' * 64)
    policy = ScenarioPlayingQualityPolicyV1.build(dict(schemaVersion=1, mode='budgeted',
        decisions={kind: dict(utility='awarded_point_margin', budgetProfile='bounded_v1',
            horizon={'unit': 'tricks' if kind == 'play' else 'plies', 'value': 1},
            boundaryEvaluator=bid_boundary if kind == 'bid' else 'current_raw_point_margin_v1')
            for kind in ('bid', 'escalation', 'project', 'play')}))
    parameters = dict(seed=10, maximumCampaignSeconds=10, maximumSeconds=.5, maximumCandidates=2,
        maximumConflicts=500, maximumLineCountLimbs=12)
    kwargs = dict(scenario=scenario, request=request, proof={'maximumNodes': 100, 'maximumDepth': 8},
        search={'maximumPrefixLiftAttempts': 2}, playing_quality=policy,
        campaign_started=module.time.perf_counter(), cancel_check=lambda: False, progress_callback=progress.append)
    rows = list(module.tail_first_proposal_stream(rules, execution, parameters, **kwargs))
    assert calls == [10, 11]
    assert len(rows) == (2 if status == 'feasible' else 0)
    for (ordinal, row), report in zip(rows, progress):
        assert ordinal == row['sourceSeed']
        assert set(row) == {'prefix', 'prefixHash', 'rulesHash', 'policyHash', 'playIndex', 'actor',
                           'action', 'sourceHash', 'sourceSeed', 'sha256'}
        assert row['sha256'] == module.canonical_sha256({key: value for key, value in row.items() if key != 'sha256'})
        assert row['sourceHash'] == module.canonical_sha256(report['constructionSource'])
        assert row['prefixHash'] == row['prefix']['sha256']
        assert len(row['prefix']['playActions']) == index and row['prefix']['stateAssertions'] == {}
        assert row['action'] == 'SA+ekka'
        assert report['proposalOnly'] is True and report['proposalCount'] == 1
    assert (scenario.to_dict(), request.to_dict(), execution.to_dict()) == before
    calls.clear()
    assert list(module.tail_first_proposal_stream(rules, execution, parameters, **{**kwargs,
        'cancel_check': lambda: True})) == []
    assert calls == []
    with pytest.raises(ValueError, match='familyOpeningMode'):
        list(module.tail_first_proposal_stream(rules,
            SimpleNamespace(family_opening_mode='required_native_opening'), parameters, **kwargs))
    with pytest.raises(ValueError, match='unchanged exact actor'):
        list(module.tail_first_proposal_stream(rules, execution, parameters, **{**kwargs,
            'request': SimpleNamespace(mode='bounded', visibility='actor_information_set')}))
