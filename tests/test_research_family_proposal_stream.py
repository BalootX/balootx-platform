import copy
import importlib.util
import json
from types import SimpleNamespace

import pytest
import yaml

from baloot_platform.research import family_proposal_stream as module
from baloot_platform.research.family_catalog import ResearchFamilyCatalogV1
from baloot_platform.research.scenario_contracts import ResearchScenarioV1, ScenarioQualityExecutionBudgetV1


def project_quality(*, mode='exact', visibility='actor_information_set'):
    from baloot_platform.research.scenario_contracts import ScenarioPlayingQualityPolicyV1
    return ScenarioPlayingQualityPolicyV1.build({'schemaVersion': 2, 'mode': mode, 'visibility': visibility,
        'decisions': {kind: {'utility': 'awarded_point_margin', 'budgetProfile': 'actor_information_set_exact_v1',
            **({'horizon': {'unit': 'plies', 'value': 1}, 'boundaryEvaluator': boundary}
               if mode == 'budgeted' else {})} for kind, boundary in {
                   'bid': 'auction_hand_margin_v1', 'escalation': 'current_raw_point_margin_v1',
                   'project': 'current_raw_point_margin_v1', 'play': 'current_raw_point_margin_v1'}.items()}})


def project_target(root, target):
    catalog = ResearchFamilyCatalogV1.load(root / 'configs/research/public_project_rank_inference.yaml')
    return module.build_family_target_v1({**{k: v for k, v in target.items() if k != 'sha256'},
        'catalog': catalog.to_dict(), 'familyId': catalog.families[0].family_id,
        'tagQuery': {'technique': ['project-inference']}})


@pytest.mark.parametrize('mode,visibility,tail,accepted', [
    ('exact', 'actor_information_set', False, True),
    ('budgeted', 'actor_information_set', True, True),
    ('budgeted', 'actor_information_set', False, False),
    ('exact', 'complete_state', True, False),
])
def test_project_query_admission_uses_the_actual_focal_proof(root, target, mode, visibility, tail, accepted):
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.research.family_catalog import project_inference_generation_supported_v1
    target = project_target(root, target)
    raw = execution(root, target)
    raw['familyCatalog'] = raw.pop('familyTarget')['catalog']
    raw['exactTailProof'] = tail
    config = ScenarioQualityExecutionBudgetV1.build(raw)
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')
    quality = project_quality(mode=mode, visibility=visibility)
    kwargs = dict(rules=rules, execution=config, playing_quality=quality)
    request = generation_request(config, technique=['project-inference'])
    before = copy.deepcopy(request)
    assert project_inference_generation_supported_v1(rules, config, quality, play_index=24) is accepted
    if accepted:
        matches = module.matching_generation_families_v1(request,
            ResearchFamilyCatalogV1.build(config.family_catalog), scenario(target), **kwargs)
        assert [family.family_id for family in matches] == ['project-rank-inference']
        frozen = module.build_family_target_v1({**{k: v for k, v in target.items() if k != 'sha256'},
            'tagQuery': request['tagQuery']})
        module.validate_family_target_scenario_v1(frozen, scenario(frozen), **kwargs)
    else:
        with pytest.raises(ValueError, match='no live producer'):
            module.matching_generation_families_v1(request,
                ResearchFamilyCatalogV1.build(config.family_catalog), scenario(target), **kwargs)
    assert module.build_family_generation_request_v1(request) == before
    if mode == 'budgeted' and tail:
        assert not project_inference_generation_supported_v1(rules, config, quality, play_index=23)


def test_project_query_stream_keeps_native_qualification_separate(root, target, monkeypatch):
    from baloot_platform.core.rules import RuleBook
    target = project_target(root, target)
    raw = execution(root, target)
    raw['familyCatalog'] = raw.pop('familyTarget')['catalog']
    config = ScenarioQualityExecutionBudgetV1.build({**raw, 'exactTailProof': True})
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')
    quality = project_quality(mode='budgeted')
    calls = []
    def construct(**kwargs):
        calls.append(kwargs)
        return {'status': 'proposal', 'deckOrder': list(range(32))}
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', construct)
    monkeypatch.setattr(module, 'family_proposal_witness_v1', lambda *args, **kw: {'qualified': False})
    request = generation_request(config, technique=['project-inference'])
    rows = list(module.family_query_proposal_stream_v1(rules, config,
        dict(seed=0, maximumCampaignSeconds=10, maximumSeconds=5, maximumCandidates=1, maximumConflicts=100),
        generation=request, scenario=scenario(target), playing_quality=quality,
        campaign_started=module.time.perf_counter(), cancel_check=lambda: False, progress_callback=lambda row: None))
    assert rows == [(0, {'qualified': False})] and len(calls) == 1
    assert calls[0]['policy_prefix_steps'] == 24


def test_project_scope_and_family_compatibility_reach_backend_and_ui_metadata(root, target):
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.research.scenario_backend import OrToolsScenarioResearchBackendV2
    from baloot_platform.research.scenario_execution_capabilities import build_scenario_execution_capabilities_v1
    from baloot_platform.research.family_catalog import project_inference_generation_supported_v1
    from jsonschema import Draft202012Validator

    project = project_target(root, target)
    raw = query_execution(root, target).to_dict()
    raw.pop('sha256')
    raw['familyCatalog'].pop('sha256')
    raw['familyCatalog']['families'].extend(project['catalog']['families'])
    config = ScenarioQualityExecutionBudgetV1.build({**raw, 'exactTailProof': True})
    before = config.to_dict()
    profile = yaml.safe_load((root / 'configs/challenge/intelligent_research.yaml').read_text(encoding='utf-8'))
    profile['scenarioResearchV4']['qualityExecution'] = before
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')
    backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile, rules=rules,
        bundle_identity={'bundleRevisionId': 'CBR-PROJECT-QUERY', 'semanticBundleHash': 'a' * 64,
                         'effectiveRuntimeConfigHash': 'b' * 64})
    request = generation_request(config, technique=['project-inference'])
    quality = project_quality(mode='budgeted')
    admitted = backend._validate_generation_request(request, scenario(project), quality.to_dict())
    assert [family.family_id for family in admitted] == ['project-rank-inference']
    mixed = generation_request(config, technique=['project-inference', 'guard/protect'])
    assert {family.family_id for family in backend._validate_generation_request(
        mixed, scenario(project), quality)} == {'project-rank-inference', 'stopper_guard'}
    metadata = build_scenario_execution_capabilities_v1(backend)['authoringCatalog']['familySearch']
    assert Draft202012Validator(metadata['requestSchema']).is_valid(request)
    assert metadata['catalog'] == before['familyCatalog']
    assert metadata['liveTechniquesByFamily']['negative-project'] == []
    assert metadata['liveTechniquesByFamily']['project-rank-inference'] == ['project-inference']
    with pytest.raises(ValueError, match='no live producer'):
        backend._validate_generation_request(request, scenario(project), project_quality(visibility='complete_state'))
    assert not project_inference_generation_supported_v1(None, config)
    assert not project_inference_generation_supported_v1(RuleBook.load(
        root / 'configs/rules/saudi_project_optional_unratified_v1.yaml'), config)
    assert not project_inference_generation_supported_v1(rules, SimpleNamespace(video_interestingness={'schemaVersion': 3}))
    assert config.to_dict() == before


@pytest.fixture
def target(root):
    raw = ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml').to_dict()
    raw.pop('sha256')
    raw['families'][0]['enabled'] = True
    return module.build_family_target_v1(dict(schemaVersion=1, catalog=raw,
        familyId='negative-project', tagQuery={'mode': ['sun'], 'trickBand': ['trick-7']},
        bindings=dict(suit='spades', seat='north', rank='seven', projectType='siri',
                      contract='sun', playIndex=24), dealer='east', buyer='north', trump=None))


def scenario(target):
    family = ResearchFamilyCatalogV1.build(target['catalog']).families[0]
    return ResearchScenarioV1.build(dict(schemaVersion=1, origin={'kind': 'search'},
        constraints=family.instantiate(target['bindings']).to_dict(),
        decisionPoint={'kind': 'play', 'selector': {'remainingCards': 2, 'trickIndex': 6,
                                                  'position': 0, 'actingSeat': 'north'}},
        outcome={'predicateId': 'PRED-OUT', 'kind': 'auction.contract', 'version': 1,
                 'hardness': 'hard', 'arguments': {'contract': target['bindings']['contract']}}))


def execution(root, target):
    raw = yaml.safe_load((root / 'configs/challenge/intelligent_research.yaml').read_text(encoding='utf-8'))
    policy = json.loads((root / 'tests/fixtures/actor_interest_calibration.json').read_text())['cases']['7189']['policy']
    return {**raw['scenarioResearchV4']['qualityExecution'], 'candidateSource': 'runtime_proposal_stream_v1',
        'proposalSeedCount': 3, 'auditConstructedTrace': True,
        'earlyPlayPolicyId': 'observer_safe_trick_control_v1', 'exactPlayFromTrick': 7,
        'videoInterestingness': policy, 'familyTarget': target}


def query_execution(root, target):
    raw = execution(root, target)
    catalog = copy.deepcopy(raw.pop('familyTarget')['catalog'])
    catalog.pop('sha256')
    for index in (0, 2):
        catalog['families'][index].update(enabled=True, contracts=['sun'])
        catalog['families'][index]['capabilityTags']['mode'] = ['sun']
    raw['familyCatalog'] = catalog
    return ScenarioQualityExecutionBudgetV1.build(raw)


def generation_request(config, **query):
    return module.build_family_generation_request_v1(dict(schemaVersion=1,
        catalogHash=config.family_catalog['sha256'], tagQuery=query))


@pytest.mark.parametrize('fixture_mode,bundle_id', [
    ('query', 'baloot_operator_family_query'), ('project-query', 'baloot_operator_project_query')])
def test_private_operator_family_fixture_uses_ratified_rules_and_strict_native_gate(
        root, tmp_path, monkeypatch, fixture_mode, bundle_id):
    from baloot_platform.configuration import SQLiteConfigurationStore, VersionedConfigurationService
    from baloot_platform.config import profile_bundle_from_resolved
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.research.scenario_backend import ScenarioResearchAuthorityV1

    spec = importlib.util.spec_from_file_location('operator_e2e_fixture', root / 'tests/browser/serve_foundry_operator_e2e.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    monkeypatch.setenv('BALOOT_OPERATOR_E2E_FAMILY_FIXTURE', fixture_mode)
    fixture._seed(tmp_path)
    configuration = VersionedConfigurationService(SQLiteConfigurationStore(tmp_path / 'runtime/configuration.sqlite3'),
        root=root, profile=yaml.safe_load((root / 'configs/configuration/control-plane.yaml').read_text(encoding='utf-8')))
    try:
        revisions = configuration.store.list_bundles(bundle_id=bundle_id, status='published')
        assert len(revisions) == 1
        resolved = configuration.resolve(revision_id=revisions[0].revision_id)
        bundle = profile_bundle_from_resolved(resolved, root=root)
        assert bundle.rules.sha256 == RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml').sha256
        policy = ScenarioResearchAuthorityV1.from_challenge_profile(bundle.component('challenge')).quality_execution
        assert policy.video_interestingness['schemaVersion'] == 4
        assert policy.video_interestingness['baselinePolicyId'] == 'observer_safe_public_two_trick_control_v6'
        assert policy.video_interestingness['minimumConsistentWorlds'] == 2
        assert policy.video_interestingness['minimumBaselineRegret'] == 20
        assert policy.video_interestingness['allowTieRuleOnlyLessons'] is False
        assert policy.exact_tail_proof is True and policy.exact_play_from_trick == 7
        if fixture_mode == 'project-query':
            assert policy.family_catalog['id'] == 'public_project_rank_inference'
            assert policy.family_catalog['families'][0]['capabilityTags']['technique'] == ['project-inference']
            authority = ScenarioResearchAuthorityV1.from_challenge_profile(bundle.component('challenge'))
            assert authority.search_budgets['profiles']['actor_information_set_exact_v1']['maximumWallSeconds'] == 60
    finally:
        configuration.store.close()


def test_query_catalog_is_opt_in_and_legacy_profiles_keep_their_hash(root, target):
    config = query_execution(root, target)
    assert ScenarioQualityExecutionBudgetV1.build(config.to_dict()).to_dict() == config.to_dict()
    assert config.family_target is None
    old = execution(root, target)
    before = ScenarioQualityExecutionBudgetV1.build(old)
    assert 'familyCatalog' not in before.to_dict()
    assert before.sha256 == module.canonical_sha256({key: value for key, value in before.to_dict().items() if key != 'sha256'})
    with pytest.raises(ValueError, match='not familyTarget'):
        ScenarioQualityExecutionBudgetV1.build({**old, 'familyCatalog': config.family_catalog})
    with pytest.raises(ValueError, match='runtime_proposal_stream'):
        raw = config.to_dict()
        raw.pop('sha256')
        ScenarioQualityExecutionBudgetV1.build({**raw, 'candidateSource': 'runtime_proposal_v1'})
    raw = config.to_dict()
    raw.pop('sha256')
    raw['familyCatalog'].pop('sha256')
    raw['familyCatalog']['families'][0]['contracts'] = ['sun', 'hokum']
    raw['familyCatalog']['families'][0]['capabilityTags']['mode'] = ['sun', 'hokum']
    with pytest.raises(ValueError, match='Sun families only'):
        ScenarioQualityExecutionBudgetV1.build(raw)


def test_quality_requests_reuse_immutable_catalog_history_validation(root, target, monkeypatch):
    from baloot_platform.research import family_catalog

    raw = query_execution(root, target).to_dict()
    raw.pop('sha256')
    raw['familyCatalog'].pop('sha256')
    raw['familyCatalog']['id'] = 'quality_history_cache_regression'
    calls = []
    validate = family_catalog.history_family_absences_v1

    def counted(expression, **kwargs):
        calls.append(expression.sha256)
        return validate(expression, **kwargs)

    monkeypatch.setattr(family_catalog, 'history_family_absences_v1', counted)
    config = ScenarioQualityExecutionBudgetV1.build(raw)
    expected = sum(row['enabled'] for row in raw['familyCatalog']['families'])
    assert len(calls) == expected
    frozen = config.to_dict()
    policy = project_quality(mode='budgeted')
    for index in (24, 25):
        rebuilt = ScenarioQualityExecutionBudgetV1.build(copy.deepcopy(frozen))
        assert rebuilt.to_dict() == frozen
        policy.request_for('play', 'west', play_index=index, execution_budget=rebuilt)
    assert len(calls) == expected
    stale = copy.deepcopy(frozen)
    stale['familyCatalog']['families'][0]['name'] = 'untrusted edit'
    with pytest.raises(ValueError, match='checksum'):
        ScenarioQualityExecutionBudgetV1.build(stale)
    changed = copy.deepcopy(raw)
    changed['familyCatalog']['revision'] += 1
    assert ScenarioQualityExecutionBudgetV1.build(changed).sha256 != config.sha256
    assert len(calls) == 2 * expected
    assert config.to_dict() == frozen
    assert ResearchFamilyCatalogV1.require_sun_history_support.cache_info().maxsize == 16


def test_query_request_is_canonical_and_matches_only_supported_frozen_families(root, target):
    config = query_execution(root, target)
    request = generation_request(config, impact=['round-flip'], mode=['sun'])
    catalog = ResearchFamilyCatalogV1.build(config.family_catalog)
    families = module.matching_generation_families_v1(request, catalog, scenario(target))
    assert {row.family_id for row in families} == {'negative-project', 'stopper_guard'}
    assert module.build_family_generation_request_v1(request) == request
    for changed in ({**request, 'catalogHash': '0' * 64}, {**request, 'unknown': True},
                    {**request, 'schemaVersion': True}, {**request, 'tagQuery': {'mode': []}}):
        with pytest.raises(ValueError):
            module.matching_generation_families_v1(changed, catalog, scenario(target))
    for query in ({'level': ['advanced']}, {'mode': ['hokum']}, {'trickBand': ['trick-5']},
                  {'technique': ['trump-control']}):
        with pytest.raises(ValueError):
            module.matching_generation_families_v1(generation_request(config, **query), catalog, scenario(target))
    with pytest.raises(ValueError, match='play boundaries'):
        module.matching_generation_families_v1(request, catalog, scenario(target), play_indices=(16,))


@pytest.mark.parametrize('query', [
    {'impact': ['project-swing']}, {'impact': ['kaboot-swing']},
    {'technique': ['project-inference']}, {'technique': ['count/void-inference']},
    {'technique': ['endplay']}, {'technique': ['trump-control']},
    {'technique': ['baloot-timing']}, {'technique': ['ace-timing']},
])
def test_query_rejects_tags_without_live_producers_before_construction(root, target, monkeypatch, query):
    config = query_execution(root, target)
    request = generation_request(config, **query)
    before = copy.deepcopy(request)
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1',
                        lambda **kwargs: pytest.fail('unsupported query reached construction'))
    parameters = dict(seed=0, maximumCampaignSeconds=10, maximumSeconds=5,
                      maximumCandidates=1, maximumConflicts=100)
    with pytest.raises(ValueError, match='generation tag .* has no live producer'):
        list(module.family_query_proposal_stream_v1(object(), config, parameters, generation=request,
            scenario=scenario(target), campaign_started=module.time.perf_counter(), cancel_check=lambda: False,
            progress_callback=lambda row: None))
    assert module.build_family_generation_request_v1(request) == before


def test_query_matches_a_family_on_a_producible_or_alternative_only(root, target):
    config = query_execution(root, target)
    catalog = ResearchFamilyCatalogV1.build(config.family_catalog)
    request = generation_request(config, impact=['project-swing', 'round-flip'],
                                 technique=['project-inference', 'guard/protect'])
    before = copy.deepcopy(request)
    matches = module.matching_generation_families_v1(request, catalog, scenario(target))
    assert [row.family_id for row in matches] == ['stopper_guard']
    assert module.build_family_generation_request_v1(request) == before
    # A producible OR alternative cannot admit a family through a different,
    # unsupported alternative on that same axis.
    matches = module.matching_generation_families_v1(
        generation_request(config, impact=['project-swing', 'points-only']), catalog, scenario(target))
    assert [row.family_id for row in matches] == ['stopper_guard']


@pytest.mark.parametrize('query', [
    {'impact': ['project-swing']}, {'technique': ['project-inference']},
    {'technique': ['project-inference', 'guard/protect']},
])
def test_frozen_target_without_a_matching_live_producer_stays_readable_but_cannot_launch(root, target, monkeypatch, query):
    raw = {key: value for key, value in target.items() if key != 'sha256'}
    historical = module.build_family_target_v1({**raw, 'tagQuery': query})
    before = copy.deepcopy(historical)
    assert module.family_target_authoring_v1(historical)['tagQuery'] == query
    with pytest.raises(ValueError, match='no live producer'):
        module.validate_family_target_scenario_v1(historical, scenario(historical))
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1',
                        lambda **kwargs: pytest.fail('unsupported target reached construction'))
    config = ScenarioQualityExecutionBudgetV1.build(execution(root, historical))
    parameters = dict(seed=0, maximumCampaignSeconds=10, maximumSeconds=5,
                      maximumCandidates=1, maximumConflicts=100)
    with pytest.raises(ValueError, match='no live producer'):
        list(module.family_proposal_stream_v1(object(), config, parameters, campaign_started=0,
            cancel_check=lambda: False, progress_callback=lambda row: None))
    assert module.build_family_target_v1(historical) == before


def test_frozen_target_keeps_supported_or_queries_without_rehashing(target):
    raw = {key: value for key, value in target.items() if key != 'sha256'}
    mixed = module.build_family_target_v1({**raw, 'tagQuery': {'impact': ['project-swing', 'round-flip']}})
    before = copy.deepcopy(mixed)
    module.validate_family_target_scenario_v1(mixed, scenario(mixed))
    assert module.build_family_target_v1(mixed) == before


@pytest.mark.parametrize('section', ['constraints', 'outcome'])
def test_query_rejects_mandatory_hokum_before_construction(root, target, monkeypatch, section):
    config = query_execution(root, target)
    request = generation_request(config)
    raw = scenario(target).to_dict()
    raw.pop('sha256')
    raw[section] = {'op': 'all', 'children': [dict(predicateId='PRED-HOKUM',
        kind='auction.contract', version=1, hardness='hard', arguments={'contract': 'hokum'})]}
    frozen = ResearchScenarioV1.build(raw)
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1',
                        lambda **kwargs: pytest.fail('unsupported contract reached construction'))
    parameters = dict(seed=0, maximumCampaignSeconds=10, maximumSeconds=5,
                      maximumCandidates=1, maximumConflicts=100)
    with pytest.raises(ValueError, match='Sun structural templates only'):
        list(module.family_query_proposal_stream_v1(object(), config, parameters, generation=request,
            scenario=frozen, campaign_started=module.time.perf_counter(), cancel_check=lambda: False,
            progress_callback=lambda row: None))


@pytest.mark.parametrize('hardness', ['prefer', 'report'])
def test_query_does_not_reject_soft_hokum(root, target, hardness):
    config = query_execution(root, target)
    raw = scenario(target).to_dict()
    raw.pop('sha256')
    raw['constraints'] = dict(predicateId='PRED-SOFT-HOKUM', kind='auction.contract',
        version=1, hardness=hardness, arguments={'contract': 'hokum'})
    frozen = ResearchScenarioV1.build(raw)
    before = frozen.to_dict()
    assert module.matching_generation_families_v1(generation_request(config),
        ResearchFamilyCatalogV1.build(config.family_catalog), frozen)
    assert frozen.to_dict() == before


@pytest.mark.parametrize('op', ['any', 'not', 'count'])
def test_query_does_not_flatten_optional_contracts(root, target, op):
    config = query_execution(root, target)
    hokum = dict(predicateId='PRED-HOKUM', kind='auction.contract', version=1,
        hardness='hard', arguments={'contract': 'hokum'})
    sun = {**hokum, 'predicateId': 'PRED-SUN', 'arguments': {'contract': 'sun'}}
    expression = {'op': op, 'children': [hokum] if op == 'not' else [hokum, sun]}
    if op == 'count':
        expression.update(minimum=1, maximum=1)
    raw = scenario(target).to_dict()
    raw.pop('sha256')
    raw['constraints'] = expression
    frozen = ResearchScenarioV1.build(raw)
    before = frozen.to_dict()
    assert module.matching_generation_families_v1(generation_request(config),
        ResearchFamilyCatalogV1.build(config.family_catalog), frozen)
    assert frozen.to_dict() == before


def test_saved_auction_hints_bind_order_without_changing_legacy_hash(root, target):
    config = query_execution(root, target)
    legacy = generation_request(config, mode=['sun'])
    assert 'auctionSeeds' not in legacy
    raw = {key: value for key, value in legacy.items() if key != 'sha256'}
    hints = [{'seed': 6, 'dealer': 'south'}, {'seed': 42, 'dealer': 'west'}]
    request = module.build_family_generation_request_v1({**raw, 'auctionSeeds': hints})
    hints[0]['seed'] = 7
    assert request['auctionSeeds'][0]['seed'] == 6
    assert request['sha256'] != legacy['sha256']
    assert module.build_family_generation_request_v1(request) == request
    reverse = module.build_family_generation_request_v1({**raw, 'auctionSeeds': request['auctionSeeds'][::-1]})
    assert request['sha256'] != reverse['sha256']
    with pytest.raises(ValueError, match='checksum'):
        module.build_family_generation_request_v1({**request, 'auctionSeeds': reverse['auctionSeeds']})


@pytest.mark.parametrize('hints', [[], None, [{'seed': True, 'dealer': 'north'}],
    [{'seed': 1.0, 'dealer': 'north'}], [{'seed': -1, 'dealer': 'north'}],
    [{'seed': '1', 'dealer': 'north'}], [{'seed': 1, 'dealer': 'NORTH'}],
    [{'seed': 1, 'dealer': 'north', 'qualified': True}],
    [{'seed': 1, 'dealer': 'north'}, {'seed': 1, 'dealer': 'north'}]])
def test_saved_auction_hints_are_closed_strict_and_unique(root, target, hints):
    config = query_execution(root, target)
    with pytest.raises(ValueError):
        module.build_family_generation_request_v1(dict(schemaVersion=1,
            catalogHash=config.family_catalog['sha256'], tagQuery={}, auctionSeeds=hints))


@pytest.mark.real_clock
def test_saved_auction_hints_are_reverified_first_and_share_the_attempt_budget(root, target, monkeypatch):
    config = query_execution(root, target)
    request = module.build_family_generation_request_v1(dict(schemaVersion=1,
        catalogHash=config.family_catalog['sha256'], tagQuery={'mode': ['sun']},
        auctionSeeds=[{'seed': 10, 'dealer': 'south'}, {'seed': 42, 'dealer': 'west'}]))
    calls, progress = [], []
    provider = object()
    monkeypatch.setattr(module.time, 'perf_counter', lambda: 100)
    def construct(**kwargs):
        assert kwargs['auction_provider'] is provider
        assert kwargs['buyer'] is None
        assert kwargs['_select_query_suit'] is True
        calls.append((kwargs['seed'], kwargs['dealer']))
        return {'status': 'seed_miss', 'reason': 'opening_mismatch'}
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', construct)
    params = dict(maximumCampaignSeconds=10, maximumSeconds=5, maximumCandidates=1,
                  maximumConflicts=100, seed=10)
    kwargs = dict(generation=request, scenario=scenario(target), campaign_started=100,
        cancel_check=lambda: False, progress_callback=progress.append, auction_provider=provider)
    assert list(module.family_query_proposal_stream_v1(object(), config, params, **kwargs)) == []
    assert calls == [(10, 'south'), (42, 'west'), (11, 'east')]
    assert [(row['screenedSeed'], row['screenedDealer']) for row in progress] == calls
    assert all(row['generationRequestHash'] == request['sha256'] for row in progress)
    assert list(module.family_query_proposal_stream_v1(object(), config, params,
        **{**kwargs, 'cancel_check': lambda: True})) == []
    assert len(calls) == 3


@pytest.mark.parametrize('reason', ['negative_project_mismatch', 'observer_suit_count_mismatch',
                                  'history_suit_count_mismatch', 'history_expression_infeasible',
                                  'fixed_deal_projects_infeasible'])
def test_query_reuses_a_qualified_opening_after_a_family_specific_miss(root, target, monkeypatch, reason):
    config = query_execution(root, target)
    request = module.build_family_generation_request_v1(dict(schemaVersion=1,
        catalogHash=config.family_catalog['sha256'], tagQuery={'mode': ['sun']},
        auctionSeeds=[{'seed': 42, 'dealer': 'west'}]))
    calls, progress = [], []
    provider = object()
    monkeypatch.setattr(module.time, 'perf_counter', lambda: 100)
    def construct(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {'status': 'seed_miss', 'reason': reason}
        return {'status': 'proposal', 'deckOrder': list(range(32))}
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', construct)
    monkeypatch.setattr(module, 'family_proposal_witness_v1', lambda proposal, **kw: {'nativeWitness': True})
    params = dict(seed=10, maximumCampaignSeconds=10, maximumSeconds=5,
                  maximumCandidates=1, maximumConflicts=100)
    rows = list(module.family_query_proposal_stream_v1(object(), config, params,
        generation=request, scenario=scenario(target), campaign_started=100,
        cancel_check=lambda: False, progress_callback=progress.append, auction_provider=provider))
    assert rows == [(10, {'nativeWitness': True})]
    assert [(call['seed'], call['dealer']) for call in calls] == [(42, 'west'), (42, 'west')]
    assert len({call['family'].family_id for call in calls}) == 2
    assert all(call['auction_provider'] is provider and call['buyer'] is None for call in calls)
    assert all(call['bindings']['seat'] == 'north' and call['bindings']['playIndex'] == 24 for call in calls)
    assert all(row['generationRequestHash'] == request['sha256'] for row in progress)
    assert [row['constructionStatus'] for row in progress] == ['seed_miss', 'proposal']


@pytest.mark.real_clock
@pytest.mark.parametrize('continuation', ['new', 'duplicate', 'incomplete', 'cancelled'])
def test_qualified_opening_enumerates_prefixes_with_one_construction_allowance(root, target, monkeypatch, continuation):
    from baloot_platform.core.cards import Card

    config = query_execution(root, target)
    request = module.build_family_generation_request_v1(dict(schemaVersion=1,
        catalogHash=config.family_catalog['sha256'], tagQuery={'technique': ['guard/protect']},
        auctionSeeds=[{'seed': 42, 'dealer': 'west'}]))
    clock, cancelled, calls = [100.0], [False], []
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    first = [{'actor': 'north', 'card': Card(i).code, 'ekka': False} for i in range(24)]
    second = first[:-2] + first[-2:][::-1]
    def construct(**kwargs):
        calls.append(kwargs)
        clock[0] += 2
        if len(calls) == 2 and continuation == 'incomplete':
            cancelled[0] = True
            return {'status': 'incomplete', 'reason': 'cancelled_or_budget'}
        prefix = first if len(calls) == 1 or continuation == 'duplicate' else second
        return {'status': 'proposal', 'deckOrder': list(range(32)),
                'decisionPrefix': {'playActions': prefix}}
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', construct)
    monkeypatch.setattr(module, 'family_proposal_witness_v1',
        lambda proposal, **kw: {'prefix': proposal['decisionPrefix'], 'qualified': False})
    params = dict(seed=10, maximumCampaignSeconds=20, maximumSeconds=5,
                  maximumCandidates=2, maximumConflicts=100)
    rows = module.family_query_proposal_stream_v1(object(), config, params,
        generation=request, scenario=scenario(target), campaign_started=100,
        cancel_check=lambda: cancelled[0], progress_callback=lambda row: None, auction_provider=object())
    assert next(rows) == (10, {'prefix': {'playActions': first}, 'qualified': False})
    clock[0] += 3  # Qualification time uses campaign time, not construction allowance twice.
    if continuation == 'cancelled':
        cancelled[0] = True
    if continuation == 'duplicate':
        with pytest.raises(ValueError, match='repeated.*decision prefix'):
            next(rows)
    elif continuation == 'new':
        assert list(rows) == [(11, {'prefix': {'playActions': second}, 'qualified': False})]
    else:
        assert list(rows) == []
    assert len(calls) == (1 if continuation == 'cancelled' else 2)
    assert calls[0]['maximum_seconds'] == 5
    if len(calls) == 2:
        assert calls[1]['maximum_seconds'] == 3
        assert calls[1]['seed'] == calls[0]['seed'] == 42
        assert calls[1]['dealer'] == calls[0]['dealer'] == 'west'
        assert not calls[1]['excluded_deck_orders']
        assert [[action.to_dict() for action in prefix] for prefix in calls[1]['excluded_play_prefixes']] == [first]


@pytest.mark.real_clock
@pytest.mark.parametrize('status,reason', [
    ('seed_miss', 'opening_mismatch'), ('seed_miss', 'unknown_miss'),
    ('incomplete', 'cancelled_or_budget'), ('infeasible', 'exact_infeasibility'),
])
def test_query_does_not_retry_other_families_on_shared_or_incomplete_misses(root, target, monkeypatch, status, reason):
    config = query_execution(root, target)
    calls = []
    monkeypatch.setattr(module.time, 'perf_counter', lambda: 100)
    def construct(**kwargs):
        calls.append((kwargs['seed'], kwargs['family'].family_id))
        return {'status': status, 'reason': reason}
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', construct)
    params = dict(seed=10, maximumCampaignSeconds=10, maximumSeconds=5,
                  maximumCandidates=1, maximumConflicts=100)
    assert list(module.family_query_proposal_stream_v1(object(), config, params,
        generation=generation_request(config, mode=['sun']), scenario=scenario(target),
        campaign_started=100, cancel_check=lambda: False, progress_callback=lambda row: None)) == []
    assert [seed for seed, _ in calls] == [10, 11, 12]


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('status,reason', [
    ('seed_miss', 'decision_actor_mismatch'), ('seed_miss', 'opening_mismatch'),
    ('seed_miss', 'native_allocation_infeasible'), ('incomplete', 'cancelled_or_budget'),
])
def test_query_native_opening_miss_does_not_skip_independent_family(
        root, target, monkeypatch, native, status, reason):
    raw = query_execution(root, target).to_dict()
    raw.pop('sha256')
    raw['proposalSeedCount'] = 1
    if native:
        raw['familyOpeningMode'] = 'native_project_auction_v1'
    config = ScenarioQualityExecutionBudgetV1.build(raw)
    calls, progress = [], []
    authored = scenario(target)
    before = authored.to_dict()
    monkeypatch.setattr(module.time, 'perf_counter', lambda: 100)

    def produce(rules, execution, parameters, **kwargs):
        calls.append((parameters, kwargs))
        first = len(calls) == 1
        kwargs['progress_callback']({'screenedSeed': parameters['seed'],
            'constructionStatus': status if first else 'proposal',
            'constructionReason': reason if first else None,
            'proposalCount': int(not first)})
        if not first:
            yield parameters['seed'], {'qualified': False}

    monkeypatch.setattr(module, '_history_family_proposals_v1', produce)
    rows = list(module.family_query_proposal_stream_v1(object(), config,
        dict(seed=10, maximumCampaignSeconds=10, maximumSeconds=5,
             maximumCandidates=1, maximumConflicts=100),
        generation=generation_request(config, mode=['sun']), scenario=authored,
        campaign_started=100, cancel_check=lambda: False, progress_callback=progress.append,
        auction_provider=object()))
    fallback = native and status == 'seed_miss'
    assert rows == ([(10, {'qualified': False})] if fallback else [])
    assert len(calls) == (2 if fallback else 1)
    assert len({kwargs['family'].family_id for _, kwargs in calls}) == len(calls)
    assert all(parameters['seed'] == 10 and kwargs['dealer'] == 'south'
               and kwargs['bindings']['seat'] == 'north' and kwargs['bindings']['playIndex'] == 24
               and kwargs['scenario'] is authored for parameters, kwargs in calls)
    assert authored.to_dict() == before and progress[0]['constructionReason'] == reason


@pytest.mark.real_clock
@pytest.mark.parametrize('stop', ['deadline', 'cancelled', 'all-families-missed'])
def test_query_family_fallback_is_bounded_by_opening_and_campaign(root, target, monkeypatch, stop):
    config = query_execution(root, target)
    now, cancelled, calls = [100], [False], []
    monkeypatch.setattr(module.time, 'perf_counter', lambda: now[0])
    def construct(**kwargs):
        calls.append((kwargs['seed'], kwargs['family'].family_id))
        assert kwargs['maximum_seconds'] <= 5
        if stop == 'deadline':
            now[0] = 110
        elif stop == 'cancelled':
            cancelled[0] = True
        return {'status': 'seed_miss', 'reason': 'history_expression_infeasible'}
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', construct)
    params = dict(seed=10, maximumCampaignSeconds=10, maximumSeconds=5,
                  maximumCandidates=1, maximumConflicts=100)
    assert list(module.family_query_proposal_stream_v1(object(), config, params,
        generation=generation_request(config, mode=['sun']), scenario=scenario(target),
        campaign_started=100, cancel_check=lambda: cancelled[0], progress_callback=lambda row: None)) == []
    if stop == 'all-families-missed':
        assert len(calls) == len(set(calls)) == 6
        assert [seed for seed, _ in calls] == [10, 10, 11, 11, 12, 12]
    else:
        assert len(calls) == 1


def test_saved_auction_hint_limit_is_frozen_before_execution(root, target, monkeypatch):
    config = query_execution(root, target)
    raw = dict(schemaVersion=1, catalogHash=config.family_catalog['sha256'], tagQuery={},
               auctionSeeds=[{'seed': seed, 'dealer': 'north'} for seed in range(4)])
    request = module.build_family_generation_request_v1(raw)
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1',
                        lambda **kwargs: pytest.fail('oversized hints reached construction'))
    with pytest.raises(ValueError, match='too long'):
        module.build_family_generation_request_v1(request, maximum_auction_seeds=config.proposal_seed_count)
    with pytest.raises(ValueError, match='too long'):
        list(module.family_query_proposal_stream_v1(object(), config, {}, generation=request,
            scenario=scenario(target), campaign_started=0, cancel_check=lambda: False,
            progress_callback=lambda row: None))


@pytest.mark.parametrize('play_indices', [None, [17, 24]])
def test_query_catalog_reaches_authoring_and_native_validation(root, target, play_indices):
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.research.scenario_backend import OrToolsScenarioResearchBackendV2
    from baloot_platform.research.scenario_execution_capabilities import build_scenario_execution_capabilities_v1
    config = query_execution(root, target)
    if play_indices is not None:
        config = ScenarioQualityExecutionBudgetV1.build({
            **{key: value for key, value in config.to_dict().items() if key != 'sha256'},
            'proposalPlayIndices': play_indices})
    before = config.to_dict()
    profile = yaml.safe_load((root / 'configs/challenge/intelligent_research.yaml').read_text(encoding='utf-8'))
    profile['scenarioResearchV4']['qualityExecution'] = config.to_dict()
    backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile,
        bundle_identity={'bundleRevisionId': 'CBR-FAMILY-QUERY', 'semanticBundleHash': 'a' * 64,
                         'effectiveRuntimeConfigHash': 'b' * 64},
        rules=RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml'))
    request = generation_request(config, mode=['sun'])
    assert len(backend._validate_generation_request(request, scenario(target), object())) == 2
    with pytest.raises(ValueError, match='playingQuality'):
        backend._validate_generation_request(request, scenario(target), None)
    authoring = build_scenario_execution_capabilities_v1(backend)['authoringCatalog']
    assert authoring['familySearch']['catalog'] == config.family_catalog
    assert authoring['familySearch']['playIndices'] == (list(range(32)) if play_indices is None else play_indices)
    assert backend.authority.quality_execution.to_dict() == before
    assert authoring['familySearch']['requestSchema']['properties']['auctionSeeds']['maxItems'] == config.proposal_seed_count
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(authoring['familySearch']['requestSchema'])
    assert validator.is_valid(request)
    unsupported = generation_request(config, impact=['project-swing'])
    assert not validator.is_valid(unsupported)
    with pytest.raises(ValueError, match='no live producer'):
        backend._validate_generation_request(unsupported, scenario(target), object())
    assert 'level' not in authoring['familySearch']['supportedAxes']
    assert 'familyGeneration' not in authoring
    oversized = dict(schemaVersion=1, catalogHash=config.family_catalog['sha256'], tagQuery={},
        auctionSeeds=[{'seed': seed, 'dealer': 'north'} for seed in range(config.proposal_seed_count + 1)])
    with pytest.raises(ValueError, match='too long'):
        backend._validate_generation_request(oversized, scenario(target), object())
    backend._rules = RuleBook.load(root / 'configs/rules/saudi_project_optional_unratified_v1.yaml')
    with pytest.raises(ValueError, match='mandatory project declarations'):
        backend._validate_generation_request(request, scenario(target), object())
    assert 'familySearch' not in build_scenario_execution_capabilities_v1(backend)['authoringCatalog']


@pytest.mark.parametrize('query,available', [({'mode': ['sun']}, True),
    ({'impact': ['project-swing']}, False), ({'technique': ['project-inference', 'guard/protect']}, False)])
def test_target_apply_control_requires_a_matching_live_producer(root, target, query, available):
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.research.scenario_backend import OrToolsScenarioResearchBackendV2
    from baloot_platform.research.scenario_execution_capabilities import build_scenario_execution_capabilities_v1

    historical = module.build_family_target_v1({**{key: value for key, value in target.items() if key != 'sha256'},
                                                'tagQuery': query})
    profile = yaml.safe_load((root / 'configs/challenge/intelligent_research.yaml').read_text(encoding='utf-8'))
    profile['scenarioResearchV4']['qualityExecution'] = execution(root, historical)
    backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile,
        bundle_identity={'bundleRevisionId': 'CBR-FAMILY-TARGET', 'semanticBundleHash': 'a' * 64,
                         'effectiveRuntimeConfigHash': 'b' * 64},
        rules=RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml'))
    catalog = build_scenario_execution_capabilities_v1(backend)['authoringCatalog']
    assert ('familyGeneration' in catalog) is available
    assert backend.authority.quality_execution.family_target == historical


@pytest.mark.real_clock
@pytest.mark.parametrize('qualified', [False, True])
def test_query_stream_rotates_families_and_does_not_spend_candidate_limit_on_seed_misses(root, target, monkeypatch, qualified):
    config = query_execution(root, target)
    raw = config.to_dict()
    raw.pop('sha256')
    raw['proposalSeedCount'] = 8
    config = ScenarioQualityExecutionBudgetV1.build(raw)
    calls, progress = [], []
    monkeypatch.setattr(module.time, 'perf_counter', lambda: 100)
    def construct(**kwargs):
        calls.append(kwargs)
        return ({'status': 'seed_miss'} if len(calls) <= 2 else
                {'status': 'proposal', 'deckOrder': list(range(32))})
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', construct)
    monkeypatch.setattr(module, 'family_proposal_witness_v1', lambda proposal, **kw: {'nativeWitness': True})
    params = dict(maximumCampaignSeconds=10, maximumSeconds=5, maximumCandidates=2,
                  maximumConflicts=100, seed=10)
    kwargs = dict(generation=generation_request(config, impact=['round-flip'], mode=['sun']),
        scenario=scenario(target), campaign_started=100, cancel_check=lambda: False,
        progress_callback=progress.append, **({'auction_provider': object()} if qualified else {}))
    rows = module.family_query_proposal_stream_v1(object(), config, params, **kwargs)
    assert not calls
    assert list(rows) == [(10, {'nativeWitness': True}), (11, {'nativeWitness': True})]
    assert len(calls) == 4
    assert {row['familyId'] for row in progress} == {'negative-project', 'stopper_guard'}
    assert [row['proposalOrdinal'] for row in progress if row['proposalCount']] == [10, 11]
    assert all(row['generationRequestHash'] == kwargs['generation']['sha256'] for row in progress)
    assert all(call['bindings']['seat'] == 'north' and call['bindings']['playIndex'] == 24 for call in calls)
    assert all(call['buyer'] is None for call in calls)
    assert all(call['_select_query_suit'] is True for call in calls)
    assert all(call['scenario'] is kwargs['scenario'] for call in calls)
    assert all('familyTargetHash' not in row for row in progress)
    assert list(module.family_query_proposal_stream_v1(object(), config, params,
        **{**kwargs, 'cancel_check': lambda: True})) == []
    assert len(calls) == 4


def test_query_allocation_ignores_disabled_families_before_assigning_floors(root, target):
    config = query_execution(root, target)
    raw = copy.deepcopy(config.family_catalog)
    raw.pop('sha256')
    for index, weight in ((0, 9), (2, 1), (1, 990)):
        raw['families'][index]['weight'] = weight
    catalog = ResearchFamilyCatalogV1.build(raw)
    request = module.build_family_generation_request_v1(dict(schemaVersion=1,
        catalogHash=catalog.sha256, tagQuery={'impact': ['round-flip']}))
    admitted = module.matching_generation_families_v1(request, catalog, scenario(target))
    shares = catalog.allocation_shares(request['tagQuery'], eligible_family_ids={row.family_id for row in admitted})
    assert shares == pytest.approx({'negative-project': .86, 'stopper_guard': .14})


def test_frozen_family_target_roundtrips_without_mutating_old_authority(root, target):
    raw = execution(root, target)
    original = copy.deepcopy(raw)
    built = ScenarioQualityExecutionBudgetV1.build(raw)
    assert raw == original
    assert ScenarioQualityExecutionBudgetV1.build(built.to_dict()).sha256 == built.sha256
    old = {key: value for key, value in raw.items() if key != 'familyTarget'}
    assert ScenarioQualityExecutionBudgetV1.build(old).sha256 != built.sha256
    assert 'familyTarget' not in ScenarioQualityExecutionBudgetV1.build(old).to_dict()
    assert module.build_family_target_v1(target) == target
    module.validate_family_target_scenario_v1(target, scenario(target))


def test_native_family_opening_mode_is_optional_frozen_and_closed(root, target):
    from jsonschema import Draft202012Validator
    from baloot_platform.research.result_authority import validate_trace_quality_evidence_v1
    raw = execution(root, target)
    legacy = ScenarioQualityExecutionBudgetV1.build(raw)
    native = ScenarioQualityExecutionBudgetV1.build({**raw, 'familyOpeningMode': 'native_project_auction_v1'})
    assert legacy.to_dict() == ScenarioQualityExecutionBudgetV1.build(raw).to_dict()
    assert 'familyOpeningMode' not in legacy.to_dict() and native.sha256 != legacy.sha256
    assert ScenarioQualityExecutionBudgetV1.build(native.to_dict()).to_dict() == native.to_dict()
    profile = yaml.safe_load((root / 'configs/challenge/intelligent_research.yaml').read_text(encoding='utf-8'))
    profile['scenarioResearchV4']['qualityExecution'] = {
        key: value for key, value in native.to_dict().items() if key != 'sha256'}
    validator = Draft202012Validator(json.loads((root / 'schemas/profiles/challenge.schema.json').read_text()))
    errors = list(validator.iter_errors(profile))
    assert not errors, [(list(error.path), error.message[:200],
                         [(list(item.path), item.message[:200]) for item in error.context[:12]])
                        for error in errors]
    assert (root / 'schemas/profiles/challenge.schema.json').read_bytes() == (
        root / 'src/baloot_platform/resources/schemas/profiles/challenge.schema.json').read_bytes()
    with pytest.raises(ValueError, match='requires allocation and auction provenance'):
        validate_trace_quality_evidence_v1({}, rules_hash='a' * 64,
            required_execution=native, require_family_opening=True)
    validate_trace_quality_evidence_v1({}, rules_hash='a' * 64, required_execution=native)
    for change in ({'familyOpeningMode': None}, {'familyOpeningMode': True},
                   {'familyOpeningMode': 'rng'}, {'familyTarget': None}):
        with pytest.raises((TypeError, ValueError)):
            ScenarioQualityExecutionBudgetV1.build({**raw, 'familyOpeningMode': 'native_project_auction_v1', **change})


@pytest.mark.real_clock
@pytest.mark.parametrize('allocation_status', ['constructed', 'seed_miss', 'incomplete'])
def test_native_stream_allocates_once_preserves_solved_buyer_and_logs_failures(root, target, monkeypatch, allocation_status):
    raw = execution(root, target)
    config = ScenarioQualityExecutionBudgetV1.build({**raw, 'familyOpeningMode': 'native_project_auction_v1'})
    allocations, histories, progress = [], [], []
    initial = {'deckOrder': list(range(32))}
    allocation = {'status': allocation_status, 'initialPrefix': initial,
        'allocation': {'modelHash': 'm', 'auction': {'completion': {'buyer': 'south'}}},
        **({'reason': 'native_allocation_' + allocation_status} if allocation_status != 'constructed' else {})}
    def allocate(**kw):
        allocations.append(kw)
        return allocation
    def construct(**kw):
        histories.append(kw)
        return {'status': 'proposal', 'decisionPrefix': {'playActions': []},
                'nativeOpeningAuction': {'sha256': 'receipt'}}
    monkeypatch.setattr(module, 'construct_family_initial_prefix_v1', allocate)
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', construct)
    monkeypatch.setattr(module, 'family_proposal_witness_v1',
        lambda proposal, **kw: {'opening': proposal['familyOpening']})
    family = ResearchFamilyCatalogV1.build(target['catalog']).families[0]
    rows = list(module._history_family_proposals_v1(object(), config,
        dict(seed=1, maximumCampaignSeconds=10, maximumSeconds=5, maximumCandidates=1, maximumConflicts=100),
        family=family, bindings=target['bindings'], dealer=target['dealer'], buyer=None, trump=None,
        campaign_started=module.time.perf_counter(), cancel_check=lambda: False,
        progress_callback=progress.append, auction_provider=object(), source_seeds=[1]))
    assert len(allocations) == 1 and allocations[0]['buyer'] is None
    assert progress[0]['currentStage'] == 'family_allocation'
    if allocation_status == 'constructed':
        assert len(rows) == len(histories) == 1
        assert histories[0]['buyer'] == 'south' and histories[0]['initial_prefix'] == initial
        assert 0 < histories[0]['maximum_seconds'] <= allocations[0]['maximum_seconds']
        assert rows[0][1]['opening']['auctionQualification']['sha256'] == 'receipt'
    else:
        assert not rows and not histories
        assert progress[0]['constructionReason'] == allocation['reason']


@pytest.mark.real_clock
def test_native_stream_shares_budget_reuses_context_and_excludes_prior_allocations(root, target, monkeypatch):
    config = ScenarioQualityExecutionBudgetV1.build({**execution(root, target),
        'familyOpeningMode': 'native_project_auction_v1'})
    now, allocations, histories, counts = [100.0], [], [], {}
    monkeypatch.setattr(module.time, 'perf_counter', lambda: now[0])
    def allocate(**kw):
        allocations.append(kw)
        now[0] += 2
        return {'status': 'constructed', 'initialPrefix': {'deckOrder': [kw['seed']]},
                'allocation': {'modelHash': 'm', 'auction': {'completion': {'buyer': 'south'}}}}
    def construct(**kw):
        histories.append(kw)
        now[0] += 1
        count = counts[kw['seed']] = counts.get(kw['seed'], 0) + 1
        if count == 3:
            return {'status': 'incomplete', 'reason': 'tail_budget'}
        return {'status': 'proposal', 'nativeOpeningAuction': {},
            'decisionPrefix': {'playActions': [{'actor': 'north', 'card': 'S7' if count == 1 else 'S8'}]}}
    monkeypatch.setattr(module, 'construct_family_initial_prefix_v1', allocate)
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', construct)
    monkeypatch.setattr(module, 'family_proposal_witness_v1', lambda *a, **kw: {})
    rows = list(module.family_proposal_stream_v1(object(), config,
        dict(seed=1, maximumCampaignSeconds=100, maximumSeconds=5, maximumCandidates=10, maximumConflicts=100),
        campaign_started=100, cancel_check=lambda: False, progress_callback=lambda row: None,
        auction_provider=object(), source_seeds=[1, 2]))
    assert len(rows) == 4 and len(allocations) == 2 and len(histories) == 6
    assert [row['excluded_deck_orders'] for row in allocations] == [(), ((1,),)]
    assert [row['maximum_seconds'] for row in histories] == [3, 2, 1, 3, 2, 1]
    assert all(not row['excluded_deck_orders'] for row in histories)


def test_repeated_family_target_validation_is_bounded_detached_and_strict(target, monkeypatch):
    module._build_family_target_cached_v1.cache_clear()
    original = module.ResearchFamilyCatalogV1.build
    calls = []
    monkeypatch.setattr(module.ResearchFamilyCatalogV1, 'build',
        lambda value: calls.append(value) or original(value))
    first = module.build_family_target_v1(target)
    first['bindings']['seat'] = 'west'
    assert module.build_family_target_v1(target) == target
    assert len(calls) == 1
    changed = copy.deepcopy(target)
    changed['bindings']['seat'] = 'west'
    with pytest.raises(ValueError, match='checksum'):
        module.build_family_target_v1(changed)
    changed = copy.deepcopy(target)
    changed['catalog']['families'] = tuple(changed['catalog']['families'])
    with pytest.raises(ValueError, match='coercion'):
        module.build_family_target_v1(changed)
    assert module._build_family_target_cached_v1.cache_info().maxsize == 16


def test_hokum_target_does_not_advertise_sun_project_inference(root, target):
    from baloot_platform.core.rules import RuleBook

    raw = {key: value for key, value in target.items() if key != 'sha256'}
    raw['catalog'].pop('sha256')
    family = raw['catalog']['families'][0]
    family['contracts'] = family['capabilityTags']['mode'] = ['hokum']
    frozen = module.build_family_target_v1({**raw, 'tagQuery': {
        'mode': ['hokum'], 'technique': ['project-inference']},
        'bindings': {**raw['bindings'], 'contract': 'hokum'}, 'trump': 'spades'})
    with pytest.raises(ValueError, match='no live producer'):
        module.validate_family_target_scenario_v1(frozen, scenario(frozen),
            rules=RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml'),
            execution=ScenarioQualityExecutionBudgetV1.build(execution(root, frozen)), playing_quality=project_quality())


@pytest.mark.parametrize('contract', ['sun', 'hokum'])
def test_visual_family_target_reuses_frozen_template_and_decision(root, target, contract):
    if contract == 'hokum':
        raw = {key: value for key, value in target.items() if key != 'sha256'}
        guard = next(row for row in raw['catalog']['families'] if row['id'] == 'stopper_guard')
        guard['enabled'] = True
        raw['catalog'].pop('sha256')
        raw['catalog']['families'] = [guard]
        target = module.build_family_target_v1({**raw, 'familyId': guard['id'],
            'tagQuery': {'mode': ['hokum']}, 'bindings': {**raw['bindings'], 'contract': 'hokum'}, 'trump': 'spades'})
    original = copy.deepcopy(target)
    authored = module.family_target_authoring_v1(target)
    expected = scenario(target)
    assert target == original
    assert authored['targetHash'] == target['sha256']
    assert authored['decisionPoint'] == expected.decision_point
    assert authored['constraints'] == expected.constraints.to_dict()
    module.validate_family_target_scenario_v1(target, expected)
    assert authored['sha256'] == module.canonical_sha256({key: value for key, value in authored.items() if key != 'sha256'})
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.research.scenario_backend import OrToolsScenarioResearchBackendV2
    from baloot_platform.research.scenario_execution_capabilities import build_scenario_execution_capabilities_v1
    profile = yaml.safe_load((root / 'configs/challenge/intelligent_research.yaml').read_text(encoding='utf-8'))
    profile['scenarioResearchV4']['qualityExecution'] = execution(root, target)
    backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile,
        bundle_identity={'bundleRevisionId': 'CBR-FAMILY-UI', 'semanticBundleHash': 'a' * 64,
                         'effectiveRuntimeConfigHash': 'b' * 64},
        rules=RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml'))
    catalog = build_scenario_execution_capabilities_v1(backend)['authoringCatalog']
    assert catalog['familyGeneration'] == authored
    assert catalog['sha256'] == module.canonical_sha256({key: value for key, value in catalog.items() if key != 'sha256'})


@pytest.mark.parametrize('mutation', [
    lambda raw: raw.update(candidateSource='runtime_proposal_v1'),
    lambda raw: raw.update(proposalPlayIndices=[23]),
    lambda raw: raw.update(acceptedLessonLibrary={}),
    lambda raw: raw['familyTarget'].update(unknown=True),
    lambda raw: raw['familyTarget'].update(sha256='0' * 64),
    lambda raw: raw['familyTarget'].update(tagQuery={'mode': ['hokum']}),
    lambda raw: raw['familyTarget'].update(familyId='positive-project'),
    lambda raw: raw['familyTarget'].update(trump='hearts'),
])
def test_family_configuration_fails_closed(root, target, mutation):
    raw = execution(root, target)
    mutation(raw)
    with pytest.raises(ValueError):
        ScenarioQualityExecutionBudgetV1.build(raw)


@pytest.mark.parametrize('mutation', [
    lambda raw: raw['decisionPoint']['selector'].update(remainingCards=9),
    lambda raw: raw['decisionPoint']['selector'].update(remainingCards=8),
    lambda raw: raw['decisionPoint']['selector'].pop('position'),
    lambda raw: raw['decisionPoint']['selector'].update(position=1),
    lambda raw: raw['decisionPoint']['selector'].update(actingSeat='east'),
    lambda raw: raw['decisionPoint'].update(kind='bid', selector={}),
    lambda raw: raw.update(constraints={'op': 'any', 'children': [raw['constraints'],
        {'predicateId': 'PRED-OTHER', 'kind': 'auction.contract', 'version': 1,
         'hardness': 'hard', 'arguments': {'contract': 'sun'}}]}),
])
def test_family_expression_cannot_be_dropped_or_made_optional(target, mutation):
    raw = scenario(target).to_dict()
    raw.pop('sha256')
    mutation(raw)
    with pytest.raises(ValueError):
        module.validate_family_target_scenario_v1(target, ResearchScenarioV1.build(raw))


@pytest.mark.parametrize('index', [16, 19, 24, 27])
def test_family_selector_agrees_with_native_per_actor_hand_count(target, index):
    raw_target = copy.deepcopy(target)
    raw_target.pop('sha256')
    raw_target['bindings']['playIndex'] = index
    raw_target['tagQuery']['trickBand'] = [f'trick-{index // 4 + 1}']
    actual = module.build_family_target_v1(raw_target)
    raw = scenario(actual).to_dict()
    raw.pop('sha256')
    raw['decisionPoint']['selector'].update(remainingCards=8 - index // 4,
        trickIndex=index // 4, position=index % 4)
    module.validate_family_target_scenario_v1(actual, ResearchScenarioV1.build(raw))


def test_unsupported_enabled_template_rejected_before_construction(target):
    raw = copy.deepcopy(target['catalog'])
    raw.pop('sha256')
    raw['families'][0]['template']['expression']['children'].append({
        'predicateId': 'PRED-UNSUPPORTED', 'kind': 'proof.winning_root_action_count',
        'version': 1, 'hardness': 'hard', 'arguments': {'minimum': 1, 'maximum': 1}})
    with pytest.raises(ValueError, match='no exact context'):
        ResearchFamilyCatalogV1.build(raw)


@pytest.mark.parametrize('contract,trump', [('hokum', None), ('sun', 'clubs')])
def test_history_generator_rejects_mismatched_contract_and_trump(target, contract, trump):
    raw = copy.deepcopy(target)
    raw.pop('sha256')
    raw['catalog'].pop('sha256')
    raw['bindings']['contract'] = contract
    raw['tagQuery']['mode'] = [contract]
    raw['trump'] = trump
    with pytest.raises(ValueError, match='Sun without trump or Hokum with trump'):
        module.build_family_target_v1(raw)


def test_guard_tag_query_can_target_an_enabled_structural_family(target):
    raw = copy.deepcopy(target)
    raw.pop('sha256')
    raw['catalog'].pop('sha256')
    guard = raw['catalog']['families'][2]
    guard['enabled'] = True
    raw['familyId'] = guard['id']
    raw['tagQuery'] = {'impact': ['round-flip'], 'technique': ['guard/protect'], 'mode': ['sun']}
    built = module.build_family_target_v1(raw)
    assert built['familyId'] == 'stopper_guard'
    # Capability targeting remains a request; result admission independently derives tags.
    assert built['tagQuery']['technique'] == ['guard/protect']


@pytest.mark.real_clock
def test_family_stream_is_lazy_bounded_and_deduplicates_decks(monkeypatch, target):
    calls, progress = [], []
    monkeypatch.setattr(module.time, 'perf_counter', lambda: 100)
    def construct(**kwargs):
        calls.append(kwargs)
        return {'status': 'proposal', 'deckOrder': list(range(32))}
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', construct)
    monkeypatch.setattr(module, 'family_proposal_witness_v1', lambda proposal, **kw: {'nativeWitness': True})
    config = SimpleNamespace(family_target=target, proposal_seed_count=4,
                             proposal_maximum_seconds=3, video_interestingness={},
                             early_play_policy_id='observer_safe_trick_control_v1')
    params = dict(maximumCampaignSeconds=10, maximumSeconds=5, maximumCandidates=2,
                  maximumConflicts=100, seed=10)
    rows = module.family_proposal_stream_v1(object(), config, params, campaign_started=100,
        cancel_check=lambda: False, progress_callback=progress.append)
    assert not calls
    assert list(rows) == [(10, {'nativeWitness': True}), (11, {'nativeWitness': True})]
    assert len(calls) == 2 and calls[0]['maximum_seconds'] == 3
    assert calls[1]['excluded_deck_orders'] == (tuple(range(32)),)
    assert all(call['buyer'] == target['buyer'] for call in calls)
    assert all('_select_query_suit' not in call for call in calls)
    assert all(row['proposalOnly'] and row['familyTargetHash'] == target['sha256'] for row in progress)
    assert list(module.family_proposal_stream_v1(object(), config, params, campaign_started=100,
        cancel_check=lambda: True, progress_callback=progress.append)) == []
    assert len(calls) == 2


@pytest.mark.real_clock
def test_family_stream_forwards_qualified_auction_generator(monkeypatch, target):
    monkeypatch.setattr(module.time, 'perf_counter', lambda: 100)
    calls, progress = [], []
    def provider(*a, **kw):
        return None
    def construct(**kwargs):
        calls.append(kwargs)
        return {'status': 'seed_miss', 'auctionQualification': {'status': 'complete', 'allPassed': False}}
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', construct)
    config = SimpleNamespace(family_target=target, proposal_seed_count=1,
        proposal_maximum_seconds=3, video_interestingness={}, early_play_policy_id='observer_safe_trick_control_v1')
    params = dict(maximumCampaignSeconds=10, maximumSeconds=5, maximumCandidates=2,
        maximumConflicts=100, seed=10)
    assert list(module.family_proposal_stream_v1(object(), config, params, campaign_started=100,
        cancel_check=lambda: False, progress_callback=progress.append, auction_provider=provider)) == []
    assert calls[0]['auction_provider'] is provider
    assert progress[0]['auctionQualification'] == {'status': 'complete', 'allPassed': False}


@pytest.mark.parametrize('mode,visibility,exact_from,expected', [
    ('budgeted', 'actor_information_set', 7, 24), ('budgeted', 'actor_information_set', 5, 16),
    ('budgeted', 'actor_information_set', 8, 24), ('budgeted', 'actor_information_set', 1, 0),
    ('exact', 'actor_information_set', 7, 0), ('budgeted', 'complete_state', 7, 0)])
def test_family_stream_constrains_only_the_declared_policy_prefix(root, monkeypatch, target, mode, visibility, exact_from, expected):
    calls = []
    def construct(**kwargs):
        calls.append(kwargs)
        return {'status': 'seed_miss'}
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', construct)
    config = SimpleNamespace(family_target=target, proposal_seed_count=1, proposal_maximum_seconds=3,
        video_interestingness={}, early_play_policy_id='observer_safe_trick_control_v1', exact_play_from_trick=exact_from)
    parameters = dict(maximumCampaignSeconds=10, maximumSeconds=5, maximumCandidates=1, maximumConflicts=100, seed=10)
    assert list(module.family_proposal_stream_v1(object(), config, parameters,
        campaign_started=module.time.perf_counter(), cancel_check=lambda: False, progress_callback=lambda row: None,
        auction_provider=object(), playing_quality=SimpleNamespace(mode=mode, visibility=visibility))) == []
    assert calls[0].get('policy_prefix_steps', 0) == expected


@pytest.mark.real_clock
@pytest.mark.parametrize('status,attempts', [('infeasible', 1), ('incomplete', 3)])
def test_only_exact_infeasibility_stops_remaining_seeds(monkeypatch, target, status, attempts):
    calls, progress = [], []
    monkeypatch.setattr(module.time, 'perf_counter', lambda: 100)
    def construct(**kwargs):
        calls.append(kwargs)
        return {'status': status}
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', construct)
    config = SimpleNamespace(family_target=target, proposal_seed_count=3,
        proposal_maximum_seconds=3, video_interestingness={},
        early_play_policy_id='observer_safe_trick_control_v1')
    params = dict(maximumCampaignSeconds=10, maximumSeconds=5, maximumCandidates=2,
        maximumConflicts=100, seed=10)
    assert list(module.family_proposal_stream_v1(object(), config, params,
        campaign_started=100, cancel_check=lambda: False, progress_callback=progress.append)) == []
    assert len(calls) == attempts
    assert all(row['constructionStatus'] == status for row in progress)


@pytest.mark.real_clock
def test_seed_miss_reason_is_retained_without_becoming_qualification(monkeypatch, target):
    progress = []
    now = [100]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: now[0])
    def construct(**kwargs):
        now[0] += 1.25
        return {'status': 'seed_miss', 'reason': 'decision_actor_mismatch',
                'constructionInfeasibleCore': ['decision_actor']}
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', construct)
    config = SimpleNamespace(family_target=target, proposal_seed_count=1,
        proposal_maximum_seconds=3, video_interestingness={},
        early_play_policy_id='observer_safe_trick_control_v1')
    params = dict(maximumCampaignSeconds=10, maximumSeconds=5, maximumCandidates=2,
        maximumConflicts=100, seed=10)
    assert list(module.family_proposal_stream_v1(object(), config, params,
        campaign_started=100, cancel_check=lambda: False, progress_callback=progress.append)) == []
    assert progress[0]['constructionReason'] == 'decision_actor_mismatch'
    assert progress[0]['constructionInfeasibleCore'] == ['decision_actor']
    assert progress[0]['constructionElapsedSeconds'] == 1.25
    assert progress[0]['proposalOnly'] and progress[0]['proposalCount'] == 0


@pytest.mark.real_clock
def test_constructed_proposal_that_exhausts_budget_has_disposition(monkeypatch, target):
    now, progress = [100], []
    monkeypatch.setattr(module.time, 'perf_counter', lambda: now[0])
    def construct(**kwargs):
        now[0] = 110
        return {'status': 'proposal', 'deckOrder': list(range(32))}
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', construct)
    config = SimpleNamespace(family_target=target, proposal_seed_count=1,
        proposal_maximum_seconds=3, video_interestingness={}, early_play_policy_id='observer_safe_trick_control_v1')
    params = dict(maximumCampaignSeconds=10, maximumSeconds=5, maximumCandidates=2,
                  maximumConflicts=100, seed=10)
    assert list(module.family_proposal_stream_v1(object(), config, params, campaign_started=100,
        cancel_check=lambda: False, progress_callback=progress.append)) == []
    assert progress[0]['proposalCount'] == 1 and progress[0]['proposalOrdinal'] == 10
    assert progress[0]['proposalDisposition'] == 'rejected_before_dispatch'
    assert progress[0]['dispositionReason'] == 'construction_budget_exhausted'
    assert progress[0]['nativeAttemptStarted'] is False


def test_seed_hints_are_bounded_reused_and_never_replace_native_auction_checks(monkeypatch, target):
    seen = []
    provider = object()
    def construct(**kwargs):
        assert kwargs['auction_provider'] is provider
        seen.append(kwargs['seed'])
        return {'status': 'incomplete', 'reason': 'auction_quality_incomplete'}
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', construct)
    config = SimpleNamespace(family_target=target, proposal_seed_count=2,
        proposal_maximum_seconds=3, video_interestingness={}, early_play_policy_id='observer_safe_trick_control_v1')
    params = dict(maximumCampaignSeconds=10, maximumSeconds=5, maximumCandidates=2, maximumConflicts=100, seed=10)
    for seeds in ((29, 31), (29, 31), ()):
        progress = []
        assert list(module.family_proposal_stream_v1(object(), config, params,
            campaign_started=module.time.perf_counter(), cancel_check=lambda: False,
            progress_callback=progress.append, auction_provider=provider, source_seeds=seeds)) == []
        assert [row['screenedSeed'] for row in progress] == list(seeds)
        assert all(row['constructionReason'] == 'auction_quality_incomplete' for row in progress)
    assert seen == [29, 31, 29, 31]


@pytest.mark.parametrize('seeds', [(1, 1), (1, 2, 3), (True,), (-1,), (1.5,), '12'])
def test_invalid_seed_hints_fail_before_construction(monkeypatch, target, seeds):
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', lambda **kw: pytest.fail('invalid seeds dispatched'))
    config = SimpleNamespace(family_target=target, proposal_seed_count=2)
    with pytest.raises(ValueError, match='source seeds'):
        list(module.family_proposal_stream_v1(object(), config, {'maximumCampaignSeconds': 10},
            campaign_started=module.time.perf_counter(), cancel_check=lambda: False,
            progress_callback=lambda row: None, source_seeds=seeds))


@pytest.mark.real_clock
@pytest.mark.parametrize('proposal_limit,expected', [(3, [3, 3, 3, 1]), (8, [5, 5])])
def test_incomplete_attempts_use_remaining_campaign_not_first_attempt_deadline(monkeypatch, target, proposal_limit, expected):
    now, allowances = [100], []
    monkeypatch.setattr(module.time, 'perf_counter', lambda: now[0])
    def construct(**kwargs):
        allowances.append(kwargs['maximum_seconds'])
        now[0] += kwargs['maximum_seconds']
        return {'status': 'incomplete'}
    monkeypatch.setattr(module, 'construct_history_family_proposal_v1', construct)
    config = SimpleNamespace(family_target=target, proposal_seed_count=10,
        proposal_maximum_seconds=proposal_limit, video_interestingness={},
        early_play_policy_id='observer_safe_trick_control_v1')
    params = dict(maximumCampaignSeconds=10, maximumSeconds=5, maximumCandidates=2,
        maximumConflicts=100, seed=10)
    assert list(module.family_proposal_stream_v1(object(), config, params,
        campaign_started=100, cancel_check=lambda: False, progress_callback=lambda row: None)) == []
    assert allowances == expected


@pytest.mark.parametrize('by_query,tail_first', [(False, False), (True, False), (False, True)])
def test_foundry_routes_frozen_family_stream_into_native_campaign(monkeypatch, root, target, by_query, tail_first):
    from baloot_platform.research import service as service_module
    from baloot_platform.research.service import ResearchFoundryServiceV1
    from baloot_platform.research.ortools_research_executor import OrToolsResearchCampaignResultV1
    from baloot_platform.research import ortools_research_executor
    monkeypatch.setattr(ortools_research_executor, 'validate_content_interest_rules_v1', lambda *a: None)
    monkeypatch.setattr(service_module, 'partition_research_shards', lambda **kw: (object(),))
    monkeypatch.setattr(service_module, '_ortools_line_count_boundaries_v1', lambda *a: {})
    service = object.__new__(ResearchFoundryServiceV1)
    service.maximum_local_shard_workers = 1
    service.maximum_shard_attempts = 1
    service._cancel_check = lambda code: False
    service._ortools_execution_parameters_v2 = lambda *a: dict(maximumCampaignSeconds=10,
        maximumSeconds=5, maximumSolveSeconds=5, seed=10, maximumCandidates=2, symmetryPolicyId='rules_preserving_v1',
        allowSymmetryOnlyVariants=False, minimumDecisionStateDistance=1, maximumProofNodes=20,
        maximumProofDepth=32, maximumHiddenWorlds=4, maximumLineCountLimbs=64,
        maximumConflicts=100, maximumPrefixLiftAttempts=1, requestedShardWorkers=1,
        cancellationPollMilliseconds=25)
    service.store = SimpleNamespace(list_shards=lambda code: (),
        get_run=lambda code: {'status': 'cancelled'}, update_run=lambda *a, **kw: None)
    events, in_progress = [], False
    service.store.get_run = lambda code: {'status': 'running' if in_progress else 'cancelled', 'events': []}
    service.store.update_run = lambda *a, **kw: events.extend(kw.get('events', []))
    documents = []
    service.artifact_store = SimpleNamespace(put=lambda payload, **kw: documents.append(json.loads(payload)) or {})
    service._runtime_proposal_queue = lambda *a, **kw: pytest.fail('organic producer used')
    service._runtime_proposal_stream = service._runtime_proposal_queue
    emitted = []
    def auction_provider(*a, **kw):
        return None
    def produce(*a, **kw):
        nonlocal in_progress
        if tail_first:
            assert kw['request'] is recipe.proof
            assert kw['proof']['maximumNodes'] == 20 and kw['proof']['maximumDepth'] == 32
            assert kw['search']['maximumPrefixLiftAttempts'] == 1
            assert 'auction_provider' not in kw
            in_progress = True
            kw['progress_callback']({'currentStage': 'tail_first_proposal', 'proposalOnly': True,
                'constructionSource': {'privateCompletePrefix': 'future cards and final score'}})
            in_progress = False
        else:
            assert kw['auction_provider'] is auction_provider
        assert kw['scenario'] is frozen
        assert kw['playing_quality'] is recipe.playing_quality
        emitted.append(True)
        yield 10, {'witness': True}
    producer = ('tail_first_proposal_stream_v1' if tail_first else
                'family_query_proposal_stream_v1' if by_query else 'family_proposal_stream_v1')
    if tail_first:
        from baloot_platform.research import ortools_tail_first_candidate as producer_module
    else:
        producer_module = module
    monkeypatch.setattr(producer_module, producer, produce)
    frozen = scenario(target)
    def execute(rules, actual, proof, **kw):
        assert actual is frozen and not emitted
        assert kw['shard_worker_count'] == 1
        assert list(kw['runtime_proposal_stream']) == [(10, {'witness': True})]
        return OrToolsResearchCampaignResultV1('not_found_within_budget', (), (), (), (),
            1, 2, 'rules_preserving_v1', 'decision_state_card_hamming_v1', 1, 0, 0)
    generation = generation_request(query_execution(root, target), mode=['sun']) if by_query else None
    backend = SimpleNamespace(authority=SimpleNamespace(quality_execution=SimpleNamespace(
        family_target=None if by_query else target, candidate_source='runtime_proposal_stream_v1',
        family_proposal_mode='native_tail_first_v1' if tail_first else None, video_interestingness=None)),
        _validate_generation_request=lambda request, *a: request == generation or pytest.fail('request changed'),
        execute_campaign=execute, qualified_auction_provider=lambda rules, quality: auction_provider)
    backend.for_run = lambda: backend
    recipe = SimpleNamespace(sha256='recipe', scenario=frozen, proof=object(), playing_quality=object(), generation=generation,
        output={'resultCount': 1}, resources={'seedStart': 10, 'seedCount': 2, 'seedsPerShard': 2})
    service._execute_v4_ortools_run({'runCode': 'run', 'runHash': 'hash', 'status': 'running', 'startedAt': 'now'},
        recipe, SimpleNamespace(bundle=SimpleNamespace(rules=object(), components={})), backend)
    assert emitted == [True]
    assert documents[-1]['runtimeProposalQueue'] == [{'witness': True}]
    if tail_first:
        source = {'privateCompletePrefix': 'future cards and final score'}
        assert documents[0] == source
        assert module.canonical_sha256(source) in documents[-1]['proposalConstructionEvidence']
        assert events and all('constructionSource' not in event for event in events)
        assert 'future cards' not in json.dumps(events)


def test_native_tail_proposal_mode_is_frozen_closed_and_schema_driven(root, target):
    from jsonschema import Draft202012Validator
    raw = execution(root, target)
    legacy = ScenarioQualityExecutionBudgetV1.build(raw)
    native = ScenarioQualityExecutionBudgetV1.build({**raw, 'familyProposalMode': 'native_tail_first_v1'})
    assert 'familyProposalMode' not in legacy.to_dict()
    assert native.sha256 != legacy.sha256
    assert ScenarioQualityExecutionBudgetV1.build(native.to_dict()).to_dict() == native.to_dict()
    assert ScenarioQualityExecutionBudgetV1.build(raw).sha256 == legacy.sha256
    profile = yaml.safe_load((root / 'configs/challenge/intelligent_research.yaml').read_text(encoding='utf-8'))
    profile['scenarioResearchV4']['qualityExecution'] = native.to_dict()
    profile['scenarioResearchV4']['qualityExecution'].pop('sha256')
    schema = json.loads((root / 'schemas/profiles/challenge.schema.json').read_text(encoding='utf-8'))
    validator = Draft202012Validator(schema)
    validator.validate(profile)
    for change in ({'familyProposalMode': None}, {'familyProposalMode': True},
                   {'familyProposalMode': 'rng'}, {'familyOpeningMode': 'native_project_auction_v1'}):
        with pytest.raises(ValueError):
            ScenarioQualityExecutionBudgetV1.build({**raw, 'familyProposalMode': 'native_tail_first_v1', **change})
        invalid = copy.deepcopy(profile)
        invalid['scenarioResearchV4']['qualityExecution'].update(change)
        assert not validator.is_valid(invalid)
    query = query_execution(root, target).to_dict()
    query.pop('sha256')
    with pytest.raises(ValueError, match='familyProposalMode'):
        ScenarioQualityExecutionBudgetV1.build({**query, 'familyProposalMode': 'native_tail_first_v1'})
    assert (root / 'schemas/profiles/challenge.schema.json').read_bytes() == (
        root / 'src/baloot_platform/resources/schemas/profiles/challenge.schema.json').read_bytes()


@pytest.mark.parametrize('mode,visibility', [('exact', 'actor_information_set'),
    ('exact', 'complete_state'), ('bounded', 'complete_state')])
def test_tail_mode_preflight_and_campaign_reject_unsupported_proof_before_construction(
        root, target, monkeypatch, mode, visibility):
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.research.scenario_backend import OrToolsScenarioResearchBackendV2
    from baloot_platform.research.scenario_contracts import ScenarioProofRequestV1
    profile = yaml.safe_load((root / 'configs/challenge/intelligent_research.yaml').read_text(encoding='utf-8'))
    profile['scenarioResearchV4']['qualityExecution'] = {
        **execution(root, target), 'familyProposalMode': 'native_tail_first_v1'}
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')
    backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile, rules=rules,
        bundle_identity={'bundleRevisionId': 'CBR-TAIL-TEST', 'semanticBundleHash': 'a' * 64,
                         'effectiveRuntimeConfigHash': 'b' * 64})
    proof = dict(schemaVersion=1, mode=mode, visibility=visibility, strategyClass='pure_deterministic',
        claim='winning_root_action_count', targetTeam='north_south', utility='awarded_point_margin',
        successThreshold=1, budgetProfile='late_game_exact_v1')
    if mode == 'bounded':
        proof.update(horizon={'unit': 'plies', 'value': 1}, boundaryEvaluator='current_raw_point_margin_v1')
    quality = project_quality(mode='budgeted')
    recipe = dict(scenario=scenario(target).to_dict(), proof=proof, playingQuality=quality.to_dict())
    errors = [row for row in backend.validate(recipe)['issues'] if row['code'] == 'family_target_inapplicable']
    supported = mode == 'exact' and visibility == 'actor_information_set'
    assert bool(errors) is not supported
    if supported:
        construction, plan = module.family_tail_first_plan_v1(rules, backend.authority.quality_execution,
            scenario=scenario(target), request=ScenarioProofRequestV1.build(proof), playing_quality=quality)
        assert plan.boundary_combinations == (('east', 'sun', 'north', None),)
        assert any(child.sha256 == scenario(target).constraints.sha256 for child in construction.constraints.children)
    else:
        monkeypatch.setattr(backend, '_new_executor', lambda *a, **kw: pytest.fail('unsupported proof launched'))
        with pytest.raises(ValueError, match='unchanged exact actor'):
            backend.execute_campaign(rules, scenario(target), ScenarioProofRequestV1.build(proof),
                playing_quality=quality, runtime_proposal_stream=iter(()))
