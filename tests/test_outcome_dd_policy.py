import copy
import time
from types import SimpleNamespace

import pytest

from baloot_platform.analysis.models import InformationState
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.outcome_generation import (
    OutcomePolicyDriver, OutcomePolicyIncomplete, load_outcome_policy, validate_outcome_policy,
)
from baloot_platform.research.outcome_spec import compile_outcome_spec


@pytest.fixture
def context(root):
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v7.yaml')
    policy = load_outcome_policy(root / 'configs/research/outcome_policy_v2.yaml')
    compiled = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'], mode='sun'), rules)
    engine = BalootEngine(rules)
    state = engine.create_round(seed=92, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state)
    engine.bid_and_complete_deal(state, forced_contract=Contract.SUN, forced_buyer=Seat.NORTH)
    engine.prepare_projects(state)
    info = InformationState.from_complete(state, state.current_player)
    return OutcomePolicyDriver(rules, compiled, policy, seed=1, deadline=time.monotonic()+30), state, info


@pytest.mark.real_clock
@pytest.mark.parametrize('second,means', [([-10, 30], {'S7': 10, 'S8': 5}),
    ([-10, 40], {'S7': 10, 'S8': 10})])
def test_pimc_is_observer_only_paired_and_uses_the_mean(context, monkeypatch, second, means):
    import baloot_platform.research.outcome_generation as module
    driver, state, info = context
    samples, seeds, calls = (object(), object()), [], []
    def sample(observation, engine, **kwargs):
        assert observation is info and isinstance(observation, InformationState)
        assert kwargs['requested_samples'] == 32
        assert kwargs['exact_allocation_limit'] == 32
        seeds.append(kwargs['seed'])
        return SimpleNamespace(states=samples, coverage_complete=True, method='exhaustive_test', attempted=2)
    def solve(world, **kwargs):
        assert world in samples and world is not state
        assert kwargs['all_root_values'] is True
        calls.append(world)
        values = [30, -20] if world is samples[0] else second
        return dict(status='exact', rootActions=[dict(action=a, value=v) for a, v in zip(('S7', 'S8'), values)])
    monkeypatch.setattr(module, 'generate_hidden_deals', sample)
    driver.informed_dd[info.observer] = SimpleNamespace(solve=solve)
    driver.dd = SimpleNamespace(solve=lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError('true-deal cache must not enter PIMC')))
    selected, evidence, guarantee, scope = driver.dd_card(state, info, 'informed')
    assert selected.code == 'S7'  # 10 mean, versus 5; not vote count or true deal.
    assert evidence['meanValues'] == means
    assert guarantee == 'sampled_policy_not_exact' and scope == info.to_dict()
    assert evidence['sampledWorlds'] == 2 and evidence['exhaustiveCohort']
    # Hidden actual hands and the construction seed cannot change the sampler.
    driver.card_cache.clear()
    driver.seed += 100
    driver.dd_card(None, info, 'informed')
    assert len(calls) == 4 and len(set(seeds)) == 1
    driver.dd_card(None, info, 'informed')
    assert len(calls) == 4  # A CEGAR revisit reuses this completed observation.


def test_frozen_dd_profile_is_packaged_without_mutating_v1(root):
    assert (root / 'configs/research/outcome_policy_v2.yaml').read_bytes() == (
        root / 'src/baloot_platform/resources/configs/research/outcome_policy_v2.yaml').read_bytes()
    assert 'cardPolicy' not in load_outcome_policy(root / 'configs/research/outcome_policy_v1.yaml')


@pytest.mark.real_clock
def test_informed_workspaces_are_separate_per_observer(context, monkeypatch):
    import baloot_platform.research.outcome_generation as module
    driver, state, first = context
    created = []
    def factory(rules):
        instance = SimpleNamespace(solve=lambda *a, **kw: dict(status='exact',
            rootActions=[dict(action='S7', value=1), dict(action='S8', value=0)]))
        created.append(instance)
        return instance
    monkeypatch.setattr(module, 'DoubleDummySolver', factory)
    monkeypatch.setattr(module, 'generate_hidden_deals', lambda *a, **kw: SimpleNamespace(
        states=(object(),), coverage_complete=True, method='test', attempted=1))
    driver.dd_card(None, first, 'informed')
    driver.engine.play_card(state, driver.engine.legal_play_actions(state)[0].card)
    second = InformationState.from_complete(state, state.current_player)
    driver.dd_card(None, second, 'informed')
    assert len(created) == 2 and driver.dd is None
    assert driver.informed_dd[first.observer] is created[0]
    assert driver.informed_dd[second.observer] is created[1]


@pytest.mark.real_clock
@pytest.mark.parametrize('partial', ['sample', 'world'])
def test_pimc_fails_closed_without_a_complete_cohort(context, monkeypatch, partial):
    import baloot_platform.research.outcome_generation as module
    driver, _, info = context
    monkeypatch.setattr(module, 'generate_hidden_deals', lambda *a, **kw: SimpleNamespace(
        states=(object(),), coverage_complete=partial == 'world', method='test', attempted=1))
    driver.informed_dd[info.observer] = SimpleNamespace(solve=lambda *a, **kw: dict(status='incomplete'))
    with pytest.raises(OutcomePolicyIncomplete, match='pimc_'):
        driver.dd_card(None, info, 'informed')
    assert not driver.card_cache and not driver.receipts


@pytest.mark.real_clock
def test_optimal_dd_has_no_fallback_and_uses_the_exact_scope(context):
    driver, state, info = context
    def solve(actual, **kwargs):
        assert actual is state and not kwargs.get('all_root_values')
        return dict(status='exact', selectedAction='S7')
    driver.dd = SimpleNamespace(solve=solve)
    selected, _, guarantee, scope = driver.dd_card(state, info, 'optimal')
    assert selected.code == 'S7' and guarantee == 'exact_complete_state'
    assert scope == state.to_dict(reveal_hands=True)
    driver.card_cache.clear()
    driver.dd = SimpleNamespace(solve=lambda *a, **kw: dict(status='incomplete'))
    with pytest.raises(OutcomePolicyIncomplete, match='dd_optimal_incomplete'):
        driver.dd_card(state, info, 'optimal')


@pytest.mark.real_clock
@pytest.mark.parametrize('key,value', [('maximumWallSeconds', 6), ('maximumCpuSeconds', 6),
    ('maximumNodes', 10000001), ('maximumCacheEntries', 1000001), ('tieBreak', 'random')])
def test_new_profile_does_not_increase_time_cpu_or_memory_bounds(context, key, value):
    driver, _, _ = context
    bad = copy.deepcopy(driver.policy)
    bad['cardPolicy'][key] = value
    with pytest.raises(ValueError):
        validate_outcome_policy(bad)


def test_batch_revision_is_packaged_and_does_not_mutate_the_serial_revisions(root):
    policy = load_outcome_policy(root / 'configs/research/outcome_policy_v3.yaml')
    assert (root / 'configs/research/outcome_policy_v3.yaml').read_bytes() == (
        root / 'src/baloot_platform/resources/configs/research/outcome_policy_v3.yaml').read_bytes()
    assert policy['cardPolicy']['maximumThreads'] == 4 and policy['cardPolicy']['particles'] == 32
    assert policy['cardPolicy']['maximumWallSeconds'] == policy['cardPolicy']['maximumCpuSeconds'] == 40
    assert policy['cardPolicy']['maximumNodes'] == 25000000
    old = load_outcome_policy(root / 'configs/research/outcome_policy_v2.yaml')['cardPolicy']
    assert old['method'] == 'native_dd_pimc_v1' and 'maximumThreads' not in old
    assert old['maximumWallSeconds'] == old['maximumCpuSeconds'] == 5
    assert old['maximumNodes'] == 10000000


@pytest.mark.real_clock
@pytest.mark.parametrize('incomplete', [False, True])
def test_pimc_batch_is_one_paired_call_and_never_accepts_a_partial_world(context, root, monkeypatch, incomplete):
    import baloot_platform.research.outcome_generation as module
    driver, state, info = context
    driver.policy = load_outcome_policy(root / 'configs/research/outcome_policy_v3.yaml')
    driver.maximum_threads = 1  # Four rollout workers each get one native worker.
    samples, calls = (object(), object()), []
    monkeypatch.setattr(module, 'generate_hidden_deals', lambda *a, **kw: SimpleNamespace(
        states=samples, coverage_complete=True, method='exhaustive_test', attempted=2))
    def solve_many(worlds, **kwargs):
        assert worlds is samples and kwargs['maximum_threads'] == 1
        assert kwargs['maximum_nodes'] == 25000000
        assert 5 < kwargs['maximum_seconds'] <= 40
        calls.append(worlds)
        return dict(status='incomplete' if incomplete else 'exact', worlds=[
            dict(status='exact', rootActions=[dict(action='S7', value=30), dict(action='S8', value=-20)]),
            dict(status='incomplete' if incomplete else 'exact', rootActions=[] if incomplete else [
                dict(action='S7', value=-10), dict(action='S8', value=30)])])
    driver.informed_dd[info.observer] = SimpleNamespace(solve_many=solve_many)
    if incomplete:
        with pytest.raises(OutcomePolicyIncomplete, match='pimc_world_incomplete'):
            driver.dd_card(state, info, 'informed')
        assert not driver.card_cache
    else:
        selected, evidence, guarantee, scope = driver.dd_card(state, info, 'informed')
        assert selected.code == 'S7' and evidence['meanValues'] == {'S7': 10, 'S8': 5}
        assert guarantee == 'sampled_policy_not_exact' and scope == info.to_dict()
    assert calls == [samples]


@pytest.mark.real_clock
def test_bounded_exhaustive_cohort_uses_all_small_worlds_or_a_deterministic_K_subset(context, root, monkeypatch):
    import baloot_platform.research.outcome_generation as module
    driver, state, info = context
    driver.policy = load_outcome_policy(root / 'configs/research/outcome_policy_v3.yaml')
    samples, seen = tuple(object() for _ in range(50)), []
    def sample(*args, **kwargs):
        assert kwargs['exact_allocation_limit'] == 2000
        return SimpleNamespace(states=samples, coverage_complete=True, method='exhaustive_test', attempted=50)
    monkeypatch.setattr(module, 'generate_hidden_deals', sample)
    def solve_many(worlds, **kwargs):
        assert len(worlds) == len(set(worlds)) == 32
        seen.append(worlds)
        return dict(status='exact', worlds=[dict(status='exact', rootActions=[dict(action='S7', value=1)]) for _ in worlds])
    driver.informed_dd[info.observer] = SimpleNamespace(solve_many=solve_many)
    _, evidence, _, _ = driver.dd_card(state, info, 'informed')
    assert not evidence['exhaustiveCohort'] and evidence['enumeratedWorlds'] == 50
    assert evidence['sampledWorlds'] == 32
    driver.card_cache.clear()
    driver.seed += 1000
    driver.dd_card(state, info, 'informed')
    assert seen[0] == seen[1]
