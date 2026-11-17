import copy
import csv
import json
import random
import time

import pytest

from baloot_platform.core.cards import Card
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.outcome_generation import OutcomePolicyDriver, load_outcome_policy
from baloot_platform.research.outcome_rollout import DealProposalStream, generate_outcomes, verify_policy_line
from baloot_platform.research.outcome_search import OutcomeSearch, critical_cards, deck_distance, perturb, target_distance
from baloot_platform.research.outcome_spec import compile_outcome_spec


def _batch_worker_ready(queue, barrier):
    import os
    queue.put(os.getpid())
    barrier.wait(timeout=120)


@pytest.fixture
def fixture(root):
    rules = RuleBook.load(root/'configs/rules/saudi_federation_2023_v7.yaml')
    policy = load_outcome_policy(root/'configs/research/outcome_policy_v3.yaml')
    compiled = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'],
        mode='sun', policies=dict(default='optimal')), rules)
    proposal = DealProposalStream(compiled, rules, policy, 20261025, time.monotonic()+20).next(10)
    context = dict(deckOrder=list(proposal.deck_order), dealer='east', auction=proposal.auction.to_dict(),
        escalation=dict(observation=dict(actions=[])))
    driver = OutcomePolicyDriver(rules, compiled, policy, seed=1, deadline=time.monotonic()+20)
    plays = driver.construct_line(context)
    okay, record = verify_policy_line(compiled, driver, context, plays)
    assert okay
    owners = {p.card.id: int(p.seat) for trick in driver.final_state.completed_tricks for p in trick.plays}
    context.update(cardOwners=[owners[c] for c in range(32)], purchaseCard=driver.final_state.purchase_card.id)
    yield compiled, rules, policy, context, driver, record
    driver.close()


@pytest.mark.real_clock
def test_critical_set_protects_named_deciding_project_and_baloot_cards(fixture):
    compiled, _, _, context, driver, _ = fixture
    named = copy.deepcopy(dict(compiled.source))
    named['outcome'] = dict(decidingTrick=dict(index=8, winningCard='C7', beatenCard='HA'))
    named = compile_outcome_spec(named, driver.rules)
    critical = critical_cards(named, driver.final_state)
    assert {Card.parse('C7').id, Card.parse('HA').id} <= critical
    assert {p.card.id for p in driver.final_state.completed_tricks[-1].plays} <= critical
    assert {c.id for p in driver.final_state.projects for c in p.cards} <= critical
    changed = perturb(context, random.Random(42), critical)
    assert changed is not None
    other, swaps = changed
    assert len(swaps) == 2 and sorted(other['deckOrder']) == list(range(32))
    assert deck_distance(context['deckOrder'], other['deckOrder']) == 4
    assert all(context['deckOrder'].index(c) == other['deckOrder'].index(c) for c in critical)
    assert changed == perturb(context, random.Random(42), critical)
    assert perturb(context, random.Random(42), range(32)) is None
    from baloot_platform.core.enums import Suit
    hokum = copy.copy(driver.final_state)
    hokum.trump_suit = Suit.HEARTS
    assert {Card.parse('HK').id, Card.parse('HQ').id} <= critical_cards(named, hokum)


@pytest.mark.real_clock
def test_target_distance_is_guidance_only_and_hill_search_uses_the_proposed_deal(fixture):
    compiled, _, _, context, driver, record = fixture
    search = OutcomeSearch(compiled, 42)
    assert search.next(lambda: context)[1]['kind'] == 'cp_sat'
    missed = copy.deepcopy(record)
    missed['failedPredicates'] = ['requested_winner']
    assert target_distance(compiled, driver.final_state, missed, False) > 0
    assert target_distance(compiled, driver.final_state, missed, True) == 0
    source = dict(kind='cp_sat')
    search.observe(context, driver.final_state, missed, False, source)
    mutated, provenance = search.next(lambda: pytest.fail('hill search restarted CP-SAT'))
    assert provenance['kind'] == 'guided'
    assert deck_distance(context['deckOrder'], mutated['deckOrder']) == 4
    search.admit(context, driver.final_state, 'parent', source)
    child, provenance = search.next(lambda: pytest.fail('neighbourhood restarted CP-SAT'))
    assert provenance['kind'] == 'neighbour'
    search.observe(child, driver.final_state, missed, False, provenance)
    assert search.neighbourhoods['parent']['tried'] == 1
    assert search.neighbourhoods['parent']['accepted'] == 0


@pytest.mark.real_clock
def test_batch_exports_are_deterministic_and_do_not_claim_sampled_is_exact(fixture, tmp_path):
    from baloot_platform.research.outcome_catalogue import catalogue_rows, write_exports
    compiled, _, _, _, _, record = fixture
    record.update(seed=1, diversityKey='a'*64)
    path = tmp_path/'receipt.json'
    path.write_text(json.dumps(record), encoding='utf-8')
    accepted = [dict(id='scenario-1', seed=1, path=str(path), sha256='b'*64, diversityKey='a'*64),
        dict(id='scenario-2', seed=2, path=str(path), sha256='b'*64, diversityKey='c'*64)]
    rows = catalogue_rows(accepted, compiled)
    assert rows == catalogue_rows(list(reversed(accepted)), compiled)
    assert rows[0]['finalScore'] == record['score']['awardedPoints']
    assert 'wins over' in rows[0]['summary']
    write_exports(tmp_path, rows)
    assert len(list(csv.DictReader((tmp_path/'catalogue.csv').open(encoding='utf-8')))) == 2
    assert 'sampled, not exact' in (tmp_path/'summaries.md').read_text(encoding='utf-8')


@pytest.mark.real_clock
def test_four_worker_batch_stops_at_distinct_count_with_replayable_exports(fixture, tmp_path, monkeypatch):
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor
    from contextlib import nullcontext
    from baloot_platform.research import outcome_rollout as module
    compiled, rules, policy, _, _, _ = fixture
    spawn = multiprocessing.get_context('spawn')
    ready = spawn.Queue()
    barrier = spawn.Barrier(4)
    # Qualify admission on real, ready workers; cold bootstrap has its own budget regression.
    with ProcessPoolExecutor(max_workers=4, mp_context=spawn,
            initializer=_batch_worker_ready, initargs=(ready, barrier)) as pool:
        probes = [pool.submit(os.getpid) for _ in range(4)]
        assert len({ready.get(timeout=120) for _ in range(4)}) == 4
        assert all(probe.result(timeout=10) for probe in probes)
        monkeypatch.setattr(module, 'ProcessPoolExecutor', lambda **kwargs: nullcontext(pool))
        result = generate_outcomes(compiled, rules, policy, count=4, seed=20261025,
            output=tmp_path/'batch', workers=4, maximum_seconds=20)
    ready.close()
    ready.join_thread()
    assert result['status'] == 'complete'
    assert result['distinctAccepted'] == len(result['plays']) == 4
    from collections import Counter
    stories = Counter(row['storyHash'] for row in result['plays'])
    assert max(stories.values()) <= 3
    assert result['storyDistinctAccepted'] == len(stories)
    decks = [json.loads(open(row['path'], encoding='utf-8').read())['exactTracePrefix']['deckOrder']
        for row in result['accepted']]
    assert all(deck_distance(a, b) >= 4 for i, a in enumerate(decks) for b in decks[i+1:])
    assert (tmp_path/'batch/catalogue.csv').exists() and (tmp_path/'batch/summaries.md').exists()
    assert (tmp_path/'batch/neighbourhoods.csv').exists()


def test_cli_generate_routes_without_building_a_media_bundle(monkeypatch):
    from baloot_platform import cli
    from baloot_platform.research import outcome_cli
    calls = []
    monkeypatch.setattr('sys.argv', ['baloot', 'generate', '--spec', 'x.yaml', '--count', '1'])
    monkeypatch.setattr(cli, 'build_bundle', lambda *_: pytest.fail('generate built a media bundle'))
    monkeypatch.setattr(cli, 'platform_root', lambda: pytest.fail('routing invoked the host launcher'))
    monkeypatch.setattr(outcome_cli, 'main', lambda args: calls.append(args))
    cli.main()
    assert calls == [['--spec', 'x.yaml', '--count', '1']]


def test_cli_help_lists_generate_and_misordered_generate_cannot_fall_through_to_video(monkeypatch, capsys):
    from baloot_platform import cli
    monkeypatch.setattr(cli, 'build_bundle', lambda *_: pytest.fail('generate reached media setup'))
    monkeypatch.setattr(cli, 'platform_root', lambda: pytest.fail('help invoked the host launcher'))
    monkeypatch.setattr('sys.argv', ['baloot', '--help'])
    with pytest.raises(SystemExit) as help_exit:
        cli.main()
    assert help_exit.value.code == 0 and 'baloot generate --help' in capsys.readouterr().out
    monkeypatch.setattr('sys.argv', ['baloot', '--seed', '1', 'generate'])
    with pytest.raises(SystemExit) as invalid_exit:
        cli.main()
    assert invalid_exit.value.code == 2


def test_script_entrypoint_needs_no_editable_install_or_pythonpath(root, tmp_path):
    import os
    import subprocess
    import sys
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, str(root/'scripts/generate_outcomes.py'), '--help'],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--spec' in result.stdout and '--count' in result.stdout


def test_shared_admission_rejects_symmetry_and_too_small_distance():
    from baloot_platform.research import outcome_rollout as module
    deck = list(range(32))
    near = deck.copy()
    near[0], near[1] = near[1], near[0]
    far = near.copy()
    far[2], far[3] = far[3], far[2]
    rows = [dict(id=i, diversityKey=key*64, deckOrder=cards, storyHash='a'*64)
        for i, key, cards in [('1', 'a', deck), ('2', 'a', deck), ('3', 'b', near), ('4', 'c', far)]]
    selected, rejected = module._merge_distinct(rows, 2, 3)
    assert [r['id'] for r in selected] == ['1', '4']
    assert rejected == {'duplicate_batch_symmetry': 1, 'duplicate_batch_deck_distance': 1}
    assert module._merge_distinct(list(reversed(rows)), 2, 3) == (selected, rejected)


def test_story_cap_reserves_across_workers_and_is_configurable():
    from baloot_platform.research import outcome_rollout as module
    rows = [dict(id=str(i), diversityKey=str(i+1)*64,
        deckOrder=list(range(i, 32))+list(range(i)), storyHash='a'*64, worker=i) for i in range(4)]
    selected, rejected = module._merge_distinct(rows, 4, 2)
    assert len(selected) == 2 and rejected == {'story_variant_cap': 2}
    assert module._merge_distinct(list(reversed(rows)), 4, 2) == (selected, rejected)
    assert len(module._merge_distinct(rows, 4, 3)[0]) == 3


@pytest.mark.real_clock
@pytest.mark.parametrize('gain_trick', [7, 8])
@pytest.mark.parametrize('tie_rule', ['buyer_makes', 'buyer_loses', None])
def test_summary_names_only_the_actual_bonus_trick_and_the_sun_tie_rule(fixture, tmp_path, gain_trick, tie_rule):
    from baloot_platform.research.outcome_catalogue import catalogue_rows, game_facts
    compiled, _, _, _, driver, record = fixture
    assert game_facts(driver.final_state, driver.engine.scoring_domain.last_trick_bonus_raw,
        tie_rule)['sunTieRule'] == (tie_rule or 'legacy_raw_units')
    record['facts'].update(biggestSwingTrick=gain_trick, finalScore=[19, 19], winner='north_south',
        sunTieRule=tie_rule)
    path = tmp_path/'receipt.json'
    path.write_text(json.dumps(record), encoding='utf-8')
    rows = catalogue_rows([dict(id='one', seed=1, path=str(path), sha256='a'*64, diversityKey='b'*64)], compiled)
    assert ('including the trick-8 bonus' in rows[0]['summary']) == (gain_trick == 8)
    assert f"Sun tie rule {tie_rule or 'legacy_raw_units'}" in rows[0]['summary']
    assert 'Sun tie rule None' not in rows[0]['summary']
    assert 'score 19-19' in rows[0]['summary']
    assert set(rows[0]['interestComponents']) == {'swing', 'closeness', 'lateness', 'rankUpset', 'declarations', 'baloot'}
    assert rows[0]['storySignature']['keyTrick'] == gain_trick
    team = record['facts']['tricks'][gain_trick-1]['team'].replace('_', '-').title()
    units = 'trick points' if gain_trick == 8 else 'raw card points'
    assert f"{team} gains {record['facts']['biggestScoreSwing']} {units} on trick {gain_trick}" in rows[0]['summary']
    assert 'raw-point gain' not in rows[0]['summary']


@pytest.mark.real_clock
def test_team_alone_is_explicitly_spec_fixed_but_an_open_spec_is_informed(fixture):
    compiled, rules, policy, _, _, _ = fixture
    base = dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'])
    for source, expected in [(base, False), ({**base, 'team': 'north_south'}, True)]:
        driver = OutcomePolicyDriver(rules, compile_outcome_spec(source, rules), policy,
            seed=1, deadline=time.monotonic()+5)
        try:
            assert driver.spec_fixed_contract == expected
        finally:
            driver.close()


@pytest.mark.real_clock
def test_suit_renamed_deal_is_rejected_globally_but_named_suits_stay_anchored(fixture):
    from baloot_platform.research import outcome_rollout as module
    from baloot_platform.research.outcome_generation import _prefix
    from baloot_platform.research.exact_state import ExactStateNormalizerV1
    compiled, rules, _, context, _, record = fixture
    prefix = record['exactTracePrefix']
    normalizer = ExactStateNormalizerV1(rules)
    original = normalizer.normalize(_prefix(context['deckOrder'], 'east', prefix['auctionActions'])).round_state
    renamed_deck = [((c//8+1) % 4)*8+c % 8 for c in context['deckOrder']]
    renamed = normalizer.normalize(_prefix(renamed_deck, 'east', prefix['auctionActions'])).round_state
    key = module._diversity_key(compiled, original, prefix['auctionActions'])
    other = module._diversity_key(compiled, renamed, prefix['auctionActions'])
    assert key == other
    named = compile_outcome_spec({**dict(compiled.source), 'outcome': {
        'decidingTrick': dict(index=8, winningCard='C7', beatenCard='HA')}}, rules)
    assert module._diversity_key(named, original, prefix['auctionActions']) != module._diversity_key(
        named, renamed, prefix['auctionActions'])
    selected, rejects = module._merge_distinct([dict(id=str(i), diversityKey=k, deckOrder=d, storyHash='a'*64)
        for i, k, d in [(0, key, context['deckOrder']), (1, other, renamed_deck)]], 2, 3)
    assert len(selected) == 1 and rejects == {'duplicate_batch_symmetry': 1}


@pytest.mark.real_clock
def test_two_four_worker_runs_have_the_same_content_ids(fixture, tmp_path):
    compiled, rules, policy, _, _, _ = fixture
    results = [generate_outcomes(compiled, rules, policy, count=2, seed=20261025, workers=4,
        maximum_seconds=60, output=tmp_path/f'run-{i}') for i in range(2)]
    assert all(r['status'] == 'complete' and r['reproducible'] for r in results)
    assert len(results[0]['plays']) == len(results[1]['plays']) == 2
    assert [p['id'] for p in results[0]['plays']] == [p['id'] for p in results[1]['plays']]
    assert [p['distinctnessHash'] for p in results[0]['plays']] == [p['distinctnessHash'] for p in results[1]['plays']]


@pytest.mark.real_clock
def test_pinned_story_fields_are_removed_and_free_history_changes_the_signature(fixture):
    from baloot_platform.research.outcome_catalogue import story_facts
    compiled, rules, _, _, _, record = fixture
    facts = copy.deepcopy(record['facts'])
    trick = facts['tricks'][7]
    winning = next(p['card'] for p in trick['plays'] if p['actor'] == trick['winner'])
    beaten = next(p['card'] for p in trick['plays'] if
        (trick['team'] == 'north_south') != (p['actor'] in {'north', 'south'}))
    spec = compile_outcome_spec({**dict(compiled.source), 'outcome': dict(winner=facts['winner'],
        decidingTrick=dict(index=8, winningCard=winning, beatenCard=beaten))}, rules)
    first = story_facts(facts, spec)
    signature = first['storySignature']
    assert not {'mode', 'keyTrick', 'winningCard', 'winner'} & set(signature)
    assert beaten not in signature['beatenCards']
    lower = copy.deepcopy(dict(spec.source))
    lower['outcome']['decidingTrick']['beatenCard'] = beaten.lower()
    assert story_facts(facts, compile_outcome_spec(lower, rules)) == first
    assert signature['granularity']['scoreBucket'] == 10
    assert len(signature['openingLedSuits']) == 3
    facts['tricks'][6]['winner'] = 'north' if facts['tricks'][6]['winner'] != 'north' else 'west'
    assert story_facts(facts, spec)['storyHash'] != first['storyHash']
    assert story_facts(facts, spec, dict(scoreBucket=5, openingTricks=1))['storyHash'] != first['storyHash']


@pytest.mark.real_clock
@pytest.mark.parametrize('pinned', ['trump', 'buyer', 'auction', 'baloot'])
def test_story_signature_excludes_the_contract_and_baloot_fields_pinned_by_the_spec(fixture, pinned):
    from baloot_platform.research.outcome_catalogue import story_facts
    compiled, rules, _, _, _, record = fixture
    source, facts = copy.deepcopy(dict(compiled.source)), copy.deepcopy(record['facts'])
    if pinned in {'trump', 'baloot'}:
        source.update(mode='hokum', trumpSuit='hearts')
        facts.update(mode='hokum', contract='hokum', trumpSuit='hearts')
    if pinned == 'buyer':
        source['buyer'] = facts['buyer']
    if pinned == 'auction':
        source['auction'] = record['exactTracePrefix']['auctionActions']
    if pinned == 'baloot':
        source['declarations'] = [dict(category='baloot')]
        facts['balootTeams'] = ['north_south']
    signature = story_facts(facts, compile_outcome_spec(source, rules))['storySignature']
    if pinned == 'trump':
        assert 'trump' not in signature['contract']
    elif pinned == 'buyer':
        assert 'buyer' not in signature
    elif pinned == 'auction':
        assert not {'buyer', 'mode', 'contract'} & signature.keys()
    else:
        assert 'baloot' not in signature


def test_help_does_not_need_the_host_launcher(monkeypatch):
    from baloot_platform.research import outcome_cli
    monkeypatch.setattr(outcome_cli, 'platform_root', lambda: pytest.fail('help requested a host launcher'))
    with pytest.raises(SystemExit) as result:
        outcome_cli.main(['--help'])
    assert result.value.code == 0


def test_real_generate_still_fails_closed_on_an_unavailable_launcher(monkeypatch):
    from baloot_platform.research import outcome_cli
    def unavailable():
        raise RuntimeError('host launcher unavailable')
    monkeypatch.setattr(outcome_cli, 'platform_root', unavailable)
    monkeypatch.setattr(outcome_cli.RuleBook, 'load', lambda *_: pytest.fail('generation bypassed the host cap'))
    with pytest.raises(RuntimeError, match='host launcher unavailable'):
        outcome_cli.main(['--spec', 'absent.yaml', '--count', '1', '--out', 'absent'])


@pytest.mark.real_clock
def test_zero_search_distance_cannot_admit_a_verifier_rejection(fixture, tmp_path, monkeypatch):
    from baloot_platform.research import outcome_rollout as module
    compiled, rules, policy, _, _, _ = fixture
    original, clock = module.verify_policy_line, time.monotonic
    def reject(*args):
        _, record = original(*args)
        record['failedPredicates'] = []  # Search distance is zero, but verification is not acceptance.
        monkeypatch.setattr(module.time, 'monotonic', lambda: clock()+1000)
        return False, record
    monkeypatch.setattr(module, 'verify_policy_line', reject)
    result = generate_outcomes(compiled, rules, policy, count=1, seed=20261025,
        output=tmp_path/'rejected', workers=1, maximum_seconds=15)
    assert result['accepted'] == [] and result['plays'] == []
    assert result['rejectReasons'] == {'outcome_miss': 1}


@pytest.mark.real_clock
@pytest.mark.parametrize('strategy', ['first_deviation', 'forward_rollout'])
def test_optional_refiners_keep_the_batch_export_contract(fixture, tmp_path, monkeypatch, strategy):
    from baloot_platform.research import outcome_rollout as module
    compiled, rules, policy, _, _, witness = fixture
    record = copy.deepcopy(witness)
    record.pop('facts')
    record['seed'] = 1
    monkeypatch.setattr(module, 'construct_outcome', lambda *args, **kwargs:
        dict(accepted=record, attempts=[dict(proposals=1)], rejectReasons={}))
    result = generate_outcomes(compiled, rules, policy, count=1, seed=1, workers=1,
        output=tmp_path/strategy, maximum_seconds=15, strategy=strategy)
    assert result['status'] == 'complete' and len(result['plays']) == 1
    assert result['plays'][0]['finalScore'] == witness['score']['awardedPoints']


@pytest.mark.real_clock
def test_process_cpu_is_not_double_counted_when_a_pool_process_runs_two_shards(fixture, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from baloot_platform.research import outcome_rollout as module
    compiled, rules, policy, _, _, _ = fixture
    class Pool:
        def __init__(self, **kwargs):
            self.calls = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, *args):
            index = self.calls
            self.calls += 1
            result = dict(worker=index, stages={}, rejectReasons={}, accepted=[], neighbourhoods={},
                processId=100 if index < 2 else 200, processCpuSeconds=[4, 11, 3, 7][index])
            return SimpleNamespace(result=lambda: result)
    monkeypatch.setattr(module, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(module.time, 'process_time', lambda: 10)
    result = generate_outcomes(compiled, rules, policy, count=4, seed=1, workers=4,
        output=tmp_path/'cpu', maximum_seconds=15)
    assert result['cpuSeconds'] == 18  # max(4, 11) + max(3, 7), not four shard totals.


@pytest.mark.real_clock
def test_cold_worker_startup_expiry_is_an_honest_partial(fixture, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from baloot_platform.research import outcome_rollout as module
    compiled, rules, policy, _, _, _ = fixture
    clock = time.monotonic
    class ColdPool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            monkeypatch.setattr(module.time, 'monotonic', lambda: clock()+1000)
            return self
        def __exit__(self, *args):
            pass
        def submit(self, function, *args):
            value = function(*args)
            return SimpleNamespace(result=lambda: value)
    monkeypatch.setattr(module, 'ProcessPoolExecutor', ColdPool)
    result = generate_outcomes(compiled, rules, policy, count=4, seed=1, workers=4,
        output=tmp_path/'cold', maximum_seconds=20)
    assert result['status'] == 'partial' and result['distinctAccepted'] == 0
    assert result['reproducible'] is False
    assert result['stages'].get('proposed', 0) == 0
    assert result['rejectReasons'] == {'worker_startup_budget': 4}
