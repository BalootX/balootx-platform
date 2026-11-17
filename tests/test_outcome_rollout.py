import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

from baloot_platform.analysis.models import BidChoice
from baloot_platform.core.enums import Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.native.accelerator import NativeAccelerator, NativeUnavailable
from baloot_platform.research.outcome_generation import OutcomePolicyDriver, OutcomePolicyIncomplete, load_outcome_policy
from baloot_platform.research.outcome_rollout import DealProposalStream, generate_outcomes, verify_policy_line
from baloot_platform.research.outcome_spec import compile_outcome_spec


@pytest.fixture
def setup(root):
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v7.yaml')
    policy = load_outcome_policy(root / 'configs/research/outcome_policy_v3.yaml')
    compiled = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'],
        mode='sun', category='big_swing', policies=dict(default='optimal')), rules)
    return rules, policy, compiled


@pytest.fixture
def open_proposal(setup, monkeypatch):
    import baloot_platform.research.outcome_generation as generation
    rules, policy, _ = setup
    bids = [dict(actor='west', action='sun')]
    proposal = SimpleNamespace(status='satisfied', deck_order=tuple(range(32)),
        auction=SimpleNamespace(to_dict=lambda: dict(completion=dict(actions=bids))),
        solution_hash='0'*64, model_hash='1'*64)
    def propose(self, maximum_seconds):
        self.model = SimpleNamespace(dealer=Seat.NORTH)
        return proposal
    def card(driver, state, info, policy):
        return driver.engine.legal_play_actions(state)[0], {'testOnly': True}, 'exact_complete_state', info.to_dict()
    monkeypatch.setattr(DealProposalStream, 'next', propose)
    monkeypatch.setattr(generation, '_informed_bid', lambda *args: (BidChoice('sun'), {'testOnly': True}))
    monkeypatch.setattr(OutcomePolicyDriver, '_informed_escalation', lambda driver, state, actor, choices:
        (driver.engine.escalation_domain.closure_action, {'observer': actor.name.lower()}, {'testOnly': True}))
    monkeypatch.setattr(OutcomePolicyDriver, 'dd_card', card)
    return rules, policy, bids


@pytest.mark.real_clock
@pytest.mark.parametrize('strategy', ['guided', 'rollout'])
@pytest.mark.parametrize('clauses', [dict(category='big_swing'), dict(outcome=dict(winner='north_south'))])
def test_open_auction_batch_replays_each_informed_bid_once(open_proposal, tmp_path, strategy, clauses):
    rules, policy, bids = open_proposal
    compiled = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'],
        policies=dict(default='optimal'), **clauses), rules)
    result = generate_outcomes(compiled, rules, policy, count=1, seed=1,
        output=tmp_path/'open', workers=1, maximum_seconds=15, strategy=strategy)
    assert result['status'] == 'complete' and result['distinctAccepted'] == 1
    assert result['rejectReasons'] == {} and result['reproducible']
    record = json.loads((tmp_path/'open/catalogue.json').read_text(encoding='utf-8'))
    scenario = json.loads(Path(record['accepted'][0]['path']).read_text(encoding='utf-8'))
    receipts = scenario['decisionReceipts']
    assert [r['action'] for r in receipts if r['decisionKind'] == 'bid'] == bids
    assert all(r['policy'] == 'informed' for r in receipts if r['decisionKind'] == 'bid')
    assert sum(r['decisionKind'] == 'play' for r in receipts) == 32
    assert sum(r['decisionKind'] == 'project' for r in receipts) == 4
    assert scenario['exactTracePrefix']['auctionActions'] == bids
    assert scenario['score']['winner'] == 'north_south'
    assert scenario['satisfaction']['status'] == 'satisfied'


@pytest.mark.real_clock
@pytest.mark.parametrize('error_type', [ValueError, AssertionError, RuntimeError])
def test_unexpected_game_error_preserves_partial_batch_and_stops_worker(open_proposal, tmp_path, monkeypatch, error_type):
    rules, policy, _ = open_proposal
    compiled = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'],
        policies=dict(default='optimal')), rules)
    original, calls = DealProposalStream.next, []
    def fail_second(self, maximum_seconds):
        calls.append(1)
        if len(calls) == 2:
            raise error_type('synthetic per-game failure')
        return original(self, maximum_seconds)
    monkeypatch.setattr(DealProposalStream, 'next', fail_second)
    result = generate_outcomes(compiled, rules, policy, count=2, seed=1,
        output=tmp_path/'partial', workers=1, maximum_seconds=15)
    reason = 'unexpected_game_error:'+error_type.__name__
    assert result['status'] == 'partial' and result['distinctAccepted'] == 1
    assert result['rejectReasons'] == {reason: 1} and not result['reproducible']
    assert len(result['plays']) == 1 and len(calls) == 2
    worker = json.loads((tmp_path/'partial/worker-0.json').read_text(encoding='utf-8'))
    assert worker['budgetCensored'] and not worker['quotaCompleted']
    assert len(worker['attempts']) == 2
    error = worker['attempts'][1]['error']
    assert error['type'] == error_type.__name__ and error['message'] == 'synthetic per-game failure'
    assert 'fail_second' in error['traceback']


@pytest.mark.parametrize('error_type', [OSError, MemoryError])
def test_resource_errors_still_abort_the_batch(open_proposal, tmp_path, monkeypatch, error_type):
    rules, policy, _ = open_proposal
    compiled = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data['profile']['id']), rules)
    def fail(self, maximum_seconds):
        raise error_type('synthetic resource failure')
    monkeypatch.setattr(DealProposalStream, 'next', fail)
    with pytest.raises(error_type, match='synthetic resource failure'):
        generate_outcomes(compiled, rules, policy, count=1, seed=1,
            output=tmp_path/'resource', workers=1, maximum_seconds=15)


@pytest.mark.parametrize('error_type', [NativeUnavailable, ImportError, ModuleNotFoundError])
def test_native_loader_errors_propagate(setup, tmp_path, monkeypatch, error_type):
    rules, policy, compiled = setup
    error = error_type('synthetic native loader failure')
    def unavailable(library):
        raise error
    monkeypatch.setattr(NativeAccelerator, '_locate_library', staticmethod(unavailable))
    with pytest.raises(error_type) as raised:
        generate_outcomes(compiled, rules, policy, count=1, seed=1,
            output=tmp_path/'unavailable', workers=1, maximum_seconds=15)
    assert raised.value is error
    assert not (tmp_path/'unavailable/catalogue.json').exists()


@pytest.mark.real_clock
@pytest.mark.parametrize('error_type', [None, ValueError, OutcomePolicyIncomplete])
def test_cli_exit_status_and_summary_preserve_written_results(open_proposal, tmp_path, monkeypatch, capsys, root, error_type):
    from baloot_platform.research import outcome_cli
    rules, _, _ = open_proposal
    source = dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'], policies=dict(default='optimal'))
    spec = tmp_path/'spec.yaml'
    spec.write_text(json.dumps(source), encoding='utf-8')
    original, calls = DealProposalStream.next, []
    def fail_second(self, maximum_seconds):
        calls.append(1)
        if len(calls) == 2:
            raise error_type('proposal:infeasible' if error_type is OutcomePolicyIncomplete else 'synthetic CLI failure')
        return original(self, maximum_seconds)
    monkeypatch.setattr(DealProposalStream, 'next', fail_second)
    monkeypatch.setattr(outcome_cli, 'platform_root', lambda: root)
    output = tmp_path/'cli'
    args = ['--spec', str(spec), '--count', '2' if error_type else '1', '--workers', '1',
        '--seconds', '15', '--strategy', 'rollout', '--out', str(output)]
    if error_type is ValueError:
        with pytest.raises(SystemExit) as exited:
            outcome_cli.main(args)
        assert exited.value.code == 1
    else:
        assert outcome_cli.main(args) is None
    summary = json.loads(capsys.readouterr().out)
    catalogue = json.loads((output/'catalogue.json').read_text(encoding='utf-8'))
    reasons = {} if error_type is None else {
        'proposal:infeasible' if error_type is OutcomePolicyIncomplete else 'unexpected_game_error:ValueError': 1}
    assert summary['rejectReasons'] == catalogue['rejectReasons'] == reasons
    assert summary['distinctAccepted'] == len(catalogue['accepted']) == 1
    assert summary['status'] == ('complete' if error_type is None else 'partial')
    assert (output/'catalogue.csv').exists() and (output/'summaries.md').exists()
    assert (output/'worker-0.json').exists() and Path(catalogue['accepted'][0]['path']).exists()


@pytest.mark.real_clock
def test_verification_unknown_is_fail_closed_and_budget_censored(open_proposal, tmp_path, monkeypatch):
    import baloot_platform.research.outcome_rollout as module
    rules, policy, _ = open_proposal
    compiled = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'],
        policies=dict(default='optimal')), rules)
    original, real_clock, calls = module._satisfaction, time.monotonic, []
    def censored(compiled, normalized):
        def unknown(solver, model):
            assert solver.parameters.num_search_workers == 1
            assert solver.parameters.max_deterministic_time == 1
            assert solver.parameters.max_time_in_seconds == float('inf')
            calls.append(1)
            monkeypatch.setattr(module.time, 'monotonic', lambda: real_clock()+1000)
            return cp_model.UNKNOWN
        monkeypatch.setattr(cp_model.CpSolver, 'solve', unknown)
        return original(compiled, normalized)
    monkeypatch.setattr(module, '_satisfaction', censored)
    result = generate_outcomes(compiled, rules, policy, count=1, seed=1,
        output=tmp_path/'censored', workers=1, maximum_seconds=15)
    assert calls == [1] and result['status'] == 'partial' and not result['accepted']
    assert result['rejectReasons'] == {'verification_budget': 1} and not result['reproducible']
    worker = json.loads((tmp_path/'censored/worker-0.json').read_text(encoding='utf-8'))
    assert worker['budgetCensored'] and not worker['quotaCompleted']


@pytest.mark.real_clock
@pytest.mark.parametrize('stage', ['policy', 'verification', 'admission'])
def test_game_errors_are_recorded_but_admission_errors_still_raise(open_proposal, tmp_path, monkeypatch, stage):
    import baloot_platform.research.outcome_rollout as module
    rules, policy, _ = open_proposal
    compiled = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'],
        policies=dict(default='optimal')), rules)
    def fail(*args):
        raise ValueError('synthetic '+stage+' failure')
    target, name = (OutcomePolicyDriver, 'construct_line') if stage == 'policy' else (
        module, 'verify_policy_line' if stage == 'verification' else 'story_facts')
    monkeypatch.setattr(target, name, fail)
    def run():
        return generate_outcomes(compiled, rules, policy, count=1, seed=1,
            output=tmp_path/'failed', workers=1, maximum_seconds=15)
    if stage == 'admission':
        with pytest.raises(ValueError, match='synthetic admission failure'):
            run()
    else:
        result = run()
        assert result['status'] == 'partial' and not result['accepted'] and not result['reproducible']
        assert result['rejectReasons'] == {'unexpected_game_error:ValueError': 1}
        worker = json.loads((tmp_path/'failed/worker-0.json').read_text(encoding='utf-8'))
        assert len(worker['attempts']) == 1 and worker['attempts'][0]['stage'] == stage
        assert worker['attempts'][0]['error']['message'] == 'synthetic '+stage+' failure'


@pytest.mark.real_clock
def test_stream_enumerates_on_the_same_native_model_and_solver(setup):
    rules, policy, compiled = setup
    stream = DealProposalStream(compiled, rules, policy, 20261025, time.monotonic()+20)
    first = stream.next(10)
    model, solver = stream.model, stream.solver
    second = stream.next(10)
    assert first.status == second.status == 'satisfied'
    assert stream.model is model and stream.solver is solver
    assert sum(a != b for a, b in zip(first.deck_order, second.deck_order, strict=True)) >= 4
    assert stream.model.canonicalize_deal_packets
    assert first.model_hash != second.model_hash


@pytest.mark.real_clock
def test_stream_retries_unknown_without_cutting_an_unknown_or_repeating_the_cut(setup, monkeypatch):
    rules, policy, compiled = setup
    stream = DealProposalStream(compiled, rules, policy, 20261025, time.monotonic()+20)
    first = stream.next(10)
    solve, calls = stream.model.solve, []
    def once_unknown(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return replace(first, status='not_found_within_budget', deck_order=None)
        return solve(**kwargs)
    monkeypatch.setattr(stream.model, 'solve', once_unknown)
    with pytest.raises(OutcomePolicyIncomplete, match='proposal:not_found_within_budget'):
        stream.next(10)
    constraints = len(stream.model.model.proto.constraints)
    assert stream.last is None
    second = stream.next(10)
    assert second.status == 'satisfied' and stream.last is second
    assert len(stream.model.model.proto.constraints) == constraints
    assert sum(a != b for a, b in zip(first.deck_order, second.deck_order, strict=True)) >= 4


@pytest.mark.real_clock
def test_rollout_default_has_real_replay_and_explains_partial_yield(setup, tmp_path):
    rules, policy, compiled = setup
    result = generate_outcomes(compiled, rules, policy, count=1, seed=20261025,
        output=tmp_path/'run', workers=1, maximum_seconds=15)
    assert result['strategy'] == 'guided'
    assert result['minimumDeckHammingDistance'] == 4
    assert result['deckDistanceScope'] == 'merged_batch'
    assert result['stages']['proposed'] > 0 and result['stages']['policyCompleted'] > 0
    assert result['stages']['proposed'] == sum(result['stages'].get(key, 0)
        for key in ('cpSatProposed', 'guidedProposed', 'neighbourProposed'))
    assert 'cp_satProposed' not in result['stages']
    assert result['stages']['verified'] == result['stages']['policyCompleted']
    assert len(result['accepted']) == result['distinctAccepted']
    assert sum(result['rejectReasons'].values())+result['stages'].get('distinctAccepted', 0) > 0
    assert (tmp_path/'run/catalogue.json').exists()
    if result['distinctAccepted']:
        assert result['cpuSecondsPerAccepted'] > 0
    else:
        assert result['status'] == 'partial' and result['cpuSecondsPerAccepted'] is None
    with pytest.raises(FileExistsError):
        generate_outcomes(compiled, rules, policy, count=1, seed=1, output=tmp_path/'run', workers=1)


@pytest.mark.real_clock
def test_actual_policy_line_cannot_bypass_the_requested_outcome(setup):
    rules, policy, compiled = setup
    stream = DealProposalStream(compiled, rules, policy, 20261025, time.monotonic()+20)
    proposal = stream.next(10)
    driver = OutcomePolicyDriver(rules, compiled, policy, seed=1, deadline=time.monotonic()+10)
    context = dict(deckOrder=list(proposal.deck_order), dealer='east', auction=proposal.auction.to_dict(),
        escalation=dict(observation=dict(actions=[])))
    try:
        plays = driver.construct_line(context)
        _, evidence = verify_policy_line(compiled, driver, context, plays)
        assert len(evidence['exactTracePrefix']['playActions']) == 32
        assert evidence['score'] == driver.final_state.round_score.to_dict()
        last = driver.final_state.completed_tricks[-1]
        expected_swing = (last.raw_points+driver.engine.scoring_domain.last_trick_bonus_raw)*(
            1 if int(last.winner.team) == 0 else -1)
        assert evidence['trajectory'][0]['value'] == expected_swing
        winning = next(play for play in last.plays if play.seat == last.winner)
        teammate = next(play for play in last.plays if play.seat.team == last.winner.team and play.seat != last.winner)
        beaten = next(play for play in last.plays if play.seat.team != last.winner.team)
        named = dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'], mode='sun',
            team=last.winner.team.name.lower(), outcome=dict(decidingTrick=dict(index=8,
                winningTeam=last.winner.team.name.lower(), winningCard=winning.card.code, beatenCard=beaten.card.code)))
        assert verify_policy_line(compile_outcome_spec(named, rules), driver, context, plays)[0]
        named['outcome']['decidingTrick']['winningCard'] = teammate.card.code
        assert not verify_policy_line(compile_outcome_spec(named, rules), driver, context, plays)[0]
        lost = driver.final_state.round_score.winner.other.name.lower()
        impossible = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'],
            mode='sun', outcome=dict(winner=lost)), rules)
        assert verify_policy_line(impossible, driver, context, plays)[0] is False
    finally:
        driver.close()


def test_worker_count_and_time_limits_fail_before_output_creation(setup, tmp_path):
    rules, policy, compiled = setup
    for overrides in (dict(workers=5), dict(maximum_seconds=300), dict(count=0), dict(strategy='approximate'),
            dict(story_settings={'scoreBucket': 0}), dict(story_settings={'openingTricks': 4}),
            dict(story_settings={'swingBucket': True}), dict(story_settings={'unknown': 1})):
        arguments = dict(count=1, seed=1, workers=1, maximum_seconds=10, output=tmp_path/'absent')
        arguments.update(overrides)
        with pytest.raises(ValueError):
            generate_outcomes(compiled, rules, policy, **arguments)
        assert not (tmp_path/'absent').exists()


@pytest.mark.real_clock
def test_completed_line_after_campaign_deadline_is_not_admitted(setup, tmp_path, monkeypatch):
    import baloot_platform.research.outcome_rollout as module
    rules, policy, _ = setup
    compiled = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'],
        mode='sun', policies=dict(default='optimal')), rules)
    original, real_clock = module.verify_policy_line, time.monotonic
    def late(*args):
        result = original(*args)
        assert result[0]
        monkeypatch.setattr(module.time, 'monotonic', lambda: real_clock()+1000)
        return result
    monkeypatch.setattr(module, 'verify_policy_line', late)
    result = generate_outcomes(compiled, rules, policy, count=1, seed=20261025,
        output=tmp_path/'late', workers=1, maximum_seconds=15)
    assert not result['accepted'] and result['status'] == 'partial'
    assert result['rejectReasons'] == {'campaign_time_budget': 1}


@pytest.mark.real_clock
def test_proposal_failure_does_not_reuse_previous_deal_decisions(setup, tmp_path, monkeypatch):
    import json
    import baloot_platform.research.outcome_rollout as module
    rules, policy, _ = setup
    compiled = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'],
        mode='sun', policies=dict(default='optimal')), rules)
    original, real_clock, calls = DealProposalStream.next, time.monotonic, []
    def fail_second(self, maximum_seconds):
        calls.append(1)
        if len(calls) == 2:
            monkeypatch.setattr(module.time, 'monotonic', lambda: real_clock()+1000)
            raise OutcomePolicyIncomplete('proposal:not_found_within_budget')
        return original(self, maximum_seconds)
    monkeypatch.setattr(DealProposalStream, 'next', fail_second)
    result = generate_outcomes(compiled, rules, policy, count=2, seed=20261025,
        output=tmp_path/'retry', workers=1, maximum_seconds=15, strategy='rollout')
    receipt = json.loads((tmp_path/'retry/worker-0.json').read_text(encoding='utf-8'))
    assert result['distinctAccepted'] == 1
    assert receipt['attempts'][0]['completedCardDecisions'] == 32
    assert receipt['attempts'][1]['stage'] == 'proposal'
    assert receipt['attempts'][1]['completedCardDecisions'] == 0
