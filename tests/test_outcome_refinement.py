import time
from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.ortools_play_tail import PlayTailSolveStopped
from baloot_platform.research.outcome_generation import (
    OutcomePolicyDeviation, OutcomePolicyDriver, OutcomePolicyIncomplete, construct_outcome, load_outcome_policy,
)
from baloot_platform.research.outcome_spec import compile_outcome_spec


@pytest.fixture
def driver(root):
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v7.yaml')
    policy = load_outcome_policy(root / 'configs/research/outcome_policy_v2.yaml')
    compiled = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'], mode='sun'), rules)
    return OutcomePolicyDriver(rules, compiled, policy, seed=1, deadline=time.monotonic()+30)


def _play(card, actor):
    return SimpleNamespace(card=Card(card), actor=Seat(actor), ekka=False,
        to_dict=lambda: dict(card=Card(card).code, actor=Seat(actor).name.lower(), ekka=False))


@pytest.mark.real_clock
def test_first_difference_is_a_conditional_cut_on_the_same_model_and_solver(driver, monkeypatch):
    model, solver = cp_model.CpModel(), cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    slots = [SimpleNamespace(card=model.new_int_var(0, 31, f'card{i}'),
        actor=model.new_int_var(0, 3, f'actor{i}'), ekka=model.new_bool_var(f'ekka{i}')) for i in range(2)]
    model.add(slots[0].actor == 0)
    model.add(slots[1].actor == 1)
    initial_prefix = model.new_bool_var('initial_prefix')
    model.add(slots[0].card == 0).only_enforce_if(initial_prefix)
    model.add_assumption(initial_prefix)
    model.add(slots[0].ekka == 0)
    build = SimpleNamespace(model=model, predicate_context=SimpleNamespace(tricks=[SimpleNamespace(actions=slots)]))
    initial = SimpleNamespace(plays=(_play(0, 0), _play(1, 1)))
    calls = []
    def policy(context, proposed):
        if len(calls) == 0:
            calls.append('deviation')
            raise OutcomePolicyDeviation(1, LegalPlayActionV1(Card(2)), 'a'*64)
        assert proposed[0].card.id == 0 and proposed[1].card.id == 2
        calls.append('accepted')
        driver.final_state = object()
    def solve(**kwargs):
        assert kwargs['solver'] is solver and build.model is model
        assert solver.solve(model) in {cp_model.OPTIMAL, cp_model.FEASIBLE}
        return SimpleNamespace(plays=tuple(_play(solver.value(s.card), solver.value(s.actor)) for s in slots))
    build.solve = solve
    monkeypatch.setattr(driver, 'construct_line', policy)
    result = driver.refine(build, {}, initial, solver, lambda: 20, 250000)
    assert calls == ['deviation', 'accepted'] and result.plays[1].card.id == 2
    assert driver.proposals == 2 and len(driver.cuts) == 1
    assert driver.cuts[0]['inputScopeHash'] == 'a'*64
    # The cut is not an unconditional card ban at this timeline index.
    other = model.clone()
    other.clear_hints()
    other.clear_assumptions()
    other.add(initial_prefix == 0)
    other.add(slots[0].card == 3)
    other.add(slots[1].card == 1)
    assert solver.solve(other) in {cp_model.OPTIMAL, cp_model.FEASIBLE}


@pytest.mark.real_clock
@pytest.mark.parametrize('reason', ['dd_optimal_incomplete', 'auction_policy_incompatible'])
def test_unknown_policy_evidence_is_not_a_deal_no_good(driver, monkeypatch, reason):
    import baloot_platform.research.outcome_generation as module
    calls = []
    class Compiler:
        def __init__(self, **kwargs):
            self.owner = kwargs['play_model_refiner'].__self__
        def __call__(self, scenario, excluded, **kwargs):
            calls.append(excluded)
            self.owner.deck = tuple(range(32))
            raise OutcomePolicyIncomplete(reason)
    monkeypatch.setattr(module, 'GenericCompleteTraceCompilerV1', Compiler)
    result = construct_outcome(driver.compiled, driver.rules, driver.policy, seed=1,
        maximum_attempts=3, maximum_seconds=10)
    assert calls == [(), (), ()] and result['rejectReasons'] == {reason: 3}
    assert [row['seed'] for row in result['attempts']] == [1, 2, 3]
    assert result['accepted'] is None


@pytest.mark.real_clock
def test_proved_unsat_policy_cuts_allow_a_new_deal(driver, monkeypatch):
    import baloot_platform.research.outcome_generation as module
    calls = []
    class Compiler:
        def __init__(self, **kwargs):
            self.owner = kwargs['play_model_refiner'].__self__
        def __call__(self, scenario, excluded, **kwargs):
            calls.append(excluded)
            self.owner.deck = tuple(range(32))
            return SimpleNamespace(candidate=None, reason='fixed_deal_play_infeasible', status='incomplete',
                partial_receipt={'exhaustionReason': 'infeasible'})
    monkeypatch.setattr(module, 'GenericCompleteTraceCompilerV1', Compiler)
    result = construct_outcome(driver.compiled, driver.rules, driver.policy, seed=1,
        maximum_attempts=2, maximum_seconds=10)
    assert len(calls) == 2 and calls[1] == (tuple(range(32)),)
    assert result['accepted'] is None


@pytest.mark.real_clock
def test_refiner_propagates_unsat_without_selecting_a_scripted_card(driver, monkeypatch):
    model = cp_model.CpModel()
    slots = [SimpleNamespace(card=model.new_int_var(0, 31, 'card'), ekka=model.new_bool_var('ekka'))]
    build = SimpleNamespace(model=model, predicate_context=SimpleNamespace(tricks=[SimpleNamespace(actions=slots)]))
    monkeypatch.setattr(driver, 'construct_line', lambda *a: (_ for _ in ()).throw(
        OutcomePolicyDeviation(0, LegalPlayActionV1(Card(2)), 'a'*64)))
    build.solve = lambda **kw: (_ for _ in ()).throw(PlayTailSolveStopped('infeasible'))
    with pytest.raises(PlayTailSolveStopped):
        driver.refine(build, {}, SimpleNamespace(plays=(_play(1, 0),)), object(), lambda: 20, 250000)
    assert driver.final_state is None and driver.cuts[0]['index'] == 0


@pytest.mark.real_clock
def test_native_refinement_can_change_the_opening_and_rebind_project_scoring(driver):
    from baloot_platform.core.auction_ir import AuctionDomainIRV1
    from baloot_platform.core.bidding_machine import BiddingMachine
    from baloot_platform.research.ortools_research_executor import GenericCompleteTraceCompilerV1

    seen = []
    def refine(build, context, first, solver, remaining, maximum_conflicts):
        assert len(first.plays) == 32 and len(context['deckOrder']) == 32
        seen.append(first.trick_winners[0])
        build.model.add(build._extract.winner_seats[0] != int(first.trick_winners[0]))
        result = build.solve(solver=solver, maximum_time_seconds=min(8, remaining()),
            maximum_conflicts=maximum_conflicts)
        assert result.trick_winners[0] != first.trick_winners[0]
        return result
    result = GenericCompleteTraceCompilerV1(rules=driver.rules, native_cp_sat_scoring=True,
        play_model_configurator=lambda build, ctx: driver.compiled.constrain_play_model(build),
        play_model_refiner=refine)(driver.compiled.scenario, (), seed=20261026, maximum_seconds=15)
    assert seen and result.candidate is not None, result.reason
    trace, engine = result.candidate.trace, driver.engine
    state = engine.create_round(seed=1, dealer=Seat.parse(trace['dealer']))
    engine.initial_deal_from_preordered(state, trace['deckOrder'])
    auction = BiddingMachine(state.dealer, state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(driver.rules.data), state.purchase_card.rank)
    for row in trace['auction']['completion']['actions']:
        auction.submit(row['action'])
    assert engine.complete_deal_from_bid_result(state, auction.result)
    for row in trace['escalation']['observation']['actions']:
        engine.apply_escalation(state, seat=Seat.parse(row['actor']), action=row['action'], mode=row.get('requestedMode'))
    engine.prepare_projects(state)
    for row in trace['play']['plays']:
        engine.play_card(state, row['card'], ekka=row.get('ekka', False))
    assert state.completed_tricks[0].winner != seen[0]
    assert list(state.round_score.project_points) == trace['projects']['combinedAwardedValue']
    assert list(state.round_score.awarded_points) == trace['score']['gamePoints']


@pytest.mark.real_clock
@pytest.mark.parametrize('outcome', ['satisfied', 'infeasible', 'unknown', 'cancelled'])
def test_forward_rollout_is_checked_natively_and_keeps_the_original_assumptions(driver, monkeypatch, outcome):
    driver.refinement = 'forward_rollout'
    model, solver = cp_model.CpModel(), cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    slots = [SimpleNamespace(card=model.new_int_var(0, 31, f'card{i}'),
        actor=model.new_int_var(0, 3, f'actor{i}'), ekka=model.new_bool_var(f'ekka{i}')) for i in range(2)]
    for i, slot in enumerate(slots):
        model.add(slot.actor == i)
        model.add(slot.ekka == 0)
    model.add(slots[0].card == 0)
    if outcome == 'infeasible':
        model.add(slots[1].card == 1)
    existing = model.new_bool_var('existing_negative_assumption')
    model.add_assumption(existing.Not())
    saved = tuple(model.proto.assumptions)
    build = SimpleNamespace(model=model, predicate_context=SimpleNamespace(tricks=[SimpleNamespace(actions=slots)]))
    def policy(context, proposed, **kwargs):
        assert kwargs['continue_after_deviation'] is True
        driver.first_difference = OutcomePolicyDeviation(1, LegalPlayActionV1(Card(2)), 'b'*64)
        driver.final_state = object()
        return [dict(actor='north', card='S7', ekka=False), dict(actor='west', card='S9', ekka=False)]
    calls = []
    def solve(**kwargs):
        assert kwargs['solver'] is solver and build.model is model
        calls.append(len(model.proto.assumptions))
        if outcome == 'unknown':
            raise PlayTailSolveStopped('unknown')
        if outcome == 'cancelled':
            raise OutcomePolicyIncomplete('candidate_time_budget')
        status = solver.solve(model)
        if status == cp_model.INFEASIBLE:
            raise PlayTailSolveStopped('infeasible')
        assert status in {cp_model.OPTIMAL, cp_model.FEASIBLE}
        return SimpleNamespace(plays=tuple(_play(solver.value(s.card), solver.value(s.actor)) for s in slots))
    build.solve = solve
    monkeypatch.setattr(driver, 'construct_line', policy)
    proposed = SimpleNamespace(plays=(_play(0, 0), _play(1, 1)))
    if outcome == 'satisfied':
        result = driver.refine(build, {}, proposed, solver, lambda: 20, 250000)
        assert result.plays[1].card.id == 2 and driver.final_state is not None
        assert driver.cuts == [] and calls == [2]
    else:
        with pytest.raises(OutcomePolicyIncomplete if outcome == 'cancelled' else PlayTailSolveStopped) as stopped:
            driver.refine(build, {}, proposed, solver, lambda: 20, 250000)
        if outcome != 'cancelled':
            assert stopped.value.solver_status == outcome
        assert tuple(model.proto.assumptions) == saved
        assert len(driver.cuts) == (1 if outcome == 'infeasible' else 0)
        assert calls == ([2, 1] if outcome == 'infeasible' else [2])
    assert driver.forward_probes == 1 and driver.proposals == 1
