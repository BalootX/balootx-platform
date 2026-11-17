import time

import pytest
from ortools.sat.python import cp_model

from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.enums import Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.ortools_scenario_model import OrToolsScenarioModelV1
from baloot_platform.research.outcome_generation import OutcomePolicyDriver, load_outcome_policy
from baloot_platform.research.outcome_spec import compile_outcome_spec


@pytest.fixture
def context(root):
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v7.yaml')
    policy = load_outcome_policy(root / 'configs/research/outcome_policy_v2.yaml')
    raw = dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'])
    return rules, policy, raw


@pytest.mark.real_clock
@pytest.mark.parametrize('fixed', [dict(mode='sun'), dict(trumpSuit='hearts'), dict(buyer='north'),
    dict(team='north_south')])
def test_spec_fixed_contract_never_consults_the_informed_bidder(context, monkeypatch, fixed):
    import baloot_platform.research.outcome_generation as module
    rules, policy, raw = context
    compiled = compile_outcome_spec({**raw, **fixed}, rules)
    driver = OutcomePolicyDriver(rules, compiled, policy, seed=92, deadline=time.monotonic()+10)
    state = driver.engine.create_round(seed=92, dealer=Seat.EAST)
    driver.engine.shuffle_and_initial_deal(state)
    machine = BiddingMachine(state.dealer, state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(rules.data), state.purchase_card.rank)
    def forbidden(*args, **kwargs):
        raise AssertionError('spec-fixed auction is not a strategic bid request')
    monkeypatch.setattr(module, '_informed_bid', forbidden)
    choice, scope, evidence, guarantee = driver.choose_bid(state, machine, dict(actor='north', action='sun'))
    machine.submit(choice.action, trump=choice.trump_suit)  # Engine legality authority.
    driver.record(Seat.NORTH, 'bid', scope, choice.to_dict(), evidence, guarantee=guarantee)
    assert driver.receipts[-1]['bid_policy'] == 'spec_fixed'
    assert driver.receipts[-1]['evidence']['legalityOnly'] is True


@pytest.mark.real_clock
def test_open_contract_refines_the_same_native_auction_model(context, monkeypatch):
    import baloot_platform.research.outcome_generation as module
    rules, policy, raw = context
    compiled = compile_outcome_spec(raw, rules)
    assert not any(p.kind == 'auction.contract' for p in compiled.scenario.predicates())
    driver = OutcomePolicyDriver(rules, compiled, policy, seed=92, deadline=time.monotonic()+20)
    assert driver.spec_fixed_contract is False
    model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    auction = model.add_native_auction_fragment()
    initial = model.model.new_bool_var('initial_pass_proposal')
    model.model.add(auction.choices[0]['pass'] == 1).only_enforce_if(initial)
    model.model.add_assumption(initial)
    solver = cp_model.CpSolver()
    attempt = model.solve(seed=92, maximum_seconds=5, solver=solver)
    assert attempt.status == 'satisfied'
    assert attempt.auction.to_dict()['completion']['actions'][0]['action'] == 'pass'
    model.model.clear_assumptions()
    model.model.add(initial == 0)
    for variable, card in zip(model.card_at_position, attempt.deck_order, strict=True):
        model.model.add(variable == card)
    original_model = model.model
    observed = []
    def bidder(scope, machine, *args):
        assert set(scope) == {'rulesHash', 'observer', 'observerHand', 'dealer', 'purchaseCard', 'matchScore', 'bids'}
        observed.append(scope)
        return next(choice for choice in module.legal_bid_choices(machine) if choice.action == 'sun'), {}
    monkeypatch.setattr(module, '_informed_bid', bidder)
    result = driver.refine_auction(model, attempt, solver, lambda: driver.remaining(), 250000)
    assert model.model is original_model and result.status == 'satisfied'
    assert result.auction.completion_contract.value == 'sun'
    assert driver.auction_proposals == 2 and len(driver.auction_cuts) == 1
    assert driver.auction_cuts[0]['binding']['status'] == 'bound'
    assert all(row['bid_policy'] == 'informed' for row in driver.receipts)
    assert observed


def test_buyer_constraint_is_lowered_and_invalid_seat_rejected(context):
    rules, _, raw = context
    compiled = compile_outcome_spec({**raw, 'mode': 'sun', 'buyer': 'west'}, rules)
    assert any(p.kind == 'auction.buyer' and p.arguments['seat'] == 'west' for p in compiled.scenario.predicates())
    with pytest.raises(ValueError, match='buyer must name a seat'):
        compile_outcome_spec({**raw, 'buyer': 'unknown'}, rules)
    with pytest.raises(ValueError, match='explicit mode'):
        compile_outcome_spec({**raw, 'declarations': [dict(category='sira')]}, rules)


@pytest.mark.real_clock
def test_informed_bid_seed_depends_on_observation_and_profile_not_construction(context, monkeypatch):
    import baloot_platform.research.outcome_generation as module
    rules, policy, raw = context
    driver = OutcomePolicyDriver(rules, compile_outcome_spec(raw, rules), policy,
        seed=1, deadline=time.monotonic()+10)
    state = driver.engine.create_round(seed=92, dealer=Seat.EAST)
    driver.engine.shuffle_and_initial_deal(state)
    machine = BiddingMachine(state.dealer, state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(rules.data), state.purchase_card.rank)
    seeds = []
    def bidder(observation, machine, rules, settings, seed, stopped):
        seeds.append(seed)
        return module.legal_bid_choices(machine)[0], {}
    monkeypatch.setattr(module, '_informed_bid', bidder)
    first = driver.choose_bid(state, machine, {})
    driver.seed = 123456
    driver.bid_cache.clear()
    assert driver.choose_bid(state, machine, {}) == first
    assert seeds[0] == seeds[1]
    driver.policy['cardPolicy']['samplingSeed'] += 1
    driver.choose_bid(state, machine, {})
    assert seeds[2] != seeds[1]
