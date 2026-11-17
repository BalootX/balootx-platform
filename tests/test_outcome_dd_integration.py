"""Small independent exact oracles and a real sampled-policy construction."""
import time

import pytest

from baloot_platform.analysis.double_dummy import DoubleDummySolver
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
from baloot_platform.research.ortools_research_executor import GenericCompleteTraceCompilerV1
from baloot_platform.research.outcome_generation import OutcomePolicyDriver, _prefix, load_outcome_policy
from baloot_platform.research.outcome_spec import compile_outcome_spec
from baloot_platform.research.scenario_contracts import ResearchScenarioV1
from scripts.validate_outcome_dd import cp_sat_oracle


def position(root, mode, seed, index=24):
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v7.yaml')
    from baloot_platform.core.engine import BalootEngine
    engine = BalootEngine(rules)
    state = engine.create_round(seed=seed, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state)
    auction = BiddingMachine(state.dealer, state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(rules.data), state.purchase_card.rank)
    while not auction.finished:
        action = mode.value if not auction.records else 'pass'
        auction.submit(action, trump=state.purchase_card.suit if action == 'hokum' else None)
    assert engine.complete_deal_from_bid_result(state, auction.result)
    bids = [dict(actor=r.seat.name.lower(), action=r.action,
        **({'trumpSuit': r.trump.name.lower()} if r.trump is not None else {})) for r in state.bid_history]
    engine.prepare_projects(state)
    state._project_declaration_state = engine.project_domain.declaration_state(state.projects, dealer=state.dealer)
    bot, plays, declarations = HeuristicBot(engine, BotProfile(seed)), [], []
    for _ in range(index):
        if not state.completed_tricks:
            action, = engine.project_domain.legal_declaration_actions(
                state._project_declaration_state, trump_suit=state.trump_suit)
            engine.apply_project_declaration_action(state, action)
            declarations.append(action.to_dict())
        played = engine.play_card(state, bot.choose(state))
        plays.append(dict(actor=played.seat.name.lower(), card=played.card.code, ekka=played.ekka))
    prefix = _prefix(state.shuffle.deck_order, 'east', bids, plays=plays, project_actions=declarations)
    return rules, engine, state, prefix


FIXED_POSITIONS = [(mode, 20261025+i, index)
    for mode, indices in ((Contract.SUN, [27, 25, 26, 24, 25, 27, 26, 24, 27, 26]),
                          (Contract.HOKUM, [26, 24, 24, 27, 25, 25, 26, 27, 25, 26]))
    for i, index in enumerate(indices)]


@pytest.mark.parametrize('mode,seed,index', FIXED_POSITIONS)
def test_fixed_dd_values_match_the_independent_cp_sat_authority(root, mode, seed, index):
    rules, engine, state, prefix = position(root, mode, seed, index)
    # The CP-SAT authority certifies forced moves without assigning their value.
    assert len(engine.legal_play_actions(state)) > 1
    policy = load_outcome_policy(root / 'configs/research/outcome_policy_v2.yaml')
    dd = DoubleDummySolver(rules)
    try:
        actual = dd.solve(state, all_root_values=True)
        assert actual['status'] == 'exact'
        oracle = cp_sat_oracle(rules, OrToolsExactStateNormalizerV1(rules, native_cp_sat=True),
            prefix, state.current_player, policy)
        assert {r['action']: r['value'] for r in actual['rootActions']} == {
            r['action']: r['value'] for r in oracle['proof']['rootActions']}
    finally:
        dd.close()


def test_multi_trick_hokum_matches_unpruned_engine_minimax(root):
    rules, engine, state, _ = position(root, Contract.HOKUM, 20261025)
    actor, visited = state.current_player, [0]
    def brute(current):
        visited[0] += 1
        assert visited[0] < 25000, 'fixed brute-force fixture grew unexpectedly'
        if current.round_score is not None:
            return current.round_score.awarded_points[0]-current.round_score.awarded_points[1]
        values = []
        for action in engine.legal_play_actions(current):
            branch = current.clone()
            engine.play_card(branch, action.card, ekka=action.ekka)
            values.append(brute(branch))
        return (max if int(current.current_player.team) == 0 else min)(values)
    expected = {}
    for action in engine.legal_play_actions(state):
        branch = state.clone()
        engine.play_card(branch, action.card, ekka=action.ekka)
        expected[action.code] = brute(branch)*(1 if int(actor.team) == 0 else -1)
    dd = DoubleDummySolver(rules)
    try:
        result = dd.solve(state, all_root_values=True)
        assert result['status'] == 'exact' and visited[0] > 8
        assert {row['action']: row['value'] for row in result['rootActions']} == expected
    finally:
        dd.close()


@pytest.mark.real_clock
@pytest.mark.parametrize('threads', [1, 4])
def test_native_cohort_matches_serial_exact_values_and_keeps_input_order(root, threads):
    rows = [position(root, mode, seed, index) for mode, seed, index in FIXED_POSITIONS]
    dd = DoubleDummySolver(rows[0][0])
    try:
        states = [row[2] for row in rows]
        expected = [dd.solve(state, all_root_values=True) for state in states]
        assert all(row['status'] == 'exact' for row in expected)
        dd.close()  # The comparison must not depend on pre-warmed serial facts.
        order = list(range(len(states)))+list(reversed(range(0, len(states), 3)))
        cohort = [states[i] for i in order]
        roots = [expected[i]['rootActions'] for i in order]
        actual = dd.solve_many(cohort, maximum_threads=threads)
        assert actual['status'] == 'exact' and actual['maximumThreads'] == threads
        assert [r['rootActions'] for r in actual['worlds']] == roots
        warm = dd.solve_many(cohort, maximum_threads=threads)
        assert warm['status'] == 'exact' and warm['nativeCache']['hits'] > 0
        assert [r['rootActions'] for r in warm['worlds']] == [r['rootActions'] for r in actual['worlds']]
        incomplete = dd.solve_many(states, maximum_threads=threads, maximum_nodes=1)
        assert incomplete['status'] == 'incomplete'
        assert all(not world['rootActions'] for world in incomplete['worlds'])
        cancelled = dd.solve_many(states, maximum_threads=threads, cancel_check=lambda: True)
        assert cancelled['status'] == 'incomplete' and cancelled['worlds'] == []
        with pytest.raises(ValueError, match='bounded'):
            dd.solve_many(states, maximum_threads=5)
    finally:
        dd.close()


@pytest.mark.real_clock
@pytest.mark.parametrize('category', ['kaboot', 'near_kaboot'])
def test_real_pimc_and_native_forward_refinement_on_a_fixed_sweep(root, category):
    from baloot_platform.research.outcome_rollout import verify_policy_line
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v7.yaml')
    policy = load_outcome_policy(root / 'configs/research/outcome_policy_v2.yaml')
    compiled = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'],
        mode='sun', category=category,
        **(dict(objective='raw_point_margin') if category == 'near_kaboot' else {}),
        policies=dict(default='optimal', **(dict(seats=dict(north='informed')) if category == 'kaboot' else {}))), rules)
    raw = compiled.scenario.to_dict()
    raw.pop('sha256')
    raw['constraints'].pop('sha256', None)
    for card in range(32):
        raw['constraints']['children'].append(dict(predicateId=f'PRED-FIXED-{card:03d}', version=1,
            hardness='hard', kind='deal.card_owner', arguments=dict(card=card,
                seat=Seat((1 if card == 7 else 0 if card == 15 else card//8)
                    if category == 'near_kaboot' else card//8).name.lower(), handStage='post_purchase')))
    driver = OutcomePolicyDriver(rules, compiled, policy, seed=20261025,
        deadline=time.monotonic()+40, refinement='forward_rollout')
    try:
        result = GenericCompleteTraceCompilerV1(rules=rules, native_cp_sat_scoring=True,
            play_model_configurator=driver.bind, play_model_refiner=driver.refine)(
                ResearchScenarioV1.build(raw), (), seed=20261025, maximum_seconds=40)
        assert result.candidate is not None, result.reason
        assert driver.proposals >= 1
        assert sum(int(trick.winner.team) == 0 for trick in driver.final_state.completed_tricks) == (
            8 if category == 'kaboot' else 7)
        samples = [r for r in driver.receipts if r['evidence'].get('method') == 'sampled_pimc_dd_v1']
        if category == 'kaboot':
            assert samples and samples[0]['evidence']['sampledWorlds'] == 32
        assert all(r['guarantee'] == 'sampled_policy_not_exact' for r in samples)
        assert result.candidate.trace['score']['gamePoints'] == list(driver.final_state.round_score.awarded_points)
        plays = [dict(actor=row['actor'], card=row['card'], ekka=row.get('ekka', False))
            for row in result.candidate.trace['play']['plays']]
        okay, evidence = verify_policy_line(compiled, driver, result.candidate.trace, plays)
        assert okay
        if category == 'near_kaboot':
            assert evidence['satisfaction']['constraintsSatisfied']
            assert not evidence['satisfaction']['hardPredicatesSatisfied']  # Count 7/8, not all eight.
    finally:
        driver.close()
