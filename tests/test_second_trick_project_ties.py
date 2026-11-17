from dataclasses import replace
import copy
import json

import jsonschema
import pytest
from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card, FULL_DECK, cards_to_mask
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, Seat, Team
from baloot_platform.core.models import Project
from baloot_platform.core.rule_source_catalog import load_exact_rule_source_catalog
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.ortools_project_fragment import attach_project_lifecycle_fragment_v1
from baloot_platform.research.ortools_project_model import OrToolsProjectLifecycleModelV1
from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
from baloot_platform.research.scenario_contracts import ExactTracePrefixV1
from test_core import _playing_state


@pytest.mark.parametrize('fixed,variable', [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_project_tie_conjunction_accepts_fixed_boolean_literals(fixed, variable):
    from baloot_platform.research.ortools_project_fragment import _and_literal
    model = cp_model.CpModel()
    declared = model.new_bool_var('declared')
    model.add(declared == variable)
    result = _and_literal(model, [declared, fixed], 'reveal_precedes')
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    assert solver.value(result) == (fixed and variable)


def _rules(root):
    return RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')


def test_v8_is_explicit_and_v7_stays_immutable(root):
    old = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v7.yaml')
    assert old.sha256 == '20abf50271d021d412ebb8f608a402aec558adc8974204578b8c2b17c0895805'
    assert old.project_domain.tie_break_order == 'declaration'
    assert 'tieBreakOrder' not in old.project_domain.to_dict()
    rules = _rules(root)
    assert rules.project_domain.to_dict()['tieBreakOrder'] == 'second_trick_reveal'
    jsonschema.validate(rules.data, json.loads((root / 'schemas/profiles/rules.schema.json').read_text(encoding='utf-8')))
    catalog = load_exact_rule_source_catalog(rules, root)
    assert catalog.binding_for('projects.tie_break_order')['ratification_status'] == 'expert_ratified'
    with pytest.raises(ValueError, match='second-trick leader'):
        rules.project_domain.score_team([])
    for relative in ('configs/rules/saudi_federation_2023_v8.yaml',
                     'configs/rules/source_catalog_saudi_federation_2023_v8.yaml'):
        assert (root / relative).read_bytes() == (root / 'src/baloot_platform/resources' / relative).read_bytes()


@pytest.mark.parametrize('leader', list(Seat))
def test_exact_tie_uses_reveal_rotation_not_declaration(root, leader):
    domain = _rules(root).project_domain
    projects = [Project(seat, 'siri', (), 4, 2, 0, True, int(seat) + 1) for seat in Seat]
    scored, winner = domain.score_team(projects, second_trick_leader=leader)
    assert winner == leader.team
    assert domain.comparison_winner(scored, second_trick_leader=leader).owner == leader
    stronger = [replace(p, high_rank=3) if p.owner == leader.previous() else p for p in projects]
    assert domain.comparison_winner(stronger, second_trick_leader=leader).owner == leader.previous()


@pytest.mark.parametrize('reveal_scope', ['scoring_team', 'comparison_winner'])
def test_runtime_actual_first_trick_winner_controls_tie_and_team_reveal(root, reveal_scope):
    payload = copy.deepcopy(_rules(root).data)
    payload['projects']['reveal_scope'] = reveal_scope
    rules = RuleBook(payload)
    engine = BalootEngine(rules)
    state = _playing_state(rules)
    hands = {seat: tuple(map(Card.parse, cards.split())) for seat, cards in (
        (Seat.NORTH, 'S7 S8 S9 HA SQ D10 C10 DA'),
        (Seat.WEST, 'H7 H8 H9 SA SK DJ CJ CA'),
        (Seat.SOUTH, 'D7 D8 D9 HK HQ DQ CQ H10'),
        (Seat.EAST, 'C7 C8 C9 SJ HJ DK CK S10'))}
    state.hands = [cards_to_mask(hands[seat]) for seat in Seat]
    state.initial_hands = list(state.hands)
    state.projects = rules.project_domain.detect(hands, Contract.SUN, Seat.EAST)
    for code in ('S7', 'SA', 'D7', 'SJ'):
        engine.play_card(state, Card.parse(code))
    assert state.first_leader == Seat.NORTH
    assert state.completed_tricks[0].winner == Seat.WEST
    assert state.project_winning_team == Team.WEST_EAST
    assert not any(p.revealed for p in state.projects)
    for _ in Seat:
        engine.play_card(state, engine.legal_moves(state, state.current_player)[0])
    expected = {Seat.WEST, Seat.EAST} if reveal_scope == 'scoring_team' else {Seat.WEST}
    assert {p.owner for p in state.projects if p.revealed} == expected
    assert all(p.to_dict()['cards'] is None for p in state.projects if p.team == Team.NORTH_SOUTH)
    while len(state.completed_tricks) < 8:
        engine.play_card(state, engine.legal_moves(state, state.current_player)[0])
    from baloot_platform.research.project_compiler import OrToolsCpSatProjectCompilerV1
    trace = state.to_dict(reveal_hands=True)
    compiler = OrToolsCpSatProjectCompilerV1()
    plan = compiler.verify_trace(trace, rules=rules)
    assert {p.owner for p in plan.projects if p.revealed} == expected
    tampered = copy.deepcopy(trace)
    plays = tampered['completedTricks'][1]['plays']
    tampered['completedTricks'][1]['plays'] = plays[2:] + plays[:2]
    with pytest.raises(ValueError, match='first trick winner'):
        compiler.verify_trace(tampered, rules=rules)


def test_lower_teammate_project_scores_and_reveals_without_loser_cards(root):
    domain = _rules(root).project_domain
    projects = [
        Project(Seat.NORTH, 'fifty', tuple(map(Card.parse, 'S9 S10 SJ SQ'.split())), 10, 5, 0, True, 1),
        Project(Seat.WEST, 'fifty', tuple(map(Card.parse, 'H9 H10 HJ HQ'.split())), 10, 5, 0, True, 2),
        Project(Seat.EAST, 'siri', tuple(map(Card.parse, 'C7 C8 C9'.split())), 4, 2, 0),
    ]
    projects, winner = domain.score_team(projects, second_trick_leader=Seat.WEST)
    assert winner == Team.WEST_EAST
    assert domain.project_points(projects) == (0, 14)
    for seat in (Seat.WEST, Seat.SOUTH, Seat.EAST, Seat.NORTH):
        projects, _ = domain.reveal_for_turn(projects, seat)
    assert {p.owner for p in projects if p.revealed} == {Seat.WEST, Seat.EAST}
    assert projects[0].to_dict()['cards'] is None


def _second_trick_disclosure(root, card_rows=None):
    rules = _rules(root)
    engine = BalootEngine(rules)
    state = _playing_state(rules)
    hands = {seat: tuple(map(Card.parse, cards.split())) for seat, cards in (card_rows or (
        (Seat.NORTH, 'S7 S8 S9 HA SQ D10 C10 DA'),
        (Seat.WEST, 'H7 H8 H9 SA SK DJ CJ CA'),
        (Seat.SOUTH, 'D7 D8 D9 HK HQ DQ CQ H10'),
        (Seat.EAST, 'C7 C8 C9 SJ HJ DK CK S10')))}
    state.hands = [cards_to_mask(hands[seat]) for seat in Seat]
    state.initial_hands = list(state.hands)
    state.projects = rules.project_domain.detect(hands, Contract.SUN, Seat.EAST)
    for code in ('S7', 'SA', 'D7', 'SJ'):
        engine.play_card(state, Card.parse(code))
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST, buyer=Seat.NORTH)
    deck = [None] * 32
    for seat in Seat:
        for position, card in zip(layout.final_hand_positions[seat], hands[seat], strict=True):
            deck[position] = card.id
    prefix = ExactTracePrefixV1.build({'schemaVersion': 1, 'deckOrder': deck, 'dealer': 'east',
        'auctionActions': [{'actor': 'north', 'action': 'sun'}], 'escalationActions': [],
        'projectDeclarations': [project.public_declaration() for project in state.projects if project.declared],
        'playActions': [{'actor': seat.name.lower(), 'card': Card.parse(card).id}
                       for seat, card in zip(Seat, ('S7', 'SA', 'D7', 'SJ'), strict=True)]})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={'kind': 'play'}, maximum_seconds=20)
    return rules, engine, state, exact


@pytest.mark.parametrize('enumerated', (False, True))
def test_native_second_trick_followup_disclosure_matches_runtime(root, enumerated, monkeypatch):
    from baloot_platform.core.exact_state_view import build_canonical_observer_information_v1
    from baloot_platform.research.ortools_play_tail import TailPrefixActionV1

    rules, engine, state, exact = _second_trick_disclosure(root)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    start = compiler.graph_compiler._parse_start(exact, expected_rules_hash=rules.sha256)
    oracle = compiler.graph_compiler._path_oracle(start, target=Team.WEST_EAST,
        utility='awarded_point_margin', play_step_count=4, boundary_evaluator='current_raw_point_margin_v1')
    original = json.dumps(exact.state, sort_keys=True)
    continuation = tuple(TailPrefixActionV1.build(seat, code) for seat, code in zip(
        (Seat.WEST, Seat.SOUTH, Seat.EAST, Seat.NORTH), ('H7', 'H10', 'HJ', 'HA'), strict=True))
    if enumerated:
        oracle.build.model.add_bool_and(oracle.action_literals[index, action.card.id, False]
                                       for index, action in enumerate(continuation))
        assert oracle.enumerate_paths(maximum_nodes=5, maximum_seconds=10, cancelled=None) is None
    path = []
    for action in continuation:
        engine.play_card(state, action.card)
        path.append(action)
        result, _ = oracle.solve(tuple(path), maximum_seconds=10, cancelled=None)
        dynamic = compiler._dynamic_state(exact.state, start, result, tuple(path))
        actor = state.current_player.name.lower()
        actual = build_canonical_observer_information_v1(complete_state=dynamic, acting_seat=actor)
        expected = build_canonical_observer_information_v1(
            complete_state=state.to_dict(reveal_hands=True), acting_seat=actor)
        assert actual['publicState']['projects'] == expected['publicState']['projects']
        assert all(row['cards'] is None for row in actual['publicState']['projects']
                   if row['team'] == 'north_south')
    assert json.dumps(exact.state, sort_keys=True) == original
    witness = (next(iter(oracle._enumerated_witnesses.values())) if enumerated else
               list(oracle._solved_witnesses.values())[-1][0])
    expected_domains = set()
    for length in (0, 2):
        prefix = continuation[:length]
        result, _ = oracle.solve(prefix, maximum_seconds=5, cancelled=None)
        expected_domains.add(compiler._path_observation_id(exact.state, start, result, prefix, {}))
    monkeypatch.setattr('baloot_platform.research.ortools_complete_state_graph._solve_cp_sat_with_cancel_v1',
        lambda *args, **kwargs: pytest.fail('native prefix disclosure must reuse the retained witness'))
    _, domains, _ = compiler._observed_policy_bindings(exact.state, oracle, witness, Team.WEST_EAST,
        {}, {}, known_domains={}, maximum_seconds=5)
    assert set(domains) == expected_domains


def test_native_second_trick_world_requires_its_own_private_receipt(root, monkeypatch, deterministic_work):
    import copy
    import time
    from concurrent.futures import Future
    from baloot_platform.research import ortools_information_set_compiler as module
    from baloot_platform.research.scenario_contracts import ScenarioProofRequestV1, canonical_sha256

    rules, engine, state, exact = _second_trick_disclosure(root)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    other = copy.deepcopy(exact.state)
    purchase = Card.parse(other['purchaseCard'])
    north = next(Card.parse(code) for code in ('D10', 'DA') if Card.parse(code) != purchase)
    east = Card.parse('DK')
    for field in ('hands', 'initialHands'):
        other[field]['north'] = [east.to_dict() if Card.parse(card) == north else card for card in other[field]['north']]
        other[field]['east'] = [north.to_dict() if Card.parse(card) == east else card for card in other[field]['east']]
    hands = {Seat.parse(seat): tuple(map(Card.parse, cards)) for seat, cards in other['initialHands'].items()}
    rows, winner, receipt = compiler._projects_with_receipt(other, hands)
    assert receipt['selectedProjects'] and all(row['cards'] for row in receipt['selectedProjects'])
    assert all(row['cards'] is None for row in rows)
    other['projects'] = rows
    other['projectWinningTeam'] = winner.name.lower()
    identity = canonical_sha256(other)
    receipt = {key: value for key, value in receipt.items() if key != 'sha256'}
    receipt['worldStateHash'] = identity
    receipt['sha256'] = canonical_sha256(receipt)
    world = replace(exact, state=other, state_hash=identity)
    with pytest.raises(ValueError, match='normalization .*identity mismatch'):
        compiler.graph_compiler._parse_start(world, expected_rules_hash=rules.sha256)
    with pytest.raises(ValueError, match='receipt differs from its world'):
        compiler.graph_compiler._parse_start(world, expected_rules_hash=rules.sha256,
            project_receipt={**receipt, 'worldStateHash': exact.state_hash})
    start = compiler.graph_compiler._parse_start(world, expected_rules_hash=rules.sha256, project_receipt=receipt)
    assert start.project_disclosure_evidence
    monkeypatch.setattr(compiler, '_enumerate_worlds', lambda *args, **kwargs:
        ((other,), {'projectLifecycleBindings': [receipt]}))
    request = ScenarioProofRequestV1.build({'schemaVersion': 1, 'mode': 'bounded',
        'visibility': 'actor_information_set', 'claim': 'outcome_reachable', 'targetTeam': 'west_east',
        'strategyClass': 'pure_deterministic',
        'utility': 'awarded_point_margin', 'successThreshold': 0, 'budgetProfile': 'test',
        'horizon': {'unit': 'plies', 'value': 4}, 'boundaryEvaluator': 'current_raw_point_margin_v1'})
    worlds, _ = compiler._prepare_observed_play_worlds(exact, request, maximum_worlds=1,
        maximum_attempts=1, maximum_model_bytes=10_000_000, maximum_seconds=15)
    assert worlds[0][1].start.project_disclosure_evidence == start.project_disclosure_evidence
    class CapturedStart(Exception):
        pass
    def capture_start(value, **kwargs):
        assert value.project_disclosure_evidence == start.project_disclosure_evidence
        raise CapturedStart
    monkeypatch.setattr(compiler.graph_compiler, '_path_oracle', capture_start)
    limits = dict(maximum_nodes=1, maximum_depth=1, deadline=deterministic_work.read("monotonic") + 10,
        cancel_check=None, requested_recall={}, histories=None,
        project_receipts={identity: receipt, 'another_world': {'worldStateHash': 'another_world'}})
    with pytest.raises(CapturedStart):
        compiler._compile_play_worlds(exact, request, (other,), **limits)
    class Pool:
        def __init__(self, **kwargs):
            monkeypatch.setattr(module, '_world_compile_cancel', kwargs['initargs'][0])
        def submit(self, worker, payload):
            assert payload[-1]['project_receipts'] == {identity: receipt}
            future = Future()
            try:
                future.set_result(worker(payload))
            except CapturedStart as error:
                future.set_exception(error)
            return future
        def shutdown(self, **kwargs):
            pass
    monkeypatch.setattr(module, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(module, '_sample_world_compile_cpu', lambda: None)
    monkeypatch.setattr(module, 'OrToolsActorInformationSetCompilerV1', lambda *args, **kwargs: compiler)
    with pytest.raises(CapturedStart):
        compiler._compile_play_worlds_parallel(exact, request, (other,), graph_workers=1, **limits)
    for card in ('H7', 'H10', 'HJ', 'HA'):
        engine.play_card(state, Card.parse(card))
    complete = state.to_dict(reveal_hands=True)
    _, _, late = compiler._projects_with_receipt(complete,
        {Seat.parse(seat): tuple(map(Card.parse, cards)) for seat, cards in complete['initialHands'].items()})
    assert 'selectedProjects' not in late


def test_native_second_trick_reveal_distinguishes_two_previously_identical_worlds(root):
    from baloot_platform.core.exact_state_view import build_canonical_observer_information_v1
    from baloot_platform.research.ortools_play_tail import TailPrefixActionV1
    from baloot_platform.research.scenario_contracts import canonical_sha256

    card_rows = ((Seat.NORTH, 'S7 S8 S9 HA SQ D10 C10 DA'),
        (Seat.WEST, 'H7 H8 H9 SA SK C7 D8 CA'),
        (Seat.SOUTH, 'D7 DJ DQ DK H10 HQ HK C9'),
        (Seat.EAST, 'SJ CJ CQ CK HJ S10 D9 C8'))
    observations = []
    swaps = {'DJ': 'CJ', 'DQ': 'CQ', 'DK': 'CK', 'CJ': 'DJ', 'CQ': 'DQ', 'CK': 'DK'}
    for swapped in (False, True):
        rows = tuple((seat, ' '.join(swaps.get(card, card) for card in cards.split())
                      if swapped and seat in (Seat.SOUTH, Seat.EAST) else cards)
                     for seat, cards in card_rows)
        rules, _, _, exact = _second_trick_disclosure(root, rows)
        compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
        hands = {Seat.parse(seat): tuple(map(Card.parse, cards))
                 for seat, cards in exact.state['initialHands'].items()}
        _, _, receipt = compiler._projects_with_receipt(exact.state, hands)
        receipt = {key: value for key, value in receipt.items() if key != 'sha256'}
        receipt['worldStateHash'] = exact.state_hash
        receipt['sha256'] = canonical_sha256(receipt)
        start = compiler.graph_compiler._parse_start(exact, expected_rules_hash=rules.sha256, project_receipt=receipt)
        oracle = compiler.graph_compiler._path_oracle(start, target=Team.WEST_EAST,
            utility='awarded_point_margin', play_step_count=4, boundary_evaluator='current_raw_point_margin_v1')
        path = tuple(TailPrefixActionV1.build(seat, card) for seat, card in
                     ((Seat.WEST, 'H7'), (Seat.SOUTH, 'H10'), (Seat.EAST, 'HJ')))
        visible = []
        for length in (2, 3):
            result, _ = oracle.solve(path[:length], maximum_seconds=10, cancelled=None)
            dynamic = compiler._dynamic_state(exact.state, start, result, path[:length])
            view = build_canonical_observer_information_v1(complete_state=dynamic, acting_seat='west')
            assert all(row['cards'] is None for row in view['publicState']['projects']
                       if row['team'] == 'north_south')
            visible.append(view)
        observations.append(visible)
    assert observations[0][0] == observations[1][0]
    assert observations[0][1]['publicState']['projects'] != observations[1][1]['publicState']['projects']
    assert canonical_sha256(observations[0][1]) != canonical_sha256(observations[1][1])


def test_exact_tie_first_revealing_team_wins_even_without_leader_project(root):
    domain = _rules(root).project_domain
    projects = [Project(seat, 'siri', (), 4, 2, 0, True, ordinal)
                for ordinal, seat in enumerate((Seat.NORTH, Seat.WEST), 1)]
    # South acts first, but has no project: North-South still wins the team tie.
    scored, winner = domain.score_team(projects, second_trick_leader=Seat.SOUTH)
    assert winner == Team.NORTH_SOUTH
    assert domain.comparison_winner(scored, second_trick_leader=Seat.SOUTH).owner == Seat.NORTH


@pytest.mark.parametrize('native', (True, False))
def test_precomparison_normalization_never_awards_future_projects(root, native):
    raw = json.loads((root / 'examples/research-scenario-v4-ace-sacrifice-two-aces.json').read_text(
        encoding='utf-8'))['exactTracePrefix']
    raw['playActions'] = raw['playActions'][:3]
    raw['projectAssertions'] = []
    raw['stateAssertions'] = {}
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(_rules(root), native_cp_sat=native)
    exact = normalizer.normalize(prefix, decision_point={'kind': 'play'}, maximum_seconds=30)
    assert not exact.state['completedTricks']
    assert all(not item['scoring'] and not item['revealed'] for item in exact.state['projects'])
    assert not any(item['type'] == 'projects_compared' for item in exact.state['events'])
    preview = normalizer.preview_legal_next_actions(prefix, maximum_seconds=30)
    assert preview
    if native:
        initial = {seat: tuple(Card.parse(card) for card in exact.state['initialHands'][seat.name.lower()])
                   for seat in Seat}
        rows, winner, receipt = OrToolsActorInformationSetCompilerV1(
            _rules(root), native_cp_sat=True)._projects_v3(
                exact.state, initial, maximum_seconds=10, cancel_check=None)
        assert winner is None
        assert all(not row['scoring'] and not row['revealed'] and row['cards'] is None for row in rows)
        assert receipt['solverStatus'] == 'optimal'


def test_declaration_only_fragment_cannot_supply_future_award_evidence(root):
    model = cp_model.CpModel()
    owners = tuple(Seat(index // 8) for index in range(32))
    fragment = attach_project_lifecycle_fragment_v1(model,
        owner_of_card=tuple(model.new_int_var(int(owner), int(owner), f'owner_{index}')
                            for index, owner in enumerate(owners)),
        fixed_card_owners=owners, contract=Contract.SUN, dealer=Seat.EAST,
        trump=None, domain=_rules(root).project_domain, declaration_only=True,
        compact_fixed_states=True)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    assert solver.solve(model) == cp_model.OPTIMAL
    assert any(solver.boolean_value(project.declared) for project in fragment.projects)
    assert all(project.awarded is None and project.comparison_winner is None for project in fragment.projects)
    with pytest.raises(ValueError, match='no comparison or award evidence'):
        fragment.extract(solver)


def test_precomparison_state_identity_ignores_future_comparison(root, monkeypatch):
    rules = _rules(root)
    hands = ('S7 S8 S9 HA SQ D10 C10 DA', 'H7 H8 H9 SA SK DJ CJ CA',
             'D7 D8 D9 HK HQ DQ CQ H10', 'C7 C8 C9 SJ HJ DK CK S10')
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST, buyer=Seat.NORTH)
    deck = [None] * 32
    for seat in Seat:
        for position, card in zip(layout.final_hand_positions[seat], hands[seat].split(), strict=True):
            deck[position] = Card.parse(card).id
    raw = {'schemaVersion': 1, 'deckOrder': deck, 'dealer': 'east',
           'auctionActions': [{'actor': 'north', 'action': 'sun'}],
           'escalationActions': [], 'projectDeclarations': [], 'playActions': []}
    prefix = ExactTracePrefixV1.build(raw)
    original = OrToolsProjectLifecycleModelV1.solve
    states = []
    winners = set()
    for future_leader in (Seat.NORTH, Seat.WEST):
        def solve(self, **kwargs):
            kwargs['second_trick_leader'] = future_leader
            result = original(self, **kwargs)
            winners.add(result.winning_team)
            return result
        monkeypatch.setattr(OrToolsProjectLifecycleModelV1, 'solve', solve)
        exact = OrToolsExactStateNormalizerV1(_rules(root), native_cp_sat=False).normalize(
            prefix, decision_point={'kind': 'play'}, maximum_seconds=30)
        states.append((exact.state_hash, exact.state))
    assert states[0] == states[1]
    assert winners == {Team.NORTH_SOUTH, Team.WEST_EAST}


@pytest.mark.parametrize('leader', list(Seat))
@pytest.mark.parametrize('leader_has_project', (True, False))
def test_native_symbolic_and_fixed_models_follow_second_leader(root, leader, leader_has_project):
    domain = _rules(root).project_domain
    hands = {seat: tuple(card for card in FULL_DECK if int(card.suit) == int(seat)) for seat in Seat}
    if not leader_has_project:
        hands = {seat: tuple(map(Card.parse, cards.split())) for seat, cards in (
            (Seat.NORTH, 'S7 S8 S9 HA SQ D10 C10 C8'),
            (Seat.WEST, 'H7 H8 H9 SA SK D8 CJ CA'),
            (Seat.SOUTH, 'D7 DJ D9 HK HQ DQ CQ H10'),
            (Seat.EAST, 'C7 DA C9 SJ HJ DK CK S10'))}
    card_owners = {card: seat for seat, cards in hands.items() for card in cards}
    model = cp_model.CpModel()
    owners = tuple(model.new_int_var(0, 3, f'owner_{card.id}') for card in FULL_DECK)
    second_leader = model.new_int_var(0, 3, 'first_trick_winner')
    fragment = attach_project_lifecycle_fragment_v1(
        model, owner_of_card=owners, fixed_card_owners=tuple(card_owners[c] for c in FULL_DECK),
        contract=Contract.SUN, dealer=Seat.EAST, trump=None, domain=domain,
        second_trick_leader=second_leader, compact_fixed_states=True)
    model.add(second_leader == int(leader))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(model) == cp_model.OPTIMAL
    native = fragment.extract(solver)
    fixed = OrToolsProjectLifecycleModelV1().solve(
        hands=hands, contract=Contract.SUN, dealer=Seat.EAST, trump=None,
        domain=domain, second_trick_leader=leader, timeout_seconds=5)
    assert native.winning_team == fixed.winning_team == leader.team
    expected_owner = leader if leader_has_project or leader in (Seat.NORTH, Seat.WEST) else leader.partner
    assert next(p.owner for p in native.projects if p.comparison_winner) == expected_owner
    assert next(p.owner for p in fixed.projects if p.comparison_winner) == expected_owner
    assert native.combined_awarded_value == fixed.combined_awarded_value
