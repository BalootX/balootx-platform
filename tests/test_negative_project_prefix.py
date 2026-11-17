import json
from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.research import negative_project_prefix as module


def _arguments(root):
    return dict(
        rules=RuleBook.load(root / "configs/rules/saudi_federation_2023_v6.yaml"),
        dealer="east", buyer="east", contract="sun", trump=None,
        purchase_card="DA", prefix=(), observer="north",
        observer_hand="S7 S9 SJ SK H7 H9 HJ HK".split(),
        public_projects={seat: [] for seat in Seat}, revealed_projects=(),
        required_remaining_owners={"DA": "east"}, maximum_seconds=10,
    )


def test_cancelled_prefix_does_not_start_construction(root, monkeypatch):
    monkeypatch.setattr(module, 'OrToolsScenarioModelV1', lambda **kw: pytest.fail('started after cancellation'))
    with pytest.raises(module.PlayTailSolveStopped) as stopped:
        module.build_negative_project_prefix_v1(**_arguments(root), cancel_check=lambda: True)
    assert stopped.value.solver_status == 'unknown'


@pytest.mark.real_clock
def test_prefix_build_time_is_subtracted_from_solve_budget(root, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    attach = module.attach_project_lifecycle_fragment_v1
    def projects(*args, **kwargs):
        result = attach(*args, **kwargs)
        now[0] += 4
        return result
    def solve(_self, **kwargs):
        assert kwargs['maximum_time_seconds'] == 6
        assert kwargs['maximum_conflicts'] == 17
        assert not kwargs['cancel_check']()
        now[0] += 6
        assert kwargs['cancel_check']()
        raise module.PlayTailSolveStopped('unknown')
    monkeypatch.setattr(module, 'attach_project_lifecycle_fragment_v1', projects)
    monkeypatch.setattr(module.OrToolsPlayTailStageV1, 'solve', solve)
    with pytest.raises(module.PlayTailSolveStopped):
        module.build_negative_project_prefix_v1(**_arguments(root), maximum_conflicts=17)


def test_solver_reallocates_hidden_ownership_and_excludes_allocations(root, monkeypatch):
    # Test the new deal/project projection with the real solver; full legal play
    # construction remains owned by the existing play-tail tests and later audit.
    def solve(_self, **kwargs):
        solver = kwargs["solver"]
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = 10
        status = solver.solve(kwargs["shared_model"])
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert kwargs["first_leader"] == Seat.NORTH
        assert kwargs['ownership'][Card.parse('DA')] == Seat.EAST
        assert all(kwargs['ownership'][Card.parse(card)] == Seat.NORTH
                   for card in _arguments(root)['observer_hand'])
        assert len(kwargs['ownership']) == 9
        return SimpleNamespace(solver_status="feasible")

    monkeypatch.setattr(module.OrToolsPlayTailStageV1, "solve", solve)
    args = _arguments(root)
    first = module.build_negative_project_prefix_v1(**args)
    second = module.build_negative_project_prefix_v1(
        **args, excluded_ownerships=(first["ownership"],),
    )
    assert first["ownership"] != second["ownership"]
    for proposal in (first, second):
        assert proposal["status"] == "proposal" and not proposal["qualified"]
        assert sorted(proposal["deckOrder"]) == list(range(32))
        assert proposal["ownership"][Card.parse("DA").id] == int(Seat.EAST)
        assert {index for index, seat in enumerate(proposal["ownership"]) if seat == int(Seat.NORTH)} == {
            Card.parse(card).id for card in args["observer_hand"]
        }


def test_rejects_incomplete_public_project_history(root):
    args = _arguments(root)
    args["public_projects"] = {Seat.NORTH: []}
    with pytest.raises(ValueError, match="all seats"):
        module.build_negative_project_prefix_v1(**args)


def test_v8_constructor_binds_reveal_order_to_native_first_trick_winner(root, monkeypatch):
    args = _arguments(root)
    args['rules'] = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')

    def solve(_self, **kwargs):
        model, solver = kwargs['shared_model'], kwargs['solver']
        assert len(kwargs['construction_plan_hash']) == 64
        kwargs['model_configurator'](SimpleNamespace(
            _extract=SimpleNamespace(winner_seats=[model.new_constant(int(Seat.SOUTH))])))
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = 10
        assert solver.solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        leader = next(index for index, var in enumerate(model.proto.variables)
                      if var.name == 'negative_project_second_trick_leader')
        assert solver.value(model.get_int_var_from_proto_index(leader)) == int(Seat.SOUTH)
        return SimpleNamespace(solver_status='feasible')

    monkeypatch.setattr(module.OrToolsPlayTailStageV1, 'solve', solve)
    assert module.build_negative_project_prefix_v1(**args)['qualified'] is False


def test_rejects_complete_hidden_fixture(root):
    args = _arguments(root)
    args["required_remaining_owners"] = {Card(index): Seat.EAST for index in range(24)}
    with pytest.raises(ValueError, match="leave hidden ownership free"):
        module.build_negative_project_prefix_v1(**args)


def test_rejects_motif_conflicting_with_observer_hand(root):
    args = _arguments(root)
    args['required_remaining_owners'] = {'S7': 'east'}
    with pytest.raises(ValueError, match='disagrees with observer hand'):
        module.build_negative_project_prefix_v1(**args)


def test_known_ownership_reduces_project_lowering_without_dropping_patterns(root, monkeypatch):
    attach = module.attach_project_lifecycle_fragment_v1

    def compare(model, **kwargs):
        reference_model = model.clone()
        reference_kwargs = {key: value for key, value in kwargs.items()
                            if key not in {'fixed_card_owners', 'compact_fixed_states',
                                           'tighten_value_bounds'}}
        reference_kwargs['owner_of_card'] = tuple(
            reference_model.get_int_var_from_proto_index(owner.index)
            for owner in kwargs['owner_of_card'])
        reference = attach(reference_model, **reference_kwargs)
        specialized = attach(model, **kwargs)
        assert [p.project_id for p in specialized.projects] == [
            p.project_id for p in reference.projects]
        assert len(model.proto.variables) < len(reference_model.proto.variables)
        assert len(model.proto.constraints) < len(reference_model.proto.constraints)
        print(f'project lowering variables: {len(reference_model.proto.variables)} -> '
              f'{len(model.proto.variables)}; constraints: '
              f'{len(reference_model.proto.constraints)} -> {len(model.proto.constraints)}')
        return specialized

    def measured(*args, **kwargs):
        raise RuntimeError('measured')

    monkeypatch.setattr(module, 'attach_project_lifecycle_fragment_v1', compare)
    monkeypatch.setattr(module.OrToolsPlayTailStageV1, 'solve', measured)
    with pytest.raises(RuntimeError, match='measured'):
        module.build_negative_project_prefix_v1(**_arguments(root))


def test_revealed_winner_does_not_exclude_hidden_awarded_teammate():
    model = cp_model.CpModel()
    projects = [SimpleNamespace(owner=seat, cards=tuple(map(Card.parse, cards)),
        awarded=model.new_constant(1), comparison_winner=model.new_constant(int(index == 0)))
        for index, (seat, cards) in enumerate(((Seat.NORTH, ['H7','H8','H9']),
                                              (Seat.SOUTH, ['D7','D8','D9'])))]
    revealed = {(Seat.NORTH, frozenset(c.id for c in projects[0].cards))}
    module._bind_revealed_projects(model, projects, revealed,
                                  SimpleNamespace(reveal_scope='comparison_winner'))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL


@pytest.mark.parametrize('seconds', [True, 0, -1, float('inf'), float('nan')])
def test_rejects_unsafe_time_limits(root, seconds):
    args = _arguments(root)
    args['maximum_seconds'] = seconds
    with pytest.raises(ValueError, match='finite and positive'):
        module.build_negative_project_prefix_v1(**args)


@pytest.mark.parametrize('hide_teammate', [False, True])
def test_scoring_team_prototype_requires_teammate_disclosure(root, hide_teammate):
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v7.yaml')
    model = cp_model.CpModel()
    projects = [SimpleNamespace(owner=seat, cards=tuple(map(Card.parse, cards.split())),
        awarded=model.new_constant(int(index < 2)), comparison_winner=model.new_constant(int(index == 0)))
        for index, (seat, cards) in enumerate(((Seat.NORTH, 'S7 S8 S9'),
            (Seat.SOUTH, 'D7 D8 D9'), (Seat.WEST, 'H7 H8 H9'), (Seat.EAST, 'C7 C8 C9')))]
    revealed = {(p.owner, frozenset(c.id for c in p.cards)) for p in projects[:1 if hide_teammate else 2]}
    module._bind_revealed_projects(model, projects, revealed, rules.project_domain)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == (cp_model.INFEASIBLE if hide_teammate else cp_model.OPTIMAL)


@pytest.fixture
def public_rank_source(root):
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')
    prefix = ExactTracePrefixV1.build(json.loads(
        (root / 'tests/fixtures/project_rank_ownership_prefix.json').read_text(encoding='utf-8')))
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
        prefix, maximum_seconds=10)
    return dict(rules=rules, source_prefix=prefix, normalized=normalized,
                project_owner='west', card='C9', expected_owner='west', maximum_seconds=10)


def test_native_public_rank_is_necessary_for_club_nine_ownership(public_rank_source, monkeypatch):
    from baloot_platform.core.auction_ir import AuctionDomainIRV1
    from baloot_platform.core.exact_state_view import build_canonical_observer_information_v1
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, canonical_sha256

    args = public_rank_source
    before = args['source_prefix'].to_dict()

    def forbidden(*args, **kwargs):
        pytest.fail('native ownership proof called Python auction semantics')

    monkeypatch.setattr(AuctionDomainIRV1, 'legal_actions', forbidden)
    monkeypatch.setattr(AuctionDomainIRV1, 'resolve_action', forbidden)
    result = module.probe_public_project_rank_ownership_v1(**args)
    assert result['status'] == 'clue_dependent' and result['qualified'] is False
    assert result['ownership'] == {'card': 'C9', 'owner': 'west'}
    assert result['sourcePrefixHash'] == args['source_prefix'].sha256
    assert result['normalizationPrefixHash'] == args['normalized'].prefix_hash
    assert result['clue'] == {'owner': 'west', 'kind': 'siri', 'highRank': 2, 'declarationOrder': 1}
    assert [(p['rankIncluded'], p['assumedOwner'], p['status']) for p in result['probes']] == [
        (True, 'west', 'proposal'), (True, 'north', 'infeasible'),
        (True, 'east', 'infeasible'), (False, 'north', 'proposal')]
    assert all(len(p['constructionModelHash']) == 64 and p['modelSize']['constraints'] > 0
               for p in result['probes'])
    assert result['sha256'] == canonical_sha256({k: v for k, v in result.items() if k != 'sha256'})
    assert args['source_prefix'].to_dict() == before
    witness = OrToolsExactStateNormalizerV1(args['rules'], native_cp_sat=True).normalize_play_or_project(
        ExactTracePrefixV1.build(result['contraryWitness']), maximum_seconds=10)
    assert Card.parse('C9').id in [Card.parse(c).id for c in witness.state['hands']['north']]

    views = [build_canonical_observer_information_v1(complete_state=state, acting_seat='south')['publicState']
             for state in (args['normalized'].state, witness.state)]
    # Mask that single announcement wherever the shared public projection repeats it.
    def mask(value):
        if isinstance(value, dict):
            if value.get('owner') == 'west' and value.get('kind') == 'siri':
                assert value.get('cards') is None
                value['highRank'] = None
            for item in value.values():
                mask(item)
        elif isinstance(value, list):
            for item in value:
                mask(item)
    assert views[0] != views[1]
    for view in views:
        mask(view)
    assert views[0] == views[1]


def test_inference_rejects_wrong_source_and_private_project_clue(public_rank_source, monkeypatch):
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    monkeypatch.setattr(module, 'build_negative_project_prefix_v1', lambda **kw: pytest.fail('started invalid probe'))
    args = public_rank_source
    changed = args['source_prefix'].to_dict()
    changed.pop('sha256')
    changed['playActions'].pop()
    with pytest.raises(ValueError, match='source authority'):
        module.probe_public_project_rank_ownership_v1(**{**args, 'source_prefix': ExactTracePrefixV1.build(changed)})
    with pytest.raises(ValueError, match='publicly announced, unrevealed rank'):
        module.probe_public_project_rank_ownership_v1(**{**args, 'project_owner': 'south'})
    with pytest.raises(ValueError, match='unseen'):
        module.probe_public_project_rank_ownership_v1(**{**args, 'card': 'C10'})


def test_inference_binds_assertion_bearing_input_separately_from_effective_prefix(public_rank_source):
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    args = public_rank_source
    payload = args['source_prefix'].to_dict()
    payload.pop('sha256')
    payload['stateAssertions'] = {'currentPlayer': 'south'}
    prefix = ExactTracePrefixV1.build(payload)
    normalized = OrToolsExactStateNormalizerV1(args['rules'], native_cp_sat=True).normalize_play_or_project(
        prefix, maximum_seconds=10)
    result = module.probe_public_project_rank_ownership_v1(**{**args, 'source_prefix': prefix, 'normalized': normalized})
    assert result['status'] == 'clue_dependent'
    assert result['sourcePrefixHash'] == prefix.sha256
    assert result['normalizationPrefixHash'] == normalized.prefix_hash
    assert result['sourcePrefixHash'] != result['normalizationPrefixHash']


def test_inference_checks_entire_remaining_public_observation(public_rank_source, monkeypatch):
    from baloot_platform.core import exact_state_view

    project = exact_state_view.build_canonical_observer_information_v1
    calls = []
    def changed_witness(**kwargs):
        result = project(**kwargs)
        if calls:
            result['publicState']['matchScore'][0] += 1
        calls.append(result)
        return result
    monkeypatch.setattr(exact_state_view, 'build_canonical_observer_information_v1', changed_witness)
    assert module.probe_public_project_rank_ownership_v1(**public_rank_source)['status'] == 'witness_observation_mismatch'


def test_inference_does_not_call_satisfiable_contrary_ownership_entailed(public_rank_source, monkeypatch):
    monkeypatch.setattr(module, 'build_negative_project_prefix_v1', lambda **kw: {
        'status': 'proposal', 'solverStatus': 'optimal', 'constructionModelHash': 'a' * 64,
        'modelSize': {'variables': 1, 'constraints': 1}})
    result = module.probe_public_project_rank_ownership_v1(**public_rank_source)
    assert result['status'] == 'not_entailed' and len(result['probes']) == 2
    assert 'contraryWitness' not in result


def test_inference_rejects_pre_reveal_context(public_rank_source, monkeypatch):
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    monkeypatch.setattr(module, 'build_negative_project_prefix_v1', lambda **kw: pytest.fail('started unsupported probe'))
    args = public_rank_source
    payload = args['source_prefix'].to_dict()
    payload.pop('sha256')
    payload['playActions'] = payload['playActions'][:7]
    prefix = ExactTracePrefixV1.build(payload)
    normalized = OrToolsExactStateNormalizerV1(args['rules'], native_cp_sat=True).normalize_play_or_project(
        prefix, maximum_seconds=10)
    with pytest.raises(ValueError, match='post-reveal plain Sun'):
        module.probe_public_project_rank_ownership_v1(**{**args, 'source_prefix': prefix, 'normalized': normalized})


@pytest.mark.parametrize('status, expected', [('infeasible', 'source_model_disagreement'), ('incomplete', 'incomplete')])
def test_inference_cannot_prove_vacuous_or_incomplete_models(public_rank_source, monkeypatch, status, expected):
    monkeypatch.setattr(module, 'build_negative_project_prefix_v1', lambda **kw: {
        'status': status, 'solverStatus': 'infeasible' if status == 'infeasible' else 'unknown',
        'constructionModelHash': 'a' * 64, 'modelSize': {'variables': 1, 'constraints': 1}})
    result = module.probe_public_project_rank_ownership_v1(**public_rank_source)
    assert result['status'] == expected
    assert len(result['probes']) == 1 and result['qualified'] is False
    assert 'contraryWitness' not in result


@pytest.mark.real_clock
@pytest.mark.parametrize('contrary_status', ['infeasible', 'proposal'])
def test_inference_keeps_one_shared_deadline(public_rank_source, monkeypatch, contrary_status):
    now, budgets = [100.0], []
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    def probe(**kwargs):
        budgets.append(kwargs['maximum_seconds'])
        now[0] += 6
        return {'status': 'proposal' if len(budgets) == 1 else contrary_status, 'solverStatus': 'optimal',
                'constructionModelHash': 'a' * 64, 'modelSize': {'variables': 1, 'constraints': 1}}
    monkeypatch.setattr(module, 'build_negative_project_prefix_v1', probe)
    result = module.probe_public_project_rank_ownership_v1(**public_rank_source)
    assert result['status'] == 'incomplete'
    assert budgets == [10.0, 4.0]


def test_cancelled_inference_does_not_start_solver(public_rank_source, monkeypatch):
    monkeypatch.setattr(module, 'build_negative_project_prefix_v1', lambda **kw: pytest.fail('started cancelled probe'))
    assert module.probe_public_project_rank_ownership_v1(
        **public_rank_source, cancel_check=lambda: True)['status'] == 'incomplete'


def test_inference_probe_does_not_turn_silence_into_no_eligible_project(root, monkeypatch):
    def projects(model, **kwargs):
        return SimpleNamespace(projects=[SimpleNamespace(owner=Seat.WEST, kind='siri', high_rank=2,
            cards=tuple(map(Card.parse, ('C7', 'C8', 'C9'))), declared=model.new_constant(0),
            eligible=model.new_constant(1), awarded=model.new_constant(0), comparison_winner=model.new_constant(0))])
    def solve(_self, **kwargs):
        solver = kwargs['solver']
        solver.parameters.num_search_workers = 1
        status = solver.solve(kwargs['shared_model'])
        if status == cp_model.INFEASIBLE:
            raise module.PlayTailSolveStopped('infeasible')
        assert status == cp_model.OPTIMAL
        return SimpleNamespace(solver_status='optimal')
    monkeypatch.setattr(module, 'attach_project_lifecycle_fragment_v1', projects)
    monkeypatch.setattr(module.OrToolsPlayTailStageV1, 'solve', solve)
    args = _arguments(root)
    args['auction_actions'] = ({'actor': 'north', 'action': 'pass'},
        {'actor': 'west', 'action': 'pass'}, {'actor': 'south', 'action': 'pass'},
        {'actor': 'east', 'action': 'pass'}, {'actor': 'north', 'action': 'pass'},
        {'actor': 'west', 'action': 'pass'}, {'actor': 'south', 'action': 'pass'},
        {'actor': 'east', 'action': 'sun'})
    with pytest.raises(module.PlayTailSolveStopped):
        module.build_negative_project_prefix_v1(**args)
    assert module.build_negative_project_prefix_v1(**args, _observer_inference_probe=True)['status'] == 'proposal'
