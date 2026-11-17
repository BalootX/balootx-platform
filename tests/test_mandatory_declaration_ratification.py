from types import SimpleNamespace

import pytest

from baloot_platform.core.rules import RuleBook
from baloot_platform.core.rule_source_catalog import load_exact_rule_source_catalog
from baloot_platform.research.service import ResearchFoundryServiceV1


def test_declaration_ratification_does_not_certify_pending_visibility(root):
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v5.yaml')
    catalog = load_exact_rule_source_catalog(rules, root)
    assert rules.project_domain.declaration_mode == 'mandatory'
    assert catalog.binding_for('projects.declaration_mode')['ratification_status'] == 'expert_ratified'
    assert catalog.binding_for('projects.top_rank_visibility')['ratification_status'] == 'pending_owner_ratification'
    assert rules.data['profile']['expert_ratified'] is False
    assert rules.data['projects']['top_rank_visibility'] == 'comparison_only'
    historical = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v4.yaml')
    assert historical.sha256 == '494d0eb9c7e4325e3337ca4c5d55e0ea7fda4295b218b2a57366dc0b39b23031'


def test_new_optional_recipe_is_rejected_before_storage():
    context = SimpleNamespace(bundle=SimpleNamespace(rules=SimpleNamespace(
        project_domain=SimpleNamespace(declaration_mode='optional'))))
    service = ResearchFoundryServiceV1.__new__(ResearchFoundryServiceV1)
    service._build_recipe_and_context = lambda _: (None, context)
    with pytest.raises(ValueError, match='historical-only'):
        ResearchFoundryServiceV1.create_recipe(service, {'schemaVersion': 4})


@pytest.mark.parametrize('backend_id', [None, 'ortools_cp_sat_v2', 'ortools_cp_sat_v3'])
def test_new_authoring_requires_native_v3(backend_id):
    context = SimpleNamespace(bundle=SimpleNamespace(rules=SimpleNamespace(
        project_domain=SimpleNamespace(declaration_mode='mandatory'))))
    backend = SimpleNamespace(identity=SimpleNamespace(backend_id=backend_id)) if backend_id else None
    service = SimpleNamespace(_cached_ortools_scenario_backend_v2=lambda _: backend)
    issue = ResearchFoundryServiceV1._new_recipe_authoring_issue(service, context)
    if backend_id == 'ortools_cp_sat_v3':
        assert issue is None
    else:
        assert issue['fieldPaths'] == ['configuration.bundleRevisionId']
        assert 'ortools_cp_sat_v3' in issue['details']['reason']


@pytest.mark.parametrize('loser_kind,loser_value,loser_rank', [('siri', 4, 7), ('fifty', 10, 5), ('fifty', 10, 6)])
def test_ratified_disclosure_only_winner_cards_and_comparison_rank(root, loser_kind, loser_value, loser_rank):
    from baloot_platform.core.models import Project
    from baloot_platform.core.cards import Card
    from baloot_platform.core.enums import Seat
    from baloot_platform.core.project_visibility import project_rank_visibility_policy, redact_project_ranks
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v6.yaml')
    import json
    import jsonschema
    jsonschema.validate(rules.data, json.loads((root / 'schemas/profiles/rules.schema.json').read_text(encoding='utf-8')))
    domain = rules.project_domain
    winner = Project(Seat.NORTH, 'fifty', tuple(map(Card.parse, ['C9', 'C10', 'CJ', 'CQ'])), 10, 6, 0, True, 0)
    loser = Project(Seat.WEST, loser_kind, tuple(map(Card.parse, ['H9', 'H10', 'HJ'])), loser_value, loser_rank, 0, True, 1)
    teammate = Project(Seat.SOUTH, 'siri', tuple(map(Card.parse, ['D9', 'D10', 'DJ'])), 4, 4, 0, True, 2)
    projects, _ = domain.score_team([winner, loser, teammate])
    assert domain.comparison_winner(projects).owner == Seat.NORTH
    for seat in Seat:
        projects, _ = domain.reveal_for_turn(projects, seat)
    assert [p.owner for p in projects if p.revealed] == [Seat.NORTH]
    public = [p.to_dict() for p in projects]
    state = {'projects': public, 'projectRankVisibility': project_rank_visibility_policy(rules), 'completedTricks': [{}]}
    redact_project_ranks(state, 'east', public)
    west = next(p for p in public if p['owner'] == 'west')
    assert west['cards'] is None
    assert 'suit' not in west
    assert west['highRank'] == (loser_rank if loser_kind == 'fifty' else None)
    catalog = load_exact_rule_source_catalog(rules, root)
    for path in ('projects.declaration_mode', 'projects.top_rank_visibility', 'projects.reveal_scope'):
        assert catalog.binding_for(path)['ratification_status'] == 'expert_ratified'


def test_scoring_team_disclosure_revision_preserves_history_and_public_losers(root):
    from baloot_platform.core.models import Project
    from baloot_platform.core.cards import Card
    from baloot_platform.core.enums import Seat
    from baloot_platform.core.project_visibility import project_rank_visibility_policy, redact_project_ranks
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v7.yaml')
    historical = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v6.yaml')
    assert historical.sha256 == '647de43ac69c77bc8159ff31a2b1778da9e0c843076a34cf0758394c22897b5c'
    assert rules.sha256 != historical.sha256
    assert rules.project_domain.reveal_scope == 'scoring_team'
    projects = [Project(seat, 'siri', tuple(map(Card.parse, cards.split())), 4, 2, 0, True, order)
                for order, (seat, cards) in enumerate(((Seat.NORTH, 'S7 S8 S9'),
                    (Seat.WEST, 'H7 H8 H9'), (Seat.SOUTH, 'D7 D8 D9'), (Seat.EAST, 'C7 C8 C9')))]
    projects, _ = rules.project_domain.score_team(projects)
    for seat in Seat:
        projects, _ = rules.project_domain.reveal_for_turn(projects, seat)
    assert {p.owner for p in projects if p.revealed} == {Seat.NORTH, Seat.SOUTH}
    public = [p.to_dict() for p in projects]
    state = {'projects': public, 'projectRankVisibility': project_rank_visibility_policy(rules), 'completedTricks': [{}]}
    redact_project_ranks(state, 'north', public)
    for row in public:
        if row['owner'] in ('west', 'east'):
            assert row['cards'] is None
            assert 'suit' not in row
            assert row['highRank'] == 2
        else:
            assert row['cards']
    catalog = load_exact_rule_source_catalog(rules, root)
    assert catalog.binding_for('projects.reveal_scope')['ratification_status'] == 'expert_ratified'
