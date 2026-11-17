from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Rank, Seat, Suit
from baloot_platform.research.ortools_project_fragment import (
    attach_project_lifecycle_fragment_v1,
)
from baloot_platform.research.scenario_contracts import ScenarioPredicateV1
from baloot_platform.research.scenario_project_outcome_lowering_v2 import (
    CpSatOutcomePredicateContextV1,
    CpSatProjectPredicateContextV1,
    OUTCOME_PREDICATE_KINDS_V2,
    PROJECT_OUTCOME_PREDICATE_KINDS_V2,
    PROJECT_PREDICATE_KINDS_V2,
    ProjectOutcomePredicateLoweringV2,
    project_outcome_predicate_registrations_v1,
)


_DECK = (
    10, 13, 23, 5, 9, 0, 7, 22, 31, 6, 27, 20, 1, 12, 4, 18,
    25, 24, 28, 8, 11, 14, 16, 17, 3, 19, 29, 26, 15, 2, 21, 30,
)


def _predicate(predicate_id: str, kind: str, arguments: dict) -> ScenarioPredicateV1:
    return ScenarioPredicateV1.build(
        {
            "predicateId": predicate_id,
            "kind": kind,
            "version": 1,
            "hardness": "hard",
            "arguments": arguments,
        }
    )


def _solve(model: cp_model.CpModel) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    assert solver.solve(model) == cp_model.OPTIMAL
    return solver


def _fixed(model: cp_model.CpModel, name: str, value: int, maximum: int):
    variable = model.new_int_var(0, maximum, name)
    model.add(variable == value)
    return variable


def _outcome_context(
    source_stage: str = "full_play",
) -> CpSatOutcomePredicateContextV1:
    model = cp_model.CpModel()
    return CpSatOutcomePredicateContextV1(
        model=model,
        context_id="full-play-terminal-state",
        source_stage=source_stage,
        source_hash="a" * 64,
        buyer_seat=_fixed(model, "buyer", int(Seat.NORTH), 3),
        round_winner=_fixed(model, "round_winner", 1, 1),
        match_winner=_fixed(model, "match_winner", 1, 2),
        raw_points=(
            _fixed(model, "raw_ns", 50, 162),
            _fixed(model, "raw_we", 112, 162),
        ),
        game_points=(
            _fixed(model, "game_ns", 0, 1000),
            _fixed(model, "game_we", 26, 1000),
        ),
        project_points=(
            _fixed(model, "project_ns", 20, 400),
            _fixed(model, "project_we", 0, 400),
        ),
        final_match_score=(
            _fixed(model, "match_ns", 140, 1000),
            _fixed(model, "match_we", 160, 1000),
        ),
        kabout=_fixed(model, "kabout", 0, 1),
        gahwa=_fixed(model, "gahwa", 0, 1),
        match_target=152,
    )


@pytest.mark.parametrize('scores,expected', [((19, 11), True), ((11, 19), True),
                                         ((26, 0), False), ((0, 26), False), ((18, 18), True)])
def test_guard_draft_contested_witness_uses_native_awarded_margin(root, scores, expected):
    from baloot_platform.research.family_catalog import ResearchFamilyCatalogV1

    family, = ResearchFamilyCatalogV1.load(root / 'configs/research/cross_suit_guard.yaml').families
    bindings = {key: values[0] for key, values in family.parameters.choices(family.contracts).items()}
    predicate, = (p for p in family.instantiate(bindings).predicates() if p.kind == 'outcome.round_margin')
    context = _outcome_context()
    context = replace(context, game_points=tuple(context.model.new_constant(value) for value in scores))
    lowering = ProjectOutcomePredicateLoweringV2().lower_predicate(predicate, context)
    solver = _solve(context.model)
    assert solver.boolean_value(lowering.literal) is expected
    assert lowering.extract(solver)['observed']['value'] == scores[0] - scores[1]


def _project_context(rules) -> CpSatProjectPredicateContextV1:
    layout = DealLayoutV1.build(
        rules.structural_domain,
        dealer=Seat.EAST,
        buyer=Seat.NORTH,
    )
    hands = {
        seat: tuple(Card(_DECK[position]) for position in layout.positions_for(seat, final=True))
        for seat in Seat
    }
    expected_owner = {
        card.id: seat for seat, cards in hands.items() for card in cards
    }
    model = cp_model.CpModel()
    owners = tuple(
        model.new_int_var(0, 3, f"owner_{card_id}") for card_id in range(32)
    )
    for card_id, variable in enumerate(owners):
        model.add(variable == int(expected_owner[card_id]))
    fragment = attach_project_lifecycle_fragment_v1(
        model,
        owner_of_card=owners,
        contract="sun",
        dealer="east",
        trump=None,
        domain=rules.project_domain,
    )
    return CpSatProjectPredicateContextV1(model, "fixed-project-state", fragment)


@pytest.mark.parametrize("contract", ("sun", "hokum"))
def test_eligibility_projection_exists_is_exact_and_rejects_award_claims(rules, contract):
    hands = {
        Seat.NORTH: [Card.of(Suit.DIAMONDS, rank) for rank in (
            Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE)],
        Seat.WEST: [Card.of(Suit.HEARTS, rank) for rank in (
            Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE)],
        Seat.SOUTH: [], Seat.EAST: [],
    }
    reserved = {card.id for hand in hands.values() for card in hand}
    remaining = iter(Card(card) for card in range(32) if card not in reserved)
    for hand in hands.values():
        hand.extend(next(remaining) for _ in range(8 - len(hand)))
    ownership = {card.id: seat for seat, hand in hands.items() for card in hand}
    outcomes = []
    for projected in (False, True):
        model = cp_model.CpModel()
        owners = tuple(model.new_int_var(0, 3, f"owner_{card}") for card in range(32))
        fragment = attach_project_lifecycle_fragment_v1(model, owner_of_card=owners,
            contract=contract, dealer="east", trump="clubs" if contract == "hokum" else None,
            domain=rules.project_domain, eligibility_only=projected)
        for card, seat in ownership.items():
            model.add(owners[card] == int(seat))
        context = CpSatProjectPredicateContextV1(model, "eligibility-parity", fragment)
        lowerer = ProjectOutcomePredicateLoweringV2()
        leaves = [lowerer.lower_predicate(_predicate(f"PRED-ELIGIBLE-{index}", "project.exists",
            dict(team=team, owner=owner, kind="siri", value=4 if contract == "sun" else 2, state="eligible")), context)
            for index, (team, owner) in enumerate((("north_south", "north"), ("west_east", "west")))]
        for leaf in leaves:
            model.add(leaf.literal == 1)
        solver = _solve(model)
        outputs = [leaf.extract(solver)["observed"] for leaf in leaves]
        outcomes.append([(row["count"], row["combinedValue"]) for row in outputs])
        if projected:
            assert all(row["constructionScope"] == "project_eligibility" for row in outputs)
            assert all("awarded" not in p for row in outputs for p in row["projects"])
            for kind, args in (("project.exists", dict(team="north_south", owner="north",
                    kind="siri", value=4 if contract == "sun" else 2, state="awarded")),
                    ("project.baloot", dict(team="north_south", achieved=True))):
                with pytest.raises(ValueError, match="cannot lower"):
                    lowerer.lower_predicate(_predicate("PRED-UNAVAILABLE", kind, args), context)
        model.add(leaves[0].literal == 0)
        assert solver.solve(model) == cp_model.INFEASIBLE
    assert outcomes[0] == outcomes[1]


def test_category_exposes_all_fifteen_stable_registrations():
    registrations = project_outcome_predicate_registrations_v1()

    assert {row.kind for row in registrations} == PROJECT_OUTCOME_PREDICATE_KINDS_V2
    assert len(registrations) == 15
    assert sum(row.context_kind == "project" for row in registrations) == 5
    assert sum(row.context_kind == "outcome" for row in registrations) == 10
    assert len({row.evidence_extractor_id for row in registrations}) == 15
    assert len(ProjectOutcomePredicateLoweringV2().manifest_hash) == 64


@pytest.mark.parametrize('contract', ['sun', 'hokum'])
@pytest.mark.parametrize('state', ['eligible', 'declared'])
def test_declaration_projection_matches_full_lifecycle_without_future_evidence(root, state, contract):
    from baloot_platform.core.rules import RuleBook

    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST, buyer=Seat.NORTH)
    ownership = {card: seat for seat in Seat
                 for card in (_DECK[position] for position in layout.positions_for(seat, final=True))}
    observations = []
    for projected in (False, True):
        model = cp_model.CpModel()
        owners = tuple(model.new_int_var(int(ownership[card]), int(ownership[card]), f'owner_{card}')
                       for card in range(32))
        fragment = attach_project_lifecycle_fragment_v1(model, owner_of_card=owners,
            fixed_card_owners=tuple(ownership[card] for card in range(32)), contract=contract,
            dealer='east', trump='clubs' if contract == 'hokum' else None, domain=rules.project_domain,
            second_trick_leader=None if projected else Seat.NORTH, declaration_only=projected,
            compact_fixed_states=True)
        context = CpSatProjectPredicateContextV1(model, 'declaration-parity', fragment)
        lowerer = ProjectOutcomePredicateLoweringV2()
        leaf = lowerer.lower_predicate(_predicate('PRED-DECLARATION', 'project.count',
            dict(team='north_south', minimum=1, maximum=16, state=state)), context)
        solver = _solve(model)
        assert solver.boolean_value(leaf.literal)
        observed = leaf.extract(solver)['observed']
        observations.append((observed['count'], observed['combinedValue'],
                             sorted(project['projectId'] for project in observed['projects'])))
        if projected:
            assert observed['constructionScope'] == 'project_declarations'
            assert all(project[state] for project in observed['projects'])
            assert all('comparisonWinner' not in project and 'awarded' not in project
                       for project in observed['projects'])
            for kind, args in [('project.count', dict(team='north_south', minimum=0, maximum=16, state=future))
                    for future in ('comparison_winner', 'awarded')] + [
                        ('project.baloot', dict(team='north_south', achieved=False))]:
                with pytest.raises(ValueError, match='cannot lower'):
                    lowerer.lower_predicate(_predicate('PRED-FUTURE', kind, args), context)
        model.add(leaf.literal == 0)
        assert solver.solve(model) == cp_model.INFEASIBLE
    assert observations[0] == observations[1]


def test_project_predicates_lower_exactly_and_keep_lifecycle_states_distinct(rules):
    context = _project_context(rules)
    lowerer = ProjectOutcomePredicateLoweringV2()
    predicates = (
        _predicate(
            "PRED-PROJECT-EXISTS",
            "project.exists",
            {
                "team": "north_south",
                "owner": "north",
                "kind": "hundred_sequence",
                "value": 20,
                "state": "awarded",
            },
        ),
        _predicate(
            "PRED-PROJECT-COUNT",
            "project.count",
            {
                "team": "north_south",
                "minimum": 2,
                "maximum": 2,
                "state": "awarded",
            },
        ),
        _predicate(
            "PRED-PROJECT-VALUE",
            "project.combined_value",
            {
                "team": "north_south",
                "minimum": 24,
                "maximum": 24,
                "state": "awarded",
            },
        ),
        _predicate(
            "PRED-PROJECT-BALOOT",
            "project.baloot",
            {"team": "north_south", "achieved": False},
        ),
        _predicate(
            "PRED-PROJECT-OVERLAP",
            "project.overlap_count",
            {"owner": "north", "minimum": 5, "maximum": 5},
        ),
    )
    lowerings = tuple(lowerer.lower_predicate(item, context) for item in predicates)
    for item in lowerings:
        context.model.add_assumption(item.literal)
    solver = _solve(context.model)
    evidence = {item.kind: item.extract(solver) for item in lowerings}

    assert set(evidence) == PROJECT_PREDICATE_KINDS_V2
    exists = evidence["project.exists"]["observed"]
    assert exists["count"] == 1
    assert exists["combinedValue"] == 20
    lifecycle = exists["projects"][0]
    assert lifecycle["eligible"] is True
    assert lifecycle["selected"] is True
    assert lifecycle["declarable"] is True
    assert lifecycle["declared"] is True
    assert lifecycle["comparisonWinner"] is True
    assert lifecycle["awarded"] is True
    assert evidence["project.count"]["observed"]["count"] == 2
    assert evidence["project.combined_value"]["observed"]["combinedValue"] == 24
    assert evidence["project.baloot"]["observed"]["achieved"] is False
    assert evidence["project.overlap_count"]["observed"]["overlapCardCount"] == 5
    assert all(item.literal.Name().startswith("scenario_assumption_") for item in lowerings)
    assert all(len(item.extract(solver)["sha256"]) == 64 for item in lowerings)


@pytest.mark.parametrize("source_stage", ["scoring", "full_play"])
def test_all_terminal_outcome_predicates_lower_from_scoring_variables(source_stage):
    context = _outcome_context(source_stage)
    lowerer = ProjectOutcomePredicateLoweringV2()
    specifications = {
        "outcome.buyer_contract": {"status": "lost"},
        "outcome.team_round": {"team": "west_east", "status": "won"},
        "outcome.team_match": {"team": "west_east", "status": "won"},
        "outcome.raw_points": {"team": "north_south", "minimum": 50, "maximum": 50},
        "outcome.game_points": {"team": "west_east", "minimum": 26, "maximum": 26},
        "outcome.project_points": {"team": "north_south", "minimum": 20, "maximum": 20},
        "outcome.round_margin": {"team": "west_east", "minimum": 25.5, "maximum": 26.5},
        "outcome.kabout": {"team": "west_east", "achieved": False},
        "outcome.gahwa": {"team": "west_east", "achieved": False},
        "outcome.match_target_crossing": {"team": "west_east", "achieved": True},
    }
    lowerings = tuple(
        lowerer.lower_predicate(
            _predicate(f"PRED-OUTCOME-{index}", kind, arguments), context
        )
        for index, (kind, arguments) in enumerate(specifications.items())
    )
    for item in lowerings:
        context.model.add_assumption(item.literal)
    solver = _solve(context.model)
    evidence = {item.kind: item.extract(solver) for item in lowerings}

    assert set(evidence) == OUTCOME_PREDICATE_KINDS_V2
    assert evidence["outcome.buyer_contract"]["observed"]["status"] == "lost"
    assert evidence["outcome.team_round"]["observed"]["status"] == "won"
    assert evidence["outcome.team_match"]["observed"]["status"] == "won"
    assert evidence["outcome.raw_points"]["observed"]["value"] == 50
    assert evidence["outcome.game_points"]["observed"]["value"] == 26
    assert evidence["outcome.project_points"]["observed"]["value"] == 20
    assert evidence["outcome.round_margin"]["observed"]["value"] == 26
    assert evidence["outcome.kabout"]["observed"]["achieved"] is False
    assert evidence["outcome.gahwa"]["observed"]["achieved"] is False
    assert evidence["outcome.match_target_crossing"]["observed"]["achieved"] is True
    assert all(
        row["observed"]["sourceStage"] == source_stage for row in evidence.values()
    )


def test_every_project_predicate_has_a_semantic_rejection_fixture(rules):
    counterexamples = {
        "project.exists": {"team": "north_south", "owner": "north",
            "kind": "hundred_sequence", "value": 4, "state": "awarded"},
        "project.count": {"team": "north_south", "minimum": 0, "maximum": 0,
            "state": "awarded"},
        "project.combined_value": {"team": "north_south", "minimum": 23,
            "maximum": 23, "state": "awarded"},
        "project.baloot": {"team": "north_south", "achieved": True},
        "project.overlap_count": {"owner": "north", "minimum": 4, "maximum": 4},
    }
    assert set(counterexamples) == PROJECT_PREDICATE_KINDS_V2
    for index, (kind, arguments) in enumerate(counterexamples.items()):
        context = _project_context(rules)
        lowering = ProjectOutcomePredicateLoweringV2().lower_predicate(
            _predicate(f"PRED-FALSE-PROJECT-{index}", kind, arguments), context)
        solver = _solve(context.model)
        assert not solver.boolean_value(lowering.literal), kind
        assert lowering.extract(solver)["satisfied"] is False, kind
        context.model.add(lowering.literal == 1)
        assert solver.solve(context.model) == cp_model.INFEASIBLE, kind


@pytest.mark.parametrize("source_stage", ["scoring", "full_play"])
def test_every_false_outcome_is_observable_and_cannot_qualify(source_stage):
    counterexamples = {
        "outcome.buyer_contract": {"status": "won"},
        "outcome.team_round": {"team": "north_south", "status": "won"},
        "outcome.team_match": {"team": "north_south", "status": "won"},
        "outcome.raw_points": {"team": "north_south", "minimum": 51, "maximum": 51},
        "outcome.game_points": {"team": "west_east", "minimum": 25, "maximum": 25},
        "outcome.project_points": {"team": "north_south", "minimum": 19, "maximum": 19},
        "outcome.round_margin": {"team": "west_east", "minimum": 24, "maximum": 25},
        "outcome.kabout": {"team": "west_east", "achieved": True},
        "outcome.gahwa": {"team": "west_east", "achieved": True},
        "outcome.match_target_crossing": {"team": "north_south", "achieved": True},
    }
    assert set(counterexamples) == OUTCOME_PREDICATE_KINDS_V2
    for index, (kind, arguments) in enumerate(counterexamples.items()):
        context = _outcome_context(source_stage)
        lowering = ProjectOutcomePredicateLoweringV2().lower_predicate(
            _predicate(f"PRED-FALSE-OUTCOME-{index}", kind, arguments), context)
        solver = _solve(context.model)
        assert solver.boolean_value(lowering.literal) is False, kind
        assert lowering.extract(solver)["satisfied"] is False, kind
        context.model.add(lowering.literal == 1)
        assert solver.solve(context.model) == cp_model.INFEASIBLE, kind


@pytest.mark.parametrize(
    "kind,arguments",
    [
        (
            "project.count",
            {"team": "north_south", "minimum": 2, "maximum": 1},
        ),
        (
            "outcome.raw_points",
            {"team": "north_south", "minimum": 2, "maximum": 1},
        ),
        (
            "outcome.team_round",
            {"team": "north_south", "status": "won", "extra": True},
        ),
    ],
)
def test_category_argument_validation_is_strict(kind, arguments, rules):
    context = (
        _project_context(rules)
        if kind in PROJECT_PREDICATE_KINDS_V2
        else _outcome_context()
    )

    with pytest.raises((TypeError, ValueError)):
        ProjectOutcomePredicateLoweringV2().lower_predicate(
            _predicate("PRED-INVALID", kind, arguments), context
        )


def test_context_kind_mismatch_fails_before_model_mutation(rules):
    outcome = _outcome_context()
    project = _project_context(rules)
    lowerer = ProjectOutcomePredicateLoweringV2()
    before_outcome = len(outcome.model.Proto().constraints)
    before_project = len(project.model.Proto().constraints)

    with pytest.raises(TypeError, match="project CP-SAT context"):
        lowerer.lower_predicate(
            _predicate(
                    "PRED-WRONG-PROJECT",
                "project.baloot",
                {"team": "north_south", "achieved": False},
            ),
            outcome,
        )
    with pytest.raises(TypeError, match="outcome CP-SAT context"):
        lowerer.lower_predicate(
            _predicate(
                "PRED-WRONG-OUTCOME",
                "outcome.buyer_contract",
                {"status": "lost"},
            ),
            project,
        )

    assert len(outcome.model.Proto().constraints) == before_outcome
    assert len(project.model.Proto().constraints) == before_project


def test_category_has_no_engine_runtime_or_semantic_verifier_imports():
    source = (
        Path(__file__).parents[1]
        / "src/baloot_platform/research/scenario_project_outcome_lowering_v2.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "core.engine",
        "BalootEngine",
        "HeuristicBot",
        "scenario_evidence",
        "proof_service",
        "reference_scoring",
        "play_compiler",
    ):
        assert forbidden not in source
