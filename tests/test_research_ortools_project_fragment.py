from __future__ import annotations

import builtins
from dataclasses import replace
from pathlib import Path

import pytest
from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, Seat, Suit, Team
from baloot_platform.core.project_ir import ProjectDeclarationActionV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.research.ortools_project_fragment import (
    OrToolsProjectEligibilityFragmentV1,
    attach_project_lifecycle_fragment_v1,
)
from baloot_platform.research.ortools_project_model import (
    OrToolsProjectLifecycleModelV1,
)
from baloot_platform.research.ortools_scoring_stage import (
    CpSatScoringInputsV1,
    attach_scoring_fragment_v1,
    attach_project_counterfactual_scoring_v1,
    extract_scoring_evidence_v1,
)


_DECK = (
    10, 13, 23, 5, 9, 0, 7, 22, 31, 6, 27, 20, 1, 12, 4, 18,
    25, 24, 28, 8, 11, 14, 16, 17, 3, 19, 29, 26, 15, 2, 21, 30,
)


def _hands(rules):
    layout = DealLayoutV1.build(
        rules.structural_domain, dealer=Seat.EAST, buyer=Seat.NORTH
    )
    return {
        seat: tuple(
            Card(_DECK[position])
            for position in layout.positions_for(seat, final=True)
        )
        for seat in Seat
    }


def _ownership_model():
    model = cp_model.CpModel()
    owners = tuple(
        model.new_int_var(0, 3, f"owner_of_card_{card_id}")
        for card_id in range(32)
    )
    for seat in Seat:
        membership = []
        for card_id, owner in enumerate(owners):
            literal = model.new_bool_var(
                f"outer_card_{card_id}_owned_by_{int(seat)}"
            )
            model.add(owner == int(seat)).only_enforce_if(literal)
            model.add(owner != int(seat)).only_enforce_if(literal.Not())
            membership.append(literal)
        model.add(sum(membership) == 8)
    return model, owners


def _solve(model):
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = 20
    assert solver.solve(model) == cp_model.OPTIMAL
    return solver


@pytest.mark.parametrize("contract", ["sun", "hokum"])
def test_project_stop_callback_preserves_completed_model(rules, contract):
    calls = []
    def keep_running():
        calls.append(None)
        return False
    results = []
    ownership = {card.id: seat for seat, cards in _hands(rules).items() for card in cards}
    for callback in (None, keep_running):
        model, owners = _ownership_model()
        fragment = attach_project_lifecycle_fragment_v1(model, owner_of_card=owners,
            contract=contract, dealer="east", trump="clubs" if contract == "hokum" else None,
            domain=rules.project_domain, fixed_card_owners=tuple(ownership[i] for i in range(32)),
            cancel_check=callback)
        results.append((str(model.proto), fragment.fragment_hash))
    assert len(calls) > 32
    assert results[0] == results[1]


@pytest.mark.parametrize("reason", ["cancelled", "maximum_seconds"])
def test_project_stop_callback_interrupts_construction(rules, reason):
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped

    model, owners = _ownership_model()
    initial_constraints = len(model.proto.constraints)
    calls = 0
    def stopped():
        nonlocal calls
        calls += 1
        if calls == 40:
            if reason == "maximum_seconds":
                raise InformationSetCompilationStopped(reason, stage="hidden_world_enumeration")
            return True
        return False
    expected = InterruptedError if reason == "cancelled" else InformationSetCompilationStopped
    with pytest.raises(expected, match=reason if reason == "maximum_seconds" else "cancelled"):
        attach_project_lifecycle_fragment_v1(model, owner_of_card=owners,
            contract="sun", dealer="east", trump=None, domain=rules.project_domain,
            cancel_check=stopped)
    assert calls == 40
    assert len(model.proto.constraints) > initial_constraints


@pytest.mark.parametrize("contract", ("sun", "hokum"))
@pytest.mark.parametrize("profile", ("saudi_common_v1.yaml", "saudi_federation_2023_v2.yaml"))
def test_eligibility_projection_preserves_native_patterns_without_lifecycle(root, profile, contract):
    rules = RuleBook.load(root / "configs/rules" / profile)
    ownership = {card.id: seat for seat, cards in _hands(rules).items() for card in cards}
    outputs, hashes, models = [], [], []
    for option in ({}, {"eligibility_only": False}, {"eligibility_only": True}):
        model, owners = _ownership_model()
        fragment = attach_project_lifecycle_fragment_v1(
            model, owner_of_card=owners, contract=contract, dealer="east",
            trump="clubs" if contract == "hokum" else None,
            domain=rules.project_domain, **option,
        )
        hashes.append(fragment.fragment_hash)
        models.append(str(model.proto))
        for card_id, owner in ownership.items():
            model.add(owners[card_id] == int(owner))
        solver = _solve(model)
        outputs.append(tuple((p.project_id, solver.boolean_value(p.literal("eligible")))
            for p in fragment.projects))
        if option.get("eligibility_only"):
            assert isinstance(fragment, OrToolsProjectEligibilityFragmentV1)
            with pytest.raises(ValueError, match="was not compiled"):
                fragment.projects[0].literal("awarded")
            with pytest.raises(ValueError, match="no declaration or award evidence"):
                fragment.extract(solver)
            for field in ("winning_team", "combined_awarded_value", "baloot_team", "declaration_actions"):
                with pytest.raises(AttributeError):
                    getattr(fragment, field)
    assert outputs[0] == outputs[1] == outputs[2]
    assert hashes[0] == hashes[1] != hashes[2]
    assert models[0] == models[1]
    assert len(models[2]) < len(models[0]) / 4


@pytest.mark.parametrize("contract", ("sun", "hokum"))
def test_partial_fixed_ownership_prunes_without_changing_project_values(rules, contract):
    ownership = {card.id: seat for seat, cards in _hands(rules).items() for card in cards}
    outputs = []
    sizes = []
    for fixed_count, compact, tight in ((0, False, False), (8, False, False),
            (32, False, False), (8, True, False), (32, True, False),
            (8, True, True), (32, True, True)):
        model, owners = _ownership_model()
        fragment = attach_project_lifecycle_fragment_v1(
            model, owner_of_card=owners, contract=contract, dealer="east",
            trump="clubs" if contract == "hokum" else None,
            domain=rules.project_domain,
            fixed_card_owners=tuple(ownership[i] if i < fixed_count else None for i in range(32)),
            compact_fixed_states=compact,
            tighten_value_bounds=tight,
        )
        for project in fragment.projects:
            if compact and any(card.id < fixed_count and ownership[card.id] != project.owner for card in project.cards):
                statuses = (project.selected, project.declarable, project.declared,
                            project.declaration_order, project.comparison_winner, project.awarded)
                assert len({variable.index for variable in statuses}) == 1
                assert all(list(variable.proto.domain) == [0, 0] and variable.Name() for variable in statuses)
        sizes.append(len(model.proto.constraints))
        for card_id, owner in ownership.items():
            model.add(owners[card_id] == int(owner))
        evidence = fragment.extract(_solve(model))
        outputs.append((evidence.projects, evidence.winning_team, evidence.combined_awarded_value))
    assert all(output == outputs[0] for output in outputs)
    assert sizes[0] > sizes[1] > sizes[2] > sizes[4]
    assert sizes[1] > sizes[3] > sizes[4]


def test_variable_ownership_fragment_matches_fixed_project_lifecycle(rules, monkeypatch):
    groups_ranked = set()

    def rank_once(group, *, key):
        # Declaration preference depends on the group, not each candidate in it.
        assert group not in groups_ranked
        groups_ranked.add(group)
        return builtins.max(group, key=key)

    monkeypatch.setattr(
        "baloot_platform.research.ortools_project_fragment.max", rank_once, raising=False,
    )
    hands = _hands(rules)
    expected_owner = {
        card.id: seat for seat, cards in hands.items() for card in cards
    }
    fixed = OrToolsProjectLifecycleModelV1().solve(
        hands=hands,
        contract="sun",
        dealer="east",
        trump=None,
        domain=rules.project_domain,
    )
    model, owners = _ownership_model()
    for card_id, owner in enumerate(owners):
        model.add(owner == int(expected_owner[card_id]))
    fragment = attach_project_lifecycle_fragment_v1(
        model,
        owner_of_card=owners,
        contract="sun",
        dealer="east",
        trump=None,
        domain=rules.project_domain,
    )
    observed = fragment.extract(_solve(model))
    assert fragment.declaration_actions is None
    with pytest.raises(ValueError, match="no native declaration action variables"):
        fragment.extract_declaration_decision(None, Seat.NORTH)
    assert groups_ranked

    fixed_by_id = {project.project_id: project for project in fixed.projects}
    assert observed.winning_team == fixed.winning_team
    assert observed.combined_awarded_value == fixed.combined_awarded_value
    assert set(project.project_id for project in observed.projects) == set(fixed_by_id)
    for project in observed.projects:
        expected = fixed_by_id[project.project_id]
        assert (
            project.selected,
            project.declared,
            project.comparison_winner,
            project.awarded,
        ) == (
            expected.selected,
            expected.declared,
            expected.comparison_winner,
            expected.awarded,
        )


def test_variable_fragment_counts_baloot_inside_official_hundred(root: Path):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    north = tuple(
        Card.parse(code)
        for code in ("HA", "D9", "D10", "DJ", "DQ", "DK", "DA", "CK")
    )
    remainder = tuple(card for card in map(Card, range(32)) if card not in north)
    hands = {
        Seat.NORTH: north,
        Seat.WEST: remainder[:8],
        Seat.SOUTH: remainder[8:16],
        Seat.EAST: remainder[16:],
    }
    expected_owner = {
        card.id: seat for seat, cards in hands.items() for card in cards
    }
    model, owners = _ownership_model()
    for card_id, owner in enumerate(owners):
        model.add(owner == int(expected_owner[card_id]))
    fragment = attach_project_lifecycle_fragment_v1(
        model,
        owner_of_card=owners,
        contract="hokum",
        dealer="east",
        trump="diamonds",
        domain=rules.project_domain,
    )

    observed = fragment.extract(_solve(model))

    assert observed.baloot_points == (rules.project_domain.baloot_value, 0)
    assert any(
        project.owner == Seat.NORTH
        and project.kind == "hundred_sequence"
        and project.awarded
        for project in observed.projects
    )


def test_fixed_owner_elimination_preserves_complete_project_evidence(rules):
    hands = _hands(rules)
    expected_owner = tuple(
        next(seat for seat, cards in hands.items() if Card(card_id) in cards)
        for card_id in range(32)
    )

    def solve(*, eliminate: bool):
        model, owners = _ownership_model()
        for card_id, owner in enumerate(owners):
            model.add(owner == int(expected_owner[card_id]))
        fragment = attach_project_lifecycle_fragment_v1(
            model,
            owner_of_card=owners,
            fixed_card_owners=expected_owner if eliminate else None,
            contract="sun",
            dealer="east",
            trump=None,
            domain=rules.project_domain,
        )
        return fragment.extract(_solve(model))

    full = solve(eliminate=False)
    reduced = solve(eliminate=True)
    assert [project.to_dict() for project in reduced.projects] == [
        project.to_dict() for project in full.projects
    ]
    assert reduced.winning_team == full.winning_team
    assert reduced.combined_awarded_value == full.combined_awarded_value


def test_awarded_buyer_hundred_forces_variable_sequence_ownership(rules):
    hands = _hands(rules)
    expected_owner = {
        card.id: seat for seat, cards in hands.items() for card in cards
    }
    target = {Card.parse(code).id for code in ("H9", "H10", "HJ", "HQ", "HK")}
    west_variable = {
        card.id for card in hands[Seat.WEST] if card.id not in target
    }
    variable_cards = target | set(sorted(west_variable)[:5])

    model, owners = _ownership_model()
    for card_id, owner in enumerate(owners):
        if card_id not in variable_cards:
            model.add(owner == int(expected_owner[card_id]))
    fragment = attach_project_lifecycle_fragment_v1(
        model,
        owner_of_card=owners,
        contract="sun",
        dealer="east",
        trump=None,
        domain=rules.project_domain,
    )
    buyer = model.new_int_var(0, 3, "buyer")
    model.add(buyer == int(Seat.NORTH))
    constraint = fragment.constrain_buyer_team_project(
        model,
        buyer_seat=buyer,
        states=("eligible", "declarable", "awarded"),
        kind="hundred_sequence",
        value=20,
        minimum=1,
        constraint_id="buyer_awarded_hundred",
    )
    solver = _solve(model)
    evidence = fragment.extract(solver)
    matching = [
        project
        for project in evidence.projects
        if project.owner.team == Seat.NORTH.team
        and project.kind == "hundred_sequence"
        and project.value == 20
        and project.eligible
        and project.declarable
        and project.awarded
    ]

    assert solver.value(constraint.count) >= 1
    assert matching
    assert any(set(card.id for card in project.cards) == target for project in matching)
    assert {evidence.card_owners[card_id] for card_id in target} == {Seat.NORTH}


def test_variable_project_fragment_has_no_runtime_research_imports():
    source = (
        Path(__file__).parents[1]
        / "src/baloot_platform/research/ortools_project_fragment.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "core.engine",
        "core.projects",
        "project_compiler",
        "reference_project",
        "HeuristicBot",
        "proof_service",
        "RuleBook",
    ):
        assert forbidden not in source


def test_optional_project_fragment_keeps_pass_and_declare_solver_choices(
    root: Path,
):
    optional = RuleBook.load(
        root / "configs/rules/saudi_project_optional_unratified_v1.yaml"
    )
    hands = _hands(optional)
    expected_owner = {
        card.id: seat for seat, cards in hands.items() for card in cards
    }

    def solve(*, require_declaration: bool):
        model, owners = _ownership_model()
        for card_id, owner in enumerate(owners):
            model.add(owner == int(expected_owner[card_id]))
        fragment = attach_project_lifecycle_fragment_v1(
            model,
            owner_of_card=owners,
            contract="sun",
            dealer="east",
            trump=None,
            domain=optional.project_domain,
        )
        declared = [project.declared for project in fragment.projects]
        model.add(sum(declared) >= 1 if require_declaration else sum(declared) == 0)
        return fragment.extract(_solve(model))

    passed = solve(require_declaration=False)
    declared = solve(require_declaration=True)

    assert not any(project.declared for project in passed.projects)
    assert passed.winning_team is None
    assert passed.combined_awarded_value == (0, 0)
    assert any(project.declared for project in declared.projects)
    assert declared.winning_team is not None


def test_optional_hundred_declaration_changes_native_round_winner(root: Path):
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    owners_by_card = {
        card.id: seat for seat, cards in _hands(rules).items() for card in cards
    }
    target_cards = tuple(card.id for card in next(
        project
        for project in rules.project_domain.candidates(
            Seat.NORTH, _hands(rules)[Seat.NORTH], Contract.SUN
        )
        if project.kind == "hundred_sequence"
    ).cards)
    scoring = ScoringDomainIRV1.from_rules(rules.data)

    def solve(declare: bool):
        # Fixed raw/trick inputs isolate native project-to-score causality;
        # full play reachability and strategy still belong to the proof graph.
        model, owners = _ownership_model()
        fragment = attach_project_lifecycle_fragment_v1(
            model,
            owner_of_card=owners,
            fixed_card_owners=tuple(owners_by_card[index] for index in range(32)),
            contract="sun",
            dealer="east",
            trump=None,
            domain=rules.project_domain,
            compact_fixed_states=True,
            tighten_value_bounds=True,
        )
        for project in fragment.projects:
            model.add(
                project.declared == int(
                    declare
                    and project.owner == Seat.NORTH
                    and project.kind == "hundred_sequence"
                    and tuple(card.id for card in project.cards) == target_cards
                )
            )
        native_score = attach_scoring_fragment_v1(
            model,
            scoring,
            CpSatScoringInputsV1(
                contract=model.new_constant(0),
                buyer_seat=model.new_constant(int(Seat.NORTH)),
                escalation=model.new_constant(0),
                last_escalating_team=model.new_constant(2),
                raw_points=(model.new_constant(60), model.new_constant(70)),
                project_points=fragment.combined_awarded_value,
                baloot_points=(model.new_constant(0), model.new_constant(0)),
                trick_winner_teams=tuple(model.new_constant(index % 2) for index in range(8)),
                starting_match_score=(model.new_constant(0), model.new_constant(0)),
            ),
            source_hash=rules.sha256,
            native_conversion=True,
        )
        counterfactual = attach_project_counterfactual_scoring_v1(native_score, scoring)
        assert counterfactual.inputs.raw_points == native_score.inputs.raw_points
        assert counterfactual.inputs.baloot_points == native_score.inputs.baloot_points
        assert counterfactual.inputs.trick_winner_teams == native_score.inputs.trick_winner_teams
        assert counterfactual.outcome_context.sha256 != native_score.outcome_context.sha256
        assert counterfactual.round_winner.name != native_score.round_winner.name
        wrong_domain = replace(native_score, domain_ir_hash="0" * 64)
        variable_count = len(model.proto.variables)
        with pytest.raises(ValueError, match="preserve the scoring domain"):
            attach_project_counterfactual_scoring_v1(wrong_domain, scoring)
        assert len(model.proto.variables) == variable_count
        solved = _solve(model)
        with pytest.raises(ValueError, match="domain differs"):
            extract_scoring_evidence_v1(solved, wrong_domain, scoring)
        assert Team(solved.value(counterfactual.round_winner)) == Team.WEST_EAST
        assert [solved.value(item) for item in counterfactual.inputs.project_points] == [0, 0]
        return fragment.extract(solved), Team(solved.value(native_score.round_winner))

    passed, pass_winner = solve(False)
    declared, declare_winner = solve(True)
    assert passed.combined_awarded_value == (0, 0)
    assert declared.combined_awarded_value == (24, 0)
    assert any(
        project.owner == Seat.NORTH
        and project.kind == "hundred_sequence"
        and tuple(card.id for card in project.cards) == target_cards
        and project.awarded
        for project in declared.projects
    )
    assert pass_winner == Team.WEST_EAST
    assert declare_winner == Team.NORTH_SOUTH


@pytest.mark.parametrize("optional", [False, True])
@pytest.mark.parametrize("contract", [Contract.SUN, Contract.HOKUM])
def test_native_declaration_choices_match_every_reference_prefix(root, rules, optional, contract):
    if optional:
        rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    domain, dealer = rules.project_domain, Seat.EAST
    trump = Suit.HEARTS if contract == Contract.HOKUM else None
    hands = _hands(rules)
    model, owners = _ownership_model()
    fragment = attach_project_lifecycle_fragment_v1(model, owner_of_card=owners,
        contract=contract, dealer=dealer, trump=trump, domain=domain,
        include_declaration_actions=True)
    for seat, cards in hands.items():
        for card in cards:
            model.add(owners[card.id] == int(seat))
    pending = [domain.declaration_state(domain.detect(hands, contract, dealer), dealer=dealer)]
    visited, choices, optional_choices = 0, set(), 0
    while pending:
        state = pending.pop()
        if state.complete:
            continue
        current = model.clone()
        for action in state.actions:
            for project in fragment.projects:
                if project.owner == action.actor:
                    current.add(project.declared == int(project.project_id == action.project_id))
        actor = state.current_actor
        expected = domain.legal_declaration_actions(state, trump_suit=trump)
        literals = fragment.declaration_actions[int(actor)]
        solved = _solve(current)
        observed = {code for code, literal in literals.items() if solved.boolean_value(literal)}
        assert observed == {action.code for action in expected}
        decision = fragment.extract_declaration_decision(solved, actor)
        assert set(decision["legalActions"]) == observed
        assert decision["selectedAction"] in observed
        choices.update(action.action for action in expected)
        visited += 1
        if optional:
            # The incumbent may announce the unique legal project, not merely
            # an eligible project blocked by an earlier public declaration.
            unique = current.clone()
            unique.add(sum(literal for code, literal in literals.items() if code != "pass") >= 2)
            uniqueness_solver = cp_model.CpSolver()
            uniqueness_solver.parameters.num_search_workers = 1
            assert uniqueness_solver.solve(unique) == cp_model.INFEASIBLE
            announce = current.clone()
            for project in fragment.projects:
                if project.owner == actor:
                    code = ProjectDeclarationActionV1(actor, "declare", project.project_id).code
                    announce.add(project.declared == literals[code])
            announced = _solve(announce)
            selected = fragment.extract_declaration_decision(announced, actor)["selectedAction"]
            assert selected == next((code for code in observed if code != "pass"), "pass")
            # Production strategy uses the same native legal literals and leaves
            # every recorded actor untouched, even after suppressing declarations.
            native_policy = current.clone()
            fragment.constrain_native_visible_policy(native_policy, team=actor.team,
                policy_id="full_hand_project_max_value_v1",
                skipped_actors=[seat for seat in Seat if seat != actor])
            native_selected = fragment.extract_declaration_decision(_solve(native_policy), actor)["selectedAction"]
            assert native_selected == selected
            # Later optional choices, including this actor choosing pass, cannot
            # change which declarations were available before that choice.
            for project in fragment.projects:
                if project.owner not in {action.actor for action in state.actions}:
                    current.add(project.declared == 0)
            passed = _solve(current)
            assert {code for code, literal in literals.items() if passed.boolean_value(literal)} == observed
            assert fragment.extract_declaration_decision(passed, actor)["selectedAction"] == "pass"
            optional_choices += int(len(expected) > 1)
        for action in expected:
            next_state, _ = domain.apply_declaration_action(state, action, trump_suit=trump)
            pending.append(next_state)
    assert visited >= 4 and "declare" in choices
    assert optional_choices > 0 if optional else visited == 4


def test_full_hand_project_policy_handles_extension_of_auction_sequence(root):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    hands = {
        Seat.NORTH: "S8 S9 S10 SJ SQ SK SA H7".split(),
        Seat.WEST: "S7 H8 H9 H10 HJ HQ HK HA".split(),
        Seat.SOUTH: "D7 D8 D9 D10 DJ DQ DK DA".split(),
        Seat.EAST: "C7 C8 C9 C10 CJ CQ CK CA".split(),
    }
    ownership = {Card.parse(code).id: seat for seat, cards in hands.items() for code in cards}
    model, owners = _ownership_model()
    for card_id, seat in ownership.items():
        model.add(owners[card_id] == int(seat))
    fragment = attach_project_lifecycle_fragment_v1(model, owner_of_card=owners,
        contract=Contract.SUN, dealer=Seat.EAST, trump=None, domain=rules.project_domain,
        include_declaration_actions=True, fixed_card_owners=tuple(ownership[i] for i in range(32)))
    auction_cards = tuple(Card.parse(code) for code in hands[Seat.NORTH][:5])
    fixed_action = fragment.known_project_policy_actions(
        "known_project_max_value_v1", Seat.NORTH, auction_cards)["north"]
    invalid = model.clone()
    invalid.add(fragment.declaration_actions[int(Seat.NORTH)].get(fixed_action, 0) == 1)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 20
    assert solver.solve(invalid) == cp_model.INFEASIBLE
    fragment.constrain_native_visible_policy(model, team=Team.NORTH_SOUTH,
        policy_id="full_hand_project_max_value_v1")
    decision = fragment.extract_declaration_decision(_solve(model), Seat.NORTH)
    assert decision["selectedAction"] in decision["legalActions"]
    assert decision["selectedAction"] != fixed_action
    assert decision["selectedAction"].startswith("declare:")


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_project_action_variable_flag_rejects_coercion_before_mutation(rules, value):
    model, owners = _ownership_model()
    original = str(model.proto)
    with pytest.raises(TypeError, match="include_declaration_actions must be bool"):
        attach_project_lifecycle_fragment_v1(model, owner_of_card=owners,
            contract="sun", dealer="east", trump=None, domain=rules.project_domain,
            include_declaration_actions=value)
    assert str(model.proto) == original
