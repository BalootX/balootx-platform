from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import pytest
import json
from ortools.sat.python import cp_model

from baloot_platform.core.enums import Contract, Seat, Team
from baloot_platform.core.models import Project, RoundState, Trick
from baloot_platform.core.project_ir import ProjectDomainIRV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring import score_round
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.research.ortools_scoring_stage import (
    CpSatScoringInputsV1,
    OrToolsScoringStageV1,
    attach_raw_conversion_v1,
    attach_scoring_fragment_v1,
)


@pytest.mark.parametrize("outcome,expected", [
    ("buyer_makes", "north_south"), ("buyer_loses", "west_east"),
])
@pytest.mark.parametrize("contract", [Contract.SUN, Contract.HOKUM])
def test_native_configured_sun_tie_does_not_call_reference_truth(rules, monkeypatch, outcome, expected, contract):
    data = deepcopy(rules.data)
    data["scoring"]["contract"]["buyer_tie_outcome"] = outcome
    domain = ScoringDomainIRV1.from_rules(data)

    def forbidden(*args, **kwargs):
        pytest.fail("native configured tie called reference semantics")

    for method in ("tie_winner", "convert_raw", "resolve_counting_team", "multiplier"):
        monkeypatch.setattr(ScoringDomainIRV1, method, forbidden)
    raw, projects = ((65, 65), (5, 5)) if contract == Contract.SUN else ((85, 77), (10, 10))
    result = OrToolsScoringStageV1(domain, native_conversion=True).solve(
        contract=contract, buyer=Seat.NORTH, escalation="normal",
        last_escalating_team=None, raw_points=raw, project_points=projects,
        baloot_points=(0, 0), trick_winner_teams=(Team.NORTH_SOUTH, Team.WEST_EAST) * 4,
        starting_match_score=(0, 0),
    )
    assert result.status == "satisfied"
    assert result.score["roundWinner"] == (expected if contract == Contract.SUN else "north_south")


@pytest.mark.parametrize(("profile", "award"), (
    ("saudi_federation_2023_v2", [88, 0]),
    ("saudi_federation_2023_sweep", [0, 44]),
))
def test_complete_swept_project_trace_preserves_profile_boundary(root, profile, award):
    from baloot_platform.core.rule_source_catalog import RuleSourceCatalogV1
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1
    rules = RuleBook.load(root / f"configs/rules/{profile}.yaml")
    catalog = RuleSourceCatalogV1.load(root / f"configs/rules/source_catalog_{profile}.yaml", rules)
    catalog.verify_retained_sources(root)
    from baloot_platform.research.domain_compiler import BalootDomainCompilerV1
    from baloot_platform.challenge.constraint_deal_search import OrToolsCpSatDealGeneratorV1
    from baloot_platform.research.auction_compiler import OrToolsCpSatAuctionCompilerV1
    from baloot_platform.research.escalation_compiler import OrToolsCpSatEscalationCompilerV1
    from baloot_platform.research.project_compiler import OrToolsCpSatProjectCompilerV1
    from baloot_platform.research.play_compiler import OrToolsCpSatPlayCompilerV1
    plugins = [cls().descriptor.to_dict() for cls in (
        OrToolsCpSatDealGeneratorV1, OrToolsCpSatAuctionCompilerV1,
        OrToolsCpSatEscalationCompilerV1, OrToolsCpSatProjectCompilerV1,
        OrToolsCpSatPlayCompilerV1,
    )]
    proofs = BalootDomainCompilerV1._behavioral_proofs(
        rules=rules.data, plugins={item["id"]: item for item in plugins})
    assert ("rules.scoring.kabout.winner" in proofs) is (
        profile == "saudi_federation_2023_sweep")
    if profile == "saudi_federation_2023_sweep":
        from baloot_platform.config import build_bundle
        from baloot_platform.core.rule_source_catalog import load_bound_rule_source_catalog
        bundle = build_bundle(root, bundle_path="configs/bundle_ortools_research_sweep.yaml")
        assert bundle.rules.sha256 == rules.sha256
        certification = bundle.component("certification")
        assert certification["certification_eligible"] is False
        assert certification["automatic_activation"] is False
        assert load_bound_rule_source_catalog(certification, rules, root).sha256 == catalog.sha256
    prefix = ExactTracePrefixV1.build(json.loads(
        (root / "examples/research-kabout-swept-projects-regression.json").read_text(encoding="utf-8")))
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "play"}, maximum_seconds=20)
    assert len(exact.state["completedTricks"]) == 8
    assert exact.state["roundScore"]["rawCardPoints"] == [0, 130]
    assert exact.state["roundScore"]["projectPoints"] == [44, 0]
    assert exact.state["roundScore"]["awardedPoints"] == award


@pytest.mark.parametrize("contract", (Contract.SUN, Contract.HOKUM))
@pytest.mark.parametrize("sweeper", tuple(Team))
@pytest.mark.parametrize("corrected", (False, True))
def test_kabout_award_belongs_to_sweeper_despite_losing_team_projects(root, contract, sweeper, corrected):
    """Independent outcome invariant, not a copy of the reference winner formula."""
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    payload = deepcopy(rules.data)
    legacy = ScoringDomainIRV1.from_rules(payload)
    if corrected:
        payload["scoring"]["kabout"]["winner"] = "trick_sweeper"
    from baloot_platform.config import validate_profile_payload
    validate_profile_payload("rules", payload, root=root)
    domain = ScoringDomainIRV1.from_rules(payload)
    assert (domain.sha256 != legacy.sha256) is corrected
    assert ("kaboutWinner" in domain.to_dict()) is corrected
    raw, projects = [0, 0], [0, 0]
    raw[int(sweeper)] = domain.raw_card_totals[contract.value]
    projects[int(sweeper.other)] = 44
    result = OrToolsScoringStageV1(domain, native_conversion=True).solve(
        contract=contract, buyer=Seat.NORTH, escalation="normal",
        last_escalating_team=None, raw_points=raw, project_points=projects,
        baloot_points=(0, 0), trick_winner_teams=(sweeper,) * 8,
        starting_match_score=(0, 0),
    )
    assert result.status == "satisfied"
    assert result.score["kabout"] is True
    expected_winner = sweeper if corrected else sweeper.other
    assert result.score["roundWinner"] == expected_winner.name.lower()
    expected = [0, 0]
    expected[int(expected_winner)] = domain.kabout_base_game_points[contract.value] + (0 if corrected else 44)
    assert result.score["gamePoints"] == expected
    winner_seat = Seat.NORTH if sweeper == Team.NORTH_SOUTH else Seat.WEST
    loser_seat = winner_seat.next()
    state = RoundState(rules.sha256, 1, 1, Seat.EAST)
    state.contract, state.buyer, state.declarer = contract, Seat.NORTH, Seat.NORTH
    state.completed_tricks = [Trick(i, winner_seat, winner=winner_seat,
        raw_points=raw[int(sweeper)] - domain.last_trick_bonus_raw if i == 0 else 0)
        for i in range(8)]
    state.projects = [Project(loser_seat, "test", (), 44, 0, 0, scoring=True)]
    runtime = score_round(state, domain, ProjectDomainIRV1.from_rules(rules.data))
    assert runtime.awarded_points == expected
    from baloot_platform.research.reference_scoring import ReferenceScoringInterpreterV1
    from baloot_platform.research.play_compiler import OrToolsCpSatPlayCompilerV1
    inputs = dict(contract=contract, declarer=Seat.NORTH, escalation="normal",
        trick_raw_points=tuple(t.raw_points for t in state.completed_tricks),
        project_points=projects)
    reference = ReferenceScoringInterpreterV1(domain).score(
        **inputs, trick_winners=(winner_seat,) * 8)
    assert list(reference.awarded_points) == expected
    probe = OrToolsCpSatPlayCompilerV1.scoring_probe(
        domain=domain, **inputs, trick_winner_teams=(sweeper,) * 8)
    assert probe["awardedPoints"] == expected


@pytest.mark.parametrize("value", (None, True, "score_total", "unknown"))
def test_kabout_winner_primitive_rejects_unknown_values(root, value):
    from baloot_platform.config import validate_profile_payload
    payload = deepcopy(RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml").data)
    payload["scoring"]["kabout"]["winner"] = value
    with pytest.raises(ValueError):
        ScoringDomainIRV1.from_rules(payload)
    with pytest.raises(ValueError):
        RuleBook(payload)
    with pytest.raises(ValueError):
        validate_profile_payload("rules", payload, root=root)


@pytest.mark.parametrize("contract", (Contract.SUN, Contract.HOKUM, Contract.ASHKAL))
def test_native_raw_conversion_exhaustive_without_reference_calls(rules, monkeypatch, contract):
    domain = ScoringDomainIRV1.from_rules(rules.data)
    maximum = domain.raw_card_totals[contract.effective.value]
    expected = [domain.convert_raw(raw, contract) for raw in range(maximum + 1)]

    def forbidden(*args, **kwargs):
        pytest.fail("native conversion called the reference interpreter")

    monkeypatch.setattr(ScoringDomainIRV1, "convert_raw", forbidden)
    model = cp_model.CpModel()
    outputs = [attach_raw_conversion_v1(
        model, model.new_constant(raw), domain=domain, contract=contract, name=f"raw_{raw}"
    ) for raw in range(maximum + 1)]
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    assert [solver.value(value) for value in outputs] == expected
    wrong = []
    for index, (output, value) in enumerate(zip(outputs, expected, strict=True)):
        differs = model.new_bool_var(f"incorrect_conversion_{index}")
        model.add(output != value).only_enforce_if(differs)
        wrong.append(differs)
    model.add_bool_or(wrong)
    assert solver.solve(model) == cp_model.INFEASIBLE


@pytest.mark.parametrize("profile", (
    "saudi_common_v1.yaml", "saudi_federation_2023.yaml", "saudi_federation_2023_v2.yaml",
    "saudi_project_optional_unratified_v1.yaml",
))
def test_normal_award_envelope_is_implied_by_unstrengthened_native_scoring(root, profile):
    from baloot_platform.research.ortools_scoring_stage import scoring_level_id_v1

    domain = ScoringDomainIRV1.from_rules(RuleBook.load(root / "configs/rules" / profile).data)
    assert all(domain.kabout_base_game_points[contract] >= domain.card_game_point_totals[contract]
               for contract in domain.card_game_point_totals)
    model = cp_model.CpModel()
    inputs = CpSatScoringInputsV1(
        model.new_int_var(0, 1, "contract"), model.new_int_var(0, 3, "buyer"),
        model.new_constant(scoring_level_id_v1("normal")), model.new_int_var(0, 2, "last_team"),
        tuple(model.new_int_var(0, max(domain.raw_card_totals.values()), f"raw_{team}") for team in range(2)),
        tuple(model.new_int_var(0, 32767, f"projects_{team}") for team in range(2)),
        tuple(model.new_int_var(0, 32767, f"baloot_{team}") for team in range(2)),
        tuple(model.new_int_var(0, 1, f"trick_{trick}") for trick in range(8)),
        tuple(model.new_int_var(0, domain.match_target - 1, f"starting_{team}") for team in range(2)),
    )
    score = attach_scoring_fragment_v1(model, domain, inputs, source_hash="a" * 64, native_conversion=True)
    violations = []
    for team in range(2):
        for winning_bound in (True, False):
            violates = model.new_bool_var(f"violates_{team}_{winning_bound}")
            model.add(score.round_winner == team).only_enforce_if(violates)
            model.add(score.awarded_points[team] < score.total_before_award[team] if winning_bound
                else score.awarded_points[1-team] > score.total_before_award[1-team]).only_enforce_if(violates)
            violations.append(violates)
    model.add_bool_or(violations)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(model) == cp_model.INFEASIBLE


@pytest.mark.parametrize("profile", ("saudi_federation_2023_v2", "saudi_federation_2023_sweep"))
def test_sun_buyer_awarded_hundred_and_two_aces_cannot_lose(root, profile):
    """Native components of the opponent's Ace-lead counterstrategy, not recipe recognition."""
    from baloot_platform.core.cards import Card, FULL_DECK
    from baloot_platform.core.enums import Rank
    from baloot_platform.research.ortools_play_tail import OrToolsPlayTailStageV1
    rules = RuleBook.load(root / f"configs/rules/{profile}.yaml")
    play = OrToolsPlayTailStageV1(native_conversion=True).build_model(
        cards=FULL_DECK, ownership=None, contract=Contract.SUN, trump=None,
        first_leader=Seat.NORTH, play_domain=rules.play_domain,
        structural_domain=rules.structural_domain, selected_play_count=4,
        decision_prefix_length=0,
    )
    play.model.add_allowed_assignments([play.selected_play_indexes[0]],
        [(card.id,) for card in FULL_DECK if card.rank == Rank.ACE])
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    assert solver.solve(play.model) == cp_model.OPTIMAL
    play.model.add(play.winner_team_variables[0] != int(Team.NORTH_SOUTH))
    assert solver.solve(play.model) == cp_model.INFEASIBLE

    from baloot_platform.research.ortools_project_fragment import attach_project_lifecycle_fragment_v1
    project_model = cp_model.CpModel()
    owners = tuple(project_model.new_int_var(0, 3, f"owner_{i}") for i in range(32))
    projects_fragment = attach_project_lifecycle_fragment_v1(project_model,
        owner_of_card=owners,
        contract=Contract.SUN, dealer=Seat.EAST, trump=None, domain=rules.project_domain,
        name="counterstrategy_projects", tighten_value_bounds=True)
    project_model.add(projects_fragment.combined_awarded_value[0] >= 20)
    control = project_model.clone()
    for card, owner in zip(FULL_DECK, owners, strict=True):
        control.add(control.get_int_var_from_proto_index(owner.index) == int(card.suit))
    assert solver.solve(control) == cp_model.OPTIMAL
    project_model.add(projects_fragment.combined_awarded_value[1] > 0)
    assert solver.solve(project_model) == cp_model.INFEASIBLE
    assert rules.project_domain.baloot_only_hokum

    domain = ScoringDomainIRV1.from_rules(rules.data)
    model = cp_model.CpModel()
    total = domain.raw_card_totals["sun"]
    raw = tuple(model.new_int_var(0, total, f"raw_{team}") for team in range(2))
    model.add(sum(raw) == total)
    ace_points = rules.play_domain.points(Card.parse("SA"), Contract.SUN, None)
    model.add(raw[0] >= 2 * ace_points)
    projects = (model.new_int_var(20, 32767, "awarded_hundred_or_more"), model.new_constant(0))
    winners = tuple(model.new_int_var(0, 1, f"winner_{i}") for i in range(8))
    model.add(sum(winners) <= 6)  # Buyer wins at least tricks 6 and 7.
    score = attach_scoring_fragment_v1(model, domain, CpSatScoringInputsV1(
        model.new_constant(0), model.new_constant(int(Seat.NORTH)),
        model.new_int_var(0, 4, "escalation"), model.new_int_var(0, 2, "last_team"),
        raw, projects, (model.new_constant(0), model.new_constant(0)), winners,
        tuple(model.new_int_var(0, domain.match_target - 1, f"match_{team}") for team in range(2))),
        source_hash="a" * 64, native_conversion=True)
    assert solver.solve(model) == cp_model.OPTIMAL
    model.add(score.round_winner == int(Team.WEST_EAST))
    assert solver.solve(model) == cp_model.INFEASIBLE


@pytest.mark.parametrize("profile", ("saudi_federation_2023_v2", "saudi_federation_2023_sweep"))
@pytest.mark.parametrize(("shared_tricks", "raw_delta"), ((True, 1), (False, 0), (False, 1)))
@pytest.mark.parametrize("contract_id", (0, 1))
def test_native_round_win_monotone_with_fixed_scoring_context(root, profile, shared_tricks, raw_delta, contract_id):
    """Binary raw-boundary probes require monotonic wins, not equal raw rankings."""
    domain = ScoringDomainIRV1.from_rules(RuleBook.load(
        root / f"configs/rules/{profile}.yaml").data)
    model = cp_model.CpModel()
    contract = model.new_constant(contract_id)
    maximum = max(domain.raw_card_totals.values())
    total = model.new_int_var(0, maximum, "total")
    model.add_allowed_assignments([contract, total], [
        (0, domain.raw_card_totals["sun"]), (1, domain.raw_card_totals["hokum"])])
    raw = model.new_int_var(0, maximum - 1, "raw_before")
    model.add(raw < total)
    buyer = model.new_int_var(0, 3, "buyer")
    escalation = model.new_int_var(0, 4, "escalation")
    prior = model.new_int_var(0, 2, "prior")
    projects = tuple(model.new_int_var(0, 32767, f"project_{t}") for t in range(2))
    baloot = tuple(model.new_int_var(0, 2, f"baloot_{t}") for t in range(2))
    winners = tuple(model.new_int_var(0, 1, f"trick_{i}") for i in range(8))
    fragments = []
    for offset in (0, 1):
        if not shared_tricks:
            winners = tuple(model.new_int_var(0, 1, f"trick_{offset}_{i}") for i in range(8))
            model.add(sum(winners) >= 1)
            model.add(sum(winners) <= 7)
        points = tuple(model.new_int_var(0, maximum, f"raw_{offset}_{t}") for t in range(2))
        model.add(points[0] == raw + offset * raw_delta)
        model.add(points[1] == total - raw - offset * raw_delta)
        inputs = CpSatScoringInputsV1(contract, buyer, escalation, prior, points,
            projects, baloot, winners, (model.new_constant(0), model.new_constant(0)))
        fragments.append(attach_scoring_fragment_v1(
            model, domain, inputs, source_hash="a" * 64, native_conversion=True))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    assert solver.solve(model) == cp_model.OPTIMAL
    model.add(fragments[0].round_winner == 0)
    model.add(fragments[1].round_winner == 1)
    # Exhaust disjoint scoring regimes rather than branch across unrelated
    # counting sides and sweep rules. Their union is the original domain.
    for level in range(5):
        for buyer_seat in range(4):
            for minimum, maximum_tricks in (((0, 0), (1, 7), (8, 8)) if shared_tricks else ((1, 7),)):
                partition = model.clone()
                partition.add(partition.get_int_var_from_proto_index(escalation.index) == level)
                partition.add(partition.get_int_var_from_proto_index(buyer.index) == buyer_seat)
                count = sum(partition.get_int_var_from_proto_index(value.index) for value in winners)
                partition.add(count >= minimum)
                partition.add(count <= maximum_tricks)
                for counting_team in Team:
                    fixed_counting = partition.clone()
                    fixed_counting.add(fixed_counting.get_int_var_from_proto_index(
                        fragments[0].counting_team.index) == int(counting_team))
                    status = solver.solve(fixed_counting)
                    if status == cp_model.UNKNOWN:
                        # Fix one decimal band to bound rounding branches; exhaust
                        # every raw value, with the same per-solve time limit.
                        for lower in range(0, maximum, 10):
                            fixed_raw = fixed_counting.clone()
                            partition_raw = fixed_raw.get_int_var_from_proto_index(raw.index)
                            fixed_raw.add(partition_raw >= lower)
                            fixed_raw.add(partition_raw <= min(lower + 9, maximum - 1))
                            assert solver.solve(fixed_raw) == cp_model.INFEASIBLE, (
                                level, buyer_seat, minimum, maximum_tricks, counting_team, lower)
                    else:
                        assert status == cp_model.INFEASIBLE, (
                            level, buyer_seat, minimum, maximum_tricks, counting_team)


@pytest.mark.parametrize("profile", ("saudi_federation_2023_v2", "saudi_federation_2023_sweep"))
def test_sun_nonsweep_awarded_margin_is_monotone_in_raw_points(root, profile):
    """A point-margin threshold may reuse raw-boundary probes only without sweeps."""
    domain = ScoringDomainIRV1.from_rules(RuleBook.load(
        root / f"configs/rules/{profile}.yaml").data)
    model = cp_model.CpModel()
    total = domain.raw_card_totals["sun"]
    raw = model.new_int_var(0, total - 1, "raw_before")
    buyer = model.new_int_var(0, 3, "buyer")
    level = model.new_int_var(0, 4, "level")
    prior = model.new_int_var(0, 2, "prior")
    projects = tuple(model.new_int_var(0, 32767, f"projects_{t}") for t in range(2))
    margins, scores = [], []
    for offset in (0, 1):
        winners = tuple(model.new_int_var(0, 1, f"winner_{offset}_{i}") for i in range(8))
        model.add(sum(winners) >= 1)
        model.add(sum(winners) <= 7)
        points = tuple(model.new_int_var(0, total, f"raw_{offset}_{t}") for t in range(2))
        model.add(points[0] == raw + offset)
        model.add(points[1] == total - raw - offset)
        score = attach_scoring_fragment_v1(model, domain, CpSatScoringInputsV1(
            model.new_constant(0), buyer, level, prior, points, projects,
            (model.new_constant(0), model.new_constant(0)), winners,
            (model.new_constant(0), model.new_constant(0))),
            source_hash="a" * 64, native_conversion=True)
        scores.append(score)
        margins.append(score.awarded_points[0] - score.awarded_points[1])
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(model) == cp_model.OPTIMAL
    model.add(margins[0] > margins[1])
    for escalation in range(5):
        for seat in range(4):
            for before, after in ((0, 0), (1, 1), (1, 0), (0, 1)):
                partition = model.clone()
                partition.add(partition.get_int_var_from_proto_index(level.index) == escalation)
                partition.add(partition.get_int_var_from_proto_index(buyer.index) == seat)
                partition.add(partition.get_int_var_from_proto_index(scores[0].round_winner.index) == before)
                partition.add(partition.get_int_var_from_proto_index(scores[1].round_winner.index) == after)
                assert solver.solve(partition) == cp_model.INFEASIBLE, (escalation, seat, before, after)


@pytest.mark.parametrize(
    "profile",
    ("saudi_common_v1.yaml", "saudi_federation_2023.yaml",
     "saudi_federation_2023_v2.yaml", "saudi_federation_2023_sweep.yaml"),
)
@pytest.mark.parametrize("contract", (Contract.SUN, Contract.HOKUM))
@pytest.mark.parametrize("level", ("normal", "double", "triple", "four", "gahwa"))
@pytest.mark.parametrize("counting_raw", (0, 1, 65))
@pytest.mark.parametrize("native_conversion", [False, True])
def test_native_scoring_matches_independent_domain_oracle(
    root: Path,
    profile: str,
    contract: Contract,
    level: str,
    counting_raw: int,
    native_conversion: bool,
):
    rules = RuleBook.load(root / "configs/rules" / profile)
    domain = ScoringDomainIRV1.from_rules(rules.data)
    raw_total = domain.raw_card_totals[contract.value]
    counting_raw = min(counting_raw, raw_total)
    buyer = Seat.SOUTH
    last_escalating_team = Team.NORTH_SOUTH if level != "normal" else None
    counting = domain.resolve_counting_team(
        declarer_team=buyer.team,
        escalation=level,
        last_escalating_team=last_escalating_team,
    )
    raw = [0, 0]
    raw[int(counting)] = counting_raw
    raw[int(counting.other)] = raw_total - counting_raw
    projects = (100, 0)
    baloot = (2, 0)
    trick_winners = (
        (Team.NORTH_SOUTH,) * 8
        if counting_raw in {0, raw_total}
        else (Team.NORTH_SOUTH, Team.WEST_EAST) * 4
    )

    result = OrToolsScoringStageV1(domain, native_conversion=native_conversion).solve(
        contract=contract,
        buyer=buyer,
        escalation=level,
        last_escalating_team=last_escalating_team,
        raw_points=raw,
        project_points=projects,
        baloot_points=baloot,
        trick_winner_teams=trick_winners,
        starting_match_score=(140, 139),
    )

    assert result.status == "satisfied"
    assert result.solver_status == "optimal"
    assert result.score is not None
    converted = domain.convert_raw(raw[int(counting)], contract)
    card_game = [0, 0]
    card_game[int(counting)] = converted
    card_game[int(counting.other)] = domain.card_game_point_totals[contract.value] - converted
    totals = tuple(card_game[index] + projects[index] + baloot[index] for index in range(2))
    winner = (
        domain.tie_winner(contract, raw[int(counting)], counting, buyer.team)
        if totals[0] == totals[1]
        else Team.NORTH_SOUTH
        if totals[0] > totals[1]
        else Team.WEST_EAST
    )
    kabout = domain.kabout_enabled and (
        Team.NORTH_SOUTH not in trick_winners or Team.WEST_EAST not in trick_winners
    )
    if kabout and domain.kabout_winner == "trick_sweeper":
        winner = trick_winners[0]
    awarded = domain.round_award(
        contract=contract,
        winner=winner,
        declarer_team=buyer.team,
        escalation=level,
        total_before_award=totals,
        project_points=projects,
        baloot_points=baloot,
        kabout=kabout,
    )
    final, match_winner = domain.apply_match_award(
        starting_score=(140, 139),
        awarded_points=awarded,
        round_winner=winner,
        gahwa=level == "gahwa",
    )
    assert result.score["cardGamePoints"] == card_game
    assert result.score["gamePoints"] == list(awarded)
    assert result.score["roundWinner"] == winner.name.lower()
    assert result.score["kabout"] is kabout
    assert result.score["finalMatchScore"] == list(final)
    assert result.score["matchWinner"] == (
        match_winner.name.lower() if match_winner is not None else None
    )
    assert result.variable_count > 0 and result.constraint_count > 0


def test_native_scoring_rejects_non_round_inputs(root: Path):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml")
    stage = OrToolsScoringStageV1(ScoringDomainIRV1.from_rules(rules.data))
    with pytest.raises(ValueError, match="round total"):
        stage.solve(
            contract="sun",
            buyer="south",
            escalation="normal",
            last_escalating_team=None,
            raw_points=(1, 1),
            project_points=(0, 0),
            baloot_points=(0, 0),
            trick_winner_teams=(Team.NORTH_SOUTH,) * 8,
            starting_match_score=(0, 0),
        )


@pytest.mark.parametrize("required_winner", tuple(Team))
@pytest.mark.parametrize("native_conversion", [False, True])
@pytest.mark.parametrize("contract_index", [0, 1])
@pytest.mark.parametrize("require_tie", [False, True])
@pytest.mark.parametrize("escalation_index,prior_team", [(level, team) for level in range(5) for team in range(3)])
def test_native_scoring_fragment_solves_from_live_model_inputs(
    root: Path,
    required_winner: Team,
    monkeypatch, native_conversion, contract_index, require_tie, escalation_index, prior_team,
):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml")
    domain = ScoringDomainIRV1.from_rules(rules.data)
    if native_conversion:
        def forbidden(*args, **kwargs):
            pytest.fail("native scoring used interpreted conversion")
        monkeypatch.setattr(ScoringDomainIRV1, "convert_raw", forbidden)
        monkeypatch.setattr(ScoringDomainIRV1, "tie_winner", forbidden)
        monkeypatch.setattr(ScoringDomainIRV1, "resolve_counting_team", forbidden)
        monkeypatch.setattr(ScoringDomainIRV1, "multiplier", forbidden)
    model = cp_model.CpModel()

    contract = model.new_int_var(0, 1, "contract")
    buyer = model.new_int_var(0, 3, "buyer")
    escalation = model.new_int_var(0, 4, "escalation")
    last_team = model.new_int_var(0, 2, "last_team")
    raw = tuple(model.new_int_var(0, 162, f"raw_{team}") for team in range(2))
    projects = tuple(
        model.new_int_var(0, 100, f"projects_{team}") for team in range(2)
    )
    baloot = tuple(model.new_int_var(0, 2, f"baloot_{team}") for team in range(2))
    trick_winners = tuple(
        model.new_int_var(0, 1, f"trick_winner_{trick}") for trick in range(8)
    )
    starting = tuple(
        model.new_int_var(0, domain.match_target - 1, f"starting_{team}")
        for team in range(2)
    )
    inputs = CpSatScoringInputsV1(
        contract,
        buyer,
        escalation,
        last_team,
        raw,
        projects,
        baloot,
        trick_winners,
        starting,
    )
    fragment = attach_scoring_fragment_v1(
        model, domain, inputs, source_hash="0" * 64, native_conversion=native_conversion
    )
    model.add(contract == contract_index)
    model.add(buyer == int(Seat.SOUTH))
    model.add(escalation == escalation_index)
    model.add(last_team == prior_team)
    model.add(sum(projects) == 0)
    model.add(sum(baloot) == 0)
    model.add(sum(starting) == 0)
    for index, winner in enumerate(trick_winners):
        model.add(winner == index % 2)
    model.add(fragment.round_winner == int(required_winner))
    if require_tie:
        model.add(fragment.total_before_award[0] == fragment.total_before_award[1])

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    if require_tie and required_winner == Team.WEST_EAST and (
        contract_index == 0 or (escalation_index != 0 and prior_team == 1)
    ):
        # Sun's odd 13 game points require exact-five raw units; V1 awards
        # that tie to this fixture's North/South buyer. Also no tie selector
        # can award West/East when buyer and counting team are both North/South.
        assert solver.solve(model) == cp_model.INFEASIBLE
        return
    assert solver.solve(model) == cp_model.OPTIMAL
    assert sum(solver.value(value) for value in raw) == domain.raw_card_totals[("sun", "hokum")[contract_index]]
    assert solver.value(fragment.round_winner) == int(required_winner)
    assert any(len(value.proto.domain) == 2 and value.proto.domain[0] < value.proto.domain[1] for value in raw)


def test_native_scoring_honors_profile_kabout_multiplier_mutation(root: Path):
    def awarded(profile: str) -> int:
        rules = RuleBook.load(root / "configs/rules" / profile)
        result = OrToolsScoringStageV1(ScoringDomainIRV1.from_rules(rules.data)).solve(
            contract="sun",
            buyer="south",
            escalation="double",
            last_escalating_team=Team.NORTH_SOUTH,
            raw_points=(130, 0),
            project_points=(100, 0),
            baloot_points=(2, 0),
            trick_winner_teams=(Team.NORTH_SOUTH,) * 8,
            starting_match_score=(0, 0),
        )
        assert result.score is not None
        return int(result.score["gamePoints"][0])

    assert awarded("saudi_common_v1.yaml") == 290
    assert awarded("saudi_federation_2023.yaml") == 246


@pytest.mark.parametrize(
    ("profile", "expected_award"),
    (
        ("saudi_common_v1.yaml", 288),
        ("saudi_federation_2023.yaml", 244),
    ),
)
def test_kabout_multiplier_cap_mutation_matches_runtime_reference(
    root: Path,
    profile: str,
    expected_award: int,
) -> None:
    rules = RuleBook.load(root / "configs" / "rules" / profile)
    scoring = ScoringDomainIRV1.from_rules(rules.data)
    state = RoundState(rules.sha256, 1, 1, Seat.EAST)
    state.contract = Contract.SUN
    state.buyer = Seat.SOUTH
    state.declarer = Seat.SOUTH
    state.escalation = "double"
    state.escalation_history = [{"team": Team.WEST_EAST}]
    state.completed_tricks = [
        Trick(index, Seat.SOUTH, winner=Seat.SOUTH, raw_points=15)
        for index in range(8)
    ]
    state.projects = [
        Project(Seat.SOUTH, "test", (), 100, 0, 0, scoring=True)
    ]
    runtime = score_round(
        state,
        scoring,
        ProjectDomainIRV1.from_rules(rules.data),
    )
    native = OrToolsScoringStageV1(scoring).solve(
        contract=Contract.SUN,
        buyer=Seat.SOUTH,
        escalation="double",
        last_escalating_team=Team.WEST_EAST,
        raw_points=runtime.raw_card_points,
        project_points=runtime.project_points,
        baloot_points=runtime.baloot_points,
        trick_winner_teams=(Team.NORTH_SOUTH,) * 8,
        starting_match_score=(0, 0),
    )

    assert runtime.awarded_points == [expected_award, 0]
    assert native.score is not None
    assert native.score["gamePoints"] == runtime.awarded_points
