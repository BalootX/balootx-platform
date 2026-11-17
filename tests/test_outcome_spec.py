from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.research.outcome_spec import compile_outcome_spec, load_outcome_spec
from baloot_platform.research.scenario_predicate_lowering_v2 import (
    CpSatActionVariablesV1, CpSatTrickVariablesV1, CpSatPredicateModelContextV1,
)


@pytest.fixture
def v7(root):
    return RuleBook.load(root / "configs/rules/saudi_federation_2023_v7.yaml")


def spec(**kwargs):
    return dict(schemaVersion=1, rulesProfile="saudi_federation_2023_v7", mode="sun", **kwargs)


def test_preset_and_named_card_compilation_structures(v7):
    from baloot_platform.research.outcome_spec import PRESETS
    requests = {kind: spec(category=kind) for kind in sorted(PRESETS)}
    requests["declaration"]["declarations"] = [{"category": "100"}]
    requests["baloot"] = {**spec(category="declaration", declarations=[{"category": "baloot"}]),
                          "mode": "hokum", "trumpSuit": "hearts"}
    for name, mode, trump, winning, beaten in [
        ("ten-over-ace", "hokum", "hearts", "H10", "CA"),
        ("seven-over-ace", "sun", None, "C7", "HA"),
    ]:
        requests[name] = {**spec(outcome={"winner": "north_south", "decidingTrick": {
            "index": 8, "winningCard": winning, "beatenCard": beaten}}), "mode": mode}
        if trump:
            requests[name]["trumpSuit"] = trump
    for name, raw in requests.items():
        compiled = compile_outcome_spec(raw, v7)
        assert compiled.scenario.origin.kind == "search"
        assert all(p.hardness == "hard" for p in compiled.scenario.predicates())
        assert compiled.rules_hash == v7.sha256
        if name in {"big_swing", "comeback", "last_trick_reversal"}:
            assert len(compiled.trajectory) == 1
        if name in {"comeback", "last_trick_reversal", "kaboot", "seven-over-ace", "ten-over-ace"}:
            assert next(p.arguments for p in compiled.scenario.predicates()
                        if p.kind == "outcome.team_round") == dict(team="north_south", status="won")
        if name == "baloot":
            assert next(p.arguments for p in compiled.scenario.predicates()
                        if p.kind == "project.baloot") == dict(team="north_south", achieved=True)


@pytest.mark.parametrize("category,expected", [
    ("big_swing", ("north_south", 7, 8, 20, 130)),
    ("comeback", ("north_south", 0, 4, -112, -1)),
    ("last_trick_reversal", ("north_south", 0, 7, -120, -1)),
])
def test_trajectory_presets_are_native_card_point_constraints(v7, category, expected):
    compiled = compile_outcome_spec(spec(category=category), v7)
    row = compiled.trajectory[0]
    assert (row.team, row.start, row.stop, row.minimum, row.maximum) == expected
    if category != "big_swing":
        assert len(compiled.trajectory) == 1
        assert any(p.kind == "outcome.team_round" for p in compiled.scenario.predicates())


@pytest.mark.parametrize("category,kind", [("near_kaboot", "play.trick_winner_matches"),
                                          ("kaboot", "outcome.kabout"),
                                          ("declaration", "project.exists")])
def test_remaining_presets_lower_to_existing_v4(v7, category, kind):
    raw = spec(category=category)
    if category == "declaration":
        raw["declarations"] = [{"category": "100", "team": "north_south"}]
    compiled = compile_outcome_spec(raw, v7)
    assert any(p.kind == kind for p in compiled.scenario.predicates())
    if category == "near_kaboot":
        count = next(e for e in compiled.scenario.constraints.children if e.op == "count")
        assert count.minimum == count.maximum == 7
        assert len(count.children) == 8
    if category == "declaration":
        projects = [p for p in compiled.scenario.predicates() if p.kind == "project.exists"]
        assert {p.arguments["kind"] for p in projects} == {"hundred_sequence", "hundred_four_kind"}
        assert {p.arguments["value"] for p in projects} == {20}
        assert all(p.arguments["state"] == "awarded" for p in projects)


@pytest.mark.parametrize("mode,trump,winning,beaten", [
    ("hokum", "hearts", "H10", "CA"), ("sun", None, "C7", "HA"),
])
def test_two_named_card_specs_bind_winner_and_opponents_card(v7, mode, trump, winning, beaten):
    raw = spec(outcome={"winner": "north_south", "decidingTrick": {
        "index": 8, "winningTeam": "north_south", "winningCard": winning, "beatenCard": beaten}})
    raw["mode"] = mode
    if trump:
        raw["trumpSuit"] = trump
    compiled = compile_outcome_spec(raw, v7)
    winner = next(p for p in compiled.scenario.predicates() if p.kind == "play.trick_winner_matches")
    loser = next(p for p in compiled.scenario.predicates() if p.kind == "play.trick_card_count")
    assert winner.arguments == dict(trickIndex=7, team="north_south", selector={"cards": [Card.parse(winning).id]})
    assert loser.arguments == dict(trickIndex=7, minimum=1, maximum=1, selector={
        "cards": [Card.parse(beaten).id], "owner": {"relation": "team", "team": "west_east"}})


@pytest.mark.parametrize("category,value", [("sira", 4), ("50", 10), ("400", 40)])
def test_project_display_categories_use_rulebook_values(v7, category, value):
    compiled = compile_outcome_spec(spec(declarations=[{"category": category}]), v7)
    assert next(p for p in compiled.scenario.predicates() if p.kind == "project.exists").arguments["value"] == value


def test_baloot_is_hokum_only_and_policy_mix_is_explicit(v7):
    raw = spec(declarations=[{"category": "baloot"}], policies={"default": "informed",
        "teams": {"west_east": "optimal"}, "seats": {"east": "informed"}})
    with pytest.raises(ValueError, match="Baloot requires Hokum"):
        compile_outcome_spec(raw, v7)
    raw.update(mode="hokum", trumpSuit="hearts")
    compiled = compile_outcome_spec(raw, v7)
    assert dict(compiled.seat_policies) == dict(north="informed", west="optimal", south="informed", east="informed")
    assert any(p.kind == "project.baloot" for p in compiled.scenario.predicates())
    raw["policies"]["default"] = "strong"
    with pytest.raises(ValueError, match="policy must be"):
        compile_outcome_spec(raw, v7)


def test_composed_ashkal_score_margin_and_declaration(v7):
    raw = spec(outcome={"winner": "west_east", "score": {"north_south": 0, "west_east": 40},
                        "awardedMargin": {"minimum": 40, "maximum": 40}},
               declarations=[{"category": "sira", "owner": "west", "team": "west_east", "state": "declared"}])
    raw["mode"] = "ashkal"
    compiled = compile_outcome_spec(raw, v7)
    points = [p.arguments for p in compiled.scenario.predicates() if p.kind == "outcome.game_points"]
    assert points == [dict(team="north_south", minimum=0, maximum=0), dict(team="west_east", minimum=40, maximum=40)]
    margin = next(p for p in compiled.scenario.predicates() if p.kind == "outcome.round_margin")
    assert margin.arguments == dict(team="west_east", minimum=40, maximum=40)
    project = next(p for p in compiled.scenario.predicates() if p.kind == "project.exists")
    assert project.arguments == dict(team="west_east", owner="west", kind="siri", value=4, state="declared")


def _native_variables(points, winners):
    model = cp_model.CpModel()
    tricks = {}
    for i, side in enumerate(winners):
        actor = model.new_constant(int(Seat.NORTH if side == 0 else Seat.WEST))
        actions = tuple(CpSatActionVariablesV1(f"PLAY-{4*i+j}", model.new_constant(4*i+j),
            actor, actor, model.new_constant(0), model.new_constant(1)) for j in range(4))
        tricks[i] = CpSatTrickVariablesV1(i, actions, actions[0])
    context = CpSatPredicateModelContextV1(model, "outcome-test", tricks, (), model.new_constant(1))
    totals, prefixes = [0, 0], [(model.new_constant(0), model.new_constant(0))]
    for index, (point, side) in enumerate(zip(points, winners, strict=True)):
        totals[side] += point + (10 if index == 7 else 0)
        prefixes.append(tuple(model.new_constant(p) for p in totals))
    return SimpleNamespace(model=model, predicate_context=context, raw_team_prefix_variables=tuple(prefixes),
        trick_point_variables=tuple(model.new_constant(p) for p in points),
        winner_team_variables=tuple(model.new_constant(w) for w in winners))


@pytest.mark.parametrize("category,points,winners,accepted", [
    ("big_swing", [1]*7+[20], [0]*8, True),
    ("big_swing", [1]*7+[9], [0]*8, False),
    ("comeback", [2]*4+[3]*4, [1]*4+[0]*4, True),
    ("comeback", [2]*4+[3]*4, [0]*8, False),
    ("last_trick_reversal", [2]*7+[20], [1]*7+[0], True),
    ("last_trick_reversal", [3]*7+[20], [1]*7+[0], True),
    ("last_trick_reversal", [2]*7+[0], [1]*8, False),
    ("near_kaboot", [2]*8, [0]*7+[1], True),
    ("near_kaboot", [2]*8, [0]*8, False),
])
def test_native_trajectory_and_count_lowering(v7, category, points, winners, accepted):
    compiled = compile_outcome_spec(spec(category=category), v7)
    build = _native_variables(points, winners)
    compiled.constrain_play_model(build)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_deterministic_time = 1
    assert solver.solve(build.model) == (cp_model.OPTIMAL if accepted else cp_model.INFEASIBLE)


def test_trajectory_requires_full_horizon_and_does_not_silently_drop(v7):
    compiled = compile_outcome_spec(spec(category="comeback"), v7)
    with pytest.raises(ValueError, match="eight-trick"):
        compiled.constrain_play_model(_native_variables([1], [0]))


def test_facade_binds_real_native_points_and_named_card_to_engine_round(v7):
    from baloot_platform.core.bots import BotProfile, HeuristicBot
    from baloot_platform.core.engine import BalootEngine
    from baloot_platform.research.ortools_play_tail import OrToolsPlayTailStageV1, TailPrefixActionV1

    engine = BalootEngine(v7)
    state = engine.create_round(seed=20261025, dealer=Seat.WEST)
    engine.shuffle_and_initial_deal(state)
    assert engine.bid_and_complete_deal(state)
    engine.prepare_projects(state)
    ownership = {card.code: seat for seat in Seat for card in state.hand(seat)}
    leader = state.current_player
    bot = HeuristicBot(engine, BotProfile(20262024))
    while state.current_player is not None:
        engine.play_card(state, bot.choose(state))
    trick = max(state.completed_tricks, key=lambda item: item.raw_points)
    winning_card = next(play.card for play in trick.plays if play.seat == trick.winner)
    beaten_card = next(play.card for play in trick.plays if play.seat.team != trick.winner.team)
    raw = spec(category="big_swing", team=trick.winner.team.name.lower(), outcome={
        "swing": {"trickIndex": trick.index + 1, "minimum": trick.raw_points},
        "decidingTrick": {"index": trick.index + 1, "winningCard": winning_card.code,
                          "beatenCard": beaten_card.code}})
    raw["mode"] = state.contract.value
    if state.trump_suit is not None:
        raw["trumpSuit"] = state.trump_suit.name.lower()
    compiled = compile_outcome_spec(raw, v7)
    build = OrToolsPlayTailStageV1(native_conversion=True).build_model(
        cards=list(ownership), ownership=ownership, contract=state.contract, trump=state.trump_suit,
        first_leader=leader, play_domain=v7.play_domain, structural_domain=v7.structural_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(v7.data),
        prefix=tuple(TailPrefixActionV1(play.seat, play.card) for play in state.played_cards))
    compiled.constrain_play_model(build)
    result = build.solve(maximum_time_seconds=2)
    assert result.trick_raw_points == tuple(t.raw_points for t in state.completed_tricks)


def test_last_trick_bonus_reverses_a_five_point_deficit(v7):
    compiled = compile_outcome_spec(spec(category="last_trick_reversal"), v7)
    # Boundary-unit golden; the real v7 deck has even card totals, so a full
    # zero-card-point final trick has an even (not five-point) preceding margin.
    build = _native_variables([55, 60, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0])
    compiled.constrain_play_model(build)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(build.model) == cp_model.OPTIMAL
    before, after = build.raw_team_prefix_variables[7], build.raw_team_prefix_variables[8]
    assert solver.value(before[0]-before[1]) == -5
    assert solver.value(after[0]-after[1]) == 5


def test_real_last_trick_reversal_wins_zero_point_trick_from_four_behind(v7):
    from baloot_platform.core.cards import FULL_DECK
    from baloot_platform.core.engine import BalootEngine
    from baloot_platform.core.serialization import replay_trusted_context
    from baloot_platform.research.ortools_play_tail import OrToolsPlayTailStageV1, TailPrefixActionV1

    # A full native construction frozen as a replayable golden, not aggregate points.
    line = ["DK", "DJ", "D9", "S8", "C10", "C9", "CJ", "CK", "CA", "CQ", "H8", "C8",
            "SK", "SA", "D10", "SJ", "H10", "DA", "HA", "HQ", "S9", "S10", "HJ", "D8",
            "SQ", "HK", "DQ", "H9", "S7", "H7", "D7", "C7"]
    leaders = [Seat.NORTH]*4+[Seat.WEST, Seat.EAST, Seat.NORTH, Seat.NORTH]
    plays = tuple(TailPrefixActionV1(Seat((int(leaders[i//4])+i%4)%4), Card.parse(card))
                  for i, card in enumerate(line))
    ownership = {play.card.code: play.actor for play in plays}
    hands = {seat.name.lower(): [card for card, owner in ownership.items() if owner == seat] for seat in Seat}
    assert len(ownership) == len(FULL_DECK)
    compiled = compile_outcome_spec(spec(category="last_trick_reversal"), v7)
    build = OrToolsPlayTailStageV1(native_conversion=True).build_model(cards=list(ownership), ownership=ownership,
        contract="sun", trump=None, first_leader="north", play_domain=v7.play_domain,
        structural_domain=v7.structural_domain, scoring_domain=ScoringDomainIRV1.from_rules(v7.data), prefix=plays)
    compiled.constrain_play_model(build)
    before = build.raw_team_prefix_variables[7]
    build.model.add(before[0] == 58)
    build.model.add(before[1] == 62)
    build.model.add(build.trick_point_variables[7] == 0)
    result = build.solve(maximum_time_seconds=5)
    state = replay_trusted_context(dict(dealer="east", contract="sun", buyer="north", firstLeader="north",
        initialHands=hands, completedTricks=[], currentTrick=None), v7)
    engine = BalootEngine(v7)
    for action in result.plays[:28]:
        engine.play_card(state, action.card, ekka=action.ekka)
    raw_team = [sum(t.raw_points for t in state.completed_tricks if int(t.winner.team) == side) for side in (0, 1)]
    assert raw_team == [58, 62]
    for action in result.plays[28:]:
        engine.play_card(state, action.card, ekka=action.ekka)
    assert state.completed_tricks[7].raw_points == 0
    assert state.completed_tricks[7].winner.team.value == 0
    assert list(state.round_score.raw_card_points) == [68, 62]
    assert result.raw_team_points == tuple(state.round_score.raw_card_points)
    assert state.round_score.winner.value == 0


def test_real_kaboot_native_play_and_outcome_solve(v7):
    from baloot_platform.core.cards import FULL_DECK
    from baloot_platform.research.ortools_play_tail import OrToolsPlayTailStageV1
    from baloot_platform.research.ortools_scoring_stage import attach_scoring_fragment_v1, CpSatScoringInputsV1
    from baloot_platform.research.scenario_predicate_manifest_v2 import shared_scenario_predicate_catalog_lowering_manifest_v2
    from baloot_platform.research.scenario_predicate_lowering_v2 import reify_scenario_expression

    compiled = compile_outcome_spec(spec(category="kaboot"), v7)
    scoring = ScoringDomainIRV1.from_rules(v7.data)
    ownership = {card.code: Seat(int(card.suit)) for card in FULL_DECK}
    build = OrToolsPlayTailStageV1(native_conversion=True).build_model(cards=list(ownership), ownership=ownership,
        contract="sun", trump=None, first_leader="north", play_domain=v7.play_domain,
        structural_domain=v7.structural_domain, scoring_domain=scoring)
    compiled.constrain_play_model(build)
    model = build.model
    fragment = attach_scoring_fragment_v1(model, scoring, CpSatScoringInputsV1(
        contract=model.new_constant(0), buyer_seat=model.new_constant(0), escalation=model.new_constant(0),
        last_escalating_team=model.new_constant(2), raw_points=build.raw_team_variables,
        project_points=(model.new_constant(0), model.new_constant(0)),
        baloot_points=(model.new_constant(0), model.new_constant(0)),
        trick_winner_teams=build.winner_team_variables,
        starting_match_score=(model.new_constant(0), model.new_constant(0))),
        source_hash=compiled.sha256, native_conversion=True)
    manifest = shared_scenario_predicate_catalog_lowering_manifest_v2()
    literals = {p.predicate_id: manifest.lower_predicate(p, fragment.outcome_context).literal
                for p in compiled.scenario.outcome.predicates()}
    model.add(reify_scenario_expression(model, compiled.scenario.outcome, literals) == 1)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(model) == cp_model.OPTIMAL
    assert all(solver.value(v) == 0 for v in build.winner_team_variables)
    assert solver.value(build.raw_team_variables[0]) == scoring.raw_card_totals["sun"]
    assert solver.value(fragment.outcome_context.kabout) == 1


@pytest.mark.parametrize("change,reason", [
    ({"team": "north_south", "outcome": {"winner": "west_east"}}, "contradict"),
    ({"category": "kaboot", "outcome": {"decidingTrick": {"index": 8, "winningTeam": "west_east"}}}, "kaboot contradicts"),
    ({"outcome": {"score": {"north_south": 10000}}}, "rulebook bound"),
    ({"category": "comeback", "outcome": {"comeback": {"afterTrick": 7, "minimumDeficit": 121}}}, "reachable bound"),
])
def test_contradictions_rejected_before_launch(v7, change, reason):
    raw = spec()
    raw.update(change)
    with pytest.raises(ValueError, match=reason):
        compile_outcome_spec(raw, v7)


@pytest.mark.parametrize("patch", [
    {"unknown": True}, {"schemaVersion": True}, {"rulesProfile": "other"},
    {"outcome": {"decidingTrick": {"index": 8, "requireRoundFlip": True}}},
    {"outcome": {"swing": {"metric": "awarded_regret"}}},
    {"outcome": {"comeback": {"minimumDeficit": 0}}},
    {"outcome": {"decidingTrick": {"index": 0}}},
    {"outcome": {"decidingTrick": {"index": 8, "winningTeam": "north_south", "winningSeat": "west"}}},
    {"declarations": [{"category": "unknown"}]}, {"category": "unknown"},
])
def test_unsupported_clauses_fail_before_native_launch(v7, patch):
    raw = spec()
    raw.update(patch)
    with pytest.raises((ValueError, TypeError)):
        compile_outcome_spec(raw, v7)


def test_yaml_determinism_and_identity_binds_rules_policies_trajectory(v7, tmp_path):
    path = tmp_path / "outcome.yaml"
    path.write_text("schemaVersion: 1\nrulesProfile: saudi_federation_2023_v7\nmode: sun\ncategory: comeback\n", encoding="utf-8")
    a = load_outcome_spec(path, v7)
    b = compile_outcome_spec(spec(category="comeback"), v7)
    assert a.to_dict() == b.to_dict()
    for change in ({"outcome": {"comeback": {"minimumDeficit": 20}}},
                   {"policies": {"default": "optimal"}}):
        raw = spec(category="comeback", **change)
        assert compile_outcome_spec(raw, v7).sha256 != a.sha256
    path.write_text("mode: sun\nmode: hokum\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique strings"):
        load_outcome_spec(path, v7)
