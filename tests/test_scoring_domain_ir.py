from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from baloot_platform.core.cards import FULL_DECK
from baloot_platform.core.enums import Contract, Seat, Suit, Team
from baloot_platform.core.project_ir import ProjectDomainIRV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring import resolve_counting_team, score_round
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.native.accelerator import NativeAccelerator, NativeUnavailable
from baloot_platform.research.play_compiler import OrToolsCpSatPlayCompilerV1
from baloot_platform.research.reference_scoring import ReferenceScoringInterpreterV1
from baloot_platform.research.service import ResearchFoundryServiceV1
from test_core import _score_state


def test_scoring_domain_ir_drives_runtime_reference_and_cp_sat(rules):
    domain = ScoringDomainIRV1.from_rules(rules.data)
    project_domain = ProjectDomainIRV1.from_rules(rules.data)
    state = _score_state(
        rules,
        Contract.HOKUM,
        Seat.EAST,
        77,
        Team.NORTH_SOUTH,
    )
    runtime = score_round(state, domain, project_domain)
    raw_rows = tuple(trick.raw_points for trick in state.completed_tricks)
    winner_seats = tuple(trick.winner for trick in state.completed_tricks)
    reference = ReferenceScoringInterpreterV1(domain).score(
        contract=Contract.HOKUM,
        declarer=Seat.EAST,
        escalation="normal",
        trick_raw_points=raw_rows,
        trick_winners=winner_seats,
    )
    cp_sat = OrToolsCpSatPlayCompilerV1.scoring_probe(
        domain=domain,
        contract=Contract.HOKUM,
        declarer=Seat.EAST,
        escalation="normal",
        trick_raw_points=raw_rows,
        trick_winner_teams=tuple(seat.team for seat in winner_seats),
    )
    assert runtime.to_dict()["awardedPoints"] == list(reference.awarded_points)
    assert runtime.to_dict()["awardedPoints"] == cp_sat["awardedPoints"]
    assert runtime.to_dict()["winner"] == reference.winner.name.lower()
    assert runtime.to_dict()["winner"] == cp_sat["winner"]


def test_scoring_runtime_rejects_legacy_parallel_rulebook(rules):
    state = _score_state(
        rules,
        Contract.SUN,
        Seat.SOUTH,
        66,
        Team.WEST_EAST,
    )
    with pytest.raises(TypeError, match="compiled ScoringDomainIRV1"):
        score_round(
            state,
            rules,  # type: ignore[arg-type]
            ProjectDomainIRV1.from_rules(rules.data),
        )


def test_scoring_domain_rejects_invalid_and_influences_all_targets(rules):
    baseline = ScoringDomainIRV1.from_rules(rules.data)
    changed = copy.deepcopy(rules.data)
    changed["scoring"]["match"]["target"] = baseline.match_target + 1
    candidate = ScoringDomainIRV1.from_rules(changed)
    assert candidate.sha256 != baseline.sha256
    assert candidate.match_target == baseline.match_target + 1

    invalid = copy.deepcopy(rules.data)
    invalid["scoring"]["contract"]["buyer_team_must_win"] = False
    with pytest.raises(ValueError, match="unsupported contract"):
        ScoringDomainIRV1.from_rules(invalid)


def test_scoring_domain_contains_all_eighteen_governed_leaves(rules):
    payload = ScoringDomainIRV1.from_rules(rules.data).to_dict()
    leaves = {
        "lastTrickBonusRaw",
        "buyerTeamMustWin",
        "lossTransfersAll",
        "winnerTakesAllWhenMultiplied",
        "projectMultiplierCap",
        "balootMultiplierCap",
        "kaboutEnabled",
        "matchTarget",
        "tieAboveTargetContinues",
        "dealerRotatesRight",
    }
    leaves.update(f"rawCardTotals.{key}" for key in payload["rawCardTotals"])
    leaves.update(
        f"cardGamePointTotals.{key}" for key in payload["cardGamePointTotals"]
    )
    leaves.update(f"conversion.{key}" for key in payload["conversion"])
    leaves.update(
        f"kaboutBaseGamePoints.{key}"
        for key in payload["kaboutBaseGamePoints"]
    )
    assert len(leaves) == 18


def test_scoring_domain_pure_primitives_cover_conversion_ties_and_awards(rules):
    domain = ScoringDomainIRV1.from_rules(rules.data)

    for raw in range(domain.raw_card_totals["sun"] + 1):
        units = raw % 10
        rounded = raw if units == 5 else ((raw + 5) // 10) * 10
        assert domain.convert_raw(raw, Contract.SUN) == rounded // 5
    for raw in range(domain.raw_card_totals["hokum"] + 1):
        tens, units = divmod(raw, 10)
        assert domain.convert_raw(raw, Contract.HOKUM) == tens + int(units >= 6)

    assert [domain.multiplier(level) for level in (
        "normal", "double", "triple", "four", "gahwa"
    )] == [1, 2, 3, 4, 1]
    with pytest.raises(ValueError, match="unsupported escalation level"):
        domain.multiplier("invalid")

    assert domain.tie_winner(
        Contract.SUN, 61, Team.NORTH_SOUTH, Team.WEST_EAST
    ) == Team.NORTH_SOUTH
    assert domain.tie_winner(
        Contract.SUN, 60, Team.NORTH_SOUTH, Team.WEST_EAST
    ) == Team.WEST_EAST
    assert domain.tie_winner(
        Contract.HOKUM, 72, Team.NORTH_SOUTH, Team.WEST_EAST
    ) == Team.NORTH_SOUTH
    assert domain.tie_winner(
        Contract.HOKUM, 71, Team.NORTH_SOUTH, Team.WEST_EAST
    ) == Team.WEST_EAST

    common = {
        "contract": Contract.SUN,
        "winner": Team.NORTH_SOUTH,
        "declarer_team": Team.NORTH_SOUTH,
        "total_before_award": (35, 25),
        "project_points": (20, 10),
        "baloot_points": (0, 0),
    }
    assert domain.round_award(
        **common, escalation="normal", kabout=False
    ) == (35, 25)
    assert domain.round_award(
        **{**common, "winner": Team.WEST_EAST},
        escalation="normal",
        kabout=False,
    ) == (0, domain.card_game_point_totals["sun"] + 30)
    assert domain.round_award(
        **common, escalation="double", kabout=False
    ) == (
        domain.card_game_point_totals["sun"] * 2 + 60,
        0,
    )
    assert domain.round_award(
        **common, escalation="double", kabout=True
    ) == (
        domain.kabout_base_game_points["sun"] * 2 + 40,
        0,
    )
    assert domain.round_award(
        **common, escalation="gahwa", kabout=True
    ) == (0, 0)


def test_scoring_domain_owns_counting_side_and_match_boundary(rules):
    domain = ScoringDomainIRV1.from_rules(rules.data)

    assert domain.resolve_counting_team(
        declarer_team=Team.NORTH_SOUTH,
        escalation="normal",
    ) == Team.WEST_EAST
    assert domain.resolve_counting_team(
        declarer_team=Team.NORTH_SOUTH,
        escalation="double",
        last_escalating_team=Team.WEST_EAST,
    ) == Team.NORTH_SOUTH

    assert domain.apply_match_award(
        starting_score=(140, 100),
        awarded_points=(12, 0),
        round_winner=Team.NORTH_SOUTH,
    ) == ((152, 100), Team.NORTH_SOUTH)
    assert domain.apply_match_award(
        starting_score=(152, 152),
        awarded_points=(0, 0),
        round_winner=Team.NORTH_SOUTH,
    ) == ((152, 152), None)
    assert domain.apply_match_award(
        starting_score=(80, 200),
        awarded_points=(0, 0),
        round_winner=Team.NORTH_SOUTH,
        gahwa=True,
    ) == ((201, 200), Team.NORTH_SOUTH)

    reference = ReferenceScoringInterpreterV1(domain).apply_match_award(
        starting_score=(80, 200),
        awarded_points=(0, 0),
        round_winner=Team.NORTH_SOUTH,
        gahwa=True,
    )
    assert reference.final_score == (201, 200)
    assert reference.winner == Team.NORTH_SOUTH


@pytest.mark.parametrize(
    ("path", "counterexample"),
    [
        (("raw_card_totals", "sun"), 129),
        (("raw_card_totals", "hokum"), 161),
        (("conversion", "sun"), "truncate"),
        (("conversion", "hokum"), "truncate"),
        (("contract", "buyer_team_must_win"), False),
        (("contract", "loss_transfers_all"), False),
        (("escalation", "winner_takes_all_when_multiplied"), False),
        (("match", "tie_above_target_continues"), False),
        (("match", "dealer_rotates_right"), False),
    ],
)
def test_scoring_fixed_variants_fail_closed_at_rulebook_and_ir(
    rules,
    path,
    counterexample,
):
    changed = copy.deepcopy(rules.data)
    cursor = changed["scoring"]
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = counterexample

    with pytest.raises(ValueError):
        RuleBook(changed)
    with pytest.raises(ValueError):
        ScoringDomainIRV1.from_rules(changed)


@pytest.mark.parametrize("escalation", ["double", "four"])
def test_multiplied_scoring_uses_the_last_escalating_team_in_all_targets(
    rules,
    escalation,
):
    domain = ScoringDomainIRV1.from_rules(rules.data)
    project_domain = ProjectDomainIRV1.from_rules(rules.data)
    state = _score_state(
        rules,
        Contract.HOKUM,
        Seat.NORTH,
        77,
        Team.NORTH_SOUTH,
    )
    state.escalation = escalation
    state.escalation_history = [{"team": "west_east"}]
    counting = resolve_counting_team(
        declarer=state.declarer,
        escalation=state.escalation,
        last_escalating_team=Team.WEST_EAST,
    )
    assert counting == Team.NORTH_SOUTH

    runtime = score_round(state, domain, project_domain).to_dict()
    raw_rows = tuple(trick.raw_points for trick in state.completed_tricks)
    winner_seats = tuple(trick.winner for trick in state.completed_tricks)
    reference = ReferenceScoringInterpreterV1(domain).score(
        contract=state.contract,
        declarer=state.declarer,
        escalation=state.escalation,
        trick_raw_points=raw_rows,
        trick_winners=winner_seats,
        counting_team=counting,
    ).to_dict()
    cp_sat = OrToolsCpSatPlayCompilerV1.scoring_probe(
        domain=domain,
        contract=state.contract,
        declarer=state.declarer,
        escalation=state.escalation,
        trick_raw_points=raw_rows,
        trick_winner_teams=tuple(seat.team for seat in winner_seats),
        counting_team=counting,
    )
    fields = (
        "rawCardPoints",
        "cardGamePoints",
        "winner",
        "awardedPoints",
        "countingTeam",
        "multiplier",
        "kabout",
        "gahwa",
    )
    assert {key: runtime[key] for key in fields} == {
        key: reference[key] for key in fields
    }
    assert {key: runtime[key] for key in fields} == {
        key: cp_sat[key] for key in fields
    }

    defaulted = OrToolsCpSatPlayCompilerV1.scoring_probe(
        domain=domain,
        contract=state.contract,
        declarer=state.declarer,
        escalation=state.escalation,
        trick_raw_points=raw_rows,
        trick_winner_teams=tuple(seat.team for seat in winner_seats),
    )
    assert defaulted["countingTeam"] != runtime["countingTeam"]


@pytest.mark.parametrize(
    ("escalation", "multiplier"),
    [("double", 2), ("four", 4)],
)
def test_complete_play_compiler_preserves_north_south_counting_team(
    rules,
    escalation,
    multiplier,
):
    hands = {
        seat: tuple(
            FULL_DECK[int(seat) * 8 : (int(seat) + 1) * 8]
        )
        for seat in Seat
    }
    plan = OrToolsCpSatPlayCompilerV1().compile(
        hands=hands,
        contract=Contract.HOKUM,
        trump=Suit.SPADES,
        first_leader=Seat.NORTH,
        declarer=Seat.NORTH,
        rules=rules,
        escalation=escalation,
        counting_team=Team.NORTH_SOUTH,
        timeout_seconds=10,
        maximum_conflicts=500_000,
    )
    assert plan.solver_status == "optimal"
    assert plan.counting_team == Team.NORTH_SOUTH
    assert plan.multiplier == multiplier
    assert plan.gahwa is False
    assert plan.to_dict()["score"]["countingTeam"] == "north_south"

    compiled = plan.to_dict()
    domain = ScoringDomainIRV1.from_rules(rules.data)
    match_probe = OrToolsCpSatPlayCompilerV1.scoring_probe(
        domain=domain,
        contract=Contract.HOKUM,
        declarer=Seat.NORTH,
        escalation=escalation,
        trick_raw_points=tuple(trick.raw_points for trick in plan.tricks),
        trick_winner_teams=tuple(trick.winner.team for trick in plan.tricks),
        project_points=plan.project_points,
        baloot_points=plan.baloot_points,
        counting_team=plan.counting_team,
        starting_match_score=(0, 0),
    )
    state = {
        "dealer": "east",
        "phase": "scored",
        "currentPlayer": None,
        "currentTrick": None,
        "contract": "hokum",
        "declarer": "north",
        "escalation": escalation,
        "completedTricks": [
            {
                "index": trick.index,
                "leader": trick.leader.name.lower(),
                "plays": [{"card": card.to_dict()} for card in trick.cards],
                "winner": trick.winner.name.lower(),
                "rawPoints": trick.raw_points,
            }
            for trick in plan.tricks
        ],
        "roundScore": {
            **copy.deepcopy(compiled["score"]),
            "matchWinner": match_probe["matchWinner"],
        },
        "matchScore": match_probe["finalMatchScore"],
        "matchWinner": match_probe["matchWinner"],
        "generation": {"dealer": "east", "matchScore": [0, 0]},
    }
    simulation = SimpleNamespace(
        final_trace=SimpleNamespace(trusted_final_state=state)
    )
    assert (
        ResearchFoundryServiceV1._compiled_play_mismatches(
            plan,
            simulation,
            scoring_domain=domain,
            expected_dealer=Seat.EAST,
            starting_match_score=(0, 0),
        )
        == ()
    )
    state["roundScore"]["countingTeam"] = "west_east"
    assert ResearchFoundryServiceV1._compiled_play_mismatches(
        plan,
        simulation,
        scoring_domain=domain,
        expected_dealer=Seat.EAST,
        starting_match_score=(0, 0),
    ) == ("compiler_parity_defect",)


@pytest.mark.parametrize("section,field", [
    ("escalation", "project_multiplier_cap"),
    ("escalation", "baloot_multiplier_cap"),
    ("kabout", "multiplier_cap"),
])
@pytest.mark.parametrize("value", [True, False, "2", 2.5, None, 0, -1])
def test_scoring_multiplier_caps_reject_coercion_and_nonpositive_values(rules, section, field, value):
    data = copy.deepcopy(rules.data)
    data["scoring"][section][field] = value
    with pytest.raises(ValueError):
        ScoringDomainIRV1.from_rules(data)


@pytest.mark.parametrize("contract,units", [
    (Contract.SUN, (1, 2, 3, 4)),
    (Contract.HOKUM, (2, 3, 4, 5)),
    (Contract.ASHKAL, (1, 2, 3, 4)),
])
def test_v1_tie_selectors_preserve_all_team_and_remainder_cases(rules, contract, units):
    domain = ScoringDomainIRV1.from_rules(rules.data)
    for counting in Team:
        for declarer in Team:
            for raw in range(163):
                assert domain.tie_winner(contract, raw, counting, declarer) == (
                    counting if raw % 10 in units else declarer
                )


def test_native_scoring_table_is_lowered_from_scoring_domain(rules):
    domain = ScoringDomainIRV1.from_rules(rules.data)
    project_domain = ProjectDomainIRV1.from_rules(rules.data)
    native = NativeAccelerator._build_rules_v3(rules, project_domain, domain)
    assert native.base.last_trick_bonus_raw == domain.last_trick_bonus_raw
    assert native.base.kabout_sun == domain.kabout_base_game_points["sun"]
    assert native.base.kabout_hokum == domain.kabout_base_game_points["hokum"]
    assert native.base.match_target == domain.match_target
    assert list(native.card_game_point_totals) == [
        domain.card_game_point_totals["sun"],
        domain.card_game_point_totals["hokum"],
    ]
    assert native.project_multiplier_cap == domain.project_multiplier_cap
    assert native.baloot_multiplier_cap == domain.baloot_multiplier_cap
    assert native.kabout_enabled == int(domain.kabout_enabled)


@pytest.mark.parametrize("outcome", ["buyer_makes", "buyer_loses"])
@pytest.mark.parametrize("counting_raw,projects", [(65, (5, 5)), (61, (4, 6))])
def test_configured_sun_tie_runtime_reference_and_both_cp_sat_paths(
    rules, outcome, counting_raw, projects,
):
    from baloot_platform.research.ortools_scoring_stage import OrToolsScoringStageV1

    data = copy.deepcopy(rules.data)
    data["scoring"]["contract"]["buyer_tie_outcome"] = outcome
    domain = ScoringDomainIRV1.from_rules(data)
    state = _score_state(rules, Contract.SUN, Seat.NORTH, counting_raw,
                         Team.WEST_EAST, project_points=projects)
    runtime = score_round(state, domain, ProjectDomainIRV1.from_rules(data))
    expected = Team.NORTH_SOUTH if outcome == "buyer_makes" else Team.WEST_EAST
    assert runtime.total_before_award == [18, 18]
    assert runtime.winner == expected
    raw_rows = tuple(trick.raw_points for trick in state.completed_tricks)
    winners = tuple(trick.winner for trick in state.completed_tricks)
    inputs = dict(contract=Contract.SUN, declarer=Seat.NORTH, escalation="normal",
                  trick_raw_points=raw_rows, project_points=projects)
    reference = ReferenceScoringInterpreterV1(domain).score(**inputs, trick_winners=winners)
    legacy_cp = OrToolsCpSatPlayCompilerV1.scoring_probe(
        domain=domain, **inputs, trick_winner_teams=tuple(seat.team for seat in winners))
    assert reference.winner == runtime.winner
    assert list(reference.awarded_points) == runtime.awarded_points == legacy_cp["awardedPoints"]
    for native_conversion in (False, True):
        result = OrToolsScoringStageV1(domain, native_conversion=native_conversion).solve(
            contract=Contract.SUN, buyer=Seat.NORTH, escalation="normal",
            last_escalating_team=None, raw_points=runtime.raw_card_points,
            project_points=projects, baloot_points=(0, 0),
            trick_winner_teams=tuple(seat.team for seat in winners), starting_match_score=(0, 0))
        assert result.status == "satisfied"
        assert result.score["gamePoints"] == runtime.awarded_points
        assert result.score["roundWinner"] == expected.name.lower()


@pytest.mark.parametrize("outcome", [None, True, False, 1, "unknown", [], {}])
def test_buyer_tie_outcome_rejects_invalid_values(rules, outcome):
    data = copy.deepcopy(rules.data)
    data["scoring"]["contract"]["buyer_tie_outcome"] = outcome
    with pytest.raises(ValueError):
        ScoringDomainIRV1.from_rules(data)


def test_optional_buyer_tie_outcome_preserves_historical_identity_and_hokum(rules):
    baseline = ScoringDomainIRV1.from_rules(rules.data)
    data = copy.deepcopy(rules.data)
    data["scoring"]["contract"]["buyer_tie_outcome"] = "buyer_makes"
    makes = ScoringDomainIRV1.from_rules(data)
    data["scoring"]["contract"]["buyer_tie_outcome"] = "buyer_loses"
    loses = ScoringDomainIRV1.from_rules(data)
    assert len({baseline.sha256, makes.sha256, loses.sha256}) == 3
    assert "buyerTieOutcome" not in baseline.to_dict()
    assert makes.to_dict()["buyerTieOutcome"] == "buyer_makes"
    assert loses.to_dict()["buyerTieOutcome"] == "buyer_loses"
    del data["scoring"]["contract"]["buyer_tie_outcome"]
    assert ScoringDomainIRV1.from_rules(data).to_dict() == baseline.to_dict()
    for raw in range(163):
        expected = baseline.tie_winner(Contract.HOKUM, raw, Team.WEST_EAST, Team.NORTH_SOUTH)
        assert makes.tie_winner(Contract.HOKUM, raw, Team.WEST_EAST, Team.NORTH_SOUTH) == expected
        assert loses.tie_winner(Contract.HOKUM, raw, Team.WEST_EAST, Team.NORTH_SOUTH) == expected
    for domain in (makes, loses):
        with pytest.raises(NativeUnavailable, match="Sun buyer tie outcomes"):
            NativeAccelerator._build_rules_v3(rules, scoring_domain=domain)


@pytest.mark.parametrize("case,counting_raw,projects,historical_winner", [
    ("542C-Spade-Jack", 36, (0, 10), Team.NORTH_SOUTH),
    ("7189-Spade-Nine-consistent-world", 28, (0, 14), Team.NORTH_SOUTH),
    ("F44B-Spade-Queen", 41, (0, 10), Team.WEST_EAST),
    ("F44B-Spade-Queen-alternative-world", 43, (0, 10), Team.WEST_EAST),
])
@pytest.mark.parametrize("outcome", [None, "buyer_makes", "buyer_loses"])
def test_audited_actor_tie_scores_bind_configured_outcome(
    rules, case, counting_raw, projects, historical_winner, outcome,
):
    from baloot_platform.research.ortools_scoring_stage import OrToolsScoringStageV1

    data = copy.deepcopy(rules.data)
    if outcome is not None:
        data["scoring"]["contract"]["buyer_tie_outcome"] = outcome
    domain = ScoringDomainIRV1.from_rules(data)
    state = _score_state(rules, Contract.SUN, Seat.NORTH, counting_raw,
                         Team.WEST_EAST, project_points=projects)
    runtime = score_round(state, domain, ProjectDomainIRV1.from_rules(data))
    expected = historical_winner if outcome is None else (
        Team.NORTH_SOUTH if outcome == "buyer_makes" else Team.WEST_EAST)
    assert runtime.total_before_award[0] == runtime.total_before_award[1], case
    assert runtime.winner == expected, case
    result = OrToolsScoringStageV1(domain, native_conversion=True).solve(
        contract=Contract.SUN, buyer=Seat.NORTH, escalation="normal",
        last_escalating_team=None, raw_points=runtime.raw_card_points,
        project_points=projects, baloot_points=(0, 0),
        trick_winner_teams=tuple(trick.winner.team for trick in state.completed_tricks),
        starting_match_score=(0, 0))
    assert result.status == "satisfied", case
    assert result.score["roundWinner"] == expected.name.lower(), case
    assert result.score["gamePoints"] == runtime.awarded_points, case
