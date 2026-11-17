from __future__ import annotations

from .enums import Contract, Seat, Team
from .models import RoundScore, RoundState
from .project_ir import ProjectDomainIRV1
from .scoring_ir import ScoringDomainIRV1


def _round_half_down_to_ten(value: int) -> int:
    return ScoringDomainIRV1.round_half_down_to_ten(value)


def _sun_game_points(raw: int) -> int:
    return ScoringDomainIRV1.sun_game_points(raw)


def _hokum_game_points(raw: int) -> int:
    return ScoringDomainIRV1.hokum_game_points(raw)


def _compiled_game_points(
    raw: int,
    contract: Contract,
    domain: ScoringDomainIRV1,
) -> int:
    return domain.convert_raw(raw, contract)


def _multiplier(level: str) -> int:
    return ScoringDomainIRV1.multiplier(level)


def resolve_counting_team(
    *,
    declarer: Seat,
    escalation: str,
    last_escalating_team: Team | str | int | None = None,
) -> Team:
    """Resolve the scoring side from one governed escalation history.

    The runtime state and Research compiler both call this function so a
    multiplied construction cannot silently score from different sides.
    """

    declarer = Seat.parse(declarer)
    return ScoringDomainIRV1.resolve_counting_team(
        declarer_team=declarer.team,
        escalation=escalation,
        last_escalating_team=last_escalating_team,
    )


def _counting_team(state: RoundState) -> Team:
    assert state.declarer is not None
    last_team = (
        state.escalation_history[-1]["team"]
        if state.escalation_history
        else None
    )
    return resolve_counting_team(
        declarer=state.declarer,
        escalation=state.escalation,
        last_escalating_team=last_team,
    )


def score_round(
    state: RoundState,
    domain: ScoringDomainIRV1,
    project_domain: ProjectDomainIRV1,
) -> RoundScore:
    if not isinstance(domain, ScoringDomainIRV1):
        raise TypeError("score_round requires a compiled ScoringDomainIRV1")
    if not isinstance(project_domain, ProjectDomainIRV1):
        raise TypeError("score_round requires a compiled ProjectDomainIRV1")
    if state.contract is None or state.declarer is None or len(state.completed_tricks) != 8:
        raise ValueError("a completed round is required")
    raw = [0, 0]
    tricks_won = [0, 0]
    for trick in state.completed_tricks:
        assert trick.winner is not None
        team = int(trick.winner.team)
        raw[team] += trick.raw_points
        tricks_won[team] += 1
    assert state.completed_tricks[-1].winner is not None
    raw[int(state.completed_tricks[-1].winner.team)] += domain.last_trick_bonus_raw
    counting = _counting_team(state)
    contract_key = state.contract.effective.value
    expected_raw = domain.raw_card_totals[contract_key]
    if sum(raw) != expected_raw:
        raise ValueError(
            f"completed trick points total {sum(raw)} does not match compiled {contract_key} total {expected_raw}"
        )
    total_cards = domain.card_game_point_totals[contract_key]
    count_gp = _compiled_game_points(
        raw[int(counting)], state.contract, domain
    )
    card_gp = [0, 0]
    card_gp[int(counting)] = count_gp
    card_gp[int(counting.other)] = total_cards - count_gp
    project_points = [0, 0]
    for project in state.projects:
        if project.scoring:
            project_points[int(project.team)] += project.value
    baloot_points = [0, 0]
    for team in state.baloot_teams:
        baloot_points[int(team)] += project_domain.baloot_value
    totals = [card_gp[i] + project_points[i] + baloot_points[i] for i in range(2)]
    declarer_team = state.declarer.team
    if totals[0] > totals[1]:
        winner = Team.NORTH_SOUTH
    elif totals[1] > totals[0]:
        winner = Team.WEST_EAST
    else:
        winner = domain.tie_winner(state.contract, raw[int(counting)], counting, declarer_team)
    multiplier = _multiplier(state.escalation)
    kabout = domain.kabout_enabled and 0 in tricks_won
    if kabout and domain.kabout_winner == "trick_sweeper":
        winner = Team.NORTH_SOUTH if tricks_won[0] else Team.WEST_EAST
    gahwa = state.escalation == "gahwa"
    explanation = [
        f"counting_team={counting.name.lower()}",
        f"card_game_points={card_gp}",
        f"projects={project_points}",
        f"baloot={baloot_points}",
    ]
    awarded = list(
        domain.round_award(
            contract=state.contract,
            winner=winner,
            declarer_team=declarer_team,
            escalation=state.escalation,
            total_before_award=totals,
            project_points=project_points,
            baloot_points=baloot_points,
            kabout=kabout,
        )
    )
    match_winner = winner if gahwa else None
    return RoundScore(raw, card_gp, project_points, baloot_points, totals, winner, awarded, counting, multiplier, kabout, gahwa, match_winner, explanation)
