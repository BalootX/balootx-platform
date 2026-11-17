"""Native CP-SAT round and match scoring for Research V4 V3."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

from ortools.sat.python import cp_model

from baloot_platform.core.enums import Contract, Seat, Team
from baloot_platform.core.scoring_ir import ScoringDomainIRV1

from .scenario_contracts import canonical_sha256
from .scenario_project_outcome_lowering_v2 import CpSatOutcomePredicateContextV1


_LEVELS = ("normal", "double", "triple", "four", "gahwa")
_EFFECTIVE_CONTRACTS = (Contract.SUN, Contract.HOKUM)


def scoring_level_id_v1(value: str) -> int:
    """Return the governed CP-SAT ordinal for one escalation level."""

    if type(value) is not str or value not in _LEVELS:
        raise ValueError("unsupported escalation level")
    return _LEVELS.index(value)


def scoring_contract_id_v1(value: Contract | str) -> int:
    """Return the governed CP-SAT ordinal for one effective contract."""

    effective = Contract.parse(value).effective
    if effective not in _EFFECTIVE_CONTRACTS:
        raise ValueError("unsupported scoring contract")
    return _EFFECTIVE_CONTRACTS.index(effective)


def attach_raw_conversion_v1(model, raw, *, domain, contract, name):
    """Compile the profile's decimal rounding primitive, without evaluated rows."""
    effective = Contract.parse(contract).effective
    mode = domain.conversion[effective.value]
    if mode not in {
        "nearest_ten_half_down_then_divide_by_ten",
        "nearest_ten_except_exact_five_then_divide_by_five",
    }:
        raise ValueError(f"unsupported scoring conversion primitive: {mode}")
    maximum = domain.raw_card_totals[effective.value]
    model.add(raw >= 0)
    model.add(raw <= maximum)
    tens = model.new_int_var(0, maximum // 10, f"{name}_tens")
    units = model.new_int_var(0, 9, f"{name}_units")
    model.add(raw == tens * 10 + units)
    up = model.new_bool_var(f"{name}_round_up")
    model.add(units >= 6).only_enforce_if(up)
    model.add(units <= 5).only_enforce_if(up.Not())
    result = model.new_int_var(0, maximum, name)
    if mode == "nearest_ten_half_down_then_divide_by_ten":
        model.add(result == tens + up)
    else:
        half = model.new_bool_var(f"{name}_exact_five")
        model.add(units == 5).only_enforce_if(half)
        model.add(units != 5).only_enforce_if(half.Not())
        model.add(result == 2 * (tens + up) + half)
    return result


@dataclass(frozen=True, slots=True)
class CpSatScoringInputsV1:
    """Live variables consumed by the native scoring fragment."""

    contract: cp_model.IntVar
    buyer_seat: cp_model.IntVar
    escalation: cp_model.IntVar
    last_escalating_team: cp_model.IntVar
    raw_points: tuple[cp_model.IntVar, cp_model.IntVar]
    project_points: tuple[cp_model.IntVar, cp_model.IntVar]
    baloot_points: tuple[cp_model.IntVar, cp_model.IntVar]
    trick_winner_teams: tuple[cp_model.IntVar, ...]
    starting_match_score: tuple[cp_model.IntVar, cp_model.IntVar]

    def __post_init__(self) -> None:
        for name in (
            "raw_points",
            "project_points",
            "baloot_points",
            "starting_match_score",
        ):
            if len(getattr(self, name)) != 2:
                raise ValueError(f"{name} requires two team variables")
        if len(self.trick_winner_teams) != 8:
            raise ValueError("complete scoring requires eight trick-winner variables")


@dataclass(frozen=True, slots=True)
class OrToolsScoringFragmentV1:
    """Derived scoring variables attached to an owning CP-SAT model."""

    model: cp_model.CpModel
    inputs: CpSatScoringInputsV1
    declarer_team: cp_model.IntVar
    counting_team: cp_model.IntVar
    card_game_points: tuple[cp_model.IntVar, cp_model.IntVar]
    total_before_award: tuple[cp_model.IntVar, cp_model.IntVar]
    round_winner: cp_model.IntVar
    multiplier: cp_model.IntVar
    kabout: cp_model.IntVar
    gahwa: cp_model.IntVar
    awarded_points: tuple[cp_model.IntVar, cp_model.IntVar]
    final_match_score: tuple[cp_model.IntVar, cp_model.IntVar]
    match_winner: cp_model.IntVar
    outcome_context: CpSatOutcomePredicateContextV1
    domain_ir_hash: str


def extract_scoring_evidence_v1(
    solution: Any,
    fragment: OrToolsScoringFragmentV1,
    domain: ScoringDomainIRV1,
    *,
    contract: Contract | str | None = None,
) -> dict[str, Any]:
    """Read immutable score evidence from one solved native fragment."""

    if not isinstance(fragment, OrToolsScoringFragmentV1):
        raise TypeError("scoring evidence requires OrToolsScoringFragmentV1")
    if not isinstance(domain, ScoringDomainIRV1):
        raise TypeError("scoring evidence requires ScoringDomainIRV1")
    if fragment.domain_ir_hash != domain.sha256:
        raise ValueError("scoring evidence domain differs from its native fragment")
    value = solution.value
    effective_contract = _EFFECTIVE_CONTRACTS[
        int(value(fragment.inputs.contract))
    ]
    declared_contract = (
        effective_contract if contract is None else Contract.parse(contract)
    )
    if declared_contract.effective != effective_contract:
        raise ValueError("declared contract disagrees with solved scoring contract")
    buyer = Seat(int(value(fragment.inputs.buyer_seat)))
    match_value = int(value(fragment.match_winner))
    evidence = {
        "schemaVersion": 1,
        "scoringDomainHash": domain.sha256,
        "buyer": buyer.name.lower(),
        "contract": declared_contract.value,
        "rawPoints": [int(value(item)) for item in fragment.inputs.raw_points],
        "cardGamePoints": [
            int(value(item)) for item in fragment.card_game_points
        ],
        "projectPoints": [
            int(value(item)) for item in fragment.inputs.project_points
        ],
        "balootPoints": [
            int(value(item)) for item in fragment.inputs.baloot_points
        ],
        "totalBeforeAward": [
            int(value(item)) for item in fragment.total_before_award
        ],
        "gamePoints": [int(value(item)) for item in fragment.awarded_points],
        "roundWinner": Team(int(value(fragment.round_winner))).name.lower(),
        "countingTeam": Team(int(value(fragment.counting_team))).name.lower(),
        "multiplier": int(value(fragment.multiplier)),
        "kabout": bool(value(fragment.kabout)),
        "gahwa": bool(value(fragment.gahwa)),
        "finalMatchScore": [
            int(value(item)) for item in fragment.final_match_score
        ],
        "matchWinner": (
            Team(match_value).name.lower() if match_value in {0, 1} else None
        ),
        "matchTarget": domain.match_target,
    }
    return {**evidence, "sha256": canonical_sha256(evidence)}


@dataclass(frozen=True, slots=True)
class OrToolsScoringResultV1:
    status: str
    solver_status: str
    score: Mapping[str, Any] | None
    domain_ir_hash: str
    model_hash: str
    evidence_hash: str | None
    variable_count: int
    constraint_count: int
    branches: int
    conflicts: int
    context: Any = field(default=None, compare=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "status": self.status,
            "solverStatus": self.solver_status,
            "score": dict(self.score) if self.score is not None else None,
            "domainIrHash": self.domain_ir_hash,
            "modelHash": self.model_hash,
            "evidenceHash": self.evidence_hash,
            "variableCount": self.variable_count,
            "constraintCount": self.constraint_count,
            "branches": self.branches,
            "conflicts": self.conflicts,
        }


def _bounds(variable: cp_model.IntVar, label: str) -> tuple[int, int]:
    domain = tuple(int(value) for value in variable.proto.domain)
    if not domain or len(domain) % 2:
        raise ValueError(f"{label} has an invalid CP-SAT domain")
    lower, upper = min(domain[::2]), max(domain[1::2])
    if lower < 0:
        raise ValueError(f"{label} must be non-negative")
    return lower, upper


def _equals(
    model: cp_model.CpModel,
    variable: cp_model.IntVar,
    value: int,
    name: str,
) -> cp_model.IntVar:
    literal = model.new_bool_var(name)
    model.add(variable == value).only_enforce_if(literal)
    model.add(variable != value).only_enforce_if(literal.Not())
    return literal


def attach_scoring_fragment_v1(
    model: cp_model.CpModel,
    domain: ScoringDomainIRV1,
    inputs: CpSatScoringInputsV1,
    *,
    source_hash: str,
    native_conversion: bool = False,
) -> OrToolsScoringFragmentV1:
    """Attach scoring directly to construction/proof variables without replay."""

    if not isinstance(model, cp_model.CpModel):
        raise TypeError("scoring fragment requires a CpModel")
    if type(native_conversion) is not bool:
        raise TypeError("native_conversion must be bool")
    if not isinstance(domain, ScoringDomainIRV1):
        raise TypeError("scoring fragment requires ScoringDomainIRV1")
    if not isinstance(inputs, CpSatScoringInputsV1):
        raise TypeError("scoring fragment requires CpSatScoringInputsV1")
    if type(source_hash) is not str or len(source_hash) != 64:
        raise ValueError("scoring fragment source_hash must be a SHA-256 digest")

    project_bounds = tuple(
        _bounds(value, f"project_points[{index}]")
        for index, value in enumerate(inputs.project_points)
    )
    baloot_bounds = tuple(
        _bounds(value, f"baloot_points[{index}]")
        for index, value in enumerate(inputs.baloot_points)
    )
    starting_bounds = tuple(
        _bounds(value, f"starting_match_score[{index}]")
        for index, value in enumerate(inputs.starting_match_score)
    )
    for index, value in enumerate(inputs.raw_points):
        _bounds(value, f"raw_points[{index}]")
    for index, value in enumerate(inputs.trick_winner_teams):
        _bounds(value, f"trick_winner_teams[{index}]")

    contract_rows = tuple(
        (
            index,
            domain.raw_card_totals[contract.value],
            domain.card_game_point_totals[contract.value],
        )
        for index, contract in enumerate(_EFFECTIVE_CONTRACTS)
    )
    raw_maximum = max(row[1] for row in contract_rows)
    game_maximum = max(row[2] for row in contract_rows)
    model.add_allowed_assignments([inputs.contract], [(0,), (1,)])
    model.add_allowed_assignments(
        [inputs.buyer_seat], [(int(seat),) for seat in Seat]
    )
    model.add_allowed_assignments(
        [inputs.escalation], [(index,) for index in range(len(_LEVELS))]
    )
    model.add_allowed_assignments(
        [inputs.last_escalating_team], [(0,), (1,), (2,)]
    )
    for variable in inputs.trick_winner_teams:
        model.add_allowed_assignments([variable], [(0,), (1,)])

    declarer_team = model.new_int_var(0, 1, "score_declarer_team")
    model.add_allowed_assignments(
        [inputs.buyer_seat, declarer_team],
        [(int(seat), int(seat.team)) for seat in Seat],
    )
    counting_team = model.new_int_var(0, 1, "score_counting_team")
    if native_conversion:
        normal_counting = _equals(model, inputs.escalation, _LEVELS.index("normal"), "score_normal_counting")
        no_prior = _equals(model, inputs.last_escalating_team, 2, "score_no_prior_escalation")
        use_buyer = model.new_bool_var("score_count_against_buyer")
        model.add_max_equality(use_buyer, [normal_counting, no_prior])
        model.add(counting_team + declarer_team == 1).only_enforce_if(use_buyer)
        model.add(counting_team + inputs.last_escalating_team == 1).only_enforce_if(use_buyer.Not())
    else:
        counting_rows = tuple(
        (
            int(declarer),
            level_index,
            2 if prior is None else int(prior),
            int(
                domain.resolve_counting_team(
                    declarer_team=declarer,
                    escalation=level,
                    last_escalating_team=prior,
                )
            ),
        )
        for declarer in Team
        for level_index, level in enumerate(_LEVELS)
        for prior in (None, *tuple(Team))
    )
        model.add_allowed_assignments(
            [declarer_team, inputs.escalation, inputs.last_escalating_team, counting_team],
            counting_rows,
        )

    raw_total = model.new_int_var(0, raw_maximum, "score_raw_total")
    game_total = model.new_int_var(0, game_maximum, "score_game_total")
    model.add_allowed_assignments(
        [inputs.contract, raw_total, game_total], contract_rows
    )
    model.add(sum(inputs.raw_points) == raw_total)
    counting_raw = model.new_int_var(0, raw_maximum, "score_counting_raw")
    model.add_element(counting_team, inputs.raw_points, counting_raw)
    converted = model.new_int_var(0, game_maximum, "score_converted_raw")
    if native_conversion:
        conversions = []
        for index, contract in enumerate(_EFFECTIVE_CONTRACTS):
            selected = _equals(model, inputs.contract, index, f"score_conversion_contract_{index}")
            branch_raw = model.new_int_var(0, domain.raw_card_totals[contract.value],
                                           f"score_conversion_raw_{index}")
            model.add(branch_raw == counting_raw).only_enforce_if(selected)
            model.add(branch_raw == 0).only_enforce_if(selected.Not())
            conversions.append(attach_raw_conversion_v1(
                model, branch_raw, domain=domain, contract=contract,
                name=f"score_conversion_value_{index}",
            ))
        model.add_element(inputs.contract, conversions, converted)
    else:
        conversion_rows = tuple(
            (contract_index, raw, domain.convert_raw(raw, contract))
            for contract_index, contract in enumerate(_EFFECTIVE_CONTRACTS)
            for raw in range(domain.raw_card_totals[contract.value] + 1)
        )
        model.add_allowed_assignments(
            [inputs.contract, counting_raw, converted], conversion_rows
        )
    game = tuple(
        model.new_int_var(0, game_maximum, f"score_card_game_{team}")
        for team in range(2)
    )
    counting_ns = _equals(model, counting_team, 0, "score_counting_ns")
    model.add(game[0] == converted).only_enforce_if(counting_ns)
    model.add(game[1] == game_total - converted).only_enforce_if(counting_ns)
    model.add(game[1] == converted).only_enforce_if(counting_ns.Not())
    model.add(game[0] == game_total - converted).only_enforce_if(counting_ns.Not())

    total_bounds = tuple(
        game_maximum + project_bounds[team][1] + baloot_bounds[team][1]
        for team in range(2)
    )
    totals = tuple(
        model.new_int_var(0, total_bounds[team], f"score_total_{team}")
        for team in range(2)
    )
    for team in range(2):
        model.add(
            totals[team]
            == game[team] + inputs.project_points[team] + inputs.baloot_points[team]
        )

    ns_greater = model.new_bool_var("score_ns_greater")
    we_greater = model.new_bool_var("score_we_greater")
    tied = model.new_bool_var("score_tied")
    model.add(totals[0] > totals[1]).only_enforce_if(ns_greater)
    model.add(totals[0] <= totals[1]).only_enforce_if(ns_greater.Not())
    model.add(totals[1] > totals[0]).only_enforce_if(we_greater)
    model.add(totals[1] <= totals[0]).only_enforce_if(we_greater.Not())
    model.add(totals[0] == totals[1]).only_enforce_if(tied)
    model.add(totals[0] != totals[1]).only_enforce_if(tied.Not())
    model.add_exactly_one(ns_greater, we_greater, tied)
    winner = model.new_int_var(0, 1, "score_round_winner")
    model.add(winner == 0).only_enforce_if(ns_greater)
    model.add(winner == 1).only_enforce_if(we_greater)
    units = model.new_int_var(0, 9, "score_counting_raw_units")
    model.add_modulo_equality(units, counting_raw, 10)
    tie_winner = model.new_int_var(0, 1, "score_tie_winner")
    if native_conversion:
        counting_selected = model.new_bool_var("score_tie_counting_selected")
        matches = []
        for contract_index, selectors in enumerate(domain.tie_counting_units):
            if contract_index == 0 and domain.buyer_tie_outcome is not None:
                continue
            contract_matches = _equals(model, inputs.contract, contract_index,
                                       f"score_tie_contract_{contract_index}")
            for unit in selectors:
                unit_matches = _equals(model, units, unit, f"score_tie_unit_{contract_index}_{unit}")
                match = model.new_bool_var(f"score_tie_match_{contract_index}_{unit}")
                model.add_min_equality(match, [contract_matches, unit_matches])
                matches.append(match)
        model.add_max_equality(counting_selected, matches)
        if domain.buyer_tie_outcome is None:
            model.add(tie_winner == counting_team).only_enforce_if(counting_selected)
            model.add(tie_winner == declarer_team).only_enforce_if(counting_selected.Not())
        else:
            sun = _equals(model, inputs.contract, 0, "score_configured_sun_tie")
            model.add(tie_winner == (declarer_team if domain.buyer_tie_outcome == "buyer_makes"
                                    else 1 - declarer_team)).only_enforce_if(sun)
            model.add(tie_winner == counting_team).only_enforce_if([sun.Not(), counting_selected])
            model.add(tie_winner == declarer_team).only_enforce_if([sun.Not(), counting_selected.Not()])
    else:
        tie_rows = tuple(
        (
            contract_index,
            unit,
            int(counting),
            int(declarer),
            int(domain.tie_winner(contract, unit, counting, declarer)),
        )
        for contract_index, contract in enumerate(_EFFECTIVE_CONTRACTS)
        for unit in range(10)
        for counting in Team
        for declarer in Team
    )
        model.add_allowed_assignments(
            [inputs.contract, units, counting_team, declarer_team, tie_winner],
            tie_rows,
        )
    model.add(winner == tie_winner).only_enforce_if(tied)
    winner_ns = _equals(model, winner, 0, "score_winner_ns")
    declarer_wins = model.new_bool_var("score_declarer_wins")
    model.add(winner == declarer_team).only_enforce_if(declarer_wins)
    model.add(winner != declarer_team).only_enforce_if(declarer_wins.Not())

    ns_trick_count = model.new_int_var(0, 8, "score_ns_trick_count")
    ns_trick_literals = tuple(
        _equals(model, variable, 0, f"score_trick_{index}_ns")
        for index, variable in enumerate(inputs.trick_winner_teams)
    )
    model.add(ns_trick_count == sum(ns_trick_literals))
    kabout = model.new_bool_var("score_kabout")
    if domain.kabout_enabled:
        model.add_allowed_assignments(
            [ns_trick_count, kabout],
            [(count, int(count in {0, 8})) for count in range(9)],
        )
    else:
        model.add(kabout == 0)

    if domain.kabout_winner == "trick_sweeper":
        total_winner = winner
        winner = model.new_int_var(0, 1, "score_effective_round_winner")
        model.add(winner == total_winner).only_enforce_if(kabout.Not())
        model.add(winner == inputs.trick_winner_teams[0]).only_enforce_if(kabout)
        winner_ns = _equals(model, winner, 0, "score_effective_winner_ns")
        declarer_wins = model.new_bool_var("score_effective_declarer_wins")
        model.add(winner == declarer_team).only_enforce_if(declarer_wins)
        model.add(winner != declarer_team).only_enforce_if(declarer_wins.Not())

    multiplier = model.new_int_var(1, 4, "score_multiplier")
    project_multiplier = model.new_int_var(1, 4, "score_project_multiplier")
    baloot_multiplier = model.new_int_var(1, 4, "score_baloot_multiplier")
    kabout_multiplier = model.new_int_var(1, 4, "score_kabout_multiplier")
    if native_conversion:
        multiplier_values = dict(domain.escalation_multipliers)
        model.add_element(inputs.escalation, [multiplier_values[level] for level in _LEVELS], multiplier)
        model.add_min_equality(project_multiplier, [multiplier, domain.project_multiplier_cap])
        model.add_min_equality(baloot_multiplier, [multiplier, domain.baloot_multiplier_cap])
        model.add_min_equality(kabout_multiplier, [multiplier, domain.kabout_multiplier_cap])
    else:
        multiplier_rows = tuple(
        (
            index,
            domain.multiplier(level),
            min(domain.multiplier(level), domain.project_multiplier_cap),
            min(domain.multiplier(level), domain.baloot_multiplier_cap),
            min(domain.multiplier(level), domain.kabout_multiplier_cap),
        )
        for index, level in enumerate(_LEVELS)
    )
        model.add_allowed_assignments(
            [inputs.escalation, multiplier, project_multiplier,
             baloot_multiplier, kabout_multiplier], multiplier_rows,
        )
    normal = _equals(model, inputs.escalation, 0, "score_normal")
    gahwa = _equals(model, inputs.escalation, 4, "score_gahwa")
    multiplied = model.new_bool_var("score_multiplied")
    model.add_allowed_assignments(
        [inputs.escalation, multiplied],
        [(index, int(index in {1, 2, 3})) for index in range(len(_LEVELS))],
    )

    project_sum_max = sum(bound[1] for bound in project_bounds)
    baloot_sum_max = sum(bound[1] for bound in baloot_bounds)
    project_sum = model.new_int_var(0, project_sum_max, "score_project_sum")
    baloot_sum = model.new_int_var(0, baloot_sum_max, "score_baloot_sum")
    model.add(project_sum == sum(inputs.project_points))
    model.add(baloot_sum == sum(inputs.baloot_points))

    kabout_base = model.new_int_var(
        0, max(domain.kabout_base_game_points.values()), "score_kabout_base"
    )
    model.add_allowed_assignments(
        [inputs.contract, kabout_base],
        [
            (index, domain.kabout_base_game_points[contract.value])
            for index, contract in enumerate(_EFFECTIVE_CONTRACTS)
        ],
    )
    kabout_base_scaled = model.new_int_var(
        0,
        max(domain.kabout_base_game_points.values()) * domain.kabout_multiplier_cap,
        "score_kabout_base_scaled",
    )
    model.add_multiplication_equality(
        kabout_base_scaled, [kabout_base, kabout_multiplier]
    )

    def scaled(
        value: cp_model.IntVar,
        value_upper: int,
        factor: cp_model.IntVar,
        factor_upper: int,
        name: str,
    ) -> cp_model.IntVar:
        result = model.new_int_var(0, value_upper * factor_upper, name)
        model.add_multiplication_equality(result, [value, factor])
        return result

    kabout_values = []
    kabout_maxima = []
    for team in range(2):
        project_scaled = scaled(
            inputs.project_points[team],
            project_bounds[team][1],
            project_multiplier,
            domain.project_multiplier_cap,
            f"score_kabout_project_{team}",
        )
        baloot_scaled = scaled(
            inputs.baloot_points[team],
            baloot_bounds[team][1],
            baloot_multiplier,
            domain.baloot_multiplier_cap,
            f"score_kabout_baloot_{team}",
        )
        maximum = (
            max(domain.kabout_base_game_points.values())
            * domain.kabout_multiplier_cap
            + project_bounds[team][1] * domain.project_multiplier_cap
            + baloot_bounds[team][1] * domain.baloot_multiplier_cap
        )
        value = model.new_int_var(0, maximum, f"score_kabout_value_{team}")
        model.add(value == kabout_base_scaled + project_scaled + baloot_scaled)
        kabout_values.append(value)
        kabout_maxima.append(maximum)

    failure_maximum = game_maximum + project_sum_max + baloot_sum_max
    failure_value = model.new_int_var(0, failure_maximum, "score_failure_value")
    model.add(failure_value == game_total + project_sum + baloot_sum)
    game_scaled = scaled(game_total, game_maximum, multiplier, 4, "score_game_scaled")
    projects_scaled = scaled(
        project_sum,
        project_sum_max,
        project_multiplier,
        domain.project_multiplier_cap,
        "score_projects_scaled",
    )
    baloots_scaled = scaled(
        baloot_sum,
        baloot_sum_max,
        baloot_multiplier,
        domain.baloot_multiplier_cap,
        "score_baloots_scaled",
    )
    multiplied_maximum = (
        game_maximum * 4
        + project_sum_max * domain.project_multiplier_cap
        + baloot_sum_max * domain.baloot_multiplier_cap
    )
    multiplied_value = model.new_int_var(
        0, multiplied_maximum, "score_multiplied_value"
    )
    model.add(multiplied_value == game_scaled + projects_scaled + baloots_scaled)
    award_maximum = max(
        failure_maximum,
        multiplied_maximum,
        *kabout_maxima,
        *total_bounds,
    )
    awarded = tuple(
        model.new_int_var(0, award_maximum, f"score_awarded_{team}")
        for team in range(2)
    )
    winner_literals = (winner_ns, winner_ns.Not())
    for team in range(2):
        model.add(awarded[team] == 0).only_enforce_if(gahwa)
        model.add(awarded[team] == kabout_values[team]).only_enforce_if(
            [gahwa.Not(), kabout, winner_literals[team]]
        )
        model.add(awarded[team] == 0).only_enforce_if(
            [gahwa.Not(), kabout, winner_literals[1 - team]]
        )
        model.add(awarded[team] == totals[team]).only_enforce_if(
            [normal, kabout.Not(), declarer_wins]
        )
        model.add(awarded[team] == failure_value).only_enforce_if(
            [normal, kabout.Not(), declarer_wins.Not(), winner_literals[team]]
        )
        model.add(awarded[team] == 0).only_enforce_if(
            [normal, kabout.Not(), declarer_wins.Not(), winner_literals[1 - team]]
        )
        model.add(awarded[team] == multiplied_value).only_enforce_if(
            [multiplied, kabout.Not(), winner_literals[team]]
        )
        model.add(awarded[team] == 0).only_enforce_if(
            [multiplied, kabout.Not(), winner_literals[1 - team]]
        )

    final_upper = max(
        max(bound[1] for bound in starting_bounds) + award_maximum,
        domain.match_target,
        max(starting_bounds[0][1], starting_bounds[1][1]) + 1,
    )
    final = tuple(
        model.new_int_var(0, final_upper, f"score_final_match_{team}")
        for team in range(2)
    )
    for team in range(2):
        gahwa_target = model.new_int_var(
            domain.match_target,
            max(domain.match_target, starting_bounds[1 - team][1] + 1),
            f"score_gahwa_target_{team}",
        )
        model.add_max_equality(
            gahwa_target,
            [domain.match_target, inputs.starting_match_score[1 - team] + 1],
        )
        model.add(final[team] == gahwa_target).only_enforce_if(
            [gahwa, winner_literals[team]]
        )
        model.add(final[team] == inputs.starting_match_score[team]).only_enforce_if(
            [gahwa, winner_literals[1 - team]]
        )
        model.add(
            final[team] == inputs.starting_match_score[team] + awarded[team]
        ).only_enforce_if(gahwa.Not())

    ns_match = model.new_bool_var("score_ns_match_winner")
    we_match = model.new_bool_var("score_we_match_winner")
    no_match = model.new_bool_var("score_no_match_winner")
    ns_reached = model.new_bool_var("score_ns_reached_target")
    ns_ahead = model.new_bool_var("score_ns_ahead")
    we_reached = model.new_bool_var("score_we_reached_target")
    we_ahead = model.new_bool_var("score_we_ahead")
    model.add(final[0] >= domain.match_target).only_enforce_if(ns_reached)
    model.add(final[0] < domain.match_target).only_enforce_if(ns_reached.Not())
    model.add(final[0] > final[1]).only_enforce_if(ns_ahead)
    model.add(final[0] <= final[1]).only_enforce_if(ns_ahead.Not())
    model.add(final[1] >= domain.match_target).only_enforce_if(we_reached)
    model.add(final[1] < domain.match_target).only_enforce_if(we_reached.Not())
    model.add(final[1] > final[0]).only_enforce_if(we_ahead)
    model.add(final[1] <= final[0]).only_enforce_if(we_ahead.Not())
    model.add_bool_and([ns_reached, ns_ahead]).only_enforce_if(ns_match)
    model.add_bool_or([ns_reached.Not(), ns_ahead.Not()]).only_enforce_if(
        ns_match.Not()
    )
    model.add_bool_and([we_reached, we_ahead]).only_enforce_if(we_match)
    model.add_bool_or([we_reached.Not(), we_ahead.Not()]).only_enforce_if(
        we_match.Not()
    )
    model.add_exactly_one(ns_match, we_match, no_match)
    match_winner = model.new_int_var(0, 2, "score_match_winner")
    model.add(match_winner == 0).only_enforce_if(ns_match)
    model.add(match_winner == 1).only_enforce_if(we_match)
    model.add(match_winner == 2).only_enforce_if(no_match)

    outcome_context = CpSatOutcomePredicateContextV1(
        model=model,
        context_id=f"native-scoring-{source_hash[:16]}",
        source_stage="scoring",
        source_hash=source_hash,
        buyer_seat=inputs.buyer_seat,
        round_winner=winner,
        match_winner=match_winner,
        raw_points=inputs.raw_points,
        game_points=awarded,
        project_points=inputs.project_points,
        final_match_score=final,
        kabout=kabout,
        gahwa=gahwa,
        match_target=domain.match_target,
    )
    return OrToolsScoringFragmentV1(
        model,
        inputs,
        declarer_team,
        counting_team,
        game,
        totals,
        winner,
        multiplier,
        kabout,
        gahwa,
        awarded,
        final,
        match_winner,
        outcome_context,
        domain.sha256,
    )


def attach_project_counterfactual_scoring_v1(
    fragment: OrToolsScoringFragmentV1, domain: ScoringDomainIRV1,
) -> OrToolsScoringFragmentV1:
    """Suppress ordinary project points only; retain the same play and Baloot.

    This is a scoring intervention, not a second legally reachable round.
    Strategic causality additionally requires paired continuation proof.
    """
    if not isinstance(fragment, OrToolsScoringFragmentV1) or not isinstance(domain, ScoringDomainIRV1):
        raise TypeError("project counterfactual requires a native scoring fragment and domain")
    if fragment.domain_ir_hash != domain.sha256:
        raise ValueError("project counterfactual must preserve the scoring domain")
    model = fragment.model
    first_variable = len(model.proto.variables)
    source_hash = canonical_sha256({
        "kind": "ordinary_project_score_suppression_v1",
        "sourceContextHash": fragment.outcome_context.sha256,
        "scoringDomainHash": domain.sha256,
    })
    zero = model.new_constant(0)
    counterfactual = attach_scoring_fragment_v1(
        model, domain, replace(fragment.inputs, project_points=(zero, zero)),
        source_hash=source_hash, native_conversion=True,
    )
    # Keep evidence names unique without changing shared source variables.
    for index in range(first_variable, len(model.proto.variables)):
        variable = model.proto.variables[index]
        variable.name = f"project_counterfactual_{index}_{variable.name}"
    return counterfactual


class OrToolsScoringStageV1:
    """Compatibility wrapper around the attachable native scoring fragment."""

    def __init__(self, domain: ScoringDomainIRV1, *, native_conversion: bool = False) -> None:
        if not isinstance(domain, ScoringDomainIRV1):
            raise TypeError("scoring stage requires ScoringDomainIRV1")
        self.domain = domain
        if type(native_conversion) is not bool:
            raise TypeError("native_conversion must be bool")
        self.native_conversion = native_conversion

    def solve(
        self,
        *,
        contract: Contract | str,
        buyer: Seat | str | int,
        escalation: str,
        last_escalating_team: Team | str | int | None,
        raw_points: Sequence[int],
        project_points: Sequence[int],
        baloot_points: Sequence[int],
        trick_winner_teams: Sequence[Team | str | int],
        starting_match_score: Sequence[int],
        maximum_seconds: float = 2.0,
        model_configurator: Callable[[Any], None] | None = None,
    ) -> OrToolsScoringResultV1:
        contract = Contract.parse(contract)
        buyer = Seat.parse(buyer)
        if escalation not in _LEVELS:
            raise ValueError("unsupported escalation level")
        raw = self._pair(raw_points, "raw points")
        projects = self._pair(project_points, "project points")
        baloot = self._pair(baloot_points, "Baloot points")
        starting = self._pair(starting_match_score, "starting match score")
        winners = tuple(self._team(item) for item in trick_winner_teams)
        if len(winners) != 8:
            raise ValueError("complete scoring requires eight trick winners")
        last_team = (
            None if last_escalating_team is None else self._team(last_escalating_team)
        )
        if type(maximum_seconds) not in {int, float} or maximum_seconds <= 0:
            raise ValueError("maximum_seconds must be positive")
        expected_raw = self.domain.raw_card_totals[contract.effective.value]
        if sum(raw) != expected_raw or max(raw) > expected_raw:
            raise ValueError("raw points do not match the governed round total")

        request = {
            "schemaVersion": 1,
            "lowering": "native_cp_sat_scoring_v2",
            **({"rawConversionAuthority": "native_arithmetic_v1"} if self.native_conversion else {}),
            "domainIrHash": self.domain.sha256,
            "contract": contract.value,
            "buyer": buyer.name.lower(),
            "escalation": escalation,
            "lastEscalatingTeam": (
                last_team.name.lower() if last_team is not None else None
            ),
            "rawPoints": list(raw),
            "projectPoints": list(projects),
            "balootPoints": list(baloot),
            "trickWinnerTeams": [team.name.lower() for team in winners],
            "startingMatchScore": list(starting),
        }
        request_hash = canonical_sha256(request)
        model = cp_model.CpModel()

        def fixed(name: str, lower: int, upper: int, value: int) -> cp_model.IntVar:
            variable = model.new_int_var(lower, upper, name)
            model.add(variable == value)
            return variable

        raw_maximum = max(self.domain.raw_card_totals.values())
        inputs = CpSatScoringInputsV1(
            contract=fixed(
                "input_contract",
                0,
                1,
                _EFFECTIVE_CONTRACTS.index(contract.effective),
            ),
            buyer_seat=fixed("input_buyer", 0, 3, int(buyer)),
            escalation=fixed(
                "input_escalation", 0, len(_LEVELS) - 1, _LEVELS.index(escalation)
            ),
            last_escalating_team=fixed(
                "input_last_escalating_team",
                0,
                2,
                2 if last_team is None else int(last_team),
            ),
            raw_points=tuple(
                fixed(f"input_raw_{team}", 0, raw_maximum, raw[team])
                for team in range(2)
            ),
            project_points=tuple(
                fixed(f"input_projects_{team}", 0, projects[team], projects[team])
                for team in range(2)
            ),
            baloot_points=tuple(
                fixed(f"input_baloot_{team}", 0, baloot[team], baloot[team])
                for team in range(2)
            ),
            trick_winner_teams=tuple(
                fixed(f"input_trick_winner_{index}", 0, 1, int(team))
                for index, team in enumerate(winners)
            ),
            starting_match_score=tuple(
                fixed(
                    f"input_starting_match_{team}",
                    0,
                    starting[team],
                    starting[team],
                )
                for team in range(2)
            ),
        )
        fragment = attach_scoring_fragment_v1(
            model, self.domain, inputs, source_hash=request_hash,
            native_conversion=self.native_conversion,
        )
        if model_configurator is not None:
            model_configurator(fragment.outcome_context)
        model_hash = canonical_sha256(
            {"requestHash": request_hash, "cpModelProto": str(model.proto)}
        )

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_time_in_seconds = float(maximum_seconds)
        status = solver.solve(model)
        solver_status = solver.status_name(status).lower()
        proto = model.proto
        variable_count = len(proto.variables)
        constraint_count = len(proto.constraints)
        if status != cp_model.OPTIMAL:
            return OrToolsScoringResultV1(
                "incomplete" if status == cp_model.UNKNOWN else "infeasible",
                solver_status,
                None,
                self.domain.sha256,
                model_hash,
                None,
                variable_count,
                constraint_count,
                solver.num_branches,
                solver.num_conflicts,
                fragment.outcome_context,
            )
        score = extract_scoring_evidence_v1(
            solver,
            fragment,
            self.domain,
            contract=contract,
        )
        evidence_hash = str(score["sha256"])
        return OrToolsScoringResultV1(
            "satisfied",
            solver_status,
            score,
            self.domain.sha256,
            model_hash,
            evidence_hash,
            variable_count,
            constraint_count,
            solver.num_branches,
            solver.num_conflicts,
            fragment.outcome_context,
        )

    @staticmethod
    def _pair(values: Sequence[int], label: str) -> tuple[int, int]:
        result = tuple(values)
        if len(result) != 2 or any(
            type(value) is not int or value < 0 for value in result
        ):
            raise ValueError(f"{label} must contain two non-negative integers")
        return result[0], result[1]

    @staticmethod
    def _team(value: Team | str | int) -> Team:
        if isinstance(value, Team):
            return value
        if type(value) is int:
            return Team(value)
        return Team[str(value).strip().upper()]


__all__ = [
    "CpSatScoringInputsV1",
    "OrToolsScoringFragmentV1",
    "OrToolsScoringResultV1",
    "OrToolsScoringStageV1",
    "attach_scoring_fragment_v1",
    "attach_project_counterfactual_scoring_v1",
    "extract_scoring_evidence_v1",
    "scoring_contract_id_v1",
    "scoring_level_id_v1",
]
