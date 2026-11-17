from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, GamePhase, Seat, Suit, Team
from baloot_platform.core.play_ir import PlayDomainIRV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.core.structural_ir import StructuralDomainIRV1
from baloot_platform.plugins.base import PluginDescriptor

from .contracts import canonical_sha256
from .reference_scoring import ReferenceScoringInterpreterV1


@dataclass(frozen=True, slots=True)
class CompiledTrickV1:
    index: int
    leader: Seat
    cards: tuple[Card, Card, Card, Card]
    winner: Seat
    raw_points: int
    ekka_lead: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "leader": self.leader.name.lower(),
            "cards": [card.to_dict() for card in self.cards],
            "winner": self.winner.name.lower(),
            "rawPoints": self.raw_points,
            "ekkaLead": self.ekka_lead,
        }


@dataclass(frozen=True, slots=True)
class PlayPlanV1:
    tricks: tuple[CompiledTrickV1, ...]
    raw_card_points: tuple[int, int]
    card_game_points: tuple[int, int]
    project_points: tuple[int, int]
    baloot_points: tuple[int, int]
    total_before_award: tuple[int, int]
    winner: Team
    awarded_points: tuple[int, int]
    counting_team: Team
    multiplier: int
    kabout: bool
    gahwa: bool
    model_hash: str
    solver_status: str
    sha256: str
    tactical_precursor: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "tricks": [trick.to_dict() for trick in self.tricks],
            "score": {
                "rawCardPoints": list(self.raw_card_points),
                "cardGamePoints": list(self.card_game_points),
                "projectPoints": list(self.project_points),
                "balootPoints": list(self.baloot_points),
                "totalBeforeAward": list(self.total_before_award),
                "winner": self.winner.name.lower(),
                "awardedPoints": list(self.awarded_points),
                "countingTeam": self.counting_team.name.lower(),
                "multiplier": self.multiplier,
                "kabout": self.kabout,
                "gahwa": self.gahwa,
            },
            "modelHash": self.model_hash,
            "solverStatus": self.solver_status,
            "tacticalPrecursor": (
                copy.deepcopy(dict(self.tactical_precursor))
                if self.tactical_precursor is not None
                else None
            ),
            "sha256": self.sha256,
        }


def _conversion_table(
    *, contract: Contract, domain: ScoringDomainIRV1
) -> tuple[tuple[int, int], ...]:
    effective = contract.effective
    mode = domain.conversion[effective.value]
    maximum = domain.raw_card_totals[effective.value]
    rows: list[tuple[int, int]] = []
    for raw in range(maximum + 1):
        if mode == "nearest_ten_half_down_then_divide_by_ten":
            tens, units = divmod(raw, 10)
            converted = tens + (1 if units >= 6 else 0)
        elif mode == "nearest_ten_except_exact_five_then_divide_by_five":
            units = raw % 10
            rounded = raw if units == 5 else ((raw + 5) // 10) * 10
            converted = rounded // 5
        else:
            raise ValueError(f"unsupported scoring conversion primitive: {mode}")
        rows.append((raw, int(converted)))
    return tuple(rows)


class OrToolsCpSatPlayCompilerV1:
    """Construct or verify a complete legal play line for one frozen deal.

    Every seat owns one card variable per trick. Leader transfer, turn order,
    remaining hands, follow/trump/overtrump obligations, trick winners, card
    points, conversion, contract loss, escalation, and Kabout are expressed in
    CP-SAT. Projects are accepted as separately compiled fixed point totals;
    project detection/reveal remains an explicit later compiler slice.
    """

    descriptor = PluginDescriptor(
        id="ortools_cp_sat_play_compiler_v1",
        kind="domain_lowering",
        version="1.0.0",
        description="Complete fixed-deal card-play, trick, and round-card scoring compiler",
        capabilities=(
            "complete_play",
            "leader_transfer",
            "follow_suit",
            "must_trump",
            "must_overtrump",
            "partner_winning_exception",
            "trick_winner",
            "card_points",
            "contract_loss",
            "escalation",
            "kabout",
            "tactical_precursor",
        ),
        deterministic=True,
        resource_class="cpu",
        timeout_seconds=60,
        maximum_memory_mb=512,
        contract_id="cp_sat_lowering",
        contract_version="1",
        safety_guards=(
            "compiled_domain_authority",
            "authoritative_legality",
            "hard_gate_preservation",
        ),
    )

    @staticmethod
    def _normalize_hands(
        hands: Mapping[Seat | str | int, Iterable[Card | int | str | dict]],
    ) -> dict[Seat, tuple[Card, ...]]:
        values: dict[Seat, tuple[Card, ...]] = {}
        for raw_seat, raw_cards in hands.items():
            seat = Seat.parse(raw_seat)
            values[seat] = tuple(Card.parse(card) for card in raw_cards)
        if set(values) != set(Seat):
            raise ValueError("complete play compilation requires all four hands")
        if any(len(cards) != 8 for cards in values.values()):
            raise ValueError("every compiled hand must contain exactly eight cards")
        all_cards = [card.id for cards in values.values() for card in cards]
        if sorted(all_cards) != list(range(32)):
            raise ValueError("compiled hands must contain every card exactly once")
        return values

    @staticmethod
    def configuration_probe(rules: RuleBook) -> dict[str, Any]:
        """Execute a bounded CP-SAT lowering probe for governed play tables.

        This is intentionally smaller than a complete round. It proves that
        the CP-SAT target consumes the configured strength/point tables and the
        discard-when-unable-to-overtrump switch as constraints.
        """
        from ortools.sat.python import cp_model

        structural = rules.structural_domain.require_integrity()
        domain = rules.play_domain.require_integrity(structural)
        model = cp_model.CpModel()
        observed: dict[str, list[Any]] = {}
        tables = {
            "sun_strength": domain.sun_strength,
            "hokum_non_trump_strength": domain.hokum_non_trump_strength,
            "hokum_trump_strength": domain.hokum_trump_strength,
            "sun_points": domain.sun_points,
            "hokum_non_trump_points": domain.hokum_non_trump_points,
            "hokum_trump_points": domain.hokum_trump_points,
        }
        for key, configured_table in tables.items():
            table = list(configured_table)
            outputs = []
            for rank in range(8):
                rank_var = model.new_int_var(rank, rank, f"{key}_rank_{rank}")
                output = model.new_int_var(
                    min(table), max(table), f"{key}_value_{rank}"
                )
                model.add_element(rank_var, table, output)
                outputs.append(output)
            observed[key] = outputs

        action = model.new_int_var(0, 1, "unable_to_overtrump_action")
        if not domain.allow_discard_when_cannot_overtrump:
            model.add(action == 0)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        status = solver.solve(model)
        if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
            raise ValueError("play configuration probe is infeasible")

        table_values = {
            key: [int(solver.value(value)) for value in outputs]
            for key, outputs in observed.items()
        }
        legal_actions: list[str] = []
        for candidate, label in ((0, "trump"), (1, "discard")):
            candidate_model = cp_model.CpModel()
            selected = candidate_model.new_int_var(0, 1, "action")
            candidate_model.add(selected == candidate)
            if not domain.allow_discard_when_cannot_overtrump:
                candidate_model.add(selected == 0)
            candidate_solver = cp_model.CpSolver()
            candidate_solver.parameters.num_search_workers = 1
            candidate_solver.parameters.random_seed = 0
            candidate_status = candidate_solver.solve(candidate_model)
            if candidate_status in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
                legal_actions.append(label)

        return {
            "tables": table_values,
            "unableToOvertrumpActions": legal_actions,
            "sha256": canonical_sha256(
                {
                    "tables": table_values,
                    "unableToOvertrumpActions": legal_actions,
                }
            ),
        }

    @staticmethod
    def resolve_first_leader(*, rules: RuleBook, dealer: Seat, buyer: Seat) -> Seat:
        """Lower the governed play policy and structural seat order to CP-SAT."""
        from ortools.sat.python import cp_model

        dealer = Seat.parse(dealer)
        buyer = Seat.parse(buyer)
        structural = rules.structural_domain.require_integrity()
        domain = rules.play_domain.require_integrity(structural)
        model = cp_model.CpModel()
        leader = model.new_int_var(0, 3, "first_leader")
        if domain.first_leader == "dealer_right":
            dealer_var = model.new_int_var(int(dealer), int(dealer), "dealer")
            model.add_allowed_assignments(
                [dealer_var, leader],
                [
                    (int(seat), int(structural.dealer_right(seat)))
                    for seat in structural.play_order
                ],
            )
        elif domain.first_leader == "buyer":
            buyer_var = model.new_int_var(int(buyer), int(buyer), "buyer")
            model.add(leader == buyer_var)
        elif domain.first_leader == "configured":
            configured = model.new_int_var(
                int(domain.configured_first_leader),
                int(domain.configured_first_leader),
                "configured_first_leader",
            )
            model.add(leader == configured)
        else:
            raise AssertionError("validated play domain has an invalid first leader")
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        status = solver.solve(model)
        if status != cp_model.OPTIMAL:
            raise ValueError("first-leader CP-SAT constraint is infeasible")
        return Seat(int(solver.value(leader)))

    @staticmethod
    def scoring_probe(
        *,
        domain: ScoringDomainIRV1,
        contract: Contract,
        declarer: Seat,
        escalation: str,
        trick_raw_points: Sequence[int],
        trick_winner_teams: Sequence[Team | int | str],
        project_points: Sequence[int] = (0, 0),
        baloot_points: Sequence[int] = (0, 0),
        counting_team: Team | None = None,
        starting_match_score: Sequence[int] = (0, 0),
    ) -> dict[str, Any]:
        """Compile a bounded round-and-match scoring observation in CP-SAT.

        The probe is intentionally independent of card allocation. It uses the
        same governed scoring leaves as the complete-play model and makes
        leaf-level counterfactual proofs fast enough to execute at compilation
        time.
        """

        from ortools.sat.python import cp_model

        if not isinstance(domain, ScoringDomainIRV1):
            raise TypeError("scoring_probe requires a compiled ScoringDomainIRV1")
        contract = Contract.parse(contract)
        declarer = Seat.parse(declarer)
        raw_rows = tuple(int(value) for value in trick_raw_points)
        winner_rows = tuple(
            Team[str(value).upper()] if isinstance(value, str) else Team(int(value))
            for value in trick_winner_teams
        )
        project = tuple(int(value) for value in project_points)
        baloot = tuple(int(value) for value in baloot_points)
        starting = tuple(int(value) for value in starting_match_score)
        if len(raw_rows) != 8 or len(winner_rows) != 8 or min(raw_rows) < 0:
            raise ValueError("scoring probe requires eight non-negative trick rows")
        if any(
            len(value) != 2 or min(value) < 0 for value in (project, baloot, starting)
        ):
            raise ValueError("scoring probe requires two non-negative team values")
        if escalation not in {"normal", "double", "triple", "four", "gahwa"}:
            raise ValueError("unsupported escalation")
        counting = Team(
            int(declarer.team.other if counting_team is None else counting_team)
        )
        contract_key = contract.effective.value
        total_cards = domain.card_game_point_totals[contract_key]
        maximum_raw = domain.raw_card_totals[contract_key]
        if sum(raw_rows) + domain.last_trick_bonus_raw != maximum_raw:
            raise ValueError(
                "scoring probe trick rows do not match the governed raw total"
            )

        model = cp_model.CpModel()
        serial = 0

        def next_name(prefix: str) -> str:
            nonlocal serial
            serial += 1
            return f"probe_{prefix}_{serial}"

        def equals_const(variable, value: int, prefix: str):
            result = model.new_bool_var(next_name(prefix))
            model.add(variable == int(value)).only_enforce_if(result)
            model.add(variable != int(value)).only_enforce_if(result.Not())
            return result

        def greater_var(left, right, prefix: str):
            result = model.new_bool_var(next_name(prefix))
            model.add(left > right).only_enforce_if(result)
            model.add(left <= right).only_enforce_if(result.Not())
            return result

        def greater_equal_const(variable, value: int, prefix: str):
            result = model.new_bool_var(next_name(prefix))
            model.add(variable >= int(value)).only_enforce_if(result)
            model.add(variable < int(value)).only_enforce_if(result.Not())
            return result

        raw_team = [
            model.new_int_var(0, maximum_raw, f"probe_raw_{team}") for team in range(2)
        ]
        tricks_won = [
            model.new_int_var(0, 8, f"probe_tricks_{team}") for team in range(2)
        ]
        bonus = domain.last_trick_bonus_raw
        for team in range(2):
            model.add(
                raw_team[team]
                == sum(
                    raw_rows[index]
                    for index, winner in enumerate(winner_rows)
                    if int(winner) == team
                )
                + bonus * int(int(winner_rows[-1]) == team)
            )
            model.add(
                tricks_won[team]
                == sum(int(int(winner) == team) for winner in winner_rows)
            )

        converted = model.new_int_var(
            0, max(total_cards, maximum_raw), "probe_converted"
        )
        model.add_allowed_assignments(
            [raw_team[int(counting)], converted],
            _conversion_table(contract=contract, domain=domain),
        )
        card_game = [
            model.new_int_var(
                0, max(total_cards, maximum_raw), f"probe_card_game_{team}"
            )
            for team in range(2)
        ]
        model.add(card_game[int(counting)] == converted)
        model.add(card_game[int(counting.other)] == total_cards - converted)
        total_bound = max(total_cards, maximum_raw) + sum(project) + sum(baloot)
        totals = [
            model.new_int_var(0, total_bound, f"probe_total_{team}")
            for team in range(2)
        ]
        for team in range(2):
            model.add(totals[team] == card_game[team] + project[team] + baloot[team])

        team_zero_greater = greater_var(totals[0], totals[1], "team_zero_greater")
        team_one_greater = greater_var(totals[1], totals[0], "team_one_greater")
        tied = model.new_bool_var("probe_tied")
        model.add(totals[0] == totals[1]).only_enforce_if(tied)
        model.add(totals[0] != totals[1]).only_enforce_if(tied.Not())
        score_winner = model.new_int_var(0, 1, "probe_score_winner")
        model.add(score_winner == 0).only_enforce_if(team_zero_greater)
        model.add(score_winner == 1).only_enforce_if(team_one_greater)
        tie_winner = model.new_int_var(0, 1, "probe_tie_winner")
        model.add_allowed_assignments(
            [raw_team[int(counting)], tie_winner],
            [
                (
                    raw,
                    int(
                        domain.tie_winner(contract, raw, counting, declarer.team)
                    ),
                )
                for raw, _ in _conversion_table(contract=contract, domain=domain)
            ],
        )
        model.add(score_winner == tie_winner).only_enforce_if(tied)

        zero_tricks = [
            equals_const(tricks_won[team], 0, "zero_tricks") for team in range(2)
        ]
        raw_kabout = model.new_bool_var("probe_raw_kabout")
        model.add_max_equality(raw_kabout, zero_tricks)
        kabout = model.new_bool_var("probe_kabout")
        if domain.kabout_enabled:
            model.add(kabout == raw_kabout)
        else:
            model.add(kabout == 0)
        if domain.kabout_winner == "trick_sweeper":
            total_winner = score_winner
            score_winner = model.new_int_var(0, 1, "probe_effective_score_winner")
            model.add(score_winner == total_winner).only_enforce_if(kabout.Not())
            for team in range(2):
                model.add(score_winner == 1 - team).only_enforce_if([kabout, zero_tricks[team]])
        multiplier = {"normal": 1, "double": 2, "triple": 3, "four": 4, "gahwa": 1}[
            escalation
        ]
        project_cap = domain.project_multiplier_cap
        baloot_cap = domain.baloot_multiplier_cap
        maximum_award = max(
            total_bound * 4,
            domain.kabout_base_game_points[contract_key] * 4
            + sum(project) * 4
            + sum(baloot) * 4,
        )
        awarded = [
            model.new_int_var(0, maximum_award, f"probe_awarded_{team}")
            for team in range(2)
        ]
        winner_flags = [
            equals_const(score_winner, team, "score_winner") for team in range(2)
        ]
        if escalation == "gahwa":
            model.add(awarded[0] == 0)
            model.add(awarded[1] == 0)
        else:
            for team in range(2):
                kabout_award = (
                    domain.kabout_base_game_points[contract_key] * multiplier
                    + project[team] * min(multiplier, project_cap)
                    + baloot[team] * min(multiplier, baloot_cap)
                )
                model.add(awarded[team] == kabout_award).only_enforce_if(
                    [kabout, winner_flags[team]]
                )
                model.add(awarded[team] == 0).only_enforce_if(
                    [kabout, winner_flags[team].Not()]
                )
            if escalation == "normal":
                declarer_wins = equals_const(
                    score_winner, int(declarer.team), "declarer_wins"
                )
                for team in range(2):
                    model.add(awarded[team] == totals[team]).only_enforce_if(
                        [kabout.Not(), declarer_wins]
                    )
                    loss_award = total_cards + sum(project) + sum(baloot)
                    model.add(awarded[team] == loss_award).only_enforce_if(
                        [kabout.Not(), declarer_wins.Not(), winner_flags[team]]
                    )
                    model.add(awarded[team] == 0).only_enforce_if(
                        [kabout.Not(), declarer_wins.Not(), winner_flags[team].Not()]
                    )
            else:
                multiplied_award = (
                    total_cards * multiplier
                    + sum(project) * min(multiplier, project_cap)
                    + sum(baloot) * min(multiplier, baloot_cap)
                )
                for team in range(2):
                    model.add(awarded[team] == multiplied_award).only_enforce_if(
                        [kabout.Not(), winner_flags[team]]
                    )
                    model.add(awarded[team] == 0).only_enforce_if(
                        [kabout.Not(), winner_flags[team].Not()]
                    )

        match_bound = max(starting) + maximum_award + domain.match_target + 1
        final_match = [
            model.new_int_var(0, match_bound, f"probe_match_{team}")
            for team in range(2)
        ]
        if escalation == "gahwa":
            for team in range(2):
                is_winner = equals_const(score_winner, team, "gahwa_winner")
                floor = model.new_int_var(0, match_bound, next_name("gahwa_floor"))
                model.add_max_equality(
                    floor,
                    [domain.match_target, starting[1 - team] + 1],
                )
                model.add(final_match[team] == floor).only_enforce_if(is_winner)
                model.add(final_match[team] == starting[team]).only_enforce_if(
                    is_winner.Not()
                )
        else:
            for team in range(2):
                model.add(final_match[team] == starting[team] + awarded[team])
        target = domain.match_target
        reaches = [
            greater_equal_const(final_match[team], target, "reaches_target")
            for team in range(2)
        ]
        leads = [
            greater_var(final_match[0], final_match[1], "north_south_leads"),
            greater_var(final_match[1], final_match[0], "west_east_leads"),
        ]
        match_winner = model.new_int_var(0, 2, "probe_match_winner")
        winner_conditions = []
        for team in range(2):
            condition = model.new_bool_var(next_name("match_winner_condition"))
            model.add_bool_and([reaches[team], leads[team]]).only_enforce_if(condition)
            model.add_bool_or([reaches[team].Not(), leads[team].Not()]).only_enforce_if(
                condition.Not()
            )
            model.add(match_winner == team).only_enforce_if(condition)
            winner_conditions.append(condition)
        no_winner = model.new_bool_var("probe_no_match_winner")
        model.add_bool_and([item.Not() for item in winner_conditions]).only_enforce_if(
            no_winner
        )
        model.add_bool_or(winner_conditions).only_enforce_if(no_winner.Not())
        model.add(match_winner == 2).only_enforce_if(no_winner)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        status = solver.solve(model)
        if status != cp_model.OPTIMAL:
            raise ValueError("CP-SAT scoring probe is infeasible")
        winner_value = int(solver.value(match_winner))
        value = {
            "rawCardPoints": [int(solver.value(item)) for item in raw_team],
            "cardGamePoints": [int(solver.value(item)) for item in card_game],
            "projectPoints": list(project),
            "balootPoints": list(baloot),
            "totalBeforeAward": [int(solver.value(item)) for item in totals],
            "winner": Team(int(solver.value(score_winner))).name.lower(),
            "awardedPoints": [int(solver.value(item)) for item in awarded],
            "countingTeam": counting.name.lower(),
            "multiplier": multiplier,
            "kabout": bool(solver.boolean_value(kabout)),
            "gahwa": escalation == "gahwa",
            "matchWinner": Team(winner_value).name.lower()
            if winner_value in {0, 1}
            else None,
            "finalMatchScore": [int(solver.value(item)) for item in final_match],
        }
        return {**value, "sha256": canonical_sha256(value)}

    def compile(
        self,
        *,
        hands: Mapping[Seat | str | int, Iterable[Card | int | str | dict]],
        contract: Contract,
        trump: Suit | None,
        first_leader: Seat | None = None,
        declarer: Seat,
        rules: RuleBook,
        dealer: Seat | None = None,
        buyer: Seat | None = None,
        escalation: str = "normal",
        counting_team: Team | None = None,
        project_points: Sequence[int] = (0, 0),
        baloot_points: Sequence[int] = (0, 0),
        locked: bool = False,
        required_cards: Sequence[Card | int | str | dict] | None = None,
        required_ekka_leads: Sequence[bool] | None = None,
        tactical_precursor: Mapping[str, Any] | None = None,
        maximum_conflicts: int = 1_000_000,
        timeout_seconds: float = 60.0,
    ) -> PlayPlanV1:
        from ortools.sat.python import cp_model

        normalized = self._normalize_hands(hands)
        structural_domain: StructuralDomainIRV1 = (
            rules.structural_domain.require_integrity()
        )
        play_domain: PlayDomainIRV1 = rules.play_domain.require_integrity(
            structural_domain
        )
        scoring_domain = ScoringDomainIRV1.from_rules(rules.data)
        contract = Contract.parse(contract)
        if dealer is not None or buyer is not None:
            if dealer is None or buyer is None:
                raise ValueError("dealer and buyer must be supplied together")
            governed_first_leader = self.resolve_first_leader(
                rules=rules,
                dealer=Seat.parse(dealer),
                buyer=Seat.parse(buyer),
            )
            if (
                first_leader is not None
                and Seat.parse(first_leader) != governed_first_leader
            ):
                raise ValueError("first leader conflicts with governed play profile")
            first_leader = governed_first_leader
        elif first_leader is not None:
            first_leader = Seat.parse(first_leader)
        else:
            raise ValueError("first leader requires dealer and buyer context")
        declarer = Seat.parse(declarer)
        if contract.effective == Contract.HOKUM and trump is None:
            raise ValueError("Hokum play compilation requires a trump suit")
        if contract.effective != Contract.HOKUM and trump is not None:
            raise ValueError("Sun/Ashkal play compilation cannot carry a trump suit")
        if escalation not in {"normal", "double", "triple", "four", "gahwa"}:
            raise ValueError("unsupported escalation")
        project = tuple(int(value) for value in project_points)
        baloot = tuple(int(value) for value in baloot_points)
        if len(project) != 2 or len(baloot) != 2 or min((*project, *baloot)) < 0:
            raise ValueError(
                "project and Baloot point inputs must contain two non-negative values"
            )
        required = (
            tuple(Card.parse(card) for card in required_cards)
            if required_cards is not None
            else None
        )
        if required is not None and len(required) != 32:
            raise ValueError("required card line must contain exactly 32 cards")
        raw_ekka_leads = (
            tuple(required_ekka_leads)
            if required_ekka_leads is not None
            else (False,) * 8
        )
        if any(type(value) is not bool for value in raw_ekka_leads):
            raise TypeError("Ekka lead evidence must contain only booleans")
        ekka_leads = tuple(raw_ekka_leads)
        if len(ekka_leads) != 8:
            raise ValueError("Ekka lead evidence must contain eight values")
        if any(ekka_leads) and not play_domain.ekka_enabled:
            raise ValueError("Ekka evidence conflicts with the play domain")
        precursor = copy.deepcopy(dict(tactical_precursor or {}))
        if precursor:
            allowed_precursor_keys = {
                "kind",
                "targetSeat",
                "maximumPayoffTricks",
                "minimumLaterRawPoints",
            }
            if set(precursor) != allowed_precursor_keys:
                raise ValueError(
                    "tactical precursor is incomplete or contains unknown keys"
                )
            if str(precursor["kind"]) != "small_card_big_gain":
                raise ValueError("unsupported tactical play precursor")
            precursor_target = Seat.parse(str(precursor["targetSeat"]))
            precursor_maximum_payoff_tricks = int(precursor["maximumPayoffTricks"])
            precursor_minimum_later_raw_points = int(precursor["minimumLaterRawPoints"])
            if not 1 <= precursor_maximum_payoff_tricks <= 4:
                raise ValueError("tactical precursor payoff depth must be in [1, 4]")
            if not 1 <= precursor_minimum_later_raw_points <= 40:
                raise ValueError("tactical precursor later points must be in [1, 40]")
        else:
            precursor_target = None
            precursor_maximum_payoff_tricks = 0
            precursor_minimum_later_raw_points = 0

        must_follow = play_domain.must_follow_suit
        must_trump = play_domain.must_trump_when_void
        must_overtrump = play_domain.must_overtrump_when_possible
        allow_discard = play_domain.allow_discard_when_cannot_overtrump
        locked_lead = play_domain.locked_hokum_blocks_trump_lead and bool(locked)

        model = cp_model.CpModel()
        serial = 0

        def next_name(prefix: str) -> str:
            nonlocal serial
            serial += 1
            return f"{prefix}_{serial}"

        def fixed_bool(value: bool, prefix: str):
            variable = model.new_bool_var(next_name(prefix))
            model.add(variable == int(value))
            return variable

        def equals_const(variable, value: int, prefix: str):
            result = model.new_bool_var(next_name(prefix))
            model.add(variable == int(value)).only_enforce_if(result)
            model.add(variable != int(value)).only_enforce_if(result.Not())
            return result

        def equals_var(left, right, prefix: str):
            result = model.new_bool_var(next_name(prefix))
            model.add(left == right).only_enforce_if(result)
            model.add(left != right).only_enforce_if(result.Not())
            return result

        def greater_var(left, right, prefix: str):
            result = model.new_bool_var(next_name(prefix))
            model.add(left > right).only_enforce_if(result)
            model.add(left <= right).only_enforce_if(result.Not())
            return result

        def any_bool(values, prefix: str):
            values = list(values)
            if not values:
                return fixed_bool(False, prefix)
            result = model.new_bool_var(next_name(prefix))
            model.add_max_equality(result, values)
            return result

        def all_bool(values, prefix: str):
            values = list(values)
            if not values:
                return fixed_bool(True, prefix)
            result = model.new_bool_var(next_name(prefix))
            model.add_min_equality(result, values)
            return result

        suit_table = [int(Card(card_id).suit) for card_id in range(32)]
        rank_table = [int(Card(card_id).rank) for card_id in range(32)]
        strength_table = [
            play_domain.strength(Card(card_id), contract, trump)
            for card_id in range(32)
        ]
        sun_strength_table = [
            play_domain.strength(Card(card_id), Contract.SUN, None)
            for card_id in range(32)
        ]
        point_table = [
            play_domain.points(Card(card_id), contract, trump) for card_id in range(32)
        ]
        maximum_strength = max(strength_table)
        maximum_trick_points = 4 * max(point_table)

        card_by_trick_seat: list[list[Any]] = []
        suit_by_trick_seat: list[list[Any]] = []
        strength_by_trick_seat: list[list[Any]] = []
        for trick_index in range(8):
            card_row: list[Any] = []
            suit_row: list[Any] = []
            strength_row: list[Any] = []
            for seat in Seat:
                domain = cp_model.Domain.from_values(
                    [card.id for card in normalized[seat]]
                )
                card_var = model.new_int_var_from_domain(
                    domain, f"card_t{trick_index}_s{int(seat)}"
                )
                suit_var = model.new_int_var(0, 3, next_name("seat_suit"))
                strength_var = model.new_int_var(
                    0, maximum_strength, next_name("seat_strength")
                )
                model.add_element(card_var, suit_table, suit_var)
                model.add_element(card_var, strength_table, strength_var)
                card_row.append(card_var)
                suit_row.append(suit_var)
                strength_row.append(strength_var)
            card_by_trick_seat.append(card_row)
            suit_by_trick_seat.append(suit_row)
            strength_by_trick_seat.append(strength_row)
        for seat in Seat:
            model.add_all_different(
                [card_by_trick_seat[index][int(seat)] for index in range(8)]
            )

        leaders: list[Any] = []
        ordered_players: list[list[Any]] = []
        ordered_cards: list[list[Any]] = []
        ordered_suits: list[list[Any]] = []
        ordered_strengths: list[list[Any]] = []
        ordered_points: list[list[Any]] = []
        winner_seats: list[Any] = []
        winner_teams: list[Any] = []
        trick_raw_points: list[Any] = []

        def build_winner(
            *,
            trick_index: int,
            upto: int,
            players: Sequence[Any],
            cards: Sequence[Any],
            suits: Sequence[Any],
            strengths: Sequence[Any],
            led_suit: Any,
        ) -> tuple[Any, Any, Any, Any, Any, Any]:
            is_led = [
                equals_var(suits[position], led_suit, "is_led")
                for position in range(upto)
            ]
            if contract.effective == Contract.HOKUM:
                is_trump = [
                    equals_const(suits[position], int(trump), "is_trump")
                    for position in range(upto)
                ]
                any_trump = any_bool(is_trump, "any_trump")
            else:
                is_trump = [fixed_bool(False, "no_trump") for _ in range(upto)]
                any_trump = fixed_bool(False, "any_trump")
            eligible: list[Any] = []
            effective_strengths: list[Any] = []
            for position in range(upto):
                eligible_var = model.new_bool_var(next_name("eligible"))
                if contract.effective == Contract.HOKUM:
                    model.add(eligible_var == is_trump[position]).only_enforce_if(
                        any_trump
                    )
                    model.add(eligible_var == is_led[position]).only_enforce_if(
                        any_trump.Not()
                    )
                else:
                    model.add(eligible_var == is_led[position])
                effective = model.new_int_var(
                    0, maximum_strength, next_name("effective_strength")
                )
                model.add(effective == strengths[position]).only_enforce_if(
                    eligible_var
                )
                model.add(effective == 0).only_enforce_if(eligible_var.Not())
                eligible.append(eligible_var)
                effective_strengths.append(effective)
            maximum = model.new_int_var(
                0, maximum_strength, next_name("maximum_strength")
            )
            model.add_max_equality(maximum, effective_strengths)
            selected = [
                model.new_bool_var(next_name("winner_position")) for _ in range(upto)
            ]
            model.add_exactly_one(selected)
            for position, selected_var in enumerate(selected):
                model.add(eligible[position] == 1).only_enforce_if(selected_var)
                model.add(effective_strengths[position] == maximum).only_enforce_if(
                    selected_var
                )
            winner_position = model.new_int_var(
                0, upto - 1, next_name("winner_position_index")
            )
            model.add(
                winner_position
                == sum(position * selected[position] for position in range(upto))
            )
            winner_seat = model.new_int_var(0, 3, next_name("winner_seat"))
            winner_team = model.new_int_var(0, 1, next_name("winner_team"))
            winner_card = model.new_int_var(0, 31, next_name("winner_card"))
            winner_suit = model.new_int_var(0, 3, next_name("winner_suit"))
            winner_strength = model.new_int_var(
                0, maximum_strength, next_name("winner_strength")
            )
            model.add_element(winner_position, list(players[:upto]), winner_seat)
            model.add_allowed_assignments(
                [winner_seat, winner_team],
                [(int(seat), int(seat.team)) for seat in structural_domain.play_order],
            )
            model.add_element(winner_position, list(cards[:upto]), winner_card)
            model.add_element(winner_position, list(suits[:upto]), winner_suit)
            model.add_element(winner_position, list(strengths[:upto]), winner_strength)
            return (
                winner_position,
                winner_seat,
                winner_team,
                winner_card,
                winner_suit,
                winner_strength,
            )

        for trick_index in range(8):
            leader = model.new_int_var(0, 3, f"leader_{trick_index}")
            if trick_index == 0:
                model.add(leader == int(first_leader))
            else:
                model.add(leader == winner_seats[trick_index - 1])
            leaders.append(leader)
            players: list[Any] = []
            cards: list[Any] = []
            suits: list[Any] = []
            strengths: list[Any] = []
            points: list[Any] = []
            for position in range(4):
                player = model.new_int_var(0, 3, f"player_t{trick_index}_p{position}")
                model.add_allowed_assignments(
                    [leader, player],
                    [
                        (int(seat), int(structural_domain.offset(seat, position)))
                        for seat in structural_domain.play_order
                    ],
                )
                card = model.new_int_var(
                    0, 31, f"ordered_card_t{trick_index}_p{position}"
                )
                suit = model.new_int_var(0, 3, next_name("ordered_suit"))
                strength = model.new_int_var(
                    0, maximum_strength, next_name("ordered_strength")
                )
                points_var = model.new_int_var(
                    0, max(point_table), next_name("ordered_points")
                )
                model.add_element(player, card_by_trick_seat[trick_index], card)
                model.add_element(card, suit_table, suit)
                model.add_element(card, strength_table, strength)
                model.add_element(card, point_table, points_var)
                players.append(player)
                cards.append(card)
                suits.append(suit)
                strengths.append(strength)
                points.append(points_var)
                if required is not None:
                    model.add(card == required[trick_index * 4 + position].id)
            ordered_players.append(players)
            ordered_cards.append(cards)
            ordered_suits.append(suits)
            ordered_strengths.append(strengths)
            ordered_points.append(points)
            led_suit = suits[0]

            if ekka_leads[trick_index]:
                if contract.effective != Contract.HOKUM or trump is None:
                    raise ValueError("Ekka may only be declared in Hokum")
                model.add(led_suit != int(trump))
                lead_sun_strength = model.new_int_var(
                    0, 7, next_name("lead_sun_strength")
                )
                model.add_element(cards[0], sun_strength_table, lead_sun_strength)
                for future_index in range(trick_index, 8):
                    for seat in structural_domain.play_order:
                        remaining_card = card_by_trick_seat[future_index][int(seat)]
                        remaining_suit = suit_by_trick_seat[future_index][int(seat)]
                        same_suit = equals_var(
                            remaining_suit, led_suit, "ekka_same_suit"
                        )
                        remaining_sun_strength = model.new_int_var(
                            0, 7, next_name("remaining_sun_strength")
                        )
                        model.add_element(
                            remaining_card,
                            sun_strength_table,
                            remaining_sun_strength,
                        )
                        model.add(
                            remaining_sun_strength <= lead_sun_strength
                        ).only_enforce_if(same_suit)

            follow_exists: dict[Seat, Any] = {}
            trump_exists: dict[Seat, Any] = {}
            non_trump_exists: dict[Seat, Any] = {}
            for seat in Seat:
                future_suits = [
                    suit_by_trick_seat[index][int(seat)]
                    for index in range(trick_index, 8)
                ]
                follow_exists[seat] = any_bool(
                    [
                        equals_var(item, led_suit, "remaining_led")
                        for item in future_suits
                    ],
                    "has_follow",
                )
                if contract.effective == Contract.HOKUM:
                    trump_flags = [
                        equals_const(item, int(trump), "remaining_trump")
                        for item in future_suits
                    ]
                    trump_exists[seat] = any_bool(trump_flags, "has_trump")
                    non_trump_exists[seat] = any_bool(
                        [flag.Not() for flag in trump_flags], "has_non_trump"
                    )
                else:
                    trump_exists[seat] = fixed_bool(False, "has_trump")
                    non_trump_exists[seat] = fixed_bool(True, "has_non_trump")

            if contract.effective == Contract.HOKUM and locked_lead:
                for seat in Seat:
                    acts = equals_const(players[0], int(seat), "lead_actor")
                    selected_is_trump = equals_const(
                        suit_by_trick_seat[trick_index][int(seat)],
                        int(trump),
                        "lead_trump",
                    )
                    model.add(selected_is_trump == 0).only_enforce_if(
                        [acts, non_trump_exists[seat]]
                    )

            for position in range(1, 4):
                (
                    _,
                    current_winner_seat,
                    current_winner_team,
                    _,
                    current_winner_suit,
                    current_winner_strength,
                ) = build_winner(
                    trick_index=trick_index,
                    upto=position,
                    players=players,
                    cards=cards,
                    suits=suits,
                    strengths=strengths,
                    led_suit=led_suit,
                )
                led_is_trump = (
                    equals_const(led_suit, int(trump), "led_is_trump")
                    if contract.effective == Contract.HOKUM
                    else fixed_bool(False, "led_is_trump")
                )
                winner_is_trump = (
                    equals_const(current_winner_suit, int(trump), "winner_is_trump")
                    if contract.effective == Contract.HOKUM
                    else fixed_bool(False, "winner_is_trump")
                )
                lead_rank = model.new_int_var(0, 7, next_name("lead_rank"))
                model.add_element(cards[0], rank_table, lead_rank)
                lead_is_ace = equals_const(lead_rank, 7, "lead_is_ace")
                for seat in Seat:
                    acts = equals_const(players[position], int(seat), "actor")
                    selected_suit = suit_by_trick_seat[trick_index][int(seat)]
                    selected_strength = strength_by_trick_seat[trick_index][int(seat)]
                    selected_is_trump = (
                        equals_const(selected_suit, int(trump), "selected_trump")
                        if contract.effective == Contract.HOKUM
                        else fixed_bool(False, "selected_trump")
                    )
                    partner_winning = equals_const(
                        current_winner_team, int(seat.team), "partner_winning"
                    )
                    if must_follow:
                        model.add(selected_suit == led_suit).only_enforce_if(
                            [acts, follow_exists[seat]]
                        )
                    if contract.effective != Contract.HOKUM or not must_trump:
                        continue
                    future_indexes = range(trick_index, 8)
                    higher_flags: list[Any] = []
                    for future_index in future_indexes:
                        future_suit = suit_by_trick_seat[future_index][int(seat)]
                        future_strength = strength_by_trick_seat[future_index][
                            int(seat)
                        ]
                        future_trump = equals_const(
                            future_suit, int(trump), "future_trump"
                        )
                        future_higher = greater_var(
                            future_strength, current_winner_strength, "future_higher"
                        )
                        higher_flags.append(
                            all_bool([future_trump, future_higher], "higher_trump_card")
                        )
                    higher_trump_exists = any_bool(higher_flags, "higher_trump_exists")
                    void_with_trump = [
                        acts,
                        follow_exists[seat].Not(),
                        trump_exists[seat],
                    ]
                    if position == 2:
                        partner_led = equals_const(
                            players[0],
                            int(structural_domain.partner(seat)),
                            "partner_led",
                        )
                        ekka_exception = fixed_bool(
                            ekka_leads[trick_index], "ekka_exception"
                        )
                        partner_exception_enabled = fixed_bool(
                            play_domain.partner_winning_exception
                            == "position_aware_ace_or_ekka",
                            "partner_exception_enabled",
                        )
                        special_cut = all_bool(
                            [
                                partner_exception_enabled,
                                partner_winning,
                                partner_led,
                                lead_is_ace.Not(),
                                ekka_exception.Not(),
                            ],
                            "special_partner_cut",
                        )
                        model.add(selected_is_trump == 1).only_enforce_if(
                            [*void_with_trump, special_cut]
                        )
                    opponent_winning = partner_winning.Not()
                    model.add(selected_is_trump == 1).only_enforce_if(
                        [*void_with_trump, opponent_winning, winner_is_trump.Not()]
                    )
                    if must_overtrump:
                        model.add(selected_is_trump == 1).only_enforce_if(
                            [
                                *void_with_trump,
                                opponent_winning,
                                winner_is_trump,
                                higher_trump_exists,
                            ]
                        )
                        model.add(
                            selected_strength > current_winner_strength
                        ).only_enforce_if(
                            [
                                *void_with_trump,
                                opponent_winning,
                                winner_is_trump,
                                higher_trump_exists,
                            ]
                        )
                        model.add(
                            selected_strength > current_winner_strength
                        ).only_enforce_if(
                            [
                                acts,
                                follow_exists[seat],
                                led_is_trump,
                                opponent_winning,
                                higher_trump_exists,
                            ]
                        )
                    if not allow_discard:
                        model.add(selected_is_trump == 1).only_enforce_if(
                            [
                                *void_with_trump,
                                opponent_winning,
                                winner_is_trump,
                                higher_trump_exists.Not(),
                            ]
                        )

            _, winner_seat, winner_team, _, _, _ = build_winner(
                trick_index=trick_index,
                upto=4,
                players=players,
                cards=cards,
                suits=suits,
                strengths=strengths,
                led_suit=led_suit,
            )
            raw_points = model.new_int_var(
                0, maximum_trick_points, f"raw_points_{trick_index}"
            )
            model.add(raw_points == sum(points))
            winner_seats.append(winner_seat)
            winner_teams.append(winner_team)
            trick_raw_points.append(raw_points)

        contract_key = contract.effective.value
        raw_maximum = scoring_domain.raw_card_totals[contract_key]
        raw_team = [
            model.new_int_var(0, raw_maximum, f"raw_team_{team}") for team in range(2)
        ]
        tricks_won = [
            model.new_int_var(0, 8, f"tricks_won_{team}") for team in range(2)
        ]
        for team in range(2):
            contributions: list[Any] = []
            wins: list[Any] = []
            for trick_index in range(8):
                won = equals_const(winner_teams[trick_index], team, "team_won")
                contribution = model.new_int_var(
                    0, maximum_trick_points, next_name("team_raw_contribution")
                )
                model.add(
                    contribution == trick_raw_points[trick_index]
                ).only_enforce_if(won)
                model.add(contribution == 0).only_enforce_if(won.Not())
                wins.append(won)
                contributions.append(contribution)
            last_won = equals_const(winner_teams[-1], team, "last_trick_team")
            model.add(
                raw_team[team]
                == sum(contributions) + scoring_domain.last_trick_bonus_raw * last_won
            )
            model.add(tricks_won[team] == sum(wins))

        counting = Team(
            int(declarer.team.other if counting_team is None else counting_team)
        )
        total_cards = scoring_domain.card_game_point_totals[contract_key]
        converted = model.new_int_var(0, total_cards, "counting_card_game_points")
        model.add_allowed_assignments(
            [raw_team[int(counting)], converted],
            _conversion_table(contract=contract, domain=scoring_domain),
        )
        card_game = [
            model.new_int_var(0, total_cards, f"card_game_{team}") for team in range(2)
        ]
        model.add(card_game[int(counting)] == converted)
        model.add(card_game[int(counting.other)] == total_cards - converted)
        totals = [
            model.new_int_var(
                0, total_cards + sum(project) + sum(baloot), f"total_{team}"
            )
            for team in range(2)
        ]
        for team in range(2):
            model.add(totals[team] == card_game[team] + project[team] + baloot[team])

        team_zero_greater = greater_var(totals[0], totals[1], "team_zero_greater")
        team_one_greater = greater_var(totals[1], totals[0], "team_one_greater")
        tied = equals_var(totals[0], totals[1], "totals_tied")
        score_winner = model.new_int_var(0, 1, "score_winner")
        model.add(score_winner == 0).only_enforce_if(team_zero_greater)
        model.add(score_winner == 1).only_enforce_if(team_one_greater)
        tie_winner = model.new_int_var(0, 1, "tie_winner")
        tie_rows = [
            (
                raw,
                int(
                    scoring_domain.tie_winner(contract, raw, counting, declarer.team)
                ),
            )
            for raw, _ in _conversion_table(contract=contract, domain=scoring_domain)
        ]
        model.add_allowed_assignments([raw_team[int(counting)], tie_winner], tie_rows)
        model.add(score_winner == tie_winner).only_enforce_if(tied)

        multiplier = {"normal": 1, "double": 2, "triple": 3, "four": 4, "gahwa": 1}[
            escalation
        ]
        kabout_zero = equals_const(tricks_won[0], 0, "kabout_zero")
        kabout_one = equals_const(tricks_won[1], 0, "kabout_one")
        raw_kabout = any_bool([kabout_zero, kabout_one], "raw_kabout")
        kabout = model.new_bool_var("kabout")
        if scoring_domain.kabout_enabled:
            model.add(kabout == raw_kabout)
        else:
            model.add(kabout == 0)
        if scoring_domain.kabout_winner == "trick_sweeper":
            total_winner = score_winner
            score_winner = model.new_int_var(0, 1, "effective_score_winner")
            model.add(score_winner == total_winner).only_enforce_if(kabout.Not())
            model.add(score_winner == 1).only_enforce_if([kabout, kabout_zero])
            model.add(score_winner == 0).only_enforce_if([kabout, kabout_one])
        awarded_maximum = (
            scoring_domain.kabout_base_game_points[contract_key] * max(1, multiplier)
            + sum(project) * max(1, multiplier)
            + sum(baloot) * max(1, multiplier)
        )
        awarded = [
            model.new_int_var(
                0,
                max(
                    total_cards * 4 + sum(project) * 4 + sum(baloot) * 4,
                    awarded_maximum,
                ),
                f"awarded_{team}",
            )
            for team in range(2)
        ]
        winner_flags = [
            equals_const(score_winner, team, "score_winner_team") for team in range(2)
        ]
        if escalation == "gahwa":
            model.add(awarded[0] == 0)
            model.add(awarded[1] == 0)
        else:
            project_cap = scoring_domain.project_multiplier_cap
            baloot_cap = scoring_domain.baloot_multiplier_cap
            for team in range(2):
                kabout_award = (
                    scoring_domain.kabout_base_game_points[contract_key] * multiplier
                    + project[team] * min(multiplier, project_cap)
                    + baloot[team] * min(multiplier, baloot_cap)
                )
                model.add(awarded[team] == kabout_award).only_enforce_if(
                    [kabout, winner_flags[team]]
                )
                model.add(awarded[team] == 0).only_enforce_if(
                    [kabout, winner_flags[team].Not()]
                )
            if escalation == "normal":
                declarer_wins = equals_const(
                    score_winner, int(declarer.team), "declarer_wins"
                )
                for team in range(2):
                    model.add(awarded[team] == totals[team]).only_enforce_if(
                        [kabout.Not(), declarer_wins]
                    )
                    loss_award = total_cards + sum(project) + sum(baloot)
                    model.add(awarded[team] == loss_award).only_enforce_if(
                        [kabout.Not(), declarer_wins.Not(), winner_flags[team]]
                    )
                    model.add(awarded[team] == 0).only_enforce_if(
                        [kabout.Not(), declarer_wins.Not(), winner_flags[team].Not()]
                    )
            else:
                multiplied_award = (
                    total_cards * multiplier
                    + sum(project) * min(multiplier, project_cap)
                    + sum(baloot) * min(multiplier, baloot_cap)
                )
                for team in range(2):
                    model.add(awarded[team] == multiplied_award).only_enforce_if(
                        [kabout.Not(), winner_flags[team]]
                    )
                    model.add(awarded[team] == 0).only_enforce_if(
                        [kabout.Not(), winner_flags[team].Not()]
                    )

        precursor_witnesses: list[tuple[int, int, Any]] = []
        if precursor_target is not None:
            for setup_trick in range(7):
                target_card = card_by_trick_seat[setup_trick][int(precursor_target)]
                target_strength = model.new_int_var(
                    0,
                    maximum_strength,
                    next_name("precursor_strength"),
                )
                model.add_element(target_card, strength_table, target_strength)
                low_card = model.new_bool_var(next_name("precursor_low_card"))
                # This stage only asks for a bottom-three governed strength
                # precursor. Observer-safe causal proof remains mandatory and
                # is intentionally outside this structural CP-SAT objective.
                model.add(target_strength <= 2).only_enforce_if(low_card)
                model.add(target_strength > 2).only_enforce_if(low_card.Not())
                setup_lost = equals_const(
                    winner_teams[setup_trick],
                    int(precursor_target.team.other),
                    "precursor_setup_lost",
                )
                payoff_stop = min(
                    8,
                    setup_trick + precursor_maximum_payoff_tricks + 1,
                )
                for payoff_trick in range(setup_trick + 1, payoff_stop):
                    payoff_won = equals_const(
                        winner_teams[payoff_trick],
                        int(precursor_target.team),
                        "precursor_payoff_won",
                    )
                    payoff_points = model.new_bool_var(
                        next_name("precursor_payoff_points")
                    )
                    model.add(
                        trick_raw_points[payoff_trick]
                        >= precursor_minimum_later_raw_points
                    ).only_enforce_if(payoff_points)
                    model.add(
                        trick_raw_points[payoff_trick]
                        < precursor_minimum_later_raw_points
                    ).only_enforce_if(payoff_points.Not())
                    witness = all_bool(
                        [low_card, setup_lost, payoff_won, payoff_points],
                        "small_card_big_gain_precursor",
                    )
                    precursor_witnesses.append((setup_trick, payoff_trick, witness))
            if not precursor_witnesses:
                raise AssertionError("tactical precursor emitted no witness choices")
            model.add(sum(item[2] for item in precursor_witnesses) >= 1)
            model.maximize(sum(item[2] for item in precursor_witnesses))

        selection_policy = "automatic_single_worker_v1"
        if precursor_target is None and required is None:
            from baloot_platform.core.engine import BalootEngine
            from baloot_platform.core.serialization import replay_trusted_context

            context_dealer = Seat.parse(dealer) if dealer is not None else first_leader
            context_buyer = Seat.parse(buyer) if buyer is not None else declarer
            hint_state = replay_trusted_context(
                {
                    "dealer": context_dealer.name.lower(),
                    "buyer": context_buyer.name.lower(),
                    "declarer": declarer.name.lower(),
                    "contract": contract.value,
                    "trumpSuit": trump.name.lower() if trump is not None else None,
                    "firstLeader": first_leader.name.lower(),
                    "initialHands": {
                        seat.name.lower(): [card.code for card in normalized[seat]]
                        for seat in Seat
                    },
                    "escalation": escalation,
                    "locked": bool(locked),
                },
                rules,
            )
            hint_engine = BalootEngine(rules)
            hint_cards: list[Card] = []
            while hint_state.current_player is not None:
                card = hint_engine.legal_actions(hint_state)[0]
                hint_cards.append(card)
                hint_engine.play_card(hint_state, card, emit_events=False)
            for index, card in enumerate(hint_cards):
                model.add(ordered_cards[index // 4][index % 4] == card.id)
            selection_policy = (
                "authoritative_legal_min_bound_line_single_worker_v1"
            )

        model_core = {
            "schemaVersion": 1,
            "rulesHash": rules.sha256,
            "hands": {
                seat.name.lower(): [card.id for card in normalized[seat]]
                for seat in Seat
            },
            "contract": contract.value,
            "trump": trump.name.lower() if trump is not None else None,
            "firstLeader": first_leader.name.lower(),
            "dealer": Seat.parse(dealer).name.lower() if dealer is not None else None,
            "buyer": Seat.parse(buyer).name.lower() if buyer is not None else None,
            "declarer": declarer.name.lower(),
            "escalation": escalation,
            "countingTeam": counting.name.lower(),
            "projectPoints": list(project),
            "balootPoints": list(baloot),
            "locked": bool(locked),
            "requiredCards": [card.id for card in required]
            if required is not None
            else None,
            "requiredEkkaLeads": list(ekka_leads),
            "tacticalPrecursor": precursor or None,
            "selectionPolicy": selection_policy,
        }
        model_hash = canonical_sha256(model_core)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_number_of_conflicts = int(maximum_conflicts)
        solver.parameters.max_time_in_seconds = float(timeout_seconds)
        status = solver.solve(model)
        if status == cp_model.INFEASIBLE:
            raise ValueError(
                "the requested fixed deal or play line is semantically infeasible"
            )
        if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
            raise RuntimeError(
                "CP-SAT play compiler did not complete: "
                f"{solver.status_name(status)}; conflicts={solver.num_conflicts}; "
                f"branches={solver.num_branches}; wallSeconds={solver.wall_time:.3f}"
            )
        solver_status = "optimal" if status == cp_model.OPTIMAL else "feasible"

        compiled_tricks: list[CompiledTrickV1] = []
        for trick_index in range(8):
            compiled_tricks.append(
                CompiledTrickV1(
                    trick_index,
                    Seat(solver.value(leaders[trick_index])),
                    tuple(
                        Card(solver.value(ordered_cards[trick_index][position]))
                        for position in range(4)
                    ),  # type: ignore[arg-type]
                    Seat(solver.value(winner_seats[trick_index])),
                    solver.value(trick_raw_points[trick_index]),
                    ekka_leads[trick_index],
                )
            )
        precursor_evidence: dict[str, Any] | None = None
        if precursor_target is not None:
            witnesses = []
            for setup_trick, payoff_trick, witness in precursor_witnesses:
                if not solver.boolean_value(witness):
                    continue
                setup_card = Card(
                    solver.value(card_by_trick_seat[setup_trick][int(precursor_target)])
                )
                witnesses.append(
                    {
                        "setupTrick": setup_trick,
                        "setupCard": setup_card.code,
                        "setupCardStrength": play_domain.strength(
                            setup_card,
                            contract,
                            trump,
                        ),
                        "payoffTrick": payoff_trick,
                        "payoffRawPoints": solver.value(trick_raw_points[payoff_trick]),
                    }
                )
            if not witnesses:
                raise AssertionError("CP-SAT tactical precursor lost its witness")
            precursor_evidence = {
                "schemaVersion": 1,
                "kind": "small_card_big_gain",
                "targetSeat": precursor_target.name.lower(),
                "witnesses": witnesses,
                "contract": contract.value,
                "trump": trump.name.lower() if trump is not None else None,
                "strengthBand": {
                    "kind": "governed_bottom_three_v1",
                    "maximum": 2,
                    "playDomainIrHash": play_domain.sha256,
                },
                "structuralPrecursorOnly": True,
                "actorSafe": False,
                "qualificationBypass": False,
            }
            precursor_evidence["sha256"] = canonical_sha256(precursor_evidence)
        core = {
            "schemaVersion": 1,
            "tricks": [item.to_dict() for item in compiled_tricks],
            "score": {
                "rawCardPoints": [solver.value(item) for item in raw_team],
                "cardGamePoints": [solver.value(item) for item in card_game],
                "projectPoints": list(project),
                "balootPoints": list(baloot),
                "totalBeforeAward": [solver.value(item) for item in totals],
                "winner": Team(solver.value(score_winner)).name.lower(),
                "awardedPoints": [solver.value(item) for item in awarded],
                "countingTeam": counting.name.lower(),
                "multiplier": multiplier,
                "kabout": solver.boolean_value(kabout),
                "gahwa": escalation == "gahwa",
            },
            "modelHash": model_hash,
            "solverStatus": solver_status,
            "tacticalPrecursor": precursor_evidence,
        }
        return PlayPlanV1(
            tuple(compiled_tricks),
            tuple(solver.value(item) for item in raw_team),  # type: ignore[arg-type]
            tuple(solver.value(item) for item in card_game),  # type: ignore[arg-type]
            project,
            baloot,
            tuple(solver.value(item) for item in totals),  # type: ignore[arg-type]
            Team(solver.value(score_winner)),
            tuple(solver.value(item) for item in awarded),  # type: ignore[arg-type]
            counting,
            multiplier,
            solver.boolean_value(kabout),
            escalation == "gahwa",
            model_hash,
            solver_status,
            canonical_sha256(core),
            precursor_evidence,
        )

    def verify_trace(
        self,
        trace_state: Mapping[str, Any],
        *,
        rules: RuleBook,
        derived_project_points: Sequence[int] | None = None,
        derived_baloot_points: Sequence[int] | None = None,
    ) -> PlayPlanV1:
        state = copy.deepcopy(dict(trace_state))
        completed = list(state.get("completedTricks") or ())
        if len(completed) != 8 or not isinstance(state.get("roundScore"), Mapping):
            raise ValueError("trace verification requires a fully scored Laffa")
        if (
            str(state.get("phase")) != GamePhase.SCORED.value
            or state.get("currentPlayer") is not None
            or state.get("currentTrick") is not None
        ):
            raise AssertionError(
                "trusted trace is not in the terminal scored lifecycle"
            )

        dealer = Seat.parse(state["dealer"])
        generation = dict(state.get("generation") or {})
        if generation.get("dealer") is not None and (
            Seat.parse(generation["dealer"]) != dealer
        ):
            raise AssertionError("trusted trace dealer differs from frozen generation")
        round_created = [
            dict(event)
            for event in state.get("events") or ()
            if dict(event).get("type") == "round_created"
        ]
        if len(round_created) != 1:
            raise ValueError("trace verification requires one round_created event")
        if Seat.parse(round_created[0]["dealer"]) != dealer:
            raise AssertionError("trusted trace dealer differs from round lifecycle")
        starting_score_sources = [
            tuple(int(value) for value in round_created[0].get("matchScore") or ())
        ]
        if generation.get("matchScore") is not None:
            starting_score_sources.append(
                tuple(int(value) for value in generation["matchScore"])
            )
        if (
            any(
                len(value) != 2 or any(item < 0 for item in value)
                for value in starting_score_sources
            )
            or len(set(starting_score_sources)) != 1
        ):
            raise AssertionError("trusted trace starting match score is inconsistent")
        starting_match_score = starting_score_sources[0]
        initial_hands = dict(state.get("initialHands") or {})
        hands = {
            Seat.parse(seat): tuple(Card.parse(card) for card in cards)
            for seat, cards in initial_hands.items()
        }
        required_cards = tuple(
            Card.parse(play["card"])
            for trick in completed
            for play in list(dict(trick).get("plays") or ())
        )
        ekka_lead_values: list[bool] = []
        for trick in completed:
            plays = list(dict(trick).get("plays") or ())
            if len(plays) != 4:
                raise ValueError("trace verification requires four plays per trick")
            for position, play in enumerate(plays):
                value = dict(play).get("ekka", False)
                if type(value) is not bool:
                    raise ValueError("trace Ekka evidence must be boolean")
                if position > 0 and value:
                    raise ValueError("Ekka is only permitted on a trick lead")
            ekka_lead_values.append(bool(dict(plays[0]).get("ekka", False)))
        ekka_leads = tuple(ekka_lead_values)
        score = dict(state["roundScore"])
        project_points = tuple(
            int(value)
            for value in (
                derived_project_points
                if derived_project_points is not None
                else score.get("projectPoints") or (0, 0)
            )
        )
        baloot_points = tuple(
            int(value)
            for value in (
                derived_baloot_points
                if derived_baloot_points is not None
                else score.get("balootPoints") or (0, 0)
            )
        )
        if project_points != tuple(
            int(value) for value in score.get("projectPoints") or ()
        ):
            raise AssertionError("derived project points differ from the trusted trace")
        if baloot_points != tuple(
            int(value) for value in score.get("balootPoints") or ()
        ):
            raise AssertionError("derived Baloot points differ from the trusted trace")
        plan = self.compile(
            hands=hands,
            contract=Contract.parse(state["contract"]),
            trump=Suit.parse(state["trumpSuit"])
            if state.get("trumpSuit") is not None
            else None,
            first_leader=Seat.parse(state["firstLeader"]),
            declarer=Seat.parse(state["declarer"]),
            rules=rules,
            dealer=dealer,
            buyer=Seat.parse(state["buyer"]),
            escalation=str(state.get("escalation") or "normal"),
            counting_team=Team[str(score["countingTeam"]).upper()],
            project_points=project_points,
            baloot_points=baloot_points,
            locked=bool(state.get("locked")),
            required_cards=required_cards,
            required_ekka_leads=ekka_leads,
        )
        for expected, actual, ekka_lead in zip(
            plan.tricks,
            completed,
            ekka_leads,
            strict=True,
        ):
            actual_trick = dict(actual)
            cards = tuple(
                Card.parse(dict(play)["card"]).id
                for play in actual_trick.get("plays") or ()
            )
            if (
                int(actual_trick.get("index", -1)) != expected.index
                or Seat.parse(actual_trick["leader"]) != expected.leader
                or cards != tuple(card.id for card in expected.cards)
                or Seat.parse(actual_trick["winner"]) != expected.winner
                or int(actual_trick.get("rawPoints", -1)) != expected.raw_points
                or bool(ekka_lead) != expected.ekka_lead
            ):
                raise AssertionError(
                    "CP-SAT trick identity differs from the trusted trace"
                )
        comparisons = {
            "rawCardPoints": plan.raw_card_points,
            "cardGamePoints": plan.card_game_points,
            "projectPoints": plan.project_points,
            "balootPoints": plan.baloot_points,
            "totalBeforeAward": plan.total_before_award,
            "awardedPoints": plan.awarded_points,
        }
        for key, actual in comparisons.items():
            expected = tuple(int(value) for value in score.get(key) or ())
            if actual != expected:
                raise AssertionError(f"CP-SAT {key} differs from the trusted trace")
        scalar_comparisons = {
            "winner": plan.winner.name.lower(),
            "countingTeam": plan.counting_team.name.lower(),
            "multiplier": plan.multiplier,
            "kabout": plan.kabout,
            "gahwa": plan.gahwa,
        }
        if any(score.get(key) != actual for key, actual in scalar_comparisons.items()):
            raise AssertionError("CP-SAT score identity differs from the trusted trace")

        scoring_probe = self.scoring_probe(
            domain=ScoringDomainIRV1.from_rules(rules.data),
            contract=Contract.parse(state["contract"]),
            declarer=Seat.parse(state["declarer"]),
            escalation=str(state.get("escalation") or "normal"),
            trick_raw_points=tuple(trick.raw_points for trick in plan.tricks),
            trick_winner_teams=tuple(trick.winner.team for trick in plan.tricks),
            project_points=plan.project_points,
            baloot_points=plan.baloot_points,
            counting_team=plan.counting_team,
            starting_match_score=starting_match_score,
        )
        probe_round_fields = (
            "rawCardPoints",
            "cardGamePoints",
            "projectPoints",
            "balootPoints",
            "totalBeforeAward",
            "winner",
            "awardedPoints",
            "countingTeam",
            "multiplier",
            "kabout",
            "gahwa",
        )
        expected_plan_score = plan.to_dict()["score"]
        if any(
            scoring_probe[key] != expected_plan_score[key] for key in probe_round_fields
        ):
            raise AssertionError("complete-play and scoring CP-SAT models disagree")
        reference = ReferenceScoringInterpreterV1(
            ScoringDomainIRV1.from_rules(rules.data)
        )
        reference_round = reference.score(
            contract=Contract.parse(state["contract"]),
            declarer=Seat.parse(state["declarer"]),
            escalation=str(state.get("escalation") or "normal"),
            trick_raw_points=tuple(trick.raw_points for trick in plan.tricks),
            trick_winners=tuple(trick.winner for trick in plan.tricks),
            project_points=plan.project_points,
            baloot_points=plan.baloot_points,
            counting_team=plan.counting_team,
        )
        reference_round_score = reference_round.to_dict()
        reference_match = reference.apply_match_award(
            starting_score=starting_match_score,
            awarded_points=reference_round.awarded_points,
            round_winner=reference_round.winner,
            gahwa=reference_round.gahwa,
        )
        reference_match_winner = (
            reference_match.winner.name.lower()
            if reference_match.winner is not None
            else None
        )
        if any(
            reference_round_score[key] != scoring_probe[key]
            for key in probe_round_fields
        ) or (
            list(reference_match.final_score) != scoring_probe["finalMatchScore"]
            or reference_match_winner != scoring_probe["matchWinner"]
        ):
            raise AssertionError("reference and CP-SAT scoring models disagree")
        if list(state.get("matchScore") or ()) != scoring_probe["finalMatchScore"]:
            raise AssertionError(
                "CP-SAT final match score differs from the trusted trace"
            )
        if (
            state.get("matchWinner") != scoring_probe["matchWinner"]
            or score.get("matchWinner") != scoring_probe["matchWinner"]
        ):
            raise AssertionError("CP-SAT match winner differs from the trusted trace")

        round_scored = [
            dict(event)
            for event in state.get("events") or ()
            if dict(event).get("type") == "round_scored"
        ]
        if len(round_scored) != 1:
            raise ValueError("trace verification requires one round_scored event")
        scored_event = round_scored[0]
        event_score = dict(scored_event.get("score") or {})
        if (
            list(scored_event.get("matchScore") or ())
            != scoring_probe["finalMatchScore"]
            or scored_event.get("matchWinner") != scoring_probe["matchWinner"]
            or any(
                event_score.get(key) != scoring_probe[key]
                for key in (*probe_round_fields, "matchWinner")
            )
        ):
            raise AssertionError(
                "round_scored event differs from CP-SAT score identity"
            )
        return plan
