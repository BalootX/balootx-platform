from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from typing import Any, Callable, Iterable, Mapping, Sequence

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, Rank, Seat, Suit, Team
from baloot_platform.core.project_ir import (
    ProjectDeclarationActionV1,
    ProjectDomainIRV1,
    project_id_v1,
)

from .ortools_information_set_proof import _solve_cp_sat_with_cancel_v1


def _hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ProjectLifecycleEvidenceV1:
    project_id: str
    owner: Seat
    kind: str
    cards: tuple[Card, ...]
    value: int
    high_rank: int
    category_priority: int
    eligible: bool
    selected: bool
    declarable: bool
    declared: bool
    declaration_order: int | None
    comparison_winner: bool
    awarded: bool

    @property
    def team(self) -> Team:
        return self.owner.team

    @property
    def card_mask(self) -> int:
        return sum(1 << card.id for card in self.cards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "projectId": self.project_id,
            "owner": self.owner.name.lower(),
            "team": self.team.name.lower(),
            "kind": self.kind,
            "cards": [card.id for card in self.cards],
            "value": self.value,
            "highRank": self.high_rank,
            "categoryPriority": self.category_priority,
            "eligible": self.eligible,
            "selected": self.selected,
            "declarable": self.declarable,
            "declared": self.declared,
            "declarationOrder": self.declaration_order,
            "comparisonWinner": self.comparison_winner,
            "awarded": self.awarded,
        }


@dataclass(frozen=True, slots=True)
class OrToolsProjectLifecycleResultV1:
    projects: tuple[ProjectLifecycleEvidenceV1, ...]
    winning_team: Team | None
    eligible_count: int
    selected_count: int
    declared_count: int
    awarded_count: int
    combined_awarded_value: tuple[int, int]
    baloot_seats: frozenset[Seat]
    baloot_teams: frozenset[Team]
    baloot_points: tuple[int, int]
    model_hash: str
    evidence_hash: str
    solver_status: str
    conflicts: int
    branches: int
    declaration_actions: tuple[ProjectDeclarationActionV1, ...] = ()
    variable_count: int = 0
    constraint_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schemaVersion": 1,
            "projects": [project.to_dict() for project in self.projects],
            "winningTeam": (
                self.winning_team.name.lower()
                if self.winning_team is not None
                else None
            ),
            "eligibleCount": self.eligible_count,
            "selectedCount": self.selected_count,
            "declaredCount": self.declared_count,
            "awardedCount": self.awarded_count,
            "combinedAwardedValue": list(self.combined_awarded_value),
            "balootSeats": [
                seat.name.lower() for seat in sorted(self.baloot_seats, key=int)
            ],
            "balootTeams": [
                team.name.lower() for team in sorted(self.baloot_teams, key=int)
            ],
            "balootPoints": list(self.baloot_points),
            "modelHash": self.model_hash,
            "evidenceHash": self.evidence_hash,
            "solverStatus": self.solver_status,
            "conflicts": self.conflicts,
            "branches": self.branches,
            "variableCount": self.variable_count,
            "constraintCount": self.constraint_count,
        }
        if self.declaration_actions:
            result["declarationActions"] = [
                action.to_dict() for action in self.declaration_actions
            ]
        return result


@dataclass(frozen=True, slots=True)
class _Candidate:
    project_id: str
    owner: Seat
    kind: str
    cards: tuple[Card, ...]
    value: int
    high_rank: int
    category_priority: int

    @property
    def team(self) -> Team:
        return self.owner.team

    @property
    def card_mask(self) -> int:
        return sum(1 << card.id for card in self.cards)


def _normalize_hands(
    hands: Mapping[Seat | str | int, Iterable[Card | int | str | dict]],
) -> dict[Seat, tuple[Card, ...]]:
    try:
        normalized = {
            Seat.parse(seat): tuple(sorted((Card.parse(card) for card in cards), key=lambda card: card.id))
            for seat, cards in hands.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("project stage hands are invalid") from exc
    if set(normalized) != set(Seat) or any(
        len(cards) != 8 for cards in normalized.values()
    ):
        raise ValueError("project stage requires four complete eight-card hands")
    if sorted(card.id for cards in normalized.values() for card in cards) != list(
        range(32)
    ):
        raise ValueError("project stage requires every card exactly once")
    return normalized


def _validate_domain(domain: ProjectDomainIRV1) -> None:
    if not isinstance(domain, ProjectDomainIRV1):
        raise TypeError("project stage requires ProjectDomainIRV1")
    supported = (
        domain.declaration_timing == "first_trick_turn"
        and domain.comparison_timing == "after_first_trick"
        and domain.reveal_timing == "during_second_trick"
        and domain.maximum_per_player > 0
        and domain.only_winning_team_scores
        and domain.baloot_only_hokum
        and domain.baloot_declare_on_second_card
        and domain.baloot_siri_or_fifty_inclusion == "automatic"
        and domain.baloot_hundred_inclusion in {"automatic", "blocked"}
        and domain.baloot_single_card_policy == "blocked"
    )
    if not supported:
        raise ValueError("active project rules are unsupported by CP-SAT project stage")


def _bind_any(model: Any, target: Any, literals: Sequence[Any]) -> None:
    if literals:
        model.add_max_equality(target, list(literals))
    else:
        model.add(target == 0)


class OrToolsProjectLifecycleModelV1:
    """Engine-free, exact CP-SAT project lifecycle for fixed complete hands."""

    def solve(
        self,
        *,
        hands: Mapping[Seat | str | int, Iterable[Card | int | str | dict]],
        contract: Contract | str,
        dealer: Seat | str | int,
        trump: Suit | str | int | None,
        domain: ProjectDomainIRV1,
        second_trick_leader: Seat | str | int | None = None,
        timeout_seconds: float = 15.0,
        maximum_conflicts: int = 250_000,
        construction_plan_hash: str | None = None,
        construction_predicate_ids: Sequence[str] = (),
        required_witness: tuple[int, str, int, Sequence[int]] | None = None,
        required_witness_state: str | None = None,
        required_declaration_actions: Sequence[
            ProjectDeclarationActionV1 | Mapping[str, Any] | str
        ] = (),
        required_declaration_actions_complete: bool = False,
        cancel_check: Callable[[], bool] | None = None,
    ) -> OrToolsProjectLifecycleResultV1:
        from ortools.sat.python import cp_model

        _validate_domain(domain)
        if domain.tie_break_order == "second_trick_reveal":
            if second_trick_leader is None:
                raise ValueError("project comparison requires the second-trick leader")
            second_trick_leader = Seat.parse(second_trick_leader)
        normalized = _normalize_hands(hands)
        contract = Contract.parse(contract)
        dealer = Seat.parse(dealer)
        parsed_trump = Suit.parse(trump) if trump is not None else None
        if contract.effective == Contract.HOKUM and parsed_trump is None:
            raise ValueError("Hokum project stage requires trump")
        if contract.effective == Contract.SUN and parsed_trump is not None:
            raise ValueError("Sun project stage cannot define trump")
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ) or float(timeout_seconds) <= 0:
            raise ValueError("timeout_seconds must be positive")
        if type(maximum_conflicts) is not int or maximum_conflicts <= 0:
            raise ValueError("maximum_conflicts must be a positive integer")
        if construction_plan_hash is not None and (
            type(construction_plan_hash) is not str
            or len(construction_plan_hash) != 64
        ):
            raise ValueError("construction_plan_hash must be a SHA-256 digest")
        predicate_ids = tuple(sorted(set(construction_predicate_ids)))
        if any(type(item) is not str or not item for item in predicate_ids):
            raise ValueError("construction predicate IDs must be non-empty strings")
        witness_key = (
            (
                int(required_witness[0]),
                str(required_witness[1]),
                int(required_witness[2]),
                tuple(int(card) for card in required_witness[3]),
            )
            if required_witness is not None
            else None
        )
        lifecycle_states = {
            "eligible",
            "selected",
            "declarable",
            "declared",
            "comparison_winner",
            "awarded",
        }
        if witness_key is None:
            if required_witness_state is not None:
                raise ValueError("required_witness_state requires a project witness")
        elif required_witness_state not in lifecycle_states:
            raise ValueError("required project witness state is unsupported")

        candidates: list[_Candidate] = []
        by_seat: dict[Seat, list[int]] = {seat: [] for seat in Seat}
        if domain.enabled:
            for seat in Seat:
                for project in domain.candidates(
                    seat, normalized[seat], contract
                ):
                    index = len(candidates)
                    candidates.append(
                        _Candidate(
                            project_id_v1(project),
                            project.owner,
                            project.kind,
                            tuple(project.cards),
                            int(project.value),
                            int(project.high_rank),
                            int(project.category_priority),
                        )
                    )
                    by_seat[seat].append(index)

        seat_order = domain.declaration_seats(dealer)
        if type(required_declaration_actions_complete) is not bool:
            raise TypeError(
                "required_declaration_actions_complete must be a boolean"
            )
        if len(required_declaration_actions) > len(seat_order):
            raise ValueError("project declaration action prefix is too long")
        required_actions = tuple(
            ProjectDeclarationActionV1.build(action, actor=seat_order[index])
            for index, action in enumerate(required_declaration_actions)
        )
        if required_declaration_actions_complete and len(required_actions) != len(
            seat_order
        ):
            raise ValueError(
                "complete project declaration actions require all four seats"
            )
        candidate_by_id = {
            candidate.project_id: candidate for candidate in candidates
        }
        for action in required_actions:
            if action.project_id is None:
                continue
            candidate = candidate_by_id.get(action.project_id)
            if candidate is None or candidate.owner != action.actor:
                raise ValueError(
                    "project declaration action references an unknown project"
                )

        model = cp_model.CpModel()
        selected = [
            model.new_bool_var(f"project_selected_{index}")
            for index in range(len(candidates))
        ]
        announcement = [
            model.new_bool_var(f"project_announcement_{index}")
            for index in range(len(candidates))
        ]
        declarable = [
            model.new_bool_var(f"project_declarable_{index}")
            for index in range(len(candidates))
        ]
        declared = [
            model.new_bool_var(f"project_declared_{index}")
            for index in range(len(candidates))
        ]
        declaration_order = [
            model.new_int_var(0, 4, f"project_declaration_order_{index}")
            for index in range(len(candidates))
        ]
        comparison_winner = [
            model.new_bool_var(f"project_comparison_winner_{index}")
            for index in range(len(candidates))
        ]
        awarded = [
            model.new_bool_var(f"project_awarded_{index}")
            for index in range(len(candidates))
        ]

        objective_terms: list[Any] = []
        groups_manifest: dict[str, list[list[str]]] = {}
        for seat in Seat:
            indexes = by_seat[seat]
            groups: list[tuple[int, ...]] = [()]
            for size in range(1, min(domain.maximum_per_player, len(indexes)) + 1):
                for group in combinations(indexes, size):
                    if domain.group_is_valid([candidates[index] for index in group]):
                        groups.append(group)
            group_literals = [
                model.new_bool_var(f"project_group_s{int(seat)}_{ordinal}")
                for ordinal in range(len(groups))
            ]
            model.add_exactly_one(group_literals)
            groups_manifest[seat.name.lower()] = [
                [candidates[index].project_id for index in group]
                for group in groups
            ]

            maximum_category = max(
                (candidates[index].category_priority for index in indexes),
                default=0,
            )
            maximum_high = max(
                (candidates[index].high_rank for index in indexes), default=0
            )
            maximum_cards = 16
            for ordinal, (group, literal) in enumerate(zip(groups, group_literals)):
                score = domain.group_score([candidates[index] for index in group])
                card_preference = maximum_cards + score[3]
                key = (
                    (
                        (
                            (score[0] * (maximum_category + 1) + score[1])
                            * (maximum_high + 1)
                            + score[2]
                        )
                        * (maximum_cards + 1)
                        + card_preference
                    )
                    * (len(groups) + 1)
                    + len(groups)
                    - ordinal
                )
                objective_terms.append(key * literal)

            for index in indexes:
                containing = [
                    literal
                    for group, literal in zip(groups, group_literals)
                    if index in group
                ]
                model.add(selected[index] == sum(containing))
                announcing = [
                    literal
                    for group, literal in zip(groups, group_literals)
                    if group
                    and index
                    == max(
                        group,
                        key=lambda item: (
                            candidates[item].value,
                            candidates[item].category_priority,
                            candidates[item].high_rank,
                        ),
                    )
                ]
                model.add(announcement[index] == sum(announcing))

        ordered_indexes = [index for seat in seat_order for index in by_seat[seat]]

        def rank_key(index: int) -> tuple[int, int, int, int]:
            return domain.rank_key(candidates[index], trump_suit=parsed_trump)

        for position, index in enumerate(ordered_indexes):
            prior = ordered_indexes[:position]
            prior_greater = [
                declared[other]
                for other in prior
                if rank_key(other) > rank_key(index)
            ]
            # Declarability is the selected group's announcement capability.
            # Whether a prior, stronger declaration blocks the actual action
            # belongs to ``declared``.  Keep this identical to the reusable
            # native fragment so generated evidence can be rebound exactly.
            model.add(declarable[index] == announcement[index])
            for other in prior:
                if rank_key(other) > rank_key(index):
                    model.add(declared[index] + declared[other] <= 1)
            model.add(declared[index] <= declarable[index])
            if domain.declaration_mode == "mandatory":
                model.add(
                    declared[index] >= declarable[index] - sum(prior_greater)
                )
            model.add(declaration_order[index] == 0).only_enforce_if(
                declared[index].Not()
            )
            model.add(
                declaration_order[index]
                == 1 + sum(declared[other] for other in prior)
            ).only_enforce_if(declared[index])

        for index in range(len(candidates)):
            if index not in ordered_indexes:
                model.add(declarable[index] == 0)
                model.add(declared[index] == 0)
                model.add(declaration_order[index] == 0)

        pass_action: dict[Seat, Any] = {}
        for seat in seat_order:
            seat_pass = model.new_bool_var(f"project_pass_{int(seat)}")
            model.add(
                seat_pass + sum(declared[index] for index in by_seat[seat]) == 1
            )
            pass_action[seat] = seat_pass
        for action in required_actions:
            if action.action == "pass":
                model.add(pass_action[action.actor] == 1)
                continue
            matching = next(
                index
                for index in by_seat[action.actor]
                if candidates[index].project_id == action.project_id
            )
            model.add(declared[matching] == 1)

        if objective_terms:
            selection_objective = sum(objective_terms)
            if domain.declaration_mode == "optional":
                model.maximize(5 * selection_objective + sum(declared))
            else:
                model.maximize(selection_objective)

        comparison_indexes = ordered_indexes
        if domain.tie_break_order == "second_trick_reveal":
            comparison_indexes = sorted(ordered_indexes, key=lambda index: (
                candidates[index].owner.team != second_trick_leader.team,
                (int(candidates[index].owner) - int(second_trick_leader)) % 4))
        for position, index in enumerate(comparison_indexes):
            disqualifying = [
                declared[other]
                for other_position, other in enumerate(comparison_indexes)
                if other != index
                and (
                    rank_key(other) > rank_key(index)
                    or (
                        rank_key(other) == rank_key(index)
                        and other_position < position
                    )
                )
            ]
            model.add(comparison_winner[index] <= declared[index])
            for literal in disqualifying:
                model.add(comparison_winner[index] + literal <= 1)
            model.add(
                comparison_winner[index]
                >= declared[index] - sum(disqualifying)
            )
        has_declaration = model.new_bool_var("project_has_declaration")
        _bind_any(model, has_declaration, declared)
        model.add(sum(comparison_winner) == has_declaration)

        winning_team = [
            model.new_bool_var(f"project_winning_team_{int(team)}")
            for team in Team
        ]
        for team in Team:
            model.add(
                winning_team[int(team)]
                == sum(
                    comparison_winner[index]
                    for index, candidate in enumerate(candidates)
                    if candidate.team == team
                )
            )
        for index, candidate in enumerate(candidates):
            team_literal = winning_team[int(candidate.team)]
            model.add(awarded[index] <= selected[index])
            model.add(awarded[index] <= team_literal)
            model.add(awarded[index] >= selected[index] + team_literal - 1)

        if witness_key is not None:
            witness_index = next(
                (
                    index
                    for index, candidate in enumerate(candidates)
                    if (
                        int(candidate.owner),
                        candidate.kind,
                        candidate.value,
                        tuple(card.id for card in candidate.cards),
                    )
                    == witness_key
                ),
                None,
            )
            if witness_index is None:
                model.add_bool_or([])
            else:
                witness_literals = {
                    "eligible": model.new_constant(1),
                    "selected": selected[witness_index],
                    "declarable": declarable[witness_index],
                    "declared": declared[witness_index],
                    "comparison_winner": comparison_winner[witness_index],
                    "awarded": awarded[witness_index],
                }
                assumption = model.new_bool_var(
                    f"assume_required_project_witness_{witness_index}"
                )
                model.add(
                    witness_literals[str(required_witness_state)] == 1
                ).only_enforce_if(assumption)
                model.add_assumption(assumption)

        combined_awarded_value = [
            model.new_int_var(0, sum(candidate.value for candidate in candidates), f"project_value_{int(team)}")
            for team in Team
        ]
        for team in Team:
            model.add(
                combined_awarded_value[int(team)]
                == sum(
                    candidate.value * awarded[index]
                    for index, candidate in enumerate(candidates)
                    if candidate.team == team
                )
            )

        baloot_seat = [
            model.new_bool_var(f"baloot_seat_{int(seat)}") for seat in Seat
        ]
        for seat in Seat:
            if (
                not domain.baloot_enabled
                or contract != Contract.HOKUM
                or parsed_trump is None
                or Card.of(parsed_trump, Rank.KING) not in normalized[seat]
                or Card.of(parsed_trump, Rank.QUEEN) not in normalized[seat]
            ):
                model.add(baloot_seat[int(seat)] == 0)
                continue
            pair = {
                Card.of(parsed_trump, Rank.KING),
                Card.of(parsed_trump, Rank.QUEEN),
            }
            included_literals: list[Any] = []
            blocked_literals: list[Any] = []
            for index in by_seat[seat]:
                candidate = candidates[index]
                card_set = set(candidate.cards)
                if (
                    candidate.kind in {"siri", "fifty"}
                    or (
                        candidate.kind == "hundred_sequence"
                        and domain.baloot_hundred_inclusion == "automatic"
                    )
                ) and pair.issubset(card_set):
                    included_literals.append(awarded[index])
                if (
                    candidate.kind == "hundred_sequence"
                    and domain.baloot_hundred_inclusion == "blocked"
                    and pair.issubset(card_set)
                ) or (
                    candidate.kind == "hundred_four_kind"
                    and candidate.cards
                    and candidate.cards[0].rank in {Rank.KING, Rank.QUEEN}
                ) or len(pair & card_set) == 1:
                    blocked_literals.append(awarded[index])
            included = model.new_bool_var(f"baloot_included_{int(seat)}")
            blocked = model.new_bool_var(f"baloot_blocked_{int(seat)}")
            _bind_any(model, included, included_literals)
            _bind_any(model, blocked, blocked_literals)
            model.add(baloot_seat[int(seat)] >= included)
            model.add(baloot_seat[int(seat)] >= 1 - blocked)
            model.add(baloot_seat[int(seat)] <= included + 1 - blocked)

        baloot_team = [
            model.new_bool_var(f"baloot_team_{int(team)}") for team in Team
        ]
        for team in Team:
            _bind_any(
                model,
                baloot_team[int(team)],
                [baloot_seat[int(seat)] for seat in Seat if seat.team == team],
            )

        model_core = {
            "schemaVersion": 1,
            "backend": "ortools_cp_sat_v2",
            "stage": "projects",
            "loweringVersion": "project_lifecycle_v1",
            "projectDomainIrHash": domain.sha256,
            "contract": contract.value,
            "dealer": dealer.name.lower(),
            "trump": parsed_trump.name.lower() if parsed_trump is not None else None,
            "hands": {
                seat.name.lower(): [card.id for card in normalized[seat]]
                for seat in Seat
            },
            "candidates": [
                {
                    "projectId": candidate.project_id,
                    "owner": candidate.owner.name.lower(),
                    "kind": candidate.kind,
                    "cards": [card.id for card in candidate.cards],
                    "value": candidate.value,
                    "highRank": candidate.high_rank,
                    "categoryPriority": candidate.category_priority,
                }
                for candidate in candidates
            ],
            "groups": groups_manifest,
        }
        if domain.tie_break_order == "second_trick_reveal":
            model_core["secondTrickLeader"] = int(second_trick_leader)
        if domain.declaration_mode_authored:
            model_core.update(
                {
                    "declarationMode": domain.declaration_mode,
                    "requiredDeclarationActions": [
                        action.to_dict() for action in required_actions
                    ],
                    "requiredDeclarationActionsComplete": (
                        required_declaration_actions_complete
                    ),
                }
            )
        if construction_plan_hash is not None:
            model_core.update(
                {
                    "constructionPlanHash": construction_plan_hash,
                    "constructionPredicateIds": list(predicate_ids),
                }
            )
        if witness_key is not None:
            model_core.update(
                {
                    "requiredWitness": [
                        witness_key[0],
                        witness_key[1],
                        witness_key[2],
                        list(witness_key[3]),
                    ],
                    "requiredWitnessState": required_witness_state,
                }
            )
        model_hash = _hash(model_core)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_time_in_seconds = float(timeout_seconds)
        solver.parameters.max_number_of_conflicts = maximum_conflicts
        status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
        if status != cp_model.OPTIMAL:
            raise RuntimeError(
                "CP-SAT project stage did not prove the exact lifecycle: "
                + solver.status_name(status)
            )

        projects = tuple(
            ProjectLifecycleEvidenceV1(
                candidate.project_id,
                candidate.owner,
                candidate.kind,
                candidate.cards,
                candidate.value,
                candidate.high_rank,
                candidate.category_priority,
                True,
                solver.boolean_value(selected[index]),
                solver.boolean_value(declarable[index]),
                solver.boolean_value(declared[index]),
                (
                    solver.value(declaration_order[index])
                    if solver.boolean_value(declared[index])
                    else None
                ),
                solver.boolean_value(comparison_winner[index]),
                solver.boolean_value(awarded[index]),
            )
            for index, candidate in enumerate(candidates)
        )
        winner = next(
            (
                team
                for team in Team
                if solver.boolean_value(winning_team[int(team)])
            ),
            None,
        )
        project_values = tuple(
            solver.value(combined_awarded_value[int(team)]) for team in Team
        )
        solved_baloot_seats = frozenset(
            seat for seat in Seat if solver.boolean_value(baloot_seat[int(seat)])
        )
        solved_baloot_teams = frozenset(
            team for team in Team if solver.boolean_value(baloot_team[int(team)])
        )
        baloot_points = tuple(
            domain.baloot_value * int(team in solved_baloot_teams) for team in Team
        )
        evidence_core = {
            "schemaVersion": 1,
            "modelHash": model_hash,
            "solverStatus": "optimal",
            "projects": [project.to_dict() for project in projects],
            "winningTeam": winner.name.lower() if winner is not None else None,
            "combinedAwardedValue": list(project_values),
            "balootSeats": [
                seat.name.lower() for seat in sorted(solved_baloot_seats, key=int)
            ],
            "balootTeams": [
                team.name.lower() for team in sorted(solved_baloot_teams, key=int)
            ],
            "balootPoints": list(baloot_points),
        }
        solved_declaration_actions = tuple(
            ProjectDeclarationActionV1(seat, "pass")
            if solver.boolean_value(pass_action[seat])
            else ProjectDeclarationActionV1(
                seat,
                "declare",
                next(
                    candidates[index].project_id
                    for index in by_seat[seat]
                    if solver.boolean_value(declared[index])
                ),
            )
            for seat in seat_order
        )
        if domain.declaration_mode_authored:
            evidence_core["declarationActions"] = [
                action.to_dict() for action in solved_declaration_actions
            ]
        return OrToolsProjectLifecycleResultV1(
            projects,
            winner,
            len(projects),
            sum(project.selected for project in projects),
            sum(project.declared for project in projects),
            sum(project.awarded for project in projects),
            (int(project_values[0]), int(project_values[1])),
            solved_baloot_seats,
            solved_baloot_teams,
            (int(baloot_points[0]), int(baloot_points[1])),
            model_hash,
            _hash(evidence_core),
            "optimal",
            solver.num_conflicts,
            solver.num_branches,
            solved_declaration_actions if domain.declaration_mode_authored else (),
            len(model.proto.variables),
            len(model.proto.constraints),
        )


__all__ = [
    "OrToolsProjectLifecycleModelV1",
    "OrToolsProjectLifecycleResultV1",
    "ProjectLifecycleEvidenceV1",
]
