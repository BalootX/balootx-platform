from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, Seat, Suit, Team
from baloot_platform.core.project_ir import ProjectDomainIRV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.plugins.base import PluginDescriptor

from .contracts import canonical_sha256
from .reference_project import ReferenceProjectInterpreterV1


@dataclass(frozen=True, slots=True)
class CompiledProjectV1:
    owner: Seat
    kind: str
    cards: tuple[Card, ...]
    value: int
    high_rank: int
    category_priority: int
    declared: bool = False
    declaration_order: int = 999
    scoring: bool = False
    revealed: bool = False

    @property
    def team(self) -> Team:
        return self.owner.team

    @property
    def card_mask(self) -> int:
        return sum(1 << card.id for card in self.cards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner.name.lower(),
            "team": self.team.name.lower(),
            "kind": self.kind,
            "cards": [card.to_dict() for card in self.cards],
            "value": self.value,
            "highRank": self.high_rank,
            "categoryPriority": self.category_priority,
            "declared": self.declared,
            "declarationOrder": self.declaration_order,
            "scoring": self.scoring,
            "revealed": self.revealed,
        }


@dataclass(frozen=True, slots=True)
class ProjectRevealV1:
    owner: Seat
    trick_index: int
    play_position: int
    project_kinds: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner.name.lower(),
            "trickIndex": self.trick_index,
            "playPosition": self.play_position,
            "projectKinds": list(self.project_kinds),
        }


@dataclass(frozen=True, slots=True)
class ProjectPlanV1:
    projects: tuple[CompiledProjectV1, ...]
    winning_team: Team | None
    project_points: tuple[int, int]
    baloot_teams: frozenset[Team]
    baloot_points: tuple[int, int]
    reveal_schedule: tuple[ProjectRevealV1, ...]
    model_hash: str
    solver_status: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "projects": [project.to_dict() for project in self.projects],
            "winningTeam": self.winning_team.name.lower() if self.winning_team is not None else None,
            "projectPoints": list(self.project_points),
            "balootTeams": [team.name.lower() for team in sorted(self.baloot_teams, key=int)],
            "balootPoints": list(self.baloot_points),
            "revealSchedule": [item.to_dict() for item in self.reveal_schedule],
            "modelHash": self.model_hash,
            "solverStatus": self.solver_status,
            "sha256": self.sha256,
        }


def _candidates(
    owner: Seat,
    hand: tuple[Card, ...],
    contract: Contract,
    domain: ProjectDomainIRV1,
) -> tuple[CompiledProjectV1, ...]:
    reference = ReferenceProjectInterpreterV1(domain)
    return tuple(
        CompiledProjectV1(
            project.owner,
            project.kind,
            project.cards,
            project.value,
            project.high_rank,
            project.category_priority,
        )
        for project in reference.candidates(owner, hand, contract)
    )


class OrToolsCpSatProjectCompilerV1:
    """Compile fixed-hand project selection, declaration, comparison and score."""

    descriptor = PluginDescriptor(
        id="ortools_cp_sat_project_compiler_v1",
        kind="domain_lowering",
        version="1.0.0",
        description="Exact fixed-hand project selection and deterministic declaration/scoring",
        capabilities=(
            "sequence_projects",
            "four_of_kind_projects",
            "non_overlapping_selection",
            "declaration_order",
            "project_comparison",
            "winning_team_scoring",
            "baloot_derivation",
            "second_trick_reveal_timing",
        ),
        deterministic=True,
        resource_class="cpu",
        timeout_seconds=15,
        maximum_memory_mb=256,
        contract_id="cp_sat_lowering",
        contract_version="1",
        safety_guards=(
            "compiled_domain_authority",
            "authoritative_legality",
            "hard_gate_preservation",
        ),
    )

    @staticmethod
    def _normalize_hands(hands: Mapping[Seat | str | int, Iterable[Card | int | str | dict]]) -> dict[Seat, tuple[Card, ...]]:
        values = {Seat.parse(seat): tuple(Card.parse(card) for card in cards) for seat, cards in hands.items()}
        if set(values) != set(Seat) or any(len(cards) != 8 for cards in values.values()):
            raise ValueError("project compilation requires four complete eight-card hands")
        if sorted(card.id for cards in values.values() for card in cards) != list(range(32)):
            raise ValueError("project compilation requires every card exactly once")
        return values

    def compile(
        self,
        *,
        hands: Mapping[Seat | str | int, Iterable[Card | int | str | dict]],
        contract: Contract,
        dealer: Seat,
        trump: Suit | None,
        rules: RuleBook,
        second_trick_leader: Seat | None = None,
        timeout_seconds: float = 15.0,
        maximum_conflicts: int = 250_000,
    ) -> ProjectPlanV1:
        from ortools.sat.python import cp_model

        normalized = self._normalize_hands(hands)
        domain = rules.project_domain
        if domain.tie_break_order == "second_trick_reveal" and second_trick_leader is None:
            raise ValueError("project compilation requires the first trick winner")
        if second_trick_leader is not None:
            second_trick_leader = Seat.parse(second_trick_leader)
        reference = ReferenceProjectInterpreterV1(domain)
        contract = Contract.parse(contract)
        dealer = Seat.parse(dealer)
        if contract.effective == Contract.HOKUM and trump is None:
            raise ValueError("Hokum project compilation requires trump")
        if not domain.enabled:
            baloot_teams = reference.complete_baloot_teams(
                normalized, (), contract, trump
            )
            baloot_points = [0, 0]
            for team in baloot_teams:
                baloot_points[int(team)] += domain.baloot_value
            core = {
                "schemaVersion": 1,
                "projects": [],
                "winningTeam": None,
                "projectPoints": [0, 0],
                "balootTeams": [
                    team.name.lower() for team in sorted(baloot_teams, key=int)
                ],
                "balootPoints": baloot_points,
                "revealSchedule": [],
                "rulesHash": rules.sha256,
                "projectDomainIrHash": domain.sha256,
            }
            return ProjectPlanV1(
                (), None, (0, 0), baloot_teams,
                (baloot_points[0], baloot_points[1]), (),
                canonical_sha256(core), "optimal",
                canonical_sha256({**core, "solverStatus": "optimal"}),
            )
        model = cp_model.CpModel()
        group_rows: dict[Seat, list[tuple[tuple[CompiledProjectV1, ...], tuple[int, int, int, int], Any]]] = {}
        objective_terms: list[Any] = []
        candidate_manifest: dict[str, list[dict[str, Any]]] = {}
        for seat in Seat:
            candidates = _candidates(seat, normalized[seat], contract, domain)
            groups: list[tuple[CompiledProjectV1, ...]] = [()]
            for size in range(
                1, min(domain.maximum_per_player, len(candidates)) + 1
            ):
                for group in combinations(candidates, size):
                    if not domain.group_is_valid(group):
                        continue
                    groups.append(group)
            selected = [model.new_bool_var(f"project_group_s{int(seat)}_{index}") for index in range(len(groups))]
            model.add_exactly_one(selected)
            maximum_category = max((project.category_priority for project in candidates), default=0)
            maximum_high = max((project.high_rank for project in candidates), default=0)
            maximum_cards = 16
            rows: list[tuple[tuple[CompiledProjectV1, ...], tuple[int, int, int, int], Any]] = []
            for ordinal, (group, literal) in enumerate(zip(groups, selected)):
                score = domain.group_score(group)
                card_preference = maximum_cards + score[3]
                key = (
                    ((((score[0] * (maximum_category + 1)) + score[1]) * (maximum_high + 1) + score[2]) * (maximum_cards + 1) + card_preference)
                    * (len(groups) + 1) + (len(groups) - ordinal)
                )
                objective_terms.append(key * literal)
                rows.append((group, score, literal))
            group_rows[seat] = rows
            candidate_manifest[seat.name.lower()] = [project.to_dict() for project in candidates]
        model.maximize(sum(objective_terms))
        model_hash = canonical_sha256({
            "schemaVersion": 1,
            "rulesHash": rules.sha256,
            "projectDomainIrHash": domain.sha256,
            "contract": contract.value,
            "dealer": dealer.name.lower(),
            **({"secondTrickLeader": second_trick_leader.name.lower()} if second_trick_leader is not None else {}),
            "trump": trump.name.lower() if trump is not None else None,
            "hands": {seat.name.lower(): [card.id for card in normalized[seat]] for seat in Seat},
            "candidates": candidate_manifest,
        })
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = float(timeout_seconds)
        solver.parameters.max_number_of_conflicts = int(maximum_conflicts)
        status = solver.solve(model)
        if status != cp_model.OPTIMAL:
            raise RuntimeError("CP-SAT project compiler did not prove the exact selection: " + solver.status_name(status))
        projects: list[CompiledProjectV1] = []
        for seat in Seat:
            chosen = [group for group, _score, literal in group_rows[seat] if solver.boolean_value(literal)]
            if len(chosen) != 1:
                raise AssertionError("project compiler did not select exactly one group")
            projects.extend(chosen[0])

        declaration_seats = list(domain.declaration_seats(dealer))
        projects, winning_team = domain.resolve_declarations(
            projects,
            dealer=dealer,
            trump_suit=trump,
            second_trick_leader=second_trick_leader,
        )
        points = domain.project_points(projects)
        baloot_teams = domain.complete_baloot_teams(
            normalized,
            projects,
            contract,
            trump,
        )
        baloot_points = [0, 0]
        for team in baloot_teams:
            baloot_points[int(team)] += domain.baloot_value
        projects.sort(key=lambda project: (declaration_seats.index(project.owner), -project.value, -project.category_priority, -project.high_rank))
        core = {
            "schemaVersion": 1,
            "projects": [project.to_dict() for project in projects],
            "winningTeam": winning_team.name.lower() if winning_team is not None else None,
            "projectPoints": list(points),
            "balootTeams": [team.name.lower() for team in sorted(baloot_teams, key=int)],
            "balootPoints": baloot_points,
            "revealSchedule": [],
            "modelHash": model_hash,
            "solverStatus": "optimal",
        }
        return ProjectPlanV1(
            tuple(projects), winning_team, points,
            baloot_teams, (baloot_points[0], baloot_points[1]),
            (), model_hash, "optimal", canonical_sha256(core),
        )

    @staticmethod
    def with_reveal_schedule(
        plan: ProjectPlanV1,
        second_trick_order: Iterable[Seat | str | int],
        *,
        domain: ProjectDomainIRV1,
        trump_suit: Suit | None = None,
    ) -> ProjectPlanV1:
        if not isinstance(domain, ProjectDomainIRV1):
            raise TypeError("project reveal requires a compiled ProjectDomainIRV1")
        order = tuple(Seat.parse(seat) for seat in second_trick_order)
        if len(order) != 4 or set(order) != set(Seat):
            raise ValueError("project reveal timing requires all four second-trick seats")
        projects = list(plan.projects)
        schedule: list[ProjectRevealV1] = []
        if not domain.should_reveal(1):
            raise ValueError("compiled project profile does not reveal during trick two")
        for position, seat in enumerate(order):
            projects, revealed_projects = domain.reveal_for_turn(
                projects, seat, trump_suit=trump_suit, second_trick_leader=order[0],
            )
            revealed = tuple(project.kind for project in revealed_projects)
            if not revealed:
                continue
            schedule.append(ProjectRevealV1(seat, 1, position, revealed))
        core = {
            "schemaVersion": 1,
            "projects": [project.to_dict() for project in projects],
            "winningTeam": plan.winning_team.name.lower() if plan.winning_team is not None else None,
            "projectPoints": list(plan.project_points),
            "balootTeams": [team.name.lower() for team in sorted(plan.baloot_teams, key=int)],
            "balootPoints": list(plan.baloot_points),
            "revealSchedule": [item.to_dict() for item in schedule],
            "modelHash": plan.model_hash,
            "solverStatus": plan.solver_status,
        }
        return ProjectPlanV1(
            tuple(projects), plan.winning_team, plan.project_points,
            plan.baloot_teams, plan.baloot_points, tuple(schedule),
            plan.model_hash, plan.solver_status, canonical_sha256(core),
        )

    def verify_trace(self, trace_state: Mapping[str, Any], *, rules: RuleBook) -> ProjectPlanV1:
        state = dict(trace_state)
        completed_tricks = list(state.get("completedTricks") or ())
        if len(completed_tricks) < 2:
            raise ValueError("project reveal verification requires a completed second trick")
        second_trick_order = [play["seat"] for play in completed_tricks[1].get("plays") or ()]
        second_trick_leader = (Seat.parse(completed_tricks[0]["winner"])
            if rules.project_domain.tie_break_order == "second_trick_reveal" else None)
        if second_trick_leader is not None and (
            not second_trick_order or Seat.parse(second_trick_order[0]) != second_trick_leader
        ):
            raise ValueError("project reveal order must start at the first trick winner")
        trump = Suit.parse(state["trumpSuit"]) if state.get("trumpSuit") is not None else None
        hands = {
            Seat.parse(seat): tuple(Card.parse(card) for card in cards)
            for seat, cards in dict(state.get("initialHands") or {}).items()
        }
        plan = self.compile(
            hands=hands,
            contract=Contract.parse(state["contract"]),
            dealer=Seat.parse(state["dealer"]),
            trump=trump,
            rules=rules,
            second_trick_leader=second_trick_leader,
        )
        plan = self.with_reveal_schedule(
            plan,
            second_trick_order,
            domain=rules.project_domain,
            trump_suit=trump,
        )
        expected_projects = list(state.get("projects") or ())
        expected_metadata = sorted(
            (
                str(project["owner"]), str(project["kind"]), int(project["value"]),
                int(project["highRank"]),
                int(project.get("declarationOrder", 999)) != 999,
                int(project.get("declarationOrder", 999)),
                bool(project.get("scoring")),
                bool(project.get("revealed")),
            )
            for project in expected_projects
        )
        actual_metadata = sorted(
            (
                project.owner.name.lower(), project.kind, project.value,
                project.high_rank, project.declared, project.declaration_order,
                project.scoring, project.revealed,
            )
            for project in plan.projects
        )
        if actual_metadata != expected_metadata:
            raise AssertionError("CP-SAT project selection or declaration differs from the trusted trace")
        expected_team = state.get("projectWinningTeam")
        actual_team = plan.winning_team.name.lower() if plan.winning_team is not None else None
        if actual_team != expected_team:
            raise AssertionError("CP-SAT project winner differs from the trusted trace")
        round_score = dict(state.get("roundScore") or {})
        if list(plan.project_points) != list(round_score.get("projectPoints") or ()):
            raise AssertionError("CP-SAT project points differ from the trusted trace")
        expected_baloot_teams = frozenset(
            Team[str(team).upper()] for team in state.get("balootTeams") or ()
        )
        if plan.baloot_teams != expected_baloot_teams:
            raise AssertionError("compiled Baloot derivation differs from the trusted trace")
        if list(plan.baloot_points) != list(round_score.get("balootPoints") or ()):
            raise AssertionError("compiled Baloot points differ from the trusted trace")
        return plan
