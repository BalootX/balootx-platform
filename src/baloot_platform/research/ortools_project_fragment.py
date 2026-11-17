from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Sequence

from baloot_platform.core.cards import FULL_DECK, Card
from baloot_platform.core.enums import Contract, Rank, Seat, Suit, Team
from baloot_platform.core.project_ir import ProjectDeclarationActionV1, ProjectDomainIRV1

from .ortools_project_model import (
    OrToolsProjectLifecycleResultV1,
    ProjectLifecycleEvidenceV1,
    _hash,
    _validate_domain,
)


_LIFECYCLE_STATES = {
    "eligible",
    "selected",
    "declarable",
    "declared",
    "comparison_winner",
    "awarded",
}

NATIVE_FULL_HAND_PROJECT_POLICY_ID = "full_hand_project_max_value_v1"
NATIVE_OBSERVED_PROJECT_POLICY_IDS = ("known_project_max_value_v1", NATIVE_FULL_HAND_PROJECT_POLICY_ID)


def validate_native_observed_project_policy_id_v1(policy_id: str) -> str:
    if type(policy_id) is not str or policy_id not in NATIVE_OBSERVED_PROJECT_POLICY_IDS:
        raise ValueError("unsupported native observed project policy")
    return policy_id


@dataclass(frozen=True, slots=True)
class ProjectPatternVariablesV1:
    project_id: str
    owner: Seat
    kind: str
    cards: tuple[Card, ...]
    value: int
    high_rank: int
    category_priority: int
    eligible: Any
    selected: Any
    declarable: Any
    declared: Any
    declaration_order: Any
    comparison_winner: Any
    awarded: Any

    @property
    def team(self) -> Team:
        return self.owner.team

    @property
    def card_mask(self) -> int:
        return sum(1 << card.id for card in self.cards)

    def literal(self, state: str) -> Any:
        if state not in _LIFECYCLE_STATES:
            raise ValueError(f"unsupported project lifecycle state: {state}")
        literal = getattr(self, state)
        if literal is None:
            raise ValueError(f"project lifecycle state was not compiled: {state}")
        return literal


@dataclass(frozen=True, slots=True)
class ProjectPatternEvidenceV1:
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
class ProjectLifecycleFragmentEvidenceV1:
    projects: tuple[ProjectPatternEvidenceV1, ...]
    winning_team: Team | None
    combined_awarded_value: tuple[int, int]
    baloot_seats: frozenset[Seat]
    baloot_teams: frozenset[Team]
    baloot_points: tuple[int, int]
    card_owners: tuple[Seat, ...]
    fragment_hash: str
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "projects": [project.to_dict() for project in self.projects],
            "winningTeam": (
                self.winning_team.name.lower()
                if self.winning_team is not None
                else None
            ),
            "combinedAwardedValue": list(self.combined_awarded_value),
            "balootSeats": [
                seat.name.lower() for seat in sorted(self.baloot_seats, key=int)
            ],
            "balootTeams": [
                team.name.lower() for team in sorted(self.baloot_teams, key=int)
            ],
            "balootPoints": list(self.baloot_points),
            "cardOwners": [seat.name.lower() for seat in self.card_owners],
            "fragmentHash": self.fragment_hash,
            "evidenceHash": self.evidence_hash,
        }


@dataclass(frozen=True, slots=True)
class ProjectBuyerConstraintV1:
    buyer_team: Any
    count: Any
    matching_literals: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class OrToolsProjectEligibilityFragmentV1:
    """Construction projection, deliberately without declaration/award variables."""

    projects: tuple[ProjectPatternVariablesV1, ...]
    owner_of_card: tuple[Any, ...]
    fragment_hash: str

    def matching_projects(self, **filters: Any) -> tuple[ProjectPatternVariablesV1, ...]:
        return OrToolsProjectLifecycleFragmentV1.matching_projects(self, **filters)

    def extract(self, solver: Any) -> ProjectLifecycleFragmentEvidenceV1:
        raise ValueError("eligibility-only construction has no declaration or award evidence")


@dataclass(frozen=True, slots=True)
class OrToolsProjectDeclarationFragmentV1(OrToolsProjectEligibilityFragmentV1):
    """Pre-comparison constraints; deliberately cannot supply scoring evidence."""

    def extract(self, solver: Any) -> ProjectLifecycleFragmentEvidenceV1:
        raise ValueError("declaration-only construction has no comparison or award evidence")


@dataclass(frozen=True, slots=True)
class OrToolsProjectLifecycleFragmentV1:
    projects: tuple[ProjectPatternVariablesV1, ...]
    owner_of_card: tuple[Any, ...]
    winning_team: tuple[Any, Any]
    combined_awarded_value: tuple[Any, Any]
    baloot_seat: tuple[Any, Any, Any, Any]
    baloot_team: tuple[Any, Any]
    baloot_value: int
    fragment_hash: str
    declaration_actions: tuple[dict[str, Any], ...] | None = None
    declaration_seats: tuple[Seat, ...] = ()

    def known_project_policy_actions(self, policy_id: str, actor: Seat,
            visible_cards: Sequence[Card]) -> dict[str, str]:
        """Choose a proof aid from public IR patterns and the actor's known cards.

        This does not establish eligibility, declaration legality, award, or utility;
        the native constant-policy query must certify those across hidden worlds.
        """
        validate_native_observed_project_policy_id_v1(policy_id)
        if policy_id == NATIVE_FULL_HAND_PROJECT_POLICY_ID:
            raise ValueError("full-hand project policy requires native symbolic declaration constraints")
        known = frozenset(card.id for card in visible_cards)
        candidates = [project for project in self.projects if project.owner == actor
                      and all(card.id in known for card in project.cards)]
        if not candidates:
            return {}
        project = min(candidates, key=lambda row: (-row.value, row.project_id))
        return {actor.name.lower(): ProjectDeclarationActionV1(actor, "declare", project.project_id).code}

    def constrain_native_visible_policy(self, model: Any, *, team: Team, policy_id: str,
            skipped_actors: Sequence[Seat] = ()) -> None:
        """Select the strongest legal announcement from own cards and public declarations.

        This is a strategy witness, not a project-optimality certificate. Eligibility
        and prior-announcement suppression remain native lifecycle variables.
        """
        if (policy_id != NATIVE_FULL_HAND_PROJECT_POLICY_ID or not isinstance(team, Team)
                or self.declaration_actions is None
                or any(not isinstance(actor, Seat) for actor in skipped_actors)):
            raise ValueError("native full-hand project policy requires a team and declaration variables")
        for actor in Seat:
            if actor.team != team or actor in skipped_actors:
                continue
            # The shared lifecycle already selects at most one legal announcement
            # per seat. Reuse it; do not maintain another project ranking.
            for project in (p for p in self.projects if p.owner == actor):
                action = "declare:" + project.project_id
                model.add(project.declared == self.declaration_actions[int(actor)].get(action, 0))

    def extract_declaration_decision(self, solver: Any, actor: Seat) -> dict[str, Any]:
        """Read native legal choices separately from the solved declaration."""
        if self.declaration_actions is None:
            raise ValueError("project fragment has no native declaration action variables")
        actor = Seat.parse(actor)
        legal = [code for code, literal in self.declaration_actions[int(actor)].items()
                 if solver.boolean_value(literal)]
        selected = [ProjectDeclarationActionV1(actor, "declare", project.project_id).code
                    for project in self.projects if project.owner == actor and solver.boolean_value(project.declared)]
        action = selected[0] if selected else "pass"
        if len(selected) > 1 or action not in legal:
            raise ValueError("native project choice disagrees with legal action variables")
        return {"actor": actor.name.lower(), "legalActions": legal, "selectedAction": action}

    @staticmethod
    def _states(states: str | Sequence[str]) -> tuple[str, ...]:
        normalized = (states,) if isinstance(states, str) else tuple(states)
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("project lifecycle states must be non-empty and unique")
        if any(state not in _LIFECYCLE_STATES for state in normalized):
            raise ValueError("unsupported project lifecycle state")
        return normalized

    def matching_projects(
        self,
        *,
        owner: Seat | str | int | None = None,
        team: Team | str | int | None = None,
        kind: str | None = None,
        value: int | None = None,
    ) -> tuple[ProjectPatternVariablesV1, ...]:
        parsed_owner = Seat.parse(owner) if owner is not None else None
        if team is None:
            parsed_team = None
        elif isinstance(team, Team):
            parsed_team = team
        elif isinstance(team, int):
            parsed_team = Team(team)
        else:
            parsed_team = Team[str(team).strip().upper()]
        if value is not None and type(value) is not int:
            raise ValueError("project value must be an integer")
        return tuple(
            project
            for project in self.projects
            if (parsed_owner is None or project.owner == parsed_owner)
            and (parsed_team is None or project.team == parsed_team)
            and (kind is None or project.kind == kind)
            and (value is None or project.value == value)
        )

    def constrain_count(
        self,
        model: Any,
        *,
        states: str | Sequence[str],
        owner: Seat | str | int | None = None,
        team: Team | str | int | None = None,
        kind: str | None = None,
        value: int | None = None,
        minimum: int = 1,
        maximum: int | None = None,
        constraint_id: str = "project_count",
    ) -> tuple[Any, tuple[Any, ...]]:
        normalized_states = self._states(states)
        if type(minimum) is not int or minimum < 0:
            raise ValueError("project count minimum must be a non-negative integer")
        if maximum is not None and (
            type(maximum) is not int or maximum < minimum
        ):
            raise ValueError("project count maximum is invalid")
        matches = self.matching_projects(
            owner=owner, team=team, kind=kind, value=value
        )
        conjunctions = tuple(
            _and_literal(
                model,
                [project.literal(state) for state in normalized_states],
                f"{constraint_id}_{index}",
            )
            for index, project in enumerate(matches)
        )
        count = model.new_int_var(
            0, len(conjunctions), f"{constraint_id}_matching_count"
        )
        model.add(count == sum(conjunctions))
        model.add(count >= minimum)
        if maximum is not None:
            model.add(count <= maximum)
        return count, conjunctions

    def constrain_buyer_team_project(
        self,
        model: Any,
        *,
        buyer_seat: Any,
        states: str | Sequence[str] = "awarded",
        kind: str | None = None,
        value: int | None = None,
        minimum: int = 1,
        maximum: int | None = None,
        constraint_id: str = "buyer_project",
    ) -> ProjectBuyerConstraintV1:
        normalized_states = self._states(states)
        if type(minimum) is not int or minimum < 0:
            raise ValueError("buyer project minimum must be a non-negative integer")
        if maximum is not None and (
            type(maximum) is not int or maximum < minimum
        ):
            raise ValueError("buyer project maximum is invalid")
        buyer_team = model.new_int_var(0, 1, f"{constraint_id}_buyer_team")
        model.add_allowed_assignments(
            [buyer_seat, buyer_team],
            [(int(seat), int(seat.team)) for seat in Seat],
        )
        candidates = self.matching_projects(kind=kind, value=value)
        matching: list[Any] = []
        for index, project in enumerate(candidates):
            same_team = model.new_bool_var(f"{constraint_id}_{index}_same_team")
            model.add(buyer_team == int(project.team)).only_enforce_if(same_team)
            model.add(buyer_team != int(project.team)).only_enforce_if(
                same_team.Not()
            )
            matching.append(
                _and_literal(
                    model,
                    [same_team, *(project.literal(state) for state in normalized_states)],
                    f"{constraint_id}_{index}_matches",
                )
            )
        count = model.new_int_var(0, len(matching), f"{constraint_id}_count")
        model.add(count == sum(matching))
        model.add(count >= minimum)
        if maximum is not None:
            model.add(count <= maximum)
        return ProjectBuyerConstraintV1(buyer_team, count, tuple(matching))

    def extract(self, solver: Any) -> ProjectLifecycleFragmentEvidenceV1:
        owners = tuple(Seat(solver.value(variable)) for variable in self.owner_of_card)
        projects = tuple(
            ProjectPatternEvidenceV1(
                project.project_id,
                project.owner,
                project.kind,
                project.cards,
                project.value,
                project.high_rank,
                project.category_priority,
                solver.boolean_value(project.eligible),
                solver.boolean_value(project.selected),
                solver.boolean_value(project.declarable),
                solver.boolean_value(project.declared),
                (
                    solver.value(project.declaration_order)
                    if solver.boolean_value(project.declared)
                    else None
                ),
                solver.boolean_value(project.comparison_winner),
                solver.boolean_value(project.awarded),
            )
            for project in self.projects
            if solver.boolean_value(project.eligible)
        )
        winner = next(
            (
                team
                for team in Team
                if solver.boolean_value(self.winning_team[int(team)])
            ),
            None,
        )
        awarded_value = tuple(
            solver.value(self.combined_awarded_value[int(team)]) for team in Team
        )
        baloot_seats = frozenset(
            seat
            for seat in Seat
            if solver.boolean_value(self.baloot_seat[int(seat)])
        )
        baloot_teams = frozenset(
            team
            for team in Team
            if solver.boolean_value(self.baloot_team[int(team)])
        )
        baloot_points = tuple(
            self.baloot_value * int(team in baloot_teams) for team in Team
        )
        core = {
            "schemaVersion": 1,
            "fragmentHash": self.fragment_hash,
            "projects": [project.to_dict() for project in projects],
            "winningTeam": winner.name.lower() if winner is not None else None,
            "combinedAwardedValue": list(awarded_value),
            "balootSeats": [
                seat.name.lower() for seat in sorted(baloot_seats, key=int)
            ],
            "balootTeams": [
                team.name.lower() for team in sorted(baloot_teams, key=int)
            ],
            "balootPoints": list(baloot_points),
            "cardOwners": [seat.name.lower() for seat in owners],
        }
        return ProjectLifecycleFragmentEvidenceV1(
            projects,
            winner,
            (int(awarded_value[0]), int(awarded_value[1])),
            baloot_seats,
            baloot_teams,
            (int(baloot_points[0]), int(baloot_points[1])),
            owners,
            self.fragment_hash,
            _hash(core),
        )


def project_lifecycle_result_v1(
    fragment: OrToolsProjectLifecycleFragmentV1,
    solver: Any,
) -> OrToolsProjectLifecycleResultV1:
    evidence = fragment.extract(solver)
    projects = tuple(
        ProjectLifecycleEvidenceV1(
            item.project_id,
            item.owner,
            item.kind,
            item.cards,
            item.value,
            item.high_rank,
            item.category_priority,
            item.eligible,
            item.selected,
            item.declarable,
            item.declared,
            item.declaration_order,
            item.comparison_winner,
            item.awarded,
        )
        for item in evidence.projects
    )
    declaration_actions = tuple(
        ProjectDeclarationActionV1(seat, "declare", project.project_id)
        if (project := next((row for row in projects if row.owner == seat and row.declared), None))
        else ProjectDeclarationActionV1(seat, "pass")
        for seat in fragment.declaration_seats
    )
    return OrToolsProjectLifecycleResultV1(
        projects,
        evidence.winning_team,
        sum(item.eligible for item in projects),
        sum(item.selected for item in projects),
        sum(item.declared for item in projects),
        sum(item.awarded for item in projects),
        evidence.combined_awarded_value,
        evidence.baloot_seats,
        evidence.baloot_teams,
        evidence.baloot_points,
        fragment.fragment_hash,
        _hash({
            "fragmentEvidenceHash": evidence.evidence_hash,
            "declarationActions": [action.to_dict() for action in declaration_actions],
        }) if declaration_actions else evidence.evidence_hash,
        solver.status_name(solver.response_proto.status).lower(),
        int(solver.num_conflicts),
        int(solver.num_branches),
        declaration_actions,
    )


def _and_literal(model: Any, literals: Sequence[Any], name: str) -> Any:
    if len(literals) == 1:
        return literals[0]
    target = model.new_bool_var(name)
    if not literals:
        model.add(target == 1)
        return target
    model.add_bool_and(list(literals)).only_enforce_if(target)
    model.add_bool_or([1 - literal if type(literal) in (int, bool) and literal in (0, 1)
                       else literal.Not() for literal in literals]).only_enforce_if(
        target.Not()
    )
    return target


def _or_literal(model: Any, literals: Sequence[Any], name: str) -> Any:
    if len(literals) == 1:
        return literals[0]
    target = model.new_bool_var(name)
    if literals:
        model.add_max_equality(target, list(literals))
    else:
        model.add(target == 0)
    return target


def _project_id(project: Any) -> str:
    return "PROJECT-" + _hash(
        {
            "owner": project.owner.name.lower(),
            "kind": project.kind,
            "cards": [card.id for card in project.cards],
            "value": int(project.value),
            "highRank": int(project.high_rank),
            "categoryPriority": int(project.category_priority),
        }
    )[:16].upper()


def constrain_exclusive_project_awards_v1(model, awarded_totals, winning_teams, has_declaration):
    """Shared aggregate constraints implied by the per-project award model."""
    for total, winner in zip(awarded_totals, winning_teams, strict=True):
        model.add(total == 0).only_enforce_if(winner.Not())
    model.add(sum(winning_teams) == has_declaration)


def attach_project_lifecycle_fragment_v1(
    model: Any,
    *,
    owner_of_card: Sequence[Any],
    fixed_card_owners: Sequence[Seat | str | int | None] | None = None,
    contract: Contract | str,
    dealer: Seat | str | int,
    trump: Suit | str | int | None,
    domain: ProjectDomainIRV1,
    second_trick_leader: Any = None,
    name: str = "projects",
    compact_fixed_states: bool = False,
    tighten_value_bounds: bool = False,
    include_declaration_actions: bool = False,
    eligibility_only: bool = False,
    declaration_only: bool = False,
    cancel_check: Callable[[], bool] | None = None,
) -> OrToolsProjectLifecycleFragmentV1 | OrToolsProjectEligibilityFragmentV1:
    def check_cancelled() -> None:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("project construction cancelled")

    check_cancelled()
    _validate_domain(domain)
    if type(compact_fixed_states) is not bool:
        raise TypeError("compact_fixed_states must be bool")
    if type(tighten_value_bounds) is not bool:
        raise TypeError("tighten_value_bounds must be bool")
    if type(include_declaration_actions) is not bool:
        raise TypeError("include_declaration_actions must be bool")
    if type(eligibility_only) is not bool:
        raise TypeError("eligibility_only must be bool")
    if type(declaration_only) is not bool:
        raise TypeError("declaration_only must be bool")
    if declaration_only and (eligibility_only or include_declaration_actions):
        raise ValueError("declaration-only construction cannot combine projection modes")
    if eligibility_only and include_declaration_actions:
        raise ValueError("eligibility-only construction cannot compile declaration actions")
    if (domain.tie_break_order == "second_trick_reveal" and not (eligibility_only or declaration_only)
            and second_trick_leader is None):
        raise ValueError("project fragment requires the second-trick leader")
    owners = tuple(owner_of_card)
    if len(owners) != 32 or any(not variable.Name() for variable in owners):
        raise ValueError("project fragment requires 32 named card-owner variables")
    fixed_owners = (
        tuple(Seat.parse(owner) if owner is not None else None for owner in fixed_card_owners)
        if fixed_card_owners is not None
        else None
    )
    if fixed_owners is not None and len(fixed_owners) != 32:
        raise ValueError("fixed project ownership requires exactly 32 card owners")
    contract = Contract.parse(contract)
    dealer = Seat.parse(dealer)
    parsed_trump = Suit.parse(trump) if trump is not None else None
    if contract.effective == Contract.HOKUM and parsed_trump is None:
        raise ValueError("Hokum project fragment requires trump")
    if contract.effective == Contract.SUN and parsed_trump is not None:
        raise ValueError("Sun project fragment cannot define trump")
    for card_id, owner in enumerate(owners):
        check_cancelled()
        model.add(owner >= 0)
        model.add(owner <= 3)
        if fixed_owners is not None and fixed_owners[card_id] is not None:
            model.add(owner == int(fixed_owners[card_id]))

    owner_is = tuple(
        tuple(
            (
                model.new_constant(int(fixed_owners[card.id] == seat))
                if fixed_owners is not None and fixed_owners[card.id] is not None
                else _owner_literal(
                    model,
                    owners[card.id],
                    seat,
                    f"{name}_c{card.id}_s{int(seat)}",
                )
            )
            for seat in Seat
        )
        for card in FULL_DECK
    )
    patterns: list[dict[str, Any]] = []
    by_seat: dict[Seat, list[int]] = {seat: [] for seat in Seat}
    active_by_seat: dict[Seat, list[int]] = {seat: [] for seat in Seat}
    if domain.enabled:
        for seat in Seat:
            for project in domain.candidates(seat, FULL_DECK, contract):
                check_cancelled()
                index = len(patterns)
                possible = fixed_owners is None or all(
                    fixed_owners[card.id] in {None, seat} for card in project.cards
                )
                patterns.append(
                    {
                        "project": project,
                        "projectId": _project_id(project),
                        "eligible": (
                            _and_literal(
                                model,
                                [
                                    owner_is[card.id][int(seat)]
                                    for card in project.cards
                                ],
                                f"{name}_p{index}_eligible",
                            )
                            if possible
                            else model.new_constant(0)
                        ),
                    }
                )
                by_seat[seat].append(index)
                if possible:
                    active_by_seat[seat].append(index)

    if eligibility_only:
        check_cancelled()
        # A structural witness needs ownership/eligibility, not a declaration policy.
        projects = tuple(ProjectPatternVariablesV1(
            row["projectId"], row["project"].owner, row["project"].kind,
            tuple(row["project"].cards), int(row["project"].value),
            int(row["project"].high_rank), int(row["project"].category_priority),
            row["eligible"], None, None, None, None, None, None,
        ) for row in patterns)
        digest = _hash({
            "schemaVersion": 1,
            "stage": "projects",
            "loweringVersion": "project_eligibility_variable_ownership_v1",
            "projectDomainIrHash": domain.sha256,
            "contract": contract.value,
            "dealer": dealer.name.lower(),
            "trump": parsed_trump.name.lower() if parsed_trump is not None else None,
            "ownerVariables": [variable.Name() for variable in owners],
            "projects": [{"projectId": p.project_id, "owner": p.owner.name.lower(),
                "kind": p.kind, "cards": [c.id for c in p.cards], "value": p.value,
                "highRank": p.high_rank, "categoryPriority": p.category_priority}
                for p in projects],
        })
        return OrToolsProjectEligibilityFragmentV1(projects, owners, digest)

    inactive = set(range(len(patterns))) - {
        index for indexes in active_by_seat.values() for index in indexes
    }
    # Preserve every catalog row, but share the proven-zero lifecycle state
    # for patterns excluded by the fixed ownership constraints above.
    impossible = model.new_int_var(0, 0, f"{name}_ineligible") if compact_fixed_states and inactive else None
    selected, declarable, declared, declaration_order, comparison_winner, awarded = (
        [impossible if compact_fixed_states and i in inactive else model.new_int_var(0, upper, f"{name}_p{i}_{suffix}")
         for i in range(len(patterns))]
        for suffix, upper in (("selected", 1), ("declarable", 1), ("declared", 1),
                              ("declaration_order", 4), ("winner", 1), ("awarded", 1))
    )
    groups_manifest: dict[str, list[dict[str, Any]]] = {}
    seat_value_bounds: dict[Seat, int] = {}

    for seat in Seat:
        check_cancelled()
        indexes = active_by_seat[seat]
        groups: list[tuple[int, ...]] = [()]
        for size in range(1, min(domain.maximum_per_player, len(indexes)) + 1):
            for group in combinations(indexes, size):
                check_cancelled()
                projects = [patterns[index]["project"] for index in group]
                if (
                    domain.group_is_valid(projects)
                    and len({card for project in projects for card in project.cards})
                    <= 8
                ):
                    groups.append(group)
        if tighten_value_bounds:
            # Exactly one group can be selected; this is a static domain bound,
            # not a prediction of eligibility, declaration, or award.
            seat_value_bounds[seat] = max(
                sum(int(patterns[index]["project"].value) for index in group)
                for group in groups
            )
        preference = sorted(
            range(len(groups)),
            key=lambda ordinal: (
                domain.group_score([patterns[index]["project"] for index in groups[ordinal]]),
                -ordinal,
            ),
        )
        rank = {ordinal: position + 1 for position, ordinal in enumerate(preference)}
        feasible: list[Any] = []
        chosen: list[Any] = []
        ranked: list[Any] = []
        for ordinal, group in enumerate(groups):
            check_cancelled()
            feasible_literal = _and_literal(
                model,
                [patterns[index]["eligible"] for index in group],
                f"{name}_s{int(seat)}_g{ordinal}_feasible",
            )
            rank_value = model.new_int_var(
                0, len(groups), f"{name}_s{int(seat)}_g{ordinal}_rank"
            )
            model.add(rank_value == rank[ordinal]).only_enforce_if(feasible_literal)
            model.add(rank_value == 0).only_enforce_if(feasible_literal.Not())
            feasible.append(feasible_literal)
            ranked.append(rank_value)
        chosen_rank = model.new_int_var(
            1, len(groups), f"{name}_s{int(seat)}_chosen_rank"
        )
        model.add_max_equality(chosen_rank, ranked)
        for ordinal in range(len(groups)):
            check_cancelled()
            literal = model.new_bool_var(f"{name}_s{int(seat)}_g{ordinal}_chosen")
            model.add(chosen_rank == rank[ordinal]).only_enforce_if(literal)
            model.add(chosen_rank != rank[ordinal]).only_enforce_if(literal.Not())
            model.add(literal <= feasible[ordinal])
            chosen.append(literal)
        model.add_exactly_one(chosen)
        groups_manifest[seat.name.lower()] = [
            {
                "projects": [patterns[index]["projectId"] for index in group],
                "preferenceRank": rank[ordinal],
            }
            for ordinal, group in enumerate(groups)
        ]
        selections = {index: [] for index in indexes}
        announcements = {index: [] for index in indexes}
        for group, literal in zip(groups, chosen):
            for index in group:
                selections[index].append(literal)
            if group:
                announcer = max(
                    group,
                    key=lambda item: (
                        patterns[item]["project"].value,
                        patterns[item]["project"].category_priority,
                        patterns[item]["project"].high_rank,
                    ),
                )
                announcements[announcer].append(literal)
        for index in indexes:
            model.add_max_equality(selected[index], selections[index])
            announcing = announcements[index]
            if announcing:
                model.add_max_equality(declarable[index], announcing)
            else:
                model.add(declarable[index] == 0)

        if not compact_fixed_states:
            for index in set(by_seat[seat]) - set(indexes):
                model.add(selected[index] == 0)
                model.add(declarable[index] == 0)
                model.add(declared[index] == 0)
                model.add(declaration_order[index] == 0)
                model.add(comparison_winner[index] == 0)
                model.add(awarded[index] == 0)

    seat_order = domain.declaration_seats(dealer)
    ordered_indexes = [
        index for seat in seat_order for index in active_by_seat[seat]
    ]

    rank_keys = {
        index: domain.rank_key(patterns[index]["project"], trump_suit=parsed_trump)
        for index in ordered_indexes
    }
    legal_declarations = {} if include_declaration_actions else None

    for position, index in enumerate(ordered_indexes):
        check_cancelled()
        prior = ordered_indexes[:position]
        prior_greater = [
            declared[other]
            for other in prior
            if rank_keys[other] > rank_keys[index]
        ]
        model.add(declared[index] <= declarable[index])
        if prior_greater:
            model.add(sum(prior_greater) == 0).only_enforce_if(declared[index])
        if domain.declaration_mode == "mandatory":
            model.add(declared[index] >= declarable[index] - sum(prior_greater))
        if legal_declarations is not None:
            # Availability is independent of an optional player's actual choice.
            # Mandatory declarations already equal this conjunction by the
            # constraints above, so reuse their literal rather than duplicate it.
            legal_declarations[index] = (declared[index] if domain.declaration_mode == "mandatory"
                else _and_literal(model, [declarable[index], *(value.Not() for value in prior_greater)],
                    f"{name}_p{index}_declaration_legal"))
        model.add(declaration_order[index] == 0).only_enforce_if(
            declared[index].Not()
        )
        model.add(
            declaration_order[index]
            == 1 + sum(declared[other] for other in prior)
        ).only_enforce_if(declared[index])

    if declaration_only:
        check_cancelled()
        projects = tuple(ProjectPatternVariablesV1(
            row["projectId"], row["project"].owner, row["project"].kind,
            tuple(row["project"].cards), int(row["project"].value),
            int(row["project"].high_rank), int(row["project"].category_priority),
            row["eligible"], selected[index], declarable[index], declared[index],
            declaration_order[index], None, None,
        ) for index, row in enumerate(patterns))
        digest = _hash({"stage": "project_declarations", "projectDomainIrHash": domain.sha256,
            "contract": contract.value, "dealer": dealer.name.lower(),
            "trump": parsed_trump.name.lower() if parsed_trump is not None else None,
            "cpModel": str(model.proto)})
        return OrToolsProjectDeclarationFragmentV1(projects, owners, digest)

    reveal_precedes = {}
    if domain.tie_break_order == "second_trick_reveal":
        if isinstance(second_trick_leader, (int, Seat)):
            second_trick_leader = Seat.parse(second_trick_leader)
        else:
            model.add(second_trick_leader >= 0)
            model.add(second_trick_leader <= 3)
        for before in Seat:
            for after in Seat:
                if before == after:
                    continue
                if isinstance(second_trick_leader, Seat):
                    reveal_precedes[before, after] = int(
                        (before.team != second_trick_leader.team,
                         (int(before) - int(second_trick_leader)) % 4)
                        < (after.team != second_trick_leader.team,
                           (int(after) - int(second_trick_leader)) % 4))
                else:
                    precedes = model.new_bool_var(
                        f"{name}_reveal_s{int(before)}_before_s{int(after)}")
                    model.add_allowed_assignments([second_trick_leader, precedes], [
                        (int(leader), int((before.team != leader.team,
                                          (int(before) - int(leader)) % 4)
                                         < (after.team != leader.team,
                                            (int(after) - int(leader)) % 4)))
                        for leader in Seat])
                    reveal_precedes[before, after] = precedes

    for position, index in enumerate(ordered_indexes):
        check_cancelled()
        disqualifying = [
            declared[other]
            for other_position, other in enumerate(ordered_indexes)
            if other != index
            and (
                rank_keys[other] > rank_keys[index]
                or (
                    rank_keys[other] == rank_keys[index]
                    and other_position < position
                    and (domain.tie_break_order == "declaration"
                         or patterns[other]["project"].owner == patterns[index]["project"].owner)
                )
            )
        ]
        if domain.tie_break_order == "second_trick_reveal":
            owner = patterns[index]["project"].owner
            disqualifying.extend(
                _and_literal(model, [declared[other], reveal_precedes[
                    patterns[other]["project"].owner, owner]],
                    f"{name}_p{other}_precedes_p{index}")
                for other in ordered_indexes
                if rank_keys[other] == rank_keys[index]
                and patterns[other]["project"].owner != owner)
        model.add(comparison_winner[index] <= declared[index])
        if disqualifying:
            model.add(sum(disqualifying) == 0).only_enforce_if(
                comparison_winner[index]
            )
        model.add(comparison_winner[index] >= declared[index] - sum(disqualifying))
    has_declaration = _or_literal(model, declared, f"{name}_has_declaration")
    model.add(sum(comparison_winner) == has_declaration)

    winning_team = tuple(
        model.new_bool_var(f"{name}_winning_team_{int(team)}") for team in Team
    )
    for team in Team:
        model.add(
            winning_team[int(team)]
            == sum(
                comparison_winner[index]
                for index in range(len(patterns))
                if patterns[index]["project"].owner.team == team
            )
        )
    for index, row in enumerate(patterns):
        check_cancelled()
        if compact_fixed_states and index in inactive:
            continue
        team_literal = winning_team[int(row["project"].owner.team)]
        model.add(awarded[index] <= selected[index])
        model.add(awarded[index] <= team_literal)
        model.add(awarded[index] >= selected[index] + team_literal - 1)

    maximum_value = sum(int(row["project"].value) for row in patterns)
    combined_awarded_value = tuple(
        model.new_int_var(0, sum(seat_value_bounds[seat] for seat in Seat if seat.team == team)
            if tighten_value_bounds else maximum_value, f"{name}_awarded_value_{int(team)}")
        for team in Team
    )
    for team in Team:
        model.add(
            combined_awarded_value[int(team)]
            == sum(
                int(row["project"].value) * awarded[index]
                for index, row in enumerate(patterns)
                if row["project"].owner.team == team
            )
        )
    if tighten_value_bounds:
        # Already implied by each award; propagate conflicting totals directly.
        constrain_exclusive_project_awards_v1(model, combined_awarded_value, winning_team, has_declaration)

    baloot_seat: list[Any] = []
    for seat in Seat:
        check_cancelled()
        target = model.new_bool_var(f"{name}_baloot_seat_{int(seat)}")
        baloot_seat.append(target)
        if not domain.baloot_enabled or contract != Contract.HOKUM or parsed_trump is None:
            model.add(target == 0)
            continue
        pair = {
            Card.of(parsed_trump, Rank.KING),
            Card.of(parsed_trump, Rank.QUEEN),
        }
        pair_owned = _and_literal(
            model,
            [owner_is[card.id][int(seat)] for card in pair],
            f"{name}_baloot_pair_{int(seat)}",
        )
        included: list[Any] = []
        blocked: list[Any] = []
        for index in active_by_seat[seat]:
            check_cancelled()
            project = patterns[index]["project"]
            project_cards = set(project.cards)
            if (
                domain.baloot_siri_or_fifty_inclusion == "automatic"
                and project.kind in {"siri", "fifty"}
                and pair.issubset(project_cards)
            ):
                included.append(awarded[index])
            if (
                project.kind == "hundred_sequence"
                and pair.issubset(project_cards)
                and domain.baloot_hundred_inclusion == "automatic"
            ):
                included.append(awarded[index])
            if (
                (
                    project.kind == "hundred_sequence"
                    and pair.issubset(project_cards)
                    and domain.baloot_hundred_inclusion == "blocked"
                )
                or (
                    project.kind == "hundred_four_kind"
                    and project.cards
                    and project.cards[0].rank in {Rank.KING, Rank.QUEEN}
                    and domain.baloot_hundred_inclusion == "blocked"
                )
                or (
                    len(pair & project_cards) == 1
                    and domain.baloot_single_card_policy == "blocked"
                )
            ):
                blocked.append(awarded[index])
        included_literal = _or_literal(
            model, included, f"{name}_baloot_included_{int(seat)}"
        )
        blocked_literal = _or_literal(
            model, blocked, f"{name}_baloot_blocked_{int(seat)}"
        )
        allowed = model.new_bool_var(f"{name}_baloot_allowed_{int(seat)}")
        model.add(allowed >= included_literal)
        model.add(allowed >= 1 - blocked_literal)
        model.add(allowed <= included_literal + 1 - blocked_literal)
        model.add(target <= pair_owned)
        model.add(target <= allowed)
        model.add(target >= pair_owned + allowed - 1)
    baloot_team = tuple(
        _or_literal(
            model,
            [baloot_seat[int(seat)] for seat in Seat if seat.team == team],
            f"{name}_baloot_team_{int(team)}",
        )
        for team in Team
    )

    project_variables = tuple(
        ProjectPatternVariablesV1(
            row["projectId"],
            row["project"].owner,
            row["project"].kind,
            tuple(row["project"].cards),
            int(row["project"].value),
            int(row["project"].high_rank),
            int(row["project"].category_priority),
            row["eligible"],
            selected[index],
            declarable[index],
            declared[index],
            declaration_order[index],
            comparison_winner[index],
            awarded[index],
        )
        for index, row in enumerate(patterns)
    )
    fragment_core = {
        "schemaVersion": 1,
        "backend": "ortools_cp_sat_v2",
        "stage": "projects",
        "loweringVersion": "project_lifecycle_variable_ownership_v1",
        "projectDomainIrHash": domain.sha256,
        "contract": contract.value,
        "dealer": dealer.name.lower(),
        "trump": parsed_trump.name.lower() if parsed_trump is not None else None,
        "ownerVariables": [variable.Name() for variable in owners],
        "projects": [
            {
                "projectId": project.project_id,
                "owner": project.owner.name.lower(),
                "kind": project.kind,
                "cards": [card.id for card in project.cards],
                "value": project.value,
                "highRank": project.high_rank,
                "categoryPriority": project.category_priority,
            }
            for project in project_variables
        ],
        "groups": groups_manifest,
        **({"seatValueUpperBounds": {seat.name.lower(): value for seat, value in seat_value_bounds.items()}}
           if tighten_value_bounds else {}),
    }
    declaration_actions = None
    if domain.tie_break_order == "second_trick_reveal":
        fragment_core["secondTrickLeader"] = (
            int(second_trick_leader) if isinstance(second_trick_leader, Seat)
            else {"variable": second_trick_leader.Name()})
    if legal_declarations is not None:
        declaration_actions = tuple({
            "pass": (model.new_constant(1) if domain.declaration_mode == "optional" else
                _and_literal(model, [declared[index].Not() for index in active_by_seat[seat]],
                    f"{name}_s{int(seat)}_pass_legal")),
            **{ProjectDeclarationActionV1(seat, "declare", patterns[index]["projectId"]).code: legal_declarations[index]
                for index in active_by_seat[seat]},
        } for seat in Seat)
        fragment_core["declarationActionVariables"] = True
    check_cancelled()
    return OrToolsProjectLifecycleFragmentV1(
        project_variables,
        owners,
        (winning_team[0], winning_team[1]),
        (combined_awarded_value[0], combined_awarded_value[1]),
        (baloot_seat[0], baloot_seat[1], baloot_seat[2], baloot_seat[3]),
        (baloot_team[0], baloot_team[1]),
        domain.baloot_value,
        _hash(fragment_core),
        declaration_actions,
        seat_order if domain.declaration_mode_authored else (),
    )


def _owner_literal(model: Any, owner: Any, seat: Seat, name: str) -> Any:
    literal = model.new_bool_var(name)
    model.add(owner == int(seat)).only_enforce_if(literal)
    model.add(owner != int(seat)).only_enforce_if(literal.Not())
    return literal


__all__ = [
    "OrToolsProjectLifecycleFragmentV1",
    "ProjectBuyerConstraintV1",
    "ProjectLifecycleFragmentEvidenceV1",
    "ProjectPatternEvidenceV1",
    "ProjectPatternVariablesV1",
    "attach_project_lifecycle_fragment_v1",
]
