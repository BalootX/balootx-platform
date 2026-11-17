"""Construct a fully specified provisional proof boundary for a solved play tail."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping, Sequence

from ortools.sat.python import cp_model

from baloot_platform.core.cards import FULL_DECK, Card
from baloot_platform.core.enums import Contract, GamePhase, Seat, Suit, Team
from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1

from .ortools_complete_state_graph import CompleteStateGraphSeedV1
from .ortools_information_set_proof import _solve_cp_sat_with_cancel_v1
from .ortools_escalation_stage import OrToolsEscalationStageV1
from .ortools_play_tail import OrToolsPlayTailResultV1, OrToolsPlayTailStageV1, PlayTailSolveStopped
from .ortools_play_state_predicates import (
    CpSatPlayPrefixActionV1,
    CpSatPlayStatePredicateContextV1,
)
from .ortools_project_fragment import (
    attach_project_lifecycle_fragment_v1,
    project_lifecycle_result_v1,
)
from .ortools_project_model import (
    OrToolsProjectLifecycleModelV1,
    OrToolsProjectLifecycleResultV1,
)
from .ortools_scenario_model import OrToolsScenarioModelV1
from .ortools_scoring_stage import (
    CpSatScoringInputsV1,
    attach_raw_conversion_v1,
    attach_scoring_fragment_v1,
    extract_scoring_evidence_v1,
    scoring_contract_id_v1,
    scoring_level_id_v1,
)
from .scenario_contracts import ResearchScenarioV1, canonical_sha256
from .scenario_deal_predicates_v2 import DealPredicateModelContextV2
from .scenario_predicate_auction_escalation_v2 import (
    CpSatEscalationPredicateContextV1,
)
from .scenario_predicate_manifest_v2 import (
    ScenarioPredicateCatalogLoweringManifestV2,
    shared_scenario_predicate_catalog_lowering_manifest_v2,
)
from .scenario_project_outcome_lowering_v2 import CpSatProjectPredicateContextV1
from .tail_first_stage_planner import TailFirstStagePlanV1


@dataclass(frozen=True, slots=True)
class TailProofBoundaryNoGoodV1:
    tail_evidence_hash: str
    boundary_combination: tuple[str, str, str, str | None]
    vector: tuple[int, ...]
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        tail_evidence_hash: str,
        boundary_combination: tuple[str, str, str, str | None],
        vector: Sequence[int],
    ) -> "TailProofBoundaryNoGoodV1":
        values = tuple(int(item) for item in vector)
        core = {
            "schemaVersion": 1,
            "tailEvidenceHash": tail_evidence_hash,
            "boundaryCombination": list(boundary_combination),
            "vector": list(values),
        }
        return cls(
            tail_evidence_hash,
            boundary_combination,
            values,
            canonical_sha256(core),
        )


@dataclass(frozen=True, slots=True)
class TailProofStateNoGoodV1:
    """Reject one exact proof state without rejecting another reachable history."""

    decision_state_hash: str
    boundary_combination: tuple[str, str, str, str | None]
    base_raw_ns: int
    base_tricks_ns: int
    awarded_project_points: tuple[int, int]
    baloot_teams: tuple[bool, bool]
    escalation_semantics: tuple[str, bool, str | None]
    proof_state_projection_hash: str
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        decision_state_hash: str,
        boundary_combination: tuple[str, str, str, str | None],
        base_raw_ns: int,
        base_tricks_ns: int,
        awarded_project_points: Sequence[int],
        baloot_teams: Sequence[bool],
        escalation_semantics: tuple[str, bool, str | None],
        proof_state_projection_hash: str,
    ) -> "TailProofStateNoGoodV1":
        project_points = tuple(int(item) for item in awarded_project_points)
        teams = tuple(bool(item) for item in baloot_teams)
        if len(project_points) != 2 or any(item < 0 for item in project_points):
            raise ValueError("proof-state project points must contain two values")
        if len(teams) != 2 or any(type(item) is not bool for item in baloot_teams):
            raise ValueError("proof-state Baloot teams must contain two booleans")
        if type(base_raw_ns) is not int or base_raw_ns < 0:
            raise ValueError("proof-state base raw must be a non-negative integer")
        if type(base_tricks_ns) is not int or base_tricks_ns < 0:
            raise ValueError("proof-state base tricks must be a non-negative integer")
        if (
            type(escalation_semantics) is not tuple
            or len(escalation_semantics) != 3
            or type(escalation_semantics[0]) is not str
            or type(escalation_semantics[1]) is not bool
            or escalation_semantics[2] not in {None, "north_south", "west_east"}
        ):
            raise ValueError("proof-state escalation semantics are invalid")
        core = {
            "schemaVersion": 1,
            "kind": "tail_proof_state_exact_no_good",
            "decisionStateHash": decision_state_hash,
            "boundaryCombination": list(boundary_combination),
            "baseRawNs": base_raw_ns,
            "baseTricksNs": base_tricks_ns,
            "awardedProjectPoints": list(project_points),
            "balootTeams": list(teams),
            "escalation": {
                "level": escalation_semantics[0],
                "locked": escalation_semantics[1],
                "lastTeam": escalation_semantics[2],
            },
            "proofStateProjectionHash": proof_state_projection_hash,
        }
        return cls(
            decision_state_hash,
            boundary_combination,
            base_raw_ns,
            base_tricks_ns,
            (project_points[0], project_points[1]),
            (teams[0], teams[1]),
            escalation_semantics,
            proof_state_projection_hash,
            canonical_sha256(core),
        )

    def cp_vector(self, escalation_semantics_code: int) -> tuple[int, ...]:
        return (
            self.base_raw_ns,
            self.base_tricks_ns,
            *self.awarded_project_points,
            *(int(item) for item in self.baloot_teams),
            escalation_semantics_code,
        )

    def verify(self) -> bool:
        return self == type(self).build(
            decision_state_hash=self.decision_state_hash,
            boundary_combination=self.boundary_combination,
            base_raw_ns=self.base_raw_ns,
            base_tricks_ns=self.base_tricks_ns,
            awarded_project_points=self.awarded_project_points,
            baloot_teams=self.baloot_teams,
            escalation_semantics=self.escalation_semantics,
            proof_state_projection_hash=self.proof_state_projection_hash,
        )


def _tail_decision_state_hash(tail: OrToolsPlayTailResultV1) -> str:
    first_leader = (
        tail.current_trick[0].actor if tail.current_trick else tail.next_actor
    )
    if first_leader is None:
        raise ValueError("tail decision state requires a live actor")
    return canonical_sha256(
        {
            "schemaVersion": 1,
            "remainingHands": [
                [card.id for card in hand] for hand in tail.remaining_hands
            ],
            "currentTrick": {
                "leader": first_leader.name.lower(),
                "plays": [item.to_dict() for item in tail.current_trick],
            },
            "nextActor": tail.next_actor.name.lower() if tail.next_actor else None,
            "legalActions": [
                item.to_dict()
                for item in sorted(
                    tail.legal_play_actions,
                    key=lambda action: action.card.id + 32 * int(action.ekka),
                )
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class OrToolsTailProofBoundaryResultV1:
    status: str
    seed: CompleteStateGraphSeedV1 | None
    no_good: TailProofBoundaryNoGoodV1 | None
    model_hash: str | None
    evidence_hash: str | None
    solver_status: str
    branches: int
    conflicts: int
    outcome_rejections: int
    reason: str | None = None
    proof_state_no_good: TailProofStateNoGoodV1 | None = None
    variable_count: int = 0
    constraint_count: int = 0
    evidence_bindings: tuple[Mapping[str, Any], ...] = ()
    predicate_evidence: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    project_result: OrToolsProjectLifecycleResultV1 | None = None
    model_build_seconds: float = 0.0
    solver_seconds: float = 0.0
    solve_attempts: int = 0


def _hard_stage_predicates(
    scenario: ResearchScenarioV1,
    manifest: ScenarioPredicateCatalogLoweringManifestV2,
    stage: str,
) -> tuple[Any, ...]:
    return tuple(
        predicate
        for predicate in scenario.predicates()
        if predicate.hardness == "hard"
        and manifest.definition(predicate.kind).stage == stage
    )


def _state_context(
    model: cp_model.CpModel,
    *,
    tail: OrToolsPlayTailResultV1,
    raw_points: Mapping[Team, Any],
    project_points: Mapping[Team, Any],
    game_points: Mapping[Team, Any],
    match_score: tuple[int, int],
    trump: Suit | None,
) -> CpSatPlayStatePredicateContextV1:
    absent = CpSatPlayPrefixActionV1(
        model.new_constant(0),
        model.new_constant(0),
        model.new_constant(0),
        model.new_constant(0),
    )
    legal_cards = {card.id for card in tail.legal_actions}
    legal_actions = {
        (action.card.id, action.ekka) for action in tail.legal_play_actions
    }
    completed = 8 - len(tail.initial_hands[0])
    current_size = len(tail.current_trick)
    absent_slots = tuple(absent for _ in range(4 - current_size))
    current_slots = (
        tuple(
            CpSatPlayPrefixActionV1(
                model.new_constant(play.card.id),
                model.new_constant(int(play.actor)),
                model.new_constant(int(play.ekka)),
                model.new_constant(1),
            )
            for play in tail.current_trick
        )
        + absent_slots
    )
    return CpSatPlayStatePredicateContextV1(
        model=model,
        context_id="tail-proof-boundary-state",
        play_slots=tuple(absent for _ in range(32)),
        prefix_length=model.new_constant(completed * 4 + current_size),
        trick_index=model.new_constant(completed),
        next_actor=model.new_constant(int(tail.next_actor)),
        remaining_hand_sizes={
            seat: model.new_constant(len(tail.remaining_hands[int(seat)]))
            for seat in Seat
        },
        current_trick_slots=current_slots,
        current_trick_size=model.new_constant(current_size),
        current_trick_active=model.new_constant(int(current_size > 0)),
        current_trick_leader=model.new_constant(
            int(tail.current_trick[0].actor) if current_size else int(tail.next_actor)
        ),
        current_trick_suit_led=model.new_constant(
            int(tail.current_trick[0].card.suit) if current_size else 0
        ),
        trump_suit=model.new_constant(int(trump) if trump is not None else 4),
        completed_trick_literals={
            trick: model.new_constant(int(trick < completed)) for trick in range(8)
        },
        completed_trick_winners={trick: model.new_constant(0) for trick in range(8)},
        legal_card_literals={
            card: model.new_constant(int(card in legal_cards)) for card in range(32)
        },
        legal_action_literals={
            (card, ekka): model.new_constant(int((card, ekka) in legal_actions))
            for card in range(32)
            for ekka in (False, True)
        },
        phase=model.new_constant(tuple(GamePhase).index(GamePhase.PLAYING)),
        match_scores={
            team: model.new_constant(match_score[int(team)]) for team in Team
        },
        raw_points=raw_points,
        project_points=project_points,
        game_points=game_points,
    )


def _attach_native_boundary_scoring(
    model: cp_model.CpModel,
    *,
    scoring: ScoringDomainIRV1,
    rules: RuleBook,
    contract: Contract,
    buyer: Seat,
    tail: OrToolsPlayTailResultV1,
    base_raw_ns: Any,
    base_raw_we: Any,
    base_tricks_ns: Any,
    completed_tricks: int,
    escalation_context: CpSatEscalationPredicateContextV1,
    projects: Any,
    starting_match_score: tuple[int, int],
    source_hash: str,
) -> Any:
    scoring_escalation = model.new_int_var(0, 4, "tail_boundary_scoring_escalation")
    model.add(
        scoring_escalation
        == (escalation_context.native["level"] if escalation_context.stage.native_cp_sat else sum(
            scoring_level_id_v1(candidate.level) * selected
            for candidate, selected in zip(
                escalation_context.candidates,
                escalation_context.selected,
                strict=True,
            )
        ))
    )
    scoring_last_team = model.new_int_var(0, 2, "tail_boundary_scoring_last_team")
    model.add(
        scoring_last_team
        == (escalation_context.native["lastTeam"] if escalation_context.stage.native_cp_sat else sum(
            (int(candidate.actions[-1].actor.team) if candidate.actions else 2)
            * selected
            for candidate, selected in zip(
                escalation_context.candidates,
                escalation_context.selected,
                strict=True,
            )
        ))
    )
    tail_raw = [0, 0]
    for winner, points in zip(
        tail.trick_winners, tail.trick_raw_points, strict=True
    ):
        tail_raw[int(winner.team)] += int(points)
    tail_raw[int(tail.trick_winners[-1].team)] += scoring.last_trick_bonus_raw
    full_raw = tuple(
        model.new_int_var(
            0,
            scoring.raw_card_totals[contract.effective.value],
            f"tail_boundary_full_raw_{int(team)}",
        )
        for team in Team
    )
    model.add(full_raw[int(Team.NORTH_SOUTH)] == base_raw_ns + tail_raw[0])
    model.add(full_raw[int(Team.WEST_EAST)] == base_raw_we + tail_raw[1])
    prefix_winner_teams = tuple(
        model.new_int_var(0, 1, f"tail_boundary_prefix_winner_{index}")
        for index in range(completed_tricks)
    )
    if prefix_winner_teams:
        prefix_winner_ns = tuple(
            model.new_bool_var(f"tail_boundary_prefix_winner_ns_{index}")
            for index in range(completed_tricks)
        )
        for winner, is_ns in zip(prefix_winner_teams, prefix_winner_ns, strict=True):
            model.add(winner == int(Team.NORTH_SOUTH)).only_enforce_if(is_ns)
            model.add(winner == int(Team.WEST_EAST)).only_enforce_if(is_ns.Not())
        model.add(base_tricks_ns == sum(prefix_winner_ns))
    trick_winner_teams = (
        *prefix_winner_teams,
        *(model.new_constant(int(winner.team)) for winner in tail.trick_winners),
    )
    if len(trick_winner_teams) != 8:
        raise RuntimeError("tail boundary does not cover all eight trick winners")
    project_points = (
        projects.combined_awarded_value
        if projects is not None
        else (model.new_constant(0), model.new_constant(0))
    )
    if projects is not None:
        baloot_points = tuple(
            model.new_int_var(
                0,
                rules.project_domain.baloot_value,
                f"tail_boundary_baloot_points_{int(team)}",
            )
            for team in Team
        )
        for team in Team:
            model.add(
                baloot_points[int(team)]
                == projects.baloot_team[int(team)] * rules.project_domain.baloot_value
            )
    else:
        baloot_points = (model.new_constant(0), model.new_constant(0))
    return attach_scoring_fragment_v1(
        model,
        scoring,
        CpSatScoringInputsV1(
            contract=model.new_constant(scoring_contract_id_v1(contract)),
            buyer_seat=model.new_constant(int(buyer)),
            escalation=scoring_escalation,
            last_escalating_team=scoring_last_team,
            raw_points=(full_raw[0], full_raw[1]),
            project_points=(project_points[0], project_points[1]),
            baloot_points=(baloot_points[0], baloot_points[1]),
            trick_winner_teams=trick_winner_teams,
            starting_match_score=tuple(
                model.new_constant(value) for value in starting_match_score
            ),
        ),
        source_hash=source_hash,
        native_conversion=escalation_context.stage.native_cp_sat,
    )


class OrToolsTailProofBoundaryV1:
    """Enumerate exact CP boundary values; reachability remains the lift's job."""

    def __init__(
        self,
        *,
        rules: RuleBook,
        starting_match_score=(0, 0),
        native_cp_sat: bool = True,
    ) -> None:
        if not isinstance(rules, RuleBook):
            raise TypeError("tail proof boundary requires RuleBook")
        if (
            type(starting_match_score) is not tuple
            or len(starting_match_score) != 2
            or any(type(item) is not int or item < 0 for item in starting_match_score)
        ):
            raise ValueError(
                "tail proof boundary match score must contain two integers"
            )
        self.rules = rules
        self.starting_match_score = starting_match_score
        self.native_cp_sat = bool(native_cp_sat)

    def solve(
        self,
        scenario: ResearchScenarioV1,
        plan: TailFirstStagePlanV1,
        tail: OrToolsPlayTailResultV1,
        *,
        fixed_card_owners: Sequence[Seat | str | int] | None = None,
        excluded: Sequence[TailProofBoundaryNoGoodV1] = (),
        excluded_proof_states: Sequence[TailProofStateNoGoodV1] = (),
        required_history: TailProofBoundaryNoGoodV1 | None = None,
        required_proof_state: TailProofStateNoGoodV1 | None = None,
        required_base_raw_ns: int | None = None,
        required_base_tricks_ns: int | None = None,
        preferred_base_raw_ns: int | None = None,
        preferred_base_tricks_ns: int | None = None,
        boundary_combination: tuple[str, str, str, str | None] | None = None,
        maximum_seconds: float = 10.0,
        maximum_conflicts: int = 250_000,
        cancel_check: Callable[[], bool] | None = None,
    ) -> OrToolsTailProofBoundaryResultV1:
        if not isinstance(scenario, ResearchScenarioV1):
            raise TypeError("tail proof boundary requires ResearchScenarioV1")
        if not isinstance(plan, TailFirstStagePlanV1) or plan.planner != "tail_first":
            raise ValueError("tail proof boundary requires a tail-first plan")
        if not isinstance(tail, OrToolsPlayTailResultV1):
            raise TypeError("tail proof boundary requires a solved play tail")
        if type(maximum_seconds) not in {int, float} or maximum_seconds <= 0:
            raise ValueError("tail proof boundary maximum_seconds must be positive")
        if type(maximum_conflicts) is not int or maximum_conflicts < 1:
            raise ValueError("tail proof boundary maximum_conflicts must be positive")
        if tail.next_actor is None or not tail.legal_play_actions:
            raise ValueError("tail proof boundary requires a live root decision")
        fixed_owners = (
            tuple(Seat.parse(owner) for owner in fixed_card_owners)
            if fixed_card_owners is not None
            else None
        )
        if fixed_owners is not None and len(fixed_owners) != 32:
            raise ValueError("tail proof boundary fixed ownership requires 32 cards")
        if any(not isinstance(item, TailProofBoundaryNoGoodV1) for item in excluded):
            raise TypeError("tail boundary exclusions must use exact boundary no-goods")
        if any(
            not isinstance(item, TailProofStateNoGoodV1) or not item.verify()
            for item in excluded_proof_states
        ):
            raise ValueError(
                "tail proof exclusions must use verified proof-state no-goods"
            )
        if required_history is not None and not isinstance(
            required_history, TailProofBoundaryNoGoodV1
        ):
            raise TypeError("required tail history must use an exact boundary no-good")
        if required_proof_state is not None and (
            not isinstance(required_proof_state, TailProofStateNoGoodV1)
            or not required_proof_state.verify()
        ):
            raise ValueError("required proof state must use a verified exact projection")
        for value, name in (
            (required_base_raw_ns, "required base raw"),
            (required_base_tricks_ns, "required base tricks"),
            (preferred_base_raw_ns, "preferred base raw"),
            (preferred_base_tricks_ns, "preferred base tricks"),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"tail proof boundary {name} must be non-negative")
        deadline = time.monotonic() + float(maximum_seconds)

        def stopped() -> bool:
            return time.monotonic() >= deadline or (cancel_check is not None and cancel_check())

        decision_state_hash = _tail_decision_state_hash(tail)
        manifest = shared_scenario_predicate_catalog_lowering_manifest_v2()
        project_predicates = _hard_stage_predicates(scenario, manifest, "projects")
        deal_predicates = _hard_stage_predicates(scenario, manifest, "deal")
        escalation_predicates = _hard_stage_predicates(scenario, manifest, "escalation")
        play_predicates = _hard_stage_predicates(scenario, manifest, "playPrefix")
        state_predicates = (
            _hard_stage_predicates(scenario, manifest, "state")
            if self.native_cp_sat
            else ()
        )
        outcome_predicates = (
            _hard_stage_predicates(scenario, manifest, "outcome")
            if self.native_cp_sat
            else ()
        )
        reveal_ordered_projects = (
            self.rules.project_domain.tie_break_order == "second_trick_reveal"
        )
        state_needs_live_projects = any(
            predicate.kind == "state.current_project_points"
            for predicate in state_predicates
        )
        outcome_needs_live_projects = any(
            predicate.kind not in {"outcome.raw_points", "outcome.kabout"}
            for predicate in outcome_predicates
        )
        scoring = ScoringDomainIRV1.from_rules(self.rules.data)
        tail_card_points = sum(int(item) for item in tail.trick_raw_points)
        outcome_rejections = 0
        saw_unknown = False
        model_build_seconds = 0.0
        solver_seconds = 0.0
        solve_attempts = 0

        for combination in plan.boundary_combinations:
            if stopped():
                saw_unknown = True
                break
            if boundary_combination is not None and combination != boundary_combination:
                continue
            if (
                required_proof_state is not None
                and required_proof_state.boundary_combination != combination
            ):
                continue
            dealer_raw, contract_raw, buyer_raw, trump_raw = combination
            contract = Contract.parse(contract_raw)
            buyer = Seat.parse(buyer_raw)
            dealer = Seat.parse(dealer_raw)
            trump = Suit.parse(trump_raw) if trump_raw is not None else None
            proof_state_exclusions = tuple(
                item
                for item in excluded_proof_states
                if item.decision_state_hash == decision_state_hash
                and item.boundary_combination == combination
            )
            if (
                required_proof_state is not None
                and required_proof_state.decision_state_hash != decision_state_hash
            ):
                raise ValueError("required proof state belongs to another decision")
            needs_live_projects = bool(
                reveal_ordered_projects
                or project_predicates
                or state_needs_live_projects
                or outcome_needs_live_projects
                or proof_state_exclusions
                or required_proof_state is not None
            )
            prefix_raw_total = (
                scoring.raw_card_totals[contract.effective.value]
                - tail_card_points
                - scoring.last_trick_bonus_raw
            )
            if prefix_raw_total < 0:
                continue
            model_started = time.perf_counter()
            model_owner = OrToolsScenarioModelV1(rules=self.rules, dealer=dealer)
            model = model_owner.model
            predicate_literals: dict[str, Any] = {}
            predicate_lowerings: dict[str, Any] = {}
            for seat in Seat:
                for card in tail.initial_hands[int(seat)]:
                    model.add(model_owner.owner_of_card[card.id] == int(seat))
            if deal_predicates:
                from .ortools_research_executor import _enforce_hard_stage_projection

                remaining_owner_by_card = {
                    card.id: int(seat)
                    for seat in Seat
                    for card in tail.remaining_hands[int(seat)]
                }
                deal_context = DealPredicateModelContextV2(
                    model_owner,
                    remaining_owner_of_card=tuple(
                        model.new_constant(remaining_owner_by_card.get(card_id, 4))
                        for card_id in range(32)
                    ),
                )
                deal_lowerings = {
                    predicate.predicate_id: manifest.lower_predicate(
                        predicate, deal_context
                    )
                    for predicate in deal_predicates
                }
                predicate_lowerings.update(deal_lowerings)
                deal_literals = {
                    predicate_id: lowering.literal
                    for predicate_id, lowering in deal_lowerings.items()
                }
                predicate_literals.update(deal_literals)
                _enforce_hard_stage_projection(
                    model,
                    scenario,
                    stage="deal",
                    predicate_literals=deal_literals,
                )
            from .ortools_research_executor import _hard_expression_requires_predicate

            predicate_fixed_owners = {
                int(predicate.arguments["card"]): Seat.parse(
                    predicate.arguments["seat"]
                )
                for predicate in deal_predicates
                if predicate.kind == "deal.card_owner"
                and predicate.arguments.get("handStage")
                in {"initial", "post_purchase"}
                and _hard_expression_requires_predicate(
                    scenario,
                    predicate.predicate_id,
                    maximum_conflicts=maximum_conflicts,
                )
            }
            known_owners = dict(predicate_fixed_owners)
            known_owners.update(
                (card.id, seat)
                for seat in Seat
                for card in tail.initial_hands[int(seat)]
            )
            project_fixed_owners = fixed_owners or tuple(
                known_owners.get(card_id) for card_id in range(32)
            )
            second_trick_leader = (
                model.new_int_var(0, 3, "tail_boundary_second_trick_leader")
                if reveal_ordered_projects
                else None
            )
            projects = (
                attach_project_lifecycle_fragment_v1(
                    model,
                    owner_of_card=model_owner.owner_of_card,
                    fixed_card_owners=project_fixed_owners,
                    contract=contract,
                    dealer=dealer,
                    trump=trump,
                    domain=self.rules.project_domain,
                    name="tail_boundary_projects",
                    second_trick_leader=second_trick_leader,
                    compact_fixed_states=True,
                    tighten_value_bounds=True,
                )
                if needs_live_projects
                else None
            )
            if project_predicates:
                assert projects is not None
                from .ortools_research_executor import _project_exists_witness_keys

                project_context = CpSatProjectPredicateContextV1(
                    model, "tail_boundary_projects", projects
                )
                project_lowerings = {
                    predicate.predicate_id: manifest.lower_predicate(
                        predicate, project_context
                    )
                    for predicate in project_predicates
                }
                predicate_lowerings.update(project_lowerings)
                project_literals = {
                    predicate_id: lowering.literal
                    for predicate_id, lowering in project_lowerings.items()
                }
                for literal in project_literals.values():
                    model.add_hint(literal, 1)
                for predicate in project_predicates:
                    if predicate.kind != "project.exists":
                        continue
                    arguments = dict(predicate.arguments)
                    if _hard_expression_requires_predicate(
                        scenario,
                        predicate.predicate_id,
                        maximum_conflicts=maximum_conflicts,
                    ):
                        witnesses = _project_exists_witness_keys(
                            predicate, self.rules, contract, dealer
                        )
                        selected_witnesses = tuple(
                            model.new_bool_var(
                                f"tail_boundary_project_witness_{index}"
                            )
                            for index in range(len(witnesses))
                        )
                        if not selected_witnesses:
                            model.add(0 == 1)
                        else:
                            model.add_bool_or(selected_witnesses)
                            for selected, witness in zip(
                                selected_witnesses, witnesses, strict=True
                            ):
                                for card_id in witness[3]:
                                    model.add(
                                        model_owner.owner_of_card[card_id]
                                        == int(witness[0])
                                    ).only_enforce_if(selected)
                    target = max(
                        projects.matching_projects(
                            owner=arguments.get("owner"),
                            team=arguments["team"],
                            kind=arguments["kind"],
                            value=int(arguments["value"]),
                        ),
                        key=lambda item: (
                            item.value,
                            item.category_priority,
                            item.high_rank,
                            item.project_id,
                        ),
                        default=None,
                    )
                    if target is not None:
                        model.add_hint(
                            target.literal(
                                str(arguments.get("state") or "eligible")
                            ),
                            1,
                        )
                        for card in target.cards:
                            model.add_hint(
                                model_owner.owner_of_card[card.id],
                                int(target.owner),
                            )
                predicate_literals.update(project_literals)
                from .ortools_research_executor import _enforce_hard_stage_projection

                _enforce_hard_stage_projection(
                    model,
                    scenario,
                    stage="projects",
                    predicate_literals=project_literals,
                )
            escalation_stage = OrToolsEscalationStageV1(
                EscalationDomainIRV1.from_rules(self.rules.data),
                declarer=buyer,
                contract=contract,
                match_score=self.starting_match_score,
                native_cp_sat=self.native_cp_sat,
            )
            escalation_context = CpSatEscalationPredicateContextV1.attach(
                escalation_stage,
                model=model,
                context_id="tail_boundary_escalation",
            )
            if self.native_cp_sat:
                model.add(escalation_context.native["terminal"] == 1)
            for candidate, selected in zip(
                escalation_context.candidates,
                escalation_context.selected,
                strict=True,
            ):
                if not candidate.terminal:
                    model.add(selected == 0)
            if escalation_predicates:
                escalation_lowerings = {
                    predicate.predicate_id: manifest.lower_predicate(
                        predicate, escalation_context
                    )
                    for predicate in escalation_predicates
                }
                predicate_lowerings.update(escalation_lowerings)
                escalation_literals = {
                    predicate_id: lowering.literal
                    for predicate_id, lowering in escalation_lowerings.items()
                }
                predicate_literals.update(escalation_literals)
                from .ortools_research_executor import _enforce_hard_stage_projection

                _enforce_hard_stage_projection(
                    model,
                    scenario,
                    stage="escalation",
                    predicate_literals=escalation_literals,
                )
            completed_tricks = (plan.tail_offset or 0) // 4
            base_raw_ns = model.new_int_var(
                0, prefix_raw_total, "tail_boundary_base_raw_ns"
            )
            base_raw_we = model.new_int_var(
                0, prefix_raw_total, "tail_boundary_base_raw_we"
            )
            model.add(base_raw_ns + base_raw_we == prefix_raw_total)
            base_tricks_ns = model.new_int_var(
                0, completed_tricks, "tail_boundary_base_tricks_ns"
            )
            base_tricks_we = model.new_int_var(
                0, completed_tricks, "tail_boundary_base_tricks_we"
            )
            model.add(base_tricks_ns + base_tricks_we == completed_tricks)
            tail_card_ids = {play.card.id for play in tail.plays}
            prefix_cards = tuple(
                Card(card_id) for card_id in range(32)
                if card_id not in tail_card_ids
            )
            collected_by_ns = tuple(
                model.new_bool_var(f"tail_boundary_prefix_card_{card.id}_ns")
                for card in prefix_cards
            )
            model.add(sum(collected_by_ns) == 4 * base_tricks_ns)
            model.add(
                base_raw_ns == sum(
                    self.rules.play_domain.points(card, contract, trump) * selected
                    for card, selected in zip(prefix_cards, collected_by_ns, strict=True)
                )
            )
            failed_history = any(
                item.tail_evidence_hash == tail.evidence_hash
                and item.boundary_combination == combination
                for item in excluded
            )
            prefix = None
            if self.native_cp_sat and (deal_predicates or failed_history) and fixed_owners is None and contract.effective == Contract.SUN:
                # Reuse legal play to constrain the provisional history when
                # authored hands or a failed lift expose unreachable aggregates.
                # The final lift still owns auction/project reachability.
                try:
                    prefix = OrToolsPlayTailStageV1(native_conversion=True).build_model(
                        cards=FULL_DECK,
                        ownership={card: seat for seat in Seat
                                   for card in tail.initial_hands[int(seat)]},
                        contract=contract, trump=trump,
                        first_leader=self.rules.play_domain.resolve_first_leader(
                            dealer=dealer, buyer=buyer,
                            structural_domain=self.rules.structural_domain,
                        ),
                        play_domain=self.rules.play_domain,
                        structural_domain=self.rules.structural_domain,
                        scoring_domain=scoring,
                        tail=tail.plays[:4],
                        play_step_count=(plan.tail_offset or 0) + 4,
                        decision_prefix_length=plan.prefix_length,
                        shared_model=model,
                        shared_owner_of_card=model_owner.owner_of_card,
                        cancel_check=stopped,
                    )
                except PlayTailSolveStopped:
                    saw_unknown = True
                    break
                for selected in prefix.selected_play_indexes[:plan.tail_offset]:
                    for card_id in tail_card_ids:
                        model.add(selected != card_id)
                context = prefix.play_state_context
                model.add(base_raw_ns == context.raw_points[Team.NORTH_SOUTH])
                model.add(base_tricks_ns == sum(
                    1 - winner for winner in prefix.winner_team_variables[:completed_tricks]
                ))
                model.add(context.next_actor == int(tail.next_actor))
                model.add(context.current_trick_size == len(tail.current_trick))
                legal = {(action.card.id, action.ekka) for action in tail.legal_play_actions}
                for action, literal in context.legal_action_literals.items():
                    model.add(literal == int(action in legal))
            elif reveal_ordered_projects:
                try:
                    prefix = OrToolsPlayTailStageV1(native_conversion=True).build_model(
                        cards=FULL_DECK,
                        ownership={card: seat for seat in Seat
                                   for card in tail.initial_hands[int(seat)]},
                        contract=contract, trump=trump,
                        first_leader=self.rules.play_domain.resolve_first_leader(
                            dealer=dealer, buyer=buyer,
                            structural_domain=self.rules.structural_domain,
                        ),
                        play_domain=self.rules.play_domain,
                        structural_domain=self.rules.structural_domain,
                        scoring_domain=scoring,
                        tail=tail.plays[:4] if not plan.tail_offset else (),
                        play_step_count=4,
                        decision_prefix_length=0,
                        shared_model=model,
                        shared_owner_of_card=model_owner.owner_of_card,
                        cancel_check=stopped,
                    )
                except PlayTailSolveStopped:
                    saw_unknown = True
                    break
                if plan.tail_offset:
                    for selected in prefix.selected_play_indexes:
                        for card_id in tail_card_ids:
                            model.add(selected != card_id)
            if second_trick_leader is not None:
                model.add(second_trick_leader == prefix._extract.winner_seats[0])
            if required_base_raw_ns is not None:
                model.add(base_raw_ns == required_base_raw_ns)
            if required_base_tricks_ns is not None:
                model.add(base_tricks_ns == required_base_tricks_ns)
            if preferred_base_raw_ns is not None:
                model.add_hint(base_raw_ns, preferred_base_raw_ns)
            if preferred_base_tricks_ns is not None:
                model.add_hint(base_tricks_ns, preferred_base_tricks_ns)
            known_prefix_winner_teams: dict[int, Team] = {}
            for predicate in play_predicates if self.native_cp_sat else ():
                if predicate.kind != "play.trick_winner_matches":
                    continue
                arguments = dict(predicate.arguments)
                trick_index = int(arguments["trickIndex"])
                if trick_index >= completed_tricks:
                    continue
                if not _hard_expression_requires_predicate(
                    scenario, predicate.predicate_id, maximum_conflicts=maximum_conflicts,
                ):
                    continue
                if arguments.get("seat") is not None:
                    required_team = Seat.parse(arguments["seat"]).team
                elif arguments.get("team") is not None:
                    required_team = Team[str(arguments["team"]).upper()]
                else:
                    continue
                prior = known_prefix_winner_teams.setdefault(
                    trick_index, required_team
                )
                if prior != required_team:
                    model.add(0 == 1)
            known_ns = sum(
                team == Team.NORTH_SOUTH
                for team in known_prefix_winner_teams.values()
            )
            unknown = completed_tricks - len(known_prefix_winner_teams)
            model.add(base_tricks_ns >= known_ns)
            model.add(base_tricks_ns <= known_ns + unknown)
            if completed_tricks and tail.current_trick:
                leader_team = tail.current_trick[0].actor.team
                model.add(
                    (
                        base_tricks_ns
                        if leader_team == Team.NORTH_SOUTH
                        else base_tricks_we
                    )
                    >= 1
                )
            escalation_index = model.new_int_var(
                0,
                (2 * 75 ** len(escalation_context.native["actionSlots"]) - 1
                 if self.native_cp_sat else len(escalation_context.selected) - 1),
                "tail_boundary_escalation_index",
            )
            model.add(
                escalation_index
                == (sum(
                    (actor * 15 + action * 3 + mode) * 75 ** index
                    for index, (actor, action, mode) in enumerate(escalation_context.native["actionSlots"])
                ) * 2 + escalation_context.native["terminal"] if self.native_cp_sat else sum(
                    index * selected
                    for index, selected in enumerate(escalation_context.selected)
                ))
            )
            escalation_semantics_by_candidate = tuple(
                (
                    candidate.level,
                    candidate.locked,
                    (
                        candidate.actions[-1].actor.team.name.lower()
                        if candidate.actions
                        else None
                    ),
                )
                for candidate in escalation_context.candidates
            )
            escalation_semantics_codes = {
                value: index
                for index, value in enumerate(
                    sorted(
                        ({(level, locked, team) for level in escalation_context.native["levels"]
                          for locked in (False, True) for team in (None, "north_south", "west_east")}
                         if self.native_cp_sat else set(escalation_semantics_by_candidate)),
                        key=lambda item: (item[0], item[1], item[2] or ""),
                    )
                )
            }
            escalation_semantics_code = model.new_int_var(
                0,
                len(escalation_semantics_codes) - 1,
                "tail_boundary_escalation_semantics",
            )
            if self.native_cp_sat:
                model.add_allowed_assignments(
                    (escalation_context.native["level"], escalation_context.native["locked"],
                     escalation_context.native["lastTeam"], escalation_semantics_code),
                    [(escalation_context.native["levels"][level], int(locked),
                      2 if team is None else int(Team[team.upper()]), code)
                     for (level, locked, team), code in escalation_semantics_codes.items()],
                )
            else:
                model.add(escalation_semantics_code == sum(
                    escalation_semantics_codes[semantics] * selected
                    for semantics, selected in zip(
                        escalation_semantics_by_candidate,
                        escalation_context.selected,
                        strict=True,
                    )
                ))
            scoring_fragment = None
            if outcome_predicates:
                scoring_source_hash = canonical_sha256(
                    {
                        "schemaVersion": 1,
                        "rulesHash": self.rules.sha256,
                        "planHash": plan.sha256,
                        "tailEvidenceHash": tail.evidence_hash,
                        "boundaryCombination": list(combination),
                    }
                )
                scoring_fragment = _attach_native_boundary_scoring(
                    model,
                    scoring=scoring,
                    rules=self.rules,
                    contract=contract,
                    buyer=buyer,
                    tail=tail,
                    base_raw_ns=base_raw_ns,
                    base_raw_we=base_raw_we,
                    base_tricks_ns=base_tricks_ns,
                    completed_tricks=completed_tricks,
                    escalation_context=escalation_context,
                    projects=projects,
                    starting_match_score=self.starting_match_score,
                    source_hash=scoring_source_hash,
                )
                outcome_lowerings = {
                    predicate.predicate_id: manifest.lower_predicate(
                        predicate, scoring_fragment.outcome_context
                    )
                    for predicate in outcome_predicates
                }
                predicate_lowerings.update(outcome_lowerings)
                outcome_literals = {
                    predicate_id: lowering.literal
                    for predicate_id, lowering in outcome_lowerings.items()
                }
                predicate_literals.update(outcome_literals)
                from .ortools_research_executor import _enforce_hard_stage_projection

                _enforce_hard_stage_projection(
                    model,
                    scenario,
                    stage="outcome",
                    predicate_literals=outcome_literals,
                )
            if state_predicates:
                base_game = {
                    team: attach_raw_conversion_v1(
                        model, base_raw_ns if team == Team.NORTH_SOUTH else base_raw_we,
                        domain=scoring, contract=contract,
                        name=f"tail_boundary_base_game_{int(team)}",
                    )
                    for team in Team
                }
                state_context = _state_context(
                    model,
                    tail=tail,
                    raw_points={
                        Team.NORTH_SOUTH: base_raw_ns,
                        Team.WEST_EAST: base_raw_we,
                    },
                    project_points={
                        team: (
                            projects.combined_awarded_value[int(team)]
                            if projects is not None
                            else model.new_constant(0)
                        )
                        for team in Team
                    },
                    game_points=base_game,
                    match_score=self.starting_match_score,
                    trump=trump,
                )
                from .ortools_research_executor import _enforce_hard_stage_projection

                state_lowerings = {
                    predicate.predicate_id: manifest.lower_predicate(
                        predicate, state_context
                    )
                    for predicate in state_predicates
                }
                predicate_lowerings.update(state_lowerings)
                state_literals = {
                    predicate_id: lowering.literal
                    for predicate_id, lowering in state_lowerings.items()
                }
                predicate_literals.update(state_literals)
                _enforce_hard_stage_projection(
                    model,
                    scenario,
                    stage="state",
                    predicate_literals=state_literals,
                )
            cut_vars = (
                *model_owner.owner_of_card,
                base_raw_ns,
                base_tricks_ns,
                escalation_index,
                *((*projects.combined_awarded_value, *projects.baloot_team)
                  if reveal_ordered_projects else ()),
            )
            if required_history is not None:
                if (
                    required_history.tail_evidence_hash != tail.evidence_hash
                    or required_history.boundary_combination != combination
                    or len(required_history.vector) != len(cut_vars)
                ):
                    raise ValueError("required tail history belongs to another boundary")
                for owner, value in zip(
                    model_owner.owner_of_card,
                    required_history.vector[: len(model_owner.owner_of_card)],
                    strict=True,
                ):
                    model.add(owner == value)
            for item in excluded:
                if (
                    item.tail_evidence_hash == tail.evidence_hash
                    and item.boundary_combination == combination
                ):
                    model.add_forbidden_assignments(cut_vars, (item.vector,))
            proof_cut_vars = None
            if proof_state_exclusions or required_proof_state is not None:
                # Tail + combination fix every other proof-seed field. These seven
                # values are the complete variable projection consumed by proof.
                proof_cut_vars = (
                    base_raw_ns,
                    base_tricks_ns,
                    *projects.combined_awarded_value,
                    *projects.baloot_team,
                    escalation_semantics_code,
                )
            if proof_state_exclusions and proof_cut_vars is not None:
                for item in proof_state_exclusions:
                    semantics_code = escalation_semantics_codes.get(
                        item.escalation_semantics
                    )
                    if semantics_code is None:
                        raise ValueError(
                            "proof-state exclusion escalation semantics are unavailable"
                        )
                    model.add_forbidden_assignments(
                        proof_cut_vars,
                        (item.cp_vector(semantics_code),),
                    )
            if required_proof_state is not None and proof_cut_vars is not None:
                semantics_code = escalation_semantics_codes.get(
                    required_proof_state.escalation_semantics
                )
                if semantics_code is None:
                    return OrToolsTailProofBoundaryResultV1(
                        "infeasible",
                        None,
                        None,
                        None,
                        None,
                        "infeasible",
                        0,
                        0,
                        outcome_rejections,
                        "required_proof_state_infeasible",
                    )
                for variable, value in zip(
                    proof_cut_vars,
                    required_proof_state.cp_vector(semantics_code),
                    strict=True,
                ):
                    model.add(variable == value)
            model.add_decision_strategy(
                list(cut_vars),
                cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE,
            )
            model_build_seconds += time.perf_counter() - model_started

            if stopped():
                saw_unknown = True
                break

            while time.monotonic() < deadline:
                if stopped():
                    saw_unknown = True
                    break
                solver = cp_model.CpSolver()
                solver.parameters.num_search_workers = 1
                solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
                solver.parameters.cp_model_probing_level = 0
                solver.parameters.max_time_in_seconds = max(
                    0.001, deadline - time.monotonic()
                )
                solver.parameters.max_number_of_conflicts = maximum_conflicts
                solver_started = time.perf_counter()
                status = _solve_cp_sat_with_cancel_v1(
                    solver, model, stopped
                )
                solver_seconds += time.perf_counter() - solver_started
                solve_attempts += 1
                solver_status = solver.status_name(status).lower()
                if status == cp_model.INFEASIBLE:
                    break
                if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
                    saw_unknown = True
                    break
                if projects is not None:
                    project_result = project_lifecycle_result_v1(projects, solver)
                else:
                    hands = {
                        seat: tuple(
                            Card(card_id)
                            for card_id, owner in enumerate(
                                fixed_owners or model_owner.owner_of_card
                            )
                            if int(
                                owner
                                if fixed_owners is not None
                                else solver.value(owner)
                            )
                            == int(seat)
                        )
                        for seat in Seat
                    }
                    project_result = OrToolsProjectLifecycleModelV1().solve(
                        hands=hands,
                        contract=contract,
                        dealer=dealer,
                        trump=trump,
                        domain=self.rules.project_domain,
                        timeout_seconds=max(0.001, deadline - time.monotonic()),
                        maximum_conflicts=maximum_conflicts,
                        cancel_check=stopped,
                    )
                    if project_result.solver_status not in {"optimal", "feasible"}:
                        saw_unknown = True
                        break
                project_points = project_result.combined_awarded_value
                baloot_points = project_result.baloot_points
                project_evidence_hash = project_result.evidence_hash
                scoring_evidence = (
                    extract_scoring_evidence_v1(
                        solver,
                        scoring_fragment,
                        scoring,
                        contract=contract,
                    )
                    if scoring_fragment is not None
                    else None
                )
                if self.native_cp_sat:
                    selected_escalation = escalation_context.extract_stage_result(
                        solver, model_hash=canonical_sha256(str(model.proto)),
                        solver_status=solver_status, branches=solver.num_branches,
                        conflicts=solver.num_conflicts,
                    ).observation
                    escalation_locked = selected_escalation.mode == "locked"
                else:
                    selected_index = int(solver.value(escalation_index))
                    selected_escalation = escalation_context.candidates[selected_index]
                    escalation_locked = selected_escalation.locked
                history = selected_escalation.actions
                last_team = history[-1].actor.team if history else None
                base_raw = (
                    int(solver.value(base_raw_ns)),
                    int(solver.value(base_raw_we)),
                )
                base_tricks = (
                    int(solver.value(base_tricks_ns)),
                    int(solver.value(base_tricks_we)),
                )
                vector = tuple(int(solver.value(value)) for value in cut_vars)
                no_good = TailProofBoundaryNoGoodV1.build(
                    tail_evidence_hash=tail.evidence_hash,
                    boundary_combination=combination,
                    vector=vector,
                )
                seed_core = {
                    "schemaVersion": 1,
                    "rulesHash": self.rules.sha256,
                    "tailEvidenceHash": tail.evidence_hash,
                    "combination": list(combination),
                    "baseRaw": list(base_raw),
                    "baseTricks": list(base_tricks),
                    "projectPoints": list(project_points),
                    "balootPoints": list(baloot_points),
                    "projectEvidenceHash": project_evidence_hash,
                    "escalation": selected_escalation.level,
                    "lastEscalatingTeam": (
                        last_team.name.lower() if last_team is not None else None
                    ),
                    "startingMatchScore": list(self.starting_match_score),
                }
                exact_state_id = "STATE-" + canonical_sha256(seed_core)[:24].upper()
                seed = CompleteStateGraphSeedV1.build(
                    exact_state_id=exact_state_id,
                    rules_hash=self.rules.sha256,
                    remaining_hands=tail.remaining_hands,
                    current_trick=tail.current_trick,
                    first_leader=(
                        tail.current_trick[0].actor
                        if tail.current_trick
                        else tail.next_actor
                    ),
                    next_actor=tail.next_actor,
                    legal_actions=tail.legal_play_actions,
                    contract=contract,
                    trump=trump,
                    buyer=buyer,
                    locked=escalation_locked,
                    base_raw=base_raw,
                    base_tricks=base_tricks,
                    awarded_project_points=project_points,
                    baloot_points=baloot_points,
                    escalation=selected_escalation.level,
                    last_escalating_team=last_team,
                    starting_match_score=self.starting_match_score,
                )
                if projects is None and any(
                    seed.proof_state_projection_hash
                    == item.proof_state_projection_hash
                    for item in proof_state_exclusions
                ):
                    model.add_forbidden_assignments(cut_vars, (vector,))
                    continue
                if (
                    required_proof_state is not None
                    and seed.proof_state_projection_hash
                    != required_proof_state.proof_state_projection_hash
                ):
                    if projects is None:
                        model.add_forbidden_assignments(cut_vars, (vector,))
                        continue
                    raise RuntimeError(
                        "required proof-state projection disagrees with the CP-SAT boundary"
                    )
                proof_state_no_good = TailProofStateNoGoodV1.build(
                    decision_state_hash=decision_state_hash,
                    boundary_combination=combination,
                    base_raw_ns=base_raw[0],
                    base_tricks_ns=base_tricks[0],
                    awarded_project_points=project_points,
                    baloot_teams=tuple(
                        team in project_result.baloot_teams for team in Team
                    ),
                    escalation_semantics=(
                        selected_escalation.level,
                        escalation_locked,
                        last_team.name.lower() if last_team is not None else None,
                    ),
                    proof_state_projection_hash=seed.proof_state_projection_hash,
                )
                model_hash = canonical_sha256(
                    {
                        "schemaVersion": 1,
                        "planHash": plan.sha256,
                        "tailModelHash": tail.model_hash,
                        "cpModelProto": str(model.proto),
                    }
                )
                predicate_by_id = {
                    predicate.predicate_id: predicate
                    for predicate in scenario.predicates()
                }
                predicate_evidence = {
                    predicate_id: lowering.extract(solver)
                    for predicate_id, lowering in sorted(
                        predicate_lowerings.items()
                    )
                }
                evidence_hash = canonical_sha256(
                    {
                        "schemaVersion": 1,
                        "seed": seed.to_dict(),
                        "nativeScoring": scoring_evidence,
                        "predicateEvidenceHashes": {
                            predicate_id: evidence["sha256"]
                            for predicate_id, evidence in sorted(
                                predicate_evidence.items()
                            )
                        },
                    }
                )
                evidence_bindings = tuple(
                    {
                        "predicateId": predicate_id,
                        "sourceEvidenceHash": evidence_hash,
                        "loweringHash": canonical_sha256(
                            {
                                "schemaVersion": 1,
                                "predicateHash": predicate_by_id[predicate_id].sha256,
                                "loweringManifestHash": manifest.manifest_hash,
                            }
                        ),
                        "modelHash": model_hash,
                        "literalName": literal.name,
                        "literalIndex": literal.index,
                        "evidenceHash": predicate_evidence[predicate_id]["sha256"],
                    }
                    for predicate_id, literal in sorted(predicate_literals.items())
                )
                return OrToolsTailProofBoundaryResultV1(
                    "feasible",
                    seed,
                    no_good,
                    model_hash,
                    evidence_hash,
                    solver_status,
                    int(solver.num_branches),
                    int(solver.num_conflicts),
                    outcome_rejections,
                    proof_state_no_good=proof_state_no_good,
                    variable_count=len(model.proto.variables),
                    constraint_count=len(model.proto.constraints),
                    evidence_bindings=evidence_bindings,
                    predicate_evidence=predicate_evidence,
                    project_result=project_result,
                    model_build_seconds=model_build_seconds,
                    solver_seconds=solver_seconds,
                    solve_attempts=solve_attempts,
                )
        if time.monotonic() >= deadline:
            saw_unknown = True
        return OrToolsTailProofBoundaryResultV1(
            "incomplete" if saw_unknown else "infeasible",
            None,
            None,
            None,
            None,
            "unknown" if saw_unknown else "infeasible",
            0,
            0,
            outcome_rejections,
            "tail_boundary_budget" if saw_unknown else "tail_boundary_infeasible",
            model_build_seconds=model_build_seconds,
            solver_seconds=solver_seconds,
            solve_attempts=solve_attempts,
        )


__all__ = [
    "OrToolsTailProofBoundaryResultV1",
    "OrToolsTailProofBoundaryV1",
    "TailProofBoundaryNoGoodV1",
    "TailProofStateNoGoodV1",
]
