"""Engine-free lift and exact proof orchestration for the flagship scenario."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ortools import __version__ as ortools_version
from ortools.sat.python import cp_model

from baloot_platform.core.cards import FULL_DECK, Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, Rank, Seat, Suit, Team
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1

from .ortools_flagship_tail import (
    FlagshipTailCandidateV1,
    OrToolsFlagshipTailSearchV1,
)
from .ortools_infeasible_core import sound_infeasible_assumption_indexes
from .ortools_information_set_proof import _solve_cp_sat_with_cancel_v1
from .ortools_project_fragment import (
    ProjectLifecycleFragmentEvidenceV1,
    attach_project_lifecycle_fragment_v1,
    constrain_exclusive_project_awards_v1,
)
from .ortools_play_tail import (
    OrToolsPlayTailStageV1,
    PlayTailSolveStopped,
    TailActionConstraintV1,
    TailPrefixActionV1,
    TailTrickCardCountConstraintV1,
    TailTrickWinnerConstraintV1,
)
from .ortools_proof import (
    OrToolsGraphProofOutcomeV1,
    OrToolsScenarioGraphProofV1,
    ProofGraphEdgeV1,
    ProofGraphNodeV1,
    ProofGraphV1,
)
from .ortools_scenario_model import OrToolsScenarioModelV1
from .ortools_scoring_stage import CpSatScoringInputsV1, attach_scoring_fragment_v1
from .scenario_contracts import (
    ResearchScenarioV1,
    ScenarioExpressionV1,
    ScenarioPredicateV1,
    ScenarioProofRequestV1,
    canonical_sha256,
)


@dataclass(frozen=True, slots=True)
class FlagshipCompletePlayV1:
    trick_index: int
    ordinal: int
    actor: Seat
    card: Card

    def to_dict(self) -> dict[str, Any]:
        return {
            "trickIndex": self.trick_index,
            "ordinal": self.ordinal,
            "actor": self.actor.name.lower(),
            "card": self.card.code,
        }


@dataclass(frozen=True, slots=True)
class FlagshipLiftV1:
    status: str
    tail: FlagshipTailCandidateV1
    dealer: Seat
    deck_order: tuple[int, ...]
    hands: tuple[tuple[Card, ...], ...]
    auction_actions: tuple[Mapping[str, Any], ...]
    projects: ProjectLifecycleFragmentEvidenceV1
    plays: tuple[FlagshipCompletePlayV1, ...]
    trick_winners: tuple[Seat, ...]
    prefix_raw_points: tuple[int, int]
    construction_model_hash: str
    lift_hash: str
    branches: int
    conflicts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "status": self.status,
            "tailCandidateHash": self.tail.candidate_hash,
            "dealer": self.dealer.name.lower(),
            "deckOrder": list(self.deck_order),
            "hands": {
                seat.name.lower(): [card.code for card in self.hands[int(seat)]]
                for seat in Seat
            },
            "auctionActions": [dict(action) for action in self.auction_actions],
            "projects": self.projects.to_dict(),
            "plays": [play.to_dict() for play in self.plays],
            "trickWinners": [seat.name.lower() for seat in self.trick_winners],
            "prefixRawPoints": list(self.prefix_raw_points),
            "constructionModelHash": self.construction_model_hash,
            "liftHash": self.lift_hash,
            "branches": self.branches,
            "conflicts": self.conflicts,
        }


@dataclass(frozen=True, slots=True)
class FlagshipResearchResultV1:
    status: str
    tail: FlagshipTailCandidateV1 | None
    lift: FlagshipLiftV1 | None
    graph: ProofGraphV1 | None
    proof: OrToolsGraphProofOutcomeV1 | None
    contradiction: FlagshipContradictionReceiptV1
    tail_no_good: tuple[int, ...] | None
    tail_no_good_hash: str | None
    rejected_tail_no_goods: tuple[tuple[int, ...], ...]
    rejected_tail_hashes: tuple[str, ...]
    stage_timings: Mapping[str, float]
    research_hash: str

    @property
    def qualified(self) -> bool:
        return self.status == "qualified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "status": self.status,
            "qualified": self.qualified,
            "tail": self.tail.to_dict() if self.tail is not None else None,
            "lift": self.lift.to_dict() if self.lift is not None else None,
            "proofGraphHash": self.graph.sha256 if self.graph is not None else None,
            "proof": self.proof.to_dict() if self.proof is not None else None,
            "contradiction": self.contradiction.to_dict(),
            "tailNoGood": list(self.tail_no_good)
            if self.tail_no_good is not None
            else None,
            "tailNoGoodHash": self.tail_no_good_hash,
            "rejectedTailHashes": list(self.rejected_tail_hashes),
            "stageTimings": dict(self.stage_timings),
            "researchHash": self.research_hash,
        }


@dataclass(frozen=True, slots=True)
class FlagshipContradictionReceiptV1:
    status: str
    theorem_id: str
    checked_acting_seats: tuple[Seat, ...]
    diagnostics: tuple[Mapping[str, Any], ...]
    solver_status: str
    model_hash: str
    variable_count: int
    constraint_count: int
    assumption_core: tuple[str, ...]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "status": self.status,
            "theoremId": self.theorem_id,
            "checkedActingSeats": [
                seat.name.lower() for seat in self.checked_acting_seats
            ],
            "reason": (
                "The opposing trick-six winner must retain one of the two opposing "
                "Aces for trick eight; as trick-seven leader they can lead that "
                "unbeatable Sun Ace, so the target team cannot force trick seven."
            ),
            "diagnostics": [dict(item) for item in self.diagnostics],
            "solverStatus": self.solver_status,
            "modelHash": self.model_hash,
            "variableCount": self.variable_count,
            "constraintCount": self.constraint_count,
            "assumptionCore": list(self.assumption_core),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class _ProofStateV1:
    hands: tuple[int, int, int, int]
    current_trick: tuple[tuple[int, int], ...]
    next_actor: Seat
    raw_points: tuple[int, int]
    completed_tricks: int


def _tail_no_good_hash(tail: FlagshipTailCandidateV1) -> str:
    return canonical_sha256(
        {
            "schemaVersion": 1,
            "kind": "flagship_tail_lift_no_good",
            "cards": list(tail.exact_no_good),
        }
    )


class OrToolsFlagshipResearchV1:
    """Lift CP tail witnesses and qualify them through exact CP graph proof."""

    def __init__(self, *, rules: RuleBook, acting_seat: Seat = Seat.WEST) -> None:
        if not isinstance(rules, RuleBook):
            raise TypeError("flagship research requires RuleBook")
        self.rules = rules
        self.acting_seat = Seat.parse(acting_seat)
        self.structural = rules.structural_domain
        self.play = rules.play_domain
        self.project = rules.project_domain
        self.scoring = ScoringDomainIRV1.from_rules(rules.data)

    def certify_round_win_counterstrategy(
        self,
        scenario: ResearchScenarioV1,
        proof_request: ScenarioProofRequestV1,
        *,
        maximum_seconds: float,
        maximum_conflicts: int,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Mapping[str, Any] | None:
        """Native evidence for the original closed form, not a V2 forced-trick claim.

        None means this theorem does not cover the request. An incomplete or
        unestablished receipt is never evidence of scenario impossibility.
        """
        if not isinstance(scenario, ResearchScenarioV1) or not isinstance(proof_request, ScenarioProofRequestV1):
            raise TypeError("counterstrategy requires typed scenario and proof")
        if (type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
                or maximum_seconds <= 0 or type(maximum_conflicts) is not int or maximum_conflicts < 1):
            raise ValueError("counterstrategy requires finite positive governed budgets")
        if (scenario.origin.kind != "search" or proof_request.mode != "exact"
                or proof_request.visibility != "complete_state" or proof_request.utility != "round_win"
                or proof_request.claim != "winning_root_action_count"
                or proof_request.target_team != "west_east" or proof_request.success_threshold != 1
                or canonical_sha256(scenario.decision_point) != canonical_sha256({
                    "kind": "play", "selector": {"trickIndex": 5, "position": 3,
                        "remainingCards": 3, "actingSeat": "west"}})
                or (self.project.baloot_enabled and not self.project.baloot_only_hokum)):
            return None
        # Applicability is deliberately exact. Other compositions still use
        # ordinary search; this optimization must not broaden their meaning.
        low = {"ranks": ["seven", "eight", "nine"]}
        ace = {"ranks": ["ace"], "owner": {"relation": "actor"}}
        pattern = (
            ("auction.contract", {"contract": "sun"}),
            ("auction.buyer", {"seat": "north"}),
            ("project.exists", {"team": "north_south", "owner": "north",
                "kind": "hundred_sequence", "value": 20, "state": "awarded"}),
            ("play.action_matches", {"trickIndex": 5, "position": 3, "actingSeat": "west", "selector": ace}),
            ("play.trick_winner_matches", {"trickIndex": 5, "team": "north_south", "selector": low}),
            ("play.trick_winner_matches", {"trickIndex": 6, "team": "west_east", "selector": low}),
            ("play.trick_winner_matches", {"trickIndex": 7, "team": "west_east", "selector": low}),
            ("play.trick_card_count", {"trickIndex": 7, "minimum": 2, "maximum": 2,
                "selector": {"ranks": ["ace"], "owner": {"relation": "team", "team": "north_south"}}}),
            ("proof.winning_root_action_matches", {"selector": ace}),
            ("proof.winning_root_action_count", {"minimum": 1, "maximum": 1}),
        )
        required = self._required_hard_predicates(scenario.constraints)
        matched = []
        for kind, arguments in pattern:
            found = next((item for item in required if item.kind == kind
                and canonical_sha256(item.arguments) == canonical_sha256(arguments)), None)
            if found is None:
                return None
            matched.append(found.predicate_id)
        started, cpu_started = time.perf_counter(), time.process_time()
        deadline = time.monotonic() + maximum_seconds
        facts, fact_timings = [], []
        conflicts = branches = 0
        status, reason = "incomplete", "counterstrategy_budget"

        def stopped():
            return time.monotonic() >= deadline or (cancel_check is not None and cancel_check())

        try:
            previous_finished = time.perf_counter()
            for fact_id, model in self._round_win_counterstrategy_models(stopped):
                ready = time.perf_counter()
                build_seconds = ready - previous_finished
                if stopped() or conflicts >= maximum_conflicts:
                    break
                model_hash = canonical_sha256({"cpModelProto": str(model.proto)})
                hashed = time.perf_counter()
                if stopped():
                    break
                solver = cp_model.CpSolver()
                solver.parameters.num_search_workers = 1
                solver.parameters.max_time_in_seconds = max(0.001, deadline - time.monotonic())
                solver.parameters.max_number_of_conflicts = maximum_conflicts - conflicts
                solved = _solve_cp_sat_with_cancel_v1(solver, model, stopped)
                previous_finished = time.perf_counter()
                fact_timings.append({"factId": fact_id, "modelBuildSeconds": build_seconds,
                    "modelHashSeconds": hashed - ready, "solveSeconds": previous_finished - hashed})
                conflicts += solver.num_conflicts
                branches += solver.num_branches
                facts.append({"factId": fact_id, "modelHash": model_hash,
                    "variableCount": len(model.proto.variables), "constraintCount": len(model.proto.constraints),
                    "solverStatus": solver.status_name(solved).lower()})
                if solved != cp_model.INFEASIBLE:
                    if solved in {cp_model.FEASIBLE, cp_model.OPTIMAL}:
                        status, reason = "not_certified", "counterstrategy_counterexample"
                    elif solved == cp_model.MODEL_INVALID:
                        status, reason = "not_certified", "counterstrategy_model_invalid"
                    break
            else:
                if not stopped():
                    status, reason = "certified", "native_round_win_counterstrategy"
        except PlayTailSolveStopped:
            pass
        core = {"schemaVersion": 1, "theoremId": "SUN-ACE-LEAD-AWARDED-PROJECT-V1",
            "status": status, "reason": reason, "solverVersion": ortools_version,
            "scenarioHash": scenario.sha256,
            "proofRequestHash": proof_request.sha256, "rulesHash": self.rules.sha256,
            "domainIrHashes": {"play": self.play.sha256, "structural": self.structural.sha256,
                "projects": self.project.sha256, "scoring": self.scoring.sha256},
            "namedCorePredicateIds": matched, "facts": facts}
        return {**core, "sha256": canonical_sha256(core), "executionReceipt": {
            "wallSeconds": time.perf_counter() - started, "cpuSeconds": time.process_time() - cpu_started,
            "conflicts": conflicts, "branches": branches,
            "maximumSeconds": maximum_seconds, "maximumConflicts": maximum_conflicts,
            "factTimings": fact_timings}}

    def _round_win_counterstrategy_models(self, stopped):
        """Counterexample models; Python never evaluates their hard truth."""
        stage = OrToolsPlayTailStageV1(native_conversion=True)
        options = dict(cards=FULL_DECK, ownership=None, contract=Contract.SUN,
            trump=None, first_leader=self.structural.offset(Seat.WEST, -3), play_domain=self.play,
            structural_domain=self.structural, selected_play_count=12,
            decision_prefix_length=3, decision_actor=Seat.WEST, cancel_check=stopped,
            action_constraints=(TailActionConstraintV1.build(step=3, ranks=[Rank.ACE]),))
        witness = stage.build_model(**options,
            winner_constraints=tuple(TailTrickWinnerConstraintV1.build(trick=index,
                team=team, ranks=[Rank.SEVEN, Rank.EIGHT, Rank.NINE])
                for index, team in enumerate((Team.NORTH_SOUTH, Team.WEST_EAST, Team.WEST_EAST))),
            card_count_constraints=(TailTrickCardCountConstraintV1.build(trick=2,
                minimum=2, maximum=2, owner_team=Team.NORTH_SOUTH, ranks=[Rank.ACE]),))
        aces = [card for card in FULL_DECK if card.rank == Rank.ACE]
        retained = witness.model.clone()
        legal = witness.native_legal_action_literals(4)
        retained.add(sum(retained.get_bool_var_from_proto_index(legal[card.code].index) for card in aces) == 0)
        yield "opposing_winner_has_legal_ace_lead", retained

        alternative = stage.build_model(**options, shared_model=witness.model,
            shared_owner_of_card=witness.initial_owner_variables)
        for left, right in zip(witness.selected_card_literals, alternative.selected_card_literals, strict=True):
            witness.model.add(left == right)
        for index in range(3):
            witness.model.add(witness.selected_play_indexes[index] == alternative.selected_play_indexes[index])
        changed_winner = witness.model.clone()
        changed_winner.add(changed_winner.get_int_var_from_proto_index(
            alternative.winner_team_variables[0].index) != int(Team.NORTH_SOUTH))
        yield "every_legal_root_ace_loses_trick_six", changed_winner

        witness.model.add_allowed_assignments([alternative.selected_play_indexes[4]], [(card.id,) for card in aces])
        failed_lead = witness.model.clone()
        failed_lead.add(failed_lead.get_int_var_from_proto_index(
            alternative.winner_team_variables[1].index) != int(Team.NORTH_SOUTH))
        yield "retained_ace_lead_wins_trick_seven", failed_lead
        insufficient_raw = witness.model.clone()
        ace_points = self.play.points(aces[0], Contract.SUN, None)
        insufficient_raw.add(sum(insufficient_raw.get_int_var_from_proto_index(value.index)
            for value in alternative.trick_point_variables[:2]) < 2 * ace_points)
        yield "counterstrategy_captures_two_aces_raw", insufficient_raw

        model = cp_model.CpModel()
        # Abstract every possible ownership/declaration into a superset of
        # totals, using the same constraints as the full native project model.
        totals = tuple(model.new_int_var(0, 32767, f"awarded_{team}") for team in Team)
        winning = tuple(model.new_bool_var(f"winning_{team}") for team in Team)
        constrain_exclusive_project_awards_v1(model, totals, winning, model.new_bool_var("has_declaration"))
        model.add(totals[0] >= 20)
        model.add(totals[1] > 0)
        yield "opposing_project_awards_are_exclusive", model

        model = cp_model.CpModel()
        total = self.scoring.raw_card_totals["sun"]
        raw = tuple(model.new_int_var(0, total, f"raw_{team}") for team in Team)
        model.add(sum(raw) == total)
        model.add(raw[0] >= 2 * ace_points)
        winners = tuple(model.new_int_var(0, 1, f"winner_{i}") for i in range(8))
        model.add(sum(winners) <= 6)
        score = attach_scoring_fragment_v1(model, self.scoring, CpSatScoringInputsV1(
            model.new_constant(0), model.new_constant(int(Seat.NORTH)),
            model.new_int_var(0, 4, "escalation"), model.new_int_var(0, 2, "last_team"),
            raw, (model.new_int_var(20, 32767, "awarded_hundred_or_more"), model.new_constant(0)),
            (model.new_constant(0), model.new_constant(0)), winners,
            tuple(model.new_int_var(0, self.scoring.match_target - 1, f"match_{team}") for team in Team)),
            source_hash=self.rules.sha256, native_conversion=True)
        model.add(score.round_winner == int(Team.WEST_EAST))
        yield "awarded_hundred_and_two_aces_prevent_target_round_win", model

    def contradiction_receipt(self) -> FlagshipContradictionReceiptV1:
        """Certify the flagship contradiction with one named CP-SAT model."""

        model = cp_model.CpModel()
        assumptions = {
            name: model.new_bool_var(name)
            for name in (
                "contract_sun",
                "opponent_wins_trick_six",
                "two_opposing_aces_in_trick_eight",
                "target_forces_trick_seven",
            )
        }
        for literal in assumptions.values():
            model.add_assumption(literal)

        acting_seat = model.new_int_var(0, 3, "acting_seat")
        acting_team = model.new_int_var(0, 1, "acting_team")
        trick_six_winner = model.new_int_var(0, 3, "trick_six_winner")
        winner_team = model.new_int_var(0, 1, "winner_team")
        model.add_modulo_equality(acting_team, acting_seat, 2)
        model.add_modulo_equality(winner_team, trick_six_winner, 2)
        model.add(winner_team != acting_team).only_enforce_if(
            assumptions["opponent_wins_trick_six"]
        )

        ace_owners = tuple(
            model.new_int_var(0, 3, f"trick_eight_ace_owner_{index}")
            for index in range(2)
        )
        ace_owner_teams = tuple(
            model.new_int_var(0, 1, f"trick_eight_ace_owner_team_{index}")
            for index in range(2)
        )
        for owner, team in zip(ace_owners, ace_owner_teams, strict=True):
            model.add_modulo_equality(team, owner, 2)
            model.add(team != acting_team).only_enforce_if(
                assumptions["two_opposing_aces_in_trick_eight"]
            )
        model.add(ace_owners[0] != ace_owners[1]).only_enforce_if(
            assumptions["two_opposing_aces_in_trick_eight"]
        )

        leader_holds_ace = tuple(
            model.new_bool_var(f"trick_seven_leader_holds_ace_{index}")
            for index in range(2)
        )
        for owner, literal in zip(ace_owners, leader_holds_ace, strict=True):
            model.add(owner == trick_six_winner).only_enforce_if(literal)
            model.add(owner != trick_six_winner).only_enforce_if(literal.Not())
        model.add(sum(leader_holds_ace) >= 1).only_enforce_if(
            (
                assumptions["opponent_wins_trick_six"],
                assumptions["two_opposing_aces_in_trick_eight"],
            )
        )

        ace_suit = model.new_int_var(0, 3, "leader_ace_suit")
        ace_strength = model.new_int_var(0, 7, "leader_ace_strength")
        model.add_allowed_assignments(
            (ace_suit, ace_strength),
            [
                (
                    int(suit),
                    self.play.strength(Card.of(suit, Rank.ACE), Contract.SUN, None),
                )
                for suit in Suit
            ],
        ).only_enforce_if(assumptions["contract_sun"])
        low_rank = model.new_int_var(0, 2, "target_low_rank")
        low_strength = model.new_int_var(0, 7, "target_low_strength")
        model.add_allowed_assignments(
            (low_rank, low_strength),
            [
                (
                    index,
                    self.play.strength(
                        Card.of(Suit.SPADES, rank), Contract.SUN, None
                    ),
                )
                for index, rank in enumerate((Rank.SEVEN, Rank.EIGHT, Rank.NINE))
            ],
        ).only_enforce_if(assumptions["contract_sun"])
        ace_beats_low = model.new_bool_var("sun_ace_beats_target_low_card")
        model.add(ace_strength > low_strength).only_enforce_if(ace_beats_low)
        model.add(ace_strength <= low_strength).only_enforce_if(ace_beats_low.Not())
        model.add(ace_beats_low == 1).only_enforce_if(assumptions["contract_sun"])

        leader_has_any_ace = model.new_bool_var("trick_seven_leader_has_any_ace")
        model.add_max_equality(leader_has_any_ace, leader_holds_ace)
        counterexample = model.new_bool_var("opponent_ace_lead_counterexample")
        model.add_min_equality(counterexample, (leader_has_any_ace, ace_beats_low))
        model.add(counterexample == 0).only_enforce_if(
            assumptions["target_forces_trick_seven"]
        )

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        status = solver.solve(model)
        if status != cp_model.INFEASIBLE:
            raise AssertionError(
                "flagship contradiction model must be exhaustively infeasible"
            )
        assumption_by_index = {
            literal.index: name for name, literal in assumptions.items()
        }
        assumption_core = tuple(
            sorted(
                assumption_by_index[index]
                for index in sound_infeasible_assumption_indexes(
                    solver, assumption_by_index
                )
            )
        )
        model_hash = canonical_sha256(
            {"schemaVersion": 1, "cpModelProto": str(model.proto)}
        )
        checked = tuple(Seat)

        diagnostics: tuple[Mapping[str, Any], ...] = (
            {
                "diagnosticId": "ALT-TARGET-WINS-TRICK-6",
                "recipeDelta": {"play.trick_winner_matches[6].team": "target_team"},
                "effect": "The target leads trick seven, removing the opposing Ace-holder lead.",
                "autoApplied": False,
            },
            {
                "diagnosticId": "ALT-ONE-OPPONENT-ACE",
                "recipeDelta": {
                    "play.trick_card_count[8].ownerTeam": "opponent_team",
                    "play.trick_card_count[8].rank": "ace",
                    "play.trick_card_count[8].minimum": 1,
                    "play.trick_card_count[8].maximum": 1,
                },
                "effect": "One opposing seat need not retain an Ace for trick eight.",
                "autoApplied": False,
            },
            {
                "diagnosticId": "ALT-POLICY-CONDITIONED-NON-ACE-LEAD",
                "recipeDelta": {
                    "proof.mode": "policy_conditioned",
                    "proof.opponentPolicy": "does_not_lead_retained_ace_on_trick_7",
                },
                "effect": "This is conditional evidence, not an exact universal proof.",
                "autoApplied": False,
            },
        )
        core = {
            "schemaVersion": 1,
            "status": "exact_structural_contradiction",
            "theoremId": "FLAGSHIP-TWO-ACES-TRICK7-LEAD-V1",
            "rulesHash": self.rules.sha256,
            "playDomainHash": self.play.sha256,
            "checkedActingSeats": [seat.name.lower() for seat in checked],
            "diagnostics": [dict(item) for item in diagnostics],
            "solverStatus": solver.status_name(status).lower(),
            "modelHash": model_hash,
            "variableCount": len(model.proto.variables),
            "constraintCount": len(model.proto.constraints),
            "assumptionCore": list(assumption_core),
        }
        return FlagshipContradictionReceiptV1(
            core["status"],
            core["theoremId"],
            checked,
            diagnostics,
            core["solverStatus"],
            model_hash,
            core["variableCount"],
            core["constraintCount"],
            assumption_core,
            canonical_sha256(core),
        )

    def contradiction_core_for_scenario(
        self,
        scenario: ResearchScenarioV1,
        proof_request: ScenarioProofRequestV1,
    ) -> tuple[str, ...] | None:
        """Recognize the historical closed form, only for frozen V2 execution.

        The theorem refutes a *forced* trick-seven win, not a witnessed line
        paired with exact round-win proof. V3 must never use this recognizer to
        declare a scenario infeasible. Only predicates required through ``all``
        nodes participate in historical recognition.
        """

        if not isinstance(scenario, ResearchScenarioV1):
            raise TypeError("flagship contradiction recognition requires a scenario")
        if not isinstance(proof_request, ScenarioProofRequestV1):
            raise TypeError("flagship contradiction recognition requires a proof")
        if (
            scenario.origin.kind != "search"
            or scenario.decision_point.get("kind") != "play"
            or proof_request.mode != "exact"
            or proof_request.visibility != "complete_state"
            or proof_request.utility != "round_win"
        ):
            return None
        selector = dict(scenario.decision_point.get("selector") or {})
        acting_seat_raw = selector.get("actingSeat")
        try:
            acting_seat = Seat.parse(acting_seat_raw)
        except (KeyError, TypeError, ValueError):
            return None
        target_team = acting_seat.team.name.lower()
        opponent_team = acting_seat.team.other.name.lower()
        if (
            selector.get("trickIndex") != 5
            or selector.get("position") != 3
            or selector.get("remainingCards") != 3
            or proof_request.target_team != target_team
        ):
            return None

        required = self._required_hard_predicates(scenario.constraints)

        def find(kind: str, predicate_match) -> ScenarioPredicateV1 | None:
            return next(
                (
                    predicate
                    for predicate in required
                    if predicate.kind == kind
                    and predicate_match(dict(predicate.arguments))
                ),
                None,
            )

        def winner_is_on_team(arguments: Mapping[str, Any], team: str) -> bool:
            if arguments.get("team") is not None:
                return arguments["team"] == team
            try:
                return Seat.parse(arguments.get("seat")).team.name.lower() == team
            except (KeyError, TypeError, ValueError):
                return False

        contract = find(
            "auction.contract", lambda arguments: arguments.get("contract") == "sun"
        )
        root_ace = find(
            "play.action_matches",
            lambda arguments: (
                arguments.get("trickIndex") == 5
                and arguments.get("position") == 3
                and arguments.get("actingSeat") == acting_seat.name.lower()
                and "ace" in dict(arguments.get("selector") or {}).get("ranks", ())
            ),
        )
        opponent_wins_six = find(
            "play.trick_winner_matches",
            lambda arguments: (
                arguments.get("trickIndex") == 5
                and winner_is_on_team(arguments, opponent_team)
            ),
        )
        target_wins_seven = find(
            "play.trick_winner_matches",
            lambda arguments: (
                arguments.get("trickIndex") == 6
                and winner_is_on_team(arguments, target_team)
            ),
        )
        two_opponent_aces_eight = find(
            "play.trick_card_count",
            lambda arguments: (
                arguments.get("trickIndex") == 7
                and arguments.get("minimum") == 2
                and arguments.get("maximum") == 2
                and "ace" in dict(arguments.get("selector") or {}).get("ranks", ())
                and dict(dict(arguments.get("selector") or {}).get("owner") or {}).get(
                    "team"
                )
                == opponent_team
            ),
        )
        unique_root = find(
            "proof.winning_root_action_count",
            lambda arguments: (
                arguments.get("minimum") == 1 and arguments.get("maximum") == 1
            ),
        )
        matched = (
            contract,
            root_ace,
            opponent_wins_six,
            target_wins_seven,
            two_opponent_aces_eight,
            unique_root,
        )
        if any(predicate is None for predicate in matched):
            return None
        return tuple(
            predicate.predicate_id for predicate in matched if predicate is not None
        )

    @classmethod
    def _required_hard_predicates(
        cls, expression: ScenarioExpressionV1
    ) -> tuple[ScenarioPredicateV1, ...]:
        if expression.predicate is not None:
            return (
                (expression.predicate,)
                if expression.predicate.hardness == "hard"
                else ()
            )
        if expression.op != "all":
            return ()
        return tuple(
            predicate
            for child in expression.children
            for predicate in cls._required_hard_predicates(child)
        )

    def _lift(
        self,
        tail: FlagshipTailCandidateV1,
        *,
        maximum_seconds: float,
    ) -> tuple[str, FlagshipLiftV1 | None]:
        dealer = self.structural.offset(tail.buyer, -1)
        model = cp_model.CpModel()
        owner_of_card = tuple(
            model.new_int_var(0, 3, f"flagship_owner_{card.id}") for card in FULL_DECK
        )
        project_fragment = attach_project_lifecycle_fragment_v1(
            model,
            owner_of_card=owner_of_card,
            contract=Contract.SUN,
            dealer=dealer,
            trump=None,
            domain=self.project,
            name="flagship_projects",
        )
        buyer_variable = model.new_int_var(
            int(tail.buyer), int(tail.buyer), "flagship_buyer"
        )
        project_fragment.constrain_buyer_team_project(
            model,
            buyer_seat=buyer_variable,
            states="awarded",
            value=self.project.sequence_value_for(Contract.SUN, "five_plus"),
            minimum=1,
            constraint_id="flagship_awarded_hundred",
        )
        first_leader = self.play.resolve_first_leader(
            dealer=dealer,
            buyer=tail.buyer,
            structural_domain=self.structural,
        )
        solver = cp_model.CpSolver()
        try:
            play_result = OrToolsPlayTailStageV1().solve(
                cards=FULL_DECK,
                ownership={play.card: play.actor for play in tail.plays},
                contract=Contract.SUN,
                trump=None,
                first_leader=first_leader,
                play_domain=self.play,
                structural_domain=self.structural,
                scoring_domain=self.scoring,
                tail=tuple(
                    TailPrefixActionV1(play.actor, play.card) for play in tail.plays
                ),
                maximum_time_seconds=maximum_seconds,
                shared_model=model,
                shared_owner_of_card=owner_of_card,
                solver=solver,
            )
        except ValueError as error:
            if str(error).endswith(": infeasible"):
                return "lift_infeasible", None
            return "lift_incomplete", None

        hands = play_result.initial_hands
        layout = DealLayoutV1.build(
            self.structural,
            dealer=dealer,
            buyer=tail.buyer,
        )
        deck_values = [-1] * 32
        for seat in Seat:
            positions = layout.positions_for(seat, final=True)
            for position, card in zip(positions, hands[int(seat)], strict=True):
                deck_values[position] = card.id
        if sorted(deck_values) != list(range(32)):
            raise AssertionError("canonical lift deck did not preserve all cards")
        deck = tuple(deck_values)

        auction_model = OrToolsScenarioModelV1(
            rules=self.rules,
            dealer=dealer,
            required_buyer=tail.buyer,
        )
        for position, card_id in enumerate(deck):
            auction_model.model.add(auction_model.card_at_position[position] == card_id)
        auction = auction_model.add_auction_fragment(
            required_actions=[
                {"roundIndex": 1, "actor": tail.buyer.name.lower(), "action": "sun"}
            ],
            required_actions_complete=True,
        )
        auction_solution = auction_model.solve(maximum_seconds=maximum_seconds)
        if auction_solution.status != "satisfied" or auction_solution.auction is None:
            raise AssertionError(
                "canonical lift deck failed exact Sun auction normalization"
            )
        model_hash = canonical_sha256(
            {
                "schemaVersion": 1,
                "kind": "flagship_prefix_lift_v1",
                "playModelHash": play_result.model_hash,
                "playEvidenceHash": play_result.evidence_hash,
                "projectFragmentHash": project_fragment.fragment_hash,
                "auctionModelHash": auction_solution.model_hash,
                "auctionDomainHash": auction.domain_hash,
            }
        )
        project_result = project_fragment.extract(solver)
        awarded_hundred = any(
            project.awarded
            and project.team == tail.buyer.team
            and project.value
            == self.project.sequence_value_for(Contract.SUN, "five_plus")
            for project in project_result.projects
        )
        if not awarded_hundred:
            raise AssertionError(
                "integrated project constraints disagreed with exact CP project stage"
            )

        plays = tuple(
            FlagshipCompletePlayV1(step // 4 + 1, step % 4 + 1, play.actor, play.card)
            for step, play in enumerate(play_result.plays)
        )
        winners = play_result.trick_winners
        prefix_raw = [0, 0]
        for winner, points in zip(
            winners[:5], play_result.trick_raw_points[:5], strict=True
        ):
            prefix_raw[int(winner.team)] += points
        prefix_raw_points = tuple(prefix_raw)
        lift_status = (
            "lift_complete"
            if prefix_raw_points == tail.prefix_raw_points
            else "lift_target_miss"
        )
        auction_evidence = auction_solution.auction
        core = {
            "schemaVersion": 1,
            "status": lift_status,
            "tailCandidateHash": tail.candidate_hash,
            "deckOrder": list(deck),
            "hands": [[card.id for card in hand] for hand in hands],
            "auction": auction_evidence.to_dict(),
            "projectEvidenceHash": project_result.evidence_hash,
            "plays": [play.to_dict() for play in plays],
            "trickWinners": [seat.name.lower() for seat in winners],
            "prefixRawPoints": list(prefix_raw_points),
            "constructionModelHash": model_hash,
        }
        lift_hash = canonical_sha256(core)
        return (
            lift_status,
            FlagshipLiftV1(
                lift_status,
                tail,
                dealer,
                deck,
                hands,
                tuple(action.to_dict() for action in auction_evidence.actions),
                project_result,
                tuple(plays),
                tuple(winners),
                prefix_raw_points,
                model_hash,
                lift_hash,
                int(solver.num_branches),
                int(solver.num_conflicts),
            ),
        )

    def _terminal_utility(
        self,
        *,
        raw_points: tuple[int, int],
        project_points: tuple[int, int],
        target_team: Team,
        buyer_team: Team,
    ) -> int:
        counting_game = self.scoring.convert_raw(
            raw_points[int(target_team)], Contract.SUN
        )
        card_game = [0, 0]
        card_game[int(target_team)] = counting_game
        card_game[int(buyer_team)] = (
            self.scoring.card_game_point_totals[Contract.SUN.value] - counting_game
        )
        totals = tuple(card_game[index] + project_points[index] for index in range(2))
        if totals[0] > totals[1]:
            winner = Team.NORTH_SOUTH
        elif totals[1] > totals[0]:
            winner = Team.WEST_EAST
        else:
            winner = self.scoring.tie_winner(
                Contract.SUN,
                raw_points[int(target_team)],
                target_team,
                buyer_team,
            )
        return 1 if winner == target_team else -1

    def _build_proof_graph(self, lift: FlagshipLiftV1) -> ProofGraphV1:
        tail = lift.tail
        first_tail = tuple(play for play in lift.plays if play.trick_index == 6)
        if len(first_tail) != 4 or first_tail[0].actor != tail.buyer:
            raise AssertionError("lifted tail does not start with the opposing buyer")
        hands = [0, 0, 0, 0]
        for play in tail.plays:
            hands[int(play.actor)] |= 1 << play.card.id
        lead = first_tail[0]
        hands[int(lead.actor)] ^= 1 << lead.card.id
        initial = _ProofStateV1(
            tuple(hands),
            ((int(lead.actor), lead.card.id),),
            self.structural.offset(lead.actor, 1),
            lift.prefix_raw_points,
            5,
        )
        projects = lift.projects.combined_awarded_value
        nodes: dict[str, ProofGraphNodeV1] = {}
        identities: dict[_ProofStateV1, str] = {}

        def identity(state: _ProofStateV1) -> str:
            existing = identities.get(state)
            if existing is not None:
                return existing
            digest = canonical_sha256(
                {
                    "hands": list(state.hands),
                    "currentTrick": [list(item) for item in state.current_trick],
                    "nextActor": state.next_actor.name.lower(),
                    "rawPoints": list(state.raw_points),
                    "completedTricks": state.completed_tricks,
                }
            )
            value = f"NODE-{digest[:24].upper()}"
            identities[state] = value
            return value

        def legal_actions(state: _ProofStateV1) -> tuple[Card, ...]:
            mask = state.hands[int(state.next_actor)]
            cards = tuple(
                Card(card_id) for card_id in range(32) if mask & (1 << card_id)
            )
            if not state.current_trick:
                return cards
            led_suit = Card(state.current_trick[0][1]).suit
            following = tuple(card for card in cards if card.suit == led_suit)
            return following or cards

        def advance(state: _ProofStateV1, card: Card) -> _ProofStateV1:
            actor = state.next_actor
            hands_value = list(state.hands)
            hands_value[int(actor)] ^= 1 << card.id
            current = (*state.current_trick, (int(actor), card.id))
            if len(current) < 4:
                return _ProofStateV1(
                    tuple(hands_value),
                    current,
                    self.structural.offset(actor, 1),
                    state.raw_points,
                    state.completed_tricks,
                )
            led_suit = Card(current[0][1]).suit
            winner_actor, _winner_card = max(
                (
                    (Seat(seat), Card(card_id))
                    for seat, card_id in current
                    if Card(card_id).suit == led_suit
                ),
                key=lambda item: self.play.strength(item[1], Contract.SUN, None),
            )
            raw = list(state.raw_points)
            raw[int(winner_actor.team)] += sum(
                self.play.points(Card(card_id), Contract.SUN, None)
                for _seat, card_id in current
            )
            completed = state.completed_tricks + 1
            if completed == 8:
                raw[int(winner_actor.team)] += self.scoring.last_trick_bonus_raw
            return _ProofStateV1(
                tuple(hands_value),
                (),
                winner_actor,
                tuple(raw),
                completed,
            )

        def build(state: _ProofStateV1) -> str:
            node_id = identity(state)
            if node_id in nodes:
                return node_id
            if state.completed_tricks == 8:
                if any(state.hands) or state.current_trick:
                    raise AssertionError("terminal proof state retained cards")
                nodes[node_id] = ProofGraphNodeV1.leaf(
                    node_id,
                    self._terminal_utility(
                        raw_points=state.raw_points,
                        project_points=projects,
                        target_team=tail.target_team,
                        buyer_team=tail.buyer.team,
                    ),
                )
                return node_id
            edges = tuple(
                ProofGraphEdgeV1.build(card.id, card.code, build(advance(state, card)))
                for card in legal_actions(state)
            )
            nodes[node_id] = ProofGraphNodeV1.decision(
                node_id,
                state.next_actor.team.name.lower(),
                edges,
                actor_seat=state.next_actor.name.lower(),
            )
            return node_id

        root = build(initial)
        return ProofGraphV1.build(
            scope="exact", root_node_id=root, nodes=tuple(nodes.values())
        )

    def _prove(
        self,
        lift: FlagshipLiftV1,
        *,
        maximum_seconds: float,
    ) -> tuple[ProofGraphV1, OrToolsGraphProofOutcomeV1]:
        graph = self._build_proof_graph(lift)
        request = ScenarioProofRequestV1.build(
            {
                "schemaVersion": 1,
                "mode": "exact",
                "visibility": "complete_state",
                "claim": "selected_action_is_unique_winner",
                "targetTeam": lift.tail.target_team.name.lower(),
                "utility": "round_win",
                "successThreshold": 1,
                "selectedAction": lift.tail.root_action.code,
                "budgetProfile": "late_game_exact_v1",
            }
        )
        proof = OrToolsScenarioGraphProofV1().prove(
            graph,
            request,
            maximum_seconds=maximum_seconds,
        )
        return graph, proof

    def _result(
        self,
        *,
        status: str,
        tail: FlagshipTailCandidateV1 | None,
        lift: FlagshipLiftV1 | None,
        graph: ProofGraphV1 | None,
        proof: OrToolsGraphProofOutcomeV1 | None,
        rejected: Sequence[FlagshipTailCandidateV1],
        stage_timings: Mapping[str, float],
    ) -> FlagshipResearchResultV1:
        contradiction = self.contradiction_receipt()
        last_rejected = rejected[-1] if rejected else None
        rejected_no_goods = tuple(item.exact_no_good for item in rejected)
        rejected_hashes = tuple(_tail_no_good_hash(item) for item in rejected)
        core = {
            "schemaVersion": 1,
            "status": status,
            "tailCandidateHash": tail.candidate_hash if tail is not None else None,
            "liftHash": lift.lift_hash if lift is not None else None,
            "graphHash": graph.sha256 if graph is not None else None,
            "proofHash": proof.sha256 if proof is not None else None,
            "contradictionHash": contradiction.sha256,
            "rejectedTailHashes": list(rejected_hashes),
        }
        return FlagshipResearchResultV1(
            status,
            tail,
            lift,
            graph,
            proof,
            contradiction,
            last_rejected.exact_no_good if last_rejected is not None else None,
            _tail_no_good_hash(last_rejected) if last_rejected is not None else None,
            rejected_no_goods,
            rejected_hashes,
            dict(stage_timings),
            canonical_sha256(core),
        )

    def search_one(
        self,
        *,
        maximum_tail_attempts: int = 1,
        tail_seconds: float = 5.0,
        lift_seconds: float = 10.0,
        proof_seconds: float = 5.0,
        excluded_exact_tails: Sequence[Sequence[int]] = (),
        excluded_suit_symmetry_keys: Sequence[str] = (),
    ) -> FlagshipResearchResultV1:
        if maximum_tail_attempts != 1:
            raise ValueError(
                "the structurally contradictory flagship permits one evidence solve only"
            )
        started = time.perf_counter()
        timings: dict[str, float] = {}
        tail_started = time.perf_counter()
        tail_result = OrToolsFlagshipTailSearchV1(
            rules=self.rules,
            acting_seat=self.acting_seat,
            excluded_exact_tails=excluded_exact_tails,
            excluded_suit_symmetry_keys=excluded_suit_symmetry_keys,
        ).solve(maximum_seconds=tail_seconds)
        timings["tailConstructionSeconds"] = time.perf_counter() - tail_started
        tail = tail_result.candidate
        if tail is None:
            timings["totalSeconds"] = time.perf_counter() - started
            return self._result(
                status=tail_result.status,
                tail=None,
                lift=None,
                graph=None,
                proof=None,
                rejected=(),
                stage_timings=timings,
            )

        lift_started = time.perf_counter()
        lift_status, lift = self._lift(tail, maximum_seconds=lift_seconds)
        timings["prefixLiftSeconds"] = time.perf_counter() - lift_started
        if lift is None:
            timings["totalSeconds"] = time.perf_counter() - started
            return self._result(
                status=lift_status,
                tail=tail,
                lift=None,
                graph=None,
                proof=None,
                rejected=(tail,),
                stage_timings=timings,
            )

        proof_started = time.perf_counter()
        graph, proof = self._prove(lift, maximum_seconds=proof_seconds)
        timings["exactProofSeconds"] = time.perf_counter() - proof_started
        timings["totalSeconds"] = time.perf_counter() - started
        exact_zero = (
            proof.completion_status == "proof_failed"
            and not proof.claim_satisfied
            and proof.winning_root_action_count == 0
            and tail.root_action.rank == Rank.ACE
        )
        status = (
            "proof_failed"
            if exact_zero
            else "proof_incomplete"
            if proof.completion_status == "proof_incomplete"
            else "model_disagreement"
        )
        return self._result(
            status=status,
            tail=tail,
            lift=lift,
            graph=graph,
            proof=proof,
            rejected=(tail,),
            stage_timings=timings,
        )

    def search_variants(
        self,
        *,
        count: int,
        maximum_tail_attempts_per_variant: int = 1,
    ) -> tuple[FlagshipResearchResultV1, ...]:
        if type(count) is not int or not 1 <= count <= 5:
            raise ValueError(
                "qualified flagship variant count must be from one through five"
            )
        if maximum_tail_attempts_per_variant != 1:
            raise ValueError(
                "the structurally contradictory flagship permits one evidence solve only"
            )
        raise ValueError(
            "FLAGSHIP-TWO-ACES-TRICK7-LEAD-V1 prevents qualified variants; "
            "review diagnostics without auto-applying them"
        )


__all__ = [
    "FlagshipCompletePlayV1",
    "FlagshipLiftV1",
    "FlagshipContradictionReceiptV1",
    "FlagshipResearchResultV1",
    "OrToolsFlagshipResearchV1",
]
