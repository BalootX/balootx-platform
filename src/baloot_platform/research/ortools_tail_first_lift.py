"""Lift a proven late-play boundary to one complete CP-SAT round."""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Mapping, Sequence

from ortools.sat.python import cp_model

from baloot_platform.core.cards import FULL_DECK, Card
from baloot_platform.core.enums import Contract, Seat, Suit, Team
from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1

from .ortools_complete_state_graph import CompleteStateGraphSeedV1
from .ortools_escalation_stage import OrToolsEscalationStageV1
from .ortools_information_set_proof import _solve_cp_sat_with_cancel_v1
from .ortools_play_tail import (
    OrToolsPlayTailResultV1,
    OrToolsPlayTailStageV1,
    PlayZeroObservationActionNoGoodV1,
    PlayTailSolveStopped,
    apply_play_zero_observation_action_no_good_v1,
)
from .ortools_project_fragment import (
    attach_project_lifecycle_fragment_v1,
    project_lifecycle_result_v1,
)
from .ortools_project_model import (
    OrToolsProjectLifecycleModelV1,
    OrToolsProjectLifecycleResultV1,
)
from .ortools_research_executor import (
    OrToolsCandidateAttemptV1,
    OrToolsNeutralCandidateV1,
    _PlayQualificationContextV1,
    _cp_sat_receipt_metadata_v1,
    _enforce_hard_stage_projection,
    _hard_expression_requires_predicate,
    _hash_document,
    _project_exists_witness_keys,
    _score_complete_candidate,
    _stage_receipt,
)
from .ortools_scenario_model import OrToolsScenarioModelV1
from .scenario_contracts import ResearchScenarioV1, canonical_sha256
from .scenario_deal_predicates_v2 import DealPredicateModelContextV2
from .scenario_predicate_auction_escalation_v2 import (
    CpSatAuctionPredicateContextV1,
    CpSatEscalationPredicateContextV1,
)
from .scenario_predicate_manifest_v2 import (
    ScenarioPredicateCatalogLoweringManifestV2,
    shared_scenario_predicate_catalog_lowering_manifest_v2,
)
from .scenario_adjudication_predicates_v3 import (
    shared_scenario_predicate_catalog_lowering_manifest_v3,
)
from .scenario_predicate_lowering_v2 import (
    CpSatActionVariablesV1,
    CpSatPredicateModelContextV1,
    CpSatTrickVariablesV1,
)
from .scenario_project_outcome_lowering_v2 import CpSatProjectPredicateContextV1
from .tail_first_stage_planner import TailFirstStagePlanV1, TailFirstStagePlannerV1


_HASH_ZERO = "0" * 64


def _plan_hash(plan: TailFirstStagePlanV1) -> str:
    payload = plan.to_dict()
    payload.pop("sha256")
    return canonical_sha256(payload)


def _tail_evidence_hash(tail: OrToolsPlayTailResultV1) -> str:
    return canonical_sha256(
        {
            "schemaVersion": 1,
            "initialHands": [
                [card.code for card in hand] for hand in tail.initial_hands
            ],
            "plays": [play.to_dict() for play in tail.plays],
            "legalActionsByStep": [
                [card.code for card in actions]
                for actions in tail.legal_actions_by_step
            ],
            "legalPlayActionsByStep": [
                [action.to_dict() for action in actions]
                for actions in tail.legal_play_actions_by_step
            ],
            "trickWinners": [seat.name.lower() for seat in tail.trick_winners],
            "trickWinningCards": [card.code for card in tail.trick_winning_cards],
            "trickRawPoints": list(tail.trick_raw_points),
            "prefixLength": tail.prefix_length,
            "remainingHands": [
                [card.code for card in hand] for hand in tail.remaining_hands
            ],
            "currentTrick": [play.to_dict() for play in tail.current_trick],
            "nextActor": (
                tail.next_actor.name.lower() if tail.next_actor is not None else None
            ),
            "legalActions": [card.code for card in tail.legal_actions],
            "legalPlayActions": [
                action.to_dict() for action in tail.legal_play_actions
            ],
            "rawTeamPoints": (
                list(tail.raw_team_points) if tail.raw_team_points is not None else None
            ),
            "cardGamePoints": (
                list(tail.card_game_points)
                if tail.card_game_points is not None
                else None
            ),
            "scoringDomainHash": tail.scoring_domain_hash,
        }
    )


def _preserve_tail_cards_before_decision(
    build: Any,
    tail: OrToolsPlayTailResultV1,
    tail_offset: int,
    cancel_check: Callable[[], bool] | None = None,
) -> bool:
    """Keep every recorded late card available until the tail begins."""
    tail_card_ids = {play.card.id for play in tail.plays}
    for selected in build.selected_play_indexes[:tail_offset]:
        if cancel_check is not None and cancel_check():
            return False
        for card_id in tail_card_ids:
            build.model.add(selected != card_id)
    return True


def _hard_predicates(
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


def _bind_stage(
    scenario: ResearchScenarioV1,
    manifest: ScenarioPredicateCatalogLoweringManifestV2,
    stage: str,
    context: Any,
) -> Mapping[str, Any]:
    predicates = _hard_predicates(scenario, manifest, stage)
    if not predicates:
        return {}
    lowerings = {
        predicate.predicate_id: manifest.lower_predicate(predicate, context)
        for predicate in predicates
    }
    literals = {
        predicate_id: lowering.literal
        for predicate_id, lowering in lowerings.items()
    }
    _enforce_hard_stage_projection(
        context.model,
        scenario,
        stage=stage,
        predicate_literals=literals,
    )
    return lowerings


def _lift_play_predicate_context(
    model: cp_model.CpModel,
    context: CpSatPredicateModelContextV1,
    tail: OrToolsPlayTailResultV1,
) -> CpSatPredicateModelContextV1:
    """Expose the lifted prefix and verified tail as logical round tricks 0..7."""

    offset = min(context.tricks)
    tricks: dict[int, CpSatTrickVariablesV1] = {}
    for absolute_index, trick in sorted(context.tricks.items()):
        logical_index = absolute_index - offset
        actions = tuple(
            CpSatActionVariablesV1(
                f"PLAY-{logical_index * 4 + position:02d}",
                action.card,
                action.actor,
                action.original_owner,
                action.ekka,
                action.present,
                action.winning,
                action.value,
            )
            for position, action in enumerate(trick.actions)
        )
        tricks[logical_index] = CpSatTrickVariablesV1(
            logical_index,
            actions,
            CpSatActionVariablesV1(
                f"WINNER-{logical_index}",
                trick.winner.card,
                trick.winner.actor,
                trick.winner.original_owner,
                trick.winner.ekka,
                trick.winner.present,
                trick.winner.winning,
                trick.winner.value,
            ),
        )

    overlap = len(tricks) - 1
    for tail_index in range(1, len(tail.trick_winners)):
        logical_index = overlap + tail_index
        plays = tail.plays[tail_index * 4 : tail_index * 4 + 4]
        actions = tuple(
            CpSatActionVariablesV1(
                f"PLAY-{logical_index * 4 + position:02d}",
                model.new_constant(play.card.id),
                model.new_constant(int(play.actor)),
                model.new_constant(int(play.actor)),
                model.new_constant(int(play.ekka)),
                model.new_constant(1),
            )
            for position, play in enumerate(plays)
        )
        winner = tail.trick_winners[tail_index]
        winning_card = tail.trick_winning_cards[tail_index]
        tricks[logical_index] = CpSatTrickVariablesV1(
            logical_index,
            actions,
            CpSatActionVariablesV1(
                f"WINNER-{logical_index}",
                model.new_constant(winning_card.id),
                model.new_constant(int(winner)),
                model.new_constant(int(winner)),
                model.new_constant(0),
                model.new_constant(1),
                model.new_constant(1),
            ),
        )
    if set(tricks) != set(range(8)):
        raise AssertionError("tail lift predicate context must cover all eight tricks")
    return CpSatPredicateModelContextV1(
        model,
        f"{context.context_id}-logical-round",
        tricks,
        (),
        model.new_constant(1),
    )


def _seed_clone(
    seed: CompleteStateGraphSeedV1, **changes: Any
) -> CompleteStateGraphSeedV1:
    values = {
        name: getattr(seed, name)
        for name in CompleteStateGraphSeedV1.__dataclass_fields__
        if name != "proof_state_projection_hash"
    }
    values.update(changes)
    return CompleteStateGraphSeedV1.build(**values)


def _base_points(
    play: OrToolsPlayTailResultV1, prefix_length: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    raw = [0, 0]
    tricks = [0, 0]
    for winner, points in zip(
        play.trick_winners[: prefix_length // 4],
        play.trick_raw_points[: prefix_length // 4],
        strict=True,
    ):
        raw[int(winner.team)] += int(points)
        tricks[int(winner.team)] += 1
    return (raw[0], raw[1]), (tricks[0], tricks[1])


def _bind_proof_seed_prefix(
    play_build: Any,
    proof_seed: CompleteStateGraphSeedV1,
    completed_tricks: int,
) -> None:
    """Keep the cheap reachability probe and full lift on the same proof state."""
    model = play_build.model
    context = play_build.play_state_context
    model.add(context.next_actor == int(proof_seed.next_actor))
    model.add(context.current_trick_size == len(proof_seed.current_trick))
    for team in Team:
        model.add(context.raw_points[team] == proof_seed.base_raw[int(team)])
        winners = []
        for trick in range(completed_tricks):
            won = model.new_bool_var(
                f"tail_lift_base_trick_{trick}_team_{int(team)}"
            )
            actor = play_build.predicate_context.tricks[trick].winner.actor
            model.add_allowed_assignments(
                [actor, won],
                [(int(seat), int(seat.team == team)) for seat in Seat],
            )
            winners.append(won)
        model.add(sum(winners) == proof_seed.base_tricks[int(team)])
    expected_actions = {
        (action.card.id, action.ekka) for action in proof_seed.legal_actions
    }
    for key, literal in context.legal_action_literals.items():
        model.add(literal == int(key in expected_actions))


def _compose_prefix_and_verified_tail(
    prefix: OrToolsPlayTailResultV1,
    tail: OrToolsPlayTailResultV1,
    *,
    tail_offset: int,
    contract: Contract,
    scoring: ScoringDomainIRV1,
    solved_card_game_points: tuple[int, int] | None = None,
) -> OrToolsPlayTailResultV1:
    """Compose one reached prefix with the already-solved legal tactical tail."""

    if len(prefix.plays) != tail_offset + 4:
        raise ValueError("prefix lift must finish the tactical root trick")
    if tuple(prefix.plays[tail_offset:]) != tuple(tail.plays[:4]):
        raise ValueError("prefix lift and verified tail root trick disagree")
    plays = tuple(prefix.plays[:tail_offset]) + tuple(tail.plays)
    winners = tuple(prefix.trick_winners[: tail_offset // 4]) + tuple(
        tail.trick_winners
    )
    winning_cards = tuple(prefix.trick_winning_cards[: tail_offset // 4]) + tuple(
        tail.trick_winning_cards
    )
    trick_points = tuple(prefix.trick_raw_points[: tail_offset // 4]) + tuple(
        tail.trick_raw_points
    )
    legal_cards = tuple(prefix.legal_actions_by_step[:tail_offset]) + tuple(
        tail.legal_actions_by_step
    )
    legal_plays = tuple(prefix.legal_play_actions_by_step[:tail_offset]) + tuple(
        tail.legal_play_actions_by_step
    )
    if not (
        len(plays) == 32
        and len(winners) == 8
        and len(winning_cards) == 8
        and len(trick_points) == 8
        and len(legal_cards) == 32
        and len(legal_plays) == 32
    ):
        raise AssertionError("composed prefix/tail does not form one complete round")
    raw = [0, 0]
    for winner, points in zip(winners, trick_points, strict=True):
        raw[int(winner.team)] += int(points)
    raw[int(winners[-1].team)] += scoring.last_trick_bonus_raw
    raw_points = (raw[0], raw[1])
    card_game_points = solved_card_game_points if solved_card_game_points is not None else tuple(
        scoring.convert_raw(raw_points[int(team)], contract) for team in Team
    )
    model_hash = canonical_sha256(
        {
            "schemaVersion": 1,
            "kind": "prefix_plus_verified_tail",
            "prefixModelHash": prefix.model_hash,
            "tailModelHash": tail.model_hash,
            "tailEvidenceHash": tail.evidence_hash,
        }
    )
    evidence_hash = canonical_sha256(
        {
            "schemaVersion": 1,
            "initialHands": [
                [card.code for card in hand] for hand in prefix.initial_hands
            ],
            "plays": [play.to_dict() for play in plays],
            "trickWinners": [winner.name.lower() for winner in winners],
            "trickRawPoints": list(trick_points),
            "rawTeamPoints": list(raw_points),
            "cardGamePoints": list(card_game_points),
        }
    )
    solver_status = (
        "optimal"
        if prefix.solver_status == "optimal" and tail.solver_status == "optimal"
        else "feasible"
    )
    semantic_hash = canonical_sha256(
        {
            "schemaVersion": 1,
            "modelHash": model_hash,
            "evidenceHash": evidence_hash,
            "solverStatus": solver_status,
        }
    )
    empty_hands = tuple(() for _ in Seat)
    return OrToolsPlayTailResultV1(
        prefix.initial_hands,
        plays,
        winners,
        winning_cards,
        trick_points,
        32,
        empty_hands,
        (),
        None,
        (),
        (),
        legal_cards,
        legal_plays,
        raw_points,
        card_game_points,
        scoring.sha256,
        model_hash,
        evidence_hash,
        solver_status,
        semantic_hash,
    )


class OrToolsTailFirstBackwardLiftV1:
    """Fix the tail/proof boundary and solve only its legal round prefix."""

    def __init__(
        self, *, rules: RuleBook, native_cp_sat_scoring: bool = False,
    ) -> None:
        if not isinstance(rules, RuleBook):
            raise TypeError("tail-first lift requires RuleBook")
        self.rules = rules
        self.native_cp_sat_scoring = bool(native_cp_sat_scoring)
        self.manifest = (
            shared_scenario_predicate_catalog_lowering_manifest_v3()
            if self.native_cp_sat_scoring
            else shared_scenario_predicate_catalog_lowering_manifest_v2()
        )

    def lift(
        self,
        scenario: ResearchScenarioV1,
        plan: TailFirstStagePlanV1,
        tail: OrToolsPlayTailResultV1,
        proof_seed: CompleteStateGraphSeedV1,
        *,
        play_hint: OrToolsPlayTailResultV1 | None = None,
        certified_projects: OrToolsProjectLifecycleResultV1 | None = None,
        excluded_decks: Sequence[Sequence[int]] = (),
        excluded_complete_plays: Sequence[Sequence[int]] = (),
        excluded_play_zero_actions: Sequence[
            PlayZeroObservationActionNoGoodV1
        ] = (),
        policy_prefix_steps: int = 0,
        policy_id: str = "observer_safe_trick_control_v1",
        require_optimal_auction_actions: bool = False,
        seed: int = 0,
        maximum_seconds: float = 15.0,
        maximum_conflicts: int = 250_000,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> OrToolsCandidateAttemptV1:
        if type(seed) is not int:
            raise TypeError("tail-first lift seed must be an integer")
        if (
            type(maximum_seconds) not in {int, float}
            or isinstance(maximum_seconds, bool)
            or maximum_seconds <= 0
        ):
            raise ValueError("tail-first lift maximum_seconds must be positive")
        if type(maximum_conflicts) is not int or maximum_conflicts < 1:
            raise ValueError("tail-first lift maximum_conflicts must be positive")
        if type(require_optimal_auction_actions) is not bool:
            raise TypeError("tail-first lift auction-quality switch must be boolean")
        if (
            type(policy_prefix_steps) is not int
            or policy_prefix_steps < 0
            or policy_prefix_steps > (plan.prefix_length or 0)
        ):
            raise ValueError("tail-first lift policy prefix must cover opening plays")
        if play_hint is not None and (
            not isinstance(play_hint, OrToolsPlayTailResultV1)
            or len(play_hint.plays) != 32
            or len(play_hint.trick_winners) != 8
        ):
            raise ValueError("tail-first lift play hint must be one complete round")
        if certified_projects is not None and not isinstance(
            certified_projects, OrToolsProjectLifecycleResultV1
        ):
            raise TypeError("certified projects must use the CP-SAT project result")
        if certified_projects is not None and (
            certified_projects.combined_awarded_value
            != proof_seed.awarded_project_points
            or certified_projects.baloot_points != proof_seed.baloot_points
        ):
            raise ValueError("certified projects disagree with the proof boundary")
        self._validate_inputs(scenario, plan, tail, proof_seed)
        normalized_play_exclusions = tuple(
            tuple(int(card) for card in play) for play in excluded_complete_plays
        )
        if any(
            len(play) != 32 or sorted(play) != list(range(32))
            for play in normalized_play_exclusions
        ):
            raise ValueError("excluded complete plays must be 32-card permutations")
        if any(
            not isinstance(item, PlayZeroObservationActionNoGoodV1)
            for item in excluded_play_zero_actions
        ):
            raise TypeError(
                "excluded play-zero actions must use the governed no-good type"
            )
        tail_cards = tuple(play.card.id for play in tail.plays)
        matching_play_exclusions = tuple(
            play
            for play in dict.fromkeys(normalized_play_exclusions)
            if proof_seed.contract.effective is not Contract.HOKUM
            and play[plan.tail_offset :] == tail_cards
        )
        if plan.tail_offset == 0 and matching_play_exclusions:
            return OrToolsCandidateAttemptV1(
                "infeasible", None, "excluded_complete_play"
            )
        manifest = self.manifest
        project_predicates = _hard_predicates(scenario, manifest, "projects")
        project_template_predicate = (
            project_predicates[0]
            if len(project_predicates) == 1
            and project_predicates[0].kind == "project.exists"
            and _hard_expression_requires_predicate(
                scenario,
                project_predicates[0].predicate_id,
                maximum_conflicts=maximum_conflicts,
            )
            else None
        )
        fixed_owner_cards = {
            int(predicate.arguments["card"])
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and predicate.kind == "deal.card_owner"
            and predicate.arguments.get("handStage")
            in {"initial", "post_purchase"}
        }
        reveal_ordered_projects = (
            self.rules.project_domain.tie_break_order == "second_trick_reveal"
        )
        live_projects = (
            reveal_ordered_projects
            or bool(project_predicates)
            or len(fixed_owner_cards) < 32
        )
        choices = tuple(
            (
                Seat.parse(dealer),
                Contract.parse(contract),
                Seat.parse(buyer),
                Suit.parse(trump) if trump is not None else None,
            )
            for dealer, contract, buyer, trump in plan.boundary_combinations
            if contract == proof_seed.contract.value
            and buyer == proof_seed.buyer.name.lower()
            and trump
            == (proof_seed.trump.name.lower() if proof_seed.trump is not None else None)
        )
        if not choices:
            raise ValueError("proof boundary is outside the planner finite domains")
        per_choice_seconds = float(maximum_seconds) / len(choices)
        all_infeasible = True
        saw_unknown = False
        saw_auction_compatible = not require_optimal_auction_actions
        mismatch_fields: set[str] = set()
        mismatch_hashes: tuple[str, str] | None = None

        def report(substage: str) -> None:
            if progress_callback is not None:
                progress_callback({"prefixLiftSubstage": substage})

        for dealer, contract, buyer, trump in choices:
            if cancel_check is not None and cancel_check():
                return OrToolsCandidateAttemptV1(
                    "incomplete", None, "cancelled"
                )
            reuse_complete_play = False
            if play_hint is not None:
                if tuple(play_hint.plays[plan.tail_offset :]) != tuple(tail.plays):
                    raise ValueError(
                        "complete play hint does not own the supplied tail"
                    )
                hinted_raw, hinted_tricks = _base_points(
                    play_hint, plan.prefix_length
                )
                reuse_complete_play = (
                    hinted_raw == proof_seed.base_raw
                    and hinted_tricks == proof_seed.base_tricks
                    and tuple(play.card.id for play in play_hint.plays)
                    not in matching_play_exclusions
                )
            choice_deadline = time.monotonic() + per_choice_seconds
            report("scenario_model")
            scenario_model = OrToolsScenarioModelV1(
                rules=self.rules,
                dealer=dealer,
                required_buyer=buyer,
                excluded_deck_orders=excluded_decks,
                canonicalize_deal_packets=True,
            )
            hinted_owners = (
                {
                    card.id: int(seat)
                    for seat in Seat
                    for card in play_hint.initial_hands[int(seat)]
                }
                if play_hint is not None
                else {}
            )
            for card_id, owner in hinted_owners.items():
                scenario_model.model.add(
                    scenario_model.owner_of_card[card_id] == owner
                )
            report("auction")
            auction_stage_predicates = tuple(
                predicate
                for predicate in scenario.predicates()
                if manifest.definition(predicate.kind).stage == "auction"
            )
            if self.native_cp_sat_scoring:
                auction_fragment = scenario_model.add_native_auction_fragment(
                    allowed_contracts=(contract,),
                    require_optimal_actions=require_optimal_auction_actions,
                )
                if trump is not None:
                    scenario_model.model.add(auction_fragment.trump == int(trump))
            else:
                auction_fragment = scenario_model.add_auction_fragment(
                    allowed_contracts=(contract,),
                    canonical_completion_only=all(
                        predicate.kind in {"auction.contract", "auction.buyer"}
                        for predicate in auction_stage_predicates
                    ),
                )
            if cancel_check is not None and cancel_check():
                return OrToolsCandidateAttemptV1(
                    "incomplete", None, "cancelled"
                )
            _bind_stage(
                scenario,
                manifest,
                "deal",
                DealPredicateModelContextV2(
                    scenario_model,
                    remaining_owner_of_card=tuple(
                        scenario_model.model.new_constant(
                            next(
                                (
                                    int(seat)
                                    for seat in Seat
                                    if Card(card_id)
                                    in proof_seed.remaining_hands[int(seat)]
                                ),
                                4,
                            )
                        )
                        for card_id in range(32)
                    ),
                ),
            )
            auction_lowerings = _bind_stage(
                scenario,
                manifest,
                "auction",
                CpSatAuctionPredicateContextV1(
                    scenario_model.model,
                    auction_fragment,
                    dealer,
                ),
            )

            if require_optimal_auction_actions:
                auction_probe = scenario_model.model.clone()
                for seat in Seat:
                    for card in tail.initial_hands[int(seat)]:
                        auction_probe.add(
                            auction_probe.get_int_var_from_proto_index(
                                scenario_model.owner_of_card[card.id].index
                            )
                            == int(seat)
                        )
                report("auction_compatibility")
                auction_solver = cp_model.CpSolver()
                auction_solver.parameters.num_search_workers = 1
                auction_solver.parameters.random_seed = seed
                auction_solver.parameters.max_time_in_seconds = max(
                    0.001, choice_deadline - time.monotonic()
                )
                auction_solver.parameters.max_number_of_conflicts = (
                    maximum_conflicts
                )
                auction_status = _solve_cp_sat_with_cancel_v1(
                    auction_solver, auction_probe, cancel_check
                )
                if auction_status == cp_model.INFEASIBLE:
                    continue
                saw_auction_compatible = True

            report("escalation")
            escalation_stage = OrToolsEscalationStageV1(
                EscalationDomainIRV1.from_rules(self.rules.data),
                declarer=buyer,
                contract=contract,
                match_score=proof_seed.starting_match_score,
                native_cp_sat=self.native_cp_sat_scoring,
            )
            escalation_context = CpSatEscalationPredicateContextV1.attach(
                escalation_stage,
                model=scenario_model.model,
                context_id="tail_lift_escalation",
                declarer_variable=auction_fragment.declarer if self.native_cp_sat_scoring else None,
            )
            if cancel_check is not None and cancel_check():
                return OrToolsCandidateAttemptV1(
                    "incomplete", None, "cancelled"
                )
            if self.native_cp_sat_scoring:
                native = escalation_context.native
                scenario_model.model.add(native["terminal"] == 1)
                scenario_model.model.add(native["level"] == native["levels"][proof_seed.escalation])
                scenario_model.model.add(native["locked"] == int(proof_seed.locked))
                scenario_model.model.add(native["lastTeam"] == (2 if proof_seed.last_escalating_team is None else int(proof_seed.last_escalating_team)))
            for candidate, selected in zip(
                escalation_context.candidates,
                escalation_context.selected,
                strict=True,
            ):
                last_team = (
                    candidate.actions[-1].actor.team if candidate.actions else None
                )
                if (
                    not candidate.terminal
                    or candidate.level != proof_seed.escalation
                    or candidate.locked != proof_seed.locked
                    or last_team != proof_seed.last_escalating_team
                ):
                    scenario_model.model.add(selected == 0)
            _bind_stage(scenario, manifest, "escalation", escalation_context)

            report("projects")
            second_trick_leader = (
                scenario_model.model.new_int_var(0, 3, "tail_lift_second_trick_leader")
                if reveal_ordered_projects
                else None
            )
            projects = (
                attach_project_lifecycle_fragment_v1(
                    scenario_model.model,
                    owner_of_card=scenario_model.owner_of_card,
                    fixed_card_owners=(
                        tuple(hinted_owners[card_id] for card_id in range(32))
                        if len(hinted_owners) == 32
                        else None
                    ),
                    contract=contract,
                    dealer=dealer,
                    trump=trump,
                    domain=self.rules.project_domain,
                    name="tail_lift_projects",
                    second_trick_leader=second_trick_leader,
                )
                if live_projects
                else None
            )
            if projects is not None:
                if project_template_predicate is not None:
                    witnesses = _project_exists_witness_keys(
                        project_template_predicate,
                        self.rules,
                        contract,
                        dealer,
                    )
                    selected_witnesses = tuple(
                        scenario_model.model.new_bool_var(
                            f"tail_lift_project_witness_{index}"
                        )
                        for index in range(len(witnesses))
                    )
                    if not selected_witnesses:
                        continue
                    scenario_model.model.add_bool_or(selected_witnesses)
                    for selected, witness in zip(
                        selected_witnesses, witnesses, strict=True
                    ):
                        for card_id in witness[3]:
                            scenario_model.model.add(
                                scenario_model.owner_of_card[card_id]
                                == int(witness[0])
                            ).only_enforce_if(selected)
                for team in Team:
                    scenario_model.model.add(
                        projects.combined_awarded_value[int(team)]
                        == proof_seed.awarded_project_points[int(team)]
                    )
                    scenario_model.model.add(
                        projects.baloot_team[int(team)] * projects.baloot_value
                        == proof_seed.baloot_points[int(team)]
                    )
                _bind_stage(
                    scenario,
                    manifest,
                    "projects",
                    CpSatProjectPredicateContextV1(
                        scenario_model.model,
                        "tail_lift_projects",
                        projects,
                    ),
                )
            if cancel_check is not None and cancel_check():
                return OrToolsCandidateAttemptV1(
                    "incomplete", None, "cancelled"
                )

            if reuse_complete_play:
                if set(hinted_owners) != set(range(32)):
                    raise ValueError(
                        "complete play hint does not own the supplied tail"
                    )
                if second_trick_leader is not None:
                    scenario_model.model.add(
                        second_trick_leader == int(play_hint.trick_winners[0])
                    )
                report("play_hint_solve")
                solver = cp_model.CpSolver()
                solver.parameters.num_search_workers = 1
                solver.parameters.random_seed = seed
                solver.parameters.max_time_in_seconds = max(
                    0.001, choice_deadline - time.monotonic()
                )
                solver.parameters.max_number_of_conflicts = maximum_conflicts
                status = _solve_cp_sat_with_cancel_v1(
                    solver, scenario_model.model, cancel_check
                )
                if status == cp_model.INFEASIBLE:
                    continue
                if status != cp_model.OPTIMAL:
                    saw_unknown = True
                    continue
                hands = {
                    seat: tuple(
                        card for card in play_hint.initial_hands[int(seat)]
                    )
                    for seat in Seat
                }
                if projects is not None:
                    project_result = project_lifecycle_result_v1(projects, solver)
                elif certified_projects is not None:
                    project_result = certified_projects
                else:
                    project_result = OrToolsProjectLifecycleModelV1().solve(
                        hands=hands,
                        contract=contract,
                        dealer=dealer,
                        trump=trump,
                        domain=self.rules.project_domain,
                        timeout_seconds=max(
                            0.001, choice_deadline - time.monotonic()
                        ),
                        maximum_conflicts=maximum_conflicts,
                        cancel_check=cancel_check,
                    )
                    if (
                        project_result.combined_awarded_value
                        != proof_seed.awarded_project_points
                        or project_result.baloot_points != proof_seed.baloot_points
                    ):
                        continue
                complete_play = play_hint
                projected = complete_play.at_prefix(plan.prefix_length)
                raw, tricks = _base_points(complete_play, plan.prefix_length)
                escalation = escalation_context.extract_stage_result(
                    solver,
                    model_hash=canonical_sha256(
                        {
                            "schemaVersion": 1,
                            "cpModelProto": str(scenario_model.model.proto),
                        }
                    ),
                    solver_status=solver.status_name(status).lower(),
                    branches=int(solver.num_branches),
                    conflicts=int(solver.num_conflicts),
                )
                derived_seed = _seed_clone(
                    proof_seed,
                    remaining_hands=projected.remaining_hands,
                    current_trick=projected.current_trick,
                    first_leader=(
                        projected.current_trick[0].actor
                        if projected.current_trick
                        else projected.next_actor
                    ),
                    next_actor=projected.next_actor,
                    legal_actions=projected.legal_play_actions,
                    base_raw=raw,
                    base_tricks=tricks,
                    awarded_project_points=project_result.combined_awarded_value,
                    baloot_points=project_result.baloot_points,
                )
                if (
                    derived_seed.proof_state_projection_hash
                    != proof_seed.proof_state_projection_hash
                ):
                    raise ValueError(
                        "complete play hint disagrees with the proof boundary"
                    )
                return self._candidate(
                    scenario,
                    scenario_model,
                    auction_fragment.extract(solver),
                    escalation,
                    project_result,
                    complete_play,
                    projected,
                    proof_seed,
                    solver,
                    auction_lowerings,
                )

            ownership = {
                card: seat for seat in Seat for card in tail.initial_hands[int(seat)]
            }
            first_leader = self.rules.play_domain.resolve_first_leader(
                dealer=dealer,
                buyer=buyer,
                structural_domain=self.rules.structural_domain,
            )
            if plan.tail_offset is None:
                raise AssertionError("tail-first lift requires a tail offset")
            reached_play_count = plan.tail_offset + 4
            native_card_game = None
            if self.native_cp_sat_scoring:
                from .ortools_scoring_stage import attach_raw_conversion_v1

                scoring_domain = ScoringDomainIRV1.from_rules(self.rules.data)
                native_card_game = []
                for team in Team:
                    full_raw = scenario_model.model.new_int_var(
                        0, scoring_domain.raw_card_totals[contract.effective.value],
                        f"lift_complete_raw_{int(team)}",
                    )
                    scenario_model.model.add(full_raw == proof_seed.base_raw[int(team)] + sum(
                        points for winner, points in zip(tail.trick_winners, tail.trick_raw_points, strict=True)
                        if winner.team == team
                    ) + scoring_domain.last_trick_bonus_raw * int(tail.trick_winners[-1].team == team))
                    native_card_game.append(attach_raw_conversion_v1(
                        scenario_model.model, full_raw, domain=scoring_domain, contract=contract,
                        name=f"lift_complete_game_{int(team)}",
                    ))
            try:
                report("play_model")
                play_build = OrToolsPlayTailStageV1(native_conversion=self.native_cp_sat_scoring).build_model(
                    cards=FULL_DECK,
                    ownership=ownership,
                    contract=contract,
                    trump=trump,
                    first_leader=first_leader,
                    play_domain=self.rules.play_domain,
                    structural_domain=self.rules.structural_domain,
                    scoring_domain=ScoringDomainIRV1.from_rules(self.rules.data),
                    locked=proof_seed.locked,
                    tail=tail.plays[:4],
                    decision_prefix_length=plan.prefix_length,
                    play_step_count=reached_play_count,
                    policy_prefix_steps=policy_prefix_steps,
                    policy_id=policy_id,
                    shared_model=scenario_model.model,
                    shared_owner_of_card=scenario_model.owner_of_card,
                    cancel_check=cancel_check,
                )
                if second_trick_leader is not None:
                    scenario_model.model.add(
                        second_trick_leader == play_build._extract.winner_seats[0]
                    )
                for index, no_good in enumerate(
                    dict.fromkeys(excluded_play_zero_actions)
                ):
                    apply_play_zero_observation_action_no_good_v1(
                        scenario_model.model,
                        owner_of_card=scenario_model.owner_of_card,
                        selected_card=play_build.selected_play_indexes[0],
                        ekka_lead=play_build.ekka_lead_variables[0],
                        no_good=no_good,
                        name=f"tail_lift_play_zero_no_good_{index}",
                    )
                for excluded in matching_play_exclusions:
                    scenario_model.model.add_forbidden_assignments(
                        play_build.selected_play_indexes[: plan.tail_offset],
                        (excluded[: plan.tail_offset],),
                    )
                if self.native_cp_sat_scoring:
                    _bind_stage(
                        scenario,
                        manifest,
                        "playPrefix",
                        _PlayQualificationContextV1(
                            play_build.play_state_context,
                            _lift_play_predicate_context(
                                scenario_model.model,
                                play_build.predicate_context,
                                tail,
                            ).tricks,
                        ),
                    )
            except PlayTailSolveStopped:
                return OrToolsCandidateAttemptV1(
                    "incomplete", None, "cancelled"
                )
            if play_hint is not None:
                hinted_owners = {
                    card.id: int(seat)
                    for seat in Seat
                    for card in play_hint.initial_hands[int(seat)]
                }
                for card_id, owner in enumerate(scenario_model.owner_of_card):
                    scenario_model.model.add_hint(owner, hinted_owners[card_id])
                for step, selected in enumerate(
                    play_build.selected_play_indexes[:reached_play_count]
                ):
                    scenario_model.model.add_hint(
                        selected, play_hint.plays[step].card.id
                    )
            if cancel_check is not None and cancel_check():
                return OrToolsCandidateAttemptV1(
                    "incomplete", None, "cancelled"
                )
            report("proof_bindings")
            if not _preserve_tail_cards_before_decision(
                play_build, tail, plan.tail_offset, cancel_check,
            ):
                return OrToolsCandidateAttemptV1("incomplete", None, "cancelled")
            _bind_proof_seed_prefix(
                play_build, proof_seed, (plan.tail_offset or 0) // 4
            )

            while time.monotonic() < choice_deadline:
                if cancel_check is not None and cancel_check():
                    saw_unknown = True
                    break
                solver = cp_model.CpSolver()
                try:
                    report("play_solve")
                    play = play_build.solve(
                        maximum_time_seconds=max(
                            0.001, choice_deadline - time.monotonic()
                        ),
                        maximum_conflicts=maximum_conflicts,
                        random_seed=seed,
                        solver=solver,
                        cancel_check=cancel_check,
                    )
                except PlayTailSolveStopped as stopped:
                    saw_unknown = saw_unknown or stopped.solver_status != "infeasible"
                    all_infeasible = (
                        all_infeasible and stopped.solver_status == "infeasible"
                    )
                    break

                hands = {
                    seat: tuple(
                        Card(card_id)
                        for card_id, owner in enumerate(
                            scenario_model.owner_of_card
                        )
                        if int(solver.value(owner)) == int(seat)
                    )
                    for seat in Seat
                }
                if projects is not None:
                    project_result = project_lifecycle_result_v1(projects, solver)
                elif certified_projects is not None:
                    project_result = certified_projects
                else:
                    try:
                        report("project_solve")
                        project_result = OrToolsProjectLifecycleModelV1().solve(
                            hands=hands,
                            contract=contract,
                            dealer=dealer,
                            trump=trump,
                            domain=self.rules.project_domain,
                            timeout_seconds=max(
                                0.001, choice_deadline - time.monotonic()
                            ),
                            maximum_conflicts=maximum_conflicts,
                            cancel_check=cancel_check,
                        )
                    except RuntimeError:
                        saw_unknown = True
                        break
                    project_mismatch = (
                        project_result.combined_awarded_value
                        != proof_seed.awarded_project_points
                        or project_result.baloot_points != proof_seed.baloot_points
                    )
                    if project_mismatch:
                        owner_vector = tuple(
                            int(solver.value(owner))
                            for owner in scenario_model.owner_of_card
                        )
                        scenario_model.model.add_forbidden_assignments(
                            scenario_model.owner_of_card, (owner_vector,)
                        )
                        continue

                all_infeasible = False
                complete_play = _compose_prefix_and_verified_tail(
                    play,
                    tail,
                    tail_offset=plan.tail_offset,
                    contract=contract,
                    scoring=ScoringDomainIRV1.from_rules(self.rules.data),
                    solved_card_game_points=(tuple(int(solver.value(value)) for value in native_card_game)
                                             if native_card_game is not None else None),
                )
                projected = complete_play.at_prefix(plan.prefix_length or 0)
                raw, tricks = _base_points(
                    complete_play, plan.prefix_length or 0
                )
                report("candidate_finalize")
                if cancel_check is not None and cancel_check():
                    return OrToolsCandidateAttemptV1(
                        "incomplete", None, "cancelled"
                    )
                escalation = escalation_context.extract_stage_result(
                    solver,
                    model_hash=play.model_hash,
                    solver_status=play.solver_status,
                    branches=int(solver.num_branches),
                    conflicts=int(solver.num_conflicts),
                )
                derived_seed = _seed_clone(
                    proof_seed,
                    remaining_hands=projected.remaining_hands,
                    current_trick=projected.current_trick,
                    first_leader=(
                        projected.current_trick[0].actor
                        if projected.current_trick
                        else projected.next_actor
                    ),
                    next_actor=projected.next_actor,
                    legal_actions=projected.legal_play_actions,
                    base_raw=raw,
                    base_tricks=tricks,
                    awarded_project_points=project_result.combined_awarded_value,
                    baloot_points=project_result.baloot_points,
                )
                if (
                    derived_seed.proof_state_projection_hash
                    != proof_seed.proof_state_projection_hash
                ):
                    mismatch_fields.update(
                        name
                        for name in CompleteStateGraphSeedV1.__dataclass_fields__
                        if name
                        not in {"exact_state_id", "proof_state_projection_hash"}
                        and getattr(derived_seed, name) != getattr(proof_seed, name)
                    )
                    mismatch_hashes = (
                        proof_seed.proof_state_projection_hash,
                        derived_seed.proof_state_projection_hash,
                    )
                    break
                report("candidate_build")
                if cancel_check is not None and cancel_check():
                    return OrToolsCandidateAttemptV1(
                        "incomplete", None, "cancelled"
                    )
                return self._candidate(
                    scenario,
                    scenario_model,
                    auction_fragment.extract(solver),
                    escalation,
                    project_result,
                    complete_play,
                    projected,
                    proof_seed,
                    solver,
                    auction_lowerings,
                )
            else:
                # The model was not proved infeasible if its time budget
                # expired before (or between) play solves.
                saw_unknown = True

        if saw_unknown:
            return OrToolsCandidateAttemptV1(
                "incomplete", None, "tail_prefix_lift_budget"
            )
        if all_infeasible:
            return OrToolsCandidateAttemptV1(
                "infeasible",
                None,
                (
                    "tail_prefix_lift_infeasible"
                    if saw_auction_compatible
                    else "tail_optimal_auction_infeasible"
                ),
            )
        reason = "tail_prefix_target_miss:" + ",".join(sorted(mismatch_fields))
        if mismatch_hashes:
            reason += f":{mismatch_hashes[0]}:{mismatch_hashes[1]}"
        return OrToolsCandidateAttemptV1("not_found", None, reason)

    def _validate_inputs(
        self,
        scenario: ResearchScenarioV1,
        plan: TailFirstStagePlanV1,
        tail: OrToolsPlayTailResultV1,
        proof_seed: CompleteStateGraphSeedV1,
    ) -> None:
        if not isinstance(scenario, ResearchScenarioV1):
            raise TypeError("tail-first lift requires ResearchScenarioV1")
        if not isinstance(plan, TailFirstStagePlanV1) or plan.planner != "tail_first":
            raise ValueError("tail-first lift requires a tail-first stage plan")
        if _plan_hash(plan) != plan.sha256:
            raise ValueError("tail-first stage plan checksum mismatch")
        expected_plan = TailFirstStagePlannerV1(
            native_cp_sat=self.native_cp_sat_scoring,
        ).plan(scenario)
        if expected_plan.sha256 != plan.sha256:
            raise ValueError("tail-first stage plan does not own the supplied scenario")
        if not isinstance(tail, OrToolsPlayTailResultV1):
            raise TypeError("tail-first lift requires OrToolsPlayTailResultV1")
        if _tail_evidence_hash(
            tail
        ) != tail.evidence_hash or tail.sha256 != canonical_sha256(
            {
                "schemaVersion": 1,
                "modelHash": tail.model_hash,
                "evidenceHash": tail.evidence_hash,
                "solverStatus": tail.solver_status,
            }
        ):
            raise ValueError("play-tail checksum mismatch")
        if (
            plan.prefix_length is None
            or plan.tail_offset is None
            or plan.tail_length != len(tail.plays)
        ):
            raise ValueError("play tail does not match the planned decision boundary")
        relative_prefix = plan.prefix_length - plan.tail_offset
        if tail.prefix_length != relative_prefix:
            raise ValueError("supplied play tail does not match its decision position")
        if not isinstance(proof_seed, CompleteStateGraphSeedV1):
            raise TypeError("tail-first lift requires CompleteStateGraphSeedV1")
        if proof_seed.rules_hash != self.rules.sha256:
            raise ValueError("proof boundary rules hash mismatch")
        if (
            _seed_clone(proof_seed).proof_state_projection_hash
            != proof_seed.proof_state_projection_hash
        ):
            raise ValueError("proof boundary projection checksum mismatch")
        if tuple(tail.current_trick) != tuple(proof_seed.current_trick):
            raise ValueError("play tail and proof boundary current trick disagree")
        expected_leader = (
            tail.current_trick[0].actor if tail.current_trick else tail.next_actor
        )
        if proof_seed.first_leader != expected_leader:
            raise ValueError("tail-first proof boundary leader disagrees")
        if tuple(tail.remaining_hands) != tuple(proof_seed.remaining_hands):
            raise ValueError("play tail and proof boundary hands disagree")
        if tail.next_actor != proof_seed.next_actor or tuple(
            tail.legal_play_actions
        ) != tuple(proof_seed.legal_actions):
            raise ValueError("play tail and proof boundary legal actions disagree")
        if sum(proof_seed.base_tricks) * 4 != plan.tail_offset:
            raise ValueError("proof boundary trick count disagrees with the plan")

    def _candidate(
        self,
        scenario: ResearchScenarioV1,
        model: OrToolsScenarioModelV1,
        auction: Any,
        escalation: Any,
        projects: OrToolsProjectLifecycleResultV1,
        play: OrToolsPlayTailResultV1,
        projected: OrToolsPlayTailResultV1,
        proof_seed: CompleteStateGraphSeedV1,
        solver: cp_model.CpSolver,
        auction_lowerings: Mapping[str, Any],
    ) -> OrToolsCandidateAttemptV1:
        if auction.completion_trump != proof_seed.trump:
            raise AssertionError("auction and tactical tail disagree on trump")
        deck = tuple(int(solver.value(item)) for item in model.card_at_position)
        buyer = auction.completion_buyer
        contract = auction.completion_contract
        scoring = ScoringDomainIRV1.from_rules(self.rules.data)
        score = _score_complete_candidate(
            rules=self.rules,
            contract=contract,
            buyer=buyer,
            escalation=escalation,
            projects=projects,
            play=play,
            starting_match_score=proof_seed.starting_match_score,
            native_cp_sat=self.native_cp_sat_scoring,
        )
        trace = {
            "schemaVersion": 1,
            "deckOrder": list(deck),
            "dealer": model.dealer.name.lower(),
            "roundNumber": 1,
            "matchScore": list(proof_seed.starting_match_score),
            "buyer": buyer.name.lower(),
            "contract": contract.value,
            "trumpSuit": (
                auction.completion_trump.name.lower()
                if auction.completion_trump is not None
                else None
            ),
            "auction": auction.to_dict(),
            "escalation": escalation.to_dict(),
            "projects": projects.to_dict(),
            "play": projected.to_dict(),
            "score": score,
        }
        trace_hash = _hash_document(trace)
        model_hash = play.model_hash
        solution_hash = canonical_sha256(
            {"deckOrder": list(deck), "modelHash": model_hash}
        )
        lift_model_hash = canonical_sha256(
            {
                "schemaVersion": 1,
                "cpModelProto": str(model.model.proto),
            }
        )
        auction_predicates = _hard_predicates(
            scenario,
            self.manifest,
            "auction",
        )
        auction_evidence = {
            predicate_id: lowering.extract(solver)
            for predicate_id, lowering in sorted(auction_lowerings.items())
        }
        auction_assumptions, auction_bindings = _cp_sat_receipt_metadata_v1(
            model.model,
            stage="auction",
            predicates=auction_predicates,
            predicate_literals={
                predicate_id: lowering.literal
                for predicate_id, lowering in auction_lowerings.items()
            },
            model_hash=lift_model_hash,
            evidence_hash=auction.trace_hash,
            lowering_manifest_hash=(
                self.manifest.manifest_hash
            ),
        )
        auction_bindings = tuple(
            {
                **dict(binding),
                "evidenceHash": auction_evidence[
                    str(binding["predicateId"])
                ]["sha256"],
            }
            for binding in auction_bindings
        )
        receipts = {
            "deal": _stage_receipt(
                "deal",
                model_hash=model_hash,
                evidence_hash=solution_hash,
                semantic={"deckOrder": list(deck)},
                solver_status=play.solver_status,
                branches=int(solver.num_branches),
                conflicts=int(solver.num_conflicts),
            ),
            "auction": _stage_receipt(
                "auction",
                model_hash=lift_model_hash,
                evidence_hash=auction.trace_hash,
                semantic=auction.to_dict(),
                solver_status=play.solver_status,
                variable_count=len(model.model.proto.variables),
                constraint_count=len(model.model.proto.constraints),
                named_assumptions=auction_assumptions,
                evidence_bindings=auction_bindings,
            ),
            "escalation": _stage_receipt(
                "escalation",
                model_hash=escalation.model_hash,
                evidence_hash=escalation.evidence_hash or _HASH_ZERO,
                semantic=escalation.to_dict(),
                solver_status=escalation.solver_status,
                branches=escalation.branches,
                conflicts=escalation.conflicts,
            ),
            "projects": _stage_receipt(
                "projects",
                model_hash=projects.model_hash,
                evidence_hash=projects.evidence_hash,
                semantic=projects.to_dict(),
                solver_status=projects.solver_status,
            ),
            "playPrefix": _stage_receipt(
                "playPrefix",
                model_hash=play.model_hash,
                evidence_hash=play.evidence_hash,
                semantic=play.to_dict(),
                solver_status=play.solver_status,
            ),
            "state": _stage_receipt(
                "state",
                model_hash=play.model_hash,
                evidence_hash=proof_seed.proof_state_projection_hash,
                semantic={
                    **projected.to_dict()["decisionState"],
                    "proofStateProjectionHash": proof_seed.proof_state_projection_hash,
                },
                solver_status=play.solver_status,
            ),
            "outcome": _stage_receipt(
                "outcome",
                model_hash=(
                    str(score["modelHash"])
                    if self.native_cp_sat_scoring
                    else canonical_sha256({"scoringDomainHash": scoring.sha256})
                ),
                evidence_hash=score["sha256"],
                semantic=score,
                solver_status="optimal",
                branches=int(score.get("branches", 0)),
                conflicts=int(score.get("conflicts", 0)),
            ),
        }
        state = {
            "schemaVersion": 1,
            "decisionPoint": copy.deepcopy(dict(scenario.decision_point)),
            "decisionState": projected.to_dict()["decisionState"],
            "proofStateProjectionHash": proof_seed.proof_state_projection_hash,
            "score": score,
            "traceHash": trace_hash,
        }
        return OrToolsCandidateAttemptV1(
            "feasible",
            OrToolsNeutralCandidateV1(
                "search",
                trace,
                state,
                deck,
                receipts,
                None,
            ),
            predicate_evidence=auction_evidence,
        )


__all__ = ["OrToolsTailFirstBackwardLiftV1"]
