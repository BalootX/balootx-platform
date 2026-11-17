"""Tail-first candidate construction for the default OR-Tools executor."""

from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from ortools.sat.python import cp_model

from baloot_platform.core.cards import FULL_DECK, Card
from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.enums import Contract, Rank, Seat, Suit, Team
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.tactics import DELAYED_PAYOFF_TACTIC_IDS, TacticalDetectorProfileV1

from .ortools_complete_state_graph import (
    CompleteStateGraphSeedV1,
    INTERNAL_RAW_POINT_MARGIN_UTILITY_V1,
    OrToolsCompleteStateGraphCompilerV1,
)
from .ortools_play_tail import (
    OrToolsPlayTailResultV1,
    OrToolsPlayTailStageV1,
    PlayZeroObservationActionNoGoodV1,
    PlayTailSolveStopped,
    apply_play_zero_observation_action_no_good_v1,
)
from .ortools_escalation_stage import OrToolsEscalationStageV1
from .ortools_information_set_compiler import InformationSetCompilationStopped
from .ortools_project_fragment import attach_project_lifecycle_fragment_v1
from .ortools_proof import (
    OrToolsGraphProofOutcomeV1,
    OrToolsScenarioGraphProofV1,
    ProofRootActionV1,
)
from .ortools_research_executor import (
    OrToolsCandidateAttemptV1,
    OrToolsNeutralCandidateV1,
    _hard_expression_requires_predicate,
    _required_search_initial_inputs_v3,
    _hard_expression_is_cross_stage_conjunction,
    _combined_expression,
    _project_exists_witness_keys,
    _enforce_hard_stage_projection,
    _solve_cp_sat_with_cancel_v1,
    _cp_sat_receipt_metadata_v1,
    _stage_receipt,
    _apply_candidate_partition_v1,
    _validated_candidate_partition_v1,
)
from .ortools_tail_first_lift import (
    OrToolsTailFirstBackwardLiftV1,
    _bind_proof_seed_prefix,
    _preserve_tail_cards_before_decision,
    _seed_clone,
)
from .ortools_tail_proof_boundary import (
    OrToolsTailProofBoundaryV1,
    TailProofBoundaryNoGoodV1,
    TailProofStateNoGoodV1,
)
from .ortools_scoring_stage import (
    CpSatScoringInputsV1,
    attach_raw_conversion_v1,
    attach_scoring_fragment_v1,
    scoring_contract_id_v1,
)
from .ortools_scenario_model import OrToolsScenarioModelV1
from .scenario_contracts import (
    ResearchScenarioV1,
    ScenarioPredicateV1,
    ScenarioProofRequestV1,
    canonical_sha256,
    validate_native_delayed_payoff_scope_v1,
)
from .scenario_predicate_manifest_v2 import (
    ScenarioPredicateCatalogLoweringManifestV2,
    shared_scenario_predicate_catalog_lowering_manifest_v2,
)
from .scenario_predicate_lowering_v2 import (
    CpSatActionVariablesV1,
    CpSatPredicateModelContextV1,
    CpSatTrickVariablesV1,
    _selector_matches,
)
from .scenario_registry_v2 import CardSelectorV1
from .scenario_proof_tactic_predicates_v2 import lower_native_proof_predicate_v3
from .scenario_deal_predicates_v2 import DealPredicateModelContextV2
from .scenario_predicate_auction_escalation_v2 import (
    CpSatAuctionPredicateContextV1,
    CpSatEscalationPredicateContextV1,
)
from .scenario_project_outcome_lowering_v2 import CpSatProjectPredicateContextV1
from .tail_first_stage_planner import TailFirstStagePlanV1

_UNIQUE_ROOT_SUIT_RELEASE_GUIDANCE = "unique_root_suit_release_single_ace_v2"


def _trick_card_count_invariant_v1(
    predicate: ScenarioPredicateV1,
    manifest: ScenarioPredicateCatalogLoweringManifestV2,
    *,
    deadline: float,
    maximum_conflicts: int,
    cancel_check: Callable[[], bool] | None,
) -> Mapping[str, Any] | None:
    """Prove a count over any complete trick using its registered complement."""
    if time.monotonic() >= deadline or cancel_check is not None and cancel_check():
        return None
    model = cp_model.CpModel()
    actions = tuple(
        CpSatActionVariablesV1(
            action_id=f"trick_count_invariant_{seat}",
            card=model.new_int_var(0, 31, f"trick_count_invariant_card_{seat}"),
            actor=model.new_constant(seat),
            original_owner=model.new_constant(seat),
            ekka=model.new_bool_var(f"trick_count_invariant_ekka_{seat}"),
            present=model.new_constant(1),
        )
        for seat in range(4)
    )
    model.add_all_different([action.card for action in actions])
    trick_index = int(predicate.arguments["trickIndex"])
    context = CpSatPredicateModelContextV1(
        model, "complete_trick_count_invariant",
        {trick_index: CpSatTrickVariablesV1(trick_index, actions, actions[0])},
        (), model.new_constant(1),
    )
    lowering = manifest.lower_predicate(predicate, context)
    model.add(lowering.literal == 0)
    remaining = deadline - time.monotonic()
    if remaining <= 0 or cancel_check is not None and cancel_check():
        return None
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = remaining
    solver.parameters.max_number_of_conflicts = maximum_conflicts
    status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
    if (status != cp_model.INFEASIBLE or time.monotonic() >= deadline
            or cancel_check is not None and cancel_check()):
        return None
    core = {
        "predicateId": predicate.predicate_id,
        "predicateHash": predicate.sha256,
        "loweringHash": lowering.lowering_hash,
        "modelHash": canonical_sha256({"proto": str(model.proto)}),
        "solverStatus": "infeasible",
    }
    return {**core, "sha256": canonical_sha256(core)}


def _unique_raw_root_matches_selectors_v1(
    root_actions: Sequence[ProofRootActionV1],
    *,
    actor: Seat,
    selectors: Sequence[CardSelectorV1],
) -> bool:
    if not root_actions:
        return False
    best = max(action.value for action in root_actions)
    winners = [action for action in root_actions if action.value == best]
    if len(winners) != 1:
        return False
    action = LegalPlayActionV1.from_code(winners[0].action_code)
    return all(
        _selector_matches(
            selector,
            card_id=action.card.id,
            actor_id=int(actor),
            owner_id=int(actor),
            ekka=action.ekka,
        )
        for selector in selectors
    )



def _reachable_tail_view_v1(
    complete: OrToolsPlayTailResultV1,
    *,
    tail_offset: int,
    decision_prefix_length: int,
) -> OrToolsPlayTailResultV1:
    """Project a solved complete round into the existing late-tail contract."""

    if len(complete.plays) != 32 or tail_offset % 4:
        raise ValueError("reachable-tail projection requires one complete round")
    tail_plays = complete.plays[tail_offset:]
    relative_prefix = decision_prefix_length - tail_offset
    if relative_prefix < 0 or relative_prefix >= len(tail_plays):
        raise ValueError("reachable-tail decision lies outside the projected tail")
    tail_start = complete.at_prefix(tail_offset)
    decision = complete.at_prefix(decision_prefix_length)
    trick_offset = tail_offset // 4
    model_hash = canonical_sha256(
        {
            "schemaVersion": 1,
            "kind": "reachable_complete_round_tail",
            "sourceModelHash": complete.model_hash,
            "tailOffset": tail_offset,
        }
    )
    evidence_hash = canonical_sha256(
        {
            "schemaVersion": 1,
            "initialHands": [
                [card.code for card in hand] for hand in tail_start.remaining_hands
            ],
            "plays": [play.to_dict() for play in tail_plays],
            "legalActionsByStep": [
                [card.code for card in actions]
                for actions in complete.legal_actions_by_step[tail_offset:]
            ],
            "legalPlayActionsByStep": [
                [action.to_dict() for action in actions]
                for actions in complete.legal_play_actions_by_step[tail_offset:]
            ],
            "trickWinners": [
                seat.name.lower() for seat in complete.trick_winners[trick_offset:]
            ],
            "trickWinningCards": [
                card.code for card in complete.trick_winning_cards[trick_offset:]
            ],
            "trickRawPoints": list(complete.trick_raw_points[trick_offset:]),
            "prefixLength": relative_prefix,
            "remainingHands": [
                [card.code for card in hand] for hand in decision.remaining_hands
            ],
            "currentTrick": [play.to_dict() for play in decision.current_trick],
            "nextActor": (
                decision.next_actor.name.lower()
                if decision.next_actor is not None
                else None
            ),
            "legalActions": [card.code for card in decision.legal_actions],
            "legalPlayActions": [
                action.to_dict() for action in decision.legal_play_actions
            ],
            "rawTeamPoints": None,
            "cardGamePoints": None,
            "scoringDomainHash": None,
        }
    )
    semantic_hash = canonical_sha256(
        {
            "schemaVersion": 1,
            "modelHash": model_hash,
            "evidenceHash": evidence_hash,
            "solverStatus": complete.solver_status,
        }
    )
    return OrToolsPlayTailResultV1(
        tail_start.remaining_hands,
        tail_plays,
        complete.trick_winners[trick_offset:],
        complete.trick_winning_cards[trick_offset:],
        complete.trick_raw_points[trick_offset:],
        relative_prefix,
        decision.remaining_hands,
        decision.current_trick,
        decision.next_actor,
        decision.legal_actions,
        decision.legal_play_actions,
        complete.legal_actions_by_step[tail_offset:],
        complete.legal_play_actions_by_step[tail_offset:],
        None,
        None,
        None,
        model_hash,
        evidence_hash,
        complete.solver_status,
        semantic_hash,
    )


def _proof_stage_projection_v1(
    scenario: ResearchScenarioV1,
    proof: OrToolsGraphProofOutcomeV1,
    manifest: ScenarioPredicateCatalogLoweringManifestV2,
    *,
    native_cp_sat: bool,
    maximum_seconds: float,
    maximum_conflicts: int,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[str, Mapping[str, Any]]:
    """Project proof/tactic claims without changing the strategic proof."""

    deadline = time.monotonic() + maximum_seconds
    predicates = tuple(
        predicate
        for predicate in scenario.predicates()
        if predicate.hardness == "hard"
        and manifest.definition(predicate.kind).stage in {"proof", "tactics"}
    )
    if not predicates:
        return "satisfied", {"predicateIds": []}
    predicate_ids = [predicate.predicate_id for predicate in predicates]
    native = getattr(proof, "native_qualification", None)
    if native is None or any(predicate_id not in native for predicate_id in predicate_ids):
        return "incomplete", {
            "predicateIds": predicate_ids,
            "reason": "native_tail_proof_bindings_unavailable",
        }
    evidence = {
        "predicateIds": predicate_ids,
        "nativeQualification": {
            predicate_id: dict(native[predicate_id])
            for predicate_id in predicate_ids
        },
    }
    # Preserve frozen V2 behavior and the cheap conjunctive V3 path.
    if not native_cp_sat or _hard_expression_is_cross_stage_conjunction(_combined_expression(scenario)):
        if any(not bool(native[predicate_id].get("satisfied")) for predicate_id in predicate_ids):
            return "false", evidence
        return "satisfied", evidence
    document = proof.model_document()
    model = cp_model.CpModel()
    proto = document.get("modelProto")
    if (canonical_sha256(document) != proof.model_hash or type(proto) is not str
            or not model.proto.parse_text_format(proto)):
        return "incomplete", {**evidence, "reason": "native_tail_proof_model_unavailable"}
    indices = document.get("nativeQualificationLiteralIndices", {})
    names = document.get("nativeQualificationVariables", {})
    if not isinstance(indices, Mapping) or not isinstance(names, Mapping):
        return "incomplete", {**evidence, "reason": "native_tail_proof_bindings_unavailable"}
    literals = {}
    for predicate_id in predicate_ids:
        index = indices.get(predicate_id)
        if (type(index) is not int or not 0 <= index < len(model.proto.variables)
                or model.proto.variables[index].name != names.get(predicate_id)
                or tuple(model.proto.variables[index].domain) not in {(0, 0), (0, 1), (1, 1)}):
            return "incomplete", {**evidence, "reason": "native_tail_proof_bindings_unavailable"}
        literals[predicate_id] = model.get_bool_var_from_proto_index(index)
    # Reuse the native constraints, not Python truth constants or a changed proof.
    _enforce_hard_stage_projection(model, scenario, stage="tail_proof", predicate_literals=literals)
    remaining = deadline - time.monotonic()
    if remaining <= 0 or cancel_check is not None and cancel_check():
        return "incomplete", {**evidence, "reason": "tail_proof_projection_budget"}
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = remaining
    solver.parameters.max_number_of_conflicts = maximum_conflicts
    status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
    evidence.update(projectionModelHash=canonical_sha256({"proto": str(model.proto)}),
                    projectionSolverStatus=solver.status_name(status).lower())
    if status == cp_model.INFEASIBLE:
        return "false", evidence
    return ("satisfied" if status == cp_model.OPTIMAL else "incomplete"), evidence
def _seed_with_target_raw_v1(
    seed: CompleteStateGraphSeedV1,
    target_team: Team,
    target_raw: int,
) -> CompleteStateGraphSeedV1:
    total_raw = sum(seed.base_raw)
    if not 0 <= target_raw <= total_raw:
        raise ValueError("target base raw is outside the fixed prefix total")
    base_raw = [0, 0]
    base_raw[int(target_team)] = target_raw
    base_raw[int(target_team.other)] = total_raw - target_raw
    values = {
        "rules_hash": seed.rules_hash,
        "remaining_hands": seed.remaining_hands,
        "current_trick": seed.current_trick,
        "first_leader": seed.first_leader,
        "next_actor": seed.next_actor,
        "legal_actions": seed.legal_actions,
        "contract": seed.contract,
        "trump": seed.trump,
        "buyer": seed.buyer,
        "locked": seed.locked,
        "base_raw": (base_raw[0], base_raw[1]),
        "base_tricks": seed.base_tricks,
        "awarded_project_points": seed.awarded_project_points,
        "baloot_points": seed.baloot_points,
        "escalation": seed.escalation,
        "last_escalating_team": seed.last_escalating_team,
        "starting_match_score": seed.starting_match_score,
    }
    provisional = CompleteStateGraphSeedV1.build(
        exact_state_id="STATE-PROOF-STATE-PROBE",
        **values,
    )
    return CompleteStateGraphSeedV1.build(
        exact_state_id=(
            "STATE-" + provisional.proof_state_projection_hash[:24].upper()
        ),
        **values,
    )


def _raw_root_state_key_v1(
    seed: CompleteStateGraphSeedV1, target_team: Team
) -> str:
    # A prefix raw-score split adds the same constant to every continuation.
    # Keep every other proof input in the key, including the fixed raw total.
    return _seed_with_target_raw_v1(seed, target_team, 0).proof_state_projection_hash


def _raw_root_uniqueness_key_v1(
    seed: CompleteStateGraphSeedV1, target_team: Team
) -> str:
    """Raw-card ranking is independent of the awarded-score context."""
    core = seed.to_dict()
    for key in (
        "exactStateId", "proofStateProjectionHash", "baseRaw", "baseTricks",
        "awardedProjectPoints", "balootPoints", "escalation",
        "lastEscalatingTeam", "startingMatchScore",
    ):
        core.pop(key)
    core.update(targetTeam=int(target_team), baseRawTotal=sum(seed.base_raw),
                completedTricks=sum(seed.base_tricks))
    return canonical_sha256(core)


def _first_winning_count_range_required_v1(
    scenario: ResearchScenarioV1, proof_predicates: Sequence[Any]
) -> bool:
    # Only count ranges accepting one winner can use the first winning split.
    return bool(proof_predicates) and all(
        predicate.kind == "proof.winning_root_action_count"
        and predicate.arguments["minimum"] == 1
        and _hard_expression_requires_predicate(scenario, predicate.predicate_id)
        for predicate in proof_predicates
    )


def _first_winning_raw_interval_v1(
    current_raw: int, raw_total: int, winning_root_count: int
) -> tuple[int, int] | None:
    """Round-winning roots cannot disappear as the target's prefix raw rises."""
    if winning_root_count:
        return 0, current_raw
    if current_raw < raw_total:
        return current_raw + 1, raw_total
    return None


def _bind_reachable_project_truth_v1(
    build: Any,
    *,
    scenario: ResearchScenarioV1,
    manifest: ScenarioPredicateCatalogLoweringManifestV2,
    project_predicates: Sequence[Any],
    seed: CompleteStateGraphSeedV1,
    tail: OrToolsPlayTailResultV1,
    rules: RuleBook,
    contract: Contract,
    dealer: Seat,
    trump: Suit | None,
) -> None:
    """Keep project evidence on the same deal as the reachable play prefix."""
    known_owners = {
        card.id: seat
        for seat in Seat
        for card in tail.initial_hands[int(seat)]
    }
    projects = attach_project_lifecycle_fragment_v1(
        build.model,
        owner_of_card=build.initial_owner_variables,
        fixed_card_owners=tuple(
            known_owners.get(card.id) for card in FULL_DECK
        ),
        contract=contract,
        dealer=dealer,
        trump=trump,
        domain=rules.project_domain,
        name="proof_seed_reachable_projects",
        second_trick_leader=(
            build._extract.winner_seats[0]
            if rules.project_domain.tie_break_order == "second_trick_reveal"
            else None
        ),
        compact_fixed_states=True,
        tighten_value_bounds=True,
    )
    for team in Team:
        build.model.add(
            projects.combined_awarded_value[int(team)]
            == seed.awarded_project_points[int(team)]
        )
        build.model.add(
            projects.baloot_team[int(team)] * projects.baloot_value
            == seed.baloot_points[int(team)]
        )
    lowerings = {
        predicate.predicate_id: manifest.lower_predicate(
            predicate,
            CpSatProjectPredicateContextV1(
                build.model, "proof_seed_reachable_projects", projects
            ),
        )
        for predicate in project_predicates
    }
    _enforce_hard_stage_projection(
        build.model,
        scenario,
        stage="projects",
        predicate_literals={
            predicate_id: lowering.literal
            for predicate_id, lowering in lowerings.items()
        },
    )


def _decision_status_vector_v1(
    tail: Any,
    *,
    prefix_length: int,
    acting_seat: Seat,
) -> tuple[int, ...]:
    decision = tail.at_prefix(prefix_length)
    if decision.next_actor != acting_seat:
        raise ValueError("tail decision actor disagrees with the scenario")
    statuses = [8] * len(FULL_DECK)
    active_cards: set[int] = set()
    for seat in Seat:
        for card in decision.remaining_hands[int(seat)]:
            if card.id in active_cards:
                raise ValueError("tail decision state duplicates an active card")
            active_cards.add(card.id)
            statuses[card.id] = (int(seat) - int(acting_seat)) % 4
    for play in decision.current_trick:
        if play.card.id in active_cards:
            raise ValueError("tail decision state duplicates an active card")
        active_cards.add(play.card.id)
        statuses[play.card.id] = 4 + (int(play.actor) - int(acting_seat)) % 4
    return tuple(statuses)


def _unique_root_suit_release_guidance_v1(
    scenario: ResearchScenarioV1,
    proof_request: ScenarioProofRequestV1,
) -> tuple[int, int, tuple[int, int]] | None:
    if proof_request.mode != "exact" or proof_request.visibility != "complete_state":
        return None
    decision = dict(scenario.decision_point["selector"])
    trick_index = int(decision["trickIndex"])
    position = int(decision.get("position", 0))
    predicates = tuple(
        predicate for predicate in scenario.predicates() if predicate.hardness == "hard"
    )
    unique = any(
        predicate.kind == "proof.winning_root_action_count"
        and int(predicate.arguments["minimum"]) == 1
        and int(predicate.arguments["maximum"]) == 1
        for predicate in predicates
    )
    root_is_ace = any(
        predicate.kind == "play.action_matches"
        and int(predicate.arguments["trickIndex"]) == trick_index
        and int(predicate.arguments["position"]) == position
        and "ace" in tuple(dict(predicate.arguments["selector"]).get("ranks", ()))
        for predicate in predicates
    )
    root_proof_is_ace = any(
        predicate.kind == "proof.winning_root_action_matches"
        and "ace" in tuple(dict(predicate.arguments["selector"]).get("ranks", ()))
        for predicate in predicates
    )
    future = tuple(
        sorted(
            int(predicate.arguments["trickIndex"])
            for predicate in predicates
            if predicate.kind == "play.trick_winner_matches"
            and predicate.arguments.get("team") == proof_request.target_team
            and int(predicate.arguments["trickIndex"]) > trick_index
            and set(dict(predicate.arguments["selector"]).get("ranks", ()))
            <= {"seven", "eight", "nine"}
        )
    )
    if (
        unique
        and root_is_ace
        and root_proof_is_ace
        and future == (trick_index + 1, trick_index + 2)
    ):
        return trick_index, position, future
    return None


def _apply_unique_root_suit_release_guidance_v1(
    build: Any,
    guidance: tuple[int, int, tuple[int, int]],
    acting_seat: Seat,
) -> None:
    trick_index, position, future = guidance
    root = build.predicate_context.tricks[trick_index].actions[position]
    first_compiled_trick = 8 - len(build.selected_play_indexes) // 4
    decision_step = (trick_index - first_compiled_trick) * 4 + position

    remaining_aces = []
    for ace in (card for card in FULL_DECK if card.rank == Rank.ACE):
        owned = build.model.new_bool_var(f"root_ace_{ace.id}_owned")
        build.model.add(
            build.initial_owner_variables[ace.id] == int(acting_seat)
        ).only_enforce_if(owned)
        build.model.add(
            build.initial_owner_variables[ace.id] != int(acting_seat)
        ).only_enforce_if(owned.Not())
        played_before_flags = []
        for step, played_card in enumerate(
            build.selected_play_indexes[:decision_step]
        ):
            played = build.model.new_bool_var(
                f"root_ace_{ace.id}_played_before_{step}"
            )
            build.model.add(played_card == ace.id).only_enforce_if(played)
            build.model.add(played_card != ace.id).only_enforce_if(played.Not())
            played_before_flags.append(played)
        played_before = build.model.new_bool_var(
            f"root_ace_{ace.id}_played_before"
        )
        build.model.add_max_equality(played_before, played_before_flags)
        selected = build.selected_card_literals[ace.id]
        remaining = build.model.new_bool_var(f"root_ace_{ace.id}_remaining")
        build.model.add_bool_and(
            [owned, selected, played_before.Not()]
        ).only_enforce_if(remaining)
        build.model.add_bool_or(
            [owned.Not(), selected.Not(), played_before]
        ).only_enforce_if(remaining.Not())
        remaining_aces.append(remaining)
    build.model.add(sum(remaining_aces) == 1)

    same_suit_pairs = tuple(
        (left.id, right.id)
        for left in FULL_DECK
        for right in FULL_DECK
        if left.suit == right.suit
    )
    for future_trick in future:
        winner = build.predicate_context.tricks[future_trick].winner
        build.model.add_allowed_assignments([root.card, winner.card], same_suit_pairs)
    release_rows = tuple(
        (actor, root_card.id, later_card.id)
        for actor in range(4)
        for root_card in FULL_DECK
        for later_card in FULL_DECK
        if actor != int(acting_seat) or later_card.suit != root_card.suit
    )
    for future_trick in future:
        for action in build.predicate_context.tricks[future_trick].actions:
            build.model.add_allowed_assignments(
                [action.actor, root.card, action.card], release_rows
            )


def _tail_policy_prefix_screen_v1(
    *,
    tail: OrToolsPlayTailResultV1,
    tail_offset: int,
    rules: RuleBook,
    contract: Contract,
    trump: Suit | None,
    first_leader: Seat,
    policy_prefix_steps: int,
    policy_id: str,
    deadline: float,
    maximum_conflicts: int,
    seed: int,
    cancel_check: Callable[[], bool],
    native_conversion: bool,
) -> tuple[str, ...]:
    """Cut a tail only if neither legal lock mode can reach it."""

    def preserve_tail_cards(build) -> None:
        if not _preserve_tail_cards_before_decision(
            build, tail, tail_offset, cancel_check,
        ):
            raise PlayTailSolveStopped(
                "unknown",
                evidence={"solverStatus": "unknown"},
            )

    statuses = []
    for locked in (False, True):
        try:
            OrToolsPlayTailStageV1(native_conversion=native_conversion).solve(
                cards=FULL_DECK,
                ownership={
                    card: seat
                    for seat in Seat
                    for card in tail.initial_hands[int(seat)]
                },
                contract=contract,
                trump=trump,
                first_leader=first_leader,
                play_domain=rules.play_domain,
                structural_domain=rules.structural_domain,
                scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
                tail=tail.plays[:4],
                locked=locked,
                play_step_count=tail_offset + 4,
                policy_prefix_steps=policy_prefix_steps,
                policy_id=policy_id,
                construction_plan_hash=canonical_sha256({
                    "schemaVersion": 1,
                    "stage": "tail_policy_reachability_v1",
                    "tailEvidenceHash": tail.evidence_hash,
                    "locked": locked,
                    "policyId": policy_id,
                    "policyPrefixSteps": policy_prefix_steps,
                }),
                model_configurator=preserve_tail_cards,
                maximum_time_seconds=max(0.001, deadline - time.monotonic()),
                maximum_conflicts=maximum_conflicts,
                random_seed=seed,
                cancel_check=cancel_check,
            )
            statuses.append("feasible")
            break
        except PlayTailSolveStopped as stopped:
            statuses.append(stopped.solver_status)
            if stopped.solver_status != "infeasible":
                break
    return tuple(statuses)


class OrToolsTailFirstCandidateCompilerV1:
    """Find, prove, and backward-lift one reachable late-play state."""

    def __init__(
        self,
        *,
        rules: RuleBook,
        plan: TailFirstStagePlanV1,
        proof_request: ScenarioProofRequestV1,
        maximum_proof_nodes: int,
        maximum_proof_depth: int,
        maximum_line_count_limbs: int,
        maximum_conflicts: int,
        maximum_prefix_lift_attempts: int,
        excluded_decision_status_vectors: Sequence[Sequence[int]] = (),
        minimum_decision_status_hamming_distance: int = 1,
        candidate_partition: tuple[int, int, int] | None = None,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        native_cp_sat_scoring: bool = False,
        tactical_detector_profile: TacticalDetectorProfileV1 | None = None,
        policy_prefix_steps: int = 0,
        policy_id: str = "observer_safe_trick_control_v1",
        require_optimal_auction_actions: bool = False,
    ) -> None:
        self.rules = rules
        self.plan = plan
        self.proof_request = proof_request
        self.maximum_proof_nodes = maximum_proof_nodes
        self.maximum_proof_depth = maximum_proof_depth
        self.maximum_line_count_limbs = maximum_line_count_limbs
        if type(maximum_conflicts) is not int or maximum_conflicts < 1:
            raise ValueError("maximum_conflicts must be a positive integer")
        self.maximum_conflicts = maximum_conflicts
        if (
            type(maximum_prefix_lift_attempts) is not int
            or maximum_prefix_lift_attempts < 1
        ):
            raise ValueError("maximum_prefix_lift_attempts must be positive")
        self.maximum_prefix_lift_attempts = maximum_prefix_lift_attempts
        self.excluded_decision_status_vectors = tuple(
            tuple(vector) for vector in excluded_decision_status_vectors
        )
        self.minimum_decision_status_hamming_distance = (
            minimum_decision_status_hamming_distance
        )
        self.candidate_partition = _validated_candidate_partition_v1(
            candidate_partition
        )
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self.native_cp_sat_scoring = bool(native_cp_sat_scoring)
        self.tactical_detector_profile = tactical_detector_profile
        if type(policy_prefix_steps) is not int or policy_prefix_steps < 0:
            raise ValueError("policy_prefix_steps must be a non-negative integer")
        self.policy_prefix_steps = policy_prefix_steps
        self.policy_id = policy_id
        self.require_optimal_auction_actions = bool(require_optimal_auction_actions)
        self._candidate_refinement: Callable[..., Mapping[str, Any] | None] | None = None
        quality_combinations = () if self.native_cp_sat_scoring else self.plan.boundary_combinations
        self._quality_rejected_complete_plays = {
            combination: [] for combination in quality_combinations
        }
        self._quality_rejected_play_zero_actions = {
            combination: [] for combination in quality_combinations
        }
        # shortcut: source cuts are factory-local; cross-campaign reuse needs the original verified miss.
        self._quality_rejected_initial_decks: dict[tuple, list[tuple[int, ...]]] = {}
        self.complete_state_graph_compiler = OrToolsCompleteStateGraphCompilerV1(
            play_domain=self.rules.play_domain,
            scoring_domain=ScoringDomainIRV1.from_rules(self.rules.data),
            structural_domain=self.rules.structural_domain,
            project_domain=self.rules.project_domain,
            native_conversion=self.native_cp_sat_scoring,
        )

    def _reject_quality_candidate(
        self,
        candidate: OrToolsNeutralCandidateV1,
        *,
        reject_play_zero_observation: bool = False,
        reject_initial_input: bool = False,
    ) -> None:
        if not isinstance(candidate, OrToolsNeutralCandidateV1):
            raise TypeError("quality rejection requires an OR-Tools candidate")
        if candidate.origin != "search":
            raise ValueError("quality rejection requires a search candidate")
        stage_plan = candidate.trace.get("tailFirstStagePlan")
        if not isinstance(stage_plan, Mapping) or stage_plan.get("sha256") != self.plan.sha256:
            raise ValueError("quality rejection candidate belongs to another tail plan")
        combination = (
            str(candidate.trace.get("dealer")),
            str(candidate.trace.get("contract")),
            str(candidate.trace.get("buyer")),
            candidate.trace.get("trumpSuit"),
        )
        if combination not in (self.plan.boundary_combinations if self.native_cp_sat_scoring
                               else self._quality_rejected_complete_plays):
            raise ValueError("quality rejection candidate has an unsupported boundary")
        quality_key = combination
        if self.native_cp_sat_scoring:
            scores = candidate.trace.get("matchScore")
            if (not isinstance(scores, (list, tuple)) or len(scores) != 2
                    or any(type(value) is not int or value < 0 for value in scores)):
                raise ValueError("quality rejection requires a valid starting match score")
            quality_key = (*combination, *scores)
        plays = candidate.trace.get("play", {}).get("plays", ())
        vector = tuple(Card.parse(play["card"]).id for play in plays)
        if len(vector) != 32 or sorted(vector) != list(range(32)):
            raise ValueError("quality rejection candidate must contain one complete play")
        if reject_initial_input:
            deck = candidate.no_good_vector
            if (not self.native_cp_sat_scoring
                    or candidate.trace.get("roundNumber", 1) != 1
                    or tuple(candidate.trace.get("deckOrder", ())) != deck
                    or len(deck) != 32 or any(type(card) is not int for card in deck)
                    or sorted(deck) != list(range(32))):
                raise ValueError("initial-input rejection requires its native complete deal")
            rejected_decks = self._quality_rejected_initial_decks.setdefault(quality_key, [])
            if deck not in rejected_decks:
                rejected_decks.append(deck)
        if Contract.parse(combination[1]).effective is not Contract.HOKUM:
            rejected = self._quality_rejected_complete_plays.setdefault(quality_key, [])
            if vector not in rejected:
                rejected.append(vector)
        if reject_play_zero_observation:
            first = plays[0]
            actor = Seat.parse(first["actor"])
            initial_hands = candidate.trace.get("play", {}).get("initialHands", {})
            cut = PlayZeroObservationActionNoGoodV1.build(
                actor=actor,
                actor_cards=initial_hands.get(actor.name.lower(), ()),
                selected_card=first["card"],
                ekka=bool(first.get("ekka", False)),
            )
            rejected = self._quality_rejected_play_zero_actions.setdefault(quality_key, [])
            if cut not in rejected:
                rejected.append(cut)

    def __call__(
        self,
        scenario: ResearchScenarioV1,
        excluded_decks: Sequence[Sequence[int]],
        *,
        seed: int,
        maximum_seconds: float,
        maximum_conflicts: int,
    ) -> OrToolsCandidateAttemptV1:
        if maximum_conflicts != self.maximum_conflicts:
            raise ValueError("tail-first conflict budget must match its frozen plan")
        validate_native_delayed_payoff_scope_v1(scenario, self.proof_request,
            self.tactical_detector_profile, native_authority=self.native_cp_sat_scoring)
        deadline = time.monotonic() + float(maximum_seconds)

        def budget_stopped() -> bool:
            return time.monotonic() >= deadline or (self.cancel_check is not None and self.cancel_check())
        starting_match_score = (0, 0)
        initial_dealer = None
        partial_initial_input_choice = False
        excluded_initial_inputs = []
        if self.native_cp_sat_scoring:
            try:
                selected_dealer, starting_match_score, partial_initial_input_choice = _required_search_initial_inputs_v3(
                    scenario, rules=self.rules, deadline=deadline, maximum_conflicts=maximum_conflicts,
                    cancel_check=self.cancel_check)
            except InformationSetCompilationStopped as stopped:
                return OrToolsCandidateAttemptV1("incomplete", None, stopped.reason,
                    partial_receipt=stopped.partial_receipt)
            if starting_match_score is None:
                return OrToolsCandidateAttemptV1("infeasible", None, "initial_score_infeasible")
            if any(predicate.hardness == "hard" and predicate.kind == "auction.dealer"
                    for predicate in scenario.predicates()):
                initial_dealer = selected_dealer
        manifest = shared_scenario_predicate_catalog_lowering_manifest_v2()
        play_predicates = tuple(
            predicate
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and manifest.definition(predicate.kind).stage == "playPrefix"
        )
        required_project_predicates = tuple(
            predicate
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and predicate.kind == "project.exists"
            and _hard_expression_requires_predicate(
                scenario,
                predicate.predicate_id,
                maximum_conflicts=maximum_conflicts,
            )
        )
        flexible_boundary_score = any(
            predicate.hardness == "hard"
            and manifest.definition(predicate.kind).stage == "outcome"
            for predicate in scenario.predicates()
        )
        project_stage_predicates = tuple(
            predicate
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and manifest.definition(predicate.kind).stage == "projects"
        )
        escalation_stage_predicates = tuple(
            predicate
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and manifest.definition(predicate.kind).stage == "escalation"
        )
        auction_stage_predicates = tuple(
            predicate
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and manifest.definition(predicate.kind).stage == "auction"
        )
        outcome_stage_predicates = tuple(
            predicate
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and manifest.definition(predicate.kind).stage == "outcome"
        )
        native_full_round_outcome = bool(
            self.native_cp_sat_scoring and outcome_stage_predicates
            # With an opening policy, the full play model is already large.
            # The proof-boundary model certifies projects and outcome natively.
            and self.policy_prefix_steps == 0
            and not any(
                predicate.hardness == "hard"
                and predicate.kind == "state.match_score"
                for predicate in scenario.predicates()
            )
        )
        # Dominant authored suits outlast the opening prefix. Construct that
        # reachable prefix once instead of repeatedly rejecting provisional tails.
        counted_seats = {
            Seat.parse(predicate.arguments["seat"])
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and predicate.kind == "deal.suit_count"
            and predicate.arguments.get("handStage") == "post_purchase"
            and predicate.arguments["minimum"] == predicate.arguments["maximum"]
            and predicate.arguments["minimum"] > self.plan.tail_offset // 4
            and _hard_expression_requires_predicate(
                scenario, predicate.predicate_id, maximum_conflicts=maximum_conflicts,
            )
        } if self.native_cp_sat_scoring else set()
        reachable_full_round = counted_seats == set(Seat)
        construction_invariants = {}
        if self.native_cp_sat_scoring and not reachable_full_round:
            for predicate in play_predicates:
                if (predicate.kind != "play.trick_card_count"
                        or int(predicate.arguments["trickIndex"]) >= self.plan.tail_offset // 4
                        or not _hard_expression_requires_predicate(
                            scenario, predicate.predicate_id,
                            maximum_conflicts=maximum_conflicts,
                            maximum_seconds=max(0.001, deadline - time.monotonic()),
                            cancel_check=self.cancel_check,
                        )):
                    continue
                invariant = _trick_card_count_invariant_v1(
                    predicate, manifest, deadline=deadline,
                    maximum_conflicts=maximum_conflicts, cancel_check=self.cancel_check,
                )
                if invariant is not None:
                    construction_invariants[predicate.predicate_id] = invariant
        construction_play_predicates = tuple(
            predicate for predicate in play_predicates
            if predicate.predicate_id not in construction_invariants
        )
        # V3 otherwise constructs only the unknown late cards here. The tail enforces
        # policy actions it contains; the lift enforces the earlier prefix.
        # The boundary and lift own full-round scoring and auction quality.
        selective_tail = bool(
            self.native_cp_sat_scoring
            and not reachable_full_round
            and self.plan.tail_offset is not None
            and all(
                predicate.kind in {
                    "play.action_matches",
                    "play.trick_card_count",
                    "play.trick_winner_matches",
                }
                and int(predicate.arguments["trickIndex"])
                >= self.plan.tail_offset // 4
                for predicate in construction_play_predicates
            )
        )
        if not selective_tail:
            construction_invariants = {}
        # Only selective tails lack the earlier tricks needed to bind current raw points.
        tail_state_predicates = tuple(
            predicate
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and (
                predicate.kind == "state.legal_action_count"
                or (not selective_tail and predicate.kind == "state.current_raw_points")
            )
        )
        tail_construction_predicates = (
            *(construction_play_predicates if selective_tail else play_predicates),
            *tail_state_predicates,
        )
        # Keep the selective model small. The mandatory native boundary owns
        # project awards and complete-round scoring before any proof or lift.
        native_full_round_outcome = native_full_round_outcome and not selective_tail
        late_only_play = bool(play_predicates) and all(
            int(predicate.arguments.get("trickIndex", -1)) >= self.plan.tail_offset // 4
            for predicate in play_predicates
        )
        inline_auction_quality = bool(
            self.require_optimal_auction_actions
            and not selective_tail
            and not late_only_play
        )
        # Bind initial-hand requirements before either model fixes the hands;
        # remaining-card predicates still belong to the reached play state.
        tail_deal_predicates = tuple(
            predicate
            for predicate in scenario.predicates()
            if self.native_cp_sat_scoring
            and predicate.hardness == "hard"
            and manifest.definition(predicate.kind).stage == "deal"
            and not predicate.kind.startswith("deal.remaining_")
            and predicate.arguments.get("handStage") != "remaining"
        )
        score_floor = None
        if not selective_tail and not native_full_round_outcome:
            for outcome in outcome_stage_predicates:
                if (outcome.kind != "outcome.team_round"
                        or outcome.arguments.get("status") != "won"
                        or not _hard_expression_requires_predicate(
                            scenario, outcome.predicate_id,
                            maximum_conflicts=maximum_conflicts,
                        )):
                    continue
                target = Team[str(outcome.arguments["team"]).upper()]
                opposing_awards = [
                    int(project.arguments["value"])
                    for project in required_project_predicates
                    if project.arguments.get("state") == "awarded"
                    and Team[str(project.arguments["team"]).upper()] == target.other
                ]
                if opposing_awards:
                    score_floor = (target, max(opposing_awards))
                break
        tail_receipt_predicates = (
            *tail_deal_predicates,
            *(auction_stage_predicates if inline_auction_quality else ()),
            *tail_construction_predicates,
            *(
                (
                    *project_stage_predicates,
                    *escalation_stage_predicates,
                    *outcome_stage_predicates,
                )
                if native_full_round_outcome
                else ()
            ),
            *(escalation_stage_predicates if self.native_cp_sat_scoring and not native_full_round_outcome else ()),
        )
        proof_predicates = tuple(
            predicate
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and manifest.definition(predicate.kind).stage == "proof"
        )
        exact_unique_root_threshold = any(
            predicate.kind == "proof.winning_root_action_count"
            and int(predicate.arguments["minimum"]) == 1
            and int(predicate.arguments["maximum"]) == 1
            for predicate in proof_predicates
        ) and all(
            predicate.kind
            in {
                "proof.winning_root_action_count",
                "proof.winning_root_action_matches",
            }
            for predicate in proof_predicates
        ) and all(
            _hard_expression_requires_predicate(scenario, predicate.predicate_id)
            for predicate in proof_predicates
        )
        winning_root_selectors = tuple(
            CardSelectorV1.build(predicate.arguments["selector"])
            for predicate in proof_predicates
            if predicate.kind == "proof.winning_root_action_matches"
        )
        first_winning_count_range = _first_winning_count_range_required_v1(
            scenario, proof_predicates
        )
        point_margin_boundary_search = bool(
            self.native_cp_sat_scoring
            and selective_tail
            and (exact_unique_root_threshold or first_winning_count_range)
            and self.proof_request.mode == "exact"
            and self.proof_request.visibility == "complete_state"
            and self.proof_request.utility == "awarded_point_margin"
        )
        flexible_boundary_score = flexible_boundary_score or point_margin_boundary_search
        suit_release_guidance = _unique_root_suit_release_guidance_v1(
            scenario, self.proof_request
        )
        suit_release_guidance_active = suit_release_guidance is not None
        action_kinds = {
            "play.action_matches",
            "play.trick_card_count",
            "play.trick_winner_matches",
        }
        exact_ownership = {
            int(predicate.arguments["card"]): Seat.parse(predicate.arguments["seat"])
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and predicate.kind == "deal.card_owner"
            and predicate.arguments.get("handStage") in {"initial", "post_purchase"}
            and (not self.native_cp_sat_scoring or _hard_expression_requires_predicate(
                scenario, predicate.predicate_id, maximum_conflicts=maximum_conflicts,
            ))
        }
        required_remaining_ownership = {
            int(predicate.arguments["card"]): Seat.parse(predicate.arguments["seat"])
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and predicate.kind == "deal.remaining_card_owner"
            and _hard_expression_requires_predicate(
                scenario,
                predicate.predicate_id,
                maximum_conflicts=maximum_conflicts,
            )
        }
        excluded_boundaries: list[TailProofBoundaryNoGoodV1] = []
        excluded_proof_states: list[TailProofStateNoGoodV1] = []
        required_history: TailProofBoundaryNoGoodV1 | None = None
        required_proof_state: TailProofStateNoGoodV1 | None = None
        excluded_tails: dict[
            tuple[str, str, str, str | None], list[tuple[int, ...]]
        ] = {combination: [] for combination in self.plan.boundary_combinations}
        excluded_complete_plays: dict[
            tuple[str, str, str, str | None], list[tuple[int, ...]]
        ] = {
            combination: list(self._quality_rejected_complete_plays.get(
                (*combination, *starting_match_score) if self.native_cp_sat_scoring else combination, ()))
            for combination in self.plan.boundary_combinations
        }
        excluded_play_zero_actions = {
            combination: list(self._quality_rejected_play_zero_actions.get(
                (*combination, *starting_match_score) if self.native_cp_sat_scoring else combination, ()))
            for combination in self.plan.boundary_combinations
        }
        excluded_initial_decks = {
            combination: list(self._quality_rejected_initial_decks.get(
                (*combination, *starting_match_score), ()))
            for combination in self.plan.boundary_combinations
        }
        rejected_decision_states: list[tuple[int, ...]] = []
        timings = {
            "tailConstruction": 0.0,
            "tailModelBuild": 0.0,
            "tailModelConfigure": 0.0,
            "tailSolver": 0.0,
            "prefixReachability": 0.0,
            "proofBoundary": 0.0,
            "proofBoundaryModelBuild": 0.0,
            "proofBoundarySolver": 0.0,
            "graph": 0.0,
            "proof": 0.0,
            "prefixLift": 0.0,
        }
        counters = {
            "tailCandidates": 0,
            "prefixReachabilityCuts": 0,
            "prefixReachabilityTailCuts": 0,
            "proofBoundaryAttempts": 0,
            "proofBoundarySolverAttempts": 0,
            "proofPredicateRejections": 0,
            "proofRejectedZeroWinningRoots": 0,
            "proofRejectedOneWinningRoot": 0,
            "proofRejectedMultipleWinningRoots": 0,
            "proofStateProbes": 0,
            "rawRootCacheHits": 0,
            "scoreProbeCacheHits": 0,
            "prefixLiftAttempts": 0,
        }
        raw_root_cache: dict[str, bool] = {}
        score_probe_context: str | None = None
        score_probe_cache: dict[int, tuple[str, OrToolsGraphProofOutcomeV1]] = {}
        latest_graph_receipt: Mapping[str, Any] | None = None
        lift_failures: list[Mapping[str, Any]] = []

        def incomplete_proof_attempt(reason: str, stage: str) -> OrToolsCandidateAttemptV1:
            partial_core = {
                "schemaVersion": 1,
                "status": "incomplete",
                "stage": stage,
                "exhaustionReason": reason,
                "evidence": {
                    "prefixLiftFailures": list(lift_failures),
                    "stageTimings": dict(timings),
                    "stageCounters": dict(counters),
                    **(
                        {"graphReceipt": dict(latest_graph_receipt)}
                        if latest_graph_receipt is not None else {}
                    ),
                },
            }
            return OrToolsCandidateAttemptV1(
                "incomplete", None, reason, stage_timings=timings,
                partial_receipt={**partial_core, "sha256": canonical_sha256(partial_core)},
            )

        def report(stage: str, evidence: Mapping[str, Any] | None = None) -> None:
            if self.progress_callback is not None:
                self.progress_callback(
                    {
                        "currentStage": stage,
                        "stageTimings": dict(timings),
                        "stageCounters": dict(counters),
                        **dict(evidence or {}),
                    }
                )

        def evaluate_proof_seed(
            seed_value: CompleteStateGraphSeedV1,
            *,
            probe: bool,
        ) -> tuple[
            str,
            OrToolsGraphProofOutcomeV1 | None,
            Any | None,
            Mapping[str, Any],
            str | None,
        ]:
            nonlocal latest_graph_receipt
            if probe:
                counters["proofStateProbes"] += 1
            graph_started = time.perf_counter()
            report("graph", {"proofStateProbe": probe})
            compilation = self.complete_state_graph_compiler.compile(
                seed_value,
                self.proof_request,
                expected_rules_hash=self.rules.sha256,
                maximum_nodes=self.maximum_proof_nodes,
                maximum_depth=self.maximum_proof_depth,
                maximum_seconds=max(0.001, deadline - time.monotonic()),
                cancelled=budget_stopped,
            )
            timings["graph"] += time.perf_counter() - graph_started
            latest_graph_receipt = compilation.receipt.to_dict()
            if not compilation.complete or compilation.graph is None:
                return (
                    "incomplete",
                    None,
                    None,
                    {},
                    "tail_proof_graph_budget" if time.monotonic() >= deadline
                    else "tail_proof_graph_" + str(compilation.receipt.exhaustion_reason),
                )
            proof_started = time.perf_counter()
            report("proof", {"proofStateProbe": probe})
            native_predicates = tuple(
                predicate
                for predicate in scenario.predicates()
                if (predicate.hardness == "hard" or self.native_cp_sat_scoring
                    and predicate.kind in {"tactic.applicable", "tactic.verified"}
                    and (predicate.arguments["kind"] in {"trump_control", "partner_enablement", "entry_unblocking"}
                         or predicate.arguments["kind"] in DELAYED_PAYOFF_TACTIC_IDS))
                and manifest.definition(predicate.kind).stage in {"proof", "tactics"}
            )
            native_same_solve_kinds = {
                "proof.winning_root_action_matches",
                "proof.winning_root_action_count",
                "proof.winning_root_card_rank_count",
                "proof.principal_variation_trick_winner",
                "proof.principal_variation_trick_team_rank_count",
            }
            native_same_solve_supported = all(
                predicate.kind in native_same_solve_kinds
                or (
                    predicate.kind == "proof.winning_continuation_line_count"
                    and self.proof_request.selected_action is not None
                    and self.proof_request.line_universe
                    == predicate.arguments["lineUniverse"]
                )
                or predicate.kind.startswith("tactic.")
                for predicate in native_predicates
            )

            def configure_tactics(bindings):
                root_node = next(node for node in compilation.graph.nodes
                                 if node.node_id == compilation.graph.root_node_id)
                root_edges = root_node.edges
                raw_payoffs = ()
                delayed_control = (isinstance(self.tactical_detector_profile, TacticalDetectorProfileV1)
                    and self.tactical_detector_profile.native_delayed_gain_metric is not None
                    and len(root_edges) > 1
                    and any(item.kind in {"tactic.applicable", "tactic.verified"}
                        and item.arguments["kind"] in DELAYED_PAYOFF_TACTIC_IDS for item in native_predicates))
                if (self.native_cp_sat_scoring and self.proof_request.mode == "exact"
                        and any(item.kind in {"tactic.applicable", "tactic.verified"} and (
                                item.arguments["kind"] in {"partner_enablement", "entry_unblocking"} and len(root_edges) > 1
                                or item.arguments["kind"] in DELAYED_PAYOFF_TACTIC_IDS
                                and (delayed_control or item.kind == "tactic.verified" and item.hardness == "report"))
                                for item in native_predicates)):
                    raw_payoffs = bindings.bind_root_raw_payoffs(
                        tuple(self.rules.play_domain.points(Card(card), seed_value.contract, seed_value.trump)
                              for card in range(32)),
                        ScoringDomainIRV1.from_rules(self.rules.data).last_trick_bonus_raw,
                        tuple(play.card.id for play in seed_value.current_trick),
                        **({"first_leader": int(seed_value.first_leader)} if len(root_edges) > 1 and (delayed_control or any(
                            item.kind in {"tactic.applicable", "tactic.verified"}
                            and item.arguments["kind"] == "entry_unblocking" for item in native_predicates)) else {}),
                        **({"delayed_control": True} if delayed_control else {}),
                    )
                trump_suffix = tuple(zip(bindings.principal_action_card_variables,
                    bindings.principal_action_actor_variables, bindings.principal_action_present_variables,
                    bindings.principal_action_trick_indexes, strict=True)) if any(
                        item.kind in {"tactic.applicable", "tactic.verified"}
                        and item.arguments["kind"] == "trump_control" for item in native_predicates) else ()
                if any(item.kind == 'proof.principal_variation_trick_team_rank_count' for item in native_predicates):
                    bindings = bindings.with_current_trick(sum(seed_value.base_tricks), tuple(
                        (play.card.id, int(play.actor)) for play in seed_value.current_trick))
                def fixed_bool(name: str, value: bool):
                    variable = bindings.model.new_bool_var(name)
                    bindings.model.add(variable == int(value))
                    return variable

                root_actions = tuple(
                    CpSatActionVariablesV1(
                        f"ROOT-{edge.action_id}",
                        bindings.root_card_variables[index],
                        bindings.root_actor_variables[index],
                        bindings.root_owner_variables[index],
                        fixed_bool(
                            f"root_ekka_{index}", edge.action_id >= 32
                        ),
                        fixed_bool(f"root_present_{index}", True),
                        bindings.root_winning_variables[index],
                        bindings.root_value_variables[index],
                    )
                    for index, edge in enumerate(root_edges)
                )
                proof_plan = SimpleNamespace(
                    completion_status="exact_complete",
                    solver_status="optimal",
                    root_actions=tuple(
                        SimpleNamespace(
                            action_code=edge.action_code,
                            satisfies=False,
                        )
                        for edge in root_edges
                    ),
                    winning_root_action_count=None,
                    selected_action=self.proof_request.selected_action,
                    sha256=compilation.graph.sha256,
                    principal_variation=(),
                )
                assert root_node.actor_team is not None
                root_team = Team[root_node.actor_team.upper()]
                target_team = Team[self.proof_request.target_team.upper()]
                context = SimpleNamespace(
                    model=bindings.model,
                    context_id=(
                        "tail-live-proof-" + compilation.graph.sha256[:16]
                    ),
                    sha256=canonical_sha256(
                        {
                            "graphHash": compilation.graph.sha256,
                            **({"tacticalDetectorProfileHash": self.tactical_detector_profile.sha256} if self.native_cp_sat_scoring and self.tactical_detector_profile is not None else {}),
                            "predicateIds": [
                                item.predicate_id
                                for item in native_predicates
                            ],
                        }
                    ),
                    proof=proof_plan,
                    native_proof_bindings=True,
                    native_cp_sat_authority=self.native_cp_sat_scoring,
                    tactical_detector_profile=self.tactical_detector_profile,
                    root_raw_payoff_variables=raw_payoffs,
                    principal_variation_tricks=(),
                    tactical_evidence=(),
                    root_actions=root_actions,
                    root_winning_variables=bindings.root_winning_variables,
                    root_value_variables=bindings.root_value_variables,
                    causal_gain_scale=compilation.graph.utility_scale,
                    causal_gain_terminal_awarded_scope=(self.proof_request.mode == "exact"
                        and self.proof_request.visibility == "complete_state" and compilation.graph.scope == "exact"
                        and self.proof_request.utility == "awarded_point_margin"),
                    root_actor_seat=int(seed_value.next_actor),
                    root_actor_team=int(root_team),
                    proof_utility=self.proof_request.utility,
                    decision_kind=str(scenario.decision_point["kind"]),
                    selected_root_action_index=(
                        bindings.selected_root_action_index
                    ),
                    root_actor_is_target=root_team == target_team,
                    root_actor_is_target_variable=(
                        bindings.root_actor_is_target_variable
                    ),
                    target_seat_variable=bindings.target_seat_variable,
                    card_strength_by_id=tuple(
                        self.rules.play_domain.strength(
                            card, seed_value.contract, seed_value.trump
                        )
                        for card in sorted(FULL_DECK, key=lambda item: item.id)
                    ),
                    first_principal_trick_winner_team=None,
                    first_principal_trick_winner_team_variable=(
                        bindings.first_principal_trick_winner_team_variable
                    ),
                    trump_suit=seed_value.trump,
                    trump_suit_variable=bindings.trump_suit_variable,
                    base_tricks=seed_value.base_tricks,
                    base_trick_variables=bindings.base_trick_variables,
                    principal_trick_winner_seat_variables=(
                        bindings.principal_trick_winner_seat_variables
                    ),
                    principal_trick_winning_card_variables=(
                        bindings.principal_trick_winning_card_variables
                    ),
                    principal_trick_indexes=bindings.principal_trick_indexes,
                    principal_action_card_variables=(
                        bindings.principal_action_card_variables
                    ),
                    principal_action_actor_variables=(
                        bindings.principal_action_actor_variables
                    ),
                    principal_action_present_variables=(
                        bindings.principal_action_present_variables
                    ),
                    principal_action_trick_indexes=(
                        bindings.principal_action_trick_indexes
                    ),
                    trump_control_suffix_actions=trump_suffix,
                    principal_variation_has_continuation_variable=(
                        bindings.principal_variation_has_continuation_variable
                    ),
                    proof_complete=bindings.proof_complete_variable,
                    continuation_line_digits=bindings.continuation_line_digits,
                    continuation_line_universe=bindings.continuation_line_universe,
                )
                return {
                    predicate.predicate_id: lower_native_proof_predicate_v3(
                        predicate, context, manifest=manifest,
                        root_selection_variables=bindings.root_selection_variables,
                    )
                    for predicate in native_predicates
                }

            try:
                proof_value = OrToolsScenarioGraphProofV1().prove(
                    compilation.graph,
                    self.proof_request,
                    maximum_seconds=max(0.001, deadline - time.monotonic()),
                    maximum_line_count_limbs=self.maximum_line_count_limbs,
                    cancel_check=budget_stopped,
                    tactic_configure=(
                        configure_tactics
                        if native_predicates and native_same_solve_supported
                        else None
                    ),
                    native_root_only=bool(native_predicates) and all(
                        predicate.kind in {
                            "proof.winning_root_action_matches",
                            "proof.winning_root_action_count",
                            "proof.winning_root_card_rank_count",
                        }
                        for predicate in native_predicates
                    ),
                    native_line_count=(
                        native_same_solve_supported
                        and any(
                            predicate.kind == "proof.winning_continuation_line_count"
                            for predicate in native_predicates
                        )
                    ),
                )
            except InterruptedError:
                return "incomplete", None, compilation.graph, {}, (
                    "tail_proof_budget" if time.monotonic() >= deadline else "cancelled"
                )
            timings["proof"] += time.perf_counter() - proof_started
            if proof_value.completion_status == "proof_incomplete":
                return (
                    "incomplete",
                    proof_value,
                    compilation.graph,
                    {},
                    "tail_proof_budget",
                )
            if proof_value.claim_satisfied:
                projection_started = time.perf_counter()
                try:
                    status, evidence = _proof_stage_projection_v1(
                        scenario,
                        proof_value,
                        manifest,
                        native_cp_sat=self.native_cp_sat_scoring,
                        maximum_seconds=max(0.001, deadline - time.monotonic()),
                        maximum_conflicts=self.maximum_conflicts,
                        cancel_check=budget_stopped,
                    )
                except ValueError:
                    status, evidence = "incomplete", {}
                if self.native_cp_sat_scoring:
                    timings["proof"] += time.perf_counter() - projection_started
            else:
                status, evidence = "false", {"claimSatisfied": False}
            report(
                "proof",
                {**dict(evidence), "proofStateProbe": probe},
            )
            return status, proof_value, compilation.graph, evidence, None

        def has_unique_raw_root(
            seed_value: CompleteStateGraphSeedV1,
        ) -> tuple[bool | None, str | None]:
            """Reject roots that no prefix score can make uniquely winning."""
            nonlocal latest_graph_receipt

            cache_key = _raw_root_uniqueness_key_v1(
                seed_value, Team[self.proof_request.target_team.upper()]
            )
            if cache_key in raw_root_cache:
                counters["rawRootCacheHits"] += 1
                return raw_root_cache[cache_key], None
            counters["proofStateProbes"] += 1
            raw_request_core = self.proof_request.to_dict()
            raw_request_core.pop("sha256", None)
            raw_request_core.update(
                {
                    "claim": "winning_root_action_count",
                    "utility": INTERNAL_RAW_POINT_MARGIN_UTILITY_V1,
                    "successThreshold": 0.0,
                }
            )
            raw_request_core.pop("selectedAction", None)
            raw_request_core.pop("lineUniverse", None)
            raw_request = replace(
                self.proof_request,
                claim="winning_root_action_count",
                utility=INTERNAL_RAW_POINT_MARGIN_UTILITY_V1,
                success_threshold=0.0,
                line_universe=None,
                selected_action=None,
                sha256=canonical_sha256(raw_request_core),
            )
            graph_started = time.perf_counter()
            report("graph", {"proofStateProbe": True, "probeKind": "raw_margin"})
            compilation = self.complete_state_graph_compiler.compile(
                seed_value,
                raw_request,
                expected_rules_hash=self.rules.sha256,
                maximum_nodes=self.maximum_proof_nodes,
                maximum_depth=self.maximum_proof_depth,
                maximum_seconds=max(0.001, deadline - time.monotonic()),
                cancelled=budget_stopped,
            )
            timings["graph"] += time.perf_counter() - graph_started
            latest_graph_receipt = compilation.receipt.to_dict()
            if not compilation.complete or compilation.graph is None:
                return None, (
                    "tail_raw_margin_graph_budget" if time.monotonic() >= deadline
                    else "tail_raw_margin_graph_" + str(compilation.receipt.exhaustion_reason)
                )
            proof_started = time.perf_counter()
            try:
                raw_proof = OrToolsScenarioGraphProofV1().prove(
                    compilation.graph,
                    raw_request,
                    maximum_seconds=max(0.001, deadline - time.monotonic()),
                    maximum_line_count_limbs=self.maximum_line_count_limbs,
                    cancel_check=budget_stopped,
                )
            except InterruptedError:
                return None, "tail_raw_margin_proof_budget" if time.monotonic() >= deadline else "cancelled"
            timings["proof"] += time.perf_counter() - proof_started
            if raw_proof.completion_status == "proof_incomplete":
                return None, "tail_raw_margin_proof_budget"
            unique = _unique_raw_root_matches_selectors_v1(
                raw_proof.root_actions,
                actor=seed_value.next_actor,
                selectors=winning_root_selectors,
            )
            raw_root_cache[cache_key] = unique
            return unique, None

        def exclude_raw_context(seed_value, original, target_team):
            # Keep the full scoring context; only vary the prefix raw split.
            current_raw = seed_value.base_raw[int(target_team)]
            for raw in range(sum(seed_value.base_raw) + 1):
                if raw == current_raw and seed_value.base_tricks[int(Team.NORTH_SOUTH)] == original.base_tricks_ns:
                    continue  # Caller already excluded the exact failed boundary.
                excluded_seed = _seed_with_target_raw_v1(seed_value, target_team, raw)
                excluded_proof_states.append(TailProofStateNoGoodV1.build(
                    decision_state_hash=original.decision_state_hash,
                    boundary_combination=original.boundary_combination,
                    base_raw_ns=excluded_seed.base_raw[int(Team.NORTH_SOUTH)],
                    base_tricks_ns=seed_value.base_tricks[int(Team.NORTH_SOUTH)],
                    awarded_project_points=original.awarded_project_points,
                    baloot_teams=original.baloot_teams,
                    escalation_semantics=original.escalation_semantics,
                    proof_state_projection_hash=excluded_seed.proof_state_projection_hash,
                ))

        while not budget_stopped():
            advanced = False
            for combination in self.plan.boundary_combinations:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                dealer_raw, contract_raw, buyer_raw, trump_raw = combination
                dealer = Seat.parse(dealer_raw)
                if initial_dealer is not None and dealer != initial_dealer:
                    continue
                contract = Contract.parse(contract_raw)
                trump = Suit.parse(trump_raw) if trump_raw is not None else None
                acting_seat = Seat.parse(
                    scenario.decision_point["selector"]["actingSeat"]
                )
                first_leader = self.rules.play_domain.resolve_first_leader(
                    dealer=dealer,
                    buyer=Seat.parse(buyer_raw),
                    structural_domain=self.rules.structural_domain,
                )
                assert self.plan.prefix_length is not None
                assert self.plan.tail_offset is not None
                tail_decision_prefix_length = (
                    self.plan.prefix_length - self.plan.tail_offset
                )
                play_plan_hash = canonical_sha256(
                    {
                        "schemaVersion": 1,
                        "stagePlanHash": self.plan.sha256,
                        "predicateIds": [
                            item.predicate_id
                            for item in tail_construction_predicates
                        ],
                        "projectWitnessPredicateIds": [
                            item.predicate_id
                            for item in required_project_predicates
                        ],
                        "nativeFullRoundPredicateIds": [
                            item.predicate_id for item in tail_receipt_predicates
                        ],
                        **({"nativeConstructionInvariants": list(construction_invariants.values())}
                           if construction_invariants else {}),
                        "candidatePartitionPolicy": "tail_play_index_mod_v2",
                        "candidatePartition": self.candidate_partition,
                        "searchGuidance": (
                            _UNIQUE_ROOT_SUIT_RELEASE_GUIDANCE
                            if suit_release_guidance_active
                            else None
                        ),
                        "observerSafePolicyPrefixSteps": self.policy_prefix_steps,
                        "observerSafePolicyId": self.policy_id,
                        **({"reachableFullRound": True} if reachable_full_round else {}),
                        **({"pointMarginBoundarySearch": True} if point_margin_boundary_search else {}),
                        "requireOptimalAuctionActions": self.require_optimal_auction_actions,
                    }
                )
                tail_lowerings: dict[str, Any] = {}
                tail_builds: list[Any] = []
                tail_scenario_model = None
                if inline_auction_quality or tail_deal_predicates or excluded_initial_decks[combination]:
                    tail_scenario_model = OrToolsScenarioModelV1(
                        rules=self.rules,
                        dealer=dealer,
                        required_buyer=Seat.parse(buyer_raw),
                        excluded_deck_orders=excluded_initial_decks[combination],
                        canonicalize_deal_packets=True,
                    )
                    if tail_deal_predicates:
                        deal_context = DealPredicateModelContextV2(
                            tail_scenario_model, context_id="tail_construction_deal"
                        )
                        deal_lowerings = {
                            predicate.predicate_id: manifest.lower_predicate(
                                predicate, deal_context
                            )
                            for predicate in tail_deal_predicates
                        }
                        tail_lowerings.update(deal_lowerings)
                        _enforce_hard_stage_projection(
                            tail_scenario_model.model, scenario, stage="deal",
                            predicate_literals={
                                predicate_id: lowering.literal
                                for predicate_id, lowering in deal_lowerings.items()
                            },
                        )
                if inline_auction_quality:
                    tail_auction = tail_scenario_model.add_native_auction_fragment(
                        allowed_contracts=(contract,),
                        require_optimal_actions=True,
                    )
                    if trump is not None:
                        tail_scenario_model.model.add(tail_auction.trump == int(trump))
                    auction_lowerings = {
                        predicate.predicate_id: manifest.lower_predicate(
                            predicate,
                            CpSatAuctionPredicateContextV1(
                                tail_scenario_model.model,
                                tail_auction,
                                dealer,
                            ),
                        )
                        for predicate in auction_stage_predicates
                    }
                    tail_lowerings.update(auction_lowerings)
                    _enforce_hard_stage_projection(
                        tail_scenario_model.model,
                        scenario,
                        stage="auction",
                        predicate_literals={
                            predicate_id: lowering.literal
                            for predicate_id, lowering in auction_lowerings.items()
                        },
                    )

                def configure(build) -> None:
                    tail_builds.append(build)
                    _apply_candidate_partition_v1(
                        build.model,
                        build.selected_play_indexes[-self.plan.tail_length :],
                        self.candidate_partition,
                        name="tail_first",
                    )
                    tail_variables = build.selected_play_indexes[-self.plan.tail_length :]
                    for vector in excluded_tails[combination]:
                        build.model.add_forbidden_assignments(
                            tail_variables, (vector,)
                        )
                    if not selective_tail:
                        for vector in excluded_complete_plays[combination]:
                            build.model.add_forbidden_assignments(
                                build.selected_play_indexes, (vector,)
                            )
                        for index, no_good in enumerate(
                            excluded_play_zero_actions[combination]
                        ):
                            apply_play_zero_observation_action_no_good_v1(
                                build.model,
                                owner_of_card=build.initial_owner_variables,
                                selected_card=build.selected_play_indexes[0],
                                ekka_lead=build.ekka_lead_variables[0],
                                no_good=no_good,
                                name=f"tail_first_play_zero_no_good_{index}",
                            )
                    for card_id, owner in required_remaining_ownership.items():
                        build.model.add(
                            build.initial_owner_variables[card_id] == int(owner)
                        )
                        if selective_tail:
                            build.model.add(build.selected_card_literals[card_id] == 1)
                        for played_card in build.selected_play_indexes[
                            : tail_decision_prefix_length if selective_tail else self.plan.prefix_length
                        ]:
                            build.model.add(played_card != card_id)
                    for predicate in (
                        ()
                        if native_full_round_outcome
                        else required_project_predicates
                    ):
                        witnesses = _project_exists_witness_keys(
                            predicate, self.rules, contract, dealer
                        )
                        selected = tuple(
                            build.model.new_bool_var(
                                f"tail_project_witness_{predicate.predicate_id}_{index}"
                            )
                            for index in range(len(witnesses))
                        )
                        if not selected:
                            build.model.add(0 == 1)
                            continue
                        build.model.add_bool_or(selected)
                        for literal, witness in zip(selected, witnesses, strict=True):
                            for card_id in witness[3]:
                                build.model.add(
                                    build.initial_owner_variables[card_id]
                                    == witness[0]
                                ).only_enforce_if(literal)
                    if tail_construction_predicates:
                        lowerings = {
                            predicate.predicate_id: manifest.lower_predicate(
                                predicate,
                                (
                                    build.predicate_context
                                    if predicate.kind in action_kinds
                                    else build.play_state_context
                                ),
                            )
                            for predicate in tail_construction_predicates
                        }
                        tail_lowerings.update(lowerings)
                        literals = {
                            predicate_id: lowering.literal
                            for predicate_id, lowering in lowerings.items()
                        }
                        _enforce_hard_stage_projection(
                            build.model,
                            scenario,
                            stage="tailConstruction",
                            predicate_literals=literals,
                        )
                    if score_floor is not None:
                        target, minimum_opposing_project = score_floor
                        scoring = ScoringDomainIRV1.from_rules(self.rules.data)
                        total = scoring.card_game_point_totals[contract.effective.value]
                        target_game = attach_raw_conversion_v1(
                            build.model, build.raw_team_variables[int(target)],
                            domain=scoring, contract=contract,
                            name="tail_optimistic_target_game",
                        )
                        opposing_game = attach_raw_conversion_v1(
                            build.model, build.raw_team_variables[int(target.other)],
                            domain=scoring, contract=contract,
                            name="tail_optimistic_opposing_game",
                        )
                        best_target_game = build.model.new_int_var(
                            0, total, "tail_optimistic_target_card_game",
                        )
                        build.model.add_max_equality(
                            best_target_game,
                            [target_game, total - opposing_game],
                        )
                        baloot_upper = (
                            self.rules.project_domain.baloot_value
                            if self.rules.project_domain.baloot_enabled
                            and (not self.rules.project_domain.baloot_only_hokum
                                 or contract.effective == Contract.HOKUM)
                            else 0
                        )
                        # Necessary only: the proof boundary certifies exact awards,
                        # counting side, tie semantics, and the round outcome.
                        build.model.add(
                            2 * best_target_game + baloot_upper
                            >= total + minimum_opposing_project
                        )
                    if (
                        suit_release_guidance_active
                        and suit_release_guidance is not None
                    ):
                        _apply_unique_root_suit_release_guidance_v1(
                            build,
                            suit_release_guidance,
                            acting_seat,
                        )
                    if native_full_round_outcome:
                        projects = attach_project_lifecycle_fragment_v1(
                            build.model,
                            owner_of_card=build.initial_owner_variables,
                            contract=contract,
                            dealer=dealer,
                            trump=trump,
                            domain=self.rules.project_domain,
                            name="tail_construction_projects",
                            second_trick_leader=(build._extract.winner_seats[0]
                                if self.rules.project_domain.tie_break_order == "second_trick_reveal"
                                else None),
                        )
                        project_lowerings = {
                            predicate.predicate_id: manifest.lower_predicate(
                                predicate,
                                CpSatProjectPredicateContextV1(
                                    build.model,
                                    "tail_construction_projects",
                                    projects,
                                ),
                            )
                            for predicate in project_stage_predicates
                        }
                        tail_lowerings.update(project_lowerings)
                        _enforce_hard_stage_projection(
                            build.model,
                            scenario,
                            stage="projects",
                            predicate_literals={
                                predicate_id: lowering.literal
                                for predicate_id, lowering in project_lowerings.items()
                            },
                        )
                    if native_full_round_outcome or (self.native_cp_sat_scoring and escalation_stage_predicates):
                        escalation = OrToolsEscalationStageV1(
                            EscalationDomainIRV1.from_rules(self.rules.data),
                            declarer=Seat.parse(buyer_raw),
                            contract=contract,
                            match_score=starting_match_score,
                            native_cp_sat=True,
                        )
                        escalation_context = CpSatEscalationPredicateContextV1.attach(
                            escalation,
                            model=build.model,
                            context_id="tail_construction_escalation",
                        )
                        build.model.add(escalation_context.native["terminal"] == 1)
                        escalation_lowerings = {
                            predicate.predicate_id: manifest.lower_predicate(
                                predicate,
                                escalation_context,
                            )
                            for predicate in escalation_stage_predicates
                        }
                        tail_lowerings.update(escalation_lowerings)
                        _enforce_hard_stage_projection(
                            build.model,
                            scenario,
                            stage="escalation",
                            predicate_literals={
                                predicate_id: lowering.literal
                                for predicate_id, lowering in escalation_lowerings.items()
                            },
                        )
                    if native_full_round_outcome:
                        baloot_points = tuple(
                            build.model.new_int_var(
                                0,
                                self.rules.project_domain.baloot_value,
                                f"tail_construction_baloot_{int(team)}",
                            )
                            for team in Team
                        )
                        for team in Team:
                            build.model.add(
                                baloot_points[int(team)]
                                == projects.baloot_team[int(team)]
                                * self.rules.project_domain.baloot_value
                            )
                        scoring = ScoringDomainIRV1.from_rules(self.rules.data)
                        scoring_fragment = attach_scoring_fragment_v1(
                            build.model,
                            scoring,
                            CpSatScoringInputsV1(
                                contract=build.model.new_constant(
                                    scoring_contract_id_v1(contract)
                                ),
                                buyer_seat=build.model.new_constant(
                                    int(Seat.parse(buyer_raw))
                                ),
                                escalation=escalation_context.native["level"],
                                last_escalating_team=escalation_context.native[
                                    "lastTeam"
                                ],
                                raw_points=(
                                    build.raw_team_variables[0],
                                    build.raw_team_variables[1],
                                ),
                                project_points=(
                                    projects.combined_awarded_value[0],
                                    projects.combined_awarded_value[1],
                                ),
                                baloot_points=(baloot_points[0], baloot_points[1]),
                                trick_winner_teams=build.winner_team_variables,
                                starting_match_score=tuple(
                                    build.model.new_constant(value) for value in starting_match_score),
                            ),
                            source_hash=play_plan_hash,
                            native_conversion=True,
                        )
                        outcome_lowerings = {
                            predicate.predicate_id: manifest.lower_predicate(
                                predicate,
                                scoring_fragment.outcome_context,
                            )
                            for predicate in outcome_stage_predicates
                        }
                        tail_lowerings.update(outcome_lowerings)
                        _enforce_hard_stage_projection(
                            build.model,
                            scenario,
                            stage="outcome",
                            predicate_literals={
                                predicate_id: lowering.literal
                                for predicate_id, lowering in outcome_lowerings.items()
                            },
                        )

                tail_started = time.perf_counter()
                report(
                    "tailConstruction",
                    {
                        "qualityPlayZeroCutCount": len(
                            excluded_play_zero_actions[combination]
                        )
                    },
                )
                try:
                    tail_solver = cp_model.CpSolver()
                    def tail_telemetry(evidence: Mapping[str, Any]) -> None:
                        if evidence.get("solverStatus") != "running":
                            timings["tailModelBuild"] += float(evidence.get("modelBuildSeconds", 0.0))
                            timings["tailModelConfigure"] += float(evidence.get("modelConfigureSeconds", 0.0))
                            timings["tailSolver"] += float(evidence.get("wallTimeSeconds", 0.0))
                        report("tailConstruction", evidence)

                    constructed_play = OrToolsPlayTailStageV1(native_conversion=self.native_cp_sat_scoring).solve(
                        cards=FULL_DECK,
                        ownership=exact_ownership or None,
                        contract=contract,
                        trump=trump,
                        first_leader=(
                            Seat((int(acting_seat) - int(scenario.decision_point["selector"]["position"])) % 4)
                            if selective_tail else first_leader
                        ),
                        play_domain=self.rules.play_domain,
                        structural_domain=self.rules.structural_domain,
                        scoring_domain=ScoringDomainIRV1.from_rules(self.rules.data),
                        **({"selected_play_count": self.plan.tail_length} if selective_tail else {"play_step_count": 32}),
                        policy_prefix_steps=(
                            max(0, min(tail_decision_prefix_length,
                                       self.policy_prefix_steps - self.plan.tail_offset))
                            if selective_tail else self.policy_prefix_steps
                        ),
                        policy_id=self.policy_id,
                        decision_prefix_length=(
                            tail_decision_prefix_length if selective_tail else self.plan.prefix_length
                        ),
                        decision_actor=acting_seat,
                        excluded_decision_status_vectors=(
                            tuple(
                                tuple(8 if status == 9 else status for status in vector)
                                for vector in self.excluded_decision_status_vectors
                            )
                        ),
                        exact_excluded_decision_status_vectors=(
                            tuple(rejected_decision_states)
                        ),
                        minimum_decision_status_hamming_distance=(
                            self.minimum_decision_status_hamming_distance
                        ),
                        construction_plan_hash=play_plan_hash,
                        model_configurator=configure,
                        shared_model=(
                            tail_scenario_model.model
                            if tail_scenario_model is not None
                            else None
                        ),
                        shared_owner_of_card=(
                            tail_scenario_model.owner_of_card
                            if tail_scenario_model is not None
                            else None
                        ),
                        solver=tail_solver,
                        random_seed=seed,
                        maximum_time_seconds=remaining,
                        maximum_conflicts=maximum_conflicts,
                        telemetry_callback=tail_telemetry,
                        cancel_check=budget_stopped,
                    )
                    tail = (
                        constructed_play if selective_tail else _reachable_tail_view_v1(
                            constructed_play,
                            tail_offset=self.plan.tail_offset,
                            decision_prefix_length=self.plan.prefix_length,
                        )
                    )
                    tail_predicate_evidence = {
                        predicate_id: lowering.extract(tail_solver)
                        for predicate_id, lowering in sorted(
                            tail_lowerings.items()
                        )
                    }
                    if len(tail_builds) != 1:
                        raise AssertionError("tail construction must expose one model")
                    tail_assumptions, tail_bindings = _cp_sat_receipt_metadata_v1(
                        tail_builds[0].model,
                        stage="tailConstruction",
                        predicates=tail_receipt_predicates,
                        predicate_literals={
                            predicate_id: lowering.literal
                            for predicate_id, lowering in tail_lowerings.items()
                        },
                        model_hash=tail.model_hash,
                        evidence_hash=tail.evidence_hash,
                        lowering_manifest_hash=manifest.manifest_hash,
                    )
                    tail_bindings = tuple(
                        {
                            **dict(binding),
                            "evidenceHash": tail_predicate_evidence[
                                str(binding["predicateId"])
                            ]["sha256"],
                        }
                        for binding in tail_bindings
                    )
                    completed_tricks = self.plan.tail_offset // 4
                    fixed_card_owners = None if selective_tail else tuple(
                        next(
                            seat
                            for seat in Seat
                            if card in constructed_play.initial_hands[int(seat)]
                        )
                        for card in FULL_DECK
                    )
                    complete_base_raw_ns = None if selective_tail else sum(
                        points
                        for winner, points in zip(
                            constructed_play.trick_winners[:completed_tricks],
                            constructed_play.trick_raw_points[:completed_tricks],
                            strict=True,
                        )
                        if winner.team == Team.NORTH_SOUTH
                    )
                    complete_base_tricks_ns = None if selective_tail else sum(
                        winner.team == Team.NORTH_SOUTH
                        for winner in constructed_play.trick_winners[:completed_tricks]
                    )
                except PlayTailSolveStopped as stopped:
                    timings["tailConstruction"] += time.perf_counter() - tail_started
                    evidence = dict(stopped.partial_receipt["evidence"])
                    if "wallTimeSeconds" not in evidence:
                        timings["tailModelBuild"] += float(evidence.get("modelBuildSeconds", 0.0))
                        timings["tailModelConfigure"] += float(evidence.get("modelConfigureSeconds", 0.0))
                    report("tailConstruction", evidence)
                    if stopped.solver_status != "infeasible":
                        partial_core = {
                            "schemaVersion": 1,
                            "status": "incomplete",
                            "stage": "tailConstruction",
                            "exhaustionReason": "tail_construction_budget",
                            "evidence": {
                                **evidence,
                                "stageTimings": dict(timings),
                                "stageCounters": dict(counters),
                            },
                        }
                        return OrToolsCandidateAttemptV1(
                            "incomplete",
                            None,
                            "tail_construction_budget",
                            stage_timings=timings,
                            partial_receipt={
                                **partial_core,
                                "sha256": canonical_sha256(partial_core),
                            },
                        )
                    continue
                timings["tailConstruction"] += time.perf_counter() - tail_started
                advanced = True
                counters["tailCandidates"] += 1
                tail_vector = tuple(play.card.id for play in tail.plays)
                decision_status_vector = _decision_status_vector_v1(
                    tail,
                    prefix_length=tail_decision_prefix_length,
                    acting_seat=acting_seat,
                )
                report(
                    "tailConstruction",
                    {
                        "decisionStateHash": canonical_sha256(
                            list(decision_status_vector)
                        ),
                        "searchGuidance": (
                            _UNIQUE_ROOT_SUIT_RELEASE_GUIDANCE
                            if suit_release_guidance_active
                            else None
                        ),
                    },
                )

                try_complete_boundary = flexible_boundary_score and not selective_tail
                generic_prefix_screened = False
                while not budget_stopped():
                    counters["proofBoundaryAttempts"] += 1
                    boundary_started = time.perf_counter()
                    report("proofBoundary")
                    constrain_to_complete = (
                        not flexible_boundary_score
                        or (
                            try_complete_boundary
                            and required_proof_state is None
                        )
                    )
                    boundary = OrToolsTailProofBoundaryV1(
                        rules=self.rules,
                        starting_match_score=starting_match_score,
                        native_cp_sat=self.native_cp_sat_scoring,
                    ).solve(
                        scenario,
                        self.plan,
                        tail,
                        fixed_card_owners=fixed_card_owners,
                        required_base_raw_ns=(
                            complete_base_raw_ns if constrain_to_complete else None
                        ),
                        required_base_tricks_ns=(
                            complete_base_tricks_ns if constrain_to_complete else None
                        ),
                        preferred_base_raw_ns=(
                            complete_base_raw_ns
                            if flexible_boundary_score and not constrain_to_complete
                            else None
                        ),
                        preferred_base_tricks_ns=(
                            complete_base_tricks_ns
                            if flexible_boundary_score and not constrain_to_complete
                            else None
                        ),
                        excluded=excluded_boundaries,
                        excluded_proof_states=excluded_proof_states,
                        required_history=required_history,
                        required_proof_state=required_proof_state,
                        boundary_combination=combination,
                        maximum_seconds=max(0.001, deadline - time.monotonic()),
                        maximum_conflicts=maximum_conflicts,
                        cancel_check=budget_stopped,
                    )
                    timings["proofBoundary"] += time.perf_counter() - boundary_started
                    timings["proofBoundaryModelBuild"] += boundary.model_build_seconds
                    timings["proofBoundarySolver"] += boundary.solver_seconds
                    counters["proofBoundarySolverAttempts"] += boundary.solve_attempts
                    report("proofBoundary", {
                        "proofBoundaryStatus": boundary.status,
                        "proofBoundaryReason": boundary.reason,
                    })
                    if boundary.status == "incomplete":
                        partial_core = {
                            "schemaVersion": 1,
                            "status": "incomplete",
                            "stage": "proofBoundary",
                            "exhaustionReason": boundary.reason or "tail_boundary_budget",
                            "evidence": {
                                "decisionStateHash": canonical_sha256(
                                    list(decision_status_vector)
                                ),
                                "decisionStateStatusVector": list(
                                    decision_status_vector
                                ),
                                "stageTimings": dict(timings),
                                "stageCounters": dict(counters),
                                **(
                                    {"graphReceipt": dict(latest_graph_receipt)}
                                    if latest_graph_receipt is not None else {}
                                ),
                            },
                        }
                        return OrToolsCandidateAttemptV1(
                            "incomplete",
                            None,
                            boundary.reason,
                            stage_timings=timings,
                            partial_receipt={
                                **partial_core,
                                "sha256": canonical_sha256(partial_core),
                            },
                        )
                    if boundary.status == "infeasible":
                        if flexible_boundary_score and constrain_to_complete:
                            excluded_complete_plays[combination].append(
                                tuple(
                                    play.card.id
                                    for play in constructed_play.plays
                                )
                            )
                            break
                        if required_proof_state is not None:
                            excluded_proof_states.append(required_proof_state)
                            required_history = None
                            required_proof_state = None
                            continue
                        excluded_tails[combination].append(tail_vector)
                        break
                    try_complete_boundary = False
                    required_history = None
                    required_proof_state = None
                    assert (
                        boundary.seed is not None
                        and boundary.no_good is not None
                        and boundary.proof_state_no_good is not None
                    )

                    target_team = Team[self.proof_request.target_team.upper()]
                    monotone_sun_score = (
                        self.proof_request.mode == "exact"
                        and boundary.seed.contract.effective == Contract.SUN
                        and all(count > 0 for count in boundary.seed.base_tricks)
                        and boundary.seed.next_actor.team == target_team
                    )
                    if (
                        exact_unique_root_threshold
                        and point_margin_boundary_search and monotone_sun_score
                        and raw_root_cache.get(_raw_root_uniqueness_key_v1(
                            boundary.seed, target_team,
                        )) is False
                    ):
                        # Scoring cannot rescue this cached nonmatching raw order in non-sweep Sun.
                        counters["rawRootCacheHits"] += 1
                        excluded_proof_states.append(boundary.proof_state_no_good)
                        completed = sum(boundary.seed.base_tricks)
                        for ns_tricks in range(1, completed):
                            context_seed = _seed_clone(boundary.seed,
                                base_tricks=(ns_tricks, completed - ns_tricks))
                            exclude_raw_context(context_seed, boundary.proof_state_no_good, target_team)
                        continue

                    proof_status, proof, proof_graph, _proof_evidence, proof_reason = (
                        evaluate_proof_seed(boundary.seed, probe=False)
                    )
                    if proof_status == "incomplete":
                        return incomplete_proof_attempt(
                            proof_reason or "tail_proof_predicate_evidence", "graph"
                            if proof_graph is None else "proof",
                        )
                    assert proof is not None
                    assert proof_graph is not None
                    if proof_status == "false":
                        counters["proofPredicateRejections"] += 1
                        root_count = proof.winning_root_action_count
                        counters[
                            "proofRejectedZeroWinningRoots"
                            if root_count == 0 else (
                                "proofRejectedOneWinningRoot"
                                if root_count == 1 else "proofRejectedMultipleWinningRoots"
                            )
                        ] += 1
                        target_team = Team[self.proof_request.target_team.upper()]
                        if (
                            exact_unique_root_threshold
                            # Raw-card ties do not imply ties after contract
                            # transfers or Kabout scoring. Never use this cut
                            # to exclude public awarded/outcome proof states.
                            and self.proof_request.utility
                            == INTERNAL_RAW_POINT_MARGIN_UTILITY_V1
                            and boundary.seed.next_actor.team == target_team
                        ):
                            unique_raw_root, raw_probe_reason = has_unique_raw_root(
                                boundary.seed
                            )
                            if unique_raw_root is None:
                                return incomplete_proof_attempt(
                                    raw_probe_reason or "tail_raw_margin_proof", "graph"
                                    if "graph" in (raw_probe_reason or "") else "proof",
                                )
                            if not unique_raw_root:
                                rejected_decision_states.append(
                                    decision_status_vector
                                )
                                break
                        excluded_proof_states.append(boundary.proof_state_no_good)
                        if (
                            (
                                exact_unique_root_threshold
                                and self.proof_request.utility in {"round_win", "contract_success"}
                            )
                            or point_margin_boundary_search and monotone_sun_score
                        ):
                            target_team = Team[self.proof_request.target_team.upper()]
                            if monotone_sun_score and exact_unique_root_threshold:
                                # No sweep and no future Sun Baloot: terminal utility
                                # is a monotone function of raw points with fixed
                                # projects. Min/max preserve that ordering, so a
                                # tied best raw root cannot become uniquely winning.
                                unique_raw_root, raw_probe_reason = has_unique_raw_root(boundary.seed)
                                if unique_raw_root is None:
                                    return incomplete_proof_attempt(
                                        raw_probe_reason or "tail_raw_margin_proof",
                                        "graph" if "graph" in (raw_probe_reason or "") else "proof",
                                    )
                                if not unique_raw_root:
                                    completed = sum(boundary.seed.base_tricks)
                                    for ns_tricks in range(1, completed):
                                        # All non-sweep histories have the same
                                        # raw ranking. Keep both sweep boundaries
                                        # and every other project context eligible.
                                        context_seed = _seed_clone(boundary.seed,
                                            base_tricks=(ns_tricks, completed - ns_tricks))
                                        exclude_raw_context(context_seed, boundary.proof_state_no_good, target_team)
                                    continue
                            raw_total = sum(boundary.seed.base_raw)
                            current_raw = boundary.seed.base_raw[int(target_team)]
                            context = _raw_root_state_key_v1(
                                boundary.seed, target_team
                            )
                            if context != score_probe_context:
                                score_probe_cache.clear()
                                score_probe_context = context
                            # The boundary solve is already an exact probe of this
                            # raw split under the same complete scoring context.
                            score_probe_cache[current_raw] = (proof_status, proof)
                            probes: dict[
                                int,
                                tuple[
                                    CompleteStateGraphSeedV1,
                                    str,
                                    OrToolsGraphProofOutcomeV1 | None,
                                    str | None,
                                ],
                            ] = {
                                current_raw: (
                                    boundary.seed, proof_status, proof, None,
                                )
                            }

                            def probe_raw(target_raw: int):
                                cached = probes.get(target_raw)
                                if cached is not None:
                                    return cached
                                seed_probe = _seed_with_target_raw_v1(
                                    boundary.seed,
                                    target_team,
                                    target_raw,
                                )
                                prior = score_probe_cache.get(target_raw)
                                if prior is not None:
                                    counters["scoreProbeCacheHits"] += 1
                                    cached = (seed_probe, prior[0], prior[1], None)
                                    probes[target_raw] = cached
                                    return cached
                                status, proof_probe, _graph, _evidence, reason = (
                                    evaluate_proof_seed(seed_probe, probe=True)
                                )
                                cached = (seed_probe, status, proof_probe, reason)
                                probes[target_raw] = cached
                                if status in {"satisfied", "false"} and proof_probe is not None:
                                    score_probe_cache[target_raw] = (status, proof_probe)
                                return cached

                            interval = _first_winning_raw_interval_v1(
                                current_raw, raw_total,
                                proof.winning_root_action_count,
                            )
                            if interval is None:
                                # The exact failed boundary was excluded above.
                                # Other project/escalation/trick contexts remain viable.
                                continue
                            low, high = interval
                            probe_incomplete_reason = None
                            while low < high:
                                middle = (low + high) // 2
                                _seed_probe, status, proof_probe, reason = probe_raw(
                                    middle
                                )
                                if status == "incomplete" or proof_probe is None:
                                    probe_incomplete_reason = reason
                                    break
                                if proof_probe.winning_root_action_count >= 1:
                                    high = middle
                                else:
                                    low = middle + 1
                            if probe_incomplete_reason is not None:
                                return incomplete_proof_attempt(
                                    probe_incomplete_reason, "graph"
                                    if "graph" in probe_incomplete_reason else "proof",
                                )
                            seed_probe, status, proof_probe, reason = probe_raw(low)
                            if status == "incomplete" or proof_probe is None:
                                return incomplete_proof_attempt(
                                    reason or "tail_proof_state_probe", "graph"
                                    if "graph" in (reason or "") else "proof",
                                )
                            if status == "satisfied":
                                required_history = boundary.no_good
                                required_proof_state = TailProofStateNoGoodV1.build(
                                    decision_state_hash=(
                                        boundary.proof_state_no_good.decision_state_hash
                                    ),
                                    boundary_combination=combination,
                                    base_raw_ns=seed_probe.base_raw[
                                        int(Team.NORTH_SOUTH)
                                    ],
                                    base_tricks_ns=seed_probe.base_tricks[
                                        int(Team.NORTH_SOUTH)
                                    ],
                                    awarded_project_points=(
                                        seed_probe.awarded_project_points
                                    ),
                                    baloot_teams=tuple(
                                        points > 0
                                        for points in seed_probe.baloot_points
                                    ),
                                    escalation_semantics=(
                                        seed_probe.escalation,
                                        seed_probe.locked,
                                        (
                                            seed_probe.last_escalating_team.name.lower()
                                            if seed_probe.last_escalating_team
                                            is not None
                                            else None
                                        ),
                                    ),
                                    proof_state_projection_hash=(
                                        seed_probe.proof_state_projection_hash
                                    ),
                                )
                            elif proof_probe.winning_root_action_count > 1:
                                # Below the first winning raw split there are no
                                # winners; here and above there are at least two.
                                # Reuse that monotone certificate only within this
                                # exact scoring context, never as a whole-tail cut.
                                exclude_raw_context(boundary.seed, boundary.proof_state_no_good, target_team)
                            # Other scoring contexts remain eligible.
                        continue

                    if selective_tail and self.policy_prefix_steps and not generic_prefix_screened:
                        generic_prefix_screened = True
                        screen_started = time.perf_counter()
                        report("prefixReachability", {"screen": "tail_policy"})
                        screen_statuses = _tail_policy_prefix_screen_v1(
                            tail=tail,
                            tail_offset=self.plan.tail_offset,
                            rules=self.rules,
                            contract=contract,
                            trump=trump,
                            first_leader=first_leader,
                            policy_prefix_steps=self.policy_prefix_steps,
                            policy_id=self.policy_id,
                            deadline=deadline,
                            maximum_conflicts=maximum_conflicts,
                            seed=seed,
                            cancel_check=budget_stopped,
                            native_conversion=self.native_cp_sat_scoring,
                        )
                        timings["prefixReachability"] += time.perf_counter() - screen_started
                        if screen_statuses == ("infeasible", "infeasible"):
                            excluded_tails[combination].append(tail_vector)
                            counters["prefixReachabilityTailCuts"] += 1
                            report("prefixReachability", {"tailPolicyScreen": "infeasible"})
                            break

                    if selective_tail and self.policy_prefix_steps:
                        probe_started = time.perf_counter()
                        report("prefixReachability")
                        reachable = False
                        inconclusive = False

                        def bind_reachable_prefix(build) -> None:
                            if not _preserve_tail_cards_before_decision(
                                build, tail, self.plan.tail_offset, budget_stopped,
                            ):
                                raise PlayTailSolveStopped(
                                    "unknown",
                                    evidence={"solverStatus": "unknown"},
                                )
                            _bind_proof_seed_prefix(
                                build, boundary.seed, self.plan.tail_offset // 4
                            )
                            _bind_reachable_project_truth_v1(
                                build,
                                scenario=scenario,
                                manifest=manifest,
                                project_predicates=project_stage_predicates,
                                seed=boundary.seed,
                                tail=tail,
                                rules=self.rules,
                                contract=contract,
                                dealer=dealer,
                                trump=trump,
                            )

                        for locked in (boundary.seed.locked,):
                            try:
                                OrToolsPlayTailStageV1(native_conversion=self.native_cp_sat_scoring).solve(
                                    cards=FULL_DECK,
                                    ownership={
                                        card: seat
                                        for seat in Seat
                                        for card in tail.initial_hands[int(seat)]
                                    },
                                    contract=contract,
                                    trump=trump,
                                    first_leader=first_leader,
                                    play_domain=self.rules.play_domain,
                                    structural_domain=self.rules.structural_domain,
                                    scoring_domain=ScoringDomainIRV1.from_rules(
                                        self.rules.data
                                    ),
                                    tail=tail.plays[:4],
                                    locked=locked,
                                    decision_prefix_length=self.plan.prefix_length,
                                    play_step_count=self.plan.tail_offset + 4,
                                    policy_prefix_steps=self.policy_prefix_steps,
                                    policy_id=self.policy_id,
                                    construction_plan_hash=canonical_sha256({
                                        "schemaVersion": 1,
                                        "stage": "proof_seed_prefix_project_reachability_v3",
                                        "proofStateHash": (
                                            boundary.seed.proof_state_projection_hash
                                        ),
                                        "tailEvidenceHash": tail.evidence_hash,
                                    }),
                                    model_configurator=bind_reachable_prefix,
                                    maximum_time_seconds=max(
                                        0.001, deadline - time.monotonic()
                                    ),
                                    maximum_conflicts=maximum_conflicts,
                                    random_seed=seed,
                                    cancel_check=budget_stopped,
                                )
                                reachable = True
                                break
                            except PlayTailSolveStopped as stopped:
                                if stopped.solver_status != "infeasible":
                                    inconclusive = True
                                    break
                        timings["prefixReachability"] += time.perf_counter() - probe_started
                        if not reachable and not inconclusive:
                            # This proof state includes its own project/score
                            # history. Another boundary for the same cards may
                            # still have a reachable prefix.
                            excluded_proof_states.append(
                                boundary.proof_state_no_good
                            )
                            counters["prefixReachabilityCuts"] += 1
                            report("prefixReachability", {"prefixReachability": "infeasible"})
                            continue

                    if (
                        counters["prefixLiftAttempts"]
                        >= self.maximum_prefix_lift_attempts
                    ):
                        partial_core = {
                            "schemaVersion": 1,
                            "status": "incomplete",
                            "stage": "prefixLift",
                            "exhaustionReason": "prefix_lift_attempt_budget",
                            "evidence": {
                                "prefixLiftFailures": list(lift_failures),
                                "stageTimings": dict(timings),
                                "stageCounters": dict(counters),
                            },
                        }
                        return OrToolsCandidateAttemptV1(
                            "incomplete",
                            None,
                            "prefix_lift_attempt_budget",
                            stage_timings=timings,
                            partial_receipt={
                                **partial_core,
                                "sha256": canonical_sha256(partial_core),
                            },
                        )
                    counters["prefixLiftAttempts"] += 1
                    lift_started = time.perf_counter()
                    report(
                        "prefixLift",
                        {
                            "qualityPlayZeroCutCount": len(
                                excluded_play_zero_actions[combination]
                            )
                        },
                    )
                    lift = OrToolsTailFirstBackwardLiftV1(
                        rules=self.rules,
                        native_cp_sat_scoring=self.native_cp_sat_scoring,
                    ).lift(
                        scenario,
                        self.plan,
                        tail,
                        boundary.seed,
                        play_hint=None if selective_tail else constructed_play,
                        certified_projects=boundary.project_result,
                        excluded_decks=(*excluded_decks, *excluded_initial_decks[combination]),
                        excluded_complete_plays=excluded_complete_plays[combination],
                        excluded_play_zero_actions=(
                            excluded_play_zero_actions[combination]
                        ),
                        policy_prefix_steps=self.policy_prefix_steps,
                        policy_id=self.policy_id,
                        require_optimal_auction_actions=(
                            self.require_optimal_auction_actions
                        ),
                        maximum_seconds=max(0.001, deadline - time.monotonic()),
                        maximum_conflicts=maximum_conflicts,
                        cancel_check=self.cancel_check,
                        progress_callback=lambda evidence: report(
                            "prefixLift", evidence
                        ),
                        seed=seed,
                    )
                    timings["prefixLift"] += time.perf_counter() - lift_started
                    if lift.status != "feasible":
                        lift_failures.append({
                            "attempt": counters["prefixLiftAttempts"],
                            "status": lift.status,
                            "reason": lift.reason,
                            "proofStateProjectionHash": boundary.seed.proof_state_projection_hash,
                            "boundaryEvidenceHash": boundary.evidence_hash,
                        })
                        report(
                            "prefixLift",
                            {
                                "prefixLiftStatus": lift.status,
                                "prefixLiftReason": lift.reason,
                            },
                        )
                    if lift.status == "incomplete":
                        partial_core = {
                            "schemaVersion": 1,
                            "status": "incomplete",
                            "stage": "prefixLift",
                            "exhaustionReason": lift.reason,
                            "evidence": {
                                "prefixLiftFailures": list(lift_failures),
                                "decisionStateHash": canonical_sha256(
                                    list(decision_status_vector)
                                ),
                                "stageTimings": dict(timings),
                                "stageCounters": dict(counters),
                            },
                        }
                        return OrToolsCandidateAttemptV1(
                            "incomplete",
                            None,
                            lift.reason,
                            stage_timings=timings,
                            partial_receipt={
                                **partial_core,
                                "sha256": canonical_sha256(partial_core),
                            },
                        )
                    if lift.reason == "tail_optimal_auction_infeasible":
                        rejected_decision_states.append(decision_status_vector)
                        break
                    if lift.status != "feasible" or lift.candidate is None:
                        excluded_boundaries.append(boundary.no_good)
                        if (
                            lift.status == "infeasible"
                            and boundary.proof_state_no_good is not None
                        ):
                            excluded_proof_states.append(
                                boundary.proof_state_no_good
                            )
                        continue

                    candidate = lift.candidate
                    receipts = {
                        **candidate.stage_receipts,
                        "tailConstruction": _stage_receipt(
                            "tailConstruction",
                            model_hash=tail.model_hash,
                            evidence_hash=tail.evidence_hash,
                            semantic={
                                "stagePlanHash": self.plan.sha256,
                                "boundaryCombination": list(combination),
                                "tail": tail.to_dict(),
                                "searchCounters": dict(counters),
                                **({"nativeConstructionInvariants": list(construction_invariants.values())}
                                   if construction_invariants else {}),
                            },
                            solver_status=tail.solver_status,
                            branches=tail_solver.num_branches,
                            conflicts=tail_solver.num_conflicts,
                            variable_count=len(tail_builds[0].model.proto.variables),
                            constraint_count=len(tail_builds[0].model.proto.constraints),
                            named_assumptions=tail_assumptions,
                            evidence_bindings=tail_bindings,
                        ),
                        "proofBoundary": _stage_receipt(
                            "proofBoundary",
                            model_hash=boundary.model_hash,
                            evidence_hash=boundary.evidence_hash,
                            semantic=boundary.seed.to_dict(),
                            solver_status=boundary.solver_status,
                            branches=boundary.branches,
                            conflicts=boundary.conflicts,
                            variable_count=boundary.variable_count,
                            constraint_count=boundary.constraint_count,
                            evidence_bindings=boundary.evidence_bindings,
                        ),
                        "prefixLift": _stage_receipt(
                            "prefixLift",
                            model_hash=canonical_sha256(
                                {
                                    "stagePlanHash": self.plan.sha256,
                                    "proofStateProjectionHash": (
                                        boundary.seed.proof_state_projection_hash
                                    ),
                                }
                            ),
                            evidence_hash=candidate.sha256,
                            semantic={
                                "stagePlanHash": self.plan.sha256,
                                "proofStateProjectionHash": (
                                    boundary.seed.proof_state_projection_hash
                                ),
                                "candidateHash": candidate.sha256,
                                "searchCounters": dict(counters),
                            },
                            solver_status=candidate.stage_receipts[
                                "deal"
                            ].solver_status,
                        ),
                    }
                    candidate = OrToolsNeutralCandidateV1(
                        candidate.origin,
                        {
                            **candidate.trace,
                            "tailFirstStagePlan": self.plan.to_dict(),
                            "tailProofBoundary": boundary.seed.to_dict(),
                            "tailFirstSearchCounters": dict(counters),
                        },
                        candidate.state,
                        candidate.no_good_vector,
                        receipts,
                        candidate.exact_state_id,
                    )
                    if self._candidate_refinement is not None:
                        miss = self._candidate_refinement(candidate, deadline - time.monotonic())
                        if miss is not None:
                            self._reject_quality_candidate(candidate, reject_initial_input=True)
                            excluded_initial_decks[combination].append(candidate.no_good_vector)
                            lift_failures.append({"attempt": counters["prefixLiftAttempts"],
                                "status": "not_found", "reason": "hard_future_card_count_miss",
                                "candidateHash": candidate.sha256, "qualityFailure": miss})
                            continue
                    return OrToolsCandidateAttemptV1(
                        "feasible",
                        candidate,
                        None,
                        proof_graph,
                        proof,
                        boundary.seed.proof_state_projection_hash,
                        timings,
                        predicate_evidence={
                            **tail_predicate_evidence,
                            **boundary.predicate_evidence,
                            **lift.predicate_evidence,
                        },
                    )
            if not advanced:
                if suit_release_guidance_active:
                    suit_release_guidance_active = False
                    report(
                        "tailConstruction",
                        {
                            "searchGuidanceFallback": (
                                _UNIQUE_ROOT_SUIT_RELEASE_GUIDANCE
                            )
                        },
                    )
                    continue
                if partial_initial_input_choice and not budget_stopped():
                    excluded_initial_inputs.append((int(selected_dealer), *starting_match_score))
                    try:
                        selected_dealer, next_scores, _partial = _required_search_initial_inputs_v3(
                            scenario, rules=self.rules, deadline=deadline, maximum_conflicts=maximum_conflicts,
                            cancel_check=self.cancel_check, excluded_inputs=excluded_initial_inputs)
                    except InformationSetCompilationStopped as stopped:
                        receipt = {**stopped.partial_receipt, "evidence": {
                            **stopped.partial_receipt.get("evidence", {}),
                            "stageTimings": dict(timings), "stageCounters": dict(counters),
                            "prefixLiftFailures": list(lift_failures),
                            **({"graphReceipt": dict(latest_graph_receipt)} if latest_graph_receipt is not None else {}),
                        }}
                        receipt["sha256"] = canonical_sha256({key: value for key, value in receipt.items() if key != "sha256"})
                        return OrToolsCandidateAttemptV1("incomplete", None, stopped.reason,
                            stage_timings=timings, partial_receipt=receipt)
                    if next_scores is None:
                        break
                    starting_match_score = next_scores
                    if initial_dealer is not None:
                        initial_dealer = selected_dealer
                    # Learned cuts belong to this initial state; campaign budgets and caller cuts do not reset.
                    excluded_boundaries.clear()
                    excluded_proof_states.clear()
                    required_history = required_proof_state = None
                    rejected_decision_states.clear()
                    score_probe_context = latest_graph_receipt = None
                    score_probe_cache.clear()
                    lift_failures.clear()
                    for combination in self.plan.boundary_combinations:
                        quality_key = (*combination, *starting_match_score)
                        excluded_tails[combination].clear()
                        excluded_complete_plays[combination] = list(self._quality_rejected_complete_plays.get(quality_key, ()))
                        excluded_play_zero_actions[combination] = list(self._quality_rejected_play_zero_actions.get(quality_key, ()))
                        excluded_initial_decks[combination] = list(self._quality_rejected_initial_decks.get(quality_key, ()))
                    suit_release_guidance_active = suit_release_guidance is not None
                    continue
                break
        cancelled = self.cancel_check is not None and self.cancel_check()
        exhausted = cancelled or time.monotonic() >= deadline
        exhaustion_reason = "cancelled" if cancelled else "tail_first_budget"
        partial_receipt = None
        if exhausted:
            partial_core = {
                "schemaVersion": 1,
                "status": "incomplete",
                "stage": "tailFirst",
                "exhaustionReason": exhaustion_reason,
                "evidence": {
                    "stageTimings": dict(timings),
                    "stageCounters": dict(counters),
                },
            }
            partial_receipt = {
                **partial_core,
                "sha256": canonical_sha256(partial_core),
            }
        if not exhausted and (self.require_optimal_auction_actions or partial_initial_input_choice
                              or self._quality_rejected_initial_decks):
            return OrToolsCandidateAttemptV1(
                "not_found", None, "initial_input_choice_no_witness" if partial_initial_input_choice
                else "tail_quality_constrained_no_witness", stage_timings=timings,
            )
        return OrToolsCandidateAttemptV1(
            "incomplete" if exhausted else "infeasible",
            None,
            exhaustion_reason if exhausted else "tail_first_infeasible",
            stage_timings=timings,
            partial_receipt=partial_receipt,
        )


def family_tail_first_plan_v1(rules, execution, *, scenario, request, playing_quality):
    """Prepare one frozen construction scope; qualification keeps the original request."""
    from .tail_first_stage_planner import TailFirstStagePlannerV1
    from .family_catalog import validate_family_target_scenario_v1

    if execution.family_opening_mode is not None:
        raise ValueError('tail-first requires an explicitly frozen execution without familyOpeningMode')
    if request.mode != 'exact' or request.visibility != 'actor_information_set':
        raise ValueError('tail-first proposals require an unchanged exact actor-information-set request')
    if playing_quality is None:
        raise ValueError('tail-first proposals require playingQuality')
    target = execution.family_target
    validate_family_target_scenario_v1(target, scenario, rules=rules, execution=execution,
                                      playing_quality=playing_quality)
    construction = scenario.to_dict()
    construction.pop('sha256')
    # Pin the requested opening and require a real choice, not a forced play.
    construction['constraints'] = {'op': 'all', 'children': [construction['constraints'], *[
        dict(predicateId=f'PRED-CONSTRUCTION-{name}', kind=kind, version=1, hardness='hard', arguments=args)
        for name, kind, args in (
            ('DEALER', 'auction.dealer', {'seat': target['dealer']}),
            ('BUYER', 'auction.buyer', {'seat': target['buyer']}),
            ('CHOICES', 'state.legal_action_count', {'minimum': 2, 'maximum': 2 * len(FULL_DECK)}))]]}
    construction = ResearchScenarioV1.build(construction)
    plan = TailFirstStagePlannerV1(native_cp_sat=True).plan(construction)
    if plan.planner != 'tail_first':
        raise ValueError(f'tail-first construction is unsupported: {plan.fallback_reason}')
    return construction, plan


def tail_first_proposal_stream_v1(rules, execution, parameters, *, scenario, request, proof, search,
                                  playing_quality, campaign_started, cancel_check, progress_callback,
                                  worker_cpu_callback=None):
    """Backward construction supplies actions only; the native consumer proves them afresh."""
    from .construction_trace import construction_trace_prefix_v1
    from .ortools_research_executor import _tail_early_policy_steps_v1
    from .scenario_contracts import ExactTracePrefixV1

    construction, plan = family_tail_first_plan_v1(rules, execution,
        scenario=scenario, request=request, playing_quality=playing_quality)
    construction_request = request.to_dict()
    construction_request.pop('sha256')
    construction_request['visibility'] = 'complete_state'
    construction_request = ScenarioProofRequestV1.build(construction_request)
    deadline = campaign_started + parameters['maximumCampaignSeconds']

    def stopped():
        return time.perf_counter() >= deadline or cancel_check()

    compiler = OrToolsTailFirstCandidateCompilerV1(rules=rules, plan=plan,
        proof_request=construction_request, maximum_proof_nodes=proof['maximumNodes'],
        maximum_proof_depth=proof['maximumDepth'], maximum_line_count_limbs=parameters['maximumLineCountLimbs'],
        maximum_conflicts=parameters['maximumConflicts'],
        maximum_prefix_lift_attempts=search['maximumPrefixLiftAttempts'],
        native_cp_sat_scoring=True, cancel_check=stopped,
        policy_prefix_steps=_tail_early_policy_steps_v1(plan.prefix_length, playing_quality, execution),
        policy_id=execution.early_play_policy_id,
        require_optimal_auction_actions=playing_quality.uses_auction_hand_boundary)
    excluded = []
    for offset in range(min(execution.proposal_seed_count, parameters['maximumCandidates'])):
        remaining = deadline - time.perf_counter()
        if remaining <= 0 or stopped():
            return
        seed = parameters['seed'] + offset
        started = time.perf_counter()
        attempt = compiler(construction, tuple(excluded), seed=seed,
            maximum_seconds=min(remaining, parameters['maximumSeconds'],
                execution.proposal_maximum_seconds or parameters['maximumSeconds']),
            maximum_conflicts=parameters['maximumConflicts'])
        progress = dict(currentStage='tail_first_proposal', screenedSeed=seed, proposalOnly=True,
            constructionStatus=attempt.status, constructionReason=attempt.reason,
            constructionElapsedSeconds=time.perf_counter() - started, proposalCount=0,
            qualificationScope='construction only; actor-information-set proof still required')
        if attempt.status != 'feasible' or attempt.candidate is None:
            progress_callback(progress)
            continue
        # Do not forward the complete-state graph, proof, state, or predicate receipts.
        complete = construction_trace_prefix_v1(attempt.candidate.trace).to_dict()
        index = plan.prefix_length
        focal = complete['playActions'][index]
        prefix = {key: value for key, value in complete.items() if key != 'sha256'}
        prefix['playActions'] = prefix['playActions'][:index]
        prefix = ExactTracePrefixV1.build(prefix)
        source = dict(constructor='native_tail_first_proposal_v1', scenario=construction.to_dict(),
            request=construction_request.to_dict(), plan=plan.to_dict(), completePrefix=complete, seed=seed,
            rulesHash=rules.sha256)
        row = dict(prefix=prefix.to_dict(), prefixHash=prefix.sha256, rulesHash=rules.sha256,
            policyHash=canonical_sha256(execution.video_interestingness), playIndex=index,
            actor=focal['actor'], action=LegalPlayActionV1(Card.parse(focal['card']), focal.get('ekka') is True).code,
            sourceHash=canonical_sha256(source), sourceSeed=seed)
        excluded.append(tuple(complete['deckOrder']))
        progress_callback({**progress, 'proposalCount': 1, 'constructionSource': source})
        yield seed, {**row, 'sha256': canonical_sha256(row)}


__all__ = ["OrToolsTailFirstCandidateCompilerV1", "family_tail_first_plan_v1", "tail_first_proposal_stream_v1"]
