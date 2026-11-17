from __future__ import annotations

import ast
from math import ceil
from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat, Suit, Team
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.research.ortools_proof import (
    ContinuationLineCountV1,
    OrToolsGraphProofOutcomeV1,
    ProofRootActionV1,
)
from baloot_platform.research.scenario_contracts import (
    ScenarioExpressionV1,
    ScenarioPredicateV1,
)
from baloot_platform.research.scenario_predicate_lowering_v2 import (
    CpSatActionVariablesV1,
)
from baloot_platform.research.scenario_proof_tactic_predicates_v2 import (
    PROOF_TACTIC_PREDICATE_REGISTRATIONS_V2,
    PrincipalVariationPlayEvidenceV1,
    PrincipalVariationTrickEvidenceV1,
    ProofTacticPredicateContextV1,
    ProofTacticPredicateLoweringV2,
    RegisteredTacticalEvidenceV1,
)
from baloot_platform.tactics import TacticalDetectorProfileV1, TacticalKind


def _outcome(
    *,
    completion_status: str = "exact_complete",
    solver_status: str = "optimal",
    lines_complete: bool = True,
    root_values: tuple[int, int, int] = (-1, -1, 1),
    forced: bool = False,
    selected_card: str = "CA",
) -> OrToolsGraphProofOutcomeV1:
    lines = (
        ContinuationLineCountV1.exact(
            line_universe="all_legal",
            count=12,
            limb_count=1,
            limb_limit=4,
            solver_status="optimal",
            model_hash="a" * 64,
        )
        if lines_complete
        else ContinuationLineCountV1.incomplete(
            line_universe="all_legal",
            upper_bound=20,
            limb_count=1,
            limb_limit=4,
            solver_status="unknown",
            model_hash="a" * 64,
            exhaustion_reason="line_count_incomplete",
        )
    )
    roots = (
        ProofRootActionV1(0, "D7", root_values[0], root_values[0] >= 1),
        ProofRootActionV1(1, "D8", root_values[1], root_values[1] >= 1),
        ProofRootActionV1(2, selected_card, root_values[2], root_values[2] >= 1),
    )
    if forced:
        roots = roots[2:]
    return OrToolsGraphProofOutcomeV1.build(
        completion_status=completion_status,
        solver_status=solver_status,
        claim_satisfied=completion_status == "exact_complete",
        selected_action=selected_card,
        winning_root_action_count=sum(row.satisfies for row in roots),
        root_actions=roots,
        principal_variation=(selected_card, "S7", "HA", "C8"),
        continuation_lines=lines,
        policies=(),
        graph_hash="b" * 64,
        model_hash="c" * 64,
        request_hash="d" * 64,
        variable_count=8,
        constraint_count=12,
    )


def _trick() -> PrincipalVariationTrickEvidenceV1:
    return PrincipalVariationTrickEvidenceV1.build(
        trick_index=7,
        plays=(
            PrincipalVariationPlayEvidenceV1.build("CA", "west"),
            PrincipalVariationPlayEvidenceV1.build("S7", "north"),
            PrincipalVariationPlayEvidenceV1.build("HA", "south"),
            PrincipalVariationPlayEvidenceV1.build("C8", "east"),
        ),
        winner_position=3,
    )


def _tactic(proof_hash: str, *, applicable: bool = True, verified: bool = True):
    return RegisteredTacticalEvidenceV1.build(
        detector_id="only_winning_move",
        detector_version="1.0.0",
        evidence_id="TACTIC-1",
        applicable=applicable,
        verified=verified,
        score=100,
        acting_seat="west",
        observer="west",
        target_team="west_east",
        causal_proof_hash="e" * 64 if verified else None,
        source_proof_hash=proof_hash,
    )


def _context(
    *,
    outcome: OrToolsGraphProofOutcomeV1 | None = None,
    evidence: tuple[RegisteredTacticalEvidenceV1, ...] | None = None,
) -> ProofTacticPredicateContextV1:
    proof = outcome or _outcome()
    return ProofTacticPredicateContextV1(
        cp_model.CpModel(),
        "proof-tactic-fixture",
        proof,
        (_trick(),),
        evidence if evidence is not None else (_tactic(proof.sha256),),
    )


def _native_tactic_context(
    *,
    first_winner: Team,
    proof_utility: str = "awarded_point_margin",
    decision_kind: str = "play",
    proof: OrToolsGraphProofOutcomeV1 | None = None,
    principal_winners: tuple[Seat, ...] = (),
    trump_suit: Suit | None = None,
    base_tricks: tuple[int, int] = (0, 0),
    root_actor_is_target: bool = True,
):
    model = cp_model.CpModel()
    proof = proof or _outcome()
    roots = tuple(
        CpSatActionVariablesV1(
            f"ROOT-{row.action_id}",
            model.new_constant(LegalPlayActionV1.from_code(row.action_code).card.id),
            model.new_constant(int(Seat.WEST)),
            model.new_constant(int(Seat.WEST)),
            model.new_constant(0),
            model.new_constant(1),
            model.new_constant(int(row.satisfies)),
            model.new_constant(row.value),
        )
        for row in proof.root_actions
    )
    winner_seats = principal_winners or (Seat.NORTH, Seat.WEST)
    return SimpleNamespace(
        model=model,
        context_id="native-tactic-fixture",
        proof=proof,
        principal_variation_tricks=(
            tuple(
                SimpleNamespace(
                    trick_index=index,
                    winner=SimpleNamespace(actor_seat=winner.name.lower()),
                )
                for index, winner in enumerate(principal_winners)
            )
            if principal_winners
            else (_trick(),)
        ),
        tactical_evidence=(),
        root_actions=roots,
        root_winning_variables=tuple(item.winning for item in roots),
        root_value_variables=tuple(item.value for item in roots),
        root_actor_seat=int(Seat.WEST),
        root_actor_team=int(Team.WEST_EAST),
        proof_utility=proof_utility,
        tactical_detector_profile=TacticalDetectorProfileV1.defaults(),
        causal_gain_scale=1,
        causal_gain_terminal_awarded_scope=proof_utility == "awarded_point_margin",
        decision_kind=decision_kind,
        selected_root_action_index=len(roots) - 1,
        root_actor_is_target_variable=model.new_constant(
            int(root_actor_is_target)
        ),
        target_seat_variable=model.new_constant(int(Seat.WEST)),
        card_strength_by_id=tuple(int(Card(card_id).rank) for card_id in range(32)),
        first_principal_trick_winner_team_variable=model.new_constant(
            int(first_winner)
        ),
        trump_suit_variable=(
            model.new_constant(int(trump_suit)) if trump_suit is not None else None
        ),
        base_trick_variables=tuple(model.new_constant(value) for value in base_tricks),
        principal_trick_winner_seat_variables=tuple(
            model.new_constant(int(winner)) for winner in winner_seats
        ),
        principal_variation_has_continuation_variable=model.new_constant(
            int(len(proof.principal_variation) > 1)
        ),
        sha256="f" * 64,
    )


def _catalog_tactic_context(kind: TacticalKind, *, applicable: bool):
    """Return one native positive or semantic non-applicability fixture."""

    if kind == TacticalKind.SMALL_CARD_BIG_GAIN:
        context = _native_tactic_context(first_winner=Team.NORTH_SOUTH)
        if applicable:
            context.card_strength_by_id = tuple(
                100 - int(Card(card_id).rank) for card_id in range(32)
            )
        return context
    if kind == TacticalKind.LARGE_CARD_SACRIFICE:
        return _native_tactic_context(
            first_winner=(Team.NORTH_SOUTH if applicable else Team.WEST_EAST)
        )
    if kind == TacticalKind.CONTRACT_RESCUE:
        return _native_tactic_context(
            first_winner=Team.NORTH_SOUTH,
            proof_utility="contract_success",
            proof=(
                _outcome(root_values=(-1, -1, 1))
                if applicable
                else _outcome(root_values=(1, 0, -1))
            ),
        )
    if kind == TacticalKind.PROJECT_BIDDING:
        return _native_tactic_context(
            first_winner=Team.NORTH_SOUTH,
            decision_kind="project",
            proof=(
                _outcome(root_values=(-1, -1, 1))
                if applicable
                else _outcome(root_values=(1, 1, 1))
            ),
        )
    if kind == TacticalKind.DEBATE:
        return _native_tactic_context(
            first_winner=Team.NORTH_SOUTH,
            proof=(
                _outcome(root_values=(-1, 1, 1))
                if applicable
                else _outcome(root_values=(1, 0, -1))
            ),
        )
    if kind == TacticalKind.ENTRY_UNBLOCKING:
        return _native_tactic_context(
            first_winner=Team.NORTH_SOUTH,
            principal_winners=(
                (Seat.NORTH, Seat.WEST)
                if applicable
                else (Seat.NORTH, Seat.NORTH)
            ),
        )
    if kind == TacticalKind.PARTNER_ENABLEMENT:
        return _native_tactic_context(
            first_winner=Team.NORTH_SOUTH,
            principal_winners=(
                (Seat.NORTH, Seat.EAST)
                if applicable
                else (Seat.NORTH, Seat.WEST)
            ),
        )
    if kind == TacticalKind.TRUMP_CONTROL:
        return _native_tactic_context(
            first_winner=Team.NORTH_SOUTH,
            trump_suit=(Suit.CLUBS if applicable else Suit.SPADES),
        )
    if kind == TacticalKind.KABOUT_CREATION:
        return _native_tactic_context(
            first_winner=Team.NORTH_SOUTH,
            proof_utility="kabout",
            base_tricks=((0, 0) if applicable else (0, 1)),
        )
    if kind == TacticalKind.KABOUT_PRESERVATION:
        return _native_tactic_context(
            first_winner=Team.NORTH_SOUTH,
            proof_utility="kabout",
            base_tricks=((0, 3) if applicable else (1, 3)),
        )
    if kind == TacticalKind.KABOUT_PREVENTION:
        return _native_tactic_context(
            first_winner=Team.NORTH_SOUTH,
            proof_utility="kabout",
            base_tricks=((3, 0) if applicable else (3, 1)),
            root_actor_is_target=False,
            proof=(
                _outcome(root_values=(1, 1, -1))
                if applicable
                else _outcome(root_values=(1, 0, -1))
            ),
        )
    if kind == TacticalKind.ONLY_WINNING_MOVE:
        return _native_tactic_context(
            first_winner=Team.NORTH_SOUTH,
            proof=(
                _outcome(root_values=(-1, -1, 1))
                if applicable
                else _outcome(root_values=(1, 0, -1))
            ),
        )
    raise AssertionError(f"missing tactical qualification fixture: {kind.value}")


def _predicate(kind: str, arguments: dict, *, predicate_id: str = "PRED-DEFAULT"):
    return ScenarioPredicateV1.build(
        {
            "predicateId": predicate_id,
            "kind": kind,
            "version": 1,
            "hardness": "hard",
            "arguments": arguments,
        }
    )


def _solve(lowering, model: cp_model.CpModel):
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    return solver, lowering.extract(solver)


@pytest.mark.parametrize(
    "kind,arguments,observed_key,observed_value",
    [
        (
            "proof.winning_root_action_count",
            {"minimum": 1, "maximum": 1},
            "winningCount",
            1,
        ),
        (
            "proof.winning_root_card_rank_count",
            {"rank": ["ace"], "minimum": 1, "maximum": 1},
            "matchingCount",
            1,
        ),
        (
            "proof.winning_continuation_line_count",
            {"lineUniverse": "all_legal", "minimum": "12", "maximum": "12"},
            "exactCount",
            "12",
        ),
        (
            "proof.principal_variation_trick_winner",
            {"trickIndex": 7, "team": "west_east", "rank": ["eight"]},
            "proofComplete",
            True,
        ),
        (
            "proof.principal_variation_trick_team_rank_count",
            {
                "trickIndex": 7,
                "team": "north_south",
                "rank": ["ace"],
                "minimum": 1,
                "maximum": 1,
            },
            "matchingCount",
            1,
        ),
        (
            "tactic.verified",
            {
                "kind": "only_winning_move",
                "actingSeat": "west",
                "observer": "west",
                "targetTeam": "west_east",
                "causalRequired": True,
                "minimumScore": 100,
                "minimumEvidence": 1,
            },
            "matchingEvidenceCount",
            2,
        ),
        (
            "tactic.applicable",
            {
                "kind": "only_winning_move",
                "actingSeat": "west",
                "applicable": True,
            },
            "matchingEvidenceCount",
            2,
        ),
    ],
)
def test_all_proof_and_tactic_predicates_lower_with_immutable_evidence(
    kind, arguments, observed_key, observed_value
):
    context = (
        _native_tactic_context(first_winner=Team.NORTH_SOUTH)
        if kind.startswith(("proof.winning_root", "tactic."))
        else _context()
    )
    lowering = ProofTacticPredicateLoweringV2().lower_predicate(
        _predicate(kind, arguments), context
    )
    solver, evidence = _solve(lowering, context.model)

    assert solver.boolean_value(lowering.literal)
    assert evidence["observed"][observed_key] == observed_value
    assert evidence["observed"]["proofHash"] == context.proof.sha256
    assert (
        evidence["evidenceExtractor"] == PROOF_TACTIC_PREDICATE_REGISTRATIONS_V2[kind]
    )
    assert len(evidence["sha256"]) == 64


@pytest.mark.parametrize(
    "winner,card_code,expected",
    [
        (Seat.NORTH, "C8", False),
        (Seat.EAST, "C8", True),
        (Seat.SOUTH, "C8", False),
        (Seat.WEST, "C8", True),
        (Seat.WEST, "CA", False),
    ],
)
def test_native_principal_trick_winner_uses_live_seat_and_card_bindings(
    winner, card_code, expected
):
    context = _native_tactic_context(first_winner=Team.NORTH_SOUTH)
    context.native_proof_bindings = True
    context.principal_trick_indexes = (7,)
    winner_var = context.model.new_int_var(0, 3, "winner")
    card_var = context.model.new_int_var(0, 31, "winning_card")
    context.principal_trick_winner_seat_variables = (winner_var,)
    context.principal_trick_winning_card_variables = (card_var,)
    lowering = ProofTacticPredicateLoweringV2().lower_predicate(
        _predicate(
            "proof.principal_variation_trick_winner",
            {"trickIndex": 7, "team": "west_east", "rank": ["eight"]},
        ),
        context,
    )
    context.model.add(winner_var == int(winner))
    context.model.add(card_var == LegalPlayActionV1.from_code(card_code).card.id)

    assert context.model.validate() == ""
    solver, evidence = _solve(lowering, context.model)
    assert solver.boolean_value(lowering.literal) is expected
    assert evidence["observed"]["proofHash"] == context.proof.sha256


@pytest.mark.parametrize(
    "actor,card_code,present,expected_count",
    [
        (Seat.NORTH, "CA", True, 1),
        (Seat.SOUTH, "CA", True, 1),
        (Seat.EAST, "CA", True, 0),
        (Seat.NORTH, "C8", True, 0),
        (Seat.NORTH, "CA", False, 0),
    ],
)
def test_native_principal_trick_rank_count_excludes_inactive_and_other_tricks(
    actor, card_code, present, expected_count
):
    context = _native_tactic_context(first_winner=Team.NORTH_SOUTH)
    context.native_proof_bindings = True
    actor_var = context.model.new_int_var(0, 3, "actor")
    card_var = context.model.new_int_var(0, 31, "card")
    present_var = context.model.new_bool_var("present")
    context.principal_action_trick_indexes = (7, 6)
    context.principal_action_actor_variables = (
        actor_var, context.model.new_constant(int(Seat.NORTH)),
    )
    context.principal_action_card_variables = (
        card_var, context.model.new_constant(LegalPlayActionV1.from_code("HA").card.id),
    )
    context.principal_action_present_variables = (
        present_var, context.model.new_constant(1),
    )
    lowering = ProofTacticPredicateLoweringV2().lower_predicate(
        _predicate(
            "proof.principal_variation_trick_team_rank_count",
            {"trickIndex": 7, "team": "north_south", "rank": ["ace"],
             "minimum": 1, "maximum": 1},
        ),
        context,
    )
    context.model.add(actor_var == int(actor))
    context.model.add(card_var == LegalPlayActionV1.from_code(card_code).card.id)
    context.model.add(present_var == int(present))

    assert context.model.validate() == ""
    solver, evidence = _solve(lowering, context.model)
    assert solver.boolean_value(lowering.literal) is (expected_count == 1)
    assert evidence["observed"]["matchingCount"] == expected_count


@pytest.mark.parametrize("completion_status", ["proof_incomplete", "model_error"])
@pytest.mark.parametrize(
    "kind,arguments",
    [
        ("proof.winning_root_action_count", {"minimum": 1, "maximum": 1}),
        (
            "proof.winning_root_card_rank_count",
            {"rank": ["ace"], "minimum": 1, "maximum": 1},
        ),
        (
            "proof.winning_continuation_line_count",
            {"lineUniverse": "all_legal"},
        ),
        (
            "proof.principal_variation_trick_winner",
            {"trickIndex": 7, "team": "west_east", "rank": ["eight"]},
        ),
        (
            "proof.principal_variation_trick_team_rank_count",
            {
                "trickIndex": 7,
                "team": "north_south",
                "rank": ["ace"],
                "minimum": 1,
                "maximum": 1,
            },
        ),
        (
            "tactic.verified",
            {"kind": "only_winning_move", "minimumScore": 0, "minimumEvidence": 0},
        ),
        (
            "tactic.applicable",
            {"kind": "only_winning_move", "applicable": True},
        ),
    ],
)
def test_incomplete_proof_makes_every_predicate_unavailable(
    completion_status, kind, arguments
):
    proof = _outcome(completion_status=completion_status, solver_status="unknown")
    context = _context(outcome=proof)
    with pytest.raises(ValueError, match="proof.*unavailable"):
        ProofTacticPredicateLoweringV2().lower_predicate(
            _predicate(kind, arguments), context
        )


def test_requested_tactic_tag_cannot_replace_native_solver_bindings():
    proof = _outcome()
    context = _context(outcome=proof)
    registry = ProofTacticPredicateLoweringV2()
    requested = (
        _predicate(
            "tactic.verified",
            {"kind": "only_winning_move", "minimumScore": 0, "minimumEvidence": 0},
            predicate_id="PRED-VERIFIED",
        ),
        _predicate(
            "tactic.applicable",
            {"kind": "only_winning_move", "applicable": False},
            predicate_id="PRED-APPLICABLE",
        ),
    )

    for item in requested:
        with pytest.raises(ValueError, match="native tactical root bindings"):
            registry.lower_predicate(item, context)


@pytest.mark.parametrize(
    ("kind", "context_kwargs", "missing_attribute", "message"),
    [
        (
            "only_winning_move",
            {},
            "root_actor_is_target_variable",
            "root-target binding",
        ),
        (
            "large_card_sacrifice",
            {},
            "first_principal_trick_winner_team_variable",
            "first-trick winner binding",
        ),
        (
            "entry_unblocking",
            {"principal_winners": (Seat.NORTH, Seat.WEST)},
            "principal_trick_winner_seat_variables",
            "winner bindings",
        ),
        (
            "trump_control",
            {"trump_suit": Suit.CLUBS},
            "trump_suit_variable",
            "trump/continuation bindings",
        ),
        (
            "kabout_creation",
            {"proof_utility": "kabout"},
            "base_trick_variables",
            "base-trick bindings",
        ),
    ],
)
def test_specialized_tactics_refuse_missing_native_semantic_bindings(
    kind, context_kwargs, missing_attribute, message
):
    context = _native_tactic_context(
        first_winner=Team.NORTH_SOUTH,
        **context_kwargs,
    )
    delattr(context, missing_attribute)
    with pytest.raises(ValueError, match=message):
        ProofTacticPredicateLoweringV2().lower_predicate(
            _predicate(
                "tactic.verified",
                {"kind": kind, "minimumScore": 0, "minimumEvidence": 0},
            ),
            context,
        )


@pytest.mark.parametrize(
    ("kind", "arguments"),
    [
        ("proof.winning_root_action_count", {"minimum": 1, "maximum": 1}),
        (
            "proof.winning_root_card_rank_count",
            {"rank": ["ace"], "minimum": 1, "maximum": 1},
        ),
    ],
)
def test_proof_root_predicates_refuse_python_outcome_fallback(kind, arguments):
    with pytest.raises(ValueError, match="native root .*bindings"):
        ProofTacticPredicateLoweringV2().lower_predicate(
            _predicate(kind, arguments), _context()
        )


def test_only_winning_move_requires_the_selected_action_to_be_the_sole_winner():
    context = _native_tactic_context(
        first_winner=Team.NORTH_SOUTH,
        proof=_outcome(root_values=(1, -1, -1)),
    )
    lowering = ProofTacticPredicateLoweringV2().lower_predicate(
        _predicate(
            "tactic.verified",
            {
                "kind": "only_winning_move",
                "minimumScore": 0,
                "minimumEvidence": 0,
            },
        ),
        context,
    )
    solver, evidence = _solve(lowering, context.model)

    assert not solver.boolean_value(lowering.literal)
    assert evidence["observed"]["verified"] is False
    assert evidence["observed"]["matchingEvidenceCount"] == 0


def test_native_tactic_target_seat_and_target_team_filters_are_exact():
    registry = ProofTacticPredicateLoweringV2()
    positive = _native_tactic_context(first_winner=Team.NORTH_SOUTH)
    matching = registry.lower_predicate(
        _predicate(
            "tactic.verified",
            {
                "kind": "only_winning_move",
                "targetSeat": "west",
                "targetTeam": "west_east",
                "minimumScore": 100,
                "minimumEvidence": 2,
            },
        ),
        positive,
    )
    mismatching = registry.lower_predicate(
        _predicate(
            "tactic.verified",
            {
                "kind": "only_winning_move",
                "targetSeat": "east",
                "targetTeam": "west_east",
                "minimumScore": 100,
                "minimumEvidence": 2,
            },
            predicate_id="PRED-TARGET-MISS",
        ),
        positive,
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(positive.model) == cp_model.OPTIMAL
    assert solver.boolean_value(matching.literal)
    assert not solver.boolean_value(mismatching.literal)

    opponent = _native_tactic_context(
        first_winner=Team.NORTH_SOUTH,
        root_actor_is_target=False,
    )
    team_filter = registry.lower_predicate(
        _predicate(
            "tactic.applicable",
            {
                "kind": "only_winning_move",
                "targetTeam": "north_south",
                "applicable": True,
            },
        ),
        opponent,
    )
    solver, _ = _solve(team_filter, opponent.model)
    assert solver.boolean_value(team_filter.literal)
    missing_target = _native_tactic_context(
        first_winner=Team.NORTH_SOUTH,
        root_actor_is_target=False,
    )
    del missing_target.target_seat_variable
    with pytest.raises(ValueError, match="target-seat binding"):
        registry.lower_predicate(
            _predicate(
                "tactic.applicable",
                {
                    "kind": "only_winning_move",
                    "targetSeat": "north",
                    "applicable": True,
                },
            ),
            missing_target,
        )


def test_causal_required_rejects_a_verified_but_all_tied_debate():
    registry = ProofTacticPredicateLoweringV2()
    context = _native_tactic_context(
        first_winner=Team.NORTH_SOUTH,
        proof=_outcome(root_values=(1, 1, 1)),
    )
    without_causal = registry.lower_predicate(
        _predicate(
            "tactic.verified",
            {
                "kind": "debate",
                "causalRequired": False,
                "minimumScore": 100,
                "minimumEvidence": 2,
            },
        ),
        context,
    )
    with_causal = registry.lower_predicate(
        _predicate(
            "tactic.verified",
            {
                "kind": "debate",
                "causalRequired": True,
                "minimumScore": 100,
                "minimumEvidence": 2,
            },
            predicate_id="PRED-CAUSAL",
        ),
        context,
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(context.model) == cp_model.OPTIMAL
    assert solver.boolean_value(without_causal.literal)
    assert not solver.boolean_value(with_causal.literal)
    assert with_causal.extract(solver)["observed"]["causalProven"] is False


def test_minimum_score_and_evidence_use_native_nonbinary_witnesses():
    registry = ProofTacticPredicateLoweringV2()
    context = _native_tactic_context(
        first_winner=Team.NORTH_SOUTH,
        proof_utility="contract_success",
        proof=_outcome(root_values=(-1, 1, 1)),
    )

    def lower(minimum_score: int, minimum_evidence: int, predicate_id: str):
        return registry.lower_predicate(
            _predicate(
                "tactic.verified",
                {
                    "kind": "contract_rescue",
                    "minimumScore": minimum_score,
                    "minimumEvidence": minimum_evidence,
                },
                predicate_id=predicate_id,
            ),
            context,
        )

    accepted = lower(50, 1, "PRED-THRESHOLD-PASS")
    score_rejected = lower(51, 1, "PRED-SCORE-FAIL")
    count_rejected = lower(50, 2, "PRED-COUNT-FAIL")
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(context.model) == cp_model.OPTIMAL
    assert solver.boolean_value(accepted.literal)
    assert not solver.boolean_value(score_rejected.literal)
    assert not solver.boolean_value(count_rejected.literal)
    evidence = accepted.extract(solver)["observed"]
    assert evidence["matchingEvidenceCount"] == 1
    assert evidence["maximumEvidenceCount"] == 2
    assert evidence["score"] == 50
    assert evidence["scorePolicy"] == "qualifying_root_witness_ratio_v1"


@pytest.mark.parametrize("tactic", tuple(TacticalKind), ids=lambda item: item.value)
def test_every_registered_tactic_has_complete_native_qualification_evidence(tactic):
    registry = ProofTacticPredicateLoweringV2()
    predicate_token = tactic.value.upper().replace("_", "-")
    positive = _catalog_tactic_context(tactic, applicable=True)
    expected_score = 50 if tactic == TacticalKind.DEBATE else 100
    expected_evidence = 1 if tactic == TacticalKind.DEBATE else 2
    verified = registry.lower_predicate(
        _predicate(
            "tactic.verified",
            {
                "kind": tactic.value,
                "minimumScore": expected_score,
                "minimumEvidence": expected_evidence,
                "causalRequired": True,
            },
            predicate_id=f"PRED-{predicate_token}-VERIFIED",
        ),
        positive,
    )
    applicable = registry.lower_predicate(
        _predicate(
            "tactic.applicable",
            {"kind": tactic.value, "applicable": True},
            predicate_id=f"PRED-{predicate_token}-APPLICABLE",
        ),
        positive,
    )
    score_rejected = None
    if expected_score < 100:
        score_rejected = registry.lower_predicate(
            _predicate(
                "tactic.verified",
                {
                    "kind": tactic.value,
                    "minimumScore": expected_score + 1,
                    "minimumEvidence": expected_evidence,
                    "causalRequired": True,
                },
                predicate_id=f"PRED-{predicate_token}-SCORE-REJECTED",
            ),
            positive,
        )
    evidence_rejected = registry.lower_predicate(
        _predicate(
            "tactic.verified",
            {
                "kind": tactic.value,
                "minimumScore": expected_score,
                "minimumEvidence": expected_evidence + 1,
                "causalRequired": True,
            },
            predicate_id=f"PRED-{predicate_token}-EVIDENCE-REJECTED",
        ),
        positive,
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(positive.model) == cp_model.OPTIMAL
    observed = verified.extract(solver)["observed"]
    assert solver.boolean_value(verified.literal)
    assert solver.boolean_value(applicable.literal)
    if score_rejected is not None:
        assert not solver.boolean_value(score_rejected.literal)
    assert not solver.boolean_value(evidence_rejected.literal)
    assert observed["applicable"] is True
    assert observed["verified"] is True
    assert observed["causalProven"] is True
    assert observed["score"] == expected_score
    assert observed["matchingEvidenceCount"] == expected_evidence

    negative = _catalog_tactic_context(tactic, applicable=False)
    non_applicable = registry.lower_predicate(
        _predicate(
            "tactic.applicable",
            {"kind": tactic.value, "applicable": False},
            predicate_id=f"PRED-{predicate_token}-NOT-APPLICABLE",
        ),
        negative,
    )
    negative_solver = cp_model.CpSolver()
    negative_solver.parameters.num_search_workers = 1
    assert negative_solver.solve(negative.model) == cp_model.OPTIMAL
    negative_observed = non_applicable.extract(negative_solver)["observed"]
    assert negative_solver.boolean_value(non_applicable.literal)
    assert negative_observed["applicable"] is False
    assert negative_observed["verified"] is False


@pytest.mark.parametrize("tactic", tuple(TacticalKind), ids=lambda item: item.value)
def test_forced_root_is_proven_non_applicable_not_incomplete(tactic):
    context = _catalog_tactic_context(tactic, applicable=True)
    proof = _outcome(forced=True)
    forced = _native_tactic_context(first_winner=Team.NORTH_SOUTH, proof=proof,
        decision_kind=context.decision_kind, proof_utility=context.proof_utility)
    forced.native_proof_bindings = True
    registry = ProofTacticPredicateLoweringV2()
    absent = registry.lower_predicate(_predicate("tactic.applicable",
        {"kind": tactic.value, "applicable": False}, predicate_id="PRED-FORCED-ABSENT"), forced)
    unverified = registry.lower_predicate(_predicate("tactic.verified",
        {"kind": tactic.value, "minimumScore": 0, "minimumEvidence": 0,
            "causalRequired": False}, predicate_id="PRED-FORCED-UNVERIFIED"), forced)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(forced.model) == cp_model.OPTIMAL
    assert solver.boolean_value(absent.literal)
    assert not solver.boolean_value(unverified.literal)
    observed = absent.extract(solver)["observed"]
    assert observed["applicable"] is False and observed["verified"] is False
    assert observed["score"] == 0 and observed["matchingEvidenceCount"] == 0
    assert observed["causalProven"] is False
    assert observed["inapplicabilityReason"] == "required_alternative_absent"
    if tactic == TacticalKind.PROJECT_BIDDING:
        for field, value in (("decision_kind", "bid"), ("proof_utility", "round_win")):
            original = getattr(forced, field)
            setattr(forced, field, value)
            with pytest.raises(ValueError, match="project-consequence"):
                registry.lower_predicate(_predicate("tactic.applicable",
                    {"kind": tactic.value, "applicable": False}), forced)
            setattr(forced, field, original)


def test_large_card_sacrifice_is_native_and_requires_losing_the_first_trick():
    registry = ProofTacticPredicateLoweringV2()
    predicate = _predicate(
        "tactic.verified",
        {
            "kind": "large_card_sacrifice",
            "minimumScore": 100,
            "minimumEvidence": 1,
        },
    )
    positive = _native_tactic_context(first_winner=Team.NORTH_SOUTH)
    lowering = registry.lower_predicate(predicate, positive)
    solver, evidence = _solve(lowering, positive.model)
    assert solver.boolean_value(lowering.literal)
    assert evidence["observed"]["qualificationSource"] == (
        "native_cp_sat_root_value_variables"
    )
    assert evidence["observed"]["detectorVersion"] == "2.0.0"
    assert evidence["observed"]["applicable"] is True
    assert evidence["observed"]["verified"] is True
    assert evidence["observed"]["score"] == 100

    negative = _native_tactic_context(first_winner=Team.WEST_EAST)
    lowering = registry.lower_predicate(predicate, negative)
    solver, _evidence = _solve(lowering, negative.model)
    assert not solver.boolean_value(lowering.literal)


def test_small_card_big_gain_uses_native_card_and_value_bindings():
    registry = ProofTacticPredicateLoweringV2()
    predicate = _predicate(
        "tactic.verified",
        {
            "kind": "small_card_big_gain",
            "minimumScore": 100,
            "minimumEvidence": 1,
        },
    )
    positive = _native_tactic_context(first_winner=Team.NORTH_SOUTH)
    positive.card_strength_by_id = tuple(
        100 - int(Card(card_id).rank) for card_id in range(32)
    )
    lowering = registry.lower_predicate(predicate, positive)
    solver, _evidence = _solve(lowering, positive.model)
    assert solver.boolean_value(lowering.literal)

    negative = _native_tactic_context(first_winner=Team.NORTH_SOUTH)
    lowering = registry.lower_predicate(predicate, negative)
    solver, _evidence = _solve(lowering, negative.model)
    assert not solver.boolean_value(lowering.literal)


@pytest.mark.parametrize('native_authority', [False, True])
@pytest.mark.parametrize('causal_required', [False, True])
@pytest.mark.parametrize('live', [False, True])
@pytest.mark.parametrize('tactic', ['small_card_big_gain', 'large_card_sacrifice'])
def test_root_advantage_without_delayed_payoff_cannot_prove_native_small_card_causality(
    native_authority, causal_required, live, tactic,
):
    context = _native_tactic_context(first_winner=Team.NORTH_SOUTH)
    context.native_proof_bindings = live
    context.native_cp_sat_authority = native_authority
    context.tactical_detector_profile = TacticalDetectorProfileV1.defaults()
    if tactic == 'small_card_big_gain':
        context.card_strength_by_id = tuple(100 - int(Card(card).rank) for card in range(32))
    predicate = _predicate('tactic.verified', {'kind': tactic,
        'minimumScore': 100, 'minimumEvidence': 1, 'causalRequired': causal_required})
    lowering = ProofTacticPredicateLoweringV2().lower_predicate(predicate, context)
    solver, evidence = _solve(lowering, context.model)
    assert solver.boolean_value(lowering.literal) is (not native_authority or not causal_required)
    assert evidence['observed']['causalProven'] is (not native_authority)
    if native_authority:
        assert evidence['observed']['causalProofStatus'] == 'missing_native_delayed_payoff_witness'


@pytest.mark.parametrize('cutoff,qualifies', [(28.57, False), (28.58, True)])
def test_native_small_card_obeys_frozen_within_suit_cutoff(cutoff, qualifies):
    context = _native_tactic_context(first_winner=Team.WEST_EAST)
    context.native_proof_bindings = True
    context.native_cp_sat_authority = True
    values = [int(Card(card).rank) for card in range(32)]
    for code, strength in (('CA', 2), ('CK', 7), ('D7', 3), ('D8', 7)):
        values[Card.parse(code).id] = strength
    context.card_strength_by_id = tuple(values)
    raw = TacticalDetectorProfileV1.defaults().to_dict()
    raw.pop('sha256')
    raw['lowCardMaxStrength'] = cutoff
    context.tactical_detector_profile = TacticalDetectorProfileV1.build(raw)
    lowering = ProofTacticPredicateLoweringV2().lower_predicate(_predicate('tactic.verified', {
        'kind': 'small_card_big_gain', 'minimumScore': 100, 'minimumEvidence': 1}), context)
    solver, evidence = _solve(lowering, context.model)
    assert solver.boolean_value(lowering.literal) is qualifies
    observed = evidence['observed']
    assert observed['detectorProfileHash'] == context.tactical_detector_profile.sha256
    assert (observed['strengthNumerator'], observed['strengthSpan']) == (200, 7)
    assert observed['allowedNumerator'] == (200 if qualifies else 199)
    assert observed['causalProven'] is False


@pytest.mark.parametrize('tactic', ['small_card_big_gain', 'large_card_sacrifice', 'project_bidding'])
def test_native_small_card_requires_an_intact_frozen_profile(tactic):
    from dataclasses import replace
    for profile in (None, replace(TacticalDetectorProfileV1.defaults(), low_card_max_strength=36)):
        context = _native_tactic_context(first_winner=Team.WEST_EAST)
        context.native_proof_bindings = True
        context.native_cp_sat_authority = True
        context.tactical_detector_profile = profile
        if tactic == 'project_bidding':
            context.decision_kind = 'project'
            context.root_project_consequence_variables = context.root_value_variables
            context.project_consequence_scale = 1
        with pytest.raises(ValueError, match='profile'):
            ProofTacticPredicateLoweringV2().lower_predicate(_predicate('tactic.verified', {
                'kind': tactic, 'minimumScore': 0, 'minimumEvidence': 0}), context)


@pytest.mark.parametrize('cutoff,qualifies', [(57.14, True), (57.15, False)])
def test_native_large_card_obeys_frozen_within_suit_cutoff(cutoff, qualifies):
    context = _native_tactic_context(first_winner=Team.NORTH_SOUTH)
    context.native_proof_bindings = True
    context.native_cp_sat_authority = True
    values = [int(Card(card).rank) for card in range(32)]
    values[Card.parse('CA').id] = 4
    values[Card.parse('CK').id] = 7
    values[Card.parse('D7').id] = 3
    context.card_strength_by_id = tuple(values)
    raw = TacticalDetectorProfileV1.defaults().to_dict()
    raw.pop('sha256')
    context.tactical_detector_profile = TacticalDetectorProfileV1.build({**raw, 'highCardMinStrength': cutoff})
    lowering = ProofTacticPredicateLoweringV2().lower_predicate(_predicate('tactic.verified', {
        'kind': 'large_card_sacrifice', 'minimumScore': 100, 'minimumEvidence': 1}), context)
    solver, evidence = _solve(lowering, context.model)
    assert solver.boolean_value(lowering.literal) is qualifies
    observed = evidence['observed']
    assert observed['detectorProfileHash'] == context.tactical_detector_profile.sha256
    assert (observed['strengthNumerator'], observed['strengthSpan']) == (400, 7)
    assert observed['requiredNumerator'] == (400 if qualifies else 401)
    assert observed['highCardMinStrength'] == cutoff
    assert observed['detectorVersion'] == '3.0.1' and observed['causalProven'] is False


def test_contract_rescue_is_native_and_requires_contract_success_utility():
    registry = ProofTacticPredicateLoweringV2()
    predicate = _predicate(
        "tactic.verified",
        {
            "kind": "contract_rescue",
            "minimumScore": 100,
            "minimumEvidence": 1,
        },
    )
    positive = _native_tactic_context(
        first_winner=Team.NORTH_SOUTH,
        proof_utility="contract_success",
    )
    lowering = registry.lower_predicate(predicate, positive)
    solver, evidence = _solve(lowering, positive.model)
    assert solver.boolean_value(lowering.literal)
    assert evidence["observed"]["qualificationSource"] == (
        "native_cp_sat_root_value_variables"
    )

    unsupported = _native_tactic_context(first_winner=Team.NORTH_SOUTH)
    with pytest.raises(ValueError, match="contract_success utility"):
        registry.lower_predicate(predicate, unsupported)


def test_project_bidding_is_native_only_at_bid_or_project_decisions():
    registry = ProofTacticPredicateLoweringV2()
    predicate = _predicate(
        "tactic.verified",
        {
            "kind": "project_bidding",
            "minimumScore": 100,
            "minimumEvidence": 1,
        },
    )
    positive = _native_tactic_context(
        first_winner=Team.NORTH_SOUTH,
        decision_kind="project",
    )
    lowering = registry.lower_predicate(predicate, positive)
    solver, _evidence = _solve(lowering, positive.model)
    assert solver.boolean_value(lowering.literal)

    unsupported = _native_tactic_context(first_winner=Team.NORTH_SOUTH)
    with pytest.raises(ValueError, match="bid or project decision"):
        registry.lower_predicate(predicate, unsupported)

    # V3 live compilation must not bypass the catalog's causal-evidence refusal.
    positive.native_proof_bindings = True
    for kind, arguments in (
        ("tactic.verified", {"minimumScore": 100, "minimumEvidence": 1}),
        ("tactic.applicable", {"applicable": True}),
    ):
        with pytest.raises(ValueError, match="project-consequence proof bindings"):
            registry.lower_predicate(
                _predicate(kind, {"kind": "project_bidding", **arguments}), positive
            )


@pytest.mark.parametrize("selected_delta,other_delta,expected", [
    (0, 0, False), (3, 0, True), (-3, 0, False), (3, 3, False)])
@pytest.mark.parametrize("actor_is_target", [True, False])
def test_live_project_tactic_requires_project_specific_advantage(selected_delta, other_delta, expected, actor_is_target):
    sign = 1 if actor_is_target else -1
    context = _native_tactic_context(first_winner=Team.NORTH_SOUTH, decision_kind="project",
        root_actor_is_target=actor_is_target, proof=_outcome(root_values=(-sign, -sign, sign)))
    context.native_proof_bindings = True
    context.native_cp_sat_authority = False
    context.root_project_consequence_variables = tuple(
        context.model.new_constant(sign * (selected_delta if index == context.selected_root_action_index else other_delta))
        for index in range(len(context.root_actions)))
    predicate = _predicate("tactic.verified", {"kind": "project_bidding",
        "minimumScore": 100, "minimumEvidence": 1, "causalRequired": True})
    lowering = ProofTacticPredicateLoweringV2().lower_predicate(predicate, context)
    solver, _ = _solve(lowering, context.model)
    assert solver.boolean_value(lowering.literal) is expected


@pytest.mark.parametrize("selected,others,cutoff,scale,expected", [
    (10, (0, 7), 5, 1, False), (12, (0, 7), 5, 1, True),
    (4, (0, 0), 4, 1, True), (4, (0, 0), 4.01, 1, False),
    (4000, (0, 0), 4, 1000, True), (4000, (0, 0), 4.001, 1000, False),
    (0, (0, 0), 0, 1, False), (1, (0, 0), 0, 1, True),
])
@pytest.mark.parametrize("actor_is_target", [True, False])
def test_native_project_gain_compares_best_alternative_in_frozen_units(
    selected, others, cutoff, scale, expected, actor_is_target,
):
    sign = 1 if actor_is_target else -1
    context = _native_tactic_context(first_winner=Team.NORTH_SOUTH, decision_kind="project",
        root_actor_is_target=actor_is_target, proof=_outcome(root_values=(-sign, -sign, sign)))
    context.native_proof_bindings = True
    context.native_cp_sat_authority = True
    raw = TacticalDetectorProfileV1.defaults().to_dict()
    raw.pop("sha256")
    context.tactical_detector_profile = TacticalDetectorProfileV1.build({**raw, "projectConsequenceDelta": cutoff})
    context.project_consequence_scale = scale
    context.root_project_consequence_variables = tuple(context.model.new_constant(sign * value)
        for value in (*others, selected))
    predicate = _predicate("tactic.verified", {"kind": "project_bidding", "minimumScore": 0,
        "minimumEvidence": 1, "causalRequired": True})
    lowering = ProofTacticPredicateLoweringV2().lower_predicate(predicate, context)
    solver, evidence = _solve(lowering, context.model)
    assert solver.boolean_value(lowering.literal) is expected
    observed = evidence["observed"]
    assert observed["verified"] is expected and observed["causalProven"] is expected
    assert observed["detectorProfileHash"] == context.tactical_detector_profile.sha256
    assert observed["projectConsequenceScale"] == scale
    assert observed["projectConsequenceUnit"] == "awarded_point_margin"
    context.native_proof_bindings = False
    with pytest.raises(ValueError, match="owned live proof bindings"):
        ProofTacticPredicateLoweringV2().lower_predicate(predicate, context)


def test_debate_is_native_only_for_an_optimal_tied_alternative():
    registry = ProofTacticPredicateLoweringV2()
    predicate = _predicate(
        "tactic.verified",
        {
            "kind": "debate",
            "minimumScore": 50,
            "minimumEvidence": 1,
        },
    )
    positive = _native_tactic_context(
        first_winner=Team.NORTH_SOUTH,
        proof=_outcome(root_values=(-1, 1, 1)),
    )
    lowering = registry.lower_predicate(predicate, positive)
    solver, _evidence = _solve(lowering, positive.model)
    assert solver.boolean_value(lowering.literal)

    negative = _native_tactic_context(first_winner=Team.NORTH_SOUTH)
    lowering = registry.lower_predicate(predicate, negative)
    solver, _evidence = _solve(lowering, negative.model)
    assert not solver.boolean_value(lowering.literal)


@pytest.mark.parametrize(
    (
        "kind",
        "positive_winners",
        "negative_winners",
        "positive_trump",
        "negative_trump",
    ),
    [
        (
            "entry_unblocking",
            (Seat.NORTH, Seat.WEST),
            (Seat.NORTH, Seat.NORTH),
            None,
            None,
        ),
        (
            "partner_enablement",
            (Seat.NORTH, Seat.EAST),
            (Seat.NORTH, Seat.WEST),
            None,
            None,
        ),
        (
            "trump_control",
            (),
            (),
            Suit.CLUBS,
            Suit.SPADES,
        ),
    ],
)
def test_continuation_mechanism_tactics_are_native_and_fail_their_counterexample(
    kind,
    positive_winners,
    negative_winners,
    positive_trump,
    negative_trump,
):
    registry = ProofTacticPredicateLoweringV2()
    predicate = _predicate(
        "tactic.verified",
        {"kind": kind, "minimumScore": 100, "minimumEvidence": 1},
    )
    positive = _native_tactic_context(
        first_winner=Team.NORTH_SOUTH,
        principal_winners=positive_winners,
        trump_suit=positive_trump,
    )
    lowering = registry.lower_predicate(predicate, positive)
    solver, _evidence = _solve(lowering, positive.model)
    assert solver.boolean_value(lowering.literal)

    negative = _native_tactic_context(
        first_winner=Team.NORTH_SOUTH,
        principal_winners=negative_winners,
        trump_suit=negative_trump,
    )
    lowering = registry.lower_predicate(predicate, negative)
    solver, _evidence = _solve(lowering, negative.model)
    assert not solver.boolean_value(lowering.literal)


@pytest.mark.parametrize("values,winners,minimum_gain,scale,actor_is_target,expected,comparator", [
    ((-2, -1, 3), (Seat.NORTH, Seat.EAST, Seat.EAST), 0.01, 1, True, False, 1),
    ((-2, -1, 3), (Seat.NORTH, Seat.NORTH, Seat.EAST), 4.001, 1, True, False, 1),
    ((-2, -1, 3), (Seat.NORTH, Seat.NORTH, Seat.EAST), 4, 1, True, True, 1),
    ((-2, -1, 3), (Seat.NORTH, Seat.NORTH, Seat.EAST), 4, 1000, False, True, 1),
    ((-2, 3, 3), (Seat.NORTH, Seat.NORTH, Seat.EAST), 0, 1, True, False, 1),
    ((-1, -1, 3), (Seat.EAST, Seat.NORTH, Seat.EAST), 0.01, 1, True, False, 0),
    ((-1, -1, 3), (Seat.NORTH, Seat.EAST, Seat.EAST), 0.01, 1, True, True, 0),
])
def test_native_partner_enablement_requires_a_paired_best_alternative(
    values, winners, minimum_gain, scale, actor_is_target, expected, comparator,
):
    context = _native_tactic_context(first_winner=Team.NORTH_SOUTH,
        principal_winners=(Seat.NORTH, Seat.EAST), root_actor_is_target=actor_is_target,
        proof=_outcome(root_values=tuple(value * scale * (1 if actor_is_target else -1) for value in values)))
    context.native_proof_bindings = True
    context.causal_gain_scale = scale
    raw = TacticalDetectorProfileV1.defaults().to_dict()
    raw.pop("sha256")
    context.tactical_detector_profile = TacticalDetectorProfileV1.build({**raw, "minimumCausalGain": minimum_gain})
    # The worse D7 alternative must not stand in for the best alternative D8.
    context.root_raw_payoff_variables = tuple({
        "trick7WinnerSeat": context.model.new_constant(int(Seat.NORTH)),
        "trick8WinnerSeat": context.model.new_constant(int(winner)),
    } for winner in winners)
    predicate = _predicate("tactic.verified", {"kind": "partner_enablement",
        "minimumScore": 0, "minimumEvidence": 1, "causalRequired": True})
    lowering = ProofTacticPredicateLoweringV2().lower_predicate(predicate, context)
    solver, evidence = _solve(lowering, context.model)
    assert solver.boolean_value(lowering.literal) is expected, evidence
    observed = evidence["observed"]
    assert observed["causalProven"] is expected
    assert observed["bestAlternativeRootIndex"] == comparator
    assert observed["minimumCausalGainScaled"] == max(1, ceil(minimum_gain * scale))
    assert observed["mechanismScope"] == "paired_canonical_continuations"


@pytest.mark.parametrize("missing", ["tactical_detector_profile", "causal_gain_terminal_awarded_scope", "root_raw_payoff_variables"])
def test_native_partner_enablement_refuses_missing_owned_bindings(missing):
    context = _native_tactic_context(first_winner=Team.NORTH_SOUTH)
    context.native_proof_bindings = True
    context.root_raw_payoff_variables = ()
    setattr(context, missing, () if missing == "root_raw_payoff_variables" else None)
    predicate = _predicate("tactic.verified", {"kind": "partner_enablement", "minimumScore": 0, "minimumEvidence": 0})
    with pytest.raises(ValueError, match="frozen tactical|terminal awarded|paired root"):
        ProofTacticPredicateLoweringV2().lower_predicate(predicate, context)


def test_native_entry_requires_paired_causality_not_a_later_actor_win():
    context = _native_tactic_context(first_winner=Team.NORTH_SOUTH,
        principal_winners=(Seat.NORTH, Seat.WEST), proof=_outcome(root_values=(-2, -1, 3)))
    context.native_proof_bindings = True
    context.root_raw_payoff_variables = tuple({
        "trick7WinnerSeat": context.model.new_constant(int(Seat.NORTH)),
        "trick8WinnerSeat": context.model.new_constant(int(Seat.WEST)),
        "trick7LeaderSeat": context.model.new_constant(int(Seat.WEST)),
        "trick8LeaderSeat": context.model.new_constant(int(Seat.NORTH)),
        "trick7LeadCard": context.model.new_constant(Card.parse("H7").id),
        "trick8LeadCard": context.model.new_constant(Card.parse("D7").id),
        "trick7CardRawMargin": context.model.new_constant(-11),
        "trick8CardRawMargin": context.model.new_constant(10),
    } for _ in context.root_actions)
    predicate = _predicate("tactic.verified", {"kind": "entry_unblocking",
        "minimumScore": 0, "minimumEvidence": 1, "causalRequired": True})
    lowering = ProofTacticPredicateLoweringV2().lower_predicate(predicate, context)
    solver, evidence = _solve(lowering, context.model)
    assert not solver.boolean_value(lowering.literal), evidence


@pytest.mark.parametrize("selected_winners,other_winners,selected_margins,other_margins,selected_lead,other_lead,minimum_gain", [
    ((Seat.NORTH, Seat.WEST), (Seat.NORTH, Seat.NORTH), (-11, 10), (-11, -10), "D7", "D7", 4),
    ((Seat.NORTH, Seat.WEST), (Seat.NORTH, Seat.WEST), (-11, 10), (-11, 10), "C7", "D7", 4),
    ((Seat.WEST, Seat.WEST), (Seat.WEST, Seat.WEST), (11, 10), (11, 10), "C7", "D7", 4),
    ((Seat.NORTH, Seat.EAST), (Seat.NORTH, Seat.WEST), (-11, 10), (-11, 10), "D7", "D7", 4),
    ((Seat.WEST, Seat.NORTH), (Seat.NORTH, Seat.NORTH), (11, -10), (-11, -10), "D7", "D7", 4),
    ((Seat.NORTH, Seat.WEST), (Seat.NORTH, Seat.NORTH), (-11, 10), (-11, -10), "D7", "D7", 4.001),
    ((Seat.NORTH, Seat.WEST, Seat.WEST), (Seat.NORTH, Seat.NORTH, Seat.WEST),
        (-40, 5, 5), (-1, -5, 5), "D7", "D7", 4),
])
def test_native_entry_mechanisms_match_independent_paired_oracle(
    selected_winners, other_winners, selected_margins, other_margins, selected_lead, other_lead, minimum_gain,
):
    from baloot_platform.challenge.oracle import derive_exact_line_causality

    context = _native_tactic_context(first_winner=selected_winners[0].team,
        principal_winners=selected_winners, proof=_outcome(root_values=(-2, -1, 3)))
    context.native_proof_bindings = True
    profile = TacticalDetectorProfileV1.defaults().to_dict()
    profile.pop("sha256")
    context.tactical_detector_profile = TacticalDetectorProfileV1.build({**profile, "minimumCausalGain": minimum_gain})
    paths = []
    for winners, margins, lead in ((other_winners, other_margins, other_lead),
                                  (other_winners, other_margins, other_lead),
                                  (selected_winners, selected_margins, selected_lead)):
        row, facts = {}, []
        for index, (winner, margin) in enumerate(zip(winners, margins, strict=True)):
            label = f"trick{9 - len(winners) + index}"
            leader = Seat.WEST if not index else winners[index - 1]
            card = "H7" if not index else lead
            for suffix, value in (("WinnerSeat", int(winner)), ("LeaderSeat", int(leader)),
                                  ("LeadCard", Card.parse(card).id), ("CardRawMargin", margin)):
                row[label + suffix] = context.model.new_constant(value)
            facts.append(SimpleNamespace(winner=winner, leader=leader, lead_card=card, relative_points=margin,
                winning_card="CA", winning_card_strength=100, opponent_trumps_remaining=0, actor_team_trumps_remaining=0))
        paths.append((row, tuple(facts)))
    context.root_raw_payoff_variables = tuple(row for row, _ in paths)
    selected = SimpleNamespace(final_margin=3, action={"suit": "clubs", "rank": "ace"},
        future_tricks=paths[2][1], card_strength=100, sha256="a" * 64)
    other = SimpleNamespace(final_margin=-1, action={"suit": "diamonds", "rank": "eight"},
        future_tricks=paths[1][1], card_strength=0, sha256="b" * 64)
    reference = derive_exact_line_causality(selected, other, acting_seat=Seat.WEST, trump_suit=None)
    expected = bool({"entry", "forced_return", "partner_enablement"}.intersection(reference.mechanisms)) and minimum_gain <= 4
    predicate = _predicate("tactic.verified", {"kind": "entry_unblocking", "minimumScore": 100,
        "minimumEvidence": 1, "causalRequired": True})
    lowering = ProofTacticPredicateLoweringV2().lower_predicate(predicate, context)
    solver, evidence = _solve(lowering, context.model)
    assert solver.boolean_value(lowering.literal) is expected
    observed = evidence["observed"]
    assert observed["payoffDepth"] == reference.payoff_depth
    assert observed["entry"] is ("entry" in reference.mechanisms)
    assert observed["forcedReturn"] is ("forced_return" in reference.mechanisms)
    assert observed["partnerEnablement"] is ("partner_enablement" in reference.mechanisms)
    assert observed["bestAlternativeRootIndex"] == 1


@pytest.mark.parametrize("tactic", ["small_card_big_gain", "large_card_sacrifice"])
@pytest.mark.parametrize("actor_is_target", [True, False])
@pytest.mark.parametrize("metric,minimum_gain,case,expected", [
    ("raw_card_points", 26, "delayed", True),
    ("raw_card_points", 26.001, "delayed", False),
    ("awarded_point_margin", 4, "delayed", True),
    ("awarded_point_margin", 4.001, "delayed", False),
    ("raw_card_points", 1, "best_alternative_has_no_mechanism", False),
    ("awarded_point_margin", 1, "best_alternative_has_no_mechanism", False),
    ("awarded_point_margin", 1, "immediate_only", False),
    ("raw_card_points", 1, "first_trick_trump_exhaustion", True),
    ("raw_card_points", 1, "promotion_only", True),
])
def test_native_delayed_payoff_uses_one_paired_comparator_and_declared_gain_units(
    tactic, actor_is_target, metric, minimum_gain, case, expected,
):
    scale = 1000
    selected_card = "C7" if tactic == "small_card_big_gain" else "CA"
    sign = 1 if actor_is_target else -1
    context = _native_tactic_context(first_winner=Team.NORTH_SOUTH,
        root_actor_is_target=actor_is_target,
        proof=_outcome(root_values=tuple(sign * value * scale for value in (-2, -1, 3)),
            selected_card=selected_card))
    context.native_proof_bindings = True
    context.causal_gain_scale = scale
    raw = context.tactical_detector_profile.to_dict()
    raw.pop("sha256")
    context.tactical_detector_profile = TacticalDetectorProfileV1.build({**raw,
        "nativeDelayedPayoff": {"gainMetric": metric, "minimumGain": minimum_gain}})
    selected_winners = (Seat.NORTH, Seat.EAST, Seat.WEST)
    other_winners = (Seat.NORTH, Seat.NORTH, Seat.NORTH)
    selected_margins, other_margins = (-20, 20, 10), (-1, -5, -10)
    if case == "immediate_only":
        other_winners, other_margins = selected_winners, (-40, 20, 10)
    if case == "first_trick_trump_exhaustion":
        selected_winners = other_winners = (Seat.NORTH, Seat.WEST)
        selected_margins, other_margins = (-20, 30), (-1, 5)
    if case == "promotion_only":
        other_winners, other_margins = selected_winners, (-19, 18, 10)
        expected = tactic == "small_card_big_gain"
    paths = []
    for index in range(3):
        use_selected = index == 2 or index == 1 and case == "best_alternative_has_no_mechanism"
        winners = selected_winners if use_selected else other_winners
        margins = selected_margins if use_selected else other_margins
        if index == 1 and case == "best_alternative_has_no_mechanism":
            margins = (-19, 18, 10)
        row = {}
        for ordinal, (winner, margin) in enumerate(zip(winners, margins, strict=True)):
            label = f"trick{ordinal + 6}"
            for suffix, value in (("WinnerSeat", int(winner)), ("LeaderSeat", int(Seat.WEST if ordinal == 0 else winners[ordinal - 1])),
                    ("LeadCard", Card.parse(selected_card if ordinal == 0 else "C7").id),
                    ("CardRawMargin", margin), ("WinningCard", Card.parse("DA" if case == "promotion_only" and index != 2
                        else "C7" if winner.team == Seat.WEST.team else "HA").id),
                    ("OpponentTrumpsRemaining", int(case == "first_trick_trump_exhaustion" and index != 2 and ordinal == 0))):
                row[label + suffix] = context.model.new_constant(value)
        paths.append(row)
    context.root_raw_payoff_variables = tuple(paths)
    lowering = ProofTacticPredicateLoweringV2().lower_predicate(_predicate("tactic.verified", {
        "kind": tactic, "minimumScore": 100, "minimumEvidence": 1, "causalRequired": True}), context)
    solver, evidence = _solve(lowering, context.model)
    assert solver.boolean_value(lowering.literal) is expected
    observed = evidence["observed"]
    assert observed["verified"] is expected and observed["causalProven"] is expected
    assert observed["bestAlternativeRootIndex"] == 1
    assert observed["detectorVersion"] == "3.1.0"
    assert observed["nativeDelayedPayoff"] == {"gainMetric": metric, "minimumGain": minimum_gain}
    assert observed["nativeGainScale"] == (1 if metric == "raw_card_points" else scale)
    if case == "delayed":
        assert observed["nativeGain"] == (26 if metric == "raw_card_points" else 4 * scale)
        assert observed["laterCardMarginGain"] == 45
        assert observed["firstTrickConcession"] == 19
    if case == "first_trick_trump_exhaustion":
        assert observed["trumpExhaustion"] is True
        assert not any(observed[flag] for flag in ("entry", "promotion", "forcedReturn", "partnerEnablement"))
    if case == "promotion_only":
        assert observed["promotion"] is expected
        assert not any(observed[flag] for flag in ("entry", "trumpExhaustion", "forcedReturn", "partnerEnablement"))


@pytest.mark.parametrize('trump,card,actor,present,trick,verified', [
    (Suit.CLUBS, 'C8', Seat.NORTH, 1, 6, True),
    (Suit.CLUBS, 'D8', Seat.NORTH, 1, 6, False),
    (Suit.CLUBS, 'C8', Seat.EAST, 1, 6, False),
    (Suit.CLUBS, 'C8', Seat.NORTH, 0, 6, False),
    (Suit.CLUBS, 'C8', Seat.NORTH, 1, 7, False),
    (Suit.SPADES, 'S8', Seat.NORTH, 1, 6, False),
])
def test_native_trump_control_requires_a_present_opponent_extraction(trump, card, actor, present, trick, verified):
    context = _native_tactic_context(first_winner=Team.WEST_EAST, trump_suit=trump)
    context.native_proof_bindings = True
    context.trump_control_suffix_actions = tuple(
        (context.model.new_constant(Card.parse(code).id), context.model.new_constant(int(seat)),
         context.model.new_constant(active), index)
        for code, seat, active, index in (('CA', Seat.WEST, 1, 6), (card, actor, present, trick)))
    predicate = _predicate('tactic.verified', {'kind': 'trump_control',
        'minimumScore': 100, 'minimumEvidence': 1, 'causalRequired': True})
    lowering = ProofTacticPredicateLoweringV2().lower_predicate(predicate, context)
    solver, evidence = _solve(lowering, context.model)
    assert solver.boolean_value(lowering.literal) is verified
    observed = evidence['observed']
    assert observed['verified'] is verified and observed['causalProven'] is verified
    assert observed['trumpExtraction'] is verified
    assert observed['mechanismScope'] == 'selected_canonical_continuation'


def test_legacy_tail_trump_detector_retains_its_frozen_semantics():
    context = _native_tactic_context(first_winner=Team.WEST_EAST, trump_suit=Suit.CLUBS)
    context.native_proof_bindings = True
    context.native_cp_sat_authority = False
    predicate = _predicate('tactic.verified', {'kind': 'trump_control',
        'minimumScore': 100, 'minimumEvidence': 1, 'causalRequired': True})
    lowering = ProofTacticPredicateLoweringV2().lower_predicate(predicate, context)
    solver, evidence = _solve(lowering, context.model)
    assert solver.boolean_value(lowering.literal)
    assert evidence['observed']['detectorVersion'] == '2.0.0'
    assert 'mechanismScope' not in evidence['observed']


def test_native_trump_extraction_rejects_missing_or_invalid_suffix_bindings():
    predicate = _predicate('tactic.verified', {'kind': 'trump_control',
        'minimumScore': 0, 'minimumEvidence': 0})
    for suffix in (None, (), ((None, None, None, 6),), ((1, 1, 1, -1),)):
        context = _native_tactic_context(first_winner=Team.WEST_EAST, trump_suit=Suit.CLUBS)
        context.native_proof_bindings = True
        if suffix is not None:
            context.trump_control_suffix_actions = suffix
        with pytest.raises(ValueError, match='trump-extraction suffix bindings'):
            ProofTacticPredicateLoweringV2().lower_predicate(predicate, context)


@pytest.mark.parametrize(
    ("kind", "positive", "negative"),
    [
        (
            "kabout_creation",
            {"base_tricks": (0, 0)},
            {"base_tricks": (0, 1)},
        ),
        (
            "kabout_preservation",
            {"base_tricks": (0, 3)},
            {"base_tricks": (1, 3)},
        ),
        (
            "kabout_prevention",
            {
                "base_tricks": (3, 0),
                "root_actor_is_target": False,
                "proof": _outcome(root_values=(1, 1, -1)),
            },
            {
                "base_tricks": (3, 0),
                "root_actor_is_target": False,
            },
        ),
    ],
)
def test_kabout_tactics_use_exact_kabout_root_variables(kind, positive, negative):
    registry = ProofTacticPredicateLoweringV2()
    predicate = _predicate(
        "tactic.verified",
        {"kind": kind, "minimumScore": 100, "minimumEvidence": 1},
    )
    positive_context = _native_tactic_context(
        first_winner=Team.NORTH_SOUTH,
        proof_utility="kabout",
        **positive,
    )
    lowering = registry.lower_predicate(predicate, positive_context)
    solver, evidence = _solve(lowering, positive_context.model)
    assert solver.boolean_value(lowering.literal)
    assert evidence["observed"]["qualificationSource"] == (
        "native_cp_sat_root_value_variables"
    )

    negative_context = _native_tactic_context(
        first_winner=Team.NORTH_SOUTH,
        proof_utility="kabout",
        **negative,
    )
    lowering = registry.lower_predicate(predicate, negative_context)
    solver, evidence = _solve(lowering, negative_context.model)
    assert not solver.boolean_value(lowering.literal)
    assert evidence["observed"]["verified"] is False
    assert evidence["observed"]["score"] == 0


def test_registered_detector_and_proof_ownership_fail_closed_at_input():
    proof = _outcome()
    with pytest.raises(ValueError, match="not registered"):
        RegisteredTacticalEvidenceV1.build(
            detector_id="requested_tag_only",
            detector_version="1",
            evidence_id="BAD",
            applicable=True,
            verified=True,
            score=100,
            source_proof_hash=proof.sha256,
        )
    wrong = _tactic("f" * 64)
    with pytest.raises(ValueError, match="different proof"):
        _context(outcome=proof, evidence=(wrong,))


def test_principal_variation_facts_must_match_the_immutable_proof():
    proof = _outcome()
    bad = PrincipalVariationTrickEvidenceV1.build(
        trick_index=7,
        plays=(
            PrincipalVariationPlayEvidenceV1.build("DA", "west"),
            PrincipalVariationPlayEvidenceV1.build("S7", "north"),
            PrincipalVariationPlayEvidenceV1.build("HA", "south"),
            PrincipalVariationPlayEvidenceV1.build("C8", "east"),
        ),
        winner_position=3,
    )
    with pytest.raises(ValueError, match="does not match proof"):
        ProofTacticPredicateContextV1(
            cp_model.CpModel(),
            "bad-pv",
            proof,
            (bad,),
            (_tactic(proof.sha256),),
        )


def test_bounded_proof_with_incomplete_line_coverage_is_unavailable():
    context = _context(
        outcome=_outcome(completion_status="bounded_complete", lines_complete=False)
    )
    registry = ProofTacticPredicateLoweringV2()
    with pytest.raises(ValueError, match="continuation-line evidence is unavailable"):
        registry.lower_predicate(
            _predicate(
                "proof.winning_continuation_line_count",
                {"lineUniverse": "all_legal"},
            ),
            context,
        )


def test_invalid_ranges_do_not_qualify():
    registry = ProofTacticPredicateLoweringV2()

    with pytest.raises(ValueError, match="minimum cannot exceed maximum"):
        registry.lower_predicate(
            _predicate(
                "proof.winning_root_action_count",
                {"minimum": 2, "maximum": 1},
            ),
            _context(),
        )


def test_missing_principal_variation_trick_is_unavailable_not_false():
    context = ProofTacticPredicateContextV1(
        cp_model.CpModel(),
        "missing-pv-trick",
        _outcome(),
        (),
        (),
    )
    predicate = _predicate(
        "proof.principal_variation_trick_winner",
        {"trickIndex": 7, "team": "west_east", "rank": ["eight"]},
    )

    with pytest.raises(ValueError, match="principal-variation trick is unavailable"):
        ProofTacticPredicateLoweringV2().lower_predicate(predicate, context)


def test_not_cannot_turn_missing_principal_variation_evidence_true():
    expression = ScenarioExpressionV1.build(
        {
            "op": "not",
            "children": [
                _predicate(
                    "proof.principal_variation_trick_winner",
                    {
                        "trickIndex": 7,
                        "team": "west_east",
                        "rank": ["eight"],
                    },
                ).to_dict()
            ],
        }
    )
    context = ProofTacticPredicateContextV1(
        cp_model.CpModel(),
        "missing-pv-under-not",
        _outcome(),
        (),
        (),
    )

    with pytest.raises(ValueError, match="principal-variation trick is unavailable"):
        ProofTacticPredicateLoweringV2().lower_predicate(
            expression.predicates()[0], context
        )


def test_complete_policy_principal_variation_remains_qualifiable():
    context = _context(outcome=_outcome(completion_status="policy_complete"))
    lowering = ProofTacticPredicateLoweringV2().lower_predicate(
        _predicate(
            "proof.principal_variation_trick_winner",
            {"trickIndex": 7, "team": "west_east", "rank": ["eight"]},
        ),
        context,
    )
    solver, evidence = _solve(lowering, context.model)

    assert solver.boolean_value(lowering.literal)
    assert evidence["observed"]["proofComplete"] is True


def test_category_exports_exactly_the_five_proof_and_two_tactic_registrations():
    assert len(PROOF_TACTIC_PREDICATE_REGISTRATIONS_V2) == 7
    assert (
        sum(
            kind.startswith("proof.")
            for kind in PROOF_TACTIC_PREDICATE_REGISTRATIONS_V2
        )
        == 5
    )
    assert (
        sum(
            kind.startswith("tactic.")
            for kind in PROOF_TACTIC_PREDICATE_REGISTRATIONS_V2
        )
        == 2
    )


def test_category_has_no_engine_runtime_or_legacy_service_dependency():
    source = __import__(
        "baloot_platform.research.scenario_proof_tactic_predicates_v2",
        fromlist=["dummy"],
    ).__file__
    assert source is not None
    text = open(source, encoding="utf-8").read()
    assert "BalootEngine" not in text
    assert "proof_service" not in text
    assert "scenario_evidence" not in text


def test_tactic_qualification_never_reinserts_python_truth_as_cp_constants():
    module = __import__(
        "baloot_platform.research.scenario_proof_tactic_predicates_v2",
        fromlist=["dummy"],
    )
    source_path = module.__file__
    assert source_path is not None
    source = open(source_path, encoding="utf-8").read()
    marker_line = next(
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if 'if predicate.kind in {"tactic.verified", "tactic.applicable"}' in line
    )
    derived_constants = [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and node.lineno > marker_line
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "new_constant"
    ]
    assert derived_constants == []
