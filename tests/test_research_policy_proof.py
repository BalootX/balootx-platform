from __future__ import annotations

import itertools

import pytest
from ortools.sat.python import cp_model  # noqa: F401

from baloot_platform.core.cards import Card
from baloot_platform.research.policy_proof import (
    OrToolsReducedPolicyProofV1,
    ReducedActorSafePolicyProblemV1,
    ReducedPolicyProfileV1,
    ReducedWorldOutcomeV1,
    exhaustive_reference_action,
)


def _problem(*, method: str = "exhaustive_reference_fixture") -> ReducedActorSafePolicyProblemV1:
    return ReducedActorSafePolicyProblemV1.build(
        information_state_hash="observer-safe-state-a",
        actor="south",
        coverage_method=method,
        profile=ReducedPolicyProfileV1(outcome_boundary=1),
        worlds=(
            ReducedWorldOutcomeV1.build("world-b", {"S7": -8, "S8": 4, "S9": -2}),
            ReducedWorldOutcomeV1.build("world-a", {"S7": 0, "S8": 7, "S9": -1}),
        ),
    )


def test_exact_only_winning_move_produces_actor_visible_cut() -> None:
    proof = OrToolsReducedPolicyProofV1().prove(_problem(), actual_action="S7")

    assert proof.selected_action == Card.parse("S8")
    assert proof.reference_action == proof.selected_action
    assert proof.model_feasibility_status == "optimal"
    assert proof.strategic_truth == "exact_reduced_actor_safe"
    assert proof.only_winning_move is not None
    assert proof.only_winning_move.selected_action == Card.parse("S8")
    assert proof.action_cut is not None
    assert proof.action_cut.information_state_hash == "observer-safe-state-a"
    assert proof.action_cut.forbidden_action == Card.parse("S7")


def test_legal_optimal_status_is_not_strategic_truth_without_exhaustive_coverage() -> None:
    proof = OrToolsReducedPolicyProofV1().prove(_problem(method="sampled_hidden_deals"))

    assert proof.model_feasibility_status == "optimal"
    assert proof.strategic_truth == "provisional"
    assert proof.only_winning_move is None


def test_unrecognized_exhaustive_claim_cannot_promote_strategic_truth() -> None:
    proof = OrToolsReducedPolicyProofV1().prove(_problem(method="exhaustive_untrusted"))

    assert proof.model_feasibility_status == "optimal"
    assert proof.strategic_truth == "provisional"


def test_problem_rejects_action_domain_drift_between_hidden_worlds() -> None:
    with pytest.raises(ValueError, match="same actor action set"):
        ReducedActorSafePolicyProblemV1.build(
            information_state_hash="safe",
            actor="south",
            coverage_method="exhaustive_reference_fixture",
            worlds=(
                ReducedWorldOutcomeV1.build("a", {"S7": 1, "S8": 2}),
                ReducedWorldOutcomeV1.build("b", {"S7": 1, "S9": 2}),
            ),
        )


def test_cp_sat_matches_exhaustive_reference_across_reduced_outcome_matrices() -> None:
    actions = ("S7", "S8", "S9")
    # Exhaustively cover every 2-world matrix over a small signed utility domain.
    for values in itertools.product((-1, 0, 1), repeat=6):
        worlds = tuple(
            ReducedWorldOutcomeV1.build(
                f"w{world_index}",
                dict(zip(actions, values[world_index * 3 : world_index * 3 + 3], strict=True)),
            )
            for world_index in range(2)
        )
        problem = ReducedActorSafePolicyProblemV1.build(
            information_state_hash="safe",
            actor="south",
            coverage_method="exhaustive_reference_fixture",
            worlds=worlds,
        )
        proof = OrToolsReducedPolicyProofV1().prove(problem)
        assert proof.selected_action == exhaustive_reference_action(problem)


def test_evidence_is_deterministic_under_world_input_order() -> None:
    original = _problem()
    reversed_problem = ReducedActorSafePolicyProblemV1.build(
        information_state_hash=original.information_state_hash,
        actor=original.actor,
        coverage_method=original.coverage_method,
        worlds=tuple(reversed(original.worlds)),
        profile=original.profile,
    )
    first = OrToolsReducedPolicyProofV1().prove(original)
    second = OrToolsReducedPolicyProofV1().prove(reversed_problem)
    assert first.problem_hash == second.problem_hash
    assert first.sha256 == second.sha256
