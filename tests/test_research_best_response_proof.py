from __future__ import annotations

import itertools

import pytest

pytest.importorskip("ortools")

from baloot_platform.core.cards import Card
from baloot_platform.research.best_response_proof import (
    OrToolsBoundedBestResponseProofV1,
    ReducedBestResponseProblemV1,
    ReducedBestResponseProfileV1,
    ReducedGameNodeV1,
    ReducedGameWorldV1,
    ReducedRootActionV1,
    exhaustive_backward_induction,
)


def _opponent(*values: int) -> ReducedGameNodeV1:
    return ReducedGameNodeV1.decision(
        "opponent",
        {index: ReducedGameNodeV1.terminal(value) for index, value in enumerate(values)},
        information_state_hash="opponent-visible-a",
    )


def _partner(*values: int) -> ReducedGameNodeV1:
    return ReducedGameNodeV1.decision(
        "partner",
        {index: ReducedGameNodeV1.terminal(value) for index, value in enumerate(values)},
        information_state_hash="partner-visible-a",
    )


def _problem(*, maximum_regret: int = 0) -> ReducedBestResponseProblemV1:
    profile = ReducedBestResponseProfileV1(
        maximum_regret=maximum_regret,
        delayed_gain_minimum=5,
        strength_profile_hash="rules-strength-v1",
    )
    roots = (
        ReducedRootActionV1.build(
            "S7",
            _opponent(14, 15),
            immediate_margin=0,
            immediate_margin_certified=True,
            strength_percentile_bps=2_000,
        ),
        ReducedRootActionV1.build(
            "S8",
            _partner(1, 3),
            immediate_margin=1,
            immediate_margin_certified=True,
            strength_percentile_bps=4_000,
        ),
    )
    return ReducedBestResponseProblemV1.build(
        information_state_hash="safe-root",
        actor="south",
        coverage_method="exhaustive_reference_fixture",
        worlds=(ReducedGameWorldV1.build("world-a", roots),),
        profile=profile,
    )


def test_partner_maximizes_opponent_minimizes_and_regret_gate_is_hard() -> None:
    proof = OrToolsBoundedBestResponseProofV1().prove(_problem(), actual_action="S8")

    assert proof.policy_proof.selected_action == Card.parse("S7")
    assert proof.policy_proof.strategic_truth == "exact_reduced_actor_safe"
    assert proof.regret is not None
    assert proof.regret.maximum_paired_regret == 11
    assert not proof.regret.passed
    assert proof.qualification_status == "rejected_maximum_regret"
    assert all(item.parity for item in proof.continuations)


def test_small_card_big_gain_requires_exact_counterfactual_and_strength_profile() -> None:
    proof = OrToolsBoundedBestResponseProofV1().prove(_problem(), actual_action="S7")

    assert proof.qualification_status == "qualified"
    assert proof.small_card_big_gain is not None
    assert proof.small_card_big_gain.selected_action == Card.parse("S7")
    assert proof.small_card_big_gain.minimum_delayed_gain == 14
    assert proof.small_card_big_gain.counterfactual_margin_gap == 11


def test_sampled_coverage_cannot_create_small_card_causal_witness() -> None:
    base = _problem()
    sampled = ReducedBestResponseProblemV1.build(
        information_state_hash=base.information_state_hash,
        actor=base.actor,
        coverage_method="sampled_hidden_deals",
        worlds=base.worlds,
        profile=base.profile,
    )
    proof = OrToolsBoundedBestResponseProofV1().prove(sampled, actual_action="S7")
    assert proof.policy_proof.strategic_truth == "provisional"
    assert proof.qualification_status == "rejected_incomplete_truth"
    assert proof.small_card_big_gain is None


def test_node_budget_fails_closed() -> None:
    base = _problem()
    with pytest.raises(ValueError, match="node budget"):
        ReducedBestResponseProblemV1.build(
            information_state_hash=base.information_state_hash,
            actor=base.actor,
            coverage_method=base.coverage_method,
            worlds=base.worlds,
            profile=ReducedBestResponseProfileV1(maximum_nodes=2),
        )


def test_future_actor_information_set_cannot_choose_by_hidden_world() -> None:
    first = ReducedGameWorldV1.build(
        "one",
        (
            ReducedRootActionV1.build("S7", _opponent(1, 2)),
            ReducedRootActionV1.build("S8", _partner(3, 4)),
        ),
    )
    second = ReducedGameWorldV1.build(
        "two",
        (
            # The same opponent information hash with different utilities
            # would permit a hidden-world-dependent response, so it fails closed.
            ReducedRootActionV1.build("S7", _opponent(-5, 8)),
            ReducedRootActionV1.build("S8", _partner(3, 4)),
        ),
    )
    with pytest.raises(ValueError, match="non-anticipativity"):
        ReducedBestResponseProblemV1.build(
            information_state_hash="safe-root",
            actor="south",
            coverage_method="exhaustive_reference_fixture",
            worlds=(first, second),
        )


def test_cp_sat_subproblems_match_exhaustive_backward_induction() -> None:
    compiler = OrToolsBoundedBestResponseProofV1()
    for values in itertools.product((-2, 0, 3), repeat=4):
        node = ReducedGameNodeV1.decision(
            "partner",
            {
                "S7": ReducedGameNodeV1.decision(
                    "opponent",
                    {"H7": ReducedGameNodeV1.terminal(values[0]), "H8": ReducedGameNodeV1.terminal(values[1])},
                    information_state_hash="opponent-after-s7",
                ),
                "S8": ReducedGameNodeV1.decision(
                    "opponent",
                    {"H7": ReducedGameNodeV1.terminal(values[2]), "H8": ReducedGameNodeV1.terminal(values[3])},
                    information_state_hash="opponent-after-s8",
                ),
            },
            information_state_hash="partner-root",
        )
        cp_margin, status, _nodes = compiler._solve_tree(node)
        reference_margin, _line = exhaustive_backward_induction(node)
        assert status == "optimal"
        assert cp_margin == reference_margin


def test_best_response_evidence_is_deterministic() -> None:
    first = OrToolsBoundedBestResponseProofV1().prove(_problem(), actual_action="S7")
    second = OrToolsBoundedBestResponseProofV1().prove(_problem(), actual_action="S7")
    assert first.sha256 == second.sha256


def test_batched_cp_sat_and_low_startup_reference_select_the_same_action() -> None:
    reference_problem = _problem()
    cp_problem = ReducedBestResponseProblemV1.build(
        information_state_hash=reference_problem.information_state_hash,
        actor=reference_problem.actor,
        coverage_method=reference_problem.coverage_method,
        worlds=reference_problem.worlds,
        profile=ReducedBestResponseProfileV1(
            maximum_regret=reference_problem.profile.maximum_regret,
            delayed_gain_minimum=reference_problem.profile.delayed_gain_minimum,
            strength_profile_hash=reference_problem.profile.strength_profile_hash,
            solver_backend="batched_cp_sat",
        ),
    )
    reference = OrToolsBoundedBestResponseProofV1().prove(reference_problem)
    cp_sat = OrToolsBoundedBestResponseProofV1().prove(cp_problem)

    assert reference.policy_proof.selected_action == cp_sat.policy_proof.selected_action
    assert all(item.solver_status == "exact_reference" for item in reference.continuations)
    assert all(item.solver_status == "optimal" for item in cp_sat.continuations)
    assert all(item.parity for item in (*reference.continuations, *cp_sat.continuations))
