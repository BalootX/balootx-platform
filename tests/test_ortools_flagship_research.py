from __future__ import annotations

import inspect

import pytest

from baloot_platform.core.enums import Rank, Seat
from baloot_platform.research.ortools_flagship_research import (
    OrToolsFlagshipResearchV1,
)


def test_original_flagship_has_executable_structural_contradiction(rules):
    research = OrToolsFlagshipResearchV1(rules=rules)

    receipt = research.contradiction_receipt()

    assert receipt.status == "exact_structural_contradiction"
    assert receipt.theorem_id == "FLAGSHIP-TWO-ACES-TRICK7-LEAD-V1"
    assert receipt.checked_acting_seats == tuple(Seat)
    assert receipt.solver_status == "infeasible"
    assert len(receipt.model_hash) == 64
    assert receipt.variable_count > 0
    assert receipt.constraint_count > 0
    assert set(receipt.assumption_core) == {
        "contract_sun",
        "opponent_wins_trick_six",
        "two_opposing_aces_in_trick_eight",
        "target_forces_trick_seven",
    }
    assert receipt == research.contradiction_receipt()
    assert [item["diagnosticId"] for item in receipt.diagnostics] == [
        "ALT-TARGET-WINS-TRICK-6",
        "ALT-ONE-OPPONENT-ACE",
        "ALT-POLICY-CONDITIONED-NON-ACE-LEAD",
    ]
    assert all(item["autoApplied"] is False for item in receipt.diagnostics)


def test_original_flagship_produces_exact_zero_winning_root_receipt(rules):
    result = OrToolsFlagshipResearchV1(rules=rules).search_one(
        tail_seconds=10,
        lift_seconds=20,
        proof_seconds=10,
    )

    assert result.status == "proof_failed"
    assert result.qualified is False
    assert result.tail is not None
    assert result.tail.root_action.rank == Rank.ACE
    assert result.lift is not None
    assert result.lift.status == "lift_target_miss"
    assert len(result.lift.deck_order) == 32
    assert len(set(result.lift.deck_order)) == 32
    assert len(result.lift.plays) == 32
    assert result.lift.plays[-12:] == tuple(
        type(result.lift.plays[0])(
            play.trick_index,
            play.ordinal,
            play.actor,
            play.card,
        )
        for play in result.tail.plays
    )
    assert any(
        project.awarded
        and project.team == result.tail.buyer.team
        and project.value == 20
        for project in result.lift.projects.projects
    )
    assert result.proof is not None
    assert result.proof.completion_status == "proof_failed"
    assert result.proof.claim_satisfied is False
    assert result.proof.winning_root_action_count == 0
    assert result.tail_no_good == result.tail.exact_no_good
    assert result.tail_no_good_hash is not None
    assert set(result.stage_timings) == {
        "tailConstructionSeconds",
        "prefixLiftSeconds",
        "exactProofSeconds",
        "totalSeconds",
    }
    assert all(value >= 0 for value in result.stage_timings.values())
    artifact = result.to_dict()
    assert artifact["proof"]["completionStatus"] == "proof_failed"
    assert artifact["proof"]["winningRootActionCount"] == 0
    assert artifact["contradiction"]["sha256"] == result.contradiction.sha256
    assert artifact["researchHash"] == result.research_hash


def test_contradictory_flagship_refuses_variant_brute_force(rules):
    research = OrToolsFlagshipResearchV1(rules=rules)

    with pytest.raises(ValueError, match="prevents qualified variants"):
        research.search_variants(count=5)


def test_flagship_research_reuses_shared_project_and_play_fragments():
    source = inspect.getsource(OrToolsFlagshipResearchV1)

    assert "attach_project_lifecycle_fragment_v1" in source
    assert "OrToolsPlayTailStageV1" in source
    assert "_add_prefix_fragment" not in source
    for forbidden in ("BalootEngine", "HeuristicBot", "proof_service"):
        assert forbidden not in source
