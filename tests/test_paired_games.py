from __future__ import annotations

import copy

import pytest
import yaml

from baloot_platform.benchmark.paired_games import (
    run_paired_duplicate_rounds,
)
from baloot_platform.config import validate_profile_payload


def _semantic(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.pop("evidenceSha256")
    return result


def test_paired_duplicate_rounds_are_deterministic_and_swapped(root):
    first = run_paired_duplicate_rounds(
        root=root,
        candidate_id="strong",
        baseline_id="heuristic",
        pair_count=2,
        seed_start=31_000_000,
        source_revision="a" * 40,
        source_tree="b" * 40,
    )
    second = run_paired_duplicate_rounds(
        root=root,
        candidate_id="strong",
        baseline_id="heuristic",
        pair_count=2,
        seed_start=31_000_000,
        source_revision="a" * 40,
        source_tree="b" * 40,
    )

    assert _semantic(first) == _semantic(second)
    assert first["evidenceSha256"] == second["evidenceSha256"]
    assert first["pairCount"] == 2
    assert first["requestedPairCount"] == 2
    assert first["completionStatus"] == "completed"
    assert first["automaticPromotion"] is False
    for pair in first["pairs"]:
        assert [
            game["candidateTeam"] for game in pair["games"]
        ] == ["north_south", "west_east"]
        assert (
            pair["games"][0]["initialStateSha256"]
            == pair["games"][1]["initialStateSha256"]
        )
        for game in pair["games"]:
            assert len(game["actions"]) == 32
            assert all(
                action["selectedAction"] in action["legalActions"]
                for action in game["actions"]
            )


def test_paired_duplicate_rounds_do_not_claim_comparison_too_early(root):
    value = run_paired_duplicate_rounds(
        root=root,
        candidate_id="strong",
        baseline_id="heuristic",
        pair_count=1,
        seed_start=31_000_100,
        source_revision="a" * 40,
        source_tree="b" * 40,
        candidate_expected_regret=0.0,
        baseline_expected_regret=1.0,
    )

    assert value["comparison"] == {
        "status": "insufficient_pairs",
        "minimumPairs": 100,
        "observedPairs": 1,
    }


def test_paired_duplicate_rounds_cancel_without_partial_pair(root):
    value = run_paired_duplicate_rounds(
        root=root,
        candidate_id="solver_fast",
        baseline_id="strong",
        pair_count=100,
        seed_start=31_000_200,
        source_revision="a" * 40,
        source_tree="b" * 40,
        candidate_expected_regret=0.0,
        baseline_expected_regret=1.0,
        cancel_check=lambda: True,
    )

    assert value["completionStatus"] == "safety_cancelled"
    assert value["requestedPairCount"] == 100
    assert value["pairCount"] == 0
    assert value["pairs"] == []
    assert value["comparison"] == {
        "status": "incomplete_run",
        "completionStatus": "safety_cancelled",
        "minimumPairs": 100,
        "observedPairs": 0,
        "requestedPairs": 100,
    }


def test_paired_benchmark_rejects_unsafe_deadline(root):
    profile = yaml.safe_load(
        (
            root / "configs/benchmark/strength.yaml"
        ).read_text(encoding="utf-8")
    )
    profile["paired_comparison"]["maximum_seconds"] = 0

    with pytest.raises(ValueError, match="maximum seconds"):
        validate_profile_payload("benchmark", profile, root=root)
