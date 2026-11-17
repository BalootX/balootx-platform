from __future__ import annotations

import pytest

from baloot_platform.research.guided_construction_pilot import (
    GuidedConstructionPilotProfileV1,
    _comparison_decision,
    _profile_definitions,
    build_guided_pilot_recipe_v1,
)


class _RecipeService:
    def capabilities(self):
        return {
            "sha256": "1" * 64,
            "bundleHash": "2" * 64,
            "rulesHash": "3" * 64,
            "conventionHash": "4" * 64,
            "research": {
                "defaultAgentRelease": "agent@1",
                "analysisProfiles": {"screening": "5" * 64},
                "defaultScoreProfileHash": "6" * 64,
            },
        }


def test_pilot_recipes_separate_hard_structure_and_tactical_proof_gates():
    service = _RecipeService()
    profile = GuidedConstructionPilotProfileV1(
        attempts_per_lane_profile=10,
        seeds_per_shard=5,
    )
    recipes = {
        definition["id"]: build_guided_pilot_recipe_v1(
            service,
            definition,
            profile,
        )
        for definition in _profile_definitions()
    }

    project = recipes["project_400"]
    assert project["situation"]["projects"] == [
        {
            "seat": "south",
            "contract": "sun",
            "kind": "four_hundred",
            "value": 400,
            "highRank": "ace",
            "mode": "hard",
        }
    ]
    assert project["tactics"] == []
    assert project["resources"]["deepVerificationPolicy"] == "off"
    assert project["resources"]["playConstructionMode"] == (
        "actor_safe_agent_verified"
    )

    kabout = recipes["kabout_opportunity"]
    assert kabout["tactics"][0]["kind"] == "kabout_creation"
    assert kabout["tactics"][0]["mode"] == "qualification_gate"
    assert kabout["resources"]["deepVerificationPolicy"] == "required_gates"

    small = recipes["small_card_delayed_gain_precursor"]
    assert small["tactics"][0]["kind"] == "small_card_big_gain"
    assert small["situation"]["contracts"] == ["hokum"]
    assert small["situation"]["trumpSuit"] == "hearts"
    assert all(
        recipe["qualification"]["requireObserverSafety"]
        and recipe["qualification"]["requireAllDecisionAudit"]
        and recipe["qualification"]["minimumSurroundingQuality"] == 70
        and recipe["qualification"]["maximumRegret"] == 0
        for recipe in recipes.values()
    )


def _lane(*, distinct, cpu, defects=0, status="completed"):
    return {
        "status": status,
        "distinctQualifiedFinalists": distinct,
        "distinctQualifiedPerCpuHour": distinct * 3600 / cpu if cpu else None,
        "cpuSecondsPerDistinctQualified": cpu / distinct if distinct else None,
        "semanticDefectCount": defects,
    }


def test_pilot_retains_only_end_to_end_improvement_without_semantic_defects():
    profile = GuidedConstructionPilotProfileV1(
        attempts_per_lane_profile=10,
        seeds_per_shard=5,
    )
    passed = _comparison_decision(
        _lane(distinct=4, cpu=4),
        _lane(distinct=1, cpu=4),
        profile,
    )
    assert passed["retainGuidedObjectiveOrConstraint"] is True
    assert passed["yieldMultiplier"] == 4

    defect = _comparison_decision(
        _lane(distinct=4, cpu=4, defects=1),
        _lane(distinct=1, cpu=4),
        profile,
    )
    assert defect["retainGuidedObjectiveOrConstraint"] is False
    assert "semantic_defect" in defect["reasons"]

    no_evidence = _comparison_decision(
        _lane(distinct=0, cpu=4),
        _lane(distinct=0, cpu=4),
        profile,
    )
    assert no_evidence["retainGuidedObjectiveOrConstraint"] is False
    assert "no_distinct_ortools_finalist" in no_evidence["reasons"]


@pytest.mark.parametrize(
    "overrides",
    (
        {"attempts_per_lane_profile": 0},
        {"attempts_per_lane_profile": 101},
        {"seeds_per_shard": 21},
        {"solver_timeout_seconds": 0},
        {"minimum_cost_reduction": 1},
    ),
)
def test_pilot_profile_rejects_invalid_budgets(overrides):
    with pytest.raises(ValueError):
        GuidedConstructionPilotProfileV1(**overrides)
