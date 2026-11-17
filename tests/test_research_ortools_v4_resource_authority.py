from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from baloot_platform.research.recipe_v4 import ResearchRecipeV4
from baloot_platform.research.scenario_contracts import canonical_sha256
from baloot_platform.service import api
from baloot_platform.service.container import build_container
from scripts.benchmark_ortools_research_v4 import _publish_explicit_bundle


_FALSE_AUTHORING_FIELDS = {
    "candidatesPerSeed",
    "maximumConflicts",
    "analysisMode",
}


def _canonical_recipe() -> dict:
    path = (
        Path(__file__).parents[1]
        / "examples"
        / "research-scenario-v4-flagship-exact-infeasible.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def canonical_built_recipe(tmp_path_factory) -> ResearchRecipeV4:
    root = Path(__file__).parents[1]
    container = build_container(
        root=root,
        writable_root=tmp_path_factory.mktemp("v4-resource-authority"),
    )
    authority, _context = _publish_explicit_bundle(
        container,
        root,
        {
            "bundle": {
                "id": "baloot_ortools_research_native_v3",
                "source": "configs/bundle_ortools_research_native_v3_1.yaml",
                "backend": "ortools_cp_sat_v3",
            }
        },
    )
    payload = _canonical_recipe()
    payload["configuration"]["bundleRevisionId"] = authority.bundle_revision_id
    payload["resources"]["generationPolicy"] = "ortools_cp_sat_v3"
    built, _context = container.research._build_recipe_and_context(payload)
    assert isinstance(built, ResearchRecipeV4)
    return built


def test_new_v4_resources_are_closed_to_frozen_ortools_authoring(
    canonical_built_recipe: ResearchRecipeV4,
) -> None:
    built = canonical_built_recipe

    assert built.resources == {
        "generationLane": "ortools",
        "generationPolicy": "ortools_cp_sat_v3",
        "seedStart": 1,
        "seedCount": 1,
        "seedsPerShard": 1,
        "solverTimeoutSeconds": 15.0,
        "proofBudgetProfile": "late_game_exact_v1",
    }
    assert not _FALSE_AUTHORING_FIELDS.intersection(built.resources)

    for field, value in (
        ("candidatesPerSeed", 1),
        ("maximumConflicts", 1),
        ("analysisMode", "cascaded"),
    ):
        payload = built.to_dict()
        payload.pop("sha256")
        payload["resources"][field] = value
        with pytest.raises(ValueError, match="unknown fields"):
            ResearchRecipeV4.build(payload)

    for field, value in (
        ("generationLane", "organic_runtime"),
        ("generationPolicy", "natural_policy"),
        ("generationPolicy", "scenario_guided_actor_safe"),
    ):
        payload = built.to_dict()
        payload.pop("sha256")
        payload["resources"][field] = value
        with pytest.raises(ValueError, match="requires generation"):
            ResearchRecipeV4.build(payload)


def test_v4_request_and_openapi_expose_only_ortools_resource_authority() -> None:
    default_resources = api.ResearchResourcesV4Request.model_validate({})
    assert default_resources.model_dump() == {
        "generationLane": "ortools",
        "generationPolicy": "ortools_cp_sat_v3",
        "seedStart": 1,
        "seedCount": 4,
        "seedsPerShard": 4,
        "shardWorkerCount": None,
        "solverTimeoutSeconds": 15.0,
        "proofBudgetProfile": "late_game_exact_v1",
    }

    for payload in (
        {"candidatesPerSeed": 1},
        {"maximumConflicts": 1},
        {"analysisMode": "cascaded"},
        {"generationLane": "organic_runtime"},
        {"generationPolicy": "natural_policy"},
        {"generationPolicy": "scenario_guided_actor_safe"},
    ):
        with pytest.raises(ValidationError):
            api.ResearchResourcesV4Request.model_validate(payload)

    schema = api.app.openapi()["components"]["schemas"][
        "ResearchResourcesV4Request"
    ]
    assert schema["additionalProperties"] is False
    assert not _FALSE_AUTHORING_FIELDS.intersection(schema["properties"])
    assert schema["properties"]["generationLane"] == {
        "type": "string",
        "const": "ortools",
        "default": "ortools",
        "title": "Generationlane",
    }
    assert schema["properties"]["generationPolicy"] == {
        "type": "string",
        "const": "ortools_cp_sat_v3",
        "default": "ortools_cp_sat_v3",
        "title": "Generationpolicy",
    }


def test_historical_sha_bound_v4_resources_round_trip_without_reauthoring(
    canonical_built_recipe: ResearchRecipeV4,
) -> None:
    historical = canonical_built_recipe.to_dict()
    historical["resources"].update(
        {
            "generationPolicy": "natural_policy",
            "candidatesPerSeed": 4,
            "maximumConflicts": 250_000,
            "analysisMode": "cascaded",
        }
    )
    historical["sha256"] = canonical_sha256(
        {key: value for key, value in historical.items() if key != "sha256"}
    )

    rebuilt = ResearchRecipeV4.build(copy.deepcopy(historical))

    assert rebuilt.to_dict() == historical
    assert _FALSE_AUTHORING_FIELDS.issubset(rebuilt.resources)
