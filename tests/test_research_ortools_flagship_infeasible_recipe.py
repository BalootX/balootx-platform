from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from baloot_platform.research.ortools_flagship_research import (
    OrToolsFlagshipResearchV1,
)
from baloot_platform.research.recipe_v4 import ResearchRecipeV4
from baloot_platform.research.scenario_predicate_manifest_v2 import (
    shared_scenario_predicate_catalog_lowering_manifest_v2,
)
from baloot_platform.research.tail_first_stage_planner import (
    TailFirstStagePlannerV1,
)
from baloot_platform.service.container import build_container
from scripts.benchmark_ortools_research_v4 import (
    _publish_explicit_bundle,
)


EXAMPLE = Path("examples/research-scenario-v4-flagship-exact-infeasible.json")


def _ui_authored_exact_request(raw: dict) -> dict:
    value = copy.deepcopy(raw)
    value["output"] = {"kind": "laffa_collection", "resultCount": 5}
    value["renderPolicy"] = {"mode": "on_demand", "videoCount": 5}
    value["resources"].update(
        {
            "seedCount": 5,
            "seedsPerShard": 1,
            "solverTimeoutSeconds": 60.0,
        }
    )
    value["diversity"]["topKPerCell"] = 5
    value["proof"]["strategyClass"] = "pure_deterministic"
    constraints = value["scenario"]["constraints"]["children"]
    opponent_winner = next(
        item
        for item in constraints
        if item["predicateId"] == "PRED-FLAGSHIP-OPPONENT-WINS-SIX"
    )
    opponent_winner["arguments"].pop("team")
    opponent_winner["arguments"]["seat"] = "north"
    for item in constraints:
        selector = item["arguments"].get("selector")
        if selector is not None:
            selector["ekka"] = "any"
    value["scenario"]["outcome"] = {
        "op": "all",
        "children": [value["scenario"]["outcome"]],
    }
    return value


def test_canonical_flagship_request_is_exact_tail_first_without_false_v3_refusal(
    root: Path,
    tmp_path: Path,
) -> None:
    raw = json.loads((root / EXAMPLE).read_text(encoding="utf-8"))
    container = build_container(root=root, writable_root=tmp_path / "runtime")
    try:
        authority, context = _publish_explicit_bundle(
            container,
            root,
            {
                "bundle": {
                    "id": "baloot_ortools_research_native_v3",
                    "source": "configs/bundle_intelligent_research.yaml",
                    "backend": "ortools_cp_sat_v3",
                }
            },
        )
        # Keep the checked-in example immutable; this test exercises the
        # freshly published bundle revision, which has a different hash.
        assert raw["configuration"]["schemaVersion"] == 1
        raw["configuration"]["bundleRevisionId"] = authority.bundle_revision_id
        assert authority.bundle_id == "baloot_ortools_research_native_v3"
        assert (
            context.bundle.component("challenge")["scenarioResearchV4"]["backend"]
            == "ortools_cp_sat_v3"
        )

        recipe, selected_context = container.research._build_recipe_and_context(raw)
        assert isinstance(recipe, ResearchRecipeV4)
        assert selected_context is not None
        assert selected_context.bundle_revision_id == authority.bundle_revision_id
        rebuilt = ResearchRecipeV4.build(recipe.to_dict())
        assert rebuilt.sha256 == recipe.sha256
        assert rebuilt.to_dict() == recipe.to_dict()

        validation = container.research.validate_recipe(recipe.to_dict())
        assert validation["valid"] is True, validation["issues"]
        assert not any(
            item["code"] == "exact_structural_infeasibility"
            for item in validation["issues"]
        )

        estimate = container.research.estimate(recipe.to_dict())
        assert estimate["valid"] is True
        assert estimate["launchAdmissible"] is True
        assert estimate["refusalReason"] is None
        assert estimate["exhaustiveSearchInfeasibilityEstablishable"] is False

        unknown = copy.deepcopy(recipe.to_dict())
        unknown["hiddenTailPins"] = []
        with pytest.raises(ValueError, match="unknown fields"):
            ResearchRecipeV4.build(unknown)

        assert recipe.output == {"kind": "laffa", "resultCount": 1}
        assert recipe.render_policy == {"mode": "disabled", "videoCount": 0}
        assert recipe.proof.mode == "exact"
        assert recipe.proof.visibility == "complete_state"
        assert recipe.proof.claim == "winning_root_action_count"

        plan = TailFirstStagePlannerV1().plan(recipe.scenario)
        assert plan.planner == "tail_first"
        assert plan.prefix_length == 23
        assert plan.tail_offset == 20
        assert plan.tail_length == 12

        manifest = shared_scenario_predicate_catalog_lowering_manifest_v2()
        for predicate in recipe.scenario.predicates():
            assert predicate.hardness == "hard"
            definition = manifest.definition(predicate.kind)
            assert definition is not None
            assert "hard" in definition.allowed_hardness
            assert definition.cp_sat_support.search_status in {"exact", "derived"}
            assert definition.cp_sat_support.exact_proof != "unsupported"
            assert definition.publication_issues() == ()

        predicates = {
            predicate.predicate_id: predicate
            for predicate in recipe.scenario.predicates()
        }
        assert (
            predicates["PRED-FLAGSHIP-OPPONENT-WINS-SIX"].arguments["team"]
            == "north_south"
        )
        for predicate_id in (
            "PRED-FLAGSHIP-TARGET-WINS-SEVEN",
            "PRED-FLAGSHIP-TARGET-WINS-EIGHT",
        ):
            assert predicates[predicate_id].arguments["team"] == "west_east"

        scenario_json = json.dumps(raw["scenario"], sort_keys=True)
        forbidden_pins = (
            '"exactTracePrefix"',
            '"deckOrder"',
            '"auctionActions"',
            '"escalationActions"',
            '"projectDeclarations"',
            '"playActions"',
            '"selectedAction"',
            '"cards"',
            '"suits"',
        )
        for forbidden in forbidden_pins:
            assert forbidden not in scenario_json

        contradiction = OrToolsFlagshipResearchV1(
            rules=context.bundle.rules
        ).contradiction_receipt()
        assert contradiction.status == "exact_structural_contradiction"
        assert contradiction.theorem_id == "FLAGSHIP-TWO-ACES-TRICK7-LEAD-V1"
        assert all(row["autoApplied"] is False for row in contradiction.diagnostics)

        ui_raw = _ui_authored_exact_request(raw)
        ui_recipe, ui_context = container.research._build_recipe_and_context(ui_raw)
        assert isinstance(ui_recipe, ResearchRecipeV4)
        assert ui_context is not None
        ui_validation = container.research.validate_recipe(ui_recipe.to_dict())
        assert ui_validation["valid"] is True, ui_validation["issues"]
        ui_estimate = container.research.estimate(ui_recipe.to_dict())
        assert ui_estimate["launchAdmissible"] is True
        assert ui_estimate["refusalReason"] is None
        assert ui_estimate["exhaustiveSearchInfeasibilityEstablishable"] is False

    finally:
        container.shutdown()
