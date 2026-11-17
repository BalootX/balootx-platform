"""Deterministic, non-mutating V3-to-V4 Research conversion preview.

The preview deliberately maps only semantics with a direct V4 equivalent.  It
does not publish or persist a recipe, and it does not invent the decision,
outcome, or proof semantics that V3 never carried.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from baloot_platform.core.cards import Card

from .contracts import ResearchRecipeV3
from .recipe_v4 import ResearchRecipeV4
from .scenario_contracts import ScenarioExpressionV1, canonical_sha256


_REQUIRED_OPERATOR_FIELDS = (
    "proof",
    "scenario.decisionPoint",
    "scenario.outcome",
)
_V4_RESOURCE_FIELDS = (
    "seedStart",
    "seedCount",
    "seedsPerShard",
    "solverTimeoutSeconds",
)
_COMPATIBILITY_ONLY_RESOURCE_DIAGNOSTICS = {
    "candidatesPerSeed": "v3_candidates_per_seed_has_no_ortools_v4_semantics",
    "maximumConflicts": "v3_solver_limit_is_replaced_by_frozen_bundle_budget",
    "analysisMode": "legacy_analyzer_identity_is_frozen_bundle_compatibility_only",
}


def _leaf_paths(value: Any, path: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        if not value:
            return [(path, {})]
        rows: list[tuple[str, Any]] = []
        for key in sorted(value, key=str):
            child_path = f"{path}.{key}" if path else str(key)
            rows.extend(_leaf_paths(value[key], child_path))
        return rows
    if isinstance(value, (list, tuple)):
        if not value:
            return [(path, [])]
        rows = []
        for index, child in enumerate(value):
            rows.extend(_leaf_paths(child, f"{path}[{index}]"))
        return rows
    return [(path, copy.deepcopy(value))]


def _path_is_within(path: str, prefix: str) -> bool:
    return (
        path == prefix
        or path.startswith(prefix + ".")
        or path.startswith(prefix + "[")
    )


def _expression(op: str, children: list[dict[str, Any]]) -> dict[str, Any]:
    if not children:
        raise ValueError("conversion expression requires at least one child")
    if len(children) == 1:
        return copy.deepcopy(children[0])
    return ScenarioExpressionV1.build(
        {"op": op, "children": copy.deepcopy(children)}
    ).to_dict()


def _predicate(
    *,
    source_hash: str,
    source_path: str,
    kind: str,
    hardness: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_arguments = copy.deepcopy(dict(arguments))
    identity = canonical_sha256(
        {
            "sourceRecipeHash": source_hash,
            "sourcePath": source_path,
            "kind": kind,
            "hardness": hardness,
            "arguments": canonical_arguments,
        }
    )
    return ScenarioExpressionV1.build(
        {
            "predicateId": f"PRED-CNV-{identity[:16].upper()}",
            "kind": kind,
            "version": 1,
            "hardness": hardness,
            "arguments": canonical_arguments,
        }
    ).to_dict()


def _create_payload_from_built(recipe: ResearchRecipeV4) -> dict[str, Any]:
    value = recipe.to_dict()
    value["configuration"] = {
        "schemaVersion": 1,
        "bundleRevisionId": recipe.configuration["bundleRevisionId"],
    }
    value.pop("capabilityPlan", None)
    value.pop("sha256", None)
    return value


def preview_v3_to_v4_conversion(
    source: Mapping[str, Any],
    *,
    decision_point: Mapping[str, Any] | None = None,
    outcome: Mapping[str, Any] | None = None,
    proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an operator-assisted V4 preview without mutating any input.

    A target identity is exposed only after all three operator-owned semantic
    sections are supplied and the resulting V4 recipe and capability plan are
    valid.  Unsupported V3 leaves remain visible as stable, sorted diagnostics.
    """

    normalized = ResearchRecipeV3.build(copy.deepcopy(dict(source))).to_dict()
    source_hash = str(normalized["sha256"])
    mapped: dict[str, dict[str, Any]] = {}
    unmapped: dict[str, dict[str, Any]] = {}
    covered: set[str] = set()

    def mark_mapped(
        source_path: str,
        target_paths: tuple[str, ...] | list[str],
        mapping_kind: str,
    ) -> None:
        covered.add(source_path)
        mapped[source_path] = {
            "sourcePath": source_path,
            "targetPaths": sorted({str(item) for item in target_paths}),
            "mappingKind": mapping_kind,
        }

    def mark_unmapped(
        source_path: str,
        value: Any,
        reason_code: str,
        *,
        requires_operator_review: bool = True,
    ) -> None:
        covered.add(source_path)
        for leaf_path, leaf_value in _leaf_paths(value, source_path):
            unmapped[leaf_path] = {
                "fieldPath": leaf_path,
                "reasonCode": reason_code,
                "requiresOperatorReview": requires_operator_review,
                "sourceValue": copy.deepcopy(leaf_value),
            }

    mark_mapped("schemaVersion", ("source.schemaVersion",), "source_identity")
    mark_mapped("code", ("source.recipeCode",), "source_identity")
    mark_mapped("sha256", ("source.recipeHash",), "source_identity")
    mark_mapped("titleAr", ("titleAr",), "exact")
    mark_mapped(
        "configuration",
        ("configuration.bundleRevisionId",),
        "frozen_bundle_authority",
    )
    mark_mapped(
        "profiles",
        ("configuration",),
        "derived_from_frozen_bundle_authority",
    )

    output = copy.deepcopy(dict(normalized["output"]))
    mark_mapped("output", ("output",), "exact")

    constraint_groups: list[dict[str, Any]] = []

    def add_predicate_group(
        source_path: str,
        specs: list[tuple[str, str, Mapping[str, Any], str]],
        *,
        group_op: str = "all",
        mapping_kind: str = "semantic_predicate",
    ) -> None:
        children = [
            _predicate(
                source_hash=source_hash,
                source_path=identity_path,
                kind=kind,
                hardness=hardness,
                arguments=arguments,
            )
            for kind, hardness, arguments, identity_path in specs
        ]
        if children:
            constraint_groups.append(_expression(group_op, children))
            mark_mapped(
                source_path,
                tuple(
                    f"scenario.constraints#{child['predicateId']}"
                    for child in children
                ),
                mapping_kind,
            )
        else:
            mark_mapped(source_path, ("scenario.constraints",), "no_constraint")

    situation = normalized["situation"]
    add_predicate_group(
        "situation.dealer",
        [
            (
                "auction.dealer",
                "hard",
                {"seat": situation["dealer"]},
                "situation.dealer",
            )
        ],
    )
    buyer = situation["buyer"]
    add_predicate_group(
        "situation.buyer",
        []
        if buyer is None
        else [
            (
                "auction.buyer",
                "hard",
                {"seat": buyer},
                "situation.buyer",
            )
        ],
    )

    objective_seat = situation["objectiveSeat"]
    if objective_seat is None:
        mark_mapped(
            "situation.objectiveSeat",
            ("scenario.constraints",),
            "no_constraint",
        )
    else:
        mark_unmapped(
            "situation.objectiveSeat",
            objective_seat,
            "v3_construction_objective_has_no_proven_v4_equivalent",
        )

    buyer_relation = situation["constructionBuyerRelation"]
    automatic_is_inert = (
        buyer_relation == "automatic"
        and (buyer is not None or not normalized["tactics"])
    )
    if automatic_is_inert:
        mark_mapped(
            "situation.constructionBuyerRelation",
            ("scenario.constraints",),
            "no_constraint",
        )
    else:
        mark_unmapped(
            "situation.constructionBuyerRelation",
            buyer_relation,
            "v3_buyer_construction_policy_has_no_proven_v4_equivalent",
        )

    purchase_round = situation["purchaseRound"]
    add_predicate_group(
        "situation.purchaseRound",
        []
        if purchase_round is None
        else [
            (
                "auction.purchase_round",
                "hard",
                {"round": purchase_round},
                "situation.purchaseRound",
            )
        ],
    )
    add_predicate_group(
        "situation.purchaseSuits",
        [
            (
                "deal.purchase_suit",
                "hard",
                {"suit": suit},
                f"situation.purchaseSuits[{index}]",
            )
            for index, suit in enumerate(situation["purchaseSuits"])
        ],
        group_op="any",
    )
    trump_suit = situation["trumpSuit"]
    add_predicate_group(
        "situation.trumpSuit",
        []
        if trump_suit is None
        else [
            (
                "auction.trump",
                "hard",
                {"suit": trump_suit},
                "situation.trumpSuit",
            )
        ],
    )
    add_predicate_group(
        "situation.contracts",
        [
            (
                "auction.contract",
                "hard",
                {"contract": contract},
                f"situation.contracts[{index}]",
            )
            for index, contract in enumerate(situation["contracts"])
        ],
        group_op="any",
    )

    match_score = situation["matchScore"]
    add_predicate_group(
        "situation.matchScore",
        [
            (
                "state.match_score",
                "hard",
                {"team": team, "minimum": score, "maximum": score},
                f"situation.matchScore[{index}]",
            )
            for index, (team, score) in enumerate(
                zip(("north_south", "west_east"), match_score, strict=True)
            )
        ],
    )

    for field_name, predicate_kind in (
        ("requiredCards", "deal.card_owner"),
        ("forbiddenCards", "deal.card_forbidden"),
    ):
        cards = situation[field_name]
        specs: list[tuple[str, str, Mapping[str, Any], str]] = []
        for seat in sorted(cards):
            for index, card in enumerate(cards[seat]):
                specs.append(
                    (
                        predicate_kind,
                        "hard",
                        {
                            "card": Card.parse(card).id,
                            "seat": seat,
                            "handStage": "post_purchase",
                        },
                        f"situation.{field_name}.{seat}[{index}]",
                    )
                )
        add_predicate_group(f"situation.{field_name}", specs)

    suit_specs: list[tuple[str, str, Mapping[str, Any], str]] = []
    for seat in sorted(situation["finalSuitCountRanges"]):
        ranges = situation["finalSuitCountRanges"][seat]
        for suit in sorted(ranges):
            bounds = ranges[suit]
            suit_specs.append(
                (
                    "deal.suit_count",
                    "hard",
                    {
                        "seat": seat,
                        "suit": suit,
                        "minimum": bounds["minimum"],
                        "maximum": bounds["maximum"],
                        "handStage": "post_purchase",
                    },
                    f"situation.finalSuitCountRanges.{seat}.{suit}",
                )
            )
    add_predicate_group("situation.finalSuitCountRanges", suit_specs)

    projects = situation["projects"]
    if projects:
        mark_unmapped(
            "situation.projects",
            projects,
            "v3_project_high_rank_equivalence_is_unproven",
        )
    else:
        mark_mapped(
            "situation.projects", ("scenario.constraints",), "no_constraint"
        )

    add_predicate_group(
        "situation.escalations",
        [
            (
                "escalation.level",
                "hard",
                {"level": level},
                f"situation.escalations[{index}]",
            )
            for index, level in enumerate(situation["escalations"])
        ],
    )

    for index, tactic in enumerate(normalized["tactics"]):
        tactic_path = f"tactics[{index}]"
        hardness = "hard" if tactic["mode"] == "qualification_gate" else "prefer"
        perspectives = tactic["perspectives"]
        tactic_specs = []
        if perspectives:
            for perspective_index, perspective in enumerate(perspectives):
                tactic_specs.append(
                    (
                        "tactic.verified",
                        hardness,
                        {
                            "kind": tactic["kind"],
                            "actingSeat": perspective,
                            "minimumScore": tactic["minimumScore"],
                            "minimumEvidence": 1,
                        },
                        f"{tactic_path}.perspectives[{perspective_index}]",
                    )
                )
        else:
            tactic_specs.append(
                (
                    "tactic.verified",
                    hardness,
                    {
                        "kind": tactic["kind"],
                        "minimumScore": tactic["minimumScore"],
                        "minimumEvidence": 1,
                    },
                    tactic_path,
                )
            )
        children = [
            _predicate(
                source_hash=source_hash,
                source_path=identity_path,
                kind=kind,
                hardness=item_hardness,
                arguments=arguments,
            )
            for kind, item_hardness, arguments, identity_path in tactic_specs
        ]
        constraint_groups.append(_expression("any", children))
        predicate_targets = tuple(
            f"scenario.constraints#{child['predicateId']}" for child in children
        )
        for field_name in ("kind", "mode", "minimumScore", "perspectives"):
            mark_mapped(
                f"{tactic_path}.{field_name}",
                predicate_targets,
                "semantic_predicate",
            )
        mark_unmapped(
            f"{tactic_path}.weight",
            tactic["weight"],
            "v4_tactic_weight_has_no_equivalent",
        )
    if not normalized["tactics"]:
        mark_mapped("tactics", ("scenario.constraints",), "no_constraint")

    constraints = _expression("all", constraint_groups)
    mark_mapped("qualification", ("qualification",), "exact")

    resources: dict[str, Any] = {}
    if normalized["resources"].get("generationLane") == "ortools":
        mark_mapped(
            "resources.generationLane",
            ("resources.generationLane",),
            "exact",
        )
    else:
        mark_unmapped(
            "resources.generationLane",
            normalized["resources"].get("generationLane"),
            "new_v4_authoring_requires_ortools_generation",
            requires_operator_review=False,
        )
    for field_name in _V4_RESOURCE_FIELDS:
        if field_name in normalized["resources"]:
            resources[field_name] = copy.deepcopy(
                normalized["resources"][field_name]
            )
            mark_mapped(
                f"resources.{field_name}",
                (f"resources.{field_name}",),
                "exact",
            )
    for field_name, reason_code in _COMPATIBILITY_ONLY_RESOURCE_DIAGNOSTICS.items():
        if field_name in normalized["resources"]:
            mark_unmapped(
                f"resources.{field_name}",
                normalized["resources"][field_name],
                reason_code,
                requires_operator_review=False,
            )
    resources["generationLane"] = "ortools"
    resources["generationPolicy"] = "ortools_cp_sat_v3"
    resources["proofBudgetProfile"] = (
        str(proof.get("budgetProfile"))
        if isinstance(proof, Mapping) and proof.get("budgetProfile")
        else "late_game_exact_v1"
    )

    diversity = {
        key: copy.deepcopy(normalized["diversity"][key])
        for key in (
            "schemaVersion",
            "cellDimensions",
            "topKPerCell",
            "rejectedAuditModulus",
        )
    }
    for key in diversity:
        mark_mapped(f"diversity.{key}", (f"diversity.{key}",), "exact")
    if "sha256" in normalized["diversity"]:
        mark_mapped(
            "diversity.sha256",
            ("sha256",),
            "derived_identity_recomputed",
        )
    mark_mapped(
        "renderPolicy",
        ("renderPolicy.mode",),
        "exact_with_v4_video_count_default",
    )
    if normalized["status"] == "draft":
        mark_mapped("status", ("status",), "exact")
    else:
        mark_unmapped(
            "status",
            normalized["status"],
            "target_lifecycle_resets_to_draft",
        )

    for leaf_path, leaf_value in _leaf_paths(normalized):
        if any(_path_is_within(leaf_path, prefix) for prefix in covered):
            continue
        unmapped[leaf_path] = {
            "fieldPath": leaf_path,
            "reasonCode": "no_proven_v4_equivalent",
            "requiresOperatorReview": True,
            "sourceValue": copy.deepcopy(leaf_value),
        }

    scenario: dict[str, Any] = {
        "schemaVersion": 1,
        "origin": {"kind": "search"},
        "constraints": constraints,
    }
    if decision_point is not None:
        scenario["decisionPoint"] = copy.deepcopy(dict(decision_point))
    if outcome is not None:
        scenario["outcome"] = copy.deepcopy(dict(outcome))

    proposed: dict[str, Any] = {
        "schemaVersion": 4,
        "titleAr": normalized["titleAr"],
        "output": output,
        "configuration": {
            "schemaVersion": 1,
            "bundleRevisionId": normalized["configuration"]["bundleRevisionId"],
        },
        "scenario": scenario,
        "qualification": copy.deepcopy(normalized["qualification"]),
        "resources": resources,
        "diversity": diversity,
        "renderPolicy": {
            **copy.deepcopy(normalized["renderPolicy"]),
            "videoCount": min(1, int(output["resultCount"])),
        },
        "status": "draft",
    }
    if proof is not None:
        proposed["proof"] = copy.deepcopy(dict(proof))

    supplied_operator_fields = {
        "proof": proof is not None,
        "scenario.decisionPoint": decision_point is not None,
        "scenario.outcome": outcome is not None,
    }
    missing_fields = [
        field_path
        for field_path in _REQUIRED_OPERATOR_FIELDS
        if not supplied_operator_fields[field_path]
    ]
    diagnostics = [
        {
            "code": "operator_input_required",
            "fieldPath": field_path,
            "message": f"operator must define {field_path} before a V4 identity exists",
        }
        for field_path in missing_fields
    ]

    target_code: str | None = None
    target_hash: str | None = None
    buildable = False
    if not missing_fields:
        internal_candidate = copy.deepcopy(proposed)
        internal_candidate["configuration"] = copy.deepcopy(
            normalized["configuration"]
        )
        try:
            built = ResearchRecipeV4.build(internal_candidate)
        except (TypeError, ValueError) as exc:
            diagnostics.append(
                {
                    "code": "target_recipe_invalid",
                    "fieldPath": "",
                    "message": str(exc),
                }
            )
        else:
            if built.capability_plan.valid:
                proposed = _create_payload_from_built(built)
                target_code = built.code
                target_hash = built.sha256
                buildable = True
            else:
                for item in built.capability_plan.diagnostics:
                    value = item.to_dict()
                    diagnostics.append(
                        {
                            "code": "target_capability_plan_invalid",
                            "fieldPath": ",".join(value.get("fieldPaths") or ()),
                            "message": str(value.get("code") or "invalid capability plan"),
                            "details": value,
                        }
                    )

    diagnostics.sort(
        key=lambda item: (
            str(item.get("fieldPath") or ""),
            str(item.get("code") or ""),
            canonical_sha256(item),
        )
    )
    mapped_rows = sorted(
        mapped.values(), key=lambda item: (item["sourcePath"], item["mappingKind"])
    )
    unmapped_rows = sorted(
        unmapped.values(),
        key=lambda item: (
            item["fieldPath"],
            item["reasonCode"],
            canonical_sha256(item["sourceValue"]),
        ),
    )
    core = {
        "schemaVersion": 1,
        "source": {
            "schemaVersion": 3,
            "recipeCode": normalized["code"],
            "recipeHash": source_hash,
        },
        "proposedRecipe": proposed,
        "targetRecipeCode": target_code,
        "targetRecipeHash": target_hash,
        "mappedFields": mapped_rows,
        "unmappedFields": unmapped_rows,
        "requiredOperatorFields": missing_fields,
        "buildDiagnostics": diagnostics,
        "exactlyRepresentable": not unmapped_rows,
        "buildable": buildable,
    }
    return {**core, "sha256": canonical_sha256(core)}


__all__ = ["preview_v3_to_v4_conversion"]
