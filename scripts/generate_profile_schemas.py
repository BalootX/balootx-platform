#!/usr/bin/env python3
"""Generate closed JSON Schemas from the authoritative profile examples.

The authored YAML remains the single source of truth. Every object is closed so
unknown or misspelled keys fail publication. Adding a supported capability
requires changing the profile and regenerating the schema in the same commit.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = yaml.safe_load((ROOT / "configs/bundle.yaml").read_text(encoding="utf-8"))
OUT = ROOT / "schemas/profiles"


def merge_schema_views(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Expose the union of published variants while their branches stay closed."""

    if left == right:
        return deepcopy(left)
    if left.get("type") == right.get("type") == "object":
        left_properties = dict(left.get("properties") or {})
        right_properties = dict(right.get("properties") or {})
        properties = {
            key: (
                merge_schema_views(left_properties[key], right_properties[key])
                if key in left_properties and key in right_properties
                else deepcopy(
                    left_properties[key]
                    if key in left_properties
                    else right_properties[key]
                )
            )
            for key in sorted(set(left_properties) | set(right_properties))
        }
        merged = {key: deepcopy(value) for key, value in left.items()
                  if key not in {"properties", "required", "oneOf"} and right.get(key) == value}
        merged.update({
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": sorted(
                set(left.get("required") or ())
                & set(right.get("required") or ())
            ),
        })
        left_branches, right_branches = left.get("oneOf"), right.get("oneOf")
        if isinstance(left_branches, list) and isinstance(right_branches, list) and len(left_branches) == len(right_branches):
            def discriminators(branch):
                return {key: field["const"] for key, field in branch.get("properties", {}).items()
                        if "const" in field}
            if all(discriminators(a) and discriminators(a) == discriminators(b)
                   for a, b in zip(left_branches, right_branches)):
                merged["oneOf"] = [merge_schema_views(a, b) for a, b in zip(left_branches, right_branches)]
        return merged
    return {**({"type": left["type"]} if left.get("type") is not None and left.get("type") == right.get("type") else {}),
            "anyOf": [deepcopy(left), deepcopy(right)]}


def authored_profile_paths() -> dict[str, list[str]]:
    paths: dict[str, list[str]] = {}
    for bundle_path in sorted((ROOT / "configs").glob("bundle*.yaml")):
        bundle = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
        for family, relative in dict(bundle["components"]).items():
            variants = paths.setdefault(str(family), [])
            if str(relative) not in variants:
                variants.append(str(relative))
    return paths


def scalar_schema(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if value is None:
        return {"type": ["null", "string", "number", "integer", "boolean", "object", "array"]}
    return {"type": "string"}


def infer(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        properties = {str(key): infer(item) for key, item in value.items()}
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": sorted(properties),
        }
    if isinstance(value, list):
        if not value:
            return {"type": "array"}
        schemas = [infer(item) for item in value]
        unique: list[dict[str, Any]] = []
        for schema in schemas:
            if schema not in unique:
                unique.append(schema)
        item_schema = unique[0] if len(unique) == 1 else {"anyOf": unique}
        return {"type": "array", "items": item_schema}
    return scalar_schema(value)


def apply_release_constraints(
    family: str,
    schema: dict[str, Any],
    authored: dict[str, Any],
) -> None:
    """Apply invariant release gates that type inference cannot express."""

    properties = schema.get("properties", {})
    if family == "agents":
        legacy = yaml.safe_load(
            (ROOT / "configs/agents/adaptive_lineup_v1.yaml").read_text(encoding="utf-8")
        )
        identity_fields = ("id", "version")
        schema["if"] = {
            "properties": {
                "profile": {
                    "properties": {
                        name: {"const": legacy["profile"][name]}
                        for name in identity_fields
                    },
                    "required": list(identity_fields),
                }
            },
            "required": ["profile"],
        }
        for branch, profiles in (
            ("then", infer(legacy["profiles"])),
            ("else", properties["profiles"]),
        ):
            rows = profiles["properties"]
            first = next(iter(rows.values()))
            if all(row == first for row in rows.values()):
                definition = "legacyAgentRow" if branch == "then" else "agentRow"
                schema.setdefault("$defs", {})[definition] = first
                profiles["properties"] = {
                    name: {"$ref": f"#/$defs/{definition}"} for name in rows
                }
            schema[branch] = {"properties": {"profiles": profiles}}
        properties["profiles"] = {"type": "object"}
    if family == "challenge":
        # Reuse generic profile metadata without duplicating complete variants.
        properties["profile"]["properties"]["description"] = {"type": "string"}
        properties["profile"]["required"] = [
            key for key in properties["profile"]["required"] if key != "description"
        ]
    if family == "challenge" and "tactical_detectors" in properties:
        tactical = properties["tactical_detectors"]
        native_payoff = {
            "type": "object", "additionalProperties": False,
            "x-messageArKey": "research_native_delayed_payoff_label",
            "x-helpArKey": "research_native_delayed_payoff_scope",
            "required": ["gainMetric", "minimumGain"],
            "properties": {
                "gainMetric": {
                    "type": "string", "enum": ["raw_card_points", "awarded_point_margin"],
                    "x-messageArKey": "research_native_delayed_payoff_metric_label",
                    "x-helpArKey": "research_native_delayed_payoff_metric_help",
                    "x-enumMessageArKeys": {
                        "raw_card_points": "research_native_delayed_payoff_raw",
                        "awarded_point_margin": "research_native_delayed_payoff_awarded",
                    },
                },
                "minimumGain": {
                    "type": "number", "minimum": 0, "maximum": 100,
                    "x-messageArKey": "research_native_delayed_payoff_minimum_label",
                    "x-helpArKey": "research_native_delayed_payoff_minimum_help",
                },
            },
        }
        tactical["properties"]["nativeDelayedPayoff"] = native_payoff
        tactical["required"] = [key for key in tactical["required"] if key != "nativeDelayedPayoff"]
        schema.setdefault("allOf", []).append({
            "if": {"required": ["tactical_detectors"], "properties": {
                "tactical_detectors": {"required": ["nativeDelayedPayoff"]},
            }},
            "then": {"required": ["scenarioResearchV4"], "properties": {
                "scenarioResearchV4": {"required": ["backend"], "properties": {
                    "backend": {"const": "ortools_cp_sat_v3"},
                }},
            }},
        })
    if family == "challenge" and "scenarioResearchV4" in properties:
        from baloot_platform.research.scenario_contracts import NativeObservedProofLimitsV1
        native_schema = NativeObservedProofLimitsV1.json_schema()
        profiles = properties["scenarioResearchV4"].get("properties", {}).get(
            "proofBudgets", {}).get("properties", {}).get("profiles", {}).get("properties", {})
        for budget in profiles.values():
            if "nativeObserved" in budget.get("properties", {}):
                budget["properties"]["nativeObserved"] = deepcopy(native_schema)
                if authored["scenarioResearchV4"]["backend"] == "ortools_cp_sat_v3":
                    budget["properties"]["policySearch"] = {"type": "string", "enum": ["time_budgeted_symbolic_v1"]}
        authoring = properties["scenarioResearchV4"].get("properties", {}).get("authoringDefaults")
        if authoring is not None:
            defaults = authoring["properties"]["playingQuality"]
            if "exactBudgetProfileByVisibility" in defaults["required"]:
                defaults["required"].remove("exactBudgetProfileByVisibility")
        quality = properties["scenarioResearchV4"].get("properties", {}).get("qualityExecution")
        if quality is not None:
            quality["properties"]["auditConstructedTrace"] = {"type": "boolean"}
            quality["properties"]["playIndexWorldCaps"] = {
                "type": "object", "additionalProperties": False, "required": ["schemaVersion", "bands"],
                "properties": {"schemaVersion": {"type": "integer", "const": 1},
                    "bands": {"type": "array", "minItems": 1, "maxItems": 32,
                        "items": {"type": "object", "additionalProperties": False,
                            "required": ["firstPlayIndex", "lastPlayIndex", "maximumHiddenWorlds"],
                            "properties": {
                                "firstPlayIndex": {"type": "integer", "minimum": 0, "maximum": 31},
                                "lastPlayIndex": {"type": "integer", "minimum": 0, "maximum": 31},
                                "maximumHiddenWorlds": {"type": "integer", "minimum": 1, "maximum": 2**31 - 1}}}}}}
            if "playIndexWorldCaps" in quality.get("required", []):
                quality["required"].remove("playIndexWorldCaps")
            if "auditConstructedTrace" in quality.get("required", []):
                quality["required"].remove("auditConstructedTrace")
        if quality is not None:
            from baloot_platform.research.ortools_play_tail import NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS
            from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1
            from baloot_platform.research.scenario_contracts import scenario_video_interestingness_schema_v1
            quality["properties"]["completeStatePlayPolicyPair"] = {"type": "boolean",
                "default": ScenarioQualityExecutionBudgetV1.__dataclass_fields__["complete_state_play_policy_pair"].default}
            quality["properties"]["exactTailProof"] = {"type": "boolean",
                "default": ScenarioQualityExecutionBudgetV1.__dataclass_fields__["exact_tail_proof"].default}
            quality["properties"]["exactTailBudgetProfile"] = {"type": "string", "enum": sorted(profiles)}
            if "exactTailBudgetProfile" in quality.get("required", []):
                quality["required"].remove("exactTailBudgetProfile")
            quality.setdefault("allOf", []).append({
                "if": {"required": ["exactTailBudgetProfile"]},
                "then": {"required": ["exactTailProof"], "properties": {"exactTailProof": {"const": True}}},
            })
            quality.setdefault("allOf", []).append({
                "if": {"required": ["exactTailProof"], "properties": {"exactTailProof": {"const": True}}},
                "then": {"required": ["exactPlayFromTrick", "earlyPlayPolicyId"],
                         "properties": {"exactPlayFromTrick": {"type": "integer", "minimum": 2}}},
            })
            quality["properties"]["videoInterestingness"] = scenario_video_interestingness_schema_v1()
            quality["properties"]["candidateSource"] = {"type": "string", "enum": ["runtime_proposal_v1", "runtime_proposal_stream_v1"]}
            from baloot_platform.research.scenario_contracts import family_opening_mode_schema_v1, family_proposal_mode_schema_v1
            quality['properties']['familyOpeningMode'] = family_opening_mode_schema_v1()
            quality['properties']['familyProposalMode'] = family_proposal_mode_schema_v1()
            quality.setdefault('allOf', []).append({
                'if': {'required': ['familyProposalMode']},
                'then': {'required': ['candidateSource', 'familyTarget'],
                         'properties': {'candidateSource': {'const': 'runtime_proposal_stream_v1'},
                                        'familyTarget': {'properties': {'bindings': {
                                            'properties': {'contract': {'const': 'sun'}}}}}},
                         'not': {'anyOf': [{'required': ['familyOpeningMode']}, {'required': ['familyCatalog']}]}},
            })
            quality.setdefault('allOf', []).append({
                'if': {'required': ['familyOpeningMode']},
                'then': {'required': ['candidateSource'],
                         'properties': {'candidateSource': {'const': 'runtime_proposal_stream_v1'}},
                         'anyOf': [{'required': ['familyTarget']}, {'required': ['familyCatalog']}]},
            })
            from baloot_platform.research.family_catalog import family_catalog_schema_v1, family_target_schema_v1
            quality["properties"]["familyTarget"] = family_target_schema_v1()
            quality["properties"]["familyCatalog"] = family_catalog_schema_v1()
            if "familyCatalog" in quality.get("required", []):
                quality["required"].remove("familyCatalog")
            quality.setdefault("allOf", []).append({
                "if": {"required": ["familyCatalog"]},
                "then": {"required": ["candidateSource"],
                         "properties": {"candidateSource": {"const": "runtime_proposal_stream_v1"}},
                         "not": {"anyOf": [{"required": ["familyTarget"]},
                                             {"required": ["acceptedLessonLibrary"]}]}},
            })
            quality.setdefault("allOf", []).append({
                "if": {"required": ["familyTarget"]},
                "then": {"required": ["candidateSource"],
                         "properties": {"candidateSource": {"const": "runtime_proposal_stream_v1"}},
                         "not": {"required": ["acceptedLessonLibrary"]}},
            })
            quality["properties"]["proposalSeedCount"] = {"type": "integer", "minimum": 1, "maximum": 1_000_000}
            quality["properties"]["proposalMaximumSeconds"] = {"type": "number", "exclusiveMinimum": 0}
            from baloot_platform.research.lesson_prefilter import LESSON_FAMILIES
            quality["properties"]["proposalPlayIndices"] = {"type": "array", "minItems": 1,
                "maxItems": 32, "uniqueItems": True, "items": {"type": "integer", "minimum": 0, "maximum": 31}}
            quality["properties"]["acceptedLessonLibrary"] = {"type": "object", "additionalProperties": False,
                "properties": {family: {"oneOf": [
                    {"type": "integer", "minimum": 0},
                    {"type": "array", "uniqueItems": True,
                     "items": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
                ]} for family in sorted(LESSON_FAMILIES)}}
            quality.setdefault("dependentRequired", {})["proposalPlayIndices"] = ["candidateSource"]
            quality.setdefault("dependentRequired", {})["acceptedLessonLibrary"] = ["candidateSource"]
            quality.setdefault("dependentRequired", {})["proposalMaximumSeconds"] = ["candidateSource"]
            quality.setdefault("dependentRequired", {})["proposalSeedCount"] = ["candidateSource"]
            quality.setdefault("allOf", []).append({
                "if": {"required": ["candidateSource"]},
                "then": {"required": ["earlyPlayPolicyId", "auditConstructedTrace", "videoInterestingness", "proposalSeedCount"],
                         "properties": {"auditConstructedTrace": {"const": True},
                                        "videoInterestingness": {"properties": {"schemaVersion": {"const": 4}}}}},
            })
            if "videoInterestingness" in quality.get("required", []):
                quality["required"].remove("videoInterestingness")
            quality["properties"]["bidOpponentPlayPolicyId"] = {
                "type": "string", "enum": sorted(NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS)}
            quality["properties"]["observedBidStrategyPlayFallbackId"] = {
                "type": "string", "enum": sorted(NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS)}
            quality["properties"]["bidAlternativeProof"] = {
                "type": "object", "additionalProperties": False,
                "required": ["schemaVersion", "upperBound", "maximumWorlds", "maximumSeconds",
                    "maximumQueries", "maximumWorldSeconds", "maximumWorldQueries"],
                "properties": {"schemaVersion": {"type": "integer", "const": 1},
                    "upperBound": {"type": "integer", "minimum": -(2**31), "maximum": 2**31 - 1},
                    **{key: {"type": "integer", "minimum": 1, "maximum": 2**31 - 1}
                       for key in ("maximumWorlds", "maximumQueries", "maximumWorldQueries")},
                    **{key: {"type": "number", "exclusiveMinimum": 0,
                        "exclusiveMaximum": authored["scenarioResearchV4"]["qualityExecution"]["maximumWallSeconds"]}
                       for key in ("maximumSeconds", "maximumWorldSeconds")}}}
            from baloot_platform.research.scenario_contracts import NATIVE_BID_WORLD_RANKING_IDS
            quality["properties"]["bidAlternativeProof"]["properties"]["worldRankingId"] = {
                "type": "string", "enum": list(NATIVE_BID_WORLD_RANKING_IDS)}
            world_alternative = deepcopy(quality["properties"]["bidAlternativeProof"])
            information_alternative = {"type": "object", "additionalProperties": False,
                "required": ["schemaVersion", "method", "upperBound", "maximumSeconds", "maximumQueries"],
                "properties": {key: deepcopy(world_alternative["properties"][key]) for key in
                    ("schemaVersion", "upperBound", "maximumSeconds", "maximumQueries")}}
            information_alternative["properties"]["schemaVersion"]["const"] = 2
            from baloot_platform.research.scenario_contracts import NATIVE_BID_INFORMATION_SET_REFUTATION_METHOD
            information_alternative["properties"]["method"] = {
                "type": "string", "enum": [NATIVE_BID_INFORMATION_SET_REFUTATION_METHOD]}
            world_alternative["not"] = {"required": ["method"]}
            information_alternative["not"] = {"anyOf": [{"required": [key]}
                for key in world_alternative["properties"] if key not in information_alternative["properties"]]}
            combined = merge_schema_views(world_alternative, information_alternative)
            combined["oneOf"] = [world_alternative, information_alternative]
            quality["properties"]["bidAlternativeProof"] = combined
            from baloot_platform.research.scenario_contracts import NATIVE_BID_REFUTATION_AUCTION_ACTIONS
            quality["properties"]["bidThresholdRefutation"] = {
                "type": "object", "additionalProperties": False,
                "required": ["schemaVersion", "maximumSeconds", "maximumQueries", "worldRankingId", "opponentAuctionAction"],
                "properties": {"schemaVersion": {"type": "integer", "const": 1},
                    "maximumSeconds": {"type": "number", "exclusiveMinimum": 0,
                        "exclusiveMaximum": authored["scenarioResearchV4"]["qualityExecution"]["maximumWallSeconds"]},
                    "maximumQueries": {"type": "integer", "minimum": 1, "maximum": 2**31 - 1},
                    "worldRankingId": {"type": "string", "enum": list(NATIVE_BID_WORLD_RANKING_IDS)},
                    "opponentAuctionAction": {"type": "string", "enum": list(NATIVE_BID_REFUTATION_AUCTION_ACTIONS)}}}
            if "bidThresholdRefutation" in quality["required"]:
                quality["required"].remove("bidThresholdRefutation")
            if "bidAlternativeProof" in quality["required"]:
                quality["required"].remove("bidAlternativeProof")
            quality["properties"]["earlyPlayPolicyId"] = {
                "type": "string", "enum": sorted(NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS)}
            if "bidOpponentPlayPolicyId" in quality["required"]:
                quality["required"].remove("bidOpponentPlayPolicyId")
            if "observedBidStrategyPlayFallbackId" in quality["required"]:
                quality["required"].remove("observedBidStrategyPlayFallbackId")
        if quality is not None and "escalationOpportunityOrder" in quality["properties"]:
            # Existing frozen quality budgets predate these optional controls.
            quality["required"].remove("escalationOpportunityOrder")
            order = authored["scenarioResearchV4"]["qualityExecution"]["escalationOpportunityOrder"]
            quality["properties"]["escalationOpportunityOrder"].update(
                minItems=len(order), maxItems=len(order), uniqueItems=True,
                items={"type": "string", "enum": sorted(order)})
            for optional in ("earlyPlayPolicyId", "exactPlayFromTrick", "featuredPlayNativeProof", "bidIncumbentProof",
                             "allowExactInputQuality", "completeStatePlayPolicyPair", "pruneOwnProjectPassWorlds"):
                if optional in quality["required"]:
                    quality["required"].remove(optional)
        if quality is not None and "bidIncumbentProof" in quality["properties"]:
            incumbent = quality["properties"]["bidIncumbentProof"]["properties"]
            incumbent["schemaVersion"]["const"] = 1
            incumbent["maximumSeconds"].update(exclusiveMinimum=0,
                exclusiveMaximum=authored["scenarioResearchV4"]["qualityExecution"]["maximumWallSeconds"])
            incumbent["maximumQueries"].update(minimum=1, maximum=2**31 - 1)
            if "playPolicies" not in incumbent:
                from baloot_platform.research.scenario_backend import materialize_default_policy_profiles_v1
                incumbent["playPolicies"] = infer(materialize_default_policy_profiles_v1(
                    authored["scenarioResearchV4"]["policyCatalog"]))
            for policy in incumbent["playPolicies"]["properties"].values():
                policy["properties"]["choices"]["maxItems"] = 0
            legacy_incumbent = deepcopy(quality["properties"]["bidIncumbentProof"])
            legacy_incumbent["properties"].pop("nativePlayPolicyId", None)
            legacy_incumbent["properties"].pop("observedProjectPolicyId", None)
            legacy_incumbent["properties"].pop("observedAuctionPolicyId", None)
            legacy_incumbent["properties"].pop("completeStateAlternativeFirst", None)
            legacy_incumbent["required"] = ["schemaVersion", "maximumSeconds", "maximumQueries", "playPolicies"]
            native_incumbent = {
                "type": "object", "additionalProperties": False,
                "properties": {key: deepcopy(incumbent[key]) for key in
                    ("schemaVersion", "maximumSeconds", "maximumQueries")},
                "required": ["schemaVersion", "maximumSeconds", "maximumQueries", "nativePlayPolicyId"],
            }
            native_incumbent["properties"]["schemaVersion"]["const"] = 2
            native_incumbent["properties"]["nativePlayPolicyId"] = {
                "type": "string", "enum": sorted(NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS)}
            from baloot_platform.research.ortools_project_fragment import NATIVE_OBSERVED_PROJECT_POLICY_IDS
            native_incumbent["properties"]["observedProjectPolicyId"] = {
                "type": "string", "enum": list(NATIVE_OBSERVED_PROJECT_POLICY_IDS)}
            from baloot_platform.research.ortools_scenario_model import NATIVE_OBSERVED_AUCTION_POLICY_IDS
            native_incumbent["properties"]["observedAuctionPolicyId"] = {
                "type": "string", "enum": list(NATIVE_OBSERVED_AUCTION_POLICY_IDS)}
            from baloot_platform.research.scenario_contracts import native_bid_nonplay_policy_schema_v1
            native_incumbent["properties"]["completeStateNonplayPolicy"] = native_bid_nonplay_policy_schema_v1()
            native_incumbent["properties"]["completeStateAlternativeFirst"] = {"type": "boolean", "default": True}
            combined = merge_schema_views(legacy_incumbent, native_incumbent)
            combined["oneOf"] = [legacy_incumbent, native_incumbent]
            quality["properties"]["bidIncumbentProof"] = combined
    if "schema_version" in properties:
        version = authored.get("schema_version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError(
                f"{family} schema_version must be an authored integer invariant"
            )
        properties["schema_version"] = {
            "const": version,
            "type": "integer",
        }

    if family == "rules":
        properties["scoring"]["properties"]["contract"]["properties"]["buyer_tie_outcome"] = {"type":"string","title":"تعادل نقاط الصن","description":"إذا تعادلت نقاط الفريقين في الصن: هل ينجح المشتري أو يخسر؟ ترك الإعداد بدون قيمة يحافظ على قاعدة النسخة المحفوظة؛ الاختيار الصريح ينشئ قاعدة جديدة.","default":"buyer_makes","anyOf":[{"const":"buyer_makes","title":"المشتري ينجح عند التعادل"},{"const":"buyer_loses","title":"المشتري يخسر عند التعادل"}]}
        contract = properties["scoring"]["properties"]["contract"]
        contract["required"] = [key for key in contract["required"] if key != "buyer_tie_outcome"]
        projects = properties["projects"]
        projects["properties"]["tie_break_order"] = {
            "type": "string", "enum": ["declaration", "second_trick_reveal"],
        }
        projects["required"] = [key for key in projects["required"] if key != "tie_break_order"]
        projects["properties"]["reveal_scope"] = {
            "type": "string", "enum": ["scoring_team", "comparison_winner"],
        }
        projects["required"] = [key for key in projects["required"] if key != "reveal_scope"]
        projects["properties"]["top_rank_visibility"] = {
            "type": "string", "enum": ["declared", "comparison_only"],
        }
        projects["required"] = [key for key in projects["required"] if key != "top_rank_visibility"]
        # Opt-in correction; omitted in immutable historical profiles.
        kabout = properties["scoring"]["properties"]["kabout"]
        kabout["properties"]["winner"] = {"type": "string", "const": "trick_sweeper"}
        kabout["required"] = [key for key in kabout["required"] if key != "winner"]
        declaration_mode = (
            schema["properties"]["projects"]["properties"].get(
                "declaration_mode"
            )
        )
        if declaration_mode is not None:
            declaration_mode.update(
                {"enum": ["mandatory", "optional"], "type": "string"}
            )
        escalation = schema["properties"]["escalation"]["properties"]
        escalation["opening_actor_policy"] = {
            "const": "opponent_team_any_seat",
            "type": "string",
        }
        escalation["closure_policy"] = {
            "const": "advance_to_projects_without_announcement",
            "type": "string",
        }
        heuristic = (
            schema["properties"]["bidding"]["properties"]["heuristics"]["properties"]
        )
        for name in ("sun_rank_weights", "hokum_trump_rank_weights"):
            heuristic[name].update({"minItems": 8, "maxItems": 8})
            heuristic[name]["items"] = {"type": "number", "minimum": 0}
        for name in (
            "sun_extra_suit_card_weight",
            "hokum_extra_trump_card_weight",
            "hokum_off_trump_ace_ten_weight",
        ):
            heuristic[name]["minimum"] = 0
        for name in (
            "sun_extra_suit_card_after",
            "hokum_extra_trump_card_after",
        ):
            heuristic[name]["minimum"] = 0
    elif family == "conventions":
        for field in schema["properties"]["forcedness"]["properties"].values():
            field.update({"minimum": 0.0, "maximum": 1.0})
        for field in schema["properties"]["relationship_reliability"][
            "properties"
        ].values():
            field.update({"minimum": 0.0, "maximum": 1.0})
        for signal in schema["properties"]["signals"]["properties"].values():
            ratios = signal["properties"]["likelihood_ratios"]["properties"]
            for field in ratios.values():
                field.update({"exclusiveMinimum": 0.0, "maximum": 4.0})
    elif family == "product":
        schema["properties"]["schema_version"] = {"const": 1, "type": "integer"}
        public_guide = schema["properties"]["public_guide"]["properties"]
        public_guide["schema_version"] = {"const": 1, "type": "integer"}
        public_guide["locale"] = {"const": "ar-SA", "type": "string"}
        public_guide["direction"] = {"const": "rtl", "type": "string"}
        foundry_copy = schema["properties"]["foundry_copy"]["properties"]
        foundry_copy["locale"] = {"const": "ar-SA", "type": "string"}
        foundry_copy["direction"] = {"const": "rtl", "type": "string"}
    elif family == "video":
        clip = schema["properties"]["exports"]["properties"]["decision_clip"]
        clip["properties"]["fps"] = {"type": "integer", "enum": [30, 60]}
        if "fps" in clip["required"]:
            clip["required"].remove("fps")
        clip["properties"]["outcome_seconds"] = {"type": "number", "minimum": 0}
        if "outcome_seconds" in clip["required"]:
            clip["required"].remove("outcome_seconds")
        # Complete-replay producers add these bounded timing controls in memory
        # without changing the authored default profile.
        clip["properties"]["setup_seconds"] = {"type": "number", "exclusiveMinimum": 0}
        clip["properties"]["play_seconds"] = {"type": "number", "exclusiveMinimum": 0}
        clip["properties"]["continue_to_round_end"] = {"type": "boolean"}
        localizations = schema["properties"]["localizations"]
        scope_keys = sorted(key for key in localizations["properties"]
                            if key.startswith("video.analysis.proof_scope.")
                            and not key.endswith("_tail"))
        for key in scope_keys:
            localizations["properties"][key].update(minLength=1, pattern=r"\S")
            localizations["required"].remove(key)
        if scope_keys:
            localizations["dependentRequired"] = {
                key: [other for other in scope_keys if other != key] for key in scope_keys}
        tail_keys = [key for key in localizations["properties"]
                     if key.endswith("_tail") and key.startswith("video.analysis.proof_scope.")]
        tail_keys += [key for key in localizations["properties"]
                      if key == "video.analysis.worst_case_awarded_margin"]
        for key in tail_keys:
            localizations["properties"][key].update(minLength=1, pattern=r"\S")
            localizations["required"].remove(key)
        if tail_keys:
            localizations.setdefault("dependentRequired", {}).update({
                key: [other for other in tail_keys if other != key] for key in tail_keys})
        # Frozen profiles predate explicit tie-rule copy.
        for key in tuple(localizations["required"]):
            if key.startswith("video.scoring.sun_tie."):
                localizations["required"].remove(key)
        render = schema["properties"]["render"]["properties"]
        render["width"] = {"const": 1080, "type": "integer"}
        render["height"] = {"const": 1920, "type": "integer"}
        render["locale"] = {"const": "ar-SA", "type": "string"}
        render["direction"] = {"const": "rtl", "type": "string"}
        schema["properties"]["motion"]["properties"]["mode"] = {
            "enum": ["normal", "reduced"],
            "type": "string",
        }
        presentation_schema = schema["properties"]["presentation"]
        presentation = presentation_schema["properties"]
        presentation["mode"] = {
            "enum": ["player", "expert"],
            "type": "string",
        }
        presentation_schema["allOf"] = [
            {
                "if": {
                    "properties": {"mode": {"const": "player"}},
                    "required": ["mode"],
                },
                "then": {
                    "properties": {
                        "show_identity_pill": {"const": True},
                        "show_state_locator": {"const": False},
                        "show_developer_card_codes": {"const": False},
                    }
                },
            }
        ]
    elif family == "analysis":
        analysis = schema["properties"]
        analysis["requested_samples"].update(
            {"minimum": 1, "maximum": 100_000}
        )
        analysis["exact_allocation_limit"].update(
            {"minimum": 1, "maximum": 100_000}
        )
        analysis["exact_search_max_remaining"].update(
            {"minimum": 0, "maximum": 32}
        )
        analysis["maximum_sampling_attempts"].update(
            {"minimum": 1, "maximum": 100_000}
        )
        analysis["objective"] = {
            "enum": [
                "contract_success",
                "cvar",
                "low_volatility",
                "match_point_margin",
                "match_win_probability",
                "round_margin",
            ],
            "type": "string",
        }
        for key in ("id", "version"):
            analysis["profile"]["properties"][key]["minLength"] = 1
    elif family == "belief":
        particles = schema["properties"]["particles"]["properties"]
        for key in ("requested", "maximum"):
            particles[key].update({"minimum": 1, "maximum": 100_000})
        particles["minimum_effective_fraction"].update(
            {"exclusiveMinimum": 0.0, "maximum": 1.0}
        )
        particles["resampling"] = {"const": "systematic", "type": "string"}
        schema["properties"]["construction"]["properties"][
            "exact_when_allocations_below"
        ].update({"minimum": 1, "maximum": 100_000})
        for key in ("id", "version"):
            schema["properties"]["profile"]["properties"][key][
                "minLength"
            ] = 1
    elif family == "challenge":
        promotion = schema["properties"]["publication"]["properties"]
        for role_field in ("reviewer_roles", "publisher_roles"):
            promotion[role_field] = {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            }

        challenge = schema["properties"]

        def versioned_identity(
            block: dict[str, Any], *, schema_version: int = 1
        ) -> None:
            properties = block["properties"]
            properties["schemaVersion"] = {
                "const": schema_version,
                "type": "integer",
            }
            properties["id"].update({"minLength": 1})
            properties["revision"].update({"minLength": 1})

        screening = challenge["screening"]["properties"]
        screening["minimum_legal_actions"].update({"minimum": 2})
        screening["shallow_samples"].update(
            {"minimum": 2, "maximum": 100_000}
        )
        screening["retain_fraction"].update(
            {"exclusiveMinimum": 0.0, "maximum": 1.0}
        )
        screening["maximum_candidates_per_seed"].update({"minimum": 1})
        screening["shallow_exact_allocation_limit"].update(
            {"minimum": 1, "maximum": 100_000}
        )
        screening["shallow_exact_search_remaining"].update(
            {"minimum": 0, "maximum": 32}
        )
        screening["maximum_sampling_attempts"].update(
            {"minimum": 2, "maximum": 100_000}
        )
        screening["exact_search_max_nodes"].update({"minimum": 1})
        screening["exact_search_max_seconds"].update(
            {"exclusiveMinimum": 0.0}
        )

        confirmation = challenge["confirmation"]["properties"]
        for key in ("bid_samples", "deep_samples"):
            confirmation[key].update({"minimum": 2, "maximum": 100_000})
        confirmation["maximum_sampling_attempts"].update(
            {"minimum": 2, "maximum": 100_000}
        )
        confirmation["exact_allocation_limit"].update(
            {"minimum": 1, "maximum": 100_000}
        )
        confirmation["exact_search_remaining"].update(
            {"minimum": 0, "maximum": 32}
        )
        confirmation["exact_search_max_nodes"].update({"minimum": 1})
        confirmation["exact_search_max_seconds"].update(
            {"exclusiveMinimum": 0.0}
        )
        confirmation["exact_search_max_depth"].update({"minimum": 1})
        confirmation["exact_search_max_cache_entries"].update({"minimum": 0})
        confirmation["minimum_confirmation_confidence"].update(
            {"minimum": 0.0, "maximum": 1.0}
        )
        confirmation["objective"] = {
            "enum": [
                "contract_success",
                "cvar",
                "low_volatility",
                "match_point_margin",
                "match_win_probability",
                "round_margin",
            ],
            "type": "string",
        }
        confirmation["require_legal_replay"] = {
            "const": True,
            "type": "boolean",
        }

        stability = challenge["stability"]["properties"]
        stability["reanalysis_samples"].update(
            {"minimum": 2, "maximum": 100_000}
        )
        stability["maximum_sampling_attempts"].update(
            {"minimum": 2, "maximum": 100_000}
        )
        stability["exact_allocation_limit"].update(
            {"minimum": 1, "maximum": 100_000}
        )
        stability["exact_search_remaining"].update(
            {"minimum": 0, "maximum": 32}
        )
        stability["exact_search_max_nodes"].update({"minimum": 1})
        stability["exact_search_max_seconds"].update(
            {"exclusiveMinimum": 0.0}
        )
        stability["minimum_hidden_deals"].update({"minimum": 2})

        executor = challenge["decision_analysis_executor"]
        versioned_identity(executor)
        for name in ("screening", "confirmation", "finalistReanalysis"):
            stage = executor["properties"][name]["properties"]
            stage["workerCount"].update({"minimum": 1, "maximum": 4})
            stage["maximumPendingPerWorker"].update(
                {"minimum": 1, "maximum": 2}
            )
            stage["maximumSeconds"].update(
                {"minimum": 0.05, "maximum": 3_600.0}
            )
            stage["pollSeconds"].update(
                {"minimum": 0.01, "maximum": 1.0}
            )

        deep = challenge["deep_analysis"]
        versioned_identity(deep)
        deep_properties = deep["properties"]
        deep_properties["maximumOracleDecisionsPerTrace"].update(
            {"minimum": 0, "maximum": 128}
        )
        deep_properties["maximumOracleRemainingCards"].update(
            {"minimum": 1, "maximum": 32}
        )
        deep_properties["maximumConfirmationDecisionsPerTrace"].update(
            {"minimum": 1, "maximum": 32}
        )
        deep_properties["minimumScreeningScore"].update(
            {"minimum": 0.0, "maximum": 100.0}
        )
        deep_properties["jobPriority"].update(
            {"minimum": 0, "maximum": 10_000}
        )

        selection = challenge["selection_policy"]
        versioned_identity(selection)
        selection_properties = selection["properties"]
        selection_properties["allowedTruthLabels"] = {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {
                "enum": [
                    "debatable",
                    "exact_optimal",
                    "robust_best_known",
                ],
                "type": "string",
            },
        }
        selection_properties["minimumConfidence"].update(
            {"minimum": 0.0, "maximum": 1.0}
        )
        selection_properties["minimumScore"].update(
            {"minimum": 0.0, "maximum": 100.0}
        )
        selection_properties["minimumSurroundingQuality"].update(
            {"minimum": 0.0, "maximum": 100.0}
        )
        selection_properties["minimumSurroundingConfidence"].update(
            {"minimum": 0.0, "maximum": 1.0}
        )
        selection_properties["minimumSurroundingMateriality"].update(
            {"minimum": 0.0, "maximum": 100.0}
        )
        for key in (
            "maximumSurroundingMaterialMistakes",
            "maximumSurroundingUnverifiedDecisions",
            "maximumSurroundingCatastrophicCrossings",
        ):
            selection_properties[key].update({"minimum": 0})
        for weight in selection_properties["weights"]["properties"].values():
            weight.update({"exclusiveMinimum": 0.0})

        thrill = challenge["thrill_profile"]
        versioned_identity(thrill)
        thrill_properties = thrill["properties"]
        thrill_properties["minimumCriticalPoints"].update({"minimum": 1})
        thrill_properties["minimumScore"].update(
            {"minimum": 0.0, "maximum": 100.0}
        )
        for weight in thrill_properties["weights"]["properties"].values():
            weight.update({"exclusiveMinimum": 0.0})

        comparison = challenge["research_comparison"]
        versioned_identity(comparison, schema_version=2)
        comparison_properties = comparison["properties"]
        comparison_properties["minimumQualifiedPerLane"].update(
            {"minimum": 1, "maximum": 1_000_000}
        )
        comparison_properties["minimumDistinctSelectedFinalistsPerLane"].update(
            {"minimum": 1, "maximum": 1_000_000}
        )
        comparison_properties["minimumAttemptedSeedsPerLane"].update(
            {"minimum": 1, "maximum": 10_000_000}
        )
        comparison_properties["minimumYieldMultiplier"].update(
            {"minimum": 1.0, "maximum": 100.0}
        )
        comparison_properties["minimumCostReduction"].update(
            {"minimum": 0.0, "exclusiveMaximum": 1.0}
        )
        comparison_properties["requireZeroSemanticDefects"] = {
            "const": True,
            "type": "boolean",
        }
        comparison_properties["requiredPromotionCpuBoundary"] = {
            "const": "isolated_process_cpu_v1",
            "type": "string",
        }

        campaign_screening = challenge["campaign_screening"]
        versioned_identity(campaign_screening)
        campaign_properties = campaign_screening["properties"]
        campaign_properties["minimumLegalActions"].update({"minimum": 2})
        for key in (
            "baseRetainBasisPoints",
            "maximumRetainBasisPoints",
            "eliteRetainBasisPoints",
        ):
            campaign_properties[key].update({"minimum": 0, "maximum": 10_000})
        campaign_properties["eliteScore"].update(
            {"minimum": 0.0, "maximum": 100.0}
        )
        campaign_properties["rejectedAuditModulus"].update(
            {"minimum": 1, "maximum": 1_000_000}
        )
        campaign_properties["compressionLevel"].update(
            {"minimum": 1, "maximum": 9}
        )
    elif family == "hitl":
        review_schema = schema["properties"]["review_clip"]
        review = review_schema["properties"]
        review["width"] = {"enum": [720, 1080], "type": "integer"}
        review["height"] = {"enum": [1280, 1920], "type": "integer"}
        review["presentation_mode"] = {
            "enum": ["player", "expert"],
            "type": "string",
        }
        review_schema["allOf"] = [
            {
                "if": {
                    "properties": {"width": {"const": 720}},
                    "required": ["width"],
                },
                "then": {"properties": {"height": {"const": 1280}}},
            },
            {
                "if": {
                    "properties": {"width": {"const": 1080}},
                    "required": ["width"],
                },
                "then": {"properties": {"height": {"const": 1920}}},
            },
            {
                "if": {
                    "properties": {
                        "presentation_mode": {"const": "player"}
                    },
                    "required": ["presentation_mode"],
                },
                "then": {
                    "properties": {
                        "show_developer_card_codes": {"const": False}
                    }
                },
            }
        ]
        schema.setdefault("allOf", []).append(
            {
                "if": {
                    "properties": {
                        "profile": {
                            "properties": {
                                "version": {"pattern": "^21\\."}
                            },
                            "required": ["version"],
                        }
                    },
                    "required": ["profile"],
                },
                "then": {
                    "properties": {
                        "review_clip": {
                            "properties": {
                                "width": {"const": 1080},
                                "height": {"const": 1920},
                            }
                        }
                    }
                },
            }
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    profile_paths = authored_profile_paths()
    for family, relative in sorted(BUNDLE["components"].items()):
        variants: list[dict[str, Any]] = []
        for variant_relative in profile_paths.get(family, [str(relative)]):
            path = ROOT / "configs" / variant_relative
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            variant = infer(data)
            apply_release_constraints(family, variant, data)
            if family == "product" and variant_relative == "product/studio_ortools.yaml":
                # Frozen published revisions predate this optional authoring control.
                fields = variant["properties"]["foundry_copy"]["properties"]["groups"]["properties"]["configuration_fields"]
                fields["required"] = [key for key in fields["required"] if key not in {
                    "label_bid_alternative_first", "label_exact_tail_proof", "label_video_interestingness",
                    "label_maximum_final_margin", "label_minimum_point_swing",
                    "label_maximum_featured_trick", "label_minimum_meaningful_decisions",
                }]
                workflow = variant["properties"]["foundry_copy"]["properties"]["groups"]["properties"]["workflow"]
                workflow["required"] = [
                    key for key in workflow["required"]
                    if not key.startswith(("research_lesson_tag", "research_auction_seeds_", "research_native_delayed_payoff_")) and key not in {
                        "research_quality_exact_tail_boundary",
                        "research_quality_exact_tail_boundary_en",
                        "research_quality_video_interest",
                        "research_quality_video_interest_en",
                        "research_scenario_estimate_full_round_unknown",
                        "research_scenario_estimate_full_round_unknown_en",
                        "research_shard_worker_count",
                        "research_shard_worker_count_en",
                        "research_scenario_help_shard_worker_count",
                        "research_scenario_help_shard_worker_count_en",
                        "research_scenario_diagnostic_schema_invalid",
                        "research_scenario_diagnostic_rules_inapplicable",
                        "research_scenario_diagnostic_unsupported_predicate",
                        "research_scenario_diagnostic_contradictory_constraints",
                        "research_scenario_diagnostic_structurally_infeasible",
                        "research_scenario_diagnostic_not_found_within_budget",
                        "research_scenario_diagnostic_construction_target_miss",
                        "research_scenario_diagnostic_semantic_postcondition_miss",
                        "research_scenario_diagnostic_proof_incomplete",
                        "research_scenario_diagnostic_proof_failed",
                        "research_scenario_diagnostic_satisfied",
                        "research_scenario_timing_tail_model_build",
                        "research_scenario_timing_tail_model_build_en",
                        "research_scenario_timing_tail_model_configure",
                        "research_scenario_timing_tail_model_configure_en",
                        "research_scenario_timing_tail_solver",
                        "research_scenario_timing_tail_solver_en",
                        "research_decision_proof_boundaries",
                        "research_decision_proof_bounded",
                        "research_decision_proof_exact",
                        "research_decision_proof_mode",
                        "research_decision_proof_scope",
                        "research_scope_action_optimality_only",
                        "research_scope_bounded",
                        "research_scope_exact",
                        "research_scope_exact_fixed_deal_terminal_tail",
                        "research_scope_observer_safe_policy_conditioned",
                        "research_scope_selected_bid_current_deal",
                        "research_scope_selected_escalation_current_deal",
                        "research_scope_selected_final_project_complete_state",
                        "research_scope_selected_play_fixed_state_terminal_continuation",
                        "research_scope_supplied_worlds_only",
                    }
                ]
            if family == "video":
                # Historical video revisions use the generic proof copy instead.
                localizations = variant["properties"]["localizations"]
                if "video.analysis.unique_awarded_margin" in localizations["required"]:
                    localizations["required"].remove("video.analysis.unique_awarded_margin")
            if variant not in variants:
                variants.append(variant)
        schema = deepcopy(variants[0])
        for variant in variants[1:]:
            schema = merge_schema_views(schema, variant)
        if len(variants) > 1:
            # Challenge revisions share closed shapes but may have overlapping budget ranges.
            schema["anyOf" if family == "challenge" else "oneOf"] = variants
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = f"https://baloot.local/schemas/profiles/{family}.schema.json"
        target = OUT / f"{family}.schema.json"
        with TemporaryDirectory(dir=OUT) as staging:
            staged = Path(staging) / target.name
            staged.write_text(
                json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            staged.replace(target)
        generated.append(str(target.relative_to(ROOT)))
    print("\n".join(generated))


if __name__ == "__main__":
    main()
