"""Configurable scenario targets; neither backward generation nor tactical proof."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

from baloot_platform.core.enums import Rank, Seat, Suit
from .scenario_contracts import (
    SCENARIO_EXPRESSION_MAXIMUM_CHILDREN,
    SCENARIO_EXPRESSION_MAXIMUM_DEPTH,
    SCENARIO_EXPRESSION_MAXIMUM_NODES,
    SCENARIO_EXPRESSION_MAXIMUM_SERIALIZED_BYTES,
    _PREDICATE_ID,
    ScenarioExpressionV1, canonical_sha256,
)
from .scenario_registry import ScenarioPredicateRegistryV1
from .scenario_predicate_lowering_v2 import ScenarioPredicateLoweringRegistryV2
from .contracts import canonical_json


GENERATOR_ADMISSION_REASON = "family_disabled"
_NATIVE_PLAY_KINDS = frozenset({
    "play.action_matches", "play.trick_card_count", "play.trick_winner_matches",
})
TAG_AXES = MappingProxyType({
    "impact": ("round-flip", "kaboot-swing", "project-swing", "points-only"),
    "technique": ("guard/protect", "unblock", "partner-entry", "count/void-inference",
                  "project-inference", "endplay", "trump-control", "baloot-timing", "ace-timing"),
    "level": ("beginner", "intermediate", "advanced"),
    "mode": ("sun", "hokum"),
    "trickBand": ("trick-5", "trick-6", "trick-7"),
})


def _mandatory_family_predicates_v1(node):
    if node.op == 'all':
        return [predicate for child in node.children for predicate in _mandatory_family_predicates_v1(child)]
    return [node.predicate] if node.op is None else []


def history_family_absences_v1(expression, *, contract, trump):
    """Validate construction support and extract optional cheap absence filters."""
    validate_family_construction_expression_v1(expression)
    absence = tuple(p for p in _mandatory_family_predicates_v1(expression) if p.kind == 'project.count'
        and p.arguments.get('state') == 'declared' and p.arguments.get('maximum') == 0
        and p.arguments.get('team') in {'north_south', 'west_east'}
        and not set(p.arguments) - {'state', 'team', 'minimum', 'maximum'})
    if contract not in {'sun', 'hokum'} or (contract == 'hokum') != (trump is not None):
        raise ValueError('history family producer requires Sun without trump or Hokum with trump')
    if trump is not None:
        Suit.parse(trump)
    return absence


def family_target_schema_v1():
    seats = {"type": "string", "enum": ["north", "east", "south", "west"]}
    properties = {"schemaVersion": {"type": "integer", "const": 1},
        "catalog": family_catalog_schema_v1(), "familyId": {"type": "string"},
        "tagQuery": tag_query_schema_v1(),
        "bindings": {"type": "object", "additionalProperties": False,
            "required": ["suit", "seat", "rank", "projectType", "contract", "playIndex"],
            "properties": {**{key: {"type": "string"} for key in
                ("suit", "seat", "rank", "projectType", "contract")},
                "playIndex": {"type": "integer", "minimum": 16, "maximum": 27}}},
        "dealer": seats, "buyer": seats,
        "trump": {"enum": [None, "spades", "hearts", "diamonds", "clubs"]}}
    return {"type": "object", "additionalProperties": False, "required": list(properties),
        "properties": {**properties, "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}}


def build_family_target_v1(value):
    # Every decision rechecks this immutable target; validate each payload once.
    serialized = canonical_json(value)
    if value != json.loads(serialized):
        raise ValueError("familyTarget requires JSON types without coercion")
    return json.loads(_build_family_target_cached_v1(serialized))


@lru_cache(maxsize=16)
def _build_family_target_cached_v1(serialized):
    value = json.loads(serialized)
    errors = list(Draft202012Validator(family_target_schema_v1()).iter_errors(value))
    if errors:
        raise ValueError(f"familyTarget: {errors[0].message}")
    raw = deepcopy(dict(value))
    if type(raw["schemaVersion"]) is not int:
        raise ValueError("familyTarget schemaVersion must be integer 1")
    declared = raw.pop("sha256", None)
    catalog = ResearchFamilyCatalogV1.build(raw["catalog"])
    family = next((row for row in catalog.matching_families(raw["tagQuery"])
                   if row.family_id == raw["familyId"] and row.enabled), None)
    if family is None:
        raise ValueError("familyTarget requires an enabled family matching its tag query")
    expression = family.instantiate(raw["bindings"])
    history_family_absences_v1(expression, contract=raw['bindings']['contract'], trump=raw['trump'])
    for axis, value in (("mode", raw["bindings"]["contract"]),
                        ("trickBand", f"trick-{raw['bindings']['playIndex'] // 4 + 1}")):
        if axis in raw["tagQuery"] and value not in raw["tagQuery"][axis]:
            raise ValueError(f"familyTarget bindings do not match tag query {axis}")
    if (raw["bindings"]["contract"] == "hokum") != (raw["trump"] is not None):
        raise ValueError("familyTarget Hokum requires trump; Sun forbids trump")
    raw["catalog"] = catalog.to_dict()
    digest = canonical_sha256(raw)
    if declared is not None and declared != digest:
        raise ValueError("familyTarget checksum mismatch")
    return canonical_json({**raw, "sha256": digest})


def family_target_authoring_v1(target):
    """Instantiate the frozen structural target for visual authoring, not proof."""
    target = build_family_target_v1(target)
    catalog = ResearchFamilyCatalogV1.build(target["catalog"])
    family = next(row for row in catalog.families if row.family_id == target["familyId"])
    expression = family.instantiate(target["bindings"])
    index = target["bindings"]["playIndex"]
    expected = {"remainingCards": 8 - index // 4, "trickIndex": index // 4,
                "position": index % 4, "actingSeat": target["bindings"]["seat"]}
    core = {"schemaVersion": 1, "familyId": family.family_id, "name": family.name,
            "targetHash": target["sha256"], "tagQuery": target["tagQuery"],
            "constraints": expression.to_dict(), "decisionPoint": {"kind": "play", "selector": expected}}
    return {**core, "sha256": canonical_sha256(core)}


def validate_family_target_scenario_v1(target, scenario, *, rules=None, execution=None, playing_quality=None):
    validate_family_target_generation_v1(target, rules=rules, execution=execution, playing_quality=playing_quality)
    authored = family_target_authoring_v1(target)
    expected = authored["decisionPoint"]["selector"]
    selector = scenario.decision_point.get("selector", {})
    if (scenario.origin.kind != "search" or scenario.decision_point["kind"] != "play"
            or selector.get("actingSeat") != expected["actingSeat"]
            or selector.get("trickIndex") != expected["trickIndex"]
            or selector.get("position") != expected["position"]
            or any(value != expected[key] for key, value in selector.items())):
        raise ValueError("familyTarget requires the exact search-play decision actor and index")
    def contains(node):
        return node.sha256 == authored["constraints"]["sha256"] or (
            node.op == "all" and any(contains(child) for child in node.children))
    if not (contains(scenario.constraints) or contains(scenario.outcome)):
        raise ValueError("familyTarget expression must be a mandatory scenario conjunct")


class _FamilyPredicateRegistryV1:
    """Keep historical definitions and admit only registered native selectors."""

    def __init__(self):
        self._legacy = ScenarioPredicateRegistryV1()
        self._native = {definition.kind: definition
                        for definition in ScenarioPredicateLoweringRegistryV2().definitions
                        if definition.kind in _NATIVE_PLAY_KINDS}

    def catalog_definitions(self):
        return [*self._legacy.catalog()["definitions"],
                *(definition.to_dict() for definition in self._native.values())]

    def require(self, predicate):
        if predicate.kind not in _NATIVE_PLAY_KINDS:
            return self._legacy.require(predicate)
        definition = self._native.get(predicate.kind)
        if definition is None or definition.version != predicate.version:
            raise ValueError(f"unsupported scenario predicate: {predicate.kind}@{predicate.version}")
        definition.validate_arguments(predicate.arguments)
        if predicate.hardness not in definition.allowed_hardness:
            raise ValueError(f"{predicate.kind} does not allow hardness {predicate.hardness}")
        return definition


def tag_query_schema_v1() -> dict[str, Any]:
    """Closed AND-of-axes, OR-within-axis query; absent axes are unrestricted."""
    return {"type": "object", "additionalProperties": False, "properties": {
        axis: {"type": "array", "minItems": 1, "uniqueItems": True,
               "items": {"type": "string", "enum": list(values)}}
        for axis, values in TAG_AXES.items()
    }}


def family_generation_request_schema_v1(*, maximum_auction_seeds=None) -> dict[str, Any]:
    auction_seeds = {"type": "array", "minItems": 1, "uniqueItems": True,
        "items": {"type": "object", "additionalProperties": False,
            "required": ["seed", "dealer"], "properties": {
                "seed": {"type": "integer", "minimum": 0},
                "dealer": {"type": "string", "enum": [seat.name.lower() for seat in Seat]}}}}
    if maximum_auction_seeds is not None:
        auction_seeds["maxItems"] = maximum_auction_seeds
    return {"type": "object", "additionalProperties": False,
            "required": ["schemaVersion", "catalogHash", "tagQuery"],
            "properties": {
                "schemaVersion": {"type": "integer", "const": 1},
                "catalogHash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "tagQuery": tag_query_schema_v1(),
                "auctionSeeds": auction_seeds,
                "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}}


def build_family_generation_request_v1(value, *, maximum_auction_seeds=None):
    errors = list(Draft202012Validator(family_generation_request_schema_v1(
        maximum_auction_seeds=maximum_auction_seeds)).iter_errors(value))
    if errors:
        raise ValueError(f"family generation request: {errors[0].message}")
    if type(value["schemaVersion"]) is not int:
        raise ValueError("family generation request schemaVersion must be integer 1")
    core = {"schemaVersion": 1, "catalogHash": value["catalogHash"],
            "tagQuery": TagQueryV1.build(value["tagQuery"]).to_dict()}
    if "auctionSeeds" in value:
        if any(type(row["seed"]) is not int for row in value["auctionSeeds"]):
            raise ValueError("family generation auction seed must be an integer")
        core["auctionSeeds"] = [{"seed": row["seed"], "dealer": row["dealer"]}
                                for row in value["auctionSeeds"]]
    digest = canonical_sha256(core)
    if "sha256" in value and value["sha256"] != digest:
        raise ValueError("family generation request checksum mismatch")
    return {**core, "sha256": digest}


def project_inference_generation_supported_v1(rules, execution, playing_quality=None, *,
                                              play_index=None, actor='north'):
    """Offer the native companion only inside its frozen, exact observer scope."""
    if (getattr(getattr(rules, 'project_domain', None), 'declaration_mode', None) != 'mandatory'
            or execution is None or not isinstance(execution.video_interestingness, Mapping)
            or execution.video_interestingness.get('schemaVersion') != 4):
        return False
    if play_index is None:
        return True  # Authoring may select an exact policy or the configured exact tail.
    if playing_quality is None or play_index < 8:
        return False
    from .scenario_contracts import ScenarioPlayingQualityPolicyV1
    if isinstance(playing_quality, Mapping):
        playing_quality = ScenarioPlayingQualityPolicyV1.build(playing_quality)
    if not isinstance(playing_quality, ScenarioPlayingQualityPolicyV1):
        return False
    request = playing_quality.request_for('play', actor, play_index=play_index, execution_budget=execution)
    return (request.mode == 'exact' and request.visibility == 'actor_information_set'
            and request.utility == 'awarded_point_margin')


def family_search_request_schema_v1(*, maximum_auction_seeds=None, project_inference_supported=False):
    """New searches need a producible OR alternative on every requested axis."""
    from .lesson_tags import live_lesson_tag_values_v1

    schema = family_generation_request_schema_v1(maximum_auction_seeds=maximum_auction_seeds)
    properties = schema['properties']['tagQuery']['properties']
    properties.pop('level')
    for axis, values in {**live_lesson_tag_values_v1(
            project_inference_supported=project_inference_supported), 'mode': ['sun']}.items():
        properties[axis]['contains'] = {'enum': values}
    return schema


def _live_generation_tag_query_v1(query, *, project_inference_supported=False):
    query = TagQueryV1.build(query).to_dict()
    if 'level' in query:
        raise ValueError('generation level requires an independent lesson audit')
    properties = family_search_request_schema_v1(
        project_inference_supported=project_inference_supported)['properties']['tagQuery']['properties']
    supported_query = {}
    for axis, values in query.items():
        supported = properties[axis].get('contains', properties[axis]['items'])['enum']
        supported_query[axis] = [value for value in values if value in supported]
        if not supported_query[axis]:
            raise ValueError(f'generation tag {axis} has no live producer for the requested alternatives')
    return supported_query


def family_generation_techniques_v1(family):
    """Historical capability labels cannot enable a structurally excluded clue."""
    techniques = family.capability_tags['technique']
    if 'project-inference' in techniques and 'sun' in family.contracts:
        bindings = {key: values[0] for key, values in family.parameters.choices(('sun',)).items()}
        expression = family.instantiate(bindings)
        if any(p.hardness == 'hard' and p.kind == 'project.count'
               and p.arguments.get('state') in {'eligible', 'declared'}
               and p.arguments.get('maximum') == 0
               for p in _mandatory_family_predicates_v1(expression)):
            techniques.remove('project-inference')
    return techniques


def family_matches_generation_query_v1(family, query):
    tags = {**family.capability_tags, 'technique': family_generation_techniques_v1(family)}
    return TagQueryV1.build(query).matches(tags)


def validate_family_target_generation_v1(target, *, rules=None, execution=None, playing_quality=None):
    """Admission only; parsing a historical frozen target never changes it."""
    target = build_family_target_v1(target)
    contract = target['bindings']['contract']
    query = _live_generation_tag_query_v1(
        {axis: values for axis, values in target['tagQuery'].items() if axis != 'mode'},
        project_inference_supported=(contract == 'sun'
        and 'project-inference' in target['tagQuery'].get('technique', ())
        and project_inference_generation_supported_v1(rules, execution, playing_quality,
            play_index=target['bindings']['playIndex'], actor=target['bindings']['seat'])))
    query['mode'] = [contract]  # Explicit targets already bind their mode and trump.
    catalog = ResearchFamilyCatalogV1.build(target['catalog'])
    if not any(family.family_id == target['familyId'] and family_matches_generation_query_v1(family, query)
               for family in catalog.matching_families(query)):
        raise ValueError('familyTarget has no live producer for its requested tag alternatives')


def matching_generation_families_v1(request, catalog, scenario, *, play_indices=None,
                                    maximum_auction_seeds=None, rules=None, execution=None, playing_quality=None):
    request = build_family_generation_request_v1(request, maximum_auction_seeds=maximum_auction_seeds)
    if request["catalogHash"] != catalog.sha256:
        raise ValueError("generation catalogHash must match the selected bundle")
    if scenario.origin.kind != "search" or scenario.decision_point["kind"] != "play":
        raise ValueError("family generation requires a search-play decision")
    if any(predicate.hardness == 'hard' and predicate.kind == 'auction.contract'
           and predicate.arguments['contract'] != 'sun'
           for expression in (scenario.constraints, scenario.outcome)
           for predicate in _mandatory_family_predicates_v1(expression)):
        raise ValueError('family generation supports Sun structural templates only')
    selector = scenario.decision_point.get("selector", {})
    if (type(selector.get("trickIndex")) is not int
            or type(selector.get("position")) is not int
            or selector.get("actingSeat") not in {seat.name.lower() for seat in Seat}):
        raise ValueError("family generation requires an exact trick, position and acting seat")
    index = selector["trickIndex"] * 4 + selector["position"]
    if (play_indices is not None and index not in play_indices
            or selector.get("remainingCards", 8 - index // 4) != 8 - index // 4):
        raise ValueError("family generation decision exceeds the frozen play boundaries")
    query = request["tagQuery"]
    supported_query = _live_generation_tag_query_v1(query, project_inference_supported=(
        'project-inference' in query.get('technique', ())
        and project_inference_generation_supported_v1(rules, execution, playing_quality,
            play_index=index, actor=selector['actingSeat'])))
    if ("trickBand" in query and f"trick-{index // 4 + 1}" not in query["trickBand"]
            or "mode" in query and "sun" not in query["mode"]):
        raise ValueError("generation tags do not match the supported decision context")
    families = tuple(family for family in catalog.matching_families(supported_query)
                     if family.enabled and "sun" in family.contracts
                     and family_matches_generation_query_v1(family, supported_query)
                     and selector["actingSeat"] in family.parameters.seats
                     and family.parameters.minimum_play_index <= index <= family.parameters.maximum_play_index)
    if not families:
        raise ValueError("no enabled supported family matches the generation request")
    return families


def lesson_tag_catalog_v1() -> dict[str, Any]:
    """Shared visual query metadata; capability tags never certify a lesson."""
    core = {"schemaVersion": 1, "axes": [
        {"id": axis, "queryParameter": f"lesson{axis[0].upper()}{axis[1:]}",
         "labelKey": f"research_lesson_tag_axis_{axis}",
         "options": [{"value": value,
                      "labelKey": f"research_lesson_tag_{axis}_{value.replace('/', '_').replace('-', '_')}"}
                     for value in values]}
        for axis, values in TAG_AXES.items()]}
    return {**core, "sha256": canonical_sha256(core)}


@dataclass(frozen=True, slots=True)
class TagQueryV1:
    selections: tuple[tuple[str, tuple[str, ...]], ...]

    @classmethod
    def build(cls, payload: Mapping[str, Any]) -> TagQueryV1:
        errors = list(Draft202012Validator(tag_query_schema_v1()).iter_errors(payload))
        if errors:
            raise ValueError(f"tag query: {errors[0].message}")
        return cls(tuple((axis, tuple(value for value in values if value in payload[axis]))
                         for axis, values in TAG_AXES.items() if axis in payload))

    def to_dict(self) -> dict[str, list[str]]:
        return {axis: list(values) for axis, values in self.selections}

    def matches(self, tags: Mapping[str, Any]) -> bool:
        return all(set(values).intersection(tags.get(axis, ()))
                   for axis, values in self.selections)


def family_catalog_schema_v1() -> dict[str, Any]:
    """The editor and loader share this closed authoring schema."""
    def obj(properties):
        return {"type": "object", "additionalProperties": False,
                "required": list(properties), "properties": properties}

    def choices(values):
        return {"type": "array", "minItems": 1, "uniqueItems": True,
                "items": {"type": "string", "enum": values}}

    identifier = {"type": "string", "pattern": "^[a-z][a-z0-9_-]{0,79}$"}
    parameters = obj({
        "suits": choices([item.name.lower() for item in Suit]),
        "seats": choices([item.name.lower() for item in Seat]),
        "ranks": choices([item.name.lower() for item in Rank]),
        "projectTypes": choices(["siri", "fifty", "hundred_sequence",
                                  "hundred_four_kind", "four_hundred"]),
        "playIndex": obj({key: {"type": "integer", "minimum": 16, "maximum": 27}
                          for key in ("minimum", "maximum")}),
    })
    entry = obj({
        "id": identifier, "name": {"type": "string", "minLength": 1, "maxLength": 160},
        "enabled": {"type": "boolean"},
        "weight": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000},
        "minimumAllocation": {"type": "number", "minimum": 0, "maximum": 1},
        "contracts": choices(["sun", "hokum"]), "parameters": parameters,
        "capabilityTags": {**tag_query_schema_v1(), "required": list(TAG_AXES)},
        "template": obj({"kind": {"const": "scenario_predicates_v1"},
                         "expression": family_expression_schema_v1()}),
    })
    schema = obj({"schemaVersion": {"const": 1, "type": "integer"}, "id": identifier,
                  "revision": {"type": "integer", "minimum": 1},
                  "families": {"type": "array", "minItems": 1, "maxItems": 128,
                               "items": entry}})
    schema["required"] = list(schema["properties"])
    schema["properties"]["sha256"] = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def family_expression_schema_v1() -> dict[str, Any]:
    """The existing profile editor consumes the same closed predicate grammar.

    Its local schema resource keeps recursive references valid when embedded
    inside a challenge profile. Binding-domain validation remains authoritative.
    """
    domains = {
        "suit": [item.name.lower() for item in Suit],
        "seat": [item.name.lower() for item in Seat],
        "rank": [item.name.lower() for item in Rank],
        "projectType": ["siri", "fifty", "hundred_sequence", "hundred_four_kind", "four_hundred"],
        "contract": ["sun", "hokum"], "playIndex": list(range(16, 28)),
    }

    def parameterize(schema):
        result = deepcopy(dict(schema))
        if result.get("type") == "object":
            result["properties"] = {key: parameterize(value)
                                    for key, value in result.get("properties", {}).items()}
        elif result.get("type") == "array" and isinstance(result.get("items"), dict):
            result["items"] = parameterize(result["items"])
        elif result.get("type") in {"string", "integer", "number"}:
            validator = Draft202012Validator(result)
            options = [key for key, values in domains.items()
                       if any(validator.is_valid(value) for value in values)]
            if options:
                return {**{key: value for key, value in result.items() if key.startswith("x-")},
                    "anyOf": [result, {
                    "type": "object", "additionalProperties": False,
                    "required": ["$parameter"], "properties": {
                        "$parameter": {"type": "string", "enum": options}},
                }], "x-familyParameterOptions": options}
        return result

    branches = []
    checksum = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    for op in ("all", "any", "not", "count"):
        properties = {"op": {"type": "string", "const": op},
                      "children": {"type": "array", "minItems": 1,
                                   "x-messageArKey": "research_scenario_expression_group",
                                   "maxItems": 1 if op == "not" else SCENARIO_EXPRESSION_MAXIMUM_CHILDREN,
                                   "items": {"$ref": "#/$defs/expression"}},
                      "sha256": checksum}
        required = ["op", "children"]
        if op == "count":
            properties.update({key: {"type": "integer", "minimum": 0,
                                    "x-messageArKey": f"research_scenario_count_{key}",
                                    "maximum": SCENARIO_EXPRESSION_MAXIMUM_CHILDREN}
                               for key in ("minimum", "maximum")})
            required += ["minimum", "maximum"]
        branches.append({"type": "object", "additionalProperties": False,
                         "properties": properties, "required": required,
                         "x-discriminator": "op", "x-labelKey": f"research_scenario_operator_{op}"})
    for definition in _FamilyPredicateRegistryV1().catalog_definitions():
        branches.append({"type": "object", "additionalProperties": False,
            "properties": {
                "predicateId": {"type": "string", "pattern": _PREDICATE_ID.pattern,
                                "x-messageArKey": "research_lesson_tag_template_predicate_id"},
                "kind": {"type": "string", "const": definition["kind"]},
                "version": {"type": "integer", "const": definition["version"]},
                "hardness": {"type": "string", "enum": definition["allowedHardness"],
                             "x-messageArKey": "research_scenario_hardness"},
                "arguments": {**parameterize(definition["argumentSchema"]),
                              "x-messageArKey": "research_lesson_tag_template_arguments"}, "sha256": checksum,
            }, "required": ["predicateId", "kind", "version", "hardness", "arguments"],
            "x-discriminator": "kind", "x-labelKey": definition["messageArKey"],
            "x-constructionSupported": family_construction_kind_supported_v1(definition["kind"])})
    expression = {"type": "object", "oneOf": branches, "x-familyExpression": True,
                  "x-expressionLimits": {"maximumDepth": SCENARIO_EXPRESSION_MAXIMUM_DEPTH,
                                         "maximumNodes": SCENARIO_EXPRESSION_MAXIMUM_NODES,
                                         "maximumSerializedBytes": SCENARIO_EXPRESSION_MAXIMUM_SERIALIZED_BYTES}}
    return {"type": "object", "x-familyExpression": True,
            "$id": "https://baloot.local/schemas/research/family-expression-v1",
            "$ref": "#/$defs/expression", "$defs": {"expression": expression}}


@dataclass(frozen=True, slots=True)
class FamilyParametersV1:
    suits: tuple[str, ...]
    seats: tuple[str, ...]
    ranks: tuple[str, ...]
    project_types: tuple[str, ...]
    minimum_play_index: int
    maximum_play_index: int

    def choices(self, contracts: tuple[str, ...]) -> dict[str, tuple]:
        return {"suit": self.suits, "seat": self.seats, "rank": self.ranks,
                "projectType": self.project_types, "contract": contracts,
                "playIndex": tuple(range(self.minimum_play_index, self.maximum_play_index + 1))}


def _substitute(value, bindings, *, depth=0):
    if depth > 24:
        raise ValueError("family template exceeds maximum nesting")
    if isinstance(value, dict):
        if "$parameter" in value:
            if set(value) != {"$parameter"} or value["$parameter"] not in bindings:
                raise ValueError("invalid family parameter reference")
            return bindings[value["$parameter"]]
        return {key: _substitute(item, bindings, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, bindings, depth=depth + 1) for item in value]
    return value


def _validate_expression(expression, registry):
    ids = set()
    def visit(node):
        if node.predicate is not None:
            if node.predicate.predicate_id in ids:
                raise ValueError("duplicate family predicateId")
            ids.add(node.predicate.predicate_id)
            registry.require(node.predicate)
        for child in node.children:
            visit(child)
    visit(expression)


def family_construction_kind_supported_v1(kind):
    return (kind.startswith(("deal.", "auction.", "project.", "outcome."))
            or kind in _NATIVE_PLAY_KINDS
            or kind in {"state.legal_action_count", "state.current_raw_points"})


def validate_family_construction_expression_v1(expression, *, _registry=None):
    """Reject predicates without an exact native construction context before solve."""
    registry = _registry if _registry is not None else _FamilyPredicateRegistryV1()
    for predicate in expression.predicates():
        registry.require(predicate)
        if predicate.hardness != "hard":
            raise ValueError("construction templates currently require hard predicates only")
        if not family_construction_kind_supported_v1(predicate.kind):
            raise ValueError(f"family construction has no exact context for {predicate.kind}")
    def visit(node, inverted=False):
        if node.predicate is not None and inverted and not registry.require(node.predicate).safe_complement:
            raise ValueError("family predicate has no safe complement")
        for child in node.children:
            visit(child, inverted or node.op == "not")
    visit(expression)


def family_construction_expression_v1(expression, scenario=None):
    """Add complete native hard conjuncts; other stages still qualify the full recipe."""
    validate_family_construction_expression_v1(expression)
    expressions = {expression.sha256: expression}

    def collect(node):
        predicates = node.predicates()
        if all(predicate.hardness == 'hard' and family_construction_kind_supported_v1(predicate.kind)
               for predicate in predicates):
            validate_family_construction_expression_v1(node)
            expressions.setdefault(node.sha256, node)
        elif node.op == 'all':
            for child in node.children:
                collect(child)
        # ANY, NOT and COUNT must never be strengthened by dropping a child.

    if scenario is not None:
        collect(scenario.constraints)
        collect(scenario.outcome)
    predicates = {}
    for node in expressions.values():
        for predicate in node.predicates():
            previous = predicates.setdefault(predicate.predicate_id, predicate.sha256)
            if previous != predicate.sha256:
                raise ValueError('conflicting family/scenario predicateId: ' + predicate.predicate_id)
    if len(expressions) == 1:
        return expression
    return ScenarioExpressionV1.build({'op': 'all',
        'children': [node.to_dict() for node in expressions.values()]})


@dataclass(frozen=True, slots=True)
class ResearchFamilyV1:
    family_id: str
    name: str
    enabled: bool
    weight: float
    minimum_allocation: float
    contracts: tuple[str, ...]
    parameters: FamilyParametersV1
    _json: str

    @property
    def generator_admission_reason(self) -> str | None:
        return None if self.enabled else GENERATOR_ADMISSION_REASON

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._json)

    @property
    def capability_tags(self) -> dict[str, list[str]]:
        """Targeting declarations, never proof that a lesson has these tags."""
        return self.to_dict()["capabilityTags"]

    def instantiate(self, bindings: Mapping[str, Any]) -> ScenarioExpressionV1:
        """Validate all bindings and lower a target, NOT a constructed position.

        The future adapter must use playIndex as its decision-prefix boundary;
        it is not encoded by a scenario predicate and is not proof of reachability.
        """
        choices = self.parameters.choices(self.contracts)
        if set(bindings) != set(choices):
            raise ValueError("family bindings must contain exactly " + ", ".join(choices))
        for key, allowed in choices.items():
            if type(bindings[key]) is not type(allowed[0]) or bindings[key] not in allowed:
                raise ValueError(f"family binding {key} is outside its configured range")
        expression = ScenarioExpressionV1.build(_substitute(
            self.to_dict()["template"]["expression"], bindings,
        ))
        _validate_expression(expression, _FamilyPredicateRegistryV1())
        return expression


@dataclass(frozen=True, slots=True)
class ResearchFamilyCatalogV1:
    catalog_id: str
    revision: int
    families: tuple[ResearchFamilyV1, ...]
    sha256: str

    @classmethod
    def build(cls, payload: Mapping[str, Any]) -> ResearchFamilyCatalogV1:
        return cls._build_cached(json.dumps(payload, allow_nan=False))

    @classmethod
    @lru_cache(maxsize=16)
    def _build_cached(cls, serialized: str) -> ResearchFamilyCatalogV1:
        # Policy requests reuse immutable catalogs, never caller-owned mappings.
        raw = json.loads(serialized)
        errors = sorted(Draft202012Validator(family_catalog_schema_v1()).iter_errors(raw),
                        key=lambda error: str(list(error.path)))
        if errors:
            raise ValueError(f"family catalog {list(errors[0].path)}: {errors[0].message}")
        if type(raw["schemaVersion"]) is not int or type(raw["revision"]) is not int:
            raise ValueError("family catalog versions must be integers")
        rows = []
        ids = set()
        registry = _FamilyPredicateRegistryV1()
        for row in raw["families"]:
            if row["id"] in ids:
                raise ValueError("duplicate family id")
            ids.add(row["id"])
            bounds = row["parameters"]["playIndex"]
            if any(type(value) is not int for value in bounds.values()):
                raise ValueError("family playIndex bounds must be integers")
            if bounds["minimum"] > bounds["maximum"]:
                raise ValueError("family playIndex bounds are inverted")
            tags = row["capabilityTags"]
            if set(tags["mode"]) != set(row["contracts"]):
                raise ValueError("family capability mode must match contracts")
            bands = {f"trick-{index // 4 + 1}" for index in
                     range(bounds["minimum"], bounds["maximum"] + 1)}
            if set(tags["trickBand"]) != bands:
                raise ValueError("family capability trickBand must match playIndex range")
            params = row["parameters"]
            if "four_hundred" in params["projectTypes"] and "hokum" in row["contracts"]:
                raise ValueError("four_hundred project range requires Sun-only family")
            family = ResearchFamilyV1(
                row["id"], row["name"], row["enabled"], row["weight"], row["minimumAllocation"],
                tuple(row["contracts"]), FamilyParametersV1(
                    tuple(params["suits"]), tuple(params["seats"]), tuple(params["ranks"]),
                    tuple(params["projectTypes"]), bounds["minimum"], bounds["maximum"],
                ), json.dumps(row, sort_keys=True, separators=(",", ":")),
            )
            choices = family.parameters.choices(family.contracts)
            # Validate each domain value, not just an example. The closed predicate
            # argument schema owns scalar constraints; no executable template code.
            baseline = {key: values[0] for key, values in choices.items()}
            for key, values in choices.items():
                for value in values:
                    expression = ScenarioExpressionV1.build(_substitute(
                        row["template"]["expression"], {**baseline, key: value},
                    ))
                    _validate_expression(expression, registry)
                    if family.enabled:
                        validate_family_construction_expression_v1(expression, _registry=registry)
                    for predicate in _predicates(expression):
                        if predicate.kind == "project.baloot" and family.contracts != ("hokum",):
                            raise ValueError("Baloot target requires Hokum-only family")
            rows.append(family)
        if math.fsum(row.minimum_allocation for row in rows) > 1:
            raise ValueError("family minimum allocations exceed one")
        declared = raw.pop("sha256", None)
        digest = canonical_sha256(raw)
        if declared is not None and declared != digest:
            raise ValueError("family catalog checksum mismatch")
        return cls(raw["id"], raw["revision"], tuple(rows), digest)

    @classmethod
    def load(cls, path: str | Path) -> ResearchFamilyCatalogV1:
        class UniqueLoader(yaml.SafeLoader):
            pass

        def mapping(loader, node):
            result = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node)
                if key in result:
                    raise ValueError(f"duplicate family YAML key: {key}")
                result[key] = loader.construct_object(value_node)
            return result

        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        return cls.build(yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueLoader))

    def to_dict(self) -> dict[str, Any]:
        return {"schemaVersion": 1, "id": self.catalog_id, "revision": self.revision,
                "families": [family.to_dict() for family in self.families], "sha256": self.sha256}

    @lru_cache(maxsize=16)
    def require_sun_history_support(self) -> None:
        # Frozen members make this content-keyed check safe to reuse across decisions.
        for family in self.families:
            if family.enabled:
                if family.contracts != ("sun",):
                    raise ValueError("familyCatalog currently supports enabled Sun families only")
                choices = family.parameters.choices(family.contracts)
                history_family_absences_v1(family.instantiate({key: values[0] for key, values in choices.items()}),
                                           contract="sun", trump=None)

    def matching_families(self, query: Mapping[str, Any]) -> tuple[ResearchFamilyV1, ...]:
        target = TagQueryV1.build(query)
        return tuple(family for family in self.families if target.matches(family.capability_tags))

    def allocation_shares(self, query: Mapping[str, Any], *,
                          audited_yield: Mapping[str, Any] | None = None,
                          eligible_family_ids: set[str] | None = None) -> dict[str, float]:
        """Authoring-only allocation preview, NOT admission or job scheduling.

        Filter before assigning floors and weighted remainder. Disabled families
        remain visible for planning; the generator admission gate is unchanged.
        Audited samples must belong to this catalog revision. Unmeasured families
        retain an exploration share; zero-yield families retain their floor.
        """
        rates = {}
        if audited_yield is not None:
            if (not isinstance(audited_yield, Mapping)
                    or set(audited_yield) != {"catalogHash", "families"}
                    or audited_yield["catalogHash"] != self.sha256
                    or not isinstance(audited_yield["families"], Mapping)):
                raise ValueError("audited yield requires matching catalog hash and family samples")
            known = {family.family_id for family in self.families}
            for family_id, sample in audited_yield["families"].items():
                if (family_id not in known or not isinstance(sample, Mapping)
                        or set(sample) != {"newLessons", "cpuSeconds"}
                        or type(sample["newLessons"]) is not int or sample["newLessons"] < 0
                        or type(sample["cpuSeconds"]) not in (int, float)
                        or not math.isfinite(sample["cpuSeconds"]) or sample["cpuSeconds"] <= 0):
                    raise ValueError("invalid audited family yield")
                rate = sample["newLessons"] / sample["cpuSeconds"]
                if not math.isfinite(rate):
                    raise ValueError("invalid audited family yield")
                rates[family_id] = rate
        families = tuple(family for family in self.matching_families(query)
                         if eligible_family_ids is None or family.family_id in eligible_family_ids)
        if not families:
            return {}
        remainder = 1 - math.fsum(family.minimum_allocation for family in families)
        best_rate = max((rates.get(family.family_id, 0) for family in families), default=0)
        weights = {family.family_id: family.weight * (
            rates.get(family.family_id, best_rate) / best_rate if best_rate else 1)
            for family in families}
        total_weight = math.fsum(weights.values())
        return {family.family_id: family.minimum_allocation + remainder * weights[family.family_id] / total_weight
                for family in families}


def _predicates(expression):
    if expression.predicate is not None:
        yield expression.predicate
    for child in expression.children:
        yield from _predicates(child)
