import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from baloot_platform.research.family_catalog import (
    GENERATOR_ADMISSION_REASON,
    ResearchFamilyCatalogV1,
    TAG_AXES,
    TagQueryV1,
    family_catalog_schema_v1,
    family_expression_schema_v1,
    family_generation_request_schema_v1,
    family_search_request_schema_v1,
    tag_query_schema_v1,
    lesson_tag_catalog_v1,
    validate_family_construction_expression_v1,
)
from baloot_platform.research.scenario_contracts import ScenarioExpressionV1, ScenarioPredicateV1, canonical_sha256
from baloot_platform.research.scenario_predicate_lowering_v2 import ScenarioPredicateLoweringRegistryV2
from baloot_platform.research.scenario_registry import ScenarioPredicateRegistryV1


CONFIG = Path(__file__).resolve().parents[1] / "configs/research/families.yaml"
NATIVE_PLAY_KINDS = {"play.action_matches", "play.trick_card_count", "play.trick_winner_matches"}


def test_search_schema_requires_a_live_producer_but_preserves_known_or_alternatives():
    from baloot_platform.research.lesson_tags import live_lesson_tag_values_v1

    schema = family_search_request_schema_v1(maximum_auction_seeds=3)
    query = schema['properties']['tagQuery']
    validator = Draft202012Validator(query)
    assert 'level' not in query['properties']
    for axis, values in live_lesson_tag_values_v1().items():
        assert query['properties'][axis]['contains']['enum'] == values
        assert query['properties'][axis]['items']['enum'] == list(TAG_AXES[axis])
        for value in TAG_AXES[axis]:
            assert validator.is_valid({axis: [value]}) == (value in values)
            assert validator.is_valid({axis: list(dict.fromkeys([value, values[0]]))})
    assert validator.is_valid({'mode': ['sun', 'hokum']})
    assert not validator.is_valid({'mode': ['hokum']})
    assert not validator.is_valid({'level': ['intermediate']})
    assert schema['properties']['auctionSeeds']['maxItems'] == 3
    historical = family_generation_request_schema_v1()['properties']['tagQuery']
    assert Draft202012Validator(historical).is_valid(
        {'impact': ['project-swing'], 'technique': ['project-inference'], 'level': ['advanced']})
    schema['properties']['tagQuery']['properties']['impact']['contains']['enum'].clear()
    assert family_search_request_schema_v1()['properties']['tagQuery']['properties']['impact']['contains']['enum']


def test_lesson_query_catalog_owns_every_axis_option_and_profile_copy(root):
    import yaml
    from baloot_platform.research.scenario_contracts import canonical_sha256
    metadata = lesson_tag_catalog_v1()
    assert metadata['sha256'] == canonical_sha256({key: value for key, value in metadata.items() if key != 'sha256'})
    copy = yaml.safe_load((root / 'configs/product/studio_ortools.yaml').read_text(encoding='utf-8'))['foundry_copy']['groups']['workflow']
    assert {axis['id'] for axis in metadata['axes']} == set(TAG_AXES)
    for axis in metadata['axes']:
        assert axis['queryParameter'] == f"lesson{axis['id'][0].upper()}{axis['id'][1:]}"
        assert [option['value'] for option in axis['options']] == list(TAG_AXES[axis['id']])
        assert copy[axis['labelKey']]
        assert all(copy[option['labelKey']] for option in axis['options'])


@pytest.mark.parametrize('version,prefixes', [
    ('1.1.30', ('research_lesson_tag', 'research_auction_seeds_')),
    ('1.1.34', ('research_auction_seeds_',)),
])
def test_historical_product_copy_can_omit_tag_controls_but_unknown_keys_fail(root, version, prefixes):
    import yaml
    schema = json.loads((root / 'schemas/profiles/product.schema.json').read_text(encoding='utf-8'))
    historical = yaml.safe_load((root / 'configs/product/studio_ortools.yaml').read_text(encoding='utf-8'))
    historical['profile']['version'] = version
    workflow = historical['foundry_copy']['groups']['workflow']
    for key in list(workflow):
        if key.startswith(prefixes):
            del workflow[key]
    validator = Draft202012Validator(schema)
    validator.validate(historical)
    workflow['research_lesson_tag_invented'] = 'unsupported'
    with pytest.raises(ValidationError, match='not valid|Additional properties'):
        validator.validate(historical)


@pytest.fixture(scope="module")
def catalog():
    return ResearchFamilyCatalogV1.load(CONFIG)


def authored(catalog):
    raw = catalog.to_dict()
    raw.pop("sha256")
    return raw


def test_repeated_catalog_validation_is_bounded_detached_and_content_keyed(catalog, monkeypatch):
    from baloot_platform.research import family_catalog as module

    raw = catalog.to_dict()
    raw['id'] = 'cached_validation_regression'
    raw.pop('sha256')
    calls = []
    schema = module.family_catalog_schema_v1
    def counted_schema():
        calls.append(True)
        return schema()
    monkeypatch.setattr(module, 'family_catalog_schema_v1', counted_schema)
    ResearchFamilyCatalogV1._build_cached.cache_clear()
    try:
        cached = ResearchFamilyCatalogV1.build(raw)
        assert ResearchFamilyCatalogV1.build(copy.deepcopy(raw)) is cached
        assert len(calls) == 1
        returned = cached.to_dict()
        returned['families'][0]['name'] = 'caller edit'
        assert cached.to_dict()['families'][0]['name'] != 'caller edit'
        original_revision = raw['revision']
        raw['revision'] += 1
        changed = ResearchFamilyCatalogV1.build(raw)
        assert changed.sha256 != cached.sha256 and cached.revision == original_revision
        assert len(calls) == 2
        stale = cached.to_dict()
        stale['revision'] += 1
        with pytest.raises(ValueError, match='checksum'):
            ResearchFamilyCatalogV1.build(stale)
        unknown = cached.to_dict()
        unknown['unsupported'] = True
        with pytest.raises(ValueError, match='Additional properties'):
            ResearchFamilyCatalogV1.build(unknown)
        assert ResearchFamilyCatalogV1._build_cached.cache_info().maxsize == 16
    finally:
        ResearchFamilyCatalogV1._build_cached.cache_clear()


def test_enabled_family_domain_validation_reuses_one_registered_authority(catalog, monkeypatch):
    from baloot_platform.research import family_catalog as module

    raw = authored(catalog)
    raw['families'] = raw['families'][:1]
    raw['families'][0]['enabled'] = True
    observed = []
    validate = module.validate_family_construction_expression_v1
    def record(expression, *, _registry=None):
        observed.append(_registry)
        return validate(expression, _registry=_registry)
    monkeypatch.setattr(module, 'validate_family_construction_expression_v1', record)

    module.ResearchFamilyCatalogV1._build_cached.cache_clear()
    result = module.ResearchFamilyCatalogV1.build(raw)
    assert len(observed) > 1 and all(registry is observed[0] for registry in observed)
    assert observed[0] is not None
    assert result.sha256 == canonical_sha256(raw)


def bindings(family):
    return {key: values[0] for key, values in family.parameters.choices(family.contracts).items()}


def native_predicate(kind, arguments, predicate_id="PRED-NATIVE"):
    return {"predicateId": predicate_id, "kind": kind, "version": 1,
            "hardness": "hard", "arguments": arguments}


def native_catalog(catalog, expression, *, enabled=False):
    raw = authored(catalog)
    raw["families"] = raw["families"][:1]
    raw["families"][0]["enabled"] = enabled
    raw["families"][0]["template"]["expression"] = expression
    return raw


def test_seeded_catalog_is_disabled_versioned_hashed_and_roundtrips(catalog):
    import yaml

    assert catalog.sha256 == canonical_sha256(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    assert {row.family_id for row in catalog.families} == {
        "negative-project", "positive-project", "stopper_guard", "overtake_unblock",
        "kaboot_prevention", "trump_management", "void_count", "endplay",
        "baloot_timing", "ekka_timing",
    }
    assert all(not row.enabled for row in catalog.families)
    assert all(row.generator_admission_reason == GENERATOR_ADMISSION_REASON for row in catalog.families)
    assert ResearchFamilyCatalogV1.build(catalog.to_dict()) == catalog
    Draft202012Validator(family_catalog_schema_v1()).validate(catalog.to_dict())
    for family in catalog.families:
        assert isinstance(family.instantiate(bindings(family)), ScenarioExpressionV1)


def test_custom_family_needs_only_config_not_a_python_enum(catalog):
    raw = authored(catalog)
    family = copy.deepcopy(raw["families"][2])
    family.update(id="custom_long_guard", name="Custom guard", minimumAllocation=0)
    family["template"]["expression"]["children"][1]["arguments"]["minimum"] = 3
    raw["families"].append(family)
    changed = ResearchFamilyCatalogV1.build(raw)
    assert changed.sha256 != catalog.sha256
    instance = changed.families[-1].instantiate({**bindings(changed.families[-1]), "suit": "hearts"})
    assert instance.children[1].predicate.arguments["suit"] == "hearts"
    assert instance.children[1].predicate.arguments["minimum"] == 3


@pytest.mark.parametrize("field,value", [
    ("enabled", "false"), ("weight", True), ("weight", "1"),
    ("weight", 0), ("weight", float("nan")), ("weight", float("inf")),
    ("minimumAllocation", -1), ("contracts", ["ashkal"]), ("contracts", ["sun", "sun"]),
    ("id", "../escape"), ("unknown", 1),
])
def test_strict_entry_validation(catalog, field, value):
    raw = authored(catalog)
    raw["families"][0][field] = value
    with pytest.raises(ValueError):
        ResearchFamilyCatalogV1.build(raw)


@pytest.mark.parametrize("mutation", [
    lambda raw: raw.update(schemaVersion=True),
    lambda raw: raw.update(schemaVersion=1.0),
    lambda raw: raw.update(revision=1.0),
    lambda raw: raw.update(revision="1"),
    lambda raw: raw["families"].append(copy.deepcopy(raw["families"][0])),
    lambda raw: raw["families"][0]["parameters"].update(suits=["bogus"]),
    lambda raw: raw["families"][0]["parameters"].update(ranks=["one"]),
    lambda raw: raw["families"][0]["parameters"].update(projectTypes=["baloot"]),
    lambda raw: raw["families"][0]["parameters"].update(projectTypes=["four_hundred"]),
    lambda raw: raw["families"][0]["parameters"]["playIndex"].update(minimum=15),
    lambda raw: raw["families"][0]["parameters"]["playIndex"].update(minimum=16.0),
    lambda raw: raw["families"][0]["capabilityTags"].pop("level"),
    lambda raw: raw["families"][0]["parameters"]["playIndex"].update(maximum=28),
    lambda raw: raw["families"][0]["parameters"]["playIndex"].update(minimum=25, maximum=20),
    lambda raw: raw["families"][0]["template"].update(kind="python"),
    lambda raw: raw["families"][0]["template"]["expression"]["children"][0].update(kind="unknown"),
    lambda raw: raw["families"][0]["template"]["expression"]["children"][0]["arguments"].update(contract={"$parameter": "unknown"}),
    lambda raw: raw["families"][8].update(contracts=["sun"]),
    lambda raw: raw["families"][0].update(minimumAllocation=1),
])
def test_invalid_catalog_domains_and_templates_fail_closed(catalog, mutation):
    raw = authored(catalog)
    mutation(raw)
    with pytest.raises((ValueError, TypeError)):
        ResearchFamilyCatalogV1.build(raw)


def test_all_domain_values_are_validated(catalog):
    raw = authored(catalog)
    expression = raw["families"][0]["template"]["expression"]
    expression["children"][0]["arguments"]["contract"] = {"$parameter": "suit"}
    with pytest.raises(ValueError):
        ResearchFamilyCatalogV1.build(raw)


def test_checksum_cannot_be_reused_after_edit_and_snapshots_are_detached(catalog):
    raw = catalog.to_dict()
    raw["families"][0]["weight"] = 2
    with pytest.raises(ValueError, match="checksum"):
        ResearchFamilyCatalogV1.build(raw)
    assert catalog.families[0].weight == 1
    original = authored(catalog)
    rebuilt = ResearchFamilyCatalogV1.build(original)
    original["families"][0]["name"] = "changed"
    assert rebuilt.families[0].name == catalog.families[0].name


@pytest.mark.parametrize("key,value", [("playIndex", True), ("playIndex", 28), ("seat", "bogus"), ("suit", "SPADES")])
def test_instantiation_rejects_out_of_range_bindings(catalog, key, value):
    family = catalog.families[0]
    with pytest.raises(ValueError):
        family.instantiate({**bindings(family), key: value})


def test_yaml_duplicate_keys_and_unsafe_tags_are_rejected(tmp_path):
    path = tmp_path / "families.yaml"
    path.write_text("schemaVersion: 1\nschemaVersion: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        ResearchFamilyCatalogV1.load(path)
    path.write_text("!!python/object/apply:os.system ['echo forbidden']", encoding="utf-8")
    import yaml
    with pytest.raises(yaml.YAMLError):
        ResearchFamilyCatalogV1.load(path)


def test_schema_is_valid_json_schema():
    schema = family_catalog_schema_v1()
    Draft202012Validator.check_schema(schema)
    assert json.loads(json.dumps(schema)) == schema


def test_template_schema_is_recursive_closed_and_valid_inside_a_profile(catalog):
    expression_schema = family_expression_schema_v1()
    schema = {"type": "object", "properties": {"target": expression_schema}}
    validator = Draft202012Validator(schema)
    expression = copy.deepcopy(catalog.families[2].to_dict()["template"]["expression"])
    nested = {"op": "any", "children": [expression, copy.deepcopy(expression)]}
    validator.validate({"target": nested})
    nested["children"][1]["children"][0]["arguments"]["unexpected"] = True
    with pytest.raises(ValidationError):
        validator.validate({"target": nested})
    leaf = expression["children"][0]
    for bad in ("arbitrary", "PRED-A"):
        leaf["predicateId"] = bad
        with pytest.raises(ValidationError):
            validator.validate({"target": expression})


def test_template_controls_reuse_registry_and_parameter_domains(catalog):
    schema = family_expression_schema_v1()["$defs"]["expression"]
    predicates = {branch["properties"]["kind"]["const"]: branch
                  for branch in schema["oneOf"] if branch["x-discriminator"] == "kind"}
    assert set(predicates) == {row["kind"] for row in ScenarioPredicateRegistryV1().catalog()["definitions"]} | NATIVE_PLAY_KINDS
    assert not predicates["tactic.verified"]["x-constructionSupported"]
    assert predicates["deal.remaining_suit_count"]["x-constructionSupported"]
    assert {kind for kind, row in predicates.items()
            if kind.startswith('state.') and row['x-constructionSupported']} == {
                'state.legal_action_count', 'state.current_raw_points'}
    assert schema["x-expressionLimits"]["maximumDepth"] == 8
    suit = predicates["deal.remaining_suit_count"]["properties"]["arguments"]["properties"]["suit"]
    assert suit["x-familyParameterOptions"] == ["suit"]
    assert suit["x-messageArKey"]
    for reference in ({"$parameter": "unknown"}, {"$parameter": "suit", "extra": True}):
        with pytest.raises(ValidationError):
            Draft202012Validator(suit).validate(reference)
    # Metadata generation cannot change frozen catalogs or their canonical hashes.
    assert ResearchFamilyCatalogV1.build(catalog.to_dict()) == catalog


def test_family_resolver_preserves_legacy_definitions_and_uses_registered_native_metadata():
    from baloot_platform.research.family_catalog import _FamilyPredicateRegistryV1
    from baloot_platform.research.scenario_registry_v2 import flagship_predicate_definitions_v2

    legacy = ScenarioPredicateRegistryV1()
    resolver = _FamilyPredicateRegistryV1()
    rows = resolver.catalog_definitions()
    historical = legacy.catalog()["definitions"]
    assert rows[:len(historical)] == historical
    registered = {row.kind: row for row in ScenarioPredicateLoweringRegistryV2().definitions}
    assert rows[len(historical):] == [registered[kind].to_dict() for kind in sorted(NATIVE_PLAY_KINDS)]
    for row in historical:
        example = ScenarioPredicateV1.build({"predicateId": "PRED-LEGACY", **row["examples"][0]})
        assert resolver.require(example) == legacy.require(example)
    schema = family_expression_schema_v1()["$defs"]["expression"]
    branches = {row["properties"]["kind"]["const"]: row for row in schema["oneOf"]
                if row["x-discriminator"] == "kind"}
    raw_definitions = {row.kind: row for row in flagship_predicate_definitions_v2()}
    for kind in NATIVE_PLAY_KINDS:
        definition = registered[kind]
        assert definition.sha256 != raw_definitions[kind].sha256
        assert definition.cp_sat_support.search_status == "exact"
        example = ScenarioPredicateV1.build({"predicateId": "PRED-NATIVE", **definition.examples[0],
                                            "hardness": "hard"})
        assert resolver.require(example) == definition
        assert branches[kind]["properties"]["hardness"]["enum"] == list(definition.allowed_hardness)
        assert branches[kind]["x-constructionSupported"]
        assert legacy.definition(kind) is None
    selector = branches["play.action_matches"]["properties"]["arguments"]["properties"]["selector"]["properties"]
    assert selector["suits"]["items"]["x-familyParameterOptions"] == ["suit"]
    assert selector["ranks"]["items"]["x-familyParameterOptions"] == ["rank"]
    assert selector["owner"]["properties"]["seat"]["x-familyParameterOptions"] == ["seat"]
    assert selector["cards"]["items"]["x-familyParameterOptions"] == ["playIndex"]


def test_native_nested_selector_bindings_roundtrip_without_mutating_templates(catalog):
    selector = {"suits": [{"$parameter": "suit"}], "ranks": [{"$parameter": "rank"}],
                "owner": {"relation": "seat", "seat": {"$parameter": "seat"}}}
    expression = {"op": "all", "children": [
        native_predicate("play.action_matches", {"trickIndex": 6, "position": 0,
            "actingSeat": {"$parameter": "seat"}, "selector": selector}, "PRED-ACTION"),
        {"op": "any", "children": [
            native_predicate("play.trick_winner_matches", {"trickIndex": 6,
                "seat": {"$parameter": "seat"}, "selector": selector}, "PRED-WINNER"),
            {"op": "count", "minimum": 0, "maximum": 1, "children": [
                {"op": "not", "children": [native_predicate("play.trick_card_count",
                    {"trickIndex": 6, "selector": selector, "minimum": 1, "maximum": 2}, "PRED-CARD-COUNT")]}]},
        ]},
    ]}
    raw = native_catalog(catalog, expression, enabled=True)
    Draft202012Validator(family_catalog_schema_v1()).validate(raw)
    changed = ResearchFamilyCatalogV1.build(raw)
    family = changed.families[0]
    original = family.instantiate(bindings(family))
    selected = {**bindings(family), "seat": "east", "suit": "hearts", "rank": "ace"}
    instance = family.instantiate(selected)
    validate_family_construction_expression_v1(instance)
    for predicate in instance.predicates():
        assert predicate.arguments["selector"] == {
            "suits": ["hearts"], "ranks": ["ace"], "owner": {"relation": "seat", "seat": "east"}}
    assert instance.predicates()[0].arguments["actingSeat"] == "east"
    assert instance.predicates()[1].arguments["seat"] == "east"
    assert instance.sha256 != original.sha256
    assert family.instantiate(selected).sha256 == instance.sha256
    assert family.to_dict()["template"]["expression"] == expression
    assert ResearchFamilyCatalogV1.build(changed.to_dict()) == changed


@pytest.mark.parametrize("kind,changes", [
    ("play.action_matches", {"selector": {}}),
    ("play.action_matches", {"selector": {"ekka": "any"}}),
    ("play.action_matches", {"selector": {"ranks": ["ace", "ace"]}}),
    ("play.action_matches", {"selector": {"suits": ["unknown"]}}),
    ("play.action_matches", {"selector": {"cards": [True]}}),
    ("play.action_matches", {"selector": {"cards": [32]}}),
    ("play.action_matches", {"selector": {"owner": {"relation": "seat"}}}),
    ("play.action_matches", {"selector": {"owner": {"relation": "actor", "seat": "north"}}}),
    ("play.action_matches", {"selector": {"owner": {"relation": "team", "team": "unknown"}}}),
    ("play.action_matches", {"position": 4}),
    ("play.action_matches", {"trickIndex": {"$parameter": "playIndex"}}),
    ("play.trick_winner_matches", {"seat": "north", "team": "north_south"}),
    ("play.trick_winner_matches", {"selector": {"ranks": []}}),
    ("play.trick_winner_matches", {"trickIndex": 8}),
    ("play.trick_card_count", {"minimum": 3, "maximum": 2}),
    ("play.trick_card_count", {"minimum": True}),
    ("play.trick_card_count", {"maximum": 5}),
    ("play.trick_card_count", {"unexpected": True}),
])
def test_native_arguments_fail_closed_in_catalog_and_instantiation(catalog, kind, changes):
    arguments = {"trickIndex": 6, "selector": {"ranks": ["ace"]}}
    if kind == "play.action_matches":
        arguments["position"] = 0
    if kind == "play.trick_card_count":
        arguments.update(minimum=1, maximum=2)
    expression = native_predicate(kind, {**arguments, **changes})
    raw = native_catalog(catalog, expression)
    with pytest.raises((ValueError, TypeError)):
        ResearchFamilyCatalogV1.build(raw)
    family = replace(catalog.families[0], _json=json.dumps(raw["families"][0]))
    with pytest.raises((ValueError, TypeError)):
        family.instantiate(bindings(family))


@pytest.mark.parametrize("kind", sorted(NATIVE_PLAY_KINDS))
@pytest.mark.parametrize("field,value", [("version", 2), ("hardness", "prefer")])
def test_native_identity_and_hardness_use_the_same_closed_resolver(catalog, kind, field, value):
    definition = ScenarioPredicateLoweringRegistryV2().definition(kind)
    expression = {"predicateId": "PRED-NATIVE", **definition.examples[0], field: value}
    with pytest.raises(ValueError):
        ResearchFamilyCatalogV1.build(native_catalog(catalog, expression))
    family = replace(catalog.families[0], _json=json.dumps(
        native_catalog(catalog, expression)["families"][0]))
    with pytest.raises(ValueError, match="unsupported scenario predicate|does not allow hardness"):
        family.instantiate(bindings(family))


def test_native_family_admission_does_not_expand_to_new_proof_kinds(catalog):
    expression = native_predicate("proof.winning_root_action_matches", {"selector": {"ranks": ["ace"]}})
    with pytest.raises(ValueError):
        ResearchFamilyCatalogV1.build(native_catalog(catalog, expression))
    family = replace(catalog.families[0], _json=json.dumps(
        native_catalog(catalog, expression)["families"][0]))
    with pytest.raises(ValueError, match="unsupported scenario predicate"):
        family.instantiate(bindings(family))


@pytest.mark.parametrize("kind", sorted(NATIVE_PLAY_KINDS))
def test_native_safe_complement_metadata_applies_below_nested_groups(catalog, kind):
    leaf = {"predicateId": "PRED-NATIVE", **ScenarioPredicateLoweringRegistryV2().definition(kind).examples[0],
            "hardness": "hard"}
    expression = {"op": "not", "children": [{"op": "any", "children": [
        {"op": "count", "minimum": 0, "maximum": 1, "children": [leaf]}]}]}
    draft = ResearchFamilyCatalogV1.build(native_catalog(catalog, expression)).families[0]
    instance = draft.instantiate(bindings(draft))
    if kind == "play.trick_card_count":
        validate_family_construction_expression_v1(instance)
        ResearchFamilyCatalogV1.build(native_catalog(catalog, expression, enabled=True))
    else:
        with pytest.raises(ValueError, match="safe complement"):
            validate_family_construction_expression_v1(instance)
        with pytest.raises(ValueError, match="safe complement"):
            ResearchFamilyCatalogV1.build(native_catalog(catalog, expression, enabled=True))


@pytest.mark.parametrize("kind", sorted(NATIVE_PLAY_KINDS))
def test_native_report_predicates_remain_draft_only(catalog, kind):
    leaf = {"predicateId": "PRED-NATIVE", **ScenarioPredicateLoweringRegistryV2().definition(kind).examples[0],
            "hardness": "report"}
    draft = ResearchFamilyCatalogV1.build(native_catalog(catalog, leaf)).families[0]
    assert draft.instantiate(bindings(draft)).predicate.hardness == "report"
    with pytest.raises(ValueError, match="hard predicates only"):
        ResearchFamilyCatalogV1.build(native_catalog(catalog, leaf, enabled=True))


def test_tag_query_filters_before_allocation_with_and_between_axes(catalog):
    query = {"impact": ["round-flip"], "technique": ["guard/protect"]}
    assert [family.family_id for family in catalog.matching_families(query)] == ["stopper_guard"]
    assert catalog.allocation_shares(query) == {"stopper_guard": 1.0}
    assert catalog.allocation_shares({"technique": ["baloot-timing"], "mode": ["sun"]}) == {}
    assert not catalog.matching_families({"technique": ["endplay"], "level": ["beginner"]})
    shares = catalog.allocation_shares({"impact": ["round-flip", "kaboot-swing"],
                                        "technique": ["guard/protect"]})
    assert shares == {"stopper_guard": 0.5, "kaboot_prevention": 0.5}
    assert sum(catalog.allocation_shares({}).values()) == pytest.approx(1)
    assert all(not family.enabled for family in catalog.matching_families(query))


@pytest.mark.parametrize("query", [
    {"impact": []}, {"impact": ["round-flip", "round-flip"]},
    {"impact": "round-flip"}, {"impact": [True]}, {"impact": ["unknown"]},
    {"unknown": ["round-flip"]}, {"operator": "or"}, {"mode": ["Sun"]},
    {"trickBand": ["trick-4"]}, {"level": None}, [], None,
])
def test_query_validation_is_closed(catalog, query):
    with pytest.raises(ValueError):
        catalog.allocation_shares(query)


def test_query_normalizes_choice_order_and_detaches_input():
    raw = {"mode": ["hokum", "sun"], "impact": ["points-only", "round-flip"]}
    query = TagQueryV1.build(raw)
    assert query.to_dict() == {"impact": ["round-flip", "points-only"], "mode": ["sun", "hokum"]}
    raw["mode"].clear()
    assert query.to_dict()["mode"] == ["sun", "hokum"]
    Draft202012Validator.check_schema(tag_query_schema_v1())
    assert set(TAG_AXES) == {"impact", "technique", "level", "mode", "trickBand"}


def test_audited_yield_allocation_rewards_lessons_per_cpu_and_preserves_floors(catalog):
    samples = {"catalogHash": catalog.sha256, "families": {
        "stopper_guard": {"newLessons": 2, "cpuSeconds": 10},
        "kaboot_prevention": {"newLessons": 20, "cpuSeconds": 1000}}}
    query = {"technique": ["guard/protect"]}
    shares = catalog.allocation_shares(query, audited_yield=samples)
    assert shares["stopper_guard"] > shares["kaboot_prevention"] >= 0.05
    assert sum(shares.values()) == pytest.approx(1)
    samples["families"]["kaboot_prevention"]["newLessons"] = 0
    assert catalog.allocation_shares(query, audited_yield=samples)["kaboot_prevention"] == 0.05
    del samples["families"]["kaboot_prevention"]
    assert catalog.allocation_shares(query, audited_yield=samples) == {"stopper_guard": 0.5, "kaboot_prevention": 0.5}
    samples["families"]["stopper_guard"]["newLessons"] = 0
    assert catalog.allocation_shares(query, audited_yield=samples) == catalog.allocation_shares(query)


@pytest.mark.parametrize("sample", [
    {"newLessons": True, "cpuSeconds": 1}, {"newLessons": -1, "cpuSeconds": 1},
    {"newLessons": 1, "cpuSeconds": 0}, {"newLessons": 1, "cpuSeconds": float('nan')},
    {"newLessons": 1, "cpuSeconds": 1, "gatePasses": 100},
])
def test_audited_yield_rejects_invalid_samples(catalog, sample):
    with pytest.raises(ValueError, match="invalid audited"):
        catalog.allocation_shares({}, audited_yield={"catalogHash": catalog.sha256,
            "families": {"stopper_guard": sample}})


def test_audited_yield_cannot_cross_catalog_revision(catalog):
    with pytest.raises(ValueError, match="catalog hash"):
        catalog.allocation_shares({}, audited_yield={"catalogHash": "old", "families": {}})


@pytest.mark.parametrize("axis,values", [
    ("impact", []), ("impact", ["unknown"]), ("mode", ["sun"]),
    ("trickBand", ["trick-5"]), ("technique", ["guard"]), ("unknown", ["beginner"]),
])
def test_capability_tags_are_closed_and_consistent(catalog, axis, values):
    raw = authored(catalog)
    raw["families"][0]["capabilityTags"][axis] = values
    with pytest.raises(ValueError):
        ResearchFamilyCatalogV1.build(raw)


def test_capability_edits_change_hash_and_custom_family_is_queryable(catalog, tmp_path):
    raw = authored(catalog)
    family = copy.deepcopy(raw["families"][2])
    family.update(id="custom_guard", minimumAllocation=0, weight=3)
    family["capabilityTags"]["level"] = ["beginner"]
    family["parameters"]["playIndex"] = {"minimum": 20, "maximum": 23}
    family["capabilityTags"]["trickBand"] = ["trick-6"]
    raw["families"].append(family)
    changed = ResearchFamilyCatalogV1.build(raw)
    assert changed.sha256 != catalog.sha256
    query = {"impact": ["round-flip"], "technique": ["guard/protect"], "trickBand": ["trick-6"]}
    shares = changed.allocation_shares(query)
    assert set(shares) == {"stopper_guard", "custom_guard"}
    assert shares["custom_guard"] > shares["stopper_guard"] >= 0.05
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(changed.to_dict()), encoding="utf-8")
    assert ResearchFamilyCatalogV1.load(path) == changed
    stale = catalog.to_dict()
    stale["families"][0]["capabilityTags"]["level"] = ["advanced"]
    with pytest.raises(ValueError, match="checksum"):
        ResearchFamilyCatalogV1.build(stale)
