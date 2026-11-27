from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from service_test_client import OperatorClient as TestClient
from pydantic import ValidationError

from baloot_platform.configuration import EffectiveRuntimeConfigV2
from baloot_platform.core.cards import Card
from baloot_platform.research import (
    ResearchConfigurationAuthorityV1,
    ResearchExecutionContextFactoryV1,
    ResearchRecipeV1,
    ResearchRecipeV2,
    ResearchRecipeV3,
    SQLiteResearchStoreV1,
    build_legacy_research_recipe_for_read,
    build_research_recipe,
)
from baloot_platform.research.contracts import (
    RESEARCH_STRATEGIC_AUCTION_DEEP_BID_SAMPLES_V3,
    RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1,
)
from baloot_platform.research.recipe_conversion import (
    preview_v3_to_v4_conversion,
)
from baloot_platform.research.recipe_v4 import ResearchRecipeV4
from baloot_platform.research.scenario_contracts import canonical_sha256
from baloot_platform.research.service import ResearchFoundryServiceV1
from baloot_platform.service.api import (
    ResearchConfigurationAuthorityResponse,
    ResearchRecipeRequest,
    ResearchRecipeResponse,
    ResearchRecipeV2Response,
    ResearchRecipeV3CreateRequest,
    ResearchRecipeV3Response,
    ResearchRecipeV4CreateRequest,
    ResearchStrategicAuctionRequest,
    ResearchStrategicAuctionV3Request,
    app,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _effective_runtime_snapshot() -> EffectiveRuntimeConfigV2:
    families = (
        "rules",
        "conventions",
        "belief",
        "agents",
        "analysis",
        "challenge",
        "product",
        "match_value",
        "performance",
    )
    revisions = {
        family: {
            "revisionId": f"CPR-{family.upper()}-0001",
            "profileId": f"{family}-profile",
            "version": "1.0.0",
            "payloadSha256": _hash(f"payload:{family}"),
        }
        for family in families
    }
    return EffectiveRuntimeConfigV2.from_resolved(
        {
            "bundle": {
                "revisionId": "CBR-RESEARCH-V3-0001",
                "bundleId": "balootx-research",
                "version": "21.2.0",
                "mode": "full",
                "contentSha256": _hash("row-bundle"),
            },
            "profiles": {
                family: {"profile": {"id": family, "version": "1.0.0"}}
                for family in families
            },
            "profileRevisions": revisions,
            "resolutionTrace": [
                {"family": family, "revisionId": revisions[family]["revisionId"]}
                for family in families
            ],
            "effectiveHash": _hash("effective-resolution"),
            "invalidationEpoch": 3,
        },
        plugin_identities=(
            {
                "id": "ortools-research",
                "version": "1.0.0",
                "artifactSha256": _hash("ortools-plugin"),
            },
        ),
        model_identities=(
            {
                "id": "balanced-agent",
                "version": "1.0.0",
                "configSha256": _hash("balanced-agent-config"),
            },
        ),
    )


def test_selected_context_builds_once_and_reuses_the_frozen_identity(monkeypatch):
    effective = _effective_runtime_snapshot()
    resolved = ResearchExecutionContextFactoryV1._resolved_from_effective(effective)
    factory = object.__new__(ResearchExecutionContextFactoryV1)
    factory.cache_size = 2
    factory._cache = OrderedDict()
    factory._cache_hits = factory._cache_misses = factory._cache_evictions = 0
    factory._lock = threading.RLock()
    resolutions = []
    builds = []
    checks = []
    bundle = SimpleNamespace(sha256=_hash("semantic-bundle"))
    backend = SimpleNamespace(descriptor=SimpleNamespace(to_dict=lambda: {
        "id": "native", "version": "1.0.0",
    }))
    factory.root = "."
    factory.host_ceiling = SimpleNamespace(validate_bundle=lambda value: checks.append(value))
    monkeypatch.setattr("baloot_platform.research.execution_context.profile_bundle_from_resolved",
                        lambda value, *, root: bundle)
    monkeypatch.setattr(factory, "_search_backend", lambda value: backend)

    def resolve(**kwargs):
        resolutions.append(kwargs)
        return copy.deepcopy(resolved)

    def build(value, *, effective_runtime_config_hash):
        builds.append(effective_runtime_config_hash)
        frozen = EffectiveRuntimeConfigV2.from_resolved(
            value, plugin_identities=effective.to_dict()["pluginIdentities"],
            model_identities=effective.to_dict()["modelIdentities"],
        )
        return SimpleNamespace(
            effective_runtime_config_hash=effective_runtime_config_hash or frozen.content_sha256,
            source_effective_hash=value["effectiveHash"],
            bundle_revision_id=value["bundle"]["revisionId"],
            plugin_identities=tuple(effective.to_dict()["pluginIdentities"]),
            model_identities=tuple(effective.to_dict()["modelIdentities"]),
            bundle=bundle, exact_search_backend=backend,
        )

    factory.configuration_service = SimpleNamespace(resolve=resolve)
    monkeypatch.setattr(factory, "_build_context", build)
    first_effective, first = factory.resolve_bundle_revision(resolved["bundle"]["revisionId"])
    assert first_effective.to_dict() == effective.to_dict()
    assert len(builds) == 1
    second_effective, second = factory.resolve_bundle_revision(resolved["bundle"]["revisionId"])
    assert second is first
    assert second_effective.to_dict() == first_effective.to_dict()
    assert len(builds) == 1
    assert len(resolutions) == 2  # Publication/retirement is checked on every lookup.
    assert checks == [bundle]
    resolved["invalidationEpoch"] += 1
    third_effective, third = factory.resolve_bundle_revision(resolved["bundle"]["revisionId"])
    assert third is not first
    assert third_effective.content_sha256 != first_effective.content_sha256
    assert len(builds) == 2

    def unavailable(value):
        raise RuntimeError("required native backend unavailable")

    monkeypatch.setattr(factory, "_search_backend", unavailable)
    with pytest.raises(RuntimeError, match="required native backend unavailable"):
        factory.resolve_bundle_revision(resolved["bundle"]["revisionId"])


@pytest.mark.parametrize("field", [
    "effective_runtime_config_hash", "source_effective_hash", "bundle_revision_id",
])
def test_prebuilt_context_rejects_cross_snapshot_identity(field):
    effective = _effective_runtime_snapshot()
    resolved = ResearchExecutionContextFactoryV1._resolved_from_effective(effective)
    built = SimpleNamespace(
        effective_runtime_config_hash=effective.content_sha256,
        source_effective_hash=resolved["effectiveHash"],
        bundle_revision_id=resolved["bundle"]["revisionId"],
    )
    setattr(built, field, "wrong-owner")
    factory = object.__new__(ResearchExecutionContextFactoryV1)
    with pytest.raises(ValueError, match="prebuilt context identity mismatch"):
        factory._cached_context(resolved, effective, built=built)


def test_research_execution_context_cache_metrics_are_bounded_and_thread_safe(
    monkeypatch: pytest.MonkeyPatch,
):
    def cache_factory(capacity: int) -> ResearchExecutionContextFactoryV1:
        factory = object.__new__(ResearchExecutionContextFactoryV1)
        factory.cache_size = capacity
        factory._cache = OrderedDict()
        factory._cache_hits = 0
        factory._cache_misses = 0
        factory._cache_evictions = 0
        factory._lock = threading.RLock()
        monkeypatch.setattr(
            factory,
            "_build_context",
            lambda _resolved, *, effective_runtime_config_hash: {
                "effectiveRuntimeConfigHash": effective_runtime_config_hash
            },
        )
        return factory

    def resolve(factory: ResearchExecutionContextFactoryV1, key: str):
        return factory._cached_context({}, SimpleNamespace(content_sha256=key))

    factory = cache_factory(2)
    assert resolve(factory, "a") == {"effectiveRuntimeConfigHash": "a"}
    assert resolve(factory, "a") == {"effectiveRuntimeConfigHash": "a"}
    assert factory.cache_metrics_snapshot() == {
        "scope": "research_execution_context",
        "hits": 1,
        "misses": 1,
        "evictions": 0,
        "entries": 1,
        "capacity": 2,
    }
    resolve(factory, "b")
    resolve(factory, "c")
    assert factory.cache_metrics_snapshot() == {
        "scope": "research_execution_context",
        "hits": 1,
        "misses": 3,
        "evictions": 1,
        "entries": 2,
        "capacity": 2,
    }

    parallel = cache_factory(2)
    keys = tuple("xy"[index % 2] for index in range(40))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(lambda key: resolve(parallel, key), keys))
    assert {item["effectiveRuntimeConfigHash"] for item in results} == {"x", "y"}
    snapshot = parallel.cache_metrics_snapshot()
    assert snapshot["hits"] + snapshot["misses"] == len(keys)
    assert snapshot["entries"] == 2
    assert snapshot["evictions"] == 0
    assert tuple(snapshot) == (
        "scope",
        "hits",
        "misses",
        "evictions",
        "entries",
        "capacity",
    )


def _configuration_authority() -> dict:
    effective_runtime_config = _effective_runtime_snapshot()
    snapshot = effective_runtime_config.to_dict()
    snapshot_bytes = json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact_hash = hashlib.sha256(snapshot_bytes).hexdigest()
    components = snapshot["componentRevisions"]

    def identity(
        role: str,
        identity_id: str,
        digest_label: str,
        source_families: tuple[str, ...],
    ) -> dict:
        return {
            "kind": role,
            "id": identity_id,
            "version": "1.0.0",
            "sha256": _hash(digest_label),
            "sourceRevisions": {
                family: components[family]["revisionId"]
                for family in sorted(source_families)
            },
        }

    snapshot_artifact = {
            "uri": f"cas://sha256/{artifact_hash}",
            "sha256": artifact_hash,
            "bytes": len(snapshot_bytes),
            "mediaType": (
                "application/vnd.balootx.effective-runtime-config-v2+json"
            ),
            "name": "effective-runtime-config-v2.json",
    }
    identities = {
        "domain": identity(
            "domain",
            "baloot-domain-v3",
            "domain",
            ("rules", "conventions", "product"),
        ),
        "rules": identity("rules", "rules-v3", "rules", ("rules",)),
        "convention": identity(
            "convention",
            "conservative-signals",
            "convention",
            ("conventions",),
        ),
        "belief": identity(
            "belief", "observer-safe-belief", "belief", ("belief",)
        ),
        "agent": identity(
            "agent",
            "balanced-agent@1.0.0",
            "agent",
            ("agents", "challenge"),
        ),
        "analyzer": identity(
            "analyzer",
            "cascaded-analysis",
            "analysis",
            (
                "challenge",
                "analysis",
                "belief",
                "match_value",
                "performance",
                "rules",
                "conventions",
            ),
        ),
        "score": identity(
            "score", "balanced-editorial", "score", ("challenge",)
        ),
        "decisionScore": identity(
            "decisionScore",
            "decision-score-balanced",
            "decision-score",
            ("challenge",),
        ),
    }
    return ResearchConfigurationAuthorityV1.freeze(
        effective_runtime_config=effective_runtime_config,
        snapshot_artifact=snapshot_artifact,
        semantic_bundle_hash=_hash("semantic-bundle"),
        identities=identities,
    ).to_dict()


def _recipe_v1(*, schema_version: int = 1) -> dict:
    configuration = _configuration_authority()
    identities = configuration["identities"]
    value = {
        "schemaVersion": schema_version,
        "titleAr": "ورقة صغيرة تفتح مكسب كبير",
        "output": {"kind": "laffa_collection", "resultCount": 2},
        "profiles": {
            "domainHash": identities["domain"]["sha256"],
            "bundleHash": configuration["semanticBundleHash"],
            "rulesHash": identities["rules"]["sha256"],
            "conventionHash": identities["convention"]["sha256"],
            "agentRelease": identities["agent"]["id"],
            "analysisProfileHash": identities["analyzer"]["sha256"],
            "scoreProfileHash": identities["score"]["sha256"],
        },
        "situation": {
            "dealer": "east",
            "buyer": "south",
            "contracts": ["sun"],
            "matchScore": [120, 118],
            "requiredCards": {"south": ["SA"]},
            "forbiddenCards": {},
            "projects": [],
            "escalations": [],
        },
        "tactics": [
            {
                "kind": "small_card_big_gain",
                "mode": "qualification_gate",
                "minimumScore": 70,
                "weight": 1,
                "perspectives": ["south"],
            }
        ],
        "qualification": {
            "minimumTruth": "robust_best_known",
            "minimumConfidence": 0.8,
            "minimumSurroundingQuality": 75,
            "requireObserverSafety": True,
            "requireAllDecisionAudit": True,
            "maximumRegret": 0,
        },
        "resources": {
            "generationLane": "ortools",
            "seedStart": 100,
            "seedCount": 20,
            "seedsPerShard": 5,
            "candidatesPerSeed": 2,
            "solverTimeoutSeconds": 10,
            "maximumConflicts": 250_000,
            "analysisMode": "cascaded",
            "deepVerificationPolicy": "required_gates",
        },
        "status": "published",
    }
    if schema_version >= 2:
        value.update(
            {
                "diversity": {
                    "schemaVersion": 1,
                    "cellDimensions": ["contract", "acting_seat", "source"],
                    "topKPerCell": 2,
                    "rejectedAuditModulus": 50,
                },
                "renderPolicy": {"mode": "on_demand"},
            }
        )
    if schema_version == 3:
        value["configuration"] = configuration
    return value


def _v3_create_payload() -> dict:
    payload = _recipe_v1(schema_version=3)
    payload.pop("profiles")
    payload["configuration"] = {
        "schemaVersion": 1,
        "bundleRevisionId": payload["configuration"]["bundleRevisionId"],
    }
    return payload


def _v4_operator_completion() -> dict[str, dict]:
    return {
        "decision_point": {
            "kind": "play",
            "selector": {"actingSeat": "south", "remainingCards": 8},
        },
        "outcome": {
            "predicateId": "PRED-CONVERT-OUTCOME",
            "kind": "outcome.team_round",
            "version": 1,
            "hardness": "hard",
            "arguments": {"team": "north_south", "status": "won"},
        },
        "proof": {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "claim": "winning_root_action_count",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": 1,
            "lineUniverse": "all_legal",
            "budgetProfile": "late_game_exact_v1",
        },
    }


def _strategic_auction_profile(*, requested_samples: int) -> dict:
    return {
        "schemaVersion": 1,
        "requestedSamples": requested_samples,
        "integerScale": 1_000,
        "maximumConflicts": 50_000,
        "maximumSelectionSeconds": 5.0,
        "objective": "expected_round_margin",
        "tieBreak": RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1,
        "requireOptimalForQualification": True,
    }


def _legacy_strategic_auction_profile(*, requested_samples: int) -> dict:
    return {
        **_strategic_auction_profile(requested_samples=requested_samples),
        "tieBreak": "win_probability_then_best_share_then_action_id",
    }


def test_research_recipe_v3_round_trips_with_frozen_configuration_authority():
    recipe = ResearchRecipeV3.build(_recipe_v1(schema_version=3))

    assert isinstance(recipe, ResearchRecipeV2)
    assert recipe.configuration["bundleRevisionId"] == "CBR-RESEARCH-V3-0001"
    assert recipe.configuration["effectiveRuntimeConfigHash"] == (
        _effective_runtime_snapshot().content_sha256
    )
    assert build_research_recipe(recipe.to_dict()) == recipe
    assert ResearchConfigurationAuthorityV1.build(
        recipe.configuration
    ).to_dict() == recipe.configuration


def test_v3_to_v4_preview_requires_each_operator_owned_semantic_section():
    source = ResearchRecipeV3.build(_recipe_v1(schema_version=3)).to_dict()
    completion = _v4_operator_completion()

    for missing_key, expected_path in (
        (None, None),
        ("decision_point", "scenario.decisionPoint"),
        ("outcome", "scenario.outcome"),
        ("proof", "proof"),
    ):
        supplied = {} if missing_key is None else copy.deepcopy(completion)
        if missing_key is not None:
            supplied.pop(missing_key)
        preview = preview_v3_to_v4_conversion(source, **supplied)

        assert preview["targetRecipeCode"] is None
        assert preview["targetRecipeHash"] is None
        assert preview["buildable"] is False
        assert preview["requiredOperatorFields"] == sorted(
            preview["requiredOperatorFields"]
        )
        if expected_path is None:
            assert preview["requiredOperatorFields"] == [
                "proof",
                "scenario.decisionPoint",
                "scenario.outcome",
            ]
        else:
            assert preview["requiredOperatorFields"] == [expected_path]
        assert all(
            diagnostic["code"] == "operator_input_required"
            for diagnostic in preview["buildDiagnostics"]
        )


def test_v3_to_v4_preview_is_deterministic_non_mutating_and_rebuildable(
    tmp_path,
):
    source = ResearchRecipeV3.build(_recipe_v1(schema_version=3)).to_dict()
    original_source = copy.deepcopy(source)
    store = SQLiteResearchStoreV1(tmp_path / "research.sqlite3")
    store.put_recipe(source)
    stored_before = store.get_recipe(source["code"])
    list_before = store.list_recipes()

    completion = _v4_operator_completion()
    first = preview_v3_to_v4_conversion(source, **completion)
    second = preview_v3_to_v4_conversion(source, **completion)
    service = object.__new__(ResearchFoundryServiceV1)
    service.store = store
    through_service = service.preview_recipe_v4_conversion(
        source["code"], **completion
    )

    assert first == second == through_service
    assert source == original_source
    assert store.get_recipe(source["code"]) == stored_before
    assert store.list_recipes() == list_before
    assert first["requiredOperatorFields"] == []
    assert first["buildDiagnostics"] == []
    assert first["buildable"] is True
    assert first["targetRecipeCode"] != source["code"]
    assert first["targetRecipeHash"] != source["sha256"]
    assert store.get_recipe(first["targetRecipeCode"]) is None
    assert first["sha256"] == canonical_sha256(
        {key: value for key, value in first.items() if key != "sha256"}
    )
    assert first["unmappedFields"] == sorted(
        first["unmappedFields"],
        key=lambda item: (
            item["fieldPath"],
            item["reasonCode"],
            canonical_sha256(item["sourceValue"]),
        ),
    )

    proposed = copy.deepcopy(first["proposedRecipe"])
    ResearchRecipeV4CreateRequest.model_validate(proposed)
    assert proposed["configuration"] == {
        "schemaVersion": 1,
        "bundleRevisionId": source["configuration"]["bundleRevisionId"],
    }
    assert proposed["output"] == source["output"]
    assert proposed["renderPolicy"] == {"mode": "on_demand", "videoCount": 1}
    assert proposed["scenario"]["decisionPoint"] == completion["decision_point"]
    assert {
        key: proposed["scenario"]["outcome"][key]
        for key in completion["outcome"]
    } == completion["outcome"]
    assert proposed["proof"] == completion["proof"] | {
        "successThreshold": 1.0,
        "sha256": proposed["proof"]["sha256"],
    }
    assert proposed["resources"]["proofBudgetProfile"] == (
        completion["proof"]["budgetProfile"]
    )
    assert proposed["resources"]["generationLane"] == "ortools"
    assert proposed["resources"]["generationPolicy"] == "ortools_cp_sat_v3"
    assert not {
        "candidatesPerSeed",
        "maximumConflicts",
        "analysisMode",
    }.intersection(proposed["resources"])
    compatibility_diagnostics = {
        item["fieldPath"]: item
        for item in first["unmappedFields"]
        if item["fieldPath"].startswith("resources.")
    }
    assert compatibility_diagnostics["resources.candidatesPerSeed"] == {
        "fieldPath": "resources.candidatesPerSeed",
        "sourceValue": source["resources"]["candidatesPerSeed"],
        "reasonCode": "v3_candidates_per_seed_has_no_ortools_v4_semantics",
        "requiresOperatorReview": False,
    }
    assert compatibility_diagnostics["resources.maximumConflicts"] == {
        "fieldPath": "resources.maximumConflicts",
        "sourceValue": source["resources"]["maximumConflicts"],
        "reasonCode": "v3_solver_limit_is_replaced_by_frozen_bundle_budget",
        "requiresOperatorReview": False,
    }
    assert compatibility_diagnostics["resources.analysisMode"] == {
        "fieldPath": "resources.analysisMode",
        "sourceValue": source["resources"]["analysisMode"],
        "reasonCode": (
            "legacy_analyzer_identity_is_frozen_bundle_compatibility_only"
        ),
        "requiresOperatorReview": False,
    }
    assert "sha256" not in proposed["diversity"]

    proposed["configuration"] = copy.deepcopy(source["configuration"])
    rebuilt = ResearchRecipeV4.build(proposed)
    assert rebuilt.capability_plan.valid is True
    assert rebuilt.code == first["targetRecipeCode"]
    assert rebuilt.sha256 == first["targetRecipeHash"]

    def predicates(expression: dict) -> list[dict]:
        if "predicateId" in expression:
            return [expression]
        return [
            predicate
            for child in expression["children"]
            for predicate in predicates(child)
        ]

    rows = predicates(proposed["scenario"]["constraints"])
    by_kind: dict[str, list[dict]] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row)
    assert by_kind["auction.dealer"][0]["arguments"] == {"seat": "east"}
    assert by_kind["auction.buyer"][0]["arguments"] == {"seat": "south"}
    assert by_kind["auction.contract"][0]["arguments"] == {"contract": "sun"}
    assert by_kind["deal.card_owner"][0]["arguments"] == {
        "card": Card.parse("SA").id,
        "seat": "south",
        "handStage": "post_purchase",
    }
    assert sorted(
        (row["arguments"] for row in by_kind["state.match_score"]),
        key=lambda arguments: arguments["team"],
    ) == [
        {"team": "north_south", "minimum": 120, "maximum": 120},
        {"team": "west_east", "minimum": 118, "maximum": 118},
    ]
    assert by_kind["tactic.verified"][0]["hardness"] == "hard"
    assert by_kind["tactic.verified"][0]["arguments"] == {
        "kind": "small_card_big_gain",
        "actingSeat": "south",
        "minimumScore": 70.0,
        "minimumEvidence": 1,
    }


def test_v3_to_v4_preview_accounts_for_every_source_leaf_and_projects():
    payload = _recipe_v1(schema_version=3)
    payload["situation"]["projects"] = [
        {
            "seat": "south",
            "contract": "sun",
            "kind": "hundred_sequence",
            "value": 100,
            "highRank": "ace",
            "mode": "hard",
        }
    ]
    source = ResearchRecipeV3.build(payload).to_dict()
    preview = preview_v3_to_v4_conversion(source)
    mapped_prefixes = [row["sourcePath"] for row in preview["mappedFields"]]
    unmapped_paths = {row["fieldPath"] for row in preview["unmappedFields"]}

    def leaf_paths(value, path=""):
        if isinstance(value, dict):
            if not value:
                return [path]
            return [
                leaf
                for key in sorted(value)
                for leaf in leaf_paths(
                    value[key], f"{path}.{key}" if path else str(key)
                )
            ]
        if isinstance(value, list):
            if not value:
                return [path]
            return [
                leaf
                for index, child in enumerate(value)
                for leaf in leaf_paths(child, f"{path}[{index}]")
            ]
        return [path]

    for leaf_path in leaf_paths(source):
        coverage = sum(
            leaf_path == prefix
            or leaf_path.startswith(prefix + ".")
            or leaf_path.startswith(prefix + "[")
            for prefix in mapped_prefixes
        ) + (leaf_path in unmapped_paths)
        assert coverage == 1, leaf_path

    project_rows = [
        row
        for row in preview["unmappedFields"]
        if row["fieldPath"].startswith("situation.projects[0].")
    ]
    assert [row["fieldPath"] for row in project_rows] == [
        "situation.projects[0].contract",
        "situation.projects[0].highRank",
        "situation.projects[0].kind",
        "situation.projects[0].mode",
        "situation.projects[0].seat",
        "situation.projects[0].value",
    ]
    assert {
        row["reasonCode"] for row in project_rows
    } == {"v3_project_high_rank_equivalence_is_unproven"}


def test_research_recipe_dispatch_preserves_v1_and_v2_read_compatibility():
    v1 = build_research_recipe(_recipe_v1(schema_version=1))
    v2 = build_research_recipe(_recipe_v1(schema_version=2))

    assert type(v1) is ResearchRecipeV1
    assert type(v2) is ResearchRecipeV2
    assert build_research_recipe(v1.to_dict()) == v1
    assert build_research_recipe(v2.to_dict()) == v2


@pytest.mark.parametrize("schema_version", (1, 2))
def test_legacy_recipe_reads_preserve_sixteen_sample_auction_profiles(
    schema_version: int,
) -> None:
    payload = _recipe_v1(schema_version=schema_version)
    payload["resources"].update(
        {
            "auctionConstructionMode": "cp_sat_actor_safe_decision_v1",
            "playConstructionMode": "actor_safe_agent_verified",
            "strategicAuction": _legacy_strategic_auction_profile(
                requested_samples=16
            ),
        }
    )

    recipe = build_legacy_research_recipe_for_read(payload)

    assert recipe.resources["strategicAuction"]["requestedSamples"] == 16
    assert recipe.resources["strategicAuction"]["tieBreak"] == (
        "win_probability_then_best_share_then_action_id"
    )


def test_v3_strategic_auction_samples_are_the_deep_profile_compatibility_mirror(
) -> None:
    canonical = _recipe_v1(schema_version=3)
    canonical["resources"].update(
        {
            "auctionConstructionMode": "cp_sat_actor_safe_decision_v1",
            "playConstructionMode": "actor_safe_agent_verified",
            "strategicAuction": _strategic_auction_profile(
                requested_samples=(
                    RESEARCH_STRATEGIC_AUCTION_DEEP_BID_SAMPLES_V3
                )
            ),
        }
    )
    built = ResearchRecipeV3.build(canonical)
    assert built.resources["strategicAuction"]["requestedSamples"] == 192
    assert built.resources["strategicAuction"]["tieBreak"] == (
        RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1
    )

    drifted = copy.deepcopy(canonical)
    drifted["resources"]["strategicAuction"]["requestedSamples"] = 191
    with pytest.raises(
        ValueError,
        match=(
            "research invariant field mismatch: "
            "resources.strategicAuction.requestedSamples"
        ),
    ):
        ResearchRecipeV3.build(drifted)

    omitted = copy.deepcopy(canonical)
    omitted["resources"]["strategicAuction"].pop("requestedSamples")
    with pytest.raises(
        ValueError,
        match="actor-safe CP-SAT auction profile is incomplete",
    ):
        ResearchRecipeV3.build(omitted)


def test_research_recipe_v3_rejects_coercion_and_non_finite_values():
    invalid_cases = []

    seed_string = _recipe_v1(schema_version=3)
    seed_string["resources"]["seedCount"] = "20"
    invalid_cases.append((seed_string, "resources.seedCount requires integer"))

    tactic_string = _recipe_v1(schema_version=3)
    tactic_string["tactics"][0]["minimumScore"] = "70"
    invalid_cases.append(
        (tactic_string, r"tactics\[\].minimumScore requires number")
    )

    gate_string = _recipe_v1(schema_version=3)
    gate_string["qualification"]["requireObserverSafety"] = "false"
    invalid_cases.append(
        (gate_string, "qualification.requireObserverSafety requires boolean")
    )

    fractional_score = _recipe_v1(schema_version=3)
    fractional_score["situation"]["matchScore"] = [1.9, 2]
    invalid_cases.append((fractional_score, "matchScore requires exactly two integers"))

    fractional_range = _recipe_v1(schema_version=3)
    fractional_range["situation"]["finalSuitCountRanges"] = {
        "south": {"spades": {"minimum": 1.9, "maximum": 8}}
    }
    invalid_cases.append((fractional_range, "requires integer bounds"))

    for non_finite in (float("nan"), float("inf"), float("-inf")):
        payload = _recipe_v1(schema_version=3)
        payload["qualification"]["maximumRegret"] = non_finite
        invalid_cases.append((payload, "maximumRegret"))

    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            ResearchRecipeV3.build(payload)

    api_payload = _v3_create_payload()
    api_payload["resources"]["seedCount"] = "20"
    with pytest.raises(ValidationError, match="seedCount"):
        ResearchRecipeV3CreateRequest.model_validate(api_payload)

    api_payload = _v3_create_payload()
    api_payload["qualification"]["maximumRegret"] = float("inf")
    with pytest.raises(ValidationError, match="maximumRegret"):
        ResearchRecipeV3CreateRequest.model_validate(api_payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("requiredCards", {"invented": ["SA"]}),
        ("requiredCards", {"south": ["ZZ"]}),
        (
            "finalSuitCountRanges",
            {"south": {"invented": {"minimum": 0, "maximum": 8}}},
        ),
        ("escalations", ["invented"]),
    ),
)
def test_research_recipe_v3_api_types_cards_seats_suits_and_escalations(
    field: str,
    value: object,
):
    api_payload = _v3_create_payload()
    api_payload["situation"][field] = value
    with pytest.raises(ValidationError, match=field):
        ResearchRecipeV3CreateRequest.model_validate(api_payload)


def test_research_recipe_v3_enforces_nested_profile_identities():
    construction = {
        "schemaVersion": 1,
        "id": "actor_safe_pre_screen_v1",
        "revision": "1",
        "enabled": False,
        "maximumPlayDecisions": 4,
        "maximumBidRegret": 0.0,
        "maximumPlayRegret": 0.0,
        "maximumProbabilityDrop": 0.0,
        "minimumConfidence": 0.0,
        "rejectCatastrophicCrossing": True,
    }
    viability = {
        "schemaVersion": 1,
        "id": "exact_tactical_viability_screen_v1",
        "revision": "1",
        "enabled": False,
        "maximumDecisions": 8,
        "requiredTacticsOnly": True,
        "rejectWhenExactAbsent": True,
    }
    tactical = {
        "schemaVersion": 1,
        "id": "tactical_play_precursor_v1",
        "revision": 1,
        "enabled": False,
        "maximumPayoffTricks": 4,
        "minimumLaterRawPoints": 10,
        "requirePrecursorForGatedTactics": True,
    }
    balance = {
        "schemaVersion": 1,
        "id": "balanced_tactical_deal_v1",
        "revision": 1,
        "enabled": False,
        "maximumTeamHighCardDifference": 2,
        "minimumHighCardsPerSeat": 2,
        "maximumHighCardsPerSeat": 4,
        "minimumTargetLowCards": 1,
    }

    cases = (
        ("constructionPreScreen", {**construction, "id": "unregistered"}, "constructionPreScreen.id"),
        ("constructionPreScreen", {**construction, "revision": "2"}, "constructionPreScreen.revision"),
        ("tacticalViabilityPreScreen", {**viability, "revision": "2"}, "tacticalViabilityPreScreen.revision"),
        ("tacticalConstruction", {**tactical, "revision": 2}, "tacticalConstruction.revision"),
        ("tacticalDealBalance", {**balance, "revision": 2}, "tacticalDealBalance.revision"),
    )
    for key, profile, field in cases:
        payload = _recipe_v1(schema_version=3)
        payload["resources"][key] = profile
        with pytest.raises(ValueError, match=field):
            ResearchRecipeV3.build(payload)

    invalid_diversity = _recipe_v1(schema_version=3)
    invalid_diversity["diversity"]["schemaVersion"] = 2
    with pytest.raises(ValueError, match="diversity.schemaVersion"):
        ResearchRecipeV3.build(invalid_diversity)

    api_payload = _v3_create_payload()
    api_payload["resources"]["constructionPreScreen"] = {
        **construction,
        "id": "unregistered",
    }
    with pytest.raises(ValidationError, match="constructionPreScreen"):
        ResearchRecipeV3CreateRequest.model_validate(api_payload)


def test_research_recipe_v3_distinguishes_omitted_and_empty_diversity():
    omitted = _recipe_v1(schema_version=3)
    omitted["diversity"].pop("cellDimensions")
    normalized = ResearchRecipeV3.build(omitted)
    assert normalized.diversity["cellDimensions"] == [
        "contract", "tactical_detector", "acting_seat", "source"
    ]

    empty = _recipe_v1(schema_version=3)
    empty["diversity"]["cellDimensions"] = []
    with pytest.raises(ValueError, match="unique and non-empty"):
        ResearchRecipeV3.build(empty)

    api_payload = _v3_create_payload()
    api_payload["diversity"]["cellDimensions"] = []
    with pytest.raises(ValidationError, match="cellDimensions"):
        ResearchRecipeV3CreateRequest.model_validate(api_payload)


def test_research_recipe_v3_rejects_unqualified_truth_and_placeholder_diversity():
    rejected_truth = _recipe_v1(schema_version=3)
    rejected_truth["qualification"]["minimumTruth"] = "rejected"
    with pytest.raises(ValueError, match="qualification.minimumTruth"):
        ResearchRecipeV3.build(rejected_truth)

    placeholder_diversity = _recipe_v1(schema_version=3)
    placeholder_diversity["diversity"]["cellDimensions"] = [
        "contract", "video_duration"
    ]
    with pytest.raises(ValueError, match="diversity.cellDimensions"):
        ResearchRecipeV3.build(placeholder_diversity)

    api_payload = _v3_create_payload()
    api_payload["qualification"]["minimumTruth"] = "rejected"
    with pytest.raises(ValidationError, match="minimumTruth"):
        ResearchRecipeV3CreateRequest.model_validate(api_payload)

    api_payload = _v3_create_payload()
    api_payload["diversity"]["cellDimensions"] = ["video_duration"]
    with pytest.raises(ValidationError, match="cellDimensions"):
        ResearchRecipeV3CreateRequest.model_validate(api_payload)


def test_research_recipe_v3_projects_require_complete_rows_and_materialize_mode():
    project = {
        "seat": "south",
        "contract": "sun",
        "kind": "siri",
        "value": 20,
        "highRank": "ace",
    }
    for missing in ("value", "highRank"):
        payload = _recipe_v1(schema_version=3)
        payload["situation"]["projects"] = [
            {key: value for key, value in project.items() if key != missing}
        ]
        with pytest.raises(ValueError, match=rf"projects\[0\].{missing}"):
            ResearchRecipeV3.build(payload)

    payload = _recipe_v1(schema_version=3)
    payload["situation"]["projects"] = [project]
    normalized = ResearchRecipeV3.build(payload)
    assert normalized.situation["projects"] == [{**project, "mode": "hard"}]
    assert ResearchRecipeV3.build(normalized.to_dict()) == normalized

    soft = _recipe_v1(schema_version=3)
    soft["situation"]["projects"] = [{**project, "mode": "preference"}]
    with pytest.raises(ValueError, match=r"projects\[\].mode"):
        ResearchRecipeV3.build(soft)


@pytest.mark.parametrize("kind", ["contextualized_laffa", "sakka"])
def test_research_recipe_v3_rejects_unimplemented_output_kinds(kind: str):
    payload = _recipe_v1(schema_version=3)
    payload["output"] = {"kind": kind, "resultCount": 1}
    with pytest.raises(ValueError, match="output.kind"):
        ResearchRecipeV3.build(payload)

    api_payload = _v3_create_payload()
    api_payload["output"] = {"kind": kind, "resultCount": 1}
    with pytest.raises(ValidationError, match="output.kind"):
        ResearchRecipeV3CreateRequest.model_validate(api_payload)


def test_legacy_research_read_adapter_preserves_historical_gates():
    for schema_version, response_model in (
        (1, ResearchRecipeResponse),
        (2, ResearchRecipeV2Response),
    ):
        payload = _recipe_v1(schema_version=schema_version)
        payload["qualification"]["requireObserverSafety"] = False
        payload["qualification"]["requireAllDecisionAudit"] = False
        legacy = build_legacy_research_recipe_for_read(payload)
        assert legacy.qualification["requireObserverSafety"] is False
        assert legacy.qualification["requireAllDecisionAudit"] is False
        serialized = response_model.model_validate(legacy.to_dict())
        assert serialized.qualification.requireObserverSafety is False
        assert serialized.qualification.requireAllDecisionAudit is False

    request_payload = _recipe_v1(schema_version=1)
    request_payload["qualification"]["requireObserverSafety"] = False
    with pytest.raises(ValidationError, match="requireObserverSafety"):
        ResearchRecipeRequest.model_validate(request_payload)
    with pytest.raises(ValueError, match="supports only V1 and V2"):
        build_legacy_research_recipe_for_read(_recipe_v1(schema_version=3))


def test_research_recipe_v3_partial_resource_defaults_are_lane_aware():
    for resources in (None, {}, {"seedCount": 20}):
        payload = _recipe_v1(schema_version=3)
        if resources is None:
            payload.pop("resources")
        else:
            payload["resources"] = resources
        recipe = ResearchRecipeV3.build(payload)
        assert recipe.resources["auctionConstructionMode"] == (
            "actor_safe_agent_verified"
        )
        assert recipe.resources["playConstructionMode"] == (
            "actor_safe_agent_verified"
        )

    organic_resources = {
        "generationLane": "organic_runtime",
        "seedStart": 100,
        "seedCount": 20,
        "seedsPerShard": 5,
        "analysisMode": "cascaded",
        "deepVerificationPolicy": "required_gates",
    }
    organic = _recipe_v1(schema_version=3)
    organic["resources"] = organic_resources
    organic["situation"]["requiredCards"] = {}
    normalized = ResearchRecipeV3.build(organic)
    assert normalized.resources["generationLane"] == "organic_runtime"
    for key in (
        "auctionConstructionMode",
        "playConstructionMode",
        "candidatesPerSeed",
        "solverTimeoutSeconds",
        "maximumConflicts",
    ):
        assert key not in normalized.resources

    explicit_solver = copy.deepcopy(organic)
    explicit_solver["resources"]["maximumConflicts"] = 100
    with pytest.raises(
        ValueError,
        match="organic_forbids_ortools_solver_controls",
    ):
        ResearchRecipeV3.build(explicit_solver)

    explicit_structural = copy.deepcopy(organic)
    explicit_structural["situation"]["requiredCards"] = {"south": ["SA"]}
    with pytest.raises(
        ValueError,
        match="organic_forbids_ortools_structural_constraints",
    ):
        ResearchRecipeV3.build(explicit_structural)

    api_organic = _v3_create_payload()
    api_organic["resources"] = {"generationLane": "organic_runtime"}
    api_model = ResearchRecipeV3CreateRequest.model_validate(api_organic)
    serialized = api_model.model_dump(mode="json", exclude_none=True)
    assert serialized["resources"] == {
        "generationLane": "organic_runtime",
        "seedStart": 1,
        "seedCount": 10,
        "seedsPerShard": 10,
        "analysisMode": "cascaded",
        "deepVerificationPolicy": "required_gates",
    }


def test_research_recipe_v3_catalog_defaults_are_executable_and_disabled_profiles_omit():
    canonical_defaults = _recipe_v1(schema_version=3)
    canonical_defaults.pop("output")
    canonical_defaults.pop("situation")
    defaulted_recipe = ResearchRecipeV3.build(canonical_defaults)
    assert defaulted_recipe.output == {"kind": "laffa", "resultCount": 1}
    assert defaulted_recipe.situation["dealer"] == "east"
    assert defaulted_recipe.situation["contracts"] == ["sun"]

    payload = _recipe_v1(schema_version=3)
    payload["situation"].pop("contracts")
    normalized = ResearchRecipeV3.build(payload)
    assert normalized.situation["contracts"] == ["sun"]

    api_payload = _v3_create_payload()
    api_payload["situation"].pop("contracts")
    api_payload["resources"]["tacticalConstruction"] = {}
    api_payload["resources"]["tacticalDealBalance"] = {}
    api_model = ResearchRecipeV3CreateRequest.model_validate(api_payload)
    assert api_model.situation.contracts == ["sun"]
    assert api_model.resources.tacticalConstruction is None
    assert api_model.resources.tacticalDealBalance is None

    minimal_api = _v3_create_payload()
    minimal_api.pop("output")
    minimal_api.pop("situation")
    minimal_model = ResearchRecipeV3CreateRequest.model_validate(minimal_api)
    assert minimal_model.output.model_dump() == {
        "kind": "laffa",
        "resultCount": 1,
    }
    assert minimal_model.situation.contracts == ["sun"]

    normal_escalation = _recipe_v1(schema_version=3)
    normal_escalation["situation"]["escalations"] = ["normal"]
    assert ResearchRecipeV3.build(normal_escalation).situation["escalations"] == []


def test_research_recipe_v3_enforces_declared_cross_field_relations():
    objective = _recipe_v1(schema_version=3)
    objective["situation"]["buyer"] = None
    objective["situation"]["constructionBuyerRelation"] = "objective"
    with pytest.raises(
        ValueError,
        match="construction_relation_requires_objective_seat",
    ):
        ResearchRecipeV3.build(objective)

    project = {
        "seat": "south",
        "contract": "sun",
        "kind": "siri",
        "value": 20,
        "highRank": "ace",
        "mode": "hard",
    }
    multiple = _recipe_v1(schema_version=3)
    multiple["situation"]["projects"] = [
        project,
        {**project, "seat": "north"},
    ]
    with pytest.raises(ValueError, match="at_most_one_hard_project"):
        ResearchRecipeV3.build(multiple)

    singular = _recipe_v1(schema_version=3)
    singular["output"] = {"kind": "laffa", "resultCount": 2}
    with pytest.raises(ValueError, match="single_laffa_requires_one_result"):
        ResearchRecipeV3.build(singular)

    suit_conflict = _recipe_v1(schema_version=3)
    suit_conflict["situation"]["requiredCards"] = {
        "south": ["S7", "S8", "S9", "S10", "SJ"]
    }
    suit_conflict["situation"]["finalSuitCountRanges"] = {
        "south": {"spades": {"minimum": 0, "maximum": 4}}
    }
    with pytest.raises(ValueError, match="required_cards_fit_suit_ranges"):
        ResearchRecipeV3.build(suit_conflict)

    for contracts in ([], ["sun", "hokum"]):
        invalid_contract_scope = _recipe_v1(schema_version=3)
        invalid_contract_scope["situation"]["contracts"] = contracts
        with pytest.raises(
            ValueError,
            match="tactics_require_exactly_one_contract",
        ):
            ResearchRecipeV3.build(invalid_contract_scope)

    split_tactic_rows = _recipe_v1(schema_version=3)
    split_tactic_rows["tactics"] = [
        {
            "kind": "only_winning_move",
            "mode": "preference",
            "minimumScore": 70,
            "weight": 1,
            "perspectives": [],
        },
        {
            "kind": "small_card_big_gain",
            "mode": "qualification_gate",
            "minimumScore": 70,
            "weight": 1,
            "perspectives": [],
        },
    ]
    split_tactic_rows["resources"]["tacticalDealBalance"] = {
        "schemaVersion": 1,
        "id": "balanced_tactical_deal_v1",
        "revision": 1,
        "enabled": True,
        "maximumTeamHighCardDifference": 2,
        "minimumHighCardsPerSeat": 2,
        "maximumHighCardsPerSeat": 4,
        "minimumTargetLowCards": 1,
    }
    balanced = ResearchRecipeV3.build(split_tactic_rows)
    assert balanced.resources["tacticalDealBalance"]["enabled"] is True


def test_research_recipe_v3_rejects_unknown_and_mismatched_identities():
    unknown = _recipe_v1(schema_version=3)
    unknown["configuration"]["hiddenOverride"] = True
    with pytest.raises(ValueError, match="unknown research configuration authority"):
        ResearchRecipeV3.build(unknown)

    wrong_source = _recipe_v1(schema_version=3)
    wrong_source["configuration"]["identities"]["belief"]["sourceRevisions"][
        "belief"
    ] = "CPR-WRONG"
    with pytest.raises(ValueError, match="belief identity source revision mismatch"):
        ResearchRecipeV3.build(wrong_source)

    omitted_source = _recipe_v1(schema_version=3)
    omitted_source["configuration"]["identities"]["analyzer"][
        "sourceRevisions"
    ].pop("performance")
    with pytest.raises(ValueError, match="analyzer identity source revisions are incomplete"):
        ResearchRecipeV3.build(omitted_source)

    stale_profile = _recipe_v1(schema_version=3)
    stale_profile["profiles"]["analysisProfileHash"] = _hash("stale-analysis")
    with pytest.raises(ValueError, match="profiles mismatch.*analysisProfileHash"):
        ResearchRecipeV3.build(stale_profile)


def test_research_recipe_v3_rejects_uncompiled_hard_tactics_without_breaking_legacy() -> None:
    current = _recipe_v1(schema_version=3)
    current["tactics"][0]["mode"] = "hard"
    with pytest.raises(ValueError, match=r"tactics\[\]\.mode"):
        ResearchRecipeV3.build(current)

    legacy = _recipe_v1(schema_version=1)
    legacy["tactics"][0]["mode"] = "hard"
    built = ResearchRecipeV1.build(legacy)
    assert built.tactics[0]["mode"] == "hard"


def test_research_recipe_v3_hashes_are_stable_and_tamper_evident():
    first = ResearchRecipeV3.build(_recipe_v1(schema_version=3))
    second = ResearchRecipeV3.build(_recipe_v1(schema_version=3))
    assert first.sha256 == second.sha256
    assert first.configuration["sha256"] == second.configuration["sha256"]

    tampered_configuration = first.to_dict()
    tampered_configuration["configuration"]["bundleId"] = "different-bundle"
    with pytest.raises(ValueError, match="configuration authority checksum mismatch"):
        ResearchRecipeV3.build(tampered_configuration)

    tampered_recipe = first.to_dict()
    tampered_recipe["output"]["resultCount"] = 3
    with pytest.raises(ValueError, match="research recipe checksum mismatch"):
        ResearchRecipeV3.build(tampered_recipe)

    snapshot_artifact = copy.deepcopy(first.configuration["snapshotArtifact"])
    snapshot_artifact["bytes"] += 1
    with pytest.raises(ValueError, match="does not match canonical bytes"):
        ResearchConfigurationAuthorityV1.freeze(
            effective_runtime_config=_effective_runtime_snapshot(),
            snapshot_artifact=snapshot_artifact,
            semantic_bundle_hash=first.configuration["semanticBundleHash"],
            identities=first.configuration["identities"],
        )


def test_research_recipe_v3_api_separates_selector_input_from_frozen_response():
    canonical = ResearchRecipeV3.build(_recipe_v1(schema_version=3)).to_dict()
    create_payload = copy.deepcopy(canonical)
    create_payload.pop("profiles")
    create_payload.pop("sha256")
    create_payload["configuration"] = {
        "schemaVersion": 1,
        "bundleRevisionId": canonical["configuration"]["bundleRevisionId"],
    }
    create = ResearchRecipeV3CreateRequest.model_validate(create_payload)
    assert create.model_dump(mode="json")["configuration"] == {
        "schemaVersion": 1,
        "bundleRevisionId": "CBR-RESEARCH-V3-0001",
    }
    assert "profiles" not in create.model_dump(mode="json")

    response = ResearchRecipeV3Response.model_validate(
        {**canonical, "lifecycleStatus": "published"}
    )
    serialized = response.model_dump(mode="json")
    assert serialized["schemaVersion"] == 3
    assert serialized["configuration"]["snapshotArtifact"]["uri"].startswith(
        "cas://sha256/"
    )
    assert serialized["lifecycleStatus"] == "published"

    invalid_selector = create.model_dump(mode="json")
    invalid_selector["configuration"]["rulesHash"] = _hash("client-override")
    with pytest.raises(ValidationError, match="rulesHash"):
        ResearchRecipeV3CreateRequest.model_validate(invalid_selector)


def test_v3_strategic_auction_uses_canonical_tie_break_and_deep_sample_mirror():
    current = ResearchStrategicAuctionV3Request.model_validate({})
    assert current.requestedSamples == (
        RESEARCH_STRATEGIC_AUCTION_DEEP_BID_SAMPLES_V3
    )
    assert current.tieBreak == RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1
    with pytest.raises(ValidationError, match="tieBreak"):
        ResearchStrategicAuctionV3Request.model_validate(
            {"tieBreak": "win_probability_then_best_share_then_action_id"}
        )
    with pytest.raises(ValidationError, match="requestedSamples"):
        ResearchStrategicAuctionV3Request.model_validate(
            {"requestedSamples": 191}
        )
    with TestClient(app) as client:
        rejected = client.post(
            "/api/v2/research-recipes:estimate",
            json={
                "schemaVersion": 3,
                "titleAr": "وصفة مرآة مزاد غير صالحة",
                "configuration": {
                    "schemaVersion": 1,
                    "bundleRevisionId": "CBR-RESEARCH-V3-0001",
                },
                "resources": {
                    "strategicAuction": {"requestedSamples": 191}
                },
            },
        )
    assert rejected.status_code == 422
    assert any(
        issue["loc"][-1] == "requestedSamples"
        for issue in rejected.json()["detail"]
    )

    legacy = ResearchStrategicAuctionRequest.model_validate({})
    assert legacy.requestedSamples == 16
    assert legacy.tieBreak == "win_probability_then_best_share_then_action_id"

    schemas = app.openapi()["components"]["schemas"]
    v3_schema = schemas["ResearchStrategicAuctionV3Request"]["properties"]
    assert v3_schema["requestedSamples"]["default"] == (
        RESEARCH_STRATEGIC_AUCTION_DEEP_BID_SAMPLES_V3
    )
    assert v3_schema["requestedSamples"]["const"] == (
        RESEARCH_STRATEGIC_AUCTION_DEEP_BID_SAMPLES_V3
    )
    assert v3_schema["tieBreak"] == {
        "type": "string",
        "const": RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1,
        "title": "Tiebreak",
        "default": RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1,
    }
    legacy_schema = schemas["ResearchStrategicAuctionRequest"]["properties"]
    assert legacy_schema["requestedSamples"]["default"] == 16
    assert legacy_schema["tieBreak"]["const"] == (
        "win_probability_then_best_share_then_action_id"
    )


def test_research_configuration_snapshot_accepts_local_and_s3_cas_only():
    local = _configuration_authority()
    assert local["snapshotArtifact"]["uri"].startswith("cas://sha256/")
    ResearchConfigurationAuthorityResponse.model_validate(local)

    s3_value = copy.deepcopy(local)
    s3_value.pop("sha256")
    artifact_hash = s3_value["snapshotArtifact"]["sha256"]
    s3_value["snapshotArtifact"]["uri"] = (
        f"s3://balootx-evidence/research/sha256/aa/bb/{artifact_hash}"
    )
    s3_authority = ResearchConfigurationAuthorityV1.build(s3_value).to_dict()
    assert s3_authority["snapshotArtifact"]["uri"].startswith(
        "s3://balootx-evidence/"
    )
    ResearchConfigurationAuthorityResponse.model_validate(s3_authority)

    for invalid_uri in (
        f"s3://balootx-evidence/research/{_hash('wrong-artifact')}",
        f"https://evidence.example/{artifact_hash}",
        f"file:///evidence/{artifact_hash}",
    ):
        tampered = copy.deepcopy(s3_authority)
        tampered.pop("sha256")
        tampered["snapshotArtifact"]["uri"] = invalid_uri
        with pytest.raises(ValueError, match="reference does not match its hash"):
            ResearchConfigurationAuthorityV1.build(tampered)
        with pytest.raises(ValidationError, match="content-addressed storage"):
            ResearchConfigurationAuthorityResponse.model_validate(tampered)


def test_research_recipe_v3_api_routes_preserve_read_retire_and_preview_only(
    monkeypatch: pytest.MonkeyPatch,
):
    from baloot_platform.service import api

    canonical = ResearchRecipeV3.build(_recipe_v1(schema_version=3)).to_dict()
    selector_payload = copy.deepcopy(canonical)
    selector_payload.pop("profiles")
    selector_payload.pop("sha256")
    selector_payload["configuration"] = {
        "schemaVersion": 1,
        "bundleRevisionId": canonical["configuration"]["bundleRevisionId"],
    }

    class FrozenRecipeApiStub:
        def _assert_selector(self, payload: dict) -> None:
            assert payload["schemaVersion"] == 3
            assert payload["configuration"] == selector_payload["configuration"]
            assert "profiles" not in payload

        def validate_recipe(self, payload: dict, *, new_authoring: bool = False) -> dict:
            assert new_authoring is True
            self._assert_selector(payload)
            return {"valid": True, "recipe": canonical, "issues": []}

        def create_recipe(self, payload: dict, *, idempotency_key: str) -> dict:
            raise AssertionError("V3 authoring must be rejected by the wire contract")

        def preview_recipe_v4_conversion(
            self,
            code: str,
            *,
            decision_point: dict | None,
            outcome: dict | None,
            proof: dict | None,
        ) -> dict | None:
            if code != canonical["code"]:
                return None
            return preview_v3_to_v4_conversion(
                canonical,
                decision_point=decision_point,
                outcome=outcome,
                proof=proof,
            )

        def list_recipe_page(
            self,
            *,
            status: str | None,
            cursor: str | None,
            limit: int,
        ) -> dict:
            assert status is None
            assert cursor is None
            assert limit == 100
            return {
                "items": [{**canonical, "lifecycleStatus": "published"}],
                "nextCursor": None,
            }

        def get_recipe(self, code: str) -> dict | None:
            return (
                {**canonical, "lifecycleStatus": "published"}
                if code == canonical["code"]
                else None
            )

        def retire_recipe(self, code: str, *, idempotency_key: str) -> dict:
            if code != canonical["code"]:
                raise KeyError(code)
            assert idempotency_key == "v3-retire-selector"
            return {**canonical, "lifecycleStatus": "retired"}

    monkeypatch.setattr(api, "RESEARCH", FrozenRecipeApiStub())
    with TestClient(api.app) as client:
        validated = client.post(
            "/api/v2/research-recipes:validate", json=selector_payload
        )
        assert validated.status_code == 200, validated.text
        assert validated.json()["recipe"]["configuration"] == canonical[
            "configuration"
        ]

        created = client.post(
            "/api/v2/research-recipes",
            json=selector_payload,
            headers={"Idempotency-Key": "v3-create-selector"},
        )
        assert created.status_code == 422, created.text

        incomplete = client.post(
            f"/api/v2/research-recipes/{canonical['code']}:preview-v4-conversion",
            json={},
        )
        assert incomplete.status_code == 200, incomplete.text
        assert incomplete.json()["targetRecipeCode"] is None
        assert incomplete.json()["targetRecipeHash"] is None

        completion = _v4_operator_completion()
        converted = client.post(
            f"/api/v2/research-recipes/{canonical['code']}:preview-v4-conversion",
            json={
                "decisionPoint": completion["decision_point"],
                "outcome": completion["outcome"],
                "proof": completion["proof"],
            },
        )
        assert converted.status_code == 200, converted.text
        assert converted.json()["buildable"] is True
        assert converted.json()["targetRecipeCode"].startswith("RCP-")
        assert converted.json()["proposedRecipe"]["status"] == "draft"

        missing = client.post(
            "/api/v2/research-recipes/RCP-MISSING:preview-v4-conversion",
            json={},
        )
        assert missing.status_code == 404

        listed = client.get("/api/v2/research-recipes")
        assert listed.status_code == 200, listed.text
        assert listed.json()["items"][0]["schemaVersion"] == 3
        assert listed.json()["nextCursor"] is None

        fetched = client.get(f"/api/v2/research-recipes/{canonical['code']}")
        assert fetched.status_code == 200, fetched.text
        assert fetched.json()["configuration"]["sha256"] == canonical[
            "configuration"
        ]["sha256"]

        retired = client.post(
            f"/api/v2/research-recipes/{canonical['code']}:retire",
            headers={"Idempotency-Key": "v3-retire-selector"},
        )
        assert retired.status_code == 200, retired.text
        assert retired.json()["status"] == "published"
        assert retired.json()["lifecycleStatus"] == "retired"


def test_v3_to_v4_conversion_preview_and_v4_only_creation_are_in_openapi():
    schema = app.openapi()
    create_request = schema["paths"]["/api/v2/research-recipes"]["post"][
        "requestBody"
    ]["content"]["application/json"]["schema"]
    assert create_request == {
        "$ref": "#/components/schemas/ResearchRecipeV4CreateRequest"
    }

    operation = schema["paths"][
        "/api/v2/research-recipes/{code}:preview-v4-conversion"
    ]["post"]
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ResearchRecipeV4ConversionPreviewRequest"
    }
    assert operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/ResearchRecipeV4ConversionPreviewResponse"}


def test_research_service_rejects_new_v3_authoring_before_validation():
    service = object.__new__(ResearchFoundryServiceV1)
    with pytest.raises(ValueError, match="creation requires schemaVersion 4"):
        service.create_recipe(
            {"schemaVersion": 3, "titleAr": "وصفة قديمة غير قابلة للإنشاء"}
        )


def test_research_recipe_routes_preserve_mixed_legacy_and_v3_responses(
    monkeypatch: pytest.MonkeyPatch,
):
    from baloot_platform.service import api

    legacy_v1_payload = _recipe_v1(schema_version=1)
    legacy_v1_payload["qualification"]["requireObserverSafety"] = False
    legacy_v1_payload["qualification"]["requireAllDecisionAudit"] = False
    v1 = ResearchRecipeV1.build(legacy_v1_payload).to_dict()
    v2 = ResearchRecipeV2.build(_recipe_v1(schema_version=2)).to_dict()
    v3 = ResearchRecipeV3.build(_recipe_v1(schema_version=3)).to_dict()
    rows = {
        item["code"]: {**item, "lifecycleStatus": "published"}
        for item in (v1, v2, v3)
    }

    class MixedRecipeApiStub:
        def list_recipe_page(
            self,
            *,
            status: str | None,
            cursor: str | None,
            limit: int,
        ) -> dict:
            assert status is None
            assert cursor is None
            assert limit == 100
            return {"items": list(rows.values()), "nextCursor": None}

        def get_recipe(self, code: str) -> dict | None:
            return rows.get(code)

        def retire_recipe(self, code: str, *, idempotency_key: str) -> dict:
            assert idempotency_key
            if code == "RCP-DRAFT00":
                raise ValueError("draft recipe cannot be retired")
            if code not in rows:
                raise KeyError(code)
            return {**rows[code], "lifecycleStatus": "retired"}

    monkeypatch.setattr(api, "RESEARCH", MixedRecipeApiStub())
    with TestClient(api.app) as client:
        listed = client.get("/api/v2/research-recipes")
        assert listed.status_code == 200, listed.text
        items = {item["schemaVersion"]: item for item in listed.json()["items"]}
        assert set(items) == {1, 2, 3}
        assert items[1]["qualification"]["requireObserverSafety"] is False
        assert items[2]["diversity"] == v2["diversity"]
        assert items[2]["renderPolicy"] == {"mode": "on_demand"}
        assert items[3]["configuration"] == v3["configuration"]

        for canonical in (v1, v2, v3):
            fetched = client.get(
                f"/api/v2/research-recipes/{canonical['code']}"
            )
            assert fetched.status_code == 200, fetched.text
            assert fetched.json()["schemaVersion"] == canonical["schemaVersion"]

        retired = client.post(
            f"/api/v2/research-recipes/{v2['code']}:retire",
            headers={"Idempotency-Key": "legacy-v2-retire"},
        )
        assert retired.status_code == 200, retired.text
        assert retired.json()["schemaVersion"] == 2
        assert retired.json()["diversity"] == v2["diversity"]
        assert retired.json()["lifecycleStatus"] == "retired"

        assert client.get("/api/v2/research-recipes/RCP-MISSING").status_code == 404
        assert client.post(
            "/api/v2/research-recipes/RCP-MISSING:retire",
            headers={"Idempotency-Key": "missing-retire"},
        ).status_code == 404
        assert client.post(
            "/api/v2/research-recipes/RCP-DRAFT00:retire",
            headers={"Idempotency-Key": "draft-retire"},
        ).status_code == 409


def test_bundle_scoped_capabilities_use_canonical_frozen_plugin_identities():
    from baloot_platform.service import api

    revision_id = api.CONTAINER.configuration.resolve()["bundle"]["revisionId"]
    capabilities = api.RESEARCH.capabilities(
        bundle_revision_id=revision_id
    )
    selection = capabilities["research"]["configurationSelection"]
    assert selection["ready"] is True
    assert selection["bundleRevisionId"] == revision_id
    assert len(selection["effectiveRuntimeConfigHash"]) == 64
    assert len(selection["contextHash"]) == 64
    assert selection["domainHash"] == capabilities["compilerV3"]["sha256"]
    assert selection["domainIrHash"] == capabilities["compilerV3"][
        "compiledDomain"
    ]["domainIr"]["sha256"]
    compilation_hash = capabilities["compilerV3"]["compiledDomain"]["sha256"]
    assert api.RESEARCH.domain_compilation(compilation_hash) is not None
    assert api.RESEARCH.domain_compilation_coverage(compilation_hash) is not None


def test_bundle_scoped_capabilities_return_typed_client_error():
    from baloot_platform.service import api

    with TestClient(api.app) as client:
        response = client.get(
            "/api/v2/domain-capabilities",
            params={"bundleRevisionId": "CBR-DOES-NOT-EXIST"},
        )
    assert response.status_code == 422
    assert response.json()["detail"]


def test_frozen_research_context_isolated_and_restart_safe_after_retirement(
    monkeypatch: pytest.MonkeyPatch,
):
    from baloot_platform.service import api

    service = api.CONTAINER.configuration
    active = service.resolve()
    active_revision = str(active["bundle"]["revisionId"])
    factory = api.RESEARCH.execution_context_factory
    assert factory is not None
    _active_effective, active_context = factory.resolve_bundle_revision(
        active_revision
    )
    selected = service.create_bundle(
        bundle_id="research-context-isolation-test",
        version="99.0.0",
        mode=str(active["bundle"]["mode"]),
        component_revisions={
            family: str(value["revisionId"])
            for family, value in active["profileRevisions"].items()
        },
        created_by="TEST",
        scope="tenant",
        tenant_id="RESEARCH-CONTEXT-TEST",
        publish=True,
        force_new_revision=True,
    )
    authority, selected_context = factory.freeze_selector(
        bundle_revision_id=str(selected["revisionId"]),
        analysis_mode="cascaded",
    )
    assert selected_context.bundle_revision_id == selected["revisionId"]
    assert selected_context.context_hash != active_context.context_hash
    assert authority.identities["domain"]["sha256"] == (
        selected_context.compiled_domain.sha256
    )
    assert authority.identities["domain"]["sha256"] != (
        selected_context.legacy_domain_v2.sha256
    )
    assert selected_context.compiled_domain.compiled.domain_ir.sha256
    assert authority.identities["agent"]["version"] == (
        selected_context.bundle.components["agents"].version
    )
    assert selected_context.model_identities == ()

    unrelated_inventory = ResearchExecutionContextFactoryV1(
        configuration_service=service,
        artifact_store=factory.artifact_store,
        root=factory.root,
        plugin_capabilities=(
            *factory.plugin_capabilities,
            {
                "kind": "test_only",
                "id": "unrelated_context_inventory",
                "version": "1.0.0",
                "capabilities": ["unrelated"],
                "deterministic": True,
            },
        ),
        shared_plugin_descriptors=factory.shared_plugin_descriptors,
        active_bundle=active_context.bundle,
        active_exact_search_backend=factory.active_exact_search_backend,
    )
    _effective, inventory_context = unrelated_inventory.resolve_bundle_revision(
        str(selected["revisionId"])
    )
    assert inventory_context.compiled_domain.sha256 == (
        selected_context.compiled_domain.sha256
    )
    assert inventory_context.context_hash == selected_context.context_hash

    active_c_analysis = copy.deepcopy(active["profiles"]["analysis"])
    active_c_analysis["requested_samples"] = max(
        1, int(active_c_analysis["requested_samples"]) - 1
    )
    active_c_revision = service.create_profile(
        family="analysis",
        payload=active_c_analysis,
        created_by="TEST",
        scope="tenant",
        tenant_id="RESEARCH-CONTEXT-ACTIVE-C",
        publish=True,
    )
    active_c_components = {
        family: str(value["revisionId"])
        for family, value in active["profileRevisions"].items()
    }
    active_c_components["analysis"] = str(active_c_revision["revisionId"])
    active_c = service.create_bundle(
        bundle_id="research-context-active-c",
        version="99.0.0",
        mode=str(active["bundle"]["mode"]),
        component_revisions=active_c_components,
        created_by="TEST",
        scope="tenant",
        tenant_id="RESEARCH-CONTEXT-ACTIVE-C",
        publish=True,
        force_new_revision=True,
    )
    _effective_c, active_c_context = factory.resolve_bundle_revision(
        str(active_c["revisionId"])
    )
    assert active_c_context.context_hash != selected_context.context_hash

    service.retire_bundle(
        str(selected["revisionId"]),
        expected_etag=int(selected["etag"]),
        actor_id="TEST",
        reason="restart hydration proof",
    )
    restarted = ResearchExecutionContextFactoryV1(
        configuration_service=service,
        artifact_store=factory.artifact_store,
        root=factory.root,
        plugin_capabilities=factory.plugin_capabilities,
        shared_plugin_descriptors=factory.shared_plugin_descriptors,
        active_bundle=active_c_context.bundle,
        active_exact_search_backend=factory.active_exact_search_backend,
        cache_size=1,
    )
    monkeypatch.setattr(
        service,
        "resolve",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("frozen hydration must not resolve mutable rows")
        ),
    )
    hydrated = restarted.resolve_authority(
        authority.to_dict(), analysis_mode="cascaded"
    )
    assert hydrated.context_hash == selected_context.context_hash
    assert hydrated.bundle_revision_id == selected["revisionId"]

    original_get = restarted.artifact_store.get
    monkeypatch.setattr(
        restarted.artifact_store,
        "get",
        lambda uri: (_ for _ in ()).throw(FileNotFoundError(uri)),
    )
    with pytest.raises(FileNotFoundError):
        restarted.resolve_authority(
            authority.to_dict(), analysis_mode="cascaded"
        )
    monkeypatch.setattr(
        restarted.artifact_store,
        "get",
        lambda uri: original_get(uri) + b"x",
    )
    with pytest.raises(ValueError, match="byte count mismatch"):
        restarted.resolve_authority(
            authority.to_dict(), analysis_mode="cascaded"
        )
