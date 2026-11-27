from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from service_test_client import OperatorClient as TestClient
from pydantic import ValidationError

from baloot_platform.research.result_authority import (
    ExecutionReceiptV1,
    ResearchAuthorityV1,
    RuntimeCompatibilityV1,
)
from baloot_platform.service import api
from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore
from baloot_platform.service.api import (
    ExecutionReceiptV1Response,
    ResearchOperatorAuditV1Response,
    ResearchPlayingQualityAuditV1Response,
    ResearchRecipeEstimateResponse,
    ResearchResourcesV4Request,
    ResearchEstimateShardingV2Response,
    ResearchSequentialSearchReceiptListResponse,
    ResearchSequentialSearchReceiptResponse,
    ResearchSequentialSearchReceiptSummaryResponse,
)


ESTIMATE_FIELDS = {
    "schemaVersion",
    "seedCount",
    "candidateCount",
    "shardCount",
    "maximumSolverSeconds",
    "analysisDecisionUpperBound",
    "warnings",
    "tacticalVerificationDecisionUpperBound",
    "valid",
    "issues",
}

ESTIMATE_V2_FIELDS = {
    "maximumCampaignSeconds",
    "sharding",
    "scenarioPlanHash",
    "capabilityPlanHash",
    "nativeCapabilityPlanHash",
    "predicatesByConstructionCapability",
    "postconditionOnlyPredicates",
    "exactInputOnlyPredicates",
    "unsupportedPredicates",
    "output",
    "proof",
    "expectedCostCategory",
    "launchAdmissible",
    "configurationAuthority",
    "backendIdentity",
    "compiledStages",
    "constructionPlan",
    "resourceTier",
    "refusalReason",
    "estimatedModelVariables",
    "estimatedModelConstraints",
    "estimatedGraphNodes",
    "estimatedGraphEdges",
    "estimatedHiddenWorlds",
    "estimatedCountLimbs",
    "firstResultCostClass",
    "requestedResultsCostClass",
    "alternativeProofModes",
    "requestedCombinationSupported",
    "exhaustiveSearchInfeasibilityEstablishable",
    "estimateHash",
}

ESTIMATE_CONSTRUCTION_PLAN_V1_FIELDS = {
    "schemaVersion",
    "planner",
    "fallbackReason",
    "prefixLength",
    "tailOffset",
    "tailLength",
    "sha256",
}

ESTIMATE_OUTPUT_V2_FIELDS = {
    "kind",
    "requestedResultCount",
    "requestedVideoCount",
    "maximumCandidateCount",
    "maximumVariantCount",
    "admissible",
    "reason",
    "recommendedResultCount",
    "requiresResourceExpansion",
}

ESTIMATE_PROOF_V2_FIELDS = {
    "requestedMode",
    "visibility",
    "strategyClass",
    "horizon",
    "admissible",
    "reason",
    "recommendedMode",
    "recommendedModes",
    "estimatedRootActionUpperBound",
    "actionDomainUpperBound",
    "estimatedEkkaActionVariantUpperBound",
    "ekkaDeclarationSequenceUpperBound",
    "targetInformationSetUpperBound",
    "opponentInformationSetUpperBound",
    "targetPolicyCountUpperBound",
    "opponentPolicyCountUpperBound",
    "policyProductUpperBound",
    "estimatedTreeStateUpperBound",
    "policyProductExceedsBudget",
    "informationTreeExceedsBudget",
    "estimatedStateSpaceClass",
    "hiddenWorldClass",
    "budgetProfile",
    "resourceBudgetProfile",
    "remainingCardCount",
    "requestedDepth",
    "budgetMaximumDepth",
    "budgetMaximumHorizonActions",
    "estimatedStateSpace",
    "estimatedHiddenWorldUpperBound",
    "budgetMaximumHiddenWorlds",
    "budgetStateSpaceLimit",
    "budgetProfileHash",
    "proofBudgetRegistryHash",
}

ESTIMATE_SHARDING_V2_FIELDS = {
    "configuredSeedsPerShard",
    "requestedShardCount",
    "actualShardCount",
    "maximumShardWorkers",
    "requestedShardWorkers",
    "effectiveShardWorkers",
    "maximumShardWallSeconds",
    "parallelismLimitedByShardPlan",
    "recommendedSeedsPerShard",
}


def test_historical_estimate_sharding_remains_readable_without_worker_override() -> None:
    historical = {
        "configuredSeedsPerShard": 1,
        "requestedShardCount": 4,
        "actualShardCount": 4,
        "maximumShardWorkers": 6,
        "effectiveShardWorkers": 4,
        "parallelismLimitedByShardPlan": False,
        "recommendedSeedsPerShard": 1,
    }
    assert ResearchEstimateShardingV2Response.model_validate(historical).requestedShardWorkers is None


def test_v4_resource_worker_override_is_optional_and_strict() -> None:
    assert "shardWorkerCount" not in ResearchResourcesV4Request().model_dump(exclude_none=True)
    assert ResearchResourcesV4Request(shardWorkerCount=2).shardWorkerCount == 2
    for invalid in (0, True, 2.5):
        with pytest.raises(ValidationError):
            ResearchResourcesV4Request(shardWorkerCount=invalid)


def _generation_request() -> dict:
    return {"schemaVersion": 1, "catalogHash": "a" * 64,
            "tagQuery": {"impact": ["round-flip"], "technique": ["guard/protect"]}}


def test_v4_generation_api_contract_is_optional_canonical_and_shared() -> None:
    from baloot_platform.research.family_catalog import (
        build_family_generation_request_v1, tag_query_schema_v1,
    )

    generation = _generation_request()
    expected = build_family_generation_request_v1(generation)
    parsed = api.ResearchGenerationV1Request.model_validate(generation)
    assert parsed.model_dump() == expected
    assert api.ResearchGenerationV1Request.model_validate(expected).model_dump() == expected
    assert "sha256" not in generation
    predicate = {"predicateId": "PRED-CONTRACT", "kind": "auction.contract",
                 "version": 1, "hardness": "hard", "arguments": {"contract": "sun"}}
    payload = {"schemaVersion": 4, "titleAr": "Scenario generation contract",
        "configuration": {"schemaVersion": 1, "bundleRevisionId": "CBR-TEST"},
        "scenario": {"origin": {"kind": "search"}, "constraints": predicate,
            "decisionPoint": {"kind": "play", "selector": {}}, "outcome": predicate},
        "proof": {"mode": "exact", "visibility": "complete_state", "claim": "outcome_reachable",
            "targetTeam": "north_south", "utility": "awarded_point_margin",
            "budgetProfile": "late_game_exact_v1"}}
    historical = api.ResearchRecipeV4CreateRequest.model_validate(payload)
    assert "generation" not in historical.model_dump(exclude_none=True)
    assert api.ResearchRecipeV4CreateRequest.model_validate({**payload, "generation": generation}).generation.model_dump() == expected
    with pytest.raises(ValidationError, match="generation must be an object"):
        api.ResearchRecipeV4CreateRequest.model_validate({**payload, "generation": None})
    for model in (api.ResearchRecipeV4CreateRequest, api.ResearchRecipeV4Response):
        schema = model.model_json_schema()
        assert "generation" not in schema["required"]
        assert schema["properties"]["generation"]["$ref"].endswith("/ResearchGenerationV1Request")
        tags = schema["$defs"]["ResearchGenerationV1Request"]["properties"]["tagQuery"]
        assert all(tags[key] == value for key, value in tag_query_schema_v1().items())


def test_v4_generation_auction_hints_are_typed_canonical_and_optional() -> None:
    from baloot_platform.research.family_catalog import build_family_generation_request_v1

    raw = {**_generation_request(), "auctionSeeds": [
        {"seed": 6, "dealer": "south"}, {"seed": 12, "dealer": "north"}]}
    parsed = api.ResearchGenerationV1Request.model_validate(raw)
    assert parsed.model_dump() == build_family_generation_request_v1(raw)
    assert isinstance(parsed.auctionSeeds[0], api.ResearchAuctionSeedV1Request)
    assert "auctionSeeds" not in api.ResearchGenerationV1Request.model_validate(_generation_request()).model_dump()
    run = {"runCode": "RUN-ABCDEF123456", "runHash": "a" * 64,
        "recipeCode": "RCP-APICONTRACT", "recipeHash": "b" * 64,
        "status": "completed", "progress": 1.0, "createdAt": "2026-10-16T00:00:00+00:00"}
    assert "auctionOpenings" not in api.ResearchRunResponse.model_validate(run).model_dump(exclude_none=True)
    response = api.ResearchRunResponse.model_validate({**run, "auctionOpenings": raw["auctionSeeds"]})
    assert response.model_dump(exclude_none=True)["auctionOpenings"] == raw["auctionSeeds"]
    schema = api.ResearchGenerationV1Request.model_json_schema()
    hint = schema["properties"]["auctionSeeds"]
    assert hint['uniqueItems'] is True
    assert "auctionSeeds" not in schema["required"]
    assert hint["items"]["$ref"].endswith("/ResearchAuctionSeedV1Request")
    assert schema["$defs"]["ResearchAuctionSeedV1Request"]["additionalProperties"] is False


@pytest.mark.parametrize("hints", [None, [], [{"seed": True, "dealer": "south"}],
    [{"seed": "6", "dealer": "south"}], [{"seed": -1, "dealer": "south"}],
    [{"seed": 6.0, "dealer": "south"}], [{"seed": 6, "dealer": "South"}],
    [{"seed": 6, "dealer": "south", "buyer": "north"}],
    [{"seed": 6, "dealer": "south"}, {"seed": 6, "dealer": "south"}]])
def test_v4_generation_api_rejects_invalid_auction_hints(hints) -> None:
    with pytest.raises(ValidationError):
        api.ResearchGenerationV1Request.model_validate({**_generation_request(), "auctionSeeds": hints})


@pytest.mark.parametrize("mutation", [
    lambda raw: raw.update(unknown=True),
    lambda raw: raw.update(schemaVersion=True),
    lambda raw: raw.update(schemaVersion="1"),
    lambda raw: raw.update(catalogHash="A" * 64),
    lambda raw: raw.update(catalogHash=1),
    lambda raw: raw.update(tagQuery=None),
    lambda raw: raw.update(tagQuery={"unknown": ["round-flip"]}),
    lambda raw: raw.update(tagQuery={"impact": "round-flip"}),
    lambda raw: raw.update(tagQuery={"impact": ["round-flip", "round-flip"]}),
    lambda raw: raw.update(tagQuery={"impact": []}),
    lambda raw: raw.update(tagQuery={"impact": [1]}),
    lambda raw: raw.update(sha256=None),
    lambda raw: raw.update(sha256="0" * 64),
])
def test_v4_generation_api_contract_rejects_invalid_or_coercive_values(mutation) -> None:
    raw = _generation_request()
    mutation(raw)
    with pytest.raises(ValidationError):
        api.ResearchGenerationV1Request.model_validate(raw)


def test_family_catalog_profile_schema_reuses_closed_catalog_and_stream_dependencies(root) -> None:
    import jsonschema
    import yaml
    from baloot_platform.research.family_catalog import family_catalog_schema_v1
    from scripts.generate_profile_schemas import apply_release_constraints, infer

    profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
    schema = infer(profile)
    apply_release_constraints("challenge", schema, profile)
    quality = schema["properties"]["scenarioResearchV4"]["properties"]["qualityExecution"]
    assert quality["properties"]["familyCatalog"] == family_catalog_schema_v1()
    assert "familyCatalog" not in quality["required"]
    catalog = yaml.safe_load((root / "configs/research/families.yaml").read_text(encoding="utf-8"))
    historical = profile["scenarioResearchV4"]["qualityExecution"]
    jsonschema.validate(historical, quality)
    import json
    interest = json.loads((root / "tests/fixtures/actor_interest_calibration.json").read_text())["cases"]["7189"]["policy"]
    request = {**historical, "familyCatalog": catalog, "candidateSource": "runtime_proposal_stream_v1",
        "earlyPlayPolicyId": "observer_safe_trick_control_v1", "auditConstructedTrace": True,
        "proposalSeedCount": 10, "videoInterestingness": interest}
    jsonschema.validate(request, quality)
    for field, value in (("candidateSource", "runtime_proposal_v1"), ("familyCatalog", None)):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**request, field: value}, quality)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({key: value for key, value in request.items() if key != "candidateSource"}, quality)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**request, "familyTarget": {}}, {"allOf": quality["allOf"]})
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**request, "acceptedLessonLibrary": {}}, {"allOf": quality["allOf"]})

SEQUENTIAL_SEARCH_SUMMARY_FIELDS = {
    "admittedSeedCount",
    "exactRejectionCount",
    "noGoodAppliedCount",
    "noGoodUnappliedCount",
    "continuationSolveCount",
    "continuationStatusDistribution",
    "stopReasonDistribution",
}


def _operator_audit_sections() -> tuple[dict, dict, dict]:
    authority = ResearchAuthorityV1.build(
        backend_id="ortools_cp_sat_v2",
        backend_version="2.0.0",
        solver_version="9.14.0",
        bundle_revision_id="CBR-OPERATOR-AUDIT",
        bundle_hash="1" * 64,
        domain_ir_hashes={"play": "2" * 64, "scoring": "3" * 64},
        lowering_manifest_hash="4" * 64,
        construction_model_hash="5" * 64,
        proof_model_hash="6" * 64,
        graph_hash="7" * 64,
        trace_hash="8" * 64,
        proof_outcome_hash="9" * 64,
    ).to_dict()
    compatibility = RuntimeCompatibilityV1.build(
        status="not_checked",
        bundle_hash=authority["bundleHash"],
        trace_hash=authority["traceHash"],
    ).to_dict()
    receipt = ExecutionReceiptV1.build(
        semantic_result_hash="a" * 64,
        stage_timings={
            "construction": 0.25,
            "finalNormalization": 0.125,
            "proof": 0.5,
        },
        cpu_seconds=0.75,
        peak_memory_bytes=1024,
        conflicts=2,
        branches=3,
        cache_bytes=128,
        completed_variants=1,
        machine={"cpu": "reference"},
    ).to_dict()
    return authority, compatibility, receipt


def test_research_authority_accepts_json_object_key_reordering() -> None:
    authority, _compatibility, _receipt = _operator_audit_sections()
    authority["domainIrHashes"] = {
        "scoring": authority["domainIrHashes"]["scoring"],
        "play": authority["domainIrHashes"]["play"],
    }

    parsed = ResearchAuthorityV1.from_dict(authority).to_dict()
    response = api.ResearchAuthorityV1Response.model_validate(authority)

    assert list(parsed["domainIrHashes"]) == ["play", "scoring"]
    assert list(response.domainIrHashes) == ["play", "scoring"]


def test_execution_receipt_accepts_json_object_key_reordering() -> None:
    _authority, _compatibility, receipt = _operator_audit_sections()
    receipt["stageTimings"] = {
        "proof": receipt["stageTimings"]["proof"],
        "finalNormalization": receipt["stageTimings"]["finalNormalization"],
        "construction": receipt["stageTimings"]["construction"],
    }

    parsed = ExecutionReceiptV1.from_dict(receipt).to_dict()
    response = ExecutionReceiptV1Response.model_validate(receipt)

    expected_stages = ["construction", "finalNormalization", "proof"]
    assert list(parsed["stageTimings"]) == expected_stages
    assert list(response.stageTimings) == expected_stages


def _estimate_response() -> dict:
    return {
        "schemaVersion": 1,
        "seedCount": 10,
        "candidateCount": 20,
        "shardCount": 2,
        "maximumSolverSeconds": 300.0,
        "analysisDecisionUpperBound": 800,
        "warnings": [],
        "tacticalVerificationDecisionUpperBound": 80,
        "valid": True,
        "issues": [],
    }


def _sequential_search_summary() -> dict:
    return {
        "admittedSeedCount": 2,
        "exactRejectionCount": 3,
        "noGoodAppliedCount": 3,
        "noGoodUnappliedCount": 0,
        "continuationSolveCount": 4,
        "continuationStatusDistribution": {
            "optimal": 3,
            "infeasible": 1,
        },
        "stopReasonDistribution": {
            "candidates_exhausted": 1,
            "maximum_solutions_reached": 0,
            "qualified": 1,
            "incomplete": 0,
            "cancelled": 0,
            "fatal_quarantine": 0,
        },
    }


def test_research_recipe_estimate_response_is_strict_and_complete() -> None:
    payload = _estimate_response()
    model = ResearchRecipeEstimateResponse.model_validate(payload)
    assert model.model_dump(mode="json", exclude_none=True) == payload
    assert not (ESTIMATE_V2_FIELDS & model.model_fields_set)

    invalid_type = deepcopy(payload)
    invalid_type["seedCount"] = "10"
    with pytest.raises(ValidationError):
        ResearchRecipeEstimateResponse.model_validate(invalid_type)

    unexpected = {**payload, "undeclared": True}
    with pytest.raises(ValidationError):
        ResearchRecipeEstimateResponse.model_validate(unexpected)

    incomplete_v2 = {**payload, "schemaVersion": 2}
    with pytest.raises(ValidationError, match="Estimate V2 fields are incomplete"):
        ResearchRecipeEstimateResponse.model_validate(incomplete_v2)


def test_historical_cp_sat_estimate_defaults_campaign_limit_to_its_solve_limit() -> None:
    payload = _estimate_response()
    payload.update(
        {
            "schemaVersion": 2,
            "scenarioPlanHash": "1" * 64,
            "capabilityPlanHash": "2" * 64,
            "predicatesByConstructionCapability": {},
            "postconditionOnlyPredicates": [],
            "exactInputOnlyPredicates": [],
            "unsupportedPredicates": [],
            "output": None,
            "proof": None,
            "expectedCostCategory": "not_compiled",
            "launchAdmissible": False,
            "configurationAuthority": None,
            "backendIdentity": {
                "schemaVersion": 1,
                "backendId": "ortools_cp_sat_v2",
                "backendVersion": "test",
                "qualificationAuthority": "cp_sat",
                "solverVersion": "test",
                "planner": "tail_first_v1",
                "deterministicExecution": True,
                "bundleRevisionId": "CBR-TEST",
                "bundleIdentityHash": "3" * 64,
                "authorityHash": "4" * 64,
                "predicateCatalogHash": "5" * 64,
                "loweringManifestHash": "4" * 64,
                "sha256": "6" * 64,
            },
            "compiledStages": [],
            "resourceTier": "test",
            "refusalReason": "historical",
            "estimatedModelVariables": 0,
            "estimatedModelConstraints": 0,
            "estimatedGraphNodes": 1,
            "estimatedGraphEdges": 0,
            "estimatedHiddenWorlds": 1,
            "estimatedCountLimbs": 0,
            "firstResultCostClass": "not_estimable",
            "requestedResultsCostClass": "not_estimable",
            "alternativeProofModes": [],
            "requestedCombinationSupported": False,
            "exhaustiveSearchInfeasibilityEstablishable": False,
            "estimateHash": "7" * 64,
        }
    )

    model = ResearchRecipeEstimateResponse.model_validate(payload)

    assert model.maximumCampaignSeconds == payload["maximumSolverSeconds"]
    payload["nativeCapabilityPlanHash"] = "8" * 64
    with pytest.raises(ValidationError, match="nativeCapabilityPlanHash requires"):
        ResearchRecipeEstimateResponse.model_validate(payload)
    payload["backendIdentity"]["backendId"] = "ortools_cp_sat_v3"
    assert ResearchRecipeEstimateResponse.model_validate(
        payload
    ).nativeCapabilityPlanHash == "8" * 64
    payload["estimatedHiddenWorlds"] = "14586641389127806976"
    assert ResearchRecipeEstimateResponse.model_validate(
        payload
    ).estimatedHiddenWorlds == payload["estimatedHiddenWorlds"]
    for invalid in ("01", "1e3", 0, True, 1.5):
        payload["estimatedHiddenWorlds"] = invalid
        with pytest.raises(ValidationError):
            ResearchRecipeEstimateResponse.model_validate(payload)


def test_estimate_v2_horizon_is_closed_and_strict() -> None:
    proof = {
        "requestedMode": "bounded",
        "visibility": "complete_state",
        "strategyClass": "pure_deterministic",
        "horizon": {"unit": "tricks", "value": 2},
        "admissible": True,
        "recommendedModes": [],
        "estimatedRootActionUpperBound": 3,
        "actionDomainUpperBound": 3,
        "estimatedEkkaActionVariantUpperBound": 0,
        "ekkaDeclarationSequenceUpperBound": "1",
        "estimatedStateSpaceClass": "late_game_manageable",
        "hiddenWorldClass": "complete_state",
        "budgetProfile": "bounded_v1",
        "resourceBudgetProfile": "bounded_v1",
        "proofBudgetRegistryHash": "1" * 64,
    }
    assert api.ResearchEstimateProofV2Response.model_validate(proof).horizon.value == 2
    with pytest.raises(ValidationError):
        api.ResearchEstimateProofV2Response.model_validate(
            {**proof, "horizon": {"unit": "tricks", "value": 2, "extra": True}}
        )
    with pytest.raises(ValidationError):
        api.ResearchEstimateProofV2Response.model_validate(
            {**proof, "estimatedRootActionUpperBound": "3"}
        )


def test_actor_information_set_strategy_class_is_explicit_and_mode_bound() -> None:
    base = {
        "schemaVersion": 1,
        "mode": "exact",
        "visibility": "actor_information_set",
        "claim": "winning_root_action_count",
        "targetTeam": "north_south",
        "utility": "awarded_point_margin",
        "successThreshold": 1.0,
        "lineUniverse": "all_legal",
        "budgetProfile": "actor_information_set_exact_v1",
    }
    with pytest.raises(ValidationError, match="strategyClass"):
        api.ScenarioProofRequestV1Request.model_validate(base)
    pure = api.ScenarioProofRequestV1Request.model_validate(
        {**base, "strategyClass": "pure_deterministic"}
    )
    assert pure.strategyClass == "pure_deterministic"
    with pytest.raises(ValidationError, match="incompatible"):
        api.ScenarioProofRequestV1Request.model_validate(
            {**base, "strategyClass": "frozen_policy"}
        )

    policies = {seat: {} for seat in ("north", "west", "south", "east")}
    frozen = api.ScenarioProofRequestV1Request.model_validate(
        {
            **base,
            "mode": "policy_conditioned",
            "strategyClass": "frozen_policy",
            "lineUniverse": "policy_conditioned",
            "policies": policies,
        }
    )
    assert frozen.strategyClass == "frozen_policy"
    complete = api.ScenarioProofRequestV1Request.model_validate(
        {
            **base,
            "visibility": "complete_state",
            "strategyClass": "pure_deterministic",
        }
    )
    assert complete.strategyClass == "pure_deterministic"
    with pytest.raises(ValidationError, match="incompatible"):
        api.ScenarioProofRequestV1Request.model_validate(
            {
                **base,
                "visibility": "complete_state",
                "strategyClass": "frozen_policy",
            }
        )


def test_sequential_search_summary_is_exact_closed_and_nested() -> None:
    payload = _sequential_search_summary()
    model = ResearchSequentialSearchReceiptSummaryResponse.model_validate(payload)
    assert model.model_dump(mode="json") == payload

    with pytest.raises(ValidationError):
        ResearchSequentialSearchReceiptSummaryResponse.model_validate(
            {**payload, "optimalityCutCount": 3}
        )

    invalid_distribution = deepcopy(payload)
    invalid_distribution["continuationStatusDistribution"]["unknown"] = 1
    with pytest.raises(ValidationError):
        ResearchSequentialSearchReceiptSummaryResponse.model_validate(
            invalid_distribution
        )

    negative = deepcopy(payload)
    negative["exactRejectionCount"] = -1
    with pytest.raises(ValidationError):
        ResearchSequentialSearchReceiptSummaryResponse.model_validate(negative)

    openapi = api.app.openapi()
    schemas = openapi["components"]["schemas"]
    summary_schema = schemas["ResearchSequentialSearchReceiptSummaryResponse"]
    assert set(summary_schema["properties"]) == SEQUENTIAL_SEARCH_SUMMARY_FIELDS
    assert set(summary_schema["required"]) == SEQUENTIAL_SEARCH_SUMMARY_FIELDS
    assert summary_schema["additionalProperties"] is False

    continuation_schema = schemas[
        "ResearchSequentialContinuationStatusDistributionResponse"
    ]
    assert set(continuation_schema["properties"]) == {"optimal", "infeasible"}
    assert set(continuation_schema["required"]) == {"optimal", "infeasible"}
    assert continuation_schema["additionalProperties"] is False

    stop_schema = schemas["ResearchSequentialStopReasonDistributionResponse"]
    expected_stop_reasons = {
        "candidates_exhausted",
        "maximum_solutions_reached",
        "qualified",
        "incomplete",
        "cancelled",
        "fatal_quarantine",
    }
    assert set(stop_schema["properties"]) == expected_stop_reasons
    assert set(stop_schema["required"]) == expected_stop_reasons
    assert stop_schema["additionalProperties"] is False

    stats_schema = schemas["ResearchStatsResponse"]
    assert "sequentialSearch" in stats_schema["required"]
    assert stats_schema["properties"]["sequentialSearch"] == {
        "$ref": "#/components/schemas/ResearchSequentialSearchReceiptSummaryResponse"
    }
    assert "optimalityCutCount" not in stats_schema["properties"]


def test_sequential_receipt_api_models_reject_all_coercive_runtime_aliases() -> None:
    summary = _sequential_search_summary()
    item = {
        "schemaVersion": 1,
        "occurrenceHash": "1" * 64,
        "receiptHash": "2" * 64,
        "runCode": "RUN-ABCDEFGHIJKL",
        "shardId": "RSH-STRICT",
        "attemptCount": 1,
        "fencingToken": 1,
        "seed": 9,
        "admitted": True,
        "createdAt": "2026-08-09T00:00:00.000000+00:00",
        "summary": summary,
        "receiptBundle": {},
    }
    page = {
        "items": [item],
        "total": 1,
        "offset": 0,
        "limit": 1,
        "nextOffset": None,
        "hasMore": False,
    }
    assert ResearchSequentialSearchReceiptResponse.model_validate(
        item
    ).model_dump(mode="json") == item
    assert ResearchSequentialSearchReceiptListResponse.model_validate(
        page
    ).model_dump(mode="json") == page

    for field, aliases in {
        "schemaVersion": (True, 1.0, "1"),
        "attemptCount": (True, 1.0, "1"),
        "fencingToken": (True, 1.0, "1"),
        "seed": (True, 9.0, "9"),
        "admitted": (1, 1.0, "true"),
    }.items():
        for alias in aliases:
            invalid = deepcopy(item)
            invalid[field] = alias
            with pytest.raises(ValidationError):
                ResearchSequentialSearchReceiptResponse.model_validate(invalid)

    oversized_seed = deepcopy(item)
    oversized_seed["seed"] = 1 << 63
    with pytest.raises(ValidationError):
        ResearchSequentialSearchReceiptResponse.model_validate(oversized_seed)

    for field, aliases in {
        "total": (True, 1.0, "1"),
        "offset": (False, 0.0, "0"),
        "limit": (True, 1.0, "1"),
        "hasMore": (0, 0.0, "false"),
    }.items():
        for alias in aliases:
            invalid = deepcopy(page)
            invalid[field] = alias
            with pytest.raises(ValidationError):
                ResearchSequentialSearchReceiptListResponse.model_validate(invalid)

    for field in (
        "admittedSeedCount",
        "exactRejectionCount",
        "noGoodAppliedCount",
        "noGoodUnappliedCount",
        "continuationSolveCount",
    ):
        for alias in (True, 1.0, "1"):
            invalid = deepcopy(summary)
            invalid[field] = alias
            with pytest.raises(ValidationError):
                ResearchSequentialSearchReceiptSummaryResponse.model_validate(invalid)
    for distribution, field in (
        ("continuationStatusDistribution", "optimal"),
        ("stopReasonDistribution", "qualified"),
    ):
        for alias in (True, 1.0, "1"):
            invalid = deepcopy(summary)
            invalid[distribution][field] = alias
            with pytest.raises(ValidationError):
                ResearchSequentialSearchReceiptSummaryResponse.model_validate(invalid)


def test_research_run_contract_separates_terminal_audit_result_identities() -> None:
    schema = api.app.openapi()["components"]["schemas"]["ResearchRunResponse"]
    assert {"resultCount", "resultCodes", "auditResultCount", "auditResultCodes"} <= set(
        schema["properties"]
    )
    assert schema["properties"]["auditResultCount"]["default"] == 0
    for field in ("resultCodes", "auditResultCodes"):
        array_schema = next(
            item
            for item in schema["properties"][field]["anyOf"]
            if item.get("type") == "array"
        )
        assert array_schema["items"] == {"type": "string"}


def test_research_estimate_route_and_openapi_use_the_typed_contract() -> None:
    route = next(
        item
        for item in api.app.routes
        if getattr(item, "path", None) == "/api/v2/research-recipes:estimate"
    )
    assert route.response_model is ResearchRecipeEstimateResponse
    assert route.response_model_exclude_none is False
    assert route.response_model_exclude_unset is True

    openapi = api.app.openapi()
    operation = openapi["paths"]["/api/v2/research-recipes:estimate"]["post"]
    request_schema = operation["requestBody"]["content"]["application/json"][
        "schema"
    ]
    assert request_schema["discriminator"]["mapping"] == {
        "1": "#/components/schemas/ResearchRecipeRequest",
        "2": "#/components/schemas/ResearchRecipeV2Request",
        "3": "#/components/schemas/ResearchRecipeV3CreateRequest",
        "4": "#/components/schemas/ResearchRecipeV4CreateRequest",
    }
    assert operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/ResearchRecipeEstimateResponse"}
    schema = openapi["components"]["schemas"][
        "ResearchRecipeEstimateResponse"
    ]
    assert set(schema["properties"]) == ESTIMATE_FIELDS | ESTIMATE_V2_FIELDS
    assert set(schema["required"]) == ESTIMATE_FIELDS
    assert schema["additionalProperties"] is False
    construction_plan_schema = openapi["components"]["schemas"][
        "ResearchEstimateConstructionPlanV1Response"
    ]
    assert set(construction_plan_schema["properties"]) == (
        ESTIMATE_CONSTRUCTION_PLAN_V1_FIELDS
    )
    assert construction_plan_schema["additionalProperties"] is False
    proof_schema = openapi["components"]["schemas"][
        "ResearchEstimateProofV2Response"
    ]
    assert set(proof_schema["properties"]) == ESTIMATE_PROOF_V2_FIELDS
    assert proof_schema["additionalProperties"] is False
    output_schema = openapi["components"]["schemas"][
        "ResearchEstimateOutputV2Response"
    ]
    assert set(output_schema["properties"]) == ESTIMATE_OUTPUT_V2_FIELDS
    assert output_schema["additionalProperties"] is False
    sharding_schema = openapi["components"]["schemas"][
        "ResearchEstimateShardingV2Response"
    ]
    assert set(sharding_schema["properties"]) == ESTIMATE_SHARDING_V2_FIELDS
    assert set(sharding_schema["required"]) == ESTIMATE_SHARDING_V2_FIELDS - {"requestedShardWorkers", "maximumShardWallSeconds"}
    assert sharding_schema["additionalProperties"] is False

    run_conflict = openapi["paths"]["/api/v2/research-runs"]["post"][
        "responses"
    ]["409"]["content"]["application/json"]["schema"]
    assert run_conflict == {
        "$ref": "#/components/schemas/ResearchRunConflictResponse"
    }
    assert openapi["components"]["schemas"]["ResearchRunConflictResponse"][
        "anyOf"
    ] == [
        {"$ref": "#/components/schemas/ResearchMutationErrorResponse"},
        {"$ref": "#/components/schemas/ResearchLaunchAdmissionErrorResponse"},
    ]
    admission = openapi["components"]["schemas"][
        "ResearchLaunchAdmissionDiagnosticResponse"
    ]
    assert set(admission["properties"]) == {
        "schemaVersion",
        "code",
        "recipeCode",
        "recipeHash",
        "configurationAuthority",
        "scenarioPlanHash",
        "estimateHash",
        "reason",
            "recommendedModes",
            "proof",
            "output",
            "sha256",
        }
    assert set(admission["required"]) == set(admission["properties"])
    assert admission["additionalProperties"] is False

    render_selected = openapi["paths"][
        "/api/v2/research-runs/{run_code}:render-selected"
    ]["post"]
    assert render_selected["requestBody"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/ResearchRenderPrepareRequest"}
    assert render_selected["responses"]["202"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/FoundryRenderBatchJobResponse"}
    assert render_selected["responses"]["409"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/ResearchMutationErrorResponse"}

    v4_request = openapi["components"]["schemas"]["ResearchRecipeV4CreateRequest"]
    assert {"output", "renderPolicy"} <= set(v4_request["properties"])
    scenario_configuration = openapi["components"]["schemas"][
        "ResearchScenarioConfigurationV1Response"
    ]
    assert scenario_configuration["properties"]["outputPolicy"] == {
        "$ref": "#/components/schemas/ScenarioOutputPolicyV1Response"
    }
    research_capability = openapi["components"]["schemas"][
        "ResearchCapabilityResponse"
    ]
    assert research_capability["properties"]["scenarioResearchV4"]["anyOf"] == [
        {
            "$ref": (
                "#/components/schemas/ScenarioResearchV4CapabilityV1Response"
            )
        },
        {"type": "null"},
    ]
    assert "scenarioResearchV4" not in research_capability["required"]
    ortools_capability = openapi["components"]["schemas"][
        "ScenarioResearchV4CapabilityV1Response"
    ]
    assert set(ortools_capability["properties"]) == {
        "schemaVersion",
        "backendIdentity",
        "executionReady",
        "launchAdmissible",
        "supportedOrigins",
        "supportedProofVisibility",
        "executionCapabilities",
        "hardPredicateCatalog",
        "localization",
        "sha256",
    }
    assert set(ortools_capability["required"]) == set(
        ortools_capability["properties"]
    ) - {"schemaVersion"}
    assert ortools_capability["additionalProperties"] is False
    execution_capabilities = openapi["components"]["schemas"][
        "ScenarioExecutionCapabilitiesV1Response"
    ]
    assert set(execution_capabilities["properties"]) == {
        "schemaVersion",
        "combinations",
        "decisionUnavailableReasons",
        "decisionSelectorSchemas",
        "authoringCatalog",
        "phase4AcceptanceGaps",
        "sha256",
    }
    assert execution_capabilities["additionalProperties"] is False
    execution_combination = openapi["components"]["schemas"][
        "ScenarioExecutionCombinationV1Response"
    ]
    assert set(execution_combination["properties"]) == {
        "origin",
        "decisionKind",
        "mode",
        "visibility",
        "claims",
        "lineUniverses",
        "strategyClasses",
        "contractValues",
    }
    assert execution_combination["additionalProperties"] is False
    assert openapi["components"]["schemas"][
        "ScenarioPredicateDefinitionV2Response"
    ]["additionalProperties"] is False
    assert openapi["components"]["schemas"][
        "ScenarioCpSatSupportV1Response"
    ]["additionalProperties"] is False
    localization = openapi["components"]["schemas"][
        "ScenarioResearchLocalizationV1Response"
    ]
    assert set(localization["properties"]) == {
        "schemaVersion",
        "bundleRevisionId",
        "productRevisionId",
        "productProfileHash",
        "locale",
        "direction",
        "groups",
        "sha256",
    }
    assert localization["additionalProperties"] is False
    assert openapi["components"]["schemas"][
        "ScenarioResearchLocalizationGroupsV1Response"
    ]["additionalProperties"] is False

    validation_recipe = openapi["components"]["schemas"][
        "ResearchRecipeValidationResponse"
    ]["properties"]["recipe"]
    assert validation_recipe["discriminator"]["mapping"] == {
        "1": "#/components/schemas/ResearchRecipeResponse",
        "2": "#/components/schemas/ResearchRecipeV2Response",
        "3": "#/components/schemas/ResearchRecipeV3Response",
        "4": "#/components/schemas/ResearchRecipeV4Response",
    }
    stored_items = openapi["components"]["schemas"][
        "ResearchRecipeListResponse"
    ]["properties"]["items"]["items"]
    assert stored_items["discriminator"]["mapping"] == {
        "1": "#/components/schemas/StoredResearchRecipeResponse",
        "2": "#/components/schemas/StoredResearchRecipeV2Response",
        "3": "#/components/schemas/StoredResearchRecipeV3Response",
        "4": "#/components/schemas/StoredResearchRecipeV4Response",
    }
    assert openapi["components"]["schemas"][
        "ExactProjectAssertionV1Request"
    ]["properties"]["highRank"] == {
        "anyOf": [{"minimum": 0.0, "type": "integer"}, {"type": "null"}],
        "title": "Highrank",
    }

    primitive_parameters = {
        item["name"]: item
        for item in openapi["paths"]["/api/v2/rule-primitives"]["get"][
            "parameters"
        ]
    }
    assert primitive_parameters["bundleRevisionId"]["in"] == "query"
    assert primitive_parameters["bundleRevisionId"]["required"] is False


def test_render_selected_run_prepares_one_frozen_batch(monkeypatch) -> None:
    items = [
        {
            "bundleArtifact": {"uri": "cas://sha256/" + digit * 64},
            "basename": f"selected-{digit}",
            "source": {"resultCode": f"LAF-{digit * 10}"},
        }
        for digit in ("1", "2")
    ]
    calls = {}

    def prepare(run_code, **options):
        calls["prepare"] = (run_code, options)
        return {"items": items}

    def submit(prepared, *, idempotency_key, playback_evidence=False):
        calls["submit"] = (prepared, idempotency_key, playback_evidence)
        return {
            "jobId": "JOB-RENDER-SELECTED",
            "tenantId": "default",
            "kind": "render_batch_v1",
            "status": "queued",
            "priority": 0,
            "progress": 0.0,
            "request": {"items": list(prepared)},
            "result": None,
            "error": None,
            "attempts": 0,
            "maxAttempts": 1,
            "leaseOwner": None,
            "leaseUntil": None,
            "cancelRequested": False,
            "createdAt": "2026-08-23T00:00:00+00:00",
            "updatedAt": "2026-08-23T00:00:00+00:00",
        }

    monkeypatch.setattr(api.RESEARCH_RENDERS, "prepare_selected_run", prepare)
    monkeypatch.setattr(api.RENDER_BATCHES, "submit", submit)
    with TestClient(api.app) as client:
        response = client.post(
            "/api/v2/research-runs/RUN-ABCDEFGHIJKL:render-selected",
            json={
                "observerSeat": "south",
                "revealPolicy": "after_commit",
                "exportPreset": "hooked_flagship",
            },
            headers={"Idempotency-Key": "selected-render-batch"},
        )
    assert response.status_code == 202, response.text
    assert calls["prepare"] == (
        "RUN-ABCDEFGHIJKL",
        {
            "observer_seat": "south",
            "reveal_policy": "after_commit",
            "export_preset": "hooked_flagship",
        },
    )
    assert calls["submit"] == (tuple(items), "selected-render-batch", True)
    assert response.json()["request"]["items"] == items


def test_v3_research_estimate_has_the_exact_public_shape() -> None:
    revision_id = api.CONTAINER.configuration.resolve()["bundle"]["revisionId"]
    payload = {
        "schemaVersion": 3,
        "titleAr": "وصفة تقدير تجريبية",
        "configuration": {
            "schemaVersion": 1,
            "bundleRevisionId": revision_id,
        },
    }
    with TestClient(api.app) as client:
        response = client.post("/api/v2/research-recipes:estimate", json=payload)
        primitives = client.get(
            "/api/v2/rule-primitives",
            params={"bundleRevisionId": revision_id},
        )
        missing_primitives = client.get(
            "/api/v2/rule-primitives",
            params={"bundleRevisionId": "CBR-DOES-NOT-EXIST"},
        )

    assert response.status_code == 200, response.text
    value = response.json()
    assert set(value) == ESTIMATE_FIELDS
    assert value["schemaVersion"] == 1
    assert value["seedCount"] == 10
    assert value["candidateCount"] == 10
    assert value["shardCount"] == 1
    ResearchRecipeEstimateResponse.model_validate(value)
    assert primitives.status_code == 200, primitives.text
    primitive_value = primitives.json()
    selection = primitive_value["configurationSelection"]
    assert selection["bundleRevisionId"] == revision_id
    assert selection["legacyDomainV2Hash"] == primitive_value["domainHash"]
    assert missing_primitives.status_code == 422


class _BundleScopedResearchStub:
    def __init__(self, base_capabilities: dict) -> None:
        self.calls: list[tuple[bool, str | None]] = []
        self.base_capabilities = deepcopy(base_capabilities)

    def capabilities(
        self,
        *,
        include_payloads: bool,
        bundle_revision_id: str | None,
    ) -> dict:
        self.calls.append((include_payloads, bundle_revision_id))
        value = deepcopy(self.base_capabilities)
        if bundle_revision_id is None:
            value["sha256"] = "0" * 64
            value["rulePrimitives"] = [
                {**value["rulePrimitives"][0], "id": "BLR-ACTIVE"}
            ]
            value["research"].pop("configurationSelection", None)
            return value
        if bundle_revision_id == "CBR-MISSING" or bundle_revision_id == "":
            raise KeyError("configuration bundle revision not found")
        selected = {
            "CBR-A": ("a" * 64, "BLR-A"),
            "CBR-B": ("b" * 64, "BLR-B"),
            "CBR-MISMATCH": ("c" * 64, "BLR-C"),
        }.get(bundle_revision_id)
        if selected is None:
            raise KeyError("configuration bundle revision not found")
        domain_hash, primitive_id = selected
        receipt_revision = (
            "CBR-B" if bundle_revision_id == "CBR-MISMATCH" else bundle_revision_id
        )
        value["sha256"] = domain_hash
        value["rulePrimitives"] = [
            {**value["rulePrimitives"][0], "id": primitive_id}
        ]
        value["research"]["configurationSelection"] = {
            "schemaVersion": 1,
            "bundleRevisionId": receipt_revision,
            "bundleId": "balootx-product",
            "bundleVersion": "1.1.18",
            "semanticBundleHash": domain_hash,
            "effectiveRuntimeConfigHash": domain_hash,
            "contextHash": domain_hash,
            "domainHash": domain_hash,
            "domainIrHash": domain_hash,
            "legacyDomainV2Hash": domain_hash,
            "ready": True,
        }
        return value


def test_rule_primitives_are_exactly_bundle_scoped_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _BundleScopedResearchStub(
        api.RESEARCH.capabilities(include_payloads=False)
    )
    monkeypatch.setattr(api, "RESEARCH", stub)

    with TestClient(api.app) as client:
        legacy = client.get("/api/v2/rule-primitives")
        capabilities_a = client.get(
            "/api/v2/domain-capabilities",
            params={"bundleRevisionId": "CBR-A"},
        )
        selected_a = client.get(
            "/api/v2/rule-primitives", params={"bundleRevisionId": "CBR-A"}
        )
        capabilities_b = client.get(
            "/api/v2/domain-capabilities",
            params={"bundleRevisionId": "CBR-B"},
        )
        selected_b = client.get(
            "/api/v2/rule-primitives", params={"bundleRevisionId": "CBR-B"}
        )
        missing = client.get(
            "/api/v2/rule-primitives",
            params={"bundleRevisionId": "CBR-MISSING"},
        )
        empty = client.get(
            "/api/v2/rule-primitives", params={"bundleRevisionId": ""}
        )
        mismatched = client.get(
            "/api/v2/rule-primitives",
            params={"bundleRevisionId": "CBR-MISMATCH"},
        )

    assert legacy.status_code == 200
    assert set(legacy.json()) == {"schemaVersion", "domainHash", "items"}
    for capabilities, response, revision_id, domain_hash, primitive_id in (
        (capabilities_a, selected_a, "CBR-A", "a" * 64, "BLR-A"),
        (capabilities_b, selected_b, "CBR-B", "b" * 64, "BLR-B"),
    ):
        assert capabilities.status_code == 200, capabilities.text
        assert response.status_code == 200, response.text
        capability_value = capabilities.json()
        value = response.json()
        assert capability_value["sha256"] == domain_hash
        assert value["domainHash"] == domain_hash
        assert [item["id"] for item in value["items"]] == [primitive_id]
        assert value["configurationSelection"] == capability_value["research"][
            "configurationSelection"
        ]
        assert value["configurationSelection"]["bundleRevisionId"] == revision_id
        assert value["configurationSelection"]["legacyDomainV2Hash"] == domain_hash
    assert missing.status_code == 422
    assert empty.status_code == 422
    assert mismatched.status_code == 422
    assert stub.calls == [
        (False, None),
        (False, "CBR-A"),
        (False, "CBR-A"),
        (False, "CBR-B"),
        (False, "CBR-B"),
        (False, "CBR-MISSING"),
        (False, ""),
        (False, "CBR-MISMATCH"),
    ]


def test_v4_result_evidence_is_typed_in_openapi() -> None:
    schemas = api.app.openapi()["components"]["schemas"]
    result = schemas["LegacyResearchResultResponse"]
    expected = {
        "scenarioSatisfaction": "ScenarioSatisfactionV1Response",
        "constraintEvidence": "ScenarioConstraintEvidenceV1Response",
        "proofOutcome": "ScenarioProofOutcomeV1Response",
        "scenarioProjection": "ScenarioProjectionV1Response",
    }
    for field, schema_name in expected.items():
        assert result["properties"][field]["anyOf"][0] == {
            "$ref": f"#/components/schemas/{schema_name}"
        }
        assert schemas[schema_name]["additionalProperties"] is False


class _ResearchResultReadStub:
    def __init__(self, result: dict | Exception | None) -> None:
        self.result = result
        self.calls: list[str] = []

    def get_result(self, code: str):
        self.calls.append(code)
        if isinstance(self.result, Exception):
            raise self.result
        return deepcopy(self.result)


def test_operator_audit_route_is_trusted_strict_and_integrity_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, compatibility, receipt = _operator_audit_sections()
    result_code = "LAF-AUDIT00001"
    stub = _ResearchResultReadStub(
        {
            "schemaVersion": 2,
            "resultCode": result_code,
            "researchAuthority": authority,
            "runtimeCompatibility": compatibility,
            "executionReceipt": receipt,
        }
    )
    monkeypatch.setattr(api, "RESEARCH", stub)
    monkeypatch.setenv("BALOOT_ENV", "production")
    monkeypatch.setenv("BALOOT_TRUSTED_API_KEY", "operator-audit-secret")
    path = f"/api/v2/research-results/{result_code}/operator-audit"

    with TestClient(api.app) as client:
        denied = client.get(path)
        wrong = client.get(path, headers={"x-baloot-trusted-key": "wrong"})
        response = client.get(
            path,
            headers={"x-baloot-trusted-key": "operator-audit-secret"},
        )

    assert denied.status_code == 403
    assert wrong.status_code == 403
    assert stub.calls == [result_code]
    assert response.status_code == 200, response.text
    expected = {
        "schemaVersion": 1,
        "resultCode": result_code,
        "researchAuthority": authority,
        "runtimeCompatibility": compatibility,
        "executionReceipt": receipt,
    }
    assert response.json() == expected
    assert ResearchOperatorAuditV1Response.model_validate(expected).model_dump(
        mode="json"
    ) == expected

    invalid = deepcopy(expected)
    invalid["researchAuthority"]["sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        ResearchOperatorAuditV1Response.model_validate(
            {**invalid, "unknown": True}
        )
    with pytest.raises(ValidationError):
        ExecutionReceiptV1Response.model_validate(
            {**receipt, "branches": True}
        )

    stub.result = {**stub.result, "researchAuthority": invalid["researchAuthority"]}
    with TestClient(api.app) as client:
        tampered = client.get(
            path,
            headers={"x-baloot-trusted-key": "operator-audit-secret"},
        )
    assert tampered.status_code == 409
    assert tampered.json() == {
        "detail": "research result authority validation failed"
    }


def test_operator_audit_route_preserves_read_time_authority_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_code = "LAF-AUDIT00002"
    stub = _ResearchResultReadStub(RuntimeError("substituted artifact"))
    monkeypatch.setattr(api, "RESEARCH", stub)
    with TestClient(api.app) as client:
        response = client.get(
            f"/api/v2/research-results/{result_code}/operator-audit"
        )
    assert response.status_code == 409
    assert response.json() == {
        "detail": "research result authority validation failed"
    }


def test_playing_quality_audit_route_is_strict_and_preserves_validation_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_code = "LAF-QUALITYAPI"
    rows = [
        {
            "ordinal": 1,
            "kind": "play",
            "actor": "north",
            "action": "H7",
            "legalActionCount": 1,
            "actualActionOptimal": True,
            "regret": 0,
            "selectionReason": "native_actor_information_set_best_value",
            "proofMode": "bounded",
            "proofVisibility": "actor_information_set",
            "proofCompletionStatus": "bounded_complete",
            "proofScope": "supplied_worlds_only",
            "proofArtifactHash": "1" * 64,
            "actionQualityHash": "2" * 64,
            "sha256": "3" * 64,
        }
    ]
    payload = {
        "schemaVersion": 1,
        "resultCode": result_code,
        "resultHash": "4" * 64,
        "scope": "all_decisions",
        "status": "complete",
        "wholeRoundCertified": False,
        "operation": "generate",
        "qualityReceiptHash": "5" * 64,
        "completeTraceHash": "6" * 64,
        "decisionCount": 1,
        "optimalDecisionCount": 1,
        "zeroRegretDecisionCount": 1,
        "countsByKind": {"bid": 0, "escalation": 0, "project": 0, "play": 1},
        "decisions": rows,
        "sha256": "7" * 64,
    }

    class Stub:
        def get_playing_quality_audit(self, code: str):
            assert code == result_code
            return deepcopy(payload)

    monkeypatch.setattr(api, "RESEARCH", Stub())
    path = f"/api/v2/research-results/{result_code}/quality-audit"
    with TestClient(api.app) as client:
        response = client.get(path)
    assert response.status_code == 200, response.text
    assert response.json() == payload
    assert ResearchPlayingQualityAuditV1Response.model_validate(payload).model_dump(
        mode="json", exclude_none=True
    ) == payload

    payload["featuredPlayProof"] = {
        "exactStateId": "STATE-" + "A" * 20,
        "playIndex": 22,
        "actor": "north",
        "action": "H7",
        "proofArtifactHash": "1" * 64,
        "proofModelKind": "native_observed_root_outcome_v1",
        "completionStatus": "bounded_complete",
        "hiddenWorlds": {"method": "supplied_worlds_only", "count": 6, "complete": False},
    }
    with TestClient(api.app) as client:
        featured_response = client.get(path)
    assert featured_response.status_code == 200, featured_response.text
    assert featured_response.json() == payload

    payload["featuredPlayProof"].update(proofModelKind="native_complete_state_root_outcome_v1",
        completionStatus="exact_complete")
    payload["featuredPlayProof"].pop("hiddenWorlds")
    with TestClient(api.app) as client:
        complete_response = client.get(path)
    assert complete_response.status_code == 200, complete_response.text
    assert complete_response.json() == payload
    for changed in (
        {"hiddenWorlds": {"method": "supplied_worlds_only", "count": 1, "complete": False}},
        {"completionStatus": "bounded_complete"},
        {"proofModelKind": "native_observed_root_outcome_v1"},
    ):
        invalid_scope = deepcopy(payload)
        invalid_scope["featuredPlayProof"].update(changed)
        with pytest.raises(ValidationError, match="scope disagree"):
            ResearchPlayingQualityAuditV1Response.model_validate(invalid_scope)

    invalid = deepcopy(payload)
    invalid["decisionCount"] = 2
    with pytest.raises(ValidationError):
        ResearchPlayingQualityAuditV1Response.model_validate(invalid)
    invalid = deepcopy(payload)
    invalid["wholeRoundCertified"] = True
    with pytest.raises(ValidationError):
        ResearchPlayingQualityAuditV1Response.model_validate(invalid)

    payload["featuredPlayProof"].update(
        proofModelKind="actor_information_set_cegis_model_v1",
        hiddenWorlds={"method": "exhaustive_cp_sat_hidden_allocations_v1", "count": 6, "complete": True},
    )
    with TestClient(api.app) as client:
        exact_actor_response = client.get(path)
    assert exact_actor_response.status_code == 200, exact_actor_response.text
    assert exact_actor_response.json() == payload
    for changed in (
        {"hiddenWorlds": None},
        {"completionStatus": "bounded_complete"},
        {"hiddenWorlds": {"method": "supplied_worlds_only", "count": 6, "complete": False}},
        {"hiddenWorlds": {"method": "exhaustive_cp_sat_hidden_allocations_v1", "count": 6, "complete": False}},
        {"hiddenWorlds": {"method": "exhaustive_cp_sat_hidden_allocations_v1", "count": 6, "complete": True, "worldState": "private"}},
    ):
        invalid_scope = deepcopy(payload)
        invalid_scope["featuredPlayProof"].update(changed)
        with pytest.raises(ValidationError):
            ResearchPlayingQualityAuditV1Response.model_validate(invalid_scope)

    class FailedStub:
        def get_playing_quality_audit(self, code: str):
            raise ValueError("substituted receipt")

    monkeypatch.setattr(api, "RESEARCH", FailedStub())
    with TestClient(api.app) as client:
        failed = client.get(path)
    assert failed.status_code == 409
    assert failed.json() == {
        "detail": "research playing-quality audit validation failed"
    }


def test_actor_public_result_adds_only_safe_authority_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, compatibility, receipt = _operator_audit_sections()
    base = {
        "schemaVersion": 2,
        "resultCode": "LAF-PUBLIC0001",
        "proofOutcome": {"claimSatisfied": True},
        "scenarioProjection": {"exactStateAvailable": True},
    }
    monkeypatch.setattr(api, "public_scenario_result_v4", lambda _value: deepcopy(base))
    public = api._public_research_result(
        {
            "schemaVersion": 2,
            "resultCode": "LAF-PUBLIC0001",
            "researchAuthority": authority,
            "runtimeCompatibility": compatibility,
            "executionReceipt": receipt,
        }
    )
    assert public == {
        **base,
        "researchBackend": "ortools_cp_sat_v2",
        "qualificationAuthority": "cp_sat",
        "runtimeCompatibilityStatus": "not_checked",
    }
    for private in (
        "researchAuthority",
        "runtimeCompatibility",
        "executionReceipt",
        "resultHash",
        "traceHash",
        "exactStateId",
    ):
        assert private not in public


@pytest.mark.parametrize("raw_roots", [None, [], [{"action": "CA", "value": 0, "satisfies": False}]])
@pytest.mark.parametrize("mutation", [None, "owner", "proof_hash"])
def test_public_incomplete_roots_require_owned_solver_values(
    tmp_path, monkeypatch, raw_roots, mutation,
):
    from test_research_result_authority_fallbacks import _v3_fixture

    raw_proof = {"completionStatus": "proof_incomplete", "claimSatisfied": False}
    if raw_roots is not None:
        raw_proof["rootActions"] = raw_roots
    result, _recipe, documents, _campaign = _v3_fixture(
        compilation_receipt=True, backend_version="3.1.136",
        terminal_status="proof_incomplete", raw_proof=raw_proof,
    )
    result.update(runCode="RUN-ABCDEF012345", recipeCode="RCP-PROOFAUDIT")
    # Model the historical derived rows, not a new legal game or solved proof.
    result["proofOutcome"]["rootActions"] = raw_roots or [
        {"action": "CA", "value": 0, "satisfies": False},
        {"action": "C7", "value": 0, "satisfies": False},
    ]
    store = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    reference = store.put(
        json.dumps(documents["proofOutcome"], ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8"),
        media_type=api.ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES["proofOutcome"],
        name="proof.json",
    )
    reference.update(ownerKind="ortools_research_semantic_artifact_v1",
                     ownerHash=result["provenance"]["resultIdentity"]["executorSemanticHash"])
    result["artifacts"]["proofOutcome"] = reference
    if mutation == "owner":
        reference["ownerHash"] = "1" * 64
    elif mutation == "proof_hash":
        result["researchAuthority"]["proofOutcomeHash"] = "2" * 64
    original = deepcopy(result)
    monkeypatch.setattr(api, "RESEARCH", SimpleNamespace(artifact_store=store))
    if mutation is not None:
        with pytest.raises(ValueError, match="public proof artifact authority mismatch"):
            api._public_research_result(result)
    else:
        public = api._public_research_result(result)
        api.ResearchScenarioPublicResultResponse.model_validate(public)
        assert public["proofOutcome"]["rootActions"] == (raw_roots or [])
        assert public["proofOutcome"]["completionStatus"] == "proof_incomplete"
        assert public["proofOutcome"]["claimSatisfied"] is False
        assert "artifacts" not in public
        assert public["proofOutcome"]["sha256"] == api.canonical_sha256({
            key: value for key, value in public["proofOutcome"].items() if key != "sha256"
        })
    assert result == original


class _ResearchResultListStub:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def list_results(self, **filters):
        self.calls.append(filters)
        return {
            "items": [],
            "total": 0,
            "offset": filters["offset"],
            "limit": filters["limit"],
            "nextOffset": None,
            "hasMore": False,
            "groupBy": None,
            "groups": [],
        }


def test_result_query_accepts_exact_input_and_all_semantic_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _ResearchResultListStub()
    monkeypatch.setattr(api, "RESEARCH", stub)
    with TestClient(api.app) as client:
        exact = client.get(
            "/api/v2/research-results",
            params={
                "generationLane": "exact_input",
                "scenarioStatus": "structurally_infeasible",
            },
        )
        budget = client.get(
            "/api/v2/research-results",
            params={"scenarioStatus": "not_found_within_budget"},
        )
        invalid = client.get(
            "/api/v2/research-results",
            params={"scenarioStatus": "completed"},
        )
    assert exact.status_code == 200, exact.text
    assert budget.status_code == 200, budget.text
    assert invalid.status_code == 422
    assert stub.calls[0]["generation_lane"] == "exact_input"
    assert stub.calls[0]["scenario_status"] == "structurally_infeasible"
    assert stub.calls[1]["scenario_status"] == "not_found_within_budget"


def test_operator_audit_openapi_contract_is_closed() -> None:
    path = "/api/v2/research-results/{result_code}/operator-audit"
    operation = api.app.openapi()["paths"][path]["get"]
    assert operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/ResearchOperatorAuditV1Response"}
    schemas = api.app.openapi()["components"]["schemas"]
    assert set(schemas["ResearchOperatorAuditV1Response"]["properties"]) == {
        "schemaVersion",
        "resultCode",
        "researchAuthority",
        "runtimeCompatibility",
        "executionReceipt",
    }
    for name in (
        "ResearchOperatorAuditV1Response",
        "ResearchAuthorityV1Response",
        "RuntimeCompatibilityV1Response",
        "ExecutionReceiptV1Response",
    ):
        assert schemas[name]["additionalProperties"] is False

    public_result = schemas["ResearchScenarioPublicResultResponse"]
    assert public_result["properties"]["proofOutcome"] == {
        "$ref": "#/components/schemas/ScenarioActorPublicProofOutcomeV1Response"
    }
    assert public_result["properties"]["scenarioProjection"] == {
        "$ref": "#/components/schemas/ScenarioActorPublicProjectionV1Response"
    }
    assert schemas["ScenarioActorPublicProjectionV1Response"]["properties"][
        "predicateEvidenceCount"
    ]["minimum"] == 0
    assert "artifacts" not in public_result["properties"]
    assert "resultHash" not in public_result["properties"]
    assert {
        "researchBackend",
        "qualificationAuthority",
        "runtimeCompatibilityStatus",
    } <= set(public_result["properties"])
    assert {
        "researchAuthority",
        "runtimeCompatibility",
        "executionReceipt",
        "traceHash",
    }.isdisjoint(public_result["properties"])

    proof = schemas["ScenarioProofOutcomeV1Response"]
    assert proof["properties"]["continuationLines"] == {
        "$ref": "#/components/schemas/ScenarioContinuationLinesV1Response"
    }
    for proof_name in (
        "ScenarioProofOutcomeV1Response",
        "ScenarioActorPublicProofOutcomeV1Response",
    ):
        properties = schemas[proof_name]["properties"]
        assert properties["winningRootActionCount"]["anyOf"][-1] == {
            "type": "null"
        }
        assert properties["winningRootActionLowerBound"]["minimum"] == 0
        assert properties["winningRootActionUpperBound"]["anyOf"][-1] == {
            "type": "null"
        }
    assert schemas["ScenarioContinuationLinesV1Response"]["additionalProperties"] is False

    validation = schemas["ResearchRecipeValidationResponse"]
    assert validation["properties"]["capabilityPlan"]["anyOf"][0] == {
        "$ref": "#/components/schemas/ScenarioCapabilityPlanV1Response"
    }
    assert validation["properties"]["nativeCapabilityPlan"]["anyOf"][0] == {
        "$ref": "#/components/schemas/NativeCapabilityPlanV3Response"
    }
    legacy_plan = schemas["ScenarioCapabilityPlanV1Response"]
    assert legacy_plan["description"] == (
        "Legacy ScenarioCompilerV1 capability plan, not native V3 admission evidence."
    )
    assert legacy_plan["properties"]["valid"] == {
        "title": "Legacy compiler validity",
        "description": (
            "Reports frozen ScenarioCompilerV1 support, not native V3 admission. "
            "It may be false for an admissible native V3 recipe; use "
            "nativeCapabilityPlan from native validation."
        ),
        "type": "boolean",
    }
    assert "valid" in legacy_plan["required"]
    assert schemas["StoredResearchRecipeV4Response"]["properties"][
        "nativeCapabilityPlan"
    ]["anyOf"][0] == {
        "$ref": "#/components/schemas/NativeCapabilityPlanV3Response"
    }
    assert validation["properties"]["normalizedExactState"]["anyOf"][0] == {
        "$ref": "#/components/schemas/NormalizedExactStateV1Response"
    }
    estimate = schemas["ResearchRecipeEstimateResponse"]
    assert estimate["properties"]["proof"]["anyOf"][0] == {
        "$ref": "#/components/schemas/ResearchEstimateProofV2Response"
    }
    for schema_name in (
        "ScenarioCapabilityPlanV1Response",
        "ScenarioCapabilityPlanRowV1Response",
        "ScenarioDiagnosticV1Response",
        "NormalizedExactStateV1Response",
        "ResearchEstimateProofV2Response",
    ):
        assert schemas[schema_name]["additionalProperties"] is False


def test_research_result_list_response_is_closed_and_strict() -> None:
    payload = {
        "items": [],
        "total": 0,
        "offset": 0,
        "limit": 100,
        "nextOffset": None,
        "hasMore": False,
        "groupBy": None,
        "groups": [],
    }
    assert api.ResearchResultListResponse.model_validate(payload).model_dump(
        mode="json"
    ) == payload
    with pytest.raises(ValidationError):
        api.ResearchResultListResponse.model_validate({**payload, "total": "0"})
    with pytest.raises(ValidationError):
        api.ResearchResultListResponse.model_validate({**payload, "unknown": True})

    group = {"value": "proof", "count": 1}
    assert api.ResearchResultGroupCountResponse.model_validate(group).model_dump(
        mode="json"
    ) == group
    with pytest.raises(ValidationError):
        api.ResearchResultGroupCountResponse.model_validate({**group, "count": "1"})
    with pytest.raises(ValidationError):
        api.ResearchResultGroupCountResponse.model_validate(
            {**group, "unknown": True}
        )
