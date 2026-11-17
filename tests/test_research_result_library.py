from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastapi.testclient import TestClient

from baloot_platform.research.postgres_store import PostgresResearchStoreV1
from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
from baloot_platform.research.result_library import (
    RESULT_PROJECTION_VERSION,
    ResearchResultQueryV1,
    result_facets,
    result_library_projection,
    result_library_safe_payload,
    stored_result_library_projection_hash_v1,
    stored_result_payload_hash_v1,
    validate_scenario_result_authority_v1,
)
from baloot_platform.research.result_authority import (
    ExecutionReceiptV1,
    _execution_receipt,
    ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES,
    ResearchAuthorityV1,
    RuntimeCompatibilityV1,
    build_ortools_research_result_v1,
)
from baloot_platform.research.scenario_contracts import (
    ExactTracePrefixV1,
    canonical_sha256,
)
from baloot_platform.research.scenario_result import public_scenario_result_v4
from baloot_platform.research.service import ResearchFoundryServiceV1
from baloot_platform.research.store import SQLiteResearchStoreV1
from baloot_platform.service import api
from baloot_platform.service.api import ResearchScenarioPublicResultResponse


def _hash(label: str) -> str:
    return (label.encode("utf-8").hex() + "0" * 64)[:64]


@pytest.mark.parametrize("field", [
    "peakMemoryBytes", "cacheBytes", "completedVariants", "completedCandidates",
])
@pytest.mark.parametrize("value", [True, "1", 1.5, -1])
def test_executor_receipt_rejects_coerced_integer_measurements(field, value):
    with pytest.raises(ValueError, match="non-negative integer"):
        _execution_receipt({field: value}, semantic_result_hash=_hash("result"), candidate=None)


@pytest.mark.parametrize("field", ["conflicts", "branches"])
@pytest.mark.parametrize("value", [True, "1", 1.5, -1])
def test_executor_receipt_validates_each_stage_before_summing(field, value):
    candidate = {"stageReceipts": {"first": {field: value}, "second": {field: 2}}}
    with pytest.raises(ValueError, match="non-negative integer"):
        _execution_receipt({}, semantic_result_hash=_hash("result"), candidate=candidate)


def test_executor_receipt_preserves_native_measurements_and_legacy_defaults():
    measured = _execution_receipt({"cpuSeconds": 0.75, "peakMemoryBytes": 123,
        "cacheBytes": 45, "completedCandidates": 2}, semantic_result_hash=_hash("result"),
        candidate={"stageReceipts": {"deal": {"conflicts": 2, "branches": 3},
            "proof": {"conflicts": 5, "branches": 7}}})
    assert measured["cpuSeconds"] == 0.75
    assert measured["peakMemoryBytes"] == 123 and measured["cacheBytes"] == 45
    assert measured["completedVariants"] == 2
    assert measured["conflicts"] == 7 and measured["branches"] == 10
    default = _execution_receipt({}, semantic_result_hash=_hash("result"), candidate=None)
    assert default["completedVariants"] == 1 and default["branches"] == 0


def _scenario_authority_fixture() -> tuple[dict, dict, dict]:
    run_code = "RUN-AUTHORITY01"
    run_hash = _hash("authority-run")
    recipe_code = "RCP-AUTHORITY"
    recipe_hash = _hash("authority-recipe")
    scenario_hash = _hash("authority-scenario")
    exact_hash = _hash("authority-exact")
    satisfaction_hash = _hash("authority-satisfaction")
    constraint_hash = _hash("authority-constraint")
    proof_hash = _hash("authority-proof")
    proof_receipt_hash = _hash("authority-proof-receipt")
    proof_artifact_hash = _hash("authority-proof-artifact")
    bundle_hash = _hash("authority-bundle")
    rules_hash = _hash("authority-rules")
    created_at = "2026-08-23T00:00:00.000000+00:00"
    configuration = {
        "schemaVersion": 1,
        "semanticBundleHash": bundle_hash,
        "identities": {"rules": {"sha256": rules_hash}},
        "sha256": _hash("authority-configuration"),
    }
    recipe = {
        "schemaVersion": 4,
        "code": recipe_code,
        "sha256": recipe_hash,
        "configuration": configuration,
        "scenario": {
            "sha256": scenario_hash,
            "origin": {"kind": "exact_trace_prefix"},
            "constraints": {
                "op": "all",
                "children": [
                    {
                        "predicateId": "PRED-PROJECT",
                        "kind": "project.exists",
                        "version": 1,
                        "hardness": "hard",
                        "arguments": {
                            "kind": "hundred_sequence",
                            "value": 100,
                            "state": "awarded",
                        },
                    }
                ],
            },
            "outcome": {
                "predicateId": "PRED-OUTCOME",
                "kind": "outcome.buyer_contract",
                "version": 1,
                "hardness": "hard",
                "arguments": {},
            },
        },
        "proof": {
            "sha256": _hash("authority-proof-request"),
            "mode": "exact",
            "visibility": "complete_state",
            "claim": "exactly_one_winning_root_action",
            "targetTeam": "west_east",
            "utility": "round_win",
            "successThreshold": 0.0,
            "budgetProfile": "late_game_exact_v1",
        },
    }
    run = {
        "runCode": run_code,
        "runHash": run_hash,
        "recipeCode": recipe_code,
        "recipeHash": recipe_hash,
        "configurationAuthority": configuration,
        "generationLane": "exact_input",
        "createdAt": created_at,
    }
    identity = {
        "schemaVersion": 1,
        "kind": "research_scenario_result_identity_v1",
        "runCode": run_code,
        "runHash": run_hash,
        "recipeCode": recipe_code,
        "recipeHash": recipe_hash,
        "configurationAuthorityHash": canonical_sha256(configuration),
        "scenarioHash": scenario_hash,
        "exactStateHash": exact_hash,
        "scenarioSatisfactionHash": satisfaction_hash,
        "constraintEvidenceHash": constraint_hash,
        "proofOutcomeHash": proof_hash,
        "artifactHashes": {"proofOutcome": proof_artifact_hash},
    }
    result_hash = canonical_sha256(identity)
    result = {
        "schemaVersion": 2,
        "resultCode": f"LAF-{result_hash[:10].upper()}",
        "resultHash": result_hash,
        "runCode": run_code,
        "recipeCode": recipe_code,
        "traceHash": _hash("authority-state"),
        "createdAt": created_at,
        "scenarioSatisfaction": {
            "scenarioHash": scenario_hash,
            "exactStateId": "STATE-AUTHORITY",
            "status": "satisfied",
            "sha256": satisfaction_hash,
        },
        "constraintEvidence": {
            "scenarioHash": scenario_hash,
            "exactStateId": "STATE-AUTHORITY",
            "exactStateHash": exact_hash,
            "sha256": constraint_hash,
        },
        "proofOutcome": {
            "requestedMode": "exact",
            "requestedVisibility": "complete_state",
            "strategyClass": "pure_deterministic",
            "claim": "exactly_one_winning_root_action",
            "targetTeam": "west_east",
            "utility": "round_win",
            "successThreshold": 0.0,
            "horizon": None,
            "boundaryEvaluator": None,
            "completionStatus": "exact_complete",
            "rootActions": [{"action": "CA"}],
            "identities": {
                "bundleHash": bundle_hash,
                "rulesHash": rules_hash,
                "stateHash": _hash("authority-state"),
                "exactStateId": "STATE-AUTHORITY",
                "proofRequestHash": _hash("authority-proof-request"),
                "strategyClass": "pure_deterministic",
                "policyHashes": {},
            },
            "semanticSha256": proof_hash,
            "sha256": proof_receipt_hash,
        },
        "scenarioProjection": {
            "scenarioTemplateId": f"SCN-{scenario_hash[:12].upper()}",
            "origin": "exact_trace_prefix",
            "projectKinds": ["hundred_sequence"],
            "projectValues": ["100"],
            "projectStates": ["awarded"],
            "exactStateId": "STATE-AUTHORITY",
        },
        "artifacts": {"proofOutcome": {"sha256": proof_artifact_hash}},
        "truthLabel": "exact_optimal",
        "confidence": 1.0,
        "score": 100.0,
        "scoreEvidence": {"score": 100.0},
        "disposition": "qualified",
        "qualification": {
            "qualified": True,
            "failures": [],
            "observerSafe": True,
            "observerSafetyEvidence": {"passed": True},
            "decisionPointAuditComplete": True,
            "scenarioSatisfactionHash": satisfaction_hash,
            "proofOutcomeHash": proof_hash,
        },
        "classification": "scenario_v4",
        "criticalDecisionIds": ["STATE-AUTHORITY"],
        "decisionCount": 1,
        "provenance": {
            "recipeHash": recipe_hash,
            "scenarioHash": scenario_hash,
            "exactStateId": "STATE-AUTHORITY",
            "configurationAuthorityHash": canonical_sha256(configuration),
            "resultIdentity": identity,
        },
    }
    return result, run, recipe


def _resign_scenario_result(result: dict) -> None:
    identity = result["provenance"]["resultIdentity"]
    result["resultHash"] = canonical_sha256(identity)
    result["resultCode"] = f"LAF-{result['resultHash'][:10].upper()}"


def _ortools_authority_fixture(
    compatibility_callback=None,
    *,
    tactic_verified: bool | None = None,
) -> tuple[dict, dict, dict]:
    _result, run, recipe = _scenario_authority_fixture()
    run["runCode"] = "RUN-AUTHORITY001"
    recipe["scenario"]["decisionPoint"] = {"kind": "play", "selector": {}}
    if tactic_verified is not None:
        recipe["scenario"]["constraints"]["children"].append(
            {
                "predicateId": "PRED-TACTIC",
                "kind": "tactic.verified",
                "version": 1,
                "hardness": "report",
                "arguments": {
                    "kind": "large_card_sacrifice",
                    "minimumScore": 100,
                    "minimumEvidence": 1,
                    "causalRequired": True,
                },
            }
        )
    recipe["proof"]["claim"] = "selected_action_is_unique_winner"
    configuration = recipe["configuration"]
    configuration.update(
        {
            "bundleRevisionId": "CBR-AUTHORITY",
            "identities": {
                **configuration["identities"],
                "domain": {"sha256": _hash("compiled-domain-authority")},
            },
        }
    )
    run["configurationAuthority"] = copy.deepcopy(configuration)
    exact_fixture = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "examples/research-scenario-v4-ace-sacrifice-two-aces.json"
        ).read_text(encoding="utf-8")
    )
    normalized = OrToolsExactStateNormalizerV1(api.ACTIVE_BUNDLE.rules).normalize(
        ExactTracePrefixV1.build(exact_fixture["exactTracePrefix"]),
        decision_point={"kind": "play", "selector": {}},
    )
    complete_normalization = normalized.to_dict()
    complete_normalization["phase"] = "scored"
    complete_normalization["state"] = {
        **complete_normalization["state"],
        "buyer": "north",
        "declarer": "north",
        "contract": "sun",
        "purchaseRound": 2,
        "trumpSuit": None,
        "roundScore": {"winner": "west_east"},
    }
    artifact_documents = {
        "constructionModel": {"schemaVersion": 1, "kind": "construction"},
        "proofModel": {"schemaVersion": 1, "kind": "proof-model"},
        "proofGraph": {"schemaVersion": 1, "kind": "proof-graph"},
        "completeTrace": {
            "schemaVersion": 1,
            "kind": "complete-trace",
            "exactNormalization": normalized.to_dict(),
            "completeExactNormalization": complete_normalization,
            "buyer": "north",
            "contract": "sun",
            "trumpSuit": None,
            "score": {"roundWinner": "west_east"},
        },
        "proofOutcome": {
            "schemaVersion": 1,
            "completionStatus": "exact_complete",
            "claimSatisfied": True,
            "rootActions": [
                {"actionId": 0, "action": "CA", "value": 1, "satisfies": True}
            ],
        },
    }
    artifact_hashes = {
        name: canonical_sha256(document)
        for name, document in artifact_documents.items()
    }
    authority = ResearchAuthorityV1.build(
        backend_id="ortools_cp_sat_v2",
        backend_version="2.0.0",
        solver_version="9.14.6206",
        bundle_revision_id="CBR-AUTHORITY",
        bundle_hash=configuration["semanticBundleHash"],
        domain_ir_hashes={
            "cards": _hash("domain-cards"),
            "play": _hash("domain-play"),
        },
        lowering_manifest_hash=_hash("authority-lowering"),
        construction_model_hash=artifact_hashes["constructionModel"],
        proof_model_hash=artifact_hashes["proofModel"],
        graph_hash=artifact_hashes["proofGraph"],
        trace_hash=artifact_hashes["completeTrace"],
        proof_outcome_hash=artifact_hashes["proofOutcome"],
    )
    candidate_core = {
        "schemaVersion": 1,
        "origin": "exact_trace_prefix",
        "trace": artifact_documents["completeTrace"],
        "state": normalized.state,
        "noGoodVector": [],
        "stageReceipts": {
            "proof": {"branches": 7, "conflicts": 3},
        },
        "exactStateId": normalized.exact_state_id,
    }
    candidate = {**candidate_core, "sha256": canonical_sha256(candidate_core)}
    predicate_evidence = []
    if tactic_verified is not None:
        evidence_core = {
            "schemaVersion": 1,
            "predicateId": "PRED-TACTIC",
            "kind": "tactic.verified",
            "satisfied": tactic_verified,
            "evidenceExtractor": "cp_sat.registered_tactic_verified_evidence_v1",
            "loweringHash": _hash("tactic-lowering"),
            "observed": {
                "detectorId": "large_card_sacrifice",
                "detectorVersion": "2.0.0",
                "applicable": tactic_verified,
                "verified": tactic_verified,
                "contextMatches": True,
                "score": 100.0 if tactic_verified else 0.0,
                "proofHash": artifact_hashes["proofOutcome"],
            },
        }
        predicate_evidence.append(
            {**evidence_core, "sha256": canonical_sha256(evidence_core)}
        )
    predicate_core = {
        "schemaVersion": 1,
        "status": "satisfied",
        "satisfied": True,
        "evidence": predicate_evidence,
        "namedCore": [],
        "modelHash": artifact_hashes["constructionModel"],
    }
    predicate = {**predicate_core, "sha256": canonical_sha256(predicate_core)}
    executor = {
        "schemaVersion": 1,
        "status": "qualified",
        "candidate": candidate,
        "predicateEvaluation": predicate,
        "proof": artifact_documents["proofOutcome"],
        "researchAuthority": authority.to_dict(),
        "semanticHash": _hash("executor-semantic-result"),
        "executionReceipt": {
            "stageTimings": {"construction": 0.5, "proof": 0.25},
            "cpuSeconds": 0.7,
            "completedCandidates": 1,
            "machine": {"cpu": "reference"},
        },
    }
    artifact_references = {}
    for name, document in artifact_documents.items():
        payload = json.dumps(
            document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = artifact_hashes[name]
        artifact_references[name] = {
            "uri": f"cas://sha256/{digest}",
            "sha256": digest,
            "bytes": len(payload),
            "mediaType": ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES[name],
            "name": f"{name}.json",
        }

    def compatibility(result_hash: str, trace_hash: str) -> dict:
        state_projection_hash = _hash("runtime-state-projection")
        trace_projection_hash = _hash("runtime-trace-projection")
        evidence_core = {
            "schemaVersion": 1,
            "kind": "runtime_compatibility_evidence_v1",
            "semanticResultHash": result_hash,
            "bundleHash": configuration["semanticBundleHash"],
            "traceHash": trace_hash,
            "engineVersion": "4.0.0",
            "rulesHash": _hash("runtime-rules"),
            "exactStateId": normalized.exact_state_id,
            "exactStateDocumentHash": _hash("runtime-exact-state-document"),
            "declaredStateHash": _hash("runtime-declared-state"),
            "derivedStateHash": _hash("runtime-derived-state"),
            "expectedStateProjectionHash": state_projection_hash,
            "derivedStateProjectionHash": state_projection_hash,
            "expectedTraceProjectionHash": trace_projection_hash,
            "derivedTraceProjectionHash": trace_projection_hash,
            "replayStatus": "completed",
            "replayFailureHash": None,
            "failureCodes": [],
        }
        evidence = {**evidence_core, "sha256": canonical_sha256(evidence_core)}
        receipt = RuntimeCompatibilityV1.build(
            status="passed",
            engine_version="4.0.0",
            evidence_hash=evidence["sha256"],
            failure_codes=(),
            bundle_hash=configuration["semanticBundleHash"],
            trace_hash=trace_hash,
        ).to_dict()
        return {"runtimeCompatibility": receipt, "evidence": evidence}

    result = build_ortools_research_result_v1(
        run=run,
        recipe=recipe,
        executor_result=executor,
        artifact_documents=artifact_documents,
        artifact_references=artifact_references,
        runtime_compatibility_check=compatibility_callback or compatibility,
        created_at=run["createdAt"],
    )
    return result, run, recipe


def test_ortools_v5_projection_uses_catalog_categories_and_terminal_state() -> None:
    result, _run, _recipe = _ortools_authority_fixture()

    assert result["scenarioProjection"]["scenarioCategories"] == [
        "outcome",
        "playPrefix",
        "projects",
        "proof",
    ]
    assert result["scenarioProjection"]["contractOutcome"] == "lost"
    assert result["libraryProjection"]["contract"] == "sun"
    assert result["libraryProjection"]["purchaseRound"] == 2
    assert result_facets(result)["scenario_category"] == (
        "outcome",
        "playprefix",
        "projects",
        "proof",
    )
    ResearchScenarioPublicResultResponse.model_validate(
        public_scenario_result_v4(result)
    )


@pytest.mark.parametrize(
    ("verified", "expected_verified"),
    ((True, ["large_card_sacrifice"]), (False, [])),
)
def test_ortools_result_projects_requested_and_solver_verified_tactics(
    verified: bool,
    expected_verified: list[str],
) -> None:
    result, _run, _recipe = _ortools_authority_fixture(tactic_verified=verified)

    assert result["tacticalTags"] == ["large_card_sacrifice"]
    assert result["verifiedTacticalTags"] == expected_verified


@pytest.mark.parametrize(
    ("callback", "failure_code"),
    (
        (
            lambda *_args: (_ for _ in ()).throw(RuntimeError("runtime unavailable")),
            "runtime_compatibility_callback_failed",
        ),
        (
            lambda *_args: {"unexpected": True},
            "runtime_compatibility_callback_malformed",
        ),
    ),
)
def test_ortools_result_keeps_cp_qualification_when_compatibility_is_quarantined(
    callback, failure_code: str
) -> None:
    baseline, _run, _recipe = _ortools_authority_fixture()
    result, _run, _recipe = _ortools_authority_fixture(callback)

    assert result["resultHash"] == baseline["resultHash"]
    assert result["resultCode"] == baseline["resultCode"]
    assert result["qualification"] == baseline["qualification"]
    assert result["disposition"] == "qualified"
    assert result["runtimeCompatibility"]["status"] == "failed"
    assert result["runtimeCompatibility"]["failureCodes"] == [failure_code]
    assert (
        result["runtimeCompatibilityEvidence"]["semanticResultHash"]
        == result["resultHash"]
    )


def test_ortools_result_rejects_forged_minimal_passed_replay_evidence() -> None:
    result, run, recipe = _ortools_authority_fixture()
    forged = copy.deepcopy(result)
    evidence_core = {
        "schemaVersion": 1,
        "semanticResultHash": result["resultHash"],
        "bundleHash": result["researchAuthority"]["bundleHash"],
        "traceHash": result["researchAuthority"]["traceHash"],
        "failureCodes": [],
    }
    evidence = {**evidence_core, "sha256": canonical_sha256(evidence_core)}
    forged["runtimeCompatibility"] = RuntimeCompatibilityV1.build(
        status="passed",
        bundle_hash=evidence_core["bundleHash"],
        trace_hash=evidence_core["traceHash"],
        engine_version="forged-engine",
        evidence_hash=evidence["sha256"],
    ).to_dict()
    forged["runtimeCompatibilityEvidence"] = evidence

    with pytest.raises(
        RuntimeError, match="passed runtime compatibility evidence is incomplete"
    ):
        validate_scenario_result_authority_v1(forged, run=run, recipe=recipe)


def _resign_ortools_sections(result: dict) -> None:
    authority = result["researchAuthority"]
    authority["sha256"] = canonical_sha256(
        {key: value for key, value in authority.items() if key != "sha256"}
    )
    identity = result["provenance"]["resultIdentity"]
    identity["researchAuthorityHash"] = authority["sha256"]
    _resign_scenario_result(result)
    receipt = result["executionReceipt"]
    receipt["semanticResultHash"] = result["resultHash"]
    receipt["sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )


def test_v4_result_authority_binds_owner_bundle_proof_and_scenario() -> None:
    result, run, recipe = _scenario_authority_fixture()
    assert (
        validate_scenario_result_authority_v1(
            result,
            run=run,
            recipe=recipe,
        )
        == result
    )

    public_projection = copy.deepcopy(result)
    public_proof = public_projection["proofOutcome"]
    proof_artifact_hash = result["artifacts"]["proofOutcome"]["sha256"]
    public_proof["artifactSha256"] = proof_artifact_hash
    for private_identity in (
        "stateHash",
        "exactStateId",
        "proofRequestHash",
        "strategyClass",
    ):
        public_proof["identities"].pop(private_identity)
    public_proof["sha256"] = canonical_sha256(
        {key: value for key, value in public_proof.items() if key != "sha256"}
    )
    assert (
        validate_scenario_result_authority_v1(
            public_projection,
            run=run,
            recipe=recipe,
        )
        == public_projection
    )
    public_projection["proofOutcome"]["claim"] = "outcome_forced"
    with pytest.raises(RuntimeError, match="public proof integrity mismatch"):
        validate_scenario_result_authority_v1(
            public_projection,
            run=run,
            recipe=recipe,
        )

    wrong_payload_owner = copy.deepcopy(result)
    wrong_payload_owner["runCode"] = "RUN-SUBSTITUTED"
    with pytest.raises(RuntimeError, match="result owner identity mismatch"):
        validate_scenario_result_authority_v1(
            wrong_payload_owner,
            run=run,
            recipe=recipe,
        )

    downgraded = copy.deepcopy(result)
    downgraded["schemaVersion"] = "2"
    with pytest.raises(RuntimeError, match="result owner identity mismatch"):
        validate_scenario_result_authority_v1(
            downgraded,
            run=run,
            recipe=recipe,
        )

    tampered_verdict = copy.deepcopy(result)
    tampered_verdict["disposition"] = "rejected"
    with pytest.raises(RuntimeError, match="verdict authority mismatch"):
        validate_scenario_result_authority_v1(
            tampered_verdict,
            run=run,
            recipe=recipe,
        )

    wrong_owner = copy.deepcopy(result)
    wrong_owner["provenance"]["resultIdentity"]["runHash"] = _hash("other-run")
    _resign_scenario_result(wrong_owner)
    with pytest.raises(RuntimeError, match="result owner identity mismatch"):
        validate_scenario_result_authority_v1(
            wrong_owner,
            run=run,
            recipe=recipe,
        )

    wrong_bundle = copy.deepcopy(result)
    proof = wrong_bundle["proofOutcome"]
    proof["identities"]["bundleHash"] = _hash("other-bundle")
    proof["sha256"] = canonical_sha256(
        {key: value for key, value in proof.items() if key != "sha256"}
    )
    wrong_bundle["provenance"]["resultIdentity"]["proofOutcomeHash"] = proof["sha256"]
    _resign_scenario_result(wrong_bundle)
    with pytest.raises(
        RuntimeError, match="semantic identity mismatch|bundle authority mismatch"
    ):
        validate_scenario_result_authority_v1(
            wrong_bundle,
            run=run,
            recipe=recipe,
        )

    substituted_scenario = copy.deepcopy(result)
    other_scenario_hash = _hash("other-scenario")
    substituted_scenario["scenarioSatisfaction"]["scenarioHash"] = other_scenario_hash
    substituted_scenario["constraintEvidence"]["scenarioHash"] = other_scenario_hash
    substituted_scenario["provenance"]["scenarioHash"] = other_scenario_hash
    substituted_scenario["provenance"]["resultIdentity"]["scenarioHash"] = (
        other_scenario_hash
    )
    _resign_scenario_result(substituted_scenario)
    with pytest.raises(RuntimeError, match="scenario authority mismatch"):
        validate_scenario_result_authority_v1(
            substituted_scenario,
            run=run,
            recipe=recipe,
        )

    substituted_projection = copy.deepcopy(result)
    substituted_projection["scenarioProjection"]["origin"] = "search"
    substituted_projection["scenarioProjection"]["projectValues"] = ["50"]
    with pytest.raises(RuntimeError, match="scenario projection authority mismatch"):
        validate_scenario_result_authority_v1(
            substituted_projection,
            run=run,
            recipe=recipe,
        )


def test_ortools_result_authority_rejects_semantic_artifact_substitution() -> None:
    result, run, recipe = _ortools_authority_fixture()
    assert (
        validate_scenario_result_authority_v1(result, run=run, recipe=recipe) == result
    )

    wrong_run = copy.deepcopy(run)
    wrong_run["configurationAuthority"]["semanticBundleHash"] = _hash("wrong-run-bundle")
    with pytest.raises(RuntimeError, match="bundle authority mismatch"):
        validate_scenario_result_authority_v1(result, run=wrong_run, recipe=recipe)

    wrong_owner = copy.deepcopy(result)
    wrong_owner["provenance"]["resultIdentity"]["runHash"] = _hash("other-run")
    _resign_scenario_result(wrong_owner)
    wrong_owner["executionReceipt"] = ExecutionReceiptV1.build(
        semantic_result_hash=wrong_owner["resultHash"],
        stage_timings={"construction": 0.5, "proof": 0.25},
        cpu_seconds=0.7,
        peak_memory_bytes=1024,
        conflicts=3,
        branches=7,
        cache_bytes=64,
        completed_variants=1,
        machine={"cpu": "reference"},
    ).to_dict()
    with pytest.raises(RuntimeError, match="owner identity mismatch"):
        validate_scenario_result_authority_v1(wrong_owner, run=run, recipe=recipe)

    wrong_backend = copy.deepcopy(result)
    wrong_backend["researchAuthority"]["backendId"] = "legacy_hybrid_v1"
    _resign_ortools_sections(wrong_backend)
    with pytest.raises(RuntimeError, match="backendId is unsupported"):
        validate_scenario_result_authority_v1(wrong_backend, run=run, recipe=recipe)

    wrong_bundle = copy.deepcopy(result)
    wrong_bundle["researchAuthority"]["bundleHash"] = _hash("other-bundle")
    wrong_bundle["runtimeCompatibility"]["bundleHash"] = _hash("other-bundle")
    wrong_bundle["runtimeCompatibility"]["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in wrong_bundle["runtimeCompatibility"].items()
            if key != "sha256"
        }
    )
    _resign_ortools_sections(wrong_bundle)
    with pytest.raises(
        RuntimeError, match="semantic identity mismatch|bundle authority mismatch"
    ):
        validate_scenario_result_authority_v1(wrong_bundle, run=run, recipe=recipe)

    for authority_field, artifact_name in (
        ("graphHash", "proofGraph"),
        ("proofModelHash", "proofModel"),
        ("traceHash", "completeTrace"),
    ):
        substituted = copy.deepcopy(result)
        replacement = _hash(f"other-{artifact_name}")
        substituted["researchAuthority"][authority_field] = replacement
        substituted["artifacts"][artifact_name]["sha256"] = replacement
        substituted["provenance"]["resultIdentity"]["artifactHashes"][artifact_name] = (
            replacement
        )
        if authority_field == "traceHash":
            substituted["traceHash"] = replacement
            substituted["runtimeCompatibility"]["traceHash"] = replacement
            substituted["runtimeCompatibility"]["sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in substituted["runtimeCompatibility"].items()
                    if key != "sha256"
                }
            )
        _resign_ortools_sections(substituted)
        with pytest.raises(
            RuntimeError,
            match=f"{artifact_name} substitution detected|authority mismatch",
        ):
            validate_scenario_result_authority_v1(substituted, run=run, recipe=recipe)

    for artifact_name in (
        "constructionModel",
        "proofModel",
        "proofGraph",
        "completeTrace",
        "proofOutcome",
    ):
        wrong_artifact_owner = copy.deepcopy(result)
        wrong_artifact_owner["artifacts"][artifact_name]["ownerHash"] = _hash(
            "other-executor"
        )
        with pytest.raises(
            RuntimeError, match=f"{artifact_name} substitution detected"
        ):
            validate_scenario_result_authority_v1(
                wrong_artifact_owner, run=run, recipe=recipe
            )

    wrong_domain_authority = copy.deepcopy(recipe)
    wrong_domain_authority["configuration"]["identities"]["domain"]["sha256"] = _hash(
        "different-compiled-domain"
    )
    wrong_domain_run = copy.deepcopy(run)
    wrong_domain_run["configurationAuthority"] = copy.deepcopy(
        wrong_domain_authority["configuration"]
    )
    with pytest.raises(RuntimeError, match="bundle authority mismatch"):
        validate_scenario_result_authority_v1(
            result, run=wrong_domain_run, recipe=wrong_domain_authority
        )


def test_ortools_execution_measurements_do_not_change_semantic_result_identity() -> (
    None
):
    result, run, recipe = _ortools_authority_fixture()
    semantic_hash = result["resultHash"]
    result["executionReceipt"] = ExecutionReceiptV1.build(
        semantic_result_hash=semantic_hash,
        stage_timings={"construction": 4.5, "proof": 2.25},
        cpu_seconds=6.7,
        peak_memory_bytes=2048,
        conflicts=30,
        branches=70,
        cache_bytes=640,
        completed_variants=1,
        machine={"cpu": "another-reference-machine"},
    ).to_dict()
    assert (
        validate_scenario_result_authority_v1(result, run=run, recipe=recipe) == result
    )
    assert result["resultHash"] == semantic_hash


def test_ortools_result_has_strict_actor_public_projection_without_exact_state_id() -> (
    None
):
    result, _run, _recipe = _ortools_authority_fixture()

    public = public_scenario_result_v4(result)
    validated = ResearchScenarioPublicResultResponse.model_validate(public)

    assert validated.scenarioProjection.exactStateAvailable is True
    assert "exactStateId" not in public["scenarioProjection"]
    assert public["decisionReference"]["resultCode"] == result["resultCode"]


def test_result_tag_view_survives_list_and_detail_without_private_evidence(monkeypatch):
    result, _run, _recipe = _ortools_authority_fixture()
    core = {
        'schemaVersion': 1, 'tags': {'impact': ['round-flip'], 'mode': ['sun']},
        'playIndex': 24, 'qualificationEvidence': False,
        'sources': {'exactStateId': 'STATE-PRIVATE', 'normalizationHash': 'a' * 64},
        'classifier': None, 'qualityEvidenceHash': 'b' * 64,
    }
    result['scenarioProjection']['lessonTags'] = {**core, 'sha256': canonical_sha256(core)}
    expected = {key: copy.deepcopy(core[key]) for key in
                ('schemaVersion', 'tags', 'playIndex', 'qualificationEvidence')}
    expected['sha256'] = canonical_sha256(expected)
    # Transport test: the service seam supplies its already-validated stored row.
    monkeypatch.setattr(api, 'RESEARCH', SimpleNamespace(
        get_result=lambda _code: copy.deepcopy(result),
        list_results=lambda **_filters: dict(items=[copy.deepcopy(result)], total=1,
            offset=0, limit=100, nextOffset=None, hasMore=False, groupBy=None, groups=[])))
    monkeypatch.setattr(api, '_require_trusted', lambda request: None)
    with TestClient(api.app) as client:
        listed = client.get('/api/v2/research-results', params={'lessonImpact': 'round-flip'})
        detail = client.get(f"/api/v2/research-results/{result['resultCode']}")
    assert listed.status_code == detail.status_code == 200, (listed.text, detail.text)
    for response in (listed.json()['items'][0], detail.json()):
        projection = response['scenarioProjection']
        assert projection['lessonTags'] == expected
        assert projection['sha256'] == canonical_sha256({
            key: value for key, value in projection.items() if key != 'sha256'})
        assert 'STATE-PRIVATE' not in json.dumps(response)
        assert 'qualityEvidenceHash' not in json.dumps(response)


def test_absent_result_tags_preserve_projection_response_shape():
    result, _run, _recipe = _ortools_authority_fixture()
    public = public_scenario_result_v4(result)
    assert 'lessonTags' not in ResearchScenarioPublicResultResponse.model_validate(
        public).model_dump()['scenarioProjection']
    assert 'lessonTags' not in api.ScenarioProjectionV1Response.model_fields


@pytest.mark.parametrize("schema_version", [None, "2"])
def test_service_rejects_v4_result_discriminator_downgrade(schema_version) -> None:
    run = {"runCode": "RUN-V4", "recipeCode": "RCP-V4"}
    recipe = {"schemaVersion": 4, "code": "RCP-V4"}
    service = SimpleNamespace(
        store=SimpleNamespace(
            get_run=lambda _code: run,
            get_recipe=lambda _code: recipe,
        )
    )
    with pytest.raises(RuntimeError, match="schema authority mismatch"):
        ResearchFoundryServiceV1._validate_stored_result(
            service,
            {"runCode": "RUN-V4", "schemaVersion": schema_version},
        )


def _seed_store(store: SQLiteResearchStoreV1) -> str:
    recipe_code = "RCP-LIBRARY"
    run_code = "RUN-LIBRARY0001"
    store.put_recipe(
        {
            "code": recipe_code,
            "sha256": _hash("recipe"),
            "status": "published",
        }
    )
    store.create_run(
        {
            "runCode": run_code,
            "runHash": _hash("run"),
            "recipeCode": recipe_code,
            "recipeHash": _hash("recipe"),
            "status": "completed",
            "progress": 1.0,
        }
    )
    rows = (
        {
            "code": "LAF-LIBRARY01",
            "score": 90.0,
            "confidence": 0.95,
            "truth": "exact_optimal",
            "disposition": "qualified",
            "created": "2026-08-01T10:00:00Z",
            "tactics": ["only_winning_move", "kabout_creation"],
            "verified": ["only_winning_move"],
            "acting": ["south"],
            "observers": ["north", "south"],
            "contract": "hokum",
            "projects": ["hundred_sequence"],
            "lane": "ortools",
        },
        {
            "code": "LAF-LIBRARY02",
            "score": 90.0,
            "confidence": 0.85,
            "truth": "robust_best_known",
            "disposition": "qualified",
            "created": "2026-08-02T10:00:00+00:00",
            "tactics": ["small_card_big_gain"],
            "verified": [],
            "acting": ["west"],
            "observers": ["east", "west"],
            "contract": "sun",
            "projects": ["four_hundred"],
            "lane": "organic_runtime",
        },
        {
            "code": "LAF-LIBRARY03",
            "score": 70.0,
            "confidence": 0.75,
            "truth": "provisional",
            "disposition": "rejected",
            "created": "2026-08-03T10:00:00Z",
            "tactics": ["kabout_creation"],
            "verified": [],
            "acting": ["south"],
            "observers": ["north", "east"],
            "contract": "hokum",
            "projects": [],
            "lane": "ortools",
        },
    )
    for item in rows:
        store.put_result(
            {
                "resultCode": item["code"],
                "resultHash": _hash(item["code"]),
                "runCode": run_code,
                "truthLabel": item["truth"],
                "confidence": item["confidence"],
                "score": item["score"],
                "disposition": item["disposition"],
                "tacticalTags": item["tactics"],
                "verifiedTacticalTags": item["verified"],
                "interestingSeats": item["acting"],
                "libraryProjection": {
                    "schemaVersion": 1,
                    "actingSeats": item["acting"],
                    "observerSeats": item["observers"],
                    "contract": item["contract"],
                    "projectKinds": item["projects"],
                    "generationLane": item["lane"],
                },
                "provenance": {"generationLane": item["lane"]},
                "createdAt": item["created"],
            }
        )
    return run_code


def test_sqlite_result_library_filters_groups_and_pages_deterministically(tmp_path):
    store = SQLiteResearchStoreV1(tmp_path / "research.sqlite3")
    try:
        run_code = _seed_store(store)
        assert [
            item["resultCode"]
            for item in store.list_results(run_code=run_code, limit=10)["items"]
        ] == ["LAF-LIBRARY01", "LAF-LIBRARY02", "LAF-LIBRARY03"]
        cases = {
            "tactical_tag": (
                {"tactical_tag": "only_winning_move"},
                ["LAF-LIBRARY01"],
            ),
            "acting_seat": (
                {"acting_seat": "south"},
                ["LAF-LIBRARY01", "LAF-LIBRARY03"],
            ),
            "observer_seat": (
                {"observer_seat": "east"},
                ["LAF-LIBRARY02", "LAF-LIBRARY03"],
            ),
            "contract": ({"contract": "sun"}, ["LAF-LIBRARY02"]),
            "project": ({"project": "four_hundred"}, ["LAF-LIBRARY02"]),
            "generation_lane": (
                {"generation_lane": "organic_runtime"},
                ["LAF-LIBRARY02"],
            ),
        }
        for filters, expected in cases.values():
            response = store.list_results(run_code=run_code, limit=10, **filters)
            assert [item["resultCode"] for item in response["items"]] == expected

        dated = store.list_results(
            run_code=run_code,
            created_from="2026-08-02T00:00:00Z",
            created_to="2026-08-02T23:59:59Z",
            limit=10,
        )
        assert [item["resultCode"] for item in dated["items"]] == ["LAF-LIBRARY02"]

        grouped = store.list_results(
            run_code=run_code, group_by="tactical_tag", limit=1
        )
        assert grouped["groups"] == [
            {"value": "kabout_creation", "count": 2},
            {"value": "only_winning_move", "count": 1},
            {"value": "small_card_big_gain", "count": 1},
        ]
        assert grouped["hasMore"] is True
        assert grouped["nextOffset"] == 1
        second = store.list_results(
            run_code=run_code, sort="score_desc", offset=1, limit=1
        )
        assert [item["resultCode"] for item in second["items"]] == ["LAF-LIBRARY02"]
        assert second["nextOffset"] == 2
    finally:
        store.close()


def test_sqlite_foundry_projection_does_not_load_full_result_payload(tmp_path):
    store = SQLiteResearchStoreV1(tmp_path / "research.sqlite3")
    try:
        _seed_store(store)
        store._db.execute(
            "UPDATE research_results SET payload_json='not-json' WHERE result_code=?",
            ("LAF-LIBRARY01",),
        )
        store._db.commit()

        statements: list[str] = []
        store._db.set_trace_callback(statements.append)
        library = store.list_result_library_projections(limit=10)
        store._db.set_trace_callback(None)
        assert library["total"] == 3
        projection_query = next(
            statement
            for statement in statements
            if "library_projection_json" in statement
            and "FROM research_results" in statement
        )
        assert "payload_json" not in projection_query
        with pytest.raises(ValueError):
            store.get_result("LAF-LIBRARY01")
        query_plan = store._db.execute(
            """EXPLAIN QUERY PLAN SELECT library_projection_json
               FROM research_results WHERE admitted=1
               ORDER BY created_at DESC,result_code ASC LIMIT 10 OFFSET 0"""
        ).fetchall()
        assert any(
            "ix_research_results_library" in str(row["detail"]) for row in query_plan
        )
    finally:
        store.close()


def test_sqlite_projection_hash_backfill_does_not_load_full_result_payload(tmp_path):
    database = tmp_path / "research.sqlite3"
    store = SQLiteResearchStoreV1(database)
    try:
        _seed_store(store)
        store._db.execute(
            """UPDATE research_results
               SET payload_json='not-json',library_projection_hash=NULL
               WHERE result_code=?""",
            ("LAF-LIBRARY01",),
        )
        store._db.commit()
    finally:
        store.close()

    reopened = SQLiteResearchStoreV1(database)
    try:
        assert reopened.list_result_library_projections(limit=10)["total"] == 3
        with pytest.raises(ValueError):
            reopened.get_result("LAF-LIBRARY01")
    finally:
        reopened.close()


def test_sqlite_result_reads_reject_payload_and_projection_row_substitution(tmp_path):
    store = SQLiteResearchStoreV1(tmp_path / "research.sqlite3")
    try:
        _seed_store(store)
        store.create_run(
            {
                "runCode": "RUN-LIBRARY0002",
                "runHash": _hash("other-run"),
                "recipeCode": "RCP-LIBRARY",
                "recipeHash": _hash("recipe"),
                "status": "completed",
                "progress": 1.0,
            }
        )
        store._db.execute(
            "UPDATE research_results SET run_code=? WHERE result_code=?",
            ("RUN-LIBRARY0002", "LAF-LIBRARY01"),
        )
        store._db.commit()
        with pytest.raises(
            RuntimeError, match="stored research result identity mismatch"
        ):
            store.get_result("LAF-LIBRARY01")
        store._db.execute(
            "UPDATE research_results SET run_code=? WHERE result_code=?",
            ("RUN-LIBRARY0001", "LAF-LIBRARY01"),
        )
        store._db.commit()
        rows = {
            row["result_code"]: row
            for row in store._db.execute(
                """SELECT result_code,payload_json,library_projection_json
                   FROM research_results
                   WHERE result_code IN (?,?)""",
                ("LAF-LIBRARY01", "LAF-LIBRARY02"),
            ).fetchall()
        }
        store._db.execute(
            "UPDATE research_results SET payload_json=? WHERE result_code=?",
            (rows["LAF-LIBRARY02"]["payload_json"], "LAF-LIBRARY01"),
        )
        store._db.commit()
        with pytest.raises(
            RuntimeError, match="stored research result identity mismatch"
        ):
            store.get_result("LAF-LIBRARY01")
        with pytest.raises(
            RuntimeError, match="stored research result identity mismatch"
        ):
            store.list_results(limit=10)

        store._db.execute(
            """UPDATE research_results
               SET payload_json=?,library_projection_json=? WHERE result_code=?""",
            (
                rows["LAF-LIBRARY01"]["payload_json"],
                rows["LAF-LIBRARY02"]["library_projection_json"],
                "LAF-LIBRARY01",
            ),
        )
        store._db.commit()
        with pytest.raises(
            RuntimeError, match="stored research result identity mismatch"
        ):
            store.list_result_library_projections(limit=10)

        tampered_projection = json.loads(
            rows["LAF-LIBRARY02"]["library_projection_json"]
        )
        tampered_projection.update(
            {
                "resultCode": "LAF-LIBRARY01",
                "resultHash": _hash("LAF-LIBRARY01"),
                "runCode": "RUN-LIBRARY0001",
                "confidence": 0.01,
            }
        )
        store._db.execute(
            "UPDATE research_results SET library_projection_json=? WHERE result_code=?",
            (json.dumps(tampered_projection), "LAF-LIBRARY01"),
        )
        store._db.commit()
        with pytest.raises(RuntimeError, match="library projection mismatch"):
            store.list_result_library_projections(limit=10)
    finally:
        store.close()


def test_sqlite_result_record_hash_rejects_coherent_companion_substitution(tmp_path):
    store = SQLiteResearchStoreV1(tmp_path / "research-record-hash.sqlite3")
    try:
        failed, run, recipe = _ortools_authority_fixture(
            lambda *_args: (_ for _ in ()).throw(RuntimeError("runtime unavailable"))
        )
        passed, _other_run, _other_recipe = _ortools_authority_fixture()
        store.put_recipe({**recipe, "status": "published"})
        store.create_run({**run, "status": "completed", "progress": 1.0})
        store.put_result(failed)

        promoted = copy.deepcopy(failed)
        for field in (
            "runtimeCompatibility",
            "runtimeCompatibilityEvidence",
            "executionReceipt",
        ):
            promoted[field] = copy.deepcopy(passed[field])
        promoted["libraryProjection"]["contract"] = "hokum"
        assert promoted["resultHash"] == failed["resultHash"]
        store._db.execute(
            "UPDATE research_results SET payload_json=? WHERE result_code=?",
            (json.dumps(promoted), failed["resultCode"]),
        )
        store._db.commit()

        with pytest.raises(RuntimeError, match="payload hash mismatch"):
            store.get_result(failed["resultCode"])
        with pytest.raises(RuntimeError, match="payload hash mismatch"):
            store.list_results(limit=10)
        assert store.list_result_library_projections(limit=10)["total"] == 1
    finally:
        store.close()


def test_postgres_result_read_rejects_payload_from_another_storage_row() -> None:
    substituted = {
        "resultCode": "LAF-OTHER",
        "resultHash": _hash("other"),
        "runCode": "RUN-EXPECTED",
    }

    class Result:
        def fetchone(self):
            return {
                "result_code": "LAF-EXPECTED",
                "result_hash": _hash("expected"),
                "run_code": "RUN-EXPECTED",
                "payload_json": substituted,
                "payload_hash": stored_result_payload_hash_v1(substituted),
            }

    class Connection:
        def execute(self, statement, _parameters=None):
            return None if "set_config" in statement else Result()

    class Checkout:
        def __enter__(self):
            return Connection()

        def __exit__(self, *_args):
            return False

    class Pool:
        def connection(self):
            return Checkout()

    store = object.__new__(PostgresResearchStoreV1)
    store.pool = Pool()
    with pytest.raises(RuntimeError, match="stored research result identity mismatch"):
        store.get_result("LAF-EXPECTED")
    substituted.update(
        {
            "resultCode": "LAF-EXPECTED",
            "resultHash": _hash("expected"),
            "runCode": "RUN-OTHER",
        }
    )
    with pytest.raises(RuntimeError, match="stored research result identity mismatch"):
        store.get_result("LAF-EXPECTED")


@pytest.mark.parametrize(
    "read",
    [
        lambda store: store.list_results(limit=10),
        lambda store: store.list_result_library_projections(limit=10),
    ],
)
def test_postgres_result_lists_reject_cross_row_substitution(read) -> None:
    row = {
        "result_code": "LAF-EXPECTED",
        "result_hash": _hash("expected"),
        "run_code": "RUN-EXPECTED",
        "payload_json": {
            "resultCode": "LAF-OTHER",
            "resultHash": _hash("other"),
            "runCode": "RUN-EXPECTED",
        },
        "library_projection_json": {
            "resultCode": "LAF-OTHER",
            "resultHash": _hash("other"),
            "runCode": "RUN-EXPECTED",
        },
        "created_at": "2026-08-23T00:00:00+00:00",
    }
    row["payload_hash"] = stored_result_payload_hash_v1(row["payload_json"])
    row["library_projection_hash"] = stored_result_library_projection_hash_v1(
        row["library_projection_json"], payload_hash=row["payload_hash"]
    )

    class Result:
        def __init__(self, *, one=None, many=()):
            self.one = one
            self.many = list(many)

        def fetchone(self):
            return self.one

        def fetchall(self):
            return self.many

    class Connection:
        def execute(self, statement, _parameters=None):
            if "set_config" in statement:
                return Result()
            if "COUNT(*)" in statement:
                return Result(one={"count": 1})
            return Result(many=[row])

    class Checkout:
        def __enter__(self):
            return Connection()

        def __exit__(self, *_args):
            return False

    class Pool:
        def connection(self):
            return Checkout()

    store = object.__new__(PostgresResearchStoreV1)
    store.pool = Pool()
    with pytest.raises(RuntimeError, match="stored research result identity mismatch"):
        read(store)


def test_postgres_projection_read_rejects_same_identity_field_tampering() -> None:
    created_at = "2026-08-23T00:00:00+00:00"
    payload = {
        "resultCode": "LAF-EXPECTED",
        "resultHash": _hash("expected"),
        "runCode": "RUN-EXPECTED",
        "truthLabel": "exact_optimal",
        "confidence": 1.0,
        "score": 100.0,
        "disposition": "qualified",
        "createdAt": created_at,
    }
    projection = result_library_projection(payload, created_at=created_at)
    projection_hash = stored_result_library_projection_hash_v1(
        projection, payload_hash=stored_result_payload_hash_v1(payload)
    )
    projection["confidence"] = 0.01
    row = {
        "result_code": payload["resultCode"],
        "result_hash": payload["resultHash"],
        "run_code": payload["runCode"],
        "payload_json": payload,
        "payload_hash": stored_result_payload_hash_v1(payload),
        "library_projection_json": projection,
        "library_projection_hash": projection_hash,
        "created_at": created_at,
    }

    class Result:
        def __init__(self, *, one=None, many=()):
            self.one = one
            self.many = list(many)

        def fetchone(self):
            return self.one

        def fetchall(self):
            return self.many

    class Connection:
        def execute(self, statement, _parameters=None):
            if "set_config" in statement:
                return Result()
            if "COUNT(*)" in statement:
                return Result(one={"count": 1})
            return Result(many=[row])

    class Checkout:
        def __enter__(self):
            return Connection()

        def __exit__(self, *_args):
            return False

    class Pool:
        def connection(self):
            return Checkout()

    store = object.__new__(PostgresResearchStoreV1)
    store.pool = Pool()
    with pytest.raises(RuntimeError, match="library projection mismatch"):
        store.list_result_library_projections(limit=10)


def test_postgres_result_read_rejects_coherent_payload_substitution() -> None:
    payload = {
        "resultCode": "LAF-EXPECTED",
        "resultHash": _hash("expected"),
        "runCode": "RUN-EXPECTED",
        "libraryProjection": {"contract": "sun"},
    }
    substituted = copy.deepcopy(payload)
    substituted["libraryProjection"]["contract"] = "hokum"
    row = {
        "result_code": payload["resultCode"],
        "result_hash": payload["resultHash"],
        "run_code": payload["runCode"],
        "payload_json": substituted,
        "payload_hash": stored_result_payload_hash_v1(payload),
    }

    class Result:
        def fetchone(self):
            return row

    class Connection:
        def execute(self, statement, _parameters=None):
            return None if "set_config" in statement else Result()

    class Checkout:
        def __enter__(self):
            return Connection()

        def __exit__(self, *_args):
            return False

    class Pool:
        def connection(self):
            return Checkout()

    store = object.__new__(PostgresResearchStoreV1)
    store.pool = Pool()
    with pytest.raises(RuntimeError, match="payload hash mismatch"):
        store.get_result(payload["resultCode"])


def test_result_projection_and_query_contract_are_shared_by_both_stores():
    payload = {
        "tacticalTags": ["Only_Winning_Move"],
        "verifiedTacticalTags": ["only_winning_move", "kabout_creation"],
        "interestingSeats": ["South"],
        "libraryProjection": {
            "actingSeats": ["south"],
            "observerSeats": ["north", "south"],
            "contract": "Hokum",
            "projectKinds": ["hundred_sequence"],
            "generationLane": "ortools",
        },
    }
    assert result_facets(payload) == {
        "tactical_tag": ("kabout_creation", "only_winning_move"),
        "acting_seat": ("south",),
        "observer_seat": ("north", "south"),
        "contract": ("hokum",),
        "project": ("hundred_sequence",),
        "generation_lane": ("ortools",),
    }
    assert inspect.signature(SQLiteResearchStoreV1.list_results) == inspect.signature(
        PostgresResearchStoreV1.list_results
    )
    query = ResearchResultQueryV1(
        tactical_tag="Only_Winning_Move",
        acting_seat="South",
        created_from="2026-08-01T00:00:00Z",
        created_to="2026-08-02T00:00:00+00:00",
        group_by="contract",
    )
    assert query.facet_filters == (
        ("tactical_tag", "only_winning_move"),
        ("acting_seat", "south"),
    )
    assert query.created_from == "2026-08-01T00:00:00.000000+00:00"


def test_v5_authority_compatibility_and_collection_facets_are_evidence_derived():
    payload = {
        "runCode": "RUN-AUTHORITY01",
        "researchAuthority": {
            "backendId": "ortools_cp_sat_v2",
            "qualificationAuthority": "cp_sat",
            "bundleRevisionId": "CBR-AUTHORITY01",
        },
        "runtimeCompatibility": {"status": "passed"},
        "scenarioProjection": {
            "exactStateId": "STATE-1234567890ABCDEF1234",
            "scenarioCategories": ["play", "proof", "play"],
        },
    }

    facets = result_facets(payload)
    assert facets["research_backend"] == ("ortools_cp_sat_v2",)
    assert facets["qualification_authority"] == ("cp_sat",)
    assert facets["runtime_compatibility_status"] == ("passed",)
    assert facets["configuration_bundle"] == ("cbr-authority01",)
    assert facets["exact_state_id"] == ("state-1234567890abcdef1234",)
    assert facets["scenario_category"] == ("play", "proof")
    assert facets["variant_collection"] == ("run-authority01",)

    legacy = result_facets({"schemaVersion": 2})
    assert not {
        "research_backend",
        "qualification_authority",
        "runtime_compatibility_status",
        "configuration_bundle",
        "variant_collection",
    } & set(legacy)

    query = ResearchResultQueryV1(
        research_backend="ORtools_CP_SAT_V2",
        qualification_authority="CP_SAT",
        runtime_compatibility_status="PASSED",
        configuration_bundle="CBR-AUTHORITY01",
        exact_state_id="STATE-1234567890ABCDEF1234",
        scenario_category="PLAY",
        variant_collection="RUN-AUTHORITY01",
    )
    assert query.facet_filters == (
        ("research_backend", "ortools_cp_sat_v2"),
        ("qualification_authority", "cp_sat"),
        ("runtime_compatibility_status", "passed"),
        ("configuration_bundle", "cbr-authority01"),
        ("exact_state_id", "state-1234567890abcdef1234"),
        ("scenario_category", "play"),
        ("variant_collection", "run-authority01"),
    )


def test_sqlite_v4_projection_rebuild_is_versioned_and_payload_preserving(tmp_path):
    database = tmp_path / "research-v4-projection.sqlite3"
    store = SQLiteResearchStoreV1(database)
    run_code = _seed_store(store)
    result_code = "LAF-SCENARIOV4"
    payload = {
        "resultCode": result_code,
        "resultHash": _hash(result_code),
        "runCode": run_code,
        "truthLabel": "exact_optimal",
        "confidence": 1.0,
        "score": 100.0,
        "disposition": "qualified",
        "scenarioProjection": {
            "scenarioTemplateId": "sun_hundred_no_salvage_v1",
            "origin": "exact_trace_prefix",
            "status": "satisfied",
            "decisionKind": "project",
            "exactStateId": "STATE-1234567890ABCDEF1234",
            "projectKinds": ["hundred_sequence"],
            "projectValues": [100],
            "projectStates": ["awarded"],
            "scenarioCategories": ["projects", "proof"],
        },
        "researchAuthority": {
            "backendId": "ortools_cp_sat_v2",
            "qualificationAuthority": "cp_sat",
            "bundleRevisionId": "CBR-PROJECTION05",
        },
        "runtimeCompatibility": {"status": "passed"},
        "proofOutcome": {
            "requestedMode": "policy_conditioned",
            "achievedMode": "policy_conditioned",
            "completionStatus": "policy_complete",
            "achievedVisibility": "complete_state",
            "targetTeam": "north_south",
            "winningRootActionCount": 0,
        },
    }
    store.put_result(payload)
    original = store.get_result(result_code)
    store._db.execute(
        "DELETE FROM research_result_facets WHERE result_code=?",
        (result_code,),
    )
    store._db.execute(
        "UPDATE research_results SET projection_version=2,payload_hash=NULL WHERE result_code=?",
        (result_code,),
    )
    store._db.commit()
    store.close()

    rebuilt = SQLiteResearchStoreV1(database)
    try:
        row = rebuilt._db.execute(
            "SELECT projection_version,payload_hash FROM research_results WHERE result_code=?",
            (result_code,),
        ).fetchone()
        assert row["projection_version"] == RESULT_PROJECTION_VERSION == 5
        assert row["payload_hash"] == stored_result_payload_hash_v1(payload)
        assert rebuilt.get_result(result_code) == original == payload
        assert [
            item["resultCode"]
            for item in rebuilt.list_results(
                scenario_template="sun_hundred_no_salvage_v1",
                decision_kind="project",
                project_value="100",
                exact_state_available="true",
                limit=10,
            )["items"]
        ] == [result_code]
        assert [
            item["resultCode"]
            for item in rebuilt.list_results(
                research_backend="ortools_cp_sat_v2",
                qualification_authority="cp_sat",
                runtime_compatibility_status="passed",
                configuration_bundle="CBR-PROJECTION05",
                exact_state_id="STATE-1234567890ABCDEF1234",
                scenario_category="proof",
                variant_collection=run_code,
                limit=10,
            )["items"]
        ] == [result_code]
    finally:
        rebuilt.close()


def test_sqlite_api_filters_all_v5_authority_facets_and_groups_categories(
    tmp_path, monkeypatch
):
    store = SQLiteResearchStoreV1(tmp_path / "research-v5-api.sqlite3")
    result, run, recipe = _ortools_authority_fixture()
    store.put_recipe({**recipe, "status": "published"})
    store.create_run({**run, "status": "completed", "progress": 1.0})
    store.put_result(result)

    class SQLiteResultLibrary:
        @staticmethod
        def list_results(**filters):
            return store.list_results(**filters)

    category = "proof"
    monkeypatch.setattr(api, "RESEARCH", SQLiteResultLibrary())
    try:
        with TestClient(api.app) as client:
            response = client.get(
                "/api/v2/research-results",
                params={
                    "researchBackend": result["researchAuthority"]["backendId"],
                    "qualificationAuthority": result["researchAuthority"][
                        "qualificationAuthority"
                    ],
                    "runtimeCompatibilityStatus": result["runtimeCompatibility"][
                        "status"
                    ],
                    "configurationBundle": result["researchAuthority"][
                        "bundleRevisionId"
                    ],
                    "exactStateId": result["scenarioProjection"]["exactStateId"],
                    "scenarioCategory": category,
                    "variantCollection": result["runCode"],
                    "groupBy": "scenario_category",
                },
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["total"] == 1
        assert [item["resultCode"] for item in body["items"]] == [result["resultCode"]]
        assert body["groupBy"] == "scenario_category"
        assert body["groups"] == [
            {"value": value, "count": 1}
            for value in ("outcome", "playprefix", "projects", "proof")
        ]
    finally:
        store.close()


def test_legacy_v4_library_projection_fails_observer_safety_closed(tmp_path):
    database = tmp_path / "research-v4-legacy-observer.sqlite3"
    store = SQLiteResearchStoreV1(database)
    try:
        run_code = _seed_store(store)
        result_code = "LAF-LEGACYV4"
        payload = {
            "schemaVersion": 2,
            "resultCode": result_code,
            "resultHash": _hash(result_code),
            "runCode": run_code,
            "truthLabel": "exact_optimal",
            "confidence": 1.0,
            "score": 100.0,
            "scoreEvidence": {"family": "scenario_v4", "score": 100.0},
            "disposition": "qualified",
            "qualification": {
                "qualified": True,
                "failures": [],
                "observerSafe": True,
            },
            "provenance": {"resultIdentity": {}},
            "createdAt": "2026-08-04T00:00:00Z",
        }
        safe = result_library_safe_payload(payload)
        assert safe["disposition"] == "rejected"
        assert safe["score"] == 100.0
        assert safe["qualification"]["qualified"] is False
        assert safe["qualification"]["legacyObserverSafetyUnverified"] is True

        store.put_result(payload)
        row = store._db.execute(
            """SELECT disposition,payload_json,library_projection_json
               FROM research_results WHERE result_code=?""",
            (result_code,),
        ).fetchone()
        assert row["disposition"] == "rejected"
        assert json.loads(row["payload_json"]) == payload
        assert json.loads(row["library_projection_json"])["disposition"] == "rejected"
        assert store.list_results(disposition="qualified", limit=10)["total"] == 2
        rejected = store.list_results(disposition="rejected", limit=10)
        legacy = next(
            item for item in rejected["items"] if item["resultCode"] == result_code
        )
        assert legacy["qualification"]["observerSafe"] is False
        assert legacy["score"] == 100.0

        store._db.execute(
            """UPDATE research_results
               SET disposition='qualified',projection_version=0,
                   library_projection_json=NULL WHERE result_code=?""",
            (result_code,),
        )
        store._db.commit()
    finally:
        store.close()

    rebuilt = SQLiteResearchStoreV1(database)
    try:
        row = rebuilt._db.execute(
            """SELECT disposition,projection_version,library_projection_json
               FROM research_results WHERE result_code=?""",
            (result_code,),
        ).fetchone()
        assert row["disposition"] == "rejected"
        assert row["projection_version"] == RESULT_PROJECTION_VERSION
        assert json.loads(row["library_projection_json"])["disposition"] == "rejected"
    finally:
        rebuilt.close()


def test_research_result_library_api_is_typed_and_forwards_server_query(monkeypatch):
    from baloot_platform.service import api

    class ResultLibraryCapture:
        def __init__(self) -> None:
            self.filters = None

        def list_results(self, **filters):
            self.filters = filters
            return {
                "items": [],
                "total": 0,
                "offset": filters["offset"],
                "limit": filters["limit"],
                "nextOffset": None,
                "hasMore": False,
                "groupBy": filters["group_by"],
                "groups": [],
            }

    capture = ResultLibraryCapture()
    monkeypatch.setattr(api, "RESEARCH", capture)
    with TestClient(api.app) as client:
        response = client.get(
            "/api/v2/research-results",
            params={
                "tacticalTag": "only_winning_move",
                "actingSeat": "south",
                "observerSeat": "north",
                "contract": "hokum",
                "project": "hundred_sequence",
                "generationLane": "ortools",
                "researchBackend": "ortools_cp_sat_v2",
                "qualificationAuthority": "cp_sat",
                "runtimeCompatibilityStatus": "passed",
                "configurationBundle": "CBR-PROJECTION05",
                "exactStateId": "STATE-1234567890ABCDEF1234",
                "scenarioCategory": "proof",
                "variantCollection": "RUN-ABCDEFGHIJKL",
                "createdFrom": "2026-08-01T00:00:00Z",
                "createdTo": "2026-08-02T00:00:00Z",
                "groupBy": "scenario_category",
                "sort": "oldest",
                "offset": 10,
                "limit": 25,
            },
        )
        assert response.status_code == 200, response.text
        assert capture.filters == {
            "run_code": None,
            "truth_label": None,
            "disposition": None,
            "minimum_score": None,
            "search": None,
            "tag_query": {},
            "tactical_tag": "only_winning_move",
            "acting_seat": "south",
            "observer_seat": "north",
            "contract": "hokum",
            "project": "hundred_sequence",
            "generation_lane": "ortools",
            "research_backend": "ortools_cp_sat_v2",
            "qualification_authority": "cp_sat",
            "runtime_compatibility_status": "passed",
            "configuration_bundle": "CBR-PROJECTION05",
            "exact_state_id": "STATE-1234567890ABCDEF1234",
            "scenario_category": "proof",
            "variant_collection": "RUN-ABCDEFGHIJKL",
            "created_from": "2026-08-01T00:00:00+00:00",
            "created_to": "2026-08-02T00:00:00+00:00",
            "group_by": "scenario_category",
            "sort": "oldest",
            "offset": 10,
            "limit": 25,
        }
        operation = client.get("/openapi.json").json()["paths"][
            "/api/v2/research-results"
        ]["get"]
        names = {item["name"] for item in operation["parameters"]}
        assert {
            "tacticalTag",
            "actingSeat",
            "observerSeat",
            "contract",
            "project",
            "generationLane",
            "researchBackend",
            "qualificationAuthority",
            "runtimeCompatibilityStatus",
            "configurationBundle",
            "exactStateId",
            "scenarioCategory",
            "variantCollection",
            "createdFrom",
            "createdTo",
            "groupBy",
            "sort",
            "offset",
            "limit",
        } <= names
        parameters = {item["name"]: item for item in operation["parameters"]}
        group_values = set(parameters["groupBy"]["schema"]["anyOf"][0]["enum"])
        assert {
            "research_backend",
            "qualification_authority",
            "runtime_compatibility_status",
            "configuration_bundle",
            "exact_state_id",
            "scenario_category",
            "variant_collection",
        } <= group_values
        schema = client.get("/openapi.json").json()["components"]["schemas"][
            "ResearchResultListResponse"
        ]
        assert {"nextOffset", "hasMore", "groupBy", "groups"} <= set(
            schema["properties"]
        )
        proof_schema = client.get("/openapi.json").json()["components"]["schemas"][
            "ScenarioProofOutcomeV1Response"
        ]
        assert "artifactSha256" in proof_schema["properties"]


def test_postgres_result_facets_use_the_psycopg_cursor_batch_api() -> None:
    class Cursor:
        def __init__(self) -> None:
            self.statement = None
            self.rows = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def executemany(self, statement, rows) -> None:
            self.statement = statement
            self.rows = list(rows)

    class Connection:
        def __init__(self) -> None:
            self.deleted = None
            self.batch = Cursor()

        def execute(self, statement, parameters):
            self.deleted = (statement, parameters)

        def cursor(self) -> Cursor:
            return self.batch

    connection = Connection()
    PostgresResearchStoreV1._replace_result_facets(
        connection,
        "LAF-V4-FACETS",
        {
            "scenarioProjection": {
                "origin": "exact_trace_prefix",
                "status": "satisfied",
                "decisionKind": "play",
                "projectKinds": ["hundred_sequence"],
                "exactStateAvailable": True,
            }
        },
    )

    assert "DELETE FROM research.result_facets" in connection.deleted[0]
    assert connection.deleted[1] == ("LAF-V4-FACETS",)
    assert "INSERT INTO research.result_facets" in connection.batch.statement
    assert (
        "LAF-V4-FACETS",
        "scenario_origin",
        "exact_trace_prefix",
    ) in connection.batch.rows
    assert ("LAF-V4-FACETS", "decision_kind", "play") in connection.batch.rows


def test_postgres_v5_facets_match_sqlite_projection_for_real_ortools_result() -> None:
    class Cursor:
        rows = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def executemany(self, _statement, rows) -> None:
            self.rows = list(rows)

    class Connection:
        def __init__(self) -> None:
            self.batch = Cursor()

        @staticmethod
        def execute(_statement, _parameters):
            return None

        def cursor(self) -> Cursor:
            return self.batch

    result, _run, _recipe = _ortools_authority_fixture()
    connection = Connection()
    PostgresResearchStoreV1._replace_result_facets(
        connection, result["resultCode"], result
    )
    expected = {
        (result["resultCode"], kind, value)
        for kind, values in result_facets(result).items()
        for value in values
    }
    assert set(connection.batch.rows or ()) == expected
