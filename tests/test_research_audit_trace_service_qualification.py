from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from baloot_platform.challenge.analysis_pipeline import TraceAnalysisProfileV1
from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.core.enums import Seat
from baloot_platform.research.audit_trace import ResearchAuditTraceRecordV1

from research_audit_trace_helpers import (
    _Harness,
    _audit_items,
    _claim_case,
    _hash,
    _payload,
    _result_items,
)
from research_audit_trace_helpers import service_harness as service_harness


def test_late_qualification_rejection_reuses_retained_complete_trace_reference(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    run, claim, _ = _claim_case(
        service_harness,
        monkeypatch,
        label="late-qualification",
        payload=_payload(
            research,
            label="late-qualification",
            lane="organic_runtime",
        ),
    )
    monkeypatch.setattr(research, "_semantic_mismatches", lambda *_a, **_k: [])
    qualification = research._qualification

    def reject_late(*args: Any, **kwargs: Any) -> dict[str, Any]:
        value = qualification(*args, **kwargs)
        return {
            **value,
            "qualified": False,
            "truthLabel": "rejected",
            "failures": [*value["failures"], "service_fixture_late_gate"],
        }

    monkeypatch.setattr(research, "_qualification", reject_late)

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    records = _audit_items(research, run["runCode"])
    results = _result_items(research, run["runCode"])
    assert len(records) == len(results) == 1
    record = records[0]
    result = results[0]
    assert record["disposition"] == result["disposition"] == "rejected"
    assert "stage_content_qualification" in record["failureCategories"]
    assert "service_fixture_late_gate" in record["failureCategories"]
    trace_reference = result["artifacts"]["completeTrace"]
    assert {
        key: value
        for key, value in trace_reference.items()
        if key not in {"ownerKind", "ownerHash"}
    } == record["completeTrace"]
    assert trace_reference["ownerKind"] == "complete_trace_v1"
    assert trace_reference["ownerHash"] == result["traceHash"]
    assert result["artifacts"]["analysis"]["ownerKind"] == "trace_analysis_v1"
    assert result["artifacts"]["auditAnalysis"]["ownerKind"] == (
        "trace_analysis_v1"
    )
    assert result["artifacts"]["decisionAudit"]["ownerKind"] == (
        "decision_audit_v2"
    )
    expected_owned_reference_keys = {
        "uri",
        "sha256",
        "bytes",
        "mediaType",
        "name",
        "ownerKind",
        "ownerHash",
    }
    for artifact_name in (
        "completeTrace",
        "analysis",
        "auditAnalysis",
        "decisionAudit",
    ):
        assert set(result["artifacts"][artifact_name]) == (
            expected_owned_reference_keys
        )

    analysis_document = json.loads(
        research.artifact_store.get(result["artifacts"]["analysis"]["uri"])
    )
    audit_analysis_document = json.loads(
        research.artifact_store.get(result["artifacts"]["auditAnalysis"]["uri"])
    )
    audit_document = json.loads(
        research.artifact_store.get(result["artifacts"]["decisionAudit"]["uri"])
    )
    assert result["artifacts"]["analysis"]["ownerHash"] == analysis_document[
        "sha256"
    ]
    assert result["artifacts"]["auditAnalysis"]["ownerHash"] == (
        audit_analysis_document["sha256"]
    )
    assert result["provenance"]["auditAnalysisHash"] == audit_analysis_document[
        "sha256"
    ]
    assert result["artifacts"]["decisionAudit"]["ownerHash"] == audit_document[
        "sha256"
    ]
    decisions = research.get_decisions(result["resultCode"])
    assert decisions == audit_document["decisions"]
    admitted_page = research.audit_traces.list_records(
        run_code=result["runCode"],
        offset=0,
        limit=500,
    )
    retained_occurrence = next(
        item
        for item in admitted_page["items"]
        if item["traceHash"] == result["traceHash"]
        and item["source"]["candidateHash"]
        == result["provenance"]["candidateHash"]
    )
    later_occurrence = copy.deepcopy(retained_occurrence)
    later_occurrence["source"]["seed"] = int(
        later_occurrence["source"]["seed"]
    ) + 1
    later_occurrence["source"]["candidateHash"] = _hash(
        "later-same-trace-candidate"
    )
    later_occurrence["source"]["candidateOrdinal"] = int(
        later_occurrence["source"]["candidateOrdinal"]
    ) + 1
    later_occurrence["createdAt"] = "2026-08-10T01:00:00+00:00"
    later_occurrence["updatedAt"] = "2026-08-10T01:00:00+00:00"
    later_occurrence["recordHash"] = canonical_sha256(
        {
            "schemaVersion": 1,
            "traceHash": later_occurrence["traceHash"],
            "source": later_occurrence["source"],
        }
    )
    later_occurrence["auditTraceCode"] = (
        f"LAF-AUD-{later_occurrence['recordHash'][:10].upper()}"
    )
    later_occurrence = ResearchAuditTraceRecordV1.build(
        later_occurrence
    ).to_dict()
    with monkeypatch.context() as scoped:
        scoped.setattr(
            research.audit_traces,
            "list_records",
            lambda **_kwargs: {
                **copy.deepcopy(admitted_page),
                "items": [
                    copy.deepcopy(later_occurrence),
                    *copy.deepcopy(admitted_page["items"]),
                ],
                "nextOffset": None,
            },
        )
        assert research.get_decisions(result["resultCode"]) == decisions
    selected = next(
        item
        for item in decisions
        if item["decisionKind"] == "play" and item["forced"] is False
    )
    prepared = service_harness.container.research_renders.prepare(
        result["resultCode"],
        decision_ordinal=int(selected["ordinal"]),
        export_preset="hooked_flagship",
    )
    source = prepared["source"]
    assert source["resultHash"] == result["resultHash"]
    assert source["traceHash"] == result["traceHash"]
    assert source["analysisHash"] == analysis_document["sha256"]
    assert source["auditHash"] == audit_document["sha256"]
    assert source["analysisProfileHash"] == analysis_document["profileHash"]
    assert source["decisionId"] == selected["decisionId"]
    assert source["exactStateId"] == selected["decisionId"]
    serialized_source = json.dumps(source, sort_keys=True)
    assert "constructionObjective" not in serialized_source
    assert "structuralObjective" not in serialized_source
    assert "solverObjective" not in serialized_source

    bundle_document = json.loads(
        research.artifact_store.get(prepared["bundleArtifact"]["uri"])
    )
    public_lifecycle = next(
        layer["content"]
        for scene in bundle_document["scenes"]
        if scene["stage"] == "continuation"
        for layer in scene["layers"]
        if layer["id"] == "public-lifecycle"
    )
    trusted_final_state = service_harness.simulation.final_trace.trusted_final_state
    assert public_lifecycle["matchScore"] == trusted_final_state["matchScore"]
    assert public_lifecycle["roundScore"] == trusted_final_state["roundScore"]
    serialized_bundle = json.dumps(bundle_document, sort_keys=True)
    assert "structuralObjectiveScore" not in serialized_bundle
    assert "constructionObjective" not in serialized_bundle

    def put_owned(
        document: dict[str, Any], *, media_type: str, owner_kind: str
    ) -> dict[str, Any]:
        payload = json.dumps(
            document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        reference = research.artifact_store.put(
            payload,
            media_type=media_type,
            name=f"{document['sha256']}.json",
        )
        return {
            **reference,
            "ownerKind": owner_kind,
            "ownerHash": document["sha256"],
        }

    def rehash_result(value: dict[str, Any]) -> None:
        provenance = dict(value["provenance"])
        identity = {
            "schemaVersion": 1,
            "runCode": value["runCode"],
            "shardId": provenance["researchShardId"],
            "shardIndex": provenance["researchShardIndex"],
            "recipeHash": provenance["recipeHash"],
            "traceHash": value["traceHash"],
            "analysisHash": value["artifacts"]["analysis"]["ownerHash"],
            "auditAnalysisHash": value["artifacts"]["auditAnalysis"][
                "ownerHash"
            ],
            "auditHash": value["artifacts"]["decisionAudit"]["ownerHash"],
            "auctionPlanHash": provenance["auctionCompilerPlanHash"],
        }
        if provenance["decisionScoreParityStatus"] == "matched":
            identity.update(
                {
                    "primaryDecisionScoreAnalysisHash": provenance[
                        "primaryDecisionScoreAnalysisHash"
                    ],
                    "decisionScoreParityHash": provenance[
                        "decisionScoreParityHash"
                    ],
                }
            )
        value["resultHash"] = canonical_sha256(identity)
        value["resultCode"] = f"LAF-{value['resultHash'][:10].upper()}"

    def expect_rejected(value: dict[str, Any], message: str) -> None:
        with monkeypatch.context() as scoped:
            scoped.setattr(
                research.store,
                "get_result",
                lambda _code: copy.deepcopy(value),
            )
            with pytest.raises(ValueError, match=message):
                research.get_decisions(value["resultCode"])

    ownerless = copy.deepcopy(result)
    ownerless["artifacts"]["completeTrace"].pop("ownerKind")
    ownerless["artifacts"]["completeTrace"].pop("ownerHash")
    expect_rejected(ownerless, "not closed or owned")

    false_owner = copy.deepcopy(result)
    false_owner["artifacts"]["completeTrace"]["ownerHash"] = "0" * 64
    expect_rejected(false_owner, "semantic owner identity mismatch")

    missing_audit_analysis = copy.deepcopy(result)
    missing_audit_analysis["artifacts"].pop("auditAnalysis")
    expect_rejected(missing_audit_analysis, "not closed or owned")
    with monkeypatch.context() as scoped:
        scoped.setattr(
            research.store,
            "get_result",
            lambda _code: copy.deepcopy(missing_audit_analysis),
        )
        with pytest.raises(ValueError, match="not closed or owned"):
            service_harness.container.research_renders.prepare(
                missing_audit_analysis["resultCode"],
                decision_ordinal=int(selected["ordinal"]),
                export_preset="hooked_flagship",
            )

    false_audit_analysis_owner = copy.deepcopy(result)
    false_audit_analysis_owner["artifacts"]["auditAnalysis"]["ownerHash"] = (
        "1" * 64
    )
    expect_rejected(
        false_audit_analysis_owner,
        "audit analysis semantic owner identity mismatch",
    )

    null_audit_analysis_owner = copy.deepcopy(result)
    null_audit_analysis_owner["artifacts"]["auditAnalysis"]["ownerKind"] = None
    null_audit_analysis_owner["artifacts"]["auditAnalysis"]["ownerHash"] = None
    expect_rejected(null_audit_analysis_owner, "semantic owner kind mismatch")
    with monkeypatch.context() as scoped:
        scoped.setattr(
            research.store,
            "get_result",
            lambda _code: copy.deepcopy(null_audit_analysis_owner),
        )
        with pytest.raises(ValueError, match="semantic owner"):
            service_harness.container.research_renders.prepare(
                null_audit_analysis_owner["resultCode"],
                decision_ordinal=int(selected["ordinal"]),
                export_preset="hooked_flagship",
            )

    false_analysis_owner = copy.deepcopy(result)
    false_analysis_owner["artifacts"]["analysis"]["ownerHash"] = "2" * 64
    expect_rejected(false_analysis_owner, "analysis semantic owner identity mismatch")

    false_audit_owner = copy.deepcopy(result)
    false_audit_owner["artifacts"]["decisionAudit"]["ownerHash"] = "3" * 64
    expect_rejected(
        false_audit_owner,
        "DecisionAuditV2 semantic owner identity mismatch",
    )

    substituted_trace_membership = copy.deepcopy(result)
    substituted_trace_membership["traceHash"] = "f" * 64
    rehash_result(substituted_trace_membership)
    expect_rejected(substituted_trace_membership, "complete-trace membership mismatch")

    substituted_analysis_membership = copy.deepcopy(result)
    substituted_analysis_membership["scoreEvidence"][
        "decisionSelectionQualification"
    ]["analysisHash"] = "e" * 64
    substituted_analysis_membership["scoreEvidence"]["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in substituted_analysis_membership[
                "scoreEvidence"
            ].items()
            if key != "sha256"
        }
    )
    expect_rejected(substituted_analysis_membership, "analysis membership mismatch")

    substituted_analysis = copy.deepcopy(analysis_document)
    substituted_analysis["analyzerId"] = "substituted_trace_analyzer_v1"
    analysis_identity = {
        "schemaVersion": substituted_analysis["schemaVersion"],
        "traceHash": substituted_analysis["traceHash"],
        "profileHash": substituted_analysis["profileHash"],
        "analyzerId": substituted_analysis["analyzerId"],
        "analyzerVersion": substituted_analysis["analyzerVersion"],
        "decisionAnalysisHashes": [
            item["analysisHash"] for item in substituted_analysis["decisions"]
        ],
        "laffaEvidenceHash": substituted_analysis["laffaEvidence"]["sha256"],
    }
    substituted_analysis["sha256"] = canonical_sha256(analysis_identity)
    substituted_audit = copy.deepcopy(audit_document)
    for item in substituted_audit["decisions"]:
        if item["decisionId"] in {
            decision["decisionId"] for decision in substituted_analysis["decisions"]
        }:
            item["provenance"]["analyzerId"] = substituted_analysis["analyzerId"]
    substituted_audit["sha256"] = canonical_sha256(
        {key: value for key, value in substituted_audit.items() if key != "sha256"}
    )
    same_game_substitution = copy.deepcopy(result)
    same_game_substitution["artifacts"]["analysis"] = put_owned(
        substituted_analysis,
        media_type="application/vnd.balootx.trace-analysis-v1+json",
        owner_kind="trace_analysis_v1",
    )
    same_game_substitution["artifacts"]["decisionAudit"] = put_owned(
        substituted_audit,
        media_type="application/vnd.balootx.decision-audit-v2+json",
        owner_kind="decision_audit_v2",
    )
    rehash_result(same_game_substitution)
    expect_rejected(same_game_substitution, "analysis authority mismatch")

    substituted_audit = copy.deepcopy(audit_document)
    substituted_row = next(
        item
        for item in substituted_audit["decisions"]
        if item["decisionId"] == selected["decisionId"]
    )
    substituted_row["recommendedAction"] = next(
        action
        for action in substituted_row["legalActions"]
        if action != substituted_row["recommendedAction"]
    )
    substituted_row["actualMatchesRecommendation"] = (
        substituted_row["actualAction"] == substituted_row["recommendedAction"]
    )
    substituted_audit["sha256"] = canonical_sha256(
        {key: value for key, value in substituted_audit.items() if key != "sha256"}
    )
    audit_substitution = copy.deepcopy(result)
    audit_substitution["artifacts"]["decisionAudit"] = put_owned(
        substituted_audit,
        media_type="application/vnd.balootx.decision-audit-v2+json",
        owner_kind="decision_audit_v2",
    )
    rehash_result(audit_substitution)
    expect_rejected(audit_substitution, "does not match its parsed analysis")

    governed_audit = copy.deepcopy(audit_document)
    governed_row = next(
        item
        for item in governed_audit["decisions"]
        if item["decisionKind"] in {"project", "escalation"}
    )
    governed_row["actualAction"] = {
        **governed_row["actualAction"],
        "eventType": "substituted_governed_event",
    }
    governed_audit["sha256"] = canonical_sha256(
        {key: value for key, value in governed_audit.items() if key != "sha256"}
    )
    governed_substitution = copy.deepcopy(result)
    governed_substitution["artifacts"]["decisionAudit"] = put_owned(
        governed_audit,
        media_type="application/vnd.balootx.decision-audit-v2+json",
        owner_kind="decision_audit_v2",
    )
    rehash_result(governed_substitution)
    expect_rejected(governed_substitution, "does not match its trace event")

    divergent_analysis = copy.deepcopy(analysis_document)
    divergent_laffa = divergent_analysis["laffaEvidence"]
    divergent_laffa["classification"] = (
        f"{divergent_laffa['classification']}_substituted"
    )
    divergent_laffa["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in divergent_laffa.items()
            if key not in {"sha256", "scenarioGraph"}
        }
    )
    divergent_analysis["sha256"] = canonical_sha256(
        {
            "schemaVersion": divergent_analysis["schemaVersion"],
            "traceHash": divergent_analysis["traceHash"],
            "profileHash": divergent_analysis["profileHash"],
            "analyzerId": divergent_analysis["analyzerId"],
            "analyzerVersion": divergent_analysis["analyzerVersion"],
            "decisionAnalysisHashes": [
                item["analysisHash"] for item in divergent_analysis["decisions"]
            ],
            "laffaEvidenceHash": divergent_laffa["sha256"],
        }
    )
    coherent_same_trace_substitution = copy.deepcopy(result)
    divergent_reference = put_owned(
        divergent_analysis,
        media_type="application/vnd.balootx.trace-analysis-v1+json",
        owner_kind="trace_analysis_v1",
    )
    coherent_same_trace_substitution["artifacts"]["analysis"] = copy.deepcopy(
        divergent_reference
    )
    coherent_same_trace_substitution["artifacts"]["auditAnalysis"] = (
        copy.deepcopy(divergent_reference)
    )
    coherent_same_trace_substitution["provenance"]["auditAnalysisHash"] = (
        divergent_analysis["sha256"]
    )
    coherent_same_trace_substitution["classification"] = divergent_laffa[
        "classification"
    ]
    coherent_selection = coherent_same_trace_substitution["scoreEvidence"][
        "decisionSelectionQualification"
    ]
    coherent_selection["analysisHash"] = divergent_analysis["sha256"]
    coherent_same_trace_substitution["scoreEvidence"]["sha256"] = (
        canonical_sha256(
            {
                key: value
                for key, value in coherent_same_trace_substitution[
                    "scoreEvidence"
                ].items()
                if key != "sha256"
            }
        )
    )
    rehash_result(coherent_same_trace_substitution)
    expect_rejected(
        coherent_same_trace_substitution,
        "not the canonical analyzer output",
    )

    substituted_simulation = service_harness.simulator.simulate(
        2,
        dealer=Seat.EAST,
        match_score=(0, 0),
    )
    substituted_trace = substituted_simulation.final_trace
    substituted_trace_document = substituted_trace.to_dict(trusted=True)
    substitution_analyzer = research.analyzers["screening"]
    profile = substitution_analyzer.profile
    heuristic_profile = TraceAnalysisProfileV1.build(
        bid_samples=profile.bid_samples,
        play_samples=profile.play_samples,
        exact_allocation_limit=profile.exact_allocation_limit,
        exact_search_max_remaining=0,
        maximum_sampling_attempts=profile.maximum_sampling_attempts,
        exact_search_max_nodes=profile.exact_search_max_nodes,
        exact_search_max_seconds=profile.exact_search_max_seconds,
        exact_search_max_depth=profile.exact_search_max_depth,
        exact_search_max_cache_entries=profile.exact_search_max_cache_entries,
        search_backend_id=profile.search_backend_id,
        objective=profile.objective,
        use_conventions=profile.use_conventions,
        belief_policy=substitution_analyzer.belief_policy,
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(substitution_analyzer, "profile", heuristic_profile)
        substituted_analysis_result = substitution_analyzer.analyze_trace(
            substituted_trace
        )
    substituted_analysis_document = substituted_analysis_result.to_dict()
    substituted_audit_document = research._decision_audit(
        substituted_trace,
        substituted_analysis_result,
    )
    coordinated_substitution = copy.deepcopy(result)
    coordinated_substitution["traceHash"] = substituted_trace.sha256
    coordinated_substitution["gameHash"] = substituted_trace.game_hash
    coordinated_substitution["artifacts"]["completeTrace"] = put_owned(
        substituted_trace_document,
        media_type="application/vnd.balootx.complete-trace-v1+json",
        owner_kind="complete_trace_v1",
    )
    coordinated_substitution["artifacts"]["analysis"] = put_owned(
        substituted_analysis_document,
        media_type="application/vnd.balootx.trace-analysis-v1+json",
        owner_kind="trace_analysis_v1",
    )
    coordinated_substitution["artifacts"]["auditAnalysis"] = put_owned(
        substituted_analysis_document,
        media_type="application/vnd.balootx.trace-analysis-v1+json",
        owner_kind="trace_analysis_v1",
    )
    coordinated_substitution["artifacts"]["decisionAudit"] = put_owned(
        substituted_audit_document,
        media_type="application/vnd.balootx.decision-audit-v2+json",
        owner_kind="decision_audit_v2",
    )
    coordinated_substitution["provenance"]["auditAnalysisHash"] = (
        substituted_analysis_result.sha256
    )
    coordinated_substitution["decisionCount"] = substituted_audit_document[
        "decisionCount"
    ]
    selection_qualification = coordinated_substitution["scoreEvidence"][
        "decisionSelectionQualification"
    ]
    selection_qualification["analysisHash"] = substituted_analysis_result.sha256
    coordinated_substitution["scoreEvidence"]["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in coordinated_substitution["scoreEvidence"].items()
            if key != "sha256"
        }
    )
    rehash_result(coordinated_substitution)
    expect_rejected(
        coordinated_substitution,
        "retained audit occurrence is unavailable",
    )
