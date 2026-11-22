from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from baloot_platform.challenge.evidence_v3 import canonical_sha256

from research_audit_trace_helpers import (
    _Harness,
    _SequentialGeneratorStub,
    _deal_spec,
    _payload,
    _result_items,
    _store_validated_legacy_recipe,
    _stub_ortools_compilers,
)
from research_audit_trace_helpers import service_harness as service_harness


def test_retained_ortools_screening_result_uses_deep_audit_analysis_for_render(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    label = "ortools-screening-audit-render"
    payload = _payload(
        research,
        label=label,
        lane="ortools",
        resources={"analysisMode": "screening"},
    )
    payload["schemaVersion"] = 3
    payload.pop("profiles")
    payload["tactics"] = []
    payload["configuration"] = {
        "schemaVersion": 1,
        "bundleRevisionId": service_harness.container.configuration.resolve()[
            "bundle"
        ]["revisionId"],
    }
    spec = _deal_spec(
        label,
        assumed_buyer=None,
        maximum_solutions=1,
    )
    monkeypatch.setattr(
        research,
        "deal_generator",
        _SequentialGeneratorStub(label, candidate_count=1),
    )
    created = _store_validated_legacy_recipe(research, payload)
    stored_recipe = research.store.get_recipe(created["code"])
    assert stored_recipe is not None
    recipe, context = research._build_recipe_and_context(stored_recipe)
    assert context is not None
    run = research._submit_recipe_run(
        recipe.code,
        idempotency_key=label,
        schedule=False,
    )
    claim = research.store.claim_shard(
        run["runCode"],
        worker_id=f"audit-service-worker:{label}",
        lease_seconds=60,
    )
    assert claim is not None
    _stub_ortools_compilers(
        research,
        monkeypatch,
        label=label,
        spec=spec,
    )
    monkeypatch.setattr(research, "_semantic_mismatches", lambda *_a, **_k: [])
    base_qualification = research._qualification

    def retain_as_rejected(*args: Any, **kwargs: Any) -> dict[str, Any]:
        value = base_qualification(*args, **kwargs)
        return {
            **value,
            "qualified": False,
            "truthLabel": "rejected",
            "failures": [*value["failures"], "service_fixture_late_gate"],
        }

    monkeypatch.setattr(research, "_qualification", retain_as_rejected)

    research._execute_claimed_shard(
        run["runCode"],
        claim,
        claim["leaseOwner"],
        context=context,
    )

    results = _result_items(research, run["runCode"])
    assert len(results) == 1
    result = results[0]
    assert result["provenance"]["generationLane"] == "ortools"
    assert result["provenance"]["finalistPrimaryPolicy"] is True
    assert result["provenance"]["decisionScoreParityStatus"] == "matched"
    assert result["provenance"]["structuralObjectiveScore"] == 1
    selected_analysis = json.loads(
        research.artifact_store.get(result["artifacts"]["analysis"]["uri"])
    )
    audit_analysis = json.loads(
        research.artifact_store.get(result["artifacts"]["auditAnalysis"]["uri"])
    )
    audit_document = json.loads(
        research.artifact_store.get(result["artifacts"]["decisionAudit"]["uri"])
    )
    assert selected_analysis["profileHash"] == context.analyzers[
        "screening"
    ].profile.sha256
    assert audit_analysis["profileHash"] == context.analyzers["deep"].profile.sha256
    assert selected_analysis["sha256"] != audit_analysis["sha256"]
    assert result["provenance"]["auditAnalysisHash"] == audit_analysis["sha256"]
    assert result["provenance"]["primaryDecisionScoreAnalysisHash"] == (
        audit_analysis["sha256"]
    )
    assert result["artifacts"]["auditAnalysis"]["ownerHash"] == audit_analysis[
        "sha256"
    ]
    expected_identity = {
        "schemaVersion": 1,
        "runCode": result["runCode"],
        "shardId": result["provenance"]["researchShardId"],
        "shardIndex": result["provenance"]["researchShardIndex"],
        "recipeHash": result["provenance"]["recipeHash"],
        "traceHash": result["traceHash"],
        "analysisHash": selected_analysis["sha256"],
        "auditAnalysisHash": audit_analysis["sha256"],
        "auditHash": audit_document["sha256"],
        "primaryDecisionScoreAnalysisHash": result["provenance"][
            "primaryDecisionScoreAnalysisHash"
        ],
        "decisionScoreParityHash": result["provenance"][
            "decisionScoreParityHash"
        ],
        "auctionPlanHash": result["provenance"]["auctionCompilerPlanHash"],
    }
    assert result["resultHash"] == canonical_sha256(expected_identity)
    assert result["resultCode"] == (
        f"LAF-{result['resultHash'][:10].upper()}"
    )

    decisions = research.get_decisions(result["resultCode"])
    assert decisions == audit_document["decisions"]
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
    assert source["analysisHash"] == audit_analysis["sha256"]
    assert source["analysisProfileHash"] == audit_analysis["profileHash"]
    assert source["analyzerId"] == audit_analysis["analyzerId"]
    assert source["analyzerVersion"] == audit_analysis["analyzerVersion"]
    assert source["auditHash"] == audit_document["sha256"]
    assert source["decisionId"] == selected["decisionId"]
    serialized_source = json.dumps(source, sort_keys=True)
    assert "structuralObjective" not in serialized_source
    assert "constructionObjective" not in serialized_source
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
    trusted = json.loads(
        research.artifact_store.get(result["artifacts"]["completeTrace"]["uri"])
    )["trustedFinalState"]
    assert public_lifecycle["matchScore"] == trusted["matchScore"]
    assert public_lifecycle["roundScore"] == trusted["roundScore"]
    serialized_bundle = json.dumps(bundle_document, sort_keys=True)
    assert "structuralObjectiveScore" not in serialized_bundle
    assert "constructionObjective" not in serialized_bundle

    def put_cas(
        document: dict[str, Any],
        *,
        media_type: str,
    ) -> dict[str, Any]:
        payload_bytes = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return dict(
            research.artifact_store.put(
                payload_bytes,
                media_type=media_type,
                name=f"{document['sha256']}.json",
            )
        )

    def rehash_ortools_result(value: dict[str, Any]) -> None:
        provenance = value["provenance"]
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
            "primaryDecisionScoreAnalysisHash": provenance[
                "primaryDecisionScoreAnalysisHash"
            ],
            "decisionScoreParityHash": provenance["decisionScoreParityHash"],
            "auctionPlanHash": provenance["auctionCompilerPlanHash"],
        }
        value["resultHash"] = canonical_sha256(identity)
        value["resultCode"] = f"LAF-{value['resultHash'][:10].upper()}"

    def expect_ortools_rejected(value: dict[str, Any], message: str) -> None:
        def unexpected_canonical_reanalysis(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError(
                "incoherent retained artifacts must fail before canonical reanalysis"
            )

        with monkeypatch.context() as scoped:
            scoped.setattr(
                research.store,
                "get_result",
                lambda _code: copy.deepcopy(value),
            )
            for analyzer_type in {
                type(analyzer) for analyzer in context.analyzers.values()
            }:
                scoped.setattr(
                    analyzer_type,
                    "analyze_trace",
                    unexpected_canonical_reanalysis,
                )
            with pytest.raises(ValueError, match=message):
                research.get_decisions(value["resultCode"])

    primary_substitution = copy.deepcopy(result)
    primary_substitution["artifacts"]["primaryDecisionScoreAnalysis"] = put_cas(
        selected_analysis,
        media_type=(
            "application/vnd.balootx."
            "primary-decision-score-analysis-v1+json"
        ),
    )
    expect_ortools_rejected(
        primary_substitution,
        "primary decision-score analysis mismatch",
    )

    parity_document = json.loads(
        research.artifact_store.get(
            result["artifacts"]["decisionScoreParity"]["uri"]
        )
    )
    substituted_parity = copy.deepcopy(parity_document)
    substituted_parity["structuralObjectiveCompared"] = True
    substituted_parity["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in substituted_parity.items()
            if key != "sha256"
        }
    )
    parity_substitution = copy.deepcopy(result)
    parity_substitution["artifacts"]["decisionScoreParity"] = put_cas(
        substituted_parity,
        media_type="application/vnd.balootx.decision-score-parity-v1+json",
    )
    parity_substitution["provenance"]["decisionScoreParityHash"] = (
        substituted_parity["sha256"]
    )
    rehash_ortools_result(parity_substitution)
    expect_ortools_rejected(
        parity_substitution,
        "decision-score parity evidence mismatch",
    )

    drifted_run = copy.deepcopy(research.store.get_run(run["runCode"]))
    assert drifted_run is not None
    drifted_run["configurationAuthority"] = copy.deepcopy(
        drifted_run["configurationAuthority"]
    )
    drifted_run["configurationAuthority"]["sha256"] = "0" * 64
    with monkeypatch.context() as scoped:
        scoped.setattr(
            research.store,
            "get_run",
            lambda _code: copy.deepcopy(drifted_run),
        )
        with pytest.raises(ValueError, match="configuration authority mismatch"):
            research.get_decisions(result["resultCode"])
