from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from baloot_platform.challenge.analysis_catalog import (
    FoundryAnalysisCandidateServiceV1,
    _complete_observer_safe_lifecycle,
)
from baloot_platform.challenge.deep_analysis import DeepAnalysisBatchSpecV1
from baloot_platform.challenge.evidence_v3 import TruthLabel, canonical_sha256
from baloot_platform.core.enums import Seat
from baloot_platform.challenge.funnel import FoundryAnalysisFunnelResultV1
from baloot_platform.challenge.profiles import ContentSelectionScoreV1
from baloot_platform.challenge.quality_diversity import (
    QualityDiversityCandidateV1,
    QualityDiversityProfileV1,
    QualityDiversitySelectorV1,
)
from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore


def _put(store, owner_kind: str, owner_hash: str, document: dict, media_type: str):
    stored = store.put(
        json.dumps(
            document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
        media_type=media_type,
    )
    return {"ownerKind": owner_kind, "ownerHash": owner_hash, **stored}


def _rehash_funnel(document: dict) -> dict:
    core = {
        "schemaVersion": document["schemaVersion"],
        "analyzerProfileHash": document["analysisProfileHash"],
        "selectionProfileHash": document["selectionProfileHash"],
        "diversityProfileHash": document["diversityProfileHash"],
        "plugins": copy.deepcopy(document["plugins"]),
        "traceHashes": list(document["traceHashes"]),
        "analysisHashes": list(document["analysisHashes"]),
        "tacticalVerificationHashes": list(document["tacticalVerificationHashes"]),
        "verifiedLaffaEvidenceHashes": list(
            document["verifiedLaffaEvidenceHashes"]
        ),
        "candidateHashes": list(document["candidateHashes"]),
        "selectionHash": document["selection"]["sha256"],
    }
    if int(document["schemaVersion"]) >= 2:
        core["surroundingCompetenceEvidenceHashes"] = list(
            document["surroundingCompetenceEvidenceHashes"]
        )
    document["sha256"] = canonical_sha256(core)
    return document


def _fixture(tmp_path):
    trace_hash = "1" * 64
    analysis_hash = "2" * 64
    score = ContentSelectionScoreV1.build(
        score_family="decision_content_selection",
        profile_hash="3" * 64,
        evidence_hash="4" * 64,
        score=88.0,
        passed=True,
    )
    candidate = QualityDiversityCandidateV1.build(
        trace_hash=trace_hash,
        decision_id="decision-1",
        observer_seat="south",
        truth_label=TruthLabel.ROBUST_BEST_KNOWN,
        confidence=0.91,
        structural_fingerprint="5" * 64,
        dimensions={"decision_kind": "play"},
        selection_score=score,
    )
    diversity = QualityDiversityProfileV1.build({
        "schemaVersion": 1,
        "cellDimensions": ["decision_kind"],
        "topKPerCell": 1,
        "rejectedAuditModulus": 10,
    })
    selection = QualityDiversitySelectorV1().select((candidate,), diversity)
    analysis = {
        "schemaVersion": 2,
        "traceHash": trace_hash,
        "decisions": [{
            "decisionId": "decision-1",
            "decisionKind": "play",
            "bestAction": {"kind": "play", "card": "AS"},
            "actualAction": {"kind": "play", "card": "AS"},
            "alternatives": [{"action": {"kind": "play", "card": "AS"}}],
        }],
        "laffaEvidence": {
            "classification": "single_perspective_masterpiece",
            "criticalDecisionIds": ["decision-1"],
        },
        "sha256": analysis_hash,
    }
    store = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    analysis_reference = _put(
        store,
        "trace_analysis_v1",
        analysis_hash,
        analysis,
        "application/vnd.balootx.trace-analysis-v1+json",
    )
    core = {
        "schemaVersion": 1,
        "analyzerProfileHash": "6" * 64,
        "selectionProfileHash": "7" * 64,
        "diversityProfileHash": diversity.sha256,
        "plugins": [],
        "traceHashes": [trace_hash],
        "analysisHashes": [analysis_hash],
        "tacticalVerificationHashes": [],
        "verifiedLaffaEvidenceHashes": [],
        "candidateHashes": [candidate.sha256],
        "selectionHash": selection.sha256,
    }
    funnel = FoundryAnalysisFunnelResultV1(
        core["analyzerProfileHash"],
        core["selectionProfileHash"],
        core["diversityProfileHash"],
        (),
        (trace_hash,),
        (analysis_hash,),
        (),
        (),
        (candidate.sha256,),
        selection,
        (analysis_reference,),
        canonical_sha256(core),
    )
    funnel_reference = _put(
        store,
        "foundry_analysis_funnel_v1",
        funnel.sha256,
        funnel.to_dict(),
        "application/vnd.balootx.foundry-analysis-funnel-v1+json",
    )
    batch = DeepAnalysisBatchSpecV1.from_dict({
        "schemaVersion": 1,
        "campaignHash": "8" * 64,
        "shardResultHashes": ["b" * 64],
        "rulesHash": "a" * 64,
        "bundleHash": "c" * 64,
        "engineVersion": "test",
        "analysisProfileHash": funnel.analysis_profile_hash,
        "perspectiveProfileHash": "d" * 64,
        "selectionProfileHash": funnel.selection_profile_hash,
        "diversityProfileHash": funnel.diversity_profile_hash,
        "deepAnalysisProfileHash": "e" * 64,
        "funnelId": "foundry_analysis_funnel_v1",
        "funnelVersion": "1.0.0",
    })
    job = {
        "jobId": "analysis-1",
        "status": "completed",
        "request": batch.to_dict(),
        "result": {
            "batchHash": batch.sha256,
            "funnelResultHash": funnel.sha256,
            "selectionHash": selection.sha256,
            "candidateHashes": [candidate.sha256],
            "artifactReferences": [analysis_reference, funnel_reference],
        },
    }
    deep = SimpleNamespace(get=lambda job_id: job if job_id == "analysis-1" else None)
    service = FoundryAnalysisCandidateServiceV1(
        deep_analysis=deep,
        campaign_store=None,
        artifact_store=store,
        publication_service=None,
        renderer=None,
        rules=None,
    )
    snapshot_hash = "9" * 64
    service.rules = SimpleNamespace(sha256="a" * 64)
    service._trace_reference = lambda *_args: {"ownerHash": trace_hash}
    service._trace = lambda _reference: SimpleNamespace(
        rules_hash="a" * 64,
        trusted_final_state={
            "generation": {"configurationSnapshotHash": snapshot_hash},
            "configurationSnapshot": {"sha256": snapshot_hash},
        },
    )
    return service, job, funnel, candidate, store


def test_completed_analysis_candidates_are_friendly_and_integrity_checked(tmp_path):
    service, _, _, candidate, _ = _fixture(tmp_path)
    result = service.list_candidates("analysis-1")
    assert result["count"] == 1
    assert result["items"][0]["shortCode"] == candidate.sha256[:8].upper()
    assert result["items"][0]["playReady"] is True
    assert result["items"][0]["compatibilityBlockers"] == []
    assert result["items"][0]["bestAction"]["card"] == "AS"


def test_legacy_candidate_is_not_reported_ready_under_different_rules(tmp_path):
    service, _, _, _, _ = _fixture(tmp_path)
    service._trace = lambda _reference: SimpleNamespace(
        rules_hash="b" * 64,
        trusted_final_state={"generation": {}},
    )

    item = service.list_candidates("analysis-1")["items"][0]

    assert item["playReady"] is False
    assert item["finalistValidationRequired"] is False
    assert item["compatibilityBlockers"] == [
        "rules_profile_mismatch",
        "frozen_configuration_snapshot_missing",
    ]


def test_complete_laffa_lifecycle_keeps_only_one_private_hand() -> None:
    plays = [
        {
            "seat": seat,
            "card": {"id": index, "code": f"C{index}"},
            "trickIndex": index // 4,
            "position": index % 4,
            "legalCards": [{"id": 999}],
        }
        for index, seat in enumerate(("north", "east", "south", "west") * 8)
    ]
    trace = SimpleNamespace(trusted_final_state={
        "roundNumber": 1,
        "dealer": "east",
        "phase": "scored",
        "matchScore": [14, 9],
        "generation": {"matchScore": [0, 0]},
        "initialHands": {
            seat: [{"id": offset + value} for value in range(8)]
            for offset, seat in zip((0, 8, 16, 24), ("north", "east", "south", "west"))
        },
        "completedTricks": [
            {"index": index, "plays": plays[index * 4:(index + 1) * 4]}
            for index in range(8)
        ],
        "projects": [],
    })

    public, observer = _complete_observer_safe_lifecycle(
        trace, observer=Seat.WEST, pause_before_play_index=24
    )

    assert sum(len(item["plays"]) for item in public["completedTricks"]) == 32
    assert "legalCards" not in json.dumps(public)
    assert "initialHands" not in json.dumps(public)
    assert observer["seat"] == "west"
    assert [item["id"] for item in observer["initialHandCards"]] == list(range(24, 32))


def test_strong_replay_nomination_is_internal_only_until_finalist_validation(tmp_path):
    service, _, _, candidate, _ = _fixture(tmp_path)
    nomination = QualityDiversityCandidateV1.build(
        trace_hash=candidate.trace_hash,
        decision_id=candidate.decision_id,
        observer_seat=candidate.observer_seat,
        truth_label=candidate.truth_label,
        confidence=candidate.confidence,
        structural_fingerprint=candidate.structural_fingerprint,
        dimensions={
            "decision_kind": "play",
            "candidate_scope": "deal_replay_nomination",
            "surrounding_competence": "needs_strong_replay",
        },
        selection_score=candidate.selection_score,
    )
    service._selected = lambda _job, _funnel=None: (nomination,)
    service._trace_reference = lambda *_args: {"ownerHash": candidate.trace_hash}
    service._trace = lambda _reference: "trusted-parent-trace"

    assert (
        service.candidate_trace("analysis-1", nomination.sha256)
        == "trusted-parent-trace"
    )
    with pytest.raises(ValueError, match="governed finalist validation"):
        service.prepare_actor_candidate("analysis-1", nomination.sha256)


def test_best_locally_qualified_finalist_is_materialized_from_descendant_evidence(tmp_path):
    service, job, _funnel, candidate, _store = _fixture(tmp_path)
    analysis_reference, funnel_reference = job["result"]["artifactReferences"]
    trusted_trace = SimpleNamespace(sha256=candidate.trace_hash)
    service._trace = lambda _reference: trusted_trace
    service._prepare_verified_candidate = lambda **values: {
        "candidateHash": values["candidate"].sha256,
        "trace": values["trace"],
        "revealPolicy": values["reveal_policy"],
    }
    finalist = {
        "status": "completed",
        "request": {"finalistAnalysisProfileHash": "6" * 64},
        "result": {
                "items": [{
                    "localPreviewReady": True,
                    "publicationReady": False,
                    "blockers": ["professional_agent_qualification_missing"],
                    "evidence": {"sha256": "replay-evidence-hash"},
                    "analysisEvidence": {
                        "descendantTraceHash": candidate.trace_hash,
                        "funnelHash": _funnel.sha256,
                        "analysisProfileHash": "6" * 64,
                        "sha256": "analysis-evidence-hash",
                    },
                    "traceArtifact": {
                    "ownerKind": "complete_trace_v1",
                    "ownerHash": candidate.trace_hash,
                },
                "analysisFunnelArtifact": funnel_reference,
                "analysisArtifacts": [analysis_reference],
            }]
        },
    }

    result = service.prepare_best_finalist_candidate(finalist)

    assert result["candidateHash"] == candidate.sha256
    assert result["trace"] is trusted_trace
    assert result["revealPolicy"] == "after_commit"
    assert result["qualification"] == {
        "schemaVersion": 1,
        "localPreviewReady": True,
        "publicationReady": False,
        "publicationBlockers": ["professional_agent_qualification_missing"],
        "finalistManifestHash": "",
        "replayEvidenceHash": "replay-evidence-hash",
        "analysisEvidenceHash": "analysis-evidence-hash",
    }


def test_best_locally_qualified_finalist_ignores_a_stronger_bid_candidate(tmp_path):
    service, job, funnel, candidate, store = _fixture(tmp_path)
    analysis_reference, _ = job["result"]["artifactReferences"]
    bid_candidate = QualityDiversityCandidateV1.build(
        trace_hash=candidate.trace_hash,
        decision_id="bid-decision",
        observer_seat=candidate.observer_seat,
        truth_label=candidate.truth_label,
        confidence=1.0,
        structural_fingerprint="9" * 64,
        dimensions={"decision_kind": "bid"},
        selection_score=ContentSelectionScoreV1.build(
            score_family="decision_content_selection",
            profile_hash="8" * 64,
            evidence_hash="7" * 64,
            score=99.0,
            passed=True,
        ),
    )
    diversity = QualityDiversityProfileV1.build({
        "schemaVersion": 1,
        "cellDimensions": ["decision_kind"],
        "topKPerCell": 1,
        "rejectedAuditModulus": 10,
    })
    selection = QualityDiversitySelectorV1().select(
        (candidate, bid_candidate), diversity
    )
    replacement_funnel = FoundryAnalysisFunnelResultV1(
        funnel.analysis_profile_hash,
        funnel.selection_profile_hash,
        diversity.sha256,
        funnel.plugins,
        funnel.trace_hashes,
        funnel.analysis_hashes,
        funnel.tactical_verification_hashes,
        funnel.verified_laffa_evidence_hashes,
        tuple(item.sha256 for item in selection.selected),
        selection,
        funnel.artifact_references,
        canonical_sha256({
            "schemaVersion": 1,
            "analyzerProfileHash": funnel.analysis_profile_hash,
            "selectionProfileHash": funnel.selection_profile_hash,
            "diversityProfileHash": diversity.sha256,
            "plugins": [dict(item) for item in funnel.plugins],
            "traceHashes": list(funnel.trace_hashes),
            "analysisHashes": list(funnel.analysis_hashes),
            "tacticalVerificationHashes": list(funnel.tactical_verification_hashes),
            "verifiedLaffaEvidenceHashes": list(funnel.verified_laffa_evidence_hashes),
            "candidateHashes": [item.sha256 for item in selection.selected],
            "selectionHash": selection.sha256,
        }),
    )
    funnel_reference = _put(
        store,
        "foundry_analysis_funnel_v1",
        replacement_funnel.sha256,
        replacement_funnel.to_dict(),
        "application/vnd.balootx.foundry-analysis-funnel-v1+json",
    )
    service._trace = lambda _reference: SimpleNamespace(sha256=candidate.trace_hash)
    service._prepare_verified_candidate = lambda **values: {
        "candidateHash": values["candidate"].sha256,
    }
    finalist = {
        "status": "completed",
        "request": {
            "finalistAnalysisProfileHash": replacement_funnel.analysis_profile_hash,
        },
        "result": {"items": [{
            "localPreviewReady": True,
            "traceArtifact": {"ownerKind": "complete_trace_v1"},
            "analysisFunnelArtifact": funnel_reference,
            "analysisArtifacts": [analysis_reference],
        }]},
    }

    result = service.prepare_best_finalist_candidate(finalist)

    assert bid_candidate.selection_score.score > candidate.selection_score.score
    assert result["candidateHash"] == candidate.sha256


def test_finalist_materialization_rejects_unqualified_descendants(tmp_path):
    service, *_ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="no locally qualified"):
        service.prepare_best_finalist_candidate({
            "status": "completed",
            "request": {"finalistAnalysisProfileHash": "6" * 64},
            "result": {"items": [{"localPreviewReady": False}]},
        })


def test_candidate_catalog_rejects_running_or_semantically_tampered_batches(tmp_path):
    service, job, funnel, _, store = _fixture(tmp_path)
    job["status"] = "running"
    with pytest.raises(RuntimeError, match="completed"):
        service.list_candidates("analysis-1")

    job["status"] = "completed"
    tampered = copy.deepcopy(funnel.to_dict())
    tampered["selection"]["selected"][0]["confidence"] = 0.2
    replacement = _put(
        store,
        "foundry_analysis_funnel_v1",
        funnel.sha256,
        tampered,
        "application/vnd.balootx.foundry-analysis-funnel-v1+json",
    )
    job["result"]["artifactReferences"][-1] = replacement
    with pytest.raises(ValueError, match="checksum"):
        service.list_candidates("analysis-1")


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("analysisProfileHash", ["0" * 64], "frozen batch"),
        ("analysisHashes", ["0" * 64], "analysis artifact membership"),
        ("traceHashes", ["0" * 64], "absent from the verified funnel"),
    ),
)
def test_catalog_rejects_fully_rehashed_funnel_authority_and_membership_substitution(
    tmp_path,
    field,
    replacement,
    message,
):
    service, job, funnel, _, store = _fixture(tmp_path)
    tampered = copy.deepcopy(funnel.to_dict())
    tampered[field] = replacement[0] if field == "analysisProfileHash" else replacement
    _rehash_funnel(tampered)
    replacement_reference = _put(
        store,
        "foundry_analysis_funnel_v1",
        tampered["sha256"],
        tampered,
        "application/vnd.balootx.foundry-analysis-funnel-v1+json",
    )
    job["result"]["artifactReferences"][-1] = replacement_reference
    job["result"]["funnelResultHash"] = tampered["sha256"]

    with pytest.raises(ValueError, match=message):
        service.list_candidates("analysis-1")


def test_candidate_reveal_reason_uses_verified_tactical_profile(tmp_path):
    _service, _job, _funnel, candidate, _store = _fixture(tmp_path)
    tactical = QualityDiversityCandidateV1.build(
        trace_hash=candidate.trace_hash,
        decision_id=candidate.decision_id,
        observer_seat=candidate.observer_seat,
        truth_label=candidate.truth_label,
        confidence=candidate.confidence,
        structural_fingerprint=candidate.structural_fingerprint,
        dimensions={
            "decision_kind": "play",
            "tactical_detector": "contract_rescue+only_winning_move",
        },
        selection_score=candidate.selection_score,
    )
    analysis = {
        "decisions": [{
            "decisionId": candidate.decision_id,
            "bestAction": {"code": "HA", "label": "A♥"},
            "actualAction": {"code": "D9", "label": "9♦"},
            "alternatives": [],
        }]
    }

    key, values = FoundryAnalysisCandidateServiceV1._reveal_reason(tactical, analysis)

    assert key == "video.analysis.only_winning_contract_rescue"
    assert values == {"best": "A♥", "alternative": "9♦"}
