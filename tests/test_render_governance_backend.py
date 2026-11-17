from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

import baloot_platform.research.render_service as render_module
from baloot_platform.service import api
from baloot_platform.audit.models import content_hash
from baloot_platform.challenge.analysis_catalog import complete_observer_safe_lifecycle
from baloot_platform.challenge.evidence_v3 import (
    DecisionLedgerV1,
    TruthLabel,
    TruthMode,
)
from baloot_platform.challenge.profiles import SelectionProfileV1
from baloot_platform.challenge.video_production_batch import (
    FoundryVideoBatchServiceV1,
    HookedLaffaBatchProfileV1,
    evaluate_video_production_candidate_v1,
)
from baloot_platform.core.enums import Seat
from baloot_platform.research.decision_audit import decode_decision_audit_v2
from baloot_platform.research.render_service import ResearchRenderPreparationServiceV1
from baloot_platform.video.batch_service import RenderBatchServiceV1
from baloot_platform.video.render_bundle import RenderBundleV1


def _sha(value: dict[str, object]) -> str:
    return content_hash(value)


def _decision_audit_fixture() -> tuple[DecisionLedgerV1, dict[str, object]]:
    ledger = DecisionLedgerV1.build(
        "a" * 64,
        (
            {
                "decisionId": "D1",
                "ordinal": 1,
                "actor": "south",
                "decisionKind": "play",
                "phase": "playing",
                "publicState": {"observer": "south"},
                "completeStateHash": "b" * 64,
                "legalActions": [
                    {"kind": "play_card", "card": {"id": 1}},
                    {"kind": "play_card", "card": {"id": 2}},
                ],
                "selectedAction": {"kind": "play_card", "card": {"id": 1}},
            },
        ),
    )
    record = ledger.records[0]
    row: dict[str, object] = {
        "schemaVersion": 1,
        "decisionId": record.decision_id,
        "decisionOrdinal": record.ordinal,
        "actingSeat": "south",
        "actingTeam": record.acting_seat.team.name.lower(),
        "decisionKind": record.decision_kind,
        "phase": record.phase,
        "publicStateHash": record.public_prefix_hash,
        "informationStateHash": "c" * 64,
        "informationSetEvidence": {
            "status": "exact_information_set",
            "observerSafe": True,
            "actionCoverageComplete": True,
            "actorChoiceBoundOnce": True,
        },
        "sequentialResponseEvidence": {
            "status": "exact_sequential_response",
            "allAlternativesEvaluated": True,
            "pairedActorSafeCohort": True,
        },
        "legalActions": [dict(item) for item in record.legal_actions],
        "rulePaths": ["engine.legal_actions"],
        "actualAction": dict(record.chosen_action),
        "actionSource": "agent",
        "forced": False,
        "recommendedAction": dict(record.chosen_action),
        "actualMatchesRecommendation": True,
        "alternatives": [],
        "expectedRegret": 0.0,
        "probabilityDrop": 0.0,
        "catastrophicCrossing": False,
        "hiddenDealSupport": 1,
        "effectiveSampleSize": 1.0,
        "confidence": 1.0,
        "convergence": "exact",
        "principalContinuation": [],
        "supportedMetrics": [],
        "unsupportedMetrics": [],
        "primaryDecisionScoreEvidence": None,
        "forcedLegalityProof": None,
        "selectionExplanation": {"available": True},
        "analyticalExplanation": {"status": "exact"},
        "forcedExplanation": None,
        "tacticalEvidence": [],
        "actualDealOracle": None,
        "truthLabel": "exact_optimal",
        "provenance": {
            "engineVersion": "1.0.0",
            "bundleHash": "d" * 64,
            "rulesHash": "e" * 64,
            "analyzerId": "adaptive_finalist_trace_analyzer_v1",
            "analyzerVersion": "1.3.0",
            "analysisProfileHash": "f" * 64,
            "analysisHash": "1" * 64,
        },
        "ordinal": 1,
    }
    core: dict[str, object] = {
        "schemaVersion": 2,
        "traceHash": "2" * 64,
        "decisionCount": 1,
        "choiceDecisionCount": 1,
        "governedActionCount": 0,
        "incompleteGovernedActionCount": 0,
        "complete": True,
        "informationSetComplete": True,
        "sequentialResponseComplete": True,
        "configurationProvenance": None,
        "decisions": [row],
    }
    return ledger, {**core, "sha256": content_hash(core)}


def _canonical_payload(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_decision_audit_decoder_is_closed_canonical_and_same_trace() -> None:
    ledger, audit = _decision_audit_fixture()

    decoded = decode_decision_audit_v2(
        _canonical_payload(audit),
        expected_trace_hash="2" * 64,
        ledger=ledger,
        expected_engine_version="1.0.0",
        expected_bundle_hash="d" * 64,
        expected_rules_hash="e" * 64,
    )
    assert decoded == audit

    drifted = copy.deepcopy(audit)
    drifted["decisions"][0]["provenance"]["rulesHash"] = "0" * 64
    drifted_core = {key: value for key, value in drifted.items() if key != "sha256"}
    drifted["sha256"] = content_hash(drifted_core)
    with pytest.raises(ValueError, match="trace provenance mismatch"):
        decode_decision_audit_v2(
            _canonical_payload(drifted),
            expected_trace_hash="2" * 64,
            ledger=ledger,
            expected_engine_version="1.0.0",
            expected_bundle_hash="d" * 64,
            expected_rules_hash="e" * 64,
        )

    extended = copy.deepcopy(audit)
    extended["decisions"][0]["foreign"] = True
    extended_core = {key: value for key, value in extended.items() if key != "sha256"}
    extended["sha256"] = content_hash(extended_core)
    with pytest.raises(ValueError, match="row schema is not closed"):
        decode_decision_audit_v2(
            _canonical_payload(extended),
            expected_trace_hash="2" * 64,
            ledger=ledger,
        )

    with pytest.raises(ValueError, match="not canonical JSON"):
        decode_decision_audit_v2(
            json.dumps(audit, ensure_ascii=False, indent=2).encode("utf-8"),
            expected_trace_hash="2" * 64,
            ledger=ledger,
        )


def test_video_lifecycle_scores_come_only_from_engine_final_state() -> None:
    plays = [
        {
            "seat": ("north", "west", "south", "east")[index % 4],
            "card": {"id": index},
            "trickIndex": index // 4,
            "position": index % 4,
        }
        for index in range(32)
    ]
    state = {
        "initialHands": {
            seat: [{"id": index} for index in range(8)]
            for seat in ("north", "west", "south", "east")
        },
        "completedTricks": [
            {"plays": plays[index : index + 4]} for index in range(0, 32, 4)
        ],
        "matchScore": [152, 91],
        "roundScore": {"us": 34, "them": 17},
        "generation": {
            "matchScore": [120, 80],
            "structuralObjectiveScore": 999_999,
            "roundScore": {"us": 999, "them": 0},
        },
    }
    public, _ = complete_observer_safe_lifecycle(
        SimpleNamespace(trusted_final_state=state),
        observer=Seat.SOUTH,
        pause_before_play_index=12,
    )

    assert public["matchScore"] == [152, 91]
    assert public["preDecisionMatchScore"] == [120, 80]
    assert public["roundScore"] == {"us": 34, "them": 17}
    assert "structuralObjectiveScore" not in public


def test_research_render_loader_requires_closed_canonical_typed_artifacts() -> None:
    media_type = "application/vnd.balootx.test+json"
    payload = _canonical_payload({"schemaVersion": 1, "value": "exact"})
    store = SimpleNamespace(get=lambda _uri: payload)
    service = ResearchRenderPreparationServiceV1(
        research=object(),
        artifact_store=store,
        publication_service=object(),
        renderer=object(),
        rules=object(),
    )
    reference = {
        "uri": "cas://sha256/test",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "mediaType": media_type,
        "name": "test.json",
    }

    assert service._load(reference, expected_media_type=media_type) == {
        "schemaVersion": 1,
        "value": "exact",
    }

    pretty = json.dumps(
        {"schemaVersion": 1, "value": "exact"},
        indent=2,
    ).encode("utf-8")
    store.get = lambda _uri: pretty
    pretty_reference = {
        **reference,
        "sha256": hashlib.sha256(pretty).hexdigest(),
        "bytes": len(pretty),
    }
    with pytest.raises(ValueError, match="not canonical JSON"):
        service._load(pretty_reference, expected_media_type=media_type)

    with pytest.raises(ValueError, match="reference is not closed"):
        service._load(
            {**reference, "foreign": True},
            expected_media_type=media_type,
        )


def test_research_result_identity_binds_trace_analysis_and_audit() -> None:
    identity = {
        "schemaVersion": 1,
        "runCode": "RUN-EXACT",
        "shardId": "SHARD-EXACT",
        "shardIndex": 2,
        "recipeHash": "1" * 64,
        "traceHash": "2" * 64,
        "analysisHash": "3" * 64,
        "auditAnalysisHash": "5" * 64,
        "auditHash": "4" * 64,
        "primaryDecisionScoreAnalysisHash": "5" * 64,
        "decisionScoreParityHash": "6" * 64,
        "auctionPlanHash": "7" * 64,
    }
    result_hash = content_hash(identity)
    result_code = f"LAF-{result_hash[:10].upper()}"
    result = {
        "resultCode": result_code,
        "resultHash": result_hash,
        "runCode": "RUN-EXACT",
        "provenance": {
            "researchShardId": "SHARD-EXACT",
            "researchShardIndex": 2,
            "recipeHash": "1" * 64,
            "auditAnalysisHash": "5" * 64,
            "decisionScoreParityStatus": "matched",
            "primaryDecisionScoreAnalysisHash": "5" * 64,
            "decisionScoreParityHash": "6" * 64,
            "auctionCompilerPlanHash": "7" * 64,
        },
    }

    assert (
        render_module._validated_research_result_hash(
            result,
            result_code=result_code,
            trace_hash="2" * 64,
            analysis_hash="3" * 64,
            audit_analysis_hash="5" * 64,
            audit_hash="4" * 64,
        )
        == result_hash
    )

    with pytest.raises(ValueError, match="identity checksum mismatch"):
        render_module._validated_research_result_hash(
            result,
            result_code=result_code,
            trace_hash="2" * 64,
            analysis_hash="8" * 64,
            audit_analysis_hash="5" * 64,
            audit_hash="4" * 64,
        )


def _challenge_profile(root: Path) -> dict[str, object]:
    return yaml.safe_load(
        (root / "configs/challenge/progressive_mining_v2.yaml").read_text(
            encoding="utf-8"
        )
    )


def _trace_with_profile(
    root: Path,
    *,
    minimum_video_score: float = 60.0,
) -> SimpleNamespace:
    challenge = copy.deepcopy(_challenge_profile(root))
    challenge["video_production_batch"]["minimumSelectionScore"] = minimum_video_score
    component = {"sha256": _sha(challenge), "payload": challenge}
    bundle_hash = "b" * 64
    snapshot_core = {
        "schemaVersion": 1,
        "bundleHash": bundle_hash,
        "mode": "test",
        "profile": {},
        "components": {"challenge": component},
    }
    snapshot = {**snapshot_core, "sha256": _sha(snapshot_core)}
    return SimpleNamespace(
        bundle_hash=bundle_hash,
        trusted_final_state={"configurationSnapshot": snapshot},
    )


def _qualified_result(
    root: Path,
    *,
    analysis_hash: str = "a" * 64,
    decision_id: str = "D1",
    score: float = 70.0,
) -> tuple[dict[str, object], SimpleNamespace, SimpleNamespace, dict[str, object]]:
    challenge = _challenge_profile(root)
    selection_profile = SelectionProfileV1.build(challenge["selection_policy"])
    row = {
        "decisionId": decision_id,
        "actingSeat": "south",
        "decisionKind": "play",
        "truthLabel": "robust_best_known",
        "confidence": 0.80,
        "score": {"score": score},
        "gatePassed": True,
        "gateFailures": [],
    }
    selection_core = {
        "schemaVersion": 1,
        "profileHash": selection_profile.sha256,
        "analysisHash": analysis_hash,
        "evaluatedDecisionCount": 1,
        "qualifiedDecisionCount": 1,
        "qualifiedDecisionIds": [decision_id],
        "passed": True,
        "rejectionReasons": [],
        "decisions": [row],
    }
    selection_hash = _sha(selection_core)
    score_core = {
        "schemaVersion": 1,
        "family": "research_content_selection",
        "decisionChallengeScores": [row],
        "decisionSelectionQualification": {
            **{
                key: copy.deepcopy(value)
                for key, value in selection_core.items()
                if key != "decisions"
            },
            "sha256": selection_hash,
        },
        "decisionProfileHash": selection_profile.sha256,
    }
    result = {
        "disposition": "qualified",
        "criticalDecisionIds": [decision_id],
        "qualification": {
            "qualified": True,
            "failures": [],
            "decisionSelection": {
                "profileHash": selection_profile.sha256,
                "evidenceHash": selection_hash,
                "qualifiedDecisionIds": [decision_id],
                "passed": True,
            },
        },
        "scoreEvidence": {**score_core, "sha256": _sha(score_core)},
    }
    analysis = SimpleNamespace(
        sha256=analysis_hash,
        laffa_evidence=SimpleNamespace(critical_decision_ids=(decision_id,)),
    )
    evidence = SimpleNamespace(
        truth_mode=TruthMode.ACTOR_OBSERVER,
        truth_label=TruthLabel.ROBUST_BEST_KNOWN,
        confidence=0.80,
    )
    audit = {"decisionId": decision_id, "decisionKind": "play"}
    return result, analysis, evidence, audit


def test_research_current_adaptive_profile_comes_from_frozen_recipe_context(
    monkeypatch,
):
    profile_hash = "a" * 64
    recipe_hash = "b" * 64
    recipe = SimpleNamespace(
        code="RCP-AUTHORITY",
        sha256=recipe_hash,
        resources={"analysisMode": "deep"},
    )
    analyzer = SimpleNamespace(
        descriptor=SimpleNamespace(
            id="adaptive_finalist_trace_analyzer_v1",
            version="1.3.0",
        ),
        profile=SimpleNamespace(sha256=profile_hash),
    )
    context = SimpleNamespace(analyzers={"deep": analyzer})
    run = {"recipeCode": recipe.code, "recipeHash": recipe.sha256}
    validated = []
    research = SimpleNamespace(
        store=SimpleNamespace(
            get_recipe=lambda code: {"code": code},
            get_run=lambda code: run if code == "RUN-AUTHORITY" else None,
        ),
        _build_recipe_and_context=lambda _value: (recipe, context),
        _assert_run_execution_context=lambda *values: validated.append(values),
    )
    service = ResearchRenderPreparationServiceV1(
        research=research,
        artifact_store=object(),
        publication_service=object(),
        renderer=object(),
        rules=object(),
    )
    trace = SimpleNamespace(sha256="c" * 64)
    document = {
        "analyzerId": "adaptive_finalist_trace_analyzer_v1",
        "analyzerVersion": "1.3.0",
    }
    result = {
        "recipeCode": recipe.code,
        "runCode": "RUN-AUTHORITY",
        "provenance": {"recipeHash": recipe.sha256},
    }
    monkeypatch.setattr(
        render_module,
        "frozen_adaptive_finalist_profile_hash_v1",
        lambda value: profile_hash if value is trace else "0" * 64,
    )

    assert (
        service._frozen_expected_analysis_profile_hash(
            result,
            trace,
            document,
        )
        == profile_hash
    )
    assert validated == [(run, recipe, context)]

    substituted = copy.deepcopy(result)
    substituted["provenance"]["recipeHash"] = "0" * 64
    with pytest.raises(ValueError, match="recipe authority mismatch"):
        service._frozen_expected_analysis_profile_hash(
            substituted,
            trace,
            document,
        )

    analyzer.descriptor = SimpleNamespace(
        id="staged_trace_analyzer_v1", version="1.3.0"
    )
    with pytest.raises(ValueError, match="does not match the adaptive artifact"):
        service._frozen_expected_analysis_profile_hash(
            result,
            trace,
            document,
        )


def test_research_historical_adaptive_analysis_has_no_current_profile_authority():
    service = ResearchRenderPreparationServiceV1(
        research=SimpleNamespace(),
        artifact_store=object(),
        publication_service=object(),
        renderer=object(),
        rules=object(),
    )

    assert (
        service._frozen_expected_analysis_profile_hash(
            {},
            SimpleNamespace(),
            {
                "analyzerId": "adaptive_finalist_trace_analyzer_v1",
                "analyzerVersion": "1.2.0",
            },
        )
        is None
    )


def test_requested_observer_selects_exact_actor_partner_or_opponent_evidence() -> None:
    decision = SimpleNamespace(decision_id="D1", acting_seat=Seat.SOUTH)
    actor = SimpleNamespace(
        decision_id="D1",
        acting_seat=Seat.SOUTH,
        observer_seat=Seat.SOUTH,
        truth_mode=TruthMode.ACTOR_OBSERVER,
    )
    partner = SimpleNamespace(
        decision_id="D1",
        acting_seat=Seat.SOUTH,
        observer_seat=Seat.NORTH,
        truth_mode=TruthMode.SEAT_EXPERIENCE,
    )
    opponent = SimpleNamespace(
        decision_id="D1",
        acting_seat=Seat.SOUTH,
        observer_seat=Seat.EAST,
        truth_mode=TruthMode.SEAT_EXPERIENCE,
    )
    analysis = SimpleNamespace(
        laffa_evidence=SimpleNamespace(
            perspectives=(actor, partner, opponent),
        )
    )

    assert (
        ResearchRenderPreparationServiceV1._select_perspective(analysis, decision, None)
        is actor
    )
    assert (
        ResearchRenderPreparationServiceV1._select_perspective(
            analysis, decision, "north"
        )
        is partner
    )
    assert (
        ResearchRenderPreparationServiceV1._select_perspective(
            analysis, decision, Seat.EAST
        )
        is opponent
    )
    assert ResearchRenderPreparationServiceV1._rendition_type(actor) == (
        "actor_challenge"
    )
    assert ResearchRenderPreparationServiceV1._rendition_type(partner) == (
        "partner_inference"
    )
    assert ResearchRenderPreparationServiceV1._rendition_type(opponent) == (
        "opponent_prediction"
    )
    with pytest.raises(ValueError, match="requested observer"):
        ResearchRenderPreparationServiceV1._select_perspective(
            analysis, decision, Seat.WEST
        )


def test_actor_actions_are_preserved_but_nonactor_input_never_receives_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    bundle = SimpleNamespace(
        bundle_sha256="f" * 64,
        value={"scenes": [], "timeline": {}},
        to_dict=lambda: {"bundleSha256": "f" * 64},
    )
    renderer = SimpleNamespace(
        build_perspective_render_request=lambda **kwargs: (
            calls.append(kwargs) or bundle
        )
    )
    artifacts = SimpleNamespace(
        put=lambda *_args, **_kwargs: {
            "uri": "cas://sha256/" + "e" * 64,
            "sha256": "e" * 64,
            "bytes": 1,
            "mediaType": "application/json",
        }
    )
    service = ResearchRenderPreparationServiceV1(
        research=SimpleNamespace(),
        artifact_store=artifacts,
        publication_service=SimpleNamespace(),
        renderer=renderer,
        rules=SimpleNamespace(),
    )
    monkeypatch.setattr(
        render_module,
        "TraceDecisionProjectionStoreV1",
        lambda *_args: SimpleNamespace(),
    )
    monkeypatch.setattr(
        render_module,
        "build_observer_projection",
        lambda *_args: SimpleNamespace(information_state={"plays": []}),
    )
    decision = SimpleNamespace(
        decision_id="D1",
        legal_actions=({"suit": "spades", "rank": "ace"},),
    )
    evidence = SimpleNamespace(
        decision_id="D1",
        acting_seat=Seat.SOUTH,
        observer_seat=Seat.NORTH,
        truth_mode=TruthMode.SEAT_EXPERIENCE,
        prompt_contract="read_partner_signal",
        recommended_action=None,
    )

    service._build_and_store(
        trace=SimpleNamespace(bundle_hash="b" * 64),
        decision=decision,
        evidence=evidence,
        analysis_hash="a" * 64,
        locator=SimpleNamespace(
            code="BX1",
            universal_link="https://example.test/c/BX1",
        ),
        reveal_policy="after_commit",
        preset="decision_clip",
        reason_key="video.observer.insight",
        rules=SimpleNamespace(),
    )

    assert calls[0]["legal_actions"] == ()
    assert calls[0]["evidence"].recommended_action is None
    assert calls[0]["variant"].rendition_type == "partner_inference"
    assert calls[0]["variant"].prompt_contract == "read_partner_signal"

    actor = SimpleNamespace(
        decision_id="D1",
        acting_seat=Seat.SOUTH,
        observer_seat=Seat.SOUTH,
        truth_mode=TruthMode.ACTOR_OBSERVER,
        prompt_contract="choose_acting_action",
        recommended_action={"suit": "spades", "rank": "ace"},
    )
    service._build_and_store(
        trace=SimpleNamespace(bundle_hash="b" * 64),
        decision=decision,
        evidence=actor,
        analysis_hash="a" * 64,
        locator=SimpleNamespace(
            code="BX1",
            universal_link="https://example.test/c/BX1",
        ),
        reveal_policy="after_commit",
        preset="decision_clip",
        reason_key="video.analysis.expert_reason",
        rules=SimpleNamespace(),
    )
    assert calls[1]["legal_actions"] == decision.legal_actions
    assert calls[1]["variant"].rendition_type == "actor_challenge"
    assert calls[1]["variant"].prompt_contract == "choose_acting_action"


def test_publication_accepts_verified_frozen_selection_and_video_evidence(
    root: Path,
) -> None:
    result, analysis, evidence, audit = _qualified_result(root)

    eligible = ResearchRenderPreparationServiceV1._publication_governance(
        result,
        audit=audit,
        analysis=analysis,
        trace=_trace_with_profile(root),
        evidence=evidence,
    )

    assert eligible[0] is True
    assert eligible[3] == []
    assert eligible[4]["decisionSelection"]["gatePassed"] is True
    assert eligible[4]["videoProduction"]["gatePassed"] is True


def test_frozen_video_profile_threshold_influences_publication(root: Path) -> None:
    result, analysis, evidence, audit = _qualified_result(root)

    strict = ResearchRenderPreparationServiceV1._publication_governance(
        result,
        audit=audit,
        analysis=analysis,
        trace=_trace_with_profile(root, minimum_video_score=80.0),
        evidence=evidence,
    )

    assert strict[0] is False
    assert "video_production_gate:below_selection_score" in strict[3]


def test_missing_decision_selection_evidence_stays_renderable_as_audit(
    root: Path,
) -> None:
    result, analysis, evidence, audit = _qualified_result(root)
    missing = copy.deepcopy(result)
    missing.pop("scoreEvidence")
    missing_evidence = ResearchRenderPreparationServiceV1._publication_governance(
        missing,
        audit=audit,
        analysis=analysis,
        trace=_trace_with_profile(root),
        evidence=evidence,
    )

    assert missing_evidence[0] is False
    assert "decision_selection_evidence_missing" in missing_evidence[3]


def test_tampered_decision_selection_hash_forces_audit_only(root: Path) -> None:
    result, analysis, evidence, audit = _qualified_result(root)
    tampered = copy.deepcopy(result)
    tampered["scoreEvidence"]["decisionChallengeScores"][0]["score"]["score"] = 99.0
    tampered_evidence = ResearchRenderPreparationServiceV1._publication_governance(
        tampered,
        audit=audit,
        analysis=analysis,
        trace=_trace_with_profile(root),
        evidence=evidence,
    )

    assert tampered_evidence[0] is False
    assert "decision_selection_score_evidence_hash_mismatch" in tampered_evidence[3]


def test_selected_decision_must_pass_its_frozen_gate(
    root: Path,
) -> None:
    result, analysis, evidence, audit = _qualified_result(root)
    score_evidence = result["scoreEvidence"]
    score_evidence["decisionChallengeScores"][0]["gatePassed"] = False
    selection_summary = score_evidence["decisionSelectionQualification"]
    selection_core = {
        **{
            key: copy.deepcopy(value)
            for key, value in selection_summary.items()
            if key != "sha256"
        },
        "decisions": copy.deepcopy(score_evidence["decisionChallengeScores"]),
    }
    selection_summary["sha256"] = _sha(selection_core)
    result["qualification"]["decisionSelection"]["evidenceHash"] = selection_summary[
        "sha256"
    ]
    score_core = {
        key: copy.deepcopy(value)
        for key, value in score_evidence.items()
        if key != "sha256"
    }
    score_evidence["sha256"] = _sha(score_core)

    governed = ResearchRenderPreparationServiceV1._publication_governance(
        result,
        audit=audit,
        analysis=analysis,
        trace=_trace_with_profile(root),
        evidence=evidence,
    )

    assert governed[0] is False
    assert "selected_decision_gate_failed" in governed[3]


def test_non_actor_is_always_audit_only_even_when_frozen_evidence_passes(
    root: Path,
) -> None:
    result, analysis, _, audit = _qualified_result(root)
    non_actor = SimpleNamespace(
        truth_mode=TruthMode.SEAT_EXPERIENCE,
        truth_label=TruthLabel.ROBUST_BEST_KNOWN,
        confidence=0.80,
    )

    governed = ResearchRenderPreparationServiceV1._publication_governance(
        result,
        audit=audit,
        analysis=analysis,
        trace=_trace_with_profile(root),
        evidence=non_actor,
    )

    assert governed[0] is False
    assert "non_actor_observer" in governed[3]


def _bundle_with_source(base: RenderBundleV1, *, audit_only: bool = False) -> RenderBundleV1:
    raw = base.to_dict()
    blockers = ["semantic_quarantine"] if audit_only else []
    if audit_only:
        decision_core = {
            "schemaVersion": 1,
            "profileHash": None,
            "evidenceHash": None,
            "decisionId": "D1",
            "gatePassed": False,
            "qualifiedDecision": False,
            "blockers": blockers,
        }
        video_core = {
            "schemaVersion": 1,
            "profileHash": None,
            "candidate": {"decisionId": "D1", "auditOnly": True},
            "gatePassed": False,
            "blockers": blockers,
        }
    else:
        decision_core = {"schemaVersion": 1, "gatePassed": True}
        video_core = {"schemaVersion": 1, "gatePassed": True}
    policy_core = {
        "schemaVersion": 1,
        "decisionSelection": {
            **decision_core,
            "sha256": _sha(decision_core),
        },
        "videoProduction": {**video_core, "sha256": _sha(video_core)},
        "blockers": blockers,
    }
    raw["source"] = {
        "sourceSchemaVersion": 2,
        "kind": "research_result",
        "code": "LAF-GOVERNED01",
        "decisionOrdinal": 1,
        "decisionId": "D1",
        "resultHash": "b" * 64,
        "analysisHash": "c" * 64,
        "auditHash": "d" * 64,
        "analysisProfileHash": "e" * 64,
        "analyzerId": "adaptive_finalist_trace_analyzer_v1",
        "analyzerVersion": "1.3.0",
        "rulesHash": "f" * 64,
        "exactStateId": "D1",
        "qrActionable": not audit_only,
        "qualificationStatus": "quarantined" if audit_only else "qualified",
        "renderPurpose": "research_audit" if audit_only else "content_candidate",
        "qualificationFailures": blockers,
        "publicationBlockers": blockers,
        "observerSeat": raw["observerSeat"],
        "auditOnly": audit_only,
        "publicationEligible": not audit_only,
        "publicationPolicy": {**policy_core, "sha256": _sha(policy_core)},
        "traceHash": "a" * 64,
    }
    core = {key: value for key, value in raw.items() if key != "bundleSha256"}
    raw["bundleSha256"] = _sha(core)
    bundle = RenderBundleV1(raw)
    assert bundle.verify()
    return bundle


def test_bundle_source_is_immutable_and_batch_rejects_caller_contradictions(
    fresh_professional_bundles,
) -> None:
    raw = _bundle_with_source(fresh_professional_bundles["decision_clip"]).to_dict()
    bundle = RenderBundleV1(raw)
    raw["source"]["renderPurpose"] = "research_audit"
    exported = bundle.to_dict()
    exported["source"]["publicationEligible"] = False

    assert bundle.value["source"]["renderPurpose"] == "content_candidate"
    assert bundle.value["source"]["auditOnly"] is False
    assert bundle.value["source"]["publicationEligible"] is True
    authoritative = RenderBatchServiceV1._validated_source_metadata(
        {"kind": "research_result", "code": "LAF-GOVERNED01"},
        bundle,
    )
    assert authoritative == bundle.value["source"]
    with pytest.raises(ValueError, match="contradicts the immutable bundle"):
        RenderBatchServiceV1._validated_source_metadata(
            {"renderPurpose": "research_audit"},
            bundle,
        )
    service = object.__new__(RenderBatchServiceV1)
    service._load_bundle = lambda _reference: bundle
    with pytest.raises(ValueError, match="contradicts the immutable bundle"):
        service.submit(
            (
                {
                    "basename": "contradictory",
                    "bundleArtifact": {},
                    "source": {"publicationEligible": False},
                },
            ),
            idempotency_key="contradictory-source",
        )

    contradictory = bundle.to_dict()
    contradictory["source"]["renderPurpose"] = "research_audit"
    contradictory_core = {
        key: value for key, value in contradictory.items() if key != "bundleSha256"
    }
    contradictory["bundleSha256"] = _sha(contradictory_core)
    assert RenderBundleV1(contradictory).verify() is False


def test_audit_only_bundle_and_batch_publication_flags_are_exact_inverses(
    fresh_professional_bundles,
) -> None:
    audit_bundle = _bundle_with_source(fresh_professional_bundles["decision_clip"], audit_only=True)
    audit_source = audit_bundle.value["source"]

    assert audit_source["qualificationStatus"] == "quarantined"
    assert audit_source["auditOnly"] is True
    assert audit_source["publicationEligible"] is False
    assert audit_source["renderPurpose"] == "research_audit"
    assert (
        RenderBatchServiceV1._validated_source_metadata(
            {
                "qualificationStatus": "quarantined",
                "auditOnly": True,
                "publicationEligible": False,
            },
            audit_bundle,
        )
        == audit_source
    )

    for caller_claim in (
        {"auditOnly": False},
        {"publicationEligible": True},
        {"renderPurpose": "content_candidate"},
    ):
        with pytest.raises(ValueError, match="contradicts the immutable bundle"):
            RenderBatchServiceV1._validated_source_metadata(
                caller_claim,
                audit_bundle,
            )

    service = object.__new__(RenderBatchServiceV1)
    service._load_bundle = lambda _reference: audit_bundle
    with pytest.raises(ValueError, match="contradicts the immutable bundle"):
        service.submit(
            (
                {
                    "basename": "quarantined-as-publishable",
                    "bundleArtifact": {},
                    "source": {
                        "qualificationStatus": "quarantined",
                        "auditOnly": False,
                        "publicationEligible": True,
                        "renderPurpose": "content_candidate",
                    },
                },
            ),
            idempotency_key="quarantined-as-publishable",
        )

    for source_changes in (
        {"auditOnly": False},
        {"publicationEligible": True},
    ):
        contradictory = audit_bundle.to_dict()
        contradictory["source"].update(source_changes)
        contradictory_core = {
            key: value for key, value in contradictory.items() if key != "bundleSha256"
        }
        contradictory["bundleSha256"] = _sha(contradictory_core)
        assert RenderBundleV1(contradictory).verify() is False

    missing_audit_only = audit_bundle.to_dict()
    missing_audit_only["source"].pop("auditOnly")
    missing_core = {
        key: value for key, value in missing_audit_only.items() if key != "bundleSha256"
    }
    missing_audit_only["bundleSha256"] = _sha(missing_core)
    assert RenderBundleV1(missing_audit_only).verify() is False

    legacy = audit_bundle.to_dict()
    legacy.pop("source")
    legacy_core = {key: value for key, value in legacy.items() if key != "bundleSha256"}
    legacy["bundleSha256"] = _sha(legacy_core)
    legacy_bundle = RenderBundleV1(legacy)
    assert legacy_bundle.verify()
    with pytest.raises(ValueError, match="quarantined render batch source"):
        RenderBatchServiceV1._validated_source_metadata(
            {
                "qualificationStatus": "quarantined",
                "auditOnly": False,
                "publicationEligible": True,
                "renderPurpose": "content_candidate",
            },
            legacy_bundle,
        )


def _retained_audit_bundle(base: RenderBundleV1) -> RenderBundleV1:
    value = _bundle_with_source(base, audit_only=True).to_dict()
    source = value["source"]
    for key in (
        "resultHash",
        "analysisHash",
        "auditHash",
        "analysisProfileHash",
        "analyzerId",
        "analyzerVersion",
    ):
        source.pop(key)
    source.update(
        {
            "kind": "laffa",
            "code": "LAF-AUD-AC1687085F",
            "decisionOrdinal": 2,
            "decisionId": "D1",
            "auditTraceCode": "LAF-AUD-AC1687085F",
            "recordHash": "8" * 64,
            "attemptStatus": "completed",
            "originKind": "research_audit_trace",
            "rulesHash": "6" * 64,
        }
    )
    core = {key: item for key, item in value.items() if key != "bundleSha256"}
    value["bundleSha256"] = _sha(core)
    bundle = RenderBundleV1(value)
    assert bundle.verify()
    return bundle


def _retained_audit_prepare_payload(bundle: RenderBundleV1) -> dict[str, object]:
    source = bundle.value["source"]
    return {
        "schemaVersion": 1,
        "laffaCode": source["code"],
        "auditTraceCode": source["auditTraceCode"],
        "recordHash": source["recordHash"],
        "traceHash": source["traceHash"],
        "decisionOrdinal": source["decisionOrdinal"],
        "decisionId": source["decisionId"],
        "observerSeat": source["observerSeat"],
        "exportPreset": "hooked_flagship",
        "qualificationStatus": source["qualificationStatus"],
        "auditOnly": True,
        "publicationEligible": False,
        "renderPurpose": "research_audit",
        "qualificationFailures": source["qualificationFailures"],
        "publicationBlockers": source["publicationBlockers"],
        "publicationPolicy": source["publicationPolicy"],
        "locator": {},
        "bundleArtifact": {
            "ownerKind": "render_bundle_v1",
            "ownerHash": bundle.bundle_sha256,
            "uri": "cas://sha256/" + "b" * 64,
            "sha256": "b" * 64,
            "bytes": 1,
            "mediaType": "application/vnd.balootx.render-bundle-v1+json",
            "name": "bundle.json",
        },
        "basename": "retained-audit",
        "preview": {},
        "source": copy.deepcopy(source),
    }


def test_retained_audit_prepare_model_and_batch_route_preserve_explicit_nulls(
    fresh_professional_bundles,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _retained_audit_bundle(fresh_professional_bundles["decision_clip"])
    source = bundle.value["source"]
    prepared = api.ResearchAuditTraceRenderPrepareResponse.model_validate(
        _retained_audit_prepare_payload(bundle)
    )
    prepared_source = prepared.source.model_dump(mode="json", exclude_unset=True)

    assert prepared_source == _retained_audit_prepare_payload(bundle)["source"]
    assert source["sourceSchemaVersion"] == 2
    assert source["exactStateId"] == source["decisionId"]
    assert source["qrActionable"] is False
    assert {
        "resultHash",
        "analysisHash",
        "auditHash",
        "analysisProfileHash",
        "analyzerId",
        "analyzerVersion",
    }.isdisjoint(source)
    assert (
        prepared_source["publicationPolicy"]["decisionSelection"]["profileHash"] is None
    )
    assert (
        prepared_source["publicationPolicy"]["decisionSelection"]["evidenceHash"]
        is None
    )
    assert (
        prepared_source["publicationPolicy"]["videoProduction"]["profileHash"] is None
    )

    body = api.FoundryRenderBatchCreateRequest.model_validate(
        {
            "items": [
                {
                    "bundleArtifact": prepared.bundleArtifact,
                    "basename": prepared.basename,
                    "source": prepared_source,
                }
            ]
        }
    )
    dumped_item = body.items[0].model_dump(mode="json", exclude_unset=True)
    assert dumped_item["source"] == prepared_source
    assert (
        RenderBatchServiceV1._validated_source_metadata(dumped_item["source"], bundle)
        == source
    )

    capture = SimpleNamespace(items=None, idempotency_key=None)

    def submit(items, *, idempotency_key, playback_evidence=False):
        capture.items = items
        capture.idempotency_key = idempotency_key
        return {"accepted": True}

    monkeypatch.setattr(api, "RENDER_BATCHES", SimpleNamespace(submit=submit))
    monkeypatch.setattr(api, "_require_trusted", lambda _request: None)
    result = api.create_foundry_render_batch(
        body,
        SimpleNamespace(headers={"idempotency-key": "retained-audit-roundtrip"}),
    )

    assert result == {"accepted": True}
    assert capture.idempotency_key == "retained-audit-roundtrip"
    assert capture.items == (dumped_item,)


def test_retained_audit_policy_omission_tamper_and_unknown_fields_fail_closed(
    fresh_professional_bundles,
) -> None:
    bundle = _retained_audit_bundle(fresh_professional_bundles["decision_clip"])
    artifact = _retained_audit_prepare_payload(bundle)["bundleArtifact"]

    omitted = copy.deepcopy(_retained_audit_prepare_payload(bundle)["source"])
    omitted["publicationPolicy"]["decisionSelection"].pop("profileHash")
    omitted["publicationPolicy"]["videoProduction"].pop("profileHash")
    omitted_body = api.FoundryRenderBatchCreateRequest.model_validate(
        {
            "items": [
                {
                    "bundleArtifact": artifact,
                    "basename": "omitted-null",
                    "source": omitted,
                }
            ]
        }
    )
    omitted_source = omitted_body.items[0].model_dump(mode="json", exclude_unset=True)[
        "source"
    ]
    assert "profileHash" not in omitted_source["publicationPolicy"]["decisionSelection"]
    assert "profileHash" not in omitted_source["publicationPolicy"]["videoProduction"]
    with pytest.raises(ValueError, match=r"\['publicationPolicy'\]"):
        RenderBatchServiceV1._validated_source_metadata(omitted_source, bundle)

    tampered = copy.deepcopy(_retained_audit_prepare_payload(bundle)["source"])
    tampered["publicationPolicy"]["decisionSelection"]["profileHash"] = "f" * 64
    tampered_body = api.FoundryRenderBatchCreateRequest.model_validate(
        {
            "items": [
                {
                    "bundleArtifact": artifact,
                    "basename": "tampered-policy",
                    "source": tampered,
                }
            ]
        }
    )
    tampered_source = tampered_body.items[0].model_dump(
        mode="json", exclude_unset=True
    )["source"]
    with pytest.raises(ValueError, match=r"\['publicationPolicy'\]"):
        RenderBatchServiceV1._validated_source_metadata(tampered_source, bundle)

    unknown = copy.deepcopy(_retained_audit_prepare_payload(bundle)["source"])
    unknown["publicationPolicy"]["decisionSelection"]["unexpected"] = True
    with pytest.raises(ValidationError):
        api.FoundryRenderBatchCreateRequest.model_validate(
            {
                "items": [
                    {
                        "bundleArtifact": artifact,
                        "basename": "unknown-policy-field",
                        "source": unknown,
                    }
                ]
            }
        )


def test_batch_and_on_demand_paths_use_the_same_video_policy_evaluator(
    root: Path,
) -> None:
    profile = HookedLaffaBatchProfileV1.build(
        _challenge_profile(root)["video_production_batch"]
    )
    candidate = {
        "decisionKind": "play",
        "critical": True,
        "truthLabel": "robust_best_known",
        "selectionScore": profile.minimum_selection_score - 0.01,
        "confidence": profile.minimum_confidence,
        "compatibilityBlockers": [],
    }

    blockers = evaluate_video_production_candidate_v1(candidate, profile)
    first_batch_blocker = FoundryVideoBatchServiceV1._candidate_gate_rejection_code(
        candidate,
        profile.to_dict(),
    )

    assert blockers == ("below_selection_score",)
    assert first_batch_blocker == blockers[0]
