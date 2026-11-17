from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.audit.service import ReplayAuditService
from baloot_platform.core.enums import Seat
from baloot_platform.hitl import ReviewGovernanceServiceV18, SQLiteReviewGovernanceStore
from baloot_platform.learning.service import LearningService
from baloot_platform.learning.store import LearningStore
from baloot_platform.service.game import GameManager
from baloot_platform.storage.game_store import SQLiteGameStore
from baloot_platform.video.hitl_review_v18 import HitlReviewClipRendererV18
from baloot_platform.video import media_validation

from hitl_helpers import hitl_profile, make_service, submit_review


def _eligible_case(root: Path, tmp_path: Path):
    service, store, _ = make_service(root, tmp_path)
    case = service.create_case("DEC-HITL-1", created_by="AUTHOR")
    service.set_consent("R1", allow_training=True)
    service.set_consent("R2", allow_training=True)
    submit_review(service, case["caseId"], "R1", rationale="عدّيت الورق وحفظ الدخول أفضل")
    submit_review(service, case["caseId"], "R2", rationale="إشارة الخوي تؤكد أن حفظ الدخول أقوى")
    return service, store, case


def test_blind_review_consensus_training_and_no_direct_engine_change(root: Path, tmp_path: Path):
    service, store, case = _eligible_case(root, tmp_path)
    blind = service.case(case["caseId"], blind=True)
    assert "productionAction" not in blind and blind["blind"] is True
    consensus = service.compute_consensus(case["caseId"])
    assert consensus["eligible"] is True and consensus["preferredAction"] == "C8"
    candidate = service.create_training_candidate(case["caseId"])
    assert candidate["automaticProductionPromotion"] is False
    service.record_challenger_evaluation(candidate["candidateId"], validator_id="SOLVER-A", validator_type="deep_solver", status="passed", metrics={"regret": 0.0})
    service.record_challenger_evaluation(candidate["candidateId"], validator_id="MODEL-B", validator_type="challenger", status="passed", metrics={"crossPlay": 0.61})
    accepted = service.accept_training_candidate(candidate["candidateId"], accepted_by="GOVERNOR")
    assert accepted["status"] == "accepted" and accepted["automaticProductionPromotion"] is False
    assert store.verify_events("case", case["caseId"])["valid"] is True


def test_hindsight_and_moderation_block_automatic_training(root: Path, tmp_path: Path):
    service, _store, _ = make_service(root, tmp_path)
    case = service.create_case("DEC-HITL-1", created_by="AUTHOR")
    service.set_consent("R1", allow_training=True)
    result = submit_review(service, case["caseId"], "R1", rationale="عرفت النتيجة ثم اقترحت البديل", hidden_seen=True, result_seen=True, disclosure="after_reveal")
    assert result["validation"]["hindsightSafe"] is False
    assert result["consensus"]["eligible"] is False
    with pytest.raises(ValueError):
        service.create_training_candidate(case["caseId"])


def test_observer_safe_review_clip_render(root: Path, tmp_path: Path):
    service, _store, _ = make_service(root, tmp_path)
    case = service.create_case("DEC-HITL-1", created_by="AUTHOR")
    profile = hitl_profile(root)["review_clip"]
    profile.update({"width": 360, "height": 640, "fps": 24, "duration_seconds": 0.5})
    renderer = HitlReviewClipRendererV18(profile)
    artifacts = renderer.render(case=service.case(case["caseId"], blind=True), output_dir=tmp_path, basename="review")
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    assert artifacts.video.stat().st_size > 1000
    assert manifest["observerSafe"] is True
    assert manifest["productionRecommendationIncluded"] is False
    assert manifest["crowdConsensusIncluded"] is False
    assert manifest["presentationMode"] == "player"
    assert manifest["developerCodesVisible"] is False
    assert manifest["caseLocatorCharacters"] == 8


@pytest.mark.skipif(os.name != "nt", reason="Windows long-path process boundary")
def test_media_command_uses_extended_executable_with_short_argv_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    executable = tmp_path / ("x" * 240) / "ffmpeg.exe"
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(media_validation.subprocess, "run", fake_run)
    media_validation.run_media_command([str(executable), "-version"])

    assert captured["command"] == ["ffmpeg.exe", "-version"]
    assert captured["kwargs"]["executable"].startswith("\\\\?\\")
    assert captured["kwargs"]["executable"].endswith("\\ffmpeg.exe")


def test_review_clip_copy_is_governed_and_not_embedded_in_renderer(root: Path):
    profile = hitl_profile(root)["review_clip"]
    assert profile["copy"]["stage_blind_choice"] == "وش تشوف اللعب الأفضل؟"
    source = (root / "src/baloot_platform/video/hitl_review_v18.py").read_text(encoding="utf-8")
    assert not any("\u0600" <= character <= "\u06ff" for character in source)


def test_hitl_rest_mcp_grpc_and_sdk_contract(root: Path):
    openapi = json.loads((root / "api/openapi.json").read_text(encoding="utf-8"))
    # The generated contract is refreshed during release verification; source surfaces must also exist.
    source = (root / "src/baloot_platform/service/routes/hitl.py").read_text(encoding="utf-8")
    mcp = (root / "src/baloot_platform/service/mcp_server.py").read_text(encoding="utf-8")
    proto = (root / "proto/baloot/v18/hitl.proto").read_text(encoding="utf-8")
    ts = (root / "sdk/typescript/src/client.ts").read_text(encoding="utf-8")
    go = (root / "sdk/go/client.go").read_text(encoding="utf-8")
    assert 'prefix="/api/v1/hitl"' in source and '@router.post("/cases"' in source
    assert "create_hitl_case" in mcp and "baloot://hitl/profile" in mcp
    assert "service HumanLearning" in proto and "SubmitReview" in proto
    assert "createHitlCase" in ts and "CreateHitlCase" in go
    assert isinstance(openapi.get("paths"), dict)


def test_player_review_ui_uses_governed_hitl_flow(root: Path):
    html = (root / "src/baloot_platform/service/static/hitl_studio.html").read_text(encoding="utf-8")
    assert "أشوف فيه لعب أفضل" in html
    assert "مراجعة عمياء" in html or "عمياء" in html
    assert "لن يغيّر المحرك مباشرة" in html or "لا يغيّر المحرك" in html
    assert "<textarea" in html and "JSON متقدم" not in html and "raw-json" not in html


def test_disclosure_timing_and_consent_revocation_are_enforced(root: Path, tmp_path: Path):
    service, store, _ = make_service(root, tmp_path)
    case = service.create_case("DEC-HITL-1", created_by="AUTHOR")
    service.set_consent("R1", allow_training=True)
    revoked = service.set_consent("R1", revoke=True)
    assert revoked["revokedAt"] and revoked["allowTraining"] is False
    events = store.list_events("consent", "R1")
    assert events[-1]["eventType"] == "consent_revoked"
    store.update_case(case["caseId"], hiddenRevealedAt="2026-07-29T00:01:00+00:00")
    result = submit_review(service, case["caseId"], "R1", rationale="مراجعة بعد كشف الورق", disclosure="after_reveal")
    assert result["validation"]["hindsightSafe"] is False


def test_post_game_reveal_metadata_blocks_hindsight_training(root: Path, tmp_path: Path):
    service, store, _ = make_service(root, tmp_path)
    case = service.create_case("DEC-HITL-1", created_by="AUTHOR", completed=True)
    store.update_case(case["caseId"], resultRevealedAt="2026-07-29T00:02:00+00:00")
    # A dishonest client cannot make an after-reveal review safe merely by setting booleans false.
    assignment = service.assign(case["caseId"], reviewer_id="R1", device_id="d1", network_id="n1")
    store_case = store.get_case(case["caseId"])
    assert store_case["resultRevealedAt"] is not None
    result = service.submit(case["caseId"], assignment_id=assignment["assignmentId"], assignment_token=assignment["assignmentToken"], reviewer_id="R1", preferred_action="C8", confidence=0.8, reason_codes=["card_counting"], rationale_ar="أراجع القرار بعد ظهور النتيجة", view_mode="observer", disclosure_timing="after_reveal")
    assert result["validation"]["hindsightSafe"] is False


def test_governed_improvement_proposal_never_auto_applies(root: Path, tmp_path: Path):
    service, _store, case = _eligible_case(root, tmp_path)
    proposal = service.create_improvement_proposal(created_by="EXPERT", target="search_configuration", title_ar="زيادة عمق البحث", rationale_ar="القرارات المتقاربة تحتاج بحثًا أعمق", source_case_ids=[case["caseId"]], requested_change={"depth": 12})
    assert proposal["automaticProductionApply"] is False
    for validator in ("VAL-A", "VAL-B"):
        service.evaluate_improvement(proposal["proposalId"], validator_id=validator, status="passed", evidence={"regression": True}, independent=True)
    approved = service.approve_improvement(proposal["proposalId"], approved_by="EXPERT", role="expert")
    assert approved["status"] == "approved_for_challenger"
    ready = service.finalize_improvement(proposal["proposalId"], release_governor_id="GOV", change_manifest={"artifactId": "ART-1", "bundleHash": "b" * 64, "rollbackManifest": {"model": "old"}, "regressionPassed": True, "shadowPassed": True, "canaryPassed": True})
    assert ready["status"] == "release_change_ready" and ready["automaticProductionApply"] is False


def test_correction_taxonomy_and_high_impact_adjudication_gate(root: Path, tmp_path: Path):
    service, _store, _ = make_service(root, tmp_path)
    with pytest.raises(ValueError, match="unsupported correction type"):
        service.create_case("DEC-HITL-1", created_by="AUTHOR", correction_type="guess")
    case = service.create_case(
        "DEC-HITL-1",
        created_by="AUTHOR",
        correction_type="tactical_convention",
        high_impact=True,
    )
    assert case["correctionType"] == "tactical_convention"
    for reviewer in ("R1", "R2"):
        service.set_consent(reviewer, allow_training=True)
        submit_review(service, case["caseId"], reviewer, rationale=f"مراجعة مستقلة {reviewer}")
    assert service.compute_consensus(case["caseId"])["eligible"] is True
    with pytest.raises(ValueError, match="requires expert adjudication"):
        service.create_training_candidate(case["caseId"])
    with pytest.raises(ValueError, match="preserve the supported blind consensus"):
        service.adjudicate(
            case["caseId"], adjudicator_id="REF", outcome="accepted",
            preferred_action="C7", rationale_ar="مراجعة الحكم النهائي",
        )
    accepted = service.adjudicate(
        case["caseId"], adjudicator_id="REF", outcome="accepted",
        preferred_action="C8", rationale_ar="مراجعة الحكم النهائي",
    )
    assert accepted["outcome"] == "accepted"
    assert service.create_training_candidate(case["caseId"])["status"] == "training_candidate"


def test_only_valid_supporting_reviews_enter_training_lineage(root: Path, tmp_path: Path):
    service, _store, case = _eligible_case(root, tmp_path)
    # This after-reveal submission is retained as evidence but cannot become a label.
    submit_review(
        service, case["caseId"], "R3", rationale="مراجعة بعد معرفة النتيجة",
        hidden_seen=True, result_seen=True, disclosure="after_reveal",
    )
    consensus = service.compute_consensus(case["caseId"])
    assert consensus["eligible"] is True
    candidate = service.create_training_candidate(case["caseId"])
    assert candidate["consentSubjects"] == ["R1", "R2"]
    assert {item["reviewerId"] for item in candidate["labelProvenance"]["human_preference"]} == {"R1", "R2"}
    assert len(candidate["labelProvenance"]["validated_submission_ids"]) == 2


def test_consent_withdrawal_invalidates_candidate_and_traces_descendants(root: Path, tmp_path: Path):
    service, store, case = _eligible_case(root, tmp_path)
    candidate = service.create_training_candidate(case["caseId"])
    service.record_challenger_evaluation(
        candidate["candidateId"], validator_id="SOLVER-A", validator_type="deep_solver",
        status="passed", metrics={"regret": 0.0},
    )
    service.record_challenger_evaluation(
        candidate["candidateId"], validator_id="MODEL-B", validator_type="challenger",
        status="passed", metrics={"crossPlay": 0.61},
    )
    service.accept_training_candidate(candidate["candidateId"], accepted_by="GOV")
    service.learning_store.list_datasets = lambda: [{
        "id": "DS-1",
        "manifest": {"metadata": {"governedHitlLineage": [{
            "candidateId": candidate["candidateId"], "consentSubjects": ["R1", "R2"],
        }]}},
    }]
    service.learning_store.list_models = lambda: [{"modelId": "MDL-1", "datasetId": "DS-1"}]

    revoked = service.set_consent("R1", revoke=True)
    impact = revoked["withdrawalImpact"]
    assert impact["candidateIds"] == [candidate["candidateId"]]
    assert impact["datasetIds"] == ["DS-1"] and impact["modelIds"] == ["MDL-1"]
    assert impact["productionMutationPerformed"] is False
    assert store.get_candidate(candidate["candidateId"])["status"] == "consent_withdrawn"
    with pytest.raises(ValueError, match="consent was withdrawn"):
        service.accept_training_candidate(candidate["candidateId"], accepted_by="GOV")
    consent_events = store.list_events("consent", "R1")
    assert any(event["eventType"] == "lineage_impact_recorded" for event in consent_events)
    assert consent_events[-1]["eventType"] == "consent_revoked"


def test_accepted_hitl_candidate_enters_dataset_with_frozen_consent_lineage(root: Path, tmp_path: Path):
    learning_store = LearningStore(tmp_path / "learning.sqlite3")
    manager = GameManager(
        root, SQLiteGameStore(tmp_path / "games.sqlite3"), learning_store=learning_store,
        agent_registry=AgentProfileRegistry.from_yaml(root / "configs/agents/adaptive_lineup_v1.yaml"),
    )
    learning = LearningService(root, tmp_path / "learning", store=learning_store, game_manager=manager)
    manager.learning_service = learning
    game = manager.create(
        seed=511, dealer=Seat.EAST,
        agent_profiles={seat: "beginner" for seat in ("north", "west", "south", "east")},
    )
    game = manager.advance_bots(game.id, stop_before=None, maximum_actions=100)
    decision = next(
        item for item in reversed(learning_store.list_decisions(game.id))
        if item["decisionKind"] == "play" and len(item["legalActions"]) > 1
    )
    audit = ReplayAuditService(root, learning_store, learning)
    report = audit.audit_decision(
        decision["decisionId"], deep=False, force=True,
        overrides={"requested_samples": 4, "maximum_sampling_attempts": 2000, "seed": 19},
    )
    governance_store = SQLiteReviewGovernanceStore(tmp_path / "hitl-bridge.sqlite3")
    service = ReviewGovernanceServiceV18(
        governance_store, learning_store, audit, hitl_profile(root),
        cluster_secret=b"hitl-bridge-cluster-secret-32byt",
    )
    case = service.create_case(decision["decisionId"], created_by="AUTHOR")
    for reviewer in ("R1", "R2"):
        service.set_consent(reviewer, allow_training=True)
        submit_review(
            service, case["caseId"], reviewer, preferred_action=report["bestAction"],
            rationale=f"مراجعة مستقلة ومعللة {reviewer}",
        )
    candidate = service.create_training_candidate(case["caseId"])
    for validator, validator_type in (("SOLVER", "deep_solver"), ("MODEL", "challenger")):
        service.record_challenger_evaluation(
            candidate["candidateId"], validator_id=validator, validator_type=validator_type,
            status="passed", metrics={"verified": True},
        )
    service.accept_training_candidate(candidate["candidateId"], accepted_by="GOV")
    learning.hitl_store = governance_store

    dataset = learning.build_dataset(minimum_confidence=0.5)
    lineage = dataset["manifestData"]["metadata"]["governedHitlLineage"]
    assert lineage == [{
        "candidateId": candidate["candidateId"], "caseId": case["caseId"],
        "decisionId": decision["decisionId"], "consentSubjects": ["R1", "R2"],
        "caseHash": case["caseHash"],
    }]
    rows = [json.loads(line) for line in Path(dataset["examples"]).read_text(encoding="utf-8").splitlines()]
    assert [row["source"] for row in rows] == ["governed_hitl_candidate"]
