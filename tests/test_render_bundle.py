from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from baloot_platform.audit.models import content_hash
from baloot_platform.video.media_validation import (
    REMOTION_RENDERER_VERSION,
    validate_renderer_identity,
)
from baloot_platform.video.render_bundle import (
    RENDER_SOURCE_KEYS,
    RenderBundleV1,
    _RETIRED_VOICE_DEFAULTS,
    build_export_timeline,
    build_render_bundle,
    normalize_render_source,
    validate_approved_v4_public_profile,
)
from baloot_platform.video.render_plan import MultiAngleRenderPlanV1, RenderBatchV1
from baloot_platform.video.scene import SceneDocumentV1, build_scene_timeline, reduce_scene_document


def _observer_view(observer: str = "south") -> dict[str, object]:
    hand = [
        {"id": 0, "rank": "7", "suit": "spades"},
        {"id": 9, "rank": "8", "suit": "hearts"},
    ]
    return {
        "observer": observer,
        "currentPlayer": observer,
        "dealer": "west",
        "buyer": "north",
        "contract": "sun",
        "trumpSuit": None,
        "matchScore": [24, 19],
        "currentTrick": {
            "leader": "north",
            "plays": [
                {"seat": "north", "card": {"id": 1, "rank": "8", "suit": "spades"}}
            ],
        },
        "hands": {
            "north": hand if observer == "north" else {"count": 3},
            "east": {"count": 3},
            "south": hand if observer == "south" else {"count": 3},
            "west": {"count": 3},
        },
    }


def _inputs(root: Path, observer: str = "south"):
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    common = {
        "observer_view": _observer_view(observer),
        "public_events": (),
        "effective_config_sha256": "a" * 64,
        "locale": "ar-SA",
        "direction": "rtl",
    }
    question = reduce_scene_document(
        stage="question",
        prompt_key="video.decision.choose_best_move",
        acting_seat=observer,
        prompt_contract="choose_acting_action",
        rendition_type="actor_challenge",
        legal_choices=(
            {"kind": "play_card", "card": {"id": 0, "rank": "7", "suit": "spades"}},
        ),
        **common,
    )
    reveal = reduce_scene_document(
        stage="reveal",
        prompt_key="video.decision.best_move",
        acting_seat=observer,
        prompt_contract="choose_acting_action",
        rendition_type="actor_challenge",
        legal_choices=(
            {"kind": "play_card", "card": {"id": 0, "rank": "7", "suit": "spades"}},
        ),
        analysis_facts={
            "recommendedAction": {
                "kind": "play_card",
                "card": {"id": 0, "rank": "7", "suit": "spades"},
            },
            "reasonKey": "video.analysis.expert_reason",
        },
        **common,
    )
    return profile, (question, reveal)


def _publication_policy(*, blockers: list[str]) -> dict[str, object]:
    decision_core = {"schemaVersion": 1, "gatePassed": not blockers}
    video_core = {"schemaVersion": 1, "gatePassed": not blockers}
    policy_core = {
        "schemaVersion": 1,
        "decisionSelection": {
            **decision_core,
            "sha256": content_hash(decision_core),
        },
        "videoProduction": {
            **video_core,
            "sha256": content_hash(video_core),
        },
        "blockers": blockers,
    }
    return {**policy_core, "sha256": content_hash(policy_core)}


@pytest.mark.parametrize("digest", ["a" * 64, None, "bad", "A" * 64, True])
def test_render_source_preserves_only_canonical_quality_evidence_hash(digest):
    source = _current_source()
    policy = source["publicationPolicy"]
    policy["qualityEvidenceHash"] = digest
    policy["sha256"] = content_hash({key: value for key, value in policy.items() if key != "sha256"})
    if digest == "a" * 64:
        normalized = normalize_render_source(source, observer="south", require_current=True)
        assert normalized["publicationPolicy"]["qualityEvidenceHash"] == digest
        policy["qualityEvidenceHash"] = "b" * 64
        with pytest.raises(ValueError, match="publication policy evidence is invalid"):
            normalize_render_source(source, observer="south", require_current=True)
    else:
        with pytest.raises(ValueError, match="quality evidence hash is invalid"):
            normalize_render_source(source, observer="south", require_current=True)


@pytest.mark.parametrize("admission", [
    {"passed": True, "evidenceHash": "a" * 64},
    {"passed": False, "evidenceHash": "b" * 64},
    None, {}, {"passed": 1, "evidenceHash": "a" * 64},
    {"passed": True, "evidenceHash": "A" * 64},
    {"passed": True, "evidenceHash": "bad"},
    {"passed": True, "evidenceHash": "a" * 64, "alternatives": ["SA"]},
])
def test_render_source_interest_admission_is_closed_strict_and_hash_bound(admission):
    source = _current_source()
    policy = source["publicationPolicy"]
    policy["interestAdmission"] = admission
    policy["sha256"] = content_hash({key: value for key, value in policy.items() if key != "sha256"})
    if isinstance(admission, dict) and type(admission.get("passed")) is bool and len(admission) == 2:
        digest = admission["evidenceHash"]
        valid = len(digest) == 64 and digest == digest.lower()
    else:
        valid = False
    if not valid:
        with pytest.raises(ValueError, match="interest admission"):
            normalize_render_source(source, observer="south", require_current=True)
        return
    normalized = normalize_render_source(source, observer="south", require_current=True)
    assert normalized["publicationPolicy"]["interestAdmission"] == admission
    policy["interestAdmission"]["passed"] = not admission["passed"]
    with pytest.raises(ValueError, match="publication policy evidence is invalid"):
        normalize_render_source(source, observer="south", require_current=True)


def test_real_public_interest_producer_normalizes_without_private_verdict():
    from baloot_platform.research.render_service import ResearchRenderPreparationServiceV1
    source = _current_source()
    policy = source["publicationPolicy"]
    policy["interestVerdict"] = {"passed": True, "sha256": "a" * 64,
        "meaningfulDecisions": [{"baselineAction": "SA", "baselineRegret": 34}]}
    source["publicationPolicy"] = ResearchRenderPreparationServiceV1._public_scenario_publication_policy(
        policy, result_code="D1")
    normalized = normalize_render_source(source, observer="south", require_current=True)
    assert normalized["publicationPolicy"]["interestAdmission"] == {"passed": True, "evidenceHash": "a" * 64}
    assert "interestVerdict" not in normalized["publicationPolicy"]


def _current_source(*, kind: str = "research_result") -> dict[str, object]:
    audit = kind != "research_result"
    blockers = ["legacy_content_qualification_not_established"] if audit else []
    source: dict[str, object] = {
        "sourceSchemaVersion": 2,
        "kind": kind,
        "code": "BX-AUDIT01" if audit else "LAF-RENDER01",
        "decisionOrdinal": 1,
        "decisionId": "D1",
        "traceHash": "1" * 64,
        "rulesHash": "2" * 64,
        "exactStateId": "D1",
        "qrActionable": not audit,
        "qualificationStatus": "legacy_replay" if audit else "qualified",
        "renderPurpose": "research_audit" if audit else "content_candidate",
        "qualificationFailures": blockers,
        "publicationBlockers": blockers,
        "observerSeat": "south",
        "auditOnly": audit,
        "publicationEligible": not audit,
        "publicationPolicy": _publication_policy(blockers=blockers),
    }
    if not audit:
        source.update(
            {
                "resultHash": "3" * 64,
                "analysisHash": "4" * 64,
                "auditHash": "5" * 64,
                "analysisProfileHash": "6" * 64,
                "analyzerId": "adaptive_finalist_trace_analyzer_v1",
                "analyzerVersion": "1.3.0",
            }
        )
    return source


def test_export_fps_override_preserves_full_and_historical_profiles(root: Path) -> None:
    from copy import deepcopy
    from baloot_platform.video.render_bundle import _export_render, _validate_export_timeline

    profile, scenes = _inputs(root)
    clip = build_export_timeline(scenes=scenes, profile=profile, export_preset="decision_clip")
    assert clip.value["frameRate"] == {"numerator": 30, "denominator": 1}
    assert clip.value["totalFrames"] == 600
    _validate_export_timeline(clip, profile=profile, export_preset="decision_clip", scenes=scenes)
    assert _export_render(profile, "hooked_flagship")["fps"] == profile["render"]["fps"] == 60
    legacy = deepcopy(profile)
    del legacy["exports"]["decision_clip"]["fps"]
    assert _export_render(legacy, "decision_clip")["fps"] == 60
    assert profile["render"]["fps"] == 60
    for invalid in (True, "30", 24):
        legacy["exports"]["decision_clip"]["fps"] = invalid
        with pytest.raises(ValueError, match="export fps"):
            build_export_timeline(scenes=scenes, profile=legacy, export_preset="decision_clip")


def test_render_bundle_is_deterministic_observer_safe_and_self_verifying(
    root: Path,
) -> None:
    profile, scenes = _inputs(root)
    timeline = build_export_timeline(
        scenes=scenes, profile=profile, export_preset="decision_clip"
    )
    kwargs = {
        "scenes": scenes,
        "timeline": timeline,
        "profile": profile,
        "export_preset": "decision_clip",
        "localizations": {
            "video.decision.choose_best_move": "وش تلعب؟",
            "video.decision.best_move": "الاختيار",
            "video.analysis.expert_reason": "يحفظ ورقة الدخول ويضغط مسار الصن.",
        },
        "identity": {
            "gameSerial": "BLT-TEST-0001",
            "sceneId": scenes[0].scene_id,
            "enrollmentUri": f"balootx://game/BLT-TEST-0001?scene={scenes[0].scene_id}",
            "players": {
                "north": "Player 01",
                "west": "Player 04",
                "east": "Player 02",
                "south": "Player 03",
            },
        },
        "repository_root": root,
    }
    first = build_render_bundle(**kwargs)
    second = build_render_bundle(**kwargs)
    encoded = json.dumps(first.to_dict(), ensure_ascii=False, sort_keys=True)

    assert first.verify()
    assert first.to_dict() == second.to_dict()
    assert first.bundle_sha256 == second.bundle_sha256
    assert first.value["renderer"]["plugin"] == "remotion_professional"
    assert first.value["renderer"]["version"] == REMOTION_RENDERER_VERSION
    assert len(first.value["renderer"]["sourceSha256"]) == 64
    assert first.value["timeline"]["totalFrames"] == 600
    assert [item["stage"] for item in first.value["timeline"]["segments"]] == [
        "hook",
        "rewind",
        "context",
        "question",
        "reveal",
        "continuation",
        "outcome",
        "hold",
    ]
    assert first.value["presentation"]["timeline"]["project_call_seconds"] == 6.0
    assert first.value["presentation"]["timeline"]["project_reveal_seconds"] == 6.0
    assert first.value["identity"]["gameSerial"] == "BLT-TEST-0001"
    assert first.value["audioCues"] == []
    audio = first.value["presentation"]["audio"]
    assert audio["voice_callouts"] is False
    assert audio["spoken_voice_pack"] is False
    assert audio["voice_pack_directory"] == ""
    assert audio["explanation_voice_localization_key"] == ""
    assert set(first.value["identity"]["players"]) == {"north", "west", "east", "south"}
    assert {item["id"] for item in first.value["assets"]} == {
        "balootx-professional-card-back",
        "goodall_modern_baloot_32",
        "noto-kufi-arabic-bold",
        "noto-sans-arabic-bold",
        "noto-sans-arabic-regular",
    }
    for forbidden in (
        "deckOrder",
        "initialHands",
        "legalCards",
        "futureCards",
        "trustedState",
    ):
        assert forbidden not in encoded

    # Opaque provenance hashes cannot authorize a substituted context base layer.
    changed = first.to_dict()
    context = reduce_scene_document(
        observer_view=_observer_view(), public_events=(), effective_config_sha256="a" * 64,
        stage="context", locale="ar-SA", direction="rtl",
    ).to_dict()
    next(layer["content"] for layer in context["layers"] if layer["id"] == "table")["matchScore"] = [99, 99]
    context["sceneId"] = content_hash({key: value for key, value in context.items()
                                        if key not in {"sceneId", "fieldDisclosures"}})
    assert SceneDocumentV1(context).verify()
    changed["scenes"].append(context)
    changed["bundleSha256"] = content_hash({key: value for key, value in changed.items() if key != "bundleSha256"})
    assert not RenderBundleV1(changed).verify()


def test_render_source_v2_is_kind_specific_and_historical_boundary_is_closed() -> None:
    research = normalize_render_source(_current_source(), observer="south")
    replay = normalize_render_source(
        _current_source(kind="laffa"),
        observer="south",
    )

    assert research["resultHash"] == "3" * 64
    assert research["exactStateId"] == research["decisionId"]
    assert research["qrActionable"] is True
    assert replay["exactStateId"] == replay["decisionId"]
    assert replay["qrActionable"] is False
    assert all("receipt" not in key.lower() for key in RENDER_SOURCE_KEYS)

    unmarked = _current_source(kind="laffa")
    del unmarked["sourceSchemaVersion"]
    with pytest.raises(ValueError, match="current-only fields"):
        normalize_render_source(unmarked, observer="south")

    foreign = _current_source(kind="laffa")
    foreign["resultHash"] = "3" * 64
    with pytest.raises(ValueError, match="Research-result lineage"):
        normalize_render_source(foreign, observer="south")

    scenario = _current_source(kind="laffa")
    scenario.update(
        {
            "kind": "research_scenario_result",
            "code": "LAF-SCENARIO01",
            "qualificationStatus": "qualified",
            "renderPurpose": "content_candidate",
            "qualificationFailures": [],
            "publicationBlockers": [],
            "auditOnly": False,
            "publicationEligible": True,
            "publicationPolicy": _publication_policy(blockers=[]),
            "qrActionable": False,
            "resultHash": "3" * 64,
            "scenarioHash": "4" * 64,
            "proofHash": "5" * 64,
            "constraintEvidenceHash": "6" * 64,
            "exactStateHash": "7" * 64,
        }
    )
    normalized_scenario = normalize_render_source(scenario, observer="south")
    assert normalized_scenario["proofHash"] == "5" * 64
    scenario["analysisHash"] = "8" * 64
    with pytest.raises(ValueError, match="Research-result lineage"):
        normalize_render_source(scenario, observer="south")

    missing_lineage = _current_source()
    del missing_lineage["analysisProfileHash"]
    with pytest.raises(ValueError, match="incomplete"):
        normalize_render_source(missing_lineage, observer="south")


def test_render_source_v3_exposes_only_public_scenario_references() -> None:
    code = "LAF-PUBLIC0001"
    decision_core = {
        "schemaVersion": 1,
        "profileHash": "8" * 64,
        "evidenceHash": None,
        "decisionId": code,
        "gatePassed": True,
        "qualifiedDecision": True,
        "blockers": [],
    }
    video_core = {"schemaVersion": 1, "gatePassed": True}
    policy_core = {
        "schemaVersion": 1,
        "decisionSelection": {
            **decision_core,
            "sha256": content_hash(decision_core),
        },
        "videoProduction": {
            **video_core,
            "sha256": content_hash(video_core),
        },
        "blockers": [],
    }
    source = {
        "sourceSchemaVersion": 3,
        "kind": "research_scenario_result",
        "code": code,
        "decisionOrdinal": 1,
        "decisionReference": {"resultCode": code, "decisionOrdinal": 1},
        "publicProjectionHash": "9" * 64,
        "qualificationStatus": "qualified",
        "renderPurpose": "content_candidate",
        "qualificationFailures": [],
        "publicationBlockers": [],
        "observerSeat": "south",
        "auditOnly": False,
        "publicationEligible": True,
        "qrActionable": False,
        "publicationPolicy": {
            **policy_core,
            "sha256": content_hash(policy_core),
        },
        "rulesHash": "2" * 64,
        "proofReference": code,
        "proofCompletionStatus": "exact_complete",
        "proofClaimSatisfied": True,
        "proofAchievedMode": "exact",
    }

    assert normalize_render_source(source, observer="south") == source
    for field in (
        "decisionId",
        "exactStateId",
        "traceHash",
        "resultHash",
        "scenarioHash",
        "proofHash",
        "constraintEvidenceHash",
        "exactStateHash",
    ):
        leaked = dict(source)
        leaked[field] = "a" * 64
        with pytest.raises(ValueError, match="private or foreign fields"):
            normalize_render_source(leaked, observer="south")

    leaked_policy = json.loads(json.dumps(source))
    leaked_decision = leaked_policy["publicationPolicy"]["decisionSelection"]
    leaked_decision["evidenceHash"] = "a" * 64
    decision_hash_core = dict(leaked_decision)
    decision_hash_core.pop("sha256")
    leaked_decision["sha256"] = content_hash(decision_hash_core)
    policy_hash_core = dict(leaked_policy["publicationPolicy"])
    policy_hash_core.pop("sha256")
    leaked_policy["publicationPolicy"]["sha256"] = content_hash(policy_hash_core)
    with pytest.raises(ValueError, match="private proof identity"):
        normalize_render_source(leaked_policy, observer="south")


def test_historical_golden_media_is_intact_but_not_currently_admissible(root: Path) -> None:
    originals = os.environ.get("BALOOTX_PRIVATE_GOLDENS_DIR")
    if not originals:
        pytest.skip("needs private original artifacts")
    output = Path(originals)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["exportPreset"] == "hooked_flagship"
    assert len(manifest["entries"]) == 3
    assert manifest["entries"][-1]["fixture"] == (
        "examples/research-scenario-v4-ace-sacrifice-two-aces.json"
    )
    for entry in manifest["entries"]:
        assert entry["exportPreset"] == "hooked_flagship"
        bundle_path = output / entry["bundle"]
        assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == entry[
            "bundleFileSha256"
        ]
        bundle = RenderBundleV1(
            json.loads(bundle_path.read_text(encoding="utf-8"))
        )
        # Historical encoded evidence is not proof against the current renderer.
        with pytest.raises(ValueError, match="current renderer source"):
            validate_renderer_identity(bundle.value["renderer"], root=root)
        assert bundle.verify() is False
        assert bundle.value["exportPreset"] == "hooked_flagship"
        assert bundle.value["timeline"]["totalFrames"] == 4320
        render_manifest = json.loads(
            (output / entry["renderManifest"]).read_text(encoding="utf-8")
        )
        assert render_manifest["exportPreset"] == "hooked_flagship"
        assert render_manifest["mediaProbe"]["durationSeconds"] == 72.0
        video_path = output / entry["video"]
        assert hashlib.sha256(video_path.read_bytes()).hexdigest() == entry[
            "videoSha256"
        ]



def test_bundle_with_internally_consistent_but_stale_renderer_identity_is_rejected(
    root: Path,
) -> None:
    profile, scenes = _inputs(root)
    bundle = build_render_bundle(
        scenes=scenes,
        timeline=build_export_timeline(
            scenes=scenes, profile=profile, export_preset="decision_clip"
        ),
        profile=profile,
        export_preset="decision_clip",
        localizations=profile["localizations"],
        identity={
            "gameSerial": "STALE1",
            "sceneId": scenes[0].scene_id,
            "enrollmentUri": "https://balootx.local/challenge/STALE1",
            "players": {"north": "N", "west": "W", "south": "S", "east": "E"},
        },
        repository_root=root,
    ).to_dict()
    bundle["renderer"]["sourceSha256"] = "0" * 64
    unsigned = dict(bundle)
    unsigned.pop("bundleSha256")
    bundle["bundleSha256"] = content_hash(unsigned)
    assert not RenderBundleV1(bundle).verify()


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("disabled", None),
        ("allowed", None),
        ("formatted_name", None),
        ("unformatted_name", "video.decision.your_turn"),
        ("justification", "outside the allowlist: video.audio.entry_control_explanation"),
        ("legacy_justification", "unsupported voice setting .*video.audio.entry_control_explanation"),
        ("retired_pack", "voice_pack_directory"),
        ("retired_asset", "assets/.*/path"),
        ("retired_provider", "retired voice pack reference at presentation/audio/voice_provider"),
        ("outside_allowlist", "outside the allowlist: video.decision.reveal_card"),
        ("wrong_stage", "voice key is not visible on screen: video.decision.choose_best_move"),
        ("different_text", "video.decision.choose_best_move"),
        ("hidden_notice", "video.notice.sentence"),
        ("retired_notice_title", "outside the allowlist: video.notice.title"),
        ("rationale_field", "video.decision.choose_best_move"),
        ("rationale_alias", "voice cue uses a rationale key: video.decision.choose_best_move"),
    ],
)
def test_render_bundle_voice_guard(root: Path, case: str, error: str | None) -> None:
    profile, scenes = _inputs(root)
    args = {
        "scenes": scenes,
        "timeline": build_export_timeline(
            scenes=scenes, profile=profile, export_preset="decision_clip"
        ),
        "profile": profile,
        "export_preset": "decision_clip",
        "localizations": profile["localizations"],
        "identity": {
            "gameSerial": "BLT-VOICE-1",
            "sceneId": scenes[0].scene_id,
            "enrollmentUri": "https://balootx.local/c/VOICE1",
            "players": {"north": "N", "west": "W", "east": "E", "south": "S"},
        },
        "repository_root": root,
    }
    justification_key = "video.audio.entry_control_explanation"
    if case == "legacy_justification":
        default_audio = profile["audio"].copy()
        profile["audio"].update(
            spoken_voice_pack=True,
            one_piece_explanation=True,
            explanation_voice_localization_key=justification_key,
        )
        with pytest.raises(ValueError, match=error):
            build_render_bundle(**args)
        profile["audio"] = default_audio
    value = build_render_bundle(**args).to_dict()
    if case == "legacy_justification":
        value["presentation"]["audio"]["voice_callouts"] = True
        value["presentation"]["audio"]["explanation_voice_localization_key"] = justification_key
    elif case == "retired_pack":
        value["presentation"]["audio"]["voice_pack_directory"] = (
            "assets/audio/voice_packs/ar_sa_baloot_v2_smooth"
        )
    elif case == "retired_asset":
        value["assets"][0]["path"] = "assets/audio/voice_packs/ar_sa_baloot_v2_smooth/clip.wav"
    elif case == "retired_provider":
        value["presentation"]["audio"]["voice_provider"] = "offline_smooth_pack_v2"
    elif case != "disabled":
        key = "video.decision.choose_best_move"
        stage = "reveal" if case == "wrong_stage" else "question"
        if case == "justification":
            key = justification_key
        elif case == "outside_allowlist":
            key = "video.decision.reveal_card"
            value["localizations"][key] = "fixture"
        elif case == "hidden_notice":
            key = "video.notice.sentence"
        elif case == "retired_notice_title":
            key = "video.notice.title"
            value["presentation"]["majlis"]["research_notice"] = True
            value["localizations"][key] = "ملاحظة بحثية"
        elif case in {"formatted_name", "unformatted_name"}:
            key = "video.decision.your_turn"
        text = value["localizations"].get(key, "fixture")
        if case == "different_text":
            text += " "
        elif case == "formatted_name":
            text = text.replace("{name}", "S")
        asset = {"id": "unit-voice", "path": "assets/audio/unit-voice.wav"}
        cue = {
            "stage": stage,
            "assetId": asset["id"],
            "path": "audio/unit-voice.wav",
            "volume": 0.5,
            "transcriptKey": key,
            "transcriptSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        if case == "rationale_field":
            cue["rationale"] = "fixture"
        elif case == "rationale_alias":
            reveal = next(scene for scene in value["scenes"] if scene["stage"] == "reveal")
            analysis = next(layer["content"] for layer in reveal["layers"] if layer["id"] == "analysis")
            analysis["reasonKey"] = key
        value["assets"].append(asset)
        value["audioCues"].append(cue)
    value.pop("bundleSha256")
    value["bundleSha256"] = content_hash(value)
    bundle = RenderBundleV1(value)
    if error is None:
        assert bundle.verify(raise_voice_errors=True)
    else:
        assert not bundle.verify()
        with pytest.raises(ValueError, match=error):
            bundle.verify(raise_voice_errors=True)


def test_public_notice_profile_builds_a_qualified_bundle_with_the_exact_notice():
    from scripts import render_research_notice as notice

    profile = notice.notice_profile()
    design = profile["approved_design"]
    assert design["variant"] == "public_v4_no_justification_red_notice"
    assert design["reference_variant"] == "public_v4_no_justification_no_notice"
    assert design["reference_mp4_sha256"] == "15492d3f389fcb07be423fc97a6e46fe0b8e385b5beb173f01073e6d11a87bf9"

    rules = notice.RuleBook.load(notice.ROOT / "configs/rules/saudi_common_v1.yaml")
    bundle = notice.build_notice_bundle(rules, profile)
    notice.validate_notice_bundle(bundle, notice.engine_round_result(rules))
    sentence = bundle.value["localizations"]["video.notice.sentence"]
    assert "video.notice.title" not in profile["localizations"]
    assert "video.notice.title" not in bundle.value["localizations"]
    assert sentence == "هذا الفيديو لأغراض البحث فقط، ويُمنع تداوله أو نشره أو مشاركته"
    assert len(sentence.encode("utf-8")) == 114
    assert hashlib.sha256(sentence.encode("utf-8")).hexdigest() == "290e2a2d03424ea7846a71afc80e019140fac3d56e465b4766a89d57a3c3b8ac"
    assert bundle.value["presentation"]["majlis"]["research_notice"] is True
    presentation = bundle.value["presentation"]["presentation"]
    assert presentation["show_justification"] is presentation["show_justification_text"] is False


@pytest.mark.parametrize("override", [None, "BALOOTX_LAYOUT_TABLE", "BALOOTX_LAYOUT_TABLE_SHA256"])
def test_renderer_rejects_unapproved_layout_table_before_render(
    root, tmp_path, monkeypatch, override
):
    from scripts import render_research_notice as notice

    profile = notice.notice_profile()
    rules = notice.RuleBook.load(root / "configs/rules/saudi_common_v1.yaml")
    bundle = notice.build_notice_bundle(rules, profile)
    renderer = notice.RemotionProfessionalRenderer(profile, repository_root=root)
    table_path = Path("scripts/approved-layout-adjacencies.json")
    approved_layout_bytes = (renderer.renderer_root / table_path).read_bytes()
    renderer.renderer_root = tmp_path
    table = tmp_path / table_path
    table.parent.mkdir()
    table.write_bytes(approved_layout_bytes if override else approved_layout_bytes + b"\n")
    if override:
        monkeypatch.setenv(override, "unapproved")
    monkeypatch.setattr(renderer, "_run_renderer", lambda **_: pytest.fail("renderer started"))

    error = "overrides require NEVER_POST" if override else "approved layout table SHA-256 mismatch"
    with pytest.raises(ValueError, match=error):
        renderer.execute(bundle, export_preset="decision_clip", output_dir=tmp_path)


@pytest.mark.parametrize(
    ("legacy_profile", "timeline_sha"),
    [
        pytest.param(
            True, "553f7f627320b073462b9eb7bfc688703c22684fe8df4929c53c50b7cba5ae0f", id="legacy-voice-profile",
        ),
        pytest.param(
            False, "906beef957f3af75da9b62f0957e4fc3d79e954922fab6aca19b9679d0d2c571", id="voice-off-profile",
        ),
    ],
)
def test_public_notice_identity_and_five_visible_voice_phrases(
    root: Path, legacy_profile: bool, timeline_sha: str,
) -> None:
    from scripts import render_research_notice as notice

    rules = notice.RuleBook.load(root / "configs/rules/saudi_common_v1.yaml")
    profile = notice.notice_profile()
    if legacy_profile:
        # Recover the pre-guard profile without copying a second justification transcript.
        profile["audio"].update(_RETIRED_VOICE_DEFAULTS, voice_callouts=True, spoken_voice_pack=True)
        key = _RETIRED_VOICE_DEFAULTS["explanation_voice_localization_key"]
        profile["audio"]["explanation_voice_transcript"] = profile["localizations"][key]
    value = notice.build_notice_bundle(rules, profile).to_dict()
    assert value["identity"]["gameSerial"] == "LAF-98FBED199A"
    assert tuple(value["identity"]["players"][seat] for seat in ("north", "west", "east", "south")) == (
        "Player 15", "Player 11", "Player 07", "Player 03",
    )
    assert content_hash(value["timeline"]) == timeline_sha
    audio = value["presentation"]["audio"]
    assert not audio.get("voice_pack_directory") and not audio.get("voice_provider")
    assert audio["voice_callouts"] is audio["spoken_voice_pack"] is False
    assert value["audioCues"] == []
    question = next(scene for scene in value["scenes"] if scene["stage"] == "question")
    assert (question["locale"], question["direction"]) == ("ar-SA", "rtl")
    prompt = next(layer["content"] for layer in question["layers"] if layer["id"] == "decision-prompt")
    name = value["identity"]["players"][prompt["actingSeat"]]
    value["assets"].append({"id": "unit-voice", "path": "assets/audio/unit-voice.wav"})
    for key, stage in (
        ("video.decision.your_turn", "question"),
        ("video.decision.choose_best_move", "question"),
        ("video.notice.sentence", "setup"),
        ("video.decision.best_move", "reveal"),
        ("video.result.title", "outcome"),
    ):
        text = value["localizations"][key].replace("{name}", name)
        value["audioCues"].append({
            "stage": stage, "assetId": "unit-voice", "path": "audio/unit-voice.wav",
            "volume": 0.5, "transcriptKey": key,
            "transcriptSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    value.pop("bundleSha256")
    value["bundleSha256"] = content_hash(value)
    assert RenderBundleV1(value).verify(raise_voice_errors=True)


def test_public_notice_identity_follows_the_deal_and_name_pool(root: Path) -> None:
    from scripts import render_research_notice as notice
    from scripts.render_research_scenario_v4_goldens import _fixture_result_code

    rules = notice.RuleBook.load(root / "configs/rules/saudi_common_v1.yaml")
    profile = notice.notice_profile()
    profile["identity"]["player_name_pool"] = list("ABCD")
    identity = notice.build_notice_bundle(rules, profile).value["identity"]
    assert identity["gameSerial"] == "LAF-98FBED199A"
    assert set(identity["players"].values()) == set("ABCD")
    assert _fixture_result_code("scenario", "deal-a") != _fixture_result_code("scenario", "deal-b")


def test_render_bundle_rejects_ungoverned_export_and_tampering(root: Path) -> None:
    profile, scenes = _inputs(root)
    with pytest.raises(ValueError, match="unsupported render export preset"):
        build_render_bundle(
            scenes=scenes,
            timeline=build_export_timeline(
                scenes=scenes, profile=profile, export_preset="decision_clip"
            ),
            profile=profile,
            export_preset="social_teaser",
            localizations={"prompt": "قرار"},
            identity={
                "gameSerial": "BLT-TEST-0001",
                "sceneId": scenes[0].scene_id,
                "enrollmentUri": "balootx://game/BLT-TEST-0001",
                "players": {
                    "north": "Player 01",
                    "west": "Player 04",
                    "east": "Player 02",
                    "south": "Player 03",
                },
            },
            repository_root=root,
        )

    bundle = build_render_bundle(
        scenes=scenes,
        timeline=build_export_timeline(
            scenes=scenes, profile=profile, export_preset="hooked_flagship"
        ),
        profile=profile,
        export_preset="hooked_flagship",
        localizations={"prompt": "قرار"},
        identity={
            "gameSerial": "BLT-TEST-0001",
            "sceneId": scenes[0].scene_id,
            "enrollmentUri": "balootx://game/BLT-TEST-0001",
            "players": {
                "north": "Player 01",
                "west": "Player 04",
                "east": "Player 02",
                "south": "Player 03",
            },
        },
        repository_root=root,
    ).to_dict()
    assert bundle["timeline"]["totalFrames"] == 4320
    bundle["presentation"]["layout"]["safe_margin"] += 1
    assert not RenderBundleV1(bundle).verify()


def test_render_bundle_rejects_incomplete_public_identity(root: Path) -> None:
    profile, scenes = _inputs(root)
    with pytest.raises(ValueError, match="one player name for every seat"):
        build_render_bundle(
            scenes=scenes,
            timeline=build_export_timeline(
                scenes=scenes, profile=profile, export_preset="decision_clip"
            ),
            profile=profile,
            export_preset="decision_clip",
            localizations={"prompt": "قرار"},
            identity={
                "gameSerial": "BLT-TEST-0001",
                "sceneId": scenes[0].scene_id,
                "enrollmentUri": "balootx://game/BLT-TEST-0001",
                "players": {"north": "Player 01"},
            },
            repository_root=root,
        )


def test_render_bundle_preserves_the_supplied_authoritative_timeline(
    root: Path,
) -> None:
    profile, scenes = _inputs(root)
    profile["exports"]["decision_clip"].pop("fps", None)
    profile["exports"]["decision_clip"].pop("outcome_seconds", None)
    profile["exports"]["decision_clip"].update(minimum_seconds=28, target_seconds=28, maximum_seconds=28)
    question, reveal = scenes
    timeline = build_scene_timeline(
        fps=60,
        width=1080,
        height=1920,
        segments=(
            ("hook", question.scene_id, 2.0),
            ("rewind", question.scene_id, 0.2),
            ("context", question.scene_id, 9.0),
            ("question", question.scene_id, 6.0),
            ("reveal", reveal.scene_id, 4.0),
            ("continuation", reveal.scene_id, 4.0),
            ("hold", reveal.scene_id, 2.8),
        ),
    )
    bundle = build_render_bundle(
        scenes=scenes,
        timeline=timeline,
        profile=profile,
        export_preset="decision_clip",
        localizations={"video.analysis.expert_reason": "سبب مختصر"},
        identity={
            "gameSerial": "BLT-TIMELINE-1",
            "sceneId": question.scene_id,
            "enrollmentUri": "https://balootx.local/c/TIMELINE1",
            "players": {
                "north": "Player 01",
                "west": "Player 04",
                "east": "Player 02",
                "south": "Player 03",
            },
        },
        repository_root=root,
    )
    assert bundle.value["timeline"] == timeline.to_dict()
    assert bundle.value["timeline"]["segments"][0]["durationFrames"] == 120
    assert bundle.value["observerSeat"] == "south"


def test_render_bundle_rejects_mixed_private_perspectives(root: Path) -> None:
    profile, scenes = _inputs(root)
    question, reveal = scenes
    north_view = _observer_view()
    north_view["observer"] = "north"
    north_view["currentPlayer"] = "north"
    north_view["hands"] = {
        "north": list(north_view["hands"]["south"]),
        "east": {"count": 3},
        "south": {"count": 3},
        "west": {"count": 3},
    }
    north_reveal = reduce_scene_document(
        observer_view=north_view,
        public_events=(),
        stage="reveal",
        prompt_key="video.decision.best_move",
        acting_seat="north",
        prompt_contract="choose_acting_action",
        rendition_type="actor_challenge",
        legal_choices=(
            {"kind": "play_card", "card": {"id": 0, "rank": "7", "suit": "spades"}},
        ),
        analysis_facts={
            "recommendedAction": {
                "kind": "play_card",
                "card": {"id": 0, "rank": "7", "suit": "spades"},
            }
        },
        effective_config_sha256="a" * 64,
        locale="ar-SA",
        direction="rtl",
    )
    with pytest.raises(ValueError, match="exactly one isolated observer"):
        build_render_bundle(
            scenes=(question, north_reveal),
            timeline=build_export_timeline(
                scenes=(question, north_reveal),
                profile=profile,
                export_preset="decision_clip",
            ),
            profile=profile,
            export_preset="decision_clip",
            localizations={"video.analysis.expert_reason": "سبب مختصر"},
            identity={
                "gameSerial": "BLT-MIXED-1",
                "sceneId": question.scene_id,
                "enrollmentUri": "https://balootx.local/c/MIXED1",
                "players": {
                    "north": "Player 01",
                    "west": "Player 04",
                    "east": "Player 02",
                    "south": "Player 03",
                },
            },
            repository_root=root,
        )


def _bundle_for_observer(root: Path, observer: str, serial: str) -> RenderBundleV1:
    profile, scenes = _inputs(root, observer)
    return build_render_bundle(
        scenes=scenes,
        timeline=build_export_timeline(
            scenes=scenes, profile=profile, export_preset="decision_clip"
        ),
        profile=profile,
        export_preset="decision_clip",
        localizations={"video.analysis.expert_reason": "سبب مختصر"},
        identity={
            "gameSerial": serial,
            "sceneId": scenes[0].scene_id,
            "enrollmentUri": f"https://balootx.local/c/{serial}",
            "players": {
                "north": "Player 01",
                "west": "Player 04",
                "east": "Player 02",
                "south": "Player 03",
            },
        },
        repository_root=root,
    )


def test_batch_and_multi_angle_plans_keep_observer_bundles_isolated(root: Path) -> None:
    south = _bundle_for_observer(root, "south", "SOUTH1")
    north = _bundle_for_observer(root, "north", "NORTH1")
    bundles = {item.bundle_sha256: item for item in (south, north)}

    batch = RenderBatchV1.create(((south, "south-actor"), (north, "north-partner")))
    assert batch.verify(bundles)
    assert [item.observer_seat for item in batch.items] == ["south", "north"]

    plan = MultiAngleRenderPlanV1.create(
        (
            (south, "actor_challenge", 0, 300, "cut", 0),
            (north, "partner_inference", 300, 550, "fade", 12),
        )
    )
    assert plan.verify(bundles)
    assert plan.total_frames == 550
    encoded = json.dumps(plan.to_dict(), sort_keys=True)
    assert "separate_render_bundles" in encoded
    assert "observer-hand" not in encoded and "scenes" not in encoded
    assert {item.observer_seat for item in plan.segments} == {"south", "north"}

    with pytest.raises(ValueError, match="at least two observer perspectives"):
        MultiAngleRenderPlanV1.create(
            (
                (south, "actor_challenge", 0, 300, "cut", 0),
                (south, "seat_narrative", 300, 550, "fade", 8),
            )
        )


@pytest.mark.parametrize("export_preset", ["hooked_flagship", "decision_clip"])
def test_render_bundle_preserves_audio_enable_toggle(root: Path, export_preset: str) -> None:
    profile, scenes = _inputs(root)
    kwargs = {
        "scenes": scenes,
        "timeline": build_export_timeline(
            scenes=scenes, profile=profile, export_preset=export_preset
        ),
        "profile": profile,
        "export_preset": export_preset,
        "localizations": {"prompt": "قرار"},
        "identity": {
            "gameSerial": "BLT-TEST-0001",
            "sceneId": scenes[0].scene_id,
            "enrollmentUri": "balootx://game/BLT-TEST-0001",
            "players": {"north": "Player 01", "west": "Player 04", "east": "Player 02", "south": "Player 03"},
        },
        "repository_root": root,
    }
    profile["audio"]["enabled"] = True
    enabled = build_render_bundle(**kwargs)
    profile["audio"]["enabled"] = False
    muted = build_render_bundle(**kwargs)

    assert enabled.value["presentation"]["audio"]["enabled"] is True
    assert muted.value["presentation"]["audio"]["enabled"] is False
    assert enabled.bundle_sha256 != muted.bundle_sha256
    assert enabled.verify() and muted.verify()
    assert enabled.value["scenes"] == muted.value["scenes"]
    assert enabled.value["timeline"] == muted.value["timeline"]


def test_approved_v4_public_gate_rejects_old_premium_theme(root: Path) -> None:
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    profile["exports"]["decision_clip"]["include_complete_replay"] = True
    profile["motion"]["flight_steps"] = 15
    validate_approved_v4_public_profile(profile)
    profile["majlis"]["theme"] = "emerald"
    with pytest.raises(ValueError, match="red Majlis number box"):
        validate_approved_v4_public_profile(profile)


def test_approved_v4_historical_public_binding_requires_no_notice(root: Path) -> None:
    profile = yaml.safe_load((root / "configs/video/premium.yaml").read_text(encoding="utf-8"))
    profile["exports"]["decision_clip"]["include_complete_replay"] = True
    profile["motion"]["flight_steps"] = 15
    assert profile["approved_design"]["variant"] == "public_v4_no_justification_no_notice"
    validate_approved_v4_public_profile(profile)

    profile["majlis"]["research_notice"] = True
    with pytest.raises(ValueError, match="research notice must match"):
        validate_approved_v4_public_profile(profile)

    profile["approved_design"]["variant"] = "unapproved_notice"
    with pytest.raises(ValueError, match="binding mismatch: variant"):
        validate_approved_v4_public_profile(profile)
