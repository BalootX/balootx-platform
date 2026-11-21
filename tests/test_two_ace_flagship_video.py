from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import yaml

from baloot_platform.research.scenario_contracts import canonical_sha256
from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer


def _golden_module(root: Path):
    path = root / "scripts/render_research_scenario_v4_goldens.py"
    spec = importlib.util.spec_from_file_location("two_ace_flagship_golden", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _two_ace_bundle(root: Path, rules):
    module = _golden_module(root)
    name, fixture_path = module.GOLDENS[-1]
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    renderer = RemotionProfessionalRenderer(profile, repository_root=root)
    return module.build_golden_bundle(
        name=name,
        fixture_path=fixture_path,
        rules=rules,
        renderer=renderer,
        video_profile_hash=canonical_sha256(profile),
    )


def test_two_ace_fixture_replays_the_authoritative_project_declaration(
    root: Path, rules
) -> None:
    module = _golden_module(root)
    fixture = json.loads(module.GOLDENS[-1][1].read_text(encoding="utf-8"))
    assert fixture["exactTracePrefix"]["projectDeclarations"] == [
        {
            "owner": "north",
            "kind": "hundred_sequence",
            "value": 20,
            "highRank": 7,
            "declarationOrder": 1,
        }
    ]
    bundle = _two_ace_bundle(root, rules)
    assert bundle.verify()


def _layer(scene: dict, layer_id: str) -> dict:
    return next(layer["content"] for layer in scene["layers"] if layer["id"] == layer_id)


def test_two_ace_flagship_progresses_observer_hand_from_eight_to_zero(
    root: Path, rules
) -> None:
    bundle = _two_ace_bundle(root, rules)
    assert bundle.verify()
    context = next(
        scene for scene in bundle.value["scenes"] if scene["stage"] == "continuation"
    )
    public = _layer(context, "public-lifecycle")
    observer = _layer(context, "observer-lifecycle")
    observer_seat = context["observer"]
    remaining = {card["id"] for card in observer["initialHandCards"]}
    sizes = [len(remaining)]
    sizes_after_observer_plays = []
    sizes_after_tricks = []
    for play_index, play in enumerate(public["plays"], start=1):
        if play["seat"] == observer_seat:
            assert play["card"]["id"] in remaining
            remaining.remove(play["card"]["id"])
            sizes_after_observer_plays.append(len(remaining))
        sizes.append(len(remaining))
        if play_index % 4 == 0:
            sizes_after_tricks.append(len(remaining))

    pause = public["pauseBeforePlayIndex"]
    assert sizes[0] == 8
    assert sizes[pause] == 3
    assert sizes[pause + 1] == 2
    assert sizes[-1] == 0
    assert sizes_after_observer_plays == list(range(7, -1, -1))
    assert sizes_after_tricks == list(range(7, -1, -1))


def test_two_ace_flagship_marks_ca_as_the_only_winner_and_holds_final_play(
    root: Path, rules
) -> None:
    module = _golden_module(root)
    assert module.GOLDENS[-1][0] == "sun-hundred-ace-sacrifice-two-aces"
    bundle = _two_ace_bundle(root, rules)
    context = next(
        scene for scene in bundle.value["scenes"] if scene["stage"] == "continuation"
    )
    reveal = next(
        scene for scene in bundle.value["scenes"] if scene["stage"] == "reveal"
    )
    public = _layer(context, "public-lifecycle")
    analysis = _layer(reveal, "analysis")
    post_decision = public["plays"][public["pauseBeforePlayIndex"] :]
    continuation = next(
        segment
        for segment in bundle.value["timeline"]["segments"]
        if segment["stage"] == "continuation"
    )

    assert analysis["completionStatus"] == "exact_complete"
    assert analysis["utility"] == "round_win"
    assert analysis["selectedAction"] == "CA"
    assert analysis["winningRootActionCount"] == 1
    assert [item["action"] for item in analysis["rootActions"] if item["satisfies"]] == [
        "CA"
    ]
    assert [play["card"]["code"] for play in post_decision] == [
        "CA",
        "C8",
        "C7",
        "D7",
        "S7",
        "C9",
        "DA",
        "D8",
        "SA",
    ]
    assert continuation["durationFrames"] == 10 * bundle.value["render"]["fps"]
    assert continuation["durationFrames"] / len(post_decision) >= bundle.value[
        "render"
    ]["fps"]

    source = (
        root / "src/baloot_platform/video/remotion/src/professional-prototype.tsx"
    ).read_text(encoding="utf-8")
    assert "canonicalCardCode(card) === selectedActionCode" in source
    assert 'data-selected-action={selectedActionCode || undefined}' in source
    assert 'className="winning-move-code"' in source
    assert '"video.analysis.only_winning_move"' in source
    assert 'analysis.utility === "round_win" || analysis.utility === "contract_success"' in source
    assert "cue.transcriptKey === analysisReasonKey" not in source
    assert "bundle.localizations[cue.transcriptKey] === spokenReason" not in source

    render_script = (
        root / "src/baloot_platform/video/remotion/scripts/render-flagship.mjs"
    ).read_text(encoding="utf-8")
    assert '"06a-sacrifice"' in render_script
    assert '"06b-final-trick"' in render_script


def test_premium_flagship_timing_is_packaged_without_drift(root: Path) -> None:
    source = root / "configs/video/premium.yaml"
    packaged = root / "src/baloot_platform/resources/configs/video/premium.yaml"
    assert packaged.read_bytes() == source.read_bytes()
    profile = yaml.safe_load(source.read_text(encoding="utf-8"))
    flagship = profile["exports"]["hooked_flagship"]
    assert flagship["continuation_seconds"] == 10.0
    assert flagship["reveal_end_seconds"] == 68.0
    assert (
        flagship["play_end_seconds"]
        + flagship["decision_seconds"]
        + flagship["reveal_seconds"]
        + flagship["continuation_seconds"]
        == flagship["reveal_end_seconds"]
    )


def test_rejected_v4_result_prepares_as_badged_non_qr_audit_media(
    root: Path, rules
) -> None:
    module = _golden_module(root)
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    renderer = RemotionProfessionalRenderer(profile, repository_root=root)
    name, fixture_path = module.AUDIT_GOLDEN
    bundle, receipt = module.build_audit_bundle(
        name=name,
        fixture_path=fixture_path,
        rules=rules,
        renderer=renderer,
        video_profile_hash=canonical_sha256(profile),
    )
    source = bundle.value["source"]
    reveal = next(
        scene for scene in bundle.value["scenes"] if scene["stage"] == "reveal"
    )
    analysis = _layer(reveal, "analysis")

    assert bundle.verify()
    assert receipt["sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    assert receipt["bundleSha256"] == bundle.bundle_sha256
    assert receipt["proofOutcome"]["completionStatus"] == "proof_failed"
    assert receipt["proofOutcome"]["claimSatisfied"] is False
    assert receipt["proofOutcome"]["resources"]["exhaustionReason"] is None
    assert analysis["reasonKey"] == "video.analysis.proof_failed"
    assert analysis["decisionReference"] == source["decisionReference"]
    assert analysis["publicProjectionHash"] == source["publicProjectionHash"]
    assert source["sourceSchemaVersion"] == 3
    for private_field in (
        "decisionId",
        "exactStateId",
        "traceHash",
        "resultHash",
        "scenarioHash",
        "proofHash",
        "constraintEvidenceHash",
        "exactStateHash",
    ):
        assert private_field not in source
    assert source["qualificationStatus"] == "rejected"
    assert source["renderPurpose"] == "research_audit"
    assert source["auditOnly"] is True
    assert source["publicationEligible"] is False
    assert source["qrActionable"] is False
    assert source["decisionReference"] == {
        "resultCode": source["code"],
        "decisionOrdinal": 1,
    }
    assert receipt["exactStateId"].startswith("STATE-")
    encoded_bundle = json.dumps(
        bundle.to_dict(), sort_keys=True, separators=(",", ":")
    )
    for private_identity in (
        receipt["exactStateId"],
        receipt["resultHash"],
        receipt["proofOutcome"]["sha256"],
        receipt["proofOutcome"].get("semanticSha256"),
    ):
        if private_identity:
            assert private_identity not in encoded_bundle
    assert set(source["publicationBlockers"]) == {
        "result_not_qualified",
        "qualification_failures_present",
        "scenario_claim_not_satisfied",
    }

    renderer_source = (
        root / "src/baloot_platform/video/remotion/src/professional-prototype.tsx"
    ).read_text(encoding="utf-8")
    assert 'localized("video.governance.research_audit_badge")' in renderer_source
    assert "show_identity_pill) || auditMedia" in renderer_source
    assert "&& !auditMedia && qrActionable" in renderer_source


def test_rejected_v4_audit_media_cli_writes_immutable_bundle_receipt(
    root: Path, tmp_path: Path
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(root / "scripts/render_research_scenario_v4_goldens.py"),
            "--audit-only",
            "--bundle-only",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "research_scenario_v4_rejected_audit_media"
    assert manifest["bundleOnly"] is True
    assert len(manifest["entries"]) == 1
    entry = manifest["entries"][0]
    assert entry["auditOnly"] is True
    assert entry["publicationEligible"] is False
    assert entry["renderPurpose"] == "research_audit"
    bundle = json.loads((tmp_path / entry["bundle"]).read_text(encoding="utf-8"))
    preparation = json.loads(
        (tmp_path / entry["preparation"]).read_text(encoding="utf-8")
    )
    assert preparation["bundleSha256"] == bundle["bundleSha256"]
    assert preparation["exactStateId"].startswith("STATE-")
    assert "exactStateId" not in bundle["source"]
    assert entry["decisionReference"] == bundle["source"]["decisionReference"]
    assert entry["publicProjectionHash"] == bundle["source"][
        "publicProjectionHash"
    ]
    public_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    assert preparation["exactStateId"] not in public_manifest
    assert preparation["resultHash"] not in public_manifest
    assert preparation["proofOutcome"]["sha256"] not in public_manifest
