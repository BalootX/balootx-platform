"""Render the research-notice publish candidate (v4) from the governed synthetic golden.

Structure (owner spec v4): pause on the decision with the question, replay the
recorded round from the bidding (the dealer's أول/ثاني round calls) up to that
decision, pause again, reveal only the winning card, play the remaining tricks
and hold on the round's final score from the engine. No justification text.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import shutil

import yaml
from PIL import Image, ImageDraw, ImageFont

from baloot_platform.core.rules import RuleBook
from baloot_platform.research.exact_state import ExactStateNormalizerV1
from baloot_platform.research.scenario_contracts import canonical_sha256
from baloot_platform.video.remotion_renderer import (
    RemotionProfessionalRenderer,
    _json_write,
)
from baloot_platform.video.render_bundle import (
    seeded_player_names,
    validate_decision_clip_admission,
    validate_viewer_buyer_roles,
    validate_approved_v4_design_binding,
    validate_approved_v4_public_profile,
)
from baloot_platform.video.typography import FONT_BODY_BOLD
from baloot_platform.video.visual_qa import assert_text_supported, text_bbox
from scripts.render_release import render_timeout_seconds
from scripts.render_research_scenario_v4_goldens import build_golden_bundle


ROOT = Path(__file__).resolve().parents[1]
DESIGN_INPUTS = (
    "configs/video/premium.yaml",
    "src/baloot_platform/video/remotion/src/professional-prototype.tsx",
    "src/baloot_platform/video/remotion/src/style.css",
)
DESIGN_BASELINE = "release/approved-v4-design-inputs.json"
GATE_RESULT = "artifacts/balootx-v4-gate.json"
FIXTURE = ROOT / "examples/research-scenario-v4-ace-sacrifice-two-aces.json"
GOLDEN = "sun-hundred-ace-sacrifice-two-aces"
# Seven is permanently barred after posting; retain its fixture only for history.
# Selection changes the actual proof/render input, never only the name draw.
DEAL_SOURCES = (
    (
        "sun-hundred-project-ace-sacrifice-two-aces-v1",
        "examples/research-scenario-v4-ace-sacrifice-two-aces.json",
        "sun-hundred-ace-sacrifice-two-aces",
    ),
    (
        "sun-single-winner-20261116-01-v1",
        "examples/research-scenario-v4-single-winner-20261116-01.json",
        "sun-single-winner-20261116-01",
    ),
    (
        "sun-single-winner-20261116-02-v1",
        "examples/research-scenario-v4-single-winner-20261116-02.json",
        "sun-single-winner-20261116-02",
    ),
    (
        "sun-single-winner-20261116-03-v1",
        "examples/research-scenario-v4-single-winner-20261116-03.json",
        "sun-single-winner-20261116-03",
    ),
    (
        "sun-single-winner-20261116-04-v1",
        "examples/research-scenario-v4-single-winner-20261116-04.json",
        "sun-single-winner-20261116-04",
    ),
    (
        "sun-single-winner-20261116-05-v1",
        "examples/research-scenario-v4-single-winner-20261116-05.json",
        "sun-single-winner-20261116-05",
    ),
    (
        "sun-single-winner-20261116-06-v1",
        "examples/research-scenario-v4-single-winner-20261116-06.json",
        "sun-single-winner-20261116-06",
    ),
)
OUTPUT = ROOT / "outputs/research-notice-4"
BASENAME = "balootx-research-notice-4"
CAPTION = (
    "ثلاث أوراق، ولعبة واحدة فقط تفوز في هالتوزيعة… وش تلعب؟ "
    "توزيع افتراضي من BalootX #بلوت #تحدي_بلوت #BalootX"
)
# Seconds per segment. Every play (replay and remaining tricks) takes play_seconds,
# 25% shorter than v3's 1.2 s; the final-score outcome takes the rest of TARGET_SECONDS.
TIMELINE = {
    "opening_hook_seconds": 4.0,
    "rewind_seconds": 0.4,
    "setup_seconds": 3.8,
    "play_seconds": 0.9,
    "decision_seconds": 2.5,
    "reveal_seconds": 2.0,
    "continue_to_round_end": True,
}
# Card flight frames: 15 of the 27 frames per play, leaving room for a visible
# trick collection (the 21-step default would squeeze collection to one frame).
FLIGHT_STEPS = 15
TARGET_SECONDS = 44
STAGES = ("hook", "rewind", "setup", "replay", "question", "reveal", "continuation", "outcome")
PHASES = ("hook", "setup", "replay", "question", "reveal", "continuation", "outcome")
STYLE = ROOT / "src/baloot_platform/video/remotion/src/style.css"
SOURCE_SHA = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:  # streamed, never read whole
        return hashlib.file_digest(handle, "sha256").hexdigest()


@contextmanager
def _gate_check(gate: dict, path: Path, name: str):
    gate["checks"][name] = {"status": "running"}
    _json_write(path, gate)
    try:
        yield
    except (Exception, SystemExit) as exc:
        gate["checks"][name] = {"status": "failed", "error": str(exc), "errorType": type(exc).__name__}
        gate["status"] = "failed"
        try:
            _json_write(path, gate)
        except OSError as write_error:
            exc.add_note(f"gate result could not be persisted: {write_error}")
        raise
    else:
        gate["checks"][name] = {"status": "passed"}
        _json_write(path, gate)


def validate_design_input_integrity(profile: dict, evidence: dict) -> None:
    """Compare measured bytes to a retained approval baseline; never create one."""
    evidence["approvedDesignBindingStatus"] = "not_validated"
    measured = evidence["measuredDesignInputs"] = {}
    for name in DESIGN_INPUTS:
        measured[name] = _sha256(ROOT / name)
    validate_approved_v4_design_binding(profile)
    evidence["approvedDesignBinding"] = dict(profile["approved_design"])
    try:
        raw = (ROOT / DESIGN_BASELINE).read_bytes()
        evidence["designBaseline"] = {"path": DESIGN_BASELINE, "sha256": hashlib.sha256(raw).hexdigest()}
        baseline = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"approved design baseline unavailable or invalid: {DESIGN_BASELINE}") from exc
    if (
        not isinstance(baseline, dict)
        or type(baseline.get("schemaVersion")) is not int
        or baseline.get("schemaVersion") != 1
        or baseline.get("status") != "owner_approved_design_baseline"
        or baseline.get("approvedDesignBinding") != profile["approved_design"]
    ):
        raise ValueError("approved design baseline binding is invalid")
    approved = baseline.get("approvedDesignInputs")
    if (
        not isinstance(approved, dict)
        or approved.keys() != measured.keys()
        or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in approved.values())
    ):
        raise ValueError("approved design baseline input hashes are invalid")
    evidence["approvedDesignInputs"] = approved
    changed = [name for name in measured if measured[name] != approved[name]]
    if changed:
        raise ValueError(f"approved Majlis design inputs changed: {', '.join(changed)}")
    evidence["approvedDesignBindingStatus"] = "matched_retained_baseline"


def _css_px(css: str, selector: str, prop: str, index: int = 0) -> int:
    rule = re.search(rf"(?:^|\n){re.escape(selector)} \{{([^}}]*)\}}", css)
    value = rule and re.search(rf"(?:^|;|\s){prop}:\s*([^;]+);", rule.group(1))
    if not value:
        raise ValueError(f"notice layout rule is missing: {selector} {prop}")
    return int(re.findall(r"(\d+)px", value.group(1))[index])


def notice_sentence_fit(text: str) -> dict[str, int]:
    """Measure the notice sentence (shaped Arabic) against the band's text column."""
    css = STYLE.read_text(encoding="utf-8")
    sentence = ".rhead-text .notice-sentence"
    if not re.search(rf"{re.escape(sentence)} \{{[^}}]*white-space:\s*nowrap", css):
        raise ValueError("notice sentence must stay on one line")
    available = (
        _css_px(css, ".rhead", "width")
        - 2 * _css_px(css, ".rhead", "padding", -1)  # horizontal padding is the last value
        - _css_px(css, ".rhead-icon", "width")
        - _css_px(css, ".rhead", "gap")
    )
    size = _css_px(css, sentence, "font-size")
    font = ImageFont.truetype(FONT_BODY_BOLD, size)
    assert_text_supported(font, text, label="research notice sentence")
    left, _, right, _ = text_bbox(
        ImageDraw.Draw(Image.new("L", (1, 1))), text, font, direction="rtl", language="ar",
    )
    return {"widthPx": right - left, "availablePx": available, "fontPx": size}


def setup_beats(lifecycle: dict) -> list[dict]:
    """The bidding beats in order (mirrors setupActions in professional-prototype.tsx)."""
    beats: list[dict] = []
    previous = None
    for bid in lifecycle["bidHistory"]:
        round_number = int(bid.get("round", 1))
        if round_number != previous:
            beats.append({"kind": "round", "seat": lifecycle["dealer"], "round": round_number})
            previous = round_number
        beats.append({"kind": "bid", "seat": bid["seat"], "action": bid["action"], "round": round_number})
    beats.append({"kind": "purchase", "seat": lifecycle["purchaseRecipient"]})
    beats.extend({"kind": "escalation", "seat": row["seat"], "action": row["action"]}
                 for row in lifecycle.get("escalationHistory") or ())
    return beats


def engine_round_result(rules: RuleBook, fixture: Path | None = None) -> dict:
    """The round's final score straight from the engine: recorded prefix plus the proof line."""
    fixture = json.loads((fixture or FIXTURE).read_text(encoding="utf-8"))
    prefix = dict(fixture["exactTracePrefix"])
    plays = list(prefix["playActions"])
    normalizer = ExactStateNormalizerV1(rules)
    for card in fixture["expected"]["principalVariation"]:
        actor = normalizer.normalize({**prefix, "playActions": plays, "stateAssertions": {}}).next_actor
        plays.append({"actor": actor, "card": card})
    state = normalizer.normalize({**prefix, "playActions": plays, "stateAssertions": {"phase": "scored"}}).state
    return {"matchScore": list(state["matchScore"]), "roundScore": dict(state["roundScore"])}


def _lifecycle(value: dict, stage: str = "continuation") -> dict:
    scene = next(scene for scene in value["scenes"] if scene["stage"] == stage)
    return next(layer["content"] for layer in scene["layers"] if layer["id"] == "public-lifecycle")


def timeline_outline(value: dict) -> list[dict]:
    """Ordered segments in seconds, with the bidding split into its equal beats."""
    fps = value["render"]["fps"]
    labels = {"hook": "opener_question", "replay": "plays_to_decision", "question": "pause_question",
              "reveal": "reveal_winning_card", "continuation": "remaining_tricks", "outcome": "final_score"}
    rows = []
    for segment in value["timeline"]["segments"]:
        start, frames = segment["startFrame"], segment["durationFrames"]
        if segment["stage"] != "setup":
            rows.append({"segment": labels.get(segment["stage"], segment["stage"]),
                         "startSeconds": start / fps, "seconds": frames / fps})
            continue
        beats = setup_beats(_lifecycle(value))
        for index, beat in enumerate(beats):
            call = value["localizations"].get(f"video.bid.round.{beat.get('round')}")
            name = (f"dealer_call_{call}" if beat["kind"] == "round"
                    else f"buying_round_{beat['round']}_{beat['seat']}_{beat['action']}" if beat["kind"] == "bid"
                    else f"{beat['kind']}_{beat['seat']}")
            beat_start = start + frames * index / len(beats)
            rows.append({"segment": name, "startSeconds": beat_start / fps,
                         "seconds": frames / len(beats) / fps})
    return rows


def notice_profile(*, public: bool = False, design_evidence: dict | None = None) -> dict:
    profile = yaml.safe_load(
        (ROOT / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    validate_design_input_integrity(profile, design_evidence if design_evidence is not None else {})
    profile["render"]["concurrency"] = 2
    profile["exports"]["decision_clip"].update({
        **TIMELINE,
        "include_complete_replay": True,
        "minimum_seconds": TARGET_SECONDS,
        "target_seconds": TARGET_SECONDS,
        "maximum_seconds": TARGET_SECONDS,
    })
    profile["motion"]["flight_steps"] = FLIGHT_STEPS
    profile["majlis"]["research_notice"] = not public
    profile["presentation"]["show_justification"] = False
    if public:
        validate_approved_v4_public_profile(profile)
    return profile


def complete_trace(
    rules: RuleBook, plays: list[dict], fixture: Path | None = None
) -> dict:
    """The recorded prefix plus the proof continuation, normalized to a scored round."""
    fixture = json.loads((fixture or FIXTURE).read_text(encoding="utf-8"))
    prefix = dict(fixture["exactTracePrefix"])
    pause = len(prefix["playActions"])
    if (
        len(plays) != 32
        or [(play["seat"], play["card"]["code"]) for play in plays[:pause]]
        != [(row["actor"], row["card"]) for row in prefix["playActions"]]
        or [play["card"]["code"] for play in plays[pause:]]
        != fixture["expected"]["principalVariation"]
    ):
        raise ValueError("notice continuation must equal the recorded prefix and proof line")
    prefix["playActions"] = [{"actor": play["seat"], "card": play["card"]["code"]} for play in plays]
    prefix["stateAssertions"] = {"phase": "scored"}
    return {
        "completeExactTracePrefix": prefix,
        "completeExactNormalization": ExactStateNormalizerV1(rules).normalize(prefix).to_dict(),
    }


def build_notice_bundle(
    rules: RuleBook,
    profile: dict | None = None,
    *,
    fixture: Path | None = None,
    golden: str | None = None,
):
    fixture = fixture or FIXTURE
    golden = golden or GOLDEN
    validate_notice_fixture_admission(fixture, rules)
    profile = profile or notice_profile()
    # First pass (plain decision clip) yields the proof continuation; the second
    # carries the complete deal so the bidding can be replayed with the real hand.
    base = yaml.safe_load((ROOT / "configs/video/premium.yaml").read_text(encoding="utf-8"))
    plain = build_golden_bundle(
        name=golden, fixture_path=fixture, rules=rules,
        renderer=RemotionProfessionalRenderer(base, repository_root=ROOT),
        video_profile_hash=canonical_sha256(base), export_preset="decision_clip",
    )
    continuation = next(scene for scene in plain.value["scenes"] if scene["stage"] == "continuation")
    plays = next(layer["content"] for layer in continuation["layers"] if layer["id"] == "public-lifecycle")["plays"]
    return build_golden_bundle(
        name=golden, fixture_path=fixture, rules=rules,
        renderer=RemotionProfessionalRenderer(profile, repository_root=ROOT),
        video_profile_hash=canonical_sha256(profile), export_preset="decision_clip",
        trusted_complete_trace=complete_trace(rules, plays, fixture),
    )


def validate_notice_bundle(
    bundle,
    engine_result: dict,
    *,
    public: bool = False,
    fixture: Path | None = None,
) -> None:
    value = bundle.value
    validate_decision_clip_admission(value)
    expected = json.loads((fixture or FIXTURE).read_text(encoding="utf-8"))["expected"]
    source = value["source"]
    reveal = next(scene for scene in value["scenes"] if scene["stage"] == "reveal")
    analysis = next(layer["content"] for layer in reveal["layers"] if layer["id"] == "analysis")
    fps = value["render"]["fps"]
    segments = {row["stage"]: row for row in value["timeline"]["segments"]}
    seconds = {stage: row["durationFrames"] / fps for stage, row in segments.items()}
    lifecycle = _lifecycle(value)
    shown_score = ("cardGamePoints", "projectPoints", "balootPoints", "awardedPoints")
    fit = notice_sentence_fit(value["localizations"]["video.notice.sentence"])
    if not (
        bundle.verify()
        and source["qualificationStatus"] == "qualified"
        and source["publicationEligible"] is True
        and source["auditOnly"] is False
        and not source["publicationBlockers"]
        and not source["qualificationFailures"]
        and analysis["completionStatus"] == "exact_complete"
        and analysis["achievedMode"] == "exact"
        and analysis["claimSatisfied"] is True
        and analysis["achievedVisibility"] == "complete_state"
        and analysis["proofScopeKey"] == "video.analysis.proof_scope.complete_state"
        and analysis["selectedAction"] == expected["selectedAction"]
        and analysis["winningRootActionCount"] == expected["winningRootActionCount"]
        and {row["action"] for row in analysis["rootActions"] if row["satisfies"]} == {expected["selectedAction"]}
        and value["localizations"]["video.decision.choose_best_move"] == "وش تلعب؟"
        and (value["render"]["width"], value["render"]["height"], fps) == (1080, 1920, 30)
        and value["timeline"]["totalFrames"] == TARGET_SECONDS * fps
        and tuple(segments) == STAGES
        and segments["hook"]["durationFrames"] == round(TIMELINE["opening_hook_seconds"] * fps)
        and 3 <= seconds["hook"] <= 5
        and 2 <= seconds["question"] <= 3
        and 2 <= seconds["outcome"] <= 3
        and len(lifecycle["plays"]) == 32
        # The final score is the engine's own round result, not the bundle's say-so.
        and list(lifecycle["matchScore"]) == engine_result["matchScore"]
        and all(lifecycle["roundScore"][key] == engine_result["roundScore"][key] for key in shown_score)
        and value["presentation"]["presentation"]["show_justification"] is False
        and value["presentation"]["presentation"]["show_justification_text"] is False
        and not any(cue.get("assetId") == "voice-explanation" for cue in value.get("audioCues") or ())
        and value["presentation"]["majlis"]["research_notice"] is (not public)
        and (not public or value["presentation"]["majlis"]["id_box"] == "number")
        and fit["widthPx"] <= fit["availablePx"]
        and len(set(value["identity"]["players"].values())) == 4
    ):
        raise ValueError("research notice requires the qualified, exact synthetic critical decision")


def validate_notice_fixture_admission(fixture: Path, rules: RuleBook | None = None) -> dict:
    rules = rules if rules is not None else RuleBook.load(ROOT / "configs/rules/saudi_common_v1.yaml")
    source = json.loads(fixture.read_text(encoding="utf-8"))
    normalized = ExactStateNormalizerV1(rules).normalize(source["exactTracePrefix"])
    viewer, buyer = normalized.next_actor, normalized.state.get("buyer")
    validate_viewer_buyer_roles(viewer, buyer)
    return {"status": "passed", "viewer": viewer, "buyer": buyer, "exactStateId": normalized.exact_state_id}


def select_deal(
    deal_id: str | None = None, seed: str | int | None = None
) -> dict[str, object]:
    """Resolve one existing governed fixture by ID or a stable catalog seed."""
    if deal_id is not None and seed is not None:
        raise ValueError("deal-id and seed are mutually exclusive")
    rows = {
        identifier: (identifier, relative, golden)
        for identifier, relative, golden in DEAL_SOURCES
    }
    if deal_id is not None:
        requested = str(deal_id).strip()
        row = rows.get(requested)
        if row is None:
            row = next(
                (
                    candidate
                    for candidate in DEAL_SOURCES
                    if requested in {Path(candidate[1]).stem, candidate[2]}
                ),
                None,
            )
        if row is None:
            raise ValueError(f"unknown governed deal identifier: {deal_id!r}")
    elif seed is None:
        try:
            relative = FIXTURE.relative_to(ROOT).as_posix()
        except ValueError:
            relative = str(FIXTURE)
        row = (DEAL_SOURCES[0][0], relative, GOLDEN)
    else:
        token = str(seed)
        if not token:
            raise ValueError("seed must not be empty")
        index = (
            int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            % len(DEAL_SOURCES)
        )
        row = DEAL_SOURCES[index]
    identifier, relative, golden = row
    fixture = FIXTURE if deal_id is None and seed is None else ROOT / relative
    if not fixture.is_file():
        raise FileNotFoundError(f"governed deal fixture is missing: {relative}")
    if deal_id is not None or seed is not None:
        try:
            source = json.loads(fixture.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"governed deal fixture is unreadable: {relative}") from exc
        if not isinstance(source, dict) or source.get("scenarioId") != identifier:
            raise ValueError(f"governed deal identifier mismatch: {relative}")
        expected = source.get("expected")
        if (
            not isinstance(expected, dict)
            or not expected.get("selectedAction")
            or expected.get("winningRootActionCount") != 1
        ):
            raise ValueError(
                f"governed deal does not satisfy the single-winning-move gate: {relative}"
            )
        prefix = source.get("exactTracePrefix")
        principal_variation = expected.get("principalVariation")
        prefix_plays = prefix.get("playActions") if isinstance(prefix, dict) else None
        if not (
            isinstance(prefix_plays, list)
            and all(
                isinstance(play, dict)
                and isinstance(play.get("actor"), str)
                and isinstance(play.get("card"), str)
                for play in prefix_plays
            )
            and isinstance(principal_variation, list)
            and principal_variation
            and all(isinstance(card, str) and card for card in principal_variation)
            and principal_variation[0] == expected["selectedAction"]
            and len(prefix_plays) + len(principal_variation) == 32
        ):
            raise ValueError(f"governed deal fixture is not buildable: {relative}")
    return {
        "identifier": identifier,
        "fixture": fixture,
        "source": relative,
        "golden": golden,
        "requestedSeed": None if seed is None else str(seed),
        "renderAdmission": validate_notice_fixture_admission(fixture),
    }


def _record_deal_manifest(
    manifest_path: Path, selection: dict[str, object], bundle, profile: dict
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = dict(manifest.get("identity") or bundle.value["identity"])
    name_seed = str(identity.get("gameSerial") or selection["identifier"])
    players = dict(identity["players"])
    if profile.get("identity") is not None and players != seeded_player_names(
        profile["identity"], name_seed
    ):
        raise ValueError("render manifest name draw is not deterministic")
    manifest["dealIdentifier"] = selection["identifier"]
    manifest["dealSource"] = selection["source"]
    manifest["dealSeed"] = selection["requestedSeed"]
    manifest["nameDraw"] = {"seed": name_seed, "players": players}
    manifest.pop("manifestSha256", None)
    manifest["manifestSha256"] = hashlib.sha256(
        json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    _json_write(manifest_path, manifest)
    return manifest


def _source_identity(
    source_head: str | None = None, source_tree: str | None = None
) -> tuple[str, str]:
    """Read the pinned source identity from the archive's retained manifest."""
    if source_head is not None or source_tree is not None:
        if not all(
            isinstance(value, str) and SOURCE_SHA.fullmatch(value)
            for value in (source_head, source_tree)
        ):
            raise ValueError("explicit source identity is invalid")
        return source_head, source_tree
    if (ROOT / ".git").exists():
        raise ValueError("explicit source identity is required for a git checkout")
    try:
        manifest = json.loads(
            (ROOT / "release/source-manifest.json").read_text(encoding="utf-8")
        )
        source = manifest["source"]
        revision = source["revision"]
        tree = source["tree"]
    except (KeyError, TypeError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("release source manifest identity is invalid") from exc
    if not all(
        isinstance(value, str) and SOURCE_SHA.fullmatch(value)
        for value in (revision, tree)
    ):
        raise ValueError("release source manifest identity is invalid")
    return revision, tree


def _write_provenance_notice(
    output: Path,
    *,
    source_head: str | None = None,
    source_tree: str | None = None,
    error: str | None = None,
) -> dict[str, object]:
    if error is not None:
        notice = {"schemaVersion": 1, "status": "failed", "error": error}
    else:
        try:
            revision, tree = _source_identity(source_head, source_tree)
        except ValueError as exc:
            notice = {"schemaVersion": 1, "status": "failed", "error": str(exc)}
        else:
            notice = {
                "schemaVersion": 1,
                "status": "passed",
                "sourceHead": revision,
                "sourceTree": tree,
            }
    notice.update({
        "scope": "source identity declaration validation only",
        "checks": {"source_identity_declaration": {"status": notice["status"]}},
        "checkDescription": "Require sourceHead and sourceTree as lowercase 40- or 64-digit hexadecimal identifiers, supplied explicitly or read from release/source-manifest.json.",
        "sourceTreeContentsVerified": False,
    })
    _json_write(output / "provenance-notice.json", notice)
    return notice


def main(
    copy_to: Path | None = None,
    *,
    public: bool = False,
    output_dir: Path | None = None,
    source_head: str | None = None,
    source_tree: str | None = None,
    deal_id: str | None = None,
    seed: str | int | None = None,
) -> None:
    if deal_id is not None and seed is not None:
        raise ValueError("deal-id and seed are mutually exclusive")
    if not public and (deal_id is not None or seed is not None):
        raise ValueError("deal-id and seed require --public")
    output = output_dir.resolve() if output_dir is not None else ROOT / ("outputs/approved-v4-public" if public else "outputs/research-notice-4")
    if any((output / name).exists() for name in ("owner-approval.json", "SHA256SUMS.txt")):
        raise ValueError(f"output contains an existing packet receipt or checksum seal; use a fresh output directory: {output}")
    selection = select_deal(deal_id, seed)
    fixture = selection["fixture"]
    golden = selection["golden"]
    gate_path = output / GATE_RESULT
    gate = {
        "schemaVersion": 1,
        "kind": "render_validation_only",
        "status": "running",
        "postingAuthorized": False,
        "ownerReleaseRequired": "explicit owner release of this exact MP4 SHA256",
        "approvedDesignBindingStatus": "not_validated",
        "public": public,
        "dealIdentifier": selection["identifier"],
        "dealSeed": selection["requestedSeed"],
        "fixture": selection["source"],
        "fixtureSha256": _sha256(fixture),
        "renderAdmission": selection["renderAdmission"],
        "checks": {},
    }
    source_identity_error = None
    try:
        source_head, source_tree = _source_identity(source_head, source_tree)
    except ValueError as exc:
        source_identity_error = str(exc)
        source_head = source_tree = None
    gate.update({"sourceHead": source_head, "sourceTree": source_tree})
    with _gate_check(gate, gate_path, "notice_profile"):
        profile = notice_profile(public=public, design_evidence=gate)
    gate["checks"]["validate_design_input_integrity"] = {"status": "passed"}
    if public:
        gate["checks"]["validate_approved_v4_public_profile"] = {"status": "passed"}
    with _gate_check(gate, gate_path, "build_notice_bundle"):
        rules = RuleBook.load(ROOT / "configs/rules/saudi_common_v1.yaml")
        renderer = RemotionProfessionalRenderer(
            profile, repository_root=ROOT, timeout_seconds=render_timeout_seconds(),
        )
        bundle = build_notice_bundle(
            rules, profile, fixture=fixture, golden=golden,
        )
    with _gate_check(gate, gate_path, "engine_round_result"):
        engine_result = engine_round_result(rules, fixture)
    with _gate_check(gate, gate_path, "validate_notice_bundle"):
        validate_notice_bundle(bundle, engine_result, public=public, fixture=fixture)
    basename = "balootx-approved-v4-public" if public else BASENAME
    with _gate_check(gate, gate_path, "RemotionProfessionalRenderer.execute"):
        artifacts = renderer.execute(
            bundle, export_preset="decision_clip", output_dir=output,
            basename=basename, playback_evidence=True,
            source_identity={"sourceHead": source_head, "sourceTree": source_tree},
        )
    with _gate_check(gate, gate_path, "render_input_integrity"):
        if (
            {name: _sha256(ROOT / name) for name in DESIGN_INPUTS} != gate["measuredDesignInputs"]
            or _sha256(fixture) != gate["fixtureSha256"]
            or _sha256(ROOT / DESIGN_BASELINE) != gate["designBaseline"]["sha256"]
        ):
            raise ValueError("render inputs changed during rendering")
    gate.update({"mp4": artifacts.video.name, "mp4Sha256": _sha256(artifacts.video),
                 "bundleSha256": bundle.bundle_sha256})
    with _gate_check(gate, gate_path, "validate_rendered_layout_geometry"):
        manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
        layout = manifest.get("layoutGeometry")
        if not isinstance(layout, dict) or layout.get("status") != "passed" or layout.get("mp4Sha256") != gate["mp4Sha256"]:
            raise ValueError("measured layout geometry gate is missing or unbound to MP4")
        if any(layout.get("binding", {}).get(key) != gate[key] for key in ("sourceHead", "sourceTree")):
            raise ValueError("layout geometry source binding mismatch")
        layout_path = output / layout["path"]
        if _sha256(layout_path) != layout["sha256"]:
            raise ValueError("layout geometry result hash mismatch")
        gate["layoutGeometry"] = layout
    with _gate_check(gate, gate_path, "render_receipt_content"):
        manifest = _record_deal_manifest(artifacts.manifest, selection, bundle, profile)
        rows = {row["checkpointPhase"]: row for row in manifest["playbackEvidence"]}
        segments = {row["stage"]: row for row in bundle.value["timeline"]["segments"]}
        # Thumbnail: the validated renderStill of the first, paused decision frame.
        thumbnail = output / f"{basename}-thumbnail.png"
        shutil.copyfile(output / manifest["files"]["playback:hook"]["name"], thumbnail)
        with Image.open(thumbnail) as image:
            if image.size != (1080, 1920) or not rows["hook"]["frame"] < segments["setup"]["startFrame"]:
                raise ValueError("thumbnail must be the portrait paused decision frame")
        stills = []
        for index, phase in enumerate(PHASES, start=1):
            still = output / f"{basename}-qa-{index:02d}-{phase}.png"
            shutil.copyfile(output / manifest["files"][f"playback:{phase}"]["name"], still)
            stills.append(still)
        fps = bundle.value["render"]["fps"]
        receipt = {
            "schemaVersion": 1,
            "status": "awaiting_owner_approval",
            "postingAuthorized": False,
            "sourceHead": source_head,
            "sourceTree": source_tree,
            "renderAdmission": gate["renderAdmission"],
            "layoutGeometry": layout,
            "approvedDesignBinding": gate["approvedDesignBinding"],
            "approvedDesignBindingStatus": gate["approvedDesignBindingStatus"],
            "approvedDesignInputs": gate["approvedDesignInputs"],
            "measuredDesignInputs": gate["measuredDesignInputs"],
            "designBaseline": gate["designBaseline"],
            "ownerReleaseRequired": gate["ownerReleaseRequired"],
            "fixture": selection["source"],
            "fixtureSha256": _sha256(fixture),
            "dealIdentifier": selection["identifier"],
            "dealSeed": selection["requestedSeed"],
            "nameDraw": manifest["nameDraw"],
            "synthetic": True,
            "language": "ar-SA",
            "caption": CAPTION,
            "headerText": None if public else {
                "title": bundle.value["localizations"]["video.notice.title"],
                "sentence": bundle.value["localizations"]["video.notice.sentence"],
                "fit": notice_sentence_fit(bundle.value["localizations"]["video.notice.sentence"]),
                "note": "Design band styling; publish-safe text instead of the audit-only wording.",
            },
            "justificationShown": False,
            "publicResearchNoticeShown": not public,
            "players": bundle.value["identity"]["players"],
            "timelineSeconds": {
                stage: segments[stage]["durationFrames"] / fps for stage in STAGES
            },
            "timelineOutline": timeline_outline(bundle.value),
            "playSeconds": TIMELINE["play_seconds"],
            "playsToRoundEnd": True,
            "finalScore": {
                "matchScore": engine_result["matchScore"],
                **{key: engine_result["roundScore"][key]
                   for key in ("cardGamePoints", "projectPoints", "balootPoints", "awardedPoints", "winner")},
            },
            "proofScope": "complete_state; this fixture only, not an actor-information-set guarantee",
            "bundleSha256": bundle.bundle_sha256,
            "decisionReference": bundle.value["source"]["decisionReference"],
            "thumbnailFrame": rows["hook"]["frame"],
            "qaStillFrames": {phase: rows[phase]["frame"] for phase in PHASES},
        }
    with _gate_check(gate, gate_path, "provenance_notice"):
        provenance = _write_provenance_notice(
            output,
            source_head=source_head,
            source_tree=source_tree,
            error=source_identity_error,
        )
    gate["checks"]["provenance_notice"] = provenance
    gate["status"] = "passed" if provenance["status"] == "passed" else "failed"
    gate["manifestSha256"] = _sha256(artifacts.manifest)
    _json_write(gate_path, gate)
    receipt["renderGate"] = {"path": GATE_RESULT, "sha256": _sha256(gate_path), "status": gate["status"]}
    deliverables = [artifacts.video, thumbnail, *stills, artifacts.manifest,
                    output / "provenance-notice.json", gate_path, layout_path,
                    *[output / manifest["files"][f"layout:{phase}"]["name"] for phase in ("hook", "question", "reveal")]]
    receipt["files"] = {path.relative_to(output).as_posix(): _sha256(path) for path in deliverables}
    approval = output / "owner-approval.json"
    _json_write(approval, receipt)
    deliverables.append(approval)
    sums = output / "SHA256SUMS.txt"
    sums.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(output).as_posix()}\n" for path in deliverables), encoding="utf-8", newline="\n",
    )
    deliverables.append(sums)
    if copy_to is not None:
        copy_to.mkdir(parents=True, exist_ok=True)
        for path in deliverables:  # one file at a time
            destination = copy_to / path.relative_to(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    print(approval)
    if provenance["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy-to", type=Path, help="Also copy the deliverables to this directory.")
    parser.add_argument("--output", type=Path, help="Write the render and evidence into this directory.")
    parser.add_argument("--public", action="store_true", help="Render the approved public v4 variant without the private notice.")
    parser.add_argument("--source-head", help="Pinned source commit SHA for archive execution.")
    parser.add_argument("--source-tree", help="Pinned source tree SHA for archive execution.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--deal-id", "--deal-identifier", dest="deal_id",
        help="Select one governed V4 deal identifier.",
    )
    selection.add_argument("--seed", help="Deterministically select a governed V4 deal from the source catalog.")
    args = parser.parse_args()
    if not args.public and (args.deal_id is not None or args.seed is not None):
        parser.error("--deal-id/--seed require --public")
    main(
        args.copy_to,
        public=args.public,
        output_dir=args.output,
        source_head=args.source_head,
        source_tree=args.source_tree,
        deal_id=args.deal_id,
        seed=args.seed,
    )
