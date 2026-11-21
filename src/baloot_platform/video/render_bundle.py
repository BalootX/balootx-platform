"""Deterministic observer-safe transport from canonical scenes to renderers.

``RenderBundleV1`` deliberately carries no game engine objects.  It packages
verified ``SceneDocumentV1`` and ``SceneTimelineV1`` values with resolved
presentation configuration, localized copy, and content-addressed assets so
any renderer plugin receives the same bounded input.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from baloot_platform.audit.models import content_hash

from .scene import SceneDocumentV1, SceneTimelineV1, build_scene_timeline
from .media_validation import (
    REMOTION_RENDERER_VERSION,
    renderer_source_sha256,
    validate_renderer_identity,
)

_EXPORT_PRESETS = frozenset({"hooked_flagship", "decision_clip"})
_VOICE_PHRASE_STAGES = {
    "video.decision.your_turn": {"hook", "question"},
    "video.decision.choose_best_move": {"hook", "question"},
    "video.notice.title": None,
    "video.notice.sentence": None,
    "video.decision.best_move": {"reveal"},
    "video.result.title": {"outcome"},
}
_RETIRED_VOICE_PACK = "ar_sa_baloot_v2_smooth"
_RETIRED_VOICE_PROVIDER = "offline_smooth_pack_v2"
_RETIRED_VOICE_FLAGS = ("voice_callouts", "spoken_voice_pack")
_RETIRED_VOICE_DEFAULTS = {
    "voice_provider": _RETIRED_VOICE_PROVIDER,
    "voice_pack_directory": f"assets/audio/voice_packs/{_RETIRED_VOICE_PACK}",
    "voice_pack_manifest": "manifest.json",
    "explanation_voice_mode": "exact_transcript_only",
    "explanation_voice_localization_key": "video.audio.entry_control_explanation",
    "explanation_voice_file": "explanation.wav",
    "explanation_voice_volume": 0.78,
}
_PRESENTATION_KEYS = (
    "audio",
    "colors",
    "layout",
    "motion",
    "timeline",
    "presentation",
    "callouts",
    "safe_areas",
    "exports",
    "typography",
    "texture",
)
MAJLIS_THEME_TOKENS = (
    "w1", "w2", "w3", "rug", "rug_line", "rug_deep", "accent",
    "m1", "m2", "c1", "c2", "wd1", "wd2",
)
MAJLIS_FIXED_TOKENS = (
    "sand", "brass", "brass_light", "card_red", "research_red", "research_tint", "white",
    "team_us", "team_us_text", "team_them", "team_them_text", "ink", "ink_warm", "cream",
    "parchment", "muted", "brown", "edge", "pill_edge", "tile", "meter", "hud_inner", "pass_bg",
    "pass_text", "escalation_bg", "escalation_red", "escalation_ink", "baloot_light",
    "baloot_edge", "ledger_line", "ledger_plus", "timer_track", "decision_red", "escalation_badge",
)
MAJLIS_DECOR_TOKENS = (
    "ceiling", "frond", "frond_light", "frond_dark", "mud", "vent", "lantern_body",
    "lantern_metal", "lantern_core", "lantern_glow", "floor_top", "floor_bottom", "smoke",
    "burner", "tray_rim", "tray", "tray_shadow", "coffee", "date_dark", "date", "date_light",
)
# Every label that can appear inside a call bubble; held to the callouts line limits.
MAJLIS_CALLOUT_KEYS = (
    "video.bid.hokum", "video.bid.sun", "video.bid.ashkal", "video.bid.pass", "video.bid.round.2",
    "video.escalation.decline", "video.escalation.double", "video.escalation.triple",
    "video.escalation.four", "video.escalation.gahwa", "video.escalation.mode.open",
    "video.escalation.mode.locked", "video.project.siri", "video.project.fifty",
    "video.project.hundred", "video.project.hundred_sequence", "video.project.hundred_four_kind",
    "video.project.four_hundred", "video.project.call_label", "video.call.baloot",
)
MAJLIS_MINIMUM_QR_CONTRAST = 7.0
_PLACEHOLDER_HANDLE = re.compile(r"^\[.*\]$")
MAJLIS_THEMES = ("red", "blue", "emerald", "charcoal")
MAJLIS_HUD_STYLES = ("plaque", "slim", "cushions")
MAJLIS_ID_BOXES = ("qr", "number")
MAJLIS_LOCALIZATION_KEYS = (
    "video.hud.trick",
    "video.hud.bidding_round",
    "video.hud.purchase",
    "video.hud.session",
    "video.decision.your_turn",
    "video.governance.research_title",
    "video.governance.not_publishable",
    "video.governance.research_short",
    "video.governance.research_audit_badge",
    "video.project.call_label",
    "video.call.baloot",
)
APPROVED_V4_SOURCE_HEAD = "730500a25ec15c3f6ffa4cd4f7d8e25f01a13850"
APPROVED_V4_SOURCE_TREE = "3a775aa0a6be8d06de83409c316793d3ef9f5f58"
APPROVED_V4_REFERENCE_MP4_SHA256 = "15492d3f389fcb07be423fc97a6e46fe0b8e385b5beb173f01073e6d11a87bf9"
APPROVED_V4_NO_NOTICE = "public_v4_no_justification_no_notice"
APPROVED_V4_RED_NOTICE = "public_v4_no_justification_red_notice"
APPROVED_V4_NOTICE_SENTENCE_SHA256 = "93ff9d2bd274a95b3160bee0e519b258abcf1702801284f9f797fc4c535d5860"
_HEX_COLOUR = re.compile(r"^#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$")


def _majlis_colours(value: Any, label: str, required: Sequence[str] | None) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping of colour tokens")
    if required is not None:
        unknown = sorted(set(value) - set(required))
        missing = sorted(set(required) - set(value))
        if unknown or missing:
            raise ValueError(f"{label} has unknown tokens {unknown} or missing tokens {missing}")
    for token, colour in value.items():
        if not isinstance(colour, str) or not _HEX_COLOUR.fullmatch(colour):
            raise ValueError(f"{label}.{token} must be a hex colour like #RRGGBB, got {colour!r}")
    return {str(token): str(colour) for token, colour in value.items()}


def _luminance(colour: str) -> float:
    digits = colour[1:]
    if len(digits) == 3:
        digits = "".join(item * 2 for item in digits)
    channels = []
    for index in (0, 2, 4):
        value = int(digits[index:index + 2], 16) / 255
        channels.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast_ratio(first: str, second: str) -> float:
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def majlis_placeholder_handles(value: Mapping[str, Any]) -> list[str]:
    """Return the social handles that are still '[...]' placeholders."""

    social = value.get("social") if isinstance(value, Mapping) else None
    social = social if isinstance(social, Mapping) else {}
    return sorted(
        key for key, item in social.items()
        if _PLACEHOLDER_HANDLE.fullmatch(str(item).strip().lstrip("@").strip())
    )


def resolve_majlis_presentation(value: Any) -> dict[str, Any]:
    """Resolve the selected Majlis theme, overrides and options into one palette."""

    if not isinstance(value, Mapping):
        raise ValueError("majlis must be a mapping")
    themes = value.get("themes")
    if not isinstance(themes, Mapping) or sorted(themes) != sorted(MAJLIS_THEMES):
        raise ValueError(f"majlis.themes must define exactly {list(MAJLIS_THEMES)}")
    theme = value.get("theme")
    if theme not in MAJLIS_THEMES:
        raise ValueError(f"majlis.theme must be one of {list(MAJLIS_THEMES)}, got {theme!r}")
    for name in MAJLIS_THEMES:
        _majlis_colours(themes[name], f"majlis.themes.{name}", MAJLIS_THEME_TOKENS)
    fixed = _majlis_colours(value.get("fixed"), "majlis.fixed", MAJLIS_FIXED_TOKENS)
    decor = _majlis_colours(value.get("decor"), "majlis.decor", MAJLIS_DECOR_TOKENS)
    palette = {**_majlis_colours(themes[theme], f"majlis.themes.{theme}", MAJLIS_THEME_TOKENS)}
    for group, label in ((fixed, "majlis.fixed"), (decor, "majlis.decor")):
        clash = sorted(set(group) & set(palette))
        if clash:
            raise ValueError(f"{label} reuses colour tokens {clash}")
        palette.update(group)
    overrides = _majlis_colours(value.get("theme_overrides") or {}, "majlis.theme_overrides", None)
    unknown = sorted(set(overrides) - set(palette))
    if unknown:
        raise ValueError(f"majlis.theme_overrides has unknown colour tokens {unknown}")
    palette.update(overrides)
    contrast = _contrast_ratio(palette["ink"], palette["sand"])
    if contrast < MAJLIS_MINIMUM_QR_CONTRAST or _luminance(palette["ink"]) > _luminance(palette["sand"]):
        raise ValueError(
            "majlis QR colours need dark ink on light sand with contrast >= "
            f"{MAJLIS_MINIMUM_QR_CONTRAST}:1, got {contrast:.2f}:1"
        )
    if value.get("hud") not in MAJLIS_HUD_STYLES:
        raise ValueError(f"majlis.hud must be one of {list(MAJLIS_HUD_STYLES)}")
    if value.get("id_box") not in MAJLIS_ID_BOXES:
        raise ValueError(f"majlis.id_box must be one of {list(MAJLIS_ID_BOXES)}")
    target = value.get("score_target")
    if type(target) is not int or target < 1:
        raise ValueError("majlis.score_target must be a positive integer")
    social = value.get("social")
    if (
        not isinstance(social, Mapping)
        or set(social) != {"tiktok", "instagram"}
        or any(not isinstance(item, str) for item in social.values())
    ):
        raise ValueError("majlis.social must give tiktok and instagram handles (empty hides the pill)")
    research_notice = value.get("research_notice", False)
    if type(research_notice) is not bool:
        raise ValueError("majlis.research_notice must be a boolean")
    return {
        "theme": theme,
        "hud": value["hud"],
        "id_box": value["id_box"],
        "research_notice": research_notice,
        "score_target": target,
        "social": {key: str(item).strip().lstrip("@") for key, item in sorted(social.items())},
        "palette": dict(sorted(palette.items())),
    }


def validate_majlis_profile(profile: Mapping[str, Any]) -> None:
    """Fail closed on a bad Majlis palette or any missing Majlis label."""

    if "majlis" not in profile:
        return
    resolve_majlis_presentation(profile["majlis"])
    strings = profile.get("localizations")
    strings = strings if isinstance(strings, Mapping) else {}
    missing = [key for key in MAJLIS_LOCALIZATION_KEYS if not str(strings.get(key) or "").strip()]
    if missing:
        raise ValueError(f"majlis video labels are missing: {missing}")
    callouts = profile.get("callouts")
    callouts = callouts if isinstance(callouts, Mapping) else {}
    max_lines = int(callouts.get("maximum_lines", 2))
    max_chars = int(callouts.get("maximum_characters_per_line", 16))
    too_long = [
        key for key in MAJLIS_CALLOUT_KEYS
        if key in strings and (
            len(str(strings[key]).splitlines()) > max_lines
            or any(len(line) > max_chars for line in str(strings[key]).splitlines())
        )
    ]
    if too_long:
        raise ValueError(f"majlis call labels exceed {max_lines} x {max_chars} characters: {too_long}")


def validate_approved_v4_public_profile(profile: Mapping[str, Any]) -> None:
    """Fail closed before production renders can fall back to the old premium look."""

    design = profile.get("approved_design")
    if not isinstance(design, Mapping):
        raise ValueError("production video requires the approved-v4 design binding")
    variant = design.get("variant")
    if variant not in (APPROVED_V4_NO_NOTICE, APPROVED_V4_RED_NOTICE):
        raise ValueError("approved-v4 design binding mismatch: variant")
    show_notice = variant == APPROVED_V4_RED_NOTICE
    expected = {
        "source_head": APPROVED_V4_SOURCE_HEAD,
        "source_tree": APPROVED_V4_SOURCE_TREE,
        "reference_mp4_sha256": APPROVED_V4_REFERENCE_MP4_SHA256,
    }
    if show_notice or "reference_variant" in design:
        expected["reference_variant"] = APPROVED_V4_NO_NOTICE
    for key, value in expected.items():
        if design.get(key) != value:
            raise ValueError(f"approved-v4 design binding mismatch: {key}")
    majlis = resolve_majlis_presentation(profile.get("majlis"))
    if majlis["theme"] != "red" or majlis["hud"] != "plaque" or majlis["id_box"] != "number":
        raise ValueError("approved-v4 public render requires the red Majlis number box")
    if majlis["research_notice"] is not show_notice:
        raise ValueError("approved-v4 public research notice must match the design variant")
    if show_notice:
        localizations = profile.get("localizations")
        sentence = localizations.get("video.notice.sentence") if isinstance(localizations, Mapping) else None
        if (
            not isinstance(sentence, str)
            or hashlib.sha256(sentence.encode("utf-8")).hexdigest() != APPROVED_V4_NOTICE_SENTENCE_SHA256
        ):
            raise ValueError("approved-v4 public research notice sentence hash mismatch")
    presentation = profile.get("presentation")
    if not isinstance(presentation, Mapping):
        raise ValueError("approved-v4 public render is missing presentation settings")
    if presentation.get("show_justification_text") is not False or presentation.get("show_justification") is not False:
        raise ValueError("approved-v4 public renders must disable all justification text")
    decision = (profile.get("exports") or {}).get("decision_clip")
    if not isinstance(decision, Mapping) or decision.get("include_complete_replay") is not True:
        raise ValueError("approved-v4 public render requires complete replay")
    if int((profile.get("motion") or {}).get("flight_steps", 0)) != 15:
        raise ValueError("approved-v4 public render requires the governed 15-step card flight")


_FORBIDDEN_KEYS = frozenset(
    {
        "seed",
        "deck",
        "deckOrder",
        "initialHands",
        "legalCards",
        "principalVariation",
        "trustedState",
        "futureCards",
    }
)

# ``sourceSchemaVersion`` was not present on historical governed render bundles.
# Its absence is therefore the immutable v1 boundary; every newly-authored
# governed source is v2 and carries the complete same-game analysis lineage.
HISTORICAL_RENDER_SOURCE_V1_KEYS = frozenset(
    {
        "kind",
        "code",
        "decisionOrdinal",
        "decisionId",
        "auditTraceCode",
        "recordHash",
        "attemptStatus",
        "observerSeat",
        "traceHash",
        "analysisHash",
        "domainHash",
        "rulesHash",
        "originKind",
        "sourceStatus",
        "qualificationStatus",
        "auditOnly",
        "renderPurpose",
        "qualificationFailures",
        "publicationBlockers",
        "publicationEligible",
        "publicationPolicy",
    }
)
CURRENT_RENDER_SOURCE_V2_COMMON_FIELDS = frozenset(
    {
        "sourceSchemaVersion",
        "exactStateId",
        "qrActionable",
    }
)
RESEARCH_RENDER_SOURCE_V2_LINEAGE_FIELDS = frozenset(
    {
        "resultHash",
        "analysisHash",
        "auditHash",
        "analysisProfileHash",
        "analyzerId",
        "analyzerVersion",
    }
)
SCENARIO_RENDER_SOURCE_V2_LINEAGE_FIELDS = frozenset(
    {
        "scenarioHash",
        "proofHash",
        "constraintEvidenceHash",
        "exactStateHash",
    }
)
CURRENT_RENDER_SOURCE_V2_FIELDS = (
    CURRENT_RENDER_SOURCE_V2_COMMON_FIELDS
    | RESEARCH_RENDER_SOURCE_V2_LINEAGE_FIELDS
    | SCENARIO_RENDER_SOURCE_V2_LINEAGE_FIELDS
)
SCENARIO_RENDER_SOURCE_V3_FIELDS = frozenset(
    {
        "sourceSchemaVersion",
        "decisionOrdinal",
        "decisionReference",
        "publicProjectionHash",
        "qrActionable",
        "rulesHash",
        "proofReference",
        "proofCompletionStatus",
        "proofClaimSatisfied",
        "proofAchievedMode",
    }
)
RENDER_SOURCE_KEYS = (
    HISTORICAL_RENDER_SOURCE_V1_KEYS
    | CURRENT_RENDER_SOURCE_V2_FIELDS
    | SCENARIO_RENDER_SOURCE_V3_FIELDS
)
RENDER_PUBLICATION_POLICY_KEYS = frozenset(
    {"schemaVersion", "decisionSelection", "videoProduction", "blockers", "sha256", "qualityEvidenceHash", "interestAdmission"}
)
RENDER_INTEREST_ADMISSION_KEYS = frozenset({"passed", "evidenceHash"})
RENDER_DECISION_SELECTION_EVIDENCE_KEYS = frozenset(
    {
        "schemaVersion",
        "profileHash",
        "evidenceHash",
        "decisionId",
        "gatePassed",
        "qualifiedDecision",
        "blockers",
        "sha256",
    }
)
RENDER_VIDEO_PRODUCTION_EVIDENCE_KEYS = frozenset(
    {"schemaVersion", "profileHash", "candidate", "gatePassed", "blockers", "sha256"}
)
RENDER_VIDEO_CANDIDATE_EVIDENCE_KEYS = frozenset(
    {
        "decisionKind",
        "critical",
        "truthLabel",
        "selectionScore",
        "confidence",
        "compatibilityBlockers",
    }
)
RENDER_AUDIT_VIDEO_CANDIDATE_EVIDENCE_KEYS = frozenset({"decisionId", "auditOnly"})

_RENDER_SOURCE_REQUIRED_GOVERNANCE = frozenset(
    {
        "kind",
        "code",
        "qualificationStatus",
        "renderPurpose",
        "qualificationFailures",
        "publicationBlockers",
        "observerSeat",
        "auditOnly",
        "publicationEligible",
        "publicationPolicy",
    }
)
_RENDER_SOURCE_V2_COMMON_REQUIRED = _RENDER_SOURCE_REQUIRED_GOVERNANCE | frozenset(
    {
        "sourceSchemaVersion",
        "decisionOrdinal",
        "decisionId",
        "traceHash",
        "rulesHash",
        "exactStateId",
        "qrActionable",
    }
)
_RESEARCH_RENDER_SOURCE_V2_REQUIRED = _RENDER_SOURCE_V2_COMMON_REQUIRED | frozenset(
    {
        "resultHash",
        "analysisHash",
        "auditHash",
        "analysisProfileHash",
        "analyzerId",
        "analyzerVersion",
    }
)
_SCENARIO_RENDER_SOURCE_V2_REQUIRED = (
    _RENDER_SOURCE_V2_COMMON_REQUIRED
    | frozenset({"resultHash", "rulesHash"})
    | SCENARIO_RENDER_SOURCE_V2_LINEAGE_FIELDS
)
_SCENARIO_RENDER_SOURCE_V3_REQUIRED = (
    _RENDER_SOURCE_REQUIRED_GOVERNANCE | SCENARIO_RENDER_SOURCE_V3_FIELDS
)
_RENDER_SOURCE_KINDS = frozenset(
    {
        "research_result",
        "research_scenario_result",
        "analysis_candidate",
        "finalist",
        "laffa",
        "research_audit_trace",
    }
)
_RENDER_SOURCE_STATUSES = frozenset(
    {"qualified", "provisional", "rejected", "quarantined", "legacy_replay"}
)
_RENDER_ATTEMPT_STATUSES = frozenset(
    {"running", "completed", "retry_scheduled", "failed", "reclaimed", "cancelled"}
)
_SOURCE_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEMANTIC_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")


def resolve_motion_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the governed normal or reduced presentation motion profile."""

    motion = dict(profile.get("motion") or {})
    mode = str(motion.pop("mode", "normal"))
    reduced = dict(motion.pop("reduced", {}) or {})
    if mode not in {"normal", "reduced"}:
        raise ValueError("motion.mode must be normal or reduced")
    if mode == "reduced":
        motion.update(reduced)
    motion["mode"] = mode
    motion["available_modes"] = ["normal", "reduced"]
    return motion


def _json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _reject_forbidden_keys(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_KEYS:
                raise ValueError(f"render bundle contains forbidden field {path}/{key}")
            if _RETIRED_VOICE_PACK in key.casefold():
                raise ValueError(f"retired voice pack reference at {path}/{key}")
            _reject_forbidden_keys(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_keys(item, f"{path}/{index}")
    elif isinstance(value, str) and _RETIRED_VOICE_PACK in value.casefold():
        raise ValueError(f"retired voice pack reference at {path}")


def _closed_mapping(
    value: Any,
    *,
    allowed_keys: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    unknown = sorted(str(key) for key in value if key not in allowed_keys)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {unknown}")
    return value


def validate_render_interest_admission(value: Any) -> None:
    admission = _closed_mapping(value, allowed_keys=RENDER_INTEREST_ADMISSION_KEYS,
                                label="render publication interest admission")
    if (set(admission) != RENDER_INTEREST_ADMISSION_KEYS
            or type(admission["passed"]) is not bool
            or not isinstance(admission["evidenceHash"], str)
            or not _SHA256.fullmatch(admission["evidenceHash"])):
        raise ValueError("render publication interest admission is invalid")


def _validate_publication_policy(
    value: Any,
    *,
    blockers: list[str],
) -> None:
    policy = _closed_mapping(
        value,
        allowed_keys=RENDER_PUBLICATION_POLICY_KEYS,
        label="render source publication policy",
    )
    policy_value = _json_copy(policy)
    if "interestAdmission" in policy_value:
        validate_render_interest_admission(policy_value["interestAdmission"])
    if "qualityEvidenceHash" in policy_value:
        quality_hash = policy_value["qualityEvidenceHash"]
        if not isinstance(quality_hash, str) or not _SHA256.fullmatch(quality_hash):
            raise ValueError("render source quality evidence hash is invalid")
    declared_policy_hash = policy_value.pop("sha256", None)
    if (
        policy_value.get("schemaVersion") != 1
        or declared_policy_hash != content_hash(policy_value)
        or policy_value.get("blockers") != blockers
    ):
        raise ValueError("render source publication policy evidence is invalid")

    decision = _closed_mapping(
        policy_value.get("decisionSelection"),
        allowed_keys=RENDER_DECISION_SELECTION_EVIDENCE_KEYS,
        label="render source publication policy decisionSelection",
    )
    video = _closed_mapping(
        policy_value.get("videoProduction"),
        allowed_keys=RENDER_VIDEO_PRODUCTION_EVIDENCE_KEYS,
        label="render source publication policy videoProduction",
    )
    for evidence in (decision, video):
        evidence_value = _json_copy(evidence)
        declared_evidence_hash = evidence_value.pop("sha256", None)
        if declared_evidence_hash != content_hash(evidence_value):
            raise ValueError("render source publication evidence checksum mismatch")

    candidate = video.get("candidate")
    if candidate is not None:
        candidate_keys = (
            RENDER_AUDIT_VIDEO_CANDIDATE_EVIDENCE_KEYS
            if isinstance(candidate, Mapping) and candidate.get("auditOnly") is True
            else RENDER_VIDEO_CANDIDATE_EVIDENCE_KEYS
        )
        _closed_mapping(
            candidate,
            allowed_keys=candidate_keys,
            label="render source publication policy videoProduction candidate",
        )


def normalize_render_source(
    value: Mapping[str, Any],
    *,
    observer: str,
    require_current: bool = False,
) -> dict[str, Any]:
    """Return one closed historical, v2, or observer-safe scenario-v3 source."""

    source = dict(
        _closed_mapping(
            _json_copy(value),
            allowed_keys=RENDER_SOURCE_KEYS,
            label="render source",
        )
    )
    schema_version = source.get("sourceSchemaVersion")
    if schema_version is None:
        required = _RENDER_SOURCE_REQUIRED_GOVERNANCE
        current_only = sorted(
            (
                (
                    CURRENT_RENDER_SOURCE_V2_FIELDS
                    | SCENARIO_RENDER_SOURCE_V3_FIELDS
                )
                - {"sourceSchemaVersion"}
            )
            & set(source)
        )
        if current_only:
            raise ValueError(
                "historical render source cannot contain current-only fields: "
                f"{current_only}"
            )
        if require_current:
            raise ValueError(
                "new governed render sources require a current sourceSchemaVersion"
            )
    elif schema_version == 2 and not isinstance(schema_version, bool):
        kind = source.get("kind")
        if require_current and kind == "research_scenario_result":
            raise ValueError(
                "new Research V4 scenario render sources require sourceSchemaVersion 3"
            )
        required = (
            _RESEARCH_RENDER_SOURCE_V2_REQUIRED
            if kind == "research_result"
            else _SCENARIO_RENDER_SOURCE_V2_REQUIRED
            if kind == "research_scenario_result"
            else _RENDER_SOURCE_V2_COMMON_REQUIRED
        )
        if kind != "research_result":
            foreign_fields = (
                (RESEARCH_RENDER_SOURCE_V2_LINEAGE_FIELDS - {"resultHash"})
            )
            if kind != "research_scenario_result":
                foreign_fields |= {"resultHash"}
            foreign_lineage = sorted(foreign_fields & set(source))
            if foreign_lineage:
                raise ValueError(
                    "non-Research render source contains Research-result lineage: "
                    f"{foreign_lineage}"
                )
        if kind != "research_scenario_result":
            foreign_scenario_lineage = sorted(
                SCENARIO_RENDER_SOURCE_V2_LINEAGE_FIELDS & set(source)
            )
            if foreign_scenario_lineage:
                raise ValueError(
                    "non-scenario render source contains scenario-result lineage: "
                    f"{foreign_scenario_lineage}"
                )
    elif schema_version == 3 and not isinstance(schema_version, bool):
        if source.get("kind") != "research_scenario_result":
            raise ValueError(
                "render source schema version 3 is reserved for Research V4 scenarios"
            )
        required = _SCENARIO_RENDER_SOURCE_V3_REQUIRED
        private_or_foreign_fields = sorted(set(source) - required)
        if private_or_foreign_fields:
            raise ValueError(
                "observer-safe scenario render source contains private or foreign "
                f"fields: {private_or_foreign_fields}"
            )
    else:
        raise ValueError("render source schema version is unsupported")
    if not required.issubset(source):
        missing = sorted(required - set(source))
        raise ValueError(f"render source governance is incomplete: {missing}")
    if source.get("kind") not in _RENDER_SOURCE_KINDS:
        raise ValueError("render source kind is invalid")
    if not isinstance(source.get("code"), str) or not _SOURCE_CODE.fullmatch(
        source["code"]
    ):
        raise ValueError("render source code is invalid")
    if source.get("qualificationStatus") not in _RENDER_SOURCE_STATUSES:
        raise ValueError("render source qualification status is invalid")
    if source.get("observerSeat") != observer:
        raise ValueError(
            "render source observer does not match the bundled perspective"
        )
    ordinal = source.get("decisionOrdinal")
    if ordinal is not None and (
        not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 1
    ):
        raise ValueError("render source decision ordinal must be positive")
    for key in (
        "decisionId",
        "auditTraceCode",
        "originKind",
        "sourceStatus",
        "proofReference",
        "proofCompletionStatus",
        "proofAchievedMode",
    ):
        item = source.get(key)
        if item is not None and (not isinstance(item, str) or not item.strip()):
            raise ValueError(f"render source {key} must be a non-empty string")
    if schema_version == 3:
        reference = source.get("decisionReference")
        if (
            not isinstance(reference, Mapping)
            or set(reference) != {"resultCode", "decisionOrdinal"}
            or reference.get("resultCode") != source.get("code")
            or type(reference.get("decisionOrdinal")) is not int
            or reference.get("decisionOrdinal") != source.get("decisionOrdinal")
            or reference.get("decisionOrdinal") != 1
        ):
            raise ValueError(
                "observer-safe scenario decision reference must use its retained "
                "result code and ordinal"
            )
        if source.get("proofReference") != source.get("code"):
            raise ValueError(
                "observer-safe scenario proof reference must use its retained result code"
            )
        if source.get("proofCompletionStatus") not in {
            "exact_complete",
            "bounded_complete",
            "policy_complete",
            "proof_incomplete",
            "proof_failed",
        }:
            raise ValueError("observer-safe scenario proof completion is invalid")
        if source.get("proofAchievedMode") not in {
            "exact",
            "bounded",
            "policy_conditioned",
            "incomplete",
        }:
            raise ValueError("observer-safe scenario proof mode is invalid")
        if type(source.get("proofClaimSatisfied")) is not bool:
            raise ValueError(
                "observer-safe scenario proof claim status must be boolean"
            )
    attempt_status = source.get("attemptStatus")
    if attempt_status is not None and attempt_status not in _RENDER_ATTEMPT_STATUSES:
        raise ValueError("render source attempt status is invalid")
    for key in (
        "recordHash",
        "traceHash",
        "analysisHash",
        "domainHash",
        "rulesHash",
        "resultHash",
        "auditHash",
        "analysisProfileHash",
        "scenarioHash",
        "proofHash",
        "constraintEvidenceHash",
        "exactStateHash",
        "publicProjectionHash",
    ):
        item = source.get(key)
        if item is not None and (
            not isinstance(item, str) or not _SHA256.fullmatch(item)
        ):
            raise ValueError(f"render source {key} must be a SHA-256 digest")
    if not isinstance(source.get("auditOnly"), bool) or not isinstance(
        source.get("publicationEligible"), bool
    ):
        raise ValueError(
            "render source audit-only and publication eligibility must be boolean"
        )
    purpose = source.get("renderPurpose")
    if purpose not in {"content_candidate", "research_audit"}:
        raise ValueError("render source purpose is invalid")

    failures = source.get("qualificationFailures")
    blockers = source.get("publicationBlockers")
    for label, items in (
        ("qualification failures", failures),
        ("publication blockers", blockers),
    ):
        if (
            not isinstance(items, list)
            or any(not isinstance(item, str) or not item.strip() for item in items)
            or len(items) != len(set(items))
        ):
            raise ValueError(f"render source {label} must be unique non-empty strings")

    _validate_publication_policy(source.get("publicationPolicy"), blockers=blockers)
    if schema_version == 3:
        decision_evidence = dict(source["publicationPolicy"]["decisionSelection"])
        if (
            decision_evidence.get("decisionId") != source.get("code")
            or decision_evidence.get("evidenceHash") is not None
        ):
            raise ValueError(
                "observer-safe scenario publication policy contains private proof identity"
            )

    eligible = source["publicationEligible"]
    if source["auditOnly"] != (not eligible):
        raise ValueError(
            "render source audit-only status contradicts publication eligibility"
        )
    if eligible != (purpose == "content_candidate"):
        raise ValueError("render source purpose contradicts publication eligibility")
    if eligible and (
        source["qualificationStatus"] != "qualified" or failures or blockers
    ):
        raise ValueError("publishable render source still has qualification blockers")
    if schema_version == 2:
        if source.get("kind") == "research_result":
            analyzer_id = source.get("analyzerId")
            analyzer_version = source.get("analyzerVersion")
            if not isinstance(analyzer_id, str) or not analyzer_id.strip():
                raise ValueError("render source analyzer id must be a non-empty string")
            if not isinstance(analyzer_version, str) or not _SEMANTIC_VERSION.fullmatch(
                analyzer_version
            ):
                raise ValueError(
                    "render source analyzer version must be semantic x.y.z"
                )
        exact_state_id = source.get("exactStateId")
        if (
            not isinstance(exact_state_id, str)
            or not exact_state_id.strip()
            or exact_state_id != source.get("decisionId")
        ):
            raise ValueError("render source exact state id must equal its decision id")
        qr_actionable = source.get("qrActionable")
        if not isinstance(qr_actionable, bool) or (qr_actionable and not eligible):
            raise ValueError(
                "render source QR actionability contradicts publication eligibility"
            )
    return source


def _scene_decision_kind(
    scenes: Sequence[SceneDocumentV1 | Mapping[str, Any]],
) -> str:
    for scene in scenes:
        value = scene.value if isinstance(scene, SceneDocumentV1) else scene
        if value.get("stage") != "reveal":
            continue
        analysis = next(
            (
                layer.get("content")
                for layer in value.get("layers", ())
                if layer.get("id") == "analysis"
            ),
            None,
        )
        if not isinstance(analysis, Mapping) or "decisionKind" not in analysis:
            return "play"
        decision_kind = str(analysis["decisionKind"])
        if decision_kind not in {"bid", "escalation", "project", "play"}:
            raise ValueError("render scene decision kind is unsupported")
        return decision_kind
    return "play"


def seeded_player_names(identity_profile: Mapping[str, Any], seed: str) -> dict[str, str]:
    """Pick four distinct pool names for one video, stable for its identity."""
    pool = identity_profile.get("player_name_pool")
    if not pool:
        return dict(identity_profile["default_player_names"])
    names = sorted(
        dict.fromkeys(str(name).strip() for name in pool if str(name).strip()),
        key=lambda name: hashlib.sha256(f"{seed}\x1f{name}".encode("utf-8")).hexdigest(),
    )
    if len(names) < 4:
        raise ValueError("identity.player_name_pool needs at least four distinct names")
    return dict(zip(("north", "west", "south", "east"), names))


def _has_preplay_setup(
    scenes: Sequence[SceneDocumentV1 | Mapping[str, Any]],
) -> bool:
    for scene in scenes:
        value = scene.value if isinstance(scene, SceneDocumentV1) else scene
        if value.get("stage") != "continuation":
            continue
        layers = {
            layer.get("id"): layer.get("content")
            for layer in value.get("layers", ())
            if isinstance(layer, Mapping)
        }
        public = layers.get("public-lifecycle")
        private = layers.get("observer-lifecycle")
        return bool(
            isinstance(public, Mapping)
            and isinstance(private, Mapping)
            and isinstance(public.get("bidHistory"), list)
            and public["bidHistory"]
            and public.get("purchaseRecipient") in {"north", "west", "south", "east"}
            and isinstance(private.get("initialFiveCards"), list)
            and len(private["initialFiveCards"]) == 5
        )
    return False


def _lifecycle_play_counts(
    scenes: Sequence[SceneDocumentV1 | Mapping[str, Any]],
) -> tuple[int, int]:
    """(plays before the decision, plays in the round) from the continuation lifecycle."""
    for scene in scenes:
        value = scene.value if isinstance(scene, SceneDocumentV1) else scene
        if value.get("stage") != "continuation":
            continue
        public = next(
            (layer.get("content") for layer in value.get("layers", ())
             if isinstance(layer, Mapping) and layer.get("id") == "public-lifecycle"),
            None,
        )
        if isinstance(public, Mapping) and isinstance(public.get("plays"), list):
            pause = public.get("pauseBeforePlayIndex")
            if type(pause) is int and 0 < pause < len(public["plays"]):
                return pause, len(public["plays"])
    raise ValueError("complete replay timing requires the round's public lifecycle")


def _export_render(profile: Mapping[str, Any], export_preset: str) -> dict[str, Any]:
    render = dict(profile["render"])
    fps = profile["exports"][export_preset].get("fps", render["fps"])
    if type(fps) is not int or fps not in {30, 60}:
        raise ValueError("export fps must be 30 or 60")
    render["fps"] = fps
    return render


def build_export_timeline(
    *,
    scenes: Sequence[SceneDocumentV1],
    profile: Mapping[str, Any],
    export_preset: str,
) -> SceneTimelineV1:
    """Resolve one frame-authoritative export timeline from governed settings."""

    render = _export_render(profile, export_preset)
    export = profile["exports"][export_preset]
    scene_ids = {str(scene.value["stage"]): scene.scene_id for scene in scenes}
    question = scene_ids.get("question") or next(iter(scene_ids.values()))
    reveal = scene_ids.get("reveal") or question
    context = scene_ids.get("context") or question
    continuation = scene_ids.get("continuation") or reveal
    outcome = scene_ids.get("outcome") or reveal
    hook = float(export["opening_hook_seconds"])
    rewind = float(export["rewind_seconds"])
    target = float(export["target_seconds"])
    decision_kind = _scene_decision_kind(scenes)
    is_nonplay = decision_kind in {"bid", "escalation", "project"}

    if export_preset == "hooked_flagship":
        play_end = float(export["play_end_seconds"])
        reveal_end = float(export["reveal_end_seconds"])
        decision = float(export["decision_seconds"])
        reveal_seconds = float(export["reveal_seconds"])
        outcome_seconds = target - reveal_end
        if is_nonplay:
            phase_key = {
                "bid": "bidding_seconds",
                "escalation": "escalation_seconds",
                "project": "project_seconds",
            }[decision_kind]
            phase_seconds = float(profile["timeline"][phase_key])
            continuation_seconds = reveal_end - (
                hook + rewind + phase_seconds + decision + reveal_seconds
            )
            if continuation_seconds <= 0:
                raise ValueError(
                    "hooked_flagship non-play phase leaves no continuation time"
                )
            segments = (
                ("hook", question, hook),
                ("rewind", question, rewind),
                ("context", context, phase_seconds),
                ("question", question, decision),
                ("reveal", reveal, reveal_seconds),
                ("continuation", continuation, continuation_seconds),
                ("outcome", outcome, outcome_seconds),
            )
        else:
            continuation_seconds = float(export["continuation_seconds"])
            setup_ready = _has_preplay_setup(scenes)
            setup_end = (
                float(export["setup_end_seconds"])
                if setup_ready else hook + rewind
            )
            replay = play_end - setup_end
            if (
                (setup_ready and setup_end <= hook + rewind)
                or replay <= 0
                or abs(
                    play_end
                    + decision
                    + reveal_seconds
                    + continuation_seconds
                    - reveal_end
                )
                > 1e-9
            ):
                raise ValueError(
                    "hooked_flagship governed stages must reconcile to reveal_end_seconds"
                )
            segments = (
                ("hook", question, hook),
                ("rewind", question, rewind),
                *((("setup", context, setup_end - hook - rewind),) if setup_ready else ()),
                ("replay", context, replay),
                ("question", question, decision),
                ("reveal", reveal, reveal_seconds),
                ("continuation", continuation, continuation_seconds),
                ("outcome", outcome, outcome_seconds),
            )
    elif export.get("include_complete_replay") is True:
        # Pause on the question, replay the recorded round from the deal up to
        # that decision (never past it), pause again, then reveal and hold.
        if is_nonplay or not _has_preplay_setup(scenes):
            raise ValueError(
                "decision_clip complete replay requires a play decision with recorded bidding"
            )
        decision = float(export["decision_seconds"])
        reveal_seconds = float(export["reveal_seconds"])
        setup_seconds = float(export["setup_seconds"])
        to_round_end = export.get("continue_to_round_end") is True
        play_seconds = export.get("play_seconds")
        if to_round_end and play_seconds is None:
            raise ValueError("decision_clip continue_to_round_end requires play_seconds")
        pause, total_plays = _lifecycle_play_counts(scenes)
        # play_seconds paces every play alike; replay_seconds is the older whole-segment form.
        replay_seconds = (
            float(play_seconds) * pause if play_seconds is not None
            else float(export["replay_seconds"])
        )
        continuation_seconds = (
            float(play_seconds) * (total_plays - pause) if to_round_end else 0.0
        )
        # The last stage (hold, or the final-score outcome) takes the rest of target_seconds.
        rest = target - (
            hook + rewind + setup_seconds + replay_seconds + decision + reveal_seconds
            + continuation_seconds
        )
        if min(setup_seconds, replay_seconds, decision, reveal_seconds, rest) <= 0 or (
            to_round_end and continuation_seconds <= 0
        ):
            raise ValueError("decision_clip complete replay stages must all be positive")
        segments = (
            ("hook", question, hook),
            ("rewind", question, rewind),
            ("setup", context, setup_seconds),
            ("replay", context, replay_seconds),
            ("question", question, decision),
            ("reveal", reveal, reveal_seconds),
            *(
                (("continuation", continuation, continuation_seconds),
                 ("outcome", continuation, rest))
                if to_round_end else (("hold", continuation, rest),)
            ),
        )
    else:
        context_seconds = float(export["context_seconds"])
        decision = float(export["decision_seconds"])
        reveal_seconds = float(export["reveal_seconds"])
        continuation_seconds = float(export["continuation_seconds"])
        outcome_seconds = float(export.get("outcome_seconds", 0))
        hold = target - (
            hook
            + rewind
            + context_seconds
            + decision
            + reveal_seconds
            + continuation_seconds
            + outcome_seconds
        )
        segments = (
            ("hook", question, hook),
            ("rewind", question, rewind),
            ("context", context, context_seconds),
            ("question", question, decision),
            ("reveal", reveal, reveal_seconds),
            ("continuation", continuation, continuation_seconds),
            *((("outcome", continuation, outcome_seconds),) if outcome_seconds > 0 else ()),
            ("hold", continuation, hold),
        )
    timeline = build_scene_timeline(
        fps=int(render["fps"]),
        width=int(render["width"]),
        height=int(render["height"]),
        segments=segments,
    )
    minimum = int(float(export["minimum_seconds"]) * int(render["fps"]))
    maximum = int(float(export["maximum_seconds"]) * int(render["fps"]))
    if not minimum <= timeline.value["totalFrames"] <= maximum:
        raise ValueError(
            f"{export_preset} timeline is outside its governed duration bounds"
        )
    return timeline


def _validate_export_timeline(
    timeline: SceneTimelineV1,
    *,
    profile: Mapping[str, Any],
    export_preset: str,
    decision_kind: str = "play",
    scenes: Sequence[SceneDocumentV1 | Mapping[str, Any]] = (),
) -> None:
    """Validate the supplied frame authority without replacing it."""
    value = timeline.value
    render = _export_render(profile, export_preset)
    if value["canvas"] != {
        "width": int(render["width"]),
        "height": int(render["height"]),
        "aspectRatio": "9:16",
    }:
        raise ValueError("render timeline canvas does not match the governed profile")
    if value["frameRate"] != {"numerator": int(render["fps"]), "denominator": 1}:
        raise ValueError(
            "render timeline frame rate does not match the governed profile"
        )
    export = profile["exports"][export_preset]
    expected_frames = round(float(export["target_seconds"]) * int(render["fps"]))
    if int(value["totalFrames"]) != expected_frames:
        raise ValueError("render timeline does not match the governed export duration")
    expected_stages = (
        ("hook", "rewind", "context", "question", "reveal", "continuation", "outcome")
        if export_preset == "hooked_flagship"
        and decision_kind in {"bid", "escalation", "project"}
        else ("hook", "rewind", "setup", "replay", "question", "reveal", "continuation", "outcome")
        if export_preset == "hooked_flagship" and _has_preplay_setup(scenes)
        else ("hook", "rewind", "replay", "question", "reveal", "continuation", "outcome")
        if export_preset == "hooked_flagship"
        else ("hook", "rewind", "setup", "replay", "question", "reveal", "continuation", "outcome")
        if export.get("include_complete_replay") is True and export.get("continue_to_round_end") is True
        else ("hook", "rewind", "setup", "replay", "question", "reveal", "hold")
        if export.get("include_complete_replay") is True
        else ("hook", "rewind", "context", "question", "reveal", "continuation", "outcome", "hold")
        if float(export.get("outcome_seconds", 0)) > 0
        else ("hook", "rewind", "context", "question", "reveal", "continuation", "hold")
    )
    actual_stages = tuple(str(item["stage"]) for item in value["segments"])
    if actual_stages != expected_stages:
        raise ValueError(
            "render timeline stage sequence does not match the export contract"
        )
    if (
        export_preset == "hooked_flagship"
        and decision_kind in {"bid", "escalation", "project"}
    ):
        phase_key = {
            "bid": "bidding_seconds",
            "escalation": "escalation_seconds",
            "project": "project_seconds",
        }[decision_kind]
        phase_seconds = float(profile["timeline"][phase_key])
        continuation_seconds = float(export["reveal_end_seconds"]) - (
            float(export["opening_hook_seconds"])
            + float(export["rewind_seconds"])
            + phase_seconds
            + float(export["decision_seconds"])
            + float(export["reveal_seconds"])
        )
        expected_durations = {
            "context": int(
                (Decimal(str(phase_seconds)) * int(render["fps"])).to_integral_value(
                    rounding=ROUND_HALF_UP
                )
            ),
            "continuation": int(
                (
                    Decimal(str(continuation_seconds)) * int(render["fps"])
                ).to_integral_value(rounding=ROUND_HALF_UP)
            ),
        }
        actual_durations = {
            str(item["stage"]): int(item["durationFrames"])
            for item in value["segments"]
        }
        if any(
            actual_durations[stage] != duration
            for stage, duration in expected_durations.items()
        ):
            raise ValueError(
                "render timeline non-play phase durations do not match the governed profile"
            )


def _asset_entry(
    root: Path, relative: str, *, asset_id: str, license_id: str
) -> dict[str, Any]:
    normalized = PurePosixPath(relative)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"asset path must be repository relative: {relative}")
    path = root.joinpath(*normalized.parts).resolve()
    repository = root.resolve()
    if repository not in path.parents or not path.is_file():
        raise ValueError(
            f"render asset is missing or outside the repository: {relative}"
        )
    payload = path.read_bytes()
    return {
        "id": asset_id,
        "path": normalized.as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "license": license_id,
    }


def validate_render_voice(value: Mapping[str, Any]) -> None:
    """Reject legacy voice settings and bind each cue to its on-screen phrase."""
    _reject_forbidden_keys(value)
    presentation = value["presentation"]
    audio = presentation.get("audio", {})
    if audio.get("voice_provider") == _RETIRED_VOICE_PROVIDER:
        raise ValueError("retired voice pack reference at presentation/audio/voice_provider")
    for field in ("one_piece_explanation", "voice_callouts", "spoken_voice_pack", "synthetic_callout_cues"):
        if audio.get(field, False) is not False:
            key = audio.get("explanation_voice_localization_key") or field
            raise ValueError(f"unsupported voice setting {field} for {key}")

    cues = value.get("audioCues")
    if not isinstance(cues, list):
        raise ValueError("audioCues must be a list")
    if not cues:
        return

    localizations = value["localizations"]
    assets = {asset["id"]: asset for asset in value["assets"]}
    scenes = {scene["stage"]: scene for scene in value["scenes"]}
    question = scenes["question"]
    prompt = next(layer["content"] for layer in question["layers"] if layer["id"] == "decision-prompt")
    table = next(layer["content"] for layer in question["layers"] if layer["id"] == "table")
    analysis = next(layer["content"] for layer in scenes["reveal"]["layers"] if layer["id"] == "analysis")
    is_play = analysis.get("decisionKind", "play") == "play"
    stages = {segment["stage"] for segment in value["timeline"]["segments"]}
    source = value.get("source")
    notice_visible = presentation.get("majlis", {}).get("research_notice") is True and (
        source is None or (
            source.get("qualificationStatus") == "qualified"
            and source.get("renderPurpose") == "content_candidate"
            and source.get("auditOnly") is False
            and source.get("publicationEligible") is True
            and source.get("qualificationFailures") == []
            and source.get("publicationBlockers") == []
        )
    )
    for cue in cues:
        key = cue.get("transcriptKey", "audioCues.transcriptKey") if isinstance(cue, Mapping) else "audioCues"
        if not isinstance(cue, Mapping) or set(cue) != {
            "stage", "assetId", "path", "volume", "transcriptKey", "transcriptSha256",
        }:
            raise ValueError(f"invalid voice cue fields for {key}")
        if not isinstance(key, str) or key not in _VOICE_PHRASE_STAGES:
            raise ValueError(f"voice key is outside the allowlist: {key}")
        if key in (analysis.get("reasonKey"), analysis.get("proofScopeKey")):
            raise ValueError(f"voice cue uses a rationale key: {key}")
        stage = cue["stage"]
        allowed_stages = _VOICE_PHRASE_STAGES[key]
        if (
            not isinstance(stage, str) or stage not in stages
            or (allowed_stages is None and not notice_visible)
            or (allowed_stages is not None and stage not in allowed_stages)
            or (allowed_stages is not None and stage == "hook" and not is_play)
            or (key == "video.decision.choose_best_move" and prompt.get("textKey", key) != key)
            or (key == "video.decision.best_move" and not analysis.get("selectedAction"))
            or (
                key == "video.result.title" and not is_play
                and (value["exportPreset"] == "decision_clip" or analysis.get("continuationComplete") is not True)
            )
        ):
            raise ValueError(f"voice key is not visible on screen: {key}")
        text = localizations.get(key)
        if isinstance(text, str) and key == "video.decision.your_turn":
            observer = question["observer"]
            players = value["identity"]["players"]
            acting_seat = prompt.get("actingSeat") or table.get("currentPlayer") or observer
            text = text.replace("{name}", players.get(acting_seat, players[observer]))
        if not isinstance(text, str) or not text.strip() or (
            hashlib.sha256(text.encode("utf-8")).hexdigest() != cue["transcriptSha256"]
        ):
            raise ValueError(f"voice transcript differs from on-screen text: {key}")
        asset = assets.get(str(cue["assetId"]))
        volume = cue["volume"]
        if (
            asset is None or asset.get("path") != f"assets/{cue['path']}"
            or not isinstance(volume, (int, float))
            or isinstance(volume, bool)
            or not 0 <= volume <= 1
        ):
            raise ValueError(f"invalid voice cue asset or volume for {key}")


@dataclass(frozen=True, slots=True, init=False)
class RenderBundleV1:
    """Renderer-neutral, observer-safe and content-addressed render request."""

    _canonical_value: str

    def __init__(self, value: Mapping[str, Any]) -> None:
        normalized = _json_copy(value)
        object.__setattr__(
            self,
            "_canonical_value",
            json.dumps(
                normalized,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    @property
    def value(self) -> dict[str, Any]:
        """Return a defensive copy so a verified bundle cannot be mutated in place."""

        return json.loads(self._canonical_value)

    @property
    def bundle_sha256(self) -> str:
        return str(self.value["bundleSha256"])

    def to_dict(self) -> dict[str, Any]:
        return self.value

    def verify(self, *, raise_voice_errors: bool = False) -> bool:
        value = self.to_dict()
        try:
            validate_render_voice(value)
        except ValueError:
            if raise_voice_errors:
                raise
            return False
        except (KeyError, AttributeError, StopIteration, TypeError):
            return False
        bundle_sha256 = value.pop("bundleSha256", None)
        if value.get("schemaVersion") != 1:
            return False
        if value.get("exportPreset") not in _EXPORT_PRESETS:
            return False
        if value.get("locale") != "ar-SA" or value.get("direction") != "rtl":
            return False
        observer = value.get("observerSeat")
        if observer not in {"north", "west", "south", "east"}:
            return False
        source = value.get("source")
        if source is not None:
            if not isinstance(source, Mapping):
                return False
            try:
                normalize_render_source(source, observer=str(observer))
            except (TypeError, ValueError):
                return False
        try:
            repository_root = Path(__file__).resolve().parents[3]
            validate_renderer_identity(
                value.get("renderer") or {}, root=repository_root
            )
        except (RuntimeError, TypeError, ValueError):
            return False
        scenes = value.get("scenes")
        timeline = value.get("timeline")
        if (
            not isinstance(scenes, list)
            or not scenes
            or not isinstance(timeline, Mapping)
        ):
            return False
        if not all(SceneDocumentV1(dict(scene)).verify() for scene in scenes):
            return False
        pre_decision = [scene for scene in scenes if scene.get("stage") in {"context", "question", "selection"}]
        if len({(scene["source"]["observerViewSha256"], scene["source"]["publicEventLineageSha256"])
                for scene in pre_decision}) > 1:
            return False
        if len({content_hash([layer for layer in scene["layers"]
                              if layer["id"] in {"table", "seats", "observer-hand", "public-events"}])
                for scene in pre_decision}) > 1:
            return False
        if {str(scene.get("observer")) for scene in scenes} != {observer}:
            return False
        if not SceneTimelineV1(dict(timeline)).verify():
            return False
        try:
            _validate_export_timeline(
                SceneTimelineV1(dict(timeline)),
                profile={
                    "render": value.get("render"),
                    "exports": value.get("presentation", {}).get("exports"),
                    "timeline": value.get("presentation", {}).get("timeline"),
                },
                export_preset=str(value.get("exportPreset")),
                decision_kind=_scene_decision_kind(scenes),
                scenes=scenes,
            )
        except (KeyError, TypeError, ValueError):
            return False
        scene_ids = {str(scene["sceneId"]) for scene in scenes}
        if any(
            str(segment["sceneId"]) not in scene_ids for segment in timeline["segments"]
        ):
            return False
        decision_index = value.get("decisionIndex")
        if not isinstance(decision_index, list):
            return False
        if any(
            not isinstance(item, Mapping)
            or set(item) != {"label", "decisionId"}
            or not isinstance(item["label"], str)
            or not item["label"].strip()
            or not isinstance(item["decisionId"], str)
            or not item["decisionId"].strip()
            for item in decision_index
        ):
            return False
        if not isinstance(value.get("assets"), list) or not isinstance(value.get("localizations"), Mapping):
            return False
        identity = value.get("identity")
        if not isinstance(identity, Mapping) or set(identity) != {
            "gameSerial",
            "sceneId",
            "enrollmentUri",
            "players",
        }:
            return False
        if any(
            not isinstance(identity.get(key), str) or not str(identity[key]).strip()
            for key in ("gameSerial", "sceneId", "enrollmentUri")
        ):
            return False
        if identity["sceneId"] not in scene_ids:
            return False
        players = identity.get("players")
        if not isinstance(players, Mapping) or set(players) != {
            "north",
            "west",
            "east",
            "south",
        }:
            return False
        if any(
            not isinstance(name, str) or not name.strip() for name in players.values()
        ):
            return False
        return bundle_sha256 == content_hash(value)


def build_render_bundle(
    *,
    scenes: Sequence[SceneDocumentV1],
    timeline: SceneTimelineV1,
    profile: Mapping[str, Any],
    export_preset: str,
    localizations: Mapping[str, str],
    identity: Mapping[str, Any],
    repository_root: Path,
    decision_index: Sequence[tuple[str, str]] = (),
    source_metadata: Mapping[str, Any] | None = None,
) -> RenderBundleV1:
    """Resolve canonical scene data into one deterministic renderer input."""

    if export_preset not in _EXPORT_PRESETS:
        raise ValueError(f"unsupported render export preset: {export_preset}")
    if not scenes or not all(scene.verify() for scene in scenes):
        raise ValueError("all render scenes must be present and verified")
    if not timeline.verify():
        raise ValueError("render timeline must be verified")
    observers = {str(scene.value["observer"]) for scene in scenes}
    if len(observers) != 1:
        raise ValueError(
            "one render bundle may contain exactly one isolated observer perspective"
        )
    observer = next(iter(observers))
    render = profile.get("render")
    if not isinstance(render, Mapping):
        raise ValueError("video profile is missing render configuration")
    if render.get("locale") != "ar-SA" or render.get("direction") != "rtl":
        raise ValueError("release render bundles require Arabic ar-SA RTL presentation")
    exports = profile.get("exports")
    if not isinstance(exports, Mapping) or export_preset not in exports:
        raise ValueError(f"video profile does not govern export preset {export_preset}")
    render = _export_render(profile, export_preset)
    _validate_export_timeline(
        timeline,
        profile=profile,
        export_preset=export_preset,
        decision_kind=_scene_decision_kind(scenes),
        scenes=scenes,
    )
    strings = {str(key): str(value) for key, value in sorted(localizations.items())}
    if not strings or any(
        not key or not value.strip() for key, value in strings.items()
    ):
        raise ValueError("localized render strings must be non-empty")
    identity_value = _json_copy(identity)
    if set(identity_value) != {"gameSerial", "sceneId", "enrollmentUri", "players"}:
        raise ValueError(
            "render identity must contain gameSerial, sceneId, enrollmentUri, and players"
        )
    if not all(
        isinstance(identity_value[key], str) and identity_value[key].strip()
        for key in ("gameSerial", "sceneId", "enrollmentUri")
    ):
        raise ValueError("render identity locator values must be non-empty strings")
    if identity_value["sceneId"] not in {scene.scene_id for scene in scenes}:
        raise ValueError("render identity sceneId must reference a bundled scene")
    players = identity_value["players"]
    if not isinstance(players, Mapping) or set(players) != {
        "north",
        "west",
        "east",
        "south",
    }:
        raise ValueError("render identity requires one player name for every seat")
    if any(not isinstance(name, str) or not name.strip() for name in players.values()):
        raise ValueError("render player names must be non-empty strings")
    decisions = [
        {"label": str(label), "decisionId": str(decision_id)}
        for label, decision_id in decision_index
    ]
    if any(
        not item["label"].strip() or not item["decisionId"].strip()
        for item in decisions
    ):
        raise ValueError("render decision index values must be non-empty strings")

    card_assets = profile.get("card_assets")
    if not isinstance(card_assets, Mapping):
        raise ValueError("video profile is missing card asset configuration")
    manifest_path = str(card_assets.get("manifest", ""))
    assets = [
        _asset_entry(
            repository_root,
            manifest_path,
            asset_id=str(card_assets.get("pack_id", "cards")),
            license_id=str(card_assets.get("license_status", "unknown")),
        ),
        _asset_entry(
            repository_root,
            "assets/fonts/NotoSansArabic-Regular.ttf",
            asset_id="noto-sans-arabic-regular",
            license_id="OFL-1.1",
        ),
        _asset_entry(
            repository_root,
            "assets/fonts/NotoSansArabic-Bold.ttf",
            asset_id="noto-sans-arabic-bold",
            license_id="OFL-1.1",
        ),
        _asset_entry(
            repository_root,
            "assets/fonts/NotoKufiArabic-Bold.ttf",
            asset_id="noto-kufi-arabic-bold",
            license_id="OFL-1.1",
        ),
        _asset_entry(
            repository_root,
            "assets/video/professional/card-back.svg",
            asset_id="balootx-professional-card-back",
            license_id="BalootX-original",
        ),
    ]
    presentation = {
        key: _json_copy(profile[key]) for key in _PRESENTATION_KEYS if key in profile
    }
    if "majlis" in profile:
        presentation["majlis"] = resolve_majlis_presentation(profile["majlis"])
    if "audio" in presentation:
        audio = presentation["audio"]
        audio["explanation_voice_transcript"] = ""
        if (
            all(audio.get(key) == default for key, default in _RETIRED_VOICE_DEFAULTS.items())
            and all(type(audio.get(key)) is bool for key in _RETIRED_VOICE_FLAGS)
            and audio.get("one_piece_explanation") is False
            and audio.get("synthetic_callout_cues") is False
        ):
            # Older profiles can still build silent bundles after their inert preset is removed.
            for key in _RETIRED_VOICE_DEFAULTS:
                del audio[key]
            audio.update(dict.fromkeys(_RETIRED_VOICE_FLAGS, False))
    core = {
        "schemaVersion": 1,
        "exportPreset": export_preset,
        "locale": "ar-SA",
        "direction": "rtl",
        "observerSeat": observer,
        "renderer": {
            "plugin": "remotion_professional",
            "contractVersion": 1,
            "version": REMOTION_RENDERER_VERSION,
            "sourceSha256": renderer_source_sha256(repository_root),
        },
        "profile": _json_copy(profile.get("profile", {})),
        "render": _json_copy(render),
        "presentation": presentation,
        "localizations": strings,
        "identity": identity_value,
        "assets": sorted(assets, key=lambda item: item["id"]),
        "audioCues": [],
        "scenes": [scene.to_dict() for scene in scenes],
        "timeline": timeline.to_dict(),
        "decisionIndex": decisions,
    }
    if source_metadata is not None:
        core["source"] = normalize_render_source(
            source_metadata,
            observer=observer,
            require_current=True,
        )
        placeholders = majlis_placeholder_handles(profile.get("majlis") or {})
        if core["source"]["publicationEligible"] and placeholders:
            raise ValueError(
                f"publishable video still has placeholder social handles: {placeholders}"
            )
    bundle = RenderBundleV1({**core, "bundleSha256": content_hash(core)})
    if not bundle.verify(raise_voice_errors=True):
        raise AssertionError("constructed render bundle failed self-verification")
    return bundle
