"""Observer-safe semantic scene and timeline contracts for media.

This module is the authority for what a renderer may see. It accepts an
already observer-filtered engine projection, public events, approved analysis
facts, and resolved presentation settings.  It deliberately has no API that
accepts a trusted ``RoundState``.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping, Sequence

from baloot_platform.audit.models import content_hash

_DISCLOSURES = frozenset({"public", "observer_private", "post_reveal_analysis"})
_SEATS = frozenset({"north", "west", "south", "east"})
_PROMPT_CONTRACTS = frozenset(
    {
        "choose_acting_action",
        "predict_opponent_play",
        "read_partner_signal",
        "identify_threat",
        "plan_next_response",
        "infer_distribution",
        "post_reveal_analysis",
    }
)
_RENDITION_TYPES = frozenset(
    {
        "actor_challenge",
        "partner_inference",
        "opponent_prediction",
        "seat_narrative",
        "post_reveal_analysis",
    }
)
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


def _json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _walk_leaves(value: Any, path: str = "") -> set[str]:
    if isinstance(value, Mapping):
        result: set[str] = set()
        for key, item in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            result.update(_walk_leaves(item, f"{path}/{escaped}"))
        return result
    if isinstance(value, list):
        result = set()
        for index, item in enumerate(value):
            result.update(_walk_leaves(item, f"{path}/{index}"))
        return result
    return {path or "/"}


def _reject_forbidden_keys(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_KEYS:
                raise ValueError(f"observer-safe scene input contains forbidden field {path}/{key}")
            _reject_forbidden_keys(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_keys(item, f"{path}/{index}")


def _validate_observer_view(view: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    safe = _json_copy(view)
    _reject_forbidden_keys(safe)
    observer = str(safe.get("observer", ""))
    hands = safe.get("hands")
    if not observer or not isinstance(hands, Mapping) or observer not in hands:
        raise ValueError("observer view must identify the observer and hands")
    for seat, value in hands.items():
        if seat == observer:
            if not isinstance(value, list):
                raise ValueError("observer hand must be the only revealed hand")
        elif not (
            isinstance(value, Mapping)
            and set(value) == {"count"}
            and isinstance(value["count"], int)
            and 0 <= value["count"] <= 8
        ):
            raise ValueError(f"non-observer hand {seat!r} must contain only a count")
    return observer, safe


def _layer(layer_id: str, kind: str, disclosure: str, content: Mapping[str, Any]) -> dict[str, Any]:
    if disclosure not in _DISCLOSURES:
        raise ValueError(f"unsupported disclosure classification: {disclosure}")
    return {
        "id": layer_id,
        "kind": kind,
        "disclosure": disclosure,
        "content": _json_copy(content),
    }


def _context_lifecycle_facts(
    view: Mapping[str, Any], events: Sequence[Mapping[str, Any]],
    observer_lifecycle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Only the decision projection can authorize pre-decision replay facts.
    observer = str(view["observer"])
    tricks = list(view.get("completedTricks") or ())
    plays = [play for trick in tricks for play in trick.get("plays", ())]
    plays.extend(dict(view.get("currentTrick") or {}).get("plays") or ())
    event_plays = [event for event in events if event.get("type") == "card_played"
                   and isinstance(event.get("card"), Mapping)]
    if len(event_plays) > len(plays):
        plays = [{key: value for key, value in event.items() if key != "type"}
                 for event in event_plays]
    public = {
        key: view[key] for key in (
            "roundNumber", "phase", "dealer", "buyer", "declarer", "contract",
            "trumpSuit", "firstLeader", "matchScore", "projects",
            "projectWinningTeam", "balootTeams", "escalation", "locked",
        ) if key in view
    }
    public.update(
        pauseBeforePlayIndex=len(plays), plays=plays, completedTricks=tricks,
        preDecisionMatchScore=view.get("matchScore", [0, 0]),
        bidHistory=view.get("biddingHistory", []),
        escalationHistory=view.get("escalationHistory", []),
    )
    purchase = next((event for event in reversed(events)
                     if event.get("type") == "contract_purchased"), {})
    for key in ("purchaseRecipient", "purchaseCard"):
        if key in purchase:
            public[key] = purchase[key]
        elif key in view:
            public[key] = view[key]
    own_cards = {card["id"]: card for card in view["hands"][observer]}
    own_cards.update({play["card"]["id"]: play["card"] for play in plays
                      if play.get("seat") == observer})
    private = {
        "seat": observer,
        "initialHandCards": [own_cards[key] for key in sorted(own_cards)],
        **({"initialFiveCards": [own_cards[card["id"]] for card in observer_lifecycle["initialFiveCards"]
                                 if card["id"] in own_cards]}
           if "initialFiveCards" in observer_lifecycle else {}),
    }
    return public, private


@dataclass(frozen=True, slots=True)
class SceneDocumentV1:
    """One canonical semantic scene; pixels and layout remain renderer concerns."""

    value: dict[str, Any]

    @property
    def scene_id(self) -> str:
        return str(self.value["sceneId"])

    def to_dict(self) -> dict[str, Any]:
        return _json_copy(self.value)

    def verify(self) -> bool:
        value = self.to_dict()
        scene_id = value.pop("sceneId")
        disclosures = value.pop("fieldDisclosures")
        if scene_id != content_hash(value):
            return False
        try:
            _reject_forbidden_keys(value)
            layers = {layer["id"]: layer for layer in value["layers"]}
            if len(layers) != len(value["layers"]):
                return False
            for layer_id, context_disclosure in (("public-lifecycle", "public"),
                                                  ("observer-lifecycle", "observer_private")):
                if layer_id in layers:
                    if value["stage"] not in {"context", "continuation"}:
                        return False
                    expected = "post_reveal_analysis" if value["stage"] == "continuation" else context_disclosure
                    if layers[layer_id]["disclosure"] != expected:
                        return False
            if value["stage"] in {"context", "question", "selection"}:
                if any(layer["disclosure"] == "post_reveal_analysis" for layer in layers.values()):
                    return False
                table = layers["table"]["content"]
                if table.get("roundScore") is not None or table.get("matchWinner") is not None:
                    return False
                events = layers["public-events"]["content"]["items"]
                if any(event.get("type") in {"round_scored", "match_completed"} for event in events):
                    return False
                for event in events:
                    if event.get("type") == "trick_completed":
                        trick = next((item for index, item in enumerate(table.get("completedTricks") or ())
                                      if item.get("index", index) == event.get("trickIndex")), None)
                        if trick is None or any(key in event and trick.get(key) != event[key]
                                                for key in ("winner", "rawPoints")):
                            return False
                    if event.get("type") == "contract_purchased":
                        if table.get("buyer") is None or any(
                            key in event and table.get(key) != event[key]
                            for key in ("buyer", "declarer", "contract", "trumpSuit", "purchaseCard")
                        ):
                            return False
                    if table.get("phase") in {"created", "shuffled", "initial_dealt", "bidding"} and event.get("type") in {
                        "final_deal_complete", "project_declared", "projects_compared", "projects_revealed",
                        "baloot_declared", "play_started", "card_played", "trick_completed",
                    }:
                        return False
                table_plays = [play for trick in table.get("completedTricks") or ()
                               for play in trick.get("plays", ())]
                table_plays.extend(dict(table.get("currentTrick") or {}).get("plays") or ())
                event_plays = [event for event in events if event.get("type") == "card_played"]
                if event_plays and [(event.get("seat"), (event.get("card") or {}).get("id", event.get("cardId")))
                                    for event in event_plays] != [
                                        (play.get("seat"), play["card"]["id"]) for play in table_plays]:
                    return False
                if "public-lifecycle" in layers or "observer-lifecycle" in layers:
                    if value["stage"] != "context":
                        return False
                    observer = value["observer"]
                    view = {**table, "observer": observer,
                            "hands": {observer: layers["observer-hand"]["content"]["cards"]}}
                    public, private = _context_lifecycle_facts(
                        view, layers["public-events"]["content"]["items"],
                        layers.get("observer-lifecycle", {}).get("content", {}),
                    )
                    if layers.get("public-lifecycle", {}).get("content") != public:
                        return False
                    if layers.get("observer-lifecycle", {}).get("content") != private:
                        return False
        except (KeyError, TypeError, ValueError):
            return False
        leaves = _walk_leaves(value)
        return (
            set(disclosures) == leaves
            and all(item in _DISCLOSURES for item in disclosures.values())
            and all(disclosures[path] == value["layers"][int(path.split("/")[2])]["disclosure"]
                    for path in leaves if path.startswith("/layers/"))
        )


def reduce_scene_document(
    *,
    observer_view: Mapping[str, Any],
    public_events: Sequence[Mapping[str, Any]],
    effective_config_sha256: str,
    stage: str,
    locale: str,
    direction: str,
    prompt_key: str | None = None,
    legal_choices: Sequence[Mapping[str, Any]] = (),
    acting_seat: str | None = None,
    prompt_contract: str | None = None,
    rendition_type: str | None = None,
    analysis_facts: Mapping[str, Any] | None = None,
    public_lifecycle_facts: Mapping[str, Any] | None = None,
    observer_lifecycle_facts: Mapping[str, Any] | None = None,
) -> SceneDocumentV1:
    """Reduce safe engine facts into one deterministic semantic scene.

    Pre-reveal stages never include analysis facts.  Text is represented by
    semantic localization keys, keeping copy out of behavioral configuration.
    """

    if len(effective_config_sha256) != 64:
        raise ValueError("effective_config_sha256 must be a SHA-256 hex digest")
    if stage not in {"context", "question", "selection", "reveal", "explanation", "continuation", "outcome"}:
        raise ValueError(f"unsupported scene stage: {stage}")
    if locale != "ar-SA" or direction != "rtl":
        raise ValueError("release scenes require Arabic ar-SA RTL presentation")
    observer, safe_view = _validate_observer_view(observer_view)
    safe_events = _json_copy(list(public_events))
    _reject_forbidden_keys(safe_events)
    safe_choices = _json_copy(list(legal_choices))
    _reject_forbidden_keys(safe_choices)
    prompt_metadata = (acting_seat, prompt_contract, rendition_type)
    if prompt_key is None:
        if safe_choices or any(item is not None for item in prompt_metadata):
            raise ValueError("prompt choices and semantics require a prompt key")
    else:
        if acting_seat not in _SEATS:
            raise ValueError("decision prompt requires a valid acting seat")
        if prompt_contract not in _PROMPT_CONTRACTS:
            raise ValueError("decision prompt contract is unsupported")
        if rendition_type not in _RENDITION_TYPES:
            raise ValueError("decision rendition type is unsupported")
        is_actor_prompt = prompt_contract == "choose_acting_action"
        if is_actor_prompt != (rendition_type == "actor_challenge"):
            raise ValueError("actor prompt and rendition semantics must agree")
        if is_actor_prompt and observer != acting_seat:
            raise ValueError("acting-action prompts require the acting observer")
        if not is_actor_prompt and observer == acting_seat:
            raise ValueError("seat-experience prompts require a non-acting observer")
        if not is_actor_prompt and safe_choices:
            raise ValueError("non-acting observer prompts cannot expose acting-player choices")
    if analysis_facts is not None and stage not in {"reveal", "explanation", "outcome"}:
        raise ValueError("analysis facts are forbidden before reveal")
    safe_analysis = _json_copy(analysis_facts or {})
    _reject_forbidden_keys(safe_analysis)
    safe_public_lifecycle = _json_copy(public_lifecycle_facts or {})
    safe_observer_lifecycle = _json_copy(observer_lifecycle_facts or {})
    _reject_forbidden_keys(safe_public_lifecycle)
    _reject_forbidden_keys(safe_observer_lifecycle)
    if (
        safe_public_lifecycle or safe_observer_lifecycle
    ) and stage not in {"context", "continuation"}:
        raise ValueError("lifecycle facts require context or post-reveal continuation")
    if stage == "context" and (safe_public_lifecycle or safe_observer_lifecycle):
        safe_public_lifecycle, safe_observer_lifecycle = _context_lifecycle_facts(
            safe_view, safe_events, safe_observer_lifecycle,
        )

    hands = safe_view["hands"]
    table = {
        key: safe_view.get(key)
        for key in (
            "roundNumber",
            "phase",
            "dealer",
            "buyer",
            "declarer",
            "contract",
            "trumpSuit",
            "purchaseCard",
            "bidding",
            "biddingHistory",
            "escalationHistory",
            "escalation",
            "locked",
            "firstLeader",
            "currentPlayer",
            "matchScore",
            "currentTrick",
            "completedTricks",
            "projects",
            "projectWinningTeam",
            "balootTeams",
            "roundScore",
            "matchWinner",
        )
        if key in safe_view
    }
    layers = [
        _layer("table", "table", "public", table),
        _layer(
            "seats",
            "seats",
            "public",
            {
                "handCounts": {
                    seat: len(value) if isinstance(value, list) else value["count"]
                    for seat, value in sorted(hands.items())
                }
            },
        ),
        _layer(
            "observer-hand",
            "cards",
            "observer_private",
            {"seat": observer, "cards": hands[observer]},
        ),
        _layer("public-events", "events", "public", {"items": safe_events}),
    ]
    if prompt_key is not None:
        layers.append(
            _layer(
                "decision-prompt",
                "prompt",
                "observer_private",
                {
                    "textKey": prompt_key,
                    "legalChoices": safe_choices,
                    "actingSeat": acting_seat,
                    "promptContract": prompt_contract,
                    "renditionType": rendition_type,
                },
            )
        )
    if safe_analysis:
        layers.append(
            _layer(
                "analysis",
                "analysis",
                "post_reveal_analysis",
                safe_analysis,
            )
        )
    if safe_public_lifecycle:
        layers.append(
            _layer(
                "public-lifecycle",
                "lifecycle",
                "post_reveal_analysis" if stage == "continuation" else "public",
                safe_public_lifecycle,
            )
        )
    if safe_observer_lifecycle:
        layers.append(
            _layer(
                "observer-lifecycle",
                "lifecycle",
                "post_reveal_analysis" if stage == "continuation" else "observer_private",
                safe_observer_lifecycle,
            )
        )

    core = {
        "schemaVersion": 1,
        "stage": stage,
        "observer": observer,
        "locale": locale,
        "direction": direction,
        "source": {
            "observerViewSha256": content_hash(safe_view),
            "publicEventLineageSha256": content_hash(safe_events),
            "effectiveConfigSha256": effective_config_sha256,
        },
        "layers": layers,
    }
    field_disclosures: dict[str, str] = {}
    for path in sorted(_walk_leaves(core)):
        if path.startswith("/layers/"):
            parts = path.split("/")
            layer = layers[int(parts[2])]
            field_disclosures[path] = str(layer["disclosure"])
        else:
            field_disclosures[path] = "public"
    value = {
        **core,
        "fieldDisclosures": field_disclosures,
        "sceneId": content_hash(core),
    }
    document = SceneDocumentV1(value)
    if not document.verify():
        raise AssertionError("constructed scene document failed self-verification")
    return document


@dataclass(frozen=True, slots=True)
class SceneTimelineV1:
    value: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _json_copy(self.value)

    def verify(self) -> bool:
        value = self.to_dict()
        timeline_sha256 = value.pop("timelineSha256")
        return timeline_sha256 == content_hash(value)


def build_scene_timeline(
    *,
    fps: int,
    width: int,
    height: int,
    segments: Sequence[tuple[str, str, Decimal | float | int | str]],
) -> SceneTimelineV1:
    """Build a frame-authoritative timeline with rational time positions."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    if width <= 0 or height <= 0 or width * 16 != height * 9:
        raise ValueError("release media must use an exact 9:16 aspect ratio")
    frame = 0
    result: list[dict[str, Any]] = []
    for stage, scene_id, seconds in segments:
        duration = Decimal(str(seconds))
        duration_frames = int((duration * fps).to_integral_value(rounding=ROUND_HALF_UP))
        if duration_frames <= 0:
            raise ValueError(f"timeline segment {stage!r} must have positive duration")
        result.append(
            {
                "stage": stage,
                "sceneId": scene_id,
                "startFrame": frame,
                "durationFrames": duration_frames,
                "startTime": {"numerator": frame, "denominator": fps},
                "durationTime": {"numerator": duration_frames, "denominator": fps},
            }
        )
        frame += duration_frames
    core = {
        "schemaVersion": 1,
        "frameRate": {"numerator": fps, "denominator": 1},
        "canvas": {"width": width, "height": height, "aspectRatio": "9:16"},
        "segments": result,
        "totalFrames": frame,
    }
    timeline = SceneTimelineV1({**core, "timelineSha256": content_hash(core)})
    if not timeline.verify():
        raise AssertionError("constructed scene timeline failed self-verification")
    return timeline


def create_signed_deep_link_payload(
    *,
    scene_id: str,
    decision_id: str,
    observer: str,
    signing_key: bytes,
) -> str:
    """Create an integrity-protected locator containing no game-state payload."""

    if len(signing_key) < 32:
        raise ValueError("deep-link signing key must contain at least 32 bytes")
    payload = {
        "schemaVersion": 1,
        "sceneId": scene_id,
        "decisionId": decision_id,
        "observer": observer,
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(signing_key, body, hashlib.sha256).digest()
    return ".".join(
        (
            base64.urlsafe_b64encode(body).rstrip(b"=").decode("ascii"),
            base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii"),
        )
    )


def verify_signed_deep_link_payload(token: str, *, signing_key: bytes) -> dict[str, Any]:
    try:
        body_part, signature_part = token.split(".", 1)
        body = base64.urlsafe_b64decode(body_part + "=" * (-len(body_part) % 4))
        signature = base64.urlsafe_b64decode(
            signature_part + "=" * (-len(signature_part) % 4)
        )
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid deep-link token") from exc
    expected = hmac.new(signing_key, body, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("deep-link signature mismatch")
    value = json.loads(body)
    if set(value) != {"schemaVersion", "sceneId", "decisionId", "observer"}:
        raise ValueError("deep-link payload contains unsupported fields")
    return value
