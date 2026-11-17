from __future__ import annotations

import copy
import hashlib
import json
from enum import StrEnum
from typing import Any, Mapping

from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.config import ProfileBundle, build_bundle
from baloot_platform.core.enums import Seat
from baloot_platform.core.shuffle import shuffle_and_cut
from baloot_platform.version import __version__


class SharedScope(StrEnum):
    """Canonical Saudi Baloot product terminology.

    * LAFFA (لفة) is one complete deal/round.
    * SAKKA (صكّة) is one complete match, normally to 152 points.

    English and legacy values remain accepted at input boundaries only.
    """

    LAFFA = "laffa"
    SAKKA = "sakka"

    @classmethod
    def parse(cls, value: str | "SharedScope") -> "SharedScope":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        aliases = {
            "round": cls.LAFFA,
            "laffa": cls.LAFFA,
            "لفة": cls.LAFFA,
            "match": cls.SAKKA,
            "sakka": cls.SAKKA,
            "صكة": cls.SAKKA,
            "صكّة": cls.SAKKA,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError("shared scope must be laffa (لفة) or sakka (صكّة)") from exc

    @property
    def legacy(self) -> str:
        return "round" if self is SharedScope.LAFFA else "match"

    @property
    def label_ar(self) -> str:
        return "لفة واحدة" if self is SharedScope.LAFFA else "صكّة كاملة"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def environment_hash(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "environmentHash"}
    return _sha(payload)


def _deal_entry(*, index: int, seed: int, dealer: Seat, cut_enabled: bool,
                cut_position: int | None = None) -> dict[str, Any]:
    deck = shuffle_and_cut(seed, cut_position, dealer=dealer, cut_enabled=cut_enabled)
    return {
        "dealAttemptIndex": int(index),
        "seed": int(seed),
        "dealer": dealer.name.lower(),
        "cutter": deck.record.cutter.name.lower() if deck.record.cutter is not None else None,
        "cutPosition": deck.record.cut_position,
        "shuffleAlgorithm": deck.record.algorithm,
        "deckCommitment": deck.record.deck_commitment,
    }


def _normalize_source_schedule(session: Any) -> list[dict[str, Any]]:
    """Return the still-relevant authoritative source schedule.

    Branching a shared game must not silently generate a new future.  The
    current deal becomes relative attempt zero and every precommitted future
    attempt is preserved, reindexed without changing seed/dealer/cut/commitment.
    """

    source = dict(getattr(session, "shared_environment", {}) or {})
    check = verify_shared_environment(source) if source else {"valid": False}
    if not check.get("valid"):
        return []
    schedule = [dict(item) for item in source.get("dealSchedule", [])]
    current = max(0, int(getattr(session, "deal_schedule_index", 0) or 0))
    if current >= len(schedule):
        return []
    result: list[dict[str, Any]] = []
    for relative, item in enumerate(schedule[current:]):
        item["sourceDealAttemptIndex"] = int(item.get("dealAttemptIndex", current + relative))
        item["dealAttemptIndex"] = relative
        result.append(item)
    return result


def _component_snapshot(bundle: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "bundleHash": bundle.sha256,
        "mode": bundle.mode,
        "rules": {
            "id": bundle.rules.id,
            "version": bundle.rules.version,
            "sha256": bundle.rules.sha256,
            "status": bundle.rules.data.get("profile", {}).get("status"),
        },
    }
    for family, artifact in sorted(bundle.components.items()):
        if family == "rules":
            continue
        result[family] = {
            "id": artifact.id,
            "version": artifact.version,
            "sha256": artifact.sha256,
            "status": artifact.data.get("profile", {}).get("status"),
        }
    return result


def build_shared_environment(
    root: Any,
    session: Any,
    registry: AgentProfileRegistry,
    *,
    scope: str | SharedScope,
    maximum_rounds: int,
    lineup: Mapping[str, str] | None = None,
    model_snapshot: Mapping[str, Any] | None = None,
    maximum_deal_attempts: int | None = None,
    assistance_policy: str = "none_during_run",
    reveal_policy: str = "after_completion",
    time_control: Mapping[str, Any] | None = None,
    scoring_profile: str = "duplicate_skill_v1",
    bundle: ProfileBundle | None = None,
) -> dict[str, Any]:
    resolved_scope = SharedScope.parse(scope)
    bundle = bundle or build_bundle(root, mode=session.mode)
    shared_cfg = dict(bundle.component("shared"))
    schedule_cfg = dict(shared_cfg.get("schedule") or {})
    model_cfg = dict(shared_cfg.get("models") or {})
    determinism_cfg = dict(shared_cfg.get("determinism") or {})

    max_rounds = 1 if resolved_scope is SharedScope.LAFFA else max(1, int(maximum_rounds))
    minimum_attempts = max(1, int(schedule_cfg.get("minimum_deal_attempts", 16)))
    factor = max(1, int(schedule_cfg.get("deal_attempts_per_maximum_round", 8)))
    maximum_allowed = max(minimum_attempts, int(schedule_cfg.get("maximum_deal_attempts", 512)))
    requested_attempts = int(maximum_deal_attempts or max(minimum_attempts, max_rounds * factor))
    max_attempts = min(maximum_allowed, max(minimum_attempts, requested_attempts))

    preserved = _normalize_source_schedule(session) if bool(
        schedule_cfg.get("preserve_source_schedule_when_branching", True)
    ) else []
    cut_enabled = bool(bundle.rules.data.get("dealing", {}).get("cut_enabled", True))
    first_dealer = session.state.dealer
    first_seed = int(session.state.seed)

    schedule: list[dict[str, Any]] = preserved[:max_attempts]
    if not schedule:
        schedule = []
        for index in range(max_attempts):
            schedule.append(_deal_entry(
                index=index,
                seed=first_seed + index,
                dealer=Seat((int(first_dealer) + index) % 4),
                cut_enabled=cut_enabled,
            ))
        # The active deal is already authoritative.  Preserve its exact cut.
        if session.state.shuffle is not None:
            record = session.state.shuffle
            schedule[0] = {
                "dealAttemptIndex": 0,
                "sourceDealAttemptIndex": int(getattr(session, "deal_schedule_index", 0) or 0),
                "seed": int(record.seed),
                "dealer": record.dealer.name.lower(),
                "cutter": record.cutter.name.lower() if record.cutter is not None else None,
                "cutPosition": record.cut_position,
                "shuffleAlgorithm": record.algorithm,
                "deckCommitment": record.deck_commitment,
            }
    elif len(schedule) < max_attempts:
        last = schedule[-1]
        last_seed = int(last["seed"])
        last_dealer = Seat.parse(last["dealer"])
        for index in range(len(schedule), max_attempts):
            offset = index - len(schedule) + 1
            schedule.append(_deal_entry(
                index=index,
                seed=last_seed + offset,
                dealer=Seat((int(last_dealer) + offset) % 4),
                cut_enabled=cut_enabled,
            ))

    seed_policy = str(determinism_cfg.get("bot_seed_policy", "state_seed_version_actor_v1"))
    requested_model = dict(model_snapshot or {
        "mode": str(model_cfg.get("default_shared_mode", "disabled_for_shared_environment")),
        "modelId": None,
        "modelVersion": None,
        "artifactHash": None,
    })
    allowed_model_modes = set(model_cfg.get("allowed_modes") or ["disabled_for_shared_environment", "frozen_artifact"])
    if str(requested_model.get("mode")) not in allowed_model_modes:
        raise ValueError("unsupported model mode for shared environment")
    if str(requested_model.get("mode")) == "frozen_artifact" and bool(
        model_cfg.get("require_artifact_hash_when_enabled", True)
    ) and not requested_model.get("artifactHash"):
        raise ValueError("frozen shared model requires artifactHash")

    agent_snapshot = registry.snapshot_lineup(
        lineup or session.agent_profiles,
        model_snapshot=requested_model,
        seed_policy=seed_policy,
    )
    target = int(bundle.rules.data.get("scoring", {}).get("match", {}).get("target", 152))
    components = _component_snapshot(bundle)
    video_profile = dict(bundle.component("video") or {})
    visual_contract = {
        "videoProfileId": str(video_profile.get("profile", {}).get("id", "")),
        "videoProfileVersion": str(video_profile.get("profile", {}).get("version", "")),
        "videoProfileHash": bundle.components["video"].sha256,
        "rendererPlugin": str(video_profile.get("renderer_plugin", "")),
        "pacingProfile": str(video_profile.get("pacing_profile", "")),
        "cardAssets": copy.deepcopy(video_profile.get("card_assets") or {}),
        "presentation": copy.deepcopy(video_profile.get("presentation") or {}),
        "callouts": copy.deepcopy(video_profile.get("callouts") or {}),
        "audio": {
            key: value for key, value in dict(video_profile.get("audio") or {}).items()
            if key not in {"voice_pack_directory"}
        },
        "render": {
            key: value for key, value in dict(video_profile.get("render") or {}).items()
            if key in {"width", "height", "fps", "locale", "direction"}
        },
        "timelineProfile": {
            key: value for key, value in dict(video_profile.get("timeline") or {}).items()
            if key in {
                "bid_call_seconds", "bid_gap_seconds", "project_call_seconds",
                "project_reveal_seconds", "card_flight_seconds",
                "between_cards_seconds", "trick_hold_seconds",
                "trick_collect_seconds", "pause_seconds",
            }
        },
    }
    visual_contract["visualContractHash"] = _sha(visual_contract)
    payload: dict[str, Any] = {
        "schemaVersion": 2,
        "terminologyVersion": "ar_sa_baloot_v2",
        "terminology": {"matchAr": "صكّة", "roundAr": "لفة", "trickAr": "أكلة"},
        "scope": resolved_scope.value,
        "scopeLegacy": resolved_scope.legacy,
        "scopeLabelAr": resolved_scope.label_ar,
        "engineVersion": __version__,
        "configuration": components,
        "rulesBundleHash": session.bundle_hash,
        "rulesProfileId": bundle.rules.id,
        "rulesProfileVersion": bundle.rules.version,
        "sharedProfileHash": bundle.components["shared"].sha256,
        "targetMatchPoints": target,
        "startingMatchScore": list(session.state.match_score),
        "startingRoundNumber": int(session.state.round_number),
        "observerSeat": "south",  # overwritten and rehashed by board builder
        "maximumRounds": max_rounds,
        "maximumDealAttempts": len(schedule),
        "dealScheduleAlgorithm": str(schedule_cfg.get("algorithm", "precommitted_deal_attempt_stream_v2")),
        "dealSchedule": schedule,
        "dealScheduleHash": _sha(schedule),
        "allPassConsumesDealAttempt": bool(schedule_cfg.get("all_pass_consumes_attempt", True)),
        "completedRoundConsumesDealAttempt": bool(schedule_cfg.get("completed_round_consumes_attempt", True)),
        "dealerProgression": str(schedule_cfg.get("dealer_progression", "counter_clockwise")),
        "agentSnapshot": agent_snapshot,
        "agentSnapshotHash": agent_snapshot["lineupHash"],
        "botSeedPolicy": agent_snapshot["seedPolicy"],
        "modelSnapshot": dict(agent_snapshot.get("modelSnapshot") or {}),
        "modelSnapshotHash": _sha(dict(agent_snapshot.get("modelSnapshot") or {})),
        "scoringProfile": str(scoring_profile),
        "timeControl": dict(time_control or {"decisionSeconds": 45}),
        "assistancePolicy": str(assistance_policy),
        "revealPolicy": str(reveal_policy),
        "visualContract": visual_contract,
        "visualContractHash": visual_contract["visualContractHash"],
        "determinism": {
            "sameActionPathSameResult": bool(determinism_cfg.get("same_action_path_same_result", True)),
            "reproducibleBotActions": bool(determinism_cfg.get("require_reproducible_bot_actions", True)),
            "allParticipantsConsumeSameAttemptStream": True,
        },
    }
    payload["environmentHash"] = environment_hash(payload)
    return payload


def _verify_deal_entry(entry: Mapping[str, Any], *, index: int) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    try:
        dealer = Seat.parse(entry.get("dealer"))
        cut_position = entry.get("cutPosition")
        cut_enabled = entry.get("cutter") is not None or cut_position is not None
        deck = shuffle_and_cut(
            int(entry.get("seed")),
            int(cut_position) if cut_position is not None else None,
            dealer=dealer,
            cut_enabled=cut_enabled,
        )
        if deck.record.deck_commitment != entry.get("deckCommitment"):
            issues.append({"code": "deck_commitment_mismatch", "index": index})
        if deck.record.algorithm != entry.get("shuffleAlgorithm"):
            issues.append({"code": "shuffle_algorithm_mismatch", "index": index})
        expected_cutter = deck.record.cutter.name.lower() if deck.record.cutter is not None else None
        if expected_cutter != entry.get("cutter"):
            issues.append({"code": "cutter_mismatch", "index": index})
    except Exception as exc:
        issues.append({"code": "invalid_deal_entry", "index": index, "message": str(exc)})
    return issues


def verify_shared_environment(value: Mapping[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    expected = str(value.get("environmentHash") or "")
    actual = environment_hash(value)
    if expected != actual:
        issues.append({"code": "environment_hash_mismatch", "expected": expected, "actual": actual})
    try:
        scope = SharedScope.parse(str(value.get("scope")))
    except ValueError as exc:
        issues.append({"code": "invalid_scope", "message": str(exc)})
        scope = None

    schedule = [dict(item) for item in value.get("dealSchedule") or []]
    if not schedule:
        issues.append({"code": "missing_deal_schedule"})
    else:
        previous_dealer: Seat | None = None
        for index, entry in enumerate(schedule):
            if int(entry.get("dealAttemptIndex", -1)) != index:
                issues.append({"code": "deal_schedule_sequence", "index": index})
            issues.extend(_verify_deal_entry(entry, index=index))
            try:
                dealer = Seat.parse(entry.get("dealer"))
                if previous_dealer is not None and str(value.get("dealerProgression")) == "counter_clockwise":
                    if dealer != previous_dealer.next():
                        issues.append({"code": "dealer_progression_mismatch", "index": index})
                previous_dealer = dealer
            except Exception:
                pass
        if _sha(schedule) != value.get("dealScheduleHash"):
            issues.append({"code": "deal_schedule_hash_mismatch"})
        if int(value.get("maximumDealAttempts", -1)) != len(schedule):
            issues.append({"code": "maximum_deal_attempts_mismatch"})

    agent = dict(value.get("agentSnapshot") or {})
    if not agent:
        issues.append({"code": "missing_agent_snapshot"})
    else:
        expected_lineup = str(value.get("agentSnapshotHash") or "")
        inline_lineup = str(agent.get("lineupHash") or "")
        if inline_lineup != expected_lineup:
            issues.append({"code": "agent_snapshot_hash_mismatch"})
        payload_without_hash = {key: item for key, item in agent.items() if key != "lineupHash"}
        if _sha(payload_without_hash) != inline_lineup:
            issues.append({"code": "agent_snapshot_content_tampered"})
        try:
            AgentProfileRegistry.from_snapshot(agent)
        except Exception as exc:
            issues.append({"code": "agent_snapshot_invalid", "message": str(exc)})

    model = dict(value.get("modelSnapshot") or {})
    if _sha(model) != value.get("modelSnapshotHash"):
        issues.append({"code": "model_snapshot_hash_mismatch"})
    if str(model.get("mode")) == "frozen_artifact" and not model.get("artifactHash"):
        issues.append({"code": "frozen_model_artifact_hash_missing"})

    configuration = dict(value.get("configuration") or {})
    if configuration and configuration.get("bundleHash") != value.get("rulesBundleHash"):
        issues.append({"code": "configuration_bundle_hash_mismatch"})

    visual = dict(value.get("visualContract") or {})
    expected_visual_hash = str(value.get("visualContractHash") or "")
    inline_visual_hash = str(visual.get("visualContractHash") or "")
    visual_without_hash = {key: item for key, item in visual.items() if key != "visualContractHash"}
    actual_visual_hash = _sha(visual_without_hash) if visual_without_hash else ""
    if not visual:
        issues.append({"code": "missing_visual_contract"})
    elif expected_visual_hash != inline_visual_hash or inline_visual_hash != actual_visual_hash:
        issues.append({"code": "visual_contract_hash_mismatch"})
    elif bool((visual.get("cardAssets") or {}).get("freeze_in_shared_environment", False)) is not True:
        issues.append({"code": "visual_card_pack_not_frozen"})

    return {
        "valid": not issues,
        "scope": scope.value if scope is not None else None,
        "scopeLabelAr": scope.label_ar if scope is not None else None,
        "environmentHash": actual,
        "dealScheduleHash": _sha(schedule) if schedule else None,
        "agentSnapshotHash": value.get("agentSnapshotHash"),
        "modelSnapshotHash": value.get("modelSnapshotHash"),
        "visualContractHash": value.get("visualContractHash"),
        "issues": issues,
    }



def public_shared_environment(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a player-safe commitment view of a frozen environment.

    Seeds, cut positions and trusted model locations can reconstruct hidden or
    future information and are therefore never exposed during a shared run.
    Participants still receive enough hashes, agent tunings and commitments to
    prove that their standardized baselines are identical.
    """
    check = verify_shared_environment(value)
    if not check["valid"]:
        raise ValueError(f"invalid shared environment: {check['issues']}")
    schedule = []
    for item in value.get("dealSchedule") or []:
        schedule.append({
            "dealAttemptIndex": int(item["dealAttemptIndex"]),
            "sourceDealAttemptIndex": item.get("sourceDealAttemptIndex"),
            "dealer": item.get("dealer"),
            "cutter": item.get("cutter"),
            "shuffleAlgorithm": item.get("shuffleAlgorithm"),
            "deckCommitment": item.get("deckCommitment"),
        })
    model = dict(value.get("modelSnapshot") or {})
    public_model = {
        "mode": model.get("mode"),
        "modelId": model.get("modelId"),
        "modelVersion": model.get("modelVersion"),
        "artifactHash": model.get("artifactHash"),
    }
    return {
        "schemaVersion": value.get("schemaVersion"),
        "terminologyVersion": value.get("terminologyVersion"),
        "terminology": dict(value.get("terminology") or {}),
        "scope": value.get("scope"),
        "scopeLegacy": value.get("scopeLegacy"),
        "scopeLabelAr": value.get("scopeLabelAr"),
        "engineVersion": value.get("engineVersion"),
        "configuration": copy.deepcopy(value.get("configuration") or {}),
        "rulesBundleHash": value.get("rulesBundleHash"),
        "rulesProfileId": value.get("rulesProfileId"),
        "rulesProfileVersion": value.get("rulesProfileVersion"),
        "sharedProfileHash": value.get("sharedProfileHash"),
        "targetMatchPoints": value.get("targetMatchPoints"),
        "startingMatchScore": list(value.get("startingMatchScore") or []),
        "startingRoundNumber": value.get("startingRoundNumber"),
        "observerSeat": value.get("observerSeat"),
        "maximumRounds": value.get("maximumRounds"),
        "maximumDealAttempts": value.get("maximumDealAttempts"),
        "dealScheduleAlgorithm": value.get("dealScheduleAlgorithm"),
        "dealSchedule": schedule,
        "dealScheduleHash": value.get("dealScheduleHash"),
        "allPassConsumesDealAttempt": value.get("allPassConsumesDealAttempt"),
        "completedRoundConsumesDealAttempt": value.get("completedRoundConsumesDealAttempt"),
        "dealerProgression": value.get("dealerProgression"),
        "agentSnapshot": copy.deepcopy(value.get("agentSnapshot") or {}),
        "agentSnapshotHash": value.get("agentSnapshotHash"),
        "botSeedPolicy": value.get("botSeedPolicy"),
        "modelSnapshot": public_model,
        "modelSnapshotHash": value.get("modelSnapshotHash"),
        "scoringProfile": value.get("scoringProfile"),
        "timeControl": copy.deepcopy(value.get("timeControl") or {}),
        "assistancePolicy": value.get("assistancePolicy"),
        "revealPolicy": value.get("revealPolicy"),
        "visualContract": copy.deepcopy(value.get("visualContract") or {}),
        "visualContractHash": value.get("visualContractHash"),
        "determinism": copy.deepcopy(value.get("determinism") or {}),
        "environmentHash": value.get("environmentHash"),
        "redaction": {
            "playerSafe": True,
            "dealSeedsHidden": True,
            "cutPositionsHidden": True,
            "futureCardsHidden": True,
        },
    }

def current_deal_entry(environment: Mapping[str, Any], current_index: int) -> dict[str, Any]:
    check = verify_shared_environment(environment)
    if not check["valid"]:
        raise ValueError(f"invalid shared environment: {check['issues']}")
    schedule = list(environment["dealSchedule"])
    index = int(current_index)
    if index < 0 or index >= len(schedule):
        raise RuntimeError("shared deal schedule index is outside the published stream")
    return copy.deepcopy(schedule[index])


def next_deal_entry(environment: Mapping[str, Any], current_index: int) -> dict[str, Any]:
    check = verify_shared_environment(environment)
    if not check["valid"]:
        raise ValueError(f"invalid shared environment: {check['issues']}")
    schedule = list(environment["dealSchedule"])
    index = int(current_index) + 1
    if index >= len(schedule):
        raise RuntimeError("shared deal schedule exhausted; publish a board with a larger schedule")
    return copy.deepcopy(schedule[index])
