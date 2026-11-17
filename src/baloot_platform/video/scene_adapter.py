"""Bounded trusted adapter from legacy EpisodeSpec to scene contracts.

Only this compatibility adapter may inspect the legacy trusted episode.  Its
output is observer-safe and can be handed to renderers, previews, screenshots,
or media QA without exposing the complete game state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.scenario import EpisodeSpec

from .scene import (
    SceneDocumentV1,
    SceneTimelineV1,
    build_scene_timeline,
    reduce_scene_document,
)
from .render_projection import (
    build_observer_safe_render_episode,
    lifecycle_scene_facts,
)


@dataclass(frozen=True, slots=True)
class DecisionScenePackageV1:
    scenes: tuple[SceneDocumentV1, ...]
    timeline: SceneTimelineV1

    def scene(self, stage: str) -> SceneDocumentV1:
        return next(item for item in self.scenes if item.value["stage"] == stage)


def _events_through_observer_state(view: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Bound a legacy event log to the public state visible at the decision."""

    events = view.get("events")
    if not isinstance(events, list) or view.get("phase") != "playing":
        return tuple(dict(item) for item in events or ())
    completed = view.get("completedTricks")
    current = view.get("currentTrick")
    completed_count = len(completed) if isinstance(completed, list) else 0
    current_plays = current.get("plays") if isinstance(current, Mapping) else ()
    current_count = len(current_plays) if isinstance(current_plays, list) else 0
    bounded: list[dict[str, Any]] = []
    for item in events:
        if not isinstance(item, Mapping):
            continue
        event_type = str(item.get("type", ""))
        trick_index = item.get("trickIndex")
        if event_type == "card_played" and isinstance(trick_index, int):
            position = item.get("position")
            if trick_index > completed_count or (
                trick_index == completed_count
                and (not isinstance(position, int) or position >= current_count)
            ):
                continue
        elif event_type == "trick_completed" and isinstance(trick_index, int):
            if trick_index >= completed_count:
                continue
        elif event_type in {"round_scored", "match_completed"}:
            continue
        bounded.append(dict(item))
    return tuple(bounded)


def observer_safe_episode_payload(
    episode: EpisodeSpec,
    package: DecisionScenePackageV1,
) -> dict[str, Any]:
    """Build the minimal public episode locator shipped beside media."""

    question = package.scene("question")
    return {
        "schemaVersion": 21,
        "episodeId": episode.episode_id,
        "bundleHash": episode.bundle_hash,
        "viewer": episode.viewer.name.lower(),
        "locale": "ar-SA",
        "direction": "rtl",
        "titleAr": episode.title_ar,
        "questionAr": episode.question_ar,
        "playbackMode": episode.playback_mode,
        "provenance": {
            "gameSerial": episode.game_serial,
            "gameRecordHash": episode.game_record_hash,
            "challengeId": episode.challenge_id,
            "analysisId": episode.analysis_id,
            "modelId": episode.model_id,
            "auditId": episode.audit_id,
            "decisionId": episode.decision_id,
        },
        "semanticScenes": {
            "questionSceneId": question.scene_id,
            "timelineSha256": package.timeline.value["timelineSha256"],
            "sceneIds": {
                str(scene.value["stage"]): scene.scene_id
                for scene in package.scenes
            },
        },
        "observerSafe": True,
        "proofBoundary": (
            "locator and public provenance only; trusted replay state, seed, "
            "deck order, hidden hands, future cards, and analysis are excluded"
        ),
    }


def _analysis_facts(episode: EpisodeSpec) -> dict[str, Any]:
    analysis = episode.analysis
    facts = {
        "recommendedAction": {
            "kind": "play_card",
            "card": analysis.recommended_card.to_dict(),
        },
        "actionEvaluations": [
            {
                "card": item.card.to_dict(),
                "expectedRoundMargin": item.expected_round_margin,
                "expectedTeamAwardedPoints": item.expected_team_awarded_points,
                "expectedOpponentAwardedPoints": item.expected_opponent_awarded_points,
                "roundLeadProbability": item.round_lead_probability,
                "drawProbability": item.draw_probability,
                "roundLeadConfidence95": [
                    item.confidence_low,
                    item.confidence_high,
                ],
                "bestShare": item.best_share,
                "contractSuccessProbability": item.contract_success_probability,
                "matchWinProbability": item.match_win_probability,
                "confidence95": [
                    item.match_confidence_low,
                    item.match_confidence_high,
                ],
                "samples": item.samples,
                "effectiveSamples": item.effective_samples,
                "exactShare": item.exact_share,
                "searchCompletionStatus": item.search_completion_status,
            }
            for item in analysis.actions
        ],
        "method": analysis.method,
        "objective": analysis.objective,
        "difficulty": analysis.difficulty,
        "debateScore": analysis.debate_score,
        "explanation": list(analysis.explanation),
        "reasonKey": "video.analysis.expert_reason",
        "analysisProfile": {
            "id": analysis.analysis_profile_id,
            "version": analysis.analysis_profile_version,
        },
    }
    summary = episode.audit_summary or {}
    shared = summary.get("sharedComparison") or summary.get("sakkaComparison")
    if isinstance(shared, Mapping) and shared.get("sameStartVerified"):
        facts["sharedComparison"] = {
            key: shared.get(key)
            for key in (
                "schemaVersion",
                "comparisonHash",
                "boardSerial",
                "sameStartVerified",
                "sharedScope",
                "playerA",
                "playerB",
                "simpleArabicExplanation",
            )
        }
    return facts


def build_legacy_episode_scene_package(
    episode: EpisodeSpec,
    profile: Mapping[str, Any],
) -> DecisionScenePackageV1:
    """Project a legacy episode into canonical observer-safe documents."""

    render = profile["render"]
    timeline_profile = profile["timeline"]
    before = BalootEngine.observer_view(
        episode.state_before_pause,
        episode.viewer,
    )
    after = BalootEngine.observer_view(episode.final_state, episode.viewer)
    legal_choices = tuple(
        {"kind": "play_card", "card": item.card.to_dict()}
        for item in episode.analysis.actions
    )
    facts = _analysis_facts(episode)
    before_events = _events_through_observer_state(before)

    common = {
        "effective_config_sha256": episode.bundle_hash,
        "locale": str(render["locale"]),
        "direction": str(render["direction"]),
    }
    question = reduce_scene_document(
        observer_view=before,
        public_events=before_events,
        stage="question",
        prompt_key="video.decision.choose_best_move",
        legal_choices=legal_choices,
        acting_seat=episode.viewer.name.lower(),
        prompt_contract="choose_acting_action",
        rendition_type="actor_challenge",
        **common,
    )
    reveal = reduce_scene_document(
        observer_view=before,
        public_events=before_events,
        stage="reveal",
        prompt_key="video.decision.best_move",
        legal_choices=legal_choices,
        acting_seat=episode.viewer.name.lower(),
        prompt_contract="choose_acting_action",
        rendition_type="actor_challenge",
        analysis_facts=facts,
        **common,
    )
    explanation = reduce_scene_document(
        observer_view=before,
        public_events=before_events,
        stage="explanation",
        analysis_facts=facts,
        **common,
    )
    projection = build_observer_safe_render_episode(episode, reveal)
    public_lifecycle, observer_lifecycle = lifecycle_scene_facts(projection)
    continuation = reduce_scene_document(
        observer_view=before,
        public_events=before_events,
        stage="continuation",
        public_lifecycle_facts=public_lifecycle,
        observer_lifecycle_facts=observer_lifecycle,
        **common,
    )
    outcome = reduce_scene_document(
        observer_view=after,
        public_events=tuple(after["events"]),
        stage="outcome",
        analysis_facts=facts,
        **common,
    )
    decision_scenes = (
        question,
        reveal,
        explanation,
        continuation,
        outcome,
    )
    context = reduce_scene_document(
        observer_view=before,
        public_events=before_events,
        stage="context",
        public_lifecycle_facts=public_lifecycle,
        observer_lifecycle_facts=observer_lifecycle,
        **common,
    )
    scenes = (context, *decision_scenes)
    scene_by_stage = {
        str(item.value["stage"]): item.scene_id for item in scenes
    }
    timeline = build_scene_timeline(
        fps=int(render["fps"]),
        width=int(render["width"]),
        height=int(render["height"]),
        segments=(
            ("question", scene_by_stage["question"], timeline_profile["pause_seconds"]),
            ("reveal", scene_by_stage["reveal"], timeline_profile["reveal_seconds"]),
            (
                "explanation",
                scene_by_stage["explanation"],
                timeline_profile["explanation_seconds"],
            ),
            (
                "continuation",
                scene_by_stage["continuation"],
                timeline_profile["continuation_seconds"],
            ),
            ("outcome", scene_by_stage["outcome"], timeline_profile["final_seconds"]),
        ),
    )
    return DecisionScenePackageV1(scenes=scenes, timeline=timeline)
