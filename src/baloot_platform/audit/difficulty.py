from __future__ import annotations

import hashlib
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import yaml

from .models import action_key, utc_now
from baloot_platform.plugins.base import PluginDescriptor


class OnlineDifficultyCalibrator:
    descriptor = PluginDescriptor(
        "online_irt_difficulty_v1",
        "difficulty_calibrator",
        "1.0.0",
        "Replayable hybrid Elo/2PL challenge difficulty and player-ability calibrator",
        ("audience_calibration", "adaptive_delivery", "response_time", "confidence_weighting"),
        "cpu",
        deterministic=True,
    )

    """Replayable audience calibration layered over immutable engine difficulty.

    The engine's predicted score is never overwritten. Responses update a
    separate regularized 2PL/Elo-style latent item difficulty and per-player
    ability estimate. Replaying the stored response log produces the same
    result, which keeps calibration auditable and suitable for ML datasets.
    """

    def __init__(self, store: Any, config: dict[str, Any]):
        self.store = store
        self.config = config

    @classmethod
    def from_yaml(cls, store: Any, path: str | Path) -> "OnlineDifficultyCalibrator":
        return cls(store, yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def _player_key(self, player_id: str | None) -> str:
        privacy = self.config.get("privacy", {})
        value = (player_id or "anonymous").strip() or "anonymous"
        if bool(privacy.get("hash_external_player_ids", True)):
            return "PLY-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24].upper()
        return value

    @staticmethod
    def _logistic(value: float) -> float:
        value = max(-30.0, min(30.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def _ability_from_skill(skill: float | None) -> float:
        if skill is None:
            return 0.0
        bounded = max(0.01, min(0.99, float(skill) / 100.0))
        return math.log(bounded / (1.0 - bounded))

    def _label(self, score: float) -> str:
        labels = self.config.get("labels", {})
        if score <= float(labels.get("easy_max", 29)):
            return "easy"
        if score <= float(labels.get("medium_max", 54)):
            return "medium"
        if score <= float(labels.get("hard_max", 74)):
            return "hard"
        if score <= float(labels.get("expert_max", 89)):
            return "expert"
        return "extreme"

    def record_attempt(
        self,
        *,
        decision_id: str,
        player_id: str | None,
        chosen_action: Any,
        accepted_actions: Iterable[Any],
        engine_difficulty: float,
        response_seconds: float | None = None,
        confidence: float | None = None,
        reported_skill: float | None = None,
        changed_answer: bool = False,
        source: str = "audience",
    ) -> dict[str, Any]:
        player_key = self._player_key(player_id)
        accepted = list(accepted_actions)
        correct = any(action_key(chosen_action) == action_key(item) for item in accepted)
        if self.store.get_difficulty_item(decision_id) is None:
            self.store.upsert_difficulty_item(
                {
                    "challengeId": decision_id,
                    "engineDifficulty": float(engine_difficulty),
                    "latentDifficulty": (float(engine_difficulty) - 50.0) / 15.0,
                    "discrimination": float(self.config.get("model", {}).get("discrimination_initial", 1.0)),
                    "responseCount": 0,
                    "distinctPlayers": 0,
                    "standardError": 15.0,
                    "calibratedScore": float(engine_difficulty),
                    "label": self._label(float(engine_difficulty)),
                    "updatedAt": utc_now(),
                }
            )
        if self.store.get_difficulty_player(player_key) is None:
            self.store.upsert_difficulty_player(
                {
                    "playerId": player_key,
                    "ability": self._ability_from_skill(reported_skill),
                    "responseCount": 0,
                    "standardError": 1.0,
                    "updatedAt": utc_now(),
                }
            )
        self.store.add_difficulty_response(
            {
                "challengeId": decision_id,
                "playerId": player_key,
                "correct": correct,
                "responseSeconds": None if response_seconds is None else max(0.0, float(response_seconds)),
                "confidence": None if confidence is None else max(0.0, min(1.0, float(confidence))),
                "chosenAction": chosen_action,
                "trace": {
                    "source": source,
                    "reportedSkill": reported_skill,
                    "changedAnswer": bool(changed_answer),
                    "acceptedActions": accepted,
                },
            }
        )
        result = self.recalibrate(decision_id, engine_difficulty=engine_difficulty)
        return {**result, "accepted": correct, "playerId": player_key}

    def recalibrate(self, decision_id: str, *, engine_difficulty: float) -> dict[str, Any]:
        model = self.config.get("model", {})
        calibration = self.config.get("calibration", {})
        responses = self.store.list_difficulty_responses(decision_id, limit=100_000)
        difficulty = (max(0.0, min(100.0, float(engine_difficulty))) - 50.0) / 15.0
        discrimination = float(model.get("discrimination_initial", 1.0))
        disc_min = float(model.get("discrimination_min", 0.35))
        disc_max = float(model.get("discrimination_max", 3.0))
        lr_ability = float(model.get("learning_rate_ability", 0.035))
        lr_difficulty = float(model.get("learning_rate_difficulty", 0.025))
        lr_disc = float(model.get("learning_rate_discrimination", 0.004))
        shrinkage = float(model.get("l2_shrinkage", 0.001))
        time_weight = float(model.get("response_time_weight", 0.12))
        confidence_weight = float(model.get("confidence_weight", 0.15))
        min_seconds = float(model.get("minimum_seconds", 1.0))
        max_seconds = float(model.get("maximum_seconds", 180.0))

        abilities: dict[str, float] = {}
        for response in responses:
            player = str(response["playerId"])
            if player in abilities:
                continue
            # Calibration is rebuilt from the immutable response log.  Never
            # seed the next run from the previously fitted ability, otherwise
            # repeatedly reading the same data would drift the model.
            reported = (response.get("trace") or {}).get("reportedSkill")
            abilities[player] = self._ability_from_skill(reported)

        # Eight deterministic passes are intentionally small and bounded. The
        # stored event log remains the source of truth; no hidden online state is
        # needed to reproduce the result.
        for _ in range(8):
            for response in responses:
                player = str(response["playerId"])
                ability = abilities.get(player, 0.0)
                target_value = 1.0 if response["correct"] else 0.0
                seconds = response.get("responseSeconds")
                if seconds is not None:
                    seconds = max(min_seconds, min(max_seconds, float(seconds)))
                    target_seconds = max(min_seconds, min(max_seconds, 8.0 + engine_difficulty * 0.10))
                    slowness = max(-1.0, min(1.0, math.log(seconds / target_seconds)))
                    if target_value > 0.5:
                        target_value = max(0.65, target_value - time_weight * max(0.0, slowness))
                    else:
                        target_value = min(0.35, target_value + time_weight * max(0.0, -slowness))
                response_confidence = response.get("confidence")
                confidence_value = float(response_confidence) if response_confidence is not None else 0.5
                weight = 1.0 + confidence_weight * (confidence_value - 0.5) * 2.0
                probability = self._logistic(discrimination * (ability - difficulty))
                residual = (target_value - probability) * weight
                abilities[player] = max(
                    -5.0,
                    min(5.0, ability + lr_ability * discrimination * residual - shrinkage * ability),
                )
                difficulty = max(
                    -5.0,
                    min(5.0, difficulty - lr_difficulty * discrimination * residual - shrinkage * difficulty),
                )
                discrimination += lr_disc * residual * (ability - difficulty)
                discrimination = max(disc_min, min(disc_max, discrimination))

        for player, ability in abilities.items():
            all_player_responses = self.store.list_difficulty_responses(None, player_id=player, limit=100_000)
            total = len(all_player_responses)
            self.store.upsert_difficulty_player(
                {
                    "playerId": player,
                    "ability": ability,
                    "responseCount": total,
                    "standardError": 1.0 / math.sqrt(max(1.0, total * discrimination)),
                    "updatedAt": utc_now(),
                }
            )

        response_count = len(responses)
        distinct = len({response["playerId"] for response in responses})
        success = (
            sum(1.0 for response in responses if response["correct"]) / response_count
            if response_count
            else None
        )
        empirical = 100.0 * (1.0 - success) if success is not None else float(engine_difficulty)
        latent_score = max(0.0, min(100.0, 50.0 + 15.0 * difficulty))
        mature = max(1, int(calibration.get("mature_response_count", 250)))
        cold_weight = float(calibration.get("cold_start_engine_weight", 0.75))
        mature_weight = float(calibration.get("mature_engine_weight", 0.20))
        fraction = min(1.0, response_count / mature)
        engine_weight = cold_weight + (mature_weight - cold_weight) * fraction
        calibrated = max(
            0.0,
            min(
                100.0,
                engine_weight * engine_difficulty
                + (1.0 - engine_weight) * (0.70 * latent_score + 0.30 * empirical),
            ),
        )
        standard_error = 15.0 / math.sqrt(max(1.0, response_count * discrimination))
        times = [float(response["responseSeconds"]) for response in responses if response.get("responseSeconds") is not None]
        item = {
            "challengeId": decision_id,
            "engineDifficulty": round(float(engine_difficulty), 3),
            "latentDifficulty": round(difficulty, 6),
            "discrimination": round(discrimination, 6),
            "responseCount": response_count,
            "distinctPlayers": distinct,
            "standardError": round(standard_error, 4),
            "calibratedScore": round(calibrated, 3),
            "calibratedDifficulty": round(calibrated, 3),
            "label": self._label(calibrated),
            "observedSuccessRate": None if success is None else round(success, 6),
            "medianResponseSeconds": round(statistics.median(times), 3) if times else None,
            "public": response_count >= int(calibration.get("minimum_responses_for_public_label", 20))
            and distinct >= int(calibration.get("minimum_distinct_players", 8)),
            "updatedAt": utc_now(),
        }
        stored = self.store.upsert_difficulty_item(item)
        return {**stored, **item}

    def adaptive_next(self, player_id: str | None, *, recent_ids: set[str] | None = None) -> dict[str, Any] | None:
        player_key = self._player_key(player_id)
        player = self.store.get_difficulty_player(player_key) or {"ability": 0.0}
        ability = float(player.get("ability", 0.0))
        target = float(self.config.get("adaptive_delivery", {}).get("target_success_probability", 0.62))
        recent_ids = recent_ids or set()
        candidates: list[tuple[float, dict[str, Any]]] = []
        for item in self.store.list_difficulty_items(limit=10_000):
            item_id = str(item["challengeId"])
            if item_id in recent_ids:
                continue
            # Adaptive challenge delivery must not leak drafts, rejected items,
            # or raw replay decisions.  Difficulty rows are shared by replay
            # decisions and challenges, so eligibility is verified against the
            # immutable challenge registry at selection time.
            get_challenge = getattr(self.store, "get_challenge", None)
            challenge = get_challenge(item_id) if callable(get_challenge) else None
            if challenge is None or str(challenge.get("status", "")) != "published":
                continue
            difficulty = float(item.get("latentDifficulty", 0.0))
            discrimination = float(item.get("discrimination", 1.0))
            probability = self._logistic(discrimination * (ability - difficulty))
            candidates.append((abs(probability - target), {**item, "predictedSuccess": probability}))
        if not candidates:
            return None
        candidates.sort(key=lambda pair: (pair[0], str(pair[1]["challengeId"])))
        return candidates[0][1]


class DifficultyCalibrationService:
    """Compatibility facade used by API/MCP integrations."""

    def __init__(self, *, store: Any, profile: dict[str, Any]):
        self.store = store
        self.profile = profile
        self.calibrator = OnlineDifficultyCalibrator(store, profile)

    def record_response(
        self,
        challenge_id: str,
        *,
        player_id: str | None,
        chosen_action: Any,
        accepted_actions: Iterable[Any],
        engine_difficulty: float,
        response_seconds: float | None = None,
        confidence: float | None = None,
        reported_skill: float | None = None,
        changed_answer: bool = False,
    ) -> dict[str, Any]:
        return self.calibrator.record_attempt(
            decision_id=challenge_id,
            player_id=player_id,
            chosen_action=chosen_action,
            accepted_actions=accepted_actions,
            engine_difficulty=engine_difficulty,
            response_seconds=response_seconds,
            confidence=confidence,
            reported_skill=reported_skill,
            changed_answer=changed_answer,
        )

    def get(self, challenge_id: str, *, engine_difficulty: float) -> dict[str, Any]:
        existing = self.store.get_difficulty_item(challenge_id)
        return existing or self.calibrator.recalibrate(challenge_id, engine_difficulty=engine_difficulty)

    def adaptive_next(self, player_id: str | None, *, recent_ids: set[str] | None = None) -> dict[str, Any] | None:
        return self.calibrator.adaptive_next(player_id, recent_ids=recent_ids)
