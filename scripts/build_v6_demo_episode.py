from __future__ import annotations

import json
import pickle
import shutil
import time
from pathlib import Path

from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.audit.service import ReplayAuditService
from baloot_platform.core.enums import Seat
from baloot_platform.learning.service import LearningService
from baloot_platform.learning.store import LearningStore
from baloot_platform.service.game import GameManager
from baloot_platform.storage.game_store import SQLiteGameStore


def mark(label: str, started: float) -> None:
    print(f"[v6-build] {label}: {time.monotonic()-started:.2f}s", flush=True)


def main() -> None:
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    runtime = root / "release" / "v6-demo-runtime"
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True)

    store = LearningStore(runtime / "learning.sqlite3")
    registry = AgentProfileRegistry.from_yaml(root / "configs/agents/adaptive_lineup_v1.yaml")
    manager = GameManager(root, SQLiteGameStore(runtime / "games.sqlite3"), learning_store=store, agent_registry=registry)
    learning = LearningService(root, runtime / "learning", store=store, game_manager=manager)
    manager.learning_service = learning
    audit = ReplayAuditService(root, store, learning)

    game = manager.create(seed=501, dealer=Seat.EAST, agent_profiles={seat: "beginner" for seat in ("north", "west", "south", "east")})
    game = manager.advance_bots(game.id, stop_before=None, maximum_actions=128)
    mark("game-completed", started)
    if game.state.round_score is None or not game.game_hash:
        raise RuntimeError("demo game did not complete")

    candidates = [
        item for item in store.list_decisions(game.id, limit=1000)
        if item.get("decisionKind") == "play" and len(item.get("legalActions") or []) > 1
    ]
    if not candidates:
        raise RuntimeError("demo game has no meaningful play decisions")

    screened = []
    for item in candidates:
        screened.append(audit.audit_decision(
            item["decisionId"], deep=False, force=True,
            overrides={
                "requested_samples": 4,
                "exact_allocation_limit": 120,
                "exact_search_remaining": 4,
                "maximum_sampling_attempts": 2500,
                "seed": 7,
            },
        ))
    mark("screen-audits-completed", started)
    selected = max(screened, key=lambda value: (
        bool(value.get("catastrophicCrossing")),
        float(value.get("winProbabilityDrop", 0.0)),
        float(value.get("expectedRegret", 0.0)),
        -int(value.get("ordinal", 0)),
    ))
    deep = audit.audit_decision(
        selected["decisionId"], deep=True, force=True,
        overrides={
            "requested_samples": 64,
            "exact_allocation_limit": 20_000,
            "exact_search_remaining": 12,
            "maximum_sampling_attempts": 80_000,
            "seed": 11,
            "use_conventions": False,
        },
    )
    mark("deep-audit-completed", started)
    episode = learning.create_historical_replay_episode(
        game.game_hash,
        decision_id=selected["decisionId"],
        analysis_id=deep.get("analysisId"),
    )
    payload = {
        "schemaVersion": 1,
        "gameId": game.id,
        "gameHash": game.game_hash,
        "gameSerial": game.serial,
        "decisionId": selected["decisionId"],
        "auditId": deep.get("auditId"),
        "analysisId": deep.get("analysisId"),
        "qualityLabel": deep.get("qualityLabel"),
        "qualityScore": deep.get("qualityScore"),
        "expectedRegret": deep.get("expectedRegret"),
        "winProbabilityDrop": deep.get("winProbabilityDrop"),
        "actualIntent": deep.get("actualIntent"),
    }
    (runtime / "episode.pkl").write_bytes(pickle.dumps(episode, protocol=pickle.HIGHEST_PROTOCOL))
    (runtime / "audit-summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    mark("episode-persisted", started)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
