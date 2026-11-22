from __future__ import annotations

from pathlib import Path


from baloot_platform.learning.service import LearningService
from baloot_platform.learning.store import LearningStore
from baloot_platform.sakka import SakkaService, SQLiteSakkaStore
from baloot_platform.service.game import GameManager
from baloot_platform.storage.game_store import SQLiteGameStore


class _Audit:
    def audit_decision(self, decision_id: str, *, deep: bool, force: bool):
        return {
            "decisionId": decision_id,
            "decisionKind": "play",
            "actualAction": {"code": "7S"},
            "bestAction": {"code": "8S"},
            "qualityLabel": "good",
            "qualityScore": 84.0,
            "forced": False,
            "expectedRegret": 2.0,
            "winProbabilityDrop": 0.03,
            "catastrophicCrossing": False,
            "reasons": [{"detailAr": "بديل محسوب أفضل بقليل."}],
        }


def _services(root: Path, tmp_path: Path):
    learning_store = LearningStore(tmp_path / "learning.sqlite3")
    manager = GameManager(
        root,
        SQLiteGameStore(tmp_path / "games.sqlite3"),
        learning_store=learning_store,
    )
    learning = LearningService(root, tmp_path / "learning", store=learning_store, game_manager=manager)
    manager.learning_service = learning
    service = SakkaService(
        root,
        SQLiteSakkaStore(tmp_path / "sakka.sqlite3"),
        game_manager=manager,
        audit_service=_Audit(),
    )
    return service, manager, learning_store
