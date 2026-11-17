from __future__ import annotations

from typing import Any

from baloot_platform.plugins.base import PluginDescriptor


class HistoryStoreAdapter:
    descriptor = PluginDescriptor(
        "immutable_history_store_v1",
        "history_store",
        "1.0.0",
        "Immutable game, decision, analysis, challenge, dataset, model, and artifact history adapter",
        ("hash_chain", "reopen", "branch", "reanalyze", "lineage"),
        resource_class="storage",
    )

    def __init__(self, store: Any) -> None:
        self.store = store

    def archive_game(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.store.archive_game(*args, **kwargs)

    def get_game(self, identifier: str) -> dict[str, Any] | None:
        return self.store.get_game(identifier)

    def verify_game(self, identifier: str) -> dict[str, Any]:
        return self.store.verify_game(identifier)

    def list_games(self, *, query: str = "", offset: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list_games(query=query, offset=offset, limit=limit)


class LearningBackendAdapter:
    descriptor = PluginDescriptor(
        "continuous_learning_backend_v1",
        "learning_backend",
        "1.0.0",
        "Observer-safe feedback, grouped datasets, deterministic training, evaluation, and promotion adapter",
        ("feedback", "active_learning", "grouped_split", "policy_value", "promotion_gates"),
        resource_class="cpu",
    )

    def __init__(self, service: Any) -> None:
        self.service = service

    def reanalyze_decision(self, decision_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.service.reanalyze_decision(decision_id, **kwargs)

    def add_feedback(self, decision_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.service.add_feedback(decision_id, **kwargs)

    def build_dataset(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.build_dataset(**kwargs)

    def train_model(self, dataset_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.service.train_model(dataset_id, **kwargs)


class GameRepositoryAdapter:
    descriptor = PluginDescriptor(
        "versioned_game_repository_v1",
        "game_repository",
        "1.0.0",
        "Immutable hash-chained authoritative game-session repository adapter",
        ("versions", "branch", "audit", "optimistic_concurrency"),
        resource_class="storage",
    )

    def __init__(self, store: Any) -> None:
        self.store = store

    def save(self, session: Any) -> Any:
        return self.store.save(session)

    def load(self, game_id: str) -> Any | None:
        return self.store.load(game_id)

    def load_version(self, game_id: str, version: int) -> Any | None:
        return self.store.load_version(game_id, version)

    def verify(self, game_id: str) -> dict[str, Any]:
        return self.store.verify(game_id)
