from __future__ import annotations

from pathlib import Path
from typing import Any

from baloot_platform.analysis.evaluator import analyze
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.plugins.base import PluginDescriptor


class BalancedPlayStrategy:
    descriptor = PluginDescriptor(
        "balanced_heuristic_v1", "play_strategy", "1.0.0",
        "Deterministic legal Baloot baseline used for rollouts and bot seats",
        ("sun", "hokum", "ashkal", "projects", "full_round"),
    )

    def choose_card(self, engine: Any, state: Any, legal_cards: tuple[Any, ...], **kwargs: Any) -> Any:
        bot = HeuristicBot(engine, BotProfile(int(kwargs.get("seed", state.seed))))
        return bot.choose(state)


class LegalHiddenDealsAnalysis:
    descriptor = PluginDescriptor(
        "legal_hidden_deals_v1", "analysis", "1.0.0",
        "Exact-or-sampled observer-safe legal hidden-deal analysis",
        ("hard_deductions", "exact_enumeration", "confidence_intervals", "convention_toggle"),
    )

    def __init__(self, rules: Any, convention: dict[str, Any]):
        self.rules = rules
        self.convention = convention

    def analyze(self, information_state: Any, **kwargs: Any) -> Any:
        return analyze(information_state, self.rules, self.convention, **kwargs)


from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore


class LocalContentAddressedStore(LocalContentAddressedArtifactStore):
    """Backward-compatible alias for the V2 local artifact repository."""

