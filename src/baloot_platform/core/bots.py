from __future__ import annotations

from dataclasses import dataclass

from .cards import Card
from .engine import BalootEngine
from .models import RoundState
from .shuffle import SplitMix64


@dataclass(frozen=True, slots=True)
class BotProfile:
    deterministic_seed: int


class HeuristicBot:
    """Deterministic, legal baseline bot.

    It is deliberately transparent: take a trick cheaply when possible,
    otherwise shed the least valuable/weakest legal card. Search and ML agents
    are separate plugins and can replace this policy without changing rules.
    """

    def __init__(self, engine: BalootEngine, profile: BotProfile):
        self.engine = engine
        self.profile = profile
        self.rng = SplitMix64(self.profile.deterministic_seed)

    def choose(self, state: RoundState) -> Card:
        legal = self.engine.legal_moves(state)
        if not legal:
            raise ValueError("no legal move")
        assert state.contract is not None
        if state.current_trick and state.current_trick.plays:
            winners: list[Card] = []
            for card in legal:
                token = None
                completed_before = len(state.completed_tricks)
                acting_seat = state.current_player
                try:
                    token = self.engine.apply(
                        state,
                        card,
                        ekka=False,
                        verify_identity=False,
                        emit_events=False,
                    )
                except Exception:
                    continue
                try:
                    if (
                        len(state.completed_tricks)
                        > completed_before
                        and state.completed_tricks[-1].winner
                        == acting_seat
                    ):
                        winners.append(card)
                finally:
                    self.engine.undo(state, token)
            if winners:
                return min(
                    winners,
                    key=lambda c: (
                        self.engine.rules.points(c, state.contract, state.trump_suit),
                        self.engine.rules.strength(c, state.contract, state.trump_suit),
                        c.id,
                    ),
                )
        return min(
            legal,
            key=lambda c: (
                self.engine.rules.points(c, state.contract, state.trump_suit),
                self.engine.rules.strength(c, state.contract, state.trump_suit),
                c.id,
            ),
        )
