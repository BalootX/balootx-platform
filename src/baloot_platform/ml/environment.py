from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from baloot_platform.analysis.models import InformationState
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, GamePhase, Seat, Suit
from baloot_platform.core.models import RoundState
from baloot_platform.core.rules import RuleBook

from .actions import ACTION_COUNT, ActionCodec, ActionKind
from .features import encode_information_state


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: np.ndarray | None
    legal_mask: tuple[int, ...]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class BalootEnvironment:
    """Deterministic, cloneable ML environment backed by the authoritative engine.

    It supports an automated auction reset and full card play. Interactive
    auction states can also be driven through `BiddingMachine`; the stable
    action codec is shared with future OpenSpiel/PettingZoo adapters.
    """

    def __init__(self, rules: RuleBook, *, observer: Seat = Seat.SOUTH):
        self.rules = rules
        self.engine = BalootEngine(rules)
        self.observer = observer
        self.state: RoundState | None = None

    def reset(
        self,
        *,
        seed: int = 1,
        dealer: Seat = Seat.EAST,
        contract: Contract | None = None,
        buyer: Seat | None = None,
        trump_suit: Suit | None = None,
    ) -> StepResult:
        state = self.engine.create_round(seed=seed, dealer=dealer)
        self.engine.shuffle_and_initial_deal(state)
        if not self.engine.bid_and_complete_deal(
            state,
            forced_contract=contract,
            forced_buyer=buyer,
            forced_trump=trump_suit,
        ):
            return StepResult(None, (0,) * ACTION_COUNT, 0.0, True, False, {"allPassed": True})
        self.engine.prepare_projects(state)
        self.state = state
        return self._result(reward=0.0)

    def clone(self) -> "BalootEnvironment":
        clone = BalootEnvironment(self.rules, observer=self.observer)
        clone.state = self.state.clone() if self.state is not None else None
        return clone

    def legal_mask(self) -> tuple[int, ...]:
        if self.state is None or self.state.current_player is None:
            return (0,) * ACTION_COUNT
        legal = self.engine.legal_moves(self.state)
        ekka_cards = tuple(
            card for card in legal
            if self.state.current_trick is not None
            and not self.state.current_trick.plays
            and self.engine._can_declare_ekka(self.state, self.state.current_player, card)
        )
        return ActionCodec.legal_play_mask(legal, ekka_cards=ekka_cards)

    def step(self, action_id: int) -> StepResult:
        if self.state is None or self.state.current_player is None:
            raise RuntimeError("environment is terminal or not reset")
        mask = self.legal_mask()
        if not 0 <= int(action_id) < ACTION_COUNT or not mask[int(action_id)]:
            raise ValueError(f"illegal action {action_id}")
        action = ActionCodec.decode(int(action_id))
        if action.kind != ActionKind.PLAY or action.card is None:
            raise ValueError("the active environment phase requires a play action")
        before = tuple(self.state.match_score)
        actor_team = self.state.current_player.team
        self.engine.play_card(self.state, action.card, ekka=action.ekka)
        reward = 0.0
        if self.state.round_score is not None:
            after = self.state.match_score
            reward = float(after[int(actor_team)] - before[int(actor_team)] - (after[int(actor_team.other)] - before[int(actor_team.other)]))
        return self._result(reward=reward)

    def _result(self, *, reward: float) -> StepResult:
        assert self.state is not None
        terminated = self.state.current_player is None
        observation = None
        if not terminated:
            observation = encode_information_state(InformationState.from_complete(self.state, self.state.current_player))
        return StepResult(
            observation,
            self.legal_mask(),
            reward,
            terminated,
            False,
            {
                "stateHash": self.state.rules_hash,
                "phase": self.state.phase.value,
                "currentPlayer": self.state.current_player.name.lower() if self.state.current_player is not None else None,
                "matchScore": list(self.state.match_score),
            },
        )
