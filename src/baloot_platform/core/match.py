from __future__ import annotations

from dataclasses import dataclass, field

from .bots import BotProfile, HeuristicBot
from .engine import BalootEngine
from .enums import Contract, Seat, Suit, Team
from .models import RoundState


@dataclass(slots=True)
class MatchResult:
    rounds: list[RoundState] = field(default_factory=list)
    final_score: list[int] = field(default_factory=lambda: [0, 0])
    winner: Team | None = None

    def to_dict(self) -> dict:
        return {
            "rounds": [r.to_dict(reveal_hands=True) for r in self.rounds],
            "finalScore": list(self.final_score),
            "winner": self.winner.name.lower() if self.winner is not None else None,
        }


def simulate_match(
    engine: BalootEngine,
    *,
    seed: int,
    dealer: Seat = Seat.EAST,
    maximum_rounds: int = 64,
    forced_contract: Contract | None = None,
    forced_trump: Suit | None = None,
) -> MatchResult:
    score = [0, 0]
    result = MatchResult(final_score=score)
    target = engine.scoring_domain.match_target
    for index in range(maximum_rounds):
        state = engine.create_round(seed=seed + index, dealer=dealer, round_number=index + 1, match_score=tuple(score))
        engine.shuffle_and_initial_deal(state)
        if not engine.bid_and_complete_deal(state, forced_contract=forced_contract, forced_trump=forced_trump):
            dealer = engine.next_round_dealer(state)
            result.rounds.append(state)
            continue
        engine.prepare_projects(state)
        bot = HeuristicBot(engine, BotProfile(seed + index + 999))
        while state.current_player is not None:
            engine.play_card(state, bot.choose(state))
        score = list(state.match_score)
        result.final_score = score
        result.rounds.append(state)
        if state.match_winner is not None:
            result.winner = state.match_winner
            break
        if max(score) >= target and score[0] != score[1]:
            result.winner = Team.NORTH_SOUTH if score[0] > score[1] else Team.WEST_EAST
            break
        dealer = engine.next_round_dealer(state)
    return result
