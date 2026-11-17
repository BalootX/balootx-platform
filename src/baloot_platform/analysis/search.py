from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Callable

from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Team
from baloot_platform.core.models import PlayedCard, RoundState, Trick

from .models import SEARCH_COMPLETION_STATUSES
# Preserve the public import paths used by older callers.
from .search_errors import (
    SearchTimeBudgetExceeded as SearchTimeBudgetExceeded,
    require_uncensored_play_trace as require_uncensored_play_trace,
)


@dataclass(frozen=True, slots=True)
class SearchResult:
    value: float
    own_points: float
    opponent_points: float
    exact: bool
    nodes: int
    principal_variation: tuple[str, ...] = ()
    contract_succeeded: bool = False
    round_lead: bool = False
    score_after_round: tuple[int, int] = (0, 0)
    match_winner: Team | None = None
    completion_status: str = "failed"

    def __post_init__(self) -> None:
        if self.completion_status not in SEARCH_COMPLETION_STATUSES:
            raise ValueError(
                f"unsupported search completion status: {self.completion_status}"
            )
        if self.exact != (self.completion_status == "exact"):
            raise ValueError("exact flag and completion status disagree")

    @property
    def margin(self) -> float:
        """Compatibility alias for older analysis plugins."""
        return self.value


def terminal_outcome(state: RoundState, team: Team) -> tuple[float, float, float]:
    if state.round_score is None:
        raise ValueError("terminal state has no score")
    own = float(state.round_score.awarded_points[int(team)])
    opponent = float(state.round_score.awarded_points[int(team.other)])
    return own - opponent, own, opponent


def _terminal_result(state: RoundState, team: Team, *, exact: bool, nodes: int, line: tuple[str, ...]) -> SearchResult:
    value, own, opponent = terminal_outcome(state, team)
    if state.round_score is None or state.declarer is None:
        raise ValueError("terminal state is missing score/declarer")
    declarer_succeeded = state.round_score.winner == state.declarer.team
    # Contract success is reported from the observer team's perspective:
    # declarer team wants success; defending team wants failure.
    contract_succeeded_for_observer = declarer_succeeded if team == state.declarer.team else not declarer_succeeded
    return SearchResult(
        value,
        own,
        opponent,
        exact,
        nodes,
        line,
        contract_succeeded_for_observer,
        value > 0,
        (int(state.match_score[0]), int(state.match_score[1])),
        state.match_winner,
        "exact" if exact else "heuristic_complete",
    )


def terminal_margin(state: RoundState, team: Team) -> float:
    return terminal_outcome(state, team)[0]


def heuristic_rollout_result(state: RoundState, engine: BalootEngine, team: Team, seed: int) -> SearchResult:
    clone = state.clone()
    clone.events = []
    bot = HeuristicBot(engine, BotProfile(seed))
    line: list[str] = []
    while clone.current_player is not None:
        card = bot.choose(clone)
        line.append(card.code)
        engine.play_card(clone, card, emit_events=False)
        clone.events = []
    return _terminal_result(clone, team, exact=False, nodes=len(line), line=tuple(line))


def heuristic_rollout(state: RoundState, engine: BalootEngine, team: Team, seed: int) -> float:
    return heuristic_rollout_result(state, engine, team, seed).value


def exact_team_value(
    state: RoundState,
    engine: BalootEngine,
    team: Team,
    *,
    max_nodes: int = 250_000,
    max_seconds: float = 2.0,
    max_depth: int | None = None,
    max_cache_entries: int = 100_000,
    cancel_check: Callable[[], bool] | None = None,
) -> SearchResult:
    """Deterministic iterative-deepening alpha-beta endgame search.

    Node, depth, cache, and time limits are authoritative budgets. Deterministic
    node/depth exhaustion may return bounded partial evidence. Wall-clock
    exhaustion raises :class:`SearchTimeBudgetExceeded` because elapsed time
    cannot safely select a canonical partial result. Explicit cooperative
    cancellation remains a separate completion state.
    """
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    if max_depth is not None and max_depth <= 0:
        raise ValueError("max_depth must be positive when provided")
    if max_cache_entries < 0:
        raise ValueError("max_cache_entries cannot be negative")

    class _NodeBudgetExceeded(Exception):
        pass

    class _SearchCancelled(Exception):
        pass

    start = monotonic()
    nodes = 0
    target_depth = sum(mask.bit_count() for mask in state.hands)
    depth_budget = min(target_depth, max_depth or target_depth)
    # value, own, opponent, contract success, round lead, score after, match winner, PV
    cache: dict[
        tuple,
        tuple[float, float, float, bool, bool, tuple[int, int], Team | None, tuple[str, ...]],
    ] = {}
    preferred_moves: dict[tuple, int] = {}
    killer_moves: dict[int, list[int]] = {}
    history_scores: dict[tuple[bool, int], int] = {}

    def key(current: RoundState) -> tuple:
        trick = tuple(
            (play.seat, play.card.id, play.ekka, play.baloot)
            for play in (current.current_trick.plays if current.current_trick else [])
        )
        completed = tuple(
            (
                item.winner,
                item.raw_points,
                tuple(
                    (play.seat, play.card.id, play.ekka, play.baloot)
                    for play in item.plays
                ),
            )
            for item in current.completed_tricks
        )
        projects = tuple(
            (
                project.owner,
                project.kind,
                tuple(card.id for card in project.cards),
                project.declared,
                project.declaration_order,
                project.scoring,
                project.revealed,
            )
            for project in current.projects
        )
        return (
            tuple(current.hands),
            current.current_player,
            current.current_trick.index if current.current_trick else -1,
            trick,
            completed,
            projects,
            current.project_winning_team,
            tuple(sorted(int(value) for value in current.baloot_teams)),
            current.trump_broken,
            tuple(current.match_score),
        )

    def check_limits() -> None:
        nonlocal nodes
        if cancel_check is not None and cancel_check():
            raise _SearchCancelled
        if monotonic() - start > max_seconds:
            raise SearchTimeBudgetExceeded(
                "exact search exceeded its wall-clock budget without canonical output"
            )
        nodes += 1
        if nodes > max_nodes:
            raise _NodeBudgetExceeded

    def completes_winning_trick(current: RoundState, card) -> bool:
        trick = current.current_trick
        acting = current.current_player
        if trick is None or acting is None or len(trick.plays) != 3:
            return False
        candidate = Trick(
            trick.index,
            trick.leader,
            plays=[
                *trick.plays,
                PlayedCard(
                    acting,
                    card,
                    trick.index,
                    3,
                ),
            ],
        )
        return engine._trick_winner(candidate, current) == acting

    def rollout_in_place(
        current: RoundState,
        seed: int,
    ) -> tuple[
        float,
        float,
        float,
        bool,
        bool,
        tuple[int, int],
        Team | None,
        tuple[str, ...],
    ]:
        bot = HeuristicBot(engine, BotProfile(seed))
        tokens = []
        line: list[str] = []
        try:
            while current.current_player is not None:
                card = bot.choose(current)
                line.append(card.code)
                tokens.append(
                    engine.apply(
                        current,
                        card,
                        verify_identity=False,
                        emit_events=False,
                    )
                )
            terminal = _terminal_result(
                current,
                team,
                exact=False,
                nodes=nodes,
                line=tuple(line),
            )
            return (
                terminal.value,
                terminal.own_points,
                terminal.opponent_points,
                terminal.contract_succeeded,
                terminal.round_lead,
                terminal.score_after_round,
                terminal.match_winner,
                terminal.principal_variation,
            )
        finally:
            for token in reversed(tokens):
                engine.undo(current, token)

    def visit(
        current: RoundState,
        depth: int,
        alpha: float,
        beta: float,
    ) -> tuple[float, float, float, bool, bool, tuple[int, int], Team | None, tuple[str, ...]]:
        check_limits()
        if current.current_player is None:
            terminal = _terminal_result(current, team, exact=True, nodes=nodes, line=())
            return (
                terminal.value,
                terminal.own_points,
                terminal.opponent_points,
                terminal.contract_succeeded,
                terminal.round_lead,
                terminal.score_after_round,
                terminal.match_winner,
                (),
            )
        if depth == 0:
            return rollout_in_place(current, nodes + 17)

        position_key = key(current)
        cache_key = (depth, position_key)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        legal = list(engine.legal_actions(current))
        preferred = preferred_moves.get(position_key)
        maximizing = current.current_player.team == team
        killers = tuple(killer_moves.get(depth, ()))
        legal.sort(
            key=lambda card: (
                card.id != preferred,
                not completes_winning_trick(current, card),
                card.id not in killers,
                -history_scores.get((maximizing, card.id), 0),
                card.id,
            )
        )
        best: tuple[float, float, float, bool, bool, tuple[int, int], Team | None, tuple[str, ...]] | None = None
        best_card = None
        cutoff = False
        for card in legal:
            token = engine.apply(
                current,
                card,
                ekka=False,
                verify_identity=False,
                emit_events=False,
            )
            try:
                (
                    value,
                    own,
                    opponent,
                    contract_ok,
                    round_lead,
                    score_after,
                    winner,
                    line,
                ) = visit(current, depth - 1, alpha, beta)
            finally:
                engine.undo(current, token)
            candidate = (value, own, opponent, contract_ok, round_lead, score_after, winner, (card.code, *line))
            if best is None or (
                (value, -card.id) > (best[0], -best_card.id)
                if maximizing
                else (value, card.id) < (best[0], best_card.id)
            ):
                best = candidate
                best_card = card
            if maximizing:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if beta <= alpha:
                cutoff = True
                depth_killers = killer_moves.setdefault(depth, [])
                if card.id not in depth_killers:
                    depth_killers.insert(0, card.id)
                    del depth_killers[2:]
                history_key = (maximizing, card.id)
                history_scores[history_key] = (
                    history_scores.get(history_key, 0) + depth * depth
                )
                break
        assert best is not None
        assert best_card is not None
        preferred_moves[position_key] = best_card.id
        if (
            not cutoff
            and max_cache_entries > 0
            and len(cache) < max_cache_entries
        ):
            cache[cache_key] = best
        return best

    def with_status(
        result: SearchResult,
        status: str,
    ) -> SearchResult:
        return SearchResult(
            result.value,
            result.own_points,
            result.opponent_points,
            status == "exact",
            nodes,
            result.principal_variation,
            result.contract_succeeded,
            result.round_lead,
            result.score_after_round,
            result.match_winner,
            status,
        )

    try:
        fallback_values = rollout_in_place(state, 17)
    except MemoryError:
        return SearchResult(
            value=0.0,
            own_points=0.0,
            opponent_points=0.0,
            exact=False,
            nodes=nodes,
            principal_variation=(),
            contract_succeeded=False,
            round_lead=False,
            score_after_round=(
                int(state.match_score[0]),
                int(state.match_score[1]),
            ),
            match_winner=state.match_winner,
            completion_status="failed",
        )
    best_result = SearchResult(
        fallback_values[0],
        fallback_values[1],
        fallback_values[2],
        False,
        nodes,
        fallback_values[7],
        fallback_values[3],
        fallback_values[4],
        fallback_values[5],
        fallback_values[6],
        "budget_exhausted",
    )

    if target_depth == 0:
        terminal = _terminal_result(
            state,
            team,
            exact=True,
            nodes=nodes,
            line=(),
        )
        return with_status(terminal, "exact")

    try:
        for depth in range(1, depth_budget + 1):
            (
                value,
                own,
                opponent,
                contract_ok,
                round_lead,
                score_after,
                winner,
                line,
            ) = visit(state, depth, float("-inf"), float("inf"))
            status = "exact" if depth == target_depth else "budget_exhausted"
            best_result = SearchResult(
                value,
                own,
                opponent,
                status == "exact",
                nodes,
                line,
                contract_ok,
                round_lead,
                score_after,
                winner,
                status,
            )
        return best_result
    except _NodeBudgetExceeded:
        return with_status(best_result, "budget_exhausted")
    except _SearchCancelled:
        return with_status(best_result, "cancelled")
    except MemoryError:
        return with_status(best_result, "failed")
