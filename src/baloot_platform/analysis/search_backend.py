"""Governed exact-search backends.

The Python implementation remains the semantic reference.  The native V3
backend is promoted only for exact results; every partial or unsupported case
falls back to the reference path so acceleration cannot weaken truth. A
wall-clock exhaustion propagates without a result so elapsed time can never
shape canonical search evidence.
"""
from __future__ import annotations

from typing import Callable, Protocol, Sequence

from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Team
from baloot_platform.core.cards import Card
from baloot_platform.core.models import RoundState
from baloot_platform.native.accelerator import NativeAccelerator, NativeUnavailable
from baloot_platform.plugins.base import PluginDescriptor

from .search import (
    SearchResult,
    exact_team_value,
    heuristic_rollout_result,
)


class ExactSearchBackend(Protocol):
    descriptor: PluginDescriptor

    def search(
        self,
        state: RoundState,
        engine: BalootEngine,
        team: Team,
        *,
        max_nodes: int,
        max_seconds: float,
        max_depth: int | None,
        max_cache_entries: int,
        cancel_check: Callable[[], bool] | None,
    ) -> SearchResult: ...


class PythonReferenceSearchBackend:
    descriptor = PluginDescriptor(
        id="python_reference_exact_search_v1",
        kind="search",
        version="1.1.0",
        description=(
            "Authoritative deterministic Python endgame search with fail-closed "
            "wall-clock exhaustion"
        ),
        capabilities=(
            "exact_search",
            "reference",
            "cancellation",
            "fail_closed_wall_clock",
        ),
        resource_class="cpu",
        deterministic=True,
    )

    search = staticmethod(exact_team_value)


class NativeExactSearchBackendV3:
    descriptor = PluginDescriptor(
        id="native_exact_v3_python_fallback_v1",
        kind="search",
        version="1.1.0",
        description=(
            "Coarse C++20 exact search with fail-closed Python reference and "
            "wall-clock fallback"
        ),
        capabilities=(
            "exact_search",
            "native_bitboards",
            "bounded_transposition_table",
            "cancellation",
            "reference_fallback",
            "fail_closed_wall_clock",
        ),
        resource_class="cpu",
        deterministic=True,
    )

    def __init__(self, native: NativeAccelerator):
        if not native.supports_v3:
            raise NativeUnavailable("native exact-search ABI V3 is unavailable")
        self.native = native
        self.supports_compact_rollout = bool(native.supports_compact_rollout)

    @staticmethod
    def _reference(
        state: RoundState,
        engine: BalootEngine,
        team: Team,
        **budgets: object,
    ) -> SearchResult:
        return exact_team_value(state, engine, team, **budgets)  # type: ignore[arg-type]

    def search(
        self,
        state: RoundState,
        engine: BalootEngine,
        team: Team,
        *,
        max_nodes: int,
        max_seconds: float,
        max_depth: int | None,
        max_cache_entries: int,
        cancel_check: Callable[[], bool] | None,
    ) -> SearchResult:
        budgets = {
            "max_nodes": max_nodes,
            "max_seconds": max_seconds,
            "max_depth": max_depth,
            "max_cache_entries": max_cache_entries,
            "cancel_check": cancel_check,
        }
        try:
            accelerated = self.native.exact_team_value(
                state,
                engine,
                team,
                **budgets,
            )
        except TimeoutError:
            raise
        except (NativeUnavailable, ValueError, OSError):
            return self._reference(state, engine, team, **budgets)
        if not accelerated.exact:
            return self._reference(state, engine, team, **budgets)
        return SearchResult(
            value=accelerated.value,
            own_points=accelerated.own_points,
            opponent_points=accelerated.opponent_points,
            exact=True,
            nodes=accelerated.nodes,
            principal_variation=accelerated.principal_variation,
            contract_succeeded=accelerated.contract_succeeded,
            round_lead=accelerated.round_lead,
            score_after_round=accelerated.score_after_round,
            match_winner=accelerated.match_winner,
            completion_status="exact",
        )

    def heuristic_rollout_batch(
        self,
        states: Sequence[RoundState],
        cards: Sequence[Card | None],
        seeds: Sequence[int],
        engine: BalootEngine,
        team: Team,
        *,
        cancel_check: Callable[[], bool] | None,
    ) -> tuple[SearchResult, ...]:
        if not (len(states) == len(cards) == len(seeds)):
            raise ValueError("batched rollout inputs must align")
        try:
            outcomes = self.native.heuristic_rollout_batch(
                states,
                cards,
                team,
                cancel_check=cancel_check,
            )
            return tuple(
                SearchResult(
                    value=item.value,
                    own_points=item.own_points,
                    opponent_points=item.opponent_points,
                    exact=False,
                    nodes=item.nodes,
                    principal_variation=item.principal_variation,
                    contract_succeeded=item.contract_succeeded,
                    round_lead=item.round_lead,
                    score_after_round=item.score_after_round,
                    match_winner=item.match_winner,
                    completion_status=item.completion_status,
                )
                for item in outcomes
            )
        except TimeoutError:
            raise
        except (NativeUnavailable, ValueError, OSError):
            fallback: list[SearchResult] = []
            for state, card, seed in zip(states, cards, seeds, strict=True):
                if cancel_check is not None and cancel_check():
                    raise InterruptedError("analysis cancelled")
                child = state.clone()
                child.events = []
                if card is not None:
                    engine.play_card(child, card, ekka=False, emit_events=False)
                fallback.append(
                    heuristic_rollout_result(child, engine, team, int(seed))
                )
            return tuple(fallback)

    def heuristic_rollout_batch_compact(
        self,
        states: Sequence[RoundState],
        cards: Sequence[Card | None],
        seeds: Sequence[int],
        engine: BalootEngine,
        team: Team,
        *,
        cancel_check: Callable[[], bool] | None,
    ) -> tuple[SearchResult, ...]:
        """Return numeric rollout evidence without retaining per-task PVs."""
        if not (len(states) == len(cards) == len(seeds)):
            raise ValueError("batched rollout inputs must align")
        try:
            outcomes = self.native.heuristic_rollout_batch_compact(
                states,
                cards,
                team,
                cancel_check=cancel_check,
            )
            return tuple(
                SearchResult(
                    value=item.value,
                    own_points=item.own_points,
                    opponent_points=item.opponent_points,
                    exact=False,
                    nodes=item.nodes,
                    principal_variation=(),
                    contract_succeeded=item.contract_succeeded,
                    round_lead=item.round_lead,
                    score_after_round=item.score_after_round,
                    match_winner=item.match_winner,
                    completion_status=item.completion_status,
                )
                for item in outcomes
            )
        except TimeoutError:
            raise
        except (NativeUnavailable, ValueError, OSError):
            return tuple(
                SearchResult(
                    value=item.value,
                    own_points=item.own_points,
                    opponent_points=item.opponent_points,
                    exact=item.exact,
                    nodes=item.nodes,
                    principal_variation=(),
                    contract_succeeded=item.contract_succeeded,
                    round_lead=item.round_lead,
                    score_after_round=item.score_after_round,
                    match_winner=item.match_winner,
                    completion_status=item.completion_status,
                )
                for item in self.heuristic_rollout_batch(
                    states,
                    cards,
                    seeds,
                    engine,
                    team,
                    cancel_check=cancel_check,
                )
            )

    heuristic_rollout_batch_details = heuristic_rollout_batch
