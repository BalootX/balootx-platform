from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import MethodType, SimpleNamespace

from baloot_platform.core.models import Contract, Seat, Suit
from baloot_platform.research.ortools_nonplay_graphs import AuctionGraphTerminalV1
from baloot_platform.research.ortools_nonplay_policy_continuation import (
    OrToolsNonPlayPolicyContinuationV1,
)
from baloot_platform.research.scenario_contracts import canonical_sha256


def test_parallel_auction_roots_cache_their_own_local_continuation() -> None:
    """A slower root must not cache a concurrently appended root's evidence."""

    first_appended = Event()
    second_cached = Event()

    class NotifyingCache(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if key[1] == Contract.HOKUM.value:
                second_cached.set()

    owner = object.__new__(OrToolsNonPlayPolicyContinuationV1)
    owner.prefix = SimpleNamespace(auction_actions=())
    owner._continuations = []
    owner._auction_cache = NotifyingCache()

    def select_escalation(self, **_kwargs):
        return (SimpleNamespace(actor=Seat.WEST),), {}

    def resolve_escalation(self, **_kwargs):
        return SimpleNamespace(level="double", locked=False)

    def round_utility(self, **kwargs):
        root_action = kwargs["root_action"]
        exact_trace = {"auctionActions": [dict(item) for item in kwargs["auction_actions"]]}
        core = {
            "rootAction": root_action,
            "stageActions": list(kwargs["stage_actions"]),
            "exactTracePrefix": exact_trace,
            "modelHash": canonical_sha256({"model": root_action}),
            "utility": 1,
        }
        continuation = {**core, "sha256": canonical_sha256(core)}
        if root_action == Contract.SUN.value:
            self._continuations.append(continuation)
            first_appended.set()
            assert second_cached.wait(5)
        else:
            assert first_appended.wait(5)
            self._continuations.append(continuation)
        return 1, continuation

    owner._select_escalation_continuation = MethodType(select_escalation, owner)
    owner._resolved_escalation_state = MethodType(resolve_escalation, owner)
    owner._round_utility = MethodType(round_utility, owner)

    def terminal(action: str, contract: Contract, trump: Suit | None = None):
        return AuctionGraphTerminalV1(
            actions=(SimpleNamespace(actor=Seat.EAST, action=action, trump=trump),),
            buyer=Seat.EAST,
            declarer=Seat.EAST,
            contract=contract,
            trump=trump,
            purchase_round=1,
            all_passed=False,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        sun = pool.submit(
            owner.auction_utility,
            terminal(Contract.SUN.value, Contract.SUN),
        )
        hokum = pool.submit(
            owner.auction_utility,
            terminal(Contract.HOKUM.value, Contract.HOKUM, Suit.DIAMONDS),
        )
        assert sun.result() == hokum.result() == 1

    sun_cached = owner._auction_cache[("east", "sun", None, 1)]
    hokum_cached = owner._auction_cache[("east", "hokum", "diamonds", 1)]
    assert sun_cached["modelHash"] != hokum_cached["modelHash"]
    assert sun_cached["exactTracePrefix"]["auctionActions"][-1] == {
        "actor": "east",
        "action": "sun",
    }
    assert hokum_cached["exactTracePrefix"]["auctionActions"][-1] == {
        "actor": "east",
        "action": "hokum",
        "trumpSuit": "diamonds",
    }
    continuation_hashes = {
        item["rootAction"]: item["sha256"] for item in owner._continuations
    }
    assert continuation_hashes["sun"] != continuation_hashes["hokum:diamonds"]
