"""Fail-closed handling for CP-SAT's non-minimal assumption cores."""

from collections.abc import Iterable
from typing import Any


def sound_infeasible_assumption_indexes(
    solver: Any, known_indexes: Iterable[int]
) -> tuple[int, ...]:
    """Return a sufficient named core, or all assumptions if CP-SAT strays."""
    known = tuple(int(index) for index in known_indexes)
    reported = tuple(
        int(index) for index in solver.sufficient_assumptions_for_infeasibility()
    )
    return reported if set(reported) <= set(known) else known
