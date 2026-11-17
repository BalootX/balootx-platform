"""Search completion refusals shared by runtime and pure evidence contracts."""
from __future__ import annotations

from typing import Mapping


class SearchTimeBudgetExceeded(TimeoutError):
    """Wall-clock exhaustion that must not produce canonical search evidence."""

    def to_detail(self) -> dict[str, str]:
        return {"reason": "time_censored", "message": str(self)}


def require_uncensored_play_trace(trace: Mapping[str, object]) -> None:
    if trace.get("analysisSolverCompletion") == "time_censored":
        raise SearchTimeBudgetExceeded(
            "time-censored play trace cannot produce canonical output"
        )
