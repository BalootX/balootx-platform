"""Rules-bound disclosure of declared project ranks."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def project_rank_visibility_policy(rules: Any) -> dict[str, Any] | None:
    if rules.data["projects"].get("top_rank_visibility", "declared") == "declared":
        return None
    return {"mode": "comparison_only", "categoryPriorities": dict(rules.project_domain.category_priorities)}


def project_rank_is_public(
    project: Mapping[str, Any], projects: Sequence[Mapping[str, Any]],
    observer: str, policy: Mapping[str, Any] | None, *, comparison_complete: bool,
) -> bool:
    if policy is None or project.get("owner") == observer or project.get("revealed"):
        return True
    if policy.get("mode") != "comparison_only":
        raise ValueError("unsupported project rank visibility")
    if not comparison_complete:
        return False
    priorities = policy["categoryPriorities"]
    def category(item: Mapping[str, Any]) -> tuple[int, int]:
        return int(item["value"]), int(priorities[item["kind"]])
    def team(item: Mapping[str, Any]) -> int:
        return int(item["owner"] in {"north", "south"})
    declared = [item for item in projects if int(item.get("declarationOrder", 999)) < 999]
    best = [max((category(item) for item in declared if team(item) == side), default=None) for side in (0, 1)]
    return best[0] is not None and best[0] == best[1] == category(project)


def redact_project_ranks(state: Mapping[str, Any], observer: str, projects: list[dict[str, Any]]) -> None:
    policy = state.get("projectRankVisibility")
    if policy is None:
        return
    complete = bool(state.get("completedTricks")) or any(event.get("type") == "projects_compared" for event in state.get("events", ()))
    for project in projects:
        if not project_rank_is_public(project, state.get("projects", ()), observer, policy, comparison_complete=complete):
            project["highRank"] = None
