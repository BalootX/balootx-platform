from __future__ import annotations

from typing import Sequence

from .cards import Card
from .enums import Contract, Seat, Suit, Team
from .models import Project
from .project_ir import ProjectDomainIRV1
from .rules import RuleBook


def _domain(rules: RuleBook) -> ProjectDomainIRV1:
    return rules.project_domain


def project_candidates(
    owner: Seat,
    cards: tuple[Card, ...],
    contract: Contract,
    rules: RuleBook,
) -> tuple[Project, ...]:
    return _domain(rules).candidates(owner, cards, contract)


def project_group_score(
    group: tuple[Project, ...],
) -> tuple[int, int, int, int]:
    return ProjectDomainIRV1.group_score(group)


def project_rank_key(
    project: Project,
    *,
    trump_suit: Suit | None = None,
    domain: ProjectDomainIRV1,
) -> tuple[int, int, int, int]:
    if not isinstance(domain, ProjectDomainIRV1):
        raise TypeError("project_rank_key requires a compiled ProjectDomainIRV1")
    return domain.rank_key(
        project,
        trump_suit=trump_suit,
    )


def baloot_project_state(
    projects: Sequence[Project],
    *,
    owner: Seat,
    trump_suit: Suit,
    domain: ProjectDomainIRV1,
) -> tuple[bool, bool]:
    if not isinstance(domain, ProjectDomainIRV1):
        raise TypeError("baloot_project_state requires a compiled ProjectDomainIRV1")
    return domain.baloot_project_state(
        projects,
        owner=owner,
        trump_suit=trump_suit,
    )


def complete_baloot_teams(
    hands: dict[Seat, tuple[Card, ...]],
    projects: Sequence[Project],
    contract: Contract,
    trump_suit: Suit | None,
    rules: RuleBook,
) -> frozenset[Team]:
    return _domain(rules).complete_baloot_teams(
        hands,
        projects,
        contract,
        trump_suit,
    )


def detect_projects(
    hands: dict[Seat, tuple[Card, ...]],
    contract: Contract,
    rules: RuleBook,
    dealer: Seat,
) -> list[Project]:
    return _domain(rules).detect(hands, contract, dealer)


def project_card_patterns(
    *,
    kind: str,
    value: int,
    high_rank: int | None,
    contract: Contract,
    rules: RuleBook,
) -> tuple[int, ...]:
    return _domain(rules).card_patterns(
        kind=kind,
        value=value,
        high_rank=high_rank,
        contract=contract,
    )


def announce_best_project(
    projects: list[Project],
    seat: Seat,
    *,
    domain: ProjectDomainIRV1,
) -> tuple[list[Project], Project | None]:
    if not isinstance(domain, ProjectDomainIRV1):
        raise TypeError("announce_best_project requires a compiled ProjectDomainIRV1")
    return domain.announce_best(projects, seat)


def announce_project_for_turn(
    projects: list[Project],
    seat: Seat,
    *,
    trump_suit: Suit | None,
    domain: ProjectDomainIRV1,
) -> tuple[list[Project], Project | None]:
    if not isinstance(domain, ProjectDomainIRV1):
        raise TypeError(
            "announce_project_for_turn requires a compiled ProjectDomainIRV1"
        )
    return domain.announce_for_turn(
        projects,
        seat,
        trump_suit=trump_suit,
    )


def score_project_team(
    projects: list[Project],
    *,
    trump_suit: Suit | None = None,
    domain: ProjectDomainIRV1,
    second_trick_leader: Seat | None = None,
) -> tuple[list[Project], Team | None]:
    if not isinstance(domain, ProjectDomainIRV1):
        raise TypeError("score_project_team requires a compiled ProjectDomainIRV1")
    return domain.score_team(
        projects,
        trump_suit=trump_suit,
        second_trick_leader=second_trick_leader,
    )


def project_lifecycle_matches(
    project: Project,
    requested_state: str,
    *,
    projects: Sequence[Project],
    winning_team: Team | None,
    trump_suit: Suit | None,
    domain: ProjectDomainIRV1,
    second_trick_leader: Seat | None = None,
) -> bool:
    """Match one authoritative project against a distinct lifecycle state."""

    if requested_state == "eligible":
        return True
    if requested_state == "declared":
        return bool(project.declared)
    if requested_state == "awarded":
        return bool(project.scoring)
    if requested_state != "comparison_winner":
        raise ValueError("unsupported project state")
    if winning_team is None:
        return False
    winner = domain.comparison_winner(
        projects, trump_suit=trump_suit, second_trick_leader=second_trick_leader
    )
    if winner is None or winner.team != winning_team:
        raise AssertionError("project comparison winner disagrees with engine state")
    return project is winner


__all__ = [
    "announce_best_project",
    "announce_project_for_turn",
    "baloot_project_state",
    "complete_baloot_teams",
    "detect_projects",
    "project_candidates",
    "project_card_patterns",
    "project_group_score",
    "project_lifecycle_matches",
    "project_rank_key",
    "score_project_team",
]
