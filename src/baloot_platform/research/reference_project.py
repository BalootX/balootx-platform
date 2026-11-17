from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, Seat, Suit, Team
from baloot_platform.core.project_ir import ProjectDomainIRV1


@dataclass(frozen=True, slots=True)
class ReferenceProjectV1:
    owner: Seat
    kind: str
    cards: tuple[Card, ...]
    value: int
    high_rank: int
    category_priority: int
    declared: bool = False
    declaration_order: int = -1
    scoring: bool = False
    revealed: bool = False

    @property
    def team(self) -> Team:
        return self.owner.team

    @property
    def card_mask(self) -> int:
        return sum(1 << card.id for card in self.cards)


class ReferenceProjectInterpreterV1:
    """Independent projection over the shared executable project DomainIR."""

    def __init__(self, domain: ProjectDomainIRV1):
        if not isinstance(domain, ProjectDomainIRV1):
            raise TypeError(
                "ReferenceProjectInterpreterV1 requires a compiled ProjectDomainIRV1"
            )
        self.domain = domain

    @classmethod
    def from_rules(cls, rules: Mapping[str, Any]) -> "ReferenceProjectInterpreterV1":
        return cls(ProjectDomainIRV1.from_rules(rules))

    def sequence_strength(self, card: Card) -> int:
        return self.domain.sequence_strength_for(card)

    def candidates(
        self, owner: Seat, cards: Sequence[Card], contract: Contract
    ) -> tuple[ReferenceProjectV1, ...]:
        return tuple(
            ReferenceProjectV1(
                project.owner,
                project.kind,
                project.cards,
                project.value,
                project.high_rank,
                project.category_priority,
            )
            for project in self.domain.candidates(owner, cards, contract)
        )

    def group_score(self, group: Sequence[Any]) -> tuple[int, int, int, int]:
        return self.domain.group_score(group)

    def select(
        self, candidates: Sequence[ReferenceProjectV1]
    ) -> tuple[ReferenceProjectV1, ...]:
        return self.domain.select(candidates)

    def rank_key(
        self,
        project: Any,
        *,
        trump_suit: Suit | None = None,
    ) -> tuple[int, int, int, int]:
        return self.domain.rank_key(
            project,
            trump_suit=trump_suit,
        )

    def complete_baloot_teams(
        self,
        hands: Mapping[Seat, Sequence[Card]],
        projects: Sequence[Any],
        contract: Contract,
        trump_suit: Suit | None,
    ) -> frozenset[Team]:
        return self.domain.complete_baloot_teams(
            hands,
            projects,
            contract,
            trump_suit,
        )

    def resolve_declarations(
        self,
        projects: Sequence[ReferenceProjectV1],
        *,
        dealer: Seat,
        trump_suit: Suit | None,
        second_trick_leader: Seat | None = None,
    ) -> tuple[list[ReferenceProjectV1], Team | None]:
        return self.domain.resolve_declarations(
            projects,
            dealer=dealer,
            trump_suit=trump_suit,
            second_trick_leader=second_trick_leader,
        )


__all__ = [
    "ProjectDomainIRV1",
    "ReferenceProjectInterpreterV1",
    "ReferenceProjectV1",
]
