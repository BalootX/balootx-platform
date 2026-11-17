from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from baloot_platform.core.enums import Contract, Seat, Team
from baloot_platform.core.scoring_ir import ScoringDomainIRV1


@dataclass(frozen=True, slots=True)
class ReferenceRoundScoreV1:
    raw_card_points: tuple[int, int]
    card_game_points: tuple[int, int]
    project_points: tuple[int, int]
    baloot_points: tuple[int, int]
    total_before_award: tuple[int, int]
    winner: Team
    awarded_points: tuple[int, int]
    counting_team: Team
    multiplier: int
    kabout: bool
    gahwa: bool
    match_winner: Team | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rawCardPoints": list(self.raw_card_points),
            "cardGamePoints": list(self.card_game_points),
            "projectPoints": list(self.project_points),
            "balootPoints": list(self.baloot_points),
            "totalBeforeAward": list(self.total_before_award),
            "winner": self.winner.name.lower(),
            "awardedPoints": list(self.awarded_points),
            "countingTeam": self.counting_team.name.lower(),
            "multiplier": self.multiplier,
            "kabout": self.kabout,
            "gahwa": self.gahwa,
            "matchWinner": (
                self.match_winner.name.lower()
                if self.match_winner is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ReferenceMatchOutcomeV1:
    starting_score: tuple[int, int]
    awarded_points: tuple[int, int]
    final_score: tuple[int, int]
    winner: Team | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "startingScore": list(self.starting_score),
            "awardedPoints": list(self.awarded_points),
            "finalScore": list(self.final_score),
            "winner": self.winner.name.lower() if self.winner is not None else None,
        }


class ReferenceScoringInterpreterV1:
    """Pure round scoring semantics independent of the mutable runtime engine."""

    def __init__(self, domain: ScoringDomainIRV1):
        self.domain = domain

    @classmethod
    def from_rules(cls, rules: Mapping[str, Any]) -> "ReferenceScoringInterpreterV1":
        return cls(ScoringDomainIRV1.from_rules(rules))

    def convert(self, raw: int, contract: Contract) -> int:
        return self.domain.convert_raw(raw, contract)

    @staticmethod
    def multiplier(level: str) -> int:
        return ScoringDomainIRV1.multiplier(level)

    def tie_winner(
        self,
        contract: Contract,
        counting_raw: int,
        counting_team: Team,
        declarer_team: Team,
    ) -> Team:
        return self.domain.tie_winner(
            contract,
            counting_raw,
            counting_team,
            declarer_team,
        )

    def apply_match_award(
        self,
        *,
        starting_score: Sequence[int],
        awarded_points: Sequence[int],
        round_winner: Team,
        gahwa: bool = False,
    ) -> ReferenceMatchOutcomeV1:
        """Apply one scored round to the configured match target.

        This is deliberately separate from round-card scoring so the reference
        interpreter exposes the match boundary without depending on the
        mutable runtime engine.
        """

        starting = tuple(int(value) for value in starting_score)
        awarded = tuple(int(value) for value in awarded_points)
        final, winner = self.domain.apply_match_award(
            starting_score=starting,
            awarded_points=awarded,
            round_winner=round_winner,
            gahwa=gahwa,
        )
        return ReferenceMatchOutcomeV1(
            starting,
            awarded,
            final,
            winner,
        )

    def score(
        self,
        *,
        contract: Contract,
        declarer: Seat,
        escalation: str,
        trick_raw_points: Sequence[int],
        trick_winners: Sequence[Seat],
        project_points: Sequence[int] = (0, 0),
        baloot_points: Sequence[int] = (0, 0),
        counting_team: Team | None = None,
    ) -> ReferenceRoundScoreV1:
        contract = Contract.parse(contract)
        declarer = Seat.parse(declarer)
        raw_rows = tuple(int(value) for value in trick_raw_points)
        winners = tuple(Seat.parse(value) for value in trick_winners)
        projects = tuple(int(value) for value in project_points)
        baloot = tuple(int(value) for value in baloot_points)
        if len(raw_rows) != 8 or len(winners) != 8:
            raise ValueError("reference scoring requires eight complete tricks")
        if len(projects) != 2 or len(baloot) != 2 or min((*projects, *baloot)) < 0:
            raise ValueError("project and Baloot points require two non-negative values")
        raw = [0, 0]
        tricks_won = [0, 0]
        for points, winner in zip(raw_rows, winners, strict=True):
            raw[int(winner.team)] += points
            tricks_won[int(winner.team)] += 1
        raw[int(winners[-1].team)] += self.domain.last_trick_bonus_raw
        counting = Team(
            int(
                declarer.team.other
                if counting_team is None
                else counting_team
            )
        )
        contract_key = contract.effective.value
        expected_raw = self.domain.raw_card_totals[contract_key]
        if sum(raw) != expected_raw:
            raise ValueError(
                f"reference trick points total {sum(raw)} does not match compiled {contract_key} total {expected_raw}"
            )
        total_cards = self.domain.card_game_point_totals[contract_key]
        converted = self.convert(raw[int(counting)], contract)
        card_game = [0, 0]
        card_game[int(counting)] = converted
        card_game[int(counting.other)] = total_cards - converted
        totals = [
            card_game[index] + projects[index] + baloot[index]
            for index in range(2)
        ]
        if totals[0] > totals[1]:
            winner = Team.NORTH_SOUTH
        elif totals[1] > totals[0]:
            winner = Team.WEST_EAST
        else:
            winner = self.tie_winner(
                contract, raw[int(counting)], counting, declarer.team
            )
        multiplier = self.multiplier(escalation)
        kabout = self.domain.kabout_enabled and 0 in tricks_won
        if kabout and self.domain.kabout_winner == "trick_sweeper":
            winner = Team.NORTH_SOUTH if tricks_won[0] else Team.WEST_EAST
        gahwa = str(escalation) == "gahwa"
        awarded = self.domain.round_award(
            contract=contract,
            winner=winner,
            declarer_team=declarer.team,
            escalation=escalation,
            total_before_award=totals,
            project_points=projects,
            baloot_points=baloot,
            kabout=kabout,
        )
        return ReferenceRoundScoreV1(
            (raw[0], raw[1]),
            (card_game[0], card_game[1]),
            projects,
            baloot,
            (totals[0], totals[1]),
            winner,
            (awarded[0], awarded[1]),
            counting,
            multiplier,
            kabout,
            gahwa,
            winner if gahwa else None,
        )
