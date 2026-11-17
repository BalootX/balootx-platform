from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Sequence

from .enums import Contract, Team


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    # Local imports keep the core state machine independent of Research.
    import hashlib
    import json

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ScoringDomainIRV1:
    """Validated scoring semantics shared by runtime, reference, and CP-SAT."""

    # Finite selectors belonging to the V1 tie primitive, not evaluated outcomes.
    tie_counting_units: ClassVar[tuple[tuple[int, ...], ...]] = (
        (1, 2, 3, 4), (2, 3, 4, 5),
    )
    escalation_multipliers: ClassVar[tuple[tuple[str, int], ...]] = (
        ("normal", 1), ("double", 2), ("triple", 3), ("four", 4), ("gahwa", 1),
    )

    last_trick_bonus_raw: int
    raw_card_totals: Mapping[str, int]
    card_game_point_totals: Mapping[str, int]
    conversion: Mapping[str, str]
    buyer_team_must_win: bool
    loss_transfers_all: bool
    winner_takes_all_when_multiplied: bool
    project_multiplier_cap: int
    baloot_multiplier_cap: int
    kabout_enabled: bool
    kabout_base_game_points: Mapping[str, int]
    kabout_multiplier_cap: int
    match_target: int
    tie_above_target_continues: bool
    dealer_rotates_right: bool
    sha256: str
    kabout_winner: str | None = None
    buyer_tie_outcome: str | None = None

    @classmethod
    def from_rules(cls, rules: Mapping[str, Any]) -> "ScoringDomainIRV1":
        try:
            scoring = dict(rules["scoring"])
            card_rules = dict(rules["card_rules"])
            raw_totals = cls._contract_values(scoring["raw_card_totals"])
            game_totals = cls._contract_values(scoring["card_game_point_totals"])
            conversion = {
                str(key): str(value) for key, value in scoring["conversion"].items()
            }
            contract = dict(scoring["contract"])
            buyer_tie_outcome = contract.get("buyer_tie_outcome")
            if "buyer_tie_outcome" in contract and buyer_tie_outcome not in {
                "buyer_makes", "buyer_loses",
            }:
                raise ValueError("unsupported buyer tie outcome")
            escalation = dict(scoring["escalation"])
            kabout = dict(scoring["kabout"])
            kabout_winner = kabout.get("winner")
            if "winner" in kabout and kabout_winner != "trick_sweeper":
                raise ValueError("unsupported Kabout winner primitive")
            match = dict(scoring["match"])
            boolean_values = (
                contract["buyer_team_must_win"],
                contract["loss_transfers_all"],
                escalation["winner_takes_all_when_multiplied"],
                kabout["enabled"],
                match["tie_above_target_continues"],
                match["dealer_rotates_right"],
            )
            if any(type(value) is not bool for value in boolean_values):
                raise TypeError("scoring switches must be booleans")
            integer_values = (
                scoring["last_trick_bonus_raw"],
                escalation["project_multiplier_cap"],
                escalation["baloot_multiplier_cap"],
                kabout.get("multiplier_cap", 4),
                match["target"],
            )
            if any(type(value) is not int for value in integer_values):
                raise TypeError("scoring integer values must be integers")
            core = {
                "schemaVersion": 1,
                "lastTrickBonusRaw": scoring["last_trick_bonus_raw"],
                "rawCardTotals": raw_totals,
                "cardGamePointTotals": game_totals,
                "conversion": conversion,
                "buyerTeamMustWin": contract["buyer_team_must_win"],
                "lossTransfersAll": contract["loss_transfers_all"],
                "winnerTakesAllWhenMultiplied": escalation[
                    "winner_takes_all_when_multiplied"
                ],
                "projectMultiplierCap": escalation["project_multiplier_cap"],
                "balootMultiplierCap": escalation["baloot_multiplier_cap"],
                "kaboutEnabled": kabout["enabled"],
                "kaboutBaseGamePoints": cls._contract_values(
                    kabout["base_game_points"]
                ),
                "kaboutMultiplierCap": kabout.get("multiplier_cap", 4),
                "matchTarget": match["target"],
                "tieAboveTargetContinues": match["tie_above_target_continues"],
                "dealerRotatesRight": match["dealer_rotates_right"],
            }
            expected_raw_totals = {
                "sun": (
                    4 * sum(cls._card_point_table(card_rules["sun_points"]))
                    + core["lastTrickBonusRaw"]
                ),
                "hokum": (
                    sum(cls._card_point_table(card_rules["hokum_trump_points"]))
                    + 3
                    * sum(
                        cls._card_point_table(
                            card_rules["hokum_non_trump_points"]
                        )
                    )
                    + core["lastTrickBonusRaw"]
                ),
            }
            if kabout_winner is not None:
                core["kaboutWinner"] = kabout_winner
            if buyer_tie_outcome is not None:
                core["buyerTieOutcome"] = buyer_tie_outcome
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError("incomplete or invalid scoring domain configuration") from exc

        expected_conversion = {
            "sun": "nearest_ten_except_exact_five_then_divide_by_five",
            "hokum": "nearest_ten_half_down_then_divide_by_ten",
        }
        if conversion != expected_conversion:
            raise ValueError("unsupported scoring conversion primitive")
        if not all((core["buyerTeamMustWin"], core["lossTransfersAll"], core["winnerTakesAllWhenMultiplied"])):
            raise ValueError("unsupported contract or multiplied-award scoring behavior")
        if not core["tieAboveTargetContinues"] or not core["dealerRotatesRight"]:
            raise ValueError("unsupported match progression behavior")
        if core["lastTrickBonusRaw"] < 0:
            raise ValueError("last-trick bonus must be non-negative")
        if core["projectMultiplierCap"] < 1 or core["balootMultiplierCap"] < 1:
            raise ValueError("scoring multiplier caps must be positive")
        if core["matchTarget"] < 1:
            raise ValueError("match target must be positive")
        if raw_totals != expected_raw_totals:
            raise ValueError(
                "scoring raw-card totals must match the governed card-point tables"
            )
        if any(value < 0 for value in (*raw_totals.values(), *game_totals.values())):
            raise ValueError("scoring totals must be non-negative")
        if any(value < 0 for value in core["kaboutBaseGamePoints"].values()):
            raise ValueError("kabout values must be non-negative")
        if core["kaboutMultiplierCap"] < 1:
            raise ValueError("Kabout multiplier cap must be positive")

        return cls(
            core["lastTrickBonusRaw"],
            raw_totals,
            game_totals,
            conversion,
            core["buyerTeamMustWin"],
            core["lossTransfersAll"],
            core["winnerTakesAllWhenMultiplied"],
            core["projectMultiplierCap"],
            core["balootMultiplierCap"],
            core["kaboutEnabled"],
            core["kaboutBaseGamePoints"],
            core["kaboutMultiplierCap"],
            core["matchTarget"],
            core["tieAboveTargetContinues"],
            core["dealerRotatesRight"],
            _canonical_sha256(core),
            kabout_winner,
            buyer_tie_outcome,
        )

    @staticmethod
    def _contract_values(value: Mapping[str, Any]) -> dict[str, int]:
        if set(value) != {"sun", "hokum"}:
            raise ValueError("scoring contract table must contain sun and hokum")
        result: dict[str, int] = {}
        for key in ("sun", "hokum"):
            item = value[key]
            if type(item) is not int:
                raise TypeError("scoring contract values must be integers")
            result[key] = item
        return result

    @staticmethod
    def _card_point_table(value: Any) -> tuple[int, ...]:
        if not isinstance(value, (list, tuple)) or len(value) != 8:
            raise TypeError("card-point tables must contain eight integers")
        if any(type(item) is not int or item < 0 for item in value):
            raise TypeError("card-point tables must contain non-negative integers")
        return tuple(value)

    @staticmethod
    def round_half_down_to_ten(value: int) -> int:
        tens, units = divmod(int(value), 10)
        return (tens + int(units >= 6)) * 10

    @staticmethod
    def sun_game_points(raw: int) -> int:
        raw = int(raw)
        units = raw % 10
        rounded = raw if units == 5 else ((raw + 5) // 10) * 10
        return rounded // 5

    @classmethod
    def hokum_game_points(cls, raw: int) -> int:
        return cls.round_half_down_to_ten(raw) // 10

    def convert_raw(self, raw: int, contract: Contract) -> int:
        raw = int(raw)
        mode = self.conversion[Contract.parse(contract).effective.value]
        if mode == "nearest_ten_half_down_then_divide_by_ten":
            return self.hokum_game_points(raw)
        if mode == "nearest_ten_except_exact_five_then_divide_by_five":
            return self.sun_game_points(raw)
        raise ValueError(f"unsupported scoring conversion primitive: {mode}")

    @classmethod
    def multiplier(cls, level: str) -> int:
        try:
            return dict(cls.escalation_multipliers)[str(level)]
        except KeyError as exc:
            raise ValueError("unsupported escalation level") from exc

    def tie_winner(
        self,
        contract: Contract,
        counting_raw: int,
        counting_team: Team,
        declarer_team: Team,
    ) -> Team:
        effective = Contract.parse(contract).effective
        if effective == Contract.SUN and self.buyer_tie_outcome is not None:
            return declarer_team if self.buyer_tie_outcome == "buyer_makes" else declarer_team.other
        units = int(counting_raw) % 10
        selector = int(effective == Contract.HOKUM)
        return counting_team if units in self.tie_counting_units[selector] else declarer_team

    @staticmethod
    def _team(value: Team | str | int) -> Team:
        return Team[str(value).strip().upper()] if isinstance(value, str) else Team(int(value))

    @classmethod
    def resolve_counting_team(
        cls,
        *,
        declarer_team: Team | str | int,
        escalation: str,
        last_escalating_team: Team | str | int | None = None,
    ) -> Team:
        declarer = cls._team(declarer_team)
        cls.multiplier(escalation)
        if str(escalation) == "normal" or last_escalating_team is None:
            return declarer.other
        return cls._team(last_escalating_team).other

    def apply_match_award(
        self,
        *,
        starting_score: Sequence[int],
        awarded_points: Sequence[int],
        round_winner: Team | str | int,
        gahwa: bool = False,
    ) -> tuple[tuple[int, int], Team | None]:
        starting = tuple(int(value) for value in starting_score)
        awarded = tuple(int(value) for value in awarded_points)
        if len(starting) != 2 or len(awarded) != 2 or min((*starting, *awarded)) < 0:
            raise ValueError("match scoring requires two non-negative score values")
        winner = self._team(round_winner)
        final = [starting[0], starting[1]]
        if bool(gahwa):
            winner_index = int(winner)
            final[winner_index] = max(self.match_target, final[1 - winner_index] + 1)
        else:
            final[0] += awarded[0]
            final[1] += awarded[1]
        match_winner = None
        if max(final) >= self.match_target and final[0] != final[1]:
            match_winner = Team.NORTH_SOUTH if final[0] > final[1] else Team.WEST_EAST
        return (final[0], final[1]), match_winner

    def round_award(
        self,
        *,
        contract: Contract,
        winner: Team,
        declarer_team: Team,
        escalation: str,
        total_before_award: Sequence[int],
        project_points: Sequence[int],
        baloot_points: Sequence[int],
        kabout: bool,
    ) -> tuple[int, int]:
        winner = Team(int(winner))
        declarer_team = Team(int(declarer_team))
        totals = tuple(int(value) for value in total_before_award)
        projects = tuple(int(value) for value in project_points)
        baloot = tuple(int(value) for value in baloot_points)
        if len(totals) != 2 or len(projects) != 2 or len(baloot) != 2:
            raise ValueError("round award inputs must contain two team values")
        contract_key = Contract.parse(contract).effective.value

        awarded = [0, 0]
        if str(escalation) == "gahwa":
            return awarded[0], awarded[1]

        multiplier = self.multiplier(escalation)
        if bool(kabout):
            awarded[int(winner)] = (
                self.kabout_base_game_points[contract_key]
                * min(multiplier, self.kabout_multiplier_cap)
                + projects[int(winner)]
                * min(multiplier, self.project_multiplier_cap)
                + baloot[int(winner)]
                * min(multiplier, self.baloot_multiplier_cap)
            )
        elif str(escalation) == "normal":
            if winner == declarer_team:
                awarded = list(totals)
            else:
                awarded[int(winner)] = (
                    self.card_game_point_totals[contract_key]
                    + sum(projects)
                    + sum(baloot)
                )
        else:
            awarded[int(winner)] = (
                self.card_game_point_totals[contract_key]
                * multiplier
                + sum(projects) * min(multiplier, self.project_multiplier_cap)
                + sum(baloot) * min(multiplier, self.baloot_multiplier_cap)
            )
        return awarded[0], awarded[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "lastTrickBonusRaw": self.last_trick_bonus_raw,
            "rawCardTotals": dict(self.raw_card_totals),
            "cardGamePointTotals": dict(self.card_game_point_totals),
            "conversion": dict(self.conversion),
            "buyerTeamMustWin": self.buyer_team_must_win,
            "lossTransfersAll": self.loss_transfers_all,
            "winnerTakesAllWhenMultiplied": self.winner_takes_all_when_multiplied,
            "projectMultiplierCap": self.project_multiplier_cap,
            "balootMultiplierCap": self.baloot_multiplier_cap,
            "kaboutEnabled": self.kabout_enabled,
            "kaboutBaseGamePoints": dict(self.kabout_base_game_points),
            "kaboutMultiplierCap": self.kabout_multiplier_cap,
            "matchTarget": self.match_target,
            "tieAboveTargetContinues": self.tie_above_target_continues,
            "dealerRotatesRight": self.dealer_rotates_right,
            "sha256": self.sha256,
            **({"kaboutWinner": self.kabout_winner} if self.kabout_winner is not None else {}),
            **({"buyerTieOutcome": self.buyer_tie_outcome} if self.buyer_tie_outcome is not None else {}),
        }


__all__ = ["ScoringDomainIRV1"]
