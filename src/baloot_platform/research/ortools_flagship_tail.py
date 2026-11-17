"""Engine-free CP-SAT construction for the late flagship tactic.

This module deliberately stops at a tail witness.  Prefix reachability,
project award, and the unique-root strategic claim remain explicit lift/proof
requirements until their owning stages certify them.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Sequence

from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, Rank, Seat, Suit, Team
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1

from .scenario_contracts import canonical_sha256


_TAIL_TRICKS = 3
_PLAYS_PER_TRICK = 4
_TAIL_CARD_COUNT = _TAIL_TRICKS * _PLAYS_PER_TRICK


@dataclass(frozen=True, slots=True)
class FlagshipTailPlayV1:
    trick_index: int
    ordinal: int
    actor: Seat
    card: Card

    def to_dict(self) -> dict[str, Any]:
        return {
            "trickIndex": self.trick_index,
            "ordinal": self.ordinal,
            "actor": self.actor.name.lower(),
            "card": self.card.code,
        }


@dataclass(frozen=True, slots=True)
class FlagshipTailCandidateV1:
    status: str
    contract: Contract
    buyer: Seat
    acting_seat: Seat
    target_team: Team
    plays: tuple[FlagshipTailPlayV1, ...]
    trick_winners: tuple[Seat, Seat, Seat]
    root_action: Card
    root_legal_actions: tuple[Card, ...]
    prefix_raw_points: tuple[int, int]
    final_raw_points: tuple[int, int]
    final_game_points: tuple[int, int]
    project_game_points: tuple[int, int]
    exact_no_good: tuple[int, ...]
    exact_no_good_hash: str
    suit_symmetry_key: str
    diversity_key: str
    model_hash: str
    candidate_hash: str
    branches: int
    conflicts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "status": self.status,
            "contract": self.contract.value,
            "buyer": self.buyer.name.lower(),
            "actingSeat": self.acting_seat.name.lower(),
            "targetTeam": self.target_team.name.lower(),
            "plays": [play.to_dict() for play in self.plays],
            "trickWinners": [seat.name.lower() for seat in self.trick_winners],
            "rootAction": self.root_action.code,
            "rootLegalActions": [card.code for card in self.root_legal_actions],
            "requiredLift": {
                "status": "not_run",
                "prefixTricks": 5,
                "prefixRawPoints": list(self.prefix_raw_points),
                "project": {
                    "state": "awarded",
                    "team": self.buyer.team.name.lower(),
                    "displayValue": 100,
                    "governedGamePoints": self.project_game_points[
                        int(self.buyer.team)
                    ],
                },
            },
            "requiredProof": {
                "status": "not_run",
                "mode": "exact",
                "visibility": "complete_state",
                "claim": "selected_action_is_unique_winner",
                "selectedAction": self.root_action.code,
                "requestedWinningRootActionCount": 1,
                "targetTeam": self.target_team.name.lower(),
                "utility": "round_win",
            },
            "terminalScoringWitness": {
                "rawPoints": list(self.final_raw_points),
                "cardGamePoints": list(self.final_game_points),
                "projectGamePoints": list(self.project_game_points),
                "winner": self.target_team.name.lower(),
            },
            "exactNoGood": list(self.exact_no_good),
            "exactNoGoodHash": self.exact_no_good_hash,
            "suitSymmetryKey": self.suit_symmetry_key,
            "diversityKey": self.diversity_key,
            "modelHash": self.model_hash,
            "candidateHash": self.candidate_hash,
            "branches": self.branches,
            "conflicts": self.conflicts,
        }


@dataclass(frozen=True, slots=True)
class FlagshipTailSearchResultV1:
    status: str
    solver_status: str
    candidate: FlagshipTailCandidateV1 | None
    model_hash: str
    exact_no_good_hashes: tuple[str, ...]
    skipped_symmetry_keys: tuple[str, ...]
    branches: int
    conflicts: int


def _normalize_exact_no_goods(
    values: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    result: set[tuple[int, ...]] = set()
    for index, value in enumerate(values):
        if len(value) != _TAIL_CARD_COUNT or any(type(card) is not int for card in value):
            raise ValueError(f"tail no-good {index} must contain twelve integer card ids")
        item = tuple(value)
        if len(set(item)) != len(item) or any(not 0 <= card < 32 for card in item):
            raise ValueError(f"tail no-good {index} must contain twelve unique card ids")
        result.add(item)
    return tuple(sorted(result))


def _canonical_suit_pattern(cards: Sequence[Card]) -> tuple[int, ...]:
    """Canonicalize every Sun suit permutation by first-seen suit identity."""

    seen: dict[Suit, int] = {}
    pattern: list[int] = []
    for card in cards:
        if card.suit not in seen:
            seen[card.suit] = len(seen)
        pattern.extend((seen[card.suit], int(card.rank)))
    return tuple(pattern)


class OrToolsFlagshipTailSearchV1:
    """Find one deterministic tricks 6-8 witness without engine execution."""

    def __init__(
        self,
        *,
        rules: RuleBook,
        acting_seat: Seat = Seat.WEST,
        excluded_exact_tails: Sequence[Sequence[int]] = (),
        excluded_suit_symmetry_keys: Sequence[str] = (),
    ) -> None:
        if not isinstance(rules, RuleBook):
            raise TypeError("flagship tail search requires RuleBook")
        self.rules = rules
        self.acting_seat = Seat.parse(acting_seat)
        self.structural_domain = rules.structural_domain
        self.buyer = self.structural_domain.offset(self.acting_seat, -1)
        if self.buyer.team == self.acting_seat.team:
            raise AssertionError("derived flagship buyer must oppose the acting seat")
        self.target_team = self.acting_seat.team
        self.no_goods = _normalize_exact_no_goods(excluded_exact_tails)
        keys = tuple(sorted(set(excluded_suit_symmetry_keys)))
        if any(
            type(key) is not str
            or len(key) != 64
            or any(character not in "0123456789abcdef" for character in key)
            for key in keys
        ):
            raise ValueError("suit-symmetry exclusions must be lowercase SHA-256 values")
        self.excluded_symmetry_keys = keys
        self.play_domain = rules.play_domain
        self.scoring_domain = ScoringDomainIRV1.from_rules(rules.data)
        self.project_value = rules.project_domain.sequence_value_for(
            Contract.SUN, "five_plus"
        )
        if self.project_value <= 0:
            raise ValueError("active Sun hundred project must have positive game points")

        self.model = cp_model.CpModel()
        self.cards = [
            self.model.new_int_var(0, 31, f"tail_card_{index}")
            for index in range(_TAIL_CARD_COUNT)
        ]
        self.model.add_all_different(self.cards)
        self.ranks = [
            self.model.new_int_var(0, 7, f"tail_rank_{index}")
            for index in range(_TAIL_CARD_COUNT)
        ]
        self.suits = [
            self.model.new_int_var(0, 3, f"tail_suit_{index}")
            for index in range(_TAIL_CARD_COUNT)
        ]
        self.strengths = [
            self.model.new_int_var(0, 7, f"tail_strength_{index}")
            for index in range(_TAIL_CARD_COUNT)
        ]
        self.points = [
            self.model.new_int_var(0, max(self.play_domain.sun_points), f"tail_points_{index}")
            for index in range(_TAIL_CARD_COUNT)
        ]
        for index, card in enumerate(self.cards):
            self.model.add_modulo_equality(self.ranks[index], card, 8)
            self.model.add_division_equality(self.suits[index], card, 8)
            self.model.add_element(
                self.ranks[index], self.play_domain.sun_strength, self.strengths[index]
            )
            self.model.add_element(
                self.ranks[index], self.play_domain.sun_points, self.points[index]
            )
        if self.no_goods:
            self.model.add_forbidden_assignments(self.cards, self.no_goods)

        self.orders = (
            self._order(self.buyer),
            self._order(self.buyer),
            self._order(self.acting_seat),
        )
        self.winners = (self.buyer, self.acting_seat, self.acting_seat)
        self._add_legal_play_and_winners()
        self._add_flagship_card_constraints()
        self._add_terminal_scoring()
        self.model.add_decision_strategy(
            [*self.cards, self.prefix_target_raw],
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MIN_VALUE,
        )
        self.model.minimize(self.prefix_target_raw)

    def _order(self, leader: Seat) -> tuple[Seat, Seat, Seat, Seat]:
        return tuple(
            self.structural_domain.offset(leader, step) for step in range(4)
        )  # type: ignore[return-value]

    @staticmethod
    def _flat(trick: int, ordinal: int) -> int:
        return trick * _PLAYS_PER_TRICK + ordinal

    def _ordinal(self, trick: int, actor: Seat) -> int:
        return self.orders[trick].index(actor)

    def _flat_for(self, trick: int, actor: Seat) -> int:
        return self._flat(trick, self._ordinal(trick, actor))

    def _add_legal_play_and_winners(self) -> None:
        for trick, order in enumerate(self.orders):
            lead = self._flat(trick, 0)
            winner = self._flat_for(trick, self.winners[trick])
            for ordinal, actor in enumerate(order):
                index = self._flat(trick, ordinal)
                follows = self.model.new_bool_var(f"tail_follows_{trick}_{ordinal}")
                self.model.add(self.suits[index] == self.suits[lead]).only_enforce_if(
                    follows
                )
                self.model.add(self.suits[index] != self.suits[lead]).only_enforce_if(
                    follows.Not()
                )
                if ordinal:
                    for remaining_trick in range(trick, _TAIL_TRICKS):
                        remaining = self._flat_for(remaining_trick, actor)
                        self.model.add(
                            self.suits[remaining] != self.suits[lead]
                        ).only_enforce_if(follows.Not())
                if index != winner:
                    self.model.add(
                        self.strengths[index] < self.strengths[winner]
                    ).only_enforce_if(follows)
            self.model.add(self.suits[winner] == self.suits[lead])

    def _add_flagship_card_constraints(self) -> None:
        root = self._flat_for(0, self.acting_seat)
        self.model.add(self.ranks[root] == int(Rank.ACE))
        lead_suit = self.suits[self._flat(0, 0)]
        for trick in range(_TAIL_TRICKS):
            self.model.add(self.suits[self._flat_for(trick, self.acting_seat)] != lead_suit)

        low = tuple(int(rank) for rank in (Rank.SEVEN, Rank.EIGHT, Rank.NINE))
        for trick in (1, 2):
            self.model.add_allowed_assignments(
                [self.ranks[self._flat_for(trick, self.acting_seat)]],
                [(rank,) for rank in low],
            )

        opponents = tuple(seat for seat in Seat if seat.team == self.buyer.team)
        for opponent in opponents:
            self.model.add(
                self.ranks[self._flat_for(2, opponent)] == int(Rank.ACE)
            )
        self.model.add(
            self.ranks[
                self._flat_for(
                    2, self.structural_domain.partner(self.acting_seat)
                )
            ]
            != int(Rank.ACE)
        )

    def _add_terminal_scoring(self) -> None:
        tail_points = self.model.new_int_var(0, 120, "tail_card_points")
        self.model.add(tail_points == sum(self.points))
        prefix_total = self.model.new_int_var(0, 120, "prefix_card_points")
        self.model.add(prefix_total + tail_points == 120)
        self.prefix_target_raw = self.model.new_int_var(0, 120, "prefix_target_raw")
        self.model.add(self.prefix_target_raw <= prefix_total)
        prefix_buyer_raw = self.model.new_int_var(0, 120, "prefix_buyer_raw")
        self.model.add(prefix_buyer_raw + self.prefix_target_raw == prefix_total)

        trick_points: list[cp_model.IntVar] = []
        for trick in range(_TAIL_TRICKS):
            value = self.model.new_int_var(0, 44, f"trick_{trick + 6}_points")
            self.model.add(
                value
                == sum(
                    self.points[self._flat(trick, ordinal)]
                    for ordinal in range(_PLAYS_PER_TRICK)
                )
            )
            trick_points.append(value)

        self.final_raw = [
            self.model.new_int_var(0, 130, f"final_raw_{team.name.lower()}")
            for team in Team
        ]
        target_index = int(self.target_team)
        buyer_index = int(self.buyer.team)
        self.model.add(
            self.final_raw[target_index]
            == self.prefix_target_raw
            + trick_points[1]
            + trick_points[2]
            + self.scoring_domain.last_trick_bonus_raw
        )
        self.model.add(
            self.final_raw[buyer_index] == prefix_buyer_raw + trick_points[0]
        )
        self.model.add(sum(self.final_raw) == 130)

        raw_to_game = [
            self.scoring_domain.convert_raw(raw, Contract.SUN) for raw in range(131)
        ]
        self.final_game = [
            self.model.new_int_var(0, max(raw_to_game), f"final_game_{team.name.lower()}")
            for team in Team
        ]
        for team in Team:
            self.model.add_element(
                self.final_raw[int(team)], raw_to_game, self.final_game[int(team)]
            )
        self.model.add(
            self.final_game[target_index]
            > self.final_game[buyer_index] + self.project_value
        )

    def _model_hash(self) -> str:
        return canonical_sha256(
            {
                "schemaVersion": 1,
                "kind": "ortools_flagship_tail_v1",
                "rulesHash": self.rules.sha256,
                "structuralDomainIrHash": self.structural_domain.sha256,
                "playDomainIrHash": self.play_domain.sha256,
                "scoringDomainIrHash": self.scoring_domain.sha256,
                "projectDomainIrHash": self.rules.project_domain.sha256,
                "actingSeat": self.acting_seat.name.lower(),
                "exactNoGoods": [list(item) for item in self.no_goods],
                "excludedSuitSymmetryKeys": list(self.excluded_symmetry_keys),
                "cpModelProtoHash": hashlib.sha256(
                    str(self.model.proto).encode("utf-8")
                ).hexdigest(),
            }
        )

    def _extract_candidate(
        self,
        solver: cp_model.CpSolver,
        *,
        model_hash: str,
    ) -> FlagshipTailCandidateV1:
        card_ids = tuple(int(solver.value(card)) for card in self.cards)
        cards = tuple(Card(card_id) for card_id in card_ids)
        plays = tuple(
            FlagshipTailPlayV1(
                6 + trick,
                ordinal + 1,
                self.orders[trick][ordinal],
                cards[self._flat(trick, ordinal)],
            )
            for trick in range(_TAIL_TRICKS)
            for ordinal in range(_PLAYS_PER_TRICK)
        )
        root = cards[self._flat_for(0, self.acting_seat)]
        root_actions = tuple(
            sorted(
                (
                    cards[self._flat_for(trick, self.acting_seat)]
                    for trick in range(_TAIL_TRICKS)
                ),
                key=lambda card: card.id,
            )
        )
        prefix_target = int(solver.value(self.prefix_target_raw))
        prefix = [0, 0]
        prefix[int(self.target_team)] = prefix_target
        tail_point_total = sum(int(solver.value(point)) for point in self.points)
        prefix[int(self.buyer.team)] = 120 - tail_point_total - prefix_target
        final_raw = tuple(int(solver.value(value)) for value in self.final_raw)
        final_game = tuple(int(solver.value(value)) for value in self.final_game)
        projects = [0, 0]
        projects[int(self.buyer.team)] = self.project_value
        exact_hash = canonical_sha256(
            {"schemaVersion": 1, "kind": "tail_exact_no_good", "cards": list(card_ids)}
        )
        symmetry_key = canonical_sha256(
            {
                "schemaVersion": 1,
                "kind": "tail_suit_symmetry",
                "pattern": list(_canonical_suit_pattern(cards)),
            }
        )
        diversity_key = canonical_sha256(
            {
                "schemaVersion": 1,
                "actingSeat": self.acting_seat.name.lower(),
                "sacrificeSuit": root.suit.name.lower(),
                "winningRanks": [
                    cards[self._flat_for(trick, self.acting_seat)].rank.name.lower()
                    for trick in (1, 2)
                ],
                "suitSymmetryKey": symmetry_key,
            }
        )
        core = {
            "schemaVersion": 1,
            "status": "tail_candidate",
            "modelHash": model_hash,
            "actingSeat": self.acting_seat.name.lower(),
            "buyer": self.buyer.name.lower(),
            "cards": list(card_ids),
            "winners": [seat.name.lower() for seat in self.winners],
            "prefixRawPoints": prefix,
            "finalRawPoints": list(final_raw),
            "finalGamePoints": list(final_game),
            "projectGamePoints": projects,
            "exactNoGoodHash": exact_hash,
            "suitSymmetryKey": symmetry_key,
            "diversityKey": diversity_key,
        }
        return FlagshipTailCandidateV1(
            "tail_candidate",
            Contract.SUN,
            self.buyer,
            self.acting_seat,
            self.target_team,
            plays,
            self.winners,
            root,
            root_actions,
            tuple(prefix),
            final_raw,
            final_game,
            tuple(projects),
            card_ids,
            exact_hash,
            symmetry_key,
            diversity_key,
            model_hash,
            canonical_sha256(core),
            int(solver.num_branches),
            int(solver.num_conflicts),
        )

    def solve(
        self,
        *,
        seed: int = 0,
        maximum_seconds: float = 15.0,
    ) -> FlagshipTailSearchResultV1:
        if type(seed) is not int:
            raise TypeError("flagship tail seed must be an integer")
        if type(maximum_seconds) not in {int, float} or isinstance(maximum_seconds, bool):
            raise TypeError("flagship tail maximum_seconds must be numeric")
        if maximum_seconds <= 0:
            raise ValueError("flagship tail maximum_seconds must be positive")

        model_hash = self._model_hash()
        skipped: list[str] = []
        branches = conflicts = 0
        deadline = time.monotonic() + float(maximum_seconds)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return FlagshipTailSearchResultV1(
                    "not_found_within_budget",
                    "unknown",
                    None,
                    model_hash,
                    tuple(
                        canonical_sha256(
                            {"schemaVersion": 1, "kind": "tail_exact_no_good", "cards": list(item)}
                        )
                        for item in self.no_goods
                    ),
                    tuple(skipped),
                    branches,
                    conflicts,
                )
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.random_seed = seed & 0x7FFFFFFF
            solver.parameters.search_branching = cp_model.FIXED_SEARCH
            solver.parameters.max_time_in_seconds = remaining
            status = solver.solve(self.model)
            branches += int(solver.num_branches)
            conflicts += int(solver.num_conflicts)
            solver_status = solver.status_name(status).lower()
            if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
                return FlagshipTailSearchResultV1(
                    "infeasible" if status == cp_model.INFEASIBLE else "not_found_within_budget",
                    solver_status,
                    None,
                    model_hash,
                    tuple(
                        canonical_sha256(
                            {"schemaVersion": 1, "kind": "tail_exact_no_good", "cards": list(item)}
                        )
                        for item in self.no_goods
                    ),
                    tuple(skipped),
                    branches,
                    conflicts,
                )
            candidate = self._extract_candidate(solver, model_hash=model_hash)
            if candidate.suit_symmetry_key not in self.excluded_symmetry_keys:
                return FlagshipTailSearchResultV1(
                    "tail_candidate",
                    solver_status,
                    candidate,
                    model_hash,
                    tuple(
                        canonical_sha256(
                            {"schemaVersion": 1, "kind": "tail_exact_no_good", "cards": list(item)}
                        )
                        for item in self.no_goods
                    ),
                    tuple(skipped),
                    branches,
                    conflicts,
                )
            skipped.append(candidate.suit_symmetry_key)
            self.model.add_forbidden_assignments(self.cards, [candidate.exact_no_good])


def find_flagship_tail_variants(
    *,
    rules: RuleBook,
    count: int,
    acting_seat: Seat = Seat.WEST,
    seed: int = 0,
    maximum_seconds_per_variant: float = 15.0,
) -> tuple[FlagshipTailCandidateV1, ...]:
    if type(count) is not int or not 1 <= count <= 64:
        raise ValueError("flagship tail variant count must be an integer from 1 through 64")
    candidates: list[FlagshipTailCandidateV1] = []
    no_goods: list[tuple[int, ...]] = []
    symmetry_keys: list[str] = []
    for _ in range(count):
        result = OrToolsFlagshipTailSearchV1(
            rules=rules,
            acting_seat=acting_seat,
            excluded_exact_tails=no_goods,
            excluded_suit_symmetry_keys=symmetry_keys,
        ).solve(seed=seed, maximum_seconds=maximum_seconds_per_variant)
        if result.candidate is None:
            break
        candidates.append(result.candidate)
        no_goods.append(result.candidate.exact_no_good)
        symmetry_keys.append(result.candidate.suit_symmetry_key)
    return tuple(sorted(candidates, key=lambda candidate: candidate.candidate_hash))


__all__ = [
    "FlagshipTailCandidateV1",
    "FlagshipTailPlayV1",
    "FlagshipTailSearchResultV1",
    "OrToolsFlagshipTailSearchV1",
    "find_flagship_tail_variants",
]
