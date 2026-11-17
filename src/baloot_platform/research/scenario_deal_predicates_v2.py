"""Exact, engine-free CP-SAT lowerings for the V1 deal predicates."""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Rank, Seat, Suit, Team

from .ortools_scenario_model import OrToolsScenarioModelV1
from .scenario_contracts import ScenarioPredicateV1, canonical_sha256
from .scenario_registry import (
    ScenarioPredicateRegistryV1,
    built_in_scenario_predicates_v1,
)


_ABSENT = 4
_HIGH_RANKS = frozenset({Rank.ACE, Rank.TEN, Rank.KING})
_LOW_RANKS = frozenset({Rank.SEVEN, Rank.EIGHT, Rank.NINE})
NATIVE_SEAT_RELATION_IMPLEMENTATION = "cp_sat.seat_relation_modulo_delta_v1"


@dataclass(frozen=True, slots=True)
class DealPredicateRegistrationV2:
    kind: str
    evidence_extractor_id: str


DEAL_PREDICATE_REGISTRATIONS_V2 = tuple(
    DealPredicateRegistrationV2(kind, f"cp_sat.{kind.replace('.', '_')}_evidence_v2")
    for kind in (
        "deal.card_owner",
        "deal.card_allowed",
        "deal.card_forbidden",
        "deal.suit_count",
        "deal.rank_count",
        "deal.high_card_count",
        "deal.low_card_count",
        "deal.purchase_card",
        "deal.purchase_suit",
        "deal.remaining_card_owner",
        "deal.remaining_suit_count",
        "deal.remaining_rank_count",
        "deal.team_high_card_balance",
        "deal.seat_relation",
    )
)


@dataclass(frozen=True, slots=True)
class DealPredicateModelContextV2:
    """Deal views bound to one existing ``OrToolsScenarioModelV1``."""

    scenario: OrToolsScenarioModelV1
    remaining_owner_of_card: tuple[cp_model.IntVar, ...] | None = None
    context_id: str = "deal"
    initial_owner_of_card: tuple[cp_model.IntVar, ...] = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, OrToolsScenarioModelV1):
            raise TypeError("deal predicate context requires OrToolsScenarioModelV1")
        if type(self.context_id) is not str or not self.context_id:
            raise ValueError("deal predicate context_id must be non-empty")
        remaining = self.remaining_owner_of_card
        if remaining is not None:
            if type(remaining) is not tuple or len(remaining) != 32:
                raise ValueError("remaining ownership must contain exactly 32 variables")
            if any(not isinstance(item, cp_model.IntVar) for item in remaining):
                raise TypeError("remaining ownership entries must be CP-SAT variables")

        layout = DealLayoutV1.build(
            self.scenario.rules.structural_domain,
            dealer=self.scenario.dealer,
            buyer=Seat.NORTH,
        )
        owner_at_position = [_ABSENT] * 32
        for seat in Seat:
            for position in layout.positions_for(seat, final=False):
                owner_at_position[position] = int(seat)
        initial: list[cp_model.IntVar] = []
        for card, position in enumerate(self.scenario.position_of_card):
            owner = self.scenario.model.new_int_var(0, _ABSENT, f"initial_owner_{card}")
            self.scenario.model.add_element(position, owner_at_position, owner)
            initial.append(owner)
        object.__setattr__(self, "initial_owner_of_card", tuple(initial))
        core = {
            "schemaVersion": 2,
            "contextId": self.context_id,
            "rulesHash": self.scenario.rules.sha256,
            "dealer": self.scenario.dealer.name.lower(),
            "requiredBuyer": (
                self.scenario.required_buyer.name.lower()
                if self.scenario.required_buyer is not None
                else None
            ),
            "remainingOwnerVariableIndexes": (
                [item.index for item in remaining] if remaining is not None else None
            ),
            "cpModelProtoHash": hashlib.sha256(
                str(self.scenario.model.proto).encode("utf-8")
            ).hexdigest(),
        }
        object.__setattr__(self, "sha256", canonical_sha256(core))

    @property
    def model(self) -> cp_model.CpModel:
        return self.scenario.model

    def owners(self, stage: str) -> tuple[cp_model.IntVar, ...]:
        if stage == "initial_five":
            return self.initial_owner_of_card
        if stage == "post_purchase":
            return tuple(self.scenario.owner_of_card)
        if stage == "remaining":
            if self.remaining_owner_of_card is None:
                raise ValueError("remaining deal predicates require bound play ownership")
            return self.remaining_owner_of_card
        raise ValueError(f"unsupported deal ownership stage: {stage}")


@dataclass(frozen=True, slots=True)
class DealPredicateLoweringV2:
    predicate_id: str
    kind: str
    literal: cp_model.IntVar
    assumption: cp_model.IntVar | None
    evidence_extractor_id: str
    lowering_hash: str
    _extract: Callable[[cp_model.CpSolver], Mapping[str, Any]] = field(
        repr=False,
        compare=False,
    )

    def extract(self, solver: cp_model.CpSolver) -> dict[str, Any]:
        core = {
            "schemaVersion": 2,
            "predicateId": self.predicate_id,
            "kind": self.kind,
            "satisfied": bool(solver.boolean_value(self.literal)),
            "evidenceExtractor": self.evidence_extractor_id,
            "loweringHash": self.lowering_hash,
            "observed": copy.deepcopy(dict(self._extract(solver))),
        }
        return {**core, "sha256": canonical_sha256(core)}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _owner_literal(
    model: cp_model.CpModel,
    owner: cp_model.IntVar,
    allowed: Sequence[int],
    name: str,
    *,
    upper_bound: int = _ABSENT,
) -> cp_model.IntVar:
    allowed_values = frozenset(int(item) for item in allowed)
    literal = model.new_bool_var(name)
    model.add_allowed_assignments(
        [owner, literal],
        [(value, int(value in allowed_values)) for value in range(upper_bound + 1)],
    )
    return literal


def _range_literal(
    model: cp_model.CpModel,
    count: cp_model.IntVar,
    *,
    upper_bound: int,
    minimum: int,
    maximum: int,
    name: str,
) -> cp_model.IntVar:
    if minimum > maximum:
        raise ValueError("deal predicate minimum cannot exceed maximum")
    literal = model.new_bool_var(name)
    model.add_allowed_assignments(
        [count, literal],
        [
            (value, int(minimum <= value <= maximum))
            for value in range(upper_bound + 1)
        ],
    )
    return literal


def _owner_name(value: int) -> str | None:
    return None if value == _ABSENT else Seat(value).name.lower()


class DealPredicateLoweringRegistryV2:
    """Closed lowerer for exactly the fourteen registered deal predicates."""

    def __init__(self) -> None:
        self._schema_registry = ScenarioPredicateRegistryV1()
        self._definitions = {
            definition.kind: definition
            for definition in built_in_scenario_predicates_v1()
            if definition.kind.startswith("deal.")
        }
        self._registrations = {
            item.kind: item for item in DEAL_PREDICATE_REGISTRATIONS_V2
        }
        if set(self._definitions) != set(self._registrations):
            raise AssertionError("deal predicate schema and lowering registrations drifted")

    @property
    def registrations(self) -> tuple[DealPredicateRegistrationV2, ...]:
        return DEAL_PREDICATE_REGISTRATIONS_V2

    @property
    def manifest_hash(self) -> str:
        return canonical_sha256(
            {
                "schemaVersion": 2,
                "registrations": [
                    {
                        "kind": item.kind,
                        "definitionHash": self._definitions[item.kind].sha256,
                        "evidenceExtractor": item.evidence_extractor_id,
                    }
                    for item in self.registrations
                ],
            }
        )

    def lower_predicate(
        self,
        predicate: ScenarioPredicateV1,
        context: DealPredicateModelContextV2,
        *,
        enforce_as_assumption: bool = False,
        native_seat_relation: bool = False,
    ) -> DealPredicateLoweringV2:
        if type(enforce_as_assumption) is not bool:
            raise TypeError("enforce_as_assumption must be boolean")
        if type(native_seat_relation) is not bool:
            raise TypeError("native_seat_relation must be boolean")
        if native_seat_relation and predicate.kind != "deal.seat_relation":
            raise ValueError("native seat relation requires deal.seat_relation")
        definition = self._schema_registry.require(predicate)
        registration = self._registrations.get(predicate.kind)
        if definition.kind not in self._definitions or registration is None:
            raise ValueError(
                f"predicate has no exact CP-SAT deal lowering: {predicate.kind}"
            )
        arguments = dict(predicate.arguments)
        minimum = arguments.get("minimum")
        maximum = arguments.get("maximum")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("deal predicate minimum cannot exceed maximum")

        name = _safe_name(predicate.predicate_id.lower())
        literal, extractor = (
            self._lower_native_seat_relation(arguments, context, name)
            if native_seat_relation
            else self._lower(predicate.kind, arguments, context, name)
        )
        assumption = None
        if enforce_as_assumption:
            assumption = context.model.new_bool_var(f"assume_{name}")
            context.model.add(literal == 1).only_enforce_if(assumption)
            context.model.add_assumption(assumption)
        lowering_hash = canonical_sha256(
            {
                "schemaVersion": 2,
                "predicateHash": predicate.sha256,
                "contextHash": context.sha256,
                "definitionHash": definition.sha256,
                "evidenceExtractor": registration.evidence_extractor_id,
                **({"nativeImplementation": NATIVE_SEAT_RELATION_IMPLEMENTATION}
                   if native_seat_relation else {}),
            }
        )
        return DealPredicateLoweringV2(
            predicate.predicate_id,
            predicate.kind,
            literal,
            assumption,
            registration.evidence_extractor_id,
            lowering_hash,
            extractor,
        )

    def _lower_native_seat_relation(
        self,
        arguments: Mapping[str, Any],
        context: DealPredicateModelContextV2,
        name: str,
    ) -> tuple[cp_model.IntVar, Callable[[cp_model.CpSolver], Mapping[str, Any]]]:
        seat = Seat.parse(arguments["seat"])
        other = Seat.parse(arguments["otherSeat"])
        relation = arguments["relation"]
        model = context.model
        seat_var = model.new_int_var(0, 3, f"{name}_seat")
        other_var = model.new_int_var(0, 3, f"{name}_other_seat")
        model.add(seat_var == int(seat))
        model.add(other_var == int(other))
        delta = model.new_int_var(0, 3, f"{name}_delta")
        model.add_modulo_equality(delta, other_var - seat_var + 4, 4)
        if relation == "opponent":
            parity = model.new_int_var(0, 1, f"{name}_parity")
            model.add_modulo_equality(parity, delta, 2)
            value, expected = parity, 1
        else:
            value, expected = delta, {"same": 0, "next": 1, "partner": 2, "previous": 3}[relation]
        literal = model.new_bool_var(f"{name}_result")
        model.add(value == expected).only_enforce_if(literal)
        model.add(value != expected).only_enforce_if(literal.Not())
        return literal, lambda solver: {
            "seat": seat.name.lower(),
            "otherSeat": other.name.lower(),
            "relation": relation,
            "matches": bool(solver.boolean_value(literal)),
        }

    def _lower(
        self,
        kind: str,
        arguments: Mapping[str, Any],
        context: DealPredicateModelContextV2,
        name: str,
    ) -> tuple[cp_model.IntVar, Callable[[cp_model.CpSolver], Mapping[str, Any]]]:
        if kind in {"deal.card_owner", "deal.card_allowed", "deal.card_forbidden"}:
            card = Card(arguments["card"])
            owners = context.owners(str(arguments["handStage"]))
            expected = (
                tuple(int(Seat.parse(seat)) for seat in arguments["seats"])
                if kind == "deal.card_allowed"
                else (int(Seat.parse(arguments["seat"])),)
            )
            if kind == "deal.card_forbidden":
                expected = tuple(value for value in range(_ABSENT + 1) if value not in expected)
            literal = _owner_literal(context.model, owners[card.id], expected, f"{name}_result")
            return literal, lambda solver: {
                "card": card.id,
                "owner": _owner_name(int(solver.value(owners[card.id]))),
                "handStage": arguments["handStage"],
            }

        if kind in {
            "deal.suit_count",
            "deal.rank_count",
            "deal.high_card_count",
            "deal.low_card_count",
            "deal.remaining_suit_count",
            "deal.remaining_rank_count",
        }:
            remaining = kind.startswith("deal.remaining_")
            stage = "remaining" if remaining else str(arguments["handStage"])
            owners = context.owners(stage)
            seat = Seat.parse(arguments["seat"])
            if kind.endswith("suit_count"):
                cards = tuple(card for card in range(32) if Card(card).suit == Suit.parse(arguments["suit"]))
            elif kind.endswith("rank_count"):
                cards = tuple(card for card in range(32) if Card(card).rank == Rank.parse(arguments["rank"]))
            elif kind == "deal.high_card_count":
                cards = tuple(card for card in range(32) if Card(card).rank in _HIGH_RANKS)
            else:
                cards = tuple(card for card in range(32) if Card(card).rank in _LOW_RANKS)
            matches = tuple(
                _owner_literal(context.model, owners[card], (int(seat),), f"{name}_card_{card}")
                for card in cards
            )
            count = context.model.new_int_var(0, len(cards), f"{name}_count")
            context.model.add(count == sum(matches))
            literal = _range_literal(
                context.model,
                count,
                upper_bound=len(cards),
                minimum=int(arguments["minimum"]),
                maximum=int(arguments["maximum"]),
                name=f"{name}_result",
            )
            return literal, lambda solver: {
                "seat": seat.name.lower(),
                "count": int(solver.value(count)),
                **({} if remaining else {"handStage": stage}),
            }

        if kind == "deal.purchase_card":
            card = Card(arguments["card"])
            literal = _owner_literal(
                context.model,
                context.scenario.purchase_card,
                (card.id,),
                f"{name}_result",
                upper_bound=31,
            )
            return literal, lambda solver: {
                "card": int(solver.value(context.scenario.purchase_card))
            }

        if kind == "deal.purchase_suit":
            suit = Suit.parse(arguments["suit"])
            literal = _owner_literal(
                context.model,
                context.scenario.purchase_suit,
                (int(suit),),
                f"{name}_result",
                upper_bound=3,
            )
            return literal, lambda solver: {
                "suit": Suit(int(solver.value(context.scenario.purchase_suit))).name.lower()
            }

        if kind == "deal.remaining_card_owner":
            card = Card(arguments["card"])
            owners = context.owners("remaining")
            literal = _owner_literal(
                context.model,
                owners[card.id],
                (int(Seat.parse(arguments["seat"])),),
                f"{name}_result",
            )
            return literal, lambda solver: {
                "card": card.id,
                "owner": _owner_name(int(solver.value(owners[card.id]))),
            }

        if kind == "deal.team_high_card_balance":
            team = Team[str(arguments["team"]).upper()]
            stage = str(arguments["handStage"])
            owners = context.owners(stage)
            seats = tuple(seat for seat in Seat if seat.team == team)
            counts: list[cp_model.IntVar] = []
            for seat in seats:
                matches = tuple(
                    _owner_literal(
                        context.model,
                        owners[card],
                        (int(seat),),
                        f"{name}_{seat.name.lower()}_{card}",
                    )
                    for card in range(32)
                    if Card(card).rank in _HIGH_RANKS
                )
                count = context.model.new_int_var(0, 12, f"{name}_{seat.name.lower()}_count")
                context.model.add(count == sum(matches))
                counts.append(count)
            difference = context.model.new_int_var(0, 12, f"{name}_difference")
            context.model.add_abs_equality(difference, counts[0] - counts[1])
            literal = _range_literal(
                context.model,
                difference,
                upper_bound=12,
                minimum=0,
                maximum=int(arguments["maximumDifference"]),
                name=f"{name}_result",
            )
            return literal, lambda solver: {
                "team": team.name.lower(),
                "counts": {
                    seat.name.lower(): int(solver.value(count))
                    for seat, count in zip(seats, counts, strict=True)
                },
                "difference": int(solver.value(difference)),
                "handStage": stage,
            }

        if kind == "deal.seat_relation":
            seat = Seat.parse(arguments["seat"])
            other = Seat.parse(arguments["otherSeat"])
            relation = str(arguments["relation"])
            matches = {
                "same": other == seat,
                "partner": other == seat.partner,
                "next": other == seat.next(),
                "previous": other == seat.previous(),
                "opponent": other.team != seat.team,
            }[relation]
            literal = context.model.new_constant(int(matches))
            return literal, lambda solver: {
                "seat": seat.name.lower(),
                "otherSeat": other.name.lower(),
                "relation": relation,
                "matches": bool(matches),
            }

        raise AssertionError(f"unregistered deal predicate: {kind}")


__all__ = [
    "DEAL_PREDICATE_REGISTRATIONS_V2",
    "DealPredicateLoweringRegistryV2",
    "DealPredicateLoweringV2",
    "DealPredicateModelContextV2",
    "DealPredicateRegistrationV2",
]
