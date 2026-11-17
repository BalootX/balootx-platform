"""Exact CP-SAT lowerings for reusable flagship scenario predicates.

The caller supplies an engine-free model context whose variables are already
bound to the authoritative Domain IR.  This module only reifies typed scenario
claims and extracts immutable solver evidence; it does not execute or import
the Baloot runtime engine.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat, Team

from .ortools_scenario_model import reify_scenario_expression
from .scenario_contracts import (
    ScenarioExpressionV1,
    ScenarioPredicateV1,
    canonical_sha256,
)
from .scenario_registry_v2 import (
    CardSelectorV1,
    ScenarioPredicateDefinitionV2,
    _registered_flagship_predicate_definitions_v2,
    scenario_predicate_definitions_v2,
)


@dataclass(frozen=True, slots=True)
class CpSatActionVariablesV1:
    """One action slot bound by the supplying CP-SAT domain model."""

    action_id: str
    card: cp_model.IntVar
    actor: cp_model.IntVar
    original_owner: cp_model.IntVar
    ekka: cp_model.IntVar
    present: cp_model.IntVar
    winning: cp_model.IntVar | None = None
    value: cp_model.IntVar | None = None

    def __post_init__(self) -> None:
        if type(self.action_id) is not str or not self.action_id:
            raise ValueError("CP-SAT action variables require a stable action_id")


@dataclass(frozen=True, slots=True)
class CpSatTrickVariablesV1:
    trick_index: int
    actions: tuple[CpSatActionVariablesV1, ...]
    winner: CpSatActionVariablesV1

    def __post_init__(self) -> None:
        if type(self.trick_index) is not int or not 0 <= self.trick_index <= 7:
            raise ValueError("CP-SAT trick index must be between zero and seven")
        if type(self.actions) is not tuple or len(self.actions) != 4:
            raise ValueError("CP-SAT trick context requires exactly four action slots")


@dataclass(frozen=True, slots=True)
class CpSatPredicateModelContextV1:
    """Finite CP variables required by the registered predicate lowerings."""

    model: cp_model.CpModel
    context_id: str
    tricks: Mapping[int, CpSatTrickVariablesV1]
    root_actions: tuple[CpSatActionVariablesV1, ...]
    proof_complete: cp_model.IntVar
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, cp_model.CpModel):
            raise TypeError("predicate lowering context requires a CpModel")
        if type(self.context_id) is not str or not self.context_id:
            raise ValueError("predicate lowering context_id must be non-empty")
        if not isinstance(self.tricks, Mapping):
            raise TypeError("predicate lowering tricks must be a mapping")
        normalized = dict(self.tricks)
        if any(key != value.trick_index for key, value in normalized.items()):
            raise ValueError("predicate lowering trick keys do not match trick indices")
        if type(self.root_actions) is not tuple:
            raise TypeError("predicate lowering root_actions must be a tuple")
        action_ids = [action.action_id for action in self.root_actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("predicate lowering root action IDs must be unique")
        if any(action.winning is None for action in self.root_actions):
            raise ValueError("every root action requires an exact winning literal")
        object.__setattr__(self, "tricks", MappingProxyType(normalized))
        core = {
            "schemaVersion": 1,
            "contextId": self.context_id,
            "tricks": [
                {
                    "trickIndex": trick.trick_index,
                    "actionIds": [action.action_id for action in trick.actions],
                    "winnerActionId": trick.winner.action_id,
                }
                for trick in sorted(normalized.values(), key=lambda row: row.trick_index)
            ],
            "rootActionIds": action_ids,
        }
        object.__setattr__(self, "sha256", canonical_sha256(core))


@dataclass(frozen=True, slots=True)
class CpSatPredicateLoweringV1:
    predicate_id: str
    kind: str
    literal: cp_model.IntVar
    evidence_extractor_id: str
    lowering_hash: str
    _extract: Callable[[cp_model.CpSolver], Mapping[str, Any]] = field(
        repr=False,
        compare=False,
    )

    def extract(self, solver: cp_model.CpSolver) -> dict[str, Any]:
        observed = copy.deepcopy(dict(self._extract(solver)))
        core = {
            "schemaVersion": 1,
            "predicateId": self.predicate_id,
            "kind": self.kind,
            "satisfied": bool(solver.boolean_value(self.literal)),
            "evidenceExtractor": self.evidence_extractor_id,
            "loweringHash": self.lowering_hash,
            "observed": observed,
        }
        return {**core, "sha256": canonical_sha256(core)}


def _action_observation(
    solver: cp_model.CpSolver,
    action: CpSatActionVariablesV1,
) -> dict[str, Any]:
    card = Card(int(solver.value(action.card)))
    actor = Seat(int(solver.value(action.actor)))
    owner = Seat(int(solver.value(action.original_owner)))
    value = {
        "actionId": action.action_id,
        "card": card.to_dict(),
        "actor": actor.name.lower(),
        "originalOwner": owner.name.lower(),
        "originalOwnerTeam": owner.team.name.lower(),
        "ekka": bool(solver.boolean_value(action.ekka)),
        "present": bool(solver.boolean_value(action.present)),
    }
    if action.winning is not None:
        value["winning"] = bool(solver.boolean_value(action.winning))
    if action.value is not None:
        value["value"] = int(solver.value(action.value))
    return value


def _selector_matches(
    selector: CardSelectorV1,
    *,
    card_id: int,
    actor_id: int,
    owner_id: int,
    ekka: bool,
) -> bool:
    card = Card(card_id)
    actor = Seat(actor_id)
    owner = Seat(owner_id)
    if selector.cards and card_id not in selector.cards:
        return False
    if selector.ranks and card.rank.name.lower() not in selector.ranks:
        return False
    if selector.suits and card.suit.name.lower() not in selector.suits:
        return False
    if selector.ekka == "required" and not ekka:
        return False
    if selector.ekka == "forbidden" and ekka:
        return False
    if selector.owner_relation == "actor" and owner != actor:
        return False
    if selector.owner_relation == "partner" and owner != actor.partner:
        return False
    if selector.owner_relation == "opponent_team" and owner.team == actor.team:
        return False
    if (
        selector.owner_relation == "seat"
        and owner != Seat.parse(selector.owner_seat)
    ):
        return False
    if (
        selector.owner_relation == "team"
        and owner.team != Team[selector.owner_team.upper()]
    ):
        return False
    return True


def _selector_literal(
    context: CpSatPredicateModelContextV1,
    action: CpSatActionVariablesV1,
    selector: CardSelectorV1,
    *,
    name: str,
) -> cp_model.IntVar:
    raw_match = context.model.new_bool_var(f"{name}_selector")
    rows = [
        [card_id, actor_id, owner_id, int(ekka), int(matched)]
        for card_id in range(32)
        for actor_id in range(4)
        for owner_id in range(4)
        for ekka in (False, True)
        for matched in (
            _selector_matches(
                selector,
                card_id=card_id,
                actor_id=actor_id,
                owner_id=owner_id,
                ekka=ekka,
            ),
        )
    ]
    context.model.add_allowed_assignments(
        [
            action.card,
            action.actor,
            action.original_owner,
            action.ekka,
            raw_match,
        ],
        rows,
    )
    result = context.model.new_bool_var(f"{name}_present_match")
    context.model.add_min_equality(result, [action.present, raw_match])
    return result


def _seat_literal(
    context: CpSatPredicateModelContextV1,
    seat: cp_model.IntVar,
    *,
    expected_seat: str | None = None,
    expected_team: str | None = None,
    name: str,
) -> cp_model.IntVar:
    if expected_seat is None and expected_team is None:
        return context.model.new_constant(1)
    literal = context.model.new_bool_var(name)
    rows: list[list[int]] = []
    for seat_id in range(4):
        current = Seat(seat_id)
        matches = (
            current == Seat.parse(expected_seat)
            if expected_seat is not None
            else current.team == Team[expected_team.upper()]
        )
        rows.append([seat_id, int(matches)])
    context.model.add_allowed_assignments([seat, literal], rows)
    return literal


def _range_literal(
    model: cp_model.CpModel,
    literals: tuple[cp_model.IntVar, ...],
    *,
    minimum: int,
    maximum: int,
    name: str,
) -> tuple[cp_model.IntVar, cp_model.IntVar]:
    count = model.new_int_var(0, len(literals), f"{name}_count")
    model.add(count == sum(literals))
    result = model.new_bool_var(f"{name}_range")
    model.add_allowed_assignments(
        [count, result],
        [
            [value, int(minimum <= value <= maximum)]
            for value in range(len(literals) + 1)
        ],
    )
    return result, count


class ScenarioPredicateLoweringRegistryV2:
    """Closed registry binding every exact kind to lowering and evidence."""

    _EXTRACTORS = MappingProxyType(
        {
            "play.action_matches": "cp_sat.play_action_matches_evidence_v1",
            "play.trick_winner_matches": "cp_sat.play_trick_winner_evidence_v1",
            "play.trick_card_count": "cp_sat.play_trick_card_count_evidence_v1",
            "proof.winning_root_action_matches": "cp_sat.winning_root_match_evidence_v1",
        }
    )

    def __init__(self) -> None:
        self._lowerers = {
            "play.action_matches": self._lower_play_action_matches,
            "play.trick_winner_matches": self._lower_play_trick_winner_matches,
            "play.trick_card_count": self._lower_play_trick_card_count,
            "proof.winning_root_action_matches": self._lower_winning_root_matches,
        }
        if set(self._lowerers) != set(self._EXTRACTORS):
            raise AssertionError("predicate lowering and evidence registries drifted")
        definitions = _registered_flagship_predicate_definitions_v2(self._EXTRACTORS)
        self._definitions = {definition.kind: definition for definition in definitions}
        if set(self._definitions) != set(self._lowerers):
            raise AssertionError("predicate catalog and lowering registries drifted")
        for definition in self._definitions.values():
            definition.require_publishable()

    @property
    def definitions(self) -> tuple[ScenarioPredicateDefinitionV2, ...]:
        return tuple(self._definitions[kind] for kind in sorted(self._definitions))

    @property
    def catalog_definitions(self) -> tuple[ScenarioPredicateDefinitionV2, ...]:
        rows = {
            definition.kind: definition
            for definition in scenario_predicate_definitions_v2()
        }
        rows.update(self._definitions)
        return tuple(rows[kind] for kind in sorted(rows))

    @property
    def manifest_hash(self) -> str:
        return canonical_sha256(
            {
                "schemaVersion": 1,
                "registrations": [
                    {
                        "kind": kind,
                        "definitionHash": self._definitions[kind].sha256,
                        "evidenceExtractor": self._EXTRACTORS[kind],
                    }
                    for kind in sorted(self._definitions)
                ],
            }
        )

    def definition(self, kind: str) -> ScenarioPredicateDefinitionV2 | None:
        return self._definitions.get(str(kind))

    def lower_predicate(
        self,
        predicate: ScenarioPredicateV1,
        context: CpSatPredicateModelContextV1,
    ) -> CpSatPredicateLoweringV1:
        definition = self.definition(predicate.kind)
        lowerer = self._lowerers.get(predicate.kind)
        if definition is None or lowerer is None or definition.version != predicate.version:
            raise ValueError(
                f"predicate has no exact CP-SAT lowering: {predicate.kind}@{predicate.version}"
            )
        definition.validate_arguments(predicate.arguments)
        return lowerer(predicate, context)

    def lower_expression(
        self,
        expression: ScenarioExpressionV1,
        context: CpSatPredicateModelContextV1,
    ) -> tuple[cp_model.IntVar, tuple[CpSatPredicateLoweringV1, ...]]:
        predicates = expression.predicates()
        ids = [predicate.predicate_id for predicate in predicates]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario expression predicate IDs must be unique")

        # Complete preflight happens before the first model mutation.
        for predicate in predicates:
            definition = self.definition(predicate.kind)
            if definition is None or predicate.kind not in self._lowerers:
                raise ValueError(
                    f"predicate has no exact CP-SAT lowering: {predicate.kind}"
                )
            if definition.version != predicate.version:
                raise ValueError(
                    f"predicate has no exact CP-SAT lowering: "
                    f"{predicate.kind}@{predicate.version}"
                )
            definition.validate_arguments(predicate.arguments)
        self._validate_safe_complements(expression)

        lowerings = tuple(
            self.lower_predicate(predicate, context) for predicate in predicates
        )
        root = reify_scenario_expression(
            context.model,
            expression,
            {item.predicate_id: item.literal for item in lowerings},
        )
        return root, lowerings

    def _validate_safe_complements(self, expression: ScenarioExpressionV1) -> None:
        def visit(node: ScenarioExpressionV1) -> None:
            if node.op == "not":
                unsafe = sorted(
                    predicate.kind
                    for predicate in node.predicates()
                    if not self._definitions[predicate.kind].safe_complement
                )
                if unsafe:
                    raise ValueError(
                        "scenario not uses predicates without a safe complement: "
                        + ", ".join(unsafe)
                    )
            for child in node.children:
                visit(child)

        visit(expression)

    def _lower_play_action_matches(
        self,
        predicate: ScenarioPredicateV1,
        context: CpSatPredicateModelContextV1,
    ) -> CpSatPredicateLoweringV1:
        arguments = dict(predicate.arguments)
        trick_index = int(arguments["trickIndex"])
        position = int(arguments["position"])
        trick = context.tricks.get(trick_index)
        if trick is None:
            raise ValueError(f"CP-SAT context does not contain trick {trick_index}")
        action = trick.actions[position]
        selector = CardSelectorV1.build(arguments["selector"])
        match = _selector_literal(
            context,
            action,
            selector,
            name=f"{predicate.predicate_id}_action",
        )
        actor_match = _seat_literal(
            context,
            action.actor,
            expected_seat=arguments.get("actingSeat"),
            name=f"{predicate.predicate_id}_actor",
        )
        literal = context.model.new_bool_var(f"{predicate.predicate_id}_result")
        context.model.add_min_equality(literal, [match, actor_match])
        return self._lowering(
            predicate,
            context,
            literal,
            lambda solver: {
                "selector": selector.to_dict(),
                "action": _action_observation(solver, action),
            },
        )

    def _lower_play_trick_winner_matches(
        self,
        predicate: ScenarioPredicateV1,
        context: CpSatPredicateModelContextV1,
    ) -> CpSatPredicateLoweringV1:
        arguments = dict(predicate.arguments)
        trick_index = int(arguments["trickIndex"])
        trick = context.tricks.get(trick_index)
        if trick is None:
            raise ValueError(f"CP-SAT context does not contain trick {trick_index}")
        selector = CardSelectorV1.build(arguments["selector"])
        match = _selector_literal(
            context,
            trick.winner,
            selector,
            name=f"{predicate.predicate_id}_winner",
        )
        seat_match = _seat_literal(
            context,
            trick.winner.actor,
            expected_seat=arguments.get("seat"),
            expected_team=arguments.get("team"),
            name=f"{predicate.predicate_id}_winner_seat",
        )
        literal = context.model.new_bool_var(f"{predicate.predicate_id}_result")
        context.model.add_min_equality(literal, [match, seat_match])
        return self._lowering(
            predicate,
            context,
            literal,
            lambda solver: {
                "selector": selector.to_dict(),
                "winner": _action_observation(solver, trick.winner),
            },
        )

    def _lower_play_trick_card_count(
        self,
        predicate: ScenarioPredicateV1,
        context: CpSatPredicateModelContextV1,
    ) -> CpSatPredicateLoweringV1:
        arguments = dict(predicate.arguments)
        trick_index = int(arguments["trickIndex"])
        trick = context.tricks.get(trick_index)
        if trick is None:
            raise ValueError(f"CP-SAT context does not contain trick {trick_index}")
        selector = CardSelectorV1.build(arguments["selector"])
        matches = tuple(
            _selector_literal(
                context,
                action,
                selector,
                name=f"{predicate.predicate_id}_card_{index}",
            )
            for index, action in enumerate(trick.actions)
        )
        literal, count = _range_literal(
            context.model,
            matches,
            minimum=int(arguments["minimum"]),
            maximum=int(arguments["maximum"]),
            name=predicate.predicate_id,
        )
        return self._lowering(
            predicate,
            context,
            literal,
            lambda solver: {
                "selector": selector.to_dict(),
                "matchingCount": int(solver.value(count)),
                "matchingActions": [
                    _action_observation(solver, action)
                    for action, matched in zip(trick.actions, matches, strict=True)
                    if solver.boolean_value(matched)
                ],
            },
        )

    def _winning_literals(
        self,
        context: CpSatPredicateModelContextV1,
        *,
        name: str,
    ) -> tuple[cp_model.IntVar, ...]:
        result: list[cp_model.IntVar] = []
        for index, action in enumerate(context.root_actions):
            if action.winning is None:
                raise AssertionError("root action winning literal disappeared")
            active = context.model.new_bool_var(f"{name}_winning_{index}")
            context.model.add_min_equality(active, [action.present, action.winning])
            result.append(active)
        return tuple(result)

    def _lower_winning_root_matches(
        self,
        predicate: ScenarioPredicateV1,
        context: CpSatPredicateModelContextV1,
    ) -> CpSatPredicateLoweringV1:
        selector = CardSelectorV1.build(dict(predicate.arguments)["selector"])
        winning = self._winning_literals(context, name=predicate.predicate_id)
        matches = tuple(
            _selector_literal(
                context,
                action,
                selector,
                name=f"{predicate.predicate_id}_root_{index}",
            )
            for index, action in enumerate(context.root_actions)
        )
        compatible: list[cp_model.IntVar] = []
        for index, (is_winning, matches_selector) in enumerate(
            zip(winning, matches, strict=True)
        ):
            item = context.model.new_bool_var(
                f"{predicate.predicate_id}_compatible_{index}"
            )
            context.model.add_allowed_assignments(
                [is_winning, matches_selector, item],
                [
                    [0, 0, 1],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 1, 1],
                ],
            )
            compatible.append(item)
        has_winner, winning_count = _range_literal(
            context.model,
            winning,
            minimum=1,
            maximum=len(winning),
            name=f"{predicate.predicate_id}_non_vacuous",
        )
        all_compatible = context.model.new_bool_var(
            f"{predicate.predicate_id}_all_compatible"
        )
        if compatible:
            context.model.add_min_equality(all_compatible, compatible)
        else:
            context.model.add(all_compatible == 1)
        literal = context.model.new_bool_var(f"{predicate.predicate_id}_result")
        context.model.add_min_equality(
            literal,
            [context.proof_complete, has_winner, all_compatible],
        )
        return self._lowering(
            predicate,
            context,
            literal,
            lambda solver: {
                "proofComplete": bool(solver.boolean_value(context.proof_complete)),
                "selector": selector.to_dict(),
                "winningCount": int(solver.value(winning_count)),
                "winningActions": [
                    _action_observation(solver, action)
                    for action, is_winning in zip(
                        context.root_actions, winning, strict=True
                    )
                    if solver.boolean_value(is_winning)
                ],
                "matchingWinningActionIds": [
                    action.action_id
                    for action, is_winning, matches_selector in zip(
                        context.root_actions, winning, matches, strict=True
                    )
                    if solver.boolean_value(is_winning)
                    and solver.boolean_value(matches_selector)
                ],
            },
        )

    def _lowering(
        self,
        predicate: ScenarioPredicateV1,
        context: CpSatPredicateModelContextV1,
        literal: cp_model.IntVar,
        extractor: Callable[[cp_model.CpSolver], Mapping[str, Any]],
    ) -> CpSatPredicateLoweringV1:
        extractor_id = self._EXTRACTORS[predicate.kind]
        lowering_hash = canonical_sha256(
            {
                "schemaVersion": 1,
                "predicateHash": predicate.sha256,
                "contextHash": context.sha256,
                "definitionHash": self._definitions[predicate.kind].sha256,
                "evidenceExtractor": extractor_id,
            }
        )
        return CpSatPredicateLoweringV1(
            predicate.predicate_id,
            predicate.kind,
            literal,
            extractor_id,
            lowering_hash,
            extractor,
        )


__all__ = [
    "CpSatActionVariablesV1",
    "CpSatPredicateLoweringV1",
    "CpSatPredicateModelContextV1",
    "CpSatTrickVariablesV1",
    "ScenarioPredicateLoweringRegistryV2",
]
