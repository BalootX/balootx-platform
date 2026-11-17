"""Engine-free CP-SAT lowerings for V1 play-prefix and state predicates."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import GamePhase, Seat, Suit, Team
from baloot_platform.core.exact_state_view import (
    build_canonical_observer_information_v1,
)

from .scenario_contracts import ScenarioPredicateV1, canonical_sha256
from .scenario_predicate_lowering_v2 import CpSatPredicateLoweringV1
from .scenario_registry_v2 import scenario_predicate_definitions_v2


_NONE = 4
_PHASE_IDS = {phase.value: index for index, phase in enumerate(GamePhase)}

PLAY_STATE_PREDICATE_KINDS_V1 = frozenset(
    {
        "play.trick_index",
        "play.current_player",
        "play.remaining_hand_size",
        "play.card_prefix",
        "play.action_prefix",
        "play.current_trick_leader",
        "play.current_trick_suit_led",
        "play.trump_use_count",
        "play.current_trick_card",
        "play.completed_trick_winner",
        "state.legal_action_count",
        "state.match_score",
        "state.phase",
        "state.next_actor",
        "state.observer_visibility",
        "state.observer_card_visibility",
        "state.current_raw_points",
        "state.current_project_points",
        "state.current_game_points",
        "state.legal_cards_include",
        "state.legal_play_actions_include",
    }
)

UNSUPPORTED_PLAY_STATE_PREDICATE_KINDS_V1 = frozenset()

PLAY_STATE_PREDICATE_REGISTRATIONS_V1 = MappingProxyType(
    {
        kind: "cp_sat.play_state_" + kind.replace(".", "_") + "_evidence_v1"
        for kind in sorted(
            PLAY_STATE_PREDICATE_KINDS_V1 - UNSUPPORTED_PLAY_STATE_PREDICATE_KINDS_V1
        )
    }
)
OBSERVER_VISIBILITY_CLASSIFICATIONS_V1 = MappingProxyType(
    {
        "complete_state": "public_fact",
        "actor_information_set": "legal_deduction",
    }
)
PLAY_STATE_PREDICATE_REGISTRATION_HASH_V1 = canonical_sha256(
    {
        "schemaVersion": 1,
        "registrations": dict(PLAY_STATE_PREDICATE_REGISTRATIONS_V1),
        "observerVisibilityClassifications": dict(
            OBSERVER_VISIBILITY_CLASSIFICATIONS_V1
        ),
    }
)


@dataclass(frozen=True, slots=True)
class CanonicalObserverScopeV1:
    """Hash-only authority for one complete world or one information set."""

    visibility_scope: str
    acting_observer: Seat
    classification: str
    observer_projection_hash: str
    hidden_world_scope_hash: str
    hidden_world_count: int
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        visibility_scope: str,
        acting_observer: Seat | str | int,
        complete_states: tuple[Mapping[str, Any], ...],
    ) -> "CanonicalObserverScopeV1":
        if visibility_scope not in OBSERVER_VISIBILITY_CLASSIFICATIONS_V1:
            raise ValueError("observer visibility scope is unsupported")
        observer = Seat.parse(acting_observer)
        if type(complete_states) is not tuple or not complete_states:
            raise ValueError("observer visibility requires complete hidden worlds")
        if visibility_scope == "complete_state" and len(complete_states) != 1:
            raise ValueError("complete_state visibility requires exactly one world")

        projections = []
        world_hashes = []
        for state in complete_states:
            if not isinstance(state, Mapping):
                raise TypeError("observer hidden world must be a complete-state object")
            if state.get("currentPlayer") != observer.name.lower():
                raise ValueError("observer visibility must bind the acting observer")
            projections.append(
                build_canonical_observer_information_v1(
                    complete_state=state,
                    acting_seat=observer.name.lower(),
                )
            )
            world_hashes.append(canonical_sha256(state))
        if len(world_hashes) != len(set(world_hashes)):
            raise ValueError("observer hidden-world scope contains duplicate worlds")
        projection_hashes = {
            str(projection["observerInformationHash"]) for projection in projections
        }
        if visibility_scope == "actor_information_set" and len(projection_hashes) != 1:
            raise ValueError("observer hidden worlds are distinguishable to the actor")

        projection_hash = (
            next(iter(projection_hashes))
            if visibility_scope == "actor_information_set"
            else canonical_sha256(
                {
                    "schemaVersion": 1,
                    "visibilityScope": "complete_state",
                    "actingObserver": observer.name.lower(),
                    "completeStateHash": world_hashes[0],
                }
            )
        )
        classification = OBSERVER_VISIBILITY_CLASSIFICATIONS_V1[visibility_scope]
        scope_core = {
            "schemaVersion": 1,
            "visibilityScope": visibility_scope,
            "actingObserver": observer.name.lower(),
            "classification": classification,
            "observerProjectionHash": projection_hash,
            "worldHashes": sorted(world_hashes),
            "hiddenWorldCount": len(world_hashes),
        }
        safe_core = {
            key: value for key, value in scope_core.items() if key != "worldHashes"
        }
        hidden_world_scope_hash = canonical_sha256(scope_core)
        safe_core["hiddenWorldScopeHash"] = hidden_world_scope_hash
        return cls(
            visibility_scope,
            observer,
            classification,
            projection_hash,
            hidden_world_scope_hash,
            len(world_hashes),
            canonical_sha256(safe_core),
        )

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "visibilityScope": self.visibility_scope,
            "actingObserver": self.acting_observer.name.lower(),
            "classification": self.classification,
            "observerProjectionHash": self.observer_projection_hash,
            "hiddenWorldScopeHash": self.hidden_world_scope_hash,
            "hiddenWorldCount": self.hidden_world_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class CpSatPlayPrefixActionV1:
    """One chronological play slot owned by the play-prefix model."""

    card: cp_model.IntVar
    actor: cp_model.IntVar
    ekka: cp_model.IntVar
    present: cp_model.IntVar


@dataclass(frozen=True, slots=True)
class CpSatPlayStatePredicateContextV1:
    """Variables exported by a full-prefix or exact-state CP-SAT model."""

    model: cp_model.CpModel
    context_id: str
    play_slots: tuple[CpSatPlayPrefixActionV1, ...]
    prefix_length: cp_model.IntVar
    trick_index: cp_model.IntVar
    next_actor: cp_model.IntVar
    remaining_hand_sizes: Mapping[Seat, cp_model.IntVar]
    current_trick_slots: tuple[CpSatPlayPrefixActionV1, ...]
    current_trick_size: cp_model.IntVar
    current_trick_active: cp_model.IntVar
    current_trick_leader: cp_model.IntVar
    current_trick_suit_led: cp_model.IntVar
    trump_suit: cp_model.IntVar
    completed_trick_literals: Mapping[int, cp_model.IntVar]
    completed_trick_winners: Mapping[int, cp_model.IntVar]
    legal_card_literals: Mapping[int, cp_model.IntVar]
    legal_action_literals: Mapping[tuple[int, bool], cp_model.IntVar]
    phase: cp_model.IntVar
    match_scores: Mapping[Team, cp_model.IntVar]
    raw_points: Mapping[Team, cp_model.IntVar]
    project_points: Mapping[Team, cp_model.IntVar]
    game_points: Mapping[Team, cp_model.IntVar]
    generic_legal_action_count: cp_model.IntVar | None = None
    card_visibility_literals: Mapping[tuple[Seat, int], cp_model.IntVar] | None = None
    observer_scope: CanonicalObserverScopeV1 | None = None
    native_observer_scope: bool = False
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, cp_model.CpModel):
            raise TypeError("play/state predicate context requires a CpModel")
        if type(self.context_id) is not str or not self.context_id:
            raise ValueError("play/state predicate context_id must be non-empty")
        if type(self.play_slots) is not tuple or len(self.play_slots) != 32:
            raise ValueError("play/state predicate context requires 32 play slots")
        if (
            type(self.current_trick_slots) is not tuple
            or len(self.current_trick_slots) != 4
        ):
            raise ValueError("play/state predicate context requires four trick slots")
        self._require_keys(self.remaining_hand_sizes, set(Seat), "remaining hands")
        self._require_keys(self.completed_trick_literals, set(range(8)), "completed")
        self._require_keys(self.completed_trick_winners, set(range(8)), "winners")
        self._require_keys(self.legal_card_literals, set(range(32)), "legal cards")
        self._require_keys(
            self.legal_action_literals,
            {(card, ekka) for card in range(32) for ekka in (False, True)},
            "legal actions",
        )
        for label, values in (
            ("match scores", self.match_scores),
            ("raw points", self.raw_points),
            ("project points", self.project_points),
            ("game points", self.game_points),
        ):
            self._require_keys(values, set(Team), label)
        if self.card_visibility_literals is not None:
            unknown = [
                key
                for key in self.card_visibility_literals
                if not isinstance(key, tuple)
                or len(key) != 2
                or not isinstance(key[0], Seat)
                or type(key[1]) is not int
                or not 0 <= key[1] < 32
            ]
            if unknown:
                raise ValueError("card visibility keys must be (Seat, card id)")
        object.__setattr__(
            self,
            "sha256",
            canonical_sha256(
                {
                    "schemaVersion": 1,
                    "contextId": self.context_id,
                    "playSlots": len(self.play_slots),
                    "currentTrickSlots": len(self.current_trick_slots),
                    "visibilityKeys": [
                        [seat.name.lower(), card]
                        for seat, card in sorted(
                            (self.card_visibility_literals or {}),
                            key=lambda item: (int(item[0]), item[1]),
                        )
                    ],
                    "observerScope": (
                        self.observer_scope.to_safe_dict()
                        if self.observer_scope is not None
                        else None
                    ),
                    **({"nativeObserverScope": True} if self.native_observer_scope else {}),
                }
            ),
        )

    @staticmethod
    def _require_keys(value: Mapping[Any, Any], expected: set[Any], label: str) -> None:
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(f"play/state predicate {label} keys are incomplete")


def _equal(model: cp_model.CpModel, variable: Any, value: int, name: str):
    literal = model.new_bool_var(name)
    model.add(variable == value).only_enforce_if(literal)
    model.add(variable != value).only_enforce_if(literal.negated())
    return literal


def _range(
    model: cp_model.CpModel,
    variable: Any,
    minimum: int,
    maximum: int,
    name: str,
):
    lower = model.new_bool_var(f"{name}_min")
    upper = model.new_bool_var(f"{name}_max")
    model.add(variable >= minimum).only_enforce_if(lower)
    model.add(variable < minimum).only_enforce_if(lower.negated())
    model.add(variable <= maximum).only_enforce_if(upper)
    model.add(variable > maximum).only_enforce_if(upper.negated())
    result = model.new_bool_var(f"{name}_range")
    model.add_min_equality(result, [lower, upper])
    return result


def _all(model: cp_model.CpModel, values: list[Any], name: str):
    if not values:
        return model.new_constant(1)
    result = model.new_bool_var(name)
    model.add_min_equality(result, values)
    return result


def _observed_action(
    solver: cp_model.CpSolver, action: CpSatPlayPrefixActionV1
) -> dict[str, Any]:
    return {
        "card": Card(int(solver.value(action.card))).to_dict(),
        "actor": Seat(int(solver.value(action.actor))).name.lower(),
        "ekka": bool(solver.boolean_value(action.ekka)),
        "present": bool(solver.boolean_value(action.present)),
    }


class OrToolsPlayStatePredicateLoweringV1:
    """Closed category lowering; integration registers its exported pairs."""

    def __init__(self) -> None:
        self._definitions = {
            row.kind: row for row in scenario_predicate_definitions_v2()
        }
        self._lowerers: dict[str, Callable[..., tuple[Any, Callable[..., Any]]]] = {
            kind: getattr(self, "_lower_" + kind.replace(".", "_"))
            for kind in PLAY_STATE_PREDICATE_REGISTRATIONS_V1
        }

    @property
    def registrations(self) -> Mapping[str, str]:
        return PLAY_STATE_PREDICATE_REGISTRATIONS_V1

    def lower_predicate(
        self,
        predicate: ScenarioPredicateV1,
        context: CpSatPlayStatePredicateContextV1,
    ) -> CpSatPredicateLoweringV1:
        lowerer = self._lowerers.get(predicate.kind)
        definition = self._definitions.get(predicate.kind)
        if lowerer is None or definition is None or predicate.version != 1:
            raise ValueError(
                f"predicate has no exact CP-SAT lowering: "
                f"{predicate.kind}@{predicate.version}"
            )
        definition.validate_arguments(predicate.arguments)
        if (
            predicate.kind == "state.observer_card_visibility"
            and context.card_visibility_literals is None
        ):
            raise ValueError(
                "state.observer_card_visibility requires exact observation variables"
            )
        if (
            predicate.kind == "state.observer_visibility"
            and context.observer_scope is None
        ):
            raise ValueError(
                "state.observer_visibility requires a canonical observer scope"
            )
        literal, extractor = lowerer(predicate, context)
        extractor_id = PLAY_STATE_PREDICATE_REGISTRATIONS_V1[predicate.kind]
        lowering_hash = canonical_sha256(
            {
                "schemaVersion": 1,
                "predicateHash": predicate.sha256,
                "contextHash": context.sha256,
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

    @staticmethod
    def _simple_range(
        predicate: ScenarioPredicateV1,
        context: CpSatPlayStatePredicateContextV1,
        variable: Any,
        label: str,
    ):
        args = predicate.arguments
        literal = _range(
            context.model,
            variable,
            int(args["minimum"]),
            int(args["maximum"]),
            predicate.predicate_id,
        )
        return literal, lambda solver: {label: int(solver.value(variable))}

    def _lower_play_trick_index(self, predicate, context):
        return self._simple_range(predicate, context, context.trick_index, "trickIndex")

    def _lower_play_current_player(self, predicate, context):
        expected = int(Seat.parse(predicate.arguments["seat"]))
        literal = _equal(
            context.model, context.next_actor, expected, predicate.predicate_id
        )
        return literal, lambda solver: {
            "currentPlayer": (
                Seat(int(solver.value(context.next_actor))).name.lower()
                if int(solver.value(context.next_actor)) != _NONE
                else None
            )
        }

    def _lower_play_remaining_hand_size(self, predicate, context):
        seat = Seat.parse(predicate.arguments["seat"])
        return self._simple_range(
            predicate,
            context,
            context.remaining_hand_sizes[seat],
            "remainingHandSize",
        )

    def _prefix(self, predicate, context, *, actions: bool):
        expected = (
            list(predicate.arguments["actions"])
            if actions
            else [{"card": card, "ekka": None} for card in predicate.arguments["cards"]]
        )
        matches: list[Any] = []
        for index, item in enumerate(expected):
            slot = context.play_slots[index]
            matches.extend(
                [
                    slot.present,
                    _equal(
                        context.model,
                        slot.card,
                        int(item["card"]),
                        f"{predicate.predicate_id}_{index}_card",
                    ),
                ]
            )
            if item["ekka"] is not None:
                matches.append(
                    _equal(
                        context.model,
                        slot.ekka,
                        int(item["ekka"]),
                        f"{predicate.predicate_id}_{index}_ekka",
                    )
                )
        matches.append(
            _equal(
                context.model,
                context.prefix_length,
                len(expected),
                f"{predicate.predicate_id}_complete",
            )
            if bool(predicate.arguments.get("complete", False))
            else _range(
                context.model,
                context.prefix_length,
                len(expected),
                32,
                f"{predicate.predicate_id}_length",
            )
        )
        literal = _all(context.model, matches, f"{predicate.predicate_id}_result")
        return literal, lambda solver: {
            "prefixLength": int(solver.value(context.prefix_length)),
            "matchedPrefix": [
                _observed_action(solver, context.play_slots[index])
                for index in range(len(expected))
            ],
        }

    def _lower_play_card_prefix(self, predicate, context):
        return self._prefix(predicate, context, actions=False)

    def _lower_play_action_prefix(self, predicate, context):
        return self._prefix(predicate, context, actions=True)

    def _lower_play_current_trick_leader(self, predicate, context):
        seat = int(Seat.parse(predicate.arguments["seat"]))
        literal = _all(
            context.model,
            [
                context.current_trick_active,
                _equal(
                    context.model,
                    context.current_trick_leader,
                    seat,
                    f"{predicate.predicate_id}_seat",
                ),
            ],
            f"{predicate.predicate_id}_result",
        )
        return literal, lambda solver: {
            "active": bool(solver.boolean_value(context.current_trick_active)),
            "leader": Seat(
                int(solver.value(context.current_trick_leader))
            ).name.lower(),
        }

    def _lower_play_current_trick_suit_led(self, predicate, context):
        suit = int(Suit.parse(predicate.arguments["suit"]))
        has_card = _range(
            context.model,
            context.current_trick_size,
            1,
            4,
            f"{predicate.predicate_id}_has_card",
        )
        literal = _all(
            context.model,
            [
                has_card,
                _equal(
                    context.model,
                    context.current_trick_suit_led,
                    suit,
                    f"{predicate.predicate_id}_suit",
                ),
            ],
            f"{predicate.predicate_id}_result",
        )
        return literal, lambda solver: {
            "currentTrickSize": int(solver.value(context.current_trick_size)),
            "suitLed": Suit(
                int(solver.value(context.current_trick_suit_led))
            ).name.lower(),
        }

    def _lower_play_trump_use_count(self, predicate, context):
        indexes = (
            range(
                int(predicate.arguments["trickIndex"]) * 4,
                int(predicate.arguments["trickIndex"]) * 4 + 4,
            )
            if "trickIndex" in predicate.arguments
            else range(32)
        )
        matches = []
        for index in indexes:
            slot = context.play_slots[index]
            match = context.model.new_bool_var(
                f"{predicate.predicate_id}_{index}_trump"
            )
            context.model.add_allowed_assignments(
                [slot.card, context.trump_suit, match],
                [
                    [card, trump, int(card // 8 == trump)]
                    for card in range(32)
                    for trump in range(5)
                ],
            )
            active = context.model.new_bool_var(
                f"{predicate.predicate_id}_{index}_active"
            )
            context.model.add_min_equality(active, [slot.present, match])
            matches.append(active)
        count = context.model.new_int_var(
            0, len(matches), f"{predicate.predicate_id}_count"
        )
        context.model.add(count == sum(matches))
        literal = _range(
            context.model,
            count,
            int(predicate.arguments["minimum"]),
            int(predicate.arguments["maximum"]),
            predicate.predicate_id,
        )
        return literal, lambda solver: {"trumpUseCount": int(solver.value(count))}

    def _lower_play_current_trick_card(self, predicate, context):
        args = predicate.arguments
        position = int(args["position"])
        slot = context.current_trick_slots[position]
        matches = [
            slot.present,
            _equal(
                context.model,
                slot.actor,
                int(Seat.parse(args["seat"])),
                f"{predicate.predicate_id}_seat",
            ),
            _equal(
                context.model,
                slot.card,
                int(args["card"]),
                f"{predicate.predicate_id}_card",
            ),
        ]
        if "ekka" in args:
            matches.append(
                _equal(
                    context.model,
                    slot.ekka,
                    int(args["ekka"]),
                    f"{predicate.predicate_id}_ekka",
                )
            )
        literal = _all(context.model, matches, f"{predicate.predicate_id}_result")
        return literal, lambda solver: {
            "position": position,
            "action": _observed_action(solver, slot),
        }

    def _lower_play_completed_trick_winner(self, predicate, context):
        args = predicate.arguments
        index = int(args["trickIndex"])
        literal = _all(
            context.model,
            [
                context.completed_trick_literals[index],
                _equal(
                    context.model,
                    context.completed_trick_winners[index],
                    int(Seat.parse(args["seat"])),
                    f"{predicate.predicate_id}_winner",
                ),
            ],
            f"{predicate.predicate_id}_result",
        )
        return literal, lambda solver: {
            "trickIndex": index,
            "completed": bool(
                solver.boolean_value(context.completed_trick_literals[index])
            ),
            "winner": Seat(
                int(solver.value(context.completed_trick_winners[index]))
            ).name.lower(),
        }

    def _lower_state_legal_action_count(self, predicate, context):
        count = context.model.new_int_var(0, 64, f"{predicate.predicate_id}_count")
        observed = context.generic_legal_action_count
        context.model.add(
            count
            == (
                observed
                if observed is not None
                else sum(context.legal_action_literals.values())
            )
        )
        return self._simple_range(predicate, context, count, "legalActionCount")

    def _team_range(self, predicate, context, values, label):
        team = Team[str(predicate.arguments["team"]).upper()]
        return self._simple_range(predicate, context, values[team], label)

    def _lower_state_match_score(self, predicate, context):
        return self._team_range(predicate, context, context.match_scores, "matchScore")

    def _lower_state_phase(self, predicate, context):
        literal = _equal(
            context.model,
            context.phase,
            _PHASE_IDS[str(predicate.arguments["state"])],
            predicate.predicate_id,
        )
        return literal, lambda solver: {
            "phase": tuple(GamePhase)[int(solver.value(context.phase))].value
        }

    def _lower_state_next_actor(self, predicate, context):
        return self._lower_play_current_player(predicate, context)

    def _lower_state_observer_visibility(self, predicate, context):
        scope = context.observer_scope
        assert scope is not None
        observer = Seat.parse(predicate.arguments["observer"])
        if context.native_observer_scope:
            classifications = tuple(self._definitions[predicate.kind].argument_schema[
                "properties"]["classification"]["enum"])
            actor = context.model.new_int_var(0, len(Seat) - 1,
                f"{predicate.predicate_id}_scope_actor")
            classification = context.model.new_int_var(0, len(classifications) - 1,
                f"{predicate.predicate_id}_scope_classification")
            context.model.add(actor == int(scope.acting_observer))
            context.model.add(classification == classifications.index(scope.classification))
            literal = _all(context.model, [
                _equal(context.model, actor, int(observer),
                    f"{predicate.predicate_id}_observer"),
                _equal(context.model, classification,
                    classifications.index(str(predicate.arguments["classification"])),
                    f"{predicate.predicate_id}_classification"),
                _equal(context.model, context.next_actor, actor,
                    f"{predicate.predicate_id}_acting_observer"),
            ], f"{predicate.predicate_id}_result")
            return literal, lambda _solver: scope.to_safe_dict()
        acting_matches = _equal(
            context.model,
            context.next_actor,
            int(scope.acting_observer),
            f"{predicate.predicate_id}_acting_observer",
        )
        literal = _all(
            context.model,
            [
                context.model.new_constant(int(observer == scope.acting_observer)),
                context.model.new_constant(
                    int(
                        str(predicate.arguments["classification"])
                        == scope.classification
                    )
                ),
                acting_matches,
            ],
            f"{predicate.predicate_id}_result",
        )
        return literal, lambda _solver: scope.to_safe_dict()

    def _lower_state_observer_card_visibility(self, predicate, context):
        args = predicate.arguments
        key = (Seat.parse(args["observer"]), int(args["card"]))
        assert context.card_visibility_literals is not None
        source = context.card_visibility_literals.get(key)
        if source is None:
            raise ValueError(
                "state.observer_card_visibility lacks the requested observation variable"
            )
        expected = int(args["visibility"] == "visible")
        literal = _equal(context.model, source, expected, predicate.predicate_id)
        return literal, lambda solver: {
            "observer": key[0].name.lower(),
            "card": Card(key[1]).to_dict(),
            "visibility": "visible" if solver.boolean_value(source) else "hidden",
        }

    def _lower_state_current_raw_points(self, predicate, context):
        return self._team_range(predicate, context, context.raw_points, "rawPoints")

    def _lower_state_current_project_points(self, predicate, context):
        return self._team_range(
            predicate, context, context.project_points, "projectPoints"
        )

    def _lower_state_current_game_points(self, predicate, context):
        return self._team_range(predicate, context, context.game_points, "gamePoints")

    def _set_inclusion(self, predicate, context, *, actions: bool):
        requested = (
            {
                (int(item["card"]), bool(item["ekka"]))
                for item in predicate.arguments["actions"]
            }
            if actions
            else {int(card) for card in predicate.arguments["cards"]}
        )
        universe = (
            context.legal_action_literals if actions else context.legal_card_literals
        )
        matches = [universe[item] for item in requested]
        if bool(predicate.arguments.get("complete", False)):
            matches.extend(
                _equal(
                    context.model,
                    literal,
                    int(item in requested),
                    f"{predicate.predicate_id}_{index}",
                )
                for index, (item, literal) in enumerate(universe.items())
            )
        literal = _all(context.model, matches, f"{predicate.predicate_id}_result")
        return literal, lambda solver: {
            "complete": bool(predicate.arguments.get("complete", False)),
            "legal": [
                (
                    {"card": Card(item[0]).to_dict(), "ekka": item[1]}
                    if actions
                    else Card(item).to_dict()
                )
                for item, available in universe.items()
                if solver.boolean_value(available)
            ],
        }

    def _lower_state_legal_cards_include(self, predicate, context):
        return self._set_inclusion(predicate, context, actions=False)

    def _lower_state_legal_play_actions_include(self, predicate, context):
        return self._set_inclusion(predicate, context, actions=True)


__all__ = [
    "CanonicalObserverScopeV1",
    "CpSatPlayPrefixActionV1",
    "CpSatPlayStatePredicateContextV1",
    "OrToolsPlayStatePredicateLoweringV1",
    "OBSERVER_VISIBILITY_CLASSIFICATIONS_V1",
    "PLAY_STATE_PREDICATE_KINDS_V1",
    "PLAY_STATE_PREDICATE_REGISTRATION_HASH_V1",
    "PLAY_STATE_PREDICATE_REGISTRATIONS_V1",
    "UNSUPPORTED_PLAY_STATE_PREDICATE_KINDS_V1",
]
