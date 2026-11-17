"""Exact engine-free auction and escalation predicate lowerings."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Mapping, Sequence

from ortools.sat.python import cp_model

from baloot_platform.core.enums import Contract, Seat, Suit, Team

from .ortools_escalation_stage import (OrToolsEscalationStageV1, attach_native_escalation_transition_v1, _LEVELS,
    OrToolsEscalationActionV1, OrToolsEscalationChoiceV1, _EscalationCandidateV1)
from .ortools_scenario_model import OrToolsScenarioAuctionFragmentV1, OrToolsNativeAuctionFragmentV1, _auction_logic_v1
from .scenario_contracts import ScenarioPredicateV1, canonical_sha256
from .scenario_predicate_lowering_v2 import CpSatPredicateLoweringV1
from .scenario_registry import ScenarioPredicateRegistryV1


@dataclass(frozen=True, slots=True)
class AuctionEscalationPredicateRegistrationV1:
    kind: str
    stage: str
    evidence_extractor_id: str


_AUCTION_KINDS = (
    "auction.dealer",
    "auction.buyer",
    "auction.ashkal_buyer_relation",
    "auction.contract",
    "auction.trump",
    "auction.purchase_round",
    "auction.round",
    "auction.action_sequence",
    "auction.legal_actions",
    "auction.pass_count",
    "auction.terminal",
)
_ESCALATION_KINDS = (
    "escalation.level",
    "escalation.action_sequence",
    "escalation.locked",
    "escalation.eligible_actor",
    "escalation.legal_actions",
    "escalation.terminal",
)

AUCTION_ESCALATION_PREDICATE_REGISTRATIONS_V1 = tuple(
    AuctionEscalationPredicateRegistrationV1(
        kind,
        kind.split(".", 1)[0],
        f"cp_sat.{kind.replace('.', '_')}_evidence_v1",
    )
    for kind in (*_AUCTION_KINDS, *_ESCALATION_KINDS)
)
_REGISTRATION_BY_KIND = {
    registration.kind: registration
    for registration in AUCTION_ESCALATION_PREDICATE_REGISTRATIONS_V1
}


@dataclass(frozen=True, slots=True)
class CpSatAuctionPredicateContextV1:
    model: cp_model.CpModel
    fragment: OrToolsScenarioAuctionFragmentV1 | OrToolsNativeAuctionFragmentV1
    dealer: Seat
    view: str = "completion"
    context_id: str = "auction"
    native: Mapping[str, Any] = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, cp_model.CpModel):
            raise TypeError("auction predicate context requires a CpModel")
        if not isinstance(self.fragment, (OrToolsScenarioAuctionFragmentV1, OrToolsNativeAuctionFragmentV1)):
            raise TypeError("auction predicate context requires an auction fragment")
        if self.view not in {"prefix", "completion"}:
            raise ValueError("auction predicate view must be prefix or completion")
        if isinstance(self.fragment, OrToolsNativeAuctionFragmentV1):
            object.__setattr__(self, "dealer", Seat.parse(self.dealer))
            self._attach_native()
            return
        if len(self.fragment.plans) != len(self.fragment.selected):
            raise ValueError("auction predicate plans and selectors disagree")
        object.__setattr__(self, "dealer", Seat.parse(self.dealer))
        values = tuple(_auction_values(self, plan) for plan in self.fragment.plans)
        selected = self.fragment.selected

        def selected_value(
            name: str, rows: list[int], lower: int, upper: int
        ) -> cp_model.IntVar:
            variable = self.model.new_int_var(lower, upper, name)
            self.model.add(
                variable
                == sum(
                    value * selector
                    for value, selector in zip(rows, selected, strict=True)
                )
            )
            return variable

        contracts = {
            None: 0,
            **{contract: index + 1 for index, contract in enumerate(Contract)},
        }
        actions = {
            action: index
            for index, action in enumerate(
                sorted({row["action"] for value in values for row in value["actions"]})
            )
        }
        maximum_steps = max((len(value["actions"]) for value in values), default=0)
        sentinel = 4
        action_slots: list[
            tuple[cp_model.IntVar, cp_model.IntVar, cp_model.IntVar, cp_model.IntVar]
        ] = []
        for step in range(maximum_steps):
            action_slots.append(
                (
                    selected_value(
                        f"auction_predicate_round_{step}",
                        [
                            value["actions"][step]["roundIndex"]
                            if step < len(value["actions"])
                            else 0
                            for value in values
                        ],
                        0,
                        2,
                    ),
                    selected_value(
                        f"auction_predicate_actor_{step}",
                        [
                            int(Seat.parse(value["actions"][step]["actor"]))
                            if step < len(value["actions"])
                            else sentinel
                            for value in values
                        ],
                        0,
                        sentinel,
                    ),
                    selected_value(
                        f"auction_predicate_action_{step}",
                        [
                            actions[value["actions"][step]["action"]]
                            if step < len(value["actions"])
                            else len(actions)
                            for value in values
                        ],
                        0,
                        len(actions),
                    ),
                    selected_value(
                        f"auction_predicate_trump_{step}",
                        [
                            int(Suit.parse(value["actions"][step]["trumpSuit"]))
                            if step < len(value["actions"])
                            and value["actions"][step].get("trumpSuit") is not None
                            else sentinel
                            for value in values
                        ],
                        0,
                        sentinel,
                    ),
                )
            )
        legal_keys = sorted(
            {
                (row["action"], row.get("trumpSuit"))
                for value in values
                for row in value["legalActions"]
            },
            key=lambda item: (item[0], item[1] or ""),
        )
        legal = {
            key: selected_value(
                "auction_predicate_legal_"
                + "_".join(str(item or "default") for item in key),
                [
                    int(
                        key
                        in {
                            (row["action"], row.get("trumpSuit"))
                            for row in value["legalActions"]
                        }
                    )
                    for value in values
                ],
                0,
                1,
            )
            for key in legal_keys
        }
        native = {
            "contracts": contracts,
            "actions": actions,
            "length": selected_value(
                "auction_predicate_length",
                [len(value["actions"]) for value in values],
                0,
                maximum_steps,
            ),
            "dealer": selected_value(
                "auction_predicate_dealer", [int(self.dealer)] * len(values), 0, 3
            ),
            "buyer": selected_value(
                "auction_predicate_buyer",
                [
                    sentinel if value["buyer"] is None else int(value["buyer"])
                    for value in values
                ],
                0,
                sentinel,
            ),
            "contract": selected_value(
                "auction_predicate_contract",
                [contracts[value["contract"]] for value in values],
                0,
                len(contracts) - 1,
            ),
            "trump": selected_value(
                "auction_predicate_trump",
                [
                    sentinel if value["trump"] is None else int(value["trump"])
                    for value in values
                ],
                0,
                sentinel,
            ),
            "purchaseRound": selected_value(
                "auction_predicate_purchase_round",
                [value["purchaseRound"] or 0 for value in values],
                0,
                2,
            ),
            "round": selected_value(
                "auction_predicate_round", [value["round"] for value in values], 1, 2
            ),
            "currentActor": selected_value(
                "auction_predicate_current_actor",
                [
                    sentinel
                    if plan.current_actor is None
                    else int(plan.current_actor)
                    for plan in self.fragment.plans
                ],
                0,
                sentinel,
            ),
            "passCount": selected_value(
                "auction_predicate_pass_count",
                [value["passCount"] for value in values],
                0,
                maximum_steps,
            ),
            "terminal": selected_value(
                "auction_predicate_terminal",
                [int(value["terminal"]) for value in values],
                0,
                1,
            ),
            "ashkalRelation": selected_value(
                "auction_predicate_ashkal_relation",
                [
                    {None: 0, "bidder": 1, "bidder_partner": 2}[
                        value["ashkalBuyerRelation"]
                    ]
                    for value in values
                ],
                0,
                2,
            ),
            "actionSlots": tuple(action_slots),
            "legal": legal,
        }
        object.__setattr__(self, "native", native)
        object.__setattr__(
            self,
            "sha256",
            canonical_sha256(
                {
                    "schemaVersion": 1,
                    "contextId": self.context_id,
                    "domainIrHash": self.fragment.domain_hash,
                    "dealer": Seat.parse(self.dealer).name.lower(),
                    "view": self.view,
                    "candidateFingerprints": [
                        plan.fingerprint for plan in self.fragment.plans
                    ],
                }
            ),
        )


    def _attach_native(self) -> None:
        model, fragment = self.model, self.fragment
        completion = self.view == "completion"
        count = len(fragment.choices) if completion else fragment.prefix_length
        state = fragment.states[-1] if completion else fragment.states[count]
        name = f"{self.context_id}_native_{self.view}"
        equal, both, _ = _auction_logic_v1(model, name)
        model.add(fragment.states[0]["actor"] == (int(self.dealer) + 1) % len(Seat))
        prefix_state = fragment.states[fragment.prefix_length]
        current_actor = model.new_int_var(0, 4, f"{name}_current_actor")
        model.add(current_actor == 4).only_enforce_if(prefix_state["terminal"])
        model.add(current_actor == prefix_state.get("actor", 0)).only_enforce_if(prefix_state["terminal"].Not())

        def scalar(key, value, low, high):
            result = model.new_int_var(low, high, f"{name}_{key}")
            model.add(result == value)
            return result

        def terminal_value(key, value, default, low, high):
            result = model.new_int_var(low, high, f"{name}_{key}")
            model.add(result == value).only_enforce_if(state["terminal"])
            model.add(result == default).only_enforce_if(state["terminal"].Not())
            return result

        actions = {action: index for index, action in enumerate(("ashkal", "hokum", "pass", "sun"))}
        slots = []
        for index, row in enumerate(fragment.choices[:count]):
            before = fragment.states[index]
            round_var = model.new_int_var(0, 2, f"{name}_round_{index}")
            actor_var = model.new_int_var(0, 4, f"{name}_actor_{index}")
            for variable, value, padding in ((round_var, before["round"], 0), (actor_var, before["actor"], 4)):
                model.add(variable == value).only_enforce_if(row["closed"].Not())
                model.add(variable == padding).only_enforce_if(row["closed"])
            action_var = scalar(f"action_{index}", sum(
                (len(actions) if code == "closed" else actions[code.split(":")[0]]) * literal
                for code, literal in row.items()), 0, len(actions))
            trump_var = scalar(f"trump_{index}", sum(
                (int(Suit.parse(code.split(":")[1])) if code.startswith("hokum:") else 4) * literal
                for code, literal in row.items()), 0, 4)
            slots.append((round_var, actor_var, action_var, trump_var))
        ashkal = both("ashkal", state["terminal"], equal(state["contract"], 2, "ashkal_contract"))
        own = equal(state["buyer"], state["caller"], "buyer_is_caller")
        relation = model.new_int_var(0, 2, f"{name}_relation")
        model.add(relation == 0).only_enforce_if(ashkal.Not())
        model.add(relation == 1).only_enforce_if([ashkal, own])
        model.add(relation == 2).only_enforce_if([ashkal, own.Not()])
        legal = {(code.split(":")[0], code.split(":")[1] if ":" in code else None):
                 model.new_constant(0) if completion else literal
                 for code, literal in fragment.legal_next_choice_literals.items()}
        native = {"contracts": {None: 0, **{contract: index + 1 for index, contract in enumerate(Contract)}},
            "actions": actions, "dealer": model.new_constant(int(self.dealer)), "currentActor": current_actor,
            "length": scalar("length", sum(row["closed"].Not() for row in fragment.choices[:count]), 0, count),
            "passCount": scalar("passes", sum(row["pass"] for row in fragment.choices[:count]), 0, count),
            "buyer": terminal_value("buyer", state["buyer"], 4, 0, 4),
            "bidder": terminal_value("bidder", state["caller"], 4, 0, 4),
            "contract": terminal_value("contract", state["contract"] + 1, 0, 0, 3),
            "trump": terminal_value("trump", state["trump"], 4, 0, 4),
            "purchaseRound": terminal_value("purchase_round", state["purchaseRound"], 0, 0, 2),
            "round": state["round"], "terminal": state["terminal"], "ashkalRelation": relation,
            "actionSlots": tuple(slots), "legal": legal}
        object.__setattr__(self, "native", native)
        object.__setattr__(self, "sha256", canonical_sha256({"schemaVersion": 1, "contextId": self.context_id,
            "domainIrHash": fragment.domain_hash, "dealer": self.dealer.name.lower(), "view": self.view,
            "lowering": "native_auction_path_v1", "cpModelProto": str(model.proto)}))

    def solved_native_values(self, solver: cp_model.CpSolver) -> dict[str, Any]:
        native = self.native
        observation = self.fragment.extract(solver)
        actions = observation.completion_actions if self.view == "completion" else observation.actions
        contracts = {value: key for key, value in native["contracts"].items()}
        def seat(key):
            value = solver.value(native[key])
            return None if value == 4 else Seat(value)
        return {"dealer": seat("dealer"), "buyer": seat("buyer"), "bidder": seat("bidder"),
            "contract": contracts[solver.value(native["contract"])],
            "trump": None if solver.value(native["trump"]) == 4 else Suit(solver.value(native["trump"])),
            "purchaseRound": solver.value(native["purchaseRound"]) or None,
            "round": solver.value(native["round"]), "actions": [action.to_dict() for action in actions],
            "passCount": solver.value(native["passCount"]), "terminal": bool(solver.value(native["terminal"])),
            "ashkalBuyerRelation": {0: None, 1: "bidder", 2: "bidder_partner"}[solver.value(native["ashkalRelation"])],
            "legalActions": [{"action": action, **({"trumpSuit": trump} if trump is not None else {})}
                             for (action, trump), literal in sorted(native["legal"].items(), key=lambda item: (item[0][0], item[0][1] or ""))
                             if solver.value(literal)]}


@dataclass(frozen=True, slots=True)
class CpSatEscalationPredicateContextV1:
    model: cp_model.CpModel
    stage: OrToolsEscalationStageV1
    candidates: tuple[Any, ...]
    selected: tuple[cp_model.IntVar, ...]
    native: Mapping[str, Any]
    context_id: str = "escalation"
    sha256: str = field(init=False)

    @classmethod
    def attach(
        cls,
        stage: OrToolsEscalationStageV1,
        *,
        model: cp_model.CpModel | None = None,
        context_id: str = "escalation",
        declarer_variable: cp_model.IntVar | None = None,
    ) -> "CpSatEscalationPredicateContextV1":
        if not isinstance(stage, OrToolsEscalationStageV1):
            raise TypeError("escalation predicate context requires an escalation stage")
        cp_model_value = model or cp_model.CpModel()
        if stage.native_cp_sat:
            return cls._attach_native(stage, cp_model_value, context_id, declarer_variable)
        if declarer_variable is not None:
            raise ValueError("symbolic declarer requires native escalation")
        candidates = stage._candidates()
        selected = tuple(
            cp_model_value.new_bool_var(f"escalation_plan_{index}")
            for index in range(len(candidates))
        )
        cp_model_value.add_exactly_one(selected)

        levels = {
            value: index
            for index, value in enumerate(
                ("normal", "double", "triple", "four", "gahwa")
            )
        }
        actions = {
            value: index for index, value in enumerate(stage.domain.supported_actions)
        }
        modes = {None: 0, "open": 1, "locked": 2}
        maximum_steps = max(
            (len(candidate.actions) for candidate in candidates), default=0
        )

        def selected_value(
            name: str, values: list[int], lower: int, upper: int
        ) -> cp_model.IntVar:
            variable = cp_model_value.new_int_var(lower, upper, name)
            cp_model_value.add(
                variable
                == sum(
                    value * selector
                    for value, selector in zip(values, selected, strict=True)
                )
            )
            return variable

        length = selected_value(
            "escalation_predicate_length",
            [len(candidate.actions) for candidate in candidates],
            0,
            maximum_steps,
        )
        level = selected_value(
            "escalation_predicate_level",
            [levels[candidate.level] for candidate in candidates],
            0,
            len(levels) - 1,
        )
        locked = selected_value(
            "escalation_predicate_locked",
            [int(candidate.locked) for candidate in candidates],
            0,
            1,
        )
        terminal = selected_value(
            "escalation_predicate_terminal",
            [int(candidate.terminal) for candidate in candidates],
            0,
            1,
        )
        sentinel = 4
        action_slots: list[
            tuple[cp_model.IntVar, cp_model.IntVar, cp_model.IntVar]
        ] = []
        for step in range(maximum_steps):
            action_slots.append(
                (
                    selected_value(
                        f"escalation_predicate_actor_{step}",
                        [
                            int(candidate.actions[step].actor)
                            if step < len(candidate.actions)
                            else sentinel
                            for candidate in candidates
                        ],
                        0,
                        sentinel,
                    ),
                    selected_value(
                        f"escalation_predicate_action_{step}",
                        [
                            actions[candidate.actions[step].action]
                            if step < len(candidate.actions)
                            else sentinel
                            for candidate in candidates
                        ],
                        0,
                        sentinel,
                    ),
                    selected_value(
                        f"escalation_predicate_mode_{step}",
                        [
                            modes[candidate.actions[step].requested_mode]
                            if step < len(candidate.actions)
                            else 0
                            for candidate in candidates
                        ],
                        0,
                        2,
                    ),
                )
            )

        seat_eligible = {
            seat: selected_value(
                f"escalation_predicate_seat_{int(seat)}_eligible",
                [int(seat in candidate.eligible_seats) for candidate in candidates],
                0,
                1,
            )
            for seat in Seat
        }
        team_eligible = {
            team: selected_value(
                f"escalation_predicate_team_{int(team)}_eligible",
                [int(team in candidate.eligible_teams) for candidate in candidates],
                0,
                1,
            )
            for team in Team
        }
        legal_keys = sorted(
            {
                (row["action"], row.get("mode"))
                for candidate in candidates
                for row in _escalation_values(candidate)["legalActions"]
            },
            key=lambda item: (item[0], item[1] or ""),
        )
        legal = {
            key: selected_value(
                "escalation_predicate_legal_"
                + "_".join(str(item or "default") for item in key),
                [
                    int(
                        key
                        in {
                            (row["action"], row.get("mode"))
                            for row in _escalation_values(candidate)["legalActions"]
                        }
                    )
                    for candidate in candidates
                ],
                0,
                1,
            )
            for key in legal_keys
        }
        native = {
            "levels": levels,
            "actions": actions,
            "modes": modes,
            "length": length,
            "level": level,
            "locked": locked,
            "terminal": terminal,
            "actionSlots": tuple(action_slots),
            "seatEligible": seat_eligible,
            "teamEligible": team_eligible,
            "legal": legal,
        }
        return cls(cp_model_value, stage, candidates, selected, native, context_id)

    @classmethod
    def _attach_native(cls, stage, model, context_id, declarer_variable=None):
        count = max(1, len(stage.domain.hokum_chain), len(stage.domain.sun_chain))
        levels = {value: index for index, value in enumerate(_LEVELS)}
        actions = {value: index for index, value in enumerate(stage.domain.supported_actions)}
        modes = {None: 0, "open": 1, "locked": 2}
        states = tuple((model.new_int_var(0, 4, f"{context_id}_level_{i}"),
                        model.new_bool_var(f"{context_id}_locked_{i}"),
                        model.new_int_var(0, 4, f"{context_id}_doubler_{i}")) for i in range(count + 1))
        for variable, value in zip(states[0], (levels[stage.domain.default_level], 0, 4)):
            model.add(variable == value)
        active, slots = [], []
        shared = dict(contract=stage.contract, declarer=declarer_variable if declarer_variable is not None else model.new_constant(int(stage.declarer)),
                      match_score=tuple(model.new_constant(value) for value in stage.match_score))
        for index in range(count):
            present = model.new_bool_var(f"{context_id}_active_{index}")
            slot = (model.new_int_var(0, 4, f"{context_id}_actor_{index}"),
                    model.new_int_var(0, 4, f"{context_id}_action_{index}"),
                    model.new_int_var(0, 2, f"{context_id}_mode_{index}"))
            attach_native_escalation_transition_v1(model, stage.domain, **shared,
                active=present, actor=slot[0], action=slot[1], requested_mode=slot[2],
                before=states[index], after=states[index + 1], name=f"{context_id}_transition_{index}")
            if active:
                model.add(active[-1] >= present)
            active.append(present)
            slots.append(slot)
        def any_literal(values, name):
            result = model.new_bool_var(f"{context_id}_{name}")
            if values:
                model.add_max_equality(result, values)
            else:
                model.add(result == 0)
            return result
        by_seat, by_action, by_choice = {}, {}, {}
        for seat in Seat:
            available = attach_native_escalation_transition_v1(model, stage.domain, **shared,
                active=model.new_constant(0), actor=model.new_constant(int(seat)), action=model.new_constant(4),
                requested_mode=model.new_constant(0), before=states[-1], after=states[-1],
                name=f"{context_id}_legal_{int(seat)}", apply_transition=False)
            retained = []
            for action in actions:
                requested = ("open", "locked") if stage.contract.effective == Contract.HOKUM and action in {"double", "four"} else (None,)
                for mode in requested:
                    literal = available[action, mode]
                    key = (action, mode if mode is not None else "open" if action in {"double", "four"} else None)
                    by_action.setdefault(key, []).append(literal)
                    by_choice[seat, *key] = literal
                    retained.append(literal)
            by_seat[seat] = any_literal(retained, f"eligible_{int(seat)}")
        has_choices = any_literal(list(by_seat.values()), "has_choices")
        closed = model.new_bool_var(f"{context_id}_closed")
        model.add(closed <= has_choices)
        model.add(closed <= int(stage.domain.closure_policy == "advance_to_projects_without_announcement"))
        terminal = any_literal([closed, has_choices.Not()], "terminal")
        _, both, _ = _auction_logic_v1(model, context_id)
        seat_eligible = {seat: both(f"visible_eligible_{int(seat)}", literal, terminal.Not()) for seat, literal in by_seat.items()}
        team_eligible = {team: any_literal([literal for seat, literal in seat_eligible.items() if seat.team == team], f"team_{int(team)}") for team in Team}
        legal = {key: both(f"visible_{key[0]}_{key[1]}", any_literal(values, f"action_{key[0]}_{key[1]}"), terminal.Not()) for key, values in by_action.items()}
        legal[stage.domain.closure_action, None] = model.new_constant(int(stage.domain.closure_policy == "advance_to_projects_without_announcement"))
        length = model.new_int_var(0, count, f"{context_id}_length")
        model.add(length == sum(active))
        last_actor = model.new_int_var(0, 4, f"{context_id}_last_actor")
        model.add_element(length, [model.new_constant(4), *(slot[0] for slot in slots)], last_actor)
        last_team = model.new_int_var(0, 2, f"{context_id}_last_team")
        model.add_element(last_actor, [int(seat.team) for seat in Seat] + [2], last_team)
        native = dict(levels=levels, actions=actions, modes=modes, length=length,
                      level=states[-1][0], locked=states[-1][1], terminal=terminal,
                      states=states,
                      stateLocked=tuple(state[1] for state in states), doubleSeat=states[-1][2],
                      lastTeam=last_team, declarer=shared["declarer"],
                      choiceLiterals={key: both(f"choice_{key}", value, terminal.Not()) for key, value in by_choice.items()},
                      actionSlots=tuple(slots), seatEligible=seat_eligible, teamEligible=team_eligible, legal=legal)
        return cls(model, stage, (), (), native, context_id)

    def attach_opportunity_path(self, *, order=None):
        """Expose every decision in a terminal escalation, including private declines.

        With no order supplied, each announcement window has a symbolic seat
        permutation. Ineligible seats and slots after an announcement are not
        decisions. Existing native transitions remain the legality authority.
        """
        if not self.stage.native_cp_sat:
            raise ValueError("escalation opportunities require native constraints")
        if order is not None:
            if (not isinstance(order, Sequence) or isinstance(order, (str, bytes))
                    or any(type(value) not in {Seat, str, int} for value in order)):
                raise ValueError("opportunity order must contain every seat once")
            order = tuple(Seat.parse(value) for value in order)
            if len(order) != 4 or len(set(order)) != 4:
                raise ValueError("opportunity order must contain every seat once")
        model, native = self.model, self.native
        name = self.context_id + "_opportunity"
        if any(variable.name.startswith(name + "_") for variable in model.proto.variables):
            raise ValueError("escalation opportunities are already attached")
        equal, both, either = _auction_logic_v1(model, name)
        model.add(native["terminal"] == 1)
        slots = []
        for window, state in enumerate(native["states"]):
            live = model.new_bool_var(f"{name}_{window}_live")
            model.add(native["length"] >= window).only_enforce_if(live)
            model.add(native["length"] < window).only_enforce_if(live.Not())
            announced = model.new_bool_var(f"{name}_{window}_has_announcement")
            model.add(native["length"] > window).only_enforce_if(announced)
            model.add(native["length"] <= window).only_enforce_if(announced.Not())
            actors = tuple(model.new_int_var(0, 3, f"{name}_{window}_order_{index}") for index in range(4))
            model.add_all_different(actors)
            for index, actor in enumerate(actors):
                if order is not None:
                    model.add(actor == int(order[index]))
                else:
                    model.add(actor == index).only_enforce_if(live.Not())
            announcer = (native["actionSlots"][window][0] if window < len(native["actionSlots"])
                         else model.new_constant(4))
            rank = model.new_int_var(0, 4, f"{name}_{window}_announcer_rank")
            model.add_element(rank, [*actors, model.new_constant(4)], announcer)
            model.add(rank == 4).only_enforce_if(announced.Not())
            by_seat = []
            for seat in Seat:
                legal = attach_native_escalation_transition_v1(model, self.stage.domain,
                    contract=self.stage.contract, declarer=native["declarer"],
                    match_score=tuple(model.new_constant(value) for value in self.stage.match_score),
                    active=model.new_constant(0), actor=model.new_constant(int(seat)),
                    action=model.new_constant(4), requested_mode=model.new_constant(0),
                    before=state, after=state, name=f"{name}_{window}_legal_{int(seat)}",
                    apply_transition=False)
                choices = {}
                for action in native["actions"]:
                    modes = ("open", "locked") if self.stage.contract.effective == Contract.HOKUM and action in {"double", "four"} else (None,)
                    for mode in modes:
                        # Sun Double has one resolved open choice; requested
                        # default/open aliases do not create different decisions.
                        resolved = mode if mode is not None else "open" if action in {"double", "four"} else None
                        code = action + (f":{resolved}" if resolved is not None else "")
                        choices[code] = legal[action, mode]
                by_seat.append(choices)
            announcements = []
            for index, actor in enumerate(actors):
                before_announcement = model.new_bool_var(f"{name}_{window}_{index}_before_announcement")
                model.add(rank >= index).only_enforce_if(before_announcement)
                model.add(rank < index).only_enforce_if(before_announcement.Not())
                available = {}
                for code in by_seat[0]:
                    selected = model.new_bool_var(f"{name}_{window}_{index}_{code}")
                    model.add_element(actor, [row[code] for row in by_seat], selected)
                    available[code] = both(f"{window}_{index}_{code}_available", live, before_announcement, selected)
                active = either(f"{window}_{index}_active", *available.values())
                at_announcement = equal(rank, index, f"{window}_{index}_at_announcement")
                announced_here = both(f"{window}_{index}_announced", active, at_announcement)
                declined = both(f"{window}_{index}_declined", active, at_announcement.Not())
                available[self.stage.domain.closure_action] = active
                slots.append({"window": window, "ordinal": index, "actor": actor,
                    "active": active, "announced": announced_here, "declined": declined,
                    "legal": available, "before": state})
                announcements.append(announced_here)
            model.add(sum(announcements) == announced)
        return tuple(slots)

    def extract_opportunity_path(self, solver, slots):
        """Privileged trace evidence, not an actor payload or a policy certificate."""
        from .ortools_nonplay_graphs import EscalationGraphActionV1, _escalation_window_key

        native = self.native
        actions = {value: key for key, value in native["actions"].items()}
        history, recall, rows = [], {seat: [] for seat in Seat}, []
        for slot in slots:
            if not solver.boolean_value(slot["active"]):
                continue
            actor = Seat(solver.value(slot["actor"]))
            window_hash = _escalation_window_key(history)
            declined = solver.boolean_value(slot["declined"])
            announcement = None
            if declined:
                code = self.stage.domain.closure_action
            else:
                window = slot["window"]
                action = actions[solver.value(native["actionSlots"][window][1])]
                mode = (("locked" if solver.boolean_value(native["stateLocked"][window + 1]) else "open")
                        if action in {"double", "four"} else None)
                code = action + (f":{mode}" if mode is not None else "")
                announcement = EscalationGraphActionV1(actor, action, mode)
            rows.append({"window": slot["window"], "ordinal": slot["ordinal"],
                "actor": actor.name.lower(), "publicAnnouncements": [
                    [row.actor.name.lower(), row.action, row.mode] for row in history],
                "ownDeclinedWindows": list(recall[actor]), "windowHash": window_hash,
                "legalActions": sorted(action for action, literal in slot["legal"].items()
                                       if solver.boolean_value(literal)), "selectedAction": code})
            if declined:
                recall[actor].append(window_hash)
            else:
                history.append(announcement)
        return rows

    def extract_stage_result(self, solver, *, model_hash, solver_status, branches, conflicts):
        if self.stage.native_cp_sat:
            native = self.native
            values = self.solved_native_values(solver)
            actions = tuple(OrToolsEscalationActionV1(Seat.parse(row["actor"]), row["action"], row.get("mode"),
                "locked" if solver.boolean_value(native["stateLocked"][index + 1]) else "open")
                for index, row in enumerate(values["actions"]))
            choices = tuple(sorted((OrToolsEscalationChoiceV1(seat, action, mode)
                for (seat, action, mode), literal in native["choiceLiterals"].items() if solver.boolean_value(literal)),
                key=lambda choice: choice.code))
            doubler = solver.value(native["doubleSeat"])
            core = {"domainIrHash": self.stage.domain.sha256,
                    "declarer": Seat(solver.value(native["declarer"])).name.lower(),
                    "contract": self.stage.contract.value, "matchScore": list(self.stage.match_score),
                    "actions": [row.to_dict() for row in actions], "level": values["level"],
                    "locked": values["locked"], "doubleSeat": Seat(doubler).name.lower() if doubler < 4 else None,
                    "terminal": values["terminal"], "legalNextChoices": [row.to_dict() for row in choices]}
            candidate = _EscalationCandidateV1(actions, values["level"], values["locked"],
                Seat(doubler) if doubler < 4 else None, values["terminal"], choices,
                self.stage.domain.closure_action if solver.boolean_value(native["legal"][self.stage.domain.closure_action, None]) else None,
                canonical_sha256(core))
        else:
            chosen = [candidate for candidate, selected in zip(self.candidates, self.selected, strict=True)
                      if solver.boolean_value(selected)]
            if len(chosen) != 1:
                raise AssertionError("escalation context did not select one plan")
            candidate = chosen[0]
        return self.stage._result(candidate, model_hash=model_hash, solver_status=solver_status,
                                  branches=branches, conflicts=conflicts)

    def bind_native_observation(self, observation: Mapping[str, Any]) -> None:
        """Constrain recorded announcements and state without candidate lookup."""
        if not self.stage.native_cp_sat:
            raise ValueError("native observation binding requires native escalation")
        native = self.native
        records = observation.get("actions")
        if not isinstance(records, (list, tuple)) or len(records) > len(native["actionSlots"]):
            raise ValueError("escalation observation action count is invalid")
        if type(observation.get("terminal")) is not bool:
            raise ValueError("escalation observation terminal must be boolean")
        if observation.get("mode") not in {"open", "locked"}:
            raise ValueError("escalation observation mode is invalid")
        if observation.get("level") not in native["levels"]:
            raise ValueError("escalation observation level is invalid")
        bindings = [(native["length"], len(records)),
                    (native["level"], native["levels"][observation["level"]]),
                    (native["locked"], int(observation["mode"] == "locked")),
                    (native["terminal"], int(observation["terminal"]))]
        if "doubleSeat" in observation:
            bindings.append((native["doubleSeat"], 4 if observation["doubleSeat"] is None else int(Seat.parse(observation["doubleSeat"]))))
        for observation_field, variables, parse in (("eligibleSeats", native["seatEligible"], Seat.parse),
                                        ("eligibleTeams", native["teamEligible"], lambda value: Team[value.upper()])):
            if observation_field in observation:
                if not isinstance(observation[observation_field], (list, tuple)):
                    raise ValueError("escalation observation eligibility must be a list")
                expected = {parse(value) for value in observation[observation_field]}
                bindings.extend((variable, int(key in expected)) for key, variable in variables.items())
        for index, record in enumerate(records):
            if not isinstance(record, Mapping) or record.get("action") not in native["actions"]:
                raise ValueError("escalation observation action is invalid")
            if record.get("requestedMode") not in native["modes"] or record.get("resolvedMode") not in {"open", "locked"}:
                raise ValueError("escalation observation action mode is invalid")
            actor = Seat.parse(record["actor"])
            if record.get("team") != actor.team.name.lower():
                raise ValueError("escalation observation actor/team metadata disagrees")
            bindings.extend(zip(native["actionSlots"][index],
                                (int(actor), native["actions"][record["action"]], native["modes"][record.get("requestedMode")])) )
            bindings.append((native["stateLocked"][index + 1], int(record["resolvedMode"] == "locked")))
        for variable, value in bindings:
            self.model.add(variable == value)

    def solved_native_values(self, solver):
        native = self.native
        action_names = {value: key for key, value in native["actions"].items()}
        mode_names = {value: key for key, value in native["modes"].items()}
        records = []
        for actor, action, mode in native["actionSlots"][:solver.value(native["length"])]:
            mode_value = mode_names[solver.value(mode)]
            records.append({"actor": Seat(solver.value(actor)).name.lower(), "action": action_names[solver.value(action)],
                            **({"mode": mode_value} if mode_value is not None else {})})
        return {"actions": records, "level": _LEVELS[solver.value(native["level"])],
                "locked": solver.boolean_value(native["locked"]), "terminal": solver.boolean_value(native["terminal"]),
                "eligibleSeats": tuple(seat for seat, value in native["seatEligible"].items() if solver.boolean_value(value)),
                "eligibleTeams": tuple(team for team, value in native["teamEligible"].items() if solver.boolean_value(value)),
                "legalActions": [{"action": action, **({"mode": mode} if mode is not None else {})}
                                 for (action, mode), value in sorted(native["legal"].items(), key=lambda item: (item[0][0], item[0][1] or "")) if solver.boolean_value(value)]}

    def __post_init__(self) -> None:
        if not isinstance(self.model, cp_model.CpModel):
            raise TypeError("escalation predicate context requires a CpModel")
        if self.stage.native_cp_sat:
            object.__setattr__(self, "sha256", canonical_sha256({"schemaVersion": 1,
                "contextId": self.context_id, "domainIrHash": self.stage.domain.sha256,
                "authority": "native_escalation_constraints_v1", "model": str(self.model.proto)}))
            return
        if len(self.candidates) != len(self.selected) or not self.candidates:
            raise ValueError("escalation predicate candidates and selectors disagree")
        object.__setattr__(
            self,
            "sha256",
            canonical_sha256(
                {
                    "schemaVersion": 1,
                    "contextId": self.context_id,
                    "domainIrHash": self.stage.domain.sha256,
                    "declarer": self.stage.declarer.name.lower(),
                    "contract": self.stage.contract.value,
                    "matchScore": list(self.stage.match_score),
                    "candidateFingerprints": [
                        candidate.fingerprint for candidate in self.candidates
                    ],
                }
            ),
        )


def _auction_values(
    context: CpSatAuctionPredicateContextV1, plan: Any
) -> dict[str, Any]:
    completion = context.view == "completion"
    actions = plan.actions if completion else plan.prefix_actions
    buyer = plan.buyer if completion else plan.prefix_buyer
    contract = plan.contract if completion else plan.prefix_contract
    trump = plan.trump if completion else plan.prefix_trump
    purchase_round = plan.purchase_round if completion else plan.prefix_purchase_round
    terminal = True if completion else plan.prefix_terminal
    round_index = (
        actions[-1].round_index if completion and actions else plan.current_round
    )
    choices = () if completion else plan.legal_next_choices
    legal_actions = []
    for choice in choices:
        if choice.startswith("hokum:"):
            legal_actions.append(
                {"action": "hokum", "trumpSuit": choice.split(":", 1)[1]}
            )
        else:
            legal_actions.append({"action": choice})
    action_rows = [action.to_dict() for action in actions]
    bidder = next(
        (
            action.actor
            for action in reversed(actions)
            if action.contract == contract and action.buyer == buyer
        ),
        None,
    )
    relation = None
    if contract == Contract.ASHKAL and bidder is not None and buyer is not None:
        relation = "bidder" if buyer == bidder else "bidder_partner"
    return {
        "dealer": context.dealer,
        "buyer": buyer,
        "contract": contract,
        "trump": trump,
        "purchaseRound": purchase_round,
        "round": round_index,
        "actions": action_rows,
        "legalActions": legal_actions,
        "passCount": sum(action["action"] == "pass" for action in action_rows),
        "terminal": terminal,
        "bidder": bidder,
        "ashkalBuyerRelation": relation,
    }


def _auction_observed(kind: str, values: Mapping[str, Any]) -> dict[str, Any]:
    if kind == "auction.dealer":
        return {"seat": values["dealer"].name.lower()}
    if kind == "auction.buyer":
        buyer = values["buyer"]
        return {"seat": buyer.name.lower() if buyer is not None else None}
    if kind == "auction.ashkal_buyer_relation":
        bidder, buyer, contract = values["bidder"], values["buyer"], values["contract"]
        return {
            "contract": contract.value if contract is not None else None,
            "bidder": bidder.name.lower() if bidder is not None else None,
            "buyer": buyer.name.lower() if buyer is not None else None,
            "relation": values["ashkalBuyerRelation"],
        }
    if kind == "auction.contract":
        contract = values["contract"]
        return {"contract": contract.value if contract is not None else None}
    if kind == "auction.trump":
        trump = values["trump"]
        return {"suit": trump.name.lower() if trump is not None else None}
    if kind == "auction.purchase_round":
        return {"round": values["purchaseRound"]}
    if kind == "auction.round":
        return {"round": values["round"]}
    if kind == "auction.action_sequence":
        return {"actions": copy.deepcopy(values["actions"])}
    if kind == "auction.legal_actions":
        return {"actions": copy.deepcopy(values["legalActions"])}
    if kind == "auction.pass_count":
        return {"count": values["passCount"]}
    if kind == "auction.terminal":
        return {"terminal": values["terminal"]}
    raise AssertionError(f"unsupported auction predicate {kind}")


def _escalation_values(candidate: Any) -> dict[str, Any]:
    actions = [
        {
            "actor": action.actor.name.lower(),
            "action": action.action,
            **(
                {"mode": action.requested_mode}
                if action.requested_mode is not None
                else {}
            ),
        }
        for action in candidate.actions
    ]
    unique: dict[tuple[str, str | None], dict[str, Any]] = {}
    if candidate.closure_action is not None:
        unique[(candidate.closure_action, None)] = {"action": candidate.closure_action}
    for choice in candidate.choices:
        row = {"action": choice.action}
        if choice.mode is not None:
            row["mode"] = choice.mode
        unique[(choice.action, choice.mode)] = row
    legal_actions = [
        unique[key] for key in sorted(unique, key=lambda row: (row[0], row[1] or ""))
    ]
    return {
        "actions": actions,
        "level": candidate.level,
        "locked": candidate.locked,
        "eligibleSeats": candidate.eligible_seats,
        "eligibleTeams": candidate.eligible_teams,
        "legalActions": legal_actions,
        "terminal": candidate.terminal,
    }


def _escalation_observed(kind: str, values: Mapping[str, Any]) -> dict[str, Any]:
    if kind == "escalation.level":
        return {"level": values["level"]}
    if kind == "escalation.action_sequence":
        return {"actions": copy.deepcopy(values["actions"])}
    if kind == "escalation.locked":
        return {"locked": values["locked"]}
    if kind == "escalation.eligible_actor":
        return {
            "eligibleSeats": [seat.name.lower() for seat in values["eligibleSeats"]],
            "eligibleTeams": [team.name.lower() for team in values["eligibleTeams"]],
        }
    if kind == "escalation.legal_actions":
        return {"actions": copy.deepcopy(values["legalActions"])}
    if kind == "escalation.terminal":
        return {"terminal": values["terminal"]}
    raise AssertionError(f"unsupported escalation predicate {kind}")


class AuctionEscalationPredicateLoweringV1:
    """Lower registered predicates to exact support sets over Domain-IR plans."""

    def __init__(self) -> None:
        self._registry = ScenarioPredicateRegistryV1()

    @property
    def registrations(self) -> tuple[AuctionEscalationPredicateRegistrationV1, ...]:
        return AUCTION_ESCALATION_PREDICATE_REGISTRATIONS_V1

    def lower_predicate(
        self,
        predicate: ScenarioPredicateV1,
        context: CpSatAuctionPredicateContextV1 | CpSatEscalationPredicateContextV1,
    ) -> CpSatPredicateLoweringV1:
        registration = _REGISTRATION_BY_KIND.get(predicate.kind)
        if registration is None:
            raise ValueError(
                f"predicate is not an auction/escalation predicate: {predicate.kind}"
            )
        definition = self._registry.require(predicate)
        expected_context = (
            CpSatAuctionPredicateContextV1
            if registration.stage == "auction"
            else CpSatEscalationPredicateContextV1
        )
        if not isinstance(context, expected_context):
            raise TypeError(f"{predicate.kind} requires a {registration.stage} context")

        if isinstance(context, CpSatAuctionPredicateContextV1):
            if not isinstance(context.fragment, OrToolsNativeAuctionFragmentV1):
                candidates = context.fragment.plans
                selected = context.fragment.selected
                values = tuple(
                    _auction_values(context, candidate) for candidate in candidates
                )
            observe: Callable[[Mapping[str, Any]], dict[str, Any]] = partial(
                _auction_observed, predicate.kind
            )
            literal = self._lower_native_auction(predicate, context)
            binding = {
                "nativeVariables": sorted(
                    variable.name
                    for variable in self._native_auction_variables(context)
                )
            }
        else:
            candidates = context.candidates
            selected = context.selected
            values = () if context.stage.native_cp_sat else tuple(_escalation_values(candidate) for candidate in candidates)
            observe = partial(_escalation_observed, predicate.kind)
            literal = self._lower_native_escalation(predicate, context)
            binding = {
                "nativeVariables": sorted(
                    variable.name
                    for variable in self._native_escalation_variables(context)
                )
            }
        lowering_hash = canonical_sha256(
            {
                "schemaVersion": 1,
                "predicateHash": predicate.sha256,
                "contextHash": context.sha256,
                "definitionHash": definition.sha256,
                "evidenceExtractor": registration.evidence_extractor_id,
                **binding,
            }
        )

        def extract(solver: cp_model.CpSolver) -> Mapping[str, Any]:
            if isinstance(context, CpSatEscalationPredicateContextV1) and context.stage.native_cp_sat:
                return _escalation_observed(predicate.kind, context.solved_native_values(solver))
            if isinstance(context, CpSatAuctionPredicateContextV1) and isinstance(context.fragment, OrToolsNativeAuctionFragmentV1):
                return _auction_observed(predicate.kind, context.solved_native_values(solver))
            chosen = [
                index
                for index, selector in enumerate(selected)
                if solver.boolean_value(selector)
            ]
            if len(chosen) != 1:
                raise AssertionError(
                    "predicate context did not select exactly one plan"
                )
            return observe(values[chosen[0]])

        return CpSatPredicateLoweringV1(
            predicate.predicate_id,
            predicate.kind,
            literal,
            registration.evidence_extractor_id,
            lowering_hash,
            extract,
        )

    @staticmethod
    def _native_auction_variables(
        context: CpSatAuctionPredicateContextV1,
    ) -> tuple[cp_model.IntVar, ...]:
        native = context.native
        return (
            native["length"],
            native["dealer"],
            native["buyer"],
            native["contract"],
            native["trump"],
            native["purchaseRound"],
            native["round"],
            native["passCount"],
            native["terminal"],
            native["ashkalRelation"],
            *((native["bidder"],) if "bidder" in native else ()),
            *(variable for slot in native["actionSlots"] for variable in slot),
            *native["legal"].values(),
        )

    @staticmethod
    def _lower_native_auction(
        predicate: ScenarioPredicateV1,
        context: CpSatAuctionPredicateContextV1,
    ) -> cp_model.IntVar:
        model, native = context.model, context.native
        arguments = dict(predicate.arguments)
        prefix = (
            f"predicate_{predicate.predicate_id}_{predicate.kind.replace('.', '_')}"
        )

        def equals(variable: cp_model.IntVar, value: int, name: str) -> cp_model.IntVar:
            result = model.new_bool_var(name)
            model.add(variable == value).only_enforce_if(result)
            model.add(variable != value).only_enforce_if(result.negated())
            return result

        def conjunction(terms: list[cp_model.IntVar], name: str) -> cp_model.IntVar:
            result = model.new_bool_var(name)
            if terms:
                model.add_min_equality(result, terms)
            else:
                model.add(result == 1)
            return result

        kind = predicate.kind
        if kind == "auction.dealer":
            return equals(native["dealer"], int(Seat.parse(arguments["seat"])), prefix)
        if kind == "auction.buyer":
            return equals(native["buyer"], int(Seat.parse(arguments["seat"])), prefix)
        if kind == "auction.contract":
            return equals(
                native["contract"],
                native["contracts"][Contract.parse(arguments["contract"])],
                prefix,
            )
        if kind == "auction.trump":
            return equals(native["trump"], int(Suit.parse(arguments["suit"])), prefix)
        if kind == "auction.purchase_round":
            return equals(native["purchaseRound"], int(arguments["round"]), prefix)
        if kind == "auction.round":
            return equals(native["round"], int(arguments["round"]), prefix)
        if kind == "auction.terminal":
            return equals(native["terminal"], int(arguments["terminal"]), prefix)
        if kind == "auction.ashkal_buyer_relation":
            relation = {"bidder": 1, "bidder_partner": 2}[arguments["relation"]]
            return equals(native["ashkalRelation"], relation, prefix)
        if kind == "auction.pass_count":
            result = model.new_bool_var(prefix)
            minimum, maximum = int(arguments["minimum"]), int(arguments["maximum"])
            if isinstance(context.fragment, OrToolsNativeAuctionFragmentV1):
                below = model.new_bool_var(prefix + "_below")
                above = model.new_bool_var(prefix + "_above")
                model.add(native["passCount"] < minimum).only_enforce_if(below)
                model.add(native["passCount"] >= minimum).only_enforce_if(below.Not())
                model.add(native["passCount"] > maximum).only_enforce_if(above)
                model.add(native["passCount"] <= maximum).only_enforce_if(above.Not())
                model.add_min_equality(result, [below.Not(), above.Not()])
                return result
            model.add_allowed_assignments(
                [native["passCount"], result],
                [
                    (value, int(minimum <= value <= maximum))
                    for value in range(len(native["actionSlots"]) + 1)
                ],
            )
            return result
        if kind == "auction.action_sequence":
            expected = list(arguments["actions"])
            if len(expected) > len(native["actionSlots"]):
                result = model.new_bool_var(prefix)
                model.add(result == 0)
                return result
            terms: list[cp_model.IntVar] = []
            if arguments["complete"]:
                terms.append(
                    equals(native["length"], len(expected), prefix + "_length")
                )
            else:
                enough = model.new_bool_var(prefix + "_length")
                model.add(native["length"] >= len(expected)).only_enforce_if(enough)
                model.add(native["length"] < len(expected)).only_enforce_if(
                    enough.negated()
                )
                terms.append(enough)
            for index, row in enumerate(expected):
                round_index, actor, action, trump = native["actionSlots"][index]
                if row["action"] not in native["actions"]:
                    impossible = model.new_bool_var(f"{prefix}_{index}_unsupported")
                    model.add(impossible == 0)
                    terms.append(impossible)
                    continue
                if "roundIndex" in row:
                    terms.append(
                        equals(
                            round_index,
                            int(row["roundIndex"]),
                            f"{prefix}_{index}_round",
                        )
                    )
                if "actor" in row:
                    terms.append(
                        equals(
                            actor,
                            int(Seat.parse(row["actor"])),
                            f"{prefix}_{index}_actor",
                        )
                    )
                terms.append(
                    equals(
                        action,
                        native["actions"][row["action"]],
                        f"{prefix}_{index}_action",
                    )
                )
                if "trumpSuit" in row:
                    expected_trump = (
                        4
                        if row["trumpSuit"] is None
                        else int(Suit.parse(row["trumpSuit"]))
                    )
                    terms.append(
                        equals(trump, expected_trump, f"{prefix}_{index}_trump")
                    )
            return conjunction(terms, prefix)
        if kind == "auction.legal_actions":
            expected = {
                (row["action"], row.get("trumpSuit")) for row in arguments["actions"]
            }
            terms = [
                equals(variable, int(key in expected), f"{prefix}_{index}")
                for index, (key, variable) in enumerate(native["legal"].items())
                if arguments["complete"] or key in expected
            ]
            if expected - set(native["legal"]):
                impossible = model.new_bool_var(prefix + "_unsupported")
                model.add(impossible == 0)
                terms.append(impossible)
            return conjunction(terms, prefix)
        raise AssertionError(f"unsupported auction predicate {kind}")

    @staticmethod
    def _native_escalation_variables(
        context: CpSatEscalationPredicateContextV1,
    ) -> tuple[cp_model.IntVar, ...]:
        native = context.native
        return (
            native["length"],
            native["level"],
            native["locked"],
            native["terminal"],
            *(variable for slot in native["actionSlots"] for variable in slot),
            *native["seatEligible"].values(),
            *native["teamEligible"].values(),
            *native["legal"].values(),
        )

    @staticmethod
    def _lower_native_escalation(
        predicate: ScenarioPredicateV1,
        context: CpSatEscalationPredicateContextV1,
    ) -> cp_model.IntVar:
        model, native = context.model, context.native
        arguments = dict(predicate.arguments)
        prefix = (
            f"predicate_{predicate.predicate_id}_{predicate.kind.replace('.', '_')}"
        )

        def equals(variable: cp_model.IntVar, value: int, name: str) -> cp_model.IntVar:
            result = model.new_bool_var(name)
            model.add(variable == value).only_enforce_if(result)
            model.add(variable != value).only_enforce_if(result.negated())
            return result

        def conjunction(terms: list[cp_model.IntVar], name: str) -> cp_model.IntVar:
            result = model.new_bool_var(name)
            if terms:
                model.add_min_equality(result, terms)
            else:
                model.add(result == 1)
            return result

        if predicate.kind == "escalation.level":
            return equals(native["level"], native["levels"][arguments["level"]], prefix)
        if predicate.kind == "escalation.locked":
            return equals(native["locked"], int(arguments["locked"]), prefix)
        if predicate.kind == "escalation.terminal":
            return equals(native["terminal"], int(arguments["terminal"]), prefix)
        if predicate.kind == "escalation.eligible_actor":
            seat = Seat.parse(arguments["seat"]) if arguments.get("seat") else None
            team = Team[str(arguments["team"]).upper()]
            variable = (
                native["seatEligible"][seat]
                if seat is not None
                else native["teamEligible"][team]
            )
            if seat is not None and seat.team != team:
                result = model.new_bool_var(prefix)
                model.add(result == int(not arguments["eligible"]))
                return result
            return equals(variable, int(arguments["eligible"]), prefix)
        if predicate.kind == "escalation.action_sequence":
            expected = list(arguments["actions"])
            if len(expected) > len(native["actionSlots"]):
                result = model.new_bool_var(prefix)
                model.add(result == 0)
                return result
            terms: list[cp_model.IntVar] = []
            if arguments.get("complete", True):
                terms.append(
                    equals(native["length"], len(expected), prefix + "_length")
                )
            else:
                enough = model.new_bool_var(prefix + "_length")
                model.add(native["length"] >= len(expected)).only_enforce_if(enough)
                model.add(native["length"] < len(expected)).only_enforce_if(
                    enough.negated()
                )
                terms.append(enough)
            for index, row in enumerate(expected):
                actor, action, mode = native["actionSlots"][index]
                if row["action"] not in native["actions"]:
                    impossible = model.new_bool_var(f"{prefix}_{index}_unsupported")
                    model.add(impossible == 0)
                    terms.append(impossible)
                    continue
                terms.extend(
                    (
                        equals(
                            actor,
                            int(Seat.parse(row["actor"])),
                            f"{prefix}_{index}_actor",
                        ),
                        equals(
                            action,
                            native["actions"][row["action"]],
                            f"{prefix}_{index}_action",
                        ),
                    )
                )
                if "mode" in row:
                    terms.append(
                        equals(
                            mode, native["modes"][row["mode"]], f"{prefix}_{index}_mode"
                        )
                    )
            return conjunction(terms, prefix)
        if predicate.kind == "escalation.legal_actions":
            expected = {
                (row["action"], row.get("mode")) for row in arguments["actions"]
            }
            terms = [
                equals(variable, int(key in expected), f"{prefix}_{index}")
                for index, (key, variable) in enumerate(native["legal"].items())
                if arguments["complete"] or key in expected
            ]
            missing = expected - set(native["legal"])
            if missing:
                impossible = model.new_bool_var(prefix + "_unsupported")
                model.add(impossible == 0)
                terms.append(impossible)
            return conjunction(terms, prefix)
        raise AssertionError(f"unsupported escalation predicate {predicate.kind}")


__all__ = [
    "AUCTION_ESCALATION_PREDICATE_REGISTRATIONS_V1",
    "AuctionEscalationPredicateLoweringV1",
    "AuctionEscalationPredicateRegistrationV1",
    "CpSatAuctionPredicateContextV1",
    "CpSatEscalationPredicateContextV1",
]
