"""Deterministic V4 structural lowering with predicate-linked diagnostics."""

from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from baloot_platform.core.cards import FULL_DECK, Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, Rank, Seat, Suit
from baloot_platform.core.rules import RuleBook

from .ortools_infeasible_core import sound_infeasible_assumption_indexes
from .scenario_contracts import (
    ResearchScenarioV1,
    ScenarioExpressionV1,
    canonical_sha256,
)
from .scenario_registry import ScenarioPredicateRegistryV1


def partial_trick_bridge_objective(
    scenario: ResearchScenarioV1,
) -> tuple[Seat, int, Suit | None, Suit | None] | None:
    decision_seats = {
        Seat.parse(predicate.arguments["seat"])
        for predicate in scenario.predicates()
        if predicate.hardness == "hard" and predicate.kind == "play.current_player"
    }
    selector = scenario.decision_point["selector"]
    if scenario.decision_point["kind"] == "play" and selector.get("actingSeat"):
        decision_seats.add(Seat.parse(selector["actingSeat"]))
    target_tricks = {
        int(predicate.arguments["minimum"])
        for predicate in scenario.predicates()
        if predicate.hardness == "hard"
        and predicate.kind == "play.trick_index"
        and predicate.arguments["minimum"] == predicate.arguments["maximum"]
    }
    if (
        len(decision_seats) != 1
        or len(target_tricks) != 1
        or "remainingCards" not in selector
    ):
        return None
    target_trick = next(iter(target_tricks))
    cards_played = 32 - (4 * target_trick) - int(selector["remainingCards"])
    if not 0 <= cards_played <= 3:
        return None
    target_leader = next(iter(decision_seats))
    for _ in range(cards_played):
        target_leader = target_leader.previous()
    protected = {
        Card(int(predicate.arguments["card"]))
        for predicate in scenario.predicates()
        if predicate.hardness == "hard"
        and predicate.kind in {"deal.remaining_card_owner", "play.current_trick_card"}
        and Seat.parse(predicate.arguments["seat"]) == target_leader
    }
    ranked_suits = sorted(
        (suit for suit in Suit if any(card.suit == suit for card in protected)),
        key=lambda suit: (
            -sum(card.suit == suit for card in protected),
            int(suit),
        ),
    )
    turnover_suit = ranked_suits[0] if len(ranked_suits) > 1 else None
    bridge_suit = (
        ranked_suits[1]
        if len(ranked_suits) > 1
        else (ranked_suits[0] if ranked_suits else None)
    )
    return target_leader, target_trick, bridge_suit, turnover_suit


@dataclass(frozen=True, slots=True)
class ScenarioConstructionPlanV1:
    scenario_hash: str
    structural_predicate_ids: tuple[str, ...]
    guided_predicate_ids: tuple[str, ...]
    postcondition_predicate_ids: tuple[str, ...]
    exact_input_predicate_ids: tuple[str, ...]
    structural_expression_complete: bool
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "scenarioHash": self.scenario_hash,
            "structuralPredicateIds": list(self.structural_predicate_ids),
            "guidedPredicateIds": list(self.guided_predicate_ids),
            "postconditionPredicateIds": list(self.postcondition_predicate_ids),
            "exactInputPredicateIds": list(self.exact_input_predicate_ids),
            "structuralExpressionComplete": self.structural_expression_complete,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ScenarioStructuralResultV1:
    status: str
    plan_hash: str
    model_hash: str
    solver_status: str
    deck_order: tuple[int, ...] | None
    predicate_evidence: tuple[Mapping[str, Any], ...]
    no_good_hashes: tuple[str, ...]
    infeasible_core_predicate_ids: tuple[str, ...]
    nodes: int
    elapsed_seconds: float
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "status": self.status,
            "planHash": self.plan_hash,
            "modelHash": self.model_hash,
            "solverStatus": self.solver_status,
            "deckOrder": list(self.deck_order) if self.deck_order is not None else None,
            "predicateEvidence": [
                copy.deepcopy(dict(item)) for item in self.predicate_evidence
            ],
            "noGoodHashes": list(self.no_good_hashes),
            "infeasibleCorePredicateIds": list(self.infeasible_core_predicate_ids),
            "nodes": self.nodes,
            "elapsedSeconds": self.elapsed_seconds,
            "sha256": self.sha256,
        }


class ScenarioStructuralCompilerV1:
    _LOWERABLE_KINDS = frozenset(
        {
            "deal.card_owner",
            "deal.card_allowed",
            "deal.card_forbidden",
            "deal.suit_count",
            "deal.rank_count",
            "deal.high_card_count",
            "deal.low_card_count",
            "deal.purchase_card",
            "deal.purchase_suit",
            "auction.dealer",
            "project.exists",
        }
    )

    def __init__(self, registry: ScenarioPredicateRegistryV1 | None = None):
        self.registry = registry or ScenarioPredicateRegistryV1()

    def plan(self, scenario: ResearchScenarioV1) -> ScenarioConstructionPlanV1:
        grouped: dict[str, list[str]] = {
            "cp_sat_exact": [],
            "incremental_actor_safe": [],
            "guided_search": [],
            "postcondition_only": [],
            "exact_input_only": [],
        }
        for predicate in scenario.predicates():
            definition = self.registry.require(predicate)
            grouped.setdefault(definition.construction_capability, []).append(
                predicate.predicate_id
            )
        structural = tuple(sorted(grouped["cp_sat_exact"]))
        guided = tuple(
            sorted(grouped["incremental_actor_safe"] + grouped["guided_search"])
        )
        postcondition = tuple(sorted(grouped["postcondition_only"]))
        exact_input = tuple(sorted(grouped["exact_input_only"]))
        complete = all(
            self.registry.require(predicate).construction_capability == "cp_sat_exact"
            and predicate.kind in self._LOWERABLE_KINDS
            for predicate in scenario.constraints.predicates()
            if predicate.hardness == "hard"
        )
        core = {
            "schemaVersion": 1,
            "scenarioHash": scenario.sha256,
            "structuralPredicateIds": list(structural),
            "guidedPredicateIds": list(guided),
            "postconditionPredicateIds": list(postcondition),
            "exactInputPredicateIds": list(exact_input),
            "structuralExpressionComplete": complete,
        }
        return ScenarioConstructionPlanV1(
            scenario.sha256,
            structural,
            guided,
            postcondition,
            exact_input,
            complete,
            canonical_sha256(core),
        )

    @staticmethod
    def _range_literal(model, expression, minimum: int, maximum: int, name: str):
        literal = model.new_bool_var(name)
        model.add(expression >= minimum).only_enforce_if(literal)
        model.add(expression <= maximum).only_enforce_if(literal)
        below = model.new_bool_var(f"{name}_below")
        above = model.new_bool_var(f"{name}_above")
        model.add(expression < minimum).only_enforce_if(below)
        model.add(expression >= minimum).only_enforce_if(~below)
        model.add(expression > maximum).only_enforce_if(above)
        model.add(expression <= maximum).only_enforce_if(~above)
        model.add_bool_or([below, above]).only_enforce_if(~literal)
        model.add(below + above == 0).only_enforce_if(literal)
        return literal

    @staticmethod
    def _project_pattern_literal(
        model,
        *,
        predicate,
        membership,
        rules: RuleBook,
        contract: Contract | None,
        buyer: Seat | None,
        name: str,
    ):
        arguments = dict(predicate.arguments)
        if (
            arguments.get("state")
            not in {"eligible", "declared", "comparison_winner", "awarded"}
            or contract is None
            or buyer is None
        ):
            return None
        team = str(arguments["team"])
        seats = [
            seat
            for seat in Seat
            if ("north_south" if int(seat.team) == 0 else "west_east") == team
        ]
        if arguments.get("owner") is not None:
            owner = Seat.parse(arguments["owner"])
            if owner not in seats:
                return model.new_constant(0)
            seats = [owner]
        patterns = {
            int(project.card_mask)
            for project in rules.project_domain.candidates(
                Seat.NORTH,
                FULL_DECK,
                contract,
            )
            if project.kind == arguments["kind"]
            and project.value == int(arguments["value"])
        }
        pattern_literals = []
        for seat in seats:
            for index, pattern in enumerate(sorted(patterns)):
                cards = [card for card in range(32) if pattern & (1 << card)]
                matched = model.new_bool_var(
                    f"{name}_{seat.name.lower()}_pattern_{index}"
                )
                required = [membership["post_purchase", card, seat] for card in cards]
                model.add_bool_and(required).only_enforce_if(matched)
                model.add_bool_or([~item for item in required]).only_enforce_if(
                    ~matched
                )
                pattern_literals.append(matched)
        if not pattern_literals:
            return model.new_constant(0)
        literal = model.new_bool_var(name)
        model.add_bool_or(pattern_literals).only_enforce_if(literal)
        model.add_bool_and([~item for item in pattern_literals]).only_enforce_if(
            ~literal
        )
        return literal

    def solve(
        self,
        scenario: ResearchScenarioV1,
        *,
        rules: RuleBook,
        dealer: Seat,
        seed: int,
        maximum_seconds: float,
        excluded_deck_orders: Sequence[Sequence[int]] = (),
    ) -> ScenarioStructuralResultV1:
        if maximum_seconds <= 0:
            raise ValueError("scenario structural maximum_seconds must be positive")
        try:
            from ortools.sat.python import cp_model
        except ImportError as exc:
            raise RuntimeError(
                "OR-Tools is required for V4 structural construction"
            ) from exc
        started = time.perf_counter()
        plan = self.plan(scenario)
        buyer_values = {
            str(predicate.arguments["seat"])
            for predicate in scenario.constraints.predicates()
            if predicate.kind == "auction.buyer" and predicate.hardness == "hard"
        }
        if len(buyer_values) > 1:
            raise ValueError("scenario declares more than one hard auction buyer")
        buyer = Seat.parse(next(iter(buyer_values))) if buyer_values else None
        contract_values = {
            str(predicate.arguments["contract"])
            for predicate in scenario.constraints.predicates()
            if predicate.kind == "auction.contract" and predicate.hardness == "hard"
        }
        contract = (
            Contract.parse(next(iter(contract_values)))
            if len(contract_values) == 1
            else None
        )
        if (
            any(
                str(predicate.arguments.get("handStage")) == "post_purchase"
                for predicate in scenario.constraints.predicates()
                if predicate.kind.startswith("deal.")
            )
            and buyer is None
        ):
            raise ValueError(
                "post-purchase structural constraints require one hard auction buyer"
            )
        layout = DealLayoutV1.build(rules.structural_domain, dealer=dealer, buyer=buyer)
        model = cp_model.CpModel()
        positions = {
            (card, position): model.new_bool_var(f"card_{card}_position_{position}")
            for card in range(32)
            for position in range(32)
        }
        for card in range(32):
            model.add_exactly_one(positions[card, position] for position in range(32))
        for position in range(32):
            model.add_exactly_one(positions[card, position] for card in range(32))
        excluded: list[tuple[int, ...]] = []
        for index, raw_deck in enumerate(excluded_deck_orders):
            deck = tuple(int(card) for card in raw_deck)
            if len(deck) != 32 or set(deck) != set(range(32)):
                raise ValueError(f"scenario no-good deck {index} is not a permutation")
            excluded.append(deck)
            model.add(
                sum(positions[card, position] for position, card in enumerate(deck))
                <= 31
            )
        no_good_hashes = tuple(canonical_sha256(list(deck)) for deck in excluded)
        membership = {}
        for hand_stage, final in (("initial_five", False), ("post_purchase", True)):
            if final and buyer is None:
                continue
            for card in range(32):
                for seat in Seat:
                    variable = model.new_bool_var(
                        f"card_{card}_{hand_stage}_{seat.name.lower()}"
                    )
                    model.add(
                        variable
                        == sum(
                            positions[card, position]
                            for position in layout.positions_for(seat, final=final)
                        )
                    )
                    membership[hand_stage, card, seat] = variable
        protected_at_decision: dict[tuple[Seat, Suit], set[int]] = {}
        decision_seats = {
            Seat.parse(predicate.arguments["seat"])
            for predicate in scenario.predicates()
            if predicate.hardness == "hard" and predicate.kind == "play.current_player"
        }
        for predicate in scenario.predicates():
            if predicate.hardness != "hard":
                continue
            args = predicate.arguments
            if predicate.kind in {
                "deal.remaining_card_owner",
                "play.current_trick_card",
            }:
                card = Card(int(args["card"]))
                protected_at_decision.setdefault(
                    (Seat.parse(args["seat"]), card.suit), set()
                ).add(card.id)
            elif (
                predicate.kind == "state.legal_cards_include"
                and len(decision_seats) == 1
            ):
                seat = next(iter(decision_seats))
                for value in args["cards"]:
                    card = Card(int(value))
                    protected_at_decision.setdefault((seat, card.suit), set()).add(
                        card.id
                    )
            elif (
                predicate.kind == "state.legal_play_actions_include"
                and len(decision_seats) == 1
            ):
                seat = next(iter(decision_seats))
                for value in args["actions"]:
                    card = Card(int(value["card"]))
                    protected_at_decision.setdefault((seat, card.suit), set()).add(
                        card.id
                    )
        bridge_objective = partial_trick_bridge_objective(scenario)
        predicate_literals: dict[str, Any] = {}
        evidence: list[dict[str, Any]] = []
        for predicate in scenario.constraints.predicates():
            if predicate.kind not in self._LOWERABLE_KINDS:
                continue
            args = dict(predicate.arguments)
            name = predicate.predicate_id.lower().replace("-", "_")
            literal = None
            hand_stage = str(args.get("handStage") or "initial_five")
            if predicate.kind == "deal.card_owner":
                literal = membership[
                    hand_stage, int(args["card"]), Seat.parse(args["seat"])
                ]
            elif predicate.kind == "deal.card_forbidden":
                literal = membership[
                    hand_stage, int(args["card"]), Seat.parse(args["seat"])
                ].Not()
            elif predicate.kind == "deal.card_allowed":
                literal = model.new_bool_var(name)
                allowed = [
                    membership[hand_stage, int(args["card"]), Seat.parse(seat)]
                    for seat in args["seats"]
                ]
                model.add(sum(allowed) == 1).only_enforce_if(literal)
                model.add(sum(allowed) == 0).only_enforce_if(~literal)
            elif predicate.kind == "deal.purchase_card":
                literal = positions[int(args["card"]), layout.purchase_position]
            elif predicate.kind == "deal.purchase_suit":
                matching = sum(
                    positions[card.id, layout.purchase_position]
                    for card in FULL_DECK
                    if card.suit == Suit.parse(args["suit"])
                )
                literal = model.new_bool_var(name)
                model.add(matching == 1).only_enforce_if(literal)
                model.add(matching == 0).only_enforce_if(~literal)
            elif predicate.kind == "auction.dealer":
                literal = model.new_constant(int(dealer == Seat.parse(args["seat"])))
            elif predicate.kind == "project.exists":
                literal = self._project_pattern_literal(
                    model,
                    predicate=predicate,
                    membership=membership,
                    rules=rules,
                    contract=contract,
                    buyer=buyer,
                    name=name,
                )
            else:
                seat = Seat.parse(args["seat"])
                if predicate.kind == "deal.suit_count":
                    cards = [
                        card.id
                        for card in (Card(index) for index in range(32))
                        if card.suit == Suit.parse(args["suit"])
                    ]
                elif predicate.kind == "deal.rank_count":
                    cards = [
                        card.id
                        for card in (Card(index) for index in range(32))
                        if card.rank == Rank.parse(args["rank"])
                    ]
                elif predicate.kind == "deal.high_card_count":
                    cards = [
                        card.id
                        for card in (Card(index) for index in range(32))
                        if card.rank in {Rank.ACE, Rank.TEN, Rank.KING}
                    ]
                else:
                    cards = [
                        card.id
                        for card in (Card(index) for index in range(32))
                        if card.rank in {Rank.SEVEN, Rank.EIGHT, Rank.NINE}
                    ]
                literal = self._range_literal(
                    model,
                    sum(membership[hand_stage, card, seat] for card in cards),
                    int(args["minimum"]),
                    int(args["maximum"]),
                    name,
                )
            if literal is None:
                continue
            predicate_literals[predicate.predicate_id] = literal
            evidence.append(
                {
                    "predicateId": predicate.predicate_id,
                    "predicateHash": predicate.sha256,
                    "lowering": (
                        (
                            "cp_sat_exact_eligible_project_pattern"
                            if predicate.arguments.get("state") == "eligible"
                            else "cp_sat_necessary_project_pattern"
                        )
                        if predicate.kind == "project.exists"
                        else "cp_sat_exact"
                    ),
                }
            )

        def expression_literal(expression: ScenarioExpressionV1):
            if expression.predicate is not None:
                if expression.predicate.hardness != "hard":
                    return None
                return predicate_literals.get(expression.predicate.predicate_id)
            children = [expression_literal(child) for child in expression.children]
            if expression.op == "all":
                available = [child for child in children if child is not None]
                if not available:
                    return None
                result = model.new_bool_var(f"expr_{expression.sha256[:12]}")
                model.add_bool_and(available).only_enforce_if(result)
                model.add_bool_or([~item for item in available]).only_enforce_if(
                    ~result
                )
                return result
            if any(child is None for child in children):
                return None
            values = [child for child in children if child is not None]
            result = model.new_bool_var(f"expr_{expression.sha256[:12]}")
            if expression.op == "any":
                model.add_bool_or(values).only_enforce_if(result)
                model.add_bool_and([~item for item in values]).only_enforce_if(~result)
            elif expression.op == "not":
                model.add(result + values[0] == 1)
            elif expression.op == "count":
                assert expression.minimum is not None and expression.maximum is not None
                result = self._range_literal(
                    model,
                    sum(values),
                    expression.minimum,
                    expression.maximum,
                    f"count_{expression.sha256[:12]}",
                )
            return result

        root_literal = expression_literal(scenario.constraints)
        lowered_ids = tuple(sorted(predicate_literals))
        assumption_predicates: dict[int, tuple[str, Any]] = {}
        flat_hard_all = scenario.constraints.op == "all" and all(
            child.predicate is not None
            and child.predicate.hardness == "hard"
            and child.predicate.predicate_id in predicate_literals
            for child in scenario.constraints.children
        )
        if flat_hard_all:
            for child in scenario.constraints.children:
                assert child.predicate is not None
                predicate_id = child.predicate.predicate_id
                assumption = model.new_bool_var(
                    f"assume_{predicate_id.lower().replace('-', '_')}"
                )
                model.add(predicate_literals[predicate_id] == 1).only_enforce_if(
                    assumption
                )
                model.add_assumption(assumption)
                assumption_predicates[assumption.index] = (
                    predicate_id,
                    assumption,
                )
        elif root_literal is not None:
            assumption = model.new_bool_var("scenario_structural_assumption")
            model.add(root_literal == 1).only_enforce_if(assumption)
            model.add_assumption(assumption)
        evidence = [
            {
                **item,
                "expressionEnforced": root_literal is not None,
            }
            for item in evidence
        ]
        final_position_owner = {
            position: seat
            for seat in Seat
            for position in (
                layout.positions_for(seat, final=True) if buyer is not None else ()
            )
        }
        bridge_hints: list[tuple[str, Seat, Card]] = []
        bridge_hint_variables: list[Any] = []
        if bridge_objective is not None and buyer is not None:
            target_leader, _, bridge_suit, turnover_suit = bridge_objective
            if bridge_suit is not None:
                fixed_owners = {
                    int(predicate.arguments["card"]): Seat.parse(
                        predicate.arguments["seat"]
                    )
                    for predicate in scenario.constraints.predicates()
                    if predicate.hardness == "hard"
                    and predicate.kind == "deal.card_owner"
                    and predicate.arguments.get("handStage") == "post_purchase"
                }
                protected_cards = {
                    card for cards in protected_at_decision.values() for card in cards
                }
                remaining_slots = {
                    seat: 8 - sum(owner == seat for owner in fixed_owners.values())
                    for seat in Seat
                }
                assigned: set[int] = set()
                lanes = (
                    ("turnover", turnover_suit, tuple(Seat)),
                    (
                        "turnover_buffer",
                        turnover_suit,
                        tuple(
                            seat
                            for (seat, suit), cards in sorted(
                                protected_at_decision.items(),
                                key=lambda item: (
                                    len(item[1]),
                                    int(item[0][0]),
                                ),
                            )
                            if suit == turnover_suit and cards
                        ),
                    ),
                    (
                        "bridge",
                        bridge_suit,
                        (target_leader,)
                        + tuple(seat for seat in Seat if seat != target_leader),
                    ),
                )
                for lane, suit, seats in lanes:
                    if suit is None:
                        continue
                    available = sorted(
                        (
                            card
                            for card in FULL_DECK
                            if card.suit == suit
                            and card.id not in protected_cards
                            and card.id not in fixed_owners
                            and card.id not in assigned
                        ),
                        key=lambda card: (
                            int(card.rank),
                            card.id,
                        ),
                    )
                    for seat in seats:
                        if remaining_slots[seat] <= 0 or not available:
                            continue
                        card = available.pop(
                            -1 if lane == "bridge" and seat == target_leader else 0
                        )
                        variable = membership["post_purchase", card.id, seat]
                        model.add_hint(variable, 1)
                        bridge_hint_variables.append(variable)
                        bridge_hints.append((lane, seat, card))
                        assigned.add(card.id)
                        remaining_slots[seat] -= 1
        costs = {
            key: int(
                hashlib.sha256(
                    f"{seed}:{plan.sha256}:{key[0]}:{key[1]}".encode("ascii")
                ).hexdigest()[:8],
                16,
            )
            % 1_000_003
            + 2_000_006
            * len(
                protected_at_decision.get(
                    (
                        final_position_owner.get(key[1]),
                        Card(key[0]).suit,
                    ),
                    (),
                )
            )
            for key in positions
        }
        model.add_decision_strategy(
            [
                positions[key]
                for key in sorted(
                    positions,
                    key=lambda key: (costs[key], key),
                )
            ],
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MIN_VALUE,
        )
        model_hash = canonical_sha256(
            {
                "schemaVersion": 1,
                "kind": "scenario_structural_model_v1",
                "selectionPolicy": "seeded_actor_safe_survival_order_v7",
                "planHash": plan.sha256,
                "rulesHash": rules.sha256,
                "dealer": dealer.name.lower(),
                "seed": int(seed),
                "costsHash": canonical_sha256([costs[key] for key in sorted(costs)]),
                "survivalBufferTargets": [
                    {
                        "seat": seat.name.lower(),
                        "suit": suit.name.lower(),
                        "protectedCards": sorted(cards),
                    }
                    for (seat, suit), cards in sorted(
                        protected_at_decision.items(),
                        key=lambda item: (int(item[0][0]), int(item[0][1])),
                    )
                ],
                "partialTrickBridgeHints": [
                    {"lane": lane, "seat": seat.name.lower(), "card": card.code}
                    for lane, seat, card in bridge_hints
                ],
                "noGoodHashes": list(no_good_hashes),
            }
        )
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = int(seed) & 0x7FFFFFFF
        solver.parameters.search_branching = (
            cp_model.HINT_SEARCH if bridge_hint_variables else cp_model.FIXED_SEARCH
        )
        solver.parameters.max_time_in_seconds = float(maximum_seconds)
        status_code = solver.solve(model)
        status_name = solver.status_name(status_code).lower()
        deck_order = None
        core: tuple[str, ...] = ()
        if status_code == cp_model.OPTIMAL:
            by_position = [-1] * 32
            for card in range(32):
                selected = [
                    position
                    for position in range(32)
                    if solver.boolean_value(positions[card, position])
                ]
                if len(selected) != 1:
                    raise AssertionError("structural solver returned an invalid deck")
                by_position[selected[0]] = card
            deck_order = tuple(by_position)
            result_status = "satisfied"
        elif status_code == cp_model.INFEASIBLE:
            result_status = "structurally_infeasible"
            sufficient = list(
                sound_infeasible_assumption_indexes(solver, assumption_predicates)
            )
            for candidate in tuple(sufficient):
                trial_indexes = [index for index in sufficient if index != candidate]
                if not trial_indexes:
                    continue
                model.clear_assumptions()
                model.add_assumptions(
                    [
                        assumption_predicates[index][1]
                        for index in trial_indexes
                        if index in assumption_predicates
                    ]
                )
                trial = cp_model.CpSolver()
                trial.parameters.num_search_workers = 1
                trial.parameters.random_seed = int(seed) & 0x7FFFFFFF
                trial.parameters.max_time_in_seconds = max(
                    0.001,
                    float(maximum_seconds) - (time.perf_counter() - started),
                )
                if trial.solve(model) == cp_model.INFEASIBLE:
                    sufficient = trial_indexes
            mapped = tuple(
                sorted(
                    {
                        assumption_predicates[index][0]
                        for index in sufficient
                        if index in assumption_predicates
                    }
                )
            )
            core = mapped or lowered_ids
        else:
            result_status = "not_found_within_budget"
        result_core = {
            "schemaVersion": 1,
            "status": result_status,
            "planHash": plan.sha256,
            "modelHash": model_hash,
            "solverStatus": status_name,
            "deckOrder": list(deck_order) if deck_order is not None else None,
            "predicateEvidence": evidence,
            "noGoodHashes": list(no_good_hashes),
            "infeasibleCorePredicateIds": list(core),
            "nodes": int(solver.num_branches),
        }
        elapsed = time.perf_counter() - started
        return ScenarioStructuralResultV1(
            result_status,
            plan.sha256,
            model_hash,
            status_name,
            deck_order,
            tuple(evidence),
            no_good_hashes,
            core,
            int(solver.num_branches),
            elapsed,
            canonical_sha256(result_core),
        )


__all__ = [
    "ScenarioConstructionPlanV1",
    "ScenarioStructuralCompilerV1",
    "ScenarioStructuralResultV1",
]
