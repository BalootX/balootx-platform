"""Pure stage planning for late, trick-boundary Research V4 searches."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

from .scenario_contracts import (
    ResearchScenarioV1,
    ScenarioExpressionV1,
    canonical_sha256,
)
from .scenario_registry_v2 import (
    ScenarioPredicateDefinitionV2,
    scenario_predicate_definitions_v2,
)


_TAIL_STAGES = frozenset({"playPrefix", "state", "proof", "tactics"})
_SUPPORTED_TAIL_LENGTHS = frozenset({4, 8, 12})


@dataclass(frozen=True, slots=True)
class TailFirstStagePlanV1:
    planner: str
    prefix_length: int | None
    tail_offset: int | None
    tail_length: int | None
    hard_tail_predicate_ids: tuple[str, ...]
    deferred_predicate_ids: tuple[str, ...]
    boundary_combinations: tuple[tuple[str, str, str, str | None], ...]
    fallback_reason: str | None
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "planner": self.planner,
            "prefixLength": self.prefix_length,
            "tailOffset": self.tail_offset,
            "tailLength": self.tail_length,
            "hardTailPredicateIds": list(self.hard_tail_predicate_ids),
            "deferredPredicateIds": list(self.deferred_predicate_ids),
            "boundaryCombinations": [
                {
                    "dealer": dealer,
                    "contract": contract,
                    "buyer": buyer,
                    "trump": trump,
                }
                for dealer, contract, buyer, trump in self.boundary_combinations
            ],
            "fallbackReason": self.fallback_reason,
            "sha256": self.sha256,
        }


class TailFirstStagePlannerV1:
    """Choose tail-first only when the expression can be split without drift."""

    def __init__(
        self,
        definitions: Sequence[ScenarioPredicateDefinitionV2] | None = None,
        *,
        default_dealer: str = "east",
        native_cp_sat: bool = False,
    ) -> None:
        rows = (
            tuple(definitions)
            if definitions is not None
            else scenario_predicate_definitions_v2()
        )
        self._definitions = {row.kind: row for row in rows}
        if len(self._definitions) != len(rows):
            raise ValueError("tail-first predicate catalog contains duplicate kinds")
        dealer_domain = self._argument_domain("auction.dealer", "seat")
        if dealer_domain is not None and default_dealer not in dealer_domain:
            raise ValueError("tail-first default dealer is outside the catalog domain")
        self._default_dealer = default_dealer
        self._supported_tail_lengths = (
            _SUPPORTED_TAIL_LENGTHS | {16} if native_cp_sat else _SUPPORTED_TAIL_LENGTHS
        )

    def plan(self, scenario: ResearchScenarioV1) -> TailFirstStagePlanV1:
        hard_tail_ids: set[str] = set()
        deferred_ids: set[str] = set()
        unresolved_kinds: set[str] = set()
        mixed_logical = False

        def stage_groups(node: ScenarioExpressionV1) -> frozenset[str]:
            nonlocal mixed_logical
            if node.predicate is not None:
                predicate = node.predicate
                if predicate.hardness != "hard":
                    return frozenset()
                definition = self._definitions.get(predicate.kind)
                if definition is None:
                    unresolved_kinds.add(predicate.kind)
                    return frozenset()
                group = "tail" if definition.stage in _TAIL_STAGES else "deferred"
                (hard_tail_ids if group == "tail" else deferred_ids).add(
                    predicate.predicate_id
                )
                return frozenset({group})

            groups = frozenset(
                group for child in node.children for group in stage_groups(child)
            )
            if node.op in {"any", "not", "count"} and len(groups) > 1:
                mixed_logical = True
            return groups

        stage_groups(scenario.constraints)
        stage_groups(scenario.outcome)

        combinations = self._boundary_combinations(scenario)
        reason: str | None = None
        prefix_length: int | None = None
        tail_length: int | None = None

        if scenario.origin.kind != "search":
            reason = "origin_not_search"
        elif scenario.decision_point["kind"] != "play":
            reason = "decision_kind_not_play"
        else:
            selector = scenario.decision_point["selector"]
            trick_index = selector.get("trickIndex")
            remaining_cards = selector.get("remainingCards")
            position = selector.get("position")
            if (
                type(trick_index) is not int
                or type(remaining_cards) is not int
                or type(position) is not int
            ):
                reason = "decision_boundary_selector_incomplete"
            elif selector.get("actingSeat") is None:
                reason = "decision_actor_selector_incomplete"
            elif not 0 <= position <= 3:
                reason = "decision_position_invalid"
            elif remaining_cards != 8 - trick_index:
                reason = "decision_remaining_cards_mismatch"
            else:
                prefix_length = trick_index * 4 + position
                tail_length = remaining_cards * 4
                if tail_length not in self._supported_tail_lengths:
                    reason = "tail_length_not_supported"

        if reason is None and not hard_tail_ids:
            reason = "tail_predicate_missing"
        if reason is None and unresolved_kinds:
            reason = "predicate_catalog_stage_unresolved"
        if reason is None and mixed_logical:
            reason = "mixed_logical_stage_projection"
        if reason is None and combinations is None:
            reason = "finite_boundary_domain_unresolved"
        elif reason is None and not combinations:
            reason = "finite_boundary_domain_empty"
        elif reason is None and any(contract == "hokum" for _, contract, _, _ in combinations):
            # ponytail: use full play for Hokum until the tail and boundary
            # bind the same escalation lock mode.
            reason = "hokum_lock_mode_requires_full_play"

        planner = "tail_first" if reason is None else "generic"
        if planner == "generic":
            prefix_length = None
            tail_length = None
        core = {
            "schemaVersion": 1,
            "planner": planner,
            "prefixLength": prefix_length,
            "tailOffset": (
                prefix_length
                - int(scenario.decision_point["selector"].get("position", 0))
                if prefix_length is not None
                else None
            ),
            "tailLength": tail_length,
            "hardTailPredicateIds": sorted(hard_tail_ids),
            "deferredPredicateIds": sorted(deferred_ids),
            "boundaryCombinations": [
                {
                    "dealer": dealer,
                    "contract": contract,
                    "buyer": buyer,
                    "trump": trump,
                }
                for dealer, contract, buyer, trump in combinations or ()
            ],
            "fallbackReason": reason,
        }
        return TailFirstStagePlanV1(
            planner=planner,
            prefix_length=prefix_length,
            tail_offset=core["tailOffset"],
            tail_length=tail_length,
            hard_tail_predicate_ids=tuple(core["hardTailPredicateIds"]),
            deferred_predicate_ids=tuple(core["deferredPredicateIds"]),
            boundary_combinations=tuple(combinations or ()),
            fallback_reason=reason,
            sha256=canonical_sha256(core),
        )

    def _boundary_combinations(
        self,
        scenario: ResearchScenarioV1,
    ) -> tuple[tuple[str, str, str, str | None], ...] | None:
        dealers = self._argument_domain("auction.dealer", "seat")
        contracts = self._argument_domain("auction.contract", "contract")
        buyers = self._argument_domain("auction.buyer", "seat")
        trumps = self._argument_domain("auction.trump", "suit")
        if dealers is None or contracts is None or buyers is None or trumps is None:
            return None
        hard_kinds = {
            predicate.kind
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
        }
        if "auction.dealer" not in hard_kinds:
            dealers = (self._default_dealer,)

        def possible(
            node: ScenarioExpressionV1,
            dealer: str,
            contract: str,
            buyer: str,
            trump: str | None,
        ) -> frozenset[bool] | None:
            if node.predicate is not None:
                predicate = node.predicate
                if predicate.hardness != "hard":
                    return None
                if predicate.kind == "auction.contract":
                    return frozenset(
                        {str(predicate.arguments.get("contract")) == contract}
                    )
                if predicate.kind == "auction.buyer":
                    return frozenset({str(predicate.arguments.get("seat")) == buyer})
                if predicate.kind == "auction.dealer":
                    return frozenset({str(predicate.arguments.get("seat")) == dealer})
                if predicate.kind == "auction.trump":
                    return frozenset({str(predicate.arguments.get("suit")) == trump})
                return frozenset({False, True})

            children = tuple(
                value
                for child in node.children
                if (value := possible(child, dealer, contract, buyer, trump))
                is not None
            )
            if not children:
                return None
            if len(children) == 1 and node.op in {"all", "any"}:
                return children[0]
            if node.op == "all":
                return frozenset(
                    value
                    for value, admitted in (
                        (True, all(True in child for child in children)),
                        (False, any(False in child for child in children)),
                    )
                    if admitted
                )
            if node.op == "any":
                return frozenset(
                    value
                    for value, admitted in (
                        (True, any(True in child for child in children)),
                        (False, all(False in child for child in children)),
                    )
                    if admitted
                )
            if node.op == "not":
                return frozenset(not value for value in children[0])
            if node.op == "count":
                if node.minimum is None or node.maximum is None:
                    raise ValueError("scenario count expression requires bounds")
                counts = {0}
                for child in children:
                    counts = {
                        count + int(value) for count in counts for value in child
                    }
                return frozenset(
                    node.minimum <= count <= node.maximum for count in counts
                )
            raise ValueError(f"unsupported scenario expression operator: {node.op}")

        universe = (
            (dealer, contract, buyer, trump)
            for dealer, contract, buyer in product(dealers, contracts, buyers)
            for trump in (trumps if contract == "hokum" else (None,))
        )
        result: list[tuple[str, str, str, str | None]] = []
        for dealer, contract, buyer, trump in universe:
            values = tuple(
                value
                for expression in (scenario.constraints, scenario.outcome)
                if (
                    value := possible(expression, dealer, contract, buyer, trump)
                )
                is not None
            )
            if all(True in value for value in values):
                result.append((dealer, contract, buyer, trump))
        return tuple(
            sorted(result, key=lambda row: (*row[:3], row[3] or ""))
        )

    def _argument_domain(self, kind: str, argument: str) -> tuple[str, ...] | None:
        definition = self._definitions.get(kind)
        if definition is None:
            return None
        properties = definition.argument_schema.get("properties")
        if not isinstance(properties, Mapping):
            return None
        schema = properties.get(argument)
        if not isinstance(schema, Mapping):
            return None
        values = schema.get("enum")
        if (
            type(values) is not list
            or not values
            or any(type(value) is not str or not value for value in values)
        ):
            return None
        return tuple(sorted(set(values)))


__all__ = ["TailFirstStagePlanV1", "TailFirstStagePlannerV1"]
