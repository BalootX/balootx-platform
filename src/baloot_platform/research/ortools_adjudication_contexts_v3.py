"""Bind governed adjudication attempts to existing auction/play CP variables.

The attempted action is a referee branch beside the ordinary legal history.
This module never appends it to auction actions or play slots.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from ortools.sat.python import cp_model

from baloot_platform.adjudication.research import AuthoredAdjudicationEventV1

from .ortools_adjudication_attempts import PurchaseAttemptTokenV1
from .scenario_adjudication_predicates_v3 import (
    ADJUDICATION_AUTHORITY_ID_V3,
    ADJUDICATION_AUTHORITY_SHA256_V3,
    AdjudicationPredicateModelContextV3,
    FALSE_EKKA_PREDICATE_V3,
    OUT_OF_TURN_PREDICATE_V3,
    PURCHASE_BEFORE_TURN_PREDICATE_V3,
    ScenarioPredicateStageContextV3,
    _matching_event,
)
from .scenario_contracts import ScenarioPredicateV1
from .ortools_scenario_model import OrToolsNativeAuctionFragmentV1
from .scenario_predicate_auction_escalation_v2 import (
    CpSatAuctionPredicateContextV1,
)


def _named_and(
    model: cp_model.CpModel,
    literals: Sequence[cp_model.IntVar],
    name: str,
) -> cp_model.IntVar:
    result = model.new_bool_var(name)
    for literal in literals:
        model.add(result <= literal)
    model.add(result >= sum(literals) - len(literals) + 1)
    return result


def _reified_equal(
    model: cp_model.CpModel,
    left: cp_model.IntVar,
    right: cp_model.IntVar,
    name: str,
) -> cp_model.IntVar:
    result = model.new_bool_var(name)
    model.add(left == right).only_enforce_if(result)
    model.add(left != right).only_enforce_if(result.Not())
    return result


def _adjudication_predicates(
    predicates: Sequence[ScenarioPredicateV1],
    kind: str,
) -> tuple[ScenarioPredicateV1, ...]:
    return tuple(predicate for predicate in predicates if predicate.kind == kind)


def wrap_auction_adjudication_context_v3(
    base_context: CpSatAuctionPredicateContextV1,
    predicates: Sequence[ScenarioPredicateV1],
    *,
    origin: str,
    exact_events: Sequence[AuthoredAdjudicationEventV1] = (),
    action_sequence: int | None = None,
    authority_id: str = ADJUDICATION_AUTHORITY_ID_V3,
    authority_sha256: str = ADJUDICATION_AUTHORITY_SHA256_V3,
) -> CpSatAuctionPredicateContextV1 | ScenarioPredicateStageContextV3:
    """Bind purchase attempts to their live or historical auction boundary."""

    selected = _adjudication_predicates(
        predicates, PURCHASE_BEFORE_TURN_PREDICATE_V3
    )
    if not selected:
        return base_context
    model = base_context.model
    contexts: dict[str, AdjudicationPredicateModelContextV3] = {}
    legal_by_token = tuple(
        base_context.fragment.legal_next_action_literals.get(
            token.name.lower(), model.new_constant(0)
        )
        for token in PurchaseAttemptTokenV1
    )
    for predicate in selected:
        attempted_actor = model.new_int_var(
            0, 3, f"{predicate.predicate_id}_attempted_actor"
        )
        attempted_action = model.new_int_var(
            0,
            len(PurchaseAttemptTokenV1) - 1,
            f"{predicate.predicate_id}_attempted_token",
        )
        available = model.new_bool_var(
            f"{predicate.predicate_id}_attempted_token_available"
        )
        context = AdjudicationPredicateModelContextV3(
            model=model,
            context_id=f"{base_context.context_id}:{predicate.predicate_id}",
            origin=origin,
            expected_actor=base_context.native["currentActor"],
            attempted_actor=attempted_actor,
            attempted_action=attempted_action,
            attempted_action_available=available,
            exact_events=tuple(exact_events),
            action_sequence=action_sequence,
            authority_id=authority_id,
            authority_sha256=authority_sha256,
        )
        fragment = base_context.fragment
        if base_context.view == "completion" and isinstance(fragment, OrToolsNativeAuctionFragmentV1):
            if not fragment.legal_choices_by_step:
                raise ValueError("historical purchase attempts require native auction boundaries")
            legal_by_boundary = []
            for index, choices in enumerate(fragment.legal_choices_by_step):
                legal = model.new_bool_var(f"{predicate.predicate_id}_token_available_{index}")
                model.add_max_equality(legal, [
                    literal for code, literal in choices.items()
                    if code.split(":")[0] == predicate.arguments["purchaseToken"]
                ] or [0])
                legal_by_boundary.append(legal)
            if origin == "exact_trace_prefix":
                context = replace(context, action_sequence=None, exact_events=tuple(
                    event for event in exact_events
                    if event.action_sequence < min(fragment.prefix_length, len(legal_by_boundary))
                ))
                event = _matching_event(predicate, context)
                boundary = event.action_sequence if event is not None else 0
                expected_actor = fragment.states[boundary]["actor"]
                model.add(available == legal_by_boundary[boundary])
            else:
                count = len(legal_by_boundary)
                witnesses = [
                    _named_and(model, (
                        legal,
                        _reified_equal(model, fragment.states[index]["actor"], attempted_actor,
                            f"{predicate.predicate_id}_same_actor_{index}").Not(),
                    ), f"{predicate.predicate_id}_witness_{index}")
                    for index, legal in enumerate(legal_by_boundary)
                ]
                first = model.new_int_var(0, count, f"{predicate.predicate_id}_first_witness")
                # Choose the first witness so negation cannot hide it at another boundary.
                model.add_min_equality(first, [
                    index + (count - index) * (1 - witness)
                    for index, witness in enumerate(witnesses)
                ])
                absent = _reified_equal(model, first, model.new_constant(count),
                    f"{predicate.predicate_id}_witness_absent")
                boundary = model.new_int_var(0, count - 1, f"{predicate.predicate_id}_boundary")
                model.add(boundary == first).only_enforce_if(absent.Not())
                model.add(boundary == 0).only_enforce_if(absent)
                expected_actor = model.new_int_var(0, 3, f"{predicate.predicate_id}_expected_actor")
                model.add_element(boundary, [state["actor"] for state in fragment.states[:count]], expected_actor)
                model.add_element(boundary, legal_by_boundary, available)
            context = replace(context,
                context_id=f"{context.context_id}:historical",
                expected_actor=expected_actor,
                action_sequence=boundary,
            )
        else:
            model.add_element(attempted_action, legal_by_token, available)
        contexts[predicate.predicate_id] = context
    return ScenarioPredicateStageContextV3(base_context, contexts)


def wrap_play_adjudication_context_v3(
    base_context: Any,
    initial_owner_variables: Sequence[cp_model.IntVar],
    predicates: Sequence[ScenarioPredicateV1],
    *,
    origin: str,
    exact_events: Sequence[AuthoredAdjudicationEventV1] = (),
    action_sequence: int | None = None,
    authority_id: str = ADJUDICATION_AUTHORITY_ID_V3,
    authority_sha256: str = ADJUDICATION_AUTHORITY_SHA256_V3,
) -> Any:
    """Attach finite rejected-play attempts beside the legal play boundary."""

    selected = tuple(
        predicate
        for predicate in predicates
        if predicate.kind
        in {OUT_OF_TURN_PREDICATE_V3, FALSE_EKKA_PREDICATE_V3}
    )
    if not selected:
        return base_context
    model = base_context.model
    if len(initial_owner_variables) != 32:
        raise ValueError("out-of-turn context requires ownership of all 32 cards")
    if any(
        not isinstance(variable, cp_model.IntVar)
        or variable.model_proto is not model.proto
        for variable in initial_owner_variables
    ):
        raise ValueError("out-of-turn ownership variables must share the play model")

    contexts: dict[str, AdjudicationPredicateModelContextV3] = {}
    for predicate in selected:
        attempted_actor = model.new_int_var(
            0, 3, f"{predicate.predicate_id}_attempted_actor"
        )
        attempted_action = model.new_int_var(
            0, 31, f"{predicate.predicate_id}_attempted_card"
        )
        if predicate.kind == FALSE_EKKA_PREDICATE_V3:
            card_id = int(predicate.arguments["attemptedCardId"])
            legal_card = base_context.legal_card_literals[card_id]
            legal_ekka = base_context.legal_action_literals[(card_id, True)]
            illegal_ekka = model.new_bool_var(
                f"{predicate.predicate_id}_illegal_ekka"
            )
            model.add(illegal_ekka + legal_ekka == 1)
            available = _named_and(
                model,
                (legal_card, illegal_ekka),
                f"{predicate.predicate_id}_false_ekka_available",
            )
        else:
            owner = model.new_int_var(0, 3, f"{predicate.predicate_id}_card_owner")
            model.add_element(attempted_action, initial_owner_variables, owner)
            owned = _reified_equal(
                model,
                owner,
                attempted_actor,
                f"{predicate.predicate_id}_owned_by_attempting_actor",
            )
            played = []
            for index, slot in enumerate(base_context.play_slots):
                same_card = _reified_equal(
                    model,
                    slot.card,
                    attempted_action,
                    f"{predicate.predicate_id}_slot_{index}_same_card",
                )
                played.append(
                    _named_and(
                        model,
                        (slot.present, same_card),
                        f"{predicate.predicate_id}_slot_{index}_already_played",
                    )
                )
            unplayed = model.new_bool_var(f"{predicate.predicate_id}_card_unplayed")
            model.add(sum(played) == 0).only_enforce_if(unplayed)
            model.add(sum(played) >= 1).only_enforce_if(unplayed.Not())
            available = _named_and(
                model,
                (owned, unplayed),
                f"{predicate.predicate_id}_attempted_card_available",
            )
        contexts[predicate.predicate_id] = AdjudicationPredicateModelContextV3(
            model=model,
            context_id=f"{base_context.context_id}:{predicate.predicate_id}",
            origin=origin,
            expected_actor=base_context.next_actor,
            attempted_actor=attempted_actor,
            attempted_action=attempted_action,
            attempted_action_available=available,
            exact_events=tuple(exact_events),
            action_sequence=action_sequence,
            authority_id=authority_id,
            authority_sha256=authority_sha256,
        )
    return ScenarioPredicateStageContextV3(base_context, contexts)


__all__ = [
    "wrap_auction_adjudication_context_v3",
    "wrap_play_adjudication_context_v3",
]
