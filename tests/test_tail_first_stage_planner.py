from __future__ import annotations

import pytest

from baloot_platform.research.scenario_contracts import (
    ResearchScenarioV1,
    canonical_sha256,
)
from baloot_platform.research.tail_first_stage_planner import (
    TailFirstStagePlannerV1,
)


def _leaf(
    predicate_id: str,
    kind: str,
    arguments: dict[str, object],
    hardness: str = "hard",
) -> dict[str, object]:
    return {
        "predicateId": predicate_id,
        "kind": kind,
        "version": 1,
        "hardness": hardness,
        "arguments": arguments,
    }


def _scenario(
    constraints: dict[str, object],
    *,
    origin: str = "search",
    decision_kind: str = "play",
    position: int | None = 0,
    acting_seat: str | None = "south",
) -> ResearchScenarioV1:
    selector = (
        {"trickIndex": 5, "remainingCards": 3,
         **({"actingSeat": acting_seat} if acting_seat is not None else {}),
         **({"position": position} if position is not None else {})}
        if decision_kind == "play"
        else {"actionIndex": 0}
    )
    origin_value: dict[str, object] = {"kind": origin}
    if origin == "exact_trace_prefix":
        origin_value["exactTracePrefix"] = {
            "schemaVersion": 1,
            "deckOrder": list(range(32)),
            "dealer": "north",
            "roundNumber": 1,
            "matchScore": [0, 0],
            "auctionActions": [],
            "escalationActions": [],
            "projectDeclarations": [],
            "projectAssertions": [],
            "playActions": [],
            "stateAssertions": {},
        }
    return ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": origin_value,
            "constraints": constraints,
            "decisionPoint": {"kind": decision_kind, "selector": selector},
            "outcome": _leaf(
                "PRED-ROUND-WIN",
                "outcome.team_round",
                {"team": "north_south", "status": "won"},
            ),
        }
    )


def _eligible_children() -> list[dict[str, object]]:
    return [
        _leaf("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
        _leaf("PRED-BUYER", "auction.buyer", {"seat": "west"}),
        _leaf(
            "PRED-TRICK",
            "play.trick_index",
            {"minimum": 5, "maximum": 5},
        ),
        _leaf(
            "PRED-REPORT",
            "state.legal_action_count",
            {"minimum": 1, "maximum": 32},
            "report",
        ),
    ]


def test_eligible_twelve_card_tail_plan_is_canonical() -> None:
    scenario = _scenario({"op": "all", "children": _eligible_children()})

    plan = TailFirstStagePlannerV1().plan(scenario)

    assert plan.planner == "tail_first"
    assert plan.prefix_length == 20
    assert plan.tail_offset == 20
    assert plan.tail_length == 12
    assert plan.hard_tail_predicate_ids == ("PRED-TRICK",)
    assert plan.deferred_predicate_ids == (
        "PRED-BUYER",
        "PRED-CONTRACT",
        "PRED-ROUND-WIN",
    )
    assert plan.boundary_combinations == (("east", "sun", "west", None),)
    assert plan.fallback_reason is None
    assert plan.to_dict()["sha256"] == plan.sha256
    assert plan.sha256 == canonical_sha256(
        {key: value for key, value in plan.to_dict().items() if key != "sha256"}
    )


def test_unconstrained_play_position_uses_full_play_fallback() -> None:
    scenario = _scenario(
        {"op": "all", "children": _eligible_children()}, position=None,
    )
    plan = TailFirstStagePlannerV1().plan(scenario)
    assert plan.planner != "tail_first"
    assert plan.fallback_reason == "decision_boundary_selector_incomplete"


def test_native_four_trick_tail_keeps_historical_planner_unchanged() -> None:
    value = _scenario({"op": "all", "children": _eligible_children()}).to_dict()
    value.pop("sha256")
    value["decisionPoint"]["selector"].update(trickIndex=4, remainingCards=4)
    scenario = ResearchScenarioV1.build(value)
    historical = TailFirstStagePlannerV1().plan(scenario)
    assert historical.planner == "generic"
    assert historical.fallback_reason == "tail_length_not_supported"
    plan = TailFirstStagePlannerV1(native_cp_sat=True).plan(scenario)
    assert (plan.planner, plan.prefix_length, plan.tail_offset, plan.tail_length) == (
        "tail_first", 16, 16, 16,
    )
    assert plan.sha256 == canonical_sha256(
        {key: item for key, item in plan.to_dict().items() if key != "sha256"}
    )


def test_unconstrained_play_actor_uses_full_play_fallback() -> None:
    scenario = _scenario(
        {"op": "all", "children": _eligible_children()}, acting_seat=None,
    )
    plan = TailFirstStagePlannerV1().plan(scenario)
    assert plan.planner == "generic"
    assert plan.fallback_reason == "decision_actor_selector_incomplete"


def test_boundary_combinations_use_default_dealer_and_contract_trump_rules() -> None:
    plan = TailFirstStagePlannerV1(default_dealer="east").plan(
        _scenario(
            {
                "op": "all",
                "children": [
                    _leaf(
                        "PRED-TRICK-ONLY",
                        "play.trick_index",
                        {"minimum": 5, "maximum": 5},
                    )
                ],
            }
        )
    )

    assert {row[0] for row in plan.boundary_combinations} == {"east"}
    assert len(plan.boundary_combinations) == 24
    assert all(
        trump is None
        for _dealer, contract, _buyer, trump in plan.boundary_combinations
        if contract in {"sun", "ashkal"}
    )
    assert {
        trump
        for _dealer, contract, _buyer, trump in plan.boundary_combinations
        if contract == "hokum"
    } == {"spades", "hearts", "diamonds", "clubs"}


def test_hokum_boundary_falls_back_until_lock_mode_is_bound() -> None:
    children = [
        _leaf("PRED-DEALER", "auction.dealer", {"seat": "north"}),
        _leaf("PRED-HOKUM", "auction.contract", {"contract": "hokum"}),
        _leaf("PRED-HOKUM-BUYER", "auction.buyer", {"seat": "south"}),
        _leaf("PRED-TRUMP", "auction.trump", {"suit": "hearts"}),
        _leaf(
            "PRED-HOKUM-TRICK",
            "play.trick_index",
            {"minimum": 5, "maximum": 5},
        ),
    ]

    plan = TailFirstStagePlannerV1().plan(
        _scenario({"op": "all", "children": children})
    )

    assert plan.planner == "generic"
    assert plan.fallback_reason == "hokum_lock_mode_requires_full_play"
    assert plan.boundary_combinations == (
        ("north", "hokum", "south", "hearts"),
    )


def test_broad_late_selector_without_a_hard_tail_predicate_uses_generic_search() -> None:
    scenario = _scenario(
        _leaf(
            "PRED-DEAL-ONLY",
            "deal.card_owner",
            {"card": 0, "seat": "north", "handStage": "initial_five"},
        )
    )

    plan = TailFirstStagePlannerV1().plan(scenario)

    assert plan.planner == "generic"
    assert plan.fallback_reason == "tail_predicate_missing"


@pytest.mark.parametrize(
    "expression",
    [
        {
            "op": "any",
            "children": [
                _leaf("PRED-ANY-TAIL", "play.trick_index", {"minimum": 5, "maximum": 5}),
                _leaf("PRED-ANY-DEFER", "auction.contract", {"contract": "sun"}),
            ],
        },
        {
            "op": "not",
            "children": [
                {
                    "op": "all",
                    "children": [
                        _leaf("PRED-NOT-TAIL", "play.trick_index", {"minimum": 5, "maximum": 5}),
                        _leaf("PRED-NOT-DEFER", "auction.contract", {"contract": "sun"}),
                    ],
                }
            ],
        },
        {
            "op": "count",
            "minimum": 1,
            "maximum": 1,
            "children": [
                _leaf("PRED-COUNT-TAIL", "play.trick_index", {"minimum": 5, "maximum": 5}),
                _leaf("PRED-COUNT-DEFER", "auction.contract", {"contract": "sun"}),
            ],
        },
    ],
)
def test_cross_stage_logical_nodes_fall_back_safely(
    expression: dict[str, object],
) -> None:
    plan = TailFirstStagePlannerV1().plan(_scenario(expression))

    assert plan.planner == "generic"
    assert plan.fallback_reason == "mixed_logical_stage_projection"
    assert plan.prefix_length is None
    assert plan.tail_offset is None
    assert plan.tail_length is None


def test_reordering_conjunctive_leaves_preserves_plan_hash() -> None:
    children = _eligible_children()
    forward = TailFirstStagePlannerV1().plan(
        _scenario({"op": "all", "children": children})
    )
    reverse = TailFirstStagePlannerV1().plan(
        _scenario({"op": "all", "children": list(reversed(children))})
    )

    assert forward.to_dict() == reverse.to_dict()


@pytest.mark.parametrize(
    ("origin", "decision_kind", "reason"),
    [
        ("exact_trace_prefix", "play", "origin_not_search"),
        ("search", "bid", "decision_kind_not_play"),
    ],
)
def test_exact_origin_and_nonplay_decisions_fall_back(
    origin: str,
    decision_kind: str,
    reason: str,
) -> None:
    plan = TailFirstStagePlannerV1().plan(
        _scenario(
            {"op": "all", "children": _eligible_children()},
            origin=origin,
            decision_kind=decision_kind,
        )
    )

    assert plan.planner == "generic"
    assert plan.fallback_reason == reason
