from __future__ import annotations

import inspect

import pytest
from ortools.sat.python import cp_model

from baloot_platform.core.enums import Seat
from baloot_platform.research import ortools_adjudication_attempts as subject
from baloot_platform.research.ortools_adjudication_attempts import (
    AdjudicationAttemptKindV1,
    AdjudicationRemedyV1,
    PurchaseAttemptTokenV1,
    add_adjudication_attempt_branch_v1,
)


_AUTHORITY_ID = "gea_baloot_2023_adjudication_research_v1"
_AUTHORITY_HASH = "685a24c8c51e0b099d5dda5fd080cbdb4bd45608c8b05382c5ed97c64fdc0b21"


def _fixed_var(model: cp_model.CpModel, value: int, name: str) -> cp_model.IntVar:
    variable = model.new_int_var(0, 31, name)
    model.add(variable == value)
    return variable


def _build_out_of_turn():
    model = cp_model.CpModel()
    expected = _fixed_var(model, int(Seat.SOUTH), "current_player")
    attempted = _fixed_var(model, int(Seat.WEST), "attempting_player")
    card = _fixed_var(model, 17, "attempted_card")
    return add_adjudication_attempt_branch_v1(
        model,
        kind=AdjudicationAttemptKindV1.OUT_OF_TURN,
        expected_actor=expected,
        attempted_actor=attempted,
        attempted_action=card,
        authority_id=_AUTHORITY_ID,
        authority_hash=_AUTHORITY_HASH,
        required_remedy=AdjudicationRemedyV1.OPPONENT_REGISTER_WITH_PROJECTS,
        assumption_id="PRED-OUT-OF-TURN",
    )


def test_out_of_turn_attempt_is_a_native_separate_branch_and_is_deterministic():
    first = _build_out_of_turn().solve()
    second = _build_out_of_turn().solve()

    assert first.status == "witness"
    assert first.event_present is True
    assert first.expected_actor == Seat.SOUTH
    assert first.attempted_actor == Seat.WEST
    assert first.attempted_action == {"type": "card", "cardId": 17}
    assert first.allowed_remedies == (
        AdjudicationRemedyV1.OPPONENT_REGISTER_WITH_PROJECTS,
        AdjudicationRemedyV1.CONTINUE_PLAY,
    )
    assert first.selected_remedy == (
        AdjudicationRemedyV1.OPPONENT_REGISTER_WITH_PROJECTS
    )
    assert first.model_hash == second.model_hash
    assert first.evidence_hash == second.evidence_hash
    assert first.to_dict()["ordinaryActionHistoryMutated"] is False


def test_named_assumption_proves_same_actor_cannot_be_out_of_turn():
    model = cp_model.CpModel()
    expected = _fixed_var(model, int(Seat.NORTH), "current_player")
    attempted = _fixed_var(model, int(Seat.NORTH), "attempting_player")
    card = _fixed_var(model, 0, "attempted_card")
    branch = add_adjudication_attempt_branch_v1(
        model,
        kind="out_of_turn",
        expected_actor=expected,
        attempted_actor=attempted,
        attempted_action=card,
        authority_id=_AUTHORITY_ID,
        authority_hash=_AUTHORITY_HASH,
        assumption_id="PRED-SAME-ACTOR-CONTRADICTION",
    )

    result = branch.solve()
    assert result.status == "infeasible"
    assert result.assumption_core == ("PRED-SAME-ACTOR-CONTRADICTION",)
    assert result.evidence_hash is None


def test_purchase_before_turn_uses_governed_tokens_and_takweesh_branch():
    model = cp_model.CpModel()
    expected = _fixed_var(model, int(Seat.EAST), "auction_actor")
    attempted = _fixed_var(model, int(Seat.NORTH), "attempting_bidder")
    token = _fixed_var(model, int(PurchaseAttemptTokenV1.SUN), "purchase_token")
    branch = add_adjudication_attempt_branch_v1(
        model,
        kind="purchase_before_turn",
        expected_actor=expected,
        attempted_actor=attempted,
        attempted_action=token,
        authority_id=_AUTHORITY_ID,
        authority_hash=_AUTHORITY_HASH,
        require_event=True,
        required_remedy="opponent_takweesh",
    )

    result = branch.solve()
    assert result.status == "witness"
    assert result.attempted_action == {"type": "auction_token", "token": "sun"}
    assert result.allowed_remedies == (
        AdjudicationRemedyV1.OPPONENT_TAKWEESH,
        AdjudicationRemedyV1.CONTINUE_PLAY,
    )
    assert result.selected_remedy == AdjudicationRemedyV1.OPPONENT_TAKWEESH
    assert result.to_dict()["ordinaryActionHistoryMutated"] is False


def test_no_attempted_actor_mismatch_means_no_event_and_no_remedy():
    model = cp_model.CpModel()
    expected = _fixed_var(model, int(Seat.EAST), "auction_actor")
    attempted = _fixed_var(model, int(Seat.EAST), "attempting_bidder")
    token = _fixed_var(model, int(PurchaseAttemptTokenV1.NO), "purchase_token")
    branch = add_adjudication_attempt_branch_v1(
        model,
        kind="purchase_before_turn",
        expected_actor=expected,
        attempted_actor=attempted,
        attempted_action=token,
        authority_id=_AUTHORITY_ID,
        authority_hash=_AUTHORITY_HASH,
    )

    result = branch.solve()
    assert result.status == "no_event"
    assert result.event_present is False
    assert result.selected_remedy is None
    assert "selectedRemedy" not in result.to_dict()


def test_unbound_remedy_is_never_reported_as_a_selected_ruling():
    model = cp_model.CpModel()
    expected = _fixed_var(model, int(Seat.SOUTH), "current_player")
    attempted = _fixed_var(model, int(Seat.WEST), "attempting_player")
    card = _fixed_var(model, 4, "attempted_card")
    branch = add_adjudication_attempt_branch_v1(
        model,
        kind="out_of_turn",
        expected_actor=expected,
        attempted_actor=attempted,
        attempted_action=card,
        authority_id=_AUTHORITY_ID,
        authority_hash=_AUTHORITY_HASH,
        require_event=True,
    )

    result = branch.solve()
    assert result.status == "witness"
    assert result.selected_remedy is None
    assert result.to_dict()["allowedRemedies"] == [
        "opponent_register_with_projects",
        "continue_play",
    ]
    assert "selectedRemedy" not in result.to_dict()


def test_kind_restricts_actions_and_remedies_before_qualification():
    model = cp_model.CpModel()
    expected = _fixed_var(model, int(Seat.EAST), "auction_actor")
    attempted = _fixed_var(model, int(Seat.NORTH), "attempting_bidder")
    unsupported_token = _fixed_var(model, 31, "unsupported_purchase_token")
    branch = add_adjudication_attempt_branch_v1(
        model,
        kind="purchase_before_turn",
        expected_actor=expected,
        attempted_actor=attempted,
        attempted_action=unsupported_token,
        authority_id=_AUTHORITY_ID,
        authority_hash=_AUTHORITY_HASH,
        require_event=True,
    )
    assert branch.solve().status == "infeasible"

    remedy_model = cp_model.CpModel()
    remedy_card = remedy_model.new_int_var(0, 31, "attempted_card")
    with pytest.raises(ValueError, match="not valid for out_of_turn"):
        add_adjudication_attempt_branch_v1(
            remedy_model,
            kind="out_of_turn",
            expected_actor=remedy_model.new_int_var(0, 3, "expected_actor"),
            attempted_actor=remedy_model.new_int_var(0, 3, "attempted_actor"),
            attempted_action=remedy_card,
            authority_id=_AUTHORITY_ID,
            authority_hash=_AUTHORITY_HASH,
            required_remedy="opponent_takweesh",
        )


def test_branch_contains_no_engine_or_python_truth_constant_shortcut():
    source = inspect.getsource(subject)
    assert "BalootEngine" not in source
    assert "new_constant" not in source
    assert "ordinaryActionHistoryMutated\": False" in source


def test_caller_owned_domain_action_is_reused_and_foreign_variables_fail_closed():
    model = cp_model.CpModel()
    expected = model.new_int_var(0, 3, "current_player")
    attempted = model.new_int_var(0, 3, "attempting_player")
    domain_card = model.new_int_var(0, 31, "existing_play_domain_card")
    branch = add_adjudication_attempt_branch_v1(
        model,
        kind="out_of_turn",
        expected_actor=expected,
        attempted_actor=attempted,
        attempted_action=domain_card,
        authority_id=_AUTHORITY_ID,
        authority_hash=_AUTHORITY_HASH,
    )
    assert branch.attempted_action is domain_card
    assert "existing\nauction or play domain variables" in (subject.__doc__ or "")

    foreign_model = cp_model.CpModel()
    foreign_card = foreign_model.new_int_var(0, 31, "foreign_card")
    with pytest.raises(ValueError, match="must belong to the supplied CP-SAT model"):
        add_adjudication_attempt_branch_v1(
            model,
            kind="out_of_turn",
            expected_actor=expected,
            attempted_actor=attempted,
            attempted_action=foreign_card,
            authority_id=_AUTHORITY_ID,
            authority_hash=_AUTHORITY_HASH,
        )
