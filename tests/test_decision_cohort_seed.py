from __future__ import annotations

import inspect

import pytest

from baloot_platform.challenge.decision_cohort import (
    derive_actor_observer_cohort_seed_v1,
)
from baloot_platform.core.enums import Seat


PUBLIC_HASH = "1" * 64
AUTHORITY_HASH = "2" * 64


def _seed(**overrides: object) -> int:
    values: dict[str, object] = {
        "public_prefix_hash": PUBLIC_HASH,
        "decision_kind": "play",
        "acting_seat": "north",
        "cohort_authority_hash": AUTHORITY_HASH,
        "cohort_index": 0,
        "governed_salt": "primary-decision-policy:v1",
    }
    values.update(overrides)
    return derive_actor_observer_cohort_seed_v1(**values)  # type: ignore[arg-type]


def test_seed_is_versioned_deterministic_unsigned_uint64() -> None:
    first = _seed()

    assert first == _seed()
    assert first == 17_385_512_570_150_350_586
    assert 0 <= first < 1 << 64


def test_signature_excludes_generation_private_and_result_identity() -> None:
    signature = inspect.signature(derive_actor_observer_cohort_seed_v1)

    assert tuple(signature.parameters) == (
        "public_prefix_hash",
        "decision_kind",
        "acting_seat",
        "cohort_authority_hash",
        "cohort_index",
        "governed_salt",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )

    excluded = (
        "root_seed",
        "attempt_index",
        "game_id",
        "trace_hash",
        "decisionId",
        "selectedAction",
        "hidden_deal",
    )
    for field in excluded:
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            _seed(**{field: "must-not-enter-the-authority"})


def test_same_actor_visible_state_ignores_hidden_deal_and_root_seed_context() -> None:
    contexts = (
        {"rootSeed": 7, "hiddenDeal": "deal-a"},
        {"rootSeed": 991, "hiddenDeal": "deal-b"},
    )

    seeds = [
        _seed()
        for context in contexts
        if context["rootSeed"] != 0 and context["hiddenDeal"]
    ]

    assert seeds == [seeds[0], seeds[0]]


@pytest.mark.parametrize(
    "override",
    (
        {"public_prefix_hash": "3" * 64},
        {"decision_kind": "bid"},
        {"acting_seat": "east"},
        {"cohort_authority_hash": "4" * 64},
        {"cohort_index": 1},
        {"governed_salt": "primary-decision-policy:v2"},
    ),
)
def test_each_authorized_input_changes_the_seed(override: dict[str, object]) -> None:
    assert _seed(**override) != _seed()


@pytest.mark.parametrize(
    ("override", "error", "message"),
    (
        ({"public_prefix_hash": "A" * 64}, ValueError, "canonical lowercase"),
        ({"public_prefix_hash": "a" * 63}, ValueError, "canonical lowercase"),
        ({"cohort_authority_hash": "z" * 64}, ValueError, "canonical lowercase"),
        ({"decision_kind": "PLAY"}, ValueError, "decision_kind"),
        ({"decision_kind": "discard"}, ValueError, "decision_kind"),
        ({"acting_seat": "NORTH"}, ValueError, "acting_seat"),
        ({"acting_seat": 0}, TypeError, "acting_seat"),
        ({"cohort_index": True}, TypeError, "cohort_index"),
        ({"cohort_index": -1}, ValueError, "cohort_index"),
        ({"cohort_index": 1 << 31}, ValueError, "cohort_index"),
        ({"governed_salt": ""}, ValueError, "governed_salt"),
        ({"governed_salt": " leading-space"}, ValueError, "governed_salt"),
        ({"governed_salt": "line\nbreak"}, ValueError, "governed_salt"),
        ({"governed_salt": "x" * 257}, ValueError, "governed_salt"),
    ),
)
def test_invalid_inputs_fail_closed(
    override: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _seed(**override)


def test_seat_enum_has_the_same_canonical_identity_as_its_name() -> None:
    assert _seed(acting_seat=Seat.NORTH) == _seed(acting_seat="north")
