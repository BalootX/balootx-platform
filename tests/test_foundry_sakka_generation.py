from __future__ import annotations

import pytest

from baloot_platform.challenge.sakka_generation import (
    NaturalSakkaGenerationProfileV1,
    NaturalSakkaGeneratorV1,
)
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.core.enums import Seat


def _profile(**overrides):
    value = {
        "schemaVersion": 1,
        "id": "natural_sakka_generation_test_v1",
        "revision": "1",
        "target": 152,
        "maximumRounds": 32,
        "seedPolicy": "sha256_round_ordinal_v1",
    }
    value.update(overrides)
    return NaturalSakkaGenerationProfileV1.build(value)


def test_natural_sakka_generator_is_complete_deterministic_and_equal_seat(root):
    simulator = NaturalLaffaSimulatorV1.from_root(
        root,
        profile_id="foundry_throughput",
        maximum_deal_attempts=8,
    )
    generator = NaturalSakkaGeneratorV1(simulator, _profile())
    first = generator.generate(20260803, initial_dealer=Seat.EAST)
    second = generator.generate(20260803, initial_dealer="east")

    assert first.sha256 == second.sha256
    assert first.sakka.sha256 == second.sakka.sha256
    assert first.sakka.starting_score == (0, 0)
    assert max(first.sakka.final_score) >= 152
    assert len(first.rounds) == len(first.sakka.rounds)
    assert len(set(first.round_seeds)) == len(first.round_seeds)
    assert all(item.publishable_laffa for item in first.rounds)
    assert all(
        len({profile for _, profile in item.final_trace.agent_profiles}) == 1
        for item in first.rounds
    )
    assert [
        item.final_trace.sha256 for item in first.rounds
    ] == [item["traceHash"] for item in first.sakka.rounds]
    assert any(item["allPassCount"] > 0 for item in first.sakka.rounds)
    assert [
        [trace.sha256 for trace in item.attempt_traces] for item in first.rounds
    ] == [item["auctionAttemptTraceHashes"] for item in first.sakka.rounds]


def test_natural_sakka_generator_rejects_unfinished_bound_and_unknown_policy(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root, maximum_deal_attempts=8)
    generator = NaturalSakkaGeneratorV1(
        simulator,
        _profile(maximumRounds=1),
    )
    with pytest.raises(RuntimeError, match="governed round bound"):
        generator.generate(1)

    with pytest.raises(ValueError, match="unknown natural Sakka generation keys"):
        NaturalSakkaGenerationProfileV1.build({
            **_profile().to_dict(),
            "unexpected": True,
        })
