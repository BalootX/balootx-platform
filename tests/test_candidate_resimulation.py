from __future__ import annotations

from itertools import count
from pathlib import Path

import pytest

from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.analysis.search import SearchTimeBudgetExceeded
from baloot_platform.challenge.candidate_resimulation import (
    CandidateResimulationProfileV1,
    CandidateResimulationV1,
)
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.config import build_bundle, platform_root


def _profile() -> CandidateResimulationProfileV1:
    return CandidateResimulationProfileV1.build(
        {
            "schemaVersion": 1,
            "id": "selective_same_deal_resimulation_v1",
            "revision": "test",
            "enabled": True,
            "agentProfileId": "foundry_balanced",
            "maximumTraceSeconds": 30,
            "requireExactDeckCommitment": True,
            "requireParentTraceLink": True,
        }
    )


def _balanced(root: Path) -> CandidateResimulationV1:
    bundle = build_bundle(root)
    registry = AgentProfileRegistry.from_bundle(root)
    simulator = NaturalLaffaSimulatorV1(
        bundle.rules,
        bundle.convention,
        bundle_hash=bundle.sha256,
        configuration_snapshot=bundle.configuration_snapshot(),
        agent_profile=registry.get("foundry_balanced"),
        generation_tier="balanced_discovery",
        maximum_deal_attempts=8,
        all_pass_dealer_policy=str(
            bundle.component("game")["all_pass"]["dealer_policy"]
        ),
    )
    return CandidateResimulationV1(simulator, _profile())


def test_profile_is_closed_and_requires_exact_lineage() -> None:
    profile = _profile()

    assert profile.to_dict()["sha256"] == profile.sha256
    with pytest.raises(ValueError, match="unknown candidate resimulation"):
        CandidateResimulationProfileV1.build(
            {**profile.to_dict(), "sha256": profile.sha256}
        )
    with pytest.raises(ValueError, match="preserve the exact deal"):
        CandidateResimulationProfileV1.build(
            {
                **{
                    key: value
                    for key, value in profile.to_dict().items()
                    if key != "sha256"
                },
                "requireExactDeckCommitment": False,
            }
        )


def test_retained_candidate_resimulation_is_deterministic_and_same_deal() -> None:
    root = platform_root()
    source = NaturalLaffaSimulatorV1.from_root(
        root,
        profile_id="foundry_throughput",
        maximum_deal_attempts=8,
    ).simulate(202608050476).final_trace
    service = _balanced(root)

    first, first_evidence = service.resimulate(source)
    second, second_evidence = service.resimulate(source)

    assert first.sha256 == second.sha256
    assert first_evidence.sha256 == second_evidence.sha256
    assert (
        first.seed_provenance["deckCommitment"]
        == source.seed_provenance["deckCommitment"]
    )
    assert first.trusted_final_state["generation"]["parentTraceHash"] == source.sha256
    assert (
        first.trusted_final_state["generation"]["kind"]
        == "candidate_resimulation_descendant"
    )
    assert (
        first.trusted_final_state["generation"]["generationTier"]
        == "balanced_discovery"
    )
    assert first_evidence.exact_deck_commitment_preserved is True
    assert first_evidence.parent_trace_link_preserved is True
    assert {value for _, value in first.agent_profiles} == {
        service.simulator.agent_release
    }


def test_retained_candidate_resimulation_honors_cancellation() -> None:
    root = platform_root()
    source = NaturalLaffaSimulatorV1.from_root(
        root,
        profile_id="foundry_throughput",
        maximum_deal_attempts=8,
    ).simulate(202608050476).final_trace

    with pytest.raises(InterruptedError):
        _balanced(root).resimulate(source, cancel_check=lambda: True)


@pytest.mark.real_clock
def test_retained_candidate_resimulation_rejects_wall_clock_exhaustion(
    monkeypatch,
) -> None:
    root = platform_root()
    source = NaturalLaffaSimulatorV1.from_root(
        root,
        profile_id="foundry_throughput",
        maximum_deal_attempts=8,
    ).simulate(202608050476).final_trace
    clock = count(0.0, 2.0)
    monkeypatch.setattr(
        "baloot_platform.analysis.search.monotonic", lambda: next(clock)
    )

    with pytest.raises(SearchTimeBudgetExceeded, match="without canonical output"):
        _balanced(root).resimulate(source)
