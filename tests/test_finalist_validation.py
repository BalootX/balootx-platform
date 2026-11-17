from __future__ import annotations

from dataclasses import replace

import pytest

from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.challenge.finalist_validation import (
    FinalistLaffaReplayV1,
    FinalistValidationProfileV1,
)
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.config import build_bundle
from baloot_platform.plugins.base import FinalistReplayPlugin


def _simulator(root, profile_id: str) -> NaturalLaffaSimulatorV1:
    bundle = build_bundle(root)
    registry = AgentProfileRegistry.from_bundle(root)
    return NaturalLaffaSimulatorV1(
        bundle.rules,
        bundle.convention,
        bundle_hash=bundle.sha256,
        configuration_snapshot=bundle.configuration_snapshot(),
        agent_profile=registry.get(profile_id),
        generation_tier="throughput_reference",
    )


def test_finalist_validation_profile_is_closed_and_governed(root):
    profile = FinalistValidationProfileV1.build(
        build_bundle(root).component("challenge")["finalist_validation"]
    )
    assert profile.agent_profile_id == "finalist_candidate"
    assert profile.maximum_traces_per_job == 4
    assert profile.maximum_confirmation_decisions_per_trace == 64
    assert profile.maximum_trace_seconds == 300
    assert profile.allow_local_preview
    assert profile.enable_exact_tactical_viability_precheck
    assert profile.require_qualified_agent_release_for_publication
    assert len(profile.sha256) == 64

    invalid = profile.to_dict()
    invalid.pop("sha256")
    invalid["undeclared"] = True
    with pytest.raises(ValueError, match="unknown finalist validation"):
        FinalistValidationProfileV1.build(invalid)


def test_finalist_replay_preserves_deal_and_blocks_unqualified_publication(root):
    parent = _simulator(root, "foundry_throughput").simulate(17).final_trace
    bundle = build_bundle(root)
    registry = AgentProfileRegistry.from_bundle(root)
    validator_profile = replace(
        registry.get("foundry_throughput"), id="test_finalist_validator"
    )
    validator = NaturalLaffaSimulatorV1(
        bundle.rules,
        bundle.convention,
        bundle_hash=bundle.sha256,
        configuration_snapshot=bundle.configuration_snapshot(),
        agent_profile=validator_profile,
        generation_tier="throughput_reference",
    )
    plugin = FinalistLaffaReplayV1(validator)
    assert isinstance(plugin, FinalistReplayPlugin)

    descendant, evidence = plugin.replay(parent)
    repeated, repeated_evidence = plugin.replay(parent)

    assert descendant.sha256 == repeated.sha256
    assert evidence.sha256 == repeated_evidence.sha256
    assert descendant.sha256 != parent.sha256
    assert descendant.seed_provenance["deckCommitment"] == parent.seed_provenance[
        "deckCommitment"
    ]
    generation = descendant.trusted_final_state["generation"]
    assert generation["kind"] == "finalist_validation_descendant"
    assert generation["parentTraceHash"] == parent.sha256
    assert set(dict(descendant.agent_profiles).values()) == {validator.agent_release}
    assert evidence.replay_complete
    assert evidence.local_preview_allowed
    assert not evidence.publication_ready
    assert evidence.validator_status == "configured_candidate"
    assert "professional_agent_qualification_missing" in evidence.blockers

    with pytest.raises(ValueError, match="agent release"):
        validator.resimulate_exact_deal(
            parent,
            match_score=(0, 0),
            round_number=1,
        )
