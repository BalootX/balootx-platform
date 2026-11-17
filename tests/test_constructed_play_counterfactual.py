from __future__ import annotations

from pathlib import Path

import pytest

from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.core.enums import Seat
from baloot_platform.research.construction_pre_screen import (
    ConstructionPreScreenProfileV1,
    ConstructionPreScreenV1,
)
from baloot_platform.service.container import build_container


def _scripted_play_provenance():
    return {
        "schemaVersion": 1,
        "pluginId": "counterfactual_fixture",
        "modelHash": "3" * 64,
        "planHash": "4" * 64,
    }


def _strategic_play_provenance():
    return {
        "schemaVersion": 1,
        "plugin": {
            "id": "actor_safe_cp_sat_late_game_v1",
            "kind": "research_play_selector",
            "version": "1.0.0",
            "deterministic": True,
        },
        "profile": {
            "schemaVersion": 1,
            "sha256": "5" * 64,
        },
    }


@pytest.mark.parametrize(
    ("key", "provenance"),
    (
        ("scriptedPlayCompiler", _scripted_play_provenance()),
        ("strategicPlaySelector", _strategic_play_provenance()),
    ),
)
def test_constructed_play_counterfactual_accepts_governed_play_sources(
    key,
    provenance,
):
    assert NaturalLaffaSimulatorV1._constructed_play_source_key(
        {key: provenance}
    ) == key


@pytest.mark.parametrize(
    "generation",
    (
        {"customPlaySource": {"schemaVersion": 1}},
        {
            "scriptedPlayCompiler": _scripted_play_provenance(),
            "strategicPlaySelector": _strategic_play_provenance(),
        },
        {"strategicPlaySelector": {"schemaVersion": 1}},
    ),
)
def test_constructed_play_counterfactual_rejects_unrecognized_or_ambiguous_sources(
    generation,
):
    with pytest.raises(ValueError, match="play source|play-source|strategic-play"):
        NaturalLaffaSimulatorV1._constructed_play_source_key(generation)


def _scripted_bids(trace):
    return tuple(
        {
            "actor": event["seat"],
            "roundIndex": event["round"],
            "action": event["action"],
            "trumpSuit": event.get("trumpSuit"),
        }
        for event in trace.trusted_final_state["events"]
        if event["type"] == "bid_made"
    )


def _scripted_plays(trace):
    return tuple(
        {
            "actor": record.acting_seat.name.lower(),
            "card": dict(record.chosen_action),
            "ekka": None,
        }
        for record in trace.decision_ledger.records
        if record.decision_kind == "play"
    )


def test_constructed_play_counterfactual_freezes_public_prefix(tmp_path):
    container = build_container(
        root=Path("src/baloot_platform/resources"),
        writable_root=tmp_path / "runtime",
    )
    try:
        simulator = container.research.simulator
        provenance = {
            "schemaVersion": 1,
            "pluginId": "counterfactual_fixture",
            "profileHash": "1" * 64,
            "candidateHash": "2" * 64,
        }
        baseline = simulator.simulate_preordered_deck(
            1,
            tuple(range(32)),
            deal_generator_provenance=provenance,
            dealer=Seat.EAST,
        ).final_trace
        assert any(
            record.decision_kind == "play"
            for record in baseline.decision_ledger.records
        )
        parent = simulator.simulate_preordered_deck(
            1,
            tuple(range(32)),
            deal_generator_provenance=provenance,
            dealer=Seat.EAST,
            scripted_bids=_scripted_bids(baseline),
            scripted_escalations=(),
            scripted_plays=_scripted_plays(baseline),
            scripted_play_provenance=_scripted_play_provenance(),
        ).final_trace

        child = simulator.resimulate_constructed_play_as_counterfactual_descendant(
            parent
        )

        assert child.seed_provenance["deckCommitment"] == parent.seed_provenance[
            "deckCommitment"
        ]
        generation = child.trusted_final_state["generation"]
        assert generation["kind"] == "constructed_play_counterfactual_descendant"
        assert generation["counterfactual"] == {
            "schemaVersion": 1,
            "scope": "play_policy_only",
            "parentPrePlayPrefixHash": (
                simulator._constructed_pre_play_prefix_hash(parent)
            ),
        }
        assert simulator._constructed_pre_play_prefix_hash(child) == (
            simulator._constructed_pre_play_prefix_hash(parent)
        )
        parent_bids = [
            (
                item.ordinal,
                item.acting_seat,
                item.complete_state_hash,
                item.public_prefix_hash,
                dict(item.chosen_action),
            )
            for item in parent.decision_ledger.records
            if item.decision_kind == "bid"
        ]
        child_bids = [
            (
                item.ordinal,
                item.acting_seat,
                item.complete_state_hash,
                item.public_prefix_hash,
                dict(item.chosen_action),
            )
            for item in child.decision_ledger.records
            if item.decision_kind == "bid"
        ]
        assert child_bids == parent_bids

        profile = ConstructionPreScreenProfileV1.build({
            "schemaVersion": 1,
            "id": "fixture_pre_screen",
            "revision": "1",
            "enabled": True,
            "maximumPlayDecisions": 4,
            "maximumBidRegret": 5,
            "maximumPlayRegret": 8,
            "maximumProbabilityDrop": 0.25,
            "minimumConfidence": 0.2,
            "rejectCatastrophicCrossing": True,
        })
        evidence = ConstructionPreScreenV1(profile).evaluate(
            parent,
            container.research.analyzers["screening"],
        )
        payload = evidence.to_dict()
        assert payload["profileHash"] == profile.sha256
        assert payload["traceHash"] == parent.sha256
        assert payload["proofBoundary"].startswith("cheap rejection evidence only")
        assert len(payload["analyzedDecisions"]) <= 12
    finally:
        container.shutdown()
