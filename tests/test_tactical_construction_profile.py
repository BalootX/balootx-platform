from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from baloot_platform.challenge import oracle as oracle_module
from baloot_platform.challenge.oracle import ActualDealOracleVerifierV1
from baloot_platform.challenge.tactical_evidence import TacticalKind
from baloot_platform.core.enums import Seat
from baloot_platform.research import TacticalPlayConstructionProfileV1
from baloot_platform.research import TacticalDealBalanceProfileV1
from baloot_platform.research.contracts import ResearchRecipeV1
from baloot_platform.research.construction_pre_screen import (
    TacticalViabilityPreScreenProfileV1,
)

from scripts.run_strategic_play_pilot import _recipe


class _ResearchStub:
    @staticmethod
    def capabilities():
        return {
            "sha256": "1" * 64,
            "bundleHash": "2" * 64,
            "rulesHash": "3" * 64,
            "conventionHash": "4" * 64,
            "research": {
                "defaultAgentRelease": "foundry_balanced@test",
                "analysisProfiles": {"screening": "5" * 64},
                "defaultScoreProfileHash": "6" * 64,
            },
        }


def _profile() -> dict:
    return {
        "schemaVersion": 1,
        "id": "tactical_play_precursor_v1",
        "revision": 1,
        "enabled": True,
        "maximumPayoffTricks": 4,
        "minimumLaterRawPoints": 10,
        "requirePrecursorForGatedTactics": True,
    }


def test_tactical_construction_profile_is_versioned_closed_and_hashed():
    profile = TacticalPlayConstructionProfileV1.build(_profile())
    assert len(profile.sha256) == 64
    assert profile.to_dict()["minimumLaterRawPoints"] == 10
    unknown = {**_profile(), "hiddenDefault": True}
    with pytest.raises(ValueError, match="unknown keys"):
        TacticalPlayConstructionProfileV1.build(unknown)


def test_tactical_construction_requires_cp_sat_feasibility_line():
    payload = _recipe(
        _ResearchStub(),
        mode="actor_safe_agent_verified",
        attempts=1,
        seed_start=1,
    )
    payload["resources"]["tacticalConstruction"] = _profile()
    with pytest.raises(ValueError, match="CP-SAT feasibility line"):
        ResearchRecipeV1.build(payload)

    valid = copy.deepcopy(payload)
    valid["resources"]["playConstructionMode"] = "cp_sat_feasibility_line"
    valid["resources"]["auctionConstructionMode"] = "cp_sat_feasibility_line"
    valid["tactics"][0].update({
        "kind": "small_card_big_gain",
        "mode": "qualification_gate",
    })
    recipe = ResearchRecipeV1.build(valid)
    assert recipe.resources["tacticalConstruction"]["enabled"] is True


def _viability_profile() -> dict:
    return {
        "schemaVersion": 1,
        "id": "exact_tactical_viability_screen_v1",
        "revision": "1",
        "enabled": True,
        "maximumDecisions": 8,
        "requiredTacticsOnly": True,
        "rejectWhenExactAbsent": True,
    }


def test_tactical_viability_pre_screen_is_closed_and_requires_a_gate():
    profile = TacticalViabilityPreScreenProfileV1.build(_viability_profile())
    assert len(profile.sha256) == 64
    unknown = {**_viability_profile(), "silentOverride": True}
    with pytest.raises(ValueError, match="unknown tactical viability"):
        TacticalViabilityPreScreenProfileV1.build(unknown)

    payload = _recipe(
        _ResearchStub(),
        mode="actor_safe_agent_verified",
        attempts=1,
        seed_start=1,
    )
    payload["resources"]["tacticalViabilityPreScreen"] = _viability_profile()
    with pytest.raises(ValueError, match="requires a hard or qualification-gate"):
        ResearchRecipeV1.build(payload)

    payload["tactics"][0]["mode"] = "qualification_gate"
    recipe = ResearchRecipeV1.build(payload)
    assert recipe.resources["tacticalViabilityPreScreen"]["enabled"] is True


def test_tactical_viability_probe_scans_past_unrelated_tactics(monkeypatch):
    def snapshot(decision_id: str, ordinal: int):
        return SimpleNamespace(
            decision=SimpleNamespace(
                decision_kind="play",
                forced=False,
                legal_actions=("left", "right"),
                decision_id=decision_id,
                acting_seat=Seat.NORTH,
                ordinal=ordinal,
            ),
            state=SimpleNamespace(hands=(1, 1, 1, 1)),
        )

    required = snapshot("DEC-REQUIRED", 1)
    unrelated = snapshot("DEC-UNRELATED", 2)

    class Store:
        def __init__(self, _trace, _rules):
            self.replay = SimpleNamespace(decisions=(required, unrelated))

    monkeypatch.setattr(oracle_module, "TraceDecisionProjectionStoreV1", Store)
    verifier = object.__new__(ActualDealOracleVerifierV1)
    verifier.rules = None
    verifier.maximum_remaining_cards = 12
    def action(snap, action_id, *, cancel_check=None):
        return SimpleNamespace(
            completion_status="exact",
            final_margin=1.0,
            sha256=f"{snap.decision.decision_id}:{action_id}",
            to_dict=lambda: {
                "completionStatus": "exact",
                "finalMargin": 1.0,
                "sha256": f"{snap.decision.decision_id}:{action_id}",
            },
        )

    verifier._action = action

    def detect(snap, _actions, _selected):
        kind = (
            TacticalKind.SMALL_CARD_BIG_GAIN
            if snap.decision.decision_id == "DEC-REQUIRED"
            else TacticalKind.ENTRY_UNBLOCKING
        )
        return (SimpleNamespace(
            kind=kind,
            causal_proof_hash=f"proof:{snap.decision.decision_id}",
            to_dict=lambda: {
                "kind": kind.value,
                "causalProofHash": f"proof:{snap.decision.decision_id}",
            },
        ),)

    verifier._detect_from_actions = detect
    trace = SimpleNamespace(sha256="a" * 64)

    result = verifier.probe_verified_tactical_play(
        trace,
        required_tactics=(TacticalKind.SMALL_CARD_BIG_GAIN,),
    )

    assert result.viable is True
    assert result.required_tactics == ("small_card_big_gain",)
    assert [item.decision_id for item in result.scanned_decisions] == [
        "DEC-UNRELATED",
        "DEC-REQUIRED",
    ]
    assert result.to_dict()["requiredTactics"] == ["small_card_big_gain"]

    absent = verifier.probe_verified_tactical_play(
        trace,
        required_tactics=(TacticalKind.KABOUT_CREATION,),
    )
    assert absent.viable is False
    assert absent.scan_complete is True
    assert absent.can_reject is True
    with pytest.raises(ValueError, match="not a valid TacticalKind"):
        verifier.probe_verified_tactical_play(
            trace,
            required_tactics=("unknown_tactic",),
        )


def test_tactical_deal_balance_is_closed_and_requires_a_supported_gate():
    raw = {
        "schemaVersion": 1,
        "id": "balanced_tactical_deal_v1",
        "revision": 1,
        "enabled": True,
        "maximumTeamHighCardDifference": 2,
        "minimumHighCardsPerSeat": 2,
        "maximumHighCardsPerSeat": 4,
        "minimumTargetLowCards": 1,
    }
    profile = TacticalDealBalanceProfileV1.build(raw)
    assert len(profile.sha256) == 64
    with pytest.raises(ValueError, match="unknown keys"):
        TacticalDealBalanceProfileV1.build({**raw, "hidden": True})

    payload = _recipe(
        _ResearchStub(),
        mode="actor_safe_agent_verified",
        attempts=1,
        seed_start=1,
    )
    payload["resources"]["tacticalDealBalance"] = raw
    payload["tactics"][0].update({
        "kind": "large_card_sacrifice",
        "mode": "qualification_gate",
    })
    with pytest.raises(ValueError, match="only-winning-move or small-card-big-gain"):
        ResearchRecipeV1.build(payload)
    for kind in ("only_winning_move", "small_card_big_gain"):
        supported = copy.deepcopy(payload)
        supported["tactics"][0]["kind"] = kind
        recipe = ResearchRecipeV1.build(supported)
        assert recipe.resources["tacticalDealBalance"]["enabled"] is True
