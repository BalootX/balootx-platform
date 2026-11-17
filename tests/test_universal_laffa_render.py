from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

from baloot_platform.challenge.evidence_v3 import DecisionLedgerV1, TruthMode, canonical_sha256
from baloot_platform.challenge.trace import CompleteGameTraceV1
from baloot_platform.core.enums import Seat
from baloot_platform.research import render_service as render_module
from baloot_platform.research.render_service import ResearchRenderPreparationServiceV1
from baloot_platform.research.render_service import _frozen_rulebook


def _trace() -> CompleteGameTraceV1:
    trusted = {
        "rulesHash": "b" * 64,
        "seed": 7,
        "dealer": "east",
        "shuffle": {
            "algorithm": "splitmix64_fisher_yates_v1",
            "seed": 7,
            "dealer": "east",
            "cutter": "south",
            "cutPosition": 9,
            "deckCommitment": "c" * 64,
            "deckOrder": list(range(32)),
        },
        "agentProfiles": {seat: "legacy" for seat in ("north", "west", "south", "east")},
        "events": [
            {"sequence": 0, "type": "round_created", "dealer": "east"},
            {"sequence": 1, "type": "card_played", "seat": "north", "card": "S7"},
        ],
    }
    ledger = DecisionLedgerV1.build(
        "a" * 64,
        ({
            "decisionId": "D1",
            "ordinal": 1,
            "actor": "north",
            "decisionKind": "play",
            "phase": "playing",
            "publicState": {"observer": "north"},
            "completeStateHash": "d" * 64,
            "legalActions": [{"suit": "spades", "rank": "seven"}, {"suit": "spades", "rank": "eight"}],
            "selectedAction": {"suit": "spades", "rank": "seven"},
        },),
    )
    game = {
        "gameHash": "a" * 64,
        "gameId": "GAME-1",
        "stateHash": canonical_sha256(trusted),
        "bundleHash": "e" * 64,
        "engineVersion": "1.0.0",
    }
    return CompleteGameTraceV1.build(
        game_record=game,
        trusted_final_state=trusted,
        decision_ledger=ledger,
    )


class _LegacyStore:
    def __init__(self, trace: CompleteGameTraceV1):
        self.trace = trace

    def get_game(self, identifier):
        return {"serial": "BX-LEGACY01", "gameHash": self.trace.game_hash}

    def complete_game_trace(self, identifier):
        return self.trace

    def list_decisions(self, identifier, *, limit):
        return [{
            "decisionId": "D1",
            "ordinal": 1,
            "decisionKind": "play",
            "actor": "north",
            "legalActions": [{"suit": "spades", "rank": "seven"}, {"suit": "spades", "rank": "eight"}],
            "selectedAction": {"suit": "spades", "rank": "seven"},
        }]

    def list_game_audits(self, identifier, *, limit):
        return []


class _LocatorService:
    def publish(self, *args, **kwargs):
        return SimpleNamespace(
            code="BX-LOCATOR1",
            universal_link="https://example.test/c/BX-LOCATOR1",
            to_dict=lambda: {
                "code": "BX-LOCATOR1",
                "universalLink": "https://example.test/c/BX-LOCATOR1",
            },
        )


def test_unqualified_legacy_laffa_prepares_as_non_publishable_audit_media(monkeypatch):
    trace = _trace()
    store = _LegacyStore(trace)
    actor = SimpleNamespace(
        truth_mode=TruthMode.ACTOR_OBSERVER,
        observer_seat=Seat.NORTH,
    )
    monkeypatch.setattr(
        render_module,
        "build_decision_perspective_evidence",
        lambda *args, **kwargs: (actor,),
    )
    monkeypatch.setattr(
        render_module,
        "TraceDecisionProjectionStoreV1",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    service = ResearchRenderPreparationServiceV1(
        research=SimpleNamespace(),
        artifact_store=SimpleNamespace(),
        publication_service=SimpleNamespace(),
        renderer=SimpleNamespace(),
        rules=SimpleNamespace(sha256=trace.rules_hash),
        learning_store=store,
        locator_service=_LocatorService(),
    )
    captured: dict[str, object] = {}

    def build_and_store(**kwargs):
        captured.update(copy.deepcopy(kwargs))
        return (
            {"ownerKind": "render_bundle_v1", "ownerHash": "f" * 64},
            {"bundleHash": "f" * 64, "sceneCount": 3},
        )

    monkeypatch.setattr(service, "_build_and_store", build_and_store)

    prepared = service.prepare_laffa("BX-LEGACY01")
    assert prepared["qualificationStatus"] == "legacy_replay"
    assert prepared["publicationEligible"] is False
    assert prepared["renderPurpose"] == "research_audit"
    assert prepared["qualificationFailures"] == [
        "analysis_unavailable",
        "legacy_content_qualification_not_established",
    ]
    assert prepared["basename"].endswith("-audit")
    response_source = prepared["source"]
    source = captured["source_metadata"]
    assert response_source == source
    assert source["sourceSchemaVersion"] == 2
    assert source["kind"] == "laffa"
    assert source["code"] == "BX-LEGACY01"
    assert source["decisionOrdinal"] == 1
    assert source["decisionId"] == prepared["decisionId"]
    assert source["exactStateId"] == prepared["decisionId"]
    assert source["qrActionable"] is False
    assert source["traceHash"] == trace.sha256
    assert source["rulesHash"] == trace.rules_hash
    assert source["renderPurpose"] == "research_audit"
    assert source["publicationEligible"] is False
    assert source["qualificationFailures"] == prepared["qualificationFailures"]
    assert {
        "resultHash",
        "analysisHash",
        "auditHash",
        "analysisProfileHash",
        "analyzerId",
        "analyzerVersion",
    }.isdisjoint(source)


def test_sakka_round_trace_reference_prepares_as_non_publishable_audit_media(monkeypatch):
    trace = _trace()
    actor = SimpleNamespace(
        truth_mode=TruthMode.ACTOR_OBSERVER,
        observer_seat=Seat.NORTH,
    )
    monkeypatch.setattr(
        render_module,
        "build_decision_perspective_evidence",
        lambda *args, **kwargs: (actor,),
    )
    monkeypatch.setattr(
        render_module,
        "TraceDecisionProjectionStoreV1",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    service = ResearchRenderPreparationServiceV1(
        research=SimpleNamespace(),
        artifact_store=SimpleNamespace(),
        publication_service=SimpleNamespace(),
        renderer=SimpleNamespace(),
        rules=SimpleNamespace(sha256=trace.rules_hash),
        locator_service=_LocatorService(),
    )
    monkeypatch.setattr(
        service,
        "_load",
        lambda _reference, **_kwargs: trace.to_dict(trusted=True),
    )
    captured: dict[str, object] = {}

    def build_and_store(**kwargs):
        captured.update(copy.deepcopy(kwargs))
        return (
            {"ownerKind": "render_bundle_v1", "ownerHash": "f" * 64},
            {"bundleHash": "f" * 64, "sceneCount": 3},
        )

    monkeypatch.setattr(service, "_build_and_store", build_and_store)

    prepared = service.prepare_trace_reference(
        {"uri": "cas://sha256/" + "1" * 64},
        code="LAF-SAKKA01",
    )
    assert prepared["publicationEligible"] is False
    assert prepared["qualificationFailures"] == [
        "analysis_unavailable",
        "content_qualification_not_established",
    ]
    response_source = prepared["source"]
    source = captured["source_metadata"]
    assert response_source == source
    assert source["sourceSchemaVersion"] == 2
    assert source["kind"] == "laffa"
    assert source["code"] == "LAF-SAKKA01"
    assert source["decisionOrdinal"] == 1
    assert source["decisionId"] == prepared["decisionId"]
    assert source["exactStateId"] == prepared["decisionId"]
    assert source["qrActionable"] is False
    assert source["traceHash"] == trace.sha256
    assert source["rulesHash"] == trace.rules_hash
    assert source["renderPurpose"] == "research_audit"
    assert source["publicationEligible"] is False
    assert source["qualificationFailures"] == prepared["qualificationFailures"]
    assert source["originKind"] == "sakka_round"
    assert {
        "resultHash",
        "analysisHash",
        "auditHash",
        "analysisProfileHash",
        "analyzerId",
        "analyzerVersion",
    }.isdisjoint(source)


def test_api_exposes_unified_laffa_prepare_and_render_linkage_paths():
    from baloot_platform.service.api import app

    paths = app.openapi()["paths"]
    assert "/api/v2/laffat/{identifier}:prepare-render" in paths
    assert "/api/v2/laffat/{identifier}/renders" in paths
    source_schema = app.openapi()["components"]["schemas"]["FoundryRenderSourceReference"]
    assert "laffa" in source_schema["properties"]["kind"]["enum"]


def test_historical_rules_accept_only_checksum_verified_retired_metadata():
    from baloot_platform.service.container import build_container

    active = build_container().research.simulator.rules
    frozen = copy.deepcopy(active.data)
    frozen["analysis_defaults"] = {"retired": True}
    expected_hash = canonical_sha256(frozen)

    restored = _frozen_rulebook(frozen, expected_hash)
    assert restored.sha256 == expected_hash
    assert "analysis_defaults" not in restored.data

    trace = replace(_trace(), rules_hash=expected_hash)
    configuration_store = SimpleNamespace(
        list_profiles=lambda **_filters: (
            SimpleNamespace(payload_sha256=expected_hash, payload=frozen),
        )
    )
    service = ResearchRenderPreparationServiceV1(
        research=SimpleNamespace(),
        artifact_store=SimpleNamespace(),
        publication_service=SimpleNamespace(),
        renderer=SimpleNamespace(),
        rules=active,
        configuration_store=configuration_store,
    )
    from_store = service._rules_for_trace(trace)
    assert from_store.sha256 == expected_hash
    assert "analysis_defaults" not in from_store.data

    frozen["invented_runtime_rule"] = True
    try:
        _frozen_rulebook(frozen, canonical_sha256(frozen))
    except ValueError as exc:
        assert "unknown rule sections" in str(exc)
    else:
        raise AssertionError("unknown historical behavior must fail closed")
