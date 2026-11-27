from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import yaml
import pytest

from baloot_platform.audit.models import content_hash
from baloot_platform.core.cards import FULL_DECK, Card, cards_to_mask
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, GamePhase, Seat, Team
from baloot_platform.core.exact_state_view import build_observer_safe_exact_state_v1
from baloot_platform.core.models import LegalPlayActionV1, RoundState
from baloot_platform.research.exact_state import (
    ExactStateNormalizerV1,
    NormalizedExactStateV1,
)
from baloot_platform.research.scenario_contracts import (
    ProofOutcomeV1,
    ResearchScenarioV1,
    canonical_sha256,
)
from baloot_platform.research.scenario_evidence import (
    ScenarioRuntimeVerifierV1,
    ScenarioSatisfactionV1,
)
from baloot_platform.research.scenario_result import (
    SCENARIO_ARTIFACT_MEDIA_TYPES,
    build_scenario_result_v4,
    canonical_json_bytes,
    public_proof_outcome_v1,
    public_scenario_result_v4,
    scenario_decision_audit_v1,
    validate_scenario_result_v4,
)
from baloot_platform.video.scenario_adapter import build_scenario_decision_scene_package
from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer
from baloot_platform.video.audio_postprocess import build_audio_cue_plan
from baloot_platform.video.render_bundle import (
    RenderBundleV1,
    build_export_timeline,
    build_render_bundle,
)


def _normalized_state(rules) -> NormalizedExactStateV1:
    state = RoundState(rules.sha256, 7, 1, Seat.EAST)
    state.phase = GamePhase.FINAL_DEALT
    state.contract = Contract.SUN
    state.buyer = Seat.NORTH
    state.declarer = Seat.NORTH
    state.first_leader = Seat.SOUTH
    state.initial_hands = [
        cards_to_mask(FULL_DECK[int(seat) * 8 : (int(seat) + 1) * 8]) for seat in Seat
    ]
    state.hands = list(state.initial_hands)
    BalootEngine(rules).prepare_projects(state)
    assert any(project.owner != Seat.SOUTH for project in state.projects)
    assert not any(project.declared or project.revealed for project in state.projects)
    state.project_winning_team = Team.NORTH_SOUTH
    payload = state.to_dict(reveal_hands=True)
    state_hash = canonical_sha256(payload)
    legal_actions = tuple(
        LegalPlayActionV1(card).to_dict() for card in state.hand(Seat.SOUTH)
    )
    core = {
        "schemaVersion": 1,
        "prefixHash": "a" * 64,
        "rulesHash": rules.sha256,
        "stateHash": state_hash,
        "phase": state.phase.value,
        "decisionKind": "play",
        "nextActor": "south",
        "legalActions": list(legal_actions),
        "eventChainHash": canonical_sha256(payload["events"]),
    }
    digest = canonical_sha256(core)
    return NormalizedExactStateV1(
        f"STATE-{digest[:20].upper()}",
        core["prefixHash"],
        rules.sha256,
        state_hash,
        state.phase.value,
        "play",
        "south",
        legal_actions,
        payload,
        0,
        0,
        core["eventChainHash"],
        digest,
        state,
        rules,
    )


def _normalized_bid_state(rules) -> NormalizedExactStateV1:
    return ExactStateNormalizerV1(rules).normalize(
        {
            "schemaVersion": 1,
            "deckOrder": list(range(32)),
            "dealer": "east",
            "roundNumber": 1,
            "matchScore": [0, 0],
            "auctionActions": [],
            "escalationActions": [],
            "projectDeclarations": [],
            "projectAssertions": [],
            "playActions": [],
            "stateAssertions": {"phase": "bidding"},
        },
        decision_point={"kind": "bid", "selector": {}},
    )


def _normalized_escalation_state(rules) -> NormalizedExactStateV1:
    prefix = {
        "schemaVersion": 1,
        "deckOrder": list(range(32)),
        "dealer": "east",
        "roundNumber": 1,
        "matchScore": [101, 0],
        "auctionActions": [
            {"action": "pass"},
            {"action": "pass"},
            {"action": "sun"},
            {"action": "pass"},
            {"action": "sun"},
        ],
        "escalationActions": [],
        "projectDeclarations": [],
        "projectAssertions": [],
        "playActions": [],
        "stateAssertions": {"phase": "final_dealt"},
    }
    playing = ExactStateNormalizerV1(rules).normalize(
        {**prefix, "stateAssertions": {}}
    )
    declarer = playing.round_state.declarer
    assert declarer is not None
    actor = next(seat for seat in Seat if seat.team != declarer.team)
    return ExactStateNormalizerV1(rules).normalize(
        prefix,
        decision_point={
            "kind": "escalation",
            "selector": {"actingSeat": actor.name.lower()},
        },
    )


def _normalized_project_state(rules) -> NormalizedExactStateV1:
    return ExactStateNormalizerV1(rules).normalize(
        {
            "schemaVersion": 1,
            "deckOrder": list(range(32)),
            "dealer": "east",
            "roundNumber": 1,
            "matchScore": [0, 0],
            "auctionActions": [
                {"action": "pass"},
                {"action": "pass"},
                {"action": "sun"},
                {"action": "pass"},
                {"action": "sun"},
            ],
            "escalationActions": [],
            "projectDeclarations": [],
            "projectAssertions": [],
            "playActions": [],
            "stateAssertions": {"phase": "playing"},
        },
        decision_point={
            "kind": "project",
            "selector": {"owner": "north", "projectIndex": 0},
        },
    )


def _completed_trace(rules, *, match_score: tuple[int, int]) -> dict[str, object]:
    engine = BalootEngine(rules)
    state = engine.create_round(
        seed=8172,
        dealer=Seat.EAST,
        match_score=match_score,
    )
    engine.initial_deal_from_preordered(state, list(range(32)))
    state.phase = GamePhase.BIDDING
    assert state.purchase_card is not None
    machine = BiddingMachine(
        state.dealer,
        state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(rules.data),
    )
    for action in ("pass", "pass", "sun", "pass", "sun"):
        machine.submit(action)
    assert machine.result is not None
    engine.complete_deal_from_bid_result(state, machine.result)
    engine.prepare_projects(state)
    bot = HeuristicBot(engine, BotProfile(8172))
    while sum(hand.bit_count() for hand in state.hands):
        engine.play_card(state, bot.choose(state))
    assert state.phase == GamePhase.SCORED
    return {
        "completeExactNormalization": {
            "rulesHash": rules.sha256,
            "state": state.to_dict(reveal_hands=True),
        }
    }


def _nonplay_production_bundle(
    root,
    rules,
    normalized,
    *,
    match_score,
    export_preset="hooked_flagship",
):
    profile = yaml.safe_load(
        (root / "configs" / "video" / "premium.yaml").read_text(encoding="utf-8")
    )
    package = build_scenario_decision_scene_package(
        exact_state=normalized.to_dict(),
        proof_outcome=_proof(normalized).to_dict(),
        scenario_projection=_projection(normalized),
        profile=profile,
        effective_config_sha256="f" * 64,
        observer_seat=str(normalized.next_actor),
        rules=rules,
        trusted_complete_trace=_completed_trace(rules, match_score=match_score),
    )
    timeline = build_export_timeline(
        scenes=package.scenes,
        profile=profile,
        export_preset=export_preset,
    )
    question = next(
        scene for scene in package.scenes if scene.value["stage"] == "question"
    )
    bundle = build_render_bundle(
        scenes=package.scenes,
        timeline=timeline,
        profile=profile,
        export_preset=export_preset,
        localizations=profile["localizations"],
        identity={
            "gameSerial": f"TEST-{normalized.decision_kind.upper()}",
            "sceneId": question.scene_id,
            "enrollmentUri": "https://example.invalid/nonplay",
            "players": dict(profile["identity"]["default_player_names"]),
        },
        repository_root=root,
    )
    return profile, package, bundle


def test_complete_play_video_reconstructs_only_the_observers_initial_five(
    root: Path, rules,
) -> None:
    prefix = {
        "schemaVersion": 1,
        "deckOrder": list(range(32)),
        "dealer": "east",
        "roundNumber": 1,
        "matchScore": [0, 0],
        "auctionActions": [
            {"action": "pass"}, {"action": "pass"}, {"action": "sun"},
            {"action": "pass"}, {"action": "sun"},
        ],
        "escalationActions": [],
        "projectDeclarations": [],
        "projectAssertions": [],
        "playActions": [],
        "stateAssertions": {},
    }
    normalized = ExactStateNormalizerV1(rules).normalize(
        prefix, decision_point={"kind": "play", "selector": {}}
    )
    complete_trace = {
        **_completed_trace(rules, match_score=(0, 0)),
        "completeExactTracePrefix": {"deckOrder": prefix["deckOrder"]},
    }
    profile = yaml.safe_load(
        (root / "configs" / "video" / "premium.yaml").read_text(encoding="utf-8")
    )
    package = build_scenario_decision_scene_package(
        exact_state=normalized.to_dict(),
        proof_outcome=_proof(normalized).to_dict(),
        scenario_projection=_projection(normalized),
        profile=profile,
        effective_config_sha256="f" * 64,
        observer_seat=normalized.next_actor,
        rules=rules,
        trusted_complete_trace=complete_trace,
    )
    context = package.scene("context").value
    layers = {layer["id"]: layer["content"] for layer in context["layers"]}
    visible = layers["observer-lifecycle"]
    assert len(visible["initialFiveCards"]) == 5
    assert {card["id"] for card in visible["initialFiveCards"]} <= {
        card["id"] for card in visible["initialHandCards"]
    }
    assert "initialFiveCards" not in layers["public-lifecycle"]
    timeline = build_export_timeline(
        scenes=package.scenes, profile=profile, export_preset="hooked_flagship"
    )
    assert [segment["stage"] for segment in timeline.to_dict()["segments"]][:4] == [
        "hook", "rewind", "setup", "replay",
    ]
    bundle = build_render_bundle(
        scenes=package.scenes,
        timeline=timeline,
        profile=profile,
        export_preset="hooked_flagship",
        localizations=profile["localizations"],
        identity={
            "gameSerial": "TEST-PREPLAY",
            "sceneId": package.scene("question").scene_id,
            "enrollmentUri": "https://example.invalid/preplay",
            "players": dict(profile["identity"]["default_player_names"]),
        },
        repository_root=root,
    )
    assert bundle.verify()
    opening_ids = sorted(card["id"] for card in visible["initialFiveCards"])
    opening_checkpoint = {
        "completedPlayCount": 0,
        "frame": 0,
        "source": "playback-00.png",
        "renderedDom": {
            "frame": 0, "completedPlayCount": 0, "cards": [],
            "hands": [
                {"seat": seat, "count": 5,
                 "faceCardIds": opening_ids if seat == normalized.next_actor else []}
                for seat in ("north", "west", "south", "east")
            ],
        },
    }
    checkpoints = {"items": [opening_checkpoint] + [
        {"completedPlayCount": count, "frame": count, "source": f"playback-{count:02d}.png"}
        for count in range(1, 33)
    ]}
    renderer = RemotionProfessionalRenderer(profile, repository_root=root)
    playback = renderer._playback_state_evidence(bundle, checkpoints)
    assert playback[0]["observerHandCardIds"] == opening_ids
    assert set(playback[0]["handCounts"].values()) == {5}
    checkpoints["items"][0]["renderedDom"]["hands"][0]["count"] = 8
    with pytest.raises(ValueError, match="observer safety"):
        renderer._playback_state_evidence(bundle, checkpoints)
    substituted = copy.deepcopy(complete_trace)
    substituted["completeExactTracePrefix"]["deckOrder"][0], substituted[
        "completeExactTracePrefix"
    ]["deckOrder"][31] = (
        substituted["completeExactTracePrefix"]["deckOrder"][31],
        substituted["completeExactTracePrefix"]["deckOrder"][0],
    )
    with pytest.raises(ValueError, match="observer hand disagrees"):
        build_scenario_decision_scene_package(
            exact_state=normalized.to_dict(),
            proof_outcome=_proof(normalized).to_dict(),
            scenario_projection=_projection(normalized),
            profile=profile,
            effective_config_sha256="f" * 64,
            observer_seat=normalized.next_actor,
            rules=rules,
            trusted_complete_trace=substituted,
        )


def _ekka_decision_prefix() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "deckOrder": [
            5,
            6,
            11,
            0,
            4,
            7,
            2,
            8,
            13,
            1,
            3,
            9,
            14,
            21,
            12,
            19,
            15,
            17,
            16,
            18,
            10,
            25,
            27,
            28,
            29,
            31,
            22,
            24,
            26,
            20,
            23,
            30,
        ],
        "dealer": "east",
        "roundNumber": 1,
        "matchScore": [0, 0],
        "auctionActions": [
            {"actor": "north", "action": "hokum"},
            {"actor": "west", "action": "pass"},
            {"actor": "south", "action": "pass"},
            {"actor": "east", "action": "pass"},
        ],
        "escalationActions": [],
        "projectDeclarations": [],
        "projectAssertions": [],
        "playActions": [
            {"actor": actor, "card": card}
            for actor, card in (
                ("north", "H10"),
                ("west", "HJ"),
                ("south", "HQ"),
                ("east", "H8"),
                ("west", "CJ"),
                ("south", "C9"),
                ("east", "CK"),
                ("north", "C10"),
                ("north", "DQ"),
                ("west", "D10"),
                ("south", "DK"),
                ("east", "DA"),
                ("east", "S10"),
                ("north", "SQ"),
                ("west", "SJ"),
                ("south", "S9"),
                ("east", "D9"),
                ("north", "HK"),
                ("west", "SA"),
                ("south", "D8"),
                ("north", "H9"),
                ("west", "CQ"),
                ("south", "HA"),
                ("east", "DJ"),
            )
        ],
        "stateAssertions": {"phase": "playing"},
    }


def _ashkal_prefix() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "deckOrder": list(range(32)),
        "dealer": "east",
        "roundNumber": 1,
        "matchScore": [0, 0],
        "auctionActions": [
            {"actor": "north", "action": "pass"},
            {"actor": "west", "action": "pass"},
            {"actor": "south", "action": "ashkal"},
            {"actor": "west", "action": "pass"},
            {"actor": "north", "action": "pass"},
        ],
        "escalationActions": [],
        "projectDeclarations": [],
        "projectAssertions": [],
        "playActions": [],
        "stateAssertions": {"phase": "playing"},
    }


def _projection(normalized: NormalizedExactStateV1) -> dict[str, object]:
    return {
        "exactStateId": normalized.exact_state_id,
        "decisionKind": normalized.decision_kind,
    }


def _resign_exact_state(exact_state: dict) -> dict:
    exact_state["stateHash"] = canonical_sha256(exact_state["state"])
    exact_state["eventChainHash"] = canonical_sha256(
        exact_state["state"].get("events") or []
    )
    exact_state["phase"] = exact_state["state"]["phase"]
    core = {
        key: copy.deepcopy(exact_state[key])
        for key in (
            "schemaVersion",
            "prefixHash",
            "rulesHash",
            "stateHash",
            "phase",
            "decisionKind",
            "nextActor",
            "legalActions",
            "eventChainHash",
        )
    }
    exact_state["sha256"] = canonical_sha256(core)
    exact_state["exactStateId"] = f"STATE-{exact_state['sha256'][:20].upper()}"
    return exact_state


def _exact_projection(exact_state: dict) -> dict[str, object]:
    return {
        "exactStateId": exact_state["exactStateId"],
        "decisionKind": exact_state["decisionKind"],
    }


def test_playable_view_accepts_legacy_project_count_identity(rules) -> None:
    exact_state = _normalized_state(rules).to_dict()
    exact_state["legalActions"] = [
        copy.deepcopy(action["card"]) for action in exact_state["legalActions"]
    ]
    core = {
        key: copy.deepcopy(exact_state[key])
        for key in (
            "schemaVersion",
            "prefixHash",
            "rulesHash",
            "stateHash",
            "phase",
            "decisionKind",
            "nextActor",
            "legalActions",
            "eventChainHash",
            "projectDeclarationsVerified",
            "projectAssertionsVerified",
        )
    }
    exact_state["sha256"] = canonical_sha256(core)
    exact_state["exactStateId"] = f"STATE-{exact_state['sha256'][:20].upper()}"

    public = build_observer_safe_exact_state_v1(
        exact_state=exact_state,
        scenario_projection=_exact_projection(exact_state),
        observer_seat=str(exact_state["nextActor"]),
    )

    assert public["decisionKind"] == exact_state["decisionKind"]
    assert [action["code"] for action in public["legalActions"]] == [
        action["code"] for action in exact_state["legalActions"]
    ]


def _hidden_project_event(normalized: NormalizedExactStateV1) -> dict[str, object]:
    state = normalized.to_dict()["state"]
    project = copy.deepcopy(
        next(item for item in state["projects"] if item["owner"] != "south")
    )
    project["cards"] = [state["initialHands"][project["owner"]][0]]
    project["revealed"] = False
    return {
        "sequence": len(state["events"]),
        "type": "projects_revealed",
        "owner": project["owner"],
        "projects": [project],
    }


def test_escalation_exact_view_rejects_actor_inconsistent_with_state(rules) -> None:
    normalized = _normalized_escalation_state(rules)
    corrupted = normalized.to_dict()
    declarer = str(corrupted["state"]["declarer"])
    assert declarer != corrupted["nextActor"]
    corrupted["nextActor"] = declarer
    _resign_exact_state(corrupted)

    with pytest.raises(ValueError, match="escalation actor mismatch"):
        build_observer_safe_exact_state_v1(
            exact_state=corrupted,
            scenario_projection=_exact_projection(corrupted),
            observer_seat=declarer,
        )


def _proof(normalized: NormalizedExactStateV1, *, exact_state: dict | None = None):
    exact_state = exact_state if exact_state is not None else normalized.to_dict()
    actions = [item["code"] for item in normalized.legal_actions]
    payload = {
        "requestedMode": "exact",
        "achievedMode": "exact",
        "requestedVisibility": "complete_state",
        "achievedVisibility": "complete_state",
        "completionStatus": "exact_complete",
        "claim": "outcome_reachable",
        "claimSatisfied": True,
        "targetTeam": "north_south",
        "winningRootActionCount": len(actions),
        "rootActions": [
            {"action": action, "value": 1.0, "satisfies": True} for action in actions
        ],
        "selectedAction": actions[0],
        "principalVariation": [actions[0]],
        "hiddenWorlds": {
            "count": 1,
            "complete": True,
            "hashes": [exact_state["stateHash"]],
            "coverage": {"method": "complete_state"},
        },
        "identities": {
            "rulesHash": exact_state["rulesHash"],
            "stateHash": exact_state["stateHash"],
            "exactStateId": exact_state["exactStateId"],
        },
        "resources": {},
    }
    return SimpleNamespace(
        payload=payload,
        sha256="b" * 64,
        to_dict=lambda: {**copy.deepcopy(payload), "sha256": "b" * 64},
    )


def _proof_outcome(normalized: NormalizedExactStateV1) -> ProofOutcomeV1:
    actions = [item["code"] for item in normalized.legal_actions]
    return ProofOutcomeV1.build(
        {
            "schemaVersion": 1,
            "requestedMode": "exact",
            "achievedMode": "exact",
            "requestedVisibility": "complete_state",
            "achievedVisibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "round_win",
            "successThreshold": 0.0,
            "horizon": None,
            "boundaryEvaluator": None,
            "rootActions": [
                {"action": action, "value": 1.0, "satisfies": True}
                for action in actions
            ],
            "selectedAction": actions[0],
            "winningRootActionCount": len(actions),
            "continuationLines": {
                "complete": True,
                "exactCount": "1",
                "lineUniverse": "all_legal",
            },
            "hiddenWorlds": {
                "count": 1,
                "complete": True,
                "hashes": [normalized.state_hash],
                "coverage": {"method": "complete_state"},
            },
            "resources": {"nodes": 1},
            "completionStatus": "exact_complete",
            "claimSatisfied": True,
            "principalVariation": [actions[0]],
            "identities": {
                "rulesHash": normalized.rules_hash,
                "bundleHash": "1" * 64,
                "stateHash": normalized.state_hash,
                "exactStateId": normalized.exact_state_id,
                "proofProfileHash": "2" * 64,
            },
        }
    )


class _ArtifactStore:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    def put(self, name: str, document: dict, media_type: str) -> dict[str, object]:
        payload = canonical_json_bytes(document)
        uri = f"memory://{name}"
        self.values[uri] = payload
        return {
            "uri": uri,
            "sha256": canonical_sha256(document),
            "bytes": len(payload),
            "mediaType": media_type,
            "name": f"{name}.json",
        }

    def get(self, uri: str) -> bytes:
        return self.values[uri]


def _validated_result(
    normalized: NormalizedExactStateV1,
    *,
    proof_override: ProofOutcomeV1 | None = None,
):
    scenario = SimpleNamespace(
        origin=SimpleNamespace(kind="exact_trace_prefix"),
        sha256="3" * 64,
        decision_point={"kind": normalized.decision_kind},
        predicates=lambda: (),
    )
    recipe = SimpleNamespace(
        code="RCP-OBSERVER-VALIDATION",
        sha256="4" * 64,
        configuration={"bundleRevisionId": "CBR-TEST"},
        scenario=scenario,
        resources={"generationLane": "exact_input"},
    )
    satisfaction_core = {
        "schemaVersion": 1,
        "scenarioHash": scenario.sha256,
        "exactStateId": normalized.exact_state_id,
        "constraintsSatisfied": True,
        "outcomeSatisfied": True,
        "hardPredicatesSatisfied": True,
        "status": "satisfied",
        "predicateEvidenceHashes": [],
    }
    satisfaction = ScenarioSatisfactionV1(
        scenario.sha256,
        normalized.exact_state_id,
        True,
        True,
        True,
        "satisfied",
        (),
        canonical_sha256(satisfaction_core),
    )
    proof = proof_override or _proof_outcome(normalized)
    constraint_core = {"schemaVersion": 1, "status": "satisfied"}
    constraint = {**constraint_core, "sha256": canonical_sha256(constraint_core)}
    documents = {
        "exactState": normalized.to_dict(),
        "scenarioSatisfaction": satisfaction.to_dict(),
        "constraintEvidence": constraint,
        "proofOutcome": proof.to_dict(),
    }
    store = _ArtifactStore()
    artifacts = {
        key: store.put(key, document, SCENARIO_ARTIFACT_MEDIA_TYPES[key])
        for key, document in documents.items()
    }
    result = build_scenario_result_v4(
        run={"runCode": "RUN-OBSERVER-VALIDATION", "runHash": "5" * 64},
        recipe=recipe,
        normalized=normalized,
        satisfaction=satisfaction,
        proof=proof,
        constraint_evidence=constraint,
        artifacts=artifacts,
        created_at="2026-08-23T00:00:00+00:00",
    )
    return result, store


@pytest.mark.parametrize(
    "proof_status", ["complete", "unknown_count", "partial_count", "known_count", "known_zero"]
)
def test_count_evidence_and_decision_reads_preserve_uncertainty(
    rules, monkeypatch, proof_status
) -> None:
    normalized = _normalized_state(rules)
    proof_value = _proof_outcome(normalized).to_dict()
    incomplete = proof_status != "complete"
    if incomplete:
        proof_value.pop("sha256")
        proof_value.pop("semanticSha256", None)
        proof_value.update({
            "completionStatus": "proof_incomplete",
            "claimSatisfied": False,
            "continuationLines": {
                "complete": False, "lowerBound": "1", "upperBound": None,
                "lineUniverse": "all_legal",
            },
        })
    uncertain_count = proof_status in {"unknown_count", "partial_count"}
    if uncertain_count:
        proof_value.update({
            "rootActions": proof_value["rootActions"][:1],
            "winningRootActionCount": None if proof_status == "unknown_count" else 1,
            "winningRootActionLowerBound": 1,
            "winningRootActionUpperBound": len(normalized.legal_actions),
        })
    elif proof_status == "known_zero":
        proof_value.update({
            "winningRootActionCount": 0,
            "winningRootActionLowerBound": 0,
            "winningRootActionUpperBound": 0,
        })
        for row in proof_value["rootActions"]:
            row.update({"satisfies": False, "value": -1.0})
    if proof_status == "partial_count":
        proof_value["resources"]["rootActionCoverageComplete"] = False
    elif proof_status in {"known_count", "known_zero"}:
        proof_value["resources"]["rootActionCoverageComplete"] = True
    proof = ProofOutcomeV1.build(proof_value)
    count = proof.payload["winningRootActionCount"]
    scenario = ResearchScenarioV1.build({
        "schemaVersion": 1,
        "origin": {"kind": "search"},
        "constraints": {"op": "all", "children": [{
            "predicateId": "PRED-CONTRACT", "kind": "auction.contract",
            "version": 1, "hardness": "hard", "arguments": {"contract": "sun"},
        }, {
            "predicateId": "PRED-RANK", "kind": "proof.winning_root_card_rank_count",
            "version": 1, "hardness": "hard",
            "arguments": {"rank": ["ace"], "minimum": 0, "maximum": 8},
        }]},
        "decisionPoint": {"kind": "play", "selector": {}},
        "outcome": {
            "predicateId": "PRED-COUNT", "kind": "proof.winning_root_action_count",
            "version": 1, "hardness": "hard",
            "arguments": {"minimum": 0, "maximum": len(normalized.legal_actions)},
        },
    })
    satisfaction = ScenarioRuntimeVerifierV1().verify(
        scenario, normalized, proof_outcome=proof
    )
    evidence = next(row for row in satisfaction.evidence if row.predicate_id == "PRED-COUNT")
    rank_evidence = next(row for row in satisfaction.evidence if row.predicate_id == "PRED-RANK")
    for row in (evidence, rank_evidence):
        assert row.status == ("not_evaluated" if uncertain_count else "satisfied")
        assert row.failure_code == ("proof_incomplete" if uncertain_count else None)
    assert satisfaction.status == (
        "semantic_postcondition_miss" if uncertain_count else "satisfied"
    )
    if uncertain_count:
        assert "count" not in evidence.observed
        assert "count" not in rank_evidence.observed
    else:
        assert evidence.observed["count"] == count
        assert rank_evidence.observed["count"] == (0 if count == 0 else 1)
    result, store = _validated_result(normalized, proof_override=proof)
    validated = validate_scenario_result_v4(result, artifact_store=store)
    assert validated["scenarioProjection"]["winningRootActionCount"] == count
    for key in ("winningRootActionLowerBound", "winningRootActionUpperBound"):
        assert validated["scenarioProjection"][key] == proof.payload[key]
    if incomplete:
        assert validated["qualification"]["qualified"] is False
        assert validated["truthLabel"] == "rejected"
        assert validated["disposition"] == "rejected"
        assert validated["confidence"] == validated["score"] == 0
    from service_test_client import OperatorClient as TestClient

    from baloot_platform.research.service import ResearchFoundryServiceV1
    from baloot_platform.service import api

    service = object.__new__(ResearchFoundryServiceV1)
    monkeypatch.setattr(service, "get_result", lambda _code: validated)
    monkeypatch.setattr(api, "RESEARCH", service)
    monkeypatch.setattr(api, "_require_trusted", lambda _request: None)
    path = f"/api/v2/research-results/{result['resultCode']}/decisions"
    with TestClient(api.app) as client:
        listed = client.get(path)
        single = client.get(f"{path}/1")
    assert listed.status_code == single.status_code == 200
    decision = single.json()
    assert listed.json()["items"] == [decision]
    assert decision["analyticalExplanation"]["winningRootActionCount"] == count
    assert decision["proof"]["winningRootActionCount"] == count
    assert decision["alternatives"] == validated["proofOutcome"]["rootActions"]
    assert decision["recommendationAvailable"] is not incomplete
    assert validated["proofOutcome"]["winningRootActionCount"] == count


def test_playable_exact_state_matches_engine_project_visibility(rules) -> None:
    normalized = _normalized_state(rules)
    exact_state = normalized.to_dict()
    exact_state["state"]["events"].append(
        {
            "sequence": len(exact_state["state"]["events"]),
            "type": "projects_compared",
            "winningTeam": "north_south",
            "declarations": [],
        }
    )
    exact_state["state"]["events"].append(_hidden_project_event(normalized))
    _resign_exact_state(exact_state)
    view = build_observer_safe_exact_state_v1(
        exact_state=exact_state,
        scenario_projection=_exact_projection(exact_state),
        observer_seat="south",
    )
    public = view["publicState"]
    canonical = BalootEngine.observer_view(normalized.round_state, Seat.SOUTH)
    assert public["projects"] == canonical["projects"]
    assert {project["owner"] for project in public["projects"]} == {"south"}
    assert public["projectWinningTeam"] is None
    assert public["projectWinningTeam"] == canonical["projectWinningTeam"]
    compared = next(
        event for event in public["events"] if event["type"] == "projects_compared"
    )
    assert compared["winningTeam"] is None
    assert not any(event["type"] == "projects_revealed" for event in public["events"])


@pytest.mark.parametrize("play_actor", ["south", "west"])
def test_playable_exact_state_strips_historical_legal_cards_from_tricks(rules, play_actor) -> None:
    normalized = _normalized_state(rules)
    exact_state = normalized.to_dict()
    card = copy.deepcopy(exact_state["state"]["hands"][play_actor][0])
    exact_state["state"]["currentTrick"] = {
        "index": 0,
        "leader": play_actor,
        "plays": [
            {
                "seat": play_actor,
                "card": card,
                "trickIndex": 0,
                "position": 0,
                "legalCards": copy.deepcopy(exact_state["state"]["hands"][play_actor]),
                "ekka": False,
                "baloot": False,
            }
        ],
        "winner": None,
        "rawPoints": 0,
    }
    exact_state["state"]["hands"][play_actor].pop(0)
    exact_state["legalActions"] = [
        LegalPlayActionV1(Card.parse(item)).to_dict()
        for item in exact_state["state"]["hands"]["south"]
    ]
    _resign_exact_state(exact_state)
    view = build_observer_safe_exact_state_v1(
        exact_state=exact_state,
        scenario_projection=_exact_projection(exact_state),
        observer_seat="south",
    )
    play = view["publicState"]["currentTrick"]["plays"][0]
    assert play["card"] == card
    assert "legalCards" not in play


@pytest.mark.parametrize("revealed", [False, True])
def test_declared_and_revealed_opponent_projects_follow_engine_visibility(
    rules,
    revealed: bool,
) -> None:
    normalized = _normalized_state(rules)
    state = normalized.round_state.clone()
    index, project = next(
        (index, project)
        for index, project in enumerate(state.projects)
        if project.owner != Seat.SOUTH
    )
    public_project = replace(
        project,
        declared=True,
        declaration_order=1,
        scoring=True,
        revealed=revealed,
    )
    state.projects[index] = public_project
    state.project_winning_team = public_project.team
    if revealed:
        state.emit(
            "projects_revealed",
            owner=public_project.owner.name.lower(),
            projects=[public_project.to_dict()],
        )
    else:
        declaration = public_project.public_declaration()
        state.emit("project_declared", **declaration)
    exact_state = normalized.to_dict()
    exact_state["state"] = state.to_dict(reveal_hands=True)
    _resign_exact_state(exact_state)
    view = build_observer_safe_exact_state_v1(
        exact_state=exact_state,
        scenario_projection=_exact_projection(exact_state),
        observer_seat="south",
    )
    public = view["publicState"]
    canonical = BalootEngine.observer_view(state, Seat.SOUTH)
    assert public["projects"] == canonical["projects"]
    visible = next(
        item
        for item in public["projects"]
        if item["owner"] == project.owner.name.lower()
    )
    assert visible["cards"] == (
        [card.to_dict() for card in public_project.cards] if revealed else None
    )
    assert public["projectWinningTeam"] == public_project.team.name.lower()
    expected_event = "projects_revealed" if revealed else "project_declared"
    assert any(event["type"] == expected_event for event in public["events"])


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("mid_round", [False, True])
def test_play_render_preserves_retained_trace_instead_of_proof_line(root, rules, corrupt, mid_round):
    normalized = (
        ExactStateNormalizerV1(rules).normalize(_ekka_decision_prefix())
        if mid_round else _normalized_state(rules)
    )
    observer = str(normalized.next_actor)
    pause = len(normalized.round_state.played_cards)
    state = copy.deepcopy(normalized.round_state)
    engine = BalootEngine(rules)
    while sum(hand.bit_count() for hand in state.hands):
        action = engine.legal_play_actions(state)[-1]
        engine.play_card(state, action.card, ekka=action.ekka)
    completed = state.to_dict(reveal_hands=True)
    if corrupt:
        first = completed["completedTricks"][0]["plays"][0]
        first["seat"] = "south" if first["seat"] != "south" else "north"
    profile = yaml.safe_load(
        (root / "configs" / "video" / "premium.yaml").read_text(encoding="utf-8")
    )
    kwargs = dict(
        exact_state=normalized.to_dict(),
        proof_outcome=_proof(normalized).to_dict(),
        scenario_projection=_projection(normalized),
        profile=profile,
        effective_config_sha256="f" * 64,
        observer_seat=observer,
        rules=rules,
        trusted_principal_variation=["invalid-proof-line"],
        trusted_complete_trace={
            "completeExactNormalization": {"rulesHash": rules.sha256, "state": completed}
        },
    )
    if corrupt:
        with pytest.raises(ValueError, match="continuation (replay|decision prefix) mismatch"):
            build_scenario_decision_scene_package(**kwargs)
        return
    package = build_scenario_decision_scene_package(**kwargs)
    layers = {layer["id"]: layer["content"] for layer in package.scene("continuation").value["layers"]}
    plays = layers["public-lifecycle"]["plays"]
    expected = [play for trick in completed["completedTricks"] for play in trick["plays"]]
    assert [(p["seat"], p["card"], p["ekka"]) for p in plays] == [
        (p["seat"], p["card"], p["ekka"]) for p in expected
    ]
    assert layers["public-lifecycle"]["pauseBeforePlayIndex"] == pause
    hand = {card["code"] for card in layers["observer-lifecycle"]["initialHandCards"]}
    assert len(hand) == 8
    for play in plays:
        if play["seat"] == observer:
            hand.remove(play["card"]["code"])
    assert not hand


def test_forced_final_card_does_not_request_missing_score_swing_alternative() -> None:
    from baloot_platform.video.scenario_adapter import _proof_reason_key

    assert _proof_reason_key(
        {
            "achievedMode": "exact",
            "completionStatus": "exact_complete",
            "utility": "awarded_point_margin",
            "winningRootActionCount": 1,
            "selectedAction": "H7",
            "rootActions": [{"action": "H7", "value": 4, "satisfies": True}],
        },
        {"video.analysis.unique_awarded_margin": "copy"},
    ) == "video.analysis.one_winning_action"


@pytest.mark.parametrize("winning", [1, 2])
def test_actor_margin_copy_does_not_present_hidden_world_guarantee_as_actual_score(winning) -> None:
    from baloot_platform.video.scenario_adapter import _proof_reason_key

    proof = {"achievedMode": "exact", "completionStatus": "exact_complete",
        "achievedVisibility": "actor_information_set", "utility": "awarded_point_margin",
        "winningRootActionCount": winning, "selectedAction": "H7",
        "rootActions": [{"action": "H7", "value": 16}, {"action": "CA", "value": -54}]}
    assert _proof_reason_key(proof, {"video.analysis.unique_awarded_margin": "actual deal copy"}) == (
        "video.analysis.one_winning_action" if winning == 1 else "video.analysis.multiple_winning_actions")


@pytest.mark.parametrize(
    "requested,achieved,completion,scope",
    [
        ("complete_state", "complete_state", "exact_complete", "complete_state"),
        ("actor_information_set", "actor_information_set", "exact_complete", "actor_information_set"),
        ("actor_information_set", "none", "proof_incomplete", "unproven"),
        ("actor_information_set", "complete_state", "exact_complete", "complete_state"),
        ("complete_state", "complete_state", "bounded_complete", "complete_state"),
        ("actor_information_set", "actor_information_set", "policy_complete", "actor_information_set"),
        ("complete_state", "complete_state", "proof_incomplete", "unproven"),
        ("complete_state", "complete_state", "proof_failed", "complete_state"),
    ],
)
def test_render_discloses_achieved_proof_scope_only(
    root: Path, rules, requested, achieved, completion, scope,
) -> None:
    normalized = _normalized_state(rules)
    profile = yaml.safe_load((root / "configs/video/premium.yaml").read_text(encoding="utf-8"))
    proof = {**_proof(normalized).to_dict(), "requestedVisibility": requested,
             "achievedVisibility": achieved, "completionStatus": completion}
    kwargs = dict(
        exact_state=normalized.to_dict(), proof_outcome=proof,
        scenario_projection=_projection(normalized), profile=profile,
        effective_config_sha256="f" * 64, observer_seat="south", rules=rules,
    )
    package = build_scenario_decision_scene_package(**kwargs)
    reveal = package.scene("reveal").to_dict()
    analysis = next(layer["content"] for layer in reveal["layers"] if layer["id"] == "analysis")
    assert analysis["proofScopeKey"] == f"video.analysis.proof_scope.{scope}"
    assert analysis["requestedVisibility"] == requested
    assert analysis["achievedVisibility"] == achieved
    assert all(layer["id"] != "analysis" for scene in package.scenes
               if scene.value["stage"] in {"context", "question"} for layer in scene.to_dict()["layers"])

    del profile["localizations"]["video.analysis.proof_scope.unproven"]
    with pytest.raises(ValueError, match="proof-scope copy is incomplete"):
        build_scenario_decision_scene_package(**kwargs)
    for key in list(profile["localizations"]):
        if key.startswith("video.analysis.proof_scope."):
            del profile["localizations"][key]
    legacy = build_scenario_decision_scene_package(**kwargs)
    legacy_reveal = legacy.scene("reveal").to_dict()
    assert "proofScopeKey" not in next(
        layer["content"] for layer in legacy_reveal["layers"] if layer["id"] == "analysis"
    )


def test_exact_margin_render_uses_governed_result_values(root: Path, rules, monkeypatch) -> None:
    normalized = _normalized_state(rules)
    profile = yaml.safe_load(
        (root / "configs" / "video" / "premium.yaml").read_text(encoding="utf-8")
    )
    proof = _proof(normalized).to_dict()
    selected = proof["selectedAction"]
    proof["utility"] = "awarded_point_margin"
    proof["winningRootActionCount"] = 1
    for root_action in proof["rootActions"]:
        root_action["value"] = 14 if root_action["action"] == selected else 10
        root_action["satisfies"] = root_action["action"] == selected
    monkeypatch.setattr(
        "baloot_platform.video.remotion_renderer.build_render_bundle",
        lambda **kwargs: SimpleNamespace(
            value={
                "localizations": kwargs["localizations"],
                "scenes": [scene.to_dict() for scene in kwargs["scenes"]],
            }
        ),
    )
    renderer = RemotionProfessionalRenderer(profile, repository_root=root)
    bundle = renderer.build_scenario_render_request(
        exact_state=normalized.to_dict(),
        proof_outcome=proof,
        scenario_projection=_projection(normalized),
        effective_config_sha256="f" * 64,
        game_serial="TEST-MARGIN",
        observer_seat="south",
        enrollment_uri="https://example.invalid/margin",
        source_metadata={},
        rules=rules,
    )
    assert bundle.value["localizations"]["video.analysis.unique_awarded_margin"] == (
        "في هالتوزيعة، الفارق المثبت 14 نقطة؛ أقرب بديل 10."
    )
    reveal = next(scene for scene in bundle.value["scenes"] if scene["stage"] == "reveal")
    analysis = next(layer["content"] for layer in reveal["layers"] if layer["id"] == "analysis")
    assert analysis["reasonKey"] == "video.analysis.unique_awarded_margin"
    assert analysis["reasonValues"] == {"selectedMargin": 14, "nextBestMargin": 10}
    assert "{" not in bundle.value["localizations"][analysis["reasonKey"]]

    legacy_profile = copy.deepcopy(profile)
    del legacy_profile["localizations"]["video.analysis.unique_awarded_margin"]
    legacy_bundle = RemotionProfessionalRenderer(
        legacy_profile, repository_root=root
    ).build_scenario_render_request(
        exact_state=normalized.to_dict(),
        proof_outcome=proof,
        scenario_projection=_projection(normalized),
        effective_config_sha256="f" * 64,
        game_serial="TEST-MARGIN",
        observer_seat="south",
        enrollment_uri="https://example.invalid/margin",
        source_metadata={},
        rules=rules,
    )
    legacy_reveal = next(
        scene for scene in legacy_bundle.value["scenes"] if scene["stage"] == "reveal"
    )
    legacy_analysis = next(
        layer["content"]
        for layer in legacy_reveal["layers"]
        if layer["id"] == "analysis"
    )
    assert legacy_analysis["reasonKey"] == "video.analysis.one_winning_action"
    assert "reasonValues" not in legacy_analysis


def test_render_lifecycle_uses_only_the_sanitized_project_projection(
    root: Path,
    rules,
) -> None:
    normalized = _normalized_state(rules)
    exact_state = normalized.to_dict()
    exact_state["state"]["events"].append(_hidden_project_event(normalized))
    _resign_exact_state(exact_state)
    profile = yaml.safe_load(
        (root / "configs" / "video" / "premium.yaml").read_text(encoding="utf-8")
    )
    package = build_scenario_decision_scene_package(
        exact_state=exact_state,
        proof_outcome=_proof(normalized, exact_state=exact_state).to_dict(),
        scenario_projection=_exact_projection(exact_state),
        profile=profile,
        effective_config_sha256="f" * 64,
        observer_seat="south",
        rules=rules,
    )
    context = next(
        scene for scene in package.scenes if scene.value["stage"] == "context"
    )
    lifecycle = next(
        layer["content"]
        for layer in context.value["layers"]
        if layer["id"] == "public-lifecycle"
    )
    assert {project["owner"] for project in lifecycle["projects"]} == {"south"}
    serialized = json.dumps(
        [scene.to_dict() for scene in package.scenes],
        sort_keys=True,
        separators=(",", ":"),
    )
    assert '"type":"projects_revealed"' not in serialized
    assert '"owner":"north","projects"' not in serialized


def test_bid_projection_and_render_reject_injected_hidden_metadata(
    root: Path,
    rules,
) -> None:
    normalized = _normalized_bid_state(rules)
    valid, _store = _validated_result(normalized)
    assert valid["qualification"]["observerSafe"] is True, valid["qualification"][
        "observerSafetyEvidence"
    ]

    state = copy.deepcopy(dict(normalized.state))
    state["bidding"]["hiddenCards"] = copy.deepcopy(state["hands"]["north"][:2])
    corrupted = replace(
        normalized,
        state=state,
        state_hash=canonical_sha256(state),
    )
    result, _store = _validated_result(corrupted)
    evidence = result["qualification"]["observerSafetyEvidence"]
    assert evidence["passed"] is False
    assert result["qualification"]["observerSafe"] is False
    assert result["qualification"]["qualified"] is False

    profile = yaml.safe_load(
        (root / "configs" / "video" / "premium.yaml").read_text(encoding="utf-8")
    )
    exact_state = corrupted.to_dict()
    _resign_exact_state(exact_state)
    with pytest.raises(ValueError, match="bidding projection is not closed"):
        build_scenario_decision_scene_package(
            exact_state=exact_state,
            proof_outcome=_proof(corrupted, exact_state=exact_state).to_dict(),
            scenario_projection=_exact_projection(exact_state),
            profile=profile,
            effective_config_sha256="f" * 64,
            observer_seat=str(corrupted.next_actor),
            rules=rules,
        )


def test_bid_render_preserves_the_five_card_pre_contract_state(
    root: Path,
    rules,
) -> None:
    normalized = _normalized_bid_state(rules)
    profile, package, bundle = _nonplay_production_bundle(
        root,
        rules,
        normalized,
        match_score=(0, 0),
    )
    context = package.scene("context")
    context_layers = {layer["id"]: layer["content"] for layer in context.value["layers"]}
    assert context_layers["public-lifecycle"]["buyer"] is None
    assert context_layers["public-lifecycle"]["contract"] is None
    assert context_layers["public-lifecycle"]["plays"] == []
    assert "roundScore" not in context_layers["public-lifecycle"]
    assert "dealSequence" not in context_layers["public-lifecycle"]
    assert len(context_layers["observer-lifecycle"]["initialHandCards"]) == 5
    context = package.scene("continuation")
    lifecycle = next(
        layer["content"]
        for layer in context.value["layers"]
        if layer["id"] == "public-lifecycle"
    )
    observer = next(
        layer["content"]
        for layer in context.value["layers"]
        if layer["id"] == "observer-lifecycle"
    )
    assert lifecycle["buyer"] == "north"
    assert lifecycle["contract"] == "sun"
    assert lifecycle["roundScore"]["awardedPoints"] == [0, 38]
    assert len(lifecycle["plays"]) == 32
    assert len(lifecycle["completedTricks"]) == 8
    assert len(observer["initialHandCards"]) == 8

    decision_codes = None
    terminal_codes = {card["code"] for card in observer["initialHandCards"]}
    for stage in ("context", "question", "reveal"):
        scene = next(item for item in package.scenes if item.value["stage"] == stage)
        table = next(
            layer["content"] for layer in scene.value["layers"] if layer["id"] == "table"
        )
        hand = next(
            layer["content"]
            for layer in scene.value["layers"]
            if layer["id"] == "observer-hand"
        )
        counts = next(
            layer["content"]["handCounts"]
            for layer in scene.value["layers"]
            if layer["id"] == "seats"
        )
        assert table["phase"] == "bidding"
        assert table["buyer"] is None
        assert table["contract"] is None
        assert table["trumpSuit"] is None
        assert table["roundScore"] is None
        assert table["completedTricks"] == []
        assert table["projects"] == []
        assert counts == {seat: 5 for seat in ("east", "north", "south", "west")}
        stage_codes = {card["code"] for card in hand["cards"]}
        assert len(stage_codes) == 5
        decision_codes = decision_codes or stage_codes
        assert stage_codes == decision_codes
    assert terminal_codes - decision_codes

    timeline = bundle.value["timeline"]["segments"]
    assert [segment["stage"] for segment in timeline] == [
        "hook",
        "rewind",
        "context",
        "question",
        "reveal",
        "continuation",
        "outcome",
    ]
    assert timeline[2]["durationFrames"] == round(
        profile["timeline"]["bidding_seconds"] * profile["render"]["fps"]
    )
    assert timeline[5]["durationFrames"] > round(
        profile["exports"]["hooked_flagship"]["continuation_seconds"]
        * profile["render"]["fps"]
    )
    assert bundle.verify()

    compressed = bundle.to_dict()
    compressed_timeline = compressed["timeline"]
    compressed_segments = compressed_timeline["segments"]
    compressed_segments[2]["durationFrames"] -= 1
    compressed_segments[5]["durationFrames"] += 1
    frame = 0
    for segment in compressed_segments:
        segment["startFrame"] = frame
        segment["startTime"]["numerator"] = frame
        segment["durationTime"]["numerator"] = segment["durationFrames"]
        frame += segment["durationFrames"]
    compressed_timeline["totalFrames"] = frame
    timeline_core = dict(compressed_timeline)
    timeline_core.pop("timelineSha256")
    compressed_timeline["timelineSha256"] = content_hash(timeline_core)
    bundle_core = dict(compressed)
    bundle_core.pop("bundleSha256")
    compressed["bundleSha256"] = content_hash(bundle_core)
    assert not RenderBundleV1(compressed).verify()

    _profile, _package, decision_clip = _nonplay_production_bundle(
        root,
        rules,
        normalized,
        match_score=(0, 0),
        export_preset="decision_clip",
    )
    assert decision_clip.verify()
    assert not {
        event["kind"] for event in build_audio_cue_plan(decision_clip)["events"]
    } & {"card", "collect"}


@pytest.mark.parametrize(
    ("normalized_factory", "match_score", "expected_phase", "phase_key"),
    [
        (_normalized_escalation_state, (101, 0), "final_dealt", "escalation_seconds"),
        (_normalized_project_state, (0, 0), "playing", "project_seconds"),
    ],
)
def test_escalation_and_project_production_bundles_keep_decision_scene_state(
    root: Path,
    rules,
    normalized_factory,
    match_score,
    expected_phase: str,
    phase_key: str,
) -> None:
    normalized = normalized_factory(rules)
    profile, package, bundle = _nonplay_production_bundle(
        root,
        rules,
        normalized,
        match_score=match_score,
    )
    question = next(
        scene for scene in package.scenes if scene.value["stage"] == "question"
    )
    table = next(
        layer["content"]
        for layer in question.value["layers"]
        if layer["id"] == "table"
    )
    hand = next(
        layer["content"]
        for layer in question.value["layers"]
        if layer["id"] == "observer-hand"
    )
    counts = next(
        layer["content"]["handCounts"]
        for layer in question.value["layers"]
        if layer["id"] == "seats"
    )
    assert table["phase"] == expected_phase
    assert table["buyer"] == "north"
    assert table["contract"] == "sun"
    assert table["roundScore"] is None
    assert table["completedTricks"] == []
    assert len(hand["cards"]) == 8
    assert counts == {seat: 8 for seat in ("east", "north", "south", "west")}
    if normalized.decision_kind == "project":
        assert table["projects"]
    else:
        assert table["projects"] == []

    context = next(
        scene for scene in package.scenes if scene.value["stage"] == "continuation"
    )
    lifecycle = next(
        layer["content"]
        for layer in context.value["layers"]
        if layer["id"] == "public-lifecycle"
    )
    assert len(lifecycle["plays"]) == 32
    assert len(lifecycle["completedTricks"]) == 8
    assert lifecycle["roundScore"] is not None

    timeline = bundle.value["timeline"]["segments"]
    assert timeline[2]["stage"] == "context"
    assert timeline[2]["durationFrames"] == round(
        profile["timeline"][phase_key] * profile["render"]["fps"]
    )
    assert timeline[5]["durationFrames"] > round(
        profile["exports"]["hooked_flagship"]["continuation_seconds"]
        * profile["render"]["fps"]
    )
    assert bundle.verify()


def test_legal_action_projection_and_render_reject_private_extra_fields(
    root: Path,
    rules,
) -> None:
    normalized = _normalized_state(rules)
    legal_actions = [copy.deepcopy(dict(item)) for item in normalized.legal_actions]
    legal_actions[0]["hiddenCards"] = copy.deepcopy(
        dict(normalized.state)["hands"]["north"][:2]
    )
    corrupted = replace(normalized, legal_actions=tuple(legal_actions))
    result, _store = _validated_result(corrupted)
    evidence = result["qualification"]["observerSafetyEvidence"]
    assert evidence["passed"] is False
    assert result["qualification"]["observerSafe"] is False
    assert result["qualification"]["qualified"] is False

    profile = yaml.safe_load(
        (root / "configs" / "video" / "premium.yaml").read_text(encoding="utf-8")
    )
    exact_state = corrupted.to_dict()
    _resign_exact_state(exact_state)
    with pytest.raises(ValueError, match="legal action is not closed"):
        build_scenario_decision_scene_package(
            exact_state=exact_state,
            proof_outcome=_proof(corrupted, exact_state=exact_state).to_dict(),
            scenario_projection=_exact_projection(exact_state),
            profile=profile,
            effective_config_sha256="f" * 64,
            observer_seat="south",
            rules=rules,
        )


@pytest.mark.parametrize("injection", ["observer_hand", "public_event"])
def test_nested_public_card_metadata_is_rejected_before_render(
    root: Path,
    rules,
    injection: str,
) -> None:
    normalized = _normalized_state(rules)
    exact_state = normalized.to_dict()
    hidden_cards = copy.deepcopy(exact_state["state"]["hands"]["north"][:2])
    if injection == "observer_hand":
        exact_state["state"]["hands"]["south"][0]["hiddenCards"] = hidden_cards
    else:
        card = copy.deepcopy(exact_state["state"]["hands"]["south"][0])
        card["hiddenCards"] = hidden_cards
        exact_state["state"]["events"].append(
            {
                "sequence": len(exact_state["state"]["events"]),
                "type": "card_played",
                "seat": "south",
                "card": card,
                "trickIndex": 0,
                "position": 0,
                "ekka": False,
                "baloot": False,
            }
        )
    _resign_exact_state(exact_state)
    with pytest.raises(ValueError, match="card is invalid"):
        build_observer_safe_exact_state_v1(
            exact_state=exact_state,
            scenario_projection=_exact_projection(exact_state),
            observer_seat="south",
        )

    profile = yaml.safe_load(
        (root / "configs" / "video" / "premium.yaml").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="card is invalid"):
        build_scenario_decision_scene_package(
            exact_state=exact_state,
            proof_outcome=_proof(normalized, exact_state=exact_state).to_dict(),
            scenario_projection=_exact_projection(exact_state),
            profile=profile,
            effective_config_sha256="f" * 64,
            observer_seat="south",
            rules=rules,
        )


def test_resigned_exact_state_cannot_swap_the_acting_observer(rules) -> None:
    normalized = _normalized_state(rules)
    exact_state = normalized.to_dict()
    exact_state["nextActor"] = "north"
    exact_state["legalActions"] = [
        LegalPlayActionV1(Card.parse(card)).to_dict()
        for card in exact_state["state"]["hands"]["north"]
    ]
    _resign_exact_state(exact_state)

    with pytest.raises(ValueError, match="play actor mismatch"):
        build_observer_safe_exact_state_v1(
            exact_state=exact_state,
            scenario_projection=_exact_projection(exact_state),
            observer_seat="north",
        )


def test_scenario_qualification_fails_closed_when_projection_evidence_mismatches(
    rules,
) -> None:
    normalized = _normalized_state(rules)
    scenario = SimpleNamespace(
        origin=SimpleNamespace(kind="exact_trace_prefix"),
        sha256="c" * 64,
        decision_point={"kind": "play"},
        predicates=lambda: (),
    )
    recipe = SimpleNamespace(
        code="RCP-OBSERVER-SAFETY",
        sha256="d" * 64,
        configuration={"bundleRevisionId": "CBR-TEST"},
        scenario=scenario,
        resources={"generationLane": "exact_input"},
    )
    satisfaction = SimpleNamespace(
        status="satisfied",
        sha256="e" * 64,
        to_dict=lambda: {"sha256": "e" * 64},
    )
    artifacts = {
        key: {"sha256": f"{index:x}" * 64}
        for index, key in enumerate(SCENARIO_ARTIFACT_MEDIA_TYPES, start=1)
    }
    valid = build_scenario_result_v4(
        run={"runCode": "RUN-OBSERVER-SAFETY", "runHash": "9" * 64},
        recipe=recipe,
        normalized=normalized,
        satisfaction=satisfaction,
        proof=_proof(normalized),
        constraint_evidence={"sha256": "8" * 64},
        artifacts=artifacts,
        created_at="2026-08-23T00:00:00+00:00",
    )
    assert valid["qualification"]["observerSafe"] is True
    assert valid["qualification"]["qualified"] is True

    corrupted_state = copy.deepcopy(dict(normalized.state))
    corrupted_state["hands"]["north"].pop()
    corrupted = replace(
        normalized,
        state=corrupted_state,
        state_hash=canonical_sha256(corrupted_state),
    )
    result = build_scenario_result_v4(
        run={"runCode": "RUN-OBSERVER-SAFETY", "runHash": "9" * 64},
        recipe=recipe,
        normalized=corrupted,
        satisfaction=satisfaction,
        proof=_proof(corrupted),
        constraint_evidence={"sha256": "8" * 64},
        artifacts=artifacts,
        created_at="2026-08-23T00:00:00+00:00",
    )
    qualification = result["qualification"]
    evidence = qualification["observerSafetyEvidence"]
    assert qualification["observerSafe"] is False
    assert qualification["qualified"] is False
    assert result["disposition"] == "rejected"
    assert "observer_safety_unverified" in qualification["failures"]
    assert evidence["passed"] is False
    assert evidence["checks"]["handsMatchEngineProjection"] is False
    assert (
        result["provenance"]["resultIdentity"]["observerSafetyEvidenceHash"]
        == (evidence["sha256"])
    )


def test_scenario_qualification_rejects_incoherent_completed_proof(rules) -> None:
    normalized = _normalized_state(rules)
    scenario = SimpleNamespace(
        origin=SimpleNamespace(kind="exact_trace_prefix"),
        sha256="c" * 64,
        decision_point={"kind": "play"},
        predicates=lambda: (),
    )
    recipe = SimpleNamespace(
        code="RCP-PROOF-SEMANTICS",
        sha256="d" * 64,
        configuration={"bundleRevisionId": "CBR-TEST"},
        scenario=scenario,
        resources={"generationLane": "exact_input"},
    )
    satisfaction = SimpleNamespace(
        status="satisfied",
        sha256="e" * 64,
        to_dict=lambda: {"sha256": "e" * 64},
    )
    proof = _proof(normalized)
    proof.payload["winningRootActionCount"] = 1
    proof.payload["rootActions"] = [
        {
            "action": proof.payload["rootActions"][0]["action"],
            "value": -1.0,
            "satisfies": False,
        }
    ]
    proof.payload["selectedAction"] = "NOT-LEGAL"
    result = build_scenario_result_v4(
        run={"runCode": "RUN-PROOF-SEMANTICS", "runHash": "9" * 64},
        recipe=recipe,
        normalized=normalized,
        satisfaction=satisfaction,
        proof=proof,
        constraint_evidence={"sha256": "8" * 64},
        artifacts={
            key: {"sha256": f"{index:x}" * 64}
            for index, key in enumerate(SCENARIO_ARTIFACT_MEDIA_TYPES, start=1)
        },
        created_at="2026-08-23T00:00:00+00:00",
    )
    assert result["qualification"]["qualified"] is False
    assert result["truthLabel"] == "rejected"
    assert result["confidence"] == 0.0
    assert result["disposition"] == "rejected"
    assert "proof_semantics_incoherent" in result["qualification"]["failures"]


def test_scenario_result_validator_binds_observer_and_score_evidence(rules) -> None:
    result, store = _validated_result(_normalized_state(rules))
    validated = validate_scenario_result_v4(result, artifact_store=store)
    assert validated["proofOutcome"] == public_proof_outcome_v1(result["proofOutcome"])
    assert (
        validated["proofOutcome"]["artifactSha256"]
        == result["artifacts"]["proofOutcome"]["sha256"]
    )
    public_core = {
        key: value
        for key, value in validated["proofOutcome"].items()
        if key != "sha256"
    }
    assert validated["proofOutcome"]["sha256"] == canonical_sha256(public_core)
    assert validate_scenario_result_v4(validated, artifact_store=store) == validated

    corrupted_score = copy.deepcopy(result)
    corrupted_score["scoreEvidence"]["score"] = 0.0
    with pytest.raises(ValueError, match="score evidence checksum mismatch"):
        validate_scenario_result_v4(corrupted_score, artifact_store=store)

    corrupted_observer = copy.deepcopy(result)
    evidence = corrupted_observer["qualification"]["observerSafetyEvidence"]
    evidence["checks"]["eventsMatchEngineProjection"] = False
    evidence_core = {key: value for key, value in evidence.items() if key != "sha256"}
    evidence["sha256"] = canonical_sha256(evidence_core)
    with pytest.raises(ValueError, match="observer-safety verdict is invalid"):
        validate_scenario_result_v4(corrupted_observer, artifact_store=store)


def test_public_result_derives_bounds_for_immutable_pre_bound_results(rules) -> None:
    result, _store = _validated_result(_normalized_state(rules))
    result["proofOutcome"].pop("winningRootActionLowerBound")
    result["proofOutcome"].pop("winningRootActionUpperBound")
    result["scenarioProjection"].pop("winningRootActionLowerBound")
    result["scenarioProjection"].pop("winningRootActionUpperBound")

    public = public_scenario_result_v4(result)
    expected = public["proofOutcome"]["winningRootActionCount"]
    assert public["proofOutcome"]["winningRootActionLowerBound"] == expected
    assert public["proofOutcome"]["winningRootActionUpperBound"] == expected
    assert public["scenarioProjection"]["winningRootActionLowerBound"] == expected
    assert public["scenarioProjection"]["winningRootActionUpperBound"] == expected


def test_scenario_result_validator_derives_resigned_verdict_and_projection(
    rules,
) -> None:
    result, store = _validated_result(_normalized_state(rules))

    tampered_verdict = copy.deepcopy(result)
    tampered_verdict["score"] = 0.0
    tampered_verdict["truthLabel"] = "rejected"
    tampered_verdict["confidence"] = 0.0
    tampered_verdict["disposition"] = "rejected"
    qualification = tampered_verdict["qualification"]
    qualification["qualified"] = False
    qualification["failures"] = ["manually_rejected"]
    score_evidence = tampered_verdict["scoreEvidence"]
    score_evidence["score"] = 0.0
    score_core = {
        key: value for key, value in score_evidence.items() if key != "sha256"
    }
    score_evidence["sha256"] = canonical_sha256(score_core)
    observer_hash = qualification["observerSafetyEvidence"]["sha256"]
    verdict = {
        "schemaVersion": 1,
        "score": 0.0,
        "truthLabel": "rejected",
        "confidence": 0.0,
        "disposition": "rejected",
        "qualification": {
            "schemaVersion": 1,
            "qualified": False,
            "failures": ["manually_rejected"],
            "scenarioSatisfactionHash": qualification["scenarioSatisfactionHash"],
            "proofOutcomeHash": qualification["proofOutcomeHash"],
            "observerSafe": True,
            "observerSafetyEvidenceHash": observer_hash,
            "decisionPointAuditComplete": True,
        },
    }
    identity = tampered_verdict["provenance"]["resultIdentity"]
    identity["scoreEvidenceHash"] = score_evidence["sha256"]
    identity["verdictHash"] = canonical_sha256(verdict)
    tampered_verdict["resultHash"] = canonical_sha256(identity)
    tampered_verdict["resultCode"] = (
        f"LAF-{tampered_verdict['resultHash'][:10].upper()}"
    )
    with pytest.raises(ValueError, match="qualification verdict mismatch"):
        validate_scenario_result_v4(tampered_verdict, artifact_store=store)

    tampered_projection = copy.deepcopy(result)
    projection = tampered_projection["scenarioProjection"]
    projection["targetTeam"] = "west_east"
    projection_core = {
        key: value for key, value in projection.items() if key != "sha256"
    }
    projection["sha256"] = canonical_sha256(projection_core)
    identity = tampered_projection["provenance"]["resultIdentity"]
    identity["scenarioProjectionHash"] = projection["sha256"]
    tampered_projection["resultHash"] = canonical_sha256(identity)
    tampered_projection["resultCode"] = (
        f"LAF-{tampered_projection['resultHash'][:10].upper()}"
    )
    with pytest.raises(ValueError, match="scenario projection authority mismatch"):
        validate_scenario_result_v4(tampered_projection, artifact_store=store)

    tampered_score_semantics = copy.deepcopy(result)
    score_evidence = tampered_score_semantics["scoreEvidence"]
    score_evidence["family"] = "structural_objective"
    score_core = {
        key: value for key, value in score_evidence.items() if key != "sha256"
    }
    score_evidence["sha256"] = canonical_sha256(score_core)
    identity = tampered_score_semantics["provenance"]["resultIdentity"]
    identity["scoreEvidenceHash"] = score_evidence["sha256"]
    tampered_score_semantics["resultHash"] = canonical_sha256(identity)
    tampered_score_semantics["resultCode"] = (
        f"LAF-{tampered_score_semantics['resultHash'][:10].upper()}"
    )
    with pytest.raises(ValueError, match="score evidence semantics mismatch"):
        validate_scenario_result_v4(tampered_score_semantics, artifact_store=store)

    extended_projection = copy.deepcopy(result)
    extended_projection["scenarioProjection"]["hiddenCards"] = [0, 1]
    projection = extended_projection["scenarioProjection"]
    projection_core = {
        key: value for key, value in projection.items() if key != "sha256"
    }
    projection["sha256"] = canonical_sha256(projection_core)
    identity = extended_projection["provenance"]["resultIdentity"]
    identity["scenarioProjectionHash"] = projection["sha256"]
    extended_projection["resultHash"] = canonical_sha256(identity)
    extended_projection["resultCode"] = (
        f"LAF-{extended_projection['resultHash'][:10].upper()}"
    )
    with pytest.raises(ValueError, match="scenario projection is not closed"):
        validate_scenario_result_v4(extended_projection, artifact_store=store)


def test_audit_only_world_witness_is_retained_in_artifact_but_never_public(
    root: Path,
    rules,
) -> None:
    normalized = _normalized_state(rules)
    proof_value = _proof_outcome(normalized).to_dict()
    proof_value.pop("sha256")
    proof_value.pop("semanticSha256", None)
    proof_value["rootActions"][0]["worstWorldWitnessHash"] = "c" * 64
    proof_value["rootActions"][0]["worstWorldWitness"] = {
        "visibility": "audit_only",
        "worldHash": "d" * 64,
        "actions": ["PRIVATE-WORLD-ACTION"],
    }
    proof = ProofOutcomeV1.build(proof_value)
    result, store = _validated_result(normalized, proof_override=proof)
    proof_uri = result["artifacts"]["proofOutcome"]["uri"]
    retained = json.loads(store.get(proof_uri).decode("utf-8"))
    assert retained["rootActions"][0]["worstWorldWitness"]["worldHash"] == ("d" * 64)

    public_result = validate_scenario_result_v4(result, artifact_store=store)
    public_root = public_result["proofOutcome"]["rootActions"][0]
    assert "worstWorldWitness" not in public_root
    assert "worstWorldWitnessHash" not in public_root
    assert "principalVariation" not in public_result["proofOutcome"]
    audit_root = scenario_decision_audit_v1(result)[0]["alternatives"][0]
    assert "worstWorldWitness" not in audit_root

    retained["privateOpponentCards"] = ["PRIVATE-TOP-LEVEL"]
    retained["rootActions"][0]["privateOpponentCards"] = ["PRIVATE-ROOT"]
    retained["resources"]["privateOpponentCards"] = ["PRIVATE-RESOURCE"]
    retained["continuationLines"]["privateOpponentCards"] = ["PRIVATE-CONTINUATION"]

    profile = yaml.safe_load(
        (root / "configs" / "video" / "premium.yaml").read_text(encoding="utf-8")
    )
    package = build_scenario_decision_scene_package(
        exact_state=normalized.to_dict(),
        proof_outcome=retained,
        scenario_projection=_projection(normalized),
        profile=profile,
        effective_config_sha256="f" * 64,
        observer_seat="south",
        rules=rules,
    )
    encoded = json.dumps(
        [scene.to_dict() for scene in package.scenes],
        sort_keys=True,
        separators=(",", ":"),
    )
    assert "PRIVATE-WORLD-ACTION" not in encoded
    assert "PRIVATE-TOP-LEVEL" not in encoded
    assert "PRIVATE-ROOT" not in encoded
    assert "PRIVATE-RESOURCE" not in encoded
    assert "PRIVATE-CONTINUATION" not in encoded
    assert '"worstWorldWitness":' not in encoded


def test_scenario_qualification_rejects_noncanonical_public_project_event(
    rules,
) -> None:
    normalized = _normalized_state(rules)
    state = copy.deepcopy(dict(normalized.state))
    state["events"].append(
        {
            "sequence": len(state["events"]),
            "type": "projects_compared",
            "winningTeam": "north_south",
            "declarations": [],
        }
    )
    injected = replace(normalized, state=state)
    result, _store = _validated_result(injected)
    evidence = result["qualification"]["observerSafetyEvidence"]
    assert evidence["checks"]["eventsMatchEngineProjection"] is False
    assert result["qualification"]["observerSafe"] is False
    assert result["qualification"]["qualified"] is False
    assert result["disposition"] == "rejected"


def test_legacy_v4_result_remains_readable_but_fails_observer_safety_closed(
    rules,
) -> None:
    result, store = _validated_result(_normalized_state(rules))
    legacy = copy.deepcopy(result)
    legacy["qualification"].pop("observerSafetyEvidence")
    legacy["qualification"]["observerSafe"] = True
    identity = legacy["provenance"]["resultIdentity"]
    identity.pop("observerSafetyEvidenceHash")
    identity.pop("scenarioProjectionHash")
    identity.pop("scoreEvidenceHash")
    identity.pop("verdictHash")
    legacy["scenarioProjection"]["publicProjectionHash"] = None
    legacy["scenarioProjection"]["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in legacy["scenarioProjection"].items()
            if key != "sha256"
        }
    )
    legacy["resultHash"] = canonical_sha256(identity)
    legacy["resultCode"] = f"LAF-{legacy['resultHash'][:10].upper()}"
    legacy["scoreEvidence"].pop("observerSafetyEvidenceHash")
    score_core = {
        key: value for key, value in legacy["scoreEvidence"].items() if key != "sha256"
    }
    legacy["scoreEvidence"]["sha256"] = canonical_sha256(score_core)

    validated = validate_scenario_result_v4(legacy, artifact_store=store)
    assert validated["qualification"]["observerSafe"] is False
    assert validated["qualification"]["qualified"] is False
    assert validated["qualification"]["observerSafetyEvidence"] is None
    assert validated["qualification"]["legacyObserverSafetyUnverified"] is True
    assert "observer_safety_unverified" in validated["qualification"]["failures"]
    assert validated["disposition"] == "rejected"
    assert "observerSafetyEvidence" not in legacy["qualification"]
    assert len(validated["scenarioProjection"]["publicProjectionHash"]) == 64


@pytest.mark.parametrize("invalid_id", [True, 1.0])
def test_playable_view_rejects_coercive_card_identifiers(
    rules,
    invalid_id: object,
) -> None:
    exact_state = _normalized_state(rules).to_dict()
    exact_state["state"]["hands"]["south"][0]["id"] = invalid_id
    _resign_exact_state(exact_state)

    with pytest.raises(ValueError, match="card is invalid"):
        build_observer_safe_exact_state_v1(
            exact_state=exact_state,
            scenario_projection=_exact_projection(exact_state),
            observer_seat="south",
        )


def test_playable_view_rejects_an_opponent_card_as_a_legal_action(rules) -> None:
    exact_state = _normalized_state(rules).to_dict()
    opponent_card = Card.parse(exact_state["state"]["hands"]["north"][0])
    exact_state["legalActions"][0] = LegalPlayActionV1(opponent_card).to_dict()
    _resign_exact_state(exact_state)

    with pytest.raises(ValueError, match="outside actor hand"):
        build_observer_safe_exact_state_v1(
            exact_state=exact_state,
            scenario_projection=_exact_projection(exact_state),
            observer_seat="south",
        )


def test_playable_view_rejects_a_forged_card_play_event(rules) -> None:
    exact_state = _normalized_state(rules).to_dict()
    exact_state["state"]["events"].append(
        {
            "sequence": len(exact_state["state"]["events"]),
            "type": "card_played",
            "seat": "west",
            "card": copy.deepcopy(exact_state["state"]["hands"]["west"][0]),
            "trickIndex": 0,
            "position": 0,
            "ekka": False,
            "baloot": False,
        }
    )
    _resign_exact_state(exact_state)

    with pytest.raises(ValueError, match="lacks a played-card owner"):
        build_observer_safe_exact_state_v1(
            exact_state=exact_state,
            scenario_projection=_exact_projection(exact_state),
            observer_seat="south",
        )


def test_indistinguishable_hidden_worlds_share_only_the_public_identifier(
    rules,
) -> None:
    first = _normalized_state(rules).to_dict()
    second = copy.deepcopy(first)
    for key in ("hands", "initialHands"):
        north = second["state"][key]["north"][0]
        west = second["state"][key]["west"][0]
        second["state"][key]["north"][0] = west
        second["state"][key]["west"][0] = north
    _resign_exact_state(second)
    assert first["exactStateId"] != second["exactStateId"]
    reference = {"resultCode": "LAF-PUBLIC0001", "decisionOrdinal": 1}
    first_view = build_observer_safe_exact_state_v1(
        exact_state=first,
        scenario_projection=_exact_projection(first),
        observer_seat="south",
        decision_reference=reference,
    )
    second_view = build_observer_safe_exact_state_v1(
        exact_state=second,
        scenario_projection=_exact_projection(second),
        observer_seat="south",
        decision_reference=reference,
    )

    assert first_view == second_view
    assert first_view["decisionReference"] == reference
    assert "exactStateId" not in first_view
    assert "stateHash" not in first_view
    assert first_view["publicProjectionHash"]
    assert first["exactStateId"].startswith("STATE-")
    assert second["exactStateId"].startswith("STATE-")


def test_actor_public_result_masks_all_full_state_identifiers(rules) -> None:
    normalized = _normalized_state(rules)
    result, store = _validated_result(normalized)
    internal = validate_scenario_result_v4(result, artifact_store=store)
    public = public_scenario_result_v4(internal)
    encoded = json.dumps(public, sort_keys=True, separators=(",", ":"))

    assert public["decisionReference"] == {
        "resultCode": result["resultCode"],
        "decisionOrdinal": 1,
    }
    assert (
        public["publicProjectionHash"]
        == result["scenarioProjection"]["publicProjectionHash"]
    )
    assert public["scenarioProjection"]["predicateEvidenceCount"] == len(
        internal["scenarioSatisfaction"]["predicates"]
    )
    assert "resultHash" not in public
    assert "traceHash" not in public
    assert "gameHash" not in public
    assert "artifacts" not in public
    assert "scenarioSatisfaction" not in public
    assert "provenance" not in public
    assert "principalVariation" not in public["proofOutcome"]
    assert "artifactSha256" not in public["proofOutcome"]
    assert "semanticSha256" not in public["proofOutcome"]
    assert normalized.exact_state_id not in encoded
    assert normalized.state_hash not in encoded
    assert result["artifacts"]["proofOutcome"]["sha256"] not in encoded
    assert internal["scenarioProjection"]["exactStateId"] == normalized.exact_state_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "forged_observer_evidence"),
        ("projectionSha256", "z" * 64),
        ("failureCode", "unexpected_failure"),
    ],
)
def test_observer_evidence_semantics_are_fail_closed(
    rules,
    field: str,
    value: object,
) -> None:
    result, store = _validated_result(_normalized_state(rules))
    corrupted = copy.deepcopy(result)
    evidence = corrupted["qualification"]["observerSafetyEvidence"]
    evidence[field] = value
    evidence["sha256"] = canonical_sha256(
        {key: item for key, item in evidence.items() if key != "sha256"}
    )

    with pytest.raises(ValueError, match="observer-safety evidence semantics"):
        validate_scenario_result_v4(corrupted, artifact_store=store)


def test_ekka_actions_survive_observer_and_video_projection(
    root: Path,
    rules,
) -> None:
    normalized = ExactStateNormalizerV1(rules).normalize(_ekka_decision_prefix())
    internal_projection = _projection(normalized)
    view = build_observer_safe_exact_state_v1(
        exact_state=normalized.to_dict(),
        scenario_projection=internal_projection,
        observer_seat="north",
    )
    assert [item["code"] for item in view["legalActions"]] == [
        "SK",
        "C8",
        "SK+ekka",
    ]
    assert view["legalActions"][2] == {
        "actionId": 38,
        "code": "SK+ekka",
        "card": Card.parse("SK").to_dict(),
        "ekka": True,
    }

    public_projection = {
        "decisionKind": "play",
        "publicProjectionHash": view["publicProjectionHash"],
        "decisionReference": {
            "resultCode": "LAF-EKKA000001",
            "decisionOrdinal": 1,
        },
    }
    profile = yaml.safe_load(
        (root / "configs" / "video" / "premium.yaml").read_text(encoding="utf-8")
    )
    package = build_scenario_decision_scene_package(
        exact_state=normalized.to_dict(),
        proof_outcome=_proof(normalized).to_dict(),
        scenario_projection=public_projection,
        profile=profile,
        effective_config_sha256="f" * 64,
        observer_seat="north",
        rules=rules,
    )
    encoded = json.dumps(
        [scene.to_dict() for scene in package.scenes],
        sort_keys=True,
        separators=(",", ":"),
    )
    assert '"code":"SK+ekka"' in encoded
    assert '"ekka":true' in encoded
    assert normalized.exact_state_id not in encoded
    assert normalized.state_hash not in encoded
    assert '"principalVariation"' not in encoded


@pytest.mark.parametrize("tie_outcome", (None, "buyer_makes", "buyer_loses"))
def test_video_analysis_binds_the_selected_sun_tie_rule(root: Path, rules, tie_outcome) -> None:
    from baloot_platform.core.rules import RuleBook

    configured = copy.deepcopy(rules.data)
    if tie_outcome is not None:
        configured["scoring"]["contract"]["buyer_tie_outcome"] = tie_outcome
    selected_rules = RuleBook(configured)
    normalized = _normalized_state(selected_rules)
    profile = yaml.safe_load((root / "configs/video/premium.yaml").read_text(encoding="utf-8"))
    package = build_scenario_decision_scene_package(
        exact_state=normalized.to_dict(), proof_outcome=_proof(normalized).to_dict(),
        scenario_projection=_projection(normalized), profile=profile,
        effective_config_sha256="f" * 64, observer_seat=normalized.next_actor,
        rules=selected_rules,
    )
    analysis = next(layer["content"] for layer in package.scene("reveal").value["layers"]
                    if layer["id"] == "analysis")
    assert analysis["sunTieOutcome"] == (tie_outcome or "legacy_raw_units")
    assert analysis["rulesProfileSha256"] == selected_rules.sha256
    assert f"video.scoring.sun_tie.{analysis['sunTieOutcome']}" in profile["localizations"]


def test_video_rejects_a_different_tie_profile_for_saved_state_or_proof(root: Path, rules) -> None:
    from baloot_platform.core.rules import RuleBook

    normalized = _normalized_state(rules)
    configured = copy.deepcopy(rules.data)
    configured["scoring"]["contract"]["buyer_tie_outcome"] = "buyer_loses"
    other_rules = RuleBook(configured)
    profile = yaml.safe_load((root / "configs/video/premium.yaml").read_text(encoding="utf-8"))
    arguments = dict(exact_state=normalized.to_dict(), proof_outcome=_proof(normalized).to_dict(),
                     scenario_projection=_projection(normalized), profile=profile,
                     effective_config_sha256="f" * 64, observer_seat=normalized.next_actor)
    with pytest.raises(ValueError, match="rules differ from the verified exact state"):
        build_scenario_decision_scene_package(**arguments, rules=other_rules)
    arguments["proof_outcome"]["identities"]["rulesHash"] = other_rules.sha256
    with pytest.raises(ValueError, match="rules differ from the verified proof"):
        build_scenario_decision_scene_package(**arguments, rules=rules)


@pytest.mark.parametrize("identity", ("exactStateId", "stateHash"))
def test_video_rejects_proof_substitution_for_another_state(root: Path, rules, identity) -> None:
    normalized = _normalized_state(rules)
    proof = _proof(normalized).to_dict()
    proof["identities"][identity] = "another-state"
    profile = yaml.safe_load((root / "configs/video/premium.yaml").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="proof references a different exact state"):
        build_scenario_decision_scene_package(
            exact_state=normalized.to_dict(), proof_outcome=proof,
            scenario_projection=_projection(normalized), profile=profile,
            effective_config_sha256="f" * 64, observer_seat=normalized.next_actor, rules=rules,
        )


def test_ashkal_exact_view_and_video_remain_supported(root: Path, rules) -> None:
    normalized = ExactStateNormalizerV1(rules).normalize(_ashkal_prefix())
    view = build_observer_safe_exact_state_v1(
        exact_state=normalized.to_dict(),
        scenario_projection=_projection(normalized),
        observer_seat="north",
    )
    assert view["publicState"]["contract"] == "ashkal"
    assert len(view["legalActions"]) == 8

    profile = yaml.safe_load(
        (root / "configs" / "video" / "premium.yaml").read_text(encoding="utf-8")
    )
    package = build_scenario_decision_scene_package(
        exact_state=normalized.to_dict(),
        proof_outcome=_proof(normalized).to_dict(),
        scenario_projection=_projection(normalized),
        profile=profile,
        effective_config_sha256="f" * 64,
        observer_seat="north",
        rules=rules,
    )
    encoded = json.dumps(
        [scene.to_dict() for scene in package.scenes],
        sort_keys=True,
        separators=(",", ":"),
    )
    assert '"contract":"ashkal"' in encoded
    assert normalized.exact_state_id not in encoded
