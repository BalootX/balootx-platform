from __future__ import annotations

import copy
import json

import pytest

from baloot_platform.core.cards import Card
from baloot_platform.research.exact_state import ExactStateNormalizerV1
from baloot_platform.research.scenario_contracts import ResearchScenarioV1
from baloot_platform.research.scenario_evidence import ScenarioRuntimeVerifierV1


def _ekka_decision_prefix() -> dict:
    """A deterministic Hokum lead where SK and SK+ekka are both legal."""

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


def test_exact_state_exposes_normal_and_ekka_as_distinct_actions(rules):
    normalized = ExactStateNormalizerV1(rules).normalize(_ekka_decision_prefix())

    expected = [
        {
            "actionId": 6,
            "code": "SK",
            "card": Card.parse("SK").to_dict(),
            "ekka": False,
        },
        {
            "actionId": 25,
            "code": "C8",
            "card": Card.parse("C8").to_dict(),
            "ekka": False,
        },
        {
            "actionId": 38,
            "code": "SK+ekka",
            "card": Card.parse("SK").to_dict(),
            "ekka": True,
        },
    ]
    assert list(normalized.legal_actions) == expected
    assert normalized.to_dict()["legalActions"] == expected
    assert len({item["actionId"] for item in expected}) == len(expected)
    assert len({item["code"] for item in expected}) == len(expected)


def test_exact_state_replay_and_hashes_bind_ekka_declaration(rules):
    normal_prefix = _ekka_decision_prefix()
    normal_prefix["playActions"].append({"actor": "north", "card": "SK", "ekka": False})
    ekka_prefix = _ekka_decision_prefix()
    ekka_prefix["playActions"].append({"actor": "north", "card": "SK", "ekka": True})

    normal = ExactStateNormalizerV1(rules).normalize(normal_prefix)
    ekka = ExactStateNormalizerV1(rules).normalize(ekka_prefix)

    normal_play = normal.state["currentTrick"]["plays"][-1]
    ekka_play = ekka.state["currentTrick"]["plays"][-1]
    assert normal_play["card"]["code"] == ekka_play["card"]["code"] == "SK"
    assert normal_play["ekka"] is False
    assert ekka_play["ekka"] is True
    assert normal.prefix_hash != ekka.prefix_hash
    assert normal.state_hash != ekka.state_hash
    assert normal.event_chain_hash != ekka.event_chain_hash
    assert normal.exact_state_id != ekka.exact_state_id
    assert normal.sha256 != ekka.sha256

    asserted = copy.deepcopy(ekka_prefix)
    asserted["stateAssertions"] = {
        "currentTrick": copy.deepcopy(ekka.state["currentTrick"])
    }
    ExactStateNormalizerV1(rules).normalize(asserted)

    contradicted = copy.deepcopy(normal_prefix)
    contradicted["stateAssertions"] = {
        "currentTrick": copy.deepcopy(ekka.state["currentTrick"])
    }
    with pytest.raises(ValueError, match="assertion mismatch at currentTrick"):
        ExactStateNormalizerV1(rules).normalize(contradicted)


def test_exact_state_identity_ignores_equivalent_ui_validation_fields(rules, root):
    fixture = json.loads(
        (root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )["exactTracePrefix"]
    ui_prefix = copy.deepcopy(fixture)
    for action in ui_prefix["playActions"]:
        action["ekka"] = False
    ui_prefix["projectAssertions"][0]["highRank"] = 7
    ui_prefix["stateAssertions"] = {}
    validation_free_prefix = copy.deepcopy(fixture)
    validation_free_prefix["projectDeclarations"] = []
    validation_free_prefix["projectAssertions"] = []
    validation_free_prefix["stateAssertions"] = {}

    authored = ExactStateNormalizerV1(rules).normalize(fixture)
    ui = ExactStateNormalizerV1(rules).normalize(ui_prefix)
    validation_free = ExactStateNormalizerV1(rules).normalize(validation_free_prefix)

    assert {authored.prefix_hash, ui.prefix_hash, validation_free.prefix_hash} == {
        authored.prefix_hash
    }
    assert {authored.state_hash, ui.state_hash, validation_free.state_hash} == {
        authored.state_hash
    }
    assert {
        authored.event_chain_hash,
        ui.event_chain_hash,
        validation_free.event_chain_hash,
    } == {authored.event_chain_hash}
    assert {
        authored.exact_state_id,
        ui.exact_state_id,
        validation_free.exact_state_id,
    } == {authored.exact_state_id}
    assert {authored.sha256, ui.sha256, validation_free.sha256} == {authored.sha256}


def test_exact_state_rejects_illegal_ekka_declaration(rules):
    legal_normal = _ekka_decision_prefix()
    legal_normal["playActions"].append({"actor": "north", "card": "C8", "ekka": False})
    ExactStateNormalizerV1(rules).normalize(legal_normal)

    illegal_ekka = _ekka_decision_prefix()
    illegal_ekka["playActions"].append({"actor": "north", "card": "C8", "ekka": True})
    with pytest.raises(
        ValueError,
        match="Ekka may only be declared on the highest remaining non-trump lead",
    ):
        ExactStateNormalizerV1(rules).normalize(illegal_ekka)


def _predicate(predicate_id: str, kind: str, arguments: dict) -> dict:
    return {
        "predicateId": predicate_id,
        "kind": kind,
        "version": 1,
        "hardness": "hard",
        "arguments": arguments,
    }


def _scenario(*constraints: dict) -> ResearchScenarioV1:
    return ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {"op": "all", "children": list(constraints)},
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-EKKA-PHASE",
                "state.phase",
                {"state": "playing"},
            ),
        }
    )


def test_scenario_predicates_distinguish_card_and_ekka_action_sets(rules):
    normalized = ExactStateNormalizerV1(rules).normalize(_ekka_decision_prefix())
    scenario = _scenario(
        _predicate(
            "PRED-LEGAL-CARDS",
            "state.legal_cards_include",
            {"cards": [Card.parse("SK").id, Card.parse("C8").id], "complete": True},
        ),
        _predicate(
            "PRED-LEGAL-PLAY-ACTIONS",
            "state.legal_play_actions_include",
            {
                "actions": [
                    {"card": Card.parse("SK").id, "ekka": False},
                    {"card": Card.parse("C8").id, "ekka": False},
                    {"card": Card.parse("SK").id, "ekka": True},
                ],
                "complete": True,
            },
        ),
    )

    result = ScenarioRuntimeVerifierV1().verify(scenario, normalized)

    assert result.status == "satisfied"
    evidence = {item.kind: item for item in result.evidence}
    assert evidence["state.legal_cards_include"].observed["cardIds"] == [6, 25]
    assert evidence["state.legal_play_actions_include"].observed["actions"] == [
        {"card": 6, "ekka": False},
        {"card": 25, "ekka": False},
        {"card": 6, "ekka": True},
    ]


def test_scenario_action_prefix_binds_ekka_declaration(rules):
    prefix = _ekka_decision_prefix()
    prefix["playActions"].append({"actor": "north", "card": "SK", "ekka": True})
    normalized = ExactStateNormalizerV1(rules).normalize(prefix)
    actions = [
        {"card": play.card.id, "ekka": bool(play.ekka)}
        for play in normalized.round_state.played_cards
    ]
    scenario = _scenario(
        _predicate(
            "PRED-ACTION-PREFIX",
            "play.action_prefix",
            {"actions": actions, "complete": True},
        )
    )

    result = ScenarioRuntimeVerifierV1().verify(scenario, normalized)

    assert result.status == "satisfied"
    evidence = next(
        item for item in result.evidence if item.kind == "play.action_prefix"
    )
    assert evidence.observed["actions"][-1] == {"card": 6, "ekka": True}
