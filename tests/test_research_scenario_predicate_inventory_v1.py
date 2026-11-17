from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import yaml

from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Seat, Team
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.exact_state import ExactStateNormalizerV1
from baloot_platform.research.scenario_construction import ScenarioStructuralCompilerV1
from baloot_platform.research.scenario_contracts import (
    ResearchScenarioV1,
)
from baloot_platform.research.scenario_evidence import ScenarioRuntimeVerifierV1
from baloot_platform.research.scenario_registry import (
    SCENARIO_RULE_APPLICABILITY_CODES,
    ScenarioPredicateDefinitionV1,
    ScenarioPredicateRegistryV1,
)
from baloot_platform.research.service import ResearchFoundryServiceV1


REQUIRED_INITIAL_KINDS = {
    "deal": {
        "deal.card_owner",
        "deal.card_allowed",
        "deal.card_forbidden",
        "deal.suit_count",
        "deal.rank_count",
        "deal.high_card_count",
        "deal.low_card_count",
        "deal.purchase_card",
        "deal.purchase_suit",
        "deal.remaining_card_owner",
        "deal.remaining_suit_count",
        "deal.remaining_rank_count",
        "deal.team_high_card_balance",
        "deal.seat_relation",
    },
    "auction": {
        "auction.dealer",
        "auction.buyer",
        "auction.ashkal_buyer_relation",
        "auction.contract",
        "auction.trump",
        "auction.purchase_round",
        "auction.round",
        "auction.action_sequence",
        "auction.legal_actions",
        "auction.pass_count",
        "auction.terminal",
    },
    "escalation": {
        "escalation.level",
        "escalation.action_sequence",
        "escalation.locked",
        "escalation.eligible_actor",
        "escalation.legal_actions",
        "escalation.terminal",
    },
    "projects": {
        "project.exists",
        "project.count",
        "project.combined_value",
        "project.baloot",
        "project.overlap_count",
    },
    "playPrefix": {
        "play.trick_index",
        "play.current_player",
        "play.remaining_hand_size",
        "play.card_prefix",
        "play.action_prefix",
        "play.current_trick_leader",
        "play.current_trick_suit_led",
        "play.trump_use_count",
        "play.current_trick_card",
        "play.completed_trick_winner",
    },
    "state": {
        "state.phase",
        "state.next_actor",
        "state.legal_action_count",
        "state.match_score",
        "state.observer_visibility",
        "state.observer_card_visibility",
        "state.current_raw_points",
        "state.current_game_points",
        "state.current_project_points",
        "state.legal_cards_include",
        "state.legal_play_actions_include",
    },
    "outcome": {
        "outcome.buyer_contract",
        "outcome.team_round",
        "outcome.team_match",
        "outcome.raw_points",
        "outcome.game_points",
        "outcome.project_points",
        "outcome.round_margin",
        "outcome.kabout",
        "outcome.gahwa",
        "outcome.match_target_crossing",
    },
    "tactics": {"tactic.verified", "tactic.applicable"},
    "proof": {
        "proof.winning_root_action_count",
        "proof.winning_root_card_rank_count",
        "proof.winning_continuation_line_count",
        "proof.principal_variation_trick_winner",
        "proof.principal_variation_trick_team_rank_count",
    },
}


def _predicate(predicate_id: str, kind: str, arguments: dict, *, hardness="hard"):
    return {
        "predicateId": predicate_id,
        "kind": kind,
        "version": 1,
        "hardness": hardness,
        "arguments": arguments,
    }


def _scenario(
    constraints: list[dict],
    *,
    phase: str,
    decision_kind: str = "play",
) -> ResearchScenarioV1:
    return ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {"op": "all", "children": constraints},
            "decisionPoint": {"kind": decision_kind, "selector": {}},
            "outcome": _predicate(
                "PRED-EXPECTED-PHASE",
                "state.phase",
                {"state": phase},
            ),
        }
    )


def _base_prefix() -> dict:
    return {
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
        "stateAssertions": {},
    }


def _flagship(root: Path) -> dict:
    return json.loads(
        (root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )


def _terminal_flagship(root: Path, rules):
    fixture = _flagship(root)
    prefix = copy.deepcopy(fixture["exactTracePrefix"])
    prefix["stateAssertions"] = {}
    normalizer = ExactStateNormalizerV1(rules)
    for card in fixture["expected"]["principalVariation"]:
        current = normalizer.normalize(prefix)
        prefix["playActions"].append({"actor": current.next_actor, "card": card})
    return normalizer.normalize(prefix)


def _definition_payload(row: dict) -> dict:
    value = copy.deepcopy(row)
    value.pop("schemaVersion", None)
    return value


def _schema_localization_keys(value) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for name, item in value.items():
            if name in {"x-messageArKey", "x-messageEnKey"}:
                keys.add(str(item))
            elif name in {"x-enumMessageArKeys", "x-enumMessageEnKeys"}:
                keys.update(str(key) for key in item.values())
            else:
                keys.update(_schema_localization_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_schema_localization_keys(item))
    return keys


def test_initial_predicate_inventory_is_complete_truthful_and_ui_safe(root: Path):
    registry = ScenarioPredicateRegistryV1()
    catalog = registry.catalog()
    rows = {row["kind"]: row for row in catalog["definitions"]}

    for stage, required in REQUIRED_INITIAL_KINDS.items():
        assert required <= {
            row["kind"] for row in catalog["definitions"] if row["stage"] == stage
        }
    assert all(row["lifecycleStatus"] == "active" for row in rows.values())
    assert all(row["constructionCapability"] != "unsupported" for row in rows.values())
    assert all(
        row["runtimeVerifier"]
        for row in rows.values()
        if "hard" in row["allowedHardness"]
    )
    assert rows["project.count"]["constructionCapability"] == "postcondition_only"
    assert rows["project.count"]["costClass"] == "linear"
    assert rows["play.card_prefix"]["hiddenInformation"] == "public"
    assert rows["state.observer_visibility"]["allowedHardness"] == ["report"]
    for row in rows.values():
        assert all((root / path).is_file() for path in row["documentation"])
        assert all((root / path).is_file() for path in row["tests"])

    product = yaml.safe_load(
        (root / "configs/product/studio_ortools.yaml").read_text(encoding="utf-8")
    )
    copy_keys = product["foundry_copy"]["groups"]["workflow"]
    assert len(SCENARIO_RULE_APPLICABILITY_CODES) == len(
        set(SCENARIO_RULE_APPLICABILITY_CODES)
    )
    for code in SCENARIO_RULE_APPLICABILITY_CODES:
        assert copy_keys[f"research_scenario_issue_{code}"].strip()
        assert copy_keys[f"research_scenario_issue_{code}_en"].strip()
    for row in rows.values():
        assert copy_keys[row["messageArKey"]].strip()
        assert copy_keys[row["messageEnKey"]].strip()
        assert all(
            copy_keys[key].strip()
            for key in _schema_localization_keys(row["argumentSchema"])
        )

    definitions = tuple(
        ScenarioPredicateDefinitionV1.build(_definition_payload(row))
        for row in catalog["definitions"]
    )
    unsupported_payload = _definition_payload(rows["deal.card_owner"])
    unsupported_payload.pop("sha256")
    unsupported_payload.update(
        {
            "kind": "deal.not_implemented",
            "allowedHardness": ["prefer", "report"],
            "constructionCapability": "unsupported",
            "runtimeVerifier": "verify_deal_not_implemented_v1",
        }
    )
    for field in ("examples", "counterexamples"):
        unsupported_payload[field][0]["kind"] = "deal.not_implemented"
        unsupported_payload[field][0]["hardness"] = "prefer"
    with_unsupported = ScenarioPredicateRegistryV1(
        (*definitions, ScenarioPredicateDefinitionV1.build(unsupported_payload))
    )
    assert with_unsupported.definition("deal.not_implemented") is not None
    assert "deal.not_implemented" not in {
        row["kind"] for row in with_unsupported.catalog()["definitions"]
    }


def test_every_catalog_predicate_reaches_a_runtime_verifier_branch(root: Path, rules):
    registry = ScenarioPredicateRegistryV1()
    rows = registry.catalog()["definitions"]
    children = []
    for index, row in enumerate(rows):
        example = copy.deepcopy(row["examples"][0])
        example["predicateId"] = f"PRED-INVENTORY-{index:03d}"
        children.append(example)
    constraints = {
        "op": "all",
        "children": [
            {"op": "all", "children": children[:64]},
            {"op": "all", "children": children[64:]},
        ],
    }
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": constraints,
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-INVENTORY-PHASE",
                "state.phase",
                {"state": "scored"},
            ),
        }
    )

    satisfaction = ScenarioRuntimeVerifierV1(registry).verify(
        scenario,
        _terminal_flagship(root, rules),
    )

    assert len(satisfaction.evidence) == len(rows) + 1
    assert {item.kind for item in satisfaction.evidence} >= {
        row["kind"] for row in rows
    }


def test_purchase_suit_is_lowered_exactly_by_structural_solver(rules):
    scenario = _scenario(
        [
            _predicate("PRED-DEALER", "auction.dealer", {"seat": "east"}),
            _predicate(
                "PRED-PURCHASE-SUIT",
                "deal.purchase_suit",
                {"suit": "hearts"},
            ),
        ],
        phase="initial_dealt",
    )

    result = ScenarioStructuralCompilerV1().solve(
        scenario,
        rules=rules,
        dealer=Seat.EAST,
        seed=4901,
        maximum_seconds=10.0,
    )

    assert result.status == "satisfied"
    assert result.deck_order is not None
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST, buyer=None)
    assert (
        Card(result.deck_order[layout.purchase_position]).suit.name.lower() == "hearts"
    )
    assert "PRED-PURCHASE-SUIT" in {
        row["predicateId"] for row in result.predicate_evidence
    }


def test_play_state_and_tactic_predicates_verify_from_authoritative_state(
    root: Path,
    rules,
):
    fixture = _flagship(root)
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    played = [
        Card.parse(item["card"]).id
        for item in fixture["exactTracePrefix"]["playActions"]
    ]
    legal = sorted({int(item["card"]["id"]) for item in normalized.legal_actions})
    legal_actions = [
        {"card": int(item["card"]["id"]), "ekka": bool(item["ekka"])}
        for item in normalized.legal_actions
    ]
    constraints = [
        _predicate("PRED-PURCHASE-SUIT", "deal.purchase_suit", {"suit": "clubs"}),
        _predicate(
            "PRED-REMAINING-SUIT",
            "deal.remaining_suit_count",
            {"seat": "west", "suit": "diamonds", "minimum": 2, "maximum": 2},
        ),
        _predicate(
            "PRED-REMAINING-RANK",
            "deal.remaining_rank_count",
            {"seat": "north", "rank": "ace", "minimum": 1, "maximum": 1},
        ),
        _predicate(
            "PRED-SEAT-RELATION",
            "deal.seat_relation",
            {"seat": "west", "otherSeat": "east", "relation": "partner"},
        ),
        _predicate("PRED-AUCTION-ROUND", "auction.round", {"round": 1}),
        _predicate("PRED-PURCHASE-ROUND", "auction.purchase_round", {"round": 1}),
        _predicate(
            "PRED-AUCTION-SEQUENCE",
            "auction.action_sequence",
            {"actions": [{"actor": "north", "action": "sun"}], "complete": True},
        ),
        _predicate(
            "PRED-HAND-SIZE",
            "play.remaining_hand_size",
            {"seat": "west", "minimum": 3, "maximum": 3},
        ),
        _predicate(
            "PRED-PLAY-PREFIX",
            "play.card_prefix",
            {"cards": played, "complete": True},
        ),
        _predicate(
            "PRED-PLAY-ACTION-PREFIX",
            "play.action_prefix",
            {
                "actions": [{"card": card, "ekka": False} for card in played],
                "complete": True,
            },
        ),
        _predicate(
            "PRED-TRUMP-COUNT",
            "play.trump_use_count",
            {"minimum": 0, "maximum": 0},
        ),
        _predicate(
            "PRED-CURRENT-GAME",
            "state.current_game_points",
            {"team": "west_east", "minimum": 14, "maximum": 14},
        ),
        _predicate("PRED-NEXT-ACTOR", "state.next_actor", {"seat": "west"}),
        _predicate(
            "PRED-LEGAL-EXACT",
            "state.legal_cards_include",
            {"cards": legal, "complete": True},
        ),
        _predicate(
            "PRED-LEGAL-ACTIONS-EXACT",
            "state.legal_play_actions_include",
            {"actions": legal_actions, "complete": True},
        ),
        _predicate(
            "PRED-VISIBLE-OWN",
            "state.observer_card_visibility",
            {"observer": "west", "card": Card.parse("CA").id, "visibility": "visible"},
        ),
        _predicate(
            "PRED-HIDDEN-OPPONENT",
            "state.observer_card_visibility",
            {"observer": "west", "card": Card.parse("SA").id, "visibility": "hidden"},
        ),
        _predicate(
            "PRED-TACTIC-VERIFIED",
            "tactic.verified",
            {
                "kind": "large_card_sacrifice",
                "actingSeat": "west",
                "observer": "west",
                "targetTeam": "west_east",
                "causalRequired": True,
                "minimumScore": 70,
                "minimumEvidence": 1,
            },
        ),
        _predicate(
            "PRED-TACTIC-APPLICABLE",
            "tactic.applicable",
            {
                "kind": "large_card_sacrifice",
                "actingSeat": "west",
                "observer": "west",
                "targetTeam": "west_east",
                "applicable": True,
            },
        ),
    ]
    scenario = _scenario(constraints, phase="playing")
    tactical = {
        "large_card_sacrifice": [
            {
                "actingSeat": "west",
                "observer": "west",
                "targetTeam": "west_east",
                "score": 80,
                "applicable": True,
                "causalProofHash": "a" * 64,
            }
        ]
    }

    satisfaction = ScenarioRuntimeVerifierV1().verify(
        scenario,
        normalized,
        tactical_evidence=tactical,
    )

    assert satisfaction.status == "satisfied"
    assert {item.status for item in satisfaction.evidence} == {"satisfied"}


def test_unavailable_hand_and_play_legal_card_queries_fail_closed(rules):
    normalized = ExactStateNormalizerV1(rules).normalize(
        _base_prefix(),
        decision_point={"kind": "bid", "selector": {"actionIndex": 0}},
    )
    purchase = normalized.round_state.purchase_card
    assert purchase is not None
    scenario = _scenario(
        [
            _predicate(
                "PRED-EXPOSED-CARD-OWNER",
                "deal.card_owner",
                {
                    "card": purchase.id,
                    "seat": "north",
                    "handStage": "initial_five",
                },
            ),
            _predicate(
                "PRED-PLAY-CARDS-DURING-BID",
                "state.legal_cards_include",
                {"cards": [purchase.id]},
            ),
        ],
        phase="bidding",
        decision_kind="bid",
    )

    result = ScenarioRuntimeVerifierV1().verify(scenario, normalized)
    evidence = {item.predicate_id: item for item in result.evidence}

    assert evidence["PRED-EXPOSED-CARD-OWNER"].status == "failed"
    assert evidence["PRED-EXPOSED-CARD-OWNER"].observed["owner"] is None
    assert evidence["PRED-PLAY-CARDS-DURING-BID"].status == "not_evaluated"
    assert (
        evidence["PRED-PLAY-CARDS-DURING-BID"].failure_code
        == "play_legal_cards_unavailable"
    )


def test_kabout_and_gahwa_outcomes_are_bound_to_the_winning_team(root: Path, rules):
    terminal = _terminal_flagship(root, rules)
    state = terminal.round_state.clone()
    assert state.round_score is not None
    state.round_score.winner = Team.WEST_EAST

    for outcome_kind in ("outcome.kabout", "outcome.gahwa"):
        state.round_score.kabout = outcome_kind == "outcome.kabout"
        state.round_score.gahwa = outcome_kind == "outcome.gahwa"
        normalized = replace(terminal, round_state=state.clone())
        scenario = _scenario(
            [
                _predicate(
                    "PRED-WINNING-TEAM-OUTCOME",
                    outcome_kind,
                    {"team": "west_east", "achieved": True},
                ),
                _predicate(
                    "PRED-LOSING-TEAM-OUTCOME",
                    outcome_kind,
                    {"team": "north_south", "achieved": False},
                ),
            ],
            phase=normalized.phase,
        )

        result = ScenarioRuntimeVerifierV1().verify(scenario, normalized)
        evidence = {item.predicate_id: item for item in result.evidence}

        assert evidence["PRED-WINNING-TEAM-OUTCOME"].status == "satisfied"
        assert evidence["PRED-LOSING-TEAM-OUTCOME"].status == "satisfied"


def test_bid_and_escalation_legal_action_sets_are_exact(rules):
    bid_prefix = _base_prefix()
    bid_prefix["auctionActions"] = [{"actor": "north", "action": "pass"}]
    bid_prefix["stateAssertions"] = {"phase": "bidding"}
    bid = ExactStateNormalizerV1(rules).normalize(
        bid_prefix,
        decision_point={"kind": "bid", "selector": {"actionIndex": 1}},
    )
    bid_actions = [
        {
            "action": item["action"],
            **(
                {"trumpSuit": item["trumpSuit"]}
                if item.get("trumpSuit") is not None
                else {}
            ),
        }
        for item in bid.legal_actions
    ]
    bid_scenario = _scenario(
        [
            _predicate("PRED-BID-ROUND", "auction.round", {"round": 1}),
            _predicate("PRED-BID-NEXT", "state.next_actor", {"seat": bid.next_actor}),
            _predicate(
                "PRED-BID-SEQUENCE",
                "auction.action_sequence",
                {"actions": [{"actor": "north", "action": "pass"}], "complete": True},
            ),
            _predicate(
                "PRED-BID-ACTIONS",
                "auction.legal_actions",
                {"actions": bid_actions, "complete": True},
            ),
            _predicate("PRED-BID-TERMINAL", "auction.terminal", {"terminal": False}),
        ],
        phase="bidding",
        decision_kind="bid",
    )
    assert ScenarioRuntimeVerifierV1().verify(bid_scenario, bid).status == "satisfied"

    escalation_prefix = _base_prefix()
    escalation_prefix["matchScore"] = [101, 0]
    escalation_prefix["auctionActions"] = [
        {"action": "pass"},
        {"action": "pass"},
        {"action": "sun"},
        {"action": "pass"},
        {"action": "sun"},
    ]
    escalation_prefix["stateAssertions"] = {"phase": "final_dealt"}
    complete = ExactStateNormalizerV1(rules).normalize(
        {**escalation_prefix, "stateAssertions": {}}
    )
    declarer = complete.round_state.declarer
    assert declarer is not None
    actor = next(seat for seat in Seat if seat.team != declarer.team)
    escalation = ExactStateNormalizerV1(rules).normalize(
        escalation_prefix,
        decision_point={
            "kind": "escalation",
            "selector": {"actionIndex": 0, "actingSeat": actor.name.lower()},
        },
    )
    escalation_actions = [
        {
            "action": item["action"],
            **({"mode": item["mode"]} if item.get("mode") is not None else {}),
        }
        for item in escalation.legal_actions
    ]
    team = actor.team.name.lower()
    escalation_scenario = _scenario(
        [
            _predicate(
                "PRED-ESCALATION-ELIGIBLE",
                "escalation.eligible_actor",
                {"team": team, "seat": actor.name.lower(), "eligible": True},
            ),
            _predicate(
                "PRED-ESCALATION-NEXT",
                "state.next_actor",
                {"seat": actor.name.lower()},
            ),
            _predicate(
                "PRED-ESCALATION-ACTIONS",
                "escalation.legal_actions",
                {"actions": escalation_actions, "complete": True},
            ),
            _predicate(
                "PRED-ESCALATION-TERMINAL",
                "escalation.terminal",
                {"terminal": False},
            ),
        ],
        phase="final_dealt",
        decision_kind="escalation",
    )
    assert (
        ScenarioRuntimeVerifierV1().verify(escalation_scenario, escalation).status
        == "satisfied"
    )


def test_ashkal_relation_and_terminal_outcomes_are_authoritative(root: Path, rules):
    prefix = _base_prefix()
    prefix["auctionActions"] = [
        {"actor": "north", "action": "pass"},
        {"actor": "west", "action": "pass"},
        {"actor": "south", "action": "ashkal"},
        {"actor": "west", "action": "pass"},
        {"actor": "north", "action": "pass"},
    ]
    prefix["stateAssertions"] = {"phase": "playing"}
    ashkal = ExactStateNormalizerV1(rules).normalize(prefix)
    ashkal_scenario = _scenario(
        [
            _predicate(
                "PRED-ASHKAL-RELATION",
                "auction.ashkal_buyer_relation",
                {"relation": "bidder_partner"},
            ),
            _predicate("PRED-ASHKAL-BUYER", "auction.buyer", {"seat": "north"}),
            _predicate(
                "PRED-ASHKAL-CONTRACT",
                "auction.contract",
                {"contract": "ashkal"},
            ),
        ],
        phase="playing",
    )
    assert (
        ScenarioRuntimeVerifierV1().verify(ashkal_scenario, ashkal).status
        == "satisfied"
    )

    terminal = _terminal_flagship(root, rules)
    terminal_scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _predicate(
                "PRED-TERMINAL-PHASE", "state.phase", {"state": "scored"}
            ),
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-PROJECT-POINTS",
                        "outcome.project_points",
                        {"team": "north_south", "minimum": 20, "maximum": 20},
                    ),
                    _predicate(
                        "PRED-MATCH-TARGET",
                        "outcome.match_target_crossing",
                        {"team": "north_south", "achieved": False},
                    ),
                    _predicate(
                        "PRED-ROUND-WINNER",
                        "outcome.team_round",
                        {"team": "west_east", "status": "won"},
                    ),
                ],
            },
        }
    )
    assert (
        ScenarioRuntimeVerifierV1().verify(terminal_scenario, terminal).status
        == "satisfied"
    )


def test_project_count_is_not_advertised_as_unused_guidance():
    scenario = _scenario(
        [
            _predicate(
                "PRED-PROJECT-COUNT",
                "project.count",
                {"team": "north_south", "minimum": 1, "maximum": 3},
            )
        ],
        phase="playing",
    )

    plan = ScenarioStructuralCompilerV1().plan(scenario)

    assert "PRED-PROJECT-COUNT" in plan.postcondition_predicate_ids
    assert "PRED-PROJECT-COUNT" not in plan.guided_predicate_ids


def test_unregistered_tactic_detector_fails_before_execution(rules):
    scenario = _scenario(
        [
            _predicate(
                "PRED-UNKNOWN-TACTIC",
                "tactic.verified",
                {
                    "kind": "not_a_registered_detector",
                    "minimumScore": 1,
                    "minimumEvidence": 1,
                },
            )
        ],
        phase="playing",
    )

    issues = ResearchFoundryServiceV1._scenario_rules_applicability_issues(
        SimpleNamespace(scenario=scenario),
        rules,
        supported_tactics=("only_winning_move",),
    )

    assert [item["code"] for item in issues] == [
        "tactic_detector_not_enabled_by_bundle"
    ]
    assert issues[0]["predicateIds"] == ["PRED-UNKNOWN-TACTIC"]


def test_bundle_rule_options_omit_disabled_contract_and_rule_variants(rules):
    data = copy.deepcopy(rules.data)
    data["contracts"]["allowed"] = ["sun", "hokum"]
    data["bidding"]["first_round"]["allowed"] = ["pass", "sun", "hokum"]
    data["bidding"]["second_round"]["allowed"] = ["pass", "sun", "hokum"]
    data["bidding"]["ashkal"]["enabled"] = False
    data["escalation"]["enabled"] = False
    data["scoring"]["kabout"]["enabled"] = False
    restricted = RuleBook(data)

    options = ResearchFoundryServiceV1._scenario_rule_options(restricted)
    play_action_options = options.pop("playActionOptions")

    assert options == {
        "contracts": ["sun", "hokum"],
        "auctionActionsByRound": [
            {"round": 1, "actions": ["pass", "sun", "hokum"]},
            {"round": 2, "actions": ["pass", "sun", "hokum"]},
        ],
        "escalationLevelsByContract": [
            {"contract": "sun", "levels": ["normal"], "actions": []},
            {"contract": "hokum", "levels": ["normal"], "actions": []},
        ],
        "ashkalBuyerRelations": [],
        "trumpContracts": ["hokum"],
        "balootContracts": ["hokum"],
        "projectsEnabled": True,
        "kaboutEnabled": False,
    }
    assert len(play_action_options) == 64
    assert [item["actionId"] for item in play_action_options] == list(range(64))
    assert play_action_options[0]["code"] == "S7"
    assert play_action_options[32]["code"] == "S7+ekka"
    assert play_action_options[32]["ekka"] is True


def test_disabled_rule_variants_fail_before_scenario_execution(rules):
    data = copy.deepcopy(rules.data)
    data["contracts"]["allowed"] = ["sun", "hokum"]
    data["bidding"]["first_round"]["allowed"] = ["pass", "sun", "hokum"]
    data["bidding"]["second_round"]["allowed"] = ["pass", "sun", "hokum"]
    data["bidding"]["ashkal"]["enabled"] = False
    data["escalation"]["enabled"] = False
    data["scoring"]["kabout"]["enabled"] = False
    restricted = RuleBook(data)
    scenario = _scenario(
        [
            _predicate(
                "PRED-DISABLED-CONTRACT",
                "auction.contract",
                {"contract": "ashkal"},
            ),
            _predicate(
                "PRED-DISABLED-ASHKAL",
                "auction.ashkal_buyer_relation",
                {"relation": "bidder_partner"},
            ),
            _predicate(
                "PRED-DISABLED-AUCTION-ACTION",
                "auction.action_sequence",
                {"actions": [{"action": "ashkal"}], "complete": False},
            ),
            _predicate(
                "PRED-DISABLED-ESCALATION-LEVEL",
                "escalation.level",
                {"level": "double"},
            ),
            _predicate(
                "PRED-DISABLED-ESCALATION-STATE",
                "escalation.locked",
                {"locked": True},
            ),
            _predicate(
                "PRED-DISABLED-KABOUT",
                "outcome.kabout",
                {"team": "north_south", "achieved": True},
            ),
        ],
        phase="playing",
    )

    issues = ResearchFoundryServiceV1._scenario_rules_applicability_issues(
        SimpleNamespace(scenario=scenario),
        restricted,
    )

    assert [item["code"] for item in issues] == [
        "contract_not_enabled_by_rules",
        "ashkal_not_enabled_by_rules",
        "auction_action_not_enabled_by_rules",
        "escalation_level_not_enabled_by_rules",
        "escalation_not_enabled_by_rules",
        "kabout_not_enabled_by_rules",
    ]
    assert all(item["class"] == "rules_inapplicable" for item in issues)
    assert all(item["severity"] == "error" for item in issues)


def test_disabled_ekka_is_omitted_and_rejected_before_execution(rules):
    data = copy.deepcopy(rules.data)
    data["play"]["ekka_enabled"] = False
    restricted = SimpleNamespace(
        data=data,
        project_domain=rules.project_domain,
        sha256="f" * 64,
    )
    options = ResearchFoundryServiceV1._scenario_rule_options(restricted)
    scenario = _scenario(
        [
            _predicate(
                "PRED-DISABLED-EKKA",
                "play.action_prefix",
                {"actions": [{"card": 6, "ekka": True}]},
            )
        ],
        phase="playing",
    )

    issues = ResearchFoundryServiceV1._scenario_rules_applicability_issues(
        SimpleNamespace(scenario=scenario),
        restricted,
    )

    assert len(options["playActionOptions"]) == 32
    assert not any(item["ekka"] for item in options["playActionOptions"])
    assert [item["code"] for item in issues] == ["ekka_not_enabled_by_rules"]
    assert issues[0]["predicateIds"] == ["PRED-DISABLED-EKKA"]


def test_sun_only_rules_omit_and_reject_inapplicable_ekka_actions(rules):
    data = copy.deepcopy(rules.data)
    data["contracts"]["allowed"] = ["sun"]
    data["bidding"]["first_round"]["allowed"] = ["pass", "sun"]
    data["bidding"]["second_round"]["allowed"] = ["pass", "sun"]
    data["bidding"]["first_round"]["hokum_uses_purchase_suit"] = False
    data["bidding"]["second_round"]["hokum_must_differ_from_purchase_suit"] = False
    restricted = RuleBook(data)
    options = ResearchFoundryServiceV1._scenario_rule_options(restricted)
    scenario = _scenario(
        [
            _predicate(
                "PRED-SUN-ONLY-EKKA",
                "state.legal_play_actions_include",
                {
                    "actions": [{"card": 6, "ekka": True}],
                    "complete": False,
                },
            )
        ],
        phase="playing",
    )

    issues = ResearchFoundryServiceV1._scenario_rules_applicability_issues(
        SimpleNamespace(scenario=scenario),
        restricted,
    )

    assert options["contracts"] == ["sun"]
    assert options["trumpContracts"] == []
    assert len(options["playActionOptions"]) == 32
    assert not any(item["ekka"] for item in options["playActionOptions"])
    assert [item["code"] for item in issues] == ["ekka_not_enabled_by_rules"]


def test_sun_constrained_recipe_rejects_ekka_from_a_full_bundle(rules):
    scenario = _scenario(
        [
            _predicate(
                "PRED-SUN-CONTRACT",
                "auction.contract",
                {"contract": "sun"},
            ),
            _predicate(
                "PRED-SUN-EKKA",
                "play.action_prefix",
                {"actions": [{"card": 6, "ekka": True}]},
            ),
        ],
        phase="playing",
    )

    issues = ResearchFoundryServiceV1._scenario_rules_applicability_issues(
        SimpleNamespace(scenario=scenario),
        rules,
    )

    assert [item["code"] for item in issues] == ["ekka_not_enabled_by_rules"]
    assert issues[0]["predicateIds"] == ["PRED-SUN-EKKA"]
