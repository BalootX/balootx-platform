from __future__ import annotations

import copy
import hashlib
import json

import pytest

from baloot_platform.core.cards import Card, FULL_DECK, cards_to_mask
from baloot_platform.core.enums import Contract, GamePhase, Seat, Team
from baloot_platform.core.exact_state_view import (
    build_canonical_observer_information_v1,
    canonical_observer_information_hash_v1,
)
from baloot_platform.core.models import BidRecord, Project, RoundScore, RoundState, Trick


def test_comparison_only_project_rank_uses_competing_best_categories():
    from baloot_platform.core.project_visibility import project_rank_is_public
    policy = {"mode": "comparison_only", "categoryPriorities": {"siri": 1, "fifty": 2, "hundred_sequence": 4, "hundred_four": 3}}
    def project(owner, kind, value):
        return {"owner": owner, "kind": kind, "value": value, "highRank": 7, "declarationOrder": 0, "revealed": False}
    north = project("north", "hundred_sequence", 100)
    west = project("west", "hundred_four", 100)
    south = project("south", "siri", 20)
    east = project("east", "siri", 20)
    rows = [north, west, south, east]
    assert not project_rank_is_public(west, rows, "north", policy, comparison_complete=True)
    assert not project_rank_is_public(east, rows, "north", policy, comparison_complete=True)
    west["kind"] = "hundred_sequence"
    assert project_rank_is_public(west, rows, "north", policy, comparison_complete=True)
    assert not project_rank_is_public(west, rows, "north", policy, comparison_complete=False)
    assert project_rank_is_public(west, rows, "west", policy, comparison_complete=False)
    west["revealed"] = True
    assert project_rank_is_public(west, rows, "north", policy, comparison_complete=False)
    assert project_rank_is_public(east, rows, "north", None, comparison_complete=False)


def test_hidden_project_rank_changes_do_not_change_observation(rules):
    state = _complete_state(rules)
    state["projectRankVisibility"] = {"mode": "comparison_only", "categoryPriorities": {"sequence": 1}}
    first = build_canonical_observer_information_v1(complete_state=state, acting_seat="south")
    changed = copy.deepcopy(state)
    changed["projects"][2]["highRank"] = 6
    for event in changed["events"]:
        if event.get("type") == "project_declared":
            event["highRank"] = 6
    second = build_canonical_observer_information_v1(complete_state=changed, acting_seat="south")
    assert first == second
    assert next(item for item in first["publicState"]["projects"] if item["owner"] == "west")["highRank"] is None


def test_runtime_rank_policy_is_optional_and_events_are_redacted(rules):
    from baloot_platform.core.engine import BalootEngine
    from baloot_platform.core.rules import RuleBook
    historical = BalootEngine(rules).create_round(seed=1)
    assert "projectRankVisibility" not in historical.to_dict()
    data = copy.deepcopy(rules.data)
    data["projects"]["top_rank_visibility"] = "comparison_only"
    engine = BalootEngine(RuleBook(data))
    state = engine.create_round(seed=1)
    state.projects = [Project(Seat.WEST, "siri", tuple(FULL_DECK[8:11]), 20, 2, 1, declared=True, declaration_order=0)]
    state.emit("project_declared", owner="west", kind="siri", value=20, highRank=2, declarationOrder=0)
    view = engine.observer_view(state, Seat.SOUTH)
    assert view["projects"][0]["highRank"] is None
    assert next(item for item in view["events"] if item["type"] == "project_declared")["highRank"] is None


def test_canonical_card_catalog_preserves_validation_and_copy_isolation():
    from baloot_platform.core.exact_state_view import _safe_card

    for card in FULL_DECK:
        source = card.to_dict()
        projected = _safe_card(source)
        assert projected == source and projected is not source
        projected["label"] = "changed"
        assert _safe_card(source) == card.to_dict()
        for field in ("code", "suit", "rank", "symbol", "label"):
            with pytest.raises(ValueError, match="not canonical"):
                _safe_card({**source, field: "invalid"})
    for invalid_id in (-1, 32, True, "0", 0.0):
        with pytest.raises(ValueError):
            _safe_card({**FULL_DECK[0].to_dict(), "id": invalid_id})
    with pytest.raises(ValueError):
        _safe_card({**FULL_DECK[0].to_dict(), "extra": "invalid"})


def _complete_state(rules) -> dict[str, object]:
    state = RoundState(rules.sha256, 91, 3, Seat.EAST)
    state.phase = GamePhase.PLAYING
    state.match_score = [42, 31]
    state.purchase_card = Card(0)
    state.initial_hands = [
        cards_to_mask(FULL_DECK[int(seat) * 8 : (int(seat) + 1) * 8])
        for seat in Seat
    ]
    state.hands = list(state.initial_hands)
    state.bid_history = [
        BidRecord(1, 0, Seat.NORTH, "sun", Contract.SUN, bidder=Seat.NORTH),
        BidRecord(1, 1, Seat.WEST, "pass"),
    ]
    state.contract = Contract.SUN
    state.buyer = Seat.NORTH
    state.declarer = Seat.NORTH
    state.escalation = "double"
    state.escalation_history = [
        {
            "seat": "west",
            "team": "west_east",
            "action": "double",
            "mode": "open",
        }
    ]
    state.first_leader = Seat.SOUTH
    state.current_player = Seat.SOUTH
    state.current_trick = Trick(0, Seat.SOUTH)
    state.projects = [
        Project(Seat.NORTH, "sequence", tuple(FULL_DECK[0:4]), 50, 3, 1),
        Project(Seat.SOUTH, "sequence", tuple(FULL_DECK[16:20]), 50, 3, 1),
        Project(
            Seat.WEST,
            "sequence",
            tuple(FULL_DECK[8:13]),
            100,
            4,
            1,
            declared=True,
            declaration_order=0,
            scoring=True,
        ),
    ]
    state.project_winning_team = Team.WEST_EAST
    state.baloot_teams = {Team.NORTH_SOUTH, Team.WEST_EAST}
    state.emit(
        "escalation_announced",
        seat="west",
        team="west_east",
        action="double",
        mode="open",
    )
    state.emit("baloot_declared", seat="south", team="north_south")
    return state.to_dict(reveal_hands=True)


def _played_state(rules) -> dict[str, object]:
    state = _complete_state(rules)
    plays = []
    for index, (seat, card_id) in enumerate((
        ("south", 16), ("east", 24), ("north", 0), ("west", 8),
        ("south", 17), ("east", 25),
    )):
        card = Card(card_id).to_dict()
        state["hands"][seat].remove(card)
        play = {"seat": seat, "card": card, "trickIndex": index // 4,
                "position": index % 4, "legalCards": [], "ekka": False, "baloot": False}
        plays.append(play)
        state["events"].append({"sequence": len(state["events"]), "type": "card_played",
                                **copy.deepcopy({key: value for key, value in play.items() if key != "legalCards"})})
        if index == 3:
            state["events"].append({"sequence": len(state["events"]), "type": "trick_completed",
                                    "trickIndex": 0, "winner": "south", "rawPoints": 0})
    state["completedTricks"] = [{"index": 0, "leader": "south", "plays": plays[:4],
                                  "winner": "south", "rawPoints": 0}]
    state["currentTrick"] = {"index": 1, "leader": "south", "plays": plays[4:],
                             "winner": None, "rawPoints": 0}
    state["currentPlayer"] = "north"
    return state


@pytest.mark.parametrize("actor,document_hash", [
    ("north", "581df577823eaa10256ab4618631e1b6a840abcd77a437575a9c2980987492f5"),
    ("west", "76c014e7f31983f9fdb7121ff7cd3fd25b77be4f66b12e1f24594d1680f5093c"),
    ("south", "0607326b5f1e443a6b7ae16abb07a8e9a914f5d1223eaed12f77535e964421a6"),
    ("east", "8957eb5214f65c944ed50d6da68d4bdb46ca4a79746377c43c12401cfc3e2d75"),
])
def test_played_history_preserves_frozen_observer_document(rules, actor, document_hash):
    # Full documents frozen from the pre-reuse implementation, without runtime Git access.
    projected = build_canonical_observer_information_v1(complete_state=_played_state(rules), acting_seat=actor)
    encoded = json.dumps(projected, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == document_hash


@pytest.mark.parametrize("actor", ["north", "west", "south", "east"])
def test_played_history_projects_each_trick_once(rules, monkeypatch, actor):
    from baloot_platform.core import exact_state_view

    state = _played_state(rules)
    original = copy.deepcopy(state)
    calls = []
    project_trick = exact_state_view._public_trick

    def counted(raw):
        calls.append(raw)
        return project_trick(raw)

    monkeypatch.setattr(exact_state_view, "_public_trick", counted)
    build_canonical_observer_information_v1(complete_state=state, acting_seat=actor)

    assert sorted(map(id, calls)) == sorted(map(id, [*state["completedTricks"], state["currentTrick"]]))
    assert state == original


def test_played_history_projection_is_detached_from_input_and_other_outputs(rules):
    state = _played_state(rules)
    original = copy.deepcopy(state)
    first = build_canonical_observer_information_v1(complete_state=state, acting_seat="north")
    expected = copy.deepcopy(first)

    first["publicState"]["completedTricks"][0]["plays"][0]["card"]["label"] = "changed output"
    first["publicState"]["currentTrick"]["plays"][0]["card"]["label"] = "changed output"
    first_event = next(event for event in first["publicState"]["events"] if event["type"] == "card_played")
    expected_event = next(event for event in expected["publicState"]["events"] if event["type"] == "card_played")
    assert first_event == expected_event
    assert state == original

    second = build_canonical_observer_information_v1(complete_state=state, acting_seat="north")
    assert second == expected
    state["completedTricks"][0]["plays"][0]["card"]["label"] = "changed input"
    state["currentTrick"]["plays"][0]["card"]["label"] = "changed input"
    next(event for event in state["events"] if event["type"] == "card_played")["card"]["label"] = "changed input"
    assert second == expected


@pytest.mark.parametrize("kind", ["contract_purchased", "projects_compared", "projects_revealed",
                                  "round_scored", "match_completed"])
def test_public_event_projection_detaches_nested_fields(kind):
    from baloot_platform.core.exact_state_view import _public_events

    project = {"owner": "west", "kind": "sequence", "value": 20, "highRank": 2,
        "declarationOrder": 0, "team": "west_east", "cards": [card.to_dict() for card in FULL_DECK[8:11]],
        "scoring": True, "revealed": True}
    score = RoundScore([80, 50], [16, 10], [0, 0], [0, 0], [16, 10],
        Team.NORTH_SOUTH, [16, 10], Team.NORTH_SOUTH, 1, explanation=["fixture"]).to_dict()
    payloads = {
        "contract_purchased": {"buyer": "north", "contract": "sun", "purchaseCard": Card(0).to_dict()},
        "projects_compared": {"winningTeam": "west_east", "declarations": [copy.deepcopy(project)]},
        "projects_revealed": {"owner": "west", "projects": [copy.deepcopy(project)]},
        "round_scored": {"score": score, "matchScore": [42, 31], "matchWinner": None},
        "match_completed": {"matchScore": [152, 31], "matchWinner": "north_south"},
    }
    state = {"events": [{"sequence": 0, "type": kind, **payloads[kind]}]}
    original, original_projects = copy.deepcopy(state), copy.deepcopy([project])
    first = _public_events(state, "west_east", [project])
    expected = copy.deepcopy(first)
    assert len(first) == 1

    def clear_nested(value):
        if isinstance(value, (dict, list)):
            for child in list(value.values()) if isinstance(value, dict) else list(value):
                clear_nested(child)
            value.clear()

    clear_nested(first)
    assert state == original and [project] == original_projects
    second = _public_events(state, "west_east", [project])
    clear_nested(state)
    clear_nested(project)
    assert second == expected


@pytest.mark.parametrize("mutation,message", [
    ("completed_card", "not canonical"),
    ("current_card", "not canonical"),
    ("trick_play", "trick play is invalid"),
    ("duplicate_trick", "trick indices are duplicated"),
    ("event_card", "not canonical"),
    ("event_owner", "played-card owner"),
    ("event_sequence", "event sequences are duplicated"),
])
def test_played_history_reuse_preserves_rejection(rules, mutation, message):
    state = _played_state(rules)
    event = next(event for event in state["events"] if event["type"] == "card_played")
    if mutation in {"completed_card", "current_card"}:
        trick = state["completedTricks"][0] if mutation == "completed_card" else state["currentTrick"]
        trick["plays"][0]["card"]["label"] = "invalid"
    elif mutation == "trick_play":
        state["currentTrick"]["plays"][0]["position"] = True
    elif mutation == "duplicate_trick":
        state["completedTricks"].append({**state["completedTricks"][0], "plays": []})
    elif mutation == "event_card":
        event["card"]["label"] = "invalid"
    elif mutation == "event_owner":
        event["seat"] = "north"
    else:
        event["sequence"] = state["events"][0]["sequence"]
    with pytest.raises(ValueError, match=message):
        build_canonical_observer_information_v1(complete_state=state, acting_seat="north")


def _swap_hidden_cards(state: dict[str, object]) -> None:
    hands = state["hands"]
    initial_hands = state["initialHands"]
    assert isinstance(hands, dict)
    assert isinstance(initial_hands, dict)
    for collection in (hands, initial_hands):
        north = collection["north"]
        west = collection["west"]
        assert isinstance(north, list)
        assert isinstance(west, list)
        north[1], west[1] = west[1], north[1]


def _swap_actor_card(state: dict[str, object]) -> None:
    hands = state["hands"]
    initial_hands = state["initialHands"]
    assert isinstance(hands, dict)
    assert isinstance(initial_hands, dict)
    for collection in (hands, initial_hands):
        south = collection["south"]
        east = collection["east"]
        assert isinstance(south, list)
        assert isinstance(east, list)
        south[0], east[0] = east[0], south[0]


def test_indistinguishable_hidden_worlds_have_one_information_hash(rules) -> None:
    first_state = _complete_state(rules)
    second_state = copy.deepcopy(first_state)
    _swap_hidden_cards(second_state)

    first = build_canonical_observer_information_v1(
        complete_state=first_state,
        acting_seat="south",
    )
    second = build_canonical_observer_information_v1(
        complete_state=second_state,
        acting_seat="south",
    )

    assert first == second
    assert first["observerInformationHash"] == canonical_observer_information_hash_v1(
        complete_state=second_state,
        acting_seat="south",
    )


def test_projection_is_canonical_for_unordered_public_collections(rules) -> None:
    original = _complete_state(rules)
    reordered = dict(reversed(list(copy.deepcopy(original).items())))
    hands = reordered["hands"]
    assert isinstance(hands, dict)
    assert isinstance(hands["south"], list)
    hands["south"].reverse()
    for key in ("projects", "balootTeams", "biddingHistory", "events"):
        value = reordered[key]
        assert isinstance(value, list)
        value.reverse()

    first = build_canonical_observer_information_v1(
        complete_state=original,
        acting_seat="south",
    )
    second = build_canonical_observer_information_v1(
        complete_state=reordered,
        acting_seat="south",
    )

    assert first == second
    assert len(first["observerInformationHash"]) == 64
    actor_ids = [card["id"] for card in first["publicState"]["hands"]["south"]]
    assert actor_ids == sorted(actor_ids)


def test_projection_contains_governed_public_state_without_hidden_cards(rules) -> None:
    complete_state = _complete_state(rules)
    projection = build_canonical_observer_information_v1(
        complete_state=complete_state,
        acting_seat="south",
    )
    public = projection["publicState"]

    assert {
        "biddingHistory",
        "winningBidder",
        "buyer",
        "contractOwner",
        "purchaseRecipient",
        "declarer",
        "contract",
        "trumpSuit",
        "escalation",
        "escalationHistory",
        "locked",
        "projects",
        "projectWinningTeam",
        "currentTrick",
        "completedTricks",
        "trumpBroken",
        "matchScore",
        "roundScore",
        "matchWinner",
    } <= set(public)
    assert isinstance(public["hands"]["south"], list)
    for seat in ("north", "west", "east"):
        assert public["hands"][seat] == {"count": 8}

    hidden_codes = {
        complete_state["hands"]["north"][1]["code"],
        complete_state["hands"]["west"][1]["code"],
        complete_state["hands"]["east"][1]["code"],
    }
    encoded = json.dumps(projection, ensure_ascii=False, sort_keys=True)
    assert all(code not in encoded for code in hidden_codes)
    assert "seed" not in public
    assert "initialHands" not in public
    assert "shuffle" not in public
    assert "dealTrace" not in public


def test_visible_public_or_private_mutation_changes_information_hash(rules) -> None:
    original = _complete_state(rules)
    public_mutation = copy.deepcopy(original)
    public_mutation["matchScore"][0] += 1
    private_mutation = copy.deepcopy(original)
    _swap_actor_card(private_mutation)

    original_hash = canonical_observer_information_hash_v1(
        complete_state=original,
        acting_seat="south",
    )
    assert original_hash != canonical_observer_information_hash_v1(
        complete_state=public_mutation,
        acting_seat="south",
    )
    assert original_hash != canonical_observer_information_hash_v1(
        complete_state=private_mutation,
        acting_seat="south",
    )
@pytest.mark.parametrize("revision", [3, 4])
def test_conservative_project_profile_has_bound_unratified_source_catalog(revision):
    import json
    from pathlib import Path
    import jsonschema
    from baloot_platform.core.rule_source_catalog import RuleSourceCatalogV1, load_exact_rule_source_catalog
    from baloot_platform.core.rules import RuleBook

    root = Path(__file__).resolve().parents[1]
    rules = RuleBook.load(root / f"configs/rules/saudi_federation_2023_v{revision}.yaml")
    catalog = RuleSourceCatalogV1.load(
        root / f"configs/rules/source_catalog_saudi_federation_2023_v{revision}.yaml", rules)
    assert load_exact_rule_source_catalog(rules, root).sha256 == catalog.sha256
    jsonschema.validate(rules.data, json.loads(
        (root / "schemas/profiles/rules.schema.json").read_text(encoding="utf-8")))
    assert rules.data["projects"]["top_rank_visibility"] == "comparison_only"
    assert rules.data["profile"]["expert_ratified"] is False
    if revision == 4:
        assert rules.data["scoring"]["contract"]["buyer_tie_outcome"] == "buyer_makes"
    for key in ("projects.declaration_mode", "projects.top_rank_visibility"):
        assert catalog.coverage[key]["ratification_status"] == "pending_owner_ratification"
