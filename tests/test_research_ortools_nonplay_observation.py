from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("ortools")

from baloot_platform.core.cards import Card, FULL_DECK, cards_to_mask
from baloot_platform.core.enums import Contract, GamePhase, Seat
from baloot_platform.core.models import Project, RoundState, Trick
from baloot_platform.core.project_ir import ProjectDomainIRV1, project_id_v1
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.ortools_complete_state_graph import (
    CompleteStateDecisionObservationV1,
)
from baloot_platform.research.ortools_exact_state import (
    OrToolsExactStateNormalizerV1,
    OrToolsExactNormalizationReceiptV1,
    OrToolsNormalizedExactStateV1,
    _effective_prefix_hash,
)
from baloot_platform.research.ortools_nonplay_complete_graph import (
    compile_nonplay_complete_graph_v1,
)
from baloot_platform.research.ortools_nonplay_observation import (
    GovernedProjectDisclosureV1,
    build_nonplay_decision_observer_v1,
    build_nonplay_root_observer_v1,
    canonical_nonplay_observation_hash_v1,
)
from baloot_platform.research.ortools_nonplay_graphs import (
    NonPlayDecisionObservationV1,
)
from baloot_platform.research.ortools_play_tail import TailPrefixActionV1
from baloot_platform.research.scenario_contracts import (
    ExactTracePrefixV1,
    ScenarioProofRequestV1,
    canonical_sha256,
)


def _state(
    rules,
    *,
    projects: tuple[Project, ...] = (),
) -> dict[str, object]:
    state = RoundState(rules.sha256, 41, 1, Seat.EAST)
    state.phase = GamePhase.PLAYING
    state.purchase_card = Card(0)
    state.initial_hands = [
        cards_to_mask(FULL_DECK[int(seat) * 8 : (int(seat) + 1) * 8]) for seat in Seat
    ]
    state.hands = list(state.initial_hands)
    state.contract = Contract.SUN
    state.buyer = Seat.NORTH
    state.declarer = Seat.NORTH
    state.first_leader = Seat.SOUTH
    state.current_player = Seat.SOUTH
    state.current_trick = Trick(0, Seat.SOUTH)
    state.projects = list(projects)
    return state.to_dict(reveal_hands=True)


def _prefix(**overrides) -> ExactTracePrefixV1:
    value = {
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
    value.update(overrides)
    return ExactTracePrefixV1.build(value)


def _exact(
    state: dict[str, object],
    rules,
    *,
    prefix: ExactTracePrefixV1 | None = None,
    decision_kind: str = "play",
) -> OrToolsNormalizedExactStateV1:
    exact_prefix = prefix or _prefix()
    state_hash = canonical_sha256(state)
    identity = canonical_sha256({"state": state_hash, "rules": rules.sha256})
    return OrToolsNormalizedExactStateV1(
        f"STATE-{identity[:20].upper()}",
        _effective_prefix_hash(exact_prefix),
        rules.sha256,
        state_hash,
        "playing",
        decision_kind,
        "south",
        (),
        state,
        0,
        0,
        canonical_sha256(state.get("events") or []),
        identity,
        OrToolsExactNormalizationReceiptV1({"sha256": "1" * 64}),
    )


def _remaining(state: dict[str, object]) -> tuple[tuple[Card, ...], ...]:
    hands = state["hands"]
    assert isinstance(hands, dict)
    return tuple(
        tuple(Card.parse(card) for card in hands[seat.name.lower()]) for seat in Seat
    )


def _history(*actions: str) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "ordinal": ordinal,
            "stage": "auction",
            "actor": "north" if ordinal % 2 == 0 else "west",
            "action": action,
        }
        for ordinal, action in enumerate(actions)
    )


def _observation(
    state: dict[str, object],
    *,
    prefix: tuple[TailPrefixActionV1, ...] = (),
    actor: Seat = Seat.SOUTH,
) -> CompleteStateDecisionObservationV1:
    return CompleteStateDecisionObservationV1(
        actor=actor,
        path=prefix,
        prefix=prefix,
        remaining_hands=_remaining(state),
        current_trick=prefix[-(len(prefix) % 4) :] if len(prefix) % 4 else (),
        trick_winners=(),
        trick_raw_points=(),
        raw=(0, 0),
        tricks=(0, 0),
    )


def test_pure_nonplay_observation_preserves_order_and_private_hand() -> None:
    hand = tuple(Card(card_id) for card_id in range(8))
    first = canonical_nonplay_observation_hash_v1(
        decision_kind="bid",
        actor="north",
        public_action_history=_history("pass", "sun"),
        actor_visible_hand=hand,
        revealed_projects=(),
    )
    reordered = canonical_nonplay_observation_hash_v1(
        decision_kind="bid",
        actor="north",
        public_action_history=_history("sun", "pass"),
        actor_visible_hand=hand,
        revealed_projects=(),
    )
    different_hand = canonical_nonplay_observation_hash_v1(
        decision_kind="bid",
        actor="north",
        public_action_history=_history("pass", "sun"),
        actor_visible_hand=(*hand[:-1], Card(8)),
        revealed_projects=(),
    )

    assert first != reordered
    assert first != different_hand
    with pytest.raises(ValueError, match="private/unknown"):
        canonical_nonplay_observation_hash_v1(
            decision_kind="bid",
            actor="north",
            public_action_history=(
                {
                    "ordinal": 0,
                    "stage": "project",
                    "actor": "north",
                    "action": "declare",
                    "projectId": "private-id",
                },
            ),
            actor_visible_hand=hand,
            revealed_projects=(),
        )
    with pytest.raises(ValueError, match="do not match"):
        canonical_nonplay_observation_hash_v1(
            decision_kind="bid",
            actor="north",
            public_action_history=(
                {
                    "ordinal": 0,
                    "stage": "auction",
                    "actor": "north",
                    "action": "pass",
                    "owner": "north",
                },
            ),
            actor_visible_hand=hand,
            revealed_projects=(),
        )


def test_factory_ignores_opponent_hands_but_binds_perfect_recall(rules) -> None:
    first_state = _state(rules)
    second_state = copy.deepcopy(first_state)
    for key in ("initialHands", "hands"):
        hands = second_state[key]
        assert isinstance(hands, dict)
        north = hands["north"]
        west = hands["west"]
        assert isinstance(north, list) and isinstance(west, list)
        north[1], west[1] = west[1], north[1]
    first = build_nonplay_decision_observer_v1(
        rules=rules,
        normalized_exact=_exact(first_state, rules),
        exact_prefix=_prefix(),
        decision_kind="bid",
        projects=(),
    )
    second = build_nonplay_decision_observer_v1(
        rules=rules,
        normalized_exact=_exact(second_state, rules),
        exact_prefix=_prefix(),
        decision_kind="bid",
        projects=(),
    )

    assert first(_observation(first_state)) == second(_observation(second_state))
    play = TailPrefixActionV1.build(Seat.SOUTH, Card.parse("D7"))
    changed = _observation(first_state, prefix=(play,))
    assert first(changed) != first(_observation(first_state))
    malformed = CompleteStateDecisionObservationV1(
        actor=Seat.SOUTH,
        path=(),
        prefix=(play,),
        remaining_hands=_remaining(first_state),
        current_trick=(play,),
        trick_winners=(),
        trick_raw_points=(),
        raw=(0, 0),
        tricks=(0, 0),
    )
    with pytest.raises(ValueError, match="perfect recall"):
        first(malformed)


def _siri(suit: str) -> Project:
    return Project(
        Seat.NORTH,
        "siri",
        tuple(Card.parse(f"{suit}{rank}") for rank in ("7", "8", "9")),
        20,
        2,
        1,
        declared=True,
        declaration_order=1,
        scoring=True,
    )


def test_project_disclosure_is_governed_and_path_dependent(rules) -> None:
    spades = _siri("S")
    hearts = _siri("H")
    first_state = _state(rules, projects=(spades,))
    second_state = _state(rules, projects=(hearts,))
    for key in ("initialHands", "hands"):
        hands = second_state[key]
        assert isinstance(hands, dict)
        hands["north"], hands["west"] = hands["west"], hands["north"]
    before = tuple(
        TailPrefixActionV1.build(seat, Card(card_id))
        for seat, card_id in zip(Seat, range(4), strict=True)
    )
    after = (*before, TailPrefixActionV1.build(Seat.NORTH, Card(4)))
    first = build_nonplay_decision_observer_v1(
        rules=rules,
        normalized_exact=_exact(first_state, rules),
        exact_prefix=_prefix(),
        decision_kind="project",
        projects=(spades,),
    )
    second = build_nonplay_decision_observer_v1(
        rules=rules,
        normalized_exact=_exact(second_state, rules),
        exact_prefix=_prefix(),
        decision_kind="project",
        projects=(hearts,),
    )
    assert first(_observation(first_state, prefix=before)) == second(
        _observation(second_state, prefix=before)
    )
    assert first(_observation(first_state, prefix=after)) != second(
        _observation(second_state, prefix=after)
    )

    disclosure = GovernedProjectDisclosureV1(
        spades.owner,
        spades.kind,
        spades.value,
        spades.high_rank,
        spades.cards,
        "owner_play_in_trick",
        1,
    )
    assert disclosure.is_revealed(before) is False
    assert disclosure.is_revealed(after) is True
    with pytest.raises(ValueError, match="timing"):
        GovernedProjectDisclosureV1.build(
            {
                **disclosure.to_dict(),
                "revealTrigger": {
                    "kind": "owner_play_in_trick",
                    "trickIndex": 8,
                },
            }
        )


def test_play_observer_rejects_hybrid_or_untrusted_inputs(rules) -> None:
    project = _siri("S")
    empty_state = _state(rules)
    with pytest.raises(ValueError, match="trusted projects"):
        build_nonplay_decision_observer_v1(
            rules=rules,
            normalized_exact=_exact(empty_state, rules),
            exact_prefix=_prefix(),
            decision_kind="project",
            projects=(project,),
        )
    with pytest.raises(ValueError, match="duplicated"):
        build_nonplay_decision_observer_v1(
            rules=rules,
            normalized_exact=_exact(_state(rules, projects=(project,)), rules),
            exact_prefix=_prefix(),
            decision_kind="project",
            projects=(project, project),
        )
    with pytest.raises(ValueError, match="ownership mismatch"):
        build_nonplay_decision_observer_v1(
            rules=rules,
            normalized_exact=_exact(empty_state, rules),
            exact_prefix=_prefix(roundNumber=2),
            decision_kind="bid",
            projects=(),
        )


def _root_public_state(
    decision_kind: str,
    actions: tuple[dict[str, object], ...],
) -> dict[str, object]:
    if decision_kind == "bid":
        return {
            "stage": "auction",
            "state": {
                "records": [{"round": 1, "trump": None, **action} for action in actions]
            },
        }
    if decision_kind == "escalation":
        return {"stage": "escalation", "actions": list(actions)}
    return {"stage": "project", "prefix": list(actions), "projects": []}


@pytest.mark.parametrize("decision_kind", ["bid", "escalation", "project"])
def test_root_observer_is_nonanticipative_and_remembers_public_actions(
    rules,
    decision_kind: str,
) -> None:
    prefix = _prefix()
    first_state = _state(rules)
    second_state = copy.deepcopy(first_state)
    for key in ("initialHands", "hands"):
        hands = second_state[key]
        assert isinstance(hands, dict)
        hands["north"], hands["west"] = hands["west"], hands["north"]
    first = build_nonplay_root_observer_v1(
        _exact(first_state, rules, prefix=prefix, decision_kind=decision_kind),
        prefix,
        decision_kind,
    )
    second = build_nonplay_root_observer_v1(
        _exact(second_state, rules, prefix=prefix, decision_kind=decision_kind),
        prefix,
        decision_kind,
    )
    empty = NonPlayDecisionObservationV1(
        "private-node-a",
        decision_kind,
        Seat.SOUTH,
        _root_public_state(decision_kind, ()),
    )
    assert first(empty) == second(
        NonPlayDecisionObservationV1(
            "different-private-node",
            decision_kind,
            Seat.SOUTH,
            _root_public_state(decision_kind, ()),
        )
    )
    action = (
        {"actor": "north", "action": "pass"}
        if decision_kind != "escalation"
        else {"actor": "north", "action": "double", "mode": None}
    )
    changed = NonPlayDecisionObservationV1(
        "private-node-b",
        decision_kind,
        Seat.SOUTH,
        _root_public_state(decision_kind, (action,)),
    )
    assert first(empty) != first(changed)
    if decision_kind == "escalation":
        from dataclasses import replace
        own_memory = replace(empty, own_private_windows=("a" * 64,))
        assert first(own_memory) != first(empty)
        assert first(own_memory) == second(own_memory)


def _graph_project(project: Project) -> dict[str, object]:
    return {
        "owner": project.owner.name.lower(),
        "kind": project.kind,
        "cards": [card.id for card in project.cards],
        "value": project.value,
        "highRank": project.high_rank,
        "categoryPriority": project.category_priority,
        "declared": project.declared,
        "declarationOrder": project.declaration_order,
        "scoring": project.scoring,
    }


def test_root_observer_hashes_used_seats_once_from_a_snapshot(rules, monkeypatch):
    from baloot_platform.research import ortools_nonplay_observation as module
    prefix = _prefix()
    state = _state(rules)
    exact = _exact(state, rules, prefix=prefix, decision_kind="project")
    value = NonPlayDecisionObservationV1("root", "project", Seat.SOUTH, _root_public_state("project", ()))
    expected = build_nonplay_root_observer_v1(exact, prefix, "project")(value)
    original = module.canonical_observer_information_hash_v1
    calls = []

    def counted(**kwargs):
        calls.append(kwargs["acting_seat"])
        return original(**kwargs)

    monkeypatch.setattr(module, "canonical_observer_information_hash_v1", counted)
    observe = build_nonplay_root_observer_v1(exact, prefix, "project")
    assert calls == []
    state["hands"]["south"], state["hands"]["east"] = state["hands"]["east"], state["hands"]["south"]
    assert observe(value) == expected
    assert observe(value) == expected
    assert calls == ["south"]
    observe(replace(value, actor=Seat.NORTH))
    assert calls == ["south", "north"]


def test_root_project_observer_does_not_leak_private_project_identity(rules) -> None:
    prefix = _prefix()
    spades = _siri("S")
    hearts = _siri("H")
    first_state = _state(rules)
    second_state = copy.deepcopy(first_state)
    for key in ("initialHands", "hands"):
        hands = second_state[key]
        assert isinstance(hands, dict)
        hands["north"], hands["west"] = hands["west"], hands["north"]
    first = build_nonplay_root_observer_v1(
        _exact(first_state, rules, prefix=prefix, decision_kind="project"),
        prefix,
        "project",
    )
    second = build_nonplay_root_observer_v1(
        _exact(second_state, rules, prefix=prefix, decision_kind="project"),
        prefix,
        "project",
    )

    def observation(project: Project, node_id: str) -> NonPlayDecisionObservationV1:
        return NonPlayDecisionObservationV1(
            node_id,
            "project",
            Seat.SOUTH,
            {
                "stage": "project",
                "prefix": [
                    {
                        "actor": "north",
                        "action": "declare",
                        "projectId": project_id_v1(project),
                    }
                ],
                "projects": [_graph_project(project)],
            },
        )

    assert first(observation(spades, "node-one")) == second(
        observation(hearts, "node-two")
    )


def test_root_observer_rejects_state_prefix_hybrid(rules) -> None:
    prefix = _prefix()
    normalized = _exact(_state(rules), rules, prefix=prefix, decision_kind="bid")
    with pytest.raises(ValueError, match="ownership mismatch"):
        build_nonplay_root_observer_v1(
            normalized,
            _prefix(roundNumber=2),
            "bid",
        )


def test_root_observer_rejects_public_history_prefix_mismatch(rules) -> None:
    prefix = _prefix(
        auctionActions=[{"actor": "north", "action": "pass"}],
    )
    observer = build_nonplay_root_observer_v1(
        _exact(_state(rules), rules, prefix=prefix, decision_kind="bid"),
        prefix,
        "bid",
    )
    with pytest.raises(ValueError, match="disagrees with exact prefix"):
        observer(
            NonPlayDecisionObservationV1(
                "node",
                "bid",
                Seat.SOUTH,
                _root_public_state("bid", ()),
            )
        )


@pytest.mark.parametrize("index", [0, 1, 2])
def test_two_decision_project_frontier_preserves_actor_recall(root, index):
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    actors = ("north", "west", "south", "east")
    prefix = _prefix(
        auctionActions=[{"actor": "north", "action": "sun"}],
        projectActions=[{"actor": actor, "action": "pass"} for actor in actors[:index]],
        playActions=[{"actor": actor, "card": card, "ekka": False}
                     for actor, card in zip(actors[:index], ("S7", "S10", "SK"))],
    )
    decision = {"kind": "project", "selector": {"owner": actors[index]}}
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix, decision_point=decision)
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "bounded", "visibility": "complete_state",
        "claim": "outcome_reachable", "targetTeam": "north_south",
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": "test",
        "horizon": {"unit": "decisions", "value": 2}, "boundaryEvaluator": "current_raw_point_margin_v1",
    })

    def no_future_awards(*_args):
        raise AssertionError("short frontier requested unresolved future project awards")

    compiled = compile_nonplay_complete_graph_v1(
        rules, normalized, prefix, decision, request,
        maximum_nodes=100, maximum_depth=2, maximum_seconds=20, native_cp_sat=True,
        base_observer_factory=lambda value, exact: build_nonplay_root_observer_v1(value, exact, "project"),
        play_observer_factory=no_future_awards,
    )
    nodes = {node.node_id: node for node in compiled.graph.nodes}
    root_node = nodes[compiled.graph.root_node_id]
    observations = compiled.decision_observation_ids
    assert set(observations) == {key for key, node in nodes.items() if not node.is_leaf}
    for edge in root_node.edges:
        play_node = nodes[edge.child_node_id]
        assert play_node.actor_seat == actors[index]
        assert observations[play_node.node_id] == canonical_sha256({
            "kind": "native_project_followup_play_v1",
            "priorObservation": observations[root_node.node_id], "priorAction": edge.action_code,
        })
        assert all(nodes[choice.child_node_id].leaf_kind == "boundary" for choice in play_node.edges)

    # Change only opponents' unplayed cards, preserving the public purchase card.
    opponents = ("west", "east") if index % 2 == 0 else ("north", "south")
    hidden_cards = [
        next(card["id"] for card in normalized.state["hands"][seat]
             if card["id"] != normalized.state["purchaseCard"]["id"])
        for seat in opponents
    ]
    alternate = prefix.to_dict()
    alternate.pop("sha256")
    deck = alternate["deckOrder"]
    left, right = (deck.index(card) for card in hidden_cards)
    deck[left], deck[right] = deck[right], deck[left]
    alternate_prefix = ExactTracePrefixV1.build(alternate)
    alternate_state = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        alternate_prefix, decision_point=decision,
    )
    assert alternate_state.state_hash != normalized.state_hash
    assert alternate_state.prefix_hash != normalized.prefix_hash
    assert alternate_state.state["hands"][actors[index]] == normalized.state["hands"][actors[index]]
    assert alternate_state.legal_actions == normalized.legal_actions
    alternate_graph = compile_nonplay_complete_graph_v1(
        rules, alternate_state, alternate_prefix, decision, request,
        maximum_nodes=100, maximum_depth=2, maximum_seconds=20, native_cp_sat=True,
        base_observer_factory=lambda value, exact: build_nonplay_root_observer_v1(value, exact, "project"),
        play_observer_factory=no_future_awards,
    )

    def visible_choices(value):
        return sorted(
            (value.decision_observation_ids[node.node_id],
             tuple(sorted(edge.action_code for edge in node.edges)))
            for node in value.graph.nodes if not node.is_leaf
        )

    assert visible_choices(alternate_graph) == visible_choices(compiled)


@pytest.mark.parametrize("horizon,unit", [(3, "decisions"), (4, "decisions"), (4, "tricks")])
def test_project_frontier_reaches_next_actors_declaration(root, horizon, unit):
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix = _prefix(
        auctionActions=[{"actor": "north", "action": "sun"}],
        projectActions=[{"actor": "north", "action": "pass"}, {"actor": "west", "action": "pass"}],
        playActions=[{"actor": "north", "card": "S7", "ekka": False},
                     {"actor": "west", "card": "S10", "ekka": False}],
    )
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    normalized = normalizer.normalize_play_or_project(prefix)
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "bounded", "visibility": "complete_state",
        "claim": "outcome_reachable", "targetTeam": "north_south",
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": "test",
        "horizon": {"unit": unit, "value": 1 if unit == "tricks" else horizon},
        "boundaryEvaluator": "current_raw_point_margin_v1",
    })

    def play_observer(value, start):
        assert horizon == 4
        return build_nonplay_decision_observer_v1(
            rules=rules, normalized_exact=value, exact_prefix=start.prefix,
            decision_kind="project", projects=start.projects,
            native_cp_sat=True, project_evidence=start.project_evidence,
        )

    result = compile_nonplay_complete_graph_v1(
        rules, normalized, prefix, {"kind": "project", "selector": {"owner": "south"}}, request,
        maximum_nodes=200, maximum_depth=horizon, maximum_seconds=30, native_cp_sat=True,
        base_observer_factory=lambda value, exact: build_nonplay_root_observer_v1(value, exact, "project"),
        play_observer_factory=play_observer,
    )
    nodes = {node.node_id: node for node in result.graph.nodes}
    assert set(result.decision_observation_ids) == {key for key, node in nodes.items() if not node.is_leaf}
    if horizon == 4:
        assert any(node.utility != 0 for node in result.graph.nodes if node.is_leaf)
    for declaration in nodes[result.graph.root_node_id].edges:
        after_declaration = normalizer.append_normalized_action(prefix, normalized, declaration.action_code)
        play_state = normalizer.normalize_play_or_project(after_declaration)
        play = nodes[declaration.child_node_id]
        assert play.actor_seat == "south"
        for card in play.edges:
            after_card = normalizer.append_normalized_action(after_declaration, play_state, card.action_code)
            next_state = normalizer.normalize_play_or_project(after_card)
            next_declaration = nodes[card.child_node_id]
            assert next_declaration.actor_seat == next_state.next_actor == "east"
            assert tuple(edge.action_code for edge in next_declaration.edges) == tuple(
                action["code"] for action in next_state.legal_actions
            )
            for edge in next_declaration.edges:
                child = nodes[edge.child_node_id]
                if horizon == 3:
                    assert child.leaf_kind == "boundary"
                    continue
                after_next_declaration = normalizer.append_normalized_action(after_card, next_state, edge.action_code)
                final_play = normalizer.normalize_play_or_project(after_next_declaration)
                assert child.actor_seat == final_play.next_actor == "east"
                assert tuple(choice.action_code for choice in child.edges) == tuple(
                    action["code"] for action in final_play.legal_actions
                )
                assert all(nodes[choice.child_node_id].leaf_kind == "boundary" for choice in child.edges)

    with pytest.raises(RuntimeError, match="budget|maximum_depth"):
        compile_nonplay_complete_graph_v1(
            rules, normalized, prefix, {"kind": "project", "selector": {"owner": "south"}}, request,
            maximum_nodes=200, maximum_depth=2, maximum_seconds=30, native_cp_sat=True,
        )


@pytest.mark.parametrize("forced_last_card", [False, True])
def test_future_forced_pass_certificate_is_not_observer_information(root, monkeypatch, forced_last_card):
    from baloot_platform.research.ortools_nonplay_play_graft import OrToolsNonPlayPlayGraphGraftV1
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "bounded", "visibility": "complete_state",
        "claim": "outcome_reachable", "targetTeam": "north_south",
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": "test",
        "horizon": {"unit": "plies", "value": 4},
        "boundaryEvaluator": "current_raw_point_margin_v1",
    })
    observations, certificates = [], []
    for forced in (False, True):
        deck = list(range(32))
        swap = 1 if forced_last_card else 21
        deck[swap], deck[9] = deck[9], deck[swap]
        if forced:
            # Only North/East private holdings change; South cannot see this.
            deck[13], deck[29] = deck[29], deck[13]
        prefix = _prefix(deckOrder=deck,
            auctionActions=[{"actor": "north", "action": "sun"}],
            projectActions=[{"actor": seat, "action": "pass"} for seat in ("north", "west")],
            playActions=[{"actor": "north", "card": "S7", "ekka": False},
                         {"actor": "west", "card": "S10", "ekka": False}])
        normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
        certificates.append(normalized.normalization_receipt.to_dict()["nativePlayNormalization"]["forcedProjectPasses"]["seats"])
        def compile_graph():
            return compile_nonplay_complete_graph_v1(
                rules, normalized, prefix, {"kind": "project", "selector": {"owner": "south"}}, request,
                maximum_nodes=1000, maximum_depth=4, maximum_seconds=30, native_cp_sat=True,
                base_observer_factory=lambda value, exact: build_nonplay_root_observer_v1(value, exact, "project"),
                play_observer_factory=lambda value, start: build_nonplay_decision_observer_v1(
                    rules=rules, normalized_exact=value, exact_prefix=start.prefix, decision_kind="project",
                    projects=start.projects, native_cp_sat=True, project_evidence=start.project_evidence))

        result = compile_graph()
        normalized.require_integrity()
        if forced:
            with monkeypatch.context() as patch:
                patch.setattr(OrToolsNonPlayPlayGraphGraftV1, "_forced_project_sequence_graph", lambda *args, **kwargs: None)
                reference = compile_graph()

            def paths(compiled):
                lookup = {node.node_id: node for node in compiled.graph.nodes}
                def walk(node_id):
                    node = lookup[node_id]
                    if node.is_leaf:
                        return (node.leaf_kind, node.utility)
                    return (node.actor_seat, compiled.decision_observation_ids[node_id],
                            tuple((edge.action_code, walk(edge.child_node_id)) for edge in node.edges))
                return walk(compiled.graph.root_node_id)

            assert paths(result) == paths(reference)
            normalized.require_integrity()
        nodes = {node.node_id: node for node in result.graph.nodes}
        root_node = nodes[result.graph.root_node_id]
        observations.append({
            "project": result.decision_observation_ids[root_node.node_id],
            "play": {edge.action_code: result.decision_observation_ids[edge.child_node_id]
                     for edge in root_node.edges},
        })
    assert "east" not in certificates[0] and "east" in certificates[1]
    assert observations[0] == observations[1]


def test_real_observers_cover_optional_project_root_and_play_graft(
    root: Path, monkeypatch,
) -> None:
    rules = RuleBook.load(
        root / "configs/rules/saudi_project_optional_unratified_v1.yaml"
    )
    prefix = _prefix(
        auctionActions=[{"actor": "north", "action": "sun"}],
        projectActions=[
            {"actor": actor, "action": "pass"} for actor in ("north", "west", "south")
        ],
        playActions=[
            {"actor": "north", "card": "S7", "ekka": False},
            {"actor": "west", "card": "S10", "ekka": False},
            {"actor": "south", "card": "SK", "ekka": False},
        ],
    )
    decision = {
        "kind": "project",
        "selector": {"owner": "east", "projectIndex": 0},
    }
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point=decision, maximum_seconds=15
    )
    request = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "bounded",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -100_000,
            "horizon": {"unit": "plies", "value": 2},
            "boundaryEvaluator": "current_raw_point_margin_v1",
            "budgetProfile": "test",
        }
    )

    captures = []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native observer called Python project scoring")

    monkeypatch.setattr(ProjectDomainIRV1, "score_team", forbidden)

    def play_factory(play_normalized, start):
        captures.append((play_normalized, start))
        return build_nonplay_decision_observer_v1(
            rules=rules,
            normalized_exact=play_normalized,
            exact_prefix=start.prefix,
            decision_kind="project",
            projects=start.projects,
            native_cp_sat=True,
            project_evidence=start.project_evidence,
        )

    # Exercise the generic evidence projector and its corruption boundaries.
    # Final-play parent projection parity is covered by the graph module.
    from baloot_platform.research.ortools_nonplay_play_graft import OrToolsNonPlayPlayGraphGraftV1
    monkeypatch.setattr(OrToolsNonPlayPlayGraphGraftV1, "_final_project_play_graph",
                        lambda *args, **kwargs: None)
    compiled = compile_nonplay_complete_graph_v1(
        rules,
        normalized,
        prefix,
        decision,
        request,
        maximum_nodes=1_000,
        maximum_depth=4,
        maximum_seconds=15,
        native_cp_sat=True,
        base_observer_factory=lambda value, exact: (
            build_nonplay_root_observer_v1(value, exact, "project")
        ),
        play_observer_factory=play_factory,
    )
    decisions = {node.node_id for node in compiled.graph.nodes if not node.is_leaf}
    assert set(compiled.decision_observation_ids) == decisions
    assert compiled.graph.root_node_id in decisions
    assert any(node_id.startswith("graft-") for node_id in decisions)
    value, start = next(item for item in captures if item[1].projects)
    arguments = dict(rules=rules, normalized_exact=value, exact_prefix=start.prefix,
                     decision_kind="project", projects=start.projects,
                     native_cp_sat=True, project_evidence=start.project_evidence)
    with pytest.raises(ValueError, match="requires CP project evidence"):
        build_nonplay_decision_observer_v1(**{**arguments, "project_evidence": None})
    with pytest.raises(ValueError, match="project evidence checksum"):
        build_nonplay_decision_observer_v1(**{
            **arguments, "project_evidence": replace(start.project_evidence, evidence_hash="0" * 64),
        })
    with pytest.raises(ValueError, match="disagree with native CP project evidence"):
        build_nonplay_decision_observer_v1(**{
            **arguments, "projects": (replace(start.projects[0], scoring=not start.projects[0].scoring), *start.projects[1:]),
        })
    receipt = value.normalization_receipt.to_dict()
    receipt["rulesHash"] = "0" * 64
    receipt.pop("sha256")
    receipt["sha256"] = canonical_sha256(receipt)
    with pytest.raises(ValueError, match="receipt ownership"):
        build_nonplay_decision_observer_v1(**{
            **arguments, "normalized_exact": replace(value, normalization_receipt=OrToolsExactNormalizationReceiptV1(receipt)),
        })
    evidence = replace(start.project_evidence, card_owners=tuple(reversed(start.project_evidence.card_owners)))
    document = evidence.to_dict()
    document.pop("evidenceHash")
    evidence = replace(evidence, evidence_hash=canonical_sha256(document))
    with pytest.raises(ValueError, match="project evidence ownership"):
        build_nonplay_decision_observer_v1(**{**arguments, "project_evidence": evidence})
