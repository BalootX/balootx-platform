from baloot_platform.analysis.hidden_deals import (
    _build_start, _build_start_reference, generate_hidden_deals,
)
from baloot_platform.analysis.ismcts_v8 import (
    analyze_information_set_mcts, information_set_key,
)
from baloot_platform.analysis.models import InformationState
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.cards import FULL_DECK
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.core.exact_state_view import build_canonical_observer_information_v1
from baloot_platform.core.projects import project_card_patterns
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.serialization import replay_trusted_context
from baloot_platform.learning.store import LearningStore, _hash
from baloot_platform.research.scenario_contracts import canonical_sha256


def test_learning_connections_commit_rollback_and_close(tmp_path):
    import sqlite3
    import pytest
    store = LearningStore(tmp_path / "learning.sqlite3")
    with store._connect() as connection:
        connection.execute("CREATE TABLE close_probe(value INTEGER)")
        connection.execute("INSERT INTO close_probe VALUES (1)")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with pytest.raises(RuntimeError):
        with store._connect() as connection:
            connection.execute("INSERT INTO close_probe VALUES (2)")
            raise RuntimeError("rollback")
    with store._connect() as connection:
        assert [row[0] for row in connection.execute("SELECT value FROM close_probe")] == [1]
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    store_path = tmp_path / "learning.sqlite3"
    store_path.rename(tmp_path / "closed.sqlite3")


def test_mcts_key_does_not_build_a_full_information_state(root, monkeypatch):
    _, state = _rank_world(root)
    expected = information_set_key(state)
    def unnecessary(*args, **kwargs):
        raise AssertionError("MCTS keys need only the shared project projection")
    monkeypatch.setattr(InformationState, "from_complete", unnecessary)
    assert information_set_key(state) == expected


def test_regenerated_comparison_only_hashes_and_learning_rows(root, tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from baloot_platform.storage.postgres_learning import PostgresLearningStore
    import json
    expected = json.loads((root / "tests/fixtures/informed-project-visibility-hashes.json").read_text(encoding="utf-8"))
    store = LearningStore(tmp_path / "learning.sqlite3")
    for high in (False, True):
        engine, state = _rank_world(root, high=high)
        info = InformationState.from_complete(state, Seat.WEST).to_dict()
        legacy = json.loads(json.dumps(info))
        for public in legacy["projects"]:
            public["highRank"] = next(p.high_rank for p in state.projects
                if p.owner.name.lower() == public["owner"] and p.kind == public["kind"])
        key = "high" if high else "low"
        assert canonical_sha256(info) == expected[key]["canonical_sha256"]
        assert canonical_sha256(legacy) == expected[key]["legacy_canonical_sha256"]
        assert info != legacy
        legal = engine.legal_actions(state)
        recorded = store.record_decision(game_id=f"rank-world-{key}", game_version=1,
            mode="sun", state=state, actor=Seat.WEST, decision_kind="play",
            legal_actions=legal, selected_action=legal[0])
        row = store.get_decision(recorded["decisionId"])
        assert row["publicState"] == info
        identity = dict(gameId=row["gameId"], gameVersion=row["gameVersion"], ordinal=row["ordinal"],
            kind=row["decisionKind"], actor=row["actor"], publicState=info,
            legalActions=row["legalActions"], selectedAction=row["selectedAction"],
            decisionMetadata=row["decisionMetadata"])
        assert row["decisionHash"] == _hash(identity) == expected[key]["decision_hash"]
        assert row["decisionId"] == f"DEC-{_hash(identity)[:20].upper()}"
        inserts = []
        postgres = object.__new__(PostgresLearningStore)
        postgres._one = lambda *args: {"value": 1}
        connection = SimpleNamespace(transaction=lambda: nullcontext(),
            execute=lambda query, values: inserts.append(values))
        postgres.pool = SimpleNamespace(connection=lambda: nullcontext(connection))
        pg_row = postgres.record_decision(game_id=f"rank-world-{key}", game_version=1,
            mode="sun", state=state, actor=Seat.WEST, decision_kind="play",
            legal_actions=legal, selected_action=legal[0])
        assert json.loads(inserts[0][9]) == info
        assert pg_row["decisionHash"] == row["decisionHash"]


def _rank_world(root, *, high=False):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v7.yaml")
    hands = {
        "north": "S7 H7 H8 H9 D7 D9 CJ CK".split(),
        "west": "S8 S9 S10 SJ SQ SK SA C7".split(),
        "south": "H10 HJ HQ HK HA D8 D10 DJ".split(),
    }
    if high:
        hands["north"][1], hands["south"][0] = hands["south"][0], hands["north"][1]
    used = {card for hand in hands.values() for card in hand}
    hands["east"] = [card.code for card in FULL_DECK if card.code not in used]
    state = replay_trusted_context(dict(
        dealer="east", contract="sun", buyer="north", firstLeader="north",
        initialHands=hands, completedTricks=[], currentTrick=None,
    ), rules)
    engine = BalootEngine(rules)
    engine.play_card(state, "S7")
    assert state.current_player == Seat.WEST
    return engine, state


def test_each_actor_input_matches_canonical_rank_disclosure_through_a_full_round(root):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v7.yaml")
    engine = BalootEngine(rules)
    state = engine.create_round(seed=20261025, dealer=Seat.WEST)
    engine.shuffle_and_initial_deal(state)
    assert engine.bid_and_complete_deal(state)
    engine.prepare_projects(state)
    assert state.projects
    bot = HeuristicBot(engine, BotProfile(20262024))
    actors, hidden_count, visible_count = set(), 0, 0
    while state.current_player is not None:
        actor = state.current_player
        actors.add(actor)
        info = InformationState.from_complete(state, actor)
        canonical = build_canonical_observer_information_v1(
            complete_state=state.to_dict(), acting_seat=actor.name.lower(),
        )["publicState"]
        expected = sorted((p["owner"], p["kind"], p["value"], p["highRank"])
                          for p in canonical["projects"]
                          if p["declarationOrder"] < 999 or p["revealed"])
        actual = sorted((p.owner.name.lower(), p.kind, p.value, p.high_rank)
                        for p in info.projects)
        assert actual == expected
        hidden_count += sum(p.high_rank is None for p in info.projects)
        visible_count += sum(p.high_rank is not None for p in info.projects)
        runtime = engine.observer_view(state, actor)
        assert sorted((p["owner"], p["kind"], p["value"], p["highRank"])
                      for p in runtime["projects"]
                      if p["declarationOrder"] < 999 or p["revealed"]) == expected
        assert runtime["currentTrick"] == canonical["currentTrick"]
        if len(state.played_cards) == 12:
            hands = {seat: state.hand(seat) for seat in Seat}
            for rebuild in (_build_start_reference, _build_start):
                reconstructed = rebuild(info, hands, engine)
                assert reconstructed.project_rank_visibility == state.project_rank_visibility
                assert InformationState.from_complete(reconstructed, actor).projects == info.projects
        engine.play_card(state, bot.choose(state))
    assert actors == set(Seat)
    assert hidden_count > 0 and visible_count > 0


def test_hidden_rank_worlds_share_input_and_mcts_key_but_own_rank_is_known(root):
    engine, low = _rank_world(root)
    _, high = _rank_world(root, high=True)
    assert low.projects[0].high_rank != high.projects[0].high_rank
    first = InformationState.from_complete(low, Seat.WEST)
    second = InformationState.from_complete(high, Seat.WEST)
    assert first.to_dict() == second.to_dict()
    assert next(p for p in first.projects if p.owner == Seat.NORTH).high_rank is None
    assert next(p for p in InformationState.from_complete(low, Seat.NORTH).projects
                if p.owner == Seat.NORTH).high_rank is not None
    assert information_set_key(low) == information_set_key(high)
    for state in (low, high):
        hands = {seat: state.hand(seat) for seat in Seat}
        rebuilt = _build_start_reference(first, hands, engine)
        assert InformationState.from_complete(rebuilt, Seat.WEST).to_dict() == first.to_dict()


def test_sampler_and_mcts_do_not_recover_or_condition_on_hidden_rank(root):
    engine, low = _rank_world(root)
    _, high = _rank_world(root, high=True)
    first = InformationState.from_complete(low, Seat.WEST)
    second = InformationState.from_complete(high, Seat.WEST)
    cohort = generate_hidden_deals(first, engine, requested_samples=8,
                                  exact_allocation_limit=1, maximum_attempts=100, seed=91)
    assert len(cohort.states) == 8
    assert len({next(p.high_rank for p in state.projects if p.owner == Seat.NORTH and p.declared)
                for state in cohort.states}) > 1
    for state in cohort.states:
        assert InformationState.from_complete(state, Seat.WEST).to_dict() == first.to_dict()
    settings = dict(particles=2, exact_allocation_limit=1, iterations=8,
                    maximum_tree_depth=2, seed=91)
    a = analyze_information_set_mcts(first, engine, **settings)
    b = analyze_information_set_mcts(second, engine, **settings)
    assert a.recommended_card == b.recommended_card
    assert a.actions == b.actions


def test_known_comparison_rank_is_retained_and_lower_category_rank_stays_hidden(root):
    engine, state = _rank_world(root)
    bot = HeuristicBot(engine, BotProfile(1))
    while len(state.played_cards) < 4:
        engine.play_card(state, bot.choose(state))
    info = InformationState.from_complete(state, Seat.WEST)
    assert next(p for p in info.projects if p.owner == Seat.NORTH).high_rank is None
    assert next(p for p in info.projects if p.owner == Seat.SOUTH).high_rank == 7


def test_unknown_rank_patterns_are_the_union_of_all_governed_rank_patterns(root):
    engine, _ = _rank_world(root)
    query = dict(kind="siri", value=4, contract=Contract.SUN, rules=engine.rules)
    expected = tuple(sorted({pattern for rank in range(8)
                             for pattern in project_card_patterns(**query, high_rank=rank)}))
    assert project_card_patterns(**query, high_rank=None) == expected
