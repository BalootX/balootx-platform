from __future__ import annotations

from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from baloot_platform.agents.profiles import (
    AdaptivePlayStrategy,
    AgentProfileRegistry,
    primary_play_decision_score_authority_v1,
)
from baloot_platform.analysis.decision_scores import canonical_sha256_v1
from baloot_platform.analysis.models import InformationState
from baloot_platform.analysis.search import SearchTimeBudgetExceeded, exact_team_value
from baloot_platform.analysis.search_backend import NativeExactSearchBackendV3, PythonReferenceSearchBackend
from baloot_platform.challenge.evidence_v3 import DecisionLedgerV1, canonical_sha256
from baloot_platform.challenge.miner import MiningRequest, ProgressiveChallengeMiner
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.challenge.trace import CompleteGameTraceV1
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.scenario import ScenarioRequest, generate_episode
from baloot_platform.learning.store import LearningStore
from baloot_platform.native.accelerator import NativeUnavailable
from baloot_platform.service.game import GameManager, GameSession


@pytest.fixture
def endgame(rules):
    engine = BalootEngine(rules)
    state = engine.create_round(seed=24001, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state, cut_position=9)
    engine.bid_and_complete_deal(
        state, forced_contract=Contract.HOKUM,
        forced_buyer=Seat.SOUTH, forced_trump=Suit.CLUBS,
    )
    engine.prepare_projects(state)
    bot = HeuristicBot(engine, BotProfile(deterministic_seed=24001))
    while sum(hand.bit_count() for hand in state.hands) > 8:
        engine.play_card(state, bot.choose(state))
    assert len(engine.legal_moves(state)) > 1
    return engine, state


@pytest.fixture
def expired_search(endgame, monkeypatch):
    engine, state = endgame
    clock = count(0.0, 2.0)
    monkeypatch.setattr("baloot_platform.analysis.search.monotonic", lambda: next(clock))

    def expire(*_args, **_kwargs):
        return exact_team_value(state, engine, state.current_player.team, max_seconds=1.0)

    return expire


def test_live_advance_bots_records_time_censored_fallback(
    root, tmp_path, endgame, expired_search,
):
    engine, state = endgame
    legal = tuple(engine.legal_moves(state))
    store = LearningStore(tmp_path / "learning.sqlite3")
    manager = GameManager(root, learning_store=store)
    game = GameSession(
        "timed-live-game", "full", engine, state, None,
        agent_profiles={seat.name.lower(): "expert" for seat in Seat},
    )
    manager.import_session(game)
    version = game.version

    advanced = manager.advance_bots(game.id, stop_before=None, maximum_actions=8)

    assert advanced.version == version + 8
    assert advanced.state.current_player is None
    assert advanced.state.round_score is not None
    decisions = store.list_decisions(game.id)
    decision = decisions[0]
    trace = decision["decisionMetadata"]
    assert trace["analysisSolverCompletion"] == "time_censored"
    assert trace["analysisCompletionStatuses"] == ["time_censored"]
    assert "search_fallback:SearchTimeBudgetExceeded" in trace["reasonTags"]
    assert trace["decisionSource"] == "transparent_heuristic"
    assert trace["observerSafe"] is True
    assert trace["analysisExhaustiveSupport"] is False
    assert trace["analysisPopulationCoverageComplete"] is False
    assert trace["selectedAction"] in [card.to_dict() for card in legal]
    with pytest.raises(SearchTimeBudgetExceeded, match="time-censored"):
        store.decision_ledger(str(advanced.game_hash))


@pytest.mark.parametrize("profile_id", [
    "expert", "expert_deterministic", "deceptive", "solver_fast",
    "finalist_candidate", "foundry_balanced",
])
def test_search_profiles_fail_closed_by_default(root, endgame, expired_search, profile_id):
    engine, state = endgame
    strategy = AdaptivePlayStrategy(AgentProfileRegistry.from_bundle(root).get(profile_id), {})
    with pytest.raises(SearchTimeBudgetExceeded, match="without canonical output"):
        strategy.choose_with_trace(engine, state)


@pytest.mark.real_clock
@pytest.mark.parametrize("allow_fallback", [False, True])
def test_native_timeout_uses_the_same_live_opt_in(root, endgame, monkeypatch, allow_fallback):
    engine, state = endgame
    strategy = AdaptivePlayStrategy(AgentProfileRegistry.from_bundle(root).get("expert"), {})
    monkeypatch.setattr(
        "baloot_platform.agents.profiles.analyze", Mock(side_effect=TimeoutError("native timed out")),
    )
    if not allow_fallback:
        with pytest.raises(TimeoutError, match="native timed out"):
            strategy.choose_with_trace(engine, state)
    else:
        _card, trace = strategy.choose_with_trace(engine, state, timeout_fallback=True)
        assert trace["analysisSolverCompletion"] == "time_censored"
        assert "search_fallback:TimeoutError" in trace["reasonTags"]


def test_scenario_does_not_skip_a_timed_out_seed(root, endgame, expired_search, monkeypatch):
    engine, _state = endgame
    collect = Mock(side_effect=expired_search)
    monkeypatch.setattr("baloot_platform.core.scenario._collect_candidates", collect)
    with pytest.raises(SearchTimeBudgetExceeded, match="without canonical output"):
        generate_episode(
            engine.rules, {}, bundle_hash="test",
            request=ScenarioRequest.governed(
                contract=Contract.SUN, buyer=Seat.SOUTH,
                require_project=False, max_seed_attempts=2,
            ),
        )
    assert collect.call_count == 1


@pytest.mark.real_clock
@pytest.mark.parametrize("stage", ["generation", "deep", "history"])
def test_miner_does_not_reject_and_continue_after_search_timeout(
    root, endgame, expired_search, monkeypatch, stage,
):
    store = SimpleNamespace(
        list_challenges=lambda **_kwargs: [],
        learning_queue=lambda **_kwargs: [{"decision": {"decisionId": "test", "decisionKind": "play"}}],
    )
    learning = SimpleNamespace(reanalyze_decision=expired_search)
    miner = ProgressiveChallengeMiner(root, store, learning=learning)
    request = MiningRequest(
        source="history" if stage == "history" else "simulation",
        decision_kind="play", seed_start=1, seeds=2, contract="sun",
        buyer_relationship="any", archetype="any", difficulty_min=0, difficulty_max=100,
        minimum_publishability=0, maximum_results=1, require_project=False,
        pause_trick_min=1, pause_trick_max=7, maximum_seconds=60,
        player_profiles=(), shallow_samples=None, deep_samples=None,
        exact_allocation_limit=None, exact_search_remaining=None, content_contract="any",
    )
    episode = SimpleNamespace(analysis=SimpleNamespace(actions=(1, 2)))
    monkeypatch.setattr(
        "baloot_platform.challenge.miner.generate_episode",
        expired_search if stage == "generation" else lambda *_args, **_kwargs: episode,
    )
    monkeypatch.setattr(miner, "_deep_confirm", expired_search)
    monkeypatch.setattr(miner, "_relationship_matches", lambda *_args: True)
    monkeypatch.setattr(miner, "_vector", lambda *_args: SimpleNamespace(
        cliff_score=1, debate_score=1, comeback_score=1, surprise_score=1, publishability_score=1,
    ))

    with pytest.raises(SearchTimeBudgetExceeded, match="without canonical output"):
        miner.mine(request)


@pytest.mark.real_clock
@pytest.mark.parametrize("operation", ["search", "heuristic_rollout_batch", "heuristic_rollout_batch_compact"])
def test_native_search_does_not_retry_a_wall_clock_timeout(
    endgame, expired_search, monkeypatch, operation,
):
    engine, state = endgame
    backend = NativeExactSearchBackendV3(SimpleNamespace(
        supports_v3=True, supports_compact_rollout=True, exact_team_value=expired_search,
        heuristic_rollout_batch=expired_search, heuristic_rollout_batch_compact=expired_search,
    ))
    reference = Mock(side_effect=AssertionError("native timeout must not select a reference retry"))
    monkeypatch.setattr(backend, "_reference", reference)

    with pytest.raises(SearchTimeBudgetExceeded, match="without canonical output"):
        if operation == "search":
            backend.search(
                state, engine, state.current_player.team,
                max_nodes=1000, max_seconds=1.0, max_depth=None,
                max_cache_entries=1000, cancel_check=None,
            )
        else:
            getattr(backend, operation)(
                (state,), (None,), (1,), engine, state.current_player.team, cancel_check=None,
            )
    reference.assert_not_called()


@pytest.mark.real_clock
@pytest.mark.parametrize("refusal", [NativeUnavailable, ValueError, OSError, "partial"])
def test_native_search_keeps_non_timeout_reference_fallback(endgame, monkeypatch, refusal):
    engine, state = endgame

    def native(*_args, **_kwargs):
        if refusal == "partial":
            return SimpleNamespace(exact=False)
        raise refusal("native refused")

    backend = NativeExactSearchBackendV3(SimpleNamespace(
        supports_v3=True, supports_compact_rollout=False, exact_team_value=native,
    ))
    reference = Mock()
    monkeypatch.setattr(backend, "_reference", reference)
    result = backend.search(
        state, engine, state.current_player.team,
        max_nodes=1000, max_seconds=1.0, max_depth=None,
        max_cache_entries=1000, cancel_check=None,
    )
    assert result is reference.return_value
    reference.assert_called_once()


@pytest.mark.real_clock
def test_card_only_wrapper_rejects_live_timeout_fallback(root, endgame, expired_search):
    engine, state = endgame
    strategy = AdaptivePlayStrategy(AgentProfileRegistry.from_bundle(root).get("expert"), {})
    with pytest.raises(SearchTimeBudgetExceeded, match="time-censored"):
        strategy.choose_card(engine, state, timeout_fallback=True)


@pytest.mark.real_clock
def test_score_evidence_rejects_timeout_even_with_live_opt_in(root, endgame, expired_search):
    engine, state = endgame
    profile = AgentProfileRegistry.from_bundle(root).get("foundry_balanced")
    authority = primary_play_decision_score_authority_v1(
        actor=state.current_player,
        public_prefix_hash=canonical_sha256_v1(
            InformationState.from_complete(state, state.current_player).to_dict()
        ),
        legal_cards=tuple(engine.legal_moves(state)), rules_hash=engine.rules.sha256,
        analysis_identity_hash="3" * 64, trace_analysis_profile_hash="2" * 64,
        cohort_index=0, cohort_seed=991, objective=profile.play_search_objective,
        backend=PythonReferenceSearchBackend.descriptor.id,
    )
    with pytest.raises(SearchTimeBudgetExceeded, match="without canonical output"):
        AdaptivePlayStrategy(profile, {}).choose_with_trace(
            engine, state, seed=991, primary_decision_score_authority=authority,
            timeout_fallback=True,
        )


@pytest.mark.parametrize("channel", ["state", "ledger"])
def test_canonical_evidence_rejects_time_censored_live_policy(endgame, channel):
    engine, state = endgame
    action = engine.legal_moves(state)[0].to_dict()
    policy = {"analysisSolverCompletion": "time_censored"}
    record = {
        "decisionId": "test", "ordinal": 1, "actor": state.current_player.name.lower(),
        "decisionKind": "play", "legalActions": [card.to_dict() for card in engine.legal_moves(state)],
        "selectedAction": action,
    }
    trusted = {
        "rulesHash": engine.rules.sha256,
        "shuffle": {"algorithm": "test", "deckCommitment": "c" * 64},
        "decisionPolicyTraces": [{"ordinal": 1, "trace": policy}],
    }
    game = {
        "gameHash": "a" * 64, "gameId": "test", "stateHash": canonical_sha256(trusted),
        "bundleHash": "b" * 64, "engineVersion": "test",
    }
    ledger = DecisionLedgerV1.build(game["gameHash"], [record])
    with pytest.raises(SearchTimeBudgetExceeded, match="time-censored"):
        if channel == "state":
            CompleteGameTraceV1.build(game_record=game, trusted_final_state=trusted, decision_ledger=ledger)
        else:
            DecisionLedgerV1.build(game["gameHash"], [{**record, "decisionMetadata": policy}])


def test_simulation_cannot_hash_a_time_censored_policy(root, monkeypatch):
    def choose(_strategy, engine, state, legal_cards=None, **_kwargs):
        return engine.legal_moves(state)[0], {"analysisSolverCompletion": "time_censored"}

    monkeypatch.setattr(AdaptivePlayStrategy, "choose_with_trace", choose)
    simulator = NaturalLaffaSimulatorV1.from_root(root, profile_id="foundry_throughput")
    with pytest.raises(SearchTimeBudgetExceeded, match="time-censored"):
        simulator.simulate(202608050476)


@pytest.mark.real_clock
@pytest.mark.parametrize("consumer", ["finalist", "symmetry", "paired", "holdout"])
def test_evidence_consumers_reject_time_censored_trace(root, endgame, monkeypatch, consumer):
    from baloot_platform.agents.symmetry_qualification import _ranked_codes
    from baloot_platform.benchmark.paired_games import _choose
    from baloot_platform.challenge.finalist_agent_agreement import FinalistAgentAgreementV1
    from scripts.benchmark_permanent_holdout import _completion

    engine, state = endgame
    trace = {"analysisSolverCompletion": "time_censored", "decisionSource": "transparent_heuristic"}
    monkeypatch.setattr(
        AdaptivePlayStrategy, "choose_with_trace",
        lambda *_args, **_kwargs: (engine.legal_moves(state)[0], trace),
    )
    with pytest.raises(SearchTimeBudgetExceeded, match="time-censored"):
        if consumer == "finalist":
            agreement = FinalistAgentAgreementV1.__new__(FinalistAgentAgreementV1)
            agreement._trace_with_analysis_identity(trace, cohort_index=0)
        elif consumer == "symmetry":
            _ranked_codes(trace)
        elif consumer == "holdout":
            _completion(trace)
        else:
            _choose(
                engine=engine, state=state, policy_id="expert",
                registry=AgentProfileRegistry.from_bundle(root), convention={}, seed=1, cancel_check=None,
            )
