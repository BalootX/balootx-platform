from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from baloot_platform.analysis.models import InformationState
from baloot_platform.analysis.hidden_deals import HiddenDealSet
from baloot_platform.core.cards import Card, cards_to_mask
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, GamePhase, Seat, Suit
from baloot_platform.core.models import RoundState, Trick
from baloot_platform.research.exact_state import ExactStateNormalizerV1
from baloot_platform.research.proof_service import (
    ProofBudgetExceeded,
    ScenarioProofBudgetRegistryV1,
    ScenarioProofBudgetV1,
    UnifiedScenarioProofServiceV1,
    decode_proof_play_action_v1,
)
from baloot_platform.research.scenario_contracts import (
    ProofOutcomeV1,
    ScenarioProofBudgetV1 as ContractProofBudgetV1,
    ScenarioProofRequestV1,
    canonical_sha256,
)


def _fixture(root: Path) -> dict:
    return json.loads(
        (root / "examples" / "research-scenario-v4-unique-saving.json").read_text(
            encoding="utf-8"
        )
    )


def _budget(profile_id: str = "test_proof_v1", **overrides) -> ScenarioProofBudgetV1:
    values = {
        "maximumNodes": 100_000,
        "maximumDepth": 16,
        "maximumHorizonActions": 16,
        "maximumHiddenWorlds": 16,
        "maximumWallSeconds": 30.0,
        "maximumCpuSeconds": 30.0,
        "maximumCacheEntries": 10_000,
        "maximumCacheBytes": 8_388_608,
        "cancellationCheckIntervalNodes": 8,
        "maximumWorkers": 1,
        "maximumLineCountNodes": 100_000,
        "maximumStateSpace": 10_000_000,
    }
    values.update(overrides)
    return ScenarioProofBudgetV1.build(profile_id, values)


@pytest.mark.real_clock
@pytest.mark.parametrize("field", ["maximumWallSeconds", "maximumCpuSeconds"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "30", 0, -1])
def test_budget_rejects_nonfinite_or_coercive_deadlines(field, value):
    with pytest.raises(ValueError, match="bound is invalid"):
        _budget(**{field: value})


def test_budget_contract_is_shared_without_changing_legacy_identity():
    assert ScenarioProofBudgetV1 is ContractProofBudgetV1
    budget = _budget()
    registry = ScenarioProofBudgetRegistryV1({budget.profile_id: budget}, "a" * 64)
    raw = registry.to_dict()["profiles"][budget.profile_id]
    assert raw.pop("sha256") == budget.sha256
    assert raw.pop("profileId") == budget.profile_id
    assert ContractProofBudgetV1.build(budget.profile_id, raw) == budget


@pytest.mark.parametrize("version", [1, 2, 3])
def test_native_budget_round_trip_and_identity(version):
    from baloot_platform.service.api import NativeObservedProofLimitsV1Response, NativeObservedProofLimitsV2Response, NativeObservedProofLimitsV3Response, ScenarioProofBudgetV1Response
    response_type = {1: NativeObservedProofLimitsV1Response, 2: NativeObservedProofLimitsV2Response, 3: NativeObservedProofLimitsV3Response}[version]
    native = {"schemaVersion": 1, "maximumIterations": 20, "maximumQueries": 40,
              "maximumObservations": 60, "maximumBoundPrefixes": 30,
              "maximumCachedPrefixesPerBranch": 10}
    if version >= 2:
        native.update(schemaVersion=version, counterexampleObjective="minimum_utility")
    if version == 3:
        native.update(playAllocationMethod="deterministic_cp_sat_batched_prefix_v1")
    legacy = _budget()
    budget = _budget(nativeObserved=native)
    assert budget.native_observed.to_dict() == native
    assert response_type.model_validate(native).model_dump() == native
    assert budget.sha256 != legacy.sha256 and legacy.native_observed is None
    raw = ScenarioProofBudgetRegistryV1({budget.profile_id: budget}, "a" * 64).to_dict()["profiles"][budget.profile_id]
    assert ScenarioProofBudgetV1Response.model_validate(raw).model_dump()["nativeObserved"] == native
    raw.pop("profileId")
    raw.pop("sha256")
    assert ContractProofBudgetV1.build(budget.profile_id, raw) == budget
    for field in native:
        for invalid in (True, "1", 0, -1, 2**31, float("inf")):
            with pytest.raises(ValueError):
                _budget(nativeObserved={**native, field: invalid})
            with pytest.raises(ValueError):
                response_type.model_validate({**native, field: invalid})
    with pytest.raises(ValueError):
        _budget(nativeObserved={**native, "unknown": 1})


def _service(budget: ScenarioProofBudgetV1) -> UnifiedScenarioProofServiceV1:
    return UnifiedScenarioProofServiceV1(
        ScenarioProofBudgetRegistryV1(
            {budget.profile_id: budget},
            canonical_sha256({"profiles": [budget.sha256]}),
        )
    )


def _request(fixture: dict, budget: ScenarioProofBudgetV1, **overrides):
    value = dict(fixture["proof"])
    value["budgetProfile"] = budget.profile_id
    value.update(overrides)
    if value.get("visibility") == "actor_information_set":
        value.setdefault(
            "strategyClass",
            "frozen_policy"
            if value.get("mode") == "policy_conditioned"
            else "pure_deterministic",
        )
    return ScenarioProofRequestV1.build(value)


def test_actor_information_set_exact_uses_one_strategy_across_worlds(
    rules,
    root: Path,
):
    source = ExactStateNormalizerV1(rules).normalize(
        _fixture(root)["exactTracePrefix"]
    )
    engine = BalootEngine(rules)

    def world(east: list[str], west: list[str]):
        hands = {
            Seat.SOUTH: ["H9", "D8"],
            Seat.NORTH: ["CA", "S10"],
            Seat.EAST: east,
            Seat.WEST: west,
        }
        state = RoundState(rules.sha256, 1, 1, Seat.EAST)
        state.contract = Contract.SUN
        state.buyer = state.declarer = Seat.SOUTH
        state.first_leader = state.current_player = Seat.SOUTH
        state.phase = GamePhase.PLAYING
        state.current_trick = Trick(6, Seat.SOUTH)
        remaining = [Card.parse(code) for codes in hands.values() for code in codes]
        remaining_raw = sum(rules.points(card, Contract.SUN, None) for card in remaining)
        prior_raw = (
            engine.scoring_domain.raw_card_totals["sun"]
            - engine.scoring_domain.last_trick_bonus_raw
            - remaining_raw
        )
        state.completed_tricks = [
            Trick(
                index,
                Seat.NORTH if index % 2 == 0 else Seat.WEST,
                winner=Seat.NORTH if index % 2 == 0 else Seat.WEST,
                raw_points=prior_raw if index == 0 else 0,
            )
            for index in range(6)
        ]
        state.hands = [
            cards_to_mask(Card.parse(code) for code in hands[seat])
            for seat in Seat
        ]
        return UnifiedScenarioProofServiceV1._normalized_hidden_world(source, state)

    worlds = (
        world(["SK", "HA"], ["C7", "C8"]),
        world(["C7", "HA"], ["SK", "C8"]),
    )
    root_actor = worlds[0].round_state.current_player
    assert root_actor == Seat.SOUTH
    observation_hash = canonical_sha256(
        InformationState.from_complete(worlds[0].round_state, root_actor).to_dict()
    )
    assert observation_hash == canonical_sha256(
        InformationState.from_complete(worlds[1].round_state, root_actor).to_dict()
    )
    after_root = []
    for item in worlds:
        state = item.round_state.clone()
        engine.play_card(state, Card.parse("H9"), emit_events=False)
        engine.play_card(state, Card.parse("HA"), emit_events=False)
        after_root.append(state)
    assert canonical_sha256(
        InformationState.from_complete(after_root[0], Seat.NORTH).to_dict()
    ) == canonical_sha256(
        InformationState.from_complete(after_root[1], Seat.NORTH).to_dict()
    )
    assert {
        card.code for card in engine.legal_actions(after_root[0])
    } == {"CA", "S10"}

    budget = _budget(
        maximumNodes=500_000,
        maximumHiddenWorlds=2,
        maximumLineCountNodes=500_000,
    )
    request = _request(
        _fixture(root),
        budget,
        visibility="actor_information_set",
        claim="selected_action_is_unique_winner",
        targetTeam="north_south",
        successThreshold=10,
        selectedAction="D8",
        lineUniverse="optimal_strategy",
    )
    coverage = {
        "method": "exhaustive_reference_fixture",
        "complete": True,
        "expectedWorldCount": 2,
        "observationHash": observation_hash,
    }
    service = _service(budget)
    with pytest.raises(ValueError, match="must be generated internally"):
        service.prove(
            worlds[0],
            request,
            bundle_hash="b" * 64,
            hidden_worlds=worlds,
            hidden_world_coverage=coverage,
        )

    def evaluate_roots(world_order):
        counter = service._counter(
            budget,
            maximum_nodes=budget.maximum_nodes,
            started_wall=time.monotonic(),
            started_cpu=time.thread_time(),
            cancel_check=None,
        )
        remaining_state_space = budget.maximum_state_space
        evaluations = {}
        for action in engine.legal_play_actions(world_order[0].round_state):
            evaluation = service._evaluate_information_set_root(
                world_order,
                root_action=action,
                request=request,
                depth=8,
                budget=budget,
                counter=counter,
                remaining_state_space=remaining_state_space,
            )
            remaining_state_space -= evaluation.forecast_state_space
            evaluations[action.code] = evaluation
        return evaluations

    evaluations = evaluate_roots(worlds)
    assert evaluations == evaluate_roots(tuple(reversed(worlds)))
    assert evaluations["H9"].value == 6
    assert sorted(evaluations["H9"].world_values) == [6, 16]
    assert evaluations["D8"].value == 12
    assert sorted(evaluations["D8"].world_values) == [12, 14]
    for evaluation in evaluations.values():
        witness_world = next(
            world
            for world in worlds
            if world.state_hash == evaluation.worst_world_hash
        )
        witness_state = witness_world.round_state.clone()
        for action in evaluation.worst_path:
            engine.play_card(witness_state, Card.parse(action), emit_events=False)
        assert witness_state.round_score is not None
        assert (
            witness_state.round_score.awarded_points[0]
            - witness_state.round_score.awarded_points[1]
        ) == evaluation.value

    generated_budget = _budget(
        maximumNodes=500_000,
        maximumHiddenWorlds=16,
        maximumLineCountNodes=500_000,
    )
    generated_request = _request(
        _fixture(root),
        generated_budget,
        visibility="actor_information_set",
    )
    generated_outcome = _service(generated_budget).prove(
        source,
        generated_request,
        bundle_hash="b" * 64,
    ).payload
    assert generated_outcome["completionStatus"] == "exact_complete"
    assert generated_outcome["achievedMode"] == "exact"
    assert generated_outcome["achievedVisibility"] == "actor_information_set"
    assert generated_outcome["strategyClass"] == "pure_deterministic"
    assert generated_outcome["resources"]["strategyClass"] == (
        "pure_deterministic"
    )
    assert generated_outcome["identities"]["proofRequestHash"] == (
        generated_request.sha256
    )
    assert generated_outcome["winningRootActionCount"] == 1
    assert generated_outcome["hiddenWorlds"]["count"] == 3
    generated_coverage = generated_outcome["hiddenWorlds"]["coverage"]
    assert generated_coverage["method"] == "exhaustive_constrained_hidden_deals"
    assert generated_coverage["complete"] is True
    assert generated_coverage["worldHashes"] == generated_outcome["hiddenWorlds"][
        "hashes"
    ]
    assert generated_coverage["sha256"] == canonical_sha256(
        {
            key: value
            for key, value in generated_coverage.items()
            if key != "sha256"
        }
    )
    assert generated_outcome["resources"]["depthLimit"] == (
        generated_budget.maximum_depth
    )
    assert generated_outcome["resources"]["wallTimeLimitSeconds"] == (
        generated_budget.maximum_wall_seconds
    )

    policies = {}
    for index, seat in enumerate(("north", "west", "south", "east")):
        core = {
            "agentId": "heuristic_bot_v1",
            "version": "1.0.0",
            "seed": index + 101,
        }
        policies[seat] = {**core, "profileHash": canonical_sha256(core)}
    policy_request = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "actor_information_set",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -1_000_000,
            "selectedAction": "D8",
            "lineUniverse": "policy_conditioned",
            "policies": policies,
            "budgetProfile": budget.profile_id,
        }
    )
    policy_outcome = service.prove(
        worlds[0],
        policy_request,
        bundle_hash="b" * 64,
        hidden_worlds=worlds,
        hidden_world_coverage=coverage,
    ).payload
    assert policy_outcome["continuationLines"] == {
        "complete": True,
        "exactCount": "2",
        "lineUniverse": "policy_conditioned",
    }
    selected_row = next(
        row for row in policy_outcome["rootActions"] if row["action"] == "D8"
    )
    witness_world = next(
        world
        for world in worlds
        if world.state_hash == selected_row["worstWorldHash"]
    )
    witness_state = witness_world.round_state.clone()
    for action in policy_outcome["principalVariation"]:
        engine.play_card(witness_state, Card.parse(action), emit_events=False)
    assert witness_state.round_score is not None
    assert (
        witness_state.round_score.awarded_points[0]
        - witness_state.round_score.awarded_points[1]
    ) == selected_row["value"]

    line_count_outcome = _service(generated_budget).prove(
        source,
        _request(
            _fixture(root),
            generated_budget,
            visibility="actor_information_set",
            claim="winning_continuation_line_count",
            targetTeam="north_south",
            successThreshold=10,
            selectedAction="H10",
            lineUniverse="all_legal",
        ),
        bundle_hash="b" * 64,
    ).payload
    assert line_count_outcome["completionStatus"] == "proof_incomplete"
    assert line_count_outcome["hiddenWorlds"]["complete"] is True
    assert line_count_outcome["hiddenWorlds"]["coverage"]["complete"] is True
    assert line_count_outcome["continuationLines"]["complete"] is False
    assert line_count_outcome["resources"]["exhaustionReason"] == (
        "actor_information_set_continuation_count_not_supported"
    )

    partial_budget = _budget(
        maximumNodes=6_000,
        maximumLineCountNodes=6_000,
        maximumHiddenWorlds=16,
    )
    partial_outcome = _service(partial_budget).prove(
        source,
        _request(
            _fixture(root),
            partial_budget,
            visibility="actor_information_set",
        ),
        bundle_hash="b" * 64,
    ).payload
    assert partial_outcome["completionStatus"] == "proof_incomplete"
    assert len(partial_outcome["rootActions"]) == 1
    assert partial_outcome["winningRootActionCount"] == sum(
        bool(row["satisfies"]) for row in partial_outcome["rootActions"]
    )
    assert partial_outcome["resources"]["rootActionCoverageComplete"] is False
    assert partial_outcome["resources"]["winningRootActionLowerBound"] == (
        partial_outcome["winningRootActionCount"]
    )
    assert partial_outcome["resources"]["winningRootActionUpperBound"] == (
        partial_outcome["winningRootActionCount"] + 1
    )
    assert partial_outcome["hiddenWorlds"]["complete"] is True

    for constrained_budget, expected_reason in (
        (
            _budget(
                maximumNodes=1,
                maximumLineCountNodes=1,
                maximumHiddenWorlds=2,
            ),
            "maximum_nodes",
        ),
        (
            _budget(
                maximumCacheBytes=1_000,
                maximumHiddenWorlds=2,
            ),
            "maximum_cache_bytes",
        ),
        (
            _budget(
                maximumStateSpace=1,
                maximumHiddenWorlds=2,
            ),
            "maximum_state_space",
        ),
    ):
        constrained_service = _service(constrained_budget)
        counter = constrained_service._counter(
            constrained_budget,
            maximum_nodes=min(
                constrained_budget.maximum_nodes,
                constrained_budget.maximum_line_count_nodes,
            ),
            started_wall=time.monotonic(),
            started_cpu=time.thread_time(),
            cancel_check=None,
        )
        with pytest.raises(ProofBudgetExceeded):
            constrained_service._evaluate_information_set_root(
                worlds,
                root_action=next(
                    action
                    for action in engine.legal_play_actions(
                        worlds[0].round_state
                    )
                    if action.code == "H9"
                ),
                request=_request(
                    _fixture(root),
                    constrained_budget,
                    visibility="actor_information_set",
                    claim="selected_action_is_unique_winner",
                    targetTeam="north_south",
                    successThreshold=10,
                    selectedAction="D8",
                    lineUniverse="optimal_strategy",
                ),
                depth=8,
                budget=constrained_budget,
                counter=counter,
                remaining_state_space=constrained_budget.maximum_state_space,
            )
        assert counter.exhaustion_reason == expected_reason
        assert counter.forecast_state_space is not None
        assert counter.forecast_state_space > 0
        assert counter.retained_cache_entries > 0
        assert counter.retained_cache_bytes > 0
        assert counter.retained_cache_bytes <= (
            constrained_budget.maximum_cache_bytes
        )
        if expected_reason == "maximum_state_space":
            assert counter.nodes <= 1


def test_exact_proof_enumerates_ekka_as_a_distinct_legal_action(
    rules,
):
    trace = {
        "schemaVersion": 1,
        "deckOrder": [
            5, 6, 11, 0, 4, 7, 2, 8, 13, 1, 3, 9, 14, 21, 12, 19,
            15, 17, 16, 18, 10, 25, 27, 28, 29, 31, 22, 24, 26, 20, 23, 30,
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
                ("north", "H10"), ("west", "HJ"), ("south", "HQ"),
                ("east", "H8"), ("west", "CJ"), ("south", "C9"),
                ("east", "CK"), ("north", "C10"), ("north", "DQ"),
                ("west", "D10"), ("south", "DK"), ("east", "DA"),
                ("east", "S10"), ("north", "SQ"), ("west", "SJ"),
                ("south", "S9"), ("east", "D9"), ("north", "HK"),
                ("west", "SA"), ("south", "D8"), ("north", "H9"),
                ("west", "CQ"), ("south", "HA"), ("east", "DJ"),
            )
        ],
        "stateAssertions": {"phase": "playing"},
    }
    normalized = ExactStateNormalizerV1(rules).normalize(trace)
    state = normalized.round_state
    state.validate_conservation()
    assert state.contract == Contract.HOKUM
    assert state.trump_suit == Suit.HEARTS
    assert state.buyer == Seat.NORTH
    assert state.projects == []
    assert [
        sorted(card.code for card in state.hand(seat)) for seat in Seat
    ] == [
        ["C8", "SK"],
        ["CA", "S7"],
        ["C7", "H7"],
        ["D7", "S8"],
    ]
    assert [action.to_dict() for action in BalootEngine(rules).legal_play_actions(state)] == [
        {"actionId": 6, "code": "SK", "card": Card.parse("SK").to_dict(), "ekka": False},
        {"actionId": 25, "code": "C8", "card": Card.parse("C8").to_dict(), "ekka": False},
        {"actionId": 38, "code": "SK+ekka", "card": Card.parse("SK").to_dict(), "ekka": True},
    ]
    invalid = state.clone()
    engine = BalootEngine(rules)
    before_invalid = engine.state_hash(invalid)
    with pytest.raises(TypeError, match="ekka must be a boolean"):
        engine.apply(invalid, Card.parse("SK"), ekka=1)
    assert engine.state_hash(invalid) == before_invalid
    with pytest.raises(TypeError, match="boolean or null"):
        engine.play_card(invalid, Card.parse("SK"), ekka="true")
    assert engine.state_hash(invalid) == before_invalid
    budget = _budget(maximumNodes=100_000, maximumLineCountNodes=100_000)
    request = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "claim": "selected_action_is_unique_winner",
            "targetTeam": "north_south",
            "utility": "contract_success",
            "successThreshold": 1,
            "selectedAction": "SK+ekka",
            "lineUniverse": "optimal_strategy",
            "budgetProfile": budget.profile_id,
        }
    )
    proof_service = _service(budget)
    proof_outcome = proof_service.prove(
        normalized,
        request,
        bundle_hash="b" * 64,
    )
    repeated_outcome = proof_service.prove(
        normalized,
        request,
        bundle_hash="b" * 64,
    )
    outcome = proof_outcome.payload

    assert outcome["completionStatus"] == "exact_complete"
    assert outcome["semanticSha256"] == repeated_outcome.payload["semanticSha256"]
    assert outcome["rootActions"] == repeated_outcome.payload["rootActions"]
    assert outcome["resources"]["nodes"] == repeated_outcome.payload["resources"][
        "nodes"
    ]
    assert outcome["winningRootActionCount"] == 1
    assert {
        row["action"]: row["value"] for row in outcome["rootActions"]
    } == {"C8": 0.0, "SK": 0.0, "SK+ekka": 1.0}
    assert outcome["selectedAction"] == "SK+ekka"
    assert outcome["principalVariation"][0] == "SK+ekka"

    changed_receipt = proof_outcome.to_dict()
    changed_receipt.pop("sha256")
    changed_receipt["resources"]["elapsedWallSeconds"] += 1.0
    rebuilt_receipt = ProofOutcomeV1.build(changed_receipt)
    assert rebuilt_receipt.payload["semanticSha256"] == outcome["semanticSha256"]
    assert rebuilt_receipt.sha256 != proof_outcome.sha256

    replay = normalized.round_state.clone()
    engine = BalootEngine(rules)
    for token in outcome["principalVariation"]:
        card, ekka = decode_proof_play_action_v1(token)
        engine.play_card(replay, card, ekka=ekka, emit_events=False)
    assert replay.round_score is not None
    assert replay.round_score.winner == Seat.NORTH.team


def test_proof_cancellation_is_polled_at_the_governed_interval(rules, root: Path):
    fixture = _fixture(root)
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    budget = _budget(cancellationCheckIntervalNodes=2)
    polls = 0

    def cancel() -> bool:
        nonlocal polls
        polls += 1
        return polls == 2

    outcome = _service(budget).prove(
        normalized,
        _request(fixture, budget),
        bundle_hash="b" * 64,
        cancel_check=cancel,
    ).payload

    assert outcome["completionStatus"] == "proof_incomplete"
    assert outcome["resources"]["cancelled"] is True
    assert outcome["resources"]["exhaustionReason"] == "cancelled"
    assert outcome["resources"]["nodes"] == 2
    assert polls == 2


def test_bounded_horizon_is_refused_instead_of_silently_clipped(rules, root: Path):
    fixture = _fixture(root)
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    budget = _budget(maximumDepth=8, maximumHorizonActions=8)
    outcome = _service(budget).prove(
        normalized,
        _request(
            fixture,
            budget,
            mode="bounded",
            horizon={"unit": "plies", "value": 9},
            boundaryEvaluator="current_raw_point_margin_v1",
        ),
        bundle_hash="b" * 64,
    ).payload

    assert outcome["completionStatus"] == "proof_incomplete"
    assert outcome["resources"]["exhaustionReason"] == "maximum_horizon_actions"
    assert outcome["resources"]["depth"] == 0


@pytest.mark.real_clock
def test_cpu_state_space_and_cache_memory_limits_are_enforced(
    rules,
    root: Path,
    monkeypatch,
):
    fixture = _fixture(root)
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])

    state_budget = _budget(maximumStateSpace=1)
    state_outcome = _service(state_budget).prove(
        normalized,
        _request(fixture, state_budget),
        bundle_hash="b" * 64,
    ).payload
    assert state_outcome["completionStatus"] == "proof_incomplete"
    assert state_outcome["resources"]["nodes"] == 0
    assert state_outcome["resources"]["forecastStateSpace"] > 1
    assert state_outcome["resources"]["exhaustionReason"] == "maximum_state_space"

    import baloot_platform.research.proof_service as proof_module

    cpu_reads = 0

    def cpu_time() -> float:
        nonlocal cpu_reads
        cpu_reads += 1
        return 0.0 if cpu_reads == 1 else 1.0

    monkeypatch.setattr(proof_module.time, "thread_time", cpu_time)
    cpu_budget = _budget(maximumCpuSeconds=0.5)
    cpu_outcome = _service(cpu_budget).prove(
        normalized,
        _request(fixture, cpu_budget),
        bundle_hash="b" * 64,
    ).payload
    assert cpu_outcome["completionStatus"] == "proof_incomplete"
    assert cpu_outcome["resources"]["exhaustionReason"] == "maximum_cpu_seconds"

    monkeypatch.undo()
    cache_budget = _budget(maximumCacheBytes=1)
    cache_outcome = _service(cache_budget).prove(
        normalized,
        _request(fixture, cache_budget),
        bundle_hash="b" * 64,
    ).payload
    assert cache_outcome["completionStatus"] == "exact_complete"
    assert cache_outcome["resources"]["cacheEntries"] == 0
    assert cache_outcome["resources"]["cacheBytes"] <= 1


def test_actor_information_set_rejects_empty_and_duplicate_world_cohorts(
    rules,
    root: Path,
):
    fixture = _fixture(root)
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    actor = normalized.round_state.current_player
    assert actor is not None
    observation_hash = canonical_sha256(
        InformationState.from_complete(normalized.round_state, actor).to_dict()
    )
    budget = _budget(maximumHiddenWorlds=2)
    policies = {}
    for index, seat in enumerate(("north", "west", "south", "east")):
        core = {
            "agentId": "heuristic_bot_v1",
            "version": "1.0.0",
            "seed": index + 1,
        }
        policies[seat] = {**core, "profileHash": canonical_sha256(core)}
    request = _request(
        fixture,
        budget,
        mode="policy_conditioned",
        visibility="actor_information_set",
        strategyClass="frozen_policy",
        lineUniverse="policy_conditioned",
        policies=policies,
    )
    coverage = {
        "method": "exhaustive_reference_fixture",
        "complete": True,
        "expectedWorldCount": 2,
        "observationHash": observation_hash,
    }
    with pytest.raises(ValueError, match="empty world cohort"):
        _service(budget).prove(
            normalized,
            request,
            bundle_hash="b" * 64,
            hidden_worlds=(),
            hidden_world_coverage={**coverage, "expectedWorldCount": 0},
        )

    class OversizedWorldSequence:
        def __len__(self):
            return budget.maximum_hidden_worlds + 1

        def __getitem__(self, _index):
            raise AssertionError("oversized world cohorts must not be iterated")

    with pytest.raises(ValueError, match="exceeds maximumHiddenWorlds"):
        _service(budget).prove(
            normalized,
            request,
            bundle_hash="b" * 64,
            hidden_worlds=OversizedWorldSequence(),
            hidden_world_coverage=coverage,
        )
    with pytest.raises(ValueError, match="must be unique"):
        _service(budget).prove(
            normalized,
            request,
            bundle_hash="b" * 64,
            hidden_worlds=(normalized, normalized),
            hidden_world_coverage=coverage,
        )


def test_policy_information_set_admission_uses_linear_policy_forecast(
    rules,
    root: Path,
    monkeypatch,
):
    fixture = _fixture(root)
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    budget = _budget(maximumHiddenWorlds=2, maximumStateSpace=100)
    policies = {}
    for index, seat in enumerate(("north", "west", "south", "east")):
        core = {
            "agentId": "heuristic_bot_v1",
            "version": "1.0.0",
            "seed": index + 1,
        }
        policies[seat] = {**core, "profileHash": canonical_sha256(core)}
    request = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "actor_information_set",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -1_000_000,
            "lineUniverse": "policy_conditioned",
            "policies": policies,
            "budgetProfile": budget.profile_id,
        }
    )

    import baloot_platform.research.proof_service as proof_module

    monkeypatch.setattr(
        proof_module,
        "generate_hidden_deals",
        lambda *_args, **_kwargs: HiddenDealSet(
            states=(normalized.round_state.clone(),),
            method="exhaustive_reference_fixture",
            attempted=1,
            coverage_complete=True,
        ),
    )
    outcome = _service(budget).prove(
        normalized,
        request,
        bundle_hash="b" * 64,
    ).payload

    assert outcome["completionStatus"] == "policy_complete"
    assert outcome["achievedMode"] == "policy_conditioned"
    assert outcome["resources"]["forecastStateSpace"] <= 100
    assert outcome["resources"]["exhaustionReason"] is None


@pytest.mark.real_clock
def test_node_wall_and_selected_action_limits_fail_closed(
    rules,
    root: Path,
    monkeypatch,
):
    fixture = _fixture(root)
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])

    node_budget = _budget(maximumNodes=1, maximumLineCountNodes=1)
    node_outcome = _service(node_budget).prove(
        normalized,
        _request(fixture, node_budget),
        bundle_hash="b" * 64,
    ).payload
    assert node_outcome["completionStatus"] == "proof_incomplete"
    assert node_outcome["resources"]["nodes"] == 1
    assert node_outcome["resources"]["nodeLimit"] == 1
    assert node_outcome["resources"]["exhaustionReason"] == "maximum_nodes"

    state_budget = _budget(maximumStateSpace=1)
    with pytest.raises(ValueError, match="not a legal root action"):
        _service(state_budget).prove(
            normalized,
            _request(fixture, state_budget, selectedAction="ZZ"),
            bundle_hash="b" * 64,
        )

    import baloot_platform.research.proof_service as proof_module

    wall_reads = 0

    def wall_time() -> float:
        nonlocal wall_reads
        wall_reads += 1
        return 0.0 if wall_reads == 1 else 1.0

    monkeypatch.setattr(proof_module.time, "monotonic", wall_time)
    wall_budget = _budget(maximumWallSeconds=0.5)
    wall_outcome = _service(wall_budget).prove(
        normalized,
        _request(fixture, wall_budget),
        bundle_hash="b" * 64,
    ).payload
    assert wall_outcome["completionStatus"] == "proof_incomplete"
    assert wall_outcome["resources"]["exhaustionReason"] == (
        "maximum_wall_seconds"
    )
    assert wall_outcome["resources"]["wallTimeLimitSeconds"] == 0.5


def test_bid_policy_horizon_includes_auction_and_full_play(rules):
    normalized = ExactStateNormalizerV1(rules).normalize(
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
        }
    )
    budget = _budget(maximumHorizonActions=32)
    policies = {}
    for index, seat in enumerate(("north", "west", "south", "east")):
        core = {
            "agentId": "heuristic_bot_v1",
            "version": "1.0.0",
            "seed": index + 1,
        }
        policies[seat] = {**core, "profileHash": canonical_sha256(core)}
    actor = Seat.parse(str(normalized.next_actor))
    request = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": (
                "north_south" if actor.team.value == 0 else "west_east"
            ),
            "utility": "awarded_point_margin",
            "successThreshold": -1_000_000,
            "lineUniverse": "policy_conditioned",
            "policies": policies,
            "budgetProfile": budget.profile_id,
        }
    )
    outcome = _service(budget).prove(
        normalized,
        request,
        bundle_hash="b" * 64,
    ).payload

    assert outcome["completionStatus"] == "proof_incomplete"
    assert outcome["resources"]["depth"] == 0
    assert outcome["resources"]["horizonActionLimit"] == 32
    assert outcome["resources"]["exhaustionReason"] == "maximum_horizon_actions"
    assert outcome["resources"]["depthLimit"] == budget.maximum_depth
    assert outcome["resources"]["wallTimeLimitSeconds"] == (
        budget.maximum_wall_seconds
    )


def _policy_receipt_request(
    normalized,
    budget: ScenarioProofBudgetV1,
) -> ScenarioProofRequestV1:
    policies = {}
    for index, seat in enumerate(("north", "west", "south", "east")):
        core = {
            "agentId": "heuristic_bot_v1",
            "version": "1.0.0",
            "seed": index + 1,
        }
        policies[seat] = {**core, "profileHash": canonical_sha256(core)}
    actor = Seat.parse(str(normalized.next_actor))
    return ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": (
                "north_south" if actor.team.value == 0 else "west_east"
            ),
            "utility": "awarded_point_margin",
            "successThreshold": -1_000_000,
            "lineUniverse": "policy_conditioned",
            "policies": policies,
            "budgetProfile": budget.profile_id,
        }
    )


def _policy_decision_prefix(
    *,
    auction_actions=(),
    match_score=(0, 0),
    phase="bidding",
):
    return {
        "schemaVersion": 1,
        "deckOrder": list(range(32)),
        "dealer": "east",
        "roundNumber": 1,
        "matchScore": list(match_score),
        "auctionActions": list(auction_actions),
        "escalationActions": [],
        "projectDeclarations": [],
        "projectAssertions": [],
        "playActions": [],
        "stateAssertions": {"phase": phase},
    }


def _assert_partial_root_receipt(
    outcome,
    *,
    expected_roots: int,
    evaluated_roots: int,
):
    rows = outcome["rootActions"]
    resources = outcome["resources"]
    completed_winners = sum(bool(row["satisfies"]) for row in rows)

    assert outcome["completionStatus"] == "proof_incomplete"
    assert outcome["claimSatisfied"] is False
    assert outcome["achievedMode"] == "none"
    assert outcome["winningRootActionCount"] == completed_winners
    assert len(rows) == evaluated_roots
    assert resources["evaluatedRootActionCount"] == evaluated_roots
    assert resources["expectedRootActionCount"] == expected_roots
    assert resources["rootActionCoverageComplete"] is False
    assert resources["winningRootActionLowerBound"] == completed_winners
    assert resources["winningRootActionUpperBound"] == (
        completed_winners + expected_roots - evaluated_roots
    )
    assert outcome["continuationLines"]["complete"] is False


def test_bid_policy_late_root_budget_exhaustion_preserves_partial_receipt(rules):
    normalized = ExactStateNormalizerV1(rules).normalize(
        _policy_decision_prefix()
    )
    budget = _budget(
        maximumNodes=20,
        maximumDepth=40,
        maximumHorizonActions=40,
        cancellationCheckIntervalNodes=1,
    )
    outcome = _service(budget).prove(
        normalized,
        _policy_receipt_request(normalized, budget),
        bundle_hash="b" * 64,
    ).payload

    _assert_partial_root_receipt(
        outcome,
        expected_roots=len(normalized.legal_actions),
        evaluated_roots=1,
    )
    assert outcome["resources"]["exhaustionReason"] == "maximum_nodes"
    assert outcome["resources"]["cancelled"] is False
    assert outcome["principalVariation"][0].startswith("bid:")


def test_escalation_policy_late_root_cancellation_preserves_partial_receipt(
    rules,
):
    auction_actions = (
        {"action": "pass"},
        {"action": "pass"},
        {"action": "sun"},
        {"action": "pass"},
        {"action": "sun"},
    )
    prefix = _policy_decision_prefix(
        auction_actions=auction_actions,
        match_score=(101, 0),
        phase="final_dealt",
    )
    playing = ExactStateNormalizerV1(rules).normalize(
        {**prefix, "stateAssertions": {}}
    )
    declarer = playing.round_state.declarer
    assert declarer is not None
    actor = next(seat for seat in Seat if seat.team != declarer.team)
    normalized = ExactStateNormalizerV1(rules).normalize(
        prefix,
        decision_point={
            "kind": "escalation",
            "selector": {
                "actionIndex": 0,
                "actingSeat": actor.name.lower(),
            },
        },
    )
    budget = _budget(
        maximumDepth=40,
        maximumHorizonActions=40,
        cancellationCheckIntervalNodes=1,
    )
    polls = 0

    def cancel_during_second_root() -> bool:
        nonlocal polls
        polls += 1
        return polls == 40

    outcome = _service(budget).prove(
        normalized,
        _policy_receipt_request(normalized, budget),
        bundle_hash="b" * 64,
        cancel_check=cancel_during_second_root,
    ).payload

    _assert_partial_root_receipt(
        outcome,
        expected_roots=len(normalized.legal_actions),
        evaluated_roots=1,
    )
    assert outcome["resources"]["exhaustionReason"] == "cancelled"
    assert outcome["resources"]["cancelled"] is True
    assert outcome["resources"]["nodes"] == 40
    assert polls == 40
    assert outcome["principalVariation"][0].startswith("escalation:")


def test_project_policy_mid_root_budget_exhaustion_reports_unknown_root(rules):
    auction_actions = (
        {"action": "pass"},
        {"action": "pass"},
        {"action": "sun"},
        {"action": "pass"},
        {"action": "sun"},
    )
    normalized = ExactStateNormalizerV1(rules).normalize(
        _policy_decision_prefix(
            auction_actions=auction_actions,
            phase="playing",
        ),
        decision_point={
            "kind": "project",
            "selector": {"owner": "north", "projectIndex": 0},
        },
    )
    budget = _budget(
        maximumNodes=20,
        maximumDepth=40,
        maximumHorizonActions=40,
        cancellationCheckIntervalNodes=1,
    )
    outcome = _service(budget).prove(
        normalized,
        _policy_receipt_request(normalized, budget),
        bundle_hash="b" * 64,
    ).payload

    _assert_partial_root_receipt(
        outcome,
        expected_roots=1,
        evaluated_roots=0,
    )
    assert outcome["resources"]["exhaustionReason"] == "maximum_nodes"
    assert outcome["resources"]["cancelled"] is False
    assert outcome["principalVariation"] == []
