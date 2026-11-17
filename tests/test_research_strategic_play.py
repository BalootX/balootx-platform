from __future__ import annotations

import json

import pytest

from baloot_platform.analysis.hidden_deals import generate_hidden_deals
from baloot_platform.analysis.models import InformationState
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.research.contracts import ResearchRecipeV1, build_research_recipe
from baloot_platform.research.strategic_play import (
    ActorSafeCpSatPlayProfileV1,
    ActorSafeCpSatPlaySelectorV1,
)
from baloot_platform.service.container import build_container


def _profile(*, exact: bool = True) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "minimumPlayedCards": 24,
        "requestedSamples": 1 if exact else 2,
        "exactAllocationLimit": 500 if exact else 1,
        "maximumSamplingAttempts": 500,
        "maximumWorlds": 500,
        "maximumContinuationSteps": 20_000,
        "maximumPlies": 8,
        "maximumSelectionSeconds": 30,
        "requireExactForQualification": True,
        "continuationPolicy": "observer_safe_heuristic_v1",
        "policyProfileId": "maximin_margin",
        "policyRevision": 1,
        "outcomeBoundary": 1,
        "utility": "actor_team_awarded_point_margin",
        "tieBreak": "maximin_then_total_then_card_id",
    }


def _late_game(rules):
    engine = BalootEngine(rules)
    state = engine.create_round(seed=2)
    engine.shuffle_and_initial_deal(state)
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.HOKUM,
        forced_buyer=Seat.SOUTH,
    )
    engine.prepare_projects(state)
    bot = HeuristicBot(engine, BotProfile(state.seed))
    while len(state.played_cards) < 24:
        engine.play_card(state, bot.choose(state))
    assert len(engine.legal_actions(state)) > 1
    return engine, state


def test_strategic_profile_and_recipe_round_trip():
    profile = ActorSafeCpSatPlayProfileV1.build(_profile())
    assert ActorSafeCpSatPlayProfileV1.build(profile.to_dict()) == profile

    recipe = {
        "schemaVersion": 1,
        "titleAr": "اختيار آمن من منظور اللاعب",
        "output": {"kind": "laffa", "resultCount": 1},
        "profiles": {
            key: "0" * 64
            for key in (
                "domainHash",
                "bundleHash",
                "rulesHash",
                "conventionHash",
                "analysisProfileHash",
                "scoreProfileHash",
            )
        }
        | {"agentRelease": "agent@frozen"},
        "situation": {
            "dealer": "east",
            "buyer": "south",
            "purchaseRound": 1,
            "trumpSuit": "clubs",
            "contracts": ["hokum"],
            "matchScore": [0, 0],
            "requiredCards": {},
            "forbiddenCards": {},
            "projects": [],
            "escalations": [],
        },
        "tactics": [],
        "qualification": {
            "minimumTruth": "provisional",
            "minimumConfidence": 0,
            "minimumSurroundingQuality": 0,
            "requireObserverSafety": True,
            "requireAllDecisionAudit": True,
            "maximumRegret": 999,
        },
        "resources": {
            "generationLane": "ortools",
            "seedStart": 1,
            "seedCount": 1,
            "seedsPerShard": 1,
            "candidatesPerSeed": 1,
            "solverTimeoutSeconds": 5,
            "maximumConflicts": 50_000,
            "analysisMode": "screening",
            "auctionConstructionMode": "actor_safe_agent_verified",
            "playConstructionMode": "cp_sat_actor_safe_late_game_v1",
            "strategicPlay": _profile(),
        },
        "status": "published",
    }
    built = ResearchRecipeV1.build(recipe)
    rebuilt = ResearchRecipeV1.build(built.to_dict())
    assert rebuilt == built
    assert rebuilt.resources["strategicPlay"]["sha256"] == profile.sha256
    assert rebuilt.resources["auctionConstructionMode"] == (
        "actor_safe_agent_verified"
    )

    from baloot_platform.service.api import (
        ResearchRecipeRequest,
        ResearchRecipeResponse,
    )

    api_request = ResearchRecipeRequest.model_validate(recipe)
    assert api_request.resources.playConstructionMode == (
        "cp_sat_actor_safe_late_game_v1"
    )
    assert api_request.resources.auctionConstructionMode == (
        "actor_safe_agent_verified"
    )
    assert api_request.resources.strategicPlay is not None
    assert api_request.resources.strategicPlay.policyProfileId == "maximin_margin"
    api_response = ResearchRecipeResponse.model_validate(built.to_dict())
    assert api_response.resources.strategicPlay is not None
    assert api_response.resources.strategicPlay.sha256 == profile.sha256


def test_actor_safe_cp_sat_selects_deterministically_across_hidden_worlds(rules):
    engine, state = _late_game(rules)
    information = InformationState.from_complete(state, state.current_player)
    cohort = generate_hidden_deals(
        information,
        engine,
        requested_samples=1,
        exact_allocation_limit=500,
        maximum_attempts=500,
        seed=99,
    )
    assert cohort.coverage_complete
    assert len(cohort.states) > 1

    selector = ActorSafeCpSatPlaySelectorV1(
        ActorSafeCpSatPlayProfileV1.build(_profile())
    )
    results = [
        selector.choose(
            engine=engine,
            state=world,
            legal_cards=engine.legal_actions(world),
            seed=77,
        )
        for world in cohort.states[:2]
    ]
    assert results[0] is not None and results[1] is not None
    assert results[0][0] == results[1][0]
    assert results[0][1]["informationStateHash"] == results[1][1][
        "informationStateHash"
    ]
    evidence = results[0][1]
    assert evidence["strategicTruth"] == "exact_policy_conditioned_actor_safe"
    assert evidence["coverageComplete"] is True
    assert evidence["allAlternativesEvaluated"] is True
    assert evidence["actorChoiceBoundOnce"] is True
    assert evidence["modelFeasibilityStatus"] == "optimal"
    assert evidence["hiddenStateIdentifiersExposed"] is False
    assert "worldFingerprint" not in json.dumps(evidence)


def test_sampled_actor_safe_selection_is_provisional_and_blocks_qualification(rules):
    engine, state = _late_game(rules)
    selector = ActorSafeCpSatPlaySelectorV1(
        ActorSafeCpSatPlayProfileV1.build(_profile(exact=False))
    )
    selected = selector.choose(
        engine=engine,
        state=state,
        legal_cards=engine.legal_actions(state),
        seed=77,
    )
    assert selected is not None
    _card, evidence = selected
    assert evidence["strategicTruth"] == "provisional_policy_conditioned_actor_safe"
    assert evidence["coverageComplete"] is False
    assert evidence["qualificationBlocked"] is True


def test_strategic_profile_rejects_unknown_or_tampered_derived_values():
    unknown = _profile()
    unknown["hiddenDefault"] = True
    with pytest.raises(ValueError, match="unknown keys"):
        ActorSafeCpSatPlayProfileV1.build(unknown)

    canonical = ActorSafeCpSatPlayProfileV1.build(_profile()).to_dict()
    canonical["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        ActorSafeCpSatPlayProfileV1.build(canonical)


def test_constructed_laffa_uses_actor_safe_cp_sat_tail_and_full_line_proof(
    tmp_path,
):
    container = build_container(writable_root=tmp_path / "runtime")
    try:
        research = container.research
        payload = {
            "schemaVersion": 3,
            "titleAr": "اختيار آمن مع تحقق المسار كامل",
            "output": {"kind": "laffa", "resultCount": 1},
            "situation": {
                "dealer": "east",
                "buyer": "north",
                "purchaseRound": 2,
                "trumpSuit": "spades",
                "contracts": ["hokum"],
                "matchScore": [0, 0],
                "requiredCards": {},
                "forbiddenCards": {},
                "projects": [],
                "escalations": [],
            },
            "tactics": [],
            "qualification": {
                "minimumTruth": "provisional",
                "minimumConfidence": 0,
                "minimumSurroundingQuality": 0,
                "requireObserverSafety": True,
                "requireAllDecisionAudit": True,
                "maximumRegret": 999,
            },
            "resources": {
                "generationLane": "ortools",
                "seedStart": 1,
                "seedCount": 1,
                "seedsPerShard": 1,
                "candidatesPerSeed": 1,
                "solverTimeoutSeconds": 5,
                "maximumConflicts": 50_000,
                "analysisMode": "screening",
                "playConstructionMode": "cp_sat_actor_safe_late_game_v1",
                "strategicPlay": _profile(),
            },
            "status": "published",
        }
        payload["configuration"] = {
            "schemaVersion": 1,
            "bundleRevisionId": container.configuration.resolve()["bundle"][
                "revisionId"
            ],
        }
        validated = research.validate_recipe(payload)
        assert validated["valid"] is True, validated["issues"]
        created = research.store.put_recipe(validated["recipe"])
        recipe = build_research_recipe(
            research.store.get_recipe(created["code"])
        )
        frozen_context = research.execution_context_factory.resolve_recipe(recipe)
        deck_order = tuple(range(32))
        auction = research._compile_auction(
            recipe,
            deck_order,
            context=frozen_context,
        )
        assert auction is not None
        escalation = research._compile_escalation_plan(
            recipe,
            auction,
            context=frozen_context,
        )
        assert escalation is not None
        selector = ActorSafeCpSatPlaySelectorV1(
            ActorSafeCpSatPlayProfileV1.build(
                recipe.resources["strategicPlay"]
            )
        )
        simulation = frozen_context.simulator.simulate_preordered_deck(
            1,
            deck_order,
            deal_generator_provenance={
                "pluginId": "test_deal_generator",
                "profileHash": "1" * 64,
                "candidateHash": "2" * 64,
            },
            dealer=Seat.EAST,
            scripted_bids=tuple(item.to_dict() for item in auction.actions),
            scripted_escalations=tuple(
                item.to_dict() for item in escalation.actions
            ),
            play_selector=selector,
        )
        trace = simulation.final_trace
        generation = trace.trusted_final_state["generation"]
        assert generation["strategicPlaySelector"]["profile"]["sha256"] == (
            selector.profile.sha256
        )
        strategic = [
            item["trace"]
            for item in trace.trusted_final_state["decisionPolicyTraces"]
            if item["trace"].get("method") == selector.descriptor.id
        ]
        assert strategic
        assert all(item["hiddenStateIdentifiersExposed"] is False for item in strategic)
        assert all(item["allAlternativesEvaluated"] is True for item in strategic)

        counterfactual = (
            frozen_context.simulator.resimulate_constructed_play_as_counterfactual_descendant(
                trace
            )
        )
        assert counterfactual.seed_provenance["deckCommitment"] == (
            trace.seed_provenance["deckCommitment"]
        )
        assert frozen_context.simulator._constructed_pre_play_prefix_hash(
            counterfactual
        ) == frozen_context.simulator._constructed_pre_play_prefix_hash(trace)

        required_cards, required_ekka_leads = research._played_line(simulation)
        _project, play_plan, verified_actions = research._compile_complete_play(
            recipe,
            deck_order,
            auction,
            escalation,
            required_cards=required_cards,
            required_ekka_leads=required_ekka_leads,
            context=frozen_context,
        )
        assert play_plan.solver_status == "optimal"
        assert len(verified_actions) == 32
        assert research._compiled_play_mismatches(play_plan, simulation) == ()

    finally:
        container.shutdown()
