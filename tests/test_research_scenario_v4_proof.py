from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import os
import subprocess
import sys

import pytest
import yaml

from baloot_platform.analysis.models import InformationState
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat, Team
from baloot_platform.research.exact_state import ExactStateNormalizerV1
from baloot_platform.research.auction_compiler import OrToolsCpSatAuctionCompilerV1
from baloot_platform.research.proof_service import (
    ScenarioProofBudgetRegistryV1,
    ScenarioProofBudgetV1,
    UnifiedScenarioProofServiceV1,
)
from baloot_platform.research.scenario_contracts import (
    ResearchScenarioV1,
    ScenarioProofRequestV1,
    canonical_sha256,
)
from baloot_platform.research.scenario_registry import (
    ScenarioCompilerV1,
    ScenarioPredicateRegistryV1,
)
from baloot_platform.research.scenario_evidence import ScenarioRuntimeVerifierV1
from baloot_platform.research.scenario_construction import ScenarioStructuralCompilerV1
from baloot_platform.research.scenario_search import (
    ScenarioSearchExecutorV1,
)
from baloot_platform.research.recipe_v4 import ResearchRecipeV4
from baloot_platform.research.service import (
    ResearchFoundryServiceV1,
)
from baloot_platform.research.scenario_result import (
    validate_scenario_result_v4,
)
from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer
from baloot_platform.video.audio_postprocess import build_audio_cue_plan

from research_scenario_v4_helpers import (
    ProofOutcomeRoundTrip,
    _exact_scenario_recipe,
    _late_prefix,
    _policy_proof_for,
    _predicate,
    _proof,
    _retained_late_play_scenario,
    _run_exact_scenario_recipe,
    _scenario,
)


def test_exact_trace_prefix_replays_and_proves_every_root_action(rules, root: Path):
    normalized = ExactStateNormalizerV1(rules).normalize(_late_prefix(rules))
    assert normalized.phase == "playing"
    assert sum(mask.bit_count() for mask in normalized.round_state.hands) == 4
    assert normalized.state_hash == canonical_sha256(normalized.state)
    assert normalized.project_declarations_verified == len(
        [
            event
            for event in normalized.state["events"]
            if event["type"] == "project_declared"
        ]
    )
    invalid_declaration = _late_prefix(rules)
    invalid_declaration["projectDeclarations"][0]["owner"] = "south"
    with pytest.raises(
        ValueError,
        match="project declarations do not match authoritative events",
    ):
        ExactStateNormalizerV1(rules).normalize(invalid_declaration)
    service = UnifiedScenarioProofServiceV1.from_repository(root)
    request = _proof(
        targetTeam=(
            "north_south"
            if normalized.round_state.current_player.team.value == 0
            else "west_east"
        )
    )
    outcome = service.prove(normalized, request, bundle_hash="a" * 64)
    payload = outcome.to_dict()
    assert payload["completionStatus"] == "exact_complete"
    assert payload["achievedMode"] == "exact"
    assert len(payload["rootActions"]) == len(normalized.legal_actions)
    assert payload["winningRootActionCount"] == sum(
        bool(item["satisfies"]) for item in payload["rootActions"]
    )
    assert payload["continuationLines"]["complete"] is True
    assert ProofOutcomeRoundTrip(payload) == payload

    terminal = normalized.round_state.clone()
    engine = BalootEngine(rules)
    for code in payload["principalVariation"]:
        engine.play_card(terminal, Card.parse(code), emit_events=False)
    assert normalized.round_state.round_score is None
    assert terminal.round_score is not None
    assert terminal.declarer is not None
    buyer_status = (
        "won"
        if terminal.round_score.winner == terminal.declarer.team
        else "lost"
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _predicate(
                "PRED-ACTOR",
                "play.current_player",
                {"seat": normalized.next_actor},
            ),
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-BUYER-OUTCOME",
                "outcome.buyer_contract",
                {"status": buyer_status},
            ),
        }
    )
    satisfaction = ScenarioRuntimeVerifierV1().verify(
        scenario,
        normalized,
        proof_outcome=outcome,
    )
    assert satisfaction.status == "satisfied"
    outcome_evidence = next(
        item for item in satisfaction.evidence if item.kind == "outcome.buyer_contract"
    )
    assert outcome_evidence.observed["stateSource"] == "proof_principal_variation"


def test_golden_sun_hundred_project_has_exactly_no_salvage(rules, root: Path):
    fixture = json.loads(
        (root / "examples" / "research-scenario-v4-golden.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    expected = fixture["expected"]
    assert normalized.phase == expected["phase"]
    assert normalized.round_state.buyer == Seat.NORTH
    assert normalized.round_state.contract.value == expected["contract"]
    awarded = [
        project
        for project in normalized.round_state.projects
        if project.owner == Seat.NORTH
        and project.kind == "hundred_sequence"
        and project.value == 20
        and project.scoring
    ]
    assert len(awarded) == 1

    proof_request = ScenarioProofRequestV1.build(fixture["proof"])
    proof = UnifiedScenarioProofServiceV1.from_repository(root).prove(
        normalized,
        proof_request,
        bundle_hash="b" * 64,
    )
    payload = proof.payload
    assert payload["completionStatus"] == expected["completionStatus"]
    assert payload["claimSatisfied"] is expected["claimSatisfied"]
    assert payload["winningRootActionCount"] == expected["winningRootActionCount"]
    assert sorted(item["action"] for item in payload["rootActions"]) == sorted(
        expected["legalRootActions"]
    )
    assert [item["value"] for item in payload["rootActions"]] == expected["rootValues"]
    assert all(item["satisfies"] is False for item in payload["rootActions"])


def test_golden_sun_hundred_project_has_one_saving_move(rules, root: Path):
    fixture = json.loads(
        (root / "examples" / "research-scenario-v4-unique-saving.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    assert normalized.round_state.buyer == Seat.NORTH
    assert normalized.round_state.contract.value == "sun"
    assert any(
        project.owner == Seat.NORTH
        and project.kind == "hundred_sequence"
        and project.value == 20
        and project.scoring
        for project in normalized.round_state.projects
    )
    proof = UnifiedScenarioProofServiceV1.from_repository(root).prove(
        normalized,
        ScenarioProofRequestV1.build(fixture["proof"]),
        bundle_hash="b" * 64,
    )
    expected = fixture["expected"]
    payload = proof.payload
    assert payload["completionStatus"] == expected["completionStatus"]
    assert payload["claimSatisfied"] is True
    assert payload["selectedAction"] == expected["selectedAction"]
    assert payload["winningRootActionCount"] == 1
    assert payload["continuationLines"] == {
        "complete": True,
        "exactCount": expected["winningContinuationLineCount"],
        "lineUniverse": "all_legal",
    }
    root_actions = {item["action"]: item for item in payload["rootActions"]}
    assert root_actions["C10"]["satisfies"] is False
    assert root_actions["C10"]["value"] == 0
    assert root_actions["H10"]["satisfies"] is True
    assert root_actions["H10"]["value"] == 10

    wrong_action_request = ScenarioProofRequestV1.build(
        {**fixture["proof"], "selectedAction": "C10"}
    )
    wrong_action = UnifiedScenarioProofServiceV1.from_repository(root).prove(
        normalized,
        wrong_action_request,
        bundle_hash="b" * 64,
    )
    assert wrong_action.payload["completionStatus"] == "proof_failed"
    assert wrong_action.payload["claimSatisfied"] is False
    assert wrong_action.payload["selectedAction"] == "C10"

    illegal_action_request = ScenarioProofRequestV1.build(
        {**fixture["proof"], "selectedAction": "D7"}
    )
    with pytest.raises(ValueError, match="not a legal root action"):
        UnifiedScenarioProofServiceV1.from_repository(root).prove(
            normalized,
            illegal_action_request,
            bundle_hash="b" * 64,
        )


def test_golden_sun_hundred_project_requires_ace_sacrifice_for_partner_seven(
    rules,
    root: Path,
):
    fixture = json.loads(
        (root / "examples" / "research-scenario-v4-ace-sacrifice-seven.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    expected = fixture["expected"]
    assert normalized.phase == expected["phase"]
    assert normalized.next_actor == expected["actingSeat"]
    assert normalized.round_state.buyer == Seat.NORTH
    assert normalized.round_state.contract.value == expected["contract"]
    assert normalized.round_state.current_trick is not None
    assert (
        sum(mask.bit_count() for mask in normalized.round_state.hands)
        + len(normalized.round_state.current_trick.plays)
        == 8
    )
    assert any(
        project.owner == Seat.NORTH
        and project.kind == "hundred_sequence"
        and project.value == 20
        and project.scoring
        for project in normalized.round_state.projects
    )

    payload = (
        UnifiedScenarioProofServiceV1.from_repository(root)
        .prove(
            normalized,
            ScenarioProofRequestV1.build(fixture["proof"]),
            bundle_hash="b" * 64,
        )
        .payload
    )
    roots = {item["action"]: item for item in payload["rootActions"]}
    assert payload["completionStatus"] == expected["completionStatus"]
    assert payload["claimSatisfied"] is True
    assert payload["selectedAction"] == expected["selectedAction"]
    assert payload["winningRootActionCount"] == 1
    assert payload["continuationLines"] == {
        "complete": True,
        "exactCount": expected["winningContinuationLineCount"],
        "lineUniverse": "all_legal",
    }
    assert {action: item["value"] for action, item in roots.items()} == expected[
        "rootValues"
    ]
    assert roots["DA"]["satisfies"] is True
    assert roots["H10"]["satisfies"] is False
    assert payload["principalVariation"] == expected["principalVariation"]


def test_golden_ace_sacrifice_is_only_win_and_final_nine_takes_two_aces(
    rules,
    root: Path,
):
    fixture = json.loads(
        (
            root
            / "examples"
            / "research-scenario-v4-ace-sacrifice-two-aces.json"
        ).read_text(encoding="utf-8")
    )
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    state = normalized.round_state
    expected = fixture["expected"]
    assert normalized.next_actor == expected["actingSeat"]
    assert sorted(action["code"] for action in normalized.legal_actions) == sorted(
        expected["legalRootActions"]
    )
    assert [
        sum(
            trick.raw_points
            for trick in state.completed_tricks
            if int(trick.winner.team) == team
        )
        for team in range(2)
    ] == expected["rawPoints"]
    assert any(
        project.owner == Seat.NORTH
        and project.kind == "hundred_sequence"
        and project.value == 20
        and project.scoring
        for project in state.projects
    )

    proof = UnifiedScenarioProofServiceV1.from_repository(root).prove(
        normalized,
        ScenarioProofRequestV1.build(fixture["proof"]),
        bundle_hash="b" * 64,
    )
    payload = proof.payload
    assert payload["completionStatus"] == expected["completionStatus"]
    assert payload["claimSatisfied"] is expected["claimSatisfied"]
    assert payload["winningRootActionCount"] == 1
    assert payload["continuationLines"]["exactCount"] == expected[
        "winningContinuationLineCount"
    ]
    assert payload["principalVariation"] == expected["principalVariation"]
    assert {row["action"]: row["value"] for row in payload["rootActions"]} == expected[
        "rootValues"
    ]

    replay = state.clone()
    engine = BalootEngine(rules)
    for action in payload["principalVariation"]:
        engine.play_card(replay, Card.parse(action), emit_events=False)
    final_three = replay.completed_tricks[-3:]
    assert [trick.winner for trick in final_three] == [
        Seat.EAST,
        Seat.EAST,
        Seat.EAST,
    ]
    assert [trick.plays[-1].card.code for trick in final_three] == ["CA", "S7", "SA"]
    assert final_three[1].plays[0].card.code == "C8"
    assert final_three[2].plays[0].card.code == "C9"
    assert sorted(
        play.card.code
        for play in final_three[2].plays
        if play.seat.team == Team.NORTH_SOUTH and play.card.rank.name == "ACE"
    ) == ["DA", "SA"]


def test_principal_variation_trick_winner_predicate_replays_exact_proof(
    rules,
    root: Path,
):
    fixture = json.loads(
        (root / "examples" / "research-scenario-v4-ace-sacrifice-seven.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    proof = UnifiedScenarioProofServiceV1.from_repository(root).prove(
        normalized,
        ScenarioProofRequestV1.build(fixture["proof"]),
        bundle_hash="b" * 64,
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-TRICK-7",
                        "proof.principal_variation_trick_winner",
                        {
                            "trickIndex": 6,
                            "team": "west_east",
                            "rank": ["seven", "eight", "nine"],
                        },
                    ),
                    _predicate(
                        "PRED-TRICK-7-ACE-COUNT",
                        "proof.principal_variation_trick_team_rank_count",
                        {
                            "trickIndex": 6,
                            "team": "west_east",
                            "rank": ["ace"],
                            "minimum": 1,
                            "maximum": 1,
                        },
                    ),
                    _predicate(
                        "PRED-ROOT-ACE",
                        "proof.winning_root_card_rank_count",
                        {
                            "rank": ["ace"],
                            "minimum": 1,
                            "maximum": 1,
                        },
                    ),
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-TRICK-8",
                        "proof.principal_variation_trick_winner",
                        {
                            "trickIndex": 7,
                            "team": "west_east",
                            "rank": ["seven", "eight", "nine"],
                        },
                    ),
                    _predicate(
                        "PRED-TRICK-8-OPPONENT-ACES",
                        "proof.principal_variation_trick_team_rank_count",
                        {
                            "trickIndex": 7,
                            "team": "north_south",
                            "rank": ["ace"],
                            "minimum": 2,
                            "maximum": 2,
                        },
                    ),
                ],
            },
        }
    )
    result = ScenarioRuntimeVerifierV1().verify(
        scenario,
        normalized,
        proof_outcome=proof,
    )
    evidence = {item.predicate_id: item for item in result.evidence}
    assert evidence["PRED-TRICK-7"].status == "satisfied"
    assert evidence["PRED-TRICK-7"].observed["card"] == "C7"
    assert evidence["PRED-TRICK-7-ACE-COUNT"].status == "satisfied"
    assert evidence["PRED-TRICK-7-ACE-COUNT"].observed["matchingCards"] == [
        {"seat": "west", "card": "DA", "rank": "ace"}
    ]
    assert evidence["PRED-ROOT-ACE"].status == "satisfied"
    assert evidence["PRED-ROOT-ACE"].observed["matchingActions"] == ["DA"]
    assert evidence["PRED-TRICK-8"].status == "failed"
    assert evidence["PRED-TRICK-8"].observed["card"] == "H10"
    assert evidence["PRED-TRICK-8-OPPONENT-ACES"].status == "failed"
    assert evidence["PRED-TRICK-8-OPPONENT-ACES"].observed["count"] == 0


def test_search_candidate_skips_illegal_selected_action_without_aborting_budget(
    rules,
    root: Path,
):
    fixture = json.loads(
        (root / "examples" / "research-scenario-v4-ace-sacrifice-seven.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    proof = _proof(
        claim="selected_action_is_unique_winner",
        selectedAction="D7",
        targetTeam="west_east",
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-CONTRACT",
                        "auction.contract",
                        {"contract": "sun"},
                    )
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-COUNT",
                "proof.winning_root_action_count",
                {"minimum": 1, "maximum": 1},
            ),
        }
    )
    recipe = _exact_scenario_recipe("RCP-V4MISS01", "مرشح غير مطابق", scenario, proof)
    candidate = ScenarioSearchExecutorV1()._finish_candidate(
        recipe,
        normalized=normalized,
        bundle_hash="b" * 64,
        seed=1,
        receipt={"schemaVersion": 1, "structural": {"status": "satisfied"}},
        auction=SimpleNamespace(to_dict=lambda: {"status": "satisfied"}),
        cancel_check=None,
        policy={"id": "test", "observerSafe": True},
    )
    assert candidate.status == "semantic_postcondition_miss"
    assert candidate.proof is None
    assert candidate.satisfaction is not None
    assert candidate.construction_receipt["failure"] == (
        "proof_selected_action_not_legal"
    )


def test_golden_media_script_builds_verified_observer_safe_bundles(
    rules,
    root: Path,
):
    path = root / "scripts" / "render_research_scenario_v4_goldens.py"
    spec = importlib.util.spec_from_file_location("scenario_v4_golden_media", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    profile_hash = canonical_sha256(profile)
    renderer = RemotionProfessionalRenderer(profile, repository_root=root)
    counts = []
    public_projection_hashes = []
    for name, fixture_path in module.GOLDENS:
        bundle = module.build_golden_bundle(
            name=name,
            fixture_path=fixture_path,
            rules=rules,
            renderer=renderer,
            video_profile_hash=profile_hash,
        )
        repeated = module.build_golden_bundle(
            name=name,
            fixture_path=fixture_path,
            rules=rules,
            renderer=renderer,
            video_profile_hash=profile_hash,
        )
        assert repeated.bundle_sha256 == bundle.bundle_sha256
        assert bundle.verify() is True
        assert bundle.value["exportPreset"] == "hooked_flagship"
        assert [
            segment["stage"] for segment in bundle.value["timeline"]["segments"]
        ] == [
            "hook",
            "rewind",
            "replay",
            "question",
            "reveal",
            "continuation",
            "outcome",
        ]
        assert bundle.value["source"]["publicationEligible"] is True
        assert bundle.value["source"]["auditOnly"] is False
        assert bundle.value["source"]["qualificationFailures"] == []
        assert bundle.value["source"]["publicationBlockers"] == []
        assert [scene["stage"] for scene in bundle.value["scenes"]] == [
            "context",
            "question",
            "reveal",
            "continuation",
            "outcome",
        ]
        context = next(
            scene for scene in bundle.value["scenes"] if scene["stage"] == "continuation"
        )
        public_lifecycle = next(
            layer["content"]
            for layer in context["layers"]
            if layer["id"] == "public-lifecycle"
        )
        observer_lifecycle = next(
            layer["content"]
            for layer in context["layers"]
            if layer["id"] == "observer-lifecycle"
        )
        assert len(observer_lifecycle["initialHandCards"]) == 8
        assert len(public_lifecycle["plays"]) == 32
        reveal = next(
            scene for scene in bundle.value["scenes"] if scene["stage"] == "reveal"
        )
        analysis = next(
            layer["content"] for layer in reveal["layers"] if layer["id"] == "analysis"
        )
        counts.append(analysis["winningRootActionCount"])
        assert "exactStateId" not in bundle.value["source"]
        public_projection_hashes.append(
            bundle.value["source"]["publicProjectionHash"]
        )
        encoded = json.dumps(bundle.to_dict(), sort_keys=True)
        assert '"deckOrder"' not in encoded
        assert '"initialHands"' not in encoded
        audio_plan = build_audio_cue_plan(bundle)
        assert {item["kind"] for item in audio_plan["events"]} == {
            "card",
            "collect",
            "outcome",
            "timer",
            "reveal",
        }
    assert counts == [0, 1, 1]
    assert len(set(public_projection_hashes)) == 3
    assert module.GOLDENS[-1][1].name == (
        "research-scenario-v4-ace-sacrifice-two-aces.json"
    )


def test_golden_media_script_cli_is_source_checkout_safe(root: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(root / "scripts" / "render_research_scenario_v4_goldens.py"),
            "--help",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Research V4 golden scenarios" in result.stdout


def test_exact_proof_budget_exhaustion_returns_only_incomplete_bounds(
    rules, root: Path
):
    fixture = json.loads(
        (root / "examples" / "research-scenario-v4-unique-saving.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    budget = ScenarioProofBudgetV1.build(
        "test_one_node_v1",
        {
            "maximumNodes": 1,
            "maximumDepth": 16,
            "maximumHorizonActions": 16,
            "maximumHiddenWorlds": 1,
            "maximumWallSeconds": 10.0,
            "maximumCpuSeconds": 10.0,
            "maximumCacheEntries": 1,
            "maximumCacheBytes": 1024,
            "cancellationCheckIntervalNodes": 1,
            "maximumWorkers": 1,
            "maximumLineCountNodes": 1,
            "maximumStateSpace": 1000000,
        },
    )
    request = dict(fixture["proof"])
    request["budgetProfile"] = budget.profile_id
    outcome = UnifiedScenarioProofServiceV1(
        ScenarioProofBudgetRegistryV1(
            {budget.profile_id: budget},
            canonical_sha256({"profiles": [budget.sha256]}),
        )
    ).prove(
        normalized,
        ScenarioProofRequestV1.build(request),
        bundle_hash="b" * 64,
    )
    assert outcome.payload["completionStatus"] == "proof_incomplete"
    assert outcome.payload["achievedMode"] == "none"
    assert outcome.payload["claimSatisfied"] is False
    assert outcome.payload["continuationLines"] == {
        "complete": False,
        "lowerBound": "0",
        "upperBound": None,
        "lineUniverse": "all_legal",
    }


def test_actor_information_set_requires_exhaustive_indistinguishable_coverage(
    rules,
    root: Path,
):
    prefix = _late_prefix(rules)
    prefix["playActions"] = prefix["playActions"][:-4]
    normalized = ExactStateNormalizerV1(rules).normalize(prefix)
    actor = normalized.round_state.current_player
    assert actor is not None
    observation_hash = canonical_sha256(
        InformationState.from_complete(normalized.round_state, actor).to_dict()
    )
    request = _proof(
        visibility="actor_information_set",
        budgetProfile="actor_information_set_exact_v1",
    )
    service = UnifiedScenarioProofServiceV1.from_repository(root)
    automatic = service.prove(
        normalized,
        request,
        bundle_hash="b" * 64,
    )
    assert automatic.payload["completionStatus"] == "exact_complete"
    assert automatic.payload["hiddenWorlds"]["count"] > 1
    assert automatic.payload["hiddenWorlds"]["coverage"]["complete"] is True
    assert automatic.payload["strategyClass"] == "pure_deterministic"
    assert automatic.payload["hiddenWorlds"]["coverage"]["worldHashes"] == (
        automatic.payload["hiddenWorlds"]["hashes"]
    )
    with pytest.raises(ValueError, match="generated internally"):
        service.prove(
            normalized,
            request,
            bundle_hash="b" * 64,
            hidden_worlds=(normalized,),
        )
    with pytest.raises(ValueError, match="generated internally"):
        service.prove(
            normalized,
            request,
            bundle_hash="b" * 64,
            hidden_worlds=(normalized,),
            hidden_world_coverage={
                "method": "exhaustive_reference_fixture",
                "complete": True,
                "expectedWorldCount": 1,
                "observationHash": observation_hash,
            },
        )


def test_bounded_and_policy_conditioned_proofs_keep_their_labels(rules, root: Path):
    normalized = ExactStateNormalizerV1(rules).normalize(_late_prefix(rules))
    service = UnifiedScenarioProofServiceV1.from_repository(root)
    bounded = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "bounded",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -1_000_000,
            "horizon": {"unit": "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1",
            "budgetProfile": "bounded_v1",
        }
    )
    bounded_outcome = service.prove(normalized, bounded, bundle_hash="c" * 64)
    assert bounded_outcome.payload["completionStatus"] == "bounded_complete"
    assert bounded_outcome.payload["achievedMode"] == "bounded"

    policies = {}
    for index, seat in enumerate(("north", "west", "south", "east")):
        core = {"agentId": "heuristic_bot_v1", "version": "1.0.0", "seed": index + 1}
        policies[seat] = {**core, "profileHash": canonical_sha256(core)}
    policy = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -1_000_000,
            "policies": policies,
            "budgetProfile": "bounded_v1",
        }
    )
    policy_outcome = service.prove(normalized, policy, bundle_hash="d" * 64)
    assert policy_outcome.payload["completionStatus"] == "policy_complete"
    assert policy_outcome.payload["achievedMode"] == "policy_conditioned"
    assert policy_outcome.payload["continuationLines"] == {
        "complete": True,
        "exactCount": "1",
        "lineUniverse": "policy_conditioned",
    }

    false_claim = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": 1_000_000,
            "budgetProfile": "late_game_exact_v1",
        }
    )
    failed = service.prove(normalized, false_claim, bundle_hash="d" * 64)
    assert failed.payload["completionStatus"] == "proof_failed"
    assert failed.payload["claimSatisfied"] is False


def test_policy_continuation_count_is_for_selected_root_not_all_roots(
    rules,
    root: Path,
):
    fixture = json.loads(
        (root / "examples" / "research-scenario-v4-unique-saving.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    policies = {}
    for index, seat in enumerate(("north", "west", "south", "east")):
        core = {
            "agentId": "heuristic_bot_v1",
            "version": "1.0.0",
            "seed": index + 101,
        }
        policies[seat] = {**core, "profileHash": canonical_sha256(core)}
    request = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -1_000_000,
            "policies": policies,
            "lineUniverse": "policy_conditioned",
            "budgetProfile": "bounded_v1",
        }
    )
    outcome = UnifiedScenarioProofServiceV1.from_repository(root).prove(
        normalized,
        request,
        bundle_hash="d" * 64,
    )
    assert len(outcome.payload["rootActions"]) == 2
    assert outcome.payload["winningRootActionCount"] == 2
    assert outcome.payload["continuationLines"]["exactCount"] == "1"


def test_runtime_satisfaction_requires_real_state_and_proof_evidence(rules, root: Path):
    normalized = ExactStateNormalizerV1(rules).normalize(_late_prefix(rules))
    target_team = (
        "north_south"
        if normalized.round_state.current_player.team.value == 0
        else "west_east"
    )
    request = _proof(targetTeam=target_team)
    proof = UnifiedScenarioProofServiceV1.from_repository(root).prove(
        normalized,
        request,
        bundle_hash="e" * 64,
    )
    count = int(proof.payload["winningRootActionCount"])
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-ACTOR",
                        "play.current_player",
                        {"seat": normalized.next_actor},
                    ),
                    _predicate(
                        "PRED-LEGAL",
                        "state.legal_action_count",
                        {
                            "minimum": len(normalized.legal_actions),
                            "maximum": len(normalized.legal_actions),
                        },
                    ),
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-PROOF",
                "proof.winning_root_action_count",
                {"minimum": count, "maximum": count},
            ),
        }
    )
    satisfied = ScenarioRuntimeVerifierV1().verify(
        scenario,
        normalized,
        proof_outcome=proof,
    )
    assert satisfied.status == "satisfied"
    assert satisfied.hard_predicates_satisfied is True
    missing = ScenarioRuntimeVerifierV1().verify(scenario, normalized)
    assert missing.status == "semantic_postcondition_miss"
    proof_row = next(
        item for item in missing.evidence if item.kind.startswith("proof.")
    )
    assert proof_row.status == "not_evaluated"


def test_runtime_verifies_extended_observable_predicate_inventory(rules):
    prefix = _late_prefix(rules)
    prefix["playActions"] = prefix["playActions"][:-1]
    normalized = ExactStateNormalizerV1(rules).normalize(prefix)
    state = normalized.round_state
    assert state.current_player is not None
    assert state.current_trick is not None
    assert state.current_trick.led_suit is not None
    assert state.completed_tricks and state.completed_tricks[0].winner is not None
    legal_card = int(normalized.legal_actions[0]["card"]["id"])
    owner = state.current_player.name.lower()
    trick = state.completed_tricks[0]
    constraints = [
        _predicate(
            "PRED-REMAINING-OWNER",
            "deal.remaining_card_owner",
            {"card": legal_card, "seat": owner},
        ),
        _predicate(
            "PRED-HIGH-BALANCE",
            "deal.team_high_card_balance",
            {
                "team": "north_south",
                "maximumDifference": 8,
                "handStage": "post_purchase",
            },
        ),
        _predicate(
            "PRED-PASSES",
            "auction.pass_count",
            {"minimum": 0, "maximum": 8},
        ),
        _predicate("PRED-AUCTION-END", "auction.terminal", {"terminal": True}),
        _predicate(
            "PRED-ESCALATION-LOCK",
            "escalation.locked",
            {"locked": state.locked},
        ),
        _predicate(
            "PRED-BALOOT",
            "project.baloot",
            {
                "team": "north_south",
                "achieved": Team.NORTH_SOUTH in state.baloot_teams,
            },
        ),
        _predicate(
            "PRED-OVERLAP",
            "project.overlap_count",
            {"owner": "north", "minimum": 0, "maximum": 32},
        ),
        _predicate(
            "PRED-LEADER",
            "play.current_trick_leader",
            {"seat": state.current_trick.leader.name.lower()},
        ),
        _predicate(
            "PRED-LED-SUIT",
            "play.current_trick_suit_led",
            {"suit": state.current_trick.led_suit.name.lower()},
        ),
        _predicate(
            "PRED-CURRENT-TRICK-CARD",
            "play.current_trick_card",
            {
                "position": 0,
                "seat": state.current_trick.plays[0].seat.name.lower(),
                "card": state.current_trick.plays[0].card.id,
            },
        ),
        _predicate(
            "PRED-TRICK-WINNER",
            "play.completed_trick_winner",
            {
                "trickIndex": trick.index,
                "seat": trick.winner.name.lower(),
            },
        ),
        _predicate(
            "PRED-RAW-NOW",
            "state.current_raw_points",
            {"team": "north_south", "minimum": 0, "maximum": 200},
        ),
        _predicate(
            "PRED-PROJECT-NOW",
            "state.current_project_points",
            {"team": "north_south", "minimum": 0, "maximum": 500},
        ),
        _predicate(
            "PRED-LEGAL-INCLUDE",
            "state.legal_cards_include",
            {"cards": [legal_card]},
        ),
        _predicate(
            "PRED-TACTIC-APPLICABLE",
            "tactic.applicable",
            {"kind": "small_card_big_gain", "applicable": True},
        ),
    ]
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "exact_trace_prefix", "exactTracePrefix": prefix},
            "constraints": {"op": "all", "children": constraints},
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-LEGAL-COUNT",
                "state.legal_action_count",
                {
                    "minimum": len(normalized.legal_actions),
                    "maximum": len(normalized.legal_actions),
                },
            ),
        }
    )
    plan = ScenarioCompilerV1().compile(scenario, _proof())
    assert plan.valid is True
    result = ScenarioRuntimeVerifierV1().verify(
        scenario,
        normalized,
        tactical_evidence={"small_card_big_gain": [{"applicable": True}]},
    )
    assert result.status == "satisfied"
    assert all(item.status == "satisfied" for item in result.evidence)
    assert all(
        row["requiredDomainPrimitives"]
        for row in ScenarioPredicateRegistryV1().catalog()["definitions"]
        if row["stage"]
        in {"deal", "auction", "escalation", "projects", "playPrefix", "outcome"}
    )


def test_structural_solver_proves_feasible_deal_and_exact_infeasible_core(rules):
    compiler = ScenarioStructuralCompilerV1()
    feasible = _scenario(
        _predicate(
            "PRED-CARD-N",
            "deal.card_owner",
            {"card": 0, "seat": "north", "handStage": "initial_five"},
        ),
        _predicate(
            "PRED-SUIT-N",
            "deal.suit_count",
            {
                "seat": "north",
                "suit": "spades",
                "minimum": 2,
                "maximum": 8,
                "handStage": "initial_five",
            },
        ),
    )
    result = compiler.solve(
        feasible,
        rules=rules,
        dealer=Seat.EAST,
        seed=91,
        maximum_seconds=10,
    )
    assert result.status == "satisfied"
    assert result.solver_status == "optimal"
    assert result.deck_order is not None
    engine = BalootEngine(rules)
    state = engine.create_round(seed=91, dealer=Seat.EAST)
    engine.initial_deal_from_preordered(state, result.deck_order)
    assert state.initial_five[int(Seat.NORTH)] & 1

    continued = compiler.solve(
        feasible,
        rules=rules,
        dealer=Seat.EAST,
        seed=91,
        maximum_seconds=10,
        excluded_deck_orders=[result.deck_order],
    )
    assert continued.status == "satisfied"
    assert continued.deck_order is not None
    assert continued.deck_order != result.deck_order
    assert continued.no_good_hashes == (canonical_sha256(list(result.deck_order)),)
    assert continued.to_dict()["noGoodHashes"] == list(continued.no_good_hashes)

    impossible = _scenario(
        _predicate(
            "PRED-CARD-N",
            "deal.card_owner",
            {"card": 0, "seat": "north", "handStage": "initial_five"},
        ),
        _predicate(
            "PRED-CARD-S",
            "deal.card_owner",
            {"card": 0, "seat": "south", "handStage": "initial_five"},
        ),
        _predicate(
            "PRED-IRRELEVANT",
            "deal.card_owner",
            {"card": 1, "seat": "east", "handStage": "initial_five"},
        ),
    )
    failed = compiler.solve(
        impossible,
        rules=rules,
        dealer=Seat.EAST,
        seed=92,
        maximum_seconds=10,
    )
    assert failed.status == "structurally_infeasible"
    assert failed.solver_status == "infeasible"
    assert failed.deck_order is None
    assert failed.infeasible_core_predicate_ids == (
        "PRED-CARD-N",
        "PRED-CARD-S",
    )


def test_structural_solver_prefers_actor_safe_buffers_for_protected_cards(rules):
    protected = Card.parse("SA")
    scenario = _scenario(
        _predicate("PRED-BUYER", "auction.buyer", {"seat": "south"}),
        _predicate("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
        _predicate(
            "PRED-OWNER",
            "deal.card_owner",
            {
                "card": protected.id,
                "seat": "north",
                "handStage": "post_purchase",
            },
        ),
        _predicate(
            "PRED-REMAINING",
            "deal.remaining_card_owner",
            {"card": protected.id, "seat": "north"},
        ),
    )
    result = ScenarioStructuralCompilerV1().solve(
        scenario,
        rules=rules,
        dealer=Seat.EAST,
        seed=93,
        maximum_seconds=10,
    )
    assert result.status == "satisfied"
    assert result.deck_order is not None
    layout = DealLayoutV1.build(
        rules.structural_domain,
        dealer=Seat.EAST,
        buyer=Seat.SOUTH,
    )
    north = [
        Card(result.deck_order[position])
        for position in layout.positions_for(Seat.NORTH, final=True)
    ]
    assert protected in north
    assert sum(card.suit == protected.suit for card in north) >= 2


def test_structural_solver_guides_multiple_engine_verified_projects(rules):
    scenario = _scenario(
        _predicate("PRED-BUYER", "auction.buyer", {"seat": "south"}),
        _predicate("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
        _predicate(
            "PRED-PROJECT",
            "project.exists",
            {
                "team": "north_south",
                "owner": "south",
                "kind": "hundred_sequence",
                "value": 20,
                "state": "eligible",
            },
        ),
        _predicate(
            "PRED-SECOND-PROJECT",
            "project.exists",
            {
                "team": "north_south",
                "owner": "north",
                "kind": "four_hundred",
                "value": 40,
                "state": "eligible",
            },
        ),
    )
    compiler = ScenarioStructuralCompilerV1()
    result = compiler.solve(
        scenario,
        rules=rules,
        dealer=Seat.EAST,
        seed=100,
        maximum_seconds=1,
    )
    assert result.status == "satisfied"
    assert result.solver_status == "optimal"
    repeated = compiler.solve(
        scenario,
        rules=rules,
        dealer=Seat.EAST,
        seed=100,
        maximum_seconds=1,
    )
    varied = compiler.solve(
        scenario,
        rules=rules,
        dealer=Seat.EAST,
        seed=101,
        maximum_seconds=1,
    )
    assert repeated.deck_order == result.deck_order
    assert varied.deck_order != result.deck_order
    project_evidence = next(
        item
        for item in result.predicate_evidence
        if item["predicateId"] == "PRED-PROJECT"
    )
    assert project_evidence["lowering"] == "cp_sat_exact_eligible_project_pattern"

    engine = BalootEngine(rules)
    state = engine.create_round(seed=100, dealer=Seat.EAST)
    engine.initial_deal_from_preordered(state, result.deck_order)
    assert state.purchase_card is not None
    machine = BiddingMachine(
        state.dealer,
        state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(rules.data),
    )
    auction = OrToolsCpSatAuctionCompilerV1().compile(
        dealer=state.dealer,
        purchase_suit=state.purchase_card.suit,
        rules=rules,
        required_buyer=Seat.SOUTH,
        required_contracts=("sun",),
    )
    assert auction is not None
    for action in auction.actions:
        machine.submit(action.action, trump=action.trump)
    assert machine.result is not None
    engine.complete_deal_from_bid_result(state, machine.result)
    engine.prepare_projects(state)
    assert any(
        project.owner == Seat.SOUTH
        and project.kind == "hundred_sequence"
        and project.value == 20
        for project in state.projects
    )
    assert any(
        project.owner == Seat.NORTH
        and project.kind == "four_hundred"
        and project.value == 40
        for project in state.projects
    )


def test_v4_rules_applicability_rejects_unknown_project_before_execution(rules):
    request = _proof()
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-CONTRACT",
                        "auction.contract",
                        {"contract": "sun"},
                    ),
                    _predicate(
                        "PRED-UNKNOWN-PROJECT",
                        "project.exists",
                        {
                            "team": "north_south",
                            "kind": "imaginary_hundred",
                            "value": 100,
                            "state": "eligible",
                        },
                    ),
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-PROOF",
                "proof.winning_root_action_count",
                {"minimum": 0, "maximum": 8},
            ),
        }
    )
    plan = ScenarioCompilerV1().compile(scenario, request)
    recipe = ResearchRecipeV4(
        "RCP-V4RULES1",
        "Ø³ÙŠÙ†Ø§Ø±ÙŠÙˆ ÙŠØ±ÙØ¶ Ù…Ø´Ø±ÙˆØ¹Ù‹Ø§ ØºÙŠØ± Ù…Ø¯Ø¹ÙˆÙ…",
        {"bundleRevisionId": "CBR-TEST"},
        scenario,
        request,
        {},
        {
            "generationLane": "ortools",
            "generationPolicy": "natural_policy",
            "seedStart": 1,
            "seedCount": 1,
            "seedsPerShard": 1,
            "candidatesPerSeed": 1,
            "solverTimeoutSeconds": 1.0,
            "maximumConflicts": 1,
            "analysisMode": "cascaded",
            "proofBudgetProfile": "late_game_exact_v1",
        },
        {},
        {"mode": "on_demand"},
        "published",
        plan,
        "f" * 64,
    )
    issues = ResearchFoundryServiceV1._scenario_rules_applicability_issues(
        recipe,
        rules,
    )
    assert [item["code"] for item in issues] == [
        "project_kind_value_not_enabled_by_rules"
    ]
    assert issues[0]["predicateIds"] == ["PRED-UNKNOWN-PROJECT"]


def test_bounded_v4_exact_prefix_retains_qualified_result(rules):
    scenario, normalized = _retained_late_play_scenario(rules)
    actor = normalized.round_state.current_player
    assert actor is not None
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "bounded",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": actor.team.name.lower(),
            "utility": "awarded_point_margin",
            "successThreshold": -1_000_000,
            "horizon": {"unit": "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1",
            "budgetProfile": "bounded_v1",
        }
    )
    recipe = _exact_scenario_recipe(
        "RCP-V4BOUNDED1",
        "إثبات محدود محفوظ",
        scenario,
        proof,
    )

    result, harness = _run_exact_scenario_recipe(rules, recipe)

    assert harness.store.run["status"] == "completed"
    assert harness.store.run["resultCount"] == 1
    assert harness.store.run["qualifiedCount"] == 1
    assert harness.store.results == [result]
    assert result["disposition"] == "qualified"
    assert result["truthLabel"] == "robust_best_known"
    assert result["scenarioSatisfaction"]["status"] == "satisfied"
    assert result["qualification"]["qualified"] is True
    assert result["qualification"]["observerSafe"] is True
    assert result["scenarioProjection"]["origin"] == "exact_trace_prefix"
    assert result["scenarioProjection"]["requestedProofMode"] == "bounded"
    assert result["scenarioProjection"]["achievedProofMode"] == "bounded"
    assert result["scenarioProjection"]["proofCompletion"] == "bounded_complete"
    assert result["scenarioProjection"]["proofVisibility"] == "complete_state"
    assert result["proofOutcome"]["claimSatisfied"] is True
    assert result["proofOutcome"]["horizon"] == {"unit": "plies", "value": 1}
    assert (
        result["proofOutcome"]["boundaryEvaluator"]
        == "current_raw_point_margin_v1"
    )
    assert (
        validate_scenario_result_v4(
            result,
            artifact_store=harness.artifact_store,
        )
        == result
    )


def test_policy_conditioned_v4_exact_prefix_retains_qualified_result(rules):
    scenario, normalized = _retained_late_play_scenario(rules)
    actor = normalized.round_state.current_player
    assert actor is not None
    proof = _policy_proof_for(actor.team.name.lower())
    recipe = _exact_scenario_recipe(
        "RCP-V4POLICY01",
        "إثبات مشروط بالسياسات محفوظ",
        scenario,
        proof,
    )

    result, harness = _run_exact_scenario_recipe(rules, recipe)

    assert harness.store.run["status"] == "completed"
    assert harness.store.run["resultCount"] == 1
    assert harness.store.run["qualifiedCount"] == 1
    assert harness.store.results == [result]
    assert result["disposition"] == "qualified"
    assert result["truthLabel"] == "robust_best_known"
    assert result["scenarioSatisfaction"]["status"] == "satisfied"
    assert result["qualification"]["qualified"] is True
    assert result["qualification"]["observerSafe"] is True
    assert result["scenarioProjection"]["origin"] == "exact_trace_prefix"
    assert (
        result["scenarioProjection"]["requestedProofMode"]
        == "policy_conditioned"
    )
    assert (
        result["scenarioProjection"]["achievedProofMode"]
        == "policy_conditioned"
    )
    assert result["scenarioProjection"]["proofCompletion"] == "policy_complete"
    assert result["scenarioProjection"]["proofVisibility"] == "complete_state"
    assert result["proofOutcome"]["claimSatisfied"] is True
    assert result["proofOutcome"]["continuationLines"] == {
        "complete": True,
        "exactCount": "1",
        "lineUniverse": "policy_conditioned",
    }
    assert result["proofOutcome"]["identities"]["policyHashes"] == {
        seat: profile["profileHash"] for seat, profile in proof.policies.items()
    }
    assert (
        validate_scenario_result_v4(
            result,
            artifact_store=harness.artifact_store,
        )
        == result
    )
