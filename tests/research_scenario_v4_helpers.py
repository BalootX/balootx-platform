from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import yaml

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import GamePhase, Seat
from baloot_platform.research.exact_state import ExactStateNormalizerV1
from baloot_platform.research.scenario_contracts import (
    ResearchScenarioV1,
    ScenarioProofRequestV1,
    ScenarioPlayingQualityPolicyV1,
    SCENARIO_DECISION_KINDS,
    canonical_sha256,
)
from baloot_platform.research.scenario_registry import (
    ScenarioCompilerV1,
)
from baloot_platform.research.recipe_v4 import ResearchRecipeV4
from baloot_platform.research.service import (
    ResearchFoundryServiceV1,
)
from baloot_platform.service import api
from scripts.benchmark_ortools_research_v4 import _publish_explicit_bundle


_ORTOOLS_V4_REVISION_ID: str | None = None


_API_CP_SAT_BUDGET_SECONDS = 900.0


def _published_ortools_v4_revision_id() -> str:
    global _ORTOOLS_V4_REVISION_ID
    if _ORTOOLS_V4_REVISION_ID is None:
        root = Path(__file__).parents[1]
        authority, context = _publish_explicit_bundle(
            api.CONTAINER,
            root,
            {
                "bundle": {
                    "id": "baloot_ortools_research_native_v3",
                    "source": "configs/bundle_intelligent_research.yaml",
                    "backend": "ortools_cp_sat_v3",
                }
            },
        )
        configuration = api.CONTAINER.configuration
        challenge = copy.deepcopy(context.bundle.component("challenge"))
        historical = yaml.safe_load((root / "configs/challenge/progressive_mining_ortools_v3_1.yaml").read_text(encoding="utf-8"))
        challenge["scenarioResearchV4"]["searchBudgets"]["profiles"]["late_game_exact_v1"] = (
            historical["scenarioResearchV4"]["searchBudgets"]["profiles"]["late_game_exact_v1"]
        )
        assert challenge["scenarioResearchV4"]["qualityExecution"]["featuredPlayNativeProof"] is True
        assert challenge["scenarioResearchV4"]["searchBudgets"]["profiles"]["late_game_exact_v1"]["maximumWallSeconds"] == 60
        _apply_measured_api_budgets(challenge)
        component = configuration.create_profile(
            family="challenge", payload=challenge, created_by="TEST-RESEARCH-API", publish=True,
            scope="environment", environment="research-api-quality-fixture",
        )
        base = configuration.store.get_bundle(authority.bundle_revision_id)
        assert base is not None
        fixture = configuration.create_bundle(
            bundle_id="baloot_research_api_quality_fixture", version="1.0.0", mode=base.mode,
            component_revisions={**base.component_revisions, "challenge": component["revisionId"]},
            created_by="TEST-RESEARCH-API", publish=True,
            scope="environment", environment="research-api-quality-fixture",
        )
        _ORTOOLS_V4_REVISION_ID = fixture["revisionId"]
    return _ORTOOLS_V4_REVISION_ID


def _apply_measured_api_budgets(challenge: dict) -> dict:
    before = copy.deepcopy(challenge)
    scenario = challenge["scenarioResearchV4"]
    scenario["qualityExecution"]["maximumWallSeconds"] = max(
        float(scenario["qualityExecution"]["maximumWallSeconds"]),
        _API_CP_SAT_BUDGET_SECONDS,
    )
    for group, keys in (
        ("proofBudgets", ("maximumWallSeconds", "maximumCpuSeconds")),
        ("searchBudgets", ("maximumWallSeconds", "maximumDeterministicTime")),
    ):
        for profile in scenario[group]["profiles"].values():
            for key in keys:
                profile[key] = max(float(profile[key]), _API_CP_SAT_BUDGET_SECONDS)
                assert profile[key] >= _API_CP_SAT_BUDGET_SECONDS

    def admission_projection(value: dict) -> dict:
        projected = copy.deepcopy(value)
        section = projected["scenarioResearchV4"]
        section["qualityExecution"].pop("maximumWallSeconds", None)
        for group, keys in (
            ("proofBudgets", ("maximumWallSeconds", "maximumCpuSeconds")),
            ("searchBudgets", ("maximumWallSeconds", "maximumDeterministicTime")),
        ):
            for profile in section[group]["profiles"].values():
                for key in keys:
                    profile.pop(key, None)
        return projected

    assert admission_projection(before) == admission_projection(challenge)
    return challenge


def _api_budgeted_quality_policy() -> dict:
    return {"schemaVersion": 1, "mode": "budgeted", "decisions": {
        kind: {
            "utility": "awarded_point_margin", "budgetProfile": "bounded_v1",
            "horizon": {"unit": "tricks" if kind == "play" else "plies", "value": 1},
            "boundaryEvaluator": "auction_hand_margin_v1" if kind == "bid" else "current_raw_point_margin_v1",
        } for kind in SCENARIO_DECISION_KINDS
    }}


def _api_budgeted_quality_prefix(root: Path) -> dict:
    """Keep positive input bound to the current bounded policy, not its filename."""
    fixture = json.loads((root / "tests/fixtures/research-native-observer-budgeted-round-v3.json").read_text(encoding="utf-8"))
    assert fixture["playingQuality"] == ScenarioPlayingQualityPolicyV1.build(_api_budgeted_quality_policy()).to_dict()
    assert fixture["maximumHiddenWorldsPerDecision"] == 8
    return fixture["exactTracePrefix"]


def _predicate(predicate_id: str, kind: str, arguments: dict) -> dict:
    return {
        "predicateId": predicate_id,
        "kind": kind,
        "version": 1,
        "hardness": "hard",
        "arguments": arguments,
    }


def _assert_public_launch_admission(admission: dict, run_code: str) -> None:
    run = api.RESEARCH.store.get_run(run_code)
    assert run is not None
    receipt = run["launchAdmission"]
    assert admission["sha256"] == receipt["sha256"] == canonical_sha256({
        key: value for key, value in receipt.items() if key != "sha256"
    })
    assert admission == api.ResearchRunLaunchAdmissionResponse.model_validate(receipt).model_dump(exclude_none=True)


def _proof(**overrides) -> ScenarioProofRequestV1:
    value = {
        "schemaVersion": 1,
        "mode": "exact",
        "visibility": "complete_state",
        "claim": "winning_root_action_count",
        "targetTeam": "north_south",
        "utility": "awarded_point_margin",
        "successThreshold": 1,
        "lineUniverse": "all_legal",
        "budgetProfile": "late_game_exact_v1",
    }
    value.update(overrides)
    if value.get("visibility") == "actor_information_set":
        value.setdefault(
            "strategyClass",
            "frozen_policy"
            if value.get("mode") == "policy_conditioned"
            else "pure_deterministic",
        )
    return ScenarioProofRequestV1.build(value)


def _published_v4_configuration():
    revision_id = api.CONTAINER.configuration.resolve(bundle_id="baloot_default")[
        "bundle"
    ]["revisionId"]
    authority, context = api.RESEARCH.execution_context_factory.freeze_selector(
        bundle_revision_id=revision_id,
        analysis_mode="cascaded",
    )
    return authority.to_dict(), context


def _scenario(*predicates: dict) -> ResearchScenarioV1:
    return ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {"op": "all", "children": list(predicates)},
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-OUTCOME",
                "outcome.team_round",
                {"team": "north_south", "status": "won"},
            ),
        }
    )


def _late_prefix(rules) -> dict:
    deck = list(range(32))
    engine = BalootEngine(rules)
    state = engine.create_round(seed=8172, dealer=Seat.EAST)
    engine.initial_deal_from_preordered(state, deck)
    assert state.purchase_card is not None
    state.phase = GamePhase.BIDDING
    machine = BiddingMachine(
        state.dealer,
        state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(rules.data),
    )
    auction_actions = [
        {"action": "pass"},
        {"action": "pass"},
        {"action": "sun"},
        {"action": "pass"},
        {"action": "sun"},
    ]
    for action in auction_actions:
        machine.submit(action["action"])
    assert machine.result is not None
    engine.complete_deal_from_bid_result(state, machine.result)
    engine.prepare_projects(state)
    bot = HeuristicBot(engine, BotProfile(8172))
    play_actions = []
    while sum(mask.bit_count() for mask in state.hands) > 4:
        assert state.current_player is not None
        card = bot.choose(state)
        play_actions.append(
            {"actor": state.current_player.name.lower(), "card": card.code}
        )
        engine.play_card(state, card)
    project_declarations = [
        {
            key: event[key]
            for key in (
                "owner",
                "kind",
                "value",
                "highRank",
                "declarationOrder",
            )
        }
        for event in state.to_dict(reveal_hands=True)["events"]
        if event["type"] == "project_declared"
    ]
    return {
        "schemaVersion": 1,
        "deckOrder": deck,
        "dealer": "east",
        "roundNumber": 1,
        "matchScore": [0, 0],
        "auctionActions": auction_actions,
        "escalationActions": [],
        "projectDeclarations": project_declarations,
        "projectAssertions": [],
        "playActions": play_actions,
        "stateAssertions": {"phase": "playing"},
    }


def _policy_proof_for(team: str) -> ScenarioProofRequestV1:
    policies = ResearchFoundryServiceV1._scenario_recipe_configuration()[
        "defaultPolicyProfiles"
    ]
    return ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": team,
            "utility": "awarded_point_margin",
            "successThreshold": -10000.0,
            "lineUniverse": "policy_conditioned",
            "policies": policies,
            "budgetProfile": "bounded_v1",
        }
    )


def ProofOutcomeRoundTrip(payload: dict) -> dict:
    from baloot_platform.research.scenario_contracts import ProofOutcomeV1

    return ProofOutcomeV1.build(payload).to_dict()


class _ScenarioArtifactStore:
    def __init__(self):
        self.values = {}

    def put(self, data, *, media_type, name=None):
        digest = hashlib.sha256(data).hexdigest()
        uri = f"cas://sha256/{digest}"
        self.values[uri] = bytes(data)
        return {
            "uri": uri,
            "sha256": digest,
            "bytes": len(data),
            "mediaType": media_type,
            "name": name,
        }

    def get(self, uri):
        return self.values[uri]


class _ScenarioRunStore:
    def __init__(self, run):
        self.run = dict(run)
        self.results = []

    def update_run(self, run_code, **changes):
        assert run_code == self.run["runCode"]
        self.run.update(changes)
        return dict(self.run)

    def get_run(self, run_code):
        assert run_code == self.run["runCode"]
        return dict(self.run)

    def put_result(self, value):
        self.results.append(dict(value))
        return dict(value)


def _run_exact_scenario_recipe(rules, recipe: ResearchRecipeV4):
    configuration, context = _published_v4_configuration()
    assert recipe.configuration == configuration
    assert context.bundle.rules.sha256 == rules.sha256
    run = {
        "runCode": "RUN-" + recipe.code.removeprefix("RCP-")[:12].ljust(12, "0"),
        "runHash": "a" * 64,
        "recipeCode": recipe.code,
        "recipeHash": recipe.sha256,
        "status": "queued",
        "progress": 0.0,
        "events": [],
        "createdAt": "2026-08-12T00:00:00.000000+00:00",
        "configurationAuthority": configuration,
    }
    harness = SimpleNamespace(
        store=_ScenarioRunStore(run),
        artifact_store=_ScenarioArtifactStore(),
        _cancel_check=lambda _run_code: False,
    )
    ResearchFoundryServiceV1._execute_v4_exact_run(harness, run, recipe, context)
    return harness.store.results[0], harness


def _exact_scenario_recipe(
    code: str,
    title: str,
    scenario: ResearchScenarioV1,
    proof: ScenarioProofRequestV1,
) -> ResearchRecipeV4:
    configuration, _context = _published_v4_configuration()
    return ResearchRecipeV4(
        code,
        title,
        configuration,
        scenario,
        proof,
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
            "proofBudgetProfile": proof.budget_profile,
        },
        {},
        {"mode": "on_demand"},
        "published",
        ScenarioCompilerV1().compile(scenario, proof),
        "f" * 64,
    )


def _retained_late_play_scenario(rules):
    prefix = _late_prefix(rules)
    normalized = ExactStateNormalizerV1(rules).normalize(prefix)
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": prefix,
            },
            "constraints": _predicate(
                "PRED-RETAINED-ACTOR",
                "play.current_player",
                {"seat": normalized.next_actor},
            ),
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-RETAINED-ROOT-COUNT",
                "proof.winning_root_action_count",
                {"minimum": 0, "maximum": len(normalized.legal_actions)},
            ),
        }
    )
    return scenario, normalized
