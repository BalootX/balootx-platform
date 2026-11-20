from __future__ import annotations

import ast
import copy
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient
from ortools.sat.python import cp_model
from pydantic import ValidationError

from baloot_platform.core.cards import FULL_DECK, Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.research import ortools_exact_state as exact_state_module
from baloot_platform.research import ortools_research_executor as executor_module
from baloot_platform.research.campaign_coordinator import (
    CampaignVariantV1,
    SEMANTIC_DISTANCE_POLICY_V1,
    build_rules_preserving_diversity_document_v1,
    decision_state_card_hamming_distance_v1,
    rules_preserving_symmetry_key_v1,
    select_campaign_variants_v1,
)
from baloot_platform.research.proof_service import ScenarioProofBudgetRegistryV1
from baloot_platform.research.recipe_v4 import ResearchRecipeV4
from baloot_platform.research.ortools_research_executor import (
    OrToolsPredicateEvaluationV1,
    OrToolsResearchCampaignResultV1,
    OrToolsResearchPreflightV1,
    OrToolsUnifiedResearchExecutorV1,
    OrToolsUnifiedResearchResultV1,
)
from baloot_platform.research.ortools_proof import (
    OrToolsGraphProofOutcomeV1,
    OrToolsScenarioGraphProofV1,
)
from baloot_platform.research.ortools_information_set_proof import (
    OrToolsActorInformationSetProofV1,
)
from baloot_platform.research.render_service import (
    ResearchRenderPreparationServiceV1,
)
from baloot_platform.research.result_authority import RuntimeCompatibilityV1
from baloot_platform.research.runtime_compatibility import (
    RuntimeCompatibilityAdmissionError,
    check_runtime_compatibility_v1,
)
from baloot_platform.research.scenario_contracts import (
    ResearchScenarioV1,
    ScenarioProofRequestV1,
    canonical_sha256,
)
from baloot_platform.research.scenario_backend import (
    ORTOOLS_SCENARIO_RESEARCH_BACKEND,
)
from baloot_platform.research.scenario_predicate_manifest_v2 import (
    ScenarioPredicateCatalogLoweringManifestV2,
)
from baloot_platform.research.service import (
    ResearchFoundryServiceV1,
    ResearchLaunchAdmissionError,
    _operator_research_run_v1,
)
from baloot_platform.research.store import SQLiteResearchStoreV1
from baloot_platform.research.tail_first_stage_planner import TailFirstStagePlannerV1
from baloot_platform.service import api
from baloot_platform.service.api import (
    ResearchRecipeEstimateResponse,
    ResearchRecipeValidationResponse,
    ScenarioResearchV4CapabilityV1Response,
)
from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore
from baloot_platform.video.batch_service import RenderBatchServiceV1
from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer


@pytest.fixture
def inline_v3_shard_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep semantic monkeypatches visible in focused persistence tests."""

    def run(task, request, *, cancellation_poll_milliseconds):
        del cancellation_poll_milliseconds
        executor = OrToolsUnifiedResearchExecutorV1(
            rules=task["rules"],
            bundle_revision_id=str(task["bundleRevisionId"]),
            bundle_hash=str(task["bundleHash"]),
            backend_id=str(task["backendId"]),
            adjudication_authority=task.get("adjudicationAuthority"),
            rule_source_catalog=task.get("ruleSourceCatalog"),
            witness_policies=task.get("witnessPolicies"),
            quality_policy=task.get("qualityPolicy"),
            quality_execution=task.get("qualityExecution"),
            quality_proof_profiles=task.get("qualityProofProfiles"),
            quality_search_profiles=task.get("qualitySearchProfiles"),
        )
        return executor._execute_seed_campaign(
            task["scenario"],
            task["proofRequest"],
            cancel_check=request.cancellation_requested,
            progress_callback=request.report_progress,
            checkpoint_callback=request.report_checkpoint,
            **dict(task["options"]),
        )

    monkeypatch.setattr(executor_module, "_run_seed_campaign_process_v1", run)


@pytest.mark.parametrize("child_reported_terminal", (False, True))
def test_v3_shard_transport_preserves_returned_terminal_receipt(
    monkeypatch: pytest.MonkeyPatch, child_reported_terminal: bool
) -> None:
    execution = SimpleNamespace(
        status="incomplete",
        candidate=None,
        proof=None,
        execution_receipt={"completedCandidates": 0, "incompleteCompilation": None},
    )
    terminal = executor_module._candidate_terminal_progress_v1(
        execution, ordinal=1, decision_kind="play"
    )
    messages = ([('progress', terminal)] if child_reported_terminal else []) + [
        ("result", SimpleNamespace(executions=(execution,)))
    ]

    class Parent:
        def poll(self, _timeout=None):
            return bool(messages)

        def recv(self):
            return messages.pop(0)

        def close(self):
            pass

    class Process:
        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    class Context:
        def Pipe(self, duplex=True):
            return Parent(), SimpleNamespace(close=lambda: None)

        def Process(self, **_kwargs):
            return Process()

    monkeypatch.setattr(executor_module.multiprocessing, "get_context", lambda _kind: Context())
    progress = []
    request = SimpleNamespace(
        report_progress=progress.append, cancellation_requested=lambda: False,
        evidence_transfer_lock=executor_module.threading.RLock(),
    )
    executor_module._run_seed_campaign_process_v1(
        {"backendId": "ortools_cp_sat_v3", "options": {},
         "scenario": SimpleNamespace(decision_point={"kind": "play"})},
        request,
        cancellation_poll_milliseconds=25,
    )
    assert progress == [terminal]


class _ChallengeBundleProxy:
    def __init__(self, bundle, challenge: dict, *, preserve_bundle_inputs: bool = False) -> None:
        self._bundle = bundle
        self._challenge = copy.deepcopy(challenge)
        self.sha256 = bundle.sha256
        self.bundle_profile = bundle.bundle_profile
        self.rules = bundle.rules if preserve_bundle_inputs else (
            RuleBook.load(
                Path(__file__).resolve().parents[1]
                / "configs/rules/saudi_federation_2023_v2.yaml"
            )
            if challenge.get("scenarioResearchV4", {}).get("backend")
            == "ortools_cp_sat_v3"
            else bundle.rules
        )
        self.certification = bundle.component("certification") if preserve_bundle_inputs else (
            yaml.safe_load((Path(__file__).resolve().parents[1]
                / "configs/certification/rules_saudi_federation_2023_v2.yaml").read_text(encoding="utf-8"))
            if challenge.get("scenarioResearchV4", {}).get("backend") == "ortools_cp_sat_v3"
            else bundle.component("certification")
        )

    def component(self, family: str):
        if family == "challenge":
            return copy.deepcopy(self._challenge)
        if family == "rules":
            return copy.deepcopy(self.rules.data)
        if family == "certification":
            return copy.deepcopy(self.certification)
        return self._bundle.component(family)

    def __getattr__(self, name: str):
        return getattr(self._bundle, name)

    def configuration_snapshot(self):
        snapshot = self._bundle.configuration_snapshot()
        for family, payload in (
            ("challenge", self._challenge), ("rules", self.rules.data),
            ("certification", self.certification),
        ):
            snapshot["components"][family] = {
                "payload": copy.deepcopy(payload),
                "sha256": self.rules.sha256 if family == "rules" else canonical_sha256(payload),
            }
        snapshot["sha256"] = canonical_sha256(
            {key: value for key, value in snapshot.items() if key != "sha256"}
        )
        return snapshot


def test_native_service_preserves_bound_catalog_revision_and_root(root, tmp_path):
    from baloot_platform.core.rule_source_catalog import RuleSourceCatalogV1

    _, context = _frozen_context_and_authority()
    bundle = _ChallengeBundleProxy(context.bundle, _ortools_challenge())
    binding = bundle.certification["rule_source_catalog"]
    document = yaml.safe_load((root / "configs" / binding["path"]).read_text(encoding="utf-8"))
    document["catalog"]["version"] = "1.0.1"
    selected = RuleSourceCatalogV1(document, bundle.rules)
    binding.update(version="1.0.1", sha256=selected.sha256)
    for relative, payload in (
        (binding["path"], document),
        ("rules/source_catalog_unselected.yaml", {**document,
            "catalog": {**document["catalog"], "version": "1.0.2"}}),
    ):
        path = tmp_path / "configs" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    for source in document["sources"]:
        if source["snapshot_url"].startswith("repo://"):
            relative = source["snapshot_url"].removeprefix("repo://")
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((root / relative).read_bytes())
    backend = ResearchFoundryServiceV1._ortools_scenario_backend_v2(
        replace(context, bundle=bundle), repository_root=tmp_path,
    )
    assert backend._rule_source_catalog == {"data": selected.data, "sha256": selected.sha256}


@pytest.mark.real_clock
@pytest.mark.parametrize("status", ["running", "queued", "completed", "failed", "cancelled"])
def test_operator_run_clock_is_live_only_while_running(monkeypatch, status) -> None:
    from datetime import datetime, timezone
    from baloot_platform.research import service

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 20, 12, 0, 10, tzinfo=timezone.utc)

    monkeypatch.setattr(service, "datetime", Clock)
    progress = {"elapsedSeconds": 3.0, "maximumCampaignSeconds": 600.0,
                "remainingCampaignSeconds": 597.0}
    raw = {"status": status, "campaignProgress": progress,
           "campaignAccounting": dict(progress), "events": [{
               "type": "progress", "at": "2026-09-20T12:00:00+00:00",
               "accounting": dict(progress)}]}
    original = copy.deepcopy(raw)
    projected = _operator_research_run_v1(raw)
    expected = 13.0 if status == "running" else 3.0
    assert projected["campaignProgress"]["elapsedSeconds"] == expected
    assert projected["campaignProgress"]["remainingCampaignSeconds"] == 600 - expected
    assert projected["campaignAccounting"]["elapsedSeconds"] == 3.0
    assert raw == original
    raw["events"][0]["at"] = "invalid"
    assert _operator_research_run_v1(raw)["campaignProgress"]["elapsedSeconds"] == 3.0


def test_operator_run_projection_is_bounded_and_preserves_owned_evidence() -> None:
    class DoNotCopy(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("operator view copied discarded evidence")

    large_trace = {"recordedTrace": "x" * 1_000_000}
    incomplete = {
        "stage": "exact_normalization",
        "exhaustionReason": "maximum_cpu_seconds",
        "sha256": "a" * 64,
        "evidence": large_trace,
    }
    raw = {
        "scenarioResearchEvidence": {"sha256": "b" * 64, "uri": "cas://evidence"},
        "performanceReceipt": DoNotCopy({"coordinatorReceipt": {"attempts": [incomplete]}}),
        "campaignProgress": {"latestIncompleteCompilation": incomplete},
        "campaignAccounting": {"latestIncompleteCompilation": incomplete},
        "events": [{"type": "completed", "proofBoundaryStatus": "infeasible",
                    "proofBoundaryReason": "tail_boundary_infeasible",
                    "campaignAccounting": DoNotCopy({"large": large_trace})},
                   {"type": "progress", "proofBoundaryReason": "x" * 1_000_000}],
    }

    projected = _operator_research_run_v1(raw)

    assert len(json.dumps(projected)) < 2_000
    assert "performanceReceipt" not in projected
    assert projected["scenarioResearchEvidence"] == raw["scenarioResearchEvidence"]
    assert projected["campaignAccounting"]["latestIncompleteCompilation"] == {
        "stage": "exact_normalization",
        "exhaustionReason": "maximum_cpu_seconds",
        "sha256": "a" * 64,
    }
    assert projected["events"] == [
        {"type": "completed", "proofBoundaryStatus": "infeasible",
         "proofBoundaryReason": "tail_boundary_infeasible"},
        {"type": "progress"},
    ]
    assert raw["campaignAccounting"]["latestIncompleteCompilation"]["evidence"] == large_trace

@pytest.mark.parametrize('status,core,expected', [
    ('seed_miss', ['decision_actor'], ['decision_actor']),
    ('infeasible', ['decision_actor', 'scenario_expression'], ['decision_actor', 'scenario_expression']),
    ('seed_miss', [], []),
    ('incomplete', ['decision_actor'], None),
    ('proposal', ['decision_actor'], None),
    ('seed_miss', ['SA'], None),
    ('seed_miss', [{'privateHand': ['SA']}], None),
    ('seed_miss', ['decision_actor', 'decision_actor'], None),
    ('seed_miss', ['scenario_expression'] * 3, None),
])
def test_operator_family_conflicts_expose_only_bounded_constraint_groups(status, core, expected):
    raw = {'status': 'completed', 'events': [{'type': 'progress', 'currentStage': 'family_proposal',
        'constructionStatus': status, 'constructionInfeasibleCore': core,
        'privateModel': {'unplayedCards': ['SA']}}]}
    event = _operator_research_run_v1(raw)['events'][0]
    assert event.get('constructionInfeasibleCore') == expected
    assert 'privateModel' not in event


def test_operator_run_auction_openings_are_safe_ordered_hints_from_completed_events() -> None:
    class DoNotCopy(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("operator view copied private auction evidence")

    def event(seed, dealer):
        return {"type": "progress", "currentStage": "family_proposal",
                "screenedSeed": seed, "screenedDealer": dealer,
                "constructionReason": "decision_actor_mismatch",
                "auctionQualification": {"status": "complete", "contract": "sun",
                    "allPassed": False, "buyer": "west",
                    "privateEvidence": DoNotCopy({"hands": "hidden"})}}

    raw = {"status": "completed", "scenarioResearchStatus": "not_found_within_budget",
           "auctionOpenings": [{"seed": 999, "dealer": "north", "private": "forged"}],
           "events": [event(6, "south"), event(6, "south"), event(6, "west"), event(8, "east")]}
    projected = _operator_research_run_v1(raw)
    assert projected["auctionOpenings"] == [
        {"seed": 6, "dealer": "south"}, {"seed": 6, "dealer": "west"},
        {"seed": 8, "dealer": "east"}]
    assert projected["events"] == [{"type": "progress", "currentStage": "family_proposal"}] * 4
    assert raw["auctionOpenings"][0]["seed"] == 999


@pytest.mark.parametrize("status", ["queued", "running", "failed", "cancelled", None])
def test_operator_run_auction_openings_are_not_exposed_before_completion(status) -> None:
    event = {"type": "progress", "currentStage": "family_proposal",
             "screenedSeed": 6, "screenedDealer": "south",
             "auctionQualification": {"status": "complete", "contract": "sun", "allPassed": False}}
    raw = {"status": status, "auctionOpenings": [{"seed": 6, "dealer": "south"}], "events": [event]}
    assert "auctionOpenings" not in _operator_research_run_v1(raw)


@pytest.mark.parametrize("mutation", [
    {"screenedSeed": True}, {"screenedSeed": "6"}, {"screenedSeed": 6.0},
    {"screenedSeed": -1}, {"screenedSeed": None}, {"screenedDealer": []},
    {"screenedDealer": "South"}, {"screenedDealer": None},
    {"type": "candidate"}, {"currentStage": "runtime_proposal"},
    {"auctionQualification": None},
    {"auctionQualification": {"status": "incomplete", "contract": "sun", "allPassed": False}},
    {"auctionQualification": {"status": "complete", "contract": "hokum", "allPassed": False}},
    {"auctionQualification": {"status": "complete", "contract": "sun", "allPassed": True}},
    {"auctionQualification": {"status": "complete", "contract": "sun", "allPassed": 0}},
    {"auctionQualification": {"status": "complete", "contract": "sun"}},
])
def test_operator_run_auction_openings_reject_malformed_or_unqualified_events(mutation) -> None:
    event = {"type": "progress", "currentStage": "family_proposal",
             "screenedSeed": 6, "screenedDealer": "south",
             "auctionQualification": {"status": "complete", "contract": "sun", "allPassed": False}}
    assert "auctionOpenings" not in _operator_research_run_v1({
        "status": "completed", "events": [{**event, **mutation}]})


@pytest.mark.parametrize("receipt", [{"coordinatorReceipt": {"private": "x" * 100_000}}, {"cpuSeconds": 1}])
def test_operator_store_reads_skip_private_receipt_before_decode(tmp_path, monkeypatch, receipt):
    from baloot_platform.research import store as store_module

    store = SQLiteResearchStoreV1(tmp_path / "operator.sqlite3")
    store.put_recipe({"code": "RCP-OPERATOR", "sha256": "a" * 64, "status": "published"})
    original = store.create_run({
        "runCode": "RUN-OPERATOR", "runHash": "b" * 64,
        "recipeCode": "RCP-OPERATOR", "recipeHash": "a" * 64,
        "status": "completed", "progress": 1, "performanceReceipt": receipt,
        "scenarioResearchEvidence": {"sha256": "c" * 64}, "events": [],
    })
    service = object.__new__(ResearchFoundryServiceV1)
    service.store = store
    decode = store_module._decode
    decoded_lengths = []

    def record_decode(value):
        decoded_lengths.append(len(value))
        return decode(value)

    monkeypatch.setattr(store_module, "_decode", record_decode)
    expected = _operator_research_run_v1(original)
    assert service.get_operator_run("RUN-OPERATOR") == expected
    assert service.list_operator_runs(status="completed", limit=1) == [expected]
    assert max(decoded_lengths) < 2000
    assert service.get_operator_run("missing") is None
    assert service.list_operator_runs(status="running") == []
    assert store.get_run("RUN-OPERATOR") == original
    assert store.list_runs() == [original]


class _FrozenContextFactory:
    root = Path(__file__).resolve().parents[1]
    def __init__(self, authority, context) -> None:
        self.authority = authority
        self.context = context

    def freeze_selector(self, **_kwargs):
        return self.authority, self.context

    def resolve_recipe(self, _recipe):
        return self.context

    def resolve_bundle_revision(self, bundle_revision_id, *, require_published=False):
        assert bundle_revision_id == self.context.bundle_revision_id
        return self.authority, self.context


def _ortools_challenge(*, quality: bool = True) -> dict:
    root = Path(__file__).resolve().parents[1]
    challenge = yaml.safe_load(
        (root / "configs/challenge/progressive_mining_ortools_v3_1.yaml").read_text(
            encoding="utf-8"
        )
    )
    if quality:
        intelligent = yaml.safe_load(
            (root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8")
        )
        execution = copy.deepcopy(intelligent["scenarioResearchV4"]["qualityExecution"])
        execution.pop("featuredPlayNativeProof", None)
        challenge["scenarioResearchV4"]["qualityExecution"] = execution
        native_profiles = intelligent["scenarioResearchV4"]["proofBudgets"]["profiles"]
        for name, profile in challenge["scenarioResearchV4"]["proofBudgets"]["profiles"].items():
            profile["nativeObserved"] = copy.deepcopy(native_profiles[name]["nativeObserved"])
    return challenge


def _apply_measured_nonplay_limits(
    challenge: dict,
    maximum_wall_seconds: float,
) -> dict:
    before = copy.deepcopy(challenge)
    scenario = challenge["scenarioResearchV4"]
    scenario["qualityExecution"]["maximumWallSeconds"] = max(
        float(scenario["qualityExecution"]["maximumWallSeconds"]),
        maximum_wall_seconds,
    )
    for name in ("interactive_validation_v1", "bounded_v1", "late_game_exact_v1"):
        proof_profile = scenario["proofBudgets"]["profiles"][name]
        search_profile = scenario["searchBudgets"]["profiles"][name]
        proof_profile.update(
            maximumWallSeconds=maximum_wall_seconds,
            maximumCpuSeconds=maximum_wall_seconds,
        )
        search_profile.update(
            maximumWallSeconds=maximum_wall_seconds,
            maximumDeterministicTime=maximum_wall_seconds,
        )
        assert proof_profile["maximumWallSeconds"] >= maximum_wall_seconds
        assert proof_profile["maximumCpuSeconds"] >= maximum_wall_seconds
        assert search_profile["maximumWallSeconds"] >= maximum_wall_seconds
        assert search_profile["maximumDeterministicTime"] >= maximum_wall_seconds
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


def _ortools_v2_challenge() -> dict:
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load(
        (root / "configs/challenge/progressive_mining_ortools.yaml").read_text(
            encoding="utf-8"
        )
    )


def _legacy_challenge() -> dict:
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load(
        (root / "configs/challenge/progressive_mining_v2.yaml").read_text(
            encoding="utf-8"
        )
    )


def _frozen_context_and_authority():
    resolved = api.CONTAINER.configuration.resolve(
        bundle_id="baloot_ortools_research_native_v3",
        environment="research-native-v3",
    )
    revision_id = resolved["bundle"]["revisionId"]
    authority, context = api.RESEARCH.execution_context_factory.freeze_selector(
        bundle_revision_id=str(revision_id),
        analysis_mode="cascaded",
    )
    return authority, replace(
        context,
        bundle=_ChallengeBundleProxy(context.bundle, _ortools_challenge()),
    )


def _intelligent_context_and_authority():
    resolved = api.CONTAINER.configuration.resolve(
        bundle_id="baloot_ortools_research_native_v3_intelligent",
        environment="research-native-v3-intelligent",
    )
    return api.RESEARCH.execution_context_factory.freeze_selector(
        bundle_revision_id=resolved["bundle"]["revisionId"], analysis_mode="cascaded",
    )


def _optional_project_context_and_authority(tmp_path: Path):
    """Load the retired optional-project test snapshot in an isolated store."""
    from baloot_platform.configuration.service import VersionedConfigurationService
    from baloot_platform.configuration.store import SQLiteConfigurationStore
    from baloot_platform.config import build_bundle
    from baloot_platform.research.execution_context import ResearchExecutionContextFactoryV1

    root = Path(__file__).resolve().parents[1]
    configuration = VersionedConfigurationService(
        SQLiteConfigurationStore(tmp_path / "configuration.sqlite3"), root=root,
        profile=yaml.safe_load((root / "configs/configuration/control-plane.yaml").read_text(encoding="utf-8")),
    )
    selected = configuration.bootstrap_packaged_bundle(
        actor_id="TEST-HISTORICAL-PROJECTS",
        bundle_path="configs/bundle_ortools_research_native_v3_optional_project_quality.yaml",
    )
    template = api.RESEARCH.execution_context_factory
    factory = ResearchExecutionContextFactoryV1(
        configuration_service=configuration,
        artifact_store=LocalContentAddressedArtifactStore(tmp_path / "artifacts"),
        research_store=SQLiteResearchStoreV1(tmp_path / "research.sqlite3"),
        root=root, plugin_capabilities=template.plugin_capabilities,
        shared_plugin_descriptors=template.shared_plugin_descriptors,
        active_bundle=build_bundle(root),
        active_exact_search_backend=template.active_exact_search_backend,
    )
    return factory.freeze_selector(
        bundle_revision_id=selected["bundle"]["revisionId"],
        analysis_mode="cascaded",
    )


def test_optional_new_authoring_is_refused_before_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, context = _optional_project_context_and_authority(tmp_path)
    service, executor = _isolated_service(tmp_path, authority, context)
    monkeypatch.setattr(api, "RESEARCH", service)
    payload = _exact_recipe_payload(context.bundle_revision_id, quality=False)
    recipe, _ = service._build_recipe_and_context(payload)
    validated_recipes = []

    def historical_validation(parsed, _context):
        validated_recipes.append(parsed.sha256)
        return {"valid": True, "recipe": parsed.to_dict(), "issues": []}

    monkeypatch.setattr(service, "_validate_recipe_v4_cached", historical_validation)
    monkeypatch.setattr(
        exact_state_module.OrToolsExactStateNormalizerV1, "normalize",
        lambda *_args, **_kwargs: pytest.fail("refused authoring must not normalize exact input"),
    )
    try:
        with TestClient(api.app) as client:
            validation = client.post("/api/v2/research-recipes:validate", json=payload)
            assert validation.status_code == 200
            validation = validation.json()
            assert validation["valid"] is False
            issue = validation["issues"][0]
            assert issue["fieldPaths"] == ["configuration.bundleRevisionId"]
            assert "mandatory project declarations" in issue["details"]["reason"]
            workflow = service._ortools_localization_projection_v2(context)["groups"]["workflow"]
            assert workflow[issue["messageArKey"]]
            assert workflow[issue["messageEnKey"]]
            ResearchRecipeValidationResponse.model_validate(validation)
            estimate = client.post("/api/v2/research-recipes:estimate", json=payload)
            assert estimate.status_code == 200
            estimate = estimate.json()
            assert estimate["valid"] is False
            assert estimate["launchAdmissible"] is False
            assert estimate["issues"] == validation["issues"]
            ResearchRecipeEstimateResponse.model_validate(estimate)
            created = client.post("/api/v2/research-recipes", json=payload,
                                  headers={"Idempotency-Key": "optional-authoring-refused"})
            assert created.status_code == 422
            assert "mandatory project declarations" in created.json()["detail"]
        assert validated_recipes == []
        assert service.store.get_recipe(recipe.code) is None
        assert service.validate_recipe(payload)["valid"] is True
        assert service.estimate(payload)["valid"] is True
        assert validated_recipes == [recipe.sha256, recipe.sha256]
    finally:
        executor.shutdown(wait=True)


def _optional_project_prefix() -> dict:
    return {
        "schemaVersion": 1,
        "deckOrder": [
            2, 3, 4, 0, 1, 8, 9, 10, 11, 12, 13, 14, 7, 15, 16, 17,
            18, 19, 20, 21, 31, 5, 6, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        ],
        "dealer": "east",
        "roundNumber": 1,
        "matchScore": [0, 0],
        "auctionActions": [{"actor": "north", "action": "sun"}],
        "escalationActions": [],
        "projectActions": [],
        "projectDeclarations": [],
        "projectAssertions": [],
        "playActions": [],
        "stateAssertions": {},
    }


def _legacy_context_and_authority():
    revision_id = api.CONTAINER.configuration.resolve()["bundle"]["revisionId"]
    authority, context = api.RESEARCH.execution_context_factory.freeze_selector(
        bundle_revision_id=str(revision_id),
        analysis_mode="cascaded",
    )
    return authority, replace(
        context,
        bundle=_ChallengeBundleProxy(context.bundle, _legacy_challenge()),
    )


def _playing_quality_payload() -> dict:
    return {
        "schemaVersion": 1,
        "mode": "budgeted",
        "decisions": {
            kind: {
                "utility": "awarded_point_margin",
                "budgetProfile": (
                    "bounded_v1" if kind == "escalation" else "late_game_exact_v1"
                ),
                "horizon": {
                    "unit": "tricks" if kind == "play" else "plies",
                    "value": 1,
                },
                "boundaryEvaluator": (
                    "auction_hand_margin_v1"
                    if kind == "bid"
                    else "current_raw_point_margin_v1"
                ),
            }
            for kind in ("bid", "escalation", "project", "play")
        },
    }


def _recipe(
    configuration: dict,
    policy: str,
    *,
    trick_index: int | None = None,
    solver_timeout_seconds: float = 1.0,
    result_count: int = 1,
    candidate_count: int = 1,
    decision_remaining_cards: int | None = None,
    decision_seat: str | None = None,
    decision_position: int | None = None,
    seeds_per_shard: int = 1,
    quality: bool = False,
    budget_profile: str = "late_game_exact_v1",
) -> ResearchRecipeV4:
    selector = {"trickIndex": trick_index} if trick_index is not None else {}
    if decision_remaining_cards is not None:
        selector["remainingCards"] = decision_remaining_cards
    if decision_seat is not None:
        selector["actingSeat"] = decision_seat
    if decision_position is not None:
        selector["position"] = decision_position
    recipe = ResearchRecipeV4.build(
        {
            "schemaVersion": 4,
            "titleAr": "بحث أورتولز مجمد للاختبار",
            **(
                {"playingQuality": _playing_quality_payload()}
                if quality or policy == "ortools_cp_sat_v3"
                else {}
            ),
            "configuration": configuration,
            "scenario": {
                "schemaVersion": 1,
                "origin": {"kind": "search"},
                "constraints": {
                    "op": "all",
                    "children": [
                        {
                            "predicateId": "PRED-ORTOOLS-DEAL",
                            "kind": "deal.card_owner",
                            "version": 1,
                            "hardness": "hard",
                            "arguments": {
                                "card": 0,
                                "seat": "north",
                                "handStage": "initial_five",
                            },
                        }
                    ],
                },
                "decisionPoint": {
                    "kind": "play",
                    "selector": selector,
                },
                "outcome": {
                    "predicateId": "PRED-ORTOOLS-REPORT",
                    "kind": "outcome.team_round",
                    "version": 1,
                    "hardness": "report",
                    "arguments": {"team": "north_south", "status": "won"},
                },
            },
            "proof": {
                "schemaVersion": 1,
                "mode": "exact",
                "visibility": "complete_state",
                "claim": "outcome_reachable",
                "targetTeam": "north_south",
                "utility": "awarded_point_margin",
                "successThreshold": -100000,
                "budgetProfile": budget_profile,
            },
            "resources": {
                "generationLane": "ortools",
                "generationPolicy": "ortools_cp_sat_v3",
                "seedStart": 1,
                "seedCount": candidate_count,
                "seedsPerShard": seeds_per_shard,
                "solverTimeoutSeconds": solver_timeout_seconds,
                "proofBudgetProfile": budget_profile,
            },
            "output": {
                "kind": "laffa" if result_count == 1 else "laffa_collection",
                "resultCount": result_count,
            },
            **(
                {
                    "diversity": {
                        "schemaVersion": 1,
                        "cellDimensions": [
                            "contract",
                            "project",
                            "decision_kind",
                            "proof_mode",
                        ],
                        "topKPerCell": result_count,
                        "rejectedAuditModulus": 100,
                    }
                }
                if result_count > 1
                else {}
            ),
            "status": "published",
        }
    )
    if policy == "ortools_cp_sat_v3":
        return recipe
    historical = recipe.to_dict()
    historical["resources"]["generationPolicy"] = policy
    if policy != "ortools_cp_sat_v2":
        historical["resources"].update(
            {
                "candidatesPerSeed": 1,
                "maximumConflicts": 1,
                "analysisMode": "cascaded",
            }
        )
    historical["sha256"] = canonical_sha256(
        {key: value for key, value in historical.items() if key != "sha256"}
    )
    return ResearchRecipeV4.build(historical)


def _isolated_service(tmp_path: Path, authority, context):
    base = api.RESEARCH
    executor = ThreadPoolExecutor(max_workers=1)
    service = ResearchFoundryServiceV1(
        domain=base.domain,
        compiled_domain=base.compiled_domain,
        store=SQLiteResearchStoreV1(tmp_path / "research.sqlite3"),
        artifact_store=LocalContentAddressedArtifactStore(tmp_path / "artifacts"),
        executor=executor,
        deal_generator=base.deal_generator,
        auction_compiler=base.auction_compiler,
        escalation_compiler=base.escalation_compiler,
        simulator=base.simulator,
        analyzers=base.analyzers,
        score_profile=base.score_profile,
        decision_score_profile=base.decision_score_profile,
        project_compiler=base.project_compiler,
        play_compiler=base.play_compiler,
        tactical_verifier=base.tactical_verifier,
        maximum_tactical_decisions=base.maximum_tactical_decisions,
        comparison_profile=base.comparison_profile,
        maximum_local_shard_workers=1,
        execution_context_factory=_FrozenContextFactory(authority, context),
    )
    return service, executor


@pytest.mark.parametrize("auction_only,recorded_decline,require_qualified", [
    (False, False, False), (True, False, False), (True, True, False),
    (False, False, True)])
def test_v3_api_roundtrips_exact_quality_recipe_without_search(tmp_path, monkeypatch, auction_only, recorded_decline, require_qualified, record_property):
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    authority, frozen_context = _frozen_context_and_authority()
    context = replace(frozen_context, bundle=_ChallengeBundleProxy(
        frozen_context.bundle, _policy_service_challenge(),
    ))
    service, executor = _isolated_service(tmp_path, authority, context)
    monkeypatch.setattr(api, "RESEARCH", service)
    payload = _exact_recipe_payload(context.bundle_revision_id)
    if require_qualified:
        payload["resources"]["solverTimeoutSeconds"] = 180.0
        payload["resources"]["proofBudgetProfile"] = "offline_heavyweight_v1"
    payload["scenario"]["origin"]["recordedEscalationOpportunities"] = []
    prefix = _optional_project_prefix()
    if not auction_only:
        normalizer = exact_state_module.OrToolsExactStateNormalizerV1(context.bundle.rules, native_cp_sat=True)
        source = ExactTracePrefixV1.build(prefix)
        project = normalizer.normalize(source,
            decision_point={"kind": "project", "selector": {"owner": "north"}}, maximum_seconds=20)
        declare = next(action["code"] for action in project.legal_actions if action["code"].startswith("declare:"))
        prefix = normalizer.append_normalized_action(source, project, declare).to_dict()
    prefix["playActions"] = []
    payload["scenario"]["origin"]["exactTracePrefix"] = prefix
    payload["scenario"]["decisionPoint"] = {"kind": "play", "selector": {}}
    payload["scenario"]["outcome"]["arguments"] = {"seat": "north"}
    if auction_only:
        prefix["projectActions"] = []
        payload["scenario"]["decisionPoint"] = {"kind": "escalation", "selector": {"actingSeat": "west"}}
        payload["scenario"]["constraints"].update(kind="auction.contract", arguments={"contract": "sun"})
        payload["scenario"]["outcome"]["arguments"] = {"seat": "west"}
    if recorded_decline:
        prefix["auctionActions"] = [{"action": action} for action in ("hokum", "pass", "pass", "pass")]
        payload["scenario"]["origin"]["recordedEscalationOpportunities"] = [{"actor": "west", "actionCode": "decline"}]
        payload["scenario"]["decisionPoint"]["selector"]["actingSeat"] = "east"
        payload["scenario"]["constraints"]["arguments"] = {"contract": "hokum"}
        payload["scenario"]["outcome"]["arguments"] = {"seat": "east"}
    payload["proof"] = {"schemaVersion": 1, "mode": "bounded", "visibility": "complete_state",
        "claim": "outcome_reachable", "targetTeam": "north_south", "utility": "round_win",
        "successThreshold": 0, "budgetProfile": "late_game_exact_v1",
        "horizon": {"unit": "plies", "value": 1}, "boundaryEvaluator": "current_raw_point_margin_v1"}
    if require_qualified:
        payload["proof"]["budgetProfile"] = "offline_heavyweight_v1"
    payload["playingQuality"] = _playing_quality_payload()
    def forbidden(*args, **kwargs):
        pytest.fail("recipe validation/storage must not launch research")
    real_campaign = OrToolsUnifiedResearchExecutorV1.execute_campaign
    monkeypatch.setattr(OrToolsUnifiedResearchExecutorV1, "execute_campaign", forbidden)
    try:
        with TestClient(api.app) as client:
            if recorded_decline:
                preview = client.post("/api/v2/research-exact-traces:preview-legal-actions", json={
                    "schemaVersion": 1, "configuration": payload["configuration"],
                    "exactTracePrefix": prefix,
                    "recordedEscalationOpportunities": payload["scenario"]["origin"]["recordedEscalationOpportunities"]})
                assert preview.status_code == 200, preview.text
                assert preview.json()["valid"] is True, preview.json()
                assert preview.json()["nextActor"] == "east"
                assert preview.json()["legalActions"]
                assert all(row["actor"] == "east" for row in preview.json()["legalActions"])
                conflicting = copy.deepcopy(payload)
                conflicting["scenario"]["decisionPoint"]["selector"]["actingSeat"] = "west"
                rejected = client.post("/api/v2/research-recipes:validate", json=conflicting)
                assert rejected.status_code == 200, rejected.text
                assert rejected.json()["valid"] is False
            response = client.post("/api/v2/research-recipes:validate", json=payload)
            assert response.status_code == 200, response.text
            assert response.json()["valid"], response.json()["issues"]
            estimated = client.post("/api/v2/research-recipes:estimate", json=payload)
            assert estimated.status_code == 200, estimated.text
            assert estimated.json()["launchAdmissible"] is True, estimated.json()
            assert estimated.json()["proof"]["remainingCardCount"] == 32
            assert estimated.json()["proof"]["requestedDepth"] == 1
            exact_payload = copy.deepcopy(payload)
            exact_payload["proof"]["mode"] = "exact"
            exact_payload["proof"].pop("horizon")
            exact_payload["proof"].pop("boundaryEvaluator")
            exact_estimated = client.post("/api/v2/research-recipes:estimate", json=exact_payload)
            assert exact_estimated.status_code == 200, exact_estimated.text
            assert exact_estimated.json()["launchAdmissible"] is False
            assert exact_estimated.json()["proof"]["requestedDepth"] == (33 if auction_only else 32)
            assert exact_estimated.json()["proof"]["admissible"] is False
            created = client.post("/api/v2/research-recipes", json=payload,
                headers={"Idempotency-Key": "exact-quality-roundtrip"})
            assert created.status_code == 201, created.text
            reopened = client.get(f"/api/v2/research-recipes/{created.json()['code']}")
            assert reopened.status_code == 200, reopened.text
            assert reopened.json()["scenario"]["origin"] == created.json()["scenario"]["origin"]
            assert reopened.json()["scenario"]["origin"]["recordedEscalationOpportunities"] == payload["scenario"]["origin"]["recordedEscalationOpportunities"]
            assert reopened.json()["playingQuality"] == created.json()["playingQuality"]
            monkeypatch.setattr(OrToolsUnifiedResearchExecutorV1, "execute_campaign", real_campaign)
            launched = client.post("/api/v2/research-runs", json={"recipeCode": created.json()["code"]},
                headers={"Idempotency-Key": "exact-quality-run"})
            assert launched.status_code == 202, launched.text
            executor.shutdown(wait=True)
            completed = client.get(f"/api/v2/research-runs/{launched.json()['runCode']}")
            assert completed.status_code == 200, completed.text
            run = completed.json()
            assert run["status"] == "completed", run.get("events")
            assert run["campaignAccounting"]["stopReason"] != "worker_failed", [
                (event.get("type"), event.get("message")) for event in run.get("events", [])
            ]
            for event in run.get("events", []):
                if event.get("type") != "progress":
                    continue
                for diagnostic in (
                    event.get("incompleteCompilation"),
                    event.get("candidateReceipt", {}).get("incompleteCompilation"),
                    event.get("accounting", {}).get("latestIncompleteCompilation"),
                ):
                    if diagnostic is not None:
                        assert set(diagnostic) <= {
                            "stage", "sourceStage", "status", "exhaustionReason",
                            "completionStatus", "sha256", "candidateHash",
                            "targetPlayIndex", "targetActor",
                            "generatedActorAtTarget",
                            "firstConstructedPlayDivergenceIndex",
                        }
            codes = [*run["resultCodes"], *run["auditResultCodes"]]
            record_property("campaignOutcome", json.dumps({
                "qualifiedCount": len(run["resultCodes"]), "auditCount": len(run["auditResultCodes"]),
                "accounting": {key: run["campaignAccounting"].get(key) for key in (
                    "stopReason", "candidateAttempts", "candidateReceiptCounts",
                    "maximumCampaignSeconds", "maximumSolveSeconds", "proofIncomplete")},
                "incomplete": [{"reason": row.get("candidateAttemptReason"),
                    "stage": row.get("incompleteCompilation", {}).get("stage"),
                    "receiptHash": row.get("incompleteCompilation", {}).get("sha256")}
                    for row in run["campaignAccounting"].get("candidateReceipts", [])],
            }, sort_keys=True))
            assert codes, {"accounting": run["campaignAccounting"], "events": run.get("events")}
            if require_qualified:
                assert run["resultCodes"], "full-round qualified result missing; see campaignOutcome receipt"
            for code in codes:
                loaded = client.get(f"/api/v2/research-results/{code}")
                assert loaded.status_code == 200, loaded.text
                result = loaded.json()
                record_property(f"qualification-{code}", json.dumps(result["qualification"], sort_keys=True))
                if code in run["auditResultCodes"]:
                    assert result["qualification"]["qualified"] is False
    finally:
        executor.shutdown(wait=True)


def _exact_recipe_payload(
    bundle_revision_id: str,
    *,
    quality: bool = True,
) -> dict:
    root = Path(__file__).resolve().parents[1]
    fixture = json.loads(
        (root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    exact_trace_prefix = json.loads(
        (root / "examples/research-scenario-bounded-quality-prefix.json").read_text(
            encoding="utf-8"
        )
    )
    exact_trace_prefix.pop("sha256", None)
    exact_trace_prefix["playActions"] = exact_trace_prefix["playActions"][:24]
    payload = {
        "schemaVersion": 4,
        "titleAr": "حالة أورتولز كاملة من التخزين المؤقت",
        "configuration": {
            "schemaVersion": 1,
            "bundleRevisionId": bundle_revision_id,
        },
        "scenario": {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": exact_trace_prefix,
                "recordedEscalationOpportunities": [],
            },
            "constraints": {
                "predicateId": "PRED-E2E-PHASE",
                "kind": "state.phase",
                "version": 1,
                "hardness": "hard",
                "arguments": {"state": "playing"},
            },
            "decisionPoint": {
                "kind": "play",
                "selector": {"trickIndex": 6, "actingSeat": "south"},
            },
            "outcome": {
                "predicateId": "PRED-E2E-ACTOR",
                "kind": "state.next_actor",
                "version": 1,
                "hardness": "hard",
                "arguments": {"seat": "south"},
            },
        },
        "proof": {
            **fixture["proof"],
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -100000,
            "selectedAction": "S10",
        },
        "resources": {
            "generationLane": "ortools",
            "generationPolicy": "ortools_cp_sat_v3",
            "seedStart": 1,
            "seedCount": 1,
            "seedsPerShard": 1,
            "solverTimeoutSeconds": 30.0,
            "proofBudgetProfile": "late_game_exact_v1",
        },
        "status": "published",
    }
    if quality:
        payload["playingQuality"] = _playing_quality_payload()
    return payload


@pytest.mark.parametrize("tactic", ["large_card_sacrifice", "small_card_big_gain"])
@pytest.mark.parametrize("section", ["constraints", "outcome"])
def test_causal_api_admission_refuses_before_normalization_or_search(tmp_path, monkeypatch, tactic, section):
    authority, context = _intelligent_context_and_authority()
    service, worker = _isolated_service(tmp_path, authority, context)
    monkeypatch.setattr(api, "RESEARCH", service)
    payload = _exact_recipe_payload(context.bundle_revision_id)
    payload["scenario"][section] = {"op": "not", "children": [{
        "predicateId": "PRED-CAUSAL", "kind": "tactic.verified", "version": 1,
        "hardness": "hard", "arguments": {
            "kind": tactic, "causalRequired": True, "minimumScore": 0, "minimumEvidence": 0,
        },
    }]}

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported causality must fail before normalization or search")

    monkeypatch.setattr(exact_state_module.OrToolsExactStateNormalizerV1, "normalize", forbidden)
    monkeypatch.setattr(OrToolsUnifiedResearchExecutorV1, "execute_campaign", forbidden)
    try:
        with TestClient(api.app) as client:
            for endpoint in ("validate", "estimate"):
                response = client.post(f"/api/v2/research-recipes:{endpoint}", json=payload)
                assert response.status_code == 200, response.text
                body = response.json()
                assert body["valid"] is False
                assert [issue["code"] for issue in body["issues"]] == ["tactic_causal_verification_unavailable"]
                assert body["issues"][0]["predicateIds"] == ["PRED-CAUSAL"]
                assert body["issues"][0]["fieldPaths"] == [
                    f"scenario.{section}.children[0].arguments.causalRequired"]
                assert "normalizedExactState" not in body
                if endpoint == "estimate":
                    assert body["launchAdmissible"] is False
    finally:
        worker.shutdown(wait=True)


def _native_qualified_exact_recipe_payload(bundle_revision_id: str, *, prefix_length: int = 24) -> dict:
    payload = _exact_recipe_payload(bundle_revision_id)
    fixture = json.loads((Path(__file__).parent / "fixtures/research-native-qualified-round-v3.json").read_text(encoding="utf-8"))
    prefix = fixture["exactTracePrefix"]
    decision = prefix["playActions"][prefix_length]
    prefix["playActions"] = prefix["playActions"][:prefix_length]
    payload["scenario"]["origin"]["exactTracePrefix"] = prefix
    payload["scenario"]["decisionPoint"]["selector"]["actingSeat"] = decision["actor"]
    payload["scenario"]["outcome"]["arguments"]["seat"] = decision["actor"]
    payload["proof"]["selectedAction"] = Card.parse(decision["card"]).code
    return payload


def test_exact_quality_routes_featured_play_to_native_proof(monkeypatch: pytest.MonkeyPatch) -> None:
    from baloot_platform.research.scenario_contracts import (
        ExactTracePrefixV1,
        ScenarioPlayingQualityPolicyV1,
        ScenarioQualityExecutionBudgetV1,
    )

    root = Path(__file__).resolve().parents[1]
    challenge = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
    section = challenge["scenarioResearchV4"]
    prefix = ExactTracePrefixV1.build(json.loads(
        (root / "examples/research-scenario-v4-intelligent-exact-prefix.json").read_text(encoding="utf-8")
    ))
    executor = OrToolsUnifiedResearchExecutorV1(
        rules=RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml"),
        bundle_revision_id="CBR-TEST", bundle_hash="a" * 64,
        backend_id="ortools_cp_sat_v3",
        quality_policy=ScenarioPlayingQualityPolicyV1.build(_playing_quality_payload()),
        quality_execution=ScenarioQualityExecutionBudgetV1.build(section["qualityExecution"]),
        quality_proof_profiles=section["proofBudgets"]["profiles"],
    )
    candidate = SimpleNamespace(
        trace={"deckOrder": list(prefix.deck_order), "dealer": prefix.dealer},
        sha256="b" * 64, origin="exact_trace_prefix",
    )

    expected = {"index": len(prefix.play_actions), "selector": {"actingSeat": "south"}}

    def capture(*_args, **kwargs):
        assert kwargs["featured_play_index"] == expected["index"]
        assert kwargs["featured_decision_point"] == {"kind": "play", "selector": expected["selector"]}
        assert kwargs["featured_selector_dynamic"] is False
        assert kwargs["search_generated"] is False
        raise RuntimeError("featured native dispatch checked")

    monkeypatch.setattr(executor_module, "resolve_quality_round_continuation_v1", capture)
    with pytest.raises(RuntimeError, match="featured native dispatch checked"):
        executor.audit_quality_candidate(
            candidate, recorded_trace=prefix, recorded_events=(), maximum_seconds=30,
            decision_point={"kind": "play", "selector": {"actingSeat": "south"}},
            continue_recorded_prefix=True,
        )
    expected.update(index=23, selector={"trickIndex": 5, "position": 3, "actingSeat": "west"})
    with pytest.raises(RuntimeError, match="featured native dispatch checked"):
        executor.audit_quality_candidate(
            candidate, recorded_trace=prefix, recorded_events=(), maximum_seconds=30,
            decision_point={"kind": "play", "selector": expected["selector"]},
            continue_recorded_prefix=True,
        )


@pytest.mark.parametrize("visibility", ("complete_state", "actor_information_set"))
@pytest.mark.parametrize("enabled", (False, True))
def test_exact_origin_quality_preflight_preserves_full_round_limits(visibility, enabled):
    authority, context = _frozen_context_and_authority()
    challenge = copy.deepcopy(context.bundle.component("challenge"))
    if enabled:
        challenge["scenarioResearchV4"]["qualityExecution"]["allowExactInputQuality"] = True
    else:
        challenge["scenarioResearchV4"]["qualityExecution"].pop("allowExactInputQuality", None)
    context = replace(context, bundle=_ChallengeBundleProxy(context.bundle, challenge))
    raw = _exact_recipe_payload(context.bundle_revision_id)
    raw["configuration"] = authority.to_dict()
    raw["scenario"]["origin"]["recordedEscalationOpportunities"] = []
    raw["playingQuality"] = {
        "schemaVersion": 2, "mode": "exact", "visibility": visibility,
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "offline_heavyweight_v1"}
                      for kind in ("bid", "escalation", "project", "play")},
    }
    recipe = ResearchRecipeV4.build(raw)
    validation = api.RESEARCH._validate_recipe_v4(recipe, context)
    if not enabled:
        assert validation["valid"] is False
        assert any(issue["code"] == "all_decision_quality_evidence_unavailable" for issue in validation["issues"])
        return
    assert validation["valid"], validation["issues"]
    estimate = api.RESEARCH._estimate_recipe_v4(recipe, validation, context)
    assert estimate["launchAdmissible"] is True, estimate
    if visibility == "actor_information_set":
        assert "playing_quality_full_round_cost_not_estimated" in estimate["warnings"]
        # A manual prefix before purchasing is not the supported recorded-bid
        # audit path; the substantive early-decision capability gate stays shut.
        unfinished = copy.deepcopy(raw)
        unfinished["scenario"]["decisionPoint"] = {"kind": "bid", "selector": {}}
        prefix = unfinished["scenario"]["origin"]["exactTracePrefix"]
        for field in ("auctionActions", "escalationActions", "projectActions", "playActions"):
            prefix[field] = []
        prefix["stateAssertions"] = {}
        prefix["projectAssertions"] = []
        backend = api.RESEARCH._ortools_scenario_backend_v2(context)
        refused = backend.estimate(unfinished)
        assert refused["launchAdmissible"] is False
        assert refused["refusalReason"] == "phase4_substantive_full_service_evidence_missing"
    else:
        assert estimate["estimatedHiddenWorlds"] == 1
    for kind in raw["playingQuality"]["decisions"]:
        shallow = copy.deepcopy(raw)
        shallow["playingQuality"]["decisions"][kind]["budgetProfile"] = "late_game_exact_v1"
        candidate = ResearchRecipeV4.build(shallow)
        checked = api.RESEARCH._validate_recipe_v4(candidate, context)
        refused = api.RESEARCH._estimate_recipe_v4(candidate, checked, context)
        assert refused["launchAdmissible"] is False
        assert refused["refusalReason"] == "playing_quality_full_round_depth_exceeds_budget"


def test_exact_quality_preflight_accepts_complete_state_bid_policy_pair():
    from baloot_platform.research.ortools_play_tail import NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS

    authority, context = _frozen_context_and_authority()
    challenge = copy.deepcopy(context.bundle.component("challenge"))
    challenge["scenarioResearchV4"]["qualityExecution"].update(
        allowExactInputQuality=True,
        bidOpponentPlayPolicyId=sorted(NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS)[0],
    )
    context = replace(context, bundle=_ChallengeBundleProxy(context.bundle, challenge))
    raw = _exact_recipe_payload(context.bundle_revision_id)
    raw["configuration"] = authority.to_dict()
    raw["scenario"]["origin"]["recordedEscalationOpportunities"] = []
    raw["playingQuality"] = {
        "schemaVersion": 2, "mode": "exact", "visibility": "complete_state",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "offline_heavyweight_v1"}
                      for kind in ("bid", "escalation", "project", "play")},
    }
    recipe = ResearchRecipeV4.build(raw)
    validation = api.RESEARCH._validate_recipe_v4(recipe, context)
    assert validation["valid"], validation["issues"]
    for issue in validation["issues"]:
        api.ScenarioDiagnosticV1Response.model_validate(issue)
    estimate = api.RESEARCH._estimate_recipe_v4(recipe, validation, context)
    assert estimate["launchAdmissible"], estimate


@pytest.mark.real_clock
@pytest.mark.parametrize("source_stage", [None, "exact_play_prefix", {"private": "not-public"}])
def test_exact_validation_timeout_is_incomplete_not_invalid(monkeypatch, source_stage):
    authority, context = _frozen_context_and_authority()
    raw = _exact_recipe_payload(context.bundle_revision_id)
    raw["configuration"] = authority.to_dict()
    recipe = ResearchRecipeV4.build(raw)
    evidence = {"futurePlays": ["not-public"], "terminalScore": [99, 0]}
    if source_stage is not None:
        evidence["sourceStage"] = source_stage
    stopped = exact_state_module.ExactNormalizationStopped("maximum_seconds", evidence=evidence)

    def exhaust(*args, **kwargs):
        raise stopped

    monkeypatch.setattr(exact_state_module.OrToolsExactStateNormalizerV1, "normalize", exhaust)
    validation = api.RESEARCH._validate_recipe_v4(recipe, context)

    assert validation["valid"] is False
    issue = next(row for row in validation["issues"] if row["code"] == "exact_normalization_incomplete")
    assert issue["class"] == "normalization_incomplete"
    assert issue["details"]["exhaustionReason"] == "maximum_seconds"
    assert issue["details"]["partialReceiptHash"] == stopped.partial_receipt["sha256"]
    expected_fields = {"backendIdentityHash", "exhaustionReason", "stage", "partialReceiptHash"}
    if isinstance(source_stage, str):
        expected_fields.add("sourceStage")
        assert issue["details"]["sourceStage"] == source_stage
    assert set(issue["details"]) == expected_fields
    assert not any(row["code"] == "exact_trace_prefix_invalid" for row in validation["issues"])
    assert api.ScenarioDiagnosticV1Response.model_validate(issue)


def test_exact_playing_quality_accepts_native_observed_exact_budget() -> None:
    from baloot_platform.research.scenario_contracts import (
        ScenarioPlayingQualityPolicyV1,
        ScenarioQualityExecutionBudgetV1,
    )

    root = Path(__file__).resolve().parents[1]
    challenge = yaml.safe_load(
        (root / "configs/challenge/intelligent_research.yaml").read_text(
            encoding="utf-8"
        )
    )
    section = challenge["scenarioResearchV4"]
    policy = {
        "schemaVersion": 1,
        "mode": "exact",
        "decisions": {
            kind: {
                "utility": "awarded_point_margin",
                "budgetProfile": "actor_information_set_exact_v1",
            }
            for kind in ("bid", "escalation", "project", "play")
        },
    }

    executor = OrToolsUnifiedResearchExecutorV1(
        rules=RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml"),
        bundle_revision_id="CBR-TEST",
        bundle_hash="a" * 64,
        backend_id="ortools_cp_sat_v3",
        quality_policy=ScenarioPlayingQualityPolicyV1.build(policy),
        quality_execution=ScenarioQualityExecutionBudgetV1.build(
            section["qualityExecution"]
        ),
        quality_proof_profiles=section["proofBudgets"]["profiles"],
    )

    assert executor.quality_policy.mode == "exact"


@pytest.mark.real_clock
def test_ortools_service_recipe_and_validation_caches_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    challenge = _ortools_challenge()
    challenge["scenarioResearchV4"]["cacheLimits"].update(
        {"maximumEntries": 1, "maximumBytes": 1_000_000}
    )
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, challenge),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        first = _recipe(authority.to_dict(), "ortools_cp_sat_v3").to_dict()
        first.pop("sha256")
        second = copy.deepcopy(first)
        second["titleAr"] += " ثانية"

        service._build_recipe_and_context(first)
        service._build_recipe_and_context(second)
        assert len(service._scenario_recipe_context_cache_v4) == 1
        assert service._scenario_recipe_context_cache_v4_bytes <= 1_000_000

        service.validate_recipe(first)
        returned = service.validate_recipe(second)
        returned["issues"].append({"not": "cached"})
        assert not any(
            "not" in item for item in service.validate_recipe(second)["issues"]
        )
        assert len(service._scenario_validation_cache_v4) == 1
        assert service._scenario_validation_cache_v4_bytes <= 1_000_000

        third = copy.deepcopy(first)
        third["titleAr"] += " ثالثة"
        original = service._validate_recipe_v4_ortools
        attempts = []

        def timeout_once(*args):
            attempts.append(1)
            if len(attempts) == 1:
                return {"valid": False, "issues": [{
                    "class": "normalization_incomplete", "code": "exact_normalization_incomplete",
                }]}
            return original(*args)

        monkeypatch.setattr(service, "_validate_recipe_v4_ortools", timeout_once)
        assert service.validate_recipe(third)["issues"][0]["code"] == "exact_normalization_incomplete"
        retry = service.validate_recipe(third)
        assert not any(item.get("class") == "normalization_incomplete" for item in retry["issues"])
        assert service.validate_recipe(third) == retry
        assert len(attempts) == 2
        assert len(service._scenario_validation_cache_v4) == 1
        assert service._scenario_validation_cache_v4_bytes <= 1_000_000
    finally:
        executor.shutdown(wait=True)


def _fixed_policy_templates() -> dict:
    policies = {}
    for seed, seat in enumerate(("north", "west", "south", "east"), start=1):
        core = {
            "schemaVersion": 1,
            "seat": seat,
            "policyId": "fixed_action_map_v1",
            "version": "1.0.0",
            "seed": seed,
            "choices": [],
            "materialization": "seeded_legal_action_order_v1",
        }
        policies[seat] = {key: value for key, value in core.items() if key != "seat"}
        policies[seat]["profileHash"] = canonical_sha256(core)
    return policies


def _policy_conditioned_recipe_payload(
    bundle_revision_id: str,
    *,
    claim: str = "winning_continuation_line_count",
    success_threshold: int = 1,
) -> dict:
    payload = _native_qualified_exact_recipe_payload(bundle_revision_id)
    proof = {
        "schemaVersion": 1,
        "mode": "policy_conditioned",
        "visibility": "complete_state",
        "strategyClass": "frozen_policy",
        "claim": claim,
        "targetTeam": "north_south",
        "utility": "awarded_point_margin",
        "successThreshold": success_threshold,
        "selectedAction": payload["proof"]["selectedAction"],
        "policies": _fixed_policy_templates(),
        "budgetProfile": "late_game_exact_v1",
    }
    if claim == "winning_continuation_line_count":
        proof["lineUniverse"] = "policy_conditioned"
    payload["proof"] = proof
    return payload


def _policy_service_challenge(*, measured_wall_seconds: float | None = None) -> dict:
    challenge = _ortools_challenge()
    if measured_wall_seconds is not None:
        return _apply_measured_nonplay_limits(challenge, measured_wall_seconds)
    # The 1s interactive tier is not reliable under the shared 5% host cap.
    challenge["scenarioResearchV4"]["searchBudgets"]["profiles"]["interactive_validation_v1"]["maximumWallSeconds"] = 5.0
    # At the 5% host cap, all 39 decisions finish but readback exhausts 60s.
    challenge["scenarioResearchV4"]["searchBudgets"]["profiles"]["late_game_exact_v1"]["maximumWallSeconds"] = 120.0
    return challenge


def _nonplay_policy_recipe_payload(
    bundle_revision_id: str,
    decision_kind: str,
) -> dict:
    payload = _exact_recipe_payload(bundle_revision_id)
    prefix = payload["scenario"]["origin"]["exactTracePrefix"]
    prefix.update(
        {
            "playActions": [],
            "projectActions": [],
            "projectDeclarations": [],
            "projectAssertions": [],
            "stateAssertions": {},
        }
    )
    if decision_kind == "bid":
        prefix["auctionActions"] = []
        prefix["escalationActions"] = []
        payload["scenario"].update(
            {
                "constraints": {
                    "op": "all",
                    "children": [
                        {
                            "predicateId": "PRED-NONPLAY-BID-PHASE",
                            "kind": "state.phase", "version": 1,
                            "hardness": "hard", "arguments": {"state": "bidding"},
                        },
                        {
                            "predicateId": "PRED-NONPLAY-BID-LEGAL-COUNT",
                            "kind": "state.legal_action_count", "version": 1,
                            "hardness": "hard",
                            "arguments": {"minimum": 1, "maximum": 8},
                        },
                        {
                            "predicateId": "PRED-NONPLAY-BID-NEXT-ACTOR",
                            "kind": "state.next_actor", "version": 1,
                            "hardness": "hard", "arguments": {"seat": "north"},
                        },
                        {
                            "predicateId": "PRED-NONPLAY-BID-MATCH",
                            "kind": "state.match_score", "version": 1,
                            "hardness": "hard",
                            "arguments": {"team": "north_south", "minimum": 0, "maximum": 0},
                        },
                    ],
                },
                "decisionPoint": {
                    "kind": "bid",
                    "selector": {"actionIndex": 0},
                },
                "outcome": {
                    "predicateId": "PRED-NONPLAY-BID-OUTCOME",
                    "kind": "outcome.round_margin",
                    "version": 1,
                    "hardness": "hard",
                    "arguments": {
                        "team": "north_south",
                        "minimum": -100000,
                        "maximum": 100000,
                    },
                },
            }
        )
    elif decision_kind == "escalation":
        prefix["matchScore"] = [101, 0]
        payload["scenario"].update(
            {
                "constraints": {
                    "op": "all",
                    "children": [
                        {
                            "predicateId": "PRED-NONPLAY-LEVEL-A",
                            "kind": "escalation.level", "version": 1,
                            "hardness": "hard", "arguments": {"level": "normal"},
                        },
                        {
                            "predicateId": "PRED-ESCALATION-PHASE",
                            "kind": "state.phase", "version": 1,
                            "hardness": "hard", "arguments": {"state": "final_dealt"},
                        },
                        {
                            "predicateId": "PRED-ESCALATION-ACTOR",
                            "kind": "state.next_actor", "version": 1,
                            "hardness": "hard", "arguments": {"seat": "west"},
                        },
                        {
                            "predicateId": "PRED-ESCALATION-LEGAL-COUNT",
                            "kind": "state.legal_action_count", "version": 1,
                            "hardness": "hard",
                            "arguments": {"minimum": 1, "maximum": 64},
                        },
                        {
                            "predicateId": "PRED-ESCALATION-MATCH",
                            "kind": "state.match_score", "version": 1,
                            "hardness": "hard",
                            "arguments": {"team": "north_south", "minimum": 101, "maximum": 101},
                        },
                    ],
                },
                "decisionPoint": {
                    "kind": "escalation",
                    "selector": {"actingSeat": "west"},
                },
                "outcome": {
                    "predicateId": "PRED-NONPLAY-LEVEL-B",
                    "kind": "escalation.level",
                    "version": 1,
                    "hardness": "hard",
                    "arguments": {"level": "normal"},
                },
            }
        )
    elif decision_kind == "project":
        payload["scenario"]["decisionPoint"] = {
            "kind": "project",
            "selector": {"owner": "north", "projectIndex": 0},
        }
        payload["scenario"]["constraints"] = {
            "op": "all",
            "children": [
                {
                    "predicateId": "PRED-NONPLAY-PROJECT-SUN",
                    "kind": "auction.contract", "version": 1,
                    "hardness": "hard", "arguments": {"contract": "sun"},
                },
                {
                    "predicateId": "PRED-NONPLAY-PROJECT-PHASE",
                    "kind": "state.phase", "version": 1,
                    "hardness": "hard", "arguments": {"state": "playing"},
                },
                {
                    "predicateId": "PRED-NONPLAY-PROJECT-ACTOR",
                    "kind": "state.next_actor", "version": 1,
                    "hardness": "hard", "arguments": {"seat": "north"},
                },
                {
                    "predicateId": "PRED-NONPLAY-PROJECT-LEGAL-COUNT",
                    "kind": "state.legal_action_count", "version": 1,
                    "hardness": "hard",
                    "arguments": {"minimum": 1, "maximum": 64},
                },
                {
                    "predicateId": "PRED-NONPLAY-PROJECT-MATCH",
                    "kind": "state.match_score", "version": 1,
                    "hardness": "hard",
                    "arguments": {"team": "north_south", "minimum": 0, "maximum": 0},
                },
            ],
        }
        payload["scenario"]["outcome"] = {
            "predicateId": "PRED-NONPLAY-PROJECT-REPORT",
            "kind": "auction.contract", "version": 1, "hardness": "report",
            "arguments": {"contract": "sun"},
        }
    else:
        raise ValueError("non-play policy fixture supports bid, escalation, or project")
    payload["proof"] = {
        "schemaVersion": 1,
        "mode": "policy_conditioned",
        "visibility": "complete_state",
        "strategyClass": "frozen_policy",
        "claim": "outcome_reachable",
        "targetTeam": "north_south",
        "utility": "awarded_point_margin",
        "successThreshold": -100000,
        "lineUniverse": "policy_conditioned",
        "policies": _fixed_policy_templates(),
        "budgetProfile": "late_game_exact_v1",
    }
    if decision_kind == "bid":
        payload["proof"]["selectedAction"] = "sun"
    return payload


def _nonplay_complete_graph_recipe_payload(
    bundle_revision_id: str,
    decision_kind: str,
    *,
    mode: str,
) -> dict:
    payload = _nonplay_policy_recipe_payload(bundle_revision_id, decision_kind)
    proof = {
        "schemaVersion": 1,
        "mode": mode,
        "visibility": "complete_state",
        "strategyClass": "pure_deterministic",
        "claim": "outcome_reachable",
        "targetTeam": "north_south",
        "utility": "awarded_point_margin",
        "successThreshold": -100000,
        "budgetProfile": "late_game_exact_v1",
    }
    if mode == "bounded":
        proof.update(
            {
                "horizon": {"unit": "plies", "value": 2},
                "boundaryEvaluator": "current_raw_point_margin_v1",
            }
        )
    payload["proof"] = proof
    return payload


def _search_nonplay_policy_recipe_payload(
    bundle_revision_id: str,
    decision_kind: str,
) -> dict:
    payload = _nonplay_policy_recipe_payload(bundle_revision_id, decision_kind)
    payload["scenario"]["origin"] = {"kind": "search"}
    if decision_kind == "bid":
        payload["scenario"]["constraints"] = {
            "op": "all",
            "children": [
                {
                    "predicateId": "PRED-SEARCH-BID-PHASE",
                    "kind": "state.phase", "version": 1,
                    "hardness": "hard", "arguments": {"state": "bidding"},
                },
                {
                    "predicateId": "PRED-SEARCH-BID-LEGAL-COUNT",
                    "kind": "state.legal_action_count", "version": 1,
                    "hardness": "hard",
                    "arguments": {"minimum": 1, "maximum": 8},
                },
                {
                    "predicateId": "PRED-SEARCH-BID-MATCH",
                    "kind": "state.match_score", "version": 1,
                    "hardness": "hard",
                    "arguments": {"team": "north_south", "minimum": 0, "maximum": 152},
                },
            ],
        }
        payload["scenario"]["decisionPoint"]["selector"] = {"actionIndex": 0}
    elif decision_kind == "escalation":
        payload["scenario"]["decisionPoint"]["selector"]["actionIndex"] = 0
        payload["scenario"]["constraints"] = {
            "op": "all",
            "children": [
                {
                    "predicateId": "PRED-SEARCH-NONPLAY-ACTIONS",
                    "kind": "escalation.level", "version": 1,
                    "hardness": "hard", "arguments": {"level": "normal"},
                },
                {
                    "predicateId": "PRED-SEARCH-ESCALATION-PHASE",
                    "kind": "state.phase", "version": 1,
                    "hardness": "hard", "arguments": {"state": "final_dealt"},
                },
                {
                    "predicateId": "PRED-SEARCH-ESCALATION-LEGAL-COUNT",
                    "kind": "state.legal_action_count", "version": 1,
                    "hardness": "hard",
                    "arguments": {"minimum": 1, "maximum": 64},
                },
                {
                    "predicateId": "PRED-SEARCH-ESCALATION-MATCH",
                    "kind": "state.match_score", "version": 1,
                    "hardness": "hard",
                    "arguments": {"team": "north_south", "minimum": 0, "maximum": 152},
                },
            ],
        }
    else:
        payload["scenario"]["constraints"] = {
            "op": "all",
            "children": [
                {
                    "predicateId": "PRED-SEARCH-NONPLAY-SUN",
                    "kind": "auction.contract", "version": 1,
                    "hardness": "hard", "arguments": {"contract": "sun"},
                },
                {
                    "predicateId": "PRED-SEARCH-NONPLAY-PHASE",
                    "kind": "state.phase", "version": 1,
                    "hardness": "hard", "arguments": {"state": "playing"},
                },
                {
                    "predicateId": "PRED-SEARCH-NONPLAY-ACTOR",
                    "kind": "state.next_actor", "version": 1,
                    "hardness": "hard", "arguments": {"seat": "north"},
                },
                {
                    "predicateId": "PRED-SEARCH-NONPLAY-LEGAL-COUNT",
                    "kind": "state.legal_action_count", "version": 1,
                    "hardness": "hard",
                    "arguments": {"minimum": 1, "maximum": 64},
                },
                {
                    "predicateId": "PRED-SEARCH-NONPLAY-MATCH",
                    "kind": "state.match_score", "version": 1,
                    "hardness": "hard",
                    "arguments": {"team": "north_south", "minimum": 0, "maximum": 152},
                },
            ],
        }
    payload["scenario"]["outcome"] = {
        "predicateId": "PRED-SEARCH-NONPLAY-OUTCOME",
        "kind": "outcome.round_margin",
        "version": 1,
        "hardness": "hard",
        "arguments": {
            "team": "north_south",
            "minimum": -100000,
            "maximum": 100000,
        },
    }
    payload["resources"]["solverTimeoutSeconds"] = 30.0
    return payload


def _actor_information_exact_recipe_payload(bundle_revision_id: str) -> dict:
    payload = _native_qualified_exact_recipe_payload(bundle_revision_id, prefix_length=27)
    payload["scenario"]["decisionPoint"]["selector"] = {
        "trickIndex": 6,
        "position": 3,
        "actingSeat": "east",
    }
    payload["scenario"]["outcome"]["arguments"] = {"seat": "east"}
    payload["proof"] = {
        "schemaVersion": 1,
        "mode": "exact",
        "visibility": "actor_information_set",
        "strategyClass": "pure_deterministic",
        "claim": "winning_root_action_count",
        "targetTeam": "west_east",
        "utility": "awarded_point_margin",
        "successThreshold": -100000,
        "budgetProfile": "actor_information_set_exact_v1",
    }
    payload["resources"]["proofBudgetProfile"] = "actor_information_set_exact_v1"
    return payload


def _nonplay_actor_information_recipe_payload(
    bundle_revision_id: str,
    decision_kind: str,
) -> dict:
    payload = _nonplay_policy_recipe_payload(bundle_revision_id, decision_kind)
    payload["proof"] = {
        "schemaVersion": 1,
        "mode": "exact",
        "visibility": "actor_information_set",
        "strategyClass": "pure_deterministic",
        "claim": "winning_root_action_count",
        "targetTeam": "north_south",
        "utility": "awarded_point_margin",
        "successThreshold": -100000,
        "budgetProfile": "actor_information_set_exact_v1",
    }
    payload["resources"]["proofBudgetProfile"] = "actor_information_set_exact_v1"
    return payload


def _nonplay_exact_line_count_recipe_payload(
    bundle_revision_id: str,
    decision_kind: str,
) -> dict:
    payload = _nonplay_policy_recipe_payload(bundle_revision_id, decision_kind)
    payload["proof"] = {
        "schemaVersion": 1,
        "mode": "exact",
        "visibility": "complete_state",
        "strategyClass": "pure_deterministic",
        "claim": "winning_continuation_line_count",
        "targetTeam": "north_south",
        "utility": "awarded_point_margin",
        "successThreshold": -100000,
        "lineUniverse": "all_legal",
        "budgetProfile": "late_game_exact_v1",
    }
    return payload


def _terminal_search_execution(
    status: str,
    manifest_hash: str,
    *,
    rules: RuleBook,
    bundle_revision_id: str,
    bundle_hash: str,
    scenario_hash: str,
    candidate_attempt_reason: str | None = None,
) -> OrToolsUnifiedResearchResultV1:
    preflight = OrToolsResearchPreflightV1(
        "search",
        "playPrefix",
        ("deal", "playPrefix", "proof"),
        ("PRED-ORTOOLS-DEAL",),
        (),
        (),
        manifest_hash,
    )
    evaluation = None
    completed_candidates = 0
    if status == "semantic_postcondition_miss":
        evaluation = OrToolsPredicateEvaluationV1(
            "target_miss",
            False,
            (
                {
                    "predicateId": "PRED-ORTOOLS-DEAL",
                    "kind": "deal.card_owner",
                    "satisfied": False,
                },
            ),
            (),
            canonical_sha256({"model": "target_miss"}),
        )
        completed_candidates = 1
    executor = OrToolsUnifiedResearchExecutorV1(
        rules,
        bundle_revision_id=bundle_revision_id,
        bundle_hash=bundle_hash,
        backend_id="ortools_cp_sat_v3",
    )
    return executor._result(
        status,
        preflight,
        None,
        evaluation,
        None,
        (),
        time.perf_counter(),
        time.process_time(),
        completed_candidates,
        scenario_hash=scenario_hash,
        candidate_attempt_reason=candidate_attempt_reason,
    )


def test_service_selects_v2_only_from_the_frozen_challenge_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, legacy_context = _legacy_context_and_authority()
    ortools_context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_v2_challenge()),
    )
    factory = api.RESEARCH.execution_context_factory
    monkeypatch.setattr(
        factory,
        "resolve_bundle_revision",
        lambda _revision_id, *, require_published=False: (None, ortools_context),
    )

    capability_response = api.RESEARCH.capabilities(
        bundle_revision_id=ortools_context.bundle_revision_id
    )
    capabilities = capability_response["research"]["scenarioResearchV4"]

    assert capabilities["backendIdentity"]["backendId"] == (
        ORTOOLS_SCENARIO_RESEARCH_BACKEND
    )
    assert capabilities["executionReady"] is True
    assert capabilities["launchAdmissible"] is True
    assert capabilities["sha256"]
    assert capability_response["research"]["scenarioConfiguration"][
        "defaultPolicyProfiles"
    ] == (_fixed_policy_templates())
    assert all(
        "agentId" not in profile
        for profile in capability_response["research"]["scenarioConfiguration"][
            "defaultPolicyProfiles"
        ].values()
    )
    policy_profile = copy.deepcopy(_fixed_policy_templates()["west"])
    api.ScenarioDeclarativePolicyProfileV1Response.model_validate(policy_profile)
    policy_profile["seed"] = "2"
    with pytest.raises(ValidationError):
        api.ScenarioDeclarativePolicyProfileV1Response.model_validate(policy_profile)
    localization = capabilities["localization"]
    localization_core = {
        key: copy.deepcopy(value)
        for key, value in localization.items()
        if key != "sha256"
    }
    assert localization["bundleRevisionId"] == ortools_context.bundle_revision_id
    assert localization["productProfileHash"] == (
        ortools_context.bundle.components["product"].sha256
    )
    assert localization["locale"] == "ar-SA"
    assert localization["direction"] == "rtl"
    assert localization["groups"] == {
        "workflow": ortools_context.bundle.component("product")["foundry_copy"][
            "groups"
        ]["workflow"]
    }
    assert localization["sha256"] == canonical_sha256(localization_core)
    manifest = ScenarioPredicateCatalogLoweringManifestV2()
    definitions = capabilities["hardPredicateCatalog"]["definitions"]
    kinds = [row["kind"] for row in definitions]
    registrations = {row.kind: row for row in manifest.registrations}
    assert len(kinds) == len(set(kinds)) == 78
    assert set(kinds) == set(registrations)
    assert "deal.card_owner" in kinds
    assert "state.observer_visibility" in kinds
    assert capabilities["backendIdentity"]["predicateCatalogHash"] == (
        manifest.manifest_hash
    )
    for definition in definitions:
        support = definition["cpSatSupport"]
        registration = registrations[definition["kind"]]
        assert support["search"]["status"] in {"exact", "derived"}
        assert support["exactInput"]["status"] == "exact"
        assert support["evidenceExtractor"] == registration.evidence_extractor_id
        assert registration.lowerer_owner
        assert registration.owning_test in definition["tests"]
    assert (
        ScenarioResearchV4CapabilityV1Response.model_validate(capabilities).model_dump(
            mode="json"
        )
        == capabilities
    )
    coercive = copy.deepcopy(capabilities)
    coercive["executionReady"] = 0
    with pytest.raises(ValidationError):
        ScenarioResearchV4CapabilityV1Response.model_validate(coercive)

    unknown = {**capabilities, "undeclared": True}
    with pytest.raises(ValidationError):
        ScenarioResearchV4CapabilityV1Response.model_validate(unknown)
    assert ResearchFoundryServiceV1._ortools_scenario_backend_v2(None) is None

    recipe = _recipe(authority.to_dict(), "ortools_cp_sat_v2")
    executor_identity = api.RESEARCH._scenario_executor_identity_v4(
        recipe,
        ortools_context,
    )
    assert executor_identity["kind"] == "ortools_scenario_execution_identity_v1"
    assert (
        executor_identity["backendIdentity"]["sha256"]
        == (capabilities["backendIdentity"]["sha256"])
    )
    assert executor_identity["executorIdentity"]["loweringManifestHash"] == (
        ScenarioPredicateCatalogLoweringManifestV2().manifest_hash
    )
    execution_parameters = api.RESEARCH._ortools_execution_parameters_v2(
        recipe,
        api.RESEARCH._ortools_scenario_backend_v2(ortools_context),
    )
    assert execution_parameters == {
        "maximumProofNodes": 2_000_000,
        "maximumProofDepth": 16,
        "maximumHiddenWorlds": 1,
        "maximumLineCountNodes": 2_000_000,
        "maximumLineCountLimbs": 128,
        "maximumCandidates": 1,
        "candidateLimit": 10_000,
        "maximumConflicts": 10_000_000,
        "maximumVariants": 100,
        "maximumPrefixLiftAttempts": 2_500,
        "maximumShardWorkers": 5,
        "requestedShardWorkers": 5,
        "cancellationPollMilliseconds": 25,
        "maximumSeconds": 1.0,
        "maximumSolveSeconds": 1.0,
        "maximumCampaignSeconds": 60.0,
        "seed": 1,
        "symmetryPolicyId": "rules_preserving_v1",
        "allowSymmetryOnlyVariants": False,
        "minimumDecisionStateDistance": 4,
        "proofBudgetHash": execution_parameters["proofBudgetHash"],
        "searchBudgetHash": execution_parameters["searchBudgetHash"],
    }
    decision_state = {
        "contract": "sun",
        "trumpSuit": None,
        "currentPlayer": "north",
        "buyer": "west",
        "dealer": "east",
        "hands": {
            seat: [Card(suit * 8 + rank).to_dict() for rank in range(2)]
            for suit, seat in enumerate(("north", "west", "south", "east"))
        },
        "currentTrick": None,
    }
    symmetry_document = build_rules_preserving_diversity_document_v1(decision_state)
    variants = tuple(
        (
            0,
            attempt,
            CampaignVariantV1.build(
                {"schemaVersion": 1, "candidate": attempt},
                symmetry_document=symmetry_document,
            ),
        )
        for attempt in (1, 2)
    )
    selected, diversity_receipts = select_campaign_variants_v1(
        variants,
        result_count=2,
        symmetry_key=rules_preserving_symmetry_key_v1,
        distance=decision_state_card_hamming_distance_v1,
        minimum_distance=execution_parameters["minimumDecisionStateDistance"],
    )
    assert len(selected) == 1
    assert [receipt.rejection_reason for receipt in diversity_receipts] == [
        None,
        "symmetry",
    ]
    validated = api.RESEARCH._validate_recipe_v4(recipe, ortools_context)
    estimate = api.RESEARCH._estimate_recipe_v4(
        recipe,
        validated,
        ortools_context,
    )

    assert validated["valid"] is True
    assert validated["issues"] == []
    ResearchRecipeValidationResponse.model_validate(validated)
    assert estimate["valid"] is True
    assert estimate["launchAdmissible"] is False
    assert estimate["proof"]["reason"] == "search_decision_selector_required"
    assert estimate["maximumSolverSeconds"] == 1.0
    assert estimate["candidateCount"] == 1
    ResearchRecipeEstimateResponse.model_validate(estimate)


def _forbid_legacy_exact_state_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy Python exact-state semantics were called")

    for name in (
        "_derive_play_prefix",
        "_replay_play_prefix",
        "_verify_project_assertions",
    ):
        monkeypatch.setattr(exact_state_module, name, forbidden)


def test_selected_validation_successor_previews_and_stores_retained_exact_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, context = _intelligent_context_and_authority()
    revision_id = context.bundle_revision_id
    challenge = context.bundle.component("challenge")
    assert challenge["profile"]["version"] == "3.14.5"
    assert challenge["scenarioResearchV4"]["searchBudgets"]["profiles"][
        "interactive_validation_v1"]["maximumWallSeconds"] == 5.0
    service, executor = _isolated_service(tmp_path, authority, context)
    _forbid_legacy_exact_state_semantics(monkeypatch)
    try:
        payload = _native_qualified_exact_recipe_payload(revision_id)
        preview = service.preview_exact_trace_legal_actions({
            "schemaVersion": 1, "configuration": payload["configuration"],
            "exactTracePrefix": payload["scenario"]["origin"]["exactTracePrefix"],
        })
        assert preview["valid"] is True, preview["diagnostics"]
        assert preview["configurationAuthority"]["bundleRevisionId"] == revision_id
        assert preview["legalActions"]
        validated = service.validate_recipe(payload)
        assert validated["valid"] is True, validated["issues"]
        stored = service.create_recipe(payload)
        assert stored["configuration"]["bundleRevisionId"] == revision_id
        assert service.get_recipe(stored["code"])["sha256"] == stored["sha256"]
    finally:
        executor.shutdown(wait=True)


def test_service_previews_only_legal_exact_trace_actions_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, context = _intelligent_context_and_authority()
    service, executor = _isolated_service(tmp_path, authority, context)
    _forbid_legacy_exact_state_semantics(monkeypatch)
    try:
        fixture = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "examples/research-scenario-v4-ace-sacrifice-two-aces.json"
            ).read_text(encoding="utf-8")
        )["exactTracePrefix"]
        fixture["playActions"] = []
        fixture["projectDeclarations"] = []
        fixture["projectAssertions"] = []
        fixture["stateAssertions"] = {}
        request = {
            "schemaVersion": 1,
            "configuration": {
                "schemaVersion": 1,
                "bundleRevisionId": context.bundle_revision_id,
            },
            "exactTracePrefix": fixture,
        }

        preview = service.preview_exact_trace_legal_actions(request)
        assert preview["valid"] is True, preview["diagnostics"]
        assert preview["currentStage"] == "project"
        assert preview["configurationAuthority"]["bundleRevisionId"] == (
            context.bundle_revision_id
        )
        assert preview["configurationAuthority"]["backendIdentityHash"]
        assert {item["stage"] for item in preview["legalActions"]} == {"play"}
        assert all(item["projectDeclarations"] for item in preview["legalActions"])

        illegal = copy.deepcopy(request)
        illegal["exactTracePrefix"]["deckOrder"][-1] = 0
        rejected = service.preview_exact_trace_legal_actions(illegal)
        assert rejected["valid"] is False
        assert rejected["legalActions"] == []
        assert rejected["currentStage"] is None
        assert rejected["diagnostics"] == [
            {
                "code": "exact_trace_prefix_invalid",
                "fieldPath": "exactTracePrefix",
            }
        ]

        mismatched = copy.deepcopy(request)
        mismatched["configuration"]["bundleRevisionId"] = "CBR-MISMATCH"
        with pytest.raises(ValueError, match="authority mismatch"):
            service.preview_exact_trace_legal_actions(mismatched)

        def exhaust(*args, **kwargs):
            raise exact_state_module.ExactNormalizationStopped("maximum_seconds")

        monkeypatch.setattr(exact_state_module.OrToolsExactStateNormalizerV1, "preview_legal_next_actions", exhaust)
        incomplete = service.preview_exact_trace_legal_actions(request)
        assert incomplete["valid"] is False
        assert incomplete["legalActions"] == []
        assert incomplete["diagnostics"] == [{
            "code": "exact_normalization_incomplete", "fieldPath": "exactTracePrefix",
        }]
        assert api.ExactTraceLegalActionsPreviewV1Response.model_validate(incomplete)
    finally:
        executor.shutdown(wait=True)


def test_exact_trace_legal_action_preview_api_is_closed_and_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision_id = "CBR-PREVIEW-ORTOOLS"
    assert api.ResearchConfigurationSelectorRequest.model_validate({
        "bundleRevisionId": revision_id,
    }).model_dump() == {"schemaVersion": 1, "bundleRevisionId": revision_id}
    request = {
        "schemaVersion": 1,
        "configuration": {"schemaVersion": 1, "bundleRevisionId": revision_id},
        "exactTracePrefix": {
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
        },
    }
    response = {
        "schemaVersion": 1,
        "configurationAuthority": {
            "bundleRevisionId": revision_id,
            "semanticBundleHash": "1" * 64,
            "effectiveRuntimeConfigHash": "2" * 64,
            "backendIdentityHash": "3" * 64,
        },
        "valid": True,
        "currentStage": "auction",
        "nextActor": "north",
        "terminal": False,
        "legalActions": [
            {
                "stage": "auction",
                "code": "sun",
                "actor": "north",
                "action": "sun",
                "trumpSuit": None,
            }
        ],
        "diagnostics": [],
        "prefixHash": "4" * 64,
        "previewStateHash": "5" * 64,
        "previewReceiptHash": "6" * 64,
        "sha256": "7" * 64,
    }
    calls: list[dict] = []
    monkeypatch.setattr(
        api.RESEARCH,
        "preview_exact_trace_legal_actions",
        lambda payload: calls.append(payload) or response,
    )

    with TestClient(api.app) as client:
        accepted = client.post(
            "/api/v2/research-exact-traces:preview-legal-actions",
            json=request,
        )
        rejected = client.post(
            "/api/v2/research-exact-traces:preview-legal-actions",
            json={**request, "unknown": True},
        )
        for invalid_version in (True, 1.0, "1"):
            for field in ("schemaVersion", "configuration"):
                malformed = copy.deepcopy(request)
                if field == "configuration":
                    malformed[field]["schemaVersion"] = invalid_version
                else:
                    malformed[field] = invalid_version
                invalid = client.post(
                    "/api/v2/research-exact-traces:preview-legal-actions",
                    json=malformed,
                )
                assert invalid.status_code == 422, (field, invalid_version, invalid.text)

        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["legalActions"][0]["code"] == "sun"
        expected_request = copy.deepcopy(request)
        expected_request["exactTracePrefix"]["adjudicationEvents"] = []
        assert calls == [expected_request]
    assert rejected.status_code == 422


def test_v3_exact_trace_validation_api_uses_native_cp_sat_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(
            legacy_context.bundle,
            _ortools_challenge(quality=True),
        ),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    _forbid_legacy_exact_state_semantics(monkeypatch)
    monkeypatch.setattr(api, "RESEARCH", service)
    payload = _exact_recipe_payload(context.bundle_revision_id)

    try:
        with TestClient(api.app) as client:
            response = client.post(
                "/api/v2/research-recipes:validate",
                json=payload,
            )

        assert response.status_code == 200, response.text
        document = response.json()
        assert document["valid"] is True, document["issues"]
        assert document["issues"] == []
        assert document["normalizedExactState"]["exactStateId"] == (
            "STATE-E4A2F2B807C9650F8DFE"
        )
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize("quality", [False, True])
def test_v3_capabilities_admit_only_frozen_full_round_quality(
    tmp_path: Path, quality: bool,
) -> None:
    authority, base_context = _frozen_context_and_authority()
    context = replace(
        base_context,
        bundle=_ChallengeBundleProxy(base_context.bundle, _ortools_challenge(quality=quality)),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        capability = service.capabilities(bundle_revision_id=context.bundle_revision_id)[
            "research"
        ]["scenarioResearchV4"]
        assert capability["executionReady"] is quality
        assert capability["launchAdmissible"] is quality
        assert (
            "playingQuality" in capability["executionCapabilities"]["authoringCatalog"]
        ) is quality
    finally:
        executor.shutdown(wait=True)


def test_v3_api_refuses_unproved_all_decision_quality_without_running_solver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, context = _frozen_context_and_authority()
    service, executor = _isolated_service(tmp_path, authority, context)
    monkeypatch.setattr(api, "RESEARCH", service)
    payload = _exact_recipe_payload(context.bundle_revision_id, quality=False)

    def forbidden_solve(*_args, **_kwargs):
        pytest.fail("Unsupported full-round quality must fail before solving")

    monkeypatch.setattr(OrToolsUnifiedResearchExecutorV1, "execute_campaign", forbidden_solve)
    try:
        with TestClient(api.app) as client:
            response = client.post("/api/v2/research-recipes:validate", json=payload)
        assert response.status_code == 200, response.text
        document = response.json()
        assert document["valid"] is False
        issue = next(
            item for item in document["issues"]
            if item["code"] == "all_decision_quality_evidence_unavailable"
        )
        assert issue["fieldPaths"] == ["qualification.requireAllDecisionAudit"]
        assert issue["severity"] == "error"
        assert issue["details"]["detail"] == "selected_decision_proof_only"
        ResearchRecipeValidationResponse.model_validate(document)
    finally:
        executor.shutdown(wait=True)


@pytest.mark.real_clock
@pytest.mark.parametrize("quality_worlds", [1, 2])
def test_v3_api_validates_frozen_budgeted_quality_search(tmp_path, monkeypatch, quality_worlds, record_property):
    authority, context = _frozen_context_and_authority()
    root = Path(__file__).resolve().parents[1]
    challenge = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
    challenge["scenarioResearchV4"]["qualityExecution"].pop("featuredPlayNativeProof", None)
    if quality_worlds == 2:
        challenge["scenarioResearchV4"]["proofBudgets"]["profiles"]["late_game_exact_v1"].update(
            maximumHiddenWorlds=2, maximumNodes=1000, maximumDepth=8,
            maximumWallSeconds=15.0, maximumCpuSeconds=15.0)
    context = replace(context, bundle=_ChallengeBundleProxy(context.bundle, challenge))
    service, executor = _isolated_service(tmp_path, authority, context)
    monkeypatch.setattr(api, "RESEARCH", service)
    payload = _exact_recipe_payload(context.bundle_revision_id)
    if quality_worlds == 2:
        # Keep this full-stack quality test on its deterministic qualifying
        # witness; target-miss no-good retries are covered by the cheap
        # campaign-coordinator tests.
        payload["resources"].update(
            seedStart=3,
            solverTimeoutSeconds=300.0,
            proofBudgetProfile="offline_heavyweight_v1",
        )
        payload["proof"] = {"schemaVersion": 1, "mode": "exact", "visibility": "complete_state",
            "claim": "outcome_reachable", "targetTeam": "west_east", "utility": "round_win",
            "successThreshold": 1.0, "budgetProfile": "offline_heavyweight_v1"}
    payload["scenario"] = {"schemaVersion": 1, "origin": {"kind": "search"},
        "constraints": {"predicateId": "PRED-CONTRACT", "kind": "auction.contract", "version": 1,
                        "hardness": "hard", "arguments": {"contract": "hokum"}},
        "decisionPoint": {"kind": "play", "selector": {"trickIndex": 6}},
        "outcome": {"predicateId": "PRED-WINNER", "kind": "outcome.team_round", "version": 1,
                    "hardness": "hard", "arguments": {"team": "west_east", "status": "won"}}}
    payload["playingQuality"] = _playing_quality_payload()
    def forbidden_solve(*args, **kwargs):
        pytest.fail("validation must not launch a campaign")
    real_campaign = OrToolsUnifiedResearchExecutorV1.execute_campaign
    monkeypatch.setattr(OrToolsUnifiedResearchExecutorV1, "execute_campaign", forbidden_solve)
    try:
        with TestClient(api.app) as client:
            response = client.post("/api/v2/research-recipes:validate", json=payload)
            assert response.status_code == 200, response.text
            assert response.json()["valid"], response.json()["issues"]
            created = client.post("/api/v2/research-recipes", json=payload,
                headers={"Idempotency-Key": "quality-api-create"})
            assert created.status_code == 201, created.text
            reopened = client.get(f"/api/v2/research-recipes/{created.json()['code']}")
            assert reopened.status_code == 200, reopened.text
            stored = reopened.json()
            assert stored["playingQuality"] == created.json()["playingQuality"]
            assert stored["playingQuality"]["decisions"] == payload["playingQuality"]["decisions"]
            assert stored["configuration"]["bundleRevisionId"] == context.bundle_revision_id
            monkeypatch.setattr(OrToolsUnifiedResearchExecutorV1, "execute_campaign", real_campaign)
            launched = client.post("/api/v2/research-runs", json={"recipeCode": stored["code"]},
                headers={"Idempotency-Key": "quality-api-run"})
            assert launched.status_code == 202, launched.text
            executor.shutdown(wait=True)
            completed = client.get(f"/api/v2/research-runs/{launched.json()['runCode']}")
            assert completed.status_code == 200, completed.text
            run_document = completed.json()
            assert run_document["status"] == "completed", run_document.get("events")
            assert run_document["campaignAccounting"]["stopReason"] != "worker_failed"
            if quality_worlds == 2:
                assert run_document["campaignAccounting"]["qualified"] == 1, run_document["campaignAccounting"]["stopReason"]
                assert len(run_document["resultCodes"]) == 1
                assert run_document["auditResultCodes"] == []
            codes = [*run_document["resultCodes"], *run_document["auditResultCodes"]]
            assert codes, run_document
            for code in codes:
                reopened_result = client.get(f"/api/v2/research-results/{code}")
                assert reopened_result.status_code == 200, reopened_result.text
                result = reopened_result.json()
                stored_result = service.get_result(code)
                assert stored_result is not None
                trace = json.loads(service.artifact_store.get(
                    stored_result["artifacts"]["completeTrace"]["uri"]
                ))
                plays = [row for row in trace["playingQualityEvidence"]["receipt"]["postAuction"]["projectPlay"]["decisions"]
                         if row["decisionKind"] == "play"]
                assert len(plays) == 32
                assert all(row["selectionReason"] == "native_actor_information_set_best_value"
                           for row in plays)
                record_property("playingQualitySeconds",
                                stored_result["executionReceipt"]["stageTimings"]["playingQuality"])
                timings = stored_result["executionReceipt"]["stageTimings"]
                assert 0 < timings["qualityAuction"] < timings["playingQuality"]
                assert 0 < timings["qualityProjectPlay"] < timings["playingQuality"]
                if quality_worlds == 2:
                    assert result["qualification"]["qualified"] is True
                    assert result["runtimeCompatibilityStatus"] == "passed"
                    assert result["proofOutcome"]["completionStatus"] == "exact_complete"
                    video_profile = yaml.safe_load(
                        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
                    )
                    prepared = ResearchRenderPreparationServiceV1(
                        research=service,
                        artifact_store=service.artifact_store,
                        publication_service=SimpleNamespace(),
                        renderer=RemotionProfessionalRenderer(video_profile, repository_root=root),
                        rules=context.bundle.rules,
                        configuration_store=api.CONTAINER.configuration_store,
                    ).prepare(code)
                    assert {row["value"] for row in result["proofOutcome"]["rootActions"]} == {1}
                    assert prepared["publicationEligible"] is False
                    assert prepared["publicationBlockers"] == [
                        "critical_decision_not_proven",
                    ]
                    assert prepared["source"]["auditOnly"] is True
                    batch = RenderBatchServiceV1(
                        jobs=None, background_executor=None, batch_executor=None,
                        artifact_store=service.artifact_store,
                        output_root=tmp_path / "render-output", maximum_attempts=1,
                    )
                    bundle = batch._load_bundle(prepared["bundleArtifact"])
                    assert bundle.verify()
                    assert batch._validated_proof_artifact(
                        prepared["proofArtifact"], bundle,
                        authority_receipt=prepared["authorityReceipt"],
                    ) == prepared["proofArtifact"]["sha256"]
                if result["proofOutcome"]["completionStatus"] == "proof_incomplete":
                    assert result["qualification"]["qualified"] is False
                    public_projection = result["scenarioProjection"]
                    assert "exactStateId" not in public_projection
                    assert ("decisionReference" in public_projection) is public_projection["exactStateAvailable"]
        assert response.status_code == 200, response.text
        document = response.json()
        assert document["valid"], document["issues"]
        ResearchRecipeValidationResponse.model_validate(document)
    finally:
        executor.shutdown(wait=True)


def test_supported_search_play_slice_is_launch_admissible() -> None:
    authority, legacy_context = _frozen_context_and_authority()
    ortools_context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge(quality=True)),
    )
    recipe = _recipe(
        authority.to_dict(),
        "ortools_cp_sat_v3",
        trick_index=5,
        decision_remaining_cards=3,
        decision_seat="south",
        decision_position=3,
        quality=True,
    )
    validated = api.RESEARCH._validate_recipe_v4(recipe, ortools_context)

    estimate = api.RESEARCH._estimate_recipe_v4(
        recipe,
        validated,
        ortools_context,
    )

    assert validated["valid"] is True
    assert estimate["launchAdmissible"] is True
    native_plan = validated["nativeCapabilityPlan"]
    assert native_plan["schemaVersion"] == 3
    assert native_plan["rows"][0]["qualificationBinding"] == "cp_sat_literal"
    assert estimate["nativeCapabilityPlanHash"] == native_plan["sha256"]
    assert api.RESEARCH._scenario_executor_identity_v4(
        recipe, ortools_context
    )["nativeCapabilityPlanHash"] == native_plan["sha256"]
    assert (
        api.ResearchRecipeValidationResponse.model_validate(validated)
        .nativeCapabilityPlan.sha256
        == native_plan["sha256"]
    )
    assert (
        api.ResearchRecipeEstimateResponse.model_validate(estimate)
        .nativeCapabilityPlanHash
        == native_plan["sha256"]
    )
    assert estimate["predicatesByConstructionCapability"]["cp_sat_exact"] == [
        "PRED-ORTOOLS-DEAL"
    ]
    assert estimate["unsupportedPredicates"] == []
    assert estimate["proof"]["admissible"] is True
    assert estimate.get("refusalReason") is None
    assert estimate["proof"]["admissible"] is True
    assert "reason" not in estimate["proof"]
    assert estimate["candidateCount"] == 1
    assert estimate["maximumSolverSeconds"] == 1.0
    assert estimate["backendIdentity"]["backendId"] == "ortools_cp_sat_v3"
    assert estimate["proof"]["remainingCardCount"] == 9
    assert estimate["proof"]["requestedDepth"] == 9
    assert estimate["estimatedGraphNodes"] == 286
    assert estimate["estimatedGraphEdges"] == 285
    assert estimate["estimatedHiddenWorlds"] == 1
    assert estimate["estimatedGraphNodes"] is not None
    assert estimate["estimatedGraphEdges"] is not None
    assert estimate["estimatedModelVariables"] > 0
    assert estimate["estimatedModelConstraints"] > 0
    assert estimate["firstResultCostClass"] == "not_estimable"
    assert estimate["requestedResultsCostClass"] == "not_estimable"
    assert "playing_quality_full_round_cost_not_estimated" in estimate["warnings"]
    ResearchRecipeEstimateResponse.model_validate(estimate)


def test_v3_1_capability_catalog_uses_frozen_shard_authoring_default() -> None:
    _authority, context = _frozen_context_and_authority()

    response = api.RESEARCH.capabilities(bundle_revision_id=context.bundle_revision_id)
    scenario = response["research"]["scenarioResearchV4"]
    catalog = scenario["executionCapabilities"]["authoringCatalog"]

    assert context.bundle.bundle_profile["profile"]["version"] == "24.0.1"
    assert context.bundle.component("challenge")["profile"]["version"] == "3.1.0"
    assert scenario["backendIdentity"]["authorityHash"]
    assert catalog["fields"]["resources.seedsPerShard"]["default"] == 1
    worker_field = catalog["fields"]["resources.shardWorkerCount"]
    assert worker_field["default"] is None
    assert worker_field["maximumByBudgetProfile"]["offline_heavyweight_v1"] == 6


def test_v3_recipe_caps_shard_workers_without_changing_seed_universe(monkeypatch) -> None:
    monkeypatch.setattr(api.RESEARCH, "maximum_local_shard_workers", 2)
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(
            legacy_context.bundle, _ortools_challenge(quality=True)
        ),
    )
    base = _recipe(
        authority.to_dict(), "ortools_cp_sat_v3", trick_index=6,
        candidate_count=4, seeds_per_shard=1, result_count=2, quality=True,
    ).to_dict()
    base.pop("code", None)
    base.pop("sha256", None)
    base["resources"]["shardWorkerCount"] = 2
    recipe = ResearchRecipeV4.build(base)
    estimate = api.RESEARCH._estimate_recipe_v4(
        recipe, api.RESEARCH._validate_recipe_v4(recipe, context), context,
    )
    assert estimate["sharding"]["requestedShardCount"] == 4
    assert estimate["sharding"]["effectiveShardWorkers"] == 2
    ResearchRecipeEstimateResponse.model_validate(estimate)
    parameters = api.RESEARCH._ortools_execution_parameters_v2(
        recipe, api.RESEARCH._ortools_scenario_backend_v2(context),
    )
    assert parameters["requestedShardWorkers"] == 2
    assert recipe.resources["seedCount"] == 4
    monkeypatch.setattr(api.RESEARCH, "maximum_local_shard_workers", 1)
    local_estimate = api.RESEARCH._estimate_recipe_v4(
        recipe, api.RESEARCH._validate_recipe_v4(recipe, context), context,
    )
    assert local_estimate["sharding"]["requestedShardWorkers"] == 2
    assert local_estimate["sharding"]["effectiveShardWorkers"] == 1
    assert local_estimate["sharding"]["parallelismLimitedByShardPlan"] is False
    assert recipe.resources["seedCount"] == 4
    for invalid in (0, True, 2.0):
        base["resources"]["shardWorkerCount"] = invalid
        with pytest.raises((TypeError, ValueError)):
            ResearchRecipeV4.build(base)
    base["resources"]["shardWorkerCount"] = 7
    invalid_validation = api.RESEARCH._validate_recipe_v4(
        ResearchRecipeV4.build(base), context,
    )
    assert invalid_validation["valid"] is False
    assert any(issue["code"] == "shard_worker_budget_exceeded" for issue in invalid_validation["issues"])


def test_hokum_actor_information_estimate_requires_full_round_quality() -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(
            legacy_context.bundle,
            _ortools_challenge(quality=False),
        ),
    )
    payload = _actor_information_exact_recipe_payload(context.bundle_revision_id)
    payload.pop("playingQuality")
    payload["configuration"] = authority.to_dict()
    recipe = ResearchRecipeV4.build(payload)
    validated = api.RESEARCH._validate_recipe_v4(recipe, context)
    assert validated["valid"] is False
    assert any(row["code"] == "all_decision_quality_evidence_unavailable" for row in validated["issues"])
    assert "normalizedExactState" not in validated
    # Supply an independently normalized state to exercise estimator arithmetic;
    # the failed recipe admission remains authoritative.
    normalized = exact_state_module.OrToolsExactStateNormalizerV1(
        context.bundle.rules, native_cp_sat=True,
    ).normalize(recipe.scenario.origin.exact_trace_prefix,
        decision_point=recipe.scenario.decision_point).to_dict()
    normalized["state"]["contract"] = "hokum"
    normalized["state"]["trumpSuit"] = "diamonds"
    validated = {**validated, "normalizedExactState": normalized}

    estimate = api.RESEARCH._estimate_recipe_v4(recipe, validated, context)

    assert estimate["launchAdmissible"] is False
    assert estimate["proof"]["admissible"] is False
    assert estimate["refusalReason"] == "frozen_quality_execution_unavailable"
    assert estimate.get("refusalReason") != (
        "actor_information_set_contract_unsupported"
    )


@pytest.mark.parametrize("contract", ("sun", "ashkal"))
def test_search_actor_information_estimate_reports_proven_service_support(contract: str) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge(quality=True)),
    )
    raw = _recipe(
        authority.to_dict(),
        "ortools_cp_sat_v3",
        trick_index=7,
        quality=True,
        decision_remaining_cards=1,
        decision_seat="south",
    ).to_dict()
    raw.pop("sha256")
    raw.pop("capabilityPlan")
    raw["scenario"].pop("sha256")
    raw["scenario"]["constraints"].pop("sha256")
    raw["proof"].pop("sha256")
    raw["scenario"]["constraints"]["children"] = [
        {
            "predicateId": "PRED-ACTOR-SEARCH-SUN",
            "kind": "auction.contract",
            "version": 1,
            "hardness": "hard",
            "arguments": {"contract": contract},
        }
    ]
    raw["proof"].update(
        {
            "visibility": "actor_information_set",
            "strategyClass": "pure_deterministic",
            "budgetProfile": "actor_information_set_exact_v1",
        }
    )
    raw["resources"]["proofBudgetProfile"] = "actor_information_set_exact_v1"
    recipe = ResearchRecipeV4.build(raw)
    validated = api.RESEARCH._validate_recipe_v4(recipe, context)

    estimate = api.RESEARCH._estimate_recipe_v4(recipe, validated, context)

    assert validated["valid"] is True
    assert estimate["launchAdmissible"] is True
    assert estimate["predicatesByConstructionCapability"]["cp_sat_exact"] == [
        "PRED-ACTOR-SEARCH-SUN"
    ]
    assert estimate["nativeCapabilityPlanHash"] == validated[
        "nativeCapabilityPlan"
    ]["sha256"]


def test_estimate_exposes_authoritative_tail_first_position_requirement() -> None:
    authority, context = _frozen_context_and_authority()
    raw = _recipe(
        authority.to_dict(),
        "ortools_cp_sat_v3",
        trick_index=6,
        decision_remaining_cards=2,
        decision_seat="south",
    ).to_dict()
    raw.pop("sha256")
    raw.pop("capabilityPlan")
    raw["scenario"].pop("sha256")
    raw["scenario"]["constraints"].pop("sha256")
    raw["scenario"]["constraints"]["children"].append({
        "predicateId": "PRED-TAIL-STATE",
        "kind": "state.legal_action_count",
        "version": 1,
        "hardness": "hard",
        "arguments": {"minimum": 2, "maximum": 8},
    })
    raw["scenario"]["constraints"]["children"].append({
        "predicateId": "PRED-TAIL-SUN",
        "kind": "auction.contract",
        "version": 1,
        "hardness": "hard",
        "arguments": {"contract": "sun"},
    })

    wildcard = ResearchRecipeV4.build(raw)
    wildcard_estimate = api.RESEARCH._estimate_recipe_v4(
        wildcard,
        api.RESEARCH._validate_recipe_v4(wildcard, context),
        context,
    )
    assert wildcard_estimate["constructionPlan"] == {
        "schemaVersion": 1,
        "planner": "generic",
        "fallbackReason": "decision_boundary_selector_incomplete",
        "prefixLength": None,
        "tailOffset": None,
        "tailLength": None,
        "sha256": wildcard_estimate["constructionPlan"]["sha256"],
    }
    api.ResearchRecipeEstimateResponse.model_validate(wildcard_estimate)

    fixed_raw = copy.deepcopy(raw)
    fixed_raw["scenario"]["decisionPoint"]["selector"]["position"] = 0
    fixed = ResearchRecipeV4.build(fixed_raw)
    fixed_estimate = api.RESEARCH._estimate_recipe_v4(
        fixed,
        api.RESEARCH._validate_recipe_v4(fixed, context),
        context,
    )
    assert fixed_estimate["constructionPlan"] == {
        "schemaVersion": 1,
        "planner": "tail_first",
        "fallbackReason": None,
        "prefixLength": 24,
        "tailOffset": 24,
        "tailLength": 8,
        "sha256": fixed_estimate["constructionPlan"]["sha256"],
    }

    open_actor_raw = copy.deepcopy(fixed_raw)
    open_actor_raw["scenario"]["decisionPoint"]["selector"].pop("actingSeat")
    open_actor = ResearchRecipeV4.build(open_actor_raw)
    open_actor_estimate = api.RESEARCH._estimate_recipe_v4(
        open_actor,
        api.RESEARCH._validate_recipe_v4(open_actor, context),
        context,
    )
    assert open_actor_estimate["constructionPlan"]["planner"] == "generic"
    assert open_actor_estimate["constructionPlan"]["fallbackReason"] == (
        "decision_actor_selector_incomplete"
    )


def test_legacy_estimate_actor_information_nonplay_uses_normalized_exact_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(
            legacy_context.bundle,
            _ortools_challenge(quality=False),
        ),
    )
    payload = _nonplay_complete_graph_recipe_payload(
        context.bundle_revision_id,
        "escalation",
        mode="exact",
    )
    payload.pop("playingQuality")
    payload["configuration"] = authority.to_dict()
    complete_state_recipe = ResearchRecipeV4.build(payload)
    validated = api.RESEARCH._validate_recipe_v4(complete_state_recipe, context)
    assert validated["valid"] is False
    assert any(row["code"] == "all_decision_quality_evidence_unavailable" for row in validated["issues"])
    assert "normalizedExactState" not in validated
    normalized = exact_state_module.OrToolsExactStateNormalizerV1(
        context.bundle.rules, native_cp_sat=True,
    ).normalize(complete_state_recipe.scenario.origin.exact_trace_prefix,
        decision_point=complete_state_recipe.scenario.decision_point).to_dict()
    validated = {**validated, "normalizedExactState": normalized}
    payload["proof"].update(
        {
            "visibility": "actor_information_set",
            "strategyClass": "pure_deterministic",
            "budgetProfile": "actor_information_set_exact_v1",
        }
    )
    payload["resources"]["proofBudgetProfile"] = "actor_information_set_exact_v1"
    recipe = ResearchRecipeV4.build(payload)
    monkeypatch.setattr(
        api.RESEARCH,
        "_cached_ortools_scenario_backend_v2",
        lambda _context: None,
    )

    estimate = api.RESEARCH._estimate_recipe_v4(recipe, validated, context)

    assert estimate["proof"]["estimatedHiddenWorldUpperBound"] is not None
    assert estimate["proof"]["estimatedHiddenWorldUpperBound"] >= 1


def test_exact_bid_bounded_estimate_preserves_depth_but_requires_quality() -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(
            legacy_context.bundle,
            _ortools_challenge(quality=False),
        ),
    )
    payload = _exact_recipe_payload(context.bundle_revision_id)
    payload.pop("playingQuality")
    prefix = payload["scenario"]["origin"]["exactTracePrefix"]
    prefix.update(
        {
            "auctionActions": [],
            "escalationActions": [],
            "projectActions": [],
            "projectDeclarations": [],
            "projectAssertions": [],
            "playActions": [],
            "stateAssertions": {},
        }
    )
    payload["configuration"] = authority.to_dict()
    payload["scenario"].update(
        {
            "constraints": {
                "predicateId": "PRED-BID-ESTIMATE-PHASE",
                "kind": "state.phase",
                "version": 1,
                "hardness": "hard",
                "arguments": {"state": "bidding"},
            },
            "decisionPoint": {"kind": "bid", "selector": {"actionIndex": 0}},
            "outcome": {
                "predicateId": "PRED-BID-ESTIMATE-REPORT",
                "kind": "state.phase",
                "version": 1,
                "hardness": "report",
                "arguments": {"state": "bidding"},
            },
        }
    )
    payload["proof"] = {
        "schemaVersion": 1,
        "mode": "bounded",
        "visibility": "complete_state",
        "strategyClass": "pure_deterministic",
        "claim": "outcome_reachable",
        "targetTeam": "north_south",
        "utility": "round_win",
        "successThreshold": 0,
        "horizon": {"unit": "plies", "value": 1},
        "boundaryEvaluator": "current_raw_point_margin_v1",
        "budgetProfile": "late_game_exact_v1",
    }
    recipe = ResearchRecipeV4.build(payload)
    validated = api.RESEARCH._validate_recipe_v4(recipe, context)

    estimate = api.RESEARCH._estimate_recipe_v4(recipe, validated, context)

    assert validated["valid"] is False
    assert estimate["launchAdmissible"] is False
    assert estimate["proof"]["admissible"] is False
    assert estimate["proof"]["remainingCardCount"] == 32
    assert estimate["proof"]["requestedDepth"] == 1
    assert estimate["refusalReason"] == "frozen_quality_execution_unavailable"


def test_exact_bid_estimate_includes_the_complete_downstream_play_graph() -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(
            legacy_context.bundle,
            _ortools_challenge(quality=False),
        ),
    )
    payload = _nonplay_complete_graph_recipe_payload(
        context.bundle_revision_id,
        "bid",
        mode="exact",
    )
    payload.pop("playingQuality")
    payload["configuration"] = authority.to_dict()
    recipe = ResearchRecipeV4.build(payload)
    validated = api.RESEARCH._validate_recipe_v4(recipe, context)

    estimate = api.RESEARCH._estimate_recipe_v4(recipe, validated, context)

    assert validated["valid"] is False
    assert any(row["code"] == "all_decision_quality_evidence_unavailable" for row in validated["issues"])
    assert estimate["proof"]["remainingCardCount"] == 32
    assert estimate["proof"]["requestedDepth"] == 33
    assert estimate["estimatedGraphNodes"] > int(
        estimate["proof"]["budgetStateSpaceLimit"]
    )
    assert estimate["launchAdmissible"] is False
    assert estimate["refusalReason"] == "frozen_quality_execution_unavailable"


def test_search_estimate_uses_frozen_shards_and_candidate_variant_limits() -> None:
    authority, legacy_context = _frozen_context_and_authority()
    challenge = _ortools_challenge(quality=True)
    challenge["scenarioResearchV4"]["searchBudgets"]["profiles"][
        "late_game_exact_v1"
    ].update({"maximumCandidates": 6, "maximumVariants": 2})
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, challenge),
    )
    recipe = _recipe(
        authority.to_dict(),
        "ortools_cp_sat_v3",
        trick_index=5,
        decision_remaining_cards=3,
        decision_seat="south",
        decision_position=3,
        candidate_count=6,
        quality=True,
        seeds_per_shard=2,
        result_count=2,
    )
    validated = api.RESEARCH._validate_recipe_v4(recipe, context)

    estimate = api.RESEARCH._estimate_recipe_v4(recipe, validated, context)

    assert estimate["launchAdmissible"] is True
    assert estimate["candidateCount"] == 6
    assert estimate["shardCount"] == 3
    # Full-round quality includes bids, escalation opportunities, declarations
    # and every card play, not only legal choices at the selected late root.
    assert estimate["analysisDecisionUpperBound"] == 6 * (
        2 * len(Seat) + len(Seat) + len(FULL_DECK)
        + challenge["scenarioResearchV4"]["qualityExecution"]["maximumEscalationDecisions"]
    )
    assert estimate["sharding"] == {
        "configuredSeedsPerShard": 2,
        "requestedShardCount": 3,
        "actualShardCount": 3,
        "maximumShardWorkers": 5,
        "requestedShardWorkers": 5,
        "effectiveShardWorkers": 3,
        "maximumShardWallSeconds": 60.0,
        "parallelismLimitedByShardPlan": False,
        "recommendedSeedsPerShard": 2,
    }
    assert estimate["output"]["maximumCandidateCount"] == 6
    assert estimate["output"]["maximumVariantCount"] == 2

    candidate_refusal = copy.deepcopy(challenge)
    candidate_refusal["scenarioResearchV4"]["searchBudgets"]["profiles"][
        "late_game_exact_v1"
    ]["maximumCandidates"] = 5
    candidate_refusal["scenarioResearchV4"]["qualityExecution"]["maximumEscalationDecisions"] = 12
    candidate_context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, candidate_refusal),
    )
    refused = api.RESEARCH._estimate_recipe_v4(
        recipe,
        api.RESEARCH._validate_recipe_v4(recipe, candidate_context),
        candidate_context,
    )
    assert refused["launchAdmissible"] is False
    assert refused["refusalReason"] == "requested_seed_count_exceeds_candidate_budget"
    assert refused["analysisDecisionUpperBound"] == 6 * (
        2 * len(Seat) + len(Seat) + len(FULL_DECK) + 12
    )

    variant_refusal = copy.deepcopy(challenge)
    variant_refusal["scenarioResearchV4"]["searchBudgets"]["profiles"][
        "late_game_exact_v1"
    ]["maximumVariants"] = 1
    variant_context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, variant_refusal),
    )
    refused = api.RESEARCH._estimate_recipe_v4(
        recipe,
        api.RESEARCH._validate_recipe_v4(recipe, variant_context),
        variant_context,
    )
    assert refused["launchAdmissible"] is False
    assert refused["refusalReason"] == "requested_result_count_exceeds_variant_budget"


@pytest.mark.parametrize("visibility", ("complete_state", "actor_information_set"))
@pytest.mark.parametrize("mode", ("exact", "budgeted"))
def test_search_admission_checks_whole_round_first_bid_budget(visibility, mode) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(legacy_context, bundle=_ChallengeBundleProxy(
        legacy_context.bundle, _ortools_challenge(quality=True),
    ))
    raw = _recipe(authority.to_dict(), "ortools_cp_sat_v3", trick_index=6,
        decision_remaining_cards=2, decision_seat="south", quality=True).to_dict()
    raw.pop("sha256", None)
    quality = _playing_quality_payload()
    quality.update(schemaVersion=2, visibility=visibility, mode=mode)
    if mode == "exact":
        for decision in quality["decisions"].values():
            decision.pop("horizon")
            decision.pop("boundaryEvaluator")
            decision["budgetProfile"] = "offline_heavyweight_v1"
    raw["playingQuality"] = quality
    recipe = ResearchRecipeV4.build(raw)
    validation = api.RESEARCH._validate_recipe_v4(recipe, context)
    assert validation["valid"] is True
    estimate = api.RESEARCH._estimate_recipe_v4(recipe, validation, context)
    assert estimate["launchAdmissible"] is True, estimate
    api.RESEARCH._require_v4_launch_admission(recipe, estimate)
    if mode == "exact" and visibility == "actor_information_set":
        assert "playing_quality_full_round_cost_not_estimated" in estimate["warnings"]
        # Without the frozen native bid tier, execution still enumerates worlds.
        for fallback in ("no_native_tier", "other_utility"):
            enumeration = copy.deepcopy(context.bundle.component("challenge"))
            enumeration_raw = copy.deepcopy(raw)
            if fallback == "no_native_tier":
                enumeration["scenarioResearchV4"]["proofBudgets"]["profiles"][
                    "offline_heavyweight_v1"
                ].pop("nativeObserved")
            else:
                enumeration_raw["playingQuality"]["decisions"]["bid"]["utility"] = "contract_success"
            enumeration_recipe = ResearchRecipeV4.build(enumeration_raw)
            enumeration_context = replace(context, bundle=_ChallengeBundleProxy(context.bundle, enumeration))
            checked = api.RESEARCH._validate_recipe_v4(enumeration_recipe, enumeration_context)
            refused = api.RESEARCH._estimate_recipe_v4(enumeration_recipe, checked, enumeration_context)
            assert refused["launchAdmissible"] is False
            assert refused["refusalReason"] == "playing_quality_first_bid_hidden_worlds_exceed_budget"
            assert int(refused["estimatedHiddenWorlds"]) > 25_000
            with pytest.raises(ResearchLaunchAdmissionError):
                api.RESEARCH._require_v4_launch_admission(enumeration_recipe, refused)
    ResearchRecipeEstimateResponse.model_validate(estimate)

    if mode == "exact":
        for kind in quality["decisions"]:
            shallow = copy.deepcopy(raw)
            shallow["playingQuality"]["decisions"][kind]["budgetProfile"] = "late_game_exact_v1"
            shallow_recipe = ResearchRecipeV4.build(shallow)
            shallow_validation = api.RESEARCH._validate_recipe_v4(shallow_recipe, context)
            assert shallow_validation["valid"] is True
            refused = api.RESEARCH._estimate_recipe_v4(shallow_recipe, shallow_validation, context)
            assert refused["launchAdmissible"] is False
            assert refused["refusalReason"] == "playing_quality_full_round_depth_exceeds_budget"
            with pytest.raises(ResearchLaunchAdmissionError):
                api.RESEARCH._require_v4_launch_admission(shallow_recipe, refused)


def test_search_estimate_warns_when_multi_result_plan_has_only_one_shard() -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge(quality=True)),
    )
    recipe = _recipe(
        authority.to_dict(),
        "ortools_cp_sat_v3",
        trick_index=6,
        decision_remaining_cards=2,
        decision_seat="south",
        candidate_count=50,
        quality=True,
        seeds_per_shard=50,
        result_count=5,
    )

    estimate = api.RESEARCH._estimate_recipe_v4(
        recipe,
        api.RESEARCH._validate_recipe_v4(recipe, context),
        context,
    )

    assert estimate["launchAdmissible"] is True
    assert estimate["shardCount"] == 1
    assert estimate["sharding"] == {
        "configuredSeedsPerShard": 50,
        "requestedShardCount": 1,
        "actualShardCount": 1,
        "maximumShardWorkers": 5,
        "requestedShardWorkers": 5,
        "effectiveShardWorkers": 1,
        "maximumShardWallSeconds": 60.0,
        "parallelismLimitedByShardPlan": True,
        "recommendedSeedsPerShard": 1,
    }
    assert estimate["firstResultCostClass"] == "not_estimable"
    assert estimate["requestedResultsCostClass"] == "not_estimable"
    assert "parallelism_limited_by_shard_plan" in estimate["warnings"]
    assert "playing_quality_full_round_cost_not_estimated" in estimate["warnings"]


def test_search_estimate_uses_independent_shards_to_find_one_rare_result() -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(
            legacy_context.bundle, _ortools_challenge(quality=True)
        ),
    )
    recipe = _recipe(
        authority.to_dict(),
        "ortools_cp_sat_v3",
        trick_index=6,
        decision_remaining_cards=2,
        decision_seat="south",
        candidate_count=4,
        quality=True,
        seeds_per_shard=1,
        result_count=1,
    )

    estimate = api.RESEARCH._estimate_recipe_v4(
        recipe,
        api.RESEARCH._validate_recipe_v4(recipe, context),
        context,
    )

    assert estimate["launchAdmissible"] is True
    assert estimate["sharding"]["actualShardCount"] == 4
    assert estimate["sharding"]["effectiveShardWorkers"] == 4
    assert estimate["sharding"]["parallelismLimitedByShardPlan"] is False


def test_search_estimate_discloses_shared_campaign_deadline_for_shards(monkeypatch) -> None:
    monkeypatch.setattr(api.RESEARCH, "maximum_local_shard_workers", 2)
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    recipe = _recipe(
        authority.to_dict(),
        "ortools_cp_sat_v3",
        trick_index=6,
        decision_remaining_cards=2,
        decision_seat="south",
        candidate_count=50,
        seeds_per_shard=1,
        solver_timeout_seconds=100.0,
        budget_profile="offline_heavyweight_v1",
    )
    estimate = api.RESEARCH._estimate_recipe_v4(
        recipe,
        api.RESEARCH._validate_recipe_v4(recipe, context),
        context,
    )

    assert estimate["maximumSolverSeconds"] == 100.0
    assert estimate["maximumCampaignSeconds"] == 600.0
    assert estimate["sharding"]["effectiveShardWorkers"] == 2
    assert estimate["sharding"]["maximumShardWallSeconds"] == 600


def test_estimate_enforces_frozen_state_and_model_refusal_thresholds() -> None:
    authority, legacy_context = _frozen_context_and_authority()
    recipe = _recipe(
        authority.to_dict(),
        "ortools_cp_sat_v3",
        trick_index=5,
        decision_remaining_cards=3,
        decision_seat="south",
        decision_position=3,
        quality=True,
    )
    baseline_context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge(quality=True)),
    )
    baseline = api.RESEARCH._estimate_recipe_v4(
        recipe,
        api.RESEARCH._validate_recipe_v4(recipe, baseline_context),
        baseline_context,
    )
    assert baseline["estimatedGraphNodes"] == 286
    mutations = (
        (
            "maximumEstimatedStates",
            baseline["estimatedGraphNodes"] - 1,
            "estimated_states_exceed_refusal_threshold",
        ),
        (
            "maximumModelVariables",
            baseline["estimatedModelVariables"] - 1,
            "model_variables_exceed_refusal_threshold",
        ),
        (
            "maximumModelConstraints",
            baseline["estimatedModelConstraints"] - 1,
            "model_constraints_exceed_refusal_threshold",
        ),
    )
    for field, limit, reason in mutations:
        challenge = _ortools_challenge(quality=True)
        challenge["scenarioResearchV4"]["refusalThresholds"][field] = limit
        context = replace(
            legacy_context,
            bundle=_ChallengeBundleProxy(legacy_context.bundle, challenge),
        )
        estimate = api.RESEARCH._estimate_recipe_v4(
            recipe,
            api.RESEARCH._validate_recipe_v4(recipe, context),
            context,
        )
        assert estimate["launchAdmissible"] is False
        assert estimate["refusalReason"] == reason


def test_service_admits_bid_proof_with_composed_team_escalation_policy() -> None:
    authority, legacy_context = _frozen_context_and_authority()
    challenge = _ortools_challenge()
    challenge["scenarioResearchV4"]["searchBudgets"]["profiles"][
        "interactive_validation_v1"
    ]["maximumWallSeconds"] = 15.0
    ortools_context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, challenge),
    )
    payload = _exact_recipe_payload(ortools_context.bundle_revision_id)
    payload["configuration"] = authority.to_dict()
    prefix = payload["scenario"]["origin"]["exactTracePrefix"]
    prefix["playActions"] = []
    prefix["stateAssertions"] = {}
    prefix["auctionActions"] = []
    prefix["projectActions"] = []
    prefix["projectDeclarations"] = []
    prefix["projectAssertions"] = []
    payload["scenario"]["decisionPoint"] = {
        "kind": "bid",
        "selector": {"actionIndex": 0},
    }
    payload["scenario"]["constraints"] = {
        "predicateId": "PRED-BID-ROUND",
        "kind": "auction.round", "version": 1, "hardness": "hard",
        "arguments": {"round": 1},
    }
    payload["scenario"]["outcome"] = {
        "predicateId": "PRED-BID-ROUND-REPORT",
        "kind": "auction.round", "version": 1, "hardness": "report",
        "arguments": {"round": 1},
    }
    payload["proof"] = {
        "schemaVersion": 1,
        "mode": "policy_conditioned",
        "visibility": "complete_state",
        "strategyClass": "frozen_policy",
        "claim": "outcome_reachable",
        "targetTeam": "north_south",
        "utility": "round_win",
        "successThreshold": 1,
        "lineUniverse": "policy_conditioned",
        "policies": _fixed_policy_templates(),
        "budgetProfile": "late_game_exact_v1",
    }
    recipe = ResearchRecipeV4.build(payload)

    validated = api.RESEARCH._validate_recipe_v4(recipe, ortools_context)
    estimate = api.RESEARCH._estimate_recipe_v4(
        recipe,
        validated,
        ortools_context,
    )

    assert validated["valid"] is True, validated["issues"]
    assert validated["normalizedExactState"]["decisionKind"] == "bid"
    assert validated["normalizedExactState"]["legalActions"]
    assert estimate["launchAdmissible"] is True
    assert estimate["proof"]["admissible"] is True
    assert "reason" not in estimate["proof"]


@pytest.mark.parametrize("decision_kind", ("bid", "escalation"))
def test_service_admits_exact_nonplay_policy_proof(decision_kind: str) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    challenge = _ortools_challenge()
    challenge["scenarioResearchV4"]["searchBudgets"]["profiles"][
        "interactive_validation_v1"
    ]["maximumWallSeconds"] = 15.0
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, challenge),
    )
    payload = _nonplay_policy_recipe_payload(
        context.bundle_revision_id,
        decision_kind,
    )
    payload["configuration"] = authority.to_dict()
    recipe = ResearchRecipeV4.build(payload)

    validated = api.RESEARCH._validate_recipe_v4(recipe, context)
    estimate = api.RESEARCH._estimate_recipe_v4(recipe, validated, context)

    assert validated["valid"] is True, validated["issues"]
    assert validated["normalizedExactState"]["decisionKind"] == decision_kind
    assert validated["normalizedExactState"]["legalActions"]
    assert estimate["launchAdmissible"] is True
    assert estimate["proof"]["admissible"] is True
    assert "reason" not in estimate["proof"]


@pytest.mark.parametrize("decision_kind", ("escalation",))
@pytest.mark.parametrize("mode", ("exact", "bounded"))
def test_service_reports_exact_input_nonplay_complete_graph_evidence_boundary(
    decision_kind: str,
    mode: str,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    payload = _nonplay_complete_graph_recipe_payload(
        context.bundle_revision_id,
        decision_kind,
        mode=mode,
    )
    payload["configuration"] = authority.to_dict()
    recipe = ResearchRecipeV4.build(payload)

    validated = api.RESEARCH._validate_recipe_v4(recipe, context)
    estimate = api.RESEARCH._estimate_recipe_v4(recipe, validated, context)

    assert validated["valid"] is True, validated["issues"]
    assert validated["normalizedExactState"]["decisionKind"] == decision_kind
    assert estimate["launchAdmissible"] is (mode == "bounded")
    assert estimate["proof"]["admissible"] is (mode == "bounded")
    if mode == "exact":
        assert estimate["refusalReason"] == (
            "phase4_substantive_full_service_evidence_missing"
        )


@pytest.mark.parametrize("mode", ("exact", "bounded", "policy_conditioned"))
def test_service_refuses_project_as_a_mandatory_declaration_root(mode: str) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    payload = (
        _nonplay_policy_recipe_payload(context.bundle_revision_id, "project")
        if mode == "policy_conditioned"
        else _nonplay_complete_graph_recipe_payload(
            context.bundle_revision_id,
            "project",
            mode=mode,
        )
    )
    payload["configuration"] = authority.to_dict()
    recipe = ResearchRecipeV4.build(payload)

    validated = api.RESEARCH._validate_recipe_v4(recipe, context)
    estimate = api.RESEARCH._estimate_recipe_v4(recipe, validated, context)

    assert validated["valid"] is True, validated["issues"]
    assert validated["normalizedExactState"]["decisionKind"] == "project"
    assert estimate["launchAdmissible"] is False
    assert estimate["refusalReason"] == (
        "project_declaration_mode_missing_or_mandatory"
    )
    assert estimate["proof"]["reason"] == (
        "project_declaration_mode_missing_or_mandatory"
    )


def test_project_information_estimate_excludes_visible_actor_hand() -> None:
    authority, context = _frozen_context_and_authority()
    payload = _nonplay_complete_graph_recipe_payload(
        context.bundle_revision_id, "project", mode="exact",
    )
    payload["configuration"] = authority.to_dict()
    payload["scenario"]["origin"]["exactTracePrefix"] = _optional_project_prefix()
    payload["proof"]["visibility"] = "actor_information_set"
    payload["proof"]["budgetProfile"] = "actor_information_set_exact_v1"
    payload["resources"]["proofBudgetProfile"] = "actor_information_set_exact_v1"
    recipe = ResearchRecipeV4.build(payload)

    validated = api.RESEARCH._validate_recipe_v4(recipe, context)
    assert validated["valid"], validated["issues"]
    normalized = validated["normalizedExactState"]
    actor = normalized["nextActor"]
    hands = normalized["state"]["hands"]
    assert actor == "north"
    opponent_sizes = [len(cards) for seat, cards in hands.items() if seat != actor]
    unknown = sum(opponent_sizes)
    expected_worlds = 1
    for size in opponent_sizes[:-1]:
        expected_worlds *= math.comb(unknown, size)
        unknown -= size

    estimate = api.RESEARCH._estimate_recipe_v4(recipe, validated, context)
    assert estimate["proof"]["estimatedHiddenWorldUpperBound"] == expected_worlds
    assert estimate["launchAdmissible"] is False
    assert estimate["refusalReason"] == "project_declaration_mode_missing_or_mandatory"


def test_v2_estimate_does_not_load_the_process_default_budget_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, legacy_context = _legacy_context_and_authority()
    ortools_context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_v2_challenge()),
    )
    monkeypatch.setattr(
        ScenarioProofBudgetRegistryV1,
        "load_default",
        classmethod(lambda cls: (_ for _ in ()).throw(AssertionError("fallback"))),
    )
    recipe = _recipe(authority.to_dict(), "ortools_cp_sat_v2")
    validated = api.RESEARCH._validate_recipe_v4(recipe, ortools_context)

    estimate = api.RESEARCH._estimate_recipe_v4(recipe, validated, ortools_context)

    assert estimate["proof"]["budgetMaximumDepth"] == 16
    assert estimate["proof"]["budgetMaximumHiddenWorlds"] == 1


def test_legacy_bundle_keeps_the_existing_validation_and_capability_surface() -> None:
    authority, context = _legacy_context_and_authority()
    recipe = _recipe(authority.to_dict(), "natural_policy")

    assert ResearchFoundryServiceV1._ortools_scenario_backend_v2(context) is None
    legacy_capabilities = api.RESEARCH.capabilities()
    assert "scenarioResearchV4" not in legacy_capabilities["research"]
    api.ResearchDomainCapabilitiesResponse.model_validate(legacy_capabilities)
    validated = api.RESEARCH._validate_recipe_v4(recipe, context)

    legacy_executor = api.RESEARCH._scenario_executor_identity_v4(recipe, context)
    assert legacy_executor == {
        "id": "research_scenario_search_v1",
        "version": "1.0.0",
        "constructionPlanHash": recipe.capability_plan.construction_hash,
        "runtimeVerificationPlanHash": recipe.capability_plan.runtime_verification_hash,
        "proofPlanHash": recipe.capability_plan.proof_plan_hash,
        "sha256": legacy_executor["sha256"],
    }

    assert "legacy_generation_policy_forbidden" not in {
        issue["code"] for issue in validated["issues"]
    }
    service_source = (
        Path(__file__).resolve().parents[1] / "src/baloot_platform/research/service.py"
    ).read_text(encoding="utf-8")
    assert '"id": "research_scenario_search_v1"' in service_source


def test_new_v4_creation_requires_ortools_but_stored_legacy_v4_remains_readable(
    tmp_path: Path,
) -> None:
    authority, context = _legacy_context_and_authority()
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _recipe(authority.to_dict(), "natural_policy").to_dict()
        with pytest.raises(ValueError, match="selecting ortools_cp_sat_v3"):
            service.create_recipe(payload)
        assert service.list_recipes() == []

        stored = service.store.put_recipe(payload)
        readable = service.get_recipe(stored["code"])
        assert readable is not None
        assert readable["sha256"] == stored["sha256"]
        raw = service.store.get_recipe(stored["code"])
        assert raw is not None
        assert service.validate_recipe(raw)["valid"] is True
        recipe, frozen_context = service._build_recipe_and_context(raw)
        assert (
            service._scenario_executor_identity_v4(
                recipe,
                frozen_context,
            )["id"]
            == "research_scenario_search_v1"
        )
    finally:
        executor.shutdown(wait=True)


def test_stored_v3_recipe_read_projects_native_plan_without_rewriting_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, context = _frozen_context_and_authority()
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _recipe(authority.to_dict(), "ortools_cp_sat_v3").to_dict()
        fixture = json.loads(
            (Path(__file__).resolve().parents[1] / "examples/research-scenario-v4-flagship-exact-infeasible.json")
            .read_text(encoding="utf-8")
        )
        play_predicate = next(
            item for item in fixture["scenario"]["constraints"]["children"]
            if item["kind"] == "play.action_matches"
        )
        payload.pop("sha256")
        payload.pop("capabilityPlan")
        payload["scenario"].pop("sha256")
        payload["scenario"]["constraints"].pop("sha256")
        payload["scenario"]["constraints"]["children"] = [
            {key: value for key, value in play_predicate.items() if key != "sha256"}
        ]
        payload = ResearchRecipeV4.build(payload).to_dict()
        assert payload["capabilityPlan"]["valid"] is False
        stored = service.store.put_recipe(payload)
        raw = service.store.get_recipe(stored["code"])
        readable = service.get_recipe(stored["code"], include_native_plan=True)
        assert raw is not None and readable is not None
        assert raw == payload
        assert "nativeCapabilityPlan" not in raw
        assert readable["sha256"] == raw["sha256"]
        assert readable["nativeCapabilityPlan"]["schemaVersion"] == 3
        assert readable["nativeCapabilityPlan"]["rows"][0]["kind"] == (
            "play.action_matches"
        )
        assert readable["nativeCapabilityPlan"]["rows"][0]["nativeStatus"] == (
            "native_exact"
        )
        api.StoredResearchRecipeV4Response.model_validate(readable)
        for listed in (
            service.list_recipes()[0],
            service.list_recipe_page(limit=1)["items"][0],
        ):
            assert listed["sha256"] == raw["sha256"]
            assert "nativeCapabilityPlan" not in listed
            api.StoredResearchRecipeV4Response.model_validate(listed)
        assert service.store.get_recipe(stored["code"]) == raw
        monkeypatch.setattr(
            service,
            "_project_recipe_read_native_v3",
            lambda _item: pytest.fail("historical API reads must not rebuild native plans"),
        )
        monkeypatch.setattr(api, "RESEARCH", service)
        with TestClient(api.app) as client:
            response = client.get(f"/api/v2/research-recipes/{stored['code']}")
        assert response.status_code == 200, response.text
        assert response.json()["sha256"] == raw["sha256"]
        assert "nativeCapabilityPlan" not in response.json()
    finally:
        executor.shutdown(wait=True)


def test_created_v3_recipe_receipt_stays_frozen_and_read_projects_native_plan(
    tmp_path: Path,
) -> None:
    authority, context = _frozen_context_and_authority()
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _recipe(
            authority.to_dict(), "ortools_cp_sat_v3", trick_index=6
        ).to_dict()
        created = service.create_recipe(payload, idempotency_key="native-plan-create")
        replayed = service.create_recipe(payload, idempotency_key="native-plan-create")
        raw = service.store.get_recipe(created["code"])
        readable = service.get_recipe(created["code"], include_native_plan=True)
        assert raw is not None
        assert readable is not None
        assert created == replayed
        assert created["sha256"] == raw["sha256"]
        assert "nativeCapabilityPlan" not in raw
        assert "nativeCapabilityPlan" not in created
        assert readable["nativeCapabilityPlan"]["schemaVersion"] == 3
        api.StoredResearchRecipeV4Response.model_validate(readable)
    finally:
        executor.shutdown(wait=True)


def test_v4_cannot_start_a_legacy_dual_backend_comparison_campaign(
    tmp_path: Path,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        recipe = _recipe(authority.to_dict(), "ortools_cp_sat_v3").to_dict()
        stored = service.store.put_recipe(recipe)

        with pytest.raises(ValueError, match="one OR-Tools qualification authority"):
            service.create_comparison_campaign(
                stored["code"],
                idempotency_key="v4-comparison-forbidden",
            )

        assert service.list_comparison_campaigns() == []
        assert service.list_runs() == []
    finally:
        executor.shutdown(wait=True)


@pytest.mark.real_clock
@pytest.mark.parametrize("cancelled_before_execution,cancel_during_proof", ((False, False), (True, False), (True, True)))
def test_incomplete_campaign_audits_incomplete_execution_not_later_empty_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled_before_execution: bool, cancel_during_proof: bool,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    backend = service._cached_ortools_scenario_backend_v2(context)
    assert backend is not None
    try:
        recipe = service.create_recipe(
            _recipe(authority.to_dict(), "ortools_cp_sat_v3", trick_index=6).to_dict(),
            idempotency_key="incomplete-before-empty-create",
        )
        args = dict(
            manifest_hash=backend.identity.predicate_catalog_hash,
            rules=context.bundle.rules,
            bundle_revision_id=context.bundle_revision_id,
            bundle_hash=context.bundle.sha256,
            scenario_hash=recipe["scenario"]["sha256"],
        )
        # Since b9516073, incomplete runs without a candidate or counterstrategy_budget reason become not_found_within_budget.
        incomplete = _terminal_search_execution(
            "incomplete", candidate_attempt_reason="counterstrategy_budget", **args)
        empty = _terminal_search_execution("not_found", **args)
        campaign = OrToolsResearchCampaignResultV1(
            "incomplete", () if cancelled_before_execution else (incomplete, empty), (), (), (), 1, 2,
            "rules_preserving_v1", "decision_state_card_hamming_v1", 1, 0.1, 0.1,
        )
        def campaign_with_solver_progress(*_args, **kwargs):
            assert kwargs["shard_worker_count"] == 1
            kwargs["progress_callback"]({
                "currentStage": "tailConstruction",
                "stageCounters": {
                    "tailCandidates": 24,
                    "proofBoundaryAttempts": 24,
                    "prefixLiftAttempts": 0,
                    "ownerOfCard": ["north"] * 32,
                },
            })
            assert service.get_run(run["runCode"])["campaignProgress"]["currentSolverCounters"] == {
                "tailCandidates": 24,
                "proofBoundaryAttempts": 24,
                "prefixLiftAttempts": 0,
            }
            # Repeated progress is one candidate, even if cancellation prevents
            # the worker from returning its completed execution document.
            for queries in (477, 478):
                kwargs["progress_callback"]({
                    "currentStage": "playingQuality", "candidateHash": "b" * 64,
                    "shardId": "SHARD-0", "attempt": 1, "completedCandidates": 1,
                    "qualitySolver": {"kind": "quality_solver_progress_v1", "phase": "counterexample",
                        "decisionKind": "bid", "proofPhase": "fallback", "masterCount": 42,
                        "bindingCount": 413, "queryCount": queries, "hands": ["private"]},
                })
            if cancelled_before_execution:
                if cancel_during_proof:
                    service.cancel(run["runCode"])
                kwargs["progress_callback"]({
                    "currentStage": "candidate_terminal", "candidateStatus": "incomplete",
                    "candidateHash": "b" * 64, "shardId": "SHARD-0", "attempt": 1,
                    "completedCandidates": 1, "executionOrdinal": 1,
                    "candidateAttemptReason": "maximum_seconds",
                })
            return campaign

        monkeypatch.setattr(backend, "execute_campaign", campaign_with_solver_progress)
        run = service._submit_recipe_run(
            recipe["code"], idempotency_key="incomplete-before-empty-run", schedule=False,
        )
        service._execute(run["runCode"])
        completed = service.get_run(run["runCode"])
        stored = service.store.get_run(run["runCode"])
        if cancel_during_proof:
            assert completed["status"] == "cancelled"
            assert completed["campaignProgress"]["state"] == "cancelled"
            assert completed["campaignProgress"]["proofIncomplete"] == 1
            assert completed["campaignProgress"]["candidateReceiptCounts"]["incomplete"] == 1
            assert completed["campaignProgress"]["qualified"] == 0
            return
        assert any(
            event.get("accounting", {}).get("currentSolverCounters") == {
                "tailCandidates": 24,
                "proofBoundaryAttempts": 24,
                "prefixLiftAttempts": 0,
            }
            for event in stored["events"]
        )
        assert completed["scenarioResearchStatus"] == "proof_incomplete"
        accounting = completed["campaignAccounting"]
        assert accounting["candidateAttempts"] == accounting["constructedCandidates"] == 1
        assert completed["attemptedCandidates"] == completed["scenarioYieldCounts"]["attempted"] == 1
        assert completed["scenarioYieldCounts"]["structurallyFeasible"] == 1
        assert accounting["proofIncomplete"] == 1
        assert accounting["currentStage"] == ("candidate_terminal" if cancelled_before_execution else "playingQuality")
        assert accounting["qualitySolver"] == (None if cancelled_before_execution else {
            "phase": "counterexample", "masterCount": 42, "queryCount": 478,
            "bindingCount": 413, "decisionKind": "bid", "proofPhase": "fallback"})
        assert accounting["currentSolverCounters"] == {
            "tailCandidates": 24, "proofBoundaryAttempts": 24, "prefixLiftAttempts": 0}
        assert completed["qualifiedCount"] == 0
        if cancelled_before_execution:
            assert accounting["latestCandidateHash"] == "b" * 64
            assert completed["auditResultCodes"] == []
            assert completed["performanceReceipt"]["completedCandidates"] == 0
            return
        assert len(completed["auditResultCodes"]) == 1
        result = service.get_result(completed["auditResultCodes"][0])
        assert result["terminalStatus"] == "proof_incomplete"
        assert result["disposition"] == "rejected"
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize("terminal_status", ["proof_failed", "semantic_postcondition_miss"])
def test_campaign_audits_matching_failure_not_later_untested_candidate(
    tmp_path, monkeypatch, terminal_status,
):
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()))
    service, executor = _isolated_service(tmp_path, authority, context)
    backend = service._cached_ortools_scenario_backend_v2(context)
    try:
        recipe = service.create_recipe(
            _recipe(authority.to_dict(), "ortools_cp_sat_v3", trick_index=6).to_dict(),
            idempotency_key="mixed-terminal-create")
        args = dict(manifest_hash=backend.identity.predicate_catalog_hash,
            rules=context.bundle.rules, bundle_revision_id=context.bundle_revision_id,
            bundle_hash=context.bundle.sha256, scenario_hash=recipe["scenario"]["sha256"])
        failed = _terminal_search_execution(terminal_status, **args)
        untested = _terminal_search_execution("not_found", **args)
        campaign = OrToolsResearchCampaignResultV1(terminal_status, (failed, untested), (), (), (), 1, 2,
            "rules_preserving_v1", "decision_state_card_hamming_v1", 1, 0.1, 0.1)
        monkeypatch.setattr(backend, "execute_campaign", lambda *a, **kw: campaign)
        run = service._submit_recipe_run(recipe["code"], idempotency_key="mixed-terminal-run", schedule=False)
        service._execute(run["runCode"])
        completed = service.get_run(run["runCode"])
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == terminal_status
        assert completed["qualifiedCount"] == 0 and len(completed["auditResultCodes"]) == 1
        result = service.get_result(completed["auditResultCodes"][0])
        assert result["terminalStatus"] == terminal_status
        assert result["researchAuthority"] == failed.research_authority
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize(
    ("origin", "expected"),
    (("search", "legacy_search"), ("exact_trace_prefix", "legacy_exact")),
)
def test_stored_legacy_v4_run_dispatches_to_its_frozen_legacy_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    origin: str,
    expected: str,
) -> None:
    authority, context = _legacy_context_and_authority()
    service, executor = _isolated_service(tmp_path, authority, context)
    calls: list[str] = []
    try:
        payload = (
            _exact_recipe_payload(context.bundle_revision_id)
            if origin == "exact_trace_prefix"
            else _recipe(
                authority.to_dict(),
                "natural_policy",
                trick_index=6,
            ).to_dict()
        )
        payload = service._build_recipe_and_context(payload)[0].to_dict()
        stored = service.store.put_recipe(payload)
        run = service._submit_recipe_run(
            stored["code"],
            idempotency_key="stored-legacy-v4-run",
            schedule=False,
        )
        monkeypatch.setattr(
            service,
            "_execute_v4_search_run",
            lambda *_args: calls.append("legacy_search"),
        )
        monkeypatch.setattr(
            service,
            "_execute_v4_exact_run",
            lambda *_args: calls.append("legacy_exact"),
        )
        monkeypatch.setattr(
            service,
            "_execute_v4_ortools_run",
            lambda *_args: calls.append("ortools"),
        )

        service._execute(run["runCode"])

        assert calls == [expected]
    finally:
        executor.shutdown(wait=True)


def test_ortools_service_methods_cannot_call_legacy_research_authorities() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "src/baloot_platform/research/service.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    service = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ResearchFoundryServiceV1"
    )
    protected_methods = {
        "_ortools_scenario_backend_v2",
        "preview_exact_trace_legal_actions",
        "_validate_recipe_v4_ortools",
        "_estimate_recipe_v4_ortools",
        "_ortools_execution_parameters_v2",
        "_execute_v4_ortools_run",
        "create_recipe",
    }
    methods = {
        node.name: node
        for node in service.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in protected_methods
    }
    forbidden_names = {
        "_execute_v4_search_run",
        "_execute_v4_exact_run",
        "ExactStateNormalizerV1",
        "ScenarioSearchExecutorV1",
        "ScenarioRuntimeVerifierV1",
        "UnifiedScenarioProofServiceV1",
        "run_sequential_candidate_search_v1",
        "BalootEngine",
        "HeuristicBot",
        "BotProfile",
    }
    forbidden_attributes = {
        "self.simulator",
        "context.simulator",
        "context.legacy_domain_v2",
        "self.deal_generator",
        "self.tactical_verifier",
        "self.analyzers",
    }

    def attribute_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            owner = attribute_name(node.value)
            return f"{owner}.{node.attr}" if owner else node.attr
        return ""

    assert set(methods) == protected_methods
    calls = {
        name: {
            attribute_name(call.func)
            for call in ast.walk(method)
            if isinstance(call, ast.Call)
        }
        for name, method in methods.items()
    }
    assert "backend.execute_campaign" in calls["_execute_v4_ortools_run"]
    campaign_call = next(
        call
        for call in ast.walk(methods["_execute_v4_ortools_run"])
        if isinstance(call, ast.Call)
        and attribute_name(call.func) == "backend.execute_campaign"
    )
    assert {keyword.arg for keyword in campaign_call.keywords} >= {
        "maximum_prefix_lift_attempts"
    }
    assert {
        method: sorted(
            name
            for name in names
            if name.rsplit(".", 1)[-1] in forbidden_names
            or any(
                name == forbidden or name.startswith(f"{forbidden}.")
                for forbidden in forbidden_attributes
            )
        )
        for method, names in calls.items()
        if any(
            name.rsplit(".", 1)[-1] in forbidden_names
            or any(
                name == forbidden or name.startswith(f"{forbidden}.")
                for forbidden in forbidden_attributes
            )
            for name in names
        )
    } == {}


@pytest.mark.parametrize("origin", ("search", "exact_trace_prefix"))
def test_ortools_run_dispatch_never_enters_legacy_v4_executors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    origin: str,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    calls: list[str] = []
    try:
        payload = (
            _exact_recipe_payload(context.bundle_revision_id)
            if origin == "exact_trace_prefix"
            else _recipe(
                authority.to_dict(),
                "ortools_cp_sat_v3",
                trick_index=6,
            ).to_dict()
        )
        recipe = service.create_recipe(payload)
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key=f"dispatch-{origin}",
            schedule=False,
        )
        monkeypatch.setattr(
            service,
            "_execute_v4_ortools_run",
            lambda *_args: calls.append("ortools"),
        )
        monkeypatch.setattr(
            service,
            "_execute_v4_exact_run",
            lambda *_args: calls.append("legacy_exact"),
        )
        monkeypatch.setattr(
            service,
            "_execute_v4_search_run",
            lambda *_args: calls.append("legacy_search"),
        )
        monkeypatch.setattr(
            service,
            "_shard_worker_loop",
            lambda *_args: calls.append("legacy_shard"),
        )

        service._execute(run["runCode"])

        assert calls == ["ortools"]
    finally:
        executor.shutdown(wait=True)


def test_historical_interest_summary_is_not_reinterpreted(monkeypatch):
    from baloot_platform.research import result_authority

    historical = {"qualification": {"interestAdmission": {"evidenceHash": "a" * 64}},
                  "provenance": {"resultIdentity": {"authorityArtifactHashes": {
                      "configurationSnapshot": "b" * 64, "loweringManifest": "c" * 64}}}}
    original = copy.deepcopy(historical)
    def forbidden_gate(*args, **kwargs):
        pytest.fail("Historical summary must not be reinterpreted by the current gate")
    monkeypatch.setattr(result_authority, "evaluate_round_interestingness_v1", forbidden_gate)
    result_authority._validate_native_interest_admission_v1(
        historical, {}, {}, verified_quality=True)
    assert historical == original


def test_diversity_rejection_response_accepts_interest_gate_only_known_reason():
    response = api.ResearchRunDiversityRejectionResponse.model_validate(
        {"resultCode": "LAF-0123456789", "reason": "interest_gate_rejected"})
    assert response.reason == "interest_gate_rejected"
    with pytest.raises(ValidationError):
        api.ResearchRunDiversityRejectionResponse.model_validate(
            {"resultCode": "LAF-0123456789", "reason": "unknown_reason"})


@pytest.mark.parametrize("version,enabled", [("3.1.54", True), ("3.1.55", True), ("3.1.55", False), ("invalid", True)])
def test_interest_artifact_requirement_is_producer_and_policy_bound(version, enabled):
    from baloot_platform.research import result_authority

    stripped = {"researchAuthority": {"backendId": "ortools_cp_sat_v3", "backendVersion": version},
                "provenance": {"resultIdentity": {"authorityArtifactHashes": {}}}}
    snapshot = {"components": {"challenge": {"payload": {"scenarioResearchV4": {
        "qualityExecution": {"videoInterestingness": {"schemaVersion": 4 if enabled else 1}}}}}}}
    if version in {"3.1.55", "invalid"} and enabled:
        with pytest.raises(ValueError, match="interest evidence artifact is missing"):
            result_authority._validate_native_interest_admission_v1(stripped, {}, snapshot, verified_quality=True)
    else:
        result_authority._validate_native_interest_admission_v1(stripped, {}, snapshot, verified_quality=True)


@pytest.mark.parametrize("retain_interest", [False, True])
def test_exact_v4_run_persists_one_owned_qualified_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retain_interest: bool,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    challenge = _policy_service_challenge()
    if retain_interest:
        calibration = json.loads((Path(__file__).parent / "fixtures/actor_interest_calibration.json").read_text())
        challenge["scenarioResearchV4"]["qualityExecution"]["videoInterestingness"] = calibration["cases"]["7189"]["policy"]
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, challenge),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _native_qualified_exact_recipe_payload(context.bundle_revision_id)
        payload["proof"]["claim"] = "winning_continuation_line_count"
        payload["proof"].pop("selectedAction", None)
        recipe = service.create_recipe(
            payload,
            idempotency_key="ortools-exact-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key="ortools-exact-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None
        api.ResearchRunResponse.model_validate(completed)
        assert completed["status"] == "completed", json.dumps(
            completed.get("events", []), ensure_ascii=False
        )
        assert completed["scenarioResearchStatus"] == "satisfied", json.dumps(
            {
                "scenarioResearchStatus": completed["scenarioResearchStatus"],
                "accounting": {key: completed.get("campaignAccounting", {}).get(key)
                    for key in ("currentStage", "stopReason", "candidateReceiptCounts", "latestIncompleteCompilation")},
                "performance": {key: completed.get("performanceReceipt", {}).get(key)
                    for key in ("status", "cpuSeconds", "wallTimeSeconds")},
                "scenarioYieldCounts": completed.get("scenarioYieldCounts"),
            },
            ensure_ascii=False,
        )
        assert completed["resultCount"] == completed["qualifiedCount"] == 1
        assert completed["scenarioYieldCounts"]["qualified"] == 1
        assert completed["performanceReceipt"]["completedCandidates"] == 1
        assert len(completed["resultCodes"]) == 1

        result = service.get_result(completed["resultCodes"][0])
        assert result is not None
        assert result["disposition"] == "qualified"
        assert result["runtimeCompatibility"]["status"] == "passed", result[
            "runtimeCompatibilityEvidence"
        ]
        assert result["researchAuthority"]["qualificationAuthority"] == "cp_sat"
        continuation = result["proofOutcome"]["continuationLines"]
        assert continuation["complete"] is True
        assert continuation["exactCount"].isdigit()
        boundary = completed["scenarioResearchEvidence"]["proofBoundaries"][0]
        assert boundary["completionStatus"] == "exact_complete"
        assert boundary["continuationLines"]["exactCount"] == continuation["exactCount"]
        assert boundary["budget"]["maximumLineCountNodes"] == 2_000_000
        assert boundary["budget"]["maximumLineCountLimbs"] == 128
        assert boundary["sha256"] == canonical_sha256(
            {key: value for key, value in boundary.items() if key != "sha256"}
        )
        assert set(result["artifacts"]) == {
            "constructionModel",
            "proofModel",
            "proofGraph",
            "completeTrace",
            "proofOutcome",
        }
        assert result["provenance"]["resultIdentity"]["kind"] == (
            "ortools_research_result_identity_v3"
        )
        assert set(result["authorityArtifacts"]) == {
            "configurationSnapshot",
            "loweringManifest",
            "resultIdentity",
            "executionReceipt",
        } | ({"interestAdmission"} if "interestAdmission" in result["qualification"] else set())
        if "interestAdmission" in result["qualification"]:
            from baloot_platform.research.result_authority import validate_native_interest_artifact_v1

            assert result["qualification"]["qualified"] is True
            assert result["qualification"]["interestAdmission"]["passed"] is False
            assert completed["diversitySelection"]["selectedCount"] == 0
            assert completed["diversitySelection"]["rejected"] == [
                {"resultCode": result["resultCode"], "reason": "interest_gate_rejected"}]
            reference = result["authorityArtifacts"]["interestAdmission"]
            evidence = json.loads(service.artifact_store.get(reference["uri"]))
            assert evidence["sha256"] == result["qualification"]["interestAdmission"]["evidenceHash"]
            assert evidence["sha256"] == canonical_sha256(
                {key: value for key, value in evidence.items() if key != "sha256"})
            assert canonical_sha256(evidence) == reference["sha256"]
            assert result["provenance"]["resultIdentity"]["authorityArtifactHashes"]["interestAdmission"] == reference["sha256"]
            trace = json.loads(service.artifact_store.get(result["artifacts"]["completeTrace"]["uri"]))
            snapshot = json.loads(service.artifact_store.get(result["authorityArtifacts"]["configurationSnapshot"]["uri"]))
            validate_native_interest_artifact_v1(result, trace, snapshot, evidence)
            changed = {**evidence, "featuredExactStateId": "STATE-SUBSTITUTED"}
            changed["sha256"] = canonical_sha256({key: value for key, value in changed.items() if key != "sha256"})
            with pytest.raises(RuntimeError, match="interest evidence binding mismatch"):
                validate_native_interest_artifact_v1(result, trace, snapshot, changed)
            for mutation in ("missing", "owner", "cas"):
                altered = copy.deepcopy(result)
                if mutation == "missing":
                    del altered["authorityArtifacts"]["interestAdmission"]
                elif mutation == "owner":
                    altered["authorityArtifacts"]["interestAdmission"]["ownerHash"] = "0" * 64
                else:
                    substituted = service.artifact_store.put(
                        json.dumps(changed, sort_keys=True, separators=(",", ":")).encode(),
                        media_type=reference["mediaType"], name="substituted.json")
                    altered["authorityArtifacts"]["interestAdmission"].update(substituted)
                with pytest.raises(RuntimeError, match="authority artifact set mismatch|interestAdmission authority substitution"):
                    service._validate_stored_scenario_result_v4(altered)
            get_artifact = service.artifact_store.get
            def missing_interest(uri):
                if uri == reference["uri"]:
                    raise FileNotFoundError(uri)
                return get_artifact(uri)
            with monkeypatch.context() as patch:
                patch.setattr(service.artifact_store, "get", missing_interest)
                with pytest.raises((FileNotFoundError, RuntimeError)):
                    service.get_result(result["resultCode"])
        assert ("interestAdmission" in result["authorityArtifacts"]) is retain_interest
        assert {
            reference["ownerHash"]
            for reference in result["authorityArtifacts"].values()
        } == {result["resultHash"]}
        semantic_owner = result["provenance"]["resultIdentity"]["executorSemanticHash"]
        assert {
            reference["ownerKind"] for reference in result["artifacts"].values()
        } == {"ortools_research_semantic_artifact_v1"}
        assert {
            reference["ownerHash"] for reference in result["artifacts"].values()
        } == {semantic_owner}
        assert result["scenarioProjection"]["exactStateId"].startswith("STATE-")

        def reject_current_execution_context(_value: dict) -> None:
            raise AssertionError(
                "immutable V3 result reads must not resolve current code"
            )

        monkeypatch.setattr(
            service,
            "_build_recipe_and_context",
            reject_current_execution_context,
        )
        assert service.get_result(result["resultCode"]) == result
        acting_seat = payload["scenario"]["decisionPoint"]["selector"]["actingSeat"]
        observer = service.get_exact_state(result["resultCode"], observer_seat=acting_seat)
        assert observer["observerSeat"] == acting_seat

        root = Path(__file__).resolve().parents[1]
        video_profile = yaml.safe_load(
            (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
        )

        def render_service(value: dict) -> ResearchRenderPreparationServiceV1:
            research = SimpleNamespace(
                get_result=lambda _code: value,
                get_recipe=lambda _code: service.get_recipe(recipe["code"]),
                get_playing_quality_audit=service.get_playing_quality_audit,
            )
            return ResearchRenderPreparationServiceV1(
                research=research,
                artifact_store=service.artifact_store,
                publication_service=SimpleNamespace(),
                renderer=RemotionProfessionalRenderer(
                    video_profile,
                    repository_root=root,
                ),
                rules=context.bundle.rules,
                configuration_store=api.CONTAINER.configuration_store,
            )

        if retain_interest:
            with pytest.raises(ValueError, match="pre-render interestingness screen rejected"):
                ResearchRenderPreparationServiceV1._require_interest_admission(
                    evidence["policy"], result["disposition"], evidence["blockers"])
            return
        prepared = render_service(result).prepare(result["resultCode"])
        import baloot_platform.research.render_service as render_module

        validate_result = render_module.validate_ortools_result_sections_v1
        for key in ("exactStateId", "stateHash"):
            def conflicting_proof(*args, **kwargs):
                validated = copy.deepcopy(validate_result(*args, **kwargs))
                validated["proofOutcome"]["identities"][key] = "wrong-state"
                return validated

            with monkeypatch.context() as patch:
                patch.setattr(render_module, "validate_ortools_result_sections_v1", conflicting_proof)
                with pytest.raises(ValueError, match="proof references a different exact state"):
                    render_service(result).prepare(result["resultCode"])
        assert prepared["publicationEligible"] is False
        assert prepared["publicationBlockers"] == [
            "critical_decision_not_proven",
        ]
        assert prepared["renderPurpose"] == "research_audit"
        assert prepared["source"]["auditOnly"] is True
        assert set(prepared["proofArtifact"]) == {
            "uri",
            "sha256",
            "bytes",
            "mediaType",
            "name",
        }
        batch = RenderBatchServiceV1(
            jobs=None,
            background_executor=None,
            batch_executor=None,
            artifact_store=service.artifact_store,
            output_root=tmp_path / "render-output",
            maximum_attempts=1,
        )
        bundle = batch._load_bundle(prepared["bundleArtifact"])
        assert (
            batch._validated_proof_artifact(
                prepared["proofArtifact"],
                bundle,
                authority_receipt=prepared["authorityReceipt"],
            )
            == prepared["proofArtifact"]["sha256"]
        )

        for status in ("not_checked", "failed"):
            audit_result = copy.deepcopy(result)
            evidence = None
            failure_codes: tuple[str, ...] = ()
            engine_version = None
            evidence_hash = None
            if status == "failed":
                failure_codes = ("forced_test_failure",)
                engine_version = "test-engine"
                evidence_core = {
                    "schemaVersion": 1,
                    "semanticResultHash": result["resultHash"],
                    "bundleHash": result["researchAuthority"]["bundleHash"],
                    "traceHash": result["researchAuthority"]["traceHash"],
                    "failureCodes": list(failure_codes),
                }
                evidence = {
                    **evidence_core,
                    "sha256": canonical_sha256(evidence_core),
                }
                evidence_hash = evidence["sha256"]
            audit_result["runtimeCompatibility"] = RuntimeCompatibilityV1.build(
                status=status,
                bundle_hash=result["researchAuthority"]["bundleHash"],
                trace_hash=result["researchAuthority"]["traceHash"],
                engine_version=engine_version,
                evidence_hash=evidence_hash,
                failure_codes=failure_codes,
            ).to_dict()
            audit_result["runtimeCompatibilityEvidence"] = evidence

            with pytest.raises(
                ValueError,
                match="runtime compatibility must pass before playable or video admission",
            ):
                render_service(audit_result).prepare(result["resultCode"])

        substituted = copy.deepcopy(result)
        substituted["artifacts"]["completeTrace"]["ownerHash"] = "0" * 64
        with pytest.raises(RuntimeError, match="completeTrace substitution detected"):
            render_service(substituted).prepare(result["resultCode"])

        for authority_name in (
            "configurationSnapshot",
            "loweringManifest",
            "resultIdentity",
            "executionReceipt",
        ):
            substituted = copy.deepcopy(result)
            substituted["authorityArtifacts"][authority_name]["ownerHash"] = "0" * 64
            with pytest.raises(
                RuntimeError,
                match=f"{authority_name} authority substitution detected",
            ):
                render_service(substituted).prepare(result["resultCode"])
    finally:
        executor.shutdown(wait=True)


def test_exact_bounded_play_run_persists_honest_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _native_qualified_exact_recipe_payload(context.bundle_revision_id)
        payload["proof"] = {
            "schemaVersion": 1,
            "mode": "bounded",
            "visibility": "complete_state",
            "strategyClass": "pure_deterministic",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "round_win",
            "successThreshold": -1,
            "horizon": {"unit": "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1",
            "budgetProfile": "late_game_exact_v1",
        }
        recipe = service.create_recipe(
            payload,
            idempotency_key="ortools-exact-play-bounded-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key="ortools-exact-play-bounded-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == "satisfied", json.dumps(
            completed, ensure_ascii=True
        )
        result = service.get_result(completed["resultCodes"][0])
        assert result["proofOutcome"]["completionStatus"] == "bounded_complete"
        assert result["proofOutcome"]["boundaryEvaluator"] == (
            "current_raw_point_margin_v1"
        )
        assert result["runtimeCompatibility"]["status"] == "passed"
        assert result["disposition"] == "qualified"
        proof = result["proofOutcome"]
        assert proof["requestedMode"] == proof["achievedMode"] == "bounded"
        assert proof["horizon"] == payload["proof"]["horizon"]
        graph = json.loads(service.artifact_store.get(result["artifacts"]["proofGraph"]["uri"]))
        assert graph["scope"] == "bounded"

        cold_service, cold_executor = _isolated_service(tmp_path, authority, context)
        try:
            assert not cold_service._result_verification_cache

            def reject_current_execution_context(*_args, **_kwargs):
                pytest.fail("cold bounded reads must use the stored configuration snapshot")

            monkeypatch.setattr(cold_service, "_build_recipe_and_context", reject_current_execution_context)
            restored = cold_service.get_result(result["resultCode"])
            assert restored["resultHash"] == result["resultHash"]
            assert restored["proofOutcome"] == proof
            assert restored["artifacts"] == result["artifacts"]
            public = api._public_research_result(restored)
            api.ResearchScenarioPublicResultResponse.model_validate(public)
            assert public["proofOutcome"]["achievedMode"] == "bounded"
            assert public["proofOutcome"]["completionStatus"] == "bounded_complete"
            assert public["proofOutcome"]["horizon"] == proof["horizon"]
            assert public["proofOutcome"]["boundaryEvaluator"] == proof["boundaryEvaluator"]
        finally:
            cold_executor.shutdown(wait=True)
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize("decision_kind", ("bid", "escalation"))
def test_exact_nonplay_policy_run_persists_complete_selected_continuation(
    tmp_path: Path,
    decision_kind: str,
) -> None:
    # Alwaleed-v2 measured p99 is 67.096s across both parameters; 3x margin rounded to 240s.
    measured_wall_seconds = 240.0
    authority, legacy_context = _frozen_context_and_authority()
    challenge = _apply_measured_nonplay_limits(_ortools_challenge(), measured_wall_seconds)
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, challenge),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _nonplay_policy_recipe_payload(
            context.bundle_revision_id,
            decision_kind,
        )
        payload["resources"]["solverTimeoutSeconds"] = measured_wall_seconds
        recipe = service.create_recipe(
            payload,
            idempotency_key=f"ortools-{decision_kind}-policy-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key=f"ortools-{decision_kind}-policy-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None
        assert completed["status"] == "completed", json.dumps(
            completed.get("events", []), ensure_ascii=True
        )
        assert completed["scenarioResearchStatus"] == "satisfied", json.dumps(
            completed.get("events", []), ensure_ascii=True
        )
        assert completed["resultCount"] == completed["qualifiedCount"] == 1
        result = service.get_result(completed["resultCodes"][0])
        assert result is not None
        assert result["proofOutcome"]["completionStatus"] == "policy_complete"
        trace = json.loads(
            service.artifact_store.get(result["artifacts"]["completeTrace"]["uri"])
        )
        quality = trace["playingQualityEvidence"]
        assert quality["receipt"]["status"] == "complete"
        assert trace["completeExactTracePrefix"] == quality["receipt"]["finalPrefix"]
        complete_normalization = trace["completeExactNormalization"]
        assert complete_normalization["phase"] == "scored"
        assert len(complete_normalization["state"]["completedTricks"]) == 8
        assert result["disposition"] == "qualified"
        assert result["runtimeCompatibility"]["status"] == "passed", result[
            "runtimeCompatibility"
        ]
        assert result["runtimeCompatibility"]["failureCodes"] == []
        exact_actions = {
            str(item["code"]) for item in trace["exactNormalization"]["legalActions"]
        }
        if decision_kind == "bid":
            count_evidence = next(
                row for row in result["predicateEvaluation"]["evidence"]
                if row["predicateId"] == "PRED-NONPLAY-BID-LEGAL-COUNT"
            )
            assert count_evidence["observed"]["legalActionCount"] == len(exact_actions)
        if decision_kind == "escalation":
            count_evidence = next(
                row for row in result["predicateEvaluation"]["evidence"]
                if row["predicateId"] == "PRED-ESCALATION-LEGAL-COUNT"
            )
            assert count_evidence["observed"]["legalActionCount"] == len(exact_actions)
        proof_actions = {
            str(item["action"]) for item in result["proofOutcome"]["rootActions"]
        }
        assert proof_actions == exact_actions, (proof_actions, exact_actions)
        compatibility = result["runtimeCompatibilityEvidence"]
        assert compatibility["replayStatus"] == "completed"
        assert (
            compatibility["expectedTraceProjectionHash"]
            == compatibility["derivedTraceProjectionHash"]
        )
        root = Path(__file__).resolve().parents[1]
        video_profile = yaml.safe_load(
            (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
        )
        render_service = ResearchRenderPreparationServiceV1(
            research=SimpleNamespace(
                get_result=lambda _code: result,
                get_recipe=lambda _code: service.get_recipe(recipe["code"]),
                get_playing_quality_audit=service.get_playing_quality_audit,
            ),
            artifact_store=service.artifact_store,
            publication_service=SimpleNamespace(),
            renderer=RemotionProfessionalRenderer(
                video_profile,
                repository_root=root,
            ),
            rules=context.bundle.rules,
            configuration_store=api.CONTAINER.configuration_store,
        )
        prepared = render_service.prepare(result["resultCode"])
        context_scene = next(
            scene
            for scene in prepared["preview"]["scenes"]
            if scene["stage"] == "continuation"
        )
        lifecycle = next(
            layer["content"]
            for layer in context_scene["layers"]
            if layer["id"] == "public-lifecycle"
        )
        assert len(lifecycle["plays"]) == 32
        assert len(lifecycle["completedTricks"]) == 8
        assert lifecycle["roundScore"]["awardedPoints"]
        assert lifecycle["pauseBeforePlayIndex"] == 0
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize(
    ("decision_kind", "wrong_actor", "wrong_fact"),
    (
        ("bid", "west", "actor"),
        ("escalation", "east", "actor"),
        ("project", "west", "actor"),
        ("project", "west", "count"),
        ("project", "west", "match"),
        ("bid", "west", "raw"),
        ("bid", "west", "game"),
        ("escalation", "east", "game"),
        ("project", "west", "project_points"),
        ("bid", "west", "visibility"),
        ("escalation", "east", "visibility"),
        ("project", "west", "visibility"),
    ),
)
def test_native_nonplay_state_rejects_false_fact(
    tmp_path: Path, decision_kind: str, wrong_actor: str, wrong_fact: str,
    inline_v3_shard_runner: None,
) -> None:
    if decision_kind == "project":
        authority, context = _optional_project_context_and_authority(tmp_path)
    else:
        authority, legacy_context = _frozen_context_and_authority()
        context = replace(
            legacy_context,
            bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
        )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _nonplay_policy_recipe_payload(
            context.bundle_revision_id, decision_kind,
        )
        if decision_kind == "project":
            payload["scenario"]["origin"]["exactTracePrefix"] = _optional_project_prefix()
        if wrong_fact == "count":
            payload["scenario"]["constraints"]["children"][3]["arguments"] = {
                "minimum": 0, "maximum": 0,
            }
        elif wrong_fact == "match":
            payload["scenario"]["constraints"]["children"][4]["arguments"] = {
                "team": "north_south", "minimum": 1, "maximum": 1,
            }
        elif wrong_fact in {"raw", "game", "project_points"}:
            kind = {
                "raw": "state.current_raw_points",
                "game": "state.current_game_points",
                "project_points": "state.current_project_points",
            }[wrong_fact]
            payload["scenario"]["constraints"]["children"].append({
                "predicateId": f"PRED-FALSE-{wrong_fact.upper()}",
                "kind": kind, "version": 1, "hardness": "hard",
                "arguments": {
                    "team": "north_south", "minimum": 1, "maximum": 1,
                },
            })
        elif wrong_fact == "visibility":
            purchase_card = payload["scenario"]["origin"]["exactTracePrefix"]["deckOrder"][20]
            payload["scenario"]["constraints"]["children"].append({
                "predicateId": "PRED-FALSE-PURCHASE-VISIBILITY",
                "kind": "state.observer_card_visibility", "version": 1,
                "hardness": "hard",
                "arguments": {
                    "observer": "west", "card": purchase_card,
                    "visibility": "hidden",
                },
            })
        else:
            payload["scenario"]["constraints"]["children"][2]["arguments"]["seat"] = wrong_actor
        if decision_kind == "project":
            with pytest.raises(ValueError, match="mandatory project declarations"):
                service.create_recipe(payload)
            recipe = service.store.put_recipe(service._build_recipe_and_context(payload)[0].to_dict())
        else:
            recipe = service.create_recipe(
                payload, idempotency_key=f"false-{decision_kind}-{wrong_fact}-create",
            )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key=f"false-{decision_kind}-{wrong_fact}-run",
            schedule=False,
        )
        service._execute(run["runCode"])
        completed = service.get_run(run["runCode"])
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == "semantic_postcondition_miss"
        assert completed["qualifiedCount"] == 0
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize("decision_kind", ("bid", "escalation", "project"))
def test_native_nonplay_current_points_are_zero_at_decision_boundary(
    tmp_path: Path, decision_kind: str, inline_v3_shard_runner: None,
) -> None:
    measured_wall_seconds = {
        "bid": 180.0,       # reference machine p99 44.796s; 3x margin rounded up.
        "escalation": 180.0,  # unclassified host p99 45.107s; 3x margin rounded up.
        "project": 240.0,   # reference machine p99 74.616s; 3x margin rounded up.
    }[decision_kind]
    if decision_kind == "project":
        authority, context = _optional_project_context_and_authority(tmp_path)
        original_rules_sha = context.bundle.rules.sha256
        original_declaration_mode = context.bundle.rules.project_domain.declaration_mode
        original_certification = copy.deepcopy(context.bundle.component("certification"))
        challenge = _apply_measured_nonplay_limits(
            copy.deepcopy(context.bundle.component("challenge")), measured_wall_seconds
        )
        context = replace(
            context,
            bundle=_ChallengeBundleProxy(
                context.bundle,
                challenge,
                preserve_bundle_inputs=True,
            ),
        )
        assert context.bundle.rules.sha256 == original_rules_sha
        assert context.bundle.rules.project_domain.declaration_mode == original_declaration_mode
        assert context.bundle.component("certification") == original_certification
    else:
        authority, legacy_context = _frozen_context_and_authority()
        context = replace(
            legacy_context,
            bundle=_ChallengeBundleProxy(
                legacy_context.bundle,
                _apply_measured_nonplay_limits(_ortools_challenge(), measured_wall_seconds),
            ),
        )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _nonplay_policy_recipe_payload(
            context.bundle_revision_id, decision_kind,
        )
        payload["resources"]["solverTimeoutSeconds"] = measured_wall_seconds
        if decision_kind == "project":
            payload["scenario"]["origin"]["exactTracePrefix"] = _optional_project_prefix()
        kinds = [
            "state.current_raw_points",
            "state.current_project_points",
            "state.current_game_points",
        ]
        for index, kind in enumerate(kinds):
            payload["scenario"]["constraints"]["children"].append({
                "predicateId": f"PRED-NONPLAY-POINTS-{index}",
                "kind": kind,
                "version": 1,
                "hardness": "hard",
                "arguments": {
                    "team": "north_south", "minimum": 0, "maximum": 0,
                },
            })
        if decision_kind in {"bid", "escalation", "project"}:
            purchase_card = payload["scenario"]["origin"]["exactTracePrefix"]["deckOrder"][20]
            payload["scenario"]["constraints"]["children"].append({
                "predicateId": "PRED-PUBLIC-PURCHASE-VISIBLE",
                "kind": "state.observer_card_visibility", "version": 1,
                "hardness": "hard",
                "arguments": {
                    "observer": "west", "card": purchase_card,
                    "visibility": "visible",
                },
            })
        recipe = (
            service.store.put_recipe(service._build_recipe_and_context(payload)[0].to_dict())
            if decision_kind == "project"
            else service.create_recipe(payload, idempotency_key=f"points-{decision_kind}-create")
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key=f"points-{decision_kind}-run",
            schedule=False,
        )
        service._execute(run["runCode"])
        completed = service.get_run(run["runCode"])
        assert completed["scenarioResearchStatus"] == "satisfied", completed["events"]
        result = service.get_result(completed["resultCodes"][0])
        observed = {
            row["predicateId"]: row["observed"]
            for row in result["predicateEvaluation"]["evidence"]
        }
        assert all(
            observed[f"PRED-NONPLAY-POINTS-{index}"][
                {"state.current_raw_points": "rawPoints",
                 "state.current_project_points": "projectPoints",
                 "state.current_game_points": "gamePoints"}[kind]
            ] == 0
            for index, kind in enumerate(kinds)
        )
    finally:
        executor.shutdown(wait=True)


def test_native_project_visibility_includes_prior_first_trick_card(
    tmp_path: Path,
) -> None:
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    authority, context = _optional_project_context_and_authority(tmp_path)
    normalizer = exact_state_module.OrToolsExactStateNormalizerV1(
        context.bundle.rules, native_cp_sat=True,
    )
    prefix = ExactTracePrefixV1.build(_optional_project_prefix())
    north = normalizer.normalize(
        prefix,
        decision_point={"kind": "project", "selector": {"owner": "north"}},
        maximum_seconds=20,
    )
    prefix = normalizer.append_normalized_action(prefix, north, "pass")
    play = normalizer.normalize(
        prefix, decision_point={"kind": "play", "selector": {}},
        maximum_seconds=20,
    )
    purchase_card = prefix.deck_order[20]
    action = next(row for row in play.legal_actions if row["code"] == "SA")
    prefix = normalizer.append_normalized_action(prefix, play, action["code"])
    played_card = Card.parse(prefix.play_actions[-1]["card"]).id
    assert played_card != purchase_card

    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _nonplay_policy_recipe_payload(
            context.bundle_revision_id, "project",
        )
        raw_prefix = prefix.to_dict()
        raw_prefix.pop("sha256", None)
        payload["scenario"]["origin"]["exactTracePrefix"] = raw_prefix
        payload["scenario"]["decisionPoint"]["selector"]["owner"] = "west"
        payload["scenario"]["constraints"]["children"][2]["arguments"]["seat"] = "west"
        payload["scenario"]["constraints"]["children"].append({
            "predicateId": "PRED-PRIOR-PLAY-PUBLIC",
            "kind": "state.observer_card_visibility", "version": 1,
            "hardness": "hard",
            "arguments": {
                "observer": "east", "card": played_card,
                "visibility": "visible",
            },
        })
        payload["scenario"]["constraints"]["children"].append({
            "predicateId": "PRED-OTHER-HAND-STILL-HIDDEN",
            "kind": "state.observer_card_visibility", "version": 1,
            "hardness": "hard",
            "arguments": {
                "observer": "east", "card": 0,
                "visibility": "hidden",
            },
        })
        payload["configuration"] = authority.to_dict()
        candidate_recipe = ResearchRecipeV4.build(payload)
        validated = service._validate_recipe_v4(candidate_recipe, context)
        assert validated["valid"], validated["issues"]
        estimate = service._estimate_recipe_v4(candidate_recipe, validated, context)
        assert estimate["proof"]["remainingCardCount"] == 31
        recipe = service.store.put_recipe(candidate_recipe.to_dict())
        run = service._submit_recipe_run(
            recipe["code"], idempotency_key="project-prior-play-visible-run",
            schedule=False,
        )
        service._execute(run["runCode"])
        completed = service.get_run(run["runCode"])
        assert completed["scenarioResearchStatus"] == "satisfied", completed["events"]
        result = service.get_result(completed["resultCodes"][0])
        observed = next(
            row["observed"] for row in result["predicateEvaluation"]["evidence"]
            if row["predicateId"] == "PRED-PRIOR-PLAY-PUBLIC"
        )
        assert observed["visibility"] == "visible"
        hidden = next(
            row["observed"] for row in result["predicateEvaluation"]["evidence"]
            if row["predicateId"] == "PRED-OTHER-HAND-STILL-HIDDEN"
        )
        assert hidden["visibility"] == "hidden"
    finally:
        executor.shutdown(wait=True)


@pytest.mark.real_clock
def test_v3_bounded_bid_materializes_one_frozen_policy_witness(
    tmp_path: Path,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    challenge = _ortools_challenge()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, challenge),
    )
    service, pool = _isolated_service(tmp_path, authority, context)
    backend = service._cached_ortools_scenario_backend_v2(context)
    assert backend is not None
    try:
        payload = _nonplay_policy_recipe_payload(
            context.bundle_revision_id,
            "bid",
        )
        scenario = ResearchScenarioV1.build(payload["scenario"])
        proof = ScenarioProofRequestV1.build(
            {
                "schemaVersion": 1,
                "mode": "bounded",
                "visibility": "complete_state",
                "strategyClass": "pure_deterministic",
                "claim": "outcome_reachable",
                "targetTeam": "north_south",
                "utility": "round_win",
                "successThreshold": 0,
                "horizon": {"unit": "plies", "value": 1},
                "boundaryEvaluator": "current_raw_point_margin_v1",
                "budgetProfile": "bounded_v1",
            }
        )

        campaign = backend.execute_campaign(
            context.bundle.rules,
            scenario,
            proof,
            result_count=1,
            symmetry_policy_id="rules_preserving_v1",
            allow_symmetry_only_variants=False,
            minimum_decision_state_distance=1,
            maximum_proof_nodes=250_000,
            maximum_proof_depth=40,
            maximum_candidates=1,
            maximum_seconds=10,
        )

        assert campaign.status == "completed"
        result = campaign.accepted_results[0]
        assert result.proof["completionStatus"] == "bounded_complete"
        trace = result.candidate.trace
        witness = trace["witnessContinuation"]
        assert witness["rootAction"] == result.proof["selectedAction"]
        assert len(trace["completeExactTracePrefix"]["playActions"]) == 32
        assert trace["completeExactNormalization"]["phase"] == "scored"
        assert result.execution_receipt["stageTimings"]["total"] < 10
    finally:
        pool.shutdown(wait=True)


@pytest.mark.parametrize("origin_kind", ("exact_trace_prefix", "search"))
def test_optional_project_policy_run_is_fully_service_qualified(
    tmp_path: Path,
    origin_kind: str,
) -> None:
    authority, context = _optional_project_context_and_authority(tmp_path)
    service, pool = _isolated_service(tmp_path, authority, context)
    try:
        payload = _exact_recipe_payload(context.bundle_revision_id)
        payload["scenario"] = {
            "schemaVersion": 1,
            "origin": (
                    {
                        "kind": "exact_trace_prefix",
                        "exactTracePrefix": _optional_project_prefix(),
                        "recordedEscalationOpportunities": [],
                    }
                if origin_kind == "exact_trace_prefix"
                else {"kind": "search"}
            ),
            "constraints": {
                "op": "all",
                "children": [
                    {
                        "predicateId": "PRED-OPTIONAL-PROJECT-SERVICE-ACTIONS",
                        "kind": "auction.contract",
                        "version": 1,
                        "hardness": "hard",
                        "arguments": {"contract": "sun"},
                    },
                    {
                        "predicateId": "PRED-OPTIONAL-PROJECT-SERVICE-ELIGIBLE",
                        "kind": "project.exists",
                        "version": 1,
                        "hardness": "hard",
                        "arguments": {
                            "team": "north_south",
                            "owner": "north",
                            "kind": "hundred_sequence",
                            "value": 20,
                            "state": "eligible",
                        },
                    },
                    {
                        "predicateId": "PRED-OPTIONAL-PROJECT-SERVICE-PHASE",
                        "kind": "state.phase", "version": 1,
                        "hardness": "hard", "arguments": {"state": "playing"},
                    },
                    {
                        "predicateId": "PRED-OPTIONAL-PROJECT-SERVICE-ACTOR",
                        "kind": "state.next_actor", "version": 1,
                        "hardness": "hard", "arguments": {"seat": "north"},
                    },
                    {
                        "predicateId": "PRED-OPTIONAL-PROJECT-SERVICE-LEGAL-COUNT",
                        "kind": "state.legal_action_count", "version": 1,
                        "hardness": "hard",
                        "arguments": {"minimum": 1, "maximum": 64},
                    },
                    {
                        "predicateId": "PRED-OPTIONAL-PROJECT-SERVICE-MATCH",
                        "kind": "state.match_score", "version": 1,
                        "hardness": "hard",
                        "arguments": {"team": "north_south", "minimum": 0, "maximum": 152},
                    },
                    {
                        "predicateId": "PRED-OPTIONAL-PROJECT-SERVICE-CARD-OWNER",
                        "kind": "deal.card_owner", "version": 1,
                        "hardness": "hard",
                        "arguments": {
                            "card": 2 if origin_kind == "exact_trace_prefix" else 0,
                            "seat": "north",
                            "handStage": "post_purchase",
                        },
                    },
                    {
                        "predicateId": "PRED-OPTIONAL-PROJECT-SERVICE-CARD-VISIBLE",
                        "kind": "state.observer_card_visibility", "version": 1,
                        "hardness": "hard",
                        "arguments": {
                            "observer": "north",
                            "card": 2 if origin_kind == "exact_trace_prefix" else 0,
                            "visibility": "visible",
                        },
                    },
                ],
            },
            "decisionPoint": {
                "kind": "project",
                    "selector": {"owner": "north", "projectIndex": 0},
            },
            "outcome": {
                "predicateId": "PRED-OPTIONAL-PROJECT-SERVICE-OUTCOME",
                "kind": "outcome.round_margin",
                "version": 1,
                "hardness": "hard",
                "arguments": {
                    "team": "west_east",
                    "minimum": -100000,
                    "maximum": 100000,
                },
            },
        }
        payload["proof"] = {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "targetTeam": "west_east",
            "utility": "round_win",
            "successThreshold": -1,
            "policies": _fixed_policy_templates(),
            "budgetProfile": "actor_information_set_exact_v1",
        }
        payload["resources"]["proofBudgetProfile"] = (
            "actor_information_set_exact_v1"
        )
        payload["resources"]["solverTimeoutSeconds"] = 60.0
        recipe = service.store.put_recipe(service._build_recipe_and_context(payload)[0].to_dict())
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key=f"optional-project-{origin_kind}-policy-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed["status"] == "completed", json.dumps(
            completed.get("events", []), ensure_ascii=True
        )
        assert completed["scenarioResearchStatus"] == "satisfied", completed[
            "events"
        ]
        result = service.get_result(completed["resultCodes"][0])
        assert result["proofOutcome"]["completionStatus"] == "policy_complete"
        trace = json.loads(
            service.artifact_store.get(result["artifacts"]["completeTrace"]["uri"])
        )
        count_evidence = next(
            row for row in result["predicateEvaluation"]["evidence"]
            if row["predicateId"] == "PRED-OPTIONAL-PROJECT-SERVICE-LEGAL-COUNT"
        )
        assert count_evidence["observed"]["legalActionCount"] == len(
            trace["exactNormalization"]["legalActions"]
        )
        match_evidence = next(
            row for row in result["predicateEvaluation"]["evidence"]
            if row["predicateId"] == "PRED-OPTIONAL-PROJECT-SERVICE-MATCH"
        )
        assert match_evidence["observed"]["matchScore"] == trace[
            "exactTracePrefix"
        ]["matchScore"][0]
        assert len(trace["completeExactTracePrefix"]["playActions"]) == 32
        assert trace["completeExactNormalization"]["phase"] == "scored"
        assert result["runtimeCompatibility"]["status"] == "passed", json.dumps(
            result["runtimeCompatibilityEvidence"], ensure_ascii=True
        )
    finally:
        pool.shutdown(wait=True)


@pytest.mark.parametrize("origin_kind", ("exact_trace_prefix", "search"))
def test_optional_project_bounded_run_is_fully_service_qualified(
    tmp_path: Path,
    origin_kind: str,
) -> None:
    authority, context = _optional_project_context_and_authority(tmp_path)
    service, pool = _isolated_service(tmp_path, authority, context)
    try:
        payload = _nonplay_complete_graph_recipe_payload(
            context.bundle_revision_id,
            "project",
            mode="bounded",
        )
        payload["scenario"]["origin"] = (
                {
                    "kind": "exact_trace_prefix",
                    "exactTracePrefix": _optional_project_prefix(),
                    "recordedEscalationOpportunities": [],
                }
            if origin_kind == "exact_trace_prefix"
            else {"kind": "search"}
        )
        payload["scenario"]["decisionPoint"] = {
            "kind": "project",
            "selector": {"owner": "north", "projectIndex": 0},
        }
        payload["scenario"]["constraints"] = {
            "op": "all",
            "children": [
                payload["scenario"]["constraints"],
                {
                    "predicateId": "PRED-OPTIONAL-PROJECT-BOUNDED-ELIGIBLE",
                    "kind": "project.exists",
                    "version": 1,
                    "hardness": "hard",
                    "arguments": {
                        "team": "north_south",
                        "owner": "north",
                        "kind": "hundred_sequence",
                        "value": 20,
                        "state": "eligible",
                    },
                },
            ],
        }
        payload["scenario"]["outcome"] = {
            "predicateId": "PRED-OPTIONAL-PROJECT-BOUNDED-OUTCOME",
            "kind": "outcome.round_margin",
            "version": 1,
            "hardness": "hard",
            "arguments": {
                "team": "west_east",
                "minimum": -100000,
                "maximum": 100000,
            },
        }
        payload["proof"]["targetTeam"] = "west_east"
        payload["proof"]["successThreshold"] = -100000
        payload["proof"]["budgetProfile"] = "actor_information_set_exact_v1"
        payload["resources"]["proofBudgetProfile"] = (
            "actor_information_set_exact_v1"
        )
        payload["resources"]["solverTimeoutSeconds"] = 60.0
        recipe = service.store.put_recipe(service._build_recipe_and_context(payload)[0].to_dict())
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key=f"optional-project-{origin_kind}-bounded-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == "satisfied", completed[
            "events"
        ]
        result = service.get_result(completed["resultCodes"][0])
        assert result["proofOutcome"]["completionStatus"] == "bounded_complete"
        assert result["runtimeCompatibility"]["status"] == "passed"
        assert result["disposition"] == "qualified"
    finally:
        pool.shutdown(wait=True)


@pytest.mark.parametrize(
    ("origin_kind", "decision_kind", "fixed_deal"),
    (
        pytest.param("exact_trace_prefix", "bid", False, id="exact_trace_prefix-bid"),
        pytest.param("exact_trace_prefix", "escalation", False, id="exact_trace_prefix-escalation"),
        pytest.param("search", "bid", False, id="search-bid"),
        pytest.param("search", "escalation", False, id="search-escalation"),
        pytest.param("search", "bid", True, id="search-bid-fixed-deal"),
        pytest.param("search", "escalation", True, id="search-escalation-fixed-deal"),
    ),
)
def test_bounded_nonplay_complete_graph_run_persists_qualified_proof(
    tmp_path: Path,
    origin_kind: str,
    decision_kind: str,
    fixed_deal: bool,
) -> None:
    # Sulaiman-v2 measured p99 is 229.816s across 52 eligible runs; 3x margin
    # rounded up to the 720s declared budget.
    measured_wall_seconds = 720.0
    authority, legacy_context = _frozen_context_and_authority()
    challenge = _ortools_challenge()
    # Reuse full-round timing without changing the proof universe.
    full_round = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/challenge/intelligent_research.yaml")
        .read_text(encoding="utf-8")
    )["scenarioResearchV4"]
    for group, name, limits in (
        ("proofBudgets", "bounded_v1", ("maximumWallSeconds", "maximumCpuSeconds")),
        ("searchBudgets", "late_game_exact_v1", ("maximumWallSeconds",)),
    ):
        challenge["scenarioResearchV4"][group]["profiles"][name].update({
            key: full_round[group]["profiles"][name][key] for key in limits
        })
    _apply_measured_nonplay_limits(challenge, measured_wall_seconds)
    proof_budget = challenge["scenarioResearchV4"]["proofBudgets"]["profiles"][
        "late_game_exact_v1"
    ]
    proof_budget.update(
        {
            "maximumNodes": 1_000,
            "maximumDepth": 4,
        }
    )
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, challenge),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _nonplay_complete_graph_recipe_payload(
            context.bundle_revision_id,
            decision_kind,
            mode="bounded",
        )
        visible_card = 0
        if origin_kind == "search":
            witness = payload["scenario"]["origin"]["exactTracePrefix"]
            search_payload = _search_nonplay_policy_recipe_payload(
                context.bundle_revision_id,
                decision_kind,
            )
            search_payload["proof"] = payload["proof"]
            payload = search_payload
            if fixed_deal:
                # Search the known finite witness; keep broad discovery cases separate.
                layout = DealLayoutV1.build(context.bundle.rules.structural_domain,
                    dealer=Seat.parse(witness["dealer"]), buyer=Seat.SOUTH)
                visible_card = witness["deckOrder"][layout.positions_for(Seat.WEST, final=False)[0]]
                constraints = payload["scenario"]["constraints"]["children"]
                for seat in Seat:
                    initial = set(layout.positions_for(seat, final=False))
                    for position in layout.positions_for(seat, final=True):
                        constraints.append({
                            "predicateId": f"PRED-FIXED-OWNER-{position}",
                            "kind": "deal.card_owner", "version": 1, "hardness": "hard",
                            "arguments": {"card": witness["deckOrder"][position],
                                "seat": seat.name.lower(),
                                "handStage": "initial_five" if position in initial else "post_purchase"},
                        })
                for predicate_id, kind, arguments in (
                    ("PRED-FIXED-DEALER", "auction.dealer", {"seat": witness["dealer"]}),
                    ("PRED-FIXED-PURCHASE", "deal.purchase_card",
                        {"card": witness["deckOrder"][layout.purchase_position]}),
                    ("PRED-FIXED-MATCH", "state.match_score",
                        {"team": "north_south", "minimum": witness["matchScore"][0],
                         "maximum": witness["matchScore"][0]}),
                ):
                    constraints.append({"predicateId": predicate_id, "kind": kind,
                        "version": 1, "hardness": "hard", "arguments": arguments})
        payload["playingQuality"]["decisions"]["play"]["budgetProfile"] = "bounded_v1"
        for kind in (
            "state.current_raw_points", "state.current_project_points",
            "state.current_game_points",
        ):
            payload["scenario"]["constraints"]["children"].append({
                "predicateId": (
                    f"PRED-{origin_kind.upper().replace('_', '-')}-"
                    f"{kind.upper().replace('.', '-').replace('_', '-')}"
                ),
                "kind": kind, "version": 1, "hardness": "hard",
                "arguments": {
                    "team": "north_south", "minimum": 0, "maximum": 0,
                },
            })
        if origin_kind == "search":
            payload["scenario"]["constraints"]["children"].extend((
                {
                    "predicateId": "PRED-SEARCH-VISIBLE-OWNER",
                    "kind": "deal.card_owner", "version": 1, "hardness": "hard",
                    "arguments": {
                        "card": visible_card, "seat": "west",
                        "handStage": (
                            "initial_five" if decision_kind == "bid"
                            else "post_purchase"
                        ),
                    },
                },
                {
                    "predicateId": "PRED-SEARCH-OWNER-VISIBLE",
                    "kind": "state.observer_card_visibility", "version": 1,
                    "hardness": "hard",
                    "arguments": {
                        "observer": "west", "card": visible_card, "visibility": "visible",
                    },
                },
            ))
        payload["resources"]["solverTimeoutSeconds"] = measured_wall_seconds
        key = f"ortools-{origin_kind}-{decision_kind}-bounded"
        recipe = service.create_recipe(
            payload,
            idempotency_key=f"{key}-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key=f"{key}-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None
        assert completed["status"] == "completed", json.dumps(
            completed.get("events", []), ensure_ascii=True
        )
        assert completed["scenarioResearchStatus"] == "satisfied", json.dumps(
            completed.get("events", []), ensure_ascii=True
        )
        assert completed["resultCount"] == completed["qualifiedCount"] == 1
        result = service.get_result(completed["resultCodes"][0])
        assert result is not None
        assert result["proofOutcome"]["completionStatus"] == "bounded_complete"
        assert result["proofOutcome"]["boundaryEvaluator"] == (
            "current_raw_point_margin_v1"
        )
        assert result["proofOutcome"]["rootActions"]
        compatibility = result["runtimeCompatibility"]
        assert compatibility["status"] == "passed", json.dumps(
            result["runtimeCompatibilityEvidence"], ensure_ascii=True
        )
        assert compatibility["failureCodes"] == []
        assert result["disposition"] == "qualified"
        graph = json.loads(
            service.artifact_store.get(result["artifacts"]["proofGraph"]["uri"])
        )
        assert graph["scope"] == "bounded"
        trace = json.loads(service.artifact_store.get(result["artifacts"]["completeTrace"]["uri"]))
        assert len(trace["completeExactTracePrefix"]["playActions"]) == 32
        assert trace["completeExactNormalization"]["phase"] == "scored"
        assert trace["playingQualityEvidence"]["receipt"]["status"] == "complete"
        assert trace["completeExactTracePrefix"] == trace["playingQualityEvidence"]["receipt"]["finalPrefix"]
    finally:
        executor.shutdown(wait=True)


def test_exact_bid_pass_root_keeps_the_intelligent_playable_trace(tmp_path: Path) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    challenge = _ortools_challenge()
    challenge["scenarioResearchV4"]["searchBudgets"]["profiles"][
        "interactive_validation_v1"
    ]["maximumWallSeconds"] = 20.0
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, challenge),
    )
    payload = _nonplay_policy_recipe_payload(context.bundle_revision_id, "bid")
    payload["scenario"]["outcome"] = {
        "predicateId": "PRED-BID-ALL-PASS-REPORT",
        "kind": "state.phase",
        "version": 1,
        "hardness": "report",
        "arguments": {"state": "bidding"},
    }
    payload["proof"]["selectedAction"] = "pass"
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        recipe = service.create_recipe(
            payload,
            idempotency_key="ortools-bid-all-pass-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key="ortools-bid-all-pass-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == "satisfied", json.dumps(
            completed.get("events", []), ensure_ascii=True
        )
        result = service.get_result(completed["resultCodes"][0])
        assert result is not None
        trace = json.loads(
            service.artifact_store.get(result["artifacts"]["completeTrace"]["uri"])
        )
        assert "policyContinuation" not in trace
        quality = trace["playingQualityEvidence"]["receipt"]
        assert trace["completeExactTracePrefix"] == quality["finalPrefix"]
        assert len(trace["completeExactTracePrefix"]["playActions"]) == 32
        assert trace["completeExactNormalization"]["phase"] == "scored"
        assert result["runtimeCompatibility"]["status"] == "passed"
        assert result["runtimeCompatibility"]["failureCodes"] == []
        assert result["disposition"] == "qualified"
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize("decision_kind", ("bid", "escalation"))
def test_search_nonplay_policy_run_qualifies_normalized_decision_and_continuation(
    tmp_path: Path,
    decision_kind: str,
) -> None:
    measured_wall_seconds = {
        "bid": 240.0,       # reference machine p99 74.404s; 3x margin rounded up.
        "escalation": 300.0,  # reference machine p99 96.047s; 3x margin rounded up.
    }[decision_kind]
    authority, legacy_context = _frozen_context_and_authority()
    challenge = _apply_measured_nonplay_limits(_ortools_challenge(), measured_wall_seconds)
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, challenge),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _search_nonplay_policy_recipe_payload(
            context.bundle_revision_id,
            decision_kind,
        )
        payload["resources"]["solverTimeoutSeconds"] = measured_wall_seconds
        recipe = service.create_recipe(
            payload,
            idempotency_key=f"ortools-search-{decision_kind}-policy-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key=f"ortools-search-{decision_kind}-policy-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None
        assert completed["status"] == "completed", json.dumps(
            completed.get("events", []), ensure_ascii=True
        )
        assert completed["scenarioResearchStatus"] == "satisfied", json.dumps(
            completed.get("events", []), ensure_ascii=True
        )
        assert completed["qualifiedCount"] == 1
        result = service.get_result(completed["resultCodes"][0])
        assert result is not None
        assert result["proofOutcome"]["completionStatus"] == "policy_complete"
        trace = json.loads(
            service.artifact_store.get(result["artifacts"]["completeTrace"]["uri"])
        )
        assert trace["exactNormalization"]["decisionKind"] == decision_kind
        assert "policyContinuation" not in trace
        quality = trace["playingQualityEvidence"]["receipt"]
        assert trace["completeExactTracePrefix"] == quality["finalPrefix"]
        assert len(trace["completeExactTracePrefix"]["playActions"]) == 32
        assert trace["score"]["gamePoints"] == trace["score"]["awardedPoints"]
        assert len(trace["score"]["cardGamePoints"]) == 2
        assert trace["completeExactNormalization"]["phase"] == "scored"
        assert result["runtimeCompatibility"]["status"] == "passed"
    finally:
        executor.shutdown(wait=True)


def test_policy_conditioned_v4_run_materializes_governed_profiles(
    tmp_path: Path,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _policy_service_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        recipe = service.create_recipe(
            _policy_conditioned_recipe_payload(context.bundle_revision_id),
            idempotency_key="ortools-policy-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key="ortools-policy-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == "satisfied"
        assert completed["resultCount"] == completed["qualifiedCount"] == 1
        result = service.get_result(completed["resultCodes"][0])
        assert result is not None
        proof = result["proofOutcome"]
        assert proof["completionStatus"] == "policy_complete"
        assert proof["continuationLines"]["lineUniverse"] == "policy_conditioned"
        assert proof["continuationLines"]["exactCount"] in {"0", "1"}
        templates = _fixed_policy_templates()
        assert proof["identities"]["policyHashes"] == {
            seat: profile["profileHash"] for seat, profile in templates.items()
        }
        proof_artifact = json.loads(
            service.artifact_store.get(result["artifacts"]["proofOutcome"]["uri"])
        )
        assert {
            seat: identity["profileHash"]
            for seat, identity in proof_artifact["policyIdentities"].items()
        } == {seat: profile["profileHash"] for seat, profile in templates.items()}
        assert all(
            identity["policyId"] == "fixed_action_map_v1"
            and identity["materialization"] == "seeded_legal_action_order_v1"
            and identity["materializedHash"] != identity["profileHash"]
            and type(identity["seed"]) is int
            for identity in proof_artifact["policyIdentities"].values()
        )
        trace = json.loads(
            service.artifact_store.get(result["artifacts"]["completeTrace"]["uri"])
        )
        source_count = len(trace["exactTracePrefix"]["playActions"])
        complete_actions = trace["completeExactTracePrefix"]["playActions"]
        assert source_count == 24
        assert len(complete_actions) == 32
        quality = trace["playingQualityEvidence"]["receipt"]
        assert quality["status"] == "complete"
        assert trace["completeExactTracePrefix"] == quality["finalPrefix"]
        assert "policyContinuation" not in trace
        assert trace["completeExactNormalization"]["phase"] == "scored"
        assert result["runtimeCompatibility"]["status"] == "passed"
        boundary = completed["scenarioResearchEvidence"]["proofBoundaries"][0]
        assert boundary["completionStatus"] == "policy_complete"
        assert (
            boundary["proofOutcomeHash"]
            == result["researchAuthority"]["proofOutcomeHash"]
        )
        public = api._public_research_result(result)
        api.ResearchScenarioPublicResultResponse.model_validate(public)
        assert public["proofOutcome"]["requestedMode"] == "policy_conditioned"
        assert public["proofOutcome"]["achievedMode"] == "policy_conditioned"
        assert public["proofOutcome"]["completionStatus"] == "policy_complete"
        assert public["proofOutcome"]["strategyClass"] == "frozen_policy"
        assert public["proofOutcome"]["identities"]["policyHashes"] == proof["identities"]["policyHashes"]
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize(
    ("claim", "success_threshold", "claim_satisfied"),
    (
        ("winning_root_action_count", 1, True),
        ("outcome_forced", -1, False),
        ("outcome_forced", -100000, True),
    ),
)
def test_policy_conditioned_v4_run_persists_broadened_claims(
    tmp_path: Path,
    claim: str,
    success_threshold: int,
    claim_satisfied: bool,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _policy_service_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _policy_conditioned_recipe_payload(
            context.bundle_revision_id,
            claim=claim,
            success_threshold=success_threshold,
        )
        assert payload["proof"]["successThreshold"] == success_threshold
        recipe = service.create_recipe(
            payload,
            idempotency_key=f"ortools-policy-{claim}-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key=f"ortools-policy-{claim}-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == ("satisfied" if claim_satisfied else "proof_failed")
        assert completed["resultCount"] == completed["qualifiedCount"] == int(claim_satisfied)
        codes = completed["resultCodes"] if claim_satisfied else completed["auditResultCodes"]
        assert len(codes) == 1
        result = service.get_result(codes[0])
        assert result is not None
        proof = result["proofOutcome"]
        assert proof["claim"] == claim
        assert proof["completionStatus"] == ("policy_complete" if claim_satisfied else "proof_failed")
        assert proof["achievedMode"] == "policy_conditioned"
        assert proof["claimSatisfied"] is claim_satisfied
        assert type(proof["winningRootActionCount"]) is int
        assert proof["continuationLines"]["complete"] is False
        assert proof["continuationLines"]["exactCount"] is None
    finally:
        executor.shutdown(wait=True)


def test_policy_conditioned_runtime_policy_fails_preflight(
    tmp_path: Path,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _policy_conditioned_recipe_payload(context.bundle_revision_id)
        payload["proof"]["policies"]["west"] = {
            "agentId": "heuristic_bot_v1",
            "version": "1.0.0",
            "seed": 2,
            "profileHash": "0" * 64,
        }

        with pytest.raises(ValueError, match="policy_profile_invalid"):
            service.create_recipe(payload)
    finally:
        executor.shutdown(wait=True)


def test_policy_conditioned_incomplete_proof_preserves_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inline_v3_shard_runner: None,
) -> None:
    real_prove = OrToolsScenarioGraphProofV1.prove

    def incomplete_target_proof(self, graph, request, **kwargs):
        outcome = real_prove(self, graph, request, **kwargs)
        if request.claim != "winning_continuation_line_count":
            return outcome
        return OrToolsGraphProofOutcomeV1.build(
            completion_status="proof_incomplete",
            solver_status="feasible",
            claim_satisfied=False,
            selected_action=None,
            winning_root_action_count=None,
            winning_root_action_lower_bound=0,
            winning_root_action_upper_bound=len(outcome.root_actions),
            root_actions=outcome.root_actions,
            principal_variation=(),
            continuation_lines=None,
            policies=outcome.policies,
            graph_hash=outcome.graph_hash,
            model_hash=outcome.model_hash,
            request_hash=outcome.request_hash,
            variable_count=outcome.variable_count,
            constraint_count=outcome.constraint_count,
            model_document=outcome.model_document(),
            hidden_worlds=outcome.hidden_worlds,
            native_qualification=outcome.native_qualification,
        )

    monkeypatch.setattr(OrToolsScenarioGraphProofV1, "prove", incomplete_target_proof)
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _policy_service_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _native_qualified_exact_recipe_payload(context.bundle_revision_id)
        proof = _policy_conditioned_recipe_payload(context.bundle_revision_id)["proof"]
        proof["selectedAction"] = payload["proof"]["selectedAction"]
        payload["proof"] = proof
        recipe = service.create_recipe(
            payload,
            idempotency_key="ortools-policy-incomplete-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key="ortools-policy-incomplete-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == "proof_incomplete"
        assert completed["resultCount"] == completed["qualifiedCount"] == 0
        assert completed["resultCodes"] == []
        assert completed["auditResultCount"] == 1
        assert len(completed["auditResultCodes"]) == 1
        audit_result = service.get_result(completed["auditResultCodes"][0])
        assert audit_result is not None
        assert audit_result["terminalStatus"] == "proof_incomplete"
        assert audit_result["disposition"] == "rejected"
        assert audit_result["qualification"]["qualified"] is False
        assert audit_result["proofOutcome"]["winningRootActionCount"] is None
        assert audit_result["proofOutcome"]["winningRootActionLowerBound"] == 0
        assert audit_result["proofOutcome"]["winningRootActionUpperBound"] == len(
            audit_result["proofOutcome"]["rootActions"]
        )
        assert audit_result["scenarioProjection"]["exactStateAvailable"] is True
        assert audit_result["runtimeCompatibility"]["status"] == "passed"
        assert set(audit_result["authorityArtifacts"]) == {
            "configurationSnapshot",
            "loweringManifest",
            "resultIdentity",
            "executionReceipt",
        }
        root = Path(__file__).resolve().parents[1]
        video_profile = yaml.safe_load(
            (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
        )
        terminal_renderer = ResearchRenderPreparationServiceV1(
            research=SimpleNamespace(
                get_result=lambda _code: audit_result,
                get_recipe=lambda _code: service.get_recipe(recipe["code"]),
                get_playing_quality_audit=service.get_playing_quality_audit,
            ),
            artifact_store=service.artifact_store,
            publication_service=SimpleNamespace(),
            renderer=RemotionProfessionalRenderer(
                video_profile,
                repository_root=root,
            ),
            rules=context.bundle.rules,
            configuration_store=api.CONTAINER.configuration_store,
        )
        prepared = terminal_renderer.prepare(audit_result["resultCode"])
        assert prepared["publicationEligible"] is False
        assert prepared["renderPurpose"] == "research_audit"
        assert prepared["source"]["proofAchievedMode"] == "incomplete"
        assert prepared["basename"].endswith("-audit")
        evidence = completed["scenarioResearchEvidence"]
        assert evidence["proofBoundaries"] == []
        campaign = json.loads(service.artifact_store.get(evidence["uri"]))
        assert len(campaign["executionEvidence"]) == 1
        execution = json.loads(
            service.artifact_store.get(campaign["executionEvidence"][0]["uri"])
        )
        assert execution["status"] == "incomplete"
        assert execution["proof"]["completionStatus"] == "proof_incomplete"
        assert {
            seat: identity["profileHash"]
            for seat, identity in execution["proof"]["policyIdentities"].items()
        } == {
            seat: profile["profileHash"]
            for seat, profile in _fixed_policy_templates().items()
        }
    finally:
        executor.shutdown(wait=True)


def test_exact_v4_line_count_exhaustion_retains_safe_audit_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inline_v3_shard_runner: None,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    challenge = _ortools_challenge()
    # The measured 60s campaign stopped during prefix auditing, before count proof.
    challenge["scenarioResearchV4"]["searchBudgets"]["profiles"]["late_game_exact_v1"]["maximumWallSeconds"] = 75.0
    proof_budget = challenge["scenarioResearchV4"]["proofBudgets"]["profiles"][
        "late_game_exact_v1"
    ]
    proof_budget["maximumNodes"] = 1_000
    proof_budget["maximumLineCountNodes"] = 100
    challenge["scenarioResearchV4"]["refusalThresholds"]["maximumLineCountLimbs"] = 1
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, challenge),
    )

    def exhausted_count(_model, *, maximum_seconds, cancel_check=None):
        assert maximum_seconds > 0
        assert cancel_check is not None
        return cp_model.CpSolver(), cp_model.UNKNOWN

    monkeypatch.setattr(
        OrToolsScenarioGraphProofV1,
        "_solve_count",
        staticmethod(exhausted_count),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _native_qualified_exact_recipe_payload(context.bundle_revision_id)
        payload["proof"]["claim"] = "winning_continuation_line_count"
        recipe = service.create_recipe(
            payload,
            idempotency_key="ortools-line-bound-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key="ortools-line-bound-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == "proof_incomplete"
        assert completed["resultCount"] == completed["qualifiedCount"] == 0
        assert completed["resultCodes"] == []
        assert completed["auditResultCount"] == 1
        assert len(completed["auditResultCodes"]) == 1
        evidence = completed["scenarioResearchEvidence"]
        boundary = evidence["proofBoundaries"][0]
        lines = boundary["continuationLines"]
        assert boundary["completionStatus"] == "proof_incomplete"
        assert boundary["claimSatisfied"] is False
        assert boundary["boundaryEvaluator"] == "terminal_utility"
        assert lines["complete"] is False
        assert lines["exactCount"] is None
        assert lines["lowerBound"] == "0"
        assert int(lines["upperBound"]) >= 1
        assert lines["exhaustionReason"] == "line_count_incomplete"
        assert boundary["budget"] == {
            "profileId": "late_game_exact_v1",
            "proofBudgetHash": boundary["budget"]["proofBudgetHash"],
            "maximumLineCountNodes": 100,
            "maximumLineCountLimbs": 1,
        }
        campaign = json.loads(service.artifact_store.get(evidence["uri"]))
        assert campaign["proofBoundaries"] == evidence["proofBoundaries"]
        execution = json.loads(
            service.artifact_store.get(campaign["executionEvidence"][0]["uri"])
        )
        assert execution["status"] == "incomplete"
        assert execution["proof"]["completionStatus"] == "proof_incomplete"
        assert execution["proof"]["continuationLines"] == lines
        assert (
            execution["researchAuthority"]["proofOutcomeHash"]
            == (boundary["proofOutcomeHash"])
        )
        result = service.get_result(completed["auditResultCodes"][0])
        assert result is not None
        cold_service, cold_executor = _isolated_service(tmp_path, authority, context)
        try:
            assert not cold_service._result_verification_cache

            def reject_current_execution_context(*_args, **_kwargs):
                pytest.fail("cold incomplete reads must use the stored configuration snapshot")

            monkeypatch.setattr(cold_service, "_build_recipe_and_context", reject_current_execution_context)
            restored = cold_service.get_result(result["resultCode"])
            assert restored["resultHash"] == result["resultHash"]
            assert restored["artifacts"] == result["artifacts"]
            assert restored["runCode"] == run["runCode"]
            assert restored["recipeCode"] == recipe["code"]
            assert restored["disposition"] != "qualified"
            proof = restored["proofOutcome"]
            assert proof == result["proofOutcome"]
            assert proof["completionStatus"] == "proof_incomplete"
            assert proof["achievedMode"] == "none"
            assert proof["claimSatisfied"] is False
            assert proof["continuationLines"] == {
                key: lines[key] for key in (
                    "complete", "exactCount", "lowerBound", "upperBound", "lineUniverse"
                )
            }
        finally:
            cold_executor.shutdown(wait=True)
    finally:
        executor.shutdown(wait=True)


def test_native_counterstrategy_persists_without_playable_state(tmp_path, inline_v3_shard_runner):
    from test_research_ortools_flagship_search_recipe import _flagship_scenario, _exact_unique_winner_proof

    authority, context = _frozen_context_and_authority()
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _exact_recipe_payload(context.bundle_revision_id)
        payload["scenario"] = _flagship_scenario().to_dict()
        payload["scenario"].pop("sha256", None)
        payload["proof"] = _exact_unique_winner_proof().to_dict()
        payload["proof"].pop("sha256", None)
        recipe = service.create_recipe(payload, idempotency_key="counterstrategy-create")
        run = service._submit_recipe_run(recipe["code"], idempotency_key="counterstrategy-run", schedule=False)
        service._execute(run["runCode"])
        completed = service.get_run(run["runCode"])
        assert completed["status"] == "completed", completed
        assert completed["scenarioResearchStatus"] == "proof_failed", completed
        assert completed["resultCodes"] == []
        assert completed["auditResultCount"] == 1
        code = completed["auditResultCodes"][0]
        result = service.get_result(code)
        assert result["terminalStatus"] == "proof_failed"
        assert "native_round_win_counterstrategy" in result["qualification"]["failures"]
        assert result["qualification"]["qualified"] is False
        assert result["scenarioProjection"]["exactStateAvailable"] is False
        assert result["proofOutcome"]["claimSatisfied"] is False
        service._result_verification_cache.clear()
        assert service.get_result(code) == result
    finally:
        executor.shutdown(wait=True)


def test_native_proof_failed_result_prepares_badged_non_publishable_audit_media(
    tmp_path: Path,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _policy_service_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _native_qualified_exact_recipe_payload(context.bundle_revision_id)
        payload["proof"]["claim"] = "selected_action_is_unique_winner"
        recipe = service.create_recipe(
            payload,
            idempotency_key="ortools-proof-failed-audit-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key="ortools-proof-failed-audit-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == "proof_failed", json.dumps(
            completed.get("events", []), ensure_ascii=False
        )
        assert completed["resultCodes"] == []
        assert completed["auditResultCount"] == 1
        audit_result = service.get_result(completed["auditResultCodes"][0])
        assert audit_result is not None
        assert audit_result["terminalStatus"] == "proof_failed"
        assert audit_result["disposition"] == "rejected"
        assert audit_result["qualification"]["qualified"] is False
        assert audit_result["proofOutcome"]["completionStatus"] == "proof_failed"
        assert audit_result["proofOutcome"]["achievedMode"] == "exact"
        assert audit_result["proofOutcome"]["claimSatisfied"] is False
        assert audit_result["scenarioProjection"]["exactStateAvailable"] is True
        assert audit_result["runtimeCompatibility"]["status"] == "passed"

        root = Path(__file__).resolve().parents[1]
        video_profile = yaml.safe_load(
            (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
        )
        render_service = ResearchRenderPreparationServiceV1(
            research=SimpleNamespace(
                get_result=lambda _code: audit_result,
                get_recipe=lambda _code: service.get_recipe(recipe["code"]),
                get_playing_quality_audit=service.get_playing_quality_audit,
            ),
            artifact_store=service.artifact_store,
            publication_service=SimpleNamespace(),
            renderer=RemotionProfessionalRenderer(
                video_profile,
                repository_root=root,
            ),
            rules=context.bundle.rules,
            configuration_store=api.CONTAINER.configuration_store,
        )
        prepared = render_service.prepare(audit_result["resultCode"])

        assert prepared["publicationEligible"] is False
        assert prepared["renderPurpose"] == "research_audit"
        assert prepared["locator"] is None
        assert prepared["basename"].endswith("-audit")
        assert prepared["source"]["auditOnly"] is True
        assert prepared["source"]["publicationEligible"] is False
        assert prepared["source"]["qrActionable"] is False
        assert prepared["source"]["proofAchievedMode"] == "exact"
        assert prepared["source"]["proofCompletionStatus"] == "proof_failed"
        assert prepared["source"]["proofClaimSatisfied"] is False
        bundle = json.loads(
            service.artifact_store.get(prepared["bundleArtifact"]["uri"])
        )
        assert bundle["source"] == prepared["source"]
        assert "scenario_claim_not_satisfied" in prepared["publicationBlockers"]
        assert video_profile["localizations"][
            "video.governance.research_audit_badge"
        ]
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize("include_stopping_certificate", (False, True))
def test_exact_actor_information_set_run_persists_exhaustive_safe_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_stopping_certificate: bool,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _actor_information_exact_recipe_payload(context.bundle_revision_id)
        if include_stopping_certificate:
            payload["proof"].update(utility="kabout", successThreshold=1)
            payload["scenario"]["outcome"] = {
                "op": "all",
                "children": [payload["scenario"]["outcome"], {
                    "predicateId": "PRED-REPORT-KABOUT-STOP",
                    "kind": "tactic.verified",
                    "version": 1,
                    "hardness": "report",
                    "arguments": {"kind": "kabout_prevention", "minimumScore": 100,
                        "minimumEvidence": 1, "causalRequired": True},
                }],
            }
        else:
            payload["scenario"]["outcome"] = {"op": "all", "children": [
                payload["scenario"]["outcome"],
                {"predicateId": "PRED-ACTOR-ROOT-RANK", "kind": "proof.winning_root_card_rank_count",
                    "version": 1, "hardness": "hard",
                    "arguments": {"rank": ["jack"], "minimum": 1, "maximum": 1}},
                {"predicateId": "PRED-ACTOR-ROOT-MATCH", "kind": "proof.winning_root_action_matches",
                    "version": 1, "hardness": "hard",
                    "arguments": {"selector": {"ranks": ["jack", "queen"]}}},
            ]}
        recipe = service.create_recipe(
            payload,
            idempotency_key="ortools-actor-information-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key="ortools-actor-information-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == "satisfied"
        assert completed["resultCount"] == completed["qualifiedCount"] == 1
        result = service.get_result(completed["resultCodes"][0])
        assert result is not None
        proof = result["proofOutcome"]
        assert proof["completionStatus"] == "exact_complete"
        assert proof["requestedVisibility"] == "actor_information_set"
        assert proof["achievedVisibility"] == "actor_information_set"
        assert proof["hiddenWorlds"]["count"] == 4
        assert proof["hiddenWorlds"]["complete"] is True
        assert result["researchAuthority"]["qualificationAuthority"] == "cp_sat"

        observer = service.get_exact_state(result["resultCode"], observer_seat="east")
        hands = observer["publicState"]["hands"]
        assert isinstance(hands["east"], list)
        assert all(
            isinstance(hands[seat], dict) and set(hands[seat]) == {"count"}
            for seat in ("north", "south", "west")
        )
        graph = service.artifact_store.get(result["artifacts"]["proofGraph"]["uri"])
        encoded = graph.decode("utf-8")
        assert '"initialHands"' not in encoded
        assert '"hands"' not in encoded

        cold_service, cold_executor = _isolated_service(tmp_path, authority, context)
        try:
            assert not cold_service._result_verification_cache

            def reject_current_execution_context(*_args, **_kwargs):
                pytest.fail("cold V3 reads must use the stored configuration snapshot")

            monkeypatch.setattr(cold_service, "_build_recipe_and_context", reject_current_execution_context)
            restored = cold_service.get_result(result["resultCode"])
            assert restored["resultHash"] == result["resultHash"]
            assert restored["proofOutcome"] == proof
            assert restored["artifacts"] == result["artifacts"]
            assert cold_service.get_exact_state(result["resultCode"], observer_seat="east") == observer
            model = json.loads(cold_service.artifact_store.get(
                restored["artifacts"]["proofModel"]["uri"]))
            assert canonical_sha256(model) == restored["artifacts"]["proofModel"]["sha256"]
            if include_stopping_certificate:
                certificate = model["opponentStopCertificate"]
                assert certificate["completionStatus"] == "exact_complete"
                assert certificate["scope"]["graphHash"] == proof["identities"]["graphHash"]
                assert certificate["sha256"] == canonical_sha256({
                    key: value for key, value in certificate.items() if key != "sha256"})
                # This round already has tricks on both sides: report-only evidence
                # must not invent an applicable tactic or qualify it as interesting.
                assert all(row["robustValue"] == 1 for row in certificate["roots"])
                report = next(row for row in restored["predicateEvaluation"]["evidence"]
                    if row["predicateId"] == "PRED-REPORT-KABOUT-STOP")
                assert report["satisfied"] is False
                assert report["observed"]["applicable"] is False
                assert report["observed"]["verified"] is False
            else:
                assert "opponentStopCertificate" not in model
                observations = {row["predicateId"]: row["observed"]
                    for row in restored["predicateEvaluation"]["evidence"]}
                assert observations["PRED-ACTOR-ROOT-RANK"]["matchingCount"] == 1
                assert {row["card"]["code"] for row in observations["PRED-ACTOR-ROOT-MATCH"]["winningActions"]} == {"DJ", "DQ"}
        finally:
            cold_executor.shutdown(wait=True)
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize(("visibility", "threshold", "expected_count"), [
    ("actor_information_set", -100000, 2),
    ("actor_information_set", 100000, 0),
    ("complete_state", -7, 1),
])
def test_native_root_count_persists_zero_one_and_multiple_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    visibility: str,
    threshold: int,
    expected_count: int,
    inline_v3_shard_runner: None,
) -> None:
    authority, base_context = _frozen_context_and_authority()
    context = replace(
        base_context,
        bundle=_ChallengeBundleProxy(
            base_context.bundle,
            _policy_service_challenge(measured_wall_seconds=420.0),
        ),
    )
    service, worker = _isolated_service(tmp_path, authority, context)
    try:
        if visibility == "actor_information_set":
            payload = _actor_information_exact_recipe_payload(context.bundle_revision_id)
        else:
            payload = _native_qualified_exact_recipe_payload(context.bundle_revision_id)
            payload["proof"]["claim"] = "winning_root_action_count"
            payload["proof"].pop("selectedAction", None)
            payload["resources"]["solverTimeoutSeconds"] = 420.0
        payload["proof"]["successThreshold"] = threshold
        payload["scenario"]["outcome"] = {"op": "all", "children": [
            payload["scenario"]["outcome"],
            {"predicateId": "PRED-EXACT-ROOT-COUNT", "kind": "proof.winning_root_action_count",
             "version": 1, "hardness": "hard",
             "arguments": {"minimum": expected_count, "maximum": expected_count}},
        ]}
        recipe = service.create_recipe(payload)
        run = service._submit_recipe_run(
            recipe["code"], idempotency_key="native-root-count-run", schedule=False,
        )
        service._execute(run["runCode"])
        completed = service.get_run(run["runCode"])
        assert completed["status"] == "completed", completed
        assert completed["scenarioResearchStatus"] == "satisfied", completed
        assert completed["resultCount"] == completed["qualifiedCount"] == 1
        result = service.get_result(completed["resultCodes"][0])
        proof = result["proofOutcome"]
        # Counting zero threshold-satisfying moves is a true claim, not a failed win claim.
        assert proof["claim"] == "winning_root_action_count"
        assert proof["claimSatisfied"] is True
        assert proof["completionStatus"] == "exact_complete"
        assert proof["winningRootActionCount"] == expected_count
        assert proof["winningRootActionLowerBound"] == expected_count
        assert proof["winningRootActionUpperBound"] == expected_count
        assert len(proof["rootActions"]) == 2
        assert sum(row["satisfies"] for row in proof["rootActions"]) == expected_count
        assert result["scenarioProjection"]["proofVisibility"] == visibility
        if visibility == "actor_information_set":
            assert proof["hiddenWorlds"]["count"] == 4
            assert proof["hiddenWorlds"]["complete"] is True
        else:
            assert {row["action"]: row["value"] for row in proof["rootActions"]} == {"HK": -6, "C8": -8}
        assert result["disposition"] == "qualified"
        assert result["runtimeCompatibility"]["status"] == "passed"
        evidence = next(row for row in result["predicateEvaluation"]["evidence"]
                        if row["predicateId"] == "PRED-EXACT-ROOT-COUNT")
        assert evidence["satisfied"] is True
        cold, cold_worker = _isolated_service(tmp_path, authority, context)
        try:
            def reject_live_context(*_args, **_kwargs):
                pytest.fail("root counts must survive reads under the frozen configuration")

            monkeypatch.setattr(cold, "_build_recipe_and_context", reject_live_context)
            restored = cold.get_result(result["resultCode"])
            assert restored == result
            public = api._public_research_result(restored)
            api.ResearchScenarioPublicResultResponse.model_validate(public)
            assert public["proofOutcome"]["winningRootActionCount"] == expected_count
        finally:
            cold_worker.shutdown(wait=True)
    finally:
        worker.shutdown(wait=True)


def test_native_false_root_count_persists_rejected_exact_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, base_context = _frozen_context_and_authority()
    context = replace(
        base_context,
        bundle=_ChallengeBundleProxy(base_context.bundle, _policy_service_challenge()),
    )
    service, worker = _isolated_service(tmp_path, authority, context)
    try:
        payload = _actor_information_exact_recipe_payload(context.bundle_revision_id)
        payload["scenario"]["outcome"] = {"op": "all", "children": [
            payload["scenario"]["outcome"],
            {"predicateId": "PRED-FALSE-ROOT-COUNT", "kind": "proof.winning_root_action_count",
             "version": 1, "hardness": "hard", "arguments": {"minimum": 1, "maximum": 1}},
        ]}
        recipe = service.create_recipe(payload)
        run = service._submit_recipe_run(
            recipe["code"], idempotency_key="native-false-root-count-run", schedule=False,
        )
        service._execute(run["runCode"])
        completed = service.get_run(run["runCode"])
        assert completed["status"] == "completed", completed
        assert completed["scenarioResearchStatus"] == "semantic_postcondition_miss", completed
        assert completed["resultCount"] == completed["qualifiedCount"] == 0
        assert completed["resultCodes"] == []
        assert completed["auditResultCount"] == 1
        result = service.get_result(completed["auditResultCodes"][0])
        assert result["terminalStatus"] == "semantic_postcondition_miss"
        assert result["disposition"] == "rejected"
        assert result["qualification"]["qualified"] is False
        proof = result["proofOutcome"]
        assert proof["completionStatus"] == "exact_complete"
        assert proof["claimSatisfied"] is True
        assert proof["winningRootActionCount"] == 2
        assert proof["winningRootActionLowerBound"] == proof["winningRootActionUpperBound"] == 2
        assert sum(row["satisfies"] for row in proof["rootActions"]) == 2
        assert proof["hiddenWorlds"]["count"] == 4
        assert proof["hiddenWorlds"]["complete"] is True
        evaluation = result["predicateEvaluation"]
        assert evaluation["satisfied"] is False
        assert evaluation["status"] == "target_miss"
        # An infeasible hard expression has an assumption core, not solved evidence variables.
        assert evaluation["namedCore"]
        cold, cold_worker = _isolated_service(tmp_path, authority, context)
        try:
            def reject_live_context(*_args, **_kwargs):
                pytest.fail("rejected exact proofs must survive reads under the frozen configuration")

            monkeypatch.setattr(cold, "_build_recipe_and_context", reject_live_context)
            restored = cold.get_result(result["resultCode"])
            assert restored == result
            assert cold.get_recipe(recipe["code"])["scenario"]["outcome"] == recipe["scenario"]["outcome"]
            public = api._public_research_result(restored)
            api.ResearchScenarioPublicResultResponse.model_validate(public)
            assert public["proofOutcome"]["winningRootActionCount"] == 2
            assert public["disposition"] == "rejected"
            trace_uri = result["artifacts"]["completeTrace"]["uri"]
            original_get = service.artifact_store.get

            def corrupted_trace(uri):
                data = original_get(uri)
                return data + b"\n" if uri == trace_uri else data

            with monkeypatch.context() as corruption:
                corruption.setattr(service.artifact_store, "get", corrupted_trace)
                corruption.setattr(cold.artifact_store, "get", corrupted_trace)
                for reader in (service, cold):
                    with pytest.raises(ValueError, match="Research result artifact integrity check failed"):
                        reader.get_result(result["resultCode"])
                corruption.setattr(api, "RESEARCH", cold)
                with TestClient(api.app) as client:
                    response = client.get(
                        f"/api/v2/research-results/{result['resultCode']}/exact-state",
                        params={"observerSeat": "east"},
                    )
                    assert response.status_code == 409, response.text
                    assert "publicState" not in response.json()
            assert cold.get_result(result["resultCode"]) == result
        finally:
            cold_worker.shutdown(wait=True)
    finally:
        worker.shutdown(wait=True)


def test_search_actor_information_set_run_persists_exhaustive_safe_proof(
    tmp_path: Path,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _recipe(
            authority.to_dict(),
            "ortools_cp_sat_v3",
            trick_index=7,
            solver_timeout_seconds=30.0,
            decision_remaining_cards=1,
            decision_seat="south",
            decision_position=0,
            candidate_count=4,
        ).to_dict()
        payload.pop("sha256", None)
        payload.pop("capabilityPlan", None)
        payload["scenario"].pop("sha256", None)
        payload["scenario"]["constraints"] = {
            "predicateId": "PRED-SEARCH-ACTOR-SUN",
            "kind": "auction.contract",
            "version": 1,
            "hardness": "hard",
            "arguments": {"contract": "sun"},
        }
        payload["proof"] = {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "actor_information_set",
            "strategyClass": "pure_deterministic",
            "claim": "winning_root_action_count",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -100000,
            "budgetProfile": "actor_information_set_exact_v1",
        }
        payload["resources"]["proofBudgetProfile"] = (
            "actor_information_set_exact_v1"
        )
        recipe = service.create_recipe(
            payload,
            idempotency_key="ortools-search-actor-information-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key="ortools-search-actor-information-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == "satisfied", completed[
            "events"
        ]
        result = service.get_result(completed["resultCodes"][0])
        proof = result["proofOutcome"]
        assert proof["completionStatus"] == "exact_complete"
        assert proof["achievedVisibility"] == "actor_information_set"
        assert proof["hiddenWorlds"]["complete"] is True
        assert proof["hiddenWorlds"]["count"] <= 6
        assert result["runtimeCompatibility"]["status"] == "passed"
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize("ratified_projects", (False, True))
def test_exact_actor_api_uses_frozen_profile_for_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ratified_projects: bool,
) -> None:
    if ratified_projects:
        from baloot_platform.config import build_bundle
        from baloot_platform.configuration import SQLiteConfigurationStore, VersionedConfigurationService
        from baloot_platform.research.execution_context import ResearchExecutionContextFactoryV1

        root = Path(__file__).resolve().parents[1]
        configuration = VersionedConfigurationService(
            SQLiteConfigurationStore(tmp_path / "configuration.sqlite3"), root=root,
            profile=yaml.safe_load((root / "configs/configuration/control-plane.yaml").read_text(encoding="utf-8")),
        )
        configuration.bootstrap_packaged_bundle(actor_id="TEST-DEFAULT")
        selected = configuration.bootstrap_packaged_bundle(
            actor_id="TEST-RATIFIED-PROJECTS",
            bundle_path="configs/bundle_ortools_research_ratified_projects.yaml",
            scope="environment", environment="private-v8-actor-api",
        )
        template = api.RESEARCH.execution_context_factory
        factory_options = dict(
            configuration_service=configuration,
            artifact_store=LocalContentAddressedArtifactStore(tmp_path / "artifacts"),
            research_store=SQLiteResearchStoreV1(tmp_path / "research.sqlite3"),
            root=root, plugin_capabilities=template.plugin_capabilities,
            shared_plugin_descriptors=template.shared_plugin_descriptors,
            active_bundle=build_bundle(root),
            active_exact_search_backend=template.active_exact_search_backend,
        )
        factory = ResearchExecutionContextFactoryV1(**factory_options)
        authority, context = factory.freeze_selector(
            bundle_revision_id=selected["bundle"]["revisionId"], analysis_mode="cascaded",
        )
        assert context.bundle.rules.id == "saudi_federation_2023_v8"
        assert configuration.resolve()["profiles"]["rules"]["profile"]["id"] != context.bundle.rules.id
    else:
        authority, legacy_context = _frozen_context_and_authority()
        context = replace(
            legacy_context,
            bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
        )
    service, executor = _isolated_service(tmp_path, authority, context)
    if ratified_projects:
        service.execution_context_factory = factory
    monkeypatch.setattr(api, "RESEARCH", service)
    payload = _actor_information_exact_recipe_payload(context.bundle_revision_id)
    try:
        with TestClient(api.app) as client:
            validation = client.post("/api/v2/research-recipes:validate", json=payload)
            assert validation.status_code == 200, validation.text
            assert validation.json()["valid"] is True, validation.json()["issues"]

            estimate = client.post("/api/v2/research-recipes:estimate", json=payload)
            assert estimate.status_code == 200, estimate.text
            estimate_document = estimate.json()
            assert estimate_document["launchAdmissible"] is True
            assert estimate_document["requestedCombinationSupported"] is True
            assert estimate_document["estimatedHiddenWorlds"] >= 2
            assert estimate_document["estimatedGraphNodes"] is not None
            assert estimate_document["estimatedGraphEdges"] is not None
            assert (
                estimate_document["proof"]["estimatedHiddenWorldUpperBound"]
                == estimate_document["estimatedHiddenWorlds"]
            )

            created = client.post(
                "/api/v2/research-recipes",
                json=payload,
                headers={"Idempotency-Key": "ortools-actor-information-api-create"},
            )
            assert created.status_code == 201, created.text
            assert created.json()["configuration"]["identities"]["rules"]["sha256"] == context.bundle.rules.sha256
            run = client.post(
                "/api/v2/research-runs",
                json={"recipeCode": created.json()["code"]},
                headers={"Idempotency-Key": "ortools-actor-information-api-run"},
            )
            assert run.status_code == 202, run.text

            executor.shutdown(wait=True)
            completed = client.get(f"/api/v2/research-runs/{run.json()['runCode']}")
            assert completed.status_code == 200, completed.text
            assert completed.json()["status"] == "completed", completed.json().get("events")
            if ratified_projects:
                # Legal historical input is not new-profile quality evidence:
                # South's S9 at index 4 loses four bounded-policy points.
                assert completed.json()["scenarioResearchStatus"] == "semantic_postcondition_miss", completed.json()
                assert completed.json()["qualifiedCount"] == 0
                assert completed.json()["campaignAccounting"]["stopReason"] != "global_campaign_timeout"
                campaign = json.loads(service.artifact_store.get(completed.json()["scenarioResearchEvidence"]["uri"]))
                assert len(campaign["executionEvidence"]) == 1
                execution = json.loads(service.artifact_store.get(campaign["executionEvidence"][0]["uri"]))
                receipt = execution["executionReceipt"]
                assert receipt["candidateAttemptReason"] == "quality_recorded_audit_target_miss"
                partial = receipt["incompleteCompilation"]
                assert partial["qualityStatus"] == "quality_failed"
                assert partial["qualityFailureReason"] == "recorded_action_not_optimal"
                decisions = partial["qualityEvidence"]["postAuction"]["projectPlay"]["decisions"]
                plays = [row for row in decisions if row["decisionKind"] == "play"]
                assert [row["playIndex"] for row in plays] == list(range(5))
                assert {key: plays[-1][key] for key in (
                    "playIndex", "actor", "action", "actualActionOptimal", "regret",
                )} == {"playIndex": 4, "actor": "south", "action": "S9", "actualActionOptimal": False, "regret": 4}
                return
            assert completed.json()["qualifiedCount"] == 1, completed.json()
            result_code = completed.json()["resultCodes"][0]
            result = client.get(f"/api/v2/research-results/{result_code}")
            assert result.status_code == 200, result.text
            public_result = result.json()
            internal_result = service.get_result(result_code)
            assert internal_result["researchAuthority"]["backendId"] == "ortools_cp_sat_v3"
            assert not {
                "resultHash", "traceHash", "gameHash", "artifacts", "provenance",
                "researchAuthority", "executionReceipt", "scenarioSatisfaction",
            } & public_result.keys()
            proof = public_result["proofOutcome"]
            assert not {"principalVariation", "artifactSha256", "semanticSha256"} & proof.keys()
            assert proof["completionStatus"] == "exact_complete"
            assert proof["achievedVisibility"] == "actor_information_set"
            assert proof["hiddenWorlds"]["complete"] is True
            assert proof["hiddenWorlds"]["count"] >= 2
            assert proof["hiddenWorlds"]["count"] == 4

            observer = client.get(
                f"/api/v2/research-results/{result_code}/exact-state",
                params={"observerSeat": "east"},
            )
            assert observer.status_code == 200, observer.text
            hands = observer.json()["publicState"]["hands"]
            assert isinstance(hands["east"], list)
            assert all(
                isinstance(hands[seat], dict) and set(hands[seat]) == {"count"}
                for seat in ("north", "south", "west")
            )
            private_identifiers = (
                internal_result["resultHash"],
                internal_result["scenarioProjection"]["exactStateId"],
                internal_result["artifacts"]["completeTrace"]["sha256"],
                internal_result["artifacts"]["proofOutcome"]["sha256"],
            )
            for document in (public_result, observer.json()):
                encoded = json.dumps(document, sort_keys=True)
                assert "exactStateId" not in encoded
                assert "stateHash" not in encoded
                assert all(identifier not in encoded for identifier in private_identifiers)

            unsupported = copy.deepcopy(payload)
            unsupported["proof"].update(
                {
                    "claim": "winning_continuation_line_count",
                    "lineUniverse": "all_legal",
                }
            )
            rejected = client.post(
                "/api/v2/research-recipes:estimate", json=unsupported
            )
            assert rejected.status_code == 200, rejected.text
            assert rejected.json()["launchAdmissible"] is False
            assert rejected.json()["proof"]["reason"] == (
                "actor_information_set_continuation_count_unsupported"
            )

            run_code = run.json()["runCode"]
            campaign_reference = copy.deepcopy(service.get_run(run_code)["scenarioResearchEvidence"])
            try:
                service.store.update_run(
                    run_code,
                    scenarioResearchEvidence={**campaign_reference, "sha256": "0" * 64},
                )
                with pytest.raises(ValueError, match="Research result CAS artifact integrity check failed"):
                    client.get(f"/api/v2/research-results/{result_code}")
            finally:
                service.store.update_run(run_code, scenarioResearchEvidence=campaign_reference)
            assert client.get(f"/api/v2/research-results/{result_code}").status_code == 200
    finally:
        executor.shutdown(wait=True)


def test_incomplete_actor_information_set_persists_nullable_count_and_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inline_v3_shard_runner: None,
) -> None:
    def unknown(_solver, _model, _deadline, statuses, _cancel_check=None):
        statuses.append(cp_model.UNKNOWN)
        return False

    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    payload = _actor_information_exact_recipe_payload(context.bundle_revision_id)
    payload["scenario"]["outcome"] = {"op": "all", "children": [
        payload["scenario"]["outcome"],
        {"predicateId": "PRED-ACTOR-INCOMPLETE-ROOT-COUNT", "kind": "proof.winning_root_action_count",
         "version": 1, "hardness": "hard", "arguments": {"minimum": 0, "maximum": 2}},
    ]}
    payload["proof"].update(claim="winning_root_action_count")
    payload["proof"].pop("selectedAction", None)
    request_hash = ScenarioProofRequestV1.build(payload["proof"]).sha256
    original = executor_module.prove_actor_information_set_v1
    interrupted_requests = []

    def incomplete(compilation, request, **kwargs):
        if request.sha256 != request_hash:
            return original(compilation, request, **kwargs)
        interrupted_requests.append(request.sha256)
        with monkeypatch.context() as patch:
            patch.setattr(OrToolsActorInformationSetProofV1, "_solve_optimal", staticmethod(unknown))
            return original(compilation, request, **kwargs)

    monkeypatch.setattr(executor_module, "prove_actor_information_set_v1", incomplete)
    try:
        recipe = service.create_recipe(
            payload,
            idempotency_key="ortools-actor-information-incomplete-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key="ortools-actor-information-incomplete-run",
            schedule=False,
        )

        service._execute(run["runCode"])
        assert interrupted_requests == [request_hash]

        completed = service.get_run(run["runCode"])
        assert completed is not None
        assert completed["scenarioResearchStatus"] == "proof_incomplete"
        result = service.get_result(completed["auditResultCodes"][0])
        assert result is not None
        trace = json.loads(service.artifact_store.get(result["artifacts"]["completeTrace"]["uri"]))
        assert trace["playingQualityEvidence"]["receipt"]["status"] == "complete"
        proof = result["proofOutcome"]
        assert proof["hiddenWorlds"]["complete"] is True
        assert proof["hiddenWorlds"]["count"] > 0
        assert proof["winningRootActionCount"] is None
        assert proof["winningRootActionLowerBound"] == 0
        assert proof["winningRootActionUpperBound"] == len(proof["rootActions"])
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize("decision_kind", ("bid", "escalation"))
def test_nonplay_actor_information_service_refuses_launch_until_substantive_evidence(
    tmp_path: Path,
    decision_kind: str,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _nonplay_actor_information_recipe_payload(
            context.bundle_revision_id,
            decision_kind,
        )
        recipe = service.create_recipe(
            payload,
            idempotency_key=f"ortools-{decision_kind}-actor-info-create",
        )
        with pytest.raises(ResearchLaunchAdmissionError) as rejected:
            service._submit_recipe_run(
                recipe["code"],
                idempotency_key=f"ortools-{decision_kind}-actor-info-run",
                schedule=False,
            )
        assert rejected.value.diagnostic["reason"] == (
            "phase4_substantive_full_service_evidence_missing"
        )
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize("decision_kind", ("bid", "escalation"))
def test_nonplay_actor_information_api_refuses_launch_until_substantive_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision_kind: str,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    monkeypatch.setattr(api, "RESEARCH", service)
    payload = _nonplay_actor_information_recipe_payload(
        context.bundle_revision_id,
        decision_kind,
    )
    try:
        with TestClient(api.app) as client:
            validation = client.post("/api/v2/research-recipes:validate", json=payload)
            assert validation.status_code == 200, validation.text
            assert validation.json()["valid"] is True, validation.json()["issues"]

            created = client.post(
                "/api/v2/research-recipes",
                json=payload,
                headers={
                    "Idempotency-Key": f"ortools-{decision_kind}-actor-info-api-create"
                },
            )
            assert created.status_code == 201, created.text
            run = client.post(
                "/api/v2/research-runs",
                json={"recipeCode": created.json()["code"]},
                headers={
                    "Idempotency-Key": f"ortools-{decision_kind}-actor-info-api-run"
                },
            )
            assert run.status_code == 409, run.text
            assert run.json()["detail"]["reason"] == (
                "phase4_substantive_full_service_evidence_missing"
            )
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize("decision_kind", ("bid", "escalation"))
def test_nonplay_line_count_service_rejects_until_substantive_evidence(
    tmp_path: Path,
    decision_kind: str,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _nonplay_exact_line_count_recipe_payload(
            context.bundle_revision_id,
            decision_kind,
        )
        recipe = service.create_recipe(
            payload,
            idempotency_key=f"ortools-{decision_kind}-line-count-create",
        )
        with pytest.raises(ResearchLaunchAdmissionError) as rejected:
            service._submit_recipe_run(
                recipe["code"],
                idempotency_key=f"ortools-{decision_kind}-line-count-run",
                schedule=False,
            )
        assert rejected.value.diagnostic["reason"] == (
            "phase4_substantive_full_service_evidence_missing"
        )
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize(
    ("failure_kind", "failure_code"),
    (
        ("callback_exception", "runtime_compatibility_callback_failed"),
        ("malformed_exact_state", "exact_state_document_malformed"),
    ),
)
def test_search_v4_run_quarantines_downstream_compatibility_without_losing_cp_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    failure_code: str,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )

    def failed_check(**kwargs):
        if failure_kind == "callback_exception":
            raise RuntimeError("downstream runtime unavailable")
        return check_runtime_compatibility_v1(**{**kwargs, "exact_state": {}})

    monkeypatch.setattr(
        "baloot_platform.research.service.check_runtime_compatibility_v1",
        failed_check,
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        recipe = service.create_recipe(
            _recipe(
                authority.to_dict(),
                "ortools_cp_sat_v3",
                trick_index=6,
                solver_timeout_seconds=30.0,
            ).to_dict(),
            idempotency_key=f"ortools-quarantine-{failure_kind}-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key=f"ortools-quarantine-{failure_kind}-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == "satisfied"
        assert completed["qualifiedCount"] == completed["resultCount"] == 1
        assert completed["quarantineCount"] == 1
        result = service.get_result(completed["resultCodes"][0])
        assert result is not None
        assert result["disposition"] == "qualified"
        assert result["qualification"]["qualified"] is True
        assert result["runtimeCompatibility"]["status"] == "failed"
        assert result["runtimeCompatibility"]["failureCodes"] == [failure_code]
        assert (
            result["runtimeCompatibilityEvidence"]["semanticResultHash"]
            == (result["resultHash"])
        )
        with pytest.raises(
            RuntimeCompatibilityAdmissionError,
            match="playable or video admission",
        ):
            service.get_exact_state(result["resultCode"], observer_seat="west")
    finally:
        executor.shutdown(wait=True)


def test_search_v4_run_lifts_and_persists_one_qualified_exact_state(
    tmp_path: Path,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    challenge = _ortools_challenge()
    # Measured full campaign: 60.47s wall / 43.83s CPU at the local cap.
    # Lead-approved 75s fixture campaign; proof budgets remain unchanged.
    challenge["scenarioResearchV4"]["searchBudgets"]["profiles"]["late_game_exact_v1"]["maximumWallSeconds"] = 75.0
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, challenge),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _recipe(
            authority.to_dict(),
            "ortools_cp_sat_v3",
            trick_index=6,
            solver_timeout_seconds=30.0,
        ).to_dict()
        payload.pop("sha256", None)
        recipe = service.create_recipe(
            payload,
            idempotency_key="ortools-search-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key="ortools-search-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == "satisfied", {
            "yield": completed.get("scenarioYieldCounts"),
            "latestIncomplete": completed.get("campaignAccounting", {}).get("latestIncompleteCompilation"),
            "database": str(tmp_path / "research.sqlite3"),
        }
        assert completed["resultCount"] == completed["qualifiedCount"] == 1
        assert completed["performanceReceipt"]["completedCandidates"] == 1
        result = service.get_result(completed["resultCodes"][0])
        assert result is not None
        assert result["disposition"] == "qualified"
        assert result["runtimeCompatibility"]["status"] == "passed"
        construction = json.loads(
            service.artifact_store.get(
                result["artifacts"]["constructionModel"]["uri"]
            )
        )
        assert construction["schemaVersion"] == 2
        assert construction["backendVersion"] == executor_module._V3_BACKEND_VERSION
        assert set(construction["compiledStages"]) <= set(
            construction["executedStages"]
        )
        assert all(
            receipt["schemaVersion"] == 2
            and receipt["variableCount"] >= 0
            and receipt["constraintCount"] >= 0
            for receipt in construction["stageReceipts"].values()
        )
        core_stage_sizes = {
            stage: (
                construction["stageReceipts"][stage]["variableCount"],
                construction["stageReceipts"][stage]["constraintCount"],
            )
            for stage in {
                "deal",
                "auction",
                "escalation",
                "projects",
                "playPrefix",
                "state",
                "outcome",
                "proof",
                "tactics",
            }
            & set(construction["stageReceipts"])
        }
        assert all(
            construction["stageReceipts"][stage]["variableCount"] > 0
            and construction["stageReceipts"][stage]["constraintCount"] > 0
            for stage in core_stage_sizes
            if stage not in {"escalation", "projects"}
        ), json.dumps(core_stage_sizes, sort_keys=True)
        assert all(
            core_stage_sizes[stage] == (0, 0)
            for stage in {"escalation", "projects"} & set(core_stage_sizes)
        )
        assert any(
            receipt["evidenceBindings"]
            for receipt in construction["stageReceipts"].values()
        )
        assert result["scenarioProjection"]["exactStateId"].startswith("STATE-")
        observer = result["libraryProjection"]["actingSeats"][0]
        exact_state = service.get_exact_state(
            result["resultCode"],
            observer_seat=observer,
        )
        assert exact_state["decisionReference"]["resultCode"] == result["resultCode"]
        assert (
            exact_state["publicProjectionHash"]
            == result["scenarioProjection"]["publicProjectionHash"]
        )
    finally:
        executor.shutdown(wait=True)


def test_search_bounded_play_run_persists_honest_boundary(
    tmp_path: Path,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _recipe(
            authority.to_dict(),
            "ortools_cp_sat_v3",
            trick_index=6,
            solver_timeout_seconds=30.0,
        ).to_dict()
        payload.pop("sha256", None)
        payload.pop("capabilityPlan", None)
        payload["scenario"].pop("sha256", None)
        payload["scenario"]["constraints"].pop("sha256", None)
        payload["scenario"]["constraints"]["children"] = [{
            "predicateId": "PRED-DECISION-LEGAL-CHOICES",
            "kind": "state.legal_action_count",
            "version": 1,
            "hardness": "hard",
            "arguments": {"minimum": 1, "maximum": 8},
        }]
        payload["proof"] = {
            "schemaVersion": 1,
            "mode": "bounded",
            "visibility": "complete_state",
            "strategyClass": "pure_deterministic",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "round_win",
            "successThreshold": -100000,
            "horizon": {"unit": "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1",
            "budgetProfile": "late_game_exact_v1",
        }
        recipe = service.create_recipe(
            payload,
            idempotency_key="ortools-search-play-bounded-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key="ortools-search-play-bounded-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == "satisfied", completed[
            "events"
        ]
        result = service.get_result(completed["resultCodes"][0])
        assert result["proofOutcome"]["completionStatus"] == "bounded_complete"
        assert result["proofOutcome"]["boundaryEvaluator"] == (
            "current_raw_point_margin_v1"
        )
        assert result["runtimeCompatibility"]["status"] == "passed"
        assert result["disposition"] == "qualified"
        assert any(
            row["predicateId"] == "PRED-DECISION-LEGAL-CHOICES"
            and row["satisfied"]
            and row["observed"]["legalActionCount"] >= 1
            for row in result["predicateEvaluation"]["evidence"]
        )
    finally:
        executor.shutdown(wait=True)


def test_search_policy_play_run_persists_complete_continuation(
    tmp_path: Path,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _recipe(
            authority.to_dict(),
            "ortools_cp_sat_v3",
            trick_index=6,
            solver_timeout_seconds=30.0,
        ).to_dict()
        payload.pop("sha256", None)
        payload.pop("capabilityPlan", None)
        payload["proof"] = {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "round_win",
            "successThreshold": -1,
            "policies": _fixed_policy_templates(),
            "budgetProfile": "late_game_exact_v1",
        }
        recipe = service.create_recipe(
            payload,
            idempotency_key="ortools-search-play-policy-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key="ortools-search-play-policy-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == "satisfied", completed[
            "events"
        ]
        result = service.get_result(completed["resultCodes"][0])
        trace = json.loads(
            service.artifact_store.get(result["artifacts"]["completeTrace"]["uri"])
        )
        assert result["proofOutcome"]["completionStatus"] == "policy_complete"
        assert len(trace["completeExactTracePrefix"]["playActions"]) == 32
        assert trace["completeExactNormalization"]["phase"] == "scored"
        assert result["runtimeCompatibility"]["status"] == "passed"
        assert result["disposition"] == "qualified"
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize("incomplete_scenario", [False, True])
@pytest.mark.parametrize("policy_version", [1, 2])
def test_search_v4_unfixed_play_position_retains_owned_quality_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, incomplete_scenario: bool, policy_version: int,
    inline_v3_shard_runner: None,
) -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/research-native-qualified-round-v3.json").read_text(encoding="utf-8")
    )
    original_model = executor_module.OrToolsScenarioModelV1

    def fixed_deal(*args, **kwargs):
        model = original_model(*args, **kwargs)
        # Pin construction only; hidden-world normalization must remain independent.
        if "excluded_deck_orders" in kwargs:
            for variable, card in zip(
                model.card_at_position, fixture["exactTracePrefix"]["deckOrder"], strict=True,
            ):
                model.model.add(variable == card)
        return model

    monkeypatch.setattr(executor_module, "OrToolsScenarioModelV1", fixed_deal)
    if incomplete_scenario:
        from baloot_platform.research import service as service_module

        original_writer = service_module.build_ortools_research_result_v3

        def incomplete_writer(**kwargs):
            rejected = original_writer(**{**kwargs, "terminal_status": "proof_incomplete"})
            receipt = kwargs["artifact_documents"]["completeTrace"]["playingQualityEvidence"]["receipt"]
            assert rejected["disposition"] == "rejected"
            assert rejected["qualification"]["failures"] == ["proof_incomplete"]
            assert rejected["qualification"]["allDecisionPolicyAuditComplete"] is True
            assert rejected["qualification"]["playingQualityEvidenceHash"] == receipt["sha256"]
            return original_writer(**kwargs)

        monkeypatch.setattr(service_module, "build_ortools_research_result_v3", incomplete_writer)
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _recipe(
            authority.to_dict(), "ortools_cp_sat_v3", trick_index=6,
            decision_remaining_cards=2, decision_seat="south",
            solver_timeout_seconds=60.0,
            budget_profile="offline_heavyweight_v1",
        ).to_dict()
        payload["resources"]["seedStart"] = 3
        payload.pop("sha256", None)
        if policy_version == 2:
            payload["playingQuality"].pop("sha256", None)
            payload["playingQuality"].update(schemaVersion=2, visibility="actor_information_set")
        assert "position" not in payload["scenario"]["decisionPoint"]["selector"]
        recipe = service.create_recipe(payload, idempotency_key="unfixed-position-create")
        run = service._submit_recipe_run(
            recipe["code"], idempotency_key="unfixed-position-run", schedule=False,
        )
        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None and completed["qualifiedCount"] == 1, completed
        result = service.get_result(completed["resultCodes"][0])
        assert result is not None and result["disposition"] == "qualified"
        trace = json.loads(service.artifact_store.get(result["artifacts"]["completeTrace"]["uri"]))
        receipt = trace["playingQualityEvidence"]["receipt"]
        assert receipt["scope"] == "all_decisions" and receipt["status"] == "complete"
        assert result["qualification"]["allDecisionPolicyAuditComplete"] is True
        assert result["qualification"]["playingQualityMode"] == "budgeted"
        if policy_version == 2:
            assert result["qualification"]["playingQualityVisibility"] == "actor_information_set"
            assert result["qualification"]["actionSelectionObserverSafe"] is True
        else:
            assert "playingQualityVisibility" not in result["qualification"]
        assert (
            result["qualification"]["playingQualityEvidenceHash"]
            == receipt["sha256"]
        )
        assert len(trace["completeExactTracePrefix"]["playActions"]) == 32
        trick_seven = trace["completeExactTracePrefix"]["playActions"][24:28]
        assert "south" in [action["actor"] for action in trick_seven]
        assert result["researchAuthority"]["traceHash"] == canonical_sha256(trace)
        attempt = completed["performanceReceipt"]["coordinatorReceipt"]["attempts"][0]
        assert "qualityCheckpoints" not in attempt["partialEvidence"]
        assert attempt["partialEvidence"]["qualityCheckpointHashes"]
        campaign = json.loads(
            service.artifact_store.get(completed["scenarioResearchEvidence"]["uri"])
        )
        execution = json.loads(
            service.artifact_store.get(campaign["executionEvidence"][0]["uri"])
        )
        assert execution["candidate"]["trace"]["playingQualityEvidence"]["receipt"]["status"] == "complete"
        from baloot_platform.research import result_authority

        quality_checks = []
        original_quality_validator = result_authority.validate_trace_quality_evidence_v1

        def count_quality_validation(*args, **kwargs):
            quality_checks.append(kwargs)
            return original_quality_validator(*args, **kwargs)

        # Force a cold read; repeated reads may reuse the verified immutable proof.
        service._result_verification_cache.clear()
        with monkeypatch.context() as audit_patch:
            audit_patch.setattr(result_authority, "validate_trace_quality_evidence_v1", count_quality_validation)
            assert service.get_playing_quality_audit(result["resultCode"])["qualityReceiptHash"] == receipt["sha256"]
        assert len(quality_checks) == 1
        assert quality_checks[0]["required_policy"] is not None
        assert quality_checks[0]["required_search_profiles"]
        assert result["scenarioProjection"]["decisionKind"] == "play"
        assert result["libraryProjection"]["actingSeats"] == ["south"]
    finally:
        executor.shutdown(wait=True)


def test_search_v4_hard_sun_contract_is_constructed_before_qualification(
    tmp_path: Path,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        payload = _recipe(
            authority.to_dict(),
            "ortools_cp_sat_v3",
            trick_index=6,
            solver_timeout_seconds=30.0,
        ).to_dict()
        payload["scenario"]["constraints"]["children"].append(
            {
                "predicateId": "PRED-ORTOOLS-CONTRACT-SUN",
                "kind": "auction.contract",
                "version": 1,
                "hardness": "hard",
                "arguments": {"contract": "sun"},
            }
        )
        payload.pop("sha256", None)
        payload.pop("capabilityPlan", None)
        payload["scenario"].pop("sha256", None)
        payload["scenario"]["constraints"].pop("sha256", None)
        recipe = service.create_recipe(
            payload,
            idempotency_key="ortools-search-sun-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key="ortools-search-sun-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None
        assert completed["scenarioResearchStatus"] == "satisfied"
        assert completed["qualifiedCount"] == 1
        result = service.get_result(completed["resultCodes"][0])
        assert result is not None
        observer = result["libraryProjection"]["actingSeats"][0]
        exact_state = service.get_exact_state(
            result["resultCode"],
            observer_seat=observer,
        )
        assert exact_state["publicState"]["contract"] == "sun"
        campaign = json.loads(
            service.artifact_store.get(completed["scenarioResearchEvidence"]["uri"])
        )
        execution = json.loads(
            service.artifact_store.get(campaign["executionEvidence"][0]["uri"])
        )
        assert execution["candidate"]["trace"]["contract"] == "sun"
        evidence = {
            item["predicateId"]: item
            for item in execution["predicateEvaluation"]["evidence"]
        }
        assert evidence["PRED-ORTOOLS-CONTRACT-SUN"]["satisfied"] is True
        assert evidence["PRED-ORTOOLS-CONTRACT-SUN"]["observed"] == {"contract": "sun"}
    finally:
        executor.shutdown(wait=True)


@pytest.mark.real_clock
def test_search_v4_full_ui_selector_campaign_persists_five_distinct_qualified_results(
    tmp_path: Path,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        recipe = service.create_recipe(
            _recipe(
                authority.to_dict(),
                "ortools_cp_sat_v3",
                trick_index=6,
                solver_timeout_seconds=60.0,
                result_count=5,
                candidate_count=50,
                seeds_per_shard=int(
                    _ortools_challenge()["scenarioResearchV4"]["authoringDefaults"][
                        "resources"
                    ]["seedsPerShard"]
                ),
                decision_remaining_cards=2,
                decision_seat="south",
                budget_profile="offline_heavyweight_v1",
            ).to_dict(),
            idempotency_key="ortools-search-five-create",
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key="ortools-search-five-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == "satisfied", {
            "yield": completed.get("scenarioYieldCounts"),
            "latestIncomplete": completed.get("campaignAccounting", {}).get("latestIncompleteCompilation"),
            "database": str(tmp_path / "research.sqlite3"),
        }
        assert completed["resultCount"] == completed["qualifiedCount"] == 5
        assert completed["selectedCount"] == 5
        assert len(completed["resultCodes"]) == 5
        assert len(set(completed["resultCodes"])) == 5
        accounting = completed["campaignAccounting"]
        assert accounting["qualified"] == 5
        assert 5 <= accounting["candidateAttempts"] <= 50
        assert accounting["constructedCandidates"] == accounting["candidateAttempts"]
        # Concurrent workers may finish extra qualified candidates before the
        # coordinator cancels them after selecting the requested five.
        assert len(accounting["provisionalResultHashes"]) >= 5
        assert len(accounting["provisionalResults"]) == 5
        receipt_counts = accounting["candidateReceiptCounts"]
        assert receipt_counts["qualified"] >= 5
        assert receipt_counts["duplicate"] == 0
        # Speculative shards can finish incomplete after the target is met;
        # their live receipts remain visible without changing qualification.
        assert receipt_counts["incomplete"] >= accounting["proofIncomplete"]
        # Concurrent workers may finish speculative candidates before the
        # coordinator cancels them after the canonical target is satisfied.
        assert receipt_counts["rejected"] >= accounting["semanticPostconditionMiss"]
        assert sum(receipt_counts.values()) == accounting["candidateReceiptCount"]
        # Speculative workers can report constructed candidates before target
        # satisfaction cancels them without a returned execution/terminal receipt.
        # Those observed attempts must not disappear from work accounting.
        completed_executions = completed["performanceReceipt"]["completedCandidates"]
        assert completed_executions <= accounting["candidateAttempts"] <= 50
        assert accounting["candidateReceiptCount"] >= completed_executions
        assert len(accounting["candidateReceipts"]) == accounting["candidateReceiptCount"]
        assert all(
            (
                item["classification"] == item["status"] == "qualified"
                and item["candidateSummary"]["candidateHash"]
                == item["candidateHash"]
            )
            or (
                item["classification"] == "rejected"
                and item["status"] in {"semantic_postcondition_miss", "result_count_reached"}
            )
            or (
                item["classification"] == "incomplete"
                and item["status"] == "incomplete"
            )
            for item in accounting["candidateReceipts"]
        )
        assert all(
            set(item)
            == {
                "schemaVersion",
                "candidateHash",
                "exactStateId",
                "decisionKind",
                "contract",
                "actingSeat",
                "proofCompletionStatus",
                "winningRootActionCount",
                "selectedAction",
            }
            for item in accounting["provisionalResults"]
        )
        assert accounting["stopReason"] == "qualified_target_satisfied"
        assert accounting["maximumSolveSeconds"] == 60.0
        assert accounting["maximumCampaignSeconds"] == 600.0
        terminal_events = [
            event
            for event in completed["events"]
            if event.get("type") == "progress"
            and event.get("currentStage") in {"candidate_terminal", "variant_selection_terminal"}
        ]
        assert {
            event["candidateReceipt"]["receiptId"]
            for event in terminal_events
            if isinstance(event.get("candidateReceipt"), dict)
            and (event.get("currentStage") == "variant_selection_terminal"
                 or isinstance(event["candidateReceipt"].get("executionOrdinal"), int))
        } == {
            item["receiptId"] for item in accounting["candidateReceipts"]
        }
        assert any(
            event.get("type") == "progress"
            and event.get("currentStage") == "candidate_terminal"
            and event.get("accounting", {}).get("provisionalResultHashes")
            and event.get("accounting", {}).get("provisionalResults")
            and event.get("candidateReceipt", {}).get("classification") == "qualified"
            for event in completed["events"]
        )
        receipt = completed["performanceReceipt"]
        assert receipt["status"] == "completed"
        coordinator_receipt = receipt["coordinatorReceipt"]
        assert coordinator_receipt["budget"]["maximumWallTimeSeconds"] == 600.0
        assert len(coordinator_receipt["shards"]) == 50
        assert coordinator_receipt["workerDeclaration"] == {
            "shardWorkerCount": 1,
            "cpSatWorkersPerSolve": 1,
        }
        assert receipt["symmetryPolicyId"] == "rules_preserving_v1"
        assert receipt["distancePolicyId"] == SEMANTIC_DISTANCE_POLICY_V1
        assert receipt["minimumDistance"] == 2
        principal_variations = [
            item["semanticDocument"]["principalVariationActionIds"]
            for item in coordinator_receipt["acceptedVariants"]
        ]
        assert all(principal_variations)
        assert len({tuple(actions) for actions in principal_variations}) == 5
        assert len(receipt["acceptedExecutionHashes"]) == 5
        assert (
            len(receipt["noGoodVectors"])
            == receipt["completedCandidates"]
        )
        assert len({tuple(item) for item in receipt["noGoodVectors"]}) == len(
            receipt["noGoodVectors"]
        )
        results = [service.get_result(code) for code in completed["resultCodes"]]
        assert all(item is not None for item in results)
        assert len({item["resultHash"] for item in results}) == 5
        assert (
            len({item["scenarioProjection"]["exactStateId"] for item in results}) == 5
        )
        assert {item["runtimeCompatibility"]["status"] for item in results} == {
            "passed"
        }
        for result in results:
            trace = json.loads(
                service.artifact_store.get(result["artifacts"]["completeTrace"]["uri"])
            )
            receipt = trace["playingQualityEvidence"]["receipt"]
            prefix = receipt["finalPrefix"]
            post = receipt["postAuction"]
            assert receipt["scope"] == "all_decisions"
            assert receipt["status"] == "complete"
            assert prefix == trace["completeExactTracePrefix"]
            audit = service.get_playing_quality_audit(result["resultCode"])
            expected_counts = {
                "bid": len(prefix["auctionActions"]),
                "escalation": len(post["escalation"]["decisions"]),
                "project": len(prefix["projectActions"]),
                "play": len(prefix["playActions"]),
            }
            assert audit["countsByKind"] == expected_counts
            assert len(prefix["playActions"]) == 32
            assert all(
                row["actualActionOptimal"] is True and row["regret"] == 0
                for row in audit["decisions"]
            )
            assert audit["qualityReceiptHash"] == receipt["sha256"]
            assert audit["decisionCount"] == sum(expected_counts.values())
            assert audit["optimalDecisionCount"] == audit["decisionCount"]
            assert audit["zeroRegretDecisionCount"] == audit["decisionCount"]
            assert result["researchAuthority"]["traceHash"] == canonical_sha256(trace)
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize(
    ("executor_status", "expected_status", "expected_attempted"),
    (
        ("not_found", "not_found_within_budget", 0),
        ("semantic_postcondition_miss", "semantic_postcondition_miss", 1),
    ),
)
def test_search_v4_terminal_outcomes_persist_without_qualified_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inline_v3_shard_runner: None,
    executor_status: str,
    expected_status: str,
    expected_attempted: int,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    backend = service._cached_ortools_scenario_backend_v2(context)
    assert backend is not None
    try:
        payload = _recipe(
            authority.to_dict(),
            "ortools_cp_sat_v3",
            trick_index=6,
        ).to_dict()
        recipe = service.create_recipe(
            payload,
            idempotency_key=f"ortools-{executor_status}-create",
        )
        execution = _terminal_search_execution(
            executor_status,
            backend.identity.predicate_catalog_hash,
            rules=context.bundle.rules,
            bundle_revision_id=context.bundle_revision_id,
            bundle_hash=context.bundle.sha256,
            scenario_hash=recipe["scenario"]["sha256"],
        )
        monkeypatch.setattr(
            "baloot_platform.research.scenario_backend.OrToolsUnifiedResearchExecutorV1.execute",
            lambda *_args, **_kwargs: execution,
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key=f"ortools-{executor_status}-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None
        assert completed["status"] == "completed", json.dumps(
            completed.get("events", []), ensure_ascii=True
        )
        assert completed["scenarioResearchStatus"] == expected_status
        assert completed["resultCount"] == completed["qualifiedCount"] == 0
        assert completed["resultCodes"] == []
        assert completed["auditResultCount"] == 1
        assert len(completed["auditResultCodes"]) == 1
        audit_result = service.get_result(completed["auditResultCodes"][0])
        assert audit_result is not None
        assert audit_result["terminalStatus"] == expected_status
        assert audit_result["disposition"] == "rejected"
        assert audit_result["qualification"]["qualified"] is False
        assert audit_result["scenarioProjection"]["exactStateAvailable"] is False
        assert audit_result["runtimeCompatibility"]["status"] == "not_checked"
        terminal_renderer = ResearchRenderPreparationServiceV1(
            research=SimpleNamespace(
                get_result=lambda _code: audit_result,
                get_recipe=lambda _code: service.get_recipe(recipe["code"]),
                get_playing_quality_audit=service.get_playing_quality_audit,
            ),
            artifact_store=service.artifact_store,
            publication_service=SimpleNamespace(),
            renderer=RemotionProfessionalRenderer(
                yaml.safe_load(
                    (
                        Path(__file__).resolve().parents[1]
                        / "configs/video/premium.yaml"
                    ).read_text(encoding="utf-8")
                ),
                repository_root=Path(__file__).resolve().parents[1],
            ),
            rules=context.bundle.rules,
            configuration_store=api.CONTAINER.configuration_store,
        )
        with pytest.raises(
            ValueError,
            match="runtime compatibility must pass before playable or video admission",
        ):
            terminal_renderer.prepare(audit_result["resultCode"])
        assert completed["attemptedCandidates"] == expected_attempted
        assert completed["scenarioYieldCounts"]["attempted"] == expected_attempted
        assert (
            completed["campaignAccounting"]["stopReason"] == "candidate_range_exhausted"
        )
        assert completed["scenarioResearchEvidence"]["sha256"]
        assert completed["performanceReceipt"]["completedCandidates"] == (
            expected_attempted
        )
        candidate_receipts = completed["campaignAccounting"]["candidateReceipts"]
        assert len(candidate_receipts) == 1
        assert candidate_receipts[0]["classification"] == "rejected"
        assert candidate_receipts[0]["status"] == executor_status
        assert any(
            event.get("candidateReceipt", {}).get("receiptId")
            == candidate_receipts[0]["receiptId"]
            for event in completed["events"]
        )
    finally:
        executor.shutdown(wait=True)


@pytest.mark.real_clock
@pytest.mark.parametrize(
    ("failure_kind", "expected_status"),
    (
        ("timeout", "not_found_within_budget"),
        ("policy_contradiction", "not_found_within_budget"),
        ("score_contradiction", "structurally_infeasible"),
    ),
)
def test_native_unsatisfied_search_persists_honest_terminal_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inline_v3_shard_runner: None,
    failure_kind: str,
    expected_status: str,
) -> None:
    authority, base_context = _frozen_context_and_authority()
    context = replace(
        base_context,
        bundle=_ChallengeBundleProxy(base_context.bundle, _ortools_challenge()),
    )
    service, worker = _isolated_service(tmp_path, authority, context)
    statuses = []
    real_solve = cp_model.CpSolver.solve

    def observe_solver(solver, model, *args, **kwargs):
        # Exercise real CP-SAT's deadline/infeasibility exits, not fabricated results.
        if failure_kind == "timeout":
            solver.parameters.max_time_in_seconds = 0.0
        status = real_solve(solver, model, *args, **kwargs)
        statuses.append(status)
        return status

    try:
        payload = _recipe(authority.to_dict(), "ortools_cp_sat_v3", trick_index=6).to_dict()
        if failure_kind != "timeout":
            payload.pop("sha256")
            payload.pop("capabilityPlan")
            payload["scenario"].pop("sha256")
            payload["scenario"]["constraints"].pop("sha256")
        if failure_kind == "policy_contradiction":
            conflict = copy.deepcopy(payload["scenario"]["constraints"]["children"][0])
            conflict.pop("sha256")
            conflict["predicateId"] = "PRED-ORTOOLS-CONFLICTING-OWNER"
            conflict["arguments"]["seat"] = "south"
            payload["scenario"]["constraints"]["children"].append(conflict)
        elif failure_kind == "score_contradiction":
            payload["scenario"]["constraints"]["children"].extend(
                {
                    "predicateId": f"PRED-ORTOOLS-SCORE-{score}",
                    "kind": "state.match_score",
                    "version": 1,
                    "hardness": "hard",
                    "arguments": {"team": "north_south", "minimum": score, "maximum": score},
                }
                for score in (0, 1)
            )
        recipe = service.create_recipe(payload, idempotency_key="native-unsatisfied-search-create")
        run = service._submit_recipe_run(
            recipe["code"], idempotency_key="native-unsatisfied-search-run", schedule=False,
        )
        with monkeypatch.context() as patch:
            patch.setattr(cp_model.CpSolver, "solve", observe_solver)
            service._execute(run["runCode"])

        if failure_kind == "timeout":
            assert statuses and all(status == cp_model.UNKNOWN for status in statuses)
        else:
            assert cp_model.INFEASIBLE in statuses
        completed = service.get_run(run["runCode"])
        assert completed["status"] == "completed", completed
        assert completed["scenarioResearchStatus"] == expected_status, completed
        assert completed["qualifiedCount"] == completed["resultCount"] == 0
        assert completed["resultCodes"] == []
        assert completed["auditResultCount"] == 1
        campaign = json.loads(service.artifact_store.get(completed["scenarioResearchEvidence"]["uri"]))
        execution = json.loads(service.artifact_store.get(campaign["executionEvidence"][0]["uri"]))
        assert execution["candidate"] is execution["proof"] is None
        assert execution["executionReceipt"]["candidateAttemptReason"] == {
            "timeout": "deal_auction_budget",
            "policy_contradiction": "auction_policy_no_witness",
            "score_contradiction": "initial_score_infeasible",
        }[failure_kind]
        result = service.get_result(completed["auditResultCodes"][0])
        assert result["terminalStatus"] == expected_status
        assert result["qualification"]["qualified"] is False
        assert result["scenarioProjection"]["exactStateAvailable"] is False
        assert result["runtimeCompatibility"]["status"] == "not_checked"

        cold_service, cold_worker = _isolated_service(tmp_path, authority, context)
        try:
            def reject_live_context(*_args, **_kwargs):
                pytest.fail("unsatisfied-search cold reads must use the frozen configuration snapshot")

            monkeypatch.setattr(cold_service, "_build_recipe_and_context", reject_live_context)
            restored = cold_service.get_result(result["resultCode"])
            assert restored == result
            api.ResearchScenarioPublicResultResponse.model_validate(api._public_research_result(restored))
        finally:
            cold_worker.shutdown(wait=True)
    finally:
        worker.shutdown(wait=True)


@pytest.mark.parametrize(
    ("campaign_status", "coordinator_status", "proof", "expected"),
    (
        ("completed", "completed", None, "qualified_target_satisfied"),
        (
            "structurally_infeasible",
            "completed",
            None,
            "exact_structural_infeasibility",
        ),
        ("incomplete", "budget_exhausted", None, "global_campaign_timeout"),
        (
            "incomplete",
            "not_found_within_budget",
            {"completionStatus": "proof_incomplete"},
            "proof_incomplete",
        ),
        (
            "proof_failed",
            "not_found_within_budget",
            None,
            "candidate_range_exhausted",
        ),
    ),
)
def test_campaign_stop_reason_is_operator_explicit(
    campaign_status: str,
    coordinator_status: str,
    proof: dict | None,
    expected: str,
) -> None:
    campaign = SimpleNamespace(
        status=campaign_status,
        executions=(SimpleNamespace(status="incomplete", proof=proof),),
    )

    assert (
        ResearchFoundryServiceV1._ortools_campaign_stop_reason_v2(
            campaign,
            {"status": coordinator_status},
        )
        == expected
    )


def test_campaign_stop_reason_distinguishes_worker_failure_from_resource_limit() -> (
    None
):
    campaign = SimpleNamespace(status="incomplete", executions=())

    assert (
        ResearchFoundryServiceV1._ortools_campaign_stop_reason_v2(
            campaign,
            {"status": "incomplete", "attempts": [{"status": "failed"}]},
        )
        == "worker_failed"
    )


@pytest.mark.real_clock
def test_cancel_terminalizes_campaign_progress_immediately(tmp_path: Path) -> None:
    authority, context = _frozen_context_and_authority()
    service, executor = _isolated_service(tmp_path, authority, context)
    try:
        recipe = service.store.put_recipe(
            _recipe(authority.to_dict(), "ortools_cp_sat_v3").to_dict()
        )
        run = service.store.create_run(
            {
                "runCode": "RUN-CANCEL-PROGRESS",
                "runHash": "c" * 64,
                "recipeCode": recipe["code"],
                "recipeHash": recipe["sha256"],
                "status": "running",
                "progress": 0.25,
                "campaignProgress": {
                    "state": "running",
                    "stopReason": None,
                    "attempts": 3,
                },
                "events": [],
            }
        )

        cancelled = service.cancel(run["runCode"])

        assert cancelled["status"] == "cancelled"
        assert cancelled["progress"] == 0.25
        assert cancelled["campaignProgress"] == {
            "state": "cancelled",
            "stopReason": "cancelled",
            "attempts": 3,
        }
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize("stop_stage", ["preflight", "graph"])
def test_v3_service_cancel_reaches_real_process_shard_and_persists_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_stage: str,
) -> None:
    import multiprocessing
    import threading

    authority, context = _frozen_context_and_authority()
    service, executor = _isolated_service(tmp_path, authority, context)
    backend = service._cached_ortools_scenario_backend_v2(context)
    assert backend is not None
    child_ready, release_progress, worker_finished = (threading.Event() for _ in range(3))
    original_process = executor_module._run_seed_campaign_process_v1

    def track_process(*args, **kwargs):
        try:
            return original_process(*args, **kwargs)
        finally:
            worker_finished.set()

    monkeypatch.setattr(executor_module, "_run_seed_campaign_process_v1", track_process)
    monkeypatch.setattr(executor_module, "time", SimpleNamespace(
        **{**vars(executor_module.time), "monotonic": time.monotonic}))
    real_execute_campaign = backend.execute_campaign

    def observe_child_progress(*args, **kwargs):
        original = kwargs["progress_callback"]

        def progress(value):
            original(value)
            if value.get("currentStage") == stop_stage:
                child_ready.set()
                release_progress.wait()

        kwargs["progress_callback"] = progress
        return real_execute_campaign(*args, **kwargs)

    monkeypatch.setattr(backend, "execute_campaign", observe_child_progress)
    existing_children = {child.pid for child in multiprocessing.active_children()}
    try:
        payload = _recipe(
            authority.to_dict(), "ortools_cp_sat_v3",
            trick_index=6,
            decision_remaining_cards=2 if stop_stage == "graph" else None,
            decision_seat="south" if stop_stage == "graph" else None,
            decision_position=0 if stop_stage == "graph" else None,
            solver_timeout_seconds=30,
        ).to_dict()
        if stop_stage == "graph":
            payload.pop("sha256")
            payload.pop("capabilityPlan")
            payload["scenario"].pop("sha256")
            payload["scenario"]["constraints"].pop("sha256")
            payload["scenario"]["constraints"]["children"].append({
                "predicateId": "PRED-TAIL-STATE",
                "kind": "state.legal_action_count",
                "version": 1,
                "hardness": "hard",
                "arguments": {"minimum": 2, "maximum": 8},
            })
            payload["scenario"]["constraints"]["children"].append({
                "predicateId": "PRED-SUN-CANCEL",
                "kind": "auction.contract",
                "version": 1,
                "hardness": "hard",
                "arguments": {"contract": "sun"},
            })
            assert TailFirstStagePlannerV1().plan(
                ResearchScenarioV1.build(payload["scenario"])
            ).planner == "tail_first"
        recipe = service.create_recipe(
            payload, idempotency_key="v3-real-process-cancel-create",
        )
        run = service._submit_recipe_run(
            recipe["code"], idempotency_key="v3-real-process-cancel-run",
            schedule=False,
        )
        future = executor.submit(service._execute, run["runCode"])
        future.add_done_callback(lambda _: child_ready.set())
        child_ready.wait()
        assert not future.done(), f"real V3 worker never reported {stop_stage}"
        cancelled = service.cancel(run["runCode"])
        release_progress.set()
        future.result()

        assert cancelled["status"] == "cancelled"
        stored = service.get_run(run["runCode"])
        assert stored is not None and stored["status"] == "cancelled"
        assert stored["scenarioResearchEvidence"]["sha256"]
        campaign = json.loads(service.artifact_store.get(
            stored["scenarioResearchEvidence"]["uri"]
        ))
        receipt = campaign["coordinatorReceipt"]
        assert receipt["status"] == "cancelled"
        assert receipt["attempts"][0]["status"] == "cancelled"
        assert receipt["attempts"][0]["partialEvidence"]["cancellation"][
            "acknowledged"
        ] is True
        worker_finished.wait()
        assert not any(
            child.pid not in existing_children
            for child in multiprocessing.active_children()
        ), "cancelled service run left a process shard alive"
    finally:
        release_progress.set()
        executor.shutdown(wait=True)


@pytest.mark.parametrize("solver_status", [cp_model.UNKNOWN, cp_model.MODEL_INVALID])
def test_native_unsolved_proof_survives_service_and_cold_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inline_v3_shard_runner: None,
    solver_status: int,
) -> None:
    authority, base_context = _frozen_context_and_authority()
    context = replace(
        base_context,
        bundle=_ChallengeBundleProxy(base_context.bundle, _policy_service_challenge()),
    )
    service, worker = _isolated_service(tmp_path, authority, context)
    real_prove = OrToolsScenarioGraphProofV1.prove
    stopped_requests = []

    class StoppedSolver:
        @staticmethod
        def status_name(status):
            return {cp_model.UNKNOWN: "UNKNOWN", cp_model.MODEL_INVALID: "MODEL_INVALID"}[status]

    try:
        payload = _native_qualified_exact_recipe_payload(context.bundle_revision_id, prefix_length=31)
        payload["scenario"]["decisionPoint"]["selector"]["trickIndex"] = 7
        payload["scenario"]["outcome"] = {"op": "all", "children": [
            payload["scenario"]["outcome"],
            {"predicateId": "PRED-E2E-ROOT-COUNT", "kind": "proof.winning_root_action_count",
             "version": 1, "hardness": "hard", "arguments": {"minimum": 0, "maximum": 1}},
        ]}
        payload["proof"].update(claim="winning_root_action_count", utility="round_win", successThreshold=1)
        payload["proof"].pop("selectedAction", None)
        recipe = service.create_recipe(payload)
        target_request_hash = recipe["proof"]["sha256"]

        def stop_target_solve(self, graph, request, **kwargs):
            if request.sha256 != target_request_hash:
                return real_prove(self, graph, request, **kwargs)
            stopped_requests.append(request.sha256)
            # Keep construction and all surrounding quality proofs real.
            with monkeypatch.context() as target_patch:
                target_patch.setattr(OrToolsScenarioGraphProofV1, "_solve", staticmethod(
                    lambda _model, **_kwargs: (StoppedSolver(), solver_status)))
                return real_prove(self, graph, request, **kwargs)

        monkeypatch.setattr(OrToolsScenarioGraphProofV1, "prove", stop_target_solve)
        run = service._submit_recipe_run(
            recipe["code"], idempotency_key="native-solver-stop-run", schedule=False)
        service._execute(run["runCode"])
        completed = service.get_run(run["runCode"])
        assert stopped_requests, completed
        assert completed["status"] == "completed", completed
        assert completed["scenarioResearchStatus"] == "proof_incomplete"
        assert completed["qualifiedCount"] == completed["resultCount"] == 0
        assert completed["resultCodes"] == []
        assert completed["auditResultCount"] == 1
        assert completed["quarantineCount"] == int(solver_status == cp_model.MODEL_INVALID)
        campaign = json.loads(service.artifact_store.get(completed["scenarioResearchEvidence"]["uri"]))
        execution = json.loads(service.artifact_store.get(campaign["executionEvidence"][0]["uri"]))
        assert execution["status"] == (
            "compiler_disagreement" if solver_status == cp_model.MODEL_INVALID else "incomplete")
        raw_proof = execution["proof"]
        assert raw_proof["completionStatus"] == (
            "model_error" if solver_status == cp_model.MODEL_INVALID else "proof_incomplete")
        assert raw_proof["solverStatus"] == (
            "model_invalid" if solver_status == cp_model.MODEL_INVALID else "unknown")
        assert raw_proof["claimSatisfied"] is False
        assert raw_proof["rootActions"] == []
        assert raw_proof["winningRootActionCount"] is None
        assert raw_proof["winningRootActionLowerBound"] == 0
        assert raw_proof["winningRootActionUpperBound"] == 1
        assert raw_proof.get("nativeQualification") is None
        assert execution["candidate"]["stageReceipts"]["proof"]["evidenceBindings"] == []
        assert execution["candidate"]["stageReceipts"]["tactics"]["evidenceBindings"] == []
        assert execution["candidate"]["trace"]["playingQualityEvidence"]["receipt"]["status"] == "complete"
        result = service.get_result(completed["auditResultCodes"][0])
        assert result["disposition"] == "rejected"
        assert result["qualification"]["qualified"] is False
        proof = result["proofOutcome"]
        assert proof["completionStatus"] == "proof_incomplete"
        assert proof["achievedMode"] == "none"
        assert proof["claimSatisfied"] is False
        assert proof["rootActions"] == []
        assert proof["winningRootActionCount"] is None
        assert proof["winningRootActionLowerBound"] == 0
        assert proof["winningRootActionUpperBound"] == 1

        cold_service, cold_worker = _isolated_service(tmp_path, authority, context)
        try:
            assert not cold_service._result_verification_cache

            def reject_live_context(*_args, **_kwargs):
                pytest.fail("cold solver-stop reads must use the stored configuration snapshot")

            monkeypatch.setattr(cold_service, "_build_recipe_and_context", reject_live_context)
            restored = cold_service.get_result(result["resultCode"])
            assert restored["resultHash"] == result["resultHash"]
            assert restored["proofOutcome"] == proof
            assert restored["artifacts"] == result["artifacts"]
            public = api._public_research_result(restored)
            api.ResearchScenarioPublicResultResponse.model_validate(public)
            assert public["proofOutcome"]["rootActions"] == []
            assert public["proofOutcome"]["winningRootActionCount"] is None
        finally:
            cold_worker.shutdown(wait=True)
    finally:
        worker.shutdown(wait=True)


@pytest.mark.real_clock
def test_actor_compiler_disagreement_is_quarantined_and_service_wires_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inline_v3_shard_runner: None,
) -> None:
    authority, legacy_context = _frozen_context_and_authority()
    context = replace(
        legacy_context,
        bundle=_ChallengeBundleProxy(legacy_context.bundle, _ortools_challenge()),
    )
    observed_cancel_checks = []

    def execute(*_args, **kwargs):
        cancel_check = kwargs.get("cancel_check")
        assert callable(cancel_check)
        observed_cancel_checks.append(cancel_check())
        return execution

    monkeypatch.setattr(
        "baloot_platform.research.scenario_backend.OrToolsUnifiedResearchExecutorV1.execute",
        execute,
    )
    service, executor = _isolated_service(tmp_path, authority, context)
    backend = service._cached_ortools_scenario_backend_v2(context)
    assert backend is not None
    try:
        payload = _recipe(
            authority.to_dict(),
            "ortools_cp_sat_v3",
            trick_index=6,
        ).to_dict()
        recipe = service.create_recipe(
            payload,
            idempotency_key="ortools-compiler-disagreement-create",
        )
        execution = _terminal_search_execution(
            "compiler_disagreement",
            backend.identity.predicate_catalog_hash,
            rules=context.bundle.rules,
            bundle_revision_id=context.bundle_revision_id,
            bundle_hash=context.bundle.sha256,
            scenario_hash=recipe["scenario"]["sha256"],
        )
        run = service._submit_recipe_run(
            recipe["code"],
            idempotency_key="ortools-compiler-disagreement-run",
            schedule=False,
        )

        service._execute(run["runCode"])

        completed = service.get_run(run["runCode"])
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["scenarioResearchStatus"] == "proof_incomplete"
        assert completed["quarantineCount"] == 1
        assert completed["resultCodes"] == []
        assert observed_cancel_checks == [False]
        campaign = json.loads(
            service.artifact_store.get(completed["scenarioResearchEvidence"]["uri"])
        )
        execution_document = json.loads(
            service.artifact_store.get(campaign["executionEvidence"][0]["uri"])
        )
        assert execution_document["status"] == "compiler_disagreement"
        assert execution_document["status"] != "proof_failed"
    finally:
        executor.shutdown(wait=True)
