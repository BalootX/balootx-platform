from __future__ import annotations

import copy
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from baloot_platform.challenge.decision_cohort import (
    PRIMARY_DECISION_COHORT_SALT_V1,
    derive_actor_observer_cohort_seed_v1,
)
from baloot_platform.challenge.finalist_validation import (
    FinalistValidationProfileV1,
)
from baloot_platform.core.enums import Seat
from baloot_platform.research import (
    ResearchExecutionContextFactoryV1,
    ResearchFoundryServiceV1,
    ResearchRecipeV3,
    SQLiteResearchStoreV1,
    build_research_recipe,
)
from baloot_platform.research.contracts import canonical_sha256
from baloot_platform.research.execution_context import (
    research_analysis_identity_hash_v1,
)


class _CapturingExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[object, tuple[object, ...]]] = []

    def submit(self, function, *args):
        self.calls.append((function, args))
        return SimpleNamespace()


class _MemoryS3ArtifactStore:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    def put(self, data: bytes, *, media_type: str, name: str | None = None):
        digest = hashlib.sha256(data).hexdigest()
        uri = f"s3://balootx-test/research/sha256/aa/bb/{digest}"
        self.values[uri] = bytes(data)
        return {
            "uri": uri,
            "sha256": digest,
            "bytes": len(data),
            "mediaType": media_type,
            "name": name,
        }

    def get(self, uri: str) -> bytes:
        return self.values[uri]


def _selector_recipe(bundle_revision_id: str) -> dict:
    return {
        "schemaVersion": 3,
        "titleAr": "اختبار سياق بحث مجمّد",
        "output": {"kind": "laffa", "resultCount": 1},
        "configuration": {
            "schemaVersion": 1,
            "bundleRevisionId": bundle_revision_id,
        },
        "situation": {
            "dealer": "east",
            "buyer": None,
            "contracts": [],
            "matchScore": [0, 0],
            "requiredCards": {},
            "forbiddenCards": {},
            "projects": [],
            "escalations": [],
        },
        "tactics": [],
        "qualification": {
            "minimumTruth": "provisional",
            "minimumConfidence": 0.0,
            "minimumSurroundingQuality": 0.0,
            "requireObserverSafety": True,
            "requireAllDecisionAudit": True,
            "maximumRegret": 100.0,
        },
        "resources": {
            "generationLane": "organic_runtime",
            "seedStart": 41,
            "seedCount": 1,
            "seedsPerShard": 1,
            "analysisMode": "screening",
            "deepVerificationPolicy": "off",
        },
        "diversity": {
            "schemaVersion": 1,
            "cellDimensions": ["contract"],
            "topKPerCell": 1,
            "rejectedAuditModulus": 10,
        },
        "renderPolicy": {"mode": "on_demand"},
        "status": "published",
    }


def _published_tenant_bundle(configuration, *, bundle_id: str) -> dict:
    active = configuration.resolve()
    analysis = copy.deepcopy(active["profiles"]["analysis"])
    analysis["requested_samples"] = max(1, int(analysis["requested_samples"]) - 1)
    analysis_revision = configuration.create_profile(
        family="analysis",
        payload=analysis,
        created_by="TEST",
        scope="tenant",
        tenant_id=bundle_id.upper(),
        publish=True,
    )
    component_revisions = {
        family: str(value["revisionId"])
        for family, value in active["profileRevisions"].items()
    }
    component_revisions["analysis"] = str(analysis_revision["revisionId"])
    return configuration.create_bundle(
        bundle_id=bundle_id,
        version="99.0.0",
        mode=str(active["bundle"]["mode"]),
        component_revisions=component_revisions,
        created_by="TEST",
        scope="tenant",
        tenant_id=bundle_id.upper(),
        publish=True,
        force_new_revision=True,
    )


def _isolated_service(base, store, executor) -> ResearchFoundryServiceV1:
    return ResearchFoundryServiceV1(
        domain=base.domain,
        compiled_domain=base.compiled_domain,
        store=store,
        artifact_store=base.artifact_store,
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
        execution_context_factory=base.execution_context_factory,
    )


def _analysis_budget(profile) -> dict[str, int | float]:
    return {
        "requested_samples": profile.play_samples,
        "exact_allocation_limit": profile.exact_allocation_limit,
        "exact_search_max_remaining": profile.exact_search_max_remaining,
        "maximum_sampling_attempts": profile.maximum_sampling_attempts,
        "exact_search_max_nodes": profile.exact_search_max_nodes,
        "exact_search_max_seconds": profile.exact_search_max_seconds,
        "exact_search_max_depth": profile.exact_search_max_depth,
        "exact_search_max_cache_entries": profile.exact_search_max_cache_entries,
    }


def _expected_analysis_identity(
    context,
    mode: str,
    *,
    descriptor: dict | None = None,
    profile=None,
) -> str:
    if descriptor is None or profile is None:
        analyzer = context.analyzers[mode]
        descriptor = analyzer.descriptor.to_dict()
        profile = analyzer.profile
    expected_projection = {
        "schemaVersion": 1,
        "mode": mode,
        "descriptor": descriptor,
        "profileHash": profile.sha256,
        "analysisPolicy": asdict(context.analysis_policy),
        "beliefPolicy": asdict(context.belief_policy),
        "matchValueProfileHash": context.bundle.components["match_value"].sha256,
        "rulesHash": context.bundle.rules.sha256,
        "conventionHash": context.bundle.components["conventions"].sha256,
        "challengeHash": context.bundle.components["challenge"].sha256,
        "performanceHash": context.bundle.components["performance"].sha256,
        "searchBackend": context.exact_search_backend.descriptor.to_dict(),
    }
    expected = canonical_sha256(expected_projection)
    assert (
        research_analysis_identity_hash_v1(
            mode=mode,
            descriptor=descriptor,
            profile_hash=profile.sha256,
            analysis_policy=context.analysis_policy,
            belief_policy=context.belief_policy,
            match_value_profile_hash=context.bundle.components["match_value"].sha256,
            rules_hash=context.bundle.rules.sha256,
            convention_hash=context.bundle.components["conventions"].sha256,
            challenge_hash=context.bundle.components["challenge"].sha256,
            performance_hash=context.bundle.components["performance"].sha256,
            search_backend_descriptor=(
                context.exact_search_backend.descriptor.to_dict()
            ),
        )
        == expected
    )
    return expected


def _assert_primary_policy_context(context) -> None:
    for mode in ("screening", "cascaded", "deep"):
        assert context.analysis_identity_hashes[mode] == (
            _expected_analysis_identity(context, mode)
        )

    screening = context.analyzers["screening"]
    cascaded = context.analyzers["cascaded"]
    deep = context.analyzers["deep"]
    assert (
        screening.cohort_authority_hash
        == (context.analysis_identity_hashes["screening"])
    )
    assert deep.cohort_authority_hash == context.analysis_identity_hashes["deep"]
    assert cascaded.screening_analyzer is screening
    assert cascaded.confirmation_analyzer is deep

    simulator = context.simulator
    agreement = simulator.decision_agreement
    assert simulator.agent_profile.id == "finalist_candidate"
    assert simulator.generation_tier == "throughput_reference"
    assert agreement is not None
    assert agreement.exact_primary_policy is True
    reanalysis = agreement.reanalysis_trace_analysis_profile
    assert reanalysis is not None
    assert context.analysis_identity_hashes["finalist_reanalysis"] == (
        _expected_analysis_identity(
            context,
            "finalist_reanalysis",
            descriptor=deep.descriptor.to_dict(),
            profile=reanalysis,
        )
    )
    assert simulator.trace_analysis_profile is deep.profile
    assert agreement.primary_trace_analysis_profile is deep.profile
    assert agreement.trace_analysis_profile is deep.profile
    assert simulator.analysis_identity_hash == context.analysis_identity_hashes["deep"]
    assert agreement.primary_analysis_identity_hash == (
        context.analysis_identity_hashes["deep"]
    )
    assert agreement.analysis_identity_hash == context.analysis_identity_hashes["deep"]
    assert agreement.reanalysis_analysis_identity_hash == (
        context.analysis_identity_hashes["finalist_reanalysis"]
    )
    assert agreement.primary_trace_analysis_profile_hash == deep.profile.sha256
    assert agreement.reanalysis_trace_analysis_profile_hash == reanalysis.sha256
    assert deep.profile.sha256 != reanalysis.sha256
    assert (
        context.analysis_identity_hashes["deep"]
        != context.analysis_identity_hashes["finalist_reanalysis"]
    )

    primary_budget = _analysis_budget(deep.profile)
    reanalysis_budget = _analysis_budget(reanalysis)
    assert primary_budget["requested_samples"] == 192
    assert reanalysis_budget["requested_samples"] == 256
    assert simulator.primary_analysis_budget == primary_budget
    assert dict(agreement.primary_analysis_budget) == primary_budget
    assert dict(agreement.independent_analysis_budget) == reanalysis_budget
    assert primary_budget != reanalysis_budget

    public_prefix_hash = "a" * 64
    expected_seed = derive_actor_observer_cohort_seed_v1(
        public_prefix_hash=public_prefix_hash,
        decision_kind="play",
        acting_seat=Seat.WEST,
        cohort_authority_hash=context.analysis_identity_hashes["deep"],
        cohort_index=0,
        governed_salt=PRIMARY_DECISION_COHORT_SALT_V1,
    )
    assert (
        simulator.primary_decision_seed(
            public_prefix_hash=public_prefix_hash,
            decision_kind="play",
            acting_seat=Seat.WEST,
        )
        == expected_seed
    )
    expected_reanalysis_seed = derive_actor_observer_cohort_seed_v1(
        public_prefix_hash=public_prefix_hash,
        decision_kind="play",
        acting_seat=Seat.WEST,
        cohort_authority_hash=context.analysis_identity_hashes[
            "finalist_reanalysis"
        ],
        cohort_index=1,
        governed_salt=PRIMARY_DECISION_COHORT_SALT_V1,
    )
    assert (
        agreement.actor_observer_seed(
            public_prefix_hash=public_prefix_hash,
            decision_kind="play",
            acting_seat=Seat.WEST,
            cohort_index=1,
        )
        == expected_reanalysis_seed
    )
    assert expected_seed != expected_reanalysis_seed

    finalist_validation = FinalistValidationProfileV1.build(
        context.bundle.component("challenge")["finalist_validation"]
    )
    expected_agent_identity = canonical_sha256(
        {
            "schemaVersion": 3,
            "authority": "research_primary_policy_v4",
            "descriptor": simulator.descriptor.to_dict(),
            "agentRelease": simulator.agent_release,
            "agentProfile": simulator.agent_profile.to_dict(),
            "generationTier": simulator.generation_tier,
            "finalistValidationProfile": finalist_validation.to_dict(),
            "decisionAgreement": {
                "descriptor": agreement.descriptor.to_dict(),
                "profile": agreement.profile.to_dict(),
                "profileHash": agreement.profile.sha256,
                "exactPrimaryPolicy": agreement.exact_primary_policy,
            },
            "deepAnalysis": {
                "profile": deep.profile.to_dict(),
                "profileHash": deep.profile.sha256,
                "analysisIdentityHash": context.analysis_identity_hashes["deep"],
                "playBudget": primary_budget,
            },
            "finalistReanalysis": {
                "profile": reanalysis.to_dict(),
                "profileHash": reanalysis.sha256,
                "analysisIdentityHash": context.analysis_identity_hashes[
                    "finalist_reanalysis"
                ],
                "playBudget": reanalysis_budget,
            },
        }
    )
    assert context.agent_identity_hash == expected_agent_identity
    expected_context_hash = canonical_sha256(
        {
            "schemaVersion": 3,
            "authority": "research_primary_policy_v4",
            "effectiveRuntimeConfigHash": context.effective_runtime_config_hash,
            "semanticBundleHash": context.bundle.sha256,
            "domainHash": context.compiled_domain.sha256,
            "legacyDomainV2Hash": context.legacy_domain_v2.sha256,
            "compiledDomainHash": context.compiled_domain.sha256,
            "agentIdentityHash": context.agent_identity_hash,
            "analysisIdentityHashes": dict(context.analysis_identity_hashes),
            "scoreProfileHash": context.score_profile.sha256,
            "decisionScoreProfileHash": context.decision_score_profile.sha256,
            "comparisonProfileHash": context.comparison_profile_hash,
            "plugins": list(context.plugin_identities),
        }
    )
    assert context.context_hash == expected_context_hash
    plugin_ids = {str(item["id"]) for item in context.plugin_identities}
    assert "finalist_agent_agreement_v1" in plugin_ids
    assert "cascaded_trace_analyzer_v1" in plugin_ids


def _legacy_v3_agent_identity_hash(context) -> str:
    simulator = context.simulator
    agreement = simulator.decision_agreement
    assert agreement is not None
    deep = context.analyzers["deep"]
    finalist_validation = FinalistValidationProfileV1.build(
        context.bundle.component("challenge")["finalist_validation"]
    )
    return canonical_sha256(
        {
            "schemaVersion": 2,
            "authority": "research_primary_policy_v3",
            "descriptor": simulator.descriptor.to_dict(),
            "agentRelease": simulator.agent_release,
            "agentProfile": simulator.agent_profile.to_dict(),
            "generationTier": simulator.generation_tier,
            "finalistValidationProfile": finalist_validation.to_dict(),
            "decisionAgreement": {
                "descriptor": agreement.descriptor.to_dict(),
                "profile": agreement.profile.to_dict(),
                "profileHash": agreement.profile.sha256,
                "exactPrimaryPolicy": agreement.exact_primary_policy,
            },
            "deepAnalysis": {
                "profile": deep.profile.to_dict(),
                "profileHash": deep.profile.sha256,
                "analysisIdentityHash": context.analysis_identity_hashes["deep"],
                "playBudget": _analysis_budget(deep.profile),
            },
        }
    )


def test_selected_bundle_flows_through_validate_estimate_submit_worker_and_compare(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    from baloot_platform.service import api

    selected = _published_tenant_bundle(
        api.CONTAINER.configuration,
        bundle_id="research-selected-b",
    )
    executor = _CapturingExecutor()
    research = _isolated_service(
        api.RESEARCH,
        SQLiteResearchStoreV1(tmp_path / "research.sqlite3"),
        executor,
    )
    selector = _selector_recipe(str(selected["revisionId"]))

    validated = research.validate_recipe(selector)
    assert validated["valid"] is True
    frozen = validated["recipe"]["configuration"]
    assert frozen["bundleRevisionId"] == selected["revisionId"]
    assert research.estimate(selector)["valid"] is True
    created = research.store.put_recipe(validated["recipe"])
    assert created["configuration"] == frozen

    run = research.submit(created["code"], idempotency_key="selected-b")
    stored_run = research.store.get_run(run["runCode"])
    assert stored_run["configurationAuthority"] == frozen
    assert stored_run["researchContextHash"]
    tampered_run = copy.deepcopy(stored_run)
    tampered_run["configurationAuthority"]["effectiveRuntimeConfigHash"] = "0" * 64
    frozen_recipe = build_research_recipe(validated["recipe"])
    context = research.execution_context_factory.resolve_recipe(frozen_recipe)
    with pytest.raises(ValueError, match="configuration authority mismatch"):
        research._assert_run_execution_context(
            tampered_run,
            frozen_recipe,
            context,
        )
    stale_context_run = copy.deepcopy(stored_run)
    stale_context_run["researchContextHash"] = "0" * 64
    with pytest.raises(ValueError, match="run context identity mismatch"):
        research._assert_run_execution_context(
            stale_context_run,
            frozen_recipe,
            context,
        )
    seen_contexts = []
    monkeypatch.setattr(
        research,
        "_shard_worker_loop",
        lambda run_code, worker_index, context=None: seen_contexts.append(context),
    )
    research._execute(run["runCode"])
    assert len(seen_contexts) == 1
    assert seen_contexts[0].bundle_revision_id == selected["revisionId"]
    assert seen_contexts[0].context_hash == stored_run["researchContextHash"]
    active_revision = str(api.CONTAINER.configuration.resolve()["bundle"]["revisionId"])
    active_context = research.execution_context_factory.resolve_bundle_revision(
        active_revision
    )[1]
    assert seen_contexts[0].analysis_policy.requested_samples != (
        active_context.analysis_policy.requested_samples
    )

    campaign = research.create_comparison_campaign(
        created["code"], idempotency_key="selected-b-comparison"
    )
    lane_contexts = []
    generation_lanes = set()
    for recipe_code in campaign["recipeCodes"].values():
        lane_recipe = build_research_recipe(research.store.get_recipe(recipe_code))
        assert isinstance(lane_recipe, ResearchRecipeV3)
        assert lane_recipe.configuration == frozen
        generation_lanes.add(str(lane_recipe.resources["generationLane"]))
        lane_contexts.append(
            research.execution_context_factory.resolve_recipe(lane_recipe)
        )
    assert generation_lanes == {"organic_runtime", "ortools"}
    assert {item.context_hash for item in lane_contexts} == {
        seen_contexts[0].context_hash
    }
    assert {item.agent_identity_hash for item in lane_contexts} == {
        seen_contexts[0].agent_identity_hash
    }
    assert {item.simulator.agent_release for item in lane_contexts} == {
        seen_contexts[0].simulator.agent_release
    }
    assert {item.simulator.agent_profile.id for item in lane_contexts} == {
        "finalist_candidate"
    }


def test_concurrent_selected_bundles_never_cross_contaminate():
    from baloot_platform.service import api

    configuration = api.CONTAINER.configuration
    factory = api.RESEARCH.execution_context_factory
    assert factory is not None
    active_revision = str(configuration.resolve()["bundle"]["revisionId"])
    selected = _published_tenant_bundle(
        configuration,
        bundle_id="research-concurrent-b",
    )
    active_context = factory.resolve_bundle_revision(active_revision)[1]
    selected_context = factory.resolve_bundle_revision(str(selected["revisionId"]))[1]
    assert active_context.context_hash != selected_context.context_hash
    assert active_context.analysis_policy.requested_samples != (
        selected_context.analysis_policy.requested_samples
    )
    assert (
        active_context.analysis_identity_hashes["deep"]
        != (selected_context.analysis_identity_hashes["deep"])
    )
    assert (
        active_context.simulator.analysis_identity_hash
        == (active_context.analysis_identity_hashes["deep"])
    )
    assert (
        selected_context.simulator.analysis_identity_hash
        == (selected_context.analysis_identity_hashes["deep"])
    )
    requested = [active_revision, str(selected["revisionId"])] * 4
    with ThreadPoolExecutor(max_workers=4) as pool:
        resolved = list(
            pool.map(
                lambda revision: factory.resolve_bundle_revision(revision)[
                    1
                ].bundle_revision_id,
                requested,
            )
        )
    assert resolved == requested


def test_retired_bundle_capabilities_remain_readable_but_not_launchable():
    from baloot_platform.service import api

    configuration = api.CONTAINER.configuration
    bundle = _published_tenant_bundle(
        configuration, bundle_id="research-retired-capabilities",
    )
    revision_id = str(bundle["revisionId"])
    configuration.retire_bundle(
        revision_id, expected_etag=int(bundle["etag"]),
        actor_id="TEST", reason="read-only historical Research audit",
    )
    analysis_revision = configuration.store.get_profile(
        str(bundle["componentRevisions"]["analysis"])
    )
    assert analysis_revision is not None
    configuration.retire_profile(
        analysis_revision.revision_id, expected_etag=analysis_revision.etag,
        actor_id="TEST", reason="read-only historical Research audit",
    )
    with pytest.raises(ValueError, match="exact bundle revision"):
        configuration.resolve(allow_retired=True)

    factory = api.RESEARCH.execution_context_factory
    assert factory is not None
    with pytest.raises(PermissionError, match="unpublished configuration bundle"):
        factory.resolve_bundle_revision(revision_id)
    capabilities = api.RESEARCH.capabilities(bundle_revision_id=revision_id)
    research = capabilities["research"]
    assert research["configurationSelection"]["bundleRevisionId"] == revision_id
    if "scenarioResearchV4" in research:
        assert research["scenarioResearchV4"]["launchAdmissible"] is False
        assert research["scenarioResearchV4"]["executionReady"] is False
    api.ResearchDomainCapabilitiesResponse.model_validate(capabilities)
    if "scenarioResearchV4" in research:
        invalid = copy.deepcopy(capabilities)
        invalid["research"]["scenarioResearchV4"]["executionReady"] = True
        with pytest.raises(ValueError, match="execution and launch capability disagree"):
            api.ResearchDomainCapabilitiesResponse.model_validate(invalid)


def test_context_freezes_finalist_primary_policy_and_exact_analysis_authorities():
    from baloot_platform.service import api

    factory = api.RESEARCH.execution_context_factory
    assert factory is not None
    active_revision = str(api.CONTAINER.configuration.resolve()["bundle"]["revisionId"])
    context = factory.resolve_bundle_revision(active_revision)[1]

    _assert_primary_policy_context(context)


def test_frozen_authority_selects_its_analyzer_without_operator_input():
    from baloot_platform.service import api

    factory = api.RESEARCH.execution_context_factory
    assert factory is not None
    active_revision = str(api.CONTAINER.configuration.resolve()["bundle"]["revisionId"])
    authority, expected = factory.freeze_selector(
        bundle_revision_id=active_revision,
        analysis_mode="cascaded",
    )

    inferred = factory.resolve_authority(authority.to_dict())

    assert inferred.context_hash == expected.context_hash
    with pytest.raises(
        ValueError,
        match="analyzer selection differs from frozen authority",
    ):
        factory.resolve_authority(authority.to_dict(), analysis_mode="screening")


def test_stored_v3_authority_with_an_old_agent_identity_fails_closed():
    from baloot_platform.service import api

    factory = api.RESEARCH.execution_context_factory
    assert factory is not None
    active_revision = str(api.CONTAINER.configuration.resolve()["bundle"]["revisionId"])
    authority, context = factory.freeze_selector(
        bundle_revision_id=active_revision,
        analysis_mode="deep",
    )
    old_authority = authority.to_dict()
    legacy_identity = _legacy_v3_agent_identity_hash(context)
    assert legacy_identity != context.agent_identity_hash
    old_authority["identities"]["agent"]["sha256"] = legacy_identity
    old_authority.pop("sha256")

    with pytest.raises(ValueError, match="Research execution identity mismatch"):
        factory.resolve_authority(old_authority, analysis_mode="deep")


def test_selected_bundle_exceeding_host_ceiling_is_rejected():
    from baloot_platform.service import api

    configuration = api.CONTAINER.configuration
    active = configuration.resolve()
    challenge = copy.deepcopy(active["profiles"]["challenge"])
    challenge["decision_analysis_executor"]["screening"]["workerCount"] += 1
    revision = configuration.create_profile(
        family="challenge",
        payload=challenge,
        created_by="TEST",
        scope="tenant",
        tenant_id="RESEARCH-HOST-CEILING",
        publish=True,
    )
    components = {
        family: str(value["revisionId"])
        for family, value in active["profileRevisions"].items()
    }
    components["challenge"] = str(revision["revisionId"])
    selected = configuration.create_bundle(
        bundle_id="research-host-ceiling",
        version="99.0.0",
        mode=str(active["bundle"]["mode"]),
        component_revisions=components,
        created_by="TEST",
        scope="tenant",
        tenant_id="RESEARCH-HOST-CEILING",
        publish=True,
        force_new_revision=True,
    )
    with pytest.raises(ValueError, match="exceeds the active host safety ceiling"):
        api.RESEARCH.execution_context_factory.resolve_bundle_revision(
            str(selected["revisionId"])
        )


@pytest.mark.parametrize(
    ("field", "increment"),
    (
        ("reanalysis_samples", 1),
        ("maximum_sampling_attempts", 1),
        ("exact_allocation_limit", 1),
        ("exact_search_remaining", 1),
        ("exact_search_max_nodes", 1),
        ("exact_search_max_seconds", 0.1),
    ),
)
def test_selected_bundle_exceeding_each_finalist_reanalysis_ceiling_is_rejected(
    field: str,
    increment: int | float,
):
    from baloot_platform.service import api

    configuration = api.CONTAINER.configuration
    active = configuration.resolve()
    challenge = copy.deepcopy(active["profiles"]["challenge"])
    challenge["stability"][field] += increment
    slug = field.replace("_", "-")
    revision = configuration.create_profile(
        family="challenge",
        payload=challenge,
        created_by="TEST",
        scope="tenant",
        tenant_id=f"RESEARCH-STABILITY-{slug}".upper(),
        publish=True,
    )
    components = {
        family: str(value["revisionId"])
        for family, value in active["profileRevisions"].items()
    }
    components["challenge"] = str(revision["revisionId"])
    selected = configuration.create_bundle(
        bundle_id=f"research-stability-{slug}",
        version="99.0.0",
        mode=str(active["bundle"]["mode"]),
        component_revisions=components,
        created_by="TEST",
        scope="tenant",
        tenant_id=f"RESEARCH-STABILITY-{slug}".upper(),
        publish=True,
        force_new_revision=True,
    )

    with pytest.raises(
        ValueError,
        match=rf"stability\.{field} exceeds the active host safety ceiling",
    ):
        api.RESEARCH.execution_context_factory.resolve_bundle_revision(
            str(selected["revisionId"])
        )


def test_s3_style_frozen_snapshot_round_trips_without_row_resolution(
    monkeypatch: pytest.MonkeyPatch,
):
    from baloot_platform.service import api

    base = api.RESEARCH.execution_context_factory
    assert base is not None
    active_revision = str(api.CONTAINER.configuration.resolve()["bundle"]["revisionId"])
    _effective, active_context = base.resolve_bundle_revision(active_revision)
    artifact_store = _MemoryS3ArtifactStore()
    factory = ResearchExecutionContextFactoryV1(
        configuration_service=api.CONTAINER.configuration,
        artifact_store=artifact_store,
        root=base.root,
        plugin_capabilities=base.plugin_capabilities,
        shared_plugin_descriptors=base.shared_plugin_descriptors,
        active_bundle=active_context.bundle,
        active_exact_search_backend=base.active_exact_search_backend,
    )
    authority, context = factory.freeze_selector(
        bundle_revision_id=active_revision,
        analysis_mode="screening",
    )
    assert authority.snapshot_artifact["uri"].startswith("s3://")
    expected_analysis_identities = dict(context.analysis_identity_hashes)
    expected_agent_identity = context.agent_identity_hash
    agreement = context.simulator.decision_agreement
    assert agreement is not None
    expected_primary_profile_hash = agreement.primary_trace_analysis_profile_hash
    expected_reanalysis_profile_hash = (
        agreement.reanalysis_trace_analysis_profile_hash
    )
    expected_primary_budget = dict(agreement.primary_analysis_budget)
    expected_reanalysis_budget = dict(agreement.independent_analysis_budget)
    monkeypatch.setattr(
        api.CONTAINER.configuration,
        "resolve",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("snapshot hydration must not resolve rows")
        ),
    )
    hydration_factory = ResearchExecutionContextFactoryV1(
        configuration_service=api.CONTAINER.configuration,
        artifact_store=artifact_store,
        root=base.root,
        plugin_capabilities=base.plugin_capabilities,
        shared_plugin_descriptors=base.shared_plugin_descriptors,
        active_bundle=active_context.bundle,
        active_exact_search_backend=base.active_exact_search_backend,
    )
    hydrated = hydration_factory.resolve_authority(
        authority.to_dict(), analysis_mode="screening"
    )
    assert hydrated.context_hash == context.context_hash
    assert dict(hydrated.analysis_identity_hashes) == expected_analysis_identities
    assert hydrated.agent_identity_hash == expected_agent_identity
    hydrated_agreement = hydrated.simulator.decision_agreement
    assert hydrated_agreement is not None
    assert (
        hydrated_agreement.primary_trace_analysis_profile_hash
        == expected_primary_profile_hash
    )
    assert (
        hydrated_agreement.reanalysis_trace_analysis_profile_hash
        == expected_reanalysis_profile_hash
    )
    assert dict(hydrated_agreement.primary_analysis_budget) == expected_primary_budget
    assert (
        dict(hydrated_agreement.independent_analysis_budget)
        == expected_reanalysis_budget
    )


def test_legacy_recipe_without_unique_retained_identity_fails_closed():
    from baloot_platform.service import api

    legacy = _selector_recipe("unused")
    legacy["schemaVersion"] = 2
    legacy.pop("configuration")
    legacy["profiles"] = {
        "domainHash": "1" * 64,
        "bundleHash": "2" * 64,
        "rulesHash": "3" * 64,
        "conventionHash": "4" * 64,
        "agentRelease": "missing@legacy",
        "analysisProfileHash": "5" * 64,
        "scoreProfileHash": "6" * 64,
    }
    recipe = build_research_recipe(legacy)
    with pytest.raises(ValueError, match="historical_record_requires_readjudication"):
        api.RESEARCH.execution_context_factory.resolve_recipe(recipe)


def test_legacy_replay_checks_schedule_integrity_only_for_possible_matches(root, tmp_path, monkeypatch):
    from baloot_platform.config import build_bundle
    from baloot_platform.configuration import SQLiteConfigurationStore, VersionedConfigurationService
    from baloot_platform.configuration.schedule import ConfigurationIntegrityError
    from baloot_platform.service import api

    base = api.RESEARCH.execution_context_factory
    store = SQLiteConfigurationStore(tmp_path / "configuration.sqlite3")
    configuration = VersionedConfigurationService(store, root=root, profile=base.configuration_service.profile)
    packaged = build_bundle(root)
    current = configuration.bootstrap_packaged_bundle(actor_id="TEST", packaged_bundle=packaged)["bundle"]
    factory = ResearchExecutionContextFactoryV1(
        configuration_service=configuration,
        artifact_store=base.artifact_store,
        root=root,
        plugin_capabilities=base.plugin_capabilities,
        shared_plugin_descriptors=base.shared_plugin_descriptors,
        active_bundle=packaged,
        active_exact_search_backend=base.active_exact_search_backend,
    )
    _, saved = factory.resolve_bundle_revision(current["revisionId"])
    raw = _selector_recipe(current["revisionId"])
    del raw["configuration"]
    raw["profiles"] = {
        "domainHash": saved.legacy_domain_v2.sha256,
        "bundleHash": saved.bundle.sha256,
        "rulesHash": saved.legacy_domain_v2.rules_hash,
        "conventionHash": saved.legacy_domain_v2.convention_hash,
        "agentRelease": saved.simulator.agent_release,
        "analysisProfileHash": saved.analyzers["screening"].profile.sha256,
        "scoreProfileHash": saved.score_profile.sha256,
    }
    v1 = {key: value for key, value in raw.items() if key not in {"diversity", "renderPolicy"}}
    recipes = [build_research_recipe({**v1, "schemaVersion": 1}), build_research_recipe({**raw, "schemaVersion": 2})]
    common = {"mode": current["mode"], "component_revisions": current["componentRevisions"],
              "created_by": "TEST", "publish": True}
    window = {"timeZone": "America/New_York", "effectiveFrom": "2026-07-01T12:00"}
    payload = copy.deepcopy(saved.bundle.component("performance"))
    payload["profile"].update(id="legacy_handoff", version="99.0.0")
    alternative = configuration.create_profile(
        family="performance", payload=payload, created_by="TEST", publish=True,
    )
    successor = configuration.create_bundle(**{
        **common, "bundle_id": current["bundleId"], "version": "99.0.0",
        "component_revisions": {**common["component_revisions"], "performance": alternative["revisionId"]},
        "local_schedule": {**window, "effectiveUntil": "2026-07-01T13:00"},
    })
    assert store.get_bundle(current["revisionId"]).status == "superseded"
    for at_time, effective_revision in (
        ("2026-07-01T15:59:59Z", current),
        ("2026-07-01T16:00:00Z", successor),
        ("2026-07-01T17:00:00Z", current),
    ):
        assert configuration.resolve(bundle_id=current["bundleId"], at_time=at_time)["bundle"]["revisionId"] == effective_revision["revisionId"]
        for recipe in recipes:
            replay = factory.resolve_recipe(recipe)
            assert replay.bundle_revision_id == saved.bundle_revision_id
            assert replay.bundle.sha256 == saved.bundle.sha256

    def corrupt(revision_id):
        with store._connect() as connection:
            connection.execute(
                "UPDATE configuration_bundle_revisions SET effective_from=? WHERE revision_id=?",
                ("2026-07-01T15:00:00+00:00", revision_id),
            )

    published = configuration.create_bundle(
        **common, bundle_id="replay-unrelated-published", version="1.0.0", local_schedule=window,
    )
    archived = configuration.create_bundle(
        **common, bundle_id="replay-unrelated-archived", version="1.0.0", local_schedule=window,
    )
    configuration.create_bundle(
        **common, bundle_id=archived["bundleId"], version="1.0.1", force_new_revision=True,
    )
    assert store.get_bundle(archived["revisionId"]).status == "superseded"
    for bundle in (published, archived):
        corrupt(bundle["revisionId"])
        with pytest.raises(ConfigurationIntegrityError):
            factory.resolve_bundle_revision(bundle["revisionId"], require_published=False)
    for recipe in recipes:
        replay = factory.resolve_recipe(recipe)
        assert replay.bundle_revision_id == saved.bundle_revision_id
        assert replay.source_effective_hash == saved.source_effective_hash
        _, exact = factory.resolve_bundle_revision(current["revisionId"], require_published=False)
        assert replay.effective_runtime_config_hash == exact.effective_runtime_config_hash

    matching = configuration.create_bundle(
        **common, bundle_id=current["bundleId"], version=current["version"], local_schedule=window,
        scope="tenant", tenant_id="TEST-REPLAY-MATCH",
    )
    legacy_page = store.legacy_bundle_page
    pages = []

    def single_row_page(after, *, limit):
        result = legacy_page(after, limit=1)
        pages.append(tuple(row.revision_id for row in result))
        return result

    monkeypatch.setattr(store, "legacy_bundle_page", single_row_page)
    for recipe in recipes:
        pages.clear()
        with pytest.raises(ValueError, match="historical_record_requires_readjudication"):
            factory.resolve_recipe(recipe)
        assert (matching["revisionId"],) in pages
        assert (current["revisionId"],) in pages
        assert len(pages) >= 2
    corrupt(matching["revisionId"])
    configuration._invalidate_cache()
    for recipe in recipes:
        with pytest.raises(ValueError, match="historical_record_requires_readjudication"):
            factory.resolve_recipe(recipe)
