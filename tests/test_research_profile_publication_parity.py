from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from baloot_platform.config import (
    build_bundle,
    validate_profile_payload,
    validate_research_profile_bundle,
)
from baloot_platform.configuration import (
    SQLiteConfigurationStore,
    VersionedConfigurationService,
)
from scripts.generate_profile_schemas import apply_release_constraints, infer


def _payload(root: Path, relative: str) -> dict[str, Any]:
    return yaml.safe_load((root / relative).read_text(encoding="utf-8"))


def _profiles(root: Path) -> dict[str, dict[str, Any]]:
    bundle = yaml.safe_load(
        (root / "configs/bundle.yaml").read_text(encoding="utf-8")
    )
    return {
        family: _payload(root, f"configs/{relative}")
        for family, relative in bundle["components"].items()
    }


def _set(value: dict[str, Any], path: str, replacement: Any) -> None:
    parts = path.split(".")
    current = value
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = replacement


def test_runtime_proposal_selector_is_optional_and_hash_bound(root: Path, rules) -> None:
    import json
    import jsonschema
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1, canonical_sha256

    profile = _payload(root, "configs/challenge/intelligent_research.yaml")
    schema = infer(profile)
    apply_release_constraints("challenge", schema, profile)
    quality_schema = schema["properties"]["scenarioResearchV4"]["properties"]["qualityExecution"]
    historical = profile["scenarioResearchV4"]["qualityExecution"]
    old = ScenarioQualityExecutionBudgetV1.build(historical)
    assert "candidateSource" not in old.to_dict()
    assert old.sha256 == canonical_sha256({key: value for key, value in old.to_dict().items() if key != "sha256"})
    policy = json.loads((root / "tests/fixtures/actor_interest_calibration.json").read_text())["cases"]["7189"]["policy"]
    proposed = {**historical, "candidateSource": "runtime_proposal_v1", "proposalSeedCount": 20000, "auditConstructedTrace": True,
                "earlyPlayPolicyId": "observer_safe_trick_control_v1", "exactPlayFromTrick": 7,
                "videoInterestingness": policy}
    parsed = ScenarioQualityExecutionBudgetV1.build(proposed)
    assert parsed.candidate_source == "runtime_proposal_v1"
    bounded = ScenarioQualityExecutionBudgetV1.build({**proposed, "proposalMaximumSeconds": 0.5})
    assert bounded.proposal_maximum_seconds == 0.5 and bounded.sha256 != parsed.sha256
    assert ScenarioQualityExecutionBudgetV1.build(bounded.to_dict()) == bounded
    jsonschema.validate({**proposed, "proposalMaximumSeconds": 0.5}, quality_schema)
    for limit in (None, True, 0, -1, float("inf"), historical["maximumWallSeconds"] + 1):
        with pytest.raises(ValueError, match="proposalMaximumSeconds"):
            ScenarioQualityExecutionBudgetV1.build({**proposed, "proposalMaximumSeconds": limit})
    assert ScenarioQualityExecutionBudgetV1.build(parsed.to_dict()) == parsed
    without = {key: value for key, value in proposed.items() if key not in {"candidateSource", "proposalSeedCount"}}
    assert parsed.sha256 != ScenarioQualityExecutionBudgetV1.build(without).sha256
    jsonschema.validate(proposed, quality_schema)
    from baloot_platform.research.ortools_research_executor import OrToolsUnifiedResearchExecutorV1
    from baloot_platform.research.scenario_contracts import ScenarioPlayingQualityPolicyV1, SCENARIO_DECISION_KINDS
    playing = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 2, "mode": "exact", "visibility": "complete_state", "decisions": {
            kind: {"utility": "awarded_point_margin", "budgetProfile": "late_game_exact_v1"}
            for kind in SCENARIO_DECISION_KINDS}})
    from types import SimpleNamespace
    from baloot_platform.core.rules import RuleBook
    rules = RuleBook(_payload(root, "configs/rules/saudi_federation_2023_v2.yaml"))
    executor = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
            bundle_revision_id="CBR-PROPOSAL-CONTRACT", bundle_hash="a" * 64,
            quality_policy=playing, quality_execution=parsed,
            quality_proof_profiles=profile["scenarioResearchV4"]["proofBudgets"]["profiles"])
    for origin, kind in (("exact_trace_prefix", "play"), ("search", "bid")):
        scenario = SimpleNamespace(origin=SimpleNamespace(kind=origin), decision_point={"kind": kind})
        executor._validate_runtime_proposal_scope(scenario)
        executor.runtime_proposals = {}
        with pytest.raises(ValueError, match="search play scenario"):
            executor._validate_runtime_proposal_scope(scenario)
        executor.runtime_proposals = None
    with pytest.raises(ValueError, match="runtime proposal queue is required"):
        executor._validate_runtime_proposal_scope(SimpleNamespace(
            origin=SimpleNamespace(kind="search"), decision_point={"kind": "play"}))
    assert executor.quality_execution.sha256 == parsed.sha256
    invalids = [{**proposed, "candidateSource": value} for value in (None, True, "unknown")]
    invalids += [{key: value for key, value in proposed.items() if key != missing}
                 for missing in ("earlyPlayPolicyId", "videoInterestingness", "auditConstructedTrace", "proposalSeedCount")]
    invalids += [{**proposed, "proposalSeedCount": value} for value in (True, 0, 1_000_001, 2.5)]
    invalids += [{**historical, "proposalSeedCount": 20000}]
    invalids += [{**proposed, "auditConstructedTrace": False},
                 {**proposed, "videoInterestingness": {"schemaVersion": 1, "maximumFinalMargin": 40,
                   "minimumPointSwing": 20, "minimumMeaningfulDecisions": 2, "maximumFeaturedTrick": 5}}]
    for invalid in invalids:
        with pytest.raises((ValueError, TypeError)):
            ScenarioQualityExecutionBudgetV1.build(invalid)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, quality_schema)


def test_tail_proof_copy_schema_preserves_historical_profiles(root: Path) -> None:
    import jsonschema

    profile = _payload(root, "configs/video/premium.yaml")
    schema = infer(profile)
    apply_release_constraints("video", schema, profile)
    clip_schema = schema["properties"]["exports"]["properties"]["decision_clip"]
    assert "outcome_seconds" not in clip_schema["required"]
    historical_clip = {key: value for key, value in profile["exports"]["decision_clip"].items()
                       if key != "outcome_seconds"}
    jsonschema.validate(historical_clip, clip_schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**historical_clip, "outcome_seconds": -1}, clip_schema)
    copy_schema = schema["properties"]["localizations"]
    labels = profile["localizations"]
    jsonschema.validate(labels, copy_schema)
    new_keys = {
        "video.analysis.proof_scope.actor_information_set_tail",
        "video.analysis.proof_scope.complete_state_tail",
        "video.analysis.worst_case_awarded_margin",
    }
    historical = {key: value for key, value in labels.items()
                  if key not in new_keys and not key.startswith("video.scoring.sun_tie.")}
    jsonschema.validate(historical, copy_schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**historical, next(iter(new_keys)): "partial"}, copy_schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**labels, "video.analysis.worst_case_awarded_margin": " "}, copy_schema)


def test_bid_opponent_policy_schema_matches_native_catalog(root: Path) -> None:
    import jsonschema
    from baloot_platform.research.ortools_play_tail import NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1
    profile = _payload(root, "configs/challenge/intelligent_research.yaml")
    schema = infer(profile)
    apply_release_constraints("challenge", schema, profile)
    quality = schema["properties"]["scenarioResearchV4"]["properties"]["qualityExecution"]
    assert quality["properties"]["bidOpponentPlayPolicyId"]["enum"] == sorted(NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS)
    original = profile["scenarioResearchV4"]["qualityExecution"]
    jsonschema.validate(original, quality)
    legacy = ScenarioQualityExecutionBudgetV1.build({key: value for key, value in original.items()
        if key != "completeStatePlayPolicyPair"})
    assert quality["properties"]["completeStatePlayPolicyPair"]["default"] is legacy.complete_state_play_policy_pair is False
    for paired in (False, True):
        configured = {**original, "completeStatePlayPolicyPair": paired}
        jsonschema.validate(configured, quality)
        budget = ScenarioQualityExecutionBudgetV1.build(configured)
        assert budget.complete_state_play_policy_pair is paired
        assert (budget.sha256 != legacy.sha256) is paired
        published = copy.deepcopy(profile)
        published["scenarioResearchV4"]["qualityExecution"] = configured
        validate_profile_payload("challenge", published, root=root)
    for invalid in (None, 1, "true", []):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**original, "completeStatePlayPolicyPair": invalid}, quality)
    for field, attribute in (("bidOpponentPlayPolicyId", "bid_opponent_play_policy_id"),
                            ("observedBidStrategyPlayFallbackId", "observed_bid_strategy_play_fallback_id")):
        assert quality["properties"][field]["enum"] == sorted(NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS)
        assert field not in quality["required"]
        assert "default" not in quality["properties"][field]
        for policy in NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS:
            configured = {**original, field: policy}
            jsonschema.validate(configured, quality)
            assert getattr(ScenarioQualityExecutionBudgetV1.build(configured), attribute) == policy
            published = copy.deepcopy(profile)
            published["scenarioResearchV4"]["qualityExecution"] = configured
            validate_profile_payload("challenge", published, root=root)
        for invalid in (None, True, "unknown", []):
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate({**original, field: invalid}, quality)
    # Native-only authoring must still generate the legacy closed wire schema.
    from baloot_platform.research.scenario_backend import materialize_default_policy_profiles_v1
    configured = {**original, "bidIncumbentProof": {"schemaVersion": 1,
        "maximumSeconds": original["bidIncumbentProof"]["maximumSeconds"],
        "maximumQueries": original["bidIncumbentProof"]["maximumQueries"],
        "playPolicies": materialize_default_policy_profiles_v1(profile["scenarioResearchV4"]["policyCatalog"])}}
    jsonschema.validate(configured, quality)
    assert ScenarioQualityExecutionBudgetV1.build(configured).bid_incumbent_proof["schemaVersion"] == 1
    search = dict(schemaVersion=1, upperBound=33, maximumWorlds=1, maximumSeconds=60,
        maximumQueries=1000, maximumWorldSeconds=30, maximumWorldQueries=500)
    configured = {key: value for key, value in original.items() if key != "bidOpponentPlayPolicyId"}
    configured["bidAlternativeProof"] = search
    jsonschema.validate(configured, quality)
    assert ScenarioQualityExecutionBudgetV1.build(configured).bid_alternative_proof["upperBound"] == 33
    ranked = {**configured, "bidAlternativeProof": {**search, "worldRankingId": "opponent_auction_value_v1"}}
    jsonschema.validate(ranked, quality)
    assert ScenarioQualityExecutionBudgetV1.build(ranked).bid_alternative_proof["worldRankingId"] == "opponent_auction_value_v1"
    shared = dict(schemaVersion=2, method="information_set_refutation_v1", upperBound=33,
        maximumSeconds=60, maximumQueries=1000)
    jsonschema.validate({**configured, "bidAlternativeProof": shared}, quality)
    assert ScenarioQualityExecutionBudgetV1.build({**configured,
        "bidAlternativeProof": shared}).bid_alternative_proof == shared
    for invalid in ({**shared, "maximumWorlds": 1}, {**shared, "method": "unknown"}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**configured, "bidAlternativeProof": invalid}, quality)
    for invalid in (None, {**search, "extra": 1}, {**search, "maximumWorlds": True}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**configured, "bidAlternativeProof": invalid}, quality)
    from baloot_platform.research.scenario_contracts import NATIVE_BID_REFUTATION_AUCTION_ACTIONS
    probe = dict(schemaVersion=1, maximumSeconds=60, maximumQueries=256,
        worldRankingId="opponent_auction_value_v1", opponentAuctionAction="sun")
    probe_schema = quality["properties"]["bidThresholdRefutation"]
    assert probe_schema["properties"]["opponentAuctionAction"]["enum"] == list(NATIVE_BID_REFUTATION_AUCTION_ACTIONS)
    configured = {**original, "bidThresholdRefutation": probe}
    jsonschema.validate(configured, quality)
    parsed = ScenarioQualityExecutionBudgetV1.build(configured)
    assert parsed.bid_threshold_refutation == probe
    assert ScenarioQualityExecutionBudgetV1.build(parsed.to_dict()).sha256 == parsed.sha256
    assert parsed.sha256 != ScenarioQualityExecutionBudgetV1.build(original).sha256
    published = copy.deepcopy(profile)
    published["scenarioResearchV4"]["qualityExecution"] = configured
    validate_profile_payload("challenge", published, root=root)
    for changed in ({"extra": 1}, {"maximumQueries": True}, {"maximumSeconds": 0},
                    {"worldRankingId": "unknown"}, {"opponentAuctionAction": "oracle"}):
        invalid = {**original, "bidThresholdRefutation": {**probe, **changed}}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, quality)
        with pytest.raises(ValueError):
            ScenarioQualityExecutionBudgetV1.build(invalid)
    unrestricted = {key: value for key, value in configured.items()
        if key != "bidOpponentPlayPolicyId"}
    jsonschema.validate(unrestricted, quality)
    parsed = ScenarioQualityExecutionBudgetV1.build(unrestricted)
    assert parsed.bid_threshold_refutation == probe
    assert ScenarioQualityExecutionBudgetV1.build(parsed.to_dict()).sha256 == parsed.sha256


def test_early_policy_schema_and_validator_use_native_catalog(root: Path) -> None:
    import jsonschema
    from baloot_platform.research.ortools_play_tail import NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1

    profile = _payload(root, "configs/challenge/intelligent_research.yaml")
    original = profile["scenarioResearchV4"]["qualityExecution"]
    schema = infer(profile)
    apply_release_constraints("challenge", schema, profile)
    quality = schema["properties"]["scenarioResearchV4"]["properties"]["qualityExecution"]
    assert quality["properties"]["earlyPlayPolicyId"]["enum"] == sorted(NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS)
    original.update(earlyPlayPolicyId=sorted(NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS)[0], exactPlayFromTrick=7)
    for policy in NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS:
        configured = {**original, "earlyPlayPolicyId": policy}
        jsonschema.validate(configured, quality)
        budget = ScenarioQualityExecutionBudgetV1.build(configured)
        assert budget.early_play_policy_id == policy
        assert ScenarioQualityExecutionBudgetV1.build(budget.to_dict()) == budget
    configured = {**original, "exactTailProof": True}
    jsonschema.validate(configured, quality)
    assert ScenarioQualityExecutionBudgetV1.build(configured).exact_tail_proof
    selected = {**configured, "exactTailBudgetProfile": "actor_information_set_exact_v1"}
    jsonschema.validate(selected, quality)
    assert ScenarioQualityExecutionBudgetV1.build(selected).exact_tail_budget_profile == "actor_information_set_exact_v1"
    for change in ({"exactTailProof": False}, {"exactTailBudgetProfile": "unknown"},
                   {"exactTailBudgetProfile": None}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**selected, **change}, quality)
    published = copy.deepcopy(profile)
    published["scenarioResearchV4"]["qualityExecution"] = configured
    validate_profile_payload("challenge", published, root=root)
    for invalid in (None, True, "unknown", [], {}):
        configured = {**original, "earlyPlayPolicyId": invalid}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(configured, quality)
        with pytest.raises(ValueError):
            ScenarioQualityExecutionBudgetV1.build(configured)


def test_video_interest_schema_roundtrips_and_preserves_historical_quality(root: Path) -> None:
    import jsonschema
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1
    profile = _payload(root, "configs/challenge/intelligent_research.yaml")
    configured = profile["scenarioResearchV4"]["qualityExecution"]
    schema = infer(profile)
    apply_release_constraints("challenge", schema, profile)
    quality = schema["properties"]["scenarioResearchV4"]["properties"]["qualityExecution"]
    jsonschema.validate(configured, quality)
    parsed = ScenarioQualityExecutionBudgetV1.build(configured)
    assert parsed.video_interestingness == configured["videoInterestingness"]
    assert ScenarioQualityExecutionBudgetV1.build(parsed.to_dict()) == parsed
    historical = {k: v for k, v in configured.items() if k != "videoInterestingness"}
    jsonschema.validate(historical, quality)
    assert ScenarioQualityExecutionBudgetV1.build(historical).sha256 != parsed.sha256
    for change in ({"minimumPointSwing": 19}, {"minimumMeaningfulDecisions": 1},
                   {"maximumFeaturedTrick": 6}, {"maximumFinalMargin": True}, {"unknown": 1}):
        invalid = {**configured, "videoInterestingness": {**configured["videoInterestingness"], **change}}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, quality)
        with pytest.raises(ValueError):
            ScenarioQualityExecutionBudgetV1.build(invalid)
    relaxed = {**configured, "videoInterestingness": {"schemaVersion": 2,
        "maximumFinalMargin": 40, "minimumPointSwing": 15,
        "maximumFeaturedTrick": 6, "minimumMeaningfulDecisions": 1}}
    jsonschema.validate(relaxed, quality)
    raw_policy = ScenarioQualityExecutionBudgetV1.build(relaxed)
    assert raw_policy.video_interestingness == relaxed["videoInterestingness"]
    assert ScenarioQualityExecutionBudgetV1.build(raw_policy.to_dict()) == raw_policy
    assert raw_policy.sha256 != parsed.sha256
    for change in ({"schemaVersion": True}, {"schemaVersion": "2"}, {"schemaVersion": 4},
                   {"minimumPointSwing": 14}, {"minimumMeaningfulDecisions": 0},
                   {"maximumFeaturedTrick": 7}, {"maximumFinalMargin": True}, {"unknown": 1}):
        invalid = {**relaxed, "videoInterestingness": {**relaxed["videoInterestingness"], **change}}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, quality)
        with pytest.raises(ValueError):
            ScenarioQualityExecutionBudgetV1.build(invalid)

    late = {**configured, "videoInterestingness": {**configured["videoInterestingness"],
        "schemaVersion": 3, "maximumFeaturedTrick": 7}}
    jsonschema.validate(late, quality)
    late_policy = ScenarioQualityExecutionBudgetV1.build(late)
    assert ScenarioQualityExecutionBudgetV1.build(late_policy.to_dict()) == late_policy
    assert late_policy.sha256 not in {parsed.sha256, raw_policy.sha256}
    owner_selected = {**late, "videoInterestingness": {**late["videoInterestingness"],
        "maximumFinalMargin": 40, "minimumMeaningfulDecisions": 1}}
    jsonschema.validate(owner_selected, quality)
    owner_policy = ScenarioQualityExecutionBudgetV1.build(owner_selected)
    assert ScenarioQualityExecutionBudgetV1.build(owner_policy.to_dict()) == owner_policy
    assert owner_policy.sha256 != late_policy.sha256
    for change in ({"maximumFeaturedTrick": 8}, {"minimumPointSwing": 19},
                   {"minimumMeaningfulDecisions": 0}):
        invalid = {**late, "videoInterestingness": {**late["videoInterestingness"], **change}}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, quality)
        with pytest.raises(ValueError):
            ScenarioQualityExecutionBudgetV1.build(invalid)


def test_published_ortools_product_revision_without_new_proof_copy_remains_readable(
    root: Path,
) -> None:
    product = _payload(root, "configs/product/studio_ortools.yaml")
    workflow = product["foundry_copy"]["groups"]["workflow"]
    newer_keys = {
        key for key in workflow
        if key.startswith(("research_decision_proof_", "research_scope_"))
    }
    assert len(newer_keys) == 15
    for key in newer_keys:
        workflow.pop(key)
    validate_profile_payload("product", product, root=root)

    workflow["research_scope_future_unknown"] = "مجهول"
    with pytest.raises(ValueError, match="Additional properties"):
        validate_profile_payload("product", product, root=root)


def test_native_delayed_payoff_copy_is_owned_and_historical_copy_is_optional(root: Path) -> None:
    product = _payload(root, "configs/product/studio_ortools.yaml")
    workflow = product["foundry_copy"]["groups"]["workflow"]
    challenge = _payload(root, "configs/challenge/intelligent_research.yaml")
    schema = infer(challenge)
    apply_release_constraints("challenge", schema, challenge)
    native = schema["properties"]["tactical_detectors"]["properties"]["nativeDelayedPayoff"]
    for node in (native, *native["properties"].values()):
        keys = [node["x-messageArKey"], node["x-helpArKey"],
                *node.get("x-enumMessageArKeys", {}).values()]
        for key in keys:
            assert workflow[key].strip()
            assert workflow[f"{key}_en"].strip()
    assert product["profile"]["version"] == "1.1.37"
    for key in list(workflow):
        if key.startswith("research_native_delayed_payoff_"):
            workflow.pop(key)
    validate_profile_payload("product", product, root=root)
    workflow["research_native_delayed_payoff_unknown"] = "مجهول"
    with pytest.raises(ValueError, match="Additional properties"):
        validate_profile_payload("product", product, root=root)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    (
        ("screening.shallow_samples", 1, "at least two samples"),
        (
            "screening.maximum_sampling_attempts",
            2,
            "sampling attempts violate safety bounds",
        ),
        ("confirmation.objective", "oracle_points", "unsupported trace analysis objective"),
        ("confirmation.deep_samples", 100_001, "sample count exceeds the safety bound"),
        ("confirmation.require_legal_replay", False, "must require legal replay"),
        (
            "decision_analysis_executor.screening.workerCount",
            5,
            "worker count must be between one and four",
        ),
        (
            "decision_analysis_executor.confirmation.maximumPendingPerWorker",
            3,
            "pending tasks per worker",
        ),
        (
            "decision_analysis_executor.finalistReanalysis.maximumSeconds",
            0.01,
            "timeout must be between",
        ),
        (
            "decision_analysis_executor.screening.pollSeconds",
            1.1,
            "poll interval",
        ),
        (
            "decision_analysis_executor.id",
            "",
            "decision analysis executor identity",
        ),
        (
            "decision_analysis_executor.schemaVersion",
            2,
            "unsupported decision analysis executor schema",
        ),
        ("selection_policy.id", "", "selection profile identity"),
        (
            "selection_policy.allowedTruthLabels",
            ["unresolved"],
            "supported truth labels",
        ),
        (
            "selection_policy.minimumSurroundingQuality",
            100.1,
            "minimum surrounding quality",
        ),
        ("thrill_profile.revision", "", "thrill profile identity"),
        ("thrill_profile.minimumCriticalPoints", 0, "thrill profile gates"),
        ("deep_analysis.id", "", "deep analysis profile identity"),
        (
            "deep_analysis.maximumOracleRemainingCards",
            33,
            "oracle endgame window",
        ),
        ("research_comparison.revision", "", "comparison profile identity"),
        (
            "research_comparison.requireZeroSemanticDefects",
            False,
            "must require zero semantic defects",
        ),
        (
            "research_comparison.minimumDistinctSelectedFinalistsPerLane",
            0,
            "distinct-finalist minimum",
        ),
        (
            "research_comparison.minimumAttemptedSeedsPerLane",
            0,
            "attempt minimum",
        ),
        (
            "research_comparison.requiredPromotionCpuBoundary",
            "wall_clock_v1",
            "isolated process CPU evidence",
        ),
    ),
)
def test_challenge_publication_uses_runtime_typed_profile_contracts(
    root: Path,
    path: str,
    replacement: Any,
    message: str,
) -> None:
    challenge = _payload(root, "configs/challenge/progressive_mining_v2.yaml")
    _set(challenge, path, replacement)
    with pytest.raises(ValueError):
        validate_profile_payload("challenge", challenge, root=root)
    profiles = _profiles(root)
    profiles["challenge"] = challenge
    with pytest.raises(ValueError, match=message):
        validate_research_profile_bundle(profiles)


@pytest.mark.parametrize(
    ("family", "path", "replacement", "message"),
    (
        ("analysis", "objective", "oracle_points", "unsupported analysis objective"),
        (
            "analysis",
            "exact_search_max_remaining",
            33,
            "exact-search threshold must be in",
        ),
        (
            "analysis",
            "maximum_sampling_attempts",
            100,
            "sampling attempts violate safety bounds",
        ),
        ("belief", "particles.resampling", "random", "systematic resampling"),
        ("belief", "particles.maximum", 100_001, "particle bounds"),
        (
            "belief",
            "particles.minimum_effective_fraction",
            0.0,
            "effective-sample fraction",
        ),
    ),
)
def test_analysis_and_belief_profiles_share_execution_builders(
    root: Path,
    family: str,
    path: str,
    replacement: Any,
    message: str,
) -> None:
    relative = {
        "analysis": "configs/analysis/deep.yaml",
        "belief": "configs/belief/particle_v1.yaml",
    }[family]
    payload = _payload(root, relative)
    _set(payload, path, replacement)
    with pytest.raises(ValueError):
        validate_profile_payload(family, payload, root=root)
    profiles = _profiles(root)
    profiles[family] = payload
    with pytest.raises(ValueError, match=message):
        validate_research_profile_bundle(profiles)


def test_bundle_validation_enforces_analysis_and_trace_belief_bounds(
    root: Path,
) -> None:
    profiles = _profiles(root)
    validate_research_profile_bundle(profiles)

    analysis_mismatch = copy.deepcopy(profiles)
    analysis_mismatch["analysis"]["requested_samples"] = 5_000
    validate_profile_payload(
        "analysis", analysis_mismatch["analysis"], root=root
    )
    with pytest.raises(ValueError, match="analysis sample count violates belief bounds"):
        validate_research_profile_bundle(analysis_mismatch)

    trace_mismatch = copy.deepcopy(profiles)
    trace_mismatch["challenge"]["confirmation"]["deep_samples"] = 5_000
    trace_mismatch["challenge"]["confirmation"][
        "maximum_sampling_attempts"
    ] = 6_000
    validate_profile_payload(
        "challenge", trace_mismatch["challenge"], root=root
    )
    with pytest.raises(ValueError, match="trace analysis sample count violates belief bounds"):
        validate_research_profile_bundle(trace_mismatch)


def test_incompatible_draft_bundle_is_rejected_before_compilation(
    root: Path,
    tmp_path: Path,
) -> None:
    governance = _payload(root, "configs/configuration/control_plane_v19.yaml")
    service = VersionedConfigurationService(
        SQLiteConfigurationStore(tmp_path / "configuration.sqlite3"),
        root=root,
        profile=governance,
    )
    resolved = service.bootstrap_packaged_bundle(actor_id="TEST")
    analysis = copy.deepcopy(resolved["profiles"]["analysis"])
    analysis["profile"] = dict(analysis["profile"])
    analysis["profile"].update(
        {"id": "incompatible_analysis", "version": "99.0.0"}
    )
    analysis["requested_samples"] = 5_000
    draft = service.create_profile(
        family="analysis", payload=analysis, created_by="TEST"
    )
    revisions = {
        family: value["revisionId"]
        for family, value in resolved["profileRevisions"].items()
    }
    revisions["analysis"] = draft["revisionId"]
    with pytest.raises(ValueError, match="analysis sample count violates belief bounds"):
        service.create_bundle(
            bundle_id="invalid_research_draft",
            version="1.0.0",
            mode="full",
            component_revisions=revisions,
            created_by="TEST",
            allow_draft_components=True,
        )
    assert service.store.list_bundles(bundle_id="invalid_research_draft") == []


def test_legacy_rows_remain_readable_but_cannot_be_requalified(
    root: Path,
    tmp_path: Path,
) -> None:
    governance = _payload(root, "configs/configuration/control_plane_v19.yaml")
    service = VersionedConfigurationService(
        SQLiteConfigurationStore(tmp_path / "legacy.sqlite3"),
        root=root,
        profile=governance,
    )
    resolved = service.bootstrap_packaged_bundle(actor_id="TEST")
    legacy_analysis = copy.deepcopy(resolved["profiles"]["analysis"])
    legacy_analysis["profile"] = {
        "id": "legacy_incompatible_analysis",
        "version": "1.0.0",
        "status": "published",
    }
    legacy_analysis["requested_samples"] = 5_000
    row = service.store.create_profile(
        family="analysis",
        profile_id="legacy_incompatible_analysis",
        version="1.0.0",
        scope="global",
        scope_key="*",
        tenant_id=None,
        environment=None,
        payload=legacy_analysis,
        status="published",
        created_by="LEGACY",
    )
    revisions = {
        family: value["revisionId"]
        for family, value in resolved["profileRevisions"].items()
    }
    revisions["analysis"] = row.revision_id
    legacy_bundle = service.store.create_bundle(
        bundle_id="legacy_incompatible_bundle",
        version="1.0.0",
        mode="full",
        scope="global",
        scope_key="*",
        tenant_id=None,
        environment=None,
        component_revisions=revisions,
        status="published",
        created_by="LEGACY",
    )
    readback = service.resolve(revision_id=legacy_bundle.revision_id)
    assert readback["profiles"]["analysis"]["requested_samples"] == 5_000
    service.set_bundle_compiler(lambda _: {"publishable": True})
    with pytest.raises(ValueError, match="analysis sample count violates belief bounds"):
        service.compile_bundle(legacy_bundle.revision_id)


def test_generated_schema_contract_matches_runtime_profile_limits(root: Path) -> None:
    profiles = _profiles(root)
    challenge_schema = infer(profiles["challenge"])
    apply_release_constraints("challenge", challenge_schema, profiles["challenge"])
    challenge = challenge_schema["properties"]
    assert challenge["confirmation"]["properties"]["objective"]["enum"] == [
        "contract_success",
        "cvar",
        "low_volatility",
        "match_point_margin",
        "match_win_probability",
        "round_margin",
    ]
    assert challenge["decision_analysis_executor"]["properties"]["screening"][
        "properties"
    ]["workerCount"]["maximum"] == 4
    assert challenge["selection_policy"]["properties"]["allowedTruthLabels"][
        "items"
    ]["enum"] == ["debatable", "exact_optimal", "robust_best_known"]
    assert challenge["research_comparison"]["properties"][
        "requireZeroSemanticDefects"
    ]["const"] is True
    comparison = challenge["research_comparison"]["properties"]
    assert comparison["schemaVersion"]["const"] == 2
    assert comparison["minimumAttemptedSeedsPerLane"]["minimum"] == 1
    assert comparison["minimumDistinctSelectedFinalistsPerLane"]["maximum"] == 1_000_000
    assert comparison["requiredPromotionCpuBoundary"]["const"] == (
        "isolated_process_cpu_v1"
    )

    analysis_schema = infer(profiles["analysis"])
    apply_release_constraints("analysis", analysis_schema, profiles["analysis"])
    assert analysis_schema["properties"]["requested_samples"]["maximum"] == 100_000

    belief_schema = infer(profiles["belief"])
    apply_release_constraints("belief", belief_schema, profiles["belief"])
    assert belief_schema["properties"]["particles"]["properties"]["resampling"][
        "const"
    ] == "systematic"


def test_packaged_bundle_still_satisfies_publication_and_runtime_contracts(
    root: Path,
) -> None:
    assert build_bundle(root).sha256
