from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from baloot_platform import config as config_module


_LEGACY_CANONICAL_SHA256 = (
    "512c9ccedbcfdda0897d9f0978961dce531e418949c87345113ea079c474380e"
)


def _load(root: Path, name: str) -> dict[str, Any]:
    return yaml.safe_load(
        (root / "configs/challenge" / name).read_text(encoding="utf-8")
    )


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_semantics(value: dict[str, Any]) -> None:
    config_module._semantic_validate("challenge", value)


def test_legacy_challenge_profile_remains_unchanged(root: Path) -> None:
    legacy = _load(root, "progressive_mining_v2.yaml")

    assert legacy["profile"]["version"] == "2.8.4"
    assert "scenarioResearchV4" not in legacy
    assert _canonical_sha256(legacy) == _LEGACY_CANONICAL_SHA256
    _validate_semantics(legacy)


def test_five_second_validation_profile_preserves_frozen_authority(root: Path) -> None:
    from baloot_platform.config import build_bundle, validate_profile_payload

    historical = _load(root, "intelligent_research.yaml")
    revised = _load(root, "intelligent_research_validation_5s.yaml")
    expected = copy.deepcopy(historical)
    expected["profile"].update(version="3.14.5", description=revised["profile"]["description"])
    expected["scenarioResearchV4"]["searchBudgets"]["profiles"]["interactive_validation_v1"]["maximumWallSeconds"] = 5.0
    assert revised == expected
    assert "2.148 seconds" in revised["profile"]["description"]
    assert "5% local CPU cap" in revised["profile"]["description"]
    assert historical["scenarioResearchV4"]["searchBudgets"]["profiles"]["interactive_validation_v1"]["maximumWallSeconds"] == 1.0
    for profile in (historical, revised):
        validate_profile_payload("challenge", profile, root=root)
        assert profile["scenarioResearchV4"]["proofBudgets"]["profiles"]["interactive_validation_v1"]["maximumWallSeconds"] == 1.0
    old_bundle = build_bundle(root, bundle_path="configs/bundle_ortools_research_native_v3_intelligent.yaml")
    new_bundle = build_bundle(root, bundle_path="configs/bundle_ortools_research_native_v3_intelligent_validation_5s.yaml")
    assert old_bundle.sha256 != new_bundle.sha256
    for family, artifact in old_bundle.components.items():
        assert (artifact.sha256 == new_bundle.components[family].sha256) is (family != "challenge")


def test_actor_tail_candidate_binds_ratified_rules_and_strict_symbolic_authority(root):
    from baloot_platform.config import build_bundle, validate_profile_payload
    from baloot_platform.research.scenario_backend import ScenarioResearchAuthorityV1
    from baloot_platform.research.scenario_contracts import ScenarioPlayingQualityPolicyV1

    profile = _load(root, "actor_visible_tail_v3.yaml")
    validate_profile_payload("challenge", profile, root=root)
    authority = ScenarioResearchAuthorityV1.from_challenge_profile(profile)
    execution = authority.quality_execution
    assert execution.exact_tail_proof and execution.exact_play_from_trick == 7
    assert execution.early_play_policy_id == "observer_safe_trick_control_v1"
    assert execution.video_interestingness == {
        "schemaVersion": 4, "maximumFinalMargin": 40, "minimumPointSwing": 20,
        "maximumFeaturedTrick": 7, "minimumMeaningfulDecisions": 1,
        "minimumConsistentWorlds": 2, "minimumBaselineRegret": 20,
        "minimumNextBestRegret": 20,
        "baselinePolicyId": "observer_safe_public_two_trick_control_v6",
        "allowTieRuleOnlyLessons": False,
    }
    historical = _load(root, "intelligent_research_validation_5s.yaml")["scenarioResearchV4"]
    expected = copy.deepcopy(historical["proofBudgets"])
    expected["profiles"]["actor_information_set_exact_v1"]["policySearch"] = "time_budgeted_symbolic_v1"
    assert profile["scenarioResearchV4"]["proofBudgets"] == expected
    assert authority.authoring_defaults["playingQuality"]["exactBudgetProfileByVisibility"][
        "actor_information_set"] == "actor_information_set_exact_v1"
    defaults = authority.authoring_defaults["playingQuality"]
    policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 2, "mode": "budgeted", "visibility": "actor_information_set",
        "decisions": {kind: {"utility": "awarded_point_margin",
            "budgetProfile": defaults["budgetProfile"],
            "horizon": {"unit": "tricks" if kind == "play" else "plies", "value": 1},
            "boundaryEvaluator": defaults["boundaryEvaluatorByDecision"][kind]}
            for kind in ("bid", "escalation", "project", "play")},
    })
    early = policy.request_for("play", "north", play_index=23, execution_budget=execution)
    assert early.mode == "bounded" and early.budget_profile == "bounded_v1"
    for kind in ("bid", "escalation", "project"):
        assert policy.request_for(kind, "north", execution_budget=execution).budget_profile == "bounded_v1"
    tail = policy.request_for("play", "north", play_index=24, execution_budget=execution)
    assert tail.mode == "exact" and tail.budget_profile == "actor_information_set_exact_v1"
    assert defaults["budgetProfile"] == historical["authoringDefaults"]["playingQuality"]["budgetProfile"]
    assert profile["scenarioResearchV4"]["searchBudgets"] == historical["searchBudgets"]
    bundle = build_bundle(root, bundle_path="configs/bundle_ortools_research_actor_tail.yaml")
    assert bundle.rules.id == "saudi_federation_2023_v8"
    assert bundle.component("challenge")["profile"]["version"] == "3.14.6"
    invalid = copy.deepcopy(profile)
    invalid["scenarioResearchV4"]["qualityExecution"]["exactTailBudgetProfile"] = "unpublished"
    with pytest.raises(ValueError, match="exact tail.*(unpublished|unavailable)"):
        _validate_semantics(invalid)
    with pytest.raises(ValueError, match="exact tail.*(unpublished|unavailable)"):
        ScenarioResearchAuthorityV1.from_challenge_profile(invalid)
    invalid = copy.deepcopy(profile)
    invalid["scenarioResearchV4"]["qualityExecution"]["unknownTailField"] = True
    with pytest.raises(ValueError):
        validate_profile_payload("challenge", invalid, root=root)


def test_delayed_payoff_draft_preserves_frozen_profiles_and_budgets(root: Path) -> None:
    from baloot_platform.config import build_bundle, validate_profile_payload
    from baloot_platform.tactics import TacticalDetectorProfileV1

    historical = _load(root, "intelligent_research_validation_5s.yaml")
    revised = _load(root, "native_delayed_payoff_v3.yaml")
    expected = copy.deepcopy(historical)
    expected["profile"].update(
        version="3.14.7", status="draft", description=revised["profile"]["description"],
    )
    expected["tactical_detectors"].update(
        revision="2", nativeDelayedPayoff={"gainMetric": "raw_card_points", "minimumGain": 8},
    )
    assert revised == expected
    validate_profile_payload("challenge", revised, root=root)
    invalid = copy.deepcopy(revised)
    invalid["profile"]["version"] = "3.14.8"
    with pytest.raises(ValueError, match="identity/version is unsupported"):
        _validate_semantics(invalid)
    selected = TacticalDetectorProfileV1.build(revised["tactical_detectors"])
    assert selected.native_delayed_gain_metric == "raw_card_points"
    assert selected.native_delayed_minimum_gain == 8
    assert "nativeDelayedPayoff" not in historical["tactical_detectors"]
    old_bundle = build_bundle(root, bundle_path="configs/bundle_ortools_research_ratified_projects.yaml")
    new_path = "configs/bundle_ortools_research_native_v3_delayed_payoff.yaml"
    new_bundle = build_bundle(root, bundle_path=new_path)
    assert new_bundle.rules.id == "saudi_federation_2023_v8"
    assert new_bundle.component("challenge") == revised
    assert yaml.safe_load((root / new_path).read_text(encoding="utf-8"))["profile"]["status"] == "draft"
    assert old_bundle.sha256 != new_bundle.sha256
    for family, artifact in old_bundle.components.items():
        assert (artifact.sha256 == new_bundle.components[family].sha256) is (family != "challenge")
    assert build_bundle(root).components["challenge"].sha256 != new_bundle.components["challenge"].sha256
    for relative in ("configs/challenge/native_delayed_payoff_v3.yaml", new_path):
        assert (root / relative).read_bytes() == (
            root / "src/baloot_platform/resources" / relative
        ).read_bytes()


@pytest.mark.parametrize("tactic", ["small_card_big_gain", "large_card_sacrifice"])
def test_delayed_payoff_draft_selects_native_causal_catalog(root: Path, tactic: str) -> None:
    from baloot_platform.research.scenario_backend import OrToolsScenarioResearchBackendV2

    manifests = []
    for name in ("intelligent_research_validation_5s.yaml", "native_delayed_payoff_v3.yaml"):
        backend = OrToolsScenarioResearchBackendV2(
            challenge_profile=_load(root, name),
            bundle_identity={
                "bundleRevisionId": "CBR-TEST-DELAYED-DRAFT",
                "semanticBundleHash": "a" * 64,
                "effectiveRuntimeConfigHash": "b" * 64,
            },
        )
        manifests.append(backend.lowering_manifest.capability_manifest)
    old, revised = manifests
    assert old.manifest_hash != revised.manifest_hash
    old_row = next(row for row in old.capability("tactic.verified").argument_contexts if row.argument_value == tactic)
    row = next(row for row in revised.capability("tactic.verified").argument_contexts if row.argument_value == tactic)
    assert old_row.causal_verification_available is False
    assert row.causal_verification_available is True
    assert row.required_utility == "awarded_point_margin"
    assert {(context.origin, context.decision_kind, context.proof_mode, context.visibility)
            for context in row.execution_contexts} == {
        (origin, "play", "exact", "complete_state")
        for origin in ("search", "exact_trace_prefix")
    }


@pytest.mark.parametrize("metric", ["raw_card_points", "awarded_point_margin"])
@pytest.mark.parametrize("gain", [0, 20.5, 100])
def test_native_delayed_payoff_config_and_optional_schema(root, metric, gain):
    from jsonschema import Draft202012Validator
    from scripts.generate_profile_schemas import apply_release_constraints, infer
    from baloot_platform.tactics import TacticalDetectorProfileV1

    profile = _load(root, "intelligent_research.yaml")
    before = copy.deepcopy(profile)
    old = TacticalDetectorProfileV1.build(profile["tactical_detectors"])
    schema = infer(profile)
    apply_release_constraints("challenge", schema, profile)
    tactical_schema = schema["properties"]["tactical_detectors"]
    assert "nativeDelayedPayoff" not in tactical_schema["required"]
    validator = Draft202012Validator(tactical_schema)
    validator.validate(profile["tactical_detectors"])
    for invalid in (None, {}, {"gainMetric": "unknown", "minimumGain": 20},
                    {"gainMetric": metric, "minimumGain": True},
                    {"gainMetric": metric, "minimumGain": -1},
                    {"gainMetric": metric, "minimumGain": 101},
                    {"gainMetric": metric, "minimumGain": 20, "extra": 1}):
        assert list(validator.iter_errors({**profile["tactical_detectors"], "nativeDelayedPayoff": invalid}))
    profile["tactical_detectors"]["nativeDelayedPayoff"] = {"gainMetric": metric, "minimumGain": gain}
    validator.validate(profile["tactical_detectors"])
    _validate_semantics(profile)
    selected = TacticalDetectorProfileV1.build(profile["tactical_detectors"])
    assert selected.sha256 != old.sha256
    assert selected.to_dict()["nativeDelayedPayoff"] == {"gainMetric": metric, "minimumGain": gain}
    assert "nativeDelayedPayoff" not in old.to_dict()
    assert TacticalDetectorProfileV1.build(before["tactical_detectors"]).sha256 == old.sha256


@pytest.mark.parametrize("value", [
    None, {}, {"gainMetric": "unknown", "minimumGain": 20},
    {"gainMetric": "raw_card_points", "minimumGain": True},
    {"gainMetric": "raw_card_points", "minimumGain": "20"},
    *({"gainMetric": "raw_card_points", "minimumGain": gain}
      for gain in (-1, 101, float("nan"), float("inf"), -float("inf"))),
    {"gainMetric": "raw_card_points", "minimumGain": 20, "extra": 1},
])
def test_native_delayed_payoff_config_reuses_canonical_validation(root, value):
    profile = _load(root, "intelligent_research.yaml")
    profile["tactical_detectors"]["nativeDelayedPayoff"] = value
    with pytest.raises(ValueError, match="native delayed payoff"):
        _validate_semantics(profile)


@pytest.mark.parametrize("filename", ["progressive_mining_v2.yaml", "progressive_mining_ortools.yaml"])
def test_native_delayed_payoff_rejects_non_native_backend(root, filename):
    from jsonschema import Draft202012Validator
    from scripts.generate_profile_schemas import apply_release_constraints, infer

    profile = _load(root, filename)
    _validate_semantics(profile)
    profile["tactical_detectors"]["nativeDelayedPayoff"] = {
        "gainMetric": "raw_card_points", "minimumGain": 20,
    }
    with pytest.raises(ValueError, match="nativeDelayedPayoff requires"):
        _validate_semantics(profile)
    schema = infer(profile)
    apply_release_constraints("challenge", schema, profile)
    assert list(Draft202012Validator(schema).iter_errors(profile))


@pytest.mark.parametrize("filename,native", [
    ("intelligent_research.yaml", True),
    ("progressive_mining_ortools_v3.yaml", True),
    ("progressive_mining_v2.yaml", False),
    ("progressive_mining_ortools.yaml", False),
])
def test_generated_challenge_schema_keeps_frozen_profiles_and_guards_native_payoff(root, filename, native):
    from jsonschema import Draft202012Validator

    profile = _load(root, filename)
    before = copy.deepcopy(profile)
    schema = json.loads((root / "schemas/profiles/challenge.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    validator.validate(profile)
    profile["tactical_detectors"]["nativeDelayedPayoff"] = {
        "gainMetric": "raw_card_points", "minimumGain": 20,
    }
    if native:
        validator.validate(profile)
    else:
        assert list(validator.iter_errors(profile))
    profile["tactical_detectors"].pop("nativeDelayedPayoff")
    assert profile == before


def test_intelligent_bid_incumbent_profile_schema_and_legacy_snapshot(root: Path) -> None:
    from jsonschema import Draft202012Validator
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1
    profile = _load(root, "intelligent_research.yaml")
    assert profile["profile"]["version"] == "3.14.4"
    schema = json.loads((root / "schemas/profiles/challenge.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    incumbent_schema = schema["properties"]["scenarioResearchV4"]["properties"]["qualityExecution"]["properties"]["bidIncumbentProof"]
    assert incumbent_schema["properties"]["completeStateAlternativeFirst"]["default"] is True
    validator.validate(profile)
    _validate_semantics(profile)
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1
    proof_raw = profile["scenarioResearchV4"]["proofBudgets"]["profiles"]["late_game_exact_v1"]
    proof_budget = ScenarioProofBudgetV1.build("late_game_exact_v1", proof_raw)
    assert proof_budget.native_observed.counterexample_objective == "minimum_utility"
    historical = copy.deepcopy(profile)
    historical["profile"]["version"] = "3.14.3"
    historical_raw = historical["scenarioResearchV4"]["proofBudgets"]["profiles"]["late_game_exact_v1"]
    historical_raw["nativeObserved"].pop("counterexampleObjective")
    historical_raw["nativeObserved"]["schemaVersion"] = 1
    _validate_semantics(historical)
    assert ScenarioProofBudgetV1.build("late_game_exact_v1", historical_raw).sha256 != proof_budget.sha256
    assert {key: value for key, value in proof_raw.items() if key != "nativeObserved"} == {
        key: value for key, value in historical_raw.items() if key != "nativeObserved"}
    execution = ScenarioQualityExecutionBudgetV1.build(profile["scenarioResearchV4"]["qualityExecution"])
    assert execution.bid_incumbent_proof["maximumSeconds"] == 20
    assert execution.bid_incumbent_proof["maximumQueries"] == 64
    assert execution.bid_incumbent_proof["schemaVersion"] == 2
    assert execution.bid_incumbent_proof["nativePlayPolicyId"] == "observer_safe_partner_feed_compact_v1"
    assert execution.bid_incumbent_proof["observedProjectPolicyId"] == "full_hand_project_max_value_v1"
    assert execution.bid_opponent_play_policy_id == "observer_safe_trick_control_v1"
    assert execution.complete_state_play_policy_pair is True
    native = copy.deepcopy(profile)
    native["scenarioResearchV4"]["qualityExecution"]["bidIncumbentProof"] = dict(schemaVersion=2,
        maximumSeconds=20.0, maximumQueries=64, nativePlayPolicyId="observer_safe_trick_control_v1")
    validator.validate(native)
    _validate_semantics(native)
    native_execution = ScenarioQualityExecutionBudgetV1.build(native["scenarioResearchV4"]["qualityExecution"])
    assert native_execution.bid_incumbent_proof["nativePlayPolicyId"] == "observer_safe_trick_control_v1"
    assert ScenarioQualityExecutionBudgetV1.build(native_execution.to_dict()) == native_execution
    for order in (True, False):
        reordered = copy.deepcopy(native)
        reordered["scenarioResearchV4"]["qualityExecution"]["bidIncumbentProof"]["completeStateAlternativeFirst"] = order
        validator.validate(reordered)
        _validate_semantics(reordered)
        parsed = ScenarioQualityExecutionBudgetV1.build(reordered["scenarioResearchV4"]["qualityExecution"])
        assert parsed.bid_incumbent_proof["completeStateAlternativeFirst"] is order
        assert parsed.sha256 != native_execution.sha256
        assert parsed.observed_bid_incumbent_proof == native_execution.observed_bid_incumbent_proof
        assert ScenarioQualityExecutionBudgetV1.build(parsed.to_dict()) == parsed
    for invalid_order in (0, 1, "false", None):
        reordered["scenarioResearchV4"]["qualityExecution"]["bidIncumbentProof"]["completeStateAlternativeFirst"] = invalid_order
        assert not validator.is_valid(reordered)
        with pytest.raises((ValueError, TypeError)):
            _validate_semantics(reordered)
    secure = copy.deepcopy(native)
    secure["scenarioResearchV4"]["qualityExecution"]["bidIncumbentProof"]["nativePlayPolicyId"] = "observer_safe_partner_feed_secure_control_v1"
    secure["scenarioResearchV4"]["qualityExecution"]["bidOpponentPlayPolicyId"] = "observer_safe_partner_feed_secure_control_v1"
    validator.validate(secure)
    _validate_semantics(secure)
    secure_execution = ScenarioQualityExecutionBudgetV1.build(secure["scenarioResearchV4"]["qualityExecution"])
    assert ScenarioQualityExecutionBudgetV1.build(secure_execution.to_dict()) == secure_execution
    assert secure_execution.sha256 != native_execution.sha256
    guard = copy.deepcopy(secure)
    guard_id = "observer_safe_partner_feed_guard_protection_v1"
    guard["scenarioResearchV4"]["qualityExecution"]["bidIncumbentProof"]["nativePlayPolicyId"] = guard_id
    guard["scenarioResearchV4"]["qualityExecution"]["bidOpponentPlayPolicyId"] = guard_id
    validator.validate(guard)
    _validate_semantics(guard)
    guard_execution = ScenarioQualityExecutionBudgetV1.build(guard["scenarioResearchV4"]["qualityExecution"])
    assert ScenarioQualityExecutionBudgetV1.build(guard_execution.to_dict()) == guard_execution
    assert guard_execution.sha256 != secure_execution.sha256
    closed = copy.deepcopy(guard)
    closed_id = "observer_safe_partner_feed_closed_trick_v1"
    closed["scenarioResearchV4"]["qualityExecution"]["bidIncumbentProof"]["nativePlayPolicyId"] = closed_id
    closed["scenarioResearchV4"]["qualityExecution"]["bidOpponentPlayPolicyId"] = closed_id
    validator.validate(closed)
    _validate_semantics(closed)
    closed_execution = ScenarioQualityExecutionBudgetV1.build(closed["scenarioResearchV4"]["qualityExecution"])
    assert ScenarioQualityExecutionBudgetV1.build(closed_execution.to_dict()) == closed_execution
    assert closed_execution.sha256 != guard_execution.sha256
    full_hand = copy.deepcopy(native)
    full_hand["scenarioResearchV4"]["qualityExecution"]["bidIncumbentProof"]["observedProjectPolicyId"] = "full_hand_project_max_value_v1"
    full_hand["scenarioResearchV4"]["qualityExecution"]["bidIncumbentProof"]["observedAuctionPolicyId"] = "prefer_legal_sun_then_pass_v1"
    validator.validate(full_hand)
    _validate_semantics(full_hand)
    root_actor = copy.deepcopy(full_hand)
    root_actor["scenarioResearchV4"]["qualityExecution"]["bidIncumbentProof"]["observedAuctionPolicyId"] = "prefer_root_actor_sun_partner_pass_v1"
    validator.validate(root_actor)
    _validate_semantics(root_actor)
    root_actor_execution = ScenarioQualityExecutionBudgetV1.build(root_actor["scenarioResearchV4"]["qualityExecution"])
    assert ScenarioQualityExecutionBudgetV1.build(root_actor_execution.to_dict()) == root_actor_execution
    assert root_actor_execution.sha256 != ScenarioQualityExecutionBudgetV1.build(full_hand["scenarioResearchV4"]["qualityExecution"]).sha256
    nonplay = {"schemaVersion": 1, "auctionAction": "pass", "escalationAction": "decline",
        "projectPolicyId": "full_hand_project_max_value_v1"}
    complete = copy.deepcopy(native)
    complete["scenarioResearchV4"]["qualityExecution"]["bidIncumbentProof"]["completeStateNonplayPolicy"] = nonplay
    validator.validate(complete)
    _validate_semantics(complete)
    complete_execution = ScenarioQualityExecutionBudgetV1.build(complete["scenarioResearchV4"]["qualityExecution"])
    assert ScenarioQualityExecutionBudgetV1.build(complete_execution.to_dict()) == complete_execution
    assert complete_execution.sha256 != native_execution.sha256
    assert complete_execution.observed_bid_incumbent_proof == native_execution.bid_incumbent_proof
    complete_execution.observed_bid_incumbent_proof["maximumQueries"] = 1
    assert complete_execution.bid_incumbent_proof["maximumQueries"] != 1
    for change in ({"schemaVersion": True}, {"auctionAction": "sun"}, {"escalationAction": None},
                   {"projectPolicyId": "known_project_max_value_v1"}, {"extra": 1}):
        invalid = copy.deepcopy(complete)
        invalid["scenarioResearchV4"]["qualityExecution"]["bidIncumbentProof"]["completeStateNonplayPolicy"].update(change)
        assert not validator.is_valid(invalid)
        with pytest.raises((ValueError, TypeError)):
            _validate_semantics(invalid)
    for change in ({"nativePlayPolicyId": "unknown"}, {"nativePlayPolicyId": None},
                   {"playPolicies": {}}, {"schemaVersion": 1},
                   {"observedAuctionPolicyId": "unknown"}, {"observedAuctionPolicyId": None},
                   {"observedProjectPolicyId": "unknown"}, {"observedProjectPolicyId": None}):
        invalid = copy.deepcopy(native)
        invalid["scenarioResearchV4"]["qualityExecution"]["bidIncumbentProof"].update(change)
        assert not validator.is_valid(invalid)
        with pytest.raises((ValueError, TypeError)):
            _validate_semantics(invalid)
    legacy = copy.deepcopy(profile)
    legacy["profile"]["version"] = "3.6.0"
    legacy["scenarioResearchV4"]["authoringDefaults"]["playingQuality"].pop("exactBudgetProfileByVisibility")
    legacy["scenarioResearchV4"]["qualityExecution"].pop("bidIncumbentProof")
    legacy["scenarioResearchV4"]["qualityExecution"].pop("allowExactInputQuality")
    validator.validate(legacy)
    _validate_semantics(legacy)
    assert ScenarioQualityExecutionBudgetV1.build(
        legacy["scenarioResearchV4"]["qualityExecution"]).bid_incumbent_proof is None
    for key, value in (("maximumSeconds", 0), ("maximumSeconds", 600),
            ("maximumSeconds", True), ("maximumQueries", 0), ("extra", 1)):
        changed = copy.deepcopy(profile)
        changed["scenarioResearchV4"]["qualityExecution"]["bidIncumbentProof"][key] = value
        assert not validator.is_valid(changed)
        with pytest.raises((ValueError, TypeError)):
            _validate_semantics(changed)


def test_ortools_challenge_profile_has_closed_deterministic_authority(
    root: Path,
) -> None:
    profile = _load(root, "progressive_mining_ortools.yaml")
    authority = profile["scenarioResearchV4"]

    assert profile["profile"]["version"] == "2.9.0"
    assert authority["backend"] == "ortools_cp_sat_v2"
    assert authority["planner"] == "tail_first_v1"
    assert authority["deterministicExecution"] is True
    assert authority["policyCatalog"] == {
        "schemaVersion": 1,
        "defaultPolicyId": "fixed_action_map_v1",
        "profiles": {
            "fixed_action_map_v1": {
                "version": "1.0.0",
                "materialization": "seeded_legal_action_order_v1",
            }
        },
    }
    assert set(authority["proofBudgets"]["profiles"]) == {
        "interactive_validation_v1",
        "bounded_v1",
        "late_game_exact_v1",
        "actor_information_set_exact_v1",
        "offline_heavyweight_v1",
    }
    assert set(authority["searchBudgets"]["profiles"]) == set(
        authority["proofBudgets"]["profiles"]
    )
    assert all(
        budget["maximumWorkers"] == 1
        for budget in authority["proofBudgets"]["profiles"].values()
    )
    assert all(
        budget["maximumWorkers"] == 1
        for budget in authority["searchBudgets"]["profiles"].values()
    )
    assert authority["searchBudgets"]["profiles"]["late_game_exact_v1"][
        "maximumShardWorkers"
    ] == 5
    _validate_semantics(profile)


def test_native_v3_1_authoring_default_is_frozen_in_new_profile_only(
    root: Path,
) -> None:
    historical = _load(root, "progressive_mining_ortools_v3.yaml")
    current = _load(root, "progressive_mining_ortools_v3_1.yaml")

    assert historical["profile"]["version"] == "3.0.0"
    assert "authoringDefaults" not in historical["scenarioResearchV4"]
    assert current["profile"] == {
        "id": "progressive_challenge_mining_v3",
        "version": "3.1.0",
        "status": "governed_candidate",
    }
    assert current["scenarioResearchV4"]["authoringDefaults"] == {
        "schemaVersion": 1,
        "resources": {"seedsPerShard": 1},
        "playingQuality": {
            "budgetProfile": "bounded_v1",
            "boundaryEvaluatorByDecision": {
                "bid": "auction_hand_margin_v1",
                "escalation": "current_raw_point_margin_v1",
                "project": "current_raw_point_margin_v1",
                "play": "current_raw_point_margin_v1",
            },
        },
    }
    _validate_semantics(historical)
    _validate_semantics(current)

    invalid = copy.deepcopy(current)
    invalid["scenarioResearchV4"]["authoringDefaults"]["resources"][
        "seedsPerShard"
    ] = 0
    with pytest.raises(ValueError, match="seedsPerShard must be a positive integer"):
        _validate_semantics(invalid)

    invalid = copy.deepcopy(current)
    invalid["scenarioResearchV4"]["authoringDefaults"]["playingQuality"][
        "budgetProfile"
    ] = "unpublished_v1"
    with pytest.raises(ValueError, match="playing quality budget profile is unpublished"):
        _validate_semantics(invalid)

    invalid = copy.deepcopy(current)
    invalid["scenarioResearchV4"]["authoringDefaults"]["playingQuality"][
        "boundaryEvaluatorByDecision"
    ]["play"] = "auction_hand_margin_v1"
    with pytest.raises(ValueError, match="playing quality boundary evaluator is unsupported"):
        _validate_semantics(invalid)


def test_ortools_profile_requires_scenario_research_authority(root: Path) -> None:
    profile = _load(root, "progressive_mining_ortools.yaml")
    profile.pop("scenarioResearchV4")

    with pytest.raises(ValueError, match="2.9.0 requires scenarioResearchV4"):
        _validate_semantics(profile)


def test_legacy_profile_rejects_scenario_research_authority(root: Path) -> None:
    legacy = _load(root, "progressive_mining_v2.yaml")
    ortools = _load(root, "progressive_mining_ortools.yaml")
    legacy["scenarioResearchV4"] = ortools["scenarioResearchV4"]

    with pytest.raises(ValueError, match="requires profile version 2.9.0"):
        _validate_semantics(legacy)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda block: block.update({"unknown": True}),
            "scenarioResearchV4 shape is invalid",
        ),
        (
            lambda block: block.update({"backend": "runtime"}),
            "backend is unsupported",
        ),
        (
            lambda block: block["proofBudgets"]["profiles"][
                "late_game_exact_v1"
            ].update({"maximumWorkers": 2}),
            "proof workers must equal one",
        ),
        (
            lambda block: block["searchBudgets"]["profiles"].pop("bounded_v1"),
            "searchBudgets.profiles shape is invalid",
        ),
        (
            lambda block: block["searchBudgets"]["profiles"][
                "late_game_exact_v1"
            ].update({"maximumShardWorkers": 33}),
            "shard workers must not exceed 32",
        ),
        (
            lambda block: block["searchBudgets"]["profiles"]["bounded_v1"].update(
                {"maximumVariants": 1001, "maximumCandidates": 1000}
            ),
            "variants cannot exceed candidates",
        ),
        (
            lambda block: block["diversityDefaults"].update(
                {"allowSymmetryOnlyVariants": True}
            ),
            "symmetry-only variants are forbidden",
        ),
        (
            lambda block: block["policyCatalog"]["profiles"][
                "fixed_action_map_v1"
            ].update({"materialization": "heuristic_bot_v1"}),
            "policy identity is unsupported",
        ),
        (
            lambda block: block["cacheLimits"].update({"maximumEntries": False}),
            "must be a positive integer",
        ),
    ),
)
def test_scenario_research_authority_fails_closed(
    root: Path,
    mutation,
    message: str,
) -> None:
    profile = _load(root, "progressive_mining_ortools.yaml")
    block = copy.deepcopy(profile["scenarioResearchV4"])
    mutation(block)
    profile["scenarioResearchV4"] = block

    with pytest.raises(ValueError, match=message):
        _validate_semantics(profile)


@pytest.mark.parametrize("section,field", [
    ("proofBudgets", "maximumWallSeconds"), ("proofBudgets", "maximumCpuSeconds"),
    ("searchBudgets", "maximumWallSeconds"), ("searchBudgets", "maximumDeterministicTime")])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_scenario_research_profile_rejects_nonfinite_limits(root, section, field, value):
    profile = _load(root, "progressive_mining_ortools.yaml")
    profile["scenarioResearchV4"][section]["profiles"]["late_game_exact_v1"][field] = value
    with pytest.raises(ValueError, match="must be positive"):
        _validate_semantics(profile)


@pytest.mark.parametrize("native_version", [1, 2, 3])
def test_native_budget_revision_is_explicit_and_complete(root, native_version):
    profile = _load(root, "progressive_mining_ortools.yaml")
    budgets = profile["scenarioResearchV4"]["proofBudgets"]
    native = {"schemaVersion": 1, "maximumIterations": 20, "maximumQueries": 40,
              "maximumObservations": 60, "maximumBoundPrefixes": 30,
              "maximumCachedPrefixesPerBranch": 10}
    if native_version >= 2:
        native.update(schemaVersion=native_version, counterexampleObjective="minimum_utility")
    if native_version == 3:
        native.update(playAllocationMethod="deterministic_cp_sat_batched_prefix_v1")
    for raw in budgets["profiles"].values():
        raw["nativeObserved"] = dict(native)
    with pytest.raises(ValueError, match="shape is invalid"):
        _validate_semantics(profile)
    budgets["version"] = "2.1.0"
    _validate_semantics(profile)
    next(iter(budgets["profiles"].values())).pop("nativeObserved")
    with pytest.raises(ValueError, match="shape is invalid"):
        _validate_semantics(profile)


def test_native_objective_generated_editor_schema_matches_api(root):
    import runpy
    from jsonschema import Draft202012Validator, ValidationError
    generator = runpy.run_path(str(root / "scripts/generate_profile_schemas.py"))
    profile = _load(root, "intelligent_research.yaml")
    schema = generator["infer"](profile)
    generator["apply_release_constraints"]("challenge", schema, profile)
    validator = Draft202012Validator(schema)
    validator.validate(profile)
    profile["scenarioResearchV4"]["qualityExecution"]["auditConstructedTrace"] = True
    validator.validate(profile)
    _validate_semantics(profile)
    profile["scenarioResearchV4"]["qualityExecution"]["auditConstructedTrace"] = "true"
    with pytest.raises(ValidationError):
        validator.validate(profile)
    with pytest.raises(ValueError):
        _validate_semantics(profile)
    profile["scenarioResearchV4"]["qualityExecution"].pop("auditConstructedTrace")
    native = profile["scenarioResearchV4"]["proofBudgets"]["profiles"]["bounded_v1"]["nativeObserved"]
    native.update(schemaVersion=2, counterexampleObjective="minimum_utility")
    validator.validate(profile)
    _validate_semantics(profile)
    native.update(schemaVersion=3, playAllocationMethod="deterministic_cp_sat_batched_prefix_v1")
    validator.validate(profile)
    _validate_semantics(profile)
    for invalid in ("unknown", "deterministic_cp_sat_prefix_v1", True, None):
        native["playAllocationMethod"] = invalid
        with pytest.raises(ValidationError):
            validator.validate(profile)
        with pytest.raises(ValueError):
            _validate_semantics(profile)
    native["playAllocationMethod"] = "deterministic_cp_sat_batched_prefix_v1"
    for invalid in ("unknown", "feasibility", True, None):
        native["counterexampleObjective"] = invalid
        with pytest.raises(ValidationError):
            validator.validate(profile)


def test_exact_quality_authoring_defaults_are_closed_and_published(root):
    profile = _load(root, "intelligent_research.yaml")
    defaults = profile["scenarioResearchV4"]["authoringDefaults"]["playingQuality"]
    assert defaults["exactBudgetProfileByVisibility"] == {
        "complete_state": "offline_heavyweight_v1",
        "actor_information_set": "offline_heavyweight_v1",
    }
    _validate_semantics(profile)
    for value in ("unknown", 1, None, []):
        invalid = copy.deepcopy(profile)
        invalid["scenarioResearchV4"]["authoringDefaults"]["playingQuality"]["exactBudgetProfileByVisibility"]["complete_state"] = value
        with pytest.raises(ValueError, match="exact budget profile is unpublished"):
            _validate_semantics(invalid)
    defaults["exactBudgetProfileByVisibility"].pop("complete_state")
    with pytest.raises(ValueError, match="shape is invalid"):
        _validate_semantics(profile)
