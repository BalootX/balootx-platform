from __future__ import annotations

import ast
import copy
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.ortools_exact_state import _effective_prefix_hash
from baloot_platform.research.recipe_v4 import _validate_selected_action
from baloot_platform.research.scenario_backend import (
    LEGACY_SCENARIO_RESEARCH_BACKEND,
    ORTOOLS_SCENARIO_RESEARCH_BACKEND,
    ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3,
    ORTOOLS_SCENARIO_GENERATION_POLICY,
    OrToolsScenarioResearchBackendV2,
    ScenarioResearchAuthorityV1,
)
from baloot_platform.research.scenario_contracts import (
    SCENARIO_DECISION_KINDS, ExactTracePrefixV1, ScenarioPlayingQualityPolicyV1, canonical_sha256,
)
from baloot_platform.research.scenario_execution_capabilities import (
    build_scenario_execution_capabilities_v1,
)


def test_native_backend_reads_certification_from_the_frozen_bundle(root, rules, monkeypatch):
    from baloot_platform.config import ProfileBundle
    from baloot_platform.research import service as module

    certification = {"binding": "frozen"}
    bundle = ProfileBundle(rules=rules, components={
        "challenge": SimpleNamespace(data={"scenarioResearchV4": {"backend": "ortools_cp_sat_v3"}}),
        "adjudication": SimpleNamespace(data={}),
        "certification": SimpleNamespace(data=certification),
    }, mode="full", bundle_profile={}, sha256="a" * 64)
    context = SimpleNamespace(bundle=bundle, bundle_revision_id="CBR-TEST",
        effective_runtime_config_hash="b" * 64)
    calls = []

    def load_catalog(profile, selected_rules, repository):
        calls.append((profile, selected_rules, repository))
        return SimpleNamespace(data={"bound": True}, sha256="c" * 64)

    monkeypatch.setattr(module, "load_bound_rule_source_catalog", load_catalog)
    monkeypatch.setattr(module, "OrToolsScenarioResearchBackendV2", lambda **kwargs: kwargs)
    result = module.ResearchFoundryServiceV1._ortools_scenario_backend_v2(context, repository_root=root)
    assert calls == [(certification, rules, root)]
    assert result["rule_source_catalog"] == {"data": {"bound": True}, "sha256": "c" * 64}
    del bundle.components["certification"]
    with pytest.raises(KeyError, match="certification"):
        module.ResearchFoundryServiceV1._ortools_scenario_backend_v2(context, repository_root=root)


@pytest.mark.parametrize("upper,threshold,expected", [(-50, 4, True), (4, 4, False), (5, 4, False), (-4, -3, True)])
def test_native_threshold_refutation_comparison(upper, threshold, expected):
    from ortools.sat.python import cp_model
    from baloot_platform.research.ortools_information_set_proof import build_native_threshold_refutation_model_v1
    model, claim = build_native_threshold_refutation_model_v1(upper, threshold)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    assert solver.boolean_value(claim) is expected
    for invalid in (True, 1.0, "1"):
        with pytest.raises(ValueError):
            build_native_threshold_refutation_model_v1(invalid, threshold)
        with pytest.raises(ValueError):
            build_native_threshold_refutation_model_v1(upper, invalid)


@pytest.mark.parametrize("change", [{"schemaVersion": True}, {"schemaVersion": 2},
    {"strategicCoverage": True}, {"branches": {}}, {"branches": [None]}])
def test_threshold_upper_reader_rejects_invalid_schema_before_solver(change, monkeypatch):
    from baloot_platform.research import result_authority as authority
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid schema reached proof validation")
    monkeypatch.setattr(authority, "_validate_streamed_bid_preparation", forbidden)
    receipt = {"kind": "native_bid_streamed_alternative_preparation_v1", "schemaVersion": 1,
        "strategicCoverage": False, "branches": [], **change}
    with pytest.raises(ValueError, match="schema mismatch"):
        authority.reverify_native_bid_world_upper_v1(receipt, rules=None, exact=None,
            source_prefix=None, action="pass", opponent_policy={}, maximum_model_bytes=100,
            maximum_branches=1, maximum_conflicts=1, maximum_seconds=1, maximum_queries=1)


@pytest.mark.parametrize("exact", [None, {}, SimpleNamespace(next_actor="north")])
def test_threshold_upper_reader_rejects_unowned_play_policy_before_solver(monkeypatch, exact):
    from baloot_platform.research import result_authority as authority
    def forbidden(*args, **kwargs):
        pytest.fail("unowned play policy reached proof validation")
    monkeypatch.setattr(authority, "_validate_streamed_bid_preparation", forbidden)
    with pytest.raises(ValueError, match="requires an owned exact state"):
        authority.reverify_native_bid_world_upper_v1(
            {"kind": "native_bid_streamed_alternative_preparation_v1"}, rules=None,
            exact=exact, source_prefix=None, action="pass", opponent_policy={},
            maximum_model_bytes=100, maximum_branches=1, maximum_conflicts=1,
            maximum_seconds=1, maximum_queries=1,
            opponent_play_policy_id="observer_safe_trick_control_v1")


def _bid_incumbent_configuration():
    policies = {}
    for seat in ("north", "south", "west", "east"):
        core = {"schemaVersion": 1, "seat": seat, "policyId": "fixed_action_map_v1", "version": "1.0.0",
            "seed": 0, "choices": [], "materialization": "seeded_legal_action_order_v1"}
        policies[seat] = {key: item for key, item in core.items() if key != "seat"}
        policies[seat]["profileHash"] = canonical_sha256(core)
    return {"schemaVersion": 1, "maximumSeconds": 2.0, "maximumQueries": 10, "playPolicies": policies}


@pytest.mark.parametrize("policy", ["observer_safe_trick_control_v1", "observer_safe_tactical_category_v1",
                                    None, True, 1, [], "unknown"])
@pytest.mark.parametrize("field,attribute", [
    ("bidOpponentPlayPolicyId", "bid_opponent_play_policy_id"),
    ("observedBidStrategyPlayFallbackId", "observed_bid_strategy_play_fallback_id")])
def test_bid_opponent_play_policy_configuration_is_owned_and_closed(policy, field, attribute):
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1
    raw = dict(schemaVersion=1, maximumWallSeconds=10, maximumEvidenceBytes=10000,
               maximumEscalationDecisions=20)
    original = ScenarioQualityExecutionBudgetV1.build(raw)
    assert field not in original.to_dict()
    configured = {**raw, field: policy}
    if type(policy) is str and policy.startswith("observer_safe_"):
        parsed = ScenarioQualityExecutionBudgetV1.build(configured)
        assert getattr(parsed, attribute) == policy
        assert parsed.sha256 != original.sha256
        assert ScenarioQualityExecutionBudgetV1.build(parsed.to_dict()) == parsed
        with pytest.raises(ValueError, match="checksum"):
            ScenarioQualityExecutionBudgetV1.build({**configured, "sha256": original.sha256})
    else:
        with pytest.raises(ValueError):
            ScenarioQualityExecutionBudgetV1.build(configured)


@pytest.mark.parametrize("mutation", (None, "native", "unknown", "null", "seconds", "boolean", "queries", "seat", "policy_hash"))
def test_bid_incumbent_configuration_is_closed_and_immutable(mutation):
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1
    raw = {"schemaVersion": 1, "maximumWallSeconds": 10, "maximumEvidenceBytes": 10000,
        "maximumEscalationDecisions": 20, "bidIncumbentProof": _bid_incumbent_configuration()}
    if mutation == "native":
        raw["bidIncumbentProof"] = dict(schemaVersion=2, maximumSeconds=2.0,
            maximumQueries=10, nativePlayPolicyId="observer_safe_trick_control_v1")
        for invalid in ({"nativePlayPolicyId": "unknown"}, {"nativePlayPolicyId": None},
                        {"playPolicies": _bid_incumbent_configuration()["playPolicies"]}, {"schemaVersion": True}):
            with pytest.raises((ValueError, TypeError)):
                ScenarioQualityExecutionBudgetV1.build({**raw, "bidIncumbentProof": {**raw["bidIncumbentProof"], **invalid}})
    elif mutation == "unknown":
        raw["bidIncumbentProof"]["extra"] = 1
    elif mutation == "null":
        raw["bidIncumbentProof"] = None
    elif mutation == "seconds":
        raw["bidIncumbentProof"]["maximumSeconds"] = 10
    elif mutation == "boolean":
        raw["bidIncumbentProof"]["maximumSeconds"] = True
    elif mutation == "queries":
        raw["bidIncumbentProof"]["maximumQueries"] = 0
    elif mutation == "seat":
        raw["bidIncumbentProof"]["playPolicies"].pop("east")
    elif mutation == "policy_hash":
        raw["bidIncumbentProof"]["playPolicies"]["east"]["seed"] = 1
    if mutation not in (None, "native"):
        with pytest.raises((ValueError, TypeError)):
            ScenarioQualityExecutionBudgetV1.build(raw)
        return
    value = ScenarioQualityExecutionBudgetV1.build(raw)
    assert ScenarioQualityExecutionBudgetV1.build(value.to_dict()) == value
    raw["bidIncumbentProof"]["maximumSeconds"] = 5
    assert value.bid_incumbent_proof["maximumSeconds"] == 2
    rendered = value.to_dict()
    rendered["bidIncumbentProof"]["maximumSeconds"] = 4
    assert value.bid_incumbent_proof["maximumSeconds"] == 2
    with pytest.raises(ValueError, match="checksum"):
        ScenarioQualityExecutionBudgetV1.build(rendered)
    assert "bidIncumbentProof" not in ScenarioQualityExecutionBudgetV1.build(
        {key: item for key, item in raw.items() if key != "bidIncumbentProof"}).to_dict()


@pytest.mark.parametrize("version", (1, 2))
def test_bid_alternative_search_configuration_is_closed_and_immutable(version):
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1
    raw = dict(schemaVersion=1, maximumWallSeconds=10, maximumEvidenceBytes=10000,
        maximumEscalationDecisions=20, maximumHiddenWorldsPerDecision=2)
    search = dict(schemaVersion=1, upperBound=-1, maximumWorlds=2, maximumSeconds=4,
        maximumQueries=30, maximumWorldSeconds=2, maximumWorldQueries=15)
    if version == 2:
        search = dict(schemaVersion=2, method="information_set_refutation_v1",
            upperBound=-1, maximumSeconds=4, maximumQueries=30)
    value = ScenarioQualityExecutionBudgetV1.build({**raw, "bidAlternativeProof": search})
    assert ScenarioQualityExecutionBudgetV1.build(value.to_dict()) == value
    assert value.sha256 != ScenarioQualityExecutionBudgetV1.build(raw).sha256
    search["upperBound"] = 0
    assert value.bid_alternative_proof["upperBound"] == -1
    rendered = value.to_dict()
    rendered["bidAlternativeProof"]["upperBound"] = 1
    with pytest.raises(ValueError, match="checksum"):
        ScenarioQualityExecutionBudgetV1.build(rendered)
    for mutation in ({"schemaVersion": True}, {"extra": 1}, {"upperBound": True},
            {"maximumWorlds": 3}, {"maximumQueries": 0}, {"maximumWorldQueries": 31},
            {"maximumSeconds": 10}, {"maximumWorldSeconds": 5}, {"maximumSeconds": float("nan")},
            {"worldRankingId": None}, {"worldRankingId": "unknown"}):
        with pytest.raises(ValueError):
            ScenarioQualityExecutionBudgetV1.build({**raw, "bidAlternativeProof": {**search, **mutation}})
    with pytest.raises(TypeError):
        ScenarioQualityExecutionBudgetV1.build({**raw, "bidAlternativeProof": None})
    with pytest.raises(ValueError, match="fixed opponent"):
        ScenarioQualityExecutionBudgetV1.build({**raw, "bidAlternativeProof": search,
            "bidOpponentPlayPolicyId": "observer_safe_trick_control_v1"})
    if version == 1:
        ranked = ScenarioQualityExecutionBudgetV1.build({**raw,
            "bidAlternativeProof": {**search, "worldRankingId": "opponent_auction_value_v1"}})
        assert ranked.sha256 != value.sha256
        assert ScenarioQualityExecutionBudgetV1.build(ranked.to_dict()) == ranked
    else:
        for mutation in ({"method": "unknown"}, {"method": None}, {"maximumWorlds": 1}):
            with pytest.raises(ValueError):
                ScenarioQualityExecutionBudgetV1.build({**raw, "bidAlternativeProof": {**search, **mutation}})


@pytest.mark.parametrize("mutation", (None, "feasible", "missing", "duplicate", "order", "owner", "scope",
    "rules", "team", "threshold", "bindings", "policy", "policy_model", "query_model", "query_scope",
    "query_team", "query_threshold", "query_mode", "unknown", "incomplete", "query_bindings", "floor_type",
    "floor_range", "floor_above_witness", "optimal_gap", "lower", "policy_hash", "scope_hash", "query_hash",
    "trusted_rules", "trusted_models", "branch_scope_hash", "query_model_missing", "extra_query",
    "query_count", "binding_count", "master"))
def test_native_incumbent_floor_reader_rejects_rehashed_substitution(mutation):
    from baloot_platform.research.result_authority import validate_native_incumbent_lower_bound_v2
    from baloot_platform.research.ortools_proof import DeclarativeActionPolicyV1

    def hashed(core):
        return {**core, "sha256": canonical_sha256(core)}

    policies = {}
    for seat in ("north", "south"):
        core = dict(schemaVersion=1, seat=seat, policyId="fixed_action_map_v1", version="1.0.0", seed=0,
                    choices=[], materialization="seeded_legal_action_order_v1")
        policies[seat] = {key: value for key, value in core.items() if key != "seat"}
        policies[seat]["profileHash"] = canonical_sha256(core)
    identities = {seat: DeclarativeActionPolicyV1.build(seat, raw).identity_dict() for seat, raw in policies.items()}
    branch_scope = hashed({"rulesHash": "rules"})
    if mutation == "branch_scope_hash":
        branch_scope["sha256"] = "corrupt"
    scope = {"rulesHash": "rules", "targetTeam": "north_south", "utilityBounds": [-10, 10],
        "incumbentPlayPoliciesHash": canonical_sha256(policies),
        "branches": [{"modelHash": source, "scope": branch_scope} for source in ("a", "b")]}
    queries = []
    for source, floor in (("a", -3), ("b", 7)):
        query = {"schemaVersion": 1, "kind": "native_round_outcome_query_v1", "baseModelHash": source + "-policy",
            "queryModelHash": source + "-optimized",
            "scope": branch_scope, "targetTeam": "north_south", "utility": "awarded_point_margin", "threshold": 11,
            "status": "witness_found", "solverStatus": "optimal", "objectiveLowerBound": floor,
            "candidate": {"utility": floor},
            **{kind + "PolicyBindings": [] for kind in ("auction", "escalation", "project", "play")}}
        policy = {"team": "north_south", "auctionAction": None, "escalationAction": None,
            "scope": "future_nonplay_constants_and_ranked_play", "playPolicies": identities,
            "playPolicyModelHash": source + "-policy"}
        receipt = {"schemaVersion": 1, "kind": "native_constant_policy_round_query_v1", "strategicCoverage": False,
            "status": "queried", "sourceModelHash": source, "scope": branch_scope, "rulesHash": "rules",
            "targetTeam": "north_south", "threshold": 11, "bindings": [], "policy": policy}
        if source == "a":
            if mutation == "feasible":
                query.update(solverStatus="feasible", candidate={"utility": 0})
            receipt_changes = {"owner": ("sourceModelHash", "other"), "scope": ("scope", {}),
                "rules": ("rulesHash", "other"), "team": ("targetTeam", "west_east"),
                "threshold": ("threshold", 10), "bindings": ("bindings", [{"status": "bound"}]),
                "incomplete": ("status", "incomplete")}
            query_changes = {"query_model": ("baseModelHash", "other"), "query_scope": ("scope", {}),
                "query_team": ("targetTeam", "west_east"), "query_threshold": ("threshold", 10),
                "query_mode": ("queryMode", "feasibility_witness"), "unknown": ("solverStatus", "unknown"),
                "query_bindings": ("playPolicyBindings", [{"status": "bound"}]),
                "floor_type": ("objectiveLowerBound", True), "floor_range": ("objectiveLowerBound", -11),
                "floor_above_witness": ("objectiveLowerBound", -2), "optimal_gap": ("objectiveLowerBound", -4)}
            if mutation in receipt_changes:
                key, value = receipt_changes[mutation]
                receipt[key] = value
            if mutation in query_changes:
                key, value = query_changes[mutation]
                query[key] = value
            if mutation == "policy":
                policy["team"] = "west_east"
            elif mutation == "policy_model":
                policy["playPolicyModelHash"] = "other"
            elif mutation == "query_model_missing":
                query.pop("queryModelHash")
        receipt["query"] = hashed(query)
        if source == "a" and mutation == "query_hash":
            receipt["query"]["sha256"] = "corrupt"
        queries.append({"branchModelHash": source, "purpose": "incumbent_lower_bound", "receipt": hashed(receipt)})
    if mutation == "missing":
        queries.pop()
    elif mutation == "duplicate":
        queries.append(queries[0])
    elif mutation == "order":
        queries.reverse()
    elif mutation == "policy_hash":
        scope["incumbentPlayPoliciesHash"] = "other"
    elif mutation == "extra_query":
        queries.append({**queries[0], "purpose": "unaccounted_query"})
    strategy = hashed({"schemaVersion": 1, "kind": "native_symbolic_round_policy_bounds_v2",
        "strategicCoverage": False, "status": "incomplete", "scope": scope,
        "queriesUsed": 0 if mutation == "query_count" else len(queries),
        "bindingQueriesUsed": 1 if mutation == "binding_count" else 0,
        "masters": [{}] if mutation == "master" else [],
        "scopeHash": "other" if mutation == "scope_hash" else canonical_sha256(scope),
        "lowerBound": 7 if mutation == "lower" else -3, "queries": queries})
    owner = {"target_team": "north_south", "play_policies": policies,
        "rules_hash": "other" if mutation == "trusted_rules" else "rules",
        "source_model_hashes": ("a", "other") if mutation == "trusted_models" else ("a", "b")}
    if mutation in (None, "feasible"):
        assert validate_native_incumbent_lower_bound_v2(strategy, **owner) == -3
        from baloot_platform.research.result_authority import validate_native_bid_dominance_claim_v1
        strategy = hashed({**{key: item for key, item in strategy.items() if key != "sha256"},
            "status": "scoped_incumbent_bound_complete", "upperBound": 10})
        alternative = {"action": "pass", "upperBound": -3, "sourceModelHashes": ["alternative"],
            "preparation": {"constructionComplete": True,
                "branches": [{"retained": True, "sourceModelHash": "alternative"}]},
            "queries": [{"status": "queried", "sourceModelHash": "alternative", "targetTeam": "west_east",
                "threshold": 3, "bindings": [], "query": {"status": "no_matching_outcome",
                    "solverStatus": "infeasible", "targetTeam": "west_east", "threshold": 3}}]}
        value = {"kind": "native_bid_selected_action_dominance_v2", "rulesHash": "rules",
            "targetTeam": "north_south", "selectedAction": "sun", "legalActions": ["pass", "sun"],
            "status": "selected_action_optimal", "incumbentPlayPolicies": policies,
            "optimalActionCount": None, "optimalActionCountLowerBound": 1, "optimalActionCountUpperBound": 2,
            "selectedValueProof": {"status": "scoped_incumbent_bound_complete", "strategy": strategy, "queriesUsed": 3,
                "preparation": {"queriesUsed": 1, "constructionComplete": True, "branches": [
                    {"retained": True, "sourceModelHash": item} for item in ("a", "b")]}},
            "alternatives": [alternative], "actionQuality": {"optimalActionCount": None,
                "optimalActionCountLowerBound": 1, "optimalActionCountUpperBound": 2,
                "optimalActionsComplete": False, "optimalActions": ["sun"], "actualAction": "sun",
                "actualActionOptimal": True, "actualValue": None, "bestValue": None, "regret": 0,
                "selectedValueBounds": {"lower": -3, "upper": 10}}}
        value["qualificationModel"] = _claim_qualification_model(value)
        validate_native_bid_dominance_claim_v1(value, legal_actions=("pass", "sun"))
        for alteration in ("exact_value", "missing_value", "overlap", "upper", "model", "usage"):
            changed = copy.deepcopy(value)
            if alteration == "exact_value":
                changed["actionQuality"]["actualValue"] = -3
            elif alteration == "missing_value":
                changed["actionQuality"].pop("bestValue")
            elif alteration == "overlap":
                changed["alternatives"][0]["upperBound"] = -2
            elif alteration == "upper":
                changed["actionQuality"]["selectedValueBounds"]["upper"] = 11
            elif alteration == "usage":
                changed["selectedValueProof"]["queriesUsed"] = 2
            else:
                changed["qualificationModel"]["modelProto"] = ""
            with pytest.raises(ValueError):
                validate_native_bid_dominance_claim_v1(changed, legal_actions=("pass", "sun"))
    else:
        with pytest.raises(ValueError, match="incumbent lower bound"):
            validate_native_incumbent_lower_bound_v2(strategy, **owner)


@pytest.mark.parametrize("mutation", (None, "threshold", "team", "model", "scope", "unknown", "bindings", "bound_type", "bound_order"))
def test_retained_infeasible_branch_requires_owned_whole_domain_proof(mutation):
    from baloot_platform.research.result_authority import _validate_native_final_branch_proofs_v2
    branch_scope = {"rulesHash": "rules"}
    query = {"status": "no_matching_outcome", "solverStatus": "infeasible", "baseModelHash": "source",
        "scope": branch_scope, "targetTeam": "west_east", "threshold": 11,
        **{kind + "PolicyBindings": [] for kind in ("auction", "escalation", "project", "play")}}
    if mutation == "threshold":
        query["threshold"] = 10
    elif mutation == "team":
        query["targetTeam"] = "north_south"
    elif mutation == "model":
        query["baseModelHash"] = "substitute"
    elif mutation == "scope":
        query["scope"] = {}
    elif mutation == "unknown":
        query["solverStatus"] = "unknown"
    elif mutation == "bindings":
        query["playPolicyBindings"] = [{"status": "bound"}]
    query["sha256"] = canonical_sha256(query)
    value = {"schemaVersion": 1, "kind": "native_symbolic_round_policy_bounds_v2", "strategicCoverage": False,
        "scope": {"rulesHash": "rules", "targetTeam": "north_south", "utilityBounds": [-10, 7],
            "branches": [{"modelHash": "source", "scope": branch_scope}]}, "policy": {}, "upperBound": 7,
        "queries": [{"branchModelHash": "source", "receipt": query}], "refinements": {},
        "finalBranchProofs": [{"branchModelHash": "source", "infeasible": True, "queryHash": canonical_sha256(query)}]}
    if mutation == "bound_type":
        value["scope"]["utilityBounds"][1] = True
    elif mutation == "bound_order":
        value["scope"]["utilityBounds"] = [8, 7]
    value["scopeHash"] = canonical_sha256(value["scope"])
    value["sha256"] = canonical_sha256(value)
    if mutation is None:
        _validate_native_final_branch_proofs_v2(value, target_team="north_south")
    else:
        with pytest.raises(ValueError, match="final branch proof"):
            _validate_native_final_branch_proofs_v2(value, target_team="north_south")


@pytest.mark.parametrize("independent", (False, True))
@pytest.mark.parametrize("threshold_mode", (False, True))
@pytest.mark.parametrize("mutation", (None, "missing", "duplicate", "query_owner", "reference", "threshold", "policy", "binding", "model", "unknown", "scope_hash", "earlier_query_scope", "binding_kind", "binding_solver", "binding_schema"))
def test_retained_final_branch_proofs_reject_rehashed_substitution(independent, threshold_mode, mutation):
    from baloot_platform.research.result_authority import (
        _validate_native_final_branch_proofs_v2, validate_native_strategy_threshold_v1)
    def hashed(core):
        return {**core, "sha256": canonical_sha256(core)}
    branch_scope = {"rulesHash": "rules"}
    binding = {"schemaVersion": 1, "kind": "native_auction_policy_binding_v1",
        "solverStatus": "INFEASIBLE", "status": "bound", "rulesHash": "rules",
        "observationHash": "observation", "action": "pass"}
    if mutation == "binding_kind":
        binding["kind"] = "native_play_policy_binding_v1"
    elif mutation == "binding_solver":
        binding["solverStatus"] = "UNKNOWN"
    elif mutation == "binding_schema":
        binding["schemaVersion"] = True
    bindings = [] if independent else [hashed(binding)]
    query = hashed({"status": "no_matching_outcome", "solverStatus": "infeasible",
        "scope": branch_scope, "baseModelHash": "source" if independent else "bound",
        "targetTeam": "north_south", "threshold": 5, "auctionPolicyBindings": bindings,
        "escalationPolicyBindings": [], "projectPolicyBindings": [], "playPolicyBindings": []})
    core = {"status": "no_adverse_round_in_branch", "sourceModelHash": "source",
        "boundModelHash": query["baseModelHash"], "scope": branch_scope, "rulesHash": "rules",
        "targetTeam": "north_south", "threshold": 5, "bindings": bindings,
        "policyHash": canonical_sha256({"observation": "pass"})}
    if mutation == "threshold":
        core["threshold"] = 3
    elif mutation == "policy":
        core["policyHash"] = "stale"
    elif mutation == "binding":
        core["bindings"] = [{"status": "bound"}]
    elif mutation == "model":
        query["baseModelHash"] = "another-model"
    elif mutation == "unknown":
        query["solverStatus"] = "unknown"
    query = hashed({k: v for k, v in query.items() if k != "sha256"})
    identity = canonical_sha256({**core, "queries": [query]})
    strategy = {"schemaVersion": 1, "kind": "native_symbolic_round_policy_bounds_v2", "strategicCoverage": False,
        "scope": {"rulesHash": "rules", "targetTeam": "north_south",
        "branches": [{"modelHash": "source", "scope": branch_scope}]}, "upperBound": 4,
        "policy": {"observation": "pass"}, "queries": [{"branchModelHash": "source", "receipt": query}],
        "refinements": {identity: {"core": core, "queryHashes": [canonical_sha256(query)]}},
        "finalBranchProofs": [{"branchModelHash": "source", "refinementHash": identity, "policyIndependent": independent}]}
    if threshold_mode:
        strategy["scope"].update(requiredThreshold=4, utilityBounds=[-10, 10],
            objective="strategy_threshold_satisfaction", coverage="supplied_deal_branches_only")
        strategy.update(kind="native_symbolic_round_strategy_threshold_v1",
            status="scoped_strategy_threshold_complete", requiredThreshold=4, lowerBound=4,
            upperBound=10, masterSearchValue=4, masters=[{"status": "optimal", "searchValue": 4}],
            domains={"observation": ["pass"]})
    expected_scope = copy.deepcopy(strategy["scope"])
    if mutation == "missing":
        strategy["finalBranchProofs"] = []
    elif mutation == "duplicate":
        strategy["finalBranchProofs"] *= 2
    elif mutation == "query_owner":
        strategy["queries"][0]["branchModelHash"] = "another-model"
    elif mutation == "reference":
        strategy["refinements"][identity]["queryHashes"] = ["missing"]
    elif mutation == "earlier_query_scope":
        earlier = hashed({**{k: v for k, v in query.items() if k != "sha256"}, "scope": {}})
        strategy["queries"].insert(0, {"branchModelHash": "source", "receipt": earlier})
        identity = canonical_sha256({**core, "queries": [earlier, query]})
        strategy["refinements"] = {identity: {"core": core, "queryHashes": [canonical_sha256(earlier), canonical_sha256(query)]}}
        strategy["finalBranchProofs"][0]["refinementHash"] = identity
    strategy["scopeHash"] = "corrupted" if mutation == "scope_hash" else canonical_sha256(strategy["scope"])
    def validate():
        if threshold_mode:
            assert validate_native_strategy_threshold_v1(hashed(strategy), expected_scope=expected_scope) == 4
        else:
            _validate_native_final_branch_proofs_v2(hashed(strategy), target_team="north_south")
    if mutation is None or mutation in {"policy", "binding_kind", "binding_solver", "binding_schema"} and independent:
        validate()
    else:
        with pytest.raises(ValueError, match="final branch proof"):
            validate()


@pytest.mark.parametrize("mutation", (None, "owner", "selected", "alternative", "quality",
    "preparation", "query", "selected_preparation", "selected_strategy", "schema_bool"))
def test_bid_dominance_retained_identity_bindings(mutation):
    from baloot_platform.research.result_authority import validate_native_bid_dominance_identity_v1
    def hashed(core):
        return {**core, "sha256": canonical_sha256(core)}
    owner = dict(rules_hash="rules", source_prefix_hash="prefix", exact_state_id="state-id",
        state_hash="state", target_team="north_south", selected_action="sun")
    selected = hashed({"testReceipt": "selected", "preparation": hashed({"testReceipt": "preparation"}),
        "strategy": hashed({"testReceipt": "strategy"})})
    alternative = {"action": "pass", "preparation": hashed({"testReceipt": "preparation"}),
        "queries": [hashed({"testReceipt": "query"})]}
    qualification = hashed({"selectedProofHash": canonical_sha256(selected), "modelProto": "test-model",
        "alternatives": [{"action": "pass", "sourceProofHash": canonical_sha256(alternative)}]})
    quality = {"qualificationModelHash": qualification["sha256"]}
    quality["sha256"] = canonical_sha256({**quality, "modelProto": "test-model"})
    core = {"kind": "native_bid_selected_action_dominance_v1", "schemaVersion": 1,
        "rulesHash": "rules", "sourcePrefixHash": "prefix", "exactStateId": "state-id", "stateHash": "state",
        "targetTeam": "north_south", "selectedAction": "sun", "visibility": "complete_state",
        "utility": "awarded_point_margin", "completionScope": "selected_bid_current_deal",
        "selectedValueProof": selected, "qualificationModel": qualification,
        "alternatives": [alternative], "actionQuality": quality}
    if mutation == "owner":
        owner["state_hash"] = "another-state"
    elif mutation == "selected":
        selected["testReceipt"] = "substituted"
    elif mutation == "alternative":
        alternative["action"] = "hokum:spades"
    elif mutation == "quality":
        quality["qualificationModelHash"] = "substituted"
    elif mutation == "schema_bool":
        core["schemaVersion"] = True
    elif mutation in {"preparation", "query", "selected_preparation", "selected_strategy"}:
        if mutation.startswith("selected_"):
            selected[mutation.removeprefix("selected_")]["testReceipt"] = "substituted"
            selected["sha256"] = canonical_sha256({k: v for k, v in selected.items() if k != "sha256"})
            qualification["selectedProofHash"] = canonical_sha256(selected)
        else:
            child = alternative["preparation"] if mutation == "preparation" else alternative["queries"][0]
            child["testReceipt"] = "substituted"
            qualification["alternatives"][0]["sourceProofHash"] = canonical_sha256(alternative)
        qualification["sha256"] = canonical_sha256({k: v for k, v in qualification.items() if k != "sha256"})
        quality["qualificationModelHash"] = qualification["sha256"]
        quality["sha256"] = canonical_sha256({**{k: v for k, v in quality.items() if k != "sha256"},
            "modelProto": "test-model"})
    value = hashed(core)
    if mutation is None:
        validate_native_bid_dominance_identity_v1(value, **owner)
    else:
        with pytest.raises(ValueError, match="bid dominance"):
            validate_native_bid_dominance_identity_v1(value, **owner)

def _claim_qualification_model(value):
    from baloot_platform.research.ortools_information_set_proof import build_native_bid_dominance_model_v1
    model, selected, claim, best, regret, bindings = build_native_bid_dominance_model_v1(
        value["selectedValueProof"]["strategy"]["lowerBound" if value.get("kind") == "native_bid_selected_action_dominance_v2" else "upperBound"], value["alternatives"])
    return {"solverStatus": "optimal", "modelProto": str(model.proto), "selectedValueVariable": selected.name,
        "claimVariable": claim.name, "claimLiteralIndex": int(claim.index), "bestValueVariable": best.name,
        "regretVariable": regret.name, "alternatives": bindings}


@pytest.mark.parametrize("mutation", (None, "gap", "count", "ties", "regret", "missing_action",
    "unknown_qualification", "inverted_bounds", "missing_preparation", "wrong_strategy_model",
    "model_substitution", "wrong_variable", "null_qualification", "invalid_team_shape", "null_scope",
    "ceiling", "ceiling_domain", "ceiling_redeal", "ceiling_redeal_scope", "ceiling_redeal_observations",
    "ceiling_policy_shape", "ceiling_binding_shape", "ceiling_missing_binding",
    "ceiling_owner", "ceiling_unknown", "ceiling_action", "ceiling_value", "ceiling_witness_only",
    "native_ceiling", "native_unknown", "native_binding", "native_missing_binding",
    "native_model", "native_scope", "native_play_policies", "native_proto",
    "missing_strategy_kind", "unknown_strategy_kind", "v2_missing_refs"))
def test_bid_dominance_claim_preserves_scope_and_unknown_ties(mutation):
    from baloot_platform.research.result_authority import validate_native_bid_dominance_claim_v1
    quality = {"optimalActionCount": None, "optimalActionsComplete": False, "optimalActions": ["sun"],
        "actualAction": "sun", "actualActionOptimal": True, "actualValue": 92, "bestValue": 92, "regret": 0}
    strategy = {"kind": "native_symbolic_round_policy_bounds_v1", "status": "scoped_bound_converged", "lowerBound": 92, "upperBound": 92}
    value = {"selectedAction": "sun", "legalActions": ["sun"], "alternatives": [],
        "status": "selected_action_optimal", "optimalActionCount": None, "targetTeam": "north_south",
        "selectedValueProof": {"strategy": strategy}, "actionQuality": quality}
    value["qualificationModel"] = _claim_qualification_model(value)
    for document in (value, quality):
        document.update(optimalActionCountLowerBound=1, optimalActionCountUpperBound=1)
    value["selectedValueProof"]["preparation"] = {"constructionComplete": True,
        "branches": [{"retained": True, "sourceModelHash": "selected-model"}]}
    strategy["scope"] = {"branches": [{"modelHash": "selected-model"}]}
    if mutation == "missing_strategy_kind":
        strategy.pop("kind")
    elif mutation == "unknown_strategy_kind":
        strategy["kind"] = "unknown"
    elif mutation == "v2_missing_refs":
        strategy["kind"] = "native_symbolic_round_policy_bounds_v2"
    if mutation is not None and mutation.startswith(("ceiling", "native")):
        def hashed(document):
            return {**document, "sha256": canonical_sha256(document)}
        policy = {"auction_action": "pass", "escalation_action": "decline"}
        bindings = [hashed({"kind": "native_constant_" + kind + "_policy_binding_v1",
            "action": action, "status": "bound"}) for kind, action in (("auction", "pass"), ("escalation", "decline"))]
        outcome = {"targetTeam": "west_east", "solverStatus": "optimal", "candidate": {"utility": -92}}
        query = {"sourceModelHash": "selected-model", "status": "queried", "targetTeam": "west_east",
            "policy": {"team": "west_east", "auctionAction": "pass", "escalationAction": "decline"}}
        if mutation.startswith("ceiling_redeal"):
            value["rulesHash"] = "rules"
            scope = {"kind": "native_all_pass_deal_branch_v1", "rulesHash": "rules", "allPassedCovered": True,
                "continuationBoundary": "current_deal_completion", "nextDealAnalyzed": mutation == "ceiling_redeal_scope"}
            query["scope"] = hashed(scope)
            bindings.pop()
            outcome.update(scope=query["scope"], candidate={"utility": 0, "allPassed": True, "awardedPoints": [0, 0],
                "escalationObservations": [], "projectObservations": [],
                "playObservations": [{}] if mutation == "ceiling_redeal_observations" else []})
            strategy.update(lowerBound=0, upperBound=0)
            quality.update(actualValue=0, bestValue=0)
        if mutation.startswith("native"):
            from baloot_platform.core.enums import Team
            from baloot_platform.research.ortools_play_tail import native_observer_safe_policy_identity_v1

            value["rulesHash"] = "rules"
            policy["native_play_policy_id"] = "observer_safe_trick_control_v1"
            identity = native_observer_safe_policy_identity_v1(policy["native_play_policy_id"], Team.WEST_EAST)
            binding = {"schemaVersion": 1, "kind": "native_visible_play_policy_binding_v1",
                "policy": identity, "sourceCpModelProtoHash": "a" * 64, "boundCpModelProtoHash": "b" * 64}
            query["scope"] = {"allPassedCovered": False}
            query["rulesHash"] = "rules"
            query["policy"].update(scope="future_nonplay_constants_and_native_visible_play",
                nativePlayPolicy=identity, playPrefixLength=0,
                nativePlayPolicyBinding=hashed(binding), playPolicyModelHash="c" * 64)
            outcome.update(scope=query["scope"], baseModelHash="c" * 64)
            if mutation == "native_unknown":
                policy["native_play_policy_id"] = "unregistered"
            elif mutation == "native_binding":
                query["policy"]["nativePlayPolicyBinding"] = hashed({**binding, "policy": {}})
            elif mutation == "native_missing_binding":
                query["policy"].pop("nativePlayPolicyBinding")
            elif mutation == "native_model":
                outcome["baseModelHash"] = "d" * 64
            elif mutation == "native_scope":
                outcome["scope"] = {"allPassedCovered": True}
            elif mutation == "native_play_policies":
                query["policy"]["playPolicies"] = {}
            elif mutation == "native_proto":
                query["policy"]["nativePlayPolicyBinding"] = hashed({
                    **binding, "boundCpModelProtoHash": "not-a-hash"})
        if mutation == "ceiling_policy_shape":
            policy = []
        elif mutation == "ceiling_binding_shape":
            bindings = [None]
        elif mutation == "ceiling_missing_binding":
            bindings.pop()
        elif mutation == "ceiling_owner":
            query["sourceModelHash"] = "other"
        elif mutation == "ceiling_unknown":
            outcome["solverStatus"] = "unknown"
        elif mutation == "ceiling_witness_only":
            outcome["queryMode"] = "feasibility_witness"
        elif mutation == "ceiling_action":
            query["policy"]["escalationAction"] = "double"
        elif mutation == "ceiling_value":
            outcome["candidate"]["utility"] = -91
        strategy["scope"].update(opponentCeilingPolicy=policy, utilityBounds=[-200, 92])
        if mutation == "ceiling_domain":
            strategy["scope"]["utilityBounds"] = [-200, 200]
        strategy["queries"] = [{"branchModelHash": "selected-model", "purpose": "opponent_policy_upper_bound",
            "receipt": hashed({**query, "bindings": bindings, "query": hashed(outcome)})}]
        if mutation.startswith("ceiling_redeal"):
            value["qualificationModel"] = _claim_qualification_model(value)
    if mutation == "null_qualification":
        value["qualificationModel"] = None
    elif mutation == "invalid_team_shape":
        value["targetTeam"] = []
    elif mutation == "null_scope":
        strategy["scope"] = None
    if mutation == "gap":
        strategy["lowerBound"] = 91
    elif mutation == "count":
        value["optimalActionCount"] = 1
    elif mutation == "ties":
        quality["optimalActionsComplete"] = True
    elif mutation == "regret":
        quality["regret"] = 1
    elif mutation == "unknown_qualification":
        value["qualificationModel"]["solverStatus"] = "unknown"
    elif mutation == "inverted_bounds":
        quality["optimalActionCountUpperBound"] = 0
    elif mutation == "missing_preparation":
        value["selectedValueProof"]["preparation"]["branches"] = []
    elif mutation == "wrong_strategy_model":
        strategy["scope"]["branches"][0]["modelHash"] = "substituted"
    elif mutation == "model_substitution":
        value["qualificationModel"]["modelProto"] = ""
    elif mutation == "wrong_variable":
        value["qualificationModel"]["claimLiteralIndex"] = 0
    actions = ("pass", "sun") if mutation == "missing_action" else ("sun",)
    if mutation in (None, "ceiling", "ceiling_domain", "ceiling_redeal", "native_ceiling"):
        validate_native_bid_dominance_claim_v1(value, legal_actions=actions)
    else:
        with pytest.raises(ValueError, match="bid dominance"):
            validate_native_bid_dominance_claim_v1(value, legal_actions=actions)


@pytest.mark.parametrize("mutation", (None, "unknown", "wrong_team", "float_threshold", "missing_model",
    "unbound", "missing_outcome", "invalid_team", "float_value"))
def test_bid_dominance_alternative_bound_consistency(mutation):
    from baloot_platform.research.result_authority import validate_native_bid_dominance_claim_v1
    outcome = {"status": "no_matching_outcome", "solverStatus": "infeasible",
        "targetTeam": "west_east", "threshold": -92}
    query = {"status": "queried", "targetTeam": "west_east", "threshold": -92,
        "sourceModelHash": "model", "bindings": [{"status": "bound"}], "query": outcome}
    row = {"action": "pass", "upperBound": 92, "sourceModelHashes": ["model"], "queries": [query]}
    quality = {"optimalActionCount": None, "optimalActionsComplete": False, "optimalActions": ["sun"],
        "actualAction": "sun", "actualActionOptimal": True, "actualValue": 92, "bestValue": 92, "regret": 0}
    value = {"selectedAction": "sun", "legalActions": ["pass", "sun"], "alternatives": [row],
        "status": "selected_action_optimal", "optimalActionCount": None, "targetTeam": "north_south",
        "selectedValueProof": {"strategy": {"kind": "native_symbolic_round_policy_bounds_v1", "status": "scoped_bound_converged", "lowerBound": 92, "upperBound": 92}},
        "actionQuality": quality}
    for document in (value, quality):
        document.update(optimalActionCountLowerBound=1, optimalActionCountUpperBound=2)
    preparation = {"constructionComplete": True, "branches": [{"retained": True, "sourceModelHash": "model"}]}
    value["selectedValueProof"]["preparation"] = copy.deepcopy(preparation)
    value["selectedValueProof"]["strategy"]["scope"] = {"branches": [{"modelHash": "model"}]}
    row["preparation"] = preparation
    value["qualificationModel"] = _claim_qualification_model(value)
    if mutation == "unknown":
        outcome["solverStatus"] = "unknown"
    elif mutation == "wrong_team":
        outcome["targetTeam"] = "north_south"
    elif mutation == "float_threshold":
        outcome["threshold"] = -92.0
    elif mutation == "missing_model":
        row["sourceModelHashes"] = []
    elif mutation == "unbound":
        query["bindings"][0]["status"] = "incomplete"
    elif mutation == "missing_outcome":
        query["query"] = None
    elif mutation == "invalid_team":
        value["targetTeam"] = "unknown"
    elif mutation == "float_value":
        quality["actualValue"] = 92.0
    if mutation is None:
        validate_native_bid_dominance_claim_v1(value, legal_actions=("pass", "sun"))
    else:
        with pytest.raises(ValueError, match="bid dominance"):
            validate_native_bid_dominance_claim_v1(value, legal_actions=("pass", "sun"))


@pytest.mark.parametrize("mutation", (None, "state", "partition", "missing", "unknown", "policy",
    "depth_complete", "depth_missing", "depth_unknown", "depth_model", "order_complete",
    "order_missing", "order_reversed", "order_invalid", "alternative_complete", "alternative_team",
    "alternative_action", "alternative_hidden", "alternative_scope", "alternative_binding",
    "compact_complete", "compact_integer", "compact_false"))
def test_bid_dominance_preparation_reader_uses_frozen_rules(rules, mutation):
    from baloot_platform.research.result_authority import validate_native_bid_dominance_preparations_v1
    from baloot_platform.research.ortools_information_set_proof import native_bid_branch_specs_v1
    from baloot_platform.core.enums import Seat
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)), "dealer": "east",
        "auctionActions": [], "escalationActions": [], "projectDeclarations": [], "playActions": []})
    partition = [None if row is None else {"contract": row[0].value,
        "trump": row[1].name.lower() if row[1] is not None else None,
        "firstLeader": row[2].name.lower(), "locked": row[3]}
        for row in native_bid_branch_specs_v1(rules, Seat.EAST)]
    scope = {"rulesHash": rules.sha256, "rootHash": "state", "visibility": "complete_state",
        "matchScore": list(prefix.match_score), "continuationBoundary": "current_deal_completion",
        "nextDealAnalyzed": False, "partition": partition, "selectedRootAction": "sun"}
    receipt = {"constructionComplete": True, "scope": scope,
        "branches": [{"partitionIndex": i, "retained": False,
            "reachability": {"solverStatus": "infeasible"}} for i in range(len(partition))]}
    value = {"targetTeam": "north_south", "selectedAction": "sun", "selectedValueProof": {"preparation": receipt},
        "opponentStrategy": {"auction_action": "pass", "escalation_action": "decline"}, "alternatives": []}
    if mutation is not None and mutation.startswith("alternative_"):
        alternate = copy.deepcopy(receipt)
        alternate["scope"].update(selectedRootAction="pass", nonplayPolicy={"team": "west_east",
            "auction_action": "pass", "escalation_action": "decline"})
        alternate["scopeHash"] = canonical_sha256(alternate["scope"])
        alternate["branches"][0].update(retained=True, sourceModelHash="model")
        branch_scope = {"rulesHash": rules.sha256, "matchScore": list(prefix.match_score),
            "allPassedCovered": True, "strategicCoverage": False, "kind": "native_all_pass_deal_branch_v1",
            "continuationBoundary": "current_deal_completion", "nextDealAnalyzed": False}
        if mutation == "alternative_scope":
            branch_scope["rulesHash"] = "other"
        branch_scope["sha256"] = canonical_sha256(branch_scope)
        query = {"sourceModelHash": "model", "targetTeam": "west_east", "rulesHash": rules.sha256,
            "scope": branch_scope, "query": {"scope": branch_scope},
            "policy": {"team": "west_east", "auctionAction": "pass", "escalationAction": "decline",
                "auctionPrefixLength": 0, "scope": "all_future_nonplay_information_sets"},
            "bindings": [{"kind": "native_constant_auction_policy_binding_v1", "action": "pass",
                "rulesHash": rules.sha256, "status": "bound"}]}
        if mutation == "alternative_team":
            query["policy"]["team"] = "north_south"
        elif mutation == "alternative_action":
            query["policy"]["auctionAction"] = "sun"
        elif mutation == "alternative_hidden":
            query["policy"]["playPolicies"] = {"north": "hidden"}
        elif mutation == "alternative_binding":
            query["bindings"] = []
        value["alternatives"] = [{"action": "pass", "preparation": alternate, "queries": [query]}]
    if mutation is not None and mutation.startswith("order_"):
        value["escalationOpportunityOrder"] = ["west", "south", "east", "north"]
        if mutation != "order_missing":
            scope["escalationOpportunityOrder"] = list(value["escalationOpportunityOrder"])
        if mutation == "order_reversed":
            scope["escalationOpportunityOrder"].reverse()
        elif mutation == "order_invalid":
            value["escalationOpportunityOrder"] = ["west"] * 4
    if mutation is not None and mutation.startswith("depth_"):
        value["limits"] = receipt["limits"] = {"maximumDepth": 40}
        query = {"kind": "native_continuation_depth_query_v1", "status": "depth_bound_proven",
            "solverStatus": "unknown" if mutation == "depth_unknown" else "infeasible",
            "maximumDepth": 40, "sourceModelHash": "wrong" if mutation == "depth_model" else "model",
            "unit": "remaining_decision_actions", "includes": ["auction", "escalation", "project", "play"]}
        query["sha256"] = canonical_sha256(query)
        receipt["branches"][0].update(retained=True, sourceModelHash="model")
        if mutation != "depth_missing":
            receipt["branches"][0]["depthQuery"] = query
    if mutation in ("compact_complete", "compact_integer", "compact_false"):
        scope["compactKnownInitialOwnership"] = {"compact_complete": True, "compact_integer": 1, "compact_false": False}[mutation]
    if mutation == "state":
        scope["rootHash"] = "wrong"
    elif mutation == "partition":
        scope["partition"] = []
    elif mutation == "missing":
        receipt["branches"].pop()
    elif mutation == "unknown":
        receipt["branches"][0]["reachability"]["solverStatus"] = "unknown"
    elif mutation == "policy":
        receipt["branches"][0]["reachability"]["policyBindings"] = [{"status": "incomplete"}]
    receipt["scopeHash"] = canonical_sha256(scope)
    if mutation in (None, "depth_complete", "order_complete", "alternative_complete", "compact_complete"):
        validate_native_bid_dominance_preparations_v1(value, rules=rules, source_prefix=prefix, state_hash="state")
    else:
        with pytest.raises(ValueError, match="bid dominance"):
            validate_native_bid_dominance_preparations_v1(value, rules=rules, source_prefix=prefix, state_hash="state")


@pytest.mark.parametrize("mutation", (None, "total", "proof_total", "bytes", "retained", "cap",
    "query_cap", "owner", "source_bytes", "outcome_bytes", "missing_query", "wrong_query", "boolean"))
def test_streamed_bid_preparation_reader_rejects_rehashed_resource_substitution(mutation):
    from baloot_platform.research.result_authority import _validate_streamed_bid_preparation

    def hashed(value):
        value.pop("sha256", None)
        return {**value, "sha256": canonical_sha256(value)}

    binding = hashed({"status": "bound", "modelBytes": {"owner": 11, "legalityProbe": 13}})
    outcome = hashed({"modelBytes": {"owner": 12, "outcomeQuery": 14}})
    query = {"sourceModelHash": "model", "bindings": [binding], "query": outcome,
        "modelBytes": {"unit": "utf8_cp_model_proto", "source": 10,
            "retainedUpperBound": 36, "includesSolverWorkingMemory": False}}
    branch = {"retained": True, "sourceModelHash": "model", "modelBytes": 10,
        "reachability": {"modelBytes": 2, "policyBindings": [hashed({"status": "bound",
            "modelBytes": {"owner": 2, "legalityProbe": 3}})]},
        "depthQuery": {"modelBytes": {"owner": 10, "depthQuery": 12}}}
    receipt = {"kind": "native_bid_streamed_alternative_preparation_v1", "constructionComplete": True,
        "modelByteUnit": "utf8_cp_model_proto", "includesSolverWorkingMemory": False,
        "alternativeUpperBound": 92, "branches": [branch], "modelBytes": 10,
        "retainedModelBytes": 36, "proofQueriesUsed": 2, "queriesUsed": 5,
        "limits": {"queries": 5, "modelBytes": 36}}
    if mutation in {"total", "proof_total", "bytes", "retained", "boolean"}:
        key = {"total": "queriesUsed", "proof_total": "proofQueriesUsed", "bytes": "modelBytes",
            "retained": "retainedModelBytes", "boolean": "queriesUsed"}[mutation]
        receipt[key] = True if mutation == "boolean" else receipt[key] + 1
    elif mutation == "cap":
        receipt["limits"]["modelBytes"] = 35
    elif mutation == "query_cap":
        receipt["limits"]["queries"] = 4
    elif mutation == "owner":
        query["sourceModelHash"] = "substituted"
    elif mutation == "source_bytes":
        query["modelBytes"]["source"] = 11
    elif mutation == "outcome_bytes":
        query["query"] = hashed({"modelBytes": {"owner": 12, "outcomeQuery": 15}})
    query = hashed(query)
    branch["alternativeQuery"] = query
    alternative = {"upperBound": 92, "queries": [query]}
    if mutation == "missing_query":
        branch.pop("alternativeQuery")
    elif mutation == "wrong_query":
        alternative["queries"] = [hashed({**query, "sourceModelHash": "substituted"})]
    receipt = hashed(receipt)
    if mutation is None:
        _validate_streamed_bid_preparation(receipt, alternative, maximum_model_bytes=36)
    else:
        with pytest.raises(ValueError, match="streamed bid"):
            _validate_streamed_bid_preparation(receipt, alternative, maximum_model_bytes=36)


@pytest.mark.parametrize("failure", (None, "identity", "claim", "preparations"))
def test_bid_dominance_combined_reader_requires_every_gate(rules, monkeypatch, failure):
    from baloot_platform.research import result_authority as authority
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)), "dealer": "east",
        "auctionActions": [], "escalationActions": [], "projectDeclarations": [], "playActions": []})
    calls = []
    value = {}
    def gate(name):
        def check(document, **context):
            assert document is value
            if name == "identity":
                assert context["rules_hash"] == rules.sha256
                assert context["source_prefix_hash"] == prefix.sha256
            elif name == "claim":
                assert context["legal_actions"] == ("sun",)
            else:
                assert context["rules"] is rules and context["source_prefix"] is prefix
                assert context["state_hash"] == "state"
            calls.append(name)
            if failure == name:
                raise ValueError("rejected certificate")
        return check
    names = ("identity", "claim", "preparations")
    for name in names:
        monkeypatch.setattr(authority, f"validate_native_bid_dominance_{name}_v1", gate(name))
    def validate():
        authority.validate_native_bid_dominance_evidence_v1(value, rules=rules, source_prefix=prefix,
            exact_state_id="exact", state_hash="state", target_team="north_south",
            selected_action="sun", legal_actions=("sun",))
    if failure is None:
        validate()
        assert calls == list(names)
    else:
        with pytest.raises(ValueError, match="rejected certificate"):
            validate()
        assert calls == list(names[:names.index(failure) + 1])


@pytest.mark.parametrize("entry", ("decision", "round", "trace"))
@pytest.mark.parametrize("invalid", ("hash", "type"))
def test_quality_readers_reject_mismatched_frozen_rules(rules, entry, invalid):
    from baloot_platform.research import result_authority as authority
    context = {"rules_hash": "0" * 64 if invalid == "hash" else rules.sha256,
        "rules": rules if invalid == "hash" else {"sha256": rules.sha256}}
    if entry == "decision":
        reader = authority.validate_native_decision_quality_evidence_v1
        context.update(source_prefix=None, request=None)
    elif entry == "round":
        reader = authority.validate_native_round_quality_evidence_v1
        context.update(source_prefix=None, policy=None, decision_point={})
    else:
        reader = authority.validate_trace_quality_evidence_v1
    with pytest.raises(ValueError, match="frozen rules mismatch"):
        reader({}, **context)


@pytest.mark.real_clock
@pytest.mark.parametrize("bid_kind", ("pass", "sun", "hokum"))
@pytest.mark.parametrize("generated", (False, True, "staged"))
def test_selected_bid_quality_artifact_does_not_invent_outcomes_or_ties(root, rules, monkeypatch, bid_kind, generated):
    from baloot_platform.core.enums import Seat
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import (
        ScenarioProofBudgetV1, ScenarioProofRequestV1, ScenarioQualityExecutionBudgetV1,
    )
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)), "dealer": "east",
        "auctionActions": [], "escalationActions": [], "projectDeclarations": [], "playActions": []})
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix, decision_point={"kind": "bid"})
    selected = next(row["code"] for row in normalized.legal_actions if row["code"].split(":")[0] == bid_kind)
    raw = {"schemaVersion": 1, "mode": "exact", "visibility": "complete_state",
        "claim": "winning_root_action_count", "targetTeam": Seat.parse(normalized.next_actor).team.name.lower(),
        "utility": "awarded_point_margin", "selectedAction": selected, "budgetProfile": "test"}
    certificate = {"sha256": "certificate", "selectedAction": selected,
        "qualificationModel": {"sha256": "model", "modelProto": "native-model"},
        "actionQuality": {"optimalActions": [selected], "optimalActionsComplete": False,
            "rootActorRelation": "target",
            "actualAction": selected, "actualActionOptimal": True, "regret": 0, "sha256": "quality",
            "optimalActionCount": None, "optimalActionCountLowerBound": 1,
            "optimalActionCountUpperBound": len(normalized.legal_actions)}}
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))["scenarioResearchV4"]
    proof_raw = settings["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    proof_budget = ScenarioProofBudgetV1.build("test", proof_raw)
    execution = ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1, "maximumWallSeconds": 30,
        "maximumEvidenceBytes": 10000, "maximumEscalationDecisions": 20,
        "escalationOpportunityOrder": list(settings["qualityExecution"]["escalationOpportunityOrder"])})
    certificate.update(kind="native_bid_selected_action_dominance_v1", queriesUsed=1,
        escalationOpportunityOrder=list(execution.escalation_opportunity_order),
        limits={"maximumDepth": min(proof_budget.maximum_depth, proof_budget.maximum_horizon_actions),
            "maximumIterations": proof_budget.native_observed.maximum_iterations,
            "maximumQueries": proof_budget.native_observed.maximum_queries,
            "maximumObservations": proof_budget.native_observed.maximum_observations,
            "maximumBranchesPerPreparation": proof_budget.maximum_state_space,
            "maximumModelBytesPerPreparation": proof_budget.maximum_cache_bytes,
            "maximumConflictsPerSolve": 123, "maximumSeconds": 10, "workersPerSolve": 1})
    if generated:
        from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
        raw["selectedAction"] = None
        actions = [row["code"] for row in normalized.legal_actions]
        prior_actions = actions[:actions.index(selected)]
        if generated == "staged":
            certificate["selectionSchedule"] = "incumbent_then_deep_v1"
            prior_actions = actions + prior_actions
        certificate["priorSelectionAttempts"] = [{"action": action, "queryAllowance": 1, "queriesCharged": 1,
            "receipt": InformationSetCompilationStopped("maximum_seconds", stage="complete_state_quality").partial_receipt}
            for action in prior_actions]
        certificate["limits"]["maximumQueries"] -= len(certificate["priorSelectionAttempts"])
    calls = []
    def validate(document, **context):
        assert document == certificate
        assert context["rules"] is rules and context["source_prefix"].sha256 == prefix.sha256
        assert context["state_hash"] == normalized.state_hash
        assert context["legal_actions"] == tuple(row["code"] for row in normalized.legal_actions)
        assert context["selected_action"] == selected
        calls.append(context)
    monkeypatch.setattr(authority, "validate_native_bid_dominance_evidence_v1", validate)
    for schedule in ([], {}, "unknown"):
        with pytest.raises(ValueError, match="selection schedule"):
            authority.build_native_bid_dominance_quality_artifact_v1(
                {**certificate, "selectionSchedule": schedule}, rules=rules,
                source_prefix=prefix, normalized=normalized, request=ScenarioProofRequestV1.build(raw))
    artifact = authority.build_native_bid_dominance_quality_artifact_v1(certificate,
        rules=rules, source_prefix=prefix, normalized=normalized, request=ScenarioProofRequestV1.build(raw))
    assert len(calls) == 1
    authority.validate_native_bid_dominance_quality_artifact_v1(artifact, rules=rules,
        request=ScenarioProofRequestV1.build(raw), source_prefix_hash=prefix.sha256,
        exact_state_id=normalized.exact_state_id)
    assert len(calls) == 2
    for field, replacement in (("claimSatisfied", True), ("winningRootActionCount", 1),
            ("completionScope", "whole_round")):
        changed = copy.deepcopy(artifact)
        changed["proof"][field] = replacement
        changed["proof"]["sha256"] = canonical_sha256({k: v for k, v in changed["proof"].items() if k != "sha256"})
        with pytest.raises(ValueError, match="artifact binding mismatch"):
            authority.validate_native_bid_dominance_quality_artifact_v1(changed, rules=rules,
                request=ScenarioProofRequestV1.build(raw), source_prefix_hash=prefix.sha256,
                exact_state_id=normalized.exact_state_id)
    with pytest.raises(ValueError, match="requires frozen rules"):
        authority.validate_native_bid_dominance_quality_artifact_v1(artifact, rules=None,
            request=ScenarioProofRequestV1.build(raw), source_prefix_hash=prefix.sha256,
            exact_state_id=normalized.exact_state_id)
    assert artifact["proof"]["completionScope"] == "selected_bid_current_deal"
    assert artifact["proof"]["claimSatisfied"] is None
    assert artifact["proof"]["winningRootActionCount"] is None
    assert all(row["value"] is None and row["satisfies"] is None for row in artifact["proof"]["rootActions"])
    assert artifact["proofModel"]["actionQuality"]["optimalActionCount"] is None
    assert artifact["graph"] == certificate and artifact["graph"] is not certificate
    for change in ({"utility": "round_win"}, {"selectedAction": "not-legal"},
            {"visibility": "actor_information_set", "strategyClass": "pure_deterministic"},
            {"targetTeam": "west_east" if raw["targetTeam"] == "north_south" else "north_south"}):
        with pytest.raises(ValueError, match="request or normalized context"):
            authority.build_native_bid_dominance_quality_artifact_v1(certificate, rules=rules,
                    source_prefix=prefix, normalized=normalized, request=ScenarioProofRequestV1.build({**raw, **change}))
    assert len(calls) == 5
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    final = normalizer.append_normalized_action(prefix, normalized, selected)
    next_state = normalizer.normalize(final, decision_point={"kind": "bid"})
    def signed(document):
        return {**document, "sha256": canonical_sha256(document)}
    row = signed({"bidIndex": 0, "decisionKind": "bid", "actor": normalized.next_actor,
        "targetTeam": raw["targetTeam"], "action": selected, "sourcePrefixHash": prefix.sha256,
        "exactStateId": normalized.exact_state_id, "nextPrefixHash": final.sha256,
        "nextExactStateId": next_state.exact_state_id, "proofArtifactHash": canonical_sha256(artifact),
        "actionQualityHash": "quality", "actualActionOptimal": True, "regret": 0,
        "optimalActionCount": None, "optimalActionCountLowerBound": 1,
        "optimalActionCountUpperBound": len(normalized.legal_actions),
        "selectionReason": "native_complete_state_best_value" if generated else "recorded_action",
        "selectionTieBreak": None})
    receipt = signed({"kind": "native_quality_auction_continuation_v1", "scope": "auction_suffix_only",
        "qualificationAuthority": "cp_sat", "rulesHash": rules.sha256, "sourcePrefixHash": prefix.sha256,
        "qualityRequestHash": ScenarioProofRequestV1.build(raw).sha256,
        "operation": "generate" if generated else "audit_recorded",
        "status": "incomplete", "finalPrefix": final.to_dict(), "coveredFromBidIndex": 0,
        "coveredToBidIndex": 1, "decisions": [row], "artifacts": {canonical_sha256(artifact): artifact},
        "finalExactStateId": next_state.exact_state_id})
    with pytest.raises(ValueError, match="requires frozen search limits"):
        authority.validate_native_decision_quality_evidence_v1(receipt, source_prefix=prefix,
            rules_hash=rules.sha256, rules=rules, request=ScenarioProofRequestV1.build(raw))
    governed_row = signed({**{key: value for key, value in row.items() if key != "sha256"},
        "proofBudgetHash": proof_budget.sha256})
    governed = signed({**{key: value for key, value in receipt.items() if key != "sha256"},
        "decisions": [governed_row], "proofBudgetProfiles": {"test": proof_raw}})
    # Envelope unit only; real solver readback/forgery tests own native truth.
    readbacks = []
    def reverify(document, **context):
        assert document == artifact
        assert context["rules"] is rules
        assert context["request"].sha256 == ScenarioProofRequestV1.build(raw).sha256
        assert context["budget"].sha256 == proof_budget.sha256
        assert context["maximum_conflicts"] == 123
        assert context["execution_budget"] is execution
        assert context["opportunity_order"] == execution.escalation_opportunity_order
        assert 0 < context["maximum_seconds"] <= min(proof_budget.maximum_wall_seconds, proof_budget.maximum_cpu_seconds)
        readbacks.append(context)
    monkeypatch.setattr(authority, "reverify_native_bid_dominance_quality_v1", reverify)
    authority.validate_native_decision_quality_evidence_v1(governed, source_prefix=prefix,
        rules_hash=rules.sha256, rules=rules, request=ScenarioProofRequestV1.build(raw),
        search_budget_profiles={"test": {"maximumConflicts": 123}}, execution_budget=execution)
    assert len(readbacks) == 1
    with pytest.raises(ValueError, match="frozen proof limits"):
        authority.validate_native_decision_quality_evidence_v1(governed, source_prefix=prefix,
            rules_hash=rules.sha256, rules=rules, request=ScenarioProofRequestV1.build(raw),
            search_budget_profiles={"test": {"maximumConflicts": 122}}, execution_budget=execution)
    substituted_row = signed({**{key: value for key, value in governed_row.items() if key != "sha256"},
        "proofBudgetHash": "0" * 64})
    substituted = signed({**{key: value for key, value in governed.items() if key != "sha256"},
        "decisions": [substituted_row]})
    with pytest.raises(ValueError, match="proof budget substitution"):
        authority.validate_native_decision_quality_evidence_v1(substituted, source_prefix=prefix,
            rules_hash=rules.sha256, rules=rules, request=ScenarioProofRequestV1.build(raw),
            search_budget_profiles={"test": {"maximumConflicts": 123}}, execution_budget=execution)
    assert len(readbacks) == 1
    if bid_kind == "sun":
        from baloot_platform.research import ortools_research_executor as executor
        monkeypatch.setattr(executor, "_complete_state_quality_artifact_v1", lambda *args, **kwargs: artifact)
        produced = executor.resolve_quality_auction_continuation_v1(rules, prefix,
            ScenarioProofRequestV1.build({key: value for key, value in raw.items() if key != "selectedAction"}),
            **({"recorded_trace": final} if not generated else {}),
            maximum_nodes=1000, maximum_depth=100, maximum_worlds=1,
            maximum_seconds=30, maximum_decision_seconds=15, maximum_evidence_bytes=10_000_000,
            proof_budget_profiles={"test": proof_raw},
            search_budget_profiles={"test": {"maximumConflicts": 123}}, execution_budget=execution)
        assert produced["status"] == "complete", produced.get("partialReceipt")
        assert produced["decisions"][0]["optimalActionCount"] is None
        assert produced["decisions"][0]["optimalActionCountUpperBound"] == len(normalized.legal_actions)
    for key, replacement in (("optimalActionCount", 1), ("optimalActionCountLowerBound", 0),
            ("selectionReason", "recorded_action" if generated else "native_complete_state_best_value")):
        changed = copy.deepcopy(governed)
        changed["decisions"][0][key] = replacement
        changed["decisions"][0] = signed({k: v for k, v in changed["decisions"][0].items() if k != "sha256"})
        changed = signed({k: v for k, v in changed.items() if k != "sha256"})
        with pytest.raises(ValueError, match="selected bid quality decision binding"):
            authority.validate_native_decision_quality_evidence_v1(changed, source_prefix=prefix,
                rules_hash=rules.sha256, rules=rules, request=ScenarioProofRequestV1.build(raw),
                search_budget_profiles={"test": {"maximumConflicts": 123}}, execution_budget=execution)


@pytest.mark.parametrize("mutation", (None, "missing", "maximumDepth", "maximumQueries",
    "maximumIterations", "maximumObservations", "maximumBranchesPerPreparation",
    "maximumModelBytesPerPreparation", "maximumConflictsPerSolve", "missingConflictCap",
    "booleanConflictCap", "workersPerSolve", "maximumSeconds", "queriesUsed", "bool",
    "missingOrder", "differentOrder", "missingFrozenOrder", "interval", "interval_time",
    "interval_queries", "interval_usage", "interval_missing", "native_ceiling",
    "native_ceiling_missing", "native_ceiling_unconfigured", "native_ceiling_malformed", "native_incumbent",
    "native_ceiling_unavailable", "native_ceiling_unavailable_single", "native_ceiling_unavailable_policy",
    "native_ceiling_unavailable_unconfigured",
    "fallback", "fallback_time", "fallback_queries", "fallback_owner", "fallback_hash", "fallback_policy",
    "fallback_shared", "fallback_shared_cap", "fallback_shared_bool", "fallback_shared_parent", "fallback_native"))
def test_selected_bid_budget_admission_uses_frozen_tier(root, mutation):
    from baloot_platform.research.result_authority import validate_native_bid_dominance_budget_v1
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1, ScenarioQualityExecutionBudgetV1
    profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
    raw = profile["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    budget = ScenarioProofBudgetV1.build("offline_heavyweight_v1", raw)
    conflicts = profile["scenarioResearchV4"]["searchBudgets"]["profiles"]["offline_heavyweight_v1"]["maximumConflicts"]
    limits = {"maximumDepth": min(budget.maximum_depth, budget.maximum_horizon_actions),
        "maximumIterations": budget.native_observed.maximum_iterations,
        "maximumQueries": budget.native_observed.maximum_queries,
        "maximumObservations": budget.native_observed.maximum_observations,
        "maximumBranchesPerPreparation": budget.maximum_state_space,
        "maximumModelBytesPerPreparation": budget.maximum_cache_bytes, "workersPerSolve": 1,
        "maximumConflictsPerSolve": conflicts,
        "maximumSeconds": min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds)}
    order = tuple(profile["scenarioResearchV4"]["qualityExecution"]["escalationOpportunityOrder"])
    certificate = {"limits": limits, "queriesUsed": 1, "escalationOpportunityOrder": list(order)}
    execution = None
    if mutation is not None and (mutation.startswith(("interval", "fallback")) or mutation == "native_incumbent"):
        certificate["kind"] = "native_bid_selected_action_dominance_v2"
        configuration = _bid_incumbent_configuration()
        if mutation in {"native_incumbent", "fallback_native"}:
            configuration.pop("playPolicies")
            configuration.update(schemaVersion=2, nativePlayPolicyId="observer_safe_trick_control_v1")
            certificate["kind"] = "native_bid_selected_action_dominance_v3"
        execution = ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1, "maximumWallSeconds": limits["maximumSeconds"],
            "maximumEvidenceBytes": 10000, "maximumEscalationDecisions": 20, "bidIncumbentProof": configuration})
        certificate.update(targetTeam="north_south", **(
            {"incumbentNativePlayPolicyId": configuration["nativePlayPolicyId"]} if configuration["schemaVersion"] == 2 else
            {"incumbentPlayPolicies": {seat: configuration["playPolicies"][seat] for seat in ("north", "south")}}))
        limits.update(incumbentMaximumSeconds=configuration["maximumSeconds"],
            incumbentMaximumQueries=configuration["maximumQueries"])
        certificate["selectedValueProof"] = {"maximumSeconds": limits["incumbentMaximumSeconds"], "queriesUsed": 1}
        if mutation == "interval_time":
            limits["incumbentMaximumSeconds"] = limits["maximumSeconds"]
        elif mutation == "interval_queries":
            limits["incumbentMaximumQueries"] = limits["maximumQueries"]
        elif mutation == "interval_usage":
            certificate["selectedValueProof"]["queriesUsed"] = limits["maximumQueries"]
        elif mutation == "interval_missing":
            limits.pop("incumbentMaximumSeconds")
        if mutation.startswith("fallback"):
            attempt = {**copy.deepcopy(certificate), "status": "incomplete", "selectedValueProof": None,
                "sourcePrefixHash": "owned-prefix"}
            if mutation == "fallback_policy":
                attempt["incumbentPlayPolicies"] = {}
            attempt["sha256"] = canonical_sha256(attempt)
            certificate = {**certificate, "kind": "native_bid_selected_action_dominance_v1",
                "sourcePrefixHash": "owned-prefix", "incumbentAttempt": attempt,
                "limits": {**limits, "maximumSeconds": limits["maximumSeconds"] - configuration["maximumSeconds"],
                    "maximumQueries": limits["maximumQueries"] - attempt["queriesUsed"]}}
            if mutation == "fallback_time":
                certificate["limits"]["maximumSeconds"] += 0.1
            elif mutation == "fallback_queries":
                certificate["limits"]["maximumQueries"] += 1
            elif mutation == "fallback_owner":
                certificate["sourcePrefixHash"] = "substituted-prefix"
            elif mutation == "fallback_hash":
                attempt["sha256"] = "invalid"
            if mutation.startswith("fallback_shared"):
                certificate["limits"].update(sharedDecisionMaximumSeconds=limits["maximumSeconds"],
                    maximumSeconds=limits["maximumSeconds"] - 1)
                if mutation == "fallback_shared_cap":
                    certificate["limits"]["sharedDecisionMaximumSeconds"] -= 1
                elif mutation == "fallback_shared_bool":
                    certificate["limits"]["sharedDecisionMaximumSeconds"] = True
                elif mutation == "fallback_shared_parent":
                    certificate.pop("incumbentAttempt")
            if mutation in {"fallback", "fallback_shared", "fallback_native"}:
                validate_native_bid_dominance_budget_v1(certificate, budget, maximum_conflicts=conflicts,
                    opportunity_order=order, execution_budget=execution)
                if mutation == "fallback_native":
                    resumed = copy.deepcopy(certificate)
                    resumed["kind"] = "native_bid_selected_action_dominance_v4"
                    resumed.pop("incumbentNativePlayPolicyId")
                    resumed["limits"]["sharedDecisionMaximumSeconds"] = limits["maximumSeconds"]
                    validate_native_bid_dominance_budget_v1(resumed, budget, maximum_conflicts=conflicts,
                        opportunity_order=order, execution_budget=execution)
                    # The threshold continuation owns only the leftover allowance;
                    # it cannot reset queries or substitute its alternative policy.
                    for key in ("queries", "deadline", "policy"):
                        changed = copy.deepcopy(resumed)
                        if key == "queries":
                            changed["limits"]["maximumQueries"] += 1
                        elif key == "deadline":
                            changed["limits"]["sharedDecisionMaximumSeconds"] += 1
                        else:
                            changed["opponentPlayPolicyId"] = "observer_safe_trick_control_v1"
                        with pytest.raises(ValueError, match="fallback|configured alternative basis"):
                            validate_native_bid_dominance_budget_v1(changed, budget,
                                maximum_conflicts=conflicts, opportunity_order=order, execution_budget=execution)
            else:
                with pytest.raises(ValueError, match="fallback|frozen incumbent"):
                    validate_native_bid_dominance_budget_v1(certificate, budget, maximum_conflicts=conflicts,
                        opportunity_order=order, execution_budget=execution)
            return
    elif mutation == "missing":
        certificate.pop("limits")
    elif mutation in {"native_ceiling", "native_ceiling_missing", "native_ceiling_unconfigured",
            "native_ceiling_malformed"}:
        from baloot_platform.research.ortools_play_tail import NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS
        policy = sorted(NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS)[0]
        if mutation != "native_ceiling_unconfigured":
            configuration = {"schemaVersion": 1, "maximumWallSeconds": limits["maximumSeconds"],
                "maximumEvidenceBytes": 10000, "maximumEscalationDecisions": 20,
                "bidOpponentPlayPolicyId": policy}
            execution = ScenarioQualityExecutionBudgetV1.build(configuration)
        if mutation != "native_ceiling_missing":
            certificate["selectedValueProof"] = {"strategy": {"scope": {
                "opponentCeilingPolicy": {"auction_action": "pass", "escalation_action": "decline",
                    "native_play_policy_id": policy}}}}
        if mutation == "native_ceiling_malformed":
            certificate["selectedValueProof"] = "not a proof"
    elif type(mutation) is str and mutation.startswith("native_ceiling_unavailable"):
        policy = "observer_safe_trick_control_v1"
        if mutation != "native_ceiling_unavailable_unconfigured":
            execution = ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1,
                "maximumWallSeconds": limits["maximumSeconds"], "maximumEvidenceBytes": 10000,
                "maximumEscalationDecisions": 20, "bidOpponentPlayPolicyId": policy})
        certificate["selectedValueProof"] = {"strategy": {"scope": {
            "branches": [{}] * (1 if mutation == "native_ceiling_unavailable_single" else 2),
            "opponentCeilingUnavailable": {"reason": "multiple_contract_branches", "policy": {
                "auction_action": "pass", "escalation_action": "decline",
                "native_play_policy_id": "wrong" if mutation == "native_ceiling_unavailable_policy" else policy}}}}}
    elif mutation == "queriesUsed":
        certificate["queriesUsed"] = limits["maximumQueries"] + 1
    elif mutation == "bool":
        limits["maximumDepth"] = True
    elif mutation == "missingConflictCap":
        conflicts = None
    elif mutation == "booleanConflictCap":
        conflicts = True
    elif mutation == "missingOrder":
        certificate.pop("escalationOpportunityOrder")
    elif mutation == "differentOrder":
        certificate["escalationOpportunityOrder"].reverse()
    elif mutation == "missingFrozenOrder":
        order = None
    elif mutation is not None:
        limits[mutation] += 1
    if mutation in (None, "interval", "native_ceiling", "native_incumbent", "native_ceiling_unavailable"):
        validate_native_bid_dominance_budget_v1(certificate, budget, maximum_conflicts=conflicts, opportunity_order=order,
            execution_budget=execution)
        if mutation == "native_incumbent":
            changed = copy.deepcopy(certificate)
            changed["incumbentNativePlayPolicyId"] = "observer_safe_tactical_category_v1"
            with pytest.raises(ValueError, match="frozen incumbent policy"):
                validate_native_bid_dominance_budget_v1(changed, budget, maximum_conflicts=conflicts,
                    opportunity_order=order, execution_budget=execution)
            changed["incumbentNativePlayPolicyId"] = configuration["nativePlayPolicyId"]
            changed["incumbentPlayPolicies"] = {}
            with pytest.raises(ValueError, match="frozen incumbent policy"):
                validate_native_bid_dominance_budget_v1(changed, budget, maximum_conflicts=conflicts,
                    opportunity_order=order, execution_budget=execution)
    else:
        with pytest.raises(ValueError, match="frozen|unconfigured"):
            validate_native_bid_dominance_budget_v1(certificate, budget, maximum_conflicts=conflicts, opportunity_order=order,
                execution_budget=execution)


@pytest.mark.real_clock
@pytest.mark.parametrize("complete", (True, False))
@pytest.mark.parametrize("interval", (True, False, "native"))
def test_complete_state_bid_quality_dispatch_uses_frozen_limits(root, monkeypatch, complete, interval):
    from types import SimpleNamespace
    from baloot_platform.research import ortools_research_executor as executor
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1
    profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
    settings = profile["scenarioResearchV4"]
    budget = ScenarioProofBudgetV1.build("offline_heavyweight_v1", settings["proofBudgets"]["profiles"]["offline_heavyweight_v1"])
    conflicts = settings["searchBudgets"]["profiles"]["offline_heavyweight_v1"]["maximumConflicts"]
    order = tuple(settings["qualityExecution"]["escalationOpportunityOrder"])
    captured, progress = {}, []
    certificate = {"status": "selected_action_optimal" if complete else "incomplete"}
    def prove(self, *args, **kwargs):
        captured.update(kwargs)
        kwargs["progress_callback"]({"kind": "quality_solver_progress_v1", "phase": "master",
            "masterCount": 1, "queryCount": 0, "bindingCount": 0})
        return certificate
    monkeypatch.setattr(executor.OrToolsActorInformationSetCompilerV1, "_prove_native_bid_selected_action", prove)
    monkeypatch.setattr(authority, "validate_native_bid_dominance_budget_v1", lambda *a, **kw: None)
    monkeypatch.setattr(authority, "build_native_bid_dominance_quality_artifact_v1", lambda *a, **kw: {"testArtifact": True})
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    arguments = dict(maximum_nodes=100, maximum_depth=64, maximum_seconds=10,
        cancel_check=lambda: False, opportunity_order=order, budget=budget, maximum_conflicts=conflicts,
        progress_callback=progress.append)
    if interval:
        from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1
        configuration = _bid_incumbent_configuration()
        if interval == "native":
            configuration = dict(schemaVersion=2, maximumSeconds=2.0, maximumQueries=10,
                nativePlayPolicyId="observer_safe_trick_control_v1")
        arguments["execution_budget"] = ScenarioQualityExecutionBudgetV1.build({**settings["qualityExecution"],
            "bidIncumbentProof": configuration})
    normalized = SimpleNamespace(decision_kind="bid", next_actor="north", legal_actions=({"code": "sun"},))
    request = SimpleNamespace(mode="exact", visibility="complete_state", utility="awarded_point_margin",
        selected_action="sun", line_universe=None)
    if complete:
        assert executor._complete_state_quality_artifact_v1(rules, normalized, None, None, request, **arguments) == {"testArtifact": True}
    else:
        with pytest.raises(executor.InformationSetCompilationStopped, match="native_bid_quality_incomplete"):
            executor._complete_state_quality_artifact_v1(rules, normalized, None, None, request, **arguments)
    assert progress == [{"kind": "quality_solver_progress_v1", "phase": "master",
        "masterCount": 1, "queryCount": 0, "bindingCount": 0,
        "decisionKind": "bid", "proofPhase": "selected"}]
    assert captured["maximum_conflicts"] == conflicts
    assert captured["opportunity_order"] == order
    assert captured["maximum_depth"] == 64
    assert captured["maximum_queries"] == budget.native_observed.maximum_queries
    assert 0 < captured["maximum_seconds"] <= 10
    if interval:
        assert captured["incumbent_maximum_seconds"] == 2
        assert captured["incumbent_maximum_queries"] == 10
        if interval == "native":
            assert captured["incumbent_native_play_policy_id"] == configuration["nativePlayPolicyId"]
            assert captured["_owned_incumbent_models"] == []
            assert "incumbent_play_policies" not in captured
        else:
            assert set(captured["incumbent_play_policies"]) == {"north", "south"}
        if complete:
            ticks = iter((0.0, 7.9, 8.1))
            with monkeypatch.context() as patch:
                patch.setattr(executor, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
                executor._complete_state_quality_artifact_v1(rules, normalized, None, None, request, **arguments)
            assert captured["maximum_seconds"] > captured["incumbent_maximum_seconds"]
            assert next(ticks) == 8.1
        # The compiler clips incumbent time to the outer decision deadline;
        # its configured sub-budget alone is not a preflight refusal.
        if complete:
            executor._complete_state_quality_artifact_v1(rules, normalized, None, None, request,
                **{**arguments, "maximum_seconds": 1})
        else:
            with pytest.raises(executor.InformationSetCompilationStopped, match="native_bid_quality_incomplete"):
                executor._complete_state_quality_artifact_v1(rules, normalized, None, None, request,
                    **{**arguments, "maximum_seconds": 1})
        assert 0 < captured["maximum_seconds"] <= 1
        oversized = arguments["execution_budget"].to_dict()
        oversized.pop("sha256")
        oversized["bidIncumbentProof"]["maximumQueries"] = budget.native_observed.maximum_queries
        with pytest.raises(executor.InformationSetCompilationStopped, match="incumbent_allowance"):
            executor._complete_state_quality_artifact_v1(rules, normalized, None, None, request,
                **{**arguments, "execution_budget": ScenarioQualityExecutionBudgetV1.build(oversized)})
    else:
        assert "incumbent_play_policies" not in captured


@pytest.mark.real_clock
@pytest.mark.parametrize("outcome", ("complete", "complete_search", "incomplete", "cancelled", "query_exhausted", "decision_exhausted",
                                     "refuted_search", "refuted_exact", "unknown_search",
                                     "cap_refuted_search", "cap_refuted_exact", "string_cause_search"))
@pytest.mark.parametrize("native_incumbent", (False, True))
def test_complete_state_bid_quality_fallback_shares_budget(root, monkeypatch, outcome, native_incumbent):
    from types import SimpleNamespace
    from baloot_platform.research import ortools_research_executor as executor
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1, ScenarioQualityExecutionBudgetV1
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))["scenarioResearchV4"]
    budget = ScenarioProofBudgetV1.build("offline_heavyweight_v1", settings["proofBudgets"]["profiles"]["offline_heavyweight_v1"])
    configuration = (_bid_incumbent_configuration() if not native_incumbent else {
        "schemaVersion": 2, "maximumSeconds": 2.0, "maximumQueries": 10,
        "nativePlayPolicyId": "observer_safe_trick_control_v1"})
    execution = ScenarioQualityExecutionBudgetV1.build({**settings["qualityExecution"],
        "bidIncumbentProof": configuration})
    calls, progress, admitted, artifacts = [], [], [], []
    attempt = {"status": "incomplete", "queriesUsed": budget.native_observed.maximum_queries if outcome == "query_exhausted" else 3}
    if native_incumbent:
        # A single legal bid has no alternative threshold to resume.
        attempt.update(kind="native_bid_selected_action_dominance_v3", alternatives=[])
    if outcome.startswith("refuted_"):
        attempt["cause"] = {"exhaustionReason": "incumbent_threshold_refuted"}
    elif outcome.startswith("cap_refuted_"):
        attempt["cause"] = {"exhaustionReason": "alternative_upper_bound_refuted"}
    elif outcome == "string_cause_search":
        attempt["cause"] = "selected_value_not_complete"
    def prove(self, *args, **kwargs):
        calls.append(kwargs)
        kwargs["progress_callback"]({"phase": "master"})
        return attempt if len(calls) == 1 else {"status": "selected_action_optimal" if outcome in {"complete", "complete_search"} else "incomplete", "queriesUsed": 2}
    monkeypatch.setattr(executor.OrToolsActorInformationSetCompilerV1, "_prove_native_bid_selected_action", prove)
    monkeypatch.setattr(authority, "validate_native_bid_dominance_budget_v1", lambda *a, **kw: admitted.append((a, kw)))
    monkeypatch.setattr(authority, "build_native_bid_dominance_quality_artifact_v1", lambda cert, **kw: artifacts.append(cert) or cert)
    ticks = iter((0.0, 0.1, budget.maximum_wall_seconds + 0.1 if outcome == "decision_exhausted" else 0.5))
    monkeypatch.setattr(executor, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    arguments = dict(maximum_nodes=100, maximum_depth=64,
        maximum_seconds=budget.maximum_wall_seconds + 10 if outcome == "decision_exhausted" else 10,
        cancel_check=lambda: outcome == "cancelled" and bool(calls),
        opportunity_order=tuple(settings["qualityExecution"]["escalationOpportunityOrder"]),
        budget=budget, maximum_conflicts=123, execution_budget=execution, progress_callback=progress.append,
        _search_generated=outcome in {"complete_search", "refuted_search", "cap_refuted_search", "unknown_search", "string_cause_search"})
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    normalized = SimpleNamespace(decision_kind="bid", next_actor="north", legal_actions=({"code": "pass"},))
    request = SimpleNamespace(mode="exact", visibility="complete_state", utility="awarded_point_margin",
        selected_action="pass", line_universe=None)
    if outcome in {"complete", "complete_search"}:
        result = executor._complete_state_quality_artifact_v1(rules, normalized, None, None, request, **arguments)
        assert result["incumbentAttempt"] is attempt and artifacts == [result]
    else:
        with pytest.raises(executor.InformationSetCompilationStopped) as stopped:
            executor._complete_state_quality_artifact_v1(rules, normalized, None, None, request, **arguments)
        retained = stopped.value.partial_receipt["evidence"]["nativeBidDominance"]
        assert retained == attempt if len(calls) == 1 else retained["incumbentAttempt"] == attempt
        assert artifacts == []
        if outcome in {"refuted_search", "cap_refuted_search", "unknown_search", "string_cause_search"}:
            assert stopped.value.reason == "native_bid_quality_incomplete"
            assert "actualActionOptimal" not in retained
    assert len(calls) == (1 if outcome in {"cancelled", "query_exhausted", "decision_exhausted"} else 2)
    assert calls[0].get("alternative_first", False) is native_incumbent
    checkpoints = [row for row in progress if row.get("kind") == "quality_bid_incumbent_checkpoint_v1"]
    assert len(checkpoints) == (0 if outcome == "query_exhausted" else 1)
    if checkpoints:
        assert checkpoints[0]["receipt"] is attempt
        assert checkpoints[0]["action"] == "pass" and checkpoints[0]["status"] == "incomplete"
    if len(calls) == 2:
        assert calls[1]["maximum_queries"] == calls[0]["maximum_queries"] - 3
        assert calls[1]["maximum_seconds"] == pytest.approx(9.5)
        assert calls[1]["maximum_seconds"] > calls[0]["maximum_seconds"] - 2
        assert (result if outcome in {"complete", "complete_search"} else retained)["limits"]["sharedDecisionMaximumSeconds"] == calls[0]["maximum_seconds"]
        assert not any(key.startswith("incumbent_") for key in calls[1])
        assert "alternative_first" not in calls[1] and "_owned_incumbent_models" not in calls[1]
        assert "_alternative_basis" not in calls[1]
        assert calls[1]["selected_root_action"] == calls[0]["selected_root_action"]
        assert progress[-1]["proofPhase"] == "fallback"
        assert admitted[0][1]["_incomplete_incumbent"] is True


@pytest.mark.real_clock
def test_complete_state_bid_quality_forwards_frozen_opponent_play_policy(root, monkeypatch):
    from types import SimpleNamespace
    from baloot_platform.research import ortools_research_executor as executor
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.ortools_play_tail import NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1, ScenarioQualityExecutionBudgetV1
    profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
    settings = profile["scenarioResearchV4"]
    budget = ScenarioProofBudgetV1.build("offline_heavyweight_v1", settings["proofBudgets"]["profiles"]["offline_heavyweight_v1"])
    policy = sorted(NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS)[0]
    execution = ScenarioQualityExecutionBudgetV1.build({**{key: value for key, value in settings["qualityExecution"].items()
        if key != "bidIncumbentProof"},
        "bidOpponentPlayPolicyId": policy})
    captured = {}
    def prove(self, *args, **kwargs):
        captured.update(kwargs)
        return {"status": "selected_action_optimal"}
    monkeypatch.setattr(executor.OrToolsActorInformationSetCompilerV1, "_prove_native_bid_selected_action", prove)
    monkeypatch.setattr(authority, "validate_native_bid_dominance_budget_v1", lambda *a, **kw: None)
    monkeypatch.setattr(authority, "build_native_bid_dominance_quality_artifact_v1", lambda *a, **kw: {})
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    request = SimpleNamespace(mode="exact", visibility="complete_state", utility="awarded_point_margin",
        selected_action="sun", line_universe=None)
    normalized = SimpleNamespace(decision_kind="bid", next_actor="north", legal_actions=({"code": "sun"},))
    executor._complete_state_quality_artifact_v1(rules, normalized,
        None, None, request, maximum_nodes=100, maximum_depth=64, maximum_seconds=10,
        cancel_check=lambda: False, opportunity_order=tuple(settings["qualityExecution"]["escalationOpportunityOrder"]),
        budget=budget, maximum_conflicts=100, execution_budget=execution)
    assert captured["opponent_play_policy_id"] == policy
    paired = ScenarioQualityExecutionBudgetV1.build({**settings["qualityExecution"],
        "bidOpponentPlayPolicyId": policy})
    executor._complete_state_quality_artifact_v1(rules, normalized,
        None, None, request, maximum_nodes=100, maximum_depth=64, maximum_seconds=30,
        cancel_check=lambda: False, opportunity_order=tuple(settings["qualityExecution"]["escalationOpportunityOrder"]),
        budget=budget, maximum_conflicts=100, execution_budget=paired)
    assert captured["opponent_play_policy_id"] == policy
    assert captured["alternative_first"] is True
    reordered = paired.to_dict()
    reordered.pop("sha256")
    reordered["bidIncumbentProof"]["completeStateAlternativeFirst"] = False
    executor._complete_state_quality_artifact_v1(rules, normalized,
        None, None, request, maximum_nodes=100, maximum_depth=64, maximum_seconds=30,
        cancel_check=lambda: False, opportunity_order=tuple(settings["qualityExecution"]["escalationOpportunityOrder"]),
        budget=budget, maximum_conflicts=100, execution_budget=ScenarioQualityExecutionBudgetV1.build(reordered))
    assert captured["alternative_first"] is False
    assert captured["incumbent_maximum_seconds"] == paired.bid_incumbent_proof["maximumSeconds"]
    nonplay = {"schemaVersion": 1, "auctionAction": "pass", "escalationAction": "decline",
        "projectPolicyId": "full_hand_project_max_value_v1"}
    all_stage = ScenarioQualityExecutionBudgetV1.build({**settings["qualityExecution"],
        "bidIncumbentProof": {**settings["qualityExecution"]["bidIncumbentProof"],
            "completeStateNonplayPolicy": nonplay}})
    executor._complete_state_quality_artifact_v1(rules, normalized,
        None, None, request, maximum_nodes=100, maximum_depth=64, maximum_seconds=30,
        cancel_check=lambda: False, opportunity_order=tuple(settings["qualityExecution"]["escalationOpportunityOrder"]),
        budget=budget, maximum_conflicts=100, execution_budget=all_stage)
    assert captured["incumbent_nonplay_policy"] == nonplay

    # A 60-second shared selection allowance must not become a 20-second
    # subdecision that rejects its frozen 20-second incumbent before solving.
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    generated_request = SimpleNamespace(mode="exact", visibility="complete_state",
        utility="awarded_point_margin", selected_action=None, line_universe=None)
    generated_state = SimpleNamespace(decision_kind="bid", next_actor="north",
        legal_actions=tuple({"code": code} for code in ("pass", "sun", "hokum:clubs")))
    for stop in ("selected_first", "selected_second", "deadline", "cancelled", "fair_deadline"):
        clock, cancelled, calls, progress = [100.0], [False], [], []
        compiler_owners = []
        def selected_bid(self, *args, **kwargs):
            compiler_owners.append(self)
            calls.append(kwargs)
            if len(calls) == 1 and stop != "selected_first":
                clock[0] += (kwargs["maximum_seconds"] if stop == "fair_deadline"
                    else 60 if stop == "deadline" else 8)
                cancelled[0] = stop == "cancelled"
                raise InformationSetCompilationStopped("maximum_seconds" if stop == "deadline" else "attempt_incomplete",
                    stage="complete_state_quality")
            return {"status": "selected_action_optimal", "selectedAction": kwargs["selected_root_action"]}
        with monkeypatch.context() as selection:
            selection.setattr(executor.time, "monotonic", lambda: clock[0])
            selection.setattr(executor.OrToolsActorInformationSetCompilerV1, "_prove_native_bid_selected_action", selected_bid)
            def run():
                return executor._select_observed_bid_quality_artifact_v1(rules,
                    generated_state, None, generated_request, budget=budget, execution_budget=paired,
                    maximum_conflicts=100, maximum_nodes=100, maximum_depth=64,
                    maximum_seconds=600 if stop == "fair_deadline" else 60,
                    opportunity_order=paired.escalation_opportunity_order, cancel_check=lambda: cancelled[0],
                    progress_callback=progress.append, _search_generated=True)
            if stop in {"deadline", "cancelled", "fair_deadline"}:
                with pytest.raises(InformationSetCompilationStopped) as ended:
                    run()
                assert ended.value.reason == ("cancelled" if stop == "cancelled" else "native_observed_bid_selection_incomplete")
            else:
                assert run() == {}  # Envelope spy only; not proof or admission.
        assert calls[0]["selected_root_action"] == "pass"
        assert calls[0]["maximum_seconds"] == (600 if stop == "fair_deadline" else 60)
        first_quota = budget.native_observed.maximum_queries // 6
        assert calls[0]["maximum_queries"] == first_quota
        assert len(calls) == (2 if stop == "selected_second" else 1)
        assert all(owner is compiler_owners[0] for owner in compiler_owners)
        if stop == "selected_second":
            assert calls[1]["selected_root_action"] == "sun"
            assert calls[1]["maximum_seconds"] == 52
            assert calls[1]["maximum_queries"] == (budget.native_observed.maximum_queries - first_quota) // 5
        if stop != "selected_first":
            assert progress[0]["attempt"]["queriesCharged"] == first_quota


def _ortools_section() -> dict:
    return {
        "schemaVersion": 1,
        "backend": ORTOOLS_SCENARIO_RESEARCH_BACKEND,
        "planner": "tail_first_v1",
        "deterministicExecution": True,
        "policyCatalog": {
            "schemaVersion": 1,
            "defaultPolicyId": "fixed_action_map_v1",
            "profiles": {
                "fixed_action_map_v1": {
                    "version": "1.0.0",
                    "materialization": "seeded_legal_action_order_v1",
                }
            },
        },
        "proofBudgets": {"schemaVersion": 1},
        "searchBudgets": {"schemaVersion": 1},
        "diversityDefaults": {"schemaVersion": 1},
        "cacheLimits": {"schemaVersion": 1},
        "refusalThresholds": {"schemaVersion": 1},
    }


def _authored_profile() -> dict:
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load(
        (root / "configs/challenge/progressive_mining_ortools.yaml").read_text(
            encoding="utf-8"
        )
    )


def _authored_v3_profile() -> dict:
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load(
        (root / "configs/challenge/progressive_mining_ortools_v3.yaml").read_text(
            encoding="utf-8"
        )
    )


def _bundle_identity() -> dict:
    return {
        "bundleRevisionId": "CBR-TEST-ORTOOLS-V2",
        "semanticBundleHash": "a" * 64,
        "effectiveRuntimeConfigHash": "b" * 64,
    }


def _recipe(*, policy: str = ORTOOLS_SCENARIO_GENERATION_POLICY) -> dict:
    return {
        "schemaVersion": 4,
        "scenario": {
            "origin": {"kind": "search"},
            "constraints": {
                "predicateId": "PRED-DEAL-OWNER",
                "kind": "deal.card_owner",
                "version": 1,
                "hardness": "hard",
                "arguments": {
                    "card": 0,
                    "seat": "north",
                    "handStage": "initial_five",
                },
            },
            "outcome": {},
        },
        "proof": {
            "mode": "exact",
            "visibility": "complete_state",
            "budgetProfile": "late_game_exact_v1",
        },
        "resources": {
            "generationLane": "ortools",
            "generationPolicy": policy,
        },
    }


def _tactic_recipe(tactic: str, *, decision_kind: str, utility: str) -> dict:
    recipe = _recipe(policy=ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3)
    recipe["scenario"]["constraints"] = {
        "predicateId": "PRED-TACTIC",
        "kind": "tactic.applicable",
        "version": 1,
        "hardness": "hard",
        "arguments": {"kind": tactic, "applicable": True},
    }
    selectors = {
        "bid": {"actionIndex": 0},
        "play": {
            "trickIndex": 5,
            "position": 3,
            "remainingCards": 3,
            "actingSeat": "west",
        },
        "project": {"owner": "south", "projectIndex": 0},
    }
    recipe["scenario"]["decisionPoint"] = {
        "kind": decision_kind,
        "selector": selectors[decision_kind],
    }
    recipe["proof"]["utility"] = utility
    return recipe


def _policy_templates() -> dict:
    result = {}
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
        result[seat] = {key: value for key, value in core.items() if key != "seat"}
        result[seat]["profileHash"] = canonical_sha256(core)
    return result


def _quality_policy(mode="budgeted", budget="late_game_exact_v1"):
    return {"schemaVersion": 1, "mode": mode, "decisions": {
        kind: {"utility": "awarded_point_margin", "budgetProfile": budget, **(
            {"horizon": {"unit": "tricks" if kind == "play" else "decisions", "value": 2},
             "boundaryEvaluator": "current_raw_point_margin_v1"} if mode == "budgeted" else {}
        )} for kind in SCENARIO_DECISION_KINDS
    }}


def _quality_bound_result():
    return {
        "researchAuthority": {"backendId": "ortools_cp_sat_v3"},
        "disposition": "qualified",
        "qualification": {
            "qualified": True,
            "failures": [],
            "allDecisionPolicyAuditComplete": True,
            "playingQualityEvidenceHash": "c" * 64,
        },
    }


@pytest.mark.parametrize("mode", ["budgeted", "exact"])
@pytest.mark.parametrize("visibility", [None, "actor_information_set", "complete_state"])
def test_playing_quality_policy_is_closed_actor_relative_and_hash_bound(mode, visibility):
    raw = _quality_policy(mode)
    if visibility is not None:
        raw.update(schemaVersion=2, visibility=visibility)
    policy = ScenarioPlayingQualityPolicyV1.build(raw)
    assert policy.to_dict() == ScenarioPlayingQualityPolicyV1.build(policy.to_dict()).to_dict()
    reordered = {**raw, "decisions": dict(reversed(list(raw["decisions"].items())))}
    assert ScenarioPlayingQualityPolicyV1.build(reordered).sha256 == policy.sha256
    for kind in SCENARIO_DECISION_KINDS:
        for actor, team in (("north", "north_south"), ("south", "north_south"),
                            ("west", "west_east"), ("east", "west_east")):
            request = policy.request_for(kind, actor)
            assert request.target_team == team
            assert request.mode == ("bounded" if mode == "budgeted" else "exact")
            assert request.visibility == (visibility or "actor_information_set")
            assert request.selected_action is None
    policy.decisions["play"]["utility"] = "round_win"
    with pytest.raises(ValueError, match="checksum"):
        policy.request_for("play", "north")


@pytest.mark.parametrize("change", [
    {"targetTeam": "west_east"}, {"visibility": "complete_state"},
    {"mode": "exact"}, {"selectedAction": "SA"}, {"budgetProfile": 10},
    {"utility": " awarded_point_margin "}, {"horizon": {"unit": "decisions", "value": True}},
])
def test_playing_quality_rejects_unsafe_or_coercive_decision_fields(change):
    raw = _quality_policy()
    raw["decisions"]["play"].update(change)
    with pytest.raises((TypeError, ValueError)):
        ScenarioPlayingQualityPolicyV1.build(raw)


def test_playing_quality_cannot_omit_decisions_or_downgrade_exact():
    raw = _quality_policy()
    del raw["decisions"]["bid"]
    with pytest.raises(ValueError, match="every decision kind"):
        ScenarioPlayingQualityPolicyV1.build(raw)
    raw = _quality_policy()
    raw["mode"] = "exact"
    with pytest.raises(ValueError, match="fields do not match"):
        ScenarioPlayingQualityPolicyV1.build(raw)


@pytest.mark.parametrize("mode", ["budgeted", "exact"])
@pytest.mark.parametrize("visibility", [None, "actor_information_set", "complete_state"])
def test_playing_quality_api_uses_the_canonical_validator(mode, visibility):
    from pydantic import ValidationError
    from baloot_platform.service.api import PlayingQualityPolicyV1Request

    raw = _quality_policy(mode)
    if visibility is not None:
        raw.update(schemaVersion=2, visibility=visibility)
    parsed = PlayingQualityPolicyV1Request.model_validate(raw)
    assert "sha256" not in PlayingQualityPolicyV1Request.model_json_schema()["required"]
    assert parsed.model_dump(exclude_none=True) == ScenarioPlayingQualityPolicyV1.build(raw).to_dict()
    for invalid in ({**raw, "mode": True}, {**raw, "sha256": "0" * 64},
                    {**raw, "decisions": {"play": raw["decisions"]["play"]}}):
        with pytest.raises(ValidationError):
            PlayingQualityPolicyV1Request.model_validate(invalid)


@pytest.mark.parametrize("change", [
    {"schemaVersion": 2}, {"schemaVersion": True}, {"schemaVersion": "2"},
    {"visibility": "complete_state"}, {"schemaVersion": 2, "visibility": None},
    {"schemaVersion": 2, "visibility": "unknown"},
])
def test_playing_quality_visibility_version_is_strict(change):
    with pytest.raises((ValueError, TypeError)):
        ScenarioPlayingQualityPolicyV1.build({**_quality_policy(), **change})


def test_playing_quality_legacy_hash_and_visibility_are_preserved():
    raw = _quality_policy()
    legacy = ScenarioPlayingQualityPolicyV1.build(raw)
    assert legacy.sha256 == canonical_sha256(raw)
    assert legacy.to_dict() == {**raw, "sha256": legacy.sha256}
    explicit = ScenarioPlayingQualityPolicyV1.build({**raw, "schemaVersion": 2,
        "visibility": "actor_information_set"})
    assert explicit.sha256 != legacy.sha256
    for kind in SCENARIO_DECISION_KINDS:
        assert explicit.request_for(kind, "west") == legacy.request_for(kind, "west")


def test_playing_quality_resolves_only_frozen_budgets_and_does_not_bypass_guard():
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(), bundle_identity=_bundle_identity(),
    )
    recipe = _recipe(policy=ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3)
    recipe["qualification"] = {"requireAllDecisionAudit": True}
    recipe["playingQuality"] = _quality_policy(budget="unpublished-budget")
    result = backend.validate(recipe)
    assert not result["valid"]
    assert any(row["code"] == "all_decision_quality_evidence_unavailable" for row in result["issues"])
    assert {row["path"] for row in result["issues"] if row["path"].startswith("playingQuality.")} == {
        f"playingQuality.decisions.{kind}.budgetProfile" for kind in SCENARIO_DECISION_KINDS
    }
    del recipe["qualification"]
    result = backend.validate(recipe)
    assert any(row["code"] == "all_decision_quality_evidence_unavailable"
               and row["path"] == "playingQuality" for row in result["issues"])


@pytest.mark.parametrize("mode,origin,order,admitted", [
    ("budgeted", "search", True, True), ("exact", "search", True, True),
    ("budgeted", "exact_trace_prefix", True, False), ("budgeted", "search", False, False)])
def test_native_quality_search_admission_requires_supported_origin_and_frozen_execution(mode, origin, order, admitted):
    profile = _authored_v3_profile()
    profile["scenarioResearchV4"]["qualityExecution"] = {
        "schemaVersion": 1, "maximumWallSeconds": 300,
        "maximumEvidenceBytes": 10_000_000, "maximumEscalationDecisions": 20,
        **({"escalationOpportunityOrder": ["west", "east", "north", "south"]} if order else {})}
    backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile, bundle_identity=_bundle_identity())
    recipe = _recipe(policy=ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3)
    recipe["scenario"]["origin"]["kind"] = origin
    recipe["scenario"]["decisionPoint"] = {"kind": "play", "selector": {"trickIndex": 6}}
    recipe["scenario"]["constraints"] = {"predicateId": "PRED-CONTRACT", "kind": "auction.contract",
        "version": 1, "hardness": "hard", "arguments": {"contract": "hokum"}}
    recipe["playingQuality"] = _quality_policy(mode)
    recipe["qualification"] = {"requireAllDecisionAudit": True}
    result = backend.validate(recipe)
    refused = any(row["code"] == "all_decision_quality_evidence_unavailable" for row in result["issues"])
    assert refused is not admitted
    if not admitted:
        expected = (
            "exact_input_recorded_quality_dispatch_unavailable" if origin == "exact_trace_prefix" else
            "frozen_quality_execution_unavailable"
        )
        assert next(row for row in result["issues"] if row["code"] == "all_decision_quality_evidence_unavailable")["detail"] == expected
    if admitted:
        assert result["valid"], result["issues"]


@pytest.mark.parametrize("count,with_policy", [(0, False), (2, True), (3, True)])
def test_recorded_quality_history_is_bounded_and_never_silently_ignored(count, with_policy):
    profile = _authored_v3_profile()
    profile["scenarioResearchV4"]["qualityExecution"] = {
        "schemaVersion": 1, "maximumWallSeconds": 300,
        "maximumEvidenceBytes": 10_000_000, "maximumEscalationDecisions": 2,
        "escalationOpportunityOrder": ["west", "east", "north", "south"],
    }
    backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile, bundle_identity=_bundle_identity())
    recipe = _recipe(policy=ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3)
    recipe["scenario"]["origin"]["recordedEscalationOpportunities"] = [
        {"actor": "west", "actionCode": "decline"} for _ in range(count)
    ]
    if with_policy:
        recipe["playingQuality"] = _quality_policy()
    result = backend.validate(recipe)
    codes = {row["code"] for row in result["issues"]}
    assert not result["valid"]
    assert "recorded_escalation_history_audit_unavailable" in codes
    assert ("recorded_escalation_history_exceeds_budget" in codes) == (count > 2)
    assert ("recorded_escalation_history_requires_quality_policy" in codes) == (not with_policy)


@pytest.mark.parametrize("history,code,field", [
    ("missing", "exact_input_project_actions_required", "projectActions"),
    ("legacy", "exact_input_project_declarations_legacy", "projectDeclarations"),
    ("canonical", None, None),
])
def test_exact_quality_project_history_diagnostic_names_the_authored_field(history, code, field):
    from baloot_platform.research.service import ResearchFoundryServiceV1

    profile = _authored_v3_profile()
    profile["scenarioResearchV4"]["qualityExecution"] = {
        "schemaVersion": 1, "maximumWallSeconds": 300,
        "maximumEvidenceBytes": 10_000_000, "maximumEscalationDecisions": 20,
        "escalationOpportunityOrder": ["west", "east", "north", "south"],
    }
    backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile, bundle_identity=_bundle_identity())
    prefix = {"schemaVersion": 1, "deckOrder": list(range(32)), "dealer": "east",
        "auctionActions": [], "escalationActions": [], "projectDeclarations": [], "playActions": []}
    if history != "missing":
        prefix["projectActions"] = []
    if history == "legacy":
        prefix["projectDeclarations"] = [{"owner": "west", "kind": "siri", "highRank": 5,
            "value": 2, "declarationOrder": 1}]
    recipe = _recipe(policy=ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3)
    recipe["scenario"]["origin"] = {"kind": "exact_trace_prefix",
        "exactTracePrefix": ExactTracePrefixV1.build(prefix).to_dict(),
        "recordedEscalationOpportunities": []}
    recipe["scenario"]["decisionPoint"] = {"kind": "play", "selector": {"trickIndex": 6}}
    recipe["playingQuality"] = _quality_policy()
    recipe["qualification"] = {"requireAllDecisionAudit": True}
    result = backend.validate(recipe)
    assert not any(row["code"] == "recorded_escalation_history_audit_unavailable" for row in result["issues"])
    if code is None:
        assert result["valid"], result["issues"]
    else:
        assert not result["valid"]
        issue = next(row for row in result["issues"] if row["code"] == code)
        assert issue["path"] == f"scenario.origin.exactTracePrefix.{field}"
        assert "native" in issue["detail"]
        assert any(row["code"] == "all_decision_quality_evidence_unavailable" for row in result["issues"])
        diagnostic = ResearchFoundryServiceV1._ortools_preflight_issue_v2(issue,
            recipe_hash="c" * 64, workflow_copy={"research_validation_failed": "validation failed"})
        assert diagnostic["class"] == "schema_invalid"
        assert diagnostic["fieldPaths"] == [issue["path"]]
        assert diagnostic["details"]["detail"] == issue["detail"]


@pytest.mark.parametrize("unit,value,actions", [("plies", 17, 17), ("decisions", 17, 17), ("tricks", 5, 20)])
@pytest.mark.parametrize("limit,code", [
    ("maximumHorizonActions", "bounded_horizon_exceeds_proof_budget"),
    ("maximumDepth", "bounded_depth_exceeds_proof_budget"),
])
def test_playing_quality_horizons_respect_each_frozen_profile(unit, value, actions, limit, code):
    profile = _authored_v3_profile()
    budget = profile["scenarioResearchV4"]["proofBudgets"]["profiles"]["late_game_exact_v1"]
    budget["maximumHorizonActions"] = 64
    budget["maximumDepth"] = 64
    budget[limit] = 16
    backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile, bundle_identity=_bundle_identity())
    recipe = _recipe(policy=ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3)
    recipe["playingQuality"] = _quality_policy()
    recipe["playingQuality"]["decisions"]["play"]["horizon"] = {"unit": unit, "value": value}
    # Mutating the caller's copy cannot increase the frozen authority's limit.
    budget[limit] = 100
    issues = backend.validate(recipe)["issues"]
    failures = [row for row in issues if row["code"] == code]
    assert failures == [{"code": code, "path": "playingQuality.decisions.play.horizon",
                         "detail": f"requestedActions={actions};{limit}=16"}]
    assert any(row["code"] == "all_decision_quality_evidence_unavailable" for row in issues)
    recipe["playingQuality"]["decisions"]["play"]["horizon"]["value"] = 4 if unit == "tricks" else 16
    assert not any(row["code"] == code for row in backend.validate(recipe)["issues"])


@pytest.mark.parametrize(("decision_kind", "unit"), [("project", "tricks"), ("play", "plies")])
def test_playing_quality_rejects_unsupported_horizon_before_execution(decision_kind, unit):
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe(policy=ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3)
    recipe["playingQuality"] = _quality_policy()
    recipe["playingQuality"]["decisions"][decision_kind]["horizon"] = {
        "unit": unit,
        "value": 1,
    }

    failures = [
        issue
        for issue in backend.validate(recipe)["issues"]
        if issue["code"] == "playing_quality_horizon_unit_unsupported"
    ]

    assert failures == [
        {
            "code": "playing_quality_horizon_unit_unsupported",
            "path": f"playingQuality.decisions.{decision_kind}.horizon",
            "detail": f"unit={unit}",
        }
    ]


@pytest.mark.parametrize("origin", ["search", "exact_trace_prefix"])
def test_native_full_round_quality_requirement_fails_before_search(origin):
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe(policy=ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3)
    recipe["scenario"]["origin"] = {"kind": origin}
    recipe["qualification"] = {
        "requireAllDecisionAudit": True,
        "minimumSurroundingQuality": 70,
        "maximumRegret": 0,
    }
    original = copy.deepcopy(recipe)
    validation = backend.preflight(recipe)
    assert validation["valid"] is False
    assert validation["executionReady"] is False
    assert validation["issues"][0] == {
        "code": "all_decision_quality_evidence_unavailable",
        "path": "qualification.requireAllDecisionAudit",
        "detail": "selected_decision_proof_only",
    }
    estimate = backend.estimate(recipe)
    assert estimate["launchAdmissible"] is False
    assert estimate["requestedCombinationSupported"] is False
    assert estimate["refusalReason"] == "all_decision_quality_evidence_unavailable"
    assert estimate["exhaustiveSearchInfeasibilityEstablishable"] is False
    assert recipe == original


def test_native_quality_admission_does_not_reinterpret_frozen_v2():
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe()
    recipe["qualification"] = {"requireAllDecisionAudit": True}
    assert not any(
        item["code"] == "all_decision_quality_evidence_unavailable"
        for item in backend.preflight(recipe)["issues"]
    )


@pytest.mark.parametrize("missing", ["both", "policy", "evidence"])
def test_native_result_writer_cannot_bypass_full_round_quality_admission(missing):
    from baloot_platform.research.result_authority import build_ortools_research_result_v3

    def forbidden_write(*_args):
        pytest.fail("Unproved full-round quality must be refused before artifact writes")

    with pytest.raises(ValueError, match="all_decision_quality_evidence_unavailable"):
        build_ortools_research_result_v3(
            run={},
            recipe={"qualification": {"requireAllDecisionAudit": True},
                    **({"playingQuality": _quality_policy()} if missing == "evidence" else {})},
            executor_result={"researchAuthority": {"backendId": "ortools_cp_sat_v3"}},
            campaign_semantic_hash="a" * 64,
            terminal_status="satisfied",
            artifact_documents=({"completeTrace": {"playingQualityEvidence": {}}}
                                if missing == "policy" else {}),
            artifact_references={},
            configuration_snapshot={},
            lowering_manifest={},
            authority_artifact_writer=forbidden_write,
            runtime_compatibility_check=forbidden_write,
            created_at="2026-09-05T00:00:00Z",
        )


@pytest.mark.parametrize("full_round", [True, False])
def test_native_single_decision_proof_does_not_approve_full_round_media(full_round):
    from baloot_platform.research.render_service import ResearchRenderPreparationServiceV1

    result = {
        "researchAuthority": {"backendId": "ortools_cp_sat_v3"},
        "disposition": "qualified",
        "qualification": {"qualified": True, "failures": []},
        "decisionCount": 1,
        "truthLabel": "exact_optimal",
    }
    original = copy.deepcopy(result)
    eligible, _, _, blockers, policy = (
        ResearchRenderPreparationServiceV1._scenario_publication_governance(
            result,
            proof={"completionStatus": "exact_complete", "claimSatisfied": True},
            exact_state_id="STATE-TEST",
            video_profile_hash="a" * 64,
            full_round=full_round,
        )
    )
    assert eligible is not full_round
    assert blockers == (["all_decision_quality_evidence_unavailable", "critical_decision_not_proven"] if full_round else [])
    assert policy["videoProduction"]["gatePassed"] is not full_round
    assert policy["videoProduction"]["candidate"]["critical"] is False
    assert result == original


@pytest.mark.parametrize("case,critical", [
    ("unique", True), ("tied", False), ("missing", False),
    ("bounded", False), ("policy", False), ("recorded_mismatch", False),
    ("recorded_loss", True), ("missing_terminal", True),
    ("unsupported_utility", True), ("unmet_threshold", True),
    ("nonfinite_threshold", True),
])
def test_flagship_criticality_requires_exact_counterfactual_proof(case, critical):
    from baloot_platform.research.render_service import ResearchRenderPreparationServiceV1

    roots = [
        {"action": "SA", "value": 10, "satisfies": True},
        {"action": "S7", "value": -10, "satisfies": False},
    ]
    if case == "tied":
        roots[1]["value"] = 10
    if case == "missing":
        roots[1].pop("value")
    mode = {"bounded": "bounded_complete", "policy": "policy_complete"}.get(case, "exact_complete")
    eligible, _, _, blockers, policy = ResearchRenderPreparationServiceV1._scenario_publication_governance(
        _quality_bound_result(),
        proof={"completionStatus": mode, "claimSatisfied": True,
               "selectedAction": "SA", "rootActions": roots,
               "targetTeam": "north_south",
               "utility": "match_win" if case == "unsupported_utility" else "round_win",
               "successThreshold": (float("nan") if case == "nonfinite_threshold"
                                    else 2 if case == "unmet_threshold" else 0)},
        exact_state_id="STATE-TEST", video_profile_hash="a" * 64, full_round=True,
        recorded_action="S7" if case == "recorded_mismatch" else "SA",
        recorded_terminal_score=(None if case == "missing_terminal" else
                                 {"winner": "west_east" if case == "recorded_loss" else "north_south"}),
    )
    assert eligible is False
    terminal_blocker = case in {"recorded_loss", "missing_terminal", "unsupported_utility",
                                "unmet_threshold", "nonfinite_threshold"}
    assert blockers == ["all_decision_quality_evidence_unavailable",
                        *([] if critical else ["critical_decision_not_proven"]),
                        *(["recorded_terminal_utility_unverified"] if terminal_blocker else [])]
    assert policy["videoProduction"]["candidate"]["critical"] is critical


def test_v3_render_rejects_recipe_proof_substitution():
    from baloot_platform.research.render_service import ResearchRenderPreparationServiceV1

    with pytest.raises(ValueError, match="differs from frozen recipe"):
        ResearchRenderPreparationServiceV1._scenario_publication_governance(
            {"researchAuthority": {"backendId": "ortools_cp_sat_v3"}},
            proof={"utility": "round_win", "targetTeam": "north_south",
                   "successThreshold": 0, "identities": {"proofRequestHash": "a" * 64}},
            proof_request={"utility": "round_win", "targetTeam": "west_east",
                           "successThreshold": 0, "sha256": "a" * 64},
            exact_state_id="STATE-TEST", video_profile_hash="b" * 64,
        )


def test_v3_full_round_budgeted_quality_receipt_passes_the_legacy_mode_gate(monkeypatch):
    from baloot_platform.research import render_service

    # Isolate the terminal gate; the quality receipt's own validator is tested separately.
    monkeypatch.setattr(render_service, "validate_trace_quality_evidence_v1", lambda *args, **kwargs: None)
    proof = {"completionStatus": "exact_complete", "claimSatisfied": True,
             "selectedAction": "SA", "targetTeam": "north_south", "utility": "round_win",
             "successThreshold": 0, "identities": {"proofRequestHash": "a" * 64},
             "rootActions": [{"action": "SA", "value": 1, "satisfies": True},
                             {"action": "S7", "value": -1, "satisfies": False}]}
    eligible, _, _, blockers, _ = render_service.ResearchRenderPreparationServiceV1._scenario_publication_governance(
        _quality_bound_result(),
        proof=proof,
        proof_request={"utility": "round_win", "targetTeam": "north_south",
                       "successThreshold": 0, "sha256": "a" * 64},
        exact_state_id="STATE-TEST", video_profile_hash="b" * 64,
        full_round=True, recorded_action="SA",
        recorded_terminal_score={"winner": "north_south"},
        quality_trace={"playingQualityEvidence": {"receipt": {"status": "complete", "sha256": "c" * 64}}},
        quality_policy=ScenarioPlayingQualityPolicyV1.build(_quality_policy()), rules_hash="d" * 64,
    )
    assert eligible is True
    assert blockers == []


@pytest.mark.parametrize("target", ["north_south", "west_east"])
@pytest.mark.parametrize("case", ["won", "unfinished", "lost", "missing", "invalid", "wrong_value", "below_threshold"])
def test_v3_match_video_requires_the_owned_match_outcome(monkeypatch, target, case):
    from baloot_platform.research import render_service

    # Exercise admission, not a substitute for the independently tested proof reader.
    monkeypatch.setattr(render_service, "validate_trace_quality_evidence_v1", lambda *args, **kwargs: None)
    other = "west_east" if target == "north_south" else "north_south"
    score = {"winner": other, "matchWinner": target}
    selected_value, threshold = 1, 1
    if case == "unfinished":
        score["matchWinner"], selected_value, threshold = None, 0, 0
    elif case == "lost":
        score.update(winner=target, matchWinner=other)
    elif case == "missing":
        score.pop("matchWinner")
    elif case == "invalid":
        score["matchWinner"] = "north"
    elif case == "wrong_value":
        selected_value, threshold = 0, 0
    elif case == "below_threshold":
        score["matchWinner"] = None
    proof = {"completionStatus": "exact_complete", "claimSatisfied": True,
             "selectedAction": "SA", "targetTeam": target, "utility": "match_win",
             "successThreshold": threshold, "identities": {"proofRequestHash": "a" * 64},
             "rootActions": [{"action": "SA", "value": selected_value, "satisfies": True},
                             {"action": "S7", "value": -1, "satisfies": False}]}
    eligible, _, _, blockers, policy = render_service.ResearchRenderPreparationServiceV1._scenario_publication_governance(
        _quality_bound_result(), proof=proof,
        proof_request={"utility": "match_win", "targetTeam": target,
                       "successThreshold": threshold, "sha256": "a" * 64},
        exact_state_id="STATE-TEST", video_profile_hash="b" * 64,
        full_round=True, recorded_action="SA", recorded_terminal_score=score,
        quality_trace={"playingQualityEvidence": {"receipt": {"status": "complete", "sha256": "c" * 64}}},
        quality_policy=ScenarioPlayingQualityPolicyV1.build(_quality_policy()), rules_hash="d" * 64,
    )
    assert eligible is (case in {"won", "unfinished"})
    assert blockers == ([] if eligible else ["recorded_terminal_utility_unverified"])
    assert policy["videoProduction"]["candidate"]["critical"] is True


def _featured_native_quality_artifact(complete_state):
    return {
        "request": {"visibility": "complete_state" if complete_state else "actor_information_set",
                    "mode": "exact" if complete_state else "bounded"},
        "proofModel": {"kind": "native_complete_state_root_outcome_v1" if complete_state else "native_observed_root_outcome_v1",
                       **({"rootActionCountScope": "exact_fixed_deal_terminal_tail"} if complete_state else {})},
        "proof": {"rootActions": [{"action": "H10"}, {"action": "SA"}],
                  "completionStatus": "exact_complete" if complete_state else "bounded_complete",
                  **({} if complete_state else {"hiddenWorlds": {
                      "method": "supplied_worlds_only", "count": 6, "complete": False}})},
    }


@pytest.mark.parametrize("complete_state", [False, True])
@pytest.mark.parametrize("mutation", [None, "mode", "visibility", "status", "kind", "forced", "worlds"])
def test_featured_native_proof_rejects_scope_substitution(complete_state, mutation):
    from baloot_platform.research.result_authority import native_featured_play_proof_supported_v1
    artifact = _featured_native_quality_artifact(complete_state)
    if mutation == "mode":
        artifact["request"]["mode"] = "bounded" if complete_state else "exact"
    elif mutation == "visibility":
        artifact["request"]["visibility"] = "actor_information_set" if complete_state else "complete_state"
    elif mutation == "status":
        artifact["proof"]["completionStatus"] = "proof_incomplete"
    elif mutation == "kind":
        artifact["proofModel"]["kind"] = "unknown"
    elif mutation == "forced":
        artifact["proof"]["rootActions"] = artifact["proof"]["rootActions"][:1]
    elif mutation == "worlds":
        artifact["proof"]["hiddenWorlds"] = {"method": "supplied_worlds_only", "count": True, "complete": False}
    assert native_featured_play_proof_supported_v1(artifact) is (mutation is None)
    assert native_featured_play_proof_supported_v1(None) is False


@pytest.mark.parametrize("complete_state", [False, True])
@pytest.mark.parametrize("case", [
    "valid", "valid_new_copy", "missing", "malformed", "mismatch", "below_threshold",
])
def test_v3_flagship_awarded_margin_requires_owned_terminal_score(monkeypatch, case, complete_state):
    from baloot_platform.research import render_service

    monkeypatch.setattr(render_service, "validate_trace_quality_evidence_v1", lambda *args, **kwargs: None)
    score = {"winner": "west_east", "awardedPoints": [8, 22]}
    if case == "missing":
        score.pop("awardedPoints")
    elif case == "malformed":
        score["awardedPoints"] = [8, True]
    elif case == "mismatch":
        score["awardedPoints"] = [8, 21]
    proof = {"completionStatus": "exact_complete", "claimSatisfied": True,
             "selectedAction": "H10", "targetTeam": "west_east",
             "utility": "awarded_point_margin", "successThreshold": 15 if case == "below_threshold" else 12,
             "identities": {"proofRequestHash": "a" * 64},
             "rootActions": [{"action": "SA", "value": 6, "satisfies": False},
                             {"action": "H10", "value": 14, "satisfies": True},
                             {"action": "DA", "value": 10, "satisfies": False}]}
    admitted, _, _, blockers, _ = render_service.ResearchRenderPreparationServiceV1._scenario_publication_governance(
        _quality_bound_result(),
        proof=proof,
        proof_request={"utility": "awarded_point_margin", "targetTeam": "west_east",
                       "successThreshold": proof["successThreshold"], "sha256": "a" * 64},
        exact_state_id="STATE-TEST", video_profile_hash="b" * 64,
        video_profile=({"localizations": {"video.analysis.unique_awarded_margin": "copy"}}
                       if case == "valid_new_copy" else None),
        full_round=True, recorded_action="H10", recorded_terminal_score=score,
        quality_trace={"exactTracePrefix": {"playActions": []},
                       "playingQualityEvidence": {"decisionPoint": {"kind": "escalation"},
                                                  "receipt": {"status": "complete", "sha256": "c" * 64,
                                                              "qualityExecution": {"featuredPlayNativeProof": True},
                                                              "postAuction": {"projectPlay": {
                                                                  "decisions": [{"exactStateId": "STATE-TEST", "action": "H10",
                                                                                 "decisionKind": "play", "playIndex": 0,
                                                                                 "proofArtifactHash": "e" * 64}],
                                                                  "artifacts": {"e" * 64: _featured_native_quality_artifact(complete_state)},
                                                              }}}}},
        quality_policy=ScenarioPlayingQualityPolicyV1.build(_quality_policy()), rules_hash="d" * 64,
    )
    assert admitted is (case == "valid_new_copy")
    assert blockers == [*(
        [] if case == "valid_new_copy" else
        ["score_swing_explanation_unavailable"] if case == "valid" else
        ["recorded_terminal_utility_unverified", "score_swing_explanation_unavailable"]
    )]


@pytest.mark.parametrize("missing", ["flag", "prefix", "forced", "wrong_action",
                                     "wrong_kind", "wrong_index", "missing_index", "bool_index"])
@pytest.mark.parametrize("complete_state", [False, True])
def test_v3_awarded_margin_without_owned_featured_proof_is_audit_only(monkeypatch, missing, complete_state):
    from baloot_platform.research import render_service

    monkeypatch.setattr(render_service, "validate_trace_quality_evidence_v1", lambda *args, **kwargs: None)
    trace = {"exactTracePrefix": {"playActions": []},
             "playingQualityEvidence": {"decisionPoint": {"kind": "play"},
                                        "receipt": {"status": "complete", "sha256": "c" * 64,
                                                    "qualityExecution": {"featuredPlayNativeProof": True},
                                                    "postAuction": {"projectPlay": {
                                                        "decisions": [{"exactStateId": "STATE-TEST", "action": "H10",
                                                                       "decisionKind": "play", "playIndex": 0,
                                                                       "proofArtifactHash": "e" * 64}],
                                                        "artifacts": {"e" * 64: _featured_native_quality_artifact(complete_state)},
                                                    }}}}}
    if missing == "flag":
        trace["playingQualityEvidence"]["receipt"]["qualityExecution"] = {}
    elif missing == "prefix":
        trace.pop("exactTracePrefix")
    if missing == "forced":
        trace["playingQualityEvidence"]["receipt"]["postAuction"]["projectPlay"]["artifacts"]["e" * 64]["proofModel"]["kind"] = "forced_legal_action_model_v1"
    if missing == "wrong_action":
        trace["playingQualityEvidence"]["receipt"]["postAuction"]["projectPlay"]["decisions"][0]["action"] = "SA"
    row = trace["playingQualityEvidence"]["receipt"]["postAuction"]["projectPlay"]["decisions"][0]
    if missing == "wrong_kind":
        row["decisionKind"] = "project"
    elif missing == "wrong_index":
        row["playIndex"] = 1
    elif missing == "missing_index":
        row.pop("playIndex")
    elif missing == "bool_index":
        row["playIndex"] = False
    eligible, _, _, blockers, _ = render_service.ResearchRenderPreparationServiceV1._scenario_publication_governance(
        _quality_bound_result(),
        proof={"completionStatus": "exact_complete", "claimSatisfied": True,
               "selectedAction": "H10", "targetTeam": "west_east", "utility": "awarded_point_margin",
               "successThreshold": 12,
               "rootActions": [{"action": "H10", "value": 14, "satisfies": True},
                               {"action": "SA", "value": 6, "satisfies": False}]},
        exact_state_id="STATE-TEST", video_profile_hash="b" * 64,
        video_profile={"localizations": {"video.analysis.unique_awarded_margin": "copy"}},
        full_round=True, recorded_action="H10",
        recorded_terminal_score={"awardedPoints": [8, 22]},
        quality_trace=trace,
        quality_policy=ScenarioPlayingQualityPolicyV1.build(_quality_policy()), rules_hash="d" * 64,
    )
    assert eligible is False
    assert blockers == [
        "featured_play_native_proof_unavailable",
    ]


def test_flagship_recorded_action_comes_from_frozen_trace(root):
    from baloot_platform.research.render_service import _recorded_scenario_action_v3

    fixture = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(encoding="utf-8"))
    complete = dict(fixture["exactTracePrefix"])
    complete.pop("sha256", None)
    source = {**complete, "playActions": complete["playActions"][:22]}
    source_prefix = ExactTracePrefixV1.build(source)
    full_prefix = ExactTracePrefixV1.build(complete)
    assert source_prefix.sha256 != _effective_prefix_hash(source_prefix)
    exact = {"decisionKind": "play", "prefixHash": _effective_prefix_hash(source_prefix),
             "nextActor": "north", "legalActions": [{"code": "D9"}, {"code": "S7"}]}
    trace = {"exactTracePrefix": source_prefix.to_dict(),
             "completeExactTracePrefix": full_prefix.to_dict()}
    assert _recorded_scenario_action_v3(trace, exact) == "D9"
    assert _recorded_scenario_action_v3(trace, {**exact, "prefixHash": "0" * 64}) is None


def test_native_decision_clip_uses_the_same_retained_trace_as_full_export(root):
    source = (root / "src/baloot_platform/research/render_service.py").read_text(
        encoding="utf-8"
    )
    assert 'if ortools_result and preset != "decision_clip"' not in source
    assert "trusted_complete_trace=(\n                complete_trace\n                if ortools_result\n" in source


@pytest.mark.parametrize("kind,field,recorded,legal,actor", [
    ("bid", "auctionActions", {"actor": "north", "action": "sun"},
     {"code": "sun", "action": "sun"}, "north"),
    ("escalation", "escalationActions", {"actor": "west", "action": "decline"},
     {"code": "decline", "action": "decline"}, "west"),
    ("project", "projectActions", {"actor": "north", "action": "pass"},
     {"code": "pass", "action": "pass"}, "north"),
])
def test_recorded_action_binding_covers_nonplay_decisions(root, kind, field, recorded, legal, actor):
    from baloot_platform.research.render_service import _recorded_scenario_action_v3

    fixture = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(encoding="utf-8"))
    raw = {**fixture["exactTracePrefix"], "playActions": [], "projectActions": [], "escalationActions": []}
    raw.pop("sha256", None)
    if kind == "bid":
        raw["auctionActions"] = []
    source = ExactTracePrefixV1.build(raw)
    complete = ExactTracePrefixV1.build({**raw, field: [recorded]})
    trace = {"exactTracePrefix": source.to_dict(), "completeExactTracePrefix": complete.to_dict()}
    exact = {"decisionKind": kind, "prefixHash": _effective_prefix_hash(source),
             "nextActor": actor, "legalActions": [legal]}
    assert _recorded_scenario_action_v3(trace, exact) == legal["code"]


def test_historical_render_criticality_admission_is_unchanged():
    from baloot_platform.research.render_service import ResearchRenderPreparationServiceV1

    eligible, _, _, blockers, policy = ResearchRenderPreparationServiceV1._scenario_publication_governance(
        {"researchAuthority": {"backendId": "historical_test_backend"},
         "disposition": "qualified", "qualification": {"qualified": True, "failures": []}},
        proof={"completionStatus": "policy_complete", "claimSatisfied": True,
               "selectedAction": "SA", "rootActions": [{"action": "SA", "value": 0, "satisfies": True}]},
        exact_state_id="STATE-TEST", video_profile_hash="a" * 64, full_round=True,
    )
    assert eligible is True and blockers == []
    assert policy["videoProduction"]["candidate"]["critical"] is True


@pytest.mark.parametrize("disposition", ["qualified", "rejected", "incomplete"])
def test_actor_interest_render_admission_cannot_be_bypassed_by_audit_disposition(disposition):
    from baloot_platform.research.render_service import ResearchRenderPreparationServiceV1

    _, _, _, blockers, _ = ResearchRenderPreparationServiceV1._scenario_publication_governance(
        {"researchAuthority": {"backendId": "ortools_cp_sat_v3"}, "disposition": disposition,
         "qualification": {"qualified": disposition == "qualified", "failures": []}},
        proof={"completionStatus": "exact_complete", "claimSatisfied": True},
        exact_state_id="STATE-TEST", video_profile_hash="a" * 64, full_round=True,
        quality_settings={"qualityExecution": {"videoInterestingness": {"schemaVersion": 4}}},
    )
    assert "interestingness_verified_evidence_unavailable" in blockers
    with pytest.raises(ValueError, match="pre-render interestingness screen rejected"):
        ResearchRenderPreparationServiceV1._require_interest_admission({"schemaVersion": 4}, disposition, blockers)
    if disposition != "qualified":
        ResearchRenderPreparationServiceV1._require_interest_admission({"schemaVersion": 3}, disposition, blockers)


def test_public_interest_admission_does_not_expose_private_alternatives():
    from baloot_platform.research.render_service import ResearchRenderPreparationServiceV1

    _, _, _, _, policy = ResearchRenderPreparationServiceV1._scenario_publication_governance(
        {"researchAuthority": {"backendId": "historical_test_backend"}, "disposition": "qualified",
         "qualification": {"qualified": True, "failures": []}},
        proof={"completionStatus": "exact_complete", "claimSatisfied": True},
        exact_state_id="STATE-PRIVATE", video_profile_hash="a" * 64, full_round=False,
    )
    policy["interestVerdict"] = {"passed": True, "sha256": "b" * 64,
                                 "meaningfulDecisions": [{"baselineAction": "CA", "baselineRegret": 34}]}
    public = ResearchRenderPreparationServiceV1._public_scenario_publication_policy(policy, result_code="RESULT-TEST")
    assert "interestVerdict" not in public
    assert public["interestAdmission"] == {"passed": True, "evidenceHash": "b" * 64}
    assert public["sha256"] == canonical_sha256({key: value for key, value in public.items() if key != "sha256"})


def test_project_selected_action_accepts_shared_domain_codes() -> None:
    _validate_selected_action("project", "pass")
    _validate_selected_action("project", "declare:PROJECT-0123456789ABCDEF")
    _validate_selected_action("project", "declare:siri:20:2")
    with pytest.raises(ValueError, match="invalid for project"):
        _validate_selected_action("project", "declare:project-0123456789abcdef")


@pytest.mark.parametrize("status", ["incomplete", "complete"])
def test_full_round_render_cannot_use_unverified_quality_receipt(status):
    from baloot_platform.research.render_service import ResearchRenderPreparationServiceV1
    kwargs = dict(result={"researchAuthority": {"backendId": "ortools_cp_sat_v3"},
                         "disposition": "qualified", "qualification": {"qualified": True, "failures": []}},
        proof={"completionStatus": "exact_complete", "claimSatisfied": True},
        exact_state_id="STATE-TEST", video_profile_hash="a" * 64, full_round=True,
        quality_trace={"playingQualityEvidence": {"receipt": {"status": status}}},
        quality_policy=ScenarioPlayingQualityPolicyV1.build(_quality_policy()), rules_hash="b" * 64)
    if status == "complete":
        with pytest.raises(ValueError, match="quality evidence shape mismatch"):
            ResearchRenderPreparationServiceV1._scenario_publication_governance(**kwargs)
    else:
        eligible, _, _, blockers, policy = ResearchRenderPreparationServiceV1._scenario_publication_governance(**kwargs)
        assert not eligible
        assert "all_decision_quality_evidence_unavailable" in blockers
        assert "qualityEvidenceHash" not in policy


@pytest.mark.parametrize("full_round", [False, True])
def test_render_reuses_only_the_result_bound_verified_quality_audit(monkeypatch, full_round):
    from baloot_platform.research import render_service
    from baloot_platform.service.api import ResearchRenderPrepareRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ResearchRenderPrepareRequest.model_validate({"_validated_quality_audit": {}})

    def unexpected_reproof(*_args, **_kwargs):
        raise AssertionError("duplicate quality reproof")

    monkeypatch.setattr(render_service, "validate_trace_quality_evidence_v1", unexpected_reproof)
    result = {**_quality_bound_result(), "resultCode": "LAF-TEST", "resultHash": "a" * 64}
    trace = {"playingQualityEvidence": {"receipt": {"status": "complete", "sha256": "c" * 64}}}
    core = {"resultCode": result["resultCode"], "resultHash": result["resultHash"],
            "status": "complete", "scope": "all_decisions",
            "completeTraceHash": render_service._canonical_sha256(trace),
            "qualityReceiptHash": "c" * 64}
    audit = {**core, "sha256": render_service._canonical_sha256(core)}
    kwargs = dict(result=result, proof={"completionStatus": "exact_complete", "claimSatisfied": True},
                  exact_state_id="STATE-TEST", video_profile_hash="b" * 64, full_round=full_round,
                  quality_trace=trace, quality_policy=ScenarioPlayingQualityPolicyV1.build(_quality_policy()),
                  rules_hash="d" * 64)
    if not full_round:
        kwargs["quality_settings"] = {"qualityExecution": {
            "schemaVersion": 1, "maximumWallSeconds": 19,
            "maximumEvidenceBytes": 23, "maximumEscalationDecisions": 7,
            "videoInterestingness": {"schemaVersion": 4, "maximumFinalMargin": 40,
                "minimumPointSwing": 20, "maximumFeaturedTrick": 7,
                "minimumMeaningfulDecisions": 1, "minimumConsistentWorlds": 2,
                "minimumNextBestRegret": 20, "minimumBaselineRegret": 20,
                "baselinePolicyId": "observer_safe_partner_feed_established_control_v1",
                "allowTieRuleOnlyLessons": False}}}
        monkeypatch.setattr(render_service, "evaluate_round_interestingness_v1", lambda *_args, **_kwargs: {
            "policy": {"schemaVersion": 4}, "blockers": [], "passed": True})
        monkeypatch.setattr(render_service, "interest_admission_summary_v1", lambda _screen: {"passed": True})
        result["qualification"]["interestAdmission"] = {"passed": True}
    check = render_service.ResearchRenderPreparationServiceV1._scenario_publication_governance
    # Standalone/historical callers still perform the full validator, never a hash-only fallback.
    with pytest.raises(AssertionError, match="duplicate quality reproof"):
        check(**kwargs)
    _, _, _, blockers, policy = check(**kwargs, _validated_quality_audit=audit)
    assert blockers == (["critical_decision_not_proven"] if full_round else [])
    assert policy["qualityEvidenceHash"] == "c" * 64
    for field in (*core, "sha256"):
        changed = {**audit, field: "substituted"}
        if field != "sha256":
            changed["sha256"] = render_service._canonical_sha256(
                {key: value for key, value in changed.items() if key != "sha256"})
        with pytest.raises(ValueError, match="quality audit binding mismatch"):
            check(**kwargs, _validated_quality_audit=changed)
    with pytest.raises(ValueError, match="quality audit binding mismatch"):
        check(**{**kwargs, "quality_trace": {**trace, "substituted": True}},
              _validated_quality_audit=audit)


@pytest.mark.parametrize("tamper", ["flag", "hash"])
def test_full_round_render_requires_result_bound_quality_receipt(monkeypatch, tamper):
    from baloot_platform.research import render_service

    monkeypatch.setattr(
        render_service, "validate_trace_quality_evidence_v1", lambda *args, **kwargs: None
    )
    result = _quality_bound_result()
    if tamper == "flag":
        result["qualification"]["allDecisionPolicyAuditComplete"] = False
    else:
        result["qualification"]["playingQualityEvidenceHash"] = "d" * 64
    eligible, _, _, blockers, _ = (
        render_service.ResearchRenderPreparationServiceV1._scenario_publication_governance(
            result,
            proof={
                "completionStatus": "exact_complete",
                "claimSatisfied": True,
                "selectedAction": "SA",
                "targetTeam": "north_south",
                "utility": "round_win",
                "successThreshold": 0,
                "rootActions": [
                    {"action": "SA", "value": 1, "satisfies": True},
                    {"action": "S7", "value": -1, "satisfies": False},
                ],
            },
            exact_state_id="STATE-TEST",
            video_profile_hash="a" * 64,
            full_round=True,
            recorded_action="SA",
            recorded_terminal_score={"winner": "north_south"},
            quality_trace={
                "playingQualityEvidence": {
                    "receipt": {"status": "complete", "sha256": "c" * 64}
                }
            },
            quality_policy=ScenarioPlayingQualityPolicyV1.build(_quality_policy()),
            rules_hash="b" * 64,
        )
    )
    assert eligible is False
    assert "all_decision_quality_result_binding_unavailable" in blockers


@pytest.mark.parametrize("interest", [False, True])
@pytest.mark.parametrize("visibility", ["complete_state", "actor_information_set"])
def test_exact_full_round_receipt_removes_policy_quality_blocker(monkeypatch, rules, interest, visibility):
    from baloot_platform.research import render_service

    calls = []
    settings = {
        "proofBudgets": {"profiles": {"frozen": {"maximumNodes": 17}}},
        "searchBudgets": {"profiles": {"frozen": {"maximumConflicts": 29}}},
        "qualityExecution": {"schemaVersion": 1, "maximumWallSeconds": 19,
                             "maximumEvidenceBytes": 23, "maximumEscalationDecisions": 7},
    }
    if interest:
        settings["qualityExecution"]["videoInterestingness"] = {
            "schemaVersion": 1, "maximumFinalMargin": 20, "minimumPointSwing": 20,
            "maximumFeaturedTrick": 5, "minimumMeaningfulDecisions": 2}
    def validate(*args, **kwargs):
        assert kwargs["rules"] is rules
        assert kwargs["rules_hash"] == rules.sha256
        assert kwargs["required_proof_profiles"] == settings["proofBudgets"]["profiles"]
        assert kwargs["required_search_profiles"] == settings["searchBudgets"]["profiles"]
        assert kwargs["required_execution"].maximum_wall_seconds == 19
        assert kwargs["require_family_opening"] is True
        calls.append(kwargs)
    monkeypatch.setattr(render_service, "validate_trace_quality_evidence_v1", validate)
    eligible, _, _, blockers, _ = (
        render_service.ResearchRenderPreparationServiceV1._scenario_publication_governance(
            _quality_bound_result(),
            proof={"completionStatus": "exact_complete", "claimSatisfied": True},
            exact_state_id="STATE-TEST",
            video_profile_hash="a" * 64,
            full_round=True,
            quality_trace={"playingQualityEvidence": {
                "receipt": {"status": "complete", "sha256": "c" * 64}
            }},
            quality_policy=ScenarioPlayingQualityPolicyV1.build({**_quality_policy("exact"),
                "schemaVersion": 2, "visibility": visibility}),
            rules_hash=rules.sha256, rules=rules,
            quality_settings=settings,
        )
    )
    assert not eligible  # No featured-action proof was supplied in this focused fixture.
    assert len(calls) == 1
    assert "all_decision_quality_evidence_unavailable" not in blockers
    assert ("interestingness_early_featured_decision_unavailable" in blockers) is interest
    assert ("interestingness_meaningful_decisions_insufficient" in blockers) is interest


def test_missing_authority_preserves_frozen_legacy_backend() -> None:
    authority = ScenarioResearchAuthorityV1.from_challenge_profile({})
    assert authority.backend == LEGACY_SCENARIO_RESEARCH_BACKEND
    assert authority.is_ortools is False
    assert authority.to_dict() == {
        "schemaVersion": 1,
        "backend": LEGACY_SCENARIO_RESEARCH_BACKEND,
        "sha256": authority.sha256,
    }


def test_explicit_authority_is_closed_deterministic_and_hashable() -> None:
    section = _ortools_section()
    authority = ScenarioResearchAuthorityV1.from_challenge_profile(
        {"scenarioResearchV4": section}
    )
    reordered = ScenarioResearchAuthorityV1.from_challenge_profile(
        {"scenarioResearchV4": dict(reversed(tuple(section.items())))}
    )
    assert authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND
    assert authority.is_ortools is True
    assert authority.sha256 == reordered.sha256
    assert authority.policy_catalog["defaultPolicyId"] == "fixed_action_map_v1"
    assert authority.to_dict()["proofBudgets"] == {"schemaVersion": 1}


def test_quality_execution_play_index_world_caps_are_frozen_and_optional():
    from baloot_platform.config import validate_profile_payload
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1
    raw = {"schemaVersion": 1, "maximumWallSeconds": 600,
           "maximumEvidenceBytes": 30_000_000, "maximumEscalationDecisions": 20}
    legacy = ScenarioQualityExecutionBudgetV1.build(raw)
    assert legacy.sha256 == canonical_sha256({**raw, "maximumWallSeconds": 600.0})
    assert "playIndexWorldCaps" not in legacy.to_dict()
    bands = {"schemaVersion": 1, "bands": [
        {"firstPlayIndex": 16, "lastPlayIndex": 23, "maximumHiddenWorlds": 2048},
        {"firstPlayIndex": 24, "lastPlayIndex": 31, "maximumHiddenWorlds": 256}]}
    budget = ScenarioQualityExecutionBudgetV1.build({**raw, "playIndexWorldCaps": bands})
    assert budget.sha256 != legacy.sha256
    assert ScenarioQualityExecutionBudgetV1.build(budget.to_dict()) == budget
    reordered = {**bands, "bands": list(reversed(bands["bands"]))}
    assert ScenarioQualityExecutionBudgetV1.build({**raw, "playIndexWorldCaps": reordered}) == budget
    for index in (16, 23):
        assert budget.resolve_hidden_world_cap(4096, play_index=index) == 2048
    for index in (24, 31):
        assert budget.resolve_hidden_world_cap(4096, play_index=index) == 256
    for index in (None, 0, 15):
        assert budget.resolve_hidden_world_cap(4096, play_index=index) == 4096
    assert budget.resolve_hidden_world_cap(8, play_index=24) == 8
    global_cap = ScenarioQualityExecutionBudgetV1.build({**raw, "playIndexWorldCaps": bands,
        "maximumHiddenWorldsPerDecision": 64})
    assert global_cap.resolve_hidden_world_cap(4096, play_index=23) == 64
    profile = yaml.safe_load((Path(__file__).resolve().parents[1] /
        "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
    profile["scenarioResearchV4"]["qualityExecution"] = {
        **raw, "maximumHiddenWorldsPerDecision": 2048, "playIndexWorldCaps": bands}
    validate_profile_payload("challenge", profile)
    authority = ScenarioResearchAuthorityV1.from_challenge_profile(profile)
    assert authority.quality_execution.resolve_hidden_world_cap(4096, play_index=23) == 2048
    assert authority.quality_execution.resolve_hidden_world_cap(4096, play_index=24) == 256
    assert ScenarioResearchAuthorityV1.from_challenge_profile(profile).sha256 == authority.sha256
    bands["bands"][0]["maximumHiddenWorlds"] = 1
    assert budget.resolve_hidden_world_cap(4096, play_index=23) == 2048


@pytest.mark.parametrize("invalid", [None, {}, {"schemaVersion": True, "bands": []},
    {"schemaVersion": 1, "bands": []}, {"schemaVersion": 1, "bands": [], "unknown": 1},
    *[{"schemaVersion": 1, "bands": [band]} for band in [
        {"firstPlayIndex": True, "lastPlayIndex": 23, "maximumHiddenWorlds": 2048},
        {"firstPlayIndex": 24, "lastPlayIndex": 23, "maximumHiddenWorlds": 2048},
        {"firstPlayIndex": 24, "lastPlayIndex": 32, "maximumHiddenWorlds": 2048},
        {"firstPlayIndex": 24, "lastPlayIndex": 31, "maximumHiddenWorlds": 0},
        {"firstPlayIndex": 24, "lastPlayIndex": 31, "maximumHiddenWorlds": True},
        {"firstPlayIndex": 24, "lastPlayIndex": 31, "maximumHiddenWorlds": 256, "extra": 1}]],
    {"schemaVersion": 1, "bands": [
        {"firstPlayIndex": 16, "lastPlayIndex": 24, "maximumHiddenWorlds": 2048},
        {"firstPlayIndex": 24, "lastPlayIndex": 31, "maximumHiddenWorlds": 256}]}])
def test_quality_execution_play_index_world_caps_fail_closed(invalid):
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1
    with pytest.raises((ValueError, TypeError)):
        ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1, "maximumWallSeconds": 600,
            "maximumEvidenceBytes": 30_000_000, "maximumEscalationDecisions": 20,
            "playIndexWorldCaps": invalid})


@pytest.mark.parametrize("native_policy", [None, "observer_safe_partner_feed_established_control_v1"])
def test_quality_execution_budget_is_frozen_optional_authority(native_policy):
    from baloot_platform.config import validate_profile_payload
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1

    profile = _authored_v3_profile()
    original = ScenarioResearchAuthorityV1.from_challenge_profile(profile)
    assert original.quality_execution is None and "qualityExecution" not in original.to_dict()
    profile = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
    raw = {"schemaVersion": 1, "maximumWallSeconds": 600,
           "maximumEvidenceBytes": 30_000_000, "maximumEscalationDecisions": 20,
           "maximumHiddenWorldsPerDecision": 8}
    if native_policy is not None:
        raw["bidIncumbentProof"] = {"schemaVersion": 2, "maximumSeconds": 20,
            "maximumQueries": 64, "nativePlayPolicyId": native_policy,
            "observedProjectPolicyId": "known_project_max_value_v1"}
    budget = ScenarioQualityExecutionBudgetV1.build(raw)
    assert budget.maximum_hidden_worlds_per_decision == 8
    assert ScenarioQualityExecutionBudgetV1.build(budget.to_dict()) == budget
    profile["scenarioResearchV4"]["qualityExecution"] = raw
    validate_profile_payload("challenge", profile)
    authority = ScenarioResearchAuthorityV1.from_challenge_profile(profile)
    assert authority.quality_execution == budget and authority.sha256 != original.sha256
    assert authority.to_dict()["qualityExecution"] == budget.to_dict()
    profile["scenarioResearchV4"]["qualityExecution"]["maximumWallSeconds"] = 900
    assert authority.quality_execution.maximum_wall_seconds == 600
    assert ScenarioResearchAuthorityV1.from_challenge_profile(profile).sha256 != authority.sha256
    legacy = _authored_profile()
    legacy["scenarioResearchV4"]["qualityExecution"] = raw
    for build in (lambda: validate_profile_payload("challenge", legacy),
                  lambda: ScenarioResearchAuthorityV1.from_challenge_profile(legacy)):
        with pytest.raises(ValueError):
            build()


def test_quality_execution_freezes_optional_early_play_policy():
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1

    budget = ScenarioQualityExecutionBudgetV1.build({
        "schemaVersion": 1,
        "maximumWallSeconds": 600,
        "maximumEvidenceBytes": 30_000_000,
        "maximumEscalationDecisions": 20,
        "earlyPlayPolicyId": "observer_safe_trick_control_v1",
        "exactPlayFromTrick": 7,
    })
    assert budget.early_play_policy_id == "observer_safe_trick_control_v1"
    assert budget.exact_play_from_trick == 7
    assert ScenarioQualityExecutionBudgetV1.build(budget.to_dict()) == budget
    tactical = ScenarioQualityExecutionBudgetV1.build({
        **{key: value for key, value in budget.to_dict().items() if key != "sha256"},
        "earlyPlayPolicyId": "observer_safe_tactical_category_v1",
    })
    assert tactical.early_play_policy_id == "observer_safe_tactical_category_v1"
    from_first = ScenarioQualityExecutionBudgetV1.build({
        **{key: value for key, value in budget.to_dict().items()
           if key not in {"sha256", "earlyPlayPolicyId"}},
        "exactPlayFromTrick": 1,
    })
    assert from_first.early_play_policy_id is None
    assert ScenarioQualityExecutionBudgetV1.build(from_first.to_dict()) == from_first
    legacy_null = {
        **{key: value for key, value in from_first.to_dict().items()
           if key not in {"sha256", "exactPlayFromTrick"}},
        "earlyPlayPolicyId": None, "exactPlayFromTrick": None,
    }
    assert ScenarioQualityExecutionBudgetV1.build(legacy_null).exact_play_from_trick is None
    from baloot_platform.config import validate_profile_payload
    profile = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
    profile["scenarioResearchV4"]["qualityExecution"].pop("earlyPlayPolicyId", None)
    profile["scenarioResearchV4"]["qualityExecution"]["exactPlayFromTrick"] = 1
    validate_profile_payload("challenge", profile)
    assert ScenarioResearchAuthorityV1.from_challenge_profile(profile).quality_execution.exact_play_from_trick == 1

    for change in (
        {"earlyPlayPolicyId": "unknown"},
        {"exactPlayFromTrick": 0},
        {"exactPlayFromTrick": 1},
        {"earlyPlayPolicyId": None},
    ):
        raw = budget.to_dict()
        raw.pop("sha256")
        raw.update(change)
        with pytest.raises(ValueError):
            ScenarioQualityExecutionBudgetV1.build(raw)


@pytest.mark.parametrize("play_index,expected_mode", [(23, "bounded"), (24, "exact"), (31, "exact")])
def test_quality_exact_tail_request_preserves_earlier_and_historical_evidence(play_index, expected_mode):
    from baloot_platform.research.scenario_contracts import ScenarioPlayingQualityPolicyV1, ScenarioQualityExecutionBudgetV1

    raw = {"schemaVersion": 1, "maximumWallSeconds": 60,
           "maximumEvidenceBytes": 30_000_000, "maximumEscalationDecisions": 20,
           "earlyPlayPolicyId": "observer_safe_trick_control_v1", "exactPlayFromTrick": 7}
    historical = ScenarioQualityExecutionBudgetV1.build(raw)
    enabled = ScenarioQualityExecutionBudgetV1.build({**raw, "exactTailProof": True})
    assert enabled.sha256 != historical.sha256
    assert ScenarioQualityExecutionBudgetV1.build(enabled.to_dict()) == enabled
    assert ScenarioQualityExecutionBudgetV1.build({**raw, "exactTailProof": False}) == historical
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2,
        "mode": "budgeted", "visibility": "actor_information_set",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "bounded_v1",
            "horizon": {"unit": "plies", "value": 1}, "boundaryEvaluator": "current_raw_point_margin_v1"}
            for kind in ("bid", "escalation", "project", "play")}})
    original = policy.request_for("play", "west")
    assert policy.request_for("play", "west", play_index=play_index, execution_budget=historical) == original
    selected = policy.request_for("play", "west", play_index=play_index, execution_budget=enabled)
    assert selected.mode == expected_mode
    assert selected.visibility == original.visibility and selected.target_team == original.target_team
    assert selected.utility == original.utility and selected.budget_profile == original.budget_profile
    assert (selected.horizon is None) == (expected_mode == "exact")
    assert (selected.boundary_evaluator is None) == (expected_mode == "exact")
    for kind in ("bid", "escalation", "project"):
        assert policy.request_for(kind, "west", execution_budget=enabled) == policy.request_for(kind, "west")
    for invalid in (None, True, -1, 32):
        with pytest.raises(ValueError):
            policy.request_for("play", "west", play_index=invalid, execution_budget=enabled)
    for invalid in (1, "true", None):
        with pytest.raises(ValueError, match="exactTailProof must be boolean"):
            ScenarioQualityExecutionBudgetV1.build({**raw, "exactTailProof": invalid})
    with pytest.raises(ValueError, match="declared late boundary"):
        ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1, "maximumWallSeconds": 60,
            "maximumEvidenceBytes": 1000, "maximumEscalationDecisions": 20, "exactTailProof": True})


def test_quality_exact_tail_uses_explicit_mixed_stage_budgets():
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1

    execution = ScenarioQualityExecutionBudgetV1.build({
        "schemaVersion": 1, "maximumWallSeconds": 1800,
        "maximumEvidenceBytes": 30_000_000, "maximumEscalationDecisions": 20,
        "earlyPlayPolicyId": "observer_safe_trick_control_v1",
        "exactPlayFromTrick": 7, "exactTailProof": True,
    })
    policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 2, "mode": "budgeted", "visibility": "actor_information_set",
        "decisions": {kind: {
            "utility": "awarded_point_margin",
            "budgetProfile": "actor_information_set_exact_v1" if kind == "play" else "bounded_v1",
            "horizon": {"unit": "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1",
        } for kind in SCENARIO_DECISION_KINDS},
    })
    for index, mode in ((23, "bounded"), (24, "exact"), (31, "exact")):
        request = policy.request_for("play", "west", play_index=index, execution_budget=execution)
        assert request.mode == mode
        assert request.budget_profile == "actor_information_set_exact_v1"
        assert (request.horizon is None) == (mode == "exact")
    for kind in ("bid", "escalation", "project"):
        request = policy.request_for(kind, "west", execution_budget=execution)
        assert request.mode == "bounded"
        assert request.budget_profile == "bounded_v1"
    assert ScenarioPlayingQualityPolicyV1.build(policy.to_dict()) == policy


def test_quality_exact_tail_budget_override_preserves_prefix_and_historical_hashes(root):
    from baloot_platform.research.ortools_research_executor import _quality_budget_snapshots
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1
    raw = {"schemaVersion": 1, "maximumWallSeconds": 1800,
        "maximumEvidenceBytes": 30_000_000, "maximumEscalationDecisions": 20,
        "earlyPlayPolicyId": "observer_safe_trick_control_v1",
        "exactPlayFromTrick": 7, "exactTailProof": True}
    historical = ScenarioQualityExecutionBudgetV1.build(raw)
    revised = ScenarioQualityExecutionBudgetV1.build({**raw,
        "exactTailBudgetProfile": "actor_information_set_exact_v1"})
    assert "exactTailBudgetProfile" not in historical.to_dict()
    assert revised.sha256 != historical.sha256
    assert ScenarioQualityExecutionBudgetV1.build(revised.to_dict()) == revised
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2, "mode": "budgeted",
        "visibility": "actor_information_set", "decisions": {kind: {
            "utility": "awarded_point_margin", "budgetProfile": "bounded_v1",
            "horizon": {"unit": "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1"} for kind in SCENARIO_DECISION_KINDS}})
    for kind in ("bid", "escalation", "project", "play"):
        kwargs = {"play_index": 23} if kind == "play" else {}
        assert policy.request_for(kind, "west", execution_budget=revised, **kwargs) == policy.request_for(
            kind, "west", execution_budget=historical, **kwargs)
    assert policy.request_for("play", "west", play_index=24,
        execution_budget=historical).budget_profile == "bounded_v1"
    assert policy.request_for("play", "west", play_index=24,
        execution_budget=revised).budget_profile == "actor_information_set_exact_v1"
    profiles = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]["proofBudgets"]["profiles"]
    budgets, snapshots = _quality_budget_snapshots({"bounded_v1"}, profiles, execution_budget=revised)
    assert set(snapshots) == {"bounded_v1", "actor_information_set_exact_v1"}
    assert budgets["bounded_v1"].maximum_wall_seconds == 20
    assert budgets["actor_information_set_exact_v1"].maximum_wall_seconds == 120
    for missing in (None, {"bounded_v1": profiles["bounded_v1"]}):
        with pytest.raises(ValueError, match="proof budget profile"):
            _quality_budget_snapshots({"bounded_v1"}, missing, execution_budget=revised)
    for invalid in (None, "", " actor_information_set_exact_v1", True):
        with pytest.raises(ValueError, match="exactTailBudgetProfile"):
            ScenarioQualityExecutionBudgetV1.build({**raw, "exactTailBudgetProfile": invalid})
    with pytest.raises(ValueError, match="exactTailBudgetProfile"):
        ScenarioQualityExecutionBudgetV1.build({**raw, "exactTailProof": False,
            "exactTailBudgetProfile": "actor_information_set_exact_v1"})
    with pytest.raises(ValueError, match="checksum"):
        ScenarioQualityExecutionBudgetV1.build({**revised.to_dict(), "exactTailBudgetProfile": "bounded_v1"})


def test_quality_execution_freezes_featured_native_play_proof():
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1

    raw = {"schemaVersion": 1, "maximumWallSeconds": 600,
           "maximumEvidenceBytes": 30_000_000, "maximumEscalationDecisions": 20}
    legacy = ScenarioQualityExecutionBudgetV1.build(raw)
    assert "featuredPlayNativeProof" not in legacy.to_dict()
    enabled = ScenarioQualityExecutionBudgetV1.build({**raw, "featuredPlayNativeProof": True})
    assert enabled.featured_play_native_proof is True
    assert ScenarioQualityExecutionBudgetV1.build(enabled.to_dict()) == enabled
    assert enabled.sha256 != legacy.sha256
    with pytest.raises(ValueError, match="featuredPlayNativeProof"):
        ScenarioQualityExecutionBudgetV1.build({**raw, "featuredPlayNativeProof": 1})


@pytest.mark.parametrize("field,value", [
    ("schemaVersion", True), ("unknown", 1), ("maximumWallSeconds", True),
    ("maximumWallSeconds", "600"), ("maximumWallSeconds", 0),
    ("maximumWallSeconds", float("inf")), ("maximumWallSeconds", float("nan")),
    ("maximumEvidenceBytes", False), ("maximumEvidenceBytes", 1.5),
    ("maximumEvidenceBytes", 0), ("maximumEvidenceBytes", 2**31),
    ("maximumEscalationDecisions", "20"), ("maximumEscalationDecisions", 0),
    ("maximumHiddenWorldsPerDecision", True), ("maximumHiddenWorldsPerDecision", 0),
    ("sha256", "0" * 64),
])
def test_quality_execution_budget_rejects_invalid_limits(field, value):
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1

    raw = {"schemaVersion": 1, "maximumWallSeconds": 600,
           "maximumEvidenceBytes": 30_000_000, "maximumEscalationDecisions": 20}
    raw[field] = value
    with pytest.raises(ValueError):
        ScenarioQualityExecutionBudgetV1.build(raw)


@pytest.mark.parametrize("order", [None, [], ["north"] * 4, ["north", "west", "south", 1], "north,west,south,east"])
def test_quality_execution_rejects_invalid_opportunity_order(order):
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1
    with pytest.raises(ValueError, match="escalationOpportunityOrder"):
        ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1, "maximumWallSeconds": 60,
            "maximumEvidenceBytes": 1000, "maximumEscalationDecisions": 20,
            "escalationOpportunityOrder": order})


def test_quality_round_uses_frozen_opportunity_order(monkeypatch):
    import baloot_platform.research.scenario_backend as module
    from baloot_platform.config import validate_profile_payload
    profile = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
    order = ["west", "east", "north", "south"]
    profile["scenarioResearchV4"]["qualityExecution"]["escalationOpportunityOrder"] = order
    validate_profile_payload("challenge", profile)
    backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile, bundle_identity=_bundle_identity())
    budget = backend.authority.quality_execution
    assert type(budget).build(budget.to_dict()) == budget
    order.reverse()
    assert budget.escalation_opportunity_order == ("west", "east", "north", "south")
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 1, "mode": "exact",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "late_game_exact_v1"}
                      for kind in SCENARIO_DECISION_KINDS}})
    def capture(*args, **kwargs):
        assert kwargs["opportunity_order"] == budget.escalation_opportunity_order
        raise RuntimeError("verified frozen dispatch")
    monkeypatch.setattr(module, "resolve_quality_round_continuation_v1", capture)
    with pytest.raises(RuntimeError, match="verified frozen dispatch"):
        backend.execute_quality_round(None, None, {}, policy)
    with pytest.raises(ValueError, match="disagrees with frozen"):
        backend.execute_quality_round(None, None, {}, policy, opportunity_order=order)


@pytest.mark.real_clock
def test_quality_round_dispatch_uses_only_frozen_profile_budgets(monkeypatch):
    import baloot_platform.research.scenario_backend as module
    from types import SimpleNamespace

    profile = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
    profile["scenarioResearchV4"]["qualityExecution"].pop("escalationOpportunityOrder", None)
    backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile, bundle_identity=_bundle_identity())
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 1, "mode": "exact",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "late_game_exact_v1"}
                      for kind in SCENARIO_DECISION_KINDS}})
    expected = copy.deepcopy(profile["scenarioResearchV4"]["proofBudgets"]["profiles"]["late_game_exact_v1"])
    expected_search = copy.deepcopy(profile["scenarioResearchV4"]["searchBudgets"]["profiles"])
    profile["scenarioResearchV4"]["qualityExecution"]["maximumWallSeconds"] = 1
    profile["scenarioResearchV4"]["proofBudgets"]["profiles"].clear()
    calls = []
    def execute(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["proof_budget_profiles"] == {"late_game_exact_v1": expected}
        assert kwargs["maximum_nodes"] == expected["maximumNodes"]
        assert kwargs["maximum_depth"] == expected["maximumDepth"]
        assert kwargs["maximum_worlds"] == expected["maximumHiddenWorlds"]
        assert kwargs["maximum_decision_seconds"] == expected["maximumWallSeconds"]
        assert kwargs["maximum_evidence_bytes"] == 100_000_000
        assert kwargs["execution_budget"].maximum_wall_seconds == 600
        return {"status": "incomplete", "sha256": canonical_sha256({"status": "incomplete"})}
    monkeypatch.setattr(module, "resolve_quality_round_continuation_v1", execute)
    validations = []
    frozen_rules = SimpleNamespace(sha256="a" * 64)
    def validate(result, **kwargs):
        validations.append(kwargs)
        assert result["status"] == "incomplete"
        assert kwargs == dict(source_prefix=None, rules_hash="a" * 64, policy=policy,
            decision_point={}, rules=frozen_rules, search_budget_profiles=expected_search)
        assert kwargs["rules"] is frozen_rules
    monkeypatch.setattr(module, "validate_native_round_quality_evidence_v1", validate)
    for seconds in (None, 15):
        result = backend.execute_quality_round(frozen_rules, None, {}, policy,
            opportunity_order=("north", "west", "south", "east"), maximum_seconds=seconds)
        assert result["status"] == "incomplete"
        assert result["configurationAuthority"] == _bundle_identity()
        assert result["backendIdentity"] == backend.identity.to_dict()
        assert result["sourceQualityHash"] == canonical_sha256({"status": "incomplete"})
        assert result["sha256"] == canonical_sha256({key: value for key, value in result.items() if key != "sha256"})
        assert calls[-1]["maximum_seconds"] == (600 if seconds is None else seconds)
    assert len(validations) == 2
    def reject(*args, **kwargs):
        raise ValueError("invalid composed receipt")
    monkeypatch.setattr(module, "validate_native_round_quality_evidence_v1", reject)
    with pytest.raises(ValueError, match="invalid composed receipt"):
        backend.execute_quality_round(SimpleNamespace(sha256="a" * 64), None, {}, policy,
                                      opportunity_order=())
    monkeypatch.setattr(module, "resolve_quality_round_continuation_v1", lambda *args, **kwargs: {"status": "complete", "sha256": "0" * 64})
    with pytest.raises(ValueError, match="producer checksum"):
        backend.execute_quality_round(None, None, {}, policy, opportunity_order=())
    old = OrToolsScenarioResearchBackendV2(challenge_profile=_authored_v3_profile(), bundle_identity=_bundle_identity())
    with pytest.raises(ValueError, match="frozen native"):
        old.execute_quality_round(None, None, {}, policy, opportunity_order=())


def test_authored_ortools_profile_resolves_the_same_closed_authority() -> None:
    profile = _authored_profile()
    authority = ScenarioResearchAuthorityV1.from_challenge_profile(profile)

    assert authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND
    assert authority.planner == "tail_first_v1"
    assert authority.proof_budgets["version"] == "2.0.0"
    assert tuple(authority.search_budgets["profiles"]) == (
        "interactive_validation_v1",
        "bounded_v1",
        "late_game_exact_v1",
        "actor_information_set_exact_v1",
        "offline_heavyweight_v1",
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (lambda value: value.update({"unknown": True}), "shape mismatch"),
        (lambda value: value.pop("proofBudgets"), "shape mismatch"),
        (lambda value: value.update({"schemaVersion": 2}), "schemaVersion"),
        (lambda value: value.update({"backend": "other"}), "backend"),
        (lambda value: value.update({"planner": "other"}), "planner"),
        (
            lambda value: value.update({"deterministicExecution": False}),
            "deterministic execution",
        ),
        (lambda value: value.update({"searchBudgets": []}), "searchBudgets"),
    ),
)
def test_explicit_authority_fails_closed(mutation, error: str) -> None:
    section = copy.deepcopy(_ortools_section())
    mutation(section)
    with pytest.raises((TypeError, ValueError), match=error):
        ScenarioResearchAuthorityV1.from_challenge_profile(
            {"scenarioResearchV4": section}
        )


@pytest.mark.parametrize("native_delayed", [False, True])
def test_native_executor_receives_the_selected_frozen_detector_profile(root, native_delayed):
    from baloot_platform.tactics import TacticalDetectorProfileV1
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v2.yaml')
    authored = _authored_v3_profile()
    if native_delayed:
        authored['tactical_detectors']['nativeDelayedPayoff'] = {'gainMetric': 'raw_card_points', 'minimumGain': 8}
    expected = TacticalDetectorProfileV1.build(authored['tactical_detectors'])
    backend = OrToolsScenarioResearchBackendV2(challenge_profile=authored, bundle_identity=_bundle_identity())
    authored['tactical_detectors']['lowCardMaxStrength'] = 0
    executor = backend._new_executor(rules, None)
    assert executor.tactical_detector_profile == expected
    assert executor.compiler.tactical_detector_profile == expected
    assert executor.lowering_manifest.manifest_hash == backend.lowering_manifest.manifest_hash


@pytest.mark.parametrize("payoff", [None, {"gainMetric": "raw_card_points", "minimumGain": 8}])
def test_direct_legacy_backend_cannot_ignore_native_delayed_settings(payoff):
    profile = _authored_profile()
    profile["tactical_detectors"]["nativeDelayedPayoff"] = payoff
    with pytest.raises(ValueError, match="nativeDelayedPayoff requires the native V3"):
        OrToolsScenarioResearchBackendV2(challenge_profile=profile, bundle_identity=_bundle_identity())


def test_backend_identity_and_v2_catalog_are_bundle_bound_and_immutable() -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )

    assert backend.identity.to_dict()["backendId"] == ORTOOLS_SCENARIO_RESEARCH_BACKEND
    assert backend.identity.to_dict()["qualificationAuthority"] == "cp_sat"
    assert backend.identity.bundle_revision_id == "CBR-TEST-ORTOOLS-V2"
    with pytest.raises(FrozenInstanceError):
        backend.identity.planner = "changed"  # type: ignore[misc]

    catalog = backend.catalog(
        origin="search",
        proof_mode="exact",
        visibility="complete_state",
    )
    kinds = {definition["kind"] for definition in catalog["definitions"]}
    assert len(kinds) == 78
    assert "deal.card_owner" in kinds
    assert "outcome.buyer_contract" in kinds
    assert "state.observer_visibility" in kinds
    assert backend.identity.predicate_catalog_hash == (
        "d36815fde96af7246bfb67153f75dd0cc7962fc5b18827a3e6582b764a9a600d"
    )
    executor_identity = backend.executor_identity
    assert executor_identity["backendIdentityHash"] == backend.identity.sha256
    assert executor_identity["loweringManifestHash"] == (
        backend.identity.predicate_catalog_hash
    )
    assert executor_identity["sha256"]


def test_v3_catalog_exposes_only_causally_supported_native_tactics(root: Path) -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(),
        bundle_identity=_bundle_identity(),
        rules=RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml"),
    )
    assert backend.identity.backend_id == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
    catalog = backend.catalog(
        origin="search", proof_mode="exact", visibility="complete_state"
    )
    tactic = next(
        item for item in catalog["definitions"] if item["kind"] == "tactic.verified"
    )
    assert tactic["argumentSchema"]["properties"]["kind"]["enum"] == [
        "contract_rescue",
        "debate",
        "entry_unblocking",
        "kabout_creation",
        "kabout_preservation",
        "kabout_prevention",
        "large_card_sacrifice",
        "only_winning_move",
        "partner_enablement",
        "project_bidding",
        "small_card_big_gain",
        "trump_control",
    ]


def test_v3_backend_enforces_tactic_decision_and_utility_prerequisites() -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(),
        bundle_identity=_bundle_identity(),
    )

    project_bid = backend.validate(
        _tactic_recipe(
            "project_bidding",
            decision_kind="bid",
            utility="awarded_point_margin",
        )
    )
    project_play = backend.validate(
        _tactic_recipe(
            "project_bidding",
            decision_kind="play",
            utility="awarded_point_margin",
        )
    )
    contract_wrong = backend.validate(
        _tactic_recipe(
            "contract_rescue",
            decision_kind="play",
            utility="round_win",
        )
    )
    contract_right = backend.validate(
        _tactic_recipe(
            "contract_rescue",
            decision_kind="play",
            utility="contract_success",
        )
    )

    assert "tactic_decision_kind_unsupported" in {
        item["code"] for item in project_bid["issues"]
    }
    assert "tactic_decision_kind_unsupported" in {
        item["code"] for item in project_play["issues"]
    }
    assert "tactic_proof_utility_unsupported" in {
        item["code"] for item in contract_wrong["issues"]
    }
    assert "tactic_proof_utility_unsupported" not in {
        item["code"] for item in contract_right["issues"]
    }


@pytest.mark.parametrize("origin", ("search", "exact_trace_prefix"))
@pytest.mark.parametrize("mode,visibility,supported", (
    ("exact", "complete_state", True),
    ("bounded", "complete_state", False),
    ("policy_conditioned", "complete_state", False),
    ("exact", "actor_information_set", False),
))
def test_v3_project_tactic_admission_is_complete_state_only(origin, mode, visibility, supported):
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(), bundle_identity=_bundle_identity(),
    )
    catalog = backend.catalog(origin=origin, proof_mode=mode, visibility=visibility)
    for definition in catalog["definitions"]:
        if definition["kind"] in {"tactic.applicable", "tactic.verified"}:
            assert ("project_bidding" in definition["argumentSchema"]["properties"]["kind"]["enum"]) is supported
    recipe = _tactic_recipe("project_bidding", decision_kind="project", utility="awarded_point_margin")
    recipe["scenario"]["origin"] = ({"kind": origin} if origin == "search"
                                    else {"kind": origin, "exactTracePrefix": {}})
    recipe["proof"].update(mode=mode, visibility=visibility)
    issues = backend.validate(recipe)["issues"]
    tactic_context_issues = {
        item["code"] for item in issues
        if item["path"] == "scenario.constraints"
        and item["code"] in {"predicate_v3_context_unsupported", "tactic_native_producer_unavailable",
                             "tactic_decision_kind_unsupported", "tactic_proof_utility_unsupported"}
    }
    assert bool(tactic_context_issues) is not supported
    recipe["proof"]["utility"] = "round_win"
    assert "tactic_proof_utility_unsupported" in {
        item["code"] for item in backend.validate(recipe)["issues"]
    }


@pytest.mark.parametrize(
    ("selected", "universe", "expected"),
    [
        (None, "all_legal", "continuation_count_requires_selected_action"),
        ("HA", "optimal_strategy", "continuation_count_line_universe_mismatch"),
        ("HA", "all_legal", None),
    ],
)
def test_v3_backend_validates_hard_line_count_request_binding(
    selected, universe, expected
) -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe(policy=ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3)
    recipe["scenario"]["constraints"] = {
        "predicateId": "PRED-LINE-COUNT",
        "kind": "proof.winning_continuation_line_count",
        "version": 1,
        "hardness": "hard",
        "arguments": {"lineUniverse": "all_legal", "minimum": "0", "maximum": "2"},
    }
    recipe["proof"].update({
        "claim": "winning_continuation_line_count",
        "lineUniverse": universe,
        **({"selectedAction": selected} if selected is not None else {}),
    })
    validation = backend.validate(recipe)
    codes = {issue["code"] for issue in validation["issues"]}
    if expected is None:
        assert not codes & {
            "continuation_count_requires_selected_action",
            "continuation_count_line_universe_mismatch",
        }
    else:
        assert expected in codes
        assert backend.estimate(recipe)["launchAdmissible"] is False


@pytest.mark.parametrize("tactic", ["large_card_sacrifice", "small_card_big_gain"])
@pytest.mark.parametrize("operator", ["leaf", "all", "not", "any", "count"])
def test_v3_rejects_unavailable_hard_causality_in_every_expression(tactic, operator):
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(), bundle_identity=_bundle_identity(),
    )
    recipe = _tactic_recipe(tactic, decision_kind="play", utility="awarded_point_margin")
    predicate = recipe["scenario"]["constraints"]
    predicate["kind"] = "tactic.verified"
    predicate["arguments"] = {
        "kind": tactic, "causalRequired": True, "minimumScore": 0, "minimumEvidence": 0,
    }
    if operator != "leaf":
        recipe["scenario"]["constraints"] = {
            "op": operator, "children": [predicate, *([{
                "predicateId": "PRED-CONTRACT", "kind": "auction.contract", "version": 1,
                "hardness": "hard", "arguments": {"contract": "sun"},
            }] if operator == "any" else [])],
            **({"minimum": 0, "maximum": 1} if operator == "count" else {}),
        }
    issues = backend.validate(recipe)["issues"]
    causal = [issue for issue in issues if issue["code"] == "tactic_causal_verification_unavailable"]
    assert len(causal) == 1
    assert causal[0]["path"].endswith(".arguments.causalRequired")
    assert causal[0]["predicateIds"] == [predicate["predicateId"]]
    assert backend.estimate(recipe)["launchAdmissible"] is False


@pytest.mark.parametrize("tactic", ["small_card_big_gain", "large_card_sacrifice"])
@pytest.mark.parametrize("origin", ["search", "exact_trace_prefix"])
@pytest.mark.parametrize("scope,issue", [
    ({}, None), ({"utility": "round_win"}, "tactic_proof_utility_unsupported"),
    ({"mode": "bounded"}, "predicate_v3_context_unsupported"),
    ({"visibility": "actor_information_set"}, "predicate_v3_context_unsupported"),
])
def test_native_delayed_profile_admission_is_bundle_bound(tactic, origin, scope, issue):
    profile = _authored_v3_profile()
    frozen = OrToolsScenarioResearchBackendV2(challenge_profile=profile, bundle_identity=_bundle_identity())
    profile["tactical_detectors"]["nativeDelayedPayoff"] = {
        "gainMetric": "raw_card_points", "minimumGain": 8,
    }
    backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile, bundle_identity=_bundle_identity())
    recipe = _tactic_recipe(tactic, decision_kind="play", utility="awarded_point_margin")
    recipe["scenario"]["origin"] = {"kind": origin}
    recipe["scenario"]["constraints"].update(kind="tactic.verified", arguments={
        "kind": tactic, "minimumScore": 100, "minimumEvidence": 1, "causalRequired": True,
    })
    recipe["proof"].update(scope)
    codes = {row["code"] for row in backend.validate(recipe)["issues"]}
    assert "tactic_causal_verification_unavailable" not in codes
    if issue is not None:
        assert issue in codes
    else:
        assert not codes & {"tactic_proof_utility_unsupported", "predicate_v3_context_unsupported"}
    assert "tactic_causal_verification_unavailable" in {
        row["code"] for row in frozen.validate(recipe)["issues"]}
    assert backend.identity.predicate_catalog_hash != frozen.identity.predicate_catalog_hash


@pytest.mark.parametrize(("kind", "hardness", "causal"), [
    ("tactic.verified", "hard", None), ("tactic.verified", "hard", False),
    ("tactic.verified", "prefer", True), ("tactic.verified", "report", True),
    ("tactic.applicable", "hard", None),
])
def test_v3_causal_admission_preserves_nonqualifying_requests(kind, hardness, causal):
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(), bundle_identity=_bundle_identity(),
    )
    recipe = _tactic_recipe("small_card_big_gain", decision_kind="play", utility="awarded_point_margin")
    predicate = recipe["scenario"]["constraints"]
    predicate.update(kind=kind, hardness=hardness)
    if kind == "tactic.verified":
        predicate["arguments"] = {"kind": "small_card_big_gain", "minimumScore": 0, "minimumEvidence": 0, **(
            {"causalRequired": causal} if causal is not None else {}
        )}
    codes = {issue["code"] for issue in backend.validate(recipe)["issues"]}
    assert not codes & {"predicate_arguments_invalid", "tactic_causal_verification_unavailable"}


def test_causal_admission_preserves_frozen_v2_requests():
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(), bundle_identity=_bundle_identity(),
    )
    recipe = _tactic_recipe("small_card_big_gain", decision_kind="play", utility="awarded_point_margin")
    recipe["resources"]["generationPolicy"] = ORTOOLS_SCENARIO_GENERATION_POLICY
    recipe["scenario"]["constraints"].update(kind="tactic.verified", arguments={
        "kind": "small_card_big_gain", "minimumScore": 0, "minimumEvidence": 0, "causalRequired": True,
    })
    codes = {issue["code"] for issue in backend.validate(recipe)["issues"]}
    assert not codes & {"predicate_arguments_invalid", "tactic_causal_verification_unavailable"}
    recipe["scenario"]["constraints"]["arguments"]["minimumScore"] = "invalid"
    invalid = backend.validate(recipe)
    assert any(issue["code"] == "predicate_arguments_invalid" for issue in invalid["issues"])
    assert all("predicateIds" not in issue for issue in invalid["issues"])


@pytest.mark.parametrize(
    "tactic",
    (
        "entry_unblocking",
        "partner_enablement",
        "trump_control",
        "small_card_big_gain",
        "large_card_sacrifice",
    ),
)
def test_v3_backend_rejects_pv_detectors_outside_play(tactic: str) -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(),
        bundle_identity=_bundle_identity(),
    )
    result = backend.validate(
        _tactic_recipe(
            tactic,
            decision_kind="bid",
            utility="awarded_point_margin",
        )
    )
    assert "tactic_decision_kind_unsupported" in {
        item["code"] for item in result["issues"]
    }


@pytest.mark.parametrize(
    "tactic",
    ("kabout_creation", "kabout_preservation", "kabout_prevention"),
)
def test_v3_backend_requires_kabout_utility_for_kabout_detectors(
    tactic: str,
) -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(),
        bundle_identity=_bundle_identity(),
    )
    wrong = backend.validate(
        _tactic_recipe(tactic, decision_kind="play", utility="round_win")
    )
    right = backend.validate(
        _tactic_recipe(tactic, decision_kind="play", utility="kabout")
    )
    assert "tactic_proof_utility_unsupported" in {
        item["code"] for item in wrong["issues"]
    }
    assert "tactic_proof_utility_unsupported" not in {
        item["code"] for item in right["issues"]
    }


def test_v3_validation_intersects_predicates_with_decision_context() -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe(policy=ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3)
    recipe["scenario"]["constraints"] = {
        "predicateId": "PRED-WINNING-ROOT-ACTION",
        "kind": "proof.winning_root_action_matches",
        "version": 1,
        "hardness": "hard",
        "arguments": {
            "selector": {"ranks": ["ace"]},
        },
    }
    recipe["scenario"]["decisionPoint"] = {
        "kind": "bid",
        "selector": {"actionIndex": 0},
    }

    rejected = backend.validate(recipe)
    assert any(
        issue["code"] == "predicate_v3_context_unsupported"
        for issue in rejected["issues"]
    )

    recipe["scenario"]["decisionPoint"] = {
        "kind": "play",
        "selector": {
            "trickIndex": 5,
            "position": 3,
            "remainingCards": 3,
            "actingSeat": "west",
        },
    }
    accepted = backend.validate(recipe)
    assert not any(
        issue["code"] == "predicate_v3_context_unsupported"
        for issue in accepted["issues"]
    )


@pytest.mark.parametrize('origin', ('search', 'exact_trace_prefix'))
@pytest.mark.parametrize('kind,arguments', (
    ('proof.winning_root_action_matches', {'selector': {'ranks': ['ace']}}),
    ('proof.winning_root_card_rank_count', {'rank': ['ace'], 'minimum': 1, 'maximum': 1}),
))
def test_v3_exact_actor_card_root_predicates_reach_catalog_and_validation(origin, kind, arguments):
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(), bundle_identity=_bundle_identity())
    recipe = _tactic_recipe('only_winning_move', decision_kind='play', utility='round_win')
    recipe['scenario']['origin'] = ({'kind': origin} if origin == 'search'
        else {'kind': origin, 'exactTracePrefix': {}})
    recipe['scenario']['constraints'] = {'predicateId': 'PRED-ACTOR-CARD-ROOT',
        'kind': kind, 'version': 1, 'hardness': 'hard', 'arguments': arguments}
    recipe['proof'].update(mode='exact', visibility='actor_information_set')
    assert not any(issue['code'] in {'predicate_v3_context_unsupported',
        'predicate_reference_world_only', 'predicate_cp_sat_unsupported'}
        for issue in backend.validate(recipe)['issues'])
    catalog = backend.catalog(origin=origin, proof_mode='exact', visibility='actor_information_set')
    definition = next(row for row in catalog['definitions'] if row['kind'] == kind)
    assert definition['cpSatSupport']['actorInformationSet'] == 'nonanticipative'
    recipe['scenario']['decisionPoint'] = {'kind': 'bid', 'selector': {'actionIndex': 0}}
    assert any(issue['code'] == 'predicate_v3_context_unsupported'
        for issue in backend.validate(recipe)['issues'])


@pytest.mark.parametrize('origin', ('search', 'exact_trace_prefix'))
@pytest.mark.parametrize('kind,arguments', (
    ('proof.principal_variation_trick_winner', {'trickIndex': 7, 'team': 'west_east', 'rank': ['nine']}),
    ('proof.principal_variation_trick_team_rank_count', {
        'trickIndex': 7, 'team': 'west_east', 'rank': ['nine'], 'minimum': 1, 'maximum': 1}),
))
def test_v3_missing_actor_evidence_is_rejected_before_execution(origin, kind, arguments):
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(), bundle_identity=_bundle_identity())
    recipe = _tactic_recipe('only_winning_move', decision_kind='play', utility='round_win')
    recipe['scenario']['origin'] = ({'kind': origin} if origin == 'search'
        else {'kind': origin, 'exactTracePrefix': {}})
    recipe['scenario']['constraints'] = {'predicateId': 'PRED-MISSING-ACTOR-PRODUCER',
        'kind': kind, 'version': 1, 'hardness': 'hard', 'arguments': arguments}
    recipe['proof'].update(mode='exact', visibility='actor_information_set')
    rejected = backend.validate(recipe)
    assert any(issue['code'] == 'predicate_v3_context_unsupported'
        and issue['path'] == 'scenario.constraints' for issue in rejected['issues'])
    catalog = backend.catalog(origin=origin, proof_mode='exact', visibility='actor_information_set')
    assert kind not in {definition['kind'] for definition in catalog['definitions']}
    recipe['proof']['visibility'] = 'complete_state'
    assert not any(issue['code'] == 'predicate_v3_context_unsupported'
        for issue in backend.validate(recipe)['issues'])


@pytest.mark.parametrize('origin', ('search', 'exact_trace_prefix'))
@pytest.mark.parametrize('predicate_kind', ('tactic.applicable', 'tactic.verified'))
@pytest.mark.parametrize('tactic_kind', ('only_winning_move', 'debate', 'contract_rescue',
    'kabout_creation', 'kabout_preservation', 'kabout_prevention', 'small_card_big_gain'))
def test_v3_actor_tactic_contexts_expose_only_owned_root_producer(origin, predicate_kind, tactic_kind):
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(), bundle_identity=_bundle_identity())
    recipe = _tactic_recipe(tactic_kind, decision_kind='play',
        utility='kabout' if tactic_kind.startswith('kabout_') else
        'contract_success' if tactic_kind == 'contract_rescue' else 'round_win')
    recipe['scenario']['origin'] = ({'kind': origin} if origin == 'search'
        else {'kind': origin, 'exactTracePrefix': {}})
    arguments = ({'kind': tactic_kind, 'applicable': True}
        if predicate_kind == 'tactic.applicable' else {
            'kind': tactic_kind, 'minimumScore': 0, 'minimumEvidence': 0})
    recipe['scenario']['constraints'] = {'predicateId': 'PRED-ACTOR-TACTIC',
        'kind': predicate_kind, 'version': 1, 'hardness': 'hard', 'arguments': arguments}
    recipe['proof'].update(mode='exact', visibility='actor_information_set')
    assert not any(issue['code'] == 'predicate_v3_context_unsupported'
                   for issue in backend.validate(recipe)['issues'])
    catalog = backend.catalog(origin=origin, proof_mode='exact', visibility='actor_information_set')
    definition = next(row for row in catalog['definitions'] if row['kind'] == predicate_kind)
    assert definition['argumentSchema']['properties']['kind']['enum'] == [
        'contract_rescue', 'debate', 'kabout_creation', 'kabout_preservation', 'kabout_prevention', 'only_winning_move',
        'small_card_big_gain']
    if tactic_kind == 'contract_rescue' or tactic_kind.startswith('kabout_'):
        recipe['proof']['utility'] = 'round_win'
        assert any(issue['code'] == 'tactic_proof_utility_unsupported'
                   for issue in backend.validate(recipe)['issues'])
        recipe['proof']['utility'] = 'kabout' if tactic_kind.startswith('kabout_') else 'contract_success'
    original_utility = recipe['proof']['utility']
    for unsupported in ('entry_unblocking',):
        arguments['kind'] = unsupported
        recipe['proof']['utility'] = 'awarded_point_margin'
        assert any(issue['code'] == 'predicate_v3_context_unsupported'
                   and issue['path'] == 'scenario.constraints'
                   for issue in backend.validate(recipe)['issues'])
    arguments['kind'] = tactic_kind
    recipe['proof']['utility'] = original_utility
    recipe['scenario']['decisionPoint'] = {'kind': 'bid', 'selector': {'actionIndex': 0}}
    expected = ('tactic_decision_kind_unsupported' if tactic_kind == 'small_card_big_gain'
                else 'predicate_v3_context_unsupported')
    assert any(issue['code'] == expected
               for issue in backend.validate(recipe)['issues'])


@pytest.mark.parametrize('origin', ('search', 'exact_trace_prefix'))
def test_v3_exact_actor_observer_scope_is_available_in_play_catalog(origin):
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(), bundle_identity=_bundle_identity())
    recipe = _tactic_recipe('only_winning_move', decision_kind='play', utility='round_win')
    recipe['scenario']['origin'] = ({'kind': origin} if origin == 'search'
        else {'kind': origin, 'exactTracePrefix': {}})
    recipe['scenario']['constraints'] = {'predicateId': 'PRED-ACTOR-SCOPE',
        'kind': 'state.observer_visibility', 'version': 1, 'hardness': 'hard',
        'arguments': {'observer': 'west', 'classification': 'legal_deduction'}}
    recipe['proof'].update(mode='exact', visibility='actor_information_set')
    assert not any(issue['code'] == 'predicate_v3_context_unsupported'
        for issue in backend.validate(recipe)['issues'])
    catalog = backend.catalog(origin=origin, proof_mode='exact', visibility='actor_information_set')
    assert 'state.observer_visibility' in {definition['kind'] for definition in catalog['definitions']}


def test_preflight_rejects_legacy_policy_and_never_claims_execution_ready() -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    valid = backend.preflight(_recipe())
    rejected = backend.preflight(_recipe(policy="scenario_guided_actor_safe"))

    assert valid["valid"] is True
    assert valid["executionReady"] is False
    assert valid["compiledStages"] == ["deal"]
    assert rejected["valid"] is False
    assert "legacy_generation_policy_forbidden" in {
        issue["code"] for issue in rejected["issues"]
    }


def test_preflight_rejects_reference_world_only_hard_actor_information_predicate() -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    definition = backend._by_kind["deal.card_owner"]
    backend._by_kind = {
        **backend._by_kind,
        definition.kind: replace(
            definition,
            cp_sat_support=replace(
                definition.cp_sat_support,
                actor_information_set="reference_world_only",
            ),
        ),
    }
    recipe = _recipe()
    recipe["scenario"].update(
        {
            "origin": {"kind": "exact_trace_prefix", "exactTracePrefix": {}},
            "decisionPoint": {
                "kind": "play",
                "selector": {"actingSeat": "north"},
            },
        }
    )
    recipe["proof"].update(
        {"mode": "exact", "visibility": "actor_information_set"}
    )

    result = backend.preflight(recipe)

    assert result["valid"] is False
    assert "predicate_reference_world_only" in {
        issue["code"] for issue in result["issues"]
    }


def test_preflight_checks_each_hard_predicate_requested_proof_mode_support() -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    definition = backend._by_kind["deal.card_owner"]
    backend._by_kind = {
        **backend._by_kind,
        definition.kind: replace(
            definition,
            cp_sat_support=replace(
                definition.cp_sat_support,
                exact_proof="unsupported",
            ),
        ),
    }

    result = backend.preflight(_recipe())

    assert result["valid"] is False
    assert "predicate_proof_mode_unsupported" in {
        issue["code"] for issue in result["issues"]
    }


def test_backend_campaign_delegates_only_to_the_frozen_unified_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    class UnifiedExecutor:
        def __init__(self, rules, *, bundle_revision_id: str, bundle_hash: str, runtime_proposals=None) -> None:
            assert runtime_proposals is None
            observed["init"] = (rules, bundle_revision_id, bundle_hash)

        def execute_campaign(self, scenario, proof, **parameters):
            observed["execute"] = (scenario, proof, parameters)
            return "campaign"

    monkeypatch.setattr(
        "baloot_platform.research.scenario_backend.OrToolsUnifiedResearchExecutorV1",
        UnifiedExecutor,
    )
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    rules = object()

    result = backend.execute_campaign(
        rules,
        "scenario",
        "proof",
        result_count=5,
        maximum_seconds=60.0,
    )

    assert result == "campaign"
    assert observed["init"] == (
        rules,
        "CBR-TEST-ORTOOLS-V2",
        "a" * 64,
    )
    assert observed["execute"] == (
        "scenario",
        "proof",
        {"result_count": 5, "maximum_seconds": 60.0},
    )


def test_campaign_quality_settings_come_from_frozen_bundle(monkeypatch, root):
    observed = {}
    executors = []
    class Executor:
        def __init__(self, rules, **parameters):
            observed.update(parameters)
            self.parameters = parameters
            executors.append(self)
        def construct_qualified_auction(self, *args, **parameters):
            return 'auction'
        def execute_campaign(self, *args, **parameters):
            assert "playing_quality" not in parameters
            return "campaign"
    monkeypatch.setattr("baloot_platform.research.scenario_backend.OrToolsUnifiedResearchExecutorV1", Executor)
    profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
    backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile, bundle_identity=_bundle_identity())
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 1, "mode": "exact", "decisions": {
        kind: {"utility": "awarded_point_margin", "budgetProfile": "late_game_exact_v1"}
        for kind in SCENARIO_DECISION_KINDS}})
    cached_backend = backend
    backend = cached_backend.for_run()
    other_run = cached_backend.for_run()
    assert backend.identity == cached_backend.identity
    assert backend._bid_quality_cache is not cached_backend._bid_quality_cache
    assert backend._bid_quality_cache is not other_run._bid_quality_cache
    provider = backend.qualified_auction_provider(None, policy)
    assert provider.__self__ is executors[0]
    assert provider(None, maximum_seconds=1) == 'auction'
    assert backend.execute_campaign(None, None, None, playing_quality=policy) == "campaign"
    assert executors[0].parameters == executors[1].parameters
    cache = observed['bid_quality_cache']
    assert cache is executors[0].parameters['bid_quality_cache'] is backend._bid_quality_cache
    assert cache.maximum_entries == backend.authority.cache_limits['maximumEntries']
    assert cache.maximum_bytes == backend.authority.cache_limits['maximumBytes']
    assert observed["quality_policy"].to_dict() == policy.to_dict()
    assert observed["quality_execution"] == backend.authority.quality_execution
    assert observed["quality_proof_profiles"] == profile["scenarioResearchV4"]["proofBudgets"]["profiles"]
    assert observed["quality_search_profiles"] == profile["scenarioResearchV4"]["searchBudgets"]["profiles"]
    from baloot_platform.research.ortools_research_executor import OrToolsUnifiedResearchExecutorV1
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    executor = OrToolsUnifiedResearchExecutorV1(rules, **observed)
    selected = "late_game_exact_v1"
    conflicts = executor.quality_search_profiles[selected]["maximumConflicts"]
    observed["quality_search_profiles"][selected]["maximumConflicts"] = True
    assert executor.quality_search_profiles[selected]["maximumConflicts"] == conflicts
    with pytest.raises(ValueError, match="matching frozen search profiles"):
        OrToolsUnifiedResearchExecutorV1(rules, **observed)
    with pytest.raises(TypeError, match="typed policy"):
        backend.execute_campaign(None, None, None, playing_quality=policy.to_dict())
    legacy = OrToolsScenarioResearchBackendV2(challenge_profile=_authored_profile(), bundle_identity=_bundle_identity())
    with pytest.raises(ValueError, match="frozen native"):
        legacy.execute_campaign(None, None, None, playing_quality=policy)


def test_policy_conditioned_preflight_requires_bundle_governed_templates() -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe()
    recipe["scenario"]["decisionPoint"] = {
        "kind": "play",
        "selector": {"trickIndex": 6},
    }
    recipe["proof"].update(
        {
            "mode": "policy_conditioned",
            "policies": _policy_templates(),
            "lineUniverse": "policy_conditioned",
        }
    )

    accepted = backend.preflight(recipe)
    assert accepted["valid"] is True
    assert not {
        item["code"] for item in accepted["issues"]
    } & {"policy_profile_invalid", "policy_identity_unsupported"}

    recipe["proof"]["policies"]["west"] = {
        "agentId": "heuristic_bot_v1",
        "version": "1.0.0",
        "seed": 2,
        "profileHash": "0" * 64,
    }
    rejected = backend.preflight(recipe)
    assert rejected["valid"] is False
    assert {item["code"] for item in rejected["issues"]} == {
        "policy_profile_invalid"
    }


def test_policy_execution_support_exposes_all_complete_state_graph_claims() -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe()
    recipe["scenario"].update(
        {
            "origin": {"kind": "exact_trace_prefix", "exactTracePrefix": {}},
            "decisionPoint": {"kind": "play", "selector": {}},
        }
    )
    recipe["proof"].update(
        {
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "winning_continuation_line_count",
            "utility": "round_win",
            "lineUniverse": "policy_conditioned",
            "policies": _policy_templates(),
        }
    )
    recipe["resources"]["proofBudgetProfile"] = "late_game_exact_v1"

    for claim in (
        "outcome_reachable",
        "outcome_forced",
        "winning_root_action_count",
        "winning_continuation_line_count",
        "selected_action_is_unique_winner",
        "target_team_has_no_salvage",
    ):
        selected = copy.deepcopy(recipe)
        selected["proof"]["claim"] = claim
        if claim != "winning_continuation_line_count":
            selected["proof"].pop("lineUniverse")
        assert backend.execution_support(selected)["executionReady"] is True

    supported_search = copy.deepcopy(recipe)
    supported_search["scenario"]["origin"] = {"kind": "search"}
    supported_search["scenario"]["decisionPoint"]["selector"] = {
        "trickIndex": 6
    }
    assert backend.execution_support(supported_search)["executionReady"] is True


@pytest.mark.parametrize("origin", ("search", "exact_trace_prefix"))
@pytest.mark.parametrize("mode", ("exact", "bounded"))
def test_root_action_count_is_admitted_for_complete_state_play_only(
    origin: str,
    mode: str,
) -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe()
    recipe["scenario"].update(
        {
            "origin": {"kind": origin},
            "decisionPoint": {
                "kind": "play",
                "selector": {"trickIndex": 6},
            },
        }
    )
    recipe["proof"].update(
        {
            "mode": mode,
            "visibility": "complete_state",
            "claim": "winning_root_action_count",
            "utility": "round_win",
        }
    )
    recipe["resources"]["proofBudgetProfile"] = "late_game_exact_v1"

    assert backend.execution_support(recipe)["executionReady"] is True

    actor_information = copy.deepcopy(recipe)
    actor_information["scenario"]["origin"] = {
        "kind": "exact_trace_prefix",
        "exactTracePrefix": {},
    }
    actor_information["proof"]["mode"] = "exact"
    actor_information["proof"]["visibility"] = "actor_information_set"
    actor_support = backend.execution_support(actor_information)
    assert actor_support["executionReady"] is True
    assert actor_support["selectionConstraints"] == {
        "contractValues": ["sun", "hokum"]
    }

    nonplay = copy.deepcopy(recipe)
    nonplay["scenario"]["origin"] = {
        "kind": "exact_trace_prefix",
        "exactTracePrefix": {},
    }
    nonplay["scenario"]["decisionPoint"] = {
        "kind": "project",
        "selector": {"owner": "south"},
    }
    nonplay["proof"].update(
        {
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "policies": _policy_templates(),
        }
    )
    assert "proof_claim_not_executable" in backend.execution_support(nonplay)[
        "reasons"
    ]


def test_bounded_kabout_requires_round_end_boundary() -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe()
    recipe["scenario"]["decisionPoint"] = {
        "kind": "play",
        "selector": {"trickIndex": 6},
    }
    recipe["proof"].update(
        {
            "mode": "bounded",
            "visibility": "complete_state",
            "claim": "outcome_forced",
            "utility": "kabout",
            "horizon": {"value": 2, "unit": "tricks"},
        }
    )

    assert "bounded_kabout_requires_round_end" in backend.execution_support(recipe)[
        "reasons"
    ]
    recipe["proof"]["horizon"] = {"value": 1, "unit": "round_end"}
    assert "bounded_kabout_requires_round_end" not in backend.execution_support(
        recipe
    )["reasons"]


def test_preflight_fails_closed_for_unsupported_visibility_combination() -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe()
    recipe["proof"].update(
        {"mode": "bounded", "visibility": "actor_information_set"}
    )
    result = backend.validate(recipe)

    assert result["valid"] is False
    assert result["issues"][0]["code"] == "unsupported_research_combination"


def test_registered_observer_visibility_can_be_requested_as_hard() -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe()
    recipe["scenario"]["constraints"] = {
        "predicateId": "PRED-OBSERVER-HARD",
        "kind": "state.observer_visibility",
        "version": 1,
        "hardness": "hard",
        "arguments": {
            "observer": "west",
            "classification": "public_fact",
        },
    }

    result = backend.validate(recipe)

    assert result["valid"] is True
    assert result["issues"] == []
    assert result["compiledStages"] == ["state"]


def test_estimate_uses_only_the_supplied_frozen_profile() -> None:
    profile = _authored_profile()
    profile["scenarioResearchV4"]["proofBudgets"]["profiles"][
        "late_game_exact_v1"
    ]["maximumNodes"] = 123_456_789
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=profile,
        bundle_identity=_bundle_identity(),
    )

    estimate = backend.estimate(_recipe())

    assert estimate["proofBudget"]["maximumNodes"] == 123_456_789
    assert estimate["estimateStatus"] == "refused"
    assert estimate["launchAdmissible"] is False
    assert estimate["refusalReason"] == "search_decision_selector_required"
    assert estimate["estimatedModelVariables"] is None
    assert estimate["estimatedModelConstraints"] is None
    assert estimate["estimatedGraphNodes"] is None
    assert estimate["estimatedGraphEdges"] is None
    assert estimate["estimatedHiddenWorlds"] is None
    assert estimate["estimatedCountLimbs"] is None
    assert estimate["resourceTier"] == "late_game_exact_v1"


def test_default_execution_slice_supports_exact_play() -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe()
    recipe["scenario"].update(
        {
            "origin": {"kind": "exact_trace_prefix", "exactTracePrefix": {}},
            "decisionPoint": {"kind": "play", "selector": {}},
        }
    )
    recipe["proof"].update(
        {
            "claim": "outcome_reachable",
            "utility": "round_win",
            "lineUniverse": "all_legal",
        }
    )
    recipe["resources"]["proofBudgetProfile"] = "late_game_exact_v1"

    support = backend.execution_support(recipe)
    estimate = backend.estimate(recipe)

    assert support["executionReady"] is True
    assert support["reasons"] == []
    assert support["executorIdentity"] == backend.executor_identity
    assert estimate["launchAdmissible"] is True
    assert estimate["estimateStatus"] == "admissible"
    search = _recipe()
    assert backend.execution_support(search)["reasons"][0] == (
        "search_decision_selector_required"
    )
    search["scenario"]["decisionPoint"] = {
        "kind": "play",
        "selector": {"trickIndex": 5},
    }
    search["proof"].update(
        {
            "claim": "outcome_reachable",
            "utility": "round_win",
            "lineUniverse": "all_legal",
        }
    )
    search["resources"]["proofBudgetProfile"] = "late_game_exact_v1"

    assert backend.execution_support(search)["executionReady"] is True
    assert backend.estimate(search)["launchAdmissible"] is True
    search["scenario"]["decisionPoint"]["selector"]["remainingCards"] = 3
    assert backend.execution_support(search)["executionReady"] is True
    assert backend.estimate(search)["launchAdmissible"] is True


def test_exact_actor_information_play_is_executable_but_line_count_is_not() -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe()
    recipe["scenario"].update(
        {
            "origin": {"kind": "exact_trace_prefix", "exactTracePrefix": {}},
            "decisionPoint": {"kind": "play", "selector": {}},
        }
    )
    recipe["proof"].update(
        {
            "mode": "exact",
            "visibility": "actor_information_set",
            "strategyClass": "pure_deterministic",
            "claim": "outcome_reachable",
            "utility": "round_win",
            "lineUniverse": None,
            "budgetProfile": "actor_information_set_exact_v1",
        }
    )
    recipe["resources"]["proofBudgetProfile"] = (
        "actor_information_set_exact_v1"
    )

    exact_support = backend.execution_support(recipe)
    assert exact_support["executionReady"] is True
    assert exact_support["selectionConstraints"] == {
        "contractValues": ["sun", "hokum"]
    }

    recipe["proof"].update(
        {
            "claim": "winning_continuation_line_count",
            "lineUniverse": "all_legal",
        }
    )
    support = backend.execution_support(recipe)
    assert support["executionReady"] is False
    assert "line_universe_not_executable" in support["reasons"]

    recipe["proof"].update(
        {"claim": "outcome_reachable", "lineUniverse": None}
    )
    recipe["scenario"]["origin"] = {"kind": "search"}
    recipe["scenario"]["decisionPoint"]["selector"] = {
        "trickIndex": 7,
        "actingSeat": "west",
        "remainingCards": 1,
    }
    support = backend.execution_support(recipe)
    assert support["executionReady"] is True, support["reasons"]


def test_bid_policy_proof_requires_a_typed_decision_ordinal() -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe()
    recipe["scenario"].update(
        {
            "origin": {"kind": "exact_trace_prefix", "exactTracePrefix": {}},
            "decisionPoint": {"kind": "bid", "selector": {}},
        }
    )
    recipe["proof"].update(
        {
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "utility": "round_win",
            "lineUniverse": "policy_conditioned",
            "policies": _policy_templates(),
        }
    )
    recipe["resources"]["proofBudgetProfile"] = "late_game_exact_v1"

    support = backend.execution_support(recipe)
    estimate = backend.estimate(recipe)

    assert support["executionReady"] is False
    assert "nonplay_terminal_continuation_unavailable" in support["reasons"]
    assert estimate["launchAdmissible"] is False
    assert estimate["refusalReason"] == (
        "nonplay_terminal_continuation_unavailable"
    )


@pytest.mark.parametrize(
    ("decision_kind", "selector"),
    (
        ("bid", {"actionIndex": 0}),
        ("escalation", {"actingSeat": "west", "actionIndex": 0}),
    ),
)
def test_exact_nonplay_policy_proof_is_executable(
    decision_kind: str,
    selector: dict,
) -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe()
    recipe["scenario"].update(
        {
            "origin": {"kind": "exact_trace_prefix", "exactTracePrefix": {}},
            "decisionPoint": {"kind": decision_kind, "selector": selector},
        }
    )
    recipe["proof"].update(
        {
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "utility": "round_win",
            "lineUniverse": "policy_conditioned",
            "policies": _policy_templates(),
        }
    )
    recipe["resources"]["proofBudgetProfile"] = "late_game_exact_v1"

    support = backend.execution_support(recipe)
    estimate = backend.estimate(recipe)

    assert support["executionReady"] is True, support["reasons"]
    assert estimate["launchAdmissible"] is True
    assert estimate["refusalReason"] is None


@pytest.mark.parametrize(
    ("decision_kind", "selector"),
    (
        ("bid", {"actionIndex": 0}),
        ("escalation", {"actingSeat": "west", "actionIndex": 0}),
    ),
)
@pytest.mark.parametrize("mode", ("exact", "bounded"))
@pytest.mark.parametrize("origin", ("search", "exact_trace_prefix"))
def test_nonplay_complete_state_graph_and_line_count_are_executable(
    decision_kind: str,
    selector: dict,
    mode: str,
    origin: str,
) -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe()
    recipe["scenario"].update(
        {
            "origin": (
                {"kind": "search"}
                if origin == "search"
                else {"kind": "exact_trace_prefix", "exactTracePrefix": {}}
            ),
            "decisionPoint": {"kind": decision_kind, "selector": selector},
        }
    )
    recipe["proof"].update(
        {
            "mode": mode,
            "visibility": "complete_state",
            "strategyClass": "pure_deterministic",
            "claim": "winning_root_action_count",
            "utility": "round_win",
            "lineUniverse": None,
        }
    )
    recipe["resources"]["proofBudgetProfile"] = "late_game_exact_v1"
    if mode == "bounded":
        recipe["proof"].update(
            {
                "horizon": {"unit": "plies", "value": 2},
                "boundaryEvaluator": "current_raw_point_margin_v1",
            }
        )

    support = backend.execution_support(recipe)

    assert support["executionReady"] is True, support["reasons"]

    recipe["proof"].update(
        {
            "claim": "winning_continuation_line_count",
            "lineUniverse": "all_legal",
        }
    )
    counted = backend.execution_support(recipe)
    assert counted["executionReady"] is True, counted["reasons"]


@pytest.mark.parametrize(
    ("decision_kind", "selector"),
    (
        ("bid", {"actionIndex": 0}),
        ("escalation", {"actingSeat": "west", "actionIndex": 0}),
    ),
)
@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize("unit", ("tricks", "round_end"))
def test_nonplay_bounded_play_horizons_require_native_complete_state(
    decision_kind: str,
    selector: dict,
    native: bool,
    unit: str,
) -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile() if native else _authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe()
    recipe["scenario"].update(
        {
            "origin": {"kind": "exact_trace_prefix", "exactTracePrefix": {}},
            "decisionPoint": {"kind": decision_kind, "selector": selector},
        }
    )
    recipe["proof"].update(
        {
            "mode": "bounded",
            "visibility": "complete_state",
            "strategyClass": "pure_deterministic",
            "claim": "outcome_reachable",
            "utility": "round_win",
            "horizon": {"unit": unit, "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1",
        }
    )
    recipe["resources"]["proofBudgetProfile"] = "late_game_exact_v1"

    support = backend.execution_support(recipe)
    assert ("nonplay_bounded_horizon_unit_unsupported" in support["reasons"]) is not native
    recipe["proof"]["visibility"] = "actor_information_set"
    refused = backend.execution_support(recipe)
    assert "nonplay_bounded_horizon_unit_unsupported" in refused["reasons"]
    recipe["proof"]["visibility"] = "complete_state"
    recipe["proof"]["horizon"] = {"unit": "plies", "value": 1}
    supported = backend.execution_support(recipe)
    assert "nonplay_bounded_horizon_unit_unsupported" not in supported["reasons"]


@pytest.mark.parametrize(
    ("decision_kind", "selector"),
    (
        ("bid", {"actionIndex": 0}),
        ("escalation", {"actingSeat": "west", "actionIndex": 0}),
    ),
)
def test_search_nonplay_policy_proof_is_executable(
    decision_kind: str,
    selector: dict,
) -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe()
    recipe["scenario"].update(
        {
            "origin": {"kind": "search"},
            "decisionPoint": {"kind": decision_kind, "selector": selector},
        }
    )
    recipe["proof"].update(
        {
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "utility": "round_win",
            "lineUniverse": "policy_conditioned",
            "policies": _policy_templates(),
        }
    )
    recipe["resources"]["proofBudgetProfile"] = "late_game_exact_v1"

    support = backend.execution_support(recipe)
    estimate = backend.estimate(recipe)

    assert support["executionReady"] is True, support["reasons"]
    assert estimate["launchAdmissible"] is True
    assert estimate["refusalReason"] is None


def test_project_strategic_root_is_hidden_without_optional_declaration_rules() -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe()
    recipe["scenario"].update(
        {
            "origin": {"kind": "exact_trace_prefix", "exactTracePrefix": {}},
            "decisionPoint": {
                "kind": "project",
                "selector": {"owner": "north", "projectIndex": 0},
            },
        }
    )
    recipe["proof"].update(
        {
            "mode": "exact",
            "visibility": "complete_state",
            "strategyClass": "pure_deterministic",
            "claim": "outcome_reachable",
            "utility": "round_win",
        }
    )
    recipe["resources"]["proofBudgetProfile"] = "late_game_exact_v1"

    support = backend.execution_support(recipe)
    estimate = backend.estimate(recipe)
    assert support["executionReady"] is False
    assert "project_declaration_mode_missing_or_mandatory" in support["reasons"]
    assert estimate["launchAdmissible"] is False
    assert estimate["refusalReason"] == (
        "project_declaration_mode_missing_or_mandatory"
    )


@pytest.mark.parametrize("mode", ("exact", "bounded", "policy_conditioned"))
def test_optional_project_final_declaration_root_requires_full_service_evidence(
    root: Path,
    mode: str,
) -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(),
        bundle_identity=_bundle_identity(),
        rules=RuleBook.load(
            root / "configs/rules/saudi_project_optional_unratified_v1.yaml"
        ),
    )
    recipe = _recipe(policy=ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3)
    recipe["scenario"].update(
        {
            "origin": {"kind": "exact_trace_prefix", "exactTracePrefix": {}},
            "decisionPoint": {
                "kind": "project",
                "selector": {"owner": "north", "projectIndex": 0},
            },
        }
    )
    recipe["proof"].update(
        {
            "mode": mode,
            "visibility": "complete_state",
            "strategyClass": (
                "frozen_policy"
                if mode == "policy_conditioned"
                else "pure_deterministic"
            ),
            "claim": "outcome_reachable",
            "utility": "round_win",
            "lineUniverse": (
                "policy_conditioned"
                if mode == "policy_conditioned"
                else "all_legal"
            ),
        }
    )
    if mode == "bounded":
        recipe["proof"].update(
            {
                "horizon": {"unit": "plies", "value": 1},
                "boundaryEvaluator": "current_raw_point_margin_v1",
            }
        )
    if mode == "policy_conditioned":
        recipe["proof"]["policies"] = _policy_templates()
    recipe["resources"].update(
        {
            "generationPolicy": ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3,
            "proofBudgetProfile": "late_game_exact_v1",
        }
    )

    support = backend.execution_support(recipe)

    if mode in {"bounded", "policy_conditioned"}:
        assert support["executionReady"] is True, support["reasons"]
        assert support["selectionConstraints"]
    else:
        assert support["executionReady"] is False
        assert "phase4_substantive_full_service_evidence_missing" in support["reasons"]
        assert support["selectionConstraints"] == {}


def test_optional_project_search_root_requires_full_service_evidence(
    root: Path,
) -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_v3_profile(),
        bundle_identity=_bundle_identity(),
        rules=RuleBook.load(
            root / "configs/rules/saudi_project_optional_unratified_v1.yaml"
        ),
    )
    recipe = _recipe(policy=ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3)
    recipe["scenario"]["decisionPoint"] = {
        "kind": "project",
        "selector": {"owner": "north", "projectIndex": 0},
    }
    recipe["proof"].update(
        {
            "mode": "exact",
            "visibility": "complete_state",
            "strategyClass": "pure_deterministic",
            "claim": "outcome_reachable",
            "utility": "round_win",
            "lineUniverse": "all_legal",
        }
    )
    recipe["resources"].update(
        {
            "generationPolicy": ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3,
            "proofBudgetProfile": "late_game_exact_v1",
        }
    )

    support = backend.execution_support(recipe)

    assert support["executionReady"] is False
    assert "phase4_substantive_full_service_evidence_missing" in support["reasons"]
    assert support["selectionConstraints"] == {}

    capabilities = build_scenario_execution_capabilities_v1(backend)
    project = [
        item
        for item in capabilities["combinations"]
        if item["decisionKind"] == "project"
    ]
    assert {
        (item["origin"], item["mode"], item["visibility"])
        for item in project
    } == {
        ("search", "bounded", "complete_state"),
        ("search", "policy_conditioned", "complete_state"),
        ("exact_trace_prefix", "bounded", "complete_state"),
        ("exact_trace_prefix", "policy_conditioned", "complete_state"),
    }
    assert "project" not in capabilities["decisionUnavailableReasons"]


def test_search_play_policy_proof_is_executable() -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe()
    recipe["scenario"].update(
        {
            "origin": {"kind": "search"},
            "decisionPoint": {
                "kind": "play",
                "selector": {
                    "trickIndex": 6,
                    "position": 0,
                    "remainingCards": 2,
                    "actingSeat": "west",
                },
            },
        }
    )
    recipe["proof"].update(
        {
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "winning_continuation_line_count",
            "utility": "round_win",
            "lineUniverse": "policy_conditioned",
            "policies": _policy_templates(),
        }
    )
    recipe["resources"]["proofBudgetProfile"] = "late_game_exact_v1"

    support = backend.execution_support(recipe)
    estimate = backend.estimate(recipe)

    assert support["executionReady"] is True, support["reasons"]
    assert estimate["launchAdmissible"] is True
    assert estimate["refusalReason"] is None


@pytest.mark.parametrize(
    ("decision_kind", "selector"),
    (
        ("bid", {}),
        ("bid", {"actionIndex": "0"}),
        ("escalation", {"actingSeat": "west"}),
        ("escalation", {"actingSeat": "west", "actionIndex": "0"}),
    ),
)
def test_search_nonplay_policy_requires_typed_selector_ordinal(
    decision_kind: str,
    selector: dict,
) -> None:
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=_authored_profile(),
        bundle_identity=_bundle_identity(),
    )
    recipe = _recipe()
    recipe["scenario"].update(
        {
            "origin": {"kind": "search"},
            "decisionPoint": {"kind": decision_kind, "selector": selector},
        }
    )
    recipe["proof"].update(
        {
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "utility": "round_win",
            "lineUniverse": "policy_conditioned",
            "policies": _policy_templates(),
        }
    )

    support = backend.execution_support(recipe)

    assert support["executionReady"] is False
    assert "search_decision_selector_required" in support["reasons"]


def test_backend_has_no_engine_bot_or_legacy_research_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    source_root = root / "src"
    pending = ["baloot_platform.research.scenario_backend"]
    visited: set[str] = set()
    modules: set[str] = set()
    sources: list[str] = []
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        path = source_root.joinpath(*module.split(".")).with_suffix(".py")
        source = path.read_text(encoding="utf-8")
        sources.append(source)
        for node in ast.walk(ast.parse(source)):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = module.split(".")[:-node.level]
                    imported.append(".".join((*base, *(node.module or "").split("."))))
                else:
                    imported.append(node.module or "")
            modules.update(imported)
            pending.extend(
                name
                for name in imported
                if name.startswith("baloot_platform.research.")
                and source_root.joinpath(*name.split(".")).with_suffix(".py").is_file()
            )
    forbidden = {
        "baloot_platform.core.engine",
        "baloot_platform.core.bots",
        "baloot_platform.research.exact_state",
        "baloot_platform.research.scenario_evidence",
        "baloot_platform.research.scenario_search",
        "baloot_platform.research.proof_service",
    }
    assert modules.isdisjoint(forbidden)
    assert all("load_default" not in source for source in sources)


def test_backend_requires_explicit_ortools_authority_and_bundle_identity() -> None:
    with pytest.raises(ValueError, match="explicit ortools_cp_sat_v2"):
        OrToolsScenarioResearchBackendV2(
            challenge_profile={},
            bundle_identity=_bundle_identity(),
        )
    with pytest.raises(ValueError, match="bundle_identity is incomplete"):
        OrToolsScenarioResearchBackendV2(
            challenge_profile=_authored_profile(),
            bundle_identity={"bundleRevisionId": "CBR-INCOMPLETE"},
        )
