from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from baloot_platform.research import BalootDomainCompilerV1
from baloot_platform.research import domain_compiler as compiler_module
from baloot_platform.research.contracts import canonical_sha256
from baloot_platform.service.container import build_container


def _source():
    container = build_container()
    return (
        container.research.domain.rules,
        container.research.domain.conventions,
        container.research.domain.localization,
        container.research.domain.bundle_hash,
        container.registry.list(),
    )


@pytest.mark.parametrize("changed_field,changed_value", [
    ("kind", "unrelated"),
    ("version", "2.0.0"),
    ("capabilities", ["different_lowering"]),
    ("deterministic", False),
])
@pytest.mark.parametrize("plugin_id", sorted({binding[0] for binding in compiler_module._CP_SAT_BINDINGS.values()}))
def test_behavioral_proof_cache_tracks_lowering_inputs_not_unrelated_plugins(
    monkeypatch: pytest.MonkeyPatch, plugin_id: str, changed_field: str, changed_value,
):
    keys: list[str] = []
    proof = {"rules.test": {"proofHash": "test-receipt"}}

    class ProofCacheProbe(dict):
        def get(self, key):
            keys.append(key)
            return proof

    monkeypatch.setattr(compiler_module, "_BEHAVIORAL_PROOF_CACHE", ProofCacheProbe())
    plugin = {
        "id": plugin_id, "kind": "domain_lowering", "version": "1.0.0",
        "capabilities": ["test_lowering"], "deterministic": True,
    }
    rules = {"schema_version": 1}
    first = BalootDomainCompilerV1._behavioral_proofs(rules=rules, plugins={plugin_id: plugin})
    first["rules.test"]["proofHash"] = "caller mutation"
    unrelated = {
        "video_renderer": {"id": "video_renderer", "version": "99.0.0"},
        plugin_id: {**plugin, "name": "display-only metadata"},
    }
    second = BalootDomainCompilerV1._behavioral_proofs(rules=rules, plugins=unrelated)
    assert second == proof
    assert keys[0] == keys[1]
    BalootDomainCompilerV1._behavioral_proofs(
        rules=rules, plugins={plugin_id: {**plugin, changed_field: changed_value}},
    )
    assert keys[-1] != keys[0]
    BalootDomainCompilerV1._behavioral_proofs(rules={"schema_version": 2}, plugins={plugin_id: plugin})
    assert keys[-1] != keys[0]
    BalootDomainCompilerV1._behavioral_proofs(rules=rules, plugins={})
    assert keys[-1] != keys[0]


@pytest.mark.parametrize("outcome", ("buyer_makes", "buyer_loses"))
def test_configured_sun_tie_leaf_has_executable_counterfactual_coverage(outcome):
    rules, conventions, localization, bundle_hash, plugins = _source()
    configured = copy.deepcopy(rules)
    configured["scoring"]["contract"]["buyer_tie_outcome"] = outcome
    spec = BalootDomainCompilerV1().compile(
        rules=configured, conventions=conventions, localization=localization,
        bundle_hash=bundle_hash, plugins=plugins,
    )
    row = next(row for row in spec.compiled.coverage.rows
               if row.primitive_id == "rules.scoring.contract.buyer_tie_outcome")
    assert row.executable_binding_status == "covered"
    assert row.behavioral_or_invariant_status == "proven_behavioral"
    assert row.qualification_eligible is True
    assert spec.compiled.target_payloads["runtimePlan"]["compiledScoring"]["buyerTieOutcome"] == outcome


def test_compiled_domain_cache_is_bounded_sanitized_and_thread_safe(
    monkeypatch: pytest.MonkeyPatch,
):
    compiler = BalootDomainCompilerV1(cache_size=2)
    uncached_calls: list[str] = []

    def fake_compile_uncached(**values):
        label = str(values["bundle_hash"])
        uncached_calls.append(label)
        return {"bundleHash": label}

    monkeypatch.setattr(compiler, "_compile_uncached", fake_compile_uncached)

    def compile_label(label: str):
        return compiler.compile(
            rules={"label": label},
            conventions={},
            localization={},
            bundle_hash=label,
        )

    first = compile_label("a")
    first["callerMutation"] = True
    assert compile_label("a") == {"bundleHash": "a"}
    assert compiler.cache_metrics_snapshot() == {
        "scope": "compiled_domain",
        "hits": 1,
        "misses": 1,
        "evictions": 0,
        "entries": 1,
        "capacity": 2,
    }

    compile_label("b")
    compile_label("c")
    assert compiler.cache_metrics_snapshot() == {
        "scope": "compiled_domain",
        "hits": 1,
        "misses": 3,
        "evictions": 1,
        "entries": 2,
        "capacity": 2,
    }
    assert uncached_calls == ["a", "b", "c"]

    parallel = BalootDomainCompilerV1(cache_size=2)
    monkeypatch.setattr(parallel, "_compile_uncached", fake_compile_uncached)
    labels = tuple("xy"[index % 2] for index in range(40))

    def compile_parallel(label: str):
        return parallel.compile(
            rules={"label": label},
            conventions={},
            localization={},
            bundle_hash=label,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(compile_parallel, labels))
    assert {item["bundleHash"] for item in results} == {"x", "y"}
    snapshot = parallel.cache_metrics_snapshot()
    assert snapshot["hits"] + snapshot["misses"] == len(labels)
    assert snapshot["entries"] == 2
    assert snapshot["evictions"] == 0
    assert tuple(snapshot) == (
        "scope",
        "hits",
        "misses",
        "evictions",
        "entries",
        "capacity",
    )


def test_domain_v3_is_deterministic_and_separates_binding_from_behavioral_proof(monkeypatch):
    rules, conventions, localization, bundle_hash, plugins = _source()
    compiler = BalootDomainCompilerV1()
    first = compiler.compile(
        rules=rules, conventions=conventions, localization=localization, bundle_hash=bundle_hash,
        plugins=plugins,
    )
    second = compiler.compile(
        rules=rules, conventions=conventions, localization=localization, bundle_hash=bundle_hash,
        plugins=plugins,
    )
    assert first == second
    assert first.sha256 == second.sha256
    lowering_ids = {binding[0] for binding in compiler_module._CP_SAT_BINDINGS.values()}
    lowering_plugins = [item for item in plugins if item["id"] in lowering_ids]

    def unexpected_project_solve(*_args, **_kwargs):
        pytest.fail("an unchanged lowering inventory must reuse the proved project lifecycle")

    with monkeypatch.context() as cached:
        cached.setattr(
            "baloot_platform.research.ortools_project_model.OrToolsProjectLifecycleModelV1.solve",
            unexpected_project_solve,
        )
        narrowed = BalootDomainCompilerV1().compile(
            rules=rules, conventions=conventions, localization=localization, bundle_hash=bundle_hash,
            plugins=lowering_plugins,
        )
    assert narrowed.compiled == first.compiled
    assert narrowed.sha256 != first.sha256
    assert narrowed.plugin_capabilities == tuple(lowering_plugins)
    assert first.plugin_capabilities == tuple(plugins)
    assert first.compiled.coverage.total > 0
    assert len({row.primitive_id for row in first.compiled.coverage.rows}) == first.compiled.coverage.total
    coverage = first.compiled.coverage
    assert coverage.executable_binding_complete is True
    assert coverage.executable_binding_covered == coverage.total
    assert coverage.behavioral_or_invariant_complete is True
    assert coverage.complete is True
    assert first.compiled.publishable is True
    assert all(row.executable_binding_status == "covered" for row in coverage.rows)
    proven = {
        row.primitive_id: row.behavioral_or_invariant_status
        for row in coverage.rows
        if row.behavioral_or_invariant_status != "unproven"
    }
    assert {
        primitive_id: status
        for primitive_id, status in proven.items()
        if primitive_id.startswith("rules.")
        and not primitive_id.startswith(
            (
                "rules.projects.",
                "rules.scoring.",
                "rules.deck.",
                "rules.seating.",
                "rules.dealing.",
                "rules.bidding.heuristics.",
            )
        )
    } == {
        "rules.bidding.ashkal.enabled": "proven_invariant",
        "rules.bidding.ashkal.eligible_priority_positions": "proven_invariant",
        "rules.bidding.ashkal.resolves_to": "proven_invariant",
        "rules.bidding.ashkal.taker": "proven_invariant",
        "rules.bidding.first_round.allowed": "proven_invariant",
        "rules.bidding.first_round.hokum_uses_purchase_suit": "proven_invariant",
        "rules.bidding.pass_allowed": "proven_invariant",
        "rules.bidding.rounds": "proven_invariant",
        "rules.bidding.second_round.allowed": "proven_invariant",
        "rules.bidding.second_round.hokum_must_differ_from_purchase_suit": "proven_invariant",
        "rules.contracts.allowed": "proven_behavioral",
        "rules.contracts.default": "proven_behavioral",
        "rules.contracts.require_preselected": "proven_behavioral",
        "rules.escalation.enabled": "proven_behavioral",
        "rules.escalation.hokum_chain": "proven_behavioral",
        "rules.escalation.sun_chain": "proven_behavioral",
        "rules.escalation.default_level": "proven_invariant",
        "rules.escalation.default_mode": "proven_behavioral",
        "rules.escalation.opening_actor_policy": "proven_invariant",
        "rules.escalation.closure_policy": "proven_invariant",
        "rules.escalation.response_actor_policy": "proven_behavioral",
        "rules.escalation.sun_score_threshold": "proven_behavioral",
        "rules.escalation.sun_double_requires_split_score": "proven_behavioral",
        "rules.disputed_options.first_leader.alternatives": "proven_invariant",
        "rules.disputed_options.double_unspecified_mode.alternatives": "proven_invariant",
        "rules.disputed_options.equal_hokum_sequence_trump_priority.alternatives": "proven_invariant",
        "rules.disputed_options.equal_hokum_sequence_trump_priority.selected": "proven_behavioral",
        "rules.disputed_options.hints_have_universal_meaning.alternatives": "proven_invariant",
        "rules.disputed_options.kawesh_saneen.alternatives": "proven_invariant",
        "rules.card_rules.hokum_non_trump_points": "proven_behavioral",
        "rules.card_rules.hokum_non_trump_strength": "proven_behavioral",
        "rules.card_rules.hokum_trump_points": "proven_behavioral",
        "rules.card_rules.hokum_trump_strength": "proven_behavioral",
        "rules.card_rules.sequence_strength": "proven_behavioral",
        "rules.card_rules.sun_points": "proven_behavioral",
        "rules.card_rules.sun_strength": "proven_behavioral",
        "rules.play.allow_discard_when_cannot_overtrump": "proven_behavioral",
        "rules.play.configured_first_leader": "proven_behavioral",
        "rules.play.ekka_enabled": "proven_invariant",
        "rules.play.first_leader": "proven_behavioral",
        "rules.play.locked_hokum_blocks_trump_lead": "proven_invariant",
        "rules.play.must_follow_suit": "proven_invariant",
        "rules.play.must_overtrump_when_possible": "proven_invariant",
        "rules.play.must_trump_when_void": "proven_invariant",
        "rules.play.partner_winning_exception": "proven_invariant",
    }
    project_proven = {
        primitive_id: status
        for primitive_id, status in proven.items()
        if primitive_id.startswith("rules.projects.")
    }
    assert len(project_proven) == 34
    assert set(project_proven) == {
        row.primitive_id
        for row in coverage.rows
        if row.primitive_id.startswith("rules.projects.")
    }
    project_invariants = {
        "rules.projects.declaration_timing",
        "rules.projects.comparison_timing",
        "rules.projects.reveal_timing",
        "rules.projects.only_winning_team_scores",
        "rules.projects.baloot.only_hokum",
        "rules.projects.baloot.declare_on_second_card",
        "rules.projects.baloot.siri_or_fifty_inclusion",
        "rules.projects.baloot.single_card_in_other_project",
    }
    assert project_proven["rules.projects.baloot.hundred_inclusion"] == (
        "proven_behavioral"
    )
    assert {
        primitive_id
        for primitive_id, status in project_proven.items()
        if status == "proven_invariant"
    } == project_invariants
    assert all(
        status == "proven_behavioral"
        for primitive_id, status in project_proven.items()
        if primitive_id not in project_invariants
    )
    structural_proven = {
        primitive_id: status
        for primitive_id, status in proven.items()
        if primitive_id.startswith(
            ("rules.deck.", "rules.seating.", "rules.dealing.")
        )
    }
    structural_behavioral = {
        "rules.dealing.allow_no_cut",
        "rules.dealing.annulling.dealer_advances",
        "rules.dealing.annulling.eligible_ranks",
        "rules.dealing.annulling.enabled",
        "rules.dealing.cut_enabled",
        "rules.dealing.initial_packets",
    }
    assert len(structural_proven) == 25
    assert set(structural_proven) == {
        row.primitive_id
        for row in coverage.rows
        if row.primitive_id.startswith(
            ("rules.deck.", "rules.seating.", "rules.dealing.")
        )
    }
    assert {
        primitive_id
        for primitive_id, status in structural_proven.items()
        if status == "proven_behavioral"
    } == structural_behavioral
    assert all(
        status == "proven_invariant"
        for primitive_id, status in structural_proven.items()
        if primitive_id not in structural_behavioral
    )
    scoring_proven = {
        primitive_id: status
        for primitive_id, status in proven.items()
        if primitive_id.startswith("rules.scoring.")
    }
    assert scoring_proven == {
        "rules.scoring.last_trick_bonus_raw": "proven_behavioral",
        "rules.scoring.raw_card_totals.sun": "proven_invariant",
        "rules.scoring.raw_card_totals.hokum": "proven_invariant",
        "rules.scoring.card_game_point_totals.sun": "proven_behavioral",
        "rules.scoring.card_game_point_totals.hokum": "proven_behavioral",
        "rules.scoring.conversion.sun": "proven_invariant",
        "rules.scoring.conversion.hokum": "proven_invariant",
        "rules.scoring.contract.buyer_team_must_win": "proven_invariant",
        "rules.scoring.contract.loss_transfers_all": "proven_invariant",
        "rules.scoring.escalation.winner_takes_all_when_multiplied": "proven_invariant",
        "rules.scoring.escalation.project_multiplier_cap": "proven_behavioral",
        "rules.scoring.escalation.baloot_multiplier_cap": "proven_behavioral",
        "rules.scoring.kabout.enabled": "proven_behavioral",
        "rules.scoring.kabout.base_game_points.sun": "proven_behavioral",
        "rules.scoring.kabout.base_game_points.hokum": "proven_behavioral",
        "rules.scoring.match.target": "proven_behavioral",
        "rules.scoring.match.tie_above_target_continues": "proven_invariant",
        "rules.scoring.match.dealer_rotates_right": "proven_invariant",
    }
    assert all(
        row.qualification_eligible
        == (
            row.behavioral_or_invariant_status
            in {"proven_behavioral", "proven_invariant"}
        )
        for row in coverage.rows
    )
    assert first.compiled.policy_plan_hash
    payloads = first.compiled.to_dict(include_payloads=True)["targetPayloads"]
    compiled_structural = [
        payloads[target]["compiledStructural"]
        for target in ("runtimePlan", "referencePlan", "cpSatBlueprint")
    ]
    assert compiled_structural[0] == compiled_structural[1] == compiled_structural[2]
    assert compiled_structural[0]["sha256"]
    compiled_projects = [
        payloads[target]["compiledProjects"]
        for target in ("runtimePlan", "referencePlan", "cpSatBlueprint")
    ]
    assert compiled_projects[0] == compiled_projects[1] == compiled_projects[2]
    assert compiled_projects[0]["sha256"]
    compiled_play = [
        payloads[target]["compiledPlay"]
        for target in ("runtimePlan", "referencePlan", "cpSatBlueprint")
    ]
    assert compiled_play[0] == compiled_play[1] == compiled_play[2]
    assert compiled_play[0]["sha256"]
    compiled_scoring = [
        payloads[target]["compiledScoring"]
        for target in ("runtimePlan", "referencePlan", "cpSatBlueprint")
    ]
    assert compiled_scoring[0] == compiled_scoring[1] == compiled_scoring[2]
    assert compiled_scoring[0]["sha256"]
    receipts = payloads["executableLowerings"]
    assert receipts
    assert all(
        receipt["verified"] is True
        for targets in receipts.values()
        for receipt in targets.values()
    )
    behavioral_proofs = payloads["behavioralProofs"]
    assert set(behavioral_proofs) == set(proven)
    assert all(
        proof["rulesHash"] == first.rules_hash
        for primitive_id, proof in behavioral_proofs.items()
        if primitive_id.startswith("rules.")
    )
    assert (
        behavioral_proofs["rules.contracts.allowed"]["proofBoundary"]
        == "exhaustive_auction_trace_set_for_counterfactual_profiles"
    )
    assert behavioral_proofs["rules.contracts.allowed"]["referenceIndependent"] is False
    assert (
        behavioral_proofs["rules.dealing.cut_enabled"]["proofKind"]
        == "structural_reference_runtime_cp_sat_counterfactual"
    )
    assert behavioral_proofs["rules.dealing.cut_enabled"]["referenceIndependent"] is True
    for primitive_id in structural_proven:
        proof = behavioral_proofs[primitive_id]
        assert proof["targetsObserved"] == ["reference", "runtime", "cpSat"]
        assert proof["referenceIndependent"] is True
        assert proof["proofKind"] == (
            "structural_reference_runtime_cp_sat_counterfactual"
            if primitive_id in structural_behavioral
            else "structural_domain_ir_counterexample_rejection"
        )
        if primitive_id not in structural_behavioral:
            assert proof["baselineObservationHash"]
    cp_sat_ids = {item["primitiveId"] for item in payloads["cpSatBlueprint"]["entries"]}
    assert any(item.startswith("rules.deck.") for item in cp_sat_ids)
    assert any(item.startswith("rules.bidding.") for item in cp_sat_ids)
    assert not any(item.startswith("rules.bidding.heuristics.") for item in cp_sat_ids)
    assert any(item.startswith("rules.card_rules.") for item in cp_sat_ids)
    assert any(item.startswith("rules.play.") for item in cp_sat_ids)
    assert any(item.startswith("rules.scoring.") for item in cp_sat_ids)
    assert any(item.startswith("rules.projects.") for item in cp_sat_ids)
    assert any(item.startswith("rules.escalation.") for item in cp_sat_ids)
    heuristic_rows = [
        item for item in first.compiled.coverage.rows
        if item.primitive_id.startswith("rules.bidding.heuristics.")
    ]
    assert heuristic_rows
    assert all(item.category == "policy" for item in heuristic_rows)
    assert all("cpSat" not in item.required_targets for item in heuristic_rows)
    expected_heuristics = {
        "rules.bidding.heuristics.sun_rank_weights",
        "rules.bidding.heuristics.sun_extra_suit_card_weight",
        "rules.bidding.heuristics.sun_extra_suit_card_after",
        "rules.bidding.heuristics.hokum_trump_rank_weights",
        "rules.bidding.heuristics.hokum_extra_trump_card_weight",
        "rules.bidding.heuristics.hokum_extra_trump_card_after",
        "rules.bidding.heuristics.hokum_off_trump_ace_ten_weight",
        "rules.bidding.heuristics.sun_threshold",
        "rules.bidding.heuristics.hokum_threshold",
        "rules.bidding.heuristics.ashkal_partner_threshold",
        "rules.bidding.heuristics.second_sun_threshold",
        "rules.bidding.heuristics.second_hokum_threshold",
    }
    assert {item.primitive_id for item in heuristic_rows} == expected_heuristics
    assert all(item.qualification_eligible for item in heuristic_rows)
    assert all(
        behavioral_proofs[item]["proofKind"] == "bidding_policy_counterfactual"
        and behavioral_proofs[item]["targetsObserved"] == ["policy"]
        and behavioral_proofs[item]["rulesHash"] == first.rules_hash
        and behavioral_proofs[item]["counterfactualRulesHash"] != first.rules_hash
        and behavioral_proofs[item]["observationHash"]
        for item in expected_heuristics
    )
    assert "analysis_defaults" not in rules
    assert "policies" not in rules
    assert not any(
        item.primitive_id.startswith(
            ("rules.analysis_defaults.", "rules.policies.")
        )
        for item in coverage.rows
    )
    assert len(payloads["uiSchema"]["fields"]) == first.compiled.coverage.total
    first.compiled.require_publishable()


def test_domain_v3_rejects_unknown_sections_and_hashes_value_changes():
    rules, conventions, localization, bundle_hash, plugins = _source()
    compiler = BalootDomainCompilerV1()
    changed = copy.deepcopy(dict(rules))
    changed["unknown_rule_family"] = {"enabled": True}
    with pytest.raises(ValueError, match="outside the governed primitive schema"):
        compiler.compile(
            rules=changed, conventions=conventions, localization=localization, bundle_hash=bundle_hash,
            plugins=plugins,
        )

    changed = copy.deepcopy(dict(rules))
    changed["play"]["invented_rule"] = True
    with pytest.raises(ValueError, match="outside the governed primitive schema"):
        compiler.compile(
            rules=changed,
            conventions=conventions,
            localization=localization,
            bundle_hash=bundle_hash,
            plugins=plugins,
        )

    baseline = compiler.compile(
        rules=rules, conventions=conventions, localization=localization, bundle_hash=bundle_hash,
        plugins=plugins,
    )
    changed = copy.deepcopy(dict(rules))
    changed["play"]["must_follow_suit"] = not bool(changed["play"]["must_follow_suit"])
    with pytest.raises(ValueError, match="unsupported play.must_follow_suit"):
        compiler.compile(
            rules=changed,
            conventions=conventions,
            localization=localization,
            bundle_hash=bundle_hash,
            plugins=plugins,
        )
    assert baseline.compiled.domain_ir.sha256


def test_domain_v3_fails_closed_when_a_required_cp_sat_plugin_is_missing():
    rules, conventions, localization, bundle_hash, plugins = _source()
    missing_play = [
        item for item in plugins
        if item["id"] != "ortools_cp_sat_play_compiler_v1"
    ]
    compiled = BalootDomainCompilerV1().compile(
        rules=rules,
        conventions=conventions,
        localization=localization,
        bundle_hash=bundle_hash,
        plugins=missing_play,
    ).compiled
    assert compiled.publishable is False
    affected = [
        row for row in compiled.coverage.rows
        if row.primitive_id.startswith(("rules.play.", "rules.card_rules.", "rules.scoring."))
        and row.primitive_id != "rules.card_rules.sequence_strength"
    ]
    assert affected
    assert all("cpSat" in row.missing_targets for row in affected)
    sequence = next(
        row
        for row in compiled.coverage.rows
        if row.primitive_id == "rules.card_rules.sequence_strength"
    )
    assert sequence.definition_kind == "rules.projects.sequence_strength.v1"
    assert "cpSat" not in sequence.missing_targets
    with pytest.raises(ValueError, match="coverage is incomplete"):
        compiled.require_publishable()


def test_domain_v3_fails_closed_when_a_plugin_lacks_the_required_capability():
    rules, conventions, localization, bundle_hash, plugins = _source()
    mutated = copy.deepcopy(plugins)
    target = next(
        item for item in mutated if item["id"] == "ortools_cp_sat_auction_compiler_v1"
    )
    target["capabilities"] = [
        value for value in target["capabilities"] if value != "complete_auction"
    ]
    compiled = BalootDomainCompilerV1().compile(
        rules=rules,
        conventions=conventions,
        localization=localization,
        bundle_hash=bundle_hash,
        plugins=mutated,
    ).compiled
    affected = [
        row for row in compiled.coverage.rows
        if row.primitive_id.startswith(("rules.contracts.", "rules.bidding."))
        and ".heuristics." not in row.primitive_id
    ]
    assert affected
    assert all("cpSat" in row.missing_targets for row in affected)


def test_auction_behavioral_receipts_require_the_real_cp_sat_binding():
    rules, conventions, localization, bundle_hash, plugins = _source()
    without_auction = [
        item for item in plugins
        if item["id"] != "ortools_cp_sat_auction_compiler_v1"
    ]
    compiled = BalootDomainCompilerV1().compile(
        rules=rules,
        conventions=conventions,
        localization=localization,
        bundle_hash=bundle_hash,
        plugins=without_auction,
    ).compiled
    auction_rows = [
        row for row in compiled.coverage.rows
        if row.primitive_id.startswith(("rules.bidding.", "rules.contracts."))
        and ".heuristics." not in row.primitive_id
    ]
    assert auction_rows
    assert all(row.behavioral_or_invariant_status == "unproven" for row in auction_rows)
    assert all(row.qualification_eligible is False for row in auction_rows)
    payloads = compiled.to_dict(include_payloads=True)["targetPayloads"]
    assert not any(
        primitive_id.startswith(("rules.bidding.", "rules.contracts."))
        for primitive_id in payloads["behavioralProofs"]
    )


def test_structural_behavioral_receipts_require_the_real_deal_cp_sat_binding():
    rules, conventions, localization, bundle_hash, plugins = _source()
    without_deal = [
        item for item in plugins
        if item["id"] != "ortools_cp_sat_deal_generator_v1"
    ]
    compiled = BalootDomainCompilerV1().compile(
        rules=rules,
        conventions=conventions,
        localization=localization,
        bundle_hash=bundle_hash,
        plugins=without_deal,
    ).compiled
    rows = {
        row.primitive_id: row
        for row in compiled.coverage.rows
        if row.primitive_id.startswith(
            ("rules.deck.", "rules.seating.", "rules.dealing.")
        )
    }
    assert len(rows) == 25
    assert all(row.behavioral_or_invariant_status == "unproven" for row in rows.values())
    assert all(row.qualification_eligible is False for row in rows.values())
    payloads = compiled.to_dict(include_payloads=True)["targetPayloads"]
    assert not any(
        primitive_id.startswith(
            ("rules.deck.", "rules.seating.", "rules.dealing.")
        )
        for primitive_id in payloads["behavioralProofs"]
    )


def test_behavioral_receipts_fail_closed_when_identity_or_hash_is_invalid():
    rules, _conventions, _localization, _bundle_hash, plugins = _source()
    compiler = BalootDomainCompilerV1()
    primitive_id = "rules.contracts.allowed"
    plugin_index = {str(item["id"]): item for item in plugins}
    valid = compiler._behavioral_proofs(
        rules=rules,
        plugins=plugin_index,
    )[primitive_id]
    definition = compiler._definition_for_path(
        section="contracts",
        path=("rules", "contracts", "allowed"),
    )

    verified = compiler._validated_behavioral_proof(
        primitive_id=primitive_id,
        proof=valid,
        rules_hash=canonical_sha256(rules),
    )
    assert compiler._instance(
        primitive_id=primitive_id,
        definition=definition,
        value=rules["contracts"]["allowed"],
        plugins=plugin_index,
        behavioral_proof=verified,
    ).qualification_eligible is True

    status_only = {"status": "proven_behavioral"}
    wrong_id = copy.deepcopy(valid)
    wrong_id["primitiveId"] = "rules.contracts.default"
    wrong_id_core = dict(wrong_id)
    wrong_id_core.pop("proofHash")
    wrong_id["proofHash"] = canonical_sha256(wrong_id_core)
    bad_hash = {**valid, "proofHash": "0" * 64}
    wrong_rules = copy.deepcopy(valid)
    wrong_rules["rulesHash"] = "1" * 64
    wrong_rules_core = dict(wrong_rules)
    wrong_rules_core.pop("proofHash")
    wrong_rules["proofHash"] = canonical_sha256(wrong_rules_core)

    for malformed in (status_only, wrong_id, bad_hash, wrong_rules):
        receipt = compiler._validated_behavioral_proof(
            primitive_id=primitive_id,
            proof=malformed,
            rules_hash=canonical_sha256(rules),
        )
        instance = compiler._instance(
            primitive_id=primitive_id,
            definition=definition,
            value=rules["contracts"]["allowed"],
            plugins=plugin_index,
            behavioral_proof=receipt,
        )
        assert instance.behavioral_or_invariant_status == "unproven"
        assert instance.qualification_eligible is False


def test_domain_v3_compilation_and_coverage_are_exposed_without_replacing_v2():
    from baloot_platform.service import api

    with TestClient(api.app) as client:
        capabilities = client.get("/api/v2/domain-capabilities")
        assert capabilities.status_code == 200
        payload = capabilities.json()
        assert payload["schemaVersion"] == 2
        assert payload["research"]["compilerQualification"] == {
            "legacyResearchAvailable": True,
            "requiresExecutableBindingCoverage": True,
            "requiresBehavioralOrInvariantCoverage": True,
            "executableBindingComplete": True,
            "behavioralOrInvariantComplete": True,
            "publicationEligible": True,
        }
        assert payload["compilerV3"]["schemaVersion"] == 3
        compiled = payload["compilerV3"]["compiledDomain"]
        response = client.get(f"/api/v2/domain-compilations/{compiled['sha256']}")
        assert response.status_code == 200
        assert response.json()["sha256"] == compiled["sha256"]
        coverage = client.get(f"/api/v2/domain-compilations/{compiled['sha256']}/coverage")
        assert coverage.status_code == 200
        coverage_payload = coverage.json()
        assert coverage_payload["executableBindingComplete"] is True
        assert coverage_payload["behavioralOrInvariantComplete"] is True
        assert coverage_payload["complete"] is True
        assert compiled["publishable"] is True
        assert client.get("/api/v2/domain-compilations/" + "0" * 64).status_code == 404
