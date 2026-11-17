from __future__ import annotations

from pathlib import Path

import pytest

from baloot_platform.research.scenario_contracts import canonical_sha256
from baloot_platform.research.scenario_predicate_capabilities_v3 import (
    NATIVE_V3_ACTOR_ROOT_TACTIC_IDS,
    NATIVE_V3_TACTIC_IDS,
    SCENARIO_PREDICATE_CAPABILITY_BACKEND_V3,
    SCENARIO_PREDICATE_CAPABILITY_REGISTRY_V3,
    ScenarioPredicateCapabilityManifestV3,
    ScenarioPredicateCapabilityV3,
    ScenarioPredicateArgumentContextV3,
    ScenarioPredicateExecutionContextV3,
    derive_scenario_predicate_capabilities_v3,
    native_scenario_predicate_definition_v3,
    shared_scenario_predicate_capability_manifest_v3,
)
from baloot_platform.research.scenario_predicate_manifest_v2 import (
    ScenarioPredicateCatalogLoweringManifestV2,
)


@pytest.mark.parametrize("available", [None, False, True])
def test_causal_argument_metadata_is_optional_and_hash_bound(available):
    row = ScenarioPredicateArgumentContextV3(
        "kind", "small_card_big_gain", ("play",), causal_verification_available=available,
    )
    wire = row.to_dict()
    if available is None:
        assert "causalVerificationAvailable" not in wire
    else:
        assert wire["causalVerificationAvailable"] is available
        historical = ScenarioPredicateArgumentContextV3("kind", "small_card_big_gain", ("play",))
        assert canonical_sha256(wire) != canonical_sha256(historical.to_dict())


@pytest.mark.parametrize("available", [0, 1, "false", {}])
def test_causal_argument_metadata_rejects_coercion(available):
    with pytest.raises(TypeError, match="must be a boolean"):
        ScenarioPredicateArgumentContextV3(
            "kind", "small_card_big_gain", ("play",), causal_verification_available=available,
        )


def _rebuild(
    row: ScenarioPredicateCapabilityV3,
    **changes,
) -> ScenarioPredicateCapabilityV3:
    values = {
        "kind": row.kind,
        "definition_sha256": row.definition_sha256,
        "registration_sha256": row.registration_sha256,
        "lowerer_owner": row.lowerer_owner,
        "evidence_extractor": row.evidence_extractor,
        "owning_test": row.owning_test,
        "qualification_binding": row.qualification_binding,
        "required_rule_primitives": row.required_rule_primitives,
        "execution_contexts": row.execution_contexts,
        "argument_contexts": row.argument_contexts,
        "native_status": row.native_status,
    }
    values.update(changes)
    return ScenarioPredicateCapabilityV3.build(**values)


def test_v3_manifest_dynamically_covers_every_exposed_lowering_registration(
    root: Path,
):
    base = ScenarioPredicateCatalogLoweringManifestV2()
    manifest = ScenarioPredicateCapabilityManifestV3(base=base)
    definitions = {item.kind: item for item in base.exposed_definitions}
    registrations = {item.kind: item for item in base.registrations}
    capabilities = {item.kind: item for item in manifest.capabilities}

    assert set(capabilities) == set(definitions) == set(registrations)
    assert len(capabilities) == len(manifest.capabilities)
    for kind, row in capabilities.items():
        definition = native_scenario_predicate_definition_v3(definitions[kind])
        registration = registrations[kind]
        assert row.definition_sha256 == definition.sha256
        assert row.registration_sha256 == canonical_sha256(registration.to_dict())
        assert row.lowerer_owner == registration.lowerer_owner
        assert row.evidence_extractor == registration.evidence_extractor_id
        assert row.owning_test == registration.owning_test
        assert (root / row.owning_test).is_file()
        assert row.required_rule_primitives == tuple(
            sorted(definition.required_domain_primitives)
        )
        assert row.native_status == "native_exact"
        assert row.execution_contexts
        if definition.stage == "state":
            expected = {"play"}
            if kind in {
                "state.phase", "state.next_actor", "state.legal_action_count",
                "state.match_score", "state.current_raw_points",
                "state.current_project_points",
            }:
                expected.update({"bid", "escalation", "project"})
            if kind == "state.current_game_points":
                expected.update({"bid", "escalation", "project"})
            if kind == "state.observer_card_visibility":
                expected.update({"bid", "escalation", "project"})
            assert {context.decision_kind for context in row.execution_contexts} == expected


def test_v3_tactic_argument_contexts_close_detector_prerequisites() -> None:
    capabilities = {
        item.kind: item
        for item in shared_scenario_predicate_capability_manifest_v3().capabilities
    }
    for predicate_kind in ("tactic.applicable", "tactic.verified"):
        rows = {
            item.argument_value: item
            for item in capabilities[predicate_kind].argument_contexts
        }
        assert set(rows) == NATIVE_V3_TACTIC_IDS
        for tactic, row in rows.items():
            expected = tactic not in {"large_card_sacrifice", "small_card_big_gain"}
            if predicate_kind == "tactic.verified":
                assert row.causal_verification_available is expected
                assert row.to_dict()["causalVerificationAvailable"] is expected
            else:
                assert "causalVerificationAvailable" not in row.to_dict()
        assert all(row.required_proof_bindings for row in rows.values())
        project = rows["project_bidding"]
        assert project.decision_kinds == ("project",)
        assert project.required_utility == "awarded_point_margin"
        assert ("root_project_consequence_variables", 2) in project.required_proof_bindings
        assert ("tactical_detector_profile", 1) in project.required_proof_bindings
        assert {(row.origin, row.proof_mode, row.visibility)
                for row in project.execution_contexts} == {
            (origin, mode, "complete_state")
            for origin in ("search", "exact_trace_prefix")
            for mode in ("exact",)
        }
        assert "executionContexts" in project.to_dict()
        assert NATIVE_V3_ACTOR_ROOT_TACTIC_IDS == {
            "only_winning_move", "debate", "contract_rescue", "kabout_creation", "kabout_preservation",
            "kabout_prevention", "small_card_big_gain"}
        for tactic in NATIVE_V3_ACTOR_ROOT_TACTIC_IDS:
            assert {(context.origin, context.decision_kind, context.proof_mode)
                    for context in rows[tactic].execution_contexts
                    if context.visibility == "actor_information_set"} == {
                (origin, "play", "exact") for origin in ("search", "exact_trace_prefix")
            }
        assert all(context.visibility == "complete_state"
                   for tactic, row in rows.items() if tactic not in NATIVE_V3_ACTOR_ROOT_TACTIC_IDS
                   for context in row.execution_contexts)
        assert rows["contract_rescue"].required_utility == "contract_success"
        assert ("trump_control_suffix_actions", 1) in rows["trump_control"].required_proof_bindings
        assert rows["trump_control"].required_utility == "awarded_point_margin"
        assert ("tactical_detector_profile", 1) in rows["trump_control"].required_proof_bindings
        assert ("causal_gain_scale", 1) in rows["trump_control"].required_proof_bindings
        assert rows["partner_enablement"].required_utility == "awarded_point_margin"
        assert ("root_raw_payoff_variables", 2) in rows["partner_enablement"].required_proof_bindings
        assert rows["entry_unblocking"].required_utility == "awarded_point_margin"
        assert ("root_raw_payoff_variables", 2) in rows["entry_unblocking"].required_proof_bindings
        assert {(row.proof_mode, row.visibility) for row in rows["entry_unblocking"].execution_contexts} == {
            ("exact", "complete_state")}
        assert {(row.proof_mode, row.visibility) for row in rows["partner_enablement"].execution_contexts} == {
            ("exact", "complete_state")}
        assert {(row.proof_mode, row.visibility) for row in rows["trump_control"].execution_contexts} == {
            ("exact", "complete_state")}
        assert ("tactical_detector_profile", 1) in rows["small_card_big_gain"].required_proof_bindings
        assert ("tactical_detector_profile", 1) in rows["large_card_sacrifice"].required_proof_bindings
        for tactic in (
            "kabout_creation",
            "kabout_preservation",
            "kabout_prevention",
        ):
            assert rows[tactic].required_utility == "kabout"
            assert ("base_trick_variables", 2) in rows[
                tactic
            ].required_proof_bindings
        for tactic in (
            "entry_unblocking",
            "partner_enablement",
            "trump_control",
            "small_card_big_gain",
            "large_card_sacrifice",
        ):
            assert rows[tactic].decision_kinds == ("play",)
            assert any(
                "principal" in binding
                for binding, _minimum in rows[tactic].required_proof_bindings
            )


def test_v3_capabilities_preserve_frozen_v2_definition_and_manifest_hashes():
    base = ScenarioPredicateCatalogLoweringManifestV2()
    base_catalog = base.catalog()
    definition_hashes = {
        item.kind: item.sha256 for item in base.exposed_definitions
    }

    first = ScenarioPredicateCapabilityManifestV3(base=base)
    second = ScenarioPredicateCapabilityManifestV3(base=base)
    catalog = first.catalog()
    declared = catalog.pop("sha256")

    assert declared == first.manifest_hash == second.manifest_hash
    assert declared == canonical_sha256(catalog)
    assert catalog["schemaVersion"] == 3
    assert catalog["registryVersion"] == SCENARIO_PREDICATE_CAPABILITY_REGISTRY_V3
    assert catalog["backendId"] == SCENARIO_PREDICATE_CAPABILITY_BACKEND_V3
    assert catalog["baseManifestHash"] == base.manifest_hash
    assert base.catalog() == base_catalog
    assert {
        item.kind: item.sha256 for item in base.exposed_definitions
    } == definition_hashes


def test_native_delayed_capability_is_opt_in_and_complete_state_exact_only():
    legacy = ScenarioPredicateCapabilityManifestV3()
    native = ScenarioPredicateCapabilityManifestV3(native_delayed_payoff=True)
    assert native.manifest_hash != legacy.manifest_hash
    assert legacy.catalog() == shared_scenario_predicate_capability_manifest_v3().catalog()
    for predicate_kind in ("tactic.applicable", "tactic.verified"):
        old_rows = {row.argument_value: row for row in legacy.capability(predicate_kind).argument_contexts}
        rows = {row.argument_value: row for row in native.capability(predicate_kind).argument_contexts}
        for tactic, row in rows.items():
            if tactic not in {"small_card_big_gain", "large_card_sacrifice"}:
                assert row == old_rows[tactic]
                continue
            assert row.required_utility == "awarded_point_margin"
            assert ("root_raw_payoff_variables", 2) in row.required_proof_bindings
            assert ("causal_gain_terminal_awarded_scope", 1) in row.required_proof_bindings
            assert {(context.origin, context.decision_kind, context.proof_mode, context.visibility)
                for context in row.execution_contexts} == {
                    (origin, "play", "exact", "complete_state")
                    for origin in ("search", "exact_trace_prefix")}
            if predicate_kind == "tactic.verified":
                assert row.causal_verification_available is True
                assert old_rows[tactic].causal_verification_available is False
    with pytest.raises(ValueError, match="argument contexts drifted"):
        ScenarioPredicateCapabilityManifestV3(native.capabilities)


@pytest.mark.parametrize("invalid", [None, 0, 1, "true"])
def test_native_delayed_capability_rejects_coercion(invalid):
    with pytest.raises(TypeError, match="must be a boolean"):
        ScenarioPredicateCapabilityManifestV3(native_delayed_payoff=invalid)


def test_v3_manifest_is_order_independent_and_shared_projection_is_immutable():
    rows = derive_scenario_predicate_capabilities_v3()
    forward = ScenarioPredicateCapabilityManifestV3(rows)
    reverse = ScenarioPredicateCapabilityManifestV3(reversed(rows))
    shared = shared_scenario_predicate_capability_manifest_v3()
    projection = shared.catalog()
    expected = shared.catalog()

    projection["capabilities"].clear()

    assert forward.manifest_hash == reverse.manifest_hash
    assert shared.catalog() == expected
    assert shared is shared_scenario_predicate_capability_manifest_v3()


def test_v3_card_root_and_principal_variation_predicates_are_play_only():
    manifest = ScenarioPredicateCapabilityManifestV3()

    for kind in (
        "proof.winning_root_action_matches",
        "proof.winning_root_card_rank_count",
        "proof.principal_variation_trick_winner",
        "proof.principal_variation_trick_team_rank_count",
    ):
        row = manifest.capability(kind)
        assert row is not None
        assert row.decision_kinds == ("play",)
        assert all(item.decision_kind == "play" for item in row.execution_contexts)


def test_v3_actor_information_capabilities_are_exact_and_uncounted():
    manifest = ScenarioPredicateCapabilityManifestV3()
    continuation = manifest.capability("proof.winning_continuation_line_count")
    root_count = manifest.capability("proof.winning_root_action_count")

    assert continuation is not None
    assert continuation.visibility_classes == ("complete_state",)
    assert root_count is not None
    assert "actor_information_set" in root_count.visibility_classes
    project_contexts = [
        context
        for row in manifest.capabilities
        for context in row.execution_contexts
        if context.decision_kind == "project"
    ]
    assert project_contexts
    assert {context.visibility for context in project_contexts} == {
        "complete_state",
        "actor_information_set",
    }
    actor_contexts = [
        context
        for row in manifest.capabilities
        for context in row.execution_contexts
        if context.visibility == "actor_information_set"
    ]
    assert actor_contexts
    assert all(
        context.proof_mode == "exact"
        and not context.line_universes
        for context in actor_contexts
    )
    assert {context.decision_kind for context in actor_contexts} == {
        "bid",
        "escalation",
        "project",
        "play",
    }


@pytest.mark.parametrize('kind', (
    'proof.principal_variation_trick_winner',
    'proof.principal_variation_trick_team_rank_count',
))
def test_v3_does_not_advertise_missing_actor_information_evidence_producers(kind):
    row = ScenarioPredicateCapabilityManifestV3().capability(kind)
    assert row is not None
    assert row.visibility_classes == ('complete_state',)
    assert row.execution_contexts


def test_v3_observer_scope_support_is_exact_actor_play_only():
    row = ScenarioPredicateCapabilityManifestV3().capability("state.observer_visibility")
    actor = [context for context in row.execution_contexts
             if context.visibility == "actor_information_set"]
    assert {context.origin for context in actor} == {"search", "exact_trace_prefix"}
    assert {(context.proof_mode, context.decision_kind) for context in actor} == {("exact", "play")}


@pytest.mark.parametrize('kind', ('proof.winning_root_action_matches', 'proof.winning_root_card_rank_count'))
def test_v3_actor_card_root_metadata_is_native_without_mutating_v2(kind):
    from baloot_platform.research.scenario_adjudication_predicates_v3 import ScenarioPredicateCatalogLoweringManifestV3

    base = ScenarioPredicateCatalogLoweringManifestV2()
    original = base.definition(kind)
    before = base.catalog()
    merged = ScenarioPredicateCatalogLoweringManifestV3(base)
    definition = merged.definition(kind)
    assert original.cp_sat_support.actor_information_set == 'reference_world_only'
    assert definition.cp_sat_support.actor_information_set == 'nonanticipative'
    assert definition.sha256 != original.sha256
    assert merged.capability_manifest.capability(kind).definition_sha256 == definition.sha256
    assert 'tests/test_research_ortools_unified_executor.py' in definition.tests
    assert base.catalog() == before
    contexts = merged.capability_manifest.capability(kind).execution_contexts
    assert {(row.origin, row.decision_kind, row.proof_mode) for row in contexts
            if row.visibility == 'actor_information_set'} == {
        (origin, 'play', 'exact') for origin in ('search', 'exact_trace_prefix')}


def test_v3_continuation_count_advertises_only_executable_line_universes():
    row = ScenarioPredicateCapabilityManifestV3().capability(
        "proof.winning_continuation_line_count"
    )
    assert row is not None

    for context in row.execution_contexts:
        assert context.visibility == "complete_state"
        assert context.line_universes == (
            ("policy_conditioned",)
            if context.proof_mode == "policy_conditioned"
            else ("all_legal", "optimal_strategy")
        )


def test_v3_capability_dimensions_are_projections_of_exact_contexts():
    manifest = ScenarioPredicateCapabilityManifestV3()

    for row in manifest.capabilities:
        contexts = row.execution_contexts
        assert row.origins == tuple(sorted({item.origin for item in contexts}))
        assert row.decision_kinds == tuple(
            sorted({item.decision_kind for item in contexts})
        )
        assert row.proof_modes == tuple(
            sorted({item.proof_mode for item in contexts})
        )
        assert row.visibility_classes == tuple(
            sorted({item.visibility for item in contexts})
        )


@pytest.mark.parametrize("mutation", ("missing", "extra", "duplicate"))
def test_v3_manifest_fails_closed_on_capability_coverage_drift(mutation: str):
    rows = list(derive_scenario_predicate_capabilities_v3())
    if mutation == "missing":
        rows.pop()
        match = "coverage is incomplete"
    elif mutation == "extra":
        rows.append(_rebuild(rows[0], kind="future.unregistered"))
        match = "coverage is incomplete"
    else:
        rows.append(rows[0])
        match = "contain duplicates"

    with pytest.raises(ValueError, match=match):
        ScenarioPredicateCapabilityManifestV3(rows)


@pytest.mark.parametrize(
    ("change", "match"),
    (
        ({"definition_sha256": "a" * 64}, "definition hash drifted"),
        ({"registration_sha256": "b" * 64}, "registration hash drifted"),
        ({"lowerer_owner": "wrong.Owner"}, "registration binding drifted"),
        ({"evidence_extractor": "cp_sat.wrong"}, "registration binding drifted"),
        ({"owning_test": "tests/wrong.py"}, "registration binding drifted"),
        ({"qualification_binding": "proof_variable"}, "qualification binding drifted"),
        ({"required_rule_primitives": ("rules.future",)}, "rule primitives drifted"),
    ),
)
def test_v3_manifest_fails_closed_on_owner_or_semantic_binding_drift(
    change: dict,
    match: str,
):
    rows = list(derive_scenario_predicate_capabilities_v3())
    index = next(
        index
        for index, row in enumerate(rows)
        if row.kind == "auction.contract"
    )
    rows[index] = _rebuild(rows[index], **change)

    with pytest.raises(ValueError, match=match):
        ScenarioPredicateCapabilityManifestV3(rows)


def test_v3_manifest_rejects_context_drift_and_tampered_row_checksum():
    rows = list(derive_scenario_predicate_capabilities_v3())
    index = next(
        index
        for index, row in enumerate(rows)
        if row.kind == "auction.contract"
    )
    reduced = rows[index].execution_contexts[:-1]
    rows[index] = _rebuild(rows[index], execution_contexts=reduced)
    with pytest.raises(ValueError, match="execution contexts drifted"):
        ScenarioPredicateCapabilityManifestV3(rows)

    rows = list(derive_scenario_predicate_capabilities_v3())
    object.__setattr__(rows[index], "sha256", "0" * 64)
    with pytest.raises(ValueError, match="checksum drifted"):
        ScenarioPredicateCapabilityManifestV3(rows)


def test_v3_unsupported_development_row_cannot_be_published():
    rows = list(derive_scenario_predicate_capabilities_v3())
    index = next(
        index
        for index, row in enumerate(rows)
        if row.kind == "auction.contract"
    )
    rows[index] = _rebuild(rows[index], native_status="unsupported")

    development = ScenarioPredicateCapabilityManifestV3(
        rows,
        require_publishable=False,
    )
    with pytest.raises(ValueError, match="not native exact"):
        development.require_publishable()
    with pytest.raises(ValueError, match="not native exact"):
        ScenarioPredicateCapabilityManifestV3(rows)


@pytest.mark.parametrize(
    "context",
    (
        {
            "origin": "search",
            "decision_kind": "play",
            "proof_mode": "bounded",
            "visibility": "actor_information_set",
        },
        {
            "origin": "search",
            "decision_kind": "play",
            "proof_mode": "exact",
            "visibility": "actor_information_set",
            "line_universes": ("all_legal",),
        },
    ),
)
def test_v3_actor_information_context_contract_fails_closed(context: dict):
    with pytest.raises(ValueError, match="exact and uncounted only"):
        ScenarioPredicateExecutionContextV3(**context)


def test_v3_actor_information_context_supports_nonplay_decisions() -> None:
    context = ScenarioPredicateExecutionContextV3(
        "exact_trace_prefix",
        "bid",
        "exact",
        "actor_information_set",
    )

    assert context.decision_kind == "bid"
