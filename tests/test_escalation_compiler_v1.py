from __future__ import annotations

import copy

import pytest

from baloot_platform.core.escalation_ir import (
    EscalationDomainIRV1,
    EscalationStateV1,
)
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.research import (
    BalootDomainCompilerV1,
    OrToolsCpSatEscalationCompilerV1,
    ReferenceEscalationInterpreterV1,
)


def test_compiled_escalation_ir_is_identical_in_all_three_domain_targets(
    rules, convention
):
    compiled = BalootDomainCompilerV1().compile(
        rules=rules.data,
        conventions=convention,
        localization={},
        bundle_hash="0" * 64,
        plugins=(),
    ).compiled
    targets = compiled.target_payloads
    runtime = targets["runtimePlan"]["compiledEscalation"]
    reference = targets["referencePlan"]["compiledEscalation"]
    cp_sat = targets["cpSatBlueprint"]["compiledEscalation"]
    assert runtime == reference == cp_sat
    assert runtime["sha256"] == EscalationDomainIRV1.from_rules(rules.data).sha256
@pytest.mark.parametrize("contract", (Contract.HOKUM, Contract.SUN, Contract.ASHKAL))
def test_escalation_domain_ir_reference_runtime_parity(contract, rules):
    score = (101, 50) if contract != Contract.HOKUM else (0, 0)
    proof = OrToolsCpSatEscalationCompilerV1.reference_runtime_parity(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=contract,
        match_score=score,
        rules=rules,
    )
    assert proof["targetsObserved"] == ["reference", "runtime"]
    assert proof["traceCount"] > 0
    assert len(proof["proofHash"]) == 64


def test_escalation_chain_mutation_influences_shared_runtime_reference_and_cp_sat(rules):
    payload = copy.deepcopy(rules.data)
    payload["escalation"]["hokum_chain"] = ["double", "triple"]
    mutated = RuleBook(payload)
    domain = EscalationDomainIRV1.from_rules(mutated.data)
    state = EscalationStateV1(domain.default_level)
    for actor, action in ((Seat.WEST, "double"), (Seat.SOUTH, "triple")):
        state = domain.transition(
            state,
            declarer=Seat.SOUTH,
            contract=Contract.HOKUM,
            match_score=(0, 0),
            actor=actor,
            action=action,
        ).state
    with pytest.raises(ValueError, match="expected None"):
        domain.transition(
            state,
            declarer=Seat.SOUTH,
            contract=Contract.HOKUM,
            match_score=(0, 0),
            actor=Seat.WEST,
            action="four",
        )
    reference = ReferenceEscalationInterpreterV1(domain)
    with pytest.raises(ValueError, match="expected None"):
        reference.apply(
            state,
            declarer=Seat.SOUTH,
            contract=Contract.HOKUM,
            match_score=(0, 0),
            actor=Seat.WEST,
            action="four",
        )
    assert OrToolsCpSatEscalationCompilerV1().compile(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        rules=mutated,
        required_level="four",
    ) is None


def test_shared_transition_and_reference_reject_legacy_parallel_construction(rules):
    with pytest.raises(TypeError, match="compiled EscalationDomainIRV1"):
        ReferenceEscalationInterpreterV1(dict(rules.data["escalation"]))
    with pytest.raises(TypeError, match="EscalationStateV1"):
        EscalationDomainIRV1.from_rules(rules.data).transition(
            {"level": "normal"},
            declarer=Seat.SOUTH,
            contract=Contract.HOKUM,
            match_score=(0, 0),
            actor=Seat.WEST,
            action="double",
        )


def test_escalation_domain_ir_rejects_invalid_raw_configuration(rules):
    wrong_type = copy.deepcopy(rules.data)
    wrong_type["escalation"]["enabled"] = "false"
    with pytest.raises(ValueError, match="invalid escalation domain"):
        EscalationDomainIRV1.from_rules(wrong_type)

    unsupported_chain = copy.deepcopy(rules.data)
    unsupported_chain["escalation"]["hokum_chain"] = ["double", "four"]
    with pytest.raises(ValueError, match="ordered supported prefix"):
        EscalationDomainIRV1.from_rules(unsupported_chain)

    unsupported_opener = copy.deepcopy(rules.data)
    unsupported_opener["escalation"]["opening_actor_policy"] = "declarer_next"
    with pytest.raises(ValueError, match="opening actor policy"):
        EscalationDomainIRV1.from_rules(unsupported_opener)

    unsupported_closure = copy.deepcopy(rules.data)
    unsupported_closure["escalation"]["closure_policy"] = "actor_pass_sequence"
    with pytest.raises(ValueError, match="closure policy"):
        EscalationDomainIRV1.from_rules(unsupported_closure)


def test_domain_owns_both_openers_and_one_actorless_closure(rules):
    domain = EscalationDomainIRV1.from_rules(rules.data)
    decisions = domain.legal_decisions(
        domain.initial_state(),
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
    )

    closures = [decision for decision in decisions if decision.is_closure]
    assert [decision.code for decision in closures] == ["decline"]
    assert {decision.actor for decision in decisions if not decision.is_closure} == {
        Seat.WEST,
        Seat.EAST,
    }


def test_compiler_proves_complete_hokum_chain_with_runtime_parity(rules):
    compiler = OrToolsCpSatEscalationCompilerV1()
    plan = compiler.compile(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        rules=rules,
        required_level="gahwa",
    )
    assert plan is not None
    assert [item.action for item in plan.actions] == [
        "double", "triple", "four", "gahwa"
    ]
    assert [item.actor for item in plan.actions] == [
        plan.double_seat, Seat.SOUTH, plan.double_seat, Seat.SOUTH
    ]
    assert plan.level == "gahwa"
    assert plan.solver_status == "optimal"


def test_compiler_models_sun_threshold_and_split_score(rules):
    compiler = OrToolsCpSatEscalationCompilerV1()
    feasible = compiler.compile(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=Contract.SUN,
        match_score=(101, 50),
        rules=rules,
        required_level="double",
    )
    impossible = compiler.compile(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=Contract.SUN,
        match_score=(80, 60),
        rules=rules,
        required_level="double",
    )
    assert feasible is not None and feasible.level == "double"
    assert impossible is None


def test_disabled_escalation_only_allows_normal(rules):
    data = copy.deepcopy(rules.data)
    data["escalation"]["enabled"] = False
    rules = RuleBook(data)
    compiler = OrToolsCpSatEscalationCompilerV1()
    normal = compiler.compile(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        rules=rules,
        required_level="normal",
    )
    doubled = compiler.compile(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        rules=rules,
        required_level="double",
    )
    assert normal is not None and not normal.actions
    assert doubled is None


def test_default_mode_and_actor_policy_change_compiled_legal_table(rules):
    locked_data = copy.deepcopy(rules.data)
    locked_data["escalation"]["default_mode"] = "locked"
    locked = RuleBook(locked_data)
    plans = OrToolsCpSatEscalationCompilerV1._legal_escalations(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        rules=locked,
    )
    unspecified_double = [
        item
        for item in plans
        if len(item.actions) == 1
        and item.actions[0].requested_mode is None
        and item.level == "double"
    ]
    assert unspecified_double and all(item.locked for item in unspecified_double)

    team_data = copy.deepcopy(rules.data)
    team_data["escalation"]["response_actor_policy"] = "team"
    team = RuleBook(team_data)
    team_plans = OrToolsCpSatEscalationCompilerV1._legal_escalations(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        rules=team,
    )
    assert any(
        len(item.actions) >= 2
        and item.actions[1].action == "triple"
        and item.actions[1].actor == Seat.NORTH
        for item in team_plans
    )


def test_compiler_is_deterministic_and_prefers_normal_when_unrestricted(rules):
    compiler = OrToolsCpSatEscalationCompilerV1()
    kwargs = dict(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        rules=rules,
    )
    first = compiler.compile(**kwargs)
    second = compiler.compile(**kwargs)
    assert first == second
    assert first is not None and first.level == "normal" and not first.actions


def test_every_escalation_rule_leaf_has_runtime_compiler_influence(rules):
    compiler = OrToolsCpSatEscalationCompilerV1()

    short_hokum_data = copy.deepcopy(rules.data)
    short_hokum_data["escalation"]["hokum_chain"] = ["double", "triple"]
    assert compiler.compile(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        rules=RuleBook(short_hokum_data),
        required_level="four",
    ) is None

    no_sun_data = copy.deepcopy(rules.data)
    no_sun_data["escalation"]["sun_chain"] = []
    assert compiler.compile(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=Contract.SUN,
        match_score=(101, 50),
        rules=RuleBook(no_sun_data),
        required_level="double",
    ) is None

    threshold_data = copy.deepcopy(rules.data)
    threshold_data["escalation"]["sun_score_threshold"] = 110
    assert compiler.compile(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=Contract.SUN,
        match_score=(101, 50),
        rules=RuleBook(threshold_data),
        required_level="double",
    ) is None

    no_split_data = copy.deepcopy(rules.data)
    no_split_data["escalation"]["sun_double_requires_split_score"] = False
    assert compiler.compile(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=Contract.SUN,
        match_score=(80, 60),
        rules=RuleBook(no_split_data),
        required_level="double",
    ) is not None

    invalid_default = copy.deepcopy(rules.data)
    invalid_default["escalation"]["default_level"] = "double"
    try:
        RuleBook(invalid_default)
    except ValueError as exc:
        assert "escalation.default_level" in str(exc)
    else:
        raise AssertionError("unsupported default escalation level did not fail closed")


def test_independent_reference_and_cp_sat_match_exact_required_sequence(rules):
    actions = (
        (Seat.WEST, "double", None),
        (Seat.SOUTH, "triple", None),
        (Seat.WEST, "four", "locked"),
    )
    expected = ReferenceEscalationInterpreterV1.from_rules(
        rules.data
    ).apply_sequence(
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        actions=actions,
    )
    plan = OrToolsCpSatEscalationCompilerV1().compile(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        rules=rules,
        required_actions=actions,
    )
    assert plan is not None
    assert plan.level == expected.state.level == "four"
    assert plan.locked is expected.state.locked is True
    assert plan.double_seat == expected.state.double_seat == Seat.WEST


def test_required_sequence_proves_response_actor_policy_counterfactual(rules):
    actions = (
        (Seat.WEST, "double", None),
        (Seat.NORTH, "triple", None),
    )
    compiler = OrToolsCpSatEscalationCompilerV1()
    assert compiler.compile(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        rules=rules,
        required_actions=actions,
    ) is None
    data = copy.deepcopy(rules.data)
    data["escalation"]["response_actor_policy"] = "team"
    assert compiler.compile(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        rules=RuleBook(data),
        required_actions=actions,
    ) is not None


def test_required_escalation_prefix_is_constructible(rules):
    compiler = OrToolsCpSatEscalationCompilerV1()
    prefix = ((Seat.WEST, "double", None),)
    plan = compiler.compile(
        dealer=Seat.EAST,
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        rules=rules,
        required_actions=prefix,
        required_actions_complete=False,
    )

    assert plan is not None
    assert tuple(
        (item.actor, item.action, item.requested_mode)
        for item in plan.actions[: len(prefix)]
    ) == prefix
