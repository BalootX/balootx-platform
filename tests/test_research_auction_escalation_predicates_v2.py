from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
from ortools.sat.python import cp_model

from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.ortools_escalation_stage import OrToolsEscalationStageV1
from baloot_platform.research.ortools_scenario_model import OrToolsScenarioModelV1
from baloot_platform.research.scenario_contracts import ScenarioPredicateV1
from baloot_platform.research.scenario_predicate_auction_escalation_v2 import (
    AUCTION_ESCALATION_PREDICATE_REGISTRATIONS_V1,
    AuctionEscalationPredicateLoweringV1,
    CpSatAuctionPredicateContextV1,
    CpSatEscalationPredicateContextV1,
)
from baloot_platform.research.scenario_registry import built_in_scenario_predicates_v1


def _predicate(index: int, kind: str, arguments: dict) -> ScenarioPredicateV1:
    return ScenarioPredicateV1.build(
        {
            "predicateId": f"PRED-STAGE-{index:02d}",
            "kind": kind,
            "version": 1,
            "hardness": "hard",
            "arguments": arguments,
        }
    )


def _solve(model: cp_model.CpModel) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    return solver


def _auction_action(action) -> dict:
    return {
        "actor": action.actor.name.lower(),
        "action": action.action,
        **(
            {"trumpSuit": action.trump.name.lower()} if action.trump is not None else {}
        ),
    }


def _auction_evidence(rules) -> dict[str, dict]:
    lowerer = AuctionEscalationPredicateLoweringV1()
    evidence: dict[str, dict] = {}

    scenario = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    fragment = scenario.add_auction_fragment()
    target = next(
        plan
        for plan in fragment.plans
        if plan.contract == Contract.HOKUM and plan.trump is not None
    )
    actions = [_auction_action(action) for action in target.actions]
    pass_count = sum(action["action"] == "pass" for action in actions)
    predicates = (
        _predicate(1, "auction.dealer", {"seat": "east"}),
        _predicate(2, "auction.buyer", {"seat": target.buyer.name.lower()}),
        _predicate(3, "auction.contract", {"contract": target.contract.value}),
        _predicate(4, "auction.trump", {"suit": target.trump.name.lower()}),
        _predicate(5, "auction.purchase_round", {"round": target.purchase_round}),
        _predicate(6, "auction.round", {"round": target.actions[-1].round_index}),
        _predicate(
            7,
            "auction.action_sequence",
            {"actions": actions, "complete": True},
        ),
        _predicate(
            8,
            "auction.pass_count",
            {"minimum": pass_count, "maximum": pass_count},
        ),
        _predicate(9, "auction.terminal", {"terminal": True}),
    )
    context = CpSatAuctionPredicateContextV1(
        scenario.model,
        fragment,
        Seat.EAST,
    )
    lowerings = tuple(
        lowerer.lower_predicate(predicate, context) for predicate in predicates
    )
    for lowering in lowerings:
        scenario.model.add_assumption(lowering.literal)
    solver = _solve(scenario.model)
    evidence.update({row.kind: row.extract(solver) for row in lowerings})

    ashkal_model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    ashkal_fragment = ashkal_model.add_auction_fragment()
    ashkal_target = next(
        plan for plan in ashkal_fragment.plans if plan.contract == Contract.ASHKAL
    )
    relation = (
        "bidder"
        if ashkal_target.buyer
        == next(
            action.actor
            for action in reversed(ashkal_target.actions)
            if action.contract == Contract.ASHKAL
        )
        else "bidder_partner"
    )
    ashkal = lowerer.lower_predicate(
        _predicate(10, "auction.ashkal_buyer_relation", {"relation": relation}),
        CpSatAuctionPredicateContextV1(
            ashkal_model.model,
            ashkal_fragment,
            Seat.EAST,
        ),
    )
    ashkal_model.model.add_assumption(ashkal.literal)
    evidence[ashkal.kind] = ashkal.extract(_solve(ashkal_model.model))

    dealer = Seat.EAST
    order = (dealer.next(), dealer.next().next(), dealer.previous(), dealer)
    prefix = tuple(
        {"roundIndex": 1, "actor": actor.name.lower(), "action": "pass"}
        for actor in order
    )
    prefix_model = OrToolsScenarioModelV1(rules=rules, dealer=dealer)
    prefix_fragment = prefix_model.add_auction_fragment(required_actions=prefix)
    prefix_target = prefix_fragment.plans[0]
    legal_actions = []
    for choice in prefix_target.legal_next_choices:
        if choice.startswith("hokum:"):
            legal_actions.append(
                {"action": "hokum", "trumpSuit": choice.split(":", 1)[1]}
            )
        else:
            legal_actions.append({"action": choice})
    legal = lowerer.lower_predicate(
        _predicate(
            11,
            "auction.legal_actions",
            {"actions": legal_actions, "complete": True},
        ),
        CpSatAuctionPredicateContextV1(
            prefix_model.model,
            prefix_fragment,
            dealer,
            view="prefix",
        ),
    )
    prefix_model.model.add_assumption(legal.literal)
    evidence[legal.kind] = legal.extract(_solve(prefix_model.model))
    return evidence


def _escalation_evidence(rules, *, native=False, actions_override=None) -> dict[str, dict]:
    stage = OrToolsEscalationStageV1(
        EscalationDomainIRV1.from_rules(rules.data),
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        native_cp_sat=native,
    )
    context = CpSatEscalationPredicateContextV1.attach(stage)
    target = next(
        candidate
        for candidate in context.candidates
        if candidate.level == "four" and candidate.locked
    ) if actions_override is None else None
    actions = actions_override if actions_override is not None else [
        {
            "actor": action.actor.name.lower(),
            "action": action.action,
            **(
                {"mode": action.requested_mode}
                if action.requested_mode is not None
                else {}
            ),
        }
        for action in target.actions
    ]
    lowerer = AuctionEscalationPredicateLoweringV1()
    predicates = (
        _predicate(12, "escalation.level", {"level": "four"}),
        _predicate(
            13, "escalation.action_sequence", {"actions": actions, "complete": True}
        ),
        _predicate(14, "escalation.locked", {"locked": True}),
        _predicate(
            15,
            "escalation.eligible_actor",
            {"team": "north_south", "seat": "south", "eligible": True},
        ),
        _predicate(
            16,
            "escalation.legal_actions",
            {
                "actions": [{"action": "decline"}, {"action": "gahwa"}],
                "complete": True,
            },
        ),
        _predicate(17, "escalation.terminal", {"terminal": False}),
    )
    lowerings = tuple(
        lowerer.lower_predicate(predicate, context) for predicate in predicates
    )
    for lowering in lowerings:
        context.model.add_assumption(lowering.literal)
    solver = _solve(context.model)
    return {row.kind: row.extract(solver) for row in lowerings}


def test_registration_tuple_covers_every_v1_auction_and_escalation_predicate():
    expected = {
        definition.kind
        for definition in built_in_scenario_predicates_v1()
        if definition.stage in {"auction", "escalation"}
    }
    actual = {
        registration.kind
        for registration in AUCTION_ESCALATION_PREDICATE_REGISTRATIONS_V1
    }
    assert actual == expected
    assert len(actual) == 17
    assert len(AUCTION_ESCALATION_PREDICATE_REGISTRATIONS_V1) == 17


def test_all_auction_predicates_are_exact_and_evidence_is_stable(rules):
    first = _auction_evidence(rules)
    second = _auction_evidence(rules)
    assert set(first) == set(_AUCTION_KINDS_FOR_TEST)
    assert first == second
    assert all(row["satisfied"] for row in first.values())


@pytest.mark.parametrize("view", ["prefix", "completion"])
@pytest.mark.parametrize("contract", ["hokum", "ashkal"])
@pytest.mark.parametrize("official", [False, True])
def test_native_auction_predicates_match_observations_without_plan_truth(root, rules, monkeypatch, view, contract, official):
    import baloot_platform.research.scenario_predicate_auction_escalation_v2 as lowering_module
    import baloot_platform.research.ortools_scenario_model as model_module
    if official:
        rules = RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml")
    trace = [{"action": action} for action in ("pass", "pass", contract, "pass", "pass", *(["pass"] if contract == "hokum" else []))]
    prefix = trace if view == "completion" else trace[:3]
    arguments = {
        "auction.dealer": [{"seat": "east"}, {"seat": "north"}],
        "auction.buyer": [{"seat": "south"}, {"seat": "north"}],
        "auction.contract": [{"contract": contract}, {"contract": "sun"}],
        "auction.trump": [{"suit": "hearts"}, {"suit": "clubs"}],
        "auction.purchase_round": [{"round": 1}, {"round": 2}],
        "auction.round": [{"round": 1}, {"round": 2}],
        "auction.action_sequence": [{"actions": prefix, "complete": True}, {"actions": [{"action": "sun"}], "complete": True}],
        "auction.legal_actions": [{"actions": [{"action": "pass"}], "complete": True}, {"actions": [{"action": "pass"}, {"action": "sun"}], "complete": True}],
        "auction.pass_count": [{"minimum": len(prefix) - 1, "maximum": len(prefix) - 1}, {"minimum": 0, "maximum": 0}],
        "auction.terminal": [{"terminal": True}, {"terminal": False}],
        "auction.ashkal_buyer_relation": [{"relation": "bidder"}, {"relation": "bidder_partner"}],
    }
    assert set(arguments) == _AUCTION_KINDS_FOR_TEST
    predicates = [_predicate(index, kind, args) for index, (kind, args) in enumerate(
        ((kind, args) for kind, rows in arguments.items() for args in rows), 1)]
    def evaluate(native):
        scenario = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
        attach = scenario.add_native_auction_fragment if native else scenario.add_auction_fragment
        fragment = attach(required_actions=prefix, required_actions_complete=view == "completion", required_purchase_card="HA")
        context = CpSatAuctionPredicateContextV1(scenario.model, fragment, Seat.EAST, view=view)
        rows = [AuctionEscalationPredicateLoweringV1().lower_predicate(predicate, context) for predicate in predicates]
        solver = _solve(scenario.model)
        assert solver.value(context.native["currentActor"]) == (4 if view == "completion" else (3 if contract == "hokum" else 1))
        evidence = [(row.extract(solver)["satisfied"], row.extract(solver)["observed"]) for row in rows]
        if native:
            verifier = cp_model.CpSolver()
            verifier.parameters.num_search_workers = 1
            for row, (satisfied, _) in zip(rows, evidence, strict=True):
                enforced = scenario.model.clone()
                enforced.add(enforced.get_bool_var_from_proto_index(row.literal.index) == 1)
                assert verifier.solve(enforced) == (
                    cp_model.OPTIMAL if satisfied else cp_model.INFEASIBLE), row.kind
        return evidence
    expected = evaluate(False)
    if view == "prefix":
        # Historical evidence accidentally identifies the last pass as bidder
        # when both unresolved contract and buyer are None. Native evidence
        # must not invent a completed buyer identity for a live auction.
        for _, observed in expected:
            if "bidder" in observed:
                observed["bidder"] = None
    def forbidden(*args, **kwargs):
        raise AssertionError("native predicate lowering/extraction used plan truth")
    monkeypatch.setattr(lowering_module, "_auction_values", forbidden)
    monkeypatch.setattr(model_module, "_enumerate_auction", forbidden)
    assert evaluate(True) == expected


def test_auction_context_exports_selected_current_actor_without_history_mutation(
    rules,
):
    model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    fragment = model.add_auction_fragment(prefix_length=0)
    context = CpSatAuctionPredicateContextV1(
        model.model,
        fragment,
        Seat.EAST,
        view="prefix",
    )
    solver = _solve(model.model)
    selected = next(
        plan
        for plan, literal in zip(fragment.plans, fragment.selected, strict=True)
        if solver.boolean_value(literal)
    )
    assert selected.current_actor is not None
    assert solver.value(context.native["currentActor"]) == int(
        selected.current_actor
    )


def test_all_escalation_predicates_are_exact_and_evidence_is_stable(rules):
    first = _escalation_evidence(rules)
    second = _escalation_evidence(rules)
    assert set(first) == set(_ESCALATION_KINDS_FOR_TEST)
    assert first == second
    assert all(row["satisfied"] for row in first.values())


def test_native_escalation_predicates_use_solved_variables(rules, monkeypatch):
    expected = _escalation_evidence(rules)
    actions = expected["escalation.action_sequence"]["observed"]["actions"]
    def forbidden(*args, **kwargs):
        pytest.fail("native predicate used candidate enumeration or Python truth")
    monkeypatch.setattr(OrToolsEscalationStageV1, "_candidates", forbidden)
    monkeypatch.setattr(EscalationDomainIRV1, "transition", forbidden)
    monkeypatch.setattr(EscalationDomainIRV1, "legal_decisions", forbidden)
    first = _escalation_evidence(rules, native=True, actions_override=actions)
    second = _escalation_evidence(rules, native=True, actions_override=actions)
    assert first == second
    assert set(first) == _ESCALATION_KINDS_FOR_TEST
    for kind, row in first.items():
        assert row["satisfied"] is True
        assert row["observed"] == expected[kind]["observed"]


def test_native_escalation_rejects_semantic_counterexample_for_every_predicate(rules):
    false_arguments = {
        "escalation.level": {"level": "triple"},
        "escalation.locked": {"locked": False},
        "escalation.terminal": {"terminal": True},
        "escalation.action_sequence": {
            "actions": [{"actor": "west", "action": "decline"}], "complete": True},
        "escalation.eligible_actor": {"team": "west_east", "seat": "west", "eligible": True},
        "escalation.legal_actions": {"actions": [{"action": "decline"}], "complete": True},
    }
    assert set(false_arguments) == _ESCALATION_KINDS_FOR_TEST
    for index, (kind, arguments) in enumerate(false_arguments.items()):
        stage = OrToolsEscalationStageV1(EscalationDomainIRV1.from_rules(rules.data),
            declarer=Seat.SOUTH, contract=Contract.HOKUM, match_score=(0, 0), native_cp_sat=True)
        result = stage.solve(required_actions=((Seat.WEST, "double", "locked"),
            (Seat.SOUTH, "triple", None), (Seat.WEST, "four", "locked")),
            required_actions_complete=True, required_terminal=False)
        assert result.status == "satisfied"
        context = CpSatEscalationPredicateContextV1.attach(stage)
        context.bind_native_observation(result.observation.to_dict())
        lowering = AuctionEscalationPredicateLoweringV1().lower_predicate(
            _predicate(index, kind, arguments), context)
        solver = _solve(context.model)
        assert not solver.boolean_value(lowering.literal), kind
        assert lowering.extract(solver)["satisfied"] is False, kind
        context.model.add(lowering.literal == 1)
        assert solver.solve(context.model) == cp_model.INFEASIBLE, kind


@pytest.mark.parametrize("mutation", [None, "level", "mode", "resolvedMode", "doubleSeat", "terminal"])
def test_native_escalation_rebinding_rejects_conflicting_observation(rules, monkeypatch, mutation):
    stage = OrToolsEscalationStageV1(EscalationDomainIRV1.from_rules(rules.data),
        declarer=Seat.SOUTH, contract=Contract.HOKUM, match_score=(0, 0), native_cp_sat=True)
    result = stage.solve(required_actions=((Seat.WEST, "double", "locked"), (Seat.SOUTH, "triple", None),
                                          (Seat.WEST, "four", "locked")), required_actions_complete=True,
                         required_terminal=False)
    assert result.status == "satisfied"
    observed = result.observation.to_dict()
    if mutation == "resolvedMode":
        observed["actions"][1]["resolvedMode"] = "locked"
    elif mutation is not None:
        observed[mutation] = {"level": "triple", "mode": "open", "doubleSeat": "east", "terminal": True}[mutation]
    def forbidden(*args, **kwargs):
        pytest.fail("native rebinding used candidate enumeration")
    monkeypatch.setattr(stage, "_candidates", forbidden)
    context = CpSatEscalationPredicateContextV1.attach(stage)
    context.bind_native_observation(observed)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(context.model) == (cp_model.OPTIMAL if mutation is None else cp_model.INFEASIBLE)


@pytest.mark.parametrize("contract", [Contract.SUN, Contract.HOKUM, Contract.ASHKAL])
@pytest.mark.parametrize("prefix_length", range(5))
def test_native_escalation_context_extracts_symbolic_declarer_observation(
    rules, monkeypatch, contract, prefix_length
):
    domain = EscalationDomainIRV1.from_rules(rules.data)
    reference = OrToolsEscalationStageV1(
        domain, declarer=Seat.SOUTH, contract=contract, match_score=(101, 0)
    )
    complete = reference.solve(
        required_level="gahwa" if contract == Contract.HOKUM else "double"
    )
    assert complete.status == "satisfied"
    assert len(complete.observation.actions) == (4 if contract == Contract.HOKUM else 1)
    actions = complete.observation.actions[:prefix_length]
    expected = reference.solve(
        required_actions=tuple(
            (action.actor, action.action, action.requested_mode) for action in actions
        ),
        required_actions_complete=True,
    )
    assert expected.status == "satisfied"

    def forbidden(*args, **kwargs):
        pytest.fail("native extraction used reference escalation evaluation")

    monkeypatch.setattr(OrToolsEscalationStageV1, "_candidates", forbidden)
    monkeypatch.setattr(EscalationDomainIRV1, "transition", forbidden)
    monkeypatch.setattr(EscalationDomainIRV1, "legal_decisions", forbidden)
    model = cp_model.CpModel()
    declarer = model.new_int_var(0, 3, "symbolic_declarer")
    model.add(declarer == int(Seat.SOUTH))
    stage = OrToolsEscalationStageV1(
        domain, declarer=Seat.WEST, contract=contract,
        match_score=(101, 0), native_cp_sat=True,
    )
    context = CpSatEscalationPredicateContextV1.attach(
        stage, model=model, declarer_variable=declarer
    )
    context.bind_native_observation(expected.observation.to_dict())
    solver = _solve(model)
    extracted = context.extract_stage_result(
        solver, model_hash=context.sha256, solver_status="OPTIMAL",
        branches=solver.num_branches, conflicts=solver.num_conflicts,
    )
    assert extracted.observation.to_dict() == expected.observation.to_dict()
    assert solver.value(context.native["lastTeam"]) == (
        int(actions[-1].actor.team) if actions else 2
    )


@pytest.mark.parametrize("contract,score,length", [
    (Contract.HOKUM, (0, 0), length) for length in range(5)
] + [(Contract.SUN, (0, 0), 0), (Contract.SUN, (101, 0), 0),
     (Contract.SUN, (101, 0), 1)])
def test_native_opportunity_path_matches_independent_decisions(rules, monkeypatch, contract, score, length):
    domain = EscalationDomainIRV1.from_rules(rules.data)
    stage = OrToolsEscalationStageV1(domain, declarer=Seat.SOUTH,
        contract=contract, match_score=score, native_cp_sat=True)
    context = CpSatEscalationPredicateContextV1.attach(stage)
    order = (Seat.EAST, Seat.NORTH, Seat.WEST, Seat.SOUTH)
    context.model.add(context.native["length"] == length)
    if length:
        context.model.add(context.native["actionSlots"][0][0] == int(Seat.WEST))
    # Lowering/extraction must not call the independent reference interpreter.
    with monkeypatch.context() as patch:
        def forbidden(*args, **kwargs):
            pytest.fail("native opportunities called a Python semantic evaluator")
        patch.setattr(EscalationDomainIRV1, "transition", forbidden)
        patch.setattr(EscalationDomainIRV1, "legal_decisions", forbidden)
        slots = context.attach_opportunity_path(order=order)
        solver = _solve(context.model)
        rows = context.extract_opportunity_path(solver, slots)
    state = domain.initial_state()
    remaining, public, recall = list(order), [], {seat: [] for seat in Seat}
    for row in rows:
        choices = domain.legal_decisions(state, declarer=Seat.SOUTH,
            contract=contract, match_score=score, include_closure=False)
        eligible = {choice.actor for choice in choices}
        actor = next(seat for seat in remaining if seat in eligible)
        assert row["actor"] == actor.name.lower()
        legal = {choice.code.split(":", 1)[1]: choice for choice in choices if choice.actor == actor}
        assert row["legalActions"] == sorted([*legal, "decline"])
        assert row["publicAnnouncements"] == public
        assert row["ownDeclinedWindows"] == recall[actor]
        if row["selectedAction"] == "decline":
            recall[actor].append(row["windowHash"])
            remaining.remove(actor)
        else:
            choice = legal[row["selectedAction"]]
            state = choice.state
            mode = choice.resolved_mode if choice.action in {"double", "four"} else None
            public.append([actor.name.lower(), choice.action, mode])
            remaining = list(order)
    assert not domain.legal_decisions(state, declarer=Seat.SOUTH, contract=contract,
        match_score=score, actors=remaining, include_closure=False)
    assert len(public) == length
    if length:
        assert rows[0]["actor"] == "east" and rows[0]["selectedAction"] == "decline"
        assert rows[1]["actor"] == "west" and rows[1]["ownDeclinedWindows"] == []


@pytest.mark.parametrize("first", [Seat.EAST, Seat.WEST])
def test_symbolic_opportunity_order_covers_either_opening_actor(rules, first):
    stage = OrToolsEscalationStageV1(EscalationDomainIRV1.from_rules(rules.data),
        declarer=Seat.SOUTH, contract=Contract.HOKUM, match_score=(0, 0), native_cp_sat=True)
    context = CpSatEscalationPredicateContextV1.attach(stage)
    slots = context.attach_opportunity_path()
    context.model.add(context.native["length"] == 0)
    context.model.add(slots[0]["actor"] == int(first))
    rows = context.extract_opportunity_path(_solve(context.model), slots)
    assert [row["actor"] for row in rows] == [first.name.lower(), first.partner.name.lower()]
    assert all(row["selectedAction"] == "decline" and not row["ownDeclinedWindows"]
               and not row["publicAnnouncements"] for row in rows)


def test_opportunity_private_recall_and_official_score_boundary(rules):
    domain = replace(EscalationDomainIRV1.from_rules(rules.data), response_actor_policy="team")
    stage = OrToolsEscalationStageV1(domain, declarer=Seat.SOUTH,
        contract=Contract.HOKUM, match_score=(0, 0), native_cp_sat=True)
    context = CpSatEscalationPredicateContextV1.attach(stage)
    slots = context.attach_opportunity_path(order=(Seat.EAST, Seat.NORTH, Seat.WEST, Seat.SOUTH))
    context.model.add(context.native["length"] == 2)
    for index, actor in enumerate((Seat.WEST, Seat.SOUTH)):
        context.model.add(context.native["actionSlots"][index][0] == int(actor))
    rows = context.extract_opportunity_path(_solve(context.model), slots)
    east = [row for row in rows if row["actor"] == "east"]
    assert len(east) == 2
    assert east[1]["ownDeclinedWindows"] == [east[0]["windowHash"]]
    assert all(not row["ownDeclinedWindows"] for row in rows if row["actor"] == "west")
    for inclusive in (False, True):
        stage = OrToolsEscalationStageV1(replace(domain, sun_lower_score_inclusive=inclusive),
            declarer=Seat.SOUTH, contract=Contract.SUN, match_score=(110, 100), native_cp_sat=True)
        context = CpSatEscalationPredicateContextV1.attach(stage)
        slots = context.attach_opportunity_path(order=tuple(Seat))
        context.model.add(context.native["length"] == 0)
        rows = context.extract_opportunity_path(_solve(context.model), slots)
        assert len(rows) == (2 if inclusive else 0)
        assert all(row["legalActions"] == ["decline", "double:open"] for row in rows)


def test_native_opportunity_order_validation_does_not_mutate_model(rules):
    stage = OrToolsEscalationStageV1(EscalationDomainIRV1.from_rules(rules.data),
        declarer=Seat.SOUTH, contract=Contract.HOKUM, match_score=(0, 0), native_cp_sat=True)
    context = CpSatEscalationPredicateContextV1.attach(stage)
    before = str(context.model.proto)
    for order in ((0, 0, 1, 2), (True, 1, 2, 3), "north", (0, 1, 2)):
        with pytest.raises(ValueError):
            context.attach_opportunity_path(order=order)
        assert str(context.model.proto) == before
    context.attach_opportunity_path()
    attached = str(context.model.proto)
    with pytest.raises(ValueError, match="already attached"):
        context.attach_opportunity_path()
    assert str(context.model.proto) == attached


def test_strict_v1_schemas_and_context_types_fail_before_lowering(rules):
    scenario = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    fragment = scenario.add_auction_fragment()
    context = CpSatAuctionPredicateContextV1(scenario.model, fragment, Seat.EAST)
    lowerer = AuctionEscalationPredicateLoweringV1()
    with pytest.raises(ValueError, match="unknown=.*oracle"):
        lowerer.lower_predicate(
            _predicate(18, "auction.dealer", {"seat": "east", "oracle": True}),
            context,
        )
    with pytest.raises(TypeError, match="must be an integer"):
        lowerer.lower_predicate(
            _predicate(19, "auction.purchase_round", {"round": True}),
            context,
        )
    with pytest.raises(TypeError, match="escalation context"):
        lowerer.lower_predicate(
            _predicate(20, "escalation.level", {"level": "normal"}),
            context,
        )


def test_new_lowering_has_no_engine_bot_or_runtime_verifier_imports():
    import baloot_platform.research.scenario_predicate_auction_escalation_v2 as module

    source = inspect.getsource(module)
    forbidden = (
        "BalootEngine",
        "HeuristicBot",
        "baloot_platform.core.engine",
        "scenario_evidence",
        "ScenarioRuntimeVerifier",
        "proof_service",
    )
    assert not any(name in source for name in forbidden)


_AUCTION_KINDS_FOR_TEST = {
    registration.kind
    for registration in AUCTION_ESCALATION_PREDICATE_REGISTRATIONS_V1
    if registration.stage == "auction"
}
_ESCALATION_KINDS_FOR_TEST = {
    registration.kind
    for registration in AUCTION_ESCALATION_PREDICATE_REGISTRATIONS_V1
    if registration.stage == "escalation"
}
