from __future__ import annotations

import copy
import inspect
import math
from dataclasses import replace
from itertools import product
from ortools.sat.python import cp_model

import pytest

import baloot_platform.research.ortools_research_executor as research_executor
from baloot_platform.core.escalation_ir import EscalationDomainIRV1, EscalationStateV1
from baloot_platform.core.enums import Contract, Seat, Team
from baloot_platform.research.ortools_escalation_stage import (
    OrToolsEscalationStageV1,
    attach_native_escalation_transition_v1,
)
from baloot_platform.research.ortools_research_executor import (
    GenericCompleteTraceCompilerV1,
)
from baloot_platform.research.reference_escalation import (
    ReferenceEscalationInterpreterV1,
)
from baloot_platform.research.scenario_contracts import ResearchScenarioV1


@pytest.mark.parametrize("contract", tuple(Contract))
@pytest.mark.parametrize("score", [(0, 0), (101, 99), (101, 100)])
@pytest.mark.parametrize("inclusive", [False, True])
@pytest.mark.parametrize("variant", ["default", "team_locked", "disabled"])
def test_native_escalation_transition_exhaustive_parity(rules, monkeypatch, contract, score, inclusive, variant):
    domain = replace(EscalationDomainIRV1.from_rules(rules.data), sun_lower_score_inclusive=inclusive)
    if variant == "team_locked":
        domain = replace(domain, response_actor_policy="team", default_mode="locked")
    elif variant == "disabled":
        domain = replace(domain, enabled=False)
    levels = ("normal", "double", "triple", "four", "gahwa")
    expected = set()
    for level, locked, doubler, actor, action_index in product(range(5), range(2), range(5), range(4), range(4)):
        action = domain.supported_actions[action_index]
        for mode_index in ((0, 1, 2) if action in {"double", "four"} else (0,)):
            try:
                outcome = domain.transition(EscalationStateV1(levels[level], bool(locked), Seat(doubler) if doubler < 4 else None),
                    declarer=Seat.SOUTH, contract=contract, match_score=score, actor=Seat(actor),
                    action=action, mode=(None, "open", "locked")[mode_index]).state
            except ValueError:
                continue
            expected.add((level, locked, doubler, actor, action_index, mode_index,
                          levels.index(outcome.level), int(outcome.locked), int(outcome.double_seat) if outcome.double_seat is not None else 4))
    def forbidden(*args, **kwargs):
        pytest.fail("native escalation lowering called the reference transition")
    monkeypatch.setattr(EscalationDomainIRV1, "transition", forbidden)
    model = cp_model.CpModel()
    before = (model.new_int_var(0, 4, "level"), model.new_bool_var("locked"), model.new_int_var(0, 4, "doubler"))
    after = (model.new_int_var(0, 4, "next_level"), model.new_bool_var("next_locked"), model.new_int_var(0, 4, "next_doubler"))
    actor, action, mode = (model.new_int_var(0, 4, "actor"), model.new_int_var(0, 4, "action"), model.new_int_var(0, 2, "mode"))
    attach_native_escalation_transition_v1(model, domain, contract=contract,
        declarer=model.new_constant(int(Seat.SOUTH)), match_score=tuple(model.new_constant(value) for value in score),
        active=model.new_constant(1), actor=actor, action=action, requested_mode=mode,
        before=before, after=after, name="native")
    variables = (*before, actor, action, mode, *after)
    class Collect(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.rows = set()
        def on_solution_callback(self):
            self.rows.add(tuple(self.value(value) for value in variables))
    collector = Collect()
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.enumerate_all_solutions = True
    status = solver.solve(model, collector)
    assert status == (cp_model.OPTIMAL if expected else cp_model.INFEASIBLE)
    assert collector.rows == expected


def _stage(rules, *, contract=Contract.HOKUM, score=(0, 0), declarer=Seat.SOUTH):
    return OrToolsEscalationStageV1(
        EscalationDomainIRV1.from_rules(rules.data),
        declarer=declarer,
        contract=contract,
        match_score=score,
    )


def test_native_escalation_stage_cancel_before_solver_retains_incomplete(rules, monkeypatch):
    monkeypatch.setattr(cp_model.CpSolver, "solve",
        lambda *args, **kwargs: pytest.fail("already cancelled model must not solve"))
    checks = iter((False, True))
    result = _stage(rules).solve(maximum_seconds=5, cancel_check=lambda: next(checks))
    assert result.status == "incomplete"
    assert result.solver_status == "unknown"
    assert result.branches == result.conflicts == 0


@pytest.mark.parametrize("contract", tuple(Contract))
@pytest.mark.parametrize("length", [0, 1, 2, 3, 4])
def test_native_escalation_stage_matches_legacy_without_interpreter(rules, monkeypatch, contract, length):
    domain = EscalationDomainIRV1.from_rules(rules.data)
    actions = ((Seat.WEST, "double", "locked"), (Seat.SOUTH, "triple", None),
               (Seat.WEST, "four", "open"), (Seat.SOUTH, "gahwa", None))[:length]
    kwargs = dict(required_actions=actions, required_actions_complete=True)
    expected = OrToolsEscalationStageV1(domain, declarer=Seat.SOUTH, contract=contract,
                                       match_score=(101, 99)).solve(**kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("native stage used Python escalation truth")
    monkeypatch.setattr(EscalationDomainIRV1, "transition", forbidden)
    monkeypatch.setattr(EscalationDomainIRV1, "legal_decisions", forbidden)
    actual = OrToolsEscalationStageV1(domain, declarer=Seat.SOUTH, contract=contract,
                                     match_score=(101, 99), native_cp_sat=True).solve(**kwargs)
    assert actual.status == expected.status
    assert (actual.observation.to_dict() if actual.observation else None) == (
        expected.observation.to_dict() if expected.observation else None)


def _search_escalation_terminal_scenario(terminal: bool) -> ResearchScenarioV1:
    return ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "predicateId": "PRED-ESCALATION-TERMINAL",
                "kind": "escalation.terminal",
                "version": 1,
                "hardness": "hard",
                "arguments": {"terminal": terminal},
            },
            "decisionPoint": {"kind": "play", "selector": {"trickIndex": 6}},
            "outcome": {
                "predicateId": "PRED-REPORT-ROUND",
                "kind": "outcome.team_round",
                "version": 1,
                "hardness": "report",
                "arguments": {"team": "north_south", "status": "won"},
            },
        }
    )


def test_exact_sequence_derives_state_response_eligibility_and_terminal(rules):
    stage = _stage(rules)
    four = stage.solve(
        required_actions=(
            (Seat.WEST, "double", None),
            (Seat.SOUTH, "triple", None),
            (Seat.WEST, "four", "locked"),
        ),
        required_actions_complete=True,
        required_level="four",
        required_mode="locked",
        required_eligible_seat=Seat.SOUTH,
        required_eligible_team=Team.NORTH_SOUTH,
        required_response="gahwa",
        required_terminal=False,
    )
    assert four.status == "satisfied"
    assert four.observation is not None
    assert four.observation.level == "four"
    assert four.observation.mode == "locked"
    assert four.observation.double_seat == Seat.WEST
    assert four.observation.eligible_seats == (Seat.SOUTH,)
    assert four.observation.eligible_teams == (Team.NORTH_SOUTH,)
    assert four.observation.legal_next_actions == ("decline", "gahwa")
    assert [choice.code for choice in four.observation.legal_next_choices] == [
        "south:gahwa"
    ]

    terminal = stage.solve(
        required_actions=(
            (Seat.WEST, "double", None),
            (Seat.SOUTH, "triple", None),
            (Seat.WEST, "four", "locked"),
            (Seat.SOUTH, "gahwa", None),
        ),
        required_actions_complete=True,
        required_terminal=True,
    )
    assert terminal.status == "satisfied"
    assert terminal.observation is not None
    assert terminal.observation.terminal is True
    assert terminal.observation.legal_next_actions == ("decline",)
    assert terminal.observation.legal_next_choices == ()


def test_partial_sequence_can_lift_to_a_required_later_state(rules):
    result = _stage(rules).solve(
        required_actions=((Seat.WEST, "double", None),),
        required_actions_complete=False,
        required_level="gahwa",
    )
    assert result.status == "satisfied"
    assert result.observation is not None
    assert [action.action for action in result.observation.actions] == [
        "double",
        "triple",
        "four",
        "gahwa",
    ]


def test_downstream_boundary_can_close_escalation_by_declining(rules):
    result = _stage(rules).solve(
        required_actions=(),
        required_actions_complete=True,
        required_terminal=True,
    )

    assert result.status == "satisfied"
    assert result.observation is not None
    assert result.observation.actions == ()
    assert result.observation.level == "normal"
    assert result.observation.terminal is True


@pytest.mark.parametrize("second", ["decline", "double:open"])
def test_opportunity_event_replay_derives_private_and_public_state(rules, second):
    from baloot_platform.research.scenario_contracts import canonical_sha256
    events = [{"actor": "west", "actionCode": "decline"}, {"actor": "east", "actionCode": second}]
    stage = _stage(rules)
    replay = stage.replay_opportunities(required_actions=(), events=events, maximum_seconds=5)
    assert replay["status"] == "satisfied"
    assert replay["completedEvents"] == 2
    assert replay["sha256"] == stage.replay_opportunities(required_actions=(), events=events, maximum_seconds=5)["sha256"]
    assert replay["sha256"] == canonical_sha256({key: value for key, value in replay.items() if key != "sha256"})
    assert replay["transitions"][0]["event"]["visibility"] == "private"
    final = replay["finalState"]
    assert len(final["observation"]["actions"]) == (0 if second == "decline" else 1)
    assert final["opportunityState"]["windowClosed"] is (second == "decline")
    assert final["opportunityState"]["nonAnnouncingSeats"] == (["west", "east"] if second == "decline" else [])
    expected = [] if second == "decline" else [{"actor": "east", "action": "double", "mode": "open"}]
    reconciled = stage.replay_opportunities(required_actions=(), events=events, maximum_seconds=5,
                                          expected_final_actions=expected)
    assert reconciled["status"] == "satisfied"
    assert "expectedFinalActions" in reconciled
    assert reconciled["sha256"] != replay["sha256"]
    wrong = [{"actor": "east", "action": "double", "mode": "open"}] if not expected else []
    with pytest.raises(ValueError, match="disagree with public escalation"):
        stage.replay_opportunities(required_actions=(), events=events, maximum_seconds=5,
                                   expected_final_actions=wrong)
    events[0]["actor"] = "north"
    assert replay["events"][0]["actor"] == "west"


@pytest.mark.parametrize("count,next_actor", [(0, "west"), (1, "east"), (2, None)])
def test_ordered_opportunity_replay_binds_next_actor(rules, count, next_actor):
    stage = _stage(rules)
    events = [{"actor": seat, "actionCode": "decline"} for seat in ("west", "east")][:count]
    order = ("west", "east", "south", "north")
    result = stage.replay_opportunities(required_actions=(), events=events,
        maximum_seconds=5, opportunity_order=order, expected_final_actions=())
    assert result["status"] == "satisfied"
    assert result["nextActor"] == next_actor
    assert result["opportunityOrder"] == list(order)
    legacy = stage.replay_opportunities(required_actions=(), events=events, maximum_seconds=5)
    assert "nextActor" not in legacy and "opportunityOrder" not in legacy
    assert result["finalState"] == legacy["finalState"]


@pytest.mark.parametrize("order", ["west", (), ("west",) * 4, (True, "east", "south", "north")])
def test_opportunity_replay_rejects_invalid_order(rules, order):
    with pytest.raises(ValueError, match="opportunity order"):
        _stage(rules).replay_opportunities(required_actions=(), events=[], maximum_seconds=5,
            opportunity_order=order)


def test_opportunity_replay_rejects_legal_actor_out_of_declared_order(rules):
    with pytest.raises(ValueError, match="disagrees with declared order"):
        _stage(rules).replay_opportunities(required_actions=(),
            events=[{"actor": "east", "actionCode": "decline"}], maximum_seconds=5,
            opportunity_order=("west", "east", "south", "north"))


@pytest.fixture
def recorded_escalation_prefix():
    prefix = {"schemaVersion": 1, "deckOrder": list(range(32)), "dealer": "east",
        "roundNumber": 1, "matchScore": [0, 0],
        "auctionActions": [{"action": action} for action in ("hokum", "pass", "pass", "pass")],
        "escalationActions": [], "projectActions": [], "projectDeclarations": [],
        "projectAssertions": [], "playActions": [], "stateAssertions": {}}
    options = {"recorded_escalation_opportunities": [{"actor": "west", "actionCode": "decline"}],
        "opportunity_order": ("west", "east", "south", "north")}
    return prefix, options


def test_exact_escalation_normalization_binds_recorded_private_history(rules, recorded_escalation_prefix):
    from baloot_platform.core.exact_state_view import build_canonical_observer_information_v1
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1

    prefix, options = recorded_escalation_prefix
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    result = normalizer.normalize(prefix, decision_point={"kind": "escalation"}, **options)
    assert result.next_actor == "east"
    assert result.legal_actions
    assert result.state["escalationHistory"] == []
    assert result.state["escalationOpportunityHistoryHash"]
    preview = normalizer.preview_legal_next_actions(prefix, **options)
    assert preview["currentStage"] == "escalation"
    assert preview["nextActor"] == "east"
    assert {row["code"] for row in preview["legalActions"]} == {row["code"] for row in result.legal_actions}
    assert all(row["stage"] == "escalation" and row["actor"] == "east" for row in preview["legalActions"])
    closed = normalizer.preview_legal_next_actions(prefix, **{**options,
        "recorded_escalation_opportunities": [
            {"actor": actor, "actionCode": "decline"} for actor in ("west", "east")]})
    assert closed["currentStage"] != "escalation"
    assert normalizer.normalize(prefix, decision_point={"kind": "escalation"}, **options).sha256 == result.sha256
    public_only = normalizer.normalize(prefix,
        decision_point={"kind": "escalation", "selector": {"actingSeat": "east"}})
    assert public_only.exact_state_id != result.exact_state_id
    # Private history binds the exact state, not another player's observation.
    observer = build_canonical_observer_information_v1(
        complete_state=result.state, acting_seat="east")
    public_observer = build_canonical_observer_information_v1(
        complete_state=public_only.state, acting_seat="east")
    assert observer == public_observer
    assert "escalationOpportunityHistoryHash" not in observer["publicState"]
    from baloot_platform.research.ortools_nonplay_graphs import compile_escalation_decision_graph_v1
    from baloot_platform.research.scenario_contracts import canonical_sha256

    replay = result.normalization_receipt.to_dict()["derivedEscalationOpportunities"]
    eligible = replay["finalState"]["observation"]["eligibleSeats"]
    graph = compile_escalation_decision_graph_v1(EscalationDomainIRV1.from_rules(rules.data),
        declarer=result.state["declarer"], contract=result.state["contract"], match_score=(0, 0),
        acting_seat=result.next_actor, native_cp_sat=True, terminal_utility=lambda _terminal: 0,
        opportunity_orders={canonical_sha256({"actions": []}): tuple(
            seat for seat in replay["opportunityOrder"] if seat in eligible)},
        prior_non_announcing_seats=replay["finalState"]["opportunityState"]["nonAnnouncingSeats"])
    root = next(node for node in graph.nodes if node.node_id == graph.root_node_id)
    assert root.actor_seat == result.next_actor
    assert {edge.action_code for edge in root.edges} == {row["code"] for row in result.legal_actions}
    with pytest.raises(ValueError, match="acting seat disagrees"):
        normalizer.normalize(prefix,
            decision_point={"kind": "escalation", "selector": {"actingSeat": "west"}}, **options)
    with pytest.raises(ValueError, match="events and opportunity order"):
        normalizer.normalize(prefix, decision_point={"kind": "escalation"},
            recorded_escalation_opportunities=[])
    with pytest.raises(ValueError, match="recorded escalation history is infeasible"):
        normalizer.normalize(prefix, decision_point={"kind": "escalation"},
            **{**options, "recorded_escalation_opportunities": [
                {"actor": "west", "actionCode": "triple"}]})


@pytest.mark.controlled_clock
def test_exact_escalation_preview_caps_replay_allowance_under_clock_rounding(
    rules, monkeypatch, recorded_escalation_prefix,
):
    import baloot_platform.research.ortools_exact_state as exact_state
    from deterministic_work import set_monotonic

    prefix, options = recorded_escalation_prefix
    declared = 15.0
    origin = math.nextafter(1024.0, 0.0)
    assert (origin + declared) - origin > declared
    set_monotonic(monkeypatch, exact_state, origin)
    replay = OrToolsEscalationStageV1.replay_opportunities
    grants = []

    def replay_after_clock_advance(self, **kwargs):
        grants.append((kwargs["maximum_seconds"], kwargs["declared_maximum_seconds"]))
        set_monotonic(monkeypatch, exact_state, origin + 4.0)
        return replay(self, **kwargs)

    monkeypatch.setattr(OrToolsEscalationStageV1, "replay_opportunities", replay_after_clock_advance)
    normalizer = exact_state.OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    preview = normalizer.preview_legal_next_actions(prefix, maximum_seconds=declared, **options)

    assert preview["currentStage"] == "escalation"
    assert preview["nextActor"] == "east"
    assert preview["legalActions"]
    # The nested normalization gets the remainder after four virtual seconds.
    assert grants == [(declared, declared), (11.0, 11.0)]


@pytest.mark.real_clock
@pytest.mark.parametrize("reason", ["cancelled", "maximum_seconds", None])
def test_exact_escalation_retains_incomplete_replay_receipt(rules, monkeypatch, reason):
    from baloot_platform.research.ortools_exact_state import (
        ExactNormalizationStopped, OrToolsExactStateNormalizerV1)
    from baloot_platform.research.scenario_contracts import canonical_sha256

    stage = _stage(rules)
    replay = stage.replay_opportunities(required_actions=(), events=[], maximum_seconds=5,
        cancel_check=lambda: True, opportunity_order=("west", "east", "south", "north"))
    if reason is None:
        replay.pop("stopReason")
    else:
        replay["stopReason"] = reason
    replay["sha256"] = canonical_sha256({key: value for key, value in replay.items() if key != "sha256"})
    monkeypatch.setattr(OrToolsEscalationStageV1, "replay_opportunities", lambda *_args, **_kwargs: replay)
    prefix = {"schemaVersion": 1, "deckOrder": list(range(32)), "dealer": "east",
        "roundNumber": 1, "matchScore": [0, 0],
        "auctionActions": [{"action": action} for action in ("hokum", "pass", "pass", "pass")],
        "escalationActions": [], "projectActions": [], "playActions": []}
    with pytest.raises(ExactNormalizationStopped) as stopped:
        OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix,
            decision_point={"kind": "escalation"}, recorded_escalation_opportunities=[],
            opportunity_order=("west", "east", "south", "north"))
    receipt = stopped.value.partial_receipt
    assert receipt["status"] == "incomplete"
    assert receipt["exhaustionReason"] == (reason or "escalation_replay_incomplete")
    assert receipt["evidence"]["escalationOpportunityReplay"] == replay
    assert receipt["sha256"] == canonical_sha256({key: value for key, value in receipt.items() if key != "sha256"})


def test_opportunity_event_replay_retains_prefix_at_illegal_or_cancelled_event(rules):
    stage = _stage(rules)
    events = [{"actor": "west", "actionCode": "decline"}] * 2
    failed = stage.replay_opportunities(required_actions=(), events=events, maximum_seconds=5)
    assert failed["status"] == "infeasible"
    assert failed["completedEvents"] == 1
    assert failed["finalState"] == failed["transitions"][0]["after"]
    stopped = stage.replay_opportunities(required_actions=(), events=events, maximum_seconds=5, cancel_check=lambda: True)
    assert stopped["status"] == "incomplete" and stopped["stopReason"] == "cancelled"
    assert stopped["completedEvents"] == 0 and stopped["transitions"] == []


@pytest.mark.parametrize("events", [[{"actor": "west", "actionCode": "decline", "visibility": "public"}],
    [{"actor": True, "actionCode": "decline"}], "decline", [{"actor": "west", "actionCode": "decline"}] * 100])
def test_opportunity_event_replay_rejects_untrusted_state_or_unbounded_input(rules, events):
    with pytest.raises(ValueError):
        _stage(rules).replay_opportunities(required_actions=(), events=events, maximum_seconds=5)


def test_opportunity_replay_shares_one_deadline_and_retains_completed_prefix(rules, monkeypatch):
    import baloot_platform.research.ortools_escalation_stage as module
    stage = _stage(rules)
    original = stage.transition_opportunity
    origin = math.nextafter(8.0, 0.0)
    clock, calls = [origin], []
    def transition(**kwargs):
        calls.append(kwargs)
        result = original(**kwargs)
        clock[0] = origin + 6
        return result
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(stage, "transition_opportunity", transition)
    result = stage.replay_opportunities(required_actions=(), maximum_seconds=5,
        events=[{"actor": "west", "actionCode": "decline"}, {"actor": "east", "actionCode": "double:open"}])
    assert len(calls) == result["completedEvents"] == 1
    assert calls[0]["maximum_seconds"] == calls[0]["declared_maximum_seconds"] == 5
    assert result["status"] == "incomplete" and result["stopReason"] == "maximum_seconds"
    assert result["finalState"] == result["transitions"][0]["after"]


def test_native_opportunity_transition_records_private_declines_and_public_announcements(rules):
    from baloot_platform.research.scenario_contracts import canonical_sha256
    stage = _stage(rules)
    kwargs = dict(required_actions=(), actor=Seat.WEST, action_code="decline", maximum_seconds=5)
    first = stage.transition_opportunity(**kwargs, non_announcements=())
    assert first["status"] == "satisfied"
    assert first["event"]["visibility"] == "private"
    assert first["after"]["observation"]["actions"] == []
    declines = first["after"]["opportunityState"]["nonAnnouncingSeats"]
    assert declines == ["west"]
    assert stage.transition_opportunity(**kwargs, non_announcements=declines)["status"] == "infeasible"
    closed = stage.transition_opportunity(**{**kwargs, "actor": Seat.EAST}, non_announcements=declines)
    assert closed["after"]["observation"]["terminal"] is True
    assert closed["after"]["observation"]["actions"] == []
    announced = stage.transition_opportunity(**{**kwargs, "actor": Seat.EAST, "action_code": "double:open"},
        non_announcements=declines)
    assert announced["status"] == "satisfied"
    assert announced["event"]["visibility"] == "public"
    assert announced["after"]["opportunityState"]["nonAnnouncingSeats"] == []
    assert announced["after"]["opportunityState"]["remainingEligibleSeats"] == ["south"]
    assert announced["after"]["observation"]["actions"][0]["actor"] == "east"
    assert first["sha256"] == stage.transition_opportunity(**kwargs, non_announcements=())["sha256"]
    assert first["sha256"] == canonical_sha256({key: value for key, value in first.items() if key != "sha256"})


def test_native_opportunity_transition_retains_cancelled_receipt(rules):
    result = _stage(rules).transition_opportunity(required_actions=(), non_announcements=(),
        actor=Seat.WEST, action_code="decline", maximum_seconds=5, cancel_check=lambda: True)
    assert result["status"] == "incomplete"
    assert result["stopReason"] == "cancelled"
    assert result["before"] is None and result["after"] is None


@pytest.mark.real_clock
def test_native_opportunity_transition_shares_wall_budget_and_retains_before(rules, monkeypatch):
    import baloot_platform.research.ortools_escalation_stage as module
    stage = _stage(rules)
    original = stage.solve
    clock = [0.0]
    calls = []
    def solve(**kwargs):
        calls.append(kwargs)
        result = original(**kwargs)
        clock[0] = 2.0
        return result
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(stage, "solve", solve)
    result = stage.transition_opportunity(required_actions=(), non_announcements=(),
        actor=Seat.WEST, action_code="decline", maximum_seconds=1)
    assert len(calls) == 1
    assert result["status"] == "incomplete"
    assert result["stopReason"] == "maximum_seconds"
    assert result["limits"]["maximumSeconds"] == 1
    assert result["before"]["status"] == "satisfied"
    assert result["after"] is None


def test_native_opportunity_transition_rejects_illegal_announcement(rules):
    result = _stage(rules).transition_opportunity(required_actions=(), non_announcements=(),
        actor=Seat.WEST, action_code="triple", maximum_seconds=5)
    assert result["before"]["status"] == "satisfied"
    assert result["after"]["status"] == result["status"] == "infeasible"


@pytest.mark.parametrize("code", ["decline:open", "gahwa:locked", "double:unsafe", "unknown", "double:open:extra"])
def test_native_opportunity_transition_rejects_unknown_syntax(rules, code):
    with pytest.raises(ValueError):
        _stage(rules).transition_opportunity(required_actions=(), non_announcements=(),
            actor=Seat.WEST, action_code=code, maximum_seconds=5)


def test_private_opportunities_close_only_after_all_eligible_seats_decline(rules):
    stage = _stage(rules)
    def solve(seats):
        return stage.solve(required_actions_complete=True, opportunity_non_announcements=seats)

    opened, one, both = solve(()), solve((Seat.WEST,)), solve((Seat.WEST, Seat.EAST))
    assert all(item.status == "satisfied" for item in (opened, one, both))
    assert opened.observation.to_dict() == one.observation.to_dict()
    assert one.opportunity_state["remainingEligibleSeats"] == ["east"]
    assert all(choice["actor"] == "east" for choice in one.opportunity_state["remainingChoices"])
    assert both.observation.terminal is True
    assert both.observation.actions == ()
    assert both.opportunity_state["remainingChoices"] == []
    assert both.opportunity_state["remainingEligibleSeats"] == []
    reverse = solve((Seat.EAST, Seat.WEST))
    assert reverse.model_hash == both.model_hash
    assert reverse.evidence_hash == both.evidence_hash
    assert one.evidence_hash != opened.evidence_hash
    view = one.to_dict()
    view["opportunityState"]["remainingEligibleSeats"].clear()
    assert one.opportunity_state["remainingEligibleSeats"] == ["east"]
    assert "opportunityState" not in stage.solve(required_actions_complete=True).to_dict()


@pytest.mark.parametrize("seats,terminal", [((Seat.NORTH,), None), ((Seat.WEST,), True), ((Seat.WEST, Seat.EAST), False)])
def test_opportunity_constraints_reject_ineligible_or_premature_closure(rules, seats, terminal):
    result = _stage(rules).solve(required_actions_complete=True,
        opportunity_non_announcements=seats, required_terminal=terminal)
    assert result.status == "infeasible"


def test_opportunities_use_current_response_window_not_previous_actors(rules):
    stage = _stage(rules)
    kwargs = dict(required_actions=((Seat.WEST, "double", "open"),), required_actions_complete=True)
    closed = stage.solve(**kwargs, opportunity_non_announcements=(Seat.SOUTH,))
    assert closed.status == "satisfied"
    assert closed.observation.terminal is True
    assert closed.observation.double_seat == Seat.WEST
    assert closed.observation.mode == "open"
    assert len(closed.observation.actions) == 1
    assert stage.solve(**kwargs, opportunity_non_announcements=(Seat.WEST,)).status == "infeasible"
    sun = _stage(rules, contract=Contract.SUN)
    assert sun.solve(required_actions_complete=True, opportunity_non_announcements=()).observation.terminal
    assert sun.solve(required_actions_complete=True, opportunity_non_announcements=(Seat.WEST,)).status == "infeasible"


@pytest.mark.parametrize("seats,error", [("west", TypeError), ((True,), TypeError), ((1.0,), TypeError), ((Seat.WEST, "west"), ValueError)])
def test_opportunity_input_is_strict(rules, seats, error):
    with pytest.raises(error):
        _stage(rules).solve(required_actions_complete=True, opportunity_non_announcements=seats)


def test_opportunities_require_fixed_unfiltered_prefix(rules):
    with pytest.raises(ValueError):
        _stage(rules).solve(opportunity_non_announcements=())
    with pytest.raises(ValueError):
        _stage(rules).solve(required_actions_complete=True, opportunity_non_announcements=(),
            allowed_candidate_fingerprints=())


def test_exact_prefix_reports_only_legal_next_response(rules):
    result = _stage(rules).solve(
        required_actions=(
            {"actor": "west", "action": "double", "requestedMode": "open"},
        ),
        required_actions_complete=True,
    )
    assert result.status == "satisfied"
    assert result.observation is not None
    assert result.observation.level == "double"
    assert result.observation.mode == "open"
    assert result.observation.legal_next_actions == ("decline", "triple")
    assert result.observation.eligible_seats == (Seat.SOUTH,)


def test_team_actor_policy_exposes_both_eligible_teammates(rules):
    data = copy.deepcopy(rules.data)
    data["escalation"]["response_actor_policy"] = "team"
    domain = EscalationDomainIRV1.from_rules(data)
    result = OrToolsEscalationStageV1(
        domain,
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
    ).solve(
        required_actions=((Seat.WEST, "double", None),),
        required_actions_complete=True,
        required_response="triple",
    )
    assert result.status == "satisfied"
    assert result.observation is not None
    assert result.observation.eligible_seats == (Seat.NORTH, Seat.SOUTH)
    assert result.observation.eligible_teams == (Team.NORTH_SOUTH,)


def test_sun_threshold_and_terminal_double_are_exact(rules):
    feasible = _stage(rules, contract=Contract.SUN, score=(101, 50)).solve(
        required_level="double",
        required_terminal=True,
    )
    blocked = _stage(rules, contract=Contract.SUN, score=(80, 60)).solve(
        required_level="double"
    )
    assert feasible.status == "satisfied"
    assert feasible.observation is not None
    assert feasible.observation.terminal is True
    assert feasible.observation.eligible_seats == ()
    assert blocked.status == "infeasible"
    assert blocked.observation is None


def test_illegal_exact_actor_sequence_is_infeasible(rules):
    result = _stage(rules).solve(
        required_actions=(
            (Seat.WEST, "double", None),
            (Seat.NORTH, "triple", None),
        ),
        required_actions_complete=True,
    )
    assert result.status == "infeasible"


def test_cp_sat_observation_matches_reference_for_microfixtures(rules):
    domain = EscalationDomainIRV1.from_rules(rules.data)
    reference = ReferenceEscalationInterpreterV1(domain)
    stage = OrToolsEscalationStageV1(
        domain,
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
    )
    fixtures = (
        (),
        ((Seat.WEST, "double", "locked"),),
        (
            (Seat.WEST, "double", None),
            (Seat.SOUTH, "triple", None),
        ),
        (
            (Seat.WEST, "double", None),
            (Seat.SOUTH, "triple", None),
            (Seat.WEST, "four", "open"),
            (Seat.SOUTH, "gahwa", None),
        ),
    )
    for actions in fixtures:
        expected = reference.apply_sequence(
            declarer=Seat.SOUTH,
            contract=Contract.HOKUM,
            match_score=(0, 0),
            actions=actions,
        )
        result = stage.solve(
            required_actions=actions,
            required_actions_complete=True,
        )
        assert result.status == "satisfied"
        assert result.observation is not None
        assert result.observation.level == expected.state.level
        assert result.observation.mode == (
            "locked" if expected.state.locked else "open"
        )
        assert result.observation.double_seat == expected.state.double_seat


def test_stage_is_deterministic_and_uses_one_worker(rules, monkeypatch):
    observed_workers: list[int] = []
    from ortools.sat.python import cp_model

    original = cp_model.CpSolver.solve

    def capture(solver, model, *args, **kwargs):
        observed_workers.append(solver.parameters.num_search_workers)
        return original(solver, model, *args, **kwargs)

    monkeypatch.setattr(cp_model.CpSolver, "solve", capture)
    stage = _stage(rules)
    first = stage.solve(required_level="four", seed=23)
    second = stage.solve(required_level="four", seed=23)
    assert first == second
    assert first.model_hash == second.model_hash
    assert first.evidence_hash == second.evidence_hash
    assert observed_workers == [1, 1]


def test_stage_rejects_coercion_and_unknown_fields(rules):
    stage = _stage(rules)
    with pytest.raises(TypeError, match="must be boolean"):
        stage.solve(required_actions_complete=1)
    with pytest.raises(ValueError, match="unknown fields"):
        stage.solve(
            required_actions=(
                {"actor": "west", "action": "double", "oracle": True},
            )
        )
    with pytest.raises(ValueError, match="only Double or Four"):
        stage.solve(required_actions=((Seat.SOUTH, "triple", "locked"),))


def test_new_escalation_stage_has_no_engine_bot_or_runtime_import():
    import baloot_platform.research.ortools_escalation_stage as module

    source = inspect.getsource(module)
    forbidden = (
        "BalootEngine",
        "HeuristicBot",
        "baloot_platform.core.engine",
        "scenario_evidence",
        "proof_service",
    )
    assert not any(name in source for name in forbidden)


@pytest.mark.parametrize("native", [False, True])
def test_complete_trace_requires_terminal_escalation_before_projects_and_play(rules, monkeypatch, native):
    scenario = _search_escalation_terminal_scenario(False)
    if native:
        def forbidden(*args, **kwargs):
            pytest.fail("native construction enumerated escalation candidates")
        monkeypatch.setattr(OrToolsEscalationStageV1, "_candidates", forbidden)
        monkeypatch.setattr(research_executor, "_eligible_escalation_fingerprints", forbidden)

    attempt = GenericCompleteTraceCompilerV1(rules=rules, native_cp_sat_scoring=native)(
        scenario,
        (),
        seed=0,
        maximum_seconds=5,
    )

    assert attempt.status == "infeasible"
    assert attempt.reason == ("deal_auction_infeasible" if native else "escalation_infeasible")


def test_native_complete_trace_lowers_escalation_before_solving(rules, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("native construction used interpreted escalation truth")
    attached_projects = []
    original_scoring = research_executor.attach_scoring_fragment_v1

    def capture_scoring(model, domain, inputs, **kwargs):
        attached_projects.append(inputs.project_points)
        return original_scoring(model, domain, inputs, **kwargs)

    monkeypatch.setattr(research_executor, "attach_scoring_fragment_v1", capture_scoring)
    monkeypatch.setattr(OrToolsEscalationStageV1, "_candidates", forbidden)
    monkeypatch.setattr(EscalationDomainIRV1, "transition", forbidden)
    monkeypatch.setattr(EscalationDomainIRV1, "legal_decisions", forbidden)
    monkeypatch.setattr(EscalationDomainIRV1, "can_close", forbidden)
    monkeypatch.setattr(research_executor, "_eligible_escalation_fingerprints", forbidden)
    attempt = GenericCompleteTraceCompilerV1(rules=rules, native_cp_sat_scoring=True)(
        _search_escalation_terminal_scenario(True), (), seed=0, maximum_seconds=5)
    assert attempt.status == "feasible", attempt.reason
    assert attempt.candidate.trace["escalation"]["observation"]["terminal"] is True
    assert len(attempt.candidate.trace["play"]["plays"]) == 32
    assert len(attached_projects) == 1
    assert all(
        value.name.startswith("play_score_projects_awarded_value_")
        for value in attached_projects[0]
    )


@pytest.mark.parametrize("buyer", [None, "west", "north"])
def test_native_complete_trace_binds_escalation_to_auction_declarer(rules, buyer):
    def leaf(identity, kind, arguments):
        return {"predicateId": identity, "kind": kind, "version": 1,
                "hardness": "hard", "arguments": arguments}

    constraints = [
        leaf("PRED-CONTRACT", "auction.contract", {"contract": "hokum"}),
        leaf("PRED-DOUBLE", "escalation.action_sequence", {"actions": [
            {"actor": "north", "action": "double", "mode": "open"},
        ], "complete": True}),
    ]
    if buyer is not None:
        constraints.append(leaf("PRED-BUYER", "auction.buyer", {"seat": buyer}))
    raw = _search_escalation_terminal_scenario(True).to_dict()
    raw.pop("sha256")
    raw["constraints"] = {"op": "all", "children": constraints}
    attempt = GenericCompleteTraceCompilerV1(rules=rules, native_cp_sat_scoring=True)(
        ResearchScenarioV1.build(raw), (), seed=0, maximum_seconds=10,
    )
    if buyer == "north":
        assert attempt.status == "infeasible", attempt.reason
        assert attempt.reason == "deal_auction_infeasible"
        return
    assert attempt.status == "feasible", attempt.reason
    assert attempt.candidate.trace["buyer"] in {"west", "east"}
    assert attempt.candidate.trace["escalation"]["observation"]["actions"] == [
        {"actor": "north", "team": "north_south", "action": "double",
         "requestedMode": "open", "resolvedMode": "open"},
    ]


def test_native_complete_trace_scores_awarded_hundred_from_project_fragment(rules, monkeypatch):
    native_solve = research_executor.OrToolsScenarioModelV1.solve
    solvers = []

    def capture_native_solver(model, **kwargs):
        solver = kwargs["solver"]
        assert solver.parameters.cp_model_probing_level == 0
        solvers.append(solver)
        return native_solve(model, **kwargs)

    monkeypatch.setattr(research_executor.OrToolsScenarioModelV1, "solve", capture_native_solver)
    scenario = ResearchScenarioV1.build({
        "schemaVersion": 1,
        "origin": {"kind": "search"},
        "constraints": {"op": "all", "children": [
            {"predicateId": "PRED-SUN", "kind": "auction.contract", "version": 1,
             "hardness": "hard", "arguments": {"contract": "sun"}},
            {"predicateId": "PRED-HUNDRED", "kind": "project.exists", "version": 1,
             "hardness": "hard", "arguments": {"team": "west_east",
             "kind": "hundred_sequence", "value": 20, "state": "awarded"}},
        ]},
        "decisionPoint": {"kind": "play", "selector": {"trickIndex": 6}},
        "outcome": {"predicateId": "PRED-ROUND-REPORT", "kind": "outcome.team_round",
                    "version": 1, "hardness": "report",
                    "arguments": {"team": "north_south", "status": "won"}},
    })
    attempt = GenericCompleteTraceCompilerV1(
        rules=rules, native_cp_sat_scoring=True
    )(scenario, (), seed=0, maximum_seconds=20)
    assert attempt.status == "feasible", attempt.reason
    assert solvers and all(solver.parameters.num_search_workers == 1 for solver in solvers)
    trace = attempt.candidate.trace
    assert trace["contract"] == "sun"
    assert trace["projects"]["combinedAwardedValue"][1] >= 20
    assert trace["score"]["projectPoints"] == trace["projects"]["combinedAwardedValue"]


def test_native_play_project_score_disagreement_fails_closed(rules, monkeypatch):
    original = research_executor.OrToolsProjectLifecycleFragmentV1.extract

    def disagree(fragment, solver):
        evidence = original(fragment, solver)
        if fragment.combined_awarded_value[0].name.startswith("play_score_projects_"):
            points = evidence.combined_awarded_value
            return replace(evidence, combined_awarded_value=(points[0] + 1, points[1]))
        return evidence

    monkeypatch.setattr(
        research_executor.OrToolsProjectLifecycleFragmentV1, "extract", disagree
    )
    with pytest.raises(AssertionError, match="play and project stages disagree"):
        GenericCompleteTraceCompilerV1(rules=rules, native_cp_sat_scoring=True)(
            _search_escalation_terminal_scenario(True), (), seed=0,
            maximum_seconds=5,
        )


def test_escalation_projection_budget_is_incomplete_not_infeasible(
    rules, monkeypatch
):
    scenario = _search_escalation_terminal_scenario(True)
    monkeypatch.setattr(
        research_executor,
        "_eligible_escalation_fingerprints",
        lambda *_args, **_kwargs: None,
    )

    attempt = GenericCompleteTraceCompilerV1(rules=rules)(
        scenario,
        (),
        seed=0,
        maximum_seconds=5,
    )

    assert attempt.status == "incomplete"
    assert attempt.reason == "escalation_predicate_budget"
