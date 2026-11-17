from types import SimpleNamespace

import pytest

from baloot_platform.research.construction_pre_screen import preflight_play_world_count_v1
from baloot_platform.research.contracts import canonical_sha256
from baloot_platform.research.ortools_information_set_compiler import (
    InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1,
)


def run(**kwargs):
    return preflight_play_world_count_v1(
        SimpleNamespace(sha256="rules"),
        SimpleNamespace(decision_kind="play", exact_state_id="state", sha256="exact"),
        maximum_worlds=kwargs.pop("maximum_worlds", 2), maximum_attempts=10,
        maximum_seconds=1, **kwargs)


@pytest.mark.parametrize("reason,count,status", [
    ("maximum_hidden_worlds", 3, "over_cap"),
    ("maximum_hidden_worlds", 2, "incomplete"),
    ("maximum_seconds", 2, "incomplete"),
    ("maximum_nodes", 0, "incomplete"),
    ("no_compatible_hidden_worlds", 0, "incomplete"),
    ("cancelled", 1, "incomplete"),
])
def test_only_compatible_count_above_cap_skips(monkeypatch, reason, count, status):
    monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "__init__", lambda *a, **k: None)
    def stopped(*args, **kwargs):
        assert kwargs["maximum_worlds"] == 2
        assert not kwargs.get("bounded_world_scope", False)
        raise InformationSetCompilationStopped(reason, stage="hidden_world_enumeration",
                                              evidence={"acceptedWorlds": count})
    monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "_enumerate_worlds", stopped)
    result = run()
    assert result["status"] == status
    assert result["worldCount"] is None
    assert result["worldCountLowerBound"] == count
    assert result["coverageComplete"] is False and result["qualified"] is False
    digest = result.pop("sha256")
    assert digest == canonical_sha256(result)


def test_exhaustion_and_changed_limit_are_bound(monkeypatch):
    monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "__init__", lambda *a, **k: None)
    monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "_enumerate_worlds",
                        lambda *a, **k: (({}, {}), {"terminalSolverStatus": "infeasible"}))
    result = run()
    assert result["status"] == "within_cap" and result["worldCount"] == 2
    assert result["coverageComplete"] is True
    assert result["sha256"] != run(maximum_worlds=2048)["sha256"]


def test_cancel_before_solver_and_invalid_limit():
    assert run(cancel_check=lambda: True)["reason"] == "cancelled"
    for limit in (True, 0, -1, 2.5):
        with pytest.raises(ValueError):
            run(maximum_worlds=limit)


@pytest.mark.real_clock
@pytest.mark.parametrize("phase", ["initialization", "ownership", "project", "project_callback", "play"])
@pytest.mark.parametrize("reason", ["maximum_seconds", "cancelled"])
def test_native_count_stops_during_model_setup(rules, monkeypatch, phase, reason):
    from baloot_platform.core.cards import FULL_DECK
    from baloot_platform.core.enums import Seat
    from baloot_platform.research import ortools_information_set_compiler as module
    from baloot_platform.research.ortools_play_tail import OrToolsPlayTailStageV1

    clock, cancelled = [0.0], [False]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def stop():
        if reason == "cancelled":
            cancelled[0] = True
        else:
            clock[0] = 2.0

    original_init = OrToolsActorInformationSetCompilerV1.__init__
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if phase == "initialization":
            stop()
    monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "__init__", initialize)
    original_model = module.cp_model.CpModel
    original_int_var = original_model.new_int_var
    def int_var(self, lower, upper, name):
        if phase == "ownership" and name == "project_owner_31":
            stop()
        return original_int_var(self, lower, upper, name)
    monkeypatch.setattr(original_model, "new_int_var", int_var)
    def model():
        assert phase != "initialization", "stopped work must not construct a model"
        return original_model()
    monkeypatch.setattr(module.cp_model, "CpModel", model)

    def projects(*args, **kwargs):
        assert phase != "ownership", "stopped ownership setup must not compile projects"
        if phase in {"project", "project_callback"}:
            stop()
        if phase == "project_callback":
            kwargs["cancel_check"]()
            pytest.fail("project compilation must propagate the typed stop")
        return SimpleNamespace()
    monkeypatch.setattr(module, "attach_project_lifecycle_fragment_v1", projects)
    monkeypatch.setattr(OrToolsActorInformationSetCompilerV1,
                        "_bind_public_project_choices", lambda *args: None)
    def play(*args, **kwargs):
        assert phase == "play", "stopped project work must not start play compilation"
        stop()
        kwargs["cancel_check"]()
        pytest.fail("play compilation must propagate the typed stop")
    monkeypatch.setattr(OrToolsPlayTailStageV1, "build_model", play)
    monkeypatch.setattr(module.cp_model, "CpSolver",
                        lambda: pytest.fail("stopped setup must not start a solver"))
    state = {
        "currentPlayer": "north", "contract": "sun", "dealer": "east",
        "firstLeader": "north", "currentTrick": None,
        "hands": {seat.name.lower(): [card.code for card in
                  FULL_DECK[8 * int(seat) + 1:8 * int(seat) + 8]] for seat in Seat},
        "completedTricks": [{"winner": "north", "plays": [
            {"seat": seat.name.lower(), "card": FULL_DECK[8 * int(seat)].code}
            for seat in Seat]}],
        "projects": [], "events": [{"type": "project_declared"}],
    }
    result = preflight_play_world_count_v1(rules,
        SimpleNamespace(decision_kind="play", exact_state_id="state", sha256="exact", state=state),
        maximum_worlds=2, maximum_attempts=10, maximum_seconds=1,
        cancel_check=lambda: cancelled[0])
    assert result["status"] == "incomplete" and result["reason"] == reason
    assert result["coverageComplete"] is False and result["worldCount"] is None
    assert result["worldCountLowerBound"] == 0 and result["qualified"] is False


def test_native_late_prefix_count_exhausts_without_graphs(rules):
    from baloot_platform.research.construction_pre_screen import _proposal_opening
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    for seed in range(16):
        engine, state, prefix = _proposal_opening(rules, seed)
        if prefix is not None:
            break
    assert prefix is not None
    for _ in range(30):
        declaration = state._project_declaration_state
        if declaration is not None and not declaration.complete and declaration.current_actor == state.current_player:
            chosen = rules.project_domain.legal_declaration_actions(declaration, trump_suit=state.trump_suit)[0]
            engine.apply_project_declaration_action(state, chosen)
            prefix['projectActions'].append(chosen.to_dict())
        action = engine.legal_play_actions(state)[0]
        actor = state.current_player.name.lower()
        engine.play_card(state, action.card, ekka=action.ekka)
        prefix['playActions'].append(dict(actor=actor, card=action.card.code, ekka=action.ekka))
    prefix['projectDeclarations'] = [
        {key: event.to_dict()[key] for key in ('owner','kind','value','highRank','declarationOrder')}
        for event in state.events if event.to_dict().get('type') == 'project_declared']
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(prefix), decision_point={'kind':'play'}, maximum_seconds=30)
    result = preflight_play_world_count_v1(rules, exact, maximum_worlds=2,
        maximum_attempts=10, maximum_seconds=30)
    assert result['status'] == 'within_cap'
    assert result['worldCount'] == 1
    assert result['receipt']['terminalSolverStatus'] == 'infeasible'
