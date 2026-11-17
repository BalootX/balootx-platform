from ortools.sat.python import cp_model

from baloot_platform.research.ortools_infeasible_core import (
    sound_infeasible_assumption_indexes,
)


def test_unregistered_solver_core_falls_back_to_full_sufficient_assumptions() -> None:
    model = cp_model.CpModel()
    value = model.new_bool_var("value")
    true_case = model.new_bool_var("true_case")
    false_case = model.new_bool_var("false_case")
    model.add(value == 1).only_enforce_if(true_case)
    model.add(value == 0).only_enforce_if(false_case)
    model.add_assumptions((true_case, false_case))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.INFEASIBLE

    class MisreportedCore:
        def sufficient_assumptions_for_infeasibility(self):
            return (999_999,)

    indexes = sound_infeasible_assumption_indexes(
        MisreportedCore(), model.proto.assumptions
    )
    assert indexes == (true_case.index, false_case.index)
    model.clear_assumptions()
    model.add_assumptions(tuple(model.get_bool_var_from_proto_index(i) for i in indexes))
    assert solver.solve(model) == cp_model.INFEASIBLE


def test_valid_empty_core_is_not_replaced() -> None:
    class EmptyCore:
        def sufficient_assumptions_for_infeasibility(self):
            return ()

    assert sound_infeasible_assumption_indexes(EmptyCore(), (1, 2)) == ()
