"""Native optimality certificates and fail-closed rolling optimizer admission."""

import highspy
import pandas as pd
import pulp
import pytest

from fpl.milp import optimize


class _Override:
    """Delegate a real solved native object, overriding only the tested evidence."""
    def __init__(self, original, **overrides):
        self.original = original
        self.overrides = overrides

    def __getattr__(self, name):
        return self.overrides[name] if name in self.overrides else getattr(self.original, name)


def _solved(solver_name="highs"):
    model = pulp.LpProblem("certificate", pulp.LpMaximize)
    choice = pulp.LpVariable("choice", lowBound=0, upBound=1, cat=pulp.LpInteger)
    model += 3 * choice
    model += choice <= 1
    solver = optimize.make_solver(solver_name, time_limit=10, threads=0, gap_rel=0)
    status = model.solve(solver)
    assert status == pulp.LpStatusOptimal
    return model, choice, status


def test_real_highs_optimum_has_native_proof():
    model, _, status = _solved()
    certificate = optimize.require_optimal_solution(model, status, "highs")
    assert certificate["native_optimality_assessment"] == "PROVEN_OPTIMAL"
    assert "Optimal" in str(certificate["native_solver_status"])
    assert certificate["mip_gap"] == 0
    assert certificate["native_objective"] == certificate["native_dual_bound"]


def test_cbc_report_is_distinguished_from_native_proof():
    model, _, status = _solved("cbc")
    certificate = optimize.require_optimal_solution(model, status, "cbc")
    assert certificate["native_optimality_assessment"] == "PULP_REPORTED_OPTIMAL"
    for field in ("mip_gap", "native_objective", "native_dual_bound", "native_solver_status"):
        assert certificate.get(field) is None


@pytest.mark.parametrize("status", [pulp.LpStatusNotSolved, pulp.LpStatusInfeasible, pulp.LpStatusUndefined])
def test_unqualified_pulp_status_rejected_even_with_finite_objective(status):
    model, _, _ = _solved()
    assert pulp.value(model.objective) == 3
    with pytest.raises(RuntimeError):
        optimize.require_optimal_solution(model, status, "highs")


def test_pulp_optimal_with_only_integer_feasible_solution_rejected():
    model, _, status = _solved()
    model.sol_status = pulp.LpSolutionIntegerFeasible
    with pytest.raises(RuntimeError):
        optimize.require_optimal_solution(model, status, "highs")


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), 0.5, 2])
def test_missing_nonfinite_fractional_or_bound_violating_variables_rejected(value):
    model, choice, status = _solved()
    choice.varValue = value
    with pytest.raises(RuntimeError):
        optimize.require_optimal_solution(model, status, "highs")


def test_nonfinite_objective_rejected():
    model, _, status = _solved()
    model.objective.constant = float("nan")
    with pytest.raises(RuntimeError):
        optimize.require_optimal_solution(model, status, "highs")


def test_invalid_pulp_constraint_solution_rejected(monkeypatch):
    model, _, status = _solved()
    monkeypatch.setattr(model, "valid", lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError):
        optimize.require_optimal_solution(model, status, "highs")


def test_native_time_limit_cannot_be_relabelled_pulp_optimal():
    model, _, status = _solved()
    model.solverModel = _Override(model.solverModel, getModelStatus=lambda: highspy.HighsModelStatus.kTimeLimit)
    with pytest.raises(RuntimeError):
        optimize.require_optimal_solution(model, status, "highs")


@pytest.mark.parametrize("field,value", [
    ("valid", False), ("mip_gap", 0.1), ("mip_gap", float("nan")),
    ("objective_function_value", float("inf")), ("mip_dual_bound", -float("inf")),
])
def test_invalid_or_nonfinite_native_certificate_fields_rejected(field, value):
    model, _, status = _solved()
    native = model.solverModel
    information = _Override(native.getInfo(), **{field: value})
    model.solverModel = _Override(native, getInfo=lambda: information)
    with pytest.raises(RuntimeError):
        optimize.require_optimal_solution(model, status, "highs")


def test_small_relative_gap_does_not_excuse_material_absolute_bound_difference():
    model, _, status = _solved()
    native = model.solverModel
    information = _Override(native.getInfo(), mip_gap=0.0,
                            mip_dual_bound=native.getInfo().objective_function_value - 0.001)
    model.solverModel = _Override(native, getInfo=lambda: information)
    with pytest.raises(RuntimeError):
        optimize.require_optimal_solution(model, status, "highs")


def test_native_invalid_solution_values_rejected():
    model, _, status = _solved()
    native = model.solverModel
    solution = _Override(native.getSolution(), value_valid=False)
    model.solverModel = _Override(native, getSolution=lambda: solution)
    with pytest.raises(RuntimeError):
        optimize.require_optimal_solution(model, status, "highs")


@pytest.mark.parametrize("option", ["mip_rel_gap", "mip_abs_gap"])
def test_native_nonzero_solver_gap_options_rejected(option):
    model, _, status = _solved()
    native = model.solverModel
    def read_option(name):
        if name == option:
            return highspy.HighsStatus.kOk, 0.01
        return native.getOptionValue(name)
    model.solverModel = _Override(native, getOptionValue=read_option)
    with pytest.raises(RuntimeError):
        optimize.require_optimal_solution(model, status, "highs")


@pytest.mark.parametrize("solver", ["highs", "cbc"])
@pytest.mark.parametrize("gap", [0.1, -0.1, float("nan"), float("inf")])
def test_solver_builder_rejects_nonzero_or_nonfinite_relative_gap(solver, gap):
    with pytest.raises(ValueError):
        optimize.make_solver(solver, time_limit=10, gap_rel=gap)


@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf")])
def test_solver_builder_rejects_nonpositive_or_nonfinite_time_limit(limit):
    with pytest.raises(ValueError):
        optimize.make_solver("highs", time_limit=limit)


@pytest.mark.parametrize("solver", ["highs", "cbc"])
def test_solver_builder_sets_both_gap_options_to_zero(solver):
    instance = optimize.make_solver(solver, time_limit=10, gap_rel=0)
    if solver == "highs":
        assert instance.gapRel == 0
        assert instance.gapAbs == 0
    else:
        assert instance.optionsDict["gapRel"] == 0
        assert instance.optionsDict["gapAbs"] == 0


@pytest.mark.parametrize("failed_round", [1, 2])
@pytest.mark.parametrize("failure", ["exception", "unqualified_status"])
def test_failed_first_or_later_solve_leaves_no_recommendation_artifacts(tmp_path, monkeypatch, failed_round, failure):
    roles = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    source, output, decisions = tmp_path / "predictions.csv", tmp_path / "squads.csv", tmp_path / "decisions.json"
    pd.DataFrame([
        {"player_id": i, "GW": gw, "name": f"player_{i}", "position": role,
         "team": f"club{i % 5}", "value": 50, "predicted_total_points": i + 1,
         "actual_total_points": i + 1, "minutes": 90}
        for gw in (1, 2) for i, role in enumerate(roles)
    ]).to_csv(source, index=False)
    original_bytes = source.read_bytes()
    actual_solve = pulp.LpProblem.solve
    calls = []
    def controlled_solve(model, solver=None, **kwargs):
        calls.append(model)
        if len(calls) == failed_round and failure == "exception":
            raise pulp.PulpSolverError("deliberate native failure")
        result = actual_solve(model, solver, **kwargs)
        return pulp.LpStatusNotSolved if len(calls) == failed_round else result
    monkeypatch.setattr(pulp.LpProblem, "solve", controlled_solve)
    args = optimize.parse_args([
        "--predictions-csv", str(source), "--start-gw", "1", "--max-gw", "2", "--horizon", "1",
        "--solver", "highs", "--output", str(output), "--decisions-json", str(decisions),
    ])
    with pytest.raises(RuntimeError):
        optimize.run(args)
    assert len(calls) == failed_round
    assert not output.exists()
    assert not decisions.exists()
    assert source.read_bytes() == original_bytes


def test_feasible_pulp_values_must_match_native_optimal_objective():
    model, choice, status = _solved()
    choice.varValue = 0  # Feasible, integral and finite, but not the native optimum.
    assert model.valid(1e-7)
    with pytest.raises(RuntimeError, match="objective"):
        optimize.require_optimal_solution(model, status, "highs")


def test_minimization_with_objective_constant_has_valid_native_certificate():
    model = pulp.LpProblem("minimum_with_constant", pulp.LpMinimize)
    choice = pulp.LpVariable("choice", lowBound=0, cat=pulp.LpInteger)
    model += 3 * choice + 5
    model += choice >= 1
    status = model.solve(optimize.make_solver("highs", time_limit=10, gap_rel=0))
    certificate = optimize.require_optimal_solution(model, status, "highs")
    assert choice.varValue == 1
    assert pulp.value(model.objective) == 8
    assert certificate["native_optimality_assessment"] == "PROVEN_OPTIMAL"
    assert certificate["native_objective"] == certificate["native_dual_bound"] == 3


@pytest.mark.parametrize("initial_ft", [0, 1, 2, 5])
def test_three_round_continuing_squad_carries_free_transfers_with_native_proof(tmp_path, initial_ft):
    roles = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    source = tmp_path / "constant_forecasts.csv"
    pd.DataFrame([
        {"player_id": i, "GW": gw, "name": f"player_{i}", "position": role,
         "team": f"club{i % 5}", "value": 50, "predicted_total_points": 3,
         "actual_total_points": 3, "minutes": 90}
        for gw in (1, 2, 3) for i, role in enumerate(roles)
    ]).to_csv(source, index=False)
    output = tmp_path / "squads.csv"
    result = optimize.run(optimize.parse_args([
        "--predictions-csv", str(source), "--start-gw", "1", "--max-gw", "3", "--horizon", "3",
        "--solver", "highs", "--initial-squad", ",".join(map(str, range(15))),
        "--initial-budget", "0", "--initial-ft", str(initial_ft), "--output", str(output),
    ]))
    cap = max(2, initial_ft)
    assert result.q_start.tolist() == [initial_ft, min(cap, initial_ft + 1), min(cap, initial_ft + 2)]
    assert result.alpha.tolist() == [0, 0, 0]
    assert result.transfers_in.tolist() == [[], [], []]
    assert result.transfers_out.tolist() == [[], [], []]
    assert result.native_optimality_assessment.tolist() == ["PROVEN_OPTIMAL"] * 3
    assert all(set(squad) == set(range(15)) for squad in result.squad)
