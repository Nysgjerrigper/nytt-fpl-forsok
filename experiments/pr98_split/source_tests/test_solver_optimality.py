"""Zero-gap settings and feasible-incumbent rejection at the legacy boundary."""
import pandas as pd
import pulp
import pytest

from fpl.milp import optimize


def test_highs_receives_explicit_zero_relative_and_absolute_gaps():
    solver = optimize.make_solver("highs", None, gap_rel=0)
    model = pulp.LpProblem("gap_options", pulp.LpMaximize)
    solver.createAndConfigureSolver(model)
    assert model.solverModel.getOptionValue("mip_rel_gap")[1] == 0.0
    assert model.solverModel.getOptionValue("mip_abs_gap")[1] == 0.0


@pytest.mark.parametrize("gap", [0.0, 0.001])
def test_cbc_receives_explicit_requested_gap(gap):
    solver = optimize.make_solver("cbc", None, gap_rel=gap)
    assert solver.optionsDict["gapRel"] == gap
    assert solver.optionsDict["gapAbs"] == 0.0


@pytest.mark.parametrize("status,solution_status", [
    (pulp.LpStatusOptimal, pulp.LpSolutionIntegerFeasible),
    (pulp.LpStatusNotSolved, pulp.LpSolutionNoSolutionFound),
    (pulp.LpStatusInfeasible, pulp.LpSolutionInfeasible),
])
def test_unproven_status_fails_before_export_even_with_objective(
    tmp_path, monkeypatch, capsys, status, solution_status,
):
    path, output = tmp_path / "predictions.csv", tmp_path / "squads.csv"
    pd.DataFrame({"player_id": [1], "GW": [1], "name": ["p1"],
                  "position": ["MID"], "team": ["A"], "value": [50.],
                  "predicted_total_points": [2.]}).to_csv(path, index=False)
    def interrupted_solve(model, solver):
        for variable in model.variables():
            variable.varValue = 0.5  # Numeric objective does not prove a valid solution.
        model.assignStatus(status, solution_status)
        return status
    monkeypatch.setattr(pulp.LpProblem, "solve", interrupted_solve)
    args = optimize.parse_args(["--predictions-csv", str(path), "--start-gw", "1",
                                "--max-gw", "1", "--output", str(output)])
    with pytest.raises(RuntimeError, match="did not prove optimality"):
        optimize.run(args)
    assert not output.exists()
    assert "Status: Optimal (" not in capsys.readouterr().out
