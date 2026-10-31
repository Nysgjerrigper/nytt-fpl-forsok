"""Lookahead free transfers must equal the rolling state's capped arithmetic."""
import pandas as pd
import pulp
import pytest

from fpl import config
from fpl.milp import optimize


@pytest.mark.parametrize("initial_ft,transfers", [(0, 0), (1, 1), (2, 0), (2, 1), (2, 3), (5, 0)])
def test_planned_ft_matches_actual_rollover(tmp_path, monkeypatch, initial_ft, transfers):
    # Two copies of a legal 15-player squad allow affordable forced transfers.
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    clubs = [0, 1] + list(range(5)) * 2 + [2, 3, 4]
    rows = [dict(player_id=i+1, GW=gw, name=f"p{i+1}", position=positions[i % 15],
                 team=str(clubs[i % 15]), value=50., predicted_total_points=1. + (i % 7))
            for gw in (1, 2) for i in range(30)]
    path, output = tmp_path / "predictions.csv", tmp_path / "squads.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    planned = []
    original_solve = pulp.LpProblem.solve
    def solve(model, solver):
        if not planned:
            incoming = [v for v in model.variables() if v.name.startswith("TransferIn_") and v.name.endswith("_1")]
            assert len(incoming) == 30
            model += pulp.lpSum(incoming) == transfers
        result = original_solve(model, solver)
        planned.append({v.name: v.varValue for v in model.variables()})
        return result
    monkeypatch.setattr(pulp.LpProblem, "solve", solve)
    args = optimize.parse_args(["--predictions-csv", str(path), "--start-gw", "1", "--max-gw", "2",
        "--horizon", "2", "--initial-squad", ",".join(map(str, range(1, 16))),
        "--initial-budget", "250", "--initial-ft", str(initial_ft), "--output", str(output)])
    result = optimize.run(args)
    expected = min(max(config.MILP_MAX_FREE_TRANSFERS, initial_ft), max(0, initial_ft-transfers)+config.MILP_FT_PER_GW)
    assert planned[0]["FreeTransfersAvailable_2"] == expected
    assert result.iloc[1].q_start == expected
    assert result.iloc[0].alpha == max(0, transfers-initial_ft)
    assert result.solver_solution_status.eq("Optimal Solution Found").all()
    assert result.solver_native_status.eq("Optimal").all()
    assert result.solver_mip_gap.eq(0).all()
    assert result.solver_gap_rel.eq(0).all()
    assert result.solver_gap_abs.eq(0).all()
