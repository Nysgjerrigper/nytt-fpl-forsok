"""Missing owned forecasts must fail before any optimizer can ignore holdings."""
import sys
import types

import pandas as pd
import pytest

from fpl import config, run_week
from fpl.milp import optimize


def test_owned_pool_guard_accepts_complete_or_empty_owned_ids_without_mutating_inputs():
    owned, forecasts = [1, 2], [3, 2, 1]
    assert optimize.validate_owned_forecast_pool(owned, forecasts) is None
    assert optimize.validate_owned_forecast_pool([], forecasts) is None
    assert owned == [1, 2] and forecasts == [3, 2, 1]


def test_owned_pool_guard_lists_missing_ids_and_explains_blank_horizon():
    with pytest.raises(ValueError) as error:
        optimize.validate_owned_forecast_pool([1, 2, 99], [1, 3])
    message = str(error.value)
    assert "2" in message and "99" in message and "missing from the forecast pool" in message
    assert "no fixtures in the requested horizon" in message and "complete owned squad" in message


def _minimal_optimizer_args(tmp_path, initial):
    path = tmp_path / "predictions.csv"
    pd.DataFrame({"player_id": [1], "GW": [1], "name": ["p1"], "position": ["MID"],
                  "team": ["A"], "value": [50.], "predicted_total_points": [2.]}).to_csv(path, index=False)
    return optimize.parse_args(["--predictions-csv", str(path), "--start-gw", "1", "--max-gw", "1",
                                "--initial-squad", initial, "--initial-budget", "250"])


def test_optimizer_missing_owned_id_rejected_before_solver_or_model_construction(tmp_path, monkeypatch):
    monkeypatch.setattr(optimize, "make_solver", lambda *a, **k: pytest.fail("missing owned ID reached solver construction"))
    monkeypatch.setattr(optimize.pulp, "LpProblem", lambda *a, **k: pytest.fail("missing owned ID reached model construction"))
    with pytest.raises(ValueError, match="Owned player IDs missing.*99"):
        optimize.run(_minimal_optimizer_args(tmp_path, "1,99"))


def test_optimizer_complete_owned_pool_reaches_existing_solver_path(tmp_path, monkeypatch):
    calls = []
    def solver(*args):
        calls.append(args)
        raise RuntimeError("valid-input sentinel")
    monkeypatch.setattr(optimize, "make_solver", solver)
    with pytest.raises(RuntimeError, match="valid-input sentinel"):
        optimize.run(_minimal_optimizer_args(tmp_path, "1"))
    assert len(calls) == 1


