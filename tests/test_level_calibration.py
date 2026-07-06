"""Guards train.fit_level_calibration: the scalar that corrects MAE-loss median-flattening
before forecasts hit the MILP's absolute-scale transfer/chip logic (RESEARCH_LOG.md 2026-07-06)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl.model.train import POSITIONS, fit_level_calibration


def _frame(n_gws=40, players_per_pos=6, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    pid = 0
    for pos in POSITIONS:
        for _ in range(players_per_pos):
            pid += 1
            for gw in range(1, n_gws + 1):
                rows.append({
                    "player_id": pid, "GW_global": gw, "position": pos,
                    "f1": rng.normal(), "f2": rng.normal(),
                    "total_points": max(0.0, rng.normal(3.0, 2.0)),
                })
    return pd.DataFrame(rows)


def test_scalars_are_finite_positive_per_position():
    df = _frame()
    weights = {pos: {"lightgbm": 1.0} for pos in POSITIONS}
    scalars = fit_level_calibration(df, ["f1", "f2"], first_holdout_gw=33, weights_by_pos=weights, window=8)
    assert set(scalars) == set(POSITIONS)
    for pos, s in scalars.items():
        assert np.isfinite(s) and s > 0, f"{pos}: {s}"


def test_deflated_forecast_gets_upscaled():
    # Noise features -> the model predicts ~the training mean. Making holdout outcomes double
    # the training-era level guarantees sum(actual) > sum(predicted), so the scalar must be > 1.
    df = _frame()
    df.loc[df["GW_global"] >= 25, "total_points"] *= 2.0
    weights = {pos: {"lightgbm": 1.0} for pos in POSITIONS}
    scalars = fit_level_calibration(df, ["f1", "f2"], first_holdout_gw=33, weights_by_pos=weights, window=8)
    assert all(s > 1.0 for s in scalars.values()), scalars


def test_empty_holdout_falls_back_to_identity():
    df = _frame(n_gws=10)
    weights = {pos: {"lightgbm": 1.0} for pos in POSITIONS}
    # Holdout window sits entirely before the data starts -> member_train is empty -> 1.0.
    scalars = fit_level_calibration(df, ["f1", "f2"], first_holdout_gw=5, weights_by_pos=weights, window=8)
    assert all(s == 1.0 for s in scalars.values())
