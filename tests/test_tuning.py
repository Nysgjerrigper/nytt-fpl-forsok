"""
Smoke test for the Optuna tuning module: it must import without optuna installed
(the dependency is optional and lazily imported), and when optuna IS present a tiny
2-trial search must complete and hand back a params dict ready to build a model with.

This is deliberately a smoke test, not a quality test - it does not check the tuned
params are actually GOOD (that needs the real dataset and a backtest). It only guards
the plumbing: expanding-window folds build, the objective runs end to end, and the
winning params come back as a dict. optuna is skipped gracefully when absent so a fresh
clone without the heavy dep still passes the suite.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl.model import tuning


def _tiny_frame():
    """One position, several gameweeks, a couple of numeric features and a target.

    Enough distinct GWs (12) that a 4-split expanding-window CV has non-empty train and
    validation blocks in every fold; small enough that a 2-trial LightGBM search is instant.
    """
    rng = np.random.default_rng(0)
    rows = []
    for player_id in range(6):
        for gw in range(1, 13):
            rows.append({
                "player_id": player_id,
                "GW_global": gw,
                "position": "MID",
                "feat_a": rng.normal(),
                "feat_b": rng.normal(),
                "total_points": rng.integers(0, 12),
            })
    return pd.DataFrame(rows)


def test_module_imports_without_optuna():
    # The import at module top already ran; this just pins the contract that importing
    # tuning never requires optuna (only calling tune_position does).
    assert hasattr(tuning, "tune_position")
    assert tuning.SUPPORTED_MODELS == ("lightgbm", "xgboost", "catboost")


def test_tune_position_returns_params_dict():
    pytest.importorskip("optuna")
    df = _tiny_frame()
    feature_cols = ["feat_a", "feat_b"]
    best = tuning.tune_position(df, feature_cols, "MID", "lightgbm", n_trials=2, n_splits=4)
    assert isinstance(best, dict)
    # The searched knobs must be present, and so must the fixed settings, so the dict
    # is directly usable as LGBMRegressor(**best).
    assert "num_leaves" in best and "learning_rate" in best
    assert best["objective"] == "regression"
