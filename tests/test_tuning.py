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


def test_save_best_params_records_cap_and_fit_model_strips_it(tmp_path, monkeypatch):
    # The params JSON must document the GW cap the search ran under (audit A2), and
    # models._tuned_params must strip that metadata before the constructor splat -
    # otherwise every tuned fit would crash on an unexpected "_meta" argument.
    from fpl import config
    from fpl.model import models

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    params = {"iterations": 10, "depth": 3, "loss_function": "MAE", "verbose": 0,
              "allow_writing_files": False, "random_seed": 0}
    path = tuning.save_best_params("MID", "catboost", params, train_max_gw=152)

    import json
    on_disk = json.loads(path.read_text())
    assert on_disk["_meta"] == {"train_max_gw": 152}

    loaded = models._tuned_params("catboost", "MID")
    assert "_meta" not in loaded
    assert loaded == params  # everything else survives the round-trip


def test_tune_position_caps_folds_at_train_max_gw():
    pytest.importorskip("optuna")
    df = _tiny_frame()
    feature_cols = ["feat_a", "feat_b"]
    # Cap below the frame's max GW (12): rows above the cap must not be seen. With the
    # cap at 8 there are only 8 distinct GWs, so a 4-split CV still builds; a cap of 1
    # leaves a single GW - no (train, val) fold can exist, so the cap being applied is
    # observable as the "not enough distinct gameweeks" refusal.
    best = tuning.tune_position(df, feature_cols, "MID", "lightgbm",
                                n_trials=2, n_splits=4, train_max_gw=8)
    assert isinstance(best, dict)
    with pytest.raises(ValueError, match="not enough distinct gameweeks"):
        tuning.tune_position(df, feature_cols, "MID", "lightgbm",
                             n_trials=2, n_splits=4, train_max_gw=1)
