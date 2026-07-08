import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl.model import probabilistic_buckets as buckets
from fpl.model import models as point_models


def test_points_to_buckets_maps_fpl_outcomes():
    y = np.array([-1, 0, 1, 2, 3, 5, 6, 9, 10, 15])
    assert buckets.points_to_buckets(y).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_points_to_buckets_supports_finer_scheme():
    y = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10, 12, 13])
    scheme = buckets.get_bucket_scheme("fine8")
    assert buckets.points_to_buckets(y, scheme).tolist() == [0, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7]


def test_expected_points_from_distribution_uses_training_bucket_values():
    proba = np.array([[0.2, 0.3, 0.0, 0.0, 0.5]])
    values = np.array([0.0, 2.0, 4.0, 8.0, 12.0])
    assert np.isclose(buckets.expected_points_from_proba(proba, values)[0], 6.6)


def test_normalize_proba_repairs_roundoff():
    proba = buckets.normalize_proba(np.array([[0.2, 0.2, 0.2, 0.2, 0.3]]))
    assert np.isclose(proba.sum(axis=1)[0], 1.0)


def test_multiclass_brier_zero_for_perfect_probabilities():
    y = np.array([0, 2, 4])
    proba = np.eye(buckets.N_BUCKETS)[y]
    assert buckets.multiclass_brier(y, proba) == 0.0


def test_cat_params_adapts_tuned_regression_params(monkeypatch):
    tuned = {"depth": 5, "iterations": 731, "learning_rate": 0.0138,
             "subsample": 0.676, "loss_function": "MAE", "random_seed": 0, "verbose": 0}
    monkeypatch.setattr(point_models, "_tuned_params", lambda name, pos: dict(tuned))
    params = buckets._cat_params(quick=False, loss_function="MultiClass", position="MID", use_tuned=True)
    assert params["loss_function"] == "MultiClass"
    assert params["depth"] == 5
    assert params["bootstrap_type"] == "Bernoulli"  # required by CatBoost when subsample is set

    quick_params = buckets._cat_params(quick=True, loss_function="Logloss", position="MID", use_tuned=True)
    assert quick_params["iterations"] == 80


def test_cat_params_without_tuning_uses_defaults():
    params = buckets._cat_params(quick=False, loss_function="MultiClass")
    assert params["loss_function"] == "MultiClass"
    assert params["iterations"] == point_models.CATBOOST_PARAMS["iterations"]


def _regression_frame():
    test_df = pd.DataFrame({
        "GW_global": [153, 153, 154, 154],
        "total_points": [0.0, 6.0, 2.0, 12.0],
    })
    return buckets.regression_prediction_frame(
        [1.0, 5.0, 2.0, 9.0], test_df, "MID", buckets.DEFAULT_BUCKET_SCHEME
    )


def test_regression_frame_scores_point_metrics_and_nans_distribution():
    metrics = buckets.evaluate_prediction_frame(_regression_frame())
    assert np.isfinite(metrics["ev_mae"])
    assert np.isfinite(metrics["bias"])
    assert np.isfinite(metrics["total_calibration"])
    assert np.isnan(metrics["bucket_logloss"])
    assert np.isnan(metrics["haul_auc"])


def test_captaincy_metrics_skip_haul_ranking_without_distribution():
    cap = buckets.captaincy_metrics(_regression_frame())
    assert np.isfinite(cap["cap_ev"])
    assert np.isnan(cap["cap_haul"])
    assert np.isnan(cap["cap_tilt"])
