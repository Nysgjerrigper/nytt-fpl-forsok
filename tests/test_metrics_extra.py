"""
Known-value checks for the mean-aligned and decision-aligned metrics. These guard the
properties that make the metrics trustworthy as a complement to MAE/MASE: RMSE must
outweigh MAE on a skewed error vector (the whole reason it exists), bias must carry the
right SIGN, and the ranking/capture metrics must reflect ordering rather than raw error.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl.model import metrics as m


def test_rmse_exceeds_mae_on_skewed_errors():
    # One big miss among small ones: RMSE punishes the outlier far harder than MAE, which
    # is exactly the mean-vs-median gap this metric is meant to surface.
    y_true = np.array([0.0, 0.0, 0.0, 12.0])
    y_pred = np.array([0.0, 0.0, 0.0, 0.0])
    assert m.rmse(y_true, y_pred) > m.mae(y_true, y_pred)
    # Sanity: with no outlier (constant error), the two coincide.
    y_true2 = np.array([1.0, 3.0, 5.0])
    y_pred2 = np.array([2.0, 4.0, 6.0])
    assert np.isclose(m.rmse(y_true2, y_pred2), m.mae(y_true2, y_pred2))


def test_rmse_known_value():
    # errors are [3, 4] -> mean(9,16)=12.5 -> sqrt = 3.5355...
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    assert np.isclose(m.rmse(y_true, y_pred), np.sqrt(12.5))


def test_bias_sign_over_and_under_prediction():
    y_true = np.array([2.0, 2.0, 2.0])
    # Over-prediction -> positive bias.
    assert m.bias(y_true, np.array([5.0, 5.0, 5.0])) == 3.0
    # Under-prediction (median-flattening on a skewed target) -> negative bias.
    assert m.bias(y_true, np.array([0.0, 0.0, 0.0])) == -2.0
    # Unbiased -> zero.
    assert m.bias(y_true, y_true) == 0.0


def test_total_calibration_ratio_and_undefined():
    y_true = np.array([2.0, 4.0, 4.0])  # sum 10
    y_pred = np.array([1.0, 2.0, 2.0])  # sum 5 -> ratio 0.5
    assert np.isclose(m.total_calibration(y_true, y_pred), 0.5)
    # Undefined when nothing was actually scored.
    assert np.isnan(m.total_calibration(np.array([0.0, 0.0]), np.array([1.0, 2.0])))


def test_spearman_by_group_perfect_and_reversed():
    # Group A ranks perfectly (rho=+1), group B is exactly reversed (rho=-1) -> mean 0.
    y_true = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])
    groups = np.array(["A", "A", "A", "B", "B", "B"])
    assert np.isclose(m.spearman_by_group(y_true, y_pred, groups), 0.0)


def test_spearman_by_group_skips_small_and_constant_groups():
    # Group "big" (>=3 rows, varying) counts with rho=+1; the 2-row group and the
    # zero-variance group are both skipped, so the mean is the big group's rho alone.
    y_true = np.array([1.0, 2.0, 3.0, 9.0, 9.0, 7.0, 8.0])
    y_pred = np.array([1.0, 2.0, 3.0, 9.0, 1.0, 7.0, 8.0])
    groups = np.array(["big", "big", "big", "flat", "flat", "small", "small"])
    #                                         ^ constant y_true (9,9)  ^ only 2 rows
    assert np.isclose(m.spearman_by_group(y_true, y_pred, groups), 1.0)


def test_spearman_by_group_nan_when_no_group_qualifies():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 2.0])
    groups = np.array(["A", "A"])  # only 2 rows -> skipped
    assert np.isnan(m.spearman_by_group(y_true, y_pred, groups))


def test_top1_capture_hand_built_two_groups():
    # Two position-gameweek groups. In gw1 our top-predicted player (pred 9) actually
    # scored 3, while the true best was 10. In gw2 we nail it: top pick scored the max, 8.
    # captured = 3 + 8 = 11; best_possible = 10 + 8 = 18; ratio = 11/18.
    df = pd.DataFrame(
        {
            "grp": ["gw1", "gw1", "gw1", "gw2", "gw2"],
            "actual": [3.0, 10.0, 1.0, 8.0, 2.0],
            "pred": [9.0, 4.0, 2.0, 7.0, 1.0],
        }
    )
    got = m.top1_capture(df, "grp", "actual", "pred")
    assert np.isclose(got, 11.0 / 18.0)


def test_top1_capture_perfect_is_one():
    # Our #1 prediction is the true top scorer in every group -> capture 1.0.
    df = pd.DataFrame(
        {
            "grp": ["a", "a", "b", "b"],
            "actual": [5.0, 1.0, 9.0, 2.0],
            "pred": [5.0, 1.0, 9.0, 2.0],
        }
    )
    assert m.top1_capture(df, "grp", "actual", "pred") == 1.0


def test_top1_capture_nan_when_nothing_scored():
    df = pd.DataFrame(
        {"grp": ["a", "a"], "actual": [0.0, 0.0], "pred": [1.0, 2.0]}
    )
    assert np.isnan(m.top1_capture(df, "grp", "actual", "pred"))
