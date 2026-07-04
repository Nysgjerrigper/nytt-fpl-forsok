"""
Sanity checks for the probabilistic (quantile) forecasting helpers. These guard
the two properties that make quantile output trustworthy: predictions must be
monotonic across quantiles (a "90th percentile" that comes out below the "10th"
is meaningless), and the scoring helpers must behave correctly on known inputs.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl.model import probabilistic as prob


def test_predict_quantiles_are_monotonic():
    # Two deliberately-crossed constant "models": the 0.9 model predicts LOWER than
    # the 0.1 model. predict_quantiles must repair the crossing so p10 <= p50 <= p90.
    class Const:
        def __init__(self, v):
            self.v = v

        def predict(self, X):
            return np.full(len(X), self.v)

    models = {0.1: Const(5.0), 0.5: Const(3.0), 0.9: Const(1.0)}
    X = np.zeros((4, 2))
    preds = prob.predict_quantiles(models, X)
    lo, mid, hi = preds[0.1], preds[0.5], preds[0.9]
    assert np.all(lo <= mid) and np.all(mid <= hi)


def test_pinball_loss_zero_on_perfect_prediction():
    y = np.array([0.0, 3.0, 10.0])
    assert prob.pinball_loss(y, y, 0.5) == 0.0


def test_pinball_loss_asymmetry():
    # For the 0.9 quantile, under-prediction (missing upside) should hurt more than
    # an equal-sized over-prediction - that asymmetry is the whole point of the loss.
    y = np.array([10.0])
    under = prob.pinball_loss(y, np.array([9.0]), 0.9)
    over = prob.pinball_loss(y, np.array([11.0]), 0.9)
    assert under > over


def test_interval_coverage_counts_correctly():
    y = np.array([0.0, 2.0, 5.0, 9.0])
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([1.0, 3.0, 4.0, 10.0])  # 5.0 falls outside [0, 4]
    assert prob.interval_coverage(y, lower, upper) == 0.75
