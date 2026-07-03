"""
Per-position ensemble: a fitted model per algorithm in fpl.model.models.FACTORIES,
blended with non-negative weights fit on held-out predictions (see
fit_blend_weights). Persisted with joblib so the whole ensemble - members and
weights - loads back as one object for live prediction in run_week.py.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import nnls


class PositionEnsemble:
    def __init__(self, members, weights):
        """members: dict[name -> fitted estimator]. weights: dict[name -> float], summing to 1."""
        self.members = members
        self.weights = weights

    def member_predictions(self, X):
        return {name: model.predict(X) for name, model in self.members.items()}

    def predict(self, X):
        preds = self.member_predictions(X)
        return sum(self.weights.get(name, 0.0) * p for name, p in preds.items())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.members, path.with_suffix(".members.joblib"))
        path.with_suffix(".weights.json").write_text(json.dumps(self.weights))

    @classmethod
    def load(cls, path):
        path = Path(path)
        members = joblib.load(path.with_suffix(".members.joblib"))
        weights = json.loads(path.with_suffix(".weights.json").read_text())
        return cls(members, weights)


def fit_blend_weights(predictions_by_model, y_true):
    """Non-negative least squares blend: finds w >= 0 minimizing
    ||sum_i w_i * pred_i - y_true||, then normalizes to sum to 1 so the
    blend can't just scale everything up to fit training noise."""
    names = list(predictions_by_model.keys())
    A = np.column_stack([predictions_by_model[n] for n in names])
    w, _ = nnls(A, y_true)
    total = w.sum()
    if total <= 0:
        # Degenerate case (e.g. all predictors uncorrelated with target): fall back to equal weights.
        w = np.ones(len(names))
        total = len(names)
    w = w / total
    return dict(zip(names, w))
