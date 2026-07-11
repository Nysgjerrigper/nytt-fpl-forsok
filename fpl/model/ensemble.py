"""
Per-position ensemble: a fitted model per algorithm in fpl.model.models.FACTORIES,
blended with non-negative weights fit on held-out predictions (see
fit_blend_weights). Never persisted: every consumer (predict.py's walk-forward,
run_week.py's live run) refits fresh through train.fit_position_ensembles, so
there is exactly one definition of the production model and no stale-artifact
path to diverge from it (a saved-but-never-loaded copy was audit finding A1).
"""
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


def fit_equal_weights_top_k(predictions_by_model, y_true, k=3):
    """Rank members by individual MAE and equal-weight the best k, zero the rest.

    Clemen (1989): with many collinear members over a short holdout, estimated
    optimal weights routinely lose to a simple average because weight-estimation
    error dwarfs the theoretical gain. A hard top-k truncation followed by an
    equal (1/k) split spends zero degrees of freedom on the weights themselves -
    only on the far more robust MAE ranking - so it barely moves between adjacent
    windows, unlike the NNLS blend that can swing wildly as members trade places."""
    names = list(predictions_by_model.keys())
    y = np.asarray(y_true, dtype=float)
    maes = {n: np.mean(np.abs(np.asarray(predictions_by_model[n], dtype=float) - y)) for n in names}
    # Effective k can't exceed the number of members available.
    k = max(1, min(k, len(names)))
    best = sorted(names, key=lambda n: maes[n])[:k]
    winners = set(best)
    return {n: (1.0 / k if n in winners else 0.0) for n in names}


def fit_ridge_stack(predictions_by_model, y_true, alpha=1.0):
    """Ridge-regularised stacking: L2-shrunk least squares of y on the members.

    NNLS lets a few members grab huge offsetting weights that fit holdout noise;
    the ridge penalty pulls the coefficient vector toward zero so no single
    collinear member dominates, trading a little bias for much lower variance in
    the estimated weights (the Clemen critique of unregularised combination).
    Negatives are clipped and the vector renormalised so the result is a convex
    combination that stays drop-in compatible with PositionEnsemble; if the
    penalty shrinks everything non-positive we fall back to an equal average."""
    names = list(predictions_by_model.keys())
    A = np.column_stack([np.asarray(predictions_by_model[n], dtype=float) for n in names])
    y = np.asarray(y_true, dtype=float)
    # Closed-form ridge (no intercept): w = (A'A + alpha*I)^-1 A'y. Small system
    # (~12 members), so solving the normal equations directly is cheap and avoids
    # pulling in a heavier sklearn estimator just for a dozen coefficients.
    n_members = A.shape[1]
    gram = A.T @ A + alpha * np.eye(n_members)
    w = np.linalg.solve(gram, A.T @ y)
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= 0:
        # Everything shrank/clipped away: fall back to a plain equal average.
        w = np.ones(n_members)
        total = n_members
    w = w / total
    return dict(zip(names, w))


# Dispatcher so callers (train.py) can switch combination strategy by name
# without importing each fitter; keeps the "which combiner" choice in one place.
_WEIGHT_FITTERS = {
    "nnls": fit_blend_weights,
    "top_k": fit_equal_weights_top_k,
    "ridge": fit_ridge_stack,
}


def fit_weights(predictions_by_model, y_true, method="nnls", **kw):
    """Thin dispatcher over the combiners above. `method` selects the strategy;
    extra keyword args (k, alpha) pass through to the chosen fitter."""
    try:
        fitter = _WEIGHT_FITTERS[method]
    except KeyError:
        raise ValueError(f"unknown combination method {method!r}; choose from {sorted(_WEIGHT_FITTERS)}")
    return fitter(predictions_by_model, y_true, **kw)
