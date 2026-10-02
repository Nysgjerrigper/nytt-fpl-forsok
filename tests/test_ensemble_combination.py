"""
Regression guards for the robust forecast-combination alternatives in
fpl.model.ensemble. These check the *properties* the combiners promise
(convexity of weights, top-k truncation, concentration on a good member,
graceful degenerate fallback) rather than exact float weights, so the tests
stay stable if the underlying solver details change.
"""

import numpy as np
import pytest

from fpl.model.ensemble import (
    fit_equal_weights_top_k,
    fit_ridge_stack,
    fit_weights,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _member_maes(preds, y):
    return {n: np.mean(np.abs(np.asarray(p) - y)) for n, p in preds.items()}


def test_top_k_picks_exactly_the_k_best_and_sums_to_one(rng):
    y = rng.normal(size=200)
    # Each member = target plus increasing noise, so MAE order is deterministic:
    # m0 best, m4 worst.
    preds = {f"m{i}": y + rng.normal(scale=0.1 * (i + 1), size=y.size) for i in range(5)}
    maes = _member_maes(preds, y)
    k = 3
    weights = fit_equal_weights_top_k(preds, y, k=k)

    nonzero = {n for n, w in weights.items() if w > 0}
    expected = set(sorted(maes, key=maes.get)[:k])
    assert nonzero == expected
    assert len(nonzero) == k
    for n in nonzero:
        assert weights[n] == pytest.approx(1.0 / k)
    assert sum(weights.values()) == pytest.approx(1.0)
    # Every member appears in the returned dict (zeros included) for drop-in use.
    assert set(weights) == set(preds)


def test_top_k_caps_k_at_member_count(rng):
    y = rng.normal(size=50)
    preds = {"a": y + rng.normal(scale=0.1, size=y.size), "b": y + rng.normal(scale=0.2, size=y.size)}
    weights = fit_equal_weights_top_k(preds, y, k=10)
    # k larger than the pool must not create phantom members or break normalisation.
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(w == pytest.approx(0.5) for w in weights.values())


def test_both_new_methods_concentrate_on_the_good_member(rng):
    y = rng.normal(size=300)
    good = y.copy()  # perfect forecaster
    noise = {f"noise{i}": rng.normal(size=y.size) for i in range(4)}  # uncorrelated with y
    preds = {"good": good, **noise}

    top_k = fit_equal_weights_top_k(preds, y, k=1)
    assert top_k["good"] == pytest.approx(1.0)
    assert all(top_k[n] == pytest.approx(0.0) for n in noise)

    ridge = fit_ridge_stack(preds, y, alpha=1.0)
    assert sum(ridge.values()) == pytest.approx(1.0)
    # The perfect member should dominate the convex combination.
    assert ridge["good"] > 0.9
    assert ridge["good"] > max(ridge[n] for n in noise)


def test_ridge_falls_back_to_equal_when_all_weights_clip(rng):
    # Members are the exact negation of the target: least-squares wants strongly
    # negative coefficients, which all clip to zero, forcing the equal-weight
    # fallback rather than a degenerate all-zero (non-summing) result.
    y = np.abs(rng.normal(size=100)) + 1.0  # strictly positive target
    preds = {"neg_a": -y, "neg_b": -2.0 * y, "neg_c": -0.5 * y}
    weights = fit_ridge_stack(preds, y, alpha=1.0)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(w == pytest.approx(1.0 / len(preds)) for w in weights.values())


def test_ridge_weights_are_a_valid_convex_combination(rng):
    y = rng.normal(size=150)
    preds = {f"m{i}": y + rng.normal(scale=0.3 * (i + 1), size=y.size) for i in range(4)}
    weights = fit_ridge_stack(preds, y, alpha=1.0)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(w >= 0.0 for w in weights.values())


def test_dispatcher_routes_and_passes_kwargs(rng):
    y = rng.normal(size=120)
    preds = {f"m{i}": y + rng.normal(scale=0.1 * (i + 1), size=y.size) for i in range(5)}

    # top_k via dispatcher must honour the forwarded k.
    disp_top = fit_weights(preds, y, method="top_k", k=2)
    direct_top = fit_equal_weights_top_k(preds, y, k=2)
    assert disp_top == direct_top
    assert sum(1 for w in disp_top.values() if w > 0) == 2

    # ridge via dispatcher must match the direct call.
    disp_ridge = fit_weights(preds, y, method="ridge", alpha=2.0)
    direct_ridge = fit_ridge_stack(preds, y, alpha=2.0)
    assert disp_ridge == pytest.approx(direct_ridge)

    # nnls is the default and stays reachable.
    default = fit_weights(preds, y)
    assert sum(default.values()) == pytest.approx(1.0)


def test_dispatcher_rejects_unknown_method(rng):
    y = rng.normal(size=10)
    preds = {"m0": y}
    with pytest.raises(ValueError):
        fit_weights(preds, y, method="does_not_exist")


class _FixedModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, X):
        return self.predictions


class _BrokenModel:
    def predict(self, X):
        raise AssertionError("An inactive member must never be evaluated")


@pytest.mark.parametrize("weights", [
    {}, {"a": 0.0}, {"a": 0.5}, {"a": 1.5},
    {"a": np.nan}, {"a": np.inf}, {"a": -np.inf},
    {"a": -0.1, "b": 1.1},
])
def test_ensemble_rejects_invalid_weights(weights):
    from fpl.model.ensemble import PositionEnsemble

    with pytest.raises(ValueError, match="weights"):
        PositionEnsemble({"a": _FixedModel([1.0]), "b": _FixedModel([2.0])}, weights)


def test_ensemble_rejects_empty_members_and_missing_positive_members():
    from fpl.model.ensemble import PositionEnsemble

    with pytest.raises(ValueError, match="at least one member"):
        PositionEnsemble({}, {"a": 1.0})
    with pytest.raises(ValueError, match="missing positively weighted members"):
        PositionEnsemble({"a": _FixedModel([1.0])}, {"a": 0.5, "missing": 0.5})


def test_ensemble_ignores_zero_weight_members_and_preserves_weighted_prediction():
    from fpl.model.ensemble import PositionEnsemble

    ensemble = PositionEnsemble(
        {"a": _FixedModel([2.0, 4.0]), "b": _FixedModel([6.0, 8.0]),
         "broken": _BrokenModel(), "nan": _FixedModel([np.nan, np.nan])},
        {"a": 0.25, "b": 0.75, "broken": 0.0, "nan": 0.0, "absent": 0.0},
    )
    np.testing.assert_array_equal(ensemble.predict(np.zeros((2, 1))), [5.0, 7.0])


@pytest.mark.parametrize("predictions", [
    np.array([[1.0], [2.0]]), [1.0], 1.0, [np.nan, 1.0], [np.inf, 1.0],
])
def test_ensemble_rejects_invalid_prediction_vectors(predictions):
    from fpl.model.ensemble import PositionEnsemble

    ensemble = PositionEnsemble({"a": _FixedModel(predictions)}, {"a": 1.0})
    with pytest.raises(ValueError, match="Member 'a'.*finite one-dimensional"):
        ensemble.predict(np.zeros((2, 1)))


def test_ensemble_does_not_normalize_approximately_unit_weights():
    from fpl.model.ensemble import PositionEnsemble

    weight = 1.0 + 1e-7
    ensemble = PositionEnsemble({"a": _FixedModel([2.0])}, {"a": weight})
    assert ensemble.predict(np.zeros((1, 1)))[0] == 2.0 * weight


def test_production_fit_preserves_every_positive_weight(monkeypatch):
    import pandas as pd
    from fpl.model import train

    calls = []

    def fit_model(name, X, y, **kwargs):
        calls.append(name)
        return _FixedModel([1.0])

    monkeypatch.setattr(train.models, "fit_model", fit_model)
    df = pd.DataFrame({"position": ["GK"], "total_points": [1.0], "x": [0.0]})
    weights = {"a": 1.0 - 1e-7, "small": 1e-7, "zero": 0.0}
    ensemble = train.fit_position_ensembles(df, ["x"], {"GK": weights})["GK"]
    assert calls == ["a", "small"]
    np.testing.assert_allclose(ensemble.predict(df[["x"]]), [1.0])
