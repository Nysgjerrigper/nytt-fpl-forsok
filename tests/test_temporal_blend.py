"""Earlier-only blend weights and mocked estimator wiring; no native fits."""
import sys
import types

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from fpl import config
from fpl.model import temporal_blend as blend


def forecasts():
    rows = []
    predictions = {"GK": (0., 4., 1.), "DEF": (0., 2., 3.), "MID": (2., 4., 0.), "FWD": (2., 2., 9.)}
    for index, position in enumerate(config.ONFIELD_POSITIONS):
        tree, cnn, actual = predictions[position]
        rows.append({"player_id": index + 1, "position": position, "origin_gw": 10, "GW": 10,
                     "fixture": 101, "actual_total_points": actual, "predicted_total_points": tree,
                     "value": 50., "minutes": 90., "team": f"Team {index}", "opponent_team": "Opponent", "was_home": 1})
    tree = pd.DataFrame(rows)
    cnn = tree.copy()
    cnn["predicted_total_points"] = [predictions[p][1] for p in tree.position]
    return cnn, tree


def test_handcomputed_clipped_weights_and_identical_prediction_tie():
    cnn, tree = forecasts()
    weights = blend.fit_earlier_weights(cnn, tree, 10, 10)
    assert weights == {"GK": .25, "DEF": 1., "MID": 0., "FWD": 0.}


def test_dgw_sum_precedes_weight_fit():
    cnn, tree = forecasts()
    # GK fixture errors cancel at round level: d=[4,-2], y-tree=[1,2].
    # Fixture fit gives w=0, whereas the summed round gives 3/2 clipped to 1.
    extra_tree = tree.iloc[[0]].assign(fixture=102, predicted_total_points=2., actual_total_points=4.)
    extra_cnn = extra_tree.assign(predicted_total_points=0.)
    tree = pd.concat([tree, extra_tree], ignore_index=True)
    cnn = pd.concat([cnn, extra_cnn], ignore_index=True)
    assert blend.fit_earlier_weights(cnn, tree, 10, 10)["GK"] == 1.


def test_future_labels_future_forecasts_and_overlapping_origins_cannot_fit_weights():
    cnn, tree = forecasts()
    expected = blend.fit_earlier_weights(cnn, tree, 10, 10)
    future_tree = tree.assign(origin_gw=11, GW=11, fixture=111, actual_total_points=1e100,
                              predicted_total_points=-1e100)
    future_cnn = future_tree.assign(predicted_total_points=1e100)
    overlapping_tree = tree.assign(origin_gw=9, actual_total_points=1e90, predicted_total_points=1e90)
    overlapping_cnn = overlapping_tree.assign(predicted_total_points=-1e90)
    actual = blend.fit_earlier_weights(pd.concat([cnn, future_cnn, overlapping_cnn]),
                                      pd.concat([tree, future_tree, overlapping_tree]), 10, 10)
    assert actual == expected


def test_blending_preserves_original_tree_order_metadata_and_exact_endpoints():
    cnn, tree = forecasts()
    tree = tree.iloc[::-1].copy()
    tree.index = [7, 6, 5, 4]
    cnn = cnn.sample(frac=1., random_state=1)
    weights = {"GK": .25, "DEF": 1., "MID": 0., "FWD": 0.}
    result = blend.blend_forecasts(cnn, tree, weights)
    columns = list(tree.columns.drop("predicted_total_points"))
    pd.testing.assert_frame_equal(result[columns], tree[columns])
    assert result.predicted_total_points.tolist() == [2., 2., 2., 1.]
    pd.testing.assert_frame_equal(blend.blend_forecasts(cnn, tree, dict.fromkeys(config.ONFIELD_POSITIONS, 0.)), tree)
    endpoint = blend.blend_forecasts(cnn, tree, dict.fromkeys(config.ONFIELD_POSITIONS, 1.))
    assert endpoint.predicted_total_points.tolist() == [2., 4., 2., 4.]


@pytest.mark.parametrize("change", ["pool", "outcomes", "value", "minutes", "fixture", "metadata", "duplicate", "positions", "origin", "nonfinite"])
def test_paired_validation_rejects_invalid_forecasts(change):
    cnn, tree = forecasts()
    if change == "pool":
        cnn = cnn.iloc[:-1]
    elif change == "outcomes":
        cnn.loc[0, "actual_total_points"] += 1
    elif change in ("value", "minutes", "fixture"):
        cnn.loc[0, change] += 1
    elif change == "metadata":
        cnn.loc[0, "opponent_team"] = "Other"
    elif change == "duplicate":
        cnn = pd.concat([cnn, cnn.iloc[[0]]])
    elif change == "positions":
        cnn.loc[0, "position"] = "unknown"
    elif change == "origin":
        cnn.loc[0, "origin_gw"] = 11
    else:
        cnn.loc[0, "predicted_total_points"] = np.nan
    with pytest.raises(ValueError):
        blend.fit_earlier_weights(cnn, tree, 10, 10)
    with pytest.raises(ValueError):
        blend.blend_forecasts(cnn, tree, dict.fromkeys(config.ONFIELD_POSITIONS, .5))


@pytest.mark.parametrize("weights", [{}, {"GK": .5}, dict.fromkeys(config.ONFIELD_POSITIONS, np.nan),
                                      dict.fromkeys(config.ONFIELD_POSITIONS, 1.1), dict.fromkeys(config.ONFIELD_POSITIONS, -1.)])
def test_invalid_weight_maps(weights):
    cnn, tree = forecasts()
    with pytest.raises(ValueError):
        blend.blend_forecasts(cnn, tree, weights)


@pytest.mark.parametrize("bounds", [(0, 10), (11, 10), (1.5, 10)])
def test_invalid_fitting_bounds(bounds):
    cnn, tree = forecasts()
    with pytest.raises(ValueError):
        blend.fit_earlier_weights(cnn, tree, *bounds)


@pytest.fixture
def mocked_members(monkeypatch):
    calls = []
    class Member:
        def __init__(self, **settings):
            self.settings = settings
            self.output = 6.
            calls.append(("cnn_constructor", settings))
        def fit(self, X, y):
            calls.append(("cnn_fit", X, y))
            return self
        def predict(self, X):
            return np.full(len(X), self.output)
        def close(self):
            calls.append(("cnn_close",))
    def fit_model(name, X, y, **settings):
        calls.append(("tree_fit", name, X, y, settings))
        model = types.SimpleNamespace(predict=lambda X: np.full(len(X), 2.))
        return model
    monkeypatch.setitem(sys.modules, "fpl.model.models", types.SimpleNamespace(fit_model=fit_model))
    monkeypatch.setitem(sys.modules, "fpl.model.cnn_process", types.SimpleNamespace(IsolatedSequenceCNNRegressor=Member))
    return calls


@pytest.mark.parametrize("weight,tree_count,cnn_count,expected", [(0., 1, 0, 2.), (1., 0, 1, 6.), (.25, 1, 1, 3.)])
def test_lazy_components_skip_unused_and_forward_explicit_params(mocked_members, weight, tree_count, cnn_count, expected):
    X, y = pd.DataFrame({"feature": [1., 2.]}), np.array([3., 4.])
    model = blend.TemporalCNNTreeBlendRegressor(cnn_weight=weight, cnn_params={"epochs": 2, "seed": 9},
                                              tree_params={"depth": 6}, seed=1, position="MID")
    assert clone(model).get_params() == model.get_params()
    model.fit(X, y)
    np.testing.assert_array_equal(model.predict(X), [expected, expected])
    tree_calls = [call for call in mocked_members if call[0] == "tree_fit"]
    cnn_calls = [call for call in mocked_members if call[0] == "cnn_constructor"]
    assert len(tree_calls) == tree_count and len(cnn_calls) == cnn_count
    if tree_calls:
        assert tree_calls[0][1] == "catboost_rmse"
        assert tree_calls[0][2] is X and tree_calls[0][3] is y
        assert tree_calls[0][4] == {"position": "MID", "params": {"depth": 6}, "seed": 1}
    if cnn_calls:
        assert cnn_calls[0][1] == {"epochs": 2, "seed": 1}
        fit = [call for call in mocked_members if call[0] == "cnn_fit"][0]
        assert fit[1] is X and fit[2] is y
    model.close()


def test_refit_closes_previous_cnn_and_invalid_member_output_fails(mocked_members):
    X, y = pd.DataFrame({"feature": [1., 2.]}), np.array([3., 4.])
    model = blend.TemporalCNNTreeBlendRegressor(cnn_weight=.5, cnn_params={}, tree_params={}).fit(X, y)
    model.fit(X, y)
    assert sum(call[0] == "cnn_close" for call in mocked_members) == 1
    model.cnn_.predict = lambda X: np.array([1.])
    with pytest.raises(ValueError, match="invalid predictions"):
        model.predict(X)
    model.close()
