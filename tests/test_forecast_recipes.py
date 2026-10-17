"""Explicit recipes preserve named loss, seed, feature order and temporal paths."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from fpl import features
from fpl.model import models, train, predict
from fpl.model.recipes import resolve_fit_recipe


POSITIONS = ("GK", "DEF", "MID", "FWD")


def _weights():
    return {position: {"catboost_mae": 1.0} for position in POSITIONS}


def _recipe():
    return {position: {"catboost_mae": {"depth": 3, "iterations": 5}} for position in POSITIONS}


def test_recipe_resolves_named_loss_and_seed_without_mutating_overrides():
    supplied = _recipe()
    before = deepcopy(supplied)
    resolved = resolve_fit_recipe(_weights(), supplied, 7)
    assert supplied == before
    for position in POSITIONS:
        params = resolved[position]["catboost_mae"]
        assert params["loss_function"] == "MAE"
        assert params["random_seed"] == 7
        estimator = models.build_registered_model("catboost_mae", params=params, seed=7)
        assert estimator.get_params()["loss_function"] == "MAE"
        assert estimator.get_params()["random_seed"] == 7
    resolved["MID"]["catboost_mae"]["depth"] = 8
    assert supplied == before


@pytest.mark.parametrize("seed", [None, True, -1, 1.5])
def test_invalid_or_missing_seed_is_rejected(seed):
    with pytest.raises(ValueError):
        resolve_fit_recipe(_weights(), _recipe(), seed)


def test_seed_without_parameters_is_rejected():
    with pytest.raises(ValueError):
        resolve_fit_recipe(_weights(), None, 0)


@pytest.mark.parametrize("alias", ["random_state", "seed"])
def test_noncanonical_seed_alias_cannot_conflict_with_registered_seed(alias):
    params = _recipe()
    params["FWD"]["catboost_mae"][alias] = 7
    with pytest.raises(ValueError, match="registered seed"):
        resolve_fit_recipe(_weights(), params, 7)


def test_real_explicit_fits_bypass_all_ambient_tuned_artifacts(monkeypatch):
    frame = pd.DataFrame([{"position": position, "value": float(i), "total_points": float(i % 4),
                           "minutes": 90, "GW_global": i + 1}
                          for position in POSITIONS for i in range(16)])
    monkeypatch.setattr(models, "_tuned_params", lambda *a, **k: pytest.fail("ambient tuned loading"))
    fitted = train.fit_position_ensembles(frame, ["value"], _weights(), params_by_pos=_recipe(), seed=7)
    assert set(fitted) == set(POSITIONS)
    for position, ensemble in fitted.items():
        result = ensemble.predict(frame.loc[frame.position == position, ["value"]])
        assert len(result) == 16 and np.isfinite(result).all()


@pytest.mark.parametrize("mutation", ["missing_position", "extra_member", "missing_member", "wrong_loss", "seed_conflict", "bad_alias", "nonfinite"])
def test_incomplete_or_mislabeled_recipes_are_rejected(mutation):
    params = _recipe()
    if mutation == "missing_position":
        params.pop("FWD")
    elif mutation == "extra_member":
        params["MID"]["catboost_rmse"] = {}
    elif mutation == "missing_member":
        params["MID"] = {}
    else:
        key, value = {"wrong_loss": ("loss_function", "RMSE"), "seed_conflict": ("random_seed", 9),
                      "bad_alias": ("objective", "MAE"), "nonfinite": ("learning_rate", float("nan"))}[mutation]
        params["MID"]["catboost_mae"][key] = value
    with pytest.raises(ValueError):
        resolve_fit_recipe(_weights(), params, 7)


def test_fitter_passes_resolved_recipe_seed_labels_and_ordered_features(monkeypatch):
    frame = pd.DataFrame([{"position": pos, "a": 1., "b": 2., "total_points": 3.,
                           "minutes": 90, "GW_global": 136} for pos in POSITIONS])
    calls = []

    class Stub:
        def predict(self, X):
            return np.zeros(len(X))

    def fit(name, X, y, **kwargs):
        calls.append((name, X.columns.tolist(), kwargs))
        return Stub()

    monkeypatch.setattr(train.models, "fit_model", fit)
    train.fit_position_ensembles(frame, ["b", "a"], _weights(), params_by_pos=_recipe(), seed=7)
    assert len(calls) == 4
    for name, columns, kwargs in calls:
        assert name == "catboost_mae" and columns == ["b", "a"]
        assert kwargs["params"]["loss_function"] == "MAE" and kwargs["params"]["random_seed"] == 7
        assert kwargs["seed"] == 7
        assert kwargs["gw"].tolist() == [136] and kwargs["minutes"].tolist() == [90]


def test_invalid_late_position_recipe_fails_before_any_member_fit(monkeypatch):
    frame = pd.DataFrame([{"position": pos, "a": 1., "total_points": 3.} for pos in POSITIONS])
    params = _recipe()
    params["FWD"]["catboost_mae"]["loss_function"] = "RMSE"
    monkeypatch.setattr(train.models, "fit_model", lambda *a, **k: pytest.fail("partial fit before admission"))
    with pytest.raises(ValueError):
        train.fit_position_ensembles(frame, ["a"], _weights(), params_by_pos=params, seed=7)


@pytest.mark.parametrize("origin", [False, True])
@pytest.mark.parametrize("options", [
    {"weight_strategy": "nnls"}, {"mid_gate": object()}, {"seed": None},
    {"feature_cols": []}, {"feature_cols": ["value", "value"]},
    {"feature_cols": ["total_points"]}, {"feature_cols": ["missing"]},
])
def test_unsupported_forecast_recipe_fails_before_hidden_fits(monkeypatch, origin, options):
    frame = pd.DataFrame({"value": [50.], "total_points": [2.]})
    monkeypatch.setattr(predict, "fit_holdout_weights", lambda *a, **k: pytest.fail("hidden holdout fit"))
    monkeypatch.setattr(predict, "fit_position_ensembles", lambda *a, **k: pytest.fail("premature member fit"))
    kwargs = {"params_by_pos": _recipe(), "seed": 7, "weight_strategy": "single:catboost_mae"}
    columns = options.get("feature_cols", ["value"])
    kwargs.update({key: value for key, value in options.items() if key != "feature_cols"})
    with pytest.raises(ValueError):
        if origin:
            predict.origin_based_predictions(frame, frame, columns, 137, 138, 2, **kwargs)
        else:
            predict.walk_forward_predictions(frame, columns, 137, 138, **kwargs)


def test_calibration_is_excluded_from_explicit_recipe_path(monkeypatch):
    monkeypatch.setattr(predict, "fit_level_calibration", lambda *a, **k: pytest.fail("hidden calibration fit"))
    with pytest.raises(ValueError):
        predict.walk_forward_predictions(pd.DataFrame({"value": [50.]}), ["value"], 137, 138,
            weight_strategy="single:catboost_mae", params_by_pos=_recipe(), seed=7, calibrate_level=True)


@pytest.mark.parametrize("origin", [False, True])
def test_forecast_loops_use_explicit_recipe_and_preserve_scoring_identity(monkeypatch, origin):
    raw = pd.DataFrame([
        {"player_id": i+1, "name": f"p{i}", "position": pos, "team": "AAA", "opponent_team": "BBB",
         "GW_global": gw, "was_home": 1, "fixture": gw, "minutes": 90., "value": 50.,
         "total_points": float(gw), **{col: 0. for col in features.FORM_STATS if col != "minutes"}}
        for gw in range(1, 6) for i, pos in enumerate(POSITIONS)
    ])
    frame = features.build_feature_frame(raw)
    calls = []
    monkeypatch.setattr(predict, "fit_holdout_weights", lambda *a, **k: pytest.fail("single recipe invoked holdout fitting"))

    def fit(training, columns, weights, **kwargs):
        calls.append((training.GW_global.max(), columns, weights, kwargs))
        return {}

    monkeypatch.setattr(predict, "fit_position_ensembles", fit)
    monkeypatch.setattr(predict, "predict_by_position", lambda rows, *a, **k: np.full(len(rows), 2.))
    kwargs = {"params_by_pos": _recipe(), "seed": 7, "weight_strategy": "single:catboost_mae", "retrain_every": 1}
    if origin:
        result = predict.origin_based_predictions(frame, raw, ["value"], 4, 5, 2, **kwargs)
        assert set(result.origin_gw) == {4, 5}
    else:
        result = predict.walk_forward_predictions(frame, ["value"], 4, 5, **kwargs)
    assert [call[0] for call in calls] == [3, 4]
    assert {"minutes", "fixture", "opponent_team", "was_home", "actual_total_points"} <= set(result)
    for _, columns, weights, options in calls:
        assert columns == ["value"] and weights == _weights()
        assert options["seed"] == 7
        assert options["params_by_pos"]["MID"]["catboost_mae"]["loss_function"] == "MAE"
