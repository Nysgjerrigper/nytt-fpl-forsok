"""Ambient tuning artifacts must not expose later labels to an earlier training fold."""
import json

import numpy as np
import pytest

from fpl.model import models


@pytest.fixture
def artifact_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(models.config, "MODELS_DIR", tmp_path)
    return tmp_path


def _save(directory, payload, name="catboost"):
    path = directory / f"tuned_params_GK_{name}.json"
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.parametrize("payload", [
    {}, {"_meta": None}, {"_meta": []}, {"_meta": {}},
    *({"_meta": {"train_max_gw": value}} for value in
      [None, True, False, "136", -1, 136.5, np.inf, np.nan, 152]),
])
def test_invalid_or_future_metadata_requires_explicit_bound(artifact_dir, payload):
    path = _save(artifact_dir, {**payload, "depth": 4})
    assert models._tuned_params("catboost", "GK") == {"depth": 4}
    with pytest.raises(ValueError, match="training cutoff 136") as exc:
        models._tuned_params("catboost", "GK", max_train_gw=136)
    assert str(path) in str(exc.value)
    assert "model='catboost', position='GK'" in str(exc.value)


@pytest.mark.parametrize("cap", [0, 135, 136, 136.0])
def test_valid_cutoff_metadata_is_removed_from_parameters(artifact_dir, cap):
    _save(artifact_dir, {"_meta": {"train_max_gw": cap}, "depth": 4})
    assert models._tuned_params("catboost", "GK", max_train_gw=136) == {"depth": 4}


def test_missing_artifact_still_uses_defaults(artifact_dir):
    assert models._tuned_params("catboost", "GK", max_train_gw=136) is None


def test_future_outcomes_cannot_change_early_fold_through_tuned_parameters(artifact_dir, monkeypatch):
    fitted_parameters = []

    class SpyEstimator:
        def __init__(self, params):
            self.params = params

        def fit(self, X, y):
            fitted_parameters.append(self.params)

    monkeypatch.setattr(models, "build_registered_model", lambda name, params, **kw: SpyEstimator(params))
    X, y, gw = np.zeros((2, 1)), np.ones(2), [135, 136]
    # Simulate a later tuning run choosing depth from future evaluation outcomes.
    for future_outcome in (2, 8):
        _save(artifact_dir, {"_meta": {"train_max_gw": 152}, "depth": future_outcome})
        with pytest.raises(ValueError, match="training cutoff 136"):
            models.fit_model("catboost", X, y, position="GK", gw=gw)
    assert fitted_parameters == []
    _save(artifact_dir, {"_meta": {"train_max_gw": 136}, "depth": 4})
    models.fit_model("catboost", X, y, position="GK", gw=gw)
    assert fitted_parameters == [{"depth": 4}]


def test_explicit_parameters_bypass_ambient_read(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Explicit params must not read ambient tuning files")

    class SpyEstimator:
        def fit(self, X, y):
            pass

    monkeypatch.setattr(models, "_tuned_params", forbidden)
    monkeypatch.setattr(models, "build_registered_model", lambda *a, **kw: SpyEstimator())
    models.fit_model("catboost", np.zeros((1, 1)), [1], position="GK", gw=[136], params={})


@pytest.mark.parametrize("name", ["catboost_hurdle", "catboost_hurdle3", "lgbm_rank"])
@pytest.mark.parametrize("kwargs", [{"params": {}}, {"seed": 0}])
def test_special_models_reject_silently_ignored_overrides(name, kwargs):
    with pytest.raises(ValueError, match="does not support explicit params or seed"):
        models.fit_model(name, np.zeros((1, 1)), [1], minutes=[90], gw=[136], **kwargs)


@pytest.mark.parametrize("name, base, estimator", [
    ("catboost_hurdle", "catboost", "TwoStageHurdle"),
    ("catboost_hurdle3", "catboost", "ThreeClassHurdle"),
    ("lgbm_rank", "lightgbm", "LambdaRankScorer"),
])
def test_special_models_validate_base_artifact_before_fit(artifact_dir, monkeypatch, name, base, estimator):
    _save(artifact_dir, {"_meta": {"train_max_gw": 152}}, name=base)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid ambient artifact must fail before estimator construction")

    monkeypatch.setattr(models, estimator, forbidden)
    with pytest.raises(ValueError, match="training cutoff 136"):
        models.fit_model(name, np.zeros((1, 1)), [1], position="GK", minutes=[90], gw=[136])


@pytest.mark.parametrize("gw", [[], [np.nan], [np.inf], [135.5], [[136]]])
def test_invalid_training_gameweeks_cannot_disable_guard(gw):
    with pytest.raises(ValueError, match="nonempty finite integer vector"):
        models.fit_model("catboost", np.zeros((1, 1)), [1], position="GK", gw=gw)
