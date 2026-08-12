"""Focused contracts for the position-specialist research registry."""
import builtins

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fpl.model import models, tuning


TOURNAMENT_EXPERTS = {
    "catboost_mae", "catboost_rmse", "lightgbm_l2", "lightgbm_l1",
    "lightgbm_huber", "xgboost_squared_error", "xgboost_absolute_error",
    "hist_gradient_boosting_absolute", "hist_gradient_boosting_squared",
    "extra_trees_tuned", "linear_svr_tuned", "realmlp", "tabm", "tabr",
}


def test_tournament_experts_have_complete_specs_without_expanding_production():
    assert TOURNAMENT_EXPERTS <= models.EXPERT_SPECS.keys()
    assert set(models.REGISTERED_MODEL_NAMES) == set(models.EXPERT_SPECS)
    assert TOURNAMENT_EXPERTS <= models.FACTORIES.keys()
    assert TOURNAMENT_EXPERTS.isdisjoint(models.MODEL_NAMES)
    for name in TOURNAMENT_EXPERTS:
        spec = models.EXPERT_SPECS[name]
        assert spec.search_space is not None
        assert isinstance(spec.default_params(7), dict)
        assert spec.preprocessing
        assert spec.seed == 0
        assert spec.provenance


def test_neural_adapter_is_lazy_and_uses_fold_local_imputer(monkeypatch):
    real_import = builtins.__import__
    seen = []

    def tracking_import(name, *args, **kwargs):
        if name.startswith("pytabkit"):
            seen.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracking_import)
    model = models.build_registered_model("tabm")
    assert isinstance(model, Pipeline)
    assert model.named_steps["impute"].strategy == "constant"
    assert seen == []


def test_missing_pytabkit_is_a_clear_optional_dependency_error(monkeypatch):
    real_import_module = models.importlib.import_module

    def missing(name, *args, **kwargs):
        if name == "pytabkit":
            raise ImportError("not installed")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(models.importlib, "import_module", missing)
    X = pd.DataFrame({"a": [np.nan, 1.0]})
    with pytest.raises(models.OptionalModelDependencyError, match="research-only expert"):
        models.fit_model("realmlp", X, pd.Series([0.0, 1.0]), position="MID")


def test_tabr_fails_cleanly_when_faiss_is_unavailable(monkeypatch):
    monkeypatch.setattr(models.importlib.util, "find_spec", lambda name: None)
    X = pd.DataFrame({"a": [0.0, 1.0]})
    with pytest.raises(models.OptionalModelDependencyError, match="FAISS"):
        models.fit_model("tabr", X, pd.Series([0.0, 1.0]), position="MID")


def test_generalized_tuning_builds_non_gbm_candidate():
    params = {"loss": "absolute_error", "max_iter": 2, "random_state": 7}
    model = tuning._build_model("hist_gradient_boosting_absolute", params,
                                seed=7, position="MID")
    assert isinstance(model, Pipeline)
    assert model.named_steps["model"].random_state == 7


def test_registered_default_uses_the_requested_seed():
    model = models.build_registered_model("extra_trees_tuned", seed=19)
    assert model.named_steps["model"].random_state == 19
