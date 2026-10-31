"""Small synthetic CPU checks; these are not historical CNN performance evidence."""
import importlib.util
import subprocess
import sys
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from fpl import config
from fpl.model.sequence_cnn import SequenceCNNRegressor
from fpl.model.sequence_features import sequence_columns


def structured_frame(rows=12):
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(rng.normal(size=(rows, len(sequence_columns()))), columns=sequence_columns())
    for step in range(config.CNN_SEQUENCE_LENGTH):
        frame[f"cnn_present_t{step}"] = float(step >= 4)
        if step < 4:
            frame[[f"cnn_{stat}_t{step}" for stat in config.CNN_SEQUENCE_STATS]] = np.nan
    frame["context"] = np.linspace(-1., 1., rows)
    frame["missing_context"] = np.nan
    return frame


def tiny_model(**params):
    return SequenceCNNRegressor(epochs=2, batch_size=4, **params)


isolated_runtime = pytest.mark.skipif(
    os.environ.get("FPL_CNN_ISOLATED_TEST") != "1",
    reason="Native CNN contracts run in a fresh subprocess; mixed GBM/Torch runtime crashed.",
)


@isolated_runtime
def test_tiny_cpu_fit_shape_units_and_deterministic_refit():
    X = structured_frame()
    y = np.linspace(1., 6., len(X))
    model = tiny_model().fit(X, y)
    prediction = model.predict(X)
    other = tiny_model().fit(X, y)
    assert prediction.shape == (len(X),) and np.isfinite(prediction).all()
    np.testing.assert_array_equal(prediction, other.predict(X))
    assert len(model.loss_history_) == 2 and np.isfinite(model.loss_history_).all()
    assert model.fit_seconds_ > 0
    assert next(model.network_.parameters()).device.type == "cpu"
    assert model.network_.convolution[0].in_channels == 17
    assert model.network_.head[0].in_features == config.CNN_CHANNELS + 4
    assert model.target_mean_ == pytest.approx(np.mean(y))
    assert model.target_scale_ == pytest.approx(np.std(y))
    assert clone(model).get_params() == model.get_params()


@isolated_runtime
def test_prediction_does_not_change_training_preprocessing():
    X = structured_frame()
    model = tiny_model().fit(X, np.arange(len(X), dtype=float))
    names = ["sequence_median_", "sequence_mean_", "sequence_scale_", "context_median_", "context_mean_", "context_scale_"]
    before = {name: getattr(model, name).copy() for name in names}
    altered = X.copy()
    altered["context"] = 1e6
    altered["cnn_total_points_t7"] = 1e5
    model.predict(altered)
    for name in names:
        np.testing.assert_array_equal(getattr(model, name), before[name])
    observed = X[[f"cnn_total_points_t{step}" for step in range(config.CNN_SEQUENCE_LENGTH)]].to_numpy()
    finite = observed[np.isfinite(observed)]
    assert model.sequence_median_[0] == pytest.approx(np.median(finite))
    assert model.sequence_mean_[0] == pytest.approx(np.mean(finite))
    assert model.sequence_scale_[0] == pytest.approx(np.std(finite))


@isolated_runtime
def test_nan_allmissing_padding_and_constant_targets_are_finite():
    X = structured_frame()
    for stat in config.CNN_SEQUENCE_STATS:
        X[[f"cnn_{stat}_t{step}" for step in range(config.CNN_SEQUENCE_LENGTH)]] = np.nan
    X[[f"cnn_present_t{step}" for step in range(config.CNN_SEQUENCE_LENGTH)]] = 0.
    model = tiny_model().fit(X, np.full(len(X), 3.))
    assert np.isfinite(model.predict(X)).all()
    np.testing.assert_array_equal(model.sequence_median_, np.zeros(8))
    np.testing.assert_array_equal(model.sequence_mean_, np.zeros(8))
    np.testing.assert_array_equal(model.sequence_scale_, np.ones(8))
    assert model.context_median_[1] == 0 and model.context_scale_[1] == 1
    assert model.target_scale_ == 1


@isolated_runtime
def test_predict_rejects_reordered_dropped_or_extra_schema():
    X = structured_frame()
    model = tiny_model().fit(X, np.arange(len(X), dtype=float))
    for invalid in (X[X.columns[::-1]], X.drop(columns="context"), X.assign(extra=1.)):
        with pytest.raises(ValueError, match="columns/order"):
            model.predict(invalid)


@pytest.mark.parametrize("kind", ["padded_value", "nonbinary", "missing_mask", "infinite", "missing_column", "empty", "nonnumeric"])
def test_invalid_inputs(kind):
    X = structured_frame()
    if kind == "padded_value":
        X.loc[0, "cnn_total_points_t0"] = 999.
    elif kind == "nonbinary":
        X.loc[0, "cnn_present_t0"] = .5
    elif kind == "missing_mask":
        X.loc[0, "cnn_present_t0"] = np.nan
    elif kind == "infinite":
        X.loc[0, "context"] = np.inf
    elif kind == "missing_column":
        X = X.drop(columns="cnn_total_points_t7")
    elif kind == "empty":
        X = X.iloc[:0]
    elif kind == "nonnumeric":
        X["context"] = "name metadata"
    with pytest.raises(ValueError):
        tiny_model().fit(X, np.arange(len(X), dtype=float))


@pytest.mark.parametrize("target", [np.zeros(11), np.zeros((12, 1)), np.full(12, np.nan), np.full(12, np.inf)])
def test_invalid_targets(target):
    with pytest.raises(ValueError, match="finite one-dimensional"):
        tiny_model().fit(structured_frame(), target)


@pytest.mark.parametrize("params", [{"epochs": 0}, {"batch_size": True}, {"channels": 1.5}, {"hidden": -1},
                                    {"seed": -1}, {"learning_rate": 0}, {"weight_decay": -1}, {"grad_clip": np.inf}])
def test_invalid_parameters(params):
    settings = {"epochs": 2, "batch_size": 4, **params}
    with pytest.raises(ValueError):
        SequenceCNNRegressor(**settings).fit(structured_frame(), np.arange(12, dtype=float))


def test_module_import_does_not_import_torch():
    code = "import sys; import fpl.model.sequence_cnn; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional requirements-cnn.txt runtime is absent")
def test_native_cpu_contracts_in_fresh_subprocess():
    env = {**os.environ, "FPL_CNN_ISOLATED_TEST": "1"}
    result = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", __file__, "-q", "-k",
         "tiny_cpu_fit or prediction_does_not or nan_allmissing or predict_rejects"],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 passed" in result.stdout
