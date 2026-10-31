"""Fresh-worker safety contracts; native CNN code never runs in a GBM parent."""
import importlib.util
import json
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

from fpl import config
from fpl.model import cnn_process
from fpl.model.cnn_process import IsolatedSequenceCNNRegressor
from fpl.model.sequence_cnn import SequenceCNNRegressor
from fpl.model.sequence_features import sequence_columns


def frame():
    result = pd.DataFrame(np.zeros((8, len(sequence_columns()))), columns=sequence_columns())
    for step in range(config.CNN_SEQUENCE_LENGTH):
        result[f"cnn_present_t{step}"] = 1.
    result["context"] = np.arange(8, dtype=float)
    return result


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional requirements-cnn.txt runtime is absent")
def test_single_fresh_worker_cpu_validation_in_gbm_parent():
    # One fresh synthetic driver validation. Never retry if its native worker fails.
    driver = r'''
import os, sys
import lightgbm, catboost, xgboost
import numpy as np
import pandas as pd
from fpl import config
from fpl.model.cnn_process import IsolatedSequenceCNNRegressor
from fpl.model.sequence_features import sequence_columns
assert 'torch' not in sys.modules
rng = np.random.default_rng(7)
X = pd.DataFrame(rng.normal(size=(12, len(sequence_columns()))), columns=sequence_columns())
for step in range(config.CNN_SEQUENCE_LENGTH):
    X[f'cnn_present_t{step}'] = float(step >= 3)
    if step < 3:
        X[[f'cnn_{stat}_t{step}' for stat in config.CNN_SEQUENCE_STATS]] = np.nan
X['context'] = np.linspace(0., 1., len(X))
X['empty_context'] = np.nan
y = np.linspace(1., 5., len(X))
model = IsolatedSequenceCNNRegressor(epochs=2, batch_size=4)
try:
    model.fit(X, y)
    initial_pid = model.worker_pid_
    assert initial_pid != os.getpid() and model.backend_ == 'isolated_cpu_worker'
    assert model.device == 'cpu' and model.convolution_input_channels == 17
    assert model.head_input_features == config.CNN_CHANNELS + 4
    prediction = model.predict(X)
    assert prediction.shape == (len(X),) and np.isfinite(prediction).all()
    arrays = ('sequence_median_', 'sequence_mean_', 'sequence_scale_', 'context_median_', 'context_mean_', 'context_scale_')
    before = {name: getattr(model, name).copy() for name in arrays}
    altered = X.copy()
    altered['context'] = 99999.
    altered['cnn_total_points_t7'] = 88888.
    model.predict(altered)
    for name in arrays:
        np.testing.assert_array_equal(before[name], getattr(model, name))
    for bad in (X[X.columns[::-1]], X.drop(columns='context')):
        try:
            model.predict(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('schema mismatch accepted')
    # Validation errors leave the retained worker usable.
    np.testing.assert_array_equal(prediction, model.predict(X))
    model.fit(X, y)
    assert model.worker_pid_ != initial_pid
    np.testing.assert_array_equal(prediction, model.predict(X))
    assert 'torch' not in sys.modules
    temporary = model._temporary_.name
finally:
    model.close()
assert not os.path.exists(temporary)
assert 'torch' not in sys.modules
print('isolated CPU validation passed')
'''
    result = subprocess.run([sys.executable, "-c", driver], cwd=config.ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, f"Fresh isolated validation failed; do not retry.\n{result.stdout}\n{result.stderr}"
    assert "isolated CPU validation passed" in result.stdout


def test_direct_fit_refuses_loaded_gbm_before_torch_import(monkeypatch):
    import builtins
    original = builtins.__import__
    def protect(name, *args, **kwargs):
        if name == "torch":
            pytest.fail("Torch import attempted after GBM load")
        return original(name, *args, **kwargs)
    monkeypatch.setitem(sys.modules, "lightgbm", object())
    monkeypatch.setattr(builtins, "__import__", protect)
    with pytest.raises(RuntimeError, match="unsafe after GBM"):
        SequenceCNNRegressor(epochs=1).fit(frame(), np.arange(8, dtype=float))


@pytest.mark.parametrize("script,expected", [
    ("import sys; sys.stderr.write('synthetic native exit evidence'); sys.stderr.flush(); sys.exit(139)", "exit code 139"),
    ("import sys; sys.stdout.write('{'); sys.stdout.flush(); sys.exit(0)", "without a complete response"),
])
def test_fake_native_exit_or_partial_protocol_fail_closed_and_cleanup(monkeypatch, script, expected):
    original = subprocess.Popen
    calls = []
    def fake_worker(args, **kwargs):
        calls.append(args)
        return original([sys.executable, "-c", script], **kwargs)
    monkeypatch.setattr(cnn_process.subprocess, "Popen", fake_worker)
    model = IsolatedSequenceCNNRegressor(epochs=1)
    with pytest.raises(RuntimeError, match=expected):
        model.fit(frame(), np.arange(8, dtype=float))
    assert len(calls) == 1 and model._worker_ is None and model._temporary_ is None
    if "139" in expected:
        assert "synthetic native exit evidence" in model.worker_stderr_


def test_partial_response_timeout_is_bounded_and_worker_cleaned(monkeypatch):
    original = subprocess.Popen
    workers = []
    def fake_worker(args, **kwargs):
        worker = original([sys.executable, "-c", "import sys,time; sys.stdout.write('{'); sys.stdout.flush(); time.sleep(30)"], **kwargs)
        workers.append(worker)
        return worker
    monkeypatch.setattr(cnn_process.subprocess, "Popen", fake_worker)
    monkeypatch.setattr(config, "CNN_WORKER_TIMEOUT_SECONDS", .05)
    model = IsolatedSequenceCNNRegressor(epochs=1)
    with pytest.raises(RuntimeError, match="timed out"):
        model.fit(frame(), np.arange(8, dtype=float))
    assert len(workers) == 1 and workers[0].poll() is not None
    assert model._worker_ is None and model._temporary_ is None


def test_input_type_errors_and_constructor_params_without_worker_fit():
    model = IsolatedSequenceCNNRegressor(epochs=2)
    assert model.get_params() == SequenceCNNRegressor(epochs=2).get_params()
    with tempfile.TemporaryDirectory() as root:
        class Temporary:
            name = root
        model._temporary_ = Temporary()
        for invalid in (np.zeros((3, 4)), pd.DataFrame(), frame().assign(context="text")):
            with pytest.raises(ValueError):
                model._save_frame(invalid)
        model._temporary_ = None
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(frame())
