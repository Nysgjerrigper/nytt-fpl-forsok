"""Process-isolated CPU CNN proxy for parents that have loaded native GBM libraries.

No Torch or tensor state enters the parent. A fresh worker owns every fitted model;
protocol failures/native exits fail closed without any automatic retry.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd

from fpl import config
from fpl.model.sequence_cnn import SequenceCNNRegressor

_MAX_RESPONSE_BYTES = 1024 * 1024
_ARRAY_ATTRIBUTES = {"feature_names_in_", "sequence_median_", "sequence_mean_", "sequence_scale_",
                     "context_median_", "context_mean_", "context_scale_"}


class IsolatedSequenceCNNRegressor(SequenceCNNRegressor):
    """Same estimator parameters as the direct CNN, with persistent fresh CPU state."""

    def _save_frame(self, X: pd.DataFrame) -> Path:
        if not isinstance(X, pd.DataFrame) or X.empty or X.columns.duplicated().any():
            raise ValueError("CNN proxy requires a nonempty DataFrame with unique columns")
        if not all(isinstance(column, str) for column in X.columns):
            raise ValueError("CNN proxy feature names must be strings")
        try:
            values = X.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("CNN proxy features must be numeric") from exc
        path = Path(self._temporary_.name) / "frame.npz"
        np.savez(path, values=values, columns=np.asarray(X.columns, dtype=str))
        return path

    def _start_worker(self) -> None:
        self._temporary_ = tempfile.TemporaryDirectory(prefix="fpl-cnn-")
        self._stderr_path_ = Path(self._temporary_.name) / "stderr.log"
        self._stderr_file_ = self._stderr_path_.open("wb")
        environment = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = str(config.RESEARCH_MODEL_THREADS)
        self._buffer_ = b""
        self._worker_ = subprocess.Popen(
            [sys.executable, "-m", "fpl.model.cnn_worker"], cwd=config.ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr_file_, env=environment)
        self.worker_pid_ = self._worker_.pid

    def _stderr_fragment(self) -> str:
        try:
            return self._stderr_path_.read_bytes()[-2048:].decode("utf-8", errors="replace")
        except (AttributeError, OSError):
            return ""

    def _worker_failure(self, reason: str) -> RuntimeError:
        code = self._worker_.poll()
        signal_detail = f" (shell exit {128 - code})" if code is not None and code < 0 else ""
        fragment = self._stderr_fragment()
        return RuntimeError(f"Isolated CNN worker {reason}; exit code {code}{signal_detail}; stderr: {fragment}")

    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        worker = self._worker_
        if worker.poll() is not None:
            raise self._worker_failure("exited before request")
        try:
            worker.stdin.write((json.dumps(request, allow_nan=False) + "\n").encode("utf-8"))
            worker.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise self._worker_failure("closed its input") from exc
        deadline = time.monotonic() + config.CNN_WORKER_TIMEOUT_SECONDS
        with selectors.DefaultSelector() as selector:
            selector.register(worker.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer_:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._worker_failure("timed out without a complete response")
                events = selector.select(min(remaining, .1))
                if events:
                    chunk = os.read(worker.stdout.fileno(), 65536)
                    if not chunk:
                        # Give a just-exiting child a bounded chance to publish its code.
                        try:
                            worker.wait(timeout=.1)
                        except subprocess.TimeoutExpired:
                            pass
                        raise self._worker_failure("closed stdout without a complete response")
                    self._buffer_ += chunk
                    if len(self._buffer_) > _MAX_RESPONSE_BYTES:
                        raise self._worker_failure("response exceeded the 1 MB protocol bound")
                elif worker.poll() is not None:
                    raise self._worker_failure("exited without a complete response")
        line, self._buffer_ = self._buffer_.split(b"\n", 1)
        try:
            response = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise self._worker_failure("returned malformed protocol JSON") from exc
        if not isinstance(response, dict) or not isinstance(response.get("ok"), bool):
            raise self._worker_failure("returned an invalid protocol response")
        if not response["ok"]:
            error = response.get("error", "unknown validation failure")
            if response.get("error_type") in ("ValueError", "TypeError"):
                raise ValueError(error)
            raise RuntimeError(f"Isolated CNN worker failed: {response.get('error_type')}: {error}")
        return response

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> IsolatedSequenceCNNRegressor:
        """Start a fresh worker and fit exact float64/Unicode NPZ inputs, never retrying."""
        self.close()
        self._validate_parameters()
        try:
            self._start_worker()
            path = self._save_frame(X)
            target = np.asarray(y, dtype=np.float64)
            if target.ndim != 1 or target.shape != (len(X),) or not np.isfinite(target).all():
                raise ValueError("CNN proxy y must be a finite one-dimensional target matching X")
            target_path = Path(self._temporary_.name) / "target.npy"
            np.save(target_path, target, allow_pickle=False)
            parameters = {key: value.item() if isinstance(value, np.generic) else value
                          for key, value in self.get_params(deep=False).items()}
            response = self._request({"command": "fit", "frame": str(path), "target": str(target_path),
                                      "parameters": parameters})
            for name, value in response["metadata"].items():
                setattr(self, name, np.asarray(value, dtype=object if name == "feature_names_in_" else float)
                        if name in _ARRAY_ATTRIBUTES else value)
            self.backend = "isolated_cpu_worker"
            self.backend_ = self.backend
            self._fitted_ = True
            return self
        except BaseException:
            self.close()
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict in the retained CPU worker; only finite NumPy points return to parent."""
        if not getattr(self, "_fitted_", False):
            raise RuntimeError("Isolated CNN estimator is not fitted or its worker was closed")
        path = self._save_frame(X)
        output_path = Path(self._temporary_.name) / "prediction.npy"
        try:
            response = self._request({"command": "predict", "frame": str(path), "output": str(output_path)})
            prediction = np.load(output_path, allow_pickle=False)
            if prediction.shape != (len(X),) or response.get("rows") != len(X) or not np.isfinite(prediction).all():
                raise RuntimeError("Isolated CNN worker returned invalid predictions")
            return prediction
        except RuntimeError:
            self.close()
            raise

    def close(self) -> None:
        """Gracefully close worker, then terminate/kill with bounded waits and clean inputs."""
        self._fitted_ = False
        worker = getattr(self, "_worker_", None)
        if worker is not None:
            if worker.poll() is None:
                try:
                    worker.stdin.write(b'{"command":"close"}\n')
                    worker.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    worker.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    worker.terminate()
                    try:
                        worker.wait(timeout=.5)
                    except subprocess.TimeoutExpired:
                        worker.kill()
                        worker.wait(timeout=.5)
            for stream in (worker.stdin, worker.stdout):
                if stream is not None:
                    stream.close()
            self._worker_ = None
        stderr = getattr(self, "_stderr_file_", None)
        if stderr is not None:
            stderr.close()
            self._stderr_file_ = None
        temporary = getattr(self, "_temporary_", None)
        if temporary is not None:
            try:
                self.worker_stderr_ = self._stderr_path_.read_text(errors="replace")
            except OSError:
                self.worker_stderr_ = ""
            temporary.cleanup()
            self._temporary_ = None

    def __enter__(self) -> IsolatedSequenceCNNRegressor:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Interpreter teardown can have already disposed subprocess primitives.
            pass
