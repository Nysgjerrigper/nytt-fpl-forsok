"""Fresh-process CPU CNN JSON-lines worker; stdout is reserved for protocol bytes."""
from __future__ import annotations

from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from fpl.model.sequence_cnn import SequenceCNNRegressor

_ARRAY_ATTRIBUTES = ("feature_names_in_", "sequence_median_", "sequence_mean_", "sequence_scale_",
                     "context_median_", "context_mean_", "context_scale_")


def _frame(path: str) -> pd.DataFrame:
    with np.load(Path(path), allow_pickle=False) as payload:
        values, columns = payload["values"], payload["columns"]
    if values.ndim != 2 or columns.ndim != 1 or values.shape[1] != len(columns):
        raise ValueError("Invalid internal CNN matrix/schema payload")
    return pd.DataFrame(values, columns=columns.tolist())


def main() -> None:
    """Keep fitted CPU state alive for fit/predict requests; never retry failures."""
    estimator = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request.get("command")
            if command == "close":
                print(json.dumps({"ok": True}), flush=True)
                return
            # Library diagnostics go to the separately preserved stderr stream.
            with redirect_stdout(sys.stderr):
                if command == "fit":
                    estimator = None
                    candidate = SequenceCNNRegressor(**request["parameters"])
                    candidate.fit(_frame(request["frame"]), np.load(request["target"], allow_pickle=False))
                    estimator = candidate
                    metadata = {name: getattr(estimator, name).tolist() for name in _ARRAY_ATTRIBUTES}
                    metadata.update(n_features_in_=estimator.n_features_in_,
                                    context_columns_=estimator.context_columns_, loss_history_=estimator.loss_history_,
                                    fit_seconds_=estimator.fit_seconds_, target_mean_=estimator.target_mean_,
                                    target_scale_=estimator.target_scale_, device="cpu",
                                    convolution_input_channels=estimator.network_.convolution[0].in_channels,
                                    head_input_features=estimator.network_.head[0].in_features)
                    response = {"ok": True, "metadata": metadata}
                elif command == "predict":
                    if estimator is None:
                        raise ValueError("CNN worker is not fitted")
                    prediction = estimator.predict(_frame(request["frame"]))
                    np.save(request["output"], prediction, allow_pickle=False)
                    response = {"ok": True, "rows": len(prediction)}
                else:
                    raise ValueError("Unknown CNN worker command")
        except Exception as exc:
            response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
