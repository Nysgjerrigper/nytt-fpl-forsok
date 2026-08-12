"""
Self-documenting experiment log.

Until now every experiment was a pile of ad-hoc prints hand-copied into
RESEARCH_LOG.md - which meant runs weren't reproducible (nobody recorded the
exact params) and the numbers drifted from the code that produced them. This
module appends one row per run to a CSV, stamping each with the wall-clock time
and the git commit it ran against, so a result can always be traced back to the
exact code and settings that produced it.

The CSV schema is deliberately not fixed up front: different experiments report
different metrics, so new metric columns are unioned in on the fly rather than
forcing every run to declare the same fields. That keeps the log honest (no
placeholder/zero-filled columns implying a metric was measured when it wasn't).
"""
import json
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

DEFAULT_RESULTS_PATH = "experiments/results.csv"

# Columns that always come first, in this order, so the human-facing CSV stays
# scannable no matter which metrics a given run happens to log.
_FIXED_COLUMNS = ["timestamp", "git_hash", "versions", "name", "params"]

# The libraries whose version can silently move a backtest number (TODO 4.2): the three
# GBMs above all, plus the numerics/solver stack they sit on. requirements.txt pins these,
# but the log records what ACTUALLY ran - a pin edit or a stray environment shows up here.
_VERSIONED_LIBS = ("catboost", "lightgbm", "xgboost", "scikit-learn", "numpy", "pandas",
                   "scipy", "pulp", "highspy", "torch", "pytabkit", "faiss-cpu")


def _library_versions():
    """{library: installed version} for the result-moving libraries, best-effort.

    A library that isn't installed (e.g. highspy on a CBC-only setup) is simply absent
    from the dict rather than crashing the logged run - same degrade-gracefully policy
    as _current_git_hash."""
    versions = {}
    for lib in _VERSIONED_LIBS:
        try:
            versions[lib] = version(lib)
        except PackageNotFoundError:
            continue
    return versions


def dataset_state(df, gw_col="GW_global"):
    """Data-state provenance for a run: how much data existed when it ran.

    Two experiments with identical code and params still aren't comparable if one saw
    three more played gameweeks (fetch.py appends new rounds in place, so the dataset
    mutates under the same path). Callers merge this into `params` so the row records
    the data snapshot alongside the knobs:
    log_result(name, {**params, **dataset_state(df)}, metrics).
    """
    return {"data_rows": int(len(df)), "data_max_gw": int(df[gw_col].max())}


def _current_git_hash():
    """Best-effort short commit hash; tolerate any failure.

    An experiment log is still useful without provenance (e.g. run from a
    tarball with no .git, or git not installed), so a missing hash must never
    be the thing that crashes a logged run - it degrades to empty instead.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def log_result(name, params, metrics, results_path=DEFAULT_RESULTS_PATH):
    """Append one row describing a single experiment run.

    ``params`` is stored as a single JSON string (an experiment's knobs are
    heterogeneous and we don't want each one spawning its own column), whereas
    each metric IS flattened into its own column so results can be sorted and
    compared across runs. When a run introduces a metric not seen before, the
    existing file's columns are unioned with the new ones and the file rewritten
    - older rows simply carry a blank in the new column, which correctly reads as
    "this metric wasn't measured that run" rather than a real zero.
    """
    row = {
        "timestamp": datetime.now().isoformat(),
        "git_hash": _current_git_hash(),
        "versions": json.dumps(_library_versions(), sort_keys=True),
        "name": name,
        "params": json.dumps(params, sort_keys=True, default=str),
    }
    # Each metric becomes its own column. dict() copies so a caller-supplied
    # mapping isn't mutated, and lets metric keys override nothing fixed above.
    for metric_name, value in dict(metrics).items():
        row[metric_name] = value

    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([row])
    if path.exists():
        # dtype=str on read-back is load-bearing: a short git hash like "37328e2"
        # is all-valid-float-syntax and pandas would silently coerce it to
        # 3732800.0, corrupting the one column whose whole job is provenance.
        # Reading everything as strings preserves prior rows verbatim; the new
        # row's own (correctly-typed) values are appended untouched.
        existing = pd.read_csv(path, dtype=str)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    # Keep the fixed provenance columns leftmost and in a stable order; append
    # any metric columns after them, preserving first-seen order for readability.
    metric_cols = [c for c in combined.columns if c not in _FIXED_COLUMNS]
    ordered = [c for c in _FIXED_COLUMNS if c in combined.columns] + metric_cols
    combined = combined[ordered]

    combined.to_csv(path, index=False)
    return path


def run_and_log(name, params, metric_fn, results_path=DEFAULT_RESULTS_PATH):
    """Sugar: run ``metric_fn(**params)``, log its returned metrics dict, return it.

    Handy for one-liner experiments where the callable and its params are the
    whole story; anything more involved should just call ``log_result`` directly.
    """
    metrics = metric_fn(**params)
    log_result(name, params, metrics, results_path=results_path)
    return metrics
