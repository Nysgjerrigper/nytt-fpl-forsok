"""
Sanity checks for the experiment log (fpl.experiment.log_result). These guard
the two properties that make the log trustworthy as a run record: it must
actually accumulate rows (one per run, header included) rather than overwrite,
and it must tolerate the schema growing - a later run that reports a brand-new
metric has to union the columns in, not crash on a mismatched CSV shape.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl.experiment import log_result


def test_two_calls_produce_header_and_two_data_rows(tmp_path):
    # Route the log to a temp CSV so the test never touches the real
    # experiments/results.csv, then log two runs and confirm both landed.
    results_path = tmp_path / "results.csv"

    log_result("first", {"lr": 0.1}, {"mae": 2.5}, results_path=str(results_path))
    log_result("second", {"lr": 0.2}, {"mae": 2.3}, results_path=str(results_path))

    assert results_path.exists()

    df = pd.read_csv(results_path)
    # Two data rows; the CSV header is columns, not a counted row.
    assert len(df) == 2
    assert "timestamp" in df.columns and "name" in df.columns
    assert list(df["name"]) == ["first", "second"]
    # The metric was flattened into its own column, not buried in params.
    assert "mae" in df.columns


def test_new_metric_key_unions_columns_without_crashing(tmp_path):
    # The whole point of the union logic: a later run reporting a metric the
    # first run never measured must widen the schema, leaving the old row blank
    # in the new column (a missing measurement, not a real zero).
    results_path = tmp_path / "results.csv"

    log_result("first", {}, {"mae": 2.5}, results_path=str(results_path))
    log_result("second", {}, {"mae": 2.3, "mase": 0.9}, results_path=str(results_path))

    df = pd.read_csv(results_path)
    assert len(df) == 2
    assert "mae" in df.columns and "mase" in df.columns
    # First run had no "mase" -> blank/NaN for that row; second run has it.
    assert pd.isna(df.loc[0, "mase"])
    assert df.loc[1, "mase"] == 0.9


def test_missing_experiments_dir_is_created(tmp_path):
    # log_result must mkdir its parent - callers shouldn't have to pre-create
    # experiments/ just to record a result.
    results_path = tmp_path / "nested" / "dir" / "results.csv"
    log_result("run", {"seed": 1}, {"score": 1.0}, results_path=str(results_path))
    assert results_path.exists()


def test_float_looking_string_column_survives_reread(tmp_path):
    # A short git hash like "37328e2" is valid float syntax; on the second
    # log_result the file is read back and re-written, and naive read_csv would
    # coerce that first-row value to 3732800.0 - silently destroying the exact
    # provenance the log exists to preserve. This pins the string round-trip.
    results_path = tmp_path / "results.csv"

    # Simulate a pre-existing log whose git_hash looks numeric-in-exponent form.
    pd.DataFrame(
        [{"timestamp": "t0", "git_hash": "37328e2", "name": "prior", "params": "{}", "mae": 2.5}]
    ).to_csv(results_path, index=False)

    log_result("next", {}, {"mae": 2.3}, results_path=str(results_path))

    # Read the git_hash column as a string, the way a human tracing a result would.
    df = pd.read_csv(results_path, dtype={"git_hash": str})
    assert df.loc[0, "git_hash"] == "37328e2"
