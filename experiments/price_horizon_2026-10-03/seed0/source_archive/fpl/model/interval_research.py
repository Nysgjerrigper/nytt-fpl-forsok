"""Predeclared static-model interval diagnostics, separate from production forecasts.

Fixture quantiles are summed into approximate player-round intervals before signed
calibration. They are not exact quantiles of a sum distribution, horizon-three
origin distributions, exchangeability guarantees, or evidence of MILP improvement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd

from fpl import config, experiment, features
from fpl.data.research import load_research_features
from fpl.model import models
from fpl.model.interval_calibration import apply_adjustment, conformal_adjustment, interval_metrics
from fpl.model.moe_tournament import protocol_cutoffs
from fpl.model.research_package import file_hash, source_hashes


def aggregate_round_predictions(frame: pd.DataFrame, quantiles: np.ndarray) -> pd.DataFrame:
    """Sum fixture targets/bounds/minutes before any round-level calibration.

    Multiple distinct DGW fixtures contribute once each; duplicate fixture identities
    fail closed. Summed fixture quantiles only approximate round quantiles.
    """
    required = {"player_id", "position", "GW_global", features.TARGET_COL, "minutes"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Interval frame missing columns: {sorted(missing)}")
    quantiles = np.asarray(quantiles, dtype=float)
    if quantiles.shape != (len(frame), 3) or not np.isfinite(quantiles).all():
        raise ValueError("Interval predictions require three finite quantiles per fixture")
    if not (quantiles[:, 0] <= quantiles[:, 1]).all() or not (quantiles[:, 1] <= quantiles[:, 2]).all():
        raise ValueError("Fixture interval bounds must bracket the median")
    if frame.empty or frame[["player_id", "position", "GW_global"]].isna().any().any():
        raise ValueError("Interval frame requires nonempty, complete player-round identity")
    for column in (features.TARGET_COL, "minutes"):
        if not np.isfinite(pd.to_numeric(frame[column], errors="coerce")).all():
            raise ValueError(f"Interval frame has nonfinite {column}")
    if (frame.minutes < 0).any():
        raise ValueError("Interval frame has negative minutes")
    keys = ["player_id", "position", "GW_global"]
    if "fixture" in frame:
        if frame.fixture.isna().any() or frame.duplicated(keys + ["fixture"]).any():
            raise ValueError("Interval frame has missing/duplicate fixture identity")
    elif frame.duplicated(keys).any():
        raise ValueError("Multiple interval fixtures require fixture identity")
    working = frame[keys + [features.TARGET_COL, "minutes"]].copy()
    working[["raw_lower", "median", "raw_upper"]] = quantiles
    return working.groupby(keys, as_index=False, sort=True).agg(
        actual_total_points=(features.TARGET_COL, "sum"), minutes=("minutes", "sum"),
        raw_lower=("raw_lower", "sum"), median=("median", "sum"), raw_upper=("raw_upper", "sum"),
        fixture_count=(features.TARGET_COL, "size"))


def _check_sources(expected: dict[str, str]) -> None:
    if source_hashes() != expected:
        raise ValueError("Pipeline source changed after interval registration")


def _parameters() -> dict[str, dict[str, Any]]:
    base = {key: value for key, value in models.LGB_PARAMS.items() if key not in ("objective", "metric")}
    base.update(objective="quantile", random_state=config.INTERVAL_SEED,
                n_jobs=config.RESEARCH_MODEL_THREADS, deterministic=True, force_col_wise=True)
    return {str(q): {**base, "alpha": q} for q in (.1, .5, .9)}


def _diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    report = {}
    for label, group in [("pooled", frame), *list(frame.groupby("position", sort=True))]:
        report[label] = {"rounds": len(group),
                         "raw": interval_metrics(group.actual_total_points, group.raw_lower, group.raw_upper,
                                                 alpha=config.INTERVAL_ALPHA),
                         "calibrated": interval_metrics(group.actual_total_points, group.calibrated_lower,
                                                        group.calibrated_upper, alpha=config.INTERVAL_ALPHA)}
    return report


def _evaluate_phase(
    frame: pd.DataFrame, columns: list[str], split: dict[str, int], parameters: dict[str, dict[str, Any]],
    source_digest: dict[str, str], directory: Path,
) -> dict[str, Any]:
    train = frame.loc[frame.GW_global <= split["train_max_gw"]]
    calibration = frame.loc[frame.GW_global.between(split["calibration_min_gw"], split["calibration_max_gw"])]
    evaluation = frame.loc[frame.GW_global.between(split["evaluation_min_gw"], split["evaluation_max_gw"])]
    calibration_rows, evaluation_rows, adjustments = [], [], {}
    fit_seconds = 0.
    prediction_seconds = 0.
    for position in config.ONFIELD_POSITIONS:
        fitting = train.loc[train.position == position]
        cal = calibration.loc[calibration.position == position]
        test = evaluation.loc[evaluation.position == position]
        if fitting.empty or cal.empty or test.empty:
            raise ValueError(f"Interval phase has missing train/calibration/evaluation rows for {position}")
        for data in (fitting, cal, test):
            if not np.isfinite(pd.to_numeric(data[features.TARGET_COL], errors="coerce")).all():
                raise ValueError("Interval phase targets must be finite")
        cal_predictions, test_predictions = [], []
        for q in (.1, .5, .9):
            _check_sources(source_digest)
            estimator = models.lgb.LGBMRegressor(**parameters[str(q)])
            started = time.perf_counter()
            estimator.fit(fitting[columns], fitting[features.TARGET_COL])
            fit_seconds += time.perf_counter() - started
            _check_sources(source_digest)
            started = time.perf_counter()
            cal_predictions.append(estimator.predict(cal[columns]))
            test_predictions.append(estimator.predict(test[columns]))
            prediction_seconds += time.perf_counter() - started
            _check_sources(source_digest)
        cal_rounds = aggregate_round_predictions(cal, np.sort(np.column_stack(cal_predictions), axis=1))
        test_rounds = aggregate_round_predictions(test, np.sort(np.column_stack(test_predictions), axis=1))
        adjustment = conformal_adjustment(cal_rounds.actual_total_points, cal_rounds.raw_lower,
                                          cal_rounds.raw_upper, alpha=config.INTERVAL_ALPHA)
        adjustments[position] = adjustment
        for rounds in (cal_rounds, test_rounds):
            lower, upper = apply_adjustment(rounds.raw_lower, rounds["median"], rounds.raw_upper, adjustment)
            rounds["calibrated_lower"], rounds["calibrated_upper"] = lower, upper
            rounds["adjustment"] = adjustment
        calibration_rows.append(cal_rounds)
        evaluation_rows.append(test_rounds)
    cal_export = pd.concat(calibration_rows, ignore_index=True)
    test_export = pd.concat(evaluation_rows, ignore_index=True)
    _check_sources(source_digest)
    directory.mkdir()
    cal_path, predictions_path = directory / "calibration.csv", directory / "predictions.csv"
    cal_export.to_csv(cal_path, index=False)
    test_export.to_csv(predictions_path, index=False)
    _check_sources(source_digest)
    return {"split": split, "adjustments": adjustments, "metrics": _diagnostics(test_export),
            "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
            "calibration_csv": str(cal_path), "calibration_sha256": file_hash(cal_path),
            "predictions_csv": str(predictions_path), "predictions_sha256": file_hash(predictions_path)}


def run(dataset_path: Path, output_dir: Path) -> dict[str, Any]:
    """Register and run two fixed chronological interval phases without production writes.

    Models are fitted on earlier fixture rows; signed adjustments use earlier
    calibration player rounds only. Features may update through evaluation origins,
    but reference model fitting/calibration never uses targets after selection.
    Preserve source bytes, registration, phase outputs, checkpoints and failures.
    """
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    if output_dir.exists():
        raise ValueError("Output directory already exists; choose a new interval run")
    output_dir.mkdir(parents=True)
    summary_path = output_dir / "summary.json"
    registration_path = output_dir / "registration.json"
    summary: dict[str, Any] = {"status": "running", "promotion_eligible": False, "phases": {}}
    started = time.perf_counter()
    try:
        cutoffs = protocol_cutoffs()
        weeks = config.INTERVAL_CALIBRATION_WEEKS
        splits = {}
        for phase, anchor, lower, upper in (
            ("selection", cutoffs["discovery_max_gw"], cutoffs["selection_min_gw"], cutoffs["selection_max_gw"]),
            ("reference", cutoffs["selection_max_gw"], cutoffs["backtest_min_gw"], cutoffs["backtest_max_gw"]),
        ):
            if weeks < 1 or anchor - weeks < 1 or not anchor < lower <= upper:
                raise ValueError("Invalid predeclared interval split")
            splits[phase] = {"train_max_gw": anchor - weeks, "calibration_min_gw": anchor - weeks + 1,
                             "calibration_max_gw": anchor, "evaluation_min_gw": lower, "evaluation_max_gw": upper}
        source_digest = source_hashes()
        archive = output_dir / "evaluated_sources"
        for name, digest in source_digest.items():
            original = config.ROOT / name
            destination = archive / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(original.read_bytes())
            if file_hash(destination) != digest:
                raise ValueError("Pipeline source changed while archiving interval sources")
        _check_sources(source_digest)
        prepared = {}
        preprocessing = {}
        columns = None
        dataset_digest = file_hash(dataset_path)
        for phase, split in splits.items():
            _check_sources(source_digest)
            phase_start = time.perf_counter()
            raw, frame, phase_columns = load_research_features(split["evaluation_max_gw"], dataset_path)
            preprocessing[phase] = time.perf_counter() - phase_start
            if raw.attrs.get("frozen_dataset_sha256") != dataset_digest or file_hash(dataset_path) != dataset_digest:
                raise ValueError("Interval dataset bytes differ from admitted frozen provenance")
            if frame.GW_global.max() > split["evaluation_max_gw"] or raw.GW_global.max() > split["evaluation_max_gw"]:
                raise ValueError("Interval feature loader failed its phase cap")
            if not phase_columns or len(set(phase_columns)) != len(phase_columns) or features.TARGET_COL in phase_columns:
                raise ValueError("Interval features must be unique, nonempty and exclude targets")
            if columns is not None and columns != phase_columns:
                raise ValueError("Interval phase ordered feature columns differ")
            columns = list(phase_columns)
            prepared[phase] = frame
            _check_sources(source_digest)
        parameters = _parameters()
        registration = {"status": "registered", "promotion_eligible": False, "seed": config.INTERVAL_SEED,
                        "alpha": config.INTERVAL_ALPHA, "quantiles": [.1, .5, .9], "parameters": parameters,
                        "splits": splits, "feature_columns": columns, "dataset": str(dataset_path.resolve()),
                        "dataset_sha256": dataset_digest, "code_sha256": source_digest,
                        "python": platform.python_version(), "versions": experiment._library_versions(),
                        "hypothesis": "Earlier signed round calibration may improve descriptive interval diagnostics.",
                        "limitations": ["Temporal dependence prevents an exchangeability coverage guarantee.",
                                        "Summed fixture quantiles approximate round intervals, not sum-distribution quantiles.",
                                        "Static one-step diagnostics, not horizon-three origin distribution exports.",
                                        "Repeatedly examined reference window; no MILP gain or production promotion claim."],
                        "preprocessing_seconds": preprocessing}
        _check_sources(source_digest)
        registration_path.write_text(json.dumps(registration, indent=2, sort_keys=True))
        experiment.log_result("interval_research_registration", registration,
                              {"status": "REGISTERED", "promotion_eligible": False})
        for phase, split in splits.items():
            _check_sources(source_digest)
            phase_started = time.perf_counter()
            result = _evaluate_phase(prepared[phase], columns, split, parameters, source_digest, output_dir / phase)
            result["preprocessing_seconds"] = preprocessing[phase]
            result["phase_seconds"] = time.perf_counter() - phase_started
            _check_sources(source_digest)
            summary["phases"][phase] = result
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        _check_sources(source_digest)
        if file_hash(dataset_path) != dataset_digest:
            raise ValueError("Interval dataset changed before completion")
        summary.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                       registration_sha256=file_hash(registration_path))
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        experiment.log_result("interval_research_complete",
                              {"registration": str(registration_path), "summary_sha256": file_hash(summary_path)},
                              {"status": "COMPLETE", "promotion_eligible": False, "phases": summary["phases"]})
        return summary
    except BaseException as exc:
        summary.update(status="incomplete", error=f"{type(exc).__name__}: {exc}",
                       elapsed_seconds=time.perf_counter() - started)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        experiment.log_result("interval_research_incomplete", {"registration": str(registration_path)},
                              {"status": "INCOMPLETE", "error": summary["error"], "promotion_eligible": False})
        raise


def main(argv: list[str] | None = None) -> None:
    """Run a fresh opt-in interval diagnostic with isolated research artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.FROZEN_RESEARCH_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
