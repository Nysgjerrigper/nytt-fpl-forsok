"""Exploratory, causal model and ensemble screening; never a promotion result.

Models use explicit registry defaults plus optional overrides, not ambient tuning
files. Each member is fitted once per refit date and reused across combiners.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl import config, features
from fpl.data.provenance import load_frozen_research_dataset
from fpl.model import metrics, models
from fpl.model.ensemble import fit_weights
from fpl.model.moe_tournament import protocol_cutoffs
from fpl.model.oof_panel import validate_oof_panel
from fpl.model.walk_forward import walk_forward_steps


def generate_oof(
    frame: pd.DataFrame, feature_cols: list[str], parameters: dict[str, dict[str, Any]],
    *, positions: tuple[str, ...], seed: int, retrain_every: int,
) -> pd.DataFrame:
    """Forecast the registered selection weeks with strictly earlier training rows.

    ``parameters`` must contain complete constructor parameters, resolved before
    fitting. Registry builders deliberately bypass the automatic tuned-file loader.
    Fit and predict on fixture rows, then sum forecasts and actuals per player round.
    The MASE denominator uses summed training rounds strictly before each refit.
    Form is updated each week; this is a one-step screen, not origin-horizon evidence.
    """
    cutoffs = protocol_cutoffs()
    start, end = cutoffs["selection_min_gw"], cutoffs["selection_max_gw"]
    if retrain_every < 1 or not parameters or not positions:
        raise ValueError("positive retrain_every, models and positions are required")
    if set(parameters) - set(models.EXPERT_SPECS):
        raise ValueError("screen models must belong to EXPERT_SPECS")
    required = {"player_id", "position", "GW_global", "total_points", *feature_cols}
    if required - set(frame):
        raise ValueError(f"missing screening columns: {sorted(required - set(frame))}")
    if "total_points" in feature_cols:
        raise ValueError("the target cannot be a feature")
    identity = ["position", "player_id", "GW_global"]
    if frame[identity + ["total_points"]].isna().any().any():
        raise ValueError("screen fixture identity and targets cannot be missing")
    if not pd.api.types.is_numeric_dtype(frame["total_points"]) or not np.isfinite(frame["total_points"]).all():
        raise ValueError("screen fixture targets must contain finite numeric values")
    multiple = frame.duplicated(identity, keep=False)
    if multiple.any() and ("fixture" not in frame or frame.loc[multiple, "fixture"].isna().any()):
        raise ValueError("multiple fixture rows require nonmissing fixture identities")
    if "fixture" in frame and frame.duplicated(identity + ["fixture"]).any():
        raise ValueError("screen contains duplicate fixture identities")
    for position in positions:
        pos = frame[frame.position == position]
        if set(pos.loc[pos.GW_global.between(start, end), "GW_global"]) != set(range(start, end + 1)):
            raise ValueError(f"{position} lacks complete selection-week coverage")
        if pos[pos.GW_global < start].empty:
            raise ValueError(f"{position} has no pre-selection training history")
    records = []
    for position in positions:
        pos = frame[frame.position == position]

        def fit_members(train: pd.DataFrame) -> tuple[dict, int, float]:
            rounds = train.groupby(["player_id", "position", "GW_global"], as_index=False)["total_points"].sum()
            scale = metrics.naive_lag1_scale(rounds)
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("training-only player-round MASE scale must be finite and positive")
            fitted = {}
            for name, params in parameters.items():
                model = models.build_registered_model(name, params=params, seed=seed, position=position)
                model.fit(train[feature_cols], train["total_points"])
                fitted[name] = model
            return fitted, int(train.GW_global.max()), scale

        for gw, (fitted, train_max, scale), test in walk_forward_steps(
            pos, start, end, retrain_every, fit_members, label=f"screen members ({position})"
        ):
            for name, model in fitted.items():
                prediction = np.asarray(model.predict(test[feature_cols]), dtype=float)
                if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
                    raise ValueError(f"{position}/{name}/GW{gw} returned invalid predictions")
                part = test[["player_id", "position", "GW_global", "total_points"]].copy()
                part = part.rename(columns={"total_points": "actual_total_points"})
                part["prediction"] = prediction
                part = part.groupby(["player_id", "position", "GW_global"], as_index=False).agg(
                    actual_total_points=("actual_total_points", "sum"),
                    prediction=("prediction", "sum"),
                    fixture_count=("prediction", "size"),
                )
                part["expert"] = name
                part["train_max_gw"], part["mase_scale"] = train_max, scale
                part["seed"], part["stage"] = seed, "exploratory_screen"
                part["params_hash"] = _json_hash(parameters[name])
                records.append(part)
    oof = pd.concat(records, ignore_index=True)
    validate_oof_panel(oof, positions=positions, experts=tuple(parameters))
    return oof


def compare_ensembles(
    oof: pd.DataFrame, *, positions: tuple[str, ...], warmup_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score all members and combiners on identical rows after a shared warmup.

    Learned weights at GW t use only member OOF forecasts and outcomes from GWs
    strictly before t. Neither member nor combiner sees its scoring row's label.
    Return diagnostics, row predictions, fitted weights, and residual correlations.
    """
    validate_oof_panel(oof, positions=positions)
    cutoffs = protocol_cutoffs()
    expected_weeks = set(range(cutoffs["selection_min_gw"], cutoffs["selection_max_gw"] + 1))
    if not (oof.train_max_gw < oof.GW_global).all():
        raise ValueError("screen OOF training cutoffs must precede prediction weeks")
    for position in positions:
        if set(oof.loc[oof.position == position, "GW_global"]) != expected_weeks:
            raise ValueError("screen OOF must cover exactly the registered selection weeks")
    if warmup_weeks < 1:
        raise ValueError("warmup_weeks must be positive")
    forecasts, weight_rows, correlations = [], [], []
    for position in positions:
        pos = oof[oof.position == position]
        names = sorted(pos.expert.unique())
        keys = ["player_id", "GW_global"]
        wide = pos.pivot(index=keys, columns="expert", values="prediction").sort_index()
        labels = pos.drop_duplicates(keys).set_index(keys).reindex(wide.index)
        weeks = sorted(wide.index.get_level_values("GW_global").unique())
        if len(weeks) <= warmup_weeks:
            raise ValueError("warmup leaves no scoring weeks")
        week_values = wide.index.get_level_values("GW_global")
        for gw in weeks[warmup_weeks:]:
            earlier, current = week_values < gw, week_values == gw
            history = {name: wide.loc[earlier, name].to_numpy() for name in names}
            y = labels.loc[earlier, "actual_total_points"].to_numpy()
            strategies = {f"single:{name}": {name: 1.0} for name in names}
            strategies["equal"] = {name: 1.0 / len(names) for name in names}
            for method in ("nnls", "ridge", "top_k"):
                strategies[method] = fit_weights(history, y, method=method)
            for strategy, weights in strategies.items():
                label_columns = ["actual_total_points", "mase_scale"]
                if "fixture_count" in labels:
                    label_columns.append("fixture_count")
                part = labels.loc[current, label_columns].reset_index()
                part["position"], part["strategy"] = position, strategy
                part["prediction"] = sum(weight * wide.loc[current, name].to_numpy() for name, weight in weights.items())
                part["weight_train_max_gw"] = max(week_values[earlier])
                forecasts.append(part)
                for name, weight in weights.items():
                    weight_rows.append({"position": position, "GW_global": gw, "strategy": strategy,
                                        "expert": name, "weight": weight,
                                        "weight_train_max_gw": max(week_values[earlier])})
        scored = week_values >= weeks[warmup_weeks]
        residuals = wide.loc[scored].subtract(labels.loc[scored, "actual_total_points"], axis=0)
        corr = residuals.corr()
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                correlations.append({"position": position, "left": left, "right": right,
                                     "residual_correlation": corr.loc[left, right]})
    predictions = pd.concat(forecasts, ignore_index=True)
    reports = []
    for (position, strategy), group in predictions.groupby(["position", "strategy"], sort=True):
        y, pred = group.actual_total_points, group.prediction
        reports.append({"position": position, "strategy": strategy, "rows": len(group),
                        "weeks": group.GW_global.nunique(), "mae": metrics.mae(y, pred),
                        "mase": float(np.mean(np.abs(y - pred) / group.mase_scale)),
                        "rmse": metrics.rmse(y, pred), "bias": metrics.bias(y, pred),
                        "total_calibration": metrics.total_calibration(y, pred),
                        "spearman": metrics.spearman_by_group(y, pred, group.GW_global),
                        "top1_capture": metrics.top1_capture(group, "GW_global", "actual_total_points", "prediction")})
    return (pd.DataFrame(reports), predictions, pd.DataFrame(weight_rows),
            pd.DataFrame(correlations, columns=["position", "left", "right", "residual_correlation"]))


def _json_hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    """Run an explicitly requested screen and persist its inputs and diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="Comma-separated EXPERT_SPECS names")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; existing runs are never overwritten")
    parser.add_argument("--dataset", type=Path, default=config.FROZEN_RESEARCH_DATASET_PATH)
    parser.add_argument("--positions", default=",".join(config.ONFIELD_POSITIONS))
    parser.add_argument("--params-json", type=Path, help="Optional JSON object mapping model names to parameter overrides")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrain-every", type=int, default=config.SCREEN_RETRAIN_EVERY)
    parser.add_argument("--warmup-weeks", type=int, default=config.SCREEN_WARMUP_WEEKS)
    args = parser.parse_args(argv)
    names, positions = tuple(args.models.split(",")), tuple(args.positions.split(","))
    if len(set(names)) != len(names) or set(names) - set(models.EXPERT_SPECS):
        parser.error(f"models must be unique members of {sorted(models.EXPERT_SPECS)}")
    if len(set(positions)) != len(positions) or set(positions) - set(config.ONFIELD_POSITIONS):
        parser.error("positions must be unique members of GK,DEF,MID,FWD")
    cutoffs = protocol_cutoffs()
    if args.retrain_every < 1 or not 1 <= args.warmup_weeks < cutoffs["selection_max_gw"] - cutoffs["selection_min_gw"] + 1:
        parser.error("retrain interval must be positive and warmup must leave scoring weeks")
    overrides = json.loads(args.params_json.read_text()) if args.params_json else {}
    if not isinstance(overrides, dict) or set(overrides) - set(names) or any(not isinstance(v, dict) for v in overrides.values()):
        parser.error("parameter overrides must map selected model names to objects")
    parameters = {name: {**models.EXPERT_SPECS[name].default_params(args.seed), **overrides.get(name, {})} for name in names}
    # Record the effective seed, including overrides, exactly as the builder uses it.
    for params in parameters.values():
        for key in ("random_state", "random_seed"):
            if key in params:
                params[key] = args.seed
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new run directory")
    args.output_dir.mkdir(parents=True)
    manifest = {"status": "running", "promotion_eligible": False,
                "purpose": "Exploratory screening only; no realized-points or promotion claim",
                "semantics": {"training": "per-fixture feature rows and targets",
                              "scoring": "summed player-position-gameweek forecasts and actuals",
                              "mase_scale": "training-only summed player-round lag-one scale"},
                "cutoffs": cutoffs, "parameters": parameters, "positions": positions, "seed": args.seed,
                "retrain_every": args.retrain_every, "warmup_weeks": args.warmup_weeks,
                "python": platform.python_version(), "dataset": str(args.dataset.resolve()),
                "dataset_sha256": None,
                "code_sha256": {str(path.relative_to(config.ROOT)): _file_hash(path) for path in sorted((config.ROOT / "fpl").rglob("*.py"))},
                "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost")}}
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    try:
        raw = load_frozen_research_dataset(args.dataset)
        manifest["dataset_sha256"] = raw.attrs["frozen_dataset_sha256"]
        raw = raw[raw.GW_global <= cutoffs["selection_max_gw"]].copy()
        frame = features.build_feature_frame(raw)
        cols = features.feature_columns(frame)
        manifest["feature_columns"] = cols
        oof = generate_oof(frame, cols, parameters, positions=positions, seed=args.seed, retrain_every=args.retrain_every)
        oof.to_csv(args.output_dir / "member_oof.csv", index=False)
        report, predictions, weights, correlations = compare_ensembles(oof, positions=positions, warmup_weeks=args.warmup_weeks)
        for name, table in (("diagnostics", report), ("predictions", predictions), ("weights", weights), ("residual_correlations", correlations)):
            table.to_csv(args.output_dir / f"{name}.csv", index=False)
        manifest["outputs"] = {path.name: _file_hash(path) for path in sorted(args.output_dir.glob("*.csv"))}
        manifest["status"] = "complete"
        print(report.to_string(index=False))
        print(f"Exploratory screen saved to {args.output_dir}; promotion NOT ASSESSED.")
    except Exception as exc:
        manifest["status"], manifest["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
