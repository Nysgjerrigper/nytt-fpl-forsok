"""
Generate a predictions CSV in the format the MILP optimizer expects
(fpl/milp/optimize.py): player_id, GW, name, position, team, value,
predicted_total_points, actual_total_points.

Two modes:
- Walk-forward backtest: for each GW in [start_gw, end_gw], train only on
  data strictly before it and predict that GW - this is what a real
  in-season run would have seen, so it's the right way to backtest the
  MILP against history (see fpl/model/train.py for the plain accuracy eval).
- Live: train on ALL available data and predict the next unplayed GW(s),
  using fixture/home-away info supplied by the caller (run_week.py fetches
  this from the official FPL API, since vaastav's historical data obviously
  has no rows yet for a gameweek that hasn't been played).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fpl import config, features
from fpl.model.train import POSITIONS, fit_holdout_weights, fit_level_calibration, fit_position_ensembles


def walk_forward_predictions(df, feature_cols, start_gw, end_gw, retrain_every=1, weight_window=16,
                             weight_strategy=config.PRODUCTION_WEIGHT_STRATEGY, calibrate_level=False):
    """Predict every GW in [start_gw, end_gw] using only data from earlier GWs.

    Combination weights are fit ONCE, on the `weight_window` gameweeks strictly before
    start_gw (members trained on data before that window) - so no weight ever sees
    a gameweek this function goes on to predict. The old scheme reused weights
    saved by train.py, which were fit INSIDE the GW153-183 test window and leaked
    into any backtest over it (RESEARCH_LOG.md 2026-07-04). Zero-weight members
    are skipped at every retrain - no point fitting a model the blend ignores.

    `weight_strategy` passes through to train.fit_holdout_weights: "nnls"/"top_k"/"ridge",
    or "single:<model>" (e.g. "single:catboost", which the GW169-226 head-to-head found
    beats every blend - see RESEARCH_LOG.md 2026-07-05).

    `calibrate_level=True` multiplies each position's predictions by a scalar fit on the same
    pre-window holdout (train.fit_level_calibration), correcting the median-flattened LEVEL of
    MAE-loss forecasts before they hit the MILP's absolute-scale transfer/chip logic. Rankings
    (and therefore MASE-style comparisons between calibrated and uncalibrated runs) are NOT
    comparable on MAE after scaling - the point of this flag is realized MILP points, not MAE.
    """
    rows = []
    models_cache = {}
    print(f"Fitting combination weights (strategy={weight_strategy}) "
          f"on holdout GW{start_gw - weight_window}-{start_gw - 1}...")
    weights_by_pos = fit_holdout_weights(df, feature_cols, first_holdout_gw=start_gw,
                                         window=weight_window, strategy=weight_strategy)
    for pos in POSITIONS:
        picked = ", ".join(f"{n}={w:.2f}" for n, w in weights_by_pos[pos].items() if w > 0.01)
        print(f"    weights ({pos}): {picked}")

    level_scalars = {pos: 1.0 for pos in POSITIONS}
    if calibrate_level:
        level_scalars = fit_level_calibration(df, feature_cols, first_holdout_gw=start_gw,
                                              weights_by_pos=weights_by_pos, window=weight_window)
        print("    level calibration scalars: "
              + ", ".join(f"{pos}={s:.3f}" for pos, s in level_scalars.items()))

    last_trained_gw = None
    for gw in range(start_gw, end_gw + 1):
        if gw not in df["GW_global"].unique():
            continue
        if last_trained_gw is None or gw - last_trained_gw >= retrain_every:
            train_df = df[df["GW_global"] < gw]
            if train_df.empty:
                continue
            models_cache.update(fit_position_ensembles(train_df, feature_cols, weights_by_pos))
            last_trained_gw = gw
            print(f"Retrained models for GW {gw}")

        test_df = df[df["GW_global"] == gw].copy()
        test_df["predicted_total_points"] = 0.0
        for pos in POSITIONS:
            mask = test_df["position"] == pos
            if mask.any() and pos in models_cache:
                test_df.loc[mask, "predicted_total_points"] = (
                    level_scalars[pos] * models_cache[pos].predict(test_df.loc[mask, feature_cols])
                )
        rows.append(test_df)

    result = pd.concat(rows, ignore_index=True)
    out_cols = ["player_id", "GW_global", "name", "position", "team", "value",
                "predicted_total_points", "total_points"]
    result = result[out_cols].rename(columns={"GW_global": "GW", "total_points": "actual_total_points"})
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start-gw", type=int, required=True)
    parser.add_argument("--end-gw", type=int, required=True)
    parser.add_argument("--retrain-every", type=int, default=4,
                         help="Retrain models every N gameweeks instead of every single GW (much faster).")
    parser.add_argument("--weight-strategy", type=str, default=config.PRODUCTION_WEIGHT_STRATEGY,
                         help="Combination strategy: nnls | top_k | ridge | single:<model>. "
                              "Defaults to the production strategy (config.PRODUCTION_WEIGHT_STRATEGY) "
                              "so backtests measure the same configuration live runs use.")
    parser.add_argument("--calibrate-level", action="store_true",
                         help="Rescale each position's predictions by sum(actual)/sum(predicted) "
                              "fit on the pre-window holdout - corrects MAE-loss median-flattening "
                              "before the MILP's absolute-scale transfer/chip decisions.")
    parser.add_argument("--output", type=str, default=str(config.PREDICTIONS_PATH))
    args = parser.parse_args()

    raw = pd.read_csv(config.MASTER_DATASET_PATH, low_memory=False)
    df = features.build_feature_frame(raw)
    feature_cols = features.feature_columns(df)

    preds = walk_forward_predictions(df, feature_cols, args.start_gw, args.end_gw, args.retrain_every,
                                     weight_strategy=args.weight_strategy,
                                     calibrate_level=args.calibrate_level)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.output, index=False)
    print(f"Saved {len(preds)} rows to {args.output}")
