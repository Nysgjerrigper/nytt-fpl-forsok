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
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fpl import config, features
from fpl.model import models as model_registry
from fpl.model.ensemble import PositionEnsemble
from fpl.model.train import POSITIONS, LGB_PARAMS, train_position_model


def _load_blend_weights(position):
    """Blend weights are fit once (fpl.model.train.evaluate_static_split) and
    reused here at every retrain point - refitting them at each walk-forward
    step would need its own held-out labelled slice per step, which isn't
    worth the extra complexity for what's a secondary refinement on top of
    already-good individual models."""
    weights_path = (config.MODELS_DIR / position).with_suffix(".weights.json")
    if weights_path.exists():
        return json.loads(weights_path.read_text())
    return None  # fall back to a plain LightGBM model if train.py hasn't been run yet


def walk_forward_predictions(df, feature_cols, start_gw, end_gw, retrain_every=1):
    """Predict every GW in [start_gw, end_gw] using only data from earlier GWs."""
    rows = []
    models_cache = {}
    last_trained_gw = None
    for gw in range(start_gw, end_gw + 1):
        if gw not in df["GW_global"].unique():
            continue
        if last_trained_gw is None or gw - last_trained_gw >= retrain_every:
            train_df = df[df["GW_global"] < gw]
            if train_df.empty:
                continue
            for pos in POSITIONS:
                pos_train = train_df[train_df["position"] == pos]
                if pos_train.empty:
                    continue
                weights = _load_blend_weights(pos)
                if weights is None:
                    models_cache[pos] = train_position_model(train_df, feature_cols, pos)
                else:
                    X, y = pos_train[feature_cols], pos_train[features.TARGET_COL]
                    members = {name: model_registry.fit_model(name, X, y) for name in weights}
                    models_cache[pos] = PositionEnsemble(members, weights)
            last_trained_gw = gw
            print(f"Retrained models for GW {gw}")

        test_df = df[df["GW_global"] == gw].copy()
        test_df["predicted_total_points"] = 0.0
        for pos in POSITIONS:
            mask = test_df["position"] == pos
            if mask.any() and pos in models_cache:
                test_df.loc[mask, "predicted_total_points"] = models_cache[pos].predict(test_df.loc[mask, feature_cols])
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
    parser.add_argument("--output", type=str, default=str(config.PREDICTIONS_PATH))
    args = parser.parse_args()

    raw = pd.read_csv(config.MASTER_DATASET_PATH, low_memory=False)
    df = features.build_feature_frame(raw)
    feature_cols = features.feature_columns(df)

    preds = walk_forward_predictions(df, feature_cols, args.start_gw, args.end_gw, args.retrain_every)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.output, index=False)
    print(f"Saved {len(preds)} rows to {args.output}")
