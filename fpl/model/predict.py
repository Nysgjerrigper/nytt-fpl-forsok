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

import numpy as np
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


def _team_form_asof(hist_df):
    """Trailing team form through the last gameweek in `hist_df`, one row per team.

    Same aggregation semantics as features.add_opponent_strength_features (goals scored =
    sum over the team's player rows, conceded = max, clean sheet = conceded == 0; rolling
    6-GW mean), but WITHOUT the shift and taken at each team's latest played GW - i.e.
    "this team's form as of now". Used to attach opponent form to future fixtures the way
    a live run could: the upcoming opponent is known from the fixture list, and their form
    through the last played round is known; their form through rounds not yet played is not.
    """
    team_gw = (
        hist_df.groupby(["team", "GW_global"], sort=True)
        .agg(team_goals=("goals_scored", "sum"), team_conceded=("goals_conceded", "max"))
        .reset_index()
        .sort_values(["team", "GW_global"], kind="mergesort")
    )
    team_gw["team_cs"] = (team_gw["team_conceded"] == 0).astype(float)
    grouped = team_gw.groupby("team", sort=False)
    rolled = pd.DataFrame({
        "team": team_gw["team"],
        "opp_attack_roll6": grouped["team_goals"].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True),
        "opp_defense_roll6": grouped["team_conceded"].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True),
        "opp_cs_rate_roll6": grouped["team_cs"].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True),
    })
    return rolled.groupby("team", sort=False).tail(1).set_index("team")


def origin_based_predictions(df, raw_df, feature_cols, start_gw, end_gw, horizon,
                             retrain_every=4, weight_window=16,
                             weight_strategy=config.PRODUCTION_WEIGHT_STRATEGY):
    """Deploy-honest backtest export: for each origin GW t, predict ALL of t..t+horizon-1
    with player form frozen at t's deadline (information through GW t-1 only).

    Why this exists (audit finding B2): the standard walk-forward gives the MILP's
    lookahead terms forecasts for t+1/t+2 that were built with the OUTCOMES of GW t and
    t+1 - the transfer locked in at t is informed by data realized after t. A live run can
    never have that; it freezes form at "now" for the whole horizon (fpl.run_week). This
    export reproduces the live information set inside the backtest, so the realized-points
    number it produces is the honest expectation for deployment; the gap to the standard
    protocol measures the lookahead optimism once.

    Mechanics per origin t:
    - members are (re)trained on GW < t (every `retrain_every` origins, like the standard
      walk-forward); combination weights are fit once, strictly before `start_gw`;
    - a synthetic-snapshot frame is built from raw data truncated to GW < t
      (run_week.build_live_snapshot - the exact live code path), giving each active
      player's form features as of the deadline;
    - each target GW's REAL rows supply what a live run legitimately knows ahead: fixture
      identity (opponent, home/away, official FDR) - plus the actual points/value used
      for scoring, exactly as in the standard protocol so value handling is identical on
      both sides of the comparison;
    - the player-shifted feature columns of those target rows are OVERWRITTEN with the
      snapshot's frozen values, and opp_* is recomputed as the actual upcoming opponent's
      form through GW t-1 (_team_form_asof) - NOT the target row's own opp_* (form through
      target-1: future information), and NOT the snapshot's opp_* (the PREVIOUS fixture's
      opponent - the staleness bug run_week itself still has, see TODO 3.6).

    Players with no snapshot row (no appearance in the trailing season before t) are
    dropped from that origin's pool - live would have no prediction for them either.

    Output matches the standard CSV plus an `origin_gw` column; fpl.milp.optimize uses
    the origin-t rows for the solve at GW t.
    """
    from fpl.run_week import build_live_snapshot

    frozen_cols = features._player_shifted_columns(df)
    print(f"Fitting combination weights (strategy={weight_strategy}) "
          f"on holdout GW{start_gw - weight_window}-{start_gw - 1}...")
    weights_by_pos = fit_holdout_weights(df, feature_cols, first_holdout_gw=start_gw,
                                         window=weight_window, strategy=weight_strategy)

    played_gws = set(df["GW_global"].unique())
    rows = []
    models_cache = {}
    last_trained_gw = None
    for origin in range(start_gw, end_gw + 1):
        if origin not in played_gws:
            continue
        if last_trained_gw is None or origin - last_trained_gw >= retrain_every:
            train_df = df[df["GW_global"] < origin]
            if train_df.empty:
                continue
            models_cache.update(fit_position_ensembles(train_df, feature_cols, weights_by_pos))
            last_trained_gw = origin
            print(f"Retrained models for origin GW {origin}")

        hist_raw = raw_df[raw_df["GW_global"] < origin]
        snapshot = build_live_snapshot(hist_raw)
        snapshot = snapshot.set_index("player_id")
        opp_form = _team_form_asof(hist_raw)

        for gw in range(origin, min(origin + horizon, end_gw + 1)):
            if gw not in played_gws:
                continue
            target = df[df["GW_global"] == gw].copy()
            known = target["player_id"].isin(snapshot.index)
            target = target[known]
            if target.empty:
                continue
            # Freeze player form at the origin deadline; keep the target row's known-ahead
            # fixture columns and its actuals (scoring/value, same as standard protocol).
            target[frozen_cols] = snapshot.loc[target["player_id"], frozen_cols].to_numpy()
            for col in ("opp_attack_roll6", "opp_defense_roll6", "opp_cs_rate_roll6"):
                mapped = target["opponent_team"].map(opp_form[col])
                target[col] = mapped.to_numpy()

            target["predicted_total_points"] = 0.0
            for pos in POSITIONS:
                mask = target["position"] == pos
                if mask.any() and pos in models_cache:
                    target.loc[mask, "predicted_total_points"] = (
                        models_cache[pos].predict(target.loc[mask, feature_cols])
                    )
            target["origin_gw"] = origin
            rows.append(target)
        print(f"Origin GW {origin}: predicted {sum(len(r) for r in rows if r['origin_gw'].iat[0] == origin)} rows "
              f"over GW{origin}-{min(origin + horizon - 1, end_gw)}")

    result = pd.concat(rows, ignore_index=True)
    out_cols = ["player_id", "origin_gw", "GW_global", "name", "position", "team", "value",
                "predicted_total_points", "total_points"]
    return result[out_cols].rename(columns={"GW_global": "GW", "total_points": "actual_total_points"})


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
    parser.add_argument("--origin-based", action="store_true",
                         help="Deploy-honest export: for each origin GW t, predict all of "
                              "t..t+horizon-1 with player form frozen at t's deadline (the live "
                              "information set), adding an origin_gw column the MILP solves "
                              "per-origin. The standard walk-forward instead gives lookahead "
                              "forecasts future information live can never have (audit B2).")
    parser.add_argument("--horizon", type=int, default=3,
                         help="Lookahead length for --origin-based (match the MILP --horizon).")
    parser.add_argument("--output", type=str, default=str(config.PREDICTIONS_PATH))
    args = parser.parse_args()

    raw = pd.read_csv(config.MASTER_DATASET_PATH, low_memory=False)
    df = features.build_feature_frame(raw)
    feature_cols = features.feature_columns(df)

    if args.origin_based:
        if args.calibrate_level:
            sys.exit("--calibrate-level is not supported with --origin-based (documented "
                     "negative result; keep the comparison surface minimal).")
        preds = origin_based_predictions(df, raw, feature_cols, args.start_gw, args.end_gw,
                                         horizon=args.horizon, retrain_every=args.retrain_every,
                                         weight_strategy=args.weight_strategy)
    else:
        preds = walk_forward_predictions(df, feature_cols, args.start_gw, args.end_gw, args.retrain_every,
                                         weight_strategy=args.weight_strategy,
                                         calibrate_level=args.calibrate_level)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.output, index=False)
    print(f"Saved {len(preds)} rows to {args.output}")
