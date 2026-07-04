"""
Train and compare several cheap, non-data-hungry model types per position
(see fpl.model.models.FACTORIES: LightGBM, Ridge, ElasticNet, Random Forest,
Extra Trees, kNN), then blend them into a per-position ensemble - replacing
the old per-position LSTM models this project used originally (a Keras/R
sequence model, since removed - see git history for `legacy/R Forecast`).

Evaluates on the same 2024-25-season-GW1-31 window the old LSTM was validated on (GW_global
153-183 now that history has been extended back to 2020-21 - see config.DEFAULT_START_SEASON;
this was GW77-107 before the extension, since GW_global numbering is season-ORDINAL, not
calendar-fixed, so it shifts whenever the start season changes)
(legacy/baseline_outputs/Validation_Predictions_Clean_v2.csv, the one file
kept from that old approach - it's still read below) so the two approaches
are directly comparable, and also reports a walk-forward (expanding window,
GW by GW) evaluation across the full dataset, which is a more honest
estimate of out-of-sample performance than one fixed split.

The ensemble blend weights are fit on the FIRST HALF of the test window and
evaluated on the SECOND HALF (a held-out split), so the reported ensemble
MAE isn't just overfit noise-chasing on the same data used to pick weights.

Both MAE and MASE (fpl.model.metrics) are reported per model/baseline/ensemble
- see that module's docstring for why MASE matters for an intermittent series
like FPL points.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fpl import config, features
from fpl.model import models
from fpl.model.ensemble import PositionEnsemble, fit_blend_weights
from fpl.model.metrics import mae, mase, naive_lag1_scale
from fpl.model.baselines import (
    add_croston_column, add_naive_drift_column, add_ses_column, add_holt_column, add_theta_column,
    add_eb_shrinkage_column, fit_ar1, predict_ar1, fit_predict_arima_per_player,
)

POSITIONS = ["GK", "DEF", "MID", "FWD"]


def load_features():
    raw = pd.read_csv(config.MASTER_DATASET_PATH, low_memory=False)
    return features.build_feature_frame(raw)


def train_position_model(train_df, feature_cols, position, model_name="lightgbm"):
    pos_df = train_df[train_df["position"] == position]
    X, y = pos_df[feature_cols], pos_df[features.TARGET_COL]
    return models.fit_model(model_name, X, y)


def evaluate_static_split(df, feature_cols, train_max_gw=152, test_min_gw=153, test_max_gw=183):
    """Same split window the old LSTM was validated on: train on GW<=152
    (2020-21 through 2023-24), test on GW153-183 (2024-25 GW1-31)."""
    # These per-player forecasts need each player's FULL prior history, not just train_df's
    # window, so compute over the whole df before splitting - still leakage-free, since each
    # row's forecast only ever uses that player's strictly earlier gameweeks (see baselines.py).
    df = add_croston_column(df)
    df = add_naive_drift_column(df)
    df = add_ses_column(df)
    df = add_holt_column(df)
    df = add_theta_column(df)
    df = add_eb_shrinkage_column(df)

    train_df = df[df["GW_global"] <= train_max_gw]
    test_df = df[(df["GW_global"] >= test_min_gw) & (df["GW_global"] <= test_max_gw)].copy()
    blend_split_gw = test_min_gw + (test_max_gw - test_min_gw) // 2
    fit_mask = test_df["GW_global"] <= blend_split_gw
    eval_mask = ~fit_mask

    baseline_pred = test_df["total_points_roll3"].fillna(test_df["total_points_season_avg"]).fillna(0)

    # In-sample scale for MASE - fit on train_df only, so the score-intermittency benchmark
    # itself can't leak information from the test window it's used to judge.
    naive_scale = naive_lag1_scale(train_df)

    # Extra per-player time-series baselines (fpl.model.baselines) reported alongside the
    # models - econometric/financial-forecasting techniques (Croston, naive drift, SES,
    # Holt's linear trend), tested honestly rather than assumed to help. See RESEARCH_LOG.md
    # for the actual verdict on each.
    extra_baseline_cols = {
        "naive_drift": "naive_drift_pred",
        "ses": "ses_pred",
        "holt": "holt_pred",
        "theta": "theta_pred",
        "croston": "croston_pred",
        "eb_shrink": "eb_shrinkage_pred",
    }

    print("\n--- Static split evaluation (GW<=152 train, GW153-183 test) ---")
    header = (f"{'Position':<8}" + "".join(f"{name:<14}" for name in models.MODEL_NAMES)
              + f"{'baseline':<14}" + "".join(f"{name:<14}" for name in extra_baseline_cols)
              + f"{'ar1':<14}{'arima':<14}{'ensemble*':<14}")
    print("MAE:")
    print(header)

    ensembles = {}
    per_model_preds_full = {pos: {} for pos in POSITIONS}
    mase_rows = {}
    for pos in POSITIONS:
        pos_mask = test_df["position"] == pos
        row = [pos]
        preds_by_model = {}
        for name in models.MODEL_NAMES:
            model = train_position_model(train_df, feature_cols, pos, name)
            preds = model.predict(test_df.loc[pos_mask, feature_cols])
            preds_by_model[name] = preds
            per_model_preds_full[pos][name] = model
            row.append(mae(test_df.loc[pos_mask, "total_points"], preds))

        y_true_pos = test_df.loc[pos_mask, "total_points"].to_numpy()
        fit_idx = fit_mask.loc[pos_mask].to_numpy()
        eval_idx = eval_mask.loc[pos_mask].to_numpy()
        weights = fit_blend_weights({n: p[fit_idx] for n, p in preds_by_model.items()}, y_true_pos[fit_idx])
        blended_eval = sum(weights[n] * p[eval_idx] for n, p in preds_by_model.items())
        ensemble_mae = mae(y_true_pos[eval_idx], blended_eval)
        baseline_pos = baseline_pred.loc[pos_mask]

        extra_preds = {name: test_df.loc[pos_mask, col] for name, col in extra_baseline_cols.items()}

        # Pooled AR(1): fit once on this position's train_df, unlike the per-player
        # recursive baselines above - the classic single-lag econometric autoregression.
        ar1_c, ar1_phi = fit_ar1(train_df[train_df["position"] == pos])
        ar1_pred = predict_ar1(test_df.loc[pos_mask], ar1_c, ar1_phi)

        # Per-player ARIMA: one fit per player (not per row) on this position's train_df,
        # forecast forward across the whole test window - see baselines.py for why this
        # doesn't re-fit every gameweek like the recursive baselines above.
        arima_pred = fit_predict_arima_per_player(
            train_df[train_df["position"] == pos], test_df.loc[pos_mask]
        )

        row.append(mae(y_true_pos, baseline_pos))
        row.extend(mae(y_true_pos, p) for p in extra_preds.values())
        row.append(mae(y_true_pos, ar1_pred))
        row.append(mae(y_true_pos, arima_pred))
        row.append(ensemble_mae)
        print(f"{row[0]:<8}" + "".join(f"{v:<14.4f}" for v in row[1:]))

        mase_rows[pos] = (
            [mase(y_true_pos, preds_by_model[name], naive_scale) for name in models.MODEL_NAMES]
            + [mase(y_true_pos, baseline_pos, naive_scale)]
            + [mase(y_true_pos, p, naive_scale) for p in extra_preds.values()]
            + [mase(y_true_pos, ar1_pred, naive_scale), mase(y_true_pos, arima_pred, naive_scale),
               mase(y_true_pos[eval_idx], blended_eval, naive_scale)]
        )

        ensembles[pos] = PositionEnsemble(per_model_preds_full[pos], weights)
        print(f"    blend weights ({pos}): " + ", ".join(f"{n}={w:.2f}" for n, w in weights.items() if w > 0.01))

    print("(*ensemble MAE measured on the 2nd half of the test window only, using weights fit on the 1st half - "
          "a genuine holdout, not the same rows the weights were chosen from.)")

    print("\nMASE (< 1 beats the naive last-gameweek forecast, scale fit on train_df only):")
    print(header)
    for pos in POSITIONS:
        print(f"{pos:<8}" + "".join(f"{v:<14.4f}" for v in mase_rows[pos]))

    # "ols" (plain, unregularized multiple linear regression) is the designated INDEX - the
    # simple textbook benchmark everything else in this project is ultimately judged against,
    # the way a passive market index is the bar an active strategy has to clear. Everything
    # above is one table; this is the one number per position that actually matters.
    ols_idx = models.MODEL_NAMES.index("ols")
    print("\n--- Index check: does the ensemble beat plain OLS regression? ---")
    for pos in POSITIONS:
        ols_mase, ensemble_mase = mase_rows[pos][ols_idx], mase_rows[pos][-1]
        verdict = "beats index" if ensemble_mase < ols_mase else "does NOT beat index"
        print(f"{pos}: OLS (index) MASE={ols_mase:.4f}  ensemble MASE={ensemble_mase:.4f}  -> {verdict}")

    old_lstm_path = config.ROOT / "legacy" / "baseline_outputs" / "Validation_Predictions_Clean_v2.csv"
    if old_lstm_path.exists():
        old = pd.read_csv(old_lstm_path)
        old_mae = mae(old["actual_total_points"], old["predicted_total_points"])
        print(f"\nOld LSTM MAE on its own GW{old['GW'].min()}-{old['GW'].max()} validation set: {old_mae:.4f}")
        print("(Not a perfectly like-for-like comparison - different train data cutoff/label noise - "
              "but same GW window and same task.)")

    return ensembles, test_df


def walk_forward_evaluate(df, feature_cols, start_gw=40, step=1, model_name="lightgbm"):
    """Expanding-window walk-forward validation: for each GW from `start_gw`
    onward, train on everything strictly before it and predict that GW only.
    More gameweeks of true out-of-sample error than a single static split."""
    # Scale fit ONCE, globally (not re-fit per fold): this is a secondary diagnostic metric,
    # not something used to pick hyperparameters, so a single fixed denominator makes MASE
    # comparable fold-to-fold - refitting it per fold would make early folds (small training
    # windows) noisy and conflate "the scale changed" with "the model got worse".
    naive_scale = naive_lag1_scale(df[df["GW_global"] < start_gw])
    gws = sorted(g for g in df["GW_global"].unique() if g >= start_gw)
    errors = []
    mase_errors = []
    for gw in gws[::step]:
        train_df = df[df["GW_global"] < gw]
        test_df = df[df["GW_global"] == gw]
        if train_df.empty or test_df.empty:
            continue
        for pos in POSITIONS:
            pos_train = train_df[train_df["position"] == pos]
            pos_test = test_df[test_df["position"] == pos]
            if pos_train.empty or pos_test.empty:
                continue
            model = train_position_model(pos_train, feature_cols, pos, model_name)
            preds = model.predict(pos_test[feature_cols])
            y_true = pos_test["total_points"]
            errors.append(mae(y_true, preds))
            mase_errors.append(mase(y_true, preds, naive_scale))
    print(f"\nWalk-forward MAE across GW{start_gw}+ (step={step}, model={model_name}): "
          f"{np.mean(errors):.4f} (n windows={len(errors)})")
    print(f"Walk-forward MASE across GW{start_gw}+ (step={step}, model={model_name}): "
          f"{np.mean(mase_errors):.4f}")
    return errors


def train_final_ensembles(df, feature_cols, blend_weights):
    """Train every registered model type on ALL available data per position,
    and save as a PositionEnsemble using blend weights estimated during
    evaluate_static_split (there's no real holdout left once we train on
    everything, so we reuse the backtested weights for production)."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for pos in POSITIONS:
        members = {name: train_position_model(df, feature_cols, pos, name) for name in models.MODEL_NAMES}
        ensemble = PositionEnsemble(members, blend_weights[pos])
        ensemble.save(config.MODELS_DIR / pos)
        print(f"Saved final {pos} ensemble to {config.MODELS_DIR / pos}.*")


if __name__ == "__main__":
    df = load_features()
    feature_cols = features.feature_columns(df)
    print(f"Loaded {len(df)} rows, {len(feature_cols)} features.")

    ensembles, _ = evaluate_static_split(df, feature_cols)
    walk_forward_evaluate(df, feature_cols, start_gw=176, step=4, model_name="lightgbm")
    train_final_ensembles(df, feature_cols, {pos: ens.weights for pos, ens in ensembles.items()})
