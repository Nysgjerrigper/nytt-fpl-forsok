"""
Train and compare several cheap, non-data-hungry model types per position
(see fpl.model.models.FACTORIES: LightGBM, Ridge, ElasticNet, Random Forest,
Extra Trees, kNN), then blend them into a per-position ensemble - replacing
the old per-position LSTM models in `legacy/R Forecast`.

Evaluates on the same GW77-107 window the old LSTM was validated on
(legacy/baseline_outputs/Validation_Predictions_Clean_v2.csv) so the two
approaches are directly comparable, and also reports a walk-forward
(expanding window, GW by GW) evaluation across the full dataset, which is a
more honest estimate of out-of-sample performance than one fixed split.

The ensemble blend weights are fit on the FIRST HALF of the test window and
evaluated on the SECOND HALF (a held-out split), so the reported ensemble
MAE isn't just overfit noise-chasing on the same data used to pick weights.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fpl import config, features
from fpl.model import models
from fpl.model.ensemble import PositionEnsemble, fit_blend_weights

POSITIONS = ["GK", "DEF", "MID", "FWD"]

# Kept for backwards compatibility with anything importing the old single-model API.
LGB_PARAMS = models.LGB_PARAMS


def load_features():
    raw = pd.read_csv(config.MASTER_DATASET_PATH, low_memory=False)
    return features.build_feature_frame(raw)


def train_position_model(train_df, feature_cols, position, model_name="lightgbm"):
    pos_df = train_df[train_df["position"] == position]
    X, y = pos_df[feature_cols], pos_df[features.TARGET_COL]
    return models.fit_model(model_name, X, y)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def evaluate_static_split(df, feature_cols, train_max_gw=76, test_min_gw=77, test_max_gw=107):
    """Same split window the old LSTM was validated on: train on GW<=76
    (2022-23 + 2023-24), test on GW77-107 (2024-25 GW1-31)."""
    train_df = df[df["GW_global"] <= train_max_gw]
    test_df = df[(df["GW_global"] >= test_min_gw) & (df["GW_global"] <= test_max_gw)].copy()
    blend_split_gw = test_min_gw + (test_max_gw - test_min_gw) // 2
    fit_mask = test_df["GW_global"] <= blend_split_gw
    eval_mask = ~fit_mask

    baseline_pred = test_df["total_points_roll3"].fillna(test_df["total_points_season_avg"]).fillna(0)

    print("\n--- Static split evaluation (GW<=76 train, GW77-107 test) ---")
    print(f"{'Position':<8}" + "".join(f"{name:<14}" for name in models.MODEL_NAMES) + f"{'baseline':<14}{'ensemble*':<14}")

    ensembles = {}
    per_model_preds_full = {pos: {} for pos in POSITIONS}
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

        row.append(mae(y_true_pos, baseline_pred.loc[pos_mask]))
        row.append(ensemble_mae)
        print(f"{row[0]:<8}" + "".join(f"{v:<14.4f}" for v in row[1:]))

        ensembles[pos] = PositionEnsemble(per_model_preds_full[pos], weights)
        print(f"    blend weights ({pos}): " + ", ".join(f"{n}={w:.2f}" for n, w in weights.items() if w > 0.01))

    print("(*ensemble MAE measured on the 2nd half of the test window only, using weights fit on the 1st half - "
          "a genuine holdout, not the same rows the weights were chosen from.)")

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
    gws = sorted(g for g in df["GW_global"].unique() if g >= start_gw)
    errors = []
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
            errors.append(mae(pos_test["total_points"], preds))
    print(f"\nWalk-forward MAE across GW{start_gw}+ (step={step}, model={model_name}): "
          f"{np.mean(errors):.4f} (n windows={len(errors)})")
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
    walk_forward_evaluate(df, feature_cols, start_gw=100, step=4, model_name="lightgbm")
    train_final_ensembles(df, feature_cols, {pos: ens.weights for pos, ens in ensembles.items()})
