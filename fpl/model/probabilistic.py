"""
Probabilistic forecasting: predict a DISTRIBUTION of points per player-gameweek,
not just a single expected value.

Why this matters for FPL specifically: two players with the same expected points
are not equivalent decisions. A steady 4-points-every-week midfielder and a
boom-or-bust forward who mostly blanks but occasionally hauls 15 can share an
expected value, yet the forward is the better captaincy pick (you want upside on
the 2x multiplier) and the worse must-not-blank pick. A single-number forecast
throws that distinction away; a distribution keeps it.

Approach: LightGBM quantile regression. For each position we fit one model per
target quantile (e.g. 0.1 / 0.5 / 0.9) using LightGBM's native `quantile`
objective, which minimizes the pinball loss for that quantile. The result per
player-gameweek is a predicted 10th percentile (near worst-case), median, and
90th percentile (upside) - i.e. a prediction interval and a point (the median)
in one. This deliberately reuses the SAME features as the point-forecast
ensemble (fpl.features) and stays completely separate from it: the production
squad optimizer still consumes the ensemble's single-number CSV; this module is
an additional, parallel view for uncertainty-aware analysis (e.g. captaincy),
not a replacement.

Evaluated with two honest, distribution-appropriate metrics rather than MAE/MASE:
- pinball (quantile) loss per quantile - the proper scoring rule these models are
  actually trained to minimize;
- interval coverage - the fraction of realized points that fall within the
  [p10, p90] band. A well-calibrated 10-90 interval should cover ~80% of
  outcomes; much less means the model is overconfident, much more means it is
  too timid.
"""
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fpl.model.models import LGB_PARAMS

DEFAULT_QUANTILES = [0.1, 0.5, 0.9]


def fit_quantile_models(X, y, quantiles=DEFAULT_QUANTILES):
    """Fit one LightGBM quantile regressor per quantile. Returns {quantile: model}.

    Shares LGB_PARAMS with the point-forecast models (fpl.model.models) but swaps
    the objective to `quantile` with the per-quantile `alpha`, so tree depth /
    learning rate / regularization stay consistent and only the loss changes.
    """
    params = {k: v for k, v in LGB_PARAMS.items() if k not in ("objective", "metric")}
    models = {}
    for q in quantiles:
        model = lgb.LGBMRegressor(objective="quantile", alpha=q, **params)
        model.fit(X, y)
        models[q] = model
    return models


def predict_quantiles(models, X):
    """Return {quantile: predictions}. Enforces monotonicity across quantiles
    (a higher quantile must not predict a lower value) by sorting per row - cheap
    insurance against the well-known quantile-crossing artifact of fitting each
    quantile independently."""
    qs = sorted(models)
    preds = np.column_stack([models[q].predict(X) for q in qs])
    preds = np.sort(preds, axis=1)  # fix any quantile crossing row-wise
    return {q: preds[:, i] for i, q in enumerate(qs)}


def pinball_loss(y_true, y_pred, quantile):
    """Pinball (quantile) loss - the proper scoring rule for a single quantile.
    Lower is better; it penalizes under- and over-prediction asymmetrically
    according to the quantile."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def interval_coverage(y_true, lower, upper):
    """Fraction of realized outcomes falling within [lower, upper]. For a 10-90
    interval this should sit near 0.80 if the model is well-calibrated."""
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))))


def evaluate_static_split(train_max_gw=152, test_min_gw=153, test_max_gw=183, quantiles=DEFAULT_QUANTILES):
    """Fit per-position quantile models on GW<=train_max_gw and evaluate on the
    same held-out window train.py uses, reporting pinball loss per quantile and
    10-90 interval coverage per position."""
    import pandas as pd
    from fpl import config, features
    from fpl.model.train import POSITIONS

    raw = pd.read_csv(config.MASTER_DATASET_PATH, low_memory=False)
    df = features.build_feature_frame(raw)
    feature_cols = features.feature_columns(df)
    train_df = df[df["GW_global"] <= train_max_gw]
    test_df = df[(df["GW_global"] >= test_min_gw) & (df["GW_global"] <= test_max_gw)]

    print(f"\n--- Probabilistic (quantile) evaluation, quantiles={quantiles} ---")
    print(f"{'Position':<8}" + "".join(f"{'pinball@' + str(q):<14}" for q in quantiles) + f"{'cov[p10,p90]':<14}")
    for pos in POSITIONS:
        tr = train_df[train_df["position"] == pos]
        te = test_df[test_df["position"] == pos]
        if tr.empty or te.empty:
            continue
        models = fit_quantile_models(tr[feature_cols], tr[features.TARGET_COL], quantiles)
        preds = predict_quantiles(models, te[feature_cols])
        y_true = te[features.TARGET_COL].to_numpy()
        row = [pinball_loss(y_true, preds[q], q) for q in quantiles]
        lo, hi = preds[min(quantiles)], preds[max(quantiles)]
        cov = interval_coverage(y_true, lo, hi)
        print(f"{pos:<8}" + "".join(f"{v:<14.4f}" for v in row) + f"{cov:<14.3f}")
    print(f"(coverage should sit near {max(quantiles) - min(quantiles):.2f} if the interval is well-calibrated)")


if __name__ == "__main__":
    evaluate_static_split()
