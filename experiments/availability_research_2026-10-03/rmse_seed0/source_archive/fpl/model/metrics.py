"""
Error metrics for point-forecast evaluation.

MAE alone is hard to read as "good" or "bad" here: FPL points are an intermittent series
- most players score 0 in most gameweeks - so an MAE of 1.2 looks identical whether the
target typically scores 8 or typically scores 0. MASE (Hyndman & Koehler, 2006) fixes this
by expressing error relative to a naive benchmark ("predict the same points a player
scored last gameweek"), scaled using only in-sample (training) data so the scale itself
can't leak test-period information. MASE < 1 means the model beats that naive floor;
MASE > 1 means it doesn't, regardless of the raw point-scale.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    """Root mean squared error - the mean-optimal complement to MAE.

    MAE is minimized by predicting the conditional MEDIAN; RMSE is minimized by the
    conditional MEAN. That distinction is load-bearing here: the MILP sums predicted
    points to build a squad, so it needs unbiased conditional means, not medians. Because
    FPL points are heavily right-skewed (mostly 0-2, occasional hauls), the median sits
    well below the mean, and a model tuned purely on MAE/MASE will systematically flatten
    those hauls. Watching RMSE alongside MAE surfaces exactly that median-vs-mean gap:
    RMSE penalizes the large misses on high scorers that MAE shrugs off.
    """
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(err ** 2)))


def bias(y_true, y_pred):
    """Signed mean error, mean(y_pred - y_true); positive = systematic over-prediction.

    MAE and MASE are unsigned, so a model that is a median predictor can look "accurate"
    while consistently sitting below the mean on a skewed target - a miscalibration the
    downstream MILP inherits as a squad-level points shortfall. Bias exposes that
    directional error: a median-flattening model on right-skewed FPL points shows up here
    as a persistent NEGATIVE bias (it under-predicts the mean), which unsigned errors hide.
    """
    return float(np.mean(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))


def total_calibration(y_true, y_pred):
    """Aggregate level ratio, sum(y_pred) / sum(y_true); ~1.0 means the totals match.

    The MILP optimizes over SUMS of predicted points, so what matters for squad value is
    whether the model gets the aggregate level right, not just per-row error. A model that
    predicts the conditional median will drive this ratio below 1.0 on right-skewed points
    (the summed medians undershoot the summed means), quantifying the total-points
    shortfall that a mean-optimal model would avoid. Returns NaN if the true total is 0,
    since the ratio is undefined there.
    """
    denom = float(np.sum(np.asarray(y_true, dtype=float)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(np.asarray(y_pred, dtype=float))) / denom


def naive_lag1_scale(df, group_col="player_id", value_col="total_points", gw_col="GW_global"):
    """In-sample MASE denominator: mean absolute first difference of the target.

    Computed as ONE global scale (pooled across all players), not per-player - a
    per-player scale would let low-scoring players' tiny naive-diff scale blow up their
    individual MASE, which isn't useful when reporting one MASE per position. Sorting by
    (group_col, gw_col) before differencing ensures a player is only ever compared against
    their own previous gameweek, never against another player's unrelated series.
    """
    sorted_df = df.sort_values([group_col, gw_col])
    diffs = sorted_df.groupby(group_col)[value_col].diff().abs()
    return float(diffs.mean())


def mase(y_true, y_pred, scale):
    if not scale or np.isnan(scale):
        return float("nan")
    return mae(y_true, y_pred) / scale


def spearman_by_group(y_true, y_pred, groups):
    """Mean within-group Spearman rank correlation between prediction and outcome.

    Absolute error (MAE/MASE) can be small while the model still orders players wrong -
    and ORDERING is what actually drives decisions: transfers and captaincy only ever
    compare players against each other, they never consume the raw point value. A
    median-flattening model that compresses every prediction toward a similar small number
    can post a good MAE yet rank players almost arbitrarily; this metric catches that.

    Grouping (e.g. by position-gameweek) keeps the ranking question local and comparable -
    "did we order THIS week's midfielders correctly" - rather than pooling across weeks
    where point levels differ for reasons unrelated to skill. Groups with < 3 rows or with
    zero variance in either vector are skipped: rank correlation is undefined or degenerate
    there (a constant vector gives NaN). Returns NaN if no group qualifies.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)
    rhos = []
    for g in pd.unique(groups):
        mask = groups == g
        gt = yt[mask]
        gp = yp[mask]
        if len(gt) < 3:
            continue
        if np.ptp(gt) == 0 or np.ptp(gp) == 0:
            continue
        rho, _ = spearmanr(gt, gp)
        if not np.isnan(rho):
            rhos.append(rho)
    if not rhos:
        return float("nan")
    return float(np.mean(rhos))


def top1_capture(df, group_col, y_true_col, y_pred_col):
    """Captaincy-quality score: actual points of our top pick vs. the true group max.

    Per group (e.g. a position-gameweek), take the player we RANKED first by prediction
    and record their ACTUAL points, then sum those across groups and divide by the sum of
    each group's true maximum. This is the decision-aligned counterpart to error metrics:
    captaincy only cares about identifying the single highest scorer, i.e. the extreme
    UPSIDE TAIL, which is precisely where a mean-fitted (let alone median-fitted) model is
    weakest. MAE/MASE reward being close on the many low scorers; this rewards being right
    about the one that matters. Score lies in ~[0, 1]: 1.0 means our #1 pick was always the
    actual top scorer. Ties in prediction resolve to the first row via idxmax. Returns NaN
    if the summed true max is 0 (no points scored anywhere, so the ratio is undefined).
    """
    captured = 0.0
    best_possible = 0.0
    for _, gdf in df.groupby(group_col):
        pick_idx = gdf[y_pred_col].idxmax()
        captured += float(gdf.loc[pick_idx, y_true_col])
        best_possible += float(gdf[y_true_col].max())
    if best_possible == 0.0:
        return float("nan")
    return captured / best_possible
