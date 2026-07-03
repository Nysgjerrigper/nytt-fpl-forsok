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


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


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
