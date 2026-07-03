"""
Cheap per-player time-series baselines, computed directly from each player's own
points history rather than fit across players like fpl.model.models' regressors.

Croston's method (Croston, 1972) was the first addition here: it's built for
intermittent series - it separates "how much when it happens" (z, exponentially
smoothed non-zero points) from "how often it happens" (a, exponentially smoothed
gap length between non-zero gameweeks) and forecasts their ratio. Tested against
the production ensemble (see RESEARCH_LOG.md) and rejected - it underperforms
everything at every position, because it's slow to react to a player going cold
(it only updates on non-zero observations).

Added afterwards: naive drift, simple exponential smoothing (SES), Holt's linear
trend (double exponential smoothing), and a pooled AR(1) - standard econometric/
financial-forecasting baselines (Hyndman & Athanasopoulos' textbook set, plus the
classic single-lag autoregression), tested the same way and reported honestly in
RESEARCH_LOG.md rather than assumed to help just because they're "more principled."
"""
import numpy as np
import pandas as pd


def croston_forecast(values, alpha=0.1):
    """One-step-ahead Croston forecasts for a single player's ordered points history.

    result[i] is computed using only values[0..i-1] (never values[i] itself), so this
    is leakage-free by construction - no separate shift step needed downstream, unlike
    the rolling-window features in fpl.features which are shifted explicitly.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    forecasts = np.zeros(n)
    z_hat = None
    a_hat = None
    periods_since_obs = 0

    for i in range(n):
        if z_hat is not None:
            forecasts[i] = z_hat / a_hat if a_hat else 0.0

        periods_since_obs += 1
        if values[i] > 0:
            if z_hat is None:
                z_hat, a_hat = values[i], float(periods_since_obs)
            else:
                z_hat += alpha * (values[i] - z_hat)
                a_hat += alpha * (periods_since_obs - a_hat)
            periods_since_obs = 0

    return forecasts


def naive_drift_forecast(values):
    """One-step-ahead naive drift forecasts: extrapolate the straight line through
    the first and last observations seen SO FAR (Hyndman & Athanasopoulos), i.e.
    x_hat[i] = x[i-1] + (x[i-1] - x[0]) / (i-1). Leakage-free the same way as
    croston_forecast - result[i] only ever uses values[0..i-1]."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    forecasts = np.zeros(n)
    for i in range(1, n):
        if i == 1:
            forecasts[i] = values[0]
        else:
            slope = (values[i - 1] - values[0]) / (i - 1)
            forecasts[i] = values[i - 1] + slope
    return forecasts


def ses_forecast(values, alpha=0.3):
    """One-step-ahead simple exponential smoothing: level_t = alpha*x_t + (1-alpha)*level_{t-1},
    forecast[i] = level_{i-1} (the smoothed level BEFORE seeing x_i). Fixed alpha rather than
    per-player-optimized, matching the "cheap, non-data-hungry" philosophy elsewhere in this repo -
    per-player MLE alpha fitting would be another axis to add later if this baseline looked promising."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    forecasts = np.zeros(n)
    level = None
    for i in range(n):
        if level is not None:
            forecasts[i] = level
        level = values[i] if level is None else alpha * values[i] + (1 - alpha) * level
    return forecasts


def holt_forecast(values, alpha=0.3, beta=0.1):
    """One-step-ahead Holt's linear trend (double exponential smoothing): tracks a level
    and a trend so it can extrapolate a player's rising/declining run of form, not just
    its current level like SES. forecast[i] = level_{i-1} + trend_{i-1}."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    forecasts = np.zeros(n)
    level, trend = None, 0.0
    for i in range(n):
        if level is not None:
            forecasts[i] = level + trend
        if level is None:
            level = values[i]
        else:
            prev_level = level
            level = alpha * values[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
    return forecasts


def _add_per_player_column(df, forecast_fn, out_col, group_col, value_col, gw_col, **kwargs):
    df = df.sort_values([group_col, gw_col])
    df[out_col] = df.groupby(group_col)[value_col].transform(
        lambda s: forecast_fn(s.to_numpy(), **kwargs)
    )
    return df


def add_croston_column(df, group_col="player_id", value_col="total_points", gw_col="GW_global", alpha=0.1):
    """Add a `croston_pred` column: each row's leakage-free Croston forecast, computed
    per player over their own sorted history."""
    return _add_per_player_column(df, croston_forecast, "croston_pred", group_col, value_col, gw_col, alpha=alpha)


def add_naive_drift_column(df, group_col="player_id", value_col="total_points", gw_col="GW_global"):
    return _add_per_player_column(df, naive_drift_forecast, "naive_drift_pred", group_col, value_col, gw_col)


def add_ses_column(df, group_col="player_id", value_col="total_points", gw_col="GW_global", alpha=0.3):
    return _add_per_player_column(df, ses_forecast, "ses_pred", group_col, value_col, gw_col, alpha=alpha)


def add_holt_column(df, group_col="player_id", value_col="total_points", gw_col="GW_global", alpha=0.3, beta=0.1):
    return _add_per_player_column(df, holt_forecast, "holt_pred", group_col, value_col, gw_col, alpha=alpha, beta=beta)


def fit_ar1(train_df, lag_col="total_points_prev", target_col="total_points"):
    """Pooled AR(1) via OLS: total_points_t = c + phi * total_points_{t-1}, fit once
    across all rows of train_df (a single position's training rows, by convention -
    see how this is called in train.py). This is the classic single-lag econometric
    autoregression, distinct from the cross-sectional ML models in fpl.model.models
    which regress on total_points_prev alongside ~70 other engineered features - this
    isolates just the pure autoregressive signal as its own comparison point."""
    x = train_df[lag_col].fillna(0.0).to_numpy()
    y = train_df[target_col].to_numpy()
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    c, phi = coef
    return c, phi


def predict_ar1(test_df, c, phi, lag_col="total_points_prev"):
    return c + phi * test_df[lag_col].fillna(0.0).to_numpy()
