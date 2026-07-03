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

Added again afterwards: the Theta method (a strong, simple M3/M4-competition
baseline - averages a linear trend line with an exponentially-smoothed
curvature-doubled line) and per-player ARIMA (via statsmodels, the one new
dependency here - see requirements.txt). Both were flagged by the Venter paper
as relatively strong individual forecasters; both tested the same honest way.
"""
import warnings

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


def theta_forecast(values, theta=2.0, alpha=0.2):
    """One-step-ahead Theta method forecasts (Assimakopoulos & Nikolopoulos, 2000):
    average a long-term linear trend line (theta=0 component) with a curvature-doubled
    line (theta=2 component, smoothed via SES) - the trend line captures the slow-moving
    signal, the doubled-curvature line reacts faster to recent swings, and averaging the
    two is what made this method a standout in the M3/M4 forecasting competitions despite
    its simplicity. Refit from scratch at each step (the trend line changes as more history
    arrives), unlike the single-pass recursions above - still leakage-free, since every
    fit only uses values[0..i-1]."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    forecasts = np.zeros(n)
    for i in range(n):
        hist = values[:i]
        m = len(hist)
        if m == 0:
            continue
        if m == 1:
            forecasts[i] = hist[0]
            continue
        t = np.arange(m, dtype=float)
        b, a = np.polyfit(t, hist, 1)
        trend = a + b * t
        theta_line = theta * hist + (1 - theta) * trend

        level = theta_line[0]
        for k in range(1, m):
            level = alpha * theta_line[k] + (1 - alpha) * level

        theta0_forecast = a + b * m
        forecasts[i] = 0.5 * theta0_forecast + 0.5 * level
    return forecasts


def fit_predict_arima_per_player(train_df, test_df, group_col="player_id", value_col="total_points",
                                  gw_col="GW_global", order=(1, 0, 1), min_obs=8):
    """Fit ONE ARIMA(order) model per player on their train_df history, then forecast
    forward however many steps that player has rows in test_df. Fit once, not re-fit at
    every gameweek like the recursive baselines above - statsmodels' MLE-based fitting
    is too slow to redo per-row per-player at this scale (hundreds of players). Players
    with fewer than `min_obs` training observations (not enough history to identify AR/MA
    parameters) fall back to their training mean."""
    from statsmodels.tsa.arima.model import ARIMA

    train_sorted = train_df.sort_values([group_col, gw_col])
    test_sorted = test_df.sort_values([group_col, gw_col])
    preds = pd.Series(index=test_sorted.index, dtype=float)

    for pid, test_rows in test_sorted.groupby(group_col):
        hist = train_sorted.loc[train_sorted[group_col] == pid, value_col].to_numpy()
        n_steps = len(test_rows)
        fallback = float(hist.mean()) if len(hist) else 0.0
        if len(hist) < min_obs:
            forecast = np.full(n_steps, fallback)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fitted = ARIMA(hist, order=order, trend="c").fit()
                    forecast = np.asarray(fitted.forecast(steps=n_steps))
                except Exception:
                    forecast = np.full(n_steps, fallback)
        preds.loc[test_rows.index] = forecast

    return preds.reindex(test_df.index)


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


def add_theta_column(df, group_col="player_id", value_col="total_points", gw_col="GW_global", theta=2.0, alpha=0.2):
    return _add_per_player_column(df, theta_forecast, "theta_pred", group_col, value_col, gw_col, theta=theta, alpha=alpha)


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
