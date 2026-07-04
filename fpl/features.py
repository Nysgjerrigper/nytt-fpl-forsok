"""
Feature engineering shared by training, backtesting and live weekly prediction.

Replaces the LSTM's learned embeddings with plain rolling-window stats per
player (form over the last few gameweeks) — this is what makes a
gradient-boosted-tree model competitive without needing thousands of rows
per player the way a sequence model would.
"""
import numpy as np
import pandas as pd

ROLLING_WINDOWS = [3, 5]

# Per-gameweek stats we track form over. Kept to well-populated, position-
# relevant columns; sparse newer stats (e.g. defensive_contribution, only
# present since 2025-26) are included but will just be 0/NaN pre-2025-26.
FORM_STATS = [
    "total_points",
    "minutes",
    "bps",
    "ict_index",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goals_conceded",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "threat",
    "creativity",
    "influence",
    "value",
    "selected",
    "defensive_contribution",
]

CATEGORICAL_FEATURES = ["was_home"]

# Fixture-difficulty columns merged in by fpl.data.fetch (official FPL FDR, 1-5).
# These are NOT shifted: the fixture list and its difficulty are published before a
# gameweek is played, so knowing GW t's opponent (and the next two) is a legitimate
# input at prediction time, not leakage.
FIXTURE_FEATURES = ["fixture_difficulty", "fixture_difficulty_next3"]

# Minutes-projection ("nailedness") features - see add_minutes_features. A player who
# won't start scores ~0, so projecting minutes from recent starts is one of the single
# most predictive signals in modern FPL models.
MINUTES_FEATURES = ["start_rate_roll5", "mins60_rate_roll5"]

TARGET_COL = "total_points"


def _ensure_sorted(df):
    return df.sort_values(["player_id", "GW_global"]).reset_index(drop=True)


def add_rolling_features(df):
    """Add, for each stat in FORM_STATS, rolling mean over the last N gameweeks
    and a season-to-date expanding mean — all shifted by one GW so a row never
    sees its own outcome (no leakage)."""
    df = _ensure_sorted(df)
    grouped = df.groupby("player_id", sort=False)

    new_cols = {}
    for col in FORM_STATS:
        if col not in df.columns:
            continue
        shifted = grouped[col].shift(1)
        new_cols[f"{col}_prev"] = shifted
        for window in ROLLING_WINDOWS:
            new_cols[f"{col}_roll{window}"] = (
                shifted.groupby(df["player_id"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
            )
        new_cols[f"{col}_season_avg"] = (
            shifted.groupby(df["player_id"]).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
        )

    new_cols["games_played_so_far"] = grouped.cumcount()
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def add_minutes_features(df):
    """Add rolling "nailedness" features projecting whether a player will get minutes.

    - start_rate_roll5: rolling fraction of the last 5 games the player started. Uses
      the `starts` column where present (2022-23 onward) and falls back to a minutes>=60
      proxy for older seasons where `starts` doesn't exist.
    - mins60_rate_roll5: rolling fraction of the last 5 games with a "full" appearance
      (>=60 minutes, the FPL clean-sheet threshold). Available for every season since
      `minutes` always exists, so it's the more robust of the two.

    Both are shifted by one gameweek (computed only from strictly-earlier games), exactly
    like the rolling form features, so a row never sees its own outcome.
    """
    df = _ensure_sorted(df)

    started = df["starts"] if "starts" in df.columns else pd.Series(np.nan, index=df.index)
    started = pd.to_numeric(started, errors="coerce").fillna((df["minutes"] >= 60).astype(float))
    flags = pd.DataFrame({
        "start_rate_roll5": started,
        "mins60_rate_roll5": (df["minutes"] >= 60).astype(float),
    }, index=df.index)

    new_cols = {}
    for out_col in flags.columns:
        shifted = flags[out_col].groupby(df["player_id"]).shift(1)
        new_cols[out_col] = (
            shifted.groupby(df["player_id"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        )
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def build_feature_frame(raw_df):
    """Full feature pipeline: clean dtypes, add rolling form features.

    `raw_df` must contain at least: player_id, GW_global, position, was_home,
    total_points, and the columns in FORM_STATS (missing ones are skipped).
    """
    df = raw_df.copy()
    for col in FORM_STATS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["was_home"] = df["was_home"].astype(int)

    df = add_rolling_features(df)
    df = add_minutes_features(df)
    return df


def feature_columns(df):
    suffixes = ("_prev", "_season_avg") + tuple(f"_roll{w}" for w in ROLLING_WINDOWS)
    cols = [c for c in df.columns if c.endswith(suffixes)]
    cols += ["games_played_so_far"] + CATEGORICAL_FEATURES + FIXTURE_FEATURES + MINUTES_FEATURES
    # start_rate_roll5 / mins60_rate_roll5 already end in _roll5, so dedupe.
    seen, deduped = set(), []
    for c in cols:
        if c in df.columns and c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped
