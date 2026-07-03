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
    return df


def feature_columns(df):
    suffixes = ("_prev", "_season_avg") + tuple(f"_roll{w}" for w in ROLLING_WINDOWS)
    cols = [c for c in df.columns if c.endswith(suffixes)]
    cols += ["games_played_so_far"] + CATEGORICAL_FEATURES
    return [c for c in cols if c in df.columns]


def split_by_position(df):
    return {pos: df[df["position"] == pos].copy() for pos in ["GK", "DEF", "MID", "FWD"]}
