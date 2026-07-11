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

# Subset of FORM_STATS we additionally smooth exponentially - see add_ewma_features.
# Baseline experiments found exponential decay tracks form better than a flat window
# (a hat-trick two games ago should weigh less than one last game), so we only pay the
# extra columns for the stats where recency matters most, not all of FORM_STATS.
EWMA_STATS = [
    "total_points",
    "minutes",
    "bps",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "goals_scored",
    "assists",
    "threat",
    "creativity",
]
EWMA_HALFLIFE = 3
EWMA_FEATURES = [f"{stat}_ewm{EWMA_HALFLIFE}" for stat in EWMA_STATS]

# Per-90 rate features - see add_per90_features. Raw rolling counts conflate "scores a lot"
# with "plays a lot"; normalising by minutes isolates finishing/creation efficiency, which
# generalises better to players whose minutes are about to change (new signing, return from
# injury).
PER90_STATS = ["goals_scored", "assists", "expected_goals", "expected_assists"]
PER90_FEATURES = [f"{stat}_per90_roll5" for stat in PER90_STATS]

# Opponent-strength features - see add_opponent_strength_features. Fixture-difficulty rating
# is the FPL's static pre-season guess; these are the opponent's *actual* recent attacking/
# defensive form, which moves within a season as teams over- or under-perform their reputation.
OPPONENT_FEATURES = ["opp_attack_roll6", "opp_defense_roll6", "opp_cs_rate_roll6"]

# Bookmaker-style prior: FPL publishes its own pre-match expected points (xP). Using shifted
# forms only (never the current GW's xP) keeps it conservative and guaranteed leakage-free even
# if the raw column ever turned out to be populated post-match.
XP_FEATURES = ["xP_prev", "xP_roll3"]

TARGET_COL = "total_points"


def _ensure_sorted(df):
    # Stable sort: rows within the same (player, GW_global) - double-gameweek fixtures -
    # must keep their input (chronological) order, or "the first fixture of the round"
    # would be arbitrary run to run. The default quicksort is not stable.
    return df.sort_values(["player_id", "GW_global"], kind="mergesort").reset_index(drop=True)


def _season_key(df):
    """Grouping key for per-season logic. Uses the real `season` column when present;
    otherwise derives the season ordinal from GW_global (38 GWs per season by construction
    of the global counter), so synthetic test frames without a season column still work."""
    if "season" in df.columns:
        return df["season"]
    return (df["GW_global"] - 1) // 38


def add_rolling_features(df):
    """Add, for each stat in FORM_STATS, rolling means over the last N gameweeks plus two
    expanding means at different horizons - all shifted by one GW so a row never sees its
    own outcome (no leakage).

    The two expanding horizons are deliberately separate features:
    - <col>_season_avg: expanding mean within the CURRENT season only - resets each August,
      "how has he been doing this campaign".
    - <col>_career_avg: expanding mean over the player's entire history in the dataset -
      the stable long-run level a season start regresses toward.
    (Until 2026-07-06 a single column named _season_avg actually computed the career mean -
    grouped by player only, never reset per season - so the name lied about the horizon.
    Both horizons carry signal, so the fix keeps both under honest names.)
    """
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
        new_cols[f"{col}_career_avg"] = (
            shifted.groupby(df["player_id"]).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
        )
        # NB: the shift above is per player ACROSS seasons, so a player's first row of a new
        # season carries his last GW of the previous season - grouping the shifted series by
        # (player, season) then means that one cross-boundary value seeds the season average.
        # That's one stale-but-real observation diluted over the season; accepted rather than
        # re-shifting within season, which would cost every player his opening-GW feature.
        new_cols[f"{col}_season_avg"] = (
            shifted.groupby([df["player_id"], _season_key(df)]).expanding(min_periods=1).mean()
            .reset_index(level=[0, 1], drop=True)
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


def add_ewma_features(df):
    """Add an exponentially-weighted moving mean (halflife=3) per player for EWMA_STATS.

    Exponential decay is used instead of another flat window because form is recency-
    weighted: last week's return should count for more than a return five weeks ago, and a
    halflife of 3 gameweeks roughly matches how quickly a player's role/fitness turns over.
    Shifted one GW first, exactly like the flat rolling features, so the value at GW t is
    built only from strictly-earlier gameweeks and never sees its own outcome.
    """
    df = _ensure_sorted(df)

    new_cols = {}
    for col in EWMA_STATS:
        if col not in df.columns:
            continue
        shifted = df.groupby("player_id", sort=False)[col].shift(1)
        new_cols[f"{col}_ewm{EWMA_HALFLIFE}"] = (
            shifted.groupby(df["player_id"]).ewm(halflife=EWMA_HALFLIFE).mean().reset_index(level=0, drop=True)
        )
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_per90_features(df):
    """Add per-90-minutes rates for PER90_STATS from shifted rolling(5) sums.

    Dividing a rolling *sum* of the stat by the matching rolling *sum* of minutes (rather than
    averaging per-game ratios) weights each game by how long the player was on the pitch, so a
    5-minute cameo can't swing the rate. A player with 0 minutes over the window gets 0.0 rather
    than a divide-by-zero. Both numerator and denominator are shifted one GW, so leakage-free.
    """
    df = _ensure_sorted(df)

    minutes_shifted = df.groupby("player_id", sort=False)["minutes"].shift(1)
    minutes_sum = (
        minutes_shifted.groupby(df["player_id"]).rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
    )

    new_cols = {}
    for col in PER90_STATS:
        if col not in df.columns:
            continue
        stat_shifted = df.groupby("player_id", sort=False)[col].shift(1)
        stat_sum = (
            stat_shifted.groupby(df["player_id"]).rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
        )
        rate = stat_sum.div(minutes_sum).mul(90.0)
        # 0 minutes over the window -> undefined rate; treat as no attacking output (0.0).
        new_cols[f"{col}_per90_roll5"] = rate.where(minutes_sum > 0, 0.0)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_opponent_strength_features(df):
    """Attach the *opponent's* recent form to each player row (how tough is this week's rival).

    Built in two stages. First collapse the player-level rows into one row per (team, GW_global):
    goals scored = sum of that team's players' goals that GW, goals conceded = max of the shared
    per-player goals_conceded (per-player because it depends on minutes on the pitch, so the max is
    the team's true concession), clean sheet = conceded == 0. A shifted rolling(6) mean of each gives
    the team's trailing attack/defense/clean-sheet form using only gameweeks STRICTLY BEFORE the
    current one (shift-by-one on the team's GW-ordered series, same discipline as the player features).

    Then merge that trailing form onto each player row keyed by the row's `opponent_team`, so the
    feature answers "how well has my opponent been playing lately" rather than anything about the
    player's own team. The merge is on the un-shifted GW_global because the team series was already
    shifted, so `opp_*_roll6` at GW t reflects the opponent's form through GW t-1 only.
    """
    if "opponent_team" not in df.columns or "team" not in df.columns:
        return df

    # One row per team per gameweek. goals_conceded is a per-player value (minutes-dependent), so max
    # recovers the actual number the team shipped; goals_scored is a per-player tally, so sum totals it.
    team_gw = (
        df.groupby(["team", "GW_global"], sort=True)
        .agg(team_goals=("goals_scored", "sum"), team_conceded=("goals_conceded", "max"))
        .reset_index()
    )
    team_gw["team_cs"] = (team_gw["team_conceded"] == 0).astype(float)
    team_gw = team_gw.sort_values(["team", "GW_global"]).reset_index(drop=True)

    grouped = team_gw.groupby("team", sort=False)
    roll_src = {
        "opp_attack_roll6": "team_goals",
        "opp_defense_roll6": "team_conceded",
        "opp_cs_rate_roll6": "team_cs",
    }
    for out_col, src_col in roll_src.items():
        shifted = grouped[src_col].shift(1)
        team_gw[out_col] = (
            shifted.groupby(team_gw["team"]).rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
        )

    opp_form = team_gw[["team", "GW_global"] + list(roll_src)].rename(columns={"team": "opponent_team"})
    return df.merge(opp_form, on=["opponent_team", "GW_global"], how="left")


def add_xp_features(df):
    """Add shifted forms of FPL's own pre-match expected-points column `xP`.

    We deliberately use the previous-GW value and a shifted rolling(3) mean instead of the current
    GW's raw xP: xP is nominally known before kickoff, but treating it as a lagged feature is the
    conservative choice that stays leakage-free regardless of when the historical dumps stamped it.
    """
    if "xP" not in df.columns:
        return df
    df = _ensure_sorted(df)

    xp_shifted = df.groupby("player_id", sort=False)["xP"].shift(1)
    new_cols = {
        "xP_prev": xp_shifted,
        "xP_roll3": (
            xp_shifted.groupby(df["player_id"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        ),
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def _player_shifted_columns(df):
    """Every player-level column built from a per-row shift(1): the _prev/rolling/expanding
    families, EWMA, per-90, minutes-projection and xP features, plus the appearance counter.
    Excludes opp_* (shifted per-TEAM-gameweek, so already strictly earlier-round) and the
    fixture/was_home columns (known-ahead, legitimately per-fixture)."""
    suffixes = (
        ("_prev", "_season_avg", "_career_avg", f"_ewm{EWMA_HALFLIFE}")
        + tuple(f"_roll{w}" for w in ROLLING_WINDOWS)
    )
    cols = [c for c in df.columns if c.endswith(suffixes) and not c.startswith("opp_")]
    cols += [c for c in MINUTES_FEATURES + ["games_played_so_far"] if c in df.columns]
    return sorted(set(cols))


def enforce_gameweek_level_shift(df):
    """Close the double-gameweek leak in all per-row-shifted player features.

    shift(1) is per ROW, so when a player has two fixtures in one GW_global (a double
    gameweek - 8.6% of rows), the second fixture's "form" includes the first fixture of
    the SAME round: information that does not exist when the squad locks before the round.
    DGW players are exactly the ones the MILP hunts, so this optimism concentrated where
    decisions are made (audit finding A3).

    Fix: within each (player_id, GW_global) group, broadcast the FIRST row's values of
    every player-shifted feature to all rows of the group. The first row's shifted
    features are built from strictly-earlier gameweeks only, and their value does not
    depend on the within-round fixture order - so after this, every fixture of a round
    is predicted with the same deadline-time form, exactly like live mode. Per-fixture
    known-ahead columns (opponent, home/away, FDR) are untouched. games_played_so_far is
    included: freezing the appearance counter at the round's start is the same
    deadline-time semantic (the second-fixture count would technically be schedule-known,
    but a one-appearance offset carries no signal worth a special case).
    """
    df = _ensure_sorted(df)
    cols = _player_shifted_columns(df)
    dup = df.duplicated(["player_id", "GW_global"], keep=False)
    if dup.any() and cols:
        firsts = (
            df.loc[dup]
            .groupby(["player_id", "GW_global"], sort=False)[cols]
            .transform("first")
        )
        df.loc[dup, cols] = firsts
    return df


def build_feature_frame(raw_df):
    """Full feature pipeline: clean dtypes, add rolling form features.

    `raw_df` must contain at least: player_id, GW_global, position, was_home,
    total_points, and the columns in FORM_STATS (missing ones are skipped).
    """
    df = raw_df.copy()
    for col in FORM_STATS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # 0-fill only within seasons where the stat was actually recorded. For seasons where
        # the column simply didn't exist yet (the Opta xG family and `starts` before 2022-23),
        # leave NaN: "this metric wasn't collected" is not the same fact as "the player
        # recorded exactly zero", and encoding it as 0 taught the models a falsehood (the old
        # behaviour - see RESEARCH_LOG.md). LightGBM/XGBoost/CatBoost branch on NaN natively;
        # the sklearn models impute it downstream via their SimpleImputer pipelines.
        season_has_stat = df.groupby(_season_key(df))[col].transform(lambda s: s.notna().any())
        df.loc[season_has_stat, col] = df.loc[season_has_stat, col].fillna(0.0)
    if "xP" in df.columns:
        df["xP"] = pd.to_numeric(df["xP"], errors="coerce")
    df["was_home"] = df["was_home"].astype(int)

    df = add_rolling_features(df)
    df = add_minutes_features(df)
    df = add_ewma_features(df)
    df = add_per90_features(df)
    df = add_opponent_strength_features(df)
    df = add_xp_features(df)
    df = enforce_gameweek_level_shift(df)
    return df


def feature_columns(df):
    suffixes = ("_prev", "_season_avg", "_career_avg") + tuple(f"_roll{w}" for w in ROLLING_WINDOWS)
    cols = [c for c in df.columns if c.endswith(suffixes)]
    cols += ["games_played_so_far"] + CATEGORICAL_FEATURES + FIXTURE_FEATURES + MINUTES_FEATURES
    cols += EWMA_FEATURES + PER90_FEATURES + OPPONENT_FEATURES + XP_FEATURES
    # Several of the above already match a suffix above (per90 ends in _roll5, xP_prev in
    # _prev, etc.); the seen-set dedupe below collapses those to a single entry.
    seen, deduped = set(), []
    for c in cols:
        if c in df.columns and c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped
