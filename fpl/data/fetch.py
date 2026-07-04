"""
Python replacement for the old `Datasett/R-Script 1 fetching data.r`.

Fetches every completed season from vaastav's FPL GitHub data repo plus
whatever gameweeks exist so far for the current/in-progress season, and
builds one clean, ascending-gameweek master dataset for the whole pipeline.

Unlike the old R script, nothing about which seasons exist or how many
gameweeks have been played is hardcoded — both are discovered at run time,
so this keeps working next season without edits.
"""
import re
import sys
import unicodedata
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fpl import config

SEASON_RE = re.compile(r"^(\d{4})-(\d{2})$")


def _strip_accents(text):
    if not isinstance(text, str):
        return text
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def discover_seasons():
    """Return all season folder names (e.g. '2022-23') available in the data repo, sorted."""
    resp = requests.get(config.GITHUB_API_SEASONS_URL, timeout=30)
    resp.raise_for_status()
    names = [entry["name"] for entry in resp.json() if entry.get("type") == "dir"]
    seasons = sorted(n for n in names if SEASON_RE.match(n))
    return seasons


def _url_exists(url):
    resp = requests.head(url, timeout=15)
    return resp.status_code == 200


def fetch_teams_map(season):
    """id -> team name for a given season, from data/{season}/teams.csv."""
    url = f"{config.GITHUB_RAW_BASE}/{season}/teams.csv"
    teams = pd.read_csv(url)
    return dict(zip(teams["id"], teams["name"]))


def fetch_fixture_difficulty(season, teams_map):
    """Build a per-(team, GW) fixture-difficulty table from data/{season}/fixtures.csv.

    Returns columns [team, GW, fixture_difficulty, fixture_difficulty_next3]:
    - fixture_difficulty: the official FPL Fixture Difficulty Rating (1=easiest,
      5=hardest) that this team faces in this gameweek. This is the home-team's
      difficulty for the home side and the away-team's difficulty for the away
      side. Double gameweeks are averaged to one value per (team, GW).
    - fixture_difficulty_next3: mean FDR over this fixture and the team's next two
      scheduled fixtures - a "fixture run" signal (an easy patch of fixtures ahead
      is a classic reason to bring a player in early).

    Both are legitimate inputs, NOT leakage: the fixture list and its difficulty
    ratings are published well before each gameweek is played, so a model
    predicting GW t genuinely knows who each team plays at t, t+1, t+2.

    Team names are corrected via config.TEAM_NAME_CORRECTIONS so they line up with
    the corrected `team` column on the player rows they'll be merged onto.
    """
    url = f"{config.GITHUB_RAW_BASE}/{season}/fixtures.csv"
    fx = pd.read_csv(url)
    fx = fx[fx["event"].notna()].copy()
    fx["event"] = fx["event"].astype(int)

    records = []
    for _, r in fx.iterrows():
        home, away = teams_map.get(r["team_h"]), teams_map.get(r["team_a"])
        records.append({"team": home, "GW": r["event"], "fixture_difficulty": r["team_h_difficulty"]})
        records.append({"team": away, "GW": r["event"], "fixture_difficulty": r["team_a_difficulty"]})
    fd = pd.DataFrame.from_records(records)
    fd["team"] = fd["team"].replace(config.TEAM_NAME_CORRECTIONS)
    fd["fixture_difficulty"] = pd.to_numeric(fd["fixture_difficulty"], errors="coerce")

    # One row per (team, GW): a double gameweek collapses to its mean difficulty.
    fd = fd.groupby(["team", "GW"], as_index=False)["fixture_difficulty"].mean()
    fd = fd.sort_values(["team", "GW"])
    # Forward-looking window: this fixture + the next two, per team. Reversing before
    # a trailing rolling mean turns it into a leading (future-facing) window.
    fd["fixture_difficulty_next3"] = (
        fd.groupby("team")["fixture_difficulty"]
        .transform(lambda s: s[::-1].rolling(3, min_periods=1).mean()[::-1])
    )
    return fd


def fetch_season_gws(season):
    """Fetch all available gameweek rows for a season as one DataFrame with a 'GW' column."""
    merged_url = f"{config.GITHUB_RAW_BASE}/{season}/gws/merged_gw.csv"
    if _url_exists(merged_url):
        df = pd.read_csv(merged_url, encoding="utf-8")
        if "GW" not in df.columns and "round" in df.columns:
            df["GW"] = df["round"]
        return df

    # Season in progress: no merged_gw.csv yet, pull gw1.csv, gw2.csv, ... until 404.
    frames = []
    gw = 1
    while True:
        url = f"{config.GITHUB_RAW_BASE}/{season}/gws/gw{gw}.csv"
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            break
        gw_df = pd.read_csv(StringIO(resp.text), encoding="utf-8")
        gw_df["GW"] = gw
        frames.append(gw_df)
        gw += 1
    if not frames:
        raise RuntimeError(f"No gameweek data found yet for season {season}.")
    return pd.concat(frames, ignore_index=True)


def clean_season(df, season):
    """Apply the same cleaning the old R script did: opponent id -> name, team name
    corrections, name corrections, latin-ascii, drop assistant managers/element col."""
    df = df.copy()

    teams_map = fetch_teams_map(season)
    df["opponent_team"] = pd.to_numeric(df["opponent_team"], errors="coerce").map(teams_map)
    df["team"] = df["team"].replace(config.TEAM_NAME_CORRECTIONS)

    # Merge official fixture-difficulty ratings onto each player row by (team, GW).
    # `GW` here is the raw per-season gameweek, which matches fixtures.csv's `event`.
    fixture_diff = fetch_fixture_difficulty(season, teams_map)
    gw_col = "GW" if "GW" in df.columns else "round"
    df["_gw_key"] = pd.to_numeric(df[gw_col], errors="coerce")
    df = df.merge(
        fixture_diff.rename(columns={"GW": "_gw_key"}), on=["team", "_gw_key"], how="left"
    ).drop(columns="_gw_key")

    df["name"] = df["name"].apply(_strip_accents)
    df["name"] = df["name"].replace(config.NAME_CORRECTIONS)
    # Ben Davies disambiguation (there are two Ben Davies in the PL - Spurs/Fulham DEF and
    # a Liverpool one who briefly appeared - see legacy R script for context).
    df.loc[(df["name"] == "Ben Davies") & (df["team"] == "Liverpool"), "name"] = "Ben Davies Liverpool"

    if "position" in df.columns:
        df = df[df["position"].isin(config.ONFIELD_POSITIONS)].copy()

    df = df.drop(columns=["element"], errors="ignore")
    df["was_home"] = df["was_home"].astype(int)
    df["season"] = season
    return df


def build_master_dataset(seasons=None, save=True):
    """Fetch + clean all seasons and concatenate into one ascending-GW dataset.

    Column set is the UNION across seasons (not the old script's positional
    "first 40 columns" hack) so newer stats - e.g. the 2025-26 defensive
    contribution columns - are kept for the seasons that have them instead
    of being silently dropped.
    """
    if seasons is None:
        seasons = [s for s in discover_seasons() if s >= config.DEFAULT_START_SEASON]
    print(f"Seasons to fetch: {seasons}")

    cleaned = []
    gw_offset = 0
    for season in seasons:
        print(f"Fetching {season}...")
        raw = fetch_season_gws(season)
        df = clean_season(raw, season)
        df["GW"] = pd.to_numeric(df["GW"])
        n_gws_this_season = int(df["GW"].max())
        df["GW_global"] = df["GW"] + gw_offset
        gw_offset += config.GWS_PER_SEASON
        print(f"  {season}: {df.shape[0]} rows, GW 1-{n_gws_this_season} "
              f"-> global GW {gw_offset - config.GWS_PER_SEASON + 1}-{gw_offset - config.GWS_PER_SEASON + n_gws_this_season}")
        cleaned.append(df)

    master = pd.concat(cleaned, ignore_index=True, sort=False)
    master["player_id"] = pd.factorize(master["name"])[0] + 1
    master = master.sort_values(["player_id", "GW_global"]).reset_index(drop=True)

    if save:
        config.MASTER_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        master.to_csv(config.MASTER_DATASET_PATH, index=False)
        print(f"Saved master dataset to {config.MASTER_DATASET_PATH} ({master.shape[0]} rows, {master.shape[1]} cols)")

    return master


if __name__ == "__main__":
    build_master_dataset()
