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

import pandas as pd
import requests

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
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
        from io import StringIO

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
