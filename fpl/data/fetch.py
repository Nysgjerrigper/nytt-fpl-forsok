"""
Python replacement for the old `Datasett/R-Script 1 fetching data.r`.

Fetches every completed season from vaastav's FPL GitHub data repo plus
whatever gameweeks exist so far for the current/in-progress season, and
builds one clean, ascending-gameweek master dataset for the whole pipeline.

Unlike the old R script, nothing about which seasons exist or how many
gameweeks have been played is hardcoded — both are discovered at run time,
so this keeps working next season without edits.
"""
import logging
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

logger = logging.getLogger(__name__)


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


def fetch_player_codes(season):
    """Per-season element id -> globally stable FPL player `code`, from players_raw.csv.

    The `element`/`id` column resets every season (the same integer is a different player
    next year), but FPL's `code` is a permanent per-person identifier carried across
    seasons - the correct cross-season join key (TODO 4.8). Name-based identity, the old
    approach, both MERGED distinct players who share a name (the two Ben Davies) and SPLIT
    one player whose name is spelled differently across seasons' dumps.
    """
    url = f"{config.GITHUB_RAW_BASE}/{season}/players_raw.csv"
    players = pd.read_csv(url, usecols=["id", "code"])
    return dict(zip(players["id"], players["code"]))


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

    # One row per (team, fixture) from each side's perspective - vectorized (the repo's
    # no-iterrows-in-ETL rule): stack the home view and the away view of the fixture list.
    fd = pd.concat([
        pd.DataFrame({"team": fx["team_h"].map(teams_map), "GW": fx["event"],
                      "fixture_difficulty": fx["team_h_difficulty"]}),
        pd.DataFrame({"team": fx["team_a"].map(teams_map), "GW": fx["event"],
                      "fixture_difficulty": fx["team_a_difficulty"]}),
    ], ignore_index=True)
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
    """Apply the same cleaning the old R script did (opponent id -> name, team name
    corrections, name corrections, latin-ascii), map the per-season `element` id to the
    stable cross-season `player_code`, and merge fixture-difficulty ratings - with
    coverage guards on every join so a silently failing merge fails loudly instead."""
    df = df.copy()
    n_rows_in = len(df)

    teams_map = fetch_teams_map(season)
    df["opponent_team"] = pd.to_numeric(df["opponent_team"], errors="coerce").map(teams_map)
    df["team"] = df["team"].replace(config.TEAM_NAME_CORRECTIONS)

    # Stable cross-season identity: per-season element id -> permanent FPL player code.
    codes_map = fetch_player_codes(season)
    df["player_code"] = pd.to_numeric(df["element"], errors="coerce").map(codes_map)

    # Merge official fixture-difficulty ratings onto each player row by (team, GW).
    # `GW` here is the raw per-season gameweek, which matches fixtures.csv's `event`.
    # validate: fixture_diff is one row per (team, GW) by construction; a violation
    # means the FDR aggregation broke and every player row would silently duplicate.
    fixture_diff = fetch_fixture_difficulty(season, teams_map)
    gw_col = "GW" if "GW" in df.columns else "round"
    df["_gw_key"] = pd.to_numeric(df[gw_col], errors="coerce")
    df = df.merge(
        fixture_diff.rename(columns={"GW": "_gw_key"}), on=["team", "_gw_key"], how="left",
        validate="many_to_one",
    ).drop(columns="_gw_key")

    df["name"] = df["name"].apply(_strip_accents)
    df["name"] = df["name"].replace(config.NAME_CORRECTIONS)
    # Ben Davies display disambiguation (two distinct players share the name; their
    # player_code already separates them - this only keeps human-readable output clear).
    df.loc[(df["name"] == "Ben Davies") & (df["team"] == "Liverpool"), "name"] = "Ben Davies Liverpool"

    if "position" in df.columns:
        df = df[df["position"].isin(config.ONFIELD_POSITIONS)].copy()

    df = df.drop(columns=["element"], errors="ignore")
    df["was_home"] = df["was_home"].astype(int)
    df["season"] = season

    # Join guards (TODO 4.8): a left merge can't drop rows, but it CAN silently miss -
    # log coverage and fail hard when a mapping goes badly wrong rather than training on
    # rows whose difficulty/identity quietly became NaN.
    fdr_cov = df["fixture_difficulty"].notna().mean()
    code_cov = df["player_code"].notna().mean()
    opp_cov = df["opponent_team"].notna().mean()
    logger.info("%s: %d rows in -> %d after cleaning; coverage fdr=%.3f code=%.3f opponent=%.3f",
                season, n_rows_in, len(df), fdr_cov, code_cov, opp_cov)
    if code_cov < 0.99 or opp_cov < 0.99:
        raise ValueError(
            f"{season}: join coverage collapsed (player_code {code_cov:.3f}, "
            f"opponent {opp_cov:.3f}) - upstream schema likely changed; refusing to build "
            "a master dataset with broken identity/opponent mappings."
        )
    if fdr_cov < 0.95:
        logger.warning("%s: fixture-difficulty coverage only %.3f - check fixtures.csv team names.",
                       season, fdr_cov)
    return df


def assign_player_ids(master):
    """Set `player_id` from the stable FPL `player_code` (TODO 4.8).

    player_id IS the FPL code: one integer per real person, identical across seasons and
    identical to bootstrap-static's `code` field, so the live path (fpl.run_week) matches
    API players by id instead of normalized-name lookups (which silently dropped players
    whose spelling differed between the API and vaastav's dumps). The rare row whose
    element id was missing from players_raw.csv falls back to a name-factorized id offset
    far above the real code range, so it stays internally consistent but can never
    collide with (or be mistaken for) a genuine FPL code.
    """
    missing = master["player_code"].isna()
    if missing.any():
        logger.warning("%d rows (%d names) lack a player_code - assigning name-based fallback ids.",
                       int(missing.sum()), master.loc[missing, "name"].nunique())
        fallback = pd.factorize(master.loc[missing, "name"])[0] + config.FALLBACK_PLAYER_ID_OFFSET
        master.loc[missing, "player_code"] = fallback
    master["player_id"] = master["player_code"].astype(int)
    return master


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
    master = assign_player_ids(master)
    master = master.sort_values(["player_id", "GW_global"]).reset_index(drop=True)

    if save:
        config.MASTER_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        master.to_csv(config.MASTER_DATASET_PATH, index=False)
        print(f"Saved master dataset to {config.MASTER_DATASET_PATH} ({master.shape[0]} rows, {master.shape[1]} cols)")

    return master


if __name__ == "__main__":
    build_master_dataset()
