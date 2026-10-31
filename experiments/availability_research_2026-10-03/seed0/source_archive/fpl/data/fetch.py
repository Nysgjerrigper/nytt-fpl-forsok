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
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import re
import unicodedata
from io import StringIO

import pandas as pd
import requests

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


class ActiveSeasonNotStartedError(RuntimeError):
    """The known API-active season has no data-checked event yet."""


def _get_json(url: str) -> Any:
    response = requests.get(url, timeout=config.FPL_API_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def active_season_from_bootstrap(bootstrap: dict[str, Any]) -> str:
    """Derive the API-active season from its published GW1 deadline, never row order."""
    first = [event for event in bootstrap["events"] if event["id"] == 1]
    if len(first) != 1 or not first[0].get("deadline_time"):
        raise ValueError("bootstrap lacks a unique GW1 deadline for active-season identity")
    try:
        year = datetime.fromisoformat(first[0]["deadline_time"].replace("Z", "+00:00")).year
    except (TypeError, ValueError) as exc:
        raise ValueError("bootstrap GW1 deadline is invalid") from exc
    return f"{year}-{str(year + 1)[-2:]}"


def _integer_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    for column in columns:
        if column not in frame:
            raise ValueError(f"{label} missing column {column}")
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any() or not np.isfinite(values).all() or not (values % 1 == 0).all():
            raise ValueError(f"{label} {column} must contain nonmissing finite integer identities")
        frame[column] = values.astype(int)


def _fixture_difficulty_from_json(fixtures: pd.DataFrame, teams_map: dict[int, str]) -> pd.DataFrame:
    """Construct archive-compatible per-team/round FDR from official fixture sides."""
    fx = fixtures[fixtures.event.notna()].copy()
    _integer_columns(fx, ["event"], "fixture")
    fd = pd.concat([
        pd.DataFrame({"team": fx.team_h.map(teams_map), "GW": fx.event,
                      "fixture_difficulty": fx.team_h_difficulty}),
        pd.DataFrame({"team": fx.team_a.map(teams_map), "GW": fx.event,
                      "fixture_difficulty": fx.team_a_difficulty}),
    ], ignore_index=True)
    fd["team"] = fd.team.replace(config.TEAM_NAME_CORRECTIONS)
    fd = fd.groupby(["team", "GW"], as_index=False).fixture_difficulty.mean().sort_values(["team", "GW"])
    fd["fixture_difficulty_next3"] = fd.groupby("team").fixture_difficulty.transform(
        lambda values: values[::-1].rolling(3, min_periods=1).mean()[::-1]
    )
    return fd


def fetch_live_season_gws(season: str, *, bootstrap: dict[str, Any] | None = None
                          ) -> tuple[pd.DataFrame, dict[int, str], dict[int, int], pd.DataFrame]:
    """Fetch exact checked rounds with stable identity and fixture-time club validation.

    Histories remain per fixture, including double gameweeks. Official fixture sides
    supply the club at each match. Current bootstrap ep_this is never used as xP.
    Any metadata, history, fixture, or worker failure aborts the entire live fetch.
    """
    bootstrap = _get_json(f"{config.FPL_API_BASE}/bootstrap-static/") if bootstrap is None else bootstrap
    active = active_season_from_bootstrap(bootstrap)
    if active != season:
        raise ValueError(f"requested season {season} does not match API-active season {active}")
    events = pd.DataFrame(bootstrap["events"])
    _integer_columns(events, ["id"], "bootstrap event")
    if events.id.duplicated().any():
        raise ValueError("bootstrap contains duplicate event identities")
    checked = set(events.loc[events.data_checked.map(lambda value: value is True), "id"])
    if not checked:
        raise ActiveSeasonNotStartedError(f"API-active season {season} has no data-checked rounds")
    teams = pd.DataFrame(bootstrap["teams"])
    _integer_columns(teams, ["id"], "bootstrap team")
    if teams.id.duplicated().any() or "name" not in teams or teams.name.isna().any():
        raise ValueError("bootstrap team metadata is missing or duplicated")
    teams_map = dict(zip(teams.id, teams.name))
    types = pd.DataFrame(bootstrap["element_types"])
    _integer_columns(types, ["id"], "bootstrap position")
    if types.id.duplicated().any() or "singular_name_short" not in types:
        raise ValueError("bootstrap position metadata is missing or duplicated")
    position_map = dict(zip(types.id, types.singular_name_short.replace({"GKP": "GK"})))
    metadata = pd.DataFrame(bootstrap["elements"])
    _integer_columns(metadata, ["id", "code", "element_type"], "bootstrap player")
    if metadata.id.duplicated().any() or metadata.code.duplicated().any():
        raise ValueError("bootstrap player identities or stable codes are duplicated")
    if not {"first_name", "second_name"}.issubset(metadata) or metadata[["first_name", "second_name"]].isna().any().any():
        raise ValueError("bootstrap player name metadata is missing")
    metadata["position"] = metadata.element_type.map(position_map)
    if not metadata.position.isin(config.ONFIELD_POSITIONS).all():
        raise ValueError("bootstrap player position metadata is missing or unknown")
    metadata["name"] = (metadata.first_name + " " + metadata.second_name).str.strip()
    metadata = metadata.rename(columns={"id": "element", "code": "player_code"})
    codes_map = dict(zip(metadata.element, metadata.player_code))
    fixtures = pd.DataFrame(_get_json(f"{config.FPL_API_BASE}/fixtures/"))
    _integer_columns(fixtures, ["id", "team_h", "team_a"], "fixture")
    required_fixture = {"event", "team_h_difficulty", "team_a_difficulty"}
    if required_fixture - set(fixtures) or fixtures.id.duplicated().any():
        raise ValueError("fixture metadata is missing or identities are duplicated")
    if not fixtures.team_h.isin(teams_map).all() or not fixtures.team_a.isin(teams_map).all() or (fixtures.team_h == fixtures.team_a).any():
        raise ValueError("fixture club metadata is missing or invalid")
    for column in ("team_h_difficulty", "team_a_difficulty"):
        fixtures[column] = pd.to_numeric(fixtures[column], errors="raise")
        if not np.isfinite(fixtures[column]).all():
            raise ValueError("fixture difficulty metadata must be finite and nonmissing")
    histories = []
    with ThreadPoolExecutor(max_workers=config.FPL_SUMMARY_WORKERS) as pool:
        pending = {pool.submit(_get_json, f"{config.FPL_API_BASE}/element-summary/{element}/"): element
                   for element in sorted(codes_map)}
        for future in as_completed(pending):
            element = pending[future]
            summary = future.result()
            if not isinstance(summary, dict) or not isinstance(summary.get("history"), list):
                raise ValueError(f"player summary {element} lacks history metadata")
            history = pd.DataFrame(summary["history"])
            if history.empty:
                continue
            _integer_columns(history, ["round"], "player history")
            history = history[history["round"].isin(checked)].copy()
            if history.empty:
                continue
            _integer_columns(history, ["element", "fixture", "opponent_team"], "player history")
            if not (history.element == element).all():
                raise ValueError(f"player history element disagrees with queried player {element}")
            if "was_home" not in history or history.was_home.isna().any() or not history.was_home.isin([0, 1]).all():
                raise ValueError("player history lacks a valid fixture-side identity")
            history["was_home"] = history.was_home.astype(int)
            histories.append(history)
    if not histories:
        raise RuntimeError("checked events have no player history; refusing stale archive fallback")
    rows = pd.concat(histories, ignore_index=True)
    if rows.duplicated(["element", "fixture"]).any():
        raise ValueError("player history contains duplicate fixture identities")
    sides = pd.concat([
        pd.DataFrame({"fixture": fixtures.id, "was_home": 1, "fixture_event": fixtures.event,
                      "fixture_team": fixtures.team_h, "fixture_opponent": fixtures.team_a}),
        pd.DataFrame({"fixture": fixtures.id, "was_home": 0, "fixture_event": fixtures.event,
                      "fixture_team": fixtures.team_a, "fixture_opponent": fixtures.team_h}),
    ], ignore_index=True)
    rows = rows.merge(sides, on=["fixture", "was_home"], how="left", validate="many_to_one", indicator=True)
    if (rows._merge != "both").any() or not (rows["round"] == rows.fixture_event).all():
        raise ValueError("player history fixture is missing or its round disagrees with fixture event")
    if not (rows.opponent_team == rows.fixture_opponent).all():
        raise ValueError("player history opponent disagrees with fixture side")
    rows["team"] = rows.fixture_team.map(teams_map)
    rows = rows.merge(metadata[["element", "name", "position"]], on="element", how="left", validate="many_to_one")
    rows["GW"] = rows["round"]
    rows["xP"] = np.nan
    rows = rows.drop(columns=["_merge", "fixture_event", "fixture_team", "fixture_opponent"])
    rows = rows.sort_values(["element", "GW", "fixture"], kind="stable").reset_index(drop=True)
    return rows, teams_map, codes_map, _fixture_difficulty_from_json(fixtures, teams_map)


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


def clean_season(df: pd.DataFrame, season: str, *, teams_map: dict[int, str] | None = None,
                 codes_map: dict[int, int] | None = None, fixture_diff: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply the same cleaning the old R script did (opponent id -> name, team name
    corrections, name corrections, latin-ascii), map the per-season `element` id to the
    stable cross-season `player_code`, and merge fixture-difficulty ratings - with
    coverage guards on every join so a silently failing merge fails loudly instead."""
    df = df.copy()
    n_rows_in = len(df)

    teams_map = fetch_teams_map(season) if teams_map is None else teams_map
    df["opponent_team"] = pd.to_numeric(df["opponent_team"], errors="coerce").map(teams_map)
    df["team"] = df["team"].replace(config.TEAM_NAME_CORRECTIONS)

    # Stable cross-season identity: per-season element id -> permanent FPL player code.
    codes_map = fetch_player_codes(season) if codes_map is None else codes_map
    df["player_code"] = pd.to_numeric(df["element"], errors="coerce").map(codes_map)

    # Merge official fixture-difficulty ratings onto each player row by (team, GW).
    # `GW` here is the raw per-season gameweek, which matches fixtures.csv's `event`.
    # validate: fixture_diff is one row per (team, GW) by construction; a violation
    # means the FDR aggregation broke and every player row would silently duplicate.
    if fixture_diff is None:
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


def build_master_dataset(seasons: Sequence[str] | None = None, save: bool = True, *,
                         include_live: bool = True) -> pd.DataFrame:
    """Build archive history plus the actual API-active season, then save atomically.

    Archive-only mode makes no official FPL API calls. Global gameweeks use each
    season's canonical offset, so requested order and missing seasons cannot shift
    chronology. Active API failures never fall back to stale archive CSVs.
    """
    if Path(config.MASTER_DATASET_PATH).resolve() == Path(config.FROZEN_RESEARCH_DATASET_PATH).resolve():
        raise ValueError("mutable master path equals frozen research path; refusing ingestion")
    if isinstance(seasons, str):
        raise ValueError("seasons must be a sequence of season names")
    bootstrap = _get_json(f"{config.FPL_API_BASE}/bootstrap-static/") if include_live else None
    active = active_season_from_bootstrap(bootstrap) if bootstrap is not None else None
    if seasons is None:
        archived = [season for season in discover_seasons()
                    if season >= config.DEFAULT_START_SEASON and (active is None or season < active)]
        seasons = archived + ([active] if active is not None and active >= config.DEFAULT_START_SEASON else [])
    normalized = sorted(set(seasons))
    for season in normalized:
        match = SEASON_RE.fullmatch(season) if isinstance(season, str) else None
        if not match or int(match[2]) != (int(match[1]) + 1) % 100 or season < config.DEFAULT_START_SEASON:
            raise ValueError(f"invalid or unsupported season {season!r}")
    print(f"Seasons to fetch: {normalized}")
    cleaned = []
    for season in normalized:
        print(f"Fetching {season}...")
        if include_live and season == active:
            try:
                raw, teams_map, codes_map, fixture_diff = fetch_live_season_gws(season, bootstrap=bootstrap)
            except ActiveSeasonNotStartedError:
                logger.warning("Skipping API-active %s because no rounds are data-checked yet.", season)
                continue
            frame = clean_season(raw, season, teams_map=teams_map, codes_map=codes_map, fixture_diff=fixture_diff)
        else:
            raw = fetch_season_gws(season)
            frame = clean_season(raw, season)
        if frame.empty:
            raise RuntimeError(f"season {season} produced an empty historical dataset")
        frame["GW"] = pd.to_numeric(frame["GW"], errors="raise")
        frame["GW_global"] = frame["GW"] + config.season_start_gw(season) - 1
        print(f"  {season}: {len(frame)} rows, global GW {frame.GW_global.min()}-{frame.GW_global.max()}")
        cleaned.append(frame)
    if not cleaned:
        raise RuntimeError("No seasons with published gameweek data were available.")
    master = assign_player_ids(pd.concat(cleaned, ignore_index=True, sort=False))
    order = ["player_id", "GW_global"] + (["fixture"] if "fixture" in master else [])
    master = master.sort_values(order, kind="stable").reset_index(drop=True)
    if save:
        destination = Path(config.MASTER_DATASET_PATH)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
            master.to_csv(temporary, index=False)
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"Saved master dataset to {destination} ({len(master)} rows, {len(master.columns)} cols)")
    return master


if __name__ == "__main__":
    build_master_dataset()
