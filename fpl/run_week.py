"""
Weekly driver: run this once a week during the season to get a squad/transfer/
captain recommendation for the next gameweek.

    python -m fpl.run_week                          # fresh squad-build advice (e.g. preseason/GW1)
    python -m fpl.run_week --team-id 1234567         # continue YOUR existing FPL team
    python -m fpl.run_week --team-id 1234567 --horizon 4

What it does, in order:
1. Refreshes the historical dataset (fpl/data/fetch.py) so this season's
   played gameweeks are included.
2. Retrains the GBM models on everything played so far.
3. Pulls the upcoming fixture(s) from the official FPL API (opponent/home-away
   per team - vaastav's historical data obviously has no rows yet for a
   gameweek that hasn't been played) and predicts points for every player
   over the next `--horizon` gameweeks, projecting each player's current form
   forward.
4. If `--team-id` is given, pulls your actual current squad/bank from the
   public FPL API and asks the MILP optimizer for the best transfers/lineup/
   captain/chip continuing from that squad. Without it, prints fresh-build
   advice as if drafting a squad from scratch.

Caveat: the official FPL API only exposes a season's players/fixtures once
that season is set up on the site (typically a few weeks before its GW1) -
this cannot be exercised end-to-end for 2026-27 until then.
"""
import argparse
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl import config, features
from fpl.data import fetch
from fpl.model.train import POSITIONS, load_features, train_position_model
from fpl.model.predict import _load_blend_weights
from fpl.model import models as model_registry
from fpl.model.ensemble import PositionEnsemble
from fpl.milp import optimize

FPL_API = "https://fantasy.premierleague.com/api"
ELEMENT_TYPE_TO_POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _normalize_name(name):
    stripped = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    stripped = config.NAME_CORRECTIONS.get(stripped, stripped)
    return stripped.strip().lower()


def fetch_bootstrap():
    resp = requests.get(f"{FPL_API}/bootstrap-static/", timeout=30)
    resp.raise_for_status()
    return resp.json()


def determine_target_gw(bootstrap):
    events = bootstrap["events"]
    unfinished = [e for e in events if not e["finished"]]
    if not unfinished:
        sys.exit("No unfinished gameweek found - season may not be set up on the FPL site yet.")
    return min(unfinished, key=lambda e: e["id"])["id"]


def fetch_fixtures(gw):
    resp = requests.get(f"{FPL_API}/fixtures/", params={"event": gw}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def build_team_fixture_map(bootstrap, gw):
    """team_name -> (opponent_team_name, was_home) for one gameweek."""
    teams_by_id = {t["id"]: t["name"] for t in bootstrap["teams"]}
    result = {}
    for fx in fetch_fixtures(gw):
        home, away = teams_by_id[fx["team_h"]], teams_by_id[fx["team_a"]]
        result[home] = (away, True)
        result[away] = (home, False)
    return result


def get_user_squad(team_id, bootstrap, name_to_player_id):
    """Public FPL API: current squad, bank and free transfers for an existing team."""
    last_finished_gw = max(e["id"] for e in bootstrap["events"] if e["finished"])
    resp = requests.get(f"{FPL_API}/entry/{team_id}/event/{last_finished_gw}/picks/", timeout=30)
    resp.raise_for_status()
    data = resp.json()

    elements_by_id = {el["id"]: el for el in bootstrap["elements"]}
    squad_ids = []
    for pick in data["picks"]:
        el = elements_by_id[pick["element"]]
        full_name = _normalize_name(f"{el['first_name']} {el['second_name']}")
        player_id = name_to_player_id.get(full_name)
        if player_id is None:
            print(f"WARNING: could not match FPL player '{el['first_name']} {el['second_name']}' "
                  f"to a known player_id - excluded from optimization.")
            continue
        squad_ids.append(player_id)

    bank = data["entry_history"]["bank"]  # in 0.1m units, matches `value` column
    return squad_ids, float(bank)


def latest_snapshot(feat_df):
    """One row per player: their most recent played gameweek, used as the
    jump-off point for projecting form into the future."""
    return feat_df.sort_values("GW_global").groupby("player_id", as_index=False).tail(1)


def build_future_predictions(feat_df, feature_cols, models, bootstrap, start_gw, horizon):
    snapshot = latest_snapshot(feat_df)
    rows = []
    for i in range(horizon):
        gw = start_gw + i
        try:
            fixture_map = build_team_fixture_map(bootstrap, gw)
        except requests.HTTPError:
            print(f"No fixtures available yet for GW {gw}, stopping horizon there.")
            break
        gw_rows = snapshot.copy()
        gw_rows["GW"] = gw
        gw_rows["GW_global"] = gw
        opponents, homes = [], []
        for team in gw_rows["team"]:
            opp, is_home = fixture_map.get(team, (None, None))
            opponents.append(opp)
            homes.append(is_home)
        gw_rows["opponent_team"] = opponents
        gw_rows["was_home"] = pd.Series(homes).fillna(False).astype(int).values
        gw_rows = gw_rows[gw_rows["opponent_team"].notna()]  # drop teams without a fixture this GW (blanks)

        gw_rows["predicted_total_points"] = 0.0
        for pos in POSITIONS:
            mask = gw_rows["position"] == pos
            if mask.any() and pos in models:
                gw_rows.loc[mask, "predicted_total_points"] = models[pos].predict(gw_rows.loc[mask, feature_cols])
        rows.append(gw_rows)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Weekly FPL squad/transfer recommendation")
    parser.add_argument("--team-id", type=int, default=None,
                         help="Your public FPL team ID, to continue your real squad. "
                              "Omit for a fresh-build recommendation.")
    parser.add_argument("--horizon", type=int, default=3, help="Gameweeks to look ahead")
    parser.add_argument("--time-limit", type=float, default=120, help="Solver time limit per GW, seconds")
    parser.add_argument("--free-transfers", type=int, default=1, help="Your free transfers, if --team-id is given")
    args = parser.parse_args()

    print("--- Refreshing historical dataset ---")
    fetch.build_master_dataset()

    print("--- Building features & training models on all data so far ---")
    feat_df = load_features()
    feature_cols = features.feature_columns(feat_df)
    models = {}
    for pos in POSITIONS:
        pos_df = feat_df[feat_df["position"] == pos]
        weights = _load_blend_weights(pos)
        if weights is None:
            print(f"No saved ensemble weights for {pos} yet - run `python -m fpl.model.train` first "
                  f"for the full ensemble; using a plain LightGBM model for now.")
            models[pos] = train_position_model(feat_df, feature_cols, pos)
        else:
            X, y = pos_df[feature_cols], pos_df[features.TARGET_COL]
            members = {name: model_registry.fit_model(name, X, y) for name in weights}
            models[pos] = PositionEnsemble(members, weights)

    bootstrap = fetch_bootstrap()
    target_gw = determine_target_gw(bootstrap)
    print(f"--- Target gameweek: {target_gw} ---")

    preds = build_future_predictions(feat_df, feature_cols, models, bootstrap, target_gw, args.horizon)
    if preds.empty:
        sys.exit("Could not build any predictions - fixtures for the target gameweek aren't published yet.")

    out_cols = ["player_id", "GW", "name", "position", "team", "value", "predicted_total_points"]
    preds = preds[out_cols]
    config.PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(config.PREDICTIONS_PATH, index=False)
    print(f"Saved {len(preds)} predicted rows to {config.PREDICTIONS_PATH}")

    opt_args = [
        "--predictions-csv", str(config.PREDICTIONS_PATH),
        "--start-gw", str(target_gw),
        "--max-gw", str(target_gw + preds["GW"].nunique() - 1),
        "--horizon", str(args.horizon),
        "--time-limit", str(args.time_limit),
    ]

    if args.team_id:
        name_to_player_id = dict(zip(feat_df["name"].apply(_normalize_name), feat_df["player_id"]))
        squad_ids, bank = get_user_squad(args.team_id, bootstrap, name_to_player_id)
        print(f"--- Continuing team {args.team_id}: {len(squad_ids)} matched players, bank={bank} ---")
        opt_args += [
            "--initial-squad", ",".join(map(str, squad_ids)),
            "--initial-budget", str(bank),
            "--initial-ft", str(args.free_transfers),
        ]
    else:
        print("--- No --team-id given: producing a fresh-build recommendation (full budget, no existing squad) ---")

    optimize.run(optimize.parse_args(opt_args))


if __name__ == "__main__":
    main()
