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
from fpl.model.train import POSITIONS, fit_holdout_weights
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
    """team_name -> (opponent_name, was_home, fixture_difficulty) for one gameweek.

    Team names go through config.TEAM_NAME_CORRECTIONS so they match the dataset's
    corrected `team` column (the API says "Spurs"/"Man Utd", the dataset says
    "Tottenham"/"Man United" - without this, lookups for those teams silently miss
    and their players get dropped from every live prediction).

    Double gameweeks: opponent/home-away come from the first fixture, difficulty is
    the mean across that GW's fixtures - matching fetch.py's historical DGW handling.
    """
    teams_by_id = {
        t["id"]: config.TEAM_NAME_CORRECTIONS.get(t["name"], t["name"]) for t in bootstrap["teams"]
    }
    per_team = {}
    for fx in fetch_fixtures(gw):
        home, away = teams_by_id[fx["team_h"]], teams_by_id[fx["team_a"]]
        per_team.setdefault(home, []).append((away, True, fx.get("team_h_difficulty")))
        per_team.setdefault(away, []).append((home, False, fx.get("team_a_difficulty")))
    result = {}
    for team, fixtures in per_team.items():
        opp, is_home, _ = fixtures[0]
        fdrs = [f[2] for f in fixtures if f[2] is not None]
        result[team] = (opp, is_home, sum(fdrs) / len(fdrs) if fdrs else None)
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


def build_live_snapshot(raw_df):
    """One synthetic next-gameweek row per active player, with form features computed
    AS OF NOW - i.e. including each player's most recent played match.

    Why not just reuse each player's last played row (the old approach)? Training rows'
    features are shifted one gameweek so a row never sees its own outcome - which means a
    played row's features EXCLUDE the match played that week. Reused as "current form",
    they silently drop every player's freshest game. A synthetic future row has no outcome
    of its own to leak, so its shifted features legitimately include everything played.

    Active = played within the last GWS_PER_SEASON global gameweeks (roughly, appeared
    during the most recent season) - keeps long-gone players out of the optimizer's pool.
    """
    max_gw = int(raw_df["GW_global"].max())
    latest = raw_df.sort_values("GW_global").groupby("player_id", as_index=False).tail(1)
    active = latest[latest["GW_global"] > max_gw - config.GWS_PER_SEASON]
    future = active.copy()
    future["GW_global"] = max_gw + 1
    combined = pd.concat([raw_df, future], ignore_index=True)
    feat = features.build_feature_frame(combined)
    return feat[feat["GW_global"] == max_gw + 1].copy()


def build_future_predictions(snapshot, feature_cols, models, bootstrap, start_gw, horizon):
    """Predict every horizon gameweek from the live snapshot, with per-GW fixture info
    (opponent, home/away, official FDR) taken from the FPL API's fixture list - the
    fixture-difficulty features MUST be per-future-GW, not copied from the player's last
    played row, or the model scores next week's fixture with last week's difficulty."""
    # Prefetch two GWs past the horizon so fixture_difficulty_next3 has a full window.
    fixture_maps = {}
    for gw in range(start_gw, start_gw + horizon + 2):
        try:
            fixture_maps[gw] = build_team_fixture_map(bootstrap, gw)
        except requests.HTTPError:
            break

    rows = []
    for i in range(horizon):
        gw = start_gw + i
        if gw not in fixture_maps:
            print(f"No fixtures available yet for GW {gw}, stopping horizon there.")
            break
        fixture_map = fixture_maps[gw]
        gw_rows = snapshot.copy()
        gw_rows["GW"] = gw
        opponents, homes, fdrs, fdr3s = [], [], [], []
        for team in gw_rows["team"]:
            opp, is_home, fdr = fixture_map.get(team, (None, None, None))
            opponents.append(opp)
            homes.append(is_home)
            fdrs.append(fdr)
            upcoming = []
            for later_gw in range(gw, gw + 3):
                later_map = fixture_maps.get(later_gw, {})
                if team in later_map and later_map[team][2] is not None:
                    upcoming.append(later_map[team][2])
            fdr3s.append(sum(upcoming) / len(upcoming) if upcoming else fdr)
        gw_rows["opponent_team"] = opponents
        gw_rows["was_home"] = pd.Series(homes).fillna(False).astype(int).values
        gw_rows["fixture_difficulty"] = fdrs
        gw_rows["fixture_difficulty_next3"] = fdr3s
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
    raw = pd.read_csv(config.MASTER_DATASET_PATH, low_memory=False)
    feat_df = features.build_feature_frame(raw)
    feature_cols = features.feature_columns(feat_df)

    # Blend weights fit on the last 16 played GWs as a genuine holdout (members trained on
    # everything before it) - fresh every run, no dependence on stale saved weights.
    max_played_gw = int(feat_df["GW_global"].max())
    print(f"--- Fitting blend weights on holdout GW{max_played_gw - 15}-{max_played_gw} ---")
    weights_by_pos = fit_holdout_weights(feat_df, feature_cols, first_holdout_gw=max_played_gw + 1)
    models = {}
    for pos in POSITIONS:
        pos_df = feat_df[feat_df["position"] == pos]
        weights = weights_by_pos[pos]
        X, y = pos_df[feature_cols], pos_df[features.TARGET_COL]
        members = {name: model_registry.fit_model(name, X, y)
                   for name, wgt in weights.items() if wgt > 1e-6}
        models[pos] = PositionEnsemble(members, weights)

    bootstrap = fetch_bootstrap()
    target_gw = determine_target_gw(bootstrap)
    print(f"--- Target gameweek: {target_gw} ---")

    snapshot = build_live_snapshot(raw)
    preds = build_future_predictions(snapshot, feature_cols, models, bootstrap, target_gw, args.horizon)
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
