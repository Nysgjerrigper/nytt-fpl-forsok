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
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl import config, features
from fpl.data import fetch
from fpl.model.train import POSITIONS, fit_holdout_weights, fit_position_ensembles
from fpl.milp import optimize

FPL_API = "https://fantasy.premierleague.com/api"
ELEMENT_TYPE_TO_POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# The dataset's player_id IS the FPL `code` (fetch.assign_player_ids, TODO 4.8), and
# bootstrap-static exposes the same `code` per element - so live API players map to
# dataset players by a direct id equality, replacing the old normalized-name matching
# that silently dropped players whose spelling differed between the two sources.


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
    """team_name -> [(opponent_name, was_home, fixture_difficulty), ...] for one gameweek.

    Team names go through config.TEAM_NAME_CORRECTIONS so they match the dataset's
    corrected `team` column (the API says "Spurs"/"Man Utd", the dataset says
    "Tottenham"/"Man United" - without this, lookups for those teams silently miss
    and their players get dropped from every live prediction).

    Returns EVERY fixture of the round per team (TODO 3.3): a double-gameweek team gets
    two entries, and build_future_predictions emits one prediction row per fixture -
    matching how historical backtest predictions represent DGWs (per-fixture rows that
    fpl.milp.optimize sums per (player, GW)). The previous first-fixture-only behaviour
    made live mode value a DGW player at roughly half his backtest-equivalent points,
    exactly the players the optimizer should be hunting.
    """
    teams_by_id = {
        t["id"]: config.TEAM_NAME_CORRECTIONS.get(t["name"], t["name"]) for t in bootstrap["teams"]
    }
    per_team = {}
    for fx in fetch_fixtures(gw):
        home, away = teams_by_id[fx["team_h"]], teams_by_id[fx["team_a"]]
        per_team.setdefault(home, []).append((away, True, fx.get("team_h_difficulty")))
        per_team.setdefault(away, []).append((home, False, fx.get("team_a_difficulty")))
    return per_team


def get_user_squad(team_id, bootstrap, known_player_ids):
    """Public FPL API: current squad, bank and free transfers for an existing team.

    Squad players are matched to dataset player_ids by FPL `code` (== player_id).
    A pick whose code is absent from `known_player_ids` (no appearance in the dataset,
    e.g. a brand-new signing with zero PL history) is excluded with a warning - the
    optimizer cannot reason about a player it has no rows for.
    """
    last_finished_gw = max(e["id"] for e in bootstrap["events"] if e["finished"])
    resp = requests.get(f"{FPL_API}/entry/{team_id}/event/{last_finished_gw}/picks/", timeout=30)
    resp.raise_for_status()
    data = resp.json()

    elements_by_id = {el["id"]: el for el in bootstrap["elements"]}
    squad_ids = []
    for pick in data["picks"]:
        el = elements_by_id[pick["element"]]
        player_id = el.get("code")
        if player_id not in known_player_ids:
            print(f"WARNING: FPL player '{el['first_name']} {el['second_name']}' (code={player_id}) "
                  f"has no rows in the dataset - excluded from optimization.")
            continue
        squad_ids.append(player_id)

    bank = data["entry_history"]["bank"]  # in 0.1m units, matches `value` column
    return squad_ids, float(bank)


def filter_to_registered(preds, bootstrap):
    """Drop prediction rows for players not registered in the current season's API.

    Found in the first real 2026-27 rehearsal (2026-07-23): the live snapshot keeps
    every player active in the trailing season of the DATASET, but between seasons many
    of those leave the league entirely (transfers abroad, retirements, relegated-club
    departures) - 264 of 710 pool players, including that summer's biggest outbound
    transfers, were not purchasable at all, and availability scaling can't catch them
    because the API has no element (hence no status) for a player it doesn't list.
    bootstrap-static's element list IS the purchasable universe, so membership in it
    (by `code` == player_id) is the filter.
    """
    registered = {el["code"] for el in bootstrap["elements"] if el.get("code") is not None}
    kept = preds[preds["player_id"].isin(registered)].copy()
    print(f"Registered-player filter: {len(preds) - len(kept)} of {len(preds)} prediction rows "
          f"dropped (players not in this season's FPL register).")
    return kept


def availability_multipliers(bootstrap):
    """Per-player availability scaling from the FPL API's own team-news fields (TODO 3.1).

    The historical dataset carries no availability signal, so without this the optimizer
    happily buys a player whose great trailing form ended in a two-month injury - form
    features cannot see a knock announced yesterday. bootstrap-static's `status` and
    `chance_of_playing_next_round` are FPL's official pre-deadline team news:

    - status 'a' (available): factor 1.0.
    - status 'd' (doubtful): chance_of_playing_next_round / 100, defaulting to 0.75 when
      FPL hasn't quantified the doubt (their UI shows unquantified doubts as 75%).
    - status 'i'/'s'/'u'/'n' (injured / suspended / unavailable / not in squad):
      chance / 100, defaulting to 0.0 - these players will not play next round.

    Returns {player_id: factor} keyed by FPL `code` (== dataset player_id); codes with no
    dataset rows simply match nothing downstream and default to 1.0. The factor is applied
    to EVERY horizon gameweek, not just the next one: `chance_of_playing_next_round`
    strictly describes the next round, but scaling later rounds too errs on the side of not
    planning transfers around a player who is out today (re-assessed on every run).
    """
    factors = {}
    for el in bootstrap["elements"]:
        player_id = el.get("code")
        if player_id is None:
            continue
        status, chance = el.get("status", "a"), el.get("chance_of_playing_next_round")
        if status == "a":
            factor = 1.0
        elif status == "d":
            factor = (chance if chance is not None else 75.0) / 100.0
        else:
            factor = (chance if chance is not None else 0.0) / 100.0
        factors[player_id] = factor
    return factors


def apply_availability(preds, factors):
    """Scale predicted_total_points by each player's availability factor (1.0 when unknown).

    Separated from main() so the scaling arithmetic is unit-testable without the API."""
    scale = preds["player_id"].map(factors).fillna(1.0)
    preds = preds.copy()
    preds["predicted_total_points"] = preds["predicted_total_points"] * scale
    flagged, zeroed = int((scale < 1.0).sum()), int((scale == 0.0).sum())
    print(f"Availability scaling: {flagged} of {len(preds)} prediction rows scaled down ({zeroed} zeroed).")
    return preds


def live_prices(bootstrap):
    """Per-player CURRENT price from bootstrap-static `now_cost` (TODO 3.2, audit A4a).

    The snapshot's `value` column is whatever price the player had on his last played
    row in vaastav's dump - potentially weeks stale for rotation players and always
    stale across a price change, so the optimizer's budget math drifts from what the
    FPL site will actually charge at the deadline. `now_cost` is in the same 0.1m
    units as the dataset's `value` (and get_user_squad's bank), so it overrides
    directly. Returns {player_id: now_cost} keyed by FPL `code` (== dataset player_id);
    codes with no dataset rows match nothing and those players keep their snapshot price.

    Known simplification (documented, TODO 3.4): this is the BUY price for everyone;
    a real squad's sell prices can differ (FPL's 50% sell-on rule). The bank figure
    from the API absorbs most of the discrepancy for continue-mode runs.
    """
    prices = {}
    for el in bootstrap["elements"]:
        player_id = el.get("code")
        if player_id is not None and el.get("now_cost") is not None:
            prices[player_id] = float(el["now_cost"])
    return prices


def apply_live_prices(preds, prices):
    """Override the stale snapshot `value` with current API prices (snapshot price when
    unknown). Separated from main() so the override is unit-testable without the API."""
    preds = preds.copy()
    live = preds["player_id"].map(prices)
    changed = int((live.notna() & (live != preds["value"])).sum())
    preds["value"] = live.fillna(preds["value"])
    print(f"Live prices: {changed} of {len(preds)} prediction rows re-priced from the API.")
    return preds


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


def build_future_predictions(snapshot, feature_cols, models, bootstrap, start_gw, horizon,
                             opp_form=None):
    """Predict every horizon gameweek from the live snapshot, with per-GW fixture info
    (opponent, home/away, official FDR) taken from the FPL API's fixture list - the
    fixture-difficulty features MUST be per-future-GW, not copied from the player's last
    played row, or the model scores next week's fixture with last week's difficulty.

    `opp_form` (features.team_form_asof frame, indexed by team) supplies each UPCOMING
    opponent's trailing form for the opp_* features. Without it the snapshot's own opp_*
    values would be carried forward, and those describe the form of the player's PREVIOUS
    fixture's opponent - the same staleness class as the fixture-difficulty bug above
    (found 2026-07-11 while building the origin-based backtest, TODO 3.6)."""
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
        # One row PER FIXTURE (not per team): a DGW team appears twice in fixture_df, and
        # the inner merge duplicates its players' snapshot rows accordingly - the optimizer
        # sums per (player, GW), so a DGW player is worth both fixtures' predicted points.
        # Teams without a fixture this GW (blanks) simply don't match and drop out, same
        # as before.
        fixture_df = pd.DataFrame([
            {"team": team, "opponent_team": opp, "was_home": int(bool(is_home)),
             "fixture_difficulty": fdr}
            for team, fixtures in fixture_map.items()
            for opp, is_home, fdr in fixtures
        ])
        # Trailing-3-GW mean difficulty per TEAM (flattened across DGW fixtures), shared
        # by all of that team's rows this GW.
        fdr3_by_team = {}
        for team in fixture_map:
            upcoming = [fdr for later_gw in range(gw, gw + 3)
                        for _, _, fdr in fixture_maps.get(later_gw, {}).get(team, [])
                        if fdr is not None]
            fdr3_by_team[team] = sum(upcoming) / len(upcoming) if upcoming else None
        gw_rows = snapshot.drop(
            columns=[c for c in ("opponent_team", "was_home", "fixture_difficulty",
                                 "fixture_difficulty_next3") if c in snapshot.columns]
        ).merge(fixture_df, on="team", how="inner")
        gw_rows["GW"] = gw
        gw_rows["fixture_difficulty_next3"] = (
            gw_rows["team"].map(fdr3_by_team).fillna(gw_rows["fixture_difficulty"])
        )
        if opp_form is not None:
            for col in features.OPPONENT_FEATURES:
                gw_rows[col] = gw_rows["opponent_team"].map(opp_form[col]).to_numpy()

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

    # Same production configuration the backtests validate (config.PRODUCTION_WEIGHT_STRATEGY),
    # fit through the same code path (train.fit_position_ensembles, position-aware so tuned
    # per-position params load). Under a blend strategy the weights come from the last 16
    # played GWs as a genuine holdout - fresh every run, no stale saved weights; under
    # single:<model> no weight fitting is needed at all.
    max_played_gw = int(feat_df["GW_global"].max())
    print(f"--- Fitting production models (strategy={config.PRODUCTION_WEIGHT_STRATEGY}) ---")
    weights_by_pos = fit_holdout_weights(feat_df, feature_cols, first_holdout_gw=max_played_gw + 1,
                                         strategy=config.PRODUCTION_WEIGHT_STRATEGY)
    models = fit_position_ensembles(feat_df, feature_cols, weights_by_pos)

    bootstrap = fetch_bootstrap()
    target_gw = determine_target_gw(bootstrap)
    print(f"--- Target gameweek: {target_gw} ---")

    snapshot = build_live_snapshot(raw)
    opp_form = features.team_form_asof(raw)
    preds = build_future_predictions(snapshot, feature_cols, models, bootstrap, target_gw, args.horizon,
                                     opp_form=opp_form)
    if preds.empty:
        sys.exit("Could not build any predictions - fixtures for the target gameweek aren't published yet.")

    preds = filter_to_registered(preds, bootstrap)
    if preds.empty:
        sys.exit("No registered players left after filtering - API/dataset identity mismatch?")
    preds = apply_availability(preds, availability_multipliers(bootstrap))
    preds = apply_live_prices(preds, live_prices(bootstrap))

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
        squad_ids, bank = get_user_squad(args.team_id, bootstrap, set(feat_df["player_id"]))
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
