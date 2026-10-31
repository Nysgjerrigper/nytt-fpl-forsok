"""
Weekly driver: run this once a week during the season to get a squad/transfer/
captain recommendation for the next gameweek.

    python -m fpl.run_week                          # fresh squad-build advice (e.g. preseason/GW1)
    python -m fpl.run_week --team-id 1234567 --free-transfers 1
    python -m fpl.run_week --team-id 1234567 --free-transfers 1 --horizon 4

What it does, in order:
1. Refreshes the historical dataset (fpl/data/fetch.py) so this season's
   played gameweeks are included.
2. Retrains the GBM models on everything played so far.
3. Pulls the upcoming fixture(s) from the official FPL API (opponent/home-away
   per team - vaastav's historical data obviously has no rows yet for a
   gameweek that hasn't been played) and predicts points for every player
   over the next `--horizon` gameweeks, projecting each player's current form
   forward.
4. If `--team-id` is given, pulls your last-deadline squad/bank from the
   public FPL API and asks the MILP optimizer for the best transfers/lineup/
   captain/chip continuing from that squad. Without it, prints fresh-build
   advice as if drafting a squad from scratch.

Caveat: public picks do not establish current private holdings, free transfers,
selling prices or remaining chips. Continuing-team output is a scenario requiring
owner verification before entry. The legacy backend uses simplified sell prices.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from fpl import config, features
from fpl.data import fetch
from fpl.model.expert_policy import (
    fit_mid_gate_experts, parse_expert_map, predict_by_position, predict_with_mid_gate,
    resolve_weight_strategy,
)
from fpl.model.mid_gate import MidGateConfig
from fpl.model.train import fit_holdout_weights, fit_position_ensembles
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


def _event_deadlines(bootstrap: dict) -> dict[int, datetime]:
    """Read unique local rounds and timezone-aware official deadlines, failing closed."""
    deadlines = {}
    for event in bootstrap["events"]:
        gw = event["id"]
        if isinstance(gw, bool) or not isinstance(gw, int) or not 1 <= gw <= config.GWS_PER_SEASON:
            raise ValueError("Invalid gameweek identity in bootstrap")
        if gw in deadlines:
            raise ValueError("Duplicate gameweek identity in bootstrap")
        try:
            deadline = datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid deadline for GW{gw}") from exc
        if deadline.utcoffset() is None:
            raise ValueError(f"Timezone-aware deadline required for GW{gw}")
        deadlines[gw] = deadline
    return deadlines


def _utc_now(now: datetime | None) -> datetime:
    """Use an aware clock so machine locale cannot change deadline admission."""
    now = datetime.now(timezone.utc) if now is None else now
    if now.utcoffset() is None:
        raise ValueError("Timezone-aware current time required")
    return now


def determine_target_gw(bootstrap: dict, *, now: datetime | None = None) -> int:
    """Select an unfinished round whose official entry deadline is still strictly open."""
    now = _utc_now(now)
    deadlines = _event_deadlines(bootstrap)
    available = [e["id"] for e in bootstrap["events"]
                 if not e["finished"] and deadlines[e["id"]] > now]
    if not available:
        raise ValueError("No open gameweek deadline found in the official schedule")
    return min(available)


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


def get_user_squad(team_id: int, bootstrap: dict, known_player_ids: set,
                   *, now: datetime | None = None) -> tuple[list[int], float]:
    """Read last-deadline picks/bank, not private current holdings or free transfers.

    A complete fifteen-player identity mapping is mandatory. A player without forecast
    history blocks continuation rather than silently disappearing from owned assets.
    """
    if (not isinstance(bootstrap, dict) or not isinstance(bootstrap.get("events"), list)
            or any(not isinstance(event, dict) for event in bootstrap["events"])):
        raise ValueError("Invalid bootstrap events for owned-squad retrieval")
    now = _utc_now(now)
    published = [gw for gw, deadline in _event_deadlines(bootstrap).items() if deadline <= now]
    if not published:
        raise ValueError("No published deadline squad exists yet; provide owner state")
    last_published_gw = max(published)

    def positive_id(value: object) -> bool:
        return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value > 0

    elements = bootstrap.get("elements")
    if not isinstance(elements, list):
        raise ValueError("Missing bootstrap elements for owned-squad identity mapping")
    elements_by_id = {}
    stable_codes = set()
    for element in elements:
        if not isinstance(element, dict):
            raise ValueError("Invalid bootstrap element for owned-squad identity mapping")
        identifier, code = element.get("id"), element.get("code")
        if not positive_id(identifier) or not positive_id(code):
            raise ValueError(f"Invalid bootstrap element/code for owned-squad mapping: element={identifier!r}, code={code!r}; "
                             "refresh and verify the official bootstrap identity mapping")
        if identifier in elements_by_id or code in stable_codes:
            raise ValueError(f"Ambiguous bootstrap element/code for owned-squad mapping: element={identifier}, code={code}; "
                             "refresh and verify the official bootstrap identity mapping")
        elements_by_id[identifier] = element
        stable_codes.add(code)
    resp = requests.get(f"{FPL_API}/entry/{team_id}/event/{last_published_gw}/picks/", timeout=30)
    resp.raise_for_status()
    data = resp.json()

    picks = data.get("picks") if isinstance(data, dict) else None
    if not isinstance(picks, list) or len(picks) != 15:
        raise ValueError("Public squad must contain exactly 15 picks")
    squad_ids = []
    for pick in picks:
        identifier = pick.get("element") if isinstance(pick, dict) else None
        if not positive_id(identifier) or identifier not in elements_by_id:
            raise ValueError(f"Owned pick element ID {identifier!r} is missing or invalid in bootstrap")
        player_id = elements_by_id[identifier]["code"]
        if player_id not in known_player_ids:
            raise ValueError(f"Owned player {identifier} (code={player_id}) has no forecast history; cannot optimize the complete owned squad")
        squad_ids.append(player_id)

    if len(set(squad_ids)) != 15:
        raise ValueError("Public squad must contain exactly 15 unique player codes")

    entry_history = data.get("entry_history")
    bank = entry_history.get("bank") if isinstance(entry_history, dict) else None
    if isinstance(bank, bool) or not isinstance(bank, Integral) or bank < 0:
        raise ValueError("Public bank must be nonnegative integer tenths")
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


def apply_live_identity(snapshot, bootstrap):
    """Overlay current FPL club and position onto historical snapshot rows.

    Player form is intentionally computed from historical rows, but club and position are
    deadline-known identity fields.  A summer transfer or FPL position reclassification
    therefore must come from ``bootstrap-static`` before fixture expansion and model
    routing.  Players are joined by stable FPL ``code`` (the dataset ``player_id``);
    unmatched rows retain their historical identity and are subsequently removed by the
    registered-player filter.
    """
    teams_by_id = {
        team["id"]: config.TEAM_NAME_CORRECTIONS.get(team["name"], team["name"])
        for team in bootstrap["teams"]
    }
    identities = {
        element["code"]: (
            teams_by_id.get(element.get("team")),
            ELEMENT_TYPE_TO_POSITION.get(element.get("element_type")),
        )
        for element in bootstrap["elements"]
        if element.get("code") is not None
    }

    updated = snapshot.copy()
    live_team = updated["player_id"].map(
        {player_id: identity[0] for player_id, identity in identities.items()}
    )
    live_position = updated["player_id"].map(
        {player_id: identity[1] for player_id, identity in identities.items()}
    )
    team_changes = int((live_team.notna() & (live_team != updated["team"])).sum())
    position_changes = int(
        (live_position.notna() & (live_position != updated["position"])).sum()
    )
    updated["team"] = live_team.fillna(updated["team"])
    updated["position"] = live_position.fillna(updated["position"])
    print(
        f"Live identity: {team_changes} club and {position_changes} position "
        f"assignments updated from the API."
    )
    return updated


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

    For cold priors, availability is heuristically an independent Bernoulli gate:
    mean becomes q*mean, variance q*variance + q*(1-q)*mean**2, and participation
    probabilities become q*p. This is not a calibrated availability distribution.
    Separated from main() so the arithmetic is unit-testable without the API."""
    scale = preds["player_id"].map(factors).fillna(1.0)
    preds = preds.copy()
    original_mean = preds["predicted_total_points"].copy()
    if "prior_std" in preds:
        marked = preds.get("is_cold_start", pd.Series(False, index=preds.index)).fillna(False).astype(bool)
        if "forecast_source" in preds:
            marked = marked | preds["forecast_source"].eq("pooled_prior")
        prior_rows = marked & np.isfinite(preds["prior_std"])
        q = scale.loc[prior_rows]
        if not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
            raise ValueError("Cold-prior availability factors must be finite probabilities in [0, 1]")
        variance = preds.loc[prior_rows, "prior_std"] ** 2
        mean = original_mean.loc[prior_rows]
        preds.loc[prior_rows, "prior_std"] = np.sqrt(q * variance + q * (1 - q) * mean ** 2)
        for column in ("p_played", "p60"):
            if column in preds:
                preds.loc[prior_rows, column] = preds.loc[prior_rows, column] * q
    preds["predicted_total_points"] = original_mean * scale
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
                             opp_form=None, mid_gate=None, mid_experts=None, *, cold_start_priors=None):
    """Predict every horizon gameweek from the live snapshot, with per-GW fixture info
    (opponent, home/away, official FDR) taken from the FPL API's fixture list - the
    fixture-difficulty features MUST be per-future-GW, not copied from the player's last
    played row, or the model scores next week's fixture with last week's difficulty.

    `opp_form` (features.team_form_asof frame, indexed by team) supplies each UPCOMING
    opponent's trailing form for the opp_* features. Without it the snapshot's own opp_*
    values would be carried forward, and those describe the form of the player's PREVIOUS
    fixture's opponent - the same staleness class as the fixture-difficulty bug above
    (found 2026-07-11 while building the origin-based backtest, TODO 3.6).

    Optional pooled fixture priors require the snapshot global origin to match their
    historical cutoff. Their spread/frequencies are empirical, uncalibrated descriptors.
    Synthetic rows bypass the normal model path; the default path is unchanged.
    """
    from fpl.model.cold_start import cold_start_mask, predict_cold_starts, validate_pooled_priors

    snapshot_cold = cold_start_mask(snapshot)
    if snapshot_cold.any() and cold_start_priors is None:
        raise ValueError("Cold-start snapshot rows require explicit pooled priors")
    global_origin = None
    if cold_start_priors is not None:
        if "GW_global" not in snapshot or snapshot.empty:
            raise ValueError("Cold-start priors require one complete snapshot global origin")
        from fpl.model.cold_start import _numbers
        origins = _numbers(snapshot.GW_global, "snapshot global origin", integer=True)
        if origins.nunique() != 1:
            raise ValueError("Cold-start priors require one snapshot global origin")
        global_origin = int(origins.iloc[0])
        validate_pooled_priors(cold_start_priors, global_origin)
    # Prefetch two GWs past the horizon so fixture_difficulty_next3 has a full window.
    if horizon < 1 or start_gw < 1 or start_gw + horizon - 1 > config.GWS_PER_SEASON:
        raise ValueError("Live prediction horizon must fit within the active season")
    fixture_maps = {}
    for gw in range(start_gw, min(start_gw + horizon + 2, config.GWS_PER_SEASON + 1)):
        # An HTTP failure is not evidence of an empty schedule. Propagate it rather
        # than publishing a shorter horizon or silently changing next-three FDR.
        fixture_maps[gw] = build_team_fixture_map(bootstrap, gw)
    for gw in range(start_gw, start_gw + horizon):
        if not fixture_maps[gw]:
            raise ValueError(f"No fixtures in requested GW{gw}; whole-round blanks are not supported")

    rows = []
    for i in range(horizon):
        gw = start_gw + i
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

        if cold_start_priors is None:
            gw_rows["predicted_total_points"] = (
                predict_with_mid_gate(gw_rows, feature_cols, models, mid_gate, mid_experts)
                if mid_gate else predict_by_position(gw_rows, feature_cols, models)
            )
        else:
            marked = cold_start_mask(gw_rows)
            gw_rows["is_cold_start"] = marked
            gw_rows["forecast_source"] = np.where(marked, "pooled_position_team_prior", "model")
            gw_rows["predicted_total_points"] = np.nan
            for column in ("prior_std", "p_played", "p60", "prior_count"):
                gw_rows[column] = np.nan
            regular = gw_rows.loc[~marked]
            if not regular.empty:
                values = (predict_with_mid_gate(regular, feature_cols, models, mid_gate, mid_experts)
                          if mid_gate else predict_by_position(regular, feature_cols, models))
                gw_rows.loc[~marked, "predicted_total_points"] = np.asarray(values)
            if marked.any():
                values = predict_cold_starts(gw_rows.loc[marked], cold_start_priors, origin_gw=global_origin)
                gw_rows.loc[marked, values.columns] = values.to_numpy()
        rows.append(gw_rows)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Weekly FPL squad/transfer recommendation")
    parser.add_argument("--team-id", type=int, default=None,
                         help="Your public FPL team ID, to continue your real squad. "
                              "Omit for a fresh-build recommendation.")
    parser.add_argument("--horizon", type=int, default=3, help="Gameweeks to look ahead")
    parser.add_argument("--time-limit", type=float, default=120, help="Solver time limit per GW, seconds")
    parser.add_argument("--free-transfers", type=int, default=None,
                        help="Explicit current free transfers (0-5), required with --team-id")
    parser.add_argument("--optimizer", choices=("legacy", "solio"), default="legacy",
                        help="Solio is the independently checked modern-rule candidate; legacy remains the verified reference.")
    parser.add_argument("--optimizer-state-json", default=None,
                        help="Solio sell_prices (tenths), chip_inventory, forced_chips and previous_free_hit_gw.")
    parser.add_argument(
        "--expert-map", type=parse_expert_map, default=None,
        help="Experimental complete position override, e.g. "
             "GK=catboost,DEF=lightgbm,MID=tabm,FWD=catboost. "
             "Production remains single:catboost when omitted.",
    )
    parser.add_argument("--mid-gate-config", default=None,
                        help="Frozen MidGateConfig JSON for optional experimental MID routing.")
    parser.add_argument("--cold-start-priors", action="store_true",
                        help="Opt-in empirical pooled priors for registered players without history; not calibrated intervals.")
    args = parser.parse_args()
    if args.optimizer_state_json and args.optimizer != "solio":
        parser.error("optimizer-state-json requires --optimizer solio")
    if args.horizon < 1:
        parser.error("horizon must be positive")
    if args.team_id is not None and (args.team_id <= 0 or args.free_transfers is None):
        parser.error("a positive team-id requires explicit --free-transfers; public picks do not expose it")
    if args.free_transfers is not None and not 0 <= args.free_transfers <= 5:
        parser.error("free-transfers must be between 0 and 5")

    # Admit the deadline before mutating data or fitting models. An in-progress round
    # is already closed for entry even when its outcomes are not finished/checked.
    bootstrap = fetch_bootstrap()
    target_gw = determine_target_gw(bootstrap)
    if target_gw + args.horizon - 1 > max(_event_deadlines(bootstrap)):
        parser.error("requested horizon extends beyond the published season")
    print(f"--- Target gameweek: {target_gw} ---")

    print("--- Refreshing historical dataset ---")
    fetch.build_master_dataset()

    print("--- Building features & training models on all data so far ---")
    raw = pd.read_csv(config.MASTER_DATASET_PATH, low_memory=False)
    feat_df = features.build_feature_frame(raw)
    feature_cols = features.feature_columns(feat_df)
    cold_start_priors = None
    cold_snapshot = None
    if args.cold_start_priors:
        from fpl.model.cold_start import add_registered_cold_starts, fit_pooled_priors
        cold_snapshot = apply_live_identity(build_live_snapshot(raw), bootstrap)
        origin = config.season_start_gw(fetch.active_season_from_bootstrap(bootstrap)) + target_gw - 1
        cold_snapshot["GW_global"] = origin
        cold_snapshot = add_registered_cold_starts(cold_snapshot, bootstrap)
        cold_start_priors = fit_pooled_priors(raw, origin_gw=origin)
    squad_ids, bank = None, 1000.0
    if args.team_id is not None:
        squad_ids, bank = get_user_squad(args.team_id, bootstrap, set(cold_snapshot["player_id"] if args.cold_start_priors else feat_df["player_id"]))
        print("OWNER VERIFICATION REQUIRED: public last-deadline picks/bank are a scenario; "
              "check current holdings, bank, selling prices and remaining chips before entry.")

    # Same production configuration the backtests validate (config.PRODUCTION_WEIGHT_STRATEGY),
    # fit through the same code path (train.fit_position_ensembles, position-aware so tuned
    # per-position params load). Under a blend strategy the weights come from the last 16
    # played GWs as a genuine holdout - fresh every run, no stale saved weights; under
    # single:<model> no weight fitting is needed at all.
    max_played_gw = int(feat_df["GW_global"].max())
    effective_strategy = resolve_weight_strategy(
        config.PRODUCTION_WEIGHT_STRATEGY, args.expert_map
    )
    label = "experimental expert map" if args.expert_map is not None else "production strategy"
    print(f"--- Fitting models ({label}={effective_strategy}) ---")
    weights_by_pos = fit_holdout_weights(feat_df, feature_cols, first_holdout_gw=max_played_gw + 1,
                                         strategy=effective_strategy)
    models = fit_position_ensembles(feat_df, feature_cols, weights_by_pos)
    mid_gate = (MidGateConfig.from_dict(json.loads(Path(args.mid_gate_config).read_text()))
                if args.mid_gate_config else None)
    mid_experts = fit_mid_gate_experts(feat_df, feature_cols, mid_gate) if mid_gate else None

    snapshot = cold_snapshot if args.cold_start_priors else apply_live_identity(build_live_snapshot(raw), bootstrap)
    opp_form = features.team_form_asof(raw)
    cold_kwargs = {"cold_start_priors": cold_start_priors} if args.cold_start_priors else {}
    preds = build_future_predictions(snapshot, feature_cols, models, bootstrap, target_gw, args.horizon,
                                     opp_form=opp_form, mid_gate=mid_gate, mid_experts=mid_experts, **cold_kwargs)
    if preds.empty:
        sys.exit("Could not build any predictions - fixtures for the target gameweek aren't published yet.")

    preds = filter_to_registered(preds, bootstrap)
    if preds.empty:
        sys.exit("No registered players left after filtering - API/dataset identity mismatch?")
    preds = apply_availability(preds, availability_multipliers(bootstrap))
    preds = apply_live_prices(preds, live_prices(bootstrap))
    if set(preds["GW"]) != set(range(target_gw, target_gw + args.horizon)):
        raise ValueError("Live forecast does not cover every requested gameweek")
    if squad_ids is not None:
        optimize.validate_owned_forecast_pool(squad_ids, list(preds["player_id"]))
    if determine_target_gw(bootstrap) != target_gw:
        raise ValueError("Target deadline passed during preparation; restart for the next open round")

    out_cols = ["player_id", "GW", "name", "position", "team", "value", "predicted_total_points"]
    out_cols.extend(col for col in ("opponent_team", "was_home") if col in preds)
    if args.cold_start_priors:
        out_cols.extend(["is_cold_start", "forecast_source", "prior_std", "p_played", "p60", "prior_count"])
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

    if args.team_id is not None:
        print(f"--- Continuing team {args.team_id}: {len(squad_ids)} matched players, bank={bank} ---")
        opt_args += [
            "--initial-squad", ",".join(map(str, squad_ids)),
            "--initial-budget", str(bank),
            "--initial-ft", str(args.free_transfers),
        ]
    else:
        print("--- No --team-id given: producing a fresh-build recommendation (full budget, no existing squad) ---")

    if args.optimizer == "solio":
        from fpl.milp.solio import run_schedule
        state = json.loads(Path(args.optimizer_state_json).read_text()) if args.optimizer_state_json else {}
        if "sell_prices" in state:
            state["sell_prices"] = {int(player): value for player, value in state["sell_prices"].items()}
        season = fetch.active_season_from_bootstrap(bootstrap)
        offset = config.season_start_gw(season) - 1
        global_preds = preds.assign(GW=preds["GW"] + offset)
        result = run_schedule(global_preds, season=season,
                              start_gw=target_gw + offset, max_gw=int(global_preds.GW.max()), horizon=args.horizon,
                              initial_squad=squad_ids, bank=bank,
                              free_transfers=args.free_transfers if squad_ids is not None else 0,
                              bootstrap=bootstrap, time_limit=args.time_limit, **state)
        output = config.SQUAD_OUTPUT_DIR / "solio_weekly.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(result[["gameweek", "captain", "transfers_in", "transfers_out", "chip_played", "budget_end"]].to_string(index=False))
        print(f"Saved independently checked Solio decisions to {output}")
    else:
        optimize.run(optimize.parse_args(opt_args))


if __name__ == "__main__":
    main()
