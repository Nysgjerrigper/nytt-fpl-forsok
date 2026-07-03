"""Walk-forward backtest of the Bayesian/MKP manager over real historical
gameweeks, mirroring fpl/model/predict.py's walk-forward discipline: the
belief model is only ever updated with data strictly before the gameweek
being decided, and the chosen lineup/captain is scored against that
gameweek's *actual* recorded results (goals_scored, assists, minutes,
clean sheets) - never the model's own simulated outcome.

Usage:
    python -m bayesian_manager.backtest --start-gw 77 --end-gw 107 --manager myopic
    python -m bayesian_manager.backtest --start-gw 77 --end-gw 107 --manager qlearning

See CLAUDE.md / README.md for what GW77-107 (2024-25 season, GW1-31) means
and why it's the reference window the production pipeline was validated on.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bayesian_manager import beliefs as beliefs_mod
from bayesian_manager import club_model, config, manager


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Backtest the Bayesian belief-state / MKP FPL manager")
    parser.add_argument("--start-gw", type=int, required=True, help="First GW_global to decide (must have GWs before it in the data)")
    parser.add_argument("--end-gw", type=int, required=True)
    parser.add_argument("--manager", choices=["myopic", "qlearning"], default="myopic")
    parser.add_argument("--n-samples", type=int, default=30, help="Monte Carlo samples per player for the myopic manager")
    parser.add_argument("--refit-every", type=int, default=1, help="Refit club strengths + exit-minute dists every N gameweeks (expensive to refit every week)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args(argv)


def _actual_points_for_lineup(gw_df, lineup_ids, captain_id):
    """Score a chosen lineup+captain against a gameweek's actual recorded
    total_points column - the ground truth the FPL site itself computed,
    same approach fpl/milp/optimize.py uses for its actual_total_points."""
    pts_by_id = dict(zip(gw_df["player_id"], gw_df["total_points"]))
    lineup_pts = sum(pts_by_id.get(i, 0) for i in lineup_ids)
    captain_pts = pts_by_id.get(captain_id, 0) if captain_id is not None else 0
    return lineup_pts + captain_pts, lineup_pts, captain_pts


def run(args):
    print(f"--- Loading {config.MASTER_DATASET_PATH} ---")
    df = pd.read_csv(config.MASTER_DATASET_PATH, low_memory=False)
    df["player_id"] = df["player_id"].astype(int)
    for col in ["team", "opponent_team", "position", "name"]:
        df[col] = df[col].astype(str).str.strip()

    rng = np.random.default_rng(args.seed)

    all_gws = sorted(df["GW_global"].unique())
    if args.start_gw not in all_gws:
        sys.exit(f"ERROR: start-gw {args.start_gw} not present in dataset.")

    player_beliefs = {}
    q_pool = []
    previous_squad, previous_budget, free_transfers = None, None, config.FREE_TRANSFERS_DEFAULT
    strengths, exit_dists, phi = None, None, None
    results = []

    for gw in range(args.start_gw, args.end_gw + 1):
        if gw not in all_gws:
            print(f"Skipping GW {gw}: not in dataset.")
            continue
        t0 = time.time()

        train_df = df[df["GW_global"] < gw]
        if train_df.empty:
            print(f"Skipping GW {gw}: no history before it to train beliefs on.")
            continue

        # Refit the shared, non-player-specific pieces (club strengths,
        # exit-minute distributions, phi) periodically rather than every
        # single gameweek - these change slowly and refitting the club
        # model's iterative-proportional-fit is the most expensive step here.
        if strengths is None or (gw - args.start_gw) % args.refit_every == 0:
            strengths = club_model.fit_club_strengths(train_df)
            exit_dists = beliefs_mod.fit_exit_minute_distributions(train_df)
            phi = beliefs_mod.compute_phi(train_df)

        # Rebuild beliefs from scratch each GW by replaying all prior
        # gameweeks. This is O(gw) per step rather than incremental, but
        # keeps the belief update logic in one place (beliefs.py) and is
        # fast enough for a season-length backtest (~150 gameweeks x a few
        # hundred rows each) - an incremental version would just apply
        # update_beliefs_from_gameweek once per new GW using saved state,
        # which is a drop-in optimisation if this is ever run over multiple
        # seasons at once.
        player_info_latest = beliefs_mod.build_player_info(df, gw)
        player_beliefs = beliefs_mod.initial_beliefs(player_info_latest)
        for hist_gw in sorted(train_df["GW_global"].unique()):
            beliefs_mod.update_beliefs_from_gameweek(player_beliefs, train_df[train_df["GW_global"] == hist_gw])

        gw_df = df[df["GW_global"] == gw].drop_duplicates("player_id", keep="first")
        if gw_df.empty:
            print(f"Skipping GW {gw}: empty gameweek slice.")
            continue

        fixtures = gw_df[["team", "opponent_team", "was_home"]].drop_duplicates()
        # Selectable pool P_i: any player beliefs.py knows about who actually
        # has a row this GW (so we know their fixture/team for this week).
        player_info = gw_df[["player_id", "name", "position", "team", "value"]].drop_duplicates("player_id")
        for pid in player_info["player_id"]:
            if pid not in player_beliefs:
                row = player_info[player_info["player_id"] == pid].iloc[0]
                player_beliefs[pid] = beliefs_mod.PlayerBelief(
                    player_id=pid, name=row["name"], position=row["position"],
                    team=row["team"], value=row["value"],
                )

        if args.manager == "myopic":
            team, exp_pts = manager.myopic_decision(
                player_info, player_beliefs, fixtures, strengths, exit_dists, phi, rng,
                previous_squad=previous_squad, previous_budget=previous_budget,
                free_transfers=free_transfers, n_samples=args.n_samples,
            )
        else:
            team, exp_pts, q_pool = manager.q_learning_decision(
                player_info, player_beliefs, fixtures, strengths, exit_dists, phi, rng,
                q_pool, previous_squad=previous_squad, previous_budget=previous_budget,
                free_transfers=free_transfers,
            )

        actual_total, actual_lineup, actual_captain = _actual_points_for_lineup(gw_df, team["lineup"], team["captain"])

        previous_squad = set(team["squad"])
        previous_budget = team["budget_remaining"]
        free_transfers = min(config.MAX_FREE_TRANSFERS, max(0, free_transfers - len(team["transfers_in"]) + 1)) if previous_squad else config.FREE_TRANSFERS_DEFAULT

        elapsed = time.time() - t0
        results.append({
            "gameweek": gw,
            "n_squad": len(team["squad"]),
            "n_transfers_in": len(team["transfers_in"]),
            "paid_transfers": team["paid_transfers"],
            "captain": team["captain"],
            "actual_total_points": actual_total,
            "actual_lineup_points": actual_lineup,
            "actual_captain_points": actual_captain,
            "solve_seconds": elapsed,
        })
        print(f"GW {gw}: actual_total_points={actual_total} (lineup={actual_lineup}, captain_bonus={actual_captain}), "
              f"transfers_in={len(team['transfers_in'])}, paid={team['paid_transfers']}, time={elapsed:.1f}s")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        output_path = args.output or str(
            config.OUTPUT_DIR / f"backtest_{args.manager}_GW{args.start_gw}-{args.end_gw}.csv"
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        total = results_df["actual_total_points"].sum()
        avg = results_df["actual_total_points"].mean()
        print(f"\nSaved results to {output_path}")
        print(f"--- Manager: {args.manager} | GW{args.start_gw}-{args.end_gw} ({len(results_df)} gameweeks) ---")
        print(f"Total actual points: {total:.1f}")
        print(f"Average actual points/GW: {avg:.1f}")
    else:
        print("No results generated.")
    return results_df


if __name__ == "__main__":
    run(parse_args())
