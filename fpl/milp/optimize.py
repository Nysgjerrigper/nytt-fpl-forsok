"""
Consolidated MILP squad-selection optimizer.

Originally replaced 8 near-identical scripts this project used to have
(MILP.py, MILP-GC.py, AUTO-MILP-GC*.py, t-auto*.py - since deleted, see git
history) with one parameterized version. The optimization model itself
(budget, formation, captaincy, transfers, chip logic) is unchanged from that
original t-auto.py - that formulation, based on Kristiansen et al., was
already correct (see the `actual_total_points` fix in git history). What
changed is that every hardcoded path/gameweek-range/chip-target is now a CLI
argument, so one script covers live weekly runs, backtests against actual
results, and sweeps over sub-horizon lengths, instead of needing a different
.py/.bat file per scenario.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd
import pulp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fpl import config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="FPL squad selection via MILP (rolling horizon)")
    parser.add_argument("--predictions-csv", type=str, default=str(config.PREDICTIONS_PATH),
                         help="CSV with columns: player_id, GW, name, position, team, value, "
                              "and the points column to optimize against.")
    parser.add_argument("--points-col", type=str, default="predicted_total_points",
                         help="Column to optimize against. Use 'actual_total_points' for "
                              "hindsight backtests (legacy t-auto-actual.py behaviour).")
    parser.add_argument("--start-gw", type=int, required=True)
    parser.add_argument("--max-gw", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=3, help="Sub-horizon length (weeks to look ahead)")
    parser.add_argument("--time-limit", type=float, default=None, help="Solver time limit per GW, seconds")
    parser.add_argument("--solver", type=str, default=config.MILP_SOLVER, choices=["cbc", "highs"],
                         help="MILP solver backend. 'highs' requires the highspy package.")
    parser.add_argument("--threads", type=int, default=config.MILP_THREADS,
                         help="Solver threads (0 = solver default/single-threaded).")
    parser.add_argument("--gap-rel", type=float, default=config.MILP_GAP_REL,
                         help="Relative MIP optimality gap (0 = prove full optimality). "
                              "A small gap (e.g. 0.001) trades a bounded objective loss for speed.")
    parser.add_argument("--wc1-gw", type=int, default=0, help="Force wildcard 1 at this absolute GW (0=disabled)")
    parser.add_argument("--wc2-gw", type=int, default=0, help="Force wildcard 2 at this absolute GW (0=disabled)")
    parser.add_argument("--tc-gw", type=int, default=0, help="Force triple captain at this absolute GW (0=disabled)")
    parser.add_argument("--fh-gw", type=int, default=0, help="Force free hit at this absolute GW (0=disabled)")
    parser.add_argument("--bb-gw", type=int, default=0, help="Force bench boost at this absolute GW (0=disabled)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: auto-named)")
    parser.add_argument("--initial-squad", type=str, default=None,
                         help="Comma-separated player_ids of an existing squad to continue from "
                              "(for live weekly use). Omit to start fresh with a full budget, as in backtests.")
    parser.add_argument("--initial-budget", type=float, default=None,
                         help="Bank balance (in the 0.1m units FPL uses) when continuing an existing squad.")
    parser.add_argument("--initial-ft", type=int, default=1, help="Free transfers available when continuing an existing squad.")
    return parser.parse_args(argv)


def make_solver(name: str, time_limit: float | None, threads: int = 0,
                gap_rel: float = 0.0) -> pulp.LpSolver:
    """Build the PuLP solver backend by name ('cbc' or 'highs').

    With gap_rel=0 both backends return proven-optimal solutions, so squad
    output is identical up to objective ties; they differ only in speed (see
    config.MILP_SOLVER for the benchmark numbers behind the default).
    """
    if name == "highs":
        kwargs = {"msg": False, "timeLimit": time_limit}
        if threads:
            kwargs["threads"] = threads
        if gap_rel:
            kwargs["gapRel"] = gap_rel
        solver = pulp.HiGHS(**kwargs)
        if not solver.available():
            sys.exit("ERROR: --solver highs requires the highspy package (pip install highspy).")
        return solver
    return pulp.PULP_CBC_CMD(msg=True, timeLimit=time_limit, threads=threads or None,
                             gapRel=gap_rel or None)


def run(args):
    solver = make_solver(args.solver, args.time_limit, args.threads, args.gap_rel)

    print(f"--- Loading predictions from {args.predictions_csv} ---")
    allesesonger = pd.read_csv(args.predictions_csv)
    essential_input_cols = ["player_id", "GW", "name", "position", "team", args.points_col, "value"]
    missing_cols = [c for c in essential_input_cols if c not in allesesonger.columns]
    if missing_cols:
        sys.exit(f"ERROR: Missing essential columns in {args.predictions_csv}: {missing_cols}")

    allesesonger["GW"] = pd.to_numeric(allesesonger["GW"])
    allesesonger["value"] = pd.to_numeric(allesesonger["value"]).fillna(50.0)

    # --- Double gameweek aggregation ---
    sum_cols = [c for c in [args.points_col, "actual_total_points"] if c in allesesonger.columns]
    first_cols = ["name", "position", "team", "value"]
    agg = {c: "sum" for c in sum_cols}
    agg.update({c: "first" for c in first_cols if c in allesesonger.columns})
    allesesonger = allesesonger.groupby(["player_id", "GW"], as_index=False).agg(agg)

    data_load_start_gw = args.start_gw - 1 if args.start_gw > 1 else args.start_gw
    data_full_range_raw = allesesonger[
        (allesesonger["GW"] >= data_load_start_gw) & (allesesonger["GW"] <= args.max_gw)
    ].copy()
    if data_full_range_raw.empty:
        sys.exit(f"ERROR: No data found for GW range {data_load_start_gw}-{args.max_gw}.")
    for col in ["team", "position", "name"]:
        data_full_range_raw[col] = data_full_range_raw[col].astype(str).str.strip()

    T_setofgameweeks_full = sorted(data_full_range_raw["GW"].unique())
    p = sorted(data_full_range_raw["player_id"].unique())
    final_player_info = data_full_range_raw.drop_duplicates(subset=["player_id"], keep="first")
    player_name_map = final_player_info.set_index("player_id")["name"].to_dict()
    pos_map = pd.Series(final_player_info["position"].values, index=final_player_info["player_id"])
    Pgk = sorted([p_ for p_ in p if pos_map.get(p_) == "GK"])
    Pdef = sorted([p_ for p_ in p if pos_map.get(p_) == "DEF"])
    Pmid = sorted([p_ for p_ in p if pos_map.get(p_) == "MID"])
    Pfwd = sorted([p_ for p_ in p if pos_map.get(p_) == "FWD"])
    P_not_gk = sorted([p_ for p_ in p if p_ not in Pgk])
    P_c = data_full_range_raw.groupby("team")["player_id"].unique().apply(list).to_dict()
    C_setofteams = sorted(P_c.keys())
    l = list(range(1, 4))
    print(f"Sets: {len(T_setofgameweeks_full)} GWs, {len(p)} players, {len(C_setofteams)} teams "
          f"({len(Pgk)} GK, {len(Pdef)} DEF, {len(Pmid)} MID, {len(Pfwd)} FWD)")

    # Parameters (Kristiansen et al. formulation)
    R_penalty, MK, MD, MM, MF, MC, E, EK = 4, 2, 5, 5, 3, 3, 11, 1
    ED, EM, EF, BS = 3, 2, 1, 1000.0
    phi = (MK + MD + MM + MF) - E
    phi_K = MK - EK
    Q_bar, Q_under_bar = 2, 1
    epsilon = 0.1
    kappa = {1: 0.01, 2: 0.005, 3: 0.001}
    M_transfer = MK + MD + MM + MF
    M_budget = BS + M_transfer * 200
    M_alpha = M_transfer + Q_bar
    M_q = Q_bar + 1

    points_matrix_df = data_full_range_raw.pivot(index="player_id", columns="GW", values=args.points_col)
    value_matrix_df = data_full_range_raw.pivot(index="player_id", columns="GW", values="value")
    points_matrix_df = points_matrix_df.reindex(index=p, columns=T_setofgameweeks_full, fill_value=0.0).fillna(0.0)
    value_matrix_df = value_matrix_df.reindex(index=p, columns=T_setofgameweeks_full)
    for player_id in p:
        value_matrix_df.loc[player_id] = value_matrix_df.loc[player_id].ffill().bfill()
    value_matrix_df.fillna(50.0, inplace=True)

    season_num = math.ceil(args.start_gw / config.GWS_PER_SEASON)
    gw1_of_this_season = (season_num - 1) * config.GWS_PER_SEASON + 1
    mid_season_split_gw = gw1_of_this_season + 19 - 1
    print(f"Season {season_num} halves: FH <= {mid_season_split_gw}, SH > {mid_season_split_gw}")

    continuing_squad = bool(args.initial_squad)
    master_results = []
    if continuing_squad:
        initial_ids = [int(pid) for pid in args.initial_squad.split(",")]
        previous_squad_dict = {pid: 1 for pid in initial_ids}
        if args.initial_budget is None:
            sys.exit("ERROR: --initial-budget is required when --initial-squad is given.")
        previous_budget = args.initial_budget
        previous_ft = args.initial_ft
        print(f"Continuing existing squad of {len(initial_ids)} players, budget={previous_budget}, FT={previous_ft}.")
    else:
        previous_squad_dict = {}
        previous_budget = BS
        previous_ft = 1
    used_chips_tracker = {"wc1": False, "wc2": False, "bb": False, "tc": False, "fh": False}
    # 0 means "disabled" (never matches a real gameweek, so the chip is always forced off);
    # a positive value forces that chip at exactly that absolute gameweek.
    chip_targets = {
        "wc1": args.wc1_gw, "wc2": args.wc2_gw,
        "tc": args.tc_gw, "fh": args.fh_gw, "bb": args.bb_gw,
    }

    for current_gw in range(args.start_gw, args.max_gw + 1):
        if current_gw not in T_setofgameweeks_full:
            print(f"Skipping GW {current_gw}: not in loaded data range.")
            continue

        print(f"\n{'=' * 15} Solving for Gameweek {current_gw} {'=' * 15}")
        loop_start = time.time()

        t_sub = sorted([gw for gw in T_setofgameweeks_full if current_gw <= gw < current_gw + args.horizon])
        if not t_sub:
            print(f"Sub-horizon empty for GW {current_gw}.")
            break
        t1_sub = t_sub[0]

        model = pulp.LpProblem(f"FPL_Opt_GW{current_gw}_Sub{args.horizon}", pulp.LpMaximize)

        x = pulp.LpVariable.dicts("Squad", (p, t_sub), cat="Binary")
        x_freehit = pulp.LpVariable.dicts("Squad_FH", (p, t_sub), cat="Binary")
        y = pulp.LpVariable.dicts("Lineup", (p, t_sub), cat="Binary")
        f = pulp.LpVariable.dicts("Captain", (p, t_sub), cat="Binary")
        h = pulp.LpVariable.dicts("ViceCaptain", (p, t_sub), cat="Binary")
        is_tc = pulp.LpVariable.dicts("TripleCaptainChipActive", (p, t_sub), cat="Binary")
        u = pulp.LpVariable.dicts("TransferOut", (p, t_sub), cat="Binary")
        e = pulp.LpVariable.dicts("TransferIn", (p, t_sub), cat="Binary")
        lambda_var = pulp.LpVariable.dicts("Aux_LineupInSquad", (p, t_sub), cat="Binary")
        g = {}
        if P_not_gk and l:
            g = pulp.LpVariable.dicts("Substitution", (P_not_gk, t_sub, l), cat="Binary")
        w = pulp.LpVariable.dicts("WildcardChipActive", t_sub, cat="Binary")
        b = pulp.LpVariable.dicts("BenchBoostChipActive", t_sub, cat="Binary")
        r = pulp.LpVariable.dicts("FreeHitChipActive", t_sub, cat="Binary")
        v = pulp.LpVariable.dicts("RemainingBudget", t_sub, lowBound=0, cat="Continuous")
        q = pulp.LpVariable.dicts("FreeTransfersAvailable", t_sub, lowBound=0, upBound=Q_bar, cat="Integer")
        alpha = pulp.LpVariable.dicts("PenalizedTransfers", t_sub, lowBound=0, upBound=M_alpha, cat="Integer")
        ft_carry = pulp.LpVariable.dicts("FT_Carry", t_sub, lowBound=0)

        points_sub = points_matrix_df.loc[p, t_sub]
        value_sub = value_matrix_df.loc[p, t_sub]

        points_from_lineup = pulp.lpSum(points_sub.loc[p_, t_] * y[p_][t_] for p_ in p for t_ in t_sub)
        points_from_captain = pulp.lpSum(points_sub.loc[p_, t_] * f[p_][t_] for p_ in p for t_ in t_sub)
        points_from_vice = pulp.lpSum(epsilon * points_sub.loc[p_, t_] * h[p_][t_] for p_ in p for t_ in t_sub)
        points_from_tc = pulp.lpSum(2 * points_sub.loc[p_, t_] * is_tc[p_][t_] for p_ in p for t_ in t_sub)
        points_from_subs = 0
        if g:
            points_from_subs = pulp.lpSum(
                kappa[l_] * points_sub.loc[p_ngk][t_] * g[p_ngk][t_][l_]
                for p_ngk in P_not_gk for t_ in t_sub for l_ in l
            )
        transfer_penalty = pulp.lpSum(R_penalty * alpha[t_] for t_ in t_sub)
        model += (points_from_lineup + points_from_captain + points_from_vice +
                  points_from_tc + points_from_subs - transfer_penalty), "Total_Expected_Points_Sub"

        # --- Chips (with timing restrictions) ---
        wc1_available = not used_chips_tracker["wc1"]
        wc2_available = not used_chips_tracker["wc2"]
        tc_available = not used_chips_tracker["tc"]
        bb_available = not used_chips_tracker["bb"]
        fh_available = not used_chips_tracker["fh"]
        for t_ in t_sub:
            is_fh_half = t_ <= mid_season_split_gw
            if is_fh_half:
                if not wc1_available or t_ != chip_targets["wc1"]:
                    model += w[t_] == 0
            else:
                model += w[t_] == 0
            is_sh_half = t_ > mid_season_split_gw
            if is_sh_half:
                if not wc2_available or t_ != chip_targets["wc2"]:
                    model += w[t_] == 0
            else:
                model += w[t_] == 0
            if not tc_available or t_ != chip_targets["tc"]:
                model += pulp.lpSum(is_tc[p_][t_] for p_ in p) == 0
            else:
                model += pulp.lpSum(is_tc[p_][t_] for p_ in p) <= 1
            if not bb_available or t_ != chip_targets["bb"]:
                model += b[t_] == 0
            if not fh_available or t_ != chip_targets["fh"]:
                model += r[t_] == 0
            model += w[t_] + pulp.lpSum(is_tc[p_][t_] for p_ in p) + b[t_] + r[t_] <= 1
        t_sub_fh = [t for t in t_sub if t <= mid_season_split_gw]
        t_sub_sh = [t for t in t_sub if t > mid_season_split_gw]
        if t_sub_fh:
            model += pulp.lpSum(w[t_] for t_ in t_sub_fh) <= (1 if wc1_available else 0)
        if t_sub_sh:
            model += pulp.lpSum(w[t_] for t_ in t_sub_sh) <= (1 if wc2_available else 0)

        squad_size_total = MK + MD + MM + MF
        for t_ in t_sub:
            if Pgk: model += pulp.lpSum(x[p_][t_] for p_ in Pgk) == MK
            if Pdef: model += pulp.lpSum(x[p_][t_] for p_ in Pdef) == MD
            if Pmid: model += pulp.lpSum(x[p_][t_] for p_ in Pmid) == MM
            if Pfwd: model += pulp.lpSum(x[p_][t_] for p_ in Pfwd) == MF
            for c_team in C_setofteams:
                players_in_team = P_c.get(c_team, [])
                if players_in_team:
                    model += pulp.lpSum(x[p_][t_] for p_ in players_in_team) <= MC
            if Pgk: model += pulp.lpSum(x_freehit[p_][t_] for p_ in Pgk) == MK * r[t_]
            if Pdef: model += pulp.lpSum(x_freehit[p_][t_] for p_ in Pdef) == MD * r[t_]
            if Pmid: model += pulp.lpSum(x_freehit[p_][t_] for p_ in Pmid) == MM * r[t_]
            if Pfwd: model += pulp.lpSum(x_freehit[p_][t_] for p_ in Pfwd) == MF * r[t_]
            model += pulp.lpSum(x_freehit[p_][t_] for p_ in p) == squad_size_total * r[t_]
            for c_team in C_setofteams:
                players_in_team = P_c.get(c_team, [])
                if players_in_team:
                    model += pulp.lpSum(x_freehit[p_][t_] for p_ in players_in_team) <= MC * r[t_]
            model += pulp.lpSum(y[p_][t_] for p_ in p) == E + phi * b[t_]
            if Pgk: model += pulp.lpSum(y[p_][t_] for p_ in Pgk) == EK + phi_K * b[t_]
            if Pdef: model += pulp.lpSum(y[p_][t_] for p_ in Pdef) >= ED
            if Pmid: model += pulp.lpSum(y[p_][t_] for p_ in Pmid) >= EM
            if Pfwd: model += pulp.lpSum(y[p_][t_] for p_ in Pfwd) >= EF
            for p_ in p:
                model += y[p_][t_] <= x_freehit[p_][t_] + lambda_var[p_][t_]
                model += lambda_var[p_][t_] <= x[p_][t_]
                model += lambda_var[p_][t_] <= 1 - r[t_]
            model += pulp.lpSum(f[p_][t_] for p_ in p) + pulp.lpSum(is_tc[p_][t_] for p_ in p) == 1
            model += pulp.lpSum(h[p_][t_] for p_ in p) == 1
            for p_ in p:
                model += f[p_][t_] + is_tc[p_][t_] + h[p_][t_] <= y[p_][t_]
                model += f[p_][t_] + h[p_][t_] <= 1
            if g:
                for p_ngk in P_not_gk:
                    is_sub = pulp.lpSum(g[p_ngk][t_][l_] for l_ in l)
                    model += y[p_ngk][t_] + is_sub <= x_freehit[p_ngk][t_] + lambda_var[p_ngk][t_]
                    model += is_sub <= 1 - y[p_ngk][t_]
                for l_ in l:
                    model += pulp.lpSum(g[p_ngk][t_][l_] for p_ngk in P_not_gk) <= 1
            for p_ in p:
                model += e[p_][t_] + u[p_][t_] <= 1
            model += pulp.lpSum(e[p_][t_] for p_ in p) == pulp.lpSum(u[p_][t_] for p_ in p)

        model += q[t1_sub] == previous_ft
        is_fresh_start = current_gw == args.start_gw and not continuing_squad
        if is_fresh_start:
            model += v[t1_sub] + pulp.lpSum(value_sub.loc[p_, t1_sub] * x[p_][t1_sub] for p_ in p) <= BS
            model += pulp.lpSum(e[p_][t1_sub] for p_ in p) == 0
            model += pulp.lpSum(u[p_][t1_sub] for p_ in p) == 0
        else:
            sales = pulp.lpSum(value_sub.loc[p_, t1_sub] * u[p_][t1_sub] for p_ in p)
            purchase = pulp.lpSum(value_sub.loc[p_, t1_sub] * e[p_][t1_sub] for p_ in p)
            model += v[t1_sub] == previous_budget + sales - purchase
            for p_ in p:
                model += x[p_][t1_sub] == previous_squad_dict.get(p_, 0) - u[p_][t1_sub] + e[p_][t1_sub]

        model += alpha[t1_sub] >= pulp.lpSum(e[p_][t1_sub] for p_ in p) - q[t1_sub]
        model += alpha[t1_sub] <= M_alpha * (1 - w[t1_sub])
        model += alpha[t1_sub] <= M_alpha * (1 - r[t1_sub])

        for idx in range(len(t_sub) - 1):
            t_curr, t_prev = t_sub[idx + 1], t_sub[idx]
            sales = pulp.lpSum(value_sub.loc[p_, t_curr] * u[p_][t_curr] for p_ in p)
            purchase = pulp.lpSum(value_sub.loc[p_, t_curr] * e[p_][t_curr] for p_ in p)
            model += v[t_curr] == v[t_prev] + sales - purchase
            for p_ in p:
                model += x[p_][t_curr] == x[p_][t_prev] - u[p_][t_curr] + e[p_][t_curr]
            cost_fh = pulp.lpSum(value_sub.loc[p_, t_curr] * x_freehit[p_][t_curr] for p_ in p)
            value_nonfh_prev = pulp.lpSum(value_sub.loc[p_, t_prev] * x[p_][t_prev] for p_ in p)
            model += cost_fh <= v[t_prev] + value_nonfh_prev + M_budget * (1 - r[t_curr])
            model += pulp.lpSum(u[p_][t_curr] for p_ in p) <= M_transfer * (1 - r[t_curr])
            model += pulp.lpSum(e[p_][t_curr] for p_ in p) <= M_transfer * (1 - r[t_curr])
            model += alpha[t_curr] >= pulp.lpSum(e[p_][t_curr] for p_ in p) - q[t_curr]
            model += alpha[t_curr] <= M_alpha * (1 - w[t_curr])
            model += alpha[t_curr] <= M_alpha * (1 - r[t_curr])
            ft_used_eff_prev = pulp.lpSum(e[p_][t_prev] for p_ in p) - alpha[t_prev]
            model += ft_carry[t_prev] >= q[t_prev] - ft_used_eff_prev
            model += ft_carry[t_prev] <= Q_bar - Q_under_bar
            chip_active_prev = w[t_prev] + r[t_prev]
            q_normal = ft_carry[t_prev] + Q_under_bar
            model += q[t_curr] <= q_normal + M_q * chip_active_prev
            model += q[t_curr] >= q_normal - M_q * chip_active_prev
            model += q[t_curr] <= Q_under_bar + M_q * (1 - chip_active_prev)
            model += q[t_curr] >= Q_under_bar - M_q * (1 - chip_active_prev)
            model += alpha[t_prev] + M_alpha * q[t_curr] <= M_alpha * Q_bar

        print(f"Solving GW {current_gw}...")
        solve_start = time.time()
        try:
            status = model.solve(solver)
        except Exception as exc:
            print(f"Solver error: {exc}")
            status = -1000
        solve_time = time.time() - solve_start
        status_str = pulp.LpStatus.get(status, "Unknown Status")
        print(f"Status: {status_str} (solve: {solve_time:.2f}s, model build: {solve_start - loop_start:.2f}s)")

        objective_value = model.objective.value() if model.objective is not None else None
        acceptable = status == pulp.LpStatusOptimal or (
            status == pulp.LpStatusNotSolved and objective_value is not None
        )

        def val(var, default=0.0):
            v_ = var.varValue
            return v_ if v_ is not None else default

        if acceptable:
            if is_fresh_start:
                previous_squad_dict = {p_: 1 for p_ in p if val(x[p_][t1_sub]) > 0.9}

            gw_results = {"gameweek": current_gw}
            gw_results["squad"] = sorted([p_ for p_ in p if val(x[p_][t1_sub]) > 0.9])
            gw_results["lineup"] = sorted([p_ for p_ in p if val(y[p_][t1_sub]) > 0.9])
            potential_captains = [p_ for p_ in p if val(f[p_][t1_sub]) > 0.9]
            potential_tc = [p_ for p_ in p if val(is_tc[p_][t1_sub]) > 0.9]
            tc_active = bool(potential_tc)
            captain_id = potential_tc[0] if potential_tc else (potential_captains[0] if potential_captains else None)
            gw_results["captain"] = [captain_id] if captain_id is not None else []
            gw_results["vice_captain"] = sorted([p_ for p_ in p if val(h[p_][t1_sub]) > 0.9])
            gw_results["transfers_in"] = sorted([p_ for p_ in p if val(e[p_][t1_sub]) > 0.9])
            gw_results["transfers_out"] = sorted([p_ for p_ in p if val(u[p_][t1_sub]) > 0.9])
            gw_results["budget_end"] = val(v[t1_sub], previous_budget)
            gw_results["budget_start"] = previous_budget
            gw_results["alpha"] = round(val(alpha[t1_sub]))
            gw_results["q_start"] = previous_ft
            gw_results["objective_value"] = objective_value

            wc_active = val(w[t1_sub]) > 0.9
            bb_active = val(b[t1_sub]) > 0.9
            fh_active = val(r[t1_sub]) > 0.9
            chip_name = None
            if wc_active: chip_name = "WC"
            if bb_active: chip_name = "BB"
            if fh_active: chip_name = "FH"
            if tc_active and captain_id is not None:
                chip_name = f"TC_{player_name_map.get(captain_id, captain_id)}"
            gw_results["chip_played"] = chip_name

            e_sum = len(gw_results["transfers_in"])
            if chip_name in ("WC", "FH"):
                next_ft = Q_under_bar
            else:
                ft_used_eff = max(0, e_sum - gw_results["alpha"])
                ft_carry_val = max(0, previous_ft - ft_used_eff)
                next_ft = min(Q_bar, math.floor(ft_carry_val + Q_under_bar))

            if fh_active:
                previous_budget = gw_results["budget_start"]
            else:
                previous_squad_dict = {p_: 1 for p_ in gw_results["squad"]}
                previous_budget = gw_results["budget_end"]
            previous_ft = next_ft

            if chip_name == "WC":
                if current_gw <= mid_season_split_gw:
                    used_chips_tracker["wc1"] = True
                else:
                    used_chips_tracker["wc2"] = True
            elif chip_name == "BB":
                used_chips_tracker["bb"] = True
            elif chip_name == "FH":
                used_chips_tracker["fh"] = True
            elif chip_name and chip_name.startswith("TC_"):
                used_chips_tracker["tc"] = True

            master_results.append(gw_results)
            print(f"GW {current_gw}: chip={chip_name}, transfers in/out={len(gw_results['transfers_in'])}, "
                  f"budget_end={previous_budget:.1f}, next FT={previous_ft}")
        else:
            print(f"No acceptable solution for GW {current_gw}; falling back to no transfers.")
            master_results.append({
                "gameweek": current_gw, "squad": sorted(previous_squad_dict.keys()),
                "lineup": [], "captain": [], "vice_captain": [], "transfers_in": [], "transfers_out": [],
                "budget_end": previous_budget, "budget_start": previous_budget, "alpha": 0,
                "q_start": previous_ft, "objective_value": None, "chip_played": "FALLBACK_NO_TRANSFERS",
            })
            previous_ft = min(Q_bar, math.floor(previous_ft + Q_under_bar))

        print(f"GW {current_gw} loop time: {time.time() - loop_start:.2f}s")

    results_df = pd.DataFrame(master_results)
    if not results_df.empty:
        id_cols = ["squad", "lineup", "captain", "vice_captain", "transfers_in", "transfers_out"]
        results_named = results_df.copy()
        for col in id_cols:
            results_named[col] = results_named[col].apply(
                lambda ids: sorted(player_name_map.get(i, f"ID:{i}") for i in ids) if isinstance(ids, list) else ids
            )
        if "actual_total_points" in data_full_range_raw.columns:
            actual_matrix = data_full_range_raw.pivot(index="player_id", columns="GW", values="actual_total_points")

            def pts(ids, gw):
                return sum(actual_matrix.loc[i, gw] for i in ids
                           if i in actual_matrix.index and gw in actual_matrix.columns
                           and not pd.isna(actual_matrix.loc[i, gw]))

            for idx, row in results_df.iterrows():
                gw = row["gameweek"]
                lineup_pts = pts(row["lineup"], gw)
                captain_bonus = 0
                if row["captain"]:
                    cap_id = row["captain"][0]
                    if cap_id in actual_matrix.index and gw in actual_matrix.columns and not pd.isna(actual_matrix.loc[cap_id, gw]):
                        base = actual_matrix.loc[cap_id, gw]
                        captain_bonus = base * 2 if (row["chip_played"] or "").startswith("TC_") else base
                if row["chip_played"] == "BB":
                    bench = [i for i in row["squad"] if i not in row["lineup"]]
                    lineup_pts += pts(bench, gw)
                results_named.at[idx, "actual_squad_points"] = pts(row["squad"], gw)
                results_named.at[idx, "actual_lineup_points"] = lineup_pts
                results_named.at[idx, "actual_captain_points"] = captain_bonus
                results_named.at[idx, "actual_total_points"] = lineup_pts + captain_bonus

        for col in id_cols:
            results_named[col] = results_named[col].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x)

        output_path = args.output or str(
            config.SQUAD_OUTPUT_DIR / f"squad_selection_W{args.start_gw}-{args.max_gw}_SHL{args.horizon}.csv"
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results_named.to_csv(output_path, index=False)
        print(f"\nSaved results to {output_path}")
        if "actual_total_points" in results_named.columns:
            print(f"Total actual points over horizon: {results_named['actual_total_points'].sum():.1f}")
    else:
        print("No results generated.")

    return results_df


if __name__ == "__main__":
    run(parse_args())
