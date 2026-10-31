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
import itertools
import math
import sys
import time
from pathlib import Path
from collections import Counter
from typing import Hashable, Mapping, Sequence

import pandas as pd
import pulp

from fpl import config


def apply_auto_substitutions(
    lineup: Sequence[Hashable], bench_order: Sequence[Hashable],
    minutes: Mapping[Hashable, float], positions: Mapping[Hashable, str],
) -> list[Hashable]:
    """Return scoring players after priority-ordered, formation-legal substitutions.

    Formation applies to all starting slots, including an absent player who cannot
    be replaced. It does not require the players who actually appeared to form an
    eleven. This allows a legal defender replacement even when another starting
    defender remains absent. Bench eligibility uses whole-round realized minutes.
    """
    active = [p for p in lineup if minutes.get(p, 0) > 0]
    if any(positions.get(p) == "GK" and minutes.get(p, 0) <= 0 for p in lineup):
        reserve = next((p for p in bench_order
                        if positions.get(p) == "GK" and minutes.get(p, 0) > 0), None)
        if reserve is not None:
            active.append(reserve)
    absent = [p for p in lineup
              if positions.get(p) != "GK" and minutes.get(p, 0) <= 0]
    available = [p for p in bench_order
                 if positions.get(p) != "GK" and minutes.get(p, 0) > 0]
    original = Counter(positions[p] for p in lineup)
    for size in range(min(len(absent), len(available)), 0, -1):
        # Combinations retain bench priority; removal choices cannot alter scores.
        for replacements in itertools.combinations(available, size):
            added = Counter(positions[p] for p in replacements)
            for removed in itertools.combinations(absent, size):
                counts = original.copy()
                counts.subtract(positions[p] for p in removed)
                counts.update(added)
                if all(config.LINEUP_MIN_COUNTS[pos] <= counts[pos] <= config.LINEUP_MAX_COUNTS[pos]
                       for pos in config.LINEUP_MIN_COUNTS):
                    return active + list(replacements)
    return active


def resolve_active_captain(
    captain: Sequence[Hashable], vice_captain: Sequence[Hashable],
    realized_lineup: Sequence[Hashable], minutes: Mapping[Hashable, float],
) -> Hashable | None:
    """Double/triple the vice only when the designated captain played no minutes."""
    cap = captain[0] if captain else None
    if cap in realized_lineup and minutes.get(cap, 0) > 0:
        return cap
    vice = vice_captain[0] if vice_captain else None
    return vice if vice in realized_lineup and minutes.get(vice, 0) > 0 else None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="FPL squad selection via MILP (rolling horizon)")
    parser.add_argument("--scoring-mode", choices=("auto", "legacy", "corrected"), default="auto",
                        help="Corrected realized scoring requires finite round minutes; auto labels legacy inputs explicitly.")
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


def validate_chip_targets(args) -> dict[str, int]:
    """Validate explicit chip scheduling against this run and standing season halves."""
    if args.start_gw < 1 or args.max_gw < args.start_gw or args.horizon < 1:
        raise ValueError("positive chronological gameweeks and horizon are required")
    targets = {name: getattr(args, f"{name}_gw", 0) for name in ("wc1", "wc2", "tc", "fh", "bb")}
    positive = [target for target in targets.values() if target > 0]
    if any(target < 0 for target in targets.values()):
        raise ValueError("chip targets must be nonnegative; 0 disables a chip")
    if len(positive) != len(set(positive)):
        raise ValueError("only one chip may be requested per gameweek")
    if any(not args.start_gw <= target <= args.max_gw for target in positive):
        raise ValueError("chip targets must lie inside the requested run range")
    season_start = ((args.start_gw - 1) // config.GWS_PER_SEASON) * config.GWS_PER_SEASON + 1
    half_end = season_start + 18
    season_end = season_start + config.GWS_PER_SEASON - 1
    for name in ("wc1", "wc2"):
        target = targets[name]
        if target and not season_start <= target <= season_end:
            raise ValueError("wildcard target must be in the starting season")
        if target and ((name == "wc1" and target > half_end)
                       or (name == "wc2" and target <= half_end)):
            raise ValueError("wildcard target is in the wrong season half")
    return targets


def run(args):
    chip_targets = validate_chip_targets(args)
    solver = make_solver(args.solver, args.time_limit, args.threads, args.gap_rel)

    print(f"--- Loading predictions from {args.predictions_csv} ---")
    allesesonger = pd.read_csv(args.predictions_csv)
    essential_input_cols = ["player_id", "GW", "name", "position", "team", args.points_col, "value"]
    missing_cols = [c for c in essential_input_cols if c not in allesesonger.columns]
    if missing_cols:
        sys.exit(f"ERROR: Missing essential columns in {args.predictions_csv}: {missing_cols}")

    scoring_mode = getattr(args, "scoring_mode", "auto")
    corrected_scoring = scoring_mode != "legacy" and "minutes" in allesesonger
    if scoring_mode == "corrected" and "minutes" not in allesesonger:
        raise ValueError("corrected scoring requires a minutes column")
    if corrected_scoring:
        minutes = pd.to_numeric(allesesonger["minutes"], errors="coerce")
        if minutes.isna().any() or ((minutes < 0) | (minutes == float("inf"))).any():
            raise ValueError("corrected scoring requires finite nonnegative minutes")
        allesesonger["minutes"] = minutes

    allesesonger["GW"] = pd.to_numeric(allesesonger["GW"])
    allesesonger["value"] = pd.to_numeric(allesesonger["value"]).fillna(50.0)

    # Origin-based predictions (fpl.model.predict --origin-based, audit B2): the CSV holds
    # one forecast per (origin_gw, player, GW), where the origin-t rows were built with only
    # information available at t's deadline. The solve at GW t then uses exclusively the
    # origin-t forecast set for its whole lookahead, mirroring what a live run knows.
    has_origin = "origin_gw" in allesesonger.columns
    if has_origin:
        allesesonger["origin_gw"] = pd.to_numeric(allesesonger["origin_gw"])
        print("origin_gw column found: solving each gameweek from its own origin forecast set.")

    # --- Double gameweek aggregation ---
    group_keys = ["player_id", "GW"] + (["origin_gw"] if has_origin else [])
    sum_cols = [c for c in [args.points_col, "actual_total_points", "minutes"] if c in allesesonger.columns]
    first_cols = ["name", "position", "team", "value"]
    agg = {c: "sum" for c in sum_cols}
    agg.update({c: "first" for c in first_cols if c in allesesonger.columns})
    allesesonger = allesesonger.groupby(group_keys, as_index=False).agg(agg)

    if any(target and target not in set(allesesonger["GW"]) for target in chip_targets.values()):
        raise ValueError("requested chip gameweek has no predictions")

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
    # Banking policy, not the site rule: config caps how many FTs the solver PLANS to bank
    # (see config.py for the backtest evidence). A real squad can arrive with more already
    # banked (the site allows 5), so the state bound honors --initial-ft rather than
    # silently clamping it / going infeasible. Trade-off: when initial_ft exceeds the
    # policy cap, the lifted bound applies to the whole horizon (the solver could in
    # principle re-bank up to the initial stack); a per-period bound isn't worth the
    # formulation complexity for a rare, live-only, few-GW situation.
    Q_bar = max(config.MILP_MAX_FREE_TRANSFERS, args.initial_ft)
    Q_under_bar = config.MILP_FT_PER_GW
    epsilon = 0.1
    kappa = {1: 0.01, 2: 0.005, 3: 0.001}
    M_transfer = MK + MD + MM + MF
    M_budget = BS + M_transfer * 200
    M_alpha = M_transfer + Q_bar
    M_q = Q_bar + 1

    # With origin_gw, (player, GW) pairs repeat across origins: values/actuals are identical
    # on every copy (they come from the target row), so the first occurrence stands in; the
    # POINTS differ per origin, so those get one matrix per origin, selected in the GW loop.
    dedup_pg = (data_full_range_raw.drop_duplicates(subset=["player_id", "GW"])
                if has_origin else data_full_range_raw)
    points_matrix_by_origin = {}
    if has_origin:
        for origin, origin_rows in data_full_range_raw.groupby("origin_gw"):
            points_matrix_by_origin[origin] = (
                origin_rows.pivot(index="player_id", columns="GW", values=args.points_col)
                .reindex(index=p).fillna(0.0)
            )
        points_matrix_df = None
    else:
        points_matrix_df = data_full_range_raw.pivot(index="player_id", columns="GW", values=args.points_col)
        points_matrix_df = points_matrix_df.reindex(index=p, columns=T_setofgameweeks_full, fill_value=0.0).fillna(0.0)
    value_matrix_df = dedup_pg.pivot(index="player_id", columns="GW", values="value")
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

        x = model.add_variable_dicts("Squad", (p, t_sub), cat="Binary")
        x_freehit = model.add_variable_dicts("Squad_FH", (p, t_sub), cat="Binary")
        y = model.add_variable_dicts("Lineup", (p, t_sub), cat="Binary")
        f = model.add_variable_dicts("Captain", (p, t_sub), cat="Binary")
        h = model.add_variable_dicts("ViceCaptain", (p, t_sub), cat="Binary")
        is_tc = model.add_variable_dicts("TripleCaptainChipActive", (p, t_sub), cat="Binary")
        u = model.add_variable_dicts("TransferOut", (p, t_sub), cat="Binary")
        e = model.add_variable_dicts("TransferIn", (p, t_sub), cat="Binary")
        lambda_var = model.add_variable_dicts("Aux_LineupInSquad", (p, t_sub), cat="Binary")
        g = {}
        if P_not_gk and l:
            g = model.add_variable_dicts("Substitution", (P_not_gk, t_sub, l), cat="Binary")
        w = model.add_variable_dicts("WildcardChipActive", t_sub, cat="Binary")
        b = model.add_variable_dicts("BenchBoostChipActive", t_sub, cat="Binary")
        r = model.add_variable_dicts("FreeHitChipActive", t_sub, cat="Binary")
        v = model.add_variable_dicts("RemainingBudget", t_sub, lowBound=0, cat="Continuous")
        q = model.add_variable_dicts("FreeTransfersAvailable", t_sub, lowBound=0, upBound=Q_bar, cat="Integer")
        alpha = model.add_variable_dicts("PenalizedTransfers", t_sub, lowBound=0, upBound=M_alpha, cat="Integer")
        ft_carry = model.add_variable_dicts("FT_Carry", t_sub, lowBound=0)

        if has_origin:
            origin_matrix = points_matrix_by_origin.get(current_gw)
            if origin_matrix is None:
                sys.exit(f"ERROR: origin-based predictions CSV has no origin_gw={current_gw} "
                         f"forecast set - regenerate it over the full GW range.")
            points_sub = origin_matrix.reindex(columns=t_sub, fill_value=0.0).fillna(0.0).loc[p]
        else:
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
            wildcard = "wc1" if t_ <= mid_season_split_gw else "wc2"
            wildcard_available = wc1_available if wildcard == "wc1" else wc2_available
            model += w[t_] == int(wildcard_available and t_ == chip_targets[wildcard])
            model += pulp.lpSum(is_tc[p_][t_] for p_ in p) == int(tc_available and t_ == chip_targets["tc"])
            model += b[t_] == int(bb_available and t_ == chip_targets["bb"])
            model += r[t_] == int(fh_available and t_ == chip_targets["fh"])
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
            # Bank is exactly the unspent fresh-build budget.
            model += v[t1_sub] + pulp.lpSum(value_sub.loc[p_, t1_sub] * x[p_][t1_sub] for p_ in p) == BS
            model += pulp.lpSum(e[p_][t1_sub] for p_ in p) == 0
            model += pulp.lpSum(u[p_][t1_sub] for p_ in p) == 0
        else:
            sales = pulp.lpSum(value_sub.loc[p_, t1_sub] * u[p_][t1_sub] for p_ in p)
            purchase = pulp.lpSum(value_sub.loc[p_, t1_sub] * e[p_][t1_sub] for p_ in p)
            model += v[t1_sub] == previous_budget + sales - purchase
            for p_ in p:
                model += x[p_][t1_sub] == previous_squad_dict.get(p_, 0) - u[p_][t1_sub] + e[p_][t1_sub]

        # A first-period Free Hit has the same budget/state limits as later periods.
        if is_fresh_start:
            available_fh_budget = BS
        else:
            available_fh_budget = previous_budget + sum(
                value_sub.loc[p_, t1_sub] * previous_squad_dict.get(p_, 0) for p_ in p
            )
        model += pulp.lpSum(value_sub.loc[p_, t1_sub] * x_freehit[p_][t1_sub] for p_ in p) <= (
            available_fh_budget + M_budget * (1 - r[t1_sub])
        )
        model += pulp.lpSum(u[p_][t1_sub] for p_ in p) <= M_transfer * (1 - r[t1_sub])
        model += pulp.lpSum(e[p_][t1_sub] for p_ in p) <= M_transfer * (1 - r[t1_sub])
        model += alpha[t1_sub] >= (
            pulp.lpSum(e[p_][t1_sub] for p_ in p) - q[t1_sub]
            - M_alpha * (w[t1_sub] + r[t1_sub])
        )
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
            model += alpha[t_curr] >= (
                pulp.lpSum(e[p_][t_curr] for p_ in p) - q[t_curr]
                - M_alpha * (w[t_curr] + r[t_curr])
            )
            model += alpha[t_curr] <= M_alpha * (1 - w[t_curr])
            model += alpha[t_curr] <= M_alpha * (1 - r[t_curr])
            ft_used_eff_prev = pulp.lpSum(e[p_][t_prev] for p_ in p) - alpha[t_prev]
            model += ft_carry[t_prev] >= (
                q[t_prev] - ft_used_eff_prev - M_q * (w[t_prev] + r[t_prev])
            )
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
            selected_squad = x_freehit if val(r[t1_sub]) > 0.9 else x
            gw_results["squad"] = sorted([p_ for p_ in p if val(selected_squad[p_][t1_sub]) > 0.9])
            gw_results["lineup"] = sorted([p_ for p_ in p if val(y[p_][t1_sub]) > 0.9])
            bench = [p_ for p_ in gw_results["squad"] if p_ not in gw_results["lineup"]]
            bench_gk = [p_ for p_ in bench if pos_map.get(p_) == "GK"]
            bench_outfield = sorted(
                [p_ for p_ in bench if pos_map.get(p_) != "GK"],
                key=lambda p_: (-points_sub.loc[p_, t1_sub], str(p_)),
            )
            gw_results["bench_order"] = bench_gk + bench_outfield
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
                "lineup": [], "bench_order": [], "captain": [], "vice_captain": [], "transfers_in": [], "transfers_out": [],
                "budget_end": previous_budget, "budget_start": previous_budget, "alpha": 0,
                "q_start": previous_ft, "objective_value": None, "chip_played": "FALLBACK_NO_TRANSFERS",
            })
            previous_ft = min(Q_bar, math.floor(previous_ft + Q_under_bar))

        print(f"GW {current_gw} loop time: {time.time() - loop_start:.2f}s")

    results_df = pd.DataFrame(master_results)
    if not results_df.empty:
        id_cols = ["squad", "lineup", "bench_order", "captain", "vice_captain", "transfers_in", "transfers_out"]
        results_named = results_df.copy()
        for col in id_cols:
            results_named[col] = results_named[col].apply(
                lambda ids: sorted(player_name_map.get(i, f"ID:{i}") for i in ids) if isinstance(ids, list) else ids
            )
        if "actual_total_points" in data_full_range_raw.columns:
            actual_matrix = dedup_pg.pivot(index="player_id", columns="GW", values="actual_total_points")
            minutes_matrix = dedup_pg.pivot(index="player_id", columns="GW", values="minutes") if corrected_scoring else None

            def pts(ids, gw):
                return sum(actual_matrix.loc[i, gw] for i in ids
                           if i in actual_matrix.index and gw in actual_matrix.columns
                           and not pd.isna(actual_matrix.loc[i, gw]))

            for idx, row in results_df.iterrows():
                gw = row["gameweek"]
                minute_map = ({p_: minutes_matrix.loc[p_, gw] for p_ in p
                               if p_ in minutes_matrix.index and gw in minutes_matrix.columns
                               and not pd.isna(minutes_matrix.loc[p_, gw])}
                              if minutes_matrix is not None else {p_: 1 for p_ in p})
                realized_lineup = ([p_ for p_ in row["squad"] if minute_map.get(p_, 0) > 0]
                                   if row["chip_played"] == "BB" else
                                   apply_auto_substitutions(row["lineup"], row["bench_order"], minute_map, pos_map))
                lineup_pts = pts(realized_lineup, gw)
                captain_bonus = 0
                active_captain = resolve_active_captain(
                    row["captain"], row["vice_captain"], realized_lineup, minute_map
                )
                if active_captain is not None:
                    cap_id = active_captain
                    if cap_id in actual_matrix.index and gw in actual_matrix.columns and not pd.isna(actual_matrix.loc[cap_id, gw]):
                        base = actual_matrix.loc[cap_id, gw]
                        chip = row["chip_played"]
                        captain_bonus = base * 2 if isinstance(chip, str) and chip.startswith("TC_") else base
                results_named.at[idx, "scoring_protocol"] = ("autosubs_v2" if corrected_scoring else "legacy_no_autosubs")
                results_named.at[idx, "actual_squad_points"] = pts(row["squad"], gw)
                results_named.at[idx, "actual_lineup"] = ", ".join(
                    sorted(player_name_map.get(p_, f"ID:{p_}") for p_ in realized_lineup)
                )
                results_named.at[idx, "actual_captain"] = (
                    player_name_map.get(active_captain, f"ID:{active_captain}")
                    if active_captain is not None else ""
                )
                results_named.at[idx, "actual_lineup_points"] = lineup_pts
                results_named.at[idx, "actual_captain_points"] = captain_bonus
                results_named.at[idx, "actual_total_points"] = lineup_pts + captain_bonus

        for col in id_cols:
            results_named[col] = results_named[col].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x)

        origin_tag = "_origin" if has_origin else ""
        output_path = args.output or str(
            config.SQUAD_OUTPUT_DIR / f"squad_selection_W{args.start_gw}-{args.max_gw}_SHL{args.horizon}{origin_tag}.csv"
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
