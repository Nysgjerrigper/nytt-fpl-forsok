"""Offline Open FPL Solver adapter with checked modern rules and legacy score lineage."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from numbers import Real
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from fpl import config
from fpl.milp import optimize
from fpl.milp.solio_inputs import prepare_inputs, _validate_forecast_admission
from fpl.milp.solio_output import normalize_solution
from fpl.milp._vendor.solio.solver import solve_multi_period_fpl

CHIPS = ("wc", "fh", "bb", "tc")


def chip_options(start_local: int, horizon: int, inventory: Mapping | None,
                 forced: Mapping[str, Sequence[int]] | None,
                 previous_free_hit_gw: int | None = None) -> dict:
    """Validate exact per-half inventories and forced local weeks, with no implied chips."""
    normalized = {chip: {half: 0 for half in (1, 2)} for chip in CHIPS}
    if previous_free_hit_gw is not None and (isinstance(previous_free_hit_gw, bool) or
        not isinstance(previous_free_hit_gw, int) or not 1 <= previous_free_hit_gw < start_local):
        raise ValueError("previous Free Hit must precede the requested local week")
    if inventory is not None:
        if set(inventory) - set(CHIPS):
            raise ValueError("unknown chip inventory")
        for chip, halves in inventory.items():
            for half, value in halves.items():
                if str(half) not in ("1", "2") or isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
                    raise ValueError("chip inventories need half 1/2 and integer 0/1 counts")
                normalized[chip][int(half)] = value
    forced = {} if forced is None else dict(forced)
    if set(forced) - set(CHIPS):
        raise ValueError("unknown forced chip")
    occupied = set()
    options = {"chip_inventory": normalized, "chip_half_boundary": config.MODERN_CHIP_HALF_BOUNDARY,
               "previous_free_hit_gw": previous_free_hit_gw}
    for chip, weeks in forced.items():
        counts = {1: 0, 2: 0}
        for week in weeks:
            if isinstance(week, bool) or not isinstance(week, int) or not start_local <= week < start_local + horizon:
                raise ValueError("forced chip target outside solver horizon")
            if week in occupied:
                raise ValueError("chip targets collide")
            occupied.add(week)
            half = 1 if week <= config.MODERN_CHIP_HALF_BOUNDARY else 2
            counts[half] += 1
            if counts[half] > normalized[chip][half]:
                raise ValueError("forced chip not available in its half")
            if chip == "wc" and week == 1:
                raise ValueError("Wildcard unavailable before GW2")
            if chip == "fh" and (week == 1 or week - 1 == previous_free_hit_gw or week - 1 in weeks):
                raise ValueError("Free Hit unavailable in GW1 or consecutive rounds")
        options[f"use_{chip}"] = list(weeks)
    return options


def solve_window(predictions: pd.DataFrame, *, season: str, start_gw: int, horizon: int,
                 initial_squad: Sequence[int] | None = None, bank: float = 1000,
                 free_transfers: int = 1, sell_prices: Mapping[int, float] | None = None,
                 bootstrap: dict | None = None, chip_inventory: Mapping | None = None,
                 forced_chips: Mapping[str, Sequence[int]] | None = None,
                 previous_free_hit_gw: int | None = None,
                 time_limit: float = 1200, seed: int = 0) -> pd.DataFrame:
    """Solve an entire window; reject nonoptimal or independently illegal decisions.

    Forecasts use global weeks/stable codes and prices/bank in tenths of a million.
    No data fetching or model fitting occurs. Chip targets use season-local weeks.
    The objective matches the legacy point/captain/vice/bench terms, with no new FT
    or cash incentive. Current transfer-bank and chip rules are separately explicit.
    """
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be finite and positive")
    week_col = "GW_global" if "GW_global" in predictions else "GW"
    if week_col not in predictions or not set(range(start_gw, start_gw+horizon)) <= set(predictions[week_col]):
        raise ValueError("every requested round needs forecasts; represent proven whole-round blanks explicitly")
    if bootstrap is not None and "events" in bootstrap:
        from fpl.data.fetch import active_season_from_bootstrap
        if active_season_from_bootstrap(bootstrap) != season:
            raise ValueError("bootstrap active season disagrees with forecast season")
    prepared = prepare_inputs(predictions, season=season, start_gw=start_gw, horizon=horizon,
                              initial_squad=initial_squad, bank=bank, free_transfers=free_transfers,
                              sell_prices=sell_prices, bootstrap=bootstrap)
    options = chip_options(prepared["next_gw"], horizon, chip_inventory, forced_chips,
                           previous_free_hit_gw)
    options.update(horizon=horizon, objective="regular", bench_weights={0: 0, 1: .01, 2: .005, 3: .001},
                   ft_value=0, itb_value=0, vcap_weight=.1, hit_cost=4,
                   preseason=initial_squad is None, gap=0, secs=time_limit,
                   num_iterations=1, random_seed=seed, solver="highs", verbose=False)
    started = time.perf_counter()
    with redirect_stdout(StringIO()):
        solutions = solve_multi_period_fpl(prepared, options)
    if len(solutions) != 1 or solutions[0]["solver_status"] != "Optimal":
        raise RuntimeError("Solio failed to return exactly one optimal solution")
    result = pd.DataFrame(normalize_solution(solutions[0], prepared))
    result["optimizer"] = "solio_modern_v1"
    result["solver_status"] = solutions[0]["solver_status"]
    result["mip_gap"] = solutions[0]["mip_gap"]
    result["solver_seconds"] = solutions[0]["solver_seconds"]
    result["window_seconds"] = time.perf_counter() - started
    result["upstream_commit"] = config.SOLIO_UPSTREAM_COMMIT
    return result


def score_decisions(decisions: pd.DataFrame, predictions: pd.DataFrame, *,
                    require_complete_outcomes: bool = False,
                    explicit_blank_outcomes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Settle corrected fixture outcomes with explicit selected-squad coverage.

    Strict mode requires an outcome row for every selected player in that round;
    absent rows are never silently assumed to be blanks. Default mode preserves
    legacy zero filling but labels that assumption. Complete coverage is not an
    independent legality/state certificate. Separate explicit_blank_outcomes must
    carry caller-verified blank evidence; this scorer checks zero arithmetic only.
    Captain and vice identities retain the existing one-element-list interface.
    """
    if type(require_complete_outcomes) is not bool:
        raise ValueError("require_complete_outcomes must be a boolean")
    required = {"player_id", "GW", "actual_total_points", "minutes", "position"}
    if predictions.empty or not predictions.columns.is_unique or not required <= set(predictions):
        raise ValueError("realized scoring requires nonempty unambiguous fixture points, minutes and identities")
    if decisions.empty or not decisions.columns.is_unique or not {"gameweek", "alpha"} <= set(decisions):
        raise ValueError("scored decisions require nonempty unique columns, gameweeks and hit counts")

    def numeric(column, name):
        if not column.map(lambda value: isinstance(value, Real) and not isinstance(value, (bool, np.bool_))).all():
            raise ValueError(f"{name} must contain non-boolean numeric values")
        result = pd.to_numeric(column, errors="raise")
        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} must contain finite values")
        return result

    out = decisions.copy().reset_index(drop=True)
    weeks = numeric(out.gameweek, "decision gameweek")
    hits = numeric(out.alpha, "decision alpha")
    if ((weeks <= 0) | (weeks % 1 != 0)).any() or weeks.duplicated().any():
        raise ValueError("decision gameweeks must be positive unique integers")
    if ((hits < 0) | (hits % 1 != 0)).any():
        raise ValueError("decision alpha must contain nonnegative integer hit counts")
    out["gameweek"] = weeks.map(int)
    out["alpha"] = hits.map(int)
    out["missing_outcome_players"] = 0
    out["explicit_blank_players"] = 0
    out["outcome_coverage_assessment"] = ""
    values = predictions.copy()
    for column in ("actual_total_points", "minutes"):
        values[column] = numeric(values[column], column)
    if (values.minutes < 0).any():
        raise ValueError("negative realized minutes")
    if not values.position.isin(config.ONFIELD_POSITIONS).all():
        raise ValueError("realized outcomes require known positions")
    if "GW_global" not in values:
        values["GW_global"] = values["GW"]
    if "origin_gw" in values:
        for origin, rows in values.groupby("origin_gw", dropna=False):
            _validate_forecast_admission(rows.copy(), int(origin))
    else:
        _validate_forecast_admission(values.copy(), int(values.GW.min()))
    keys = (["origin_gw"] if "origin_gw" in values else []) + ["player_id", "GW"]
    if (values.groupby(keys).position.nunique() != 1).any():
        raise ValueError("positions disagree within a realized player round")
    grouped = values.groupby(keys).agg(
        actual_total_points=("actual_total_points", "sum"), minutes=("minutes", "sum"),
        position=("position", "first"))
    if not np.isfinite(grouped[["actual_total_points", "minutes"]].to_numpy(dtype=float)).all():
        raise ValueError("aggregated realized points/minutes must remain finite")
    if "origin_gw" in values:
        panel = grouped.reset_index()
        counts = panel.groupby(["player_id", "GW"])[["actual_total_points", "minutes", "position"]].nunique()
        if (counts > 1).any().any():
            raise ValueError("realized rounds disagree across forecast origins")
        rounds = panel.drop_duplicates(["player_id", "GW"]).set_index(["player_id", "GW"])
    else:
        rounds = grouped
    # Blank outcomes are separate player-round evidence, never invented fixtures
    # or zero forecasts. The caller must verify their external evidence receipts.
    if explicit_blank_outcomes is not None:
        blanks = explicit_blank_outcomes.copy()
        required_blank = {"player_id", "GW", "position", "actual_total_points", "minutes", "outcome_basis"}
        if (not blanks.columns.is_unique or not required_blank <= set(blanks)
                or not blanks.outcome_basis.eq("EXPLICIT_RETROSPECTIVE_BLANK").all()
                or not blanks.position.isin(config.ONFIELD_POSITIONS).all()):
            raise ValueError("invalid explicit blank outcome schema")
        for column in ("player_id", "GW", "actual_total_points", "minutes"):
            blanks[column] = numeric(blanks[column], f"blank {column}")
        identities = blanks[["player_id", "GW"]]
        if (not blanks[["actual_total_points", "minutes"]].eq(0).all().all()
                or not ((identities > 0) & (identities < 2**63) & (identities % 1 == 0)).all().all()
                or blanks.duplicated(["player_id", "GW"]).any()):
            raise ValueError("explicit blanks require unique integer identities and zero outcomes")
        blanks = blanks.set_index(["player_id", "GW"])
        if len(rounds.index.intersection(blanks.index)):
            raise ValueError("explicit blanks cannot replace observed player outcomes")
        rounds = pd.concat([rounds, blanks[["actual_total_points", "minutes", "position"]]])
    for index, row in out.iterrows():
        gw = row.gameweek
        current = rounds.xs(gw, level="GW") if gw in rounds.index.get_level_values("GW") else rounds.iloc[0:0]
        present = set(current.index.get_level_values("player_id")) if isinstance(current.index, pd.MultiIndex) else set(current.index)
        missing = [player for player in row.squad if player not in present]
        if require_complete_outcomes and missing:
            raise ValueError(f"Missing outcomes for GW{gw}, selected player IDs: {missing}")
        out.at[index, "missing_outcome_players"] = len(missing)
        if explicit_blank_outcomes is not None:
            out.at[index, "explicit_blank_players"] = sum((player, gw) in blanks.index for player in row.squad)
        out.at[index, "outcome_coverage_assessment"] = "LEGACY_ZERO_FILL" if missing else "COMPLETE_SELECTED_SQUAD"
        minutes = current.minutes.to_dict()
        positions = current.position.to_dict()
        for player in row.squad:
            positions.setdefault(player, next(iter(values.loc[values.player_id == player, "position"]), None))
        active = (list(row.squad) if row.chip_played == "BB" else
                  optimize.apply_auto_substitutions(row.lineup, row.bench_order, minutes, positions))
        captain = optimize.resolve_active_captain(row.captain, row.vice_captain, active, minutes)
        points = current.actual_total_points.to_dict()
        gross = sum(points.get(player, 0) for player in active)
        triple = isinstance(row.chip_played, str) and (row.chip_played == "TC" or row.chip_played.startswith("TC_"))
        gross += points.get(captain, 0) * (2 if triple else 1)
        out.at[index, "actual_total_points"] = gross
        out.at[index, "actual_net_points"] = gross - 4 * row.alpha
        out.at[index, "scoring_protocol"] = "autosubs_v2"
    out.index = decisions.index
    return out


def run_schedule(predictions: pd.DataFrame, *, season: str, start_gw: int, max_gw: int,
                 horizon: int = 3, initial_squad: Sequence[int] | None = None,
                 bank: float = 1000, free_transfers: int = 1,
                 sell_prices: Mapping[int, float] | None = None, chip_inventory: Mapping | None = None,
                 forced_chips: Mapping[str, Sequence[int]] | None = None,
                 bootstrap: dict | None = None, time_limit: float = 1200,
                 previous_free_hit_gw: int | None = None,
                 origin_price_captures: Mapping[int, dict] | None = None,
                 purchase_prices: Mapping[int, int] | None = None) -> pd.DataFrame:
    """Roll decisions forward with permanent FH state and explicit price assumptions.

    Input metadata/prices must be explicitly static over the evaluated window. Origin
    forecasts are selected by their exact origin before solving; no alternative origin
    is substituted. Future planned chips do not consume actual inventory.
    Optional admitted origin captures replace future-row metadata/prices, and
    acquisition costs determine half-profit owned selling prices. This mode returns
    decisions only; realized outcomes must be settled in a separate replay.
    """
    if max_gw < start_gw or horizon < 1:
        raise ValueError("invalid evaluation range or horizon")
    offset = config.season_start_gw(season) - 1
    if origin_price_captures is not None:
        if set(origin_price_captures) != set(range(start_gw, max_gw + 1)):
            raise ValueError("origin price captures must cover the exact evaluation range")
        if "origin_gw" not in predictions:
            raise ValueError("origin price mode requires explicitly origin-indexed forecasts")
        if bootstrap is not None or sell_prices is not None:
            raise ValueError("origin captures use acquisition basis, not separate bootstrap/sell prices")
        if initial_squad is None and purchase_prices:
            raise ValueError("fresh origin-price runs cannot have initial purchase prices")
    elif purchase_prices is not None:
        raise ValueError("purchase_prices requires admitted origin price captures")
    if previous_free_hit_gw is not None and (isinstance(previous_free_hit_gw, bool) or
        not isinstance(previous_free_hit_gw, int) or not 1 <= previous_free_hit_gw < start_gw-offset):
        raise ValueError("previous Free Hit must precede the requested local week")
    inventory = chip_options(start_gw-offset, max_gw-start_gw+1, chip_inventory,
                             forced_chips, previous_free_hit_gw)["chip_inventory"]
    state = None if initial_squad is None else list(initial_squad)
    prices = {} if sell_prices is None else dict(sell_prices)
    previous_fh = previous_free_hit_gw
    acquisition = {} if purchase_prices is None else dict(purchase_prices)
    rows = []
    for gw in range(start_gw, max_gw + 1):
        length = min(horizon, max_gw - gw + 1)
        frame = predictions
        if "origin_gw" in frame:
            frame = frame[frame.origin_gw == gw]
            if frame.empty:
                raise ValueError(f"missing forecast origin {gw}")
        frame = frame[frame.GW.between(gw, gw + length - 1)]
        if frame.empty or not (frame.GW == gw).any():
            raise ValueError(f"missing prediction round {gw}")
        current_bootstrap, current_sells = bootstrap, prices or None
        price_receipt = None
        if origin_price_captures is not None:
            from fpl.milp.origin_prices import prepare_origin_frame, selling_prices
            capture = origin_price_captures[gw]
            frame, price_receipt = prepare_origin_frame(frame, capture, origin_gw=gw, owned=state)
            current_bootstrap = capture["bootstrap"]
            current_sells = selling_prices(capture, state or [], acquisition)
        forced = {chip: [week for week in weeks if gw-offset <= week < gw-offset+length]
                  for chip, weeks in (forced_chips or {}).items()}
        window = solve_window(frame, season=season, start_gw=gw, horizon=length, initial_squad=state,
                              bank=bank, free_transfers=free_transfers, sell_prices=current_sells,
                              bootstrap=current_bootstrap, chip_inventory=inventory, forced_chips=forced,
                              previous_free_hit_gw=previous_fh, time_limit=time_limit)
        row = window.iloc[0].to_dict()
        if price_receipt is not None:
            row["price_policy"] = "origin_capture_half_profit_static_horizon_v1"
            row["pool_policy"] = "registered_forecast_rows"
            row["price_capture_at"] = price_receipt["captured_at"]
            row["price_event_deadline"] = price_receipt["event_deadline"]
            row["price_payload_sha256"] = price_receipt["payload_sha256"]
            row["price_descriptor_sha256"] = price_receipt["descriptor_sha256"]
            row["price_input_receipt"] = price_receipt
            row["solver_outcome_columns"] = "ABSENT"
        rows.append(row)
        played = row["chip_played"]
        if played:
            key = "tc" if played.startswith("TC_") else played.lower()
            inventory[key][1 if gw-offset <= config.MODERN_CHIP_HALF_BOUNDARY else 2] = 0
        if played == "FH":
            previous_fh = gw-offset
        else:
            state = row["squad"]
            prices = {p: price for p, price in prices.items() if p in state and p not in row["transfers_in"]}
            bank = row["budget_end"]
            if origin_price_captures is not None:
                for player in row["transfers_out"]:
                    acquisition.pop(player, None)
                for player in row["transfers_in"]:
                    acquisition[player] = int(capture["metadata"].loc[player, "value"])
        free_transfers = row["q_end"]
    result = pd.DataFrame(rows)
    if origin_price_captures is None and "actual_total_points" in predictions:
        result = score_decisions(result, predictions)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline modern-rule Open FPL optimizer")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--season", required=True)
    parser.add_argument("--start-gw", required=True, type=int)
    parser.add_argument("--max-gw", required=True, type=int)
    parser.add_argument("--horizon", default=3, type=int)
    parser.add_argument("--initial-squad", default="")
    parser.add_argument("--initial-budget", default=1000, type=float)
    parser.add_argument("--initial-ft", default=1, type=int)
    parser.add_argument("--state-json", help="Explicit sell_prices or origin-mode purchase_prices (tenths), chip inventory and optional bootstrap")
    parser.add_argument("--origin-price-captures", type=Path, help="Pinned predeadline register descriptor; decisions-only modern replay")
    parser.add_argument("--time-limit", default=1200, type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decisions-json", type=Path, help="New pre-settlement stable-ID snapshot (origin-price mode)")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.decisions_json is not None:
        if not args.origin_price_captures:
            raise ValueError("--decisions-json requires origin-price mode before settlement")
        if args.decisions_json.exists() or args.decisions_json.resolve() == output.resolve():
            raise ValueError("decision snapshot requires a new path distinct from output")
    state = json.loads(Path(args.state_json).read_text()) if args.state_json else {}
    if "sell_prices" in state:
        state["sell_prices"] = {int(p): price for p, price in state["sell_prices"].items()}
    if "purchase_prices" in state:
        basis = state["purchase_prices"]
        if (not isinstance(basis, dict) or any(not p.isdecimal() or str(int(p)) != p or int(p) <= 0 for p in basis)):
            raise ValueError("purchase_prices requires canonical positive stable-ID keys")
        state["purchase_prices"] = {int(p): price for p, price in basis.items()}
    if args.origin_price_captures:
        from fpl.milp.origin_prices import load_origin_captures
        state["origin_price_captures"] = load_origin_captures(args.origin_price_captures,
            season=args.season, origins=list(range(args.start_gw, args.max_gw + 1)))
    decisions = run_schedule(pd.read_csv(args.predictions_csv), season=args.season, start_gw=args.start_gw,
                             max_gw=args.max_gw, horizon=args.horizon,
                             initial_squad=[int(p) for p in args.initial_squad.split(",")] if args.initial_squad else None,
                             bank=args.initial_budget, free_transfers=args.initial_ft,
                             time_limit=args.time_limit, **state)
    if args.decisions_json is not None:
        from fpl.milp.optimize import write_decision_snapshot
        write_decision_snapshot(decisions, args.decisions_json)
    exported = decisions.copy()
    for col in ("squad", "lineup", "bench_order", "captain", "vice_captain", "transfers_in", "transfers_out",
                "permanent_squad_start", "permanent_squad_end"):
        exported[col] = exported[col].map(json.dumps)
    if "price_input_receipt" in exported:
        exported["price_input_receipt"] = exported.price_input_receipt.map(
            lambda value: json.dumps(value, sort_keys=True, allow_nan=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        exported.to_csv(stream, index=False)
    print(f"Saved {len(decisions)} optimal, independently checked decisions to {output}")
    if args.origin_price_captures:
        print("Origin capture prices; solver outcome columns absent. Settle saved decisions separately.")
    if "actual_total_points" in decisions:
        print(f"Gross={decisions.actual_total_points.sum():.0f}; net={decisions.actual_net_points.sum():.0f}")


if __name__ == "__main__":
    main()
