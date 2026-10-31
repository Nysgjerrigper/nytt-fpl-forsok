"""Validate upstream decisions and translate element IDs to stable player codes."""

from collections import Counter
from itertools import combinations
from math import isfinite
from numbers import Real

import pandas as pd

from fpl import config


def _integer(value: object, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be an integer")
    number = float(value)
    if not isfinite(number) or not number.is_integer() or not low <= number <= high:
        raise ValueError(f"Invalid {label}: {value}")
    return int(number)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{label} must be finite numeric data")
    return float(value)


def normalize_solution(solution: dict, prepared: dict) -> list[dict]:
    """Check squad, transfer, budget and modern FT state against prepared inputs.

    Prices and banks in the upstream interface are pounds in millions. Returned
    banks use the legacy interface's integer tenths; IDs use stable FPL codes.
    Patched Bench Boost decisions retain a legal XI and four bench players.
    Legacy upstream 15-player lineups can also be converted to a legal XI for
    our realized scorer, which separately awards all four bench players' points.
    """
    picks = solution.get("picks")
    required = {"id", "week", "squad", "lineup", "bench", "captain", "vicecaptain",
                "transfer_in", "transfer_out", "chip"}
    if not isinstance(picks, pd.DataFrame) or picks.empty or not required <= set(picks):
        raise ValueError("Solution picks must contain all decision columns")
    picks = picks.copy()
    for column in required - {"chip"}:
        low, high = (1, 10**12) if column == "id" else ((1, 38) if column == "week"
                    else ((-1, 3) if column == "bench" else (0, 1)))
        picks[column] = [_integer(v, column, low, high) for v in picks[column]]
    if picks.duplicated(["id", "week"]).any():
        raise ValueError("Duplicate player-week decisions")
    if not picks["chip"].isin(["", "WC", "FH", "BB", "TC"]).all():
        raise ValueError("Unknown chip in picks")
    metadata = prepared["merged_data"]
    mapping = prepared["element_to_stable"]
    if (not isinstance(metadata, pd.DataFrame) or not metadata.index.is_unique
            or not {"element_type", "name"} <= set(metadata)):
        raise ValueError("Player metadata must have a unique element index")
    positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    ids = set(picks["id"]) | set(prepared["initial_squad"])
    if not ids <= set(mapping) or not ids <= set(metadata.index):
        raise ValueError("Missing player identity metadata")
    stable = {p: _integer(mapping[p], "stable player ID", 1, 10**12) for p in ids}
    if len(set(stable.values())) != len(stable):
        raise ValueError("Stable player IDs must be unique")
    pos = {p: positions[_integer(metadata.loc[p, "element_type"], "position", 1, 4)] for p in ids}
    clubs = {p: metadata.loc[p, "name"] for p in ids}
    if any(pd.isna(v) or not str(v).strip() for v in clubs.values()):
        raise ValueError("Missing club metadata")
    if not ids <= set(prepared["buy_price"]):
        raise ValueError("Missing buy price for a selected or owned player")
    if not set(prepared["initial_squad"]) <= set(prepared["sell_price"]):
        raise ValueError("Missing initial owned-player selling price")
    buy = {p: _number(prepared["buy_price"][p], "buy price") for p in ids}
    sell = {p: _number(prepared["sell_price"].get(p, buy[p]), "sell price") for p in ids}
    if any(v <= 0 for v in buy.values()) or any(sell[p] <= 0 or sell[p] > buy[p] for p in ids):
        raise ValueError("Invalid player prices")
    permanent = set(prepared["initial_squad"])
    if len(permanent) != len(prepared["initial_squad"]) or len(permanent) not in (0, 15):
        raise ValueError("Initial squad must be empty or contain 15 distinct players")
    unsold_originals = permanent.copy()
    bank = _number(prepared["itb"], "initial bank")
    if bank < 0:
        raise ValueError("Initial bank cannot be negative")
    ft = _integer(prepared["ft"], "initial free transfers", 0, 5)
    weeks = sorted(set(picks["week"]))
    if weeks != list(range(prepared["next_gw"], max(weeks) + 1)):
        raise ValueError("Decision weeks must be contiguous from next_gw")
    score = _number(solution["score"], "objective score")
    results = []

    def legal_xi(players: set[int]) -> bool:
        counts = Counter(pos[p] for p in players)
        return len(players) == 11 and all(config.LINEUP_MIN_COUNTS[k] <= counts[k]
            <= config.LINEUP_MAX_COUNTS[k] for k in config.LINEUP_MIN_COUNTS)

    for week in weeks:
        rows = picks[picks["week"] == week]
        stat = solution["statistics"].get(week)
        if not isinstance(stat, dict):
            raise ValueError(f"Missing statistics for GW{week}")
        chip = stat.get("chip")
        chip = "" if chip is None else chip
        if chip not in ("", "WC", "FH", "BB", "TC"):
            raise ValueError("Unknown statistics chip")
        row_chips = set(rows["chip"]) - {""}
        if row_chips != ({chip} if chip else set()):
            raise ValueError("Picks and statistics chips disagree")
        squad = set(rows.loc[rows.squad == 1, "id"])
        lineup = set(rows.loc[rows.lineup == 1, "id"])
        captains = rows.loc[rows.captain == 1, "id"].tolist()
        vice = rows.loc[rows.vicecaptain == 1, "id"].tolist()
        ins = set(rows.loc[rows.transfer_in == 1, "id"])
        outs = set(rows.loc[rows.transfer_out == 1, "id"])
        if Counter(pos[p] for p in squad) != Counter({"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}):
            raise ValueError("Squad must have the 2/5/5/3 position structure")
        if max(Counter(clubs[p] for p in squad).values()) > 3:
            raise ValueError("Squad exceeds three players from a club")
        if len(captains) != 1 or len(vice) != 1 or captains[0] == vice[0]:
            raise ValueError("Exactly one distinct captain and vice captain required")
        required_captains = set(captains + vice)
        if not required_captains <= lineup or not lineup <= squad:
            raise ValueError("Captain, vice or lineup outside selected squad")
        if chip == "BB" and len(lineup) == 15:
            if lineup != squad or not (rows.bench == -1).all():
                raise ValueError("Legacy Bench Boost lineup must contain the whole squad")
            if f"{week}_Pts" not in metadata:
                raise ValueError("Missing Bench Boost reconstruction forecasts")
            forecasts = {p: _number(metadata.loc[p, f"{week}_Pts"], "forecast") for p in squad}
            candidates = [set(c) for c in combinations(sorted(squad), 11)
                          if required_captains <= set(c) and legal_xi(set(c))]
            if not candidates:
                raise ValueError("No legal Bench Boost XI containing captain and vice")
            lineup = max(candidates, key=lambda c: (sum(forecasts[p] for p in c), tuple(sorted(c))))
            remaining = squad - lineup
            bench = sorted(remaining, key=lambda p: (pos[p] != "GK", -forecasts[p], p))
        else:
            bench_rows = rows[rows.bench >= 0]
            if len(bench_rows) != 4 or set(bench_rows.bench) != set(range(4)):
                raise ValueError("Bench orders must cover slots 0 through 3 once")
            bench = bench_rows.sort_values("bench")["id"].tolist()
        if not legal_xi(lineup) or set(bench) != squad - lineup or pos[bench[0]] != "GK":
            raise ValueError("Illegal XI or bench composition")
        if any(pos[p] == "GK" for p in bench[1:]):
            raise ValueError("Only bench slot zero may contain a goalkeeper")
        reported_ft = _integer(stat["ft"], "statistics free transfers", 0, 5)
        hits = _integer(stat["pt"], "penalized transfers", 0, 15)
        transfer_count = _integer(stat["nt"], "transfer count", 0, 15)
        fresh = not permanent and week == weeks[0]
        expected_hits = 0 if chip in ("WC", "FH") or fresh else max(0, len(ins) - ft)
        if reported_ft != ft or hits != expected_hits or transfer_count != len(outs):
            raise ValueError("Transfer statistics disagree with independently derived state")
        if ins & outs or not outs <= permanent or ins & permanent:
            raise ValueError("Invalid permanent transfers")
        sale_price = lambda p: sell[p] if p in unsold_originals else buy[p]
        next_bank = bank
        if chip == "FH":
            if ins or outs or fresh:
                raise ValueError("Free Hit cannot make permanent transfers or build a fresh team")
            temporary_cost = sum(sale_price(p) if p in permanent else buy[p] for p in squad)
            wealth = bank + sum(sale_price(p) for p in permanent)
            if temporary_cost > wealth + 1e-5:
                raise ValueError("Free Hit temporary squad exceeds available wealth")
        else:
            if squad != (permanent - outs) | ins:
                raise ValueError("Squad does not match permanent transfer state")
            next_bank += sum(sale_price(p) for p in outs) - sum(buy[p] for p in ins)
        reported_bank = _number(stat["itb"], "statistics bank")
        if next_bank < -1e-5 or abs(reported_bank - next_bank) > 1e-5:
            raise ValueError("Bank disagrees with transfer-price arithmetic")
        next_ft = max(1, ft) if chip in ("WC", "FH") else max(1, min(5, ft - len(ins) + 1))
        to_stable = lambda players: [stable[p] for p in players]
        captain_name = (metadata.loc[captains[0], "web_name"]
                        if "web_name" in metadata else str(captains[0]))
        results.append({"gameweek": week + prepared["season_offset"],
            "squad": to_stable(sorted(squad)), "lineup": to_stable(sorted(lineup)),
            "bench_order": to_stable(bench), "captain": to_stable(captains),
            "vice_captain": to_stable(vice), "transfers_in": to_stable(sorted(ins)),
            "transfers_out": to_stable(sorted(outs)), "budget_start": round(bank * 10, 6),
            "budget_end": round(next_bank * 10, 6), "alpha": hits, "q_start": ft,
            "q_end": next_ft, "objective_value": score,
            "chip_played": f"TC_{captain_name}" if chip == "TC" else chip or None})
        if chip != "FH":
            permanent = squad
            unsold_originals -= outs
            bank = max(0.0, next_bank)
        ft = next_ft
    return results
