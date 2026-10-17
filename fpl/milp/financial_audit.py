"""Independent legacy no-chip transfer/bank/free-transfer arithmetic.

This module does not import an optimizer or scorer. Correct arithmetic does not
certify official rules, historical price availability, deadline information or
purchase-price provenance; those remain caller responsibilities.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import pandas as pd

from fpl import config


def _count(value, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} requires an integer >= {minimum}")
    return int(value)


def _money(value, label: str, *, positive: bool = False) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value % 1 or value < (1 if positive else 0)):
        raise ValueError(f"{label} requires finite {'positive' if positive else 'nonnegative'} integer tenths")
    return int(value)


def _ids(value, label: str, *, length: int | None = None) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{label} requires an explicit list of stable IDs")
    ids = [_count(player, label, minimum=1) for player in value]
    if len(set(ids)) != len(ids) or (length is not None and len(ids) != length):
        raise ValueError(f"{label} requires distinct stable IDs" + (f" with length {length}" if length is not None else ""))
    return ids


def _reported(value, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{label} requires a finite numeric reported bank")
    return float(value)


def _purchase_prices(state: dict, squad: set[int], policy: str) -> dict[int, int | None]:
    source = state.get("purchase_prices")
    if source is None:
        if policy == "half_profit" and squad:
            raise ValueError("half_profit requires explicit initial purchase_prices for all owned players")
        return {player: None for player in squad}
    if not isinstance(source, Mapping):
        raise ValueError("initial purchase_prices must be a stable-code mapping")
    result = {}
    for key, value in source.items():
        if isinstance(key, str):
            if not key.isdecimal() or str(int(key)) != key:
                raise ValueError("purchase_prices JSON keys require canonical positive integer codes")
            player = _count(int(key), "purchase_prices stable code", minimum=1)
        else:
            player = _count(key, "purchase_prices stable code", minimum=1)
        if player in result or player not in squad:
            raise ValueError("purchase_prices contains duplicate or unowned stable codes")
        result[player] = _money(value, "purchase price", positive=True)
    if policy == "half_profit" and set(result) != squad:
        raise ValueError("half_profit requires purchase_prices for all fifteen initial holdings")
    return {player: result.get(player) for player in squad}


def audit_legacy_finances(snapshot: dict, prices: pd.DataFrame, *, initial_state: dict,
                          ft_cap: int, ft_gain: int, selling_policy: str = "current_price",
                          compare_reported: bool = True) -> pd.DataFrame:
    """Audit fixed no-chip decisions against explicit prices in integer tenths.

    Sales use current prices or the declared half-profit policy. Acquisitions reset
    purchase basis; initial half-profit holdings require explicit acquisition costs.
    Sensitivity mode still enforces transfers, hits, free transfers and affordability,
    but does not require reconstructed banks to match original reported banks.
    Neither mode proves full financial/rule legality or source/time provenance.
    """
    if selling_policy not in ("current_price", "half_profit") or type(compare_reported) is not bool:
        raise ValueError("explicit supported selling_policy and boolean compare_reported are required")
    cap, gain = _count(ft_cap, "ft_cap", minimum=1), _count(ft_gain, "ft_gain", minimum=1)
    if (not isinstance(initial_state, dict) or not {"mode", "squad", "budget", "free_transfers"} <= set(initial_state)
            or initial_state["mode"] not in ("fresh", "continuing")):
        raise ValueError("explicit initial_state mode, squad, budget and free_transfers are required")
    initial_mode = initial_state["mode"]
    owned = set(_ids(initial_state["squad"], "initial squad", length=0 if initial_mode == "fresh" else 15))
    bank = _money(initial_state["budget"], "initial bank")
    free = _count(initial_state["free_transfers"], "initial free transfers")
    if cap < free:
        raise ValueError("ft_cap must honor the explicit initial free-transfer stack")
    basis = _purchase_prices(initial_state, owned, selling_policy)
    if (not isinstance(snapshot, dict) or type(snapshot.get("schema_version")) is not int
            or snapshot["schema_version"] != 1 or snapshot.get("settlement_completed") is not False
            or not isinstance(snapshot.get("decisions"), list) or not snapshot["decisions"]):
        raise ValueError("a nonempty pre-settlement schema-1 decision snapshot is required")
    if not prices.columns.is_unique or prices.empty or not {"player_id", "GW", "value"} <= set(prices):
        raise ValueError("prices require nonempty unambiguous player_id/GW/value columns")
    ledger = prices[["player_id", "GW", "value"]].copy()
    # Price CSV identities may use integer-valued numeric dtypes, but never strings/booleans.
    for column in ("player_id", "GW"):
        ledger[column] = ledger[column].map(lambda value: _money(value, column, positive=True))
    ledger["value"] = ledger.value.map(lambda value: _money(value, "current price", positive=True))
    if ledger.duplicated(["player_id", "GW"]).any():
        raise ValueError("prices require unique player/gameweek identities")
    price_map = ledger.set_index(["player_id", "GW"]).value.to_dict()
    weeks = [_count(row.get("gameweek"), "decision gameweek", minimum=1)
             for row in snapshot["decisions"] if isinstance(row, dict)]
    if (len(weeks) != len(snapshot["decisions"]) or weeks != list(range(weeks[0], weeks[0] + len(weeks)))):
        raise ValueError("decision gameweeks must be unique, chronological and contiguous")
    results = []
    tolerance = config.MILP_CERTIFICATE_ABS_TOL
    for index, decision in enumerate(snapshot["decisions"]):
        if any(key.startswith("actual_") or key == "gross_actual_total_points" for key in decision):
            raise ValueError("decision snapshot must precede outcome settlement")
        week = weeks[index]
        if decision.get("chip_played") not in (None, ""):
            raise ValueError("financial auditor supports legacy no-chip decisions only")
        squad = set(_ids(decision.get("squad"), "selected squad", length=15))
        ins, outs = set(_ids(decision.get("transfers_in"), "transfers_in")), set(_ids(decision.get("transfers_out"), "transfers_out"))
        alpha = _count(decision.get("alpha"), "alpha")
        reported_ft_start = _count(decision.get("q_start"), "q_start")
        reported_ft_end = _count(decision.get("q_end"), "q_end")
        fresh = index == 0 and initial_mode == "fresh"
        if fresh:
            if ins or outs or alpha:
                raise ValueError("fresh-build snapshot requires empty transfer lists and zero hits")
        elif ins != squad - owned or outs != owned - squad or len(ins) != len(outs) or ins & outs:
            raise ValueError("transfer lists must exactly match the permanent squad change")
        transfers = 0 if fresh else len(ins)
        expected_alpha = max(0, transfers - free)
        next_free = min(cap, max(0, free - transfers) + gain)
        if alpha != expected_alpha or reported_ft_start != free or reported_ft_end != next_free:
            raise ValueError("hits or free-transfer state disagree with independent arithmetic")
        needed = squad | outs
        missing = sorted(player for player in needed if (player, week) not in price_map)
        if missing:
            raise ValueError(f"missing current prices in GW{week} for selected/sold players {missing}")
        prices_now = {player: price_map[(player, week)] for player in needed}
        sale_total = 0
        for player in outs:
            current = prices_now[player]
            purchase = basis.get(player)
            if selling_policy == "half_profit":
                if purchase is None:
                    raise ValueError("half_profit sale lacks an explicit purchase basis")
                sale_total += current if current <= purchase else purchase + (current - purchase) // 2
            else:
                sale_total += current
        acquisitions = squad if fresh else ins
        purchase_total = sum(prices_now[player] for player in acquisitions)
        next_bank = bank + sale_total - purchase_total
        if next_bank < 0:
            raise ValueError(f"fixed decisions are unaffordable in GW{week}: reconstructed bank {next_bank}")
        mode = "fresh" if fresh else "continuing"
        if "state_mode" in decision and decision["state_mode"] != ("fresh_build" if fresh else "continuing"):
            raise ValueError("optional state_mode disagrees with reconstructed mode")
        for field, expected in (("permanent_squad_start", owned), ("permanent_squad_end", squad)):
            if field in decision and set(_ids(decision[field], field, length=len(expected))) != expected:
                raise ValueError(f"optional {field} disagrees with reconstructed holdings")
        reported = {}
        for field, expected in (("budget_start", bank), ("budget_end", next_bank)):
            if compare_reported and field not in decision:
                raise ValueError(f"reported comparison requires {field}")
            reported[field] = _reported(decision[field], field) if field in decision else None
            if compare_reported and abs(reported[field] - expected) > tolerance:
                raise ValueError(f"reported {field} disagrees with reconstructed bank")
        if compare_reported:
            for field, expected in (("permanent_bank_start", bank), ("permanent_bank_end", next_bank)):
                if field in decision and abs(_reported(decision[field], field) - expected) > tolerance:
                    raise ValueError(f"optional {field} disagrees with reconstructed bank")
            if decision.get("temporary_selection_bank_end") is not None:
                raise ValueError("no-chip decisions cannot have a temporary selection bank")
        results.append({"gameweek": week, "mode": mode,
            "permanent_squad_start": sorted(owned), "permanent_squad_end": sorted(squad),
            "derived_bank_start": bank, "derived_bank_end": next_bank,
            "reported_budget_start": reported["budget_start"], "reported_budget_end": reported["budget_end"],
            "bank_start_difference": None if reported["budget_start"] is None else bank - reported["budget_start"],
            "bank_end_difference": None if reported["budget_end"] is None else next_bank - reported["budget_end"],
            "sales_proceeds": sale_total, "purchases_cost": purchase_total, "transfer_count": transfers,
            "alpha": alpha, "transfer_hit_points": config.TRANSFER_HIT_POINTS * alpha,
            "q_start": free, "q_end": next_free, "selling_policy": selling_policy,
            "scope": "LEGACY_NO_CHIP_FINANCIAL_ARITHMETIC" if compare_reported else "FIXED_DECISION_SELLING_POLICY_SENSITIVITY",
            "arithmetic_assessment": "PASS"})
        for player in outs:
            basis.pop(player, None)
        for player in acquisitions:
            basis[player] = prices_now[player]
        owned, bank, free = squad, next_bank, next_free
    return pd.DataFrame(results)
