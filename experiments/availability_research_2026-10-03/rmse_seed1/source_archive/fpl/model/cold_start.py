"""Opt-in empirical fixture priors for registered players without snapshot rows.

These spreads describe historical outcomes, not calibrated prediction intervals.
Synthetic player rows are forecast inputs only and must never enter training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fpl import config

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
PRIOR_COLUMNS = ["position", "team", "predicted_total_points", "prior_std", "p_played", "p60", "prior_count"]


def fit_pooled_priors(
    history: pd.DataFrame, origin_gw: int,
    prior_strength: float = config.COLD_START_PRIOR_STRENGTH,
) -> pd.DataFrame:
    """Mix team-position and position fixture distributions before ``origin_gw``.

    Include DNP outcomes. Team weight is n/(n+strength); spread uses the mixed
    second moment, with population variance. A null team denotes the pooled prior.
    """
    if isinstance(origin_gw, bool) or not np.isfinite(origin_gw) or origin_gw < 1 or int(origin_gw) != origin_gw:
        raise ValueError("origin_gw must be a positive integer")
    if not np.isfinite(prior_strength) or prior_strength <= 0:
        raise ValueError("prior_strength must be finite and positive")
    required = {"player_id", "position", "team", "GW_global", "total_points", "minutes"}
    if missing := required.difference(history.columns):
        raise ValueError(f"Prior history missing columns: {sorted(missing)}")
    rounds = pd.to_numeric(history["GW_global"], errors="coerce")
    if not np.isfinite(rounds).all() or (rounds < 1).any() or (rounds % 1 != 0).any():
        raise ValueError("Prior history has invalid GW_global")
    past = history.loc[rounds < origin_gw].copy()
    if past.empty:
        return pd.DataFrame(columns=PRIOR_COLUMNS)
    for col in ("player_id", "position", "team"):
        if past[col].isna().any():
            raise ValueError(f"Prior history has missing {col}")
    if not past["position"].isin(POSITIONS.values()).all() or (past["team"].astype(str).str.strip() == "").any():
        raise ValueError("Prior history has invalid position/team identity")
    ids = pd.to_numeric(past["player_id"], errors="coerce")
    if not np.isfinite(ids).all() or (ids <= 0).any() or (ids % 1 != 0).any():
        raise ValueError("Prior history has invalid stable player codes")
    for col in ("total_points", "minutes"):
        past[col] = pd.to_numeric(past[col], errors="coerce")
        if not np.isfinite(past[col]).all():
            raise ValueError(f"Prior history has nonfinite {col}")
    if (past["minutes"] < 0).any():
        raise ValueError("Prior history has negative minutes")
    past["second"] = past["total_points"] ** 2
    past["played"] = (past["minutes"] > 0).astype(float)
    past["sixty"] = (past["minutes"] >= 60).astype(float)
    agg = {"total_points": "mean", "second": "mean", "played": "mean", "sixty": "mean", "minutes": "size"}
    pooled = past.groupby("position").agg(agg)
    teams = past.groupby(["position", "team"]).agg(agg)
    rows = []
    for position, stats in pooled.iterrows():
        rows.append([position, None, stats.total_points, np.sqrt(max(0, stats.second - stats.total_points ** 2)), stats.played, stats.sixty, int(stats.minutes)])
    # Only four positions and team-position groups; this is aggregation, not row ETL.
    for (position, team), stats in teams.iterrows():
        base = pooled.loc[position]
        weight = stats.minutes / (stats.minutes + prior_strength)
        mixed = weight * stats + (1 - weight) * base
        rows.append([position, team, mixed.total_points, np.sqrt(max(0, mixed.second - mixed.total_points ** 2)), mixed.played, mixed.sixty, int(stats.minutes)])
    return pd.DataFrame(rows, columns=PRIOR_COLUMNS)


def add_registered_cold_starts(snapshot: pd.DataFrame, bootstrap: dict) -> pd.DataFrame:
    """Append registered stable codes absent from snapshot with unknown form/outcomes.

    Known rows retain all their values, with an added false cold-start marker.
    Current identity and price come directly from the validated API register.
    """
    if "player_id" not in snapshot or snapshot["player_id"].isna().any() or snapshot["player_id"].duplicated().any():
        raise ValueError("Snapshot requires unique nonmissing stable player codes")
    ids = pd.to_numeric(snapshot["player_id"], errors="coerce")
    if not np.isfinite(ids).all() or (ids <= 0).any() or (ids % 1 != 0).any():
        raise ValueError("Snapshot requires positive integer stable player codes")
    teams = {}
    for team in bootstrap["teams"]:
        if team["id"] in teams or not team.get("name"):
            raise ValueError("Duplicate or invalid API team mapping")
        teams[team["id"]] = config.TEAM_NAME_CORRECTIONS.get(team["name"], team["name"])
    seen = set()
    additions = []
    existing = set(snapshot["player_id"])
    for element in bootstrap["elements"]:
        code = element.get("code")
        if isinstance(code, bool) or not isinstance(code, (int, np.integer)) or code <= 0 or code in seen:
            raise ValueError("API register requires unique positive stable player codes")
        seen.add(code)
        if element.get("team") not in teams or element.get("element_type") not in POSITIONS:
            raise ValueError(f"Invalid API team/position mapping for code {code}")
        price = element.get("now_cost")
        if price is None or not np.isfinite(price) or price <= 0:
            raise ValueError(f"Invalid API current price for code {code}")
        if code in existing:
            continue
        row = dict.fromkeys(snapshot.columns, np.nan)
        row.update(player_id=code, position=POSITIONS[element["element_type"]], team=teams[element["team"]],
                   name=" ".join(str(element.get(k, "")) for k in ("first_name", "second_name")).strip() or element.get("web_name", str(code)),
                   value=float(price), total_points=np.nan, minutes=np.nan, is_cold_start=True)
        additions.append(row)
    result = snapshot.copy()
    result["is_cold_start"] = False
    if additions:
        result = pd.concat([result, pd.DataFrame(additions)], ignore_index=True)
    return result


def predict_cold_starts(frame: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    """Resolve team-position priors, falling back to pooled position history.

    Only position/team identity is read from inputs; target outcomes are ignored.
    Returned rows preserve input order/index, including duplicated fixture indices.
    """
    values = ["predicted_total_points", "prior_std", "p_played", "p60", "prior_count"]
    if missing := set(PRIOR_COLUMNS).difference(priors.columns):
        raise ValueError(f"Priors missing columns: {sorted(missing)}")
    if priors.duplicated(["position", "team"]).any():
        raise ValueError("Ambiguous duplicated prior identity")
    pooled = priors[priors.team.isna()].set_index("position")
    missing_positions = set(frame.position).difference(pooled.index)
    if missing_positions:
        raise ValueError(f"No pooled history for positions: {sorted(missing_positions)}")
    lookup = priors[priors.team.notna()].set_index(["position", "team"])
    keys = pd.MultiIndex.from_frame(frame[["position", "team"]])
    result = lookup.reindex(keys)[values].reset_index(drop=True)
    fallback = pooled.reindex(frame.position)[values].reset_index(drop=True)
    result = result.fillna(fallback)
    result.index = frame.index
    return result


def retrospective_cold_start_predictions(
    raw: pd.DataFrame, start_gw: int, end_gw: int,
) -> pd.DataFrame:
    """Forecast first-recorded player rounds using strictly earlier fixture priors.

    The cohort includes DNPs and every fixture in a player's first recorded DGW.
    This retrospective diagnostic does not establish historical registration or
    optimizer performance; empirical spreads are not calibrated intervals.
    Rows later than ``end_gw`` are excluded before cohort grouping or fitting.
    """
    for bound in (start_gw, end_gw):
        if isinstance(bound, bool) or not np.isfinite(bound) or bound < 1 or int(bound) != bound:
            raise ValueError("Cold-start diagnostic bounds must be positive integers")
    if start_gw > end_gw:
        raise ValueError("Cold-start diagnostic requires start_gw <= end_gw")
    required = {"player_id", "position", "team", "GW_global", "total_points", "minutes"}
    if missing := required.difference(raw.columns):
        raise ValueError(f"Cold-start diagnostic missing columns: {sorted(missing)}")
    identity_columns = ["player_id", "position", "team", "GW_global"]
    if "fixture" in raw:
        identity_columns.append("fixture")
    output_columns = identity_columns + ["actual_total_points", "minutes"] + PRIOR_COLUMNS[2:] + ["origin_gw"]
    rounds = pd.to_numeric(raw["GW_global"], errors="coerce")
    if not np.isfinite(rounds).all() or (rounds < 1).any() or (rounds % 1 != 0).any():
        raise ValueError("Cold-start diagnostic has invalid GW_global")
    admitted = raw.loc[rounds <= end_gw].copy()
    admitted["GW_global"] = rounds.loc[rounds <= end_gw]
    if admitted["player_id"].isna().any():
        raise ValueError("Cold-start diagnostic has missing player identity")
    first_round = admitted.groupby("player_id")["GW_global"].transform("min")
    cohort = admitted.loc[(admitted["GW_global"] == first_round) & (admitted["GW_global"] >= start_gw)]
    if cohort.empty:
        return pd.DataFrame(columns=output_columns)
    rows = []
    # One fitting operation per origin round; no player-level iteration or synthetic history.
    for gw, target in cohort.groupby("GW_global", sort=True):
        target = target.reset_index(drop=True)
        for column in ("total_points", "minutes"):
            numeric = pd.to_numeric(target[column], errors="coerce")
            if not np.isfinite(numeric).all() or (column == "minutes" and (numeric < 0).any()):
                raise ValueError(f"Cold-start diagnostic has invalid target {column}")
            target[column] = numeric
        prediction = predict_cold_starts(target[["position", "team"]], fit_pooled_priors(admitted, int(gw)))
        result = target[identity_columns + ["total_points", "minutes"]].rename(columns={"total_points": "actual_total_points"})
        result = pd.concat([result, prediction], axis=1)
        result["origin_gw"] = int(gw)
        rows.append(result)
    return pd.concat(rows, ignore_index=True)[output_columns]
