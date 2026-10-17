"""Exact-coverage player-round forecast diagnostics from saved paired panels.

These diagnostics describe the supplied population. They do not establish training
cutoffs, causal validity, original model coverage, realized captaincy or promotion.
"""
from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from fpl import config
from fpl.model import metrics


def _numeric(series: pd.Series, name: str, *, integer: bool = False) -> pd.Series:
    """Admit finite numeric values, retaining exact integer identity when required."""
    if series.map(lambda value: isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)).any():
        raise ValueError(f"{name} must contain numeric values, not booleans or missing values")
    values = pd.to_numeric(series, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must contain finite values")
    if integer:
        if ((values <= 0) | (values >= 2**63) | (values % 1 != 0)).any():
            raise ValueError(f"{name} must contain positive integers")
        try:
            return values.astype("int64")
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{name} integer identity is out of range") from exc
    return values.astype(float)


def _prepare(frame: pd.DataFrame, protocol: str, grain: str, weeks: list[int]) -> tuple[pd.DataFrame, list[str]]:
    if not frame.columns.is_unique:
        raise ValueError("forecast panel has duplicate column labels")
    required = {"player_id", "GW", "position", "predicted_total_points", "actual_total_points"}
    origin = protocol == "origin"
    if origin:
        required.add("origin_gw")
    elif "origin_gw" in frame.columns:
        raise ValueError("standard panels cannot contain origin_gw")
    if grain == "fixture":
        required.add("fixture")
    if frame.empty or not required <= set(frame.columns):
        raise ValueError("forecast panel is empty or missing required fields")
    result = frame.loc[:, sorted(required)].copy()
    integers = ["player_id", "GW"] + (["origin_gw"] if origin else []) + (["fixture"] if grain == "fixture" else [])
    for column in integers:
        result[column] = _numeric(result[column], column, integer=True)
    for column in ("predicted_total_points", "actual_total_points"):
        result[column] = _numeric(result[column], column)
    if not result.position.isin(config.ONFIELD_POSITIONS).all():
        raise ValueError("forecast positions must be known on-field positions")
    if sorted(result.GW.unique().tolist()) != sorted(weeks):
        raise ValueError("forecast panel gameweeks must match requested exact coverage")
    if origin and (result.origin_gw > result.GW).any():
        raise ValueError("forecast origins cannot follow their target round")
    keys = (["origin_gw"] if origin else []) + ["player_id", "GW"]
    full_keys = keys + (["fixture"] if grain == "fixture" else [])
    if result.duplicated(full_keys).any():
        raise ValueError("forecast panel contains duplicate or ambiguous identities")
    if (result.groupby(keys).position.nunique() != 1).any():
        raise ValueError("position differs within a player-round")
    return result.sort_values(full_keys).reset_index(drop=True), keys


def _aggregate(frame: pd.DataFrame, keys: list[str], grain: str) -> pd.DataFrame:
    if grain == "fixture":
        result = frame.groupby(keys, as_index=False).agg(
            position=("position", "first"), predicted_total_points=("predicted_total_points", "sum"),
            actual_total_points=("actual_total_points", "sum"), fixture_rows=("fixture", "size"))
    else:
        result = frame.copy()
    if not np.isfinite(result[["predicted_total_points", "actual_total_points"]].to_numpy()).all():
        raise ValueError("aggregated forecast or actual points are nonfinite")
    if "origin_gw" in keys:
        shared = result.groupby(["player_id", "GW"])
        if (shared.actual_total_points.nunique() != 1).any():
            raise ValueError("realized targets disagree across forecast origins")
        if (shared.position.nunique() != 1).any():
            raise ValueError("target positions disagree across forecast origins")
    return result


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _stats(frame: pd.DataFrame, grain: str) -> dict[str, Any]:
    ordered = frame.sort_values("player_id", kind="stable").reset_index(drop=True).copy()
    columns = ["position", "GW"] + (["origin_gw"] if "origin_gw" in ordered else [])
    ordered["ranking_group"] = ordered[columns].astype(str).agg("/".join, axis=1)
    grouped = ordered.groupby("ranking_group", sort=True)
    sizes = grouped.size()
    truth_range = grouped.actual_total_points.max() - grouped.actual_total_points.min()
    prediction_range = grouped.predicted_total_points.max() - grouped.predicted_total_points.min()
    used = int(((sizes >= 3) & (truth_range > 0) & (prediction_range > 0)).sum())
    maxima = grouped.predicted_total_points.transform("max")
    tied = int(ordered.loc[ordered.predicted_total_points == maxima].groupby("ranking_group").size().gt(1).sum())
    true, predicted = ordered.actual_total_points, ordered.predicted_total_points
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return {
            "player_rounds": len(ordered),
            "fixture_rows": int(ordered.fixture_rows.sum()) if grain == "fixture" else None,
            "ranking_groups": len(sizes), "spearman_used_groups": used,
            "spearman_skipped_groups": len(sizes) - used, "tied_top_groups": tied,
            "mae": _finite(metrics.mae(true, predicted)), "rmse": _finite(metrics.rmse(true, predicted)),
            "bias": _finite(metrics.bias(true, predicted)),
            "total_calibration": _finite(metrics.total_calibration(true, predicted)),
            "spearman": _finite(metrics.spearman_by_group(true, predicted, ordered.ranking_group)),
            "top1_capture": _finite(metrics.top1_capture(ordered, "ranking_group", "actual_total_points", "predicted_total_points")),
            "aggregate_true_total": _finite(float(true.sum())),
            "top1_true_max_sum": _finite(float(grouped.actual_total_points.max().sum())),
        }


def _summary(frame: pd.DataFrame, grain: str) -> dict[str, Any]:
    return {"overall": _stats(frame, grain), "by_position": {
        position: _stats(frame.loc[frame.position == position], grain)
        for position in config.ONFIELD_POSITIONS if (frame.position == position).any()
    }}


def summarize_forecast_pair(candidate: pd.DataFrame, control: pd.DataFrame, *,
                            information_protocol: str, grain: str, gameweeks: list[int],
                            population: str) -> dict[str, Any]:
    """Summarize exact paired forecast coverage, aggregating fixtures before scoring.

    Origin forecasts are reported separately by GW minus origin_gw. No repeated-target
    pooled metric or training-scaled MASE is inferred from an evaluation panel.
    """
    if information_protocol not in ("standard", "origin") or grain not in ("player_round", "fixture"):
        raise ValueError("unknown information protocol or forecast grain")
    if not isinstance(population, str) or not population.strip():
        raise ValueError("forecast population requires a descriptive label")
    if (not isinstance(gameweeks, list) or not gameweeks or
            any(type(week) is not int or week <= 0 for week in gameweeks) or len(set(gameweeks)) != len(gameweeks)):
        raise ValueError("gameweeks must be explicit unique positive integers")
    a, keys = _prepare(candidate, information_protocol, grain, gameweeks)
    b, _ = _prepare(control, information_protocol, grain, gameweeks)
    comparison = keys + (["fixture"] if grain == "fixture" else []) + ["position", "actual_total_points"]
    if not a[comparison].equals(b[comparison]):
        raise ValueError("candidate/control full keys, positions and realized outcomes must match exactly")
    a, b = _aggregate(a, keys, grain), _aggregate(b, keys, grain)
    summaries = []
    leads = sorted((a.GW - a.origin_gw).unique().tolist()) if information_protocol == "origin" else [None]
    for lead in leads:
        left = a.loc[a.GW - a.origin_gw == lead] if lead is not None else a
        right = b.loc[b.GW - b.origin_gw == lead] if lead is not None else b
        summaries.append({"lead": int(lead) if lead is not None else None,
                          "candidate": _summary(left, grain), "control": _summary(right, grain)})
    return {
        "status": "MEASURED", "metric_version": "forecast_player_round_v1",
        "information_protocol": information_protocol, "grain": grain, "gameweeks": sorted(gameweeks),
        "population": population, "coverage_comparison": "EXACT_MATCH",
        "original_model_coverage": "NOT ASSESSED",
        "training_scaled_mase": {"status": "NOT MEASURED", "reason": "Training-only scales and cutoffs were not supplied."},
        "summaries": summaries,
        "notes": ["Ratios are undefined when their reported denominator is zero; negative outcomes can yield ratios outside [0, 1].",
                  "Top1 capture is a within-position ranking diagnostic, not realized captaincy.",
                  "These metrics do not establish causal validity or promotion eligibility."],
    }
