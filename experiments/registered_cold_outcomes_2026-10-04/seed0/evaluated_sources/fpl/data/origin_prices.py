"""Strictly lagged observed player prices for opt-in origin research.

This catalog is a lagged observed-price proxy. It does not prove archived source
publication or the actual price available at a historical deadline. Frozen input
admission and registered research windows remain the calling runner's responsibility.
"""
from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import numpy as np
import pandas as pd

_COLUMNS = ["player_id", "origin_gw", "value", "price_observed_gw"]


def _integers(values: pd.Series, label: str) -> pd.Series:
    """Validate numeric whole positive values, including rejecting mixed booleans."""
    if values.empty:
        return pd.Series(index=values.index, dtype="int64", name=values.name)
    canonical = values.astype("string").str.fullmatch(r"[+]?\d+(?:\.0+)?", na=False)
    if not canonical.all() or not pd.api.types.is_numeric_dtype(values.dtype):
        raise ValueError(f"{label} requires positive integer numeric IDs/gameweeks")
    numeric = pd.to_numeric(values, errors="coerce")
    if not np.isfinite(numeric).all() or (numeric <= 0).any() or (numeric % 1 != 0).any():
        raise ValueError(f"{label} requires positive integer numeric IDs/gameweeks")
    return numeric.astype("int64")


def _requested(values: Iterable[int], label: str) -> np.ndarray:
    try:
        sequence = list(values)
    except TypeError as exc:
        raise ValueError(f"{label} must be an iterable of positive integer IDs/gameweeks") from exc
    if not sequence:
        raise ValueError(f"{label} must be nonempty")
    return np.sort(_integers(pd.Series(sequence), label).unique())


def _prices(values: pd.Series) -> None:
    if values.empty:
        return
    if pd.api.types.is_bool_dtype(values) or not pd.api.types.is_numeric_dtype(values):
        raise ValueError("Prices must be finite positive numeric values")
    numeric = pd.to_numeric(values, errors="coerce")
    if not np.isfinite(numeric).all() or (numeric <= 0).any():
        raise ValueError("Prices must be finite positive numeric values")


def build_origin_price_catalog(
    raw: pd.DataFrame, player_ids: Iterable[int], start_gw: int, end_gw: int,
) -> pd.DataFrame:
    """Choose the latest observed player price strictly before every requested origin.

    Deduplicate identical same-round fixture prices and fail on conflicts. Missing
    prior prices omit that player/origin rather than using a future value or fallback.
    Rows at/after end_gw are excluded before price grouping, and only requested stable
    codes contribute. The caller admits frozen bytes and its registered window.
    """
    for bound in (start_gw, end_gw):
        if isinstance(bound, bool) or not isinstance(bound, Integral) or bound < 1:
            raise ValueError("Origin-price bounds must be positive integer gameweeks")
    if start_gw > end_gw:
        raise ValueError("Origin-price start_gw must not exceed end_gw")
    if not isinstance(raw, pd.DataFrame) or not raw.columns.is_unique:
        raise ValueError("Origin-price input requires a DataFrame with unique columns")
    if missing := {"player_id", "GW_global", "value"}.difference(raw.columns):
        raise ValueError(f"Origin-price input missing columns: {sorted(missing)}")
    requested = _requested(player_ids, "player_ids")
    rows = raw[["player_id", "GW_global", "value"]].copy()
    rows["player_id"] = _integers(rows.player_id, "player_id")
    rows["GW_global"] = _integers(rows.GW_global, "GW_global")
    rows = rows.loc[(rows.GW_global < end_gw) & rows.player_id.isin(requested)]
    _prices(rows.value)
    if rows.groupby(["player_id", "GW_global"]).value.nunique().gt(1).any():
        raise ValueError("Ambiguous prices across fixtures in the same player round")
    historical = rows.drop_duplicates(["player_id", "GW_global"]).rename(columns={"GW_global": "price_observed_gw"})
    if historical.empty:
        return pd.DataFrame(columns=_COLUMNS)
    grid = pd.MultiIndex.from_product([range(start_gw, end_gw + 1), requested], names=["origin_gw", "player_id"]).to_frame(index=False)
    # merge_asof compares time globally, with exact stable-code matching inside each origin.
    catalog = pd.merge_asof(grid.sort_values(["origin_gw", "player_id"]),
                            historical.sort_values(["price_observed_gw", "player_id"]),
                            left_on="origin_gw", right_on="price_observed_gw", by="player_id",
                            direction="backward", allow_exact_matches=False)
    catalog = catalog.loc[catalog.value.notna()].copy()
    catalog["price_observed_gw"] = catalog.price_observed_gw.astype("int64")
    return catalog[_COLUMNS].sort_values(["origin_gw", "player_id"]).reset_index(drop=True)


def validate_origin_price_catalog(
    catalog: pd.DataFrame, player_ids: Iterable[int], origins: Iterable[int],
) -> pd.DataFrame:
    """Validate a unique lagged-price catalog and return its canonical four columns.

    Every requested origin must have at least one known price; players with unknown
    prices remain ineligible rather than receiving invented defaults. Extra source
    metadata belongs in separate registration, not in this optimizer catalog.
    """
    if not isinstance(catalog, pd.DataFrame) or not catalog.columns.is_unique:
        raise ValueError("Origin-price catalog requires a DataFrame with unique columns")
    if set(catalog.columns) != set(_COLUMNS):
        raise ValueError("Origin-price catalog requires exactly the four canonical columns")
    requested_players = _requested(player_ids, "player_ids")
    requested_origins = _requested(origins, "origins")
    result = catalog[_COLUMNS].copy()
    for column in ("player_id", "origin_gw", "price_observed_gw"):
        result[column] = _integers(result[column], column)
    _prices(result.value)
    if (result.price_observed_gw >= result.origin_gw).any():
        raise ValueError("Observed price gameweek must strictly precede its origin")
    if result.duplicated(["player_id", "origin_gw"]).any():
        raise ValueError("Duplicate player/origin price catalog entry")
    if not result.player_id.isin(requested_players).all() or not result.origin_gw.isin(requested_origins).all():
        raise ValueError("Origin-price catalog contains unknown player IDs/origins")
    if set(result.origin_gw) != set(requested_origins):
        raise ValueError("Every requested origin requires a nonempty price catalog")
    return result.sort_values(["origin_gw", "player_id"]).reset_index(drop=True)
