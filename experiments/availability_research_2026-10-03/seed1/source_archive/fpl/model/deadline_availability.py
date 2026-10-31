"""Pure research features from admitted pre-deadline availability captures.

The caller independently admits archive proofs. A capture before the round's
deadline may legitimately describe that round without shifted outcomes; this
transform reads no outcomes. It makes no price, roster completeness, or model
improvement claim and never substitutes unavailable source data with zeros.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fpl import config

_STATUSES = ("a", "d", "i", "s", "u", "n")
_ROSTER_COLUMNS = (
    "player_id", "origin_gw", "status", "chance_of_playing_next_round",
    "snapshot_age_hours", "captured_at", "deadline",
)


def availability_columns() -> list[str]:
    """Return the fixed ordered eight-column research feature schema."""
    return [f"archive_status_{status}" for status in _STATUSES] + [
        "archive_chance_next", "archive_capture_age_hours",
    ]


def _keys(values: pd.Series, name: str) -> pd.Series:
    if values.empty:
        return pd.Series(index=values.index, dtype="int64")
    # Canonical numeric text rejects booleans even inside mixed object arrays.
    canonical = values.astype("string").str.fullmatch(r"[+]?\d+(?:\.0+)?", na=False)
    numeric = pd.to_numeric(values, errors="coerce")
    if (not canonical.all() or not np.isfinite(numeric).all()
            or (numeric <= 0).any() or (numeric % 1 != 0).any()
            or (numeric > np.iinfo(np.int64).max).any()):
        raise ValueError(f"{name} must contain positive whole IDs/gameweeks excluding booleans")
    return numeric.astype("int64")


def _timestamps(values: pd.Series, name: str) -> pd.Series:
    if values.empty:
        return pd.Series(index=values.index, dtype="datetime64[ns, UTC]")
    text = values.astype("string")
    explicit_zone = text.str.contains(r"[T ]\d{2}:\d{2}.*(?:Z|[+-]\d{2}:?\d{2})$", na=False)
    parsed = pd.to_datetime(values, format="mixed", utc=True, errors="coerce")
    if not explicit_zone.all() or parsed.isna().any():
        raise ValueError(f"{name} must contain valid timezone-aware timestamps")
    return parsed


def add_availability_features(frame: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Join admitted availability by stable code and same-round origin gameweek.

    Preserve all fixture rows, their order/index, and unrelated columns. Missing
    player/origin archive coverage produces eight NaNs. Every supplied roster row
    must be valid, unique and captured strictly before its deadline, at most the
    configured 24 hours earlier. Claimed age must match timestamp elapsed hours.
    The caller verifies source evidence and freezes these inputs at each origin.
    """
    for value, label in ((frame, "frame"), (rosters, "rosters")):
        if not isinstance(value, pd.DataFrame) or not value.columns.is_unique:
            raise ValueError(f"{label} must be a DataFrame with unique columns")
    if missing := {"player_id", "GW_global"}.difference(frame.columns):
        raise ValueError(f"frame missing required columns: {sorted(missing)}")
    if missing := set(_ROSTER_COLUMNS).difference(rosters.columns):
        raise ValueError(f"rosters missing required columns: {sorted(missing)}")
    columns = availability_columns()
    if collisions := set(columns).intersection(frame.columns):
        raise ValueError(f"Availability feature column collisions: {sorted(collisions)}")

    keys = pd.DataFrame({
        "player_id": _keys(frame.player_id, "frame.player_id").to_numpy(),
        "origin_gw": _keys(frame.GW_global, "frame.GW_global").to_numpy(),
    })
    admitted = rosters[list(_ROSTER_COLUMNS)].copy().reset_index(drop=True)
    admitted["player_id"] = _keys(admitted.player_id, "rosters.player_id")
    admitted["origin_gw"] = _keys(admitted.origin_gw, "rosters.origin_gw")
    if admitted.duplicated(["player_id", "origin_gw"]).any():
        raise ValueError("Ambiguous duplicate player_id/origin_gw roster rows")
    if not admitted.status.isin(_STATUSES).all():
        raise ValueError("Roster status must be one of a/d/i/s/u/n")
    chances = admitted.chance_of_playing_next_round
    numeric_chances = pd.to_numeric(chances, errors="coerce")
    canonical_chances = chances.astype("string").str.fullmatch(r"\d+(?:\.0+)?", na=False)
    if ((chances.notna() & (~canonical_chances | ~numeric_chances.isin([0, 25, 50, 75, 100]))).any()):
        raise ValueError("Roster chance must be 0/25/50/75/100 or null, excluding booleans")
    ages = pd.to_numeric(admitted.snapshot_age_hours, errors="coerce")
    canonical_ages = admitted.snapshot_age_hours.astype("string").str.fullmatch(
        r"[+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", na=False,
    )
    if (not canonical_ages.all() or not np.isfinite(ages).all()
            or (ages <= 0).any() or (ages > config.DEADLINE_SNAPSHOT_MAX_AGE_HOURS).any()):
        raise ValueError("Roster capture age must be finite, positive and within the configured maximum")
    captured = _timestamps(admitted.captured_at, "captured_at")
    deadline = _timestamps(admitted.deadline, "deadline")
    elapsed = (deadline - captured).dt.total_seconds() / 3600
    if (captured >= deadline).any() or not np.isclose(elapsed, ages, rtol=0, atol=1e-6).all():
        raise ValueError("Roster capture must precede deadline with consistent snapshot_age_hours")

    features = admitted[["player_id", "origin_gw"]].copy()
    for status in _STATUSES:
        features[f"archive_status_{status}"] = admitted.status.eq(status).astype(float)
    features["archive_chance_next"] = numeric_chances.astype(float) / 100
    features["archive_capture_age_hours"] = ages.astype(float)
    joined = keys.merge(features, on=["player_id", "origin_gw"], how="left", sort=False, validate="many_to_one")
    result = frame.copy()
    result[columns] = joined[columns].to_numpy()
    result.attrs = {key: value for key, value in frame.attrs.items() if key != "frozen_dataframe_fingerprint"}
    return result
