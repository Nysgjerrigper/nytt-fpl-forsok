"""Causal sequences of recorded player rounds for the optional CNN research lane.

Histories use stable player codes and span seasons. A step is an observed player
round, not a fixed calendar week; gaps do not create invented zero observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fpl import config


def sequence_columns() -> list[str]:
    """Return channel-major columns, oldest to newest within each channel."""
    length = config.CNN_SEQUENCE_LENGTH
    return ([f"cnn_{stat}_t{step}" for stat in config.CNN_SEQUENCE_STATS
             for step in range(length)]
            + [f"cnn_present_t{step}" for step in range(length)])


def _rounds(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate fixture rows and sum each statistic into observed player rounds."""
    if frame.empty:
        raise ValueError("sequence input must be nonempty")
    required = {"player_id", "GW_global", *config.CNN_SEQUENCE_STATS}
    if missing := required - set(frame.columns):
        raise ValueError(f"sequence input lacks required columns: {sorted(missing)}")
    if not frame.columns.is_unique:
        raise ValueError("sequence input must have unique column names")
    if collision := set(sequence_columns()) & set(frame.columns):
        raise ValueError(f"sequence input already has generated columns: {sorted(collision)}")
    for column in ("player_id", "GW_global"):
        values = frame[column]
        if (not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
                or values.isna().any() or not np.isfinite(values).all() or not (values > 0).all()
                or not (values % 1 == 0).all()):
            raise ValueError(f"{column} must contain finite positive integers")
    keys = ["player_id", "GW_global"]
    repeated = frame.duplicated(keys, keep=False)
    if repeated.any():
        if "fixture" not in frame or frame.loc[repeated, "fixture"].isna().any():
            raise ValueError("multiple player-round rows require fixture identities")
        if frame.duplicated(keys + ["fixture"]).any():
            raise ValueError("duplicate player-round fixture identity")
    numeric = frame[keys].copy()
    for stat in config.CNN_SEQUENCE_STATS:
        try:
            values = pd.to_numeric(frame[stat], errors="raise").to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{stat} must contain numeric values or NaN") from exc
        if np.isinf(values).any():
            raise ValueError(f"{stat} must not contain infinities")
        numeric[stat] = values
    return numeric.groupby(keys, as_index=False, sort=True)[list(config.CNN_SEQUENCE_STATS)].sum(min_count=1)


def _sequences(rounds: pd.DataFrame, *, include_latest: bool) -> pd.DataFrame:
    """Vectorize fixed channel/lag shifts over chronologically sorted histories."""
    length = config.CNN_SEQUENCE_LENGTH
    groups = rounds.groupby("player_id", sort=False)
    offset = 0 if include_latest else 1
    columns = {}
    for stat in config.CNN_SEQUENCE_STATS:
        for step in range(length):
            lag = length - 1 - step + offset
            columns[f"cnn_{stat}_t{step}"] = groups[stat].shift(lag).to_numpy()
    count = groups.cumcount().to_numpy()
    for step in range(length):
        lag = length - 1 - step + offset
        columns[f"cnn_present_t{step}"] = (count >= lag).astype(float)
    return pd.DataFrame(columns, index=rounds.index)


def add_sequence_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append strictly prior recorded-round sequences to every raw fixture row.

    Same-round fixtures receive identical histories; no current fixture outcome
    enters another fixture's sequence. Round sums retain NaN when every fixture
    statistic is missing. Padding values are NaN and their presence flag is zero.
    Original rows, order and index (including duplicate index labels) are preserved.
    """
    rounds = _rounds(frame)
    sequences = _sequences(rounds, include_latest=False)
    keys = ["player_id", "GW_global"]
    positions = pd.MultiIndex.from_frame(rounds[keys]).get_indexer(pd.MultiIndex.from_frame(frame[keys]))
    appended = sequences.iloc[positions].copy()
    appended.index = frame.index
    return pd.concat([frame.copy(), appended], axis=1)


def sequence_snapshot(raw_history: pd.DataFrame) -> pd.DataFrame:
    """Return each stable player's latest recorded rounds for an origin snapshot.

    The latest historical round is included. The caller must supply history
    strictly before the origin deadline. Only player_id and sequence columns are
    returned, with one row per player; empty history returns that empty schema.
    """
    if raw_history.empty:
        return pd.DataFrame(columns=["player_id", *sequence_columns()])
    rounds = _rounds(raw_history)
    sequences = _sequences(rounds, include_latest=True)
    last = rounds.groupby("player_id", sort=False).tail(1).index
    return pd.concat([rounds.loc[last, ["player_id"]], sequences.loc[last]], axis=1).reset_index(drop=True)
