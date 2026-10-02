"""Shared identity, comparability, and provenance checks for research OOF panels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fpl import config
from fpl.model import models


def validate_oof_panel(oof: pd.DataFrame, *, positions: tuple[str, ...] | None = None,
                       experts: tuple[str, ...] | None = None) -> None:
    """Require comparable expert forecasts on identical player-gameweek rows.

    Identity, finite targets/scales, and fixed parameter provenance are mandatory.
    Causal selection-window checks remain the responsibility of compare_ensembles.
    """
    required = {"player_id", "position", "expert", "GW_global", "prediction",
                "actual_total_points", "mase_scale", "train_max_gw", "seed",
                "params_hash", "stage"}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"OOF lineage missing columns for panel: {sorted(missing)}")
    if oof.empty:
        raise ValueError("OOF panel cannot be empty")
    if oof[list(required)].isna().any().any():
        raise ValueError("OOF panel identity, values, and provenance cannot be missing")
    for column in ("prediction", "actual_total_points", "mase_scale",
                   "GW_global", "train_max_gw", "seed"):
        if not pd.api.types.is_numeric_dtype(oof[column]) or not np.isfinite(oof[column]).all():
            raise ValueError(f"OOF panel {column} must contain finite numeric values")
    for column in ("GW_global", "train_max_gw", "seed"):
        if pd.api.types.is_bool_dtype(oof[column]) or not (oof[column] % 1 == 0).all():
            raise ValueError(f"OOF panel {column} must contain integer values")
    if not (oof["mase_scale"] > 0).all():
        raise ValueError("OOF panel mase_scale must be positive")
    if not oof["params_hash"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("OOF panel params_hash must contain nonempty strings")
    expected_positions = set(config.ONFIELD_POSITIONS if positions is None else positions)
    expected_experts = set(oof["expert"] if experts is None else experts)
    if not expected_positions or not expected_positions.issubset(config.ONFIELD_POSITIONS):
        raise ValueError("OOF panel requested positions must be known positions")
    if not expected_experts or not expected_experts.issubset(models.REGISTERED_MODEL_NAMES):
        raise ValueError("OOF panel contains unknown experts")
    if set(oof["position"]) != expected_positions:
        raise ValueError("OOF panel has missing, extra, or unknown positions")
    if set(oof["expert"]) != expected_experts:
        raise ValueError("OOF panel has missing or extra experts")
    if oof["seed"].nunique(dropna=False) != 1:
        raise ValueError("OOF panel must contain exactly one seed")
    identity = ["position", "expert", "player_id", "GW_global"]
    if oof.duplicated(identity).any():
        raise ValueError("OOF panel contains duplicate position/expert/player/gameweek rows")
    expert_counts = oof.groupby("position", dropna=False)["expert"].nunique(dropna=False)
    if not (expert_counts == len(expected_experts)).all():
        raise ValueError("OOF panel must contain the same expert set for every position")
    row_groups = oof.groupby(["position", "player_id", "GW_global"], dropna=False)
    if not (row_groups["expert"].nunique(dropna=False) == len(expected_experts)).all():
        raise ValueError("OOF panel experts must forecast identical player/gameweek rows")
    shared = ["actual_total_points", "mase_scale", "train_max_gw"]
    if "fixture_count" in oof:
        counts = oof["fixture_count"]
        if (not pd.api.types.is_numeric_dtype(counts) or pd.api.types.is_bool_dtype(counts)
                or not np.isfinite(counts).all() or not ((counts > 0) & (counts % 1 == 0)).all()):
            raise ValueError("OOF panel fixture_count must contain positive integers")
        shared.append("fixture_count")
    if not (row_groups[shared].nunique(dropna=False) == 1).all().all():
        raise ValueError("OOF panel has inconsistent actuals, mase_scale, or train_max_gw across experts")
    hashes = oof.groupby(["position", "expert", "seed"], dropna=False)["params_hash"].nunique(dropna=False)
    if not (hashes == 1).all():
        raise ValueError("OOF panel params_hash must be constant per position/expert/seed")
