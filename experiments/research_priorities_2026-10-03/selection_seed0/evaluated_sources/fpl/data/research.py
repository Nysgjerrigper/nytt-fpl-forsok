"""Frozen-input admission and causal cutoff for research feature construction."""
from numbers import Integral
from pathlib import Path

import pandas as pd

from fpl import config, features
from fpl.data.provenance import load_frozen_dataset_manifest, load_frozen_research_dataset


def validate_gameweek(value: int, name: str) -> None:
    """Reject noninteger or nonpositive research gameweek bounds."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer gameweek")


def validate_static_window(train_max_gw: int, test_min_gw: int, test_max_gw: int) -> None:
    """Require capped training and a strictly later, nonempty evaluation window."""
    for name, value in (("train_max_gw", train_max_gw), ("test_min_gw", test_min_gw),
                        ("test_max_gw", test_max_gw)):
        validate_gameweek(value, name)
    validate_tuning_cutoff(train_max_gw)
    if not train_max_gw < test_min_gw <= test_max_gw:
        raise ValueError("require train_max_gw < test_min_gw <= test_max_gw")


def validate_tuning_cutoff(train_max_gw: int) -> None:
    """Keep static fitting and tuning inside the registered selection ceiling."""
    validate_gameweek(train_max_gw, "train_max_gw")
    if train_max_gw > config.TUNING_TRAIN_MAX_GW:
        raise ValueError(f"training/tuning cutoff must be <= {config.TUNING_TRAIN_MAX_GW}")


def validate_walk_forward_window(start_gw: int, end_gw: int, retrain_every: int) -> None:
    """Require an ordered walk-forward window and a positive retrain interval."""
    for name, value in (("start_gw", start_gw), ("end_gw", end_gw),
                        ("retrain_every", retrain_every)):
        validate_gameweek(value, name)
    if start_gw > end_gw:
        raise ValueError("require start_gw <= end_gw")


def load_research_features(
    max_gw: int, dataset_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Validate complete frozen bytes, then build features only through ``max_gw``.

    The subset retains the verified source byte hash for provenance, but loses the
    whole-frame fingerprint: it cannot impersonate the complete admitted snapshot.
    No feature-cache admission is reused for either the subset or derived features.
    """
    raw = load_frozen_research_dataset(dataset_path)
    validate_gameweek(max_gw, "max_gw")
    frozen_max = load_frozen_dataset_manifest()["max_gw_global"]
    if max_gw > frozen_max:
        raise ValueError(f"max_gw must be <= frozen cutoff {frozen_max}")
    subset = raw.loc[raw["GW_global"].between(1, max_gw)].copy()
    subset.attrs.pop("frozen_dataframe_fingerprint", None)
    frame = features.build_feature_frame(subset)
    frame.attrs.update(subset.attrs)
    frame.attrs.pop("frozen_dataframe_fingerprint", None)
    return subset, frame, features.feature_columns(frame)
