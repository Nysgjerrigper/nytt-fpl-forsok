"""Admit bounded experimental inputs from the canonical frozen research snapshot."""
from __future__ import annotations

from numbers import Integral

import pandas as pd

from fpl import config
from fpl.data.provenance import load_frozen_dataset_manifest, load_frozen_research_dataset


def load_research_raw(end_gw: int, *, start_gw: int = 1,
                      train_max_gw: int | None = None, tuning: bool = False) -> pd.DataFrame:
    """Validate research windows before admitting frozen bytes and building features.

    Retain all history through end_gw for causal feature construction and training;
    start_gw bounds evaluation, not the retained history. Static training must end
    before evaluation. Tuning cannot pass the standing training/selection boundary.
    The full frozen feature cache cannot accept this subset: its complete-frame
    admission fingerprint is removed while original byte provenance is retained.
    Admission changes alone do not establish new model results or promotion evidence.
    """
    for name, value in (("start_gw", start_gw), ("end_gw", end_gw)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    start_gw, end_gw = int(start_gw), int(end_gw)
    if start_gw > end_gw:
        raise ValueError("research window must have start_gw <= end_gw")
    if train_max_gw is not None:
        if (isinstance(train_max_gw, bool) or not isinstance(train_max_gw, Integral)
                or not 0 < train_max_gw < start_gw):
            raise ValueError("static train_max_gw must be a positive integer before start_gw")
    cutoff = load_frozen_dataset_manifest()["max_gw_global"]
    if end_gw > cutoff:
        raise ValueError("research window exceeds the frozen dataset cutoff")
    if tuning and end_gw > config.TUNING_TRAIN_MAX_GW:
        raise ValueError("tuning cannot exceed the standing training/selection boundary")
    raw = load_frozen_research_dataset()
    bounded = raw.loc[raw["GW_global"] <= end_gw].copy()
    if bounded.empty:
        raise ValueError("research window contains no admitted rows")
    bounded.attrs = raw.attrs.copy()
    bounded.attrs.pop("frozen_dataframe_fingerprint", None)
    bounded.attrs["research_max_gw"] = end_gw
    return bounded
