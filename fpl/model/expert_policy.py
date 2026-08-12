"""Position-specialist model policy and shared prediction routing.

The production policy remains the scalar ``single:catboost`` strategy.  Research
runs may override it with a complete expert map, for example::

    GK=catboost,DEF=lightgbm,MID=tabm,FWD=catboost

A complete map is deliberate: silently filling an omitted position from the
production default would make experiment provenance ambiguous.
"""
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from fpl import config
from fpl.model import models as model_registry
from fpl.model.mid_gate import MidGateConfig


POSITIONS = tuple(config.ONFIELD_POSITIONS)


def parse_expert_map(value, allowed_models=None):
    """Parse and validate a complete ``POSITION=model`` expert map.

    ``value`` may be the CLI string form or an existing mapping. Position keys
    are case-insensitive; model names use the registry's canonical spelling.
    The returned dict always follows the stable GK/DEF/MID/FWD order.
    """
    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, str):
        items = []
        for entry in value.split(","):
            if "=" not in entry:
                raise ValueError(
                    f"Invalid expert-map entry {entry!r}; expected POSITION=model"
                )
            position, model_name = entry.split("=", 1)
            items.append((position, model_name))
    else:
        raise TypeError("expert map must be a POSITION=model string or mapping")

    parsed = {}
    for raw_position, raw_model in items:
        position = str(raw_position).strip().upper()
        model_name = str(raw_model).strip()
        if position not in POSITIONS:
            raise ValueError(
                f"Unknown position {position!r}; expected one of {', '.join(POSITIONS)}"
            )
        if position in parsed:
            raise ValueError(f"Duplicate expert-map position {position}")
        if not model_name:
            raise ValueError(f"Missing model name for position {position}")
        parsed[position] = model_name

    missing = [position for position in POSITIONS if position not in parsed]
    if missing:
        raise ValueError("Expert map must define every position; missing: " + ", ".join(missing))

    # ``MODEL_NAMES`` deliberately remains the production bake-off.  Research
    # experts are accepted only through this explicit complete-map override.
    allowed = set(model_registry.REGISTERED_MODEL_NAMES if allowed_models is None else allowed_models)
    unknown = sorted({name for name in parsed.values() if name not in allowed})
    if unknown:
        raise ValueError(
            "Unknown expert model(s): " + ", ".join(unknown)
            + "; registered models: " + ", ".join(sorted(allowed))
        )
    return {position: parsed[position] for position in POSITIONS}


def expert_map_strategies(expert_map):
    """Convert a validated expert map to existing per-position strategies."""
    parsed = parse_expert_map(expert_map)
    return {position: f"single:{model_name}" for position, model_name in parsed.items()}


def resolve_weight_strategy(weight_strategy, expert_map=None):
    """Return the effective strategy while preserving scalar compatibility.

    An explicitly supplied expert map is an experimental override. Without one,
    scalar strategies (including production ``single:catboost``) and legacy
    per-position strategy dictionaries pass through unchanged.
    """
    if expert_map is None:
        return weight_strategy
    return expert_map_strategies(expert_map)


def predict_by_position(frame, feature_cols: Sequence[str], fitted_by_position,
                        level_scalars=None):
    """Route rows to their fitted positional expert and return aligned predictions."""
    predictions = pd.Series(0.0, index=frame.index, dtype=float)
    scalars = level_scalars or {}
    for position in POSITIONS:
        mask = frame["position"] == position
        fitted = fitted_by_position.get(position)
        if mask.any() and fitted is not None:
            values = np.asarray(fitted.predict(frame.loc[mask, feature_cols]), dtype=float)
            predictions.loc[mask] = float(scalars.get(position, 1.0)) * values
    return predictions


def fit_mid_gate_experts(train_frame, feature_cols: Sequence[str], gate: MidGateConfig):
    """Fit exactly the MID experts named by a frozen, causal gate."""
    mid = train_frame[train_frame["position"] == "MID"]
    return {name: model_registry.fit_model(name, mid[feature_cols], mid["total_points"],
                                           position="MID", minutes=mid.get("minutes"),
                                           gw=mid.get("GW_global")) for name in gate.candidates}


def predict_with_mid_gate(frame, feature_cols: Sequence[str], fitted_by_position, gate: MidGateConfig,
                          mid_experts, level_scalars=None):
    """Route only MID rows through a frozen gate; output remains one scalar per row."""
    predictions = predict_by_position(frame, feature_cols, fitted_by_position, level_scalars)
    mask = frame["position"] == "MID"
    if mask.any():
        mid = frame.loc[mask]
        candidate_predictions = {name: model.predict(mid[feature_cols])
                                 for name, model in mid_experts.items()}
        predictions.loc[mask] = gate.predict(mid, candidate_predictions)
        if level_scalars:
            predictions.loc[mask] *= float(level_scalars.get("MID", 1.0))
    return predictions
