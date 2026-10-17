"""Validate explicit registered-model fit recipes independently of fitting."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math
from numbers import Integral, Real
from typing import Any

from fpl import config
from fpl.model.models import EXPERT_SPECS


def _validate_json(value: Any) -> None:
    """Reject nonfinite/non-JSON parameters and non-string mapping keys."""
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("recipe parameter keys must be strings")
        for nested in value.values():
            _validate_json(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_json(nested)
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("recipe parameters must be finite JSON values") from exc


def resolve_fit_recipe(weights_by_pos: Mapping, params_by_pos: Mapping | None,
                       seed: int | None) -> dict | None:
    """Deep-copy complete registered defaults plus overrides for an explicit seed.

    All positions/members are validated before fitting. The optional legacy path
    returns None only when both parameters and seed are absent. Explicit recipes
    cannot change a registry member's declared loss or contradict the supplied seed.
    """
    if params_by_pos is None and seed is None:
        return None
    if params_by_pos is None or seed is None:
        raise ValueError("explicit parameters and seed must be supplied together")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("recipe seed must be a nonnegative integer")
    if (not isinstance(weights_by_pos, Mapping) or not weights_by_pos
            or not isinstance(params_by_pos, Mapping) or not params_by_pos
            or set(weights_by_pos) != set(params_by_pos)
            or not set(weights_by_pos) <= set(config.ONFIELD_POSITIONS)):
        raise ValueError("recipe and weights require identical nonempty known-position mappings")
    result = {}
    for position, weights in weights_by_pos.items():
        if not isinstance(weights, Mapping) or not weights:
            raise ValueError("each position requires a nonempty weights mapping")
        if any(isinstance(weight, bool) or not isinstance(weight, Real)
               or not math.isfinite(weight) or weight < 0 for weight in weights.values()):
            raise ValueError("weights must be finite nonnegative numeric values")
        active = {name for name, weight in weights.items() if weight > 0}
        overrides_by_model = params_by_pos[position]
        if (not active or not isinstance(overrides_by_model, Mapping)
                or set(overrides_by_model) != active or not active <= set(EXPERT_SPECS)):
            raise ValueError("recipe model keys must exactly match active registered experts")
        result[position] = {}
        for name in weights:
            if name not in active:
                continue
            overrides = overrides_by_model[name]
            if not isinstance(overrides, dict):
                raise ValueError("each model override must be a dictionary")
            _validate_json(overrides)
            defaults = EXPERT_SPECS[name].default_params(int(seed))
            for loss_key in ("loss_function", "objective", "loss"):
                if loss_key in overrides and (loss_key not in defaults or overrides[loss_key] != defaults[loss_key]):
                    raise ValueError(f"{name} recipe cannot change its registered {loss_key}")
            for seed_key in ("random_state", "random_seed", "seed"):
                if seed_key in overrides:
                    if seed_key not in defaults:
                        raise ValueError(f"{name} recipe must use its registered seed parameter")
                    value = overrides[seed_key]
                    if isinstance(value, bool) or not isinstance(value, Integral) or value != seed:
                        raise ValueError(f"{seed_key} must match the recipe seed")
            resolved = deepcopy({**defaults, **overrides})
            for seed_key in ("random_state", "random_seed"):
                if seed_key in resolved:
                    resolved[seed_key] = int(seed)
            _validate_json(resolved)
            result[position][name] = resolved
    return result
