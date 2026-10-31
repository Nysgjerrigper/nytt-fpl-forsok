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


def resolve_pipeline_recipe(weights_by_pos: Mapping, params_by_pos: Mapping | None,
                            seed: int | None) -> dict | None:
    """Route opt-in exploratory members without extending the registered universe.

    Registered recipes retain their existing validator unchanged. Exploratory
    recipes require explicit complete parameters and a seed, and cannot be mixed
    with registered members inside the same positional ensemble.
    """
    from fpl.model.models import EXPLORATORY_SPECS

    if not isinstance(weights_by_pos, Mapping):
        return resolve_fit_recipe(weights_by_pos, params_by_pos, seed)
    exploratory = any(
        name in EXPLORATORY_SPECS and isinstance(weight, Real)
        and not isinstance(weight, bool) and math.isfinite(weight) and weight > 0
        for weights in weights_by_pos.values() if isinstance(weights, Mapping)
        for name, weight in weights.items()
    )
    if not exploratory:
        return resolve_fit_recipe(weights_by_pos, params_by_pos, seed)
    if (isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0
            or not isinstance(params_by_pos, Mapping)
            or set(params_by_pos) != set(weights_by_pos)
            or not set(weights_by_pos) <= set(config.ONFIELD_POSITIONS)):
        raise ValueError("exploratory recipes require explicit matching positions and a seed")
    result = {}
    for position, weights in weights_by_pos.items():
        if (not isinstance(weights, Mapping) or not weights
                or any(isinstance(value, bool) or not isinstance(value, Real)
                       or not math.isfinite(value) or value < 0 for value in weights.values())):
            raise ValueError("exploratory weights must be finite nonnegative values")
        active = {name: value for name, value in weights.items() if value > 0}
        if not any(name in EXPLORATORY_SPECS for name in active):
            result.update(resolve_fit_recipe({position: weights}, {position: params_by_pos[position]}, seed))
            continue
        if len(active) != 1 or not set(active) <= set(EXPLORATORY_SPECS):
            raise ValueError("exploratory recipes require a single exploratory member per position")
        name = next(iter(active))
        overrides = params_by_pos[position]
        if not isinstance(overrides, Mapping) or set(overrides) != {name} or not isinstance(overrides[name], dict):
            raise ValueError("exploratory recipe must exactly cover its active member")
        params = deepcopy(overrides[name])
        _validate_json(params)
        if "seed" in params and (isinstance(params["seed"], bool) or params["seed"] != seed):
            raise ValueError("exploratory member seed must match recipe seed")
        if name == "sequence_cnn":
            defaults = EXPLORATORY_SPECS[name].default_params(int(seed))
            if set(params) - (set(defaults) | {"seed"}):
                raise ValueError("unknown exploratory CNN parameter")
            params = {**defaults, **params}
        elif set(params) != {"cnn_weight", "cnn_params", "tree_params"}:
            raise ValueError("temporal blend requires explicit weight and both member parameters")
        result[position] = {name: params}
    return result
