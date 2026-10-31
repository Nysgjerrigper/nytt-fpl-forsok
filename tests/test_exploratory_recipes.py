"""Exploratory recipe routing stays explicit and outside registered promotion."""
from copy import deepcopy

import pytest

from fpl.model import models
from fpl.model.recipes import resolve_fit_recipe, resolve_pipeline_recipe


@pytest.mark.parametrize("weight", ["bad", None, float("nan")])
def test_dispatch_preserves_registered_invalid_weight_admission(weight):
    with pytest.raises(ValueError, match="weights"):
        resolve_pipeline_recipe({"GK": {"catboost_mae": weight}},
                                {"GK": {"catboost_mae": {}}}, 0)


def _cnn(overrides=None):
    return {"MID": {"sequence_cnn": {} if overrides is None else overrides}}


def _blend():
    return {"MID": {"temporal_cnn_blend": {
        "cnn_weight": 0.25,
        "cnn_params": {"epochs": 2},
        "tree_params": {"iterations": 5, "loss_function": "RMSE"},
    }}}


def test_registered_routing_retains_legacy_and_explicit_validation():
    weights = {"MID": {"catboost_mae": 1.0}}
    overrides = {"MID": {"catboost_mae": {"depth": 3}}}
    assert resolve_pipeline_recipe(weights, None, None) is None
    assert resolve_pipeline_recipe(weights, overrides, 2) == resolve_fit_recipe(weights, overrides, 2)
    bad = deepcopy(overrides)
    bad["MID"]["catboost_mae"]["loss_function"] = "RMSE"
    with pytest.raises(ValueError, match="registered loss_function"):
        resolve_pipeline_recipe(weights, bad, 2)


def test_registered_admission_does_not_accept_exploratory_models():
    with pytest.raises(ValueError, match="active registered experts"):
        resolve_fit_recipe({"MID": {"sequence_cnn": 1.0}}, _cnn(), 2)


def test_cnn_resolves_complete_defaults_and_copies_overrides():
    supplied = _cnn({"epochs": 2})
    before = deepcopy(supplied)
    resolved = resolve_pipeline_recipe({"MID": {"sequence_cnn": 1.0}}, supplied, 2)
    expected = {**models.EXPLORATORY_SPECS["sequence_cnn"].default_params(2), "epochs": 2}
    assert resolved == {"MID": {"sequence_cnn": expected}}
    resolved["MID"]["sequence_cnn"]["epochs"] = 99
    assert supplied == before


@pytest.mark.parametrize("params,seed", [
    (None, 2), (_cnn(), None), (_cnn(), True), (_cnn(), -1),
    (_cnn({"seed": 1}), 2), (_cnn({"seed": True}), 1),
    (_cnn({"unknown": 1}), 2), (_cnn({"learning_rate": float("nan")}), 2),
    (_cnn({"weight_decay": float("inf")}), 2),
])
def test_cnn_rejects_implicit_conflicting_or_invalid_parameters(params, seed):
    with pytest.raises(ValueError):
        resolve_pipeline_recipe({"MID": {"sequence_cnn": 1.0}}, params, seed)


def test_cnn_explicit_matching_seed_is_retained():
    result = resolve_pipeline_recipe({"MID": {"sequence_cnn": 1.0}}, _cnn({"seed": 2}), 2)
    assert result["MID"]["sequence_cnn"]["seed"] == 2


def test_temporal_blend_copies_complete_nested_member_parameters():
    supplied = _blend()
    before = deepcopy(supplied)
    result = resolve_pipeline_recipe({"MID": {"temporal_cnn_blend": 1.0}}, supplied, 2)
    assert result == before
    result["MID"]["temporal_cnn_blend"]["cnn_params"]["epochs"] = 99
    result["MID"]["temporal_cnn_blend"]["tree_params"]["iterations"] = 99
    assert supplied == before


@pytest.mark.parametrize("mutation", ["missing_weight", "missing_cnn", "missing_tree", "extra", "nonfinite_nested"])
def test_temporal_blend_requires_exact_complete_finite_parameter_packet(mutation):
    supplied = _blend()
    params = supplied["MID"]["temporal_cnn_blend"]
    if mutation.startswith("missing_"):
        key = {"missing_weight": "cnn_weight", "missing_cnn": "cnn_params", "missing_tree": "tree_params"}[mutation]
        params.pop(key)
    elif mutation == "extra":
        params["extra"] = 1
    else:
        params["tree_params"]["learning_rate"] = float("nan")
    with pytest.raises(ValueError):
        resolve_pipeline_recipe({"MID": {"temporal_cnn_blend": 1.0}}, supplied, 2)


def test_same_position_cannot_mix_registered_and_exploratory_members():
    weights = {"MID": {"sequence_cnn": 0.5, "catboost_mae": 0.5}}
    params = {"MID": {"sequence_cnn": {}, "catboost_mae": {}}}
    with pytest.raises(ValueError, match="single exploratory member"):
        resolve_pipeline_recipe(weights, params, 2)


def test_positions_can_route_to_registered_or_exploratory_members_independently():
    weights = {"GK": {"catboost_mae": 1.0}, "MID": {"sequence_cnn": 1.0}}
    params = {"GK": {"catboost_mae": {"depth": 3}}, **_cnn({"epochs": 2})}
    result = resolve_pipeline_recipe(weights, params, 2)
    assert result["GK"] == resolve_fit_recipe({"GK": weights["GK"]}, {"GK": params["GK"]}, 2)["GK"]
    assert result["MID"]["sequence_cnn"]["epochs"] == 2
    assert set(result["MID"]["sequence_cnn"]) == set(models.EXPLORATORY_SPECS["sequence_cnn"].default_params(2))
