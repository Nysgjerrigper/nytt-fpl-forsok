"""A registered override must never silently leave auxiliary fits ambient."""
import pandas as pd
import pytest

from fpl.model import predict


@pytest.mark.parametrize("origin", [False, True])
@pytest.mark.parametrize("overrides", [{"seed": 2}, {"parameters_by_pos": {}}])
@pytest.mark.parametrize("auxiliary", [
    {"mid_gate": object()}, {"weight_strategy": "nnls"},
    {"weight_strategy": {"GK": "single:catboost", "MID": "ridge"}},
])
def test_unsupported_auxiliary_fits_rejected_before_any_fitting(monkeypatch, origin, overrides, auxiliary):
    monkeypatch.setattr(predict, "fit_holdout_weights", lambda *a, **k: pytest.fail("ambient auxiliary fit"))
    frame = pd.DataFrame()
    with pytest.raises(ValueError, match="auxiliary fits do not support"):
        if origin:
            predict.origin_based_predictions(frame, frame, [], 1, 2, 3, **overrides, **auxiliary)
        else:
            predict.walk_forward_predictions(frame, [], 1, 2, **overrides, **auxiliary)


def test_level_calibration_rejects_explicit_overrides_before_fitting(monkeypatch):
    monkeypatch.setattr(predict, "fit_holdout_weights", lambda *a, **k: pytest.fail("ambient fit"))
    with pytest.raises(ValueError, match="level calibration"):
        predict.walk_forward_predictions(pd.DataFrame(), [], 1, 2, seed=2, calibrate_level=True)


def test_explicit_single_member_positional_strategies_remain_supported():
    predict._validate_explicit_fit_settings({"GK": "single:catboost_mae", "MID": "single:catboost_rmse"}, {}, 2)
    predict._validate_explicit_fit_settings("nnls", None, None)
