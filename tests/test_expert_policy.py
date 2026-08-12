"""Focused guards for the experimental position-specialist policy."""
import numpy as np
import pandas as pd
import pytest

from fpl.model import expert_policy
from fpl.model.train import fit_position_ensembles


def test_parse_complete_expert_map_in_canonical_order():
    result = expert_policy.parse_expert_map(
        "mid=lightgbm, GK=catboost,FWD=xgboost,DEF=catboost",
        allowed_models={"catboost", "lightgbm", "xgboost"},
    )
    assert result == {
        "GK": "catboost", "DEF": "catboost", "MID": "lightgbm", "FWD": "xgboost"
    }


def test_parse_complete_expert_map_accepts_research_expert_by_default():
    result = expert_policy.parse_expert_map(
        "GK=catboost,DEF=lightgbm,MID=tabm,FWD=catboost"
    )
    assert result["MID"] == "tabm"


@pytest.mark.parametrize("value, message", [
    ("GK=catboost,DEF=catboost,MID=catboost", "missing: FWD"),
    ("GK=catboost,DEF=catboost,MID=catboost,FWD=unknown", "Unknown expert model"),
    ("GK=catboost,DEF=catboost,MID=catboost,MID=xgboost,FWD=catboost", "Duplicate"),
    ("GK=catboost,DEF=catboost,MID,FWD=catboost", "expected POSITION=model"),
])
def test_parse_rejects_invalid_maps(value, message):
    with pytest.raises(ValueError, match=message):
        expert_policy.parse_expert_map(value, allowed_models={"catboost", "xgboost"})


def test_expert_map_overrides_scalar_without_changing_scalar_default():
    assert expert_policy.resolve_weight_strategy("single:catboost") == "single:catboost"
    resolved = expert_policy.resolve_weight_strategy(
        "nnls", {"GK": "catboost", "DEF": "xgboost", "MID": "lightgbm", "FWD": "catboost"}
    )
    assert resolved == {
        "GK": "single:catboost", "DEF": "single:xgboost",
        "MID": "single:lightgbm", "FWD": "single:catboost",
    }


def test_shared_router_preserves_row_order_and_position_assignment():
    class Constant:
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return np.full(len(X), self.value)

    frame = pd.DataFrame({
        "position": ["MID", "GK", "FWD", "DEF"],
        "feature": [1.0, 2.0, 3.0, 4.0],
    }, index=[8, 3, 12, 1])
    fitted = {position: Constant(value) for position, value in {
        "GK": 1, "DEF": 2, "MID": 3, "FWD": 4
    }.items()}
    result = expert_policy.predict_by_position(frame, ["feature"], fitted)
    assert result.index.tolist() == [8, 3, 12, 1]
    assert result.tolist() == [3.0, 1.0, 4.0, 2.0]


def test_fit_trains_only_selected_positional_experts(monkeypatch):
    calls = []

    class Stub:
        def predict(self, X):
            return np.zeros(len(X))

    def fake_fit(name, X, y, **kwargs):
        calls.append((kwargs["position"], name))
        return Stub()

    monkeypatch.setattr("fpl.model.train.models.fit_model", fake_fit)
    rows = [
        {"position": position, "x": 1.0, "total_points": 0.0, "minutes": 90, "GW_global": 1}
        for position in expert_policy.POSITIONS
    ]
    weights = {
        "GK": {"catboost": 1.0}, "DEF": {"xgboost": 1.0},
        "MID": {"lightgbm": 1.0}, "FWD": {"catboost": 1.0},
    }
    fit_position_ensembles(pd.DataFrame(rows), ["x"], weights)
    assert calls == [
        ("GK", "catboost"), ("DEF", "xgboost"),
        ("MID", "lightgbm"), ("FWD", "catboost"),
    ]
