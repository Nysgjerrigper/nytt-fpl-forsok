"""Static evaluation must reject invalid time partitions before any fitting."""
import pandas as pd
import pytest

from fpl.model import train


@pytest.fixture
def no_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid partitions must fail before fitting or baseline generation")

    monkeypatch.setattr(train, "train_position_model", forbidden)
    monkeypatch.setattr(train, "add_croston_column", forbidden)


@pytest.mark.parametrize("bounds, message", [
    ((2, 2, 3), "strictly before"),
    ((3, 2, 4), "strictly before"),
    ((1, 2, 2), "distinct blend and evaluation gameweeks"),
    ((1, 3, 2), "distinct blend and evaluation gameweeks"),
])
def test_invalid_bounds_rejected_before_accessing_data(no_fitting, bounds, message):
    with pytest.raises(ValueError, match=message):
        train.evaluate_static_split(pd.DataFrame(), [], *bounds, include_baselines=True)


@pytest.mark.parametrize("position", train.POSITIONS)
@pytest.mark.parametrize("missing_gw, partition", [(1, "training"), (2, "blend"), (3, "evaluation")])
def test_missing_position_partition_rejected_before_any_fit(no_fitting, position, missing_gw, partition):
    df = pd.DataFrame([
        {"position": pos, "GW_global": gw}
        for pos in train.POSITIONS for gw in (1, 2, 3)
        if (pos, gw) != (position, missing_gw)
    ])
    with pytest.raises(ValueError, match=f"No {partition} rows for position {position}"):
        train.evaluate_static_split(df, [], 1, 2, 3, include_baselines=True)


def test_baseline_reordering_preserves_chronological_blend_rows(monkeypatch):
    import numpy as np

    df = pd.DataFrame([
        {"position": pos, "player_id": i, "GW_global": gw, "total_points": gw,
         "total_points_roll3": 0., "total_points_season_avg": 0.}
        for i, pos in enumerate(train.POSITIONS) for gw in (3, 1, 2)
    ], index=range(101, 113))

    def enrich(frame):
        result = frame.sort_values(["player_id", "GW_global"]).reset_index(drop=True)
        for col in ("croston_pred", "naive_drift_pred", "ses_pred", "holt_pred",
                    "theta_pred", "eb_shrinkage_pred"):
            result[col] = 0.
        return result

    for name in ("add_croston_column", "add_naive_drift_column", "add_ses_column",
                 "add_holt_column", "add_theta_column", "add_eb_shrinkage_column"):
        monkeypatch.setattr(train, name, enrich)
    monkeypatch.setattr(train.models, "MODEL_NAMES", ("ols",))
    monkeypatch.setattr(train, "naive_lag1_scale", lambda _: 1.)

    class Predictor:
        def predict(self, X):
            return X["GW_global"].to_numpy()

    monkeypatch.setattr(train, "train_position_model", lambda *args: Predictor())

    class BlendChecked(Exception):
        pass

    def inspect_blend(predictions, actual, method):
        np.testing.assert_array_equal(actual, [2])
        np.testing.assert_array_equal(predictions["ols"], [2])
        raise BlendChecked

    monkeypatch.setattr(train, "fit_weights", inspect_blend)
    with pytest.raises(BlendChecked):
        train.evaluate_static_split(df, ["GW_global"], 1, 2, 3, include_baselines=True)
