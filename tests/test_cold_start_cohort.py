"""Causal, opt-in fixture priors and API cold-start coverage."""
import numpy as np
import pandas as pd
import pytest

from fpl import run_week
from fpl.model.cold_start import add_registered_cold_starts, fit_pooled_priors, predict_cold_starts


def history():
    return pd.DataFrame({"player_id": [11, 12, 13, 14], "position": ["MID"] * 4,
                         "team": ["A", "A", "B", "B"], "GW_global": [1, 2, 1, 2],
                         "total_points": [0., 4., 2., 6.], "minutes": [0., 60., 30., 90.]})


def snapshot():
    return pd.DataFrame({"player_id": [11], "position": ["MID"], "team": ["A"],
                         "name": ["Known"], "value": [50.], "form": [3.],
                         "total_points": [4.], "minutes": [60.]})


def bootstrap():
    return {"teams": [{"id": 1, "name": "A"}, {"id": 2, "name": "Spurs"}],
            "elements": [{"id": 1, "code": 11, "team": 1, "element_type": 3, "now_cost": 55},
                         {"id": 2, "code": 99, "team": 2, "element_type": 3, "now_cost": 60,
                          "first_name": "New", "second_name": "Player"}]}


@pytest.mark.parametrize("q", [0., .5, 1.])
def test_availability_bernoulli_gate_moments(q):
    preds = pd.DataFrame({"player_id": [99, 11], "predicted_total_points": [3., 8.],
                          "prior_std": [2., np.nan], "p_played": [.8, np.nan],
                          "p60": [.5, np.nan], "forecast_source": ["pooled_prior", "model"]})
    result = run_week.apply_availability(preds, {99: q, 11: .5})
    assert result.predicted_total_points.tolist() == [3. * q, 4.]
    assert result.prior_std.iloc[0] == pytest.approx(np.sqrt(q * 4. + q * (1 - q) * 9.))
    assert result.p_played.iloc[0] == pytest.approx(q * .8)
    assert result.p60.iloc[0] == pytest.approx(q * .5)
    assert pd.isna(result.prior_std.iloc[1])
    assert preds.prior_std.iloc[0] == 2.


@pytest.mark.parametrize("q", [-.1, 1.1, np.inf])
def test_invalid_prior_availability_gate(q):
    preds = pd.DataFrame({"player_id": [99], "predicted_total_points": [3.],
                          "prior_std": [2.], "is_cold_start": [True]})
    with pytest.raises(ValueError, match="availability factors"):
        run_week.apply_availability(preds, {99: q})


def cohort_history():
    past = history().assign(fixture=[1, 2, 1, 2])
    newcomers = pd.DataFrame({"player_id": [99, 99, 99, 100], "position": ["MID"] * 4,
                              "team": ["A"] * 4, "GW_global": [3, 3, 4, 3],
                              "total_points": [4., 0., 10., 0.], "minutes": [90., 0., 90., 0.],
                              "fixture": [3, 4, 5, 3]})
    return pd.concat([past, newcomers], ignore_index=True)


def test_retrospective_first_round_dgw_and_dnp():
    from fpl.model.cold_start import retrospective_cold_start_predictions
    result = retrospective_cold_start_predictions(cohort_history(), 3, 4)
    assert result.player_id.tolist() == [99, 99, 100]
    assert result.fixture.tolist() == [3, 4, 3]
    assert result.GW_global.tolist() == result.origin_gw.tolist() == [3, 3, 3]
    assert result.minutes.tolist() == [90., 0., 0.]
    assert result.actual_total_points.tolist() == [4., 0., 0.]
    assert result.predicted_total_points.nunique() == 1


def test_retrospective_targets_and_future_do_not_influence_forecasts():
    from fpl.model.cold_start import retrospective_cold_start_predictions
    raw = cohort_history()
    result = retrospective_cold_start_predictions(raw, 3, 3)
    perturbed = raw.copy()
    perturbed.loc[perturbed.GW_global >= 3, "total_points"] = 999.
    perturbed.loc[perturbed.GW_global > 3, ["player_id", "position", "team", "minutes"]] = [None, None, None, np.nan]
    other = retrospective_cold_start_predictions(perturbed, 3, 3)
    cols = ["predicted_total_points", "prior_std", "p_played", "p60", "prior_count"]
    pd.testing.assert_frame_equal(result[cols], other[cols])


def test_retrospective_empty_cohort_and_optional_fixture_schema():
    from fpl.model.cold_start import retrospective_cold_start_predictions
    result = retrospective_cold_start_predictions(cohort_history(), 4, 4)
    assert result.empty and "fixture" in result and "origin_gw" in result
    result = retrospective_cold_start_predictions(history(), 3, 3)
    assert result.empty and "fixture" not in result


@pytest.mark.parametrize("bounds", [(0, 3), (4, 3), (1.5, 3), (1, np.inf)])
def test_retrospective_invalid_bounds(bounds):
    from fpl.model.cold_start import retrospective_cold_start_predictions
    with pytest.raises(ValueError):
        retrospective_cold_start_predictions(history(), *bounds)


@pytest.mark.parametrize("column,value", [("total_points", np.nan), ("minutes", np.inf), ("minutes", -1)])
def test_retrospective_invalid_targets(column, value):
    from fpl.model.cold_start import retrospective_cold_start_predictions
    raw = cohort_history()
    raw.loc[raw.player_id == 100, column] = value
    with pytest.raises(ValueError, match="invalid target"):
        retrospective_cold_start_predictions(raw, 3, 3)
