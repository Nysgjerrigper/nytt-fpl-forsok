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


def test_mixed_moments_include_dnp_and_shrink():
    priors = fit_pooled_priors(history(), 3, prior_strength=2.)
    resolved = predict_cold_starts(pd.DataFrame({"position": ["MID", "MID"], "team": ["A", "Unknown"]}), priors)
    assert resolved.predicted_total_points.tolist() == [2.5, 3.]
    assert resolved.prior_std.iloc[0] == pytest.approx(np.sqrt(11. - 2.5 ** 2))
    assert resolved.p_played.tolist() == [.625, .75]
    assert resolved.p60.tolist() == [.5, .5]
    assert resolved.prior_count.tolist() == [2, 4]


def test_future_excluded_and_prediction_does_not_read_outcomes():
    h = history()
    future = h.iloc[[0]].assign(GW_global=3, total_points=np.nan, minutes=np.nan)
    pd.testing.assert_frame_equal(fit_pooled_priors(h, 3), fit_pooled_priors(pd.concat([h, future]), 3))
    frame = snapshot()
    pd.testing.assert_frame_equal(predict_cold_starts(frame, fit_pooled_priors(h, 3)),
                                  predict_cold_starts(frame.assign(total_points=999, minutes=-999), fit_pooled_priors(h, 3)))


@pytest.mark.parametrize("kwargs", [{"origin_gw": 0}, {"origin_gw": 1.5}, {"origin_gw": np.nan},
                                     {"origin_gw": 3, "prior_strength": 0}, {"origin_gw": 3, "prior_strength": np.inf}])
def test_invalid_fit_arguments(kwargs):
    with pytest.raises(ValueError):
        fit_pooled_priors(history(), **kwargs)


@pytest.mark.parametrize("column,value", [("minutes", np.nan), ("total_points", np.inf),
                                            ("player_id", None), ("position", "bad"), ("team", None)])
def test_invalid_admitted_history(column, value):
    h = history()
    h.loc[0, column] = value
    with pytest.raises(ValueError):
        fit_pooled_priors(h, 3)


def test_registration_preserves_known_and_unknown_form_and_targets():
    old = snapshot()
    extended = add_registered_cold_starts(old, bootstrap())
    pd.testing.assert_frame_equal(extended.loc[[0], old.columns], old)
    new = extended.iloc[1]
    assert new.player_id == 99 and new.team == "Tottenham" and new.position == "MID"
    assert new.value == 60 and new["name"] == "New Player" and new.is_cold_start
    assert pd.isna(new.form) and pd.isna(new.total_points) and pd.isna(new.minutes)
    assert not extended.iloc[0].is_cold_start
    register = bootstrap()
    register["elements"] = register["elements"][:1]
    assert len(add_registered_cold_starts(old, register)) == 1


@pytest.mark.parametrize("kind", ["duplicate", "code", "team", "position", "price"])
def test_invalid_register(kind):
    register = bootstrap()
    if kind == "duplicate":
        register["elements"].append(register["elements"][0])
    else:
        key = {"code": "code", "team": "team", "position": "element_type", "price": "now_cost"}[kind]
        register["elements"][1][key] = None
    with pytest.raises(ValueError):
        add_registered_cold_starts(snapshot(), register)


def test_missing_position_history_fails_clearly():
    with pytest.raises(ValueError, match="No pooled history"):
        predict_cold_starts(pd.DataFrame({"position": ["GK"], "team": ["A"]}), fit_pooled_priors(history(), 3))


def test_live_direct_priors_dgw_blank_and_default(monkeypatch):
    calls = []
    def predict(frame, feature_cols, models):
        assert not frame.get("is_cold_start", pd.Series(False, index=frame.index)).any()
        calls.append(frame.player_id.tolist())
        return np.full(len(frame), 8.)
    monkeypatch.setattr(run_week, "predict_by_position", predict)
    monkeypatch.setattr(run_week, "build_team_fixture_map", lambda bootstrap, gw: {
        "A": [("B", True, 2)], "Tottenham": [("A", True, 3), ("B", False, 4)]} if gw == 1 else {})
    regular = snapshot()
    default = run_week.build_future_predictions(regular, ["form"], {}, bootstrap(), 1, 2)
    assert len(default) == 1 and default.predicted_total_points.iloc[0] == 8
    assert "forecast_source" not in default
    extended = add_registered_cold_starts(regular, bootstrap())
    preds = run_week.build_future_predictions(extended, ["form"], {}, bootstrap(), 1, 2,
                                             cold_start_priors=fit_pooled_priors(history(), 3))
    assert len(preds) == 3 and preds.GW.tolist() == [1, 1, 1]
    assert preds.forecast_source.tolist() == ["model", "pooled_prior", "pooled_prior"]
    assert preds.predicted_total_points.tolist() == [8., 3., 3.]
    assert pd.isna(preds.prior_std.iloc[0])
    scaled = run_week.apply_availability(preds, {99: .5})
    assert scaled.prior_std.iloc[1] == pytest.approx(
        np.sqrt(.5 * preds.prior_std.iloc[1] ** 2 + .25 * preds.predicted_total_points.iloc[1] ** 2))
    assert calls == [[11], [11]]


def test_empty_register_and_invalid_snapshot_ids():
    old = snapshot()
    result = add_registered_cold_starts(old, {"teams": [], "elements": []})
    pd.testing.assert_frame_equal(result[old.columns], old)
    assert not result.is_cold_start.any()
    for invalid in (pd.concat([old, old]), old.assign(player_id=-1)):
        with pytest.raises(ValueError):
            add_registered_cold_starts(invalid, bootstrap())


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
