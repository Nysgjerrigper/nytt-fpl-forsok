"""History-only prior arithmetic, identity admission and synthetic-row separation."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from fpl import features, run_week
from fpl.model import cold_start, train


def history():
    return pd.DataFrame({"player_id": [11, 12, 13, 14], "position": ["MID"] * 4,
                         "team": ["A", "A", "B", "B"], "GW_global": [1, 2, 1, 2],
                         "total_points": [0., 4., 2., 6.], "minutes": [0., 60., 30., 90.]})


def snapshot():
    return pd.DataFrame({"player_id": [11], "position": ["MID"], "team": ["A"],
                         "name": ["Known"], "value": [50.], "form": [3.],
                         "GW_global": [3], "total_points": [4.], "minutes": [60.]})


def bootstrap():
    return {"teams": [{"id": 1, "name": "A"}, {"id": 2, "name": "Spurs"}],
            "elements": [{"code": 11, "team": 1, "element_type": 3, "now_cost": 55},
                         {"code": 99, "team": 2, "element_type": 3, "now_cost": 60,
                          "first_name": "New", "second_name": "Player"}]}


def test_mixed_moments_shrinkage_dnp_and_complete_team_fallback():
    priors = cold_start.fit_pooled_priors(history(), 3, prior_strength=2.)
    frame = pd.DataFrame({"position": ["MID", "MID"], "team": ["A", "Unknown"]}, index=[7, 7])
    predicted = cold_start.predict_cold_starts(frame, priors, origin_gw=3)
    assert predicted.index.tolist() == [7, 7]
    assert predicted.predicted_total_points.tolist() == [2.5, 3.]
    assert predicted.prior_std.iloc[0] == pytest.approx(np.sqrt(11. - 2.5 ** 2))
    assert predicted.p_played.tolist() == [.625, .75]
    assert predicted.p60.tolist() == [.5, .5]
    assert predicted.prior_count.tolist() == [2, 4]
    assert priors.attrs["origin_gw"] == 3 and priors.attrs["history_max_gw"] == 2
    assert priors.attrs["prior_strength"] == 2.


def test_future_rows_and_target_outcomes_do_not_change_priors_or_fingerprint():
    original = history()
    future = original.iloc[[0]].assign(GW_global=3, player_id="invalid future identity",
                                       total_points=np.nan, minutes=-999.)
    expected = cold_start.fit_pooled_priors(original, 3)
    actual = cold_start.fit_pooled_priors(pd.concat([original, future], ignore_index=True), 3)
    pd.testing.assert_frame_equal(expected, actual)
    assert expected.attrs == actual.attrs
    target = snapshot()
    pd.testing.assert_frame_equal(
        cold_start.predict_cold_starts(target, expected, origin_gw=3),
        cold_start.predict_cold_starts(target.assign(total_points=np.inf, minutes=-999), expected, origin_gw=3))


@pytest.mark.parametrize("kwargs", [{"origin_gw": True}, {"origin_gw": 0}, {"origin_gw": 1.5},
                                     {"origin_gw": np.nan}, {"origin_gw": 3, "prior_strength": True},
                                     {"origin_gw": 3, "prior_strength": 0},
                                     {"origin_gw": 3, "prior_strength": np.inf}])
def test_invalid_fit_arguments(kwargs):
    with pytest.raises(ValueError):
        cold_start.fit_pooled_priors(history(), **kwargs)


@pytest.mark.parametrize("column,value", [("player_id", True), ("player_id", "11"),
    ("player_id", None), ("GW_global", True), ("position", "bad"), ("team", " "),
    ("total_points", np.inf), ("minutes", -1), ("minutes", np.nan)])
def test_invalid_admitted_history_rejected(column, value):
    frame = history().astype({column: object})
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        cold_start.fit_pooled_priors(frame, 3)


def test_duplicate_history_rejected_but_distinct_dgw_fixtures_admitted():
    frame = history()
    repeated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        cold_start.fit_pooled_priors(repeated, 3)
    repeated["fixture"] = [101, 201, 102, 202, 103]
    prior = cold_start.fit_pooled_priors(repeated, 3)
    assert prior.loc[prior.team.isna(), "prior_count"].iloc[0] == 5
    repeated.loc[4, "fixture"] = 101
    with pytest.raises(ValueError):
        cold_start.fit_pooled_priors(repeated, 3)


def test_empty_history_has_schema_metadata_and_cannot_invent_position_priors():
    prior = cold_start.fit_pooled_priors(history().iloc[:0], 3)
    assert prior.empty and prior.attrs["history_max_gw"] is None
    assert set(prior) == set(cold_start.PRIOR_COLUMNS)
    with pytest.raises(ValueError, match="pooled history"):
        cold_start.predict_cold_starts(snapshot(), prior, origin_gw=3)


@pytest.mark.parametrize("change", ["origin", "future_cutoff", "missing_metadata", "changed_mean"])
def test_prior_binding_and_mutation_rejected(change):
    priors = cold_start.fit_pooled_priors(history(), 3)
    if change == "origin":
        origin = 4
    else:
        origin = 3
        if change == "future_cutoff":
            priors.attrs["history_max_gw"] = 3
        elif change == "missing_metadata":
            priors.attrs.clear()
        else:
            priors.loc[0, "predicted_total_points"] = 999.
    with pytest.raises(ValueError):
        cold_start.predict_cold_starts(snapshot(), priors, origin_gw=origin)


def test_prior_json_roundtrip_preserves_binding_and_predictions():
    priors = cold_start.fit_pooled_priors(history(), 3)
    payload = json.loads(json.dumps({"metadata": priors.attrs, "rows": priors.to_dict("records")}, allow_nan=False))
    restored = pd.DataFrame(payload["rows"], columns=cold_start.PRIOR_COLUMNS)
    restored.attrs = payload["metadata"]
    pd.testing.assert_frame_equal(
        cold_start.predict_cold_starts(snapshot(), priors, origin_gw=3),
        cold_start.predict_cold_starts(snapshot(), restored, origin_gw=3))


@pytest.mark.parametrize("column,value", [("prior_std", -1.), ("p_played", 1.1), ("p60", .99),
                                         ("prior_count", 0), ("prior_count", 1.5),
                                         ("predicted_total_points", np.nan)])
def test_semantically_invalid_priors_rejected_even_with_updated_digest(column, value):
    priors = cold_start.fit_pooled_priors(history(), 3)
    priors[column] = priors[column].astype(float)
    priors.loc[0, column] = value
    priors.attrs["table_sha256"] = cold_start._table_digest(priors)
    with pytest.raises(ValueError):
        cold_start.validate_pooled_priors(priors, 3)


def test_duplicate_prior_keys_and_missing_pooled_position_rejected():
    original = cold_start.fit_pooled_priors(history(), 3)
    for priors in (pd.concat([original, original.iloc[[1]]], ignore_index=True),
                   original.loc[original.team.notna()].copy()):
        priors.attrs = original.attrs.copy()
        priors.attrs["table_sha256"] = cold_start._table_digest(priors)
        with pytest.raises(ValueError):
            cold_start.validate_pooled_priors(priors, 3)


def test_registration_appends_only_missing_codes_without_fabricating_form():
    original = snapshot()
    result = cold_start.add_registered_cold_starts(original, bootstrap())
    pd.testing.assert_frame_equal(result.loc[[0], original.columns], original)
    new = result.iloc[1]
    assert new.player_id == 99 and new.team == "Tottenham" and new.position == "MID"
    assert new.GW_global == 3 and new.value == 60 and new["name"] == "New Player"
    assert pd.isna(new.form) and pd.isna(new.total_points) and pd.isna(new.minutes)
    assert result.is_cold_start.tolist() == [False, True]
    assert "is_cold_start" not in original
    with pytest.raises(ValueError):
        cold_start.add_registered_cold_starts(result, bootstrap())


@pytest.mark.parametrize("field,value", [("code", True), ("code", "99"), ("team", True),
    ("element_type", True), ("now_cost", True), ("now_cost", 60.5), ("now_cost", 0)])
def test_invalid_registered_identities_and_price(field, value):
    register = bootstrap()
    register["elements"][1][field] = value
    with pytest.raises(ValueError):
        cold_start.add_registered_cold_starts(snapshot(), register)


def test_duplicate_register_and_string_snapshot_identity_rejected():
    register = bootstrap()
    register["elements"].append(deepcopy(register["elements"][0]))
    with pytest.raises(ValueError):
        cold_start.add_registered_cold_starts(snapshot(), register)
    with pytest.raises(ValueError):
        cold_start.add_registered_cold_starts(snapshot().assign(player_id="11"), bootstrap())
    register = bootstrap()
    register["teams"].append({"id": 3, "name": "Tottenham"})
    with pytest.raises(ValueError):
        cold_start.add_registered_cold_starts(snapshot(), register)


@pytest.mark.parametrize("identity", [2**63, 2**100, float(2**63)])
def test_out_of_range_ids_fail_without_integer_wraparound(identity):
    frame = history().astype({"player_id": object})
    frame.loc[0, "player_id"] = identity
    with pytest.raises(ValueError):
        cold_start.fit_pooled_priors(frame, 3)
    register = bootstrap()
    register["elements"][1]["code"] = identity
    with pytest.raises(ValueError):
        cold_start.add_registered_cold_starts(snapshot(), register)


@pytest.mark.parametrize("flag", [1, "False", None])
def test_synthetic_marker_is_strict_boolean(flag):
    with pytest.raises(ValueError):
        cold_start.cold_start_mask(snapshot().assign(is_cold_start=flag))


@pytest.mark.parametrize("entrypoint", ["features", "ensemble", "weights", "calibration"])
def test_synthetic_rows_rejected_before_any_feature_build_or_fit(monkeypatch, entrypoint):
    frame = snapshot().assign(is_cold_start=True)
    monkeypatch.setattr(train.models, "fit_model", lambda *a, **k: pytest.fail("synthetic training row reached fit"))
    weights = {p: {"catboost": 1.} for p in ("GK", "DEF", "MID", "FWD")}
    with pytest.raises(ValueError, match="[Ss]ynthetic|cold.start"):
        if entrypoint == "features":
            features.build_feature_frame(frame)
        elif entrypoint == "ensemble":
            train.fit_position_ensembles(frame, ["form"], weights)
        elif entrypoint == "weights":
            train.fit_holdout_weights(frame, ["form"], first_holdout_gw=3)
        else:
            train.fit_level_calibration(frame, ["form"], first_holdout_gw=3, weights_by_pos=weights)


@pytest.mark.parametrize("use_gate", [False, True])
def test_fixture_expansion_routes_only_known_rows_to_models_and_preserves_default(monkeypatch, use_gate):
    calls = []
    def predict(frame, *args, **kwargs):
        assert not cold_start.cold_start_mask(frame).any()
        calls.append(frame.player_id.tolist())
        return np.full(len(frame), 8.)
    monkeypatch.setattr(run_week, "predict_by_position", predict)
    monkeypatch.setattr(run_week, "predict_with_mid_gate", predict)
    monkeypatch.setattr(run_week, "build_team_fixture_map", lambda bootstrap, gw: {
        "A": [("B", True, 2)], "Tottenham": [("A", True, 3), ("B", False, 4)]})
    original = snapshot()
    options = {"mid_gate": object()} if use_gate else {}
    default = run_week.build_future_predictions(original, ["form"], {}, bootstrap(), 1, 1, **options)
    assert default.predicted_total_points.tolist() == [8.] and "forecast_source" not in default
    extended = cold_start.add_registered_cold_starts(original, bootstrap())
    prior = cold_start.fit_pooled_priors(history(), 3)
    result = run_week.build_future_predictions(extended, ["form"], {}, bootstrap(), 1, 1,
                                              cold_start_priors=prior, **options)
    assert calls == [[11], [11]]
    assert result.player_id.tolist() == [11, 99, 99]
    assert result.predicted_total_points.tolist() == [8., 3., 3.]
    assert result.forecast_source.tolist() == ["model", "pooled_position_team_prior", "pooled_position_team_prior"]
    assert pd.isna(result.prior_std.iloc[0]) and result.prior_count.iloc[1] == 4
    pd.testing.assert_frame_equal(result.loc[[0], default.columns], default)


@pytest.mark.parametrize("kind", ["missing", "wrong_origin", "missing_snapshot_origin"])
def test_invalid_prior_use_fails_before_fixture_fetch_or_model(monkeypatch, kind):
    extended = cold_start.add_registered_cold_starts(snapshot(), bootstrap())
    prior = cold_start.fit_pooled_priors(history(), 4 if kind == "wrong_origin" else 3)
    if kind == "missing_snapshot_origin":
        extended = extended.drop(columns="GW_global")
    monkeypatch.setattr(run_week, "build_team_fixture_map", lambda *a: pytest.fail("invalid prior use fetched fixtures"))
    kwargs = {} if kind == "missing" else {"cold_start_priors": prior}
    with pytest.raises(ValueError):
        run_week.build_future_predictions(extended, ["form"], {}, bootstrap(), 1, 1, **kwargs)


def test_all_cold_fixture_pool_never_calls_models_and_blank_teams_disappear(monkeypatch):
    extended = cold_start.add_registered_cold_starts(snapshot(), bootstrap())
    only_new = extended.loc[extended.is_cold_start]
    prior = cold_start.fit_pooled_priors(history(), 3)
    monkeypatch.setattr(run_week, "predict_by_position", lambda *a, **k: pytest.fail("all-cold pool reached models"))
    monkeypatch.setattr(run_week, "build_team_fixture_map", lambda bootstrap, gw: {"Tottenham": [("A", True, 3)]})
    result = run_week.build_future_predictions(only_new, ["form"], {}, bootstrap(), 1, 1, cold_start_priors=prior)
    assert result.player_id.tolist() == [99] and result.predicted_total_points.tolist() == [3.]
    # Known team A has a blank: it contributes no fixture row to the same pool.
    result = run_week.build_future_predictions(extended, ["form"], {}, bootstrap(), 1, 1, cold_start_priors=prior)
    assert result.player_id.tolist() == [99]
