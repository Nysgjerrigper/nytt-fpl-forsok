"""Recorded-round sequence alignment, fixture causality and origin snapshots."""
import numpy as np
import pandas as pd
import pytest

from fpl import config
from fpl.model.sequence_features import add_sequence_features, sequence_columns, sequence_snapshot


@pytest.fixture(autouse=True)
def sequence_config(monkeypatch):
    monkeypatch.setattr(config, "CNN_SEQUENCE_LENGTH", 8, raising=False)
    monkeypatch.setattr(config, "CNN_SEQUENCE_STATS", ("total_points", "minutes", "goals_scored", "assists",
                                                     "bps", "influence", "creativity", "threat"), raising=False)


def _frame(gws=(1, 2, 2, 3), players=None):
    frame = pd.DataFrame({"player_id": players or [1] * len(gws), "GW_global": gws,
                          "fixture": np.arange(len(gws)) + 100, "season": "2024-25"})
    for stat in config.CNN_SEQUENCE_STATS:
        frame[stat] = np.arange(len(gws), dtype=float) + 1
    return frame


def test_column_order_and_configurable_length(monkeypatch):
    columns = sequence_columns()
    assert len(columns) == 72
    assert columns[:8] == [f"cnn_total_points_t{i}" for i in range(8)]
    assert columns[-8:] == [f"cnn_present_t{i}" for i in range(8)]
    monkeypatch.setattr(config, "CNN_SEQUENCE_LENGTH", 2)
    assert len(sequence_columns()) == 18
    assert sequence_columns()[:2] == ["cnn_total_points_t0", "cnn_total_points_t1"]


def test_dgw_prior_sums_identical_current_sequences_and_no_own_outcome():
    raw = _frame()
    result = add_sequence_features(raw)
    assert result.loc[1, "cnn_total_points_t7"] == 1
    pd.testing.assert_series_equal(result.loc[1, sequence_columns()], result.loc[2, sequence_columns()], check_names=False)
    assert result.loc[3, "cnn_total_points_t7"] == 5
    assert result.loc[3, "cnn_total_points_t6"] == 1
    changed = raw.copy()
    changed.loc[[1, 2], "total_points"] = [100, 200]
    later = add_sequence_features(changed)
    pd.testing.assert_frame_equal(result.iloc[:3][sequence_columns()], later.iloc[:3][sequence_columns()])
    assert later.loc[3, "cnn_total_points_t7"] == 300


def test_future_perturbation_leaves_earlier_sequences_unchanged():
    raw = _frame(tuple(range(1, 12)))
    original = add_sequence_features(raw)
    altered = raw.copy()
    altered.loc[altered.GW_global >= 9, list(config.CNN_SEQUENCE_STATS)] = -1000
    changed = add_sequence_features(altered)
    pd.testing.assert_frame_equal(original.iloc[:9][sequence_columns()], changed.iloc[:9][sequence_columns()])
    assert original.loc[10, "cnn_total_points_t0"] == 3
    assert original.loc[10, "cnn_total_points_t7"] == 10


def test_rows_order_and_duplicate_index_preserved():
    raw = _frame().iloc[[3, 1, 0, 2]].copy()
    raw.index = pd.Index([9, 9, 4, 4], name="original")
    result = add_sequence_features(raw)
    pd.testing.assert_frame_equal(result[raw.columns], raw)
    assert result.index.equals(raw.index)
    assert result.iloc[0]["cnn_total_points_t7"] == 5
    assert result.iloc[1]["cnn_total_points_t7"] == result.iloc[3]["cnn_total_points_t7"] == 1


def test_long_absence_and_season_transition_use_stable_recorded_history():
    raw = _frame((10, 80, 81))
    raw["season"] = ["2020-21", "2022-23", "2022-23"]
    result = add_sequence_features(raw)
    assert result.loc[1, "cnn_total_points_t7"] == 1
    assert result.loc[2, "cnn_total_points_t6"] == 1
    assert result.loc[2, "cnn_total_points_t7"] == 2
    assert result.loc[2, "cnn_present_t6"] == 1
    assert result.loc[2, "cnn_present_t5"] == 0


def test_missing_is_not_zero_or_padding():
    raw = _frame((1, 1, 2, 3))
    raw["total_points"] = [np.nan, np.nan, 0, 2]
    result = add_sequence_features(raw)
    assert np.isnan(result.loc[2, "cnn_total_points_t7"])
    assert result.loc[2, "cnn_present_t7"] == 1
    assert result.loc[2, "cnn_present_t6"] == 0
    assert result.loc[3, "cnn_total_points_t7"] == 0
    assert result.loc[3, "cnn_present_t7"] == 1
    raw.loc[1, "total_points"] = 5
    assert add_sequence_features(raw).loc[2, "cnn_total_points_t7"] == 5


def test_snapshot_matches_synthetic_next_round_transform():
    raw = _frame((1, 2, 2, 10, 2, 6), [1, 1, 1, 1, 2, 2])
    snapshot = sequence_snapshot(raw).set_index("player_id")
    next_rows = _frame((11, 11), [1, 2])
    transformed = add_sequence_features(pd.concat([raw, next_rows], ignore_index=True)).tail(2).set_index("player_id")
    pd.testing.assert_frame_equal(snapshot[sequence_columns()], transformed[sequence_columns()])
    assert list(sequence_snapshot(raw).columns) == ["player_id", *sequence_columns()]
    assert len(snapshot) == 2


def test_single_history_row_snapshot_includes_latest_transform_excludes_it():
    raw = _frame((10,))
    snapshot = sequence_snapshot(raw)
    assert snapshot.loc[0, "cnn_total_points_t7"] == 1
    assert snapshot.loc[0, "cnn_present_t7"] == 1
    assert snapshot.loc[0, "cnn_present_t6"] == 0
    transformed = add_sequence_features(raw)
    assert transformed.filter(regex="^cnn_present_").to_numpy().sum() == 0
    assert transformed.filter(regex="^cnn_total_points_").isna().all().all()


def test_empty_snapshot_schema_and_empty_transform_refusal():
    assert sequence_snapshot(pd.DataFrame()).columns.tolist() == ["player_id", *sequence_columns()]
    with pytest.raises(ValueError, match="nonempty"):
        add_sequence_features(pd.DataFrame())


@pytest.mark.parametrize("change", ["duplicate_fixture", "missing_fixture", "nan_fixture", "collision", "missing_stat"])
def test_invalid_fixture_schema_refused(change):
    raw = _frame()
    if change == "duplicate_fixture":
        raw.loc[2, "fixture"] = raw.loc[1, "fixture"]
    elif change == "missing_fixture":
        raw = raw.drop(columns="fixture")
    elif change == "nan_fixture":
        raw.loc[2, "fixture"] = np.nan
    elif change == "collision":
        raw[sequence_columns()[0]] = 0
    else:
        raw = raw.drop(columns="minutes")
    with pytest.raises(ValueError):
        add_sequence_features(raw)
    with pytest.raises(ValueError):
        sequence_snapshot(raw)


@pytest.mark.parametrize("column", ["player_id", "GW_global"])
@pytest.mark.parametrize("value", [0, -1, 1.5, np.nan, np.inf])
def test_invalid_positive_integer_identifiers(column, value):
    raw = _frame()
    raw[column] = raw[column].astype(float)
    raw.loc[0, column] = value
    with pytest.raises(ValueError, match="positive integers"):
        add_sequence_features(raw)


@pytest.mark.parametrize("value", [np.inf, -np.inf, "invalid"])
def test_invalid_statistics_refused(value):
    raw = _frame()
    raw["minutes"] = raw["minutes"].astype(object)
    raw.loc[0, "minutes"] = value
    with pytest.raises(ValueError):
        add_sequence_features(raw)


@pytest.mark.parametrize("column", ["player_id", "GW_global"])
def test_nullable_missing_identifiers_and_boolean_codes_refused(column):
    raw = _frame()
    raw[column] = raw[column].astype("Int64")
    raw.loc[0, column] = pd.NA
    with pytest.raises(ValueError, match="positive integers"):
        add_sequence_features(raw)
    raw[column] = True
    with pytest.raises(ValueError, match="positive integers"):
        add_sequence_features(raw)


def test_all_missing_nullable_statistic_retains_observed_presence():
    raw = _frame((1, 2))
    raw["minutes"] = pd.Series([pd.NA, pd.NA], dtype="Float64")
    result = add_sequence_features(raw)
    assert np.isnan(result.loc[1, "cnn_minutes_t7"])
    assert result.loc[1, "cnn_present_t7"] == 1
