"""Synthetic feature-contract checks; these do not prove archive admissibility."""
import numpy as np
import pandas as pd
import pytest

from fpl.model.deadline_availability import add_availability_features, availability_columns


def roster():
    return pd.DataFrame({
        "player_id": [101, 102, 101], "origin_gw": [137, 137, 138],
        "status": ["a", "i", "d"], "chance_of_playing_next_round": [None, 0, 75],
        "snapshot_age_hours": [2.0, 2.0, 1.0],
        "captured_at": ["2024-01-01T10:00:00Z", "2024-01-01T10:00:00Z", "2024-01-08T11:00:00Z"],
        "deadline": ["2024-01-01T12:00:00Z", "2024-01-01T12:00:00Z", "2024-01-08T12:00:00Z"],
    })


def fixtures():
    return pd.DataFrame({
        "player_id": [102, 101, 103, 101, 101], "GW_global": [137, 137, 137, 137, 138],
        "fixture": [2, 1, 3, 4, 5], "total_points": [5, 8, 0, 9, 3],
    }, index=pd.Index([9, 9, 2, 9, 1], name="original"))


def test_order_dgw_metadata_null_zero_and_absence():
    frame = fixtures()
    frame.attrs = {"frozen_dataframe_fingerprint": "old", "source": "archive"}
    before = frame.copy(deep=True)
    result = add_availability_features(frame, roster())
    assert availability_columns() == [f"archive_status_{status}" for status in "adisun"] + [
        "archive_chance_next", "archive_capture_age_hours",
    ]
    pd.testing.assert_frame_equal(result[frame.columns], frame)
    pd.testing.assert_frame_equal(frame, before)
    assert result.attrs == {"source": "archive"}
    assert frame.attrs["frozen_dataframe_fingerprint"] == "old"
    assert result.iloc[0].archive_chance_next == 0
    assert np.isnan(result.iloc[1].archive_chance_next)
    assert result.iloc[2][availability_columns()].isna().all()
    np.testing.assert_equal(result.iloc[1][availability_columns()].to_numpy(), result.iloc[3][availability_columns()].to_numpy())
    assert result.iloc[4].archive_chance_next == .75
    assert result.iloc[0].archive_status_i == 1
    assert result.iloc[1].archive_status_a == 1
    assert result.iloc[4].archive_status_d == 1
    assert result.iloc[[0, 1, 3, 4]][availability_columns()[:6]].sum(axis=1).eq(1).all()


def test_empty_canonical_archive_and_empty_fixture_frame():
    result = add_availability_features(fixtures(), roster().iloc[:0])
    assert result[availability_columns()].isna().all().all()
    empty = fixtures().iloc[:0]
    result = add_availability_features(empty, roster())
    assert result.empty
    assert result.index.equals(empty.index)
    assert list(result.columns) == list(empty.columns) + availability_columns()


def test_future_and_outcome_perturbations_do_not_affect_earlier_features():
    frame = fixtures()
    original = add_availability_features(frame, roster())
    changed_roster = roster()
    changed_roster.loc[2, ["status", "chance_of_playing_next_round"]] = ["s", 25]
    changed_frame = frame.copy()
    changed_frame["total_points"] = [999, np.nan, -50, np.inf, 100]
    changed = add_availability_features(changed_frame, changed_roster)
    pd.testing.assert_frame_equal(
        original.loc[original.GW_global.eq(137), availability_columns()],
        changed.loc[changed.GW_global.eq(137), availability_columns()],
    )


@pytest.mark.parametrize("status", list("adisun"))
def test_every_known_status_is_one_hot(status):
    records = roster().iloc[:1].copy()
    records["status"] = status
    result = add_availability_features(fixtures().iloc[[1]], records)
    assert result[availability_columns()[:6]].iloc[0].to_dict() == {
        f"archive_status_{candidate}": float(candidate == status) for candidate in "adisun"
    }


@pytest.mark.parametrize("column,value,match", [
    ("player_id", True, "positive whole"), ("player_id", 0, "positive whole"),
    ("player_id", 101.5, "positive whole"), ("origin_gw", False, "positive whole"),
    ("origin_gw", np.nan, "positive whole"), ("status", "x", "status"),
    ("status", None, "status"), ("chance_of_playing_next_round", True, "chance"),
    ("chance_of_playing_next_round", 60, "chance"), ("chance_of_playing_next_round", np.inf, "chance"),
    ("snapshot_age_hours", 25, "capture age"), ("snapshot_age_hours", 0, "capture age"),
    ("snapshot_age_hours", np.nan, "capture age"), ("snapshot_age_hours", True, "capture age"),
    ("snapshot_age_hours", 3, "consistent"),
    ("captured_at", "2024-01-01T12:00:00Z", "precede"),
    ("captured_at", "2024-01-01T13:00:00Z", "precede"),
    ("captured_at", "2024-01-01T10:00:00", "timezone"),
    ("deadline", "2024-01-01", "timezone"), ("deadline", "invalid", "timezone"),
])
def test_malformed_rosters_fail_even_when_not_in_requested_frame(column, value, match):
    records = roster().astype(object)
    records.loc[2, column] = value
    with pytest.raises(ValueError, match=match):
        add_availability_features(fixtures().iloc[:1], records)


@pytest.mark.parametrize("column,value", [("player_id", True), ("GW_global", False), ("player_id", -1), ("GW_global", 1.2)])
def test_invalid_frame_keys(column, value):
    frame = fixtures().astype(object)
    frame.iloc[0, frame.columns.get_loc(column)] = value
    with pytest.raises(ValueError, match="positive whole"):
        add_availability_features(frame, roster())


def test_duplicates_required_schema_and_collisions():
    with pytest.raises(ValueError, match="duplicate"):
        add_availability_features(fixtures(), pd.concat([roster(), roster().iloc[:1]]))
    for frame, records in (
        (fixtures().drop(columns="GW_global"), roster()),
        (fixtures(), roster().drop(columns="status")),
    ):
        with pytest.raises(ValueError, match="required columns"):
            add_availability_features(frame, records)
    for frame, records in (
        (pd.concat([fixtures(), fixtures()[["fixture"]]], axis=1), roster()),
        (fixtures(), pd.concat([roster(), roster()[["status"]]], axis=1)),
    ):
        with pytest.raises(ValueError, match="unique columns"):
            add_availability_features(frame, records)
    with pytest.raises(ValueError, match="collisions"):
        add_availability_features(fixtures().assign(archive_status_a=0), roster())


def test_timezone_normalization_and_exact_maximum_age():
    records = roster().iloc[:1].copy()
    records["snapshot_age_hours"] = 24.0
    records["captured_at"] = "2023-12-31T14:00:00+02:00"
    result = add_availability_features(fixtures().iloc[[1]], records)
    assert result.iloc[0].archive_capture_age_hours == 24
