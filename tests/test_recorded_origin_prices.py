"""Lagged-price admission; these checks do not prove historical publication timing."""
import numpy as np
import pandas as pd
import pytest

from fpl.data.origin_prices import build_origin_price_catalog, validate_origin_price_catalog


def history():
    return pd.DataFrame({"player_id": [1, 1, 1, 1, 2, 2, 3], "GW_global": [1, 2, 2, 4, 1, 3, 5],
                         "value": [50., 51., 51., 99., 60., 62., 70.]})


def test_latest_strictly_prior_prices_deduplicate_dgw_and_omit_unknowns():
    result = build_origin_price_catalog(history(), [1, 2, 3, 999], 2, 4)
    assert result.columns.tolist() == ["player_id", "origin_gw", "value", "price_observed_gw"]
    assert result.to_dict("records") == [
        {"player_id": 1, "origin_gw": 2, "value": 50., "price_observed_gw": 1},
        {"player_id": 2, "origin_gw": 2, "value": 60., "price_observed_gw": 1},
        {"player_id": 1, "origin_gw": 3, "value": 51., "price_observed_gw": 2},
        {"player_id": 2, "origin_gw": 3, "value": 60., "price_observed_gw": 1},
        {"player_id": 1, "origin_gw": 4, "value": 51., "price_observed_gw": 2},
        {"player_id": 2, "origin_gw": 4, "value": 62., "price_observed_gw": 3}]
    assert 999 not in result.player_id.values and 3 not in result.player_id.values
    pd.testing.assert_frame_equal(validate_origin_price_catalog(result, [1, 2, 3, 999], [2, 3, 4]), result)


def test_current_round_and_future_price_changes_cannot_backdate_catalog():
    raw = history()
    expected = build_origin_price_catalog(raw, [1, 2, 3], 2, 4)
    perturbed = raw.copy()
    for future_price in (np.nan, np.inf):
        perturbed.loc[perturbed.GW_global >= 4, "value"] = future_price
        pd.testing.assert_frame_equal(build_origin_price_catalog(perturbed, [1, 2, 3], 2, 4), expected)
    at_three = build_origin_price_catalog(raw, [2], 3, 3)
    assert at_three.value.tolist() == [60.]
    assert at_three.price_observed_gw.tolist() == [1]


def test_ambiguous_same_round_prices_fail():
    raw = history()
    raw.loc[2, "value"] = 52.
    with pytest.raises(ValueError, match="Ambiguous"):
        build_origin_price_catalog(raw, [1], 2, 4)


def test_no_prior_price_is_empty_without_any_fallback():
    result = build_origin_price_catalog(history(), [3, 999], 2, 4)
    assert result.empty and result.columns.tolist() == ["player_id", "origin_gw", "value", "price_observed_gw"]
    with pytest.raises(ValueError, match="nonempty price catalog"):
        validate_origin_price_catalog(result, [3, 999], [2, 3, 4])


@pytest.mark.parametrize("bounds", [(0, 4), (4, 2), (2.5, 4), (True, 4)])
def test_invalid_bounds(bounds):
    with pytest.raises(ValueError):
        build_origin_price_catalog(history(), [1], *bounds)


@pytest.mark.parametrize("column,value", [("player_id", 0), ("player_id", 1.5), ("GW_global", np.nan),
                                         ("value", 0.), ("value", np.inf), ("value", np.nan)])
def test_invalid_raw_values(column, value):
    raw = history()
    raw[column] = raw[column].astype(float)
    raw.loc[0, column] = value
    with pytest.raises(ValueError):
        build_origin_price_catalog(raw, [1], 2, 4)


@pytest.mark.parametrize("column", ["player_id", "GW_global", "value"])
def test_mixed_boolean_numeric_raw_values_rejected(column):
    raw = history()
    raw[column] = raw[column].astype(object)
    raw.loc[0, column] = True
    with pytest.raises(ValueError):
        build_origin_price_catalog(raw, [1], 2, 4)


@pytest.mark.parametrize("players", [[], [True, 2], [0], [1.5], ["1"]])
def test_invalid_requested_ids(players):
    with pytest.raises(ValueError):
        build_origin_price_catalog(history(), players, 2, 4)


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown_player", "unknown_origin", "same_round", "future_round", "nonfinite", "boolean"])
def test_invalid_catalog(change):
    result = build_origin_price_catalog(history(), [1, 2], 2, 4)
    if change == "missing":
        result = result.drop(columns="price_observed_gw")
    elif change == "duplicate":
        result = pd.concat([result, result.iloc[[0]]])
    elif change == "unknown_player":
        result.loc[0, "player_id"] = 99
    elif change == "unknown_origin":
        result.loc[0, "origin_gw"] = 9
    elif change == "same_round":
        result.loc[0, "price_observed_gw"] = 2
    elif change == "future_round":
        result.loc[0, "price_observed_gw"] = 3
    elif change == "nonfinite":
        result.loc[0, "value"] = np.inf
    else:
        result["origin_gw"] = result.origin_gw.astype(object)
        result.loc[0, "origin_gw"] = True
    with pytest.raises(ValueError):
        validate_origin_price_catalog(result, [1, 2], [2, 3, 4])


def test_missing_origin_rejected_but_missing_player_allowed_and_extras_rejected():
    result = build_origin_price_catalog(history(), [1], 2, 4)
    validated = validate_origin_price_catalog(result, [1, 2], [2, 3, 4])
    assert validated.player_id.tolist() == [1, 1, 1]
    with pytest.raises(ValueError, match="nonempty price catalog"):
        validate_origin_price_catalog(result.iloc[:2], [1, 2], [2, 3, 4])
    with pytest.raises(ValueError, match="exactly the four"):
        validate_origin_price_catalog(result.assign(note="lagged proxy"), [1, 2], [2, 3, 4])


def test_empty_admitted_history_returns_empty_canonical_schema():
    raw = pd.DataFrame(columns=["player_id", "GW_global", "value"])
    result = build_origin_price_catalog(raw, [1], 2, 3)
    assert result.empty and result.columns.tolist() == ["player_id", "origin_gw", "value", "price_observed_gw"]
