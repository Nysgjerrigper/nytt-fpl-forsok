"""Offline contract tests for upstream prepared forecast inputs."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from fpl.milp import solio_inputs
from fpl.milp.solio_inputs import prepare_inputs


@pytest.fixture
def forecasts():
    return pd.DataFrame([
        {"player_id": 100, "name": "Keeper", "position": "GK", "team": "Arsenal", "value": 50,
         "fixture": 101, "GW_global": 153, "predicted_total_points": 2.0, "predicted_minutes": 60.0, "actual_total_points": 999},
        {"player_id": 100, "name": "Keeper", "position": "GK", "team": "Arsenal", "value": 50,
         "fixture": 102, "GW_global": 153, "predicted_total_points": 3.0, "predicted_minutes": 70.0, "actual_total_points": 999},
        {"player_id": 200, "name": "Forward", "position": "FWD", "team": "Chelsea", "value": 60,
         "fixture": 103, "GW_global": 154, "predicted_total_points": -1.0, "predicted_minutes": 45.0, "actual_total_points": 999},
    ])


def prepare(frame, **kwargs):
    return prepare_inputs(frame, season="2024-25", start_gw=153, horizon=2, **kwargs)


def squad_frame():
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    return pd.DataFrame({"player_id": range(1, 16), "name": [f"Player {i}" for i in range(15)],
                         "position": positions, "team": [f"Club {i % 5}" for i in range(15)],
                         "value": [50] * 15, "GW_global": [153] * 15,
                         "predicted_total_points": [4.0] * 15})


def bootstrap():
    return {"teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Chelsea"}],
            "elements": [{"id": 9, "code": 100, "team": 2, "element_type": 2,
                          "now_cost": 55, "web_name": "Current keeper"},
                         {"id": 3, "code": 200, "team": 1, "element_type": 4,
                          "now_cost": 65, "web_name": "Current forward"}]}


def test_dgw_sums_blanks_and_global_mapping_without_mutating_input(forecasts):
    original = forecasts.copy(deep=True)
    result = prepare(forecasts)
    table = result["merged_data"]
    assert table.loc[100, "1_Pts"] == 5
    assert table.loc[100, "1_xMins"] == 130
    assert table.loc[100, "2_Pts"] == table.loc[100, "2_xMins"] == 0
    assert table.loc[200, "1_Pts"] == 0
    assert table.loc[200, "2_Pts"] == -1
    assert result["next_gw"] == 1 and result["season_offset"] == 152
    assert result["buy_price"] == {100: 5.0, 200: 6.0}
    assert result["itb"] == 100 and result["ft"] == result["ft_base"] == 0
    assert result["fixtures"] == result["initial_squad"] == []
    assert result["max_players_from_team"] == 0
    assert result["type_data"].squad_min_play.tolist() == [1, 3, 2, 1]
    assert result["type_data"].squad_max_play.tolist() == [1, 5, 5, 3]
    assert result["type_data"].squad_select.tolist() == [2, 5, 5, 3]
    assert not any("actual" in column for column in table)
    pd.testing.assert_frame_equal(forecasts, original)


def test_csv_gw_alias_and_explicit_display_minutes_setting(forecasts, monkeypatch):
    monkeypatch.setattr(solio_inputs, "DEFAULT_DISPLAY_MINUTES", 75.0)
    result = prepare(forecasts.drop(columns="predicted_minutes").rename(columns={"GW_global": "GW"}))
    assert result["merged_data"].loc[100, "1_xMins"] == 75
    assert result["merged_data"].loc[100, "2_xMins"] == 0


def test_bootstrap_authoritative_overlay_and_reverse_mapping(forecasts):
    result = prepare(forecasts, bootstrap=bootstrap())
    table = result["merged_data"]
    assert table.index.tolist() == [3, 9]
    assert result["stable_to_element"] == {100: 9, 200: 3}
    assert result["element_to_stable"] == {9: 100, 3: 200}
    assert table.loc[9, ["name", "web_name", "Pos", "element_type"]].tolist() == ["Chelsea", "Current keeper", "D", 2]
    assert table.loc[9, "1_Pts"] == 5
    assert result["buy_price"] == {9: 5.5, 3: 6.5}


def test_owned_state_units_and_missing_round_completion():
    frame = squad_frame()
    result = prepare(frame, initial_squad=list(range(1, 16)), bank=15, free_transfers=5,
                     sell_prices={1: 45})
    assert result["initial_squad"] == list(range(1, 16))
    assert result["itb"] == 1.5
    assert result["ft"] == result["ft_base"] == 5
    assert result["sell_price"][1] == 4.5
    assert result["sell_price"][2] == 5
    assert result["price_modified_players"] == [1]
    assert result["max_players_from_team"] == 3
    assert result["merged_data"]["2_Pts"].eq(0).all()


@pytest.mark.parametrize("column,value", [
    ("predicted_total_points", np.nan), ("predicted_total_points", np.inf),
    ("value", np.inf), ("value", -1), ("player_id", np.nan),
    ("player_id", 1.5), ("GW_global", 153.5), ("GW_global", 191),
    ("position", "UNK"), ("team", None), ("name", ""),
    ("predicted_minutes", np.inf), ("predicted_minutes", -1),
])
def test_invalid_forecast_or_metadata_rejected(forecasts, column, value):
    frame = forecasts.astype({column: object})
    frame.loc[0, column] = value
    with pytest.raises((ValueError, TypeError)):
        prepare(frame)


@pytest.mark.parametrize("column,value", [("value", 51), ("team", "Liverpool"),
                                           ("position", "DEF"), ("name", "Other")])
def test_static_metadata_changes_rejected(forecasts, column, value):
    forecasts.loc[1, column] = value
    with pytest.raises(ValueError, match="static-price limitation"):
        prepare(forecasts)


@pytest.mark.parametrize("kwargs", [{"bank": -1}, {"bank": np.inf}, {"free_transfers": 6},
                                     {"free_transfers": -1}, {"free_transfers": 1.5},
                                     {"free_transfers": True}, {"sell_prices": {100: 45}}])
def test_invalid_fresh_state_rejected(forecasts, kwargs):
    with pytest.raises(ValueError):
        prepare(forecasts, **kwargs)


@pytest.mark.parametrize("kind", ["duplicate", "missing", "positions", "clubs", "empty"])
def test_invalid_owned_squad_rejected(kind):
    frame = squad_frame()
    squad = list(range(1, 16))
    if kind == "duplicate":
        squad[-1] = squad[0]
    elif kind == "missing":
        squad[-1] = 999
    elif kind == "positions":
        frame.loc[0, "position"] = "FWD"
    elif kind == "clubs":
        frame.loc[:3, "team"] = "Same club"
    else:
        squad = []
    with pytest.raises(ValueError):
        prepare(frame, initial_squad=squad)


@pytest.mark.parametrize("sell", [np.nan, np.inf, -1, 51])
def test_invalid_owned_sell_price_rejected(sell):
    with pytest.raises(ValueError):
        prepare(squad_frame(), initial_squad=list(range(1, 16)), sell_prices={1: sell})


@pytest.mark.parametrize("kind", ["code", "id", "unregistered", "club", "position", "price"])
def test_invalid_bootstrap_rejected(forecasts, kind):
    source = deepcopy(bootstrap())
    if kind in {"code", "id"}:
        source["elements"][1][kind] = source["elements"][0][kind]
    elif kind == "unregistered":
        source["elements"][0]["code"] = 999
    elif kind == "club":
        source["elements"][0]["team"] = 99
    elif kind == "position":
        source["elements"][0]["element_type"] = 5
    else:
        source["elements"][0]["now_cost"] = np.inf
    with pytest.raises(ValueError):
        prepare(forecasts, bootstrap=source)


@pytest.mark.parametrize("season,start,horizon", [("2024-26", 153, 2), ("2024-25", 152, 2),
                                                 ("2024-25", 190, 2), ("2024-25", 153, 0)])
def test_invalid_season_window_rejected(forecasts, season, start, horizon):
    with pytest.raises(ValueError):
        prepare_inputs(forecasts, season=season, start_gw=start, horizon=horizon)


def test_conflicting_global_week_alias_rejected(forecasts):
    forecasts["GW"] = forecasts.GW_global + 1
    with pytest.raises(ValueError, match="disagree"):
        prepare(forecasts)


def test_team_name_correction(forecasts):
    forecasts.loc[forecasts.player_id.eq(100), "team"] = "Spurs"
    result = prepare(forecasts)
    assert result["merged_data"].loc[100, "name"] == "Tottenham"


def test_fresh_fixed_budget_and_no_transfers(forecasts):
    assert prepare(forecasts, free_transfers=5)["ft"] == 0
    with pytest.raises(ValueError, match="fresh mode requires"):
        prepare(forecasts, bank=999)


def test_owned_bootstrap_maps_squad_and_sell_prices():
    frame = squad_frame()
    clubs = [{"id": i + 1, "name": f"Club {i}"} for i in range(5)]
    positions = frame.position.map({"GK": 1, "DEF": 2, "MID": 3, "FWD": 4})
    source = {"teams": clubs, "elements": [
        {"code": code, "id": code + 100, "team": (code - 1) % 5 + 1,
         "element_type": int(positions.iloc[code - 1]), "now_cost": 55}
        for code in range(1, 16)]}
    result = prepare(frame, initial_squad=list(range(1, 16)), bootstrap=source,
                     sell_prices={1: 50}, bank=25, free_transfers=0)
    assert result["initial_squad"] == list(range(101, 116))
    assert result["sell_price"][101] == 5.0
    assert result["buy_price"][101] == 5.5
    assert result["price_modified_players"] == [101]
    assert result["ft"] == result["ft_base"] == 0 and result["itb"] == 2.5


def test_input_order_does_not_change_prepared_tables(forecasts):
    first = prepare(forecasts)
    shuffled = prepare(forecasts.sample(frac=1, random_state=2))
    for key in ("merged_data", "team_data", "type_data"):
        pd.testing.assert_frame_equal(first[key], shuffled[key])


@pytest.mark.parametrize("origins", [[153, 153, 153], [152, 153, 153],
                                     [152, 152, 152], [153, None, 153], [153.5] * 3])
def test_single_requested_origin_admission(forecasts, origins):
    forecasts["origin_gw"] = origins
    if origins == [153, 153, 153]:
        assert prepare(forecasts)["merged_data"].loc[100, "1_Pts"] == 5
    else:
        with pytest.raises(ValueError):
            prepare(forecasts)


@pytest.mark.parametrize("kind", ["fixture", "forecast_fixture_id", "opponent_home"])
def test_distinct_fixture_identity_forms_and_duplicate_rejection(forecasts, kind):
    if kind == "forecast_fixture_id":
        forecasts = forecasts.drop(columns="fixture")
        forecasts["forecast_fixture_id"] = ["match-a", "match-b", "match-c"]
    elif kind == "opponent_home":
        forecasts = forecasts.drop(columns="fixture")
        forecasts["opponent_team"] = ["Liverpool", "Liverpool", "Arsenal"]
        forecasts["was_home"] = [True, False, True]
    assert prepare(forecasts)["merged_data"].loc[100, "1_Pts"] == 5
    repeated = pd.concat([forecasts, forecasts.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate player/gameweek/fixture"):
        prepare(repeated)


def test_repeated_rounds_without_fixture_identity_rejected(forecasts):
    with pytest.raises(ValueError, match="ambiguous repeated"):
        prepare(forecasts.drop(columns="fixture"))


def test_single_round_rows_need_no_fixture_identity(forecasts):
    frame = forecasts.drop_duplicates(["player_id", "GW_global"]).drop(columns="fixture")
    assert prepare(frame)["merged_data"].loc[100, "1_Pts"] == 2


@pytest.mark.parametrize("identity,value", [("fixture", None), ("fixture", 101.5),
                                            ("forecast_fixture_id", None),
                                            ("forecast_fixture_id", ""),
                                            ("forecast_fixture_id", 101),
                                            ("opponent_team", None),
                                            ("was_home", None), ("was_home", 2)])
def test_malformed_fixture_identity_rejected(forecasts, identity, value):
    if identity != "fixture":
        forecasts = forecasts.drop(columns="fixture")
        if identity == "forecast_fixture_id":
            forecasts["forecast_fixture_id"] = ["a", "b", "c"]
        else:
            forecasts["opponent_team"] = ["Liverpool", "Chelsea", "Arsenal"]
            forecasts["was_home"] = [True, False, True]
    forecasts = forecasts.astype({identity: object})
    forecasts.loc[0, identity] = value
    with pytest.raises(ValueError):
        prepare(forecasts)


def test_origin_panel_cannot_be_summed_as_double_gameweek(forecasts):
    panel = pd.concat([forecasts.assign(origin_gw=153), forecasts.assign(origin_gw=154)])
    with pytest.raises(ValueError, match="origin_gw must equal"):
        prepare(panel)


def test_partial_opponent_home_identity_rejected_for_repeated_rounds(forecasts):
    frame = forecasts.drop(columns="fixture").assign(opponent_team="Liverpool")
    with pytest.raises(ValueError, match="ambiguous repeated"):
        prepare(frame)
