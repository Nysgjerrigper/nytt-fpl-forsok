"""Offline conversion of existing point forecasts into upstream prepared inputs.

Prices remain static across the horizon. DEFAULT_DISPLAY_MINUTES is an explicit
adapter setting used only when predicted_minutes is absent; it is a display
placeholder for zero bench-weight runs and never evidence for realized autosubs.
"""
from __future__ import annotations

import re
from collections import Counter
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fpl import config

DEFAULT_DISPLAY_MINUTES = 90.0
_PRICE_SCALE = 10.0
_FRESH_BUDGET_TENTHS = 1000.0
_MAX_CLUB_PLAYERS = 3
_POSITION_TYPES = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
_POSITION_LABELS = {"GK": "G", "DEF": "D", "MID": "M", "FWD": "F"}
_SQUAD_COUNTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
_SEASON_PATTERN = re.compile(r"^(\d{4})-(\d{2})$")


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _money(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _integer_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame:
        raise ValueError(f"missing identity column {column}")
    values = pd.to_numeric(frame[column], errors="raise")
    if (pd.api.types.is_bool_dtype(values) or not np.isfinite(values).all()
            or not ((values >= 1) & (values % 1 == 0)).all()):
        raise ValueError(f"{column} must contain nonmissing positive integer identities")
    frame[column] = values.astype(int)



def _validate_forecast_admission(frame: pd.DataFrame, start: int) -> None:
    """Reject mixed origins and ambiguous/duplicate player-fixture forecasts."""
    if "origin_gw" in frame:
        _integer_column(frame, "origin_gw")
        if not frame.origin_gw.eq(start).all():
            raise ValueError("origin_gw must equal the requested start_gw for every forecast")
    round_keys = ["player_id", "GW_global"]
    identity_keys: list[str] = []
    if "fixture" in frame:
        _integer_column(frame, "fixture")
        identity_keys = ["fixture"]
    elif "forecast_fixture_id" in frame:
        if not frame.forecast_fixture_id.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError("forecast_fixture_id must contain nonmissing nonempty strings")
        identity_keys = ["forecast_fixture_id"]
    elif {"opponent_team", "was_home"} <= set(frame):
        if not frame.opponent_team.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError("opponent_team must contain nonmissing nonempty strings")
        if not frame.was_home.map(
            lambda value: isinstance(value, (bool, np.bool_, Integral, Real))
            and pd.notna(value) and value in (0, 1)
        ).all():
            raise ValueError("was_home must contain nonmissing boolean or 0/1 values")
        frame["opponent_team"] = frame.opponent_team.replace(config.TEAM_NAME_CORRECTIONS)
        identity_keys = ["opponent_team", "was_home"]
    if identity_keys:
        if frame.duplicated(round_keys + identity_keys).any():
            raise ValueError("duplicate player/gameweek/fixture forecast identity")
    elif frame.duplicated(round_keys).any():
        raise ValueError("ambiguous repeated player/gameweek forecasts require fixture identity")


def prepare_inputs(
    predictions: pd.DataFrame, *, season: str, start_gw: int, horizon: int,
    initial_squad: Sequence[int] | None = None, bank: float = 1000,
    free_transfers: int = 1, sell_prices: Mapping[int, float] | None = None,
    bootstrap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare upstream tables/state without fetching data, fitting, or solving.

    player_id is stable FPL code and GW_global (or the pipeline CSV's GW alias)
    is global gameweek. Per-fixture forecasts sum into player rounds; absent round
    forecasts are blanks with zero points/minutes. Repeated player rounds require
    distinct fixture identities; optional origin_gw must equal start_gw throughout.
    Actual points/minutes never
    enter these inputs. Price, bank and optional owned sell prices are tenths of
    a million pounds and are divided by ten for upstream monetary dictionaries.

    Metadata must be constant within the requested horizon: changing prices,
    clubs or positions require an explicit adapter design beyond this static-price
    interface. Fresh mode requires the upstream fixed £100m budget and starts
    with zero banked transfers; continuing mode preserves the provided state.
    Bootstrap, when supplied, authoritatively overlays current element
    IDs, club, position and buy price; no unregistered forecast player is dropped.
    """
    match = _SEASON_PATTERN.fullmatch(season) if isinstance(season, str) else None
    if not match or int(match[2]) != (int(match[1]) + 1) % 100:
        raise ValueError("season must be a consecutive YYYY-YY season")
    offset = config.season_start_gw(season) - 1
    start = _integer(start_gw, "start_gw", minimum=1)
    length = _integer(horizon, "horizon", minimum=1)
    if start <= offset or start + length - 1 > offset + config.GWS_PER_SEASON:
        raise ValueError("requested horizon must lie within the same season")
    bank_value = _money(bank, "bank")
    if initial_squad is None and bank_value != _FRESH_BUDGET_TENTHS:
        raise ValueError("fresh mode requires bank=1000 (upstream fixed £100m budget)")
    ft = _integer(free_transfers, "free_transfers")
    if ft > config.MODERN_MAX_FREE_TRANSFERS:
        raise ValueError("free_transfers must lie between 0 and 5")
    if initial_squad is None:
        ft = 0
    required = {"player_id", "name", "position", "team", "value", "predicted_total_points"}
    if missing := required - set(predictions):
        raise ValueError(f"predictions missing metadata/forecast columns: {sorted(missing)}")
    frame = predictions.copy()
    if "GW_global" not in frame:
        if "GW" not in frame:
            raise ValueError("predictions missing GW_global (or global GW alias)")
        frame["GW_global"] = frame["GW"]
    elif "GW" in frame and not frame.GW_global.equals(frame.GW):
        raise ValueError("GW and GW_global disagree")
    for column in ("player_id", "GW_global"):
        _integer_column(frame, column)
    _validate_forecast_admission(frame, start)
    if not frame.GW_global.between(offset + 1, offset + config.GWS_PER_SEASON).all():
        raise ValueError("forecast gameweeks must lie within the same season")
    frame = frame[frame.GW_global.between(start, start + length - 1)].copy()
    if frame.empty:
        raise ValueError("requested horizon has no forecast metadata")
    for column in ("predicted_total_points", "value"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{column} must be finite and nonmissing")
    if (frame.value < 0).any():
        raise ValueError("value must be nonnegative")
    for column in ("name", "position", "team"):
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{column} metadata must contain nonmissing strings")
    frame["team"] = frame.team.replace(config.TEAM_NAME_CORRECTIONS)
    if not frame.position.isin(_POSITION_TYPES).all():
        raise ValueError("unknown position metadata")
    if not (frame.groupby("player_id")[["name", "position", "team", "value"]].nunique(dropna=False) == 1).all().all():
        raise ValueError("static-price limitation: player metadata/price must be constant across the horizon")
    metadata = frame.drop_duplicates("player_id").set_index("player_id")[["name", "position", "team", "value"]].sort_index()
    stable_to_element = {int(code): int(code) for code in metadata.index}
    if bootstrap is not None:
        elements = pd.DataFrame(bootstrap.get("elements", []))
        teams = pd.DataFrame(bootstrap.get("teams", []))
        for column in ("id", "code", "team", "element_type"):
            _integer_column(elements, column)
        _integer_column(teams, "id")
        if elements.id.duplicated().any() or elements.code.duplicated().any():
            raise ValueError("bootstrap element IDs or stable codes are duplicated")
        if teams.id.duplicated().any() or "name" not in teams:
            raise ValueError("bootstrap team IDs/names are missing or duplicated")
        if not teams.name.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError("bootstrap team names must be nonmissing strings")
        teams["name"] = teams.name.replace(config.TEAM_NAME_CORRECTIONS)
        if teams.name.duplicated().any():
            raise ValueError("bootstrap normalized team names are duplicated")
        if "now_cost" not in elements:
            raise ValueError("bootstrap missing current buy prices")
        elements["now_cost"] = pd.to_numeric(elements.now_cost, errors="raise")
        if not np.isfinite(elements.now_cost).all() or (elements.now_cost < 0).any():
            raise ValueError("bootstrap buy prices must be finite and nonnegative")
        if not elements.team.isin(teams.id).all() or not elements.element_type.isin(_POSITION_TYPES.values()).all():
            raise ValueError("bootstrap has missing club or unknown position metadata")
        elements = elements.set_index("code")
        if missing := set(metadata.index) - set(elements.index):
            raise ValueError(f"unregistered forecast stable codes: {sorted(missing)}")
        current = elements.reindex(metadata.index)
        stable_to_element = {int(code): int(element) for code, element in current.id.items()}
        metadata["team"] = current.team.map(teams.set_index("id").name)
        metadata["position"] = current.element_type.map({value: key for key, value in _POSITION_TYPES.items()})
        metadata["value"] = current.now_cost
        if "web_name" in current:
            names = current.web_name
            if not names.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
                raise ValueError("bootstrap player display names must be nonmissing strings")
            metadata["name"] = names
        team_data = teams[["id", "name"]].sort_values("id").reset_index(drop=True)
    else:
        names = sorted(metadata.team.unique())
        team_data = pd.DataFrame({"id": range(1, len(names) + 1), "name": names})
    owned = [] if initial_squad is None else [_integer(code, "initial squad player", minimum=1) for code in initial_squad]
    if len(set(owned)) != len(owned):
        raise ValueError("initial squad contains duplicate players")
    if missing := set(owned) - set(metadata.index):
        raise ValueError(f"initial owned players lack forecast metadata: {sorted(missing)}")
    owned_metadata = metadata.loc[owned]
    if initial_squad is not None:
        if len(owned) != sum(_SQUAD_COUNTS.values()) or owned_metadata.position.value_counts().to_dict() != _SQUAD_COUNTS:
            raise ValueError("initial squad must contain 15 players with positions 2/5/5/3")
        if (owned_metadata.team.value_counts() > _MAX_CLUB_PLAYERS).any():
            raise ValueError("initial squad exceeds the three-player club limit")
    supplied_sells = {} if sell_prices is None else dict(sell_prices)
    for code in supplied_sells:
        _integer(code, "sell price player", minimum=1)
    if set(supplied_sells) - set(owned):
        raise ValueError("sell prices may be supplied only for owned players")
    buy_price = {stable_to_element[int(code)]: float(price) / _PRICE_SCALE for code, price in metadata.value.items()}
    sell_price = {}
    for code in owned:
        sell = _money(supplied_sells.get(code, metadata.loc[code, "value"]), "sell price")
        if sell > metadata.loc[code, "value"]:
            raise ValueError("owned sell price cannot exceed current buy price")
        sell_price[stable_to_element[code]] = sell / _PRICE_SCALE
    local_weeks = list(range(start - offset, start - offset + length))
    frame["local_week"] = frame.GW_global - offset
    points = frame.groupby(["player_id", "local_week"]).predicted_total_points.sum().unstack(fill_value=0)
    points = points.reindex(index=metadata.index, columns=local_weeks, fill_value=0).fillna(0)
    if not np.isfinite(points.to_numpy()).all():
        raise ValueError("summed player-round points must be finite")
    if "predicted_minutes" in frame:
        frame["predicted_minutes"] = pd.to_numeric(frame.predicted_minutes, errors="raise")
        if not np.isfinite(frame.predicted_minutes).all() or (frame.predicted_minutes < 0).any():
            raise ValueError("predicted_minutes must be finite and nonnegative")
        minutes = frame.groupby(["player_id", "local_week"]).predicted_minutes.sum().unstack(fill_value=0)
        minutes = minutes.reindex(index=metadata.index, columns=local_weeks, fill_value=0).fillna(0)
    else:
        display_minutes = _money(DEFAULT_DISPLAY_MINUTES, "DEFAULT_DISPLAY_MINUTES")
        minutes = frame.groupby(["player_id", "local_week"]).size().unstack(fill_value=0)
        minutes = minutes.reindex(index=metadata.index, columns=local_weeks, fill_value=0).fillna(0).gt(0).astype(float) * display_minutes
    if not np.isfinite(minutes.to_numpy()).all():
        raise ValueError("summed player-round minutes must be finite")
    merged_data = pd.DataFrame(index=metadata.index)
    merged_data["element_type"] = metadata.position.map(_POSITION_TYPES)
    merged_data["Pos"] = metadata.position.map(_POSITION_LABELS)
    merged_data["name"] = metadata.team
    merged_data["web_name"] = metadata["name"]
    merged_data["now_cost"] = metadata.value
    merged_data["team"] = metadata.team.map(team_data.set_index("name").id)
    for week in local_weeks:
        merged_data[f"{week}_Pts"] = points[week]
        merged_data[f"{week}_xMins"] = minutes[week]
    merged_data.index = pd.Index([stable_to_element[int(code)] for code in merged_data.index], name="element")
    merged_data["ID"] = merged_data.index
    merged_data = merged_data.sort_index()
    type_data = pd.DataFrame({
        "singular_name_short": list(_POSITION_TYPES),
        "squad_min_play": [config.LINEUP_MIN_COUNTS[position] for position in _POSITION_TYPES],
        "squad_max_play": [config.LINEUP_MAX_COUNTS[position] for position in _POSITION_TYPES],
        "squad_select": [_SQUAD_COUNTS[position] for position in _POSITION_TYPES],
    }, index=pd.Index(list(_POSITION_TYPES.values()), name="id"))
    return {"merged_data": merged_data, "team_data": team_data, "type_data": type_data,
            "next_gw": start - offset, "initial_squad": [stable_to_element[code] for code in owned],
            "itb": bank_value / _PRICE_SCALE, "ft": ft, "ft_base": ft,
            "buy_price": buy_price, "sell_price": sell_price,
            "price_modified_players": [element for element, sell in sell_price.items() if sell != buy_price[element]],
            "max_players_from_team": max(Counter(owned_metadata.team).values(), default=0), "fixtures": [],
            "stable_to_element": stable_to_element,
            "element_to_stable": {element: code for code, element in stable_to_element.items()},
            "season_offset": offset}
