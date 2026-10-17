"""Admit pinned origin registers and overlay their static-horizon identities/prices.

Capture hashes bind supplied bytes, not archive authenticity, exact-deadline roster
identity, chip/rule legality or fixture-publication chronology. No forecast, blank
outcome or purchase basis is inferred for an absent registered player.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import gzip
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import re

import numpy as np
import pandas as pd

from fpl import config

_POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
_FILES = {"manifest.json", "bootstrap-static.json.gz"}


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json(content: bytes) -> dict:
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("capture JSON must be an object")
    json.dumps(value, allow_nan=False)
    return value


def _canonical_sha(value: dict) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _frame_sha(frame: pd.DataFrame) -> str:
    """Runtime-bound ordered dataframe fingerprint, including index and columns."""
    return _sha(frame.to_json(orient="split", date_format="iso", double_precision=15).encode())


def _integer(value, label: str, *, maximum: int = 2**63 - 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 < value <= maximum:
        raise ValueError(f"{label} requires a positive non-boolean integer")
    return int(value)


def _identity_column(column: pd.Series, label: str) -> pd.Series:
    if (not column.map(lambda value: isinstance(value, Real) and not isinstance(value, (bool, np.bool_))).all()
            or not np.isfinite(column.to_numpy(dtype=float)).all()
            or not ((column > 0) & (column < 2**63) & (column % 1 == 0)).all()):
        raise ValueError(f"{label} requires finite positive integer identities")
    return column.astype("int64")


def _numeric(column: pd.Series, label: str) -> pd.Series:
    if (not column.map(lambda value: isinstance(value, Real) and not isinstance(value, (bool, np.bool_))).all()
            or not np.isfinite(column.to_numpy(dtype=float)).all()):
        raise ValueError(f"{label} requires finite non-boolean numeric forecasts")
    return column.astype(float)


def _timestamp(value, label: str) -> pd.Timestamp:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} requires an aware timestamp")
    try:
        result = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError(f"{label} requires an aware timestamp")
    return result.tz_convert("UTC")


def _metadata(bootstrap: dict) -> pd.DataFrame:
    if not isinstance(bootstrap.get("teams"), list) or len(bootstrap["teams"]) != 20 or not isinstance(bootstrap.get("elements"), list):
        raise ValueError("origin register requires twenty clubs and explicit elements")
    teams = {}
    for item in bootstrap["teams"]:
        identifier = _integer(item.get("id"), "API team id")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("API club names must be nonblank")
        name = config.TEAM_NAME_CORRECTIONS.get(name.strip(), name.strip())
        if identifier in teams or name in teams.values():
            raise ValueError("duplicate API club id or normalized name")
        teams[identifier] = name
    rows, codes, elements = [], set(), set()
    for item in bootstrap["elements"]:
        code = _integer(item.get("code"), "stable code")
        element = _integer(item.get("id"), "API element id")
        club = _integer(item.get("team"), "element club")
        role = _integer(item.get("element_type"), "element type", maximum=4)
        price = _integer(item.get("now_cost"), "current price")
        name = item.get("web_name")
        if code in codes or element in elements or club not in teams:
            raise ValueError("duplicate or unknown register element identity")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("registered web_name must be nonblank")
        codes.add(code)
        elements.add(element)
        rows.append({"player_id": code, "name": name.strip(), "team": teams[club],
                     "position": _POSITIONS[role], "value": price})
    if not rows:
        raise ValueError("origin register must contain players")
    return pd.DataFrame(rows).set_index("player_id").sort_index()


def _capture_guard(capture: dict) -> None:
    if not isinstance(capture, dict) or not {"bootstrap", "metadata", "receipt"} <= set(capture):
        raise ValueError("admitted capture bootstrap, metadata and receipt are required")
    receipt, metadata = capture["receipt"], capture["metadata"]
    if not isinstance(receipt, dict) or not isinstance(metadata, pd.DataFrame):
        raise ValueError("invalid capture metadata/receipt")
    if (receipt.get("bootstrap_canonical_sha256") != _canonical_sha(capture["bootstrap"])
            or receipt.get("metadata_sha256") != _frame_sha(metadata)):
        raise ValueError("capture bootstrap or metadata changed since admission")
    expected = _metadata(capture["bootstrap"])
    if not metadata.equals(expected):
        raise ValueError("capture metadata differs from its admitted bootstrap")


def load_origin_captures(descriptor_path: str | Path, *, season: str, origins: list[int]) -> dict[int, dict]:
    """Validate exact requested capture coverage and recent predeadline byte identities."""
    match = re.fullmatch(r"(\d{4})-(\d{2})", season) if isinstance(season, str) else None
    if not match or int(match[2]) != (int(match[1]) + 1) % 100:
        raise ValueError("season must be consecutive YYYY-YY")
    if not isinstance(origins, list) or not origins:
        raise ValueError("explicit requested origins are required")
    requested = [_integer(value, "origin_gw") for value in origins]
    first = config.season_start_gw(season)
    if len(set(requested)) != len(requested) or any(not first <= value < first + 38 for value in requested):
        raise ValueError("origins must be unique and belong to the declared season")
    path = Path(descriptor_path).resolve()
    content = path.read_bytes()
    descriptor = _json(content)
    if (set(descriptor) != {"schema_version", "season", "captures"}
            or type(descriptor["schema_version"]) is not int or descriptor["schema_version"] != 1
            or descriptor["season"] != season or not isinstance(descriptor["captures"], dict)
            or set(descriptor["captures"]) != {str(value) for value in requested}):
        raise ValueError("invalid descriptor schema, season or exact requested capture coverage")
    result = {}
    for origin in requested:
        entry = descriptor["captures"][str(origin)]
        if (not isinstance(entry, dict) or set(entry) != {"folder", "input_sha256"}
                or not isinstance(entry["folder"], str) or not entry["folder"]
                or not isinstance(entry["input_sha256"], dict) or set(entry["input_sha256"]) != _FILES):
            raise ValueError("capture requires a relative folder and both pinned inputs")
        relative = Path(entry["folder"])
        folder = path.parent / relative
        if relative.is_absolute() or ".." in relative.parts or not folder.resolve().is_relative_to(path.parent):
            raise ValueError("capture folder escapes descriptor directory")
        inputs = {}
        for name in _FILES:
            source = folder / name
            if not source.resolve().is_relative_to(path.parent):
                raise ValueError("capture input symlink escapes descriptor directory")
            inputs[name] = source.read_bytes()
            if _sha(inputs[name]) != entry["input_sha256"][name]:
                raise ValueError("capture input hash differs from descriptor")
        manifest = _json(inputs["manifest.json"])
        try:
            payload = gzip.decompress(inputs["bootstrap-static.json.gz"])
        except (OSError, EOFError) as exc:
            raise ValueError("invalid compressed register payload") from exc
        bootstrap = _json(payload)
        files = manifest.get("files")
        if not isinstance(files, list):
            raise ValueError("capture manifest requires endpoint files")
        bound = [item for item in files if isinstance(item, dict) and item.get("name") == "bootstrap-static.json.gz"]
        if len(bound) != 1 or bound[0].get("endpoint") != "/bootstrap-static/" or bound[0].get("sha256") != _sha(payload):
            raise ValueError("official bootstrap payload hash/endpoint differs from manifest")
        local = origin - first + 1
        if (_integer(manifest.get("event"), "manifest event", maximum=38) != local
                or not isinstance(manifest.get("backfill"), dict) or manifest["backfill"].get("season") != season):
            raise ValueError("manifest event/season differs from requested global origin")
        events = {}
        if not isinstance(bootstrap.get("events"), list):
            raise ValueError("register events are required")
        for item in bootstrap["events"]:
            identifier = _integer(item.get("id"), "register event", maximum=38)
            if identifier in events:
                raise ValueError("duplicate register event")
            events[identifier] = _timestamp(item.get("deadline_time"), "register deadline")
        if local not in events or 1 not in events or events[1].year != int(match[1]):
            raise ValueError("register first deadline/year or requested event differs from season")
        capture_time = _timestamp(manifest.get("captured_at"), "captured_at")
        deadline = _timestamp(manifest.get("event_deadline"), "event_deadline")
        if (deadline != events[local] or capture_time > deadline
                or (deadline - capture_time).total_seconds() > config.REPLAY_REGISTER_MAX_AGE_HOURS * 3600):
            raise ValueError("capture requires matching recent predeadline register timing")
        metadata = _metadata(bootstrap)
        receipt = {"origin_gw": origin, "event": local, "captured_at": capture_time.isoformat(),
                   "event_deadline": deadline.isoformat(), "input_sha256": deepcopy(entry["input_sha256"]),
                   "payload_sha256": _sha(payload), "descriptor_sha256": _sha(content),
                   "bootstrap_canonical_sha256": _canonical_sha(bootstrap), "metadata_sha256": _frame_sha(metadata)}
        result[origin] = {"bootstrap": bootstrap, "metadata": metadata, "receipt": receipt}
    return result


def prepare_origin_frame(frame: pd.DataFrame, capture: dict, *, origin_gw: int,
                         owned: list[int] | None = None) -> tuple[pd.DataFrame, dict]:
    """Filter registered forecast rows and overlay one origin's static horizon prices.

    Raw future values, identities and all outcomes are discarded. Missing players
    receive neither forecasts nor zero outcomes. Fixture causality remains unassessed.
    """
    origin = _integer(origin_gw, "origin_gw")
    _capture_guard(capture)
    if capture["receipt"].get("origin_gw") != origin:
        raise ValueError("capture does not match the requested forecast origin")
    required = {"player_id", "GW", "origin_gw", "predicted_total_points"}
    if frame.empty or not frame.columns.is_unique or not required <= set(frame):
        raise ValueError("origin frame requires nonempty unambiguous forecast fields")
    allowed = ["player_id", "GW", "GW_global", "origin_gw", "predicted_total_points", "predicted_minutes",
               "fixture", "forecast_fixture_id", "opponent_team", "was_home"]
    values = frame[[name for name in allowed if name in frame]].copy()
    for name in ("player_id", "GW", "origin_gw"):
        values[name] = _identity_column(values[name], name)
    if not values.origin_gw.eq(origin).all() or (values.GW < origin).any():
        raise ValueError("every row must match the origin and target it or later rounds")
    if "GW_global" in values:
        values["GW_global"] = _identity_column(values.GW_global, "GW_global")
        if not values.GW_global.eq(values.GW).all():
            raise ValueError("GW_global and GW must use the same global identity")
    for name in ("predicted_total_points", "predicted_minutes"):
        if name in values:
            values[name] = _numeric(values[name], name)
    metadata = capture["metadata"]
    known = values.player_id.isin(metadata.index)
    excluded = sorted(set(values.loc[~known, "player_id"]))
    prepared = values.loc[known].copy()
    available = set(prepared.player_id)
    if owned is not None:
        if not isinstance(owned, list):
            raise ValueError("owned requires an explicit stable-code list")
        owner = [_integer(player, "owned stable code") for player in owned]
        if len(owner) != len(set(owner)) or not set(owner) <= set(metadata.index) or not set(owner) <= available:
            raise ValueError("owned players must be registered and have explicit horizon forecasts")
    if prepared.empty:
        raise ValueError("no registered forecast rows remain at the origin")
    for name in ("name", "team", "position", "value"):
        prepared[name] = prepared.player_id.map(metadata[name])
    absent = sorted(set(metadata.index) - available)
    receipt = {**deepcopy(capture["receipt"]), "price_policy": "origin_capture_half_profit_static_horizon_v1",
               "pool_policy": "registered_forecast_rows", "input_forecast_rows": len(frame),
               "registered_forecast_rows": len(prepared), "excluded_forecast_rows": int((~known).sum()),
               "excluded_player_ids": excluded, "registered_without_forecasts": len(absent),
               "registered_without_forecast_ids": absent, "prepared_frame_sha256": _frame_sha(prepared),
               "fixture_causality": "NOT ASSESSED", "chip_rule_legality": "NOT ASSESSED",
               "archive_authenticity": "NOT ASSESSED"}
    return prepared, receipt


def selling_prices(capture: dict, owned: list[int], purchase_prices: Mapping[int, int]) -> dict[int, int]:
    """Calculate half-profit prices from exact explicit owned acquisition costs."""
    _capture_guard(capture)
    if not isinstance(owned, list) or not isinstance(purchase_prices, Mapping):
        raise ValueError("owned list and explicit acquisition-cost mapping are required")
    players = [_integer(player, "owned stable code") for player in owned]
    keys = [_integer(player, "purchase-price stable code") for player in purchase_prices]
    if (len(players) != len(set(players)) or set(keys) != set(players)
            or not set(players) <= set(capture["metadata"].index)):
        raise ValueError("purchase basis must cover exactly the registered owned codes")
    result = {}
    for player in players:
        purchase = _integer(purchase_prices[player], "purchase price")
        current = _integer(capture["metadata"].loc[player, "value"], "captured current price")
        result[player] = min(current, purchase + max(current - purchase, 0) // 2)
    return result
