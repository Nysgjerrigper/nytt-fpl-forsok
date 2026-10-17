"""Explicit retrospective blank evidence, separate from forecast fixture rows.

The caller must capture and pin the register, teams and season fixture ledger.
This helper does not establish capture timing, historical roster identity, financial
legality or outcome-blind execution, and never manufactures forecast/fixture IDs.
"""
from __future__ import annotations

from numbers import Real
import re

import numpy as np
import pandas as pd

from fpl import config

_AUDIT_COLUMNS = ["player_id", "GW", "team", "position", "actual_total_points", "minutes", "outcome_basis"]
_POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _integer(value, label: str, *, minimum: int = 1) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < minimum or value >= 2**63 or value % 1):
        raise ValueError(f"{label} requires finite non-boolean integer values >= {minimum}")
    return int(value)


def _name(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("team names must be nonblank strings")
    value = value.strip()
    return config.TEAM_NAME_CORRECTIONS.get(value, value)


def _deadline(value) -> pd.Timestamp:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("register event deadline must be an explicit timestamp")
    try:
        result = pd.to_datetime(value, errors="raise", utc=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid register event deadline") from exc
    if pd.isna(result):
        raise ValueError("invalid register event deadline")
    return result


def _fixture_ledger(fixtures: pd.DataFrame, teams: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    required = {"id", "event", "team_h", "team_a", "team_h_score", "team_a_score", "finished", "kickoff_time"}
    if (not fixtures.columns.is_unique or not teams.columns.is_unique
            or not required <= set(fixtures) or not {"id", "name"} <= set(teams)
            or len(fixtures) != 380 or len(teams) != 20):
        raise ValueError("blank evidence requires a complete 20-club 380-fixture season ledger")
    team_ids = teams.id.map(lambda value: _integer(value, "team id"))
    names = teams.name.map(_name)
    if team_ids.duplicated().any() or names.duplicated().any():
        raise ValueError("duplicate team identities or normalized names")
    mapping = dict(zip(team_ids, names))
    frame = fixtures.copy()
    for column in ("id", "event", "team_h", "team_a"):
        frame[column] = frame[column].map(lambda value: _integer(value, column))
    for column in ("team_h_score", "team_a_score"):
        frame[column] = frame[column].map(lambda value: _integer(value, column, minimum=0))
    if frame.id.duplicated().any() or set(frame.event) != set(range(1, 39)):
        raise ValueError("fixture ids must be unique and cover events 1 through 38")
    if not frame.finished.map(lambda value: isinstance(value, (bool, np.bool_)) and bool(value)).all():
        raise ValueError("every retrospective fixture must be explicitly finished")
    if not frame.kickoff_time.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("fixture kickoff times must be explicit parseable timestamps")
    try:
        kickoff = pd.to_datetime(frame.kickoff_time, errors="raise", utc=True, format="mixed")
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid fixture kickoff timestamp") from exc
    if kickoff.isna().any():
        raise ValueError("missing fixture kickoff timestamp")
    frame["_kickoff_utc"] = kickoff
    pairs = list(zip(frame.team_h, frame.team_a))
    expected = {(home, away) for home in mapping for away in mapping if home != away}
    if len(set(pairs)) != 380 or set(pairs) != expected:
        raise ValueError("each ordered home/away club pair must occur exactly once")
    appearances = pd.concat([frame.team_h, frame.team_a]).value_counts()
    if set(appearances.index) != set(mapping) or not appearances.eq(38).all():
        raise ValueError("every club requires exactly 38 season fixtures")
    return frame, mapping


def explicit_blank_outcomes(snapshot: dict, predictions: pd.DataFrame, *, fixtures: pd.DataFrame,
                            teams: pd.DataFrame, register: dict, season: str,
                            first_global_gw: int, event: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify missing selected-player zeros for one documented blank event only.

    Return an unchanged copy of original outcomes and separate explicit zero audit
    rows. The audit never receives fixture IDs or predicted points. Register names
    are matched to the complete season ledger; their historical applicability and
    capture timing remain caller responsibilities, not independent proof here.
    """
    match = re.fullmatch(r"(\d{4})-(\d{2})", season) if isinstance(season, str) else None
    if not match or int(match[2]) != (int(match[1]) + 1) % 100:
        raise ValueError("season must be a consecutive YYYY-YY season")
    first = _integer(first_global_gw, "first_global_gw")
    local = _integer(event, "event")
    if local > 38 or first != config.season_start_gw(season):
        raise ValueError("event or global season mapping is invalid")
    target = first + local - 1
    ledger, clubs = _fixture_ledger(fixtures, teams)
    if not isinstance(register, dict) or not {"teams", "elements", "events"} <= set(register):
        raise ValueError("explicit official register teams, elements and events are required")
    register_teams = {}
    for team in register["teams"]:
        identifier = _integer(team.get("id"), "register team id")
        name = _name(team.get("name"))
        if identifier in register_teams or name in register_teams.values():
            raise ValueError("duplicate register club identity")
        register_teams[identifier] = name
    if set(register_teams.values()) != set(clubs.values()):
        raise ValueError("register team names differ from the season ledger")
    events = {}
    for item in register["events"]:
        identifier = _integer(item.get("id"), "register event")
        if identifier > 38 or identifier in events:
            raise ValueError("duplicate or invalid register event")
        events[identifier] = item
    if 1 not in events or local not in events:
        raise ValueError("register must identify first and requested event deadlines")
    first_deadline = _deadline(events[1].get("deadline_time"))
    target_deadline = _deadline(events[local].get("deadline_time"))
    if first_deadline.year != int(match[1]) or target_deadline < first_deadline:
        raise ValueError("register deadline year/order differs from the declared season")
    season_year = int(match[1])
    kickoff = ledger["_kickoff_utc"]
    target_kickoffs = kickoff.loc[ledger.event == local]
    if (not kickoff.dt.year.isin([season_year, season_year + 1]).all()
            or kickoff.min().year != season_year or kickoff.min() < first_deadline
            or (target_kickoffs < target_deadline).any()):
        raise ValueError("fixture kickoff calendar/deadlines differ from the declared season/event")
    identities = {}
    for element in register["elements"]:
        code = _integer(element.get("code"), "register stable code")
        team = _integer(element.get("team"), "registered player club")
        role = _integer(element.get("element_type"), "registered player position")
        if code in identities or team not in register_teams or role not in _POSITIONS:
            raise ValueError("duplicate stable codes or invalid register player identity")
        identities[code] = (register_teams[team], _POSITIONS[role])
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("decisions"), list):
        raise ValueError("snapshot decisions must be explicit")
    selected = None
    for decision in snapshot["decisions"]:
        if _integer(decision.get("gameweek"), "decision gameweek") != target:
            continue
        if selected is not None or not isinstance(decision.get("squad"), list) or not decision["squad"]:
            raise ValueError("requested event requires one nonempty selected squad")
        selected = [_integer(value, "selected stable code") for value in decision["squad"]]
        if len(set(selected)) != len(selected):
            raise ValueError("selected squad contains duplicated stable codes")
    if selected is None:
        raise ValueError("snapshot has no decision for the requested event")
    if not predictions.columns.is_unique or not {"player_id", "GW", "team", "position"} <= set(predictions):
        raise ValueError("original outcomes require explicit player/round/team/position identities")
    weeks = predictions.GW.map(lambda value: _integer(value, "outcome gameweek"))
    codes = predictions.player_id.map(lambda value: _integer(value, "outcome stable code"))
    current = predictions.loc[weeks == target].copy()
    current["player_id"] = codes.loc[weeks == target]
    playing = set(ledger.loc[ledger.event == local, "team_h"]) | set(ledger.loc[ledger.event == local, "team_a"])
    playing_names = {clubs[identifier] for identifier in playing}
    rows = []
    for code in selected:
        if code not in identities:
            raise ValueError(f"selected stable code {code} is missing from the explicit register")
        team, position = identities[code]
        present = current.loc[current.player_id == code]
        if not present.empty:
            if not present.team.map(_name).eq(team).all() or not present.position.eq(position).all():
                raise ValueError(f"selected target identity contradicts the register for player {code}")
            if team not in playing_names:
                for column in ("actual_total_points", "minutes"):
                    if column in present and not present[column].eq(0).all():
                        raise ValueError(f"selected outcome contradicts blank club ledger for player {code}")
            continue
        if team in playing_names:
            raise ValueError(f"missing outcomes in GW{target} for player {code} at a club with a fixture")
        rows.append({"player_id": code, "GW": target, "team": team, "position": position,
                     "actual_total_points": 0.0, "minutes": 0.0,
                     "outcome_basis": "EXPLICIT_RETROSPECTIVE_BLANK"})
    return predictions.copy(), pd.DataFrame(rows, columns=_AUDIT_COLUMNS)
