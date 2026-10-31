"""Offline admission of hash-bound pre-deadline bootstrap roster captures.

This verifies the preserved archive chain and capture timing. It does not establish
complete publication-time coverage, exact selling prices, or a modeling benefit.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Iterable
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from fpl import config
from fpl.data.provenance import load_frozen_dataset_manifest

MAX_BODY_BYTES = config.DEADLINE_SNAPSHOT_MAX_BODY_BYTES
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
COLUMNS = ["player_id", "fpl_element_id", "origin_gw", "season", "event", "team",
           "position", "name", "value", "status", "chance_of_playing_next_round",
           "captured_at", "deadline", "snapshot_age_hours", "source_body_sha256"]


class SnapshotUnavailableError(ValueError):
    """A requested origin has no usable capture, without invalidating evidence.

    Only absent captures, excessive age and an earlier next-event alignment use
    this exception. Hash, clock, identity and structural failures remain fatal.
    """

    def __init__(self, origin_gw: int, reason: str) -> None:
        self.origin_gw = origin_gw
        self.reason = reason
        super().__init__(f"Snapshot unavailable for origin {origin_gw}: {reason}")


def _bytes(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return data


def _path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Capture paths must be relative and confined under proof directory")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Capture path escapes proof directory")
    return path


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{label} must be a positive integer, excluding bool")
    return int(value)


def _origin(capture: dict) -> int:
    season = capture["season"]
    if not isinstance(season, str) or not re.fullmatch(r"\d{4}-\d{2}", season):
        raise ValueError("Invalid capture season")
    year, end = season.split("-")
    if (int(year) + 1) % 100 != int(end):
        raise ValueError("Capture season years must be sequential")
    event = _integer(capture["event"], "event")
    if event > config.GWS_PER_SEASON:
        raise ValueError("Capture event exceeds season rounds")
    return config.season_start_gw(season) + event - 1


def _time(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise ValueError("Invalid capture/deadline timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Capture/deadline timestamp requires explicit timezone")
    return result.astimezone(timezone.utc)


def _official(url: str) -> bool:
    parsed = urlsplit(url)
    return (parsed.scheme in ("http", "https") and parsed.netloc == "fantasy.premierleague.com"
            and parsed.path == "/api/bootstrap-static/" and not parsed.query and not parsed.fragment)


def _unique(records: list[dict], key: str, label: str) -> dict[int, dict]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"Missing {label}")
    result = {}
    for record in records:
        identifier = _integer(record[key], label)
        if identifier in result:
            raise ValueError(f"Duplicate {label}")
        result[identifier] = record
    return result


def _capture_rows(capture: dict, origin: int, root: Path, index: list[dict], max_age: float) -> tuple[list[dict], int]:
    timestamp = capture["wayback_timestamp"]
    if not isinstance(timestamp, str) or not re.fullmatch(r"\d{14}", timestamp):
        raise ValueError("Invalid Wayback timestamp")
    captured = _time(capture["captured_at"])
    if captured != datetime.strptime(timestamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc):
        raise ValueError("Capture timestamp differs from Wayback timestamp")
    deadline = _time(capture["deadline"])
    age = (deadline - captured).total_seconds() / 3600
    if age <= 0:
        raise ValueError("Capture must be strictly before deadline")
    matching = [row for row in index if row["timestamp"] == timestamp
                and row["digest"] == capture["cdx_digest"] and str(row["statuscode"]) == "200"
                and row["mimetype"] == "application/json" and _official(row["original"])]
    if not matching:
        raise ValueError("Capture has no matching official successful JSON CDX record")
    source = capture["source_url"]
    prefix = f"https://web.archive.org/web/{timestamp}id_/"
    if not isinstance(source, str) or not source.startswith(prefix) or not _official(source[len(prefix):]):
        raise ValueError("Capture source is not the exact raw Wayback official endpoint")
    if capture.get("http_status") != 200 or isinstance(capture.get("http_status"), bool):
        raise ValueError("Capture HTTP status must be 200")
    if capture.get("final_url", source) != source:
        raise ValueError("Capture redirected away from its exact source URL")
    headers = {key.lower(): value for key, value in capture.get("headers", {}).items()}
    if headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        raise ValueError("Capture content type must be application/json")
    original = _bytes(_path(root, capture["original_payload"]), capture["original_payload_sha256"])
    if base64.b32encode(hashlib.sha1(original).digest()).decode() != capture["cdx_digest"]:
        raise ValueError("Original response SHA1 differs from CDX digest")
    if original.startswith(b"\x1f\x8b"):
        with gzip.GzipFile(fileobj=io.BytesIO(original)) as stream:
            body = stream.read(MAX_BODY_BYTES + 1)
    else:
        body = original
    if len(body) > MAX_BODY_BYTES:
        raise ValueError("Bootstrap body exceeds maximum bytes")
    body_digest = hashlib.sha256(body).hexdigest()
    if body_digest != capture["body_sha256"]:
        raise ValueError("Bootstrap body SHA256 mismatch")
    manifest = json.loads(_bytes(_path(root, capture["manifest"]), capture["manifest_sha256"]))
    backfill = manifest["backfill"]
    if (manifest["event"] != capture["event"] or isinstance(manifest["event"], bool)
            or backfill["season"] != capture["season"] or backfill["source"] != source
            or backfill["wayback_timestamp"] != timestamp or backfill["digest"] != capture["cdx_digest"]
            or _time(manifest["captured_at"]) != captured or _time(manifest["event_deadline"]) != deadline):
        raise ValueError("Source manifest identity/timing differs from capture")
    files = [file for file in manifest["files"] if file.get("endpoint") == "/bootstrap-static/"]
    if len(files) != 1 or files[0]["sha256"] != body_digest or files[0]["bytes"] != len(body):
        raise ValueError("Source manifest bootstrap SHA/byte count mismatch")
    payload = json.loads(body)
    events = _unique(payload["events"], "id", "event ID")
    if any(event > config.GWS_PER_SEASON for event in events):
        raise ValueError("Payload event ID exceeds season rounds")
    if any(not isinstance(event.get("is_next"), bool) for event in events.values()):
        raise ValueError("Payload is_next flags must be boolean")
    target = events.get(capture["event"])
    next_events = [event for event, record in events.items() if record.get("is_next") is True]
    if (target is None or target.get("finished") is not False or len(next_events) != 1
            or _time(target["deadline_time"]) != deadline):
        raise ValueError("Payload target deadline/finished/is_next is inadmissible")
    # Do not turn tampered bytes or invalid clocks into ordinary missing signals:
    # structural and roster validation must complete before availability checks.
    earlier_next = next_events[0] < capture["event"]
    if next_events[0] != capture["event"] and not earlier_next:
        raise ValueError("Payload next event lies after the requested target")
    if 1 not in events or _time(events[1]["deadline_time"]).year != int(capture["season"][:4]):
        raise ValueError("Payload GW1 deadline year differs from source season")
    teams = _unique(payload["teams"], "id", "team ID")
    types = _unique(payload["element_types"], "id", "element type ID")
    if not set(POSITIONS).issubset(types) or set(types).difference(POSITIONS) not in (set(), {5}):
        raise ValueError("Payload element types require GK/DEF/MID/FWD and only optional manager type 5")
    if 5 in types:
        label = types[5].get("singular_name")
        if not isinstance(label, str) or label.casefold() not in ("manager", "assistant manager"):
            raise ValueError("Optional element type 5 must be labelled Manager or Assistant Manager")
    elements = _unique(payload["elements"], "id", "element ID")
    _unique(payload["elements"], "code", "stable player code")
    rows = []
    excluded_nonplayer_count = 0
    for element in elements.values():
        team = _integer(element["team"], "player team")
        position = _integer(element["element_type"], "player element type")
        cost = _integer(element["now_cost"], "now_cost")
        if team not in teams or position not in types:
            raise ValueError("Player team/element type is not in payload")
        status = element["status"]
        if status not in ("a", "d", "i", "s", "u", "n"):
            raise ValueError("Invalid player status")
        chance = element["chance_of_playing_next_round"]
        if chance is not None and (isinstance(chance, bool) or not isinstance(chance, Real)
                                  or chance not in (0, 25, 50, 75, 100)):
            raise ValueError("Invalid chance_of_playing_next_round")
        name, team_name = element["web_name"], teams[team]["name"]
        if not isinstance(name, str) or not name or not isinstance(team_name, str) or not team_name:
            raise ValueError("Player/team name must be nonempty text")
        if position == 5:
            excluded_nonplayer_count += 1
            continue
        rows.append(dict(player_id=element["code"], fpl_element_id=element["id"], origin_gw=origin,
                         season=capture["season"], event=capture["event"],
                         team=config.TEAM_NAME_CORRECTIONS.get(team_name, team_name), position=POSITIONS[position],
                         name=name, value=cost, status=status, chance_of_playing_next_round=np.nan if chance is None else chance,
                         captured_at=capture["captured_at"], deadline=capture["deadline"], snapshot_age_hours=age,
                         source_body_sha256=body_digest))
    if age > max_age:
        raise SnapshotUnavailableError(origin, f"capture age {age:g}h exceeds maximum {max_age:g}h")
    if earlier_next:
        raise SnapshotUnavailableError(origin, f"next event {next_events[0]} precedes requested event {capture['event']}")
    return rows, excluded_nonplayer_count


def _load_deadline_rosters(proof_path: Path, index_path: Path, expected_proof_sha256: str,
                          origins: Iterable[int], *, max_age_hours: float = config.DEADLINE_SNAPSHOT_MAX_AGE_HOURS) -> pd.DataFrame:
    """Admit exactly requested frozen-window origins from a preserved archive proof.

    Hash verification precedes parsing. Each roster's original response is bound to
    the CDX SHA1 and its decoded body to the source manifest SHA256. No live fetch,
    stale substitution, target statistics, or invented observation round is used.
    Optional validated manager elements are excluded and counted in frame attrs.
    """
    proof_path, index_path = Path(proof_path), Path(index_path)
    proof = json.loads(_bytes(proof_path, expected_proof_sha256))
    if proof.get("status") not in ("verified", "incomplete"):
        raise ValueError("Archive proof must be verified or incomplete with valid requested captures")
    requested = [_integer(origin, "origin") for origin in origins]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Requested origins must be nonempty and unique")
    frozen_max = load_frozen_dataset_manifest()["max_gw_global"]
    if max(requested) > frozen_max:
        raise ValueError("Requested origin exceeds frozen research cutoff")
    if isinstance(max_age_hours, bool) or not isinstance(max_age_hours, Real) or not math.isfinite(max_age_hours) or max_age_hours <= 0:
        raise ValueError("max_age_hours must be positive and finite")
    index_digest = proof["index"]["sha256"]
    table = json.loads(_bytes(index_path, index_digest))
    required = {"timestamp", "original", "statuscode", "mimetype", "digest"}
    if not isinstance(table, list) or not table or not required.issubset(table[0]) or len(set(table[0])) != len(table[0]):
        raise ValueError("Invalid CDX index header")
    if any(not isinstance(row, list) or len(row) != len(table[0]) for row in table[1:]):
        raise ValueError("Invalid CDX index rows")
    index = [dict(zip(table[0], row)) for row in table[1:]]
    captures: dict[int, list[dict]] = {origin: [] for origin in requested}
    for capture in proof["captures"]:
        origin = _origin(capture)
        if origin in captures:
            captures[origin].append(capture)
    rows = []
    excluded_by_origin = {}
    for origin in requested:
        if not captures[origin]:
            raise SnapshotUnavailableError(origin, "no verified capture")
        if len(captures[origin]) != 1:
            raise ValueError(f"Requested origin {origin} requires exactly one matching capture")
        capture_rows, excluded = _capture_rows(captures[origin][0], origin, proof_path.parent, index, max_age_hours)
        rows.extend(capture_rows)
        excluded_by_origin[origin] = excluded
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame.attrs.update(proof_sha256=expected_proof_sha256, index_sha256=index_digest,
                       excluded_nonplayer_count=sum(excluded_by_origin.values()),
                       excluded_nonplayer_count_by_origin=excluded_by_origin)
    return frame


def load_deadline_rosters(proof_path: Path, index_path: Path, expected_proof_sha256: str,
                          origins: Iterable[int], *, max_age_hours: float = config.DEADLINE_SNAPSHOT_MAX_AGE_HOURS) -> pd.DataFrame:
    """Load hash-bound offline rosters, refusing any unavailable requested origin.

    Only absent captures, excessive age and stale earlier-event alignment raise
    SnapshotUnavailableError. Malformed evidence and hash/identity/timing failures
    raise ValueError and must not be treated as missing external information.
    """
    try:
        return _load_deadline_rosters(proof_path, index_path, expected_proof_sha256,
                                      origins, max_age_hours=max_age_hours)
    except (KeyError, TypeError, OSError, EOFError) as exc:
        raise ValueError(f"Invalid archive evidence: {type(exc).__name__}: {exc}") from exc
