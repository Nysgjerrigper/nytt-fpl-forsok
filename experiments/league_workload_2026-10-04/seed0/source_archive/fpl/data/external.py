"""Research-only byte, identity and deadline admission for external fixture records.

Checks establish consistency of claimed publication evidence and input bytes, not
factual proof of source availability. Real publication timing needs an independent
source audit. This module does not fetch data or infer global gameweek boundaries.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from fpl import config

_EXTERNAL_KEY = ["season", "external_player_id", "event_id"]
_VERSION_KEY = _EXTERNAL_KEY + ["available_at"]
_POSITIONS = {"GK", "DEF", "MID", "FWD"}


def _timestamps(values: pd.Series, label: str) -> pd.Series:
    """Require explicit ISO timezone evidence and normalize a timestamp column to UTC."""
    text = values.astype("string")
    explicit = text.str.match(r"^\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}.*(?:[zZ]|[+-]\d{2}:?\d{2})$", na=False)
    if not explicit.all():
        raise ValueError(f"{label} requires timezone-aware ISO timestamps")
    parsed = pd.to_datetime(text, utc=True, errors="coerce", format="mixed")
    if parsed.isna().any():
        raise ValueError(f"{label} contains invalid timestamps")
    return parsed


def _timestamp(value: str, label: str) -> pd.Timestamp:
    return _timestamps(pd.Series([value]), label).iloc[0]


def _required(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    if missing := set(columns).difference(frame.columns):
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _opaque_ids(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    """Require actual strings, preserving identity spelling and leading zeroes."""
    for column in columns:
        series = frame[column]
        # Vectorized pandas string operations reject numeric identity values as missing.
        nonempty = (series.str.len().notna() & series.str.strip().ne("")) if pd.api.types.is_string_dtype(series.dtype) else pd.Series(False, index=series.index)
        if not nonempty.all():
            raise ValueError(f"{label} {column} must contain nonempty string identifiers")


def _positive_ids(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    for column in columns:
        # Canonical numeric text validation excludes scalar booleans even when
        # pandas stores a mixed bool/integer column as object dtype.
        canonical = frame[column].astype("string").str.fullmatch(r"[+]?\d+(?:\.0+)?", na=False)
        if not canonical.all():
            raise ValueError(f"{label} {column} requires positive integer identifiers")
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(numeric).all() or (numeric <= 0).any() or (numeric % 1 != 0).any():
            raise ValueError(f"{label} {column} requires positive integer identifiers")
        frame[column] = numeric.astype("int64")


def load_external_records(
    csv_path: Path, manifest_path: Path, *, max_available_at: str,
) -> pd.DataFrame:
    """Admit verified external bytes and claimed publication timestamps for research.

    The caller supplies its registered capped historical boundary, not a live date.
    Availability must reflect archived capture/provider publication with independent
    audit evidence; a match timestamp or Git commit timestamp is insufficient.
    Missing statistics stay missing, and opaque source IDs retain leading zeroes.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != config.EXTERNAL_SOURCE_SCHEMA:
        raise ValueError("Unsupported external source manifest schema")
    for field in ("source", "revision", "sha256", "snapshot_available_at", "timing_basis", "publication_evidence"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"External manifest requires nonempty {field}")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", manifest["revision"]):
        raise ValueError("External manifest revision must be 40 hexadecimal characters")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", manifest["sha256"]):
        raise ValueError("External manifest sha256 must be 64 hexadecimal characters")
    if manifest["timing_basis"] not in {"archived_capture", "provider_publication"}:
        raise ValueError("External source requires archived_capture/provider_publication timing evidence")
    evidence = urlparse(manifest["publication_evidence"])
    if evidence.scheme != "https" or not evidence.netloc:
        raise ValueError("External publication_evidence must be an HTTPS URL")
    snapshot = _timestamp(manifest["snapshot_available_at"], "snapshot_available_at")
    boundary = _timestamp(max_available_at, "max_available_at")
    if snapshot >= boundary:
        raise ValueError("External snapshot must precede the registered historical boundary")
    data = Path(csv_path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest["sha256"].lower():
        raise ValueError("External CSV SHA256 mismatch")
    # Parse the very bytes just hashed, preventing a read/hash time-of-check mismatch.
    from io import BytesIO
    frame = pd.read_csv(BytesIO(data), dtype={key: "string" for key in _EXTERNAL_KEY},
                        keep_default_na=False, na_values=[""])
    _required(frame, _VERSION_KEY + ["event_time"], "External records")
    _opaque_ids(frame, _EXTERNAL_KEY, "External records")
    frame["event_time"] = _timestamps(frame["event_time"], "event_time")
    frame["available_at"] = _timestamps(frame["available_at"], "available_at")
    if (frame.event_time > frame.available_at).any() or (frame.available_at > snapshot).any():
        raise ValueError("External timing requires event_time <= available_at <= snapshot_available_at")
    if frame.duplicated(_VERSION_KEY).any():
        raise ValueError("Ambiguous duplicate external identity/version")
    frame.attrs.update(input_sha256=digest, source=manifest["source"], revision=manifest["revision"],
                       timing_basis=manifest["timing_basis"])
    return frame


def map_external_players(
    records: pd.DataFrame, mapping: pd.DataFrame, roster: pd.DataFrame, *,
    mapping_available_at: str, roster_available_at: str, allow_unmapped: bool = False,
) -> pd.DataFrame:
    """Join exact season-scoped source IDs to stable FPL codes without fuzzy names.

    Mapping/register snapshots must be one-to-one within each season. Their own
    publication times advance record availability rather than backdating statistics.
    Preserve source_available_at separately so later identity snapshots cannot
    collapse the identity of distinct provider revisions. Optional unmatched rows
    retain unknown IDs with explicit mapping_status.
    """
    _required(records, ["season", "external_player_id", "available_at"], "External records")
    _required(mapping, ["season", "external_player_id", "fpl_element_id"], "External mapping")
    _required(roster, ["season", "fpl_element_id", "player_id", "position"], "FPL roster")
    if {"player_id", "position", "fpl_element_id", "mapping_status", "source_available_at"}.intersection(records.columns):
        raise ValueError("External records already contain mapped identity columns")
    source = records.copy()
    links = mapping[["season", "external_player_id", "fpl_element_id"]].copy()
    players = roster[["season", "fpl_element_id", "player_id", "position"]].copy()
    _opaque_ids(source, ["season", "external_player_id"], "External records")
    _opaque_ids(links, ["season", "external_player_id"], "External mapping")
    _opaque_ids(players, ["season"], "FPL roster")
    _positive_ids(links, ["fpl_element_id"], "External mapping")
    _positive_ids(players, ["fpl_element_id", "player_id"], "FPL roster")
    if not players.position.isin(_POSITIONS).all():
        raise ValueError("FPL roster has unsupported positions")
    for table, keys in ((links, ["season", "external_player_id"]), (links, ["season", "fpl_element_id"]),
                        (players, ["season", "fpl_element_id"]), (players, ["season", "player_id"])):
        if table.duplicated(keys).any():
            raise ValueError("External identity mapping must be one-to-one per season")
    source["available_at"] = _timestamps(source["available_at"], "available_at")
    source["source_available_at"] = source["available_at"]
    mapping_time = _timestamp(mapping_available_at, "mapping_available_at")
    roster_time = _timestamp(roster_available_at, "roster_available_at")
    # Left merges preserve input order; many historical events per identity are valid.
    result = source.merge(links, on=["season", "external_player_id"], how="left", sort=False, validate="many_to_one")
    result = result.merge(players, on=["season", "fpl_element_id"], how="left", sort=False, validate="many_to_one")
    matched = result.player_id.notna()
    if not allow_unmapped and not matched.all():
        raise ValueError("External records have unmatched exact player IDs")
    result["mapping_status"] = np.where(matched, "mapped", "unmapped")
    for known_at in (mapping_time, roster_time):
        result["available_at"] = result["available_at"].where(result["available_at"] >= known_at, known_at)
    result.index = records.index
    result.attrs = records.attrs.copy()
    return result


def latest_available_records(
    records: pd.DataFrame, deadline: str, *, excluded_event_ids: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Take each event's latest strictly pre-deadline version, excluding current fixtures.

    Both event and composed publication times must precede the deadline. Revisions
    use original source publication time when available, preserving their order
    when a later identity snapshot gives multiple revisions equal composed times.
    Without mapping metadata, available_at remains the revision timestamp.
    """
    _required(records, _VERSION_KEY + ["event_time"], "External records")
    _opaque_ids(records, _EXTERNAL_KEY, "External records")
    result = records.copy()
    result["event_time"] = _timestamps(result["event_time"], "event_time")
    result["available_at"] = _timestamps(result["available_at"], "available_at")
    boundary = _timestamp(deadline, "deadline")
    version_time = "available_at"
    if "source_available_at" in result:
        result["source_available_at"] = _timestamps(result["source_available_at"], "source_available_at")
        if (result.source_available_at > result.available_at).any():
            raise ValueError("source_available_at must not exceed composed available_at")
        version_time = "source_available_at"
    if result.duplicated(_EXTERNAL_KEY + [version_time]).any():
        raise ValueError("Ambiguous duplicate external identity/version")
    result = result.loc[(result.event_time < boundary) & (result.available_at < boundary)
                        & ~result.event_id.isin(excluded_event_ids)]
    result = result.sort_values(version_time, kind="stable").drop_duplicates(_EXTERNAL_KEY, keep="last")
    result.attrs = records.attrs.copy()
    return result
