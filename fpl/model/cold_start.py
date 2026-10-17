"""Opt-in pooled fixture priors, including DNPs, for registered missing players.

Empirical spreads and participation frequencies are historical descriptors, not
calibrated intervals or live probabilities. Synthetic rows are prediction inputs
only. Caller-controlled source manifests remain necessary for external provenance.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral, Real

import numpy as np
import pandas as pd

from fpl import config

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
PRIOR_COLUMNS = ["position", "team", "predicted_total_points", "prior_std", "p_played", "p60", "prior_count"]
_VALUES = PRIOR_COLUMNS[2:]


def cold_start_mask(frame: pd.DataFrame) -> pd.Series:
    """Return literal boolean markers, rejecting missing or coerced numeric flags."""
    if "is_cold_start" not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame["is_cold_start"]
    if not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("is_cold_start requires a literal boolean for every row")
    return values.astype(bool)


def _positive(value, name: str, *, strict: bool = False) -> int:
    kind = Integral if strict else Real
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, kind)
            or value <= 0 or value >= 2**63 or not np.isfinite(value) or value % 1):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _numbers(values: pd.Series, name: str, *, integer: bool = False) -> pd.Series:
    if not values.map(lambda value: isinstance(value, Real) and not isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"{name} requires non-boolean numeric values")
    numeric = pd.to_numeric(values, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} requires finite values")
    if integer and ((numeric <= 0) | (numeric >= 2**63) | (numeric % 1 != 0)).any():
        raise ValueError(f"{name} requires positive integer identities")
    try:
        return numeric.astype("int64" if integer else float)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"invalid numeric range for {name}") from exc


def _identity(frame: pd.DataFrame) -> None:
    if not {"position", "team"} <= set(frame):
        raise ValueError("position and team identity are required")
    if not frame.position.isin(config.ONFIELD_POSITIONS).all() or not frame.team.map(
            lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("known position and nonblank team strings are required")


def _prior_strength(value: Real) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("prior_strength must be finite and positive")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("prior_strength must be finite and positive") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError("prior_strength must be finite and positive")
    return result


def _digest(frame: pd.DataFrame) -> str:
    """Runtime-dependent pandas record fingerprint; not external authentication."""
    digest = hashlib.sha256(json.dumps(list(frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _table_digest(priors: pd.DataFrame) -> str:
    """Hash ordered columns and records, canonicalizing numeric values, not attrs."""
    canonical = priors.copy()
    if "team" in canonical:
        canonical["team"] = canonical.team.astype(object).where(canonical.team.notna(), None)
    for column in _VALUES:
        if column in canonical:
            canonical[column] = pd.to_numeric(canonical[column], errors="raise").astype(float)
    return _digest(canonical)


def fit_pooled_priors(history: pd.DataFrame, origin_gw: int,
                      prior_strength: float = config.COLD_START_PRIOR_STRENGTH) -> pd.DataFrame:
    """Mix past team-position and position fixture moments before a global origin.

    Include DNP fixtures. Team weight n/(n+strength) mixes means, participation and
    second moments; standard deviation is the resulting population spread. Validate
    global round identities first, then exclude future rows before other labels.
    The history digest covers admitted relevant fields with canonical numeric dtypes;
    pandas fingerprints depend on runtime and do not authenticate external history.
    """
    origin = _positive(origin_gw, "origin_gw")
    prior_strength = _prior_strength(prior_strength)
    required = ["player_id", "position", "team", "GW_global", "total_points", "minutes"]
    if not set(required) <= set(history) or not history.columns.is_unique:
        raise ValueError("prior history requires unambiguous identity and outcome columns")
    if cold_start_mask(history).any():
        raise ValueError("synthetic cold-start rows cannot enter prior history")
    rounds = _numbers(history.GW_global, "GW_global", integer=True)
    columns = required + (["fixture"] if "fixture" in history else [])
    past = history.loc[rounds < origin, columns].copy()
    for name in ("player_id", "GW_global") + (("fixture",) if "fixture" in past else ()):
        past[name] = _numbers(past[name], name, integer=True)
    for name in ("total_points", "minutes"):
        past[name] = _numbers(past[name], name)
    _identity(past)
    if (past.minutes < 0).any():
        raise ValueError("prior history minutes must be nonnegative")
    keys = ["player_id", "GW_global"] + (["fixture"] if "fixture" in past else [])
    if past.duplicated(keys).any():
        raise ValueError("duplicate fixtures or ambiguous player rounds require unique fixture identities")
    history_digest = _digest(past)
    if past.empty:
        table = pd.DataFrame(columns=PRIOR_COLUMNS)
    else:
        squared = past.total_points ** 2
        if not np.isfinite(squared).all():
            raise ValueError("prior second moments must be finite")
        observations = past.assign(second=squared, played=(past.minutes > 0).astype(float), sixty=(past.minutes >= 60).astype(float))
        aggregations = {"mean": ("total_points", "mean"), "second": ("second", "mean"),
                        "played": ("played", "mean"), "sixty": ("sixty", "mean"), "count": ("minutes", "size")}
        pooled = observations.groupby("position", as_index=False).agg(**aggregations)
        teams = observations.groupby(["position", "team"], as_index=False).agg(**aggregations)
        mixed = teams.merge(pooled, on="position", suffixes=("", "_pooled"), validate="many_to_one")
        weight = mixed["count"] / (mixed["count"] + float(prior_strength))
        for field in ("mean", "second", "played", "sixty"):
            mixed[field] = weight * mixed[field] + (1 - weight) * mixed[field + "_pooled"]
        pooled["team"] = None
        def table_rows(values):
            return pd.DataFrame({"position": values.position, "team": values.team,
                "predicted_total_points": values["mean"],
                "prior_std": np.sqrt((values.second - values["mean"] ** 2).clip(lower=0)),
                "p_played": values.played, "p60": values.sixty, "prior_count": values["count"].astype("int64")})
        table = pd.concat([table_rows(pooled), table_rows(mixed)], ignore_index=True)[PRIOR_COLUMNS]
    table.attrs = {"origin_gw": origin, "history_max_gw": int(past.GW_global.max()) if not past.empty else None,
                   "prior_strength": float(prior_strength), "history_sha256": history_digest,
                   "table_sha256": _table_digest(table)}
    validate_pooled_priors(table, origin)
    return table


def validate_pooled_priors(priors: pd.DataFrame, origin_gw: int) -> None:
    """Check bound origin, unchanged table bytes and independent prior semantics."""
    origin = _positive(origin_gw, "origin_gw")
    metadata = priors.attrs
    required = {"origin_gw", "history_max_gw", "prior_strength", "history_sha256", "table_sha256"}
    if not required <= set(metadata) or list(priors.columns) != PRIOR_COLUMNS:
        raise ValueError("prior table is missing binding metadata or exact schema")
    try:
        json.dumps(metadata, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("prior metadata must be finite JSON") from exc
    if _positive(metadata["origin_gw"], "fitted origin") != origin:
        raise ValueError("prior table was fitted for a different global origin")
    _prior_strength(metadata["prior_strength"])
    for name in ("history_sha256", "table_sha256"):
        value = metadata[name]
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("invalid prior fingerprint")
    if metadata["table_sha256"] != _table_digest(priors):
        raise ValueError("prior table changed after fitting")
    cutoff = metadata["history_max_gw"]
    if cutoff is None:
        if not priors.empty:
            raise ValueError("nonempty priors require admitted history")
        return
    if _positive(cutoff, "history cutoff") >= origin or priors.empty:
        raise ValueError("prior history must precede origin and support a nonempty table")
    if (not priors.position.isin(config.ONFIELD_POSITIONS).all()
            or not (priors.team.isna() | priors.team.map(lambda team: isinstance(team, str) and bool(team.strip()))).all()
            or priors.duplicated(["position", "team"]).any()):
        raise ValueError("invalid or duplicate prior position/team identity")
    pooled = priors.loc[priors.team.isna()]
    if set(pooled.position) != set(priors.position) or pooled.position.duplicated().any():
        raise ValueError("every represented position requires exactly one pooled row")
    numeric = {name: _numbers(priors[name], name, integer=name == "prior_count") for name in _VALUES}
    if ((numeric["prior_std"] < 0).any() or (numeric["p60"] < 0).any()
            or (numeric["p60"] > numeric["p_played"]).any() or (numeric["p_played"] > 1).any()):
        raise ValueError("invalid prior spreads or participation frequencies")


def predict_cold_starts(frame: pd.DataFrame, priors: pd.DataFrame, *, origin_gw: int) -> pd.DataFrame:
    """Resolve complete fixture-prior rows using only position/team input identity.

    Fall back to the pooled position only when the team key is absent. Preserve
    row order and index, including duplicated indices from fixture expansion.
    """
    validate_pooled_priors(priors, origin_gw)
    _identity(frame)
    pooled = priors.loc[priors.team.isna()].set_index("position")
    if not set(frame.position) <= set(pooled.index):
        raise ValueError("no pooled history for a requested position")
    teams = priors.loc[priors.team.notna()].set_index(["position", "team"])
    keys = pd.MultiIndex.from_frame(frame[["position", "team"]])
    selected = teams.reindex(keys)[_VALUES].reset_index(drop=True)
    absent = ~keys.isin(teams.index)
    fallback = pooled.reindex(frame.position)[_VALUES].reset_index(drop=True)
    selected.loc[absent, _VALUES] = fallback.loc[absent, _VALUES].to_numpy()
    selected.index = frame.index
    return selected


def add_registered_cold_starts(snapshot: pd.DataFrame, bootstrap: dict) -> pd.DataFrame:
    """Append missing API stable codes with identity/price and unknown form/labels.

    Known values are unchanged except the false marker. New global origins are
    populated only when the input snapshot has one complete valid global round.
    """
    if "player_id" not in snapshot or not snapshot.columns.is_unique:
        raise ValueError("snapshot requires unambiguous stable player codes")
    if cold_start_mask(snapshot).any():
        raise ValueError("cannot re-register an already synthetic snapshot")
    ids = _numbers(snapshot.player_id, "snapshot player_id", integer=True)
    if ids.duplicated().any():
        raise ValueError("snapshot stable player codes must be unique")
    teams, aliases = {}, set()
    for team in bootstrap["teams"]:
        identifier = _positive(team.get("id"), "API team id", strict=True)
        name = team.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("invalid API team name")
        name = config.TEAM_NAME_CORRECTIONS.get(name, name)
        if identifier in teams or name in aliases:
            raise ValueError("duplicate API team ids or normalized aliases")
        teams[identifier] = name
        aliases.add(name)
    global_origin = None
    if "GW_global" in snapshot and not snapshot.empty:
        try:
            rounds = _numbers(snapshot.GW_global, "snapshot global origin", integer=True)
            if rounds.nunique() == 1:
                global_origin = int(rounds.iloc[0])
        except ValueError:
            pass
    existing, seen, additions = set(ids.tolist()), set(), []
    for element in bootstrap["elements"]:
        code = _positive(element.get("code"), "API stable code", strict=True)
        team = _positive(element.get("team"), "API element team", strict=True)
        role = _positive(element.get("element_type"), "API element type", strict=True)
        price = _positive(element.get("now_cost"), "API price", strict=True)
        if code in seen or team not in teams or role not in POSITIONS:
            raise ValueError("duplicate stable codes or invalid API identity mappings")
        seen.add(code)
        if code in existing:
            continue
        row = dict.fromkeys(snapshot.columns, np.nan)
        row.update(player_id=code, position=POSITIONS[role], team=teams[team],
                   name=" ".join(str(element.get(key, "")) for key in ("first_name", "second_name")).strip() or element.get("web_name", str(code)),
                   value=float(price), total_points=np.nan, minutes=np.nan, is_cold_start=True)
        if "GW_global" in snapshot and global_origin is not None:
            row["GW_global"] = global_origin
        additions.append(row)
    result = snapshot.copy()
    result["is_cold_start"] = False
    return pd.concat([result, pd.DataFrame(additions)], ignore_index=True) if additions else result
