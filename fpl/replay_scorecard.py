"""Read-only summaries of verified saved settlement inputs, without re-scoring."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from numbers import Real
from pathlib import Path
import re

import numpy as np
import pandas as pd

from fpl import config


_GATES = (
    "price_information_validity", "forecast_blank_planning", "fixture_causality",
    "full_outcome_blind_execution", "native_certificate", "independent_legality",
)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _weeks(values, label: str) -> list[int]:
    if (not isinstance(values, list) or not values
            or any(type(value) is not int or value <= 0 for value in values)
            or values != sorted(set(values))):
        raise ValueError(f"{label} requires sorted unique positive integer gameweeks")
    return values


def _number(value, label: str, *, nonnegative: bool = False) -> float:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or (nonnegative and value < 0)):
        raise ValueError(f"invalid finite numeric {label}")
    return float(value)


def _transfers(row: dict, field: str) -> int:
    values = row.get(field)
    if (not isinstance(values, list)
            or any(type(value) is not int or value <= 0 for value in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{field} requires unique positive integer IDs")
    return len(values)


def _recorded(snapshot: dict, rows: list[dict]) -> dict:
    result = {}
    for field in ("solver_status", "mip_gap", "solver_seconds"):
        present = [row[field] for row in rows if row.get(field) is not None]
        if field != "solver_status":
            for value in present:
                _number(value, field, nonnegative=True)
        result[field] = ({"status": "RECORDED", "values": present}
                         if len(present) == len(rows) else {"status": "NOT MEASURED"})
    for field in ("price_policy", "pool_policy", "solver_outcome_columns"):
        observed = {}
        if field in snapshot:
            observed["snapshot"] = snapshot[field]
        if any(field in row for row in rows):
            observed["decisions"] = [row.get(field) for row in rows]
        if observed:
            result[field] = {"status": "RECORDED", **observed}
    return result


def _side(directory: str | Path, expected_sha: str, weeks: list[int]) -> dict:
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("comparison input SHA256 must be a lowercase hexadecimal digest")
    from fpl.milp.replay import read_replay

    root = Path(directory)
    receipt_bytes = (root / "complete.json").read_bytes()
    receipt = read_replay(root)
    if json.loads(receipt_bytes) != receipt:
        raise ValueError("replay receipt changed during admission")
    captured = {name: (root / name).read_bytes()
                for name in ("decisions.json", "replayed.csv", "predictions.csv")}
    for name, content in captured.items():
        if _sha(content) != receipt["output_sha256"].get(name):
            raise ValueError(f"saved replay bytes changed: {name}")
    if (root / "complete.json").read_bytes() != receipt_bytes:
        raise ValueError("replay receipt changed during summary")
    if _sha(captured["replayed.csv"]) != expected_sha:
        raise ValueError("comparison input SHA256 does not match replayed.csv")
    if _weeks(receipt.get("gameweeks"), "receipt") != weeks:
        raise ValueError("receipt gameweek coverage differs from comparison")
    snapshot = json.loads(captured["decisions.json"])
    json.dumps(snapshot, allow_nan=False)
    rows = snapshot.get("decisions")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("invalid decision snapshot")
    if _weeks([row.get("gameweek") for row in rows], "snapshot") != weeks:
        raise ValueError("snapshot gameweek coverage differs from comparison")
    frame = pd.read_csv(BytesIO(captured["replayed.csv"]))
    required = ["gameweek", "actual_total_points", "actual_net_points", "alpha"]
    if not set(required).issubset(frame):
        raise ValueError("replayed CSV lacks settlement columns")
    csv_weeks = []
    for value in frame.gameweek:
        number = _number(value, "CSV gameweek", nonnegative=True)
        if number <= 0 or not number.is_integer():
            raise ValueError("invalid CSV gameweek")
        csv_weeks.append(int(number))
    if _weeks(csv_weeks, "CSV") != weeks:
        raise ValueError("CSV gameweek coverage differs from comparison")
    gross = np.array([_number(value, "gross points") for value in frame.actual_total_points])
    net = np.array([_number(value, "net points") for value in frame.actual_net_points])
    alpha = np.array([_number(value, "alpha", nonnegative=True) for value in frame.alpha])
    if np.any(alpha != np.floor(alpha)):
        raise ValueError("alpha must contain integer hits")
    snapshot_alpha = [_number(row.get("alpha"), "snapshot alpha", nonnegative=True) for row in rows]
    if any(not value.is_integer() for value in snapshot_alpha) or not np.array_equal(alpha, snapshot_alpha):
        raise ValueError("snapshot and settlement alpha disagree")
    hits = config.TRANSFER_HIT_POINTS * alpha
    if not np.allclose(gross - net, hits, rtol=0, atol=1e-9):
        raise ValueError("gross/net transfer-hit accounting disagrees")
    for field, total in (("gross_points", gross.sum()), ("net_points", net.sum())):
        if not np.isclose(_number(receipt.get(field), field), total, rtol=0, atol=1e-9):
            raise ValueError(f"receipt {field} disagrees with settlement CSV")
    chips = {}
    for row in rows:
        chip = row.get("chip_played")
        if chip is None:
            continue
        if not isinstance(chip, str) or not chip:
            raise ValueError("invalid recorded chip label")
        chips[chip] = chips.get(chip, 0) + 1
    return {
        "gross_points": float(gross.sum()), "net_points": float(net.sum()),
        "hit_points": float(hits.sum()), "decision_rounds": len(rows),
        "transfers_in": sum(_transfers(row, "transfers_in") for row in rows),
        "transfers_out": sum(_transfers(row, "transfers_out") for row in rows),
        "chip_usage": chips, "receipt_sha256": _sha(receipt_bytes),
        "snapshot_sha256": _sha(captured["decisions.json"]),
        "comparison_input_sha256": expected_sha,
        "prediction_sha256": _sha(captured["predictions.csv"]),
        "recorded": _recorded(snapshot, rows),
    }


def summarize_replay_pair(candidate_dir: str | Path, control_dir: str | Path, *, candidate_sha256: str,
                          control_sha256: str, gameweeks: list[int], metric: str) -> dict:
    """Verify captured settlement bundles and describe their fixed decisions.

    Recorded solver fields are observations only. Settlement does not validate
    forecast blank planning, source timing, independent legality or native proof.
    """
    if metric not in ("actual_total_points", "actual_net_points"):
        raise ValueError("unsupported settlement comparison metric")
    weeks = _weeks(gameweeks, "comparison")
    result = {
        "schema_version": 1, "status": "VERIFIED_SETTLEMENT_INPUTS",
        "gameweeks": list(weeks), "metric": metric, "promotion_eligible": False,
        "sides": {"candidate": _side(candidate_dir, candidate_sha256, weeks),
                  "control": _side(control_dir, control_sha256, weeks)},
        "checks": {gate: "NOT ASSESSED" for gate in _GATES},
        "limitations": "Saved-input settlement verification only. Recorded policies and solver fields are claims, "
                       "not independent certificates. Settlement does not validate zero-fill blank planning, "
                       "source timing, fixture causality or full outcome-blind execution; no promotion is inferred.",
    }
    json.dumps(result, allow_nan=False)
    return result
