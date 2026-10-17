"""Hash-bound comparison assumptions; compatibility is not a promotion verdict.

Contracts bind the actual bytes used for a comparison and declared experimental
assumptions. They do not establish pre-registration, causal correctness or native
solver optimality; those require independent evidence under the registered protocol.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fpl import config

_CONTEXT = {
    "season", "start_season", "gameweeks", "information_protocol", "horizon",
    "scoring_protocol", "price_policy", "sell_price_policy", "initial_state",
    "rule_set", "chip_inventory", "runtime",
}
_ARTIFACTS = {
    "dataset", "outcomes", "player_pool", "predictions", "parameters",
    "features", "model_source", "optimizer_source", "scorer_source",
}
_DIFFERENCES = {
    "forecast": {"predictions", "parameters", "model_source"},
    "features": {"predictions", "parameters", "features", "model_source"},
    "optimizer": {"optimizer_source"},
    "reporting": set(),
}


def _json_bytes(path: Path) -> tuple[dict, bytes]:
    content = path.read_bytes()
    value = json.loads(content)
    # Reject non-standard NaN/Infinity, even in nested declared assumptions.
    json.dumps(value, allow_nan=False)
    if not isinstance(value, dict):
        raise ValueError("comparison contract must be an object")
    return value, content


def _load(path: str, backtest_sha: str, frame: pd.DataFrame, metric: str) -> dict[str, Any]:
    location = Path(path).resolve()
    spec, content = _json_bytes(location)
    required = {"schema_version", "purpose", "run_id", "intervention", "backtest_sha256",
                "metric", "context", "artifacts"}
    if set(spec) != required or type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("comparison contract has invalid fields/schema_version")
    if spec["purpose"] not in ("framework_validation", "registered_research"):
        raise ValueError("contract must declare a known purpose")
    if not isinstance(spec["run_id"], str) or not spec["run_id"].strip():
        raise ValueError("contract requires run_id")
    if spec["intervention"] not in _DIFFERENCES:
        raise ValueError("contract requires a supported intervention")
    if spec["backtest_sha256"] != backtest_sha or spec["metric"] != metric:
        raise ValueError("backtest hash or metric differs from comparison contract")
    context, artifacts = spec["context"], spec["artifacts"]
    if not isinstance(context, dict) or set(context) != _CONTEXT:
        raise ValueError("contract context fields are incomplete or unknown")
    weeks = context["gameweeks"]
    if (not isinstance(weeks, list) or not weeks or any(type(gw) is not int or gw <= 0 for gw in weeks)
            or weeks != sorted(set(weeks)) or weeks != sorted(frame.gameweek.astype(int).tolist())):
        raise ValueError("contract gameweeks do not match exact backtest coverage")
    if context["information_protocol"] not in ("standard", "origin"):
        raise ValueError("unknown information protocol")
    if (type(context["horizon"]) is not int or context["horizon"] <= 0
            or context["scoring_protocol"] not in config.BACKTEST_SCORING_PROTOCOLS):
        raise ValueError("invalid horizon or scoring protocol")
    if "scoring_protocol" not in frame or set(frame.scoring_protocol) != {context["scoring_protocol"]}:
        raise ValueError("backtest scoring protocol does not match contract")
    for name in ("season", "start_season", "price_policy", "sell_price_policy", "rule_set"):
        if not isinstance(context[name], str) or not context[name].strip():
            raise ValueError(f"contract requires {name}")
    if not set(weeks) <= set(range(config.season_start_gw(context["season"]),
                                   config.season_start_gw(context["season"]) + config.GWS_PER_SEASON)):
        raise ValueError("contract weeks do not belong to the declared season")
    if context["start_season"] != config.DEFAULT_START_SEASON:
        raise ValueError("contract global-week mapping differs from configuration")
    for name in ("initial_state", "chip_inventory", "runtime"):
        if not isinstance(context[name], dict) or not context[name]:
            raise ValueError(f"contract requires explicit {name}")
    if not isinstance(artifacts, dict) or set(artifacts) != _ARTIFACTS:
        raise ValueError("contract must bind every required input/source artifact")
    verified = {}
    for name, record in artifacts.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"invalid artifact record: {name}")
        if not isinstance(record["path"], str) or not record["path"]:
            raise ValueError(f"missing artifact path: {name}")
        source = (location.parent / record["path"]).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"artifact hash mismatch: {name}")
        verified[name] = digest
    return {"context": context, "artifacts": verified, "purpose": spec["purpose"],
            "intervention": spec["intervention"], "run_id": spec["run_id"],
            "contract_sha256": hashlib.sha256(content).hexdigest()}


def validate_pair(path_a: str, path_b: str, hashes: list[str],
                  frames: list[pd.DataFrame], metric: str) -> dict:
    """Verify both manifests against parsed backtest bytes and allowed interventions.

    All context must match. Only artifacts affected by the declared intervention
    may differ; outcome, dataset, pool and scoring bytes must always match.
    """
    a, b = [_load(path, digest, frame, metric)
            for path, digest, frame in zip((path_a, path_b), hashes, frames)]
    if a["purpose"] != b["purpose"] or a["intervention"] != b["intervention"]:
        raise ValueError("contract purpose/intervention differs")
    if a["context"] != b["context"]:
        raise ValueError("comparison context differs")
    allowed = _DIFFERENCES[a["intervention"]]
    for name in _ARTIFACTS - allowed:
        if a["artifacts"][name] != b["artifacts"][name]:
            raise ValueError(f"undeclared artifact difference: {name}")
    return {"a": a, "b": b, "promotion_assessment": "NOT ASSESSED",
            "limitation": "Byte identity and declared comparability only; not proof of causal or solver validity."}
