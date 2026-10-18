"""Opt-in immutable wrapper checkpoints; no native proof or outcome-blind claim."""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd

from fpl import config
from fpl.artifacts import sha256_file, write_json_new


def finite_value(value: Any) -> Any:
    """Convert wrapper state into finite JSON without coercing missing numbers."""
    if isinstance(value, np.generic):
        return finite_value(value.item())
    if isinstance(value, dict):
        return {str(key): finite_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def frame_identity(frame: pd.DataFrame) -> dict:
    """Bind exact ordered frame/index CSV representation and its pandas runtime."""
    content = frame.to_csv(index=True, lineterminator="\n").encode("utf-8")
    return {"sha256": hashlib.sha256(content).hexdigest(), "rows": len(frame),
            "columns": list(frame.columns), "encoding": "pandas_csv_with_index_v1"}


def input_identity(value: Any) -> Any:
    """Bind prepared tables and capture objects without writing their outcome rows."""
    if isinstance(value, pd.DataFrame):
        return frame_identity(value)
    if isinstance(value, dict):
        return {str(key): input_identity(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [input_identity(item) for item in value]
    return finite_value(value)


def source_runtime() -> dict:
    """Capture relevant wrapper/vendor bytes and execution-library versions."""
    files = {config.ROOT / "fpl/config.py", config.ROOT / "fpl/artifacts.py",
             config.ROOT / "fpl/milp/optimize.py", config.ROOT / "fpl/milp/origin_prices.py"}
    files.update((config.ROOT / "fpl/milp").glob("solio*.py"))
    files.update((config.ROOT / "fpl/milp/_vendor/solio").rglob("*.py"))
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "pulp", "highspy", "sasoptpy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"source_sha256": {str(path.relative_to(config.ROOT)): sha256_file(path)
                              for path in sorted(files)}, "runtime": versions}


class ScheduleReceipts:
    """Checkpoint wrapper calls, retaining completed decisions when a later call fails."""

    def __init__(self, directory: str | Path, predictions: pd.DataFrame, settings: dict) -> None:
        self.root = Path(directory).resolve()
        if any(self.root.is_relative_to(path.resolve()) for path in (config.DATASETT_DIR, config.ROOT / "fpl")):
            raise ValueError("checkpoint directory cannot target production code or datasets")
        self.origin = None
        self.sequence = 0
        self.attempted = 0
        self.completed = 0
        self.decisions = 0
        self.stage = "schedule-validation"
        self.requested_origins = list(range(settings["start_gw"], settings["max_gw"]+1))
        settings = dict(settings)
        for name in ("bootstrap", "origin_price_captures"):
            if settings.get(name) is not None:
                content = json.dumps(input_identity(settings[name]), sort_keys=True, allow_nan=False).encode()
                settings[name] = {"sha256": hashlib.sha256(content).hexdigest(), "basis": "canonical_input_identity_json"}
        started = {"schema_version": 1, "status": "STARTED", "input_frame": frame_identity(predictions),
                   "settings": input_identity(settings), **source_runtime(),
                   "outcome_access_assessment": "NOT ASSESSED",
                   "call_count_scope": "VENDOR_WRAPPER_INVOCATIONS"}
        # Validate before creating the directory, then refuse every existing path.
        json.dumps(finite_value(started), allow_nan=False)
        self.root.mkdir(parents=True, exist_ok=False)
        write_json_new(self.root / "started.json", finite_value(started))
        self.run_identity = sha256_file(self.root / "started.json")
        self.sources = {key: started[key] for key in ("source_sha256", "runtime")}

    def emit(self, event: str, values: dict) -> None:
        """Create one immutable event before committing in-memory progress counters."""
        origin = values["origin_gw"] if event == "origin-start" else self.origin
        attempted = self.attempted + (event == "vendor-start")
        completed = self.completed + (event == "vendor-complete")
        decisions = self.decisions + (event == "decision")
        sequence = self.sequence + 1
        record = {"schema_version": 1, "event": event, "origin_gw": origin,
                  "sequence": sequence, "started_sha256": self.run_identity,
                  "attempted_vendor_calls": attempted,
                  "completed_vendor_calls": completed, **values}
        write_json_new(self.root / f"{sequence:04d}-{event}.json", finite_value(record))
        self.origin, self.stage, self.sequence = origin, event, sequence
        self.attempted, self.completed, self.decisions = attempted, completed, decisions

    def terminal(self, error: Exception | None = None) -> None:
        """Seal all recorded files, preserving failures without a completion claim."""
        if error is None and self.sources != source_runtime():
            raise ValueError("checkpoint source or runtime changed during schedule")
        record = {"schema_version": 1, "status": "FAILED" if error else "COMPLETED",
                  "attempted_vendor_calls": self.attempted, "completed_vendor_calls": self.completed,
                  "completed_origins": self.decisions, "requested_origins": self.requested_origins,
                  "started_sha256": self.run_identity,
                  "call_count_scope": "VENDOR_WRAPPER_INVOCATIONS",
                  "outcome_access_assessment": "NOT ASSESSED",
                  "receipt_sha256": {str(path.relative_to(self.root)): sha256_file(path)
                                     for path in sorted(self.root.rglob("*")) if path.is_file()}}
        if error is not None:
            record.update(failed_origin=self.origin, failed_stage=self.stage, exception_type=type(error).__name__, exception=str(error),
                          failed_native_seconds="NOT MEASURED", failed_build_seconds="NOT MEASURED")
        write_json_new(self.root / ("failed.json" if error else "complete.json"), finite_value(record))
