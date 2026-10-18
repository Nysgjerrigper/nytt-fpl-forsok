"""Verify captured operational observations without importing or running any solver.

Returned cards contain kind/status, original provenance, measured timing count/total
pairs, boundary counts and saved origins. Native acceptance is a recorded wrapper
claim separate from diagnostic completion. No card provides points, rank or promotion.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SCHEDULE_TIMES = ("admission_seconds", "preparation_seconds", "vendor_wall_seconds", "native_seconds",
                   "non_native_overhead_seconds", "normalization_seconds", "state_update_seconds",
                   "build_seconds", "failed_native_seconds", "failed_build_seconds")
_NATIVE_TIMES = ("model_build_seconds", "native_call_wall_seconds", "native_runtime_seconds")
_STEPS = ("origin-start", "origin-admitted", "prepared", "vendor-start", "vendor-complete",
          "normalization-start", "normalization-complete", "state-update-start", "decision")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _int(value: Any, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, "invalid operational integer")
    return value


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _path(name: Any) -> str:
    _require(isinstance(name, str) and bool(name) and "\\" not in name
             and not Path(name).is_absolute() and all(part not in ("", ".", "..") for part in name.split("/")),
             "unsafe operational manifest path")
    return name


def _json(content: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON key in operational evidence")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    value = json.loads(content, object_pairs_hook=pairs, parse_constant=constant)
    _require(isinstance(value, dict), "operational JSON must contain an object")
    json.dumps(value, allow_nan=False)
    return value


def _files(directory: Path) -> dict[str, bytes]:
    _require(not directory.is_symlink() and directory.is_dir(), "operational root must be a real directory")
    result = {}
    def visit(folder: Path) -> None:
        for item in os.scandir(folder):
            mode = item.stat(follow_symlinks=False).st_mode
            _require(not stat.S_ISLNK(mode), "symlinks are not operational evidence")
            path = Path(item.path)
            if stat.S_ISDIR(mode):
                visit(path)
            else:
                _require(stat.S_ISREG(mode), "nonregular operational evidence file")
                descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(descriptor, "rb") as stream:
                    _require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "nonregular evidence file")
                    result[path.relative_to(directory).as_posix()] = stream.read()
    visit(directory)
    return result


def _manifest(manifest: Any, files: dict[str, bytes], expected: set[str]) -> None:
    _require(isinstance(manifest, dict), "missing operational file manifest")
    for name, digest in manifest.items():
        _path(name)
        _require(isinstance(digest, str) and _DIGEST.fullmatch(digest) is not None, "invalid operational SHA256")
    _require(set(manifest) == expected, "operational manifest file coverage differs")
    for name, digest in manifest.items():
        _require(name in files and _sha(files[name]) == digest, f"operational artifact hash differs: {name}")


def _provenance(started: dict, started_sha: str, terminal_sha: str | None) -> dict:
    sources, runtime = started.get("source_sha256"), started.get("runtime")
    _require(isinstance(sources, dict) and bool(sources), "missing original source provenance")
    for name, digest in sources.items():
        _path(name)
        _require(isinstance(digest, str) and _DIGEST.fullmatch(digest) is not None, "invalid source digest")
    _require(isinstance(runtime, dict) and bool(runtime) and all(
        isinstance(name, str) and bool(name) and (value is None or isinstance(value, str) and bool(value))
        for name, value in runtime.items()), "missing original runtime provenance")
    return {"source_sha256": sources, "runtime": runtime, "started_sha256": started_sha,
            "terminal_sha256": terminal_sha, "basis": "ORIGINAL_CAPTURE_NOT_CURRENT_EXECUTION"}


class _Timings:
    def __init__(self, names: tuple[str, ...]):
        self.values = {name: [] for name in names}

    def add(self, record: dict) -> None:
        for name, values in self.values.items():
            if name not in record:
                continue
            value = record[name]
            if value is None or value == "NOT MEASURED":
                values.append(None)
            else:
                _require(_number(value) and value >= 0, f"invalid elapsed timing: {name}")
                values.append(value)

    def result(self) -> dict:
        result = {}
        for name, values in self.values.items():
            measured = [value for value in values if value is not None]
            total = sum(measured) if measured else None
            _require(total is None or math.isfinite(total), "elapsed timing total is nonfinite")
            result[name] = {"count": len(measured), "total": total,
                            "observations": len(values), "not_measured": len(values)-len(measured)}
        return result


def _schedule(files: dict[str, bytes], docs: dict[str, dict], terminal: str | None) -> dict:
    started = docs["started.json"]
    _require(started.get("status") == "STARTED" and started.get("call_count_scope") == "VENDOR_WRAPPER_INVOCATIONS",
             "invalid schedule start receipt")
    settings = started.get("settings")
    _require(isinstance(settings, dict), "missing schedule settings")
    start, end = _int(settings.get("start_gw"), minimum=1), _int(settings.get("max_gw"), minimum=1)
    _require(end >= start, "invalid requested schedule range")
    requested = list(range(start, end+1))
    identity = _sha(files["started.json"])
    names = sorted(set(files) - {"started.json", terminal})
    completed, attempted, returned, failed = [], 0, 0, 0
    origin, last, step = None, "schedule-validation", 0
    scoring = False
    timings = _Timings(_SCHEDULE_TIMES)
    for sequence, name in enumerate(names, 1):
        record = docs.get(name)
        _require(isinstance(record, dict), "unexpected schedule artifact")
        event = record.get("event")
        _require(event in (*_STEPS, "vendor-failed", "schedule-scoring-start")
                 and name == f"{sequence:04d}-{event}.json", "schedule event filename/sequence differs")
        _require(type(record.get("sequence")) is int and record["sequence"] == sequence
                 and record.get("started_sha256") == identity, "schedule event run identity/sequence differs")
        if event == "schedule-scoring-start":
            _require(not scoring and step == 0 and completed == requested, "scoring before all saved origins")
            _require(type(record.get("completed_origins")) is int and record["completed_origins"] == len(completed),
                     "scoring origin count differs")
            scoring = True
        else:
            _require(not scoring, "schedule event after scoring boundary")
            _require(step >= 0, "schedule event after failed vendor call")
            expected = _STEPS[step]
            _require(event == expected or expected == "vendor-complete" and event == "vendor-failed",
                     "invalid schedule event transition")
            if event == "origin-start":
                _require(len(completed) < len(requested), "extra schedule origin")
                origin = requested[len(completed)]
                _require(isinstance(record.get("state"), dict), "origin start lacks state")
            elif event == "vendor-start":
                attempted += 1
            elif event == "vendor-complete":
                returned += 1
            elif event == "vendor-failed":
                failed += 1
            elif event == "decision":
                decision = record.get("decision")
                _require(isinstance(decision, dict) and type(decision.get("gameweek")) is int
                         and decision["gameweek"] == origin and record.get("settlement_completed") is False
                         and isinstance(record.get("next_state"), dict)
                         and not any(key.startswith("actual_") or key == "gross_actual_total_points" for key in decision),
                         "saved decision is not a matching pre-settlement origin")
                completed.append(origin)
            step = 0 if event == "decision" else step+1
            if event == "vendor-failed":
                step = -1  # No continuation after a failed wrapper invocation.
        _require(type(record.get("origin_gw")) is int and record["origin_gw"] == origin,
                 "schedule event origin differs")
        _require(type(record.get("attempted_vendor_calls")) is int and record["attempted_vendor_calls"] == attempted
                 and type(record.get("completed_vendor_calls")) is int and record["completed_vendor_calls"] == returned,
                 "schedule event call counters differ")
        timings.add(record)
        last = event
    status = "INCOMPLETE"
    failed_origin = failed_stage = recorded_error = None
    if terminal:
        receipt = docs[terminal]
        status = "FAILED" if terminal == "failed.json" else "COMPLETED"
        _require(receipt.get("status") == status and receipt.get("started_sha256") == identity
                 and receipt.get("call_count_scope") == "VENDOR_WRAPPER_INVOCATIONS", "invalid schedule terminal")
        _require(receipt.get("requested_origins") == requested
                 and all(type(gw) is int for gw in receipt["requested_origins"]), "terminal requested range differs")
        for key, value in (("attempted_vendor_calls", attempted), ("completed_vendor_calls", returned),
                           ("completed_origins", len(completed))):
            _require(type(receipt.get(key)) is int and receipt[key] == value, "schedule terminal counters differ")
        if status == "COMPLETED":
            _require(completed == requested and last in ("decision", "schedule-scoring-start"),
                     "incomplete schedule cannot have complete terminal")
        else:
            _require((receipt.get("failed_origin") is None if origin is None else type(receipt.get("failed_origin")) is int
                      and receipt["failed_origin"] == origin) and receipt.get("failed_stage") == last,
                     "schedule failure origin/stage differs")
            failed_origin, failed_stage = origin, last
            if "exception_type" in receipt or "exception" in receipt:
                _require(isinstance(receipt.get("exception_type"), str) and bool(receipt["exception_type"])
                         and isinstance(receipt.get("exception"), str), "invalid recorded schedule error")
                recorded_error = {"type": receipt["exception_type"], "message": receipt["exception"]}
        timings.add(receipt)
    return {"status": status, "requested_origins": requested, "completed_origins": completed,
            "saved_origins": len(completed), "failed_origin": failed_origin, "failed_stage": failed_stage,
            "recorded_error": recorded_error,
            "execution_settings": {key: settings[key] for key in ("season", "start_gw", "max_gw", "horizon", "time_limit")
                                   if key in settings},
            "counts": {"attempted_vendor_calls": attempted, "returned_vendor_calls": returned,
                       "failed_vendor_calls": failed, "unknown_in_flight_calls": attempted-returned-failed,
                       "saved_origins": len(completed)},
            "count_scope": "VENDOR_WRAPPER_INVOCATIONS", "timings": timings.result()}


def _linear_model(model: dict) -> None:
    """Validate the stored mathematical schema structurally without native libraries."""
    _require(type(model.get("schema_version")) is int and model["schema_version"] == 1
             and isinstance(model.get("lp"), dict) and isinstance(model.get("matrix"), dict), "invalid linear model JSON")
    lp, sparse = model["lp"], model["matrix"]
    cols, rows = _int(lp.get("num_col")), _int(lp.get("num_row"))
    _require(type(sparse.get("num_col")) is int and sparse["num_col"] == cols
             and type(sparse.get("num_row")) is int and sparse["num_row"] == rows, "model dimensions differ")
    _require(all(isinstance(lp.get(key), list) for key in ("col_cost", "col_lower", "col_upper", "row_lower", "row_upper"))
             and all(isinstance(sparse.get(key), list) for key in ("start", "p_end", "index", "value")),
             "missing captured linear model arrays")


def _native(files: dict[str, bytes], docs: dict[str, dict], terminal: str | None) -> dict:
    started = docs["started.json"]
    _require(started.get("status") == "DIAGNOSTICS_STARTED" and isinstance(started.get("context"), dict),
             "invalid native diagnostics start")
    origin = _int(started["context"].get("start_gw"), minimum=1)
    identity = _sha(files["started.json"])
    names = sorted(name for name in files if re.fullmatch(r"\d{4,}-native-(?:start|finish|error)-0-event\.json", name))
    known = {"started.json"} | ({terminal} if terminal else set())
    events, info, options = [], None, None
    timings = _Timings(_NATIVE_TIMES)
    for sequence, name in enumerate(names, 1):
        record = docs[name]
        stage = record.get("stage")
        prefix = f"{sequence:04d}-{stage}-0"
        _require(name == prefix+"-event.json" and type(record.get("iteration")) is int and record["iteration"] == 0
                 and type(record.get("sequence")) is int and record["sequence"] == sequence
                 and record.get("started_sha256") == identity, "native event name/run/iteration differs")
        _require(stage == "native-start" and not events or stage in ("native-finish", "native-error")
                 and events == ["native-start"], "invalid native event order")
        artifacts = record.get("artifact_sha256")
        _require(isinstance(artifacts, dict), "missing native artifact references")
        expected = {prefix+"-model.json"} if stage == "native-start" else {prefix+"-native-info.json"}
        if stage == "native-start" and prefix+"-model.mps" in artifacts:
            expected.add(prefix+"-model.mps")
        _manifest(artifacts, files, expected)
        known.update(expected | {name})
        if stage == "native-start":
            _linear_model(docs[prefix+"-model.json"])
            if "model_authority" in record:
                _require(record["model_authority"] == "LOSSLESS_JSON", "unknown native model authority claim")
            if "mps_write_status" in record or "mps_interpretation" in record:
                _require(prefix+"-model.mps" in artifacts and bool(files[prefix+"-model.mps"])
                         and type(record.get("mps_write_status")) is int and record["mps_write_status"] in (0, 1),
                         "MPS review status requires a nonempty referenced view")
            options = record.get("effective_options")
            _require(isinstance(options, dict) and bool(options), "missing effective native options")
            for option, observed in options.items():
                _require(isinstance(option, str) and bool(option) and isinstance(observed, dict)
                         and set(observed) == {"status", "value"} and type(observed["status"]) is int
                         and observed["status"] in (-1, 0, 1), "invalid native option observation")
                value = observed["value"]
                _require(value is None or type(value) in (str, bool) or _number(value), "invalid native option value")
                _require(observed["status"] == 0 or value is None, "unavailable native option has a value")
        else:
            info = docs[prefix+"-native-info.json"]
            _require(isinstance(info.get("status_name"), str) and bool(info["status_name"])
                     and type(info.get("status_value")) is int
                     and all(type(info.get(key)) is bool for key in ("info_valid", "primal_value_valid", "dual_value_valid")),
                     "invalid native status/validity observations")
            _require(isinstance(info.get("values"), dict) and isinstance(info.get("availability"), dict)
                     and set(info["values"]) == set(info["availability"]), "invalid native metric availability")
            for key, value in info["values"].items():
                _require(value is None or _number(value), "invalid recorded native metric")
                _require(value is None or info["info_valid"], "invalid native info cannot contain measured metrics")
                if key == "objective_function_value" and value is not None:
                    _require(info["primal_value_valid"], "objective requires valid primal values")
                if value is not None and ("count" in key or "infeasibil" in key or "violation" in key or "error" in key or key == "mip_gap"):
                    _require(value >= 0, "native nonnegative metric is negative")
                _require(info["availability"][key] == ("NOT AVAILABLE" if value is None else "RECORDED"),
                         "native metric availability differs")
            _require(info.get("caller_timings") == record.get("caller_timings"), "native timing references differ")
            timings.add(info)
        caller = record.get("caller_timings")
        _require(isinstance(caller, dict), "invalid native caller timings")
        timings.add(caller)
        events.append(stage)
    uncommitted = set(files) - known
    if uncommitted:
        if terminal is not None:
            receipt = docs[terminal]
            error = receipt.get("error")
            _require(receipt.get("solver_result_accepted") is False and isinstance(error, dict)
                     and isinstance(error.get("type"), str) and bool(error["type"])
                     and isinstance(error.get("message"), str) and bool(error["message"]),
                     "sealed uncommitted native artifacts require a recorded rejection error")
        if not events:
            prefix = "0001-native-start-0"
            _require(uncommitted <= {prefix+"-model.json", prefix+"-model.mps"}
                     and prefix+"-model.json" in uncommitted, "unknown pending native-start artifacts")
        elif events == ["native-start"]:
            allowed = {"0002-native-finish-0-native-info.json", "0002-native-error-0-native-info.json"}
            _require(len(uncommitted) == 1 and uncommitted <= allowed, "unknown or conflicting pending native outcomes")
        else:
            raise ValueError("unexpected artifacts after a recorded native outcome")
    accepted = recorded_error = None
    if terminal:
        receipt = docs[terminal]
        _require(receipt.get("status") == "DIAGNOSTICS_RECORDED" and receipt.get("started_sha256") == identity
                 and type(receipt.get("solver_result_accepted")) is bool, "invalid native diagnostic terminal")
        accepted = receipt["solver_result_accepted"]
        recorded_error = receipt.get("error")
        _require(recorded_error is None or isinstance(recorded_error, dict)
                 and isinstance(recorded_error.get("type"), str) and bool(recorded_error["type"])
                 and isinstance(recorded_error.get("message"), str), "invalid recorded native error")
        if accepted:
            _require(events == ["native-start", "native-finish"] and info is not None
                     and info["status_name"] == "Optimal" and info["status_value"] == 7 and info["info_valid"] and info["primal_value_valid"]
                     and _number(info["values"].get("mip_gap")) and 0 <= info["values"]["mip_gap"] <= 1e-9
                     and receipt.get("error") is None, "accepted native result lacks qualified recorded completion")
            for option in ("mip_rel_gap", "time_limit"):
                _require(isinstance(options.get(option), dict) and options[option]["status"] == 0
                         and _number(options[option]["value"]), "accepted native result needs recorded gap/time options")
            _require(options["mip_rel_gap"]["value"] == 0 and options["time_limit"]["value"] > 0,
                     "accepted native result has incompatible relative gap/time settings")
    return {"status": "DIAGNOSTICS_RECORDED" if terminal else "INCOMPLETE",
            "solver_result_accepted": accepted, "requested_origins": [origin], "native_info": info,
            "recorded_error": recorded_error,
            "execution_settings": {key: started["context"][key] for key in ("season", "start_gw", "horizon")
                                   if key in started["context"]},
            "effective_options": options, "uncommitted_artifacts": sorted(uncommitted),
            "count_scope": "NATIVE_RUN_BOUNDARIES",
            "counts": {"native_started": int(bool(events)), "native_finished": int("native-finish" in events),
                       "native_error": int("native-error" in events), "native_run_unknown": int(events == ["native-start"])},
            "timings": timings.result()}


def inspect_evidence(directory: Path, *, kind: str, terminal_sha256: str | None) -> dict:
    """Verify pinned operational receipts and return a directory-independent card.

    Provenance describes original source/runtime, never equality with today's code.
    Schedule counts refer to vendor wrapper boundaries; native counts refer to
    native run boundaries. Unsealed valid prefixes are INCOMPLETE, with no inferred
    execution status. Timing totals contain only measured elapsed values.
    """
    _require(kind in ("solio_schedule", "native_diagnostics"), "unsupported operational evidence kind")
    files = _files(Path(directory))
    _require("started.json" in files, "missing operational started receipt")
    docs = {name: _json(content) for name, content in files.items() if name.endswith(".json")}
    _require(all(type(doc.get("schema_version")) is int and doc["schema_version"] == 1
                 for name, doc in docs.items() if not name.endswith("native-info.json")), "invalid operational schema")
    terminals = [name for name in ("complete.json", "failed.json", "diagnostics-complete.json") if name in files]
    _require(len(terminals) <= 1, "conflicting operational terminals")
    terminal = terminals[0] if terminals else None
    if terminal:
        _require(terminal in (("complete.json", "failed.json") if kind == "solio_schedule" else ("diagnostics-complete.json",)),
                 "terminal does not match evidence kind")
        _require(isinstance(terminal_sha256, str) and _DIGEST.fullmatch(terminal_sha256) is not None
                 and _sha(files[terminal]) == terminal_sha256, "missing or mismatched terminal SHA256 pin")
        _manifest(docs[terminal].get("receipt_sha256" if kind == "solio_schedule" else "file_sha256"),
                  files, set(files)-{terminal})
    else:
        _require(terminal_sha256 is None, "terminal pin supplied for an incomplete capture")
    started_sha = _sha(files["started.json"])
    result = {"schema_version": 1, "kind": kind,
              "provenance": _provenance(docs["started.json"], started_sha, terminal_sha256),
              "assessment": "RECORDED_OPERATIONAL_EVIDENCE_ONLY", "promotion_eligible": False,
              "independent_legality": "NOT ASSESSED", "performance_assessment": "NOT ASSESSED"}
    result.update((_schedule if kind == "solio_schedule" else _native)(files, docs, terminal))
    json.dumps(result, allow_nan=False)
    return result
