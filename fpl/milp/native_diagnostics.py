"""Lossless linear-model capture and immutable native diagnostic observations.

Artifacts describe native state only. Recording diagnostics, including an incumbent,
never certifies an accepted recommendation or runs/modifies a solver.
"""
from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import highspy

from fpl import config
from fpl.artifacts import sha256_file, write_json_new
from fpl.milp.solio_receipts import finite_value, source_runtime


_LP_FIELDS = ("num_col", "num_row", "col_cost", "col_lower", "col_upper", "row_lower",
              "row_upper", "integrality", "col_names", "row_names", "model_name", "sense", "offset")
_MATRIX_FIELDS = ("format", "num_col", "num_row", "start", "p_end", "index", "value")
_BOUND_FIELDS = {"col_lower", "col_upper", "row_lower", "row_upper"}
_OPTIONS = ("time_limit", "mip_rel_gap", "mip_abs_gap", "presolve", "parallel", "threads", "random_seed")


def _encode_number(value: Real, *, allow_infinite: bool = False) -> float | str:
    number = float(value)
    if math.isnan(number):
        raise ValueError("NaN is not an admissible linear model coefficient/bound")
    if math.isinf(number):
        if not allow_infinite:
            raise ValueError("linear coefficients and offsets must be finite")
        return "+inf" if number > 0 else "-inf"
    return number


def _decode_number(value: Any, *, allow_infinite: bool = False) -> float:
    if allow_infinite and isinstance(value, str) and value in ("+inf", "-inf"):
        return math.inf if value == "+inf" else -math.inf
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError("model numeric fields require finite numbers or explicit infinity strings")
    return float(value)


def _integer(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("model dimensions/indices require nonnegative JSON integers")
    return value


def _enum(value: Any, enum_type):
    if type(value) is not int:
        raise ValueError("model enum requires an exact JSON integer")
    if value not in {int(member) for member in enum_type.__members__.values()}:
        raise ValueError("unknown native model enum")
    return enum_type(value)


def serialize_linear_model(model: highspy.HighsModel) -> dict:
    """Preserve the complete mathematical linear model and sparse ordering.

    Bound-only infinity tokens keep the payload finite JSON; Python JSON doubles preserve
    every finite binary64 value. Quadratic Hessians are deliberately unsupported.
    Solver basis, presolve state and native internal caches are outside this format.
    """
    hessian = model.hessian_
    if hessian.dim_ or len(hessian.value_) or len(hessian.index_) or list(hessian.start_) not in ([], [0]):
        raise ValueError("nonempty quadratic Hessian is unsupported")
    lp, matrix = model.lp_, model.lp_.a_matrix_
    fields = {"num_col": int(lp.num_col_), "num_row": int(lp.num_row_),
              "integrality": [int(value) for value in lp.integrality_],
              "col_names": list(lp.col_names_), "row_names": list(lp.row_names_),
              "model_name": lp.model_name_, "sense": int(lp.sense_), "offset": _encode_number(lp.offset_)}
    for name in ("col_cost", "col_lower", "col_upper", "row_lower", "row_upper"):
        fields[name] = [_encode_number(value, allow_infinite=name in _BOUND_FIELDS)
                        for value in getattr(lp, name + "_")]
    sparse = {"format": int(matrix.format_), "num_col": int(matrix.num_col_), "num_row": int(matrix.num_row_)}
    for name in ("start", "p_end", "index"):
        sparse[name] = [int(value) for value in getattr(matrix, name + "_")]
    sparse["value"] = [_encode_number(value) for value in matrix.value_]
    payload = {"schema_version": 1, "lp": fields, "matrix": sparse}
    deserialize_linear_model(payload)  # Validate structural/schema invariants before publishing.
    json.dumps(payload, allow_nan=False)
    return payload


def deserialize_linear_model(payload: dict) -> highspy.HighsModel:
    """Reconstruct a new HighsModel without solving, setting options or using pickle."""
    if (not isinstance(payload, dict) or set(payload) != {"schema_version", "lp", "matrix"}
            or type(payload["schema_version"]) is not int or payload["schema_version"] != 1):
        raise ValueError("invalid linear-model schema")
    lp, sparse = payload["lp"], payload["matrix"]
    if not isinstance(lp, dict) or set(lp) != set(_LP_FIELDS) or not isinstance(sparse, dict) or set(sparse) != set(_MATRIX_FIELDS):
        raise ValueError("invalid linear-model fields")
    ncol, nrow = _integer(lp["num_col"]), _integer(lp["num_row"])
    if (_integer(sparse["num_col"]), _integer(sparse["num_row"])) != (ncol, nrow):
        raise ValueError("matrix and LP dimensions disagree")
    arrays = {}
    for name, length in (("col_cost", ncol), ("col_lower", ncol), ("col_upper", ncol),
                         ("row_lower", nrow), ("row_upper", nrow)):
        if not isinstance(lp[name], list) or len(lp[name]) != length:
            raise ValueError(f"invalid {name} length")
        arrays[name] = [_decode_number(value, allow_infinite=name in _BOUND_FIELDS) for value in lp[name]]
    for name, length in (("col_names", ncol), ("row_names", nrow), ("integrality", ncol)):
        if not isinstance(lp[name], list) or len(lp[name]) not in (0, length):
            raise ValueError(f"invalid {name} length")
    if (not isinstance(lp["model_name"], str)
            or any(not isinstance(name, str) for name in lp["col_names"] + lp["row_names"])):
        raise ValueError("native model names must be strings")
    integrality = [_enum(value, highspy.HighsVarType) for value in lp["integrality"]]
    sense = _enum(lp["sense"], highspy.ObjSense)
    fmt = _enum(sparse["format"], highspy.MatrixFormat)
    for name in ("start", "p_end", "index", "value"):
        if not isinstance(sparse[name], list):
            raise ValueError("native sparse arrays must be JSON lists")
    start, end, index = ([_integer(value) for value in sparse[name]] for name in ("start", "p_end", "index"))
    values = [_decode_number(value) for value in sparse["value"]]
    major = ncol if fmt == highspy.MatrixFormat.kColwise else nrow
    minor = nrow if fmt == highspy.MatrixFormat.kColwise else ncol
    if (len(start) != major + 1 or start[0] != 0 or start[-1] != len(index)
            or len(index) != len(values) or start != sorted(start) or any(value >= minor for value in index)
            or (end and (fmt != highspy.MatrixFormat.kRowwisePartitioned or len(end) != major
                         or any(not start[i] <= value <= start[i+1] for i, value in enumerate(end))))):
        raise ValueError("invalid sparse matrix structure")
    model = highspy.HighsModel()
    native = model.lp_
    native.num_col_, native.num_row_ = ncol, nrow
    for name, value in arrays.items():
        setattr(native, name + "_", value)
    native.col_names_, native.row_names_ = lp["col_names"], lp["row_names"]
    native.model_name_, native.sense_, native.offset_ = lp["model_name"], sense, _decode_number(lp["offset"])
    native.integrality_ = integrality
    matrix = native.a_matrix_
    matrix.num_col_, matrix.num_row_, matrix.format_ = ncol, nrow, fmt
    matrix.start_, matrix.p_end_, matrix.index_, matrix.value_ = start, end, index, values
    return model


def _available_number(value: Any, *, nonnegative: bool = False) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        return None
    if nonnegative and value < 0:
        return None
    return int(value) if isinstance(value, int) else float(value)


def _native_info(model: highspy.Highs, timings: dict) -> dict:
    info, solution, status = model.getInfo(), model.getSolution(), model.getModelStatus()
    values, availability = {}, {}
    for name in sorted(dir(info)):
        if name.startswith("_") or name == "valid":
            continue
        value = getattr(info, name)
        if not isinstance(value, Real):
            continue
        nonnegative = ("count" in name or "infeasibil" in name or "violation" in name
                       or "error" in name or name == "mip_gap")
        available = bool(info.valid)
        if name == "objective_function_value":
            available = available and bool(solution.value_valid)
        measured = _available_number(value, nonnegative=nonnegative) if available else None
        values[name] = measured
        availability[name] = "RECORDED" if measured is not None else "NOT AVAILABLE"
    runtime = _available_number(model.getRunTime(), nonnegative=True)
    return {"status_name": model.modelStatusToString(status), "status_value": int(status),
            "info_valid": bool(info.valid), "primal_value_valid": bool(solution.value_valid),
            "dual_value_valid": bool(solution.dual_valid), "values": values, "availability": availability,
            "native_runtime_seconds": runtime,
            "native_runtime_availability": "RECORDED" if runtime is not None else "NOT AVAILABLE",
            "native_runtime_scope": "HIGHS_REPORTED_CUMULATIVE", "caller_timings": finite_value(timings),
            "interpretation": "DIAGNOSTIC_ONLY"}


def _mps_review_writer(payload: dict) -> highspy.Highs:
    """Create a disposable review writer; never change the original native state.

    HiGHS MPS export can generate names in its model. Only this disposable copy
    receives the model and log option, and it is never run.
    """
    writer = highspy.Highs()
    if writer.setOptionValue("output_flag", False) != highspy.HighsStatus.kOk:
        raise RuntimeError("MPS review writer logging option could not be configured")
    if writer.passModel(deserialize_linear_model(payload)) != highspy.HighsStatus.kOk:
        raise RuntimeError("authoritative model could not be passed to MPS review writer")
    return writer


class NativeDiagnostics:
    """Record immutable observations without modifying native state or accepting a solve."""

    def __init__(self, directory: str | Path, context: dict) -> None:
        if not isinstance(context, dict):
            raise ValueError("diagnostic context must be a dictionary")
        self.root = Path(directory).resolve()
        if any(self.root.is_relative_to(path.resolve()) for path in (config.ROOT / "fpl", config.DATASETT_DIR)):
            raise ValueError("diagnostic directory cannot target production code or datasets")
        sources = source_runtime()
        sources["source_sha256"][str(Path(__file__).resolve().relative_to(config.ROOT))] = sha256_file(__file__)
        started = {"schema_version": 1, "status": "DIAGNOSTICS_STARTED", "context": finite_value(context),
                   **sources, "highs_version": highspy.Highs().version(), "interpretation": "DIAGNOSTIC_ONLY"}
        json.dumps(started, allow_nan=False)
        self.root.mkdir(parents=True, exist_ok=False)
        write_json_new(self.root / "started.json", started)
        self.identity = sha256_file(self.root / "started.json")
        self.seen: set[tuple[str, int]] = set()
        self.sequence = 0
        self.sealed = False

    def observe(self, stage: str, model: highspy.Highs, iteration: int, timings: dict) -> None:
        """Capture boundaries without changing live native state; MPS uses a disposable copy."""
        if self.sealed:
            raise ValueError("native diagnostics are already sealed")
        if stage not in ("native-start", "native-finish", "native-error"):
            raise ValueError("unsupported native diagnostic stage")
        if type(iteration) is not int or iteration < 0 or not isinstance(timings, dict):
            raise ValueError("native diagnostic iteration/timings are invalid")
        if (stage, iteration) in self.seen:
            raise ValueError("duplicate native diagnostic stage/iteration")
        if stage != "native-start":
            if ("native-start", iteration) not in self.seen:
                raise ValueError("native outcome requires a recorded native-start")
            if any((outcome, iteration) in self.seen for outcome in ("native-finish", "native-error")):
                raise ValueError("native-finish/native-error are mutually exclusive")
        caller_timings = finite_value(timings)
        json.dumps(caller_timings, allow_nan=False)
        sequence = self.sequence + 1
        prefix = f"{sequence:04d}-{stage}-{iteration}"
        artifacts = {}
        if stage == "native-start":
            model_path = self.root / f"{prefix}-model.json"
            payload = serialize_linear_model(model.getModel())
            write_json_new(model_path, payload)
            artifacts[model_path.name] = sha256_file(model_path)
            mps = self.root / f"{prefix}-model.mps"
            if mps.exists():
                raise FileExistsError(mps)
            status_mps = _mps_review_writer(payload).writeModel(str(mps))
            if (status_mps not in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning)
                    or not mps.is_file() or mps.stat().st_size == 0):
                raise RuntimeError("native diagnostic MPS review view could not be written")
            artifacts[mps.name] = sha256_file(mps)
            options = {}
            for name in _OPTIONS:
                status, value = model.getOptionValue(name)
                options[name] = {"status": int(status), "value": finite_value(value) if status == highspy.HighsStatus.kOk else None}
                if isinstance(options[name]["value"], float) and math.isinf(options[name]["value"]):
                    options[name]["value"] = _encode_number(options[name]["value"], allow_infinite=True)
            detail = {"effective_options": options, "caller_timings": caller_timings,
                      "mps_write_status": int(status_mps), "mps_write_status_name": status_mps.name,
                      "mps_interpretation": "REVIEW_ONLY_GENERATED_NAMES_POSSIBLE",
                      "model_authority": "LOSSLESS_JSON", "native_runtime_seconds": None,
                      "native_runtime_availability": "NOT MEASURED BEFORE RUN"}
        else:
            info_path = self.root / f"{prefix}-native-info.json"
            write_json_new(info_path, _native_info(model, caller_timings))
            artifacts[info_path.name] = sha256_file(info_path)
            detail = {"caller_timings": caller_timings}
        write_json_new(self.root / f"{prefix}-event.json", {"schema_version": 1, "stage": stage,
            "iteration": iteration, "sequence": sequence, "started_sha256": self.identity,
            "artifact_sha256": artifacts, "interpretation": "DIAGNOSTIC_ONLY", **detail})
        self.sequence = sequence
        self.seen.add((stage, iteration))

    def seal(self, accepted: bool, error: Exception | None = None) -> None:
        """Seal recorded evidence; completion never implies solver result acceptance."""
        if self.sealed:
            raise ValueError("native diagnostics are already sealed")
        if type(accepted) is not bool:
            raise ValueError("solver-result acceptance must be explicit boolean")
        record = {"schema_version": 1, "status": "DIAGNOSTICS_RECORDED",
                  "solver_result_accepted": accepted, "started_sha256": self.identity,
                  "interpretation": "DIAGNOSTIC_ONLY", "error": None if error is None else {
                      "type": type(error).__name__, "message": str(error)},
                  "file_sha256": {str(path.relative_to(self.root)): sha256_file(path)
                                  for path in sorted(self.root.rglob("*")) if path.is_file()}}
        write_json_new(self.root / "diagnostics-complete.json", record)
        self.sealed = True
