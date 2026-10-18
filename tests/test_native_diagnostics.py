"""Tiny native-model round trips and immutable diagnostic evidence; no FPL fits."""
from copy import deepcopy
import json
import math

import highspy
import pytest

from fpl import config
from fpl.artifacts import sha256_file
from fpl.milp import native_diagnostics
from fpl.milp.native_diagnostics import NativeDiagnostics, deserialize_linear_model, serialize_linear_model


def linear_model():
    model = highspy.HighsModel()
    lp = model.lp_
    lp.num_col_, lp.num_row_ = 3, 2
    lp.col_cost_ = [1.1234567890123457, 2., -3.]
    lp.col_lower_, lp.col_upper_ = [-0., 0., 0.], [math.inf, 3., 4.]
    lp.row_lower_, lp.row_upper_ = [1., -math.inf], [math.inf, 5.]
    lp.col_names_, lp.row_names_ = ["a", "b", "c"], ["r0", "r1"]
    lp.model_name_, lp.sense_, lp.offset_ = "tiny-model", highspy.ObjSense.kMinimize, 2.25
    lp.integrality_ = [highspy.HighsVarType.kInteger, highspy.HighsVarType.kContinuous, highspy.HighsVarType.kInteger]
    matrix = lp.a_matrix_
    matrix.num_col_, matrix.num_row_, matrix.format_ = 3, 2, highspy.MatrixFormat.kColwise
    matrix.start_, matrix.index_, matrix.value_ = [0, 2, 3, 5], [1, 0, 0, 1, 0], [2., 1., 2., 1., 3.]
    return model


def native(model=None):
    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    assert solver.passModel(linear_model() if model is None else model) == highspy.HighsStatus.kOk
    return solver


def read(path):
    return json.loads(path.read_bytes())


def test_lossless_roundtrip_through_json_and_fresh_native_model():
    solver = native()
    original = serialize_linear_model(solver.getModel())
    payload = json.loads(json.dumps(original, allow_nan=False))
    assert payload["matrix"]["index"] == [1, 0, 0, 1, 0]  # Preserve unsorted native sparse order.
    assert payload["lp"]["col_upper"][0] == "+inf"
    assert payload["lp"]["row_lower"][1] == "-inf"
    restored = deserialize_linear_model(payload)
    assert serialize_linear_model(restored) == original
    fresh = native(restored)
    assert serialize_linear_model(fresh.getModel()) == original
    assert math.copysign(1, restored.lp_.col_lower_[0]) == -1
    assert fresh.run() == highspy.HighsStatus.kOk
    assert solver.run() == highspy.HighsStatus.kOk
    assert fresh.getObjectiveValue() == solver.getObjectiveValue()


def test_changed_bounds_and_coefficients_change_model_and_optimum():
    original = serialize_linear_model(linear_model())
    changed = deepcopy(original)
    changed["lp"]["col_upper"][2] = 0.
    changed["lp"]["col_cost"][1] = 10.
    changed["matrix"]["value"][0] = 4.
    one, two = native(deserialize_linear_model(original)), native(deserialize_linear_model(changed))
    one.run()
    two.run()
    assert one.getObjectiveValue() != two.getObjectiveValue()
    assert serialize_linear_model(two.getModel()) == changed


@pytest.mark.parametrize("fmt", [highspy.MatrixFormat.kRowwise, highspy.MatrixFormat.kRowwisePartitioned])
def test_rowwise_formats_and_partition_end_preserved(fmt):
    model = linear_model()
    matrix = model.lp_.a_matrix_
    matrix.format_, matrix.start_ = fmt, [0, 3, 5]
    matrix.index_, matrix.value_ = [2, 0, 1, 2, 0], [3., 1., 2., 1., 2.]
    matrix.p_end_ = [2, 4] if fmt == highspy.MatrixFormat.kRowwisePartitioned else []
    payload = serialize_linear_model(model)
    restored = deserialize_linear_model(json.loads(json.dumps(payload)))
    assert serialize_linear_model(restored) == payload
    assert payload["matrix"]["p_end"] == list(matrix.p_end_)


def test_all_integrality_enums_sense_offset_and_absent_names_preserved():
    model = linear_model()
    model.lp_.sense_, model.lp_.offset_ = highspy.ObjSense.kMaximize, -2.5
    model.lp_.col_names_, model.lp_.row_names_ = [], []
    for value in highspy.HighsVarType.__members__.values():
        model.lp_.integrality_ = [value] * 3
        payload = serialize_linear_model(model)
        assert serialize_linear_model(deserialize_linear_model(payload)) == payload
    model.lp_.integrality_ = []
    payload = serialize_linear_model(model)
    assert serialize_linear_model(deserialize_linear_model(payload)) == payload


@pytest.mark.parametrize("where", ["cost", "bounds", "matrix", "offset", "hessian"])
def test_nan_and_quadratic_models_rejected(where):
    model = linear_model()
    if where == "cost": model.lp_.col_cost_ = [math.nan, 2., -3.]
    elif where == "bounds": model.lp_.row_lower_ = [math.nan, -math.inf]
    elif where == "matrix": model.lp_.a_matrix_.value_ = [math.nan, 1., 2., 1., 3.]
    elif where == "offset": model.lp_.offset_ = math.nan
    else:
        model.hessian_.dim_ = 3
        model.hessian_.start_ = [0, 1, 1, 1]
        model.hessian_.index_ = [0]
        model.hessian_.value_ = [1.]
    with pytest.raises(ValueError): serialize_linear_model(model)


@pytest.mark.parametrize("change", ["schema", "extra", "nan", "inf", "bool", "enum", "dimension", "start", "index"])
def test_untrusted_payload_rejected(change):
    payload = serialize_linear_model(linear_model())
    if change == "schema": payload["schema_version"] = True
    elif change == "extra": payload["lp"]["extra"] = 1
    elif change == "nan": payload["lp"]["offset"] = math.nan
    elif change == "inf": payload["matrix"]["value"][0] = math.inf
    elif change == "bool": payload["lp"]["col_upper"][0] = True
    elif change == "enum": payload["lp"]["integrality"][0] = 99
    elif change == "dimension": payload["matrix"]["num_row"] = 3
    elif change == "start": payload["matrix"]["start"] = [0, 4, 3, 5]
    else: payload["matrix"]["index"][0] = 99
    with pytest.raises(ValueError): deserialize_linear_model(payload)


def test_native_observations_bind_options_models_info_and_exhaustive_hashes(tmp_path):
    solver = native()
    solver.setOptionValue("random_seed", 37)
    context = {"origin_gw": 153, "scope": "tiny-native-test"}
    diagnostic = NativeDiagnostics(tmp_path / "diagnostic", context)
    diagnostic.observe("native-start", solver, 0, {"model_build_seconds": .01})
    start = read(next(diagnostic.root.glob("*-native-start-*-event.json")))
    assert start["effective_options"]["random_seed"]["value"] == 37
    assert start["native_runtime_seconds"] is None
    assert start["model_authority"] == "LOSSLESS_JSON"
    saved = read(next(diagnostic.root.glob("*-model.json")))
    assert saved == serialize_linear_model(solver.getModel())
    assert next(diagnostic.root.glob("*.mps")).is_file()
    solver.run()
    diagnostic.observe("native-finish", solver, 0, {"native_call_wall_seconds": .02})
    info = read(next(diagnostic.root.glob("*-native-info.json")))
    assert info["status_name"] == "Optimal"
    assert info["info_valid"] is True and info["primal_value_valid"] is True
    assert info["values"]["objective_function_value"] == solver.getObjectiveValue()
    assert info["native_runtime_seconds"] >= 0
    assert info["interpretation"] == "DIAGNOSTIC_ONLY"
    diagnostic.seal(False, RuntimeError("wrapper rejected a later invariant"))
    receipt = read(diagnostic.root / "diagnostics-complete.json")
    assert receipt["status"] == "DIAGNOSTICS_RECORDED"
    assert receipt["solver_result_accepted"] is False
    assert receipt["error"]["type"] == "RuntimeError"
    files = {str(path.relative_to(diagnostic.root)) for path in diagnostic.root.rglob("*") if path.is_file()}
    assert set(receipt["file_sha256"]) == files - {"diagnostics-complete.json"}
    for name, digest in receipt["file_sha256"].items():
        assert sha256_file(diagnostic.root / name) == digest
    for path in diagnostic.root.glob("*-event.json"):
        assert read(path)["started_sha256"] == sha256_file(diagnostic.root / "started.json")
    json.dumps(receipt, allow_nan=False)


def test_invalid_info_and_nonfinite_unknown_fields_are_null(tmp_path):
    solver = native()
    diagnostic = NativeDiagnostics(tmp_path / "diagnostic", {})
    diagnostic.observe("native-start", solver, 0, {})
    diagnostic.observe("native-error", solver, 0, {"native_call_wall_seconds": .1})
    info = read(next(diagnostic.root.glob("*-native-info.json")))
    assert info["info_valid"] is False
    assert info["values"]["objective_function_value"] is None
    assert info["availability"]["mip_gap"] == "NOT AVAILABLE"
    assert not (diagnostic.root / "diagnostics-complete.json").exists()
    diagnostic.seal(False, ValueError("native exception"))


def test_nonfinite_and_unavailable_info_remain_diagnostic_only(tmp_path):
    solver = native()
    solver.run()
    real_info = solver.getInfo()
    real_info.mip_gap = math.inf
    real_info.mip_dual_bound = math.nan
    real_info.mip_node_count = -1
    class Probe:
        def getInfo(self): return real_info
        def getSolution(self): return solver.getSolution()
        def getModelStatus(self): return solver.getModelStatus()
        def modelStatusToString(self, status): return solver.modelStatusToString(status)
        def getRunTime(self): return math.inf
    diagnostic = NativeDiagnostics(tmp_path / "diagnostic", {})
    diagnostic.observe("native-start", solver, 0, {})
    diagnostic.observe("native-finish", Probe(), 0, {})
    info = read(next(diagnostic.root.glob("*-native-info.json")))
    assert info["values"]["mip_gap"] is None
    assert info["values"]["mip_dual_bound"] is None
    assert info["values"]["mip_node_count"] is None
    assert info["native_runtime_seconds"] is None
    assert info["interpretation"] == "DIAGNOSTIC_ONLY"


def test_duplicate_events_seals_and_directory_reuse_refused(tmp_path):
    root = tmp_path / "diagnostic"
    diagnostic = NativeDiagnostics(root, {})
    solver = native()
    diagnostic.observe("native-start", solver, 0, {})
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(ValueError, match="duplicate"):
        diagnostic.observe("native-start", solver, 0, {})
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(FileExistsError): NativeDiagnostics(root, {})
    diagnostic.seal(True)
    assert read(root / "diagnostics-complete.json")["solver_result_accepted"] is True
    with pytest.raises(ValueError): diagnostic.seal(True)
    with pytest.raises(ValueError): diagnostic.observe("native-finish", solver, 0, {})


@pytest.mark.parametrize("folder", ["source", "dataset"])
def test_unsafe_destination_rejected_before_creation(folder):
    target = (config.ROOT / "fpl" if folder == "source" else config.DATASETT_DIR) / "forbidden-native-diagnostic-test"
    with pytest.raises(ValueError, match="cannot target"):
        NativeDiagnostics(target, {})
    assert not target.exists()


def test_model_export_failure_is_not_silently_accepted(tmp_path, monkeypatch):
    solver = native()
    class BadExport:
        def getModel(self): return solver.getModel()
        def writeModel(self, path): return highspy.HighsStatus.kError
    diagnostic = NativeDiagnostics(tmp_path / "diagnostic", {})
    monkeypatch.setattr(native_diagnostics, "_mps_review_writer", lambda payload: BadExport())
    with pytest.raises(RuntimeError, match="MPS"):
        diagnostic.observe("native-start", solver, 0, {})
    assert not (diagnostic.root / "diagnostics-complete.json").exists()


def test_objective_requires_valid_info_and_primal_values(tmp_path):
    solver = native()
    solver.run()
    solution = solver.getSolution()
    solution.value_valid = False
    class MissingPrimal:
        def getInfo(self): return solver.getInfo()
        def getSolution(self): return solution
        def getModelStatus(self): return solver.getModelStatus()
        def modelStatusToString(self, status): return solver.modelStatusToString(status)
        def getRunTime(self): return solver.getRunTime()
    diagnostic = NativeDiagnostics(tmp_path / "diagnostic", {})
    diagnostic.observe("native-start", solver, 0, {})
    diagnostic.observe("native-finish", MissingPrimal(), 0, {})
    info = read(next(diagnostic.root.glob("*-native-info.json")))
    assert info["info_valid"] is True
    assert info["primal_value_valid"] is False
    assert info["values"]["objective_function_value"] is None
    assert info["availability"]["objective_function_value"] == "NOT AVAILABLE"


def test_invalid_context_and_timings_fail_closed(tmp_path):
    with pytest.raises(ValueError):
        NativeDiagnostics(tmp_path / "bad-context", {"value": math.nan})
    assert not (tmp_path / "bad-context").exists()
    diagnostic = NativeDiagnostics(tmp_path / "diagnostic", {})
    with pytest.raises(ValueError):
        diagnostic.observe("native-start", native(), 0, {"model_build_seconds": math.nan})
    assert list(path.name for path in diagnostic.root.iterdir()) == ["started.json"]
    assert not (diagnostic.root / "diagnostics-complete.json").exists()


@pytest.mark.parametrize("field", ["col_cost", "matrix", "offset"])
@pytest.mark.parametrize("value", [math.inf, -math.inf])
def test_infinite_coefficients_and_offsets_rejected(field, value):
    model = linear_model()
    if field == "col_cost": model.lp_.col_cost_ = [value, 2., -3.]
    elif field == "matrix": model.lp_.a_matrix_.value_ = [value, 1., 2., 1., 3.]
    else: model.lp_.offset_ = value
    with pytest.raises(ValueError, match="finite"):
        serialize_linear_model(model)


@pytest.mark.parametrize("field", ["col_cost", "matrix", "offset"])
@pytest.mark.parametrize("token", ["+inf", "-inf"])
def test_symbolic_infinite_coefficients_and_offsets_rejected(field, token):
    payload = serialize_linear_model(linear_model())
    if field == "col_cost": payload["lp"]["col_cost"][0] = token
    elif field == "matrix": payload["matrix"]["value"][0] = token
    else: payload["lp"]["offset"] = token
    with pytest.raises(ValueError): deserialize_linear_model(payload)


def test_native_outcome_stage_order_and_exclusivity(tmp_path):
    diagnostic = NativeDiagnostics(tmp_path / "diagnostic", {})
    solver = native()
    with pytest.raises(ValueError, match="native-start"):
        diagnostic.observe("native-finish", solver, 0, {})
    with pytest.raises(ValueError, match="native-start"):
        diagnostic.observe("native-error", solver, 0, {})
    assert list(path.name for path in diagnostic.root.iterdir()) == ["started.json"]
    diagnostic.observe("native-start", solver, 0, {})
    diagnostic.observe("native-error", solver, 0, {})
    with pytest.raises(ValueError, match="mutually exclusive"):
        diagnostic.observe("native-finish", solver, 0, {})


def test_real_unnamed_rows_warning_preserves_authoritative_model(tmp_path):
    model = linear_model()
    model.lp_.row_names_ = []
    solver = native(model)
    before = serialize_linear_model(solver.getModel())
    diagnostic = NativeDiagnostics(tmp_path / "diagnostic", {})
    diagnostic.observe("native-start", solver, 0, {})
    event = read(next(diagnostic.root.glob("*-event.json")))
    assert event["mps_write_status"] == int(highspy.HighsStatus.kWarning)
    assert event["mps_write_status_name"] == "kWarning"
    assert event["mps_interpretation"] == "REVIEW_ONLY_GENERATED_NAMES_POSSIBLE"
    assert next(diagnostic.root.glob("*.mps")).stat().st_size > 0
    assert read(next(diagnostic.root.glob("*-model.json"))) == before
    assert serialize_linear_model(solver.getModel()) == before


@pytest.mark.parametrize("status", [highspy.HighsStatus.kWarning, highspy.HighsStatus.kOk])
@pytest.mark.parametrize("empty_file", [False, True])
def test_export_status_without_nonempty_file_is_rejected(tmp_path, status, empty_file, monkeypatch):
    solver = native()
    class BadExport:
        def getModel(self): return solver.getModel()
        def writeModel(self, path):
            if empty_file:
                from pathlib import Path
                Path(path).touch()
            return status
    diagnostic = NativeDiagnostics(tmp_path / "diagnostic", {})
    monkeypatch.setattr(native_diagnostics, "_mps_review_writer", lambda payload: BadExport())
    with pytest.raises(RuntimeError, match="MPS"):
        diagnostic.observe("native-start", solver, 0, {})
    assert not list(diagnostic.root.glob("*-event.json"))
    assert not (diagnostic.root / "diagnostics-complete.json").exists()
