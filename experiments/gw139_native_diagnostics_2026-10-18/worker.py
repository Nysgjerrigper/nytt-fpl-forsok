"""Registered GW139 diagnostic-only worker: one fixed vendor window, no settlement.

This file is prepared for separate review/registration; importing it does not run
anything. Execution requires a pinned plan and two completely verified prior seals.
"""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
import time
import traceback

import pandas as pd

from fpl.artifacts import sha256_file, write_json_new
from fpl.milp.native_diagnostics import deserialize_linear_model, serialize_linear_model
from fpl.milp.origin_prices import load_origin_captures, prepare_origin_frame, selling_prices
from fpl.milp.solio import solve_window
from fpl.milp.solio_inputs import prepare_inputs
from fpl.milp.solio_receipts import finite_value, frame_identity, input_identity
from fpl.research_runner import _source_hashes, _runtime_versions, _verify_outputs


_PRIOR_SEAL = "24760644efaaa7e59c0f856c80e8acf75fb432c952d65007f743a8a6fc9138f7"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    json.dumps(value, allow_nan=False)
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def verify_seal(directory: Path, expected: str) -> dict:
    seal_path = directory / "attempt-seal.json"
    require(sha256_file(seal_path) == expected, f"prior attempt seal SHA mismatch: {directory}")
    seal = read_json(seal_path)
    _verify_outputs(directory, seal["output_sha256"], {"attempt-seal.json"})
    return seal


def canonical_basis(values: dict) -> dict[int, int]:
    require(isinstance(values, dict), "purchase basis must be a mapping")
    require(all(isinstance(key, str) and key.isdecimal() and str(int(key)) == key and int(key) > 0
                and type(value) is int and value > 0 for key, value in values.items()),
            "purchase basis needs canonical positive stable codes and integer tenths")
    return {int(key): value for key, value in values.items()}


def verify_native(directory: Path) -> dict:
    receipt_path = directory / "diagnostics-complete.json"
    result = {"sealed": False, "lossless_model_roundtrips": 0}
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        require(receipt.get("status") == "DIAGNOSTICS_RECORDED", "invalid native diagnostic seal")
        _verify_outputs(directory, receipt["file_sha256"], {"diagnostics-complete.json"})
        result.update(sealed=True, receipt_sha256=sha256_file(receipt_path),
                      solver_result_accepted=receipt["solver_result_accepted"])
    if directory.exists():
        for path in sorted(directory.glob("*-model.json")):
            payload = read_json(path)
            require(serialize_linear_model(deserialize_linear_model(payload)) == payload,
                    "authoritative native JSON failed a lossless model roundtrip")
            result["lossless_model_roundtrips"] += 1
    return result


def main(root: Path) -> None:
    root = root.resolve()
    plan_path = root / "plan.json"
    plan = read_json(plan_path)
    plan_sha = sha256_file(plan_path)
    # Refuse reuse even after failure or interruption; never resume this worker.
    output = root / "first"
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    counters = {"attempted_vendor_calls": 0, "completed_vendor_calls": 0,
                "failed_vendor_calls": 0, "prepared_callbacks": 0}
    callback_sequence = 0
    prepared_checked = False
    accepted = False
    error = None
    returned_rows = None
    stage = "registration-verification"
    sources = runtime = None
    post_verification = {}
    write_json_new(output / "worker-started.json", {"schema_version": 1, "plan_sha256": plan_sha,
        "worker_sha256": sha256_file(__file__), "status": "STARTED", "scope": "DIAGNOSTIC_ONLY",
        "origin_gw": 139, "call_count_scope": "VENDOR_WRAPPER_INVOCATIONS", "promotion_eligible": False})
    try:
        require(sha256_file(__file__) == plan["worker_sha256"], "worker bytes differ from registered plan")
        sources, runtime = _source_hashes(), _runtime_versions()
        require(sources == plan["source_sha256"], "source bytes differ from registered plan")
        require(runtime == plan["runtime"], "runtime differs from registered plan")
        require(plan["prior_attempt_seal_sha256"] == _PRIOR_SEAL, "unexpected checkpoint attempt seal pin")
        prior = Path(plan["prior_bundle"]).resolve()
        forecast = Path(plan["forecast_bundle"]).resolve()
        prior_seal = verify_seal(prior, _PRIOR_SEAL)
        forecast_seal = verify_seal(forecast, plan["forecast_attempt_seal_sha256"])
        prior_plan = read_json(prior / "plan.json")
        require(Path(prior_plan["prior_bundle"]).resolve() == forecast,
                "forecast bundle differs from checkpoint attempt lineage")
        require(prior_plan["prior_attempt_seal_sha256"] == plan["forecast_attempt_seal_sha256"],
                "forecast seal differs from checkpoint lineage")
        checkpoints = prior / "first/candidate-checkpoints"
        origin_start = read_json(checkpoints / "0019-origin-start.json")
        admitted = read_json(checkpoints / "0020-origin-admitted.json")
        expected_prepared = read_json(checkpoints / "0021-prepared.json")
        expected_vendor = read_json(checkpoints / "0022-vendor-start.json")
        require(origin_start["origin_gw"] == admitted["origin_gw"] == expected_prepared["origin_gw"] == 139,
                "saved checkpoints do not describe GW139")
        state = origin_start["state"]
        require(state["bank"] == 18 and state["free_transfers"] == 2
                and state["previous_free_hit_gw"] is None and state["sell_prices"] == {},
                "unexpected registered GW139 state")
        squad = state["squad"]
        require(isinstance(squad, list) and len(squad) == 15 and len(set(squad)) == 15
                and all(type(code) is int and code > 0 for code in squad), "invalid captured owned squad")
        basis = canonical_basis(state["purchase_prices"])
        require(set(basis) == set(squad), "captured purchase basis does not cover exact squad")
        inventory = {chip: {int(half): value for half, value in halves.items()}
                     for chip, halves in state["chip_inventory"].items()}
        stage = "origin-reconstruction"
        captures = load_origin_captures(forecast / "captures/captures.json", season="2023-24",
                                        origins=list(range(137, 153)))
        raw_path = forecast / "candidate-predictions.csv"
        raw_bytes = raw_path.read_bytes()
        require(sha256_file(raw_path) == forecast_seal["output_sha256"]["candidate-predictions.csv"],
                "forecast bytes changed after prior seal verification")
        raw = pd.read_csv(BytesIO(raw_bytes))
        frame = raw.drop(columns=[name for name in raw if name.startswith("actual_") or name == "minutes"])
        frame = frame.loc[(frame.origin_gw == 139) & frame.GW.between(139, 141)]  # Keep original CSV index.
        capture = captures[139]
        frame, price_receipt = prepare_origin_frame(frame, capture, origin_gw=139, owned=squad)
        require(frame_identity(frame) == admitted["admitted_frame"], "reconstructed frame identity differs")
        require(price_receipt == admitted["price_receipt"], "reconstructed price receipt differs")
        sells = selling_prices(capture, squad, basis)
        kwargs = dict(season="2023-24", start_gw=139, horizon=3, initial_squad=squad,
                      bank=18, free_transfers=2, sell_prices=sells, bootstrap=capture["bootstrap"])
        prepared = prepare_inputs(frame, **kwargs)
        require(input_identity(prepared) == expected_prepared["prepared_inputs"],
                "reconstructed prepared inputs differ")
        write_json_new(output / "reconstruction.json", {"schema_version": 1,
            "scope": "DIAGNOSTIC_ONLY", "admitted_frame": frame_identity(frame), "price_receipt": price_receipt,
            "prepared_inputs": input_identity(prepared), "state": state,
            "prior_seal_sha256": _PRIOR_SEAL, "forecast_seal_sha256": plan["forecast_attempt_seal_sha256"],
            "verified_prior_files": len(prior_seal["output_sha256"]),
            "verified_forecast_files": len(forecast_seal["output_sha256"]),
            "promotion_eligible": False})

        def callback(event: str, values: dict) -> None:
            nonlocal callback_sequence, prepared_checked, stage
            stage = event
            if event == "prepared":
                require(counters["prepared_callbacks"] == 0, "duplicate preparation callback")
                require(values["prepared_inputs"] == expected_prepared["prepared_inputs"],
                        "solve_window preparation differs before vendor")
                require(finite_value(values["options"]) == expected_prepared["options"],
                        "solve_window options differ before vendor")
                counters["prepared_callbacks"] += 1
                prepared_checked = True
            elif event == "vendor-start":
                require(prepared_checked and counters["attempted_vendor_calls"] == 0,
                        "unverified or duplicate vendor invocation")
                require(values["time_limit_seconds"] == expected_vendor["time_limit_seconds"] == 120
                        and values["relative_gap"] == expected_vendor["relative_gap"] == 0,
                        "vendor budget or gap differs from registered boundary")
                counters["attempted_vendor_calls"] += 1
            elif event == "vendor-complete":
                counters["completed_vendor_calls"] += 1
            elif event == "vendor-failed":
                counters["failed_vendor_calls"] += 1
            callback_sequence += 1
            write_json_new(output / f"callback-{callback_sequence:04d}-{event}.json", finite_value({
                "schema_version": 1, "event": event, "origin_gw": 139, "values": values,
                "counters": dict(counters), "scope": "DIAGNOSTIC_ONLY"}))

        stage = "solve-window"
        decisions = solve_window(frame, **kwargs, chip_inventory=inventory, forced_chips={},
            previous_free_hit_gw=None, time_limit=120, seed=0,
            native_diagnostics_dir=output / "native", _event_callback=callback)
        accepted = True
        returned_rows = len(decisions)
        # No decision export, settlement, scoring, model training or second solve.
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "stage": stage,
                 "traceback": traceback.format_exc()}
    try:
        post_verification["plan_unchanged"] = sha256_file(plan_path) == plan_sha
        post_verification["worker_unchanged"] = sha256_file(__file__) == plan["worker_sha256"]
        post_verification["source_unchanged"] = _source_hashes() == plan["source_sha256"]
        post_verification["runtime_unchanged"] = _runtime_versions() == plan["runtime"]
        require(all(post_verification.values()), "source/runtime/plan/worker changed during attempt")
        post_verification["native"] = verify_native(output / "native")
    except Exception as exc:
        post_verification["error"] = {"type": type(exc).__name__, "message": str(exc),
                                      "traceback": traceback.format_exc()}
    result = {"schema_version": 1, "status": "DIAGNOSTIC_ATTEMPT_RECORDED", "scope": "DIAGNOSTIC_ONLY",
        "plan_sha256": plan_sha, "origin_gw": 139, "solver_result_accepted": accepted,
        "returned_decision_rows": returned_rows, "error": error, "post_verification": post_verification,
        "wall_seconds": time.perf_counter() - started, "counters": counters,
        "call_count_scope": "VENDOR_WRAPPER_INVOCATIONS", "settlement_performed": False,
        "model_fits": 0, "promotion_eligible": False}
    write_json_new(output / "worker-result.json", result)
    write_json_new(output / "result-seal.json", {"schema_version": 1, "status": "DIAGNOSTIC_EVIDENCE_SEALED",
        "scope": "DIAGNOSTIC_ONLY", "promotion_eligible": False,
        "output_sha256": {str(path.relative_to(output)): sha256_file(path)
                          for path in sorted(output.rglob("*")) if path.is_file()}})
    if error is not None or "error" in post_verification:
        raise SystemExit(2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: fpl-gw139-diagnostic-worker.py BUNDLE_ROOT")
    main(Path(sys.argv[1]))
