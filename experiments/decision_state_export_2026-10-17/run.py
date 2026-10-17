"""Bounded synthetic export control; four solves, no research model claims."""
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

from fpl import experiment
from fpl.artifacts import sha256_file, write_json_new
from fpl.research_runner import _source_hashes, _runtime_versions


def main(plan_path: Path, output: Path) -> None:
    plan = json.loads(plan_path.read_bytes())
    if (plan["source_sha256"] != _source_hashes() or plan["runtime"] != _runtime_versions()
            or plan["script_sha256"] != sha256_file(Path(__file__))):
        raise ValueError("registered source/runtime/script changed")
    bundle = Path(plan["input_bundle"])
    for name, digest in plan["input_sha256"].items():
        if sha256_file(bundle / name) != digest:
            raise ValueError("registered synthetic control input changed")
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output / "plan.json", plan)
    frames = {}
    for label, name in (("baseline", "baseline_optimize.py"), ("candidate", "candidate_optimize.py")):
        spec = importlib.util.spec_from_file_location(label + "_export_control", bundle / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module.parse_args(["--predictions-csv", str(bundle / "predictions.csv"),
            "--start-gw", "153", "--max-gw", "154", "--horizon", "1", "--fh-gw", "153",
            "--initial-squad", ",".join(map(str, range(1, 16))), "--initial-budget", "250",
            "--initial-ft", "0", "--scoring-mode", "corrected",
            "--output", str(output / (label + ".csv")),
            "--decisions-json", str(output / (label + ".json"))])
        with (output / (label + ".log")).open("x") as stream, redirect_stdout(stream):
            frames[label] = module.run(args)
        if not frames[label].native_optimality_assessment.eq("PROVEN_OPTIMAL").all():
            raise ValueError("synthetic solve lacks native optimality proof")
    old, new = frames["baseline"], frames["candidate"]
    pd.testing.assert_frame_equal(old, new[old.columns], check_exact=False, atol=1e-9, rtol=0)
    if plan["source_sha256"] != _source_hashes() or plan["runtime"] != _runtime_versions():
        raise ValueError("source/runtime changed during export control")
    result = {"status": "SYNTHETIC_EXPORT_CONTROL_PASS", "model_fits": 0, "optimizer_solves": 4,
        "compared_rounds": 2, "original_columns_unchanged": old.columns.tolist(),
        "added_columns": [name for name in new if name not in old],
        "native_optimality": "PROVEN_OPTIMAL", "promotion_eligible": False,
        "limitation": "Synthetic state/export engineering check; not empirical FPL model performance or independent financial-rule validation."}
    experiment.log_result("decision-state-export-control", {"plan_sha256": sha256_file(plan_path)},
        {"status": result["status"], "optimizer_solves": 4}, results_path=output / "registry.csv")
    result["output_sha256"] = {str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*")) if path.is_file()}
    write_json_new(output / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
