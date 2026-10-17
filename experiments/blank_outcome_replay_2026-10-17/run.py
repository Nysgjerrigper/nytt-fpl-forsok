"""Registered explicit-blank qualification; preserve raw forecasts and failed attempts."""
import json
from pathlib import Path
import sys

from fpl import experiment
from fpl.artifacts import sha256_file, write_json_new
from fpl.milp.replay import run_replay, read_replay
from fpl.research_runner import _source_hashes, _runtime_versions, read_result


def main(plan_path: Path, output: Path) -> None:
    plan = json.loads(plan_path.read_bytes())
    if (plan["source_sha256"] != _source_hashes() or plan["runtime"] != _runtime_versions()
            or plan["script_sha256"] != sha256_file(Path(__file__))):
        raise ValueError("registered source/runtime changed")
    original = Path(plan["original_bundle"])
    read_result(original)
    if sha256_file(original / "complete.json") != plan["original_receipt_sha256"]:
        raise ValueError("original pilot receipt changed")
    for case in plan["cases"]:
        for name, digest in case["input_sha256"].items():
            if sha256_file(original / name) != digest:
                raise ValueError("registered replay input changed")
    evidence = Path(plan["blank_evidence_path"])
    for name, digest in plan["blank_evidence_sha256"].items():
        if sha256_file(evidence.parent / name) != digest:
            raise ValueError("registered blank evidence changed")
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output / "plan.json", plan)
    write_json_new(output / "started.json", {"status": "STARTED", "plan_sha256": sha256_file(plan_path)})
    results = {}
    for case in plan["cases"]:
        identifier = case["id"]
        try:
            target = run_replay(original / case["decisions"], original / case["predictions"],
                                output / identifier, reference_path=original / case["reference"],
                                blank_evidence_path=evidence)
            receipt = read_replay(target)
            results[identifier] = {"status": "SETTLEMENT_ONLY_VERIFIED", "gross_points": receipt["gross_points"],
                "net_points": receipt["net_points"], "reference_reconciliation": receipt["reference_reconciliation"],
                "explicit_blank_outcomes": receipt["explicit_blank_outcomes"],
                "registration_identity": receipt["registration_identity"]}
        except (ValueError, KeyError, TypeError, OSError) as exc:
            results[identifier] = {"status": "NOT QUALIFIED", "reason": str(exc)}
        experiment.log_result("decision-replay-" + identifier,
            {"scope": "saved_pilot_settlement_qualification", "plan_sha256": sha256_file(plan_path),
             "inputs": case["input_sha256"]}, results[identifier], results_path=output / "registry.csv")
    if plan["source_sha256"] != _source_hashes() or plan["runtime"] != _runtime_versions():
        raise ValueError("source/runtime changed during registered replays")
    read_result(original)
    for name, digest in plan["blank_evidence_sha256"].items():
        if sha256_file(evidence.parent / name) != digest:
            raise ValueError("blank evidence changed during replay")
    result = {"schema_version": 1, "status": "QUALIFICATION_MATRIX_COMPLETED", "promotion_eligible": False,
              "fresh_holdout": False, "model_fits": 0, "optimizer_solves": 0, "cases": results,
              "preserved_bundle": str(output.resolve()),
              "original_receipt_sha256": plan["original_receipt_sha256"],
              "independent_financial_rule_legality": "NOT ASSESSED", "outcome_blind_execution": "NOT ASSESSED",
              "limitation": "Static structure and retrospective settlement only. Capture-time registration is not exact-deadline identity proof. Financial/rule legality, causal decision execution and promotion remain unassessed. Prior failed attempts are preserved."}
    result["output_sha256"] = {str(path.relative_to(output)): sha256_file(path)
                                for path in sorted(output.rglob("*")) if path.is_file()}
    write_json_new(output / "result.json", result)
    print(json.dumps(result["cases"], indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
