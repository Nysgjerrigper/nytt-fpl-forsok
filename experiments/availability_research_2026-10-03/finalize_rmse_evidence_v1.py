"""Finalize verified RMSE evidence without repeating training or optimization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path: Path) -> str:
    """Return the byte SHA256 of a preserved artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Require the completed pilot proof, then bind aggregate and timing reports."""
    directory = Path(__file__).resolve().parent
    repository = directory.parents[1]
    package_directory = repository / "experiments/research_priorities_2026-10-03"
    pilot_report_path = directory / "rmse_pilot_verification.json"
    report = json.loads(pilot_report_path.read_bytes())
    if report.get("status") != "verified":
        raise ValueError("Pilot must have completed independent verification")
    protocol_path = directory / "rmse_pilot_registration.json"
    protocol = json.loads(protocol_path.read_bytes())
    assert digest(protocol_path) == report["protocol_sha256"]
    for path, expected in report["input_sha256"].items():
        assert digest(Path(path)) == expected, path
    for path, expected in protocol["expected_code_sha256"].items():
        assert digest(repository / path) == expected, path
    study = directory / "rmse_seed0"
    assert digest(study / "summary.json") == report["summary_sha256"]
    assert digest(study / "verification.json") == report["verification_sha256"]
    summary = json.loads((study / "summary.json").read_bytes())
    assert summary["status"] in {"complete", "selection_rejected"}
    assert summary["status"] == report["pilot_status"]
    assert summary["advanced"] == report["advanced"]
    assert not summary["promotion_eligible"]
    package_path = package_directory / "package_with_rmse_availability_verification.json"
    timing_path = package_directory / "compute_timings_with_rmse_verified.json"
    output = directory / "rmse_evidence_finalization.json"
    if any(path.exists() for path in (package_path, timing_path, output)):
        raise ValueError("Preserve previous finalization reports; no overwrite")
    subprocess.run([sys.executable, "-B", str(package_directory / "verify_package.py"),
                    "--include-availability", "--include-rmse-availability"], check=True)
    package = json.loads(package_path.read_bytes())
    key = "availability_research_2026-10-03/rmse_seed0"
    assert key in package["studies"]
    assert package["studies"][key]["summary_sha256"] == report["summary_sha256"]
    subprocess.run([sys.executable, "-B", str(package_directory / "summarize_compute.py"),
                    "--package", str(package_path), "--output", str(timing_path)], check=True)
    result = {"status": "verified", "promotion_eligible": False,
              "pilot_verification_sha256": digest(pilot_report_path),
              "package_verification_sha256": digest(package_path),
              "compute_ledger_sha256": digest(timing_path),
              "finalizer_sha256": digest(Path(__file__)),
              "aggregate_optimal_solves": package["verified_optimal_solves"],
              "pilot_elapsed_seconds": summary["elapsed_seconds"],
              "comparisons": summary["comparisons"],
              "limitations": ["Repeatedly examined historical reference, not a fresh holdout.",
                              "Selected crawl-time information and reconstructed price/roster limitations remain.",
                              "No production change, promotion, commit, push or merge."]}
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
