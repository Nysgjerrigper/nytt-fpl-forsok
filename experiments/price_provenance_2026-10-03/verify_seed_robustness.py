"""Independently verify the completed, preregistered cached price repetitions."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from fpl import experiment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parent = Path(__file__).resolve().parent
    protocol_path = parent / "seed_robustness_protocol.json"
    summary_path = parent / "seed_robustness_summary.json"
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "complete" and not summary["promotion_eligible"]
    assert summary["protocol_sha256"] == digest(protocol_path)
    assert protocol["launcher_sha256"] == digest(parent / "seed_robustness_launcher.py")
    assert protocol["seeds"] == [1, 2]
    assert protocol["reference_window"] == [153, 183]
    assert protocol["new_model_fits"] == 0
    assert protocol["reference_only"] and protocol["include_current_proxy"] and protocol["skip_lagged_proxy"]
    assert protocol["policies"] == ["target-replay", "origin-current-proxy"]
    assert protocol["dataset_sha256"] == digest(protocol["dataset"])
    sealed = {**protocol["cached_input_sha256"], **protocol["source_archive_sha256"],
              **summary["immutable_output_sha256"]}
    for path, expected in sealed.items():
        assert digest(path) == expected, path
    total_optimal = total_budgets = 0
    reports = {}
    for seed in protocol["seeds"]:
        recorded = summary["seeds"][str(seed)]
        child = Path(recorded["output_dir"])
        result = json.loads((child / "summary.json").read_text())
        verification = json.loads((child / "verification.json").read_text())
        registration = json.loads((child / "registration.json").read_text())
        assert recorded["summary_sha256"] == digest(child / "summary.json") == verification["summary_sha256"]
        assert verification["status"] == "verified" and not verification["promotion_eligible"]
        assert result["status"] == "complete" and registration["seed"] == seed
        assert recorded["comparisons"] == result["comparisons"]["reference"]
        points = {name: entry["total_points"] for name, entry in result["phases"]["reference"].items()}
        assert recorded["results"] == points == verification["results"]["reference"]
        assert set(points) == {"mae_replay", "mae_current", "rmse_replay", "rmse_current"}
        assert verification["optimal_solves"] == verification["independently_reconstructed_current_budgets"] == 124
        total_optimal += verification["optimal_solves"]
        total_budgets += verification["independently_reconstructed_current_budgets"]
        reports[str(seed)] = {"summary_sha256": recorded["summary_sha256"],
                              "verification_sha256": digest(child / "verification.json"), "points": points}
    assert total_optimal == protocol["expected_optimal_solves"] == 248
    report = {"status": "verified", "promotion_eligible": False,
              "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "summary_sha256": digest(summary_path), "protocol_sha256": digest(protocol_path),
              "sealed_files": len(sealed), "source_files": len(protocol["code_sha256"]),
              "optimal_solves": total_optimal, "independently_reconstructed_current_budgets": total_budgets,
              "elapsed_seconds": summary["elapsed_seconds"], "seeds": reports,
              "limitations": protocol["limitations"]}
    (parent / "seed_robustness_verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    runtime_path = parent / "seed_robustness_runtime.json"
    runtime = json.loads(runtime_path.read_text())
    runtime.update(classification="complete; owned exec session 1453 returned exit code 0; no active jobs",
                   exit_code=0, classified_at_utc=datetime.now(timezone.utc).isoformat())
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    experiment.log_result("price_seed_robustness_verification", {"summary_sha256": digest(summary_path)},
                          {"status": "VERIFIED", "promotion_eligible": False, "optimal_solves": total_optimal})
    print(json.dumps({key: report[key] for key in ("status", "sealed_files", "optimal_solves", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
