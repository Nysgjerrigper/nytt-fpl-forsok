"""Run the fixed RMSE availability pilot and independently verify its evidence."""
import json
from pathlib import Path
import subprocess
import sys

from fpl import config
from fpl.model import availability_research, research_package


def main():
    root = Path(__file__).resolve().parent
    protocol_path = root / "rmse_pilot_registration.json"
    protocol = json.loads(protocol_path.read_bytes())
    output = root / "rmse_seed0"
    lock_path = root / "rmse_pilot_input_lock.json"
    if output.exists() or lock_path.exists():
        raise ValueError("Preserve previous RMSE pilot outputs; no overwrite")
    inputs = dict(protocol["input_sha256"])
    inputs[protocol["dataset"]] = protocol["dataset_sha256"]
    inputs[str(protocol_path.resolve())] = research_package.file_hash(protocol_path)
    inputs[str((root / "rmse_extension.patch").resolve())] = protocol["planned_extension_sha256"]
    inputs[str((root / "verify_loss_study.py").resolve())] = protocol["verification_script_sha256"]
    inputs[str(Path(__file__).resolve())] = research_package.file_hash(Path(__file__))
    lock_path.write_text(json.dumps({"input_sha256": inputs}, indent=2, sort_keys=True) + "\n")
    inputs[str(lock_path.resolve())] = research_package.file_hash(lock_path)

    def check():
        assert research_package.source_hashes() == protocol["expected_code_sha256"]
        for path, digest in inputs.items():
            assert research_package.file_hash(Path(path)) == digest, path

    check()
    summary = availability_research.run(protocol["dataset"], output,
                                       audit_dir=config.ROOT / "experiments/deadline_snapshot_audit_2026-10-03",
                                       seed=protocol["seed"], model_name=protocol["model_name"])
    check()
    registered = json.loads((output / "registration.json").read_bytes())
    for field in ("model_name", "seed", "parameters", "feature_variants", "windows", "dataset_sha256",
                  "input_sha256", "coverage_sha256", "coverage_mask_sha256", "roster_sha256", "origin_price_catalog_sha256"):
        assert registered[field] == protocol[field], field
    result = subprocess.run([sys.executable, "-B", str(root / "verify_loss_study.py"), str(output)],
                            check=True, text=True, capture_output=True)
    check()
    print(result.stdout, flush=True)
    report = {"status": "verified", "promotion_eligible": False,
              "protocol_sha256": research_package.file_hash(protocol_path), "input_sha256": inputs,
              "summary_sha256": research_package.file_hash(output / "summary.json"),
              "verification_sha256": research_package.file_hash(output / "verification.json"),
              "pilot_status": summary["status"], "advanced": summary["advanced"]}
    (root / "rmse_pilot_verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
