"""Execute the predeclared serial availability seeds; stop on any invalid evidence."""
import json
from pathlib import Path
import subprocess
import sys
import time

from fpl import config, experiment
from fpl.model import availability_research, research_package


def main():
    root = Path(__file__).resolve().parent
    parent_path = root / "robustness_registration.json"
    parent = json.loads(parent_path.read_bytes())
    parent_digest = research_package.file_hash(parent_path)
    summary_path = root / "robustness_summary.json"
    lock_path = root / "robustness_input_lock.json"
    if summary_path.exists() or lock_path.exists():
        raise ValueError("Preserve existing robustness outputs; no resume or overwrite")
    seed0 = root / "seed0"
    previous = json.loads((seed0 / "summary.json").read_bytes())
    verification = json.loads((seed0 / "verification.json").read_bytes())
    assert previous["status"] == "complete" and previous["advanced"]
    assert verification["status"] == "verified"
    assert verification["summary_sha256"] == research_package.file_hash(seed0 / "summary.json")
    immutable = {str(parent_path.resolve()): parent_digest}
    for name in ("registration.json", "selection_lock.json", "summary.json", "verification.json"):
        path = seed0 / name
        immutable[str(path.resolve())] = research_package.file_hash(path)
    assert immutable[str((seed0 / "registration.json").resolve())] == parent["seed0_registration_sha256"]
    assert immutable[str((seed0 / "selection_lock.json").resolve())] == parent["seed0_selection_lock_sha256"]
    immutable.update(parent["input_sha256"])
    immutable[parent["dataset"]] = parent["dataset_sha256"]
    patch = root / "robustness_extension.patch"
    assert research_package.file_hash(patch) == parent["planned_extension_sha256"]
    immutable[str(patch.resolve())] = parent["planned_extension_sha256"]
    lock_path.write_text(json.dumps({"parent_registration_sha256": parent_digest, "input_sha256": immutable}, indent=2, sort_keys=True) + "\n")
    immutable[str(lock_path.resolve())] = research_package.file_hash(lock_path)
    summary = {"status": "running", "promotion_eligible": False, "seeds": {},
               "parent_registration_sha256": parent_digest, "input_lock_sha256": immutable[str(lock_path.resolve())]}
    began = time.perf_counter()

    def check():
        assert research_package.source_hashes() == parent["expected_code_sha256"], "Current sources differ from predeclared extension"
        for path, digest in immutable.items():
            assert research_package.file_hash(Path(path)) == digest, path

    try:
        check()
        for seed in parent["seeds"]:
            output = root / f"seed{seed}"
            study = availability_research.run(parent["dataset"], output,
                                              audit_dir=config.ROOT / "experiments/deadline_snapshot_audit_2026-10-03", seed=seed)
            check()
            registered = json.loads((output / "registration.json").read_bytes())
            assert registered["parameters"] == parent["parameters"][str(seed)]
            for field in ("input_sha256", "dataset_sha256", "feature_variants", "windows", "coverage_sha256",
                          "coverage_mask_sha256", "roster_sha256", "origin_price_catalog_sha256"):
                assert registered[field] == parent[field], field
            result = subprocess.run([sys.executable, "-B", str(root / "verify_study.py"), str(output)],
                                    check=True, text=True, capture_output=True)
            print(result.stdout, flush=True)
            report = json.loads((output / "verification.json").read_bytes())
            summary["seeds"][str(seed)] = {"status": study["status"], "advanced": study["advanced"],
                                           "selection": {arm: value["total_points"] for arm, value in study["selection"].items()},
                                           "comparisons": study["comparisons"], "optimal_solves": report["optimal_solves"],
                                           "summary_sha256": research_package.file_hash(output / "summary.json"),
                                           "verification_sha256": research_package.file_hash(output / "verification.json")}
            immutable[str((output / "summary.json").resolve())] = research_package.file_hash(output / "summary.json")
            immutable[str((output / "verification.json").resolve())] = research_package.file_hash(output / "verification.json")
            immutable.update(study["immutable_output_sha256"])
            check()
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        check()
        summary.update(status="complete", elapsed_seconds=time.perf_counter() - began,
                       immutable_input_output_sha256=immutable,
                       limitations=parent["limitations"])
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        experiment.log_result("availability_seed_robustness_completed", {"registration": str(parent_path),
                              "summary_sha256": research_package.file_hash(summary_path)},
                              {"status": "COMPLETE", "promotion_eligible": False, "seeds": summary["seeds"]})
    except BaseException as exc:
        summary.update(status="incomplete", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter() - began)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        experiment.log_result("availability_seed_robustness_incomplete", {"registration": str(parent_path)},
                              {"status": "INCOMPLETE", "error": summary["error"], "promotion_eligible": False})
        raise


if __name__ == "__main__":
    main()
