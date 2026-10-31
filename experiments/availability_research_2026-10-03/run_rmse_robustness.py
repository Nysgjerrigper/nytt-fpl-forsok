"""Execute registered RMSE availability refit seeds serially; stop on invalid evidence."""
import json
from pathlib import Path
import subprocess
import sys
import time

if not __debug__:
    raise RuntimeError("RMSE robustness requires normal Python with assertion checks enabled")

from fpl import config, experiment
from fpl.model import availability_research, research_package


def main():
    if not __debug__:
        raise RuntimeError("RMSE robustness requires normal Python")
    root = Path(__file__).resolve().parent
    parent_path = root / "rmse_robustness_registration.json"
    summary_path = root / "rmse_robustness_summary.json"
    lock_path = root / "rmse_robustness_input_lock.json"
    if summary_path.exists() or lock_path.exists():
        raise ValueError("Preserve existing RMSE robustness outputs; no resume or overwrite")
    for seed in (1, 2):
        if (root / f"rmse_seed{seed}").exists():
            raise ValueError(f"Preserve existing rmse_seed{seed}; no resume or overwrite")
    immutable = {}
    parent = {}
    summary = {"status": "running", "promotion_eligible": False, "seeds": {}}
    began = time.perf_counter()

    def bind(path, expected=None):
        path = Path(path).resolve()
        key = str(path)
        actual = research_package.file_hash(path)
        if expected is not None and actual != expected:
            raise ValueError(f"RMSE parent-bound artifact hash mismatch: {path}")
        if key in immutable and immutable[key] != actual:
            raise ValueError(f"RMSE preserved artifact changed: {path}")
        immutable[key] = actual
        return actual

    def check():
        if research_package.source_hashes() != parent["expected_code_sha256"]:
            raise ValueError("Current pipeline differs from RMSE parent registration")
        if experiment._library_versions() != parent["versions"]:
            raise ValueError("Current software versions differ from RMSE parent registration")
        for path, expected in immutable.items():
            if research_package.file_hash(Path(path)) != expected:
                raise ValueError(f"RMSE preserved artifact changed: {path}")

    def save():
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    try:
        parent_digest = bind(parent_path)
        parent = json.loads(parent_path.read_bytes())
        if parent.get("model_name") != "catboost_rmse" or parent.get("seeds") != [1, 2] or any(type(seed) is not int for seed in parent["seeds"]):
            raise ValueError("RMSE parent must register canonical model and ordered integer seeds 1, 2")
        for seed in parent["seeds"]:
            if parent["comparison_by_seed"][str(seed)] != {"block_len": 3, "n_boot": 10000, "seed": seed}:
                raise ValueError("Parent comparison must use block length 3, 10000 draws and the requested seed")
        bind(Path(__file__), parent["driver_sha256"])
        verifier = root / "verify_loss_study.py"
        bind(verifier, parent["verification_script_sha256"])
        seed0 = root / "rmse_seed0"
        for name in ("registration.json", "selection_lock.json", "summary.json", "verification.json"):
            bind(seed0 / name)
        bind(seed0 / "registration.json", parent["seed0_registration_sha256"])
        bind(seed0 / "selection_lock.json", parent["seed0_selection_lock_sha256"])
        previous = json.loads((seed0 / "summary.json").read_bytes())
        previous_registration = json.loads((seed0 / "registration.json").read_bytes())
        previous_report = json.loads((seed0 / "verification.json").read_bytes())
        if (previous.get("status") != "complete" or previous.get("advanced") is not True
                or previous_report.get("status") != "verified"
                or type(previous_registration.get("seed")) is not int
                or previous_registration["seed"] != 0
                or previous_registration.get("model_name") != "catboost_rmse"):
            raise ValueError("RMSE seed-zero source must be complete, advanced and independently verified")
        bind(seed0 / "summary.json", previous_report["summary_sha256"])
        if "registration_sha256" in previous_report:
            bind(seed0 / "registration.json", previous_report["registration_sha256"])
        for name, digest in previous["immutable_output_sha256"].items():
            path = Path(name)
            bind(path if path.is_absolute() else seed0 / path, digest)
        for relative, digest in previous_registration["code_sha256"].items():
            bind(seed0 / previous_registration["source_archive"] / relative, digest)
        for name, digest in parent["input_sha256"].items():
            path = Path(name)
            bind(path if path.is_absolute() else config.ROOT / path, digest)
        bind(parent["dataset"], parent["dataset_sha256"])
        check()
        with lock_path.open("x") as handle:
            json.dump({"parent_registration_sha256": parent_digest, "input_sha256": dict(immutable)}, handle, indent=2, sort_keys=True)
            handle.write("\n")
        lock_digest = bind(lock_path)
        summary.update(parent_registration_sha256=parent_digest, input_lock_sha256=lock_digest)
        check()
        for seed in parent["seeds"]:
            check()
            output = root / f"rmse_seed{seed}"
            study = availability_research.run(parent["dataset"], output,
                        audit_dir=config.ROOT / "experiments/deadline_snapshot_audit_2026-10-03",
                        seed=seed, model_name="catboost_rmse")
            check()
            registration_path = output / "registration.json"
            registration = json.loads(registration_path.read_bytes())
            if registration.get("seed") != seed or registration.get("model_name") != parent["model_name"]:
                raise ValueError("Child model/refit seed differs from parent")
            if registration["parameters"] != parent["parameters"][str(seed)]:
                raise ValueError("Child exact parameters differ from parent")
            for field in ("input_sha256", "dataset_sha256", "feature_variants", "windows", "coverage_sha256",
                          "coverage_mask_sha256", "roster_sha256", "origin_price_catalog_sha256", "versions",
                          "arms", "extra_player_features", "retrain_every", "horizon", "chips", "gap_rel", "solver",
                          "initial_budget", "scoring_protocol", "points_accounting", "origin_price_policy",
                          "standard_price_policy", "max_capture_age_hours", "selection_rule", "cutoffs"):
                if registration[field] != parent[field]:
                    raise ValueError(f"Child registration differs from parent: {field}")
            if registration["comparison"] != parent["comparison_by_seed"][str(seed)]:
                raise ValueError("Child comparison settings differ from parent")
            if registration["code_sha256"] != parent["expected_code_sha256"]:
                raise ValueError("Child source hashes differ from parent")
            bind(registration_path, study["registration_sha256"])
            for path, digest in study["immutable_output_sha256"].items():
                bind(path, digest)
            child_summary_path = output / "summary.json"
            child_summary_digest = bind(child_summary_path)
            if study["status"] not in ("complete", "selection_rejected"):
                raise ValueError("Child did not reach a valid terminal state")
            check()
            process = subprocess.run([sys.executable, "-B", str(verifier), str(output)],
                                     check=True, text=True, capture_output=True)
            print(process.stdout, flush=True)
            check()
            report_path = output / "verification.json"
            report = json.loads(report_path.read_bytes())
            if report.get("status") != "verified":
                raise ValueError("Child independent verification is not verified")
            bind(child_summary_path, report["summary_sha256"])
            if report["summary_sha256"] != child_summary_digest:
                raise ValueError("Child summary changed before independent verification")
            if "registration_sha256" in report:
                bind(registration_path, report["registration_sha256"])
            report_digest = bind(report_path)
            verified_summary = json.loads(child_summary_path.read_bytes())
            for path, digest in verified_summary["immutable_output_sha256"].items():
                bind(path, digest)
            check()
            summary["seeds"][str(seed)] = {
                "status": verified_summary["status"], "advanced": verified_summary["advanced"],
                "selection": {arm: value["total_points"] for arm, value in verified_summary["selection"].items()},
                "comparisons": verified_summary["comparisons"], "optimal_solves": report["optimal_solves"],
                "summary_sha256": child_summary_digest, "verification_sha256": report_digest}
            save()
            check()
        check()
        summary.update(status="complete", elapsed_seconds=time.perf_counter() - began,
                       immutable_input_output_sha256=dict(immutable), limitations=parent["limitations"])
        save()
        experiment.log_result("availability_rmse_seed_robustness_completed",
                              {"registration": str(parent_path), "summary_sha256": research_package.file_hash(summary_path)},
                              {"status": "COMPLETE", "promotion_eligible": False, "seeds": summary["seeds"]})
        check()
    except BaseException as exc:
        summary.update(status="incomplete", error=f"{type(exc).__name__}: {exc}",
                       elapsed_seconds=time.perf_counter() - began, immutable_input_output_sha256=dict(immutable))
        save()
        experiment.log_result("availability_rmse_seed_robustness_incomplete", {"registration": str(parent_path)},
                              {"status": "INCOMPLETE", "error": summary["error"], "promotion_eligible": False})
        raise


if __name__ == "__main__":
    main()
