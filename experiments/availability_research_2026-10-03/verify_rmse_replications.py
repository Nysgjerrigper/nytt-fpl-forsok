"""Offline byte/protocol verification of completed RMSE refit replications.

Per-seed independent reports supply numerical reconstruction. This script checks
their preserved bytes and registered protocol without refitting or solving.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    repository = directory.parents[1]
    output = directory / "rmse_robustness_verification.json"
    if output.exists():
        raise ValueError("Preserve existing replication verification; no overwrite")
    bound = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"SHA256 mismatch: {path}")
        if str(path) in bound and bound[str(path)] != actual:
            raise ValueError(f"Artifact changed: {path}")
        bound[str(path)] = actual
        return actual

    def require(condition, label):
        if not condition:
            raise ValueError(label)

    parent_path = directory / "rmse_robustness_registration.json"
    summary_path = directory / "rmse_robustness_summary.json"
    bind(parent_path)
    bind(summary_path)
    parent = json.loads(parent_path.read_bytes())
    summary = json.loads(summary_path.read_bytes())
    require(parent["model_name"] == "catboost_rmse" and parent["seeds"] == [1, 2]
            and all(type(seed) is int for seed in parent["seeds"]), "Invalid parent model/seeds")
    require(summary["status"] == "complete" and summary["promotion_eligible"] is False, "Parent incomplete/promoted")
    bind(parent_path, summary["parent_registration_sha256"])
    lock_path = directory / "rmse_robustness_input_lock.json"
    bind(lock_path, summary["input_lock_sha256"])
    lock = json.loads(lock_path.read_bytes())
    require(lock["parent_registration_sha256"] == digest(parent_path), "Input lock parent differs")
    for mapping in (lock["input_sha256"], summary["immutable_input_output_sha256"]):
        for path, expected in mapping.items():
            bind(path, expected)
    for path, expected in parent["input_sha256"].items():
        path = Path(path)
        bind(path if path.is_absolute() else repository / path, expected)
    for path, expected in parent["prior_evidence_sha256"].items():
        bind(directory / path, expected)
    bind(parent["dataset"], parent["dataset_sha256"])
    bind(directory / "run_rmse_robustness.py", parent["driver_sha256"])
    bind(directory / "verify_loss_study.py", parent["verification_script_sha256"])
    fields = ("input_sha256", "dataset_sha256", "feature_variants", "windows", "coverage_sha256",
              "coverage_mask_sha256", "roster_sha256", "origin_price_catalog_sha256", "versions",
              "arms", "extra_player_features", "retrain_every", "horizon", "chips", "gap_rel", "solver",
              "initial_budget", "scoring_protocol", "points_accounting", "origin_price_policy",
              "standard_price_policy", "max_capture_age_hours", "selection_rule", "cutoffs")
    children = {}
    for seed in (0, 1, 2):
        child = directory / f"rmse_seed{seed}"
        registration_path = child / "registration.json"
        child_summary_path, report_path = child / "summary.json", child / "verification.json"
        registration_sha = bind(registration_path)
        summary_sha, report_sha = bind(child_summary_path), bind(report_path)
        registration = json.loads(registration_path.read_bytes())
        study, report = json.loads(child_summary_path.read_bytes()), json.loads(report_path.read_bytes())
        require(type(registration["seed"]) is int and registration["seed"] == seed
                and registration["model_name"] == "catboost_rmse", f"Child {seed} model/seed differs")
        require(report["status"] == "verified" and report["promotion_eligible"] is False
                and study["promotion_eligible"] is False, f"Child {seed} unverified/promoted")
        bind(registration_path, report["registration_sha256"])
        bind(registration_path, study["registration_sha256"])
        bind(child_summary_path, report["summary_sha256"])
        for path, expected in study["immutable_output_sha256"].items():
            path = Path(path)
            bind(path if path.is_absolute() else child / path, expected)
        for relative, expected in registration["code_sha256"].items():
            bind(child / registration["source_archive"] / relative, expected)
        if seed:
            for field in fields:
                require(registration[field] == parent[field], f"Child {seed} registration differs: {field}")
            require(registration["parameters"] == parent["parameters"][str(seed)], "Child exact parameters differ")
            require(registration["code_sha256"] == parent["expected_code_sha256"], "Child archived code differs")
            require(registration["comparison"] == parent["comparison_by_seed"][str(seed)], "Child comparison differs")
            recorded = summary["seeds"][str(seed)]
            require(recorded["summary_sha256"] == summary_sha and recorded["verification_sha256"] == report_sha,
                    "Parent child hashes differ")
            require(recorded["comparisons"] == study["comparisons"], "Parent child comparisons differ")
            require(recorded["selection"] == {arm: result["total_points"] for arm, result in study["selection"].items()},
                    "Parent child selection totals differ")
        else:
            bind(registration_path, parent["seed0_registration_sha256"])
            bind(child / "selection_lock.json", parent["seed0_selection_lock_sha256"])
        control, candidate = registration["arms"]
        points = [study["selection"][arm]["total_points"] for arm in (control, candidate)]
        require(all(math.isfinite(point) for point in points), "Nonfinite selection points")
        advanced = points[1] > points[0]
        require(study["advanced"] is advanced, "Strict selection gate differs")
        gate = json.loads((child / "selection_lock.json").read_bytes())
        bind(child / "selection_lock.json")
        bind(child / "selection_summary.json", gate["selection_summary_sha256"])
        require(gate["advanced"] is advanced and gate["parameters"] == registration["parameters"]
                and gate["feature_variants"] == registration["feature_variants"], "Finalist lock differs")
        require(study["status"] == ("complete" if advanced else "selection_rejected"), "Child terminal status differs")
        if not advanced:
            require(not study["evaluation"] and not gate["finalists"] and not (child / "reference").exists(),
                    "Rejected candidate has reference evaluation")
        require(report["optimal_solves"] == (156 if advanced else 32), "Child optimal count differs")
        if seed:
            require(not advanced and summary["seeds"][str(seed)]["optimal_solves"] == 32
                    and summary["seeds"][str(seed)]["status"] == study["status"]
                    and summary["seeds"][str(seed)]["advanced"] is advanced, "Replication status/count differs")
        children[str(seed)] = {"status": study["status"], "advanced": advanced, "optimal_solves": report["optimal_solves"],
                               "summary_sha256": summary_sha, "verification_sha256": report_sha,
                               "registration_sha256": registration_sha, "comparisons": study["comparisons"]}
    require(sum(value["optimal_solves"] for value in children.values()) == 220, "RMSE family total differs")
    for path, expected in bound.items():
        require(digest(path) == expected, f"Verification-time drift: {path}")
    result = {"status": "verified", "promotion_eligible": False, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "parent_registration_sha256": digest(parent_path), "parent_summary_sha256": digest(summary_path),
              "verifier_sha256": digest(Path(__file__)), "optimal_solves": 64, "rmse_family_optimal_solves": 220,
              "seeds": children, "rechecked_input_output_sha256": bound,
              "limitations": ["Per-seed reports supply independent numerical reconstruction.",
                              "Refit seeds share outcomes and seed-zero tuning; no independent-sample pooling.",
                              "Archived sources checked; current pipeline may have subsequent research additions.",
                              "No production promotion or fresh holdout claim."]}
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "optimal_solves", "rmse_family_optimal_solves")}))


if __name__ == "__main__":
    main()
