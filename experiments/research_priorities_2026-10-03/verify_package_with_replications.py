"""Bind all sixteen completed studies and RMSE replication proof offline.

No optional study omissions, repeated fits, optimizers or changes to sealed reports.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fpl import config


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_maps(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "immutable_output_sha256":
                yield child
            else:
                yield from immutable_maps(child)
    elif isinstance(value, list):
        for child in value:
            yield from immutable_maps(child)


def main():
    output = Path(__file__).resolve().parent / "package_with_availability_replications_verification.json"
    if output.exists():
        raise ValueError("Preserve previous aggregate verification; no overwrite")
    directories = [f"research_priorities_2026-10-03/selection_seed{seed}" for seed in (0, 1, 2)] + [
        "interval_calibration_2026-10-03/seed0", "cnn_research_2026-10-03/seed0",
        "temporal_blend_2026-10-03/seed0", "price_provenance_2026-10-03/seed0",
        "price_horizon_2026-10-03/seed0", "price_provenance_2026-10-03/reference_seed1",
        "price_provenance_2026-10-03/reference_seed2"] + [
        f"availability_research_2026-10-03/seed{seed}" for seed in (0, 1, 2)] + [
        f"availability_research_2026-10-03/rmse_seed{seed}" for seed in (0, 1, 2)]
    studies, outputs, bound = {}, {}, {}

    def require(condition, label):
        if not condition:
            raise ValueError(label)

    def bind(path, expected=None):
        path = Path(path).resolve()
        actual = digest(path)
        require(expected is None or expected == actual, f"SHA mismatch: {path}")
        require(str(path) not in bound or bound[str(path)] == actual, f"Artifact changed: {path}")
        bound[str(path)] = actual
        return actual

    parent_report_path = ROOT / "experiments/availability_research_2026-10-03/rmse_robustness_verification.json"
    parent_report_sha = bind(parent_report_path)
    parent_report = json.loads(parent_report_path.read_bytes())
    require(parent_report["status"] == "verified" and parent_report["promotion_eligible"] is False
            and parent_report["optimal_solves"] == 64 and parent_report["rmse_family_optimal_solves"] == 220,
            "RMSE replication verification differs")
    for path, expected in parent_report["rechecked_input_output_sha256"].items():
        bind(path, expected)
    total_optimal = 0
    for index, relative in enumerate(directories):
        directory = ROOT / "experiments" / relative
        summary_path, report_path = directory / "summary.json", directory / "verification.json"
        summary_sha, report_sha = bind(summary_path), bind(report_path)
        summary, report = json.loads(summary_path.read_bytes()), json.loads(report_path.read_bytes())
        require(str(report["status"]).lower() in {"pass", "verified"}, f"Unverified study: {relative}")
        require(summary["status"] in {"complete", "selection_rejected"}
                and not summary.get("promotion_eligible", False), f"Incomplete/promoted study: {relative}")
        require(report.get("summary_sha256", report.get("summary_file_sha256")) == summary_sha,
                f"Summary proof differs: {relative}")
        registration_path = directory / "registration.json"
        if "registration_sha256" in report:
            bind(registration_path, report["registration_sha256"])
        for mapping in immutable_maps(summary):
            for path, expected in mapping.items():
                artifact = Path(path)
                artifact = artifact if artifact.is_absolute() else directory / artifact
                bind(artifact, expected)
                key = str(artifact.resolve())
                require(key not in outputs or outputs[key] == expected, f"Conflicting immutable output: {key}")
                outputs[key] = expected
        optimal = report.get("optimal_solves", 0)
        require(type(optimal) is int and optimal >= 0, f"Invalid optimal solve count: {relative}")
        total_optimal += optimal
        if index == 9:
            require(total_optimal == 1642, "Original ten-study count differs")
        studies[relative] = {"status": summary["status"], "summary_sha256": summary_sha,
                             "verification_sha256": report_sha, "optimal_solves": optimal}
    require(len(studies) == 16 and total_optimal == 2082, "Complete replication package count differs")
    require(config.PRODUCTION_WEIGHT_STRATEGY == "single:catboost", "Production strategy differs")
    audit = ROOT / "experiments/external_source_audit_2026-10-03/audit.json"
    audit_sha = bind(audit)
    for path, expected in bound.items():
        require(digest(path) == expected, f"Aggregate verification-time drift: {path}")
    result = {"status": "verified", "promotion_eligible": False,
              "verified_at_utc": datetime.now(timezone.utc).isoformat(), "studies": studies,
              "verified_optimal_solves": total_optimal, "omitted_availability_studies": {},
              "rechecked_unique_immutable_outputs": len(outputs), "external_audit_sha256": audit_sha,
              "rmse_robustness_verification_sha256": parent_report_sha,
              "aggregate_verifier_sha256": digest(Path(__file__)),
              "current_pipeline_source_sha256": {str(path.relative_to(ROOT)): digest(path)
                                                 for path in sorted((ROOT / "fpl").rglob("*.py"))},
              "production_strategy": config.PRODUCTION_WEIGHT_STRATEGY,
              "limitations": ["Per-study independent reports remain evidence for numerical reconstruction.",
                              "Historical reference outcomes repeatedly examined; no fresh holdout.",
                              "Refit seeds share outcomes and fixed seed-zero tuning; no independent-sample pooling.",
                              "Historical prices, registered rosters and external publication times remain incompletely verified.",
                              "Current source hashes describe subsequent state; archived sources bind completed studies.",
                              "No model promotion, production change, commit, push or merge."]}
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "verified_optimal_solves", "rechecked_unique_immutable_outputs")}))


if __name__ == "__main__":
    main()
