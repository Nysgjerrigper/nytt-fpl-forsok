"""Bind completed scientific evidence without rerunning fits or optimizers."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path

from fpl import config
from fpl.model import research_package


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-availability', action='store_true')
    parser.add_argument('--include-rmse-availability', action='store_true')
    args = parser.parse_args()
    if args.include_rmse_availability and not args.include_availability:
        parser.error('--include-rmse-availability requires --include-availability')
    root = Path(__file__).resolve().parents[2]
    directories = [f"research_priorities_2026-10-03/selection_seed{s}" for s in (0, 1, 2)] + [
        "interval_calibration_2026-10-03/seed0", "cnn_research_2026-10-03/seed0",
        "temporal_blend_2026-10-03/seed0", "price_provenance_2026-10-03/seed0",
        "price_horizon_2026-10-03/seed0", "price_provenance_2026-10-03/reference_seed1",
        "price_provenance_2026-10-03/reference_seed2"]
    omitted = {}
    original_count = len(directories)
    if args.include_availability:
        for seed in (0, 1, 2):
            relative = f'availability_research_2026-10-03/seed{seed}'
            directory = root / 'experiments' / relative
            summary_path, report_path = directory / 'summary.json', directory / 'verification.json'
            if not summary_path.exists() or not report_path.exists():
                omitted[relative] = 'not yet complete and independently verified'
                continue
            summary = json.loads(summary_path.read_text())
            if summary['status'] not in {'complete', 'selection_rejected'}:
                omitted[relative] = summary['status']
                continue
            directories.append(relative)
    if args.include_rmse_availability:
        relative = 'availability_research_2026-10-03/rmse_seed0'
        directory = root / 'experiments' / relative
        summary_path, report_path = directory / 'summary.json', directory / 'verification.json'
        if not summary_path.exists() or not report_path.exists():
            omitted[relative] = 'not yet complete and independently verified'
        else:
            summary = json.loads(summary_path.read_text())
            report = json.loads(report_path.read_text())
            if summary.get('status') not in {'complete', 'selection_rejected'}:
                omitted[relative] = summary.get('status', 'missing summary status')
            elif str(report.get('status', '')).lower() not in {'pass', 'verified'}:
                omitted[relative] = 'independent verification is not pass/verified'
            else:
                directories.append(relative)
    studies = {}
    outputs = {}
    total_optimal = 0
    for index, relative in enumerate(directories):
        directory = root / "experiments" / relative
        summary_path, report_path = directory / "summary.json", directory / "verification.json"
        summary, report = json.loads(summary_path.read_text()), json.loads(report_path.read_text())
        assert report["status"].lower() in {"pass", "verified"}, relative
        assert summary["status"] in {"complete", "selection_rejected"}, relative
        assert not summary.get("promotion_eligible", False), relative
        expected = report.get("summary_sha256", report.get("summary_file_sha256"))
        assert expected == digest(summary_path), relative
        registration_path = directory / "registration.json"
        if "registration_sha256" in report:
            assert report["registration_sha256"] == digest(registration_path), relative
        for mapping in immutable_maps(summary):
            for path, sha in mapping.items():
                artifact = Path(path)
                if not artifact.is_absolute():
                    artifact = directory / artifact
                key = str(artifact.resolve())
                assert digest(artifact) == sha, artifact
                assert key not in outputs or outputs[key] == sha, artifact
                outputs[key] = sha
        optimal = report.get("optimal_solves", 0)
        total_optimal += optimal
        if index == original_count - 1:
            assert total_optimal == 1642
        studies[relative] = {"status": summary["status"], "summary_sha256": digest(summary_path),
                             "verification_sha256": digest(report_path), "optimal_solves": optimal}
    assert config.PRODUCTION_WEIGHT_STRATEGY == "single:catboost"
    audit = root / "experiments/external_source_audit_2026-10-03/audit.json"
    result = {"status": "verified", "promotion_eligible": False,
              "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "studies": studies, "verified_optimal_solves": total_optimal,
              "omitted_availability_studies": omitted,
              "rechecked_unique_immutable_outputs": len(outputs),
              "external_audit_sha256": digest(audit),
              "current_pipeline_source_sha256": research_package.source_hashes(),
              "production_strategy": config.PRODUCTION_WEIGHT_STRATEGY,
              "limitations": ["Per-study independent verification reports remain the evidence for numerical reconstruction.",
                              "Historical reference outcomes were repeatedly examined; no fresh holdout.",
                              "Refit seeds share outcomes and fixed seed-zero tuning; no independent-sample pooling.",
                              "Historical deadline prices, registered rosters and external publication times remain incompletely verified.",
                              "Cold-start priors improve registered-player coverage; no established realized-point gain.",
                              "No model promotion, production change, commit, push or merge."]}
    if args.include_rmse_availability:
        name = 'package_with_rmse_availability_verification.json'
    else:
        name = 'package_with_availability_verification.json' if args.include_availability else 'package_verification.json'
    output = Path(__file__).resolve().parent / name
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "verified_optimal_solves", "rechecked_unique_immutable_outputs")}))


if __name__ == "__main__":
    main()
