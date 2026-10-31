"""Registered coverage check for existing priors, without scoring target outcomes."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from fpl import config, experiment
from fpl.data.deadline_snapshots import load_deadline_rosters, SnapshotUnavailableError
from fpl.data.provenance import load_frozen_research_dataset
from fpl.model.cold_start import fit_pooled_priors, predict_cold_starts
from fpl.model import research_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.FROZEN_RESEARCH_DATASET_PATH)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    registration_path = root / "registered_priors_registration.json"
    output_path = root / "registered_priors_audit.json"
    if registration_path.exists() or output_path.exists():
        raise ValueError("Preserve the completed coverage audit; refuse overwrite")
    proof = root / "wayback_verification.json"
    sources = research_package.source_hashes()
    registration = {"status": "registered", "promotion_eligible": False,
                    "hypothesis": "Existing pooled priors cover registered codes absent from the active history snapshot.",
                    "windows": {"selection": [137, 152], "reference": [153, 183]},
                    "prior_strength": config.COLD_START_PRIOR_STRENGTH,
                    "dataset": str(args.dataset.resolve()), "dataset_sha256": research_package.file_hash(args.dataset),
                    "proof_sha256": research_package.file_hash(proof), "code_sha256": sources,
                    "script_sha256": research_package.file_hash(Path(__file__)),
                    "admission_verification_sha256": research_package.file_hash(root / "admission_verification.json"),
                    "decision_evaluation": False, "target_outcomes_read": False,
                    "limitations": ["Coverage at selected crawl time, not exact deadline or player eligibility.",
                                    "Prior distributions are empirical and uncalibrated; no accuracy or MILP gain assessment.",
                                    "Repeated players across origins are not independent observations."]}
    registration_path.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
    registration_digest = research_package.file_hash(registration_path)
    experiment.log_result("registered_prior_coverage_registration", registration, {"status": "REGISTERED", "promotion_eligible": False})
    raw = load_frozen_research_dataset(args.dataset)
    raw = raw.loc[raw.GW_global <= 183].copy()
    results, exports = [], []
    fit_seconds = 0.
    for origin in range(137, 184):
        try:
            roster = load_deadline_rosters(proof, root / "inventory/wayback_index.json", registration["proof_sha256"], [origin])
        except SnapshotUnavailableError as exc:
            results.append({"origin": origin, "status": "missing", "reason": exc.reason})
            continue
        past = raw.loc[raw.GW_global < origin]
        latest = past.groupby("player_id").GW_global.max()
        active = latest.index[latest > past.GW_global.max() - config.GWS_PER_SEASON]
        absent = roster.loc[~roster.player_id.isin(active)].reset_index(drop=True)
        began = time.perf_counter()
        priors = fit_pooled_priors(past, origin)
        predicted = predict_cold_starts(absent[["position", "team"]], priors)
        fit_seconds += time.perf_counter() - began
        assert np.isfinite(predicted.to_numpy(dtype=float)).all()
        assert predicted.prior_std.ge(0).all()
        assert predicted.p_played.between(0, 1).all() and predicted.p60.between(0, 1).all()
        assert (predicted.p60 <= predicted.p_played).all()
        exported = pd.concat([absent[["player_id", "origin_gw", "position", "team", "status", "value"]], predicted], axis=1)
        exports.append(exported)
        results.append({"origin": origin, "status": "admitted", "registered_players": len(roster),
                        "covered_historical_codes": len(roster) - len(absent),
                        "pooled_prior_codes": len(absent), "registered_codes_after_fallback": len(roster),
                        "prior_mean_min": float(predicted.predicted_total_points.min()) if len(absent) else None,
                        "prior_mean_max": float(predicted.predicted_total_points.max()) if len(absent) else None})
    for path, expected in [(proof, registration["proof_sha256"]), (args.dataset, registration["dataset_sha256"]),
                           (registration_path, registration_digest), (Path(__file__), registration["script_sha256"]),
                           (root / "admission_verification.json", registration["admission_verification_sha256"])]:
        assert research_package.file_hash(path) == expected
    assert research_package.source_hashes() == sources
    export_path = root / "registered_priors.csv"
    exported = pd.concat(exports, ignore_index=True)
    exported.to_csv(export_path, index=False)
    result = {"status": "complete", "promotion_eligible": False, "registration_sha256": registration_digest,
              "origins": results, "prior_fit_seconds": fit_seconds,
              "covered_player_origin_rows": len(exported), "unique_covered_codes": int(exported.player_id.nunique()),
              "export_sha256": research_package.file_hash(export_path),
              "limitations": registration["limitations"]}
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    experiment.log_result("registered_prior_coverage_completed", {"registration": str(registration_path),
                          "summary_sha256": research_package.file_hash(output_path)},
                          {"status": "COMPLETE", "promotion_eligible": False,
                           "covered_player_origin_rows": len(exported), "prior_fit_seconds": fit_seconds})
    print(json.dumps({key: result[key] for key in ("status", "covered_player_origin_rows", "unique_covered_codes", "prior_fit_seconds")}))


if __name__ == "__main__":
    main()
