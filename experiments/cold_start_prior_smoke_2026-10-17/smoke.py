"""Execute a pre-registered, two-computation engineering check; no model selection."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

from fpl import config, experiment
from fpl.artifacts import sha256_file, write_json_new
from fpl.data.provenance import load_frozen_research_dataset
from fpl.model.cold_start import fit_pooled_priors, predict_cold_starts, validate_pooled_priors, PRIOR_COLUMNS
from fpl.research_runner import _source_hashes, _runtime_versions


def main(plan_path: Path, output: Path) -> None:
    """Verify pinned sources, compute priors twice, and preserve explicit limitations."""
    plan = json.loads(plan_path.read_bytes())
    if (plan["source_sha256"] != _source_hashes() or plan["runtime"] != _runtime_versions()
            or plan["script_sha256"] != sha256_file(Path(__file__))
            or sha256_file(plan["dataset"]) != plan["dataset_sha256"]):
        raise ValueError("registered inputs or runtime changed")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        write_json_new(output / "plan.json", plan)
        write_json_new(output / "started.json", {"status": "STARTED", "plan_sha256": sha256_file(plan_path)})
        raw = load_frozen_research_dataset(plan["dataset"])
        origin, strength = plan["origin_gw"], plan["prior_strength"]
        priors = fit_pooled_priors(raw, origin, strength)
        changed = raw.copy()
        changed["player_id"] = changed.player_id.astype(object)
        future = changed.GW_global >= origin
        changed.loc[future, "player_id"] = "excluded future identity"
        changed.loc[future, "total_points"] = np.nan
        changed.loc[future, "minutes"] = -999.
        perturbed = fit_pooled_priors(changed, origin, strength)
        pd.testing.assert_frame_equal(priors, perturbed)
        assert priors.attrs == perturbed.attrs
        payload = {"metadata": priors.attrs, "rows": priors.to_dict("records")}
        write_json_new(output / "priors.json", payload)
        restored_payload = json.loads((output / "priors.json").read_bytes())
        restored = pd.DataFrame(restored_payload["rows"], columns=PRIOR_COLUMNS)
        restored.attrs = restored_payload["metadata"]
        validate_pooled_priors(restored, origin)
        # Fictitious team probes check pooled fallback, not player eligibility or performance.
        probe = pd.DataFrame({"position": config.ONFIELD_POSITIONS, "team": "__UNSEEN_TEAM__"})
        predicted = predict_cold_starts(probe, restored, origin_gw=origin)
        pd.testing.assert_frame_equal(predicted, predict_cold_starts(
            probe.assign(total_points=np.inf, minutes=-999), restored, origin_gw=origin))
        write_json_new(output / "probe-predictions.json", {"purpose": "synthetic fallback probes", "rows": predicted.to_dict("records")})
        if plan["source_sha256"] != _source_hashes() or plan["runtime"] != _runtime_versions():
            raise ValueError("source or runtime changed during smoke")
        result = {"status": "ENGINEERING_SMOKE_PASS", "promotion_eligible": False,
                  "fresh_holdout": False, "prior_computations": 2, "gbm_fits": 0, "milp_solves": 0,
                  "origin_gw": origin, "history_max_gw": priors.attrs["history_max_gw"],
                  "history_rows": int((raw.GW_global < origin).sum()), "prior_rows": len(priors),
                  "positions": sorted(priors.position.unique()), "seconds": time.monotonic() - started,
                  "limitations": "Engineering cutoff/serialization check only; synthetic probes are not eligible players. Empirical spreads/frequencies are uncalibrated. Forecast benefit and downstream FPL points NOT ASSESSED."}
        experiment.log_result("cold-start-prior-engineering-smoke", {"plan_sha256": sha256_file(plan_path),
            "scope": "framework_validation", "origin_gw": origin, "prior_strength": strength}, result,
            results_path=output / "registry.csv")
        result["output_sha256"] = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
        write_json_new(output / "result.json", result)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        write_json_new(output / "failed.json", {"status": "FAILED", "reason": str(exc), "promotion_eligible": False})
        raise


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
