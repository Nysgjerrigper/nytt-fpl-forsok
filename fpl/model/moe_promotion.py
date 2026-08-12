"""Pre-registered promotion decision for the position-specialist MoE.

This module deliberately operates only on finished MILP squad-selection CSVs.
Forecast screening and model selection happen elsewhere; by the time a candidate
reaches this gate, the only question is whether its realized decisions beat the
standing CatBoost control under both standard and deploy-honest protocols.

Manifest format::

    {
      "control": {"standard": "...csv", "origin": "...csv"},
      "candidates": {
        "position_map": {
          "standard": "...csv", "origin": "...csv",
          "seed_diffs": [12, 8, 15], "complexity": 0
        },
        "mid_gate": {
          "standard": "...csv", "origin": "...csv",
          "seed_diffs": [18, 4, 11], "complexity": 1
        }
      }
    }

Only the two pre-registered final candidates belong here.  One-position
ablations are diagnostic and must not be added to the multiple-testing family.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from fpl.experiment import log_result
from fpl.milp.compare_backtests import compare, holm_bonferroni, promotion_gate
from fpl.model.moe_tournament import artifact_hash, seed_differences_from_artifacts, SEEDS, protocol_cutoffs, spent_window


def _headline_paths(finalists_path: str, seed: int) -> tuple[str, str, str, str]:
    """Resolve and revalidate frozen candidate/control headline files for one seed."""
    payload = json.loads(Path(finalists_path).read_text())
    if payload.get("artifact_sha256") != artifact_hash({k: v for k, v in payload.items() if k != "artifact_sha256"}):
        raise ValueError("frozen finalists manifest hash mismatch")
    if seed not in SEEDS or str(seed) not in payload.get("seeds", {}) or str(seed) not in payload.get("controls", {}):
        raise ValueError("unregistered headline seed")
    result = []
    expected = set(range(protocol_cutoffs()["backtest_min_gw"], protocol_cutoffs()["backtest_max_gw"] + 1))
    low, high = spent_window()
    for group in ("seeds", "controls"):
        for protocol in ("standard", "origin"):
            record = payload[group][str(seed)][protocol]
            path = Path(record["path"])
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("frozen headline artifact bytes changed")
            frame = pd.read_csv(path)
            gw = "gameweek" if "gameweek" in frame else "GW"
            if gw not in frame or set(frame[gw]) != expected or frame[gw].duplicated().any() or frame[gw].between(low, high).any():
                raise ValueError("frozen headline artifact has invalid gameweek lineage")
            result.append(str(path))
    return result[0], result[1], result[2], result[3]


def validate_promotion_family(manifest: dict) -> None:
    """Require a hash-bound registered finalist family before Holm is applied."""
    candidates = manifest.get("candidates", {})
    names = set(candidates)
    if names not in ({"position_map"}, {"position_map", "mid_gate"}):
        raise ValueError("Holm family must be exactly {position_map} or {position_map, mid_gate}")
    frozen = manifest.get("frozen_finalists")
    if not frozen:
        raise ValueError("promotion manifest must bind candidates to frozen finalists")
    frozen_payload = json.loads(Path(frozen).read_text())
    if frozen_payload.get("artifact_sha256") != artifact_hash({k: v for k, v in frozen_payload.items() if k != "artifact_sha256"}):
        raise ValueError("frozen finalists have no artifact hash")
    for name, spec in candidates.items():
        if spec.get("finalists_sha256") != frozen_payload["artifact_sha256"]:
            raise ValueError(f"candidate {name!r} is not bound to frozen finalist artifacts")


def assess_manifest(manifest: dict, alpha: float = 0.05, n_boot: int = 10000,
                    block_len: int = 3, seed: int = 0) -> dict:
    validate_promotion_family(manifest)
    candidates = manifest.get("candidates", {})
    if not 1 <= len(candidates) <= 2:
        raise ValueError("promotion family must contain one or two final candidates")

    reports = {}
    one_sided_p = []
    names = list(candidates)
    for name in names:
        spec = candidates[name]
        required = {"finalists", "headline_seed"}
        missing = required - set(spec)
        if missing:
            raise ValueError(f"candidate {name!r} missing {sorted(missing)}")
        candidate_standard, candidate_origin, control_standard, control_origin = _headline_paths(spec["finalists"], int(spec["headline_seed"]))
        standard = compare(candidate_standard, control_standard, n_boot, block_len, seed)
        origin = compare(candidate_origin, control_origin, n_boot, block_len, seed)
        reports[name] = {"standard": standard, "origin": origin}
        # Holm needs null p-values.  Use the exact paired sign test under the
        # centered 50/50 win-probability null, not a complement of a bootstrap
        # probability from the observed (uncentered) differences.
        one_sided_p.append(float(standard["sign_test"]["p_a_better"]))

    holm = holm_bonferroni(one_sided_p, alpha=alpha)
    decisions = {}
    for name, holm_pass in zip(names, holm):
        spec = candidates[name]
        derived_diffs = seed_differences_from_artifacts(spec["finalists"])
        decisions[name] = promotion_gate(
            reports[name]["standard"], reports[name]["origin"],
            derived_diffs, holm_pass=holm_pass,
        )
        decisions[name]["complexity"] = int(spec.get("complexity", 0))
        decisions[name]["one_sided_p"] = one_sided_p[names.index(name)]

    passing = [name for name in names if decisions[name]["promote"]]
    # Prefer the simpler qualifying architecture, then the higher standard
    # realized total.  This encodes the plan's regularization preference and
    # avoids a post-result judgment call.
    winner = min(
        passing,
        key=lambda name: (decisions[name]["complexity"],
                          -reports[name]["standard"]["total_a"]),
    ) if passing else None
    return {"winner": winner, "decisions": decisions, "reports": reports,
            "holm_alpha": float(alpha), "candidate_order": names}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Assess final MoE candidates against CatBoost")
    parser.add_argument("manifest", help="JSON manifest of control and final candidate CSVs")
    parser.add_argument("--output", required=True, help="Decision JSON output")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--block-len", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-csv", default="experiments/results.csv")
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text())
    decision = assess_manifest(manifest, args.alpha, args.n_boot, args.block_len, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2, sort_keys=True))

    metrics = {"promoted": int(decision["winner"] is not None)}
    for name, report in decision["reports"].items():
        metrics[f"{name}_standard_diff"] = report["standard"]["total_diff"]
        metrics[f"{name}_origin_diff"] = report["origin"]["total_diff"]
        metrics[f"{name}_standard_ci_low"] = report["standard"]["ci_low"]
    log_result("position_moe_promotion_gate", {"manifest": manifest, "alpha": args.alpha,
                                               "winner": decision["winner"]}, metrics,
               results_path=args.results_csv)
    print(f"Promotion winner: {decision['winner'] or 'none; retain single:catboost'}")
    print(f"Saved decision evidence to {output}")


if __name__ == "__main__":
    main()
