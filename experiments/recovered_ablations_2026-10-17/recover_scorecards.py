"""Qualify archived ablations and regenerate scorecards without fitting or solving.

Usage: PYTHONPATH=. python experiments/recovered_ablations_2026-10-17/recover_scorecards.py ORIGINAL_BUNDLE NEW_OUTPUT_DIR

Archived Python is hashed, never imported or executed. Original input files stay
unchanged. Accounting normalization is not independent scoring or decision replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpl import config, experiment
from fpl.artifacts import sha256_file, write_json_new
from fpl.data.provenance import load_frozen_dataset_manifest
from fpl.playground import league, read_run, render_league, run_experiment

_SELECTION_RULE = "Highest selection-origin MILP total among MAE ablations; full control retained on ties. RMSE is predeclared."
_VARIANTS = {"mae_full", "rmse_full", "mae_without_fixture", "mae_without_minutes", "mae_without_opponent", "mae_without_xp"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    _require(isinstance(value, dict), f"JSON object required: {path}")
    json.dumps(value, allow_nan=False)
    return value


def _member(root: Path, name: str) -> Path:
    path = root / name
    _require(not Path(name).is_absolute() and path.resolve().is_relative_to(root.resolve()), "unsafe archived input path")
    _require(path.is_file(), f"missing archived input: {name}")
    return path


def _pin(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _normalize(frame: pd.DataFrame, registration: dict, label: str) -> pd.DataFrame:
    _require(registration["points_accounting"] == "net_after_transfer_hits", "unrecognized registered accounting")
    required = {"gameweek", "alpha", "actual_total_points", "gross_actual_total_points", "points_accounting", "scoring_protocol"}
    _require(not frame.empty and required <= set(frame), f"missing accounting columns: {label}")
    _require(frame.points_accounting.eq("net_after_transfer_hits").all(), f"mixed legacy accounting: {label}")
    _require(frame.scoring_protocol.eq(registration["scoring_protocol"]).all(), f"scoring protocol mismatch: {label}")
    for name in ("gameweek", "alpha", "actual_total_points", "gross_actual_total_points"):
        _require(pd.api.types.is_numeric_dtype(frame[name]) and not pd.api.types.is_bool_dtype(frame[name])
                 and np.isfinite(frame[name]).all(), f"nonfinite/non-numeric legacy accounting: {label}/{name}")
    _require(((frame.alpha >= 0) & (frame.alpha % 1 == 0)).all(), f"invalid hit counts: {label}")
    _require(np.array_equal(frame.actual_total_points.to_numpy(),
                           (frame.gross_actual_total_points - 4 * frame.alpha).to_numpy()),
             f"legacy net/gross/hits disagreement: {label}")
    result = frame.copy()
    result["actual_net_points"] = frame.actual_total_points
    result["actual_total_points"] = frame.gross_actual_total_points
    result["transfer_hit_points"] = 4 * frame.alpha
    result["points_accounting"] = "gross_actual_total_points_and_explicit_net"
    return result


def _preflight(original: Path) -> dict:
    registration = _json(_member(original, "registration.json"))
    source_bundle = _json(_member(original, "source_bundle.json"))
    verification = _json(_member(original, "verification.json"))
    lock = _json(_member(original, "selection_lock.json"))
    summary_path = _member(original, "summary.json")
    selection_path = _member(original, "selection_summary.json")
    summary, selection = _json(summary_path), _json(selection_path)
    sources = registration["code_sha256"]
    _require(isinstance(sources, dict) and len(sources) == 37 and sources == source_bundle["source_sha256"],
             "registered/evaluated source identities differ or are incomplete")
    archived_files = {str(path.relative_to(original / "evaluated_sources"))
                      for path in (original / "evaluated_sources").rglob("*") if path.is_file()}
    _require(archived_files == set(sources), "archived evaluated-source coverage differs")
    for name, digest in sources.items():
        _require(sha256_file(_member(original / "evaluated_sources", name)) == digest, f"evaluated source changed: {name}")
    _require(source_bundle["status"] == "verified" and verification["status"] == "PASS", "archive verification is not PASS")
    _require(verification["evaluated_runner_sha256"] == sources["fpl/model/research_package.py"], "evaluated runner hash mismatch")
    _require(sha256_file(summary_path) == verification["summary_file_sha256"], "final summary file hash mismatch")
    canonical = hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()
    _require(canonical == verification["original_summary_canonical_payload_sha256"], "final summary payload hash mismatch")
    selection_hash = sha256_file(selection_path)
    _require(selection_hash == verification["selection_summary_sha256"] == lock["selection_summary_sha256"], "selection lock hash mismatch")
    _require(registration["selection_rule"] == lock["selection_rule"] == _SELECTION_RULE, "unrecognized selection rule")
    _require(registration["dataset_sha256"] == load_frozen_dataset_manifest()["dataset_sha256"], "archive dataset differs from canonical frozen manifest")
    _require(registration["scoring_protocol"] == "autosubs_v2" and registration["horizon"] == 3
             and registration["gap_rel"] == 0 and registration["chips"] == "disabled"
             and registration["promotion_eligible"] is False, "unsupported registered experiment assumptions")
    _require(summary["status"] == "complete" and summary["promotion_eligible"] is False, "original experiment is incomplete")
    _require(selection["selection"] == summary["selection"] and set(selection["selection"]) == _VARIANTS,
             "selection population changed after locking")
    totals = {name: record["total_points"] for name, record in selection["selection"].items()}
    _require(all(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)
                 for value in totals.values()), "nonfinite selection totals")
    ranked = sorted((name for name in _VARIANTS if name.startswith("mae_without_")), key=lambda name: (-totals[name], name))
    selected = ranked[0] if totals[ranked[0]] > totals["mae_full"] else None
    finalists = ["mae_full", "rmse_full"] + ([selected] if selected else [])
    _require(lock["selected_ablation"] == summary["selected_ablation"] == selected
             and lock["finalists"] == finalists and set(summary["evaluation"]) == set(finalists),
             "finalists disagree with the locked selection rule")
    runs = {}
    expected_files = set()
    verified_names = set()
    for stage in ("selection", "evaluation"):
        variants = sorted(_VARIANTS) if stage == "selection" else finalists
        protocols = ("origin",) if stage == "selection" else ("standard", "origin")
        cutoffs = registration["cutoffs"]
        start = cutoffs["selection_min_gw" if stage == "selection" else "backtest_min_gw"]
        end = cutoffs["selection_max_gw" if stage == "selection" else "backtest_max_gw"]
        for variant in variants:
            model = "catboost_rmse" if variant == "rmse_full" else "catboost_mae"
            for protocol in protocols:
                prefix = f"{variant}_{protocol}"
                label = f"{stage}/{prefix}"
                result_path = _member(original, f"{label}.result.json")
                expected_files.add(result_path)
                result = _json(result_path)
                admission = _json(_member(original, f"{label}.admission.json"))
                expected_result = summary[stage][variant] if stage == "selection" else summary[stage][variant][protocol]
                _require(result == expected_result, f"result differs from pinned summary: {label}")
                _require(result["variant"] == variant and result["origin"] is (protocol == "origin"), f"result identity mismatch: {label}")
                _require(result["predictions"] == prefix + ".predictions.csv" and result["squads"] == prefix + ".squads.csv", f"unexpected artifact names: {label}")
                predictions = _member(original / stage, result["predictions"])
                squads = _member(original / stage, result["squads"])
                _require(sha256_file(predictions) == result["prediction_sha256"] == admission["prediction_sha256"], f"forecast hash mismatch: {label}")
                _require(sha256_file(squads) == result["squad_sha256"], f"squad hash mismatch: {label}")
                expected_admission = {"variant": variant, "origin": protocol == "origin", "seed": registration["seed"],
                    "horizon": registration["horizon"], "start_gw": start, "end_gw": end,
                    "feature_columns": registration["feature_variants"][variant],
                    "parameters": registration["parameters"][model], "prediction_sha256": result["prediction_sha256"]}
                _require(admission == expected_admission, f"admission differs from registration: {label}")
                frame = pd.read_csv(squads)
                normalized = _normalize(frame, registration, label)
                _require(frame.gameweek.tolist() == list(range(start, end + 1)), f"squad round coverage mismatch: {label}")
                _require(result["total_points"] == float(frame.actual_total_points.sum())
                         and result["gross_points"] == float(frame.gross_actual_total_points.sum())
                         and result["transfer_hit_points"] == float(4 * frame.alpha.sum()), f"result accounting totals mismatch: {label}")
                runs[(stage, variant, protocol)] = {"predictions": predictions, "squads": squads, "normalized": normalized,
                                                    "result": result, "weeks": list(range(start, end + 1))}
                verified_names.add(f"{stage}/{variant}" if stage == "selection" else f"{stage}/{variant}/{protocol}")
    actual_results = set(original.glob("selection/*.result.json")) | set(original.glob("evaluation/*.result.json"))
    _require(actual_results == expected_files and len(runs) == 12, "unexpected archived result-file coverage")
    _require(verified_names == set(verification["verified_runs"]), "verification run identities differ")
    hashes = {str(path.relative_to(original)): sha256_file(path) for path in sorted(original.rglob("*")) if path.is_file()}
    return {"registration": registration, "selected": selected, "finalists": finalists,
            "runs": runs, "input_sha256": hashes, "verified_sources": len(sources)}


def recover(original_bundle: str | Path, output_dir: str | Path) -> Path:
    """Generate qualified saved-outcome reports while preserving all original bytes."""
    original, output = Path(original_bundle).resolve(), Path(output_dir).resolve()
    _require(not output.is_relative_to(original), "output must be separate from the original bundle")
    _require(not any(output.is_relative_to(path.resolve()) for path in (config.DATASETT_DIR, config.ROOT / "fpl")), "protected output directory")
    output.mkdir(parents=True, exist_ok=False)
    try:
        evidence = _preflight(original)
        normalized = {}
        normalization = {}
        for key, run in evidence["runs"].items():
            stage, variant, protocol = key
            target = output / "normalized" / stage / f"{variant}_{protocol}.squads.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8", newline="") as stream:
                run["normalized"].to_csv(stream, index=False)
            normalized[key] = target
            normalization["/".join(key)] = {"source": str(run["squads"]), "source_sha256": sha256_file(run["squads"]),
                "output": str(target.relative_to(output)), "output_sha256": sha256_file(target),
                "transformation": "Restore gross actual_total_points; preserve old actual_total_points as explicit actual_net_points; hits unchanged."}
        contrasts = [("selection", variant, "origin") for variant in sorted(_VARIANTS - {"mae_full"})]
        contrasts += [("evaluation", variant, protocol) for variant in evidence["finalists"] if variant != "mae_full"
                      for protocol in ("standard", "origin")]
        _require(len(contrasts) == 9, "locked archive does not contain the expected nine contrasts")
        comparisons, reports = {}, {}
        for stage, variant, protocol in contrasts:
            key, control = (stage, variant, protocol), (stage, "mae_full", protocol)
            run, baseline = evidence["runs"][key], evidence["runs"][control]
            name = "RMSE tuned-loss-and-capacity contrast" if variant == "rmse_full" else f"Matched MAE feature removal: {variant.removeprefix('mae_without_')}"
            identifier = f"recovered-{stage}-{variant}-{protocol}"
            spec = {"schema_version": 1, "id": identifier, "purpose": "historical_reanalysis", "metric": "actual_net_points",
                "hypothesis": name + "; recovered historical results, not fresh evidence or a promotion assessment.",
                "candidate": {"name": name, "input": _pin(normalized[key])},
                "control": {"name": "Explicit tuned MAE full-feature refit", "input": _pin(normalized[control])},
                "bootstrap": {"n_boot": 10000, "block_len": 3, "seed": evidence["registration"]["seed"]},
                "forecasts": {"information_protocol": protocol, "grain": "fixture",
                    "population": "Exact common archived forecast pool; source-run coverage and causal publication times are not independently assessed",
                    "candidate": _pin(run["predictions"]), "control": _pin(baseline["predictions"])}}
            spec_path = output / f"{identifier}.json"
            write_json_new(spec_path, spec)
            directory = run_experiment(spec_path, runs_dir=output / "runs")
            card = read_run(directory)
            _require(card["status"] == "UNVERIFIED" and card["verdict"]["promotion_eligible"] is False,
                     "recovered results must remain unverified and ineligible for promotion")
            comparisons[identifier] = card["decision_outcomes"]
            reports[identifier] = str(directory.relative_to(output))
            experiment.log_result(identifier,
                {"scope": "recovered historical reanalysis", "phase": stage, "variant": variant, "protocol": protocol,
                 "contrast": name, "registration_sha256": evidence["input_sha256"]["registration.json"],
                 "verified_source_count": evidence["verified_sources"], "promotion_eligible": False},
                {"status": "UNVERIFIED", "candidate_net_points": card["decision_outcomes"]["total_a"],
                 "control_net_points": card["decision_outcomes"]["total_b"], "net_difference": card["decision_outcomes"]["total_diff"],
                 "ci_low": card["decision_outcomes"]["ci_low"], "ci_high": card["decision_outcomes"]["ci_high"]},
                results_path=output / "registry.csv")
        current = {str(path.relative_to(original)): sha256_file(path) for path in sorted(original.rglob("*")) if path.is_file()}
        _require(current == evidence["input_sha256"], "original archive changed during report generation")
        standings = league(output / "runs")
        _require(all(row["rank"] is None for group in standings for row in group["entries"]), "recovered results must remain unranked")
        with (output / "LEAGUE.md").open("x", encoding="utf-8") as stream:
            stream.write(render_league(standings))
        qualification = {"schema_version": 1, "status": "QUALIFIED_HISTORICAL_REANALYSIS",
            "original_bundle": str(original), "input_sha256": evidence["input_sha256"],
            "source_bundle_verified_count": evidence["verified_sources"], "verified_result_count": len(evidence["runs"]),
            "selected_ablation": evidence["selected"], "finalists": evidence["finalists"],
            "normalization": normalization, "reports": reports, "comparisons": comparisons,
            "native_optimality": "NOT ASSESSED", "independent_legality": "NOT ASSESSED",
            "causal_validity": "NOT ASSESSED", "fresh_holdout": False, "promotion_eligible": False,
            "recovery_script_sha256": sha256_file(Path(__file__)),
            "limitations": ["Accounting/report recovery only: no fitting, optimizer solve, decision replay or independent settlement.",
                "Original log Optimal labels do not establish a native solver certificate.",
                "Only seed 0 was evaluated; historical windows were already examined.",
                "RMSE differs in tuned loss and capacity; it is not a loss-only experiment.",
                "MAE ablations retain their matched registered parameters.",
                "Original fitted-model coverage, historical fixture publication times, price policy and independent FPL legality are not established."]}
        qualification["output_sha256"] = {str(path.relative_to(output)): sha256_file(path)
                                           for path in sorted(output.rglob("*")) if path.is_file()}
        write_json_new(output / "qualification.json", qualification)
        return output
    except Exception as exc:
        write_json_new(output / "failed.json", {"status": "FAILED", "type": type(exc).__name__, "reason": str(exc), "promotion_eligible": False})
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_bundle", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    print(recover(args.original_bundle, args.output_dir))


if __name__ == "__main__":
    main()
