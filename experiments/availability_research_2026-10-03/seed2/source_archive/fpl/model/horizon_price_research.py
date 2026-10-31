"""Cached horizon-price sensitivity conditional on reconstructed current prices.

Only the new current-origin-frozen horizon policy is scored. Verified replay and
lagged-asof controls are reused verbatim. Reconstructed current prices can retain
legacy forward/back filling; this experiment does not verify deadline prices.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd

from fpl import config, experiment
from fpl.data.origin_prices import validate_origin_price_catalog
from fpl.data.provenance import load_frozen_research_dataset
from fpl.model import blend_research, cnn_research, price_research, research_package
from fpl.model.moe_tournament import protocol_cutoffs


ARMS = {"mae_horizon": ("mae_full", "catboost_mae", "mae"),
        "rmse_horizon": ("rmse_full", "catboost_rmse", "rmse")}
COMPARISONS = (("mae_horizon_vs_replay", "mae_horizon", "mae_replay"),
               ("rmse_horizon_vs_replay", "rmse_horizon", "rmse_replay"),
               ("mae_horizon_vs_asof", "mae_horizon", "mae_asof"),
               ("rmse_horizon_vs_asof", "rmse_horizon", "rmse_asof"),
               ("rmse_horizon_vs_mae_horizon", "rmse_horizon", "mae_horizon"))


def _source_price_study(source_dir: Path, cutoffs: dict) -> tuple[dict, dict, dict[str, str]]:
    """Require a completed independently verified price study and seal all artifacts."""
    read, bind = blend_research._json, blend_research._bind
    verification_path = source_dir / "verification.json"
    if not verification_path.is_file():
        raise ValueError("completed price study requires verification.json")
    original = read(source_dir / "registration.json")
    summary = read(source_dir / "summary.json")
    verification = read(verification_path)
    expected_windows = {"selection": [cutoffs["selection_min_gw"], cutoffs["selection_max_gw"]],
                        "reference": [cutoffs["backtest_min_gw"], cutoffs["backtest_max_gw"]]}
    if (summary.get("status") != "complete" or verification.get("status") != "verified"
            or original.get("seed") != 0 or original.get("windows") != expected_windows
            or original.get("gap_rel") != 0 or original.get("chips") != "disabled"
            or original.get("horizon") != 3 or original.get("retrain_every") != 4
            or original.get("solver") != "highs" or original.get("initial_state") != "fresh_full_budget"
            or original.get("scoring_protocol") != "autosubs_v2"
            or original.get("points_accounting") != "net_after_transfer_hits"
            or original.get("eligibility_basis") != "phase_optimizer_union_with_known_lagged_prices"):
        raise ValueError("source is not the completed verified registered price protocol")
    inputs: dict[str, str] = {}
    bind(inputs, source_dir / "registration.json", verification["registration_sha256"])
    if summary.get("registration_sha256") != verification["registration_sha256"]:
        raise ValueError("price summary registration hash differs from verification")
    bind(inputs, source_dir / "summary.json", verification["summary_sha256"])
    bind(inputs, verification_path)
    lock = read(source_dir / "catalog_lock.json")
    if (lock.get("registration_sha256") != verification["registration_sha256"]
            or lock.get("origin_price_catalog_sha256") != original["origin_price_catalog_sha256"]):
        raise ValueError("verified price catalog lock differs from registration")
    bind(inputs, source_dir / "catalog_lock.json")
    if not summary.get("immutable_output_sha256") or not original.get("cached_input_sha256"):
        raise ValueError("verified price study lacks sealed source/output manifests")
    for path, digest in {**original["cached_input_sha256"], **summary["immutable_output_sha256"]}.items():
        bind(inputs, Path(path), digest)
    archived = original.get("code_sha256", {})
    if not archived:
        raise ValueError("verified price study has no archived evaluated sources")
    for relative, digest in archived.items():
        bind(inputs, source_dir / original["source_archive"] / relative, digest)
    for phase in expected_windows:
        for arm in price_research.ARMS:
            result = summary["phases"][phase][arm]
            directory = source_dir / phase
            bind(inputs, directory / result["predictions"], result["prediction_sha256"])
            bind(inputs, directory / result["squads"], result["squad_sha256"])
            prefix = directory / f"{arm}_origin"
            for suffix in (".admission.json", ".result.json", ".log"):
                path = prefix.with_suffix(suffix)
                if str(path.resolve()) not in summary["immutable_output_sha256"]:
                    raise ValueError("verified price study has an unsealed arm metadata/log artifact")
                bind(inputs, path, summary["immutable_output_sha256"][str(path.resolve())])
    return original, summary, inputs


def run(dataset_path: str | Path, output_dir: str | Path, *, source_dir: str | Path | None = None) -> dict[str, Any]:
    """Score only two new horizon-frozen arms per phase against verified controls.

    Original cached forecasts use the same parsing and export method as the price
    study. Exact export hashes are checked before scoring and against old controls
    after scoring. Both phase catalogs retain identical known-price eligibility.
    No features, forecast refits, tuning or adaptive admission are performed.
    """
    started = time.perf_counter()
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    source_dir = Path(source_dir) if source_dir is not None else config.ROOT / "experiments/price_provenance_2026-10-03/seed0"
    if output_dir.exists():
        raise ValueError("output directory already exists; preserve it and choose a new run")
    cutoffs = protocol_cutoffs()
    current_hashes = research_package.source_hashes()
    original, old_summary, inputs = _source_price_study(source_dir, cutoffs)
    cached_registration, cached_hashes, frames = price_research._source_inputs(Path(original["source_run"]), cutoffs)
    if (cached_registration != original["source_registration"]
            or cached_registration["dataset_sha256"] != original["dataset_sha256"]
            or cached_registration["parameters"] != original["parameters"]):
        raise ValueError("verified price source differs from the original registered cached model settings")
    for path, digest in cached_hashes.items():
        blend_research._bind(inputs, Path(path), digest)
    raw = load_frozen_research_dataset(dataset_path)
    if raw.attrs["frozen_dataset_sha256"] != original["dataset_sha256"]:
        raise ValueError("newly admitted dataset differs from verified price study")
    phases = {name: original["windows"][name] for name in ("selection", "reference")}
    parameters = copy.deepcopy(original["parameters"])
    columns = list(original["feature_columns"])
    if columns != cached_registration["feature_variants"]["mae_full"]:
        raise ValueError("verified price study's ordered feature columns differ from source models")
    catalogs, phase_raw = {}, {}
    for phase, (start, end) in phases.items():
        catalog_path = source_dir / f"{phase}_origin_prices.csv"
        blend_research._bind(inputs, catalog_path, original["origin_price_catalog_sha256"][phase])
        players = sorted(frames[phase]["mae_full"].player_id.unique().tolist())
        validate_origin_price_catalog(pd.read_csv(catalog_path), players, range(start, end + 1))
        catalogs[phase] = catalog_path
        phase_raw[phase] = raw.loc[raw.GW_global.between(1, end)].copy()
        for _, (variant, _, prefix) in ARMS.items():
            exported_sha = hashlib.sha256(frames[phase][variant].to_csv(index=False).encode()).hexdigest()
            replay, asof = (old_summary["phases"][phase][f"{prefix}_{policy}"] for policy in ("replay", "asof"))
            if exported_sha != replay["prediction_sha256"] or exported_sha != asof["prediction_sha256"]:
                raise ValueError("cached horizon forecasts cannot reproduce verified price prediction bytes")
    output_dir.mkdir(parents=True)
    registration = {
        "status": "registered", "promotion_eligible": False, "seed": 0,
        "hypothesis": "Freezing reconstructed current-origin prices across the horizon may change cached forecast decisions.",
        "dataset": str(dataset_path.resolve()), "dataset_sha256": raw.attrs["frozen_dataset_sha256"],
        "source_run": str(source_dir.resolve()), "source_registration": original,
        "cached_input_sha256": inputs, "code_sha256": current_hashes, "source_archive": "source_archive",
        "python": platform.python_version(), "versions": experiment._library_versions(),
        "parameters": parameters, "feature_columns": columns, "windows": phases, "arms": ARMS,
        "new_score_calls": len(ARMS) * len(phases), "expected_optimal_solves": len(ARMS) * sum(end - start + 1 for start, end in phases.values()),
        "comparison_labels": [label for label, _, _ in COMPARISONS],
        "comparison": {"n_boot": 10000, "block_len": 3, "seed": 0},
        "horizon": 3, "retrain_every": 4, "cached_replay_only": True, "gap_rel": 0,
        "chips": "disabled", "solver": "highs", "initial_budget": 1000, "initial_state": "fresh_full_budget",
        "origin_price_policy": "origin-current-proxy", "scoring_protocol": "autosubs_v2",
        "points_accounting": "net_after_transfer_hits", "eligibility_basis": original["eligibility_basis"],
        "origin_price_catalog_sha256": original["origin_price_catalog_sha256"],
        "catalog_diagnostics": original["catalog_diagnostics"],
        "limitations": ["Conditional horizon sensitivity uses the reconstructed replay current-origin vector, not verified deadline prices.",
                        "Current prices can retain legacy forward/back filling; roster and selling-price rules remain historical approximations.",
                        "Known-lagged-price eligibility remains identical to the verified price study, with no adaptive age filter.",
                        "The earlier price proxy experiment changed both current and horizon vectors; this policy isolates the horizon conditional on its reconstructed current vector.",
                        "Both windows are repeatedly examined historical evidence; no complete PIT, availability or promotion claim.",
                        "Only new horizon arms are scored; verified replay/asof controls are reused without model fitting or tuning."],
    }
    summary: dict[str, Any] = {"status": "running", "promotion_eligible": False, "phases": {}, "comparisons": {},
                               "reused_controls": old_summary["phases"], "catalog_diagnostics": original["catalog_diagnostics"]}
    outputs: dict[str, str] = {}
    registration_path, summary_path = output_dir / "registration.json", output_dir / "summary.json"
    try:
        cnn_research._archive_sources(output_dir / "source_archive", current_hashes)
        for relative, digest in current_hashes.items():
            blend_research._bind(outputs, output_dir / "source_archive" / relative, digest)
        cnn_research._write_json(registration_path, registration)
        blend_research._bind(outputs, registration_path)
        cnn_research._write_json(output_dir / "input_lock.json", {
            "registration_sha256": research_package.file_hash(registration_path),
            "cached_input_sha256": inputs, "origin_price_catalog_sha256": original["origin_price_catalog_sha256"]})
        blend_research._bind(outputs, output_dir / "input_lock.json")
        price_research._check(registration, outputs)
        experiment.log_result("horizon_price_research_registration", {"registration": str(registration_path), **registration},
                              {"status": "REGISTERED", "promotion_eligible": False})
        for phase, (start, end) in phases.items():
            directory = output_dir / phase
            directory.mkdir()
            summary["phases"][phase] = {}
            for arm, (variant, model, prefix) in ARMS.items():
                price_research._check(registration, outputs)
                cached = frames[phase][variant]
                result = research_package.evaluate_variant(
                    phase_raw[phase], phase_raw[phase], columns, arm, parameters, start, end, True, directory, 0,
                    model_name=model, retrain_every=4, forecast_provider=lambda cached=cached: cached.copy(),
                    origin_price_catalog=catalogs[phase], origin_price_policy="origin-current-proxy")
                summary["phases"][phase][arm] = result
                if not np.isfinite(result["total_points"]):
                    raise ValueError("horizon price net points must be finite")
                for key, hash_key in (("predictions", "prediction_sha256"), ("squads", "squad_sha256")):
                    blend_research._bind(outputs, directory / result[key], result[hash_key])
                metadata_prefix = directory / f"{arm}_origin"
                for suffix in (".admission.json", ".result.json", ".log"):
                    blend_research._bind(outputs, metadata_prefix.with_suffix(suffix))
                for policy in ("replay", "asof"):
                    baseline = old_summary["phases"][phase][f"{prefix}_{policy}"]
                    research_package.validate_paired_forecasts(directory / result["predictions"],
                                                               source_dir / phase / baseline["predictions"])
                    if result["prediction_sha256"] != baseline["prediction_sha256"]:
                        raise ValueError("horizon price policy changed verified model prediction bytes")
                price_research._check(registration, outputs)
                cnn_research._write_json(summary_path, summary)
            summary["comparisons"][phase] = {}
            for label, candidate_arm, control_arm in COMPARISONS:
                price_research._check(registration, outputs)
                candidate = summary["phases"][phase][candidate_arm]
                control = summary["phases"][phase].get(control_arm, old_summary["phases"][phase].get(control_arm))
                control_dir = directory if control_arm in ARMS else source_dir / phase
                research_package.validate_paired_forecasts(directory / candidate["predictions"], control_dir / control["predictions"])
                summary["comparisons"][phase][label] = research_package.compare_backtests.compare(
                    str(directory / candidate["squads"]), str(control_dir / control["squads"]), **registration["comparison"])
                price_research._check(registration, outputs)
            phase_summary = output_dir / f"{phase}_summary.json"
            cnn_research._write_json(phase_summary, {"results": summary["phases"][phase], "comparisons": summary["comparisons"][phase]})
            blend_research._bind(outputs, phase_summary)
        price_research._check(registration, outputs)
        summary.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                       immutable_output_sha256=dict(outputs), registration_sha256=research_package.file_hash(registration_path))
        cnn_research._write_json(summary_path, summary)
        experiment.log_result("horizon_price_research_completed", {"registration": str(registration_path),
                              "summary_sha256": research_package.file_hash(summary_path)},
                              {"status": "COMPLETE", "promotion_eligible": False, "comparisons": summary["comparisons"]})
        price_research._check(registration, outputs)
        print("Horizon price sensitivity completed: four new cached score calls; verified controls reused.", flush=True)
    except BaseException as exc:
        summary.update(status="incomplete", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter() - started,
                       immutable_output_sha256=dict(outputs), registration_sha256=outputs.get(str(registration_path.resolve())))
        cnn_research._write_json(summary_path, summary)
        experiment.log_result("horizon_price_research_incomplete", {"registration": str(registration_path)},
                              {"status": "INCOMPLETE", "error": summary["error"], "promotion_eligible": False})
        raise
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run four new cached horizon arms from a portable verified price study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.FROZEN_RESEARCH_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args(argv)
    run(args.dataset, args.output_dir, source_dir=args.source_dir)


if __name__ == "__main__":
    main()
