"""Bounded earlier-only CNN/RMSE blending with a later cached decision pilot.

Weights use current-origin player rounds in the first eight selection weeks.
The later block is scored from cached origins with fresh identical MILP state;
a strict net-points gain is required before constructing historical reference
features. This exploratory lane does not change production or promotion policy.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd

from fpl import config, experiment
from fpl.data.research import load_research_features
from fpl.model import cnn_research, research_package, temporal_blend
from fpl.model.moe_tournament import protocol_cutoffs
from fpl.model.sequence_features import sequence_columns


BLEND_NAME = "temporal_cnn_blend"
VARIANTS = {"blend": BLEND_NAME, "rmse_sequence": "catboost_rmse", "mae_full": "catboost_mae"}


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _bind(manifest: dict[str, str], path: Path, expected: str | None = None) -> None:
    digest = research_package.file_hash(path)
    if expected is not None and digest != expected:
        raise ValueError(f"cached source provenance hash mismatch: {path}")
    key = str(path.resolve())
    if key in manifest and manifest[key] != digest:
        raise ValueError(f"previously bound artifact changed: {path}")
    manifest[key] = digest


def _source_inputs(source_dir: Path, cutoffs: dict) -> tuple[dict, dict[str, str], pd.DataFrame, pd.DataFrame]:
    """Admit old completed member artifacts against their preserved source archive."""
    registration = _json(source_dir / "registration.json")
    summary = _json(source_dir / "summary.json")
    lock = _json(source_dir / "selection_lock.json")
    earlier_summary = _json(source_dir / "selection_summary.json")
    if (summary.get("status") not in ("complete", "selection_rejected") or registration.get("seed") != 0
            or registration.get("cutoffs") != cutoffs or registration.get("horizon") != 3
            or registration.get("retrain_every") != config.CNN_RETRAIN_EVERY
            or registration.get("points_accounting") != "net_after_transfer_hits"):
        raise ValueError("cached CNN source is not the completed registered fixed-fit seed-zero protocol")
    if earlier_summary.get("selection") != summary.get("selection"):
        raise ValueError("cached completed selection differs from the immutable selection summary")
    if registration.get("sequence_feature_columns") != sequence_columns():
        raise ValueError("cached sequence schema differs from the current registered sequence schema")
    inputs: dict[str, str] = {}
    for name in ("registration.json", "summary.json", "selection_lock.json"):
        _bind(inputs, source_dir / name)
    _bind(inputs, source_dir / "selection_summary.json", lock["selection_summary_sha256"])
    # Old code is checked against its own archived bytes; it need not equal current
    # runner/model source, which is independently registered and archived below.
    archived = registration.get("code_sha256", {})
    if not archived:
        raise ValueError("cached source registration has no archived source provenance")
    archive_dir = source_dir / registration["source_archive"]
    for relative, digest in archived.items():
        _bind(inputs, archive_dir / relative, digest)
    verification_path = source_dir / "verification.json"
    if verification_path.exists():
        verification = _json(verification_path)
        if verification.get("status") != "VERIFIED":
            raise ValueError("cached CNN source verification is not VERIFIED")
        for name, key in (("registration.json", "registration_sha256"), ("summary.json", "summary_sha256"),
                          ("selection_lock.json", "selection_lock_sha256")):
            _bind(inputs, source_dir / name, verification[key])
        _bind(inputs, verification_path)
    paths = {}
    for variant in ("cnn_sequence", "rmse_sequence"):
        result = summary["selection"][variant]
        path = source_dir / "selection" / result["predictions"]
        _bind(inputs, path, result["prediction_sha256"])
        research_package.validate_forecasts(pd.read_csv(path), cutoffs["selection_min_gw"], cutoffs["selection_max_gw"], True)
        paths[variant] = path
    research_package.validate_paired_forecasts(paths["cnn_sequence"], paths["rmse_sequence"])
    return registration, inputs, pd.read_csv(paths["cnn_sequence"]), pd.read_csv(paths["rmse_sequence"])


def _check(registration: dict, immutable_outputs: dict[str, str] | None = None) -> None:
    """Recheck cached artifacts, dataset bytes and all current sources around scoring."""
    if research_package.source_hashes() != registration["code_sha256"]:
        raise ValueError("current pipeline source changed after blend registration")
    if research_package.file_hash(Path(registration["dataset"])) != registration["dataset_sha256"]:
        raise ValueError("dataset bytes changed after blend registration")
    for path, digest in registration["cached_input_sha256"].items():
        if research_package.file_hash(Path(path)) != digest:
            raise ValueError(f"cached blend input changed after registration: {path}")
    for path, digest in (immutable_outputs or {}).items():
        if research_package.file_hash(Path(path)) != digest:
            raise ValueError(f"immutable blend output changed after locking: {path}")


def run(dataset_path: str | Path, output_dir: str | Path, *, source_dir: str | Path | None = None) -> dict[str, Any]:
    """Register earlier weights, score the later pilot, then admit reference on gain.

    Exact member settings come from the completed CNN registration. Reference
    implementations are present in the archived current source before pilot scoring.
    Every run is new; existing artifacts cannot be resumed or overwritten.
    """
    started = time.perf_counter()
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    source_dir = Path(source_dir) if source_dir is not None else config.ROOT / "experiments/cnn_research_2026-10-03/seed0"
    if output_dir.exists():
        raise ValueError("output directory already exists; preserve it and choose a new run")
    cutoffs = protocol_cutoffs()
    weight_start = cutoffs["selection_min_gw"]
    weight_end = weight_start + config.BLEND_WEIGHT_WEEKS - 1
    pilot_start, pilot_end = weight_end + 1, cutoffs["selection_max_gw"]
    if config.BLEND_WEIGHT_WEEKS < 1 or pilot_start > pilot_end:
        raise ValueError("weight-fitting block must leave a nonempty later selection pilot")
    if config.CNN_RETRAIN_EVERY <= max(pilot_end - pilot_start + 1,
                                      cutoffs["backtest_max_gw"] - cutoffs["backtest_min_gw"] + 1):
        raise ValueError("blend reference must use one fixed pre-window fit")
    code_hashes = research_package.source_hashes()
    original, input_hashes, cnn, tree = _source_inputs(source_dir, cutoffs)
    raw, base_frame, base_columns = load_research_features(pilot_end, dataset_path)
    if (raw.attrs["frozen_dataset_sha256"] != original["dataset_sha256"]
            or base_columns != original["base_feature_columns"]):
        raise ValueError("cached member dataset or ordered base feature schema differs")
    # The helper uses only first-block current-origin observations for fitting.
    weights = temporal_blend.fit_earlier_weights(cnn, tree, weight_start, weight_end)
    parameters = copy.deepcopy(original["parameters"])
    parameters[BLEND_NAME] = {
        position: {BLEND_NAME: {"cnn_weight": weights[position],
                               "cnn_params": copy.deepcopy(parameters["sequence_cnn"][position]["sequence_cnn"]),
                               "tree_params": copy.deepcopy(parameters["catboost_rmse"][position]["catboost_rmse"])}}
        for position in config.ONFIELD_POSITIONS}
    late_mask = tree.origin_gw.between(pilot_start, pilot_end) & tree.GW.between(pilot_start, pilot_end)
    late_tree = tree.loc[late_mask].copy()
    cnn_mask = cnn.origin_gw.between(pilot_start, pilot_end) & cnn.GW.between(pilot_start, pilot_end)
    late_cnn = cnn.loc[cnn_mask].copy()
    late_blend = temporal_blend.blend_forecasts(late_cnn, late_tree, weights)
    seq_columns = sequence_columns()
    variants = {variant: list(base_columns) + ([] if variant == "mae_full" else seq_columns)
                for variant in VARIANTS}
    output_dir.mkdir(parents=True)
    registration = {
        "status": "registered", "promotion_eligible": False, "seed": 0,
        "hypothesis": "Earlier-only convex CNN/RMSE weights may improve later origin MILP net points.",
        "dataset": str(dataset_path.resolve()), "dataset_sha256": raw.attrs["frozen_dataset_sha256"],
        "source_run": str(source_dir.resolve()), "source_registration": original,
        "cached_input_sha256": input_hashes, "code_sha256": code_hashes, "source_archive": "source_archive",
        "python": platform.python_version(), "versions": experiment._library_versions(),
        "parameters": parameters, "weights": weights, "feature_variants": variants,
        "base_feature_columns": list(base_columns), "sequence_feature_columns": seq_columns,
        "windows": {"weights": [weight_start, weight_end], "pilot": [pilot_start, pilot_end],
                    "reference": [cutoffs["backtest_min_gw"], cutoffs["backtest_max_gw"]]},
        "cutoffs": cutoffs, "retrain_every": config.CNN_RETRAIN_EVERY, "horizon": 3,
        "scoring_protocol": "autosubs_v2", "points_accounting": "net_after_transfer_hits",
        "chips": "disabled", "gap_rel": 0, "solver": "highs", "initial_state": "fresh_full_budget",
        "comparison": {"n_boot": 10000, "block_len": 3, "seed": 0},
        "selection_rule": "Advance only if cached later origin blend net points strictly exceed cached RMSE sequence net points; ties reject.",
        "runtime_policy": {"cnn_backend": "isolated_cpu_worker", "device": "cpu", "pytabkit_attempts": 0},
        "limitations": ["Weight fitting uses earlier current-origin player rounds only; pilot reuses historical selection data.",
                        "The full selection window was examined by the CNN study and selected its strong tree; this later replay is exploratory continuation, not untouched independent validation. Earlier-only weight causality does not remove member-selection reuse.",
                        "Cached members were fitted through the original discovery cutoff; the later pilot resets MILP state.",
                        "Recorded sequence steps span seasons and are not calendar weeks.",
                        "Archived prices and fixtures lack complete deadline publication-time provenance.",
                        "Reference is repeatedly examined historical data, not an untouched holdout.",
                        "Fixed-fit seed-zero exploratory comparison; no production or promotion claim."],
    }
    summary: dict[str, Any] = {"status": "running", "promotion_eligible": False,
                               "pilot": {}, "evaluation": {}, "comparisons": {}}
    registration_path, summary_path = output_dir / "registration.json", output_dir / "summary.json"
    immutable_outputs: dict[str, str] = {}
    try:
        cnn_research._archive_sources(output_dir / "source_archive", code_hashes)
        for relative, digest in code_hashes.items():
            _bind(immutable_outputs, output_dir / "source_archive" / relative, digest)
        _check(registration, immutable_outputs)
        cnn_research._write_json(registration_path, registration)
        cnn_research._write_json(output_dir / "weights.json", {"weights": weights, "window": [weight_start, weight_end],
                                                               "fitting_rows": "current_origin_only"})
        weight_lock = {"weights_sha256": research_package.file_hash(output_dir / "weights.json"),
                       "parameters": parameters[BLEND_NAME], "reference_variants": variants,
                       "selection_rule": registration["selection_rule"]}
        cnn_research._write_json(output_dir / "weights_lock.json", weight_lock)
        for path in (registration_path, output_dir / "weights.json", output_dir / "weights_lock.json"):
            _bind(immutable_outputs, path)
        _check(registration, immutable_outputs)
        experiment.log_result("blend_research_registration", {"registration": str(registration_path), **registration},
                              {"status": "REGISTERED", "promotion_eligible": False})
        pilot_dir = output_dir / "pilot"
        pilot_dir.mkdir()
        for variant, cached in (("rmse_sequence", late_tree), ("blend", late_blend)):
            _check(registration, immutable_outputs)
            summary["pilot"][variant] = research_package.evaluate_variant(
                raw, base_frame, variants[variant], variant, parameters, pilot_start, pilot_end, True, pilot_dir, 0,
                model_name=VARIANTS[variant], retrain_every=config.CNN_RETRAIN_EVERY,
                snapshot_builder=cnn_research.build_sequence_snapshot, extra_player_features=tuple(seq_columns),
                forecast_provider=lambda cached=cached: cached.copy())
            result = summary["pilot"][variant]
            _bind(immutable_outputs, pilot_dir / result["predictions"], result["prediction_sha256"])
            _bind(immutable_outputs, pilot_dir / result["squads"], result["squad_sha256"])
            if not np.isfinite(summary["pilot"][variant]["total_points"]):
                raise ValueError("pilot net points must be finite")
            research_package.validate_paired_forecasts(pilot_dir / summary["pilot"][variant]["predictions"],
                                                       pilot_dir / summary["pilot"]["rmse_sequence"]["predictions"])
            _check(registration, immutable_outputs)
            cnn_research._write_json(summary_path, summary)
        blend_result, tree_result = summary["pilot"]["blend"], summary["pilot"]["rmse_sequence"]
        _check(registration, immutable_outputs)
        summary["comparisons"]["pilot_blend_vs_rmse"] = research_package.compare_backtests.compare(
            str(pilot_dir / blend_result["squads"]), str(pilot_dir / tree_result["squads"]), **registration["comparison"])
        _check(registration, immutable_outputs)
        advances = blend_result["total_points"] > tree_result["total_points"]
        pilot_summary_path = output_dir / "pilot_summary.json"
        cnn_research._write_json(pilot_summary_path, summary)
        lock = {"advances": advances, "finalists": list(VARIANTS) if advances else [],
                "pilot_summary_sha256": research_package.file_hash(pilot_summary_path),
                "weights_sha256": weight_lock["weights_sha256"], "parameters": parameters,
                "feature_variants": variants, "selection_rule": registration["selection_rule"]}
        cnn_research._write_json(output_dir / "pilot_lock.json", lock)
        for path in (pilot_summary_path, output_dir / "pilot_lock.json"):
            _bind(immutable_outputs, path)
        _check(registration, immutable_outputs)
        summary["advances"] = advances
        if not advances:
            summary["status"] = "pilot_rejected"
            print(f"Blend pilot rejected: {blend_result['total_points']:.0f} vs RMSE {tree_result['total_points']:.0f} net points; reference skipped.", flush=True)
        else:
            reference_raw, reference_base, reference_columns = load_research_features(cutoffs["backtest_max_gw"], dataset_path)
            if (reference_columns != base_columns
                    or reference_raw.attrs["frozen_dataset_sha256"] != registration["dataset_sha256"]):
                raise ValueError("reference dataset or ordered feature schema differs from registration")
            reference_frame = cnn_research.attach_sequences(reference_raw, reference_base)
            _check(registration, immutable_outputs)
            reference_dir = output_dir / "evaluation"
            reference_dir.mkdir()
            for variant in VARIANTS:
                summary["evaluation"][variant] = {}
                for origin in (False, True):
                    _check(registration, immutable_outputs)
                    mode = "origin" if origin else "standard"
                    summary["evaluation"][variant][mode] = research_package.evaluate_variant(
                        reference_raw, reference_frame, variants[variant], variant, parameters,
                        cutoffs["backtest_min_gw"], cutoffs["backtest_max_gw"], origin, reference_dir, 0,
                        model_name=VARIANTS[variant], retrain_every=config.CNN_RETRAIN_EVERY,
                        snapshot_builder=None if variant == "mae_full" else cnn_research.build_sequence_snapshot,
                        extra_player_features=() if variant == "mae_full" else tuple(seq_columns))
                    result = summary["evaluation"][variant][mode]
                    _bind(immutable_outputs, reference_dir / result["predictions"], result["prediction_sha256"])
                    _bind(immutable_outputs, reference_dir / result["squads"], result["squad_sha256"])
                    _check(registration, immutable_outputs)
                    cnn_research._write_json(summary_path, summary)
            for control in ("rmse_sequence", "mae_full"):
                label = f"reference_blend_vs_{control}"
                summary["comparisons"][label] = {}
                for mode in ("standard", "origin"):
                    _check(registration, immutable_outputs)
                    candidate = summary["evaluation"]["blend"][mode]
                    baseline = summary["evaluation"][control][mode]
                    research_package.validate_paired_forecasts(reference_dir / candidate["predictions"],
                                                               reference_dir / baseline["predictions"])
                    summary["comparisons"][label][mode] = research_package.compare_backtests.compare(
                        str(reference_dir / candidate["squads"]), str(reference_dir / baseline["squads"]),
                        **registration["comparison"])
                    _check(registration, immutable_outputs)
            summary["status"] = "complete"
            print("Blend reference completed; paired net-points reports preserved.", flush=True)
        _check(registration, immutable_outputs)
        summary["elapsed_seconds"] = time.perf_counter() - started
        cnn_research._write_json(summary_path, summary)
        experiment.log_result("blend_research_completed", {"registration": str(registration_path),
                              "summary_sha256": research_package.file_hash(summary_path)},
                              {"status": summary["status"].upper(), "promotion_eligible": False,
                               "comparisons": summary["comparisons"]})
        _check(registration, immutable_outputs)
    except BaseException as exc:
        summary.update(status="incomplete", error=f"{type(exc).__name__}: {exc}",
                       elapsed_seconds=time.perf_counter() - started)
        cnn_research._write_json(summary_path, summary)
        experiment.log_result("blend_research_incomplete", {"registration": str(registration_path)},
                              {"status": "INCOMPLETE", "error": summary["error"], "promotion_eligible": False})
        raise
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run a new registered cached-pilot blend study without changing production."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.FROZEN_RESEARCH_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args(argv)
    run(args.dataset, args.output_dir, source_dir=args.source_dir)


if __name__ == "__main__":
    main()
