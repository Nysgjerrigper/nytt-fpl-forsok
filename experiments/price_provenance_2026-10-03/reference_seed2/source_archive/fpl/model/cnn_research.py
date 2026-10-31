"""Bounded CNN research with earlier selection and immutable reference admission.

This optional lane makes no promotion claim. Its fixed fits, losses and capacity
are different from the standing rolling-refit control and do not isolate a pure
architecture effect. Recorded player rounds span seasons and are not calendar weeks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd

from fpl import config, experiment
from fpl.data.research import load_research_features
from fpl.model import models, research_package
from fpl.model.moe_tournament import protocol_cutoffs
from fpl.model.sequence_features import add_sequence_features, sequence_columns, sequence_snapshot
from fpl.run_week import build_live_snapshot


TREE_VARIANTS = ("mae_full", "mae_sequence", "rmse_sequence")
MODEL_NAMES = {"mae_full": "catboost_mae", "mae_sequence": "catboost_mae",
               "rmse_sequence": "catboost_rmse", "cnn_sequence": "sequence_cnn"}


def attach_sequences(raw: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Append prior-round channels to existing feature rows without rebuilding them.

    Align by stable player code and global round, preserving fixture rows, order
    and index. Every fixture in a round receives the same prior-round sequence.
    """
    columns = sequence_columns()
    if set(columns) & set(frame):
        raise ValueError("feature frame already contains generated sequence columns")
    keys = ["player_id", "GW_global"]
    history = add_sequence_features(raw)[keys + columns].drop_duplicates(keys)
    positions = pd.MultiIndex.from_frame(history[keys]).get_indexer(pd.MultiIndex.from_frame(frame[keys]))
    if (positions < 0).any():
        raise ValueError("feature rows lack matching raw player-round history")
    aligned = history.iloc[positions][columns].copy()
    aligned.index = frame.index
    return pd.concat([frame.copy(), aligned], axis=1)


def build_sequence_snapshot(hist_raw: pd.DataFrame) -> pd.DataFrame:
    """Add exact recorded-round sequences to a history-only live form snapshot.

    No data is fetched. The caller must pass raw rows strictly before the origin.
    Keep precisely the base snapshot's player universe and index.
    """
    snapshot = build_live_snapshot(hist_raw)
    columns = sequence_columns()
    if set(columns) & set(snapshot):
        raise ValueError("base snapshot already contains sequence columns")
    history = sequence_snapshot(hist_raw).set_index("player_id")
    positions = history.index.get_indexer(snapshot["player_id"])
    if (positions < 0).any():
        raise ValueError("snapshot players lack matching sequence history")
    aligned = history.iloc[positions][columns].copy()
    aligned.index = snapshot.index
    return pd.concat([snapshot.copy(), aligned], axis=1)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _archive_sources(directory: Path, expected: dict[str, str]) -> None:
    """Preserve the exact registered Python source bytes before the first fit."""
    for relative, digest in expected.items():
        content = (config.ROOT / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("pipeline source changed during source archival")
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as output:
            output.write(content)


def _bind_output(manifest: dict[str, str], path: Path, expected: str | None = None) -> None:
    """Bind preserved output bytes without allowing a changed file to be rebound."""
    digest = research_package.file_hash(path)
    key = str(path.resolve())
    if expected is not None and digest != expected:
        raise ValueError(f"CNN output hash mismatch: {path}")
    if key in manifest and manifest[key] != digest:
        raise ValueError(f"previously bound CNN output changed: {path}")
    manifest[key] = digest


def _check_provenance(registration: dict, immutable_outputs: dict[str, str] | None = None) -> None:
    """Fail on source, full dataset byte or tuned-artifact drift during a run."""
    if research_package.source_hashes() != registration["code_sha256"]:
        raise ValueError("pipeline source changed after CNN registration")
    if research_package.file_hash(Path(registration["dataset"])) != registration["dataset_sha256"]:
        raise ValueError("dataset bytes changed after CNN registration")
    for relative, digest in registration["tuned_artifact_sha256"].items():
        if research_package.file_hash(config.ROOT / relative) != digest:
            raise ValueError("tuned artifacts changed after CNN registration")
    for path, digest in (immutable_outputs or {}).items():
        if research_package.file_hash(Path(path)) != digest:
            raise ValueError(f"immutable CNN output changed after locking: {path}")


def run(dataset_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Register and run fixed-fit selection, admitting reference only on strict gain.

    CNN must beat the best predeclared tree's earlier origin MILP net points.
    Reference settings are locked before reference feature construction. Existing
    outputs cannot be resumed or overwritten; failures preserve an incomplete summary.
    """
    started = time.perf_counter()
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    if output_dir.exists():
        raise ValueError("output directory already exists; preserve it and choose a new run")
    cutoffs = protocol_cutoffs()
    retrain = config.CNN_RETRAIN_EVERY
    if (isinstance(retrain, bool) or not isinstance(retrain, int)
            or retrain <= max(cutoffs["selection_max_gw"] - cutoffs["selection_min_gw"] + 1,
                              cutoffs["backtest_max_gw"] - cutoffs["backtest_min_gw"] + 1)):
        raise ValueError("CNN retrain interval must exceed both windows for fixed pre-window fits")
    code_hashes = research_package.source_hashes()
    # Earlier inputs are the only feature frame built until the finalist lock exists.
    sel_raw, sel_base, base_columns = load_research_features(cutoffs["selection_max_gw"], dataset_path)
    sel_frame = attach_sequences(sel_raw, sel_base)
    seq_columns = sequence_columns()
    parameters, artifact_hashes = research_package.resolve_parameters("selection", 0)
    cnn_params = models.EXPLORATORY_SPECS["sequence_cnn"].default_params(0)
    parameters["sequence_cnn"] = {position: {"sequence_cnn": dict(cnn_params)}
                                  for position in config.ONFIELD_POSITIONS}
    variants = {name: list(base_columns) + ([] if name == "mae_full" else seq_columns)
                for name in MODEL_NAMES}
    output_dir.mkdir(parents=True)
    registration = {
        "status": "registered", "promotion_eligible": False, "seed": 0,
        "hypothesis": "A bounded sequence CNN may improve earlier origin decisions over fixed-fit tree controls.",
        "dataset": str(dataset_path.resolve()), "dataset_sha256": sel_raw.attrs["frozen_dataset_sha256"],
        "code_sha256": code_hashes, "source_archive": "source_archive",
        "python": platform.python_version(), "versions": experiment._library_versions(),
        "runtime_policy": {"cnn_backend": "isolated_cpu_worker", "device": "cpu", "pytabkit_attempts": 0},
        "tuned_artifact_sha256": artifact_hashes, "parameters": parameters,
        "base_feature_columns": list(base_columns), "sequence_feature_columns": seq_columns,
        "sequence_length": config.CNN_SEQUENCE_LENGTH, "sequence_stats": list(config.CNN_SEQUENCE_STATS),
        "architecture": {"convolution_layers": 2, "kernel_size": 3, "padding": 1,
                         "channels": cnn_params.get("channels", config.CNN_CHANNELS),
                         "hidden": cnn_params.get("hidden", config.CNN_HIDDEN),
                         "activation": "relu", "pooling": "presence_masked_mean",
                         "sequence_missing_flags": True, "context_missing_flags": True},
        "training_objective": {"loss": "MSE", "target_standardization": "training_only_mean_std",
                               "preprocessing": "training_only_imputation_and_scaling"},
        "feature_variants": variants, "model_names": MODEL_NAMES, "cutoffs": cutoffs,
        "train_max_gw": {"selection": cutoffs["selection_min_gw"] - 1,
                         "reference": cutoffs["backtest_min_gw"] - 1},
        "retrain_every": retrain, "horizon": 3, "chips": "disabled", "solver": "highs", "gap_rel": 0,
        "scoring_protocol": "autosubs_v2", "points_accounting": "net_after_transfer_hits",
        "comparison": {"n_boot": 10000, "block_len": 3, "seed": 0},
        "selection_rule": "Best tree by earlier origin net points; tree ties prefer mae_full then mae_sequence then rmse_sequence. CNN advances only on a strict net-points gain.",
        "limitations": ["Recorded player rounds span seasons; sequence steps are not calendar weeks.",
                        "Archived prices and fixture ratings lack complete deadline publication-time provenance.",
                        "Reference is repeatedly examined historical data, not a fresh untouched holdout.",
                        "Loss, capacity and input differences do not isolate an architecture effect.",
                        "Fixed pre-window fits differ from the standing rolling-refit controls.",
                        "Seed zero only; no tuning, production change or promotion eligibility."],
    }
    summary: dict[str, Any] = {"status": "running", "promotion_eligible": False,
                               "selection": {}, "evaluation": {}, "comparisons": {}}
    summary_path = output_dir / "summary.json"
    registration_path = output_dir / "registration.json"
    immutable_outputs: dict[str, str] = {}
    try:
        _archive_sources(output_dir / "source_archive", code_hashes)
        for relative, digest in code_hashes.items():
            _bind_output(immutable_outputs, output_dir / "source_archive" / relative, digest)
        _check_provenance(registration, immutable_outputs)
        _write_json(registration_path, registration)
        _bind_output(immutable_outputs, registration_path)
        experiment.log_result("cnn_research_registration", {"registration": str(registration_path), **registration},
                              {"status": "REGISTERED", "promotion_eligible": False})
        selection_dir = output_dir / "selection"
        selection_dir.mkdir()
        for variant, columns in variants.items():
            _check_provenance(registration, immutable_outputs)
            summary["selection"][variant] = research_package.evaluate_variant(
                sel_raw, sel_frame, columns, variant, parameters,
                cutoffs["selection_min_gw"], cutoffs["selection_max_gw"], True, selection_dir, 0,
                model_name=MODEL_NAMES[variant], retrain_every=retrain,
                snapshot_builder=None if variant == "mae_full" else build_sequence_snapshot,
                extra_player_features=() if variant == "mae_full" else tuple(seq_columns))
            result = summary["selection"][variant]
            _bind_output(immutable_outputs, selection_dir / result["predictions"], result["prediction_sha256"])
            _bind_output(immutable_outputs, selection_dir / result["squads"], result["squad_sha256"])
            research_package.validate_paired_forecasts(
                selection_dir / summary["selection"][variant]["predictions"],
                selection_dir / summary["selection"]["mae_full"]["predictions"])
            if not np.isfinite(summary["selection"][variant]["total_points"]):
                raise ValueError("selection net points must be finite")
            _check_provenance(registration, immutable_outputs)
            _write_json(summary_path, summary)
        tree_best = max(TREE_VARIANTS, key=lambda name: summary["selection"][name]["total_points"])
        cnn_points = summary["selection"]["cnn_sequence"]["total_points"]
        admitted = cnn_points > summary["selection"][tree_best]["total_points"]
        selection_summary = output_dir / "selection_summary.json"
        _write_json(selection_summary, summary)
        finalists = (["mae_full"] + ([tree_best] if tree_best != "mae_full" else []) + ["cnn_sequence"]
                     if admitted else [])
        lock = {"tree_best": tree_best, "cnn_advances": admitted, "finalists": finalists,
                "selection_summary_sha256": research_package.file_hash(selection_summary),
                "selection_rule": registration["selection_rule"], "retrain_every": retrain,
                "settings": {variant: {"model_name": MODEL_NAMES[variant], "feature_columns": variants[variant],
                                       "parameters": parameters[MODEL_NAMES[variant]], "seed": 0}
                             for variant in finalists}}
        _write_json(output_dir / "selection_lock.json", lock)
        for path in (selection_summary, output_dir / "selection_lock.json"):
            _bind_output(immutable_outputs, path)
        _check_provenance(registration, immutable_outputs)
        summary.update(tree_best=tree_best, cnn_advances=admitted)
        if not admitted:
            summary["status"] = "selection_rejected"
            print(f"CNN selection rejected: {cnn_points:.0f} net points vs {tree_best} "
                  f"{summary['selection'][tree_best]['total_points']:.0f}; reference skipped.", flush=True)
        else:
            raw, base_frame, reference_columns = load_research_features(cutoffs["backtest_max_gw"], dataset_path)
            if reference_columns != base_columns or raw.attrs["frozen_dataset_sha256"] != registration["dataset_sha256"]:
                raise ValueError("reference input differs from registered dataset or feature schema")
            frame = attach_sequences(raw, base_frame)
            _check_provenance(registration, immutable_outputs)
            evaluation_dir = output_dir / "evaluation"
            evaluation_dir.mkdir()
            for variant in finalists:
                summary["evaluation"][variant] = {}
                for origin in (False, True):
                    _check_provenance(registration, immutable_outputs)
                    mode = "origin" if origin else "standard"
                    summary["evaluation"][variant][mode] = research_package.evaluate_variant(
                        raw, frame, variants[variant], variant, parameters,
                        cutoffs["backtest_min_gw"], cutoffs["backtest_max_gw"], origin, evaluation_dir, 0,
                        model_name=MODEL_NAMES[variant], retrain_every=retrain,
                        snapshot_builder=None if variant == "mae_full" else build_sequence_snapshot,
                        extra_player_features=() if variant == "mae_full" else tuple(seq_columns))
                    result = summary["evaluation"][variant][mode]
                    _bind_output(immutable_outputs, evaluation_dir / result["predictions"], result["prediction_sha256"])
                    _bind_output(immutable_outputs, evaluation_dir / result["squads"], result["squad_sha256"])
                    _check_provenance(registration, immutable_outputs)
                    _write_json(summary_path, summary)
            for label, control_name in (("cnn_vs_selected_tree", tree_best), ("cnn_vs_mae_full", "mae_full")):
                summary["comparisons"][label] = {}
                for mode in ("standard", "origin"):
                    _check_provenance(registration, immutable_outputs)
                    candidate = summary["evaluation"]["cnn_sequence"][mode]
                    control = summary["evaluation"][control_name][mode]
                    research_package.validate_paired_forecasts(evaluation_dir / candidate["predictions"],
                                                               evaluation_dir / control["predictions"])
                    summary["comparisons"][label][mode] = research_package.compare_backtests.compare(
                        str(evaluation_dir / candidate["squads"]), str(evaluation_dir / control["squads"]),
                        **registration["comparison"])
                    _check_provenance(registration, immutable_outputs)
            summary["status"] = "complete"
            print(f"CNN reference completed against {tree_best} and mae_full; paired net-points reports preserved.", flush=True)
        _check_provenance(registration, immutable_outputs)
        summary["elapsed_seconds"] = time.perf_counter() - started
        _write_json(summary_path, summary)
        experiment.log_result("cnn_research_completed", {"registration": str(registration_path),
                              "summary_sha256": research_package.file_hash(summary_path)},
                              {"status": summary["status"].upper(), "promotion_eligible": False,
                               "comparisons": summary["comparisons"]})
        _check_provenance(registration, immutable_outputs)
    except BaseException as exc:
        summary.update(status="incomplete", error=f"{type(exc).__name__}: {exc}",
                       elapsed_seconds=time.perf_counter() - started)
        _write_json(summary_path, summary)
        experiment.log_result("cnn_research_incomplete", {"registration": str(registration_path)},
                              {"status": "INCOMPLETE", "error": summary["error"], "promotion_eligible": False})
        raise
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run the bounded CNN lane in a new artifact directory without production changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.FROZEN_RESEARCH_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
