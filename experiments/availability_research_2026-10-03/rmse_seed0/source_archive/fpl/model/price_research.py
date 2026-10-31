"""Descriptive cached-forecast price-provenance comparisons without model fitting.

All registered price policies use the same lagged-price-known eligibility. Latest recorded
prices are a proxy, not verified historical deadline prices or full point-in-time
availability. Reference results are repeatedly examined historical evidence and
cannot promote a model or price policy to production.
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
from fpl.data.origin_prices import build_origin_price_catalog, validate_origin_price_catalog
from fpl.data.provenance import load_frozen_research_dataset
from fpl.model import blend_research, cnn_research, research_package
from fpl.model.moe_tournament import protocol_cutoffs


ARMS = {"mae_replay": ("mae_full", "catboost_mae", "target-replay"),
        "mae_asof": ("mae_full", "catboost_mae", "origin-asof"),
        "rmse_replay": ("rmse_full", "catboost_rmse", "target-replay"),
        "rmse_asof": ("rmse_full", "catboost_rmse", "origin-asof")}


def _validate_seed(seed: int) -> None:
    """Require one of the predeclared integer source seeds."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in (0, 1, 2):
        raise ValueError("seed must be an integer in 0, 1, 2")


def _source_inputs(source_dir: Path, cutoffs: dict, seed: int = 0) -> tuple[dict, dict[str, str], dict[str, dict[str, pd.DataFrame]]]:
    """Validate cached forecasts against their seed-specific verified archived sources."""
    _validate_seed(seed)
    read, bind = blend_research._json, blend_research._bind
    original = read(source_dir / "registration.json")
    summary = read(source_dir / "summary.json")
    lock = read(source_dir / "selection_lock.json")
    selection_summary = read(source_dir / "selection_summary.json")
    bundle = read(source_dir / "source_bundle.json") if seed == 0 else None
    if (summary.get("status") != "complete" or original.get("seed") != seed
            or original.get("cutoffs") != cutoffs or original.get("retrain_every") != 4
            or original.get("horizon") != 3 or original.get("scoring_protocol") != "autosubs_v2"
            or original.get("points_accounting") != "net_after_transfer_hits"
            or original.get("gap_rel") != 0 or original.get("chips") != "disabled"):
        raise ValueError("price source must be the completed registered matching-seed origin protocol")
    if original["feature_variants"]["mae_full"] != original["feature_variants"]["rmse_full"]:
        raise ValueError("cached MAE/RMSE ordered feature columns differ")
    if selection_summary.get("selection") != summary.get("selection"):
        raise ValueError("source selection differs from its immutable selection summary")
    hashes = original.get("code_sha256", {})
    if not hashes:
        raise ValueError("original registered source hashes are required")
    inputs: dict[str, str] = {}
    for name in ("registration.json", "summary.json", "selection_lock.json"):
        bind(inputs, source_dir / name)
    bind(inputs, source_dir / "selection_summary.json", lock["selection_summary_sha256"])
    verification_path = source_dir / "verification.json"
    if not verification_path.is_file():
        raise ValueError("original source verification.json is required")
    verification = read(verification_path)
    if seed == 0:
        if bundle.get("status") != "verified" or bundle.get("source_sha256") != hashes:
            raise ValueError("evaluated source bundle does not match original registered source hashes")
        bind(inputs, source_dir / "source_bundle.json")
        if verification.get("status") != "PASS":
            raise ValueError("original source verification must be PASS")
        bind(inputs, source_dir / "summary.json", verification["summary_file_sha256"])
        bind(inputs, source_dir / "selection_summary.json", verification["selection_summary_sha256"])
        if "evaluated_runner_sha256" in verification:
            bind(inputs, source_dir / "evaluated_sources/fpl/model/research_package.py",
                 verification["evaluated_runner_sha256"])
    else:
        if verification.get("status") != "verified" or original.get("source_archive") != "evaluated_sources":
            raise ValueError("modern source requires verified status and evaluated_sources archive")
        bind(inputs, source_dir / "summary.json", verification["summary_sha256"])
        bind(inputs, source_dir / "registration.json", verification["registration_sha256"])
        if summary.get("registration_sha256") != verification["registration_sha256"]:
            raise ValueError("modern summary registration hash differs")
        immutable = summary.get("immutable_output_sha256")
        if not immutable:
            raise ValueError("modern immutable output manifest is required")
        for relative, digest in immutable.items():
            path = Path(relative)
            bind(inputs, path if path.is_absolute() else source_dir / path, digest)
        for relative, digest in original.get("tuned_artifact_sha256", {}).items():
            bind(inputs, config.ROOT / relative, digest)
        for model in ("catboost_mae", "catboost_rmse"):
            for position in config.ONFIELD_POSITIONS:
                if original["parameters"][model][position][model].get("random_seed") != seed:
                    raise ValueError("modern member parameter seed differs from registered seed")
    for relative, digest in hashes.items():
        bind(inputs, source_dir / "evaluated_sources" / relative, digest)
    bind(inputs, verification_path)
    # Bind every available scored artifact, including nonmember diagnostics and
    # standard exports: the completed source study is an immutable input bundle.
    for directory in ("selection", "evaluation"):
        for path in sorted((source_dir / directory).glob("*")):
            if path.is_file() and (path.suffix in (".csv", ".json", ".log")):
                bind(inputs, path)
    frames = {}
    for phase, start, end in (("selection", cutoffs["selection_min_gw"], cutoffs["selection_max_gw"]),
                               ("reference", cutoffs["backtest_min_gw"], cutoffs["backtest_max_gw"])):
        frames[phase] = {}
        paths = {}
        for variant in ("mae_full", "rmse_full"):
            result = summary["selection"][variant] if phase == "selection" else summary["evaluation"][variant]["origin"]
            directory = "selection" if phase == "selection" else "evaluation"
            path = source_dir / directory / result["predictions"]
            bind(inputs, path, result["prediction_sha256"])
            exported = pd.read_csv(path)
            # The optimizer's pool is the requested phase, excluding forecasts beyond
            # its final target round even if a cached horizon export includes them.
            frame = exported.loc[exported.GW.between(start, end) & exported.origin_gw.between(start, end)].copy()
            research_package.validate_forecasts(frame, start, end, True)
            frames[phase][variant] = frame
            paths[variant] = path
        research_package.validate_paired_forecasts(paths["mae_full"], paths["rmse_full"])
    return original, inputs, frames


def catalog_diagnostics(catalog: pd.DataFrame, player_ids: list[int]) -> list[dict[str, Any]]:
    """Describe phase-union eligible/unknown counts and recorded-price age by origin.

    These counts concern lagged-price knowledge only; they do not claim deadline
    availability or that every phase-union player had a fixture at every origin.
    """
    age = catalog.origin_gw - catalog.price_observed_gw
    work = catalog.assign(age=age)
    groups = work.groupby("origin_gw", sort=True).agg(
        eligible=("player_id", "size"), age_min=("age", "min"), age_mean=("age", "mean"),
        age_median=("age", "median"), age_max=("age", "max")).reset_index()
    groups["unknown"] = len(player_ids) - groups.eligible
    return groups.to_dict(orient="records")


def _check(registration: dict, outputs: dict[str, str]) -> None:
    """Rehash all sealed inputs, catalogs, archives and outputs around scoring."""
    if research_package.source_hashes() != registration["code_sha256"]:
        raise ValueError("pipeline source changed after price registration")
    if research_package.file_hash(Path(registration["dataset"])) != registration["dataset_sha256"]:
        raise ValueError("dataset bytes changed after price registration")
    for path, digest in {**registration["cached_input_sha256"], **outputs}.items():
        if research_package.file_hash(Path(path)) != digest:
            raise ValueError(f"sealed price research artifact changed: {path}")


def run(dataset_path: str | Path, output_dir: str | Path, *, source_dir: str | Path | None = None,
        seed: int = 0, reference_only: bool = False, include_current_proxy: bool = False,
        skip_lagged_proxy: bool = False) -> dict[str, Any]:
    """Seal requested catalogs, then replay the predeclared cached price/model arms.

    No model fits or feature construction occur. Cached predictions are unchanged;
    the optimizer receives the same known-price eligibility catalog under all
    registered policies. Optional current proxies freeze reconstructed current prices
    across the horizon; they may retain legacy backfills and are not deadline prices.
    All arms start from fresh identical state.
    """
    _validate_seed(seed)
    if any(not isinstance(flag, bool) for flag in (reference_only, include_current_proxy, skip_lagged_proxy)):
        raise ValueError("research price switches must be booleans")
    started = time.perf_counter()
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    source_dir = Path(source_dir) if source_dir is not None else config.ROOT / f"experiments/research_priorities_2026-10-03/selection_seed{seed}"
    if output_dir.exists():
        raise ValueError("output directory already exists; preserve it and choose a new run")
    cutoffs = protocol_cutoffs()
    code_hashes = research_package.source_hashes()
    original, inputs, frames = _source_inputs(source_dir, cutoffs, seed=seed)
    raw = load_frozen_research_dataset(dataset_path)
    if raw.attrs["frozen_dataset_sha256"] != original["dataset_sha256"]:
        raise ValueError("frozen dataset differs from the cached source registration")
    parameters = copy.deepcopy(original["parameters"])
    columns = list(original["feature_variants"]["mae_full"])
    phases = {"selection": (cutoffs["selection_min_gw"], cutoffs["selection_max_gw"]),
              "reference": (cutoffs["backtest_min_gw"], cutoffs["backtest_max_gw"])}
    if reference_only:
        phases = {"reference": phases["reference"]}
    arms = dict(ARMS)
    comparisons = [
        ("mae_asof_vs_replay", "mae_asof", "mae_replay"),
        ("rmse_asof_vs_replay", "rmse_asof", "rmse_replay"),
        ("rmse_vs_mae_replay", "rmse_replay", "mae_replay"),
        ("rmse_vs_mae_asof", "rmse_asof", "mae_asof")]
    if include_current_proxy:
        arms.update(mae_current=("mae_full", "catboost_mae", "origin-current-proxy"),
                    rmse_current=("rmse_full", "catboost_rmse", "origin-current-proxy"))
        comparisons.extend([
            ("mae_current_vs_replay", "mae_current", "mae_replay"),
            ("mae_current_vs_asof", "mae_current", "mae_asof"),
            ("rmse_current_vs_replay", "rmse_current", "rmse_replay"),
            ("rmse_current_vs_asof", "rmse_current", "rmse_asof"),
            ("rmse_vs_mae_current", "rmse_current", "mae_current")])
    if skip_lagged_proxy:
        arms = {name: settings for name, settings in arms.items() if settings[2] != "origin-asof"}
        comparisons = [comparison for comparison in comparisons
                       if comparison[1] in arms and comparison[2] in arms]
    catalogs, diagnostics, phase_raw = {}, {}, {}
    # Prepare and validate both phase artifacts before the first solve. Catalog
    # construction sees only the phase's capped raw rows, never later target rows.
    for phase, (start, end) in phases.items():
        phase_raw[phase] = raw.loc[raw.GW_global.between(1, end)].copy()
        player_ids = sorted(frames[phase]["mae_full"].player_id.unique().tolist())
        catalog = build_origin_price_catalog(phase_raw[phase], player_ids, start, end)
        catalogs[phase] = validate_origin_price_catalog(catalog, player_ids, range(start, end + 1))
        diagnostics[phase] = catalog_diagnostics(catalogs[phase], player_ids)
    output_dir.mkdir(parents=True)
    registration = {
        "status": "registered", "promotion_eligible": False, "seed": seed,
        "hypothesis": "Budget price assumptions may change cached forecast decisions under fixed price-known eligibility.",
        "dataset": str(dataset_path.resolve()), "dataset_sha256": raw.attrs["frozen_dataset_sha256"],
        "source_run": str(source_dir.resolve()), "source_registration": original,
        "cached_input_sha256": inputs, "code_sha256": code_hashes, "source_archive": "source_archive",
        "python": platform.python_version(), "versions": experiment._library_versions(),
        "parameters": parameters, "feature_columns": columns, "windows": phases,
        "arms": arms, "retrain_every": 4, "cached_replay_only": True,
        "horizon": 3, "gap_rel": 0, "solver": "highs", "chips": "disabled",
        "scoring_protocol": "autosubs_v2", "points_accounting": "net_after_transfer_hits",
        "initial_state": "fresh_full_budget", "eligibility_basis": "phase_optimizer_union_with_known_lagged_prices",
        "comparison": {"n_boot": 10000, "block_len": 3, "seed": seed},
        "comparison_labels": [label for label, _, _ in comparisons],
        "comparisons": comparisons, "reference_only": reference_only,
        "include_current_proxy": include_current_proxy, "skip_lagged_proxy": skip_lagged_proxy,
        "evaluation_jobs": len(arms) * len(phases),
        "expected_optimal_solves": len(arms) * sum(end - start + 1 for start, end in phases.values()),
        "catalog_diagnostics": diagnostics,
        "limitations": ["Latest recorded prices are descriptive proxies, not actual verified historical deadline prices.",
                        "Archived fixtures and prices lack complete publication-time provenance; no full PIT or availability claim.",
                        "All policies restrict eligibility to known lagged prices, so these controls differ from unrestricted historical controls.",
                        "Selection and reference are repeatedly examined historical data, not fresh independent evidence.",
                        "Optional lagged-proxy exclusion reflects season-repricing budget incompatibility, not adaptive model selection.",
                        "Current-origin proxies may retain legacy forward/backfills and approximate roster/selling rules.",
                        "No fitting, tuning, production changes or promotion assessment."],
    }
    summary: dict[str, Any] = {"status": "running", "promotion_eligible": False,
                               "phases": {}, "comparisons": {}, "catalog_diagnostics": diagnostics}
    outputs: dict[str, str] = {}
    registration_path, summary_path = output_dir / "registration.json", output_dir / "summary.json"
    try:
        cnn_research._archive_sources(output_dir / "source_archive", code_hashes)
        for relative, digest in code_hashes.items():
            blend_research._bind(outputs, output_dir / "source_archive" / relative, digest)
        catalog_hashes = {}
        for phase in phases:
            path = output_dir / f"{phase}_origin_prices.csv"
            catalogs[phase].to_csv(path, index=False)
            blend_research._bind(outputs, path)
            catalog_hashes[phase] = research_package.file_hash(path)
        registration["origin_price_catalog_sha256"] = catalog_hashes
        cnn_research._write_json(registration_path, registration)
        blend_research._bind(outputs, registration_path)
        lock_path = output_dir / "catalog_lock.json"
        cnn_research._write_json(lock_path, {"registration_sha256": research_package.file_hash(registration_path),
                                            "origin_price_catalog_sha256": catalog_hashes, "windows": phases, "arms": arms})
        blend_research._bind(outputs, lock_path)
        _check(registration, outputs)
        experiment.log_result("price_research_registration", {"registration": str(registration_path), **registration},
                              {"status": "REGISTERED", "promotion_eligible": False})
        for phase, (start, end) in phases.items():
            directory = output_dir / phase
            directory.mkdir()
            summary["phases"][phase] = {}
            catalog_path = output_dir / f"{phase}_origin_prices.csv"
            for arm, (variant, model, policy) in arms.items():
                _check(registration, outputs)
                cached = frames[phase][variant]
                result = research_package.evaluate_variant(
                    phase_raw[phase], phase_raw[phase], columns, arm, parameters, start, end, True, directory, seed,
                    model_name=model, retrain_every=4, forecast_provider=lambda cached=cached: cached.copy(),
                    origin_price_catalog=catalog_path, origin_price_policy=policy)
                summary["phases"][phase][arm] = result
                if not np.isfinite(result["total_points"]):
                    raise ValueError("price research net points must be finite")
                for name, digest_key in (("predictions", "prediction_sha256"), ("squads", "squad_sha256")):
                    blend_research._bind(outputs, directory / result[name], result[digest_key])
                prefix = directory / f"{arm}_origin"
                for suffix in (".admission.json", ".result.json", ".log"):
                    blend_research._bind(outputs, prefix.with_suffix(suffix))
                replay_arm = "mae_replay" if variant == "mae_full" else "rmse_replay"
                replay = summary["phases"][phase][replay_arm]
                research_package.validate_paired_forecasts(directory / result["predictions"], directory / replay["predictions"])
                if result["prediction_sha256"] != replay["prediction_sha256"]:
                    raise ValueError("price policies must preserve identical cached forecast bytes")
                _check(registration, outputs)
                cnn_research._write_json(summary_path, summary)
            summary["comparisons"][phase] = {}
            for label, candidate_arm, control_arm in comparisons:
                _check(registration, outputs)
                candidate, control = summary["phases"][phase][candidate_arm], summary["phases"][phase][control_arm]
                research_package.validate_paired_forecasts(directory / candidate["predictions"], directory / control["predictions"])
                summary["comparisons"][phase][label] = research_package.compare_backtests.compare(
                    str(directory / candidate["squads"]), str(directory / control["squads"]), **registration["comparison"])
                _check(registration, outputs)
            phase_summary = output_dir / f"{phase}_summary.json"
            cnn_research._write_json(phase_summary, {"results": summary["phases"][phase],
                                                    "comparisons": summary["comparisons"][phase]})
            blend_research._bind(outputs, phase_summary)
        _check(registration, outputs)
        summary.update(status="complete", elapsed_seconds=time.perf_counter() - started)
        summary["immutable_output_sha256"] = dict(outputs)
        summary["registration_sha256"] = research_package.file_hash(registration_path)
        cnn_research._write_json(summary_path, summary)
        experiment.log_result("price_research_completed", {"registration": str(registration_path),
                              "summary_sha256": research_package.file_hash(summary_path)},
                              {"status": "COMPLETE", "promotion_eligible": False, "comparisons": summary["comparisons"]})
        _check(registration, outputs)
        print("Price proxy replay completed: registered windows, unchanged cached forecasts, paired net-points reports preserved.", flush=True)
    except BaseException as exc:
        summary.update(status="incomplete", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter() - started)
        summary["immutable_output_sha256"] = dict(outputs)
        summary["registration_sha256"] = outputs.get(str(registration_path.resolve()))
        cnn_research._write_json(summary_path, summary)
        experiment.log_result("price_research_incomplete", {"registration": str(registration_path)},
                              {"status": "INCOMPLETE", "error": summary["error"], "promotion_eligible": False})
        raise
    return summary


def main(argv: list[str] | None = None) -> None:
    """Replay a new price proxy study from portable cached source artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.FROZEN_RESEARCH_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--include-current-proxy", action="store_true")
    parser.add_argument("--skip-lagged-proxy", action="store_true")
    args = parser.parse_args(argv)
    options: dict[str, Any] = {"source_dir": args.source_dir}
    if args.seed:
        options["seed"] = args.seed
    for name in ("reference_only", "include_current_proxy", "skip_lagged_proxy"):
        if getattr(args, name):
            options[name] = True
    run(args.dataset, args.output_dir, **options)


if __name__ == "__main__":
    main()
