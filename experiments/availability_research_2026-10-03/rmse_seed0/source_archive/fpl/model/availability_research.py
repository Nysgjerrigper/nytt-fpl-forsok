"""Registered offline availability challenger, with explicit archive coverage gaps.

This descriptive historical study does not establish complete deadline rosters,
point-in-time prices, production promotion, or modern transfer/selling policies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd

from fpl import config, experiment
from fpl.data import deadline_snapshots, provenance
from fpl.data.origin_prices import build_origin_price_catalog, validate_origin_price_catalog
from fpl.data.research import load_research_features
from fpl.model import blend_research, cnn_research, research_package
from fpl.model.deadline_availability import add_availability_features, availability_columns
from fpl.model.moe_tournament import protocol_cutoffs
from fpl.run_week import build_live_snapshot

ARMS = ("mae_full", "mae_availability")


def _check(registration: dict, outputs: dict[str, str]) -> None:
    """Rehash admitted dataset, sources, tuning artifacts and immutable study bytes."""
    if research_package.source_hashes() != registration["code_sha256"]:
        raise ValueError("Availability pipeline source changed after registration")
    if research_package.file_hash(Path(registration["dataset"])) != registration["dataset_sha256"]:
        raise ValueError("Availability dataset changed after admission")
    for name, digest in {**registration["input_sha256"], **outputs}.items():
        if research_package.file_hash(Path(name)) != digest:
            raise ValueError(f"Sealed availability artifact changed: {name}")


def _rosters(audit_dir: Path, cutoffs: dict) -> tuple[pd.DataFrame, list[dict], dict[str, str]]:
    """Admit each origin once; only declared unavailable snapshots become missing."""
    inputs: dict[str, str] = {}
    frames, coverage = [], []
    for proof_name, index_name, lower, upper in (
        ("discovery_wayback_verification.json", "discovery_wayback_index.json", 1, cutoffs["discovery_max_gw"]),
        ("wayback_verification.json", "wayback_index.json", cutoffs["selection_min_gw"], cutoffs["backtest_max_gw"]),
    ):
        proof_path, index_path = audit_dir / proof_name, audit_dir / "inventory" / index_name
        blend_research._bind(inputs, proof_path)
        proof_digest = inputs[str(proof_path.resolve())]
        proof = json.loads(proof_path.read_bytes())
        blend_research._bind(inputs, index_path, proof["index"]["sha256"])
        for capture in proof["captures"]:
            if lower <= deadline_snapshots._origin(capture) <= upper:
                for path_key, digest_key in (("original_payload", "original_payload_sha256"), ("manifest", "manifest_sha256")):
                    path = deadline_snapshots._path(proof_path.parent, capture[path_key])
                    blend_research._bind(inputs, path, capture[digest_key])
        for origin in range(lower, upper + 1):
            try:
                frame = deadline_snapshots.load_deadline_rosters(proof_path, index_path, proof_digest, [origin])
            except deadline_snapshots.SnapshotUnavailableError as exc:
                coverage.append({"origin_gw": origin, "available": False, "reason": exc.reason,
                                 "roster_players": 0, "excluded_nonplayer_count": 0})
            else:
                frames.append(frame)
                coverage.append({"origin_gw": origin, "available": True, "reason": None,
                                 "roster_players": len(frame),
                                 "excluded_nonplayer_count": frame.attrs.get("excluded_nonplayer_count", 0)})
    roster = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=deadline_snapshots.COLUMNS)
    if roster.duplicated(["player_id", "origin_gw"]).any():
        raise ValueError("Availability roster has duplicate player/origin identity")
    return roster, coverage, inputs


def run(dataset: str | Path, output_dir: str | Path, *, audit_dir: str | Path, seed: int = 0,
        model_name: str = "catboost_mae") -> dict[str, Any]:
    """Register an equal-capacity loss-specific challenger and advance on strict earlier net gain.

    Missing admitted archive signals stay NaN while every historical outcome row
    is retained. Origin features freeze the exact origin's dated availability.
    Origin arms share known-prior-price eligibility and current-price horizon proxies;
    standard reference forecasts use target-price replay as historical diagnostics.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in (0, 1, 2):
        raise ValueError("seed must be an integer in 0, 1, 2")
    if not isinstance(model_name, str) or model_name not in ("catboost_mae", "catboost_rmse"):
        raise ValueError("model_name must be catboost_mae or catboost_rmse")
    prefix = "mae" if model_name == "catboost_mae" else "rmse"
    control_arm, candidate_arm = f"{prefix}_full", f"{prefix}_availability"
    arms = (control_arm, candidate_arm)
    dataset, output_dir, audit_dir = Path(dataset), Path(output_dir), Path(audit_dir)
    if output_dir.exists():
        raise ValueError("Output directory already exists; choose a new availability study")
    started = time.perf_counter()
    outputs: dict[str, str] = {}
    summary: dict[str, Any] = {"status": "running", "promotion_eligible": False, "selection": {}, "evaluation": {}, "comparisons": {}}
    output_dir.mkdir(parents=True)
    registration_path, summary_path = output_dir / "registration.json", output_dir / "summary.json"
    try:
        cutoffs = protocol_cutoffs()
        code_hashes = research_package.source_hashes()
        dataset_digest = research_package.file_hash(dataset)
        rosters, coverage, inputs = _rosters(audit_dir, cutoffs)
        blend_research._bind(inputs, provenance.FROZEN_MANIFEST_PATH)
        parameters, tuned_hashes = research_package.resolve_parameters("selection", seed)
        parameters = {model_name: parameters[model_name]}
        for relative, digest in tuned_hashes.items():
            blend_research._bind(inputs, config.ROOT / relative, digest)
        windows = {"selection": (cutoffs["selection_min_gw"], cutoffs["selection_max_gw"]),
                   "reference": (cutoffs["backtest_min_gw"], cutoffs["backtest_max_gw"])}
        prepared, catalogs, masks, coverage_counts = {}, {}, [], {}
        base_columns = None
        extra = availability_columns()
        for phase, (start, end) in windows.items():
            raw, frame, columns = load_research_features(end, dataset)
            if raw.attrs.get("frozen_dataset_sha256") != dataset_digest or raw.GW_global.max() > end or frame.GW_global.max() > end:
                raise ValueError("Availability features violate frozen provenance or phase cutoff")
            if not columns or len(set(columns)) != len(columns) or 'total_points' in columns or set(columns) & set(extra):
                raise ValueError("Availability base feature schema is invalid")
            if base_columns is not None and columns != base_columns:
                raise ValueError("Availability phase ordered base columns differ")
            base_columns = list(columns)
            extended = add_availability_features(frame, rosters)
            if len(extended) != len(frame) or not extended.index.equals(frame.index):
                raise ValueError("Availability transform dropped or reordered outcome rows")
            prepared[phase] = (raw, extended)
            player_ids = sorted(frame.loc[frame.GW_global.between(start, end), "player_id"].unique().tolist())
            catalogs[phase] = validate_origin_price_catalog(build_origin_price_catalog(raw, player_ids, start, end),
                                                            player_ids, range(start, end + 1))
            keys = [column for column in ("player_id", "GW_global", "fixture") if column in frame]
            mask = frame[keys].copy()
            mask["phase"] = phase
            mask["availability_known"] = extended.archive_status_a.notna().to_numpy()
            masks.append(mask)
            evaluation_mask = frame.GW_global.between(start, end)
            coverage_counts[phase] = {"fixture_rows": len(frame), "covered_fixture_rows": int(mask.availability_known.sum()),
                                      "evaluation_fixture_rows": int(evaluation_mask.sum()),
                                      "evaluation_covered_fixture_rows": int((evaluation_mask.to_numpy() & mask.availability_known.to_numpy()).sum()),
                                      "evaluation_missing_fixture_rows": int((evaluation_mask.to_numpy() & ~mask.availability_known.to_numpy()).sum()),
                                      "missing_fixture_rows": int((~mask.availability_known).sum())}
        variants = {control_arm: base_columns, candidate_arm: base_columns + extra}
        registration = {"status": "registered", "seed": seed, "model_name": model_name, "promotion_eligible": False,
                        "parameter_protocol": f"Same registered seed-zero {model_name} tuned capacity; requested seed overrides refitting only, no new HPO.",
                        "hypothesis": f"Pre-deadline availability may improve equal-capacity {prefix.upper()} decisions despite incomplete archive coverage.",
                        "dataset": str(dataset.resolve()), "dataset_sha256": dataset_digest,
                        "code_sha256": code_hashes, "source_archive": "source_archive", "input_sha256": inputs,
                        "tuned_artifact_sha256": tuned_hashes, "parameters": parameters, "feature_variants": variants,
                        "cutoffs": cutoffs, "windows": windows, "arms": list(arms), "retrain_every": 4,
                        "horizon": 3, "chips": "disabled", "gap_rel": 0, "solver": "highs", "initial_budget": 1000,
                        "scoring_protocol": "autosubs_v2", "points_accounting": "net_after_transfer_hits",
                        "origin_price_policy": "origin-current-proxy", "standard_price_policy": "target-replay",
                        "extra_player_features": extra, "max_capture_age_hours": config.DEADLINE_SNAPSHOT_MAX_AGE_HOURS,
                        "comparison_labels": ["selection_origin", "reference_standard", "reference_origin"], "coverage": coverage, "coverage_counts": coverage_counts,
                        "selection_rule": "Candidate advances only if selection-origin net points strictly exceed control; ties reject.",
                        "comparison": {"n_boot": 10000, "block_len": 3, "seed": seed},
                        "python": platform.python_version(), "versions": experiment._library_versions(),
                        "limitations": ["Incomplete temporal capture coverage; absent signals stay NaN and all outcome rows remain.",
                                        "Status and prices describe selected crawl time, not the deadline itself.",
                                        "Selected captures are not asserted to be the latest eligible captures.",
                                        "Forecast pool remains history-based, not a registered cold-start roster universe.",
                                        "No complete historical roster or publication-time/PIT price claim.",
                                        "Current-price proxies retain reconstructed/backfilled prices and old FT/selling policies.",
                                        "Repeatedly examined historical selection/reference; standard forecasts are diagnostics.",
                                        "Seeds and loss objectives share historical outcomes and are not independent pooled samples.",
                                        "No new hyperparameter search, family selection changes or production promotion.",
                                        "No production change or promotion assessment."]}
        cnn_research._archive_sources(output_dir / 'source_archive', code_hashes)
        for relative, digest in code_hashes.items():
            blend_research._bind(outputs, output_dir / 'source_archive' / relative, digest)
        rosters.to_csv(output_dir / 'rosters.csv', index=False)
        pd.concat(masks, ignore_index=True).to_csv(output_dir / 'coverage_mask.csv', index=False)
        cnn_research._write_json(output_dir / 'coverage.json', {"origins": coverage, "counts": coverage_counts})
        for name in ('rosters.csv','coverage_mask.csv','coverage.json'):
            blend_research._bind(outputs, output_dir / name)
        catalog_hashes = {}
        for phase, catalog in catalogs.items():
            path = output_dir / f'{phase}_origin_prices.csv'
            catalog.to_csv(path, index=False)
            blend_research._bind(outputs, path)
            catalog_hashes[phase] = research_package.file_hash(path)
        registration.update(origin_price_catalog_sha256=catalog_hashes,
                            roster_sha256=research_package.file_hash(output_dir/'rosters.csv'),
                            coverage_mask_sha256=research_package.file_hash(output_dir/'coverage_mask.csv'),
                            coverage_sha256=research_package.file_hash(output_dir/'coverage.json'))
        cnn_research._write_json(registration_path, registration)
        blend_research._bind(outputs, registration_path)
        _check(registration, outputs)
        experiment.log_result('availability_research_registration', registration, {'status':'REGISTERED','promotion_eligible':False})

        def snapshot(hist_raw: pd.DataFrame, origin: int) -> pd.DataFrame:
            if not hist_raw.empty and hist_raw.GW_global.max() >= origin:
                raise ValueError('Availability snapshot history must be strictly before origin')
            base = build_live_snapshot(hist_raw)
            base['GW_global'] = origin
            return add_availability_features(base, rosters)

        def score(phase: str, arm: str, origin: bool) -> dict:
            _check(registration, outputs)
            raw, frame = prepared[phase]
            directory = output_dir / phase
            directory.mkdir(exist_ok=True)
            kwargs = {}
            if origin:
                kwargs.update(origin_price_catalog=output_dir/f'{phase}_origin_prices.csv', origin_price_policy='origin-current-proxy')
                if arm == candidate_arm:
                    kwargs.update(origin_snapshot_builder=snapshot, extra_player_features=extra)
            result = research_package.evaluate_variant(raw, frame, variants[arm], arm, parameters,
                       *windows[phase], origin, directory, seed, model_name=model_name, retrain_every=4, **kwargs)
            if not np.isfinite(result['total_points']):
                raise ValueError('Availability net points must be finite')
            for name, digest_key in (('predictions','prediction_sha256'),('squads','squad_sha256')):
                blend_research._bind(outputs, directory/result[name], result[digest_key])
            prefix = directory / f'{arm}_{"origin" if origin else "standard"}'
            for suffix in ('.admission.json','.result.json','.log'):
                blend_research._bind(outputs,prefix.with_suffix(suffix))
            _check(registration,outputs)
            return result

        def compare(phase: str, candidate: dict, control: dict) -> dict:
            _check(registration,outputs)
            directory = output_dir/phase
            research_package.validate_paired_forecasts(directory/candidate['predictions'],directory/control['predictions'])
            report = research_package.compare_backtests.compare(str(directory/candidate['squads']),str(directory/control['squads']),**registration['comparison'])
            _check(registration,outputs)
            return report

        for arm in arms:
            summary['selection'][arm] = score('selection',arm,True)
            cnn_research._write_json(summary_path,summary)
        summary['comparisons']['selection_origin'] = compare('selection',summary['selection'][candidate_arm],summary['selection'][control_arm])
        advanced = summary['selection'][candidate_arm]['total_points'] > summary['selection'][control_arm]['total_points']
        selection_path = output_dir/'selection_summary.json'
        cnn_research._write_json(selection_path,{"selection":summary['selection'],"comparison":summary['comparisons']['selection_origin'],"advanced":advanced})
        blend_research._bind(outputs,selection_path)
        cnn_research._write_json(output_dir/'selection_lock.json',{"selection_summary_sha256":research_package.file_hash(selection_path),
                      "advanced":advanced,"finalists":list(arms) if advanced else [],"feature_variants":variants,"parameters":parameters})
        blend_research._bind(outputs,output_dir/'selection_lock.json')
        _check(registration,outputs)
        if advanced:
            for arm in arms:
                summary['evaluation'][arm] = {}
                for origin in (False,True):
                    summary['evaluation'][arm]['origin' if origin else 'standard'] = score('reference',arm,origin)
                    cnn_research._write_json(summary_path,summary)
            for mode in ('standard','origin'):
                summary['comparisons'][f'reference_{mode}'] = compare('reference',summary['evaluation'][candidate_arm][mode],summary['evaluation'][control_arm][mode])
        _check(registration,outputs)
        summary.update(status='complete' if advanced else 'selection_rejected',advanced=advanced,
                       registration_sha256=research_package.file_hash(registration_path),
                       immutable_output_sha256=dict(outputs),elapsed_seconds=time.perf_counter()-started)
        cnn_research._write_json(summary_path,summary)
        experiment.log_result('availability_research_completed',{'registration':str(registration_path),'summary_sha256':research_package.file_hash(summary_path)},
                              {'status':summary['status'].upper(),'promotion_eligible':False,'comparisons':summary['comparisons']})
        _check(registration,outputs)
        return summary
    except BaseException as exc:
        summary.update(status='incomplete',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-started,
                       immutable_output_sha256=dict(outputs),registration_sha256=outputs.get(str(registration_path.resolve())))
        cnn_research._write_json(summary_path,summary)
        experiment.log_result('availability_research_incomplete',{'registration':str(registration_path)}, {'status':'INCOMPLETE','error':summary['error'],'promotion_eligible':False})
        raise


def main(argv: list[str] | None = None) -> None:
    """Run a fresh offline availability study from explicit preserved audit artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=config.FROZEN_RESEARCH_DATASET_PATH)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=(0,1,2),default=0)
    parser.add_argument('--model-name',choices=('catboost_mae','catboost_rmse'),default='catboost_mae')
    args=parser.parse_args(argv)
    options: dict[str, Any] = {"audit_dir": args.audit_dir}
    if args.seed:
        options["seed"] = args.seed
    if args.model_name != "catboost_mae":
        options["model_name"] = args.model_name
    run(args.dataset,args.output_dir,**options)


if __name__ == '__main__':
    main()
