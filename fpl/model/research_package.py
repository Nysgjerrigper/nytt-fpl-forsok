"""Registered, research-only loss/feature comparisons with preserved MILP evidence.

This is an exploratory family, not the registered MoE promotion protocol. Selection
uses origin MILP points before the reference window; production is never changed.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd

from fpl import config, experiment, features
from fpl.data import provenance
from fpl.data.research import load_research_features
from fpl.milp import compare_backtests, optimize
from fpl.model import metrics, models, predict
from fpl.model.moe_tournament import protocol_cutoffs
from fpl.model.tuning import load_validated_params


def file_hash(path: Path) -> str:
    """Hash preserved bytes, including parameters and generated predictions."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    """Bind the experiment to every pipeline source, including uncommitted edits."""
    return {str(p.relative_to(config.ROOT)): file_hash(p)
            for p in sorted((config.ROOT / 'fpl').rglob('*.py'))}


def feature_variants(columns: list[str]) -> dict[str, list[str]]:
    """Predeclare loss controls and one-at-a-time removals from existing features.

    Hyperparameters stay fixed within the MAE ablation family. Removing minutes
    includes every lagged minutes statistic as well as the start/60-minute rates.
    """
    groups = {
        'fixture': set(features.FIXTURE_FEATURES),
        'minutes': {c for c in columns if c.startswith('minutes_')} | set(features.MINUTES_FEATURES),
        'opponent': set(features.OPPONENT_FEATURES),
        'xp': set(features.XP_FEATURES),
    }
    if not columns or len(columns) != len(set(columns)) or features.TARGET_COL in columns:
        raise ValueError('research features must be unique, nonempty and exclude the target')
    variants = {'mae_full': list(columns), 'rmse_full': list(columns)}
    for family in config.RESEARCH_ABLATION_FAMILIES:
        removed = groups[family] & set(columns)
        if not removed:
            raise ValueError(f'feature family {family} is absent; cannot claim an ablation')
        variants[f'mae_without_{family}'] = [c for c in columns if c not in removed]
    return variants


def resolve_parameters(mode: str, seed: int) -> tuple[dict[str, dict[str, dict]], dict[str, str]]:
    """Resolve exact four-position parameters without any ambient artifact fallback.

    Selection mode validates the preserved seed-0 studies. A seed override changes
    refitting only and is explicitly not a new tuned study or promotion evidence.
    The five-thread runtime override is recorded in the resolved parameters.
    """
    resolved, artifact_hashes = {}, {}
    if mode not in ('defaults', 'selection') or type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError('parameters must be defaults/selection and seed must be 0/1/2')
    for name in ('catboost_mae', 'catboost_rmse'):
        resolved[name] = {}
        for position in config.ONFIELD_POSITIONS:
            if mode == 'selection':
                path = config.MODELS_DIR / f'tuned_params_{position}_{name}.json'
                payload = load_validated_params(path, position=position, model_name=name,
                                                seed=0, stage='selection',
                                                max_train_gw=protocol_cutoffs()['discovery_max_gw'])
                params = {k: v for k, v in payload.items() if not k.startswith('_')}
                artifact_hashes[str(path.relative_to(config.ROOT))] = file_hash(path)
            else:
                params = models.EXPERT_SPECS[name].default_params(seed)
            params = {**params, 'random_seed': seed,
                      'thread_count': config.RESEARCH_MODEL_THREADS, 'allow_writing_files': False}
            resolved[name][position] = {name: params}
    return resolved, artifact_hashes


def prediction_diagnostics(predictions: pd.DataFrame, raw: pd.DataFrame) -> list[dict[str, Any]]:
    """Score fixture-summed rounds, using strictly earlier training-round MASE scales.

    Origin diagnostics use only each origin's current round, never counting the
    overlapping future horizon copies as independent observations.
    """
    current = (predictions.loc[predictions.GW == predictions.origin_gw]
               if 'origin_gw' in predictions else predictions)
    rounds = current.groupby(['player_id', 'position', 'GW'], as_index=False).agg(
        actual_total_points=('actual_total_points', 'sum'),
        prediction=('predicted_total_points', 'sum'))
    reports = []
    for position, group in rounds.groupby('position'):
        errors = []
        for gw, week in group.groupby('GW'):
            train = raw.loc[(raw.position == position) & (raw.GW_global < gw)]
            train = train.groupby(['player_id', 'GW_global'], as_index=False).total_points.sum()
            scale = metrics.naive_lag1_scale(train)
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('invalid training-round MASE scale')
            errors.extend((np.abs(week.actual_total_points - week.prediction) / scale).tolist())
        y, pred = group.actual_total_points, group.prediction
        reports.append({'position': position, 'rounds': len(group), 'mae': metrics.mae(y, pred),
                        'mase': float(np.mean(errors)), 'rmse': metrics.rmse(y, pred),
                        'bias': metrics.bias(y, pred), 'total_calibration': metrics.total_calibration(y, pred),
                        'spearman': metrics.spearman_by_group(y, pred, group.GW),
                        'top1_capture': metrics.top1_capture(group, 'GW', 'actual_total_points', 'prediction')})
    return reports


def validate_forecasts(frame: pd.DataFrame, start: int, end: int, origin: bool) -> None:
    """Reject ambiguous fixtures, missing rounds and nonfinite corrected-scoring inputs."""
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError('forecast frame must be nonempty with unique columns')
    required = {'player_id', 'position', 'GW', 'predicted_total_points', 'actual_total_points', 'minutes', 'value'}
    if required - set(frame):
        raise ValueError(f'missing forecast columns: {sorted(required - set(frame))}')
    if origin != ('origin_gw' in frame):
        raise ValueError('mixed or missing origin semantics')
    identities = ['player_id', 'GW'] + ([ 'origin_gw'] if origin else []) + (['fixture'] if 'fixture' in frame else [])
    for col in identities + ['predicted_total_points', 'actual_total_points', 'minutes', 'value']:
        values = frame[col]
        if not pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype) or values.isna().any() or pd.api.types.is_complex_dtype(values.dtype) or not np.isfinite(values).all():
            raise ValueError(f'nonfinite or nonnumeric research forecast {col}')
        if col in identities and ((values <= 0).any() or (values % 1 != 0).any()):
            raise ValueError(f'forecast {col} requires positive integer identities')
    if not frame.position.isin(config.ONFIELD_POSITIONS).all():
        raise ValueError('invalid forecast position')
    if frame.groupby(['player_id', 'GW'], dropna=False).position.nunique(dropna=False).gt(1).any():
        raise ValueError('conflicting player-round positions')
    if (frame.minutes < 0).any() or set(frame.GW) != set(range(start, end + 1)):
        raise ValueError('negative minutes or incomplete forecast rounds')
    if (frame.value <= 0).any():
        raise ValueError('forecast value must be positive')
    keys = ['player_id', 'GW'] + (['origin_gw'] if origin else [])
    if 'fixture' in frame:
        if frame.duplicated(keys + ['fixture']).any():
            raise ValueError('missing or duplicate fixture identities')
    elif frame.duplicated(keys).any():
        raise ValueError('multiple fixture rows require fixture identities')
    if origin:
        expected = {(o, gw) for o in range(start, end + 1)
                    for gw in range(o, min(o + 3, end + 1))}
        if set(zip(frame.origin_gw, frame.GW)) != expected:
            raise ValueError('incomplete or ambiguous origin horizon coverage')
        fixture_keys = ['player_id', 'GW'] + (['fixture'] if 'fixture' in frame else [])
        metadata = ['actual_total_points', 'minutes', 'value'] + [c for c in frame.columns
                    if c == 'name' or c.endswith('_name') or 'team' in c or 'opponent' in c or 'home' in c]
        if frame.groupby(fixture_keys, dropna=False)[metadata].nunique(dropna=False).gt(1).any().any():
            raise ValueError('conflicting fixture scoring or metadata across origins')


def validate_paired_forecasts(candidate_path: Path, control_path: Path) -> None:
    """Require identical exported pools, outcomes, prices and fixture metadata.

    Prediction values and row order may differ; every other exported input must
    match. Individual forecast admission precedes this paired-input check.
    """
    candidate, control = pd.read_csv(candidate_path), pd.read_csv(control_path)
    required = {'player_id', 'position', 'GW', 'predicted_total_points',
                'actual_total_points', 'minutes', 'value'}
    if set(candidate) != set(control) or required - set(candidate):
        raise ValueError('paired forecast schemas differ or lack required scoring inputs')
    keys = ['player_id', 'GW'] + [c for c in ('origin_gw', 'fixture') if c in candidate]
    columns = sorted(set(candidate) - {'predicted_total_points'})
    if candidate.duplicated(keys).any() or control.duplicated(keys).any():
        raise ValueError('paired forecasts contain duplicate fixture identities')
    a = candidate.sort_values(keys)[columns].reset_index(drop=True)
    b = control.sort_values(keys)[columns].reset_index(drop=True)
    if not a.equals(b):
        raise ValueError('paired forecasts differ in universe, identity or scoring/price metadata')


def evaluate_variant(raw: pd.DataFrame, frame: pd.DataFrame, columns: list[str],
                     variant: str, parameters: dict, start: int, end: int,
                     origin: bool, directory: Path, seed: int, *, model_name: str | None = None,
                     retrain_every: int = 4, snapshot_builder=None, extra_player_features=(),
                     forecast_provider=None, origin_price_catalog: Path | None = None,
                     origin_price_policy: str = 'target-replay', origin_snapshot_builder=None) -> dict:
    """Preserve predictions before exact-optimal, chip-disabled corrected MILP scoring.

    A registered cached replay can provide ``forecast_provider()`` instead of
    refitting. The caller binds its source prediction bytes and settings before
    evaluation; admission and exact corrected scoring remain identical.
    """
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError('seed must be an integer in 0, 1, 2')
    name = model_name or ('catboost_rmse' if variant == 'rmse_full' else 'catboost_mae')
    if origin_price_policy not in ('target-replay', 'origin-asof', 'origin-current-proxy'):
        raise ValueError('unknown origin price policy')
    if origin_price_catalog is not None and not origin:
        raise ValueError('origin price research requires origin forecasts')
    if origin_price_policy != 'target-replay' and origin_price_catalog is None:
        raise ValueError(f'{origin_price_policy} requires an origin price catalog')
    price_sha = file_hash(origin_price_catalog) if origin_price_catalog is not None else None
    prefix = directory / f'{variant}_{"origin" if origin else "standard"}'
    if any(prefix.with_suffix(suffix).exists() for suffix in ('.predictions.csv', '.squads.csv', '.admission.json', '.result.json', '.log')):
        raise ValueError('evaluation prefix artifacts already exist; preserve recorded evidence')
    predictions_path, squads_path = prefix.with_suffix('.predictions.csv'), prefix.with_suffix('.squads.csv')
    log = StringIO()
    started = time.perf_counter()
    with redirect_stdout(log):
        kwargs = dict(retrain_every=retrain_every, weight_strategy=f'single:{name}',
                      params_by_pos=parameters[name], seed=seed)
        if forecast_provider is not None:
            if not callable(forecast_provider):
                raise ValueError('cached forecast provider must be callable')
            forecasts = forecast_provider()
        elif origin:
            adapter_kwargs = (dict(snapshot_builder=snapshot_builder, extra_player_features=extra_player_features)
                              if snapshot_builder is not None or extra_player_features or origin_snapshot_builder is not None else {})
            if origin_snapshot_builder is not None:
                adapter_kwargs['origin_snapshot_builder'] = origin_snapshot_builder
            forecasts = predict.origin_based_predictions(frame, raw, columns, start, end,
                                                         horizon=3, **kwargs, **adapter_kwargs)
        else:
            forecasts = predict.walk_forward_predictions(frame, columns, start, end, **kwargs)
    fit_seconds = time.perf_counter() - started
    validate_forecasts(forecasts, start, end, origin)
    forecasts.to_csv(predictions_path, index=False)
    prediction_sha = file_hash(predictions_path)
    admission = {'prediction_sha256': prediction_sha, 'variant': variant, 'origin': origin,
                 'feature_columns': columns, 'parameters': parameters[name], 'seed': seed,
                 'start_gw': start, 'end_gw': end, 'horizon': 3, 'retrain_every': retrain_every,
                 'extra_player_features': list(extra_player_features),
                 'dated_origin_snapshot': origin_snapshot_builder is not None,
                 'cached_replay': forecast_provider is not None}
    if price_sha is not None:
        admission.update(origin_price_policy=origin_price_policy, origin_price_catalog_sha256=price_sha)
    prefix.with_suffix('.admission.json').write_text(json.dumps(admission, indent=2, sort_keys=True))
    diagnostic = prediction_diagnostics(forecasts, raw)
    price_args = (['--origin-price-catalog', str(origin_price_catalog), '--origin-price-policy', origin_price_policy]
                  if origin_price_catalog is not None else [])
    if price_sha is not None and file_hash(origin_price_catalog) != price_sha:
        raise ValueError('origin price catalog changed before evaluation')
    args = optimize.parse_args(['--predictions-csv', str(predictions_path), '--start-gw', str(start),
                                '--max-gw', str(end), '--horizon', '3', '--scoring-mode', 'corrected',
                                '--solver', 'highs', '--gap-rel', '0', '--output', str(squads_path)] + price_args)
    solve_start = time.perf_counter()
    with redirect_stdout(log):
        optimize.run(args)
    prefix.with_suffix('.log').write_text(log.getvalue())
    solve_seconds = time.perf_counter() - solve_start
    if file_hash(predictions_path) != prediction_sha:
        raise ValueError('predictions changed during evaluation')
    if price_sha is not None and file_hash(origin_price_catalog) != price_sha:
        raise ValueError('origin price catalog changed during evaluation')
    if log.getvalue().count('Status: Optimal (') != end - start + 1:
        raise ValueError('not every research solve was proven optimal')
    squads = pd.read_csv(squads_path)
    if not {'solver_status', 'solver_solution_status'}.issubset(squads.columns) or not squads.solver_status.eq('Optimal').all() or not squads.solver_solution_status.eq('Optimal Solution Found').all():
        raise ValueError('every research solve requires structured proven-optimal status')
    if set(squads.gameweek) != set(range(start, end + 1)) or not (squads.scoring_protocol == 'autosubs_v2').all():
        raise ValueError('incomplete or inconsistent corrected scoring')
    # The legacy scorer exports gross points. Research decisions pay their hits;
    # keep both columns so no historical gross score is silently relabelled.
    squads['gross_actual_total_points'] = squads['actual_total_points']
    squads['actual_total_points'] = squads['gross_actual_total_points'] - 4 * squads['alpha']
    squads['points_accounting'] = 'net_after_transfer_hits'
    squads.to_csv(squads_path, index=False)
    result = {'variant': variant, 'origin': origin, 'total_points': float(squads.actual_total_points.sum()),
              'fit_seconds': fit_seconds, 'milp_seconds': solve_seconds, 'diagnostics': diagnostic,
              'predictions': predictions_path.name, 'prediction_sha256': prediction_sha,
              'squads': squads_path.name, 'squad_sha256': file_hash(squads_path)}
    result['gross_points'] = float(squads.gross_actual_total_points.sum())
    result['transfer_hit_points'] = float(4 * squads.alpha.sum())
    if price_sha is not None:
        result.update(origin_price_policy=origin_price_policy, origin_price_catalog_sha256=price_sha)
    prefix.with_suffix('.result.json').write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f'{directory.name}/{variant}/{"origin" if origin else "standard"}: '
          f'{result["total_points"]:.0f} points; fits {fit_seconds:.1f}s, MILP {solve_seconds:.1f}s', flush=True)
    return result


def select_ablation(selection: dict[str, dict]) -> str | None:
    """Lock the best ablation by earlier origin points; retain full features on ties."""
    control = selection['mae_full']['total_points']
    candidates = sorted((name for name in selection if name.startswith('mae_without_')),
                        key=lambda name: (-selection[name]['total_points'], name))
    return candidates[0] if candidates and selection[candidates[0]]['total_points'] > control else None


def run_package(dataset: Path, output_dir: Path, *, parameter_mode: str = 'selection', seed: int = 0) -> dict:
    """Register, select and evaluate one exploratory loss/ablation family, fail closed.

    Existing outputs are never overwritten. Reference-window findings cannot change
    the frozen finalist set. Seed-0 studies are reused explicitly, never claimed to
    reproduce the missing historical production artifact namespace.
    """
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError('seed must be an integer in 0, 1, 2')
    if output_dir.exists():
        raise ValueError('output directory already exists; preserve it and choose a new run')
    code_digest = source_hashes()
    cutoffs = protocol_cutoffs()
    manifest_path = Path(provenance.FROZEN_MANIFEST_PATH).resolve()
    manifest_digest = file_hash(manifest_path)
    dataset_digest = file_hash(dataset)
    sel_raw, sel_frame, columns = load_research_features(cutoffs['selection_max_gw'], dataset)
    if file_hash(manifest_path) != manifest_digest:
        raise ValueError('frozen manifest changed during selection admission')
    if file_hash(dataset) != dataset_digest or sel_raw.attrs.get('frozen_dataset_sha256') != dataset_digest:
        raise ValueError('selection frozen dataset provenance differs from admitted bytes')
    variants = feature_variants(columns)
    parameters, artifact_hashes = resolve_parameters(parameter_mode, seed)
    output_dir.mkdir(parents=True)
    registration = {
        'status': 'registered', 'promotion_eligible': False, 'issue': 13,
        'hypothesis': 'Loss objective and existing feature families may affect realized decisions differently from forecast errors.',
        'control': 'new explicit catboost_mae refit; not historical 2088/1901 replay',
        'parameter_mode': parameter_mode, 'seed': seed, 'parameters': parameters,
        'tuned_artifact_sha256': artifact_hashes, 'cutoffs': cutoffs, 'feature_variants': variants,
        'dataset': str(dataset.resolve()), 'dataset_sha256': sel_raw.attrs['frozen_dataset_sha256'],
        'code_sha256': code_digest, 'source_archive': 'evaluated_sources',
        'manifest_path': str(manifest_path), 'manifest_sha256': manifest_digest, 'feature_columns': list(columns),
        'python': platform.python_version(),
        'versions': experiment._library_versions(), 'horizon': 3, 'retrain_every': 4,
        'scoring_protocol': 'autosubs_v2', 'points_accounting': 'net_after_transfer_hits',
        'gap_rel': 0, 'chips': 'disabled',
        'selection_rule': 'Highest selection-origin MILP total among MAE ablations; full control retained on ties. RMSE is predeclared.',
        'limitations': ['Repeatedly examined historical reference, not a fresh holdout.',
                       'No GW232+ admission, no spent-window selection.',
                       'Archive fixtures/prices lack full historical publication-time provenance.',
                       'Only the explicitly requested refit seed is evaluated; no promotion assessment.'],
    }
    path = output_dir / 'registration.json'
    summary: dict[str, Any] = {'status': 'running', 'selection': {}, 'evaluation': {}, 'comparisons': {}}
    summary_path = output_dir / 'summary.json'
    immutable_outputs: dict[Path, str] = {}
    registration_digest: str | None = None

    def check_file(file: Path, expected: str, label: str) -> None:
        try:
            actual = file_hash(file)
        except OSError as exc:
            raise ValueError(f'{label} is missing or unreadable: {file}') from exc
        if actual != expected:
            raise ValueError(f'{label} changed after registration: {file}')

    def check_provenance() -> None:
        if source_hashes() != registration['code_sha256']:
            raise ValueError('pipeline source changed after registration')
        check_file(dataset, registration['dataset_sha256'], 'frozen dataset')
        check_file(manifest_path, registration['manifest_sha256'], 'frozen manifest')
        for name, digest in artifact_hashes.items():
            check_file(config.ROOT / name, digest, 'pinned tuned artifact')
        if registration_digest is not None:
            check_file(path, registration_digest, 'registration file')
        for file, digest in immutable_outputs.items():
            check_file(file, digest, 'immutable research output')

    def bind_result(result: dict, directory: Path) -> None:
        for filename_key, hash_key in (('predictions', 'prediction_sha256'), ('squads', 'squad_sha256')):
            if filename_key not in result or hash_key not in result:
                raise ValueError('evaluated results require prediction/squad byte hashes')
            file = directory / result[filename_key]
            check_file(file, result[hash_key], 'evaluated prediction/squad result')
            if file in immutable_outputs and immutable_outputs[file] != result[hash_key]:
                raise ValueError('evaluated result attempted to replace an immutable output')
            immutable_outputs[file] = result[hash_key]
        # The normal evaluator also preserves its result JSON; bind its exact bytes
        # in addition to the prediction/scoring artifacts named by the result.
        for file in [*directory.glob('*.result.json'), *directory.glob('*.admission.json'), *directory.glob('*.log')]:
            if file not in immutable_outputs:
                immutable_outputs[file] = file_hash(file)

    try:
        check_provenance()
        archive = output_dir / registration['source_archive']
        for name, digest in code_digest.items():
            destination = archive / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((config.ROOT / name).read_bytes())
            check_file(destination, digest, 'archived pipeline source')
            immutable_outputs[destination] = digest
        check_provenance()
        path.write_text(json.dumps(registration, indent=2, sort_keys=True))
        registration_digest = file_hash(path)
        experiment.log_result('research_package_registration', {'registration': str(path), **registration},
                              {'status': 'REGISTERED', 'promotion_eligible': False})
        check_provenance()
        selection_dir = output_dir / 'selection'
        selection_dir.mkdir()
        for variant, cols in variants.items():
            check_provenance()
            summary['selection'][variant] = evaluate_variant(
                sel_raw, sel_frame, cols, variant, parameters,
                cutoffs['selection_min_gw'], cutoffs['selection_max_gw'], True, selection_dir, seed)
            bind_result(summary['selection'][variant], selection_dir)
            check_provenance()
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        check_provenance()
        control_path = selection_dir / summary['selection']['mae_full']['predictions']
        for variant, result in summary['selection'].items():
            if variant != 'mae_full':
                validate_paired_forecasts(selection_dir / result['predictions'], control_path)
                check_provenance()
        ablation = select_ablation(summary['selection'])
        check_provenance()
        finalists = ['mae_full', 'rmse_full'] + ([ablation] if ablation else [])
        selection_summary_path = output_dir / 'selection_summary.json'
        selection_summary_path.write_bytes(summary_path.read_bytes())
        immutable_outputs[selection_summary_path] = file_hash(selection_summary_path)
        lock = {'finalists': finalists, 'selected_ablation': ablation,
                'selection_summary': selection_summary_path.name,
                'selection_summary_sha256': file_hash(selection_summary_path),
                'selection_rule': registration['selection_rule']}
        lock_path = output_dir / 'selection_lock.json'
        lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True))
        immutable_outputs[lock_path] = file_hash(lock_path)
        check_provenance()
        raw, frame, reference_columns = load_research_features(cutoffs['backtest_max_gw'], dataset)
        if raw.attrs.get('frozen_dataset_sha256') != registration['dataset_sha256']:
            raise ValueError('reference frozen dataset provenance differs from registration')
        if reference_columns != registration['feature_columns']:
            raise ValueError('reference ordered feature columns differ from registration')
        check_provenance()
        evaluation_dir = output_dir / 'evaluation'
        evaluation_dir.mkdir()
        for variant in finalists:
            check_provenance()
            summary['evaluation'][variant] = {}
            for origin in (False, True):
                check_provenance()
                mode = 'origin' if origin else 'standard'
                summary['evaluation'][variant][mode] = evaluate_variant(
                    raw, frame, variants[variant], variant, parameters,
                    cutoffs['backtest_min_gw'], cutoffs['backtest_max_gw'], origin, evaluation_dir, seed)
                bind_result(summary['evaluation'][variant][mode], evaluation_dir)
                check_provenance()
                summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
            if variant != 'mae_full':
                summary['comparisons'][variant] = {}
                for mode in ('standard', 'origin'):
                    check_provenance()
                    candidate = summary['evaluation'][variant][mode]
                    control = summary['evaluation']['mae_full'][mode]
                    validate_paired_forecasts(evaluation_dir / candidate['predictions'],
                                              evaluation_dir / control['predictions'])
                    check_provenance()
                    summary['comparisons'][variant][mode] = compare_backtests.compare(
                        str(evaluation_dir / candidate['squads']), str(evaluation_dir / control['squads']),
                        n_boot=10000, block_len=3, seed=seed)
                    check_provenance()
                    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        check_provenance()
        summary['registration_sha256'] = registration_digest
        summary['immutable_output_sha256'] = {str(file.relative_to(output_dir)): digest
                                              for file, digest in immutable_outputs.items()}
        summary.update(status='complete', promotion_eligible=False, selected_ablation=ablation)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        summary_digest = file_hash(summary_path)
        experiment.log_result('research_package_completed', {'registration': str(path), 'seed': seed,
                              'summary_sha256': summary_digest},
                              {'status': 'COMPLETE', 'promotion_eligible': False,
                               'comparisons': summary['comparisons']})
        check_provenance()
        check_file(summary_path, summary_digest, 'completed summary')
    except BaseException as exc:
        summary.update(status='incomplete', error=f'{type(exc).__name__}: {exc}', promotion_eligible=False)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        experiment.log_result('research_package_incomplete', {'registration': str(path)},
                              {'status': 'INCOMPLETE', 'error': summary['error'], 'promotion_eligible': False})
        raise
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run a new registered package; all artifacts stay separate from production."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=config.FROZEN_RESEARCH_DATASET_PATH)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--parameters', choices=('defaults', 'selection'), default='selection')
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args(argv)
    run_package(args.dataset, args.output_dir, parameter_mode=args.parameters, seed=args.seed)


if __name__ == '__main__':
    main()
