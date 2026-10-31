"""Registered fixed EPL calendar workload challenger; historical schedule proxy only."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd

from fpl import config, experiment
from fpl.data import provenance
from fpl.data.origin_prices import build_origin_price_catalog, validate_origin_price_catalog
from fpl.data.research import load_research_features
from fpl.model import blend_research, cnn_research, research_package
from fpl.model.moe_tournament import protocol_cutoffs
from fpl.run_week import build_live_snapshot

HELPER_PATH = Path(__file__).resolve().with_name('league_workload.py')
_LOADED_DRIVER_SHA = research_package.file_hash(Path(__file__))
_LOADED_HELPER_SHA = research_package.file_hash(HELPER_PATH)
_spec = importlib.util.spec_from_file_location('registered_league_workload', HELPER_PATH)
if _spec is None or _spec.loader is None:
    raise ValueError('Workload helper cannot be loaded')
_helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helper)
COLUMNS = tuple(_helper.COLUMNS)
ARMS = ('mae_full', 'mae_league_workload')


def build_context(raw: pd.DataFrame, max_gw: int) -> pd.DataFrame:
    """Collapse aligned workload values to unique team/rounds, including DGW NaNs."""
    aligned = _helper.build_league_workload(raw, max_gw)
    if len(aligned) != len(raw) or not aligned.index.equals(raw.index):
        raise ValueError('Workload helper did not preserve admitted raw row alignment')
    rows = raw[['team', 'GW_global']].copy()
    rows[list(COLUMNS)] = aligned[list(COLUMNS)].to_numpy()
    for column in COLUMNS:
        if rows.groupby(['team', 'GW_global'], dropna=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError('Workload DGW team/round values conflict')
    return rows.drop_duplicates(['team', 'GW_global']).reset_index(drop=True)


def attach_context(frame: pd.DataFrame, context: pd.DataFrame, *, origin: int | None = None) -> pd.DataFrame:
    """Join fixed team/round context without dropping/reordering rows or inventing clocks."""
    if set(COLUMNS) & set(frame.columns):
        raise ValueError('Workload feature collision')
    if context.duplicated(['team','GW_global']).any():
        raise ValueError('Workload context must have unique team/rounds')
    keys = frame[['team']].copy()
    keys['GW_global'] = frame.GW_global.to_numpy() if origin is None else origin
    joined = keys.merge(context[['team','GW_global',*COLUMNS]], on=['team','GW_global'], how='left', sort=False, validate='many_to_one')
    extended = frame.copy()
    extended[list(COLUMNS)] = joined[list(COLUMNS)].to_numpy()
    extended.attrs = dict(frame.attrs)
    return extended


def _check(registration: dict, outputs: dict[str, str]) -> None:
    if research_package.source_hashes() != registration['code_sha256']:
        raise ValueError('Workload pipeline sources changed')
    for name,digest in {**registration['input_sha256'],**outputs}.items():
        if research_package.file_hash(Path(name)) != digest:
            raise ValueError(f'Sealed workload bytes changed: {name}')


def run(dataset: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Preregister seed-zero equal-capacity MAE; advance only on strict origin net gain."""
    dataset,output_dir = Path(dataset),Path(output_dir)
    if output_dir.exists():
        raise ValueError('Workload output already exists; choose a fresh directory')
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    outputs: dict[str,str] = {}
    summary: dict[str,Any] = dict(status='running',promotion_eligible=False,selection={},evaluation={},comparisons={})
    summary_path = output_dir/'summary.json'
    registration_path = output_dir/'registration.json'
    try:
        cutoffs = protocol_cutoffs()
        windows = {'selection':(cutoffs['selection_min_gw'],cutoffs['selection_max_gw']),
                   'reference':(cutoffs['backtest_min_gw'],cutoffs['backtest_max_gw'])}
        parameters,tuned = research_package.resolve_parameters('selection',0)
        parameters = {'catboost_mae':parameters['catboost_mae']}
        if any(pos['catboost_mae'].get('thread_count') != 5 or pos['catboost_mae'].get('random_seed') != 0 for pos in parameters['catboost_mae'].values()):
            raise ValueError('Workload requires registered seed-zero capacity and five threads')
        inputs: dict[str,str] = {}
        for path in (dataset,provenance.FROZEN_MANIFEST_PATH,Path(__file__),HELPER_PATH):
            blend_research._bind(inputs,Path(path), _LOADED_DRIVER_SHA if Path(path).resolve() == Path(__file__).resolve() else _LOADED_HELPER_SHA if Path(path).resolve() == HELPER_PATH else None)
        for relative,digest in tuned.items():
            blend_research._bind(inputs,config.ROOT/relative,digest)
        code = research_package.source_hashes()
        registration = dict(status='registered',seed=0,model_name='catboost_mae',parameters=parameters,
            tuned_artifact_sha256=tuned,code_sha256=code,source_archive='source_archive',input_sha256=inputs,
            dataset=str(dataset.resolve()),dataset_sha256=inputs[str(dataset.resolve())],windows=windows,cutoffs=cutoffs,
            arms=list(ARMS),extra_player_features=list(COLUMNS),base_feature_count=115,
            base_feature_recipe='Full ordered fpl.features.feature_columns on capped frozen frame; exact phase columns sealed before first fit.',
            retrain_every=4,horizon=3,chips='disabled',gap_rel=0,solver='highs',initial_budget=1000,
            scoring_protocol='autosubs_v2',points_accounting='net_after_transfer_hits',
            origin_price_policy='origin-current-proxy',standard_price_policy='target-replay',
            selection_rule='Candidate advances only if selection-origin net points strictly exceed control; ties reject.',
            comparison=dict(n_boot=10000,block_len=3,seed=0),comparison_labels=['selection_origin','reference_standard','reference_origin'],
            python=platform.python_version(),versions=experiment._library_versions(),promotion_eligible=False,
            hypothesis='Three prior EPL calendar workload inputs improve equal-capacity MAE decisions.',
            limitations=['EPL only; no cup, Europe or Elo features.',
                'Recorded kickoff times are historical schedule proxies; publication timing is unproven.',
                'Current-origin team workload freezes across the horizon; missing team/origin context stays NaN.',
                'History-based player pool and reconstructed/backfilled current-price proxies are not complete PIT evidence.',
                'Repeatedly examined selection/reference outcomes; standard forecasts are historical target-price diagnostics.',
                'Old FT and simplified selling-price policies remain; no fresh holdout or production promotion.',
                'One fixed seed, tracked capacity, no new HPO or adaptive feature-family selection.'])
        cnn_research._archive_sources(output_dir/'source_archive',code)
        for relative,digest in code.items():
            blend_research._bind(outputs,output_dir/'source_archive'/relative,digest)
        for path in (Path(__file__),HELPER_PATH):
            destination = output_dir/'source_archive'/'driver'/path.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(path.read_bytes())
            blend_research._bind(outputs,destination,inputs[str(path.resolve())])
        cnn_research._write_json(registration_path,registration)
        blend_research._bind(outputs,registration_path)
        _check(registration,outputs)
        experiment.log_result('league_workload_registration',registration,{'status':'REGISTERED','promotion_eligible':False})
        _check(registration,outputs)
        base_columns: list[str] | None = None

        def prepare(phase: str) -> tuple[pd.DataFrame,pd.DataFrame,dict,Path,pd.DataFrame]:
            nonlocal base_columns
            _check(registration,outputs)
            start,end = windows[phase]
            raw,frame,columns = load_research_features(end,dataset)
            if raw.attrs.get('frozen_dataset_sha256') != registration['dataset_sha256'] or not raw.GW_global.between(1,end).all() or not frame.GW_global.between(1,end).all():
                raise ValueError('Workload frozen admission/cutoff mismatch')
            if len(columns) != 115 or len(set(columns)) != len(columns) or 'total_points' in columns or set(columns)&set(COLUMNS):
                raise ValueError('Workload ordered base schema mismatch')
            if base_columns is not None and base_columns != list(columns):
                raise ValueError('Workload phase ordered base features differ')
            base_columns = list(columns)
            context = build_context(raw,end)
            extended = attach_context(frame,context)
            directory = output_dir/phase
            directory.mkdir()
            context.to_csv(directory/'context.csv',index=False)
            calendar = raw[['team','GW_global','fixture','kickoff_time']].drop_duplicates()
            calendar.to_csv(directory/'calendar.csv',index=False)
            variants = {ARMS[0]:list(columns),ARMS[1]:list(columns)+list(COLUMNS)}
            cnn_research._write_json(directory/'columns.json',variants)
            ids = sorted(frame.loc[frame.GW_global.between(start,end),'player_id'].unique().tolist())
            catalog = validate_origin_price_catalog(build_origin_price_catalog(raw,ids,start,end),ids,range(start,end+1))
            catalog_path = directory/'origin_prices.csv'
            catalog.to_csv(catalog_path,index=False)
            for name in ('context.csv','calendar.csv','columns.json','origin_prices.csv'):
                blend_research._bind(outputs,directory/name)
            _check(registration,outputs)
            return raw,extended,variants,catalog_path,context

        def score(phase: str,arm: str,origin: bool,prepared: tuple) -> dict:
            _check(registration,outputs)
            raw,frame,variants,catalog,context = prepared
            directory = output_dir/phase
            kwargs: dict[str,Any] = {}
            if origin:
                kwargs.update(origin_price_catalog=catalog,origin_price_policy='origin-current-proxy')
                if arm == ARMS[1]:
                    def snapshot(hist_raw: pd.DataFrame,origin_gw: int) -> pd.DataFrame:
                        if not hist_raw.empty and hist_raw.GW_global.max() >= origin_gw:
                            raise ValueError('Workload snapshot history must precede origin')
                        return attach_context(build_live_snapshot(hist_raw),context,origin=origin_gw)
                    kwargs.update(origin_snapshot_builder=snapshot,extra_player_features=list(COLUMNS))
            result = research_package.evaluate_variant(raw,frame,variants[arm],arm,parameters,
                *windows[phase],origin,directory,0,model_name='catboost_mae',retrain_every=4,**kwargs)
            if not np.isfinite(result['total_points']):
                raise ValueError('Workload net points must be finite')
            for field,digest_key in (('predictions','prediction_sha256'),('squads','squad_sha256')):
                blend_research._bind(outputs,directory/result[field],result[digest_key])
            prefix = directory/f'{arm}_{"origin" if origin else "standard"}'
            for suffix in ('.admission.json','.result.json','.log'):
                blend_research._bind(outputs,prefix.with_suffix(suffix))
            _check(registration,outputs)
            return result

        def compare(phase: str,candidate: dict,control: dict,label: str) -> dict:
            _check(registration,outputs)
            directory = output_dir/phase
            research_package.validate_paired_forecasts(directory/candidate['predictions'],directory/control['predictions'])
            report = research_package.compare_backtests.compare(str(directory/candidate['squads']),str(directory/control['squads']),**registration['comparison'])
            path = output_dir/f'{label}_comparison.json'
            cnn_research._write_json(path,report)
            blend_research._bind(outputs,path)
            _check(registration,outputs)
            return report

        selection = prepare('selection')
        for arm in ARMS:
            summary['selection'][arm] = score('selection',arm,True,selection)
            cnn_research._write_json(summary_path,summary)
        summary['comparisons']['selection_origin'] = compare('selection',summary['selection'][ARMS[1]],summary['selection'][ARMS[0]],'selection_origin')
        advanced = summary['selection'][ARMS[1]]['total_points'] > summary['selection'][ARMS[0]]['total_points']
        selection_path = output_dir/'selection_summary.json'
        cnn_research._write_json(selection_path,dict(selection=summary['selection'],comparison=summary['comparisons']['selection_origin'],advanced=advanced))
        blend_research._bind(outputs,selection_path)
        cnn_research._write_json(output_dir/'selection_lock.json',dict(selection_summary_sha256=research_package.file_hash(selection_path),
            advanced=advanced,finalists=list(ARMS) if advanced else [],parameters=parameters,feature_variants=selection[2]))
        blend_research._bind(outputs,output_dir/'selection_lock.json')
        _check(registration,outputs)
        if advanced:
            reference = prepare('reference')
            for arm in ARMS:
                summary['evaluation'][arm] = {}
                for origin in (False,True):
                    summary['evaluation'][arm]['origin' if origin else 'standard'] = score('reference',arm,origin,reference)
                    cnn_research._write_json(summary_path,summary)
            for mode in ('standard','origin'):
                label = f'reference_{mode}'
                summary['comparisons'][label] = compare('reference',summary['evaluation'][ARMS[1]][mode],summary['evaluation'][ARMS[0]][mode],label)
        _check(registration,outputs)
        summary.update(status='complete' if advanced else 'selection_rejected',advanced=advanced,
            elapsed_seconds=time.perf_counter()-started,registration_sha256=research_package.file_hash(registration_path),immutable_output_sha256=dict(outputs))
        cnn_research._write_json(summary_path,summary)
        summary_hash = research_package.file_hash(summary_path)
        experiment.log_result('league_workload_completed',{'registration':str(registration_path),'summary_sha256':summary_hash},
            {'status':summary['status'].upper(),'promotion_eligible':False,'comparisons':summary['comparisons']})
        _check(registration,outputs)
        if research_package.file_hash(summary_path) != summary_hash:
            raise ValueError('Workload final summary changed')
        return summary
    except BaseException as exc:
        summary.update(status='incomplete',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-started,
            immutable_output_sha256=dict(outputs),registration_sha256=outputs.get(str(registration_path.resolve())))
        cnn_research._write_json(summary_path,summary)
        experiment.log_result('league_workload_incomplete',{'registration':str(registration_path)},{'status':'INCOMPLETE','error':summary['error'],'promotion_eligible':False})
        raise


def main(argv: list[str] | None = None) -> None:
    """Run an explicitly frozen, fresh workload research study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=config.FROZEN_RESEARCH_DATASET_PATH)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args(argv)
    run(args.dataset,args.output_dir)


if __name__ == '__main__':
    main()
