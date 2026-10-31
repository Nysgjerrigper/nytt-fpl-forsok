"""Predeclared saved-forecast MAE/RMSE replays under corrected optimizer logic."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import time
import shutil
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from fpl import config, experiment
from fpl.data.provenance import FROZEN_MANIFEST_PATH, load_frozen_research_dataset
from fpl.milp import compare_backtests, optimize
from fpl.model import research_package as rp

ROOT = config.ROOT
OUTPUT = Path(__file__).resolve().parent / 'replay_v3'
DATASET = Path('/Users/perandersmolberg/Developer/MSc-BusAdm-25-EBA-PAM/Datasett/frozen/master_dataset_gw231.csv')
SEEDS = (0, 1, 2)
ARMS = ('mae_full', 'rmse_full')
MODES = ('standard', 'origin')
START, END = 153, 183


def score_case(case: dict, directory: Path) -> dict:
    """Score exact original forecast bytes, preserving gross and hit-adjusted points."""
    prefix=directory/f"{case['arm']}_{case['mode']}"
    if any(directory.glob(prefix.name+'.*')):
        raise ValueError('Replay case artifacts exist; preserve them')
    prediction=prefix.with_suffix('.predictions.csv');squad=prefix.with_suffix('.squads.csv')
    shutil.copyfile(case['prediction'],prediction)
    digest=rp.file_hash(prediction)
    if digest!=rp.file_hash(Path(case['prediction'])):
        raise ValueError('Replay forecast bytes differ from source')
    frame=pd.read_csv(prediction)
    rp.validate_forecasts(frame,START,END,case['mode']=='origin')
    write(prefix.with_suffix('.admission.json'),dict(prediction_sha256=digest,
          source_prediction=case['prediction'],parameters=case['parameters'],seed=case['seed'],
          horizon=3,start_gw=START,end_gw=END,cached_replay=True,byte_identical=True))
    args=optimize.parse_args(['--predictions-csv',str(prediction),'--start-gw',str(START),
         '--max-gw',str(END),'--horizon','3','--scoring-mode','corrected',
         '--solver','highs','--gap-rel','0','--output',str(squad)])
    started=time.perf_counter()
    with prefix.with_suffix('.log').open('w',buffering=1) as log,redirect_stdout(log):
        optimize.run(args)
    if rp.file_hash(prediction)!=digest:raise ValueError('Replay forecasts changed during solve')
    squads=pd.read_csv(squad)
    if squads.gameweek.tolist()!=list(range(START,END+1)) or not squads.scoring_protocol.eq('autosubs_v2').all():
        raise ValueError('Incomplete or incompatible replay scoring')
    if squads.chip_played.notna().any():raise ValueError('Replay chips not disabled')
    squads['gross_actual_total_points']=squads.actual_total_points
    squads['actual_total_points']=squads.gross_actual_total_points-4*squads.alpha
    squads['points_accounting']='net_after_transfer_hits'
    squads.to_csv(squad,index=False)
    result=dict(variant=case['arm'],origin=case['mode']=='origin',total_points=float(squads.actual_total_points.sum()),
        fit_seconds=0.,milp_seconds=time.perf_counter()-started,predictions=prediction.name,
        prediction_sha256=digest,squads=squad.name,squad_sha256=rp.file_hash(squad),
        gross_points=float(squads.gross_actual_total_points.sum()),transfer_hit_points=float(4*squads.alpha.sum()),
        diagnostics=case['forecast_diagnostics'])
    write(prefix.with_suffix('.result.json'),result)
    print(f"seed{case['seed']}/{case['arm']}/{case['mode']}: {result['total_points']:.0f} net points; MILP {result['milp_seconds']:.1f}s",flush=True)
    return result


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def run() -> dict:
    """Seal all inputs before scoring; never refit models or overwrite a prior run."""
    if OUTPUT.exists():
        raise ValueError('Replay directory exists; preserve it')
    inputs, cases = {}, []
    for seed in SEEDS:
        base = ROOT / f'experiments/research_priorities_2026-10-03/selection_seed{seed}'
        registration = json.loads((base / 'registration.json').read_text())
        summary = json.loads((base / 'summary.json').read_text())
        for path in (base / 'registration.json', base / 'summary.json'):
            inputs[str(path.resolve())] = rp.file_hash(path)
        for arm in ARMS:
            for mode in MODES:
                old = summary['evaluation'][arm][mode]
                path = base / 'evaluation' / old['predictions']
                if rp.file_hash(path) != old['prediction_sha256']:
                    raise ValueError('Saved forecast bytes do not match original result')
                inputs[str(path.resolve())] = rp.file_hash(path)
                cases.append(dict(seed=seed, arm=arm, mode=mode, prediction=str(path.resolve()),
                                  previous_total=old['total_points'], parameters=registration['parameters'], forecast_diagnostics=old['diagnostics']))
        for mode in MODES:
            rp.validate_paired_forecasts(base / 'evaluation' / summary['evaluation']['rmse_full'][mode]['predictions'],
                                         base / 'evaluation' / summary['evaluation']['mae_full'][mode]['predictions'])
    verifier = ROOT / 'experiments/league_workload_2026-10-04/verify_v2.py'
    for path in (DATASET, FROZEN_MANIFEST_PATH, verifier, Path(__file__)):
        inputs[str(path.resolve())] = rp.file_hash(path)
    sources = rp.source_hashes()
    OUTPUT.mkdir()
    registration = dict(status='registered', issue_refs=[6,13], promotion_eligible=False,
        hypothesis='Correct zero-gap admission and exact free-transfer lookahead may change earlier saved-forecast decisions.',
        cases=cases, input_sha256=inputs, code_sha256=sources, versions=experiment._library_versions(),
        start_gw=START,end_gw=END,horizon=3,seeds=list(SEEDS),chips='disabled',gap_rel=0,gap_abs=0,
        points_accounting='net_after_transfer_hits',scoring_protocol='autosubs_v2',
        roundoff_relative_tolerance=64*np.finfo(float).eps,
        forecast_input_policy='Byte-identical copies of original CSVs; no pandas reserialization before solving.',
        prior_incomplete_attempts=['replay','replay_v2'],
        limitations=['Previously examined historical reference; no untouched holdout.',
                     'Fixed saved forecasts/prices; no new training or hyperparameter selection.',
                     'Does not isolate solver tolerance from the free-transfer correction.',
                     'Historical publication-time/price and legacy chip policy limitations persist.'])
    write(OUTPUT/'registration.json',registration)
    registration_digest=rp.file_hash(OUTPUT/'registration.json')
    archive=OUTPUT/'evaluated_sources'
    for name,digest in sources.items():
        target=archive/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
        assert rp.file_hash(target)==digest
    outputs={str(path.resolve()):rp.file_hash(path) for path in archive.rglob('*.py')}
    summary=dict(status='running',promotion_eligible=False,results=[],comparisons={})
    def check():
        if rp.source_hashes()!=sources or rp.file_hash(OUTPUT/'registration.json')!=registration_digest:
            raise ValueError('Replay registration/source drift')
        for name,digest in {**inputs,**outputs}.items():
            if rp.file_hash(Path(name))!=digest:raise ValueError(f'Replay artifact drift: {name}')
    def bind(directory):
        for path in directory.glob('*'):
            if path.is_file():outputs[str(path.resolve())]=rp.file_hash(path)
    try:
        check()
        experiment.log_result('corrected_optimizer_replay_registered',{'registration':str(OUTPUT/'registration.json'),'registration_sha256':registration_digest},{'status':'REGISTERED','promotion_eligible':False})
        raw=load_frozen_research_dataset(DATASET)
        raw=raw.loc[raw.GW_global<=END].copy()
        spec=importlib.util.spec_from_file_location('independent_workload_verifier',verifier)
        independent=importlib.util.module_from_spec(spec);spec.loader.exec_module(independent)
        started=time.perf_counter()
        for case in cases:
            check()
            directory=OUTPUT/f"seed{case['seed']}";directory.mkdir(exist_ok=True)
            name='catboost_mae' if case['arm']=='mae_full' else 'catboost_rmse'
            frame=pd.read_csv(case['prediction'])
            result=score_case(case,directory)
            squads=pd.read_csv(directory/result['squads'])
            for column,expected in [('solver_status','Optimal'),('solver_solution_status','Optimal Solution Found'),('solver_native_status','Optimal')]:
                if not squads[column].eq(expected).all():raise ValueError('Native optimality not established')
            if not squads[['solver_gap_rel','solver_gap_abs']].eq(0).all().all():
                raise ValueError('Nonzero requested solver gap')
            bounds=squads[['solver_primal_bound','solver_dual_bound']].to_numpy(dtype=float)
            gaps=squads.solver_mip_gap.to_numpy(dtype=float)
            tolerance=registration['roundoff_relative_tolerance']
            scale=np.maximum(1.,np.abs(bounds).max(axis=1))
            if (not np.isfinite(bounds).all() or not np.isfinite(gaps).all() or (gaps<0).any()
                    or (gaps>tolerance).any() or (np.abs(bounds[:,0]-bounds[:,1])>tolerance*scale).any()):
                raise ValueError('Native gap/bound difference exceeds floating-point roundoff')
            missing=[]
            count=independent.budgets_and_points(frame,squads,START,END,None,missing_evidence=missing)
            result.update(seed=case['seed'],mode=case['mode'],previous_total=case['previous_total'],
                          difference_from_previous=result['total_points']-case['previous_total'],
                          independently_reconstructed_rounds=count,missing_actual_row_evidence=missing)
            summary['results'].append(result);bind(directory);check()
            write(OUTPUT/'summary.json',summary)
        for seed in SEEDS:
            summary['comparisons'][str(seed)]={}
            for mode in MODES:
                directory=OUTPUT/f'seed{seed}'
                report=compare_backtests.compare(str(directory/f'rmse_full_{mode}.squads.csv'),str(directory/f'mae_full_{mode}.squads.csv'),n_boot=10000,block_len=3,seed=seed)
                summary['comparisons'][str(seed)][mode]=report
        check()
        summary.update(status='complete',optimal_solves=len(cases)*(END-START+1),elapsed_seconds=time.perf_counter()-started,
                       registration_sha256=registration_digest,immutable_output_sha256=outputs)
        write(OUTPUT/'summary.json',summary);digest=rp.file_hash(OUTPUT/'summary.json')
        experiment.log_result('corrected_optimizer_replay_completed',{'registration':str(OUTPUT/'registration.json'),'summary_sha256':digest},{'status':'COMPLETE','promotion_eligible':False,'comparisons':summary['comparisons']})
        check()
        if rp.file_hash(OUTPUT/'summary.json')!=digest:raise ValueError('Final replay summary changed')
        return summary
    except BaseException as exc:
        summary.update(status='incomplete',error=f'{type(exc).__name__}: {exc}',immutable_output_sha256=outputs)
        write(OUTPUT/'summary.json',summary)
        experiment.log_result('corrected_optimizer_replay_incomplete',{'registration':str(OUTPUT/'registration.json')},{'status':'INCOMPLETE','promotion_eligible':False,'error':summary['error']})
        raise


if __name__=='__main__':run()
