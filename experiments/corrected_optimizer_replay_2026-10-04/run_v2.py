"""Predeclared saved-forecast MAE/RMSE replays under corrected optimizer logic."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import time
import shutil

import numpy as np
import pandas as pd

from fpl import config, experiment
from fpl.data.provenance import FROZEN_MANIFEST_PATH, load_frozen_research_dataset
from fpl.milp import compare_backtests
from fpl.model import research_package as rp

ROOT = config.ROOT
OUTPUT = Path(__file__).resolve().parent / 'replay_v2'
RECOVERY = Path(__file__).resolve().parent / 'replay'
DATASET = Path('/Users/perandersmolberg/Developer/MSc-BusAdm-25-EBA-PAM/Datasett/frozen/master_dataset_gw231.csv')
SEEDS = (0, 1, 2)
ARMS = ('mae_full', 'rmse_full')
MODES = ('standard', 'origin')
START, END = 153, 183


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
                                  previous_total=old['total_points'], parameters=registration['parameters']))
        for mode in MODES:
            rp.validate_paired_forecasts(base / 'evaluation' / summary['evaluation']['rmse_full'][mode]['predictions'],
                                         base / 'evaluation' / summary['evaluation']['mae_full'][mode]['predictions'])
    verifier = ROOT / 'experiments/league_workload_2026-10-04/verify_v2.py'
    for path in (DATASET, FROZEN_MANIFEST_PATH, verifier, Path(__file__)):
        inputs[str(path.resolve())] = rp.file_hash(path)
    sources = rp.source_hashes()
    if RECOVERY is not None:
        earlier=json.loads((RECOVERY/'registration.json').read_text())
        if earlier['code_sha256']!=sources:
            raise ValueError('Recovered optimizer sources differ')
        for path in [RECOVERY/'registration.json',RECOVERY/'summary.json',*sorted((RECOVERY/'seed0').glob('*'))]:
            if path.is_file():inputs[str(path.resolve())]=rp.file_hash(path)
    OUTPUT.mkdir()
    registration = dict(status='registered', issue_refs=[6,13], promotion_eligible=False,
        hypothesis='Correct zero-gap admission and exact free-transfer lookahead may change earlier saved-forecast decisions.',
        cases=cases, input_sha256=inputs, code_sha256=sources, versions=experiment._library_versions(),
        start_gw=START,end_gw=END,horizon=3,seeds=list(SEEDS),chips='disabled',gap_rel=0,gap_abs=0,
        points_accounting='net_after_transfer_hits',scoring_protocol='autosubs_v2',
        roundoff_relative_tolerance=64*np.finfo(float).eps,
        continuation_of=str(RECOVERY) if RECOVERY is not None else None,
        recovery_reason='First complete control stopped at a checker requiring literal zero rather than floating-point roundoff; original run retained.',
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
            if RECOVERY is not None and case['seed']==0 and case['arm']=='mae_full' and case['mode']=='standard':
                prefix='mae_full_standard'
                result=json.loads((RECOVERY/'seed0'/f'{prefix}.result.json').read_text())
                for path in sorted((RECOVERY/'seed0').glob(f'{prefix}.*')):
                    shutil.copyfile(path,directory/path.name)
                if rp.file_hash(directory/result['predictions'])!=inputs[case['prediction']]:
                    raise ValueError('Recovered control forecasts differ from pinned forecasts')
                if rp.file_hash(directory/result['squads'])!=result['squad_sha256']:
                    raise ValueError('Recovered control squad digest mismatch')
                result['reused_scored_output']=True
            else:
                result=rp.evaluate_variant(raw,pd.DataFrame(),[],case['arm'],case['parameters'],START,END,
                    case['mode']=='origin',directory,case['seed'],model_name=name,
                    forecast_provider=lambda frame=frame:frame.copy())
                result['reused_scored_output']=False
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
