"""Read-only reconstruction of completed byte-preserving optimizer replays."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpl.model import research_package as rp
from fpl.milp import compare_backtests


def verify(directory: Path) -> dict:
    """Rehash immutable evidence and reconstruct cash, realized points and paired CIs."""
    registration_path,summary_path=directory/'registration.json',directory/'summary.json'
    registration=json.loads(registration_path.read_text());summary=json.loads(summary_path.read_text())
    if summary['status']!='complete':raise ValueError('Replay is not complete')
    if rp.file_hash(registration_path)!=summary['registration_sha256']:raise ValueError('Registration mismatch')
    summary_digest=rp.file_hash(summary_path)
    evidence={**registration['input_sha256'],**summary['immutable_output_sha256']}
    for name,digest in evidence.items():
        if rp.file_hash(Path(name))!=digest:raise ValueError(f'Evidence drift: {name}')
    root=Path(__file__).resolve().parents[2]
    helper=root/'experiments/league_workload_2026-10-04/verify_v2.py'
    if str(helper.resolve()) not in registration['input_sha256']:raise ValueError('Independent accounting source not sealed')
    spec=importlib.util.spec_from_file_location('replay_accounting',helper)
    accounting=importlib.util.module_from_spec(spec);spec.loader.exec_module(accounting)
    start,end=registration['start_gw'],registration['end_gw']
    results={(r['seed'],r['variant'],r['mode']):r for r in summary['results']}
    if len(results)!=len(registration['cases']):raise ValueError('Missing or duplicated replay cases')
    total=0;maximum_gap=0.;missing_count=0
    for case in registration['cases']:
        result=results[case['seed'],case['arm'],case['mode']]
        location=directory/f"seed{case['seed']}"
        prediction=location/result['predictions'];squad=location/result['squads']
        if rp.file_hash(prediction)!=registration['input_sha256'][case['prediction']]:
            raise ValueError('Original prediction bytes not preserved')
        if rp.file_hash(squad)!=result['squad_sha256']:raise ValueError('Squad bytes differ')
        frame,squads=pd.read_csv(prediction),pd.read_csv(squad)
        rp.validate_forecasts(frame,start,end,case['mode']=='origin')
        if squads.gameweek.tolist()!=list(range(start,end+1)):raise ValueError('Round coverage')
        for column,label in [('solver_status','Optimal'),('solver_solution_status','Optimal Solution Found'),('solver_native_status','Optimal'),('scoring_protocol','autosubs_v2'),('points_accounting','net_after_transfer_hits')]:
            if not squads[column].eq(label).all():raise ValueError(f'Unexpected {column}')
        if not squads[['solver_gap_rel','solver_gap_abs']].eq(0).all().all():raise ValueError('Nonzero requested gap')
        gap=squads.solver_mip_gap.to_numpy(dtype=float)
        bounds=squads[['solver_primal_bound','solver_dual_bound']].to_numpy(dtype=float)
        epsilon=64*np.finfo(float).eps
        if (not np.isfinite(gap).all() or not np.isfinite(bounds).all() or (gap<0).any()
            or (gap>epsilon).any() or (np.abs(bounds[:,0]-bounds[:,1])>epsilon*np.maximum(1.,np.abs(bounds).max(axis=1))).any()):
            raise ValueError('Native gap exceeds machine roundoff')
        maximum_gap=max(maximum_gap,float(gap.max()))
        if squads.chip_played.notna().any():raise ValueError('Research chips enabled')
        if not np.array_equal(squads.actual_total_points,squads.gross_actual_total_points-4*squads.alpha):raise ValueError('Net hit accounting')
        if squads.actual_total_points.sum()!=result['total_points']:raise ValueError('Total mismatch')
        previous_ft=1
        for row in squads.to_dict('records'):
            transfers=[] if pd.isna(row['transfers_in']) else str(row['transfers_in']).split(', ')
            if row['q_start']!=previous_ft:raise ValueError('Carried FT state mismatch')
            if row['alpha']!=max(0,len(transfers)-previous_ft):raise ValueError('Hit count mismatch')
            previous_ft=min(2,max(0,previous_ft-len(transfers))+1)
        missing=[]
        count=accounting.budgets_and_points(frame,squads,start,end,None,missing_evidence=missing)
        if missing!=result['missing_actual_row_evidence']:raise ValueError('Missing-outcome disclosure mismatch')
        missing_count+=sum(len(row['missing_all_origin_actual_rows']) for row in missing)
        total+=count
    for seed in registration['seeds']:
        for mode in ('standard','origin'):
            base=directory/f'seed{seed}'
            rp.validate_paired_forecasts(base/f'rmse_full_{mode}.predictions.csv',base/f'mae_full_{mode}.predictions.csv')
            rebuilt=compare_backtests.compare(str(base/f'rmse_full_{mode}.squads.csv'),str(base/f'mae_full_{mode}.squads.csv'),n_boot=10000,block_len=3,seed=seed)
            if rebuilt!=summary['comparisons'][str(seed)][mode]:raise ValueError('Paired comparison mismatch')
    for name,digest in evidence.items():
        if rp.file_hash(Path(name))!=digest:raise ValueError('Evidence changed during verification')
    if rp.file_hash(summary_path)!=summary_digest:raise ValueError('Summary drift')
    if total!=summary['optimal_solves']:raise ValueError('Optimal count mismatch')
    return dict(status='verified',promotion_eligible=False,independently_reconstructed_rounds=total,
        rechecked_artifacts=len(evidence),summary_sha256=summary_digest,maximum_achieved_gap=maximum_gap,
        roundoff_relative_tolerance=64*np.finfo(float).eps,missing_all_origin_squad_rows=missing_count,
        verifier_sha256=rp.file_hash(Path(__file__)),comparisons=summary['comparisons'],
        limitations=['Historical reference is repeatedly examined, not an untouched holdout.',
                     'Floating-point native optimality certificate; requested tolerances are exactly zero.',
                     'No refits, production promotion or revised modern-chip policy.'])


if __name__=='__main__':
    directory=Path(__file__).resolve().parent/'replay_v3'
    target=directory/'verification.json'
    if target.exists():raise ValueError('Preserve existing verification')
    report=verify(directory)
    target.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:report[k] for k in ('status','independently_reconstructed_rounds','rechecked_artifacts','maximum_achieved_gap')}))
