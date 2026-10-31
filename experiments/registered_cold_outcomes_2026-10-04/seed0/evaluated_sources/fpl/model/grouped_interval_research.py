"""Fixed grouped interval diagnostics from verified quantile exports, without refitting.

Grouping uses shifted minutes propensity only. Observed played status is a reported
post-outcome diagnostic, never a grouping input. Neither temporal exchangeability,
MILP improvement, production promotion nor a fresh holdout is established.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd

from fpl import config, experiment
from fpl.data import provenance
from fpl.data.research import load_research_features
from fpl.model import blend_research, cnn_research, research_package
from fpl.model.grouped_interval_calibration import fit_grouped_calibration, apply_grouped_calibration
from fpl.model.interval_calibration import conformal_adjustment, apply_adjustment, interval_metrics

KEYS = ['player_id', 'position', 'GW_global']
FEATURE = 'mins60_rate_roll5'


def minutes_groups(values: pd.Series, thresholds: tuple[float, float]) -> np.ndarray:
    """Use explicit low/mid/high boundaries and a separate missing category."""
    low, high = thresholds
    if not 0 <= low < high <= 1:
        raise ValueError('Invalid minutes group thresholds')
    numeric = pd.to_numeric(values, errors='coerce')
    if (values.notna() & numeric.isna()).any() or not np.isfinite(numeric.dropna()).all() or not numeric.dropna().between(0,1).all():
        raise ValueError('Observed shifted minutes propensity must be finite in [0,1]')
    return np.select([numeric.isna(),numeric < low,numeric < high],['missing','low','mid'],default='high')


def join_minutes(base: pd.DataFrame, frame: pd.DataFrame, thresholds: tuple[float,float]) -> pd.DataFrame:
    """Collapse consistent DGW features, then join exactly one row per base identity."""
    if set(KEYS)-set(base) or set(KEYS+[FEATURE])-set(frame) or base.duplicated(KEYS).any() or FEATURE in base or 'minutes_group' in base:
        raise ValueError('Missing feature keys or duplicate base player rounds')
    if frame[KEYS].isna().any().any():
        raise ValueError('Feature identities must be complete')
    minutes_groups(frame[FEATURE],thresholds)
    if frame.groupby(KEYS,dropna=False)[FEATURE].nunique(dropna=False).gt(1).any():
        raise ValueError('DGW shifted minutes propensity differs within a player round')
    collapsed=frame[KEYS+[FEATURE]].drop_duplicates(KEYS)
    joined=base.merge(collapsed,on=KEYS,how='left',sort=False,validate='one_to_one',indicator=True)
    if len(joined)!=len(base) or not joined[KEYS].reset_index(drop=True).equals(base[KEYS].reset_index(drop=True)) or not joined['_merge'].eq('both').all():
        raise ValueError('Base intervals lack exact matching feature rows or join reordered keys')
    joined=joined.drop(columns='_merge');joined.index=base.index
    joined['minutes_group']=minutes_groups(joined[FEATURE],thresholds)
    return joined


def _validate_base(frame: pd.DataFrame, start: int, end: int) -> None:
    required=set(KEYS+['actual_total_points','minutes','raw_lower','median','raw_upper',
                       'calibrated_lower','calibrated_upper','adjustment'])
    if frame.empty or not frame.columns.is_unique or required-set(frame) or frame.duplicated(KEYS).any():
        raise ValueError('Invalid or duplicate base interval schema')
    if frame.position.isna().any() or not frame.position.isin(config.ONFIELD_POSITIONS).all():
        raise ValueError('Invalid base position')
    for name in ('player_id','GW_global'):
        values=frame[name]
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values) or not np.isfinite(values).all() or (values<=0).any() or (values%1!=0).any():
            raise ValueError('Base identity must be positive numeric whole IDs/gameweeks')
    for name in required-set(KEYS):
        values=frame[name]
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values) or not np.isfinite(values).all():
            raise ValueError(f'Base interval numeric values invalid: {name}')
    if not frame.GW_global.between(start,end).all() or (frame.minutes<0).any():
        raise ValueError('Base CSV violates declared phase window or minutes')
    if (frame.raw_lower>frame['median']).any() or (frame['median']>frame.raw_upper).any() or (frame.calibrated_lower>frame['median']).any() or (frame['median']>frame.calibrated_upper).any():
        raise ValueError('Base interval bounds must bracket median')


def _scores(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    return upper-lower + 2/alpha*(np.maximum(lower-y,0)+np.maximum(y-upper,0))


def paired_interval_score_ci(gameweeks: np.ndarray, differences: np.ndarray, *, n_boot: int,
                             block_len: int, seed: int) -> dict[str,Any]:
    """Noncircular moving blocks preserve whole rounds and player-round weights."""
    gameweeks=np.asarray(gameweeks);differences=np.asarray(differences,dtype=float)
    if gameweeks.ndim!=1 or differences.shape!=gameweeks.shape or not len(gameweeks) or not np.isfinite(differences).all():
        raise ValueError('Paired scores require aligned finite nonempty arrays')
    if not np.issubdtype(gameweeks.dtype,np.number) or not np.isfinite(gameweeks).all() or (gameweeks<=0).any() or (gameweeks%1!=0).any():
        raise ValueError('Bootstrap gameweeks must be positive integers')
    if type(n_boot) is not int or n_boot<1 or type(block_len) is not int or block_len<1 or type(seed) is not int or seed<0:
        raise ValueError('Invalid bootstrap settings')
    rounds,inverse=np.unique(gameweeks,return_inverse=True)
    n=len(rounds)
    if n<block_len or not np.array_equal(rounds,np.arange(rounds[0],rounds[-1]+1)):
        raise ValueError('Moving blocks require enough contiguous recorded gameweeks')
    sums=np.bincount(inverse,weights=differences);counts=np.bincount(inverse)
    starts=np.random.default_rng(seed).integers(0,n-block_len+1,size=(n_boot,(n+block_len-1)//block_len))
    indices=(starts[:,:,None]+np.arange(block_len)).reshape(n_boot,-1)[:,:n]
    draws=sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
    lo,hi=np.quantile(draws,[.025,.975])
    return {'mean_interval_score_delta':float(differences.mean()),'ci_low':float(lo),'ci_high':float(hi),
            'rounds':n,'player_rounds':len(differences),'n_boot':n_boot,'block_len':block_len,'seed':seed,
            'direction':'candidate minus control; lower is better'}


def _check(registration: dict, outputs: dict[str,str]) -> None:
    if research_package.source_hashes()!=registration['code_sha256']:
        raise ValueError('Grouped interval current sources changed')
    for name,digest in {**registration['input_sha256'],**outputs}.items():
        if research_package.file_hash(Path(name))!=digest:
            raise ValueError(f'Sealed grouped interval artifact changed: {name}')


def _base_inputs(base_dir: Path, dataset: Path) -> tuple[dict,dict,dict[str,str]]:
    read=lambda name:json.loads((base_dir/name).read_bytes())
    summary,registration,verification=read('summary.json'),read('registration.json'),read('verification.json')
    if summary.get('status')!='complete' or str(verification.get('status','')).lower() not in {'pass','verified'}:
        raise ValueError('Base interval study must be complete and independently verified')
    inputs={}
    for name,digest in [('summary.json',verification['summary_sha256']),('registration.json',verification['registration_sha256']),('verification.json',None)]:
        blend_research._bind(inputs,base_dir/name,digest)
    if summary['registration_sha256']!=verification['registration_sha256']:
        raise ValueError('Base summary registration hash differs')
    for path,digest in summary.get('immutable_output_sha256',{}).items():
        path=Path(path);blend_research._bind(inputs,path if path.is_absolute() else base_dir/path,digest)
    for relative,digest in registration['code_sha256'].items():
        blend_research._bind(inputs,base_dir/registration.get('source_archive','evaluated_sources')/relative,digest)
    blend_research._bind(inputs,dataset,registration['dataset_sha256'])
    blend_research._bind(inputs,provenance.FROZEN_MANIFEST_PATH)
    if registration['alpha']!=config.INTERVAL_ALPHA or registration['seed']!=config.INTERVAL_SEED:
        raise ValueError('Base alpha/seed differs from fixed grouped protocol')
    if set(registration['splits'])!={'selection','reference'}:
        raise ValueError('Base requires both declared phases')
    for phase,split in registration['splits'].items():
        if not split['train_max_gw']<split['calibration_min_gw']<=split['calibration_max_gw']<split['evaluation_min_gw']<=split['evaluation_max_gw']:
            raise ValueError('Base splits are not strictly chronological')
        if summary['phases'][phase]['split']!=split:
            raise ValueError('Base phase split differs from registration')
        for name,key in [('calibration.csv','calibration_sha256'),('predictions.csv','predictions_sha256')]:
            blend_research._bind(inputs,base_dir/phase/name,summary['phases'][phase][key])
    return registration,summary,inputs


def _control(calibration: pd.DataFrame, evaluation: pd.DataFrame, alpha: float) -> None:
    if set(evaluation.position)-set(calibration.position):
        raise ValueError('Evaluation position has no earlier calibration rows')
    for position in np.unique(calibration.position):
        cal=calibration.loc[calibration.position==position]
        if cal.empty:raise ValueError('Position has no earlier calibration rows')
        adjustment=conformal_adjustment(cal.actual_total_points,cal.raw_lower,cal.raw_upper,alpha=alpha)
        for frame in (cal,evaluation.loc[evaluation.position==position]):
            if frame.empty:continue
            lower,upper=apply_adjustment(frame.raw_lower,frame['median'],frame.raw_upper,adjustment)
            for actual,expected in [(frame.calibrated_lower,lower),(frame.calibrated_upper,upper),(frame.adjustment,np.full(len(frame),adjustment))]:
                if not np.allclose(actual,expected,rtol=0,atol=1e-12):
                    raise ValueError('Stored position-only control differs from exact reconstruction')


def _report(frame: pd.DataFrame, alpha: float) -> dict:
    def metrics(data):
        if data.empty:return None
        return {'player_rounds':len(data),'control':interval_metrics(data.actual_total_points,data.calibrated_lower,data.calibrated_upper,alpha=alpha),
                'grouped':interval_metrics(data.actual_total_points,data.grouped_lower,data.grouped_upper,alpha=alpha)}
    return {'all':metrics(frame), 'by_position':{str(p):metrics(g) for p,g in frame.groupby('position',sort=True)},
            'by_minutes_group':{str(p):metrics(g) for p,g in frame.groupby('minutes_group',sort=True)},
            'by_position_and_group':[{ 'position':str(p),'group':str(k),'metrics':metrics(g)} for (p,k),g in frame.groupby(['position','minutes_group'],sort=True)],
            'post_outcome_played_diagnostic_only':metrics(frame.loc[frame.minutes>0])}


def run(dataset: str|Path, output_dir: str|Path, *, base_dir: str|Path) -> dict[str,Any]:
    """Register fixed grouped diagnostics before preprocessing, then reuse exports."""
    dataset,output_dir,base_dir=Path(dataset),Path(output_dir),Path(base_dir)
    if output_dir.exists():raise ValueError('Output already exists; preserve completed evidence')
    output_dir.mkdir(parents=True)
    started=time.perf_counter();outputs={};summary={'status':'running','promotion_eligible':False,'phases':{}}
    registration_path=output_dir/'registration.json';summary_path=output_dir/'summary.json'
    try:
        base,base_summary,inputs=_base_inputs(base_dir,dataset)
        hashes=research_package.source_hashes()
        registration={'status':'registered','promotion_eligible':False,'dataset':str(dataset.resolve()),'dataset_sha256':base['dataset_sha256'],
          'base_dir':str(base_dir.resolve()),'base_registration':base,'input_sha256':inputs,'code_sha256':hashes,'source_archive':'source_archive',
          'splits':base['splits'],'alpha':config.INTERVAL_ALPHA,'seed':config.INTERVAL_SEED,
          'min_group_size':config.GROUPED_INTERVAL_MIN_CALIBRATION_ROWS,'minutes_thresholds':list(config.GROUPED_INTERVAL_MINUTES_THRESHOLDS),
          'grouping_feature':FEATURE,'bootstrap':{'n_boot':config.GROUPED_INTERVAL_BOOTSTRAP_SAMPLES,'block_len':config.GROUPED_INTERVAL_BLOCK_LENGTH,'seed':config.INTERVAL_SEED},
          'python':platform.python_version(),'versions':experiment._library_versions(),
          'limitations':['No quantile model refit, retuning or new MILP evidence.','Fixed groups use shifted deadline-known minutes propensity only.',
                         'Both phases retain the fixed candidate regardless of selection diagnostic direction.',
                         'Temporal dependence supplies no exchangeability coverage guarantee; summed quantiles are approximations.',
                         'Played-only reports use observed minutes after outcomes and are diagnostics only.',
                         'Repeatedly examined historical reference, no fresh holdout, production change or promotion.',
                         'Bootstrap intervals are descriptive uncertainty, not null p-values.']}
        cnn_research._archive_sources(output_dir/'source_archive',hashes)
        for relative,digest in hashes.items():blend_research._bind(outputs,output_dir/'source_archive'/relative,digest)
        cnn_research._write_json(registration_path,registration);blend_research._bind(outputs,registration_path)
        _check(registration,outputs)
        experiment.log_result('grouped_interval_research_registration',registration,{'status':'REGISTERED','promotion_eligible':False})
        for phase in ('selection','reference'):
            _check(registration,outputs);split=registration['splits'][phase]
            raw,features,_=load_research_features(split['evaluation_max_gw'],dataset)
            if raw.attrs.get('frozen_dataset_sha256')!=registration['dataset_sha256'] or raw.GW_global.max()>split['evaluation_max_gw'] or features.GW_global.max()>split['evaluation_max_gw']:
                raise ValueError('Grouped interval admitted feature cap/provenance differs')
            _check(registration,outputs)
            frames=[]
            for name,low,high in [('calibration.csv',split['calibration_min_gw'],split['calibration_max_gw']),('predictions.csv',split['evaluation_min_gw'],split['evaluation_max_gw'])]:
                frame=pd.read_csv(base_dir/phase/name);_validate_base(frame,low,high)
                frames.append(join_minutes(frame,features,tuple(registration['minutes_thresholds'])))
            cal,test=frames;_control(cal,test,registration['alpha'])
            fitted=fit_grouped_calibration(cal.actual_total_points,cal.raw_lower,cal.raw_upper,cal.position,cal.minutes_group,
                                         alpha=registration['alpha'],min_group_size=registration['min_group_size'])
            _check(registration,outputs)
            directory=output_dir/phase;directory.mkdir()
            metadata=asdict(fitted);cnn_research._write_json(directory/'calibration_lock.json',metadata);blend_research._bind(outputs,directory/'calibration_lock.json')
            _check(registration,outputs)
            for frame in frames:
                frame['grouped_lower'],frame['grouped_upper'],frame['group_adjustment'],frame['group_source']=apply_grouped_calibration(
                    fitted,frame.raw_lower,frame['median'],frame.raw_upper,frame.position,frame.minutes_group)
            y=test.actual_total_points.to_numpy();delta=_scores(y,test.grouped_lower.to_numpy(),test.grouped_upper.to_numpy(),registration['alpha'])-_scores(y,test.calibrated_lower.to_numpy(),test.calibrated_upper.to_numpy(),registration['alpha'])
            ci=paired_interval_score_ci(test.GW_global.to_numpy(),delta,**registration['bootstrap'])
            for frame,name in zip(frames,('calibration.csv','predictions.csv')):
                frame.to_csv(directory/name,index=False);blend_research._bind(outputs,directory/name)
            report={'metrics':_report(test,registration['alpha']),'paired_interval_score':ci,'calibration':metadata,
                    'calibration_sha256':research_package.file_hash(directory/'calibration.csv'),'predictions_sha256':research_package.file_hash(directory/'predictions.csv')}
            cnn_research._write_json(directory/'phase_summary.json',report);blend_research._bind(outputs,directory/'phase_summary.json')
            _check(registration,outputs);summary['phases'][phase]=report;cnn_research._write_json(summary_path,summary)
        _check(registration,outputs)
        summary.update(status='complete',elapsed_seconds=time.perf_counter()-started,registration_sha256=research_package.file_hash(registration_path),immutable_output_sha256=dict(outputs))
        cnn_research._write_json(summary_path,summary)
        experiment.log_result('grouped_interval_research_completed',{'registration':str(registration_path),'summary_sha256':research_package.file_hash(summary_path)}, {'status':'COMPLETE','promotion_eligible':False,'phases':summary['phases']})
        _check(registration,outputs);return summary
    except BaseException as exc:
        summary.update(status='incomplete',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-started,immutable_output_sha256=dict(outputs))
        cnn_research._write_json(summary_path,summary)
        experiment.log_result('grouped_interval_research_incomplete',{'registration':str(registration_path)}, {'status':'INCOMPLETE','error':summary['error'],'promotion_eligible':False})
        raise


def main(argv: list[str]|None=None) -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=config.FROZEN_RESEARCH_DATASET_PATH)
    parser.add_argument('--base-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv);run(args.dataset,args.output_dir,base_dir=args.base_dir)


if __name__=='__main__':main()
