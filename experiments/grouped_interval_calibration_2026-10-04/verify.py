"""Independently reconstruct completed grouped intervals; never overwrite study files."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from math import ceil
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path.cwd()))
from fpl import config
from fpl.data.research import load_research_features

KEYS=['player_id','position','GW_global']
FEATURE='mins60_rate_roll5'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_bytes())

def need(condition,message):
    if not condition:raise ValueError(message)


def equal(actual,expected,label):
    if isinstance(expected,dict):
        need(isinstance(actual,dict) and set(actual)==set(expected),f'{label}: report keys differ')
        for key,value in expected.items():equal(actual[key],value,f'{label}/{key}')
    elif isinstance(expected,list):
        need(isinstance(actual,list) and len(actual)==len(expected),f'{label}: report list differs')
        for index,value in enumerate(expected):equal(actual[index],value,f'{label}/{index}')
    elif isinstance(expected,float):
        need(np.isfinite(actual) and np.isclose(actual,expected,rtol=0,atol=1e-12),f'{label}: numerical result differs')
    else:need(actual==expected,f'{label}: value differs')


def groups(values,thresholds):
    v=values.to_numpy(dtype=float);low,high=thresholds
    need(0<=low<high<=1,'Invalid thresholds')
    need(np.isfinite(v[~np.isnan(v)]).all() and ((v[~np.isnan(v)]>=0)&(v[~np.isnan(v)]<=1)).all(),'Invalid grouping feature')
    out=np.full(len(v),'missing',dtype='U7');observed=~np.isnan(v)
    out[observed & (v<low)]='low';out[observed & (v>=low)&(v<high)]='mid';out[observed & (v>=high)]='high'
    return out


def adjustment(frame,alpha):
    y=frame.actual_total_points.to_numpy();lower=frame.raw_lower.to_numpy();upper=frame.raw_upper.to_numpy()
    need(len(y)>0 and 0<alpha<1,'Invalid calibration sample/alpha')
    scores=np.maximum(lower-y,y-upper);rank=ceil((len(scores)+1)*(1-alpha))
    need(rank<=len(scores),'Insufficient exact-rank calibration sample')
    return float(np.sort(scores)[rank-1])


def bounds(frame,adjustments):
    return np.minimum(frame.raw_lower.to_numpy()-adjustments,frame['median'].to_numpy()),np.maximum(frame.raw_upper.to_numpy()+adjustments,frame['median'].to_numpy())


def scores(frame,lower,upper,alpha):
    y=frame.actual_total_points.to_numpy()
    return upper-lower+2/alpha*np.where(y<lower,lower-y,0)+2/alpha*np.where(y>upper,y-upper,0)


def metrics(frame,lower,upper,alpha):
    y=frame.actual_total_points.to_numpy();below=y<lower;above=y>upper
    return {'coverage':float(np.mean(~(below|above))),'mean_width':float(np.mean(upper-lower)),
            'interval_score':float(np.mean(scores(frame,lower,upper,alpha))),
            'lower_miss_rate':float(np.mean(below)),'upper_miss_rate':float(np.mean(above))}


def reports(frame,alpha):
    def report(data):
        if data.empty:return None
        return {'player_rounds':len(data), 'control':metrics(data,data.calibrated_lower.to_numpy(),data.calibrated_upper.to_numpy(),alpha),
                'grouped':metrics(data,data.grouped_lower.to_numpy(),data.grouped_upper.to_numpy(),alpha)}
    return {'all':report(frame),'by_position':{str(p):report(g) for p,g in frame.groupby('position',sort=True)},
            'by_minutes_group':{str(p):report(g) for p,g in frame.groupby('minutes_group',sort=True)},
            'by_position_and_group':[{'position':str(p),'group':str(k),'metrics':report(g)} for (p,k),g in frame.groupby(['position','minutes_group'],sort=True)],
            'post_outcome_played_diagnostic_only':report(frame.loc[frame.minutes>0])}


def bootstrap(frame,alpha,parameters):
    differences=scores(frame,frame.grouped_lower.to_numpy(),frame.grouped_upper.to_numpy(),alpha)-scores(frame,frame.calibrated_lower.to_numpy(),frame.calibrated_upper.to_numpy(),alpha)
    aggregated=pd.DataFrame({'gw':frame.GW_global.to_numpy(),'delta':differences}).groupby('gw',sort=True).agg(total=('delta','sum'),count=('delta','size'))
    n=len(aggregated);length=parameters['block_len'];rounds=aggregated.index.to_numpy()
    need(n>=length and np.array_equal(rounds,np.arange(rounds[0],rounds[-1]+1)),'Noncontiguous bootstrap rounds')
    rng=np.random.default_rng(parameters['seed']);draws=[]
    # One draw at a time deliberately provides an independently structured
    # reconstruction of the runner's vectorized noncircular moving blocks.
    for _ in range(parameters['n_boot']):
        starts=rng.integers(0,n-length+1,size=ceil(n/length))
        sampled=np.concatenate([np.arange(start,start+length) for start in starts])[:n]
        draws.append(float(aggregated.total.to_numpy()[sampled].sum()/aggregated['count'].to_numpy()[sampled].sum()))
    low,high=np.quantile(draws,[.025,.975])
    return {'mean_interval_score_delta':float(differences.mean()),'ci_low':float(low),'ci_high':float(high),
            'rounds':n,'player_rounds':len(frame),'n_boot':parameters['n_boot'],'block_len':length,'seed':parameters['seed'],
            'direction':'candidate minus control; lower is better'}


def verify(directory):
    directory=Path(directory).resolve();registration=read(directory/'registration.json')
    summary_bytes=(directory/'summary.json').read_bytes();summary=json.loads(summary_bytes)
    initial_summary_digest=hashlib.sha256(summary_bytes).hexdigest()
    need(summary.get('status')=='complete','Grouped study is not terminal complete')
    need(not summary.get('promotion_eligible') and not registration.get('promotion_eligible'),'Promotion must remain disabled')
    need(summary['registration_sha256']==digest(directory/'registration.json'),'Registration hash mismatch')
    checked={}
    def bind(path,expected):
        path=Path(path).resolve();actual=digest(path);need(actual==expected,f'Hash mismatch: {path}')
        need(str(path) not in checked or checked[str(path)]==actual,f'Conflicting preserved artifact: {path}');checked[str(path)]=actual
    bind(directory/'summary.json',initial_summary_digest)
    for path,expected in {**registration['input_sha256'],**summary['immutable_output_sha256']}.items():bind(path,expected)
    for relative,expected in registration['code_sha256'].items():
        bind(directory/registration['source_archive']/relative,expected)
        # Feature reconstruction uses current code only when it is byte-identical
        # to this study's preserved implementation, never an older base model.
        bind(config.ROOT/relative,expected)
    base_dir=Path(registration['base_dir']);base_reg=read(base_dir/'registration.json');base_summary=read(base_dir/'summary.json');base_verifier=read(base_dir/'verification.json')
    equal(registration['base_registration'],base_reg,'base registration')
    need(base_summary['status']=='complete' and str(base_verifier['status']).lower() in {'pass','verified'},'Base verification not complete/verified')
    bind(base_dir/'summary.json',base_verifier['summary_sha256']);bind(base_dir/'registration.json',base_verifier['registration_sha256'])
    for relative,expected in base_reg['code_sha256'].items():bind(base_dir/base_reg.get('source_archive','evaluated_sources')/relative,expected)
    for name,expected in base_summary.get('immutable_output_sha256',{}).items():
        path=Path(name);bind(path if path.is_absolute() else base_dir/path,expected)
    dataset=Path(registration['dataset']);bind(dataset,registration['dataset_sha256'])
    need(registration['dataset_sha256']==base_reg['dataset_sha256'],'Base dataset differs')
    need(registration['grouping_feature']==FEATURE,'Unregistered grouping input')
    need(type(registration['min_group_size']) is int and registration['min_group_size']>0,'Invalid fixed minimum')
    need(type(registration['seed']) is int and registration['seed']==0,'Unexpected seed')
    alpha=registration['alpha'];minimum=registration['min_group_size'];phase_reports={}
    for phase in ('selection','reference'):
        split=registration['splits'][phase]
        equal(split,base_reg['splits'][phase],f'{phase} fixed split')
        need(split['train_max_gw']<split['calibration_min_gw']<=split['calibration_max_gw']<split['evaluation_min_gw']<=split['evaluation_max_gw'],'Chronology violation')
        with redirect_stdout(sys.stderr):
            raw,features,_=load_research_features(split['evaluation_max_gw'],dataset)
        need(raw.attrs['frozen_dataset_sha256']==registration['dataset_sha256'],'Admitted data hash mismatch')
        need(raw.GW_global.max()<=split['evaluation_max_gw'] and features.GW_global.max()<=split['evaluation_max_gw'],'Future features admitted')
        need(not features.groupby(KEYS,dropna=False)[FEATURE].nunique(dropna=False).gt(1).any(),'DGW feature inconsistency')
        feature_rows=features[KEYS+[FEATURE]].drop_duplicates(KEYS);frames=[]
        for name,start,end,base_key in [('calibration.csv',split['calibration_min_gw'],split['calibration_max_gw'],'calibration_sha256'),('predictions.csv',split['evaluation_min_gw'],split['evaluation_max_gw'],'predictions_sha256')]:
            bind(base_dir/phase/name,base_summary['phases'][phase][base_key])
            original=pd.read_csv(base_dir/phase/name);saved=pd.read_csv(directory/phase/name)
            need(len(saved)==len(original) and not saved.duplicated(KEYS).any(),'Original round coverage/uniqueness differs')
            need(saved.GW_global.between(start,end).all(),'Saved rows exceed declared chronological window')
            extra={FEATURE,'minutes_group','grouped_lower','grouped_upper','group_adjustment','group_source'}
            need(set(saved)==set(original)|extra and not set(original)&extra,'Unexpected output field changes')
            pd.testing.assert_frame_equal(saved[list(original)],original,check_dtype=False,rtol=0,atol=1e-12)
            joined=original[KEYS].merge(feature_rows,on=KEYS,how='left',sort=False,validate='one_to_one',indicator=True)
            need(joined['_merge'].eq('both').all() and joined[KEYS].equals(original[KEYS]),'Missing or reordered grouping matches')
            np.testing.assert_allclose(saved[FEATURE],joined[FEATURE],rtol=0,atol=1e-12,equal_nan=True)
            expected_groups=groups(joined[FEATURE],registration['minutes_thresholds'])
            need(np.array_equal(saved.minutes_group.to_numpy(),expected_groups),'Saved group labels differ from shifted feature')
            for field in ['actual_total_points','minutes','raw_lower','median','raw_upper','calibrated_lower','calibrated_upper','adjustment','grouped_lower','grouped_upper','group_adjustment']:
                need(np.isfinite(saved[field]).all(),f'Nonfinite numeric output: {field}')
            need((saved.raw_lower<=saved['median']).all() and (saved['median']<=saved.raw_upper).all(),'Raw bounds cross median')
            frames.append(saved)
        cal,test=frames;cells=[];counts=[];lookup={}
        for position in sorted(cal.position.unique()):
            subset=cal.loc[cal.position==position];fallback=adjustment(subset,alpha);lookup[position,None]=fallback
            cells.append({'position':position,'group':None,'n':len(subset),'adjustment':fallback})
            for group in sorted(subset.minutes_group.unique()):
                grouped=subset.loc[subset.minutes_group==group];counts.append([position,group,len(grouped)])
                if len(grouped)>=minimum:
                    value=adjustment(grouped,alpha);lookup[position,group]=value
                    cells.append({'position':position,'group':group,'n':len(grouped),'adjustment':value})
        metadata={'alpha':alpha,'min_group_size':minimum,'cells':cells,'group_counts':counts}
        equal(read(directory/phase/'calibration_lock.json'),metadata,f'{phase} signed rank metadata')
        for frame in frames:
            control=np.empty(len(frame));grouped=np.empty(len(frame));sources=np.empty(len(frame),dtype='U8')
            for position,group in frame[['position','minutes_group']].drop_duplicates().itertuples(index=False,name=None):
                mask=(frame.position==position)&(frame.minutes_group==group)
                need((position,None) in lookup,'Evaluation-only position')
                control[mask]=lookup[position,None];use_group=(position,group) in lookup
                grouped[mask]=lookup[position,group] if use_group else lookup[position,None];sources[mask]='group' if use_group else 'position'
            lower,upper=bounds(frame,control)
            np.testing.assert_allclose(frame.adjustment,control,rtol=0,atol=1e-12)
            np.testing.assert_allclose(frame.calibrated_lower,lower,rtol=0,atol=1e-12);np.testing.assert_allclose(frame.calibrated_upper,upper,rtol=0,atol=1e-12)
            lower,upper=bounds(frame,grouped)
            np.testing.assert_allclose(frame.group_adjustment,grouped,rtol=0,atol=1e-12)
            np.testing.assert_allclose(frame.grouped_lower,lower,rtol=0,atol=1e-12);np.testing.assert_allclose(frame.grouped_upper,upper,rtol=0,atol=1e-12)
            need(np.array_equal(frame.group_source,sources),'Group fallback source differs')
        expected={'metrics':reports(test,alpha),'paired_interval_score':bootstrap(test,alpha,registration['bootstrap']),
                  'calibration':metadata,'calibration_sha256':digest(directory/phase/'calibration.csv'),
                  'predictions_sha256':digest(directory/phase/'predictions.csv')}
        equal(read(directory/phase/'phase_summary.json'),expected,f'{phase} saved phase report')
        equal(summary['phases'][phase],expected,f'{phase} summary report')
        phase_reports[phase]={'player_rounds':len(test),'cells':len(cells),'observed_groups':len(counts),
                             'paired_interval_score':expected['paired_interval_score']}
    for name,expected in checked.items():need(digest(name)==expected,f'Artifact changed during verification: {name}')
    return {'status':'verified','promotion_eligible':False,'verified_at_utc':datetime.now(timezone.utc).isoformat(),
            'summary_sha256':digest(directory/'summary.json'),'registration_sha256':digest(directory/'registration.json'),
            'rechecked_artifacts':len(checked),'reports':phase_reports,
            'checks':['Input, output, archived/current feature-source and base proof hashes.',
                      'Capped frozen feature grouping, DGW consistency, exact original CSV fields and round order.',
                      'Independent signed finite-order calibration cells/counts and conservative median brackets.',
                      'Independent inclusive coverage, width, interval scores and all descriptive subgroups.',
                      'Independent noncircular moving-block weighted score differences and 95% confidence intervals.'],
            'limitations':['No training or optimizer execution; descriptive historical diagnostics only.',
                           'No exchangeability guarantee, fresh holdout, null bootstrap p-value or production promotion.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args();print(json.dumps(verify(args.directory),indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
