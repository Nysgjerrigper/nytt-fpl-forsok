"""Read-only independent reconstruction of registered EPL workload research.

No driver/helper execution, model fitting, optimizer run, or study output writes.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from math import ceil, comb
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fpl import config
from fpl.data.provenance import load_frozen_research_dataset, load_frozen_dataset_manifest
from fpl.data.research import load_research_features
from fpl.model import research_package
from fpl.model.moe_tournament import protocol_cutoffs

COLUMNS = ['league_rest_days','league_matches_last7_days','league_matches_last14_days']
ARMS = ['mae_full','mae_league_workload']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def equal(actual: Any, expected: Any, label: str = 'report') -> None:
    if isinstance(expected,dict):
        need(isinstance(actual,dict) and actual.keys()==expected.keys(),f'{label} keys differ')
        for key,value in expected.items():equal(actual[key],value,f'{label}.{key}')
    elif isinstance(expected,list):
        need(isinstance(actual,list) and len(actual)==len(expected),f'{label} lengths differ')
        for i,(a,b) in enumerate(zip(actual,expected)):equal(a,b,f'{label}[{i}]')
    elif isinstance(expected,(float,np.floating)):
        need(bool(np.isclose(actual,expected,rtol=1e-10,atol=1e-12,equal_nan=True)),f'{label} numbers differ')
    else:
        need(actual==expected,f'{label} differs')


def context_from_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    """Direct team/round reconstruction from unique fixtures, not helper math."""
    fixtures=calendar.drop_duplicates(['team','GW_global','fixture']).copy()
    fixtures['clock']=pd.to_datetime(fixtures.kickoff_time,utc=True,format='mixed')
    rows=[]
    for (team,gw),current in fixtures.groupby(['team','GW_global'],sort=False):
        clock=current.clock.min()
        prior=fixtures.loc[(fixtures.team==team)&(fixtures.GW_global<gw)&(fixtures.clock<clock),'clock']
        if prior.empty:values=[np.nan,np.nan,np.nan]
        else:
            days=(clock-prior).dt.total_seconds()/86400.
            values=[float(days.min()),float((days<=7).sum()),float((days<=14).sum())]
        rows.append([team,int(gw),*values])
    return pd.DataFrame(rows,columns=['team','GW_global',*COLUMNS])


def diagnostics(predictions: pd.DataFrame,raw: pd.DataFrame) -> list[dict]:
    """Independent fixture-to-round error/rank diagnostics with earlier MASE scales."""
    current=predictions.loc[predictions.GW==predictions.origin_gw] if 'origin_gw' in predictions else predictions
    rounds=current.groupby(['player_id','position','GW'],as_index=False).agg(actual_total_points=('actual_total_points','sum'),prediction=('predicted_total_points','sum'))
    reports=[]
    for position,group in rounds.groupby('position'):
        normalized=[];rank_values=[];captured=0.;possible=0.
        for gw,week in group.groupby('GW'):
            train=raw.loc[(raw.position==position)&(raw.GW_global<gw)].groupby(['player_id','GW_global'],as_index=False).total_points.sum().sort_values(['player_id','GW_global'])
            scale=float(train.groupby('player_id').total_points.diff().abs().mean())
            need(np.isfinite(scale) and scale>0,'Invalid earlier MASE scale')
            normalized.extend((abs(week.actual_total_points-week.prediction)/scale).tolist())
            y,p=week.actual_total_points.to_numpy(),week.prediction.to_numpy()
            if len(week)>=3 and np.ptp(y)>0 and np.ptp(p)>0:rank_values.append(float(spearmanr(y,p).statistic))
            captured+=float(y[np.argmax(p)]);possible+=float(y.max())
        y,p=group.actual_total_points.to_numpy(),group.prediction.to_numpy()
        error=p-y
        reports.append(dict(position=position,rounds=len(group),mae=float(abs(error).mean()),mase=float(np.mean(normalized)),
            rmse=float(np.sqrt(np.mean(error**2))),bias=float(error.mean()),
            total_calibration=float(p.sum()/y.sum()) if y.sum()!=0 else float('nan'),
            spearman=float(np.mean(rank_values)) if rank_values else float('nan'),top1_capture=captured/possible if possible else float('nan')))
    return reports


def comparison(candidate: pd.DataFrame,control: pd.DataFrame,*,n_boot: int,block_len: int,seed: int) -> dict:
    """Direct contiguous noncircular bootstrap plus exact binomial sign probabilities."""
    a=candidate.set_index('gameweek').actual_total_points.sort_index()
    b=control.set_index('gameweek').actual_total_points.sort_index()
    need(a.index.equals(b.index) and not a.index.has_duplicates,'Unpaired score windows')
    differences=(a-b).to_numpy();n=len(differences);length=min(block_len,n)
    rng=np.random.default_rng(seed)
    totals=[]
    for _ in range(n_boot):
        starts=rng.integers(0,n-length+1,size=ceil(n/length))
        indices=np.concatenate([np.arange(start,start+length) for start in starts])[:n]
        totals.append(float(differences[indices].sum()))
    totals=np.asarray(totals);low,high=np.percentile(totals,[2.5,97.5])
    wa,wb=int((differences>0).sum()),int((differences<0).sum());effective=wa+wb
    if effective:
        probability=comb(effective,wa)/2**effective
        two=sum(comb(effective,k)/2**effective for k in range(effective+1) if comb(effective,k)/2**effective<=probability*(1+1e-12))
        one=sum(comb(effective,k)/2**effective for k in range(wa,effective+1))
    else:two=one=1.
    return dict(total_diff=float(differences.sum()),boot_mean=float(totals.mean()),ci_low=float(low),ci_high=float(high),
        p_a_better=float((totals>0).mean()),n_gws=n,block_len=length,
        sign_test=dict(wins_a=wa,wins_b=wb,ties=int((differences==0).sum()),p_value=float(min(two,1.)),p_a_better=float(one)),
        total_a=float(a.sum()),total_b=float(b.sum()))


def price_catalog(raw: pd.DataFrame,ids: np.ndarray,start: int,end: int) -> pd.DataFrame:
    """Direct strictly-earlier latest-price reconstruction without the catalog builder."""
    past=raw.loc[raw.player_id.isin(ids),['player_id','GW_global','value']].drop_duplicates()
    need(not past.duplicated(['player_id','GW_global']).any(),'Ambiguous player-round prices')
    pieces=[]
    for origin in range(start,end+1):
        observed=past.loc[past.GW_global<origin].sort_values('GW_global').drop_duplicates('player_id',keep='last').copy()
        observed=observed.rename(columns={'GW_global':'price_observed_gw'})
        observed['origin_gw']=origin
        pieces.append(observed[['player_id','origin_gw','value','price_observed_gw']])
    return pd.concat(pieces,ignore_index=True).sort_values(['origin_gw','player_id']).reset_index(drop=True)


def budgets_and_points(predictions: pd.DataFrame,squads: pd.DataFrame,start: int,end: int,catalog: pd.DataFrame | None) -> int:
    """Reconstruct exported current-price cash conservation and realized gross/net sums."""
    key=['player_id','GW']+(['origin_gw'] if 'origin_gw' in predictions else [])
    grouped=predictions.groupby(key,as_index=False).agg(value=('value','first'),name=('name','first'))
    restricted=grouped.loc[grouped.GW.between(start-1,end)].sort_values(key)
    metadata=restricted.drop_duplicates('player_id').set_index('player_id')
    need(metadata.name.is_unique,'Ambiguous exported player names')
    reverse={str(name).strip():int(code) for code,name in metadata.name.items()}
    prices=restricted.drop_duplicates(['player_id','GW']).pivot(index='player_id',columns='GW',values='value').sort_index(axis=1).ffill(axis=1).bfill(axis=1).fillna(50.)
    prior_ids=catalog.groupby('origin_gw').player_id.agg(set).to_dict() if catalog is not None else None
    outcomes=predictions.loc[predictions.GW==predictions.origin_gw] if 'origin_gw' in predictions else predictions
    actual=outcomes.groupby(['name','GW']).actual_total_points.sum()
    previous=set();bank=1000.
    for row in squads.to_dict('records'):
        gw=int(row['gameweek']);names=[name.strip() for name in row['squad'].split(', ')];codes=[reverse[name] for name in names]
        need(len(codes)==len(set(codes))==15,'Invalid exported squad cardinality')
        if prior_ids is not None:need(set(codes).issubset(prior_ids[gw]),'Squad violated known-prior-price eligibility')
        need(np.isclose(row['budget_start'],bank),'Budget-start cash mismatch')
        starting_value=prices.loc[list(previous),gw].sum() if previous else 0.
        need(np.isclose(prices.loc[codes,gw].sum()+row['budget_end'],bank+starting_value),'Current-price budget mismatch')
        lineup=[name.strip() for name in row['actual_lineup'].split(', ') if name.strip()]
        lineup_points=sum(float(actual.loc[(name,gw)]) for name in lineup)
        captain=row['actual_captain'];captain_points=0. if pd.isna(captain) else float(actual.loc[(str(captain).strip(),gw)])
        need(np.isclose(lineup_points,row['actual_lineup_points']) and np.isclose(captain_points,row['actual_captain_points']),'Realized lineup/captain score mismatch')
        need(np.isclose(sum(float(actual.loc[(name,gw)]) for name in names),row['actual_squad_points']),'Realized whole-squad score mismatch')
        need(np.isclose(lineup_points+captain_points,row['gross_actual_total_points']),'Gross score mismatch')
        previous=set(codes);bank=float(row['budget_end'])
    return len(squads)


def verify(directory: Path) -> dict[str,Any]:
    """Verify completed study byte locks and independently rebuild numerical evidence."""
    directory=directory.resolve();summary_path=directory/'summary.json';registration_path=directory/'registration.json'
    checked={str(summary_path):digest(summary_path)}
    summary=json.loads(summary_path.read_bytes());registration=json.loads(registration_path.read_bytes())
    need(summary['status'] in ('complete','selection_rejected'),'Study is not terminal')
    need(summary['promotion_eligible'] is False and registration['promotion_eligible'] is False,'Invalid promotion claim')
    need(registration['arms']==ARMS and registration['extra_player_features']==COLUMNS,'Unexpected workload family')
    need(type(registration['seed']) is int and registration['seed']==0 and registration['model_name']=='catboost_mae','Unexpected model/seed')
    need(registration['horizon']==3 and registration['retrain_every']==4 and registration['gap_rel']==0 and registration['chips']=='disabled','Unexpected runtime protocol')
    need(registration['comparison']==dict(n_boot=10000,block_len=3,seed=0),'Unexpected comparison settings')
    cutoffs=protocol_cutoffs();equal(registration['cutoffs'],cutoffs)
    equal(registration['windows'],dict(selection=[cutoffs['selection_min_gw'],cutoffs['selection_max_gw']],reference=[cutoffs['backtest_min_gw'],cutoffs['backtest_max_gw']]))
    frozen_max=load_frozen_dataset_manifest()['max_gw_global']
    need(summary['registration_sha256']==digest(registration_path),'Registration digest mismatch')
    checked[str(registration_path)]=summary['registration_sha256']
    for relative,sha in registration['code_sha256'].items():checked[str(directory/registration['source_archive']/relative)]=sha
    checked.update(registration['input_sha256']);checked.update(summary['immutable_output_sha256'])
    for name,sha in checked.items():need(digest(Path(name))==sha,f'Changed sealed bytes: {name}')
    need(research_package.source_hashes()==registration['code_sha256'],'Current pipeline differs from registered source')
    driver_inputs=[Path(name) for name in registration['input_sha256'] if Path(name).name in ('run.py','league_workload.py')]
    need({p.name for p in driver_inputs}=={'run.py','league_workload.py'},'Missing own driver/helper input hashes')
    for path in driver_inputs:need(digest(directory/'source_archive'/'driver'/path.name)==digest(path),'Driver/helper archive mismatch')
    for position,payload in registration['parameters']['catboost_mae'].items():
        need(payload['catboost_mae']['thread_count']==5 and payload['catboost_mae']['random_seed']==0,f'Invalid capacity runtime {position}')
    with redirect_stdout(sys.stderr):raw=load_frozen_research_dataset(registration['dataset'])
    need(raw.attrs['frozen_dataset_sha256']==registration['dataset_sha256'],'Dataset admission SHA mismatch')
    selected=summary['selection'];need(set(selected)==set(ARMS),'Incomplete selection arms')
    advanced=selected[ARMS[1]]['total_points']>selected[ARMS[0]]['total_points']
    need(summary['advanced']==advanced,'Strict selection gate mismatch')
    lock=json.loads((directory/'selection_lock.json').read_bytes())
    need(lock['advanced']==advanced and lock['selection_summary_sha256']==digest(directory/'selection_summary.json'),'Selection lock mismatch')
    selection_summary=json.loads((directory/'selection_summary.json').read_bytes())
    equal(selection_summary,dict(selection=selected,comparison=summary['comparisons']['selection_origin'],advanced=advanced))
    equal(lock['parameters'],registration['parameters'])
    need(lock['finalists']==(ARMS if advanced else []),'Finalists mismatch')
    if not advanced:need(not summary['evaluation'] and not (directory/'reference').exists(),'Rejected study loaded reference artifacts')
    else:need(set(summary['evaluation'])==set(ARMS),'Incomplete reference arms')
    phases=['selection','reference'] if advanced else ['selection'];base=None;contexts={};catalogs={}
    for phase in phases:
        start,end=registration['windows'][phase]
        need(1<=start<=end<=frozen_max,'Invalid phase cutoff')
        capped=raw.loc[raw.GW_global.between(1,end)]
        location=directory/phase
        calendar=pd.read_csv(location/'calendar.csv')
        expected=capped[['team','GW_global','fixture','kickoff_time']].drop_duplicates()
        keys=['team','GW_global','fixture','kickoff_time']
        pd.testing.assert_frame_equal(calendar.sort_values(keys).reset_index(drop=True),expected.sort_values(keys).reset_index(drop=True),check_dtype=False)
        rebuilt=context_from_calendar(calendar);stored=pd.read_csv(location/'context.csv')
        need(not stored.duplicated(['team','GW_global']).any(),'Duplicate stored context')
        pd.testing.assert_frame_equal(stored.sort_values(['team','GW_global']).reset_index(drop=True),rebuilt.sort_values(['team','GW_global']).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-12)
        for filename in ('calendar.csv','context.csv','columns.json','origin_prices.csv'):
            need(str((location/filename).resolve()) in checked,f'Unsealed phase artifact: {filename}')
        columns=json.loads((location/'columns.json').read_bytes())
        need(len(columns[ARMS[0]])==115 and len(set(columns[ARMS[0]]))==115 and 'total_points' not in columns[ARMS[0]],'Invalid base columns')
        need(columns[ARMS[1]]==columns[ARMS[0]]+COLUMNS,'Invalid workload ordered columns')
        with redirect_stdout(sys.stderr):_,frame,exact_columns=load_research_features(end,registration['dataset'])
        need(columns[ARMS[0]]==exact_columns,'Base feature recipe mismatch')
        if base is None:base=exact_columns
        else:need(base==exact_columns,'Phase base schema mismatch')
        if phase=='selection':equal(lock['feature_variants'],columns)
        ids=frame.loc[frame.GW_global.between(start,end),'player_id'].unique()
        catalog=pd.read_csv(location/'origin_prices.csv')
        pd.testing.assert_frame_equal(catalog,price_catalog(capped,ids,start,end),check_dtype=False)
        contexts[phase]=len(stored);catalogs[phase]=catalog
    entries=[('selection','origin',arm,selected[arm]) for arm in ARMS]
    if advanced:
        for arm in ARMS:
            need(set(summary['evaluation'][arm])=={'standard','origin'},'Missing reference modes')
            entries.extend(('reference',mode,arm,result) for mode,result in summary['evaluation'][arm].items())
    optimal=budgets=0;scores={};squad_frames={}
    for phase,mode,arm,result in entries:
        location=directory/phase;start,end=registration['windows'][phase]
        predictions=pd.read_csv(location/result['predictions']);squads=pd.read_csv(location/result['squads'])
        need(digest(location/result['predictions'])==result['prediction_sha256'] and digest(location/result['squads'])==result['squad_sha256'],'Scored output digest mismatch')
        research_package.validate_forecasts(predictions,start,end,mode=='origin')
        need(squads.gameweek.tolist()==list(range(start,end+1)),'Scored round coverage mismatch')
        need(squads.scoring_protocol.eq('autosubs_v2').all() and squads.points_accounting.eq('net_after_transfer_hits').all(),'Scoring protocol mismatch')
        need(squads.chip_played.isna().all(),'Research chips enabled')
        need(np.array_equal(squads.actual_total_points.to_numpy(),(squads.gross_actual_total_points-4*squads.alpha).to_numpy()),'Net hit accounting mismatch')
        equal(result['total_points'],float(squads.actual_total_points.sum()));equal(result['gross_points'],float(squads.gross_actual_total_points.sum()));equal(result['transfer_hit_points'],float(4*squads.alpha.sum()))
        prefix=location/f'{arm}_{mode}';admission=json.loads(prefix.with_suffix('.admission.json').read_bytes())
        for filename in (result['predictions'],result['squads'],prefix.with_suffix('.admission.json').name,prefix.with_suffix('.result.json').name,prefix.with_suffix('.log').name):
            need(str((location/filename).resolve()) in checked,f'Unsealed scored artifact: {filename}')
        equal(json.loads(prefix.with_suffix('.result.json').read_bytes()),result)
        columns=json.loads((location/'columns.json').read_bytes())
        need(admission['feature_columns']==columns[arm],'Admitted columns mismatch')
        equal(admission['parameters'],registration['parameters']['catboost_mae'])
        need(admission['seed']==0 and admission['horizon']==3 and admission['retrain_every']==4 and not admission['cached_replay'],'Forecast settings mismatch')
        need(admission['start_gw']==start and admission['end_gw']==end and admission['origin']==(mode=='origin'),'Forecast windows mismatch')
        if mode=='origin':
            need(admission['origin_price_policy']=='origin-current-proxy' and admission['origin_price_catalog_sha256']==digest(location/'origin_prices.csv'),'Origin price policy mismatch')
            need(admission['dated_origin_snapshot']==(arm==ARMS[1]) and admission['extra_player_features']==(COLUMNS if arm==ARMS[1] else []),'Origin freeze adapter mismatch')
        count=prefix.with_suffix('.log').read_text().count('Status: Optimal (')
        need(count==end-start+1,'Incomplete optimal solve evidence');optimal+=count
        budgets+=budgets_and_points(predictions,squads,start,end,catalogs[phase] if mode=='origin' else None)
        equal(result['diagnostics'],diagnostics(predictions,raw.loc[raw.GW_global<=end]),f'{phase}/{arm}/{mode} diagnostics')
        scores[f'{phase}/{arm}/{mode}']=result['total_points'];squad_frames[(phase,mode,arm)]=squads
    for phase in phases:
        modes=['origin'] if phase=='selection' else ['standard','origin']
        for mode in modes:
            reports=selected if phase=='selection' else {arm:summary['evaluation'][arm][mode] for arm in ARMS}
            research_package.validate_paired_forecasts(directory/phase/reports[ARMS[1]]['predictions'],directory/phase/reports[ARMS[0]]['predictions'])
            rebuilt=comparison(squad_frames[(phase,mode,ARMS[1])],squad_frames[(phase,mode,ARMS[0])],**registration['comparison'])
            equal(summary['comparisons'][f'{phase}_{mode}'],rebuilt)
            equal(json.loads((directory/f'{phase}_{mode}_comparison.json').read_bytes()),rebuilt)
    expected=2*(registration['windows']['selection'][1]-registration['windows']['selection'][0]+1)
    if advanced:expected+=4*(registration['windows']['reference'][1]-registration['windows']['reference'][0]+1)
    need(optimal==budgets==expected,'Total scored optimal/budget counts mismatch')
    for name,sha in checked.items():need(digest(Path(name))==sha,f'Bytes drifted during verification: {name}')
    need(research_package.source_hashes()==registration['code_sha256'],'Source drift during verification')
    need(config.PRODUCTION_WEIGHT_STRATEGY=='single:catboost','Unexpected production strategy')
    return dict(status='verified',verified_at_utc=datetime.now(timezone.utc).isoformat(),promotion_eligible=False,
        summary_sha256=digest(summary_path),registration_sha256=digest(registration_path),source_files=len(registration['code_sha256']),
        rechecked_artifacts=len(checked),optimal_solves=optimal,independently_reconstructed_current_budgets=budgets,
        independently_reconstructed_context_rows=contexts,results=scores,
        checks=['All input/output/own driver/helper and archived pipeline byte hashes.',
            'Frozen capped fixture calendars, strict earlier round/date context and earliest DGW clock.',
            'Exact ordered115/118 features, strict earlier gate and sealed finalist lock.',
            'Paired forecast metadata, corrected gross-minus-four-hits and optimal solve counts.',
            'Current-price budgets/eligibility and exported realized squad/lineup/captain sums.',
            'Independent player-round forecast diagnostics, block-bootstrap CIs and exact sign tests.'],
        limitations=['Historical EPL schedule proxy; no publication-time or all-competition claim.',
            'No model refits or optimizer execution; no untouched holdout or promotion assessment.'])


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args();print(json.dumps(verify(args.directory),indent=2,sort_keys=True))


if __name__=='__main__':main()
